"""Efficiency: vanilla softmax attention against plain linear attention, one card, same model.

    python .edullm/bench_attention_vs_linear.py --seq-lens 1024,2048,4096,8192,16384,32768

WHAT IS HELD FIXED. One `olmo3_370M` backbone, identical d_model / n_layers / n_heads /
head_dim / vocab, identical batch tokens per step, identical dtype. The ONLY difference between
arms is the block's sequence mixer. Weights are randomly initialised on purpose: step time,
throughput and peak memory do not depend on what a weight contains, so this needs no checkpoint
and is not waiting on one.

WHY "VANILLA" HAD TO BE BUILT RATHER THAN TAKEN FROM THE FACTORY. `TransformerConfig.olmo3_370M`
does not give you vanilla attention. It sets a `SlidingWindowAttentionConfig` with pattern
`[4096, 4096, 4096, -1]`, so three layers in four are windowed -- already sub-quadratic. Timing
linear attention against that and calling the result "linear beats quadratic" would be measuring
against something that is not quadratic. So the `full` arm sets `sliding_window=None` explicitly,
and the quadratic term is really there.

WHY THE STOCK SLIDING-WINDOW ARM IS NOT INCLUDED. It would need flash-attn 2, whose windowed
kernel is what makes SWA fast. `A100-MFU-PLAYBOOK.md` B7 records that flash-attn 2 is NOT in the
platform image and that `assert_supported()` raises rather than degrading -- hard-pinning it
"broke the build outright". On torch SDPA a window is a materialised mask over full attention, so
an SWA arm here would report full-attention cost under a windowed name. A number that misleads is
worse than a number that is missing.

THE BACKBONE'S OWN BACKEND WOULD ALSO HAVE FAILED. `olmo3_370M` pins
`attn_backend=flash_2`, for the same absent library. Both arms are therefore pinned to `torch`
SDPA, uniformly across every layer, and the uniformity is asserted rather than assumed -- B7 again,
where an unpinned backend resolved per layer and turned a 3:1 window layout into a silent 3:1
*kernel* split, biasing the very number the benchmark existed to produce.

MEASUREMENT, FOLLOWING THE PLAYBOOK'S SECTION A BECAUSE HALF ITS "WINS" WERE INSTRUMENT BUGS:

  * `--warmup` steps are discarded and never enter a statistic. A rate whose denominator
    includes one-time setup is a different quantity from the rate you extrapolate with.
  * MEDIAN step time over the measured window, with min/p90/max beside it. A median alone hides
    a bimodal step time; the spread is what shows one.
  * `torch.cuda.synchronize()` only at the window's edges, so timing does not serialise the work.
  * ARM ORDER IS COUNTERBALANCED -- forwards, then reversed -- because a sequential loop on a
    warming GPU biases whichever arm ran last, measured rather than hypothetical (B8). Both
    passes are reported separately and the drift between them is printed. If drift exceeds the
    effect, the comparison has not measured the effect.
  * torch.compile is OFF by default and that is deliberate. Its eager fallback is process-sticky
    and `backend_used` records what was REQUESTED, not what executed (A4), so a compiled
    benchmark can silently report eager for the second arm onward -- up to 43x error in the
    playbook's case. `--compile` turns it on and calls `torch._dynamo.reset()` between arms.
  * tokens/sec is the ranking metric, NOT MFU. The playbook is explicit: never rank configs
    that differ in seq_len by MFU. MFU is printed for scale only, and the attention arm's FLOP
    counter overcounts its score term by ~9% by using a non-causal form for a causal kernel.
  * An out-of-memory is a RESULT, not a crash: it is recorded against that arm and length and
    the sweep continues. Where vanilla attention stops fitting is half the finding.
"""

import argparse
import gc
import json
import logging
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "linear-attn-vs-gdn"
    ),
)

import olmo_linear_attn  # noqa: E402,F401  (registers the "linear_attention" mixer)
from olmo_linear_attn import LinearAttentionConfig  # noqa: E402

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerBlockConfig  # noqa: E402

log = logging.getLogger("bench")

L2_BYTES = {"A100": 40 << 20, "L40S": 96 << 20, "A10G": 6 << 20, "L4": 48 << 20, "T4": 4 << 20}


def build(arm: str, opts) -> TransformerConfig:
    """One backbone, one mixer swapped, one backend pinned everywhere."""
    vocab = TokenizerConfig.dolma2().padded_vocab_size()
    # sliding_window=None is what makes the `full` arm actually quadratic; see the module
    # docstring. attn_backend=torch is what makes it runnable in this image at all.
    cfg = TransformerConfig.olmo3_370M(
        vocab_size=vocab,
        sliding_window=None,
        attn_backend=AttentionBackendName.torch,
    )
    if arm == "linear":
        mixer = LinearAttentionConfig(
            n_heads=opts.n_heads,
            n_v_heads=opts.n_heads,
            head_dim=opts.head_dim,
            expand_v=1.0,
            conv_size=4,
            qk_l2norm=True,
            # normalize=False is the baseline's setting: the pure ungated cumulative sum, which
            # is the configuration the trained linear model actually used. Matching it keeps this
            # efficiency number attached to the model the other eval scores.
            normalize=False,
        )
        blocks = cfg.block if isinstance(cfg.block, dict) else {"_": cfg.block}
        for b in blocks.values():
            if isinstance(b, TransformerBlockConfig):
                b.sequence_mixer = mixer
    return cfg


def assert_uniform_backend(model) -> str:
    """B7: one kernel on every layer, asserted rather than hoped for."""
    seen = set()
    for m in model.modules():
        backend = getattr(m, "backend", None)
        if backend is not None and hasattr(backend, "__class__"):
            seen.add(type(backend).__name__)
    if len(seen) > 1:
        raise RuntimeError(f"attention backend is not uniform across layers: {sorted(seen)}")
    return next(iter(seen)) if seen else "n/a"


def working_set_report(model, device) -> str:
    """A5: a benchmark whose working set fits in L2 can report the wrong sign."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    l2 = next((v for k, v in L2_BYTES.items() if k in name), 40 << 20)
    ratio = param_bytes / l2
    warn = (
        "  *** WORKING SET FITS IN L2 -- TREAT EVERY NUMBER BELOW AS SUSPECT ***"
        if ratio < 1
        else ""
    )
    return (
        f"params {param_bytes / 2**30:.2f} GiB, L2 {l2 / 2**20:.0f} MiB, ratio {ratio:.1f}x{warn}"
    )


def time_arm(arm: str, seq_len: int, opts, device) -> Dict[str, Any]:
    """Median steady-state step time for one arm at one length, or an OOM record."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    cfg = build(arm, opts)
    try:
        model = cfg.build(init_device=str(device)).to(torch.bfloat16)
        model.train()
        backend = assert_uniform_backend(model)
        if opts.compile:
            torch._dynamo.reset()  # A4: the fallback is process-sticky across arms
            model.compile()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)

        bs = max(1, opts.batch_tokens // seq_len)
        vocab = TokenizerConfig.dolma2().padded_vocab_size()
        ids = torch.randint(0, vocab, (bs, seq_len), device=device, dtype=torch.long)
        # get_labels rather than ids.clone(): the model warns loudly that identical labels are a
        # COPY TASK whose loss collapses without erroring. The FLOPs are the same either way, so
        # this does not move a timing number -- but a benchmark that trips a correctness warning
        # on every step is one whose output nobody should trust on sight.
        labels = get_labels({"input_ids": ids})

        def step():
            opt.zero_grad(set_to_none=True)
            # forward returns LMOutputWithLoss (a NamedTuple), not a tensor: .loss is the one to
            # optimize, .ce_loss and .z_loss are for logging.
            out = model(input_ids=ids, labels=labels)
            loss = out.loss if hasattr(out, "loss") else out
            (loss if loss.ndim == 0 else loss.mean()).backward()
            opt.step()

        for _ in range(opts.warmup):
            step()
        torch.cuda.synchronize()

        times: List[float] = []
        for _ in range(opts.steps):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        med = statistics.median(times)
        tokens = bs * seq_len
        peak = torch.cuda.max_memory_allocated() / 2**30
        fpt = model.num_flops_per_token(seq_len)
        out = {
            "arm": arm,
            "seq_len": seq_len,
            "batch": bs,
            "tokens_per_step": tokens,
            "median_s": med,
            "min_s": min(times),
            "p90_s": sorted(times)[int(0.9 * len(times)) - 1],
            "max_s": max(times),
            "spread": max(times) / med,
            "tokens_per_s": tokens / med,
            "peak_gib": peak,
            "flops_per_token": fpt,
            "tflops": fpt * tokens / med / 1e12,
            "backend": backend,
            "oom": False,
        }
        # No `del` of the closed-over locals: `step` holds them in a cell, and the function
        # returns here anyway, so the `finally` below is what actually reclaims the memory.
        return out
    except torch.cuda.OutOfMemoryError:
        # A finding, not a failure. Where the quadratic arm stops fitting is half the point.
        return {"arm": arm, "seq_len": seq_len, "oom": True}
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="vanilla attention vs linear attention, one card.")
    p.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    p.add_argument("--seq-lens", default="1024,2048,4096,8192,16384,32768")
    p.add_argument(
        "--batch-tokens",
        type=int,
        default=16384,
        help="Tokens per step, held equal across arms and lengths.",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--output", default=None)
    opts = p.parse_args()

    if not torch.cuda.is_available():
        print("FATAL no CUDA device; this benchmark measures a GPU")
        return 2
    device = torch.device("cuda")
    seq_lens = [int(s) for s in opts.seq_lens.split(",") if s]

    print(f"device {torch.cuda.get_device_name(device)}")
    print(f"batch tokens/step {opts.batch_tokens}  warmup {opts.warmup}  measured {opts.steps}")
    print(f"compile {opts.compile}  (off by default: see A4 in the playbook)")
    probe = build("full", opts).build(init_device="meta")
    print(f"model {probe.num_params:,} params; {working_set_report(probe, device)}")
    del probe

    # COUNTERBALANCED: forwards, then reversed. A warming GPU biases whichever arm ran last.
    results: Dict[str, List[Dict[str, Any]]] = {"forward": [], "reverse": []}
    for pass_name, order in (("forward", ["full", "linear"]), ("reverse", ["linear", "full"])):
        for seq_len in seq_lens:
            for arm in order:
                r = time_arm(arm, seq_len, opts, device)
                r["pass"] = pass_name
                results[pass_name].append(r)
                if r["oom"]:
                    print(f"  {pass_name:>7} {arm:>6} L={seq_len:<6} OOM")
                else:
                    print(
                        f"  {pass_name:>7} {arm:>6} L={seq_len:<6} "
                        f"{r['median_s'] * 1e3:8.1f} ms  {r['tokens_per_s']:>9,.0f} tok/s  "
                        f"{r['peak_gib']:5.2f} GiB  spread {r['spread']:.2f}x"
                    )

    # ---- summary LAST: `edullm logs` shows only the final fifty lines -----------------------
    def find(pass_name: str, arm: str, L: int) -> Optional[Dict[str, Any]]:
        return next((r for r in results[pass_name] if r["arm"] == arm and r["seq_len"] == L), None)

    print()
    print("=" * 78)
    print("VANILLA (full softmax, SDPA) vs LINEAR ATTENTION -- median steady-state")
    print("=" * 78)
    print(
        f"{'seq':>7} {'vanilla tok/s':>14} {'linear tok/s':>13} {'speedup':>8} "
        f"{'van GiB':>8} {'lin GiB':>8} {'drift':>7}"
    )
    table = []
    for L in seq_lens:
        f_full, f_lin = find("forward", "full", L), find("forward", "linear", L)
        r_lin = find("reverse", "linear", L)
        if not f_full or not f_lin:
            continue
        v_oom, l_oom = f_full["oom"], f_lin["oom"]
        v = "OOM" if v_oom else f"{f_full['tokens_per_s']:,.0f}"
        ln = "OOM" if l_oom else f"{f_lin['tokens_per_s']:,.0f}"
        sp = "-" if (v_oom or l_oom) else f"{f_lin['tokens_per_s'] / f_full['tokens_per_s']:.2f}x"
        vg = "-" if v_oom else f"{f_full['peak_gib']:.2f}"
        lg = "-" if l_oom else f"{f_lin['peak_gib']:.2f}"
        # drift: how much the reversed pass moved the same arm. If this rivals `speedup`,
        # order effects are the size of the effect and the comparison is not clean.
        drift = "-"
        if r_lin and not r_lin["oom"] and not l_oom:
            drift = (
                f"{abs(r_lin['tokens_per_s'] - f_lin['tokens_per_s']) / f_lin['tokens_per_s']:.1%}"
            )
        print(f"{L:>7} {v:>14} {ln:>13} {sp:>8} {vg:>8} {lg:>8} {drift:>7}")
        table.append({"seq_len": L, "vanilla": f_full, "linear": f_lin})

    print()
    print("speedup is linear tok/s / vanilla tok/s at equal tokens per step. Above 1.00 means")
    print("linear attention is faster. drift is the same arm's change when the order reversed;")
    print("if drift approaches speedup, read neither.")
    print("Ranked by tokens/sec, not MFU -- the arms differ in seq_len and in FLOP formula.")

    out = opts.output or f"/tmp/bench_attn_vs_linear_{opts.run_name}.json"
    with open(out, "w") as f:
        json.dump({"opts": vars(opts), "results": results}, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
