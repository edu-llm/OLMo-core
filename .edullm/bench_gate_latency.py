"""Full-model inference latency for the three P1 gate structures, on one A100.

WHAT THIS ANSWERS, AND WHY IT IS THE LAST MISSING NUMBER
  The 1B-token grid answered P1's QUALITY question and returned a well-powered null:
  ``F-r128`` and ``G-grouped`` reach the same held-out cross-entropy as dense ``L0`` while
  removing 15,728,640 parameters (75% of the gate budget, 4.03% of the model). A null on
  quality is only interesting if the parameters bought something, so the claim is an
  EFFICIENCY claim and it rests on a latency number this project does not yet have.

  Everything measured so far was a MICROBENCHMARK of the gate projections in isolation, on an
  L40S. Two facts make those numbers unusable as the headline:

    1. The first one (``experiments/liv/probes/p1_launch_bench.py``) held 40 MiB of gate
       weights against the L40S's 96 MiB L2 and replayed a CUDA graph with nothing to evict
       them. It reported low-rank as 8.2% SLOWER. That figure was RETRACTED on 2026-08-01.
       The tell sat in its own output unexamined: dense achieved 744.7 GB/s, and the L40S HBM
       peak is 864 GB/s -- 86% of peak on a working set that fits in cache is a cache
       measurement, not a bandwidth measurement.
    2. The residency-scaled re-test (``p1_scaled.py``) fixed the regime and flipped the sign:
       past L2, ``F-r128`` is +31.3% to +40.4% and ``G-grouped`` is +47.6% to +54.4%. But
       both still time SEVEN LINEAR LAYERS IN A LOOP, not a model. Share-weighting a +40%
       subgraph win by the 5.38% of parameters it touches gives ~+1.8% end-to-end, and that
       arithmetic is a prediction, not a measurement.

  So this harness times the WHOLE MODEL, built through the same ``build_arm`` path the
  training run used, on the card the study ran on. It is the only configuration in which the
  end-to-end number is measured rather than inferred.

THE FOUR RECEIPTS, WHICH ARE THE POINT OF THE FILE
  A latency delta on its own is not checkable, and this project has already been burned by
  one that replicated cleanly while measuring the wrong regime. Every row therefore carries:

    working_set_mib      -- bytes of weights the timed region reads
    achieved_gbs         -- working_set / elapsed
    pct_of_hbm_peak      -- achieved_gbs / 1555 GB/s (A100-SXM4-40GB)
    conv_path            -- 'fla' or 'nn.Conv1d', ASSERTED equal across arms

  ``pct_of_hbm_peak`` above 100 means the timed region read cache, and the row is marked
  ``CACHE_RESIDENT`` and excluded from the headline. That is the check whose absence caused
  the retraction. It is not advisory here: ``--fail-on-cache-resident`` makes it an error.

  ``conv_path`` exists because ``ShortConv.use_fla`` defaults to True and dispatches on
  ``has_fla() and x.is_cuda``. If ``fla`` is present for one arm's code path and absent for
  another's, the contrast compares a fused kernel against ``nn.Conv1d`` and attributes the
  difference to gate structure. Verified 2026-08-05: ``fla`` is NOT in the research image, so
  every arm should report ``nn.Conv1d``. This pins ``use_fla`` identically per arm anyway and
  asserts the realised path, because "available" is a property of the environment and not of
  the arm.

WHAT IS MEASURED, AND THE ONE THING THAT IS NOT
  Two regimes, because the gates sit in different company in each:

    prefill   (batch, seq_len) forward, compute-bound, GEMMs are large and square-ish.
              This is where a smaller gate buys the LEAST, so it is the conservative rung.
    decode    seq_len=1 forward at several batch sizes, memory-bound, GEMMs are skinny.
              This is where weight bytes dominate and where P1's claim lives.

  DECODE HERE IS NOT AUTOREGRESSIVE, AND THAT IS A REAL LIMIT, STATED IN THE OUTPUT.
  ``ShortConv`` implements no conv-state cache and ``forward`` takes no ``past_key_values``;
  attention layers in this config likewise run without a KV cache. A true single-token step
  would reuse both. So ``decode`` here is a seq_len=1 forward: it reproduces the skinny-GEMM,
  weight-bytes-dominated regime that the gate structures differ in, and it does NOT include
  the KV-cache traffic a served decode would add. Since that traffic is identical across
  arms (all three share attention geometry exactly), it enters both numerator and denominator
  of the ratio and DILUTES the measured delta -- so the end-to-end figure reported here is an
  UPPER BOUND on the served-decode speedup, and the report says so.

  The prefill rung has no such caveat, which is why both are reported.

WHY ONE CARD AND WHY IT IS AN EIGHT-CARD MACHINE ANYWAY
  This platform prices no single-A100 shape -- the ``compute_profile`` dropdown offers
  ``gpu-8xa100`` (p4d.24xlarge) and nothing smaller with an A100 in it. Inference latency is
  a per-card property and this harness uses ONE device, so seven cards sit idle. That is
  waste, and it is unavoidable if the number is to be measured on the card the study used.

  The alternative is ``gpu-1xl40s`` at $1.8610/hour, which is 11.8x cheaper and measures a
  DIFFERENT card. It is worth running as a cross-check precisely because it is cheap, and
  ``--device-peak-gbs`` exists so the bandwidth receipt is correct on either.

  Because seven cards are idle, the run needs ``EDULLM_LAUNCH_CHECK=waived``: the platform
  requires processes == devices on a multi-GPU shape, and one process on eight cards is
  exactly the benchmark case that waiver is for.

  srun -p gpu --gres=gpu:1 -c 8 --mem=64G -t 00:40:00 python .edullm/bench_gate_latency.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("bench_gate_latency")

# ---------------------------------------------------------------------------------------------
# Cards, and the peak bandwidth that makes the cache check meaningful.
#
# These are vendor HBM figures, not achieved. `achieved/peak > 1` is impossible for a genuinely
# HBM-bound region and therefore proves cache residency, which is the only inference drawn from
# them. A missing card is an ERROR rather than a default: silently assuming a peak would restore
# exactly the blind spot this file exists to close, and `speed_monitor.py` has already shipped
# one MFU number inflated 1.175x by a missing per-card entry.
# ---------------------------------------------------------------------------------------------
HBM_PEAK_GBS: Dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-PCIE-40GB": 1555.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA L4": 300.0,
    "NVIDIA A10G": 600.0,
    "Tesla T4": 320.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
}

#: L2 sizes, reported alongside the working set so a reader can see the margin rather than
#: trust the ratio. Informational only -- the pass/fail check is the bandwidth ratio.
L2_MIB: Dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 40.0,
    "NVIDIA A100-SXM4-80GB": 40.0,
    "NVIDIA A100-PCIE-40GB": 40.0,
    "NVIDIA L40S": 96.0,
    "NVIDIA L4": 48.0,
    "NVIDIA A10G": 6.0,
    "Tesla T4": 4.0,
    "NVIDIA H100 80GB HBM3": 50.0,
}

#: The three arms the 1B-token grid trained. Order matters: the baseline is first and every
#: delta is computed against it.
ARM_NAMES: Tuple[str, ...] = ("L0", "F-r128", "G-grouped")

#: Prefill shapes as ``(batch_size, seq_len)``. 4096 is the training sequence length.
PREFILL_SHAPES: Tuple[Tuple[int, int], ...] = ((1, 4096), (4, 4096))

#: Decode batch sizes, at ``seq_len=1``. Batch 1 is the latency-critical single-stream case;
#: larger batches are where a served system actually runs and where the skinny GEMMs widen.
DECODE_BATCHES: Tuple[int, ...] = (1, 8, 32)


@dataclass
class Row:
    """One (arm, regime, shape) measurement, with the receipts that make it checkable."""

    arm: str
    regime: str
    batch_size: int
    seq_len: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    iters: int
    params_total: int
    #: Weight bytes the timed region reads. For a full forward every weight is read once.
    working_set_mib: float
    achieved_gbs: float
    pct_of_hbm_peak: float
    cache_resident: bool
    conv_path: str
    #: Populated for non-baseline arms once the baseline is known.
    vs_baseline_pct: Optional[float] = None
    tokens_per_s: Optional[float] = None


@dataclass
class Report:
    device: str
    hbm_peak_gbs: float
    l2_mib: Optional[float]
    dtype: str
    vocab_size: int
    torch_version: str
    use_fla_pinned: bool
    fla_importable: bool
    rows: List[Row] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Weight bytes actually read by a forward pass.
#
# This is computed from the BUILT MODULE, by walking parameters, rather than from the
# analytic formula in `ShortConvConfig.num_params`. A formula re-derived here would agree with
# the code by construction and could not detect a build that differs from the config -- the
# same shape as a test that recomputes what it is checking. Walking the real parameters means a
# gate structure that failed to apply shows up as a working set identical to dense.
# ---------------------------------------------------------------------------------------------
def weight_bytes(model) -> int:
    """Bytes of distinct parameter storage a single forward pass reads.

    Tied parameters are counted ONCE, keyed by storage identity. The embedding is tied to the
    unembedding in every arm, and double-counting it would add ~205 MiB of phantom traffic to
    all three arms equally -- which would not change the ratio, but would make
    ``pct_of_hbm_peak`` overstate and could mask a genuinely cache-resident row.
    """
    seen = set()
    total = 0
    for p in model.parameters():
        key = p.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += p.numel() * p.element_size()
    return total


def count_params(model) -> int:
    """Distinct parameters, tied weights counted once."""
    seen = set()
    total = 0
    for p in model.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel()
    return total


def realised_conv_path(model) -> str:
    """Which convolution implementation the LIV layers will actually execute.

    Reads the dispatch condition out of the built modules rather than assuming it, because the
    condition includes ``has_fla()`` -- environment state, not a property of the arm. Returns a
    single string and RAISES if the layers disagree with each other, since a model where some
    LIV layers fuse and others do not is not a configuration anybody chose.
    """
    from olmo_core.nn.attention.flash_linear_attn_api import has_fla
    from olmo_core.nn.attention.short_conv import ShortConv

    paths = set()
    for module in model.modules():
        if isinstance(module, ShortConv):
            paths.add("fla" if (module.use_fla and has_fla()) else "nn.Conv1d")
    if not paths:
        raise RuntimeError(
            "no ShortConv layers found in the built model -- the per-layer mixer override did "
            "not apply. Note the field is `sequence_mixer`, NOT `attention`: setting "
            "`.attention` on a block config silently creates a new attribute and leaves every "
            "layer as attention, producing a model that runs and answers a different question."
        )
    if len(paths) > 1:
        raise RuntimeError(f"LIV layers disagree on convolution path: {sorted(paths)}")
    return paths.pop()


def build_arm_config(arm_name: str, *, vocab_size: int, use_fla: bool, seed: int):
    """The ``TransformerConfig`` for one arm, with ``use_fla`` pinned on every LIV layer.

    Goes through ``arms_for_vocab`` rather than ``ARMS`` directly. ``A16-P`` and ``N-narrow``
    carry geometry SOLVED against a target that moves with the vocabulary, and ``build_arm``
    does not re-solve it. The three arms measured here do not have derived geometry, so this
    is belt-and-braces -- but reaching for the un-resolved dict is the habit that silently
    reports a capacity control as parameter-matched when it is not.

    ``init_seed`` is set on the CONFIG because that is the only place it is read from:
    ``Transformer.init_weights`` takes no generator and seeds itself from ``self.init_seed``.
    Passing a generator would be a silent no-op via ``**kwargs``.
    """
    from dataclasses import replace as dc_replace

    from olmo_core.nn.attention.short_conv import ShortConvConfig
    from olmo_core.nn.transformer.liv_arms import arms_for_vocab, build_arm

    arms = arms_for_vocab(vocab_size)
    if arm_name not in arms:
        raise KeyError(f"unknown arm {arm_name!r}; known: {sorted(arms)}")

    cfg = build_arm(arms[arm_name], vocab_size=vocab_size, init_device="meta")
    cfg.init_seed = seed

    # Pin `use_fla` on every ShortConv config the builder produced. `build_arm` never sets it,
    # so it sits at its `True` default and dispatches on whether `fla` happens to be
    # importable. Pinning it per arm is what makes the contrast fair; asserting the realised
    # path afterwards is what makes the pin verifiable.
    def pin(block_cfg):
        mixer = getattr(block_cfg, "sequence_mixer", None)
        if isinstance(mixer, ShortConvConfig):
            return dc_replace(block_cfg, sequence_mixer=dc_replace(mixer, use_fla=use_fla))
        return block_cfg

    cfg.block = pin(cfg.block)
    if cfg.block_overrides:
        cfg.block_overrides = {i: pin(b) for i, b in cfg.block_overrides.items()}
    return cfg


def build_one_arm(arm_name: str, *, vocab_size: int, dtype, use_fla: bool, device, seed: int):
    """Build one arm on ``device``, INITIALISED, with gates pinned identically.

    ``TransformerConfig.build`` constructs modules without initialising them, and
    ``init_weights`` must run before anything is timed. Uninitialised CUDA memory is often all
    zeros, and a kernel fed zeros or subnormals can take a different amount of time than one
    fed real values -- so timing an uninitialised model is timing something other than the
    model, and a dead branch would look merely inert rather than broken.

    Order matters and is not interchangeable: ``init_weights`` calls ``to_empty(device)``,
    which allocates FRESH storage and then re-establishes weight tying. Casting to ``dtype``
    before that call would have its result thrown away. So build -> init -> cast -> eval.
    """
    cfg = build_arm_config(arm_name, vocab_size=vocab_size, use_fla=use_fla, seed=seed)
    model = cfg.build(init_device="meta")
    model.init_weights(device=device)
    model.to(dtype=dtype)
    model.eval()
    return model


def time_forward(
    model,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int,
    iters: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Median, p10 and p90 milliseconds for one forward pass.

    Timed with CUDA events around each individual call, with a synchronize BEFORE the start
    event so queued work from the previous iteration cannot land inside this one's window.

    Returns percentiles rather than a mean because a single interfering kernel or a clock
    excursion moves a mean and leaves a median alone. Reporting p10/p90 alongside means a
    reader can see whether the median is stable rather than take it on faith -- a spread wider
    than the between-arm delta would mean the measurement cannot resolve the effect.
    """
    import torch

    dev = next(model.parameters()).device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Fixed input across arms: same tokens, so any difference is the model and not the data.
    tokens = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=gen, dtype=torch.long
    ).to(dev)

    def once():
        model(tokens)

    with torch.no_grad():
        for _ in range(warmup):
            once()
        torch.cuda.synchronize()

        samples: List[float] = []
        for _ in range(iters):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            once()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))

    samples.sort()
    return (
        statistics.median(samples),
        samples[max(0, int(0.10 * len(samples)) - 1)],
        samples[min(len(samples) - 1, int(0.90 * len(samples)))],
    )


def measure_arm(
    arm_name: str,
    *,
    vocab_size: int,
    dtype,
    use_fla: bool,
    device,
    peak_gbs: float,
    warmup: int,
    iters: int,
    seed: int,
    prefill_shapes: Sequence[Tuple[int, int]],
    decode_batches: Sequence[int],
) -> Tuple[List[Row], str, int]:
    """Build one arm, measure every regime, free it. Returns (rows, conv_path, params)."""
    import torch

    model = build_one_arm(
        arm_name, vocab_size=vocab_size, dtype=dtype, use_fla=use_fla, device=device, seed=seed
    )

    conv_path = realised_conv_path(model)
    params = count_params(model)
    wbytes = weight_bytes(model)
    working_mib = wbytes / 2**20

    log.info(
        "%s: %s params, %.1f MiB of weights, conv path %s",
        arm_name,
        f"{params:,}",
        working_mib,
        conv_path,
    )

    rows: List[Row] = []
    plan = [("prefill", b, s) for b, s in prefill_shapes] + [
        ("decode", b, 1) for b in decode_batches
    ]

    for regime, batch_size, seq_len in plan:
        median_ms, p10, p90 = time_forward(
            model,
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            warmup=warmup,
            iters=iters,
            seed=seed,
        )
        achieved = wbytes / (median_ms / 1e3) / 1e9  # GB/s, decimal, matching vendor figures
        pct_peak = achieved / peak_gbs * 100.0
        rows.append(
            Row(
                arm=arm_name,
                regime=regime,
                batch_size=batch_size,
                seq_len=seq_len,
                median_ms=median_ms,
                p10_ms=p10,
                p90_ms=p90,
                iters=iters,
                params_total=params,
                working_set_mib=working_mib,
                achieved_gbs=achieved,
                pct_of_hbm_peak=pct_peak,
                cache_resident=pct_peak > 100.0,
                conv_path=conv_path,
                tokens_per_s=batch_size * seq_len / (median_ms / 1e3),
            )
        )
        log.info(
            "  %-8s b=%-3d s=%-5d  %8.3f ms  [p10 %.3f p90 %.3f]  %7.1f GB/s (%5.1f%% peak)",
            regime,
            batch_size,
            seq_len,
            median_ms,
            p10,
            p90,
            achieved,
            pct_peak,
        )

    del model
    torch.cuda.empty_cache()
    return rows, conv_path, params


def attach_deltas(rows: List[Row], baseline_arm: str) -> None:
    """Fill ``vs_baseline_pct`` in place, matching rows by (regime, batch, seq_len).

    Positive means FASTER than the baseline. A missing baseline row is an error rather than a
    skipped delta: silently omitting a comparison is how an empty comparison set comes to read
    as a clean result.
    """
    key = lambda r: (r.regime, r.batch_size, r.seq_len)  # noqa: E731
    base = {key(r): r for r in rows if r.arm == baseline_arm}
    for row in rows:
        if row.arm == baseline_arm:
            continue
        b = base.get(key(row))
        if b is None:
            raise RuntimeError(
                f"no {baseline_arm} row for {key(row)} -- cannot compute a delta, and "
                f"reporting the row without one would read as a result."
            )
        row.vs_baseline_pct = (b.median_ms - row.median_ms) / b.median_ms * 100.0


def summarise(report: Report, baseline_arm: str) -> None:
    """Print the table, the caveats, and the share-weighting arithmetic."""
    print(f"\ndevice: {report.device}")
    print(
        f"HBM peak {report.hbm_peak_gbs:.0f} GB/s"
        + (f", L2 {report.l2_mib:.0f} MiB" if report.l2_mib else "")
        + f", dtype {report.dtype}, torch {report.torch_version}"
    )
    print(
        f"use_fla pinned to {report.use_fla_pinned} on every arm; "
        f"fla importable: {report.fla_importable}"
    )

    print(
        f"\n{'arm':<11} {'regime':<8} {'b':>4} {'seq':>6} {'median ms':>10} "
        f"{'GB/s':>8} {'%peak':>7} {'vs base':>9}  conv"
    )
    for r in report.rows:
        delta = "  baseline" if r.vs_baseline_pct is None else f"{r.vs_baseline_pct:+8.2f}%"
        flag = " CACHE_RESIDENT" if r.cache_resident else ""
        print(
            f"{r.arm:<11} {r.regime:<8} {r.batch_size:>4} {r.seq_len:>6} "
            f"{r.median_ms:>10.3f} {r.achieved_gbs:>8.1f} {r.pct_of_hbm_peak:>6.1f}% "
            f"{delta:>9}  {r.conv_path}{flag}"
        )

    # Parameter savings, which is the other half of the efficiency claim and is exact.
    by_arm = {}
    for r in report.rows:
        by_arm.setdefault(r.arm, r.params_total)
    base_params = by_arm.get(baseline_arm)
    if base_params:
        print(f"\nparameters (tied embedding counted once), baseline {baseline_arm}:")
        for arm, n in by_arm.items():
            d = base_params - n
            print(
                f"  {arm:<11} {n:>13,}"
                + (f"   -{d:,} ({d / base_params * 100:.2f}%)" if d else "   baseline")
            )

    print("\nnotes:")
    for n in report.notes:
        print(f"  - {n}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--baseline", default="L0")
    ap.add_argument("--vocab-size", type=int, default=100_352)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--use-fla",
        default="false",
        choices=("true", "false"),
        help="Pinned identically on every arm. Default false: `fla` is absent from the "
        "research image, so leaving the module default (True) would let the dispatch "
        "condition depend on the environment rather than on the declared config.",
    )
    ap.add_argument(
        "--device-peak-gbs",
        type=float,
        default=None,
        help="Override the HBM peak. Only needed for a card missing from HBM_PEAK_GBS; the "
        "run refuses rather than guessing.",
    )
    ap.add_argument(
        "--fail-on-cache-resident",
        action="store_true",
        default=True,
        help="Exit non-zero if any row exceeds 100%% of HBM peak. On by default: a "
        "cache-resident row is the exact failure that produced the retracted -8.2%%.",
    )
    ap.add_argument("--allow-cache-resident", dest="fail_on_cache_resident", action="store_false")
    ap.add_argument("--out", default=None, help="Where to write the JSON report.")
    opts = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )

    try:
        import torch
    except ImportError:
        print("ERROR: torch is not importable.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA device. This measures per-card inference latency and is "
            "meaningless on CPU -- a CPU number would not be a slower version of the answer, "
            "it would be a different question. Run on the platform (gpu-8xa100 with "
            "EDULLM_LAUNCH_CHECK=waived, or gpu-1xl40s) or on a FarmShare GPU node.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)

    peak = opts.device_peak_gbs or HBM_PEAK_GBS.get(device_name)
    if peak is None:
        print(
            f"ERROR: no HBM peak recorded for {device_name!r}. Pass --device-peak-gbs with the "
            f"vendor figure. Refusing to default: the peak is the denominator of the only check "
            f"that can detect a cache-resident measurement, and assuming one would restore the "
            f"blind spot that produced the retracted -8.2% result.",
            file=sys.stderr,
        )
        return 1

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        opts.dtype
    ]
    use_fla = opts.use_fla == "true"

    try:
        from olmo_core.nn.attention.flash_linear_attn_api import has_fla

        fla_importable = has_fla()
    except Exception:
        fla_importable = False

    arms = [a.strip() for a in opts.arms.split(",") if a.strip()]
    if opts.baseline not in arms:
        print(
            f"ERROR: baseline {opts.baseline!r} is not in --arms ({arms}). Every delta is "
            f"computed against it, so a run without it produces no comparisons at all.",
            file=sys.stderr,
        )
        return 1

    report = Report(
        device=device_name,
        hbm_peak_gbs=peak,
        l2_mib=L2_MIB.get(device_name),
        dtype=opts.dtype,
        vocab_size=opts.vocab_size,
        torch_version=torch.__version__,
        use_fla_pinned=use_fla,
        fla_importable=fla_importable,
    )

    log.info("run %s on %s (%s), %s", opts.run_id, device_name, platform.node(), opts.dtype)

    conv_paths: Dict[str, str] = {}
    for arm_name in arms:
        rows, conv_path, _ = measure_arm(
            arm_name,
            vocab_size=opts.vocab_size,
            dtype=dtype,
            use_fla=use_fla,
            device=device,
            peak_gbs=peak,
            warmup=opts.warmup,
            iters=opts.iters,
            seed=opts.seed,
            prefill_shapes=PREFILL_SHAPES,
            decode_batches=DECODE_BATCHES,
        )
        report.rows.extend(rows)
        conv_paths[arm_name] = conv_path

    # The fairness assertion. A contrast in which one arm fused its convolution and another
    # did not would attribute a kernel difference to gate structure, and the bias points
    # toward the hypothesis -- which is the direction that gets believed.
    if len(set(conv_paths.values())) > 1:
        print(
            f"ERROR: arms executed different convolution paths: {conv_paths}. The contrast "
            f"would compare kernels, not gate structures. Pin --use-fla and re-run.",
            file=sys.stderr,
        )
        return 1

    attach_deltas(report.rows, opts.baseline)

    report.notes = [
        f"conv path was {next(iter(conv_paths.values()))} on every arm ({conv_paths}).",
        "decode rows are seq_len=1 forwards, NOT autoregressive steps: ShortConv has no "
        "conv-state cache and attention runs without a KV cache. A served decode adds "
        "KV-cache traffic identical across arms, which dilutes the ratio -- so the decode "
        "delta here is an UPPER BOUND on the served-decode speedup.",
        "prefill rows carry no such caveat and are the conservative rung.",
        "share-weight before quoting: the gates are ~5.38% of parameters, so a large "
        "subgraph win is a small end-to-end win. These rows are already end-to-end.",
        f"cache check: every row's achieved bandwidth compared against {peak:.0f} GB/s; "
        f"{sum(1 for r in report.rows if r.cache_resident)} row(s) exceeded peak.",
    ]

    summarise(report, opts.baseline)

    out_path = opts.out or f"gate_latency_{opts.run_id}.json"
    payload = {
        "run_id": opts.run_id,
        "device": report.device,
        "hbm_peak_gbs": report.hbm_peak_gbs,
        "l2_mib": report.l2_mib,
        "dtype": report.dtype,
        "vocab_size": report.vocab_size,
        "torch_version": report.torch_version,
        "use_fla_pinned": report.use_fla_pinned,
        "fla_importable": report.fla_importable,
        "baseline": opts.baseline,
        "warmup": opts.warmup,
        "iters": opts.iters,
        "notes": report.notes,
        "rows": [asdict(r) for r in report.rows],
    }
    try:
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {out_path}")
    except OSError as exc:  # a read-only cwd must not lose the numbers already printed
        log.warning("could not write %s (%s); the table above is the record", out_path, exc)

    # Also drop a copy beside the checkpoint dir, which is the path that survives the job.
    ckpt = os.environ.get("EDULLM_CHECKPOINT_DIR")
    if ckpt:
        try:
            os.makedirs(ckpt, exist_ok=True)
            side = os.path.join(ckpt, os.path.basename(out_path))
            with open(side, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"wrote {side}")
        except OSError as exc:
            log.warning("could not write into EDULLM_CHECKPOINT_DIR (%s)", exc)

    resident = [r for r in report.rows if r.cache_resident]
    if resident and opts.fail_on_cache_resident:
        print(
            f"\nERROR: {len(resident)} row(s) read above {peak:.0f} GB/s, which is impossible "
            f"for an HBM-bound region and therefore proves the timed region was cache "
            f"resident. That is the exact failure behind the retracted -8.2% figure. The "
            f"numbers above are NOT a valid latency comparison. Rerun at a larger shape, or "
            f"pass --allow-cache-resident if you are deliberately measuring the cached "
            f"regime.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
