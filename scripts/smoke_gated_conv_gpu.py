"""
GPU smoke test for the gated short convolution inside KDA.

WHAT THIS PROVES, AND WHAT IT DOES NOT
    It proves the three arms **build, run forward and backward, and produce live gradients on a
    real GPU with the fused kernels**, and it measures the peak memory each one costs. It does
    **not** measure quality: 30 steps on synthetic tokens says nothing about held-out CE, and any
    loss difference at this length is noise.

    This box is L40S / sm_89. The training target is A100 / sm_80. So a pass here means the stack
    works, not that the numbers transfer -- ``fla``'s Triton kernels are compiled per architecture
    and KDA's backward had an sm-dependent illegal-access bug (fla #802, fixed in 0.5.0). The
    A100 probe is a separate, paid run.

THE CHECKS THAT CAN ACTUALLY FAIL
    Each is a gate with a stated threshold, and each is reachable in this regime -- a guard that
    cannot fire in the regime it runs in is worse than no guard, because it reads as a pass.

    1. **The fused kernel is really used.** The gated module falls back to ``torch.nn.Conv1d`` when
       ``fla`` is missing or the tensor is on CPU. If the plain arm got the fused path and the
       gated arm got the fallback, the arms would differ in numerics and speed for a reason that
       has nothing to do with gating. This project has already been burned by exactly that
       (``short_conv.py`` defaulting ``use_fla=True`` against an absent ``fla``). Asserted by
       reading the realised path, not by assuming it.
    2. **The gate is alive after one real optimizer step.** Against a floor relative to a
       known-live parameter, not an absolute number, and not ``is not None``.
    3. **The gate has actually moved off neutral by the end.** A gate that receives gradient and
       still sits at exactly 1.0 after 30 steps is decorative.
    4. **Peak memory is recorded per arm.** This is the real cost of the experiment and the number
       a run gets sized from. Predicted against measured, so a bad prediction is visible.
    5. **The loss is in the right absolute band.** A fresh model on a uniform-random vocabulary
       must start near ``ln(vocab)``. Checking only that loss *decreased* passes for a model
       reading its own targets.

USAGE
    srun -p gpu --gres=gpu:1 -c 8 --mem=64G -t 00:40:00 \
        python scripts/smoke_gated_conv_gpu.py --out smoke.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

# Argv handling stays ABOVE the torch import so --help and the refusals work without a GPU, and so
# a test can reach them. Putting them below is how a guard becomes unreachable.
_ARMS = ("kda-plain", "kda-gated", "kda-gated-silu")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--arms", default=",".join(_ARMS))
    ap.add_argument("--gate-structure", default="depthwise", choices=("depthwise", "lowrank"))
    ap.add_argument("--gate-rank", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--allow-eager-conv",
        action="store_true",
        help="permit the unfused torch.nn.Conv1d fallback; without this a fallback is a refusal",
    )
    return ap


#: How far below a known-live parameter's gradient a gate gradient may sit before the branch is
#: dead in practice. bf16 keeps 8 mantissa bits, so anything more than 2**-9 down is lost to
#: accumulation. Relative, because an absolute threshold fires on the input's variance instead.
GATE_LIVENESS_RATIO = 2.0**-9

#: How far the gate must have moved off its neutral 1.0 by the end of the run. A gate that gets
#: gradient and does not move is decorative, and reports a clean null.
GATE_MOVEMENT_FLOOR = 1e-4

#: A fresh model on uniform-random targets must start near ln(vocab). Wider than it looks because
#: a few steps have already run by the time the first loss is recorded.
INITIAL_LOSS_TOLERANCE = 0.5


def loss_band(vocab_size: int) -> tuple[float, float]:
    """
    The absolute band a fresh model's first loss must land in.

    Gating on "loss decreased" is what let a ``targets == inputs`` bug pass 74 tests in this
    project, because a model reading its own input decreases beautifully. ``ln(vocab)`` is the
    entropy of a uniform draw, which is what an untrained model on random data must score.

    :param vocab_size: The vocabulary size.

    :returns: The ``(low, high)`` band.
    """
    expected = math.log(vocab_size)
    return expected - INITIAL_LOSS_TOLERANCE, expected + INITIAL_LOSS_TOLERANCE


def gate_is_alive(gate_grad: Optional[float], reference_grad: Optional[float]) -> tuple[bool, str]:
    """
    Whether a gate's gradient is large enough to produce a usable update.

    Extracted so a CPU test can call it, and so the threshold is one named thing rather than an
    inline comparison in the middle of a training loop.

    :param gate_grad: Max absolute gradient on the gate parameters.
    :param reference_grad: Max absolute gradient on a parameter known to be training.

    :returns: ``(alive, explanation)``.
    """
    if gate_grad is None:
        return False, "no gate gradient was recorded"
    if reference_grad is None or reference_grad <= 0.0:
        # Fail closed. If the reference is itself dead the comparison carries no information, and
        # calling that a pass is the guard-that-cannot-fire failure.
        return False, "the reference gradient is zero or missing, so liveness is unmeasurable"
    floor = GATE_LIVENESS_RATIO * reference_grad
    if gate_grad <= floor:
        return False, (
            f"gate gradient {gate_grad:.3e} is at or below {floor:.3e} "
            f"({GATE_LIVENESS_RATIO:.1e} of the reference {reference_grad:.3e}): "
            "the branch cannot accumulate a usable update"
        )
    return True, f"gate gradient {gate_grad:.3e} vs floor {floor:.3e}"


def conv_path_is_fused(realised_paths: List[str], *, allow_eager: bool) -> tuple[bool, str]:
    """
    Whether every convolution ran the fused kernel.

    A mixed run is the dangerous case: the arms then differ in numerics *and* speed for a reason
    unrelated to the gate, and nothing in a loss curve shows it. An empty list is a refusal, not a
    pass -- an empty comparison set reporting success is a failure mode this project has shipped
    four times in one build.

    :param realised_paths: One entry per convolution actually executed.
    :param allow_eager: Whether the unfused fallback is permitted.

    :returns: ``(ok, explanation)``.
    """
    if not realised_paths:
        return False, "no convolution path was recorded, so this check proves nothing"
    kinds = sorted(set(realised_paths))
    if kinds == ["fused"]:
        return True, f"all {len(realised_paths)} convolutions fused"
    if allow_eager:
        return True, f"paths {kinds} (eager permitted by --allow-eager-conv)"
    return False, (
        f"convolution paths were {kinds} across {len(realised_paths)} calls; an unfused "
        "fallback makes the arms incomparable in numerics and speed"
    )


def main(argv: Optional[List[str]] = None) -> int:
    opts = build_parser().parse_args(argv)

    arms = [a.strip() for a in opts.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in _ARMS]
    if unknown:
        print(f"ERROR: unknown arms {unknown}; choose from {list(_ARMS)}", file=sys.stderr)
        return 2
    if len(arms) < 2:
        print(
            "ERROR: at least two arms are needed. One arm cannot show that the gate changes "
            "anything, so a single-arm run reports a pass while measuring nothing.",
            file=sys.stderr,
        )
        return 2

    # TRITON_F32_DEFAULT is checked before torch, because it participates in kernel codegen: a
    # cache built without it is not reusable with it. The torch tf32 flag does NOT control Triton.
    triton_f32 = os.environ.get("TRITON_F32_DEFAULT")

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print(
            "ERROR: no GPU. This script exists to exercise the fused fla kernels, which do not "
            "run on CPU -- a CPU pass here would be the unfused fallback and would prove nothing.",
            file=sys.stderr,
        )
        return 3

    from olmo_core.nn.attention import KimiDeltaAttentionConfig
    from olmo_core.nn.attention.flash_linear_attn_api import has_fla
    from olmo_core.nn.gated_convolution import GatedCausalConv1d
    from olmo_core.nn.transformer.init import InitMethod

    if not has_fla():
        print(
            "ERROR: fla is not importable, so KimiDeltaAttention cannot be built.", file=sys.stderr
        )
        return 3

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} sm_{props.major}{props.minor}, {props.total_memory/2**30:.1f} GiB")
    print(f"torch {torch.__version__}, TRITON_F32_DEFAULT={triton_f32!r}")
    if triton_f32 != "ieee":
        print(
            "  NOTE: TRITON_F32_DEFAULT is not 'ieee'. Triton lowers tl.dot to TF32 on Ampere+ by "
            "its own default, which measured 166x worse fp32 accuracy in this project. It does "
            "not affect this smoke test's verdicts, but a real run must set it."
        )
    print(f"loss band for vocab {opts.vocab_size}: {loss_band(opts.vocab_size)}")
    print()

    def make_config(arm: str) -> KimiDeltaAttentionConfig:
        common: Dict[str, Any] = dict(
            n_heads=opts.n_heads,
            head_dim=opts.d_model // opts.n_heads,
            expand_v=1.0,
            conv_size=4,
        )
        if arm == "kda-plain":
            return KimiDeltaAttentionConfig(**common)
        cfg: Dict[str, Any] = dict(
            common,
            gated_conv=True,
            gate_structure=opts.gate_structure,
            gated_conv_activation="silu" if arm == "kda-gated-silu" else None,
        )
        if opts.gate_structure == "lowrank":
            cfg["gate_rank"] = opts.gate_rank
        return KimiDeltaAttentionConfig(**cfg)

    class TinyLM(torch.nn.Module):
        """Embedding, N KDA layers with a residual and a norm, and a tied-free head."""

        def __init__(self, cfg: KimiDeltaAttentionConfig):
            super().__init__()
            self.embed = torch.nn.Embedding(opts.vocab_size, opts.d_model)
            self.norms = torch.nn.ModuleList(
                [torch.nn.RMSNorm(opts.d_model) for _ in range(opts.n_layers)]
            )
            self.mixers = torch.nn.ModuleList(
                [
                    cfg.build(opts.d_model, layer_idx=i, n_layers=opts.n_layers)
                    for i in range(opts.n_layers)
                ]
            )
            self.out_norm = torch.nn.RMSNorm(opts.d_model)
            self.head = torch.nn.Linear(opts.d_model, opts.vocab_size, bias=False)

        def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
            h = self.embed(tokens)
            for norm, mixer in zip(self.norms, self.mixers):
                h = h + mixer(norm(h))
            return self.head(self.out_norm(h))

    records: List[Dict[str, Any]] = []
    failures: List[str] = []

    for arm in arms:
        print(f"=== {arm}")
        torch.manual_seed(opts.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        cfg = make_config(arm)
        # Built ON THE DEVICE and initialized there, with a device-matched generator. torch's
        # 'uniform_' refuses a CPU generator against a CUDA tensor, so building on CPU and moving
        # afterwards would work while initializing on the device would not -- the failure is
        # 'Expected a cuda device type for generator but found cpu' and it fires on the FIRST arm,
        # which means nothing about the module itself gets exercised.
        model = TinyLM(cfg).to(device)
        gen = torch.Generator(device=device).manual_seed(opts.seed)
        for i, mixer in enumerate(model.mixers):
            mixer.init_weights(
                init_method=InitMethod.normal,
                d_model=opts.d_model,
                block_idx=i,
                num_blocks=opts.n_layers,
                generator=gen,
            )
        model = model.to(torch.bfloat16)

        gated_convs = [
            c
            for m in model.mixers
            for c in (m.q_conv1d, m.k_conv1d, m.v_conv1d)
            if isinstance(c, GatedCausalConv1d)
        ]
        expected_gated = 0 if arm == "kda-plain" else 3 * opts.n_layers
        if len(gated_convs) != expected_gated:
            failures.append(
                f"{arm}: found {len(gated_convs)} gated convolutions, expected {expected_gated}"
            )

        # Record which path each convolution really takes, by instrumenting the module rather than
        # inferring it from a flag. 'use_fla=True' records the REQUEST; this records the execution.
        realised: List[str] = []
        for conv in gated_convs:
            original = conv._conv

            def wrapped(u, cu, _conv=conv, _orig=original):
                realised.append("fused" if (_conv.use_fla and has_fla() and u.is_cuda) else "eager")
                return _orig(u, cu)

            conv._conv = wrapped  # type: ignore[method-assign]

        gate_params = [
            p for c in gated_convs for n, p in c.named_parameters() if not n.startswith("conv.")
        ]
        gate_at_init = (
            None
            if not gate_params
            else max(float(p.detach().float().abs().max()) for p in gate_params)
        )

        optim = torch.optim.AdamW(model.parameters(), lr=opts.lr)
        losses: List[float] = []
        step1_gate_grad: Optional[float] = None
        step1_ref_grad: Optional[float] = None

        for step in range(opts.steps):
            tokens = torch.randint(
                0, opts.vocab_size, (opts.batch_size, opts.seq_len + 1), device=device
            )
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(
                logits.float().reshape(-1, opts.vocab_size), tokens[:, 1:].reshape(-1)
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()

            if step == 0:
                # The reference is the head, which is unambiguously training.
                if model.head.weight.grad is not None:
                    step1_ref_grad = float(model.head.weight.grad.float().abs().max())
                grads = [
                    float(p.grad.float().abs().max()) for p in gate_params if p.grad is not None
                ]
                step1_gate_grad = max(grads) if grads else None

            optim.step()
            losses.append(float(loss))
            if step % 10 == 0 or step == opts.steps - 1:
                print(f"  step {step:3d}  loss {loss:.4f}")

        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        gate_now = (
            None
            if not gate_params
            else max(float(p.detach().float().abs().max()) for p in gate_params)
        )
        gate_moved = None if gate_now is None or gate_at_init is None else gate_now - gate_at_init

        low, high = loss_band(opts.vocab_size)
        rec: Dict[str, Any] = {
            "arm": arm,
            "gate_structure": opts.gate_structure if arm != "kda-plain" else None,
            "num_params": sum(p.numel() for p in model.parameters()),
            "mixer_params_predicted": cfg.num_params(opts.d_model) * opts.n_layers,
            "mixer_params_actual": sum(p.numel() for m in model.mixers for p in m.parameters()),
            "gate_params": cfg.gate_params(opts.d_model) * opts.n_layers,
            "gate_bytes_predicted": cfg.gate_activation_bytes(
                opts.d_model, batch_size=opts.batch_size, seq_len=opts.seq_len
            )
            * opts.n_layers,
            "peak_gib": peak_gib,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "gate_grad_step1": step1_gate_grad,
            "ref_grad_step1": step1_ref_grad,
            "gate_movement": gate_moved,
            "conv_paths": sorted(set(realised)),
            "conv_calls": len(realised),
            "gated_conv_count": len(gated_convs),
        }

        # --- the gates, each with a stated threshold and each reachable in this regime ---

        if rec["mixer_params_predicted"] != rec["mixer_params_actual"]:
            failures.append(
                f"{arm}: config predicts {rec['mixer_params_predicted']} mixer params, "
                f"module has {rec['mixer_params_actual']}"
            )

        if not (low <= losses[0] <= high):
            failures.append(
                f"{arm}: first loss {losses[0]:.4f} is outside [{low:.3f}, {high:.3f}] = "
                f"ln({opts.vocab_size}) +/- {INITIAL_LOSS_TOLERANCE}. A fresh model on random "
                "targets must start at the uniform entropy; outside the band means the data or "
                "the target alignment is wrong, and 'loss went down' would not have caught it."
            )

        if not math.isfinite(losses[-1]):
            failures.append(f"{arm}: final loss is {losses[-1]}")

        if arm != "kda-plain":
            ok, why = conv_path_is_fused(realised, allow_eager=opts.allow_eager_conv)
            print(f"  conv path: {why}")
            if not ok:
                failures.append(f"{arm}: {why}")

            ok, why = gate_is_alive(step1_gate_grad, step1_ref_grad)
            print(f"  gate liveness: {why}")
            if not ok:
                failures.append(f"{arm}: {why}")

            print(f"  gate moved {gate_moved!r} off neutral over {opts.steps} steps")
            if gate_moved is None or gate_moved < GATE_MOVEMENT_FLOOR:
                failures.append(
                    f"{arm}: the gate moved {gate_moved!r}, below {GATE_MOVEMENT_FLOOR}. It "
                    "receives gradient but is not actually changing the function, so the arm is "
                    "the control wearing a different name."
                )

        print(f"  peak memory {peak_gib:.3f} GiB, params {rec['num_params']:,}")
        print()
        records.append(rec)

        del model, optim
        torch.cuda.empty_cache()

    # --- cross-arm checks. These need every arm, so they are here rather than in the loop. ---

    by_arm = {r["arm"]: r for r in records}
    if "kda-plain" in by_arm and "kda-gated" in by_arm:
        p, g = by_arm["kda-plain"], by_arm["kda-gated"]
        delta = g["mixer_params_actual"] - p["mixer_params_actual"]
        print(f"parameter delta gated - plain: {delta:,} ({delta/p['mixer_params_actual']:.4%})")
        if delta != g["gate_params"]:
            failures.append(
                f"the measured parameter delta {delta} does not equal the predicted gate cost "
                f"{g['gate_params']}"
            )
        mem_ratio = g["peak_gib"] / p["peak_gib"] if p["peak_gib"] else float("inf")
        print(f"peak memory ratio gated / plain: {mem_ratio:.3f}x")
        print(
            f"  predicted gate activation bytes: {g['gate_bytes_predicted']/2**30:.3f} GiB, "
            f"measured increase: {g['peak_gib']-p['peak_gib']:.3f} GiB"
        )
        if mem_ratio < 1.0:
            failures.append(
                f"the gated arm used LESS peak memory than the plain arm ({mem_ratio:.3f}x). The "
                "gates retain backward activations, so this means the measurement is wrong, not "
                "that gating is free."
            )

    if {"kda-gated", "kda-gated-silu"} <= by_arm.keys():
        a, b = by_arm["kda-gated"], by_arm["kda-gated-silu"]
        if a["mixer_params_actual"] != b["mixer_params_actual"]:
            failures.append(
                "kda-gated and kda-gated-silu must be parameter-identical, so that any difference "
                "between them is the activation and nothing else"
            )
        if a["loss_last"] == b["loss_last"]:
            failures.append(
                "kda-gated and kda-gated-silu produced an identical final loss, which means the "
                "activation is not reaching the module and the two arms are the same run"
            )

    payload = {
        "device": props.name,
        "sm": f"{props.major}{props.minor}",
        "torch": torch.__version__,
        "triton_f32_default": triton_f32,
        "opts": vars(opts),
        "loss_band": list(loss_band(opts.vocab_size)),
        "records": records,
        "failures": failures,
    }
    if opts.out:
        with open(opts.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {opts.out}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("All checks passed. Note this is sm_89; sm_80 must be verified separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
