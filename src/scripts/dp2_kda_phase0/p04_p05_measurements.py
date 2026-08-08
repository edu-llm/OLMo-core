"""Runbook §4.6 decay stress and §4.7 R1/R2 timing + peak-memory measurements.

Two things the §4.7 named-test table does *not* name, and which are therefore not covered by the
13 gate tests:

* **§4.6 "Decay stress"** -- near-zero retention, strong decay, and the separate reflection-beta
  regime. Each regime is compared against the float64 oracle and the realized error reported.
* **The measurement block at the end of §4.7** -- R1 and R2 at the probe geometry, forward-only /
  backward-only / forward+backward, bf16 inputs with float32 recurrence accumulation, warm kernels
  excluded from timing, peak allocated and reserved memory, and throughput expressed in **logical
  real tokens per second** (``B * T``), never virtual positions per second (``B * T * R``).

Run under the Phase-0 environment (``$KDA_PROBES_DIR`` and ``PYTHONPATH`` set) on an L40S.
Emits JSON on stdout. This is a measurement harness, not a pass/fail gate.
"""

import argparse
import json
import statistics
import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F


def _torch_backend() -> Callable[..., Tuple[torch.Tensor, Any]]:
    """:returns: the differentiable pure-torch KDA-Householder backend."""
    from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

    return kda_householder_torch


def _triton_backend() -> Callable[..., Tuple[torch.Tensor, Any]]:
    """:returns: the dispatching entry point ``chunk_kda_householder``."""
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    return chunk_kda_householder


# ------------------------------------------------------------------------------------------
# §4.6 decay stress
# ------------------------------------------------------------------------------------------


def decay_stress(device: torch.device) -> List[Dict[str, Any]]:
    """Stress the decay/beta extremes of the §4.6 table against two independent references.

    Three regimes are exercised, per the §4.6 table row:

    * ``near_zero_retention`` -- ``g`` far below zero, so ``exp(g) ~ 0`` and the state is wiped
      every step. This is where a decay applied the wrong number of times shows up loudest.
    * ``strong_decay`` -- ``g`` moderately negative, a realistic aggressive-forgetting regime.
    * ``reflection_beta`` -- ``beta`` in ``(0, 2)`` (the ``allow_neg_eigval`` doubling) rather than
      ``(0, 1)``, which makes ``I - beta k k^T`` a reflection rather than a contraction.

    .. important::
       ``kda_householder_torch`` versus the probe oracle is **not** an independent comparison.
       §4.5 states it explicitly and the measurement confirms it: the realized difference is
       exactly ``0.0`` in every regime, because the torch backend is a transcription of the probe
       using the same einsum calls in the same order -- they are *one* oracle, not two. That
       column is retained only as a transcription check and its zeros must not be read as
       evidence about the decay regimes.

       The load-bearing column is ``triton_vs_torch_rel``, which compares the fused Triton kernel
       -- genuinely separate code -- against the reference under the same extreme inputs. Triton
       runs in bf16 on GPU, so its floor is bf16 round-off, not float64 ulp.

    :param device: device for the float64 reference arm (CPU is fine; Triton always uses CUDA).

    :returns: one record per (regime, R).
    """
    oracle = _load_oracle_recurrence()
    kda_torch = _torch_backend()
    out: List[Dict[str, Any]] = []

    B, T, H, K, V = 2, 32, 2, 16, 16
    for regime in ("near_zero_retention", "strong_decay", "reflection_beta", "baseline"):
        for R in (1, 2, 3):
            gen = torch.Generator().manual_seed(hash((regime, R)) % (2**31))

            def rnd(*shape: int) -> torch.Tensor:
                return torch.randn(*shape, generator=gen, dtype=torch.float64)

            q = F.normalize(rnd(B, T, H, K), p=2, dim=-1)
            k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
            v = rnd(B, T * R, H, V)

            if regime == "near_zero_retention":
                # exp(g) ~ e^-30: the state is annihilated between tokens.
                g = torch.full((B, T, H, K), -30.0, dtype=torch.float64)
                beta = rnd(B, T * R, H).sigmoid()
            elif regime == "strong_decay":
                g = torch.full((B, T, H, K), -3.0, dtype=torch.float64) + 0.1 * rnd(B, T, H, K)
                beta = rnd(B, T * R, H).sigmoid()
            elif regime == "reflection_beta":
                g = F.logsigmoid(rnd(B, T, H, K))
                # The allow_neg_eigval doubling: beta in (0, 2), so I - beta k k^T reflects.
                beta = 2.0 * rnd(B, T * R, H).sigmoid()
            else:
                g = F.logsigmoid(rnd(B, T, H, K))
                beta = rnd(B, T * R, H).sigmoid()

            scale = K**-0.5
            o, s = kda_torch(
                q, k, v, g, beta, num_householder=R, scale=scale, output_final_state=True
            )
            o_ref, s_ref = oracle(q, k, v, g, beta, R, scale, None, torch.float64)

            denom = o_ref.abs().max().item()
            diff = (o.double() - o_ref.double()).abs().max().item()
            sdiff = (s.double() - s_ref.double()).abs().max().item()

            rec: Dict[str, Any] = {
                "regime": regime,
                "R": R,
                # Transcription check only -- see the note in this function's docstring.
                "torch_vs_probe_abs_err": diff,
                "torch_vs_probe_rel_err": diff / denom if denom > 0 else float("nan"),
                "torch_vs_probe_state_abs_err": sdiff,
                "output_absmax": denom,
                "min_exp_g": float(g.exp().min().item()),
                "beta_max": float(beta.max().item()),
                "torch_finite": bool(torch.isfinite(o).all() and torch.isfinite(s).all()),
            }
            rec.update(_triton_decay_arm(q, k, v, g, beta, R=R, scale=scale, o_ref=o_ref))
            out.append(rec)
    return out


def _triton_decay_arm(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    R: int,
    scale: float,
    o_ref: torch.Tensor,
) -> Dict[str, Any]:
    """Run the fused Triton kernel on the same extreme inputs and compare against the reference.

    This is the genuinely independent arm of the decay stress: the Triton kernel shares no code
    with the probe oracle. It consumes bf16 on CUDA, so the achievable agreement is floored by
    bf16 round-off rather than by float64 ulp.

    :param q: float64 queries ``[B, T, H, K]``.
    :param k: float64 keys ``[B, T * R, H, K]``.
    :param v: float64 values ``[B, T * R, H, V]``.
    :param g: float64 per-channel log-decay ``[B, T, H, K]``.
    :param beta: float64 write strengths ``[B, T * R, H]``.
    :param R: number of Householder factors.
    :param scale: query scale.
    :param o_ref: the float64 reference output to compare against.

    :returns: a dict of Triton-arm metrics, or a reason string if CUDA/triton is unavailable.
    """
    if not torch.cuda.is_available():
        return {"triton_skipped": "no CUDA device"}
    try:
        from olmo_core.nn.attention.kda_householder import chunk_kda_householder
    except ImportError as exc:  # pragma: no cover - environment-dependent
        return {"triton_skipped": f"import failed: {exc}"}

    dev = torch.device("cuda")
    o_tri, s_tri = chunk_kda_householder(
        q=q.to(dev, torch.bfloat16),
        k=k.to(dev, torch.bfloat16),
        v=v.to(dev, torch.bfloat16),
        g=g.to(dev, torch.float32),  # gate stays fp32: the fp32 accumulation path
        beta=beta.to(dev, torch.bfloat16),
        num_householder=R,
        scale=scale,
        output_final_state=True,
        backend="triton",
    )
    denom = o_ref.abs().max().item()
    d = (o_tri.double().cpu() - o_ref.double()).abs().max().item()
    return {
        "triton_vs_ref_abs": d,
        "triton_vs_ref_rel": d / denom if denom > 0 else float("nan"),
        "triton_finite": bool(torch.isfinite(o_tri).all() and torch.isfinite(s_tri).all()),
    }


def _load_oracle_recurrence() -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
    """Import the out-of-repo probe oracle, mirroring the test suite's loader.

    The dtype-parameterised ``_recurrence`` body is used rather than the public entry point,
    which hard-codes float32 accumulation and would floor the comparison at ~5e-8.

    :returns: ``_recurrence(q, k, v, g, beta, R, scale, initial_state, dtype)``.
    """
    import os
    from pathlib import Path

    env_dir = os.environ.get("KDA_PROBES_DIR")
    if not env_dir:
        raise SystemExit("KDA_PROBES_DIR is not set; refusing to guess the oracle location.")
    p = Path(env_dir)
    if not (p / "naive_kda_householder.py").is_file():
        raise SystemExit(f"No naive_kda_householder.py under {p}")
    if str(p) not in sys.path:
        sys.path.append(str(p))
    from naive_kda_householder import _recurrence  # type: ignore[import-not-found]

    return _recurrence


# ------------------------------------------------------------------------------------------
# §4.7 timing and peak memory at probe geometry
# ------------------------------------------------------------------------------------------


def _bench_one(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> Tuple[float, float, int, int]:
    """Time ``fn`` with warm kernels excluded and report peak memory.

    :param fn: nullary callable performing one measured unit of work.
    :param warmup: iterations run before the memory counters are reset and timing starts.
    :param iters: measured iterations.

    :returns: ``(median_ms, p90_ms, peak_allocated_bytes, peak_reserved_bytes)``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Reset AFTER warmup so the reported peak reflects steady state, and so autotuning
    # scratch from the first calls is not attributed to the measured region.
    torch.cuda.reset_peak_memory_stats()

    times: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    p90 = times[min(len(times) - 1, int(0.9 * len(times)))]
    return (
        statistics.median(times),
        p90,
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )


def timing_and_memory(
    device: torch.device, *, warmup: int, iters: int, seq_lens: List[int]
) -> List[Dict[str, Any]]:
    """Measure R1 and R2 forward / backward / fwd+bwd at the Phase-1 probe geometry.

    Probe geometry comes from ``probes/train_probe.py`` defaults: ``batch=64``, ``d_model=256``,
    ``n_heads=4``, ``head_dim=64``, evaluation lengths up to 512. Inputs are bf16; ``g`` is float32
    because both the kernel and the reference consume the gate in float32, which is the
    "fp32 recurrence accumulation" the runbook asks for.

    Throughput is reported as **logical real tokens per second**, ``B * T / seconds`` -- the R2 arm
    consumes ``T * R`` virtual positions but produces ``T`` real outputs, and quoting virtual
    positions would inflate R2 by exactly ``R``.

    :param device: CUDA device.
    :param warmup: warmup iterations excluded from timing.
    :param iters: measured iterations.
    :param seq_lens: sequence lengths to sweep.

    :returns: one record per (R, T, mode).
    """
    B, H, K, V = 64, 4, 64, 64
    records: List[Dict[str, Any]] = []
    for T in seq_lens:
        for R in (1, 2):
            records.extend(
                _bench_config(device, B=B, H=H, K=K, V=V, T=T, R=R, warmup=warmup, iters=iters)
            )
            torch.cuda.empty_cache()
    return records


def _bench_config(
    device: torch.device,
    *,
    B: int,
    H: int,
    K: int,
    V: int,
    T: int,
    R: int,
    warmup: int,
    iters: int,
) -> List[Dict[str, Any]]:
    """Benchmark one ``(T, R)`` configuration in its own scope so tensors free on return.

    :param device: CUDA device.
    :param B: batch size.
    :param H: head count.
    :param K: per-head key dimension.
    :param V: per-head value dimension.
    :param T: real sequence length (the kernel consumes ``T * R`` virtual positions).
    :param R: number of Householder factors.
    :param warmup: warmup iterations excluded from timing.
    :param iters: measured iterations.

    :returns: three records -- forward, backward, forward_backward.
    """
    kda = _triton_backend()
    gen = torch.Generator(device=device).manual_seed(1234 + R)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16)
    v = rnd(B, T * R, H, V).to(torch.bfloat16)
    beta = rnd(B, T * R, H).sigmoid().to(torch.bfloat16)
    g = F.logsigmoid(rnd(B, T, H, K))  # float32 gate == fp32 recurrence accumulation

    def fwd_only() -> torch.Tensor:
        with torch.no_grad():
            o, _ = kda(q=q, k=k, v=v, g=g, beta=beta, num_householder=R, backend="triton")
        return o

    grads = [t.detach().clone().requires_grad_(True) for t in (q, k, v, beta)]
    g_req = g.detach().clone().requires_grad_(True)

    def fwd_for_bwd() -> torch.Tensor:
        # backend="triton" is the fused training path Phase 1 would actually run.
        o, _ = kda(
            q=grads[0],
            k=grads[1],
            v=grads[2],
            g=g_req,
            beta=grads[3],
            num_householder=R,
            backend="triton",
        )
        return o

    def fwd_bwd() -> None:
        for t in grads + [g_req]:
            t.grad = None
        fwd_for_bwd().float().pow(2).mean().backward()

    torch.cuda.empty_cache()
    f_med, f_p90, f_alloc, f_res = _bench_one(fwd_only, warmup=warmup, iters=iters)

    torch.cuda.empty_cache()
    fb_med, fb_p90, fb_alloc, fb_res = _bench_one(fwd_bwd, warmup=warmup, iters=iters)

    # The backward cannot run standalone, so it is timed with the forward graph retained. The
    # reported peak memory for this mode therefore includes the retained forward activations.
    torch.cuda.empty_cache()
    loss = fwd_for_bwd().float().pow(2).mean()

    def bwd_only() -> None:
        for t in grads + [g_req]:
            t.grad = None
        loss.backward(retain_graph=True)

    b_med, b_p90, b_alloc, b_res = _bench_one(bwd_only, warmup=warmup, iters=iters)

    real_tokens = B * T
    out: List[Dict[str, Any]] = []
    for mode, (med, p90, alloc, res) in (
        ("forward", (f_med, f_p90, f_alloc, f_res)),
        ("backward", (b_med, b_p90, b_alloc, b_res)),
        ("forward_backward", (fb_med, fb_p90, fb_alloc, fb_res)),
    ):
        out.append(
            {
                "R": R,
                "T": T,
                "mode": mode,
                "median_ms": med,
                "p90_ms": p90,
                "peak_allocated_MiB": alloc / 2**20,
                "peak_reserved_MiB": res / 2**20,
                "real_tokens": real_tokens,
                "virtual_positions": B * T * R,
                # Logical REAL tokens/s. Never B*T*R/s -- that would inflate R2 by exactly R.
                "real_tokens_per_s": real_tokens / (med / 1e3),
            }
        )
    return out


def main() -> int:
    """Run both measurement blocks and print a JSON report.

    :returns: process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()

    report: Dict[str, Any] = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    report["decay_stress"] = decay_stress(torch.device("cpu"))
    if not args.skip_timing and torch.cuda.is_available():
        report["probe_geometry"] = {
            "batch": 64,
            "d_model": 256,
            "n_heads": 4,
            "head_dim": 64,
            "source": "probes/train_probe.py argparse defaults",
        }
        report["timing_memory"] = timing_and_memory(
            torch.device("cuda"), warmup=args.warmup, iters=args.iters, seq_lens=args.seq_lens
        )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
