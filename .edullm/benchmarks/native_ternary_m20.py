#!/usr/bin/env python
"""
Deferred A100/H100 benchmark for the exact Maple M20 expert geometry.

This harness is intentionally not a CI benchmark. It reports pack, packed forward, packed
input-gradient, ordinary grouped identity-STE weight-gradient, peak memory, and a three-
projection SwiGLU training step separately. Run it on both SM80 and SM90 before considering any
change to the opt-in status of ``native_packed``.
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict

# Benchmark the checkout that supplied this script, not the possibly older package baked into
# the capacity-block image. The launcher clones the repository to /work, but resolving from this
# file keeps the harness valid in worktrees and on developer machines as well.
_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src",
)
if os.path.isdir(_REPO_SRC):
    sys.path.insert(0, _REPO_SRC)

import torch
import torch.nn.functional as F

from olmo_core.nn.moe.mlp import MoEMLP
from olmo_core.nn.quantization import QuantBackend, QuantConfig
from olmo_core.ops.ternary import PackedTWNCache, native_packed_status

D_MODEL = 2048
EXPERT_HIDDEN = 512
NUM_EXPERTS = 256
TOP_K = 8


def _measure(fn: Callable[[], None], *, warmup: int, repeats: int) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _measure_full_step(
    *,
    quant: QuantConfig | None,
    capacity: int,
    tokens: int,
    warmup: int,
    repeats: int,
) -> Dict[str, object]:
    device = torch.device("cuda")
    mlp = MoEMLP(
        d_model=D_MODEL,
        hidden_size=EXPERT_HIDDEN,
        num_experts=NUM_EXPERTS,
        dtype=torch.bfloat16,
        init_device="cuda",
        quant=quant,
    )
    step_x = torch.randn(
        NUM_EXPERTS,
        capacity,
        D_MODEL,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-4)

    def full_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        step_x.grad = None
        loss = F.mse_loss(mlp(step_x).float(), torch.zeros((), device=device))
        loss.backward()
        # Mutation forces the native cache to repack on the next step.
        optimizer.step()

    torch.cuda.reset_peak_memory_stats()
    timings = _measure(full_step, warmup=warmup, repeats=repeats)
    metrics: Dict[str, object] = dict(timings)
    if quant is not None and quant.backend == QuantBackend.native_packed:
        pack_misses = sum(cache.misses for cache in mlp._native_pack_caches.values())
        if mlp._last_resolved_backend != QuantBackend.native_packed.value or pack_misses == 0:
            raise RuntimeError(
                "native_packed benchmark did not execute packed kernels; refusing to report "
                "fallback timings"
            )
        metrics["native_pack_misses"] = pack_misses
    metrics["resolved_backend"] = mlp._last_resolved_backend
    metrics["tokens_per_second"] = tokens / (timings["median_ms"] / 1000)
    metrics["peak_memory_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    # The capacity-block launcher always uses torchrun. Select the assigned card explicitly;
    # otherwise every local rank defaults to cuda:0 and eight independent harnesses OOM it.
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    status = native_packed_status()
    if not status["available"]:
        raise SystemExit(f"native packed backend unavailable: {status['reason']}")
    from olmo_core.kernels import ternary as kernels

    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        raise SystemExit(f"SM80 or newer is required, got compute capability {capability}")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    # Capacity path has M_e = ceil(tokens * top_k / experts), matching the exact M20 routing
    # geometry while keeping this kernel harness independent of a router distribution.
    capacity = (args.tokens * TOP_K + NUM_EXPERTS - 1) // NUM_EXPERTS
    x = torch.randn(NUM_EXPERTS, capacity, D_MODEL, device=device, dtype=dtype)
    weight = torch.randn(
        NUM_EXPERTS, D_MODEL, EXPERT_HIDDEN, device=device, dtype=dtype, requires_grad=True
    )
    grad_output = torch.randn(NUM_EXPERTS, capacity, EXPERT_HIDDEN, device=device, dtype=dtype)
    cache = PackedTWNCache()

    def pack_once() -> None:
        cache.clear()
        cache.get_or_pack(weight, in_dim=1, orientation="m20_capacity_w1")

    pack_metrics = _measure(pack_once, warmup=args.warmup, repeats=args.repeats)
    packed = cache.get_or_pack(weight, in_dim=1, orientation="m20_capacity_w1")

    def forward_once() -> None:
        kernels.fixed_grouped_packed_matmul(x, packed.codes, packed.alpha, D_MODEL)

    forward_metrics = _measure(forward_once, warmup=args.warmup, repeats=args.repeats)

    def grad_input_once() -> None:
        kernels.fixed_grouped_packed_matmul_transpose(
            grad_output, packed.codes_t, packed.alpha, D_MODEL
        )

    grad_input_metrics = _measure(grad_input_once, warmup=args.warmup, repeats=args.repeats)

    def grad_weight_once() -> None:
        # Identity STE necessarily retains this ordinary grouped BF16 GEMM.
        torch.bmm(grad_output.transpose(1, 2), x)

    grad_weight_metrics = _measure(grad_weight_once, warmup=args.warmup, repeats=args.repeats)

    gc.collect()
    torch.cuda.empty_cache()

    native_quant = QuantConfig(
        enabled=True,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
    )
    fake_quant = QuantConfig(enabled=True, backend=QuantBackend.fake_quant)
    full_steps = {
        "native_packed": _measure_full_step(
            quant=native_quant,
            capacity=capacity,
            tokens=args.tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "fake_quant": _measure_full_step(
            quant=fake_quant,
            capacity=capacity,
            tokens=args.tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        "ordinary_bf16": _measure_full_step(
            quant=None,
            capacity=capacity,
            tokens=args.tokens,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
    }

    print(
        json.dumps(
            {
                "status": "deferred_hardware_measurement",
                "gpu": torch.cuda.get_device_name(),
                "compute_capability": capability,
                "torch": torch.__version__,
                "geometry": {
                    "d_model": D_MODEL,
                    "expert_hidden": EXPERT_HIDDEN,
                    "num_experts": NUM_EXPERTS,
                    "top_k": TOP_K,
                    "tokens": args.tokens,
                    "capacity_per_expert": capacity,
                },
                "pack": pack_metrics,
                "forward": forward_metrics,
                "grad_input": grad_input_metrics,
                "grad_weight_grouped_bf16": grad_weight_metrics,
                "full_steps": full_steps,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
