"""
Microbenchmark the b=3 SO(3) prefix product at production shapes.

The b=3 arm spends ~75% of its step in this one scan, so the question that decides whether the
``associative_scan`` rewrite is worth finishing is narrow: **is the tree scan actually faster than
the chunked sequential-product form, forward and forward+backward?** That was never measured -- the
training run died at step 1 on a NaN gradient, and throughput is not reported until step 10.

This isolates the scan from the rest of the model and answers it in ~2 minutes. It also reports
whether each variant's gradient is finite, which localises the NaN to a specific combine_mode.

Run on an idle GPU::

    CUDA_VISIBLE_DEVICES=1 python src/scripts/b200/scan_bench.py
"""

import time
from typing import Callable, Optional

import torch

# Import the symbols directly: `olmo_core.nn.mamba3` re-exports a *function* named
# `mamba3_ssd_fast`, which shadows the submodule of the same name.
from olmo_core.nn.mamba3.mamba3_ssd_api import _cumulative_block_rotation
from olmo_core.nn.mamba3.mamba3_ssd_fast import (
    _so3_pointwise_combine,
    associative_autograd_cumulative_block_rotation,
    fast_block_rotations,
)

# batch, seq_len, n_groups, n_blocks (= d_state // 3), angles per so(3) block.
# Mirrors the real 32-seq microbatch at d_state=192, seq 4096.
SHAPE = (32, 4096, 1, 64, 3)
REPEATS = 5
WARMUP = 2


def _assoc(rot: torch.Tensor, combine_mode: str) -> torch.Tensor:
    """The associative_scan prefix product, with combine_mode forced."""
    from torch._higher_order_ops.associative_scan import associative_scan

    leaves = tuple(rot[..., i, j].contiguous() for i in range(3) for j in range(3))
    scanned = associative_scan(
        _so3_pointwise_combine, leaves, dim=1, combine_mode=combine_mode
    )
    return torch.stack(tuple(scanned), dim=-1).unflatten(-1, (3, 3))


def _time(fn: Callable[[], None]) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(REPEATS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / REPEATS * 1e3


def _bench(
    label: str,
    scan: Callable[[torch.Tensor], torch.Tensor],
    *,
    backward: bool,
    compile_it: bool,
    baseline_ms: Optional[float] = None,
) -> Optional[float]:
    torch.manual_seed(0)
    theta = (torch.randn(*SHAPE, device="cuda", dtype=torch.float32) * 0.1).requires_grad_(backward)

    def once() -> None:
        out = scan(fast_block_rotations(theta, 3))
        if backward:
            theta.grad = None
            out.sum().backward()

    fn = torch.compile(once) if compile_it else once
    try:
        ms = _time(fn)
    except Exception as exc:  # a variant failing must not abort the sweep
        print(f"  {label:38s}   FAILED  {type(exc).__name__}: {str(exc)[:70]}")
        return None

    note = ""
    if backward:
        grad = theta.grad
        if grad is None or not torch.isfinite(grad).all():
            note = "   <-- NON-FINITE GRAD"
    speedup = f"  ({baseline_ms / ms:4.2f}x)" if baseline_ms else ""
    print(f"  {label:38s} {ms:8.1f} ms{speedup}{note}")
    return ms


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    print(f"device: {torch.cuda.get_device_name()}   torch {torch.__version__}")
    print(f"shape:  {SHAPE}  (batch, T, groups, blocks, angles)  fp32\n")

    chunked = lambda rot: _cumulative_block_rotation(rot, chunk_size=32)  # noqa: E731
    variants = [
        ("chunked (chunk=32, today)", chunked),
        ("associative pointwise", lambda r: _assoc(r, "pointwise")),
        ("associative generic", lambda r: _assoc(r, "generic")),
        # Same forward as "associative pointwise", but with `associative_scan`'s own autograd
        # replaced by an analytic backward that is itself one scan. This is the only row expected
        # to report a finite gradient *and* a tree-scan forward, so it is the one to decide on.
        ("associative + analytic bwd", associative_autograd_cumulative_block_rotation),
    ]

    for compile_it in (False, True):
        for backward in (False, True):
            head = f"{'compiled' if compile_it else 'eager'} / {'fwd+bwd' if backward else 'fwd'}"
            print(f"{head}:")
            base = None
            for label, scan in variants:
                ms = _bench(label, scan, backward=backward, compile_it=compile_it, baseline_ms=base)
                if base is None:
                    base = ms
            print()

    print(
        "Decide on 'compiled / fwd+bwd'. The chunked row is what the 33,468 tok/s run uses; the scan\n"
        "is ~75% of the b=3 step, so a 2x here is roughly 1.6x end-to-end. Under ~1.3x is not worth\n"
        "spending the remaining window on."
    )


if __name__ == "__main__":
    main()
