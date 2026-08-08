"""Benchmark the Lngram counterfactual backward on one CUDA device.

This compares the exact PyTorch reference with the optional Triton grad-z path.
It does not train or read a corpus.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
from collections.abc import Callable, Iterator, Sequence

import torch

import olmo_core.nn.memory.counterfactual as counterfactual_module
from olmo_core.nn.memory.counterfactual import counterfactual_lookup
from olmo_core.ops.lngram import has_lngram_triton


@contextlib.contextmanager
def _backend(*, triton_enabled: bool) -> Iterator[None]:
    original = counterfactual_module.try_counterfactual_grad_z
    if triton_enabled:

        def require_triton(*args, **kwargs):
            result = original(*args, **kwargs)
            if result is None:
                raise RuntimeError("benchmark inputs did not dispatch to Triton")
            return result

        counterfactual_module.try_counterfactual_grad_z = require_triton
    else:
        counterfactual_module.try_counterfactual_grad_z = lambda *args, **kwargs: None
    try:
        yield
    finally:
        counterfactual_module.try_counterfactual_grad_z = original


def _run_once(
    lookup: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, ...],
    ],
    z: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    upstreams: tuple[torch.Tensor, torch.Tensor],
) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iteration_z = z.detach().requires_grad_()
    iteration_tables = tuple(table.detach().requires_grad_() for table in tables)
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    outputs = lookup(iteration_z, iteration_tables[0], iteration_tables[1])
    start.record()
    torch.autograd.backward(outputs, upstreams)
    end.record()
    end.synchronize()

    elapsed_ms = start.elapsed_time(end)
    peak_gib = (torch.cuda.max_memory_allocated() - baseline_bytes) / 1024**3
    return elapsed_ms, peak_gib


def _benchmark(
    z: torch.Tensor,
    tables: tuple[torch.Tensor, torch.Tensor],
    upstreams: tuple[torch.Tensor, torch.Tensor],
    *,
    triton_enabled: bool,
    compiled: bool,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    def lookup(
        input_z: torch.Tensor,
        table_order_2: torch.Tensor,
        table_order_3: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return counterfactual_lookup(
            input_z,
            (table_order_2, table_order_3),
            (2, 3),
            bits_per_route=4,
            require_triton=triton_enabled,
        )

    with _backend(triton_enabled=triton_enabled):
        benchmark_lookup = torch.compile(lookup, fullgraph=False) if compiled else lookup
        for _ in range(warmup):
            _run_once(benchmark_lookup, z, tables, upstreams)
        measurements = [
            _run_once(benchmark_lookup, z, tables, upstreams) for _ in range(iterations)
        ]
    return (
        statistics.median(elapsed for elapsed, _ in measurements),
        max(peak for _, peak in measurements),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--memory-dim", type=int, default=61)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--eager",
        action="store_true",
        help="disable torch.compile; compiled mode matches the training path",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    opts = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not has_lngram_triton():
        raise SystemExit("Triton is not available")
    if opts.d_model % 4:
        raise SystemExit("--d-model must be divisible by four")
    if not 0 < opts.memory_dim <= 64:
        raise SystemExit("--memory-dim must be between 1 and 64 for Triton")

    dtype = getattr(torch, opts.dtype)
    device = torch.device("cuda")
    num_routes = opts.d_model // 4
    z = torch.randn(
        opts.batch_size,
        opts.sequence_length,
        opts.d_model,
        device=device,
        dtype=dtype,
    )
    tables = (
        torch.randn(
            num_routes * 16**2,
            opts.memory_dim,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            num_routes * 16**3,
            opts.memory_dim,
            device=device,
            dtype=dtype,
        ),
    )
    upstreams = (
        torch.randn(
            opts.batch_size,
            opts.sequence_length,
            num_routes * opts.memory_dim,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            opts.batch_size,
            opts.sequence_length,
            num_routes * opts.memory_dim,
            device=device,
            dtype=dtype,
        ),
    )

    reference_ms, reference_peak = _benchmark(
        z,
        tables,
        upstreams,
        triton_enabled=False,
        compiled=not opts.eager,
        warmup=opts.warmup,
        iterations=opts.iterations,
    )
    triton_ms, triton_peak = _benchmark(
        z,
        tables,
        upstreams,
        triton_enabled=True,
        compiled=not opts.eager,
        warmup=opts.warmup,
        iterations=opts.iterations,
    )
    mode = "eager" if opts.eager else "compiled"
    print(
        f"mode:      {mode}\n"
        f"reference: {reference_ms:.2f} ms, incremental peak {reference_peak:.2f} GiB\n"
        f"triton:    {triton_ms:.2f} ms, incremental peak {triton_peak:.2f} GiB\n"
        f"speedup:   {reference_ms / triton_ms:.2f}x"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
