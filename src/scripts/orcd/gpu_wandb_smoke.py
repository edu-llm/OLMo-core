"""Run a bounded CUDA access probe and publish its metrics to Weights & Biases."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MEBIBYTE = 1024 * 1024


@dataclass(frozen=True)
class MatmulMeasurement:
    """One timed CUDA matrix multiplication."""

    iteration: int
    milliseconds: float
    tflops: float
    result_mean: float


def calculate_tflops(matrix_size: int, elapsed_seconds: float) -> float:
    """
    Calculate throughput for one square matrix multiplication.

    :param matrix_size: Width and height of both input matrices.
    :param elapsed_seconds: Wall-clock duration of the operation.

    :returns: Approximate floating-point operations per second in trillions.
    """
    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    return (2 * matrix_size**3) / elapsed_seconds / 1e12


def run_cuda_matmuls(matrix_size: int, iterations: int) -> list[MatmulMeasurement]:
    """
    Run a small sequence of CUDA matrix multiplications.

    :param matrix_size: Width and height of each input matrix.
    :param iterations: Number of timed operations.

    :returns: Per-iteration timing, throughput, and result measurements.

    :raises RuntimeError: If CUDA is unavailable or a result is not finite.
    """
    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the allocated job")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    left = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    right = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)

    # Compile kernels and establish memory allocations outside the timed samples.
    result = left @ right
    torch.cuda.synchronize()

    measurements: list[MatmulMeasurement] = []
    for iteration in range(1, iterations + 1):
        started = time.perf_counter()
        result = left @ right
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        result_mean = float(result.mean().item())
        if not math.isfinite(result_mean):
            raise RuntimeError("CUDA matrix multiplication produced a non-finite result")
        measurements.append(
            MatmulMeasurement(
                iteration=iteration,
                milliseconds=elapsed_seconds * 1000,
                tflops=calculate_tflops(matrix_size, elapsed_seconds),
                result_mean=result_mean,
            )
        )
    return measurements


def build_report(matrix_size: int, measurements: list[MatmulMeasurement]) -> dict[str, Any]:
    """
    Build the public, secret-free smoke report.

    :param matrix_size: Width and height of the tested matrices.
    :param measurements: Successful CUDA timing samples.

    :returns: Device details and aggregate smoke metrics.
    """
    if not measurements:
        raise ValueError("at least one measurement is required")
    properties = torch.cuda.get_device_properties(0)
    milliseconds = [measurement.milliseconds for measurement in measurements]
    tflops = [measurement.tflops for measurement in measurements]
    return {
        "passed": True,
        "device": {
            "name": properties.name,
            "count": torch.cuda.device_count(),
            "total_memory_mib": properties.total_memory / MEBIBYTE,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "matrix_size": matrix_size,
        "iterations": len(measurements),
        "metrics": {
            "matmul_ms_mean": statistics.fmean(milliseconds),
            "matmul_ms_min": min(milliseconds),
            "tflops_mean": statistics.fmean(tflops),
            "tflops_max": max(tflops),
            "gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / MEBIBYTE,
            "gpu_memory_reserved_mib": torch.cuda.max_memory_reserved() / MEBIBYTE,
        },
        "samples": [asdict(measurement) for measurement in measurements],
    }


def log_report(run: Any, report: dict[str, Any]) -> None:
    """
    Log the probe report to an initialized W&B run.

    :param run: An initialized W&B run.
    :param report: The report produced by :func:`build_report`.
    """
    for sample in report["samples"]:
        run.log(
            {
                "smoke/iteration": sample["iteration"],
                "smoke/matmul_ms": sample["milliseconds"],
                "smoke/tflops": sample["tflops"],
                "smoke/result_mean": sample["result_mean"],
                "smoke/gpu_memory_allocated_mib": report["metrics"]["gpu_memory_allocated_mib"],
                "smoke/gpu_memory_reserved_mib": report["metrics"]["gpu_memory_reserved_mib"],
                "smoke/gpu_available": 1,
                "smoke/passed": 1,
            },
            step=sample["iteration"],
        )
    run.summary.update(
        {
            "smoke/status": "passed",
            "smoke/device_name": report["device"]["name"],
            "smoke/matmul_ms_mean": report["metrics"]["matmul_ms_mean"],
            "smoke/tflops_mean": report["metrics"]["tflops_mean"],
            "smoke/passed": 1,
        }
    )


def main() -> None:
    """Run the CUDA probe, write a local report, and publish W&B metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    import wandb

    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        group=os.environ.get("WANDB_GROUP", "orcd-gpu-access"),
        name=args.run_name,
        id=os.environ.get("WANDB_RUN_ID", args.run_name),
        job_type="gpu-access-smoke",
        tags=["orcd", "gpu-access", "l40s", "smoke"],
        mode="online",
        resume="never",
        config={
            "matrix_size": args.matrix_size,
            "iterations": args.iterations,
            "git_commit": os.environ.get("EDULLM_COMMIT_SHA"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    )
    exit_code = 1
    try:
        if not torch.cuda.is_available():
            run.log({"smoke/gpu_available": 0, "smoke/passed": 0}, step=0)
            run.summary.update({"smoke/status": "failed", "smoke/passed": 0})
            raise RuntimeError("CUDA is not available in the allocated job")

        measurements = run_cuda_matmuls(args.matrix_size, args.iterations)
        report = build_report(args.matrix_size, measurements)
        log_report(run, report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"W&B run: {run.url}")
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
