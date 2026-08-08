"""Strict subprocess benchmark for native Mamba-3 SISO PD chunk sizes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Optional

CHUNK_CANDIDATES = (32, 64, 128)
ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/olmo_core/nn/flash_pd_native"


def production_source_hashes() -> dict[str, str]:
    """Hash every source file that defines the measured native path."""
    paths = {
        "api.py": PACKAGE / "api.py",
        "cuda.py": PACKAGE / "cuda.py",
        "mamba3_siso.py": PACKAGE / "mamba3_siso.py",
        "flash_pd_native.cpp": PACKAGE / "csrc/flash_pd_native.cpp",
        "flash_pd_native_cuda.cu": PACKAGE / "csrc/flash_pd_native_cuda.cu",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def select_a100_winner(
    gpu_name: str,
    measurements: list[dict[str, Any]],
) -> Optional[int]:
    """Select the measured median winner only when the device is an A100."""
    if "A100" not in gpu_name.upper() or not measurements:
        return None
    return int(min(measurements, key=lambda result: result["median_ms"])["chunk_size"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, choices=CHUNK_CANDIDATES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--dictionary-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    return parser


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def _worker(opts: argparse.Namespace) -> dict[str, Any]:
    import torch

    from olmo_core.config import DType
    from olmo_core.nn.flash_pd_native import (
        NativeFlashPDMamba3SISOMixerConfig,
        NativePDMode,
        get_backend_counters,
        native_cuda_capability,
        reset_backend_counters,
    )
    from olmo_core.nn.transformer import InitMethod

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; the benchmark never records a fallback")
    capability = native_cuda_capability()
    if not capability.available:
        raise RuntimeError(capability.reason)
    if opts.d_model != opts.n_heads * opts.d_state:
        raise ValueError("d-model must equal n-heads * d-state")
    torch._dynamo.reset()
    torch.manual_seed(6198)
    torch.cuda.manual_seed_all(6198)
    device = torch.device("cuda")
    config = NativeFlashPDMamba3SISOMixerConfig(
        n_heads=opts.n_heads,
        d_state=opts.d_state,
        dictionary_size=opts.dictionary_size,
        chunk_size=opts.chunk_size,
        backend="cuda",
        mode=NativePDMode.GENERAL_SCATTER,
        dtype=DType.bfloat16,
    )
    mixer = config.build(
        opts.d_model,
        layer_idx=0,
        n_layers=1,
        init_device="cuda",
    )
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=opts.d_model,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator(device=device).manual_seed(6198),
    )
    # Learned colliding maps are the production route. Force every source onto
    # destination zero so a permutation specialization cannot be measured by accident.
    with torch.no_grad():
        mixer.dictionary_logits.fill_(-1)
        mixer.dictionary_logits[:, :, 0, :].fill_(1)
    x = torch.randn(
        opts.batch_size,
        opts.sequence_length,
        opts.d_model,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    measured_mixer = mixer
    if opts.compile:
        measured_mixer = torch.compile(mixer, fullgraph=True)

    def iteration() -> None:
        mixer.zero_grad(set_to_none=True)
        x.grad = None
        output = measured_mixer(x)
        output.float().square().mean().backward()

    reset_backend_counters()
    for _ in range(opts.warmup):
        iteration()
    torch.cuda.synchronize()
    elapsed_ms = []
    for _ in range(opts.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        iteration()
        end.record()
        end.synchronize()
        elapsed_ms.append(float(start.elapsed_time(end)))

    counters = get_backend_counters()
    expected_dispatches = opts.warmup + opts.iterations
    realized = counters.get("cuda_mamba3_siso_general_scatter", 0)
    if realized != expected_dispatches:
        raise RuntimeError(
            "native general-scatter dispatch was not realized exactly: "
            f"expected {expected_dispatches}, counters={counters}"
        )
    accounting = mixer.accounting(
        batch_size=opts.batch_size,
        sequence_length=opts.sequence_length,
        element_size=2,
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in mixer.parameters()
    )
    working_set_bytes = (
        parameter_bytes + accounting.saved_tensor_bytes + accounting.peak_workspace_bytes
    )
    properties = torch.cuda.get_device_properties(device)
    l2_bytes = int(getattr(properties, "l2_cache_size", 0))
    tokens = opts.batch_size * opts.sequence_length
    median_ms = statistics.median(elapsed_ms)
    return {
        "chunk_size": opts.chunk_size,
        "gpu_name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": "bfloat16",
        "mode": "general_scatter",
        "compiled": opts.compile,
        "warmup": opts.warmup,
        "iterations": opts.iterations,
        "median_ms": median_ms,
        "p95_ms": _percentile(elapsed_ms, 0.95),
        "tokens_per_second": tokens / (median_ms / 1000.0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "working_set_bytes": working_set_bytes,
        "l2_cache_bytes": l2_bytes,
        "working_set_over_l2": (working_set_bytes / l2_bytes if l2_bytes else None),
        "model_flops_per_token": accounting.flops_per_token,
        "model_flops_per_sequence": accounting.model_flops_per_sequence,
        "nonlinear_evaluations_per_sequence": (accounting.nonlinear_evaluations_per_sequence),
        "route_comparisons_per_sequence": (accounting.route_comparisons_per_sequence),
        "backend_counters": counters,
        "source_hashes": production_source_hashes(),
    }


def _parent(opts: argparse.Namespace) -> dict[str, Any]:
    measurements = []
    for chunk_size in CHUNK_CANDIDATES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--chunk-size",
            str(chunk_size),
            "--batch-size",
            str(opts.batch_size),
            "--sequence-length",
            str(opts.sequence_length),
            "--d-model",
            str(opts.d_model),
            "--n-heads",
            str(opts.n_heads),
            "--d-state",
            str(opts.d_state),
            "--dictionary-size",
            str(opts.dictionary_size),
            "--warmup",
            str(opts.warmup),
            "--iterations",
            str(opts.iterations),
        ]
        if opts.compile:
            command.append("--compile")
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            check=True,
            text=True,
            capture_output=True,
        )
        measurements.append(json.loads(completed.stdout.splitlines()[-1]))
    gpu_names = {measurement["gpu_name"] for measurement in measurements}
    if len(gpu_names) != 1:
        raise RuntimeError(f"subprocesses used different GPUs: {sorted(gpu_names)}")
    gpu_name = next(iter(gpu_names))
    return {
        "gpu_name": gpu_name,
        "measurements": measurements,
        "a100_winner": select_a100_winner(gpu_name, measurements),
        "source_hashes": production_source_hashes(),
    }


def main() -> None:
    opts = build_parser().parse_args()
    if opts.warmup < 20 or opts.iterations < 50:
        raise SystemExit("final benchmark requires at least 20 warmups and 50 measurements")
    if opts.worker and opts.chunk_size is None:
        raise SystemExit("--worker requires --chunk-size")
    result = _worker(opts) if opts.worker else _parent(opts)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
