#!/usr/bin/env python3
"""Measure HPO-style throughput with one isolated training process per GPU.

The default launcher starts eight independent ``WORLD_SIZE=1`` workers. Each worker sees
exactly one physical GPU through ``CUDA_VISIBLE_DEVICES`` and trains its own model, matching
the ordinary HPO probe topology. Synthetic local data avoids dataset and network overhead.

Example::

    python .edullm/runpod/throughput_smoke.py \
      --profile olmoe-hpo --gpu-count 8 --warmup-steps 2 --bench-steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ThroughputProfile:
    """Model and per-trial batch geometry to benchmark."""

    name: str
    model_factory: str
    sequence_length: int
    global_batch_size: int
    rank_microbatch_size: int


@dataclass(frozen=True)
class BenchmarkVariant:
    """Optimizer and memory/performance knobs for one independent worker."""

    name: str
    optimizer: str
    rank_microbatch_size: int
    activation_checkpointing: bool = False
    compile_model: bool = False
    fused_loss: bool = False


@dataclass(frozen=True)
class WorkerProcessSpec:
    """Command and isolated distributed environment for one trial worker."""

    gpu_id: int
    argv: tuple[str, ...]
    env: dict[str, str]
    log_path: Path


PROFILES: dict[str, ThroughputProfile] = {
    "190m-probe": ThroughputProfile(
        name="190m-probe",
        model_factory="olmo2_190M",
        sequence_length=2048,
        global_batch_size=32768,
        rank_microbatch_size=4096,
    ),
    "olmoe-hpo": ThroughputProfile(
        name="olmoe-hpo",
        model_factory="olmoe_1B_7B",
        sequence_length=2048,
        global_batch_size=32768,
        rank_microbatch_size=4096,
    ),
}

VARIANTS: dict[str, BenchmarkVariant] = {
    "adam-mb2048": BenchmarkVariant("adam-mb2048", "adam", 2048),
    "adam-mb4096": BenchmarkVariant("adam-mb4096", "adam", 4096),
    "adam8-mb2048": BenchmarkVariant("adam8-mb2048", "adam8", 2048),
    "adam8-mb4096": BenchmarkVariant("adam8-mb4096", "adam8", 4096),
    "adam8-mb8192": BenchmarkVariant("adam8-mb8192", "adam8", 8192),
    "adam8-mb16384": BenchmarkVariant("adam8-mb16384", "adam8", 16384),
    "adam8-ac-mb2048": BenchmarkVariant(
        "adam8-ac-mb2048", "adam8", 2048, activation_checkpointing=True
    ),
    "adam8-ac-mb4096": BenchmarkVariant(
        "adam8-ac-mb4096", "adam8", 4096, activation_checkpointing=True
    ),
    "adam8-ac-mb8192": BenchmarkVariant(
        "adam8-ac-mb8192", "adam8", 8192, activation_checkpointing=True
    ),
    "adam8-ac-mb16384": BenchmarkVariant(
        "adam8-ac-mb16384", "adam8", 16384, activation_checkpointing=True
    ),
    "adam8-compile-mb4096": BenchmarkVariant(
        "adam8-compile-mb4096", "adam8", 4096, compile_model=True
    ),
    "adam8-compile-mb8192": BenchmarkVariant(
        "adam8-compile-mb8192", "adam8", 8192, compile_model=True
    ),
    "adam8-fused-mb2048": BenchmarkVariant(
        "adam8-fused-mb2048", "adam8", 2048, fused_loss=True
    ),
    "adam8-fused-mb4096": BenchmarkVariant(
        "adam8-fused-mb4096", "adam8", 4096, fused_loss=True
    ),
    "adam8-fused-ac-mb2048": BenchmarkVariant(
        "adam8-fused-ac-mb2048",
        "adam8",
        2048,
        activation_checkpointing=True,
        fused_loss=True,
    ),
    "adam8-fused-ac-mb4096": BenchmarkVariant(
        "adam8-fused-ac-mb4096",
        "adam8",
        4096,
        activation_checkpointing=True,
        fused_loss=True,
    ),
    "adam8-fused-compile-mb2048": BenchmarkVariant(
        "adam8-fused-compile-mb2048",
        "adam8",
        2048,
        compile_model=True,
        fused_loss=True,
    ),
    "adam8-fused-compile-mb4096": BenchmarkVariant(
        "adam8-fused-compile-mb4096",
        "adam8",
        4096,
        compile_model=True,
        fused_loss=True,
    ),
    "adam-fused-ac-mb2048": BenchmarkVariant(
        "adam-fused-ac-mb2048",
        "adam",
        2048,
        activation_checkpointing=True,
        fused_loss=True,
    ),
    "adam-fused-ac-mb4096": BenchmarkVariant(
        "adam-fused-ac-mb4096",
        "adam",
        4096,
        activation_checkpointing=True,
        fused_loss=True,
    ),
}


def worker_process_specs(
    *,
    script: Path,
    profile: str,
    gpu_count: int,
    warmup_steps: int,
    bench_steps: int,
    work_dir: Path,
    variants: list[str] | None = None,
) -> list[WorkerProcessSpec]:
    """Build one non-``torchrun``, world-size-one worker command per GPU."""

    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    if variants is None:
        variants = ["adam-mb4096"] * gpu_count
    if len(variants) != gpu_count:
        raise ValueError("variants must contain exactly one entry per GPU")
    unknown_variants = set(variants) - set(VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")

    specs: list[WorkerProcessSpec] = []
    for gpu_id in range(gpu_count):
        worker_dir = work_dir / f"worker-{gpu_id}"
        argv = (
            sys.executable,
            str(script),
            "--worker",
            "--gpu-id",
            str(gpu_id),
            "--profile",
            profile,
            "--variant",
            variants[gpu_id],
            "--warmup-steps",
            str(warmup_steps),
            "--bench-steps",
            str(bench_steps),
            "--work-dir",
            str(worker_dir),
        )
        env = {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(29600 + gpu_id),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "NUM_NODES": "1",
        }
        specs.append(
            WorkerProcessSpec(
                gpu_id=gpu_id,
                argv=argv,
                env=env,
                log_path=work_dir / "logs" / f"worker-{gpu_id}.log",
            )
        )
    return specs


def _write_synthetic_corpus(path: Path, num_tokens: int) -> None:
    import numpy as np

    from olmo_core.data.utils import write_document_indices

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint32
    eos_token_id = 0
    tokens = np.random.default_rng(42).integers(1, 32000, size=num_tokens, dtype=dtype)
    tokens[-1] = eos_token_id
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(num_tokens,))
    mmap[:] = tokens
    mmap.flush()
    write_document_indices(path, dtype=dtype, eos_token_id=eos_token_id)


def _build_trainer(
    profile: ThroughputProfile,
    variant: BenchmarkVariant,
    *,
    work_dir: Path,
    total_steps: int,
    warmup_steps: int,
    bench_steps: int,
) -> tuple[Any, Any]:
    import time

    import torch

    from olmo_core.config import DType
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import (
        TransformerConfig,
        TransformerActivationCheckpointingMode,
        TransformerDataParallelWrappingStrategy,
    )
    from olmo_core.optim import AdamWConfig, CosWithWarmup
    from olmo_core.optim.config import OptimConfig
    from olmo_core.train import Duration, TrainerConfig
    from olmo_core.train.callbacks import Callback, CheckpointerCallback
    from olmo_core.train.train_module import (
        TransformerActivationCheckpointingConfig,
        TransformerDataParallelConfig,
        TransformerTrainModuleConfig,
    )

    class BenchmarkTimerCallback(Callback):
        """Time only the post-warmup optimizer steps, including batch loading."""

        def __init__(self) -> None:
            self.started_at: float | None = None
            self.elapsed: float | None = None

        def pre_train(self) -> None:
            if warmup_steps == 0:
                torch.cuda.synchronize()
                self.started_at = time.perf_counter()

        def post_step(self) -> None:
            if self.trainer.global_step == warmup_steps:
                torch.cuda.synchronize()
                self.started_at = time.perf_counter()
            elif self.trainer.global_step == warmup_steps + bench_steps:
                torch.cuda.synchronize()
                if self.started_at is None:
                    raise RuntimeError("benchmark timer never started")
                self.elapsed = time.perf_counter() - self.started_at

    tokenizer = TokenizerConfig.dolma2()
    factory = getattr(TransformerConfig, profile.model_factory)
    model_config = factory(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=AttentionBackendName.flash_3,
    )
    if variant.fused_loss:
        model_config.lm_head.loss_implementation = LMLossImplementation.fused_linear

    data_path = work_dir / "synthetic.npy"
    min_tokens = profile.global_batch_size * (total_steps + 2)
    _write_synthetic_corpus(data_path, min_tokens)
    dataset = NumpyFSLDatasetConfig(
        paths=[str(data_path)],
        sequence_length=profile.sequence_length,
        tokenizer=tokenizer,
        work_dir=str(work_dir),
    ).build()
    data_loader = NumpyDataLoaderConfig(
        global_batch_size=profile.global_batch_size,
        seed=12345,
        num_workers=2,
    )
    if variant.optimizer == "adam":
        optimizer = AdamWConfig(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    elif variant.optimizer == "adam8":
        from torchao.optim import AdamW8bit

        @dataclass
        class AdamW8bitConfig(OptimConfig):
            lr: float = 3e-4
            betas: tuple[float, float] = (0.9, 0.95)
            eps: float = 1e-8
            weight_decay: float = 0.1

            @classmethod
            def optimizer(cls):
                return AdamW8bit

        optimizer = AdamW8bitConfig()
    else:
        raise ValueError(f"unsupported optimizer {variant.optimizer!r}")

    ac_config = None
    if variant.activation_checkpointing:
        ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_ops
        )

    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=variant.rank_microbatch_size,
        max_sequence_length=profile.sequence_length,
        optim=optimizer,
        compile_model=variant.compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        ac_config=ac_config,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=max(1, min(5, total_steps // 4))),
    )
    timer = BenchmarkTimerCallback()
    trainer_config = (
        TrainerConfig(
            save_folder=str(work_dir / "checkpoints"),
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(total_steps),
        )
        .with_callback("checkpointer", CheckpointerCallback(enabled=False))
        .with_callback("benchmark_timer", timer)
    )

    model = model_config.build(init_device="meta")
    module = train_module.build(model)
    loader = data_loader.build(dataset, dp_process_group=module.dp_process_group)
    return trainer_config.build(module, loader), timer


def _run_worker(args: argparse.Namespace) -> int:
    import torch

    from olmo_core.train import prepare_training_environment, teardown_training_environment

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"worker {args.gpu_id} must see exactly one GPU, saw {torch.cuda.device_count()}"
        )
    profile = PROFILES[args.profile]
    variant = VARIANTS[args.variant]
    total_steps = args.warmup_steps + args.bench_steps
    work_dir = Path(args.work_dir)

    prepare_training_environment()
    trainer, timer = _build_trainer(
        profile,
        variant,
        work_dir=work_dir,
        total_steps=total_steps,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
    )
    try:
        trainer.fit()
        if timer.elapsed is None or timer.elapsed <= 0:
            raise RuntimeError("benchmark timer produced no duration")
        result = {
            "gpu_id": args.gpu_id,
            "profile": profile.name,
            "variant": variant.name,
            "model_factory": profile.model_factory,
            "sequence_length": profile.sequence_length,
            "global_batch_size": profile.global_batch_size,
            "rank_microbatch_size": variant.rank_microbatch_size,
            "optimizer": variant.optimizer,
            "activation_checkpointing": variant.activation_checkpointing,
            "compile_model": variant.compile_model,
            "fused_loss": variant.fused_loss,
            "world_size": torch.distributed.get_world_size(),
            "visible_gpu_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0),
            "warmup_steps": args.warmup_steps,
            "bench_steps": args.bench_steps,
            "elapsed_seconds": timer.elapsed,
            "tokens_per_second": profile.global_batch_size * args.bench_steps / timer.elapsed,
        }
        print("THROUGHPUT_SMOKE_RESULT", json.dumps(result, sort_keys=True), flush=True)
    finally:
        teardown_training_environment()
    return 0


def _parse_result(log_path: Path) -> dict[str, Any] | None:
    marker = "THROUGHPUT_SMOKE_RESULT "
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith(marker):
            return json.loads(line.removeprefix(marker))
    return None


def _run_launcher(args: argparse.Namespace) -> int:
    import torch

    if torch.cuda.device_count() < args.gpu_count:
        raise RuntimeError(
            f"requested {args.gpu_count} workers but only {torch.cuda.device_count()} GPUs are visible"
        )
    work_dir = Path(args.work_dir)
    specs = worker_process_specs(
        script=Path(__file__).resolve(),
        profile=args.profile,
        gpu_count=args.gpu_count,
        warmup_steps=args.warmup_steps,
        bench_steps=args.bench_steps,
        work_dir=work_dir,
        variants=args.variants,
    )
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)

    processes: list[tuple[WorkerProcessSpec, subprocess.Popen[Any], Any]] = []
    for spec in specs:
        log_file = spec.log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(spec.env)
        process = subprocess.Popen(
            spec.argv,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((spec, process, log_file))
        print(f"WORKER_START gpu={spec.gpu_id} pid={process.pid} log={spec.log_path}", flush=True)

    failures: list[int] = []
    results: list[dict[str, Any]] = []
    for spec, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        result = _parse_result(spec.log_path) if return_code == 0 else None
        if result is None:
            failures.append(spec.gpu_id)
        else:
            results.append(result)
        print(f"WORKER_EXIT gpu={spec.gpu_id} code={return_code}", flush=True)

    if failures:
        print(
            "THROUGHPUT_SMOKE_FAILED "
            + json.dumps(
                {
                    "failed_gpu_ids": failures,
                    "logs": [str(specs[gpu_id].log_path) for gpu_id in failures],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1

    worker_tps = [float(result["tokens_per_second"]) for result in results]
    summary = {
        "profile": args.profile,
        "topology": "independent_world_size_one_workers",
        "gpu_count": args.gpu_count,
        "worker_tokens_per_second": worker_tps,
        "worker_variants": [str(result["variant"]) for result in results],
        "per_gpu_tps_min": min(worker_tps),
        "per_gpu_tps_median": statistics.median(worker_tps),
        "per_gpu_tps_max": max(worker_tps),
        "aggregate_tokens_per_second": sum(worker_tps),
    }
    print("THROUGHPUT_SMOKE_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="olmoe-hpo")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="adam-mb4096")
    parser.add_argument(
        "--variants",
        type=lambda value: value.split(","),
        help="Comma-separated per-GPU variants; launcher only",
    )
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--bench-steps", type=int, default=5)
    parser.add_argument(
        "--work-dir",
        default=os.environ.get("THROUGHPUT_SMOKE_WORK_DIR", "/workspace/throughput-smoke"),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu_count <= 0:
        parser.error("--gpu-count must be positive")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be non-negative")
    if args.bench_steps <= 0:
        parser.error("--bench-steps must be positive")
    if not args.worker and args.variants is not None and len(args.variants) != args.gpu_count:
        parser.error("--variants must provide exactly one variant per GPU")
    return args


def main(argv: list[str] | None = None) -> int:
    """Run either one isolated worker or the multi-worker launcher."""

    args = _parse_args(argv)
    return _run_worker(args) if args.worker else _run_launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
