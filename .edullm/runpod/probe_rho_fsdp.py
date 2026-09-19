#!/usr/bin/env python3
"""Eight-rank CUDA/FSDP validation probe for GPU-local RHO reference scoring."""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.distributed as dist
from torch import Tensor, nn

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import seed_all

import token_selection_370m.selection as selection
from runpod.entrypoint import manifest, resolve_local_corpus
from token_selection_370m.arms import get_arm
from token_selection_370m.recipe import (
    GLOBAL_BATCH_TOKENS,
    RANK_MICROBATCH_TOKENS,
    build_trainer,
)
from token_selection_370m.selection import WeightShadow, selection_weights
from token_selection_370m.train_module import TokenWeightedTrainModule, load_flat_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/workspace/edullm-inputs/token-selection/ready.json"),
    )
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--scorer-repeats", type=int, default=5)
    parser.add_argument("--target-tokens-per-sec-gpu", type=float, default=45_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def local(parameter: Tensor) -> Tensor:
    return selection._local(parameter)


def clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def sync_elapsed(operation) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return time.perf_counter() - start


@contextlib.contextmanager
def pre_materialized_reference(
    model: nn.Module, reference_state: Mapping[str, Tensor]
) -> Iterator[None]:
    """Recreate the old per-score reference materialization as an oracle."""
    saved: dict[str, Tensor] = {}
    try:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                saved[name] = local(parameter).detach().clone()
                selection._write(parameter, reference_state[name])
        yield
    finally:
        with torch.no_grad():
            selection._reshard(model)
            for name, parameter in model.named_parameters():
                local(parameter).copy_(saved[name])


@contextlib.contextmanager
def forbid_hot_path_materialization() -> Iterator[dict[str, int]]:
    """Fail on the old full-tensor helpers or an explicit device-to-CPU transfer."""
    from torch.distributed.tensor import DTensor

    calls = {"full_tensor": 0, "cpu": 0, "to_cpu": 0, "write": 0, "snapshot": 0}
    original_full_tensor = DTensor.full_tensor
    original_cpu = torch.Tensor.cpu
    original_to = torch.Tensor.to
    original_write = selection._write
    original_snapshot = selection._snapshot

    def forbidden(name: str):
        def fail(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"optimized scoring called forbidden {name}")

        return fail

    def guarded_to(tensor, *args, **kwargs):
        device = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device, int)):
            device = args[0]
        if device is not None:
            try:
                if torch.device(device).type == "cpu":
                    calls["to_cpu"] += 1
                    raise AssertionError("optimized scoring attempted a CPU transfer")
            except (TypeError, RuntimeError):
                pass
        return original_to(tensor, *args, **kwargs)

    DTensor.full_tensor = forbidden("full_tensor")  # type: ignore[method-assign]
    torch.Tensor.cpu = forbidden("cpu")  # type: ignore[method-assign]
    torch.Tensor.to = guarded_to  # type: ignore[method-assign]
    selection._write = forbidden("write")
    selection._snapshot = forbidden("snapshot")
    try:
        yield calls
    finally:
        DTensor.full_tensor = original_full_tensor  # type: ignore[method-assign]
        torch.Tensor.cpu = original_cpu  # type: ignore[method-assign]
        torch.Tensor.to = original_to  # type: ignore[method-assign]
        selection._write = original_write
        selection._snapshot = original_snapshot


def assert_restored(model: nn.Module, expected: Mapping[str, Tensor]) -> None:
    failures = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(local(parameter), expected[name])
    ]
    if failures:
        raise AssertionError(f"training shards changed after reference scoring: {failures[:8]}")


def score_oracle(
    module: TokenWeightedTrainModule,
    reference_state: Mapping[str, Tensor],
    ids: Tensor,
    labels: Tensor,
    kwargs: Mapping[str, Any],
) -> Tensor:
    with (
        module._score_mode(),
        torch.no_grad(),
        pre_materialized_reference(module.model, reference_state),
    ):
        output = module.model_forward(
            ids,
            labels=labels,
            ignore_index=module.label_ignore_index,
            loss_reduction="none",
            return_logits=False,
            **kwargs,
        )
    if not isinstance(output, LMOutputWithLoss):
        raise AssertionError("oracle did not return per-token loss")
    module.model.reset_auxiliary_metrics()
    return module._loss_tensor(output, labels, "ce_loss").detach().clone()


def training_backward(
    module: TokenWeightedTrainModule,
    ids: Tensor,
    labels: Tensor,
    kwargs: Mapping[str, Any],
    weights: Tensor,
) -> None:
    module.zero_grads()
    with module._train_microbatch_context(0, 1):
        output = module.model_forward(
            ids,
            labels=labels,
            ignore_index=module.label_ignore_index,
            loss_reduction="none",
            z_loss_multiplier=module.z_loss_multiplier,
            return_logits=False,
            **kwargs,
        )
        if not isinstance(output, LMOutputWithLoss):
            raise AssertionError("training forward did not return per-token loss")
        loss = (module._loss_tensor(output, labels, "loss").float() * weights).sum()
        loss.div_(weights.sum()).backward()
    module.model.post_batch(dry_run=True)


def main() -> None:
    args = parse_args()
    os.environ["EDULLM_RUNPOD_INPUT_MANIFEST"] = str(args.manifest)
    os.environ["WANDB_MODE"] = "disabled"
    prepare_training_environment(seed=6198)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != 8 or torch.cuda.device_count() < 8:
        raise AssertionError(
            f"probe requires one 8-GPU node, got world_size={world_size}, "
            f"visible_cuda={torch.cuda.device_count()}"
        )

    output_path = args.output
    try:
        torch.set_float32_matmul_precision("high")
        seed_all(6198)
        payload = manifest()
        if payload.get("arm") != "rho-1":
            raise AssertionError(f"staged arm is {payload.get('arm')!r}, not 'rho-1'")
        reference_path = Path(payload["references"]["reference"])
        if not reference_path.is_file():
            raise AssertionError(f"staged reference is missing: {reference_path}")
        arm = get_arm("rho-1")
        corpus_record = payload["corpora"][arm.dataset_id]
        corpus = resolve_local_corpus(
            dataset_id=arm.dataset_id,
            version=str(corpus_record["version"]),
            tokenizer_id="tokenizer/dolma2-bpe",
        )

        root = Path(f"/tmp/rho-fsdp-probe-{os.environ.get('MASTER_PORT', 'default')}")
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True)
        dist.barrier()
        try:
            trainer = build_trainer(
                arm,
                corpus,
                max_tokens=GLOBAL_BATCH_TOKENS * max(args.timed_steps, 2),
                save_folder=root / "checkpoints",
                work_dir=root / "work",
                progress_dir=root / "progress",
                task_loss_script=root / "no-task-loss.py",
                reference_path=str(reference_path),
                production=False,
            )
            module = trainer.train_module
            if not isinstance(module, TokenWeightedTrainModule):
                raise AssertionError(f"unexpected train module: {type(module).__name__}")
            shadow = module.selection_state.reference
            if not isinstance(shadow, WeightShadow):
                raise AssertionError("RHO reference shadow was not initialized")

            parameters = dict(module.model.named_parameters())
            local_checks = {
                name: (
                    tensor.device.type == "cuda"
                    and tensor.device.index == local_rank
                    and tensor.shape == local(parameters[name]).shape
                    and tensor.dtype == local(parameters[name]).dtype
                    and not tensor.requires_grad
                )
                for name, tensor in shadow.weights.items()
            }
            if set(local_checks) != set(parameters) or not all(local_checks.values()):
                bad = [name for name, valid in local_checks.items() if not valid]
                raise AssertionError(f"reference is not rank-local CUDA state: {bad[:8]}")

            trainer.data_loader.reshuffle(1)
            batch = next(iter(trainer.data_loader))
            if "labels" not in batch:
                batch["labels"] = get_labels(batch, label_ignore_index=module.label_ignore_index)
            micro_batches = split_batch(
                batch, RANK_MICROBATCH_TOKENS // batch["input_ids"].shape[1]
            )
            ids, labels, model_kwargs = module._prepare_batch(dict(micro_batches[0]))
            if labels is None:
                raise AssertionError("probe microbatch has no labels")
            valid = (labels != module.label_ignore_index).to(module.device, non_blocking=True)
            reference_state = load_flat_weights(reference_path)

            training_before = {
                name: local(parameter).detach().clone()
                for name, parameter in module.model.named_parameters()
            }
            oracle_loss = score_oracle(module, reference_state, ids, labels, model_kwargs)
            assert_restored(module.model, training_before)
            optimized_loss = module._score(shadow, ids, labels, model_kwargs)
            if not torch.equal(optimized_loss, oracle_loss):
                max_abs = (optimized_loss - oracle_loss).abs().max().item()
                raise AssertionError(f"reference CE differs from oracle, max_abs={max_abs}")

            current_output = module.model_forward(
                ids,
                labels=labels,
                ignore_index=module.label_ignore_index,
                loss_reduction="none",
                z_loss_multiplier=module.z_loss_multiplier,
                return_logits=False,
                **model_kwargs,
            )
            if not isinstance(current_output, LMOutputWithLoss):
                raise AssertionError("current model did not return per-token loss")
            current_ce = module._loss_tensor(current_output, labels, "ce_loss").detach()
            selection._reshard(module.model)
            oracle_mask = selection_weights(
                "rho_excess",
                valid=valid,
                keep_fraction=arm.keep_fraction,
                step=0,
                seed=42,
                current=current_ce,
                reference=oracle_loss,
            )
            optimized_mask = selection_weights(
                "rho_excess",
                valid=valid,
                keep_fraction=arm.keep_fraction,
                step=0,
                seed=42,
                current=current_ce,
                reference=optimized_loss,
            )
            if not torch.equal(optimized_mask, oracle_mask):
                raise AssertionError("optimized RHO mask differs from oracle mask")
            module.model.reset_auxiliary_metrics()
            assert_restored(module.model, training_before)

            for _ in range(3):
                module._score(shadow, ids, labels, model_kwargs)
                assert_restored(module.model, training_before)
            try:
                with shadow.swap_to(module.model):
                    raise RuntimeError("intentional scoring exception")
            except RuntimeError as error:
                if str(error) != "intentional scoring exception":
                    raise
            assert_restored(module.model, training_before)

            module.zero_grads()
            with forbid_hot_path_materialization() as forbidden_calls:
                guarded_loss = module._score(shadow, ids, labels, model_kwargs)
            if forbidden_calls != {
                "full_tensor": 0,
                "cpu": 0,
                "to_cpu": 0,
                "write": 0,
                "snapshot": 0,
            }:
                raise AssertionError(f"forbidden hot-path calls observed: {forbidden_calls}")
            if not torch.equal(guarded_loss, optimized_loss):
                raise AssertionError("guarded scorer changed reference loss")

            if any(parameter.grad is not None for parameter in module.model.parameters()):
                raise AssertionError("reference scoring populated training gradients")
            if any(weight.grad is not None or weight.requires_grad for weight in shadow.weights.values()):
                raise AssertionError("reference shards participate in autograd")
            training_backward(module, ids, labels, model_kwargs, optimized_mask)
            local_grad_count = sum(
                parameter.grad is not None for parameter in module.model.parameters()
            )
            if local_grad_count == 0:
                raise AssertionError("training backward produced no parameter gradients")
            if any(weight.grad is not None or weight.requires_grad for weight in shadow.weights.values()):
                raise AssertionError("training backward reached reference shards")
            module.zero_grads()
            assert_restored(module.model, training_before)

            del reference_state, training_before, oracle_loss, current_output, current_ce
            gc.collect()
            torch.cuda.empty_cache()

            for _ in range(2):
                module._score(shadow, ids, labels, model_kwargs)
            scorer_times = [
                sync_elapsed(lambda: module._score(shadow, ids, labels, model_kwargs))
                for _ in range(args.scorer_repeats)
            ]

            torch.cuda.reset_peak_memory_stats()
            step_times: list[float] = []
            step_batch = clone_batch(batch)
            for step_index in range(args.timed_steps):
                trainer.global_step = step_index + 1

                def run_step() -> None:
                    module.train_batch(clone_batch(step_batch))
                    module.optim_step()
                    module.zero_grads()

                step_times.append(sync_elapsed(run_step))

            local_tokens = int(step_batch["input_ids"].numel())
            local_result = {
                "rank": rank,
                "local_rank": local_rank,
                "reference_parameters": len(shadow.weights),
                "reference_local_numel": sum(tensor.numel() for tensor in shadow.weights.values()),
                "reference_cuda_device": str(next(iter(shadow.weights.values())).device),
                "reference_all_local_cuda": all(local_checks.values()),
                "oracle_loss_exact": True,
                "oracle_mask_exact": True,
                "restoration_exact": True,
                "exception_restoration_exact": True,
                "training_grad_parameter_count": local_grad_count,
                "reference_grad_count": sum(
                    weight.grad is not None for weight in shadow.weights.values()
                ),
                "forbidden_hot_path_calls": forbidden_calls,
                "microbatch_tokens": int(ids.numel()),
                "scorer_seconds": scorer_times,
                "scorer_tokens_per_second": [
                    float(ids.numel()) / elapsed for elapsed in scorer_times
                ],
                "step_tokens": local_tokens,
                "step_seconds": step_times,
                "step_tokens_per_second_per_gpu": [
                    local_tokens / elapsed for elapsed in step_times
                ],
                "memory_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "memory_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "memory_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                "memory_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            }

            gathered: list[dict[str, Any] | None] = [None] * world_size
            dist.all_gather_object(gathered, local_result)
            if rank == 0:
                rank_results = [result for result in gathered if result is not None]
                worst_step_seconds = [
                    max(result["step_seconds"][index] for result in rank_results)
                    for index in range(args.timed_steps)
                ]
                conservative_throughput = [
                    local_tokens / elapsed for elapsed in worst_step_seconds
                ]
                summary = {
                    "status": "PASS",
                    "world_size": world_size,
                    "reference_path": str(reference_path),
                    "manifest_total_bytes": int(payload["total_bytes"]),
                    "target_tokens_per_second_per_gpu": args.target_tokens_per_sec_gpu,
                    "conservative_step_tokens_per_second_per_gpu": conservative_throughput,
                    "target_met_each_timed_step": [
                        value >= args.target_tokens_per_sec_gpu
                        for value in conservative_throughput
                    ],
                    "max_peak_allocated_gib": max(
                        result["memory_peak_allocated_gib"] for result in rank_results
                    ),
                    "max_peak_reserved_gib": max(
                        result["memory_peak_reserved_gib"] for result in rank_results
                    ),
                    "ranks": rank_results,
                }
                rendered = json.dumps(summary, indent=2, sort_keys=True)
                print("RHO_FSDP_PROBE_RESULT=" + rendered, flush=True)
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(rendered + "\n", encoding="utf-8")
        finally:
            dist.barrier()
            if rank == 0:
                shutil.rmtree(root, ignore_errors=True)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
