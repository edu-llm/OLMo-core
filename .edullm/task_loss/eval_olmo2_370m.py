#!/usr/bin/env python3
"""Self-contained 20-label OLMES BPB evaluator for OLMo2-370M checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.distributed as dist
from olmo.config import EvaluatorConfig, EvaluatorType, ModelConfig, TokenizerConfig, TrainConfig
from olmo.eval import build_evaluator
from olmo.eval.downstream import label_to_task_map
from olmo.tokenizer import Tokenizer
from olmo.util import prepare_cli_environment
from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.transformer import TransformerConfig

LOG = logging.getLogger("final_validation_task_loss")
EMBEDDING_SIZE = 100_352
TASK_LABELS = (
    "arc_challenge_val_rc_5shot_bpb",
    "arc_challenge_test_rc_5shot_bpb",
    "arc_easy_val_rc_5shot_bpb",
    "arc_easy_test_rc_5shot_bpb",
    "boolq_val_rc_5shot_bpb",
    "csqa_val_rc_5shot_bpb",
    "hellaswag_val_rc_5shot_bpb",
    "openbookqa_val_rc_5shot_bpb",
    "openbookqa_test_rc_5shot_bpb",
    "piqa_val_rc_5shot_bpb",
    "socialiqa_val_rc_5shot_bpb",
    "winogrande_val_rc_5shot_bpb",
    "mmlu_stem_val_rc_5shot_bpb",
    "mmlu_stem_test_rc_5shot_bpb",
    "mmlu_humanities_val_rc_5shot_bpb",
    "mmlu_humanities_test_rc_5shot_bpb",
    "mmlu_social_sciences_val_rc_5shot_bpb",
    "mmlu_social_sciences_test_rc_5shot_bpb",
    "mmlu_other_val_rc_5shot_bpb",
    "mmlu_other_test_rc_5shot_bpb",
)


class EvaluatorContractError(RuntimeError):
    """The checkpoint or installed evaluator violates the fixed task-loss suite."""


def build_eval_config(device_eval_batch_size: int) -> TrainConfig:
    """Construct the complete OLMo2-370M evaluator config."""

    config = TrainConfig(
        model=ModelConfig(
            d_model=1_024,
            n_heads=16,
            n_layers=16,
            mlp_ratio=8,
            vocab_size=100_278,
            embedding_size=EMBEDDING_SIZE,
            eos_token_id=100_257,
            pad_token_id=100_277,
        ),
        tokenizer=TokenizerConfig(identifier="allenai/dolma2-tokenizer"),
        global_train_batch_size=8,
        device_train_microbatch_size=1,
        device_eval_batch_size=int(device_eval_batch_size),
        seed=6_198,
    )
    config.evaluators = []
    return config


def require_suite() -> None:
    """Fail before loading a checkpoint when the pinned suite is incomplete."""

    missing = [label for label in TASK_LABELS if label not in label_to_task_map]
    if missing:
        raise EvaluatorContractError(
            "installed ai2-olmo lacks the fixed OLMES BPB suite "
            f"({len(missing)} missing; first={missing[0]!r})"
        )


def build_model() -> torch.nn.Module:
    """Build the exact stock model used by final validation."""

    config = TransformerConfig.olmo2_370M(
        vocab_size=EMBEDDING_SIZE,
        attn_backend=AttentionBackendName.torch,
    )
    model = config.build(init_device="cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def materialize_model_eval(checkpoint: Path) -> Path:
    """Convert an OLMo-core distributed checkpoint to one CPU model file."""

    output = checkpoint / "model_eval.pt"
    if output.is_file():
        return output
    source = checkpoint / "model_and_optim"
    if not (source / ".metadata").is_file():
        raise EvaluatorContractError(f"missing checkpoint metadata under {source}")
    if dist.is_initialized():
        raise EvaluatorContractError("checkpoint unsharding must precede process-group init")
    step_file = checkpoint / "step.txt"
    step = (
        int(step_file.read_text(encoding="utf-8").strip())
        if step_file.is_file()
        else int(checkpoint.name.removeprefix("step").split("-")[0])
    )
    temporary = Path(tempfile.mkdtemp(prefix="final-validation-unshard-"))
    try:
        unshard_checkpoint(
            dir=str(source),
            target_dir=str(temporary),
            optim=False,
            save_overwrite=True,
        )
        candidates = (
            temporary / "model.pt",
            temporary / "model.pth",
            temporary / "model_and_optim" / "model.pt",
        )
        model_path = next((path for path in candidates if path.is_file()), None)
        if model_path is None:
            model_files = sorted(temporary.rglob("*.pt"))
            if not model_files:
                raise EvaluatorContractError("unsharding produced no model file")
            model_path = model_files[0]
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and isinstance(raw.get("model"), dict):
            raw = raw["model"]
        if not isinstance(raw, dict):
            raise EvaluatorContractError("unsharded model state is not a mapping")
        clean = {
            str(name): value.detach().cpu()
            for name, value in raw.items()
            if torch.is_tensor(value)
        }
        torch.save({"step": step, "model": clean}, output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return output


def load_model(checkpoint: Path, model: torch.nn.Module) -> int:
    """Load and structurally validate a materialized 370M model."""

    path = checkpoint / "model_eval.pt"
    if not path.is_file():
        raise EvaluatorContractError(f"missing materialized evaluator state: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise EvaluatorContractError(f"invalid model evaluator payload: {path}")
    model_state = payload["model"]
    embedding = model_state.get("embeddings.weight")
    if embedding is None or tuple(embedding.shape) != (EMBEDDING_SIZE, 1_024):
        shape = None if embedding is None else tuple(embedding.shape)
        raise EvaluatorContractError(f"unexpected embeddings.weight shape: {shape}")
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if unexpected:
        LOG.warning("unexpected checkpoint keys (first 8): %s", unexpected[:8])
    if len(missing) > max(4, int(0.05 * (len(model_state) + len(missing)))):
        raise EvaluatorContractError(f"too many missing model keys: {len(missing)}")
    return int(payload["step"])


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Return logits across compatible OLMo-core output forms."""

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, return_logits=True)
    if hasattr(output, "logits"):
        return output.logits
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise EvaluatorContractError(f"unexpected model output: {type(output)}")


def evaluate_label(
    model: torch.nn.Module,
    config: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    label: str,
    device_eval_batch_size: int,
) -> float:
    """Evaluate one OLMES BPB label."""

    evaluator = build_evaluator(
        config,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=device_eval_batch_size,
        ),
        tokenizer,
        device,
    )
    evaluator.reset_metrics()
    batches = 0
    for batch in evaluator.eval_loader:
        batch = {
            key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        evaluator.eval_metric.update(batch, model_logits(model, batch["input_ids"]))
        batches += 1
    if batches == 0:
        raise EvaluatorContractError(f"{label}: evaluator produced no batches")
    metrics = evaluator.compute_metrics()
    values = list(metrics.values())
    if len(values) != 1:
        raise EvaluatorContractError(f"{label}: expected one metric, got {list(metrics)}")
    return float(values[0])


def warm_inputs(
    config: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    device_eval_batch_size: int,
) -> None:
    """Serialize cache population to avoid eight-rank cache lock contention."""

    if dist.get_rank() == 0:
        for label in TASK_LABELS:
            evaluator = build_evaluator(
                config,
                EvaluatorConfig(
                    label=label,
                    type=EvaluatorType.downstream,
                    device_eval_batch_size=device_eval_batch_size,
                ),
                tokenizer,
                device,
            )
            try:
                next(iter(evaluator.eval_loader))
            except StopIteration:
                pass
    dist.barrier()


def run(
    checkpoint: Path,
    output: Path,
    run_name: str,
    device_eval_batch_size: int,
) -> dict[str, Any]:
    """Run the fixed distributed suite and write one rank-zero result."""

    if not dist.is_initialized():
        raise EvaluatorContractError("distributed process group is not initialized")
    prepare_cli_environment()
    require_suite()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = build_model()
    step = load_model(checkpoint, model)
    model.to(device)
    config = build_eval_config(device_eval_batch_size)

    if rank == 0:
        tokenizer = Tokenizer.from_train_config(config)
    dist.barrier()
    if rank != 0:
        tokenizer = Tokenizer.from_train_config(config)
    dist.barrier()
    warm_inputs(config, tokenizer, device, device_eval_batch_size)

    local_results = {
        label: evaluate_label(
            model,
            config,
            tokenizer,
            device,
            label,
            device_eval_batch_size,
        )
        for label in TASK_LABELS
    }
    gathered: list[dict[str, float] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_results)
    labels: dict[str, float] = {}
    for result in gathered:
        if result:
            labels.update(result)
    if tuple(labels) != TASK_LABELS:
        missing = [label for label in TASK_LABELS if label not in labels]
        raise EvaluatorContractError(f"incomplete gathered suite: {missing}")
    payload: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "format": "state_pt",
        "step": step,
        "world_size": dist.get_world_size(),
        "task_loss_bpb": labels,
        "labels": labels,
        "macro_mean": sum(labels.values()) / len(TASK_LABELS),
        "raw_label_count": len(TASK_LABELS),
        "suite_complete": True,
    }
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    dist.barrier()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--format", choices=("auto", "distcp", "state_pt"), default="auto")
    parser.add_argument("--device-eval-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    evaluator_state = args.checkpoint / "model_eval.pt"
    if not evaluator_state.is_file():
        if local_rank == 0:
            materialize_model_eval(args.checkpoint)
        else:
            for _ in range(3_600):
                if evaluator_state.is_file():
                    break
                time.sleep(1)
            else:
                raise EvaluatorContractError(f"timed out waiting for {evaluator_state}")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    run(args.checkpoint, args.out, args.run_name, args.device_eval_batch_size)


if __name__ == "__main__":
    main()
