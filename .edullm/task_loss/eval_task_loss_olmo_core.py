#!/usr/bin/env python3
"""Exact 20-label OLMES BPB evaluation for curriculum MoE checkpoints.

This is a self-contained branch packaging of the evaluator used by the
curriculum methodology at edu-llm/edullm commit
``b435cbe9c352399fc4ab54b310f36d28f6c9746f``. It supports OLMo-core DCP
checkpoints and the post-hoc ``model_eval.pt`` format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
import torch.distributed as dist
import yaml
from olmo.config import EvaluatorConfig, EvaluatorType, ModelConfig, TokenizerConfig, TrainConfig
from olmo.eval import build_evaluator
from olmo.tokenizer import Tokenizer
from olmo.torch_util import get_local_rank
from olmo.util import prepare_cli_environment
from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.nn.attention import AttentionBackendName

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))
from curriculum_model import build_model_config  # noqa: E402
from task_loss.item_identity import normalize as normalize_item_text  # noqa: E402
from task_loss.item_identity import normalizer_fingerprint, sha256  # noqa: E402
from task_loss.per_item import reduce_per_item, write_per_item  # noqa: E402

try:
    from olmo.util import add_cached_path_clients
except ImportError:

    def add_cached_path_clients() -> None:
        return None


log = logging.getLogger("curriculum_task_loss")
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


def build_model() -> torch.nn.Module:
    try:
        backend = AttentionBackendName.torch
    except Exception:
        backend = None
    kwargs: dict[str, Any] = {"vocab_size": EMBEDDING_SIZE}
    if backend is not None:
        kwargs["attn_backend"] = backend
    model = build_model_config(**kwargs).build(init_device="cuda")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _local_tensor(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor state, got {type(value)}")
    full = getattr(value, "full_tensor", None)
    if callable(full):
        return full().detach().cpu()
    local = getattr(value, "to_local", None)
    if callable(local):
        return local().detach().cpu()
    return value.detach().cpu()


def _model_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint model payload must be a mapping, got {type(payload)}")
    if isinstance(payload.get("train_module"), dict):
        payload = payload["train_module"]
    if isinstance(payload.get("model"), dict):
        payload = payload["model"]
    prefixed = {
        key.removeprefix("model."): _local_tensor(value)
        for key, value in payload.items()
        if key.startswith("model.") and torch.is_tensor(value)
    }
    if prefixed:
        return prefixed
    tensors = {
        str(key): _local_tensor(value) for key, value in payload.items() if torch.is_tensor(value)
    }
    if not tensors:
        raise RuntimeError("checkpoint contains no model tensors")
    return tensors


def _checkpoint_step(checkpoint: Path, payload: dict[str, Any] | None = None) -> int:
    if payload is not None and payload.get("step") is not None:
        return int(payload["step"])
    step_file = checkpoint / "step.txt"
    if step_file.is_file():
        return int(step_file.read_text(encoding="utf-8").strip())
    return int(checkpoint.name.removeprefix("step").split("-")[0])


def materialize_distcp_model_eval(checkpoint: Path) -> Path:
    """Unshard DCP before process-group initialization."""
    if dist.is_initialized():
        raise RuntimeError("DCP materialization must happen before distributed initialization")
    target = checkpoint / "model_eval.pt"
    if target.is_file():
        return target
    source = checkpoint / "model_and_optim"
    if not (source / ".metadata").is_file():
        raise FileNotFoundError(f"missing DCP metadata under {source}")
    temporary = Path(tempfile.mkdtemp(prefix="curriculum-task-loss-"))
    try:
        model_path, _ = unshard_checkpoint(
            source,
            temporary,
            optim=False,
            save_overwrite=True,
            quiet=True,
        )
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        torch.save(
            {"step": _checkpoint_step(checkpoint), "model": _model_state(payload)},
            target,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return target


def load_model(checkpoint: Path, model: torch.nn.Module) -> int:
    path = checkpoint / "model_eval.pt"
    if not path.is_file():
        state_path = checkpoint / "state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"no model_eval.pt or state.pt under {checkpoint}")
        path = state_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = _model_state(payload)
    embeddings = state.get("embeddings.weight")
    expected_embeddings = model.state_dict().get("embeddings.weight")
    if embeddings is not None:
        if expected_embeddings is None:
            raise RuntimeError("model has no embeddings.weight parameter")
        expected_shape = tuple(expected_embeddings.shape)
        if tuple(embeddings.shape) != expected_shape:
            raise RuntimeError(
                f"invalid embeddings.weight shape: {tuple(embeddings.shape)}; "
                f"expected {expected_shape}"
            )
    # strict=True: a partially-loaded checkpoint must fail loudly, not
    # silently score a model with up to 5% randomly-initialized tensors
    # (audit finding 1e -- the prior 5%-missing-key tolerance could produce
    # a plausible-looking but wrong bpb with no error at all).
    model.load_state_dict(state, strict=True)
    return _checkpoint_step(checkpoint, payload if isinstance(payload, dict) else None)


def load_base_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"expected mapping in evaluator config: {path}")
    model = raw.get("model") or {}
    tokenizer = raw.get("tokenizer") or {}
    config = TrainConfig(
        model=ModelConfig(
            d_model=int(model.get("d_model", 2048)),
            n_heads=int(model.get("n_heads", 16)),
            n_layers=int(model.get("n_layers", 16)),
            mlp_ratio=int(model.get("mlp_ratio", 8)),
            vocab_size=int(model.get("vocab_size", 100_278)),
            embedding_size=int(model.get("embedding_size", EMBEDDING_SIZE)),
            eos_token_id=int(model.get("eos_token_id", 100_257)),
            pad_token_id=int(model.get("pad_token_id", 100_277)),
            # Was omitted entirely -> ai2-olmo's ModelConfig default, which
            # is confirmed to be 1024 at the pinned commit
            # (090253dac6688f2532509daa7aa2eb5fae50e956, olmo/config.py:
            # `max_sequence_length: int = 1024`), silently truncating 5-shot
            # ICL context at half the model's trained 2048 sequence length
            # (audit finding 1e). The field name is correct: ModelConfig is
            # a dataclass, so a wrong keyword would raise at construction.
            max_sequence_length=int(model.get("max_sequence_length", 2048)),
        ),
        tokenizer=TokenizerConfig(
            identifier=str(tokenizer.get("identifier", "allenai/dolma2-tokenizer"))
        ),
        global_train_batch_size=int(raw.get("global_train_batch_size", 8)),
        device_train_microbatch_size=int(raw.get("device_train_microbatch_size", 1)),
        device_eval_batch_size=int(raw.get("device_eval_batch_size", 4)),
        seed=int(raw.get("seed", 42)),
    )
    config.evaluators = []
    return config


def require_labels() -> None:
    from olmo.eval.downstream import label_to_task_map

    missing = [label for label in TASK_LABELS if label not in label_to_task_map]
    if missing:
        raise RuntimeError(
            f"installed ai2-olmo lacks {len(missing)} required BPB labels; "
            f"first missing label is {missing[0]!r}"
        )


def model_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(input_ids, return_logits=True)
    if hasattr(output, "logits"):
        return output.logits
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and torch.is_tensor(output[0]):
        return output[0]
    raise RuntimeError(f"unexpected model output type: {type(output)}")


def capture_per_item(evaluator: Any) -> list[tuple[int, int, float]]:
    """Read per-document values off the ICL metric before they are reduced.

    `ICLMetric.update()` accumulates `(doc_id, cont_id, log_likelihood)`
    triples, and for a `*_bpb` label that third value is ALREADY the
    per-continuation bits-per-byte -- the metric divides by
    `cont_byte_len` at update time. `compute()` then keeps only the gold
    continuation per document and averages, discarding everything else.

    So the per-item numbers we want already exist; they are thrown away one
    line before the caller sees them. This reads them out instead.

    Returns this rank's LOCAL shard only. The eval loader uses a
    `DistributedSampler`, so the caller must gather across ranks, and must
    expect duplicate `doc_id`s because that sampler pads to divisibility.
    """
    state = getattr(evaluator.eval_metric, "loglikelihoods", None)
    if state is None:
        raise RuntimeError(
            "ICLMetric has no `loglikelihoods` state; the installed ai2-olmo "
            "has changed shape and per-item capture would silently record "
            "nothing. Refusing to continue."
        )
    rows: list[tuple[int, int, float]] = []
    for entry in state:
        doc_id, cont_id, value = (float(x) for x in entry.tolist())
        rows.append((int(doc_id), int(cont_id), float(value)))
    return rows


ITEM_IDENTITY_SAMPLE = 64


def capture_item_identity(evaluator: Any, tokenizer: Tokenizer) -> dict[str, Any]:
    """Record which items this label scored, so hits can be joined to bpb.

    `doc_id` is a bare positional integer from ai2-olmo's dataset
    construction, and nothing else in the per-item output identifies the item.
    Without this, a contamination hit cannot be tied to a per-item bpb number,
    and an off-by-one join produces entirely plausible numbers.

    Two things are recorded:

      - `n_docs` and a digest of the full sorted doc_id list, which catches a
        different item set or a reordering.
      - normalized-text hashes for `ITEM_IDENTITY_SAMPLE` deterministically
        spaced doc_ids, which catches the case where the doc_ids line up but
        the underlying oe-eval request text has changed.

    Only a sample is hashed because decoding every context at every permanent
    checkpoint would cost real time for no extra safety -- a text change that
    spares 64 spread-out items is not a realistic failure. The hash is of the
    FULL context rather than the preamble-stripped stem: stripping needs every
    context in the label decoded to find the shared prefix, and the section 4a
    dump records `full_context_sha256` for exactly this comparison.
    """
    task = evaluator.eval_loader.dataset
    by_doc: dict[int, Any] = {}
    for sample in task.samples:
        by_doc.setdefault(int(sample["doc_id"]), sample)
    doc_ids = sorted(by_doc)
    stride = max(1, len(doc_ids) // ITEM_IDENTITY_SAMPLE)
    sampled: dict[str, dict[str, str]] = {}
    for doc_id in doc_ids[::stride][:ITEM_IDENTITY_SAMPLE]:
        sample = by_doc[doc_id]
        sampled[str(doc_id)] = {
            "full_context_sha256": sha256(
                normalize_item_text(tokenizer.decode(list(sample["ctx"])))
            ),
            "gold_sha256": sha256(
                normalize_item_text(tokenizer.decode(list(sample["continuation"])))
            ),
        }
    return {
        "n_docs": len(doc_ids),
        "doc_id_digest": sha256(",".join(str(d) for d in doc_ids)),
        "sampled": sampled,
    }


def evaluate_label(
    model: torch.nn.Module,
    config: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    label: str,
    batch_size: int,
) -> tuple[float, list[tuple[int, int, float]], dict[str, Any]]:
    evaluator = build_evaluator(
        config,
        EvaluatorConfig(
            label=label,
            type=EvaluatorType.downstream,
            device_eval_batch_size=batch_size,
        ),
        tokenizer,
        device,
    )
    evaluator.reset_metrics()
    for batch in evaluator.eval_loader:
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        evaluator.eval_metric.update(batch, model_logits(model, batch["input_ids"]))
    # Capture BEFORE compute_metrics(): torchmetrics syncs state across ranks
    # inside compute() and then unsyncs, so reading afterwards would give the
    # local shard anyway -- but reading first keeps the intent explicit and
    # avoids depending on that restore behaviour.
    per_item = capture_per_item(evaluator)
    identity = capture_item_identity(evaluator, tokenizer)
    metrics = evaluator.compute_metrics()
    if len(metrics) != 1:
        raise RuntimeError(f"{label} returned unexpected metrics: {sorted(metrics)}")
    return float(next(iter(metrics.values()))), per_item, identity


def run_suite(
    checkpoint: Path,
    output: Path,
    run_name: str,
    *,
    base_config: Path,
    batch_size: int,
) -> dict[str, Any]:
    if not dist.is_initialized():
        raise RuntimeError("20-label evaluation requires an initialized process group")
    prepare_cli_environment()
    add_cached_path_clients()
    require_labels()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = build_model()
    step = load_model(checkpoint, model)
    model.to(device)
    config = load_base_config(base_config)
    config.device_eval_batch_size = batch_size

    tokenizer: Tokenizer | None = None
    if rank == 0:
        tokenizer = Tokenizer.from_train_config(config)
    dist.barrier()
    if tokenizer is None:
        tokenizer = Tokenizer.from_train_config(config)
    dist.barrier()

    # Warm shared caches on one rank before every DistributedSampler is built.
    if rank == 0:
        for label in TASK_LABELS:
            evaluator = build_evaluator(
                config,
                EvaluatorConfig(
                    label=label,
                    type=EvaluatorType.downstream,
                    device_eval_batch_size=batch_size,
                ),
                tokenizer,
                device,
            )
            try:
                next(iter(evaluator.eval_loader))
            except StopIteration:
                pass
    dist.barrier()

    local_results = {
        label: evaluate_label(model, config, tokenizer, device, label, batch_size)
        for label in TASK_LABELS
    }
    gathered: list[dict[str, tuple[float, list, dict]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_results)
    merged: dict[str, float] = {}
    # The scalar is already synced across ranks by compute_metrics(), so
    # overwriting is fine for it -- but the per-item rows are rank-local
    # shards and must be concatenated, not overwritten.
    raw_rows: dict[str, list[tuple[int, int, float]]] = {label: [] for label in TASK_LABELS}
    # The identity record describes the dataset, which every rank holds in
    # full, so it is rank-invariant and overwriting is correct. Disagreement
    # between ranks would mean the ranks built different datasets, which is
    # asserted rather than assumed.
    identities: dict[str, dict[str, Any]] = {}
    for result in gathered:
        if not result:
            continue
        for label, (value, rows, identity) in result.items():
            merged[label] = float(value)
            raw_rows[label].extend(tuple(row) for row in rows)
            previous = identities.setdefault(label, identity)
            if previous != identity:
                raise RuntimeError(
                    f"{label}: ranks disagree about the scored item set "
                    f"(doc_id digest {previous['doc_id_digest'][:12]} vs "
                    f"{identity['doc_id_digest'][:12]}); per-item rows from "
                    f"different ranks would describe different items"
                )
    missing = [label for label in TASK_LABELS if label not in merged]
    if missing:
        raise RuntimeError(f"incomplete 20-label suite; first missing label: {missing[0]}")
    labels = {label: float(merged[label]) for label in TASK_LABELS}
    per_item, padding_duplicates = reduce_per_item(raw_rows, labels)
    per_item_path = output.parent / f"{output.stem}_per_item.jsonl.gz"
    payload: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "step": step,
        "world_size": world_size,
        "task_loss_bpb": labels,
        "labels": labels,
        "macro_mean": sum(labels.values()) / len(labels),
        "raw_label_count": len(labels),
        "suite_complete": True,
        "per_item_file": per_item_path.name,
        "per_item_rows": sum(len(values) for values in per_item.values()),
        "sampler_padding_duplicates": padding_duplicates,
        # Section 4b: what makes a contamination hit joinable to a bpb number.
        "item_identity": identities,
        "normalizer_fingerprint": normalizer_fingerprint(),
    }
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_per_item(per_item_path, per_item, step)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dist.barrier()
    return payload


def detect_format(checkpoint: Path) -> str:
    if (checkpoint / "model_eval.pt").is_file() or (checkpoint / "state.pt").is_file():
        return "state_pt"
    if (checkpoint / "model_and_optim" / ".metadata").is_file():
        return "distcp"
    raise RuntimeError(f"cannot detect checkpoint format under {checkpoint}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--run-name", required=True)
    result.add_argument("--format", choices=("auto", "state_pt", "distcp"), default="auto")
    result.add_argument("--base-config", type=Path)
    result.add_argument("--device-eval-batch-size", type=int, default=4)
    return result


def main() -> None:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_config = args.base_config or Path(
        os.environ.get(
            "LADDER_BASE_CONFIG",
            Path(__file__).with_name("ladder_base_config.yaml"),
        )
    )
    if not base_config.is_file():
        raise SystemExit(f"missing packaged ladder config: {base_config}")
    checkpoint_format = detect_format(args.checkpoint) if args.format == "auto" else args.format
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if checkpoint_format == "distcp" and not (args.checkpoint / "model_eval.pt").is_file():
        if local_rank == 0:
            materialize_distcp_model_eval(args.checkpoint)
        else:
            target = args.checkpoint / "model_eval.pt"
            for _ in range(3600):
                if target.is_file():
                    break
                time.sleep(1)
            else:
                raise SystemExit(f"timed out waiting for {target}")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
    run_suite(
        args.checkpoint,
        args.out,
        args.run_name,
        base_config=base_config,
        batch_size=args.device_eval_batch_size,
    )


if __name__ == "__main__":
    main()
