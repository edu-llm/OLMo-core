"""Helpers for cloning historical W&B runs with scaled optimizer-step axes."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

EVAL_ARTIFACT_RE = re.compile(r"^eval-step(\d+):")
SKIP_HISTORY_KEYS = frozenset({"_step", "_timestamp", "_runtime", "_wandb"})
STEP_METRIC_KEYS = frozenset({"checkpoint/step"})
TRAIN_LOSS_METRIC_MAP = {
    "train_loss": "train/loss",
    "tok_per_s": "train/tok_per_s",
    "tok_per_s_avg": "train/tok_per_s_avg",
    "tokens_seen": "train/tokens_seen",
}


def history_step_from_row(row: Mapping[str, Any]) -> int | None:
    """Read the W&B history step from a scanned row."""

    value = row.get("_step")
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if math.isnan(float(value)):
        return None
    return int(value)


def normalize_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    """Drop bookkeeping fields and alias legacy mixlaw loss names."""

    metrics: dict[str, float] = {}
    for key, value in row.items():
        if key in SKIP_HISTORY_KEYS:
            continue
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            continue
        metrics[key] = number
    if "train/CE loss" not in metrics and "train/loss" in metrics:
        metrics["train/CE loss"] = metrics["train/loss"]
    return metrics


def scale_step_metrics(metrics: Mapping[str, float], step_multiplier: int) -> dict[str, float]:
    """Scale logged step fields so control batches align with HPO step counts."""

    if step_multiplier == 1:
        return dict(metrics)
    scaled = dict(metrics)
    for key in STEP_METRIC_KEYS:
        if key in scaled:
            scaled[key] = float(scaled[key]) * step_multiplier
    return scaled


def merge_history_rows(row_groups: Iterable[Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Merge multiple history sources keyed by optimizer step."""

    merged: dict[int, dict[str, Any]] = {}
    for rows in row_groups:
        for row in rows:
            history_step = history_step_from_row(row)
            if history_step is None:
                continue
            bucket = merged.setdefault(history_step, {"_step": history_step})
            for key, value in row.items():
                if key == "_step" or value is None:
                    continue
                bucket[key] = value
    return [merged[step] for step in sorted(merged)]


def row_from_train_loss_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a mixlaw ``train_loss.jsonl`` record into a W&B history row."""

    row: dict[str, Any] = {"_step": int(record["step"])}
    for source_key, metric_key in TRAIN_LOSS_METRIC_MAP.items():
        if source_key in record:
            row[metric_key] = record[source_key]
    return row


def parse_train_loss_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse newline-delimited mixlaw training-loss records."""

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(row_from_train_loss_record(json.loads(line)))
    return rows


def eval_metrics_from_task_loss(payload: Mapping[str, Any]) -> dict[str, float]:
    """Convert a task-loss JSON payload into W&B eval metrics."""

    labels = payload.get("labels")
    if not isinstance(labels, Mapping):
        return {}
    metrics = {f"eval/bpb/{label}": float(value) for label, value in labels.items()}
    macro_mean = payload.get("macro_mean")
    if isinstance(macro_mean, (int, float)) and not isinstance(macro_mean, bool):
        metrics["eval/macro_bpb"] = float(macro_mean)
    elif metrics:
        metrics["eval/macro_bpb"] = sum(metrics.values()) / len(metrics)
    return metrics


def row_from_eval_task_loss(
    payload: Mapping[str, Any],
    *,
    history_step: int,
    global_batch_tokens: int,
) -> dict[str, Any]:
    """Build a history row for one checkpoint evaluation."""

    row: dict[str, Any] = {
        "_step": history_step,
        "checkpoint/step": history_step,
        "checkpoint/tokens_seen": history_step * global_batch_tokens,
    }
    row.update(eval_metrics_from_task_loss(payload))
    return row


def parse_eval_artifact_name(name: str) -> int | None:
    """Parse ``eval-step0000125:v0`` into checkpoint step 125."""

    match = EVAL_ARTIFACT_RE.match(name)
    if match is None:
        return None
    return int(match.group(1))


def iter_clone_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    step_multiplier: int = 1,
) -> list[tuple[int, dict[str, float]]]:
    """Convert history rows into `(wandb_step, metrics)` pairs."""

    if step_multiplier <= 0:
        raise ValueError("step_multiplier must be positive")

    grouped: dict[int, dict[str, float]] = {}
    for row in rows:
        history_step = history_step_from_row(row)
        if history_step is None:
            continue
        wandb_step = history_step * step_multiplier
        metrics = scale_step_metrics(normalize_metrics(row), step_multiplier)
        if not metrics:
            continue
        if wandb_step not in grouped:
            grouped[wandb_step] = metrics
        else:
            grouped[wandb_step].update(metrics)
    return sorted(grouped.items())


def clone_config(
    source_config: Mapping[str, Any],
    *,
    cloned_from: str,
    step_multiplier: int = 1,
) -> dict[str, Any]:
    """Copy a source config and annotate provenance."""

    config: dict[str, Any] = dict(source_config)
    config["cloned_from"] = cloned_from
    config["control_run"] = True
    config["clone_step_multiplier"] = step_multiplier
    return config


def merge_tags(source_tags: Sequence[str] | None, extra: Sequence[str]) -> list[str]:
    """Preserve source tags while adding clone metadata."""

    merged = list(source_tags or [])
    for tag in extra:
        if tag not in merged:
            merged.append(tag)
    return merged
