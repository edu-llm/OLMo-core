"""Tests for W&B clone helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "wandb_token_axis",
    REPO / ".edullm" / "wandb_token_axis.py",
)
assert SPEC and SPEC.loader
clone_helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clone_helpers)


def test_history_step_from_row() -> None:
    assert clone_helpers.history_step_from_row({"_step": 125}) == 125
    assert clone_helpers.history_step_from_row({"train/loss": 1.0}) is None


def test_normalize_metrics_aliases_train_loss() -> None:
    metrics = clone_helpers.normalize_metrics(
        {
            "_step": 10,
            "train/loss": 2.5,
            "train/tokens_seen": 4194304,
        }
    )
    assert metrics["train/CE loss"] == 2.5
    assert metrics["train/tokens_seen"] == 4194304.0
    assert "_step" not in metrics


def test_merge_history_rows_combines_sources() -> None:
    merged = clone_helpers.merge_history_rows(
        (
            [{"_step": 10, "train/loss": 3.0}],
            [{"_step": 10, "train/tok_per_s": 100.0}, {"_step": 125, "eval/macro_bpb": 1.5}],
        )
    )
    assert merged == [
        {"_step": 10, "train/loss": 3.0, "train/tok_per_s": 100.0},
        {"_step": 125, "eval/macro_bpb": 1.5},
    ]


def test_parse_train_loss_jsonl() -> None:
    rows = clone_helpers.parse_train_loss_jsonl(
        json.dumps(
            {
                "step": 20,
                "train_loss": 7.5,
                "tok_per_s": 100.0,
                "tok_per_s_avg": 90.0,
                "tokens_seen": 83886080,
            }
        )
    )
    assert rows == [
        {
            "_step": 20,
            "train/loss": 7.5,
            "train/tok_per_s": 100.0,
            "train/tok_per_s_avg": 90.0,
            "train/tokens_seen": 83886080,
        }
    ]


def test_iter_clone_rows_scales_and_merges_same_step() -> None:
    logged = clone_helpers.iter_clone_rows(
        [
            {"_step": 125, "train/loss": 2.0},
            {
                "_step": 125,
                "checkpoint/step": 125,
                "eval/macro_bpb": 1.5,
            },
        ],
        step_multiplier=16,
    )
    assert logged == [
        (
            2000,
            {
                "train/loss": 2.0,
                "train/CE loss": 2.0,
                "checkpoint/step": 2000.0,
                "eval/macro_bpb": 1.5,
            },
        )
    ]


def test_row_from_eval_task_loss() -> None:
    row = clone_helpers.row_from_eval_task_loss(
        {
            "labels": {"arc_easy_val_rc_5shot_bpb": 1.2},
            "macro_mean": 1.2,
        },
        history_step=125,
        global_batch_tokens=4_194_304,
    )
    assert row["_step"] == 125
    assert row["checkpoint/step"] == 125
    assert row["checkpoint/tokens_seen"] == 125 * 4_194_304
    assert row["eval/bpb/arc_easy_val_rc_5shot_bpb"] == 1.2
    assert row["eval/macro_bpb"] == 1.2


def test_clone_config_annotates_provenance() -> None:
    config = clone_helpers.clone_config(
        {"mix_name": "mix01"},
        cloned_from="https://wandb.ai/run",
        step_multiplier=16,
    )
    assert config["mix_name"] == "mix01"
    assert config["cloned_from"] == "https://wandb.ai/run"
    assert config["control_run"] is True
    assert config["clone_step_multiplier"] == 16
