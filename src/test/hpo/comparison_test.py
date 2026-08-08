import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from olmo_core.hpo.comparison import (
    COMPARISON_HELDOUT_LABEL,
    COMPARISON_HELDOUT_METRIC,
    DEFAULT_RECIPE_HPS,
    build_comparison_experiment,
    comparison_dataset_from_read,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_comparison_dataset_uses_sealed_train_and_heldout_splits():
    read = SimpleNamespace(
        paths=["s3://bucket/train-00000.u32le.bin"],
        train=["s3://bucket/train-00000.u32le.bin"],
        val=["s3://bucket/val-00000.u32le.bin"],
        dtype="uint32",
        byte_order="little",
        header_bytes=0,
    )
    dataset = comparison_dataset_from_read(
        read,
        dataset_id="pretrain/reservoir-dolma2",
        version="v1",
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    assert dataset.train_paths == tuple(read.paths)
    assert dataset.val_paths == tuple(read.val)
    assert set(dataset.train_paths).isdisjoint(dataset.val_paths)
    assert dataset.dtype.value == "uint32"


def test_comparison_dataset_uses_reader_hardened_train_paths():
    read = SimpleNamespace(
        paths=["s3://bucket/train-00000.u32le.bin"],
        train=[
            "s3://bucket/train-00000.u32le.bin",
            "s3://bucket/val-incorrectly-declared-train.u32le.bin",
        ],
        val=["s3://bucket/val-00000.u32le.bin"],
        dtype="uint32",
        byte_order="little",
        header_bytes=0,
    )
    dataset = comparison_dataset_from_read(
        read,
        dataset_id="pretrain/reservoir-dolma2",
        version="v1",
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    assert dataset.train_paths == tuple(read.paths)


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"val": None}, "held-out"),
        ({"header_bytes": 128}, "header"),
        ({"byte_order": "big"}, "byte order"),
    ],
)
def test_comparison_dataset_rejects_unsafe_or_missing_validation(changes, match):
    values = {
        "paths": ["s3://bucket/train-00000.u32le.bin"],
        "train": ["s3://bucket/train-00000.u32le.bin"],
        "val": ["s3://bucket/val-00000.u32le.bin"],
        "dtype": "uint32",
        "byte_order": "little",
        "header_bytes": 0,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=match):
        comparison_dataset_from_read(
            SimpleNamespace(**values),
            dataset_id="pretrain/reservoir-dolma2",
            version="v1",
            tokenizer_id="tokenizer/dolma2-bpe",
        )


def test_comparison_factory_builds_matched_train_and_eval_config(
    monkeypatch,
):
    read = SimpleNamespace(
        paths=["s3://bucket/train-00000.u32le.bin"],
        train=["s3://bucket/train-00000.u32le.bin"],
        val=["s3://bucket/val-00000.u32le.bin"],
        dtype="uint32",
        byte_order="little",
        header_bytes=0,
    )
    fake_package = types.ModuleType("edullm_data")
    fake_read = types.ModuleType("edullm_data.read")
    fake_s3 = types.ModuleType("edullm_data.s3")
    fake_read.dataset_paths = lambda dataset_id, version, *, s3: read

    class Boto3S3:
        @classmethod
        def default(cls):
            return object()

    fake_s3.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", fake_package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", fake_read)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", fake_s3)
    monkeypatch.setenv("EDULLM_DATASET_ID", "pretrain/reservoir-dolma2")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "/tmp/checkpoints")

    config = build_comparison_experiment(
        sequence_length=2048,
        global_batch_size=32768,
        rank_microbatch_size=4096,
        eval_steps=2,
    )
    assert config.dataset.paths == read.paths
    evaluator = config.trainer.callbacks["search_validation"]
    assert evaluator.eval_dataset.paths == read.val
    assert evaluator.eval_dataset.metadata == [{"label": COMPARISON_HELDOUT_LABEL}]
    assert evaluator.eval_on_finish is True
    assert evaluator.eval_duration == evaluator.eval_duration.steps(2)
    assert config.data_loader.global_batch_size == 32768
    assert config.init_seed == 110007
    assert config.train_module.compile_model is False
    assert COMPARISON_HELDOUT_METRIC == (f"eval/lm/{COMPARISON_HELDOUT_LABEL}/CE loss")


def test_comparison_specs_match_aggregate_budget_and_platform_contract():
    root = _repo_root()
    baseline = yaml.safe_load((root / ".edullm/run-hpo-comparison-baseline.yaml").read_text())
    hybrid = yaml.safe_load((root / ".edullm/run-hpo-comparison-hybrid.yaml").read_text())
    spec = json.loads((root / ".edullm/hpo-comparison-hybrid.json").read_text())

    assert baseline["suggested_compute"] == "gpu-1xa10g"
    assert hybrid["suggested_compute"] == "gpu-4xa10g"
    assert "--param-dtype bfloat16" in baseline["command"]
    assert "--param-dtype bfloat16" in hybrid["command"]
    assert "$EDULLM_CHECKPOINT_DIR" in baseline["command"]
    assert "$EDULLM_CHECKPOINT_DIR" in hybrid["command"]
    assert "EDULLM_LAUNCH_CHECK=waived" in hybrid["command"]
    assert spec["controller"]["worker_count"] == 4
    assert spec["controller"]["target_tokens"] == 327_680
    assert spec["controller"]["quantum"] == 163_840
    assert spec["controller"]["budget_tokens"] == 1_146_880
    assert spec["controller"]["budget_tokens"] == 7 * spec["controller"]["quantum"]
    assert "--target-tokens 1146880" in baseline["command"]
    assert spec["btt"]["min_fidelity"] <= spec["controller"]["quantum"]
    assert spec["ipbt"]["update_interval_init"] <= spec["controller"]["quantum"]
    assert spec["require_final_winner"] is True
    assert spec["normalizer"]["ce_at_zero"] >= 16.0
    assert spec["heldout_metric"] == COMPARISON_HELDOUT_METRIC
    assert spec["fixed_hps"] == {
        key: value
        for key, value in DEFAULT_RECIPE_HPS.items()
        if key not in {"lr", "weight_decay", "max_grad_norm"}
    }
    assert "s3://" not in json.dumps(spec)
    assert spec["experiment_factory"].endswith(":build_comparison_experiment")


def test_runbook_has_non_dispatching_checks_and_pinned_dataset_contract():
    text = (_repo_root() / "HPO_COMPARISON_RUNS.md").read_text()
    assert text.count("edullm check --json") >= 2
    assert "edullm submit" in text
    assert text.count("--hours 1") >= 4
    assert "reservoir-dolma2-v1" in text
    assert "38bf831a6c3f445e394784018441fd59288b876c" in text
    assert "pretrain-tokens/v1" in text
    assert "functional smoke" in text.lower()
