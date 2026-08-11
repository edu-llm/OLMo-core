import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

# bettermap currently imports a Python <=3.13-only multiprocessing symbol on Windows.
if sys.version_info >= (3, 14):
    sys.modules.setdefault(
        "bettermap",
        SimpleNamespace(
            ordered_map_per_thread=lambda function, values, **kwargs: map(function, values)
        ),
    )

from olmo_core.hpo.comparison import (
    COMPARISON_HELDOUT_LABEL,
    COMPARISON_HELDOUT_METRIC,
    DEFAULT_RECIPE_HPS,
    build_comparison_experiment,
    build_olmoe_hpo_experiment,
    build_umup_hpo_experiment,
    comparison_dataset_from_read,
    comparison_heldout_label,
    comparison_heldout_metric,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_comparison_heldout_helpers_follow_dataset_id():
    assert comparison_heldout_label("pretrain/regmix-10b") == "regmix-10b-val"
    assert comparison_heldout_metric("pretrain/regmix-10b") == ("eval/lm/regmix-10b-val/CE loss")


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
        dataset_id="pretrain/regmix-10b",
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
        dataset_id="pretrain/regmix-10b",
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
            dataset_id="pretrain/regmix-10b",
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
    data_buckets = []

    def dataset_paths(dataset_id, version, *, s3, data_bucket=None):
        del dataset_id, version, s3
        data_buckets.append(data_bucket)
        return read

    fake_read.dataset_paths = dataset_paths

    class Boto3S3:
        @classmethod
        def default(cls):
            return object()

    fake_s3.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", fake_package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", fake_read)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", fake_s3)
    monkeypatch.setenv("EDULLM_DATASET_ID", "pretrain/regmix-10b")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "/tmp/checkpoints")

    config = build_comparison_experiment(
        sequence_length=2048,
        global_batch_size=32768,
        rank_microbatch_size=4096,
        eval_steps=2,
        data_bucket="edullm-data-us-east-2",
    )
    assert data_buckets == ["edullm-data-us-east-2"]
    assert config.dataset.paths == read.paths
    evaluator = config.trainer.callbacks["search_validation"]
    assert evaluator.eval_dataset.paths == read.val
    assert evaluator.eval_dataset.metadata == [{"label": COMPARISON_HELDOUT_LABEL}]
    assert evaluator.eval_on_finish is True
    assert evaluator.eval_duration == evaluator.eval_duration.steps(2)
    assert config.data_loader.global_batch_size == 32768
    assert config.init_seed == 110007
    assert config.train_module.compile_model is False
    assert config.model.n_layers == 12
    assert config.umup_backend is None
    assert COMPARISON_HELDOUT_METRIC == (f"eval/lm/{COMPARISON_HELDOUT_LABEL}/CE loss")

    umup_config = build_umup_hpo_experiment(
        sequence_length=2048,
        global_batch_size=32768,
        rank_microbatch_size=4096,
        eval_steps=2,
    )
    assert umup_config.model.n_layers == 16
    assert umup_config.umup_backend == "unit-scaling"
    assert umup_config.umup_parity_validated is True
    assert umup_config.umup_metadata["proxy_depth"] == 16
    assert umup_config.train_module.optim.__class__.__name__ == "UMuPAdamWConfig"


def test_olmoe_factory_is_separate_and_builds_mesh_from_launched_world(monkeypatch):
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
    monkeypatch.setenv("EDULLM_DATASET_ID", "pretrain/regmix-10b")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "/tmp/checkpoints")

    dense = build_comparison_experiment()
    olmoe = build_olmoe_hpo_experiment()

    assert dense.model.n_layers == 12
    assert dense.model.block.feed_forward_moe is None
    assert dense.data_loader.global_batch_size == 32_768
    assert dense.train_module.rank_microbatch_size == 4_096
    assert str(dense.train_module.dp_config.name) == "fsdp"
    assert dense.train_module.ep_config is None

    moe = olmoe.model.block.feed_forward_moe
    assert moe is not None
    assert olmoe.model.n_layers == 16
    assert moe.num_experts == 64
    assert moe.router.top_k == 8
    assert moe.hidden_size == 1_024
    assert olmoe.data_loader.global_batch_size == 262_144
    assert olmoe.dataset.sequence_length == 2_048
    assert olmoe.train_module.rank_microbatch_size == 32_768
    assert olmoe.train_module.max_sequence_length == 2_048
    assert 262_144 // (8 * olmoe.train_module.rank_microbatch_size) == 1
    assert olmoe.train_module.compile_model is True
    assert str(olmoe.train_module.dp_config.name) == "hsdp"
    assert olmoe.train_module.dp_config.num_replicas == 1
    assert olmoe.train_module.dp_config.get_replicate_and_shard_degree(8) == (1, 8)
    assert str(olmoe.train_module.dp_config.param_dtype) == "bfloat16"
    assert str(olmoe.train_module.dp_config.reduce_dtype) == "float32"
    assert olmoe.train_module.ep_config.degree == -1
    assert olmoe.train_module.optim.__class__.__name__ == "SkipStepAdamWConfig"
    assert olmoe.train_module.optim.lr == 4e-4
    assert olmoe.train_module.scheduler.warmup == 24
    assert olmoe.train_module.scheduler.alpha_f == 0.1
    assert olmoe.train_module.z_loss_multiplier == 1e-5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"global_batch_size": 524_288},
        {"rank_microbatch_size": 16_384},
        {"sequence_length": 4_096},
    ],
)
def test_olmoe_factory_rejects_changes_to_fixed_batch_contract(monkeypatch, kwargs):
    monkeypatch.setenv("EDULLM_DATASET_ID", "unused")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "unused")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "unused")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "unused")

    with pytest.raises(ValueError, match="fixed"):
        build_olmoe_hpo_experiment(**kwargs)


def test_olmoe_probe_specs_are_separate_fixed_batch_capacity_block_arms():
    root = _repo_root() / ".edullm"
    no_proxy = json.loads((root / "hpo-olmoe-no-proxy.json").read_text())
    no_centaur = json.loads((root / "hpo-olmoe-no-centaur.json").read_text())
    expected_dimensions = {
        "lr",
        "weight_decay",
        "beta2_gap",
        "eps",
        "warmup_fraction",
        "decay_fraction",
        "terminal_lr_ratio",
        "max_grad_norm",
    }

    for arm, spec in (("olmoe_no_proxy", no_proxy), ("olmoe_no_centaur", no_centaur)):
        assert spec["arm"] == arm
        assert spec["algorithm"] == "brainlift"
        assert {dimension["name"] for dimension in spec["search_space"]} == expected_dimensions
        assert "global_batch_mult" not in json.dumps(spec["search_space"])
        assert spec["base_global_batch_size"] == 262_144
        assert spec["factory_kwargs"]["global_batch_size"] == 262_144
        assert spec["factory_kwargs"]["rank_microbatch_size"] == 32_768
        assert spec["factory_kwargs"]["sequence_length"] == 2_048
        assert spec["experiment_factory"].endswith(":build_olmoe_hpo_experiment")
        assert spec["model_parameterization"] == {
            "kind": "standard",
            "architecture": "olmoe_1B_7B",
            "depth": 16,
            "backend": "none",
        }
        assert spec["fidelity"] == {"kind": "exact"}
        assert spec["launch_backend"] == "capacity_block"
        assert spec["capacity_block"]["branch"] == "edullm/hpo-complex"
        assert spec["capacity_block"]["repository"] == "edu-llm/OLMo-core"
        assert spec["worker_world_size"] == 8
        assert spec["max_workers"] == 6
        assert spec["capacity_block"]["branch"] == "edullm/hpo-complex"
        assert spec["capacity_block"]["repository"] == "edu-llm/OLMo-core"
        assert spec["controller"]["worker_count"] == 6
        assert spec["controller"]["quantum"] == 49_807_360
        assert spec["controller"]["target_tokens"] == 499_908_608
        assert spec["controller"]["budget_tokens"] == 2_000_158_720
        assert spec["btt"]["min_fidelity"] == spec["controller"]["quantum"]
        assert spec["ipbt"]["update_interval_init"] == spec["controller"]["quantum"]
        assert spec["controller"]["quantum"] % 262_144 == 0
        assert spec["controller"]["target_tokens"] % 262_144 == 0
        assert spec["controller"]["budget_tokens"] % 262_144 == 0
        assert "proxy_evidence_contract" not in spec
        assert "proxy_admission" not in spec

    assert no_proxy["centaur"]["ratio"] == 0.3
    assert no_centaur["centaur"] is None


def test_olmoe_curriculum_no_proxy_spec_combines_required_contracts():
    spec = json.loads((_repo_root() / ".edullm/hpo-olmoe-curriculum-no-proxy.json").read_text())

    assert spec["arm"] == "olmoe_curriculum_no_proxy"
    assert spec["centaur"]["ratio"] == 0.3
    assert spec["centaur"]["scope"] == "multi_action"
    assert spec["experiment_factory"].endswith(":build_olmoe_curriculum_hpo_experiment")
    assert spec["curriculum_identity"]["pacing"] == "arm9_warmup_quadratic_n10_token_fraction_v1"
    assert spec["model_parameterization"]["architecture"] == "olmoe_1B_7B"
    assert "global_batch_mult" not in json.dumps(spec["search_space"])
    assert spec["base_global_batch_size"] == 262_144
    assert spec["factory_kwargs"]["rank_microbatch_size"] == 32_768
    assert spec["factory_kwargs"]["data_bucket"] == "edullm-data-us-east-2"
    assert spec["controller"]["worker_count"] == spec["max_workers"] == 6
    assert spec["controller"]["quantum"] == 50_331_648
    assert spec["controller"]["target_tokens"] == 503_316_480
    assert spec["controller"]["budget_tokens"] == 2_013_265_920
    assert spec["worker_environment"] == {
        "EDULLM_DATASET_ID": "pretrain/opt-with-synthetic-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
    }


def test_dense_probe_specs_remain_byte_for_byte_intact():
    root = _repo_root() / ".edullm"
    expected_sha256 = {
        "hpo-no-proxy.json": "39b6dfe87768436a45b256dd27417d964dca91e46f21d90755a876fd2bcb5225",
        "hpo-no-centaur.json": "34f47bdd806f48948b48c3e2e1d3d972192d3a06287bad5a052cc53d972283b9",
    }

    for name, expected in expected_sha256.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_olmoe_runbook_pins_parallel_node_dispatch_and_launch_safety():
    text = (_repo_root() / ".edullm/OLMOE_PARALLEL_HPO.md").read_text()

    assert "one `block-run.yml` dispatch per IDLE node" in text
    assert "`block-run-distributed.yml`" in text
    assert "`processes=all` prepends `torchrun`" in text
    assert "No launcher in `command`" in text
    assert "`--moe-shard-degree`" in text
    assert "`--moe-num-replicas`" in text
    assert "`mesh_flags=false`" in text
    assert "explicit `--flag=value`" in text
    assert "dry-run" in text
    assert "refuse-busy" in text
    assert "262,144" in text
    assert "32,768" in text
    assert "2,048" in text


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
    assert "regmix-10b-v1" in text
    assert "38bf831a6c3f445e394784018441fd59288b876c" in text
    assert "pretrain-tokens/v1" in text
    assert "functional smoke" in text.lower()
