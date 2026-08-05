"""Build the complete platform config without AWS or a GPU.

This catches controls that are present in YAML but silently dropped while adapting
`.edullm/train_on_corpus.py`. The resulting config is exactly what `--dry-run`
prints; model construction and actual bytes are covered elsewhere.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "src" / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("p3_train_platform", SCRIPTS / "train_platform.py")
assert spec and spec.loader
platform = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = platform
spec.loader.exec_module(platform)

from provenance import (  # noqa: E402
    TOKENIZER_ARTIFACT,
    TOKENIZER_ARTIFACT_ID,
    TOKENIZER_ARTIFACT_VERSION,
    TOKENIZER_COMPOSITE_SHA256,
    TOKENIZER_FILE_SHA256,
    TOKENIZERS_VERSION,
)

from olmo_core.data import NumpyDatasetDType  # noqa: E402
from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.lm_head import LMLossImplementation  # noqa: E402
from olmo_core.nn.transformer.qwen import (  # noqa: E402
    QWEN2_0_5B_HF_ID,
    QWEN2_0_5B_HF_REVISION,
    QWEN2_0_5B_HF_WEIGHTS_SHA256,
    QWEN2_0_5B_HF_WEIGHTS_SIZE,
    qwen2_0_5b_config,
    qwen2_tokenizer_config,
)
from olmo_core.train import DurationUnit, Trainer  # noqa: E402

FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
V3_TRAIN_SHARD_SEQUENCE_COUNTS = {
    (
        "s3://edullm-data/pretrain/formal-proof-premises-500m/v3/"
        f"tokens/{family}/train-{shard:05d}.u32le.bin"
    ): 2_600
    + family_index * 10
    + shard
    for family_index, family in enumerate(FAMILIES)
    for shard in range(2)
}
V3_TRAIN_PATHS = list(V3_TRAIN_SHARD_SEQUENCE_COUNTS)
V3_AGGREGATE_ROWS = sum(V3_TRAIN_SHARD_SEQUENCE_COUNTS.values()) * 16_384
V3_BATCHES_PER_EPOCH = V3_AGGREGATE_ROWS // (16 * 16_384)
V3_STEPS = V3_BATCHES_PER_EPOCH * 13


@pytest.fixture
def repaired_v3_read():
    return SimpleNamespace(
        paths=V3_TRAIN_PATHS,
        dtype="uint32",
        byte_order=sys.byteorder,
        header_bytes=0,
        rows=V3_AGGREGATE_ROWS,
    )


@pytest.fixture
def built(monkeypatch, repaired_v3_read):
    corpus = platform.corpus_from_manifest(
        repaired_v3_read,
        dataset_id="pretrain/formal-proof-premises-500m",
        version="v3",
        tokenizer_id="tokenizer/qwen25-vendored/v1",
    )
    monkeypatch.setattr(platform, "resolve_corpus", lambda **_: corpus)

    def sealed_olmo_config():
        config = qwen2_tokenizer_config()
        config.identifier = TOKENIZER_ARTIFACT
        return config

    sealed = SimpleNamespace(
        artifact_id=TOKENIZER_ARTIFACT_ID,
        artifact_version=TOKENIZER_ARTIFACT_VERSION,
        file_sha256=TOKENIZER_FILE_SHA256,
        composite_sha256=TOKENIZER_COMPOSITE_SHA256,
        tokenizers_version=TOKENIZERS_VERSION,
        eos_token_id=151_643,
        pad_token_id=151_643,
        separator_ids=lambda _text: [10952, 15513, 969],
        olmo_config=sealed_olmo_config,
        provenance_dict=lambda: {
            "tokenizer_artifact_id": TOKENIZER_ARTIFACT_ID,
            "tokenizer_artifact_version": TOKENIZER_ARTIFACT_VERSION,
            "tokenizer_file_sha256": TOKENIZER_FILE_SHA256,
            "tokenizer_composite_sha256": TOKENIZER_COMPOSITE_SHA256,
            "tokenizers_version": TOKENIZERS_VERSION,
            "tokenizer_eos_token_id": 151_643,
            "tokenizer_pad_token_id": 151_643,
        },
    )
    monkeypatch.setattr(platform, "fetch_tokenizer_artifact", lambda *_args, **_kwargs: sealed)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("EDULLM_DATASET_RELEASE", "formal-proof-premises-500m-v3")
    monkeypatch.setenv("EDULLM_COMMIT_SHA", "a" * 40)
    args = platform.build_parser().parse_args(
        [
            "test-run",
            "--arm",
            "split",
            "--config",
            str(SCRIPTS / "configs" / "split.yaml"),
            "--dataset-id",
            corpus.dataset_id,
            "--dataset-version",
            corpus.version,
            "--dataset-tokenizer",
            "tokenizer/qwen25-vendored/v1",
            "--save-folder",
            "s3://checkpoints/test-run/",
        ]
    )
    return platform.build_config(args, []), args


@pytest.fixture
def stale_v2_invocation(built):
    cfg, base = built
    args = copy.copy(base)
    args.dataset_version = "v2"
    cfg = copy.copy(cfg)
    cfg.dataset_version = "v2"
    cfg.dataset_release = "formal-proof-premises-500m-v2"
    return cfg, args


def test_tokenize_corpus_ast_import_and_cli_startup():
    producer = SCRIPTS / "tokenize_corpus.py"
    source = producer.read_text(encoding="utf-8")
    ast.parse(source, filename=str(producer))

    module_spec = importlib.util.spec_from_file_location("p3_tokenize_corpus_startup", producer)
    assert module_spec and module_spec.loader
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.FAMILIES == FAMILIES

    startup = subprocess.run(
        [sys.executable, str(producer), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert startup.returncode == 0, startup.stderr
    assert "--corpus-contract-root" in startup.stdout
    assert "--test-only-corpus-dir" in startup.stdout


def test_repaired_v3_fixture_consumes_every_multi_shard_path_and_aggregate_count(built):
    cfg, args = built

    assert cfg.dataset_version == "v3"
    assert cfg.dataset.paths == V3_TRAIN_PATHS
    assert set(cfg.dataset.paths) == set(V3_TRAIN_SHARD_SEQUENCE_COUNTS)
    assert all(
        sum(f"/tokens/{family}/" in path for path in cfg.dataset.paths) > 1 for family in FAMILIES
    )
    assert V3_AGGREGATE_ROWS == sum(V3_TRAIN_SHARD_SEQUENCE_COUNTS.values()) * 16_384
    assert args.steps == V3_STEPS


def test_reader_requests_train_and_preserves_every_v3_manifest_path(monkeypatch, repaired_v3_read):
    calls = []
    s3 = object()

    def dataset_paths(dataset_id, version, *, split, s3):
        calls.append((dataset_id, version, split, s3))
        return repaired_v3_read

    monkeypatch.setitem(
        sys.modules,
        "edullm_data.read",
        SimpleNamespace(dataset_paths=dataset_paths, resolve_latest=lambda *_args, **_kwargs: "v3"),
    )
    monkeypatch.setitem(
        sys.modules,
        "edullm_data.s3",
        SimpleNamespace(Boto3S3=SimpleNamespace(default=lambda: s3)),
    )

    corpus = platform.resolve_corpus(
        dataset_id="pretrain/formal-proof-premises-500m",
        version="v3",
        tokenizer_id="tokenizer/qwen25-vendored/v1",
    )

    assert calls == [("pretrain/formal-proof-premises-500m", "v3", "train", s3)]
    assert corpus.paths == V3_TRAIN_PATHS
    assert corpus.rows == sum(V3_TRAIN_SHARD_SEQUENCE_COUNTS.values()) * 16_384


def test_publish_command_documents_profile_sequence_length_contract():
    readme = (SCRIPTS / "README.md").read_text(encoding="utf-8")

    assert 'group_meta={"tokens": {"seq_len": 16384}},' in readme


def test_dataset_and_loader_controls(built):
    cfg, _ = built
    assert cfg.dataset.paths == V3_TRAIN_PATHS
    assert cfg.dataset.dtype == NumpyDatasetDType.uint32
    assert cfg.dataset.sequence_length == 16_384
    assert cfg.dataset.generate_doc_lengths is True
    assert cfg.data_loader.global_batch_size == 262_144
    assert cfg.data_loader.seed == 42
    assert cfg.data_loader.num_workers == 2
    assert cfg.dataset.tokenizer.identifier == TOKENIZER_ARTIFACT


def test_model_and_mask_controls(built):
    cfg, _ = built
    default_qwen = qwen2_0_5b_config()
    assert default_qwen.lm_head is not None
    assert default_qwen.lm_head.loss_implementation == LMLossImplementation.default
    assert cfg.model.vocab_size == 151_936
    assert cfg.model.init_seed == 42
    assert cfg.model.tie_word_embeddings is True
    assert cfg.model.block.sequence_mixer.backend == AttentionBackendName.flash_2
    assert cfg.model.lm_head is not None
    assert cfg.model.lm_head.loss_implementation == LMLossImplementation.fused_linear
    assert cfg.train_module.arm == "split"
    assert cfg.train_module.separator_ids == [10952, 15513, 969]
    assert cfg.train_module.eos_token_id == 151_643
    assert cfg.train_module.pad_token_id == 151_643
    assert cfg.train_module.fixed_loss_div_factor == 262_144.0


def test_p3_refuses_to_fall_back_to_materialized_logits(built, tmp_path):
    _, base = built
    args = copy.copy(base)
    explicit = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    explicit["shared"]["loss_implementation"] = "default"
    config_path = tmp_path / "split.yaml"
    config_path.write_text(yaml.safe_dump(explicit), encoding="utf-8")
    args.config = str(config_path)

    with pytest.raises(platform.Refusal, match="fixed to 'fused_linear'"):
        platform.apply_arm_config(args)


def test_optimizer_schedule_and_checkpoint_controls(built):
    cfg, args = built
    assert cfg.train_module.rank_microbatch_size == 16_384
    assert cfg.train_module.optim.lr == 2e-5
    assert cfg.train_module.optim.betas == (0.9, 0.95)
    assert cfg.train_module.optim.eps == 1e-8
    assert cfg.train_module.optim.weight_decay == 0.0
    assert cfg.train_module.max_grad_norm == 1.0
    assert cfg.train_module.scheduler.warmup == 2_400
    assert cfg.train_module.scheduler.alpha_f == 0.1
    assert args.steps == V3_STEPS
    assert cfg.trainer.max_duration.unit == DurationUnit.epochs
    assert cfg.trainer.max_duration.value == 13
    assert cfg.trainer.callbacks["checkpointer"].save_interval == 2_000
    assert cfg.trainer.callbacks["checkpointer"].ephemeral_save_interval is None
    assert cfg.trainer.callbacks["checkpointer"].max_checkpoints is None
    assert cfg.trainer.save_folder == "s3://checkpoints/test-run/"


def test_checkpoint_config_persists_model_tokenizer_and_launch_provenance(built):
    cfg, _ = built
    assert cfg.arm == "split"
    assert cfg.model_factory == "qwen2_0_5b"
    assert cfg.loss_implementation == "fused_linear"
    assert cfg.base_model_id == QWEN2_0_5B_HF_ID
    assert cfg.base_model_revision == QWEN2_0_5B_HF_REVISION
    assert cfg.base_model_weight_sha256 == QWEN2_0_5B_HF_WEIGHTS_SHA256
    assert cfg.base_model_weight_size == QWEN2_0_5B_HF_WEIGHTS_SIZE
    assert cfg.tokenizer_artifact_id == TOKENIZER_ARTIFACT_ID
    assert cfg.tokenizer_artifact_version == TOKENIZER_ARTIFACT_VERSION
    assert cfg.tokenizer_file_sha256 == TOKENIZER_FILE_SHA256
    assert cfg.tokenizer_composite_sha256 == TOKENIZER_COMPOSITE_SHA256
    assert cfg.tokenizers_version == TOKENIZERS_VERSION
    assert cfg.tokenizer_eos_token_id == cfg.tokenizer_pad_token_id == 151_643
    assert cfg.dataset_id == "pretrain/formal-proof-premises-500m"
    assert cfg.dataset_version == "v3"
    assert cfg.dataset_release == "formal-proof-premises-500m-v3"
    assert cfg.world_size == 1
    assert cfg.launch_contract["supported_compute_profiles"] == [
        "gpu-8xa100",
        "gpu-8xh100",
    ]
    assert cfg.launch_contract["recommended_compute_profile"] == "gpu-8xh100"
    assert cfg.launch_contract["final_world_size"] == 8
    assert cfg.launch_contract["config_preflight_compute_profile"] == "gpu-1xa10g"
    assert cfg.source_commit == "a" * 40


def test_non_dry_config_requires_source_commit_but_dry_run_reports_unavailable(built, monkeypatch):
    _, base = built
    monkeypatch.delenv("EDULLM_COMMIT_SHA")

    non_dry = copy.copy(base)
    non_dry.dry_run = False
    with pytest.raises(platform.Refusal, match="EDULLM_COMMIT_SHA"):
        platform.build_config(non_dry, [])

    dry_run = copy.copy(base)
    dry_run.dry_run = True
    config = platform.build_config(dry_run, [])
    assert config.run_mode == "dry-run"
    assert config.source_commit == ""


def test_platform_manifest_identity_is_saved_when_exposed(built, monkeypatch):
    _, base = built
    monkeypatch.setenv("EDULLM_RUN_MANIFEST_ID", "manifest-dense-123")
    monkeypatch.setenv("EDULLM_RUN_MANIFEST_SHA256", "d" * 64)

    config = platform.build_config(copy.copy(base), [])

    assert config.platform_run_manifest_id == "manifest-dense-123"
    assert config.platform_run_manifest_sha256 == "d" * 64


def test_platform_manifest_hash_must_be_a_bound_sha256(built, monkeypatch):
    _, base = built
    monkeypatch.setenv("EDULLM_RUN_MANIFEST_ID", "manifest-dense-123")
    monkeypatch.setenv("EDULLM_RUN_MANIFEST_SHA256", "not-a-digest")

    with pytest.raises(platform.Refusal, match="manifest.*SHA-256"):
        platform.build_config(copy.copy(base), [])


def test_dense_and_split_configs_record_identical_downloaded_tokenizer_seal(built):
    split_config, base = built
    dense_args = platform.parse_cli_args(
        [
            "test-dense",
            "--arm",
            "dense",
            "--config",
            str(SCRIPTS / "configs" / "dense.yaml"),
            "--dataset-id",
            base.dataset_id,
            "--dataset-version",
            base.dataset_version,
            "--dataset-tokenizer",
            base.dataset_tokenizer,
            "--save-folder",
            base.save_folder,
        ]
    )
    dense_config = platform.build_config(dense_args, [])

    for field in (
        "tokenizer_artifact_id",
        "tokenizer_artifact_version",
        "tokenizer_file_sha256",
        "tokenizer_composite_sha256",
        "tokenizers_version",
        "tokenizer_eos_token_id",
        "tokenizer_pad_token_id",
    ):
        assert getattr(dense_config, field) == getattr(split_config, field)


def test_wandb_comes_from_platform_form_not_local_yaml(built, monkeypatch):
    _, args = built
    assert platform.wandb_project(args) is None
    monkeypatch.setenv("WANDB_PROJECT", "p3-math")
    assert platform.wandb_project(args) == "p3-math"


def test_crash_wandb_record_is_rank_zero_before_init_and_after_teardown(monkeypatch):
    initialized = []

    class Run:
        def __init__(self):
            self.summary = {}
            self.finished = []

        def finish(self, *, exit_code):
            self.finished.append(exit_code)

    wandb = SimpleNamespace(run=None)

    def init(**kwargs):
        initialized.append(kwargs)
        wandb.run = Run()
        return wandb.run

    wandb.init = init
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setenv("WANDB_PROJECT", "p3")
    monkeypatch.setattr(
        platform,
        "get_rank",
        lambda: (_ for _ in ()).throw(AssertionError("env rank must work without a process group")),
    )

    for rank in range(7, -1, -1):
        monkeypatch.setenv("RANK", str(rank))
        platform.leave_the_reason_in_wandb(
            run_name="fake-8-rank",
            stage=platform.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            explanation="fixture",
        )

    assert len(initialized) == 1
    assert wandb.run.finished == [int(platform.Stage.THE_CONFIG_WOULD_NOT_BUILD)]


def test_dotlist_unknown_and_scientific_flag_overrides_are_rejected(built):
    _, args = built
    with pytest.raises(platform.Refusal, match="override"):
        platform.build_config(args, ["trainer.max_duration.value=100"])

    base = [
        "test",
        "--arm",
        "split",
        "--config",
        str(SCRIPTS / "configs" / "split.yaml"),
    ]
    for override in (
        ["trainer.max_duration.value=100"],
        ["--learning-rate", "9e-4"],
        ["--unknown-setting", "value"],
    ):
        with pytest.raises(platform.Refusal, match="override"):
            platform.parse_cli_args(base + override)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_factory", "qwen3_0_6b", "model factory"),
        ("dataset_id", "pretrain/other", "dataset"),
        ("dataset_version", "latest", "version"),
        ("dataset_tokenizer", "tokenizer/qwen25-vendored/v2", "tokenizer"),
        ("data_seed", 7, "seed"),
    ],
)
def test_fixed_submission_controls_refuse_drift(built, field, value, message):
    _, args = built
    setattr(args, field, value)
    with pytest.raises(platform.Refusal, match=message):
        platform.validate_submission_controls(args)


def test_arm_must_use_its_canonical_config(built, tmp_path):
    _, args = built
    copied = tmp_path / "split.yaml"
    copied.write_text(Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")
    args.config = str(copied)
    with pytest.raises(platform.Refusal, match="canonical"):
        platform.validate_submission_controls(args)


def test_no_cost_a10g_config_preflight_accepts_one_process(built):
    cfg, _ = built
    assert cfg.world_size == 1
    assert cfg.launch_contract["final_world_size"] == 8


@pytest.mark.parametrize(
    ("dry_run", "dataset_version", "rank", "expects_warning"),
    [
        pytest.param(True, "v2", 0, True, id="dry-v2"),
        pytest.param(False, "v2", 0, True, id="train-v2"),
        pytest.param(False, "v2", 1, False, id="train-v2-nonzero-rank"),
        pytest.param(True, "v3", 0, False, id="dry-v3"),
        pytest.param(False, "v3", 0, False, id="train-v3"),
        pytest.param(True, "v12", 0, False, id="dry-v12"),
        pytest.param(False, "v12", 0, False, id="train-v12"),
    ],
)
def test_stale_v2_warning_is_warning_only_for_dry_and_non_dry_invocations(
    stale_v2_invocation,
    monkeypatch,
    caplog,
    dry_run,
    dataset_version,
    rank,
    expects_warning,
):
    cfg, base = stale_v2_invocation
    args = copy.copy(base)
    args.dry_run = dry_run
    args.dataset_version = dataset_version
    cfg = copy.copy(cfg)
    cfg.dataset_version = dataset_version
    events = []

    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(platform, "parse_cli_args", lambda: args)
    monkeypatch.setattr(platform, "build_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(platform, "show", lambda _config: events.append("show"))
    monkeypatch.setattr(platform, "prepare_training_environment", lambda: events.append("prepare"))
    monkeypatch.setattr(platform, "train", lambda *_args, **_kwargs: events.append("train"))
    monkeypatch.setattr(
        platform, "teardown_training_environment", lambda: events.append("teardown")
    )

    with caplog.at_level(logging.WARNING, logger=platform.log.name):
        platform.main()

    warnings = [
        record
        for record in caplog.records
        if record.name == platform.log.name and "SCIENTIFIC WARNING" in record.getMessage()
    ]
    assert len(warnings) == int(expects_warning)
    if expects_warning:
        message = warnings[0].getMessage()
        assert "scientifically stale" in message
        assert "forbidden for final conclusions" in message
        assert "warning-only" in message
    assert events == (["show"] if dry_run else ["prepare", "train", "teardown"])


def test_one_rank_dry_run_reaches_show_without_training_setup(built, monkeypatch):
    cfg, base = built
    args = copy.copy(base)
    args.dry_run = True
    events = []
    monkeypatch.setattr(platform, "parse_cli_args", lambda: args)
    monkeypatch.setattr(platform, "build_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(platform, "show", lambda _config: events.append("show"))
    monkeypatch.setattr(
        platform,
        "prepare_training_environment",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run must not initialize training")),
    )

    platform.main()

    assert cfg.world_size == 1
    assert events == ["show"]


@pytest.mark.parametrize("runtime_smoke", [False, True], ids=["final-train", "runtime-smoke"])
def test_one_rank_non_dry_invocations_refuse_before_training_setup(
    built, monkeypatch, runtime_smoke
):
    cfg, base = built
    args = copy.copy(base)
    args.dry_run = False
    args.runtime_smoke = runtime_smoke
    setup_called = False

    def prepare():
        nonlocal setup_called
        setup_called = True

    monkeypatch.setattr(platform, "parse_cli_args", lambda: args)
    monkeypatch.setattr(platform, "build_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(platform, "prepare_training_environment", prepare)

    with pytest.raises(platform.Refusal, match=r"WORLD_SIZE.*8"):
        platform.main()

    assert not setup_called


def test_eight_rank_single_node_contract_is_accepted_and_recorded(built, monkeypatch):
    _, base = built
    args = copy.copy(base)
    args.dry_run = False
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    cfg = platform.build_config(args, [])
    events = []
    monkeypatch.setattr(platform, "parse_cli_args", lambda: args)
    monkeypatch.setattr(platform, "build_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(platform, "prepare_training_environment", lambda: events.append("prepare"))
    monkeypatch.setattr(platform, "train", lambda *_args, **_kwargs: events.append("train"))
    monkeypatch.setattr(
        platform,
        "teardown_training_environment",
        lambda: events.append("teardown"),
    )

    platform.main()

    assert cfg.world_size == 8
    assert events == ["prepare", "train", "teardown"]


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("LOCAL_WORLD_SIZE", "4", "LOCAL_WORLD_SIZE"),
        ("RANK", "8", "RANK"),
        ("LOCAL_RANK", "8", "LOCAL_RANK"),
    ],
)
def test_eight_rank_launch_refuses_incompatible_process_contract(
    built, monkeypatch, name, value, message
):
    _, args = built
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv(name, value)

    with pytest.raises(platform.Refusal, match=message):
        platform.validate_runtime_launch_contract()


def test_closed_runtime_smoke_mode_uses_yaml_declared_100_step_profile(built):
    _, base = built
    smoke = platform.parse_cli_args(
        [
            "test-smoke",
            "--arm",
            "split",
            "--config",
            base.config,
            "--dataset-id",
            base.dataset_id,
            "--dataset-version",
            base.dataset_version,
            "--dataset-tokenizer",
            base.dataset_tokenizer,
            "--save-folder",
            base.save_folder,
            "--runtime-smoke",
        ]
    )

    cfg = platform.build_config(smoke, [])

    assert cfg.run_mode == "runtime-smoke"
    assert cfg.trainer.max_duration.unit == DurationUnit.steps
    assert cfg.trainer.max_duration.value == 100
    assert cfg.train_module.scheduler.warmup == 10
    assert cfg.trainer.callbacks["checkpointer"].save_interval == 50


def test_versioned_published_tokenizer_resolves_to_local_config():
    """The platform pins dependencies as tokenizer/<name>/vN."""
    read = SimpleNamespace(
        paths=["s3://bucket/tokens/x/train-00000.u32le.bin"],
        dtype="uint32",
        byte_order=sys.byteorder,
        header_bytes=0,
        rows=16_384,
    )
    corpus = platform.corpus_from_manifest(
        read,
        dataset_id="pretrain/formal-proof-premises-500m",
        version="v3",
        tokenizer_id="tokenizer/qwen25-vendored/v1",
    )
    assert corpus.tokenizer.vocab_size == 151_936


@pytest.mark.parametrize(
    ("rows", "global_batch", "epochs", "expected_batches", "expected_steps"),
    [
        pytest.param(
            V3_AGGREGATE_ROWS,
            16 * 16_384,
            13,
            V3_BATCHES_PER_EPOCH,
            V3_STEPS,
            id="v3-aggregate-manifest-count",
        ),
        pytest.param(10, 4, 3, 2, 6, id="non-divisible-small"),
        pytest.param(12, 4, 3, 3, 9, id="exact-divisible-small"),
    ],
)
def test_loader_epoch_step_counts_drop_the_tail_each_epoch(
    rows, global_batch, epochs, expected_batches, expected_steps
):
    assert platform.loader_epoch_step_counts(rows, global_batch, epochs) == (
        expected_batches,
        expected_steps,
    )


def test_trainer_and_scheduler_resolve_the_v3_epoch_horizon(built, monkeypatch):
    cfg, args = built
    trainer = Trainer.__new__(Trainer)
    trainer.max_duration = cfg.trainer.max_duration
    trainer.data_loader = SimpleNamespace(
        total_batches=V3_BATCHES_PER_EPOCH,
        batches_processed=0,
        batches_in_epoch=lambda _epoch: V3_BATCHES_PER_EPOCH,
    )
    trainer.epoch = 1
    trainer.global_step = 0
    trainer.global_train_tokens_seen = 0

    assert trainer.max_steps == args.steps == V3_STEPS

    seen = {}
    scheduler = cfg.train_module.scheduler

    def capture_t_max(initial_lr, current, t_max):
        seen["t_max"] = t_max
        return initial_lr

    monkeypatch.setattr(scheduler, "get_lr", capture_t_max)
    scheduler.set_lr({"lr": 2e-5, "params": []}, trainer)
    assert seen["t_max"] == V3_STEPS


def test_epoch_horizon_validation_rejects_loader_manifest_drift(built):
    cfg, args = built
    trainer = Trainer.__new__(Trainer)
    trainer.max_duration = cfg.trainer.max_duration
    trainer.data_loader = SimpleNamespace(
        total_batches=V3_BATCHES_PER_EPOCH - 1,
        batches_processed=0,
        batches_in_epoch=lambda _epoch: V3_BATCHES_PER_EPOCH - 1,
    )
    trainer.epoch = 1
    trainer.global_step = 0
    trainer.global_train_tokens_seen = 0

    with pytest.raises(RuntimeError, match="loader resolves"):
        platform.validate_epoch_horizon(trainer, args.steps)


def test_deliberate_max_steps_uses_step_duration(built, tmp_path):
    _, args = built
    explicit = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    explicit["shared"]["max_steps"] = 2_501
    config_path = tmp_path / "split.yaml"
    config_path.write_text(yaml.safe_dump(explicit), encoding="utf-8")
    args.config = str(config_path)

    # This unit test exercises YAML semantics directly. The CLI separately refuses
    # non-canonical config paths for an actual P3 submission.
    platform.apply_arm_config(args)
    cfg = platform.build_config(args, [], validate_controls=False)

    assert args.steps == 2_501
    assert cfg.trainer.max_duration.unit == DurationUnit.steps
    assert cfg.trainer.max_duration.value == 2_501


def test_runtime_summary_records_observed_world_and_provenance(built, monkeypatch, capsys):
    cfg, args = built
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(platform, "get_rank", lambda: 0)
    trainer = SimpleNamespace(
        train_module=SimpleNamespace(model=SimpleNamespace(parameters=lambda: [])),
        global_step=100,
    )
    losses = SimpleNamespace(first=3.0, last=2.0, wandb_url="")

    platform.summarise(opts=args, config=cfg, trainer=trainer, losses=losses, seconds=1.5)

    summary = json.loads(capsys.readouterr().out)
    assert summary["world_size"] == 8
    assert summary["local_world_size"] == 8
    assert summary["base_model_id"] == QWEN2_0_5B_HF_ID
    assert summary["base_model_revision"] == QWEN2_0_5B_HF_REVISION
    assert summary["base_model_weight_sha256"] == QWEN2_0_5B_HF_WEIGHTS_SHA256
    assert summary["loss_implementation"] == "fused_linear"
    assert summary["tokenizer_artifact_id"] == TOKENIZER_ARTIFACT_ID
    assert summary["tokenizer_file_sha256"] == TOKENIZER_FILE_SHA256
