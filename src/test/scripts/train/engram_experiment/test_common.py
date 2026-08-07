import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerConfig,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from scripts.train.engram_experiment import common


def tiny_model_config() -> TransformerConfig:
    return TransformerConfig.olmo2_1M(vocab_size=common.PADDED_VOCAB_SIZE)


@dataclass
class FakeManifest:
    paths: list[str] = field(
        default_factory=lambda: [
            "s3://edullm-data/pretrain/regmix-10b/v1/tokens/wiki/train-000.u32le.bin"
        ]
    )
    dtype: object = "uint32"
    byte_order: object = "little"
    header_bytes: object = 0


def test_shared_token_and_batch_constants_are_derived_once():
    assert common.TOKENIZER_CONFIG.padded_vocab_size() == 100_352
    assert common.PADDED_VOCAB_SIZE == 100_352
    assert common.SEQUENCE_LENGTH == 2_048
    assert 9_000_000_000 <= common.TARGET_TOKENS <= 11_000_000_000
    assert common.GLOBAL_BATCH_SIZE % (common.WORLD_SIZE * common.RANK_MICROBATCH_SIZE) == 0
    assert common.WORLD_SIZE == 8
    assert common.MAX_STEPS == math.ceil(common.TARGET_TOKENS / common.GLOBAL_BATCH_SIZE)


def test_default_config_is_local_config_only_and_records_the_data_contract(monkeypatch):
    monkeypatch.setattr(
        common,
        "resolve_corpus_from_environment",
        Mock(side_effect=AssertionError("default config construction reached the registry")),
    )

    config = common.build_experiment_config(tiny_model_config())

    assert isinstance(config.dataset, NumpyFSLDatasetConfig)
    assert isinstance(config.data_loader, NumpyDataLoaderConfig)
    assert config.dataset_id == common.SEALED_DATASET_ID
    assert config.dataset_version == common.SEALED_DATASET_VERSION
    assert config.dataset_tokenizer == common.SEALED_DATASET_TOKENIZER
    assert config.dataset_paths == [common.LOCAL_DATASET_PLACEHOLDER]
    assert config.dataset.paths == config.dataset_paths
    assert config.dataset.dtype is NumpyDatasetDType.uint32
    assert config.dataset.work_dir == "/tmp/dataset-cache"
    assert config.dataset.sequence_length == common.SEQUENCE_LENGTH
    assert config.dataset.tokenizer.padded_vocab_size() == 100_352
    assert config.data_loader.global_batch_size == common.GLOBAL_BATCH_SIZE
    assert config.as_config_dict()["dataset_paths"] == [common.LOCAL_DATASET_PLACEHOLDER]


def test_train_module_uses_adamw_cosine_fsdp2_and_no_other_parallelism():
    train_module = common.build_experiment_config(tiny_model_config()).train_module

    assert isinstance(train_module.optim, AdamWConfig)
    assert isinstance(train_module.scheduler, CosWithWarmup)
    assert train_module.rank_microbatch_size == common.RANK_MICROBATCH_SIZE
    assert train_module.max_sequence_length == common.SEQUENCE_LENGTH
    assert train_module.dp_config is not None
    assert train_module.dp_config.name is DataParallelType.fsdp
    assert train_module.dp_config.param_dtype is DType.bfloat16
    assert train_module.dp_config.reduce_dtype is DType.float32
    assert train_module.tp_config is None
    assert train_module.cp_config is None
    assert train_module.pp_config is None
    assert train_module.ep_config is None
    assert train_module.ac_config is not None
    assert train_module.ac_config.mode is TransformerActivationCheckpointingMode.selected_ops


def test_memory_optimizer_override_is_opt_in_and_five_times_base_lr():
    without_memory = common.build_experiment_config(tiny_model_config()).train_module.optim
    with_memory = common.build_experiment_config(
        tiny_model_config(), with_memory_optimizer=True
    ).train_module.optim

    assert isinstance(without_memory, AdamWConfig)
    assert isinstance(with_memory, AdamWConfig)
    assert all(
        override.params != ["blocks.*.memory.*"]
        for override in (without_memory.group_overrides or [])
    )
    memory_override = next(
        override
        for override in (with_memory.group_overrides or [])
        if override.params == ["blocks.*.memory.*"]
    )
    assert memory_override.opts == {
        "lr": pytest.approx(common.BASE_LEARNING_RATE * 5),
        "weight_decay": 0.0,
    }


def test_trainer_checkpoint_and_wandb_callbacks_follow_platform_environment(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    monkeypatch.delenv("EDULLM_RUN_ID", raising=False)
    config = common.build_experiment_config(tiny_model_config(), save_folder="s3://checkpoints/run")

    assert config.trainer.save_folder == "s3://checkpoints/run"
    assert config.trainer.save_overwrite is False
    assert config.trainer.max_duration.value == common.MAX_STEPS
    assert config.trainer.no_evals is True
    assert isinstance(config.trainer.callbacks["config_saver"], ConfigSaverCallback)
    checkpointer = config.trainer.callbacks["checkpointer"]
    assert isinstance(checkpointer, CheckpointerCallback)
    assert checkpointer.save_async is True
    assert checkpointer.max_checkpoints is None
    assert checkpointer.ephemeral_save_interval is None
    assert "wandb" not in config.trainer.callbacks
    assert not any("eval" in name for name in config.trainer.callbacks)

    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "memory-ablation")
    monkeypatch.setenv("EDULLM_RUN_ID", "run-123")
    monkeypatch.setenv("WANDB_RUN_GROUP", "platform-group")
    with_wandb = common.build_experiment_config(tiny_model_config())
    wandb = with_wandb.trainer.callbacks["wandb"]
    assert isinstance(wandb, WandBCallback)
    assert wandb.project == "memory-ablation"
    assert wandb.name == "run-123"
    assert wandb.group is None


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("dtype", None),
        ("dtype", "uint16"),
        ("byte_order", None),
        ("byte_order", "big"),
        ("header_bytes", None),
        ("header_bytes", 4),
        ("paths", []),
    ],
)
def test_manifest_mismatches_are_refused_not_inferred(field_name, bad_value):
    manifest = FakeManifest()
    setattr(manifest, field_name, bad_value)

    with pytest.raises(common.CorpusContractError):
        common.corpus_from_manifest(manifest)


def test_healthy_manifest_is_kept_as_headerless_little_endian_uint32():
    corpus = common.corpus_from_manifest(FakeManifest())

    assert corpus.dataset_id == common.SEALED_DATASET_ID
    assert corpus.version == common.SEALED_DATASET_VERSION
    assert corpus.tokenizer_id == common.SEALED_DATASET_TOKENIZER
    assert corpus.dtype is NumpyDatasetDType.uint32
    assert corpus.paths == FakeManifest().paths


def test_registry_resolution_uses_only_sealed_environment_identity():
    seen = {}

    def registry_reader(dataset_id, version):
        seen["identity"] = (dataset_id, version)
        return FakeManifest()

    corpus = common.resolve_corpus_from_environment(
        environ={
            "EDULLM_DATASET_ID": common.SEALED_DATASET_ID,
            "EDULLM_DATASET_VERSION": common.SEALED_DATASET_VERSION,
            "EDULLM_DATASET_TOKENIZER": common.SEALED_DATASET_TOKENIZER,
        },
        registry_reader=registry_reader,
    )

    assert seen["identity"] == (common.SEALED_DATASET_ID, common.SEALED_DATASET_VERSION)
    assert corpus.paths == FakeManifest().paths


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("EDULLM_DATASET_ID", ""),
        ("EDULLM_DATASET_ID", "pretrain/something-else"),
        ("EDULLM_DATASET_VERSION", "latest"),
        ("EDULLM_DATASET_TOKENIZER", "tokenizer/gpt2"),
    ],
)
def test_train_resolution_requires_the_exact_sealed_identity(name, value):
    environ = {
        "EDULLM_DATASET_ID": common.SEALED_DATASET_ID,
        "EDULLM_DATASET_VERSION": common.SEALED_DATASET_VERSION,
        "EDULLM_DATASET_TOKENIZER": common.SEALED_DATASET_TOKENIZER,
    }
    environ[name] = value

    with pytest.raises(common.CorpusContractError, match=name):
        common.resolve_corpus_from_environment(
            environ=environ,
            registry_reader=Mock(side_effect=AssertionError("invalid identity reached registry")),
        )


def test_parser_keeps_immutable_train_flags_explicit(monkeypatch):
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "s3://default-checkpoints")
    opts, overrides = common.parse_cli_args([])
    assert opts.command is None
    assert opts.save_folder == "s3://default-checkpoints"
    assert opts.param_dtype == "bfloat16"
    assert overrides == []

    opts, overrides = common.parse_cli_args(
        [
            "train",
            "run-123",
            "--param-dtype",
            "bfloat16",
            "--save-folder",
            "s3://explicit-checkpoints",
            "trainer.metrics_collect_interval=7",
        ]
    )
    assert opts.command == "train"
    assert opts.run_name == "run-123"
    assert opts.param_dtype == "bfloat16"
    assert opts.save_folder == "s3://explicit-checkpoints"
    assert overrides == ["trainer.metrics_collect_interval=7"]


def test_default_dispatch_builds_and_prints_counts_without_registry_or_training(
    monkeypatch, capsys
):
    forbidden = Mock(side_effect=AssertionError("default dispatch crossed a train boundary"))
    monkeypatch.setattr(common, "resolve_corpus_from_environment", forbidden)
    monkeypatch.setattr(common, "prepare_training_environment", forbidden)
    monkeypatch.setattr(common, "train", forbidden)

    config = common.dispatch(tiny_model_config, argv=[])

    assert isinstance(config, common.ExperimentConfig)
    assert config.dataset_paths == [common.LOCAL_DATASET_PLACEHOLDER]
    output = capsys.readouterr().out
    assert '"total_parameters"' in output
    assert '"active_parameters"' in output
    forbidden.assert_not_called()


def test_only_exact_train_resolves_then_starts_distributed_training(monkeypatch):
    events = []
    corpus = common.corpus_from_manifest(FakeManifest())

    monkeypatch.setenv("EDULLM_RUN_ID", "run-123")
    monkeypatch.setattr(
        common,
        "resolve_corpus_from_environment",
        Mock(side_effect=lambda: events.append("resolve") or corpus),
    )
    monkeypatch.setattr(common, "prepare_training_environment", lambda: events.append("prepare"))
    monkeypatch.setattr(common, "train", lambda config: events.append(("train", config)))
    monkeypatch.setattr(common, "teardown_training_environment", lambda: events.append("teardown"))

    config = common.dispatch(
        tiny_model_config,
        argv=[
            "train",
            "run-123",
            "--param-dtype",
            "bfloat16",
            "--save-folder",
            "s3://checkpoints/run-123",
        ],
    )

    assert events[0:2] == ["resolve", "prepare"]
    assert events[2] == ("train", config)
    assert events[3] == "teardown"
    assert config.dataset_paths == FakeManifest().paths
    assert config.trainer.save_folder == "s3://checkpoints/run-123"

    with pytest.raises(SystemExit):
        common.dispatch(tiny_model_config, argv=["dry_run"])


def test_fit_trainer_sets_saved_config_then_repairs_resumes_and_fits(monkeypatch):
    events = []
    config = common.build_experiment_config(tiny_model_config())
    config_saver = SimpleNamespace(config=None)

    class FakeTrainer:
        save_folder = "s3://checkpoints/run"
        callbacks = {"config_saver": config_saver}

        def maybe_load_checkpoint(self):
            events.append("load")

        def fit(self):
            events.append("fit")

    monkeypatch.setattr(
        common,
        "remove_torn_checkpoints",
        lambda folder: events.append(("repair", folder)) or [],
    )

    common.fit_trainer(config, FakeTrainer())

    assert config_saver.config == config.as_config_dict()
    assert events == [("repair", "s3://checkpoints/run"), "load", "fit"]


def test_torn_step_detection_keeps_complete_checkpoints_and_unrelated_directories(monkeypatch):
    children = [
        "s3://checkpoints/run/step10",
        "s3://checkpoints/run/step20",
        "s3://checkpoints/run/logs",
    ]
    monkeypatch.setattr(common, "list_directory", lambda *_args, **_kwargs: children)
    monkeypatch.setattr(
        common.Checkpointer,
        "dir_is_checkpoint",
        staticmethod(lambda path: path.endswith("step10")),
    )

    assert common.torn_step_directories("s3://checkpoints/run") == ["s3://checkpoints/run/step20"]
