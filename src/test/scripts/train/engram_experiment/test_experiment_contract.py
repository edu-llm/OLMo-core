import math
from unittest.mock import Mock

import pytest

from olmo_core.data import NumpyDatasetDType
from olmo_core.nn.memory import EngramConfig, LngramConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from olmo_core.optim import CosWithWarmup
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
)
from scripts.train.engram_experiment import base_moe, common, engram_moe, lngram_moe

ARMS = {
    "base": base_moe,
    "engram": engram_moe,
    "lngram": lngram_moe,
}
MEMORY_OPTIMIZER_PARAMS = ["blocks.*.memory.tables.*"]


def test_all_arms_have_exact_model_shapes_overrides_and_parameter_counts():
    models = {name: arm.build_model_config() for name, arm in ARMS.items()}

    base = models["base"]
    assert base.block_overrides is None
    assert base.block.name is TransformerBlockType.moe_reordered_norm
    assert base.block.memory is None
    assert base.block.feed_forward_moe.router.top_k == 4

    expected_memory = {
        "engram": (TransformerBlockType.moe_engram_reordered_norm, EngramConfig),
        "lngram": (TransformerBlockType.moe_lngram_reordered_norm, LngramConfig),
    }
    for name, (block_type, memory_type) in expected_memory.items():
        model = models[name]
        assert model.block.name is TransformerBlockType.moe_reordered_norm
        assert model.block.memory is None
        assert model.block_overrides is not None
        assert list(model.block_overrides) == [1, 5]
        assert all(
            override.name is block_type and isinstance(override.memory, memory_type)
            for override in model.block_overrides.values()
        )
        memory_params = sum(
            override.memory.num_params(model.d_model) for override in model.block_overrides.values()
        )
        assert 60_000_000 <= memory_params <= 75_000_000

    counts = {}
    for name, model in models.items():
        arm = ARMS[name]
        assert model.checkpoint_revision == common.EXPERIMENT_REVISION
        expected_total, expected_active = common.parameter_counts(model)
        assert (arm.EXPECTED_TOTAL_PARAMETERS, arm.EXPECTED_ACTIVE_PARAMETERS) == (
            expected_total,
            expected_active,
        )
        assert abs(expected_total - 400_000_000) / 400_000_000 <= 0.05
        assert model.block.feed_forward_moe.router.top_k == 4
        counts[name] = (expected_total, expected_active)

    totals = [total for total, _ in counts.values()]
    assert (max(totals) - min(totals)) / min(totals) <= 0.01
    base_active = counts["base"][1]
    for name in ("engram", "lngram"):
        assert abs(counts[name][1] - base_active) / base_active <= 0.10
    assert "dense" in lngram_moe.ACTIVE_PARAMETER_BUMP_REASON


def test_all_arms_share_the_sealed_data_batch_and_schedule_contract(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    configs = {name: arm.build_config() for name, arm in ARMS.items()}

    assert common.TARGET_TOKENS == 10_000_000_000
    assert common.SEQUENCE_LENGTH == 2_048
    assert common.PADDED_VOCAB_SIZE == 100_352
    assert common.GLOBAL_BATCH_SIZE == 256 * common.SEQUENCE_LENGTH
    assert common.RANK_MICROBATCH_SIZE == 8 * common.SEQUENCE_LENGTH
    assert common.MAX_STEPS == math.ceil(common.TARGET_TOKENS / common.GLOBAL_BATCH_SIZE)

    for config in configs.values():
        assert config.dataset.dtype is NumpyDatasetDType.uint32
        assert config.dataset.sequence_length == common.SEQUENCE_LENGTH
        assert config.dataset.tokenizer.padded_vocab_size() == common.PADDED_VOCAB_SIZE
        assert config.data_loader.global_batch_size == common.GLOBAL_BATCH_SIZE
        assert config.data_loader.seed == common.DATA_SEED
        assert config.init_seed == common.INIT_SEED
        assert config.train_module.rank_microbatch_size == common.RANK_MICROBATCH_SIZE
        assert config.train_module.max_sequence_length == common.SEQUENCE_LENGTH
        assert isinstance(config.train_module.scheduler, CosWithWarmup)
        assert config.train_module.scheduler.warmup == common.WARMUP_STEPS
        assert config.trainer.max_duration.value == common.MAX_STEPS


def test_memory_optimizer_group_is_present_only_on_memory_arms(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    configs = {name: arm.build_config() for name, arm in ARMS.items()}

    for name, config in configs.items():
        overrides = config.train_module.optim.group_overrides or []
        memory_overrides = [
            override for override in overrides if override.params == MEMORY_OPTIMIZER_PARAMS
        ]
        assert len(memory_overrides) == (0 if name == "base" else 1)
        if memory_overrides:
            assert memory_overrides[0].opts == {
                "lr": pytest.approx(common.BASE_LEARNING_RATE * 5),
                "weight_decay": 0.0,
            }


def test_all_arms_share_checkpoint_callbacks_and_disable_evaluation(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)

    for arm in ARMS.values():
        trainer = arm.build_config().trainer
        assert trainer.save_overwrite is False
        assert trainer.no_evals is True
        assert set(trainer.callbacks) == {"gpu_monitor", "checkpointer", "config_saver"}
        assert isinstance(trainer.callbacks["gpu_monitor"], GPUMemoryMonitorCallback)
        assert isinstance(trainer.callbacks["config_saver"], ConfigSaverCallback)
        checkpointer = trainer.callbacks["checkpointer"]
        assert isinstance(checkpointer, CheckpointerCallback)
        assert checkpointer.save_interval == common.SAVE_INTERVAL
        assert checkpointer.ephemeral_save_interval is None
        assert checkpointer.max_checkpoints is None
        assert checkpointer.save_async is True
        assert not any("eval" in name.lower() for name in trainer.callbacks)


@pytest.mark.parametrize("arm", ARMS.values(), ids=ARMS)
def test_default_main_never_resolves_data_or_trains(arm, monkeypatch, capsys):
    resolve = Mock(side_effect=AssertionError("default main resolved the corpus"))
    train = Mock(side_effect=AssertionError("default main trained"))
    monkeypatch.setattr(common, "resolve_corpus_from_environment", resolve)
    monkeypatch.setattr(common, "train", train)

    config = arm.main([])

    assert config.dataset.paths == [common.LOCAL_DATASET_PLACEHOLDER]
    resolve.assert_not_called()
    train.assert_not_called()
    capsys.readouterr()
