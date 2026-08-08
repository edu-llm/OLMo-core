from unittest.mock import Mock

from olmo_core.nn.memory import LngramConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from scripts.train.engram_experiment import base_moe, common, lngram_moe


def test_model_has_exact_copied_lngram_overrides_and_reduced_experts():
    model = lngram_moe.build_model_config()

    assert isinstance(model, TransformerConfig)
    assert model.vocab_size == common.PADDED_VOCAB_SIZE == 100_352
    assert model.d_model == base_moe.MODEL_DIM == 384
    assert model.n_layers == base_moe.NUM_LAYERS == 12
    assert model.block.sequence_mixer.n_heads == base_moe.NUM_HEADS == 6
    assert model.block.name is TransformerBlockType.moe_reordered_norm
    assert model.block.memory is None
    assert model.block.feed_forward_moe is not None
    assert model.block.feed_forward_moe.num_experts == lngram_moe.NUM_EXPERTS
    assert lngram_moe.NUM_EXPERTS == 51 < base_moe.NUM_EXPERTS
    assert model.block.feed_forward_moe.hidden_size == base_moe.EXPERT_HIDDEN_SIZE == 336
    assert model.block_overrides is not None
    assert set(model.block_overrides) == {1, model.n_layers // 2 - 1}

    first, second = model.block_overrides.values()
    for override in (first, second):
        assert override is not model.block
        assert override.name is TransformerBlockType.moe_lngram_reordered_norm
        assert isinstance(override.memory, LngramConfig)
        assert override.memory.orders == (2, 3)
        assert override.memory.bits_per_route == 4
        assert override.memory.memory_dim == lngram_moe.MEMORY_DIM
        assert override.memory.conv_dilation == 3
        assert override.memory.norm_eps == 1e-6
        assert override.memory.require_triton is True
        assert override.sequence_mixer is not model.block.sequence_mixer
        assert override.feed_forward_moe is not model.block.feed_forward_moe
        assert override.feed_forward_moe is not None
        assert override.feed_forward_moe.num_experts == lngram_moe.NUM_EXPERTS
        assert override.feed_forward_moe.hidden_size == base_moe.EXPERT_HIDDEN_SIZE

    assert first is not second
    assert first.sequence_mixer is not second.sequence_mixer
    assert first.feed_forward_moe is not second.feed_forward_moe
    assert first.memory is not second.memory


def test_parameter_accounting_is_exact_and_near_400m():
    model = lngram_moe.build_model_config()

    assert model.num_params == lngram_moe.EXPECTED_TOTAL_PARAMETERS == 392_198_016
    assert model.num_active_params == lngram_moe.EXPECTED_ACTIVE_PARAMETERS == 122_942_208
    assert "dense" in lngram_moe.ACTIVE_PARAMETER_BUMP_REASON
    assert (
        abs(model.num_params - base_moe.EXPECTED_TOTAL_PARAMETERS)
        / (base_moe.EXPECTED_TOTAL_PARAMETERS)
        <= 0.01
    )
    assert (
        abs(model.num_active_params - base_moe.EXPECTED_ACTIVE_PARAMETERS)
        / (base_moe.EXPECTED_ACTIVE_PARAMETERS)
        <= 0.10
    )


def test_build_config_uses_shared_pipeline_and_memory_optimizer():
    config = lngram_moe.build_config()

    assert config.dataset.paths == [common.LOCAL_DATASET_PLACEHOLDER]
    assert config.dataset.sequence_length == common.SEQUENCE_LENGTH == 2_048
    assert config.data_loader.global_batch_size == common.GLOBAL_BATCH_SIZE
    assert config.trainer.max_duration.value == common.MAX_STEPS
    assert common.TARGET_TOKENS == 10_000_000_000
    assert config.train_module.rank_microbatch_size == common.RANK_MICROBATCH_SIZE
    assert config.train_module.scheduler.warmup == common.WARMUP_STEPS

    memory_override = next(
        override
        for override in (config.train_module.optim.group_overrides or [])
        if override.params == ["blocks.*.memory.tables.*"]
    )
    assert memory_override.opts == {
        "lr": common.BASE_LEARNING_RATE * 5,
        "weight_decay": 0.0,
    }


def test_main_delegates_only_to_shared_memory_dispatch(monkeypatch):
    dispatch = Mock(return_value="config")
    monkeypatch.setattr(common, "dispatch", dispatch)

    assert lngram_moe.main([]) == "config"
    dispatch.assert_called_once_with(
        lngram_moe.build_model_config,
        with_memory_optimizer=True,
        argv=[],
        prog=lngram_moe.__file__,
    )
