from unittest.mock import Mock

from olmo_core.nn.memory import DOLMA2_COMPRESSION_MAP_NAME, EngramConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from scripts.train.engram_experiment import base_moe, common, engram_moe


def test_model_has_exact_independent_engram_overrides():
    model = engram_moe.build_model_config()

    assert isinstance(model, TransformerConfig)
    assert model.vocab_size == common.PADDED_VOCAB_SIZE == 100_352
    assert model.d_model == base_moe.MODEL_DIM
    assert model.n_layers == base_moe.NUM_LAYERS
    assert model.block.feed_forward_moe is not None
    assert model.block.feed_forward_moe.hidden_size == base_moe.EXPERT_HIDDEN_SIZE
    assert model.block.feed_forward_moe.num_experts == engram_moe.NUM_EXPERTS
    assert engram_moe.NUM_EXPERTS < base_moe.NUM_EXPERTS
    assert model.block_overrides is not None
    assert list(model.block_overrides) == [1, model.n_layers // 2 - 1]

    first, second = model.block_overrides.values()
    for override in (first, second):
        assert override is not model.block
        assert override.name is TransformerBlockType.moe_engram_reordered_norm
        assert isinstance(override.memory, EngramConfig)
        assert override.memory.orders == (2, 3)
        assert override.memory.num_hash_heads == 8
        assert override.memory.table_sizes == engram_moe.TABLE_SIZES
        assert override.memory.embedding_dim == engram_moe.EMBEDDING_DIM
        assert override.memory.vocab_size == common.PADDED_VOCAB_SIZE
        assert override.memory.compression_map_name == DOLMA2_COMPRESSION_MAP_NAME
        assert override.memory.conv_dilation == 3
        assert override.sequence_mixer is not model.block.sequence_mixer
        assert override.feed_forward_moe is not model.block.feed_forward_moe

    assert model.block.memory is None
    assert model.block.name is TransformerBlockType.moe_reordered_norm
    assert first is not second
    assert first.sequence_mixer is not second.sequence_mixer
    assert first.feed_forward_moe is not second.feed_forward_moe
    assert first.memory is not second.memory


def test_engram_table_sizes_are_prime():
    for table_size in engram_moe.TABLE_SIZES:
        assert table_size > 1
        assert all(table_size % divisor for divisor in range(2, int(table_size**0.5) + 1))


def test_parameter_accounting_is_exact_and_near_base():
    model = engram_moe.build_model_config()

    assert model.num_params == engram_moe.EXPECTED_TOTAL_PARAMETERS
    assert model.num_active_params == engram_moe.EXPECTED_ACTIVE_PARAMETERS
    assert abs(model.num_params - base_moe.EXPECTED_TOTAL_PARAMETERS) <= 1_000_000
    assert (
        abs(model.num_active_params - base_moe.EXPECTED_ACTIVE_PARAMETERS)
        / base_moe.EXPECTED_ACTIVE_PARAMETERS
        <= 0.05
    )


def test_build_config_uses_shared_pipeline_and_memory_optimizer():
    config = engram_moe.build_config()

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


def test_default_main_builds_without_resolving_or_training(monkeypatch):
    resolve = Mock(side_effect=AssertionError("default main resolved the corpus"))
    train = Mock(side_effect=AssertionError("default main trained"))
    monkeypatch.setattr(common, "resolve_corpus_from_environment", resolve)
    monkeypatch.setattr(common, "train", train)

    config = engram_moe.main([])

    assert config.dataset.paths == [common.LOCAL_DATASET_PLACEHOLDER]
    resolve.assert_not_called()
    train.assert_not_called()


def test_main_delegates_only_to_shared_memory_dispatch(monkeypatch):
    dispatch = Mock(return_value="config")
    monkeypatch.setattr(common, "dispatch", dispatch)

    assert engram_moe.main(["train"]) == "config"
    dispatch.assert_called_once_with(
        engram_moe.build_model_config,
        with_memory_optimizer=True,
        argv=["train"],
        prog=engram_moe.__file__,
    )
