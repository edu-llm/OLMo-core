from unittest.mock import Mock

from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockType, TransformerType
from scripts.train.engram_experiment import base_moe, common


def test_model_is_reordered_norm_moe_at_the_400m_100m_scale():
    model = base_moe.build_model_config()

    assert isinstance(model, TransformerConfig)
    assert model.name is TransformerType.moe
    assert model.vocab_size == common.PADDED_VOCAB_SIZE == 100_352
    assert model.block.name is TransformerBlockType.moe_reordered_norm
    assert model.block.feed_forward_moe is not None
    assert model.block.feed_forward_moe.num_experts == base_moe.NUM_EXPERTS
    assert model.block.feed_forward_moe.hidden_size == base_moe.EXPERT_HIDDEN_SIZE
    assert model.block_overrides is None
    assert model.block_pattern is None
    assert 380_000_000 <= model.num_params <= 420_000_000
    assert 90_000_000 <= model.num_active_params <= 115_000_000
    assert model.num_params == base_moe.EXPECTED_TOTAL_PARAMETERS
    assert model.num_active_params == base_moe.EXPECTED_ACTIVE_PARAMETERS


def test_build_config_uses_shared_local_pipeline_without_memory_overrides():
    config = base_moe.build_config()

    assert config.model is not base_moe.build_model_config()
    assert config.dataset.paths == [common.LOCAL_DATASET_PLACEHOLDER]
    assert all(
        override.params != ["blocks.*.memory.tables.*"]
        for override in (config.train_module.optim.group_overrides or [])
    )


def test_default_main_builds_locally_without_resolving_or_training(monkeypatch):
    resolve = Mock(side_effect=AssertionError("default main resolved the corpus"))
    train = Mock(side_effect=AssertionError("default main trained"))
    monkeypatch.setattr(common, "resolve_corpus_from_environment", resolve)
    monkeypatch.setattr(common, "train", train)

    config = base_moe.main([])

    assert config.dataset.paths == [common.LOCAL_DATASET_PLACEHOLDER]
    resolve.assert_not_called()
    train.assert_not_called()


def test_explicit_train_is_delegated_to_shared_dispatch(monkeypatch):
    dispatch = Mock(return_value="config")
    monkeypatch.setattr(common, "dispatch", dispatch)

    assert base_moe.main(["train"]) == "config"
    dispatch.assert_called_once_with(
        base_moe.build_model_config,
        argv=["train"],
        prog=base_moe.__file__,
    )
