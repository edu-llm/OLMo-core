import importlib
from typing import Any

import pytest

memory_module = importlib.import_module("olmo_core.nn.memory")

transformer_config_module = importlib.import_module("olmo_core.nn.transformer.config")
transformer_block_module = importlib.import_module("olmo_core.nn.transformer.block")
OLMoConfigurationError = importlib.import_module("olmo_core.exceptions").OLMoConfigurationError

TransformerBlockConfig = transformer_config_module.TransformerBlockConfig
TransformerBlockType = transformer_config_module.TransformerBlockType
EngramConfig = transformer_config_module.EngramConfig
LngramConfig = transformer_config_module.LngramConfig


class _CountConfig:
    def __init__(self, total: int, active: int | None = None):
        self.total = total
        self.active = total if active is None else active

    def num_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.total

    def num_active_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.active


class _EngramMemoryConfig(EngramConfig):
    def __init__(self, total: int = 19, active: int = 5):
        self.total = total
        self.active = active

    def num_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.total

    def num_active_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.active


class _LngramMemoryConfig(LngramConfig):
    def __init__(self, total: int = 23, active: int = 7):
        self.total = total
        self.active = active

    def num_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.total

    def num_active_params(self, d_model: int) -> int:
        assert d_model == 32
        return self.active


class _RecordingMemoryBlock:
    def __init__(self, *, memory: Any, **kwargs: Any):
        self.memory = memory
        self.kwargs = {"memory": memory, **kwargs}


class _RecordingOrdinaryMoEBlock:
    def __init__(
        self,
        *,
        sequence_mixer: Any,
        layer_norm: Any,
        feed_forward_moe: Any,
        d_model: int,
        block_idx: int,
        n_layers: int,
        init_device: str,
        cache: Any,
    ):
        self.kwargs = {
            "sequence_mixer": sequence_mixer,
            "layer_norm": layer_norm,
            "feed_forward_moe": feed_forward_moe,
            "d_model": d_model,
            "block_idx": block_idx,
            "n_layers": n_layers,
            "init_device": init_device,
            "cache": cache,
        }


@pytest.fixture
def recording_block_classes(monkeypatch: pytest.MonkeyPatch) -> dict[str, type]:
    class RecordingEngramBlock(_RecordingMemoryBlock):
        pass

    class RecordingLngramBlock(_RecordingMemoryBlock):
        pass

    classes = {
        "MoEEngramReorderedNormTransformerBlock": RecordingEngramBlock,
        "MoELngramReorderedNormTransformerBlock": RecordingLngramBlock,
        "MoEReorderedNormTransformerBlock": _RecordingOrdinaryMoEBlock,
    }
    for name, block_class in classes.items():
        monkeypatch.setattr(transformer_block_module, name, block_class, raising=False)
    return classes


def _moe_block_config(
    *,
    name: Any,
    memory: Any = None,
) -> tuple[Any, Any, Any, Any]:
    sequence_mixer = _CountConfig(11)
    layer_norm = _CountConfig(3)
    feed_forward_moe = _CountConfig(101, 17)
    config = TransformerBlockConfig(
        name=name,
        sequence_mixer=sequence_mixer,
        layer_norm=layer_norm,
        feed_forward_moe=feed_forward_moe,
        memory=memory,
    )
    return config, sequence_mixer, layer_norm, feed_forward_moe


def test_memory_block_enum_values_and_default() -> None:
    assert TransformerBlockType.moe_engram_reordered_norm == "moe_engram_reordered_norm"
    assert TransformerBlockType.moe_lngram_reordered_norm == "moe_lngram_reordered_norm"

    config, _, _, _ = _moe_block_config(name=TransformerBlockType.moe_reordered_norm)
    assert config.memory is None


@pytest.mark.parametrize(
    ("block_type", "memory", "class_name"),
    [
        (
            TransformerBlockType.moe_engram_reordered_norm,
            _EngramMemoryConfig(),
            "MoEEngramReorderedNormTransformerBlock",
        ),
        (
            TransformerBlockType.moe_lngram_reordered_norm,
            _LngramMemoryConfig(),
            "MoELngramReorderedNormTransformerBlock",
        ),
    ],
)
def test_builds_explicit_memory_block_with_all_config(
    block_type: Any,
    memory: Any,
    class_name: str,
    recording_block_classes: dict[str, type],
) -> None:
    config, sequence_mixer, layer_norm, feed_forward_moe = _moe_block_config(
        name=block_type, memory=memory
    )
    cache = object()

    block = config.build(
        d_model=32,
        block_idx=2,
        n_layers=7,
        init_device="meta",
        cache=cache,
    )

    assert type(block) is recording_block_classes[class_name]
    assert block.kwargs == {
        "sequence_mixer": sequence_mixer,
        "layer_norm": layer_norm,
        "feed_forward_moe": feed_forward_moe,
        "memory": memory,
        "d_model": 32,
        "block_idx": 2,
        "n_layers": 7,
        "init_device": "meta",
        "cache": cache,
    }


def test_ordinary_moe_build_is_unchanged(
    recording_block_classes: dict[str, type],
) -> None:
    config, sequence_mixer, layer_norm, feed_forward_moe = _moe_block_config(
        name=TransformerBlockType.moe_reordered_norm
    )
    cache = object()

    block = config.build(
        d_model=32,
        block_idx=1,
        n_layers=4,
        init_device="cpu",
        cache=cache,
    )

    assert type(block) is recording_block_classes["MoEReorderedNormTransformerBlock"]
    assert block.kwargs == {
        "sequence_mixer": sequence_mixer,
        "layer_norm": layer_norm,
        "feed_forward_moe": feed_forward_moe,
        "d_model": 32,
        "block_idx": 1,
        "n_layers": 4,
        "init_device": "cpu",
        "cache": cache,
    }


def test_memory_parameter_accounting_is_exact_and_independently_active() -> None:
    memory = _EngramMemoryConfig(total=19, active=5)
    config, _, _, _ = _moe_block_config(
        name=TransformerBlockType.moe_engram_reordered_norm,
        memory=memory,
    )

    assert config.num_params(32) == 11 + (2 * 3) + 101 + 19
    assert config.num_active_params(32) == 11 + (2 * 3) + 17 + 5


@pytest.mark.parametrize(
    ("block_type", "memory"),
    [
        (TransformerBlockType.moe_engram_reordered_norm, None),
        (TransformerBlockType.moe_engram_reordered_norm, _LngramMemoryConfig()),
        (TransformerBlockType.moe_lngram_reordered_norm, None),
        (TransformerBlockType.moe_lngram_reordered_norm, _EngramMemoryConfig()),
        (TransformerBlockType.moe_reordered_norm, _EngramMemoryConfig()),
    ],
)
def test_invalid_memory_and_block_type_combinations_raise_configuration_error(
    block_type: Any,
    memory: Any,
    recording_block_classes: dict[str, type],
) -> None:
    config, _, _, _ = _moe_block_config(name=block_type, memory=memory)

    with pytest.raises(OLMoConfigurationError):
        config.build(d_model=32, block_idx=0, n_layers=1)
