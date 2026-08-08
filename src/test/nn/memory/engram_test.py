import json
from dataclasses import is_dataclass

import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.memory.engram import DOLMA2_COMPRESSION_MAP_NAME, Engram, EngramConfig


def _config(**kwargs) -> EngramConfig:
    values = {
        "orders": (2, 3),
        "num_hash_heads": 2,
        "table_sizes": (5, 7),
        "embedding_dim": 3,
        "vocab_size": 5,
    }
    values.update(kwargs)
    return EngramConfig(**values)


def _single_order_config(**kwargs) -> EngramConfig:
    values = {
        "orders": (2,),
        "num_hash_heads": 1,
        "table_sizes": (5,),
        "embedding_dim": 2,
        "vocab_size": 5,
    }
    values.update(kwargs)
    return EngramConfig(**values)


def _manual_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _configure_math_fixture(module: Engram) -> None:
    with torch.no_grad():
        module.tables[0].weight.zero_()
        module.tables[0].weight[0] = torch.tensor([9.0, 9.0])
        module.tables[0].weight[1] = torch.tensor([1.0, 2.0])
        module.tables[0].weight[2] = torch.tensor([-2.0, 1.0])
        module.key_proj.weight.copy_(torch.eye(2))
        module.value_proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))


def _manual_gated_output(
    module: Engram,
    hidden_states: torch.Tensor,
    hash_indices: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    retrieved = torch.stack(
        [
            torch.zeros(2),
            module.tables[0].weight[1].detach(),
            module.tables[0].weight[2].detach(),
        ]
    ).unsqueeze(0)
    key = F.linear(retrieved, module.key_proj.weight)
    value = F.linear(retrieved, module.value_proj.weight)
    query_norm = _manual_rms_norm(hidden_states, module.query_norm.eps)
    key_norm = _manual_rms_norm(key, module.key_norm.eps)
    alpha = torch.sigmoid((query_norm * key_norm).sum(dim=-1, keepdim=True) / (2**0.5))
    return alpha * value


def test_config_defaults_and_clean_serialization() -> None:
    defaults = EngramConfig()
    assert is_dataclass(defaults)
    assert isinstance(defaults, ModuleConfig)
    assert defaults.orders == (2, 3)
    assert defaults.num_hash_heads == 8
    assert defaults.vocab_size == 100352
    assert defaults.tokenizer_compression is True
    assert defaults.conv_dilation == 1

    config = _config(compression_map=(0, 0, 1, 2, 3))
    dumped = config.as_config_dict()
    assert dumped["orders"] == [2, 3]
    assert dumped["table_sizes"] == [5, 7]
    assert dumped["compression_map"] == [0, 0, 1, 2, 3]
    json.dumps(dumped)
    assert EngramConfig.from_dict(dumped).as_config_dict() == dumped


@pytest.mark.parametrize(
    "kwargs",
    [
        {"orders": ()},
        {"orders": (2, 2), "table_sizes": (5, 7)},
        {"orders": (1,), "table_sizes": (5,)},
        {"orders": (2,), "table_sizes": (5, 7)},
        {"orders": (2,), "table_sizes": (4,)},
        {"num_hash_heads": 0},
        {"embedding_dim": 0},
        {"vocab_size": 0},
        {"compression_map": (0, 1)},
        {"compression_map": (0, 1, 2, 3, 5)},
        {"compression_map": (0, 0, 2, 2, 2)},
        {"tokenizer_compression": False, "compression_map": (0, 0, 1, 2, 3)},
        {"compression_map_name": "unknown"},
        {
            "compression_map": (0, 0, 1, 2, 3),
            "compression_map_name": DOLMA2_COMPRESSION_MAP_NAME,
        },
        {
            "tokenizer_compression": False,
            "compression_map_name": DOLMA2_COMPRESSION_MAP_NAME,
        },
        {"conv_dilation": 0},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(OLMoConfigurationError):
        _config(**kwargs)


@pytest.mark.parametrize("d_model", [0, -1, True])
def test_invalid_d_model_is_rejected(d_model) -> None:
    config = _config()
    with pytest.raises(OLMoConfigurationError):
        config.build(d_model)
    with pytest.raises(OLMoConfigurationError):
        config.num_params(d_model)
    with pytest.raises(OLMoConfigurationError):
        config.num_active_params(d_model)
    with pytest.raises(OLMoConfigurationError):
        config.num_flops_per_token(d_model)


def test_cpu_meta_construction_and_exact_parameter_accounting() -> None:
    config = _config()
    d_model = 4
    module = config.build(d_model, init_device="cpu")

    assert isinstance(module, Engram)
    assert isinstance(module.conv, CausalConv1d)
    assert module.conv.kernel_size == (4,)
    assert module.conv.dilation == (1,)
    assert module.conv.bias is None
    assert module.conv.activation is None
    assert torch.count_nonzero(module.conv.weight) == 0
    assert config.num_params(d_model) == sum(parameter.numel() for parameter in module.parameters())

    full_table_params = sum(
        table_size * config.num_hash_heads * config.embedding_dim
        for table_size in config.table_sizes
    )
    active_table_params = len(config.orders) * config.num_hash_heads * config.embedding_dim
    expected_active = config.num_params(d_model) - full_table_params + active_table_params
    assert config.num_active_params(d_model) == expected_active
    assert module.num_active_params() == expected_active
    assert module.num_flops_per_token(seq_len=17) == config.num_flops_per_token(d_model)

    meta_module = config.build(d_model, init_device="meta")
    assert all(parameter.device.type == "meta" for parameter in meta_module.parameters())
    assert meta_module.compression_map.device.type == "meta"
    meta_module.to_empty(device="cpu")
    meta_module.reset_parameters()
    torch.testing.assert_close(meta_module.compression_map, module.compression_map)
    torch.testing.assert_close(meta_module.hash_multipliers, module.hash_multipliers)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
    for actual, expected in zip(
        meta_module.compute_hash_indices(input_ids),
        module.compute_hash_indices(input_ids),
    ):
        torch.testing.assert_close(actual, expected)


def test_compression_map_config_override_hook_and_identity_fallback() -> None:
    identity = _config().build(4)
    torch.testing.assert_close(identity.compression_map, torch.arange(5))

    compressed = _config(compression_map=(0, 0, 1, 2, 3)).build(4)
    torch.testing.assert_close(compressed.compression_map, torch.tensor([0, 0, 1, 2, 3]))

    overridden = _config(compression_map=(0, 0, 1, 2, 3)).build(4, compression_map=(0, 1, 1, 2, 3))
    torch.testing.assert_close(overridden.compression_map, torch.tensor([0, 1, 1, 2, 3]))

    calls = []

    def build_map(vocab_size: int) -> tuple[int, ...]:
        calls.append(vocab_size)
        return (0, 0, 1, 2, 3)

    hooked = _config().build(4, compression_map_hook=build_map)
    assert calls == [5]
    torch.testing.assert_close(hooked.compression_map, torch.tensor([0, 0, 1, 2, 3]))

    disabled = _config(tokenizer_compression=False).build(4)
    torch.testing.assert_close(disabled.compression_map, torch.arange(5))


def test_sealed_dolma2_compression_map_is_loaded_and_surjective() -> None:
    config = EngramConfig(
        orders=(2,),
        num_hash_heads=1,
        table_sizes=(5,),
        embedding_dim=1,
        vocab_size=100_352,
        compression_map_name=DOLMA2_COMPRESSION_MAP_NAME,
    )
    module = config.build(4)
    unique_ids = torch.unique(module.compression_map)

    assert module.compression_map_persistent is False
    assert "compression_map" not in module.state_dict()
    assert module.compression_map.shape == (100_352,)
    assert unique_ids.numel() == 62_421
    assert unique_ids[0].item() == 0
    assert unique_ids[-1].item() == 62_420
    torch.testing.assert_close(unique_ids, torch.arange(62_421))
    padded = module.compression_map[100_278:]
    assert torch.unique(padded).numel() == 74

    explicit = EngramConfig(
        orders=(2,),
        num_hash_heads=1,
        table_sizes=(5,),
        embedding_dim=1,
        vocab_size=5,
        compression_map=(0, 0, 1, 2, 3),
    ).build(4)
    assert explicit.compression_map_persistent is True
    assert "compression_map" in explicit.state_dict()

    legacy = EngramConfig(
        orders=(2,),
        num_hash_heads=1,
        table_sizes=(5,),
        embedding_dim=1,
        vocab_size=100_352,
        compression_map=tuple(range(100_352)),
    ).build(4)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        module.load_state_dict(legacy.state_dict(), strict=True)


def test_compute_hash_indices_is_exact_deterministic_and_masks_prefixes() -> None:
    module = _config(compression_map=(0, 0, 1, 2, 3)).build(4)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)

    first = module.compute_hash_indices(input_ids)
    second = module.compute_hash_indices(input_ids.clone())

    expected_bigram = torch.tensor([[[0, 0], [0, 0], [2, 1], [3, 0], [3, 1]]])
    expected_trigram = torch.tensor([[[0, 0], [0, 0], [2, 1], [3, 5], [0, 0]]])
    assert len(first) == 2
    torch.testing.assert_close(first[0], expected_bigram)
    torch.testing.assert_close(first[1], expected_trigram)
    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    assert all(indices.dtype == torch.int64 for indices in first)

    identity_hashes = _config().build(4).compute_hash_indices(input_ids)
    assert not torch.equal(identity_hashes[0], first[0])


def test_incomplete_ngrams_are_masked_after_lookup_even_when_row_zero_is_nonzero() -> None:
    module = _config().build(4)
    with torch.no_grad():
        for table in module.tables:
            table.weight.fill_(1.0)

    hashes = tuple(
        torch.zeros((1, 3, module.num_hash_heads), dtype=torch.long) for _ in module.orders
    )
    retrieved = module.retrieve_embeddings(hashes)
    order_width = module.num_hash_heads * module.embedding_dim

    torch.testing.assert_close(retrieved[:, 0], torch.zeros_like(retrieved[:, 0]))
    torch.testing.assert_close(
        retrieved[:, 1, :order_width], torch.ones_like(retrieved[:, 1, :order_width])
    )
    torch.testing.assert_close(
        retrieved[:, 1, order_width:], torch.zeros_like(retrieved[:, 1, order_width:])
    )
    torch.testing.assert_close(retrieved[:, 2], torch.ones_like(retrieved[:, 2]))


def test_zero_initialized_convolution_returns_exact_gated_value_without_hidden_residual() -> None:
    module = _single_order_config().build(2)
    _configure_math_fixture(module)
    hidden_states = torch.tensor([[[3.0, 4.0], [1.0, -2.0], [-3.0, 1.0]]])
    hashes = (torch.tensor([[[0], [1], [2]]], dtype=torch.long),)

    expected = _manual_gated_output(module, hidden_states, hashes)
    actual = module(hidden_states, hash_indices=hashes)

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual, hidden_states)
    torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]))


def test_forward_applies_external_silu_to_causal_convolution() -> None:
    module = _single_order_config().build(2)
    _configure_math_fixture(module)
    with torch.no_grad():
        module.conv.weight.copy_(torch.tensor([[[0.1, 0.2, 0.3, 0.4]], [[-0.2, 0.1, 0.2, 0.3]]]))

    hidden_states = torch.tensor([[[3.0, 4.0], [1.0, -2.0], [-3.0, 1.0]]])
    hashes = (torch.tensor([[[0], [1], [2]]], dtype=torch.long),)
    gated = _manual_gated_output(module, hidden_states, hashes)
    normalized = _manual_rms_norm(gated, module.output_norm.eps)
    convolved = F.conv1d(
        normalized.transpose(1, 2),
        module.conv.weight,
        bias=None,
        padding=3,
        groups=2,
    )[..., : hidden_states.shape[1]].transpose(1, 2)
    expected = gated + F.silu(convolved)

    torch.testing.assert_close(module(hidden_states, hash_indices=hashes), expected)


def test_hash_input_validation() -> None:
    module = _config().build(4)

    with pytest.raises(ValueError, match="shape"):
        module.compute_hash_indices(torch.zeros(3, dtype=torch.long))
    with pytest.raises(TypeError, match="int64"):
        module.compute_hash_indices(torch.zeros((1, 3), dtype=torch.int32))
    with pytest.raises(ValueError, match="range"):
        module.compute_hash_indices(torch.tensor([[0, -1]], dtype=torch.long))
    with pytest.raises(ValueError, match="range"):
        module.compute_hash_indices(torch.tensor([[0, 5]], dtype=torch.long))


def test_forward_input_and_index_validation() -> None:
    module = _config().build(4)
    hidden_states = torch.zeros((1, 3, 4))
    valid = (
        torch.zeros((1, 3, 2), dtype=torch.long),
        torch.zeros((1, 3, 2), dtype=torch.long),
    )

    with pytest.raises(ValueError, match="hidden_states.*shape"):
        module(torch.zeros((1, 4)), hash_indices=valid)
    with pytest.raises(ValueError, match="d_model"):
        module(torch.zeros((1, 3, 5)), hash_indices=valid)
    with pytest.raises(TypeError, match="floating"):
        module(torch.zeros((1, 3, 4), dtype=torch.long), hash_indices=valid)
    with pytest.raises(ValueError, match="one tensor per"):
        module(hidden_states, hash_indices=valid[:1])
    with pytest.raises(ValueError, match="shape"):
        module(
            hidden_states,
            hash_indices=(valid[0][:, :, :1], valid[1]),
        )
    with pytest.raises(TypeError, match="int64"):
        module(
            hidden_states,
            hash_indices=(valid[0].to(torch.int32), valid[1]),
        )
    with pytest.raises(ValueError, match="range"):
        invalid_range = valid[0].clone()
        invalid_range[0, 1, 0] = 5
        module(hidden_states, hash_indices=(invalid_range, valid[1]))
