import math

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.layer_norm import RMSNorm
from olmo_core.nn.memory.lngram import Lngram, LngramConfig


def _assert_finite_nonzero(gradient: torch.Tensor | None) -> None:
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_config_roundtrip_and_validation() -> None:
    config = LngramConfig(
        orders=(3, 2),
        bits_per_route=4,
        memory_dim=2,
        surrogate_temperature=0.5,
        surrogate_scale=0.0,
        conv_kernel_size=3,
    )
    restored = LngramConfig.from_dict(config.as_config_dict())
    assert restored == config
    assert restored.orders == (3, 2)

    invalid_kwargs = [
        {"orders": ()},
        {"orders": (1, 2)},
        {"orders": (2, 2)},
        {"orders": (2, 3, 4)},
        {"bits_per_route": 3},
        {"memory_dim": 0},
        {"memory_dim": 1.5},
        {"surrogate_temperature": 0.0},
        {"surrogate_temperature": math.inf},
        {"surrogate_scale": -1.0},
        {"surrogate_scale": math.nan},
        {"conv_kernel_size": 0},
        {"conv_dilation": 0},
        {"norm_eps": 0.0},
        {"norm_eps": math.inf},
        {"require_triton": "yes"},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(OLMoConfigurationError):
            LngramConfig(**kwargs)

    for d_model in (0, 2, 6, 4.0):
        with pytest.raises(OLMoConfigurationError):
            LngramConfig(memory_dim=1).build(d_model)  # type: ignore[arg-type]


def test_required_triton_can_build_portable_module() -> None:
    module = LngramConfig(memory_dim=1, require_triton=True).build(4)
    assert module.require_triton is True


def test_builds_on_cpu_and_meta_with_exact_shapes_and_shared_readouts() -> None:
    config = LngramConfig(memory_dim=2)
    module = config.build(8)

    assert isinstance(module, Lngram)
    assert isinstance(module.input_norm, RMSNorm)
    assert isinstance(module.query_norm, RMSNorm)
    assert isinstance(module.key_norm, RMSNorm)
    assert isinstance(module.conv_norm, RMSNorm)
    assert isinstance(module.conv, CausalConv1d)
    assert module.input_norm.bias is None
    assert module.query_norm.bias is None
    assert module.key_norm.bias is None
    assert module.conv_norm.bias is None
    assert module.w_q.bias is None
    assert module.w_k.bias is not None
    assert module.w_v.bias is not None
    assert module.conv.bias is None
    assert module.conv.activation is None
    assert module.conv.dilation == (1,)
    assert module.input_norm.eps == config.norm_eps == 1e-6
    assert torch.count_nonzero(module.conv.weight) == 0
    assert torch.count_nonzero(module.w_k.bias) == 0
    assert torch.count_nonzero(module.w_v.bias) == 0
    assert all(torch.count_nonzero(table) == 0 for table in module.tables)

    assert len(module.tables) == 2
    assert tuple(module.tables[0].shape) == (2 * 16**2, 2)
    assert tuple(module.tables[1].shape) == (2 * 16**3, 2)
    assert module.w_k.in_features == 2 * 2
    assert module.w_v.in_features == 2 * 2

    calls = {"key": 0, "value": 0}
    key_hook = module.w_k.register_forward_hook(
        lambda *_args: calls.__setitem__("key", calls["key"] + 1)
    )
    value_hook = module.w_v.register_forward_hook(
        lambda *_args: calls.__setitem__("value", calls["value"] + 1)
    )
    try:
        output = module(torch.randn(2, 4, 8))
    finally:
        key_hook.remove()
        value_hook.remove()
    assert output.shape == (2, 4, 8)
    torch.testing.assert_close(output, torch.zeros_like(output))
    assert calls == {"key": len(config.orders), "value": len(config.orders)}

    meta_module = config.build(8, init_device="meta")
    assert all(parameter.is_meta for parameter in meta_module.parameters())
    assert tuple(meta_module.tables[1].shape) == (2 * 16**3, 2)


def test_noop_initialization_still_updates_tables() -> None:
    torch.manual_seed(5)
    module = LngramConfig(memory_dim=2).build(4)
    hidden_states = torch.randn(2, 5, 4, requires_grad=True)
    output = module(hidden_states)

    torch.testing.assert_close(output, torch.zeros_like(output))
    (output * torch.randn_like(output)).sum().backward()
    for table in module.tables:
        _assert_finite_nonzero(table.grad)
    assert module.w_q.weight.grad is not None
    torch.testing.assert_close(
        module.w_q.weight.grad,
        torch.zeros_like(module.w_q.weight.grad),
    )


def test_exact_little_endian_addresses_mask_prefix_and_zero_conv_branch() -> None:
    module = LngramConfig(memory_dim=1).build(4)
    with torch.no_grad():
        module.w_q.weight.copy_(torch.eye(4))
        for table in module.tables:
            table.zero_()

        # Codes are 1, 2, and 3. For one route:
        # bigram addresses are 1 + 2*16 = 33 and 2 + 3*16 = 50;
        # the trigram address is 1 + 2*16 + 3*16**2 = 801.
        module.tables[0][33] = 2.0
        module.tables[0][50] = 5.0
        module.tables[1][801] = 7.0
        module.w_k.weight.zero_()
        module.w_k.bias.zero_()
        module.w_v.weight.fill_(1.0)
        module.w_v.bias.zero_()

    hidden_states = torch.tensor(
        [
            [
                [1.0, -1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
            ]
        ]
    )
    actual = module(hidden_states)
    expected = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [6.0, 6.0, 6.0, 6.0],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)


def test_zero_conv_output_is_manually_computed_shared_gated_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.lngram as lngram_module

    module = LngramConfig(orders=(2, 3), memory_dim=1).build(4)
    retrieved = (
        torch.tensor([[[0.0], [1.0], [2.0]]]).expand(-1, -1, 1),
        torch.tensor([[[0.0], [0.0], [3.0]]]).expand(-1, -1, 1),
    )

    def fake_lookup(z, tables, orders, **kwargs):
        assert z.shape == (1, 3, 4)
        assert tuple(tables) == tuple(module.tables)
        assert tuple(orders) == module.orders
        assert kwargs == {
            "bits_per_route": 4,
            "temperature": 1.0,
            "scale": 1.0,
            "require_triton": False,
        }
        return retrieved

    monkeypatch.setattr(lngram_module, "counterfactual_lookup", fake_lookup)
    hidden_states = torch.randn(1, 3, 4)

    expected = torch.zeros_like(hidden_states)
    normalized_h = module.query_norm(hidden_states)
    for order, memory in zip(module.orders, retrieved):
        key = module.w_k(memory)
        value = module.w_v(memory)
        alpha = torch.sigmoid(
            (normalized_h.float() * module.key_norm(key).float()).sum(dim=-1, keepdim=True)
            / math.sqrt(module.d_model)
        ).to(value.dtype)
        contribution = alpha * value
        contribution[:, : order - 1] = 0
        expected = expected + contribution

    torch.testing.assert_close(module(hidden_states), expected)


def test_gate_similarity_accumulates_in_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.lngram as lngram_module

    module = LngramConfig(orders=(2,), memory_dim=1).build(
        4,
        dtype=torch.bfloat16,
    )
    retrieved = torch.randn(1, 3, 1, dtype=torch.bfloat16)
    monkeypatch.setattr(
        lngram_module,
        "counterfactual_lookup",
        lambda *args, **kwargs: (retrieved,),
    )
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16)
    key = module.w_k(retrieved)
    value = module.w_v(retrieved)
    gate_logits = (module.query_norm(hidden_states).float() * module.key_norm(key).float()).sum(
        dim=-1, keepdim=True
    ) / math.sqrt(module.d_model)
    expected = torch.sigmoid(gate_logits).to(value.dtype) * value
    expected[:, :1] = 0

    actual = module(hidden_states)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected)


def test_incomplete_prefixes_stay_zero_with_affine_readout_biases() -> None:
    module = LngramConfig(orders=(2, 3), memory_dim=1).build(4)
    with torch.no_grad():
        module.w_q.weight.zero_()
        for table in module.tables:
            table.zero_()
        module.w_k.weight.zero_()
        module.w_k.bias.zero_()
        module.w_v.weight.zero_()
        module.w_v.bias.fill_(2.0)

    output = module(torch.ones(1, 3, 4))

    torch.testing.assert_close(output[:, 0], torch.zeros_like(output[:, 0]))
    torch.testing.assert_close(output[:, 1], torch.ones_like(output[:, 1]))
    torch.testing.assert_close(output[:, 2], torch.full_like(output[:, 2], 2.0))


def test_total_active_and_flop_accounting_are_exact() -> None:
    d_model = 8
    memory_dim = 2
    config = LngramConfig(memory_dim=memory_dim, conv_kernel_size=3)
    module = config.build(d_model)
    routes = d_model // 4

    table_params = sum(table.numel() for table in module.tables)
    dense_params = (
        d_model  # input RMSNorm
        + d_model  # query RMSNorm
        + d_model * d_model  # W_q
        + 2 * (d_model * routes * memory_dim + d_model)  # shared W_K and W_V
        + d_model  # key RMSNorm
        + d_model  # convolution RMSNorm
        + d_model * config.conv_kernel_size  # depthwise convolution
    )
    expected_total = dense_params + table_params
    expected_active = dense_params + len(config.orders) * routes * memory_dim

    assert sum(parameter.numel() for parameter in module.parameters()) == expected_total
    assert config.num_params(d_model) == expected_total
    assert config.num_active_params(d_model) == expected_active
    assert module.num_active_params() == expected_active
    assert config.num_flops_per_token(d_model) == 6 * expected_active
    assert module.num_flops_per_token() == 6 * expected_active


def test_normal_and_counterfactual_gradients_are_finite_and_nonzero() -> None:
    torch.manual_seed(17)
    module = LngramConfig(memory_dim=2).build(4)
    with torch.no_grad():
        for table in module.tables:
            table.normal_(std=0.02)
        module.conv.weight.normal_(std=0.1)

    hidden_states = torch.randn(2, 5, 4, requires_grad=True)
    output = module(hidden_states)
    (output * torch.randn_like(output)).sum().backward()

    for table in module.tables:
        _assert_finite_nonzero(table.grad)
    _assert_finite_nonzero(module.w_q.weight.grad)
    _assert_finite_nonzero(module.w_k.weight.grad)
    _assert_finite_nonzero(module.w_k.bias.grad)
    _assert_finite_nonzero(module.w_v.weight.grad)
    _assert_finite_nonzero(module.w_v.bias.grad)
    _assert_finite_nonzero(module.conv.weight.grad)
    _assert_finite_nonzero(module.input_norm.weight.grad)
    _assert_finite_nonzero(module.query_norm.weight.grad)
    _assert_finite_nonzero(module.key_norm.weight.grad)
    _assert_finite_nonzero(module.conv_norm.weight.grad)
    _assert_finite_nonzero(hidden_states.grad)


@pytest.mark.parametrize(
    "hidden_states,match",
    [
        (torch.randn(2, 4), "rank 3"),
        (torch.randn(2, 3, 8), "last dimension"),
        (torch.ones(2, 3, 4, dtype=torch.long), "floating point"),
    ],
)
def test_forward_validates_inputs(hidden_states: torch.Tensor, match: str) -> None:
    module = LngramConfig(memory_dim=1).build(4)
    with pytest.raises(ValueError, match=match):
        module(hidden_states)


def test_public_docstrings_cite_the_paper() -> None:
    assert "arXiv:2605.24869" in (LngramConfig.__doc__ or "")
    assert "arXiv:2605.24869" in (Lngram.__doc__ or "")
