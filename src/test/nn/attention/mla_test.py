from typing import Any, Dict

import pytest
import torch

from olmo_core.nn.attention import MLAConfig, MultiheadLatentAttention
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.utils import seed_all


def _small_config(**overrides: Any) -> MLAConfig:
    """A small MLA config suitable for fast CPU tests."""
    kwargs: Dict[str, Any] = dict(
        n_heads=4,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )
    kwargs.update(overrides)
    return MLAConfig(**kwargs)


@pytest.mark.parametrize(
    "mla_config",
    [
        pytest.param(_small_config(), id="default"),
        pytest.param(_small_config(q_lora_rank=24), id="q-lora"),
        pytest.param(_small_config(bias=True), id="bias"),
        pytest.param(_small_config(q_lora_rank=24, bias=True), id="q-lora-bias"),
        pytest.param(_small_config(norm=None), id="no-norm"),
        pytest.param(_small_config(q_lora_rank=24, norm=None), id="q-lora-no-norm"),
        pytest.param(_small_config(rope=None), id="no-rope"),
        pytest.param(
            _small_config(norm=LayerNormConfig(name=LayerNormType.rms, bias=True)),
            id="rms-norm-with-bias",
        ),
        pytest.param(
            _small_config(v_head_dim=24, qk_nope_head_dim=24, qk_rope_head_dim=16),
            id="uneven-head-dims",
        ),
    ],
)
def test_mla_config_num_params(mla_config: MLAConfig):
    d_model = 128
    module = mla_config.build(d_model, layer_idx=0, n_layers=12, init_device="cpu")

    # The estimated number of params must match the actual number of params.
    n_params = sum(p.numel() for p in module.parameters())
    assert mla_config.num_params(d_model) == n_params


@pytest.mark.parametrize("batch_size, seq_len", [(2, 16), (1, 8)])
@pytest.mark.parametrize(
    "mla_config",
    [
        pytest.param(_small_config(), id="default"),
        pytest.param(_small_config(q_lora_rank=24), id="q-lora"),
        pytest.param(_small_config(bias=True), id="bias"),
        pytest.param(_small_config(norm=None), id="no-norm"),
        pytest.param(_small_config(rope=None), id="no-rope"),
        pytest.param(_small_config(rope=RoPEConfig(name=RoPEType.complex)), id="complex-rope"),
        pytest.param(
            _small_config(v_head_dim=24, qk_nope_head_dim=24, qk_rope_head_dim=16),
            id="uneven-head-dims",
        ),
    ],
)
def test_mla_forward_shape_and_backward(mla_config: MLAConfig, batch_size: int, seq_len: int):
    seed_all(0)
    d_model = 128

    module = mla_config.build(d_model, layer_idx=0, n_layers=12, init_device="cpu")
    assert isinstance(module, MultiheadLatentAttention)

    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    y = module(x)

    # Output shape must equal the input shape.
    assert y.shape == x.shape

    # The output must be differentiable and gradients must flow back to the input.
    assert y.requires_grad
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_mla_forward_matches_config_num_params():
    """The forward path and the config's num_params estimate must agree on the same module."""
    d_model = 96
    config = _small_config(q_lora_rank=24, n_heads=3)
    module = config.build(d_model, layer_idx=0, n_layers=4, init_device="cpu")

    n_params = sum(p.numel() for p in module.parameters())
    assert config.num_params(d_model) == n_params

    x = torch.randn(2, 8, d_model)
    assert module(x).shape == x.shape


def test_mla_rejects_odd_rope_dim():
    from olmo_core.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError):
        _small_config(qk_rope_head_dim=7).build(128, layer_idx=0, n_layers=1, init_device="cpu")
