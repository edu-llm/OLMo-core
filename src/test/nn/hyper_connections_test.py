"""
CPU-only tests for hyper-connections and manifold-constrained hyper-connections.

Everything here runs on CPU by design: the correctness properties being checked (doubly
stochastic mixers, exact parameter counts, baseline-equivalent initialisation, symmetry
breaking, float32 routing) are all device-independent, and gating them behind a GPU would mean
never running them.
"""

import copy
import dataclasses
import math
from typing import cast

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.hyper_connections import (
    HyperConnection,
    HyperConnectionConfig,
    ResidualMixerType,
    StreamCollapseConfig,
    StreamCollapseType,
    sinkhorn_log_space,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.transformer import (
    HyperConnectionTransformerBlock,
    ReorderedNormTransformerBlock,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.utils import seed_all

# The parameter counts from the mHC ablation spec, per wrapped sub-layer at n=4. Each is
# 2n gate logits plus the mixer's own parameters.
EXPECTED_PARAM_COUNTS_N4 = {
    ResidualMixerType.identity: 8,
    ResidualMixerType.unconstrained: 24,
    ResidualMixerType.sinkhorn: 24,
    ResidualMixerType.birkhoff: 32,
    ResidualMixerType.kronecker: 12,
}

DOUBLY_STOCHASTIC_MIXERS = [
    ResidualMixerType.identity,
    ResidualMixerType.sinkhorn,
    ResidualMixerType.birkhoff,
    ResidualMixerType.kronecker,
]

ALL_MIXERS = list(ResidualMixerType)


def _hc(mixer: ResidualMixerType, **kwargs) -> HyperConnection:
    config = HyperConnectionConfig(
        n_streams=kwargs.pop("n_streams", 4),
        mixer=mixer,
        init_noise_std=kwargs.pop("init_noise_std", 0.0),
        residual_dropout_p=kwargs.pop("residual_dropout_p", 0.0),
        **kwargs,
    )
    return config.build()


def _block(mixer: ResidualMixerType, *, d_model: int = 64, **hc_kwargs):
    hc_config = HyperConnectionConfig(
        n_streams=hc_kwargs.pop("n_streams", 4),
        mixer=mixer,
        init_noise_std=hc_kwargs.pop("init_noise_std", 0.0),
        residual_dropout_p=hc_kwargs.pop("residual_dropout_p", 0.0),
        **hc_kwargs,
    )
    return HyperConnectionTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(n_heads=4),
        feed_forward=FeedForwardConfig(hidden_size=2 * d_model),
        layer_norm=LayerNormConfig(),
        hyper_connection=hc_config,
        init_device="cpu",
    )


def _reordered_norm_block(*, d_model: int = 64) -> ReorderedNormTransformerBlock:
    return ReorderedNormTransformerBlock(
        d_model=d_model,
        block_idx=0,
        n_layers=1,
        sequence_mixer=AttentionConfig(n_heads=4),
        feed_forward=FeedForwardConfig(hidden_size=2 * d_model),
        layer_norm=LayerNormConfig(),
        init_device="cpu",
    )


def _hc_transformer_config(
    mixer: ResidualMixerType,
    *,
    base: TransformerConfig,
    n_streams: int = 4,
    init_noise_std: float = 0.0,
    residual_dropout_p: float = 0.0,
    collapse: StreamCollapseType = StreamCollapseType.mean,
) -> TransformerConfig:
    hc_config = HyperConnectionConfig(
        n_streams=n_streams,
        mixer=mixer,
        init_noise_std=init_noise_std,
        residual_dropout_p=residual_dropout_p,
    )
    block = dataclasses.replace(
        cast(TransformerBlockConfig, base.block),
        name=TransformerBlockType.hyper_connection,
        hyper_connection=hc_config,
    )
    return dataclasses.replace(
        base,
        block=block,
        name=TransformerType.hyper_connection,
        stream_collapse=StreamCollapseConfig(n_streams=n_streams, policy=collapse),
    )


def _base_transformer_config(*, d_model: int = 64, n_layers: int = 2) -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=4,
        vocab_size=256,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
        layer_norm_eps=1e-6,
        hidden_size_multiplier=1.5,
    )


@pytest.mark.parametrize("mixer", DOUBLY_STOCHASTIC_MIXERS)
@pytest.mark.parametrize("n_streams", [2, 4])
def test_constrained_mixers_are_doubly_stochastic(mixer: ResidualMixerType, n_streams: int):
    seed_all(23)
    hc = _hc(mixer, n_streams=n_streams, init_noise_std=0.5)
    # Push the logits well off their initialisation: the constraint has to hold everywhere on
    # the manifold, not only at the identity-preserving starting point.
    with torch.no_grad():
        if hc.h_res_logits is not None:
            hc.h_res_logits.normal_(mean=0.0, std=3.0)

    h_res = hc.residual_mixer()

    assert h_res.shape == (n_streams, n_streams)
    assert (h_res >= 0).all(), f"{mixer} produced negative entries"
    torch.testing.assert_close(h_res.sum(dim=-1), torch.ones(n_streams), atol=1e-5, rtol=0)
    torch.testing.assert_close(h_res.sum(dim=-2), torch.ones(n_streams), atol=1e-5, rtol=0)


def test_unconstrained_mixer_is_not_doubly_stochastic():
    # The point of the `unconstrained` arm is that nothing holds it on the Birkhoff polytope.
    # If this ever starts passing the doubly stochastic check, the instability control has
    # quietly turned into a second mHC arm and the ablation no longer measures anything.
    seed_all(23)
    hc = _hc(ResidualMixerType.unconstrained, n_streams=4)
    assert hc.h_res_logits is not None
    with torch.no_grad():
        hc.h_res_logits.normal_(mean=0.0, std=3.0)

    h_res = hc.residual_mixer()

    assert (h_res < 0).any(), "expected a raw learned matrix to be able to go negative"
    assert not torch.allclose(h_res.sum(dim=-1), torch.ones(4), atol=1e-3)


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_param_counts_match_config(mixer: ResidualMixerType):
    config = HyperConnectionConfig(n_streams=4, mixer=mixer)
    hc = config.build()

    measured = sum(p.numel() for p in hc.parameters())

    assert measured == EXPECTED_PARAM_COUNTS_N4[mixer]
    assert config.num_params() == EXPECTED_PARAM_COUNTS_N4[mixer]


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_block_routing_param_count_is_twice_the_sublayer_count(mixer: ResidualMixerType):
    block = _block(mixer)
    baseline = _reordered_norm_block()

    n_hc = sum(p.numel() for p in block.parameters())
    n_base = sum(p.numel() for p in baseline.parameters())

    assert n_hc - n_base == 2 * EXPECTED_PARAM_COUNTS_N4[mixer]
    assert block.num_routing_params == 2 * EXPECTED_PARAM_COUNTS_N4[mixer]


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_identity_preserving_init_reproduces_the_unwrapped_block(mixer: ResidualMixerType):
    seed_all(42)
    d_model = 64
    block = _block(mixer, d_model=d_model)
    baseline = _reordered_norm_block(d_model=d_model)
    baseline.load_state_dict(
        {
            k: v
            for k, v in block.state_dict().items()
            if not k.startswith(("attention_hc", "feed_forward_hc"))
        },
        strict=False,
    )
    block.eval()
    baseline.eval()

    x = torch.randn(2, 8, d_model)
    with torch.no_grad():
        streams = block(x)
        expected = baseline(x)

    assert streams.shape == (2, 8, 4, d_model)
    for stream_idx in range(4):
        torch.testing.assert_close(streams[:, :, stream_idx], expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_hc_model_matches_baseline_model_at_init(mixer: ResidualMixerType):
    seed_all(17)
    base = _base_transformer_config()
    baseline = base.build()
    baseline.init_weights()
    baseline.eval()

    model = _hc_transformer_config(mixer, base=base).build()
    model.init_weights()
    model.eval()
    _, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected

    input_ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids), baseline(input_ids), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_forward_shape_and_backward_reaches_routing_params(mixer: ResidualMixerType):
    seed_all(7)
    d_model = 32
    block = _block(mixer, d_model=d_model, init_noise_std=1e-2)

    out = block(torch.randn(2, 6, 4, d_model))
    assert out.shape == (2, 6, 4, d_model)

    # Not `out.sum()`: see `test_doubly_stochastic_mixers_preserve_the_stream_sum` for why that
    # particular loss carries no gradient at all into a doubly stochastic mixer.
    out.square().mean().backward()

    routing_params = {
        name: param
        for name, param in block.named_parameters()
        if name.startswith(("attention_hc", "feed_forward_hc"))
    }
    assert routing_params, "the block has no routing parameters"
    for name, param in routing_params.items():
        assert param.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient on {name}"
        assert param.grad.abs().sum() > 0, f"zero gradient on {name}"


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_streams_stay_identical_without_symmetry_breaking(mixer: ResidualMixerType):
    seed_all(3)
    d_model = 32
    block = _block(mixer, d_model=d_model, init_noise_std=0.0)
    optim = torch.optim.SGD(block.parameters(), lr=0.1)

    x = torch.randn(2, 6, d_model)
    block(x).square().mean().backward()
    optim.step()
    optim.zero_grad()

    out = block(x)
    spread = (out - out.mean(dim=-2, keepdim=True)).abs().max().item()
    assert spread < 1e-6, f"streams diverged without symmetry breaking (spread {spread})"


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_symmetry_breaking_makes_streams_diverge(mixer: ResidualMixerType):
    seed_all(3)
    d_model = 32
    block = _block(mixer, d_model=d_model, init_noise_std=1e-2)
    optim = torch.optim.SGD(block.parameters(), lr=0.1)

    x = torch.randn(2, 6, d_model)
    block(x).square().mean().backward()
    optim.step()
    optim.zero_grad()

    out = block(x)
    spread = (out - out.mean(dim=-2, keepdim=True)).abs().max().item()
    assert spread > 1e-5, f"streams did not diverge with symmetry breaking (spread {spread})"


def test_read_in_gate_stays_a_convex_combination_after_noise():
    seed_all(11)
    hc = _hc(ResidualMixerType.sinkhorn, init_noise_std=1e-1)

    h_pre = hc.read_in_gate()

    assert (h_pre > 0).all()
    torch.testing.assert_close(h_pre.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)
    # The noise has to actually land somewhere, otherwise the renormalisation has erased it.
    assert h_pre.std() > 0


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_routing_math_stays_float32_in_bfloat16(mixer: ResidualMixerType):
    seed_all(5)
    d_model = 32
    hc = _hc(mixer, init_noise_std=1e-2).to(torch.bfloat16)

    assert hc.h_pre_logits.dtype == torch.bfloat16
    assert hc.residual_mixer().dtype == torch.float32
    assert hc.read_in_gate().dtype == torch.float32
    assert hc.write_out_gate().dtype == torch.float32

    streams = torch.randn(2, 4, 4, d_model, dtype=torch.bfloat16)
    out = hc(streams, lambda x: x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()


def test_kronecker_rejects_non_power_of_two_streams():
    with pytest.raises(OLMoConfigurationError, match="power of two"):
        HyperConnectionConfig(n_streams=3, mixer=ResidualMixerType.kronecker)
    with pytest.raises(OLMoConfigurationError, match="power of two"):
        HyperConnection(n_streams=6, mixer=ResidualMixerType.kronecker)


def test_birkhoff_rejects_impractical_stream_counts():
    with pytest.raises(OLMoConfigurationError, match="one parameter per"):
        HyperConnectionConfig(n_streams=8, mixer=ResidualMixerType.birkhoff)


@pytest.mark.parametrize("mixer", DOUBLY_STOCHASTIC_MIXERS)
def test_doubly_stochastic_mixers_preserve_the_stream_sum(mixer: ResidualMixerType):
    """
    A doubly stochastic ``H_res`` leaves the sum over streams untouched, since summing
    ``H_res @ Z`` over rows weights each stream by its column sum, which is 1.

    This is worth pinning down because of what it means for a loss: any objective that only
    sees the sum over streams has exactly zero gradient with respect to a ``birkhoff`` or
    ``kronecker`` mixer's parameters, and a "does the gradient reach the routing parameters"
    check written against such a loss reports a failure that is not there.
    """
    seed_all(29)
    hc = _hc(mixer, init_noise_std=0.5)
    with torch.no_grad():
        if hc.h_res_logits is not None:
            hc.h_res_logits.normal_(mean=0.0, std=2.0)

    streams = torch.randn(2, 3, 4, 8)
    mixed = torch.einsum("nm,btmd->btnd", hc.residual_mixer(), streams)

    torch.testing.assert_close(mixed.sum(dim=-2), streams.sum(dim=-2), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("scale", [1e2, 1e4, 1e8])
def test_sinkhorn_is_finite_for_large_magnitude_logits(scale: float):
    """
    Extreme logits must not produce ``inf`` or ``nan``, which a probability-space Sinkhorn does.

    Only finiteness, nonnegativity and the column sums are asserted here. Twenty iterations do
    not reach the doubly stochastic fixed point once the logits are this far from zero, so the
    row sums genuinely drift; see the warning on :func:`sinkhorn_log_space`. The converged
    regime is covered by ``test_constrained_mixers_are_doubly_stochastic``.
    """
    seed_all(2)
    logits = torch.randn(4, 4) * scale

    out = sinkhorn_log_space(logits)

    assert torch.isfinite(out).all()
    assert (out >= 0).all()
    torch.testing.assert_close(out.sum(dim=-2), torch.ones(4), atol=1e-5, rtol=0)


def test_sinkhorn_handles_masked_out_entries():
    logits = torch.zeros(4, 4)
    logits[0, 0] = float("-inf")

    out = sinkhorn_log_space(logits)

    assert torch.isfinite(out).all()
    assert out[0, 0] == 0.0
    torch.testing.assert_close(out.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)


def test_residual_logit_dropout_is_training_only_and_guards_full_rows():
    seed_all(1)
    hc = _hc(ResidualMixerType.sinkhorn, residual_dropout_p=0.9)

    hc.eval()
    torch.testing.assert_close(hc.residual_mixer(), torch.full((4, 4), 0.25), atol=1e-5, rtol=0)

    hc.train()
    saw_a_zero = False
    for _ in range(30):
        h_res = hc.residual_mixer()
        assert torch.isfinite(h_res).all(), "the row/column guard let a mask through"
        torch.testing.assert_close(h_res.sum(dim=-1), torch.ones(4), atol=1e-4, rtol=0)
        torch.testing.assert_close(h_res.sum(dim=-2), torch.ones(4), atol=1e-4, rtol=0)
        saw_a_zero = saw_a_zero or bool((h_res < 1e-6).any())
    assert saw_a_zero, "dropout at p=0.9 never masked anything"


@pytest.mark.parametrize("mixer", ALL_MIXERS)
def test_state_dict_round_trip(mixer: ResidualMixerType):
    seed_all(13)
    d_model = 32
    block = _block(mixer, d_model=d_model, init_noise_std=1e-2)
    x = torch.randn(2, 5, 4, d_model)
    block.eval()
    with torch.no_grad():
        before = block(x)

    state = copy.deepcopy(block.state_dict())
    restored = _block(mixer, d_model=d_model, init_noise_std=1e-2)
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        after = restored(x)

    torch.testing.assert_close(before, after)


@pytest.mark.parametrize("collapse", list(StreamCollapseType))
def test_stream_collapse_starts_as_the_mean(collapse: StreamCollapseType):
    config = StreamCollapseConfig(n_streams=4, policy=collapse)
    module = config.build()
    streams = torch.randn(2, 3, 4, 8)

    torch.testing.assert_close(module(streams), streams.mean(dim=-2))
    assert sum(p.numel() for p in module.parameters()) == config.num_params()


def test_hc_transformer_reports_routing_params():
    base = _base_transformer_config(n_layers=2)
    config = _hc_transformer_config(
        ResidualMixerType.birkhoff, base=base, collapse=StreamCollapseType.softmax
    )
    model = config.build()
    model.init_weights()

    # 2 blocks x 2 sub-layers x 32 + 4 readout logits.
    expected = 2 * 2 * EXPECTED_PARAM_COUNTS_N4[ResidualMixerType.birkhoff] + 4
    assert config.num_routing_params == expected
    assert model.num_routing_params == expected
    assert config.num_params == model.num_params


def test_permutation_count_matches_factorial():
    for n in (2, 3, 4):
        config = HyperConnectionConfig(n_streams=n, mixer=ResidualMixerType.birkhoff)
        assert config.num_residual_mixer_params() == math.factorial(n)


def test_hyper_connection_rejects_a_bad_stream_dimension():
    hc = _hc(ResidualMixerType.sinkhorn)
    with pytest.raises(ValueError, match="expected 4 streams"):
        hc(torch.randn(2, 3, 5, 8), lambda x: x)
    with pytest.raises(ValueError, match="3D .* or 4D"):
        hc(torch.randn(2, 3), lambda x: x)


def test_stream_collapse_config_only_valid_on_hc_models():
    base = _base_transformer_config()
    with pytest.raises(OLMoConfigurationError, match="stream_collapse"):
        dataclasses.replace(base, stream_collapse=StreamCollapseConfig(n_streams=4))


def test_hyper_connection_block_config_only_valid_on_hc_blocks():
    base = _base_transformer_config()
    with pytest.raises(OLMoConfigurationError, match="hyper_connection"):
        dataclasses.replace(
            cast(TransformerBlockConfig, base.block), hyper_connection=HyperConnectionConfig()
        )


def test_single_stream_path_is_unchanged():
    """
    The hyper-connection work added an ``expand_residual_streams``/``collapse_residual_streams``
    pair to ``Transformer.forward``. Both are the identity on the default model, and this pins
    that down: an ordinary transformer must produce bit-identical logits and see no extra
    parameters.
    """
    seed_all(19)
    config = _base_transformer_config(n_layers=2)
    model = config.build()
    model.init_weights()
    model.eval()

    input_ids = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        h = model.embeddings(input_ids)
        torch.testing.assert_close(model.expand_residual_streams(h), h)
        torch.testing.assert_close(model.collapse_residual_streams(h), h)
        logits = model(input_ids)

    assert logits.shape == (2, 8, 256)
    assert config.num_routing_params == 0
    assert model.num_params == config.num_params
