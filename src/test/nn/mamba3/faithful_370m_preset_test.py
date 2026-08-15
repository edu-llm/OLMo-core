"""
Tests for :meth:`Mamba3Config.mamba3_faithful_olmo3_370M`, the published-SISO 370M preset.

The preset exists so a ``b=2`` versus ``b=3`` comparison can be run on an architecture that is
actually Mamba-3, rather than on :meth:`Mamba3Config.mamba3_olmo3_370M`, which departs from
published SISO in seven ways beyond the rotation (see ``MAMBA3_B2_VS_B3.md``). Its contract is
narrow and worth pinning: every field is the published one except three recorded deviations, and
``rotation_block_size`` is the only field a caller is expected to move.

Everything here builds on ``meta`` or reads configs, so it allocates nothing and needs no GPU.
"""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.mamba3.mixer import Mamba3MixerConfig
from olmo_core.nn.transformer.init import InitMethod

VOCAB = 100_352
D_MODEL = 1024

#: One ``theta_proj`` column per extra angle: ``b=3`` needs 96 angles where ``b=2`` needs 48, over
#: twelve Mamba layers of width 1024. This is the whole intrinsic parameter cost of the treatment.
EXPECTED_B3_PARAMETER_COST = (96 - 48) * D_MODEL * 12


def mixer_of(config: Mamba3Config) -> Mamba3MixerConfig:
    """The Mamba-3 mixer config out of a built hybrid config."""
    assert isinstance(config.block, dict), "the hybrid preset uses named blocks"
    mixer = config.block["mamba3"].sequence_mixer
    assert isinstance(mixer, Mamba3MixerConfig)
    return mixer


# ------------------------------------------------------------------------------------------
# Fidelity to published SISO
# ------------------------------------------------------------------------------------------


def test_preset_reproduces_the_published_siso_mixer():
    """
    Every published-SISO field, checked in one place.

    Each of these was found missing by the August fidelity audit and restored; a regression on any
    one of them silently turns the arm back into something that is not Mamba-3, while every printed
    field still reads correctly.
    """
    mixer = mixer_of(Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB))

    assert mixer.head_dim == 64, "published head dimension"
    assert mixer.n_heads * mixer.head_dim == 2 * D_MODEL, "published expand factor of 2"
    assert mixer.mimo_rank == 1, "SISO"
    assert mixer.n_groups == 1, "MVA: B and C are shared across heads"
    assert mixer.bc_norm is True, "BCNorm"
    assert mixer.bc_bias is False, "the pre-BCNorm linear bias is not the published one"
    assert mixer.bc_bias_after_norm is True, "published bias is applied after BCNorm"
    assert mixer.dynamic_a is True, "published A is token-dependent"
    assert mixer.d_skip is True, "published D skip"
    assert mixer.norm_before_gate is True, "published hybrid output ordering"
    assert mixer.dt_scaled_rotation is True, "published angle is tanh(theta) * pi * dt"
    assert mixer.rope_fraction == 0.5, "published default rotates half the state"
    assert mixer.theta_max is None, "the paper bounds the angle with tanh, not a clamp"


def test_preset_records_its_three_deviations_and_no_others():
    """
    The deviations are deliberate, so they are asserted rather than left to drift.

    ``d_state`` is forced (128 cannot express ``b=3``), ``group_mean`` is a measured 18.8x speed
    decision, and the static ``A_log`` baseline under ``dynamic_a`` is a carry-over. All three are
    shared by both block sizes, so none is a confounder for the comparison -- but a fourth
    appearing without anybody noticing would be.
    """
    mixer = mixer_of(Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB))

    assert mixer.d_state == 192
    assert mixer.rotation_timescale == "group_mean"
    assert mixer.a_log_init_min is not None and mixer.a_log_init_max is not None


def test_preset_uses_the_pre_norm_block():
    """Section 3.4 specifies a Llama-style pre-normalized backbone, not OLMo-2 reordered-norm."""
    config = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB)

    assert isinstance(config.block, dict)
    for name, block in config.block.items():
        assert block.name == "default", f"{name} block must be pre-norm, got {block.name}"


def test_preset_keeps_the_olmo3_370m_shell():
    """
    The backbone is OLMo-3-370M's, so the arm sits beside the repository's other 370M runs.

    Depth, width, the 3:1 substitution pattern, RoPE theta and QK-norm all follow the reference;
    only the sub-quadratic layer is replaced.
    """
    config = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB)

    assert config.d_model == D_MODEL
    assert config.n_layers == 16
    assert config.block_pattern == ["mamba3", "mamba3", "mamba3", "attn"]
    assert isinstance(config.block, dict)
    attn = config.block["attn"].sequence_mixer
    assert attn.rope is not None and attn.rope.theta == 500_000
    assert attn.qk_norm is not None


# ------------------------------------------------------------------------------------------
# The block size, which is the only field a caller moves
# ------------------------------------------------------------------------------------------


def test_preset_defaults_to_the_tc0_baseline():
    """``b=2`` is the control, so it is the default; the treatment has to be asked for."""
    mixer = mixer_of(Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB))

    assert mixer.rotation_block_size == 2


@pytest.mark.parametrize("block_size", [2, 3])
def test_preset_threads_the_block_size_to_the_mixer(block_size):
    """The ablation's core failure mode is a block size that never reaches the mixer."""
    config = Mamba3Config.mamba3_faithful_olmo3_370M(
        vocab_size=VOCAB, rotation_block_size=block_size
    )

    assert mixer_of(config).rotation_block_size == block_size


def test_preset_rejects_a_block_size_the_state_cannot_express():
    """``d_state`` must be divisible by ``b``; 5 does not divide 192."""
    with pytest.raises(OLMoConfigurationError):
        Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB, rotation_block_size=5).build(
            init_device="meta"
        )


@pytest.mark.parametrize(
    "block_size,expected_rotated_channels",
    [(2, 96), (3, 96)],
)
def test_both_block_sizes_rotate_the_same_state_channels(block_size, expected_rotated_channels):
    """
    ``rope_fraction=0.5`` covers half the state at either block size.

    This is what keeps the comparison about the *group* rather than about how much of the state is
    rotated: 48 planes of 2 and 32 blocks of 3 both come to 96 of 192 channels.
    """
    model = Mamba3Config.mamba3_faithful_olmo3_370M(
        vocab_size=VOCAB, rotation_block_size=block_size
    ).build(init_device="meta")
    # `.attention` is the built block's sequence-mixer slot, named for checkpoint compatibility.
    mixer = model.blocks["0"].attention

    assert mixer.n_rotated_blocks * mixer.rotation_block_size == expected_rotated_channels


def test_b3_costs_exactly_the_extra_angle_projection_and_nothing_else():
    """
    The b=3 parameter surcharge must be the angle projection alone.

    Anything else showing up here means the treatment is moving a second thing, which is the one
    failure this comparison cannot absorb.
    """
    b2 = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB, rotation_block_size=2)
    b3 = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB, rotation_block_size=3)

    assert b3.num_params - b2.num_params == EXPECTED_B3_PARAMETER_COST

    b2_shapes = {
        name: tuple(p.shape) for name, p in b2.build(init_device="meta").named_parameters()
    }
    b3_shapes = {
        name: tuple(p.shape) for name, p in b3.build(init_device="meta").named_parameters()
    }
    assert set(b2_shapes) == set(b3_shapes), "b must not add or remove a parameter tensor"
    differing = {name for name in b2_shapes if b2_shapes[name] != b3_shapes[name]}
    assert differing and all(
        name.endswith("theta_proj.weight") for name in differing
    ), f"only theta_proj may change shape with b, got {sorted(differing)}"


def test_intermediate_size_is_settable_for_parameter_matching():
    """
    The FFN width is the knob both the 370M target and optional param-matching turn.

    The paper does the same thing -- its MIMO variants are ``parameter-matched to their SISO
    counterparts by reducing the MLP width``.
    """
    narrow = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB, intermediate_size=3456)
    wide = Mamba3Config.mamba3_faithful_olmo3_370M(vocab_size=VOCAB, intermediate_size=3468)

    assert isinstance(narrow.block, dict) and isinstance(wide.block, dict)
    for name in ("mamba3", "attn"):
        feed_forward = narrow.block[name].feed_forward
        assert feed_forward is not None and feed_forward.hidden_size == 3456
    assert wide.num_params > narrow.num_params


# ------------------------------------------------------------------------------------------
# It has to run
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [2, 3])
def test_the_mixer_runs_and_differentiates_at_the_production_geometry(block_size):
    """
    Forward and backward at the arm's real mixer dimensions, on CPU.

    ``b=2`` reaching the published options at all is the thing worth checking: every one of them
    was added for the ``b=3`` arm, ``bc_bias_after_norm`` and ``dt_scaled_rotation`` are coupled by
    a config-time check, and the control arm is the one path nobody exercised. Finite gradients
    matter as much as finite outputs -- the small-angle branch of the rotation is exactly the init
    regime, and getting it wrong is NaN gradients at step 1 rather than a slow drift.

    ``rotation_scan_impl`` is left unset because the quaternion scan refuses to run outside CUDA
    rather than fall back silently; this exercises the same rotation through the chunked path.
    """
    torch.manual_seed(0)
    mixer = Mamba3MixerConfig(
        n_heads=32,
        head_dim=64,
        d_state=192,
        n_groups=1,
        mimo_rank=1,
        rotation_block_size=block_size,
        norm_eps=1e-6,
        bc_norm=True,
        bc_bias=False,
        dynamic_a=True,
        d_skip=True,
        norm_before_gate=True,
        bc_bias_after_norm=True,
        dt_scaled_rotation=True,
        rope_fraction=0.5,
        rotation_timescale="group_mean",
    ).build(D_MODEL, layer_idx=0, n_layers=2, init_device="cpu")
    mixer.init_weights(init_method=InitMethod.normal, d_model=D_MODEL, block_idx=0, num_blocks=2)

    x = torch.randn(1, 24, D_MODEL, requires_grad=True)
    y = mixer(x)
    y.square().mean().backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all(), f"b={block_size} produced a non-finite output"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, f"b={block_size}: {name} received no gradient"
        assert torch.isfinite(
            parameter.grad
        ).all(), f"b={block_size}: {name} gradient is not finite"


# ------------------------------------------------------------------------------------------
# The preset it must not disturb
# ------------------------------------------------------------------------------------------


def test_the_original_370m_preset_is_untouched():
    """
    ``mamba3_olmo3_370M`` is live in another script and its own tests; adding the faithful preset
    must not move it. Pinned by parameter count and by the fields the faithful arm changes.
    """
    original = Mamba3Config.mamba3_olmo3_370M(vocab_size=VOCAB)
    mixer = mixer_of(original)

    assert original.num_params == 467_717_248
    assert original.num_non_embedding_params == 364_956_800
    assert mixer.rotation_block_size == 2
    assert mixer.mimo_rank == 1
    assert mixer.n_heads == 16 and mixer.head_dim is None
    assert mixer.bc_bias is True
    assert mixer.dynamic_a is False
    assert mixer.d_skip is False
    assert mixer.norm_before_gate is False
    assert mixer.bc_bias_after_norm is False
    assert mixer.dt_scaled_rotation is False
    assert mixer.rope_fraction == 1.0
    assert mixer.rotation_timescale == "per_head"
    assert isinstance(original.block, dict)
    assert original.block["mamba3"].name == "reordered_norm"
