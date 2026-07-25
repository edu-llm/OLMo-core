"""
Tests for the ``d_state`` constraint helpers.

These exist because the *choice* of ``d_state`` is made in several places -- every preset plus
the ``A_5`` harness -- while the rule governing it lived only as a negative check inside
``_validate_dims`` and as prose repeated across a dozen docstrings. Stating the rule positively
is only worth doing if the positive and negative forms cannot drift apart, so most of what is
below is a consistency assertion between the two.
"""

import pytest

from olmo_core.nn.mamba3 import (
    DEFAULT_D_STATE,
    Mamba3Config,
    Mamba3MixerConfig,
    admissible_block_sizes,
    kernel_padded_width,
)


@pytest.mark.parametrize(
    "d_state, expected",
    [
        (48, (2, 3, 4, 6, 8)),
        (96, (2, 3, 4, 6, 8)),
        (128, (2, 4, 8)),
        (144, (2, 3, 4, 6, 8)),
        (192, (2, 3, 4, 6, 8)),
        (256, (2, 4, 8)),
        (12, (2, 3, 4, 6)),
        (1, ()),
        (0, ()),
    ],
)
def test_admissible_block_sizes(d_state: int, expected: tuple):
    assert admissible_block_sizes(d_state) == expected


def test_admissible_block_sizes_agrees_with_the_validator():
    """
    The positive and negative forms of the same rule must never disagree.

    ``admissible_block_sizes`` is only safe to cite in a docstring or a preset if building the
    config it describes actually succeeds, so this drives the real constructor for every pair
    rather than re-deriving the arithmetic.
    """
    for d_state in (12, 48, 96, 128, 192):
        admissible = admissible_block_sizes(d_state)
        for b in range(2, 9):
            config = Mamba3MixerConfig(n_heads=4, d_state=d_state, rotation_block_size=b)
            if b in admissible:
                assert config.num_params(d_model=64) > 0
            else:
                with pytest.raises(ValueError, match="divisible"):
                    config.num_params(d_model=64)


def test_default_d_state_admits_the_full_baseline_and_nc1_sweep():
    """
    The property the default is *chosen* for, asserted next to the constant itself.

    An NC^1 ablation is only single-variable if the TC^0 baseline and the NC^1 arm can share one
    state size, which requires the default to admit both ``b=2`` and ``b=3``. This previously
    pinned the opposite -- that 128 could not express ``b=3`` -- and it is inverted rather than
    deleted because the failure it guards against is the same one: someone picking a ``d_state``
    on parameter count or roundness alone and silently forcing the ablation to change two fields.
    """
    admissible = admissible_block_sizes(DEFAULT_D_STATE)
    assert 2 in admissible, "the TC^0 baseline must be expressible at the default"
    assert 3 in admissible, "the NC^1 arm must be expressible at the default, with no override"


def test_every_preset_reports_the_same_default():
    """
    The default is defined once; assert no preset has drifted from it.

    It was previously written out in six places, which is exactly the shape of bug that stays
    invisible until two of the six disagree.
    """
    presets = (
        Mamba3Config.mamba3_hybrid_190M,
        Mamba3Config.mamba3_hybrid_1B,
        Mamba3Config.mamba3_olmo3_370M,
    )
    for preset in presets:
        config = preset(vocab_size=1024)
        assert isinstance(config.block, dict)
        mixer = config.block["mamba3"].sequence_mixer
        assert isinstance(mixer, Mamba3MixerConfig)
        assert mixer.d_state == DEFAULT_D_STATE, f"{preset.__name__} drifted from the default"


# ------------------------------------------------------------------------------------------
# The power-of-two / divisible-by-three conflict
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dim, expected", [(8, 16), (16, 16), (96, 128), (128, 128), (144, 256), (192, 256), (256, 256)]
)
def test_kernel_padded_width(dim: int, expected: int):
    assert kernel_padded_width(dim) == expected


def test_kernel_padded_width_matches_the_adapter():
    """The diagnostic is worthless if it disagrees with the padding the kernel adapter applies."""
    from olmo_core.nn.mamba3.mamba3_ssd_fast import _padded

    for dim in (8, 12, 16, 20, 96, 128, 144, 192, 240, 256):
        assert kernel_padded_width(dim) == _padded(dim)


def test_no_admissible_b3_d_state_avoids_kernel_padding():
    """
    The conflict itself, as an assertion rather than a comment.

    No power of two is divisible by 3, so *every* ``d_state`` that can express ``b=3`` pays
    padding on the official kernel path. That is a fact about arithmetic, not about this
    codebase, and pinning it stops anyone hunting for the zero-waste ``b=3`` configuration that
    cannot exist.
    """
    candidates = [d for d in range(2, 4096) if 3 in admissible_block_sizes(d)]
    assert candidates, "sanity: some d_state must admit b=3"
    assert all(kernel_padded_width(d) != d for d in candidates)


def test_padding_waste_is_reported_but_not_counted_as_flops():
    """
    Padding must show up as a diagnostic, never inside the FLOP count.

    Padded lanes carry zeros. Counting them as model FLOPs would *raise* reported MFU for the
    configuration that wastes more hardware, which inverts the metric. The correct behaviour is
    an unchanged FLOP count plus a visible waste figure.
    """
    mixer = Mamba3MixerConfig(n_heads=4, head_dim=64, d_state=192, rotation_block_size=3).build(
        d_model=256, layer_idx=0, n_layers=4, init_device="meta"
    )
    waste = mixer.kernel_padding_waste()

    assert waste["d_state"] == 192 and waste["d_state_padded"] == 256
    assert waste["d_state_waste"] == pytest.approx(0.25)
    # head_dim 64 is already a power of two, so it is free.
    assert waste["head_dim_padded"] == 64 and waste["head_dim_waste"] == 0.0

    unpadded = Mamba3MixerConfig(n_heads=4, head_dim=64, d_state=128, rotation_block_size=4).build(
        d_model=256, layer_idx=0, n_layers=4, init_device="meta"
    )
    assert unpadded.kernel_padding_waste()["d_state_waste"] == 0.0

    # The FLOP count tracks the logical width only: 192 vs 128 differs by the real 1.5x, not by
    # the padded 2x.
    assert mixer.num_flops_per_token(512) > unpadded.num_flops_per_token(512)
