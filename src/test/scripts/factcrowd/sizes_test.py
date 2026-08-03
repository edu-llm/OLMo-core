"""
What the ladder guarantees: fixed depth, clean widths, and a parameter count nobody guessed.

The FFN multiplier is the trap this file exists for. ``d_ffn`` is not ``8 * d_model / 3`` -- every
``olmo2_*`` factory multiplies that by 1.5 and rounds up to a multiple of 256 -- and a count that
misses it reports exactly 75% of the real count, so the model is a third bigger than planned.
Since cost scales as the square of parameters at fixed rho, a grid built on the wrong count would
cost 1.78x its budget and every cell would sit at the wrong rho.

:func:`factcrowd.ladder.sizes.build` is what checks this against a real model, and it needs
``torch``; the tests here cover the closed-form arithmetic it is checked *against*, which does not.
"""

import pytest
from factcrowd.ladder import sizes as S

from olmo_core.exceptions import OLMoConfigurationError


def test_depth_is_fixed_across_the_whole_ladder():
    """
    Width is the only thing that varies, because a depth-scaled ladder confounds the two axes.

    Reasoning capability tracks depth and fact capacity tracks total parameters, so if depth moved
    with width, a reasoning change could not be attributed to fact load.
    """
    assert S.N_LAYERS == 12
    assert len({S.N_LAYERS for _ in S.LADDER}) == 1


def test_every_row_has_a_clean_head_count():
    """Head dim 64 at every width, so no row needs a special case."""
    for ladder_row in S.LADDER:
        assert ladder_row.d_model % S.HEAD_DIM == 0
        assert ladder_row.n_heads == ladder_row.d_model // 64


def test_head_count_is_refused_when_the_width_does_not_divide():
    """A width that needs a ragged head dim is a config error, not something to round."""
    with pytest.raises(OLMoConfigurationError, match="not a multiple of the head"):
        S.LadderRow("odd", 300, 1).n_heads


def test_ffn_hidden_size_includes_the_multiplier_and_the_rounding():
    """
    The trap, asserted directly: 4x d_model at these widths, not 2.67x.

    Worked example at d=256: ``int(8 * 256 / 3)`` is 682, times 1.5 is 1023, rounded up to a
    multiple of 256 is 1024.
    """
    assert S.feed_forward_hidden_size(256) == 1024
    assert S.feed_forward_hidden_size(384) == 1536
    assert S.feed_forward_hidden_size(576) == 2304
    assert S.feed_forward_hidden_size(768) == 3072

    for ladder_row in S.LADDER:
        assert ladder_row.d_ffn == 4 * ladder_row.d_model


@pytest.mark.parametrize("d_model", [256, 384, 576, 768, 320, 448, 640, 896])
def test_dropping_the_multiplier_undercounts_by_exactly_a_quarter(d_model):
    """
    Quantifies the trap, so the figures in the docstrings are checked rather than asserted in prose.

    A formula using a plain ``8 * d / 3`` FFN reports exactly 75% of the real count -- the real
    model is a third bigger -- and since cost goes as the square, a grid planned that way costs
    1.78x its budget. Exact at every width tried, including the 320/448/640/896 set an earlier
    draft of the ladder used, which is how the error was found.
    """
    naive_ffn = S.feed_forward_hidden_size(d_model, multiplier=1.0, multiple_of=1)
    naive = S.non_embedding_params(d_model, d_ffn=naive_ffn)
    real = S.non_embedding_params(d_model)

    assert naive / real == pytest.approx(0.750, abs=0.002)
    assert (real / naive) ** 2 == pytest.approx(1.78, abs=0.02)


def test_the_ladder_reproduces_its_own_declared_counts():
    """
    Each row's expected count is what the arithmetic produces, to the parameter.

    The counts are restated on the rows rather than computed, so this catches a change to either the
    widths or the formula instead of letting the two agree quietly on a new number.
    """
    for ladder_row in S.LADDER:
        assert S.non_embedding_params(ladder_row.d_model) == (
            ladder_row.expected_non_embedding_params
        ), ladder_row.label


@pytest.mark.parametrize(
    "preset, d_model, n_layers, total_with_one_embedding",
    [
        ("olmo2_100M", 512, 12, 100_000_000),
        ("olmo2_190M", 768, 12, 190_000_000),
        ("olmo2_370M", 1024, 16, 370_000_000),
    ],
)
def test_the_formula_reproduces_olmo_cores_own_preset_names(
    preset, d_model, n_layers, total_with_one_embedding
):
    """
    Validation against numbers we did not choose: the presets' names are their parameter counts.

    Non-embedding params plus one 100,278-wide embedding table should land on the label. Agreement
    to 2% across three presets is what makes the closed form trustworthy enough to budget from --
    and it is how the ladder's original 320/448/640/896 widths were found to be 35% over target.
    """
    dolma2_vocab = 100_278
    estimate = S.non_embedding_params(d_model, n_layers=n_layers) + dolma2_vocab * d_model
    assert estimate == pytest.approx(total_with_one_embedding, rel=0.02), preset


def test_the_ladder_steps_evenly_except_at_the_top():
    """
    2.25x, 2.25x, then 1.78x. The short last step is a choice, so it is asserted rather than
    tolerated.

    768 is the width OLMo-core's own ``olmo2_190M`` uses, so ``d_ffn`` matches the preset exactly,
    and the top row is two cells whose job is to break the size confound rather than to extend the
    progression. Holding 2.1x would need ``d_model=832`` and 38% more cost on the grid's most
    expensive row.
    """
    counts = [r.expected_non_embedding_params for r in S.LADDER]
    ratios = [b / a for a, b in zip(counts, counts[1:])]

    assert ratios[0] == pytest.approx(2.25, abs=0.01)
    assert ratios[1] == pytest.approx(2.25, abs=0.01)
    assert ratios[2] == pytest.approx(1.78, abs=0.01)


def test_relative_costs_across_the_ladder_match_the_prd():
    """
    Cost goes as P^2 at fixed rho, which is why the top row is two cells and 250M is absent.

    PRD section 3.1 quotes 5x / 26x / 81x relative to the 13M row; those numbers decide the budget,
    so they are checked here rather than trusted.
    """
    base = S.LADDER[0].expected_non_embedding_params
    factors = [(r.expected_non_embedding_params / base) ** 2 for r in S.LADDER]
    assert factors[1] == pytest.approx(5, abs=0.5)
    assert factors[2] == pytest.approx(26, abs=1.5)
    assert factors[3] == pytest.approx(81, abs=4)


def test_row_lookup_by_label():
    """Configs name a row by label, so the lookup is part of the contract."""
    assert S.row("64M").d_model == 576
    assert S.row("113M").expected_non_embedding_params == 113_283_840


def test_row_lookup_refuses_an_unknown_label_and_lists_the_real_ones():
    """The error fires in front of somebody editing a config, so it names the alternatives."""
    with pytest.raises(OLMoConfigurationError, match="no ladder row '60M'") as excinfo:
        S.row("60M")
    assert "13M" in str(excinfo.value)


@pytest.mark.parametrize("bad", [0, -1])
def test_degenerate_widths_and_depths_are_refused(bad):
    """Each would produce a model with no parameters and a rho of infinity."""
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        S.feed_forward_hidden_size(bad)
    with pytest.raises(OLMoConfigurationError, match="must be positive"):
        S.non_embedding_params(256, n_layers=bad)


def test_embedding_share_is_why_the_vocab_is_32k_and_the_embeddings_are_tied():
    """
    At the smallest rung a tied 32k table is 39% of the model and an untied one is 57%.

    rho is defined against non-embedding capacity, so an embedding table larger than the thing being
    measured is indefensible to a reader even when the arithmetic is right. This pins the figures
    quoted in PRD section 7.1.
    """
    expected_share = {"13M": 39, "28M": 30, "64M": 22, "113M": 18}
    for ladder_row in S.LADDER:
        tied = 32_000 * ladder_row.d_model
        total = tied + ladder_row.expected_non_embedding_params
        assert round(100 * tied / total) == pytest.approx(
            expected_share[ladder_row.label], abs=1
        ), ladder_row.label

    smallest = S.LADDER[0]
    untied = 2 * 32_000 * smallest.d_model
    assert untied > smallest.expected_non_embedding_params
