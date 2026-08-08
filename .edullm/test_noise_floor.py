"""Every estimator the noise floor rests on, against a truth it was not shown.

Run with ``pytest -v .edullm/test_noise_floor.py``.

TWO KINDS OF TEST HERE AND THE SECOND KIND IS THE POINT. The first kind checks an estimator
recovers a planted answer, which is ordinary. The second kind pins the *analysis conventions*
-- the error df of a blocked design, the bias of a standard deviation, the numbers printed in
the pre-registration's own table -- because those are what were wrong before, they were wrong
silently, and nothing in the repository could have said so. A pre-registration whose numbers
are asserted against the code that generates them cannot drift from it, and drift is the whole
failure mode: the paired MDE table stood at the wrong df for as long as it stood, and it was
found by recomputing it rather than by reading it.
"""

import json
import math
import os
import pathlib
import re
import sys
from typing import List

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import noise_floor as nf  # noqa: E402

PRE_REGISTRATION = pathlib.Path(_HERE) / "hyper-connections.md"


def markdown_table(text: str, header_contains: str) -> List[List[str]]:
    """
    Pull one markdown table out of the pre-registration by something in its header.

    :param text: The whole document.
    :param header_contains: A substring identifying the header row.

    :returns: The body rows, each a list of stripped cells.

    :raises AssertionError: If no table has that header, which is itself the regression --
        a table cannot be silently renamed out from under its test.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if header_contains in line), None)
    assert start is not None, f"no table header containing {header_contains!r}"
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


# ---------------------------------------------------------------------------------------
# The df convention. This is the correction, so it gets the most tests.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_arms, n_seeds, expected",
    [(3, 3, 4), (3, 5, 8), (2, 5, 4), (3, 2, 2), (2, 3, 2), (4, 5, 12)],
)
def test_the_paired_error_df_is_the_randomized_block_count(n_arms, n_seeds, expected):
    """
    ``(k-1)(n-1)``, because blocking on seed spends ``n-1`` df on the seed main effect.

    The pre-registration used ``k(n-1)`` for the paired table -- the unpaired count -- which at
    three arms and three seeds is 6 against a true 4.
    """
    assert nf.error_df(n_arms, n_seeds, paired=True) == expected
    assert nf.error_df(n_arms, n_seeds, paired=True) == (n_arms - 1) * (n_seeds - 1)


@pytest.mark.parametrize("n_arms, n_seeds, expected", [(3, 3, 6), (3, 5, 12), (2, 5, 8), (3, 4, 9)])
def test_the_unpaired_error_df_is_n_minus_k(n_arms, n_seeds, expected):
    """``N - k = k(n-1)``, the pooled within-arm df, with no block term to pay for."""
    assert nf.error_df(n_arms, n_seeds, paired=False) == expected
    assert nf.error_df(n_arms, n_seeds, paired=False) == n_arms * n_seeds - n_arms


def test_the_paired_design_always_has_fewer_error_df_than_the_unpaired_one():
    """
    Which is the content of the correction: pairing costs df, and it is never free.

    A version of this file that made pairing free again -- by using ``k(n-1)`` for both, which
    is exactly the mistake -- would make these two equal and fail here.
    """
    for n_arms in range(2, 6):
        for n_seeds in range(2, 8):
            assert nf.error_df(n_arms, n_seeds, True) < nf.error_df(n_arms, n_seeds, False)


@pytest.mark.parametrize("n_arms, n_seeds", [(1, 5), (3, 1), (0, 0), (2, 1)])
def test_a_design_with_no_error_df_is_refused_rather_than_returned(n_arms, n_seeds):
    with pytest.raises(ValueError):
        nf.error_df(n_arms, n_seeds, paired=True)


def test_the_df_error_was_worth_about_eleven_percent_at_three_by_three():
    """
    What the correction cost, held to the figure the document prints.

    The wrong df is reproduced deliberately -- same standard error, ``k(n-1)`` instead of
    ``(k-1)(n-1)`` -- so that the *size* of the error is pinned and not only its direction.
    """
    from scipy import optimize

    sigma, n_seeds = 0.010, 3
    for rho in (0.0, 0.3, 0.5, 0.7):
        se = nf.contrast_se(sigma, n_seeds, rho, paired=True)
        wrong = optimize.brentq(
            lambda d: nf.power_of(d, se, nf.error_df(3, n_seeds, False)) - 0.80, 1e-12, se * 40
        )
        right = nf.mde(sigma, n_seeds, 3, rho, paired=True)
        assert right > wrong
        assert right / wrong == pytest.approx(1.117, abs=0.002)


def test_the_same_error_at_five_seeds_would_have_been_smaller_and_still_wrong():
    """
    4.9%, because the block eats a smaller share of a larger df. Pinned because it is the
    reason moving to five seeds would have hidden the mistake rather than fixed it.
    """
    from scipy import optimize

    se = nf.contrast_se(0.010, 5, 0.0, paired=True)
    wrong = optimize.brentq(lambda d: nf.power_of(d, se, 12) - 0.80, 1e-12, se * 40)
    right = nf.mde(0.010, 5, 3, 0.0, paired=True)
    assert right / wrong == pytest.approx(1.049, abs=0.002)


# ---------------------------------------------------------------------------------------
# The pre-registration's own tables, asserted against the estimator that generates them.
# ---------------------------------------------------------------------------------------


def test_the_paired_mde_table_in_the_document_is_the_one_the_estimator_produces():
    """
    Cell by cell, at the sigma the document quotes.

    THE REGRESSION GUARD FOR DELIVERABLE 2. The table can be edited and the estimator can be
    edited, and this fails unless both move together.
    """
    rows = markdown_table(
        PRE_REGISTRATION.read_text(), "| ρ | 3 pairs (df 4) | 4 pairs (df 6) | 5 pairs (df 8) |"
    )
    seed_counts = (3, 4, 5)
    assert len(rows) == 4
    for row in rows:
        rho = float(row[0])
        for column, n_seeds in enumerate(seed_counts, start=1):
            assert float(row[column]) == pytest.approx(
                nf.mde(nf.PLANNING_SIGMA_NATS, n_seeds, 3, rho, paired=True), abs=5e-4
            ), f"rho {rho}, {n_seeds} pairs"


def test_the_document_shows_what_the_old_df_convention_cost_rather_than_only_the_new_numbers():
    """
    The correction has to be visible as a correction. A document that quietly held the right
    numbers would pass every other test here and would still have lost the thing worth
    keeping, which is that a reader can tell which version they read.
    """
    rows = markdown_table(
        PRE_REGISTRATION.read_text(), "| ρ | as printed, df = 6 | correct, df = 4 |"
    )
    assert len(rows) == 4
    for row in rows:
        rho, printed, correct = float(row[0]), float(row[1]), float(row[2])
        assert correct == pytest.approx(nf.mde(0.010, 3, 3, rho, paired=True), abs=5e-4)
        assert correct > printed
        assert row[3].startswith("11.7")


def test_the_document_states_the_convention_and_not_only_the_answer():
    text = re.sub(r"\s+", " ", PRE_REGISTRATION.read_text())
    assert "randomized complete block design" in text
    assert "(k − 1)(n − 1)" in text
    assert "**4, not 6**" in text


def test_the_document_carries_the_a100_throughput_row_it_was_measured_at():
    """Deliverable 3: the fact is recorded, with the shape and the row count behind it."""
    text = PRE_REGISTRATION.read_text()
    assert "run_019fe262-778d" in text
    assert "2.91" in text
    assert "72 rows" in text or "over 72" in text
    assert "39.63" in text, "the step the instrument ran on has to be visible beside the median"


def test_the_render_matches_the_estimator():
    rendered = nf.render_mde_table(0.010)
    assert "df 8" in rendered
    assert f"{nf.mde(0.010, 5, 3, 0.7, True):.3f}" in rendered


# ---------------------------------------------------------------------------------------
# The frozen artifact, and the document's account of it.
# ---------------------------------------------------------------------------------------

FROZEN = pathlib.Path(_HERE) / "noise-floor.json"


def frozen() -> dict:
    """
    The frozen noise floor, or a skip where stage 1 has not been frozen yet.

    Skipped rather than failed so this file still runs in a checkout made before the freeze,
    which is most of the ones it was written in.
    """
    if not FROZEN.exists():
        pytest.skip("stage 1 has not been frozen")
    return json.loads(FROZEN.read_text())


def test_the_frozen_artifact_is_the_thing_it_claims_to_be():
    """
    Five distinct seeds, the final step, nothing provisional, and the runs named.

    ``--freeze`` refuses a provisional reading, so in principle this cannot fail. That is the
    reason to assert it: the refusal is one ``if`` in one file, the artifact is what stage 2
    is read against for the rest of the module, and a claim this cheap to check should not
    rest on a code path nobody looks at again.
    """
    f = frozen()
    assert f["label"] == "measured"
    assert f["provisional"] == []
    assert f["n_seeds"] == 5
    assert f["sigma_df"] == 4
    assert f["final_step"] == f["horizon"]
    assert sorted(f["seeds"]) == [0, 1, 2, 3, 4]
    assert len(set(f["runs"])) == 5, "the artifact has to name the five cells it was built on"
    assert all(r.startswith(f["submission"]) for r in f["runs"])
    assert f["sigma_bpb_unbiased"] == pytest.approx(f["sigma_bpb"] / nf.c4(4))
    assert len(f["sources"]) == len(f["weights"]["weights"]) == 7
    assert sum(f["weights"]["weights"]) == pytest.approx(1.0)


def test_the_measured_mde_table_in_the_document_is_the_one_the_estimator_produces():
    """
    Cell by cell, at the frozen sigma, for the four-arm design that is actually funded.

    THE SAME GUARD AS THE PLANNING TABLE ABOVE, POINTED AT THE MEASUREMENT. That table is the
    one the tranche was priced against and this one is what it can actually detect, and the
    second is the one a reader will quote. Both are generated numbers sitting in prose, so
    both need the estimator asserting against them or they drift the way the paired table did.
    """
    f = frozen()
    unweighted = f["sigma_bpb_unbiased"] * nf.NATS_PER_BPB
    weighted = f["weights"]["weighted_sigma"] / nf.c4(f["sigma_df"]) * nf.NATS_PER_BPB

    rows = markdown_table(
        PRE_REGISTRATION.read_text(), "| analysis | df | MDE, unweighted | MDE, strata-weighted |"
    )
    assert len(rows) == 6

    for row in rows:
        analysis, df = row[0], int(row[1])
        paired = analysis.startswith("paired")
        rho = float(analysis.split("=")[1]) if paired else 0.0
        assert df == nf.error_df(4, 5, paired), analysis
        for column, sigma in enumerate((unweighted, weighted), start=2):
            printed = float(row[column].strip("*"))
            assert printed == pytest.approx(
                nf.mde(sigma, 5, 4, rho, paired), abs=5e-5
            ), f"{analysis}, column {column}"


def test_the_document_reports_the_spike_split_rather_than_only_the_sigma_it_produced():
    """
    99% of the measured variance is two runs out of five taking a loss spike, and a σ̂ quoted
    without that is a number a reader will take for seed jitter and plan against as if buying
    replicates would average it down.
    """
    text = re.sub(r"\s+", " ", PRE_REGISTRATION.read_text())
    assert "99.0% of the endpoint variance" in text
    assert "1376–1418" in text and "1726–1773" in text, "the episodes have to be locatable"
    assert "SkipStepAdamW" in text, "the mitigation that exists and is not enabled"
    assert "0.0040 nats unpaired" in text, "the counterfactual is what makes the finding actionable"


# ---------------------------------------------------------------------------------------
# sigma-hat.
# ---------------------------------------------------------------------------------------


def test_pooled_sigma_counts_its_df_and_pools_sums_of_squares():
    one = nf.pooled_sigma([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert one.df == 4
    assert one.n_groups == 1
    assert one.sigma == pytest.approx(float(np.std([1, 2, 3, 4, 5], ddof=1)))

    three = nf.pooled_sigma([[1.0, 2.0, 3.0]] * 3)
    assert three.df == 6
    assert three.sigma == pytest.approx(1.0)


def test_pooled_sigma_skips_groups_that_carry_no_information_rather_than_erroring():
    """A partially-landed fan-out has runs with one point in it, and one point has no spread."""
    estimate = nf.pooled_sigma([[1.0], [1.0, 2.0, 3.0], []])
    assert estimate.df == 2
    assert estimate.n_groups == 1


def test_pooled_sigma_refuses_when_there_is_no_variance_estimate_at_all():
    with pytest.raises(ValueError):
        nf.pooled_sigma([[1.0], [2.0]])


@pytest.mark.parametrize("df, expected", [(2, 12.07), (4, 4.80), (6, 3.42), (12, 2.30)])
def test_variance_interval_spans_match_what_the_documents_quote(df, expected):
    """
    The df = 2 span is 12.1 and the df = 6 span is 3.4, both of which the pre-registration
    states correctly. ``run.baseline-stage.yaml`` attributes the 3.4 to df = 2, which is what
    this test would have caught.
    """
    assert nf.variance_interval_span(df) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("df, expected", [(1, 0.7979), (4, 0.9400), (6, 0.9594), (12, 0.9794)])
def test_c4_is_the_bias_of_a_standard_deviation(df, expected):
    assert nf.c4(df) == pytest.approx(expected, abs=1e-4)
    assert nf.c4(df) < 1.0


def test_the_variance_is_unbiased_and_the_standard_deviation_is_not():
    """
    The second correction, measured rather than cited: at five seeds ``s`` runs 6% low.

    A tool that reported the raw ``s`` as sigma would price every MDE 6% optimistically, which
    is half the size of the df error and the same kind of mistake.
    """
    truth, replicates = 0.010, 2000
    estimates = [
        nf.pooled_sigma([nf.synthetic_pair(5, truth, 0.0, 0.0, r)[0]]) for r in range(replicates)
    ]
    assert np.mean([e.sigma**2 for e in estimates]) / truth**2 == pytest.approx(1.0, abs=0.04)
    assert np.mean([e.sigma for e in estimates]) / truth == pytest.approx(nf.c4(4), abs=0.02)
    assert np.mean([e.sigma_unbiased for e in estimates]) / truth == pytest.approx(1.0, abs=0.02)


def test_the_chi_square_interval_covers_at_about_its_nominal_rate():
    truth, replicates = 0.010, 2000
    covered = sum(
        1
        for r in range(replicates)
        if (lambda e: e.ci_low <= truth <= e.ci_high)(
            nf.pooled_sigma([nf.synthetic_pair(5, truth, 0.0, 0.0, r)[0]])
        )
    )
    assert covered / replicates == pytest.approx(0.95, abs=0.02)


def test_the_trajectory_sees_a_sigma_that_is_still_moving():
    """
    The reason intermediate checkpoints are read at all. A settled trajectory and a strongly
    moving one have to come back with different verdicts, or the field is decoration.
    """
    moving, sources, steps = nf.synthetic_baseline(n_seeds=5, settling=4.0, rng_seed=11)
    verdict = nf.sigma_trajectory(moving.mean(axis=2), steps, bootstrap=800)
    assert verdict.ratio < 0.6, "sigma planted to fall by a factor should be seen falling"
    assert not verdict.settled

    flat, _, steps = nf.synthetic_baseline(n_seeds=5, settling=0.0, rng_seed=11)
    assert nf.sigma_trajectory(flat.mean(axis=2), steps, bootstrap=800).settled


def test_the_trajectory_needs_aligned_seeds_because_the_bootstrap_resamples_rows():
    with pytest.raises(ValueError):
        nf.sigma_trajectory(np.zeros((5, 3)), [1, 2])
    with pytest.raises(ValueError):
        nf.sigma_trajectory(np.zeros((1, 3)), [1, 2, 3])


# ---------------------------------------------------------------------------------------
# Per-source weights.
# ---------------------------------------------------------------------------------------


def test_the_covariance_is_singular_at_five_seeds_which_is_why_this_is_not_gls():
    """
    The arithmetic behind the constraint, asserted rather than asserted-in-prose.

    Five seeds give a 7x7 sample covariance of rank at most 4. ``Sigma^-1 1`` does not exist,
    so a weighting derived from it would be a weighting derived from whatever regularization
    was reached for.
    """
    values, sources, _ = nf.synthetic_baseline(n_seeds=5)
    final = values[:, -1, :]
    covariance = np.cov(final, rowvar=False)
    assert covariance.shape == (len(sources), len(sources))
    assert np.linalg.matrix_rank(covariance) <= 4
    assert np.linalg.matrix_rank(covariance) < len(sources)


@pytest.mark.parametrize("scheme", ["strata", "inverse-variance"])
def test_weights_are_normalized_and_ordered_against_the_per_source_sigma(scheme):
    values, sources, _ = nf.synthetic_baseline(n_seeds=5, shared_fraction=0.0, rng_seed=5)
    weights = nf.inverse_variance_weights(values[:, -1, :], sources, scheme)
    assert sum(weights.weights) == pytest.approx(1.0)
    assert all(w > 0 for w in weights.weights)
    order = np.argsort(weights.sigma)
    ranked = [weights.weights[i] for i in order]
    assert ranked == sorted(ranked, reverse=True), "a noisier source never gets a larger weight"


def test_strata_pool_the_df_behind_each_weight_and_plain_inverse_variance_does_not():
    """The whole argument for the default: a weight on df = 4 is not a weight."""
    values, sources, _ = nf.synthetic_baseline(n_seeds=5)
    final = values[:, -1, :]
    strata = nf.inverse_variance_weights(final, sources, "strata")
    plain = nf.inverse_variance_weights(final, sources, "inverse-variance")
    assert set(plain.df_per_weight) == {4}
    assert min(strata.df_per_weight) > 4
    assert len(set(strata.strata)) == 2


def test_the_strata_split_does_not_strand_one_source_on_its_own_df():
    """
    Planted with one source 25 times noisier than the rest, which is the case both the
    largest-gap rule and an unconstrained minimum-scatter rule isolate into a stratum of one --
    putting that weight back on df = 4, for the source whose variance is least well determined.
    """
    sources = list(nf.HELD_OUT_SOURCES)
    sigma = {s: 0.004 + 0.0005 * i for i, s in enumerate(sources)}
    sigma[sources[-1]] = 0.10
    values, _, _ = nf.synthetic_baseline(
        n_seeds=5, sigma_by_source=sigma, shared_fraction=0.0, rng_seed=2
    )
    weights = nf.inverse_variance_weights(values[:, -1, :], sources, "strata")
    counts = [weights.strata.count(0), weights.strata.count(1)]
    assert min(counts) >= 2, f"a stratum of one puts a weight back on df = 4: {weights.strata}"


def test_realised_sigma_is_the_composite_taken_directly():
    values, sources, _ = nf.synthetic_baseline(n_seeds=5)
    final = values[:, -1, :]
    weights = np.full(len(sources), 1.0 / len(sources))
    assert nf.realised_sigma(final, weights) == pytest.approx(float(final.mean(axis=1).std(ddof=1)))


def test_weighting_beats_the_unweighted_mean_when_the_sources_differ_by_an_order_of_magnitude():
    """
    Both in sample and out of it. The cross-validated figure is the one that matters, because
    it is the arrangement stage 2 is in -- a frozen vector meeting data it did not see.
    """
    values, sources, _ = nf.synthetic_baseline(n_seeds=5, shared_fraction=0.2, rng_seed=4)
    weights = nf.inverse_variance_weights(values[:, -1, :], sources, "strata")
    assert weights.weighted_sigma < weights.unweighted_sigma
    assert weights.variance_reduction > 1.5
    assert weights.cross_validated_variance_reduction > 1.0


def test_a_shared_seed_effect_puts_a_floor_under_what_any_diagonal_weighting_buys():
    """
    Which is why the realised reduction is reported beside the diagonal one. Weighting cannot
    remove a component every source carries equally, and the gap between the two numbers is
    exactly that component.
    """
    values, sources, _ = nf.synthetic_baseline(n_seeds=5, shared_fraction=0.9, rng_seed=6)
    weights = nf.inverse_variance_weights(values[:, -1, :], sources, "strata")
    assert weights.variance_reduction < weights.diagonal_variance_reduction


def test_weights_refuse_a_shape_that_does_not_match_its_sources():
    values, sources, _ = nf.synthetic_baseline(n_seeds=5)
    with pytest.raises(ValueError):
        nf.inverse_variance_weights(values[:, -1, :], sources[:3], "strata")
    with pytest.raises(ValueError):
        nf.inverse_variance_weights(np.zeros((5, len(sources))), sources, "strata")
    with pytest.raises(ValueError):
        nf.inverse_variance_weights(values[:, -1, :], sources, "gls")


# ---------------------------------------------------------------------------------------
# rho-hat, written before the data that will use it.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("planted", [0.0, 0.3, 0.5, 0.7, 0.9])
def test_the_paired_correlation_recovers_a_planted_rho(planted):
    """
    Averaged over replicates, because at five pairs one draw of a correlation says nothing.
    The tolerance is wide on purpose: a Pearson r at n = 5 carries a real downward bias, which
    is documented on the dataclass and is the reason sigma_delta and not rho is what the
    analysis consumes.
    """
    estimates = [
        nf.paired_correlation(*nf.synthetic_pair(5, 0.010, planted, 0.0, r)).rho_pearson
        for r in range(2000)
    ]
    assert float(np.mean(estimates)) == pytest.approx(planted, abs=0.06)


@pytest.mark.parametrize("planted", [0.0, 0.3, 0.7])
def test_sigma_delta_squared_is_unbiased_for_twice_sigma_squared_one_minus_rho(planted):
    """
    The primitive the paired test actually runs on, and unlike the two rho estimators it has
    no small-sample bias to apologise for.
    """
    truth = 0.010
    squares = [
        nf.paired_correlation(*nf.synthetic_pair(5, truth, planted, 0.0, r)).sigma_delta ** 2
        for r in range(2000)
    ]
    assert float(np.mean(squares)) == pytest.approx(2 * truth**2 * (1 - planted), rel=0.06)


def test_the_two_rho_estimators_agree_when_the_two_arms_have_equal_sample_variances():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    estimate = nf.paired_correlation(a, b)
    assert estimate.rho_pearson == pytest.approx(1.0)
    assert estimate.rho_variance_components == pytest.approx(1.0)


def test_pairing_is_refused_when_the_seeds_do_not_line_up():
    with pytest.raises(ValueError):
        nf.paired_correlation([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        nf.paired_correlation([1.0, 2.0], [1.0, 2.0])


def test_the_paired_standard_error_beats_the_unpaired_one_only_above_a_positive_rho():
    """
    The finding the correction produced. At rho = 0 the block costs df and returns nothing, so
    pairing is a net loss; break-even at five seeds is around 0.09.
    """
    unpaired = nf.mde(0.010, 5, 3, 0.0, paired=False)
    assert nf.mde(0.010, 5, 3, 0.0, paired=True) > unpaired
    assert nf.mde(0.010, 5, 3, 0.09, paired=True) == pytest.approx(unpaired, abs=2e-4)
    assert nf.mde(0.010, 5, 3, 0.3, paired=True) < unpaired


# ---------------------------------------------------------------------------------------
# The MDE solve.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_seeds", [2, 5, 8])
@pytest.mark.parametrize("rho", [0.0, 0.9])
@pytest.mark.parametrize("power", [0.8, 0.95])
def test_the_mde_inverts_its_own_power_function(n_seeds, rho, power):
    effect = nf.mde(0.010, n_seeds, 3, rho, True, power=power)
    se = nf.contrast_se(0.010, n_seeds, rho, True)
    assert nf.power_of(effect, se, nf.error_df(3, n_seeds, True)) == pytest.approx(power, abs=1e-8)


def test_the_mde_is_linear_in_sigma():
    """Which is why a sigma 6% low makes an MDE 6% low, and why c4 is applied."""
    assert nf.mde(0.020, 5, 3, 0.5, True) == pytest.approx(2 * nf.mde(0.010, 5, 3, 0.5, True))


def test_mde_from_applies_the_bias_correction_and_mde_does_not():
    estimate = nf.pooled_sigma([[1.0, 2.0, 3.0, 4.0, 5.0]])
    assert nf.mde_from(estimate, 5, 3, 0.5, True) == pytest.approx(
        nf.mde(estimate.sigma / nf.c4(4), 5, 3, 0.5, True)
    )
    assert nf.mde_from(estimate, 5) > nf.mde(estimate.sigma, 5)


def test_the_noncentral_t_is_not_the_normal_approximation():
    """
    At df = 8 they differ by several percent, which is the same order as the df error. A tool
    that quietly used z would pass every other test in this file.
    """
    from scipy import stats

    exact = nf.mde(0.010, 5, 3, 0.0, True)
    se = nf.contrast_se(0.010, 5, 0.0, True)
    approximate = se * (stats.norm.ppf(0.975) + stats.norm.ppf(0.80))
    assert exact > approximate
    assert exact / approximate == pytest.approx(1.14, abs=0.02)


def test_the_quadrature_agrees_with_scipy_wherever_scipy_is_finite():
    """
    The fallback has to be the *same* distribution function, or it is a second opinion rather
    than a repair. Checked across the df and noncentralities this design reaches.
    """
    from scipy import stats

    compared = 0
    for df in (1, 2, 4, 8, 12):
        nodes, weights = nf._quadrature(df)
        for t in (-4.3, 0.0, 2.5, 4.3):
            for ncp in np.linspace(-10, 10, 11):
                reference = float(stats.nct.cdf(t, df, ncp))
                if not math.isfinite(reference):
                    continue
                quadrature = float(weights @ stats.norm.cdf(t * np.sqrt(nodes / df) - ncp))
                assert quadrature == pytest.approx(reference, abs=1e-6), (df, t, ncp)
                compared += 1
    assert compared > 150


def test_scipy_returns_nan_at_ordinary_arguments_which_is_why_the_fallback_exists():
    """
    Pinned because it is surprising and because the day scipy fixes it this test tells you the
    fallback is now dead code rather than leaving it there forever. The holes are not at the
    extremes: df = 2 at a noncentrality of 7.750 is a NaN and 7.756 is not.
    """
    from scipy import stats

    holes = sum(
        1
        for df in (2, 4, 8)
        for t in (-4.3, 0.0, 4.3)
        for ncp in np.linspace(-10, 10, 21)
        if not math.isfinite(float(stats.nct.cdf(t, df, ncp)))
    )
    assert holes > 0, "scipy no longer needs the fallback; _noncentral_t_cdf can be simplified"


def test_power_survives_the_regime_where_scipy_returns_nothing_usable():
    """
    The failure this replaced: a NaN clamped to 1.0 put spikes into the power curve at
    scattered points, and brentq converged onto one of them and returned a minimum detectable
    effect 0.6% wrong with every appearance of having succeeded.
    """
    assert nf.power_of(1.0, 1e-4, 8) == pytest.approx(1.0)
    assert math.isfinite(nf.mde(1e-6, 5, 3, 0.0, True))

    se, df = 0.01, 2
    curve = [nf.power_of(d, se, df) for d in np.linspace(0.05, 0.12, 400)]
    assert all(math.isfinite(p) for p in curve)
    assert all(b >= a - 1e-9 for a, b in zip(curve, curve[1:])), "power must be monotone in delta"


def test_the_contrast_standard_error_refuses_an_impossible_correlation():
    with pytest.raises(ValueError):
        nf.contrast_se(0.010, 5, 1.0, paired=True)
    with pytest.raises(ValueError):
        nf.contrast_se(0.010, 5, -1.5, paired=True)


# ---------------------------------------------------------------------------------------
# Reading the runs, and refusing to read them too early.
# ---------------------------------------------------------------------------------------


def series(seed: int, per_source):
    return nf.SeedSeries(
        run_name=f"run-{seed}",
        seed=seed,
        arm="baseline",
        state="running",
        last_step=0,
        per_source=per_source,
    )


def test_aligned_matrix_drops_runs_that_logged_no_evaluation():
    """
    The group holds two dead runs from an earlier submission with no history at all, and
    intersecting their empty step sets with everyone else's returns nothing -- which reads as
    "the fan-out has not landed" when in fact three cells have.
    """
    live = {0: {"arxiv": 1.0}, 500: {"arxiv": 0.9}}
    values, steps, seeds = nf.aligned_matrix(
        [series(0, {}), series(1, live), series(2, live)], ["arxiv"]
    )
    assert steps == (0, 500)
    assert seeds == (1, 2)
    assert values.shape == (2, 2, 1)


def test_aligned_matrix_keeps_only_steps_every_run_reached():
    a = {0: {"arxiv": 1.0}, 500: {"arxiv": 0.9}, 1000: {"arxiv": 0.8}}
    b = {0: {"arxiv": 1.1}, 500: {"arxiv": 0.95}}
    values, steps, _ = nf.aligned_matrix([series(0, a), series(1, b)], ["arxiv"])
    assert steps == (0, 500)
    assert values.shape == (2, 2, 1)


def test_aligned_matrix_is_empty_when_nothing_overlaps():
    values, steps, _ = nf.aligned_matrix(
        [series(0, {0: {"arxiv": 1.0}}), series(1, {500: {"arxiv": 1.0}})], ["arxiv"]
    )
    assert steps == ()
    assert values.size == 0


def test_contributing_is_the_same_subset_aligned_matrix_keeps():
    """
    The frozen artifact names the runs it was computed from, and it would be worth nothing if
    that list were assembled by a second copy of this rule that could drift from the first.
    """
    live = {0: {"arxiv": 1.0}, 500: {"arxiv": 0.9}}
    entries = [series(0, {}), series(1, live), series(2, live)]
    _, _, seeds = nf.aligned_matrix(entries, ["arxiv"])
    assert tuple(entry.seed for entry in nf.contributing(entries)) == seeds


# The three cancelled L40S cells and the five live A100 ones, exactly as W&B holds them: one
# experiment slug, two submissions, and a display name that says nothing because the cells of
# a fan-out share it and the cancelled ones were all renamed to '...-died'.
L40S = "run_019fe279-4ef0-7035-9432-4e24d23fba97"
A100 = "run_019fe2f4-f528-70a8-9242-d22f358ede0a"


@pytest.mark.parametrize(
    "run_id, submission, expected",
    [
        (f"{A100}-cell-0", "run_019fe2f4-f528", True),
        (f"{A100}-cell-4", "run_019fe2f4-f528", True),
        (f"{L40S}-cell-0", "run_019fe2f4-f528", False),
        (f"{L40S}-cell-3", "run_019fe279", True),
        ("run_019fdf85-b356-7060-be18-c5fcd4119776", "run_019fe2f4-f528", False),
        # A run that is not a fan-out cell still answers for its own id.
        ("run_019fe008-5877-7048-8078-525707d6ae32", "run_019fe008", True),
        # No submission named is the behaviour from before the argument existed.
        (f"{L40S}-cell-0", None, True),
        (f"{L40S}-cell-0", "", True),
    ],
)
def test_belongs_to_submission_matches_the_id_and_not_the_name(run_id, submission, expected):
    assert nf.belongs_to_submission(run_id, submission) is expected


def test_naming_the_submission_is_what_separates_two_attempts_at_the_same_seeds():
    """
    THE FAILURE THIS EXISTS TO STOP, AND IT IS NOT THE ONE IT LOOKS LIKE. Reading the slug
    whole returns seeds 0, 0, 1, 1, 2, 3, 3, 4 across the two submissions, and the
    distinct-seed refusal in ``main`` fires on it. That refusal is correct and the duplicates
    are real -- but they are two attempts at the same replicate rather than one attempt run
    twice, so the thing at fault is the query. Selecting the submission resolves it to the
    five cells the pre-registration names, and leaves the refusal free to catch the failure it
    was written for.
    """
    whole_group = [f"{L40S}-cell-{i}" for i in (0, 1, 3)] + [f"{A100}-cell-{i}" for i in range(5)]
    selected = [r for r in whole_group if nf.belongs_to_submission(r, "run_019fe2f4-f528")]
    assert selected == [f"{A100}-cell-{i}" for i in range(5)]


def test_sources_come_out_of_the_run_config_through_source_label():
    """
    Read back through the same function that put the labels there, so a config whose paths and
    whose metadata labels disagree is visible rather than silently trusted.
    """
    config = {
        "trainer": {
            "callbacks": {
                "held_out": {
                    "eval_dataset": {
                        "paths": [
                            "s3://b/pretrain/regmix-10b/v1/tokens/wiki/val-00000.u32le.bin",
                            "s3://b/pretrain/regmix-10b/v1/tokens/arxiv/val-00000.u32le.bin",
                        ]
                    }
                }
            }
        }
    }
    assert nf.sources_from_config(config) == ("arxiv", "wiki")
    assert nf.sources_from_config({}) == nf.HELD_OUT_SOURCES


def test_sources_fall_back_to_the_metadata_labels_when_there_are_no_paths():
    config = {
        "trainer": {
            "callbacks": {
                "held_out": {"eval_dataset": {"metadata": [{"label": "dclm"}, {"label": "wiki"}]}}
            }
        }
    }
    assert nf.sources_from_config(config) == ("dclm", "wiki")


@pytest.mark.parametrize(
    "n_seeds, final_step, expected",
    [(5, 6000, 0), (3, 6000, 1), (5, 0, 1), (5, 500, 1), (3, 0, 2), (2, 0, 2)],
)
def test_provisional_reasons_name_every_way_a_reading_is_not_the_answer(
    n_seeds, final_step, expected
):
    """
    Ten minutes after admission every cell has one evaluation, of an untrained model, and a
    sigma over those is a real number that means nothing. It has to announce itself.
    """
    assert len(nf.provisional_reasons(n_seeds, final_step, 6000)) == expected


def test_the_arm_is_recovered_from_the_model_and_not_from_a_label():
    """
    No ``--arm`` string reaches the saved config, so a run is identified by what it is. That
    also means a mislabelled run cannot be counted into the wrong arm's noise floor.
    """
    assert nf._arm_of({}) == "baseline"
    assert nf._arm_of({"model": {"block": {}}}) == "baseline"
    assert (
        nf._arm_of({"model": {"block": {"hyper_connections": {"mode": "output"}}}}) == "output-only"
    )
    assert nf._arm_of({"model": {"block": {"hyper_connections": {"mode": "full"}}}}) == "faithful"


# ---------------------------------------------------------------------------------------
# The synthetic generator, which every test above rests on.
# ---------------------------------------------------------------------------------------


def test_the_generator_plants_the_per_source_sigmas_it_claims_to():
    """Averaged over many draws, since five seeds resolve nothing on their own."""
    sources = nf.HELD_OUT_SOURCES
    stacked = np.stack(
        [
            nf.synthetic_baseline(n_seeds=5, shared_fraction=0.0, rng_seed=r)[0][:, -1, :]
            for r in range(400)
        ]
    )
    measured = np.sqrt(np.mean(stacked.var(axis=1, ddof=1), axis=0))
    planted = np.array([0.0035, 0.0060, 0.0240, 0.0180, 0.0048, 0.0300, 0.0042])
    lookup = dict(
        zip(
            ["dclm", "arxiv", "algebraic-stack", "open-web-math", "pes2o", "starcoder", "wiki"],
            planted,
        )
    )
    for j, source in enumerate(sources):
        assert measured[j] == pytest.approx(lookup[source], rel=0.10), source


def test_the_shared_seed_effect_correlates_the_sources():
    independent, sources, _ = nf.synthetic_baseline(n_seeds=200, shared_fraction=0.0, rng_seed=1)
    shared, _, _ = nf.synthetic_baseline(n_seeds=200, shared_fraction=0.9, rng_seed=1)

    def mean_off_diagonal(values):
        correlation = np.corrcoef(values[:, -1, :], rowvar=False)
        return float(correlation[np.triu_indices(len(sources), 1)].mean())

    assert abs(mean_off_diagonal(independent)) < 0.10
    assert mean_off_diagonal(shared) > 0.20
    assert mean_off_diagonal(shared) > 3 * abs(mean_off_diagonal(independent))


def test_the_self_test_passes():
    """
    The thing a person runs before trusting any of this, run as a test as well so that it
    cannot rot. Fewer replicates than the command-line default, which widens its own Monte
    Carlo tolerances to match rather than becoming a stricter test on a smaller sample.
    """
    assert nf.self_test(replicates=800) == 0
