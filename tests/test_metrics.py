"""Divergence, noise-floor and compute-accounting properties."""

import math

import numpy as np
import pytest

from memsplit import metrics
from memsplit.metrics import Crossing


def _dists(n=200, v=64, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(v) * 0.5, size=n)
    q = rng.dirichlet(np.ones(v) * 0.5, size=n)
    return p, q


def test_jsd_is_bounded_by_ln2():
    """The bound is the point: a saturated value is interpretable, 6.7 nats isn't."""
    p, q = _dists()
    j = metrics.jsd(p, q)
    assert (j >= 0).all()
    assert (j <= metrics.LN2 + 1e-9).all(), j.max()

    # Disjoint supports saturate exactly at ln 2.
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([[0.0, 1.0, 0.0]])
    assert metrics.jsd(a, b)[0] == pytest.approx(metrics.LN2, abs=1e-6)


def test_jsd_is_symmetric_and_zero_on_identity():
    p, q = _dists()
    assert np.allclose(metrics.jsd(p, q), metrics.jsd(q, p))
    assert np.allclose(metrics.jsd(p, p), 0.0, atol=1e-9)


def test_kl_is_asymmetric_so_direction_must_be_stated():
    p, q = _dists()
    assert not np.allclose(metrics.kl(p, q), metrics.kl(q, p))


def test_kl_blows_up_where_jsd_saturates():
    """Explains the 6.7-nat figure: near-disjointness, not a large 'distance'."""
    a = np.array([[1.0, 1e-30, 1e-30]])
    b = np.array([[1e-30, 1.0, 1e-30]])
    assert metrics.kl(a, b)[0] > 10.0
    assert metrics.jsd(a, b)[0] == pytest.approx(metrics.LN2, abs=1e-6)


def test_divergence_report_stratifies_and_gives_tails_not_just_means():
    p, q = _dists(n=120)
    roles = ["plain"] * 60 + ["payload"] * 60
    # Make payload positions genuinely divergent.
    q[60:] = np.roll(q[60:], 7, axis=1)
    rep = metrics.divergence_report(p, q, roles=roles)
    assert rep["all"]["n"] == 120
    for key in ("role:plain", "role:payload"):
        assert {"mean", "median", "p90", "p99", "max"} <= set(rep[key]["jsd"])
    assert rep["jsd_ceiling_nats"] == pytest.approx(math.log(2))


def test_rank_shift_classes_use_the_published_thresholds():
    ranks = np.array([1, 1, 2, 3, 4, 100])
    cls = metrics.rank_shift_classes(ranks)
    assert cls["unshifted"] == pytest.approx(2 / 6)
    assert cls["marginal"] == pytest.approx(2 / 6)
    assert cls["shifted"] == pytest.approx(2 / 6)
    assert cls["unshifted_or_marginal"] == pytest.approx(4 / 6)


def test_bits_per_byte_conversion_lands_on_the_interpretive_scale():
    """0.08 nats/token at ~4 bytes/token is ~0.029 bits/byte -- below the ~0.1 floor."""
    bpb = metrics.nats_to_bits_per_byte(0.08, n_bytes=4000, n_tokens=1000)
    assert 0.02 < bpb < 0.04, bpb
    assert bpb < 0.1, "this is the point: it sits under the across-seed floor"


def test_seed_floor_flags_an_indistinguishable_arm_difference():
    """H2 as previously stated may be absence of evidence. Make that explicit."""
    weak = metrics.seed_floor_report(0.03, [0.028, 0.031, 0.030])
    assert not weak.distinguishable
    assert weak.ratio < 1.2

    strong = metrics.seed_floor_report(6.7, [0.028, 0.031, 0.030])
    assert strong.distinguishable
    assert strong.ratio > 100


def test_checkpoint_noise_floor_is_free_and_gives_an_mde():
    vals = [0.310, 0.313, 0.309, 0.315, 0.311, 0.312, 0.308, 0.314]
    out = metrics.checkpoint_noise_floor(vals)
    assert out["n_checkpoints"] == 8
    assert out["relative_sd"] > 0
    assert out["implied_mde_pp_at_80_power"] > 0
    with pytest.raises(ValueError):
        metrics.checkpoint_noise_floor([0.3, 0.31])


def test_compute_to_threshold_refuses_an_unbracketed_crossing():
    """This is the '10-15x' failure: first point already at ceiling."""
    c = metrics.compute_to_threshold(
        steps=[47, 94, 141, 940],
        accuracy=[0.998, 1.0, 1.0, 0.998],
        threshold=0.90,
        tokens_per_step=524288,
        flops_per_token=1.0,
    )
    assert not c.bracketed
    assert "censored lower bound" in c.note
    assert c.tokens is None


def test_compute_to_threshold_interpolates_a_real_crossing():
    c = metrics.compute_to_threshold(
        steps=[190, 380, 570, 760, 950],
        accuracy=[0.006, 0.005, 0.403, 0.761, 0.807],
        threshold=0.80,
        tokens_per_step=524288,
        flops_per_token=2.0,
    )
    assert c.bracketed
    assert 760 < c.steps < 950
    assert c.tokens == pytest.approx(c.steps * 524288)
    assert c.flops == pytest.approx(c.tokens * 2.0)


def test_threshold_ratio_refuses_when_either_side_is_unbracketed():
    good = metrics.compute_to_threshold(
        [1, 2, 4, 8], [0.0, 0.2, 0.6, 0.95], 0.8, 100.0, 1.0
    )
    bad = Crossing(0.8, False, None, None, None, note="unbracketed")
    out = metrics.threshold_ratio(bad, good)
    assert out["usable"] is False
    ok = metrics.threshold_ratio(good, good)
    assert ok["usable"] and ok["tokens_ratio"] == pytest.approx(1.0)


def test_inference_overhead_surfaces_the_mfu_asymmetry():
    out = metrics.inference_overhead(20.0, 60.0)
    assert out["answer_token_ratio"] == pytest.approx(3.0)
    assert out["train_to_inference_mfu_penalty"] == pytest.approx(50.0)
    assert "not" in out["note"]
