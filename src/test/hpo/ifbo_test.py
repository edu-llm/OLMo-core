import numpy as np
import pytest

from olmo_core.hpo.ftpfn import ObservedCurve, PosteriorInput
from olmo_core.hpo.ifbo import (
    Candidate,
    IfBOCandidateGenerator,
    MFPIRandom,
    observed_f_best,
)
from olmo_core.hpo.objective import CENormalizer
from olmo_core.hpo.types import CurvePoint, ProposalSource


class _TPosterior:
    """Deterministic fake: PI increases with the query fidelity coordinate."""

    def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
        return np.clip(x.query_t, 0.0, 1.0)


class _ConstPosterior:
    def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
        return np.full(x.query_hp.shape[0], 0.5, dtype=np.float64)


def _norm():
    return CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)


def _observed():
    return [
        ObservedCurve(1, (0.2,), (CurvePoint(1024, 5.0), CurvePoint(2048, 4.0))),
        ObservedCurve(2, (0.6,), (CurvePoint(1024, 4.5),)),
    ]


def _candidates():
    return [
        Candidate(
            "t1", 1, (0.2,), base_tokens=2048, is_continuation=True, source=ProposalSource.CMA
        ),
        Candidate(
            "t2", 2, (0.6,), base_tokens=1024, is_continuation=True, source=ProposalSource.CMA
        ),
        Candidate(
            "new-a",
            0,
            (0.9,),
            base_tokens=1024,
            is_continuation=False,
            source=ProposalSource.RANDOM,
        ),
    ]


def test_observed_f_best_uses_observed_points_only():
    # Best (lowest) CE observed is 4.0 -> y = 0.5.
    assert observed_f_best(_observed(), _norm()) == pytest.approx(0.5)


def test_threshold_uses_observed_f_best_and_log_uniform_tau():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    f_best = observed_f_best(_observed(), _norm())
    sel = mfpi.select(_observed(), _candidates(), rng=np.random.default_rng(0), f_best=f_best)
    tau = (sel.threshold - f_best) / (1.0 - f_best)
    assert 1e-4 <= tau <= 1e-1  # log-uniform in [10^-4, 10^-1]
    assert 1 <= sel.horizon <= 4  # inclusive horizon range


def test_selection_is_deterministic_for_a_seed():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    f_best = observed_f_best(_observed(), _norm())
    a = mfpi.select(_observed(), _candidates(), rng=np.random.default_rng(123), f_best=f_best)
    b = mfpi.select(_observed(), _candidates(), rng=np.random.default_rng(123), f_best=f_best)
    assert a.chosen_index == b.chosen_index
    assert a.horizon == b.horizon
    assert a.threshold == pytest.approx(b.threshold)
    assert np.allclose(a.scores, b.scores)


def test_ties_break_deterministically_by_candidate_key():
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    cands = _candidates()
    sel = mfpi.select(_observed(), cands, rng=np.random.default_rng(7), f_best=0.5)
    # All PI equal -> smallest key wins ("new-a" < "t1" < "t2").
    assert cands[sel.chosen_index].key == "new-a"


def test_horizon_advances_query_time_and_clamps_at_one():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    # A continuation already at the target token ceiling stays clamped at t=1.0.
    cand = [
        Candidate(
            "t1", 1, (0.2,), base_tokens=4096, is_continuation=True, source=ProposalSource.CMA
        )
    ]
    sel = mfpi.select(_observed(), cand, rng=np.random.default_rng(1), f_best=0.5)
    assert sel.scores[0] == pytest.approx(1.0)


def test_constructor_rejects_invalid_target_and_horizon_bounds():
    with pytest.raises(ValueError):
        MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=0, normalizer=_norm())
    for bounds in ((0, 1), (2, 1), (1, 5)):
        with pytest.raises(ValueError):
            MFPIRandom(
                _ConstPosterior(),
                n_fidelity_bins=4,
                target_tokens=4096,
                normalizer=_norm(),
                horizon_bounds=bounds,
            )


@pytest.mark.parametrize(
    "scores",
    (
        np.array([float("nan"), 0.9, 0.8]),
        np.array([float("inf"), 0.9, 0.8]),
        np.array([-0.1, 0.9, 0.8]),
        np.array([1.1, 0.9, 0.8]),
    ),
)
def test_select_rejects_nonfinite_or_out_of_range_pi(scores):
    class InvalidPosterior:
        def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
            return scores

    mfpi = MFPIRandom(InvalidPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    with pytest.raises(ValueError):
        mfpi.select(_observed(), _candidates(), rng=np.random.default_rng(0), f_best=0.5)


def test_select_batch_picks_distinct_candidates_and_fantasizes_pending():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    observed = _observed()
    picks = mfpi.select_batch(
        observed, _candidates(), count=2, rng=np.random.default_rng(0), f_best=0.5
    )
    assert len(picks) == 2
    # A worker slot cannot be assigned the same candidate twice in one batch.
    assert picks[0].chosen_index != picks[1].chosen_index
    # chosen_index refers back into the original candidate list.
    assert all(0 <= p.chosen_index < 3 for p in picks)


def test_select_batch_is_deterministic_for_a_seed():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    a = mfpi.select_batch(
        _observed(), _candidates(), count=3, rng=np.random.default_rng(5), f_best=0.5
    )
    b = mfpi.select_batch(
        _observed(), _candidates(), count=3, rng=np.random.default_rng(5), f_best=0.5
    )
    assert [p.chosen_index for p in a] == [p.chosen_index for p in b]


def test_new_fantasy_id_avoids_all_observed_curve_ids():
    class RecordingPosterior:
        def __init__(self):
            self.calls = []

        def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
            self.calls.append(x)
            return np.full(len(x.query_t), 0.5)

    posterior = RecordingPosterior()
    mfpi = MFPIRandom(posterior, n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    observed = [ObservedCurve(1, (0.2,), (CurvePoint(1024, 4.0),))]
    candidates = [
        Candidate("new-a", 0, (0.8,), 1024, False, ProposalSource.RANDOM),
        Candidate("new-b", 0, (0.9,), 1024, False, ProposalSource.RANDOM),
    ]
    mfpi.select_batch(observed, candidates, count=2, rng=np.random.default_rng(0), f_best=0.5)
    assert posterior.calls[1].context_ids.tolist() == [1, 2]


def test_select_batch_rejects_duplicate_candidate_keys():
    candidates = [
        Candidate("same", 0, (0.1,), 1024, False, ProposalSource.RANDOM),
        Candidate("same", 0, (0.9,), 1024, False, ProposalSource.RANDOM),
    ]
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    with pytest.raises(ValueError):
        mfpi.select_batch(
            _observed(), candidates, count=2, rng=np.random.default_rng(0), f_best=0.5
        )


def test_select_batch_rejects_duplicate_new_configs_before_scoring():
    class RecordingPosterior(_ConstPosterior):
        def __init__(self):
            self.calls = 0

        def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
            self.calls += 1
            return super().pi(x, threshold)

    posterior = RecordingPosterior()
    candidates = [
        Candidate("a", 0, (0.5,), 1024, False, ProposalSource.RANDOM),
        Candidate("b", 0, (0.5,), 1024, False, ProposalSource.RANDOM),
    ]
    mfpi = MFPIRandom(posterior, n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    with pytest.raises(ValueError):
        mfpi.select_batch(
            _observed(), candidates, count=2, rng=np.random.default_rng(0), f_best=0.5
        )
    assert posterior.calls == 0


def test_select_batch_allows_duplicate_config_only_with_shared_positive_curve_id():
    candidates = [
        Candidate("donor", 1, (0.2,), 2048, True, ProposalSource.CMA),
        Candidate("pure-copy", 1, (0.2,), 2048, False, ProposalSource.IPBT_META),
    ]
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    picks = mfpi.select_batch(
        _observed(), candidates, count=2, rng=np.random.default_rng(0), f_best=0.5
    )
    assert len(picks) == 2


def test_select_batch_rejects_invalid_fantasy_y_before_scoring():
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    for fantasy_y in (float("nan"), -0.1, 1.1):
        with pytest.raises(ValueError):
            mfpi.select_batch(
                _observed(),
                _candidates(),
                count=2,
                rng=np.random.default_rng(0),
                f_best=0.5,
                fantasy_y=fantasy_y,
            )


def test_context_capacity_has_deterministic_retention_policy():
    observed = [
        ObservedCurve(
            curve_id=index + 1,
            unit_config=(index / 1000.0,),
            points=(CurvePoint(tokens=1024 + index, ce=4.0),),
        )
        for index in range(1000)
    ]
    candidates = [
        Candidate("new-a", 0, (0.9995,), 1024, False, ProposalSource.RANDOM),
        Candidate("new-b", 0, (1.0,), 1024, False, ProposalSource.RANDOM),
    ]
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    picks = mfpi.select_batch(
        observed, candidates, count=2, rng=np.random.default_rng(0), f_best=0.5
    )
    assert len(picks) == 2


def test_context_capacity_retains_curves_needed_by_resume_queries():
    observed = [
        ObservedCurve(
            curve_id=index + 1,
            unit_config=(index / 1000.0,),
            points=(CurvePoint(tokens=1024 + index, ce=4.0),),
        )
        for index in range(1000)
    ]
    candidates = [
        Candidate("resume-oldest", 1, (0.0,), 1024, True, ProposalSource.IFBO),
        Candidate("new", 0, (0.9995,), 1024, False, ProposalSource.IFBO),
    ]
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    assert (
        len(
            mfpi.select_batch(
                observed,
                candidates,
                count=1,
                rng=np.random.default_rng(0),
                f_best=0.5,
            )
        )
        == 1
    )


def test_capacity_reindexes_retained_global_curve_ids():
    observed = [
        ObservedCurve(
            curve_id=index + 2,
            unit_config=(index / 1000.0,),
            points=(CurvePoint(tokens=1024 + index, ce=4.0),),
        )
        for index in range(1000)
    ]
    candidate = Candidate(
        "resume-high-id",
        1001,
        (0.999,),
        2048,
        True,
        ProposalSource.IFBO,
    )
    mfpi = MFPIRandom(_ConstPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    assert (
        mfpi.select_batch(
            observed,
            [candidate],
            count=1,
            rng=np.random.default_rng(0),
            f_best=0.5,
        )[0].chosen_index
        == 0
    )


def test_full_context_single_pick_scores_resume_and_new_slate():
    observed = [
        ObservedCurve(
            curve_id=index + 1,
            unit_config=(index / 1000.0,),
            points=(CurvePoint(tokens=1024, ce=4.0),),
        )
        for index in range(1000)
    ]
    candidates = [
        Candidate(
            f"resume-{index}",
            index + 1,
            (index / 1000.0,),
            1024,
            True,
            ProposalSource.IFBO,
        )
        for index in range(1000)
    ] + [Candidate("new", 0, (0.9995,), 1024, False, ProposalSource.IFBO)]
    mfpi = MFPIRandom(
        _ConstPosterior(),
        n_fidelity_bins=4,
        target_tokens=4096,
        normalizer=_norm(),
    )
    assert (
        len(
            mfpi.select_batch(
                observed,
                candidates,
                count=1,
                rng=np.random.default_rng(0),
                f_best=0.5,
            )
        )
        == 1
    )


def test_remapped_resume_ids_leave_fantasy_capacity():
    observed = [
        ObservedCurve(
            curve_id=index + 2,
            unit_config=(index / 1000.0,),
            points=(CurvePoint(tokens=1024, ce=4.0),),
        )
        for index in range(999)
    ]
    candidates = [
        Candidate("resume-a", 1000, (0.998,), 1024, True, ProposalSource.IFBO),
        Candidate("resume-b", 999, (0.997,), 1024, True, ProposalSource.IFBO),
        Candidate("new", 0, (0.9995,), 1024, False, ProposalSource.IFBO),
    ]
    mfpi = MFPIRandom(
        _ConstPosterior(),
        n_fidelity_bins=4,
        target_tokens=4096,
        normalizer=_norm(),
    )
    assert (
        len(
            mfpi.select_batch(
                observed,
                candidates,
                count=2,
                rng=np.random.default_rng(0),
                f_best=0.5,
            )
        )
        == 2
    )


def test_ifbo_candidate_generator_is_seeded_bounded_and_replayable():
    first = IfBOCandidateGenerator(ndim=3, seed=7)
    initial = first.ask(8)
    assert initial[0] == (0.0, 0.0, 0.0)
    assert initial[1] == (1.0, 1.0, 1.0)
    assert all(
        len(config) == 3 and all(0.0 <= value <= 1.0 for value in config) for config in initial
    )
    state = first.state_dict()
    expected = first.ask(4, incumbent=(0.5, 0.5, 0.5))

    restored = IfBOCandidateGenerator(ndim=3, seed=999)
    restored.load_state_dict(state)
    assert restored.ask(4, incumbent=(0.5, 0.5, 0.5)) == expected
    assert restored.proposal_source is ProposalSource.IFBO


def test_ifbo_candidate_generator_never_repeats_across_rounds():
    generator = IfBOCandidateGenerator(ndim=2, seed=11)
    first = set(generator.ask(16))
    second = set(generator.ask(16, incumbent=(0.5, 0.5)))
    assert first.isdisjoint(second)
    restored = IfBOCandidateGenerator(ndim=2, seed=0)
    restored.load_state_dict(generator.state_dict())
    third = set(restored.ask(16, incumbent=(0.5, 0.5)))
    assert third.isdisjoint(first | second)


def test_batch_selection_scores_keep_original_candidate_identity():
    mfpi = MFPIRandom(_TPosterior(), n_fidelity_bins=4, target_tokens=4096, normalizer=_norm())
    picks = mfpi.select_batch(
        _observed(), _candidates(), count=2, rng=np.random.default_rng(0), f_best=0.5
    )
    for pick in picks:
        score_position = pick.score_indices.index(pick.chosen_index)
        assert pick.scores[score_position] == pytest.approx(pick.mfpi_score)
