import pytest

from olmo_core.hpo.bttackler import (
    BTTCalibrationProfile,
    BTTConfig,
    BTTDiagnoser,
    BTTMode,
    BTTObservation,
)
from olmo_core.hpo.types import BTTDisposition, BTTVerdictKind


def _obs(**over):
    base = dict(
        trial_id="t0",
        completed_fidelity=10_000,
        observation_hash="h",
        grad_norm_history=tuple([1.0] * 8),
        loss_history=tuple([4.0 - 0.1 * i for i in range(8)]),  # steadily decreasing
        activation_ratio=0.5,
        non_finite=False,
    )
    base.update(over)
    return BTTObservation(**base)


def _diag(**cfg):
    return BTTDiagnoser(BTTConfig(min_fidelity=1000, **cfg))


def test_healthy_when_learning_normally():
    v = _diag().diagnose(_obs())
    assert v.kind is BTTVerdictKind.HEALTHY
    assert v.profile_version == "btt-v1"
    assert v.binding_key == ("t0", 10_000, "h")


def test_agv_non_finite_is_fatal_and_bypasses_min_fidelity():
    # Non-finite training is always fatal, even below the minimum-fidelity gate.
    v = _diag().diagnose(_obs(completed_fidelity=10, non_finite=True))
    assert v.kind is BTTVerdictKind.FATAL
    assert "AGV" in v.indicators


def test_eag_exponentially_amplified_gradients_degraded():
    hist = tuple([1.0, 1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 40.0])  # blowing up
    v = _diag().diagnose(_obs(grad_norm_history=hist))
    assert v.kind is BTTVerdictKind.DEGRADED
    assert "EAG" in v.indicators


def test_erg_exponentially_reduced_gradients_degraded():
    hist = tuple([10.0, 8.0, 5.0, 2.0, 1.0, 0.4, 0.1, 0.02])  # vanishing
    v = _diag().diagnose(_obs(grad_norm_history=hist))
    assert v.kind is BTTVerdictKind.DEGRADED
    assert "ERG" in v.indicators


def test_short_gradient_history_uses_disjoint_windows():
    amplified = _diag(window=4).diagnose(_obs(grad_norm_history=(1.0, 1.0, 1.0, 40.0)))
    reduced = _diag(window=4).diagnose(_obs(grad_norm_history=(40.0, 1.0, 1.0, 1.0)))
    assert "EAG" in amplified.indicators
    assert "ERG" in reduced.indicators


def test_plc_passive_loss_is_degraded():
    flat = tuple([4.0] * 8)  # never learned
    v = _diag().diagnose(_obs(loss_history=flat))
    assert v.kind is BTTVerdictKind.DEGRADED
    assert "PLC" in v.indicators


def test_ulc_loss_spike_is_degraded():
    spike = tuple([4.0, 3.5, 3.0, 2.8, 2.7, 2.6, 3.5, 5.0])  # diverged at the end
    v = _diag().diagnose(_obs(loss_history=spike))
    assert v.kind is BTTVerdictKind.DEGRADED
    assert "ULC" in v.indicators


def test_ulc_uses_recent_preterminal_baseline_and_plc_excludes_deterioration():
    old_low_outlier = _diag(window=4).diagnose(_obs(loss_history=(4.0, 1.0, 2.0, 1.8, 1.6, 1.5)))
    assert "ULC" not in old_low_outlier.indicators

    worsening = _diag(window=4).diagnose(_obs(loss_history=(4.0, 3.0, 2.0, 2.5, 5.0)))
    assert "ULC" in worsening.indicators
    assert "PLC" not in worsening.indicators


def test_lar_low_activation_ratio_degraded_only_when_available():
    v = _diag().diagnose(_obs(activation_ratio=0.01))
    assert v.kind is BTTVerdictKind.DEGRADED
    assert "LAR" in v.indicators
    # When activation telemetry is absent, LAR simply does not apply.
    v2 = _diag().diagnose(_obs(activation_ratio=None))
    assert "LAR" not in v2.indicators


def test_nmg_plateau_is_saturated_and_incumbent():
    # Learned, then flat over the final window.
    curve = tuple([4.0, 3.0, 2.5, 2.2, 2.19, 2.185, 2.184, 2.1839])
    v = _diag().diagnose(_obs(loss_history=curve))
    assert v.kind is BTTVerdictKind.SATURATED
    assert "NMG" in v.indicators
    assert v.is_incumbent_candidate() is True


def test_min_fidelity_gate_protects_early_trials():
    # A degraded signal below min fidelity is suppressed (not enough evidence yet).
    v = _diag().diagnose(_obs(completed_fidelity=10, loss_history=tuple([4.0] * 8)))
    assert v.kind is BTTVerdictKind.HEALTHY


def test_late_bloomer_reserve_spares_would_be_degraded():
    # reserve=1.0 spares every would-be termination deterministically.
    spared = BTTDiagnoser(BTTConfig(min_fidelity=1000, late_bloomer_reserve=1.0)).diagnose(
        _obs(loss_history=tuple([4.0] * 8))
    )
    assert spared.kind is BTTVerdictKind.HEALTHY
    assert spared.spared_by_reserve is True
    # reserve=0.0 does not spare.
    cut = _diag(late_bloomer_reserve=0.0).diagnose(_obs(loss_history=tuple([4.0] * 8)))
    assert cut.kind is BTTVerdictKind.DEGRADED


def test_fatal_is_never_spared_by_reserve():
    v = BTTDiagnoser(BTTConfig(min_fidelity=1000, late_bloomer_reserve=1.0)).diagnose(
        _obs(non_finite=True)
    )
    assert v.kind is BTTVerdictKind.FATAL


def test_paper_binary_vs_adapted_modes_agree_on_nmg_candidacy():
    curve = tuple([4.0, 3.0, 2.5, 2.2, 2.19, 2.185, 2.184, 2.1839])
    paper = BTTDiagnoser(BTTConfig(min_fidelity=1000, mode=BTTMode.PAPER_BINARY)).diagnose(
        _obs(loss_history=curve)
    )
    adapted = BTTDiagnoser(BTTConfig(min_fidelity=1000, mode=BTTMode.ADAPTED_RECYCLE)).diagnose(
        _obs(loss_history=curve)
    )
    # NMG "stops but preserves candidacy" in both arms.
    assert paper.kind is BTTVerdictKind.SATURATED
    assert adapted.kind is BTTVerdictKind.SATURATED
    assert paper.is_incumbent_candidate() and adapted.is_incumbent_candidate()


def test_paper_binary_stops_while_adapted_mode_recycles_degraded_trial():
    flat = _obs(loss_history=tuple([4.0] * 8))
    paper = BTTDiagnoser(BTTConfig(min_fidelity=1000, mode=BTTMode.PAPER_BINARY)).diagnose(flat)
    adapted = BTTDiagnoser(BTTConfig(min_fidelity=1000, mode=BTTMode.ADAPTED_RECYCLE)).diagnose(
        flat
    )
    assert paper.disposition is BTTDisposition.STOP
    assert adapted.disposition is BTTDisposition.RECYCLE


def test_calibration_profile_is_required_and_applied_when_configured():
    config = BTTConfig(min_fidelity=1000, require_calibration=True)
    with pytest.raises(ValueError):
        BTTDiagnoser(config)
    profile = BTTCalibrationProfile(
        profile_version="completed-v2",
        completed_run_ids=("cal-1", "cal-2"),
        thresholds={"plc_min_rel_improve": 0.01},
    )
    diagnoser = BTTDiagnoser(config, calibration=profile)
    assert diagnoser.config.profile_version == "completed-v2"
    assert diagnoser.config.plc_min_rel_improve == pytest.approx(0.01)


def test_same_fidelity_top_trial_is_protected_with_counterfactual_evidence():
    diagnoser = _diag(same_fidelity_top_fraction=0.5)
    observations = [
        _obs(trial_id="best", loss_history=tuple([4.0] * 8)),
        _obs(trial_id="worst", loss_history=tuple([4.0] * 8)),
    ]
    verdicts = diagnoser.diagnose_cohort(observations, scores={"best": 0.9, "worst": 0.1})
    assert verdicts["best"].kind is BTTVerdictKind.HEALTHY
    assert verdicts["best"].protected_by_peer_rank is True
    assert "PLC" in verdicts["best"].indicators
    assert verdicts["worst"].kind is BTTVerdictKind.DEGRADED


def test_calibration_profile_detaches_thresholds_and_config_rejects_invalid_ranges():
    thresholds = {"eag_ratio": 5.0}
    run_ids = ["run-1"]
    profile = BTTCalibrationProfile(
        profile_version="v1",
        completed_run_ids=run_ids,
        thresholds=thresholds,
    )
    thresholds["eag_ratio"] = -1.0
    run_ids.append("run-2")
    assert profile.thresholds["eag_ratio"] == 5.0
    assert profile.completed_run_ids == ("run-1",)

    for kwargs in (
        {"agv_max_grad_norm": 0.0},
        {"eag_ratio": 1.0},
        {"erg_ratio": 1.0},
        {"erg_ratio": 0.0},
        {"ulc_spike_ratio": 1.0},
        {"lar_min_active": 1.1},
        {"plc_min_rel_improve": -0.1},
        {"nmg_min_rel_improve": 1.1},
    ):
        with pytest.raises(ValueError):
            BTTConfig(**kwargs)
