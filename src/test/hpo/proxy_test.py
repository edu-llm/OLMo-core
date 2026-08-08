import pytest

from olmo_core.hpo.proxy import (
    AdmitDecision,
    ExactTokenScreen,
    FrozenLayerProxy,
    ProxyAdmission,
    ProxyKind,
    ProxyMetrics,
    UMuPArm,
    lcb,
    output_suffix_freeze_patterns,
    rank_correlation,
    top_k_recall,
)


def test_exact_token_screen_is_the_mandatory_first_fidelity():
    screen = ExactTokenScreen(tokens=10_000)
    assert screen.is_mandatory_baseline is True
    assert screen.fidelity_rank == 0
    with pytest.raises(TypeError):
        ExactTokenScreen(tokens=10_000, is_mandatory_baseline=False)


def test_rank_correlation_bounds():
    assert rank_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_rank_correlation_is_tie_aware_and_rejects_invalid_inputs():
    assert rank_correlation([1, 1, 2], [1, 2, 2]) == pytest.approx(0.5)
    for a, b in (
        ([1, 1, 1], [1, 2, 3]),
        ([1, float("nan"), 3], [1, 2, 3]),
    ):
        with pytest.raises(ValueError):
            rank_correlation(a, b)


def test_top_k_recall():
    # full best-3 are ids 0,1,2; proxy recovers 2 of them in its top 3.
    full_order = [0, 1, 2, 3, 4]
    proxy_order = [0, 1, 3, 2, 4]
    assert top_k_recall(proxy_order, full_order, k=3) == pytest.approx(2 / 3)


def test_lcb_penalizes_variance():
    assert lcb(mean=0.8, std=0.0, n=100) == pytest.approx(0.8)
    assert lcb(mean=0.8, std=0.3, n=10) < 0.8


def test_frozen_layer_proxy_requires_full_retrain_and_freezes_prefix():
    proxy = FrozenLayerProxy(n_layers=12, train_last_k=2)
    assert proxy.requires_full_retrain is True
    with pytest.raises(TypeError):
        FrozenLayerProxy(n_layers=12, train_last_k=2, requires_full_retrain=False)
    patterns = output_suffix_freeze_patterns(n_layers=12, train_last_k=2)
    # The last two blocks are trainable; blocks 0..9 and embeddings are frozen.
    assert "blocks.0.*" in patterns
    assert "blocks.9.*" in patterns
    assert "blocks.10.*" not in patterns
    assert "blocks.11.*" not in patterns
    assert any(p.startswith("embeddings") for p in patterns)
    assert "embedding_norm.*" in patterns


def test_umup_arm_forbids_depth_reduction():
    UMuPArm(width_factor=0.5, depth_factor=1.0)  # width-reduced, same depth: ok
    with pytest.raises(ValueError):
        UMuPArm(width_factor=0.5, depth_factor=0.5)  # depth reduction is the weakest transfer axis
    with pytest.raises(TypeError):
        UMuPArm(width_factor=0.5, depth_factor=1.0, validate_parity_first=False)
    assert UMuPArm(width_factor=0.5, depth_factor=1.0).validate_parity_first is True


def test_admission_requires_beating_exact_screen_at_equal_budget():
    gate = ProxyAdmission(min_rank_corr=0.7, min_top_k_recall=0.6)
    good = ProxyMetrics(
        rank_corr_mean=0.9,
        rank_corr_std=0.05,
        top_k_recall=0.8,
        n=50,
        net_compute_savings=0.4,
        beats_exact_at_equal_budget=True,
        top_k_recall_std=0.05,
        proxy_kind=ProxyKind.UMUP,
        parity_validated=True,
    )
    assert gate.decide(good) is AdmitDecision.PRUNE_PROMOTE

    # Fails if it does not beat exact-model screening at equal budget.
    not_better = ProxyMetrics(0.9, 0.05, 0.8, 50, 0.4, False, top_k_recall_std=0.05)
    assert gate.decide(not_better) is AdmitDecision.REPORTING_ONLY

    # Fails if rank-correlation LCB is below threshold.
    noisy = ProxyMetrics(0.75, 0.4, 0.8, 5, 0.4, True, top_k_recall_std=0.05)
    assert gate.decide(noisy) is AdmitDecision.REPORTING_ONLY

    # Fails if there are no net compute savings.
    no_savings = ProxyMetrics(0.9, 0.05, 0.8, 50, -0.1, True, top_k_recall_std=0.05)
    assert gate.decide(no_savings) is AdmitDecision.REPORTING_ONLY


def test_admission_rejects_invalid_uncertainty_insufficient_samples_and_missing_parity():
    gate = ProxyAdmission(min_rank_corr=0.7, min_top_k_recall=0.6, min_samples=10)
    with pytest.raises(ValueError):
        gate.decide(ProxyMetrics(0.9, -1.0, 0.8, 50, 0.4, True))

    one_sample = ProxyMetrics(0.9, 0.0, 0.6, 1, 0.4, True, top_k_recall_std=0.0)
    assert gate.decide(one_sample) is AdmitDecision.REPORTING_ONLY

    missing_parity = ProxyMetrics(
        0.9,
        0.01,
        0.8,
        50,
        0.4,
        True,
        top_k_recall_std=0.01,
        proxy_kind=ProxyKind.UMUP,
        parity_validated=False,
    )
    assert gate.decide(missing_parity) is AdmitDecision.REPORTING_ONLY


def test_proxy_metrics_and_admission_thresholds_validate_domains():
    for kwargs in (
        {"rank_corr_mean": 2.0},
        {"top_k_recall": 1.1},
        {"rank_corr_std": float("inf")},
        {"top_k_recall_std": -0.1},
        {"net_compute_savings": float("inf")},
    ):
        values = {
            "rank_corr_mean": 0.8,
            "rank_corr_std": 0.1,
            "top_k_recall": 0.8,
            "n": 20,
            "net_compute_savings": 0.2,
            "beats_exact_at_equal_budget": True,
            "top_k_recall_std": 0.1,
        }
        values.update(kwargs)
        with pytest.raises(ValueError):
            ProxyMetrics(**values)

    for kwargs in (
        {"min_rank_corr": 2.0, "min_top_k_recall": 0.5},
        {"min_rank_corr": 0.5, "min_top_k_recall": 1.1},
    ):
        with pytest.raises(ValueError):
            ProxyAdmission(**kwargs)


def test_admission_requires_explicit_recall_uncertainty():
    gate = ProxyAdmission(min_rank_corr=0.7, min_top_k_recall=0.6)
    metrics = ProxyMetrics(0.9, 0.05, 0.8, 50, 0.4, True)
    assert gate.decide(metrics) is AdmitDecision.REPORTING_ONLY
