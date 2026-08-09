from contextlib import ExitStack
from unittest import mock

import pytest
import torch

from olmo_core.hpo.proxy import (
    AdmitDecision,
    ExactTokenScreen,
    FrozenLayerProxy,
    ProxyAdmission,
    ProxyEvidenceContract,
    ProxyKind,
    ProxyMetrics,
    UMuPArm,
    evaluate_paired_proxy_bundle,
    evaluate_paired_proxy_observations,
    lcb,
    output_suffix_freeze_patterns,
    preregistered_cohort,
    rank_correlation,
    top_k_recall,
)
from olmo_core.hpo.umup import (
    UMuPAdamWConfig,
    apply_umup_model,
    apply_umup_parameter_metadata,
    build_same_depth_umup_proxy,
    require_official_umup_forward,
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


def test_half_layer_proxy_freezes_embeddings_and_exactly_first_eight_blocks():
    patterns = FrozenLayerProxy(n_layers=16, train_last_k=8).freeze_patterns()
    frozen_blocks = [pattern for pattern in patterns if pattern.startswith("blocks.")]
    assert frozen_blocks == [f"blocks.{index}.*" for index in range(8)]
    assert "embeddings.*" in patterns


def test_paired_proxy_evidence_requires_common_preregistered_cohort():
    contract = ProxyEvidenceContract(
        cohort_id="v1",
        config_ids=("a", "b", "c", "d", "e", "f"),
        first_rung_tokens=50_003_968,
        top_k=2,
    )
    cohort = preregistered_cohort(contract)
    assert set(cohort) == set(contract.config_ids)
    assert all(len(unit_config) == 9 for unit_config in cohort.values())
    reference = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0}
    proxy = {"a": 1.1, "b": 2.1, "c": 4.0, "d": 3.0, "e": 5.1, "f": 6.1}
    metrics = evaluate_paired_proxy_bundle(
        contract,
        proxy_ce=proxy,
        reference_ce=reference,
        net_compute_savings=0.2,
    )
    assert metrics.proxy_kind is ProxyKind.PROXY_BUNDLE
    assert metrics.paired_reference_complete is True
    assert metrics.n == 6
    assert metrics.top_k_recall == 1.0
    with pytest.raises(ValueError, match="cohort"):
        evaluate_paired_proxy_bundle(
            contract,
            proxy_ce={key: value for key, value in proxy.items() if key != "f"},
            reference_ce=reference,
            net_compute_savings=0.2,
        )

    proxy_observations = {
        key: {"tokens": contract.first_rung_tokens, "ce": value, "accelerator_seconds": 8.0}
        for key, value in proxy.items()
    }
    reference_observations = {
        key: {"tokens": contract.first_rung_tokens, "ce": value, "accelerator_seconds": 10.0}
        for key, value in reference.items()
    }
    raw_metrics = evaluate_paired_proxy_observations(
        contract,
        proxy_observations=proxy_observations,
        reference_observations=reference_observations,
    )
    assert raw_metrics.net_compute_savings == pytest.approx(0.2)
    assert (
        ProxyAdmission(min_rank_corr=0.5, min_top_k_recall=0.5, min_samples=6).decide(raw_metrics)
        is AdmitDecision.PRUNE_PROMOTE
    )
    proxy_observations["a"]["tokens"] -= 1
    with pytest.raises(ValueError, match="tokens"):
        evaluate_paired_proxy_observations(
            contract,
            proxy_observations=proxy_observations,
            reference_observations=reference_observations,
        )


def test_umup_arm_forbids_depth_reduction():
    UMuPArm(width_factor=0.5, depth_factor=1.0)  # width-reduced, same depth: ok
    with pytest.raises(ValueError):
        UMuPArm(width_factor=0.5, depth_factor=0.5)  # depth reduction is the weakest transfer axis
    with pytest.raises(TypeError):
        UMuPArm(width_factor=0.5, depth_factor=1.0, validate_parity_first=False)
    assert UMuPArm(width_factor=0.5, depth_factor=1.0).validate_parity_first is True


def test_same_depth_umup_is_counted_from_370m_and_near_190m_target():
    config, metadata = build_same_depth_umup_proxy(vocab_size=100_352)
    assert config.n_layers == metadata.source_depth == metadata.proxy_depth == 16
    assert metadata.source_architecture == "olmo2_370M"
    assert metadata.backend == "unit-scaling"
    assert metadata.proxy_non_embedding_params == config.num_non_embedding_params
    assert metadata.relative_parameter_error <= metadata.parity_tolerance
    assert config.d_model != 768  # not the stock, depth-confounded 12-layer 190M config


def test_official_umup_forward_path_is_available():
    require_official_umup_forward()


def test_official_umup_executes_cpu_forward_and_backward():
    import unit_scaling.functional as U

    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerBlockType

    config = TransformerConfig.llama_like(
        d_model=32,
        hidden_size_multiplier=2.0,
        n_layers=2,
        n_heads=4,
        vocab_size=64,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
    )
    model = config.build(init_device="meta")
    apply_umup_model(model, n_layers=config.n_layers)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights(device=torch.device("cpu"))

    operation_names = (
        "cross_entropy",
        "embedding",
        "linear",
        "linear_readout",
        "residual_add",
        "residual_split",
        "rms_norm",
        "scaled_dot_product_attention",
        "silu_glu",
    )
    with ExitStack() as stack:
        operations = {
            name: stack.enter_context(mock.patch.object(U, name, wraps=getattr(U, name)))
            for name in operation_names
        }
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.roll(-1, dims=1)
        labels[:, -1] = -100
        output = model(input_ids, labels=labels)
        output.loss.backward()

    assert torch.isfinite(output.loss)
    assert all(operation.call_count > 0 for operation in operations.values())
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert model.embeddings.weight.std().item() == pytest.approx(1.0, rel=0.15)
    assert model.lm_head.w_out.weight.mup_type == "output"
    assert model.lm_head.norm.weight.mup_type == "norm"


def test_official_umup_supports_unreduced_eval_loss():
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerBlockType

    config = TransformerConfig.llama_like(
        d_model=32,
        hidden_size_multiplier=2.0,
        n_layers=2,
        n_heads=4,
        vocab_size=64,
        block_name=TransformerBlockType.reordered_norm,
        qk_norm=True,
    )
    model = config.build(init_device="meta")
    apply_umup_model(model, n_layers=config.n_layers)
    model.to_empty(device=torch.device("cpu"))
    model.init_weights(device=torch.device("cpu"))

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.roll(-1, dims=1)
    labels[:, -1] = -100
    output = model(input_ids, labels=labels, loss_reduction="none")

    assert output.loss.shape == labels.shape
    assert output.ce_loss.shape == labels.shape
    assert torch.isfinite(output.loss).all()
    assert torch.equal(output.loss[:, -1], torch.zeros(2))


def test_umup_metadata_is_complete_and_marks_depth_and_readout():
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])
    model.lm_head = torch.nn.Linear(4, 8, bias=False)
    apply_umup_parameter_metadata(model, n_layers=16)
    metadata = {
        name: (parameter.mup_type, parameter.mup_scaling_depth)
        for name, parameter in model.named_parameters()
    }
    assert metadata["blocks.0.weight"] == ("weight", 16)
    assert metadata["lm_head.weight"] == ("output", None)

    groups = UMuPAdamWConfig(lr=1.0, weight_decay=0.1).build_groups(model)
    lrs = [group["lr"] for group in groups]
    assert min(lrs) < max(lrs)
    assert max(lrs) == pytest.approx(1.0)  # readout uses the width-stable base LR


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
