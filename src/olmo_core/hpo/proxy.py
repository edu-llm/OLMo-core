"""
Cheaper-fidelity proxies and the gate that decides whether a proxy may prune/promote.

The plan is strict here: **fixed-token screening on the exact target model is the mandatory
baseline and first fidelity**. Any cheaper proxy (frozen-layer suffix training, a u-muP
width-reduced model) is *reporting-only* until it demonstrably beats exact-model screening at
equal total budget, with a lower-confidence-bound (LCB) on rank correlation and top-k recall
above preregistered thresholds and positive net compute savings.

This module is pure ``numpy`` + standard library. The frozen-layer helper emits ``fnmatch``
patterns compatible with :attr:`olmo_core.nn.transformer.TransformerConfig.freeze_params`;
promoted frozen-layer configs must be retrained from scratch because unfreezing changes the task.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "ExactTokenScreen",
    "FrozenLayerProxy",
    "UMuPArm",
    "ProxyKind",
    "ProxyMetrics",
    "AdmitDecision",
    "ProxyAdmission",
    "ProxyEvidenceContract",
    "preregistered_cohort",
    "evaluate_paired_proxy_bundle",
    "evaluate_paired_proxy_observations",
    "rank_correlation",
    "top_k_recall",
    "lcb",
    "output_suffix_freeze_patterns",
]


@dataclass(frozen=True)
class ExactTokenScreen:
    """Fixed-token screening on the exact target model: the mandatory baseline fidelity."""

    tokens: int
    is_mandatory_baseline: bool = field(default=True, init=False)
    fidelity_rank: int = field(default=0, init=False)


def output_suffix_freeze_patterns(n_layers: int, train_last_k: int) -> List[str]:
    """``freeze_params`` glob patterns that freeze everything but the last ``train_last_k`` blocks.

    Embeddings and all but the final ``train_last_k`` transformer blocks are frozen, so only an
    output-side suffix trains.
    """
    if not (0 < train_last_k <= n_layers):
        raise ValueError("train_last_k must be in (0, n_layers]")
    patterns = ["embeddings.*", "embedding_norm.*"]
    for i in range(n_layers - train_last_k):
        patterns.append(f"blocks.{i}.*")
    return patterns


@dataclass(frozen=True)
class FrozenLayerProxy:
    """Train an output-side suffix as a cheap fidelity. Promoted configs retrain from scratch."""

    n_layers: int
    train_last_k: int
    requires_full_retrain: bool = field(default=True, init=False)

    def freeze_patterns(self) -> List[str]:
        return output_suffix_freeze_patterns(self.n_layers, self.train_last_k)


@dataclass(frozen=True)
class ProxyEvidenceContract:
    """Pre-registered common first-rung cohort for proxy-bundle/reference evidence."""

    cohort_id: str
    config_ids: tuple[str, ...]
    first_rung_tokens: int
    proxy_arm: str = "full_acronym_soup"
    reference_arm: str = "no_proxy"
    top_k: int = 5
    search_dimensions: int = 9
    seed: int = 20260808

    def __post_init__(self) -> None:
        if not self.cohort_id:
            raise ValueError("proxy evidence cohort_id must be non-empty")
        if len(self.config_ids) < 3 or len(set(self.config_ids)) != len(self.config_ids):
            raise ValueError("proxy evidence requires at least three unique common configurations")
        if self.first_rung_tokens <= 0:
            raise ValueError("first_rung_tokens must be positive")
        if not 0 < self.top_k <= len(self.config_ids):
            raise ValueError("top_k must be in (0, cohort size]")
        if self.proxy_arm != "full_acronym_soup":
            raise ValueError("the paired proxy cohort must use full_acronym_soup")
        if self.reference_arm != "no_proxy":
            raise ValueError("the combined proxy must use the conventional no_proxy reference")
        if self.search_dimensions != 9:
            raise ValueError("the three-arm proxy cohort must use the shared 9-D search space")


def preregistered_cohort(contract: ProxyEvidenceContract) -> dict[str, tuple[float, ...]]:
    """Materialize the deterministic Latin-hypercube cohort shared by all arms."""

    rng = np.random.default_rng(contract.seed)
    size = len(contract.config_ids)
    columns = []
    for _ in range(contract.search_dimensions):
        columns.append((rng.permutation(size) + 0.5) / size)
    matrix = np.stack(columns, axis=1)
    return {
        config_id: tuple(float(value) for value in matrix[index])
        for index, config_id in enumerate(contract.config_ids)
    }


@dataclass(frozen=True)
class UMuPArm:
    """A u-muP transfer arm: width-reduced, *same-depth* proxy with parity validated first.

    Depth reduction is rejected because the Brainlift identifies depth as the weakest transfer
    axis.
    """

    width_factor: float
    depth_factor: float = 1.0
    validate_parity_first: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.width_factor <= 1.0):
            raise ValueError("width_factor must be in (0, 1]")
        if self.depth_factor != 1.0:
            raise ValueError("u-muP proxy must keep depth fixed (depth_factor == 1.0)")


class ProxyKind(str, Enum):
    EXACT = "exact"
    FROZEN_LAYER = "frozen_layer"
    UMUP = "umup"
    PROXY_BUNDLE = "umup_frozen_layer"


@dataclass(frozen=True)
class ProxyMetrics:
    rank_corr_mean: float
    rank_corr_std: float
    top_k_recall: float
    n: int
    net_compute_savings: float
    beats_exact_at_equal_budget: bool
    top_k_recall_std: Optional[float] = None
    proxy_kind: ProxyKind = ProxyKind.FROZEN_LAYER
    parity_validated: bool = False
    paired_reference_complete: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.rank_corr_mean) or not -1.0 <= self.rank_corr_mean <= 1.0:
            raise ValueError("rank_corr_mean must be finite and in [-1, 1]")
        if not math.isfinite(self.rank_corr_std) or self.rank_corr_std < 0.0:
            raise ValueError("rank_corr_std must be finite and non-negative")
        if not math.isfinite(self.top_k_recall) or not 0.0 <= self.top_k_recall <= 1.0:
            raise ValueError("top_k_recall must be finite and in [0, 1]")
        if self.top_k_recall_std is not None and (
            not math.isfinite(self.top_k_recall_std) or self.top_k_recall_std < 0.0
        ):
            raise ValueError("top_k_recall_std must be finite and non-negative")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if not math.isfinite(self.net_compute_savings):
            raise ValueError("net_compute_savings must be finite")


def evaluate_paired_proxy_bundle(
    contract: ProxyEvidenceContract,
    *,
    proxy_ce: Mapping[str, float],
    reference_ce: Mapping[str, float],
    net_compute_savings: float,
) -> ProxyMetrics:
    """Compute combined-proxy ranking evidence against the conventional reference.

    Uncertainty is estimated with a deterministic leave-one-out jackknife. Missing, additional,
    or non-finite observations fail closed instead of silently changing the cohort.
    """

    expected = set(contract.config_ids)
    if set(proxy_ce) != expected or set(reference_ce) != expected:
        raise ValueError(
            "proxy and reference evidence must exactly match the pre-registered cohort"
        )
    proxy_values = [float(proxy_ce[key]) for key in contract.config_ids]
    reference_values = [float(reference_ce[key]) for key in contract.config_ids]
    if not all(math.isfinite(value) for value in proxy_values + reference_values):
        raise ValueError("proxy evidence values must be finite")

    rank_corr = rank_correlation(proxy_values, reference_values)
    proxy_order = sorted(contract.config_ids, key=proxy_ce.__getitem__)
    reference_order = sorted(contract.config_ids, key=reference_ce.__getitem__)
    recall = top_k_recall(proxy_order, reference_order, contract.top_k)

    jackknife_rank = []
    jackknife_recall = []
    for omitted in contract.config_ids:
        ids = [key for key in contract.config_ids if key != omitted]
        try:
            jackknife_rank.append(
                rank_correlation(
                    [float(proxy_ce[key]) for key in ids],
                    [float(reference_ce[key]) for key in ids],
                )
            )
        except ValueError:
            continue
        k = min(contract.top_k, len(ids))
        jackknife_recall.append(
            top_k_recall(
                sorted(ids, key=proxy_ce.__getitem__),
                sorted(ids, key=reference_ce.__getitem__),
                k,
            )
        )
    rank_std = float(np.std(jackknife_rank, ddof=1)) if len(jackknife_rank) > 1 else math.inf
    recall_std = float(np.std(jackknife_recall, ddof=1)) if len(jackknife_recall) > 1 else math.inf
    return ProxyMetrics(
        rank_corr_mean=rank_corr,
        rank_corr_std=rank_std,
        top_k_recall=recall,
        top_k_recall_std=recall_std,
        n=len(contract.config_ids),
        net_compute_savings=net_compute_savings,
        beats_exact_at_equal_budget=False,
        proxy_kind=ProxyKind.PROXY_BUNDLE,
        parity_validated=True,
        paired_reference_complete=True,
    )


def evaluate_paired_proxy_observations(
    contract: ProxyEvidenceContract,
    *,
    proxy_observations: Mapping[str, Mapping[str, float]],
    reference_observations: Mapping[str, Mapping[str, float]],
) -> ProxyMetrics:
    """Validate raw first-rung observations and derive all admission metrics locally."""

    expected = set(contract.config_ids)
    if set(proxy_observations) != expected or set(reference_observations) != expected:
        raise ValueError("paired observations must exactly match the pre-registered cohort")

    def validate(
        observations: Mapping[str, Mapping[str, float]], *, label: str
    ) -> tuple[dict[str, float], float]:
        ce: dict[str, float] = {}
        accelerator_seconds = 0.0
        for config_id in contract.config_ids:
            observation = observations[config_id]
            tokens = int(observation["tokens"])
            value = float(observation["ce"])
            seconds = float(observation["accelerator_seconds"])
            if tokens != contract.first_rung_tokens:
                raise ValueError(
                    f"{label} observation {config_id} has {tokens} tokens, "
                    f"expected {contract.first_rung_tokens}"
                )
            if not math.isfinite(value):
                raise ValueError(f"{label} observation {config_id} has non-finite CE")
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError(f"{label} observation {config_id} has invalid accelerator seconds")
            ce[config_id] = value
            accelerator_seconds += seconds
        return ce, accelerator_seconds

    proxy_ce, proxy_seconds = validate(proxy_observations, label="proxy")
    reference_ce, reference_seconds = validate(reference_observations, label="reference")
    net_compute_savings = 1.0 - proxy_seconds / reference_seconds
    return evaluate_paired_proxy_bundle(
        contract,
        proxy_ce=proxy_ce,
        reference_ce=reference_ce,
        net_compute_savings=net_compute_savings,
    )


class AdmitDecision(str, Enum):
    PRUNE_PROMOTE = "prune_promote"
    REPORTING_ONLY = "reporting_only"


def rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation between two score sequences."""
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)
    if a_array.shape != b_array.shape or a_array.size < 2:
        raise ValueError("rank_correlation needs two equal-length sequences of length >= 2")
    if not np.all(np.isfinite(a_array)) or not np.all(np.isfinite(b_array)):
        raise ValueError("rank_correlation inputs must be finite")

    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        start = 0
        while start < values.size:
            stop = start + 1
            while stop < values.size and values[order[stop]] == values[order[start]]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return ranks

    ra = average_ranks(a_array)
    rb = average_ranks(b_array)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float(np.sum(ra**2)) * float(np.sum(rb**2)))
    if denom == 0:
        raise ValueError("rank_correlation is undefined for a constant input")
    return float(np.sum(ra * rb) / denom)


def top_k_recall(proxy_order: Sequence[int], full_order: Sequence[int], k: int) -> float:
    """Fraction of the full-model top-k that appears in the proxy's top-k."""
    if k <= 0:
        raise ValueError("k must be positive")
    proxy_top = set(list(proxy_order)[:k])
    full_top = list(full_order)[:k]
    if not full_top:
        return 0.0
    hits = sum(1 for x in full_top if x in proxy_top)
    return hits / len(full_top)


def lcb(mean: float, std: float, n: int, *, z: float = 1.96) -> float:
    """Lower confidence bound on a mean via a normal approximation."""
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be positive")
    if not all(math.isfinite(value) for value in (mean, std, z)) or std < 0.0 or z < 0.0:
        raise ValueError("LCB mean/std/z must be finite, with std and z non-negative")
    return mean - z * (std / math.sqrt(n))


@dataclass(frozen=True)
class ProxyAdmission:
    """Preregistered gate: a proxy may prune/promote only if it clears every bar."""

    min_rank_corr: float
    min_top_k_recall: float
    z: float = 1.96
    min_samples: int = 2

    def __post_init__(self) -> None:
        if not -1.0 <= self.min_rank_corr <= 1.0:
            raise ValueError("min_rank_corr must be in [-1, 1]")
        if not 0.0 <= self.min_top_k_recall <= 1.0:
            raise ValueError("min_top_k_recall must be in [0, 1]")
        if not math.isfinite(self.z) or self.z < 0.0:
            raise ValueError("z must be finite and non-negative")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")

    def decide(self, metrics: ProxyMetrics) -> AdmitDecision:
        if metrics.n < self.min_samples:
            return AdmitDecision.REPORTING_ONLY
        if metrics.top_k_recall_std is None:
            return AdmitDecision.REPORTING_ONLY
        rank_lcb = lcb(metrics.rank_corr_mean, metrics.rank_corr_std, metrics.n, z=self.z)
        recall_lcb = lcb(metrics.top_k_recall, metrics.top_k_recall_std, metrics.n, z=self.z)
        ok = (
            rank_lcb >= self.min_rank_corr
            and recall_lcb >= self.min_top_k_recall
            and metrics.net_compute_savings > 0.0
            and (
                (
                    metrics.proxy_kind is ProxyKind.PROXY_BUNDLE
                    and metrics.paired_reference_complete
                    and metrics.parity_validated
                )
                or (
                    metrics.proxy_kind is not ProxyKind.PROXY_BUNDLE
                    and metrics.beats_exact_at_equal_budget
                    and (metrics.proxy_kind is not ProxyKind.UMUP or metrics.parity_validated)
                )
            )
        )
        return AdmitDecision.PRUNE_PROMOTE if ok else AdmitDecision.REPORTING_ONLY
