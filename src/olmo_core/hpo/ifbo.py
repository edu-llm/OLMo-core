"""
MFPI-random allocation (the ifBO controller logic), re-implemented for deterministic replay.

MFPI-random is the acquisition policy from *In-context Freeze-Thaw Bayesian Optimization*:
for each decision it samples a forecast **horizon** and an improvement **threshold**
``T = f_best + tau * (1 - f_best)`` with log-uniform ``tau``, then advances every candidate's
fidelity coordinate by the horizon and asks the FT-PFN posterior for the probability that the
forecast objective exceeds ``T``. The candidate with the highest probability wins.

This module fixes the NePS reference defects the plan calls out:

- ``f_best`` is computed from **observed** points only (:func:`observed_f_best`), never from
  fantasized ones.
- the horizon is sampled from an **inclusive** integer range.
- selection is fully deterministic given the RNG: ties break on a stable candidate key.

It depends on :mod:`olmo_core.hpo.ftpfn` for input assembly and a :class:`Posterior`, both of
which are ``numpy``-only, so this module is testable with a fake posterior and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .artifacts import FTPFN_MAX_CONTEXT_CURVES, FTPFN_MAX_HP_DIMS
from .ftpfn import ObservedCurve, Posterior, QueryPoint, assemble_posterior_input
from .objective import CENormalizer
from .types import CurvePoint, ProposalSource

__all__ = [
    "Candidate",
    "Selection",
    "IfBOCandidateGenerator",
    "MFPIRandom",
    "observed_f_best",
]


@dataclass(frozen=True)
class Candidate:
    """A thing MFPI-random may allocate to.

    :param key: Stable identifier used for deterministic tie-breaking.
    :param curve_id: FT-PFN context id (>=1 for a continuation, 0 for a new config).
    :param unit_config: The configuration as a unit vector.
    :param base_tokens: Absolute tokens already trained (continuation) or the minimum
        fidelity (new config).
    :param is_continuation: Whether this resumes an existing trial.
    :param source: Where a new configuration came from (ignored for continuations).
    """

    key: str
    curve_id: int
    unit_config: Tuple[float, ...]
    base_tokens: int
    is_continuation: bool
    source: ProposalSource


@dataclass(frozen=True)
class Selection:
    """The outcome of one MFPI-random decision, with everything needed to replay/audit it."""

    chosen_index: int
    horizon: int
    threshold: float
    mfpi_score: float
    scores: np.ndarray
    score_indices: Tuple[int, ...]
    tie_break: Tuple[float, str]


class IfBOCandidateGenerator:
    """Seeded global/local/boundary candidate stream for FT-PFN ifBO.

    Candidate generation is deliberately acquisition-agnostic: FT-PFN/MFPI scores the returned
    vectors and remains the sole optimizer. Boundary and local candidates prevent a purely
    uniform stream from missing edge optima or failing to refine the incumbent.
    """

    proposal_source = ProposalSource.IFBO

    def __init__(self, ndim: int, *, seed: int = 0, local_scale: float = 0.1) -> None:
        if ndim < 1 or ndim > FTPFN_MAX_HP_DIMS:
            raise ValueError(f"ndim must be in [1, {FTPFN_MAX_HP_DIMS}]")
        if not np.isfinite(local_scale) or local_scale <= 0.0:
            raise ValueError("local_scale must be finite and positive")
        self.ndim = ndim
        self.local_scale = float(local_scale)
        self._rng = np.random.default_rng(seed)
        self._draw_count = 0
        self._seen_configs: set[Tuple[float, ...]] = set()

    def ask(self, n: int, *, incumbent: Sequence[float] | None = None) -> List[Tuple[float, ...]]:
        if n < 0:
            raise ValueError("n must be non-negative")
        incumbent_array = None
        if incumbent is not None:
            incumbent_array = np.asarray(incumbent, dtype=np.float64)
            if incumbent_array.shape != (self.ndim,) or np.any(~np.isfinite(incumbent_array)):
                raise ValueError("incumbent must be a finite vector with ndim coordinates")
            if np.any(incumbent_array < 0.0) or np.any(incumbent_array > 1.0):
                raise ValueError("incumbent must lie in the unit cube")

        configs: List[Tuple[float, ...]] = []
        while len(configs) < n:
            draw = self._draw_count
            self._draw_count += 1
            if draw == 0:
                vector = np.zeros(self.ndim)
            elif draw == 1:
                vector = np.ones(self.ndim)
            elif draw % 3 == 0 and incumbent_array is not None:
                vector = np.clip(
                    incumbent_array + self._rng.normal(0.0, self.local_scale, self.ndim),
                    0.0,
                    1.0,
                )
            elif draw % 3 == 1:
                vector = self._rng.integers(0, 2, self.ndim).astype(np.float64)
            else:
                vector = self._rng.random(self.ndim)
            config = tuple(float(value) for value in vector)
            if config in self._seen_configs:
                continue
            self._seen_configs.add(config)
            configs.append(config)
        return configs

    def state_dict(self) -> dict:
        return {
            "bit_generator_state": self._rng.bit_generator.state,
            "draw_count": self._draw_count,
            "seen_configs": [list(config) for config in sorted(self._seen_configs)],
        }

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator_state"]
        self._draw_count = int(state["draw_count"])
        self._seen_configs = {
            tuple(float(value) for value in config) for config in state.get("seen_configs", [])
        }


def observed_f_best(observed: Sequence[ObservedCurve], normalizer: CENormalizer) -> float:
    """The best (highest) normalized objective over **observed** points only.

    Returns ``0.0`` when there is no observation yet, matching an empty-history prior.
    """
    best = None
    for curve in observed:
        for p in curve.points:
            y = normalizer.to_y(p.ce)
            if best is None or y > best:
                best = y
    return 0.0 if best is None else float(best)


class MFPIRandom:
    """Deterministic MFPI-random selector over a fixed posterior.

    :param posterior: A :class:`~olmo_core.hpo.ftpfn.Posterior`.
    :param n_fidelity_bins: Number of fidelity rungs; the horizon is measured in these bins.
    :param target_tokens: The final per-lineage token horizon (denominator of ``t``).
    :param normalizer: The frozen CE->[0,1] map.
    :param tau_log_bounds: ``log10(tau)`` bounds for the log-uniform threshold multiplier.
    :param horizon_bounds: Inclusive integer horizon range in fidelity bins.
    """

    def __init__(
        self,
        posterior: Posterior,
        *,
        n_fidelity_bins: int,
        target_tokens: int,
        normalizer: CENormalizer,
        tau_log_bounds: Tuple[float, float] = (-4.0, -1.0),
        horizon_bounds: Tuple[int, int] | None = None,
    ) -> None:
        if n_fidelity_bins < 1:
            raise ValueError("n_fidelity_bins must be >= 1")
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        if not all(np.isfinite(tau_log_bounds)) or tau_log_bounds[0] > tau_log_bounds[1]:
            raise ValueError(f"invalid tau_log_bounds: {tau_log_bounds}")
        bounds = horizon_bounds or (1, n_fidelity_bins)
        if bounds[0] < 1 or bounds[0] > bounds[1] or bounds[1] > n_fidelity_bins:
            raise ValueError(f"horizon_bounds must lie within [1, {n_fidelity_bins}], got {bounds}")
        self.posterior = posterior
        self.n_fidelity_bins = n_fidelity_bins
        self.target_tokens = target_tokens
        self.normalizer = normalizer
        self.tau_log_bounds = tau_log_bounds
        self.horizon_bounds = bounds

    def _sample_horizon(self, rng: np.random.Generator) -> int:
        lo, hi = self.horizon_bounds
        return int(rng.integers(lo, hi + 1))  # inclusive upper bound

    def _sample_threshold(self, rng: np.random.Generator, f_best: float) -> float:
        lo, hi = self.tau_log_bounds
        tau = float(10.0 ** rng.uniform(lo, hi))
        return f_best + tau * (1.0 - f_best)

    def select(
        self,
        observed: Sequence[ObservedCurve],
        candidates: Sequence[Candidate],
        *,
        rng: np.random.Generator,
        f_best: float,
    ) -> Selection:
        if not candidates:
            raise ValueError("select requires at least one candidate")
        if not np.isfinite(f_best) or not 0.0 <= f_best <= 1.0:
            raise ValueError(f"f_best must be finite and in [0, 1], got {f_best}")

        horizon = self._sample_horizon(rng)
        threshold = self._sample_threshold(rng, f_best)
        increment = horizon / self.n_fidelity_bins

        queries: List[QueryPoint] = []
        for c in candidates:
            base_t = c.base_tokens / self.target_tokens
            queries.append(QueryPoint(c.curve_id, c.unit_config, t=min(base_t + increment, 1.0)))

        x = assemble_posterior_input(
            observed, queries, target_tokens=self.target_tokens, normalizer=self.normalizer
        )
        scores = np.asarray(self.posterior.pi(x, threshold), dtype=np.float64)
        if scores.shape != (len(candidates),):
            raise ValueError(
                f"posterior returned {scores.shape} scores for {len(candidates)} candidates"
            )
        if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("posterior PI scores must be finite probabilities in [0, 1]")

        # Deterministic argmax: maximize PI, break ties on the smallest candidate key.
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-float(scores[i]), candidates[i].key),
        )
        chosen = order[0]
        return Selection(
            chosen_index=chosen,
            horizon=horizon,
            threshold=threshold,
            mfpi_score=float(scores[chosen]),
            scores=scores,
            score_indices=tuple(range(len(candidates))),
            tie_break=(-float(scores[chosen]), candidates[chosen].key),
        )

    def select_batch(
        self,
        observed: Sequence[ObservedCurve],
        candidates: Sequence[Candidate],
        *,
        count: int,
        rng: np.random.Generator,
        f_best: float,
        fantasy_y: float | None = None,
    ) -> List[Selection]:
        """Choose ``count`` distinct actions sequentially, fantasizing each pick into context.

        This is how a single controller feeds ``count`` concurrent workers without proposing the
        same continuation to two of them: after each pick, a fantasy curve point for the chosen
        candidate (objective ``fantasy_y``, defaulting to the current ``f_best``) is appended to
        the FT-PFN context so the next pick conditions on the pending work.

        Returned :class:`Selection` objects carry ``chosen_index`` in terms of the *original*
        ``candidates`` list.
        """
        if count < 0:
            raise ValueError("count must be non-negative")
        keys = [candidate.key for candidate in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate keys must be unique within an allocation batch")
        config_ids: dict[Tuple[float, ...], set[int]] = {}
        for candidate in candidates:
            config_ids.setdefault(candidate.unit_config, set()).add(candidate.curve_id)
        for config, curve_ids in config_ids.items():
            matching_count = sum(candidate.unit_config == config for candidate in candidates)
            if matching_count > 1 and (len(curve_ids) != 1 or next(iter(curve_ids)) <= 0):
                raise ValueError("duplicate candidate configs require one shared positive curve_id")
        fantasy_y = f_best if fantasy_y is None else fantasy_y
        if not np.isfinite(fantasy_y) or not 0.0 <= fantasy_y <= 1.0:
            raise ValueError("fantasy_y must be finite and in [0, 1]")
        fantasy_ce = self.normalizer.ce_for_y(fantasy_y)

        new_slots = min(
            max(0, count - 1),
            sum(not candidate.is_continuation for candidate in candidates),
        )
        context_capacity = FTPFN_MAX_CONTEXT_CURVES - new_slots
        if context_capacity < 0:
            raise ValueError("allocation batch exceeds FT-PFN context capacity")
        context_ids = {curve.curve_id for curve in observed}
        required_ids = {candidate.curve_id for candidate in candidates if candidate.curve_id > 0}
        if not required_ids <= context_ids:
            missing = sorted(required_ids - context_ids)
            raise ValueError(f"resume candidates reference missing context curves: {missing}")
        if len(required_ids) > context_capacity:
            raise ValueError("resume candidates exceed retained FT-PFN context capacity")
        if len(context_ids) > context_capacity:
            latest_tokens = {
                curve_id: max(
                    point.tokens
                    for curve in observed
                    if curve.curve_id == curve_id
                    for point in curve.points
                )
                for curve_id in context_ids
            }
            additional = [
                curve_id
                for curve_id, _ in sorted(
                    latest_tokens.items(),
                    key=lambda item: (-item[1], -item[0]),
                )
                if curve_id not in required_ids
            ][: context_capacity - len(required_ids)]
            retained_ids = required_ids | set(additional)
            context = [curve for curve in observed if curve.curve_id in retained_ids]
        else:
            context = list(observed)
        old_to_new_id = {
            old_id: index + 1
            for index, old_id in enumerate(sorted({curve.curve_id for curve in context}))
        }
        context = [
            ObservedCurve(
                curve_id=old_to_new_id[curve.curve_id],
                unit_config=curve.unit_config,
                points=curve.points,
            )
            for curve in context
        ]
        working_candidates = [
            Candidate(
                key=candidate.key,
                curve_id=(0 if candidate.curve_id <= 0 else old_to_new_id[candidate.curve_id]),
                unit_config=candidate.unit_config,
                base_tokens=candidate.base_tokens,
                is_continuation=candidate.is_continuation,
                source=candidate.source,
            )
            for candidate in candidates
        ]
        remaining = list(range(len(candidates)))
        used_curve_ids = [curve.curve_id for curve in context]
        used_curve_ids.extend(candidate.curve_id for candidate in working_candidates)
        free_fantasy_ids = iter(
            sorted(
                set(range(1, FTPFN_MAX_CONTEXT_CURVES + 1))
                - {curve_id for curve_id in used_curve_ids if curve_id > 0}
            )
        )
        picks: List[Selection] = []

        total_picks = min(count, len(candidates))
        for pick_index in range(total_picks):
            sub = [working_candidates[i] for i in remaining]
            sel = self.select(context, sub, rng=rng, f_best=f_best)
            orig = remaining[sel.chosen_index]
            chosen = working_candidates[orig]
            picks.append(
                Selection(
                    chosen_index=orig,
                    horizon=sel.horizon,
                    threshold=sel.threshold,
                    mfpi_score=sel.mfpi_score,
                    scores=sel.scores,
                    score_indices=tuple(remaining),
                    tie_break=sel.tie_break,
                )
            )
            remaining.remove(orig)

            if pick_index == total_picks - 1:
                continue
            # Fantasize the pending outcome at the scored horizon.
            increment = sel.horizon / self.n_fidelity_bins
            base_t = chosen.base_tokens / self.target_tokens
            scored_tokens = int(round(min(base_t + increment, 1.0) * self.target_tokens))
            fantasy_id = chosen.curve_id if chosen.curve_id > 0 else next(free_fantasy_ids)
            context.append(
                ObservedCurve(
                    curve_id=fantasy_id,
                    unit_config=chosen.unit_config,
                    points=(CurvePoint(tokens=max(scored_tokens, 1), ce=fantasy_ce),),
                )
            )

        return picks
