"""
Clean-room IPBT (Iterated Population Based Training) outer population mechanics.

Re-implemented from the paper because the reference repository declares no license. IPBT keeps a
population, exploits top members into weak slots, explores by re-assigning hyperparameters and
weights, and periodically performs a *task-agnostic restart* while doubling the update interval.

This module owns only the **population lineage, donor selection, weight-transition policy, and
restart/update-interval mechanics** -- it never spends compute (ifBO does) or censors trials
(BTTackler does). Every decision is a typed plan the controller executes.

Safety rules the plan requires and this module enforces:

- **Optimizer-state policy follows the weight policy.** A pure checkpoint copy may retain
  optimizer moments; shrink-perturb and fresh reset must reset them (:func:`optimizer_reset_for`).
- **Online mutation is restricted to state-safe keys** (LR, weight decay, max-grad norm). Changing
  beta/epsilon/batch/schedule shape raises :class:`NewLineageRequired` -- it must start a new
  lineage with an explicit state policy, not mutate an existing one.
- **Fixed, preregistered split ratios.** The fresh-reset vs shrink-perturb and random vs BO splits
  are assigned by index (not a per-slot coin) so they are exact and reproducible.
- **Ranking happens only within comparable lineage/fidelity strata** (:meth:`group_by_stratum`);
  from-scratch and inherited curves are never pooled just because their tokens match.

Pure ``numpy`` + standard library; no ``torch``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "WeightPolicy",
    "HPSource",
    "NewLineageRequired",
    "Member",
    "Descendant",
    "GenerationPlan",
    "RestartPlan",
    "IPBTConfig",
    "RestartTracker",
    "IPBTController",
    "optimizer_reset_for",
]

# The only hyperparameters that may be mutated online without resetting optimizer/schedule state.
SAFE_ONLINE_KEYS = frozenset({"lr", "weight_decay", "max_grad_norm"})


class WeightPolicy(str, Enum):
    PURE_COPY = "pure_copy"
    SHRINK_PERTURB = "shrink_perturb"
    FRESH_RESET = "fresh_reset"


class HPSource(str, Enum):
    INHERIT = "inherit"
    RANDOM = "random"
    BO = "bo"


class NewLineageRequired(Exception):
    """Raised when a mutation cannot be applied online and must start a new lineage."""


def optimizer_reset_for(policy: WeightPolicy) -> bool:
    """Whether optimizer moments must be reset for a given weight-transition policy."""
    return policy is not WeightPolicy.PURE_COPY


@dataclass(frozen=True)
class Member:
    """One population slot's current state."""

    member_id: str
    lineage_id: str
    unit_config: Tuple[float, ...]
    score: float
    fidelity: int
    checkpoint_ref: str
    optimizer_state_valid: bool
    comparison_stratum: str = "from_scratch"


@dataclass(frozen=True)
class Descendant:
    """A plan to (re)seed a weak slot from a donor via a declared weight/HP policy."""

    slot_id: str
    donor_id: Optional[str]
    lineage_id: str
    weight_policy: WeightPolicy
    optimizer_reset: bool
    hp_source: HPSource
    unit_config: Tuple[float, ...]
    parent_lineage_id: Optional[str]
    weight_scale: float
    schedule_age_tokens: int


@dataclass(frozen=True)
class GenerationPlan:
    kept: List[Member]
    descendants: List[Descendant]


@dataclass(frozen=True)
class RestartPlan:
    copies: List[Descendant]
    descendants: List[Descendant]


@dataclass
class IPBTConfig:
    population_size: int = 16
    top_quantile: float = 0.25
    bottom_quantile: float = 0.25
    reset_fraction: float = 0.5
    random_hp_fraction: float = 0.5
    shrink_perturb_factor: float = 0.4
    update_interval_init: int = 100
    restart_patience: int = 5
    initial_oversample: Optional[int] = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.population_size, bool)
            or not isinstance(self.population_size, int)
            or self.population_size < 2
        ):
            raise ValueError("population_size must be an integer >= 2")
        for name in ("top_quantile", "bottom_quantile", "reset_fraction", "random_hp_fraction"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if self.top_quantile <= 0.0 or self.bottom_quantile <= 0.0:
            raise ValueError("top_quantile and bottom_quantile must be positive")
        if self.top_quantile + self.bottom_quantile > 1.0:
            raise ValueError("top_quantile + bottom_quantile must be <= 1")
        if not 0.0 < self.shrink_perturb_factor <= 1.0:
            raise ValueError("shrink_perturb_factor must be in (0, 1]")
        if self.update_interval_init <= 0 or self.restart_patience <= 0:
            raise ValueError("update_interval_init and restart_patience must be positive")
        if self.initial_oversample is not None and self.initial_oversample < self.population_size:
            raise ValueError("initial_oversample must be >= population_size")


class RestartTracker:
    """Task-agnostic restart trigger with update-interval doubling.

    Fires a restart once the best score has failed to improve for ``patience`` consecutive
    updates, then doubles the update interval (the "iterated" in IPBT).
    """

    def __init__(self, patience: int, interval: int, *, max_interval: Optional[int] = None) -> None:
        self.patience = patience
        self.interval = interval
        self.max_interval = max_interval
        self._best: Optional[float] = None
        self._stale = 0

    def update(self, best_score: float) -> bool:
        if self._best is None or best_score > self._best + 1e-12:
            self._best = best_score
            self._stale = 0
            return False
        self._stale += 1
        if self._stale >= self.patience:
            self._stale = 0
            self.interval *= 2
            if self.max_interval is not None:
                self.interval = min(self.interval, self.max_interval)
            return True
        return False


class IPBTController:
    def __init__(self, config: IPBTConfig) -> None:
        self.config = config
        self.restart_tracker = RestartTracker(
            patience=config.restart_patience,
            interval=config.update_interval_init,
        )
        self._lineage_counter = 0
        self._used_lineage_ids: set[str] = set()
        self._assignment_counts: Dict[Tuple[WeightPolicy, HPSource], int] = {
            (WeightPolicy.FRESH_RESET, HPSource.RANDOM): 0,
            (WeightPolicy.FRESH_RESET, HPSource.BO): 0,
            (WeightPolicy.SHRINK_PERTURB, HPSource.RANDOM): 0,
            (WeightPolicy.SHRINK_PERTURB, HPSource.BO): 0,
        }

    def _new_lineage_id(self) -> str:
        while True:
            lineage_id = f"Lnew{self._lineage_counter}"
            self._lineage_counter += 1
            if lineage_id not in self._used_lineage_ids:
                self._used_lineage_ids.add(lineage_id)
                return lineage_id

    def state_dict(self) -> Dict[str, Any]:
        return {
            "lineage_counter": self._lineage_counter,
            "used_lineage_ids": sorted(self._used_lineage_ids),
            "restart_tracker": {
                "best": self.restart_tracker._best,
                "stale": self.restart_tracker._stale,
                "interval": self.restart_tracker.interval,
            },
            "assignment_counts": {
                f"{policy.value}:{source.value}": count
                for (policy, source), count in self._assignment_counts.items()
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._lineage_counter = int(state["lineage_counter"])
        self._used_lineage_ids = set(state["used_lineage_ids"])
        tracker = state["restart_tracker"]
        assert isinstance(tracker, Mapping)
        best = tracker["best"]
        self.restart_tracker._best = None if best is None else float(best)
        self.restart_tracker._stale = int(tracker["stale"])
        self.restart_tracker.interval = int(tracker["interval"])
        for key, count in state.get("assignment_counts", {}).items():
            policy, source = str(key).split(":", 1)
            self._assignment_counts[(WeightPolicy(policy), HPSource(source))] = int(count)

    def select_initial_population(self, candidates: Sequence[Member]) -> List[Member]:
        """Apply the optional oversampling variant before reference IPBT starts."""
        expected = self.config.initial_oversample or self.config.population_size
        if len(candidates) != expected:
            raise ValueError(f"expected {expected} initial candidates, got {len(candidates)}")
        if len({candidate.member_id for candidate in candidates}) != len(candidates):
            raise ValueError("initial candidates must have unique member ids")
        if any(not math.isfinite(candidate.score) for candidate in candidates):
            raise ValueError("initial candidate scores must be finite")
        return self._rank(candidates)[: self.config.population_size]

    def group_by_stratum(self, members: Sequence[Member]) -> Dict[Tuple[str, int], List[Member]]:
        """Group by declared lineage kind and fidelity, never by raw objective alone."""
        groups: Dict[Tuple[str, int], List[Member]] = {}
        for m in members:
            groups.setdefault((m.comparison_stratum, m.fidelity), []).append(m)
        return groups

    def _rank(self, members: Sequence[Member]) -> List[Member]:
        # Deterministic: best score first, ties broken by member id.
        return sorted(members, key=lambda m: (-m.score, m.member_id))

    def _validate_members(self, members: Sequence[Member]) -> int:
        if len(members) != self.config.population_size:
            raise ValueError(
                f"expected population_size={self.config.population_size}, got {len(members)}"
            )
        if len({member.member_id for member in members}) != len(members):
            raise ValueError("population member_id values must be unique")
        if not members:
            raise ValueError("population must not be empty")
        ndim = len(members[0].unit_config)
        if ndim < 1:
            raise ValueError("population configs must have at least one dimension")
        for member in members:
            if len(member.unit_config) != ndim:
                raise ValueError("population configs must have one stable dimensionality")
            if not math.isfinite(member.score):
                raise ValueError("population scores must be finite")
            if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in member.unit_config):
                raise ValueError("population configs must lie in the finite unit cube")
            if member.fidelity <= 0:
                raise ValueError("population fidelity must be positive")
        self._used_lineage_ids.update(member.lineage_id for member in members)
        return ndim

    def _cohorts(
        self, members: Sequence[Member]
    ) -> Tuple[List[Member], List[Tuple[Member, List[Member]]]]:
        bottom_with_donors: List[Tuple[Member, List[Member]]] = []
        bottom_ids: set[str] = set()
        for stratum_members in self.group_by_stratum(members).values():
            ranked = self._rank(stratum_members)
            top_k = max(1, math.ceil(self.config.top_quantile * len(ranked)))
            bottom_k = max(1, math.ceil(self.config.bottom_quantile * len(ranked)))
            if top_k + bottom_k > len(ranked):
                continue
            donors = ranked[:top_k]
            for slot in ranked[-bottom_k:]:
                bottom_with_donors.append((slot, donors))
                bottom_ids.add(slot.member_id)
        kept = [member for member in members if member.member_id not in bottom_ids]
        return kept, bottom_with_donors

    def _factorial_assignments(
        self, n: int, rng: np.random.Generator
    ) -> List[Tuple[WeightPolicy, HPSource]]:
        reset = self.config.reset_fraction
        random = self.config.random_hp_fraction
        reset_target = reset * n
        random_target = random * n
        if n > 1 and reset_target.is_integer() and random_target.is_integer():
            n_reset = int(round(reset * n))
            n_random = int(round(random * n))
            lower = max(0, n_reset + n_random - n)
            upper = min(n_reset, n_random)
            both = min(
                max(int(round(n_reset * n_random / n)), lower),
                upper,
            )
            assignments = (
                [(WeightPolicy.FRESH_RESET, HPSource.RANDOM)] * both
                + [(WeightPolicy.FRESH_RESET, HPSource.BO)] * (n_reset - both)
                + [(WeightPolicy.SHRINK_PERTURB, HPSource.RANDOM)] * (n_random - both)
                + [(WeightPolicy.SHRINK_PERTURB, HPSource.BO)] * (n - n_reset - n_random + both)
            )
            for assignment in assignments:
                self._assignment_counts[assignment] += 1
            rng.shuffle(assignments)
            return assignments

        probabilities = {
            (WeightPolicy.FRESH_RESET, HPSource.RANDOM): reset * random,
            (WeightPolicy.FRESH_RESET, HPSource.BO): reset * (1.0 - random),
            (WeightPolicy.SHRINK_PERTURB, HPSource.RANDOM): (1.0 - reset) * random,
            (WeightPolicy.SHRINK_PERTURB, HPSource.BO): (1.0 - reset) * (1.0 - random),
        }
        cells = list(probabilities)
        assignments = []
        for _ in range(n):
            next_total = sum(self._assignment_counts.values()) + 1
            assignment = max(
                cells,
                key=lambda cell: (
                    probabilities[cell] * next_total - self._assignment_counts[cell],
                    -cells.index(cell),
                ),
            )
            self._assignment_counts[assignment] += 1
            assignments.append(assignment)
        rng.shuffle(assignments)
        return assignments

    @staticmethod
    def _next_bo_config(bo_iter, ndim: int) -> Tuple[float, ...]:
        try:
            config = tuple(float(value) for value in next(bo_iter))
        except StopIteration as exc:
            raise ValueError("not enough BO configs for the declared IPBT split") from exc
        if len(config) != ndim or any(
            not math.isfinite(value) or value < 0.0 or value > 1.0 for value in config
        ):
            raise ValueError("BO configs must match the population's finite unit cube")
        return config

    def _make_descendants(
        self,
        slots_and_donors: Sequence[Tuple[Member, List[Member]]],
        *,
        rng: np.random.Generator,
        bo_configs: Sequence[Tuple[float, ...]],
        ndim: int,
        forbidden_configs: set[Tuple[float, ...]],
        force_fresh: bool = False,
        force_fresh_slots: Optional[set[str]] = None,
    ) -> List[Descendant]:
        assignments = self._factorial_assignments(len(slots_and_donors), rng)
        bo_iter = iter(bo_configs)
        descendants: List[Descendant] = []
        for (slot, donors), (policy, hp_source) in zip(slots_and_donors, assignments):
            donor = donors[int(rng.integers(0, len(donors)))]
            must_force_fresh = force_fresh or (
                force_fresh_slots is not None and slot.member_id in force_fresh_slots
            )
            if must_force_fresh and policy is not WeightPolicy.FRESH_RESET:
                self._assignment_counts[(policy, hp_source)] -= 1
                self._assignment_counts[(WeightPolicy.FRESH_RESET, hp_source)] += 1
                policy = WeightPolicy.FRESH_RESET
            fresh = policy is WeightPolicy.FRESH_RESET
            if hp_source is HPSource.RANDOM:
                for _ in range(100):
                    unit_config = tuple(float(value) for value in rng.random(ndim))
                    if unit_config not in forbidden_configs:
                        break
                else:
                    raise ValueError("could not sample a unique random IPBT config")
            else:
                while True:
                    unit_config = self._next_bo_config(bo_iter, ndim)
                    if unit_config not in forbidden_configs:
                        break
            forbidden_configs.add(unit_config)
            descendants.append(
                Descendant(
                    slot_id=slot.member_id,
                    donor_id=None if fresh else donor.member_id,
                    lineage_id=self._new_lineage_id(),
                    weight_policy=policy,
                    optimizer_reset=True,
                    hp_source=hp_source,
                    unit_config=unit_config,
                    parent_lineage_id=None if fresh else donor.lineage_id,
                    weight_scale=0.0 if fresh else self.config.shrink_perturb_factor,
                    schedule_age_tokens=0 if fresh else donor.fidelity,
                )
            )
        return descendants

    def plan_generation(
        self,
        members: Sequence[Member],
        *,
        rng: np.random.Generator,
        bo_configs: Sequence[Tuple[float, ...]],
    ) -> GenerationPlan:
        ndim = self._validate_members(members)
        kept, bottom_with_donors = self._cohorts(members)
        descendants = self._make_descendants(
            bottom_with_donors,
            rng=rng,
            bo_configs=bo_configs,
            ndim=ndim,
            forbidden_configs={member.unit_config for member in members},
        )
        return GenerationPlan(kept=kept, descendants=descendants)

    def restart_population(
        self,
        members: Sequence[Member],
        *,
        rng: np.random.Generator,
        bo_configs: Sequence[Tuple[float, ...]],
    ) -> RestartPlan:
        ndim = self._validate_members(members)
        restart_slots: List[Tuple[Member, List[Member]]] = []
        force_fresh_slots: set[str] = set()
        for stratum_members in self.group_by_stratum(members).values():
            ranked = self._rank(stratum_members)
            if not any(member.optimizer_state_valid and member.checkpoint_ref for member in ranked):
                force_fresh_slots.update(member.member_id for member in ranked)
                restart_slots.extend((member, [ranked[0]]) for member in ranked)
                continue
            top_k = max(1, math.ceil(self.config.top_quantile * len(ranked)))
            if top_k >= len(ranked):
                continue
            donors = [
                member
                for member in ranked[:top_k]
                if member.optimizer_state_valid and member.checkpoint_ref
            ]
            if not donors:
                donors = ranked[:top_k]
                force_fresh_slots.update(member.member_id for member in ranked)
                restart_slots.extend((member, donors) for member in ranked)
                continue
            restart_slots.extend((slot, donors) for slot in ranked[top_k:])
        descendants = self._make_descendants(
            restart_slots,
            rng=rng,
            bo_configs=bo_configs,
            ndim=ndim,
            forbidden_configs={member.unit_config for member in members},
            force_fresh_slots=force_fresh_slots,
        )
        return RestartPlan(copies=[], descendants=descendants)

    def fresh_restart_population(
        self,
        members: Sequence[Member],
        *,
        rng: np.random.Generator,
        bo_configs: Sequence[Tuple[float, ...]],
    ) -> RestartPlan:
        """Replace every slot from fresh weights when no safe donor exists."""
        ndim = self._validate_members(members)
        slots = [(member, [member]) for member in members]
        descendants = self._make_descendants(
            slots,
            rng=rng,
            bo_configs=bo_configs,
            ndim=ndim,
            forbidden_configs={member.unit_config for member in members},
            force_fresh=True,
        )
        return RestartPlan(copies=[], descendants=descendants)

    def mutate_online(
        self,
        hps: Mapping[str, float],
        changes: Mapping[str, float],
        *,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        """Apply online HP changes, refusing any that are not state-safe."""
        unsafe = [k for k in changes if k not in SAFE_ONLINE_KEYS]
        if unsafe:
            raise NewLineageRequired(
                f"cannot mutate {unsafe} online; changing these starts a new lineage with an "
                f"explicit state policy. State-safe keys are {sorted(SAFE_ONLINE_KEYS)}"
            )
        out = dict(hps)
        out.update(changes)
        return out
