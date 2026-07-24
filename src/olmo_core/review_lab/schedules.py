from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _deduplicate_positions(raw: Sequence[int], first: int, last: int) -> List[int]:
    """Turn rounded positions into a strictly increasing, endpoint-matched sequence."""

    count = len(raw)
    if count == 0:
        return []
    positions: List[int] = []
    for index, value in enumerate(raw):
        lower = first if index == 0 else positions[-1] + 1
        upper = last - (count - index - 1)
        positions.append(min(max(value, lower), upper))
    positions[0] = first
    positions[-1] = last
    return positions


def fixed_review_steps(
    kind: str,
    *,
    events: int,
    first_step: int,
    last_step: int,
    expansion_ratio: float = 2.0,
) -> List[int]:
    """Return compute- and endpoint-matched fixed review steps.

    ``cramming`` is the time-reversal of ``expanding``: gaps contract so that
    reviews become denser near the shared final endpoint.
    """

    if events == 0:
        return []
    if events == 1:
        return [first_step]
    if first_step >= last_step:
        raise ValueError("At least two events require first_step < last_step")
    if events > last_step - first_step + 1:
        raise ValueError("Not enough unique steps for the requested review events")

    span = last_step - first_step
    if kind == "uniform":
        raw = [round(first_step + span * index / (events - 1)) for index in range(events)]
    elif kind == "expanding":
        if expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must be > 1")
        denominator = expansion_ratio ** (events - 1) - 1.0
        raw = [
            round(first_step + span * (expansion_ratio**index - 1.0) / denominator)
            for index in range(events)
        ]
    elif kind == "cramming":
        if expansion_ratio <= 1.0:
            raise ValueError("expansion_ratio must be > 1")
        denominator = expansion_ratio ** (events - 1) - 1.0
        raw = [
            round(last_step - span * (expansion_ratio ** (events - 1 - index) - 1.0) / denominator)
            for index in range(events)
        ]
    else:
        raise ValueError(f"Unsupported fixed schedule: {kind}")

    return _deduplicate_positions(raw, first_step, last_step)


@dataclass(frozen=True)
class ReviewDecision:
    review: bool
    skill: Optional[str] = None
    reason: str = "new-data"


@dataclass
class SkillSelector:
    skills: Tuple[str, ...]
    adaptive: bool = False
    _baseline: Dict[str, float] = field(default_factory=dict)
    _current: Dict[str, float] = field(default_factory=dict)
    _cursor: int = 0

    def set_baseline(self, losses: Mapping[str, float]) -> None:
        self._baseline = {skill: float(losses[skill]) for skill in self.skills}
        self._current = dict(self._baseline)

    def observe(self, losses: Mapping[str, float]) -> None:
        for skill in self.skills:
            if skill in losses:
                self._current[skill] = float(losses[skill])

    def deterioration(self) -> Dict[str, float]:
        return {
            skill: self._current.get(skill, self._baseline.get(skill, 0.0))
            - self._baseline.get(skill, 0.0)
            for skill in self.skills
        }

    def choose(self) -> str:
        if self.adaptive and self._baseline:
            deltas = self.deterioration()
            # Stable lexical tie-breaking keeps paired runs reproducible.
            return max(self.skills, key=lambda skill: (deltas[skill], skill))
        skill = self.skills[self._cursor % len(self.skills)]
        self._cursor += 1
        return skill


class ReviewController:
    """Choose both review timing and the old-data skill to review."""

    def __init__(
        self,
        condition: str,
        skills: Iterable[str],
        *,
        total_steps: int,
        events: int,
        first_step: int,
        last_step: int,
        expansion_ratio: float,
        adaptive_opportunity_interval: int,
        loss_delta_threshold: float,
        min_gap: int,
    ) -> None:
        self.condition = condition
        self.skills = tuple(sorted(skills))
        if not self.skills:
            raise ValueError("At least one skill is required")
        self.total_steps = total_steps
        self.events = events
        self.first_step = first_step
        self.last_step = last_step
        self.loss_delta_threshold = loss_delta_threshold
        self.min_gap = min_gap
        self._used = 0
        self._last_review = -(10**9)
        self._review_steps: List[int] = []
        self._selector = SkillSelector(
            self.skills, adaptive=condition in {"adaptive_mix", "adaptive_due"}
        )

        if condition in {"uniform", "adaptive_mix"}:
            self._fixed_steps = set(
                fixed_review_steps(
                    "uniform",
                    events=events,
                    first_step=first_step,
                    last_step=last_step,
                    expansion_ratio=expansion_ratio,
                )
            )
        elif condition in {"expanding", "cramming"}:
            self._fixed_steps = set(
                fixed_review_steps(
                    condition,
                    events=events,
                    first_step=first_step,
                    last_step=last_step,
                    expansion_ratio=expansion_ratio,
                )
            )
        else:
            self._fixed_steps = set()

        if adaptive_opportunity_interval < 1:
            raise ValueError("adaptive_opportunity_interval must be positive")
        self._opportunities = tuple(
            sorted(
                set(range(first_step, last_step + 1, adaptive_opportunity_interval))
                | {first_step, last_step}
            )
        )
        if condition == "adaptive_due" and len(self._opportunities) < events:
            raise ValueError(
                "Adaptive schedule has fewer review opportunities than its review budget"
            )

    @property
    def review_steps(self) -> Tuple[int, ...]:
        return tuple(self._review_steps)

    @property
    def used_events(self) -> int:
        return self._used

    def set_baseline(self, losses: Mapping[str, float]) -> None:
        self._selector.set_baseline(losses)

    def observe(self, losses: Mapping[str, float]) -> None:
        self._selector.observe(losses)

    def _adaptive_due(self, step: int) -> Tuple[bool, str]:
        if step not in self._opportunities or self._used >= self.events:
            return False, "new-data"
        if step - self._last_review < self.min_gap:
            return False, "minimum-gap"

        # Match the first and final review recency of the fixed schedules. The internal events are
        # the only timing degrees of freedom, preventing recency from masquerading as spacing.
        if self.events == 1:
            return (step == self.last_step), "matched-final-endpoint"
        if step == self.first_step and self._used == 0:
            return True, "matched-first-endpoint"
        if step == self.last_step:
            return True, "matched-final-endpoint"
        if self._used >= self.events - 1:
            return False, "reserved-final-endpoint"

        remaining_budget = (self.events - 1) - self._used
        remaining_opportunities = sum(
            step <= candidate < self.last_step for candidate in self._opportunities
        )
        force_budget = remaining_budget >= remaining_opportunities
        max_delta = max(self._selector.deterioration().values(), default=-math.inf)
        due = max_delta >= self.loss_delta_threshold
        if force_budget:
            return True, "budget-deadline"
        if due:
            return True, f"loss-delta={max_delta:.4f}"
        return False, f"below-threshold={max_delta:.4f}"

    def decide(self, step: int) -> ReviewDecision:
        if not 0 <= step < self.total_steps:
            raise ValueError(f"Step {step} is outside [0, {self.total_steps})")
        if self.condition == "no_review" or self.events == 0:
            return ReviewDecision(False)

        if self.condition == "adaptive_due":
            review, reason = self._adaptive_due(step)
        else:
            review = step in self._fixed_steps
            reason = f"{self.condition}-schedule" if review else "new-data"

        if not review:
            return ReviewDecision(False, reason=reason)
        if self._used >= self.events:
            return ReviewDecision(False, reason="budget-exhausted")

        skill = self._selector.choose()
        self._used += 1
        self._last_review = step
        self._review_steps.append(step)
        return ReviewDecision(True, skill=skill, reason=reason)
