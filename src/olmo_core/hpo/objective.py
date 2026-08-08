"""
Held-out objective normalization and the evaluator gate.

FT-PFN's ``T = f_best + tau * (1 - f_best)`` threshold makes the *meaning of 1* operational,
so the map from held-out cross-entropy (CE) to the ``[0, 1]`` objective must be **frozen before
HPO** and fit only on disjoint calibration runs. :class:`CENormalizer` is that frozen affine map
(lower CE is better, so higher ``y`` is better). It deliberately does **not** clip: values
outside the calibration range are rejected for FT-PFN input rather than silently pinned to 0 or
1, which would create tie pile-ups that distort MFPI ranking.

:class:`EvaluatorGate` encodes the plan's selection/evidence separation: configurations are
selected on a fixed *search-validation* split, and the frozen winner is measured once on an
*untouched* split. The gate fails closed if the search-validation evaluator is not present at a
segment boundary.

Pure ``math`` + standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Set

__all__ = ["CENormalizer", "EvaluatorGate"]


@dataclass(frozen=True)
class CENormalizer:
    """Frozen affine map from held-out CE to a maximization objective in ``[0, 1]``.

    :param ce_at_zero: The CE that maps to ``y = 0`` (the calibration *worst*).
    :param ce_at_one: The CE that maps to ``y = 1`` (the calibration *best*); must be lower.
    """

    ce_at_zero: float
    ce_at_one: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.ce_at_zero) and math.isfinite(self.ce_at_one)):
            raise ValueError("CENormalizer endpoints must be finite")
        if not (self.ce_at_one < self.ce_at_zero):
            raise ValueError(
                f"CENormalizer requires ce_at_one < ce_at_zero (lower CE is better), got "
                f"best={self.ce_at_one}, worst={self.ce_at_zero}"
            )

    def to_y(self, ce: float) -> float:
        """The raw affine objective. May fall outside ``[0, 1]`` for out-of-calibration CE."""
        ce = float(ce)
        if not math.isfinite(ce):
            raise ValueError(f"CE must be finite, got {ce}")
        return (self.ce_at_zero - ce) / (self.ce_at_zero - self.ce_at_one)

    def ce_for_y(self, y: float) -> float:
        """Inverse of :meth:`to_y`: the CE that would produce objective ``y``.

        Used to synthesize fantasy learning-curve points for pending (in-flight) workers so the
        allocator can account for them before choosing the next action.
        """
        y = float(y)
        if not math.isfinite(y):
            raise ValueError(f"y must be finite, got {y}")
        return self.ce_at_zero - y * (self.ce_at_zero - self.ce_at_one)

    def to_ftpfn_y(self, ce: float) -> float:
        """The FT-PFN-safe objective. Rejects out-of-range values instead of clipping."""
        y = self.to_y(ce)
        if y < 0.0 or y > 1.0:
            raise ValueError(
                f"CE {ce} maps to y={y} outside [0, 1]; recalibrate rather than clip "
                "(clipping produces ties that corrupt MFPI ranking)"
            )
        return y

    @classmethod
    def from_calibration(cls, ces: Sequence[float], *, margin: float = 0.0) -> "CENormalizer":
        """Fit endpoints from disjoint calibration CE values.

        :param ces: CE values from calibration-only runs (never the search runs).
        :param margin: Optional fractional widening of the range on both ends.
        """
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError(f"calibration margin must be finite and non-negative, got {margin}")
        finite = [float(c) for c in ces if math.isfinite(c)]
        if len(finite) < 2:
            raise ValueError("from_calibration needs at least two finite CE values")
        lo, hi = min(finite), max(finite)
        if hi <= lo:
            raise ValueError("calibration CE values are degenerate (all equal)")
        span = hi - lo
        return cls(ce_at_zero=hi + margin * span, ce_at_one=lo - margin * span)


@dataclass(frozen=True)
class EvaluatorGate:
    """Names the search-validation and untouched evaluators and enforces their use.

    :param search_validation: Evaluator name that drives every HPO decision.
    :param untouched: Evaluator name used exactly once on the frozen winner.
    """

    search_validation: str
    untouched: str

    def __post_init__(self) -> None:
        if not self.search_validation or not self.untouched:
            raise ValueError("EvaluatorGate requires both evaluator names")
        if self.search_validation == self.untouched:
            raise ValueError("search-validation and untouched evaluators must be different splits")

    def require_ready(self, available: Set[str]) -> None:
        """Fail closed unless the search-validation evaluator is present."""
        if self.search_validation not in available:
            raise RuntimeError(
                f"search-validation evaluator '{self.search_validation}' is not configured; "
                "refusing to emit HPO curves without a held-out objective"
            )
