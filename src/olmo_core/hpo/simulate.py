"""
Deterministic synthetic-curve simulator for verifying the controller before any GPU work.

The plan requires a deterministic synthetic simulator as the *first* verification stage. This
provides:

- :class:`SyntheticObjective` -- a reproducible learning-curve oracle. CE decays with tokens
  toward a per-config asymptote that is lower for configs nearer a hidden optimum, so a working
  controller measurably converges toward that optimum.
- :class:`OracleFTPFN` -- a :class:`~olmo_core.hpo.ftpfn.Posterior` stand-in that scores query
  points from the simulator's expected CE. It is explicitly **not** the real FT-PFN surrogate; it
  exists only to close the control loop deterministically in tests.
- :class:`RandomProposer` -- a seeded uniform new-configuration proposer used when CMA/Centaur is
  not part of an arm.

Pure ``numpy`` + standard library.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .ftpfn import PosteriorInput
from .objective import CENormalizer

__all__ = ["SyntheticObjective", "OracleFTPFN", "RandomProposer"]


@dataclass(frozen=True)
class SyntheticObjective:
    """A deterministic learning-curve oracle over the unit cube."""

    optimum: Tuple[float, ...]
    floor_ce: float = 2.0
    ceil_ce: float = 6.0
    hardness: float = 2.0
    approach_rate: float = 4.0
    noise: float = 0.0
    seed: int = 0

    def asymptote(self, unit_config: Sequence[float]) -> float:
        cfg = np.asarray(unit_config, dtype=np.float64)
        opt = np.asarray(self.optimum, dtype=np.float64)
        dist = float(np.linalg.norm(cfg - opt))
        # Bounded in [floor_ce, ceil_ce): near the optimum -> floor, far -> ceil.
        return self.floor_ce + (self.ceil_ce - self.floor_ce) * math.tanh(self.hardness * dist)

    def _noise(self, unit_config: Sequence[float], t: float) -> float:
        if self.noise == 0.0:
            return 0.0
        key = f"{self.seed}:{tuple(round(float(x), 9) for x in unit_config)}:{round(float(t), 9)}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        u = int(digest[:16], 16) / float(1 << 64)  # in [0, 1)
        return (u - 0.5) * 2.0 * self.noise

    def ce(self, unit_config: Sequence[float], t: float) -> float:
        """Held-out CE for ``unit_config`` after fidelity fraction ``t`` in ``(0, 1]``."""
        if not (0.0 < t <= 1.0):
            raise ValueError(f"t must be in (0, 1], got {t}")
        a = self.asymptote(unit_config)
        val = a + (self.ceil_ce - a) * math.exp(-self.approach_rate * t)
        return val + self._noise(unit_config, t)


class OracleFTPFN:
    """A posterior stand-in that scores query points from the simulator (tests only)."""

    def __init__(
        self, objective: SyntheticObjective, normalizer: CENormalizer, *, scale: float = 12.0
    ) -> None:
        self.objective = objective
        self.normalizer = normalizer
        self.scale = scale

    def pi(self, x: PosteriorInput, threshold: float) -> np.ndarray:
        out: List[float] = []
        for j in range(x.query_hp.shape[0]):
            cfg = tuple(float(v) for v in x.query_hp[j])
            t = float(x.query_t[j])
            pred_ce = self.objective.ce(cfg, t)
            pred_y = self.normalizer.to_y(pred_ce)
            # Smooth probability that predicted objective exceeds the threshold.
            out.append(1.0 / (1.0 + math.exp(-self.scale * (pred_y - threshold))))
        return np.asarray(out, dtype=np.float64)


class RandomProposer:
    """A seeded uniform proposer of new unit-cube configurations."""

    def __init__(self, ndim: int, *, seed: int = 0) -> None:
        self.ndim = ndim
        self._rng = np.random.default_rng(seed)

    def ask(self, n: int) -> List[Tuple[float, ...]]:
        return [tuple(float(v) for v in self._rng.random(self.ndim)) for _ in range(n)]

    def state_dict(self) -> dict:
        return {"bit_generator_state": self._rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        self._rng.bit_generator.state = state["bit_generator_state"]
