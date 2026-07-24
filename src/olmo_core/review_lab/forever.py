from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


# Source-faithful constants from the official FOREVER implementation.
FOREVER_CALIBRATION_STEPS = 24
FOREVER_TRIGGER_DAYS: Tuple[int, ...] = (1, 2, 4, 7, 15, 30, 60, 90, 120)
FOREVER_MEMORY_RATIO = 0.02
FOREVER_MEMORY_EPOCHS = 2
FOREVER_EMA_ALPHA = 0.05
FOREVER_REGULARIZATION_COEFFICIENT = 0.001
FOREVER_SCALE_MIN = 0.5
FOREVER_SCALE_MAX = 3.0


@dataclass
class ForeverClock:
    """Track FOREVER's model-centric time and return newly crossed review days."""

    calibration_steps: int = FOREVER_CALIBRATION_STEPS
    trigger_days: Sequence[int] = FOREVER_TRIGGER_DAYS
    ema_alpha: float = FOREVER_EMA_ALPHA
    tau: float = 0.0
    current_steps: int = 0
    model_day: Optional[float] = None
    mu0: Optional[float] = None
    mu: Optional[float] = None
    next_trigger_index: int = 0
    triggered_days: List[int] = field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return self.model_day is not None

    @property
    def replay_scale(self) -> float:
        if self.mu0 is None or self.mu is None:
            return 1.0
        ratio = self.mu / max(self.mu0, 1e-12)
        return min(FOREVER_SCALE_MAX, max(FOREVER_SCALE_MIN, ratio))

    def observe_update(self, delta: float) -> List[int]:
        if delta < 0:
            raise ValueError("Parameter-change norm cannot be negative")
        self.current_steps += 1
        self.tau += float(delta)

        if not self.calibrated:
            if self.current_steps < self.calibration_steps:
                return []
            self.model_day = max(self.tau, 1e-8)
            self.mu0 = self.tau / float(self.current_steps)
            self.mu = self.mu0
        else:
            assert self.mu is not None
            self.mu = (1.0 - self.ema_alpha) * self.mu + self.ema_alpha * float(delta)

        crossed: List[int] = []
        assert self.model_day is not None
        while self.next_trigger_index < len(self.trigger_days):
            day = int(self.trigger_days[self.next_trigger_index])
            if self.tau < day * self.model_day:
                break
            crossed.append(day)
            self.triggered_days.append(day)
            self.next_trigger_index += 1
        return crossed

