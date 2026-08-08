import logging
from dataclasses import dataclass

from olmo_core.nn.mamba3.mamba3_ssd_api import (
    get_backend_counters,
    reset_backend_counters,
)

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class Mamba3BackendMonitorCallback(Callback):
    """Fail closed unless the requested official Mamba-3 backend executes."""

    expected_backend: str = "official_fast"
    realization_check_step: int = 1
    backend_realized: bool = False
    _steps_seen: int = 0

    def pre_train(self):
        if self.realization_check_step <= 0:
            raise ValueError("'realization_check_step' must be positive")
        reset_backend_counters()
        self.backend_realized = False
        self._steps_seen = 0
        log.info("Mamba-3 runtime backend required: %s", self.expected_backend)

    def post_step(self):
        self._steps_seen += 1
        counters = get_backend_counters()
        for backend, count in sorted(counters.items()):
            self.trainer.record_metric(f"mamba3/backend {backend} calls", count)
        if not self.backend_realized and self._steps_seen >= self.realization_check_step:
            if counters.get(self.expected_backend, 0) <= 0:
                raise RuntimeError(
                    f"Mamba-3 backend '{self.expected_backend}' was not realized; "
                    f"runtime counters={counters}"
                )
            self.backend_realized = True
            log.info("Mamba-3 runtime backend realized: %s", counters)
