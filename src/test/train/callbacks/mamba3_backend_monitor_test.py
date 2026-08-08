from collections import Counter

import pytest

from olmo_core.train.callbacks import mamba3_backend_monitor as monitor_module
from olmo_core.train.callbacks.mamba3_backend_monitor import Mamba3BackendMonitorCallback


class _Trainer:
    def __init__(self):
        self.metrics = {}

    def record_metric(self, name, value, **_kwargs):
        self.metrics[name] = value


def test_mamba_backend_monitor_requires_and_records_official_fast_realization(monkeypatch):
    counters = Counter()
    monkeypatch.setattr(monitor_module, "reset_backend_counters", counters.clear)
    monkeypatch.setattr(monitor_module, "get_backend_counters", lambda: dict(counters))
    callback = Mamba3BackendMonitorCallback(realization_check_step=1)
    trainer = _Trainer()
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_train()

    counters["official_fast"] = 12
    callback.post_step()

    assert callback.backend_realized is True
    assert trainer.metrics["mamba3/backend official_fast calls"] == 12


def test_mamba_backend_monitor_fails_closed_on_chunked_fallback(monkeypatch):
    monkeypatch.setattr(monitor_module, "reset_backend_counters", lambda: None)
    monkeypatch.setattr(monitor_module, "get_backend_counters", lambda: {"chunked": 12})
    callback = Mamba3BackendMonitorCallback(realization_check_step=1)
    callback.trainer = _Trainer()  # type: ignore[assignment]
    callback.pre_train()

    with pytest.raises(RuntimeError, match="official_fast.*not realized"):
        callback.post_step()
