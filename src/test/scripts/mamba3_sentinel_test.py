"""
Tests for the Mamba-3 silent-failure sentinel.

The governing constraint is that this callback must never be the reason a run dies. It exists
to notice silent failures; a monitor that raises from its own bookkeeping converts a
recoverable situation into an outage, and does so from inside the component you were relying
on to tell you what went wrong.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC

MODULE_PATH = Path("src/scripts/train/smoketests/mamba3_sentinel.py")


@pytest.fixture(scope="module")
def sentinel_mod() -> ModuleType:
    assert MODULE_PATH.exists(), f"{MODULE_PATH} not found; run pytest from the repo root"
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("mamba3_sentinel", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["mamba3_sentinel"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


class _FakeTrainer:
    global_step = 0

    def __init__(self):
        self.cancelled: list = []

    def cancel_run(self, reason: str):
        self.cancelled.append(reason)


def _make(sentinel_mod, **kwargs):
    callback = sentinel_mod.Mamba3SentinelCallback(run_dir=tempfile.mkdtemp(), **kwargs)
    callback.trainer = _FakeTrainer()
    return callback


@pytest.mark.parametrize("skip_rate_window", [0, 1, 2, 25])
def test_degenerate_skip_rate_window_does_not_raise(sentinel_mod, skip_rate_window: int):
    """
    A zero-length window must not divide by zero.

    ``skip_rate_window=0`` leaves the deque empty after the trim, and the rate computation then
    divides by ``len(...) == 0``. Reachable by misconfiguration, and it raises straight out of
    ``pre_log_metrics`` into the training loop.
    """
    callback = _make(sentinel_mod, skip_rate_window=skip_rate_window)
    for step in range(5):
        callback.pre_log_metrics(step, {sentinel_mod.STEP_SKIPPED_METRIC: 1.0})


@pytest.mark.parametrize("plateau_window", [0, 1, 2, 3, 50])
def test_degenerate_plateau_window_does_not_raise(sentinel_mod, plateau_window: int):
    """
    Windows below 2 must not divide by zero.

    The plateau check splits its history in half; at ``plateau_window`` of 0 or 1 that half is
    empty and the mean divides by zero.
    """
    callback = _make(sentinel_mod, plateau_window=plateau_window)
    for step in range(6):
        callback.pre_log_metrics(step, {TRAIN_CE_LOSS_METRIC: 3.0 - step * 1e-9})


def test_sentinel_survives_an_unwritable_run_dir(sentinel_mod):
    """Losing the telemetry destination must degrade to a log line, not an exception."""
    callback = sentinel_mod.Mamba3SentinelCallback(run_dir="/proc/nonexistent/nope")
    callback.trainer = _FakeTrainer()
    callback.alert("kind", "message")
    callback._write_heartbeat(status="training")


def test_sentinel_still_detects_what_it_is_for(sentinel_mod):
    """Hardening must not blunt detection: the real signals still have to fire."""
    callback = _make(sentinel_mod, skip_rate_window=4, skip_rate_threshold=0.5)
    callback.pre_log_metrics(0, {OPTIM_GRAD_NORM_METRIC: float("nan")})
    assert "nonfinite_grad_norm" in callback._alerts_seen

    callback = _make(sentinel_mod, skip_rate_window=4, skip_rate_threshold=0.5)
    for step in range(6):
        callback.pre_log_metrics(step, {sentinel_mod.STEP_SKIPPED_METRIC: 1.0})
    assert "high_step_skip_rate" in callback._alerts_seen
