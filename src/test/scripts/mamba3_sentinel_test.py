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


class _FakeMixer:
    def __init__(self, mimo_rank: int = 1, prefer_official_kernel=None):
        self.mimo_rank = mimo_rank
        self.prefer_official_kernel = prefer_official_kernel


def _kernel_callback(sentinel_mod, monkeypatch, *, installed: bool, cuda: bool, mixers, expected):
    """A sentinel whose SSD-eligibility inputs are all pinned, so the check is deterministic."""
    callback = _make(sentinel_mod, expected_official_kernel=expected)
    monkeypatch.setattr(
        "olmo_core.nn.mamba3.mamba3_ssd_api.has_mamba3", lambda: installed, raising=True
    )
    monkeypatch.setattr(sentinel_mod.torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(
        type(callback),
        "_iter_mixers",
        lambda self: iter([(f"blocks.{i}", m) for i, m in enumerate(mixers)]),
    )
    return callback


def test_kernel_check_passes_when_the_official_path_is_live(sentinel_mod, monkeypatch):
    callback = _kernel_callback(
        sentinel_mod, monkeypatch, installed=True, cuda=True, mixers=[_FakeMixer()], expected=True
    )
    callback._check_ssd_kernel()
    assert "ssd_kernel_mismatch" not in callback._alerts_seen


@pytest.mark.parametrize(
    "installed, cuda, mixer",
    [
        (False, True, _FakeMixer()),  # mamba-ssm absent
        (True, False, _FakeMixer()),  # CPU-only
        (True, True, _FakeMixer(mimo_rank=4)),  # MIMO is ineligible
        (True, True, _FakeMixer(prefer_official_kernel=False)),  # explicitly forbidden
    ],
    ids=["not-installed", "no-cuda", "mimo", "forbidden"],
)
def test_kernel_check_catches_a_silent_downgrade(sentinel_mod, monkeypatch, installed, cuda, mixer):
    """
    The failure this exists for: asking for the fused kernel and silently getting the reference
    one, so the run goes green having exercised code the real runs never touch.
    """
    callback = _kernel_callback(
        sentinel_mod, monkeypatch, installed=installed, cuda=cuda, mixers=[mixer], expected=True
    )
    callback._check_ssd_kernel()
    assert "ssd_kernel_mismatch" in callback._alerts_seen


def test_kernel_check_is_inert_when_nothing_is_expected(sentinel_mod, monkeypatch):
    """Default runs only log which path is live; they must not fail on the fallback."""
    callback = _kernel_callback(
        sentinel_mod, monkeypatch, installed=False, cuda=False, mixers=[_FakeMixer()], expected=None
    )
    callback._check_ssd_kernel()
    assert callback._alerts_seen == {}
