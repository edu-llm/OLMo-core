"""
Tests for W&B tracking (``olmo_core.latentcot.tracking``) and the ``train_arm`` ``on_log`` hook.

The point of most of these is the *fail-open* contract: a five-arm run is a day of A100 time and
a metrics sidecar must never be able to end one. So every way W&B can fail is asserted to leave
training running, and to say so rather than fail silently.
"""

import json
import sys

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.tracking import ArmTracker, resolve_project
from olmo_core.latentcot.train_driver import train_arm

from .test_train_driver import _tiny_model  # a plain 2-layer CPU model builder


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture
def dataset(tok, tmp_path):
    path = tmp_path / "conversations" / "train-00000.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        for s in range(6):
            ex = generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2))
            f.write(json.dumps(to_sft_record(ex)) + "\n")
    return LatentCotDataset(path, num_continuous_thoughts=2)


class _FakeRun:
    """Stands in for a ``wandb.Run``: records what it was given."""

    def __init__(self) -> None:
        self.logged: list = []
        self.summary: dict = {}
        self.finished_with = None
        self.url = "https://wandb.ai/fake/run"

    def log(self, values, step=None):
        self.logged.append((step, values))

    def finish(self, exit_code=0):
        self.finished_with = exit_code


# --------------------------------------------------------------------------------------
# resolve_project: the platform's env var, and the explicit override
# --------------------------------------------------------------------------------------


def test_resolve_project_prefers_explicit_over_env(monkeypatch):
    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "from-platform")
    assert resolve_project("from-flag") == "from-flag"


def test_resolve_project_falls_back_to_platform_env(monkeypatch):
    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "from-platform")
    assert resolve_project(None) == "from-platform"


def test_resolve_project_is_none_when_nothing_names_one(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    assert resolve_project(None) is None
    assert resolve_project("") is None


# --------------------------------------------------------------------------------------
# The fail-open contract
# --------------------------------------------------------------------------------------


def test_no_project_yields_inert_tracker_with_a_reason():
    tracker = ArmTracker.start(project=None, name="A2-seed1")
    assert not tracker.active
    assert tracker.reason  # non-empty, so the caller can print WHY it is off
    # Every method must be safe on an inert tracker.
    tracker.log({"step": 0, "loss": 1.0})
    tracker.summarize({"overall_acc": 0.5})
    tracker.finish()
    assert tracker.url == ""


def test_missing_wandb_package_does_not_raise(monkeypatch):
    """An image without `wandb` must train untracked, not die at startup."""
    monkeypatch.setitem(sys.modules, "wandb", None)  # forces ImportError on `import wandb`
    tracker = ArmTracker.start(project="p", name="A2-seed1")
    assert not tracker.active
    assert tracker.reason


def test_init_failure_does_not_raise(monkeypatch):
    """A failed `wandb.init` (no API key, no network, timeout) must degrade to untracked."""

    class _Boom:
        @staticmethod
        def init(**kwargs):
            raise RuntimeError("no API key")

    monkeypatch.setitem(sys.modules, "wandb", _Boom)
    tracker = ArmTracker.start(project="p", name="A2-seed1")
    assert not tracker.active
    assert "no API key" in tracker.reason


def test_log_failure_does_not_raise(monkeypatch):
    """A mid-run log failure must not end a run that is hours in."""

    class _BadRun(_FakeRun):
        def log(self, values, step=None):
            raise RuntimeError("connection reset")

    tracker = ArmTracker(run=_BadRun())
    tracker.log({"step": 1, "loss": 2.0})  # must not raise
    assert tracker.active  # and the run stays nominally open


def test_summarize_and_finish_failures_do_not_raise():
    class _BadRun:  # not a _FakeRun subclass: `summary` has to be a raising property
        @property
        def summary(self):
            raise RuntimeError("nope")

        def finish(self, exit_code=0):
            raise RuntimeError("nope")

    tracker = ArmTracker(run=_BadRun())
    tracker.summarize({"a": 1})
    tracker.finish()
    assert not tracker.active  # finish() clears the run even when it failed


# --------------------------------------------------------------------------------------
# What actually reaches W&B
# --------------------------------------------------------------------------------------


def test_log_passes_step_separately_and_drops_non_numerics():
    run = _FakeRun()
    tracker = ArmTracker(run=run)
    tracker.log({"step": 40, "loss": 1.5, "grad_norm": 0.25, "arm": "A2_codi", "ok": True})
    (step, values) = run.logged[0]
    assert step == 40
    assert values == {"loss": 1.5, "grad_norm": 0.25}  # 'step' is the axis; str and bool dropped


def test_summarize_writes_every_key():
    run = _FakeRun()
    ArmTracker(run=run).summarize({"overall_acc": 0.7, "solve_rate/depth_4": 0.5})
    assert run.summary == {"overall_acc": 0.7, "solve_rate/depth_4": 0.5}


def test_finish_records_exit_code():
    run = _FakeRun()
    ArmTracker(run=run).finish(exit_code=1)
    assert run.finished_with == 1


# --------------------------------------------------------------------------------------
# The train_arm hook
# --------------------------------------------------------------------------------------


def test_train_arm_calls_on_log_once_per_history_entry(dataset):
    torch.manual_seed(0)
    seen: list = []
    history = train_arm(
        _tiny_model(),
        ARMS["A2"],
        dataset,
        steps=4,
        batch_size=2,
        warmup_steps=1,
        log_every=1,
        on_log=seen.append,
    )
    assert len(seen) == len(history)
    assert seen == history  # same dicts, so W&B sees exactly what metrics.json records
    assert all("loss" in entry and "step" in entry for entry in seen)


def test_train_arm_survives_a_raising_on_log(dataset):
    """The last line of defense: even a sink that throws cannot end the run."""

    def boom(_entry):
        raise RuntimeError("sink exploded")

    torch.manual_seed(0)
    history = train_arm(
        _tiny_model(),
        ARMS["A2"],
        dataset,
        steps=3,
        batch_size=2,
        warmup_steps=1,
        log_every=1,
        on_log=boom,
    )
    assert len(history) == 3  # training completed regardless


def test_train_arm_without_on_log_is_unchanged(dataset):
    torch.manual_seed(0)
    history = train_arm(
        _tiny_model(), ARMS["A2"], dataset, steps=2, batch_size=2, warmup_steps=1, log_every=1
    )
    assert len(history) == 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
