import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from olmo_core.exceptions import OLMoEnvironmentError
from olmo_core.train.callbacks import WandBCallback


class Unreachable(Exception):
    """Stands in for ``wandb.errors.CommError``.

    A plain exception rather than the real one on purpose: the guard in ``pre_train`` is
    deliberately not type-specific, because no failure of the metrics backend should be the
    thing that ends a training run.
    """


def make_callback(tmp_path: Path, fake_wandb: Mock) -> WandBCallback:
    callback = WandBCallback()
    callback._trainer = Mock(work_dir=tmp_path)
    callback._wandb = fake_wandb
    return callback


def test_a_successful_init_records_the_run_path(tmp_path, monkeypatch):
    """The ordinary path still works, so a guard that disabled everything would fail here."""
    monkeypatch.setenv("WANDB_API_KEY", "stand-in")
    fake_wandb = Mock()
    fake_wandb.run.path = "eduLLM/edullm-platform-smoke/run_019f"

    callback = make_callback(tmp_path, fake_wandb)
    callback.pre_train()

    assert callback.enabled
    assert callback.run_path == "eduLLM/edullm-platform-smoke/run_019f"
    fake_wandb.init.assert_called_once()


def test_a_failed_init_does_not_end_the_run(tmp_path, monkeypatch, caplog):
    """Eight GPU runs died here on 2026-08-01 because a stored API key was unusable."""
    monkeypatch.setenv("WANDB_API_KEY", "stand-in")
    fake_wandb = Mock()
    fake_wandb.init.side_effect = Unreachable("user is not logged in")

    callback = make_callback(tmp_path, fake_wandb)
    with caplog.at_level(logging.ERROR):
        callback.pre_train()

    assert not callback.enabled
    assert callback.run_path is None
    assert "trains without metrics logging" in caplog.text


def test_a_run_that_lost_wandb_logs_no_metrics(tmp_path, monkeypatch):
    """Disabling has to be complete, or every later step raises where init did once."""
    monkeypatch.setenv("WANDB_API_KEY", "stand-in")
    fake_wandb = Mock()
    fake_wandb.init.side_effect = Unreachable("user is not logged in")

    callback = make_callback(tmp_path, fake_wandb)
    callback.pre_train()
    callback.log_metrics(1, {"train/CE loss": 1.0})
    callback.close()

    fake_wandb.log.assert_not_called()
    fake_wandb.finish.assert_not_called()


def test_a_missing_api_key_still_ends_the_run(tmp_path, monkeypatch):
    """Unchanged, and the asymmetry is the point: this one is certain and costs no network."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    callback = make_callback(tmp_path, Mock())
    with pytest.raises(OLMoEnvironmentError):
        callback.pre_train()
