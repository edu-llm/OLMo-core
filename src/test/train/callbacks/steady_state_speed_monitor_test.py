import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from olmo_core.train.callbacks import speed_monitor as speed_monitor_module
from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback
from olmo_core.train.train_module.transformer import TransformerTrainModule


class _RecordingTrainer:
    def __init__(self):
        self.dp_process_group = None
        self.device = torch.device("cpu")
        self.train_module = object()
        self.global_train_tokens_seen = None
        self.global_train_petaflops = 0.0
        self.metrics = {}

    def record_metric(self, name, value, **_kwargs):
        self.metrics[name] = value


def _run_step(callback, batch):
    callback.pre_step(batch)
    callback.post_step()


def test_declared_steady_state_window_reports_auditable_medians(monkeypatch, caplog):
    callback = SpeedMonitorCallback(
        num_flops_per_token=10,
        num_params=4,
        device_peak_flops_per_second=1000,
        steady_state_warmup_steps=1,
        steady_state_window_steps=3,
        measured_gemm_flops_per_second=500,
        working_set_bytes=400,
        device_l2_bytes=100,
        flops_formula_provenance="fixture_model.py:7",
    )
    trainer = _RecordingTrainer()
    callback.trainer = trainer  # type: ignore[assignment]
    times = iter([0, 0, 0, 2, 2, 4, 4, 7, 7, 11])
    monkeypatch.setattr(speed_monitor_module.time, "perf_counter", lambda: next(times))
    batch = {"input_ids": torch.zeros((1, 10), dtype=torch.long)}

    caplog.set_level(logging.INFO)
    callback.pre_train()
    _run_step(callback, batch)  # establishes the post-startup clock
    _run_step(callback, batch)  # declared warmup exclusion: 2 seconds
    _run_step(callback, batch)  # measured: 2 seconds
    _run_step(callback, batch)  # measured: 3 seconds
    _run_step(callback, batch)  # measured: 4 seconds

    assert trainer.metrics["throughput/steady_state/step time median (s)"] == 3
    assert trainer.metrics["throughput/steady_state/TPS median"] == pytest.approx(10 / 3)
    assert trainer.metrics["throughput/steady_state/MFU median"] == pytest.approx(10 / 3)
    assert trainer.metrics["throughput/steady_state/MFU measured GEMM median"] == pytest.approx(
        20 / 3
    )
    assert trainer.metrics["throughput/steady_state/working set bytes"] == 400
    assert trainer.metrics["throughput/steady_state/working set over L2"] == 4
    assert "window=[1,4)" in caplog.text
    assert "excluded={'warmup': 1}" in caplog.text
    assert "formula=fixture_model.py:7" in caplog.text
    assert "working_set_bytes=400" in caplog.text
    assert "device_l2_bytes=100" in caplog.text
    assert "measured_gemm_flops_per_second=500" in caplog.text


def test_incomplete_steady_state_window_refuses_a_result(caplog):
    callback = SpeedMonitorCallback(
        steady_state_warmup_steps=1,
        steady_state_window_steps=3,
    )
    callback._steady_state_step_times = [2.0, 3.0]
    callback._steady_state_excluded = {"warmup": 1}

    caplog.set_level(logging.WARNING)
    callback.post_train()

    assert "incomplete steady-state window" in caplog.text
    assert "2/3" in caplog.text


def test_pre_train_derives_formula_parameter_working_set_and_cuda_l2(monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

        @property
        def num_non_embedding_params(self):
            return self.weight.numel()

        def num_flops_per_token(self, seq_len):
            return seq_len * 2

    train_module = object.__new__(TransformerTrainModule)
    object.__setattr__(train_module, "model", _Model())
    object.__setattr__(train_module, "_dp_config", None)
    object.__setattr__(train_module, "autocast_precision", None)
    trainer = _RecordingTrainer()
    trainer.device = torch.device("cuda")
    trainer.train_module = train_module
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(L2_cache_size=40)
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "Unknown Test GPU")

    callback = SpeedMonitorCallback()
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_train()

    assert callback.working_set_bytes == 8
    assert callback.device_l2_bytes == 40
    assert callback.flops_formula_provenance is not None
    assert "steady_state_speed_monitor_test.py:" in callback.flops_formula_provenance
