from collections import Counter, defaultdict
from types import SimpleNamespace

import pytest

from olmo_core.train.callbacks import compile_monitor as compile_monitor_module
from olmo_core.train.callbacks.compile_monitor import CompileMonitorCallback


class _Trainer:
    def __init__(self, *, compile_enabled):
        self.train_module = SimpleNamespace(model=SimpleNamespace(compile_enabled=compile_enabled))
        self.metrics = {}

    def record_metric(self, name, value, **_kwargs):
        self.metrics[name] = value


@pytest.fixture
def dynamo_counters(monkeypatch):
    counters = defaultdict(Counter)
    monkeypatch.setattr(compile_monitor_module, "dynamo_counters", counters)
    return counters


def test_compile_monitor_proves_runtime_realization_and_records_diagnostics(dynamo_counters):
    dynamo_counters["stats"].update({"unique_graphs": 2, "calls_captured": 6})
    dynamo_counters["graph_break"].update({"unsupported custom op": 1})
    callback = CompileMonitorCallback(realization_check_step=1)
    trainer = _Trainer(compile_enabled=True)
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_train()

    dynamo_counters["stats"].update({"unique_graphs": 1, "calls_captured": 4})
    dynamo_counters["graph_break"].update({"tensor.item": 2})
    callback.post_step()

    assert callback.compile_realized is True
    assert trainer.metrics["compile/runtime unique graphs"] == 1
    assert trainer.metrics["compile/runtime calls captured"] == 4
    assert trainer.metrics["compile/runtime graph breaks"] == 2


def test_compile_monitor_refuses_requested_compile_without_a_realized_graph(dynamo_counters):
    callback = CompileMonitorCallback(realization_check_step=1)
    trainer = _Trainer(compile_enabled=True)
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_train()

    with pytest.raises(RuntimeError, match="compile was requested.*no runtime graph"):
        callback.post_step()


def test_compile_monitor_does_not_claim_realization_for_eager_config(dynamo_counters):
    callback = CompileMonitorCallback(realization_check_step=1)
    trainer = _Trainer(compile_enabled=False)
    callback.trainer = trainer  # type: ignore[assignment]
    callback.pre_train()
    callback.post_step()

    assert callback.compile_realized is False
    assert trainer.metrics["compile/runtime unique graphs"] == 0


def test_cuda_training_requires_and_records_triton_ieee_policy(dynamo_counters, monkeypatch):
    trainer = _Trainer(compile_enabled=False)
    trainer.device = SimpleNamespace(type="cuda")
    callback = CompileMonitorCallback()
    callback.trainer = trainer  # type: ignore[assignment]
    monkeypatch.delenv("TRITON_F32_DEFAULT", raising=False)

    with pytest.raises(RuntimeError, match="TRITON_F32_DEFAULT=ieee"):
        callback.pre_train()

    monkeypatch.setenv("TRITON_F32_DEFAULT", "ieee")
    callback.pre_train()
    assert trainer.metrics["numerics/TRITON_F32_DEFAULT is ieee"] == 1.0
