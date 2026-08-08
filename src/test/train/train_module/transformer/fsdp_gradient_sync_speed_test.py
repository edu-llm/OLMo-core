import contextlib
import json
import inspect
from unittest.mock import patch

import pytest

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.optim import AdamWConfig
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)
from olmo_core.train.train_module.transformer import train_module as train_module_lib


class _MockFSDP:
    def __init__(self):
        self.gradient_sync_calls = []
        self.all_reduce_calls = []
        self.last_backward_calls = []

    def set_requires_gradient_sync(self, value: bool):
        self.gradient_sync_calls.append(value)

    def set_requires_all_reduce(self, value: bool):
        self.all_reduce_calls.append(value)

    def set_is_last_backward(self, value: bool):
        self.last_backward_calls.append(value)


class _MockDDP:
    def __init__(self):
        self.no_sync_calls = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_calls += 1
        yield


def _config() -> TransformerTrainModuleConfig:
    return TransformerTrainModuleConfig(
        rank_microbatch_size=16,
        max_sequence_length=16,
        optim=AdamWConfig(),
    )


def _module(model, *, dp_type: DataParallelType, enabled: bool) -> TransformerTrainModule:
    module = object.__new__(TransformerTrainModule)
    object.__setattr__(module, "model", model)
    object.__setattr__(module, "_dp_config", TransformerDataParallelConfig(name=dp_type))
    object.__setattr__(module, "accumulate_grads_without_comm", enabled)
    return module


def _run_microbatches(module: TransformerTrainModule, count: int) -> None:
    for index in range(count):
        with module._train_microbatch_context(index, count):
            pass


def test_gradient_sync_escape_hatch_is_serialized_and_forwarded(tmp_path):
    config = _config()
    assert config.accumulate_grads_without_comm is False
    payload = config.as_config_dict()
    assert payload["accumulate_grads_without_comm"] is False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = TransformerTrainModuleConfig.from_json(
        path, overrides=["accumulate_grads_without_comm=true"]
    )
    assert restored.accumulate_grads_without_comm is True

    with patch.object(train_module_lib, "TransformerTrainModule") as constructor:
        restored.build(object())  # type: ignore[arg-type]
    assert constructor.call_args.kwargs["accumulate_grads_without_comm"] is True


def test_gradient_sync_option_does_not_shift_existing_constructor_positions():
    parameters = list(inspect.signature(TransformerTrainModule.__init__).parameters)
    assert parameters[:6] == [
        "self",
        "model",
        "optim",
        "rank_microbatch_size",
        "max_sequence_length",
        "compile_model",
    ]


@pytest.mark.parametrize(
    "count,enabled,expected",
    [(1, False, []), (1, True, []), (3, False, []), (3, True, [False, False, True])],
)
def test_fsdp_syncs_gradients_only_on_final_microbatch_when_enabled(count, enabled, expected):
    model = _MockFSDP()
    module = _module(model, dp_type=DataParallelType.fsdp, enabled=enabled)
    with patch.object(train_module_lib, "FSDPModule", _MockFSDP):
        _run_microbatches(module, count)

    assert model.gradient_sync_calls == expected
    assert model.last_backward_calls == [index == count - 1 for index in range(count)]
    assert model.all_reduce_calls == []


def test_hsdp_preserves_all_reduce_gating_and_restores_sync_after_error():
    model = _MockFSDP()
    module = _module(model, dp_type=DataParallelType.hsdp, enabled=True)
    with patch.object(train_module_lib, "FSDPModule", _MockFSDP):
        _run_microbatches(module, 3)
    assert model.gradient_sync_calls == [False, False, True]
    assert model.all_reduce_calls == [False, False, True]

    failing_model = _MockFSDP()
    failing_module = _module(failing_model, dp_type=DataParallelType.fsdp, enabled=True)
    with patch.object(train_module_lib, "FSDPModule", _MockFSDP):
        with pytest.raises(RuntimeError, match="backward failed"):
            with failing_module._train_microbatch_context(0, 2):
                raise RuntimeError("backward failed")
    assert failing_model.gradient_sync_calls == [False, True]


def test_ddp_still_suppresses_sync_only_on_non_final_microbatches():
    model = _MockDDP()
    module = _module(model, dp_type=DataParallelType.ddp, enabled=True)

    with patch.object(train_module_lib, "FSDPModule", _MockFSDP), patch.object(
        train_module_lib, "DDP", _MockDDP
    ):
        _run_microbatches(module, 3)

    assert model.no_sync_calls == 2


def test_gradient_sync_methods_are_not_called_on_unrelated_model_types():
    class UnrelatedModel:
        def __init__(self):
            self.calls = []

        def set_requires_gradient_sync(self, value: bool):
            self.calls.append(("gradient_sync", value))

        def set_requires_all_reduce(self, value: bool):
            self.calls.append(("all_reduce", value))

        def set_is_last_backward(self, value: bool):
            self.calls.append(("last_backward", value))

    model = UnrelatedModel()
    module = _module(model, dp_type=DataParallelType.fsdp, enabled=True)

    with patch.object(train_module_lib, "FSDPModule", _MockFSDP), patch.object(
        train_module_lib, "DDP", _MockDDP
    ):
        _run_microbatches(module, 3)

    assert model.calls == []
