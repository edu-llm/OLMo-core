import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict

from torch._dynamo.utils import counters as dynamo_counters

from .callback import Callback

log = logging.getLogger(__name__)


def _counter(category: str, name: str) -> int:
    return int(dynamo_counters.get(category, {}).get(name, 0))


def _category_total(category: str) -> int:
    return sum(int(value) for value in dynamo_counters.get(category, {}).values())


def _recompile_total() -> int:
    return sum(
        int(value)
        for category, values in dynamo_counters.items()
        if "recompil" in category.lower()
        for value in values.values()
    )


@dataclass
class CompileMonitorCallback(Callback):
    """Prove that requested regional compilation realizes runtime graphs."""

    priority: ClassVar[int] = -1

    realization_check_step: int = 1
    compile_realized: bool = False

    _compile_requested: bool = False
    _steps_seen: int = 0
    _baseline: Dict[str, int] = field(default_factory=dict)

    def pre_train(self):
        if self.realization_check_step <= 0:
            raise ValueError("'realization_check_step' must be positive")
        model = getattr(self.trainer.train_module, "model", None)
        self._compile_requested = bool(getattr(model, "compile_enabled", False))
        self.compile_realized = False
        self._steps_seen = 0
        self._baseline = self._snapshot()

        triton_f32_default = os.environ.get("TRITON_F32_DEFAULT")
        triton_ieee = triton_f32_default == "ieee"
        self.trainer.record_metric("numerics/TRITON_F32_DEFAULT is ieee", float(triton_ieee))
        device = getattr(self.trainer, "device", None)
        if getattr(device, "type", None) == "cuda" and not triton_ieee:
            raise RuntimeError(
                "CUDA training with Triton kernels requires TRITON_F32_DEFAULT=ieee "
                "before Python starts"
            )
        torch_logs = {part.strip() for part in os.environ.get("TORCH_LOGS", "").split(",")}
        diagnostics_enabled = {"graph_breaks", "recompiles"} <= torch_logs
        self.trainer.record_metric(
            "compile/graph break and recompile logs enabled", float(diagnostics_enabled)
        )
        log.info(
            "Compile runtime contract: requested=%s TRITON_F32_DEFAULT=%s "
            "TORCH_LOGS_graph_breaks_recompiles=%s baseline=%s",
            self._compile_requested,
            triton_f32_default,
            diagnostics_enabled,
            self._baseline,
        )

    def _snapshot(self) -> Dict[str, int]:
        return {
            "unique_graphs": _counter("stats", "unique_graphs"),
            "calls_captured": _counter("stats", "calls_captured"),
            "graph_breaks": _category_total("graph_break"),
            "recompiles": _recompile_total(),
        }

    def post_step(self):
        self._steps_seen += 1
        current = self._snapshot()
        runtime = {name: current[name] - self._baseline[name] for name in current}
        self.trainer.record_metric("compile/runtime unique graphs", runtime["unique_graphs"])
        self.trainer.record_metric("compile/runtime calls captured", runtime["calls_captured"])
        self.trainer.record_metric("compile/runtime graph breaks", runtime["graph_breaks"])
        self.trainer.record_metric("compile/runtime recompiles", runtime["recompiles"])

        if (
            self._compile_requested
            and not self.compile_realized
            and self._steps_seen >= self.realization_check_step
        ):
            if runtime["unique_graphs"] <= 0:
                raise RuntimeError(
                    "compile was requested, but no runtime graph was realized; "
                    "configuration alone is not compilation proof"
                )
            self.compile_realized = True
            log.info(
                "Compile runtime realized at least one regional graph: %s. "
                "This proves realization, not full-model graph coverage.",
                runtime,
            )
