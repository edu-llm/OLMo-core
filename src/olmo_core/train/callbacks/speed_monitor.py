import inspect
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

import torch

from olmo_core.config import DType
from olmo_core.distributed.utils import get_world_size

from ..common import ReduceType
from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


def _dense_bf16_peak_flops(device_name: str) -> Optional[int]:
    """Return an explicitly audited dense-BF16 peak, with no device fallback."""
    dense_correction = 0.5
    if "H100" in device_name:
        if "NVL" in device_name:
            return int(1671e12 * dense_correction)
        if "PCIe" in device_name:
            return int(1513e12 * dense_correction)
        return int(1979e12 * dense_correction)
    if "B200" in device_name:
        return int(4.5e15 * dense_correction)
    if "A100" in device_name:
        return int(312e12)
    if "L40S" in device_name:
        return int(362e12 * dense_correction)
    if "A10G" in device_name:
        return int(125e12 * dense_correction)
    if "L4" in device_name:
        return int(121e12 * dense_correction)
    return None


def _percentile(values: List[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class SpeedMonitorCallback(Callback):
    """
    Monitors throughput.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    priority: ClassVar[int] = -2

    num_flops_per_token: Optional[int] = None
    num_params: Optional[int] = None
    device_peak_flops_per_second: Optional[int] = None
    steady_state_warmup_steps: int = 20
    steady_state_window_steps: int = 50
    measured_gemm_flops_per_second: Optional[float] = None
    working_set_bytes: Optional[int] = None
    device_l2_bytes: Optional[int] = None
    flops_formula_provenance: Optional[str] = None

    _total_steps: int = 0
    _total_tokens: int = 0
    _total_flops: int = 0
    _start_time: float = 0.0
    _first_step: bool = True
    _step_last_logged: float = 0.0
    _batch_load_start: float = 0.0
    _batch_load_time: float = 0.0
    _step_tokens: int = 0
    _step_seq_len: int = 0
    _step_flops: int = 0
    _parallel_degree: int = 1
    _bps_avg: Optional[float] = None
    _tps_avg: Optional[float] = None
    _mfu_avg: Optional[float] = None
    _steady_state_steps_seen: int = 0
    _steady_state_step_times: List[float] = field(default_factory=list)
    _steady_state_tps: List[float] = field(default_factory=list)
    _steady_state_flops_ps: List[float] = field(default_factory=list)
    _steady_state_excluded: Dict[str, int] = field(default_factory=dict)
    _steady_state_reported: bool = False

    def reset(self):
        self._first_step = True
        self._bps_avg = None
        self._steady_state_steps_seen = 0
        self._steady_state_step_times.clear()
        self._steady_state_tps.clear()
        self._steady_state_flops_ps.clear()
        self._steady_state_excluded.clear()
        self._steady_state_reported = False

    @property
    def bps_avg(self) -> Optional[float]:
        return self._bps_avg

    @property
    def tps_avg(self) -> Optional[float]:
        return self._tps_avg

    @property
    def mfu_avg(self) -> Optional[float]:
        return self._mfu_avg

    def _get_num_flops_per_token(self, seq_len: int) -> Optional[int]:
        if self.num_flops_per_token is not None:
            return self.num_flops_per_token
        elif isinstance(self.trainer.train_module, TransformerTrainModule):
            return self.trainer.train_module.num_flops_per_token(seq_len)
        else:
            return None

    def pre_train(self):
        self.reset()
        if self.steady_state_warmup_steps < 0:
            raise ValueError("'steady_state_warmup_steps' must be non-negative")
        if self.steady_state_window_steps <= 0:
            raise ValueError("'steady_state_window_steps' must be positive")
        if (
            self.measured_gemm_flops_per_second is not None
            and self.measured_gemm_flops_per_second <= 0
        ):
            raise ValueError("'measured_gemm_flops_per_second' must be positive")

        if self.trainer.dp_process_group is not None:
            self._parallel_degree = get_world_size() // get_world_size(
                self.trainer.dp_process_group
            )

        if self.num_params is None and isinstance(
            self.trainer.train_module, TransformerTrainModule
        ):
            self.num_params = self.trainer.train_module.model.num_non_embedding_params

        if isinstance(self.trainer.train_module, TransformerTrainModule):
            model = self.trainer.train_module.model
            if self.working_set_bytes is None:
                self.working_set_bytes = sum(
                    parameter.numel() * parameter.element_size() for parameter in model.parameters()
                )
            if self.flops_formula_provenance is None:
                formula = model.num_flops_per_token
                try:
                    source_file = inspect.getsourcefile(formula)
                    _, source_line = inspect.getsourcelines(formula)
                except (OSError, TypeError):
                    source_file = None
                    source_line = 0
                if source_file is not None:
                    self.flops_formula_provenance = f"{source_file}:{source_line}"
        if self.device_l2_bytes is None and self.trainer.device.type == "cuda":
            l2_cache_size = getattr(
                torch.cuda.get_device_properties(self.trainer.device), "L2_cache_size", None
            )
            if isinstance(l2_cache_size, int) and l2_cache_size > 0:
                self.device_l2_bytes = l2_cache_size

        if (
            self.device_peak_flops_per_second is None
            and self.trainer.device.type == "cuda"
            and isinstance(self.trainer.train_module, TransformerTrainModule)
        ):
            device_name = torch.cuda.get_device_name(self.trainer.device)

            tm = self.trainer.train_module
            using_half_precision = tm.autocast_precision == torch.bfloat16 or (
                tm.dp_config is not None and tm.dp_config.param_dtype == DType.bfloat16
            )
            if using_half_precision:
                self.device_peak_flops_per_second = _dense_bf16_peak_flops(device_name)
                if self.device_peak_flops_per_second is None:
                    log.warning(
                        "No audited dense-BF16 peak FLOP/s for device '%s'; MFU will be omitted",
                        device_name,
                    )
            log.info(
                f"Device: {device_name}, Device peak Flops/s: {self.device_peak_flops_per_second}"
            )

        window_end = self.steady_state_warmup_steps + self.steady_state_window_steps
        log.info(
            "Steady-state throughput denominator: "
            "dense_bf16_device_peak_flops_per_second=%s "
            "measured_gemm_flops_per_second=%s window=[%d,%d)",
            self.device_peak_flops_per_second,
            self.measured_gemm_flops_per_second,
            self.steady_state_warmup_steps,
            window_end,
        )
        log.info(
            "Steady-state FLOPs formula provenance: formula=%s",
            self.flops_formula_provenance or "unavailable",
        )
        log.info(
            "Steady-state working set: working_set_bytes=%s device_l2_bytes=%s",
            self.working_set_bytes,
            self.device_l2_bytes,
        )
        if self.working_set_bytes is not None:
            self.trainer.record_metric(
                "throughput/steady_state/working set bytes", self.working_set_bytes
            )
            if self.device_l2_bytes is not None and self.device_l2_bytes > 0:
                working_set_over_l2 = self.working_set_bytes / self.device_l2_bytes
                self.trainer.record_metric(
                    "throughput/steady_state/working set over L2", working_set_over_l2
                )
                if working_set_over_l2 <= 1:
                    log.warning(
                        "Steady-state working set fits in L2; production HBM conclusions are invalid"
                    )

    def pre_load_batch(self):
        self._batch_load_start = time.perf_counter()

    def pre_step(self, batch: Dict[str, Any]):
        self._batch_load_time = time.perf_counter() - self._batch_load_start

        if self._first_step:
            # We don't record the first batch since the first one tends to take
            # unusually long.
            return

        self._total_steps += 1
        if "input_ids" in batch:
            tokens_in_batch = batch["input_ids"].numel()
            self._step_tokens = tokens_in_batch // self._parallel_degree
            self._step_seq_len = batch["input_ids"].shape[1]
            self._total_tokens += self._step_tokens

            self._step_flops = 0
            if (
                num_flops_per_token := self._get_num_flops_per_token(self._step_seq_len)
            ) is not None:
                self._step_flops = num_flops_per_token * self._step_tokens
                self._total_flops += self._step_flops

    def post_step(self):
        counter = time.perf_counter()
        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time, reduce_type=ReduceType.max
        )

        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._total_flops = 0
            self._start_time = counter
            self._first_step = False
            self._step_last_logged = counter
            return

        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        tps: Optional[float] = None
        if self._step_tokens and self._total_tokens:
            tps = self._step_tokens / step_time
            tps_avg = self._total_tokens / total_time
            self._tps_avg = tps_avg
            self.trainer.record_metric("throughput/device/TPS", tps)
            self.trainer.record_metric("throughput/device/TPS (actual avg)", tps_avg)

        if self.trainer.global_train_tokens_seen is not None:
            self.trainer.record_metric(
                "throughput/total tokens", self.trainer.global_train_tokens_seen
            )
            if self.num_params is not None:
                self.trainer.record_metric(
                    "throughput/chinchilla multiple",
                    self.trainer.global_train_tokens_seen / (20 * self.num_params),
                )

        flops_ps: Optional[float] = None
        flops_ps_avg: Optional[float] = None
        if self._step_flops and self._total_flops:
            flops_ps = self._step_flops / step_time
            flops_ps_avg = self._total_flops / total_time
            self.trainer.record_metric("throughput/device/flopsPS", flops_ps)
            self.trainer.record_metric("throughput/device/flopsPS (actual avg)", flops_ps_avg)
            self.trainer.record_metric(
                "throughput/total petaflops", self.trainer.global_train_petaflops
            )

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        self._bps_avg = bps_avg
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (actual avg)", bps_avg)

        data_pct = 100 * self._batch_load_time / step_time
        self.trainer.record_metric(
            "throughput/device/data loading (%)", data_pct, reduce_type=ReduceType.max
        )

        if (
            self.device_peak_flops_per_second is not None
            and flops_ps is not None
            and flops_ps_avg is not None
        ):
            # model FLOPS utilization
            # For its definition and calculation, please refer to the PaLM paper:
            # https://arxiv.org/abs/2204.02311
            # MFU is computed from FLOPs/sec. This stays correct even if sequence length changes.
            mfu = 100 * flops_ps / self.device_peak_flops_per_second
            mfu_avg = 100 * flops_ps_avg / self.device_peak_flops_per_second
            self._mfu_avg = mfu_avg
            self.trainer.record_metric("throughput/device/MFU", mfu)
            self.trainer.record_metric("throughput/device/MFU (actual avg)", mfu_avg)

        self._record_steady_state(step_time, tps, flops_ps)

    def _record_steady_state(
        self, step_time: float, tps: Optional[float], flops_ps: Optional[float]
    ) -> None:
        sample_index = self._steady_state_steps_seen
        self._steady_state_steps_seen += 1
        window_start = self.steady_state_warmup_steps
        window_end = window_start + self.steady_state_window_steps

        if sample_index < window_start:
            self._steady_state_excluded["warmup"] = self._steady_state_excluded.get("warmup", 0) + 1
            return
        if sample_index >= window_end:
            self._steady_state_excluded["outside_window"] = (
                self._steady_state_excluded.get("outside_window", 0) + 1
            )
            return
        if not math.isfinite(step_time) or step_time <= 0:
            self._steady_state_excluded["invalid_step_time"] = (
                self._steady_state_excluded.get("invalid_step_time", 0) + 1
            )
        else:
            self._steady_state_step_times.append(step_time)
        if tps is None or not math.isfinite(tps):
            self._steady_state_excluded["missing_tokens"] = (
                self._steady_state_excluded.get("missing_tokens", 0) + 1
            )
        else:
            self._steady_state_tps.append(tps)
        if flops_ps is None or not math.isfinite(flops_ps):
            self._steady_state_excluded["missing_flops"] = (
                self._steady_state_excluded.get("missing_flops", 0) + 1
            )
        else:
            self._steady_state_flops_ps.append(flops_ps)

        if self._steady_state_steps_seen == window_end:
            self._report_steady_state()

    def _report_steady_state(self) -> None:
        if self._steady_state_reported:
            return
        sample_count = len(self._steady_state_step_times)
        if sample_count != self.steady_state_window_steps:
            log.warning(
                "Refusing incomplete steady-state window: %d/%d valid step times; excluded=%s",
                sample_count,
                self.steady_state_window_steps,
                self._steady_state_excluded,
            )
            return

        median_step_time = statistics.median(self._steady_state_step_times)
        maximum_step_time = max(self._steady_state_step_times)
        metrics = {
            "throughput/steady_state/step time median (s)": median_step_time,
            "throughput/steady_state/step time min (s)": min(self._steady_state_step_times),
            "throughput/steady_state/step time p90 (s)": _percentile(
                self._steady_state_step_times, 0.9
            ),
            "throughput/steady_state/step time max (s)": maximum_step_time,
            "throughput/steady_state/max over median": maximum_step_time / median_step_time,
        }
        if len(self._steady_state_tps) == self.steady_state_window_steps:
            metrics["throughput/steady_state/TPS median"] = statistics.median(
                self._steady_state_tps
            )
        if len(self._steady_state_flops_ps) == self.steady_state_window_steps:
            median_flops_ps = statistics.median(self._steady_state_flops_ps)
            metrics["throughput/steady_state/flopsPS median"] = median_flops_ps
            if self.device_peak_flops_per_second is not None:
                metrics["throughput/steady_state/MFU median"] = (
                    100 * median_flops_ps / self.device_peak_flops_per_second
                )
            if self.measured_gemm_flops_per_second is not None:
                metrics["throughput/steady_state/MFU measured GEMM median"] = (
                    100 * median_flops_ps / self.measured_gemm_flops_per_second
                )

        for name, value in metrics.items():
            self.trainer.record_metric(name, value)
        window_end = self.steady_state_warmup_steps + self.steady_state_window_steps
        log.info(
            "RESULT steady_state window=[%d,%d) median_step_time_s=%g p90_step_time_s=%g "
            "excluded=%s formula=%s working_set_bytes=%s device_l2_bytes=%s "
            "dense_bf16_device_peak_flops_per_second=%s measured_gemm_flops_per_second=%s",
            self.steady_state_warmup_steps,
            window_end,
            median_step_time,
            metrics["throughput/steady_state/step time p90 (s)"],
            self._steady_state_excluded,
            self.flops_formula_provenance or "unavailable",
            self.working_set_bytes,
            self.device_l2_bytes,
            self.device_peak_flops_per_second,
            self.measured_gemm_flops_per_second,
        )
        if metrics["throughput/steady_state/max over median"] > 1.5:
            log.warning("Steady-state step-time max exceeds 1.5x the median")
        self._steady_state_reported = True

    def post_train(self):
        if not self._steady_state_reported:
            self._report_steady_state()
