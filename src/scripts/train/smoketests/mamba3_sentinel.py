"""
In-process guard against *silent* Mamba-3 training failures.

A silent failure is one where the GPU stays busy, the step counter advances, and nothing
raises -- but the run has stopped producing anything useful. The trainer already fails loudly
on a non-finite ``train/CE loss`` (``trainer.py``, ``_check_and_pass_on_metrics``), so this
covers the modes it does *not*:

1. **Non-finite gradient norm.** The trainer's finiteness check reads only the CE loss metric.
   A NaN/Inf ``optim/total grad norm`` never trips it.
2. **Poisoned skip-step optimizer.** :class:`SkipStepOptimizer.get_step_factor` compares the
   latest loss/grad-norm against rolling statistics. A single NaN in *either* series makes
   ``torch.std_mean`` return NaN, and every subsequent comparison against NaN is ``False``, so
   the step factor is 0. Measured at the production default ``rolling_interval_length=128``:
   one NaN causes **129 consecutive skipped optimizer steps**, and the loss looks perfectly
   healthy throughout because the weights simply stop moving.
3. **Vanishing memory horizon.** ``A_log ~ log(Uniform(a_log_init_min, a_log_init_max))`` with the default
   16 gives ``alpha ~ 0.92``, so a signal injected at position 0 is at ``~1e-9`` by position
   256. The model still trains -- it learns local statistics and the loss falls -- while every
   long-range capability the block-rotation work exists to buy is unreachable. Nothing errors.
4. **Silently-dropped configuration.** If ``rotation_block_size`` fails to reach the kernel the
   model stays abelian while every metric looks nominal.
5. **A wedged process.** No metric can report that the trainer has stopped stepping, so this
   writes a heartbeat for the external watchdog to read.

Design constraint: this **detects and notifies, it never terminates**. It does not kill the
process, and it contains no AWS calls of any kind, so it cannot stop or terminate an instance.
The strongest action available is :meth:`Trainer.cancel_run`, a graceful run-level stop, and it
is opt-in via ``cancel_on_alert`` (default off).
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import torch

from olmo_core.train.callbacks import Callback
from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC

log = logging.getLogger(__name__)

STEP_SKIPPED_METRIC = "optim/step skipped"


@dataclass
class Mamba3SentinelCallback(Callback):
    """
    Detect silent training failures and record them where a watchdog can see them.

    :param run_dir: Directory for ``heartbeat.json`` and ``alerts.jsonl``. Keep this on local
        disk; the watchdog is responsible for shipping it anywhere else.
    :param expected_rotation_block_size: Fail the preflight if the built model disagrees.
    :param sequence_length: Used to judge whether the decay horizon covers the context.
    :param skip_rate_window: Window over which the skipped-step rate is measured.
    :param skip_rate_threshold: Alert when the skipped-step rate over the window exceeds this.
    :param plateau_window: Number of logged steps over which loss improvement is measured.
    :param plateau_min_improvement: Alert when relative loss improvement over the window falls
        below this. Set to ``0`` to disable.
    :param cancel_on_alert: If set, gracefully cancel the *run* on a critical alert. This never
        stops the machine. Default off: notify and keep training.
    """

    priority: ClassVar[int] = -100

    run_dir: str = "."
    expected_rotation_block_size: Optional[int] = None
    sequence_length: Optional[int] = None
    skip_rate_window: int = 25
    skip_rate_threshold: float = 0.5
    plateau_window: int = 50
    plateau_min_improvement: float = 0.001
    cancel_on_alert: bool = False
    heartbeat_every_steps: int = 1

    _skips: List[float] = field(default_factory=list, repr=False)
    _losses: List[float] = field(default_factory=list, repr=False)
    _alerts_seen: Dict[str, int] = field(default_factory=dict, repr=False)
    _last_step_time: float = 0.0
    _started_at: float = 0.0

    @property
    def _heartbeat_path(self) -> Path:
        return Path(self.run_dir) / "heartbeat.json"

    @property
    def _alerts_path(self) -> Path:
        return Path(self.run_dir) / "alerts.jsonl"

    def alert(self, kind: str, message: str, *, critical: bool = False, **detail: Any) -> None:
        """
        Record an alert to the log and to ``alerts.jsonl``.

        Repeated alerts of the same kind are counted but only written every 25 occurrences, so a
        persistent condition cannot flood the file and mask a later, different failure.
        """
        seen = self._alerts_seen.get(kind, 0) + 1
        self._alerts_seen[kind] = seen
        if seen != 1 and seen % 25 != 0:
            return

        record = {
            "ts": time.time(),
            "step": getattr(self.trainer, "global_step", -1) if self._trainer else -1,
            "kind": kind,
            "critical": critical,
            "occurrences": seen,
            "message": message,
            **detail,
        }
        (log.error if critical else log.warning)("SENTINEL %s: %s", kind, message)
        try:
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)
            with self._alerts_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:  # never let telemetry break training
            log.warning("could not write alert file: %s", exc)

        if critical and self.cancel_on_alert:
            # Graceful, run-level only. Does not touch the process or the machine.
            self.trainer.cancel_run(f"sentinel: {kind}")

    def pre_train(self):
        self._started_at = time.time()
        self._last_step_time = time.time()
        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        self._check_rotation_block_size()
        self._check_memory_horizon()
        self._write_heartbeat(status="training")

    def _iter_mixers(self):
        try:
            model = self.trainer.train_module.model
        except (AttributeError, AssertionError):
            return
        for name, module in model.named_modules():
            if type(module).__name__ == "Mamba3Mixer":
                yield name, module

    def _check_rotation_block_size(self) -> None:
        """A config flag that never reaches the kernel leaves the model abelian, silently."""
        if self.expected_rotation_block_size is None:
            return
        found = sorted(
            {
                int(b)
                for _, m in self._iter_mixers()
                if (b := getattr(m, "rotation_block_size", None)) is not None
            }
        )
        if not found:
            self.alert(
                "no_mamba3_mixers",
                "expected Mamba-3 mixers but found none; the model may be pure attention",
                critical=True,
            )
        elif found != [self.expected_rotation_block_size]:
            self.alert(
                "rotation_block_size_mismatch",
                f"expected rotation_block_size={self.expected_rotation_block_size} but the built "
                f"model has {found}",
                critical=True,
                found=found,
            )
        else:
            log.info(
                "sentinel: confirmed rotation_block_size=%d on all Mamba-3 mixers",
                self.expected_rotation_block_size,
            )

    def _check_memory_horizon(self) -> None:
        """
        Report the horizon over which the state retains a signal, from the init distribution.

        ``alpha = exp(dt * A)`` with ``A = -exp(A_log)`` and ``dt = softplus(dt_bias)``. The
        number of steps until an injected signal decays below fp32 resolution is
        ``log(eps) / log(alpha)``. If that is short relative to the sequence length, the run
        will look healthy and learn nothing at long range.
        """
        horizons: List[float] = []
        for _, mixer in self._iter_mixers():
            try:
                with torch.no_grad():
                    a_log = mixer.A_log.detach().float()
                    dt = torch.nn.functional.softplus(mixer.dt_bias.detach().float())
                    alpha = torch.exp(dt * -torch.exp(a_log)).clamp(max=1 - 1e-12)
                    horizon = math.log(1e-6) / torch.log(alpha).clamp(max=-1e-12).min().item()
                horizons.append(horizon)
            except (AttributeError, RuntimeError, ValueError):
                continue
        if not horizons:
            return

        best = max(horizons)
        log.info(
            "sentinel: decay horizon (steps until signal < 1e-6) min=%.0f max=%.0f",
            min(horizons),
            best,
        )
        if self.sequence_length is not None and best < self.sequence_length:
            self.alert(
                "short_memory_horizon",
                f"the longest-memory head retains signal for only ~{best:.0f} steps but the "
                f"sequence length is {self.sequence_length}. Training will appear healthy while "
                f"long-range state tracking stays unreachable. Lower both ends of the A_log "
                f"range ('a_log_init_min' and 'a_log_init_max') together.",
                horizon_steps=round(best),
                sequence_length=self.sequence_length,
            )

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        del step
        self._check_grad_norm(metrics)
        self._check_skip_rate(metrics)
        self._check_plateau(metrics)

    def _check_grad_norm(self, metrics: Dict[str, float]) -> None:
        grad_norm = metrics.get(OPTIM_GRAD_NORM_METRIC)
        if grad_norm is None:
            return
        if not math.isfinite(grad_norm):
            self.alert(
                "nonfinite_grad_norm",
                f"gradient norm is {grad_norm}. The trainer only checks the CE loss, so this "
                f"would not otherwise raise. It also poisons SkipStepOptimizer's rolling "
                f"statistics, silently skipping the next ~129 optimizer steps.",
                critical=True,
                grad_norm=str(grad_norm),
            )
        elif grad_norm == 0.0:
            self.alert(
                "zero_grad_norm",
                "gradient norm is exactly 0; the model is not receiving gradient",
                critical=True,
            )

    @property
    def _skip_window(self) -> int:
        """
        At least 1.

        A zero-length window trims the deque to empty and then averages over ``len() == 0``.
        Clamping here rather than validating in the constructor keeps the contract that this
        callback never raises out of the training loop, whatever it is handed.
        """
        return max(1, self.skip_rate_window)

    @property
    def _plateau_span(self) -> int:
        """At least 2: the plateau check halves its history, so both halves need an element."""
        return max(2, self.plateau_window)

    def _check_skip_rate(self, metrics: Dict[str, float]) -> None:
        skipped = metrics.get(STEP_SKIPPED_METRIC)
        if skipped is None:
            return
        window = self._skip_window
        self._skips.append(float(skipped))
        if len(self._skips) > window:
            self._skips.pop(0)
        if len(self._skips) < window:
            return
        rate = sum(self._skips) / len(self._skips)
        if rate > self.skip_rate_threshold:
            self.alert(
                "high_step_skip_rate",
                f"{rate:.0%} of the last {window} optimizer steps were skipped. "
                f"The GPU is busy but the weights are barely moving. A single NaN loss or grad "
                f"norm poisons the rolling statistics for 129 steps.",
                critical=True,
                skip_rate=round(rate, 3),
            )

    def _check_plateau(self, metrics: Dict[str, float]) -> None:
        loss = metrics.get(TRAIN_CE_LOSS_METRIC)
        if loss is None or not math.isfinite(loss):
            return
        span = self._plateau_span
        self._losses.append(loss)
        if len(self._losses) > span:
            self._losses.pop(0)
        if self.plateau_min_improvement <= 0 or len(self._losses) < span:
            return

        half = span // 2
        earlier = sum(self._losses[:half]) / half
        later = sum(self._losses[half:]) / (len(self._losses) - half)
        if earlier <= 0:
            return
        improvement = (earlier - later) / abs(earlier)
        if improvement < self.plateau_min_improvement:
            self.alert(
                "loss_plateau",
                f"CE loss improved {improvement:+.4%} over the last {span} logged "
                f"steps (from {earlier:.4f} to {later:.4f}), below the "
                f"{self.plateau_min_improvement:.2%} threshold",
                improvement=round(improvement, 6),
            )

    def post_step(self):
        if self.step % max(1, self.heartbeat_every_steps) == 0:
            self._write_heartbeat(status="training")
        self._last_step_time = time.time()

    def _write_heartbeat(self, *, status: str, **extra: Any) -> None:
        """
        Write liveness state for the external watchdog.

        Written atomically via a temp file and rename, so the watchdog can never read a
        half-written file and misdiagnose a healthy run as corrupt.
        """
        now = time.time()
        payload = {
            "ts": now,
            "status": status,
            "step": getattr(self.trainer, "global_step", -1) if self._trainer else -1,
            "seconds_since_last_step": round(now - self._last_step_time, 3),
            "uptime_seconds": round(now - self._started_at, 1),
            "recent_loss": self._losses[-1] if self._losses else None,
            "recent_skip_rate": (
                round(sum(self._skips) / len(self._skips), 3) if self._skips else None
            ),
            "alert_counts": dict(self._alerts_seen),
            **extra,
        }
        try:
            Path(self.run_dir).mkdir(parents=True, exist_ok=True)
            tmp = self._heartbeat_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self._heartbeat_path)
        except OSError as exc:
            log.warning("could not write heartbeat: %s", exc)

    def on_error(self, exc: BaseException):
        """Record the failure so the watchdog reports it even though the process is gone."""
        self.alert("training_error", f"{type(exc).__name__}: {exc}", critical=False)
        self._write_heartbeat(status="error", error=f"{type(exc).__name__}: {exc}")

    def post_train(self):
        self._write_heartbeat(status="finished")
