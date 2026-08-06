"""
Steady-state throughput reporting: a fixed measurement window, a median, and a printed
denominator.

WHAT THIS ADAPTS RATHER THAN REPLACES.

``SpeedMonitorCallback`` already computes everything expensive -- FLOPs per token from the model
itself (``speed_monitor.py:145-151``), the device BF16 peak with a correct per-card table and no
fall-through default (``:104-149``), tokens per step accounting for parallel degree (``:168``), and
both an instantaneous and a running-average MFU (``:239-249``). **None of that is reimplemented
here.** This callback reads ``SpeedMonitorCallback``'s own per-step quantities and does the one
thing it structurally cannot: report a **median over a fixed window that starts after steady
state**, with the denominator printed alongside.

Why ``SpeedMonitorCallback`` structurally cannot. Its two MFU metrics are:

* ``throughput/device/MFU`` -- instantaneous, from the single most recent step's wall time. Noisy;
  a checkpoint-save step or an eval step lands in it and is indistinguishable from a slow step.
* ``throughput/device/MFU (actual avg)`` -- cumulative from ``_start_time``, which is set at the
  *second* step (``:186-193`` skips only the first). So it averages in ``torch.compile``'s
  multi-minute step 2, every cold-shard stall, every checkpoint write, and every eval, and it can
  never forget them -- a cumulative mean is monotonically contaminated.

**That is why those two differed by ~6 points on a sibling run, and the gap IS the
warmup-contamination signal** rather than a bug in either. This callback resolves the ambiguity by
reporting a third thing whose exclusions are explicit and logged.

WHAT IT REFUSES TO DO.

It does not report a tokens/s number without its denominator. Every ``RESULT `` line it emits
carries the FLOPs formula, the peak it divided by, the window boundaries, and the count of steps
excluded and why. A throughput figure with no denominator is how a benchmark reports the wrong sign
and reproduces cleanly, and this project has paid for that once already.

It also does not report a *bandwidth* figure it cannot substantiate. ``torch.cuda`` exposes no
achieved-bytes counter, so an "achieved GB/s" here would be a model masquerading as a measurement.
Instead it prints the **working set** -- which it can compute exactly -- so a reader can see at a
glance whether the measured subgraph is large enough that a cache-resident result is impossible.
That is the specific failure this project recorded: a subgraph that fits in cache reports the wrong
sign and replicates cleanly. Working set is the diagnostic that catches it; a fabricated bandwidth
number is not.
"""

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional

import torch

from olmo_core.distributed.utils import get_world_size

from ..train_module import TransformerTrainModule
from .callback import Callback
from .speed_monitor import SpeedMonitorCallback

log = logging.getLogger(__name__)

__all__ = ["SteadyStateThroughputCallback"]


@dataclass
class SteadyStateThroughputCallback(Callback):
    """
    Reports median step time and MFU over a fixed, explicitly-bounded window that begins after
    steady state, excluding checkpoint-save and evaluation steps.

    Add alongside :class:`SpeedMonitorCallback`, which it reads from rather than duplicating.
    """

    # Lower than SpeedMonitorCallback's -2 so that this runs AFTER it within a step and can read
    # the per-step quantities it just computed. Callbacks run highest-priority-first
    # (callback.py:24-27).
    priority: ClassVar[int] = -3

    warmup_steps: int = 50
    """
    Steps to discard before the window opens. 50 is the project's standing figure and it is not
    arbitrary: ``torch.compile`` alone can take minutes on the first compiled step, and the
    S3-mmap loader is I/O-bound early.
    """

    window_steps: int = 100
    """
    Steps in the measurement window. With ``warmup_steps=50`` this is the documented 50-150 window.
    """

    _step_times: List[float] = field(default_factory=list)
    _step_flops: List[int] = field(default_factory=list)
    _step_tokens: List[int] = field(default_factory=list)
    _last_step_end: Optional[float] = None
    _excluded: Dict[str, int] = field(default_factory=dict)
    _reported: bool = False
    _saved_this_step: bool = False
    _evaled_this_step: bool = False

    @property
    def _speed_monitor(self) -> Optional[SpeedMonitorCallback]:
        for cb in self.trainer.callbacks.values():
            if isinstance(cb, SpeedMonitorCallback):
                return cb
        return None

    def _exclude(self, reason: str):
        self._excluded[reason] = self._excluded.get(reason, 0) + 1

    def pre_train(self):
        sm = self._speed_monitor
        if sm is None:
            log.warning(
                "SteadyStateThroughputCallback found no SpeedMonitorCallback to read from; "
                "no steady-state throughput will be reported."
            )
            return

        # Print the denominator ONCE, up front, before any number depends on it. If the run dies
        # later this line is still in the log and the numbers that did appear are interpretable.
        tm = self.trainer.train_module
        peak = sm.device_peak_flops_per_second
        print(
            f"RESULT throughput_denominator device="
            f"{torch.cuda.get_device_name(self.trainer.device) if self.trainer.device.type == 'cuda' else self.trainer.device.type} "
            f"peak_bf16_dense_flops_per_s={peak} "
            f"window=[{self.warmup_steps},{self.warmup_steps + self.window_steps}) "
            f"world_size={get_world_size()}",
            flush=True,
        )
        if peak is None:
            print(
                "RESULT throughput_denominator WARNING peak is None -- this device is not in "
                "speed_monitor.py's table, so MFU is deliberately not reported rather than "
                "reported against a borrowed peak. tokens/s is still valid.",
                flush=True,
            )

        # The FLOPs formula, printed so MFU is auditable rather than trusted.
        print(
            "RESULT flops_formula per_token = sum_over_blocks[ 6*block_params "
            "+ 12*n_heads*head_dim*min(window_size, seq_len) ] + 6*lm_head_params ; "
            "MoE blocks contribute 6*router_params + 6*int(expert_params*top_k/num_experts). "
            "Source: nn/transformer/model.py:1018-1029, nn/attention/__init__.py:791-809, "
            "nn/moe/moe.py:337-343, nn/lm_head.py:422-425.",
            flush=True,
        )
        print(
            "RESULT flops_formula NOTE this is NOT '6*(active_params - embed_params)'. That "
            "expression omits the attention score term and mis-credits the head; at R3 it is "
            "0.906x the counted figure, i.e. 9.4% low. Quote the counted formula above, since it "
            "is what the reported MFU actually divides.",
            flush=True,
        )

        if isinstance(tm, TransformerTrainModule):
            self._print_working_set(tm)

    def _print_working_set(self, tm: TransformerTrainModule):
        """
        Print the working set, so a reader can rule out a cache-resident measurement by inspection.

        A subgraph that fits in cache reports the wrong sign and replicates cleanly. This is the
        check that catches it. An A100's L2 is 40 MiB; anything here in the GiB is unambiguously
        HBM-resident, and saying so with a number beats asserting it.
        """
        model = tm.model
        params = model.num_params
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        mb = tm.rank_microbatch_size

        l2_bytes = None
        if self.trainer.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.trainer.device)
            l2_bytes = getattr(props, "L2_cache_size", None)

        msg = (
            f"RESULT working_set params={params} param_bytes={param_bytes} "
            f"rank_microbatch_tokens={mb}"
        )
        if l2_bytes:
            ratio = param_bytes / l2_bytes
            msg += f" l2_cache_bytes={l2_bytes} param_bytes_over_l2={ratio:.1f}x"
            if param_bytes < l2_bytes:
                msg += (
                    " WARNING the parameter working set FITS IN L2. A measurement of this "
                    "subgraph can report the wrong sign and reproduce cleanly. Do not quote a "
                    "throughput number from it without saying so."
                )
        else:
            msg += " l2_cache_bytes=unknown"
        print(msg, flush=True)

    def post_checkpoint_saved(self, path):
        """
        Mark this step as a checkpoint step so it is excluded from the window.

        The signature takes ``path`` because the base hook does (``callback.py:127``); it is
        unused here. Getting this wrong would be a silent miss rather than an error -- Python
        would raise ``TypeError`` at the first save, i.e. 30 minutes into a paid run -- so it is
        matched deliberately rather than by memory.
        """
        del path
        self._saved_this_step = True

    def pre_step(self, batch):
        """
        Detect an eval step by watching the evaluator callbacks' own trigger condition.

        There is **no** ``post_eval_batch`` hook on ``Callback`` -- the eval callbacks drive
        evaluation from inside their own ``post_step`` (``evaluator_callback.py:107-114``). So an
        eval's cost lands in the *next* interval this callback measures, and the only way to
        exclude it without inventing a hook is to notice, at the start of a step, that the
        previous step was one an evaluator would have fired on.

        This is checked against the evaluators actually attached rather than assumed, so a run
        with evals disabled (which is the platform default -- both evaluators die at trainer
        construction per the project's platform constraints) excludes nothing and pays nothing.
        """
        del batch
        from .evaluator_callback import EvaluatorCallback

        prev_step = self.trainer.global_step - 1
        if prev_step <= 0:
            return
        for cb in self.trainer.callbacks.values():
            if not isinstance(cb, EvaluatorCallback):
                continue
            interval = getattr(cb, "eval_interval", None)
            fixed = getattr(cb, "fixed_steps", None) or ()
            if (interval and prev_step % interval == 0) or prev_step in fixed:
                self._evaled_this_step = True
                return

    def post_step(self):
        sm = self._speed_monitor
        if sm is None:
            return

        now = time.perf_counter()
        prev, self._last_step_end = self._last_step_end, now

        step = self.trainer.global_step
        saved, evaled = self._saved_this_step, self._evaled_this_step
        self._saved_this_step = self._evaled_this_step = False

        if prev is None:
            # No interval to measure yet.
            return
        if step <= self.warmup_steps:
            self._exclude("warmup")
            return
        if len(self._step_times) >= self.window_steps:
            if not self._reported:
                self._report()
            return
        if saved:
            self._exclude("checkpoint_save")
            return
        if evaled:
            self._exclude("eval")
            return

        self._step_times.append(now - prev)
        self._step_flops.append(sm._step_flops)
        self._step_tokens.append(sm._step_tokens)

    def post_train(self):
        if not self._reported:
            self._report()

    def _report(self):
        self._reported = True
        n = len(self._step_times)
        if n == 0:
            print(
                f"RESULT steady_state INCOMPLETE steps_in_window=0 "
                f"excluded={self._excluded} "
                f"-- the run ended before the window opened. Report NO throughput number from "
                f"this run rather than reporting the contaminated cumulative average.",
                flush=True,
            )
            return

        med = statistics.median(self._step_times)
        # Report the spread too. A median with no spread hides a bimodal step time, which is what
        # a periodic cold-shard stall looks like -- and the loader touches new shards MID-RUN, so
        # warmup does not cover it.
        lo, hi = min(self._step_times), max(self._step_times)
        p90 = sorted(self._step_times)[int(0.9 * (n - 1))]

        tokens = statistics.median(self._step_tokens) if self._step_tokens else 0
        flops = statistics.median(self._step_flops) if self._step_flops else 0

        tps = tokens / med if med > 0 else 0.0
        fps = flops / med if med > 0 else 0.0

        sm = self._speed_monitor
        peak = sm.device_peak_flops_per_second if sm is not None else None

        print(
            f"RESULT steady_state steps_in_window={n} "
            f"window=[{self.warmup_steps},{self.warmup_steps + self.window_steps}) "
            f"median_step_s={med:.4f} min_s={lo:.4f} p90_s={p90:.4f} max_s={hi:.4f} "
            f"max_over_median={hi / med:.2f}x "
            f"excluded={self._excluded}",
            flush=True,
        )
        print(
            f"RESULT steady_state tokens_per_s_per_device={tps:.1f} "
            f"counted_flops_per_s_per_device={fps:.4e} "
            f"counted_flops_per_step={flops}",
            flush=True,
        )
        if peak:
            print(
                f"RESULT steady_state MFU_median={100 * fps / peak:.3f}% "
                f"peak_used={peak} "
                f"-- median-based, warmup/checkpoint/eval excluded. Compare against "
                f"'throughput/device/MFU (actual avg)', which includes all three.",
                flush=True,
            )
        else:
            print(
                "RESULT steady_state MFU=not_reported (no known BF16 peak for this device; "
                "see speed_monitor.py:128-149 for why it is not defaulted)",
                flush=True,
            )
        if hi / med > 1.5:
            print(
                f"RESULT steady_state WARNING max/median={hi / med:.2f}x -- the step time is not "
                f"unimodal inside the window. Most likely a cold S3-mmap shard boundary. The "
                f"median is still the right statistic but the mean would be wrong, and a "
                f"throughput claim from this run should quote the spread.",
                flush=True,
            )
