import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import torch

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class GPUMemoryMonitorCallback(Callback):
    """
    Adds metrics for GPU memory statistics.
    """

    priority: ClassVar[int] = -1
    device_id: Optional[int] = None
    _num_alloc_retries: int = 0

    # RUNNING MAXIMA ACROSS THE WHOLE RUN, held here because after `post_step` nothing else can
    # know them.
    #
    # `post_step` calls `reset_peak_memory_stats()` on **every** step -- deliberately, so that the
    # per-step series is a per-step peak rather than a monotone staircase. The cost is that
    # `torch.cuda.max_memory_allocated()` read AFTER `fit()` returns reports only the *final
    # step's* tail. That is a truncated window which looks exactly like a whole-run peak, and it
    # is the field a reader naturally reaches for as a fit criterion. Accumulating immediately
    # before the reset is the only point at which the information still exists.
    #
    # Bytes, not GiB, so the running max is exact integer arithmetic rather than float.
    peak_active_bytes: int = 0
    peak_reserved_bytes: int = 0

    def current_allocated_bytes(self) -> int:
        """Bytes currently allocated -- **resident state, not a peak.**

        `memory_allocated()` is instantaneous, so unlike every other number this callback
        publishes it does not mix persistent state with transient activation high-water marks.
        Sampled from `post_step`, i.e. after the optimizer step and before the next forward, it
        reads persistent state with activations at their floor: parameters, gradients, optimizer
        state and any accumulator that survives the step boundary, and nothing else.

        It is therefore the only unambiguous reading of resident state this instrument has, and
        the reason it exists: a peak attributes to no single term, so a peak that comes back at a
        plausible value cannot be checked against a term-by-term memory model.

        **Not counted by this or any other allocator-based number:** the CUDA context and NCCL
        communication buffers, which live outside the caching allocator entirely. The gap between
        this and `reserved` is allocator slack and fragmentation, not those.

        Uses the public `memory_allocated` rather than indexing `memory_stats` for
        `"allocated_bytes.all.current"`, which is what it returns -- a stat-dict key is an
        internal string and a rename would be a `KeyError` at step 1 of a paid run.
        """
        return torch.cuda.memory_allocated(self.device)

    @property
    def device(self) -> torch.device:
        return (
            torch.device("cuda")
            if self.device_id is None
            else torch.device(f"cuda:{self.device_id}")
        )

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device)

    @property
    def device_capacity(self) -> int:
        return torch.cuda.get_device_properties(self.device).total_memory

    def pre_train(self):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        log.info(
            f"GPU capacity: {self.device_name} with {self._to_gib(self.device_capacity):.2f}GiB memory "
            f"of which {self._to_gib(torch.cuda.memory_allocated()):.2f}GiB is currently allocated and "
            f"{self._to_gib(torch.cuda.memory_reserved()):.2f}GiB is currently reserved."
        )

    def post_step(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)
        self.trainer.record_metric("gpu_memory/GPU active mem (GiB)", max_active_gib)
        self.trainer.record_metric("gpu_memory/GPU active mem (%)", max_active_pct)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)
        self.trainer.record_metric("gpu_memory/GPU reserved mem (GiB)", max_reserved_gib)
        self.trainer.record_metric("gpu_memory/GPU reserved mem (%)", max_reserved_pct)

        # CURRENT-ALLOCATED -- resident state, the only unambiguous reading here.
        #
        # Two names emitted where one would do, because the four existing series above are all
        # peaks and NONE of them says so in its name. `record_metric` is where the naming rule
        # ("name the denominator where there is any doubt") is actually consumed, and "GPU active
        # mem (GiB)" reads like current allocation while being a per-step peak. The new names
        # spell out `current` and `step peak` so no reader has to know which is which; the four
        # legacy names are left alone because W&B history and every existing gate depend on them.
        current_allocated = self.current_allocated_bytes()
        self.trainer.record_metric(
            "gpu_memory/current allocated (GiB)", self._to_gib(current_allocated)
        )
        self.trainer.record_metric(
            "gpu_memory/current allocated (%)", self._to_pct(current_allocated)
        )
        self.trainer.record_metric("gpu_memory/step peak active (GiB)", max_active_gib)
        self.trainer.record_metric("gpu_memory/step peak reserved (GiB)", max_reserved_gib)

        # Running maxima, accumulated BEFORE the reset below -- the whole-run peaks that
        # `max_memory_allocated()` after `fit()` cannot see. Emitted per step as well as held on
        # the callback, so a run killed by the wall clock or an OOM still leaves its true peak in
        # the log rather than only in a summary that never printed.
        self.peak_active_bytes = max(self.peak_active_bytes, max_active)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, max_reserved)
        self.trainer.record_metric(
            "gpu_memory/run peak active (GiB)", self._to_gib(self.peak_active_bytes)
        )
        self.trainer.record_metric(
            "gpu_memory/run peak reserved (GiB)", self._to_gib(self.peak_reserved_bytes)
        )

        num_retries = cuda_info["num_alloc_retries"]
        if num_retries > self._num_alloc_retries:
            log.warning(f"{num_retries} CUDA memory allocation retries.")
            self._num_alloc_retries = num_retries

        num_ooms = cuda_info["num_ooms"]
        if num_ooms > 0:
            log.warning(f"{num_ooms} CUDA OOM errors thrown.")

        torch.cuda.reset_peak_memory_stats()

    def state_dict(self) -> Dict[str, Any]:
        """Carry the running maxima across a resume.

        Without this, "run peak" silently means "peak since the last restart". The platform's
        training profile mandates a 30-minute checkpoint interval with `resume_required`, so a
        long run is a *sequence* of processes and the un-checkpointed version of this number would
        understate the true peak by however much the pre-resume segment exceeded the post-resume
        one -- an under-report, in the direction that makes a shape look like it fits.
        """
        return {
            "peak_active_bytes": self.peak_active_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # `.get` rather than `[]`: a checkpoint written before this field existed has no such key,
        # and resuming from one must not raise. Zero is the correct starting value there.
        self.peak_active_bytes = state_dict.get("peak_active_bytes", 0)
        self.peak_reserved_bytes = state_dict.get("peak_reserved_bytes", 0)

    def _to_pct(self, memory: float) -> float:
        return 100 * memory / self.device_capacity

    def _to_gib(self, memory_in_bytes: int) -> float:
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib
