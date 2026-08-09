import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ...optim import SkipStepOptimizer
from ..common import OPTIM_GRAD_NORM_METRIC, OPTIM_STEP_SKIPPED_METRIC
from .callback import Callback

log = logging.getLogger(__name__)

#: Cumulative count of steps the optimizer declined to apply. Monotone, so its last value is
#: the run's total and no aggregation over the series is needed to read it.
STEPS_SKIPPED_METRIC = "stability/steps skipped"

#: The gradient norm on a step that was skipped. RECORDED ONLY ON THOSE STEPS, so the set of
#: steps this key exists at *is* the list of skipped steps, and no separate index is needed.
SKIP_GRAD_NORM_METRIC = "stability/grad norm at a skipped step"

#: Running maximum of the above. This is the statistic that separates a benign skip from the
#: onset of a loss spike, and one number per run rather than a series to be scanned.
MAX_SKIP_GRAD_NORM_METRIC = "stability/largest grad norm at a skipped step"


@dataclass
class SkipStepMonitorCallback(Callback):
    """
    Record what a :class:`~olmo_core.optim.SkipStepOptimizer` actually did, as a per-run
    outcome rather than as a per-step flag nobody aggregates.

    WHY THIS IS AN OUTCOME AND NOT INSTRUMENTATION. Turning spike skipping on removes an
    instability from the loss, and a loss that no longer carries the instability is a loss the
    instability cannot be read off. An arm that would have spiked in four runs of five instead
    reads a couple of hundredths of a nat worse and gets written up as a result about quality,
    when it is a result about stability. The two are different claims and both are worth
    having, so the skipping is logged with enough detail to make the second one testable:
    **how many** steps were skipped, **which** steps, and **at what gradient norm**.

    Three metrics, and between them they answer all three:

    - :data:`STEPS_SKIPPED_METRIC`, cumulative, so the final value is the count.
    - :data:`SKIP_GRAD_NORM_METRIC`, recorded *only* on a step that was skipped, so the steps
      it exists at are the steps that were skipped and its value is what triggered each.
    - :data:`MAX_SKIP_GRAD_NORM_METRIC`, running, because the count alone does not separate a
      run that skipped a dozen unremarkable steps from one that skipped the first step of a
      spike. Magnitude does: measured over five 370M runs, the largest trigger on a run that
      never spiked was 0.35, against 9.30 and 20.45 on the two that did.

    The numbers are read out of the metrics the train module has already recorded and reduced,
    in :meth:`pre_log_metrics`, so this adds no collective, no host-device sync and no read of
    optimizer state. That is also why it cannot disagree with the optimizer: it reports the
    decision that was taken rather than re-deriving one.

    .. seealso::
        :class:`StabilityMonitorCallback` is the other half and a different question. It is a
        *detector*, applying its own threshold to loss and gradient norm whether or not the
        optimizer acted. This is a *recorder* of what the optimizer did.
    """

    enabled: bool = True

    _skipped_steps: List[int] = field(default_factory=list, repr=False)
    _skipped_grad_norms: List[float] = field(default_factory=list, repr=False)
    _max_grad_norm_at_skip: float = 0.0

    def post_attach(self):
        """
        :raises RuntimeError: If the run's optimizer can never skip a step, so that the count
            this callback reports would be a constant zero that reads like a stable run.
        """
        if not self.enabled:
            return
        if not self._optimizers_can_skip():
            raise RuntimeError(
                f"{type(self).__name__} was added to a run whose optimizer is not a "
                "SkipStepOptimizer, so it would report zero skipped steps for every arm and "
                "the stability outcome would read as a null rather than as an absent "
                "measurement. Build the run with SkipStepAdamWConfig, or drop this callback."
            )

    def _optimizers_can_skip(self) -> bool:
        train_module = self.trainer.train_module
        candidates = [getattr(train_module, "optim", None)]
        candidates.extend(getattr(train_module, "optimizers", None) or [])
        return any(isinstance(optim, SkipStepOptimizer) for optim in candidates)

    @property
    def steps_skipped(self) -> int:
        """How many steps the optimizer has declined to apply so far."""
        return len(self._skipped_steps)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "skipped_steps": list(self._skipped_steps),
            "skipped_grad_norms": list(self._skipped_grad_norms),
            "max_grad_norm_at_skip": self._max_grad_norm_at_skip,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # A second attempt resumes from the last checkpoint, and a count that restarted at
        # zero there would under-report the arm by however much of the run the first attempt
        # got through -- silently, and only on the cells that lost a host.
        self._skipped_steps = list(state_dict.get("skipped_steps", []))
        self._skipped_grad_norms = list(state_dict.get("skipped_grad_norms", []))
        self._max_grad_norm_at_skip = state_dict.get("max_grad_norm_at_skip", 0.0)

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self.enabled:
            return

        skipped = metrics.get(OPTIM_STEP_SKIPPED_METRIC)
        if skipped is None:
            # A step that recorded no optimizer metrics -- the evaluations at step 0, and
            # anything else that reaches the metric flush without an optimizer step.
            return

        grad_norm = metrics.get(OPTIM_GRAD_NORM_METRIC)
        if skipped > 0.5:
            self._skipped_steps.append(step)
            if grad_norm is not None:
                self._skipped_grad_norms.append(grad_norm)
                self._max_grad_norm_at_skip = max(self._max_grad_norm_at_skip, grad_norm)
                metrics[SKIP_GRAD_NORM_METRIC] = grad_norm
            log.warning(
                "Optimizer skipped step %d. Gradient norm %s, %d skipped so far, largest "
                "trigger %.4g.",
                step,
                "unrecorded" if grad_norm is None else f"{grad_norm:.4g}",
                len(self._skipped_steps),
                self._max_grad_norm_at_skip,
            )

        metrics[STEPS_SKIPPED_METRIC] = float(len(self._skipped_steps))
        metrics[MAX_SKIP_GRAD_NORM_METRIC] = self._max_grad_norm_at_skip

    def post_train(self):
        log.info("Skipped-step summary: %s", self.summary())

    def summary(self) -> str:
        """
        One line naming the count, the largest trigger and where the skips fell.

        Written to the run log at the end of training so the stability outcome survives in the
        container's own output as well as in W&B.
        """
        if not self._skipped_steps:
            return "no steps skipped"
        shown = ", ".join(str(s) for s in self._skipped_steps[:32])
        if len(self._skipped_steps) > 32:
            shown += f", ... ({len(self._skipped_steps) - 32} more)"
        return (
            f"{len(self._skipped_steps)} step(s) skipped, largest triggering gradient norm "
            f"{self._max_grad_norm_at_skip:.4g}, at steps {shown}"
        )
