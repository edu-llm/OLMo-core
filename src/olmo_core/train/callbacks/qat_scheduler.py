import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["QATSchedulerCallback", "set_quant_enabled", "count_quant_modules"]


def _quant_modules(model: nn.Module) -> List[nn.Module]:
    """Every module carrying a ``quant_enabled`` switch, quantized or not."""
    return [m for m in model.modules() if hasattr(m, "quant_enabled")]


def count_quant_modules(model: nn.Module) -> Tuple[int, int]:
    """
    Count quantizable modules and how many currently have the quantizer on.

    :param model: The model to inspect.

    :returns: ``(enabled, total)``.
    """
    modules = _quant_modules(model)
    return sum(1 for m in modules if m.quant_enabled), len(modules)


def set_quant_enabled(model: nn.Module, enabled: bool) -> int:
    """
    Turn ternary QAT on or off across a whole model.

    :param model: The model to modify.
    :param enabled: Whether the quantizer should fire.

    :returns: How many modules were switched.

    :raises OLMoConfigurationError: If the model has no quantizable modules, which means it was
        built with ``quantize=None`` and no schedule can turn a quantizer on later.
    """
    modules = _quant_modules(model)
    if not modules:
        raise OLMoConfigurationError(
            "no quantizable modules found -- the model was built with `quantize=None`, which "
            "produces plain `nn.Linear` and stacked expert weights with no quantizer to "
            "schedule. Build with `quantize='control'` to get the modules in place and let "
            "this callback switch them on."
        )
    switched = 0
    for module in modules:
        if module.quant_enabled is not enabled:
            module.quant_enabled = enabled
            switched += 1
    return switched


@dataclass
class QATSchedulerCallback(Callback):
    """
    Train in full precision first, then switch ternary QAT on for the tail of the run.

    Quantization-aware training from step 0 is the expensive way to reach a ternary model and
    not the accurate one. The QAT scaling-law literature finds that the loss penalty is smallest
    when full-precision training establishes the weights and QAT occupies only a final fraction
    of the token budget, and that the penalty *grows* with the number of tokens trained under
    quantization. Running the quantizer for the whole budget therefore pays the QAT tax on every
    step and lands at a worse loss than spending the same compute on a full-precision phase
    followed by a QAT phase.

    The model must be built with the quantizer present but able to start off -- ``quantize`` set
    to the control state -- so that module classes, parameter names and state-dict keys are
    fixed for the whole run and the switch changes arithmetic only.

    .. important::
        This callback changes what the model computes at :data:`start_step`. Loss will step up
        at the transition; that is the quantizer engaging, not a bug.

    :param start_step: Step at which the quantizer turns on. Mutually exclusive with
        :data:`start_fraction`.
    :param start_fraction: Fraction of the total training run after which the quantizer turns
        on, e.g. ``0.7`` for the last 30% of steps. Requires the trainer to know its own
        duration in steps.
    :param enabled: Set ``False`` to disable the schedule and leave the model as built.
    """

    start_step: Optional[int] = None
    start_fraction: Optional[float] = None
    enabled: bool = True

    _resolved_start: Optional[int] = None
    _switched: bool = False

    def pre_train(self):
        if not self.enabled:
            return

        if (self.start_step is None) == (self.start_fraction is None):
            raise OLMoConfigurationError(
                "the QAT scheduler needs exactly one of `start_step` or `start_fraction`"
            )

        if self.start_step is not None:
            if self.start_step < 0:
                raise OLMoConfigurationError("`start_step` must not be negative")
            self._resolved_start = self.start_step
        else:
            assert self.start_fraction is not None
            if not 0.0 <= self.start_fraction <= 1.0:
                raise OLMoConfigurationError("`start_fraction` must be within [0, 1]")
            max_steps = self.trainer.max_steps
            if max_steps is None:
                raise OLMoConfigurationError(
                    "`start_fraction` needs a run whose duration is known in steps; this "
                    "trainer's `max_duration` does not resolve to one, so use `start_step`"
                )
            self._resolved_start = int(self.start_fraction * max_steps)

        # Fail at setup rather than silently never firing.
        enabled_now, total = count_quant_modules(self.trainer.train_module.model)
        if total == 0:
            raise OLMoConfigurationError(
                "the QAT scheduler found no quantizable modules -- the model was built with "
                "`quantize=None`, which produces plain `nn.Linear` and stacked expert weights "
                "with no quantizer to schedule. Build with the control state of `quantize` so "
                "the modules are in place and this callback can switch them on."
            )

        if self.step >= self._resolved_start:
            # Resuming past the transition: switch on immediately so a restart matches an
            # uninterrupted run.
            self._engage()
        elif enabled_now:
            log.warning(
                "the QAT scheduler will run a full-precision phase until step %d, but the "
                "model was built with the quantizer already on; turning it off for that phase",
                self._resolved_start,
            )
            set_quant_enabled(self.trainer.train_module.model, False)

    def pre_step(self, batch):
        del batch
        if not self.enabled or self._switched:
            return
        assert self._resolved_start is not None
        if self.step >= self._resolved_start:
            self._engage()

    def _engage(self) -> None:
        switched = set_quant_enabled(self.trainer.train_module.model, True)
        _, total = count_quant_modules(self.trainer.train_module.model)
        self._switched = True
        log.info(
            "ternary QAT engaged at step %d: %d of %d quantizable modules switched on",
            self.step,
            switched,
            total,
        )
