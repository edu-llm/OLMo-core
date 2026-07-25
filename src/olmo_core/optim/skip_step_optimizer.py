from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from typing_extensions import TypeAlias

from olmo_core.utils import get_default_device, move_to_device

ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class SkipStepOptimizer(Optimizer):
    """
    A :class:`SkipStepOptimizer` is an optimizer that can skip updates when the loss or gradient
    norm for a step is above a certain threshold of standard deviations computed over a rolling
    interval.

    Non-finite losses and gradient norms are skipped and are also excluded from the rolling
    statistics, so a single bad step costs exactly one step rather than poisoning the window.

    .. important::
        When using a :class:`SkipStepOptimizer` you must always set :data:`latest_loss` and
        :data:`latest_grad_norm` to the current loss and grad norm, respectively, *before* calling
        :meth:`step()`.

        The :class:`~olmo_core.train.train_module.TransformerTrainModule` will automatically set
        the :data:`latest_loss` and :data:`latest_grad_norm` whenever its optimizer is a subclass of
        :class:`SkipStepOptimizer`.

    .. tip::
        When implementing a :class:`SkipStepOptimizer` you should be careful to avoid host-device
        syncs. You can use :meth:`get_step_factor()` within your :meth:`step()` method to do this.
        See the implementation of :class:`SkipStepLion` for an example.
    """

    def __init__(
        self,
        params: ParamsT,
        defaults: Dict[str, Any],
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ) -> None:
        super().__init__(params, defaults)
        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.numel() > 0:
                        self._device = p.device
                        break
            if self._device is None:
                self._device = get_default_device()
        return self._device

    @property
    def latest_loss(self) -> Optional[torch.Tensor]:
        if not self._losses:
            return None
        else:
            return self._losses[-1]

    @latest_loss.setter
    def latest_loss(self, loss: torch.Tensor):
        self._losses.append(loss)
        while len(self._losses) > self.rolling_interval_length + 1:
            self._losses.pop(0)

    @property
    def latest_grad_norm(self) -> Optional[torch.Tensor]:
        if not self._grad_norms:
            return None
        else:
            return self._grad_norms[-1]

    @latest_grad_norm.setter
    def latest_grad_norm(self, grad_norm: torch.Tensor):
        self._grad_norms.append(grad_norm)
        while len(self._grad_norms) > self.rolling_interval_length + 1:
            self._grad_norms.pop(0)

    def _passes_threshold(
        self, history: List[torch.Tensor], latest: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Whether ``latest`` is within ``sigma_factor`` standard deviations of ``history``.

        Non-finite entries in ``history`` are excluded from the statistics. This matters a great
        deal: ``torch.std_mean`` over a window containing a single NaN returns NaN for both the
        mean and the std, and every subsequent ``<=`` comparison against NaN is ``False``. That
        made one bad step skip the next ``rolling_interval_length + 1`` steps (129 at the
        default) rather than just itself, with the loss looking perfectly healthy throughout
        because the weights had simply stopped moving.

        A non-finite ``latest`` is still rejected -- skipping that step is the whole point.

        Kept free of host-device syncs (no ``.item()``, no Python branching on tensor values),
        as required by the class contract.
        """
        assert latest is not None
        values = torch.stack(history)
        finite = torch.isfinite(values)
        count = finite.sum()
        zeros = torch.zeros_like(values)

        mean = torch.where(finite, values, zeros).sum() / count.clamp(min=1)
        variance = torch.where(finite, (values - mean) ** 2, zeros).sum() / (count - 1).clamp(min=1)
        threshold = mean + self.sigma_factor * variance.sqrt()

        # +inf fails every comparison, so this rejects NaN and -inf as well as +inf.
        current = torch.where(torch.isfinite(latest), latest, torch.full_like(latest, float("inf")))
        judged = current <= threshold
        # Fewer than two finite samples leaves the std undefined, so there is no threshold to
        # judge against and the step is skipped. That matches the previous behaviour, which got
        # there by accident: `torch.std_mean` of a single sample returns NaN, and every
        # comparison against NaN is False. It is also the right call on its own terms -- once
        # the outer guard has passed, reaching this branch means nearly the whole window is
        # non-finite. It self-heals, because finite losses keep refilling the window.
        return torch.where(count >= 2, judged, torch.zeros_like(judged))

    @torch._dynamo.disable()
    def get_step_factor(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the optimizer should proceed with the step
        and `0.0` if the optimizer should skip the step.

        The tensor can be used within the optimizer's step computation to essentially skip a step
        without a host-device sync.
        """
        if len(self._losses) < max(2, self.rolling_interval_length // 2):
            return move_to_device(torch.tensor(1.0), self.device)

        step_factor = self._passes_threshold(self._losses[:-1], self.latest_loss)
        if self._grad_norms:
            step_factor = torch.logical_and(
                step_factor,
                self._passes_threshold(self._grad_norms[:-1], self.latest_grad_norm),
            )

        return step_factor.float()

    @property
    def step_skipped(self) -> torch.Tensor:
        """
        Returns a float tensor which will be `1.0` if the step was skipped and `0.0` otherwise.
        """
        return 1 - self.get_step_factor()
