"""
The diagnostics that make a hyper-connection run interpretable, and without which a null
result is not a result.

Every quantity here answers a question that the loss curve cannot. The one that matters most
is the first: a public mHC reproduction measured the residual mixer's gradient norm at about
``1e-9`` against ``1.84`` on the branch weights and concluded the matrix never left its
initialisation. A run that reports only a loss cannot tell that story apart from any other, and
"finite and nonzero" gradient checks pass a ``1e-9`` gradient without comment.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from olmo_core.config import Config

from .callback import Callback, CallbackConfig

__all__ = ["HyperConnectionMonitorCallback", "HyperConnectionMonitorCallbackConfig"]


@dataclass
class HyperConnectionMonitorCallback(Callback):
    """
    Log what the residual mixers are doing, beside a reference parameter so the numbers mean
    something.

    :param interval: How many steps between readings of the scalar diagnostics.
    :param matrix_interval: How many steps between logging the ``H_res`` matrices themselves,
        entry by entry. Much rarer than ``interval`` because it is ``n * n`` series per wrapped
        sub-layer.
    :param reference_parameter: A substring identifying the parameter every gradient norm is
        reported relative to. The attention output projection is the default: it is in the same
        block, it is always present, and it is the parameter the reproduction that produced the
        ``1.84`` used.
    """

    interval: int = 50
    matrix_interval: int = 500
    reference_parameter: str = "attention.w_out.weight"

    # A ClassVar, as `Callback` declares it: a dataclass field of the same name would shadow
    # the class variable the trainer sorts on. After the optimizer step would be too late --
    # `pre_optim_step` is the last moment this step's gradients are still on the parameters.
    priority: ClassVar[int] = -1

    def __post_init__(self):
        self._initial: Dict[str, torch.Tensor] = {}

    def pre_train(self):
        """
        Snapshot every mixer at initialisation, so displacement has something to be measured
        from.

        Taken here rather than in ``__init__`` because a resumed run has already loaded its
        checkpoint by this point, and a displacement measured from a resumed state would silently
        restart at zero half way through a run.
        """
        from olmo_core.nn.hyper_connections import HyperConnection

        for name, module in self.trainer.train_module.model.named_modules():
            if isinstance(module, HyperConnection):
                self._initial[name] = module.residual_mixer().detach().float().clone()

    def _hyper_connections(self) -> List[Tuple[str, "torch.nn.Module"]]:
        from olmo_core.nn.hyper_connections import HyperConnection

        return [
            (name, module)
            for name, module in self.trainer.train_module.model.named_modules()
            if isinstance(module, HyperConnection)
        ]

    def _reference_gradient_norm(self) -> Optional[float]:
        total = 0.0
        found = False
        for name, parameter in self.trainer.train_module.model.named_parameters():
            if self.reference_parameter in name and parameter.grad is not None:
                total += float(parameter.grad.detach().float().norm() ** 2)
                found = True
        return math.sqrt(total) if found else None

    def pre_optim_step(self):
        """
        Read the mixers and their gradients, while the gradients are still there.
        """
        step = self.trainer.global_step
        if self.interval <= 0 or step % self.interval != 0:
            return

        reference = self._reference_gradient_norm()
        connections = self._hyper_connections()
        if not connections:
            return

        displacements: List[float] = []
        gradient_norms: List[float] = []
        largest_logit = 0.0
        worst_row_error = 0.0

        for name, module in connections:
            mixer = module.residual_mixer().detach().float()
            initial = self._initial.get(name)
            if initial is not None and float(initial.norm()) > 0:
                displacement = float((mixer - initial).norm() / initial.norm())
                displacements.append(displacement)
                self.trainer.record_metric(f"hc/{name}/mixer displacement", displacement)

            logits = getattr(module, "h_res_logits", None)
            if logits is not None:
                largest_logit = max(largest_logit, float(logits.detach().abs().max()))
                if logits.grad is not None:
                    norm = float(logits.grad.detach().float().norm())
                    gradient_norms.append(norm)
                    self.trainer.record_metric(f"hc/{name}/H_res grad norm", norm)

            # The doubly stochastic guarantee, checked rather than assumed. Sinkhorn's twenty
            # iterations converge only while the logits stay small: measured over 200 draws the
            # row-sum error is already 1.3e-2 at a maximum absolute logit of about 6.5. The
            # column sums survive at every scale because they are normalised last, so the row
            # sums are the reading that matters, and a row sum below 1 shrinks that stream's
            # residual -- at which point the run is no longer doing what the method says.
            worst_row_error = max(worst_row_error, float((mixer.sum(-1) - 1).abs().max()))

        if displacements:
            self.trainer.record_metric(
                "hc/mixer displacement", sum(displacements) / len(displacements)
            )
        if gradient_norms:
            mean_gradient = sum(gradient_norms) / len(gradient_norms)
            self.trainer.record_metric("hc/H_res grad norm", mean_gradient)
            if reference is not None and reference > 0:
                # THE HEADLINE NUMBER. A bare gradient norm is unreadable because it scales with
                # the loss and the batch; the ratio against a parameter in the same block is
                # what the 1e-9-against-1.84 reading was, and it is what the treatment is
                # predicted to move.
                self.trainer.record_metric(
                    "hc/H_res grad norm over reference", mean_gradient / reference
                )
        if reference is not None:
            self.trainer.record_metric("hc/reference grad norm", reference)
        self.trainer.record_metric("hc/largest residual logit", largest_logit)
        self.trainer.record_metric("hc/doubly stochastic row error", worst_row_error)

        if self.matrix_interval > 0 and step % self.matrix_interval == 0:
            for name, module in connections:
                mixer = module.residual_mixer().detach().float()
                for row in range(mixer.shape[0]):
                    for column in range(mixer.shape[1]):
                        self.trainer.record_metric(
                            f"hc/{name}/H_res[{row}][{column}]", float(mixer[row, column])
                        )


@dataclass
class HyperConnectionMonitorCallbackConfig(CallbackConfig, Config):
    """
    A config for building a :class:`HyperConnectionMonitorCallback`.
    """

    interval: int = 50
    matrix_interval: int = 500
    reference_parameter: str = "attention.w_out.weight"

    def build(self, trainer) -> Optional[Callback]:
        """
        :param trainer: The trainer, unused.

        :returns: The callback.
        """
        del trainer
        return HyperConnectionMonitorCallback(
            interval=self.interval,
            matrix_interval=self.matrix_interval,
            reference_parameter=self.reference_parameter,
        )
