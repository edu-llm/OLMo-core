import dataclasses
import functools as ft
import logging
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional, Tuple

import torch

from olmo_core.distributed.utils import get_full_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.residual_stream import HyperConnectionStream, sinkhorn_knopp

from ..common import MetricMergeStrategy, ReduceType
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class HyperConnectionMonitorCallback(Callback):
    """
    Instrumentation for a run whose residual stream has been widened by
    :class:`~olmo_core.nn.residual_stream.HyperConnectionStream`.

    Four things are logged, each of which answers a question a final loss number cannot:

    - **Per-lane norm**, the spread across lanes, and the dispersion of the lanes about their
      own mean. If the lanes never differentiate then the mechanism is inert, and neither a
      positive nor a negative downstream result says anything about hyper-connections. This is
      the primary guard, and :data:`fail_closed_by_step` turns it into an error rather than a
      plot nobody looks at.
    - **Spectral radius of the lane-mixing matrix** per layer. Parcae (arXiv 2604.12946) found
      diverging looped runs learn a radius at or above 1 while converging ones stay below, and
      Tencent's 3B divergence had a multi-lane drift signature. This is how you see it coming.
    - **Condition number of the composite mapping** across depth, i.e. of the product of every
      layer's mixing matrix. mHC's argument for constraining that matrix to the Birkhoff polytope
      is that doubly stochastic matrices are closed under multiplication, so the composite stays
      well conditioned. Measuring it is how that gets tested rather than cited.
    - **Hidden-state norm per layer**. RMSNorm readouts are scale-invariant, so cross-entropy
      cannot see hidden-state scale at all, and pre-norm stacks have been measured driving norms
      into the 10^3 to 10^4 range invisibly (arXiv 2606.24898).

    The matrices are read from the static parameters at :meth:`pre_optim_step`, outside the
    forward pass, so this stays correct under FSDP without a collective inside a hook. Under DHC
    the per-token term is a small perturbation around the static matrix, and it is the static
    matrix that drifts.
    """

    enabled: bool = True

    interval: int = 50
    """
    How often, in steps, to measure. Everything here is norms plus one small eigendecomposition
    per layer, but the activation statistics do cost a pass over the lane tensor.
    """

    fail_closed_by_step: Optional[int] = None
    """
    If set, raise once this step is reached and the lanes have still not differentiated. A
    rehearsal should set this; a real run generally should not, because by then the question has
    been answered.
    """

    min_lane_norm_spread: float = 5e-3
    """
    The relative spread below which one block's lanes count as identical.

    Set from measurement rather than from caution. The rehearsal read 6.4e-4 at step 10, while
    the lanes were still separating, and settled in the 2e-2 to 4e-2 band once they had. The
    original 1e-3 sat below everything it would ever see and could only have caught a total
    failure; this sits an order of magnitude above the inert reading and four times below the
    working one.

    The name is kept from when this floor was applied to the standard deviation of the per-lane
    norms over their mean. It is now applied to the lane dispersion instead, and the calibration
    carries over unchanged because the two coincide exactly when the lanes are parallel and the
    dispersion is the larger of the two otherwise.
    """

    min_differentiated_fraction: float = 0.5
    """
    The fraction of blocks that must clear :data:`min_lane_norm_spread` for the mechanism to
    count as alive.

    This quantity is bimodal with nothing in the middle, which is what makes a threshold on it
    safe. At initialization every lane holds a bit-identical copy of the residual stream, so it
    reads exactly 0.0 in both runs measured so far. Once training starts it reads 0.875 to 1.0
    in the 370M probe and 1.0 from step 20 onward in the 8-block rehearsal. A half is the middle
    of that empty band: eight of sixteen blocks would have to go undifferentiated before the run
    is refused, and no run yet measured has had more than two.
    """

    _handles: Optional[list] = dataclasses.field(default=None, repr=False)
    _lane_spreads: Dict[str, float] = dataclasses.field(default_factory=dict, repr=False)
    _lane_dispersions: Dict[str, float] = dataclasses.field(default_factory=dict, repr=False)
    _in_training_step: bool = dataclasses.field(default=False, repr=False)

    def post_attach(self):
        if not self.enabled:
            return
        if not self._hyper_connection_blocks():
            raise OLMoConfigurationError(
                f"{type(self).__name__} was added to a run whose model has no hyper-connection "
                "blocks, so there is nothing for it to measure."
            )

    def _hyper_connection_blocks(self) -> List[Tuple[str, torch.nn.Module]]:
        model = getattr(self.trainer.train_module, "model", None)
        if model is None:
            return []
        blocks = []
        for name, module in model.named_modules():
            streams = [
                child for child in module.children() if isinstance(child, HyperConnectionStream)
            ]
            if streams:
                blocks.append((name, module))
        return blocks

    def _measuring(self) -> bool:
        return self.enabled and self.step % self.interval == 0

    def pre_step(self, batch):
        del batch
        self._in_training_step = True

    def post_train_batch(self):
        # Cleared before post_step, which is where the evaluators run. Everything below this
        # line in the step is a forward pass over held-out data, and the forward hook must not
        # read it.
        self._in_training_step = False

    def pre_train(self):
        if not self.enabled:
            return
        handles = []
        for name, block in self._hyper_connection_blocks():
            handles.append(
                block.register_forward_hook(ft.partial(self._activation_hook, block_name=name))
            )
        self._handles = handles

    @torch._dynamo.disable()
    @torch.no_grad()
    def _activation_hook(self, module, args, output, block_name: str):
        del module, args
        # `_in_training_step` and not `module.training`: the evaluator's own comment says it
        # means to put the model in eval mode and the line is commented out, so the module
        # flag stays True through an evaluation. Reading the eval forward pass instead of the
        # training one understated lane spread by 11-50% in the rehearsal, worst at the step
        # that lands in the run summary.
        if not self._in_training_step:
            return
        if not self._measuring() or not isinstance(output, torch.Tensor) or output.ndim != 4:
            return

        # (batch, seq, lanes, d_model) -> one norm per lane, averaged over tokens.
        lanes = output.detach().float()
        token_lane_norms = lanes.norm(dim=-1)
        lane_norms = token_lane_norms.mean(dim=(0, 1))
        mean_norm = lane_norms.mean()

        # A forward hook fires once per micro-batch, so every one of these names is recorded
        # several times against the same step -- eight times at the rehearsal's batch shape.
        # Without a merge strategy that is a warning per metric per micro-batch, and the
        # default keeps the *first* value and discards the rest. These are per-token means
        # already, so the last micro-batch of the step is as good an estimate as any and is a
        # quantity that can be named, which an iterated pairwise mean is not.
        label = _block_label(block_name)
        for lane, norm in enumerate(lane_norms):
            self.trainer.record_metric(
                f"hc/{label}/lane {lane} norm",
                norm,
                reduce_type=ReduceType.mean,
                merge_strategy=MetricMergeStrategy.latest,
            )
        self.trainer.record_metric(
            f"hc/{label}/hidden norm",
            mean_norm,
            reduce_type=ReduceType.mean,
            merge_strategy=MetricMergeStrategy.latest,
        )

        spread = (lane_norms.std(unbiased=False) / mean_norm.clamp_min(1e-12)).item()
        self.trainer.record_metric(
            f"hc/{label}/lane norm spread",
            spread,
            reduce_type=ReduceType.mean,
            merge_strategy=MetricMergeStrategy.latest,
        )
        self._lane_spreads[block_name] = spread

        # Lanes that differ only by a rotation have identical norms, so the spread above reads
        # zero on them and cannot tell "the lanes are one vector" from "the lanes are the same
        # length". This can: mean_k||x_k - x_bar||^2 = mean_k||x_k||^2 - ||x_bar||^2, so it is
        # zero if and only if every lane equals the lane mean, and per token it is never below
        # the spread, with equality exactly when the lanes are parallel.
        lane_mean_norm = lanes.mean(dim=-2).norm(dim=-1)
        about_mean = token_lane_norms.pow(2).mean(dim=-1) - lane_mean_norm.pow(2)
        dispersion = (
            (about_mean.clamp_min(0).sqrt() / lane_mean_norm.clamp_min(1e-12)).mean().item()
        )
        self.trainer.record_metric(
            f"hc/{label}/lane dispersion",
            dispersion,
            reduce_type=ReduceType.mean,
            merge_strategy=MetricMergeStrategy.latest,
        )
        self._lane_dispersions[block_name] = dispersion

    @torch.no_grad()
    def pre_optim_step(self):
        if not self._measuring():
            return

        composite: Optional[torch.Tensor] = None
        for name, block in self._hyper_connection_blocks():
            label = _block_label(name)
            for kind in ("attention", "feed_forward"):
                stream = getattr(block, f"{kind}_residual_stream", None)
                if not isinstance(stream, HyperConnectionStream):
                    continue
                matrix = get_full_tensor(stream.hc_static_alpha_r.detach()).float()
                if stream.doubly_stochastic:
                    matrix = sinkhorn_knopp(matrix, num_iters=stream.sinkhorn_iters)

                radius = torch.linalg.eigvals(matrix).abs().max()
                self.trainer.record_metric(f"hc/{label}/rho(A_r) {kind}", radius)

                composite = matrix if composite is None else composite @ matrix

        if composite is not None and composite.shape[-1] > 1:
            self.trainer.record_metric(
                "hc/composite condition number", torch.linalg.cond(composite)
            )
            self.trainer.record_metric(
                "hc/composite spectral radius", torch.linalg.eigvals(composite).abs().max()
            )

        self._check_lanes_differentiated()

    def _check_lanes_differentiated(self):
        if not self._lane_dispersions:
            return

        self.trainer.record_metric("hc/min lane norm spread", min(self._lane_spreads.values()))

        dispersions = self._lane_dispersions
        flat = sorted(n for n, v in dispersions.items() if v < self.min_lane_norm_spread)
        fraction = 1.0 - len(flat) / len(dispersions)
        self.trainer.record_metric("hc/min lane dispersion", min(dispersions.values()))
        self.trainer.record_metric("hc/median lane dispersion", median(dispersions.values()))
        self.trainer.record_metric("hc/differentiated block fraction", fraction)

        # A block that stays flat while its neighbours separate is a result about depth, not a
        # reason to stop, so it is said out loud on every measurement instead of only in the
        # exception that no longer fires for it.
        if flat:
            log.warning(
                "Lanes are undifferentiated in %d of %d blocks at step %d, dispersion under the "
                "%.3g floor: %s.",
                len(flat),
                len(dispersions),
                self.step,
                self.min_lane_norm_spread,
                ", ".join(f"{n} ({dispersions[n]:.3g})" for n in flat),
            )

        if self.fail_closed_by_step is None or self.step < self.fail_closed_by_step:
            return
        if fraction >= self.min_differentiated_fraction:
            log.info(
                "Lane differentiation check passed at step %d: %.0f%% of blocks are over the "
                "%.3g floor, against the %.0f%% required.",
                self.step,
                100 * fraction,
                self.min_lane_norm_spread,
                100 * self.min_differentiated_fraction,
            )
            self.fail_closed_by_step = None
            return

        raise RuntimeError(
            f"Hyper-connection lanes are still identical at step {self.step}: only "
            f"{100 * fraction:.0f}% of blocks carry lanes that differ by more than "
            f"{self.min_lane_norm_spread:.3g}, against the "
            f"{100 * self.min_differentiated_fraction:.0f}% required, with the median block at "
            f"{median(dispersions.values()):.3g}. The mechanism is inert across the model, so no "
            "downstream number from this configuration would be interpretable. Failing here "
            "rather than spending a full arm to find out."
        )

    def close(self):
        if self._handles is not None:
            for handle in self._handles:
                handle.remove()
            self._handles = None


def _block_label(block_name: str) -> str:
    # 'blocks.11' -> 'block 11', so the metrics sort next to the model's own per-block metrics.
    leaf = block_name.rsplit(".", 1)[-1]
    return f"block {int(leaf):02d}" if leaf.isdigit() else block_name
