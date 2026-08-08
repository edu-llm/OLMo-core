import dataclasses
import functools as ft
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from olmo_core.distributed.utils import get_full_tensor
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.residual_stream import HyperConnectionStream, sinkhorn_knopp

from ..common import ReduceType
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class HyperConnectionMonitorCallback(Callback):
    """
    Instrumentation for a run whose residual stream has been widened by
    :class:`~olmo_core.nn.residual_stream.HyperConnectionStream`.

    Four things are logged, each of which answers a question a final loss number cannot:

    - **Per-lane norm** and the spread across lanes. If the lanes never differentiate then the
      mechanism is inert, and neither a positive nor a negative downstream result says anything
      about hyper-connections. This is the primary guard, and :data:`fail_closed_by_step` turns
      it into an error rather than a plot nobody looks at.
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

    min_lane_norm_spread: float = 1e-3
    """
    The relative spread -- standard deviation of the per-lane norms over their mean -- below
    which the lanes count as identical and the mechanism as inert.
    """

    _handles: Optional[list] = dataclasses.field(default=None, repr=False)
    _lane_spreads: Dict[str, float] = dataclasses.field(default_factory=dict, repr=False)

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
        if not self._measuring() or not isinstance(output, torch.Tensor) or output.ndim != 4:
            return

        # (batch, seq, lanes, d_model) -> one norm per lane, averaged over tokens.
        lane_norms = output.detach().float().norm(dim=-1).mean(dim=(0, 1))
        mean_norm = lane_norms.mean()

        label = _block_label(block_name)
        for lane, norm in enumerate(lane_norms):
            self.trainer.record_metric(
                f"hc/{label}/lane {lane} norm", norm, reduce_type=ReduceType.mean
            )
        self.trainer.record_metric(
            f"hc/{label}/hidden norm", mean_norm, reduce_type=ReduceType.mean
        )

        spread = (lane_norms.std(unbiased=False) / mean_norm.clamp_min(1e-12)).item()
        self.trainer.record_metric(
            f"hc/{label}/lane norm spread", spread, reduce_type=ReduceType.mean
        )
        self._lane_spreads[block_name] = spread

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
        if not self._lane_spreads:
            return

        worst = min(self._lane_spreads.values())
        self.trainer.record_metric("hc/min lane norm spread", worst)

        if self.fail_closed_by_step is None or self.step < self.fail_closed_by_step:
            return
        if worst >= self.min_lane_norm_spread:
            log.info(
                "Lane differentiation check passed at step %d: smallest relative spread across "
                "lanes is %.3g, over the %.3g floor.",
                self.step,
                worst,
                self.min_lane_norm_spread,
            )
            self.fail_closed_by_step = None
            return

        inert = sorted(n for n, s in self._lane_spreads.items() if s < self.min_lane_norm_spread)
        raise RuntimeError(
            f"Hyper-connection lanes are still identical at step {self.step}: the smallest "
            f"relative spread across lanes is {worst:.3g}, under the {self.min_lane_norm_spread:.3g} "
            f"floor, in {len(inert)} block(s) starting with '{inert[0]}'. The mechanism is inert, "
            "so no downstream number from this configuration would be interpretable. Failing here "
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
