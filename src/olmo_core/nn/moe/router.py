import logging
import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core.config import DType, StrEnum
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from ..config import ModuleConfig
from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "MoERouter",
    "MoELinearRouter",
    "MoERouterConfig",
    "MoERouterType",
    "MoERouterGatingFunction",
]


log = logging.getLogger(__name__)


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign items/tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        del ctx
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment: Callable[
    [torch.Tensor, int], torch.Tensor
] = _UniformExpertAssignment.apply  # type: ignore


class MoERouterType(StrEnum):
    """
    An enumeration of the different MoE router implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoELinearRouter`
    """


class MoERouterGatingFunction(StrEnum):
    softmax = "softmax"
    sigmoid = "sigmoid"


@dataclass
class MoERouterConfig(ModuleConfig):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    name: MoERouterType = MoERouterType.default
    """
    The name of the implementation.
    """
    top_k: int = 1
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    dtype: Optional[DType] = None

    def num_params(self, d_model: int, num_experts: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0
        if self.name == MoERouterType.default:
            num_params += d_model * num_experts
        else:
            raise NotImplementedError

        return num_params

    def build(
        self,
        d_model: int,
        num_experts,
        *,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "MoERouter":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            num_experts=num_experts,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
        )
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == MoERouterType.default:
                return MoELinearRouter(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoERouter(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[dist.DeviceMesh] = None
        self.tp_mesh: Optional[dist.DeviceMesh] = None
        # Set by a capacity-based ParallelMLP so the router can report the drop rate it causes.
        # Left None by the dropless path, where no token is ever dropped and the metric would be
        # a constant zero pretending to be a measurement.
        self.expert_capacity: Optional[int] = None
        # How many micro-batches are folded into `batch_size_per_expert` right now. The
        # histogram accumulates until `reset_metrics`, while `expert_capacity` is per
        # micro-batch, so the drop rate needs both to be over the same span.
        self.num_accumulated_micro_batches: int = 0
        # EXACT drop accounting handed over by the capacity-based ParallelMLP (B3, L3's
        # `drop_accounting.py`), accumulated the same way the histogram above is: a running sum of
        # dropped and total assignments over the logging interval, so the reported rate is a real
        # interval rate rather than whichever micro-batch happened to be last.
        #
        # Two counters rather than a running mean of `drop_frac`, because micro-batches differ in
        # their total assignment count and a mean of per-micro-batch rates would weight them
        # equally. Left at None by the dropless path, where the metric is omitted rather than
        # reported as a zero indistinguishable from a measured zero.
        #
        # This closes a seam that the six-branch merge left open: `ParallelMLP` computed
        # `last_drop_accounting` every step and NOTHING IN THE TREE READ IT, so the exact
        # `drop_frac` the telemetry contract registers was computed and then discarded. Found by
        # the anti-vacuity guard in `MetricAssertionCallback`, which is exactly what it is for.
        self._drop_dropped_sum: Optional[torch.Tensor] = None
        self._drop_total_sum: Optional[torch.Tensor] = None

        if self.bias_gamma is not None:
            assert self.bias_gamma > 0
            self.register_buffer("score_bias", torch.zeros(self.num_experts, device=init_device))
        else:
            self.register_buffer("score_bias", None)

        # NOTE: we don't use buffers for t hese because we don't want FSDP to manage them, and we
        # don't use a BufferCache because `torch.compile()` doesn't handle that well when we're modifying
        # values in the cache.
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=init_device)
        )
        self._score_bias_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._load_balancing_loss: Optional[_HiddenTensor] = None
        self._z_loss: Optional[_HiddenTensor] = None
        # Sum of per-token gate mass -- the row-sum of the post-normalisation expert weights --
        # accumulated over the logging interval, so the metric reports a mean over the interval
        # rather than whatever the last micro-batch happened to be.  Hidden from torch for the
        # same reason as the histogram above: FSDP must not manage it, and a BufferCache does
        # not survive `torch.compile()` when the value is mutated in place.
        self._gate_mass_sum = hide_from_torch(torch.zeros([], device=init_device))
        # The divisor is a PLAIN PYTHON INT, not a tensor, and deliberately so.  It counts
        # tokens, which is a function of tensor SHAPES and therefore already known on the host
        # -- so keeping it here costs nothing, whereas reading a device-side counter to decide
        # whether the metric exists would force a host-device sync inside the training step.
        # The trainer runs the step under `set_sync_debug_mode("warn")`, so that sync would be
        # both a slowdown and a warning on every step.  `num_accumulated_micro_batches` above
        # is a plain int for the same reason.
        self._gate_mass_tokens: int = 0

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )
        # Re-created on the real device alongside the histogram. Built on `init_device` in
        # `__init__`, which is "meta" for an FSDP build, and a meta tensor accumulates nothing.
        self._gate_mass_sum = hide_from_torch(torch.zeros([], device=self.device))
        self._gate_mass_tokens = 0

        if self.bias_gamma is not None:
            assert self.score_bias is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()
            self._score_bias_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.lb_loss_weight is not None:
            self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.z_loss_weight is not None:
            self._z_loss = hide_from_torch(torch.zeros([], device=self.device))

    @property
    def device(self) -> torch.device:
        return get_default_device()

    @property
    def score_bias_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if self.bias_gamma is not None:
            if self._score_bias_batch_size_per_expert is None:
                self._score_bias_batch_size_per_expert = hide_from_torch(
                    torch.zeros(self.num_experts, device=self.device)
                )
            elif self._score_bias_batch_size_per_expert.device != self.device:
                self._score_bias_batch_size_per_expert = self._score_bias_batch_size_per_expert.to(
                    self.device
                )
        return (
            None
            if self._score_bias_batch_size_per_expert is None
            else unhide_from_torch(self._score_bias_batch_size_per_expert)
        )

    @score_bias_batch_size_per_expert.setter
    def score_bias_batch_size_per_expert(self, value: torch.Tensor):
        self._score_bias_batch_size_per_expert = hide_from_torch(value)

    @property
    def batch_size_per_expert(self) -> torch.Tensor:
        if self._batch_size_per_expert.device != self.device:
            self._batch_size_per_expert = self._batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._batch_size_per_expert)

    @batch_size_per_expert.setter
    def batch_size_per_expert(self, value: torch.Tensor):
        self._batch_size_per_expert = hide_from_torch(value)

    @torch.no_grad()
    def accumulate_drop_accounting(
        self, dropped_count: torch.Tensor, total_count: torch.Tensor
    ) -> None:
        """
        Fold one micro-batch's **exact** drop counts into the interval accumulator.

        Called by the capacity-based ``ParallelMLP`` right after it computes them, so the reported
        ``drop_frac`` is the exact ratio over the same span as every other metric here rather than
        the accumulated upper bound. Device tensors in, no host sync: the counts stay on device and
        are only divided at ``compute_metrics`` time.
        """
        if self._drop_dropped_sum is None or self._drop_dropped_sum.device != dropped_count.device:
            self._drop_dropped_sum = torch.zeros((), dtype=torch.float, device=dropped_count.device)
            self._drop_total_sum = torch.zeros((), dtype=torch.float, device=dropped_count.device)
        assert self._drop_total_sum is not None
        self._drop_dropped_sum += dropped_count.detach().float()
        self._drop_total_sum += total_count.detach().float()

    @property
    def gate_mass_mean(self) -> Optional[torch.Tensor]:
        """
        Mean gate mass -- the row-sum of the post-normalisation expert weights, averaged over
        every token routed since the last :meth:`reset_metrics()`.

        ``None`` before any token has been routed, so that a pre-training zero is not reported
        as a measured collapse. Under ``normalize_expert_weights=1.0`` this is 1.0 by
        construction; the sibling track measured **0.161** with the flag at its stock ``None``.
        """
        if self._gate_mass_tokens <= 0:
            return None
        if self._gate_mass_sum.device != self.device:
            self._gate_mass_sum = self._gate_mass_sum.to(self.device)
        return unhide_from_torch(self._gate_mass_sum) / self._gate_mass_tokens

    @property
    def load_balancing_loss(self) -> Optional[torch.Tensor]:
        if self.lb_loss_weight is not None:
            if self._load_balancing_loss is None:
                self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._load_balancing_loss.device != self.device:
                self._load_balancing_loss = self._load_balancing_loss.to(self.device)
        return (
            None
            if self._load_balancing_loss is None
            else unhide_from_torch(self._load_balancing_loss)
        )

    @load_balancing_loss.setter
    def load_balancing_loss(self, value: torch.Tensor):
        self._load_balancing_loss = hide_from_torch(value)

    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        if self.z_loss_weight is not None:
            if self._z_loss is None:
                self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._z_loss.device != self.device:
                self._z_loss = self._z_loss.to(self.device)
        return None if self._z_loss is None else unhide_from_torch(self._z_loss)

    @z_loss.setter
    def z_loss(self, value: torch.Tensor):
        self._z_loss = hide_from_torch(value)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if self.bias_gamma is None or not self.training:
            return

        assert self.score_bias is not None
        assert self.score_bias_batch_size_per_expert is not None
        score_bias = cast(torch.Tensor, self.score_bias)
        batch_size_per_expert = self.score_bias_batch_size_per_expert

        # Maybe reduce across the process group.
        if is_distributed():
            dist.all_reduce(batch_size_per_expert, group=self.group)

        ideal_batch_size_per_expert = batch_size_per_expert.mean(
            dim=0, keepdim=True, dtype=torch.float32
        )
        bias_delta = self.bias_gamma * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
        # NOTE: have to be careful here to manage the case where `score_bias` is a DTensor.
        bias_delta = distribute_like(score_bias, bias_delta)

        if not dry_run:
            get_local_tensor(score_bias).add_(get_local_tensor(bias_delta))

        # Reset the accumulator.
        batch_size_per_expert.zero_()

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.top_k, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    @abstractmethod
    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the un-normalized expert scores.

        :returns: The expert logits, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self.batch_size_per_expert
        out["load imbalance"] = (
            batch_size_per_expert.max() / batch_size_per_expert.mean(dtype=torch.float),
            ReduceType.max,
        )

        # ---------------------------------------------------------------------------------
        # B2 -- E-COMPARABLE BALANCE METRICS.  See `maple/agents/contracts/telemetry-schema.md`,
        # which registers these names.  They exist because `load imbalance` above CANNOT BE
        # COMPARED BETWEEN TWO CONFIGURATIONS WITH DIFFERENT EXPERT COUNTS, and the expert
        # count is the only axis of the rung ladder these metrics are built to compare -- so
        # using max/mean naively invalidates the central result of the whole series.
        #
        # Why max/mean is not comparable: it is the maximum of `num_experts` positively-skewed
        # counts, so its expectation rises with `num_experts` even when routing is perfectly
        # uniform.  A multinomial draw of 4,096 assignments gives 1.06 at E=8/k=2 and 1.19 at
        # E=32/k=8 -- a 12% "worsening" caused by nothing but counting more experts.  Reducing
        # it with ReduceType.max across data-parallel ranks skews it further, since each rank
        # contributes its own local maximum rather than a pooled count.  A difference in that
        # metric between two rungs is therefore not evidence of a difference in their routing.
        #
        # `load imbalance` is kept rather than replaced: it is what every prior run recorded,
        # and a metric that silently changes meaning between runs is worse than one that is
        # merely awkward to compare.  It must not be used for a cross-rung comparison.
        #
        # EVERY KEY BELOW IS TAGGED ReduceType.max, AND NOT BECAUSE A MAXIMUM IS THE STATISTIC
        # WE WANT.  `MoETransformer.compute_auxiliary_metrics` emits each key twice: once as
        # `block NN/<key>`, which is the series to read, and once folded across every MoE block
        # under the bare `<key>`.  For `mean` and `sum` it folds by ADDING -- which is right for
        # the auxiliary losses, the only things stock ever tagged that way, and wrong for
        # anything bounded.  Tagged `mean`, a normalised entropy of ~0.998 in each of 16 blocks
        # was reported as ~15.97: a quantity defined on [0, 1] left the trainer above 15.  That
        # is not hypothetical, it is what the sibling track logged.  `max` folds with
        # `torch.max`, so the bare key reads as the WORST block -- lowest-entropy, highest-CV,
        # most-dead -- which is the right summary for a health gate.
        #
        # The bare folded key is still under-labelled: `expert_load_cv` with no block prefix
        # means "worst block", which the name does not say.  Explicitly-named cross-block
        # aggregates (`moe/expert_load_cv_max`, `moe/expert_load_cv_mean`, ...) are computed by
        # `MetricAssertionCallback` from the `block NN/` series, where a mean is a real mean
        # rather than a sum.  Read those, or read the per-block series.  Never read a bare
        # cross-block key and assume it is an average.
        counts = batch_size_per_expert.to(torch.float)
        num_experts = counts.numel()
        tiny = torch.finfo(torch.float32).tiny
        mean = counts.mean()
        total = counts.sum()

        # Coefficient of variation: population std over mean, and the scale-free quantity the
        # load-balancing loss is a smooth surrogate for. Zero under perfect balance at every E.
        cv = counts.std(unbiased=False) / mean.clamp_min(tiny)
        out["expert_load_cv"] = (cv, ReduceType.max)

        # CV EXCESS OVER THE UNIFORM-ROUTING NULL, AND THIS IS THE ONE TO COMPARE ACROSS RUNGS.
        #
        # `expert_load_cv` is scale-free in the mean LOAD but NOT in the SAMPLE SIZE, and the
        # ladder varies the sample size by 4x -- so comparing raw CV across rungs is the same
        # class of error as comparing max/mean, merely smaller and much less obvious. Measured
        # on FarmShare and confirmed in closed form:
        #
        #   counts ~ Multinomial(n, 1/E) under perfect uniform routing, so
        #   sd = sqrt(n/E * (1 - 1/E)), mean = n/E, and therefore
        #       CV_null = sqrt((1 - 1/E) / mean)
        #
        # Total assignments are tokens*k and k=8 at every rung, so assignments/expert FALLS as E
        # rises: 2048 on the sibling probe, then 1024 (R1), 512 (R2), 256 (R3). The null CV
        # therefore rises 0.0207 -> 0.0310 -> 0.0440 -> 0.0624 across the ladder -- it TRIPLES
        # from the sibling to R3 under identical, perfect routing. A raw-CV comparison would read
        # that as balance degrading with granularity, which is exactly the false conclusion the
        # E-comparability requirement exists to prevent.
        #
        # The ratio is 1.0 under perfect balance at every E and every sample size, which is what
        # "comparable across rungs" has to mean. Above 1.0 is real imbalance in units of "times
        # worse than chance"; a value below 1.0 is a router more uniform than a fair multinomial
        # draw, which is what a load-balancing loss actually produces and is not an error.
        #
        # `mean` here is the ACCUMULATED mean over the logging interval, which is correct without
        # dividing by the micro-batch count: the null formula takes whatever sample the observed
        # CV was computed from, and both come from the same accumulated histogram.
        cv_null = ((1.0 - 1.0 / num_experts) / mean.clamp_min(tiny)).sqrt()
        out["expert_load_cv_excess"] = (cv / cv_null.clamp_min(tiny), ReduceType.max)

        # Normalised routing entropy deficit, 1 - H(p)/log(E) over the realised assignment
        # distribution. 0.0 is perfect balance and 1.0 is collapse onto a single expert, at
        # every E, which makes this the primary cross-rung readout. Stated as a deficit rather
        # than as entropy so that folding with `max` keeps the WORST block rather than the best.
        #
        # This one carries the same finite-sample bias as the CV above -- a fair multinomial draw
        # has entropy strictly below log(E), so the null deficit is (E-1)/(2*n*ln E) rather than
        # zero -- but here the bias is negligible rather than merely small: it runs 1.0e-4 at the
        # sibling's E=8/2048 to 3.5e-4 at R3, i.e. THREE ORDERS OF MAGNITUDE below the [0, 0.06]
        # band of interest. So no null correction is emitted for it, and the sibling's measured
        # worst-block 0.0663 stands as ~645x its own null: overwhelmingly real imbalance, not a
        # sampling artefact. Recorded here so the asymmetry with the CV is a documented decision
        # rather than an oversight.
        probs = counts / total.clamp_min(tiny)
        entropy = -(probs * probs.clamp_min(tiny).log()).sum()
        normalized_entropy = entropy / math.log(num_experts) if num_experts > 1 else entropy
        out["entropy_deficit"] = (1.0 - normalized_entropy, ReduceType.max)

        # Dead experts as a fraction, because the stop criteria name dead experts and neither
        # metric above separates "one expert idle" from "all slightly uneven". A fraction
        # rather than a count so that it, too, is comparable across E. Guarded against an
        # all-zero histogram, which happens before any tokens have been routed: the unguarded
        # form returns 1.0 there, and 1.0 is indistinguishable from total expert collapse.
        out["dead_expert_frac"] = (
            (
                (counts == 0).sum(dtype=torch.float) / num_experts
                if total > 0
                else torch.zeros((), dtype=torch.float, device=counts.device)
            ),
            ReduceType.max,
        )

        # Assignments per expert per micro-batch -- the DENOMINATOR every band in the telemetry
        # contract is quoted against, logged so those bands are auditable in-run rather than
        # taken on faith from a planning document.  The ladder's whole balance risk is stated in
        # these units: 2048 assignments/expert on the sibling probe, 1024 at R1, 512 at R2, 256
        # at R3, against a capacity of 312 at factor 1.2 (3.50 sigma) or ~512 at the funded
        # factor 2.0.  If this number is not what the rung table predicts then the batch, the
        # micro-batch split or the expert count is not what was configured, and every band below
        # is being applied to the wrong regime.
        #
        # Divided by the accumulated micro-batch count because `batch_size_per_expert` sums over
        # every micro-batch since the last `reset_metrics`, while the capacity it is compared
        # against is per-micro-batch.  Reporting the raw accumulated mean is how a balanced
        # router comes to look like it is dropping 44% of its assignments.
        if self.num_accumulated_micro_batches > 0:
            out["assignments_per_expert_mean"] = (
                mean / self.num_accumulated_micro_batches,
                ReduceType.max,
            )

        # GATE MASS -- THE GUARD ON `normalize_expert_weights`, WHICH IS MEASURED-BROKEN BY
        # DEFAULT.  `normalize_expert_weights=None` is the stock default and zero of five
        # shipped recipes set it; the sibling track measured the resulting gate mass at 0.161
        # against an intended 1.000, a 6.2x error that TRAINS HAPPILY and shows up only as a
        # quietly worse loss.  This is Maple's `norm_topk_prob`, so it is an architectural
        # requirement here, not a tuning knob.
        #
        # Accumulated in `forward` after the normalisation block, so this measures the weights
        # the experts were actually scaled by rather than the config value that was supposed to
        # produce them.  A config assertion cannot catch a normalisation that runs and produces
        # the wrong norm; this can.
        if (gate_mass := self.gate_mass_mean) is not None:
            out["gate_mass_mean"] = (gate_mass, ReduceType.max)

        # DROPPED TOKENS, WHICH NOTHING COMPUTED BEFORE.
        # A capacity-based MoE pads every expert to `expert_capacity` slots and silently
        # discards assignments beyond it -- `binned_gather` keeps the first `bin_size` of each
        # bin and the rest never reach an expert. The discard leaves no trace: the loss is
        # slightly worse and nothing says why. That matters more than usual here, because a
        # drop rate is not a constant of the model: it depends on how uneven the router is,
        # so two configurations with different expert counts can drop at different rates and
        # the difference is invisible while looking exactly like a quality difference.
        #
        # `expert_capacity` is set by the capacity-based ParallelMLP each step. The dropless
        # path leaves it None and the metric is omitted rather than reported as zero, because a
        # constant zero and a measured zero should not look the same in a dashboard.
        #
        # Note the sort in `indices_and_bins` is by expert id only, so the assignments that
        # survive are the earliest by position, not the highest-weighted -- overflow drops the
        # end of the sequence. That is a separate issue from measuring it.
        # THE EXACT RATE, which is the one the telemetry contract registers as `drop_frac` and the
        # one the 1% ceiling asserts against. Emitted whenever the capacity path handed over real
        # counts; the upper bound below is emitted alongside it so a reader can see how loose the
        # accumulated estimate is against the truth.
        if self._drop_total_sum is not None and self._drop_dropped_sum is not None:
            out["drop_frac"] = (
                self._drop_dropped_sum / self._drop_total_sum.clamp_min(1.0),
                ReduceType.max,
            )

        if self.expert_capacity is not None and self.num_accumulated_micro_batches > 0:
            # Capacity over the same span as the counts. See the note beside
            # `num_accumulated_micro_batches`: comparing an interval's worth of assignments
            # against one forward's capacity reports ~44% drops for a router measured, by the
            # entropy beside it, as within 1% of uniform.
            #
            # This is an upper bound rather than the exact rate, because a per-micro-batch
            # overflow that the accumulated histogram averages away is not recoverable from a
            # summed count. It is tight when the router is stable across an interval, which is
            # the regime the gate cares about, and it cannot under-report.
            span_capacity = self.expert_capacity * self.num_accumulated_micro_batches
            overflow = (counts - span_capacity).clamp_min(0).sum()
            # RENAMED from "dropped token fraction (upper bound)" to match the `drop_frac`
            # family that `telemetry-schema.md` registers, and to lose the spaces and
            # parentheses that make a metric name awkward to glob and awkward to assert on.
            # The "_upper_bound" suffix is load-bearing and stays: this is computed from the
            # ACCUMULATED histogram against a per-micro-batch capacity summed over the same
            # span, so a per-micro-batch overflow that the accumulation averages away is not
            # recoverable from it.  It cannot under-report, and it is tight when the router is
            # stable across an interval, which is the regime the gate cares about.
            #
            # L3 owns the true per-micro-batch `drop_frac` if it produces one.  Both names are
            # registered; a run that emits only this one is asserted against this one.  An
            # assertion pointed at a metric nobody emits is a green run that verified nothing.
            out["drop_frac_upper_bound"] = (
                overflow / total.clamp_min(1.0),
                ReduceType.max,
            )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            assert self.load_balancing_loss is not None
            out["load balancing loss"] = (
                self.lb_loss_weight * self.load_balancing_loss,
                ReduceType.mean,
            )
            out["load balancing loss unscaled"] = (
                self.load_balancing_loss.clone(),
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            assert self.z_loss is not None
            out["router Z loss"] = (self.z_loss_weight * self.z_loss, ReduceType.mean)
            out["router Z loss unscaled"] = (self.z_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        if (bz_per_expert := self.batch_size_per_expert) is not None:
            bz_per_expert.zero_()
        # Cleared with the histogram it counts, or the capacity span would keep growing while
        # the counts restarted from zero and the drop rate would fall to zero and stay there.
        self.num_accumulated_micro_batches = 0
        # Same reasoning for gate mass: the sum and its token count must be cleared together
        # or the reported mean decays toward zero across a run while the true mass is 1.0.
        unhide_from_torch(self._gate_mass_sum).zero_()
        self._gate_mass_tokens = 0
        # And the exact drop counters, for the same reason: numerator and denominator must be
        # cleared together or the rate drifts toward the run's lifetime average.
        if self._drop_dropped_sum is not None:
            self._drop_dropped_sum.zero_()
        if self._drop_total_sum is not None:
            self._drop_total_sum.zero_()
        if (lb_loss := self.load_balancing_loss) is not None:
            lb_loss.zero_()
        if (z_loss := self.z_loss) is not None:
            z_loss.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        # GATE MASS, MEASURED HERE AND NOWHERE ELSE.  This is the one point in the model where
        # the weights the experts will actually be scaled by are visible: after `get_top_k` and
        # after the normalisation above.  Accumulated unconditionally rather than only when
        # `normalize_expert_weights` is set, because the case worth catching is precisely the
        # one where the flag is UNSET -- stock default `None`, zero of five shipped recipes set
        # it, measured mass 0.161 against an intended 1.000.  A metric that only exists when the
        # knob is configured cannot detect the knob being missing.
        #
        # Summed over all but the last dimension so the divisor is a token count: `top_k`
        # weights per token sum to the mass of one token's gate.  No `.item()` and no
        # comparison against a device tensor anywhere in here -- the whole accumulator is one
        # fused add, and the token count is taken from `.shape`, which is host-side already.
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                if self._gate_mass_sum.device != expert_weights.device:
                    self._gate_mass_sum = self._gate_mass_sum.to(expert_weights.device)
                # `.add_()` rather than `+=`, deliberately. `unhide_from_torch` returns the
                # WRAPPED tensor itself, so `+=` on the returned local would in fact mutate the
                # stored tensor -- but only because `Tensor.__iadd__` happens to be in-place.
                # That is too subtle to rely on in a metric whose whole job is to catch a silent
                # error: if it ever rebound instead of mutating, this accumulator would read a
                # permanent zero and the gate-mass assertion would pass vacuously forever.
                # `.add_()` cannot be misread.
                unhide_from_torch(self._gate_mass_sum).add_(expert_weights.detach().float().sum())
                self._gate_mass_tokens += expert_weights.numel() // expert_weights.shape[-1]

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        batch_size_per_expert=batch_size_per_expert,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.load_balancing_loss += lb_loss.detach()

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

            self.batch_size_per_expert += batch_size_per_expert
            # Count the micro-batches folded into that accumulator. `expert_capacity` is a
            # per-micro-batch bound, so a drop rate computed from the accumulated histogram
            # has to compare against the capacity summed over the same micro-batches.
            # Without this the numerator spans a whole logging interval and the denominator
            # spans one forward, and a perfectly balanced router reports a large drop rate.
            self.num_accumulated_micro_batches += 1
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )
        self.tp_mesh = tp_mesh

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh


class MoELinearRouter(MoERouter):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(init_device=init_device, **kwargs)
        # NOTE: this parameter needs to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP. So we flatten it to a single dimension tensor.
        # And for that reason we don't support a 'bias' option.
        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x.float(), get_local_tensor(self.weight).view(self.num_experts, self.d_model).float()
        )

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        super().apply_tp(tp_mesh, float8_enabled=float8_enabled)
        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )
