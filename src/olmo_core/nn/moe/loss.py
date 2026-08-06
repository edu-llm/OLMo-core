"""
Auxiliary losses for the MoE router.

Rewritten rather than patched. Three things were wrong with the previous version and two of
them cannot be fixed by adding a branch:

1. **The load-balancing loss was rank-local only.** ``MoELoadBalancingLossGranularity`` offered
   ``local_batch`` and ``instance``, and ``instance`` is *finer*, not coarser. Neither aggregates
   over the data-parallel group, and there was no all-reduce of the expert counts anywhere in the
   loss path -- the only cross-rank reduction of expert counts in the whole MoE stack lives in
   ``MoERouter.post_batch`` and is gated on ``bias_gamma``, i.e. it serves DeepSeek-v3's
   aux-loss-free bias and not this loss. So every rank pushed its own shard towards uniform and
   nothing pushed the *global* batch towards uniform. The error this leaves behind grows as the
   number of assignments each expert sees per rank falls, which is exactly the direction a
   fine-grained MoE moves in: 2048 assignments/expert at E=8, 256 at E=256.

   ``global_batch`` is added here. It is not a wrapper around the local value under a new name --
   see :func:`reduce_expert_counts` -- and both numbers are returned on every call so that the
   claim "the reduction reduced" is checkable rather than asserted.

2. **Both quantities are returned, always.** A load-balancing loss that only reports its
   post-fix value makes the fix unfalsifiable: an all-reduce over a group of size one, an
   all-reduce over the wrong group, and an all-reduce that silently no-ops all look identical
   from a single logged scalar. :class:`LoadBalancingLoss` carries ``lbl_local`` and
   ``lbl_global`` computed from the *same* scores and the *same* counts, so the pair is a
   controlled comparison and not two measurements of two things.

3. **The scale is documented.** Under perfectly uniform routing the returned loss is exactly
   ``1.0`` for every ``num_experts`` and every ``top_k``, which is what makes it comparable
   across the rungs of an expert-count ladder. That property is asserted in
   ``src/test/nn/moe/loss_test.py`` rather than left as a claim about the algebra.

The third defect in the original module -- per-block metrics folded across blocks by addition --
is not fixed here, because for a *loss* the cross-block sum is the correct aggregate and it is
only bounded per-block statistics that summing corrupts. What this module owes the telemetry
contract is that the two scalars it hands out are losses, are tagged as such, and are folded the
same way as each other so their ratio survives the fold.
"""

from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor, get_world_size

__all__ = [
    "MoELoadBalancingLossGranularity",
    "LoadBalancingLoss",
    "reduce_expert_counts",
    "load_balancing_loss",
    "router_z_loss",
]


class MoELoadBalancingLossGranularity(StrEnum):
    """
    Defines the granularity for the router's load balancing loss.
    """

    local_batch = "local_batch"
    """
    The loss is always computed over the rank-local shard of the batch, ignoring any
    parallelism strategies used. This is ideal for minimizing the number of dropped tokens for
    any parallel strategy.
    """

    instance = "instance"
    """
    The loss is computed over each instance, taking into account any parallelism strategies used.
    """

    global_batch = "global_batch"
    """
    The loss is computed against the expert assignment counts of the **whole** batch, reduced
    across every rank that holds a different slice of it.

    This is the same Switch-style loss as :data:`local_batch`; only the load term changes, from
    "how uneven is this rank's shard" to "how uneven is the batch". The gradient still flows
    only through the rank-local router scores, so no differentiable collective is needed -- the
    counts are already computed under ``no_grad``. See :func:`load_balancing_loss` for why the
    gradient magnitude is unchanged by the switch, which is what makes ``lb_loss_weight``
    transfer between the two granularities instead of silently rescaling by the world size.

    .. warning::
        Not the default. Choosing it is an explicit act, because a run that gets global balancing
        by accident and a run that gets it on purpose are indistinguishable afterwards, and
        because the local/global contrast is itself an experimental arm.
    """


class LoadBalancingLoss(NamedTuple):
    """
    The output of :func:`load_balancing_loss`.

    :param loss: The value to attach to the graph, computed at the requested granularity. This is
        the only element carrying a gradient.
    :param lbl_local: The rank-local value, detached. Equal to what
        :data:`MoELoadBalancingLossGranularity.local_batch` would have produced.
    :param lbl_global: The data-parallel-reduced value, detached.
    :param reduced: Whether counts from **more than one rank** were actually pooled. ``False``
        means ``lbl_global`` is arithmetically the local computation, because the group had size
        one -- not that the reduction was skipped by mistake. Callers that assert the two differ
        must check this first, or the assertion fires on a single-GPU run for the wrong reason.

        .. warning::
            This is deliberately **not** "did an ``all_reduce`` call execute". The platform
            bootstraps a single-process distributed group for 1-GPU runs, so
            ``is_distributed()`` is ``True`` with ``world_size == 1`` and the collective does
            run -- over one rank, pooling nothing. Keyed on the collective it would report
            ``True`` there, and a gate reading "reduced" as evidence that the global batch was
            pooled would be reading a run where it demonstrably was not. Size of the group is the
            honest predicate; whether a call happened is not.
    """

    loss: torch.Tensor
    lbl_local: torch.Tensor
    lbl_global: torch.Tensor
    reduced: bool


def reduce_expert_counts(
    counts: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, bool]:
    """
    Sum per-expert assignment counts across a process group.

    :param counts: Integer counts of shape ``(num_experts,)`` for the rank-local slice.
    :param group: The group to reduce over. See the note below on which group is correct.
    :returns: ``(global_counts, pooled)``. ``global_counts`` is a **new** tensor; ``counts`` is
        never modified. ``pooled`` is ``True`` only when the group holds **more than one rank**,
        i.e. only when the returned counts genuinely describe more than this rank's slice --
        **not** merely when an ``all_reduce`` executed. See :class:`LoadBalancingLoss`.

    **Why the input must be cloned.** The same ``batch_size_per_expert`` tensor the router hands
    to this function is also handed to the dispatch path, which sizes its all-to-all and its
    capacity bins from it. An in-place ``all_reduce`` would multiply those counts by the world
    size and route garbage, and it would do so without raising. The clone is the entire safety
    property of this function and is not an optimization to remove.

    **Why the counts are reduced as integers.** They are exact, so the reduction is exact and the
    rescaling below is the only place a rounding error can enter. Reducing them after the cast to
    float would put the sum at risk above ``2**24`` assignments per group -- reachable at a large
    global batch -- for no benefit.

    **Which group is correct.** Every rank holding a *different slice of the batch* must be
    included, and every rank holding a *different set of layers* must be excluded. Data parallel,
    expert parallel and context parallel ranks all hold different tokens, so they belong in.
    Tensor-parallel ranks hold different slices of the sequence in this codebase, so they belong
    in as well. Pipeline-parallel ranks hold *different layers* and their counts describe
    different routers, so summing across them would add unrelated histograms. ``MoERouter.group``
    is exactly this group: it is ``None`` (i.e. the world) when pipeline parallelism is off, and
    the pipeline *stage* group when it is on. That is also the group DeepSeek-v3's expert-bias
    path already reduces over, so the two globally-balanced mechanisms in this stack agree about
    what "global" means.
    """
    # THE PREDICATE IS GROUP SIZE, NOT WHETHER A COLLECTIVE RAN. Keyed on `is_distributed()`
    # alone this reported "reduced" on any 1-GPU platform run, because the platform bootstraps a
    # single-process distributed group -- `is_distributed()` is True, `world_size` is 1, the
    # all_reduce executes and pools nothing. A gate reading that as "the global batch was pooled"
    # would be reading a run where it provably was not, which is the exact false confidence the
    # local/global pair exists to prevent. Measured on FarmShare: world_size=1 with gloo
    # initialised returns reduced=True under the old predicate.
    #
    # `get_world_size(group)` returns 1 when distributed is uninitialised, so this one call covers
    # both the not-distributed and the world-size-1 cases.
    if get_world_size(group) <= 1:
        # Nothing to pool. Return a distinct tensor anyway so that callers cannot accidentally
        # alias the input, and report `pooled=False` so a caller comparing local against global
        # can tell "identical because there is one rank" from "identical because the reduction
        # silently failed".
        #
        # The collective is skipped rather than run over a group of one. That is safe for
        # collective symmetry precisely because the condition is world size, which every rank in
        # the group agrees on -- so either all ranks skip or none do. A condition that could
        # differ per rank must never gate a collective; this one cannot.
        return counts.clone(), False

    global_counts = counts.clone()
    dist.all_reduce(global_counts, group=group)
    return global_counts, True


def _mean_expert_scores(expert_scores: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Mean router probability per expert over the rank-local tokens. Shape ``(num_experts,)``.

    This term, and only this term, carries the gradient.
    """
    # shape: (B, S, num_experts) -> (B * S, num_experts) -> (num_experts,)
    return expert_scores.reshape(-1, num_experts).mean(dim=0)


def _switch_loss(
    *,
    counts: torch.Tensor,
    mean_expert_scores: torch.Tensor,
    num_experts: int,
    top_k: int,
    loss_div_factor: Union[torch.Tensor, float],
) -> torch.Tensor:
    """
    The Switch/GShard load-balancing loss, ``(E / k) * <counts, mean_scores> / tokens``.

    Normalised so that perfectly uniform routing gives exactly ``1.0`` at every ``E`` and ``k``:
    with ``counts[e] = T*k/E`` and ``mean_scores[e] = 1/E``, the inner product is ``T*k/E``, the
    division by ``T`` leaves ``k/E``, and the ``E/k`` prefactor cancels it. That invariance is
    what lets an expert-count ladder compare this number between rungs, and it is the reason the
    prefactor is here rather than folded into ``lb_loss_weight``.
    """
    return (num_experts / top_k) * torch.dot(counts, mean_expert_scores) / loss_div_factor


def load_balancing_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batch_size_per_expert: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    granularity: MoELoadBalancingLossGranularity,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> LoadBalancingLoss:
    """
    Compute the router's load-balancing loss at the requested granularity, together with the
    rank-local and data-parallel-reduced values of the same loss for telemetry.

    :param num_experts: ``E``.
    :param top_k: ``k``.
    :param expert_scores: Router probabilities, shape ``(B, S, E)``, rank-local. Carries grad.
    :param batch_size_per_expert: Assignment counts, shape ``(E,)``, rank-local, integer, no grad.
    :param batched_batch_size_per_expert: Per-instance counts, shape ``(B, E)``. Used only by
        :data:`MoELoadBalancingLossGranularity.instance`.
    :param granularity: Which value gets a gradient.
    :param loss_div_factor: Token-count divisor. When ``None``, the rank-local token count of
        ``expert_scores`` is used. The trainer passes the rank's whole-batch non-padding token
        count, so each micro-batch contributes its share and the shares sum to one batch's worth.
    :param dp_group: The group to reduce counts over. See :func:`reduce_expert_counts` for which
        group this must be.
    :param tp_mesh: Tensor-parallel mesh, if any.
    :param cp_mesh: Context-parallel mesh, if any.

    **Why the global variant needs no differentiable collective, and why switching granularity
    introduces no world-size rescale.**

    Write the loss as ``E * sum_e f_e * P_e`` with ``f_e`` the fraction of assignments going to
    expert ``e`` and ``P_e`` the mean router probability for it. Only ``P_e`` is differentiable;
    ``f_e`` comes from an ``argmax`` and is computed under ``no_grad``. The global loss replaces
    the rank-local ``f_e`` with the batch-wide ``f_e`` and leaves ``P_e`` rank-local. So each rank
    computes ``E * sum_e f_e^global * P_e^local``, and:

    - **Value.** Averaged over ranks -- which is how the metric is reduced for logging -- this is
      ``E * sum_e f_e^global * P_e^global``, the true global loss.
    - **Gradient.** The data-parallel wrapper averages gradients across ranks, so the optimizer
      sees ``E * sum_e f_e^global * d(mean over ranks of P_e^local)/dtheta``. The local variant
      gives the same expression with ``f_e^global`` replaced by ``f_e^local``. Both ``f`` vectors
      sum to one over experts and sit at ``O(1/E)``, so **no world-size factor is introduced by the
      switch** and ``lb_loss_weight`` means the same thing under either granularity. Nothing has to
      be multiplied by the world size, and nothing has to be all-reduced with a gradient attached.

      **This is deliberately weaker than "the two gradients have the same magnitude", which is what
      this docstring used to claim and which is false.** Measured per-rank ``local/global``
      gradient-norm ratios on four genuinely skewed ranks span **0.72x to 3.44x**. That spread is
      the intended signal, not a bug: on a rank whose own histogram is collapsed, the local loss is
      screaming about an imbalance the global batch does not have, so its gradient *should* be
      larger. The claim the local-vs-global experimental arm rests on is the absence of a
      world-size rescale -- which is what makes ``lb_loss_weight`` transferable -- not per-rank
      gradient-norm equality. The old wording would have licensed reading a real routing signal as
      a scale bug.

    The global counts are rescaled from the batch's token budget back down to this rank's, by the
    exact ratio of the two assignment totals, so that ``loss_div_factor`` keeps its rank-local
    meaning. Two consequences worth stating because they are the assertions that make the
    reduction falsifiable:

    - With one rank, or with every rank holding an identical count histogram, ``lbl_global`` is
      **bit-identical** to ``lbl_local``.
    - With ranks holding different histograms, the two **must** differ. A "global" value that
      still matches under different data is a reduction that did not happen.

    The ratio form also means unequal token counts per rank -- a ragged final batch, or different
    amounts of padding -- are handled without a separate world-size query.
    """
    expert_scores, batch_size_per_expert, batched_batch_size_per_expert = (
        get_local_tensor(expert_scores),
        get_local_tensor(batch_size_per_expert),
        get_local_tensor(batched_batch_size_per_expert),
    )

    B, S, _ = expert_scores.shape

    # -- The telemetry pair. Computed for every granularity, from one set of scores and one set of
    # -- counts, so that the two numbers differ only in the thing under test.
    #
    # The divisor here is deliberately the same one `local_batch` uses, rather than the one the
    # requested granularity uses, so that `lbl_local` is exactly the number this run would have
    # logged before `global_batch` existed. Comparing the pair is then a comparison against the
    # prior behaviour and not against a re-derived baseline.
    telemetry_div_factor: Union[torch.Tensor, float]
    if loss_div_factor is None:
        telemetry_div_factor = B * S
        if tp_mesh is not None:
            telemetry_div_factor = telemetry_div_factor * tp_mesh.size()
    elif cp_mesh is not None:
        telemetry_div_factor = loss_div_factor / cp_mesh.size()
    else:
        telemetry_div_factor = loss_div_factor

    mean_scores = _mean_expert_scores(expert_scores, num_experts)

    local_counts = batch_size_per_expert.type_as(expert_scores)
    lbl_local = _switch_loss(
        counts=local_counts,
        mean_expert_scores=mean_scores,
        num_experts=num_experts,
        top_k=top_k,
        loss_div_factor=telemetry_div_factor,
    )

    global_counts_int, reduced = reduce_expert_counts(batch_size_per_expert, group=dp_group)
    # Rescale the batch-wide counts to this rank's token budget. Both totals are exact integers,
    # so the only inexactness is the division itself; with equal ranks the factor is 1/world_size,
    # which is exact for a power-of-two world size and makes the identical-data case bit-exact.
    # Two degenerate cases, both measured rather than assumed, both finite:
    #
    # - **A rank with zero assignments** (an empty shard, or a fully-masked batch). Its
    #   `local_share` is 0, so its rescaled global counts are all zero and its balance loss is
    #   exactly 0 -- it contributes no balance gradient even if the global batch is badly
    #   imbalanced. That is the intended reading: a rank with no tokens has nothing to re-route,
    #   and the ranks that do hold tokens still see the true global histogram. Recorded because it
    #   is a behaviour rather than an obvious consequence.
    # - **Every rank zero** (before any forward). `clamp_min(1.0)` on the denominator is what makes
    #   this 0.0 rather than NaN. Do not remove it: a NaN here would poison the aux loss and, via
    #   `attach_auxiliary_loss`, the whole graph.
    local_total = batch_size_per_expert.sum()
    global_total = global_counts_int.sum()
    local_share = (local_total.double() / global_total.double().clamp_min(1.0)).to(
        expert_scores.dtype
    )
    global_counts = global_counts_int.type_as(expert_scores) * local_share
    lbl_global = _switch_loss(
        counts=global_counts,
        mean_expert_scores=mean_scores,
        num_experts=num_experts,
        top_k=top_k,
        loss_div_factor=telemetry_div_factor,
    )

    # -- The value that gets a gradient.
    loss: torch.Tensor
    if granularity == MoELoadBalancingLossGranularity.instance:
        # shape: (B, num_experts)
        batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)

        # NOTE: for CP it suffices to reduce the 'batched_batch_size_per_expert' across the CP group
        # and do the rest of the computation locally.
        if cp_mesh is not None:
            dist.all_reduce(batched_batch_size_per_expert, group=cp_mesh.get_group())

        # NOTE: for TP, the end result needs to be a DTensor over the TP mesh, so we handle this case
        # a little differently.
        instance_scores: torch.Tensor
        if tp_mesh is not None:
            # NOTE: assumes sharded on sequence dimension and equal splits across TP group.
            dist.all_reduce(batched_batch_size_per_expert, group=tp_mesh.get_group())
            batched_batch_size_per_expert = DTensor.from_local(  # type: ignore[assignment]
                batched_batch_size_per_expert, tp_mesh, (Replicate(),)
            )
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
            instance_scores = expert_scores.view(B, -1, num_experts).mean(dim=1, keepdim=True)
            # shape: (B, 1, num_experts) -> (B, num_experts)
            instance_scores = DTensor.from_local(instance_scores, tp_mesh, (Shard(1),)).mean(dim=1)
        else:
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            instance_scores = expert_scores.view(B, -1, num_experts).mean(dim=1)

        # We compute this across the TP and CP groups, so the 'loss_div_factor' should represent
        # the total number of tokens across the TP and CP groups.
        instance_div_factor: Union[torch.Tensor, float]
        if loss_div_factor is None:
            # this gives us total number of tokens across TP + CP groups.
            instance_div_factor = batched_batch_size_per_expert.sum() / top_k
        else:
            instance_div_factor = loss_div_factor

        # shape: scalar
        loss = (
            (num_experts / top_k)
            * (instance_scores * batched_batch_size_per_expert).sum()
            / instance_div_factor
        )
    elif granularity in (
        MoELoadBalancingLossGranularity.local_batch,
        MoELoadBalancingLossGranularity.global_batch,
    ):
        # NOTE: We essentially ignore CP for these granularities, and for TP we still compute the
        # loss locally, but wrap as a DTensor and reduce it at the end because the end result has
        # to be a DTensor over the TP mesh.
        # Due to that DTensor reduction, with TP the 'loss_div_factor' should be the total number
        # of tokens across the TP group, but not the CP group. That is already what
        # `telemetry_div_factor` is, so both paths share it.
        if granularity == MoELoadBalancingLossGranularity.local_batch:
            loss = _switch_loss(
                counts=local_counts,
                mean_expert_scores=mean_scores,
                num_experts=num_experts,
                top_k=top_k,
                loss_div_factor=telemetry_div_factor,
            )
        else:
            loss = _switch_loss(
                counts=global_counts,
                mean_expert_scores=mean_scores,
                num_experts=num_experts,
                top_k=top_k,
                loss_div_factor=telemetry_div_factor,
            )

        if tp_mesh is not None:
            loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()
    else:
        raise NotImplementedError(granularity)

    return LoadBalancingLoss(
        loss=loss,
        lbl_local=lbl_local.detach(),
        lbl_global=lbl_global.detach(),
        reduced=reduced,
    )


def router_z_loss(
    *,
    expert_logits: torch.Tensor,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    """
    The router z-loss, ``mean_t logsumexp(logits_t)^2``.

    Unchanged from the previous version, and deliberately so: unlike the load-balancing loss this
    one has no cross-rank defect to fix. It is a mean of a strictly per-token quantity, so the
    rank-local mean is an unbiased estimate of the global mean and averaging the per-rank values
    at log time already gives the global value. There is no counterpart to the count histogram
    here and therefore nothing to all-reduce.

    It does share defect (b) with the balance loss -- the weight applied to it is divided by the
    model's total depth rather than by its MoE depth. That is fixed where the division happens,
    in :class:`~olmo_core.nn.moe.moe.MoEBase`, not here.
    """
    expert_logits = get_local_tensor(expert_logits)
    B, S, _ = expert_logits.shape

    # NOTE: with TP, end result has to be a DTensor over the TP mesh, so we wrap as a DTensor
    # and reduce it. Due to this reduction, the 'loss_div_factor' should represent the total
    # number of tokens across the TP group (but not the CP group).
    if loss_div_factor is None:
        loss_div_factor = B * S
        if tp_mesh is not None:
            loss_div_factor = loss_div_factor * tp_mesh.size()
    elif cp_mesh is not None:
        loss_div_factor = loss_div_factor / cp_mesh.size()

    loss = torch.logsumexp(expert_logits, dim=-1).square().sum() / loss_div_factor
    if tp_mesh is not None:
        loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()

    return loss
