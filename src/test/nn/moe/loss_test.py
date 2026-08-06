"""
Tests for the MoE auxiliary losses.

Every assertion here is on a **magnitude**, not on existence. "the loss is finite" and "the metric
is logged" are the shape of check that let a 6.2x gate-mass error and a 15.97-valued normalised
entropy ship, so they are not used.

The load-balancing loss is normalised to exactly ``1.0`` under uniform routing at every ``E`` and
``k``, which gives every test below a closed-form expected value rather than a golden number.

The all-reduce is tested with ``run_distributed_test`` over the **gloo** backend, which needs no
GPU. The adversarial case -- a global value that is secretly the local one under a new name -- is
:func:`test_lbl_global_differs_from_local_under_skewed_ranks`, which constructs ranks that disagree
and requires the two to differ by a computed amount.
"""

import math
import os

import pytest
import torch
import torch.distributed as dist

from olmo_core.nn.moe.loss import (
    MoELoadBalancingLossGranularity,
    load_balancing_loss,
    reduce_expert_counts,
    router_z_loss,
)
from olmo_core.nn.moe.router import MoELinearRouter
from olmo_core.testing import run_distributed_test


def _uniform_inputs(num_experts: int, top_k: int, B: int, S: int):
    """
    Router scores and counts for perfectly uniform routing.

    ``scores`` is exactly ``1/E`` everywhere and ``counts`` is exactly ``T*k/E`` everywhere, so the
    expected loss is the closed-form ``1.0`` rather than an approximation of it.
    """
    tokens = B * S
    assert (tokens * top_k) % num_experts == 0, "choose sizes that divide evenly"
    scores = torch.full((B, S, num_experts), 1.0 / num_experts, dtype=torch.float32)
    counts = torch.full((num_experts,), tokens * top_k // num_experts, dtype=torch.float32)
    batched = torch.full((B, num_experts), S * top_k / num_experts, dtype=torch.float32)
    return scores, counts, batched


def _call(
    *,
    num_experts,
    top_k,
    scores,
    counts,
    batched,
    granularity=MoELoadBalancingLossGranularity.local_batch,
    loss_div_factor=None,
):
    return load_balancing_loss(
        num_experts=num_experts,
        top_k=top_k,
        expert_scores=scores,
        batch_size_per_expert=counts,
        batched_batch_size_per_expert=batched,
        granularity=granularity,
        loss_div_factor=loss_div_factor,
    )


# ---------------------------------------------------------------------------------------------
# Scale. The property that makes the loss comparable across an expert-count ladder.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_experts,top_k",
    [(8, 2), (16, 2), (32, 8), (64, 8), (128, 8), (256, 8)],
)
@pytest.mark.parametrize(
    "granularity",
    [
        MoELoadBalancingLossGranularity.local_batch,
        MoELoadBalancingLossGranularity.global_batch,
        MoELoadBalancingLossGranularity.instance,
    ],
)
def test_uniform_routing_gives_exactly_one_at_every_E(num_experts, top_k, granularity):
    """
    Under uniform routing the loss is 1.0 at every E and k, for every granularity.

    This is the invariance the E-sweep depends on. Without it, a rung at E=256 would report a
    different balance loss than a rung at E=8 for reasons having nothing to do with routing --
    which is the exact defect that makes stock ``load imbalance`` (max/mean) unusable across E:
    measured 1.0446 at E=8 versus 1.1060 at E=64 under *identical* uniform routing.
    """
    B, S = 2, 256
    scores, counts, batched = _uniform_inputs(num_experts, top_k, B, S)
    out = _call(
        num_experts=num_experts,
        top_k=top_k,
        scores=scores,
        counts=counts,
        batched=batched,
        granularity=granularity,
    )
    torch.testing.assert_close(out.loss, torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.lbl_local, torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.lbl_global, torch.tensor(1.0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_experts", [8, 64, 256])
def test_total_collapse_gives_exactly_E(num_experts):
    """
    All tokens and all score mass on one expert gives exactly E, the loss's maximum.

    Fixes the top of the range as well as the bottom, so a claim like "the balance loss barely
    moved" can be read against a known scale instead of against intuition.
    """
    B, S, top_k = 2, 64, 1
    scores = torch.zeros(B, S, num_experts)
    scores[..., 0] = 1.0
    counts = torch.zeros(num_experts)
    counts[0] = B * S * top_k
    batched = torch.zeros(B, num_experts)
    batched[:, 0] = S * top_k

    out = _call(num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched)
    torch.testing.assert_close(out.loss, torch.tensor(float(num_experts)), rtol=1e-5, atol=1e-5)


def test_loss_is_monotone_in_skew():
    """
    Concentrating routing raises the loss. Sanity on the sign, which a normalisation error flips
    without changing anything else about how the number looks.
    """
    num_experts, top_k, B, S = 8, 1, 2, 64
    tokens = B * S
    values = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        # `frac` of the mass on expert 0, remainder spread over the rest.
        rest = (1.0 - frac) / (num_experts - 1)
        scores = torch.full((B, S, num_experts), rest)
        scores[..., 0] = frac
        counts = torch.full((num_experts,), tokens * rest)
        counts[0] = tokens * frac
        batched = counts.unsqueeze(0).expand(B, -1) / B
        out = _call(
            num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched
        )
        values.append(out.loss.item())

    assert values == sorted(values), values
    # And the endpoints are the closed-form ones.
    assert math.isclose(values[0], 1.0, rel_tol=1e-5), values[0]
    assert math.isclose(values[-1], float(num_experts), rel_tol=1e-5), values[-1]


# ---------------------------------------------------------------------------------------------
# Single-process behaviour of the pair. Must agree, and must say why.
# ---------------------------------------------------------------------------------------------


def test_pair_is_bit_identical_on_one_rank_and_reports_not_reduced():
    """
    With one process the pair must agree **bit-exactly** and ``reduced`` must be False.

    Both halves matter. Agreement on one rank is the null case, and if it did not hold, a
    multi-rank difference could be the arithmetic rather than the reduction. ``reduced=False`` is
    what stops a single-GPU run from being read as evidence that the reduction works -- there was
    nothing to reduce, and a gate that compares the pair has to run on more than one rank.
    """
    scores, counts, batched = _uniform_inputs(64, 8, 2, 128)
    # Deliberately skewed, so equality is not an artifact of symmetry.
    counts = counts.clone()
    counts[0] *= 3
    counts[1] = 0
    out = _call(num_experts=64, top_k=8, scores=scores, counts=counts, batched=batched)
    assert out.reduced is False
    assert out.lbl_local.item() == out.lbl_global.item()


def test_reduce_expert_counts_never_mutates_its_input():
    """
    The counts handed to this function are the same tensor the dispatch path sizes its all-to-all
    and its capacity bins from. An in-place reduce would multiply them by the world size and route
    garbage, silently. This is the safety property of the function.
    """
    counts = torch.tensor([3.0, 5.0, 0.0, 8.0])
    before = counts.clone()
    reduced, did = reduce_expert_counts(counts)
    torch.testing.assert_close(counts, before)
    assert reduced.data_ptr() != counts.data_ptr()
    assert did is False


def test_loss_does_not_mutate_the_count_histogram():
    """The same property, through the public entry point."""
    scores, counts, batched = _uniform_inputs(32, 8, 2, 64)
    counts[3] = 0
    before = counts.clone()
    _call(num_experts=32, top_k=8, scores=scores, counts=counts, batched=batched)
    torch.testing.assert_close(counts, before)


def test_only_the_requested_granularity_carries_grad():
    """
    The telemetry pair must be detached. If it were not, logging it would add a second gradient
    path through the router and the loss actually optimized would depend on whether metrics were
    enabled.
    """
    scores, counts, batched = _uniform_inputs(16, 2, 2, 64)
    scores = scores.clone().requires_grad_(True)
    out = _call(num_experts=16, top_k=2, scores=scores, counts=counts, batched=batched)
    assert out.loss.requires_grad
    assert not out.lbl_local.requires_grad
    assert not out.lbl_global.requires_grad


def test_global_granularity_gradient_matches_local_in_scale():
    """
    Switching granularity must not silently rescale the gradient, or ``lb_loss_weight`` would mean
    two different things under the two settings and the local-vs-global arm would be confounded
    with an effective learning-rate change on the router.

    On one rank the two are the same computation, so the gradients must be identical. That is the
    strongest form of the check available without ranks; the multi-rank version below asserts the
    *values* diverge while the scale stays O(1).
    """
    num_experts, top_k = 32, 8
    grads = []
    for granularity in (
        MoELoadBalancingLossGranularity.local_batch,
        MoELoadBalancingLossGranularity.global_batch,
    ):
        torch.manual_seed(0)
        raw = torch.randn(2, 64, num_experts, requires_grad=True)
        scores = raw.softmax(dim=-1)
        counts = torch.full((num_experts,), 2 * 64 * top_k / num_experts)
        batched = torch.full((2, num_experts), 64 * top_k / num_experts)
        out = _call(
            num_experts=num_experts,
            top_k=top_k,
            scores=scores,
            counts=counts,
            batched=batched,
            granularity=granularity,
        )
        out.loss.backward()
        assert raw.grad is not None
        grads.append(raw.grad.clone())

    torch.testing.assert_close(grads[0], grads[1])


# ---------------------------------------------------------------------------------------------
# The z-loss. Closed form, so no golden number.
# ---------------------------------------------------------------------------------------------


def test_z_loss_closed_form_on_uniform_logits():
    """
    With all logits equal to ``c``, ``logsumexp = c + ln E`` and the loss is ``(c + ln E)**2``.
    """
    num_experts, B, S, c = 64, 2, 32, 0.5
    logits = torch.full((B, S, num_experts), c)
    expected = (c + math.log(num_experts)) ** 2
    loss = router_z_loss(expert_logits=logits)
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_z_loss_is_zero_when_logsumexp_is_zero():
    """
    ``logits = -ln E`` everywhere puts ``logsumexp`` at exactly 0, so the loss is exactly 0. Pins
    the lower end, which a sign or square error would move off zero.
    """
    num_experts, B, S = 16, 2, 8
    logits = torch.full((B, S, num_experts), -math.log(num_experts))
    loss = router_z_loss(expert_logits=logits)
    torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# Distributed. gloo, CPU, no GPU needed.
# ---------------------------------------------------------------------------------------------


def _run_identical_ranks():
    """
    Every rank gets the same skewed histogram. The reduced counts are then ``world_size`` times the
    local ones and the rescale divides by exactly ``world_size``, so the global value must come back
    **bit-identical** to the local one -- while ``reduced`` is True.

    This separates the two ways the pair can agree. If this test failed, a difference in the skewed
    test below could be the rescaling arithmetic rather than the reduction.
    """
    num_experts, top_k, B, S = 16, 2, 2, 64
    scores, counts, batched = _uniform_inputs(num_experts, top_k, B, S)
    counts = counts.clone()
    counts[0] *= 4
    counts[1] = 0

    out = _call(num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched)
    assert out.reduced is True, "collective did not run"
    assert out.lbl_local.item() == out.lbl_global.item(), (
        f"identical ranks must give identical values, got "
        f"local={out.lbl_local.item()!r} global={out.lbl_global.item()!r}"
    )


def test_lbl_global_equals_local_under_identical_ranks():
    run_distributed_test(_run_identical_ranks, world_size=4, backend="gloo")


def _run_skewed_ranks():
    """
    THE ADVERSARIAL CHECK. Each rank sends every token to a *different* expert, so the local
    histograms are maximally disjoint and the pooled histogram is uniform.

    With ``world_size == num_experts``:

    - Rank ``r``'s local counts are ``T*k`` on expert ``r`` and zero elsewhere, and its scores put
      all mass on expert ``r``. So the local loss is the collapse value, exactly ``E``.
    - The pooled counts are ``T*k`` on every expert; rescaled to this rank's budget that is
      ``T*k/E`` each, i.e. uniform. Dotted against a one-hot score vector the global loss is
      exactly ``1.0``.

    So local is ``E`` and global is ``1.0``: a factor of ``E`` apart, both closed-form. A "global"
    path that returned the local value under a new name would read ``E`` here. So would one that
    reduced over a group of size one, or over the wrong group. This is the test that makes the fix
    falsifiable, and it is why it asserts an exact ratio rather than merely ``!=``.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_experts = world_size
    top_k, B, S = 1, 2, 32
    tokens = B * S

    scores = torch.zeros(B, S, num_experts)
    scores[..., rank] = 1.0
    counts = torch.zeros(num_experts)
    counts[rank] = tokens * top_k
    batched = torch.zeros(B, num_experts)
    batched[:, rank] = S * top_k

    out = _call(num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched)

    assert out.reduced is True, "collective did not run"
    torch.testing.assert_close(
        out.lbl_local, torch.tensor(float(num_experts)), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(out.lbl_global, torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    assert (
        out.lbl_global.item() < out.lbl_local.item()
    ), "the global path returned the local value -- the reduction did not happen"


def test_lbl_global_differs_from_local_under_skewed_ranks():
    run_distributed_test(_run_skewed_ranks, world_size=4, backend="gloo")


def _run_global_granularity_is_the_global_value():
    """
    Asking for ``global_batch`` must return the global number, not merely report it alongside.

    A fix that computed the pair correctly for telemetry but kept optimizing the local value would
    pass every test above. So this one checks the *optimized* loss.
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    num_experts = world_size
    top_k, B, S = 1, 2, 32
    tokens = B * S

    scores = torch.zeros(B, S, num_experts, requires_grad=False)
    scores[..., rank] = 1.0
    counts = torch.zeros(num_experts)
    counts[rank] = tokens * top_k
    batched = torch.zeros(B, num_experts)
    batched[:, rank] = S * top_k

    local = _call(
        num_experts=num_experts,
        top_k=top_k,
        scores=scores,
        counts=counts,
        batched=batched,
        granularity=MoELoadBalancingLossGranularity.local_batch,
    )
    glob = _call(
        num_experts=num_experts,
        top_k=top_k,
        scores=scores,
        counts=counts,
        batched=batched,
        granularity=MoELoadBalancingLossGranularity.global_batch,
    )
    torch.testing.assert_close(local.loss, local.lbl_local, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(glob.loss, glob.lbl_global, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(local.loss, torch.tensor(float(num_experts)), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(glob.loss, torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_global_granularity_optimizes_the_global_value():
    run_distributed_test(_run_global_granularity_is_the_global_value, world_size=4, backend="gloo")


def _run_ragged_ranks():
    """
    Unequal token counts per rank -- a ragged final batch, or different amounts of padding.

    The rescale is a ratio of assignment totals rather than a division by the world size, so this
    must work without a special case. Rank ``r`` holds ``(r+1)`` sequences. Routing is uniform
    everywhere, so both values must still be exactly 1.0: the *shape* of the reduction must not
    introduce a bias when the ranks are unbalanced in size.
    """
    rank = dist.get_rank()
    num_experts, top_k, S = 16, 2, 64
    B = rank + 1
    scores, counts, batched = _uniform_inputs(num_experts, top_k, B, S)

    out = _call(num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched)
    assert out.reduced is True
    torch.testing.assert_close(out.lbl_local, torch.tensor(1.0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.lbl_global, torch.tensor(1.0), rtol=1e-5, atol=1e-5)


def test_ragged_rank_sizes_do_not_bias_the_reduction():
    run_distributed_test(_run_ragged_ranks, world_size=4, backend="gloo")


def _run_router_pair_accumulates():
    """
    End to end through a real router: the pair accumulates across micro-batches, over the same
    span as the loss it mirrors, and clears with it.

    The span mismatch is not hypothetical -- the drop-rate metric on this branch reported 44% on a
    router its own entropy put within 1% of uniform, because its numerator accumulated over a
    logging interval and its denominator covered one forward.
    """
    router = MoELinearRouter(
        d_model=32,
        num_experts=8,
        top_k=2,
        lb_loss_weight=0.01,
        lb_loss_granularity=MoELoadBalancingLossGranularity.global_batch,
        z_loss_weight=0.001,
    )
    router.train()
    router.reset_parameters()

    assert router.lbl_local is not None and router.lbl_global is not None
    assert router.lbl_local.item() == 0.0
    assert router.lbl_global.item() == 0.0

    x = torch.randn(2, 16, 32)
    router(x)
    after_one = (router.lbl_local.item(), router.lbl_global.item())
    assert after_one[0] > 0.0 and after_one[1] > 0.0
    assert router.lbl_pooled is True

    router(x)
    after_two = (router.lbl_local.item(), router.lbl_global.item())
    # Same input twice, so each accumulator must have exactly doubled -- the same span as
    # `load_balancing_loss`, which is the property that makes the ratio meaningful.
    assert math.isclose(after_two[0], 2 * after_one[0], rel_tol=1e-5), after_two
    assert math.isclose(after_two[1], 2 * after_one[1], rel_tol=1e-5), after_two
    accumulated = router.load_balancing_loss
    assert accumulated is not None
    assert math.isclose(
        accumulated.item(), 2 * after_one[1], rel_tol=1e-5
    ), "the accumulated loss must equal the accumulated global value at global granularity"

    metrics = router.compute_metrics(reset=True)
    # Both spellings, and they must be the SAME NUMBER. L5's registry takes the bare key (it
    # becomes `block NN/...`); `telemetry-schema.md` registers the flat `moe/...` scalar that a
    # gate asserts on. Two names for one accumulator is safe; two implementations would not be, so
    # this asserts they cannot have drifted.
    for bare, flat in (
        ("lbl_local", "moe/lbl_local"),
        ("lbl_global", "moe/lbl_global"),
        ("lbl_global_over_local", "moe/lbl_global_over_local"),
        ("lbl_not_reduced", "moe/lbl_not_reduced"),
        ("lbl_pooled_world_size", "moe/lbl_pooled_world_size"),
        ("lb_loss_weight_effective", "moe/lb_loss_weight_effective"),
    ):
        assert bare in metrics, bare
        assert flat in metrics, flat
        assert metrics[bare][0].item() == metrics[flat][0].item(), (bare, flat)
        assert metrics[bare][1] == metrics[flat][1], (bare, flat)

    assert metrics["moe/lbl_not_reduced"][0].item() == 0.0
    # The weight this router was handed, reported rather than implied.
    assert metrics["lb_loss_weight_effective"][0].item() == pytest.approx(0.01)
    # And NOT the divisor: this router was constructed directly, so nothing divided its weight and
    # there is no divisor to audit. Emitting one here would claim a division that never happened.
    # `MoEBase` sets it; `aux_weight_test.py` covers that path.
    assert "moe/aux_loss_divisor" not in metrics
    assert "aux_loss_divisor" not in metrics

    # Cleared with the loss.
    assert router.lbl_local.item() == 0.0
    assert router.lbl_global.item() == 0.0


def test_router_accumulates_and_clears_the_pair():
    run_distributed_test(_run_router_pair_accumulates, world_size=2, backend="gloo")


def test_router_reports_not_reduced_on_a_single_process():
    """
    Single-process: the flag must say nothing was pooled. A gate that reads ``lbl_local ==
    lbl_global`` as proof of anything on one rank is reading this flag wrong, and the metric exists
    to make that mistake visible.
    """
    router = MoELinearRouter(
        d_model=32,
        num_experts=8,
        top_k=2,
        lb_loss_weight=0.01,
    )
    router.train()
    router.reset_parameters()
    router(torch.randn(2, 16, 32))
    metrics = router.compute_metrics(reset=False)
    assert metrics["moe/lbl_not_reduced"][0].item() == 1.0
    assert metrics["moe/lbl_pooled_world_size"][0].item() == 1.0
    assert router.lbl_pooled is False


def _run_world_size_one_is_honest():
    """
    THE REGRESSION THAT SHIPPED AND WAS CAUGHT AFTER MERGE.

    ``reduced`` was keyed on ``is_distributed()``. But the platform bootstraps a **single-process
    distributed group** for 1-GPU runs -- the trainer prints "Distributed launch env vars not
    found; bootstrapping a single-process distributed setup" -- so ``is_distributed()`` is ``True``
    with ``world_size == 1``, the ``all_reduce`` executes over one rank, and it pools nothing.

    Under the old predicate ``lbl_not_reduced`` therefore read **0.0** on every 1-GPU platform run,
    which is the value this project's own guidance said means "the global batch was pooled". A gate
    asserting ``lbl_not_reduced == 0.0`` would have passed on a run where no pooling occurred --
    manufacturing exactly the confidence the local/global pair exists to deny.

    The predicate is now group SIZE. This test runs with ``world_size=1`` under an initialised
    process group, which is the platform's 1-GPU shape, and requires the honest answer.
    """
    assert dist.is_initialized(), "this test is meaningless without an initialised group"
    assert dist.get_world_size() == 1, "this test must run at world_size 1"

    num_experts, top_k, B, S = 16, 2, 2, 64
    scores, counts, batched = _uniform_inputs(num_experts, top_k, B, S)
    counts = counts.clone()
    counts[0] *= 3
    counts[1] = 0

    out = _call(num_experts=num_experts, top_k=top_k, scores=scores, counts=counts, batched=batched)
    assert out.reduced is False, (
        "world_size==1 with distributed initialised must report pooled=False: the collective runs "
        "but pools nothing, and reporting True there is the false-confidence bug"
    )
    # And the arithmetic must still be right -- the skip must not change the value.
    assert out.lbl_local.item() == out.lbl_global.item()

    router = MoELinearRouter(d_model=32, num_experts=8, top_k=2, lb_loss_weight=0.01)
    router.train()
    router.reset_parameters()
    router(torch.randn(2, 16, 32))
    metrics = router.compute_metrics(reset=False)
    assert (
        metrics["moe/lbl_not_reduced"][0].item() == 1.0
    ), "a 1-GPU platform run must report NOT pooled"
    assert metrics["moe/lbl_pooled_world_size"][0].item() == 1.0


def test_world_size_one_reports_not_pooled_even_though_distributed_is_initialized():
    # NOT via `run_distributed_test`, which asserts `world_size > 1` -- and that assertion is
    # itself why this bug survived: the harness cannot construct the shape the platform actually
    # runs 1-GPU jobs in, so no existing test could have covered it. Initialise the group directly.
    if dist.is_initialized():
        pytest.skip("must own the process group to initialise it at world_size 1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)
    try:
        _run_world_size_one_is_honest()
    finally:
        dist.destroy_process_group()


def _run_pooled_world_size_counts_ranks():
    """
    ``lbl_pooled_world_size`` must equal the real rank count, not just be nonzero.

    ``lbl_not_reduced == 0.0`` cannot tell a fully-formed 4-rank group from a partially-formed
    2-rank one, and a partial group pools half the batch while reading as success on a boolean. A
    gate should assert this equals the rank count it asked for.
    """
    world_size = dist.get_world_size()
    router = MoELinearRouter(d_model=32, num_experts=8, top_k=2, lb_loss_weight=0.01)
    router.train()
    router.reset_parameters()
    router(torch.randn(2, 16, 32))
    metrics = router.compute_metrics(reset=False)
    assert router.lbl_pooled is True
    assert router.lbl_pooled_world_size == world_size
    assert metrics["moe/lbl_pooled_world_size"][0].item() == float(world_size)
    assert metrics["moe/lbl_not_reduced"][0].item() == 0.0


def test_pooled_world_size_equals_the_rank_count():
    run_distributed_test(_run_pooled_world_size_counts_ranks, world_size=4, backend="gloo")
