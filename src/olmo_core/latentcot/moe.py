"""
Making the latent-CoT arms correct on a Mixture-of-Experts base.

The experiment forks a pretrained checkpoint. When that checkpoint is an MoE, three things the
framework :class:`~olmo_core.train.Trainer` does for a normal run are nobody's job in the Phase-8
direct loop — and the first is a confound, not a rough edge.

**1. The auxiliary losses are per-forward, and the arms do wildly different numbers of forwards.**
Each MoE router computes a load-balancing loss (and optionally a router z-loss) on every forward
and welds it onto the returned activation with :func:`~olmo_core.ops.attach_auxiliary_loss`. That
is an autograd trick: the aux loss receives gradient ``1.0`` whenever the activation is backwarded
through, *regardless of how the main loss was scaled*. Each forward also normalizes its own aux
loss by its own token count, so every forward contributes an O(1) term and **the aux pressure per
optimizer step is proportional to the number of forwards**. ``codi_loss``'s division by the
example count does not touch it.

Count the forwards per example:

===============  ============================================  ===============
Arm              Forwards                                      At ``K=10``
===============  ============================================  ===============
A0 explicit_cot  1 (the written-out CoT)                        1
A1 no_cot        1 (the direct view)                            1
A2/A3/A4 codi    1 teacher + ``K`` thought steps + 1 student    12
===============  ============================================  ===============

So the router would feel **twelve times** the balancing pressure in A2–A4 as in A0. Gate A is
defined on ``acc(A2) − acc(A0)``; a systematic difference in how hard the router is pushed between
exactly those two arms is not a nuisance, it is an alternative explanation for the result. It is
the same species of bug as the pre-run thought-norm drift, which was also arm-dependent.

:func:`normalized_aux_losses` fixes it by dividing each router's ``lb_loss_weight`` and
``z_loss_weight`` by the step's forward count for the duration of the step, so the summed aux
gradient is what a single-forward step would have produced, for every arm alike.

.. warning::
   **Not** ``loss_div_factor``, which is the obvious-looking lever and the wrong one.
   :meth:`~olmo_core.nn.lm_head.LMHead.forward` passes it to ``_finalize_loss``, so it divides the
   **cross-entropy** as well as the router losses — setting it to the step's token count silently
   rescales the language-modelling objective by that factor and with it the effective learning
   rate. Measured on a dense 2-layer model: ``loss_div_factor=1234`` took the loss from 11.67 to
   0.00945, exactly 1/1234. The router weights are the only knob that moves the aux term alone.

**2. ``post_batch()`` is never called.** That is where a router with ``bias_gamma`` set — the
aux-loss-free, DeepSeek-style balancing — applies its score-bias update. Without it that mechanism
silently does nothing and the pretrained routing drifts with no corrective term.

**3. ``reset_auxiliary_metrics()`` is never called**, so the routers' accumulators grow
monotonically for the whole run. They are detached (see ``Router.forward``), so this leaks no
graph and no memory — but the numbers stop meaning "this step", and they are precisely the
expert-balance signals worth watching on a fine-tune that could quietly collapse the routing.

**Requires a GPU, unavoidably.** Every MoE path in this repository routes through
``olmo_core.kernels.moe``, which is Triton, so it is CUDA-only — ``import triton`` fails outright
on macOS. This repo's own MoE tests are all :func:`~olmo_core.testing.requires_gpu` for the same
reason, and the MoE tests for this module follow. What can be checked on CPU is the arithmetic and
the plumbing, and that is what the CPU tests cover.
"""

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

__all__ = [
    "is_moe_model",
    "count_forwards",
    "normalized_aux_losses",
    "reset_router_state",
    "finish_step",
    "collect_router_metrics",
    "describe_moe",
]


def is_moe_model(model: Any) -> bool:
    """
    Whether ``model`` is an MoE transformer, i.e. whether any of this module applies.

    Reads the ``is_moe`` property :class:`~olmo_core.nn.transformer.MoETransformer` sets, and
    treats its absence as "dense" so a plain :class:`~olmo_core.nn.transformer.Transformer` and
    the tiny test stand-ins both answer ``False`` without a try/except at every call site.

    :param model: A built transformer.
    :returns: ``True`` for an MoE model.
    """
    return bool(getattr(model, "is_moe", False))


def count_forwards(examples: List[Dict[str, Any]], *, mode: str) -> int:
    """
    How many model forwards an arm performs for one optimizer step.

    This is the factor the routers' auxiliary-loss weights must be divided by, because each
    forward contributes an independently-normalized O(1) aux term — see the module docstring for
    why leaving it uncorrected is a confound rather than a detail.

    Per example: 1 for ``explicit_cot`` and ``no_cot``; for ``codi``, ``K + 2`` (one teacher
    branch, ``K`` thought-loop steps, one assembled student forward).

    :param examples: The encoded examples for this step (as ``encode_example`` returns them).
    :param mode: ``"explicit_cot"``, ``"no_cot"`` or ``"codi"``.

    :returns: The forward count, at least 1 (never 0, since it is used as a divisor).

    :raises ValueError: On an unknown ``mode``, rather than silently returning a wrong divisor.
    """
    if mode not in ("explicit_cot", "no_cot", "codi"):
        raise ValueError(f"unknown arm mode: {mode!r} (expected explicit_cot | no_cot | codi)")
    if mode in ("explicit_cot", "no_cot"):
        return max(len(examples), 1)
    return max(sum(int(ex["num_continuous_thoughts"]) + 2 for ex in examples), 1)


def _routers(model: Any) -> List[Any]:
    """Every MoE router in the model, or an empty list for a dense one."""
    if not is_moe_model(model):
        return []
    routers = []
    for block in getattr(model, "blocks", {}).values():
        if not getattr(block, "is_moe", False):
            continue
        router = getattr(getattr(block, "feed_forward_moe", None), "router", None)
        if router is not None:
            routers.append(router)
    return routers


@contextmanager
def normalized_aux_losses(model: Any, num_forwards: int) -> Iterator[None]:
    """
    Divide every router's auxiliary-loss weight by ``num_forwards`` for the enclosed step.

    Wrap the loss computation in this. Each of the ``num_forwards`` forwards inside then attaches
    ``weight / num_forwards`` of aux loss, so the total is the ``weight`` a single-forward step
    would apply — making the balancing pressure identical across arms that do very different
    amounts of work.

    A no-op for a dense model, for ``num_forwards <= 1``, and for routers with the weight unset
    (``None`` means that term is off, and off must stay off). The original weights are restored on
    the way out even if the body raises, so a failed step cannot leave the model permanently
    detuned.

    :param model: A built transformer.
    :param num_forwards: From :func:`count_forwards`.
    """
    routers = _routers(model) if num_forwards > 1 else []
    saved = [(r, r.lb_loss_weight, r.z_loss_weight) for r in routers]
    try:
        for router, lb, z in saved:
            if lb is not None:
                router.lb_loss_weight = lb / num_forwards
            if z is not None:
                router.z_loss_weight = z / num_forwards
        yield
    finally:
        for router, lb, z in saved:
            router.lb_loss_weight = lb
            router.z_loss_weight = z


def reset_router_state(model: Any) -> None:
    """
    Clear every router's accumulated metrics, so what is read back describes one step.

    A no-op on a dense model.

    :param model: A built transformer.
    """
    if is_moe_model(model):
        model.reset_auxiliary_metrics()


def finish_step(model: Any, *, dry_run: bool = False) -> None:
    """
    Run the model's end-of-batch MoE bookkeeping — the ``bias_gamma`` score-bias update.

    Call once per optimizer step, after ``backward()``. A no-op on a dense model, and on an MoE
    whose routers have no ``bias_gamma``.

    :param model: A built transformer.
    :param dry_run: Forwarded to ``post_batch``; computes the update without applying it.
    """
    if is_moe_model(model):
        model.post_batch(dry_run=dry_run)


def collect_router_metrics(model: Any, *, reset: bool = True) -> Dict[str, float]:
    """
    Read the routers' metrics back as plain floats, ready for a log line or W&B.

    Keys come from the model and are prefixed ``moe/``. The per-block entries the model also
    reports (``block 00/…``) are dropped: at ``log_every`` resolution the totals are what show
    routing collapse, and one series per block per metric would swamp the run.

    Note that the load-balancing figure reported here is measured under the scaled weights
    :func:`normalized_aux_losses` installs, so it is comparable across arms in the same way the
    gradient is.

    :param model: A built transformer.
    :param reset: Clear the accumulators, so the next read describes the next window.

    :returns: ``{"moe/<metric>": value}``, empty for a dense model.
    """
    if not is_moe_model(model):
        return {}
    out: Dict[str, float] = {}
    for name, (value, _reduction) in model.compute_auxiliary_metrics(reset=reset).items():
        if name.startswith("block "):
            continue
        try:
            out[f"moe/{name}".replace(" ", "_")] = float(value)
        except BaseException:  # noqa: BLE001 -- a metric is never worth the run
            continue
    return out


def describe_moe(model: Any) -> Optional[Dict[str, Any]]:
    """
    A small summary of the MoE configuration actually built, for the run record.

    Worth recording rather than trusting the flags: the arms fork a pretrained checkpoint whose
    expert count and top-k come from *its* config, and a mismatch between what was intended and
    what was loaded is the kind of thing that should be visible in ``metrics.json`` and on the
    W&B run rather than reconstructed afterwards.

    :param model: A built transformer.
    :returns: ``{"num_moe_blocks", "num_experts", "top_k", "lb_loss_weight", "z_loss_weight"}``,
        or ``None`` for a dense model.
    """
    if not is_moe_model(model):
        return None
    routers = _routers(model)
    if not routers:
        return {"num_moe_blocks": 0}
    router = routers[0]
    return {
        "num_moe_blocks": len(routers),
        "num_experts": getattr(router, "num_experts", None),
        "top_k": getattr(router, "top_k", None),
        "lb_loss_weight": getattr(router, "lb_loss_weight", None),
        "z_loss_weight": getattr(router, "z_loss_weight", None),
    }
