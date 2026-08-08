"""
Muon with a selectable post-orthogonalization constraint, so that *Hyperball* (MuonH) and
decoupled weight decay (MuonW) can be compared without changing anything else.

Hyperball (https://arxiv.org/abs/2606.16899) is an optimizer *wrapper* rather than an
optimizer. It fixes two Frobenius norms that decoupled weight decay only controls
indirectly: the norm of each constrained weight matrix, and the norm of the update applied
to it. Writing :math:`\\widehat{X} := X / \\lVert X \\rVert_F`, one step is

.. math::

    W_{t+1} = R \\cdot \\widehat{\\left( W_t - \\eta_t R \\, \\widehat{u_t} \\right)}

where :math:`u_t` is whatever update the base optimizer produced and :math:`R` is a radius
held fixed at :math:`\\lVert W_0 \\rVert_F`. The step therefore has length exactly
:math:`\\eta_t R` before the radial projection, which is what makes :math:`\\eta_t` a
*relative* step size and (per the paper) roughly transferable across model scale.

MuonH is this wrapper with Muon's :math:`\\operatorname{msign}(M_t)` as the base update.
Because step 2 renormalizes :math:`u_t`, every per-matrix scalar Muon might apply --
``adjust_lr``, Moonlight's :math:`0.2\\sqrt{\\max(d_{in}, d_{out})}` -- cancels exactly, so
only the *direction* :math:`\\operatorname{msign}(M_t)` survives. Hyperball matrices take no
weight decay: the constraint replaces it.

**Why one class and not two.** :class:`MuonHConfig` and :class:`MuonWConfig` build the same
:class:`MuonH` optimizer and differ only in ``constraint``. Both arms therefore share the
identical momentum and Newton-Schulz code path, which is the point when the two are being
compared -- a baseline that reached ``msign`` by a different route would confound the
comparison with an implementation difference. :class:`~olmo_core.optim.muon.MuonConfig`,
which wraps the external ``dion`` package, is left alone and is the faster choice for
production runs at scale; see "Cost" below.

**Blocked matrices, which MoE makes mandatory.** OLMo-core stores MoE expert weights with
the expert dimension folded into rows: ``w1`` is ``(num_experts * d_model, hidden_size)``.
That is one 2D tensor holding ``num_experts`` independent matrices, so orthogonalizing it
whole is wrong -- it mixes experts, and it computes ``adjust_lr`` and the Hyperball radius
from the stacked shape instead of an expert's. Set ``block_rows`` on the param group and
every operation here (``msign``, both Frobenius norms, the radius, ``adjust_lr``) runs
per block instead. :meth:`MuonHConfig.default_group_overrides` derives it from the owning
module's ``num_experts``, for both arms alike.

This blocking is an extension, not something the paper prescribes: Hyperball's paper says
nothing about MoE or about grouped weight tensors, and constrains "attention and MLP weight
matrices" one individual matrix at a time. Treating each expert as its own matrix is the
reading that follows from that; treating the stacked tensor as one matrix is not.

**Initialization is part of the method.** :math:`R` comes from :math:`W_0`, so the
initializer decides the step scale: the paper initializes constrained matrices with standard
deviation :math:`1/\\sqrt{d_{in}}`, which puts :math:`R = \\lVert W_0 \\rVert_F` at
:math:`\\sqrt{d_{out}}` and is what its theory assumes. That is
:attr:`~olmo_core.nn.transformer.init.InitMethod.fan_in` in this repository, and it is not
:attr:`~olmo_core.nn.transformer.init.InitMethod.normal`, which is what ``llama_like`` and so
every ``olmo2_*`` factory default to. Choose it deliberately, and choose the same one for both
arms. :attr:`_MuonHybridConfig.radius_scale` rescales :math:`R` if a different sphere is
wanted without changing the initializer.

Note also that the paper's ``u_t`` is ``msign`` of a plain EMA with no Nesterov term, which is
why ``nesterov`` defaults to ``False`` here even though the reference Muon implementation
defaults to ``True``.

**Cost.** ``msign`` needs whole matrices, so a matrix sharded across ranks is all-gathered,
updated, and sliced back. Expert weights are the exception and it is the case that matters:
FSDP shards them on the expert-major row dimension, so whenever the shard boundary lands on
a block boundary every block is already rank-local and the step needs no communication at
all. ``dion``'s Muon avoids the gather for dense matrices too, with a megabatched all-to-all
this module does not attempt. At 370M the largest gathered matrix is a few MB and both arms
pay it identically, so relative timings stay fair; at larger scale prefer ``dion``.
"""

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import Optimizer

from ..config import StrEnum
from ..distributed.utils import distribute_like, get_full_tensor, get_local_tensor
from .adamw import adamw_step
from .config import MatrixAwareOptimConfig, OptimConfig, OptimGroupOverride

__all__ = [
    "MuonConstraint",
    "MuonH",
    "MuonHConfig",
    "MuonWConfig",
    "newton_schulz_msign",
]

log = logging.getLogger(__name__)

NS5_COEFFICIENTS: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
"""
The quintic coefficients of the Newton-Schulz iteration used by Muon, applied for
``ns_steps`` iterations. They do not converge to the exact matrix sign -- they are tuned to
push the singular values into a band around 1 in very few steps, which is all the update
direction needs.
"""


class MuonConstraint(StrEnum):
    """
    What to do with a weight matrix once its update has been orthogonalized.
    """

    hyperball = "hyperball"
    """
    Fix ``||W||_F`` and the update length, per https://arxiv.org/abs/2606.16899. This is the
    "H" in MuonH. Weight decay does not apply and ``adjust_lr`` has no effect, because the
    update is renormalized.
    """

    weight_decay = "weight_decay"
    """
    Standard Muon: scale the orthogonalized update by ``adjust_lr`` and apply decoupled
    weight decay. This is the "W" in MuonW, and the control arm for Hyperball.
    """


class MuonAdjustLR(StrEnum):
    """
    How to scale the orthogonalized update, for :attr:`MuonConstraint.weight_decay` only.
    """

    rms_norm = "rms_norm"
    """
    ``0.2 * sqrt(max(d_out, d_in))``, which puts Muon's update on the same RMS scale as
    Adam's so that one learning rate suits both (Moonlight, https://arxiv.org/abs/2502.16982).
    """

    spectral_norm = "spectral_norm"
    """
    ``max(1, sqrt(d_out / d_in))``, for learning-rate transfer across model scale. This is the
    ``s_mu`` of Hyperball's paper and of the reference Muon implementation.
    """


def newton_schulz_msign(
    G: torch.Tensor, *, steps: int = 5, eps: float = 1e-30, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """
    Approximate ``msign(G) = P @ Q.T`` for ``G = P @ S @ Q.T``, by the quintic Newton-Schulz
    iteration Muon uses.

    Batched over every leading dimension, which is how blocked (MoE expert) matrices are
    handled: pass ``(num_blocks, rows, cols)`` and each block is orthogonalized on its own.

    :param G: A tensor of shape ``(..., m, n)``.
    :param steps: Number of quintic iterations.
    :param eps: Floor on the Frobenius norm before dividing, so that an all-zero input (an
        expert that received no tokens, or step 0 with zero momentum) yields zero rather than
        NaN. A floor rather than an additive term on purpose -- see below.
    :param dtype: Compute dtype for the iteration. bfloat16 matches Muon and is ample for a
        direction. The initial normalization is done in ``G``'s own dtype regardless.

    :returns: A tensor shaped like ``G``, in ``G``'s dtype, whose singular values are all
        approximately 1.
    """
    if G.ndim < 2:
        raise ValueError(f"newton_schulz_msign expects at least 2 dims, got {G.ndim}")

    a, b, c = NS5_COEFFICIENTS
    out_dtype = G.dtype
    X = G

    # The iteration costs less when the Gram matrix is over the shorter side.
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT

    # Bound the spectral norm at 1 so the iteration is in its region of convergence. The norm
    # is per matrix, not over the whole batch, so one dead expert cannot rescale the others.
    #
    # Two departures from the reference implementation, both to keep msign scale-free, which
    # it is in exact arithmetic and which Hyperball leans on much harder than weight decay
    # does: Hyperball renormalizes the update, so the momentum's magnitude is *only* ever a
    # scale -- and it drifts by orders of magnitude between warmup and the end of decay.
    #
    #  1. Normalize before the cast, not after. Rounding ``c * G`` to bfloat16 is not ``c``
    #     times the rounding of ``G``, so casting first makes the returned direction depend
    #     slightly on the input's magnitude.
    #  2. Floor the divisor instead of adding to it. ``norm + eps`` shrinks the result by
    #     ``eps/norm``, which is a different amount for differently-scaled inputs; the
    #     reference's 1e-7 is negligible against a raw gradient and is not negligible against
    #     one that has decayed. ``clamp_min`` leaves any nonzero norm exactly alone and still
    #     turns an all-zero block into zero rather than NaN.
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X.to(dtype)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(out_dtype)


def _blocked_view(x: torch.Tensor, block_rows: Optional[int]) -> torch.Tensor:
    """
    View a 2D tensor as ``(num_blocks, block_rows, cols)``, sharing storage so that in-place
    writes reach the original. ``block_rows=None`` means the whole tensor is one block.
    """
    if x.ndim != 2:
        raise ValueError(f"expected a 2D tensor, got shape {tuple(x.shape)}")
    if block_rows is None:
        return x.unsqueeze(0)
    rows = x.shape[0]
    if rows % block_rows != 0:
        raise ValueError(f"{rows} rows do not divide into blocks of {block_rows}")
    return x.view(rows // block_rows, block_rows, x.shape[1])


def _adjust_lr_factor(rows: int, cols: int, strategy: Optional[Union[MuonAdjustLR, str]]) -> float:
    """
    The per-matrix multiplier applied to the learning rate under
    :attr:`MuonConstraint.weight_decay`. ``rows``/``cols`` are one block's dimensions.
    """
    if strategy is None:
        return 1.0
    strategy = MuonAdjustLR(strategy)
    if strategy == MuonAdjustLR.rms_norm:
        return 0.2 * math.sqrt(max(rows, cols))
    if strategy == MuonAdjustLR.spectral_norm:
        return max(1.0, math.sqrt(rows / cols))
    raise NotImplementedError(strategy)


def _sharded_block_local(p: torch.Tensor, block_rows: Optional[int]) -> bool:
    """
    Whether every block of ``p`` lives wholly inside this rank's shard, so the update needs
    no communication.

    True when the parameter is not sharded at all, and -- the case that matters -- when it is
    sharded evenly on dim 0 and each rank's row count is a whole number of blocks. FSDP
    shards MoE expert weights on the expert-major dimension, so this holds for any world size
    that divides the expert count.
    """
    if not isinstance(p, DTensor):
        return True

    placements = p.placements
    if len(placements) != 1 or not placements[0].is_shard():
        # Replicated, or a mesh this module does not reason about. Fall back to the gather.
        return all(pl.is_replicate() for pl in placements)

    if placements[0].dim != 0:  # type: ignore[attr-defined]
        return False
    if block_rows is None:
        return False

    local_rows = get_local_tensor(p).shape[0]
    full_rows = p.shape[0]
    world_size = p.device_mesh.size()
    # Uneven sharding pads the last shard, which puts the shard offsets off a block boundary.
    if full_rows != local_rows * world_size:
        return False
    return local_rows % block_rows == 0


class MuonH(Optimizer):
    """
    Muon whose post-orthogonalization step is either Hyperball (MuonH) or decoupled weight
    decay (MuonW), selected per param group by ``constraint``.

    Only matrix parameters belong here. Embeddings, the LM head and every gain or bias are
    optimized with AdamW, in groups carrying ``algorithm="adamw"`` --
    :meth:`MuonHConfig.default_group_overrides` builds that split.

    Per-group options: ``algorithm`` (``"muon"`` or ``"adamw"``), ``lr``, ``mu``, ``betas``,
    ``eps``, ``weight_decay``, ``nesterov``, ``ns_steps``, ``constraint``, ``adjust_lr``,
    ``block_rows``, ``radius_scale``.

    .. note::
        The radius is deliberately absent from :meth:`state_dict`. Hyperball's own invariant
        is ``||W_b||_F == R_b``, so the radius is recoverable from the weights themselves and
        recomputing it on the first step after a resume gives back the same number it had.
        That keeps the checkpointed state identical to plain Muon's -- momentum only.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        mu: float = 0.95,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        nesterov: bool = False,
        ns_steps: int = 5,
        constraint: Union[MuonConstraint, str] = MuonConstraint.hyperball,
        adjust_lr: Optional[Union[MuonAdjustLR, str]] = MuonAdjustLR.rms_norm,
        radius_scale: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= mu <= 1.0:
            raise ValueError(f"invalid mu: {mu}")
        if not all(0.0 <= beta <= 1.0 for beta in betas):
            raise ValueError(f"invalid betas: {betas}")
        if ns_steps < 1:
            raise ValueError(f"invalid ns_steps: {ns_steps}")
        if radius_scale <= 0.0:
            raise ValueError(f"invalid radius_scale: {radius_scale}")

        defaults = dict(
            algorithm="muon",
            lr=lr,
            mu=mu,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            constraint=MuonConstraint(constraint),
            adjust_lr=None if adjust_lr is None else MuonAdjustLR(adjust_lr),
            block_rows=None,
            radius_scale=radius_scale,
        )
        super().__init__(params, defaults)

        # Keyed by parameter identity rather than held in ``self.state``, so that the
        # checkpointed optimizer state stays shape-compatible with plain Muon's. See the
        # class docstring for why losing it on resume is harmless.
        self._radii: Dict[torch.Tensor, torch.Tensor] = {}
        self._last_step_metrics: Dict[str, float] = {}
        self._norm_accumulator: List[torch.Tensor] = []
        self._drift_accumulator: List[torch.Tensor] = []

    def latest_metrics(self) -> Dict[str, float]:
        """
        Diagnostics from the most recent :meth:`step`, for logging.

        Every number here is a by-product of the step itself rather than extra work, which is
        why it is worth logging every step.

        ``matrix_norm_mean`` is the mean Frobenius norm over constrained blocks.
        ``radius_relative_drift_max`` is the largest ``| ||W_b||_F / R_b - 1 |`` over blocks and
        is only present under Hyperball -- it is the constraint's own invariant, so it should sit
        at the floor of fp32 accumulation for the whole run. If it climbs, the constraint is not
        holding and the arm is no longer testing what it claims to.

        Values are this rank's blocks only. For a sharded parameter that means a slice, so
        treat these as a monitor rather than a global reduction.
        """
        return dict(self._last_step_metrics)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        self._norm_accumulator = []
        self._drift_accumulator = []

        for group in self.param_groups:
            if group.get("algorithm", "muon") == "adamw":
                self._adamw_group_step(group)
            else:
                self._muon_group_step(group)

        metrics: Dict[str, float] = {}
        if self._norm_accumulator:
            norms = torch.cat(self._norm_accumulator)
            metrics["matrix_norm_mean"] = norms.mean().item()
            metrics["matrix_norm_min"] = norms.min().item()
            metrics["matrix_norm_max"] = norms.max().item()
        if self._drift_accumulator:
            metrics["radius_relative_drift_max"] = torch.cat(self._drift_accumulator).max().item()
        self._last_step_metrics = metrics
        self._norm_accumulator = []
        self._drift_accumulator = []

    def _adamw_group_step(self, group: Dict[str, Any]) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] = torch.tensor(0.0, device=p.device)
            adamw_step(
                p,
                p.grad,
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                exp_avg=state["exp_avg"],
                exp_avg_sq=state["exp_avg_sq"],
                step=state["step"],
                step_factor=torch.ones((), device=p.device),
            )

    def _muon_group_step(self, group: Dict[str, Any]) -> None:
        mu = group["mu"]
        block_rows: Optional[int] = group.get("block_rows")

        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim != 2:
                raise RuntimeError(
                    f"MuonH expects 2D parameters in a muon group, got {p.ndim}D "
                    f"{tuple(p.shape)}; route it to an adamw group instead"
                )

            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(p)
            momentum = state["momentum"]

            # Momentum and the Nesterov mix are element-wise, so they run on whatever layout
            # the parameter already has -- sharded or not, no communication.
            grad = p.grad
            momentum.lerp_(grad, 1 - mu)
            update = grad.lerp(momentum, mu) if group["nesterov"] else momentum

            if _sharded_block_local(p, block_rows):
                self._apply_blocked_(
                    get_local_tensor(p), get_local_tensor(update), p, group, block_rows
                )
            else:
                # msign needs whole matrices, so gather, update a full copy, and slice this
                # rank's rows back out.
                full_weight = get_full_tensor(p).clone()
                self._apply_blocked_(full_weight, get_full_tensor(update), p, group, block_rows)
                get_local_tensor(p).copy_(get_local_tensor(distribute_like(p, full_weight)))

    def _radius_for(
        self, key: torch.Tensor, weight_blocks: torch.Tensor, radius_scale: float
    ) -> torch.Tensor:
        """
        The fixed Hyperball radius per block, ``R_b = radius_scale * ||W_{0,b}||_F``.

        Computed once, on the first step that touches ``key``. On a fresh run that first step
        sees the initialization and so measures exactly what the paper defines; after a resume
        it sees weights the constraint has been holding at ``R_b`` all along and measures the
        same number back.
        """
        radius = self._radii.get(key)
        if radius is None:
            radius = weight_blocks.norm(dim=(-2, -1), keepdim=True).to(torch.float32)
            if radius_scale != 1.0:
                # W_0 is on the sphere of radius ||W_0||_F by definition, so it only needs
                # moving onto the sphere we actually asked for.
                radius = radius * radius_scale
                weight_blocks.mul_(
                    (radius / weight_blocks.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)).to(
                        weight_blocks.dtype
                    )
                )
            self._radii[key] = radius
        return radius

    def _apply_blocked_(
        self,
        weight: torch.Tensor,
        update: torch.Tensor,
        key: torch.Tensor,
        group: Dict[str, Any],
        block_rows: Optional[int],
    ) -> None:
        """
        Orthogonalize ``update`` and write the constrained result into ``weight``, in place,
        one independent matrix block at a time. Both are plain 2D tensors.
        """
        weight_blocks = _blocked_view(weight, block_rows)
        update_blocks = _blocked_view(update, block_rows)

        ortho = newton_schulz_msign(update_blocks, steps=group["ns_steps"], eps=group["eps"]).to(
            torch.float32
        )

        lr = group["lr"]
        if isinstance(lr, torch.Tensor):
            lr = lr.item()

        if MuonConstraint(group["constraint"]) == MuonConstraint.hyperball:
            radius = self._radius_for(key, weight_blocks, group["radius_scale"])

            # u_hat, so the step length below is exactly lr * R and every scalar Muon might
            # have applied to the update cancels.
            ortho.div_(ortho.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30))

            weight_blocks.add_((ortho * (-lr * radius)).to(weight_blocks.dtype))
            # Radial projection back onto the sphere: W <- R * W / ||W||_F.
            unprojected = weight_blocks.norm(dim=(-2, -1), keepdim=True).to(torch.float32)
            weight_blocks.mul_((radius / unprojected.clamp_min(1e-30)).to(weight_blocks.dtype))

            # Free: the projection already needed both norms. The drift is measured AFTER the
            # projection, so it reports what the constraint actually achieved rather than what
            # it was handed.
            projected = weight_blocks.norm(dim=(-2, -1)).to(torch.float32)
            self._norm_accumulator.append(projected.flatten())
            self._drift_accumulator.append(
                (projected / radius.flatten().clamp_min(1e-30) - 1.0).abs().flatten()
            )
        else:
            factor = _adjust_lr_factor(
                weight_blocks.shape[-2], weight_blocks.shape[-1], group["adjust_lr"]
            )
            # Decoupled weight decay at the base lr, with the update at the adjusted one --
            # the same split dion's Muon makes.
            weight_blocks.mul_(1 - lr * group["weight_decay"])
            weight_blocks.add_(ortho.to(weight_blocks.dtype), alpha=-lr * factor)
            # No radius to compare against on this arm, but the norm itself is the interesting
            # half: weight decay lets it move, and where it settles is what Hyperball pins.
            self._norm_accumulator.append(
                weight_blocks.norm(dim=(-2, -1)).to(torch.float32).flatten()
            )


@dataclass
class _MuonHybridConfig(MatrixAwareOptimConfig[MuonH]):
    """
    Shared configuration for the two arms. Subclasses fix ``constraint``.
    """

    lr: float = 0.01
    """
    Learning rate for the matrix (Muon) groups.

    Under :attr:`MuonConstraint.hyperball` this is a *relative* step size: the update has
    length ``lr * ||W||_F`` before projection, so it is dimensionless and does not need
    rescaling with matrix shape. Under :attr:`MuonConstraint.weight_decay` it is scaled per
    matrix by :attr:`adjust_lr`.
    """

    adamw_lr: Optional[float] = None
    """
    Learning rate for the AdamW groups (embeddings, LM head, gains). ``None`` uses
    :attr:`lr`, which is rarely what you want -- a Muon learning rate is much larger than an
    AdamW one. Set it explicitly.
    """

    mu: float = 0.95
    """Momentum coefficient for the matrix groups."""

    betas: Tuple[float, float] = (0.9, 0.95)
    """Betas for the AdamW groups."""

    eps: float = 1e-8
    """Epsilon for AdamW, and the Newton-Schulz normalization floor."""

    weight_decay: float = 0.1
    """
    Weight decay. Applied to the AdamW groups always, and to the matrix groups only under
    :attr:`MuonConstraint.weight_decay` -- Hyperball replaces it with the norm constraint.
    """

    nesterov: bool = False
    """
    Whether to mix the gradient back into the momentum before orthogonalizing. ``False``
    matches Hyperball's paper, whose ``u_t`` is ``msign`` of the plain EMA; the reference Muon
    implementation defaults this to ``True``, so it is a flag rather than a constant.
    """

    ns_steps: int = 5
    """Newton-Schulz iterations per step."""

    adjust_lr: Optional[MuonAdjustLR] = MuonAdjustLR.rms_norm
    """
    Per-matrix learning-rate scaling, for :attr:`MuonConstraint.weight_decay` only. It has no
    effect under Hyperball, which renormalizes the update.
    """

    radius_scale: float = 1.0
    """
    Multiplier on the radius taken from initialization, ``R = radius_scale * ||W_0||_F``.
    Hyperball only. ``1.0`` is the paper's definition; anything else rescales the weights onto
    the requested sphere on the first step.
    """

    @classmethod
    def constraint(cls) -> MuonConstraint:
        raise NotImplementedError

    @classmethod
    def optimizer(cls) -> Type[MuonH]:
        return MuonH

    def _expert_block_rows(self, model: nn.Module) -> Dict[str, int]:
        """
        Map each stacked expert weight's FQN to the row count of one expert's matrix.

        OLMo-core folds the expert dimension into rows, so ``w1`` of an ``E``-expert layer is
        ``(E * d_model, hidden_size)``. Dividing the row count by ``E`` recovers one expert's
        matrix for any of ``w1``/``w2``/``w3`` without hard-coding which is which.
        """
        block_rows: Dict[str, int] = {}
        for module_name, module in model.named_modules():
            num_experts = getattr(module, "num_experts", None)
            if not isinstance(num_experts, int) or num_experts <= 1:
                continue
            for param_name, param in module.named_parameters(recurse=False):
                if param.ndim != 2:
                    continue
                rows = param.shape[0]
                if rows % num_experts != 0:
                    raise RuntimeError(
                        f"{module_name}.{param_name}: {rows} rows do not divide by "
                        f"{num_experts} experts, so per-expert blocks cannot be derived"
                    )
                block_rows[f"{module_name}.{param_name}"] = rows // num_experts
        return block_rows

    def default_group_overrides(self, model: nn.Module) -> List[OptimGroupOverride]:
        """
        Split the parameters into Muon groups (2D hidden weights, blocked per expert where
        applicable) and AdamW groups (embeddings, LM head, gains and biases).
        """
        params = self.categorize_parameters(model)
        adamw_lr = self.adamw_lr if self.adamw_lr is not None else self.lr

        block_rows = self._expert_block_rows(model)
        # Group the blocked matrices by their block size; everything else is one plain group.
        by_block_size: Dict[int, List[str]] = OrderedDict()
        plain: List[str] = []
        for name in params["matrix"]:
            if name in block_rows:
                by_block_size.setdefault(block_rows[name], []).append(name)
            else:
                plain.append(name)

        overrides: List[OptimGroupOverride] = []
        if plain:
            overrides.append(OptimGroupOverride(params=plain, opts=dict()))
        for size, names in by_block_size.items():
            overrides.append(OptimGroupOverride(params=names, opts=dict(block_rows=size)))

        # Embeddings take no weight decay, matching the recipes elsewhere in this repository.
        overrides.append(
            OptimGroupOverride(
                params=params["embed"],
                opts=dict(algorithm="adamw", lr=adamw_lr, weight_decay=0.0),
            )
        )
        overrides.append(
            OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw", lr=adamw_lr))
        )
        overrides.append(
            OptimGroupOverride(params=params["lm_head"], opts=dict(algorithm="adamw", lr=adamw_lr))
        )
        return [o for o in overrides if o.params]

    def create_optimizer(self, model: nn.Module, strict: bool = True, **kwargs) -> MuonH:
        # ``adamw_lr`` is spent building the groups, so it is not a constructor argument.
        kwargs.pop("adamw_lr", None)
        return MuonH(
            self.build_groups(model, strict=strict), constraint=self.constraint(), **kwargs
        )


@OptimConfig.register("muon_h")
@dataclass
class MuonHConfig(_MuonHybridConfig):
    """
    MuonH: Muon wrapped in Hyperball (https://arxiv.org/abs/2606.16899).

    Matrix parameters get ``msign``-orthogonalized momentum, renormalized to unit Frobenius
    norm, a step of length ``lr * R``, and a radial projection back onto the sphere of radius
    ``R = ||W_0||_F``. They take no weight decay. Everything else is AdamW.

    :attr:`lr` here is a relative step size and is not comparable to an AdamW learning rate,
    nor to :class:`MuonWConfig`'s. Sweep it.
    """

    @classmethod
    def constraint(cls) -> MuonConstraint:
        return MuonConstraint.hyperball


@OptimConfig.register("muon_w")
@dataclass
class MuonWConfig(_MuonHybridConfig):
    """
    MuonW: standard Muon with decoupled weight decay, and the control arm for
    :class:`MuonHConfig`.

    Identical to :class:`MuonHConfig` up to the post-orthogonalization step, deliberately, so
    that a comparison between the two isolates Hyperball.
    """

    @classmethod
    def constraint(cls) -> MuonConstraint:
        return MuonConstraint.weight_decay
