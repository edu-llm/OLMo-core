"""
Hyper-Connections (HC) and Manifold-Constrained Hyper-Connections (mHC).

A hyper-connection replaces the single residual stream of a transformer sub-layer with ``n``
parallel streams. Writing :math:`Z_{l-1}` for the ``n x d`` stream matrix of one token, one
wrapped sub-layer ``f`` computes

.. math::

    x   &= h_{pre}^{T} Z_{l-1} \\\\
    Z_l &= H_{res} Z_{l-1} + h_{post} \\otimes f(x)

where :math:`h_{pre}` (length ``n``) is the read-in vector, :math:`h_{post}` (length ``n``) is
the write-out vector, and :math:`H_{res}` (``n x n``) is the residual mixer.

The gates come from learned logits:

.. math::

    h_{pre} = \\sigma(\\theta_{pre}), \\qquad h_{post} = 2 \\sigma(\\theta_{post})

and the residual mixer is one of five parameterisations, enumerated by
:class:`ResidualMixerType`. Four of the five constrain :math:`H_{res}` to the Birkhoff polytope
(the doubly stochastic matrices) — that constraint is what "manifold-constrained" refers to —
and the fifth, :data:`ResidualMixerType.unconstrained`, is the original Hyper-Connections
formulation, kept here as the instability control for ablations.

This module implements the *static* (input-independent) routing only. Dynamic, input-dependent
routing, expert-parallel/MoE integration, tensor and pipeline parallelism, and fused kernels
are deliberately out of scope; see the implementation map in the project's mHC dossier.

References:

- Zhu et al. 2025, `Hyper-Connections <https://arxiv.org/abs/2409.19606>`_ (HC).
- Xie et al. 2026, `Manifold-Constrained Hyper-Connections <https://arxiv.org/abs/2512.24880>`_
  (mHC, Sinkhorn).
- Yang & Gao 2026, mHC-lite (Birkhoff-von Neumann mixture of permutations).
- Zhou et al. 2026, KromHC (Kronecker-factored mixer).
"""

import math
from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.ops import attach_auxiliary_loss

from .config import ModuleConfig

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "ResidualMixerType",
    "StreamCollapseType",
    "StreamUtilisationType",
    "StreamBalanceLossType",
    "HyperConnectionConfig",
    "HyperConnection",
    "StreamCollapseConfig",
    "StreamCollapse",
    "sinkhorn_log_space",
    "permutation_matrices",
    "stream_utilisation",
    "stream_balance_loss",
]


# The largest ``n`` for which enumerating all ``n!`` permutation matrices is sane. ``7! = 5040``
# coefficients per wrapped sub-layer already dwarfs every other routing parameter.
MAX_BIRKHOFF_STREAMS = 6


class ResidualMixerType(StrEnum):
    """
    An enumeration of the residual-mixer (:math:`H_{res}`) parameterisations.

    The parameter counts quoted below are for :math:`H_{res}` alone and exclude the ``2n``
    read-in/write-out gate logits that every variant carries.
    """

    identity = "identity"
    """
    :math:`H_{res} = I_n`. Zero parameters. Each stream keeps its own residual and the streams
    only ever interact through the shared branch input.
    """

    unconstrained = "unconstrained"
    """
    :math:`H_{res}` is a raw learned ``n x n`` matrix, i.e. the original Hyper-Connections of
    Zhu et al. 2025. ``n^2`` parameters. Deliberately *not* doubly stochastic: nothing bounds
    the spectral radius, so this is the instability control an mHC ablation is measured against.
    """

    sinkhorn = "sinkhorn"
    """
    :math:`H_{res} = \\mathrm{Sinkhorn}(\\Theta)`, the exact mHC of Xie et al. 2026. ``n^2``
    parameters. Sinkhorn-Knopp alternates row and column normalisation, here in log space, and
    converges to a doubly stochastic matrix.
    """

    birkhoff = "birkhoff"
    """
    mHC-lite (Yang & Gao 2026). By the Birkhoff-von Neumann theorem every doubly stochastic
    matrix is a convex combination of permutation matrices, so
    :math:`H_{res} = \\sum_k a_k P_k` with :math:`a = \\mathrm{softmax}(\\theta)` over all
    ``n!`` permutations. ``n!`` parameters (24 at ``n = 4``). Exactly doubly stochastic with no
    iteration.
    """

    kronecker = "kronecker"
    """
    KromHC (Zhou et al. 2026). :math:`H_{res} = A_1 \\otimes \\cdots \\otimes A_{\\log_2 n}`
    where each ``2 x 2`` factor is :math:`A_k = [[p, 1-p], [1-p, p]]` for
    :math:`p = \\mathrm{softmax}(\\theta_k)_0`. ``2 \\log_2 n`` parameters (4 at ``n = 4``).
    Kronecker products of doubly stochastic matrices are doubly stochastic. Requires ``n`` to
    be a power of two.
    """


class StreamCollapseType(StrEnum):
    """
    An enumeration of the policies for collapsing ``n`` residual streams back to one.
    """

    mean = "mean"
    """
    Unweighted mean over the stream dimension. Zero parameters.
    """

    softmax = "softmax"
    """
    Learned readout: a softmax over ``n`` logits gives convex weights over the streams.
    ``n`` parameters.
    """


class StreamUtilisationType(StrEnum):
    """
    How "how much is this stream being used" is measured, for the stream-balancing loss.

    The two differ in one respect and it decides whether the loss does anything at all. See
    :func:`stream_utilisation`.
    """

    dispersion = "dispersion"
    """
    The share of residual energy each stream carries **that no other stream carries**, with the
    energy common to all of them counted as one more bin.

    This is the one the treatment uses. Its uniform point is ``n`` streams carrying equal and
    distinct content; its worst point is ``n`` streams carrying the same vector, which is
    exactly stream collapse.
    """

    energy = "energy"
    """
    The share of total residual energy each stream carries. The literal mirror of MoE's
    load-balancing loss with streams in place of experts.

    **Kept as a control and deliberately not the default, because it is degenerate on the case
    it is meant to fix.** ``n`` identical streams have exactly equal energy, so this statistic
    is already uniform at full collapse, the loss is already at its minimum, and the gradient it
    contributes is zero precisely when the problem is worst. Anybody mirroring MoE's loss
    without noticing that would ship a treatment that cannot work and a null result that means
    nothing, which is why the alternative is in the enum rather than in a comment.
    """


class StreamBalanceLossType(StrEnum):
    """
    The functional form of the stream-balancing penalty. See :func:`stream_balance_loss` for
    why the two are not interchangeable.
    """

    entropy = "entropy"
    """
    The normalised entropy deficit, ``1 - H(p) / log K``. The default: its gradient grows as the
    utilisation vector concentrates, so it pushes hardest exactly where collapse is worst.
    """

    squared_share = "squared_share"
    """
    ``(K * sum p^2 - 1) / (K - 1)``, which is the shape MoE's load-balancing loss takes when its
    hard assignment and its soft probability coincide. Nearly flat at the concentrated end, and
    kept so that the choice of form is an arm rather than an assumption.
    """


def permutation_matrices(n: int, *, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Enumerate every ``n x n`` permutation matrix.

    :param n: The number of streams.
    :param device: The device to build the matrices on.

    :returns: A tensor of shape ``(n!, n, n)``, ordered by :func:`itertools.permutations`, so
        index 0 is always the identity.
    """
    index = torch.tensor(list(permutations(range(n))), dtype=torch.long, device=device)
    return torch.eye(n, dtype=torch.float32, device=device)[index]


def sinkhorn_log_space(
    logits: torch.Tensor,
    *,
    n_iters: int = 20,
    eps: float = 1e-6,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Project a matrix of logits onto the doubly stochastic matrices with Sinkhorn-Knopp.

    The alternating row/column normalisation runs entirely in log space, which is what keeps the
    result finite for large-magnitude logits: the equivalent probability-space implementation
    exponentiates first and overflows.

    .. warning::
        20 iterations, the count the mHC paper specifies and the default here, reaches the
        doubly stochastic fixed point only while the logits stay near zero. Sinkhorn's
        convergence rate degrades as the logits grow and the fixed point approaches a
        permutation matrix; past roughly ``|logit| ~ 10`` the result is still finite and
        nonnegative, and its column sums are still exactly 1 because the column normalisation
        is applied last, but its row sums drift away from 1 — by a factor of two or more at
        ``|logit| ~ 100``. A row sum below 1 shrinks that stream's residual, so a run whose
        ``H_res`` logits grow that large is no longer doing what the method says it does.
        Watching the largest absolute residual logit is the cheap way to notice.

    :param logits: Logits of shape ``(..., n, n)``. Entries may be ``-inf`` (that is how the
        residual logit dropout masks an entry out), provided no row or column is entirely
        ``-inf``.
    :param n_iters: The maximum number of Sinkhorn iterations.
    :param eps: A floor for the final normalisation denominators.
    :param tol: Stop early once no log-space entry moves by more than this in an iteration.

    :returns: A nonnegative tensor of the same shape whose row sums and column sums are 1, in
        ``float32`` regardless of the input dtype.
    """
    z = logits.float()
    # Shift by the max for numerical stability. This is a no-op on the normalised result.
    z = z - z.amax(dim=(-2, -1), keepdim=True)

    for _ in range(n_iters):
        z_prev = z
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)
        if (z - z_prev).abs().amax() < tol:
            break

    p = z.exp()
    # A finite number of iterations leaves a small residual asymmetry; clean it up.
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = p / p.sum(dim=-2, keepdim=True).clamp_min(eps)
    return p


def stream_utilisation(
    streams: torch.Tensor, *, statistic: StreamUtilisationType = StreamUtilisationType.dispersion
) -> torch.Tensor:
    """
    How the residual energy is shared out over the streams, as a probability vector.

    :param streams: A tensor of shape ``(batch_size, seq_len, n_streams, d_model)``.
    :param statistic: Which utilisation definition to use.

    :returns: A ``float32`` vector summing to 1 — length ``n_streams`` for
        :data:`StreamUtilisationType.energy` and ``n_streams + 1`` for
        :data:`StreamUtilisationType.dispersion`, whose extra leading entry is the share carried
        by the component common to every stream.

    :raises NotImplementedError: If the statistic is not recognised.

    **Why `dispersion` has an extra bin, which is the whole design of this function.** The
    obvious statistic — each stream's share of the total energy — is uniform when the streams
    are identical, because identical streams have identical energy. That is the collapsed state,
    so a balancing loss built on it is already satisfied exactly where the problem is. Splitting
    each stream into the part every stream shares and the part only it carries fixes that: write
    ``Z_i = Zbar + D_i`` with ``Zbar`` the mean over streams, put ``n * E||Zbar||^2`` in bin zero
    and ``E||D_i||^2`` in bin ``i``, and full collapse puts all the mass in bin zero, which is
    as far from uniform as the vector goes.

    Computed in float32 whatever the activations are, for the reason the rest of the routing is:
    a sum of squares over ``d_model`` in bfloat16 loses most of the precision the ratio needs.
    """
    values = streams.float()
    if statistic == StreamUtilisationType.energy:
        mass = values.pow(2).sum(dim=-1).mean(dim=(0, 1))
    elif statistic == StreamUtilisationType.dispersion:
        n_streams = values.shape[-2]
        common = values.mean(dim=-2, keepdim=True)
        deviation = values - common
        # `n *` on the common bin so that the vector is uniform exactly when each stream's own
        # content matches the content it shares, rather than when it matches n times as much.
        common_mass = n_streams * common.pow(2).sum(dim=-1).squeeze(-1).mean(dim=(0, 1))
        deviation_mass = deviation.pow(2).sum(dim=-1).mean(dim=(0, 1))
        mass = torch.cat([common_mass.reshape(1), deviation_mass], dim=0)
    else:
        raise NotImplementedError(statistic)
    return mass / mass.sum().clamp_min(torch.finfo(mass.dtype).tiny)


def stream_balance_loss(
    utilisation: torch.Tensor,
    *,
    loss_type: "StreamBalanceLossType" = None,  # type: ignore[assignment]
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    How far a utilisation vector is from uniform, on a scale where 0 is uniform and 1 is fully
    concentrated on one bin.

    :param utilisation: A probability vector, as returned by :func:`stream_utilisation`.
    :param loss_type: The functional form. Defaults to
        :data:`StreamBalanceLossType.entropy`.
    :param eps: The floor under a share before its logarithm is taken, which is what bounds the
        entropy form's gradient.

    :returns: A scalar in ``[0, 1]``.

    :raises NotImplementedError: If the loss type is not recognised.

    **The two forms differ in how they behave as the vector concentrates, and the difference
    is measured rather than argued.** Both are zero at uniform and one at full concentration and
    both are smooth, so the choice reads as stylistic. Differentiated with respect to the
    unnormalised masses the loss is really a function of, with ``d`` for one small bin's share:

    ======  ==================  ==================
    ``d``   ``entropy``         ``squared_share``
    ======  ==================  ==================
    1e-1    0.73                0.82
    1e-2    2.65                2.20
    1e-4    5.72                2.50
    1e-6    8.58                2.50
    1e-8    11.45               2.50
    ======  ==================  ==================

    ``squared_share`` saturates at a constant; ``entropy`` grows as ``log(1/d)``. So the
    entropy form pushes harder the worse the collapse is, and the MoE-shaped form does not push
    any harder at ``1e-8`` than at ``1e-4``. It is a factor of 4.6 at the deep end rather than
    orders of magnitude, and it compounds: over 200 optimizer steps in
    ``src/test/nn/stream_balance_test.py`` the entropy form reaches about 18 times the stream
    dispersion the squared-share form does, from the same initialisation.

    Note the first row. At a share of 0.1 the squared-share form is very slightly the stronger
    of the two, so this is not a claim that one dominates everywhere -- it is a claim about the
    regime a collapsed hyper-connection is actually in, which is the last row and not the first.
    """
    if loss_type is None:
        loss_type = StreamBalanceLossType.entropy
    bins = utilisation.shape[-1]
    if loss_type == StreamBalanceLossType.squared_share:
        return (bins * utilisation.pow(2).sum(dim=-1) - 1.0) / (bins - 1)
    if loss_type == StreamBalanceLossType.entropy:
        clamped = utilisation.clamp_min(eps)
        entropy = -(clamped * clamped.log()).sum(dim=-1)
        return 1.0 - entropy / math.log(bins)
    raise NotImplementedError(loss_type)


@dataclass
class HyperConnectionConfig(ModuleConfig):
    """
    A config for building a :class:`HyperConnection`.
    """

    n_streams: int = 4
    """
    The number of residual streams, ``n``.
    """

    mixer: ResidualMixerType = ResidualMixerType.sinkhorn
    """
    The residual-mixer parameterisation.
    """

    init_noise_std: float = 1e-2
    """
    The standard deviation of the Gaussian noise added to every gating logit at initialisation.

    .. note::
        All ``n`` streams start out identical, which leaves an :math:`S_n` permutation symmetry
        that the gradients preserve exactly: without symmetry breaking the streams stay
        identical for the whole run and ``n > 1`` buys nothing. Appendix C of the mHC paper
        gives ``1e-2`` for this noise and Table 13 of the same paper gives ``1e-4``. The two
        are not reconciled anywhere in the paper. ``1e-2`` is the default here because it is
        the value in the prose that explains the mechanism, but the discrepancy is real and the
        field is exposed so an ablation can settle it rather than inherit a guess.
    """

    residual_dropout_p: float = 0.1
    """
    The probability of masking out an individual residual-mixer logit during training. Masked
    entries are set to ``-inf`` *before* the constraint map, so the surviving entries are
    renormalised rather than merely rescaled. Disabled at eval. See
    :meth:`HyperConnection.residual_mixer` for what this means for each mixer.
    """

    collapse: StreamCollapseType = StreamCollapseType.mean
    """
    Unused by :class:`HyperConnection` itself; carried here so that a single config object can
    describe a whole hyper-connected model. See :class:`StreamCollapseConfig`.
    """

    stream_balance_loss_weight: float = 0.0
    """
    The weight on the stream-balancing auxiliary loss. **Zero, and zero is a hard off switch
    rather than a small number.**

    This is the treatment in ``docs/hc-ablation/EXPERIMENT-DESIGN.md`` and the one isolated
    change the experiment turns on. At zero, :meth:`HyperConnection.write_out` does not compute
    the statistic, does not attach a loss, and does not record a metric, so the untreated path
    is bit-identical to the path that existed before this field did — which
    ``src/test/nn/stream_balance_test.py`` asserts by running the code with the statistic
    replaced by something that raises.

    **What it is for.** Every constrained residual mixer starts at the uniform doubly stochastic
    matrix, which averages the streams, which drives them toward carrying the same vector; and
    the gradient that survives the constraint map is exactly the part proportional to how far
    apart they are. Measured on a block at initialisation, that leaves the mixer's gradient norm
    seven to eight orders of magnitude below the unconstrained mixer's on the same inputs, which
    is the mechanism behind the ``1e-9`` gradient a public mHC reproduction reported. A loss
    that keeps the streams apart is what keeps the mixer learning. See
    ``src/test/nn/hc_moe_block_test.py::test_constrained_mixer_gradient_is_orders_below_the_unconstrained_one``.

    A sensible nonzero value is ``0.01``, matching ``MoEConfig.lb_loss_weight``, and it is a
    guess: the losses are on different scales and nothing has tuned this.
    """

    stream_balance_statistic: StreamUtilisationType = StreamUtilisationType.dispersion
    """
    Which utilisation statistic the balancing loss is computed on. See
    :class:`StreamUtilisationType`; the default is the one that is not degenerate at collapse.
    """

    stream_balance_loss_type: StreamBalanceLossType = StreamBalanceLossType.entropy
    """
    The functional form of the penalty. See :class:`StreamBalanceLossType` and
    :func:`stream_balance_loss`; the default is the one whose gradient does not vanish where
    collapse is worst.
    """

    sinkhorn_iters: int = 20
    """
    The number of Sinkhorn-Knopp iterations, for :data:`ResidualMixerType.sinkhorn`.
    """

    sinkhorn_eps: float = 1e-6
    """
    The denominator floor in the final Sinkhorn normalisation.
    """

    dtype: DType = DType.float32
    """
    The dtype the routing quantities (``h_pre``, ``h_post``, ``H_res``, the Sinkhorn iteration)
    are computed in, independent of the activation dtype. Keep this at ``float32``: in
    ``bfloat16`` the Sinkhorn fixed point is reached to about two decimal digits and the row and
    column sums drift far enough off 1 that the doubly stochastic guarantee, which is the whole
    point of mHC, no longer holds.
    """

    def __post_init__(self):
        if self.n_streams < 1:
            raise OLMoConfigurationError(f"'n_streams' must be at least 1, got {self.n_streams}")
        if self.mixer == ResidualMixerType.kronecker and not _is_power_of_two(self.n_streams):
            raise OLMoConfigurationError(
                f"the '{ResidualMixerType.kronecker}' residual mixer factors H_res into 2x2 "
                f"blocks and so requires 'n_streams' to be a power of two, got {self.n_streams}"
            )
        if self.mixer == ResidualMixerType.birkhoff and self.n_streams > MAX_BIRKHOFF_STREAMS:
            raise OLMoConfigurationError(
                f"the '{ResidualMixerType.birkhoff}' residual mixer carries one parameter per "
                f"permutation, so 'n_streams' above {MAX_BIRKHOFF_STREAMS} is impractical "
                f"({math.factorial(self.n_streams):,d} parameters per wrapped sub-layer at "
                f"n_streams={self.n_streams})"
            )
        if not 0.0 <= self.residual_dropout_p < 1.0:
            raise OLMoConfigurationError(
                f"'residual_dropout_p' must be in [0, 1), got {self.residual_dropout_p}"
            )
        if self.init_noise_std < 0.0:
            raise OLMoConfigurationError(
                f"'init_noise_std' must be non-negative, got {self.init_noise_std}"
            )
        if self.stream_balance_loss_weight < 0.0:
            raise OLMoConfigurationError(
                "'stream_balance_loss_weight' must be non-negative, got "
                f"{self.stream_balance_loss_weight}"
            )
        if self.stream_balance_loss_weight > 0.0 and self.n_streams < 2:
            raise OLMoConfigurationError(
                "'stream_balance_loss_weight' is meaningless with one stream: there is nothing "
                "to balance and the loss is identically zero"
            )

    def num_residual_mixer_params(self) -> int:
        """
        The number of learned parameters in :math:`H_{res}` alone.

        :returns: The parameter count.
        """
        n = self.n_streams
        if self.mixer == ResidualMixerType.identity:
            return 0
        elif self.mixer in (ResidualMixerType.unconstrained, ResidualMixerType.sinkhorn):
            return n * n
        elif self.mixer == ResidualMixerType.birkhoff:
            return math.factorial(n)
        elif self.mixer == ResidualMixerType.kronecker:
            return 2 * max(int(math.log2(n)), 0)
        else:
            raise NotImplementedError(self.mixer)

    def num_params(self) -> int:
        """
        The number of routing parameters one wrapped sub-layer adds.

        This is ``2n`` gate logits plus the mixer's own parameters, giving 8 / 24 / 24 / 32 / 12
        for identity / unconstrained / sinkhorn / birkhoff / kronecker at ``n = 4``.

        :returns: The parameter count.
        """
        return 2 * self.n_streams + self.num_residual_mixer_params()

    def build(self, *, init_device: str = "cpu") -> "HyperConnection":
        """
        Build the corresponding :class:`HyperConnection`.

        :param init_device: The device to allocate the routing parameters on.

        :returns: The module.
        """
        return HyperConnection(
            n_streams=self.n_streams,
            mixer=self.mixer,
            init_noise_std=self.init_noise_std,
            residual_dropout_p=self.residual_dropout_p,
            sinkhorn_iters=self.sinkhorn_iters,
            sinkhorn_eps=self.sinkhorn_eps,
            stream_balance_loss_weight=self.stream_balance_loss_weight,
            stream_balance_statistic=self.stream_balance_statistic,
            stream_balance_loss_type=self.stream_balance_loss_type,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class HyperConnection(nn.Module):
    """
    A static hyper-connection around one transformer sub-layer.

    The module owns the read-in gate, the write-out gate and the residual mixer, but not the
    sub-layer itself: :meth:`forward` takes the branch as a callable so that the read-in and the
    write-out cannot drift apart.

    At initialisation the whole thing is the identity on top of an ordinary residual: ``h_pre``
    is the uniform average, ``h_post`` is all ones and every constrained mixer starts at the
    uniform doubly stochastic matrix, so with streams that are all copies of ``z`` the update is
    ``z + f(z)`` in every stream — exactly the unwrapped backbone. Setting
    ``init_noise_std > 0`` perturbs that by design; see :data:`HyperConnectionConfig.init_noise_std`.

    :param n_streams: The number of residual streams, ``n``.
    :param mixer: The residual-mixer parameterisation.
    :param init_noise_std: The standard deviation of the symmetry-breaking noise on the gating
        logits.
    :param residual_dropout_p: The residual-mixer logit dropout probability, applied during
        training only.
    :param sinkhorn_iters: The number of Sinkhorn iterations.
    :param sinkhorn_eps: The Sinkhorn normalisation floor.
    :param dtype: The dtype the routing quantities are computed in.
    :param init_device: The device to allocate parameters on.

    :raises OLMoConfigurationError: If the mixer and ``n_streams`` are incompatible.
    """

    def __init__(
        self,
        *,
        n_streams: int = 4,
        mixer: ResidualMixerType = ResidualMixerType.sinkhorn,
        init_noise_std: float = 1e-2,
        residual_dropout_p: float = 0.1,
        sinkhorn_iters: int = 20,
        sinkhorn_eps: float = 1e-6,
        stream_balance_loss_weight: float = 0.0,
        stream_balance_statistic: StreamUtilisationType = StreamUtilisationType.dispersion,
        stream_balance_loss_type: StreamBalanceLossType = StreamBalanceLossType.entropy,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()

        # Route every argument through the config so that the validation lives in one place.
        config = HyperConnectionConfig(
            n_streams=n_streams,
            mixer=ResidualMixerType(mixer),
            init_noise_std=init_noise_std,
            residual_dropout_p=residual_dropout_p,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_eps=sinkhorn_eps,
            stream_balance_loss_weight=stream_balance_loss_weight,
            stream_balance_statistic=StreamUtilisationType(stream_balance_statistic),
            stream_balance_loss_type=StreamBalanceLossType(stream_balance_loss_type),
            dtype=DType.from_pt(dtype),
        )

        self.n_streams = config.n_streams
        self.mixer = config.mixer
        self.init_noise_std = config.init_noise_std
        self.residual_dropout_p = config.residual_dropout_p
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_eps = config.sinkhorn_eps
        self.stream_balance_loss_weight = config.stream_balance_loss_weight
        self.stream_balance_statistic = config.stream_balance_statistic
        self.stream_balance_loss_type = config.stream_balance_loss_type
        self.routing_dtype = dtype
        # Where the last forward's diagnostics land, for `compute_metrics` to read. Plain
        # attributes rather than buffers: they are per-step readings and have no place in a
        # checkpoint, and a buffer would have to be sharded by FSDP for no reason.
        self._balance_loss: Optional[torch.Tensor] = None
        self._utilisation: Optional[torch.Tensor] = None
        # Turned on for one step at a time by `HyperConnectionMonitorCallback`, and off the
        # rest of the time.
        #
        # **THIS IS WHAT LETS AN UNTREATED ARM REPORT ITS COLLAPSE, AND WITHOUT IT THE
        # EXPERIMENT CANNOT BE READ.** The balancing loss computes the utilisation as a side
        # effect, so a treated arm logged a dispersion share and a control arm logged nothing --
        # which is to say the comparison the whole design rests on had a number on one side of
        # it. Under this flag the statistic is computed under `no_grad` and recorded, and
        # nothing is attached to the graph, so a control arm stays numerically a control arm.
        self.diagnostics_enabled: bool = False

        n = self.n_streams
        self.h_pre_logits = nn.Parameter(torch.empty(n, device=init_device, dtype=dtype))
        self.h_post_logits = nn.Parameter(torch.empty(n, device=init_device, dtype=dtype))

        self.h_res_logits: Optional[nn.Parameter]
        if self.mixer == ResidualMixerType.identity:
            self.register_parameter("h_res_logits", None)
            self.register_buffer(
                "_eye", torch.eye(n, device=init_device, dtype=dtype), persistent=False
            )
        elif self.mixer in (ResidualMixerType.unconstrained, ResidualMixerType.sinkhorn):
            self.h_res_logits = nn.Parameter(torch.empty(n, n, device=init_device, dtype=dtype))
        elif self.mixer == ResidualMixerType.birkhoff:
            self.h_res_logits = nn.Parameter(
                torch.empty(math.factorial(n), device=init_device, dtype=dtype)
            )
            self.register_buffer(
                "_perm_mats",
                permutation_matrices(n, device=torch.device(init_device)),
                persistent=False,
            )
        elif self.mixer == ResidualMixerType.kronecker:
            self.n_factors = max(int(math.log2(n)), 0)
            self.h_res_logits = nn.Parameter(
                torch.empty(self.n_factors, 2, device=init_device, dtype=dtype)
            )
        else:
            raise NotImplementedError(self.mixer)

        if init_device != "meta":
            self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """
        Apply the identity-preserving initialisation, then the symmetry-breaking noise.

        ``h_pre`` is set to the uniform average ``1/n`` and renormalised after the noise so that
        it stays a convex combination; ``h_post`` is set to all ones; and every mixer's logits
        are zeroed, which lands ``sinkhorn``, ``birkhoff`` and ``kronecker`` on the uniform
        doubly stochastic matrix and ``unconstrained`` on the same matrix set directly.

        :param generator: An optional RNG for the noise. Note that
            :meth:`~olmo_core.nn.transformer.Transformer.init_weights` calls
            ``reset_parameters()`` with no arguments, so unless a caller passes a generator the
            noise is drawn from the global RNG rather than from the model's ``init_seed``.
            :class:`~olmo_core.nn.transformer.HyperConnectionTransformer` re-seeds it after the
            fact.
        """
        n = self.n_streams

        # `Transformer.init_weights` calls `to_empty()` before this, which reallocates every
        # buffer, non-persistent ones included, so the constant tables have to be rewritten here
        # and not only in `__init__` — otherwise H_res is read out of uninitialized memory.
        if self.mixer == ResidualMixerType.identity:
            self._eye.copy_(torch.eye(n, device=self._eye.device, dtype=self._eye.dtype))
        elif self.mixer == ResidualMixerType.birkhoff:
            self._perm_mats.copy_(
                permutation_matrices(n, device=self._perm_mats.device).to(self._perm_mats.dtype)
            )

        def noise_like(t: torch.Tensor) -> torch.Tensor:
            if self.init_noise_std <= 0.0:
                return torch.zeros_like(t)
            return (
                torch.randn(t.shape, generator=generator, device=t.device, dtype=torch.float32)
                * self.init_noise_std
            ).to(t.dtype)

        # h_pre = sigmoid(logits) should average to 1/n so that reading in from n identical
        # streams reproduces the single stream exactly.
        target = 1.0 / n
        self.h_pre_logits.fill_(math.log(target / (1.0 - target)))
        self.h_pre_logits.add_(noise_like(self.h_pre_logits))
        # The noise moved sum(h_pre) off 1; put it back, which is what keeps the read-in a
        # convex combination rather than a scaled one.
        h_pre = torch.sigmoid(self.h_pre_logits.float())
        h_pre = (h_pre / h_pre.sum()).clamp(1e-6, 1.0 - 1e-6)
        self.h_pre_logits.copy_(torch.log(h_pre / (1.0 - h_pre)).to(self.h_pre_logits.dtype))

        # h_post = 2 * sigmoid(0) = 1.
        self.h_post_logits.zero_()
        self.h_post_logits.add_(noise_like(self.h_post_logits))

        if self.h_res_logits is None:
            return

        if self.mixer == ResidualMixerType.unconstrained:
            # No constraint map, so the uniform doubly stochastic matrix has to be written in
            # directly rather than reached from zero logits.
            self.h_res_logits.fill_(1.0 / n)
        else:
            self.h_res_logits.zero_()
        self.h_res_logits.add_(noise_like(self.h_res_logits))

    def extra_repr(self) -> str:
        return (
            f"n_streams={self.n_streams}, mixer='{self.mixer}', "
            f"init_noise_std={self.init_noise_std}, residual_dropout_p={self.residual_dropout_p}"
        )

    def _drop_residual_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Mask individual residual-mixer logits to ``-inf`` with probability
        ``residual_dropout_p``, guarding against a fully masked group.
        """
        if not self.training or self.residual_dropout_p <= 0.0:
            return logits

        mask = torch.bernoulli(torch.full_like(logits, self.residual_dropout_p)).bool()
        if logits.dim() == 2:
            # An entirely masked row or column makes the Sinkhorn logsumexp -inf and the next
            # subtraction NaN, so any group that lost everything is restored whole.
            mask = mask & ~mask.all(dim=-1, keepdim=True) & ~mask.all(dim=-2, keepdim=True)
        else:
            mask = mask & ~mask.all(dim=-1, keepdim=True)
        return logits.masked_fill(mask, float("-inf"))

    def residual_mixer(self, *, deterministic: bool = False) -> torch.Tensor:
        """
        Build the residual mixer :math:`H_{res}`.

        :param deterministic: Skip the residual-logit dropout even in training mode. **Anything
            that reads this matrix to measure it rather than to use it must pass ``True``.**
            The dropout masks entries to ``-inf`` before the constraint map, so a matrix read
            during training is a draw and not the matrix: measured on an unchanged parameter,
            successive reads differ from the initialisation by 0.44 to 0.86 in relative
            Frobenius norm while the underlying logits have not moved at all. A displacement
            monitor that did not pass this would report dropout as learning.

        Computed in :data:`~HyperConnectionConfig.dtype` (``float32`` by default) whatever dtype
        the parameters and activations are in.

        For ``sinkhorn`` the logit dropout masks entries of the ``n x n`` logit matrix, with the
        row/column guard the Sinkhorn iteration needs. For ``birkhoff`` and ``kronecker`` the
        logits are coefficients rather than matrix entries, so the same Bernoulli mask is
        applied per coefficient group with a guard that at least one coefficient survives. For
        ``identity`` there are no logits and for ``unconstrained`` there is no constraint map to
        apply the mask in front of, so neither uses the dropout.

        :returns: An ``(n, n)`` tensor. Nonnegative with unit row and column sums for every
            mixer except ``unconstrained``.
        """
        n = self.n_streams
        if self.mixer == ResidualMixerType.identity:
            return self._eye.to(self.routing_dtype)

        assert self.h_res_logits is not None
        logits = self.h_res_logits.to(self.routing_dtype)

        if self.mixer == ResidualMixerType.unconstrained:
            return logits

        if not deterministic:
            logits = self._drop_residual_logits(logits)

        if self.mixer == ResidualMixerType.sinkhorn:
            return sinkhorn_log_space(
                logits, n_iters=self.sinkhorn_iters, eps=self.sinkhorn_eps
            ).to(self.routing_dtype)
        elif self.mixer == ResidualMixerType.birkhoff:
            weights = F.softmax(logits.float(), dim=-1)
            perms = self._perm_mats.to(device=weights.device, dtype=weights.dtype)
            return torch.einsum("r,rij->ij", weights, perms).to(self.routing_dtype)
        elif self.mixer == ResidualMixerType.kronecker:
            # Each factor is the convex combination p*I + (1-p)*swap of the two 2x2
            # permutations, which is doubly stochastic, and so is any Kronecker product of them.
            p = F.softmax(logits.float(), dim=-1)[..., 0]
            factors = torch.stack(
                [torch.stack([p, 1.0 - p], dim=-1), torch.stack([1.0 - p, p], dim=-1)], dim=-2
            )
            result = factors[0]
            for k in range(1, self.n_factors):
                result = torch.einsum("ac,bd->abcd", result, factors[k]).reshape(
                    result.shape[-2] * 2, result.shape[-1] * 2
                )
            if self.n_factors == 0:
                result = torch.ones(1, 1, device=logits.device, dtype=torch.float32)
            assert result.shape == (n, n)
            return result.to(self.routing_dtype)
        else:
            raise NotImplementedError(self.mixer)

    def read_in_gate(self) -> torch.Tensor:
        """
        The read-in vector :math:`h_{pre} = \\sigma(\\theta_{pre})`, shape ``(n,)``, computed in
        the routing dtype.

        :returns: The gate.
        """
        return torch.sigmoid(self.h_pre_logits.to(self.routing_dtype))

    def write_out_gate(self) -> torch.Tensor:
        """
        The write-out vector :math:`h_{post} = 2\\sigma(\\theta_{post})`, shape ``(n,)``,
        computed in the routing dtype.

        :returns: The gate.
        """
        return 2.0 * torch.sigmoid(self.h_post_logits.to(self.routing_dtype))

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lift a single-stream hidden state into ``n`` identical streams.

        :param x: A tensor of shape ``(batch_size, seq_len, d_model)``.

        :returns: A tensor of shape ``(batch_size, seq_len, n_streams, d_model)``.
        """
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n_streams, x.shape[-1])

    def read_in(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Mix the streams down to the single branch input :math:`x = h_{pre}^{T} Z`.

        :param streams: A tensor of shape ``(batch_size, seq_len, n_streams, d_model)``.

        :returns: A tensor of shape ``(batch_size, seq_len, d_model)``.
        """
        h_pre = self.read_in_gate().to(streams.dtype)
        return torch.einsum("n,btnd->btd", h_pre, streams)

    def write_out(self, streams: torch.Tensor, branch_out: torch.Tensor) -> torch.Tensor:
        """
        Mix the residual streams and add the branch output to each of them, i.e.
        :math:`H_{res} Z + h_{post} \\otimes f(x)`.

        :param streams: The incoming streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        :param branch_out: The branch output, shape ``(batch_size, seq_len, d_model)``.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.
        """
        if self.mixer == ResidualMixerType.identity:
            mixed = streams
        else:
            h_res = self.residual_mixer().to(streams.dtype)
            mixed = torch.einsum("nm,btmd->btnd", h_res, streams)

        h_post = self.write_out_gate().to(branch_out.dtype)
        out = mixed + torch.einsum("n,btd->btnd", h_post, branch_out)
        return self._maybe_balance_streams(out)

    def _maybe_balance_streams(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Attach the stream-balancing auxiliary loss to the outgoing streams, if it is turned on.

        **The zero-weight path is the one that has to stay free, and it does: this returns
        before touching anything.** No statistic is computed, no tensor is allocated, nothing is
        attached to the autograd graph and no metric is recorded, so a model with the weight at
        its default is numerically the model that existed before this method did.

        Measured on the streams this hyper-connection *writes*, not the ones it reads, for two
        reasons. It is the quantity the next sub-layer sees, so it is what collapse means at
        this depth. And the mixer is between the two, so a loss on the outgoing streams reaches
        ``H_res`` directly rather than only through the next block.

        :param streams: The outgoing streams, shape
            ``(batch_size, seq_len, n_streams, d_model)``.

        :returns: The same tensor, with the auxiliary loss attached when the loss is on.
        """
        if self.stream_balance_loss_weight <= 0.0:
            if self.diagnostics_enabled:
                with torch.no_grad():
                    self._utilisation = stream_utilisation(
                        streams, statistic=self.stream_balance_statistic
                    )
            return streams

        utilisation = stream_utilisation(streams, statistic=self.stream_balance_statistic)
        loss = stream_balance_loss(utilisation, loss_type=self.stream_balance_loss_type)
        # Kept for `compute_metrics`, detached so that holding it cannot extend the graph's
        # lifetime past the backward pass.
        self._balance_loss = loss.detach()
        self._utilisation = utilisation.detach()
        return attach_auxiliary_loss(streams, self.stream_balance_loss_weight * loss)

    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        """
        What the last forward pass measured about this hyper-connection.

        Reported whether or not the balancing loss is on, because the diagnostics are the point
        even in the control arm: a null result on an arm whose streams never collapsed says
        nothing about a treatment for collapse. The loss itself is reported only where there is
        one.

        Mirrors :meth:`~olmo_core.nn.moe.MoEBase.compute_metrics` in shape and in the
        scaled/unscaled pairing, so that a panel written for the MoE router reads these too.

        :param reset: Whether to clear the recorded values afterwards.

        :returns: A mapping from metric name to (value, reduction).
        """
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        if self._utilisation is not None:
            utilisation = self._utilisation
            bins = utilisation.shape[-1]
            # Normalised entropy, so 1 is perfectly spread and 0 is everything in one bin. The
            # complement of the balance loss rather than a second reading of it: the loss is the
            # squared-share form MoE uses and this is the information-theoretic one, and they
            # disagree about which of two unequal vectors is worse.
            entropy = -(utilisation.clamp_min(1e-12).log() * utilisation).sum()
            out["stream usage entropy"] = (
                entropy / math.log(bins),
                ReduceType.mean,
            )
            out["stream usage imbalance"] = (
                utilisation.max() * bins,
                ReduceType.max,
            )
            if self.stream_balance_statistic == StreamUtilisationType.dispersion:
                # The share of residual energy that is NOT common to every stream. This is the
                # quantity the mixer's surviving gradient is proportional to, so it is the one
                # to watch: it going to zero is stream collapse, and it is what the treatment
                # exists to hold up.
                # `utilisation[1:].sum()` and NOT `1 - utilisation[0]`. The two are equal in
                # exact arithmetic and not in float32: at initialisation the share is around
                # 1e-6, so the subtraction cancels every significant digit and logs a clean
                # 0.000000 for a quantity that is small and nonzero. The metric that told a
                # reader the streams were exactly collapsed when they were not is the one whose
                # movement this whole treatment is judged on.
                out["stream dispersion share"] = (utilisation[1:].sum(), ReduceType.mean)

        read_concentration, write_concentration = self.gate_concentration()
        out["read gate concentration"] = (read_concentration, ReduceType.mean)
        out["write gate concentration"] = (write_concentration, ReduceType.mean)

        if self._balance_loss is not None:
            out["stream balance loss"] = (
                self.stream_balance_loss_weight * self._balance_loss,
                ReduceType.mean,
            )
            out["stream balance loss unscaled"] = (self._balance_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self) -> None:
        """
        Forget what the last forward pass measured.
        """
        self._balance_loss = None
        self._utilisation = None

    def stream_norms(self, streams: torch.Tensor) -> torch.Tensor:
        """
        The mean L2 norm of each stream, which is the collapse probe stated plainly.

        :param streams: A tensor of shape ``(batch_size, seq_len, n_streams, d_model)``.

        :returns: A ``float32`` vector of length ``n_streams``.
        """
        return streams.float().norm(dim=-1).mean(dim=(0, 1))

    def gate_concentration(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        How concentrated the read-in and write-out gates are, on ``[0, 1]``.

        Alimaskina et al.'s stream-dominance probe: the gates, not the content. Zero is uniform
        over the streams and one is everything on a single stream, so the two together separate
        "one stream dominates because the gates say so" from "the streams carry the same thing".

        :returns: ``(read_concentration, write_concentration)``.
        """
        results = []
        for gate in (self.read_in_gate(), self.write_out_gate()):
            shares = gate.float() / gate.float().sum().clamp_min(1e-12)
            bins = shares.shape[-1]
            results.append((bins * shares.pow(2).sum() - 1.0) / (bins - 1))
        return results[0], results[1]

    def forward(
        self,
        streams: torch.Tensor,
        branch: Callable[..., torch.Tensor],
        *branch_args,
        **branch_kwargs,
    ) -> torch.Tensor:
        """
        Read in from the streams, run the branch, and write the result back out.

        :param streams: The residual streams, shape
            ``(batch_size, seq_len, n_streams, d_model)``. A three-dimensional
            ``(batch_size, seq_len, d_model)`` input is lifted into ``n`` identical streams
            first, which is what makes a single wrapped sub-layer usable on its own.
        :param branch: The wrapped sub-layer. Called as
            ``branch(x, *branch_args, **branch_kwargs)`` with ``x`` of shape
            ``(batch_size, seq_len, d_model)`` and must return the same shape.
        :param branch_args: Positional arguments forwarded to ``branch``.
        :param branch_kwargs: Keyword arguments forwarded to ``branch``.

        :returns: The updated streams, shape ``(batch_size, seq_len, n_streams, d_model)``.

        :raises ValueError: If ``streams`` is neither 3- nor 4-dimensional, or if its stream
            dimension does not match ``n_streams``.
        """
        if streams.dim() == 3:
            streams = self.expand(streams)
        elif streams.dim() != 4:
            raise ValueError(
                f"expected a 3D (batch, seq, d_model) or 4D (batch, seq, streams, d_model) "
                f"input, got shape {tuple(streams.shape)}"
            )
        if streams.shape[-2] != self.n_streams:
            raise ValueError(
                f"expected {self.n_streams} streams, got {streams.shape[-2]} "
                f"(input shape {tuple(streams.shape)})"
            )

        branch_out = branch(self.read_in(streams), *branch_args, **branch_kwargs)
        return self.write_out(streams, branch_out)


@dataclass
class StreamCollapseConfig(ModuleConfig):
    """
    A config for building a :class:`StreamCollapse`.
    """

    n_streams: int = 4
    """
    The number of residual streams to collapse.
    """

    policy: StreamCollapseType = StreamCollapseType.mean
    """
    How to reduce over the stream dimension.
    """

    dtype: DType = DType.float32
    """
    The dtype the readout weights are computed in.
    """

    def num_params(self) -> int:
        """
        The number of parameters the collapse adds to the model.

        :returns: ``0`` for the mean policy and ``n_streams`` for the learned softmax readout.
        """
        return 0 if self.policy == StreamCollapseType.mean else self.n_streams

    def build(self, *, init_device: str = "cpu") -> "StreamCollapse":
        """
        Build the corresponding :class:`StreamCollapse`.

        :param init_device: The device to allocate parameters on.

        :returns: The module.
        """
        return StreamCollapse(
            n_streams=self.n_streams,
            policy=self.policy,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class StreamCollapse(nn.Module):
    """
    Reduce ``n`` residual streams back to the single hidden state the LM head expects.

    :param n_streams: The number of streams.
    :param policy: The collapse policy.
    :param dtype: The dtype the readout weights are computed in.
    :param init_device: The device to allocate parameters on.
    """

    def __init__(
        self,
        *,
        n_streams: int = 4,
        policy: StreamCollapseType = StreamCollapseType.mean,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.n_streams = n_streams
        self.policy = StreamCollapseType(policy)
        self.readout_dtype = dtype

        self.readout_logits: Optional[nn.Parameter]
        if self.policy == StreamCollapseType.softmax:
            self.readout_logits = nn.Parameter(
                torch.zeros(n_streams, device=init_device, dtype=dtype)
            )
        else:
            self.register_parameter("readout_logits", None)

    @torch.no_grad()
    def reset_parameters(self, generator: Optional[torch.Generator] = None) -> None:
        """
        Zero the readout logits, which makes the learned softmax readout start as the mean.

        :param generator: Unused; accepted so that the signature matches
            :meth:`HyperConnection.reset_parameters`.
        """
        del generator
        if self.readout_logits is not None:
            self.readout_logits.zero_()

    def extra_repr(self) -> str:
        return f"n_streams={self.n_streams}, policy='{self.policy}'"

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        """
        Collapse the stream dimension.

        :param streams: A tensor of shape ``(batch_size, seq_len, n_streams, d_model)``.

        :returns: A tensor of shape ``(batch_size, seq_len, d_model)``.

        :raises ValueError: If ``streams`` does not have the expected stream dimension.
        """
        if streams.dim() != 4 or streams.shape[-2] != self.n_streams:
            raise ValueError(
                f"expected a (batch, seq, {self.n_streams}, d_model) input, got shape "
                f"{tuple(streams.shape)}"
            )
        if self.readout_logits is None:
            return streams.mean(dim=-2)
        weights = F.softmax(self.readout_logits.to(self.readout_dtype), dim=-1)
        return torch.einsum("n,btnd->btd", weights.to(streams.dtype), streams)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
