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
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.config import DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError

from .config import ModuleConfig

__all__ = [
    "ResidualMixerType",
    "StreamCollapseType",
    "HyperConnectionConfig",
    "HyperConnection",
    "StreamCollapseConfig",
    "StreamCollapse",
    "sinkhorn_log_space",
    "permutation_matrices",
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
            dtype=DType.from_pt(dtype),
        )

        self.n_streams = config.n_streams
        self.mixer = config.mixer
        self.init_noise_std = config.init_noise_std
        self.residual_dropout_p = config.residual_dropout_p
        self.sinkhorn_iters = config.sinkhorn_iters
        self.sinkhorn_eps = config.sinkhorn_eps
        self.routing_dtype = dtype

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

    def residual_mixer(self) -> torch.Tensor:
        """
        Build the residual mixer :math:`H_{res}`.

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
        return mixed + torch.einsum("n,btd->btnd", h_post, branch_out)

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
