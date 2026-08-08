from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from ..config import StrEnum
from ..distributed.utils import distribute_like, get_local_tensor
from .config import ModuleConfig

__all__ = [
    "ResidualStream",
    "HyperConnectionMode",
    "HyperConnectionConfig",
    "HyperConnectionStream",
    "MHARConfig",
    "MHARRoutingSite",
    "expand_residual_lanes",
    "reduce_residual_lanes",
    "output_init_scale",
    "sinkhorn_knopp",
    "HC_STATIC_PARAM_GLOB",
    "HC_DYNAMIC_PARAM_GLOB",
]


#: Glob matching the static hyper-connection parameters (``B``, ``A_m``, ``A_r``), which
#: ByteDance excludes from weight decay.
HC_STATIC_PARAM_GLOB = "*.hc_static_*"

#: Glob matching the dynamic hyper-connection parameters, which do take weight decay.
HC_DYNAMIC_PARAM_GLOB = "*.hc_dynamic_*"


class ResidualStream(nn.Module):
    """
    A parameter-free module that just handles a residual stream connection, like those in a transformer
    block. The benefit of using this module instead of a direct add operation is that the flexible
    to configure hooks for logging or other purposes, like with the
    :class:`olmo_core.train.callbacks.GAPMonitorCallback`.
    """

    def __init__(self, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.add(residual, self.dropout(x), alpha=self.alpha)


class HyperConnectionMode(StrEnum):
    """
    Which half of the hyper-connection mechanism is enabled.
    """

    full = "full"
    """
    The faithful method: the sublayer input is a learned (and, for DHC, input-dependent)
    weighted sum of the lanes, via ``A_m``.
    """

    output = "output"
    """
    Output-side mixing only. The sublayer input is the unweighted mean of the lanes, so ``A_m``
    does not exist. This is the variant a shared residual interface forces on you when it hands
    the residual module the sublayer *output* and never the sublayer input, which is the reason
    Tencent (arXiv 2605.20798) gave for their reimplementation being incomplete.
    """


@dataclass
class HyperConnectionConfig(ModuleConfig):
    """
    A config for building :class:`HyperConnectionStream`, and for the model-level choices that
    widening the residual stream forces: how the lanes are collapsed before the unembedding, and
    how the output modules are initialized to compensate.

    See :class:`HyperConnectionStream` for the per-stream fields.
    """

    n_lanes: int = 4
    """
    The expansion rate ``n``. ByteDance found 4 best, 8 barely better, and 1 worse than baseline.
    """

    mode: HyperConnectionMode = HyperConnectionMode.full
    """
    Whether the input side is mixed as well as the output side.
    """

    dynamic: bool = True
    """
    DHC rather than SHC. DHC is the paper's default and beats SHC at ``n=4``.
    """

    tanh: bool = True
    """
    Squash the dynamic term.
    """

    dynamic_scale_init: float = 0.01
    """
    Initial value of ``s_beta`` and ``s_alpha``.
    """

    doubly_stochastic: bool = False
    """
    Constrain ``A_r`` to the Birkhoff polytope (mHC).
    """

    sinkhorn_iters: int = 8
    """
    Sweeps used by the Sinkhorn-Knopp projection.
    """

    birkhoff_init_logit: float = 8.0
    """
    Diagonal of ``A_r`` at initialization when ``doubly_stochastic`` is set.
    """

    output_init_exponent: float = 0.5
    """
    Output modules are initialized with their standard deviation multiplied by
    ``n_lanes ** -output_init_exponent``. See :func:`output_init_scale`. Set to ``0.0`` to turn
    the correction off, which is the arm that tests whether it is load-bearing.
    """

    average_lanes: bool = False
    """
    Average rather than sum when collapsing the lanes before the final norm.
    """

    eps: float = 1e-6
    """
    Epsilon for the norm on the lanes.
    """

    def build(
        self,
        *,
        d_model: int,
        block_idx: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_device: str = "cpu",
    ) -> "HyperConnectionStream":
        return HyperConnectionStream(
            d_model=d_model,
            block_idx=block_idx,
            n_lanes=self.n_lanes,
            mode=self.mode,
            dynamic=self.dynamic,
            tanh=self.tanh,
            dynamic_scale_init=self.dynamic_scale_init,
            doubly_stochastic=self.doubly_stochastic,
            sinkhorn_iters=self.sinkhorn_iters,
            birkhoff_init_logit=self.birkhoff_init_logit,
            alpha=alpha,
            dropout=dropout,
            eps=self.eps,
            init_device=init_device,
        )

    def num_params(self, d_model: int) -> int:
        """
        Parameters added per block, i.e. twice :meth:`HyperConnectionStream.expected_num_params`
        because a block has one stream for attention and one for the feed-forward.
        """
        return 2 * HyperConnectionStream.expected_num_params(
            d_model=d_model, n_lanes=self.n_lanes, mode=self.mode, dynamic=self.dynamic
        )

    def optim_group_overrides(self, weight_decay: float = 0.0) -> list:
        """
        The parameter-group split ByteDance describe: "the static component does not utilize
        weight decay, whereas the dynamic component does".

        The dynamic group is named explicitly rather than left to fall through to the default
        group, so that a run whose parameter names drift will fail loudly at
        :meth:`~olmo_core.optim.OptimConfig.build_groups` instead of quietly decaying ``A_r``.

        :param weight_decay: Weight decay for the dynamic component. Should match the value the
            rest of the model is trained with.

        :returns: Overrides to extend
            :data:`~olmo_core.optim.OptimConfig.group_overrides` with.
        """
        from ..optim import OptimGroupOverride

        overrides = [OptimGroupOverride(params=[HC_STATIC_PARAM_GLOB], opts=dict(weight_decay=0.0))]
        if self.dynamic:
            overrides.append(
                OptimGroupOverride(
                    params=[HC_DYNAMIC_PARAM_GLOB], opts=dict(weight_decay=weight_decay)
                )
            )
        return overrides


@dataclass
class MHARConfig(ModuleConfig):
    """
    A config for :class:`MHARRoutingSite`, multi-head attention residuals
    (`arXiv 2607.27230 <https://arxiv.org/abs/2607.27230>`_).

    :param n_route_heads: ``H``. The paper's optimum is flat over 4-8, and their own 1B numbers
        put ``H=4`` and ``H=8`` 0.003 nats apart -- inside anyone's noise -- so a sweep should
        step further than that to learn anything.
    """

    n_route_heads: int = 8
    eps: float = 1e-6
    zero_init_query: bool = True
    """
    Zero-initialized queries make every depth softmax start as a uniform average over sources.
    Load-bearing rather than tidy: a random query makes the softmax arbitrarily peaked at step
    zero, and the paper's own web-corpus tables were measured before they fixed this.
    """

    def build(self, *, d_model: int, init_device: str = "cpu") -> "MHARRoutingSite":
        return MHARRoutingSite(
            d_model=d_model,
            n_route_heads=self.n_route_heads,
            eps=self.eps,
            zero_init_query=self.zero_init_query,
            init_device=init_device,
        )

    def num_params(self, d_model: int) -> int:
        """
        Per routing site: one query and one key-norm gain, both of width ``d_model``.

        There are ``2L + 1`` sites in a model -- two per block plus one after the stack -- so a
        16-layer model at ``d_model`` 1024 adds 67,584 parameters, about 0.014%.

        This is where the widely repeated "zero added parameters" claim needs care. It is true
        of ``H`` specifically: the ``H`` queries are a reshape of the same ``d_model`` numbers,
        so ``H=8`` is iso-parameter, iso-FLOP and iso-wall-clock with ``H=1``. It is not true
        against a plain transformer, which has no routing sites at all.
        """
        return 2 * d_model


class MHARRoutingSite(nn.Module):
    """
    One depth-routing site: a softmax over every preceding sublayer output, per head.

    Replaces the additive residual rather than augmenting it. Where an ordinary block reads
    ``x`` and writes ``x + T(norm(x))``, a routed block reads a learned mixture of *all* sources
    produced so far and appends its output as a new source. Eq. 3 of the paper, with ``s_i``
    the sources and coordinates split into ``H`` contiguous heads of width ``d/H``:

    .. code-block::

        alpha_i^(h) = softmax_i( q_h . RMSNorm(s_i)_[h] )
        out_[h]     = sum_i alpha_i^(h) s_i,[h]

    Three details decide whether this is the published mechanism or something near it. The
    normalization is over the **full** ``d_model`` row and then sliced, not per head -- the
    authors' reference and their fused kernel agree on this against their own paper figure, and
    per-head normalization would strip the magnitude signal the logits read. The **values are
    raw**, only the keys are normalized. And there is no ``1/sqrt(d)`` on the logits.

    Sources are taken as a list and never stacked. Stacking is not a style preference: at
    ``d_model`` 1024, 16 layers, sequence 4096 and batch 8, the 33 sources are 2.1 GiB held once
    each, and 35 GiB if every site materializes its own stacked copy for autograd. That
    quadratic is what makes the reference implementation run out of memory above 1B.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_route_heads: int = 8,
        eps: float = 1e-6,
        zero_init_query: bool = True,
        init_device: str = "cpu",
    ):
        super().__init__()
        if d_model % n_route_heads != 0:
            raise ValueError(f"d_model {d_model} is not divisible by n_route_heads {n_route_heads}")
        self.d_model = d_model
        self.n_route_heads = n_route_heads
        self.head_dim = d_model // n_route_heads
        self.eps = eps
        self.zero_init_query = zero_init_query

        self.mhar_query = nn.Parameter(torch.empty(d_model, device=init_device))
        self.mhar_key_gain = nn.Parameter(torch.empty(d_model, device=init_device))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.zero_init_query:
                nn.init.zeros_(self.mhar_query)
            else:
                nn.init.normal_(self.mhar_query, mean=0.0, std=0.02)
            nn.init.ones_(self.mhar_key_gain)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_route_heads={self.n_route_heads}"

    def _logits(self, source: torch.Tensor) -> torch.Tensor:
        """Per-head scores for one source, shape ``(batch_size, seq_len, n_route_heads)``."""
        variance = source.float().pow(2).mean(dim=-1, keepdim=True)
        key = (source.float() * torch.rsqrt(variance + self.eps)).to(source.dtype)
        key = key * self.mhar_key_gain
        shape = (*key.shape[:-1], self.n_route_heads, self.head_dim)
        return torch.einsum(
            "hk,...hk->...h",
            self.mhar_query.view(self.n_route_heads, self.head_dim),
            key.view(shape),
        )

    def forward(self, sources: List[torch.Tensor]) -> torch.Tensor:
        """
        Route the sources into the vector the next sublayer sees.

        :param sources: Every sublayer output so far, plus the embeddings, each of shape
            ``(batch_size, seq_len, d_model)``. Never stacked; see the class docstring.

        :returns: The routed mixture, of the same shape as one source.
        """
        if not sources:
            raise ValueError("MHAR needs at least one source to route")

        # Two passes over the list. The first is over (batch, seq, heads) scores, which are
        # d_model/head_dim times smaller than a source, so the only full-size tensors alive at
        # once are the sources themselves -- which the caller already holds.
        weights = torch.softmax(
            torch.stack([self._logits(s) for s in sources], dim=0).float(), dim=0
        )

        routed = torch.zeros_like(sources[0])
        flat = (*routed.shape[:-1], self.n_route_heads, self.head_dim)
        view = routed.view(flat)
        for i, source in enumerate(sources):
            view = view + weights[i].unsqueeze(-1).to(source.dtype) * source.view(flat)
        return view.reshape(routed.shape)


def _set_full(param: torch.Tensor, value: torch.Tensor):
    """
    Write a full tensor into a parameter that may be sharded.

    ``ones_`` and ``zeros_`` are safe on a :class:`~torch.distributed.tensor.DTensor` because
    they are the same everywhere, but the identity and the one-hot read are position-dependent
    and a rank holding rows 2 and 3 would otherwise get the wrong ones. Same approach as
    ``olmo_core.nn.transformer.init._apply_init``, written out here rather than imported to
    avoid a cycle -- the transformer config imports this module.
    """
    if isinstance(param, DTensor):
        get_local_tensor(param).copy_(get_local_tensor(distribute_like(param, value)))
    else:
        param.copy_(value)


def expand_residual_lanes(h: torch.Tensor, n_lanes: int) -> torch.Tensor:
    """
    Replicate a hidden state into ``n_lanes`` hyper hidden vectors, giving ``H^0`` of eq. 1
    in `Hyper-Connections <https://arxiv.org/abs/2409.19606>`_.

    :param h: Hidden states of shape ``(batch_size, seq_len, d_model)``.
    :param n_lanes: The expansion rate ``n``.

    :returns: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.
    """
    return h.unsqueeze(-2).expand(*h.shape[:-1], n_lanes, h.shape[-1]).contiguous()


def reduce_residual_lanes(hidden: torch.Tensor, *, average: bool = False) -> torch.Tensor:
    """
    Collapse hyper hidden vectors back to a single hidden state, which the paper does by
    summing row-wise before the final norm and unembedding.

    :param hidden: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.
    :param average: Average instead of summing. The final norm is scale-invariant so this does
        not change the forward pass, but it does change the scale the lanes are carried at.

    :returns: Hidden states of shape ``(batch_size, seq_len, d_model)``.
    """
    return hidden.mean(dim=-2) if average else hidden.sum(dim=-2)


def sinkhorn_knopp(logits: torch.Tensor, *, num_iters: int = 8) -> torch.Tensor:
    """
    Project a matrix of logits onto the Birkhoff polytope of doubly stochastic matrices by
    alternating row and column normalization in the log domain, as mHC
    (`arXiv 2512.24880 <https://arxiv.org/abs/2512.24880>`_) does.

    The result is exactly column-stochastic and approximately row-stochastic; the row residual
    shrinks geometrically in ``num_iters``. Doubly stochastic matrices have spectral radius
    exactly 1 and are closed under multiplication, which is the property mHC relies on to keep
    the composite mapping across depth well conditioned.

    :param logits: Logits of shape ``(..., n, n)``.
    :param num_iters: Number of alternating normalization sweeps.

    :returns: A nonnegative matrix of the same shape whose columns sum to 1.
    """
    for _ in range(num_iters):
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logits = logits - torch.logsumexp(logits, dim=-2, keepdim=True)
    return logits.exp()


class HyperConnectionStream(nn.Module):
    """
    `Hyper-connections <https://arxiv.org/abs/2409.19606>`_ (ByteDance Seed, ICLR 2025) in place
    of a residual connection, with the manifold-constrained variant of
    `mHC <https://arxiv.org/abs/2512.24880>`_ available behind ``doubly_stochastic``.

    The residual stream is widened into ``n_lanes`` hyper hidden vectors ``H``. Each sublayer
    reads one vector out of them, and its output is written back across all of them while the
    lanes are also mixed with each other. Following eqs. 2-5 of the paper, with ``T`` the
    sublayer:

    .. code-block::

        h_0 = A_m^T H                     # read   (width connection)
        H'  = A_r^T H + B^T T(h_0)^T      # write  (width + depth connections)

    Unlike :class:`ResidualStream` this cannot be a single call, because the read happens before
    the sublayer runs and the write happens after. Use :meth:`read` and then :meth:`write`; the
    coefficients are computed once, in :meth:`read`, and handed to :meth:`write`.

    For dynamic hyper-connections (DHC, the paper's default and best variant) every coefficient
    is the sum of a static parameter and a token-dependent term predicted from the normalized
    lanes, per eqs. 10-13.

    At initialization this is exactly equivalent to the residual stack it replaces: ``A_m`` reads
    lane ``block_idx % n_lanes``, ``B`` writes to every lane, ``A_r`` is the identity, and the
    dynamic projections are zero, so every lane holds an identical copy of the ordinary residual
    stream. :func:`reduce_residual_lanes` then sums to ``n_lanes`` times the ordinary hidden
    state, which the final norm is invariant to.

    :param d_model: The model dimensionality.
    :param n_lanes: The expansion rate ``n``. ``n_lanes=1`` is the paper's seesaw control, which
        they found does *not* beat the baseline.
    :param block_idx: Position of the block in the model, which sets which lane is read at
        initialization.
    :param mode: See :class:`HyperConnectionMode`.
    :param dynamic: Use DHC rather than static SHC.
    :param tanh: Squash the dynamic term. The paper's ``DHC x8 w/o tanh`` was their best single
        number, so this is worth having as a knob.
    :param dynamic_scale_init: Initial value of the learnable scales ``s_beta`` and ``s_alpha``.
    :param doubly_stochastic: Project ``A_r`` onto the Birkhoff polytope (mHC).
    :param sinkhorn_iters: Sweeps used by that projection.
    :param birkhoff_init_logit: Diagonal of ``A_r`` at initialization when ``doubly_stochastic``
        is set. ``A_r`` is then read as logits, so the identity has to be expressed as a large
        diagonal for the projection to return something close to the identity.
    :param alpha: A scaling factor applied to the sublayer output, matching
        :class:`ResidualStream`.
    :param dropout: Dropout probability applied to the sublayer output.
    :param eps: Epsilon for the norm on the lanes.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_lanes: int,
        block_idx: int,
        mode: HyperConnectionMode = HyperConnectionMode.full,
        dynamic: bool = True,
        tanh: bool = True,
        dynamic_scale_init: float = 0.01,
        doubly_stochastic: bool = False,
        sinkhorn_iters: int = 8,
        birkhoff_init_logit: float = 8.0,
        alpha: float = 1.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        init_device: str = "cpu",
    ):
        super().__init__()
        if n_lanes < 1:
            raise ValueError(f"n_lanes must be at least 1, got {n_lanes}")

        self.d_model = d_model
        self.n_lanes = n_lanes
        self.block_idx = block_idx
        self.mode = HyperConnectionMode(mode)
        self.dynamic = dynamic
        self.tanh = tanh
        self.dynamic_scale_init = dynamic_scale_init
        self.doubly_stochastic = doubly_stochastic
        self.sinkhorn_iters = sinkhorn_iters
        self.birkhoff_init_logit = birkhoff_init_logit
        self.alpha = alpha
        self.eps = eps
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Static component (eq. 1). Excluded from weight decay -- see HC_STATIC_PARAM_GLOB.
        self.hc_static_beta = nn.Parameter(torch.empty(n_lanes, device=init_device))
        self.hc_static_alpha_r = nn.Parameter(torch.empty(n_lanes, n_lanes, device=init_device))
        if self.mode == HyperConnectionMode.full:
            self.hc_static_alpha_m = nn.Parameter(torch.empty(n_lanes, device=init_device))
        else:
            self.register_parameter("hc_static_alpha_m", None)

        # Dynamic component (eqs. 11-13). One scale for beta and one shared by A_m and A_r,
        # which is what the paper's parameter count in eq. 25 implies.
        if dynamic:
            self.hc_dynamic_w_beta = nn.Parameter(torch.empty(d_model, device=init_device))
            self.hc_dynamic_w_r = nn.Parameter(torch.empty(d_model, n_lanes, device=init_device))
            # Shape (1,) rather than a true scalar: FSDP shards on dim 0 and cannot shard a
            # 0-dim parameter at all. One element either way, which is what eq. 24 counts.
            self.hc_dynamic_scale_beta = nn.Parameter(torch.empty(1, device=init_device))
            self.hc_dynamic_scale_alpha = nn.Parameter(torch.empty(1, device=init_device))
            if self.mode == HyperConnectionMode.full:
                self.hc_dynamic_w_m = nn.Parameter(torch.empty(d_model, device=init_device))
            else:
                self.register_parameter("hc_dynamic_w_m", None)
        else:
            for name in (
                "hc_dynamic_w_beta",
                "hc_dynamic_w_m",
                "hc_dynamic_w_r",
                "hc_dynamic_scale_beta",
                "hc_dynamic_scale_alpha",
            ):
                self.register_parameter(name, None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Apply the initialization of eq. 14, under which the whole mechanism collapses to the
        residual stack it replaces.
        """
        with torch.no_grad():
            nn.init.ones_(self.hc_static_beta)

            identity = torch.eye(
                self.n_lanes,
                device=self.hc_static_alpha_r.device,
                dtype=self.hc_static_alpha_r.dtype,
            )
            # In mHC, A_r is read as logits, so a plain identity would come back out of the
            # Sinkhorn projection as a near-uniform matrix.
            _set_full(
                self.hc_static_alpha_r,
                identity * self.birkhoff_init_logit if self.doubly_stochastic else identity,
            )

            if self.hc_static_alpha_m is not None:
                read = torch.zeros_like(identity[0])
                read[self.block_idx % self.n_lanes] = 1.0
                _set_full(self.hc_static_alpha_m, read)

            if self.dynamic:
                nn.init.zeros_(self.hc_dynamic_w_beta)
                nn.init.zeros_(self.hc_dynamic_w_r)
                nn.init.constant_(self.hc_dynamic_scale_beta, self.dynamic_scale_init)
                nn.init.constant_(self.hc_dynamic_scale_alpha, self.dynamic_scale_init)
                if self.hc_dynamic_w_m is not None:
                    nn.init.zeros_(self.hc_dynamic_w_m)

    def extra_repr(self) -> str:
        return (
            f"n_lanes={self.n_lanes}, mode={self.mode}, dynamic={self.dynamic}, "
            f"doubly_stochastic={self.doubly_stochastic}"
        )

    def _normalize_lanes(self, hidden: torch.Tensor) -> torch.Tensor:
        # Parameter-free RMS norm, eq. 10. The paper notes |theta_norm| = 0 for OLMo.
        variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
        return (hidden.float() * torch.rsqrt(variance + self.eps)).to(hidden.dtype)

    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x) if self.tanh else x

    def coefficients(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ``(A_m, B, A_r)`` for this token, per eqs. 11-13.

        The coefficient tensors are tiny -- ``n`` and ``n x n`` per token -- so they are
        assembled in float32 and cast back, which costs nothing and keeps the lane mixing out of
        bfloat16. Tencent's 3B divergence showed up as multi-lane drift, so this is the one place
        worth spending precision.

        :param hidden: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.

        :returns: ``alpha_m`` of shape ``(..., n_lanes)``, ``beta`` of shape ``(..., n_lanes)``,
            and ``alpha_r`` of shape ``(..., n_lanes, n_lanes)``, where ``alpha_r[..., i, j]``
            is the weight from lane ``i`` into lane ``j``.
        """
        dtype = hidden.dtype
        beta = self.hc_static_beta.float()
        alpha_r = self.hc_static_alpha_r.float()
        alpha_m = None if self.hc_static_alpha_m is None else self.hc_static_alpha_m.float()

        if self.dynamic:
            normed = self._normalize_lanes(hidden)
            beta = beta + self.hc_dynamic_scale_beta.float() * self._squash(
                torch.matmul(normed, self.hc_dynamic_w_beta).float()
            )
            alpha_r = alpha_r + self.hc_dynamic_scale_alpha.float() * self._squash(
                torch.matmul(normed, self.hc_dynamic_w_r).float()
            )
            if alpha_m is not None:
                assert self.hc_dynamic_w_m is not None
                alpha_m = alpha_m + self.hc_dynamic_scale_alpha.float() * self._squash(
                    torch.matmul(normed, self.hc_dynamic_w_m).float()
                )

        if self.doubly_stochastic:
            alpha_r = sinkhorn_knopp(alpha_r, num_iters=self.sinkhorn_iters)

        if alpha_m is None:
            # Output-side-only: a fixed uniform read. The mean rather than the sum, because the
            # reordered-norm block hands the sublayer its input unnormalized, so a sum would
            # feed it n times the baseline scale and the arm would stop being a control.
            alpha_m = torch.full(
                (self.n_lanes,), 1.0 / self.n_lanes, device=hidden.device, dtype=torch.float32
            ).expand(*hidden.shape[:-2], self.n_lanes)

        return alpha_m.to(dtype), beta.to(dtype), alpha_r.to(dtype)

    def read(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collapse the lanes into the vector the sublayer sees, ``h_0 = A_m^T H`` (eq. 3).

        :param hidden: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.

        :returns: The sublayer input of shape ``(batch_size, seq_len, d_model)``, plus ``beta``
            and ``alpha_r`` to hand back to :meth:`write` so they are only computed once.
        """
        alpha_m, beta, alpha_r = self.coefficients(hidden)
        sublayer_input = torch.einsum("...i,...id->...d", alpha_m, hidden)
        return sublayer_input, beta, alpha_r

    def write(
        self,
        hidden: torch.Tensor,
        sublayer_output: torch.Tensor,
        beta: torch.Tensor,
        alpha_r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mix the lanes and add the sublayer output back across them,
        ``H' = A_r^T H + B^T T(h_0)^T`` (eq. 5).

        :param hidden: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.
        :param sublayer_output: The sublayer output of shape ``(batch_size, seq_len, d_model)``.
        :param beta: ``B``, from :meth:`read`.
        :param alpha_r: ``A_r``, from :meth:`read`.

        :returns: Lanes of shape ``(batch_size, seq_len, n_lanes, d_model)``.
        """
        mixed = torch.einsum("...ij,...id->...jd", alpha_r, hidden)
        scattered = torch.einsum(
            "...j,...d->...jd", beta, self.dropout(sublayer_output) * self.alpha
        )
        return mixed + scattered

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del residual, x
        raise NotImplementedError(
            "HyperConnectionStream splits the residual connection into read() before the "
            "sublayer and write() after it, so it cannot stand in for a ResidualStream call. "
            "Use HyperConnectionTransformerBlock."
        )

    def num_params(self) -> int:
        """
        The parameter count of eq. 25, which for OLMo-1B-DHC x4 gives the paper's 394,048 once
        doubled per block and multiplied by the layer count.
        """
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def expected_num_params(
        *,
        d_model: int,
        n_lanes: int,
        mode: HyperConnectionMode = HyperConnectionMode.full,
        dynamic: bool = True,
    ) -> int:
        """
        Eq. 25 in closed form, for config-time parameter accounting.
        """
        count = n_lanes + n_lanes * n_lanes  # B, A_r
        if dynamic:
            count += d_model + d_model * n_lanes + 2  # W_beta, W_r, s_beta, s_alpha
        if HyperConnectionMode(mode) == HyperConnectionMode.full:
            count += n_lanes  # A_m
            if dynamic:
                count += d_model  # W_m
        return count


def output_init_scale(n_lanes: int, exponent: float) -> float:
    """
    The factor to multiply an output module's initialized weights by when the residual stream
    has been widened to ``n_lanes``.

    ByteDance keep "the standard deviation of the output ... consistent with the original" by
    scaling the second linear of the feed-forward network and the attention output projector
    "by a factor of sqrt(n)". Hyper-connections make the pre-unembedding hidden state *larger*,
    since the lanes are summed, so the factor has to be a divisor: ``n ** -exponent``.

    ``exponent=0.5`` is the paper's sqrt(n). ``exponent=1.0`` is what exactly cancels the sum at
    initialization, where the lanes are still identical copies. ``exponent=0.0`` turns the
    correction off.
    """
    return float(n_lanes) ** (-exponent) if n_lanes > 1 else 1.0
