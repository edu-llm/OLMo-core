import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

from .mamba3_ssd_api import dispatch_mamba3_ssd

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "Mamba3Mixer",
    "Mamba3MixerConfig",
    "DEFAULT_D_STATE",
    "admissible_block_sizes",
    "kernel_padded_width",
]

#: The default SSM state size ``N``, defined once so the mixer, its config, and every preset in
#: :mod:`olmo_core.nn.mamba3.config` cannot drift apart.
#:
#: 192 rather than the more obvious 128 because it is the smallest value that admits ``b`` in
#: ``{2, 3, 4}`` (:func:`admissible_block_sizes`), which is what lets a TC^0 baseline and an
#: NC^1 arm share one state size -- so ``rotation_block_size`` is genuinely the only field that
#: differs between them. 128 admits only ``{2, 4, 8}`` and would force an NC^1 arm to change a
#: second field or settle for ``b=4``. It also happens to sit closer to the OLMo-3-370M
#: reference parameter count (1.70% against 2.23%), because the Mamba arm is below the
#: reference and widening the state closes the gap.
#:
#: The cost is that the official kernel zero-pads it to 256
#: (:func:`kernel_padded_width`); no power of two is divisible by 3, so this is unavoidable for
#: any ``b=3`` configuration rather than a property of this particular number.
DEFAULT_D_STATE = 192

#: Largest ``b`` :func:`admissible_block_sizes` will report. ``A_5 subset SO(3)`` already gives
#: NC^1-hardness, so nothing above this is load-bearing; the cap just keeps the answer readable.
_MAX_REPORTED_BLOCK_SIZE = 8


def admissible_block_sizes(
    d_state: int, *, max_block_size: int = _MAX_REPORTED_BLOCK_SIZE
) -> tuple[int, ...]:
    """
    The rotation block sizes a given ``d_state`` can actually express.

    The positive form of the constraint :func:`_validate_dims` enforces negatively. Stating it
    once as a function matters because the *choice* of ``d_state`` is made independently in
    several places -- every preset in :mod:`~olmo_core.nn.mamba3.config` and the ``A_5`` harness
    -- and each one has to know which ``b`` sweep its choice permits. Prior to this helper that
    fact was restated in prose in a dozen docstrings and encoded once as a bare ``48``.

    The practical trap it exists to make visible: ``128`` admits only ``{2, 4}``, so an NC^1 arm
    at ``b=3`` needs a different ``d_state`` (``192`` admits ``2``, ``3``, ``4`` and ``6``).

    :param d_state: The SSM state size ``N``.
    :param max_block_size: Largest ``b`` to report.

    :returns: Ascending block sizes ``b >= 2`` that divide ``d_state``.
    """
    if d_state < 2:
        return ()
    return tuple(b for b in range(2, max_block_size + 1) if d_state % b == 0)


def kernel_padded_width(dim: int, *, min_width: int = 16) -> int:
    """
    The width the official ``mamba-ssm`` kernel will actually run ``dim`` at.

    ``mamba3_siso_combined`` needs a power-of-two head dimension for TMA, so
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_fast.mamba3_ssd_fast` zero-pads up to one. The padding
    is numerically exact -- a zero column of ``B`` never enters the state -- but it is *wasted
    work*, and it collides with :func:`admissible_block_sizes`: no power of two is divisible by
    3, so every ``b=3`` configuration pays some. ``d_state=192`` runs at 256, a quarter of the
    ``Q``/``K`` lanes carrying zeros.

    Only the official/fast path pads; the chunked and reference paths use ``dim`` as given.

    :param dim: The logical width (``d_state`` or ``head_dim``).
    :param min_width: Floor imposed by the kernel's ``tl.dot`` contraction.
    """
    out = min_width
    while out < dim:
        out *= 2
    return out


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square normalization over the last dimension, in float32."""
    orig_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(orig_dtype)


def _validate_dims(
    *,
    n_heads: int,
    d_state: int,
    n_groups: int,
    mimo_rank: int,
    rotation_block_size: int,
    a_log_init_max: float,
) -> None:
    """
    Validate the mixer's shape-determining options.

    Shared by :meth:`Mamba3Mixer.__init__` and :meth:`Mamba3MixerConfig.num_params` so that
    sizing a config fails exactly when building it would: ``num_params`` is read long before any
    module exists (it is what :meth:`Mamba3Config.build` logs and what sizing scripts print), and
    its integer arithmetic would otherwise report a plausible number for a config that cannot be
    constructed.

    :raises ValueError: If any option is out of range or the dimensions are incompatible.
    """
    if rotation_block_size < 2:
        raise ValueError(f"rotation_block_size must be >= 2, got {rotation_block_size}")
    if d_state < rotation_block_size:
        # Divisibility alone would wave ``d_state=0`` through (``0 % b == 0``), leaving zero
        # rotation blocks and zero-width B/C -- a mixer that returns exactly zero for every
        # input rather than failing.
        raise ValueError(
            f"d_state ({d_state}) must be at least rotation_block_size "
            f"({rotation_block_size}); a smaller state leaves no rotation blocks at all"
        )
    if d_state % rotation_block_size != 0:
        raise ValueError(
            f"d_state ({d_state}) must be divisible by rotation_block_size "
            f"({rotation_block_size}) for the Mamba-3 rotation"
        )
    if a_log_init_max <= 0:
        raise ValueError(f"a_log_init_max must be > 0, got {a_log_init_max}")
    if n_groups < 1:
        raise ValueError(f"n_groups must be >= 1, got {n_groups}")
    if n_heads < 1:
        raise ValueError(f"n_heads must be >= 1, got {n_heads}")
    if n_heads % n_groups != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})")
    if mimo_rank < 1:
        raise ValueError(f"mimo_rank must be >= 1, got {mimo_rank}")


class Mamba3Mixer(SequenceMixer):
    """
    A Mamba-3 state-space sequence mixer, implementing the three innovations from
    `Mamba-3: Improved Sequence Modeling using State Space Principles
    <https://arxiv.org/abs/2603.15569>`_:

    1. Exponential-trapezoidal discretization (§3.1) - a 2nd-order recurrence whose implicit
       width-2 convolution on the state-input, together with ``B``/``C`` bias, removes the
       short causal convolution used by Mamba-2 / GatedDeltaNet.
    2. Complex-valued state via the data-dependent RoPE trick (§3.2) - a cumulative rotation
       applied to the ``B`` and ``C`` projections (the SSD analogs of attention's ``K``/``Q``).
    3. MIMO (§3.3) - rank-``R`` input/output projections (``mimo_rank``); ``R == 1`` is SISO.

    This module drops into a :class:`~olmo_core.nn.transformer.block.TransformerBlock` in place
    of attention (it is a :class:`~olmo_core.nn.attention.base.SequenceMixer`), which is how the
    1:3 attention-to-Mamba-3 hybrid is assembled.

    :param d_model: The model hidden size.
    :param n_heads: The number of SSM heads.
    :param head_dim: Per-head value dimension. Defaults to ``d_model // n_heads``.
    :param d_state: The SSM state dimension ``N`` (must be a positive multiple of
        ``rotation_block_size``).
    :param n_groups: Number of ``(B, C)`` groups shared across heads.
    :param mimo_rank: The MIMO rank ``R`` (``1`` == SISO).
    :param rotation_block_size: Size ``b`` of the orthogonal transition blocks. ``2`` is the
        paper's complex diagonal and the default. ``b >= 3`` makes the transition monoid
        non-solvable (``A_5 subset SO(3)``), which lifts the layer out of TC^0 and is what
        state-tracking tasks need; it costs a prefix product over ``SO(b)`` in place of a
        cumulative sum of angles. Must be one of :func:`admissible_block_sizes` for this
        ``d_state``. ``b=3`` is the cheapest non-solvable choice and the only one with a closed
        form (:func:`~olmo_core.nn.mamba3.mamba3_ssd_fast.fast_block_rotations`); ``b=4`` adds no
        hardness over it, falls back to ``matrix_exp``, and has been observed to be sensitive to
        learning rate and seed on the ``A_5`` task.
    :param norm_eps: Epsilon for the internal RMS norms.
    :param bc_norm: Whether to apply BCNorm (QK-norm analog) to ``B`` and ``C``.
    :param bc_bias: Whether the ``B``/``C`` projections use a bias term.
    :param a_log_init_max: Upper bound of the ``A_log`` init distribution. The default of 16
        gives ``alpha ~ 0.92`` at init, so a signal injected at position 0 has decayed to
        ``~1e-9`` by position 256 -- below fp32 resolution, which makes long-horizon state
        tracking near-untrainable regardless of ``rotation_block_size``. Lower it (``~0.1``)
        for state-tracking runs.
    :param dtype: The default parameter dtype.
    :param init_device: The device to initialize parameters on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 8,
        head_dim: Optional[int] = None,
        d_state: int = DEFAULT_D_STATE,
        n_groups: int = 1,
        mimo_rank: int = 4,
        rotation_block_size: int = 2,
        norm_eps: float = 1e-5,
        bc_norm: bool = True,
        bc_bias: bool = True,
        a_log_init_max: float = 16.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        _validate_dims(
            n_heads=n_heads,
            d_state=d_state,
            n_groups=n_groups,
            mimo_rank=mimo_rank,
            rotation_block_size=rotation_block_size,
            a_log_init_max=a_log_init_max,
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.d_state = d_state
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups
        self.mimo_rank = mimo_rank
        self.rotation_block_size = rotation_block_size
        self.n_rotation_blocks = d_state // rotation_block_size
        self.angles_per_block = rotation_block_size * (rotation_block_size - 1) // 2
        self.norm_eps = norm_eps
        self.bc_norm_enabled = bc_norm
        self.bc_bias = bc_bias
        self.a_log_init_max = a_log_init_max

        inner = self.n_heads * self.head_dim
        bc_out = self.n_groups * self.mimo_rank * self.d_state

        self.in_x = nn.Linear(d_model, inner, bias=False, dtype=dtype, device=init_device)
        self.in_z = nn.Linear(d_model, inner, bias=False, dtype=dtype, device=init_device)
        self.in_B = nn.Linear(d_model, bc_out, bias=bc_bias, dtype=dtype, device=init_device)
        self.in_C = nn.Linear(d_model, bc_out, bias=bc_bias, dtype=dtype, device=init_device)
        self.dt_proj = nn.Linear(d_model, self.n_heads, bias=False, dtype=dtype, device=init_device)
        self.lam_proj = nn.Linear(
            d_model, self.n_heads, bias=False, dtype=dtype, device=init_device
        )
        # b*(b-1)//2 angles per block spans so(b). At b == 2 this is one angle per channel pair,
        # i.e. exactly the output width of the pre-blocked implementation.
        self.theta_proj = nn.Linear(
            d_model,
            self.n_groups * self.n_rotation_blocks * self.angles_per_block,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.out_proj = nn.Linear(inner, d_model, bias=False, dtype=dtype, device=init_device)

        # SSM parameters are kept in float32 for stability, like GatedDeltaNet.
        self.A_log = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32, device=init_device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32, device=init_device)
        )

        # Norm weights default to ones so the module is usable even before ``init_weights``.
        self.o_norm_weight = nn.Parameter(
            torch.ones(self.head_dim, dtype=dtype, device=init_device)
        )
        if bc_norm:
            self.bc_norm_b = nn.Parameter(torch.ones(self.d_state, dtype=dtype, device=init_device))
            self.bc_norm_c = nn.Parameter(torch.ones(self.d_state, dtype=dtype, device=init_device))
        else:
            self.register_parameter("bc_norm_b", None)
            self.register_parameter("bc_norm_c", None)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply Mamba-3 sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths. Only a single-document batch is
            supported; anything describing a packed multi-document batch raises, since the SSD
            scan carries state across the whole sequence and would leak it across document
            boundaries.

        :raises NotImplementedError: If ``cu_doc_lens`` describes more than one document.

        :returns: The output of shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs

        if cu_doc_lens is not None and cu_doc_lens.numel() > 2:
            # ``cu_doc_lens`` is a flat ``[0, ..., batch_size * seq_len]`` over the whole batch
            # (see ``get_cumulative_document_lengths``), so it holds one entry per document plus
            # the leading zero: 2 entries means a single document and needs no masking. Checking
            # the size rather than the values keeps this free of a host-device sync.
            raise NotImplementedError(
                f"Mamba3Mixer does not support intra-document masking, but 'cu_doc_lens' "
                f"describes {cu_doc_lens.numel() - 1} documents. The Mamba-3 SSD scan would "
                f"carry state across document boundaries, silently corrupting packed training. "
                f"Train Mamba-3 without intra-document masking (unset 'doc_lens'/'max_doc_lens') "
                f"until masking is implemented."
            )

        batch, seq_len, _ = x.shape
        H, P, G, N, R = (
            self.n_heads,
            self.head_dim,
            self.n_groups,
            self.d_state,
            self.mimo_rank,
        )

        xv = self.in_x(x).view(batch, seq_len, H, P)
        z = self.in_z(x).view(batch, seq_len, H, P)
        Bm = self.in_B(x).view(batch, seq_len, G, R, N)
        Cm = self.in_C(x).view(batch, seq_len, G, R, N)

        dt = F.softplus(self.dt_proj(x).float() + self.dt_bias)  # (batch, T, H), > 0
        lam = torch.sigmoid(self.lam_proj(x))  # (batch, T, H) in (0, 1)
        theta = self.theta_proj(x).view(
            batch, seq_len, G, self.n_rotation_blocks, self.angles_per_block
        )
        A = -torch.exp(self.A_log.float())  # (H,), < 0

        if self.bc_norm_enabled:
            # Ordering is immaterial: the rotation is block-diagonal orthogonal, so it preserves
            # the l2 norm of the full N-vector that bc_norm normalizes.
            Bm = _rms_norm(Bm, self.bc_norm_b, self.norm_eps)
            Cm = _rms_norm(Cm, self.bc_norm_c, self.norm_eps)

        y = dispatch_mamba3_ssd(
            xv,
            Bm,
            Cm,
            dt,
            A,
            lam,
            theta,
            heads_per_group=self.heads_per_group,
            block_size=self.rotation_block_size,
        )  # (batch, T, H, P)

        # Gated RMS norm (Mamba-style): normalize the gated output.
        y = _rms_norm(y * F.silu(z), self.o_norm_weight, self.norm_eps)
        y = y.reshape(batch, seq_len, H * P)
        return self.out_proj(y)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for Mamba3Mixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        # Context parallelism (Ulysses/ring) is deferred for the Mamba-3 mixer, matching the
        # tensor-parallel treatment. The recurrent scan would require gathering the full
        # sequence via all-to-all; this is left as a follow-up.
        raise NotImplementedError("Context parallelism is not yet implemented for Mamba3Mixer")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for Mamba3Mixer"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.in_x, self.in_z, self.in_B, self.in_C, self.dt_proj, self.lam_proj):
            init_linear(w, std=std, generator=generator)
        # Rotation angle projection starts small so early training is near-identity. This holds
        # for any block size: small angles put exp(S) near I regardless of b.
        init_linear(self.theta_proj, std=std * 0.1, generator=generator)

        # A_log ~ log(Uniform(0, a_log_init_max)); matches GatedDeltaNet / Mamba conventions.
        self.A_log.copy_(
            nn.init.uniform_(self.A_log, a=0.0, b=self.a_log_init_max, generator=generator).log()
        )

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
        # Inverse of softplus.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        self.o_norm_weight.fill_(1.0)
        if self.bc_norm_enabled:
            assert self.bc_norm_b is not None and self.bc_norm_c is not None
            self.bc_norm_b.fill_(1.0)
            self.bc_norm_c.fill_(1.0)

        # Depth-scale the output projection like GatedDeltaNet / Llama.
        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.out_proj, std=std, generator=generator)

    def kernel_padding_waste(self) -> dict:
        """
        What the official kernel's power-of-two padding costs this configuration.

        Reported separately from :meth:`num_flops_per_token` on purpose: the padded lanes carry
        zeros, so they are wasted work rather than model FLOPs. Counting them as FLOPs would
        *raise* reported MFU for a less efficient configuration, which is backwards. Padding
        shows up correctly as extra wall-clock against an unchanged FLOP count, and this method
        exists so the cause is visible at config time instead of being rediscovered from a
        disappointing throughput number.

        Only :func:`~olmo_core.nn.mamba3.mamba3_ssd_fast.mamba3_ssd_fast` and
        :func:`~olmo_core.nn.mamba3.mamba3_ssd_official.mamba3_ssd_official` pad; the chunked and
        reference paths report zero waste regardless of what this says.

        :returns: Logical and padded widths plus the fraction of lanes carrying zeros.
        """
        d_state_padded = kernel_padded_width(self.d_state)
        head_dim_padded = kernel_padded_width(self.head_dim)
        return {
            "d_state": self.d_state,
            "d_state_padded": d_state_padded,
            "d_state_waste": 1.0 - self.d_state / d_state_padded,
            "head_dim": self.head_dim,
            "head_dim_padded": head_dim_padded,
            "head_dim_waste": 1.0 - self.head_dim / head_dim_padded,
        }

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Approximate FLOPs per token: dominated by the linear projections, plus the rank-R SSD
        state update/readout and the block-rotation preprocessing.

        These are *logical* FLOPs, at the configured ``d_state``/``head_dim``. The official
        kernel may run wider after zero-padding (:meth:`kernel_padding_waste`), and that is
        deliberately excluded: padded lanes compute zeros, so counting them would inflate MFU
        for the configuration that wastes more hardware.
        """
        linear_flops = 2 * sum(
            m.weight.numel()
            for m in (
                self.in_x,
                self.in_z,
                self.in_B,
                self.in_C,
                self.dt_proj,
                self.lam_proj,
                self.theta_proj,
                self.out_proj,
            )
        )
        # State-input outer product + readout, each ~2 FLOPs per element of the rank-R state.
        state_size = self.n_heads * self.mimo_rank * self.d_state * self.head_dim
        recurrent_flops = 2 * 2 * state_size

        b = self.rotation_block_size
        # Applying Q^T to B and C: a b x b matvec per block, per rank, per group, for each of
        # the two. Scales as N*b, so it is monotone in the block size.
        rotation_flops = 2 * 2 * self.n_groups * self.mimo_rank * self.n_rotation_blocks * b * b
        # Prefix product over SO(b): ceil(log2(T)) levels of b x b matmuls per block. The b == 2
        # path collapses to a cumulative sum of angles and does not pay this.
        scan_flops = 0
        if b > 2:
            levels = math.ceil(math.log2(max(seq_len, 2)))
            scan_flops = 2 * levels * self.n_groups * self.n_rotation_blocks * b**3
        return int(linear_flops + recurrent_flops + rotation_flops + scan_flops)


@SequenceMixerConfig.register("mamba3")
@dataclass
class Mamba3MixerConfig(SequenceMixerConfig[Mamba3Mixer]):
    """
    Configuration for :class:`Mamba3Mixer`.

    See :class:`Mamba3Mixer` for a description of the options.
    """

    n_heads: int = 8
    """The number of SSM heads."""
    head_dim: Optional[int] = None
    """Per-head value dimension. Defaults to ``d_model // n_heads``."""
    d_state: int = DEFAULT_D_STATE
    """The SSM state dimension ``N`` (must be a positive multiple of ``rotation_block_size``)."""
    n_groups: int = 1
    """Number of ``(B, C)`` groups shared across heads."""
    mimo_rank: int = 4
    """The MIMO rank ``R``. ``1`` recovers the SISO variant."""
    rotation_block_size: int = 2
    """
    Size ``b`` of the orthogonal transition blocks. ``2`` (the default) is the paper's complex
    diagonal and keeps the layer in TC^0; ``b >= 3`` gives a non-solvable transition monoid.
    Must be one of :func:`admissible_block_sizes` for the chosen ``d_state`` -- notably
    :data:`DEFAULT_D_STATE` cannot express ``b=3``.
    """
    norm_eps: float = 1e-5
    """Epsilon for the internal RMS norms."""
    bc_norm: bool = True
    """Whether to apply BCNorm (QK-norm analog) to ``B`` and ``C``."""
    bc_bias: bool = True
    """Whether the ``B``/``C`` projections use a bias term."""
    a_log_init_max: float = 16.0
    """Upper bound of the ``A_log`` init distribution. Lower it (``~0.1``) for state tracking."""
    dtype: DType = DType.float32
    """The default parameter dtype."""

    def num_params(self, d_model: int) -> int:
        """
        The number of params the :class:`Mamba3Mixer` will have once built.

        :raises ValueError: If the options are ones :meth:`build` would reject.
        """
        _validate_dims(
            n_heads=self.n_heads,
            d_state=self.d_state,
            n_groups=self.n_groups,
            mimo_rank=self.mimo_rank,
            rotation_block_size=self.rotation_block_size,
            a_log_init_max=self.a_log_init_max,
        )
        H = self.n_heads
        P = self.head_dim if self.head_dim is not None else d_model // H
        G = self.n_groups
        N = self.d_state
        R = self.mimo_rank
        inner = H * P
        bc_out = G * R * N

        params = 0
        params += d_model * inner  # in_x
        params += d_model * inner  # in_z
        params += d_model * bc_out  # in_B
        params += d_model * bc_out  # in_C
        if self.bc_bias:
            params += bc_out  # in_B bias
            params += bc_out  # in_C bias
        params += d_model * H  # dt_proj
        params += d_model * H  # lam_proj
        b = self.rotation_block_size
        params += d_model * (G * (N // b) * (b * (b - 1) // 2))  # theta_proj
        params += inner * d_model  # out_proj
        params += H  # A_log
        params += H  # dt_bias
        params += P  # o_norm_weight
        if self.bc_norm:
            params += 2 * N  # bc_norm_b + bc_norm_c
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> Mamba3Mixer:
        """Build the :class:`Mamba3Mixer` module."""
        del layer_idx, n_layers, cache  # unused
        return Mamba3Mixer(
            d_model=d_model,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            d_state=self.d_state,
            n_groups=self.n_groups,
            mimo_rank=self.mimo_rank,
            rotation_block_size=self.rotation_block_size,
            norm_eps=self.norm_eps,
            bc_norm=self.bc_norm,
            bc_bias=self.bc_bias,
            a_log_init_max=self.a_log_init_max,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
