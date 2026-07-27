import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

from .mamba3_ssd_api import dispatch_mamba3_ssd, kernel_padded_width
from .mamba3_ssd_fast import resolve_rotation_scan_impl

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "Mamba3Mixer",
    "Mamba3MixerConfig",
    "DEFAULT_D_STATE",
    "admissible_block_sizes",
    "kernel_padded_width",
    "mamba3_modules_to_ignore_for_fp8",
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


def mamba3_modules_to_ignore_for_fp8(model: nn.Module) -> set:
    """
    Fully-qualified names of every Mamba-3 SSM-parameterising projection in ``model``.

    This is the set to pass as ``Float8Config.modules_to_ignore`` so fp8 conversion skips exactly
    the projections in :attr:`Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS` -- the ones that decide the
    recurrence rather than carry its FLOPs. Names are derived from the built model, so they stay
    correct across depth, block pattern, and layer index. ``Float8Config.apply_float8_linear``
    hard-errors on an ignored name that does not resolve to a module, which turns a stale hardcoded
    list into a conversion-time crash; deriving the list here avoids that entirely.

    :param model: A (possibly hybrid) model that may contain :class:`Mamba3Mixer` modules.

    :returns: FQNs of the sensitive projections; empty if ``model`` has no Mamba-3 mixers.
    """
    ignore = set()
    for name, module in model.named_modules():
        if isinstance(module, Mamba3Mixer):
            for proj in Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS:
                if isinstance(getattr(module, proj, None), nn.Linear):
                    ignore.add(f"{name}.{proj}" if name else proj)
    return ignore


# kernel_padded_width is defined in mamba3_ssd_api (the single source of the padding rule) and
# re-exported here, where it was originally public, so existing imports keep working.


def no_weight_decay_param_names(module: nn.Module) -> List[str]:
    """
    Fully-qualified names of parameters that must not be weight-decayed.

    Follows ``mamba_ssm``, which tags ``A_log``, ``dt_bias`` and ``D`` with
    ``_no_weight_decay = True``. These set the recurrence's timescale rather than its magnitude,
    so shrinking them toward zero moves the model's forgetting rate rather than regularizing it.

    Feed the result to :class:`~olmo_core.optim.OptimGroupOverride` with ``weight_decay=0.0``.
    """
    return [n for n, p in module.named_parameters() if getattr(p, "_no_weight_decay", False)]


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
    a_log_init_min: float,
    a_log_init_max: float,
) -> None:
    """
    Validate the mixer's shape-determining options.

    Shared by :meth:`Mamba3Mixer.__init__` and :meth:`Mamba3MixerConfig.num_params` so that
    sizing a config fails exactly when building it would: ``num_params`` is read long before any
    module exists (it is what :meth:`Mamba3Config.build` logs and what sizing scripts print), and
    its integer arithmetic would otherwise report a plausible number for a config that cannot be
    constructed.

    :raises OLMoConfigurationError: If any option is out of range or the dimensions are
        incompatible. This is a configuration fault, so it uses the repo's typed config error
        rather than a bare ``ValueError`` -- callers that gate on ``OLMoConfigurationError``
        (the framework's own convention, e.g. the attention and transformer configs) then catch
        it uniformly. Runtime tensor-contract violations inside the SSD kernels stay
        ``ValueError``; those are programming errors, not configuration ones.
    """
    if rotation_block_size < 2:
        raise OLMoConfigurationError(f"rotation_block_size must be >= 2, got {rotation_block_size}")
    if d_state < rotation_block_size:
        # Divisibility alone would wave ``d_state=0`` through (``0 % b == 0``), leaving zero
        # rotation blocks and zero-width B/C -- a mixer that returns exactly zero for every
        # input rather than failing.
        raise OLMoConfigurationError(
            f"d_state ({d_state}) must be at least rotation_block_size "
            f"({rotation_block_size}); a smaller state leaves no rotation blocks at all"
        )
    if d_state % rotation_block_size != 0:
        raise OLMoConfigurationError(
            f"d_state ({d_state}) must be divisible by rotation_block_size "
            f"({rotation_block_size}) for the Mamba-3 rotation"
        )
    if a_log_init_min <= 0:
        raise OLMoConfigurationError(f"a_log_init_min must be > 0, got {a_log_init_min}")
    if a_log_init_max <= a_log_init_min:
        raise OLMoConfigurationError(
            f"a_log_init_max must be > a_log_init_min, got "
            f"({a_log_init_min}, {a_log_init_max})"
        )
    if n_groups < 1:
        raise OLMoConfigurationError(f"n_groups must be >= 1, got {n_groups}")
    if n_heads < 1:
        raise OLMoConfigurationError(f"n_heads must be >= 1, got {n_heads}")
    if n_heads % n_groups != 0:
        raise OLMoConfigurationError(
            f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})"
        )
    if mimo_rank < 1:
        raise OLMoConfigurationError(f"mimo_rank must be >= 1, got {mimo_rank}")


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
    :param a_log_init_min: Lower bound of the ``A_log`` init distribution. ``A`` is drawn as
        ``-Uniform(a_log_init_min, a_log_init_max)``, so this floors the decay rate. The default
        of 1.0 matches ``mamba_ssm``'s ``A_init_range=(1, 16)``. A bound of 0 admits heads with
        ``A ~ 0``, which never decay and act as accumulators over the whole sequence, and makes
        ``log(0) = -inf`` reachable.
    :param a_log_init_max: Upper bound of the ``A_log`` init distribution; see
        ``a_log_init_min``. Together the default ``(1, 16)`` spreads the per-head memory horizon
        across roughly three orders of magnitude once ``dt in [0.001, 0.1]`` is folded in, which
        is what gives the layer both short- and long-horizon heads at init.
    :param rotation_scan_impl: Which of
        :data:`~olmo_core.nn.mamba3.mamba3_ssd_fast.ROTATION_SCAN_IMPLS` computes the ``b >= 3``
        prefix product. ``None`` (the default) defers to ``MAMBA3_ROTATION_SCAN_IMPL``, so the
        historical behaviour is unchanged; naming it here instead puts the choice in the config a
        checkpoint saves and a run logs. It only reaches the fast official-kernel path, so on CPU
        or under activation checkpointing it is recorded and ignored.
    :param dtype: The default parameter dtype.
    :param init_device: The device to initialize parameters on.
    """

    FP8_SENSITIVE_PROJECTIONS: tuple[str, ...] = (
        "in_B",
        "in_C",
        "dt_proj",
        "lam_proj",
        "theta_proj",
    )
    """
    Projections that parameterise the state-space recurrence and must stay out of fp8.

    These carry almost no FLOPs but decide the SSM's behaviour: ``in_B``/``in_C`` are the state
    read/write matrices, ``dt_proj`` the timestep, ``lam_proj`` the trapezoidal blend, and
    ``theta_proj`` the ``SO(b)`` rotation angles that make ``b >= 3`` non-solvable. fp8 rounding
    here is all risk (it perturbs decay, stability, and the NC^1 rotation) and no reward (the
    speedup lives in the big GEMMs ``in_x``/``in_z``/``out_proj``, which are *not* listed here).
    :func:`mamba3_modules_to_ignore_for_fp8` turns this into ``Float8Config.modules_to_ignore``.
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
        a_log_init_min: float = 0.05,
        a_log_init_max: float = 16.0,
        prefer_official_kernel: Optional[bool] = None,
        rotation_scan_impl: Optional[str] = None,
        theta_max: Optional[float] = None,
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
            a_log_init_min=a_log_init_min,
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
        self.a_log_init_min = a_log_init_min
        self.a_log_init_max = a_log_init_max
        # Kernel selection, forwarded to `dispatch_mamba3_ssd`. ``None`` (default) uses the fast
        # official kernel whenever it is eligible (CUDA, ``mimo_rank == 1``, reduced precision) --
        # the intended main-run path. ``False`` forces the chunked PyTorch form, which is what an
        # activation-checkpointed run must use: the official kernel's ``autograd.Function`` is
        # incompatible with non-reentrant `checkpoint_wrapper` (see `dispatch_mamba3_ssd`). ``True``
        # requires the official kernel and errors if it cannot run.
        self.prefer_official_kernel = prefer_official_kernel
        # Checked here, not on the first forward: the rotation is only reached once training has
        # paid for the model build, the compile warmup and a step, by which point a typo has cost
        # a GPU-hour to discover. Stored un-defaulted -- `None` has to survive as `None` so the
        # environment still decides at call time for callers that never set this.
        if rotation_scan_impl is not None:
            try:
                rotation_scan_impl = resolve_rotation_scan_impl(rotation_scan_impl)
            except ValueError as e:
                # Config faults use the framework's typed error (see `_validate_dims`); the
                # `ValueError` the kernel module raises is the runtime-contract flavour.
                raise OLMoConfigurationError(str(e)) from e
        self.rotation_scan_impl = rotation_scan_impl
        if theta_max is not None and theta_max <= 0:
            raise OLMoConfigurationError(f"theta_max must be positive, got {theta_max}")
        self.theta_max = theta_max

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
        # Both set the recurrence's timescale, not its magnitude, so weight decay on them is
        # not regularization -- it is a pull toward a fixed forgetting rate. Decaying A_log
        # toward 0 drives |A| to 1; decaying dt_bias toward 0 drives dt to softplus(0)=0.693,
        # ~70x the init median. Since the memory horizon is 1/(dt*|A|), that squeezes it from
        # both ends. `mamba_ssm` marks the same parameters exempt; see
        # `no_weight_decay_param_names`, which the training script turns into optimizer groups.
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

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
        if self.theta_max is not None:
            # Bound the per-step rotation angle, because an unbounded one silently
            # destroys the state channel it exists to carry. The cumulative rotation
            # Q_t Q_s^T = R_t ... R_{s+1} is a random walk on the *non-abelian* SO(b), which mixes
            # to Haar measure in ~1/||theta||^2 steps; past that, C_t^T (R_t ... R_{s+1}) B_s
            # averages to zero and no information survives the gap. Measured on this code at
            # seq 4096, the lag at which tr(Q_t Q_{t-d}^T) falls to half is 72 steps at
            # ||theta|| = 0.1, 746 at 0.03, and beyond the sequence at 0.01 -- and the default
            # init (theta_proj std = 1e-3 * ...) already emits ||theta|| ~ 0.1 per block, i.e. a
            # 72-token horizon on a 4096-token sequence, before a single gradient step. Keeping
            # ||theta|| <~ 1/sqrt(seq_len) keeps the mixing time past the sequence length.
            #
            # `c * tanh(x / c)` rather than a clamp: it is the identity near zero (so it does not
            # perturb the near-identity init the rotation is designed around) and saturates
            # smoothly, keeping the gradient finite everywhere -- a hard clamp would zero the
            # gradient for every angle that needed correcting most.
            #
            # b == 2 is bounded on the same grounds, despite its walk being abelian. A cumulative
            # scalar angle still diffuses: E[cos(sum theta)] = prod E[cos theta_u] decays
            # geometrically in the gap, so the 2-D phase relaxes to the uniform measure on the
            # circle exactly as the SO(b >= 3) walk mixes to Haar. Periodicity bounds the rotation
            # matrix, not the information it carries across a gap.
            theta = self.theta_max * torch.tanh(theta / self.theta_max)
        A = -torch.exp(self.A_log.float())  # (H,), < 0

        if self.bc_norm_enabled:
            # Normalize *before* the rotation, and note the order is load-bearing. The rotation
            # is block-diagonal orthogonal so it preserves the l2 norm, which makes the
            # normalization step itself commute -- but the learned per-channel `bc_norm_b/c`
            # scale does not. The two orders agree only while those weights are still at their
            # all-ones init; once trained they diverge (measured ~0.85 absolute on unit-scale
            # inputs). Swapping them is a silent model change, not a refactor.
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
            prefer_official_kernel=self.prefer_official_kernel,
            rotation_scan_impl=self.rotation_scan_impl,
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

        # A = -Uniform(a_log_init_min, a_log_init_max), stored as its log. The default (1, 16)
        # is mamba_ssm's A_init_range. The lower bound is load-bearing: at 0 a head can draw
        # A ~ 0, which never decays and turns that channel into a document-mean accumulator.
        self.A_log.copy_(
            nn.init.uniform_(
                self.A_log, a=self.a_log_init_min, b=self.a_log_init_max, generator=generator
            ).log()
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

        These are *model* FLOPs -- the arithmetic the layer is defined to do -- not the FLOPs a
        particular kernel happens to execute. Two implementation overheads are deliberately
        excluded on the same principle, because counting either would raise reported MFU for the
        configuration that wastes more hardware:

        - the official kernel's zero-padding to a power-of-two width
          (:meth:`kernel_padding_waste`);
        - the chunked path's intra-chunk ``Q x Q`` form, which trades extra arithmetic for
          parallelism and so does more FLOPs than the recurrence it implements.

        Both show up correctly as extra wall-clock against an unchanged numerator.
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

        del seq_len  # every term below is per-token and sequence-length independent

        b = self.rotation_block_size
        # Applying Q^T to B and C: a b x b matvec per block, per rank, per group, for each of
        # the two. Scales as N*b, so it is monotone in the block size.
        rotation_flops = 2 * 2 * self.n_groups * self.mimo_rank * self.n_rotation_blocks * b * b

        # Prefix product over SO(b), plus building the per-step rotations. The b == 2 path
        # collapses to a cumulative sum of angles and pays neither.
        #
        # This previously multiplied by `ceil(log2(seq_len))`, which confused the *depth* of the
        # scan with its *work* and overstated the term by 4.5-6.5x at production lengths. An
        # associative scan over T elements costs O(T) compositions, not O(T log T): the
        # implementation does one b x b matmul per token in the sequential pass and one more
        # applying the chunk carry, so ~2 per token. The Hillis-Steele stage runs only over the
        # T/chunk chunk totals and is a rounding error against that.
        scan_flops = 0
        if b > 2:
            per_token_matmuls = 2  # sequential pass + carry broadcast
            # Building R_t itself: Rodrigues is two 3x3 matmuls plus elementwise work; the
            # `matrix_exp` fallback at other b costs several times this and is not modelled.
            per_token_matmuls += 2
            scan_flops = 2 * per_token_matmuls * self.n_groups * self.n_rotation_blocks * b**3

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
    a_log_init_min: float = 0.05
    """Lower bound of the ``A_log`` init distribution. Must be ``> 0``: at 0 a head can draw
    ``A ~ 0``, never decay, and act as an accumulator over the whole sequence, and ``log(0)``
    is reachable. ``mamba_ssm`` uses 1.0; 0.05 is lower on purpose, so ~6% of heads start
    slow-decaying. Measured on the 4.8B b=2/b=3 runs, that inherited tail was the *only* source
    of long-horizon heads either arm ended up with, and nothing in training recreated it."""
    a_log_init_max: float = 16.0
    """Upper bound of the ``A_log`` init distribution."""
    prefer_official_kernel: Optional[bool] = None
    """
    SSD kernel selection. ``None`` (default, the intended main-run setting) uses the fast
    official ``mamba-ssm`` kernel whenever eligible. Set ``False`` for activation-checkpointed
    runs -- the official kernel's ``autograd.Function`` is incompatible with non-reentrant
    activation checkpointing, so AC runs must take the chunked PyTorch path. ``True`` forces the
    official kernel and errors if it cannot run.
    """
    rotation_scan_impl: Optional[str] = None
    """
    Which of :data:`~olmo_core.nn.mamba3.mamba3_ssd_fast.ROTATION_SCAN_IMPLS` computes the
    ``b >= 3`` prefix product, or ``None`` (the default) to defer to ``MAMBA3_ROTATION_SCAN_IMPL``.

    Naming it here rather than only in the environment is what gets the choice into the saved
    checkpoint config and the startup log. While it lived only in the environment, a resume in a
    fresh shell that lost the export silently fell back to ``chunked`` at 33,468 tok/s instead of
    ``quaternion``'s 75,040 -- a 2.2x regression that raised nothing and left no record of which
    scan the run had actually used.
    """
    theta_max: Optional[float] = None
    """
    Upper bound on the per-step rotation angle, applied as ``theta_max * tanh(theta / theta_max)``
    at every ``rotation_block_size``. ``None`` (the default) leaves ``theta`` unbounded, which is
    the historical behaviour.

    At ``b >= 3`` the cumulative rotation is a random walk on a non-abelian group and mixes to Haar
    measure in ``~1/||theta||^2`` steps, after which the state channel carries nothing. Measured at
    seq 4096: the half-life of ``tr(Q_t Q_{t-d}^T)`` is 72 steps at ``||theta|| = 0.1``, 746 at
    ``0.03``, and past the sequence at ``0.01``. The default init already emits ``||theta|| ~ 0.1``,
    so an unbounded ``b >= 3`` run has a ~72-token memory horizon from step zero and shrinks from
    there as ``theta_proj`` grows. Set this to about ``1/sqrt(seq_len)`` (``~0.015`` at 4096).

    Exempt at ``b == 2`` by construction: that rotation is RoPE on a cumulative scalar angle, where
    wrapping is periodic and harmless, so a bound would only cost frequency range.
    """
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
            a_log_init_min=self.a_log_init_min,
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
            a_log_init_min=self.a_log_init_min,
            a_log_init_max=self.a_log_init_max,
            prefer_official_kernel=self.prefer_official_kernel,
            rotation_scan_impl=self.rotation_scan_impl,
            theta_max=self.theta_max,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
