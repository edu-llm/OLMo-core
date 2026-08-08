import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_chunk_gated_delta_rule,
    dispatch_chunk_kda,
    has_fla,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction
from olmo_core.nn.functional import l2_normalize

if TYPE_CHECKING:
    from olmo_core.nn.gated_convolution import GateStructure
    from olmo_core.nn.transformer.init import InitMethod


def _init_short_conv(
    conv: nn.Module, *, std: float, generator: Optional[torch.Generator] = None
) -> None:
    """
    Initialize one short convolution, whether it is plain or gated.

    ``init_linear`` writes ``m.weight`` and expects an :class:`torch.nn.Conv1d`, which a
    :class:`~olmo_core.nn.gated_convolution.GatedCausalConv1d` is not — it *holds* one. This
    resolves that and then zeroes the gate.

    **The convolution weight is drawn first, unconditionally, from the shared generator**, so a
    plain arm and a gated arm draw the same convolution values at the same point in the random
    stream. This matters more than it looks: if a gated arm consumed randomness *before* the
    convolution draw, every subsequent parameter in the model would differ too, and that confound
    does not show up anywhere in a loss curve.

    The ``"depthwise"`` gate draws nothing, so a ``depthwise`` gated arm shares the entire random
    stream with the plain arm and differs only by the gate. The ``"lowrank"`` gate **must** draw
    its shared down-projection -- zeroing both factors of a product kills the branch permanently
    -- so a ``lowrank`` arm's later parameters do differ from the plain arm's. That is reported
    rather than assumed:
    :meth:`~olmo_core.nn.gated_convolution.GatedCausalConv1d.init_gate_weights` returns whether it
    drew, and this function surfaces it on the module as ``_gate_init_consumed_randomness``.

    .. warning::
        ``"lowrank"`` has a **second, separate** divergence channel that this function cannot fix.
        :meth:`~olmo_core.nn.transformer.model.Transformer.init_weights` runs
        ``for module in self.modules(): module.reset_parameters()`` (``model.py:290``) **before**
        the seeded generator is created at ``:299``, so that sweep draws from the **global** RNG.
        ``"lowrank"`` adds three ``nn.Linear`` submodules per convolution -- nine per layer -- each
        with its own ``reset_parameters``, so the global RNG lands in a different state than the
        plain arm's. Parameter *values* are still safe (they all come from the passed generator),
        but anything downstream that draws from the global RNG is not.

        ``"depthwise"`` is exempt: ``pre_scale`` and ``post_scale`` are bare
        :class:`torch.nn.Parameter` objects with no ``reset_parameters``, and
        :class:`~olmo_core.nn.gated_convolution.GatedCausalConv1d` holds exactly one
        :class:`torch.nn.Conv1d`, matching what
        :class:`~olmo_core.nn.convolution.CausalConv1d` *is*.

        So a ``"lowrank"`` arm is **not seed-comparable** to a plain arm. Pair its cells on data
        seed and treat the init draw as a nuisance, or accept that the two arms differ by more than
        the gate.

    :param conv: The convolution module.
    :param std: The standard deviation for the convolution weight.
    :param generator: The random generator, shared across the whole model.

    :raises TypeError: If ``conv`` is neither a plain nor a gated short convolution.
    """
    from olmo_core.nn.gated_convolution import GatedCausalConv1d
    from olmo_core.nn.transformer.init import init_linear

    # Dispatch on TYPE, not on attribute presence. 'getattr(conv, "conv", conv)' would look right
    # and be wrong the moment any Conv1d subclass grew a '.conv' attribute -- it would initialize
    # that and leave the real weight at torch's default, which trains.
    if isinstance(conv, GatedCausalConv1d):
        init_linear(conv.conv, std=std, generator=generator)
        drew = conv.init_gate_weights(std=std, generator=generator)
        conv._gate_init_consumed_randomness = drew
    elif isinstance(conv, nn.Conv1d):
        init_linear(conv, std=std, generator=generator)
    else:
        raise TypeError(f"cannot initialize a short convolution of type {type(conv).__name__}")


class GatedDeltaNet(SequenceMixer):
    """
    The layer implementation for `Gated Delta Networks <https://arxiv.org/abs/2412.06464>`_.

    Modified from: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/layers/gated_deltanet.py#L34

    This is a linear attention variant that uses a gated delta rule for recurrent
    state updates, providing efficient O(n) sequence modeling.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 2.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``True``. If set to ``True``, the beta
        will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs Through Negative
        Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param gate_init: Initialization strategy for the recurrent decay gate. ``"default"`` preserves
        the original GatedDeltaNet initialization. ``"halflife"`` initializes the no-input decay
        with log-spaced half-lives between ``gate_min_halflife`` and ``gate_max_halflife``.
        ``"halflife_random"`` samples those half-lives log-uniformly.
        ``"halflife_random_a"`` also samples target half-lives log-uniformly,
        but preserves a default-like random ``A_log`` scale. ``"halflife_a"`` and
        ``"halflife_a_permuted"`` preserve that random ``A_log`` scale with exact
        log-spaced half-life coverage.
    :param gate_min_halflife: Minimum initial half-life in tokens when ``gate_init="halflife"``.
    :param gate_max_halflife: Maximum initial half-life in tokens when ``gate_init="halflife"``.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        gate_init: str = "default",
        gate_min_halflife: float = 8.0,
        gate_max_halflife: float = 4096.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.gate_init = gate_init
        self.gate_min_halflife = gate_min_halflife
        self.gate_max_halflife = gate_max_halflife
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0
        assert gate_init in {
            "default",
            "halflife",
            "halflife_random",
            "halflife_random_a",
            "halflife_a",
            "halflife_a_permuted",
        }
        assert gate_min_halflife > 0
        assert gate_max_halflife >= gate_min_halflife

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.w_a = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)
        self.w_b = nn.Linear(d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device)

        self.A_log = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.n_v_heads, dtype=dtype, device=init_device))

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.w_g = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, device=init_device)  # type: ignore
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False
        self.uly: Optional[UlyssesContextParallelStyle] = None

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply gated delta network sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T_og, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_v_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0
        g = -self.A_log.float().exp() * F.softplus(self.w_a(x).float() + self.dt_bias)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T_local, C] -> [B, T_total, C/CP]
            q, k = all_to_all_cp2hp([q, k], self._cp_group)
            v = all_to_all_single_cp2hp(v, self._cp_group)
            g, beta = all_to_all_cp2hp([g, beta], self._cp_group)

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        T = q.size(1)
        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        o, _ = dispatch_chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu_doc_lens, use_qk_l2norm_in_kernel=True
        )

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            o = all_to_all_single_hp2cp(o, self._cp_group)

        g = self.w_g(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, g).view(B, T_og, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for GatedDeltaNet")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        if ring is not None:
            raise NotImplementedError("Ring context parallelism is not supported for GatedDeltaNet")
        assert uly is not None

        cp_world_size = cp_mesh.size()
        if cp_world_size == 1:
            return

        # Ulysses CP requires divisibility by CP world size for:
        # 1. n_v_heads - for head partitioning in the recurrent kernel
        # 2. key_dim and value_dim - for channel partitioning in the conv layers
        assert self.n_v_heads % cp_world_size == 0
        assert self.key_dim % cp_world_size == 0
        assert self.value_dim % cp_world_size == 0

        self.uly = uly
        self._cp_mesh = cp_mesh
        self._cp_group = cp_mesh.get_group()
        self.cp_enabled = True

        self.q_conv1d.apply_cp(cp_mesh)
        self.k_conv1d.apply_cp(cp_mesh)
        self.v_conv1d.apply_cp(cp_mesh)

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
                f"init method '{init_method}' is not supported for GatedDeltaNet"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g):
            init_linear(w, std=std, generator=generator)
        for w in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(w, std=std, generator=generator)

        if self.gate_init == "default":
            self.A_log.copy_(nn.init.uniform_(self.A_log, a=0, b=16, generator=generator).log())
            dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
            dt = torch.exp(
                nn.init.uniform_(self.dt_bias, generator=generator)
                * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min),
            ).clamp(min=dt_init_floor)
        elif self.gate_init == "halflife":
            self.A_log.zero_()
            half_life = torch.logspace(
                math.log10(self.gate_min_halflife),
                math.log10(self.gate_max_halflife),
                steps=self.n_v_heads,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )
            dt = (math.log(2.0) / half_life).to(dtype=self.dt_bias.dtype)
        elif self.gate_init == "halflife_random":
            self.A_log.zero_()
            log_half_life = torch.empty(
                self.n_v_heads,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )
            log_half_life.uniform_(
                math.log(self.gate_min_halflife),
                math.log(self.gate_max_halflife),
                generator=generator,
            )
            dt = (math.log(2.0) / log_half_life.exp()).to(dtype=self.dt_bias.dtype)
        elif self.gate_init == "halflife_random_a":
            A = torch.empty(self.n_v_heads, device=self.A_log.device, dtype=torch.float32)
            nn.init.uniform_(A, a=0, b=16, generator=generator)
            # Preserve the default A scale distribution without allowing exact zero,
            # since dt must solve A * softplus(dt_bias) for the requested half-life.
            A.clamp_(min=1e-4)
            self.A_log.copy_(A.log().to(dtype=self.A_log.dtype))
            log_half_life = torch.empty(
                self.n_v_heads,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )
            log_half_life.uniform_(
                math.log(self.gate_min_halflife),
                math.log(self.gate_max_halflife),
                generator=generator,
            )
            dt = (math.log(2.0) / (A * log_half_life.exp())).to(dtype=self.dt_bias.dtype)
        elif self.gate_init in {"halflife_a", "halflife_a_permuted"}:
            A = torch.empty(self.n_v_heads, device=self.A_log.device, dtype=torch.float32)
            nn.init.uniform_(A, a=0, b=16, generator=generator)
            A.clamp_(min=1e-4)
            self.A_log.copy_(A.log().to(dtype=self.A_log.dtype))
            half_life = torch.logspace(
                math.log10(self.gate_min_halflife),
                math.log10(self.gate_max_halflife),
                steps=self.n_v_heads,
                device=self.dt_bias.device,
                dtype=torch.float32,
            )
            if self.gate_init == "halflife_a_permuted":
                half_life = half_life[
                    torch.randperm(self.n_v_heads, device=half_life.device, generator=generator)
                ]
            dt = (math.log(2.0) / (A * half_life)).to(dtype=self.dt_bias.dtype)
        else:
            raise RuntimeError(f"unexpected gate_init '{self.gate_init}'")
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for Gated Delta Net.

        This accounts for:
        - Linear projections (w_q, w_k, w_v, w_a, w_b, w_g, w_out)
        - Short convolutions (q, k, v)
        - Gated delta rule recurrent computation
        - Gated RMS normalization
        """
        del seq_len
        # Linear projection FLOPs (2 ops per multiply-add)
        linear_flops = 2 * sum(
            m.weight.numel()
            for m in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g, self.w_out)
        )

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output)
        conv_flops = (
            2
            * self.conv_size
            * (self.key_dim + self.key_dim + self.value_dim)  # q_conv1d  # k_conv1d  # v_conv1d
        )

        # Gated delta rule recurrent computation per token:
        # - Outer product k ⊗ v: n_v_heads * head_k_dim * head_v_dim
        # - State decay: n_v_heads * head_k_dim * head_v_dim
        # - Beta scaling: n_v_heads * head_k_dim * head_v_dim
        # - Query-state matmul: n_v_heads * head_k_dim * head_v_dim
        # Each is 2 FLOPs per element (multiply-add or similar)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 4 * state_size

        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("gated_delta_net")
@dataclass
class GatedDeltaNetConfig(SequenceMixerConfig[GatedDeltaNet]):
    """
    Configuration for :class:`GatedDeltaNet`.

    See :class:`GatedDeltaNet` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    n_v_heads: Optional[int] = None
    """
    The number of value heads. If ``None``, defaults to ``n_heads``.
    If ``n_v_heads`` > ``n_heads``, GVA (Grouped Value Attention) is applied.

    GVA is preferred over GQA for linear RNNs like GDN because the recurrent state
    has shape ``(n_v_heads, head_k_dim, head_v_dim)``. Unlike softmax attention where
    the KV cache grows with sequence length (motivating GQA to reduce it), the linear
    RNN state is constant size regardless of sequence length. Since there's no memory
    scaling issue to solve, we instead can opt to increase the state size to improve the model's
    capacity to compress long-range context. Increasing ``n_v_heads`` directly
    increases this fixed state size.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 2.0
    """
    The expansion ratio for the value dimension (``head_v_dim = head_dim * expand_v``).
    Like ``n_v_heads``, this increases the constant-size recurrent state, improving
    capacity without memory scaling concerns.
    """
    allow_neg_eigval: bool = True
    """
    Allow negative eigenvalues in the recurrent dynamics.
    """
    gate_init: str = "default"
    """
    Initialization strategy for the recurrent decay gate. ``"default"`` preserves the original
    GatedDeltaNet initialization. ``"halflife"`` initializes the no-input decay with log-spaced
    half-lives between :attr:`gate_min_halflife` and :attr:`gate_max_halflife`.
    ``"halflife_random"`` samples those half-lives log-uniformly.
    ``"halflife_random_a"`` samples target half-lives log-uniformly while preserving a
    default-like random ``A_log`` scale. ``"halflife_a"`` and ``"halflife_a_permuted"``
    preserve that random ``A_log`` scale with exact log-spaced half-life coverage.
    """
    gate_min_halflife: float = 8.0
    """
    Minimum initial half-life in tokens when :attr:`gate_init` is ``"halflife"``.
    """
    gate_max_halflife: float = 4096.0
    """
    Maximum initial half-life in tokens when :attr:`gate_init` is ``"halflife"``.
    """
    conv_size: int = 4
    """
    The kernel size of the short convolution.
    """
    conv_bias: bool = False
    """
    Whether to use bias in the short convolution.
    """
    norm_eps: float = 1e-5
    """
    The epsilon value for the normalization layer.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the GatedDeltaNet will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_a, w_b, w_g, w_out
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += d_model * n_v_heads  # w_a
        params += d_model * n_v_heads  # w_b
        params += d_model * value_dim  # w_g
        params += value_dim * d_model  # w_out

        # A_log and dt_bias parameters
        params += n_v_heads  # A_log
        params += n_v_heads  # dt_bias

        # Short convolutions (kernel_size * hidden_size for each)
        params += self.conv_size * key_dim  # q_conv1d
        params += self.conv_size * key_dim  # k_conv1d
        params += self.conv_size * value_dim  # v_conv1d
        if self.conv_bias:
            params += key_dim  # q_conv1d bias
            params += key_dim  # k_conv1d bias
            params += value_dim  # v_conv1d bias

        # FusedRMSNormGated (weight only, no bias)
        params += head_v_dim  # o_norm

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> GatedDeltaNet:
        """
        Build the GatedDeltaNet module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return GatedDeltaNet(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            gate_init=self.gate_init,
            gate_min_halflife=self.gate_min_halflife,
            gate_max_halflife=self.gate_max_halflife,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class KimiDeltaAttention(SequenceMixer):
    """
    The layer implementation for `Kimi Delta Attention (KDA)
    <https://arxiv.org/abs/2510.26692>`_.

    Modified from: https://github.com/fla-org/flash-linear-attention/blob/v0.4.1/fla/layers/kda.py

    KDA is a close cousin of :class:`GatedDeltaNet`: it shares the short-convolution / beta /
    output-gate / gated-RMSNorm skeleton, but replaces the *per-head* scalar forget gate with a
    *per-channel* one. Concretely, the decay applied to the recurrent state is

    .. math::
        g = -\\exp(A_\\text{log}) \\cdot \\text{softplus}(f(x) + \\text{dt\\_bias})

    where :math:`f(x)` has one value per (head, key-channel) pair rather than one per head. The
    gate input :math:`f(x)` and the output gate are both produced by *low-rank* projections that
    bottleneck through ``head_v_dim``, which keeps the extra parameter cost small.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 1.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``False``. If set to ``True``,
        the beta will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs
        Through Negative Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param gated_conv: Replace the three plain short convolutions with LFM2/LIV-style *gated*
        ones (:class:`~olmo_core.nn.gated_convolution.GatedCausalConv1d`). Default: ``False``,
        which is the shipped KDA operator. **Do not change this default** — every measurement in
        ``KDA/HANDOFF.md`` was taken with plain convolutions, and a default that silently moved
        would invalidate all of them.
    :param gated_conv_activation: The activation inside the gated convolution. ``None`` matches
        LFM2, whose block is activation-free; ``"silu"`` keeps KDA's activation *and* adds the
        gate, which is the arm that separates "gating helps" from "removing silu helps". Ignored
        when ``gated_conv=False``.
    :param gate_structure: ``"depthwise"`` (per-channel, ~0.06% parameter cost, keeps arms
        parameter-matched) or ``"lowrank"``. See
        :data:`~olmo_core.nn.gated_convolution.GateStructure`.
    :param gate_rank: The gate bottleneck width, required when ``gate_structure="lowrank"``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 1.0,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        conv_activation: Optional[str] = ActivationFunction.silu.value,
        gated_conv: bool = False,
        gated_conv_activation: Optional[str] = None,
        gate_structure: "GateStructure" = "depthwise",
        gate_rank: Optional[int] = None,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.conv_activation = conv_activation
        self.gated_conv = gated_conv
        self.gated_conv_activation = gated_conv_activation
        self.gate_structure: "GateStructure" = gate_structure
        self.gate_rank = gate_rank
        self.norm_eps = norm_eps

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions.
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0
        # The KDA kernel materializes the key head dim in a single Triton block.
        assert self.head_k_dim <= 256, "KDA only supports a key head dim <= 256"

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.w_b = nn.Linear(d_model, self.n_heads, bias=False, dtype=dtype, device=init_device)

        # NOTE: unlike GatedDeltaNet, the forget-gate projection is a low-rank bottleneck through
        # 'head_v_dim' that produces one value per (head, key-channel) pair.
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False, dtype=dtype, device=init_device),
        )

        # NOTE: 'A_log' is per-head, but 'dt_bias' is per (head, key-channel) and is kept *flat*
        # with shape '(n_heads * head_k_dim,)' since that's what the fused KDA gate kernel expects.
        self.A_log = nn.Parameter(torch.empty(self.n_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=dtype, device=init_device))

        # The three short convolutions, either the shipped plain ones or the LIV-style gated
        # ones. Both classes take '(x=..., cu_seqlens=...)', so the forward pass is unchanged
        # apart from threading 'gate_input' when the gate reads the layer input.
        self.q_conv1d = self._build_conv(
            hidden_size=self.key_dim, dtype=dtype, init_device=init_device
        )
        self.k_conv1d = self._build_conv(
            hidden_size=self.key_dim, dtype=dtype, init_device=init_device
        )
        self.v_conv1d = self._build_conv(
            hidden_size=self.value_dim, dtype=dtype, init_device=init_device
        )

        # NOTE: like 'f_proj', the output gate is a low-rank bottleneck, and its second projection
        # carries a bias (unlike every other projection in this layer).
        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, dtype=dtype, device=init_device),
        )
        # NOTE: KDA gates the output norm with a sigmoid, whereas GatedDeltaNet uses the default
        # swish.
        self.o_norm = FusedRMSNormGated(  # type: ignore
            self.head_v_dim,
            activation="sigmoid",
            eps=norm_eps,
            device=init_device,
            dtype=dtype,
        )
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def _build_conv(self, *, hidden_size: int, dtype: torch.dtype, init_device: str) -> nn.Module:
        """
        Build one short convolution, plain or gated.

        Factored out so all three streams are guaranteed to get the same kind of convolution.
        Building them inline three times is how one stream ends up plain while the other two are
        gated, which trains fine and is not the operator under test.

        :param hidden_size: The channel count for this stream.
        :param dtype: Parameter dtype.
        :param init_device: Device to initialize on.

        :returns: A :class:`~olmo_core.nn.convolution.CausalConv1d` or a
            :class:`~olmo_core.nn.gated_convolution.GatedCausalConv1d`.
        """
        if not self.gated_conv:
            return CausalConv1d(
                hidden_size=hidden_size,
                kernel_size=self.conv_size,
                bias=self.conv_bias,
                # From the config, NOT hard-coded to silu. Hard-coding it made the
                # no-activation arm unbuildable, and that arm is the only one that isolates the
                # gate -- see 'conv_activation' on the config.
                activation=self.conv_activation,  # type: ignore[arg-type]
                dtype=dtype,
                init_device=init_device,
            )

        from olmo_core.nn.gated_convolution import GatedCausalConv1d

        return GatedCausalConv1d(
            hidden_size=hidden_size,
            kernel_size=self.conv_size,
            gate_structure=self.gate_structure,
            d_model=self.d_model,
            gate_rank=self.gate_rank,
            bias=self.conv_bias,
            # NOTE: passed explicitly rather than defaulted. 'CausalConv1d' defaults to 'silu'
            # and 'GatedCausalConv1d' defaults to None, so relying on either default here would
            # make the arm's operator depend on which class was constructed.
            activation=self.gated_conv_activation,  # type: ignore[arg-type]
            dtype=dtype,
            init_device=init_device,
        )

    def _conv_kwargs(self, x: torch.Tensor) -> dict:
        """
        Extra keyword arguments the convolutions need for this forward pass.

        A ``"lowrank"`` gate reads the mixer input, so it needs ``gate_input=x``; nothing else
        does. Computed once here rather than at three call sites.

        :param x: The mixer input.

        :returns: Keyword arguments to pass to each convolution.
        """
        if self.gated_conv and self.gate_structure == "lowrank":
            return {"gate_input": x}
        return {}

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply Kimi Delta Attention sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Requires ``batch_size == 1``.

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.

        :raises RuntimeError: If ``cu_doc_lens`` is given with a batch size greater than 1.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T, _ = x.shape

        if cu_doc_lens is not None and B != 1:
            raise RuntimeError(
                "The KDA kernel requires a batch size of 1 when 'cu_doc_lens' is given "
                f"(got batch size {B}). Flatten variable-length inputs into a single sequence "
                "first, or turn off intra-document masking with 'generate_doc_lengths=False'."
            )

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_v_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # shape: (batch_size, seq_len, n_heads)
        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        conv_kwargs = self._conv_kwargs(x)
        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens, **conv_kwargs)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens, **conv_kwargs)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens, **conv_kwargs)

        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        # The *raw* (pre-activation) forget gate. The kernel turns this into the log-space decay
        # '-exp(A_log) * softplus(raw + dt_bias)' internally when 'use_gate_in_kernel=True'.
        # shape: (batch_size, seq_len, n_heads, head_k_dim)
        raw = self.f_proj(x).view(B, T, -1, self.head_k_dim)

        A_log, dt_bias = self.A_log, self.dt_bias
        if self.n_v_heads > self.n_heads:
            # For grouped-value attention we repeat the key-side inputs for simplicity.
            #
            # NOTE: we also have to repeat the gate parameters, which the reference layer does
            # *not* do. The fused gate kernel infers its head count from ``g.shape[-2]`` (which is
            # ``n_v_heads`` after the repeat) and indexes ``A_log``/``dt_bias`` with it, so passing
            # the unexpanded parameters would read out of bounds.
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)
            raw = raw.repeat_interleave(repeat_factor, dim=-2)
            beta = beta.repeat_interleave(repeat_factor, dim=-1)
            A_log = A_log.repeat_interleave(repeat_factor, dim=0)
            dt_bias = (
                dt_bias.view(self.n_heads, self.head_k_dim)
                .repeat_interleave(repeat_factor, dim=0)
                .reshape(-1)
            )

        # shape: (batch_size, seq_len, n_v_heads, head_v_dim)
        o, _ = dispatch_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu_doc_lens,
        )

        # shape: (batch_size, seq_len, n_v_heads, head_v_dim)
        g = self.g_proj(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, g).view(B, T, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError(
            "Tensor parallelism is not yet implemented for KimiDeltaAttention"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        raise NotImplementedError(
            "Context parallelism is not yet implemented for KimiDeltaAttention"
        )

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
                f"init method '{init_method}' is not supported for KimiDeltaAttention"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.w_q, self.w_k, self.w_v, self.w_b, *self.f_proj, *self.g_proj):
            assert isinstance(w, nn.Linear)
            init_linear(w, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            _init_short_conv(conv, std=std, generator=generator)

        # The reference KDA initialization: 'A_log = log(U(1, 16))' with a zero 'dt_bias'.
        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1.0, b=16.0, generator=generator).log())
        self.dt_bias.zero_()

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for Kimi Delta Attention.

        This accounts for:

        - Linear projections (``w_q``, ``w_k``, ``w_v``, ``w_b``, ``f_proj``, ``g_proj``,
          ``w_out``)
        - Short convolutions (q, k, v), and their gates when ``gated_conv=True``
        - The delta rule recurrent computation
        - Gated RMS normalization

        :param seq_len: The sequence length (unused, since KDA is linear in the sequence length).

        :returns: The number of FLOPs per token.
        """
        del seq_len
        # Linear projection FLOPs (2 ops per multiply-add).
        linears: list[nn.Linear] = [self.w_q, self.w_k, self.w_v, self.w_b, self.w_out]
        for seq in (self.f_proj, self.g_proj):
            for m in seq:
                assert isinstance(m, nn.Linear)
                linears.append(m)
        linear_flops = 2 * sum(m.weight.numel() for m in linears)

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output).
        conv_channels = self.key_dim + self.key_dim + self.value_dim  # q, k, v
        conv_flops = 2 * self.conv_size * conv_channels

        # Gate FLOPs, when gated. Each stream gets two gates; each gate is a sigmoid plus a
        # multiply onto the stream, and 'lowrank' also pays its two projections. Small against
        # the projections, but reported so an arm matched on this quantity is matched honestly
        # rather than by omission.
        gate_flops = 0
        if self.gated_conv:
            if self.gate_structure == "lowrank":
                assert self.gate_rank is not None
                # Shared down-projection once per stream, then one up-projection per gate.
                gate_flops += 2 * 3 * self.d_model * self.gate_rank
                gate_flops += 2 * 2 * self.gate_rank * conv_channels
            else:
                # ONE multiply per channel per gate to form the pre-activation, so 2 FLOPs each
                # over 2 gates -- not '2 * 2 * channels', which double-counted. Under 0.01% of the
                # layer either way, but the test bound is '< 1%' and so cannot catch it; arms are
                # matched on this quantity, so it should be honest rather than merely harmless.
                gate_flops += 2 * conv_channels
            # The elementwise 2*sigmoid and the multiply onto the stream, both gates.
            gate_flops += 2 * 2 * conv_channels

        # Delta rule recurrent computation per token:
        # - Outer product k ⊗ v: n_v_heads * head_k_dim * head_v_dim
        # - State decay: n_v_heads * head_k_dim * head_v_dim
        # - Beta scaling: n_v_heads * head_k_dim * head_v_dim
        # - Query-state matmul: n_v_heads * head_k_dim * head_v_dim
        # Each is 2 FLOPs per element (multiply-add or similar).
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 4 * state_size

        return int(linear_flops + conv_flops + gate_flops + recurrent_flops)


@SequenceMixerConfig.register("kimi_delta_attention")
@dataclass
class KimiDeltaAttentionConfig(SequenceMixerConfig[KimiDeltaAttention]):
    """
    Configuration for :class:`KimiDeltaAttention`.

    See :class:`KimiDeltaAttention` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    n_v_heads: Optional[int] = None
    """
    The number of value heads. If ``None``, defaults to ``n_heads``.
    If ``n_v_heads`` > ``n_heads``, GVA (Grouped Value Attention) is applied.

    Like :attr:`expand_v`, this increases the constant-size recurrent state, improving the
    model's capacity to compress long-range context without any memory scaling concerns.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 1.0
    """
    The expansion ratio for the value dimension (``head_v_dim = head_dim * expand_v``).
    """
    allow_neg_eigval: bool = False
    """
    Allow negative eigenvalues in the recurrent dynamics.
    """
    conv_size: int = 4
    """
    The kernel size of the short convolution.
    """
    conv_bias: bool = False
    """
    Whether to use bias in the short convolution.
    """
    conv_activation: Optional[str] = "silu"
    """
    The activation inside the *plain* short convolution. ``"silu"`` is what KDA ships.

    Exists so ``None`` -- a convolution with no activation at all -- is reachable. Without it the
    activation was hard-coded and the only arm that **isolates the gate** could not be built. See
    :attr:`gated_conv_activation` for why that arm is necessary rather than nice to have.
    """
    gated_conv: bool = False
    """
    Replace the three plain short convolutions with LFM2/LIV-style *gated* ones.

    ``False`` is the shipped KDA operator and the only value any existing measurement was taken
    at. **Do not change this default.** Every number in ``KDA/HANDOFF.md`` — the arm ledger, the
    285,832 tok/s throughput, the 5.169 GiB peak — assumes plain convolutions, and a default that
    moved would silently invalidate all of them while every test still passed.
    """
    gated_conv_activation: Optional[str] = None
    """
    The activation inside the gated convolution, ignored when :attr:`gated_conv` is ``False``.

    .. important::
        **With ``gate_structure="depthwise"``, ``None`` does NOT mean activation-free.** The
        depthwise pre-gate satisfies ``2*sigmoid(a*u)*u == (2/a)*silu(a*u)`` exactly, so it *is* a
        SiLU with a learnable per-channel slope, moved to before the convolution, with its
        amplitude absorbed into the convolution taps. See
        :data:`~olmo_core.nn.gated_convolution.GateStructure` for the derivation.

        This is why the arm set needs **four** cells, not three. LIV changes two things at once --
        it adds gating and it removes the activation -- and with a depthwise gate the "removes the
        activation" half does not actually happen:

    ==============================  ================  ==========================  ==================
    arm                             ``gated_conv``    ``gated_conv_activation``   ``conv_activation``
    ==============================  ================  ==========================  ==================
    ``kda-plain`` (as shipped)      ``False``         n/a                         ``"silu"``
    ``kda-plain-noact``             ``False``         n/a                         ``None``
    ``kda-gated``                   ``True``          ``None``                    n/a
    ``kda-gated-silu``              ``True``          ``"silu"``                  n/a
    ==============================  ================  ==========================  ==================

    The contrast that isolates the gate is **``kda-gated`` minus ``kda-plain-noact``**, not
    ``kda-gated`` minus ``kda-plain``: the latter varies the activation's position, its
    learnability, and the post gate all together, so a difference cannot be attributed. Against
    this project's measured within-arm SD of 0.01463 nats you would see a real effect and be
    unable to say what caused it.
    """
    gate_structure: str = "depthwise"
    """
    ``"depthwise"`` or ``"lowrank"``. See
    :data:`~olmo_core.nn.gated_convolution.GateStructure`.

    ``"depthwise"`` costs ``2 * hidden_size`` per convolution — about 0.06% of the layer's
    projections — so the arms are parameter-matched for free. A full dense gate projection would
    be +60% and would confound the mechanism with capacity.
    """
    gate_rank: Optional[int] = None
    """
    The gate bottleneck width, required when :attr:`gate_structure` is ``"lowrank"``.
    """
    norm_eps: float = 1e-5
    """
    The epsilon value for the normalization layer.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def gate_params(self, d_model: int) -> int:
        """
        Parameters the convolution gates add, over the three streams.

        Zero when :attr:`gated_conv` is ``False``. Exposed separately from :meth:`num_params` so
        an arm's parameter delta can be asserted directly, rather than inferred by subtracting
        two large totals where a mistake of a few thousand disappears into rounding.

        :param d_model: The model dimensionality.

        :returns: The number of gate parameters.

        :raises ValueError: Via :meth:`validate_gate_options`. An incoherent config must not be
            able to produce a plausible-looking parameter count -- ``num_params`` is what solves
            FFN widths for parameter matching, so a number returned here from a config that could
            never build would move the anchor for every arm in the ledger.
        """
        self.validate_gate_options()
        if not self.gated_conv:
            return 0

        from olmo_core.nn.gated_convolution import gate_param_count

        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        return sum(
            gate_param_count(
                hidden_size=h,
                structure=self.gate_structure,  # type: ignore[arg-type]
                d_model=d_model,
                gate_rank=self.gate_rank,
            )
            for h in (key_dim, key_dim, value_dim)
        )

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the KimiDeltaAttention will have once built.

        :param d_model: The model dimensionality.

        :returns: The number of parameters.
        """
        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_b, w_out.
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += d_model * n_heads  # w_b
        params += value_dim * d_model  # w_out

        # Low-rank forget gate projection.
        params += d_model * head_v_dim  # f_proj[0]
        params += head_v_dim * key_dim  # f_proj[1]

        # Low-rank output gate projection (the second projection has a bias).
        params += d_model * head_v_dim  # g_proj[0]
        params += head_v_dim * value_dim + value_dim  # g_proj[1]

        # A_log is per-head while dt_bias is per (head, key channel).
        params += n_heads  # A_log
        params += key_dim  # dt_bias

        # Short convolutions (kernel_size * hidden_size for each).
        params += self.conv_size * key_dim  # q_conv1d
        params += self.conv_size * key_dim  # k_conv1d
        params += self.conv_size * value_dim  # v_conv1d
        if self.conv_bias:
            params += key_dim  # q_conv1d bias
            params += key_dim  # k_conv1d bias
            params += value_dim  # v_conv1d bias

        # FusedRMSNormGated (weight only, no bias).
        params += head_v_dim  # o_norm

        # Convolution gates, zero unless 'gated_conv' is set.
        params += self.gate_params(d_model)

        return params

    def gate_activation_bytes(
        self,
        d_model: int,
        *,
        batch_size: int,
        seq_len: int,
        bytes_per_element: int = 2,
    ) -> int:
        """
        Extra activation bytes one gated KDA layer holds for its backward pass.

        **The parameter delta is not the cost of this experiment; this is.** At
        ``d_model=2048, n_heads=16, expand_v=1.0`` with 8192 tokens per rank in bf16 — the
        microbatch KDA's 285,832 tok/s was measured at — this is **384 MiB per layer**, so a
        28-layer model pays about **10.5 GiB** on top of KDA's measured 5.169 GiB peak. That is
        roughly 3x peak, not a rounding error: it fits on a 40 GiB card and would not fit at
        ``seq_len=32768``. Size a run from a measured peak on the gated arm.

        :param d_model: The model dimensionality.
        :param batch_size: The per-rank batch size.
        :param seq_len: The sequence length.
        :param bytes_per_element: 2 for bf16, 4 for fp32.

        :returns: The number of extra bytes, or 0 when :attr:`gated_conv` is ``False``.
        """
        if not self.gated_conv:
            return 0

        from olmo_core.nn.gated_convolution import gate_activation_bytes

        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * int(head_dim * self.expand_v)

        return sum(
            gate_activation_bytes(
                hidden_size=h,
                batch_size=batch_size,
                seq_len=seq_len,
                bytes_per_element=bytes_per_element,
            )
            for h in (key_dim, key_dim, value_dim)
        )

    def validate_gate_options(self) -> None:
        """
        Check the gate options for coherence, without building anything.

        **Separate from :meth:`build` on purpose, and it is not a style choice.**
        :class:`KimiDeltaAttention`'s constructor opens with ``assert has_fla()``, and ``fla``
        needs CUDA — so a check living inside ``build`` is unreachable on any machine without a
        GPU, and no cheap test can show it fires. Mutation M12 exploited exactly that: deleting
        the check produced an ``AssertionError`` on a laptop (which read as "caught, for the wrong
        reason") and nothing at all on a GPU host, where
        :class:`~olmo_core.nn.gated_convolution.GatedCausalConv1d`'s own identically-worded error
        satisfied the test.

        Callable on a bare config, so the refusals are verifiable for free.

        :raises ValueError: If gate options are set while :attr:`gated_conv` is ``False`` — a
            config that reads as a treatment arm in a diff and trains as the control — or if
            ``gate_structure="lowrank"`` carries no :attr:`gate_rank`.
        """
        if self.conv_activation not in (None, "silu", "swish"):
            raise ValueError(
                f"unsupported conv_activation '{self.conv_activation}'; use None, 'silu' or 'swish'"
            )
        if self.gated_conv_activation not in (None, "silu", "swish"):
            raise ValueError(
                f"unsupported gated_conv_activation '{self.gated_conv_activation}'; "
                "use None, 'silu' or 'swish'"
            )
        if not self.gated_conv:
            if self.gate_rank is not None:
                raise ValueError("'gate_rank' is set but 'gated_conv' is False")
            if self.gated_conv_activation is not None:
                raise ValueError("'gated_conv_activation' is set but 'gated_conv' is False")
            if self.gate_structure != "depthwise":
                # This is the case the comment above was written for and the guard originally
                # missed: 'gate_structure="lowrank", gated_conv=False' reads as a treatment arm in
                # a YAML diff, builds three plain convolutions, and gives two identically-trained
                # controls with different names -- a guaranteed null. "depthwise" is the field
                # default, so it is the only value that can mean "not set".
                raise ValueError(
                    f"'gate_structure' is {self.gate_structure!r} but 'gated_conv' is False"
                )
            return
        if self.gate_structure not in ("depthwise", "lowrank"):
            raise ValueError(f"unknown gate structure '{self.gate_structure}'")
        if self.gate_structure == "lowrank" and self.gate_rank is None:
            # Worded differently from GatedCausalConv1d's check for the same condition, so a test
            # can tell which one fired. Both are real; this one fires first, on 'meta', before any
            # convolution is allocated.
            raise ValueError(
                "'gate_rank' is required when gate_structure='lowrank' "
                "(refused by KimiDeltaAttentionConfig before the module is constructed)"
            )

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> KimiDeltaAttention:
        """
        Build the KimiDeltaAttention module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).

        :returns: The built module.

        :raises ValueError: Via :meth:`validate_gate_options`.
        """
        del layer_idx, n_layers, cache  # Unused

        self.validate_gate_options()

        return KimiDeltaAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            conv_activation=self.conv_activation,
            gated_conv=self.gated_conv,
            gated_conv_activation=self.gated_conv_activation,
            gate_structure=self.gate_structure,  # type: ignore[arg-type]
            gate_rank=self.gate_rank,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class KimiDeltaHouseholder(SequenceMixer):
    """
    Kimi Delta Attention with ``R`` **Householder (DeltaProduct) factors** per token.

    This is :class:`KimiDeltaAttention` -- the same short-convolution / beta / low-rank-gate /
    gated-RMSNorm skeleton, and the same *per-channel* forget gate

    .. math::
        g = -\\exp(A_\\text{log}) \\cdot \\text{softplus}(f(x) + \\text{dt\\_bias})

    -- generalized so that each token applies ``R`` successive rank-1 delta updates to the
    recurrent state instead of one. See `DeltaProduct: Improving State-Tracking in Linear RNNs via
    Householder Products <https://arxiv.org/abs/2502.10297>`_ for the ``R``-factor generalization
    and `Kimi Linear <https://arxiv.org/abs/2510.26692>`_ for the per-channel gate. The
    combination is implemented by :func:`olmo_core.nn.attention.kda_householder.chunk_kda_householder`.

    The decay is applied **once per token**, not once per factor, so only the *key side* of the
    layer widens with ``R``: ``w_k``, ``w_v``, ``w_b`` and the ``k``/``v`` short convolutions
    produce ``R`` factors per token, while ``w_q``, ``q_conv1d``, ``f_proj`` (the forget gate),
    ``g_proj`` (the output gate), ``o_norm`` and ``w_out`` are unchanged from
    :class:`KimiDeltaAttention`. At ``num_householder=1`` this layer has exactly the same
    parameters as :class:`KimiDeltaAttention`.

    .. warning::
        The Triton backward allocates an ``O(batch * seq_len * n_heads * head_k_dim *
        head_v_dim)`` float32 workspace -- the full per-token state history. It is proportional to
        sequence length and can OOM at production shapes even when the forward fits comfortably.
        See :func:`~olmo_core.nn.attention.kda_householder.kda_householder_bwd`.

    .. note::
        Unlike :class:`KimiDeltaAttention`, which lets ``fla``'s fused kernel build the gate and
        L2-normalize ``q``/``k`` internally, both are done explicitly here: the kernel takes a
        **raw per-token** log-decay ``g`` (no cumsum) and does not normalize ``q``/``k`` itself.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param num_householder: The number of Householder / delta factors ``R`` applied per token.
        Default: 2.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 1.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``False``. If set to ``True``,
        the beta will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs
        Through Negative Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        num_householder: int = 2,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 1.0,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        backend: Literal["triton", "torch"] = "triton",
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        assert backend in ("triton", "torch")
        self.backend = backend
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_householder = num_householder
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions.
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0
        assert num_householder >= 1, "'num_householder' must be at least 1"
        # The kernel materializes the key head dim in a single Triton block.
        assert self.head_k_dim <= 256, "KimiDeltaHouseholder only supports a key head dim <= 256"

        R = num_householder

        # NOTE: the key side produces 'R' factors per token, so 'w_k'/'w_v'/'w_b' and the
        # 'k'/'v' convolutions are 'R' times wider than in KimiDeltaAttention. The query and both
        # gates stay per-token.
        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, R * self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(
            d_model, R * self.value_dim, bias=False, dtype=dtype, device=init_device
        )
        self.w_b = nn.Linear(d_model, R * self.n_heads, bias=False, dtype=dtype, device=init_device)

        # The forget-gate projection is a low-rank bottleneck through 'head_v_dim' that produces
        # one value per (head, key-channel) pair, once per token.
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False, dtype=dtype, device=init_device),
        )

        # NOTE: 'A_log' is per-head and 'dt_bias' is per (head, key-channel), kept *flat* with
        # shape '(n_heads * head_k_dim,)'. Neither scales with 'R': the decay is applied once per
        # token, shared by that token's 'R' factors.
        self.A_log = nn.Parameter(torch.empty(self.n_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=dtype, device=init_device))

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=R * self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=R * self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )

        # Like 'f_proj', the output gate is a low-rank bottleneck, and its second projection
        # carries a bias (unlike every other projection in this layer).
        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, dtype=dtype, device=init_device),
        )
        # KDA gates the output norm with a sigmoid, whereas GatedDeltaNet uses the default swish.
        self.o_norm = FusedRMSNormGated(  # type: ignore
            self.head_v_dim,
            activation="sigmoid",
            eps=norm_eps,
            device=init_device,
            dtype=dtype,
        )
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply KDA + ``R``-Householder sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``. Must not be
            ``float32`` at the point the kernel is called, so either run this layer under
            ``torch.autocast(..., dtype=torch.bfloat16)`` or build it with a ``bfloat16`` dtype.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``). These are in
            *token* units; the kernel scales them by ``R`` internally for the interleaved
            tensors. Requires ``batch_size == 1``.

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.

        :raises RuntimeError: If ``cu_doc_lens`` is given with a batch size greater than 1.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface

        # NOTE: imported lazily because 'kda_householder' imports triton at module scope, and we
        # still want this layer to be constructible (and its 'num_params' checkable) on a machine
        # without triton installed.
        from olmo_core.nn.attention.kda_householder import chunk_kda_householder

        B, T, _ = x.shape
        R = self.num_householder
        H = self.n_heads

        if cu_doc_lens is not None and B != 1:
            raise RuntimeError(
                "The KDA Householder kernel requires a batch size of 1 when 'cu_doc_lens' is "
                f"given (got batch size {B}). Flatten variable-length inputs into a single "
                "sequence first, or turn off intra-document masking with "
                "'generate_doc_lengths=False'."
            )

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, R * n_heads * head_k_dim),
        #        (batch_size, seq_len, R * n_v_heads * head_v_dim)
        q = self.q_conv1d(x=self.w_q(x), cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=self.w_k(x), cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=self.w_v(x), cu_seqlens=cu_doc_lens)

        # shape: (batch_size, seq_len, R * n_heads)
        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # The per-channel log-decay, computed in float32 here rather than in the kernel: unlike
        # 'fla.ops.kda.chunk_kda' this kernel has no fused-gate path, and it wants the *raw*
        # per-token 'g' (no cumsum).
        # shape: (batch_size, seq_len, n_heads, head_k_dim)
        g = -self.A_log.float().exp().unsqueeze(-1) * F.softplus(
            self.f_proj(x).view(B, T, H, self.head_k_dim).float()
            + self.dt_bias.float().view(H, self.head_k_dim)
        )

        # Split the factor axis out. The 'R' factors of a token are the *outer* axis of each
        # projection's output, i.e. '[B, T, (R h d)] -> [B, T, R, h, d]', which is what makes the
        # '[B, T, R, h, d] -> [B, T * R, h, d]' reshape below equal to einops
        # "b t (n h d) -> b (t n) h d" -- the interleaved layout the kernel expects.
        q = q.view(B, T, H, self.head_k_dim)
        k = k.view(B, T, R, H, self.head_k_dim)
        v = v.view(B, T, R, self.n_v_heads, self.head_v_dim)
        beta = beta.view(B, T, R, H)

        if self.n_v_heads > H:
            # For grouped-value attention we repeat the key-side inputs for simplicity. The gate
            # is repeated *after* it has been built from the per-head 'A_log' / 'dt_bias', so
            # (unlike KimiDeltaAttention, which hands the raw gate to a fused kernel) those
            # parameters never need to be repeated.
            repeat_factor = self.n_v_heads // H
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)
            beta = beta.repeat_interleave(repeat_factor, dim=-1)
            g = g.repeat_interleave(repeat_factor, dim=-2)

        # shape: (batch_size, seq_len * R, n_v_heads, head_k_dim),
        #        (batch_size, seq_len * R, n_v_heads, head_v_dim),
        #        (batch_size, seq_len * R, n_v_heads)
        k = k.reshape(B, T * R, -1, self.head_k_dim)
        v = v.reshape(B, T * R, -1, self.head_v_dim)
        beta = beta.reshape(B, T * R, -1)

        # The kernel has no 'use_qk_l2norm_in_kernel' flag, so normalize here instead. This
        # matches KimiDeltaAttention, where the fused kernel normalizes *after* the convolution.
        q = l2_normalize(q)
        k = l2_normalize(k)

        # shape: (batch_size, seq_len, n_v_heads, head_v_dim)
        o, _ = chunk_kda_householder(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_householder=R,
            cu_seqlens=cu_doc_lens,
            backend=self.backend,
        )

        # shape: (batch_size, seq_len, n_v_heads, head_v_dim)
        gate = self.g_proj(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, gate).view(B, T, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError(
            "Tensor parallelism is not yet implemented for KimiDeltaHouseholder"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        # The interleaved '[B, T * R, ...]' key-side layout does not line up with the Ulysses
        # all-to-all, which assumes one entry per token on every tensor it redistributes.
        raise NotImplementedError(
            "Context parallelism is not supported for KimiDeltaHouseholder: the interleaved "
            "'T * R' key/value layout is incompatible with the Ulysses all-to-all"
        )

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
                f"init method '{init_method}' is not supported for KimiDeltaHouseholder"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.w_q, self.w_k, self.w_v, self.w_b, *self.f_proj, *self.g_proj):
            assert isinstance(w, nn.Linear)
            init_linear(w, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(conv, std=std, generator=generator)

        # The reference KDA initialization: 'A_log = log(U(1, 16))' with a zero 'dt_bias'.
        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1.0, b=16.0, generator=generator).log())
        self.dt_bias.zero_()

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for KDA + ``R`` Householder factors.

        This accounts for:

        - Linear projections (``w_q``, ``w_k``, ``w_v``, ``w_b``, ``f_proj``, ``g_proj``,
          ``w_out``). ``w_k``/``w_v``/``w_b`` are already ``R`` times wider, so their FLOPs scale
          with ``R`` automatically.
        - Short convolutions (q, k, v), where the ``k``/``v`` convolutions are ``R`` times wider.
        - The delta rule recurrent computation, which applies ``R`` rank-1 updates per token.
        - Gated RMS normalization.

        :param seq_len: The sequence length (unused, since this layer is linear in the sequence
            length).

        :returns: The number of FLOPs per token.
        """
        del seq_len
        R = self.num_householder

        # Linear projection FLOPs (2 ops per multiply-add).
        linears: list[nn.Linear] = [self.w_q, self.w_k, self.w_v, self.w_b, self.w_out]
        for seq in (self.f_proj, self.g_proj):
            for m in seq:
                assert isinstance(m, nn.Linear)
                linears.append(m)
        linear_flops = 2 * sum(m.weight.numel() for m in linears)

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output). The
        # 'k'/'v' convolutions run over 'R' times as many channels.
        conv_flops = (
            2
            * self.conv_size
            * (
                self.key_dim  # q_conv1d
                + R * self.key_dim  # k_conv1d
                + R * self.value_dim  # v_conv1d
            )
        )

        # Recurrent computation per token, in units of the state size
        # 'n_v_heads * head_k_dim * head_v_dim':
        # - State decay: once per token.
        # - Query-state matmul: once per token.
        # - Outer product k ⊗ v: once per Householder factor.
        # - Beta-scaled delta 'beta * (v - k @ S)': once per Householder factor.
        # Each is 2 FLOPs per element (multiply-add or similar). At 'R == 1' this reduces to the
        # '2 * 4 * state_size' of KimiDeltaAttention.
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * (2 + 2 * R) * state_size

        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("kimi_delta_householder")
@dataclass
class KimiDeltaHouseholderConfig(SequenceMixerConfig[KimiDeltaHouseholder]):
    """
    Configuration for :class:`KimiDeltaHouseholder`.

    See :class:`KimiDeltaHouseholder` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    num_householder: int = 2
    """
    The number of Householder / delta factors ``R`` applied per token. Each token contributes
    ``R`` rank-1 updates to the recurrent state, which widens ``w_k``, ``w_v``, ``w_b`` and the
    ``k``/``v`` short convolutions by a factor of ``R``. ``num_householder=1`` recovers the
    parameterization of :class:`KimiDeltaAttentionConfig`.
    """
    n_v_heads: Optional[int] = None
    """
    The number of value heads. If ``None``, defaults to ``n_heads``.
    If ``n_v_heads`` > ``n_heads``, GVA (Grouped Value Attention) is applied.

    Like :attr:`expand_v`, this increases the constant-size recurrent state, improving the
    model's capacity to compress long-range context without any memory scaling concerns.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 1.0
    """
    The expansion ratio for the value dimension (``head_v_dim = head_dim * expand_v``).
    """
    allow_neg_eigval: bool = False
    """
    Allow negative eigenvalues in the recurrent dynamics.
    """
    backend: Literal["triton", "torch"] = "triton"
    """
    Kernel backend. ``"triton"`` is the fast fused-recurrent path; both its forward and backward
    are validated, and it is what you want for training. ``"torch"`` selects the reference
    pure-PyTorch recurrence -- much slower, but the only path that runs on CPU, supports
    ``float64`` (hence ``torch.autograd.gradcheck``), and is twice differentiable.
    """
    conv_size: int = 4
    """
    The kernel size of the short convolution.
    """
    conv_bias: bool = False
    """
    Whether to use bias in the short convolution.
    """
    norm_eps: float = 1e-5
    """
    The epsilon value for the normalization layer.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the KimiDeltaHouseholder will have once built.

        :param d_model: The model dimensionality.

        :returns: The number of parameters.
        """
        n_heads = self.n_heads
        R = self.num_householder
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_b, w_out. The key side produces 'R' factors per
        # token, so 'w_k'/'w_v'/'w_b' are 'R' times wider.
        params += d_model * key_dim  # w_q
        params += d_model * R * key_dim  # w_k
        params += d_model * R * value_dim  # w_v
        params += d_model * R * n_heads  # w_b
        params += value_dim * d_model  # w_out

        # Low-rank forget gate projection (per token, so no 'R').
        params += d_model * head_v_dim  # f_proj[0]
        params += head_v_dim * key_dim  # f_proj[1]

        # Low-rank output gate projection (per token; the second projection has a bias).
        params += d_model * head_v_dim  # g_proj[0]
        params += head_v_dim * value_dim + value_dim  # g_proj[1]

        # A_log is per-head while dt_bias is per (head, key channel). Neither scales with 'R':
        # the decay is applied once per token.
        params += n_heads  # A_log
        params += key_dim  # dt_bias

        # Short convolutions (kernel_size * hidden_size for each). The 'k'/'v' convolutions run
        # over 'R' times as many channels.
        params += self.conv_size * key_dim  # q_conv1d
        params += self.conv_size * R * key_dim  # k_conv1d
        params += self.conv_size * R * value_dim  # v_conv1d
        if self.conv_bias:
            params += key_dim  # q_conv1d bias
            params += R * key_dim  # k_conv1d bias
            params += R * value_dim  # v_conv1d bias

        # FusedRMSNormGated (weight only, no bias).
        params += head_v_dim  # o_norm

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> KimiDeltaHouseholder:
        """
        Build the KimiDeltaHouseholder module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).

        :returns: The built module.
        """
        del layer_idx, n_layers, cache  # Unused

        return KimiDeltaHouseholder(
            d_model=d_model,
            n_heads=self.n_heads,
            num_householder=self.num_householder,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            backend=self.backend,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
