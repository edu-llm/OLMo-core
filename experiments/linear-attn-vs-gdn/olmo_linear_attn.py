"""
Plain (ungated) linear attention as a drop-in OLMo-core sequence mixer, purpose
built so it can be compared apples-to-apples against
:class:`olmo_core.nn.attention.recurrent.GatedDeltaNet` on the **same** ``fla``
chunked-scan Triton kernel family.

This is intentionally a faithful *ablation* of ``GatedDeltaNet``:

  * identical ``w_q`` / ``w_k`` / ``w_v`` / ``w_out`` projections and head layout,
  * identical short causal convolutions (silu) on q/k/v,
  * identical QK L2-normalization,
  * identical output RMSNorm placement,

and it differs in exactly one thing -- the recurrence. GatedDeltaNet applies the
*gated delta rule*

    S_t = (diag(a_t) - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T

whereas this module applies the *ungated linear-attention* update

    S_t = S_{t-1} + k_t v_t^T ,   o_t = q_t S_t

with no decay gate ``g`` (a_t), no delta strength ``beta``, and a non-gated output
norm. Both recurrences are evaluated by the same ``fla`` chunked-scan machinery
(:func:`fla.ops.linear_attn.chunk_linear_attn` vs
:func:`fla.ops.gated_delta_rule.chunk_gated_delta_rule`), so the *only*
experimental variable is the gated delta mechanism itself.

Defined as an importable module (NOT ``__main__``) so OLMo-core config
(de)serialization -- which stores a config by its fully-qualified class import
path -- can reconstruct it on checkpoint resume.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import has_fla
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction

try:  # available on the training box; guarded so the module still imports for --dry-run
    from fla.ops.linear_attn import chunk_linear_attn
except ImportError:
    chunk_linear_attn = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class LinearAttention(SequenceMixer):
    """
    Plain (ungated) linear-attention sequence mixer -- a gate/delta ablation of
    :class:`olmo_core.nn.attention.recurrent.GatedDeltaNet`.

    :param d_model: The model hidden size.
    :param n_heads: The number of (key/query) heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim (``head_v_dim = head_dim * expand_v``).
    :param conv_size: The kernel size of the short convolution.
    :param conv_bias: Whether to use bias in the short convolution.
    :param use_short_conv: Whether to apply the q/k/v short convolutions (kept ``True`` to
        match GatedDeltaNet's token mixing; only the gating is ablated, not the conv).
    :param qk_l2norm: L2-normalize q and k per head before the kernel (matches GatedDeltaNet's
        ``use_qk_l2norm_in_kernel=True``).
    :param normalize: Apply the classic linear-attention denominator normalization inside the
        kernel. Kept ``False`` by default so the update is the pure ungated cumulative sum
        (the honest gate/delta ablation); the output RMSNorm handles scale growth.
    :param norm_eps: The epsilon value for the output normalization layer.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 16,
        n_v_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        expand_v: float = 2.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        use_short_conv: bool = True,
        qk_l2norm: bool = True,
        normalize: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla(), "flash-linear-attention (fla) is required for LinearAttention"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.conv_size = conv_size
        self.use_short_conv = use_short_conv
        self.qk_l2norm = qk_l2norm
        self.normalize = normalize

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_v_heads * self.head_v_dim)

        # Consistency checks: ensure expand_v produces integer dimensions
        assert math.isclose(self.n_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5)
        assert math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5)
        assert self.n_v_heads >= self.n_heads and self.n_v_heads % self.n_heads == 0

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)

        if self.use_short_conv:
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

        # Non-gated RMSNorm (GatedDeltaNet uses FusedRMSNormGated; the gate is the ablated part).
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps, dtype=dtype, device=init_device)
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply plain (ungated) linear-attention sequence mixing.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Ignored -- the plain linear-attention kernel has no variable-length
            (document-boundary reset) path. Both mixers in this experiment therefore run without
            intra-sequence resets, keeping the comparison symmetric.

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.
        """
        del cu_doc_lens, kwargs  # no var-len path; symmetric with GDN run (also no resets)
        assert chunk_linear_attn is not None, "fla.ops.linear_attn.chunk_linear_attn unavailable"

        B, T_og, _ = x.shape

        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.use_short_conv:
            q = self.q_conv1d(x=q, cu_seqlens=None)
            k = self.k_conv1d(x=k, cu_seqlens=None)
            v = self.v_conv1d(x=v, cu_seqlens=None)

        T = q.size(1)
        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        if self.qk_l2norm:
            q = F.normalize(q, p=2.0, dim=-1)
            k = F.normalize(k, p=2.0, dim=-1)

        # Same fla chunked-scan kernel family as chunk_gated_delta_rule, minus gate + delta.
        # NO head_first= HERE, AND ITS ABSENCE IS THE POINT. This module was written against
        # flash-linear-attention 0.4.1, where chunk_linear_attn took head_first to choose between
        # a [B, H, T, D] and a [B, T, H, D] layout. 0.5.x removed the argument and standardised on
        # [B, T, H, D] -- exactly what head_first=False selected -- so dropping the keyword keeps
        # the same layout and the same numerics. Passing it against 0.5.2 is a TypeError, which is
        # how run_019ff4db died after the softmax arm had already been timed.
        #
        # The pin moved because GatedDeltaNet2 needs fla.ops.gdn2, which does not exist before
        # 0.5.0. So bumping the extra for one mixer broke a sibling mixer's call site, in a file
        # that no test imports.
        o, _ = chunk_linear_attn(q=q, k=k, v=v, normalize=self.normalize)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o).view(B, T_og, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for LinearAttention")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        raise NotImplementedError("Context parallelism is not implemented for LinearAttention")

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
                f"init method '{init_method}' is not supported for LinearAttention"
            )

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for w in (self.w_q, self.w_k, self.w_v):
            init_linear(w, std=std, generator=generator)
        if self.use_short_conv:
            for w in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                init_linear(w, std=std, generator=generator)

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for plain linear attention (GDN minus gate/delta).
        """
        del seq_len
        # Linear projection FLOPs (2 ops per multiply-add)
        linear_flops = 2 * sum(m.weight.numel() for m in (self.w_q, self.w_k, self.w_v, self.w_out))

        conv_flops = 0
        if self.use_short_conv:
            conv_flops = 2 * self.conv_size * (self.key_dim + self.key_dim + self.value_dim)

        # Ungated linear-attention recurrent computation per token:
        # - Outer product k ⊗ v: n_v_heads * head_k_dim * head_v_dim
        # - Query-state matmul:   n_v_heads * head_k_dim * head_v_dim
        # (no state decay, no beta scaling -- that's the ablated part)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 2 * state_size

        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("linear_attention")
@dataclass
class LinearAttentionConfig(SequenceMixerConfig[LinearAttention]):
    """
    Configuration for :class:`LinearAttention`.

    See :class:`LinearAttention` for a description of the configuration options.
    """

    n_heads: int = 16
    n_v_heads: Optional[int] = None
    head_dim: Optional[int] = None
    expand_v: float = 2.0
    conv_size: int = 4
    conv_bias: bool = False
    use_short_conv: bool = True
    qk_l2norm: bool = True
    normalize: bool = False
    norm_eps: float = 1e-5
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the LinearAttention module will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_out
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += value_dim * d_model  # w_out

        # Short convolutions (kernel_size * hidden_size for each)
        if self.use_short_conv:
            params += self.conv_size * key_dim  # q_conv1d
            params += self.conv_size * key_dim  # k_conv1d
            params += self.conv_size * value_dim  # v_conv1d
            if self.conv_bias:
                params += key_dim  # q_conv1d bias
                params += key_dim  # k_conv1d bias
                params += value_dim  # v_conv1d bias

        # Output RMSNorm (weight only, no bias)
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
    ) -> LinearAttention:
        """
        Build the LinearAttention module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return LinearAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            use_short_conv=self.use_short_conv,
            qk_l2norm=self.qk_l2norm,
            normalize=self.normalize,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
