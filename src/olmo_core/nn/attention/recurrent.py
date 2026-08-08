import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_chunk_gated_delta_rule,
    dispatch_chunk_gdn2,
    has_fla,
)
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


def document_reversal_index(
    seq_len: int,
    cu_doc_lens: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    Build the index that reverses token order *within each document*.

    A linear recurrence is directional, so the only way to give one a right-to-left view is to
    run it over reversed input. Reversing the whole packed sequence would be wrong: OLMo-core
    packs many documents into one row, so a whole-row flip moves tokens across document
    boundaries and the first document's tokens end up conditioning on the last document's. This
    is the same distinction `DeltaFlow <https://arxiv.org/abs/2608.01240>`_ draws when it
    reverses only the *body* tokens and leaves its prefix in place.

    The returned index is an involution -- applying it twice is the identity -- which is what
    lets the caller use one index to reverse on the way in and un-reverse on the way out.
    Document spans are unchanged by the reversal, so ``cu_doc_lens`` stays valid for the
    convolutions and the kernel downstream.

    :param seq_len: The (packed) sequence length to build the index over.
    :param cu_doc_lens: Cumulative document lengths, a 1D tensor with one more element than
        there are documents, whose first element is ``0``. When ``None`` there is no
        intra-document structure to respect and the whole sequence is reversed.
    :param device: The device to build the index on.

    :returns: A 1D ``int64`` tensor of length ``seq_len``.
    """
    pos = torch.arange(seq_len, device=device)
    if cu_doc_lens is None:
        return pos.flip(0)

    # `cu_doc_lens[1:]` are the exclusive ends, so the number of ends at or below a position is
    # that position's document index. `right=True` puts a position that equals an end into the
    # *next* document, which is what an exclusive end means.
    doc = torch.searchsorted(cu_doc_lens[1:].contiguous(), pos, right=True)
    start = cu_doc_lens[doc].long()
    end = cu_doc_lens[doc + 1].long()
    # Reflect each position through the midpoint of its own document.
    return start + end - 1 - pos


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
        self.conv_size = conv_size

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

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=0, b=16, generator=generator).log())
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
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
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward), which is the convention every
        # other module here uses -- see `Attention`, `FeedForward` and `LMHead`. It covers the
        # short convolutions exactly as well as the projections: a depthwise conv does one
        # multiply-add per (channel, tap) per output, so its 2 * kernel_size * channels forward
        # FLOPs are 2 per parameter, the same ratio as a matmul.
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # Gated delta rule recurrent computation per token, on a state of shape
        # (n_v_heads, head_k_dim, head_v_dim):
        # - Outer product k ⊗ v
        # - State decay
        # - Beta scaling
        # - Query-state matmul
        # 24x multiplier: 4 passes * 2 ops each * 3 for forward+backward
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 24 * state_size

        return int(param_flops + recurrent_flops)


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
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class GatedDeltaNet2(SequenceMixer):
    """
    The layer implementation for `Gated DeltaNet-2: Decoupling Erase and Write in Linear
    Attention <https://arxiv.org/abs/2605.22791>`_.

    Modelled on the reference layer in ``fla.layers.gdn2``, itself adapted from
    https://github.com/NVlabs/GatedDeltaNet-2, but written against OLMo-core's
    :class:`~olmo_core.nn.convolution.CausalConv1d` and init/parallelism conventions so that it
    lines up with :class:`GatedDeltaNet`.

    :class:`GatedDeltaNet` drives both halves of the state edit from a single scalar
    ``beta_t`` per head: how much of the old key direction to erase, and how much of the new
    value to write. GDN-2 splits those into two independent channel-wise gates -- an erase gate
    ``b_t`` over the ``K`` (key) axis and a write gate ``w_t`` over the ``V`` (value) axis -- on
    top of KDA's channel-wise decay ``g_t``. The recurrence on the matrix state
    ``S in R^{K x V}`` is

    .. math::
        S_t = \\left(I - k_t (b_t \\odot k_t)^\\top\\right) \\mathrm{diag}(\\exp(g_t))\\, S_{t-1}
              + k_t (w_t \\odot v_t)^\\top

    Collapsing ``b_t`` and ``w_t`` to a shared scalar recovers KDA exactly; collapsing the decay
    to a scalar as well recovers :class:`GatedDeltaNet`.

    .. note::
        The decay is per key *channel* rather than per head, so ``dt_bias`` has ``key_dim``
        entries here where :class:`GatedDeltaNet` has ``n_v_heads``, and the decay
        pre-activation comes from a low-rank bottleneck through ``head_v_dim`` rather than a
        single ``d_model -> n_v_heads`` projection.

    :param d_model: The model hidden size.
    :param n_heads: The number of QK heads.
    :param n_v_heads: The number of value heads. If ``None``, defaults to ``n_heads``.
        GVA is applied if ``n_v_heads`` > ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 1.0, which is the ratio the
        GDN-2 reference implementation ships with.
    :param allow_neg_eigval: Allow negative eigenvalues, by widening the *erase* gate ``b`` from
        ``[0, 1]`` to ``[0, 2]``. The write gate ``w`` is left alone: the sign effect concerns the
        state transition, not the magnitude of what is written. Default: ``False``, unlike
        :class:`GatedDeltaNet` -- the GDN-2 paper's headline model keeps ``b`` in ``[0, 1]``, and
        its Table 5 ablation finds the widened range gives no consistent gain at 1.3B. See
        reference: `Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues
        <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param reverse_scan: Run the recurrence right-to-left within each document instead of
        left-to-right. This is what makes a bidirectional stack possible: alternate the flag
        across layers and the stack sees both directions, which is DeltaFlow's *alternating
        scan* variant. Costs nothing over the causal layer -- it is one scan either way.
        Default: ``False``, so an unconfigured layer is the causal one, unchanged.
    :param noise_conditioned: Condition the decay and gate logits on a diffusion noise level
        passed to :meth:`forward` as ``noise_level``. Required for diffusion training; see the
        note below. Default: ``False``.
    :param noise_embed_dim: Width of the sinusoidal noise-level embedding, used only when
        ``noise_conditioned`` is set. Default: 64.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.

    .. note::
        **On bidirectionality and diffusion.** A causal recurrence cannot carry a masked
        diffusion objective: DeltaFlow reports that unidirectional GDN "suffers from entropy
        collapse". ``reverse_scan`` supplies the missing direction, and the short convolutions
        need no change to follow -- fed reversed input, a
        :class:`~olmo_core.nn.convolution.CausalConv1d` is an anti-causal convolution in the
        original coordinates, which is the pairing DiffuMamba describes.

        DeltaFlow also finds the bidirectional core *alone* insufficient: it lowers perplexity
        but "still produces overly concentrated generations", and restoring diversity needs the
        decay to know how corrupted its input is. That is ``noise_conditioned``. The projections
        are zero-initialised, so at initialisation the layer is bit-exactly the
        noise-independent one and the conditioning is something the model learns to use rather
        than something imposed on it.
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
        norm_eps: float = 1e-5,
        reverse_scan: bool = False,
        noise_conditioned: bool = False,
        noise_embed_dim: int = 64,
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
        self.reverse_scan = reverse_scan
        self.noise_conditioned = noise_conditioned
        self.noise_embed_dim = noise_embed_dim

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

        # Decay pre-activation, through a low-rank bottleneck: d_model -> head_v_dim -> key_dim.
        self.w_a = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False, dtype=dtype, device=init_device),
        )
        # The two gates GDN-2 decouples: erase over the K axis, write over the V axis.
        self.w_b = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_w = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)

        # Per-QK-head decay rate, and a per-key-channel softplus bias.
        self.A_log = nn.Parameter(torch.empty(self.n_heads, dtype=dtype, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=dtype, device=init_device))

        # Noise-adaptive memory control. DeltaFlow shifts two logits -- its decay `a` and its
        # write rate `beta` -- by a per-head function of the noise level. GDN-2 has decoupled
        # `beta` into an erase gate over the key axis and a write gate over the value axis, so
        # `beta`'s single shift becomes two here and all three are declared together.
        #
        # These are the only parameters in the layer deliberately initialised to zero, and
        # `reset_parameters` leaves them alone for that reason.
        if noise_conditioned:
            if noise_embed_dim % 2 != 0:
                raise OLMoConfigurationError(
                    f"noise_embed_dim must be even, since the sinusoidal embedding splits it "
                    f"into a cosine and a sine half (got {noise_embed_dim})"
                )
            self.u_a = nn.Linear(
                noise_embed_dim, self.n_heads, bias=True, dtype=dtype, device=init_device
            )
            self.u_b = nn.Linear(
                noise_embed_dim, self.n_heads, bias=True, dtype=dtype, device=init_device
            )
            self.u_w = nn.Linear(
                noise_embed_dim, self.n_v_heads, bias=True, dtype=dtype, device=init_device
            )
            for proj in (self.u_a, self.u_b, self.u_w):
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)

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

        # Output gate, also a low-rank bottleneck, and unlike the others it carries a bias.
        self.w_g = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True, dtype=dtype, device=init_device),
        )
        # `swish` (== SiLU), which is FusedRMSNormGated's default and so is spelled by omission
        # here, the same way GatedDeltaNet spells it. Section 3.5 of the paper: "The recurrent
        # output is RMS-normalized, multiplied by a separate SiLU output gate, and projected back
        # to the model dimension", and NVIDIA's reference layer uses FusedRMSNormSwishGate.
        # `fla.layers.gdn2` passes activation="sigmoid" instead, which is a deviation from both;
        # the two activations are not interchangeable, so this follows the paper.
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, device=init_device)  # type: ignore
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def _noise_embedding(self, noise_level: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Sinusoidally embed a per-sequence noise level.

        :param noise_level: The noise level per sequence, shape ``(batch_size,)``.
        :param dtype: The dtype to return, matching whichever projection consumes it.

        :returns: Shape ``(batch_size, noise_embed_dim)``.
        """
        half = self.noise_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=noise_level.device, dtype=torch.float32)
            / half
        )
        args = noise_level.float().reshape(-1, 1) * freqs.reshape(1, -1)
        return torch.cat([args.cos(), args.sin()], dim=-1).to(dtype)

    def _noise_shift(
        self, proj: nn.Linear, noise_level: torch.Tensor, per_head_dim: int
    ) -> torch.Tensor:
        """
        Per-head logit shift from the noise level, broadcast to a channel-wise logit's shape.

        :param proj: The zero-initialised projection for this logit.
        :param noise_level: The noise level per sequence, shape ``(batch_size,)``.
        :param per_head_dim: How many channels each head owns in the logit being shifted.

        :returns: Shape ``(batch_size, 1, n_heads * per_head_dim)``, broadcasting over sequence.
        """
        emb = self._noise_embedding(noise_level, dtype=proj.weight.dtype)
        return proj(emb).repeat_interleave(per_head_dim, dim=-1).unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply GDN-2 sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
        :param noise_level: The diffusion noise level per sequence, shape ``(batch_size,)``.
            Required when the layer was built with ``noise_conditioned``, ignored otherwise.

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.

        :raises RuntimeError: If ``noise_conditioned`` was set and no ``noise_level`` arrived,
            or if a reversed scan is combined with context parallelism.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T_og, _ = x.shape

        if self.noise_conditioned and noise_level is None:
            raise RuntimeError(
                "this GatedDeltaNet2 was built with noise_conditioned=True but forward() got no "
                "noise_level, so the decay and gates would silently fall back to the "
                "noise-independent recurrence -- which is a diffusion run quietly training the "
                "wrong model rather than a slower one"
            )

        # A reversed scan needs the whole document in one place to reflect it. Under context
        # parallelism each rank holds a slice, so the reflection would run inside the slice and
        # produce something that is neither direction. Refuse rather than compute it.
        if self.reverse_scan and self.cp_enabled:
            raise RuntimeError(
                "reverse_scan is not supported together with context parallelism: each rank "
                "holds a slice of the sequence, so reversing within the slice is not reversing "
                "within the document"
            )

        rev_index: Optional[torch.Tensor] = None
        if self.reverse_scan:
            # One index reverses on the way in and un-reverses on the way out, because
            # `document_reversal_index` is an involution. Everything between these two calls --
            # projections, convolutions, the kernel -- runs in reversed coordinates, which is
            # what turns the CausalConv1d into an anti-causal one in the original order.
            rev_index = document_reversal_index(T_og, cu_doc_lens, x.device)
            x = x.index_select(1, rev_index)

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_v_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Channel-wise gates, both squashed to [0, 1]. `b` erases along K, `w` writes along V.
        b_logit = self.w_b(x)
        w_logit = self.w_w(x)
        a_logit = self.w_a(x).float() + self.dt_bias
        if self.noise_conditioned:
            assert noise_level is not None
            b_logit = b_logit + self._noise_shift(self.u_b, noise_level, self.head_k_dim)
            w_logit = w_logit + self._noise_shift(self.u_w, noise_level, self.head_v_dim)
            a_logit = a_logit + self._noise_shift(self.u_a, noise_level, self.head_k_dim).float()

        b = b_logit.sigmoid()
        if self.allow_neg_eigval:
            b = b * 2.0
        w = w_logit.sigmoid()

        # Channel-wise log-decay in fp32, kept flat over `key_dim`. The per-head A_log rate is
        # folded in here, *before* any context-parallel exchange: after the exchange only
        # n_heads/CP heads are local, so a per-head multiply would have to be sliced to match.
        # shape: (batch_size, seq_len, n_heads * head_k_dim)
        g = -self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(a_logit)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T_local, C] -> [B, T_total, C/CP]
            q, k, g, b = all_to_all_cp2hp([q, k, g, b], self._cp_group)
            v = all_to_all_single_cp2hp(v, self._cp_group)
            w = all_to_all_single_cp2hp(w, self._cp_group)

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        T = q.size(1)
        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        g = g.view(B, T, -1, self.head_k_dim)
        b = b.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)
        w = w.view(B, T, -1, self.head_v_dim)

        if self.n_v_heads > self.n_heads:
            # GVA: broadcast every QK-side tensor across the value-head groups.
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)
            g = g.repeat_interleave(repeat_factor, dim=-2)
            b = b.repeat_interleave(repeat_factor, dim=-2)

        o, _ = dispatch_chunk_gdn2(
            q=q, k=k, v=v, g=g, b=b, w=w, cu_seqlens=cu_doc_lens, use_qk_l2norm_in_kernel=True
        )

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            o = all_to_all_single_hp2cp(o, self._cp_group)

        g_out = self.w_g(x).view(B, T_og, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        out = self.w_out(self.o_norm(o, g_out).view(B, T_og, -1))

        if rev_index is not None:
            # Back to the original token order, so the residual stream this returns into is in
            # the same coordinates every other layer reads.
            out = out.index_select(1, rev_index)

        return out

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for GatedDeltaNet2")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        if ring is not None:
            raise NotImplementedError(
                "Ring context parallelism is not supported for GatedDeltaNet2"
            )
        assert uly is not None

        cp_world_size = cp_mesh.size()
        if cp_world_size == 1:
            return

        # Ulysses CP requires divisibility by CP world size for:
        # 1. n_heads and n_v_heads - for head partitioning in the recurrent kernel
        # 2. key_dim and value_dim - for channel partitioning in the conv layers and in the
        #    channel-wise gates, which are exchanged as flat channel tensors
        assert self.n_heads % cp_world_size == 0
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

        # `fan_in` is std = 1/sqrt(d_in) read off each layer's own input width, which is what it
        # means everywhere else in `transformer/init.py`. It is implemented here rather than
        # refused because MuonH needs it: Hyperball fixes each constrained matrix's radius at
        # R = ||W_0||_F, so the initialiser sets the absolute step length and a run on a
        # different one is not running the method the paper describes.
        fan_in = init_method == InitMethod.fan_in

        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        def linear_std(module: nn.Linear) -> float:
            return module.in_features**-0.5 if fan_in else std

        for w in (self.w_q, self.w_k, self.w_v, self.w_b, self.w_w):
            init_linear(w, std=linear_std(w), generator=generator)
        # `w_a` and `w_g` are two-layer bottlenecks, so init each leaf rather than the container.
        # Under `fan_in` the two leaves get different widths, which is the point of asking each.
        for seq in (self.w_a, self.w_g):
            for w in seq:
                init_linear(w, std=linear_std(w), generator=generator)
        for w in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            # Depthwise (`groups=hidden_size`), so an output channel sees `conv_size` inputs and
            # not the full width.
            init_linear(w, std=self.conv_size**-0.5 if fan_in else std, generator=generator)

        # `u_a`, `u_b` and `u_w` are deliberately absent from every loop above. They are the
        # noise-conditioning shifts and must stay at the zeros `__init__` gave them, so that an
        # untrained model is exactly the noise-independent recurrence.

        # Uniform on (1, 16) rather than (0, 16): log(0) is -inf, and exp(A_log) is the decay
        # rate, so a zero sample would silently freeze one head's state.
        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1, b=16, generator=generator).log())
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias.copy_(inv_dt)

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=linear_std(self.w_out), generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for GDN-2.

        This accounts for:
        - Linear projections (w_q, w_k, w_v, w_a, w_b, w_w, w_g, w_out)
        - Short convolutions (q, k, v)
        - The GDN-2 recurrent computation
        - Gated RMS normalization
        """
        del seq_len
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward), the convention `Attention`,
        # `FeedForward` and `LMHead` all use. Covering every parameter rather than an enumerated
        # list of projections is also what keeps the two low-rank bottlenecks (`w_a`, `w_g`) and
        # the short convolutions counted without naming their leaves: a depthwise conv does one
        # multiply-add per (channel, tap) per output, which is 2 forward FLOPs per parameter --
        # the same ratio as a matmul.
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # GDN-2 recurrent computation per token, on a state of shape
        # (n_v_heads, head_k_dim, head_v_dim):
        # - Channel-wise decay diag(exp(g)) S: one pass over the state
        # - Erase term k (b ⊙ k)^T S: two passes (the read, then the rank-1 subtraction)
        # - Outer product k ⊗ (w ⊙ v): one pass
        # - Query-state matmul: one pass
        # 30x multiplier: 5 passes * 2 ops each * 3 for forward+backward. One pass more than
        # GatedDeltaNet, which folds erase and write into a single beta-scaled rank-1 update.
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 30 * state_size

        return int(param_flops + recurrent_flops)


@SequenceMixerConfig.register("gated_delta_net_2")
@dataclass
class GatedDeltaNet2Config(SequenceMixerConfig[GatedDeltaNet2]):
    """
    Configuration for :class:`GatedDeltaNet2`.

    See :class:`GatedDeltaNet2` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of QK heads.
    """
    n_v_heads: Optional[int] = None
    """
    The number of value heads. If ``None``, defaults to ``n_heads``.
    If ``n_v_heads`` > ``n_heads``, GVA (Grouped Value Attention) is applied.

    As with :class:`GatedDeltaNetConfig`, GVA is preferred over GQA here because the recurrent
    state has shape ``(n_v_heads, head_k_dim, head_v_dim)`` and does not grow with sequence
    length, so raising ``n_v_heads`` buys long-range capacity rather than costing memory.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 1.0
    """
    The expansion ratio for the value dimension (``head_v_dim = head_dim * expand_v``).

    Defaults to 1.0, which is what the GDN-2 reference implementation ships with -- note that
    :class:`GatedDeltaNetConfig` defaults to 2.0, so the two are not parameter-matched at their
    respective defaults.
    """
    allow_neg_eigval: bool = False
    """
    Allow negative eigenvalues in the recurrent dynamics, by widening the erase gate ``b`` from
    ``[0, 1]`` to ``[0, 2]``. The write gate ``w`` is left alone.

    Defaults to ``False`` where :class:`GatedDeltaNetConfig` defaults to ``True``, and the
    difference is the paper rather than an oversight: the GDN-2 headline model keeps ``b`` in
    ``[0, 1]``, and Table 5 reports the widened range as an ablation with no consistent gain at
    1.3B (15.95 vs 15.90 WikiText ppl, 53.04 vs 53.11 common-sense average).
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
    reverse_scan: bool = False
    """
    Run the recurrence right-to-left within each document rather than left-to-right.

    Alternating this across the GDN-2 layers of a stack is DeltaFlow's *alternating scan*, and is
    how a stack of linear recurrences comes to see both directions at all. It costs nothing over
    the causal layer, because it is still one scan per layer. Not usable with context
    parallelism, which splits the sequence a reversal needs whole.
    """
    noise_conditioned: bool = False
    """
    Condition the decay and both gates on a diffusion noise level supplied to
    :meth:`GatedDeltaNet2.forward` as ``noise_level``.

    Set this for diffusion training. DeltaFlow finds a bidirectional core alone lowers perplexity
    but "still produces overly concentrated generations": the decay has to know how corrupted its
    input is before it can tell a correction from noise. The projections are zero-initialised, so
    this adds nothing to the function computed at initialisation.
    """
    noise_embed_dim: int = 64
    """
    Width of the sinusoidal noise-level embedding. Must be even. Only read when
    ``noise_conditioned`` is set.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the GatedDeltaNet2 will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_v_heads = self.n_v_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_b, w_w, w_out
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += d_model * key_dim  # w_b (erase gate, K axis)
        params += d_model * value_dim  # w_w (write gate, V axis)
        params += value_dim * d_model  # w_out

        # The two low-rank bottlenecks. Only w_g's second layer carries a bias.
        params += d_model * head_v_dim + head_v_dim * key_dim  # w_a (decay)
        params += d_model * head_v_dim + head_v_dim * value_dim + value_dim  # w_g (output gate)

        # A_log is per QK head; dt_bias is per key channel.
        params += n_heads  # A_log
        params += key_dim  # dt_bias

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

        # Noise-conditioning shifts: one per-head bias-carrying projection for the decay, the
        # erase gate and the write gate. Counted because they are parameters, negligible because
        # they are `noise_embed_dim x n_heads` and nothing else.
        if self.noise_conditioned:
            params += 2 * (self.noise_embed_dim * n_heads + n_heads)  # u_a, u_b
            params += self.noise_embed_dim * n_v_heads + n_v_heads  # u_w

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> GatedDeltaNet2:
        """
        Build the GatedDeltaNet2 module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return GatedDeltaNet2(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            reverse_scan=self.reverse_scan,
            noise_conditioned=self.noise_conditioned,
            noise_embed_dim=self.noise_embed_dim,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
