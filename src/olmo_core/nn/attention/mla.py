import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.backend import AttentionBackend, AttentionBackendName
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.layer_norm import LayerNorm, LayerNormConfig, LayerNormType
from olmo_core.nn.rope import ComplexRotaryEmbedding, RoPEConfig, RotaryEmbedding

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "MLAConfig",
    "MultiheadLatentAttention",
]

log = logging.getLogger(__name__)


def _default_mla_norm() -> LayerNormConfig:
    # DeepSeek applies a (bias-free) RMSNorm to the compressed latent(s).
    return LayerNormConfig(name=LayerNormType.rms, bias=False)


@SequenceMixerConfig.register("mla")
@dataclass
class MLAConfig(SequenceMixerConfig["MultiheadLatentAttention"]):
    """
    Configuration for :class:`MultiheadLatentAttention` (DeepSeek Multi-head Latent Attention).

    See :class:`MultiheadLatentAttention` for a full description of the mechanism and of each
    configuration option.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    kv_lora_rank: int = 512
    """
    The dimensionality of the compressed joint key/value latent vector (the width of the
    down-projection ``W_DKV``). This is the quantity that is cached per token at inference time,
    which is what makes MLA's KV cache small.
    """
    q_lora_rank: Optional[int] = None
    """
    The dimensionality of the compressed query latent. If ``None`` (the default) the queries are
    projected directly from ``d_model`` without low-rank compression. DeepSeek-V2/V3 set this to a
    non-``None`` value for the larger models to save query-projection parameters.
    """
    qk_nope_head_dim: int = 128
    """
    The per-head width of the "no-position" (non-RoPE) part of the query/key, reconstructed from
    the compressed latent.
    """
    qk_rope_head_dim: int = 64
    """
    The per-head width of the decoupled RoPE part of the query/key. RoPE is applied only to this
    sub-vector, which bypasses the low-rank compression. Must be even.
    """
    v_head_dim: int = 128
    """
    The per-head width of the value, reconstructed from the compressed latent.
    """
    norm: Optional[LayerNormConfig] = field(default_factory=_default_mla_norm)
    """
    The norm applied to the compressed query and key/value latents. Defaults to a bias-free
    :class:`~olmo_core.nn.layer_norm.RMSNorm`, matching DeepSeek. Set to ``None`` to disable.
    """
    rope: Optional[RoPEConfig] = field(default_factory=RoPEConfig)
    """
    The config for the decoupled RoPE applied to the ``qk_rope_head_dim`` sub-vector. If ``None``,
    no positional information is added (the decoupled dimensions are still present but not rotated).
    """
    bias: bool = False
    """
    Whether to include biases on the linear projections. DeepSeek uses no biases.
    """
    dropout: float = 0.0
    """
    The dropout probability applied inside the attention backend.
    """
    softmax_scale: Optional[float] = None
    """
    The scale applied to the attention logits. If ``None``, defaults to ``1 / sqrt(qk_head_dim)``
    where ``qk_head_dim = qk_nope_head_dim + qk_rope_head_dim``.
    """
    backend: Optional[AttentionBackendName] = None
    """
    The attention backend to use. If not set, it will be chosen automatically (``torch`` on CPU).
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the MLA module will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        bias = self.bias

        params = 0

        # Query path.
        if self.q_lora_rank is None:
            # Direct query projection: d_model -> n_heads * qk_head_dim.
            params += d_model * n_heads * qk_head_dim
            if bias:
                params += n_heads * qk_head_dim
        else:
            # Down-projection W_DQ: d_model -> q_lora_rank.
            params += d_model * self.q_lora_rank
            if bias:
                params += self.q_lora_rank
            # Up-projection W_UQ: q_lora_rank -> n_heads * qk_head_dim.
            params += self.q_lora_rank * n_heads * qk_head_dim
            if bias:
                params += n_heads * qk_head_dim
            # Query latent norm.
            if self.norm is not None:
                params += self.norm.num_params(self.q_lora_rank)

        # KV down-projection W_DKV: d_model -> kv_lora_rank + qk_rope_head_dim.
        # (The extra qk_rope_head_dim carries the decoupled RoPE key, shared across heads.)
        params += d_model * (self.kv_lora_rank + self.qk_rope_head_dim)
        if bias:
            params += self.kv_lora_rank + self.qk_rope_head_dim

        # KV latent norm (applied to the compressed latent only, not the decoupled RoPE key).
        if self.norm is not None:
            params += self.norm.num_params(self.kv_lora_rank)

        # KV up-projection W_UKV: kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim).
        params += self.kv_lora_rank * n_heads * (self.qk_nope_head_dim + self.v_head_dim)
        if bias:
            params += n_heads * (self.qk_nope_head_dim + self.v_head_dim)

        # Output projection: n_heads * v_head_dim -> d_model.
        params += n_heads * self.v_head_dim * d_model
        if bias:
            params += d_model

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "MultiheadLatentAttention":
        """
        Build the :class:`MultiheadLatentAttention` module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache, reused for RoPE/backend buffers.
        """
        del layer_idx, n_layers  # unused

        try:
            return MultiheadLatentAttention(
                d_model=d_model,
                n_heads=self.n_heads,
                kv_lora_rank=self.kv_lora_rank,
                q_lora_rank=self.q_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                norm=self.norm,
                rope=self.rope,
                bias=self.bias,
                dropout=self.dropout,
                softmax_scale=self.softmax_scale,
                backend=self.backend,
                dtype=self.dtype.as_pt(),
                init_device=init_device,
                cache=cache,
            )
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.__class__.__name__}', {e}"
            ) from e


class MultiheadLatentAttention(SequenceMixer):
    """
    An implementation of `Multi-head Latent Attention (MLA)
    <https://arxiv.org/abs/2405.04434>`_ from DeepSeek-V2/V3, registered as a
    :class:`~olmo_core.nn.attention.base.SequenceMixer` under the name ``"mla"``.

    MLA replaces the standard per-head key/value projections with a *low-rank joint compression*.
    A single down-projection ``W_DKV`` maps the input to a small latent vector ``c_kv`` of width
    ``kv_lora_rank``, and an up-projection ``W_UKV`` reconstructs the per-head keys and values from
    that latent. At inference time only ``c_kv`` (plus the small decoupled RoPE key) needs to be
    cached, so the KV cache is roughly ``kv_lora_rank + qk_rope_head_dim`` numbers per token instead
    of ``2 * n_kv_heads * head_dim`` — the legible payoff of the mechanism.

    **Decoupled RoPE.** Rotary embeddings cannot be applied to the compressed latent directly
    (the up-projection would mix positions across the rank), so MLA carries position information on
    a *separate* sub-vector. Queries and keys are each formed of two parts concatenated along the
    head dimension:

    - a "nope" part of width ``qk_nope_head_dim`` reconstructed from the low-rank latent, which
      carries no positional information, and
    - a "rope" part of width ``qk_rope_head_dim`` that bypasses the compression and receives RoPE.

    The decoupled key is a single shared head (produced alongside ``c_kv`` by ``W_DKV``) that is
    broadcast across all query heads, following DeepSeek. The value head width ``v_head_dim`` may
    differ from the query/key head width ``qk_nope_head_dim + qk_rope_head_dim``.

    Queries may optionally be compressed the same way via ``q_lora_rank``; if it is ``None`` the
    queries are projected directly.

    .. note::
        This module implements the training forward path. An optimized inference path that caches
        the compressed latent ``c_kv`` (the whole point of MLA) is future work: :meth:`forward`
        currently reconstructs the full per-head keys/values on every call and does not accept a
        KV cache. The core kwargs used during training (intra-document masking via ``cu_doc_lens``
        / ``max_doc_len`` and pre-sharded RoPE buffers) are supported; ``cache_leftpad`` is not.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param kv_lora_rank: The width of the compressed joint key/value latent.
    :param q_lora_rank: The width of the compressed query latent, or ``None`` to project queries
        directly from ``d_model``.
    :param qk_nope_head_dim: The per-head width of the non-RoPE query/key part.
    :param qk_rope_head_dim: The per-head width of the decoupled RoPE query/key part. Must be even.
    :param v_head_dim: The per-head width of the value.
    :param norm: The norm applied to the compressed latent(s), or ``None`` to disable.
    :param rope: The RoPE config for the decoupled part, or ``None`` for no positional information.
    :param bias: Include biases with linear layers.
    :param dropout: Dropout probability applied inside the attention backend.
    :param softmax_scale: Override for the attention logit scale. Defaults to
        ``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)``.
    :param backend: The attention backend to use. Defaults to ``torch`` on CPU.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    :param cache: Optional buffer cache reused for RoPE/backend buffers.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int] = None,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        norm: Optional[LayerNormConfig] = None,
        rope: Optional[RoPEConfig] = None,
        bias: bool = False,
        dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        backend: Optional[AttentionBackendName] = None,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ):
        super().__init__()

        if qk_rope_head_dim % 2 != 0:
            raise OLMoConfigurationError(
                f"'qk_rope_head_dim' must be even for RoPE (got {qk_rope_head_dim})"
            )
        for name, value in (
            ("kv_lora_rank", kv_lora_rank),
            ("qk_nope_head_dim", qk_nope_head_dim),
            ("qk_rope_head_dim", qk_rope_head_dim),
            ("v_head_dim", v_head_dim),
        ):
            if value <= 0:
                raise OLMoConfigurationError(f"'{name}' must be positive (got {value})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # Query path: either a direct projection or a low-rank (down, norm, up) compression.
        self.q_norm: Optional[LayerNorm] = None
        if q_lora_rank is None:
            self.w_q = nn.Linear(
                d_model, n_heads * self.qk_head_dim, bias=bias, dtype=dtype, device=init_device
            )
        else:
            self.w_dq = nn.Linear(d_model, q_lora_rank, bias=bias, dtype=dtype, device=init_device)
            self.w_uq = nn.Linear(
                q_lora_rank, n_heads * self.qk_head_dim, bias=bias, dtype=dtype, device=init_device
            )
            if norm is not None:
                self.q_norm = norm.build(size=q_lora_rank, init_device=init_device)

        # Joint KV down-projection. Produces the compressed latent ``c_kv`` (kv_lora_rank) plus the
        # single-head decoupled RoPE key (qk_rope_head_dim).
        self.w_dkv = nn.Linear(
            d_model,
            kv_lora_rank + qk_rope_head_dim,
            bias=bias,
            dtype=dtype,
            device=init_device,
        )
        self.kv_norm: Optional[LayerNorm] = None
        if norm is not None:
            self.kv_norm = norm.build(size=kv_lora_rank, init_device=init_device)

        # KV up-projection: reconstruct the per-head non-RoPE keys and the values from ``c_kv``.
        self.w_ukv = nn.Linear(
            kv_lora_rank,
            n_heads * (qk_nope_head_dim + v_head_dim),
            bias=bias,
            dtype=dtype,
            device=init_device,
        )

        self.w_out = nn.Linear(
            n_heads * v_head_dim, d_model, bias=bias, dtype=dtype, device=init_device
        )

        # Decoupled RoPE, built for the small ``qk_rope_head_dim`` sub-vector.
        self.rope: Optional[Union[RotaryEmbedding, ComplexRotaryEmbedding]] = None
        if rope is not None:
            if rope.name == "fused":
                raise OLMoConfigurationError(
                    f"fused RoPE is not compatible with {self.__class__.__name__}"
                )
            rope_module = rope.build(self.qk_rope_head_dim, cache=cache)
            assert isinstance(rope_module, (RotaryEmbedding, ComplexRotaryEmbedding))
            self.rope = rope_module

        # Reuse the shared SDPA backend. MLA reconstructs full (non-grouped) per-head keys and
        # values, so ``n_kv_heads == n_heads``. The value head dim may differ from the query/key
        # head dim, which the SDPA backends handle. Default to the torch backend, which works
        # everywhere; other backends are chosen only when explicitly requested.
        if backend is not None:
            backend = AttentionBackendName(backend)
        else:
            backend = AttentionBackendName.torch
        if not torch.cuda.is_available() and backend != AttentionBackendName.torch:
            warnings.warn(
                f"Backend is set to {backend}, but GPUs are not available. Defaulting to torch."
            )
            backend = AttentionBackendName.torch
        backend.assert_supported()
        log.info(f"Using attention backend '{backend}'")
        self.backend: AttentionBackend = backend.build(
            head_dim=self.qk_head_dim,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            scale=softmax_scale,
            dropout_p=dropout,
            cache=cache,
        )

    @property
    def cp_enabled(self) -> bool:
        return self.backend.cp_enabled

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cu_doc_lens: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None
        rope_kwargs = {}
        if cu_doc_lens is not None:
            if not isinstance(self.rope, RotaryEmbedding):
                raise NotImplementedError(
                    "Intra-document RoPE (cu_doc_lens) is only supported by RotaryEmbedding; "
                    f"got {type(self.rope).__name__}"
                )
            rope_kwargs["cu_doc_lens"] = cu_doc_lens
        return self.rope(
            q,
            k,
            head_first=False,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            **rope_kwargs,
        )

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
    ) -> torch.Tensor:
        # shape: (batch_size, seq_len, n_heads, v_head_dim)
        return self.backend(
            (q, k, v),
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply multi-head latent attention to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).
            Required together with ``max_doc_len`` when using intra-document masking.
        :param max_doc_len: The maximum document length in the input ``x``.
            Required together with ``cu_doc_lens`` when using intra-document masking.

        :returns: The output of attention with shape ``(batch_size, seq_len, d_model)``.
        """
        if cache_leftpad is not None:
            raise NotImplementedError(
                f"cache_leftpad (inference KV caching) is not supported by {self.__class__.__name__}"
            )

        B, T, _ = x.shape

        # Query path -> (batch_size, seq_len, n_heads, qk_head_dim).
        if self.q_lora_rank is None:
            q = self.w_q(x)
        else:
            c_q = self.w_dq(x)
            if self.q_norm is not None:
                c_q = self.q_norm(c_q)
            q = self.w_uq(c_q)
        q = q.view(B, T, self.n_heads, self.qk_head_dim)
        # Split into the non-RoPE ("nope") and decoupled-RoPE parts.
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Joint KV compression -> compressed latent ``c_kv`` plus the decoupled RoPE key.
        compressed = self.w_dkv(x)
        c_kv, k_pe = torch.split(compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        if self.kv_norm is not None:
            c_kv = self.kv_norm(c_kv)

        # Reconstruct per-head non-RoPE keys and values from the latent.
        kv = self.w_ukv(c_kv).view(B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        # shape: (batch_size, seq_len, n_heads, qk_nope_head_dim), (..., v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # The decoupled RoPE key is a single shared head.
        # shape: (batch_size, seq_len, 1, qk_rope_head_dim)
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim)

        if self.rope is not None:
            if self.cp_enabled and pos_sin is None and pos_cos is None and freqs_cis is None:
                raise RuntimeError(
                    "RoPE buffers must be passed through to attention after being properly "
                    "sharded by the context parallel load balancer"
                )
            q_pe, k_pe = self._apply_rope(q_pe, k_pe, pos_sin, pos_cos, freqs_cis, cu_doc_lens)

        # Broadcast the single decoupled key head across all query heads.
        # shape: (batch_size, seq_len, n_heads, qk_rope_head_dim)
        k_pe = k_pe.expand(B, T, self.n_heads, self.qk_rope_head_dim)

        # Assemble the full query/key by concatenating the nope and rope parts.
        # shape: (batch_size, seq_len, n_heads, qk_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # shape: (batch_size, seq_len, n_heads, v_head_dim)
        att = self.sdpa(
            q,
            k,
            v,
            cu_doc_lens=cu_doc_lens,
            cu_doc_lens_q=cu_doc_lens_q,
            cu_doc_lens_k=cu_doc_lens_k,
            max_doc_len=max_doc_len,
            max_doc_len_q=max_doc_len_q,
            max_doc_len_k=max_doc_len_k,
            local_k_slice=local_k_slice,
        )

        # shape: (batch_size, seq_len, n_heads * v_head_dim)
        att = att.reshape(B, T, self.n_heads * self.v_head_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(att)

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
            "Tensor parallelism is not yet implemented for MultiheadLatentAttention"
        )

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly
        raise NotImplementedError(
            "Context parallelism is not yet implemented for MultiheadLatentAttention"
        )

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

        # All "input" projections (everything except the output projection).
        input_projections = []
        if self.q_lora_rank is None:
            input_projections.append(self.w_q)
        else:
            input_projections.extend([self.w_dq, self.w_uq])
        input_projections.extend([self.w_dkv, self.w_ukv])

        if init_method == InitMethod.fan_in:
            for w in input_projections:
                init_linear(w, std=w.in_features**-0.5, generator=generator)
        else:
            if init_method == InitMethod.normalized:
                std = d_model**-0.5
            for w in input_projections:
                init_linear(w, std=std, generator=generator)

        # Output projection with the usual residual/depth scaling.
        if init_method == InitMethod.fan_in:
            std = self.w_out.in_features**-0.5
        elif init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Estimate the FLOPs per token.

        This accounts for the linear projections (down/up compression and output) via a
        6-FLOPs-per-parameter estimate, plus the attention score and value aggregation
        (``QK^T`` over ``qk_head_dim`` and ``softmax(QK^T) @ V`` over ``v_head_dim``).
        """
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward).
        param_flops = 6 * sum(p.numel() for p in self.parameters())

        # Attention computation: QK^T (qk_head_dim) and Attn @ V (v_head_dim).
        # 6x = 2 ops * 3 for forward+backward.
        attn_flops = 6 * self.n_heads * seq_len * (self.qk_head_dim + self.v_head_dim)

        return param_flops + attn_flops
