"""
xLSTM[1:0] (mLSTM-only) as an OLMo-core sequence mixer.

This implements the optimized post-up-projection mLSTM layer used by xLSTM 7B:
dense q/k/v projections, per-head exponential input and forget gates, an
output gate, head-wise LayerNorm, and a dense output projection. The
parallel matrix-memory recurrence is provided by the official
``mlstm-kernels`` package.

The module intentionally omits the channel-wise convolution and learnable skip
connection from the original pre-up-projection xLSTM architecture. Those
operations were removed in the optimized xLSTM 7B architecture, whose block
layout already matches OLMo's sequence-mixer-plus-SwiGLU transformer block.

Defined as an importable module (rather than under ``__main__``) so OLMo-core
can reconstruct its registered config when resuming a checkpoint.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
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

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


MLSTM_KERNELS_VERSION = "2.0.4"
CHECKPOINT_LAYOUT = "olmo-fused-v1"
OFFICIAL_CHECKPOINT_LAYOUT = "xlstm-large-single-v1"

_OFFICIAL_MLSTM_KEYS = (
    "q.weight",
    "k.weight",
    "v.weight",
    "ogate_preact.weight",
    "igate_preact.weight",
    "fgate_preact.weight",
    "igate_preact.bias",
    "fgate_preact.bias",
    "multihead_norm.weight",
    "out_proj.weight",
)
_PACKED_MLSTM_KEYS = (
    "w_qk.weight",
    "w_vo.weight",
    "w_if.weight",
    "w_if.bias",
    "o_norm.weight",
    "w_out.weight",
)


def _reject_optimizer_state(optimizer_state_dict: Mapping[str, Any] | None) -> None:
    if optimizer_state_dict is not None:
        raise NotImplementedError(
            "optimizer-state conversion is unsupported because projection packing changes "
            "parameter identity and cardinality; convert model state only and initialize a "
            "new optimizer"
        )


def _sorted_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: state_dict[key] for key in sorted(state_dict)}


def convert_mlstm_official_to_packed_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Convert official xLSTM-Large single-projection model state to OLMo packed state.

    Convolution parameters are preserved unchanged. Optimizer state cannot be
    converted because six official projection parameters become three packed
    parameters.
    """
    _reject_optimizer_state(optimizer_state_dict)
    marker = "ogate_preact.weight"
    prefixes = sorted({key[: -len(marker)] for key in state_dict if key.endswith(marker)})
    if not prefixes:
        packed = sorted(
            key for key in state_dict if any(key.endswith(suffix) for suffix in _PACKED_MLSTM_KEYS)
        )
        if packed:
            raise ValueError(
                f"expected official mLSTM checkpoint layout, found packed keys: {packed}"
            )
        raise ValueError("no official mLSTM projection layout was found")
    converted = dict(state_dict)
    for prefix in prefixes:
        required = {f"{prefix}{suffix}" for suffix in _OFFICIAL_MLSTM_KEYS}
        missing = sorted(required.difference(converted))
        if missing:
            raise ValueError(f"incomplete official mLSTM layout at '{prefix}': missing {missing}")
        packed_keys = {f"{prefix}{suffix}" for suffix in _PACKED_MLSTM_KEYS}
        collisions = sorted(packed_keys.intersection(converted))
        if collisions:
            raise ValueError(f"mixed mLSTM checkpoint layouts at '{prefix}': found {collisions}")
        unsupported_biases = sorted(
            key
            for key in (
                f"{prefix}q.bias",
                f"{prefix}k.bias",
                f"{prefix}v.bias",
                f"{prefix}ogate_preact.bias",
                f"{prefix}multihead_norm.bias",
                f"{prefix}out_proj.bias",
            )
            if key in converted
        )
        if unsupported_biases:
            raise ValueError(
                "official mLSTM projection biases have no packed OLMo counterpart: "
                f"{unsupported_biases}"
            )

        updates = {
            f"{prefix}w_qk.weight": torch.cat(
                (converted[f"{prefix}q.weight"], converted[f"{prefix}k.weight"]),
                dim=0,
            ),
            f"{prefix}w_vo.weight": torch.cat(
                (
                    converted[f"{prefix}v.weight"],
                    converted[f"{prefix}ogate_preact.weight"],
                ),
                dim=0,
            ),
            f"{prefix}w_if.weight": torch.cat(
                (
                    converted[f"{prefix}igate_preact.weight"],
                    converted[f"{prefix}fgate_preact.weight"],
                ),
                dim=0,
            ),
            f"{prefix}w_if.bias": torch.cat(
                (
                    converted[f"{prefix}igate_preact.bias"],
                    converted[f"{prefix}fgate_preact.bias"],
                ),
                dim=0,
            ),
            f"{prefix}o_norm.weight": converted[f"{prefix}multihead_norm.weight"],
            f"{prefix}w_out.weight": converted[f"{prefix}out_proj.weight"],
        }
        for key in required:
            del converted[key]
        converted.update(updates)
    return _sorted_state_dict(converted)


def convert_mlstm_packed_to_official_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    *,
    optimizer_state_dict: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert OLMo packed mLSTM model state to official xLSTM-Large projections."""
    _reject_optimizer_state(optimizer_state_dict)
    marker = "w_qk.weight"
    prefixes = sorted({key[: -len(marker)] for key in state_dict if key.endswith(marker)})
    if not prefixes:
        official = sorted(
            key
            for key in state_dict
            if any(key.endswith(suffix) for suffix in _OFFICIAL_MLSTM_KEYS)
        )
        if official:
            raise ValueError(
                f"expected packed mLSTM checkpoint layout, found official keys: {official}"
            )
        raise ValueError("no packed mLSTM projection layout was found")
    converted = dict(state_dict)
    for prefix in prefixes:
        required = {f"{prefix}{suffix}" for suffix in _PACKED_MLSTM_KEYS}
        missing = sorted(required.difference(converted))
        if missing:
            raise ValueError(f"incomplete packed mLSTM layout at '{prefix}': missing {missing}")
        official_keys = {f"{prefix}{suffix}" for suffix in _OFFICIAL_MLSTM_KEYS}
        collisions = sorted(official_keys.intersection(converted))
        if collisions:
            raise ValueError(f"mixed mLSTM checkpoint layouts at '{prefix}': found {collisions}")

        split_keys = {
            "w_qk.weight": ("q.weight", "k.weight"),
            "w_vo.weight": ("v.weight", "ogate_preact.weight"),
            "w_if.weight": ("igate_preact.weight", "fgate_preact.weight"),
            "w_if.bias": ("igate_preact.bias", "fgate_preact.bias"),
        }
        updates = {}
        for packed_suffix, official_suffixes in split_keys.items():
            tensor = converted[f"{prefix}{packed_suffix}"]
            if tensor.shape[0] % 2:
                raise ValueError(
                    f"packed mLSTM tensor '{prefix}{packed_suffix}' has odd leading dimension"
                )
            first, second = tensor.chunk(2, dim=0)
            updates[f"{prefix}{official_suffixes[0]}"] = first
            updates[f"{prefix}{official_suffixes[1]}"] = second
        updates[f"{prefix}multihead_norm.weight"] = converted[f"{prefix}o_norm.weight"]
        updates[f"{prefix}out_proj.weight"] = converted[f"{prefix}w_out.weight"]
        for key in required:
            del converted[key]
        converted.update(updates)
    return _sorted_state_dict(converted)


class _MultiHeadLayerNorm(nn.Module):
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        *,
        eps: float,
        dtype: torch.dtype,
        init_device: str,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim, dtype=dtype, device=init_device))

    def forward(self, x: torch.Tensor, *, head_first: bool = False) -> torch.Tensor:
        if head_first:
            if x.ndim != 4 or x.shape[1] != self.n_heads or x.shape[-1] != self.head_dim:
                raise ValueError(
                    f"expected shape (batch, {self.n_heads}, seq, {self.head_dim}), "
                    f"got {tuple(x.shape)}"
                )
            shape = x.shape
            x = F.group_norm(
                x.reshape(-1, self.head_dim),
                num_groups=1,
                weight=None,
                bias=None,
                eps=self.eps,
            )
            x = x.reshape(shape)
            return x * self.weight[None, :, None, :]

        if x.shape[-2:] != (self.n_heads, self.head_dim):
            raise ValueError(
                f"expected trailing shape {(self.n_heads, self.head_dim)}, "
                f"got {tuple(x.shape[-2:])}"
            )
        shape = x.shape
        x = x.reshape(-1, self.n_heads * self.head_dim)
        x = F.group_norm(
            x,
            num_groups=self.n_heads,
            weight=self.weight.reshape(-1),
            bias=None,
            eps=self.eps,
        )
        return x.reshape(shape)


class _UnavailableMLSTMBackend(nn.Module):
    def __init__(self, kernel: str, reason: str):
        super().__init__()
        self.kernel = kernel
        self.reason = reason

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            f"mLSTM kernel '{self.kernel}' is unavailable in this environment. "
            "The Triton xLSTM backend requires the CUDA training image. "
            f"Backend error: {self.reason}"
        )


def _build_mlstm_backend(
    *,
    chunkwise_kernel: str,
    chunk_size: int,
    autocast_kernel_dtype: str,
    eps: float,
    mode: str,
) -> nn.Module:
    try:
        from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig
    except ImportError as exc:
        raise ImportError("xLSTM requires Python >= 3.11 and mlstm-kernels==2.0.4") from exc

    try:
        return mLSTMBackend(
            mLSTMBackendConfig(
                chunkwise_kernel=chunkwise_kernel,
                mode=mode,
                chunk_size=chunk_size,
                return_last_states=False,
                autocast_kernel_dtype=autocast_kernel_dtype,
                eps=eps,
            )
        )
    except ValueError as exc:
        if "triton" in chunkwise_kernel and "Unknown mlstm kernel backend" in str(exc):
            # CPU-only environments do not register Triton backends. Keep meta-model
            # construction and config dry-runs available, but never silently change
            # the requested training kernel.
            return _UnavailableMLSTMBackend(chunkwise_kernel, str(exc))
        raise


class XLSTMMixer(SequenceMixer):
    """
    Optimized mLSTM sequence mixer from the xLSTM[1:0] architecture.

    :param d_model: The model hidden size.
    :param n_heads: The number of independent mLSTM matrix-memory heads.
    :param qk_dim_factor: Total q/k dimensionality relative to ``d_model``.
    :param v_dim_factor: Total value dimensionality relative to ``d_model``.
    :param conv_size: Causal depthwise-convolution kernel size for q/k features.
    :param gate_soft_cap: Soft cap applied to input- and forget-gate
        preactivations. Set to ``0`` to disable it.
    :param input_gate_bias: Initial bias for the exponential input gate.
    :param forget_gate_bias_min: Initial forget-gate bias for the first head.
    :param forget_gate_bias_max: Initial forget-gate bias for the last head.
    :param chunkwise_kernel: The ``mlstm-kernels`` training backend.
    :param chunk_size: Chunk size passed to the training backend.
    :param autocast_kernel_dtype: Internal autocast dtype used by the kernel.
    :param norm_eps: Epsilon used by the output LayerNorm and mLSTM recurrence.
    :param dtype: The parameter dtype.
    :param init_device: The device on which parameters are initialized.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 4,
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        conv_size: int = 4,
        gate_soft_cap: float = 15.0,
        input_gate_bias: float = -10.0,
        forget_gate_bias_min: float = 3.0,
        forget_gate_bias_max: float = 6.0,
        chunkwise_kernel: str = "chunkwise--triton_xl_chunk",
        chunk_size: int = 256,
        autocast_kernel_dtype: str = "bfloat16",
        norm_eps: float = 1e-6,
        checkpoint_layout: str = CHECKPOINT_LAYOUT,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        if n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {n_heads}")
        if conv_size <= 0:
            raise ValueError(f"conv_size must be positive, got {conv_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if gate_soft_cap is None or gate_soft_cap < 0:
            raise ValueError(
                f"gate_soft_cap must be non-negative; use 0 to disable it, got {gate_soft_cap}"
            )
        if qk_dim_factor <= 0 or v_dim_factor <= 0:
            raise ValueError("qk_dim_factor and v_dim_factor must be positive")
        if forget_gate_bias_min > forget_gate_bias_max:
            raise ValueError("forget_gate_bias_min must not exceed forget_gate_bias_max")
        if checkpoint_layout != CHECKPOINT_LAYOUT:
            raise ValueError(
                f"unsupported mLSTM checkpoint layout '{checkpoint_layout}'; "
                f"only '{CHECKPOINT_LAYOUT}' is implemented"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.conv_size = conv_size
        self.key_dim = int(d_model * qk_dim_factor)
        self.value_dim = int(d_model * v_dim_factor)
        self.gate_soft_cap = gate_soft_cap
        self.input_gate_bias = input_gate_bias
        self.forget_gate_bias_min = forget_gate_bias_min
        self.forget_gate_bias_max = forget_gate_bias_max
        self.chunkwise_kernel = chunkwise_kernel
        self.chunk_size = chunk_size
        self.checkpoint_layout = checkpoint_layout

        if not math.isclose(d_model * qk_dim_factor, self.key_dim, rel_tol=1e-5):
            raise ValueError(
                f"qk_dim_factor must produce an integer dimension, got {qk_dim_factor}"
            )
        if not math.isclose(d_model * v_dim_factor, self.value_dim, rel_tol=1e-5):
            raise ValueError(f"v_dim_factor must produce an integer dimension, got {v_dim_factor}")
        if self.key_dim % n_heads != 0 or self.value_dim % n_heads != 0:
            raise ValueError("q/k and value dimensions must be divisible by n_heads")
        self.head_k_dim = self.key_dim // n_heads
        self.head_v_dim = self.value_dim // n_heads

        self.w_qk = nn.Linear(
            d_model,
            2 * self.key_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.w_vo = nn.Linear(
            d_model,
            2 * self.value_dim,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.conv1d = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_size,
            groups=d_model,
            bias=True,
            padding=conv_size - 1,
            dtype=dtype,
            device=init_device,
        )
        self.w_if = nn.Linear(
            d_model,
            2 * n_heads,
            bias=True,
            dtype=dtype,
            device=init_device,
        )
        self.o_norm = _MultiHeadLayerNorm(
            n_heads,
            self.head_v_dim,
            eps=norm_eps,
            dtype=dtype,
            init_device=init_device,
        )
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.mlstm_backend = _build_mlstm_backend(
            chunkwise_kernel=chunkwise_kernel,
            chunk_size=chunk_size,
            autocast_kernel_dtype=autocast_kernel_dtype,
            eps=norm_eps,
            mode="train",
        )
        self.mlstm_padded_backend = _build_mlstm_backend(
            chunkwise_kernel=chunkwise_kernel,
            chunk_size=chunk_size,
            autocast_kernel_dtype=autocast_kernel_dtype,
            eps=norm_eps,
            mode="train_with_padding",
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        local_keys = {key[len(prefix) :] for key in state_dict if key.startswith(prefix)}
        if local_keys and "w_qk.weight" not in local_keys:
            official_markers = (
                "q.weight",
                "k.weight",
                "v.weight",
                "igate_preact.weight",
                "fgate_preact.weight",
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "proj_up.weight",
                "ogate_preact.weight",
            )
            detected = (
                "official xLSTM projection keys"
                if any(marker in local_keys for marker in official_markers)
                else "an unrecognized projection layout"
            )
            raise RuntimeError(
                "mLSTM checkpoint layout mismatch: checkpoint uses "
                f"{detected}, but this module requires {CHECKPOINT_LAYOUT}. "
                "Call convert_mlstm_official_to_packed_state_dict explicitly for the "
                f"{OFFICIAL_CHECKPOINT_LAYOUT} projection layout; optimizer-state conversion "
                "is unsupported."
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def backend_identity(self) -> str:
        """Return the exact dependency, kernel, and aligned-sequence mode requested."""
        return self.backend_identity_for_sequence(self.chunk_size)

    def backend_identity_for_sequence(self, seq_len: int) -> str:
        """Return the exact backend identity selected for ``seq_len``."""
        mode = "train" if seq_len % self.chunk_size == 0 else "train_with_padding"
        return f"mlstm-kernels=={MLSTM_KERNELS_VERSION}:{self.chunkwise_kernel}:{mode}"

    def _soft_cap_gates(self, gates: torch.Tensor) -> torch.Tensor:
        if self.gate_soft_cap == 0:
            return gates
        return self.gate_soft_cap * torch.tanh(gates / self.gate_soft_cap)

    def _project_qk_head_first(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.w_qk.weight.view(
            2,
            self.n_heads,
            self.head_k_dim,
            self.d_model,
        )
        qk = torch.einsum("bsd,phkd->pbhsk", x, weight).contiguous()
        return qk.unbind(0)

    def _project_vo_head_first(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.w_vo.weight.view(
            2,
            self.n_heads,
            self.head_v_dim,
            self.d_model,
        )
        vo = torch.einsum("bsd,phvd->pbhsv", x, weight).contiguous()
        return vo.unbind(0)

    def _project_if_head_first(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.w_if.bias is not None
        weight = self.w_if.weight.view(2, self.n_heads, self.d_model)
        bias = self.w_if.bias.view(2, self.n_heads)
        gates = torch.einsum("bsd,phd->pbhs", x, weight).contiguous()
        gates = self._soft_cap_gates(gates + bias[:, None, :, None])
        return gates.unbind(0)

    def _normalize_gate_and_project(
        self,
        hidden: torch.Tensor,
        output_gate: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.o_norm(hidden, head_first=True)
        hidden = hidden * output_gate.sigmoid()
        weight = self.w_out.weight.view(self.d_model, self.n_heads, self.head_v_dim)
        return torch.einsum("bhsv,ohv->bso", hidden, weight)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply mLSTM matrix-memory sequence mixing.

        :param x: Input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Unsupported. The mLSTM backend does not expose
            intra-sequence state resets, so supplying boundaries fails closed.

        :returns: Output of shape ``(batch_size, seq_len, d_model)``.
        """
        if cu_doc_lens is not None:
            raise RuntimeError(
                "mLSTM document boundaries are not supported until segmented state resets exist"
            )
        del kwargs
        _, seq_len, _ = x.shape

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len]
        x_conv = torch.nn.functional.silu(x_conv.transpose(1, 2))
        q, k = self._project_qk_head_first(x_conv)
        v, output_gate = self._project_vo_head_first(x)
        i, f = self._project_if_head_first(x)

        backend = (
            self.mlstm_backend if seq_len % self.chunk_size == 0 else self.mlstm_padded_backend
        )
        hidden = backend(
            q=q,
            k=k,
            v=v,
            i=i,
            f=f,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]

        return self._normalize_gate_and_project(hidden, output_gate)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for XLSTMMixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        del ring, uly
        if cp_mesh.size() == 1:
            return
        raise NotImplementedError("Context parallelism is not implemented for XLSTMMixer")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for XLSTMMixer"
            )
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for projection in (self.w_qk, self.w_vo):
            init_linear(projection, std=std, generator=generator)
        self.w_vo.weight[self.value_dim :].zero_()
        init_linear(self.conv1d, std=std, generator=generator)
        self.w_if.weight.zero_()
        assert self.w_if.bias is not None
        self.w_if.bias[: self.n_heads].fill_(self.input_gate_bias)
        self.w_if.bias[self.n_heads :].copy_(
            torch.linspace(
                self.forget_gate_bias_min,
                self.forget_gate_bias_max,
                steps=self.n_heads,
                dtype=self.w_if.bias.dtype,
                device=self.w_if.bias.device,
            )
        )
        self.o_norm.weight.fill_(1.0)

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5
        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Estimate training FLOPs per token for the projections and mLSTM cell.
        """
        del seq_len
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward).
        param_flops = 6 * sum(param.numel() for param in self.parameters())

        state_size = self.n_heads * self.head_k_dim * self.head_v_dim
        # Cell decay/write, normalizer decay/write, and cell/normalizer reads.
        # The 6x multiplier includes multiply-adds for forward and backward.
        recurrent_flops = 6 * (3 * state_size + 3 * self.key_dim)
        return int(param_flops + recurrent_flops)


@SequenceMixerConfig.register("xlstm")
@dataclass
class XLSTMMixerConfig(SequenceMixerConfig[XLSTMMixer]):
    """
    Configuration for :class:`XLSTMMixer`.

    See :class:`XLSTMMixer` for a description of each option.
    """

    n_heads: int = 4
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    conv_size: int = 4
    gate_soft_cap: float = 15.0
    input_gate_bias: float = -10.0
    forget_gate_bias_min: float = 3.0
    forget_gate_bias_max: float = 6.0
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    chunk_size: int = 256
    autocast_kernel_dtype: str = "bfloat16"
    norm_eps: float = 1e-6
    checkpoint_layout: str = CHECKPOINT_LAYOUT
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        """
        Return the number of parameters in the built mixer.

        :param d_model: The model dimensionality.
        """
        key_dim = int(d_model * self.qk_dim_factor)
        value_dim = int(d_model * self.v_dim_factor)

        params = 2 * d_model * key_dim  # q and k
        params += d_model * value_dim  # v
        params += d_model * (self.conv_size + 1)  # depthwise conv weight+bias
        params += 2 * (d_model * self.n_heads + self.n_heads)  # i and f
        params += d_model * value_dim  # output gate
        params += value_dim  # independently affine head-wise LayerNorm
        params += value_dim * d_model  # output projection
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: BufferCache | None = None,
    ) -> XLSTMMixer:
        """
        Build the configured xLSTM mixer.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The number of layers (unused).
        :param init_device: Device on which parameters are initialized.
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache
        return XLSTMMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            conv_size=self.conv_size,
            gate_soft_cap=self.gate_soft_cap,
            input_gate_bias=self.input_gate_bias,
            forget_gate_bias_min=self.forget_gate_bias_min,
            forget_gate_bias_max=self.forget_gate_bias_max,
            chunkwise_kernel=self.chunkwise_kernel,
            chunk_size=self.chunk_size,
            autocast_kernel_dtype=self.autocast_kernel_dtype,
            norm_eps=self.norm_eps,
            checkpoint_layout=self.checkpoint_layout,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
