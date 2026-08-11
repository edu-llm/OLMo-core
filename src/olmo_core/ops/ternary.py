"""
Packed ternary-weight training operations.

The public Maple artifacts establish the forward quantizer, but not its backward rule. The
custom autograd functions in this module therefore implement the branch's explicit replication
policy: a plain identity STE. Forward and input-gradient use packed add/subtract Triton kernels;
the latent-weight gradient remains an ordinary BF16 GEMM, as identity STE requires.

Packing is ephemeral. :class:`PackedTWNCache` is an ordinary Python object, not a module or
buffer, and packed codes never appear in a state dict.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

try:
    from olmo_core.kernels import ternary as kernels

    _KERNEL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # Triton is an optional, CUDA-only dependency.
    kernels = None  # type: ignore[assignment]
    _KERNEL_IMPORT_ERROR = exc

__all__ = [
    "TWN_NEGATIVE_CODE",
    "TWN_ZERO_CODE",
    "TWN_POSITIVE_CODE",
    "PackedTWN",
    "PackedTWNCache",
    "pack_twn_reference",
    "unpack_twn_codes",
    "dequantize_packed_twn",
    "native_packed_status",
    "native_packed_linear",
    "native_packed_grouped_linear",
]


# DeepGrove MLX stores ``ternary + 1`` in two-bit affine codes, LSB first.
TWN_NEGATIVE_CODE = 0
TWN_ZERO_CODE = 1
TWN_POSITIVE_CODE = 2
_TRITS_PER_WORD = 16


@dataclass(frozen=True)
class PackedTWN:
    """
    Ephemeral packed representation of a logical ``[..., out_features, in_features]`` weight.

    ``codes`` is forward-friendly and ``codes_t`` is input-gradient-friendly. Both pack 16
    two-bit codes per ``uint32`` word, least-significant code first. ``alpha`` is rounded to
    BF16 once per logical output row.
    """

    codes: torch.Tensor
    codes_t: torch.Tensor
    alpha: torch.Tensor
    in_features: int
    out_features: int
    num_experts: int
    materialized: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _PackCacheKey:
    version: int
    data_ptr: int
    storage_id: int
    storage_offset: int
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    device: torch.device
    dtype: torch.dtype
    in_dim: int
    orientation: str


PackFn = Callable[..., PackedTWN]


class PackedTWNCache:
    """
    One-weight ephemeral packed cache.

    The key deliberately includes the latent tensor version, storage identity/pointer/offset,
    shape/stride, device, input axis, and operator orientation. Optimizer updates normally change
    ``_version`` while checkpoint loads and storage replacement change either version or storage.
    FSDP2 may reuse both storage and version across updated unshards, so owning modules explicitly
    clear this cache before every FSDP-managed forward.
    """

    def __init__(self) -> None:
        self._key: Optional[_PackCacheKey] = None
        self._packed: Optional[PackedTWN] = None
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(weight: torch.Tensor, in_dim: int, orientation: str) -> _PackCacheKey:
        normalized_in_dim = in_dim % weight.ndim
        storage = weight.untyped_storage()
        return _PackCacheKey(
            version=int(getattr(weight, "_version", 0)),
            data_ptr=weight.data_ptr(),
            storage_id=int(getattr(storage, "_cdata", id(storage))),
            storage_offset=weight.storage_offset(),
            shape=tuple(weight.shape),
            stride=tuple(weight.stride()),
            device=weight.device,
            dtype=weight.dtype,
            in_dim=normalized_in_dim,
            orientation=orientation,
        )

    def clear(self) -> None:
        """Invalidate the pack while retaining native output buffers for safe repacking."""
        self._key = None

    def get_or_pack(
        self,
        weight: torch.Tensor,
        *,
        in_dim: int,
        orientation: str,
        packer: Optional[PackFn] = None,
    ) -> PackedTWN:
        """Return a valid cached pack, or repack after any key change."""
        key = self._make_key(weight, in_dim, orientation)
        if key == self._key and self._packed is not None:
            self.hits += 1
            return self._packed

        if packer is None:
            if kernels is None:
                detail = f": {_KERNEL_IMPORT_ERROR}" if _KERNEL_IMPORT_ERROR is not None else ""
                raise RuntimeError(f"native packed TWN kernels are unavailable{detail}")
            packer = kernels.pack_twn

        with torch.no_grad():
            if kernels is not None and packer is kernels.pack_twn_forward_only:
                packed = packer(weight, in_dim, out=self._packed)
            else:
                packed = packer(weight, in_dim)
        self._key = key
        self._packed = packed
        self.misses += 1
        return packed


def _canonical_weight(weight: torch.Tensor, in_dim: int) -> torch.Tensor:
    in_dim %= weight.ndim
    if weight.ndim not in (2, 3):
        raise ValueError(f"packed TWN expects a 2-D or 3-D weight, got {weight.ndim}-D")
    if weight.ndim == 3 and in_dim == 0:
        raise ValueError("the expert axis cannot be the packed TWN input-feature axis")
    return weight.movedim(in_dim, -1).contiguous()


def _pack_codes_reference(codes: torch.Tensor) -> torch.Tensor:
    k = codes.shape[-1]
    packed_k = (k + _TRITS_PER_WORD - 1) // _TRITS_PER_WORD
    padded_k = packed_k * _TRITS_PER_WORD
    if padded_k != k:
        # Padding is the zero trit. Kernels mask the logical width, but using code 1 also makes
        # direct unpacking of the padded tail unsurprising.
        codes = torch.nn.functional.pad(codes, (0, padded_k - k), value=float(TWN_ZERO_CODE))
    lanes = codes.to(torch.int64).reshape(*codes.shape[:-1], packed_k, _TRITS_PER_WORD)
    shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=codes.device)
    words = torch.sum(lanes << shifts, dim=-1)
    return words.to(torch.uint32)


def pack_twn_reference(weight: torch.Tensor, in_dim: int) -> PackedTWN:
    """
    CPU/GPU Torch reference for Maple-compatible TWN packing.

    Statistics are FP32, the threshold comparison is strict, alpha is the survivor mean rounded
    to BF16, and codes are DeepGrove-compatible ``negative=0, zero=1, positive=2`` packed LSB
    first. This function is a correctness oracle; native execution uses the Triton packer.
    """
    logical = _canonical_weight(weight, in_dim)
    w32 = logical.detach().to(torch.float32)
    absw = w32.abs()
    delta = 0.7 * absw.mean(dim=-1, keepdim=True)
    survivors = absw > delta
    count = survivors.sum(dim=-1).clamp(min=1)
    alpha = ((absw * survivors).sum(dim=-1) / count).to(torch.bfloat16).contiguous()
    codes = torch.where(
        w32 > delta,
        TWN_POSITIVE_CODE,
        torch.where(w32 < -delta, TWN_NEGATIVE_CODE, TWN_ZERO_CODE),
    )
    packed = _pack_codes_reference(codes)
    packed_t = _pack_codes_reference(codes.transpose(-2, -1).contiguous())
    num_experts = logical.shape[0] if logical.ndim == 3 else 1
    return PackedTWN(
        codes=packed,
        codes_t=packed_t,
        alpha=alpha,
        in_features=logical.shape[-1],
        out_features=logical.shape[-2],
        num_experts=num_experts,
    )


def unpack_twn_codes(packed: torch.Tensor, logical_width: int) -> torch.Tensor:
    """Unpack DeepGrove-compatible words to integer codes along the last axis."""
    words = packed.to(torch.int64).unsqueeze(-1)
    shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=packed.device)
    codes = ((words >> shifts) & 0x3).reshape(*packed.shape[:-1], -1)
    return codes[..., :logical_width]


def dequantize_packed_twn(packed: PackedTWN) -> torch.Tensor:
    """Materialize the BF16 logical weight for tests and parity checks only."""
    codes = unpack_twn_codes(packed.codes, packed.in_features)
    trits = torch.where(
        codes == TWN_POSITIVE_CODE,
        1.0,
        torch.where(codes == TWN_NEGATIVE_CODE, -1.0, 0.0),
    ).to(torch.bfloat16)
    return trits * packed.alpha.unsqueeze(-1)


def _autocast_activation_dtype(activation: torch.Tensor) -> torch.dtype:
    if (
        activation.device.type == "cuda"
        and activation.dtype in (torch.float16, torch.float32, torch.bfloat16)
        and torch.is_autocast_enabled("cuda")
    ):
        return torch.get_autocast_dtype("cuda")
    return activation.dtype


def _cast_activation_for_native(activation: torch.Tensor) -> torch.Tensor:
    if _autocast_activation_dtype(activation) is torch.bfloat16:
        return activation.to(torch.bfloat16)
    return activation


def native_packed_status(
    weight: Optional[torch.Tensor] = None, activation: Optional[torch.Tensor] = None
) -> Dict[str, object]:
    """Describe whether native packed execution is usable for ``weight``."""
    reason: Optional[str] = None
    if kernels is None:
        reason = (
            f"Triton kernels failed to import: {_KERNEL_IMPORT_ERROR}"
            if _KERNEL_IMPORT_ERROR is not None
            else "Triton kernels are unavailable"
        )
    elif not torch.cuda.is_available():
        reason = "CUDA is unavailable"
    elif weight is not None and weight.device.type not in ("cuda", "meta"):
        reason = f"weight is on {weight.device.type}, not CUDA"
    elif activation is not None and activation.device.type not in ("cuda", "meta"):
        reason = f"activation is on {activation.device.type}, not CUDA"
    elif torch.cuda.get_device_capability(
        weight.device
        if weight is not None and weight.device.type == "cuda"
        else (
            activation.device
            if activation is not None and activation.device.type == "cuda"
            else torch.cuda.current_device()
        )
    ) < (8, 0):
        reason = "native packed TWN requires SM80 or newer for BF16 execution"
    elif weight is not None and weight.dtype is not torch.bfloat16:
        reason = f"native packed TWN requires BF16 weights, got {weight.dtype}"
    elif activation is not None and _autocast_activation_dtype(activation) is not torch.bfloat16:
        reason = (
            "native packed TWN requires BF16 activations"
            f", got {_autocast_activation_dtype(activation)}"
        )
    elif (
        weight is not None
        and activation is not None
        and weight.device.type != "meta"
        and activation.device.type != "meta"
        and weight.device != activation.device
    ):
        reason = f"weight is on {weight.device}, but activation is on {activation.device}"
    return {
        "available": reason is None,
        "reason": reason,
        "kernel": "triton_packed_add_sub" if reason is None else "fake_quant_bf16",
    }


def _require_native(weight: torch.Tensor, activation: torch.Tensor) -> None:
    status = native_packed_status(weight, activation)
    if not status["available"]:
        raise RuntimeError(f"native packed TWN cannot run: {status['reason']}")


def _restore_weight_orientation(logical: torch.Tensor, in_dim: int) -> torch.Tensor:
    return logical.movedim(-1, in_dim)


def _grouped_grad_weight(
    grad_output: torch.Tensor, x: torch.Tensor, offsets: torch.Tensor, num_experts: int
) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.new_zeros((num_experts, grad_output.shape[-1], x.shape[-1]))

    grouped_mm = getattr(torch, "_grouped_mm", None)
    if grouped_mm is not None:
        try:
            # Both operands are 2-D and partitioned by the same cumulative offsets:
            # [O, sum(M_e)] @ [sum(M_e), K] -> [E, O, K].
            return grouped_mm(grad_output.transpose(0, 1).contiguous(), x.contiguous(), offsets)
        except RuntimeError:
            # Older torch builds expose the operator but lack the 2-D x 2-D kernel on SM80.
            # Keep a correctness fallback; the benchmark reports this separately.
            pass

    pieces = []
    start = 0
    for end in offsets.to("cpu").tolist():
        pieces.append(
            grad_output[start:end].transpose(0, 1) @ x[start:end]
            if end > start
            else x.new_zeros((grad_output.shape[-1], x.shape[-1]))
        )
        start = end
    return torch.stack(pieces)


class _NativePackedLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        cache: PackedTWNCache,
        orientation: str,
    ) -> torch.Tensor:
        _require_native(weight, x)
        assert kernels is not None
        packed = cache.get_or_pack(
            weight,
            in_dim=-1,
            orientation=orientation,
            packer=kernels.pack_twn_forward_only,
        )
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        if packed.materialized is None:
            raise RuntimeError("forward-only TWN pack did not produce a BF16 materialization")
        materialized = packed.materialized
        out = x2 @ materialized.transpose(0, 1)
        if bias is not None:
            out = out + bias
        ctx.save_for_backward(x, materialized)
        ctx.weight_shape = weight.shape
        ctx.has_bias = bias is not None
        return out.reshape(*x.shape[:-1], packed.out_features)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, materialized = ctx.saved_tensors
        grad2 = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        x2 = x.reshape(-1, x.shape[-1]).contiguous()
        grad_input = (grad2 @ materialized).reshape_as(x)
        # Identity STE replication policy: dL/dW_latent is the ordinary linear dL/dW_q.
        grad_weight = (grad2.transpose(0, 1) @ x2).reshape(ctx.weight_shape)
        grad_bias = grad2.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None


class _NativePackedFixedGroupedLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        in_dim: int,
        cache: PackedTWNCache,
        orientation: str,
    ) -> torch.Tensor:
        _require_native(weight, x)
        assert kernels is not None
        packed = cache.get_or_pack(
            weight,
            in_dim=in_dim,
            orientation=orientation,
            packer=kernels.pack_twn_forward_only,
        )
        # M20's fine-grained experts make decode-inside-MMA substantially slower than decoding
        # each packed weight once and handing both GEMMs to cuBLAS. Keep the ephemeral packed
        # representation as the source of truth, then retain only its BF16 materialization until
        # backward so forward and input-gradient share it.
        if packed.materialized is None:
            raise RuntimeError("forward-only TWN pack did not produce a BF16 materialization")
        materialized = packed.materialized
        out = torch.bmm(x.contiguous(), materialized.transpose(1, 2))
        ctx.save_for_backward(x, materialized)
        ctx.in_dim = in_dim
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, materialized = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.bmm(grad_output, materialized)
        grad_logical = torch.bmm(grad_output.transpose(1, 2), x)
        grad_weight = _restore_weight_orientation(grad_logical, ctx.in_dim)
        return grad_input, grad_weight, None, None, None


class _NativePackedJaggedGroupedLinear(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        batch_sizes: torch.Tensor,
        in_dim: int,
        cache: PackedTWNCache,
        orientation: str,
    ) -> torch.Tensor:
        _require_native(weight, x)
        assert kernels is not None
        packed = cache.get_or_pack(weight, in_dim=in_dim, orientation=orientation)
        if batch_sizes.ndim != 1 or batch_sizes.numel() != packed.num_experts:
            raise ValueError(
                f"batch_sizes must have shape ({packed.num_experts},), "
                f"got {tuple(batch_sizes.shape)}"
            )
        if batch_sizes.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"batch_sizes must be int32 or int64, got {batch_sizes.dtype}")
        batch_sizes_i32 = batch_sizes.to(device=x.device, dtype=torch.int32)
        offsets = torch.cumsum(batch_sizes_i32, 0, dtype=torch.int32)
        expert_ids = torch.repeat_interleave(
            torch.arange(packed.num_experts, device=x.device, dtype=torch.int32),
            batch_sizes_i32,
            output_size=x.shape[0],
        )
        out = kernels.jagged_grouped_packed_matmul(
            x.contiguous(), packed.codes, packed.alpha, expert_ids, packed.in_features
        )
        ctx.save_for_backward(x, packed.codes_t, packed.alpha, expert_ids, offsets)
        ctx.in_dim = in_dim
        ctx.num_experts = packed.num_experts
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        assert kernels is not None
        x, codes_t, alpha, expert_ids, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = kernels.jagged_grouped_packed_matmul_transpose(
            grad_output, codes_t, alpha, expert_ids, x.shape[-1]
        )
        grad_logical = _grouped_grad_weight(grad_output, x, offsets, ctx.num_experts)
        grad_weight = _restore_weight_orientation(grad_logical, ctx.in_dim)
        return grad_input, grad_weight, None, None, None, None


@torch._dynamo.disable()
def native_packed_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    cache: PackedTWNCache,
    orientation: str = "dense",
) -> torch.Tensor:
    """Packed dense linear with identity-STE latent-weight gradients."""
    x = _cast_activation_for_native(x)
    return _NativePackedLinear.apply(x, weight, bias, cache, orientation)  # type: ignore[no-any-return]


@torch._dynamo.disable()
def native_packed_grouped_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    in_dim: int,
    cache: PackedTWNCache,
    orientation: str,
    batch_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Packed expert-grouped linear for fixed-capacity or jagged dropless inputs.

    ``batch_sizes=None`` selects fixed ``(experts, capacity, in_features)`` input. A vector
    selects contiguous jagged ``(tokens, in_features)`` input and supports empty experts.
    """
    x = _cast_activation_for_native(x)
    if batch_sizes is None:
        return _NativePackedFixedGroupedLinear.apply(  # type: ignore[no-any-return]
            x, weight, in_dim, cache, orientation
        )
    return _NativePackedJaggedGroupedLinear.apply(  # type: ignore[no-any-return]
        x, weight, batch_sizes, in_dim, cache, orientation
    )
