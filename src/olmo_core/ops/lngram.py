"""Optional Triton acceleration for Lngram."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
from torch.distributed.tensor import DTensor

try:
    import triton  # type: ignore

    from olmo_core.kernels import lngram as kernels
except (ImportError, RuntimeError):
    triton = None  # type: ignore
    kernels = None  # type: ignore

__all__ = ["has_lngram_triton"]

_SUPPORTED_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def has_lngram_triton() -> bool:
    """Return whether the optional Lngram Triton kernel imported successfully."""
    return triton is not None and kernels is not None


if has_lngram_triton():

    @torch.library.triton_op(
        "olmo_core::lngram_counterfactual_grad_z",
        mutates_args={},
    )
    def _counterfactual_grad_z_triton(
        z: torch.Tensor,
        codes: torch.Tensor,
        table_order_2: torch.Tensor,
        table_order_3: torch.Tensor,
        upstream_order_2: torch.Tensor,
        upstream_order_3: torch.Tensor,
        temperature: float,
        scale: float,
    ) -> torch.Tensor:
        grad_z = torch.empty_like(z, memory_format=torch.contiguous_format)
        batch_size, sequence_length, channels = z.shape
        num_routes = channels // 4
        memory_dim = table_order_2.shape[1]

        def grid(meta):
            return (
                batch_size * sequence_length,
                triton.cdiv(num_routes, meta["BLOCK_R"]),
            )

        torch.library.wrap_triton(kernels.counterfactual_grad_z_kernel)[grid](
            z,
            codes,
            table_order_2,
            table_order_3,
            upstream_order_2,
            upstream_order_3,
            grad_z,
            batch_size,
            sequence_length,
            num_routes,
            memory_dim,
            temperature,
            scale,
            z.stride(0),
            z.stride(1),
            z.stride(2),
            codes.stride(0),
            codes.stride(1),
            codes.stride(2),
            table_order_2.stride(0),
            table_order_2.stride(1),
            table_order_3.stride(0),
            table_order_3.stride(1),
            upstream_order_2.stride(0),
            upstream_order_2.stride(1),
            upstream_order_2.stride(2),
            upstream_order_3.stride(0),
            upstream_order_3.stride(1),
            upstream_order_3.stride(2),
            grad_z.stride(0),
            grad_z.stride(1),
            grad_z.stride(2),
            BLOCK_R=4,
            BLOCK_D=64,
            num_warps=4,
        )
        return grad_z

else:
    _counterfactual_grad_z_triton = None


def _eligible_float_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and tensor.dtype in _SUPPORTED_DTYPES and not isinstance(tensor, DTensor)


def _eligible_codes(codes: torch.Tensor) -> bool:
    return (
        codes.is_cuda
        and codes.dtype in (torch.uint8, torch.int32, torch.int64)
        and not isinstance(codes, DTensor)
    )


def _try_counterfactual_grad_z(
    z: torch.Tensor,
    codes: torch.Tensor,
    tables: Sequence[torch.Tensor],
    orders: Sequence[int],
    grad_outputs: Sequence[Optional[torch.Tensor]],
    *,
    bits_per_route: int,
    temperature: float,
    scale: float,
) -> Optional[torch.Tensor]:
    """Run the fused CUDA path when its exact contract is satisfied."""
    if (
        _counterfactual_grad_z_triton is None
        or tuple(orders) != (2, 3)
        or bits_per_route != 4
        or len(tables) != 2
        or len(grad_outputs) != 2
        or grad_outputs[0] is None
        or grad_outputs[1] is None
        or z.numel() == 0
        or scale == 0
    ):
        return None

    table_order_2, table_order_3 = tables
    upstream_order_2, upstream_order_3 = grad_outputs
    assert upstream_order_2 is not None
    assert upstream_order_3 is not None
    if (
        z.ndim != 3
        or codes.ndim != 3
        or table_order_2.ndim != 2
        or table_order_3.ndim != 2
        or upstream_order_2.ndim != 3
        or upstream_order_3.ndim != 3
        or z.shape[2] == 0
        or z.shape[2] % 4
    ):
        return None

    float_tensors = (
        z,
        table_order_2,
        table_order_3,
        upstream_order_2,
        upstream_order_3,
    )
    if not all(_eligible_float_tensor(tensor) for tensor in float_tensors):
        return None
    if not _eligible_codes(codes):
        return None
    if any(tensor.device != z.device for tensor in (*float_tensors, codes)):
        return None
    if table_order_2.shape[1] != table_order_3.shape[1]:
        return None
    if not 0 < table_order_2.shape[1] <= 64:
        return None
    if upstream_order_2.shape != upstream_order_3.shape:
        return None
    expected_upstream_shape = (
        z.shape[0],
        z.shape[1],
        (z.shape[2] // 4) * table_order_2.shape[1],
    )
    if upstream_order_2.shape != expected_upstream_shape:
        return None
    if codes.shape != (z.shape[0], z.shape[1], z.shape[2] // 4):
        return None
    num_routes = z.shape[2] // 4
    if table_order_2.shape[0] != num_routes * 16**2:
        return None
    if table_order_3.shape[0] != num_routes * 16**3:
        return None

    return _counterfactual_grad_z_triton(
        z,
        codes,
        table_order_2,
        table_order_3,
        upstream_order_2,
        upstream_order_3,
        float(temperature),
        float(scale),
    )
