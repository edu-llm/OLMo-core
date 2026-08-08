"""Static storage accounting for the fused Mamba-3 ``b == 3`` rotation path."""

from math import prod
from typing import Sequence


def saved_tensor_accounting(
    *,
    B_shape: Sequence[int],
    C_shape: Sequence[int],
    theta_shape: Sequence[int],
    bc_element_size: int,
    prefix_element_size: int = 4,
) -> dict[str, int]:
    """
    Account for generated tensors retained by the fused quaternion autograd boundary.

    Inputs are shapes and element sizes so callers can use this helper during configuration or
    profiling without importing Torch, allocating tensors, changing the recurrence API, or adding
    work to a compiled graph. The current implementation retains only the compact FP32 quaternion
    prefix as generated storage; ``B``, ``C``, and ``theta`` are existing input references.

    :param B_shape: Shape ``(batch, sequence, groups, rank, d_state)``.
    :param C_shape: Shape matching ``B_shape``.
    :param theta_shape: Shape ``(batch, sequence, groups, d_state // 3, 3)``.
    :param bc_element_size: Bytes per B/C element, normally 2 under BF16 autocast.
    :param prefix_element_size: Bytes per prefix element. The FP32 floor makes the default 4.

    :returns: Byte counts for retained generated storage and eliminated alternatives.
    """
    B_shape = tuple(B_shape)
    C_shape = tuple(C_shape)
    theta_shape = tuple(theta_shape)
    if len(B_shape) != 5 or len(C_shape) != 5 or len(theta_shape) != 5:
        raise ValueError("B, C, and theta shapes must all be five-dimensional")
    if B_shape != C_shape:
        raise ValueError(f"B and C shapes must match, got {B_shape} and {C_shape}")
    if B_shape[:3] != theta_shape[:3]:
        raise ValueError("B/C and theta batch, sequence, and group dimensions must match")
    if theta_shape[-1] != 3 or B_shape[-1] != theta_shape[-2] * 3:
        raise ValueError("theta must carry three angles for every b=3 state block")
    if any(dim < 1 for dim in B_shape + theta_shape):
        raise ValueError("all b=3 accounting dimensions must be positive")
    if bc_element_size < 1 or prefix_element_size < 1:
        raise ValueError("element sizes must be positive byte counts")

    bc_elements = prod(B_shape) + prod(C_shape)
    prefix_positions = prod(theta_shape[:-1])
    compact_prefix_bytes = prefix_positions * 4 * prefix_element_size
    materialized_bc_bytes = bc_elements * bc_element_size
    matrix_prefix_bytes = prefix_positions * 9 * prefix_element_size
    return {
        "generated_saved_bytes": compact_prefix_bytes,
        "compact_prefix_saved_bytes": compact_prefix_bytes,
        "materialized_bc_saved_bytes_avoided": materialized_bc_bytes,
        "matrix_prefix_bytes_avoided": matrix_prefix_bytes,
        "prior_generated_saved_bytes": compact_prefix_bytes + materialized_bc_bytes,
    }
