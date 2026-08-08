"""Correctness-first recurrent and three-phase chunkwise Flash PD-SSM scans."""

from typing import Optional

import torch

from .transition import (
    SparseAffineTransition,
    apply_sparse_affine,
    compose_sparse_affine,
)

__all__ = [
    "affine_chunkwise_reference",
    "affine_recurrent_reference",
    "compose_affine",
    "sparse_chunkwise_reference",
    "sparse_recurrent_reference",
]


def compose_affine(
    later_matrix: torch.Tensor,
    later_bias: torch.Tensor,
    earlier_matrix: torch.Tensor,
    earlier_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compose dense affine maps as ``later(earlier(state))``.

    :param later_matrix: Matrix of the transition applied second.
    :param later_bias: Bias of the transition applied second.
    :param earlier_matrix: Matrix of the transition applied first.
    :param earlier_bias: Bias of the transition applied first.

    :returns: The matrix and bias of the composed affine map.
    """
    matrix = later_matrix @ earlier_matrix
    bias = torch.einsum("...ij,...j->...i", later_matrix, earlier_bias) + later_bias
    return matrix, bias


def _validate_dense_scan_inputs(
    transition: torch.Tensor,
    bias: torch.Tensor,
    initial: Optional[torch.Tensor],
) -> torch.Tensor:
    if transition.ndim < 3:
        raise ValueError("transition must have shape (..., time, state, state)")
    if transition.shape[-1] != transition.shape[-2]:
        raise ValueError("transition matrices must be square")
    expected_bias_shape = transition.shape[:-1]
    if bias.shape != expected_bias_shape:
        raise ValueError(f"bias shape must be {expected_bias_shape}, got {bias.shape}")
    expected_initial_shape = transition.shape[:-3] + transition.shape[-1:]
    if initial is None:
        return torch.zeros(expected_initial_shape, dtype=bias.dtype, device=bias.device)
    if initial.shape != expected_initial_shape:
        raise ValueError(f"initial shape must be {expected_initial_shape}, got {initial.shape}")
    return initial


def affine_recurrent_reference(
    transition: torch.Tensor,
    bias: torch.Tensor,
    *,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate ``x_t = A_t x_(t-1) + b_t`` sequentially.

    This dense path intentionally preserves autograd through both slope-annealed STE
    selections. It is the numerical oracle, not the memory-efficient production path.

    :param transition: Matrices of shape ``(..., time, state, state)``.
    :param bias: Affine terms of shape ``(..., time, state)``.
    :param initial: Optional initial state of shape ``(..., state)``.

    :returns: States of shape ``(..., time, state)``.
    """
    state = _validate_dense_scan_inputs(transition, bias, initial)
    states = []
    for token_idx in range(transition.shape[-3]):
        state = (
            torch.einsum("...ij,...j->...i", transition[..., token_idx, :, :], state)
            + bias[..., token_idx, :]
        )
        states.append(state)
    if not states:
        return bias.clone()
    return torch.stack(states, dim=-2)


def affine_chunkwise_reference(
    transition: torch.Tensor,
    bias: torch.Tensor,
    *,
    chunk_size: int,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate the affine recurrence with the paper's three-phase chunkwise algorithm.

    Phase A independently builds local prefix transforms and one aggregate per chunk. Phase B
    propagates exclusive carries between chunks. Phase C applies each local prefix to its
    incoming carry. Python loops make this implementation a transparent autograd oracle; they
    are not presented as a performance kernel.

    :param transition: Matrices of shape ``(..., time, state, state)``.
    :param bias: Affine terms of shape ``(..., time, state)``.
    :param chunk_size: Positive number of timesteps per chunk.
    :param initial: Optional initial state of shape ``(..., state)``.

    :returns: States of shape ``(..., time, state)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    initial_state = _validate_dense_scan_inputs(transition, bias, initial)
    time = transition.shape[-3]
    if time == 0:
        return bias.clone()

    state_size = transition.shape[-1]
    batch_shape = transition.shape[:-3]
    identity = torch.eye(state_size, dtype=transition.dtype, device=transition.device)
    identity = identity.expand(*batch_shape, state_size, state_size)
    zero = torch.zeros(*batch_shape, state_size, dtype=bias.dtype, device=bias.device)

    # Phase A: local scan and aggregate, independently for every chunk.
    local_prefixes: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    aggregates: list[tuple[torch.Tensor, torch.Tensor]] = []
    for chunk_start in range(0, time, chunk_size):
        prefix_matrix, prefix_bias = identity, zero
        chunk_prefixes = []
        for token_idx in range(chunk_start, min(chunk_start + chunk_size, time)):
            prefix_matrix, prefix_bias = compose_affine(
                transition[..., token_idx, :, :],
                bias[..., token_idx, :],
                prefix_matrix,
                prefix_bias,
            )
            chunk_prefixes.append((prefix_matrix, prefix_bias))
        local_prefixes.append(chunk_prefixes)
        aggregates.append((prefix_matrix, prefix_bias))

    # Phase B: exclusive global carry at every chunk boundary.
    carries = []
    carry = initial_state
    for aggregate_matrix, aggregate_bias in aggregates:
        carries.append(carry)
        carry = torch.einsum("...ij,...j->...i", aggregate_matrix, carry) + aggregate_bias

    # Phase C: correct every independent local prefix with its incoming carry.
    states = []
    for carry, chunk_prefixes in zip(carries, local_prefixes):
        for prefix_matrix, prefix_bias in chunk_prefixes:
            states.append(torch.einsum("...ij,...j->...i", prefix_matrix, carry) + prefix_bias)
    return torch.stack(states, dim=-2)


def _validate_sparse_scan_inputs(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
    initial: Optional[torch.Tensor],
) -> torch.Tensor:
    if destination.shape != diagonal.shape or destination.shape != bias.shape:
        raise ValueError(
            "destination, diagonal, and bias shapes must match, got "
            f"{destination.shape}, "
            f"{diagonal.shape}, and {bias.shape}"
        )
    if destination.ndim < 2:
        raise ValueError("sparse scan inputs must have shape (..., time, state)")
    expected_initial_shape = destination.shape[:-2] + destination.shape[-1:]
    if initial is None:
        return torch.zeros(expected_initial_shape, dtype=bias.dtype, device=bias.device)
    if initial.shape != expected_initial_shape:
        raise ValueError(f"initial shape must be {expected_initial_shape}, got {initial.shape}")
    return initial


def _token_sparse_transition(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
) -> SparseAffineTransition:
    return SparseAffineTransition(
        destination=destination,
        scale=diagonal,
        bias=bias,
    )


def sparse_recurrent_reference(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
    *,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate the Flash PD recurrence directly from integer transition indices.

    :param destination: Destination per source, shape ``(..., time, state)``.
    :param diagonal: Complex diagonal factors with the same shape.
    :param bias: Complex affine terms with the same shape.
    :param initial: Optional state of shape ``(..., state)``.

    :returns: Complex states of shape ``(..., time, state)``.
    """
    state = _validate_sparse_scan_inputs(destination, diagonal, bias, initial)
    states = []
    for token_idx in range(destination.shape[-2]):
        transition = _token_sparse_transition(
            destination[..., token_idx, :],
            diagonal[..., token_idx, :],
            bias[..., token_idx, :],
        )
        state = apply_sparse_affine(transition, state)
        states.append(state)
    if not states:
        return bias.clone()
    return torch.stack(states, dim=-2)


def sparse_chunkwise_reference(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
    *,
    chunk_size: int,
    initial: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Evaluate the three-phase algorithm while retaining sparse transition triples.

    :param destination: Destination per source, shape ``(..., time, state)``.
    :param diagonal: Complex diagonal factors with the same shape.
    :param bias: Complex affine terms with the same shape.
    :param chunk_size: Positive number of timesteps per chunk.
    :param initial: Optional state of shape ``(..., state)``.

    :returns: Complex states of shape ``(..., time, state)``.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    initial_state = _validate_sparse_scan_inputs(destination, diagonal, bias, initial)
    time, state_size = destination.shape[-2:]
    if time == 0:
        return bias.clone()

    batch_shape = destination.shape[:-2]
    identity_destination = torch.arange(state_size, device=destination.device)
    identity_destination = identity_destination.expand(*batch_shape, state_size)
    identity = SparseAffineTransition(
        destination=identity_destination,
        scale=torch.ones(*batch_shape, state_size, dtype=diagonal.dtype, device=diagonal.device),
        bias=torch.zeros(*batch_shape, state_size, dtype=bias.dtype, device=bias.device),
    )

    local_prefixes: list[list[SparseAffineTransition]] = []
    aggregates: list[SparseAffineTransition] = []
    for chunk_start in range(0, time, chunk_size):
        prefix = identity
        chunk_prefixes = []
        for token_idx in range(chunk_start, min(chunk_start + chunk_size, time)):
            token = _token_sparse_transition(
                destination[..., token_idx, :],
                diagonal[..., token_idx, :],
                bias[..., token_idx, :],
            )
            prefix = compose_sparse_affine(token, prefix)
            chunk_prefixes.append(prefix)
        local_prefixes.append(chunk_prefixes)
        aggregates.append(prefix)

    carries = []
    carry = initial_state
    for aggregate in aggregates:
        carries.append(carry)
        carry = apply_sparse_affine(aggregate, carry)

    states = [
        apply_sparse_affine(prefix, carry)
        for carry, chunk_prefixes in zip(carries, local_prefixes)
        for prefix in chunk_prefixes
    ]
    return torch.stack(states, dim=-2)
