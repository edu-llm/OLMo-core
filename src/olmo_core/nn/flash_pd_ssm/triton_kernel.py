"""Optional forward-only Triton prototype for the three-phase Flash PD scan."""

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in environments without Triton.
    triton = None
    tl = None

__all__ = ["TritonCapability", "flash_pd_triton_scan", "triton_capability"]


@dataclass(frozen=True)
class TritonCapability:
    """
    Result of probing whether the prototype Triton scan can run.

    :param available: Whether all current constraints are satisfied.
    :param reason: Human-readable capability or fallback explanation.
    """

    available: bool
    reason: str


def triton_capability(
    destination: Optional[torch.Tensor] = None,
    diagonal: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = 128,
    requires_autograd: bool = False,
) -> TritonCapability:
    """
    Probe the forward-only Triton implementation without silently dispatching.

    The prototype currently requires Triton with ``tl.gather``, CUDA tensors, complex64 values,
    a state size no larger than one warp (32), zero initial state, and no requested autograd.
    The one-warp limit makes each collision reduction synchronizable inside one program. Unsupported
    training calls must use one of the PyTorch references so selector and dictionary STE
    gradients remain correct.

    :param destination: Optional source-to-destination index tensor.
    :param diagonal: Optional complex diagonal tensor.
    :param bias: Optional complex affine tensor.
    :param chunk_size: Timesteps per chunk.
    :param requires_autograd: Whether callers need gradients through the recurrence or selectors.

    :returns: An explicit capability result and reason.
    """
    if chunk_size < 1 or chunk_size > 128:
        return TritonCapability(False, "chunk_size must be in [1, 128]")
    if requires_autograd:
        return TritonCapability(False, "prototype is forward-only and does not implement autograd")
    if destination is None or diagonal is None or bias is None:
        if triton is None or tl is None:
            return TritonCapability(False, "Triton is not installed")
        if not hasattr(tl, "gather"):
            return TritonCapability(False, "the installed Triton does not provide tl.gather")
        if not torch.cuda.is_available():
            return TritonCapability(False, "CUDA is not available")
        return TritonCapability(True, "Triton and CUDA are available; tensor constraints unprobed")
    if destination.shape != diagonal.shape or destination.shape != bias.shape:
        return TritonCapability(False, "destination, diagonal, and bias shapes must match")
    if destination.ndim != 4:
        return TritonCapability(
            False, "prototype expects tensors shaped (batch, heads, time, state)"
        )
    if destination.dtype not in (torch.int32, torch.int64):
        return TritonCapability(False, "destination indices must be int32 or int64")
    if diagonal.dtype != torch.complex64 or bias.dtype != torch.complex64:
        return TritonCapability(False, "prototype supports complex64 values only")
    if destination.shape[-2] < 1:
        return TritonCapability(False, "prototype requires a non-empty sequence")
    if destination.shape[-1] < 1 or destination.shape[-1] > 32:
        return TritonCapability(False, "prototype supports state_size <= 32")
    if destination.requires_grad or diagonal.requires_grad or bias.requires_grad:
        return TritonCapability(
            False,
            "prototype is forward-only and does not implement autograd",
        )
    if triton is None or tl is None:
        return TritonCapability(False, "Triton is not installed")
    if not hasattr(tl, "gather"):
        return TritonCapability(False, "the installed Triton does not provide tl.gather")
    if not destination.is_cuda or not diagonal.is_cuda or not bias.is_cuda:
        return TritonCapability(False, "CUDA tensors are required")
    return TritonCapability(True, "supported by the forward-only three-phase Triton prototype")


if triton is not None and tl is not None:

    @triton.jit
    def _phase_a_kernel(
        destination_ptr,
        diagonal_real_ptr,
        diagonal_imag_ptr,
        bias_real_ptr,
        bias_imag_ptr,
        aggregate_source_ptr,
        aggregate_scale_real_ptr,
        aggregate_scale_imag_ptr,
        aggregate_bias_real_ptr,
        aggregate_bias_imag_ptr,
        time,
        state,
        n_chunks,
        chunk_size,
        BLOCK_STATE: tl.constexpr,
    ):
        """Fold each chunk independently into one sparse affine transform."""
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_STATE)
        lane_mask = offsets < state
        identity_source = tl.where(lane_mask, offsets, 0)

        acc_source = identity_source
        acc_scale_real = tl.where(lane_mask, 1.0, 0.0)
        acc_scale_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        acc_bias_real = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        acc_bias_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)

        for local_idx in range(0, chunk_size):
            token = chunk * chunk_size + local_idx
            token_mask = lane_mask & (token < time)
            token_offset = (row * time + token) * state + offsets
            token_destination = tl.load(
                destination_ptr + token_offset, mask=token_mask, other=0
            ).to(tl.int32)
            token_destination = tl.where(token_mask, token_destination, identity_source)
            token_diagonal_real = tl.load(
                diagonal_real_ptr + token_offset, mask=token_mask, other=1.0
            ).to(tl.float32)
            token_diagonal_imag = tl.load(
                diagonal_imag_ptr + token_offset, mask=token_mask, other=0.0
            ).to(tl.float32)
            token_bias_real = tl.load(bias_real_ptr + token_offset, mask=token_mask, other=0.0).to(
                tl.float32
            )
            token_bias_imag = tl.load(bias_imag_ptr + token_offset, mask=token_mask, other=0.0).to(
                tl.float32
            )

            next_source = tl.gather(token_destination, acc_source, axis=0)
            later_scale_real = tl.gather(token_diagonal_real, acc_source, axis=0)
            later_scale_imag = tl.gather(token_diagonal_imag, acc_source, axis=0)

            next_scale_real = acc_scale_real * later_scale_real - acc_scale_imag * later_scale_imag
            next_scale_imag = acc_scale_real * later_scale_imag + acc_scale_imag * later_scale_real

            # Apply the token transition to the accumulated bias. Multiple source lanes may
            # target one destination, so this must be a collision-preserving scatter reduction.
            scratch_offset = (row * n_chunks + chunk) * state + offsets
            tl.store(
                aggregate_bias_real_ptr + scratch_offset,
                token_bias_real,
                mask=lane_mask,
            )
            tl.store(
                aggregate_bias_imag_ptr + scratch_offset,
                token_bias_imag,
                mask=lane_mask,
            )
            tl.debug_barrier()
            contribution_real = (
                token_diagonal_real * acc_bias_real - token_diagonal_imag * acc_bias_imag
            )
            contribution_imag = (
                token_diagonal_real * acc_bias_imag + token_diagonal_imag * acc_bias_real
            )
            destination_offset = (row * n_chunks + chunk) * state + token_destination
            tl.atomic_add(
                aggregate_bias_real_ptr + destination_offset,
                contribution_real,
                mask=lane_mask,
            )
            tl.atomic_add(
                aggregate_bias_imag_ptr + destination_offset,
                contribution_imag,
                mask=lane_mask,
            )
            tl.debug_barrier()

            acc_source = next_source
            acc_scale_real = next_scale_real
            acc_scale_imag = next_scale_imag
            acc_bias_real = tl.load(
                aggregate_bias_real_ptr + scratch_offset,
                mask=lane_mask,
                other=0.0,
            )
            acc_bias_imag = tl.load(
                aggregate_bias_imag_ptr + scratch_offset,
                mask=lane_mask,
                other=0.0,
            )

        chunk_offset = (row * n_chunks + chunk) * state + offsets
        tl.store(aggregate_source_ptr + chunk_offset, acc_source, mask=lane_mask)
        tl.store(aggregate_scale_real_ptr + chunk_offset, acc_scale_real, mask=lane_mask)
        tl.store(aggregate_scale_imag_ptr + chunk_offset, acc_scale_imag, mask=lane_mask)
        tl.store(aggregate_bias_real_ptr + chunk_offset, acc_bias_real, mask=lane_mask)
        tl.store(aggregate_bias_imag_ptr + chunk_offset, acc_bias_imag, mask=lane_mask)

    @triton.jit
    def _phase_b_kernel(
        aggregate_source_ptr,
        aggregate_scale_real_ptr,
        aggregate_scale_imag_ptr,
        aggregate_bias_real_ptr,
        aggregate_bias_imag_ptr,
        prefix_source_ptr,
        prefix_scale_real_ptr,
        prefix_scale_imag_ptr,
        prefix_bias_real_ptr,
        prefix_bias_imag_ptr,
        state,
        n_chunks,
        BLOCK_STATE: tl.constexpr,
    ):
        """Propagate exclusive sparse affine prefixes between chunks."""
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_STATE)
        lane_mask = offsets < state
        identity_source = tl.where(lane_mask, offsets, 0)

        prefix_source = identity_source
        prefix_scale_real = tl.where(lane_mask, 1.0, 0.0)
        prefix_scale_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        prefix_bias_real = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        prefix_bias_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)

        for chunk in range(0, n_chunks):
            chunk_offset = (row * n_chunks + chunk) * state + offsets
            tl.store(prefix_source_ptr + chunk_offset, prefix_source, mask=lane_mask)
            tl.store(prefix_scale_real_ptr + chunk_offset, prefix_scale_real, mask=lane_mask)
            tl.store(prefix_scale_imag_ptr + chunk_offset, prefix_scale_imag, mask=lane_mask)
            tl.store(prefix_bias_real_ptr + chunk_offset, prefix_bias_real, mask=lane_mask)
            tl.store(prefix_bias_imag_ptr + chunk_offset, prefix_bias_imag, mask=lane_mask)

            chunk_destination = tl.load(
                aggregate_source_ptr + chunk_offset,
                mask=lane_mask,
                other=0,
            ).to(tl.int32)
            chunk_scale_real = tl.load(
                aggregate_scale_real_ptr + chunk_offset, mask=lane_mask, other=1.0
            ).to(tl.float32)
            chunk_scale_imag = tl.load(
                aggregate_scale_imag_ptr + chunk_offset, mask=lane_mask, other=0.0
            ).to(tl.float32)
            chunk_bias_real = tl.load(
                aggregate_bias_real_ptr + chunk_offset, mask=lane_mask, other=0.0
            ).to(tl.float32)
            chunk_bias_imag = tl.load(
                aggregate_bias_imag_ptr + chunk_offset, mask=lane_mask, other=0.0
            ).to(tl.float32)

            next_source = tl.gather(chunk_destination, prefix_source, axis=0)
            later_scale_real = tl.gather(chunk_scale_real, prefix_source, axis=0)
            later_scale_imag = tl.gather(chunk_scale_imag, prefix_source, axis=0)
            next_scale_real = (
                prefix_scale_real * later_scale_real - prefix_scale_imag * later_scale_imag
            )
            next_scale_imag = (
                prefix_scale_real * later_scale_imag + prefix_scale_imag * later_scale_real
            )

            # Reuse this consumed aggregate's bias storage as collision-reduction scratch.
            tl.store(
                aggregate_bias_real_ptr + chunk_offset,
                chunk_bias_real,
                mask=lane_mask,
            )
            tl.store(
                aggregate_bias_imag_ptr + chunk_offset,
                chunk_bias_imag,
                mask=lane_mask,
            )
            tl.debug_barrier()
            contribution_real = (
                chunk_scale_real * prefix_bias_real - chunk_scale_imag * prefix_bias_imag
            )
            contribution_imag = (
                chunk_scale_real * prefix_bias_imag + chunk_scale_imag * prefix_bias_real
            )
            destination_offset = (row * n_chunks + chunk) * state + chunk_destination
            tl.atomic_add(
                aggregate_bias_real_ptr + destination_offset,
                contribution_real,
                mask=lane_mask,
            )
            tl.atomic_add(
                aggregate_bias_imag_ptr + destination_offset,
                contribution_imag,
                mask=lane_mask,
            )
            tl.debug_barrier()

            prefix_source = next_source
            prefix_scale_real = next_scale_real
            prefix_scale_imag = next_scale_imag
            prefix_bias_real = tl.load(
                aggregate_bias_real_ptr + chunk_offset,
                mask=lane_mask,
                other=0.0,
            )
            prefix_bias_imag = tl.load(
                aggregate_bias_imag_ptr + chunk_offset,
                mask=lane_mask,
                other=0.0,
            )

    @triton.jit
    def _phase_c_kernel(
        destination_ptr,
        diagonal_real_ptr,
        diagonal_imag_ptr,
        bias_real_ptr,
        bias_imag_ptr,
        prefix_bias_real_ptr,
        prefix_bias_imag_ptr,
        output_real_ptr,
        output_imag_ptr,
        time,
        state,
        n_chunks,
        chunk_size,
        BLOCK_STATE: tl.constexpr,
    ):
        """Replay each chunk from its globally correct incoming state."""
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_STATE)
        lane_mask = offsets < state
        chunk_offset = (row * n_chunks + chunk) * state + offsets
        state_real = tl.load(prefix_bias_real_ptr + chunk_offset, mask=lane_mask, other=0.0).to(
            tl.float32
        )
        state_imag = tl.load(prefix_bias_imag_ptr + chunk_offset, mask=lane_mask, other=0.0).to(
            tl.float32
        )

        for local_idx in range(0, chunk_size):
            token = chunk * chunk_size + local_idx
            token_mask = lane_mask & (token < time)
            token_offset = (row * time + token) * state + offsets
            token_destination = tl.load(
                destination_ptr + token_offset,
                mask=token_mask,
                other=0,
            ).to(tl.int32)
            diagonal_real = tl.load(
                diagonal_real_ptr + token_offset,
                mask=token_mask,
                other=1.0,
            ).to(tl.float32)
            diagonal_imag = tl.load(
                diagonal_imag_ptr + token_offset,
                mask=token_mask,
                other=0.0,
            ).to(tl.float32)
            bias_real = tl.load(bias_real_ptr + token_offset, mask=token_mask, other=0.0).to(
                tl.float32
            )
            bias_imag = tl.load(bias_imag_ptr + token_offset, mask=token_mask, other=0.0).to(
                tl.float32
            )
            tl.store(output_real_ptr + token_offset, bias_real, mask=token_mask)
            tl.store(output_imag_ptr + token_offset, bias_imag, mask=token_mask)
            tl.debug_barrier()
            contribution_real = diagonal_real * state_real - diagonal_imag * state_imag
            contribution_imag = diagonal_real * state_imag + diagonal_imag * state_real
            destination_offset = (row * time + token) * state + token_destination
            tl.atomic_add(
                output_real_ptr + destination_offset,
                contribution_real,
                mask=token_mask,
            )
            tl.atomic_add(
                output_imag_ptr + destination_offset,
                contribution_imag,
                mask=token_mask,
            )
            tl.debug_barrier()
            next_real = tl.load(
                output_real_ptr + token_offset,
                mask=token_mask,
                other=0.0,
            )
            next_imag = tl.load(
                output_imag_ptr + token_offset,
                mask=token_mask,
                other=0.0,
            )
            state_real = tl.where(token_mask, next_real, state_real)
            state_imag = tl.where(token_mask, next_imag, state_imag)


def flash_pd_triton_scan(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
    *,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Run the custom three-launch Triton forward prototype.

    Kernel A folds independent local chunks, Kernel B propagates exclusive chunk prefixes, and
    Kernel C replays/corrects every chunk. This is intentionally not an autograd implementation;
    callers requiring gradients must use the PyTorch recurrent or chunkwise reference.

    :param destination: Destination per source, shaped ``(batch, heads, time, state)``.
    :param diagonal: Complex64 diagonal factors with the same shape.
    :param bias: Complex64 affine terms with the same shape.
    :param chunk_size: Timesteps per independently processed chunk, at most 128.

    :returns: Complex64 states with the same shape.

    :raises RuntimeError: If Triton or a required tensor capability is unavailable.
    """
    capability = triton_capability(
        destination,
        diagonal,
        bias,
        chunk_size=chunk_size,
    )
    if not capability.available:
        raise RuntimeError(f"Flash PD Triton scan unavailable: {capability.reason}")
    if chunk_size < 1 or chunk_size > 128:
        raise ValueError(f"chunk_size must be in [1, 128], got {chunk_size}")
    assert triton is not None

    destination = destination.contiguous()
    diagonal = diagonal.contiguous()
    bias = bias.contiguous()
    batch, heads, time, state = destination.shape
    rows = batch * heads
    n_chunks = triton.cdiv(time, chunk_size)
    block_state = triton.next_power_of_2(state)

    destination_flat = destination.reshape(rows, time, state)
    diagonal_flat = diagonal.reshape(rows, time, state)
    bias_flat = bias.reshape(rows, time, state)
    diagonal_real = diagonal_flat.real.contiguous()
    diagonal_imag = diagonal_flat.imag.contiguous()
    bias_real, bias_imag = bias_flat.real.contiguous(), bias_flat.imag.contiguous()

    chunk_shape = (rows, n_chunks, state)
    aggregate_source = torch.empty(
        chunk_shape,
        dtype=torch.int32,
        device=destination.device,
    )
    aggregate_scale_real = torch.empty(
        chunk_shape,
        dtype=torch.float32,
        device=destination.device,
    )
    aggregate_scale_imag = torch.empty_like(aggregate_scale_real)
    aggregate_bias_real = torch.empty_like(aggregate_scale_real)
    aggregate_bias_imag = torch.empty_like(aggregate_scale_real)
    prefix_source = torch.empty_like(aggregate_source)
    prefix_scale_real = torch.empty_like(aggregate_scale_real)
    prefix_scale_imag = torch.empty_like(aggregate_scale_real)
    prefix_bias_real = torch.empty_like(aggregate_scale_real)
    prefix_bias_imag = torch.empty_like(aggregate_scale_real)
    output_real = torch.empty_like(diagonal_real)
    output_imag = torch.empty_like(diagonal_imag)

    with torch.cuda.device(destination.device):
        _phase_a_kernel[(rows, n_chunks)](
            destination_flat,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            aggregate_source,
            aggregate_scale_real,
            aggregate_scale_imag,
            aggregate_bias_real,
            aggregate_bias_imag,
            time,
            state,
            n_chunks,
            chunk_size,
            BLOCK_STATE=block_state,
            num_warps=1,
        )
        _phase_b_kernel[(rows,)](
            aggregate_source,
            aggregate_scale_real,
            aggregate_scale_imag,
            aggregate_bias_real,
            aggregate_bias_imag,
            prefix_source,
            prefix_scale_real,
            prefix_scale_imag,
            prefix_bias_real,
            prefix_bias_imag,
            state,
            n_chunks,
            BLOCK_STATE=block_state,
            num_warps=1,
        )
        _phase_c_kernel[(rows, n_chunks)](
            destination_flat,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            prefix_bias_real,
            prefix_bias_imag,
            output_real,
            output_imag,
            time,
            state,
            n_chunks,
            chunk_size,
            BLOCK_STATE=block_state,
            num_warps=1,
        )

    return torch.complex(output_real, output_imag).reshape(batch, heads, time, state)
