"""Shared-state Triton kernel for the fused Mamba-3 Flash-PD readout."""

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in environments without Triton.
    triton = None
    tl = None

MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES = frozenset({(8, 0), (12, 0)})
"""CUDA capabilities validated for this kernel: A100/sm80 and local sm120."""

MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE = 16
"""Largest bounded replay stride accepted by the production Triton path."""

_KERNEL_COUNTS = {"forward": 0, "backward": 0}

__all__ = [
    "MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE",
    "MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES",
    "Mamba3FlashPDTritonCapability",
    "estimate_mamba3_flash_pd_checkpoint_bytes",
    "estimate_mamba3_flash_pd_replay_work",
    "get_mamba3_flash_pd_kernel_counts",
    "mamba3_flash_pd_triton_capability",
    "mamba3_flash_pd_triton_readout",
    "reset_mamba3_flash_pd_kernel_counts",
]


@dataclass(frozen=True)
class Mamba3FlashPDTritonCapability:
    """
    Result of probing the production shared-state Flash-PD kernel.

    :param available: Whether all kernel constraints are satisfied.
    :param reason: Human-readable capability or fallback explanation.
    """

    available: bool
    reason: str


def _effective_checkpoint_stride(chunk_size: int, checkpoint_stride: int) -> int:
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if checkpoint_stride < 1:
        raise ValueError(f"checkpoint_stride must be positive, got {checkpoint_stride}")
    return min(chunk_size, checkpoint_stride)


def estimate_mamba3_flash_pd_replay_work(
    *,
    time: int,
    chunk_size: int,
    checkpoint_stride: int,
) -> int:
    """
    Count bounded replay-loop bodies per ``(batch, head, payload)`` program.

    Rank is absent because the recurrent state and its replay are shared across MIMO rank.
    Triton's runtime loops include masked tail iterations, so the count is
    ``ceil(T / C) * C * min(C, S)``.
    """
    if time < 1:
        raise ValueError(f"time must be positive, got {time}")
    effective_stride = _effective_checkpoint_stride(chunk_size, checkpoint_stride)
    n_chunks = (time + chunk_size - 1) // chunk_size
    return n_chunks * chunk_size * effective_stride


def estimate_mamba3_flash_pd_checkpoint_bytes(
    *,
    batch: int,
    heads: int,
    time: int,
    state: int,
    payload: int,
    chunk_size: int,
    checkpoint_stride: int,
) -> int:
    """Estimate saved float32 real/imag shared-state checkpoint bytes."""
    dimensions = {
        "batch": batch,
        "heads": heads,
        "time": time,
        "state": state,
        "payload": payload,
    }
    for name, value in dimensions.items():
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    effective_stride = _effective_checkpoint_stride(chunk_size, checkpoint_stride)
    n_chunks = (time + chunk_size - 1) // chunk_size
    checkpoints_per_chunk = (chunk_size + effective_stride - 1) // effective_stride
    elements = batch * heads * n_chunks * checkpoints_per_chunk * state * payload
    return elements * 2 * 4


def get_mamba3_flash_pd_kernel_counts() -> dict[str, int]:
    """Return process-local successful forward and backward kernel counts."""
    return dict(_KERNEL_COUNTS)


def reset_mamba3_flash_pd_kernel_counts() -> None:
    """Reset process-local kernel identity counters."""
    _KERNEL_COUNTS["forward"] = 0
    _KERNEL_COUNTS["backward"] = 0


def _selected_destinations(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dictionary_destination = dictionary_logits.argmax(dim=-2)
    route = selector_logits.argmax(dim=-1)
    batch, time, heads = route.shape
    state = dictionary_destination.shape[-1]
    dictionary = dictionary_destination.view(
        1,
        1,
        heads,
        dictionary_destination.shape[1],
        state,
    ).expand(batch, time, -1, -1, -1)
    index = route[..., None, None].expand(-1, -1, -1, 1, state)
    destination = torch.gather(dictionary, dim=3, index=index).squeeze(3)
    return destination.permute(0, 2, 1, 3).contiguous(), route


def mamba3_flash_pd_triton_capability(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    c_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
    *,
    chunk_size: int,
    checkpoint_stride: int = MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE,
) -> Mamba3FlashPDTritonCapability:
    """
    Probe whether compact shared-state Mamba-3 inputs are eligible for Triton.

    Rank participates only in the B/C boundary contractions. The recurrent state, checkpoints,
    collision handling, and bounded replay all contain ``state * payload`` elements independent
    of rank.
    """
    if chunk_size < 1 or chunk_size > 128:
        return Mamba3FlashPDTritonCapability(False, "chunk_size must be in [1, 128]")
    if checkpoint_stride < 1 or checkpoint_stride > MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE:
        return Mamba3FlashPDTritonCapability(
            False,
            "checkpoint_stride must be in " f"[1, {MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE}]",
        )
    if diagonal.dtype != torch.complex64:
        return Mamba3FlashPDTritonCapability(
            False,
            "shared-state MIMO Triton requires complex64 diagonal values and float32 real inputs",
        )
    real_tensors = (
        dictionary_logits,
        selector_logits,
        value,
        b_projection,
        c_projection,
        mimo_x,
        mimo_o,
        dt,
        lam,
    )
    if any(tensor.dtype != torch.float32 for tensor in real_tensors):
        return Mamba3FlashPDTritonCapability(False, "MIMO Triton supports float32 inputs only")
    if dictionary_logits.ndim != 4 or selector_logits.ndim != 4:
        return Mamba3FlashPDTritonCapability(False, "invalid dictionary or selector rank")
    heads, dictionary_size, state_rows, state = dictionary_logits.shape
    batch, time, selector_heads, selector_dictionary = selector_logits.shape
    if state_rows != state or (selector_heads, selector_dictionary) != (
        heads,
        dictionary_size,
    ):
        return Mamba3FlashPDTritonCapability(False, "dictionary and selector shapes do not match")
    if value.ndim != 4 or value.shape[:3] != (batch, heads, time):
        return Mamba3FlashPDTritonCapability(False, "value must have shape (batch, heads, time, P)")
    if b_projection.ndim != 5 or b_projection.shape[:3] != (batch, heads, time):
        return Mamba3FlashPDTritonCapability(
            False,
            "B must have shape (batch, heads, time, rank, state)",
        )
    if c_projection.shape != b_projection.shape or b_projection.shape[-1] != state:
        return Mamba3FlashPDTritonCapability(False, "B/C projection shapes do not match state")
    rank = b_projection.shape[-2]
    payload = value.shape[-1]
    if mimo_x.shape != (heads, rank, payload) or mimo_o.shape != mimo_x.shape:
        return Mamba3FlashPDTritonCapability(
            False,
            "MIMO write/read factors must have shape (heads, rank, payload)",
        )
    if diagonal.shape != (batch, heads, time, state):
        return Mamba3FlashPDTritonCapability(False, "diagonal shape does not match compact inputs")
    if dt.shape != (batch, heads, time) or lam.shape != dt.shape:
        return Mamba3FlashPDTritonCapability(False, "dt and lambda shapes do not match inputs")
    if min(batch, heads, time, rank, state, payload) < 1:
        return Mamba3FlashPDTritonCapability(False, "all MIMO dimensions must be non-empty")
    if state > 32:
        return Mamba3FlashPDTritonCapability(False, "MIMO Triton supports state size <= 32")
    if dictionary_size > 32:
        return Mamba3FlashPDTritonCapability(False, "MIMO Triton supports dictionary size <= 32")
    if rank * state > 256:
        return Mamba3FlashPDTritonCapability(
            False,
            "MIMO boundary contraction supports rank * state <= 256",
        )
    devices = {tensor.device for tensor in (diagonal, *real_tensors)}
    if len(devices) != 1:
        return Mamba3FlashPDTritonCapability(False, "all MIMO inputs must share one device")
    if triton is None or tl is None:
        return Mamba3FlashPDTritonCapability(False, "Triton is not installed")
    if not hasattr(tl, "gather") or not hasattr(tl, "range"):
        return Mamba3FlashPDTritonCapability(
            False, "installed Triton lacks gather or dynamic range"
        )
    if not diagonal.is_cuda:
        return Mamba3FlashPDTritonCapability(False, "CUDA tensors are required")
    capability = torch.cuda.get_device_capability(diagonal.device)
    if capability not in MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES:
        supported = ", ".join(
            f"sm{major}{minor}"
            for major, minor in sorted(MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES)
        )
        return Mamba3FlashPDTritonCapability(
            False,
            f"CUDA capability sm{capability[0]}{capability[1]} is unvalidated; supported: {supported}",
        )
    return Mamba3FlashPDTritonCapability(
        True,
        f"supported float32 shared-state MIMO kernel on sm{capability[0]}{capability[1]}",
    )


if triton is not None and tl is not None:

    @triton.jit
    def _mimo_shared_forward_kernel(
        destination_ptr,
        diagonal_real_ptr,
        diagonal_imag_ptr,
        value_ptr,
        b_projection_ptr,
        c_projection_ptr,
        mimo_x_ptr,
        mimo_o_ptr,
        dt_ptr,
        lam_ptr,
        checkpoint_real_ptr,
        checkpoint_imag_ptr,
        workspace_real_ptr,
        workspace_imag_ptr,
        output_ptr,
        time,
        heads,
        rank,
        state,
        payload,
        n_chunks,
        checkpoints_per_chunk,
        chunk_size,
        checkpoint_stride,
        BLOCK_RANK: tl.constexpr,
        BLOCK_STATE: tl.constexpr,
    ):
        """Run the recurrence with one ``N x P`` state and save bounded checkpoints."""
        row = tl.program_id(0)
        payload_index = tl.program_id(1)
        head_index = row % heads
        rank_offsets = tl.arange(0, BLOCK_RANK)
        state_offsets = tl.arange(0, BLOCK_STATE)
        rank_mask = rank_offsets < rank
        state_mask = state_offsets < state
        matrix_offsets = rank_offsets[:, None] * state + state_offsets[None, :]
        matrix_mask = rank_mask[:, None] & state_mask[None, :]
        workspace_base = row * state * payload
        workspace_offsets = workspace_base + state_offsets * payload + payload_index
        mimo_offsets = (head_index * rank + rank_offsets) * payload + payload_index
        mimo_x = tl.load(mimo_x_ptr + mimo_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        mimo_o = tl.load(mimo_o_ptr + mimo_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        state_real = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        state_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)

        for chunk in tl.range(0, n_chunks):
            for local_index in tl.range(0, chunk_size):
                token = chunk * chunk_size + local_index
                active = token < time
                row_token = row * time + token
                checkpoint = local_index // checkpoint_stride
                checkpoint_base = (
                    ((row * n_chunks + chunk) * checkpoints_per_chunk + checkpoint)
                    * state
                    * payload
                )
                checkpoint_offsets = checkpoint_base + state_offsets * payload + payload_index
                checkpoint_mask = state_mask & active & ((local_index % checkpoint_stride) == 0)
                tl.store(
                    checkpoint_real_ptr + checkpoint_offsets,
                    state_real,
                    mask=checkpoint_mask,
                )
                tl.store(
                    checkpoint_imag_ptr + checkpoint_offsets,
                    state_imag,
                    mask=checkpoint_mask,
                )

                previous_active = active & (token > 0)
                previous_row_token = row_token - 1
                token_dt = tl.load(dt_ptr + row_token, mask=active, other=0.0).to(tl.float32)
                token_lam = tl.load(lam_ptr + row_token, mask=active, other=0.0).to(tl.float32)
                previous_scale = (1.0 - token_lam) * token_dt
                current_scale = token_lam * token_dt
                previous_value = tl.load(
                    value_ptr + previous_row_token * payload + payload_index,
                    mask=previous_active,
                    other=0.0,
                ).to(tl.float32)
                current_value = tl.load(
                    value_ptr + row_token * payload + payload_index,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                previous_b = tl.load(
                    b_projection_ptr + previous_row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & previous_active,
                    other=0.0,
                ).to(tl.float32)
                current_b = tl.load(
                    b_projection_ptr + row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & active,
                    other=0.0,
                ).to(tl.float32)
                previous_drive = tl.sum(previous_b * mimo_x[:, None], axis=0) * previous_value
                current_drive = tl.sum(current_b * mimo_x[:, None], axis=0) * current_value
                source_real = state_real + previous_scale * previous_drive
                source_imag = state_imag
                diagonal_real = tl.load(
                    diagonal_real_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=1.0,
                ).to(tl.float32)
                diagonal_imag = tl.load(
                    diagonal_imag_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)
                contribution_real = diagonal_real * source_real - diagonal_imag * source_imag
                contribution_imag = diagonal_real * source_imag + diagonal_imag * source_real
                destination = tl.load(
                    destination_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=state_offsets,
                ).to(tl.int32)
                destination_offsets = workspace_base + destination * payload + payload_index
                tl.store(
                    workspace_real_ptr + workspace_offsets,
                    current_scale * current_drive,
                    mask=state_mask & active,
                )
                tl.store(
                    workspace_imag_ptr + workspace_offsets,
                    0.0,
                    mask=state_mask & active,
                )
                tl.debug_barrier()
                tl.atomic_add(
                    workspace_real_ptr + destination_offsets,
                    contribution_real,
                    mask=state_mask & active,
                )
                tl.atomic_add(
                    workspace_imag_ptr + destination_offsets,
                    contribution_imag,
                    mask=state_mask & active,
                )
                tl.debug_barrier()
                next_real = tl.load(
                    workspace_real_ptr + workspace_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)
                next_imag = tl.load(
                    workspace_imag_ptr + workspace_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)
                state_real = tl.where(active, next_real, state_real)
                state_imag = tl.where(active, next_imag, state_imag)

                token_c = tl.load(
                    c_projection_ptr + row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & active,
                    other=0.0,
                ).to(tl.float32)
                rank_readout = tl.sum(token_c * state_real[None, :], axis=1)
                readout = tl.sum(
                    tl.where(rank_mask, rank_readout * mimo_o, 0.0),
                    axis=0,
                )
                tl.store(
                    output_ptr + row_token * payload + payload_index,
                    readout,
                    mask=active,
                )

    @triton.jit
    def _mimo_shared_backward_kernel(
        destination_ptr,
        dictionary_destination_ptr,
        route_ptr,
        diagonal_real_ptr,
        diagonal_imag_ptr,
        value_ptr,
        b_projection_ptr,
        c_projection_ptr,
        mimo_x_ptr,
        mimo_o_ptr,
        dt_ptr,
        lam_ptr,
        checkpoint_real_ptr,
        checkpoint_imag_ptr,
        workspace_real_ptr,
        workspace_imag_ptr,
        grad_readout_ptr,
        grad_diagonal_real_ptr,
        grad_diagonal_imag_ptr,
        grad_value_ptr,
        grad_b_projection_ptr,
        grad_c_projection_ptr,
        grad_mimo_x_ptr,
        grad_mimo_o_ptr,
        grad_dt_ptr,
        grad_lam_ptr,
        raw_selector_ptr,
        raw_dictionary_ptr,
        time,
        heads,
        dictionary_size,
        rank,
        state,
        payload,
        n_chunks,
        checkpoints_per_chunk,
        chunk_size,
        checkpoint_stride,
        BLOCK_RANK: tl.constexpr,
        BLOCK_STATE: tl.constexpr,
        BLOCK_DICTIONARY: tl.constexpr,
    ):
        """Reverse the shared recurrence with replay bounded by ``checkpoint_stride``."""
        row = tl.program_id(0)
        payload_index = tl.program_id(1)
        batch_index = row // heads
        head_index = row % heads
        rank_offsets = tl.arange(0, BLOCK_RANK)
        state_offsets = tl.arange(0, BLOCK_STATE)
        dictionary_offsets = tl.arange(0, BLOCK_DICTIONARY)
        rank_mask = rank_offsets < rank
        state_mask = state_offsets < state
        dictionary_mask = dictionary_offsets < dictionary_size
        matrix_offsets = rank_offsets[:, None] * state + state_offsets[None, :]
        matrix_mask = rank_mask[:, None] & state_mask[None, :]
        workspace_base = row * state * payload
        workspace_offsets = workspace_base + state_offsets * payload + payload_index
        mimo_offsets = (head_index * rank + rank_offsets) * payload + payload_index
        mimo_x = tl.load(mimo_x_ptr + mimo_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        mimo_o = tl.load(mimo_o_ptr + mimo_offsets, mask=rank_mask, other=0.0).to(tl.float32)
        carry_real = tl.zeros((BLOCK_STATE,), dtype=tl.float32)
        carry_imag = tl.zeros((BLOCK_STATE,), dtype=tl.float32)

        for reverse_chunk in tl.range(0, n_chunks):
            chunk = n_chunks - 1 - reverse_chunk
            for reverse_index in tl.range(0, chunk_size):
                local_index = chunk_size - 1 - reverse_index
                token = chunk * chunk_size + local_index
                active = token < time
                checkpoint = local_index // checkpoint_stride
                checkpoint_base = (
                    ((row * n_chunks + chunk) * checkpoints_per_chunk + checkpoint)
                    * state
                    * payload
                )
                checkpoint_offsets = checkpoint_base + state_offsets * payload + payload_index
                previous_state_real = tl.load(
                    checkpoint_real_ptr + checkpoint_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)
                previous_state_imag = tl.load(
                    checkpoint_imag_ptr + checkpoint_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)

                for replay_index in tl.range(0, checkpoint_stride):
                    replay_local = checkpoint * checkpoint_stride + replay_index
                    replay_token = chunk * chunk_size + replay_local
                    replay_active = active & (replay_local < local_index) & (replay_token < time)
                    replay_row_token = row * time + replay_token
                    replay_previous_active = replay_active & (replay_token > 0)
                    replay_previous_row_token = replay_row_token - 1
                    replay_dt = tl.load(
                        dt_ptr + replay_row_token,
                        mask=replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_lam = tl.load(
                        lam_ptr + replay_row_token,
                        mask=replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_previous_value = tl.load(
                        value_ptr + replay_previous_row_token * payload + payload_index,
                        mask=replay_previous_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_current_value = tl.load(
                        value_ptr + replay_row_token * payload + payload_index,
                        mask=replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_previous_b = tl.load(
                        b_projection_ptr
                        + replay_previous_row_token * rank * state
                        + matrix_offsets,
                        mask=matrix_mask & replay_previous_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_current_b = tl.load(
                        b_projection_ptr + replay_row_token * rank * state + matrix_offsets,
                        mask=matrix_mask & replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_previous_drive = (
                        tl.sum(replay_previous_b * mimo_x[:, None], axis=0) * replay_previous_value
                    )
                    replay_current_drive = (
                        tl.sum(replay_current_b * mimo_x[:, None], axis=0) * replay_current_value
                    )
                    replay_source_real = (
                        previous_state_real + (1.0 - replay_lam) * replay_dt * replay_previous_drive
                    )
                    replay_source_imag = previous_state_imag
                    replay_diagonal_real = tl.load(
                        diagonal_real_ptr + replay_row_token * state + state_offsets,
                        mask=state_mask & replay_active,
                        other=1.0,
                    ).to(tl.float32)
                    replay_diagonal_imag = tl.load(
                        diagonal_imag_ptr + replay_row_token * state + state_offsets,
                        mask=state_mask & replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_contribution_real = (
                        replay_diagonal_real * replay_source_real
                        - replay_diagonal_imag * replay_source_imag
                    )
                    replay_contribution_imag = (
                        replay_diagonal_real * replay_source_imag
                        + replay_diagonal_imag * replay_source_real
                    )
                    replay_destination = tl.load(
                        destination_ptr + replay_row_token * state + state_offsets,
                        mask=state_mask & replay_active,
                        other=state_offsets,
                    ).to(tl.int32)
                    replay_destination_offsets = (
                        workspace_base + replay_destination * payload + payload_index
                    )
                    tl.store(
                        workspace_real_ptr + workspace_offsets,
                        replay_lam * replay_dt * replay_current_drive,
                        mask=state_mask & replay_active,
                    )
                    tl.store(
                        workspace_imag_ptr + workspace_offsets,
                        0.0,
                        mask=state_mask & replay_active,
                    )
                    tl.debug_barrier()
                    tl.atomic_add(
                        workspace_real_ptr + replay_destination_offsets,
                        replay_contribution_real,
                        mask=state_mask & replay_active,
                    )
                    tl.atomic_add(
                        workspace_imag_ptr + replay_destination_offsets,
                        replay_contribution_imag,
                        mask=state_mask & replay_active,
                    )
                    tl.debug_barrier()
                    replay_next_real = tl.load(
                        workspace_real_ptr + workspace_offsets,
                        mask=state_mask & replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    replay_next_imag = tl.load(
                        workspace_imag_ptr + workspace_offsets,
                        mask=state_mask & replay_active,
                        other=0.0,
                    ).to(tl.float32)
                    previous_state_real = tl.where(
                        replay_active,
                        replay_next_real,
                        previous_state_real,
                    )
                    previous_state_imag = tl.where(
                        replay_active,
                        replay_next_imag,
                        previous_state_imag,
                    )

                row_token = row * time + token
                output_token = (batch_index * time + token) * heads + head_index
                previous_active = active & (token > 0)
                previous_row_token = row_token - 1
                token_dt = tl.load(dt_ptr + row_token, mask=active, other=0.0).to(tl.float32)
                token_lam = tl.load(lam_ptr + row_token, mask=active, other=0.0).to(tl.float32)
                previous_scale = (1.0 - token_lam) * token_dt
                current_scale = token_lam * token_dt
                previous_value = tl.load(
                    value_ptr + previous_row_token * payload + payload_index,
                    mask=previous_active,
                    other=0.0,
                ).to(tl.float32)
                current_value = tl.load(
                    value_ptr + row_token * payload + payload_index,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                previous_b = tl.load(
                    b_projection_ptr + previous_row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & previous_active,
                    other=0.0,
                ).to(tl.float32)
                current_b = tl.load(
                    b_projection_ptr + row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & active,
                    other=0.0,
                ).to(tl.float32)
                previous_drive = tl.sum(previous_b * mimo_x[:, None], axis=0) * previous_value
                current_drive = tl.sum(current_b * mimo_x[:, None], axis=0) * current_value
                source_real = previous_state_real + previous_scale * previous_drive
                source_imag = previous_state_imag
                diagonal_real = tl.load(
                    diagonal_real_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=1.0,
                ).to(tl.float32)
                diagonal_imag = tl.load(
                    diagonal_imag_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)
                transitioned_source_real = diagonal_real * source_real - diagonal_imag * source_imag
                transitioned_source_imag = diagonal_real * source_imag + diagonal_imag * source_real
                token_destination = tl.load(
                    destination_ptr + row_token * state + state_offsets,
                    mask=state_mask & active,
                    other=state_offsets,
                ).to(tl.int32)
                token_destination_offsets = (
                    workspace_base + token_destination * payload + payload_index
                )
                tl.store(
                    workspace_real_ptr + workspace_offsets,
                    current_scale * current_drive,
                    mask=state_mask & active,
                )
                tl.store(
                    workspace_imag_ptr + workspace_offsets,
                    0.0,
                    mask=state_mask & active,
                )
                tl.debug_barrier()
                tl.atomic_add(
                    workspace_real_ptr + token_destination_offsets,
                    transitioned_source_real,
                    mask=state_mask & active,
                )
                tl.atomic_add(
                    workspace_imag_ptr + token_destination_offsets,
                    transitioned_source_imag,
                    mask=state_mask & active,
                )
                tl.debug_barrier()
                token_state_real = tl.load(
                    workspace_real_ptr + workspace_offsets,
                    mask=state_mask & active,
                    other=0.0,
                ).to(tl.float32)

                readout_grad = tl.load(
                    grad_readout_ptr + output_token * payload + payload_index,
                    mask=active,
                    other=0.0,
                ).to(tl.float32)
                token_c = tl.load(
                    c_projection_ptr + row_token * rank * state + matrix_offsets,
                    mask=matrix_mask & active,
                    other=0.0,
                ).to(tl.float32)
                tl.atomic_add(
                    grad_c_projection_ptr + row_token * rank * state + matrix_offsets,
                    readout_grad * mimo_o[:, None] * token_state_real[None, :],
                    mask=matrix_mask & active,
                )
                rank_readout = tl.sum(token_c * token_state_real[None, :], axis=1)
                tl.atomic_add(
                    grad_mimo_o_ptr + mimo_offsets,
                    readout_grad * rank_readout,
                    mask=rank_mask & active,
                )
                readout_state_grad = tl.sum(token_c * mimo_o[:, None], axis=0) * readout_grad
                total_real = carry_real + readout_state_grad
                total_imag = carry_imag

                grad_current_scale = tl.sum(
                    tl.where(state_mask, total_real * current_drive, 0.0),
                    axis=0,
                )
                grad_current_drive = current_scale * total_real
                tl.atomic_add(
                    grad_b_projection_ptr + row_token * rank * state + matrix_offsets,
                    grad_current_drive[None, :] * mimo_x[:, None] * current_value,
                    mask=matrix_mask & active,
                )
                grad_mimo_x_current = (
                    tl.sum(grad_current_drive[None, :] * current_b, axis=1) * current_value
                )
                tl.atomic_add(
                    grad_mimo_x_ptr + mimo_offsets,
                    grad_mimo_x_current,
                    mask=rank_mask & active,
                )
                grad_current_value = tl.sum(
                    tl.where(
                        matrix_mask,
                        grad_current_drive[None, :] * current_b * mimo_x[:, None],
                        0.0,
                    ),
                    axis=1,
                )
                tl.atomic_add(
                    grad_value_ptr + row_token * payload + payload_index,
                    tl.sum(grad_current_value, axis=0),
                    mask=active,
                )

                grad_at_destination_real = tl.gather(
                    total_real,
                    token_destination,
                    axis=0,
                )
                grad_at_destination_imag = tl.gather(
                    total_imag,
                    token_destination,
                    axis=0,
                )
                tl.atomic_add(
                    grad_diagonal_real_ptr + row_token * state + state_offsets,
                    grad_at_destination_real * source_real + grad_at_destination_imag * source_imag,
                    mask=state_mask & active,
                )
                tl.atomic_add(
                    grad_diagonal_imag_ptr + row_token * state + state_offsets,
                    grad_at_destination_imag * source_real - grad_at_destination_real * source_imag,
                    mask=state_mask & active,
                )
                grad_source_real = (
                    grad_at_destination_real * diagonal_real
                    + grad_at_destination_imag * diagonal_imag
                )
                grad_source_imag = (
                    grad_at_destination_imag * diagonal_real
                    - grad_at_destination_real * diagonal_imag
                )
                grad_previous_scale = tl.sum(
                    tl.where(state_mask, grad_source_real * previous_drive, 0.0),
                    axis=0,
                )
                grad_previous_drive = previous_scale * grad_source_real
                tl.atomic_add(
                    grad_b_projection_ptr + previous_row_token * rank * state + matrix_offsets,
                    grad_previous_drive[None, :] * mimo_x[:, None] * previous_value,
                    mask=matrix_mask & previous_active,
                )
                grad_mimo_x_previous = (
                    tl.sum(grad_previous_drive[None, :] * previous_b, axis=1) * previous_value
                )
                tl.atomic_add(
                    grad_mimo_x_ptr + mimo_offsets,
                    grad_mimo_x_previous,
                    mask=rank_mask & previous_active,
                )
                grad_previous_value = tl.sum(
                    tl.where(
                        matrix_mask,
                        grad_previous_drive[None, :] * previous_b * mimo_x[:, None],
                        0.0,
                    ),
                    axis=1,
                )
                tl.atomic_add(
                    grad_value_ptr + previous_row_token * payload + payload_index,
                    tl.sum(grad_previous_value, axis=0),
                    mask=previous_active,
                )
                tl.atomic_add(
                    grad_dt_ptr + row_token,
                    grad_previous_scale * (1.0 - token_lam) + grad_current_scale * token_lam,
                    mask=active,
                )
                tl.atomic_add(
                    grad_lam_ptr + row_token,
                    (grad_current_scale - grad_previous_scale) * token_dt,
                    mask=active,
                )

                candidate_destination = tl.load(
                    dictionary_destination_ptr
                    + (head_index * dictionary_size + dictionary_offsets[:, None]) * state
                    + state_offsets[None, :],
                    mask=dictionary_mask[:, None] & state_mask[None, :] & active,
                    other=state_offsets[None, :],
                ).to(tl.int32)
                candidate_total_real = tl.gather(
                    tl.broadcast_to(
                        total_real[None, :],
                        (BLOCK_DICTIONARY, BLOCK_STATE),
                    ),
                    candidate_destination,
                    axis=1,
                )
                candidate_total_imag = tl.gather(
                    tl.broadcast_to(
                        total_imag[None, :],
                        (BLOCK_DICTIONARY, BLOCK_STATE),
                    ),
                    candidate_destination,
                    axis=1,
                )
                route_score = tl.sum(
                    tl.where(
                        dictionary_mask[:, None] & state_mask[None, :],
                        candidate_total_real * transitioned_source_real[None, :]
                        + candidate_total_imag * transitioned_source_imag[None, :],
                        0.0,
                    ),
                    axis=1,
                )
                selector_token = (batch_index * time + token) * heads + head_index
                tl.atomic_add(
                    raw_selector_ptr + selector_token * dictionary_size + dictionary_offsets,
                    route_score,
                    mask=dictionary_mask & active,
                )

                dictionary_outer = (
                    total_real[:, None] * transitioned_source_real[None, :]
                    + total_imag[:, None] * transitioned_source_imag[None, :]
                )
                selected_route = tl.load(
                    route_ptr + selector_token,
                    mask=active,
                    other=0,
                ).to(tl.int32)
                dictionary_matrix_offsets = (
                    (head_index * dictionary_size + selected_route) * state + state_offsets[:, None]
                ) * state + state_offsets[None, :]
                tl.atomic_add(
                    raw_dictionary_ptr + dictionary_matrix_offsets,
                    dictionary_outer,
                    mask=state_mask[:, None] & state_mask[None, :] & active,
                )
                carry_real = tl.where(active, grad_source_real, carry_real)
                carry_imag = tl.where(active, grad_source_imag, carry_imag)

    @triton.jit
    def _mimo_shared_ste_kernel(
        selector_logits_ptr,
        dictionary_logits_ptr,
        raw_selector_ptr,
        raw_dictionary_ptr,
        grad_selector_ptr,
        grad_dictionary_ptr,
        router_rows,
        heads,
        dictionary_size,
        state,
        temperature,
        BLOCK_DICTIONARY: tl.constexpr,
        BLOCK_STATE: tl.constexpr,
    ):
        """Apply both tempered-softmax STE Jacobians."""
        program = tl.program_id(0)
        dictionary_rows = heads * dictionary_size * state
        if program < router_rows:
            dictionary_offsets = tl.arange(0, BLOCK_DICTIONARY)
            dictionary_mask = dictionary_offsets < dictionary_size
            logits = (
                tl.load(
                    selector_logits_ptr + program * dictionary_size + dictionary_offsets,
                    mask=dictionary_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                / temperature
            )
            logits -= tl.max(logits, axis=0)
            probability = tl.exp(logits)
            probability /= tl.sum(probability, axis=0)
            raw = tl.load(
                raw_selector_ptr + program * dictionary_size + dictionary_offsets,
                mask=dictionary_mask,
                other=0.0,
            ).to(tl.float32)
            centered = raw - tl.sum(probability * raw, axis=0)
            tl.store(
                grad_selector_ptr + program * dictionary_size + dictionary_offsets,
                probability * centered / temperature,
                mask=dictionary_mask,
            )
        elif program < router_rows + dictionary_rows:
            dictionary_program = program - router_rows
            source = dictionary_program % state
            dictionary_program //= state
            dictionary_index = dictionary_program % dictionary_size
            head = dictionary_program // dictionary_size
            state_offsets = tl.arange(0, BLOCK_STATE)
            state_mask = state_offsets < state
            offsets = (
                (head * dictionary_size + dictionary_index) * state + state_offsets
            ) * state + source
            logits = (
                tl.load(
                    dictionary_logits_ptr + offsets,
                    mask=state_mask,
                    other=-float("inf"),
                ).to(tl.float32)
                / temperature
            )
            logits -= tl.max(logits, axis=0)
            probability = tl.exp(logits)
            probability /= tl.sum(probability, axis=0)
            raw = tl.load(
                raw_dictionary_ptr + offsets,
                mask=state_mask,
                other=0.0,
            ).to(tl.float32)
            centered = raw - tl.sum(probability * raw, axis=0)
            tl.store(
                grad_dictionary_ptr + offsets,
                probability * centered / temperature,
                mask=state_mask,
            )


def _launch_mimo_forward(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    c_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
    *,
    chunk_size: int,
    checkpoint_stride: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the shared-state forward without host-side token loops."""
    assert triton is not None
    destination = destination.contiguous()
    diagonal = diagonal.contiguous()
    value = value.contiguous()
    b_projection = b_projection.contiguous()
    c_projection = c_projection.contiguous()
    mimo_x = mimo_x.contiguous()
    mimo_o = mimo_o.contiguous()
    dt = dt.contiguous()
    lam = lam.contiguous()
    batch, heads, time, payload = value.shape
    rank, state = b_projection.shape[-2:]
    rows = batch * heads
    n_chunks = triton.cdiv(time, chunk_size)
    checkpoint_stride = _effective_checkpoint_stride(chunk_size, checkpoint_stride)
    checkpoints_per_chunk = triton.cdiv(chunk_size, checkpoint_stride)
    block_rank = triton.next_power_of_2(rank)
    block_state = triton.next_power_of_2(state)
    block_matrix = block_rank * block_state
    num_warps = (
        1 if block_matrix <= 32 else 2 if block_matrix <= 64 else 4 if block_matrix <= 128 else 8
    )

    destination_flat = destination.reshape(rows, time, state)
    diagonal_flat = diagonal.reshape(rows, time, state)
    value_flat = value.reshape(rows, time, payload)
    b_flat = b_projection.reshape(rows, time, rank, state)
    c_flat = c_projection.reshape(rows, time, rank, state)
    dt_flat = dt.reshape(rows, time)
    lam_flat = lam.reshape(rows, time)
    diagonal_real = diagonal_flat.real.contiguous()
    diagonal_imag = diagonal_flat.imag.contiguous()
    checkpoint_shape = (
        rows,
        n_chunks,
        checkpoints_per_chunk,
        state,
        payload,
    )
    checkpoint_real = torch.empty(
        checkpoint_shape,
        dtype=torch.float32,
        device=value.device,
    )
    checkpoint_imag = torch.empty_like(checkpoint_real)
    workspace_shape = (rows, state, payload)
    workspace_real = torch.empty(workspace_shape, dtype=torch.float32, device=value.device)
    workspace_imag = torch.empty_like(workspace_real)
    output = torch.empty((rows, time, payload), dtype=torch.float32, device=value.device)

    with torch.cuda.device(value.device):
        _mimo_shared_forward_kernel[(rows, payload)](
            destination_flat,
            diagonal_real,
            diagonal_imag,
            value_flat,
            b_flat,
            c_flat,
            mimo_x,
            mimo_o,
            dt_flat,
            lam_flat,
            checkpoint_real,
            checkpoint_imag,
            workspace_real,
            workspace_imag,
            output,
            time,
            heads,
            rank,
            state,
            payload,
            n_chunks,
            checkpoints_per_chunk,
            chunk_size,
            checkpoint_stride,
            BLOCK_RANK=block_rank,
            BLOCK_STATE=block_state,
            num_warps=num_warps,
        )

    readout = output.reshape(batch, heads, time, payload).permute(0, 2, 1, 3).contiguous()
    saved_shape = (
        batch,
        heads,
        n_chunks,
        checkpoints_per_chunk,
        state,
        payload,
    )
    return (
        readout,
        checkpoint_real.reshape(saved_shape),
        checkpoint_imag.reshape(saved_shape),
    )


def _launch_mimo_backward(
    destination: torch.Tensor,
    dictionary_destination: torch.Tensor,
    route: torch.Tensor,
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    c_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
    checkpoint_real: torch.Tensor,
    checkpoint_imag: torch.Tensor,
    grad_readout: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    checkpoint_stride: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Launch bounded shared-state replay and every analytic/STE gradient."""
    assert triton is not None
    destination = destination.contiguous()
    dictionary_destination = dictionary_destination.contiguous()
    route = route.contiguous()
    dictionary_logits = dictionary_logits.contiguous()
    selector_logits = selector_logits.contiguous()
    diagonal = diagonal.contiguous()
    value = value.contiguous()
    b_projection = b_projection.contiguous()
    c_projection = c_projection.contiguous()
    mimo_x = mimo_x.contiguous()
    mimo_o = mimo_o.contiguous()
    dt = dt.contiguous()
    lam = lam.contiguous()
    checkpoint_real = checkpoint_real.contiguous()
    checkpoint_imag = checkpoint_imag.contiguous()
    grad_readout = grad_readout.contiguous()
    batch, heads, time, payload = value.shape
    dictionary_size = dictionary_logits.shape[1]
    rank, state = b_projection.shape[-2:]
    rows = batch * heads
    n_chunks = triton.cdiv(time, chunk_size)
    checkpoint_stride = _effective_checkpoint_stride(chunk_size, checkpoint_stride)
    checkpoints_per_chunk = triton.cdiv(chunk_size, checkpoint_stride)
    block_rank = triton.next_power_of_2(rank)
    block_state = triton.next_power_of_2(state)
    block_dictionary = triton.next_power_of_2(dictionary_size)
    block_matrix = block_rank * block_state
    num_warps = (
        1 if block_matrix <= 32 else 2 if block_matrix <= 64 else 4 if block_matrix <= 128 else 8
    )

    destination_flat = destination.reshape(rows, time, state)
    diagonal_flat = diagonal.reshape(rows, time, state)
    value_flat = value.reshape(rows, time, payload)
    b_flat = b_projection.reshape(rows, time, rank, state)
    c_flat = c_projection.reshape(rows, time, rank, state)
    dt_flat = dt.reshape(rows, time)
    lam_flat = lam.reshape(rows, time)
    diagonal_real = diagonal_flat.real.contiguous()
    diagonal_imag = diagonal_flat.imag.contiguous()
    checkpoint_real_flat = checkpoint_real.reshape(
        rows,
        n_chunks,
        checkpoints_per_chunk,
        state,
        payload,
    )
    checkpoint_imag_flat = checkpoint_imag.reshape_as(checkpoint_real_flat)
    workspace_shape = (rows, state, payload)
    workspace_real = torch.empty(workspace_shape, dtype=torch.float32, device=value.device)
    workspace_imag = torch.empty_like(workspace_real)
    grad_dictionary = torch.empty_like(dictionary_logits)
    grad_selector = torch.empty_like(selector_logits)
    grad_diagonal_real = torch.zeros_like(diagonal_real)
    grad_diagonal_imag = torch.zeros_like(diagonal_imag)
    grad_value = torch.zeros_like(value_flat)
    grad_b_projection = torch.zeros_like(b_flat)
    grad_c_projection = torch.zeros_like(c_flat)
    grad_mimo_x = torch.zeros_like(mimo_x)
    grad_mimo_o = torch.zeros_like(mimo_o)
    grad_dt = torch.zeros_like(dt_flat)
    grad_lam = torch.zeros_like(lam_flat)
    raw_selector = torch.zeros_like(selector_logits)
    raw_dictionary = torch.zeros_like(dictionary_logits)
    router_rows = batch * time * heads
    dictionary_rows = heads * dictionary_size * state

    with torch.cuda.device(value.device):
        _mimo_shared_backward_kernel[(rows, payload)](
            destination_flat,
            dictionary_destination,
            route,
            diagonal_real,
            diagonal_imag,
            value_flat,
            b_flat,
            c_flat,
            mimo_x,
            mimo_o,
            dt_flat,
            lam_flat,
            checkpoint_real_flat,
            checkpoint_imag_flat,
            workspace_real,
            workspace_imag,
            grad_readout,
            grad_diagonal_real,
            grad_diagonal_imag,
            grad_value,
            grad_b_projection,
            grad_c_projection,
            grad_mimo_x,
            grad_mimo_o,
            grad_dt,
            grad_lam,
            raw_selector,
            raw_dictionary,
            time,
            heads,
            dictionary_size,
            rank,
            state,
            payload,
            n_chunks,
            checkpoints_per_chunk,
            chunk_size,
            checkpoint_stride,
            BLOCK_RANK=block_rank,
            BLOCK_STATE=block_state,
            BLOCK_DICTIONARY=block_dictionary,
            num_warps=num_warps,
        )
        _mimo_shared_ste_kernel[(router_rows + dictionary_rows,)](
            selector_logits,
            dictionary_logits,
            raw_selector,
            raw_dictionary,
            grad_selector,
            grad_dictionary,
            router_rows,
            heads,
            dictionary_size,
            state,
            temperature,
            BLOCK_DICTIONARY=block_dictionary,
            BLOCK_STATE=block_state,
            num_warps=1,
        )

    grad_diagonal = torch.complex(grad_diagonal_real, grad_diagonal_imag).reshape_as(diagonal)
    return (
        grad_dictionary,
        grad_selector,
        grad_diagonal,
        grad_value.reshape_as(value),
        grad_b_projection.reshape_as(b_projection),
        grad_c_projection.reshape_as(c_projection),
        grad_mimo_x,
        grad_mimo_o,
        grad_dt.reshape_as(dt),
        grad_lam.reshape_as(lam),
    )


class _Mamba3FlashPDTritonReadout(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        dictionary_logits: torch.Tensor,
        selector_logits: torch.Tensor,
        diagonal: torch.Tensor,
        value: torch.Tensor,
        b_projection: torch.Tensor,
        c_projection: torch.Tensor,
        mimo_x: torch.Tensor,
        mimo_o: torch.Tensor,
        dt: torch.Tensor,
        lam: torch.Tensor,
        temperature: float,
        chunk_size: int,
        checkpoint_stride: int,
    ) -> torch.Tensor:
        destination, _ = _selected_destinations(dictionary_logits, selector_logits)
        readout, checkpoint_real, checkpoint_imag = _launch_mimo_forward(
            destination.detach(),
            diagonal.detach(),
            value.detach(),
            b_projection.detach(),
            c_projection.detach(),
            mimo_x.detach(),
            mimo_o.detach(),
            dt.detach(),
            lam.detach(),
            chunk_size=chunk_size,
            checkpoint_stride=checkpoint_stride,
        )
        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            diagonal,
            value,
            b_projection,
            c_projection,
            mimo_x,
            mimo_o,
            dt,
            lam,
            checkpoint_real,
            checkpoint_imag,
        )
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.checkpoint_stride = checkpoint_stride
        _KERNEL_COUNTS["forward"] += 1
        return readout

    @staticmethod
    def backward(
        ctx: Any,
        grad_readout: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        (
            dictionary_logits,
            selector_logits,
            diagonal,
            value,
            b_projection,
            c_projection,
            mimo_x,
            mimo_o,
            dt,
            lam,
            checkpoint_real,
            checkpoint_imag,
        ) = ctx.saved_tensors
        destination, route = _selected_destinations(dictionary_logits, selector_logits)
        dictionary_destination = dictionary_logits.argmax(dim=-2)
        gradients = _launch_mimo_backward(
            destination.detach(),
            dictionary_destination.detach(),
            route.detach(),
            dictionary_logits.detach(),
            selector_logits.detach(),
            diagonal.detach(),
            value.detach(),
            b_projection.detach(),
            c_projection.detach(),
            mimo_x.detach(),
            mimo_o.detach(),
            dt.detach(),
            lam.detach(),
            checkpoint_real.detach(),
            checkpoint_imag.detach(),
            grad_readout.detach(),
            temperature=ctx.temperature,
            chunk_size=ctx.chunk_size,
            checkpoint_stride=ctx.checkpoint_stride,
        )
        _KERNEL_COUNTS["backward"] += 1
        return (*gradients, None, None, None)


def mamba3_flash_pd_triton_readout(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    c_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
    checkpoint_stride: int = MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE,
) -> torch.Tensor:
    """
    Run the shared-state Triton forward and bounded-replay backward.

    The only recurrent/checkpoint/workspace shape is ``(B, H, N, P)``. Rank remains in B/C and
    the learned ``mimo_x``/``mimo_o`` boundary contractions, matching official Mamba-3.

    :raises RuntimeError: If any input or device capability is unsupported.
    """
    capability = mamba3_flash_pd_triton_capability(
        dictionary_logits,
        selector_logits,
        diagonal,
        value,
        b_projection,
        c_projection,
        mimo_x,
        mimo_o,
        dt,
        lam,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    if not capability.available:
        raise RuntimeError(f"Mamba-3 Flash-PD Triton unavailable: {capability.reason}")
    return _Mamba3FlashPDTritonReadout.apply(
        dictionary_logits,
        selector_logits,
        diagonal,
        value,
        b_projection,
        c_projection,
        mimo_x,
        mimo_o,
        dt,
        lam,
        temperature,
        chunk_size,
        checkpoint_stride,
    )
