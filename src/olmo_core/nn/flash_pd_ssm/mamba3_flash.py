"""
Correctness-first Mamba-3 discretization with collision-preserving Flash-PD transitions.

The fused readout auto-dispatches eligible float32 CUDA inputs to the bounded-scratch MIMO
Triton kernel and retains the compact checkpointed PyTorch recurrence as its fallback. This is
not a throughput claim. Importing the module directly registers the ``mamba3_flash_pd``
sequence-mixer config without changing the existing Mamba-3 or Flash-PD modules.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

from .mamba3_flash_triton import (
    MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE,
    mamba3_flash_pd_triton_capability,
    mamba3_flash_pd_triton_readout,
)

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

__all__ = [
    "Mamba3FlashPDSSMMixer",
    "Mamba3FlashPDSSMMixerConfig",
    "mamba3_flash_pd_readout",
    "mamba3_flash_pd_scan",
]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    original_dtype = x.dtype
    x_float = x.float()
    normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * weight.float()).to(original_dtype)


def _validate_options(
    *,
    d_model: int,
    n_heads: int,
    head_dim: Optional[int],
    d_state: int,
    n_groups: int,
    mimo_rank: int,
    dictionary_size: int,
    ste_temperature: float,
    norm_eps: float,
    a_log_init_min: float,
    a_log_init_max: float,
) -> int:
    if d_model < 1:
        raise OLMoConfigurationError(f"d_model must be positive, got {d_model}")
    if n_heads < 1:
        raise OLMoConfigurationError(f"n_heads must be positive, got {n_heads}")
    if head_dim is None:
        if d_model % n_heads != 0:
            raise OLMoConfigurationError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) when "
                "head_dim is omitted"
            )
        resolved_head_dim = d_model // n_heads
    else:
        resolved_head_dim = head_dim
    if resolved_head_dim < 1:
        raise OLMoConfigurationError(f"head_dim must be positive, got {resolved_head_dim}")
    if d_state < 1:
        raise OLMoConfigurationError(f"d_state must be positive, got {d_state}")
    if n_groups < 1:
        raise OLMoConfigurationError(f"n_groups must be positive, got {n_groups}")
    if n_heads % n_groups != 0:
        raise OLMoConfigurationError(
            f"n_heads ({n_heads}) must be divisible by n_groups ({n_groups})"
        )
    if mimo_rank < 1:
        raise OLMoConfigurationError(f"mimo_rank must be positive, got {mimo_rank}")
    if dictionary_size < 1:
        raise OLMoConfigurationError(f"dictionary_size must be positive, got {dictionary_size}")
    if ste_temperature <= 0:
        raise OLMoConfigurationError(f"ste_temperature must be positive, got {ste_temperature}")
    if norm_eps <= 0:
        raise OLMoConfigurationError(f"norm_eps must be positive, got {norm_eps}")
    if a_log_init_min <= 0:
        raise OLMoConfigurationError(f"a_log_init_min must be positive, got {a_log_init_min}")
    if a_log_init_max <= a_log_init_min:
        raise OLMoConfigurationError(
            "a_log_init_max must be greater than a_log_init_min, got "
            f"{a_log_init_max} <= {a_log_init_min}"
        )
    return resolved_head_dim


def _validate_scan_inputs(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    previous_input: torch.Tensor,
    current_input: torch.Tensor,
    *,
    temperature: float,
) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if dictionary_logits.ndim != 4:
        raise ValueError("dictionary_logits must have shape (heads, dictionary_size, state, state)")
    heads, dictionary_size, state_rows, state_columns = dictionary_logits.shape
    if state_rows != state_columns:
        raise ValueError("dictionary transition logits must be square")
    if selector_logits.ndim != 4:
        raise ValueError("selector_logits must have shape (batch, time, heads, dictionary_size)")
    batch, time, selector_heads, selector_dictionary_size = selector_logits.shape
    if (selector_heads, selector_dictionary_size) != (heads, dictionary_size):
        raise ValueError(
            "selector heads/dictionary dimensions do not match dictionary_logits: "
            f"{(selector_heads, selector_dictionary_size)} != {(heads, dictionary_size)}"
        )
    if diagonal.shape != (batch, heads, time, state_columns):
        raise ValueError(
            "diagonal must have shape (batch, heads, time, state), got " f"{tuple(diagonal.shape)}"
        )
    if previous_input.ndim != 5:
        raise ValueError("previous_input must have shape (batch, heads, time, state, payload)")
    if previous_input.shape[:3] != (batch, heads, time):
        raise ValueError("previous_input batch/head/time dimensions do not match selectors")
    if previous_input.shape[3] != state_columns:
        raise ValueError("previous_input state dimension does not match dictionary_logits")
    if current_input.shape != previous_input.shape:
        raise ValueError(
            "current_input and previous_input must have identical shapes, got "
            f"{tuple(current_input.shape)} and {tuple(previous_input.shape)}"
        )
    if time < 1:
        raise ValueError("the sparse recurrence requires at least one token")
    if not diagonal.is_complex():
        raise TypeError(f"diagonal must be complex, got {diagonal.dtype}")
    if not previous_input.is_complex() or not current_input.is_complex():
        raise TypeError("previous_input and current_input must be complex")
    if previous_input.dtype != diagonal.dtype or current_input.dtype != diagonal.dtype:
        raise TypeError("diagonal and payload tensors must have the same complex dtype")


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


class _Mamba3FlashPDSparseScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        dictionary_logits: torch.Tensor,
        selector_logits: torch.Tensor,
        diagonal: torch.Tensor,
        previous_input: torch.Tensor,
        current_input: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        destination, route = _selected_destinations(dictionary_logits, selector_logits)
        batch, heads, time, state_size, payload_size = previous_input.shape
        state = torch.zeros_like(current_input[:, :, 0])
        states = []
        for token_idx in range(time):
            source = state + previous_input[:, :, token_idx]
            scaled_source = diagonal[:, :, token_idx, :, None] * source
            token_destination = destination[:, :, token_idx]
            index = token_destination[..., None].expand(batch, heads, state_size, payload_size)
            state = torch.zeros_like(state).scatter_add(
                dim=2,
                index=index,
                src=scaled_source,
            )
            state = state + current_input[:, :, token_idx]
            states.append(state)
        output = torch.stack(states, dim=2)
        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            diagonal,
            previous_input,
            output,
            destination,
            route,
        )
        ctx.temperature = temperature
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None,]:
        (
            dictionary_logits,
            selector_logits,
            diagonal,
            previous_input,
            states,
            destination,
            route,
        ) = ctx.saved_tensors
        temperature = ctx.temperature
        batch, heads, time, state_size, payload_size = previous_input.shape
        dictionary_size = dictionary_logits.shape[1]

        grad_diagonal = torch.zeros_like(diagonal)
        grad_previous_input = torch.zeros_like(previous_input)
        grad_current_input = torch.zeros_like(previous_input)
        grad_selector = torch.zeros_like(selector_logits)
        dictionary_transition_grad = torch.zeros_like(dictionary_logits)
        carry = torch.zeros_like(previous_input[:, :, 0])
        zero_state = torch.zeros_like(carry)

        dictionary_destination = dictionary_logits.argmax(dim=-2)
        hard_dictionary = F.one_hot(
            dictionary_destination,
            num_classes=state_size,
        ).movedim(-1, -2)
        hard_dictionary = hard_dictionary.to(dictionary_logits.dtype)
        selector_probability = torch.softmax(selector_logits / temperature, dim=-1)

        for token_idx in range(time - 1, -1, -1):
            total_grad = grad_states[:, :, token_idx] + carry
            grad_current_input[:, :, token_idx] = total_grad
            previous_state = zero_state if token_idx == 0 else states[:, :, token_idx - 1]
            source = previous_state + previous_input[:, :, token_idx]
            token_destination = destination[:, :, token_idx]
            index = token_destination[..., None].expand(batch, heads, state_size, payload_size)
            grad_at_destination = torch.gather(total_grad, dim=2, index=index)
            token_diagonal = diagonal[:, :, token_idx]

            grad_diagonal[:, :, token_idx] = (grad_at_destination * source.conj()).sum(dim=-1)
            grad_source = grad_at_destination * token_diagonal[..., None].conj()
            grad_previous_input[:, :, token_idx] = grad_source
            carry = grad_source

            transitioned_source = token_diagonal[..., None] * source
            transition_grad = torch.einsum(
                "bhip,bhqp->bhiq",
                total_grad.conj(),
                transitioned_source,
            ).real.to(dictionary_logits.dtype)

            route_score = torch.einsum(
                "bhiq,hkiq->bhk",
                transition_grad,
                hard_dictionary,
            )
            probability = selector_probability[:, token_idx]
            grad_selector[:, token_idx] = (
                probability
                * (route_score - (probability * route_score).sum(dim=-1, keepdim=True))
                / temperature
            )

            hard_route = F.one_hot(
                route[:, token_idx],
                num_classes=dictionary_size,
            ).to(dictionary_logits.dtype)
            dictionary_transition_grad.add_(
                torch.einsum(
                    "bhk,bhiq->hkiq",
                    hard_route,
                    transition_grad,
                )
            )

        dictionary_probability = torch.softmax(
            dictionary_logits / temperature,
            dim=-2,
        )
        grad_dictionary = (
            dictionary_probability
            * (
                dictionary_transition_grad
                - (dictionary_probability * dictionary_transition_grad).sum(dim=-2, keepdim=True)
            )
            / temperature
        )
        return (
            grad_dictionary,
            grad_selector,
            grad_diagonal,
            grad_previous_input,
            grad_current_input,
            None,
        )


def mamba3_flash_pd_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    previous_input: torch.Tensor,
    current_input: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """
    Run a collision-preserving Flash-PD recurrence over one shared MIMO state.

    ``destination[..., q]`` is a source-to-destination map, so colliding source states are
    summed. The recurrence is ``h_t = A_t (h_(t-1) + previous_input_t) +
    current_input_t``. For Mamba-3, callers set ``previous_input_t`` to
    ``(1-lambda_t) dt_t v_(t-1)`` and ``current_input_t`` to ``lambda_t dt_t v_t``.

    The custom backward reproduces both hard selectors' tempered-softmax STE gradients while
    saving compact destinations and ``(batch, heads, time, state, payload)`` states. It
    never saves a per-token square transition. The Python loop is a correctness path, not a
    production-throughput kernel.

    :param dictionary_logits: Transition dictionary logits, shape ``(H, K, N, N)``.
    :param selector_logits: Per-token router logits, shape ``(B, T, H, K)``.
    :param diagonal: Complex source scales, shape ``(B, H, T, N)``.
    :param previous_input: Payload transformed by ``A_t``, shape ``(B, H, T, N, P)``.
    :param current_input: Payload added after ``A_t``, shape ``(B, H, T, N, P)``.
    :param temperature: Positive STE softmax temperature.

    :returns: Complex states with shape ``(B, H, T, N, P)``.
    """
    _validate_scan_inputs(
        dictionary_logits,
        selector_logits,
        diagonal,
        previous_input,
        current_input,
        temperature=temperature,
    )
    return _Mamba3FlashPDSparseScan.apply(
        dictionary_logits,
        selector_logits,
        diagonal,
        previous_input,
        current_input,
        temperature,
    )


def _validate_readout_inputs(
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
    checkpoint_stride: int,
) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if checkpoint_stride < 1:
        raise ValueError(f"checkpoint_stride must be positive, got {checkpoint_stride}")
    if dictionary_logits.ndim != 4:
        raise ValueError("dictionary_logits must have shape (heads, dictionary_size, state, state)")
    heads, dictionary_size, state_rows, state_columns = dictionary_logits.shape
    if state_rows != state_columns:
        raise ValueError("dictionary transition logits must be square")
    if selector_logits.ndim != 4:
        raise ValueError("selector_logits must have shape (batch, time, heads, dictionary_size)")
    batch, time, selector_heads, selector_dictionary_size = selector_logits.shape
    if (selector_heads, selector_dictionary_size) != (heads, dictionary_size):
        raise ValueError("selector dimensions do not match dictionary_logits")
    if diagonal.shape != (batch, heads, time, state_columns):
        raise ValueError("diagonal must have shape (batch, heads, time, state)")
    if value.ndim != 4 or value.shape[:3] != (batch, heads, time):
        raise ValueError("value must have shape (batch, heads, time, payload)")
    if b_projection.ndim != 5 or b_projection.shape[:3] != (batch, heads, time):
        raise ValueError("b_projection must have shape (batch, heads, time, rank, state)")
    if b_projection.shape[-1] != state_columns:
        raise ValueError("b_projection state dimension does not match dictionary_logits")
    if c_projection.shape != b_projection.shape:
        raise ValueError("b_projection and c_projection must have identical shapes")
    rank = b_projection.shape[-2]
    payload = value.shape[-1]
    if mimo_x.shape != (heads, rank, payload):
        raise ValueError(
            "mimo_x must have shape (heads, rank, payload), got " f"{tuple(mimo_x.shape)}"
        )
    if mimo_o.shape != mimo_x.shape:
        raise ValueError("mimo_x and mimo_o must have identical shapes")
    if dt.shape != (batch, heads, time) or lam.shape != dt.shape:
        raise ValueError("dt and lam must have shape (batch, heads, time)")
    if time < 1:
        raise ValueError("the sparse recurrence requires at least one token")
    if not diagonal.is_complex():
        raise TypeError(f"diagonal must be complex, got {diagonal.dtype}")
    real_dtype = diagonal.real.dtype
    for name, tensor in (
        ("dictionary_logits", dictionary_logits),
        ("selector_logits", selector_logits),
        ("value", value),
        ("b_projection", b_projection),
        ("c_projection", c_projection),
        ("mimo_x", mimo_x),
        ("mimo_o", mimo_o),
        ("dt", dt),
        ("lam", lam),
    ):
        if tensor.dtype != real_dtype:
            raise TypeError(f"{name} must have dtype {real_dtype}, got {tensor.dtype}")


def _compact_drive(
    value: torch.Tensor,
    b_projection: torch.Tensor,
    mimo_x: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bhrn,hrp,bhp->bhnp", b_projection, mimo_x, value)


def _compact_step(
    state: torch.Tensor,
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    previous_value: Optional[torch.Tensor],
    previous_b_projection: Optional[torch.Tensor],
    dt: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    drive = _compact_drive(value, b_projection, mimo_x)
    if previous_value is None or previous_b_projection is None:
        previous_drive = torch.zeros_like(drive)
    else:
        previous_drive = _compact_drive(previous_value, previous_b_projection, mimo_x)
    previous_scale = (1.0 - lam) * dt
    current_scale = lam * dt
    source = state + previous_scale[..., None, None] * previous_drive
    scaled_source = diagonal[..., None] * source
    batch, heads, state_size, payload = scaled_source.shape
    index = destination[..., None].expand(batch, heads, state_size, payload)
    transitioned = torch.zeros_like(state).scatter_add(
        dim=2,
        index=index,
        src=scaled_source,
    )
    return transitioned + current_scale[..., None, None] * drive


class _Mamba3FlashPDCheckpointedReadout(torch.autograd.Function):
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
    ) -> torch.Tensor:
        destination, _ = _selected_destinations(dictionary_logits, selector_logits)
        batch, heads, time, payload = value.shape
        state_size = b_projection.shape[-1]
        state = diagonal.new_zeros((batch, heads, state_size, payload))
        readout = value.new_empty((batch, time, heads, payload))
        boundaries = []

        for token_idx in range(time):
            previous_idx = token_idx - 1
            state = _compact_step(
                state,
                destination[:, :, token_idx],
                diagonal[:, :, token_idx],
                value[:, :, token_idx],
                b_projection[:, :, token_idx],
                mimo_x,
                None if token_idx == 0 else value[:, :, previous_idx],
                None if token_idx == 0 else b_projection[:, :, previous_idx],
                dt[:, :, token_idx],
                lam[:, :, token_idx],
            )
            readout[:, token_idx] = torch.einsum(
                "bhrn,bhnp,hrp->bhp",
                c_projection[:, :, token_idx],
                state.real,
                mimo_o,
            )
            if (token_idx + 1) % chunk_size == 0 and token_idx + 1 < time:
                boundaries.append(state)

        if boundaries:
            boundary_states = torch.stack(boundaries, dim=2)
        else:
            boundary_states = state.new_empty((batch, heads, 0, state_size, payload))
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
            boundary_states,
        )
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
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
            boundary_states,
        ) = ctx.saved_tensors
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        batch, heads, time, payload = value.shape
        state_size = b_projection.shape[-1]
        dictionary_size = dictionary_logits.shape[1]

        grad_dictionary_transition = torch.zeros_like(dictionary_logits)
        grad_selector = torch.zeros_like(selector_logits)
        grad_diagonal = torch.zeros_like(diagonal)
        grad_value = torch.zeros_like(value)
        grad_b_projection = torch.zeros_like(b_projection)
        grad_c_projection = torch.zeros_like(c_projection)
        grad_mimo_x = torch.zeros_like(mimo_x)
        grad_mimo_o = torch.zeros_like(mimo_o)
        grad_dt = torch.zeros_like(dt)
        grad_lam = torch.zeros_like(lam)

        destination, route = _selected_destinations(dictionary_logits, selector_logits)
        dictionary_destination = dictionary_logits.argmax(dim=-2)
        selector_probability = torch.softmax(selector_logits / temperature, dim=-1)
        num_chunks = math.ceil(time / chunk_size)
        zero_state = diagonal.new_zeros((batch, heads, state_size, payload))
        carry = torch.zeros_like(zero_state)

        for chunk_idx in range(num_chunks - 1, -1, -1):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, time)
            start_state = zero_state if chunk_idx == 0 else boundary_states[:, :, chunk_idx - 1]
            state = start_state
            states = []
            for token_idx in range(start, end):
                previous_idx = token_idx - 1
                state = _compact_step(
                    state,
                    destination[:, :, token_idx],
                    diagonal[:, :, token_idx],
                    value[:, :, token_idx],
                    b_projection[:, :, token_idx],
                    mimo_x,
                    None if token_idx == 0 else value[:, :, previous_idx],
                    None if token_idx == 0 else b_projection[:, :, previous_idx],
                    dt[:, :, token_idx],
                    lam[:, :, token_idx],
                )
                states.append(state)

            for local_idx in range(end - start - 1, -1, -1):
                token_idx = start + local_idx
                token_state = states[local_idx]
                token_readout_grad = grad_readout[:, token_idx]
                token_c = c_projection[:, :, token_idx]
                grad_c_projection[:, :, token_idx] = torch.einsum(
                    "bhp,hrp,bhnp->bhrn",
                    token_readout_grad,
                    mimo_o,
                    token_state.real,
                )
                grad_mimo_o.add_(
                    torch.einsum(
                        "bhp,bhrn,bhnp->hrp",
                        token_readout_grad,
                        token_c,
                        token_state.real,
                    )
                )
                total_grad = carry + torch.einsum(
                    "bhrn,hrp,bhp->bhnp",
                    token_c,
                    mimo_o,
                    token_readout_grad,
                )

                token_value = value[:, :, token_idx]
                token_b = b_projection[:, :, token_idx]
                drive = _compact_drive(token_value, token_b, mimo_x)
                current_scale = lam[:, :, token_idx] * dt[:, :, token_idx]
                grad_current_scale = (total_grad.real * drive).sum(dim=(2, 3))
                grad_drive = current_scale[..., None, None] * total_grad.real
                grad_b_projection[:, :, token_idx].add_(
                    torch.einsum("bhnp,hrp,bhp->bhrn", grad_drive, mimo_x, token_value)
                )
                grad_mimo_x.add_(
                    torch.einsum("bhnp,bhrn,bhp->hrp", grad_drive, token_b, token_value)
                )
                grad_value[:, :, token_idx].add_(
                    torch.einsum("bhnp,bhrn,hrp->bhp", grad_drive, token_b, mimo_x)
                )

                previous_state = start_state if local_idx == 0 else states[local_idx - 1]
                if token_idx == 0:
                    previous_drive = torch.zeros_like(drive)
                else:
                    previous_value = value[:, :, token_idx - 1]
                    previous_b = b_projection[:, :, token_idx - 1]
                    previous_drive = _compact_drive(previous_value, previous_b, mimo_x)
                previous_scale = (1.0 - lam[:, :, token_idx]) * dt[:, :, token_idx]
                source = previous_state + previous_scale[..., None, None] * previous_drive
                token_diagonal = diagonal[:, :, token_idx]
                transitioned_source = token_diagonal[..., None] * source
                token_destination = destination[:, :, token_idx]
                index = token_destination[..., None].expand(batch, heads, state_size, payload)
                grad_at_destination = torch.gather(total_grad, dim=2, index=index)
                grad_diagonal[:, :, token_idx] = (grad_at_destination * source.conj()).sum(dim=-1)
                grad_source = grad_at_destination * token_diagonal[..., None].conj()
                carry = grad_source
                grad_previous_scale = (grad_source.real * previous_drive).sum(dim=(2, 3))
                if token_idx > 0:
                    grad_previous_drive = previous_scale[..., None, None] * grad_source.real
                    grad_b_projection[:, :, token_idx - 1].add_(
                        torch.einsum(
                            "bhnp,hrp,bhp->bhrn",
                            grad_previous_drive,
                            mimo_x,
                            previous_value,
                        )
                    )
                    grad_mimo_x.add_(
                        torch.einsum(
                            "bhnp,bhrn,bhp->hrp",
                            grad_previous_drive,
                            previous_b,
                            previous_value,
                        )
                    )
                    grad_value[:, :, token_idx - 1].add_(
                        torch.einsum(
                            "bhnp,bhrn,hrp->bhp",
                            grad_previous_drive,
                            previous_b,
                            mimo_x,
                        )
                    )
                grad_dt[:, :, token_idx] = (
                    grad_previous_scale * (1.0 - lam[:, :, token_idx])
                    + grad_current_scale * lam[:, :, token_idx]
                )
                grad_lam[:, :, token_idx] = (grad_current_scale - grad_previous_scale) * dt[
                    :, :, token_idx
                ]

                route_scores = []
                for dictionary_idx in range(dictionary_size):
                    candidate_destination = dictionary_destination[:, dictionary_idx]
                    candidate_index = candidate_destination.view(1, heads, state_size, 1).expand(
                        batch, heads, state_size, payload
                    )
                    candidate_grad = torch.gather(total_grad, dim=2, index=candidate_index)
                    route_scores.append(
                        (candidate_grad.conj() * transitioned_source).real.sum(dim=(2, 3))
                    )
                route_score = torch.stack(route_scores, dim=-1)
                probability = selector_probability[:, token_idx]
                grad_selector[:, token_idx] = (
                    probability
                    * (route_score - (probability * route_score).sum(dim=-1, keepdim=True))
                    / temperature
                )

                hard_route = F.one_hot(
                    route[:, token_idx],
                    num_classes=dictionary_size,
                ).to(dictionary_logits.dtype)
                for dictionary_idx in range(dictionary_size):
                    grad_dictionary_transition[:, dictionary_idx].add_(
                        torch.einsum(
                            "bh,bhip,bhqp->hiq",
                            hard_route[:, :, dictionary_idx],
                            total_grad.conj(),
                            transitioned_source,
                        ).real.to(dictionary_logits.dtype)
                    )

        dictionary_probability = torch.softmax(
            dictionary_logits / temperature,
            dim=-2,
        )
        grad_dictionary = (
            dictionary_probability
            * (
                grad_dictionary_transition
                - (dictionary_probability * grad_dictionary_transition).sum(
                    dim=-2,
                    keepdim=True,
                )
            )
            / temperature
        )
        return (
            grad_dictionary,
            grad_selector,
            grad_diagonal,
            grad_value,
            grad_b_projection,
            grad_c_projection,
            grad_mimo_x,
            grad_mimo_o,
            grad_dt,
            grad_lam,
            None,
            None,
        )


def mamba3_flash_pd_readout(
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
    chunk_size: int = 128,
    checkpoint_stride: int = MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE,
    use_triton: Optional[bool] = None,
) -> torch.Tensor:
    """
    Run the fused recurrence and C readout from compact projected inputs.

    The Triton path saves compact inputs and hierarchical subchunk states for backward. Token
    states and rank-contracted drives are rebuilt in tiles bounded by ``checkpoint_stride``,
    using ``O(B * H * ceil(T / checkpoint_stride) * N * P)`` saved state rather than a full
    token history. The PyTorch fallback retains chunk-boundary checkpointing.

    :param dictionary_logits: Transition dictionary logits, shape ``(H, K, N, N)``.
    :param selector_logits: Per-token router logits, shape ``(B, T, H, K)``.
    :param diagonal: Complex source scales, shape ``(B, H, T, N)``.
    :param value: Per-head payload values, shape ``(B, H, T, P)``.
    :param b_projection: MIMO input factors, shape ``(B, H, T, R, N)``.
    :param c_projection: MIMO readout factors, shape ``(B, H, T, R, N)``.
    :param mimo_x: Learned rank-to-value write factors, shape ``(H, R, P)``.
    :param mimo_o: Learned rank-to-output read factors, shape ``(H, R, P)``.
    :param dt: Positive recurrence timesteps, shape ``(B, H, T)``.
    :param lam: Trapezoidal interpolation weights, shape ``(B, H, T)``.
    :param temperature: Positive STE softmax temperature.
    :param chunk_size: Number of token states recomputed together during backward.
    :param checkpoint_stride: Maximum Triton token-state replay tile, bounded at 16 on the
        production path. Values above 16 auto-fallback and fail in strict Triton mode.
    :param use_triton: ``None`` auto-dispatches eligible float32 CUDA inputs, ``True`` requires
        Triton without fallback, and ``False`` selects the compact checkpointed PyTorch path.

    :returns: Real readouts with shape ``(B, T, H, P)``.
    """
    _validate_readout_inputs(
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
        temperature=temperature,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    if use_triton is not False:
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
        if capability.available:
            return mamba3_flash_pd_triton_readout(
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
                temperature=temperature,
                chunk_size=chunk_size,
                checkpoint_stride=checkpoint_stride,
            )
        if use_triton:
            raise RuntimeError(
                "Mamba-3 Flash-PD Triton path required but unavailable: " f"{capability.reason}"
            )
    return _Mamba3FlashPDCheckpointedReadout.apply(
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
    )


class Mamba3FlashPDSSMMixer(SequenceMixer):
    """
    A single SSM layer fusing Mamba-3 discretization with Flash-PD transitions.

    Each source state ``q`` has one destination ``p_t[q]`` and complex scale
    ``d_t[q] = exp(dt_t * a_h) * exp(i * dt_t * theta_t[q])``, where ``a_h < 0``.
    The source-to-destination orientation preserves collisions by scatter-add and stays closed
    under transition composition; no dense ``SO(3)`` factor is applied.

    The MIMO state has shape ``(B, H, N, P)``, independent of rank. With
    ``v_t[n,p] = sum_r B_t[r,n] mimo_x[r,p] x_t[p]``, the layer computes
    ``h_t = A_t h_(t-1) + (1-lambda_t) dt_t A_t v_(t-1) +
    lambda_t dt_t v_t``. Readout first applies ``C_t[r,n]`` to the shared state and then
    contracts rank with ``mimo_o[r,p]``. Its real component feeds the existing SiLU gate,
    RMS normalization, and output projection. There is no short causal convolution.

    Eligible float32 CUDA inputs use the three-phase MIMO Triton forward and hierarchically
    checkpointed GPU backward. Unsupported dtypes, devices, shapes, and architectures use the
    compact chunk-checkpointed PyTorch fallback. Neither path saves full token-state history.

    :param d_model: Input and output width.
    :param n_heads: Number of independent SSM heads.
    :param head_dim: Per-head payload width, defaulting to ``d_model // n_heads``.
    :param d_state: Number of complex source states per head.
    :param n_groups: Number of MIMO ``B/C`` groups shared across heads.
    :param mimo_rank: MIMO rank ``R``.
    :param dictionary_size: Number of source-to-destination maps per head.
    :param ste_temperature: Backward softmax temperature for dictionary and router STEs.
    :param norm_eps: Epsilon for B/C and output RMS normalization.
    :param bc_norm: Apply RMS normalization to B and C projections.
    :param bc_bias: Include bias in the fused B/C projection.
    :param exempt_timescale_params_from_weight_decay: Tag A-log and dt-bias parameters.
    :param a_log_init_min: Minimum initial positive decay rate magnitude.
    :param a_log_init_max: Maximum initial positive decay rate magnitude.
    :param scan_chunk_size: Top-level recurrence chunk size.
    :param scan_checkpoint_stride: Maximum hierarchical Triton replay tile. The safe production
        default bounds replay-loop work at 16 bodies per token.
    :param dtype: Projection and dictionary parameter dtype.
    :param init_device: Initialization device, including ``"meta"``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 8,
        head_dim: Optional[int] = None,
        d_state: int = 64,
        n_groups: int = 1,
        mimo_rank: int = 4,
        dictionary_size: int = 16,
        ste_temperature: float = 1.0,
        norm_eps: float = 1e-5,
        bc_norm: bool = True,
        bc_bias: bool = True,
        exempt_timescale_params_from_weight_decay: bool = True,
        a_log_init_min: float = 0.05,
        a_log_init_max: float = 16.0,
        scan_chunk_size: int = 128,
        scan_checkpoint_stride: int = MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        resolved_head_dim = _validate_options(
            d_model=d_model,
            n_heads=n_heads,
            head_dim=head_dim,
            d_state=d_state,
            n_groups=n_groups,
            mimo_rank=mimo_rank,
            dictionary_size=dictionary_size,
            ste_temperature=ste_temperature,
            norm_eps=norm_eps,
            a_log_init_min=a_log_init_min,
            a_log_init_max=a_log_init_max,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = resolved_head_dim
        self.d_state = d_state
        self.n_groups = n_groups
        self.heads_per_group = n_heads // n_groups
        self.mimo_rank = mimo_rank
        self.dictionary_size = dictionary_size
        self.ste_temperature = ste_temperature
        self.norm_eps = norm_eps
        self.bc_norm_enabled = bc_norm
        self.bc_bias = bc_bias
        self.exempt_timescale_params_from_weight_decay = exempt_timescale_params_from_weight_decay
        self.a_log_init_min = a_log_init_min
        self.a_log_init_max = a_log_init_max
        if scan_chunk_size < 1:
            raise OLMoConfigurationError(f"scan_chunk_size must be positive, got {scan_chunk_size}")
        if scan_checkpoint_stride < 1:
            raise OLMoConfigurationError(
                f"scan_checkpoint_stride must be positive, got {scan_checkpoint_stride}"
            )
        self.scan_chunk_size = scan_chunk_size
        self.scan_checkpoint_stride = scan_checkpoint_stride
        self.last_backend: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None

        inner = n_heads * resolved_head_dim
        bc_width = n_groups * mimo_rank * d_state
        dynamics_width = n_heads * (dictionary_size + d_state + 2)
        self.dictionary_logits = nn.Parameter(
            torch.empty(
                n_heads,
                dictionary_size,
                d_state,
                d_state,
                dtype=dtype,
                device=init_device,
            )
        )
        self.xz_proj = nn.Linear(
            d_model,
            2 * inner,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.bc_proj = nn.Linear(
            d_model,
            2 * bc_width,
            bias=bc_bias,
            dtype=dtype,
            device=init_device,
        )
        self.dynamics_proj = nn.Linear(
            d_model,
            dynamics_width,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.out_proj = nn.Linear(
            inner,
            d_model,
            bias=False,
            dtype=dtype,
            device=init_device,
        )

        self.A_log = nn.Parameter(torch.empty(n_heads, dtype=torch.float32, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(n_heads, dtype=torch.float32, device=init_device))
        if exempt_timescale_params_from_weight_decay:
            self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
            self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.o_norm_weight = nn.Parameter(
            torch.ones(resolved_head_dim, dtype=dtype, device=init_device)
        )
        mimo_init = 1.0 / mimo_rank
        self.mimo_x = nn.Parameter(
            torch.full(
                (n_heads, mimo_rank, resolved_head_dim),
                mimo_init,
                dtype=dtype,
                device=init_device,
            )
        )
        self.mimo_o = nn.Parameter(
            torch.full(
                (n_heads, mimo_rank, resolved_head_dim),
                mimo_init,
                dtype=dtype,
                device=init_device,
            )
        )
        if bc_norm:
            self.bc_norm_b = nn.Parameter(torch.ones(d_state, dtype=dtype, device=init_device))
            self.bc_norm_c = nn.Parameter(torch.ones(d_state, dtype=dtype, device=init_device))
        else:
            self.register_parameter("bc_norm_b", None)
            self.register_parameter("bc_norm_c", None)

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
        if prefix + "xz_proj.weight" in state_dict and (
            prefix + "mimo_x" not in state_dict or prefix + "mimo_o" not in state_dict
        ):
            error_msgs.append(
                f"{prefix or '<root>'}: rank-expanded Mamba-3 Flash-PD checkpoints are "
                "incompatible with the shared-state MIMO recurrence; expected learned "
                "'mimo_x' and 'mimo_o' parameters"
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

    def set_ste_temperature(self, temperature: float) -> None:
        """
        Set the backward softmax temperature used by both hard selectors.

        :param temperature: Positive STE temperature.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.ste_temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply the fused Mamba-3 plus Flash-PD SSM layer.

        Packed documents, explicit reset masks, recurrent state, and decode caching are rejected
        because carrying state across those boundaries would silently mix unrelated sequences.

        :param x: Input tensor with shape ``(batch, time, d_model)``.
        :param cu_doc_lens: Packed-document boundaries, currently unsupported.
        :param kwargs: Transformer compatibility arguments; recognized stateful modes fail closed.

        :returns: A tensor with the same shape and dtype as ``x``.

        :raises NotImplementedError: For packed/reset/decode/recurrent-state operation.
        """
        if cu_doc_lens is not None:
            raise NotImplementedError(
                "Mamba3FlashPDSSMMixer does not support packed documents or cu_doc_lens"
            )
        if kwargs.get("initial_state") is not None:
            raise NotImplementedError(
                "Mamba3FlashPDSSMMixer does not support initial_state or recurrent caching"
            )
        if kwargs.get("decode", False):
            raise NotImplementedError(
                "Mamba3FlashPDSSMMixer does not support decode mode or recurrent caching"
            )
        reset_arguments = (
            "reset_mask",
            "state_reset",
            "sequence_id",
            "document_id",
            "packed_document_mask",
        )
        for argument in reset_arguments:
            if kwargs.get(argument) is not None:
                raise NotImplementedError(
                    f"Mamba3FlashPDSSMMixer does not support state reset argument '{argument}'"
                )
        del kwargs
        self.last_backend = None
        self.last_fallback_reason = None

        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, time, {self.d_model}), got {tuple(x.shape)}"
            )
        batch, time, _ = x.shape
        H, P, G, R, N, K = (
            self.n_heads,
            self.head_dim,
            self.n_groups,
            self.mimo_rank,
            self.d_state,
            self.dictionary_size,
        )

        xz = self.xz_proj(x).view(batch, time, 2, H, P)
        value, gate = xz[:, :, 0], xz[:, :, 1]
        bc = self.bc_proj(x).view(batch, time, 2, G, R, N)
        b_projection, c_projection = bc[:, :, 0], bc[:, :, 1]
        if self.bc_norm_enabled:
            assert self.bc_norm_b is not None and self.bc_norm_c is not None
            b_projection = _rms_norm(
                b_projection,
                self.bc_norm_b,
                self.norm_eps,
            )
            c_projection = _rms_norm(
                c_projection,
                self.bc_norm_c,
                self.norm_eps,
            )

        dynamics = self.dynamics_proj(x).float()
        selector_width = H * K
        selector_flat, dt_logits, lambda_logits, theta_flat = dynamics.split(
            (selector_width, H, H, H * N),
            dim=-1,
        )
        selector_logits = selector_flat.view(batch, time, H, K)
        dt = F.softplus(dt_logits + self.dt_bias.view(1, 1, H))
        lam = torch.sigmoid(lambda_logits)
        theta = theta_flat.view(batch, time, H, N)

        decay = -torch.exp(self.A_log.float())
        magnitude = torch.exp(dt * decay.view(1, 1, H))
        phase = dt[..., None] * theta
        diagonal = torch.polar(magnitude[..., None].expand_as(phase), phase)

        b_projection = b_projection.repeat_interleave(
            self.heads_per_group,
            dim=2,
        ).float()
        c_projection = c_projection.repeat_interleave(
            self.heads_per_group,
            dim=2,
        ).float()
        readout_inputs = (
            self.dictionary_logits.float(),
            selector_logits,
            diagonal.permute(0, 2, 1, 3),
            value.float().permute(0, 2, 1, 3),
            b_projection.permute(0, 2, 1, 3, 4),
            c_projection.permute(0, 2, 1, 3, 4),
            self.mimo_x.float(),
            self.mimo_o.float(),
            dt.permute(0, 2, 1),
            lam.permute(0, 2, 1),
        )
        triton_capability = mamba3_flash_pd_triton_capability(
            *readout_inputs,
            chunk_size=self.scan_chunk_size,
            checkpoint_stride=self.scan_checkpoint_stride,
        )
        readout = mamba3_flash_pd_readout(
            *readout_inputs,
            temperature=self.ste_temperature,
            chunk_size=self.scan_chunk_size,
            checkpoint_stride=self.scan_checkpoint_stride,
            use_triton=triton_capability.available,
        )
        if triton_capability.available:
            effective_stride = min(self.scan_chunk_size, self.scan_checkpoint_stride)
            self.last_backend = f"triton_mimo_shared_hierarchical_s{effective_stride}"
        else:
            self.last_backend = "pytorch_checkpointed"
            self.last_fallback_reason = triton_capability.reason
        normalized = _rms_norm(
            readout * F.silu(gate.float()),
            self.o_norm_weight,
            self.norm_eps,
        )
        return self.out_proj(normalized.reshape(batch, time, H * P).to(x.dtype))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        """Reject tensor parallelism, which is not implemented for this mixer."""
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for Mamba3FlashPDSSMMixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """Reject context parallelism, which requires a distributed recurrent scan."""
        del cp_mesh, ring, uly
        raise NotImplementedError(
            "Context parallelism is not implemented for Mamba3FlashPDSSMMixer"
        )

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
        """
        Initialize projections, selectors, decay rates, timesteps, and norm weights.

        :param init_method: OLMo transformer initialization policy.
        :param d_model: Model width.
        :param block_idx: Zero-based block index.
        :param num_blocks: Number of transformer blocks.
        :param std: Base projection standard deviation.
        :param generator: Optional deterministic random generator.
        """
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for Mamba3FlashPDSSMMixer"
            )
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        nn.init.normal_(self.dictionary_logits, std=std, generator=generator)
        for projection in (self.xz_proj, self.bc_proj, self.dynamics_proj):
            init_linear(projection, std=std, generator=generator)
        phase_start = self.n_heads * (self.dictionary_size + 2)
        self.dynamics_proj.weight[phase_start:].mul_(0.1)

        self.A_log.copy_(
            nn.init.uniform_(
                self.A_log,
                a=self.a_log_init_min,
                b=self.a_log_init_max,
                generator=generator,
            ).log()
        )
        dt_min, dt_max, dt_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_floor)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        self.o_norm_weight.fill_(1.0)
        self.mimo_x.fill_(1.0 / self.mimo_rank)
        self.mimo_o.fill_(1.0 / self.mimo_rank)
        if self.bc_norm_enabled:
            assert self.bc_norm_b is not None and self.bc_norm_c is not None
            self.bc_norm_b.fill_(1.0)
            self.bc_norm_c.fill_(1.0)

        output_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            output_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std = std / (2 * (block_idx + 1)) ** 0.5
        init_linear(self.out_proj, std=output_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Estimate logical training-model FLOPs per token.

        The estimate counts projection matmuls, the MIMO outer product/readout, complex sparse
        recurrence, routing, and amortized dictionary hardening. It excludes Python-loop overhead
        and does not imply production throughput.

        :param seq_len: Sequence length used to amortize dictionary hardening.

        :returns: Approximate model FLOPs per token.
        """
        projection_flops = 6 * sum(
            projection.weight.numel()
            for projection in (
                self.xz_proj,
                self.bc_proj,
                self.dynamics_proj,
                self.out_proj,
            )
        )
        shared_state_elements = self.n_heads * self.d_state * self.head_dim
        rank_boundary_elements = self.mimo_rank * shared_state_elements
        outer_and_readout_flops = 3 * 6 * rank_boundary_elements
        complex_recurrence_flops = 3 * 10 * shared_state_elements
        routing_flops = self.n_heads * self.dictionary_size
        dictionary_flops = (
            self.n_heads * self.dictionary_size * self.d_state * self.d_state
        ) // max(seq_len, 1)
        return int(
            projection_flops
            + outer_and_readout_flops
            + complex_recurrence_flops
            + routing_flops
            + dictionary_flops
        )


@SequenceMixerConfig.register("mamba3_flash_pd")
@dataclass
class Mamba3FlashPDSSMMixerConfig(SequenceMixerConfig[Mamba3FlashPDSSMMixer]):
    """Registered configuration for :class:`Mamba3FlashPDSSMMixer`."""

    n_heads: int = 8
    """Number of Flash-PD SSM heads."""
    head_dim: Optional[int] = None
    """Per-head payload width; defaults to ``d_model // n_heads``."""
    d_state: int = 64
    """Number of complex states per head."""
    n_groups: int = 1
    """Number of B/C projection groups shared across heads."""
    mimo_rank: int = 4
    """MIMO rank, with one recovering the SISO shape."""
    dictionary_size: int = 16
    """Number of source-to-destination transition maps per head."""
    ste_temperature: float = 1.0
    """Tempered-softmax gradient temperature for both hard selectors."""
    norm_eps: float = 1e-5
    """RMS normalization epsilon."""
    bc_norm: bool = True
    """Apply RMS normalization to B and C projections."""
    bc_bias: bool = True
    """Include bias in the fused B/C projection."""
    exempt_timescale_params_from_weight_decay: bool = True
    """Tag A-log and dt-bias parameters as weight-decay exempt."""
    a_log_init_min: float = 0.05
    """Minimum initial positive decay-rate magnitude."""
    a_log_init_max: float = 16.0
    """Maximum initial positive decay-rate magnitude."""
    scan_chunk_size: int = 128
    """Top-level recurrence chunk size."""
    scan_checkpoint_stride: int = MAMBA3_FLASH_PD_MAX_CHECKPOINT_STRIDE
    """Maximum hierarchical Triton replay tile; defaults to the production-safe bound."""
    dtype: DType = DType.float32
    """Projection and dictionary parameter dtype."""

    def num_params(self, d_model: int) -> int:
        """
        Count parameters exactly as :meth:`build` creates them.

        :param d_model: Input/output model width.

        :returns: Exact parameter count.
        """
        P = _validate_options(
            d_model=d_model,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            d_state=self.d_state,
            n_groups=self.n_groups,
            mimo_rank=self.mimo_rank,
            dictionary_size=self.dictionary_size,
            ste_temperature=self.ste_temperature,
            norm_eps=self.norm_eps,
            a_log_init_min=self.a_log_init_min,
            a_log_init_max=self.a_log_init_max,
        )
        H, G, R, N, K = (
            self.n_heads,
            self.n_groups,
            self.mimo_rank,
            self.d_state,
            self.dictionary_size,
        )
        inner = H * P
        bc_width = G * R * N
        dynamics_width = H * (K + N + 2)

        params = H * K * N * N
        params += d_model * 2 * inner
        params += d_model * 2 * bc_width
        if self.bc_bias:
            params += 2 * bc_width
        params += d_model * dynamics_width
        params += inner * d_model
        params += 2 * H
        params += P
        params += 2 * H * R * P
        if self.bc_norm:
            params += 2 * N
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> Mamba3FlashPDSSMMixer:
        """
        Build a fused Mamba-3 plus Flash-PD sequence mixer.

        :param d_model: Input/output model width.
        :param layer_idx: Layer index, accepted for the mixer factory contract.
        :param n_layers: Layer count, accepted for the mixer factory contract.
        :param init_device: Initialization device, including ``"meta"``.
        :param cache: Shared buffer cache, unused because decode is unsupported.

        :returns: A configured :class:`Mamba3FlashPDSSMMixer`.
        """
        del layer_idx, n_layers, cache
        return Mamba3FlashPDSSMMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            d_state=self.d_state,
            n_groups=self.n_groups,
            mimo_rank=self.mimo_rank,
            dictionary_size=self.dictionary_size,
            ste_temperature=self.ste_temperature,
            norm_eps=self.norm_eps,
            bc_norm=self.bc_norm,
            bc_bias=self.bc_bias,
            exempt_timescale_params_from_weight_decay=(
                self.exempt_timescale_params_from_weight_decay
            ),
            a_log_init_min=self.a_log_init_min,
            a_log_init_max=self.a_log_init_max,
            scan_chunk_size=self.scan_chunk_size,
            scan_checkpoint_stride=self.scan_checkpoint_stride,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
