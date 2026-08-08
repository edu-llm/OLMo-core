"""Linear-memory autograd for collision-preserving Flash PD transitions."""

from typing import Any

import torch
from torch.nn import functional as F

from .reference import sparse_recurrent_reference
from .transition import (
    destination_to_column_one_hot,
    selected_transition_destination,
)
from .triton_kernel import flash_pd_triton_scan

__all__ = ["sparse_ste_scan"]


class _SparseSTEScan(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        dictionary_logits: torch.Tensor,
        selector_logits: torch.Tensor,
        diagonal: torch.Tensor,
        bias: torch.Tensor,
        temperature: float,
        use_triton: bool,
        chunk_size: int,
    ) -> torch.Tensor:
        destination_bt = selected_transition_destination(
            dictionary_logits,
            selector_logits,
        )
        destination = destination_bt.permute(0, 2, 1, 3).contiguous()
        route = selector_logits.argmax(dim=-1)

        if use_triton:
            states = flash_pd_triton_scan(
                destination,
                diagonal.detach(),
                bias.detach(),
                chunk_size=chunk_size,
            )
        else:
            states = sparse_recurrent_reference(destination, diagonal, bias)

        ctx.save_for_backward(
            dictionary_logits,
            selector_logits,
            diagonal,
            states,
            destination,
            route,
        )
        ctx.temperature = temperature
        return states

    @staticmethod
    def backward(
        ctx: Any,
        grad_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None,]:
        (
            dictionary_logits,
            selector_logits,
            diagonal,
            states,
            destination,
            route,
        ) = ctx.saved_tensors
        temperature = ctx.temperature
        batch, heads, time, state = diagonal.shape
        dictionary_size = dictionary_logits.shape[1]

        grad_diagonal = torch.zeros_like(diagonal)
        grad_bias = torch.zeros_like(grad_states)
        grad_selector = torch.zeros_like(selector_logits)
        dictionary_transition_grad = torch.zeros_like(dictionary_logits)
        carry = torch.zeros(
            batch,
            heads,
            state,
            dtype=grad_states.dtype,
            device=grad_states.device,
        )
        zero_state = torch.zeros_like(carry)

        dictionary_destination = dictionary_logits.argmax(dim=-2)
        hard_dictionary = destination_to_column_one_hot(
            dictionary_destination,
            dtype=dictionary_logits.dtype,
        )
        selector_probability = torch.softmax(selector_logits / temperature, dim=-1)

        for token_idx in range(time - 1, -1, -1):
            total_grad = grad_states[:, :, token_idx] + carry
            grad_bias[:, :, token_idx] = total_grad
            previous_state = zero_state if token_idx == 0 else states[:, :, token_idx - 1]
            token_destination = destination[:, :, token_idx].long()
            token_diagonal = diagonal[:, :, token_idx]

            grad_at_destination = torch.gather(
                total_grad,
                dim=-1,
                index=token_destination,
            )
            grad_diagonal[:, :, token_idx] = grad_at_destination * previous_state.conj()
            carry = grad_at_destination * token_diagonal.conj()

            transitioned_source = token_diagonal * previous_state
            transition_grad = torch.einsum(
                "bhi,bhq->bhiq",
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
                - (dictionary_probability * dictionary_transition_grad).sum(
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
            grad_bias,
            None,
            None,
            None,
        )


def sparse_ste_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    bias: torch.Tensor,
    *,
    temperature: float,
    use_triton: bool = False,
    chunk_size: int = 128,
) -> torch.Tensor:
    """
    Run the hard Flash PD recurrence with exact state gradients and STE selector gradients.

    Unlike the dense correctness oracle, this stores only states and compact integer transitions
    per token. Dictionary-gradient outer products are formed one timestep at a time in backward,
    keeping activation memory linear in ``state_size``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return _SparseSTEScan.apply(
        dictionary_logits,
        selector_logits,
        diagonal,
        bias,
        temperature,
        use_triton,
        chunk_size,
    )
