"""Structured sparse transition algebra for Flash PD-SSM."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

__all__ = [
    "SparseAffineTransition",
    "apply_sparse_affine",
    "column_one_hot_to_destination",
    "compose_sparse_affine",
    "destination_diagonal_to_dense",
    "destination_to_column_one_hot",
    "selected_transition_destination",
    "selected_transition_matrix",
    "slope_annealed_hardmax",
    "sparse_affine_to_dense",
]


@dataclass(frozen=True)
class SparseAffineTransition:
    """
    A column-sparse affine map with one destination per source coordinate.

    ``destination[..., q]`` names the output coordinate receiving source coordinate ``q``;
    ``scale[..., q]`` is its (possibly complex) coefficient, and ``bias[..., i]`` is the
    affine term at destination ``i``. Multiple sources may share one destination, in which
    case their contributions are summed.

    :param destination: Integer destination indices of shape ``(..., state_size)``.
    :param scale: Coefficients with the same shape as ``destination``.
    :param bias: Affine terms with the same shape as ``destination``.
    """

    destination: torch.Tensor
    scale: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if self.destination.shape != self.scale.shape or self.destination.shape != self.bias.shape:
            raise ValueError(
                "destination, scale, and bias must have identical shapes, got "
                f"{self.destination.shape}, {self.scale.shape}, and {self.bias.shape}"
            )
        if self.destination.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"destination must contain integer indices, got {self.destination.dtype}"
            )


def slope_annealed_hardmax(
    logits: torch.Tensor,
    *,
    dim: int,
    temperature: float,
) -> torch.Tensor:
    """
    Return an argmax one-hot tensor with tempered-softmax surrogate gradients.

    The forward value is exactly discrete. In the backward pass it has the Jacobian of
    ``softmax(logits / temperature)``. Decreasing ``temperature`` over training implements
    slope annealing, since the equivalent softmax slope is ``1 / temperature``.

    :param logits: Values to select from.
    :param dim: Selection dimension.
    :param temperature: Positive backward-pass softmax temperature.

    :returns: A one-hot tensor with the same shape and dtype as ``logits``.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    soft = torch.softmax(logits / temperature, dim=dim)
    indices = logits.argmax(dim=dim, keepdim=True)
    hard = torch.zeros_like(logits).scatter_(dim, indices, 1)
    return (hard - soft).detach() + soft


def destination_to_column_one_hot(
    destination: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Materialize a column-one-hot matrix from source-to-destination indices.

    The returned matrix has ``matrix[..., destination[q], q] == 1``. Multiple sources may select
    the same destination, as required for general deterministic finite-state transitions.

    :param destination: Destination indices of shape ``(..., state_size)``.
    :param dtype: Output matrix dtype.

    :returns: A matrix of shape ``(..., state_size, state_size)``.
    """
    if destination.ndim < 1:
        raise ValueError("destination must have at least one dimension")
    state_size = destination.shape[-1]
    return F.one_hot(destination.long(), num_classes=state_size).movedim(-1, -2).to(dtype=dtype)


def column_one_hot_to_destination(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compress a column-one-hot matrix into source-to-destination indices.

    :param matrix: Matrix of shape ``(..., state_size, state_size)``.

    :returns: Active row indices of shape ``(..., state_size)``.
    """
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"matrix must be square on its last two axes, got {matrix.shape}")
    return matrix.argmax(dim=-2)


def selected_transition_matrix(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """
    Select one column-one-hot transition dictionary entry per token with two STEs.

    ``dictionary_logits`` is hardened column-wise (Equation 5 of the paper), then
    ``selector_logits`` chooses one dictionary member per token (Equation 7). Both argmaxes are
    exactly discrete in the forward pass and use slope-annealed softmax gradients.

    :param dictionary_logits: Dense trainable dictionary of shape
        ``(heads, dictionary_size, state_size, state_size)``.
    :param selector_logits: Per-token dictionary scores of shape
        ``(batch, time, heads, dictionary_size)``.
    :param temperature: Positive STE softmax temperature.

    :returns: Selected column-one-hot matrices of shape
        ``(batch, time, heads, state_size, state_size)``.
    """
    if dictionary_logits.ndim != 4:
        raise ValueError("dictionary_logits must have shape (heads, dictionary_size, state, state)")
    if dictionary_logits.shape[-1] != dictionary_logits.shape[-2]:
        raise ValueError("dictionary transition matrices must be square")
    if selector_logits.ndim != 4:
        raise ValueError("selector_logits must have shape (batch, time, heads, dictionary_size)")
    if selector_logits.shape[-2:] != dictionary_logits.shape[:2]:
        raise ValueError(
            "selector heads/dictionary dimensions do not match dictionary_logits: "
            f"{selector_logits.shape[-2:]} != {dictionary_logits.shape[:2]}"
        )

    dictionary = slope_annealed_hardmax(
        dictionary_logits,
        dim=-2,
        temperature=temperature,
    )
    selector = slope_annealed_hardmax(
        selector_logits,
        dim=-1,
        temperature=temperature,
    )
    return torch.einsum("bthk,hknm->bthnm", selector, dictionary)


def selected_transition_destination(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Select compact hard source-to-destination maps without per-token square matrices.

    This inference-only helper intentionally has no straight-through gradient. Training paths
    must use :func:`selected_transition_matrix`.
    """
    if dictionary_logits.ndim != 4:
        raise ValueError("dictionary_logits must have shape (heads, dictionary_size, state, state)")
    if selector_logits.ndim != 4:
        raise ValueError("selector_logits must have shape (batch, time, heads, dictionary_size)")
    if selector_logits.shape[-2:] != dictionary_logits.shape[:2]:
        raise ValueError(
            "selector heads/dictionary dimensions do not match dictionary_logits: "
            f"{selector_logits.shape[-2:]} != {dictionary_logits.shape[:2]}"
        )

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
    return torch.gather(dictionary, dim=3, index=index).squeeze(3)


def destination_diagonal_to_dense(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    """
    Build the dense transition ``P @ diag(D)`` used by the reference path.

    This orientation implements ``x_t[i] = b_t[i] + sum_{q:p_t[q]=i} D_t[q] x_(t-1)[q]``.

    :param destination: Destination per source, shape ``(..., state_size)``.
    :param diagonal: Complex or real diagonal values with the same shape.

    :returns: Dense matrices of shape ``(..., state_size, state_size)``.
    """
    if destination.shape != diagonal.shape:
        raise ValueError(
            "destination and diagonal shapes differ: " f"{destination.shape} != {diagonal.shape}"
        )
    column_one_hot = destination_to_column_one_hot(destination, dtype=diagonal.dtype)
    return column_one_hot * diagonal.unsqueeze(-2)


def apply_sparse_affine(
    transition: SparseAffineTransition,
    state: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a structured sparse affine transition to a state.

    :param transition: Transition to apply.
    :param state: State with the same shape as ``transition.destination``.

    :returns: The transformed state.
    """
    if state.shape != transition.destination.shape:
        raise ValueError(
            f"state shape must match transition shape, got {state.shape} and "
            f"{transition.destination.shape}"
        )
    transformed = torch.zeros_like(state).scatter_add(
        dim=-1,
        index=transition.destination.long(),
        src=transition.scale * state,
    )
    return transition.bias + transformed


def compose_sparse_affine(
    later: SparseAffineTransition,
    earlier: SparseAffineTransition,
) -> SparseAffineTransition:
    """
    Compose two sparse affine maps as ``later(earlier(state))``.

    This is the structured form of ``(A2, b2) o (A1, b1) =
    (A2 A1, A2 b1 + b2)``. Source-to-destination maps are closed under this operation, so
    composition remains linear in the state size and is associative.

    :param later: Transition applied second.
    :param earlier: Transition applied first.

    :returns: Their structured sparse composition.
    """
    if later.destination.shape != earlier.destination.shape:
        raise ValueError(
            "transition shapes differ: " f"{later.destination.shape} != {earlier.destination.shape}"
        )
    earlier_destination = earlier.destination.long()
    destination = torch.gather(later.destination, dim=-1, index=earlier_destination)
    later_scale = torch.gather(later.scale, dim=-1, index=earlier_destination)
    transformed_bias = torch.zeros_like(earlier.bias).scatter_add(
        dim=-1,
        index=later.destination.long(),
        src=later.scale * earlier.bias,
    )
    return SparseAffineTransition(
        destination=destination,
        scale=earlier.scale * later_scale,
        bias=later.bias + transformed_bias,
    )


def sparse_affine_to_dense(
    transition: SparseAffineTransition,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Materialize a structured affine transition for validation.

    :param transition: Sparse transition to materialize.

    :returns: ``(matrix, bias)`` where ``matrix`` has shape
        ``(..., state_size, state_size)``.
    """
    matrix = destination_to_column_one_hot(
        transition.destination,
        dtype=transition.scale.dtype,
    )
    matrix = matrix * transition.scale.unsqueeze(-2)
    return matrix, transition.bias
