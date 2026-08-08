"""Compact hard-map construction and permutation preflight checks."""

import torch

from .contracts import BijectionProof, HardSelection


def _validate_compact_shapes(destination: torch.Tensor, routes: torch.Tensor) -> None:
    if destination.ndim != 3:
        raise ValueError("destination must have shape (heads, dictionary, state)")
    if routes.ndim != 3:
        raise ValueError("routes must have shape (batch, heads, time)")
    if routes.shape[1] != destination.shape[0]:
        raise ValueError("route heads must match destination heads")
    state = destination.shape[-1]
    dictionary_size = destination.shape[1]
    if state < 1 or state >= 1024:
        raise ValueError(f"state size must be in [1, 1024), got {state}")
    if dictionary_size < 1:
        raise ValueError("dictionary size must be positive")
    if destination.numel():
        minimum = int(destination.min().item())
        maximum = int(destination.max().item())
        if minimum < 0 or maximum >= state:
            raise ValueError(
                f"destination indices must be in [0, {state}), got [{minimum}, {maximum}]"
            )
    if routes.numel():
        minimum = int(routes.min().item())
        maximum = int(routes.max().item())
        if minimum < 0 or maximum >= dictionary_size:
            raise ValueError(
                f"route indices must be in [0, {dictionary_size}), got [{minimum}, {maximum}]"
            )


def compact_hard_selection(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
) -> HardSelection:
    """
    Build the paper's compact Equation-5 maps and Equation-7 routes.

    ``destination[h, k, source]`` is the selected destination row for that source
    column. No per-token ``N x N`` tensor is materialized.
    """
    if dictionary_logits.ndim != 4:
        raise ValueError("dictionary_logits must have shape (heads, dictionary, state, state)")
    if dictionary_logits.shape[-1] != dictionary_logits.shape[-2]:
        raise ValueError("dictionary matrices must be square")
    if selector_logits.ndim != 4:
        raise ValueError("selector_logits must have shape (batch, time, heads, dictionary)")
    if selector_logits.shape[-2:] != dictionary_logits.shape[:2]:
        raise ValueError("selector heads/dictionary dimensions do not match dictionary_logits")
    state = dictionary_logits.shape[-1]
    dictionary_size = dictionary_logits.shape[1]
    if state >= 1024:
        raise ValueError(f"int16 compact maps require state size below 1024, got {state}")
    if dictionary_size >= 32768:
        raise ValueError(
            f"int16 compact routes require dictionary size below 32768, got {dictionary_size}"
        )

    destination = dictionary_logits.argmax(dim=-2).to(torch.int16)
    routes = selector_logits.argmax(dim=-1).permute(0, 2, 1).contiguous().to(torch.int16)
    return HardSelection(destination=destination.contiguous(), routes=routes)


def prove_selected_maps_bijective(
    destination: torch.Tensor,
    routes: torch.Tensor,
) -> BijectionProof:
    """
    Prove bijectivity for every dictionary entry selected by ``routes``.

    The returned inverse is indexed by destination and is the only map admitted by
    the Appendix-E gather path. Unselected colliding dictionary entries do not block
    a call, but selecting one on a later call triggers a fresh failed proof.
    """
    _validate_compact_shapes(destination, routes)
    heads, dictionary_size, state = destination.shape
    inverse = torch.empty_like(destination, dtype=torch.int16)
    expected = torch.arange(state, device=destination.device, dtype=torch.long)

    for head in range(heads):
        selected = torch.unique(routes[:, head].long())
        for dictionary_idx_tensor in selected:
            dictionary_idx = int(dictionary_idx_tensor.item())
            values = destination[head, dictionary_idx].long()
            if not torch.equal(torch.sort(values).values, expected):
                return BijectionProof(
                    proven=False,
                    inverse_destination=None,
                    failing_head=head,
                    failing_dictionary=dictionary_idx,
                )
            inverse[head, dictionary_idx, values] = expected.to(torch.int16)

        unused = torch.ones(dictionary_size, dtype=torch.bool, device=destination.device)
        unused[selected] = False
        inverse[head, unused] = 0

    return BijectionProof(proven=True, inverse_destination=inverse.contiguous())
