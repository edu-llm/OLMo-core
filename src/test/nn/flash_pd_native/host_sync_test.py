"""Host-synchronization contract for native Flash PD-SSM compact route validation."""

from typing import Any, Callable, Dict, List

import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    NativePDBackend,
    NativePDMode,
    compact_hard_selection,
    paper_surrogate_scan,
)
from olmo_core.nn.flash_pd_native.routes import _validate_compact_shapes

# Every way a tensor element can be pulled back to the host as a Python scalar.
# On CUDA each one drains the stream, which is what this file exists to forbid.
_READBACKS = ("item", "tolist", "__bool__", "__float__", "__index__", "__int__")

_HEADS = 2
_DICTIONARY = 3
_STATE = 4
_BATCH = 2
_TIME = 6


class _ReadbackCounter:
    """Count device-to-host scalar reads made inside the ``with`` block."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self._originals: Dict[str, Any] = {}

    def __enter__(self) -> "_ReadbackCounter":
        for name in _READBACKS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            setattr(torch.Tensor, name, self._wrap(name, original))
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)

    def _wrap(self, name: str, original: Callable) -> Callable:
        def wrapper(tensor, *args, **kwargs):
            self.calls.append(name)
            return original(tensor, *args, **kwargs)

        return wrapper


def _selection_logits(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    dictionary_logits = torch.randn(_HEADS, _DICTIONARY, _STATE, _STATE, device=device)
    selector_logits = torch.randn(_BATCH, _TIME, _HEADS, _DICTIONARY, device=device)
    return dictionary_logits, selector_logits


def _split_values(device: torch.device) -> list[torch.Tensor]:
    return [torch.randn(_BATCH, _HEADS, _TIME, _STATE, device=device) * 0.1 for _ in range(4)]


def test_argmax_selection_validation_makes_no_host_readback():
    selection = compact_hard_selection(*_selection_logits(torch.device("cpu")))

    with _ReadbackCounter() as counter:
        _validate_compact_shapes(selection.destination, selection.routes)

    assert counter.calls == []


def test_paper_surrogate_dispatch_makes_no_host_readback():
    dictionary_logits, selector_logits = _selection_logits(torch.device("cpu"))
    values = _split_values(torch.device("cpu"))

    with _ReadbackCounter() as counter:
        paper_surrogate_scan(
            dictionary_logits,
            selector_logits,
            *values,
            mode=NativePDMode.GENERAL_SCATTER,
            backend=NativePDBackend.REFERENCE,
        )

    assert counter.calls == []


@pytest.mark.parametrize(
    "destination_values, routes_values, message",
    [
        ([[[0, 4]]], [[[0, 0, 0]]], "destination indices must be in"),
        ([[[0, -1]]], [[[0, 0, 0]]], "destination indices must be in"),
        ([[[0, 1]]], [[[0, 3, 0]]], "route indices must be in"),
        ([[[0, 1]]], [[[0, -2, 0]]], "route indices must be in"),
    ],
)
def test_externally_supplied_out_of_range_indices_still_raise(
    destination_values: list, routes_values: list, message: str
):
    destination = torch.tensor(destination_values, dtype=torch.int16)
    routes = torch.tensor(routes_values, dtype=torch.int16)

    with pytest.raises(ValueError, match=message):
        _validate_compact_shapes(destination, routes)


def test_in_place_mutation_revokes_structural_trust():
    selection = compact_hard_selection(*_selection_logits(torch.device("cpu")))
    _validate_compact_shapes(selection.destination, selection.routes)

    selection.destination[0, 0, 0] = _STATE

    with pytest.raises(ValueError, match="destination indices must be in"):
        _validate_compact_shapes(selection.destination, selection.routes)


def test_derived_copies_are_not_trusted_by_provenance():
    selection = compact_hard_selection(*_selection_logits(torch.device("cpu")))
    forged = selection.destination.clone()
    forged[0, 0, 0] = _STATE + 1

    with pytest.raises(ValueError, match="destination indices must be in"):
        _validate_compact_shapes(forged, selection.routes)


@pytest.mark.gpu
def test_cuda_paper_surrogate_dispatch_never_synchronizes_the_stream():
    device = torch.device("cuda")
    dictionary_logits, selector_logits = _selection_logits(device)
    values = _split_values(device)

    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with _ReadbackCounter() as counter:
            paper_surrogate_scan(
                dictionary_logits,
                selector_logits,
                *values,
                mode=NativePDMode.GENERAL_SCATTER,
                backend=NativePDBackend.AUTO,
            )
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert counter.calls == []


@pytest.mark.gpu
def test_cuda_out_of_range_indices_are_rejected_without_readback():
    device = torch.device("cuda")
    destination = torch.tensor([[[0, 4]]], dtype=torch.int16, device=device)
    routes = torch.zeros((1, 1, 3), dtype=torch.int16, device=device)

    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        with _ReadbackCounter() as counter:
            _validate_compact_shapes(destination, routes)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert counter.calls == []
    with pytest.raises(RuntimeError):
        torch.cuda.synchronize()
