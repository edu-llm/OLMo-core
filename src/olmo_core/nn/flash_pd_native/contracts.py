"""Public contracts for the native Flash PD-SSM scan."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

import torch


class NativePDMode(str, Enum):
    """Transition orientation used by the scan."""

    AUTO = "auto"
    GENERAL_SCATTER = "general_scatter"
    PERMUTATION_GATHER = "permutation_gather"


class NativePDBackend(str, Enum):
    """Available native Flash PD-SSM backends."""

    AUTO = "auto"
    REFERENCE = "reference"
    CUDA = "cuda"


@dataclass(frozen=True)
class HardSelection:
    """Compact hard dictionary maps and per-token routes."""

    destination: torch.Tensor
    routes: torch.Tensor


@dataclass(frozen=True)
class BijectionProof:
    """Result of proving that every selected source-to-destination map is bijective."""

    proven: bool
    inverse_destination: Optional[torch.Tensor]
    failing_head: Optional[int] = None
    failing_dictionary: Optional[int] = None


@dataclass(frozen=True)
class ScanMetadata:
    """Observable backend and Appendix-E working-set information."""

    backend: str
    mode: NativePDMode
    forward_launches: int
    backward_launches: int
    state_shape: tuple[int, int, int, int]
    scratch_elements: int
    shared_memory_bytes: int
    training_sequence_elements: int = 0
    dictionary_storage_elements: int = 0
    payload_axes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SISOScanCache:
    """One-token recurrent state for the Mamba-3 SISO PD recurrence."""

    h_real: torch.Tensor
    h_imag: torch.Tensor
    v_real: torch.Tensor
    v_imag: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over cache tensors in stable checkpoint order."""
        return iter((self.h_real, self.h_imag, self.v_real, self.v_imag))


@dataclass(frozen=True)
class SelectorTelemetry:
    """Observable hard-router statistics without an auxiliary balancing loss."""

    route_entropy: torch.Tensor
    dead_entries: torch.Tensor
    ties: torch.Tensor
    route_churn: torch.Tensor


@dataclass(frozen=True)
class SISOAccounting:
    """Exact model-work and native scan storage accounting."""

    parameters: int
    flops_per_token: int
    model_flops_per_sequence: int
    nonlinear_evaluations_per_sequence: int
    route_comparisons_per_sequence: int
    saved_tensor_bytes: int
    forward_workspace_bytes: int
    backward_workspace_bytes: int
    peak_workspace_bytes: int
