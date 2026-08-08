"""Backend dispatch for the native Flash PD-SSM scan."""

from collections import Counter
from typing import overload

import torch

from .contracts import NativePDBackend, NativePDMode, ScanMetadata
from .cuda import (
    native_cuda_capability,
    native_cuda_mamba3_siso_surrogate_scan,
    native_cuda_paper_surrogate_scan,
    native_cuda_scan,
)
from .reference import (
    paper_surrogate_reference_scan,
    reference_scan,
    trapezoidal_proposition2_reference_scan,
)
from .routes import compact_hard_selection, prove_selected_maps_bijective

_BACKEND_COUNTERS: Counter[str] = Counter()


def reset_backend_counters() -> None:
    """Reset process-local native scan dispatch counters."""
    _BACKEND_COUNTERS.clear()


def get_backend_counters() -> dict[str, int]:
    """Return a copy of process-local native scan dispatch counters."""
    return dict(_BACKEND_COUNTERS)


def _resolve_mode(
    destination: torch.Tensor,
    routes: torch.Tensor,
    mode: NativePDMode,
) -> NativePDMode:
    if mode != NativePDMode.AUTO:
        return mode
    proof = prove_selected_maps_bijective(destination, routes)
    return NativePDMode.PERMUTATION_GATHER if proof.proven else NativePDMode.GENERAL_SCATTER


def _metadata(
    values: torch.Tensor,
    *,
    chunk_size: int,
    backend: str,
    mode: NativePDMode,
) -> ScanMetadata:
    batch, heads, time, state = values.shape
    chunks = (time + chunk_size - 1) // chunk_size
    # Aggregate and exclusive-prefix triples each hold one int16 map and four FP32 values.
    scratch_elements = 2 * batch * heads * chunks * state * 5
    return ScanMetadata(
        backend=backend,
        mode=mode,
        forward_launches=0 if backend == "reference" else 3,
        backward_launches=0 if backend == "reference" else 1,
        state_shape=(batch, heads, time, state),
        scratch_elements=scratch_elements,
        shared_memory_bytes=28 * state,
    )


@overload
def flash_pd_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    chunk_size: int = 128,
    mode: NativePDMode | str = NativePDMode.AUTO,
    backend: NativePDBackend | str = NativePDBackend.AUTO,
    return_metadata: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    ...


@overload
def flash_pd_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    chunk_size: int = 128,
    mode: NativePDMode | str = NativePDMode.AUTO,
    backend: NativePDBackend | str = NativePDBackend.AUTO,
    return_metadata: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, ScanMetadata]:
    ...


def flash_pd_scan(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    chunk_size: int = 128,
    mode: NativePDMode | str = NativePDMode.AUTO,
    backend: NativePDBackend | str = NativePDBackend.AUTO,
    return_metadata: bool = False,
):
    """
    Evaluate the native vector-state recurrence with explicit safe dispatch.

    Auto dispatch never changes transition semantics: colliding maps select the
    scatter path, while the gather path is selected only after a bijection proof.
    """
    if chunk_size < 1 or chunk_size > 128:
        raise ValueError(f"chunk_size must be in [1, 128], got {chunk_size}")
    requested_backend = NativePDBackend(backend)
    resolved_mode = _resolve_mode(destination, routes, NativePDMode(mode))

    capability = native_cuda_capability(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size=chunk_size,
    )
    use_cuda = capability.available and requested_backend != NativePDBackend.REFERENCE
    if requested_backend == NativePDBackend.CUDA and not capability.available:
        _BACKEND_COUNTERS["cuda_rejected"] += 1
        raise RuntimeError(capability.reason)
    if use_cuda:
        output = native_cuda_scan(
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            chunk_size=chunk_size,
            mode=resolved_mode,
        )
        backend_name = "cuda"
        _BACKEND_COUNTERS[f"cuda_{resolved_mode.value}"] += 1
    else:
        if requested_backend == NativePDBackend.AUTO and diagonal_real.is_cuda:
            _BACKEND_COUNTERS["cuda_fallback"] += 1
        else:
            _BACKEND_COUNTERS["reference"] += 1
        output = reference_scan(
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            mode=resolved_mode,
        )
        backend_name = "reference"
    if not return_metadata:
        return output
    return (
        output[0],
        output[1],
        _metadata(
            diagonal_real,
            chunk_size=chunk_size,
            backend=backend_name,
            mode=resolved_mode,
        ),
    )


def paper_surrogate_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    temperature: float = 1.0,
    chunk_size: int = 128,
    mode: NativePDMode | str = NativePDMode.AUTO,
    backend: NativePDBackend | str = NativePDBackend.AUTO,
    return_metadata: bool = False,
):
    """Run hard Flash-PD forward with the Appendix-C activated-selection backward."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    selection = compact_hard_selection(dictionary_logits, selector_logits)
    resolved_mode = _resolve_mode(selection.destination, selection.routes, NativePDMode(mode))
    requested_backend = NativePDBackend(backend)
    capability = native_cuda_capability(
        selection.destination,
        selection.routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size=chunk_size,
    )
    use_cuda = capability.available and requested_backend != NativePDBackend.REFERENCE
    if requested_backend == NativePDBackend.CUDA and not capability.available:
        _BACKEND_COUNTERS["cuda_training_rejected"] += 1
        raise RuntimeError(capability.reason)
    if use_cuda:
        output = native_cuda_paper_surrogate_scan(
            dictionary_logits,
            selector_logits,
            selection.destination,
            selection.routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            temperature=temperature,
            chunk_size=chunk_size,
            mode=resolved_mode,
        )
        backend_name = "cuda_paper_training"
        _BACKEND_COUNTERS[f"cuda_training_{resolved_mode.value}"] += 1
    else:
        output = paper_surrogate_reference_scan(
            dictionary_logits,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            temperature=temperature,
            mode=resolved_mode,
        )
        backend_name = "reference_paper_surrogate"
        _BACKEND_COUNTERS["reference_paper_surrogate"] += 1
    if not return_metadata:
        return output
    metadata = _metadata(
        diagonal_real,
        chunk_size=chunk_size,
        backend=backend_name,
        mode=resolved_mode,
    )
    if use_cuda:
        batch, heads, time, state = diagonal_real.shape
        dictionary_size = dictionary_logits.shape[1]
        metadata = ScanMetadata(
            backend=metadata.backend,
            mode=metadata.mode,
            forward_launches=3,
            backward_launches=5,
            state_shape=metadata.state_shape,
            scratch_elements=metadata.scratch_elements,
            shared_memory_bytes=metadata.shared_memory_bytes,
            training_sequence_elements=(batch * heads * time + heads * dictionary_size * state),
            dictionary_storage_elements=(heads * dictionary_size * state * state),
        )
    return output[0], output[1], metadata


def mamba3_siso_surrogate_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    *,
    dictionary_temperature: float,
    router_temperature: float,
    chunk_size: int = 64,
    mode: NativePDMode | str = NativePDMode.AUTO,
    backend: NativePDBackend | str = NativePDBackend.AUTO,
    return_metadata: bool = False,
):
    """Run the exact Mamba-3 SISO trapezoidal recurrence with hard PD routing."""
    if chunk_size not in (32, 64, 128):
        raise ValueError(f"chunk_size must be one of (32, 64, 128), got {chunk_size}")
    if dictionary_temperature <= 0 or router_temperature <= 0:
        raise ValueError("dictionary and router temperatures must be positive")
    selection = compact_hard_selection(dictionary_logits, selector_logits)
    resolved_mode = _resolve_mode(selection.destination, selection.routes, NativePDMode(mode))
    requested_backend = NativePDBackend(backend)
    capability = native_cuda_capability(
        selection.destination,
        selection.routes,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        chunk_size=chunk_size,
    )
    use_cuda = capability.available and requested_backend != NativePDBackend.REFERENCE
    if requested_backend == NativePDBackend.CUDA and not capability.available:
        _BACKEND_COUNTERS["cuda_mamba3_siso_rejected"] += 1
        raise RuntimeError(capability.reason)
    if use_cuda:
        output = native_cuda_mamba3_siso_surrogate_scan(
            dictionary_logits,
            selector_logits,
            selection.destination,
            selection.routes,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
            dictionary_temperature=dictionary_temperature,
            router_temperature=router_temperature,
            chunk_size=chunk_size,
            mode=resolved_mode,
        )
        backend_name = "cuda_mamba3_siso"
        _BACKEND_COUNTERS[f"cuda_mamba3_siso_{resolved_mode.value}"] += 1
    else:
        output = trapezoidal_proposition2_reference_scan(
            dictionary_logits,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
            dictionary_temperature=dictionary_temperature,
            router_temperature=router_temperature,
            chunk_size=chunk_size,
            mode=resolved_mode,
        )
        backend_name = "reference_mamba3_siso_proposition2"
        _BACKEND_COUNTERS["reference_mamba3_siso"] += 1
    if not return_metadata:
        return output
    metadata = _metadata(
        diagonal_real,
        chunk_size=chunk_size,
        backend=backend_name,
        mode=resolved_mode,
    )
    batch, heads, time, state = diagonal_real.shape
    chunks = (time + chunk_size - 1) // chunk_size
    dictionary_size = dictionary_logits.shape[1]
    metadata = ScanMetadata(
        backend=metadata.backend,
        mode=metadata.mode,
        forward_launches=3 if use_cuda else 0,
        backward_launches=2 if use_cuda else 0,
        state_shape=metadata.state_shape,
        scratch_elements=5 * batch * heads * chunks * state,
        shared_memory_bytes=max(28 * state, 16 * state),
        training_sequence_elements=(heads * dictionary_size * state),
        dictionary_storage_elements=(heads * dictionary_size * state * state),
    )
    return output[0], output[1], metadata
