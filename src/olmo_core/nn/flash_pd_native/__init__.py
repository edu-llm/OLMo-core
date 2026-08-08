"""
Paper-faithful native Flash PD-SSM vector-state path.

The package is intentionally separate from ``flash_pd_ssm.mamba3_flash``. Its
state is exactly ``(batch, heads, time, state)`` and has no MIMO payload axis.
"""

from .api import (
    flash_pd_scan,
    get_backend_counters,
    mamba3_siso_surrogate_scan,
    paper_surrogate_scan,
    reset_backend_counters,
)
from .contracts import (
    BijectionProof,
    HardSelection,
    NativePDBackend,
    NativePDMode,
    ScanMetadata,
    SelectorTelemetry,
    SISOAccounting,
    SISOScanCache,
)
from .cuda import NativeCUDACapability, native_cuda_capability
from .mamba3_siso import (
    NativeFlashPDMamba3SISOMixer,
    NativeFlashPDMamba3SISOMixerConfig,
)
from .mixer import NativeFlashPDMixer, NativeFlashPDMixerConfig
from .reference import dense_scan_oracle, trapezoidal_reference_scan
from .routes import compact_hard_selection, prove_selected_maps_bijective

__all__ = [
    "BijectionProof",
    "HardSelection",
    "NativePDBackend",
    "NativeCUDACapability",
    "NativeFlashPDMamba3SISOMixer",
    "NativeFlashPDMamba3SISOMixerConfig",
    "NativeFlashPDMixer",
    "NativeFlashPDMixerConfig",
    "NativePDMode",
    "ScanMetadata",
    "SelectorTelemetry",
    "SISOAccounting",
    "SISOScanCache",
    "compact_hard_selection",
    "dense_scan_oracle",
    "flash_pd_scan",
    "get_backend_counters",
    "mamba3_siso_surrogate_scan",
    "native_cuda_capability",
    "paper_surrogate_scan",
    "prove_selected_maps_bijective",
    "reset_backend_counters",
    "trapezoidal_reference_scan",
]
