"""
Isolated Flash PD-SSM prototype for OLMo-core.

The package is additive: importing it registers ``FlashPDSSMMixerConfig`` as the ``"flash_pd"``
sequence mixer without changing any existing Mamba-3 module, config, factory, or kernel default.
Use :func:`mamba3_olmo3_370m_with_state_tracker` or :func:`replace_state_tracker` for an explicit
opt-in hybrid replacement.
"""

from .autograd import sparse_ste_scan
from .hybrid import (
    StateTracker,
    StateTrackerConfig,
    mamba3_flash_pd_mixed_olmo3_370m,
    mamba3_flash_pd_olmo3_370m,
    mamba3_olmo3_370m_with_state_tracker,
    replace_mamba3_with_fused_flash_pd,
    replace_state_tracker,
)
from .mamba3_flash import (
    Mamba3FlashPDSSMMixer,
    Mamba3FlashPDSSMMixerConfig,
    mamba3_flash_pd_scan,
)
from .mixer import FlashPDSSMImplementation, FlashPDSSMMixer, FlashPDSSMMixerConfig
from .reference import (
    affine_chunkwise_reference,
    affine_recurrent_reference,
    compose_affine,
    sparse_chunkwise_reference,
    sparse_recurrent_reference,
)
from .transition import (
    SparseAffineTransition,
    apply_sparse_affine,
    column_one_hot_to_destination,
    compose_sparse_affine,
    destination_diagonal_to_dense,
    destination_to_column_one_hot,
    selected_transition_destination,
    selected_transition_matrix,
    slope_annealed_hardmax,
    sparse_affine_to_dense,
)
from .triton_kernel import TritonCapability, flash_pd_triton_scan, triton_capability

__all__ = [
    "FlashPDSSMImplementation",
    "FlashPDSSMMixer",
    "FlashPDSSMMixerConfig",
    "Mamba3FlashPDSSMMixer",
    "Mamba3FlashPDSSMMixerConfig",
    "SparseAffineTransition",
    "StateTracker",
    "StateTrackerConfig",
    "TritonCapability",
    "affine_chunkwise_reference",
    "affine_recurrent_reference",
    "apply_sparse_affine",
    "column_one_hot_to_destination",
    "compose_affine",
    "compose_sparse_affine",
    "destination_diagonal_to_dense",
    "destination_to_column_one_hot",
    "flash_pd_triton_scan",
    "mamba3_flash_pd_mixed_olmo3_370m",
    "mamba3_flash_pd_olmo3_370m",
    "mamba3_flash_pd_scan",
    "mamba3_olmo3_370m_with_state_tracker",
    "replace_mamba3_with_fused_flash_pd",
    "replace_state_tracker",
    "selected_transition_destination",
    "selected_transition_matrix",
    "slope_annealed_hardmax",
    "sparse_affine_to_dense",
    "sparse_chunkwise_reference",
    "sparse_recurrent_reference",
    "sparse_ste_scan",
    "triton_capability",
]
