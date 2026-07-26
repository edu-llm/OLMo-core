"""
A Mamba-3 / attention hybrid model, mirroring :mod:`olmo_core.nn.transformer`.

This package provides :class:`Mamba3Config` (a config type like
:class:`~olmo_core.nn.transformer.config.TransformerConfig`) that builds a :class:`Mamba3`
model interleaving attention and Mamba-3 state-space layers - by default at a 1:3
attention-to-Mamba-3 ratio (Nemotron-H / Jamba style). The Mamba-3 mixer implements the three
innovations from `Mamba-3 <https://arxiv.org/abs/2603.15569>`_ (exponential-trapezoidal
discretization, complex state via the RoPE trick, and MIMO), backed by an in-repo reference SSD
kernel with an optional fast-kernel dispatch.
"""

from .block import Mamba3Block, ReorderedNormMamba3Block
from .config import Mamba3BlockConfig, Mamba3BlockType, Mamba3Config, Mamba3Type
from .init import InitMethod
from .mamba3_ssd_api import dispatch_mamba3_ssd, has_mamba3, mamba3_ssd_reference
from .mamba3_ssd_fast import (
    fast_block_rotations,
    fast_cumulative_block_rotation,
    fast_mamba3_is_available,
    mamba3_ssd_fast,
)
from .mixer import (
    DEFAULT_D_STATE,
    Mamba3Mixer,
    Mamba3MixerConfig,
    admissible_block_sizes,
    kernel_padded_width,
    mamba3_modules_to_ignore_for_fp8,
)
from .model import Mamba3

__all__ = [
    "Mamba3Type",
    "Mamba3Config",
    "Mamba3",
    "Mamba3BlockType",
    "Mamba3BlockConfig",
    "Mamba3Block",
    "ReorderedNormMamba3Block",
    "Mamba3Mixer",
    "Mamba3MixerConfig",
    "DEFAULT_D_STATE",
    "admissible_block_sizes",
    "kernel_padded_width",
    "mamba3_modules_to_ignore_for_fp8",
    "InitMethod",
    "has_mamba3",
    "mamba3_ssd_reference",
    "dispatch_mamba3_ssd",
    "mamba3_ssd_fast",
    "fast_mamba3_is_available",
    "fast_block_rotations",
    "fast_cumulative_block_rotation",
]
