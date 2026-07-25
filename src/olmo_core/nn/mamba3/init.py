"""
Weight initialization for the Mamba-3 hybrid.

The hybrid reuses the transformer :class:`~olmo_core.nn.transformer.init.InitMethod` verbatim
(the :class:`~olmo_core.nn.mamba3.mixer.Mamba3Mixer` implements the ``init_weights`` contract it
expects). This module re-exports it so the ``mamba3`` package mirrors the ``transformer`` layout.
"""

from ..transformer.init import InitMethod, init_linear

__all__ = ["InitMethod", "init_linear"]
