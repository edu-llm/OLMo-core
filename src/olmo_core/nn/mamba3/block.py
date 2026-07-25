"""
Transformer-style blocks for the Mamba-3 hybrid.

These mirror :mod:`olmo_core.nn.transformer.block` and are intentionally thin subclasses:
a block is behaviorally identical whether its sequence mixer is attention or a
:class:`~olmo_core.nn.mamba3.mixer.Mamba3Mixer` - the *mixer* determines the layer type, so
the 1:3 attention-to-Mamba-3 hybrid is expressed purely via named blocks + ``block_pattern``
(see :class:`~olmo_core.nn.mamba3.config.Mamba3Config`). Subclassing keeps all parallelism,
residual, and forward logic inherited from the transformer blocks with zero drift.
"""

from ..transformer.block import ReorderedNormTransformerBlock, TransformerBlock

__all__ = ["Mamba3Block", "ReorderedNormMamba3Block"]


class Mamba3Block(TransformerBlock):
    """
    A Llama-style pre-norm block for the Mamba-3 hybrid (``norm -> sequence_mixer -> residual ->
    norm -> feed_forward -> residual``). Identical to :class:`TransformerBlock`; the sequence
    mixer may be attention or a :class:`~olmo_core.nn.mamba3.mixer.Mamba3Mixer`.
    """


class ReorderedNormMamba3Block(ReorderedNormTransformerBlock):
    """
    The OLMo2-style reordered-norm variant of :class:`Mamba3Block` (norm applied after each
    sublayer's output). Identical to :class:`ReorderedNormTransformerBlock`.
    """
