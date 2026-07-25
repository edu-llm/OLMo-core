"""
The Mamba-3 hybrid model.

:class:`Mamba3` mirrors :class:`~olmo_core.nn.transformer.model.Transformer` but is a subclass,
so all of the training-critical machinery - input preparation, RoPE buffer handling, weight
init, and the ``apply_tp``/``apply_cp``/``apply_fsdp``/``apply_ddp`` parallelism hooks - is
inherited unchanged. Crucially, subclassing also makes ``isinstance(model, Transformer)`` true,
which the transformer train module relies on (``parallelize_model``), so a :class:`Mamba3`
trains via the existing :class:`~olmo_core.train.train_module.TransformerTrainModule` with no
changes.

The 1:3 attention-to-Mamba-3 interleaving is not a property of the model class; it is expressed
by the named blocks + ``block_pattern`` on :class:`~olmo_core.nn.mamba3.config.Mamba3Config`.
"""

from ..transformer.model import Transformer

__all__ = ["Mamba3"]


class Mamba3(Transformer):
    """
    A hybrid attention + Mamba-3 language model.

    See :class:`~olmo_core.nn.transformer.model.Transformer` for constructor parameters; this
    subclass adds no new behavior. Build one from :class:`~olmo_core.nn.mamba3.config.Mamba3Config`.
    """
