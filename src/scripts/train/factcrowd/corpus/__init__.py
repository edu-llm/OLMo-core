"""
Corpus generation: the entity table, the renderer, the reasoning slices, and the mixture.

Nothing here imports ``torch``. These modules hold the arithmetic the experiment's validity rests
on -- exact bit accounting, exposure counts, absolute-token mixing -- and an invariant that can
only be checked on a GPU node is an invariant that stops being checked.
"""
