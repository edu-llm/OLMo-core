"""
Measurement over saved checkpoints: achieved bits, recall, and the reasoning endpoints.

Nothing here trains. Every module reads a checkpoint that ``train_cell.py`` wrote, rebuilds the exact
corpus that produced it from the cell record saved beside the weights, and scores it. That split is
deliberate: :mod:`factcrowd.measure.recall` needs free-running generation, which
``TransformerGenerationModule`` cannot provide from inside a training callback without re-parallelising
the model, so measurement is a post-hoc job over checkpoints rather than a callback (PRD 8.2).

The layering, innermost first:

``spans``
    Where a token's cross-entropy lives. One rule, one place, because an off-by-one here silently
    corrupts every bit count and every endpoint score.
``endpoints``
    :class:`~factcrowd.measure.endpoints.EndpointResult` -- three counts, an accuracy, answer-token CE
    in bits, and the measured floor. The shape every endpoint reports in.
``reasoning``, ``bits``, ``recall``
    The three measurements. Each takes a ``forward`` callable rather than a model, so all of the logic
    is testable without a GPU or a checkpoint.
``gates``
    PRD 8.6's G1-G8. An endpoint that has not passed them cannot be read.
``collect``
    Walks a run prefix, joins each checkpoint's cell record with its scores, and emits one tidy row per
    (cell, replicate, step) for analysis.
"""
