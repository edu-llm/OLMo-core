"""
Score a reasoning endpoint on one checkpoint.

The dependent variable. Everything the design does to make this readable happens elsewhere -- fixed-width
items, a single-token answer at a known position, a train-disjoint ``eval`` split, a measured floor -- so
what is left here is a loop and an argmax.

That is the point. PRD 1 lists four endpoints in this programme that produced uninterpretable nulls, and
every one failed in the scoring rather than in the model: an eval that graded one integer and discarded
the derivation, a parser that read a truncated continuation as a wrong answer, a macro-average over
families with floors from 0 to 0.5. None of those failures is expressible here. There is no continuation
to truncate, no string to parse, and one endpoint per result.

The scorer takes a ``forward`` callable rather than a model, so every line below is testable on a stub
that returns chosen logits -- including the cases a real model would almost never produce.
"""

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..corpus.tasks import ReasoningTask
from .endpoints import EndpointAccumulator, EndpointResult
from .spans import predicted_token, span_bits

Forward = Callable[[np.ndarray], Tuple["np.ndarray", "np.ndarray"]]
"""
A batched forward pass.

Takes token ids of shape ``(batch, sequence)`` and returns ``(ce_loss, logits)`` with shapes
``(batch, sequence)`` and ``(batch, sequence, vocab)``. Whatever indexes like that will do -- torch
tensors, numpy arrays, or a stub -- which is what keeps the scoring logic free of the GPU stack.
"""


def score_reasoning(
    task: ReasoningTask,
    forward: Forward,
    *,
    n_items: int,
    batch_size: int = 64,
    floor_sample: int = 20_000,
    floor: Optional[float] = None,
    degenerate_answer: Optional[Tuple[str, ...]] = None,
) -> EndpointResult:
    """
    Score ``n_items`` held-out items of one task.

    :param task: The task, **constructed with** ``split="eval"``. Passing a training-split task is
        refused rather than silently scoring items the model was trained on.
    :param forward: See :data:`Forward`.
    :param n_items: How many items to score. The frozen evaluation set is the first ``n_items`` of the
        eval split, so the same items are scored at every checkpoint and every cell.
    :param batch_size: Items per forward pass.
    :param floor_sample: Items to draw when measuring the degenerate baseline.
    :param floor: Pre-measured floor, to avoid re-measuring it per checkpoint. Measured here if omitted.
    :param degenerate_answer: Pre-measured best fact-free answer, paired with ``floor``.

    :returns: The endpoint's score.

    :raises OLMoConfigurationError: If the task is on the training split, or ``n_items`` is not positive.
    """
    if task.split != "eval":
        raise OLMoConfigurationError(
            f"endpoint '{task.name}' was handed a task on the '{task.split}' split. Scoring the split "
            f"the model trained on measures memorisation, not reasoning -- build the task with "
            f"split='eval'."
        )
    if n_items <= 0:
        raise OLMoConfigurationError(f"n_items must be positive, got {n_items}")

    # `floor is None` alone, not `or degenerate_answer is None`. With `or`, passing a pre-measured floor
    # without its answer silently re-measured and discarded it -- and re-measuring costs seconds per
    # endpoint per checkpoint, which is a minute per cell of pure repetition over ten checkpoints.
    if floor is None:
        label, measured = task.degenerate_baseline(floor_sample)
        floor = measured
        if degenerate_answer is None:
            degenerate_answer = _answer_of(task, label)

    accumulator = EndpointAccumulator(task.name, floor=floor, degenerate_answer=degenerate_answer)
    for start in range(0, n_items, batch_size):
        stop = min(start + batch_size, n_items)
        items = [task.item(index) for index in range(start, stop)]
        batch = np.stack([item.tokens for item in items])
        ce_loss, logits = forward(batch)
        _check_shapes(ce_loss, logits, batch)
        for row, item in enumerate(items):
            predicted, parseable = _decode_answer(task, logits[row], item)
            accumulator.add(
                predicted=predicted,
                expected=item.answer,
                ce_bits=span_bits(ce_loss[row], item.answer_start, item.answer_end),
                parseable=parseable,
            )
    return accumulator.result()


def _decode_answer(task: ReasoningTask, logits, item) -> Tuple[Tuple[str, ...], bool]:
    """
    Read the model's answer, or report that there was not one.

    **The output layer is wider than the vocabulary.** ``padded_size()`` rounds up to a multiple of 128
    for the matmul, so a model can argmax into the padding -- 65 ids on the entropy axis, 31 on the count
    axis -- where no word exists. An untrained checkpoint does this readily.

    That is exactly what ``n_unparseable`` is for, and it is the only way the count becomes non-zero on
    these endpoints: the model emitted something that is not a word, which is neither correct nor a wrong
    answer. Indexing the word list without checking raised ``IndexError`` and took the whole scoring job
    down, which is a worse failure than the one it was hiding.

    :param task: The task, for its vocabulary.
    :param logits: One sequence's logits.
    :param item: The item being graded.

    :returns: The answer as words, and whether every position decoded to a real word. On failure the
        words decoded so far are still returned, for the log.
    """
    words: List[str] = []
    parseable = True
    real_words = task.vocabulary.size
    for position in range(item.answer_start, item.answer_end):
        token = predicted_token(logits, position)
        if not 0 <= token < real_words:
            parseable = False
            continue
        words.append(task.vocabulary.words[token])
    return tuple(words), parseable


def _answer_of(task: ReasoningTask, label: str) -> Optional[Tuple[str, ...]]:
    """
    Recover the winning policy's answer from a :meth:`degenerate_baseline` label.

    Only a constant policy names an answer the model could emit. A copy policy is a rule about position
    rather than a fixed string, so there is nothing to count occurrences of and the degenerate count is
    left at zero -- with the floor itself still reported, which is what the gate reads.
    """
    if label.startswith("constant:"):
        return tuple(label[len("constant:") :].split(" "))
    return None


def _check_shapes(ce_loss, logits, batch: np.ndarray) -> None:
    """Fail loudly on a ``forward`` that does not return what :data:`Forward` promises."""
    if tuple(ce_loss.shape) != tuple(batch.shape):
        raise OLMoConfigurationError(
            f"forward returned ce_loss of shape {tuple(ce_loss.shape)} for a batch of "
            f"{tuple(batch.shape)}; per-token loss must align with the input"
        )
    if tuple(logits.shape[:2]) != tuple(batch.shape):
        raise OLMoConfigurationError(
            f"forward returned logits of shape {tuple(logits.shape)} for a batch of "
            f"{tuple(batch.shape)}"
        )


def score_all(
    tasks: Sequence[ReasoningTask], forward: Forward, *, n_items: int, batch_size: int = 64
) -> Tuple[EndpointResult, ...]:
    """
    Score several endpoints on one checkpoint.

    :param tasks: The tasks, each on the ``eval`` split.
    :param forward: See :data:`Forward`.
    :param n_items: Items per endpoint.
    :param batch_size: Items per forward pass.

    :returns: One result per task, in order.
    """
    return tuple(
        score_reasoning(task, forward, n_items=n_items, batch_size=batch_size) for task in tasks
    )
