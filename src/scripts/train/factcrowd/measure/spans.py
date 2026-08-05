"""
Where a token's cross-entropy lives, and the conversion to bits.

This module exists to hold **one rule** in one place. OLMo-core builds labels with
:func:`olmo_core.data.utils.get_labels`, which is ``pad(input_ids[..., 1:], (0, 1), value=-100)`` --
so position ``t`` of the loss scores the token at position ``t + 1`` of the input, and the cost of the
token at position ``p`` is ``ce_loss[p - 1]``.

Getting that backwards would shift every measurement by one token: a bit count would attribute a value
token's cost to the literal before it, and an endpoint would grade the token before the answer. Neither
would look wrong. So the arithmetic is written once, exercised directly by tests against a manually
computed cross-entropy, and every caller goes through it.
"""

import math
from typing import TYPE_CHECKING, Tuple

from olmo_core.exceptions import OLMoConfigurationError

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import torch

BITS_PER_NAT: float = 1.0 / math.log(2.0)
"""
Nats to bits. OLMo-core's cross-entropy is in nats; Allen-Zhu's bound is in bits.

The same ``1/ln2`` PRD 8.1 names. It is a constant rather than an inline literal because a bit count is
the experiment's x-axis and a factor of 1.44 in it would be invisible in a plot.
"""


def predictor_slice(start: int, end: int, sequence_length: int) -> Tuple[int, int]:
    """
    The loss positions that score the tokens in ``[start, end)``.

    :param start: First token position being scored. Must be at least 1 -- position 0 is predicted from
        nothing, so it has no cost and cannot be part of a measured span.
    :param end: One past the last token position being scored.
    :param sequence_length: Length of the sequence the span lives in.

    :returns: ``(start - 1, end - 1)``, ready to slice a loss row.

    :raises OLMoConfigurationError: If the span is empty, inverted, starts at 0, or runs past the end.
    """
    if end <= start:
        raise OLMoConfigurationError(f"span [{start}, {end}) is empty or inverted")
    if start < 1:
        raise OLMoConfigurationError(
            f"span starts at {start}, but position 0 is predicted from nothing and has no loss. A "
            f"span that includes it is a sign the answer or value was rendered without a prefix."
        )
    if end > sequence_length:
        raise OLMoConfigurationError(
            f"span [{start}, {end}) runs past the sequence length {sequence_length}"
        )
    return start - 1, end - 1


def span_nats(ce_loss: "torch.Tensor", start: int, end: int) -> float:
    """
    Summed cross-entropy, in nats, over the tokens at positions ``[start, end)``.

    **Summed, never averaged.** Allen-Zhu's estimator is a total over the value tokens; averaging would
    make the result independent of how many facts the corpus holds, which is the quantity being swept.

    :param ce_loss: One sequence's per-token loss, shape ``(sequence_length,)``, as returned by a
        forward pass with ``loss_reduction="none"``.
    :param start: First token position to charge for.
    :param end: One past the last.

    :returns: The sum, in nats.
    """
    low, high = predictor_slice(start, end, ce_loss.shape[-1])
    return float(ce_loss[low:high].sum())


def span_bits(ce_loss: "torch.Tensor", start: int, end: int) -> float:
    """
    :func:`span_nats` converted to bits.

    :param ce_loss: One sequence's per-token loss.
    :param start: First token position to charge for.
    :param end: One past the last.

    :returns: The sum, in bits.
    """
    return span_nats(ce_loss, start, end) * BITS_PER_NAT


def predicted_token(logits: "torch.Tensor", position: int) -> int:
    """
    The greedy prediction for the token at ``position``, under teacher forcing.

    Reads ``logits[position - 1]`` for the same reason :func:`predictor_slice` subtracts one.

    Both reasoning endpoints render a **single-token** answer, so a teacher-forced argmax here is not an
    approximation of free generation -- it is identical to it, and there is no continuation to truncate
    and no string to parse. That is what removes the failure mode PRD 1 lists four times: an eval whose
    score is bounded by its parser rather than by the model.

    :param logits: One sequence's logits, shape ``(sequence_length, vocab_size)``.
    :param position: The token position being predicted.

    :returns: The argmax token id.

    :raises OLMoConfigurationError: If ``position`` is 0 or past the end.
    """
    low, _ = predictor_slice(position, position + 1, logits.shape[0])
    return int(logits[low].argmax())
