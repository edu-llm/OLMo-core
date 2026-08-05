"""
The one rule the whole measurement layer rests on: which loss position pays for which token.

OLMo-core builds labels as `pad(input_ids[..., 1:])`, so `ce_loss[t]` scores `input_ids[t+1]` and the
cost of the token at position `p` is `ce_loss[p-1]`. Off by one in either direction and nothing raises:
a bit count would charge a value token's cost to the literal before it, and an endpoint would grade the
token before the answer. Both would produce plausible numbers.

So the convention is checked against a *manually computed* cross-entropy from a real model rather than
against itself, and against OLMo-core's own `get_labels` rather than against a restatement of it.
"""

import numpy as np
import pytest
from factcrowd.measure import spans

from olmo_core.exceptions import OLMoConfigurationError


def test_a_token_is_paid_for_by_the_position_before_it():
    """One token, and the slice that scores it."""
    assert spans.predictor_slice(5, 6, 16) == (4, 5)
    assert spans.predictor_slice(1, 2, 16) == (0, 1)
    assert spans.predictor_slice(3, 7, 16) == (2, 6)


@pytest.mark.parametrize(
    "start,end,length,match",
    [
        (0, 3, 16, "predicted from nothing"),
        (5, 5, 16, "empty or inverted"),
        (7, 3, 16, "empty or inverted"),
        (10, 17, 16, "past the sequence length"),
    ],
)
def test_an_unmeasurable_span_is_refused(start, end, length, match):
    """
    Each of these would otherwise read outside the span or silently score nothing.

    Position 0 matters most: it is predicted from no context, so it has no loss. A span including it is a
    sign that an answer or value was rendered with no prefix, which is a corpus bug rather than a
    measurement one, and it should surface as a refusal rather than as a suspiciously low bit count.
    """
    with pytest.raises(OLMoConfigurationError, match=match):
        spans.predictor_slice(start, end, length)


def test_summing_is_summing_and_bits_are_nats_over_ln_two():
    """
    Summed, never averaged -- a mean over value tokens is independent of how many facts the corpus
    holds, which is the quantity being swept.
    """
    ce = np.arange(10, dtype=np.float64)  # ce[t] = t
    # Tokens 3..5 are paid for by positions 2..4, i.e. 2 + 3 + 4.
    assert spans.span_nats(ce, 3, 6) == pytest.approx(9.0)
    assert spans.span_bits(ce, 3, 6) == pytest.approx(9.0 / np.log(2))
    assert spans.BITS_PER_NAT == pytest.approx(1.4426950408889634)
    # A one-token span is one position, not zero and not two.
    assert spans.span_nats(ce, 4, 5) == pytest.approx(3.0)


def test_the_predicted_token_is_read_from_the_position_before_it():
    """A prediction for position p comes from logits[p-1], the same shift as the loss."""
    logits = np.zeros((6, 4))
    logits[2, 3] = 1.0  # position 2 predicts token 3 -> that is the prediction FOR position 3
    assert spans.predicted_token(logits, 3) == 3
    with pytest.raises(OLMoConfigurationError, match="predicted from nothing"):
        spans.predicted_token(logits, 0)


def test_the_convention_matches_olmo_cores_own_labels_and_a_manual_cross_entropy():
    """
    The check that makes the rest of this module trustworthy, run against a real model.

    `get_labels` is OLMo-core's, the forward pass is OLMo-core's, and the reference cross-entropy is
    computed here by hand from the logits. If `span_nats` used the wrong offset, this is the only test in
    the suite that could notice.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from factcrowd.ladder import sizes

    from olmo_core.data.utils import get_labels

    model = sizes.build(sizes.row("13M"), 128, init_seed=11).build(init_device="cpu")
    model.eval()
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(3))
    labels = get_labels({"input_ids": ids})
    # OLMo-core's own shift, restated nowhere: labels[t] is the token at t+1.
    assert labels[0, :5].tolist() == ids[0, 1:6].tolist()
    assert labels[0, -1].item() == -100

    with torch.no_grad():
        out = model(
            ids, labels=labels, ignore_index=-100, loss_reduction="none", return_logits=True
        )
    assert tuple(out.ce_loss.shape) == tuple(ids.shape)

    for position in (1, 4, 7, 11):
        manual = F.cross_entropy(
            out.logits[0, position - 1].float().unsqueeze(0),
            ids[0, position].view(1),
            reduction="none",
        )
        assert spans.span_nats(out.ce_loss[0], position, position + 1) == pytest.approx(
            manual.item(), rel=1e-4
        ), position

    # And a multi-token span is the sum of its single-token spans.
    total = sum(spans.span_nats(out.ce_loss[0], p, p + 1) for p in (3, 4, 5))
    assert spans.span_nats(out.ce_loss[0], 3, 6) == pytest.approx(total, rel=1e-5)
