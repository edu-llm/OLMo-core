"""
The shape every endpoint reports in.

Four numbers and a floor, because that is what it takes to read a reasoning score honestly. This
programme has produced four uninterpretable nulls (PRD 1) and every one of them would have been caught
by reporting these together rather than a bare accuracy:

- **three counts**, so a score can be checked against its own denominator. An eval that silently drops
  items reports a fraction of a number nobody sees.
- **the measured floor**, so "above chance" is a subtraction rather than an assumption. One previous
  eval scored *below* its own floor and reported the number anyway.
- **answer-token CE in bits**, a continuous quantity that moves before accuracy does. A 2pp effect on a
  five-point grid needs every bit of resolution available -- though it is not commensurate with stored
  information just because both are in bits (PRD 16.5).
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from olmo_core.exceptions import OLMoConfigurationError


@dataclass(frozen=True)
class EndpointResult:
    """
    One endpoint's score at one checkpoint.

    :param name: Slice name, e.g. ``"mano"``.
    :param n_total: Items scored.
    :param n_correct: Items whose predicted answer span matched exactly.
    :param n_modal: Items where the prediction equalled this model's own most common prediction. Catches
        collapse to *any* constant, which ``n_degenerate`` cannot: it matches one pre-measured answer, so a
        model that settled on a different constant reports zero degeneracy while being entirely constant.
    :param n_degenerate: Items where the prediction matched the endpoint's best fact-free policy. Not a
        subset of the incorrect ones -- a degenerate answer is sometimes right, which is the whole point
        of measuring the floor.
    :param n_unparseable: Items whose answer could not be read. Structurally zero here, because both
        endpoints render a fixed-width single-token answer at a known position; kept because PRD 8.6's
        G7 requires it under 5% and a future multi-token endpoint could break that.
    :param answer_ce_bits: Mean cross-entropy of the answer tokens, in bits.
    :param floor: The measured degenerate baseline, as a fraction. From
        :meth:`factcrowd.corpus.tasks.ReasoningTask.degenerate_baseline`, never assumed.
    """

    name: str
    n_total: int
    n_correct: int
    n_degenerate: int
    n_unparseable: int
    answer_ce_bits: float
    floor: float
    #: Defaulted so records written before this field existed still load.
    n_modal: int = 0

    def __post_init__(self) -> None:
        if self.n_total <= 0:
            raise OLMoConfigurationError(f"endpoint '{self.name}' scored {self.n_total} items")
        for field, value in (
            ("n_correct", self.n_correct),
            ("n_degenerate", self.n_degenerate),
            ("n_modal", self.n_modal),
            ("n_unparseable", self.n_unparseable),
        ):
            if not 0 <= value <= self.n_total:
                raise OLMoConfigurationError(
                    f"endpoint '{self.name}': {field} is {value}, outside [0, {self.n_total}]"
                )
        if not 0.0 <= self.floor <= 1.0:
            raise OLMoConfigurationError(
                f"endpoint '{self.name}': floor {self.floor} is not a fraction"
            )

    @property
    def accuracy(self) -> float:
        """Fraction of items answered exactly."""
        return self.n_correct / self.n_total

    @property
    def modal_rate(self) -> float:
        """
        Share of items answered with this model's own most common prediction.

        1.0 means the model is a constant function, whatever that constant is. Read this before accuracy:
        a collapsed model still scores the frequency of its chosen answer, which on a 23-way task is a few
        percent and looks like a floor rather than like a failure to answer at all.
        """
        return 0.0 if self.n_total == 0 else self.n_modal / self.n_total

    @property
    def degenerate_rate(self) -> float:
        """Fraction of items where the model emitted the best fact-free answer."""
        return self.n_degenerate / self.n_total

    @property
    def unparseable_rate(self) -> float:
        """Fraction of items whose answer could not be read."""
        return self.n_unparseable / self.n_total

    @property
    def above_floor(self) -> float:
        """
        Accuracy minus the measured floor, in percentage points.

        **Negative is a defect, not a result.** A score below its own floor means the endpoint is
        measuring its own instrumentation, which is how a previous deduction eval in this programme came
        to report a number under 0.500 on a two-way task.
        """
        return 100.0 * (self.accuracy - self.floor)

    @property
    def headroom(self) -> float:
        """
        Percentage points between the floor and a perfect score -- the range an effect could occupy.

        PRD 8.6's G4 wants an effect measured against this rather than against a nominal 0-100.
        """
        return 100.0 * (1.0 - self.floor)

    def summary(self) -> Dict[str, object]:
        """
        A flat mapping for logging and for :mod:`factcrowd.measure.collect`.

        :returns: Every field and every derived quantity, so a collected row needs no recomputation.
        """
        return {
            "endpoint": self.name,
            "n_total": self.n_total,
            "n_correct": self.n_correct,
            "n_degenerate": self.n_degenerate,
            "n_modal": self.n_modal,
            "modal_rate": round(self.modal_rate, 6),
            "n_unparseable": self.n_unparseable,
            "accuracy": round(self.accuracy, 6),
            "degenerate_rate": round(self.degenerate_rate, 6),
            "unparseable_rate": round(self.unparseable_rate, 6),
            "answer_ce_bits": round(self.answer_ce_bits, 6),
            "floor": round(self.floor, 6),
            "above_floor_pp": round(self.above_floor, 4),
            "headroom_pp": round(self.headroom, 4),
        }


class EndpointAccumulator:
    """
    Counts items as a scorer walks them, then produces an :class:`EndpointResult`.

    A separate object rather than a running tuple so the scoring loop stays a loop over items and the
    bookkeeping cannot drift between endpoints.

    :param name: Slice name.
    :param floor: The measured degenerate baseline.
    :param degenerate_answer: The best fact-free answer, for counting how often the model emits it.
    """

    def __init__(
        self, name: str, *, floor: float, degenerate_answer: Optional[Tuple[str, ...]] = None
    ) -> None:
        self.name = name
        self.floor = floor
        self.degenerate_answer = degenerate_answer
        self._total = 0
        self._correct = 0
        self._degenerate = 0
        self._unparseable = 0
        self._ce_bits = 0.0
        # COLLAPSE TO *ANY* CONSTANT, NOT JUST THE BEST ONE. `n_degenerate` matches one pre-measured
        # answer -- the best fact-free policy -- so a model that collapses to a different constant reads
        # as perfectly non-degenerate. Observed across the first grid: eleven cells scored exactly
        # 1,342/30,000, which is the number of eval items answered `<n0>`, and one scored 1,339, the count
        # for `<n12>`. Every one of them reported degenerate_rate 0.0. The single cell that *was* flagged
        # at 99.86% is the one that happened to pick `<n16>`, the answer the detector checks.
        self._predictions: Counter = Counter()

    def add(
        self,
        *,
        predicted: Tuple[str, ...],
        expected: Tuple[str, ...],
        ce_bits: float,
        parseable: bool = True,
    ) -> None:
        """
        Record one item.

        :param predicted: The model's answer as words.
        :param expected: The correct answer as words.
        :param ce_bits: Cross-entropy of the answer tokens, in bits.
        :param parseable: Whether the answer could be read at all.
        """
        self._total += 1
        self._ce_bits += ce_bits
        if not parseable:
            self._unparseable += 1
            return
        if predicted == expected:
            self._correct += 1
        if self.degenerate_answer is not None and predicted == self.degenerate_answer:
            self._degenerate += 1
        self._predictions[predicted] += 1

    def result(self) -> EndpointResult:
        """
        Freeze the counts.

        :returns: The endpoint's score.

        :raises OLMoConfigurationError: If nothing was recorded.
        """
        return EndpointResult(
            name=self.name,
            n_total=self._total,
            n_correct=self._correct,
            n_degenerate=self._degenerate,
            n_modal=(max(self._predictions.values()) if self._predictions else 0),
            n_unparseable=self._unparseable,
            answer_ce_bits=self._ce_bits / max(1, self._total),
            floor=self.floor,
        )
