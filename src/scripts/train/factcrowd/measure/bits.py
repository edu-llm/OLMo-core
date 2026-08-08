"""
Achieved fact bits: Allen-Zhu's estimator over the value spans the renderer already returns.

The x-axis of the experiment is *demanded* bits per parameter, which is arithmetic (``ladder/rho.py``).
This module measures what the model actually **stored**, which is the mediator, and the two are reported
against each other.

The method is Physics 3.3's, and PRD 8.1 states the two things that make it right: **sum, never average**
-- a mean over value tokens is independent of how many facts the corpus holds, which is the swept
quantity -- and convert nats to bits with ``1/ln2``.

Two honesty constraints are enforced here rather than left to the reader:

- **Achieved bits cannot exceed the capacity bound.** ``R <= R_max`` is asserted, and a violation means
  the estimator is measuring something other than storage -- most likely context leakage.
- **With intra-document masking off, this is an upper bound** (PRD 7.3). A biography packed after another
  can attend to it, so some of what looks stored was read from context. :attr:`AchievedBits.is_upper_bound`
  carries that forward into the collected row instead of leaving it in a commit message.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..ladder import rho
from .spans import span_bits


@dataclass(frozen=True)
class AchievedBits:
    """
    What one checkpoint stored, measured over a sample of its own fact corpus.

    :param n_entities_sampled: Entities the estimate is built from.
    :param n_entities_total: Entities in the cell's corpus, which the estimate scales to.
    :param value_bits_sampled: Summed CE over the sampled entities' value tokens, in bits. This is the
        *residual* -- what the model still has to be told -- so stored bits are the difference from the
        prior, not this number.
    :param prior_bits_per_entity: Bits an entity's values carry with no model at all, i.e. the schema's
        own ``sum(log2(pool))``. The reference the residual is subtracted from.
    :param non_embedding_params: Denominator for the per-parameter figure.
    :param is_upper_bound: True while intra-document masking is off.
    :param per_entity_bits: **Signed, unclamped** ``prior - residual_i`` for each sampled entity, for the
        distribution PRD 8.1 asks to be logged. A mean hides a corpus where a few entities are memorised
        and the rest are not -- but these are a diagnostic, not a second estimate of the headline. Their
        mean is upward biased against it, so they are deliberately left signed: a negative value is
        visible as "worse than uniform about this entity" rather than silently floored.
    """

    n_entities_sampled: int
    n_entities_total: int
    value_bits_sampled: float
    prior_bits_per_entity: float
    non_embedding_params: int
    is_upper_bound: bool
    per_entity_bits: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.n_entities_sampled <= 0:
            raise OLMoConfigurationError("no entities were sampled")
        if self.n_entities_total < self.n_entities_sampled:
            raise OLMoConfigurationError(
                f"sampled {self.n_entities_sampled:,} entities out of {self.n_entities_total:,}"
            )
        if self.prior_bits_per_entity < 0 or self.non_embedding_params <= 0:
            raise OLMoConfigurationError("prior bits and parameter count must be positive")

    @property
    def residual_bits_per_entity(self) -> float:
        """Mean bits still needed to specify one entity's values after seeing the model."""
        return self.value_bits_sampled / self.n_entities_sampled

    @property
    def stored_bits_per_entity(self) -> float:
        """
        The headline: ``max(0, prior - mean(residual))``, clamped **once, in aggregate**.

        This is Allen-Zhu's estimator, which subtracts an aggregate expected loss. Clamping per entity and
        then averaging is a different quantity and is **upward biased by Jensen's inequality** -- every
        entity the model is worse than uniform about contributes 0 instead of its true negative, so the
        mean of clamps exceeds the clamp of the mean whenever any residual exceeds the prior. That is the
        norm early in training, where a real checkpoint measured 88-90 bits of residual against a
        47.59-bit prior, and a case exists where the per-entity form reports 2 stored bits while the
        aggregate is 0.

        An earlier revision used the per-entity form to make this agree with
        :attr:`signed_bits_per_entity`. Agreement was the right instinct and the wrong direction: the fix
        is that the distribution is now published **signed and unclamped**, so it cannot be mistaken for
        a second estimate of the same number.
        """
        return max(0.0, self.prior_bits_per_entity - self.residual_bits_per_entity)

    @property
    def stored_bits_total(self) -> float:
        """Stored bits across the whole corpus, scaled from the sample."""
        return self.stored_bits_per_entity * self.n_entities_total

    @property
    def achieved_per_param(self) -> float:
        """Stored bits per non-embedding parameter -- the achieved R(F)."""
        return self.stored_bits_total / self.non_embedding_params

    def check_against_demand(self, demanded_bits: float) -> None:
        """
        Assert the achieved figure sits under the only bound that is a theorem.

        A model cannot supply more information about a corpus than the corpus contains, so
        ``achieved <= demanded`` is a hard fact about the dataset. Exceeding it means the estimator is
        crediting the model for something it did not store -- most likely a biography attending to its
        neighbour, because intra-document masking is off (PRD 7.3).

        :param demanded_bits: Total demanded bits for this cell, from
            :func:`factcrowd.ladder.rho.demanded_bits`.

        :raises OLMoConfigurationError: If achieved exceeds demanded.
        """
        if demanded_bits > 0 and self.stored_bits_total > demanded_bits * (1.0 + 1e-6):
            raise OLMoConfigurationError(
                f"achieved {self.stored_bits_total:,.0f} bits exceeds the {demanded_bits:,.0f} the "
                f"corpus contains. A model cannot supply more information than its data holds, so this "
                f"is a measurement fault: check span alignment (measure.spans) and that packed documents "
                f"are masked from each other."
            )

    def capacity_warning(self, r_max: float = rho.R_E_MAX) -> Optional[str]:
        """
        A note when the achieved figure passes the published capacity estimate. **Not an error.**

        An earlier revision raised here, which was wrong twice over. Physics 3.3's ~2 bits/parameter is an
        empirical measurement at 1,000 exposures, not a theorem -- and three of the six entropy cells
        *demand* 2.2 to 4.2 bits/param, so a model that stored what it was given would have been refused
        by the instrument built to measure it. That censors the quantity M0 exists to discover.

        :param r_max: The published estimate, in bits per parameter.

        :returns: A warning string, or ``None``.
        """
        if self.achieved_per_param > r_max:
            return (
                f"achieved {self.achieved_per_param:.3f} bits/param is above the published ~{r_max} "
                f"estimate. Worth checking span alignment and document masking, but not impossible: "
                f"this cell may simply demand more than that, and the estimate is empirical."
            )
        return None

    def summary(self) -> Dict[str, object]:
        """A flat mapping for logging and collection."""
        distribution: Dict[str, object] = {}
        if self.per_entity_bits:
            values = np.asarray(self.per_entity_bits, dtype=np.float64)
            distribution = {
                "signed_bits_p10": round(float(np.percentile(values, 10)), 4),
                "signed_bits_median": round(float(np.median(values)), 4),
                "signed_bits_p90": round(float(np.percentile(values, 90)), 4),
                "signed_bits_sd": round(float(values.std(ddof=1)) if values.size > 1 else 0.0, 4),
                "signed_bits_fraction_positive": round(float((values > 0).mean()), 4),
            }
        return {
            "n_entities_sampled": self.n_entities_sampled,
            "n_entities_total": self.n_entities_total,
            "prior_bits_per_entity": round(self.prior_bits_per_entity, 4),
            "residual_bits_per_entity": round(self.residual_bits_per_entity, 4),
            "stored_bits_per_entity": round(self.stored_bits_per_entity, 4),
            "stored_bits_total": round(self.stored_bits_total, 1),
            "achieved_bits_per_param": round(self.achieved_per_param, 6),
            "bits_is_upper_bound": self.is_upper_bound,
            **distribution,
        }


def achieved_bits(
    value_bits_per_entity: Sequence[float],
    *,
    n_entities_total: int,
    prior_bits_per_entity: float,
    non_embedding_params: int,
    is_upper_bound: bool = True,
) -> AchievedBits:
    """
    Assemble an :class:`AchievedBits` from per-entity residuals.

    :param value_bits_per_entity: Summed CE over each sampled entity's value tokens, in bits.
    :param n_entities_total: Entities in the cell's corpus.
    :param prior_bits_per_entity: The schema's ``sum(log2(pool))``.
    :param non_embedding_params: Denominator for the per-parameter figure.
    :param is_upper_bound: Whether packed documents can see each other.

    :returns: The measurement.
    """
    residuals = list(value_bits_per_entity)
    if not residuals:
        raise OLMoConfigurationError("no per-entity value bits were supplied")
    # Signed and unclamped: see AchievedBits.per_entity_bits. Clamping here and averaging is the biased
    # estimator, and having both a clamped distribution and a clamped mean is what let two disagreeing
    # numbers into one summary.
    signed = [prior_bits_per_entity - residual for residual in residuals]
    return AchievedBits(
        n_entities_sampled=len(residuals),
        n_entities_total=n_entities_total,
        value_bits_sampled=float(sum(residuals)),
        prior_bits_per_entity=prior_bits_per_entity,
        non_embedding_params=non_embedding_params,
        is_upper_bound=is_upper_bound,
        per_entity_bits=tuple(signed),
    )


def value_bits_of_batch(ce_loss, spans_per_row: Iterable[Sequence[Tuple[int, int]]]) -> List[float]:
    """
    Summed CE over each row's value spans, in bits.

    The spans come from :meth:`factcrowd.corpus.render.Renderer.render_run`, which already returns where
    every value landed -- so nothing here re-derives token positions, and the estimator cannot disagree
    with the renderer about which tokens are the facts.

    :param ce_loss: Per-token loss, shape ``(batch, sequence)``.
    :param spans_per_row: For each row, the ``(start, end)`` spans of its value tokens.

    :returns: One summed figure per row, in bits.
    """
    out: List[float] = []
    for row, spans in enumerate(spans_per_row):
        out.append(sum(span_bits(ce_loss[row], start, end) for start, end in spans))
    return out


def demanded_vs_achieved(
    achieved: AchievedBits,
    *,
    n_entities: int,
    bits_per_entity: float,
    name_space: Optional[int],
) -> Dict[str, float]:
    """
    Put the achieved figure beside the demanded one, on the same definition.

    Reuses :func:`factcrowd.ladder.rho.demanded_bits` rather than restating the formula, so the two
    halves of the comparison cannot drift -- which they did once already, when two tables in the PRD
    quoted the axis on different definitions of the name term.

    :param achieved: The measurement.
    :param entity_offset: First entity to sample. Non-zero avoids the ``<compare>`` probe window.
    :param n_entities: Entities in the corpus.
    :param bits_per_entity: The schema's attribute bits.
    :param name_space: Name universe, or ``None`` to drop the name term.

    :returns: Demanded, achieved and their ratio, all per non-embedding parameter.
    """
    demanded = rho.demanded_bits(n_entities, bits_per_entity, name_space=name_space)
    demanded_per_param = demanded / achieved.non_embedding_params
    return {
        "demanded_bits_per_param": demanded_per_param,
        "achieved_bits_per_param": achieved.achieved_per_param,
        "achieved_over_demanded": (
            achieved.achieved_per_param / demanded_per_param if demanded_per_param > 0 else math.nan
        ),
    }


def score_checkpoint(
    loaded,
    forward,
    *,
    n_entities: int = 2_000,
    entity_offset: int = 0,
    batch_size: int = 32,
    exposure: int = 0,
    is_upper_bound: bool = True,
) -> Optional[AchievedBits]:
    """
    Measure what one checkpoint stored, over a sample of its own entities.

    Renders each sampled entity's biography with
    :meth:`~factcrowd.corpus.render.Renderer.render`, which returns the value spans alongside the
    tokens, and charges only those spans. Nothing here re-derives a token position, so the estimator
    cannot disagree with the renderer about which tokens are the facts.

    The sample is a **prefix** of the entity table, so the same entities are measured at every
    checkpoint and every cell -- a bit curve over steps is then a curve about the model rather than
    about which entities got sampled. The table is generated from the cell's seed, so a prefix is a
    fixed set.

    :param loaded: An opened checkpoint, from :func:`factcrowd.measure.checkpoint.load`.
    :param forward: The ``(ce_loss, logits)`` callable from
        :func:`factcrowd.measure.checkpoint.forward_fn`.
    :param n_entities: How many entities to sample. Capped at the corpus size.
    :param batch_size: Biographies per forward pass.
    :param exposure: Which exposure's phrasing to render. Fixed at 0 so the measurement is not also
        sampling over templates.
    :param is_upper_bound: Whether packed documents can see each other during training. True while
        intra-document masking is off (PRD 7.3), and carried into the result rather than dropped.

    :returns: The measurement, or ``None`` for the reasoning-only control, which has no facts to store.
    """
    corpus = loaded.corpus
    if corpus.renderer is None or corpus.table is None:
        return None  # the control: no entities, so nothing to have stored

    total = loaded.resolved.n_entities
    # THE DEFAULT COHORT IS THE ONE <compare> SUPERVISES, WHICH IS WHY THIS IS A PARAMETER.
    # `CompareTask` draws its pairs from entities 0..24,999 and its answer is the earlier *birth year*,
    # so every entity in that window receives ~211 direct year constraints. Measured on the first count
    # grid, birth_year then reconstructs at 328x chance in a cell where the best other attribute is
    # 1.1x -- so a prefix sample beginning at 0 measures Compare's supervision as much as biography
    # storage. An offset above the probe window gives an uncontaminated cohort from the same checkpoint.
    first = max(0, entity_offset)
    sampled = max(0, min(n_entities, total - first))
    prior = corpus.corpus_schema.schema.bits_per_entity

    residuals: List[float] = []
    for start in range(0, sampled, batch_size):
        stop = min(start + batch_size, sampled)
        rendered = [
            corpus.renderer.render(entity, exposure)
            for entity in range(first + start, first + stop)
        ]
        width = max(tokens.size for tokens, _ in rendered)
        # Right-padded with the pad id and charged only over the value spans, so the padding never
        # enters the sum. Ragged biographies are the norm on the count axis -- the templates span
        # 21 to 152 tokens by design.
        batch = np.full((len(rendered), width), corpus.vocabulary.pad_id, dtype=np.int64)
        for row, (tokens, _) in enumerate(rendered):
            batch[row, : tokens.size] = tokens
        ce_loss, _ = forward(batch)
        residuals.extend(
            value_bits_of_batch(
                ce_loss, [[(s.start, s.end) for s in spans] for _, spans in rendered]
            )
        )

    return achieved_bits(
        residuals,
        n_entities_total=total,
        prior_bits_per_entity=prior,
        non_embedding_params=loaded.cell.non_embedding_params,
        is_upper_bound=is_upper_bound,
    )
