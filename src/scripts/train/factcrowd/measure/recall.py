"""
Fact recall by generation and by recognition.

Two questions that come apart, which is why both are asked. **Generation** is "what is this person's
birth city?" answered into the whole vocabulary. **Recognition** restricts the choice to the attribute's
own pool, so it asks whether the model can pick the right value out of the candidates even when it cannot
produce it unprompted. A model can be at chance on the first and well above it on the second, and only
the pair distinguishes "the fact is absent" from "the fact is there but not retrievable".

PRD 8.2 puts this in a post-hoc job rather than a training callback, and the reason is structural:
``TransformerGenerationModule.__init__`` re-parallelises the model and mutates KV-cache state, so there
is no safe way to generate from inside a trainer.

No KV cache is needed here regardless. Every biography is scored in **one teacher-forced forward pass**:
the prompt is the real rendered biography, so the value tokens sit at positions the renderer already
reports, and reading the argmax at those positions is what a greedy decode would have produced from the
same prefix. That holds exactly for a one-token value and position-by-position for a longer one.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from .spans import predicted_token


@dataclass(frozen=True)
class RecallResult:
    """
    Recall for one attribute, or pooled across attributes.

    :param attribute: Which attribute, or ``"all"`` when pooled.
    :param n_probed: Entity-attribute pairs asked about.
    :param n_generated: Times the model's unrestricted argmax was the right value.
    :param n_recognised: Times the right value outranked every other value **in its own pool**.
    :param pool_size: Candidates in the pool, so recognition has a stated chance level.
    """

    attribute: str
    n_probed: int
    n_generated: int
    n_recognised: int
    pool_size: int

    def __post_init__(self) -> None:
        if self.n_probed <= 0:
            raise OLMoConfigurationError(f"attribute '{self.attribute}': nothing was probed")
        for name, value in (("n_generated", self.n_generated), ("n_recognised", self.n_recognised)):
            if not 0 <= value <= self.n_probed:
                raise OLMoConfigurationError(
                    f"attribute '{self.attribute}': {name} is {value}, outside [0, {self.n_probed}]"
                )

    @property
    def generation(self) -> float:
        """Fraction produced unprompted."""
        return self.n_generated / self.n_probed

    @property
    def recognition(self) -> float:
        """Fraction picked correctly out of the pool."""
        return self.n_recognised / self.n_probed

    @property
    def recognition_chance(self) -> float:
        """Chance level for recognition: one over the pool size."""
        return 1.0 / self.pool_size if self.pool_size > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        """A flat mapping for collection."""
        return {
            f"recall_{self.attribute}_generation": round(self.generation, 6),
            f"recall_{self.attribute}_recognition": round(self.recognition, 6),
            f"recall_{self.attribute}_chance": round(self.recognition_chance, 6),
            f"recall_{self.attribute}_n": self.n_probed,
        }


def score_recall(
    loaded: Any,
    forward,
    *,
    n_entities: int = 1_000,
    batch_size: int = 32,
    exposure: int = 0,
) -> Tuple[RecallResult, ...]:
    """
    Probe generation and recognition over a prefix of the entity table.

    A prefix rather than a random sample, for the same reason the bit count uses one: the same entities
    are probed at every checkpoint and in every cell, so a difference is about the model.

    :param loaded: An opened checkpoint from :func:`factcrowd.measure.checkpoint.load`.
    :param forward: The ``(ce_loss, logits)`` callable from
        :func:`factcrowd.measure.checkpoint.forward_fn`.
    :param n_entities: Entities to probe, capped at the corpus size.
    :param batch_size: Biographies per forward pass.
    :param exposure: Which phrasing to render, fixed so the probe is not also sampling templates.

    :returns: One result per attribute, then a pooled ``"all"``. Empty for the reasoning-only control,
        which has no facts to recall.
    """
    corpus = loaded.corpus
    if corpus.renderer is None or corpus.table is None:
        return ()

    pool_ids = corpus.vocabulary.pool_token_ids
    spec_pools = {spec.name: spec.pool_names for spec in corpus.corpus_schema.values}
    probed: Dict[str, List[Tuple[bool, bool]]] = {name: [] for name in spec_pools}

    total = min(n_entities, loaded.resolved.n_entities)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        rendered = [corpus.renderer.render(entity, exposure) for entity in range(start, stop)]
        width = max(tokens.size for tokens, _ in rendered)
        batch = np.full((len(rendered), width), corpus.vocabulary.pad_id, dtype=np.int64)
        for row, (tokens, _) in enumerate(rendered):
            batch[row, : tokens.size] = tokens
        _, logits = forward(batch)

        for row, (tokens, spans) in enumerate(rendered):
            for span in spans:
                candidates = _candidate_ids(pool_ids, spec_pools.get(span.attribute, ()))
                outcome = _score_span(logits[row], tokens, span, candidates)
                probed.setdefault(span.attribute, []).append(outcome)

    results: List[RecallResult] = []
    pooled_gen = pooled_rec = pooled_n = 0
    for attribute, outcomes in probed.items():
        if not outcomes:
            continue
        generated = sum(1 for was_generated, _ in outcomes if was_generated)
        recognised = sum(1 for _, was_recognised in outcomes if was_recognised)
        pool = len(_candidate_ids(pool_ids, spec_pools.get(attribute, ())))
        results.append(
            RecallResult(
                attribute=attribute,
                n_probed=len(outcomes),
                n_generated=generated,
                n_recognised=recognised,
                pool_size=pool,
            )
        )
        pooled_gen += generated
        pooled_rec += recognised
        pooled_n += len(outcomes)

    if pooled_n:
        results.append(
            RecallResult(
                attribute="all",
                n_probed=pooled_n,
                n_generated=pooled_gen,
                n_recognised=pooled_rec,
                pool_size=int(np.mean([r.pool_size for r in results])) if results else 0,
            )
        )
    return tuple(results)


def _candidate_ids(pool_ids: Dict[str, np.ndarray], pool_names: Sequence[str]) -> np.ndarray:
    """Every token id an attribute's value could take, across the pools that compose it."""
    if not pool_names:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.asarray(pool_ids[name], dtype=np.int64) for name in pool_names])


def _score_span(
    logits: np.ndarray, tokens: np.ndarray, span: Any, candidates: np.ndarray
) -> Tuple[bool, bool]:
    """
    Whether one value was generated, and whether it was recognised.

    Generation is the unrestricted argmax at every position of the span. Recognition restricts the
    comparison to the attribute's own pool, which is what makes it a different question rather than an
    easier version of the same one.

    :returns: ``(generated, recognised)``.
    """
    generated = True
    recognised = True
    for position in range(span.start, span.end):
        truth = int(tokens[position])
        if predicted_token(logits, position) != truth:
            generated = False
        if candidates.size:
            row = logits[position - 1]
            best = candidates[int(np.argmax(row[candidates]))]
            if int(best) != truth:
                recognised = False
        else:
            recognised = False
    return generated, recognised


def pooled(results: Sequence[RecallResult]) -> Optional[RecallResult]:
    """
    The pooled ``"all"`` row, if present.

    :param results: What :func:`score_recall` returned.

    :returns: The pooled result, or ``None``.
    """
    for result in results:
        if result.attribute == "all":
            return result
    return None
