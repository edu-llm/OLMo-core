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
    :param chance: Probability of getting every position right by guessing uniformly within each
        position's own reachable pool -- the product over positions of ``1 / active_size``, not one over
        the candidate count. Recognition requires the *whole* value, so a four-word value from four
        four-word pools is one chance in 256, not one in sixteen.
    """

    attribute: str
    n_probed: int
    n_generated: int
    n_recognised: int
    chance: float

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
        """The stated chance level, so "above chance" is a subtraction rather than an assumption."""
        return self.chance

    def summary(self) -> Dict[str, Any]:
        """A flat mapping for collection."""
        return {
            f"recall_{self.attribute}_generation": round(self.generation, 6),
            f"recall_{self.attribute}_recognition": round(self.recognition, 6),
            f"recall_{self.attribute}_chance": round(self.recognition_chance, 9),
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

    spec_pools = {spec.name: spec.pool_names for spec in corpus.corpus_schema.values}
    # Built once per attribute rather than per span: the pools do not change between biographies, and
    # slicing to the reachable prefix on every one of 25,000 entities is the kind of cost that turns a
    # minute of scoring into an hour.
    candidates_by_attribute = {
        name: _candidates_per_position(corpus, pools) for name, pools in spec_pools.items()
    }
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
                candidates = candidates_by_attribute.get(span.attribute, ())
                outcome = _score_span(logits[row], tokens, span, candidates)
                probed.setdefault(span.attribute, []).append(outcome)

    results: List[RecallResult] = []
    pooled_gen = pooled_rec = pooled_n = 0
    for attribute, outcomes in probed.items():
        if not outcomes:
            continue
        generated = sum(1 for was_generated, _ in outcomes if was_generated)
        recognised = sum(1 for _, was_recognised in outcomes if was_recognised)
        results.append(
            RecallResult(
                attribute=attribute,
                n_probed=len(outcomes),
                n_generated=generated,
                n_recognised=recognised,
                chance=_chance_of(candidates_by_attribute.get(attribute, ())),
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
                # The mean of the per-attribute chances, not one over the mean pool size. Those differ
                # -- mean(1/n) against 1/mean(n) -- and the second understated the pooled chance by 3.8x
                # on bioS, which reported an untrained model as 4.15x above chance when it was at it.
                chance=float(np.mean([result.chance for result in results])) if results else 0.0,
            )
        )
    return tuple(results)


def _chance_of(candidates: Sequence[np.ndarray]) -> float:
    """
    Probability of getting a whole value right by guessing within each position's reachable pool.

    The product over positions, because recognition requires every one of them. A four-word value drawn
    from four four-word pools is one chance in 256, not one in sixteen.

    :param candidates: One id array per position.

    :returns: The chance level, or 0.0 when there is nothing to guess among.
    """
    if not candidates or any(array.size == 0 for array in candidates):
        return 0.0
    chance = 1.0
    for array in candidates:
        chance /= float(array.size)
    return chance


def _candidates_per_position(corpus: Any, pool_names: Sequence[str]) -> Tuple[np.ndarray, ...]:
    """
    The ids each position of a value may legally take, one array per position.

    **Per position, and reachable only.** Two mistakes are easy here and both inflate the apparent
    difficulty of recognition:

    - Concatenating an attribute's pools and taking one argmax over the union lets position 1 be beaten by
      a word only pool 2 contains. That is not a wrong answer, it is a question nobody asked -- the
      corpus never puts pool 2's words in position 1.
    - On the entropy axis a pool holds the sweep's union of 256 words while only ``2**(b/4)`` are ever
      assigned. The rest are never trained and their embeddings are still at init, so letting them compete
      measures initialisation noise.

    :param corpus: The rebuilt corpus, for its schema and vocabulary.
    :param pool_names: The pools composing one attribute, in position order.

    :returns: One id array per position.
    """
    pools = {pool.name: pool for pool in corpus.corpus_schema.schema.attributes}
    ids = corpus.vocabulary.pool_token_ids
    out = []
    for name in pool_names:
        reachable = pools[name].active_size
        out.append(np.asarray(ids[name], dtype=np.int64)[:reachable])
    return tuple(out)


def _score_span(
    logits: np.ndarray,
    tokens: np.ndarray,
    span: Any,
    candidates: Sequence[np.ndarray],
) -> Tuple[bool, bool]:
    """
    Whether one value was generated, and whether it was recognised.

    Generation is the unrestricted argmax at every position. Recognition restricts each position to *its
    own* reachable pool, which is what makes it a different question rather than an easier version of the
    same one. Both require every position, because half a value is not the value.

    :param logits: One sequence's logits.
    :param tokens: The rendered biography, for the truth.
    :param span: Where the value sits.
    :param candidates: One id array per position of the span.

    :returns: ``(generated, recognised)``.
    """
    generated = True
    recognised = bool(candidates)
    for offset, position in enumerate(range(span.start, span.end)):
        truth = int(tokens[position])
        if predicted_token(logits, position) != truth:
            generated = False
        if offset < len(candidates) and candidates[offset].size:
            allowed = candidates[offset]
            best = int(allowed[int(np.argmax(logits[position - 1][allowed]))])
            if best != truth:
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
