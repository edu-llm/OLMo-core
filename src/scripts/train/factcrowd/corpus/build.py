"""
Build the corpus a cell implies -- once, for both training and measurement.

``train_cell.py`` needs it to feed the trainer; :mod:`factcrowd.measure` needs it to rebuild the exact
corpus a checkpoint was trained on before scoring it. Those two must agree token for token or a
measurement is describing a different experiment from the run, so there is one implementation and both
import it.

The only thing that differs between the two callers is the reasoning **split**: training generates
``"train"`` items, measurement generates the disjoint ``"eval"`` ones. That is a constructor argument,
and the split reaches the item key, so the two can never overlap (PRD 16.4).
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from olmo_core.exceptions import OLMoConfigurationError

from .. import cells as cell_module
from . import render as render_module
from . import stream as stream_module
from . import tasks as tasks_module
from . import values as values_module
from . import vocab as vocab_module

DOMAIN_TOKENS: Tuple[str, ...] = ("<facts>", "<mano>", "<compare>")
"""One per corpus slice. Prepended to every segment; there is no flag to disable it."""

MANO_LENGTH = 10
"""
Expression length for Mano. Ten, not thirteen.

At thirteen the task sits about a point above its own degenerate policy at 13M-28M, which fails
the 20-80% admission band it is meant to pass. At ten Physics 4.1 reports 47.8 to 66.0 from
scratch at our exact twelve layers, moving 18.2 points across the parameter range.
"""

TASK_WORDS: Tuple[str, ...] = tasks_module.all_required_words(
    (tasks_module.ManoTask, tasks_module.CompareTask)
)
"""
Words the reasoning tasks need, taken off the classes before anything is built.

The ordering is forced: a task cannot be constructed without a vocabulary containing its tokens,
and the vocabulary cannot be built without knowing which tokens to reserve -- and the pool
allocator has to avoid them too, or a generated city could collide with an operator.
"""


class BuiltCorpus:
    """
    The corpus objects a cell implies, built once and shared by the dry run and the real run.

    :param resolved: The resolved cell.
    :param work_dir: Local scratch for the entity table and the offset index.
    :param split: Which reasoning split to generate, ``"train"`` or ``"eval"``. Measurement passes
        ``"eval"``; the two sets are disjoint by construction.
    :param with_streams: Build the packed token volumes -- the fact
        :class:`~factcrowd.corpus.stream.BioStream` with its token-offset index, and the
        :class:`~factcrowd.corpus.tasks.TaskStream` volumes. Training needs them; measurement does not.

        The switch used to cover only the task streams, so ``with_streams=False`` still built an offset
        index over billions of fact tokens -- measured at 4.7s against 0.3s on a 127k-entity cell, paid
        on every checkpoint scored. Measurement reads :attr:`tasks` and :attr:`renderer`, which are built
        either way: the renderer renders a biography and reports its value spans, which is all that bits
        and template reconstruction need.
    """

    def __init__(
        self,
        resolved: cell_module.ResolvedCell,
        work_dir: Path,
        *,
        split: str = "train",
        with_streams: bool = True,
    ) -> None:
        from . import entities as entities_module

        spec = resolved.spec
        if spec.sweep == "count":
            self.templates = render_module.BIOS_TEMPLATES
            literals = render_module.literal_words_of(self.templates)
            self.corpus_schema = values_module.bios_schema(
                reserved=tuple(literals) + TASK_WORDS + vocab_module.SPECIALS + DOMAIN_TOKENS
            )
        else:
            assert spec.bits_per_attribute is not None
            self.templates = render_module.entropy_templates(
                values_module.ENTROPY_ATTRIBUTES, values_module.ENTROPY_WORDS_PER_VALUE
            )
            literals = render_module.literal_words_of(self.templates)
            self.corpus_schema = values_module.entropy_schema(
                spec.bits_per_attribute,
                reserved=tuple(literals) + TASK_WORDS + vocab_module.SPECIALS + DOMAIN_TOKENS,
            )

        self.vocabulary = vocab_module.Vocabulary.build(
            self.corpus_schema.schema,
            literal_words=tuple(literals) + TASK_WORDS,
            domain_tokens=DOMAIN_TOKENS,
        )
        # The reasoning-only control has no entities, so it has no table, no renderer and no fact
        # stream. It keeps the schema and the vocabulary above: the model must be architecturally
        # identical to the row's other cells, and dropping ~1,700 unused word embeddings would change
        # the parameter count of the one cell the others are compared against.
        self.table: Optional[Any] = None
        self.renderer: Optional[Any] = None
        self.stream: Optional[Any] = None
        self.spec_cell_id = spec.cell_id
        self.split = split
        if not spec.is_control:
            table_entities = spec.table_entities or resolved.n_entities
            self.table = entities_module.EntityTable.build(
                self.corpus_schema.schema, table_entities, spec.seed
            )
            self.renderer = render_module.Renderer(
                self.table,
                self.corpus_schema,
                self.vocabulary,
                self.templates,
                domain_token=DOMAIN_TOKENS[0],
                seed=spec.seed + 1,
                min_templates=1 if spec.sweep == "entropy" else 20,
            )
        if self.renderer is not None and with_streams:
            self.stream = stream_module.BioStream(
                self.renderer,
                n_entities=resolved.n_entities,
                exposures=spec.exposures,
                work_dir=work_dir,
                seed=spec.seed + 2,
            )

        # The reasoning slices. Each is stated in absolute tokens and is identical in every cell that
        # carries it, which is the invariant that stops reasoning volume moving with fact load -- hold
        # the ratio instead and the result is confounded in both directions.
        #
        # Per slice, not split between them. Splitting a fixed total was the first thing written here
        # and it silently broke the control: the control carries no related slice, so an evenly split
        # total gave it 2x the Mano exposure of every cell it is the reference for, and its Mano score
        # would have been higher for that reason alone. A cell's reasoning total therefore varies with
        # how many slices it carries, and that is correct -- what an endpoint's score must be compared
        # at is exposure to *that endpoint's* data.
        self.task_streams: Tuple[tasks_module.TaskStream, ...] = ()
        self.tasks: Tuple[tasks_module.ReasoningTask, ...] = ()
        if spec.reasoning_tokens > 0:
            built: list = [
                tasks_module.ManoTask(
                    self.vocabulary,
                    domain_token="<mano>",
                    length=MANO_LENGTH,
                    seed=spec.seed + 4,
                    split=split,
                )
            ]
            # The related-reasoning slice asks about the fact schema, so it exists only where that
            # schema has fields to ask about. The entropy axis's attributes are abstract by
            # construction -- six positional attributes of four words each, with no ordinal field to
            # compare -- so that axis carries unrelated reasoning alone. It costs nothing: the entropy
            # axis's job is the identified demand sweep, and the related-reasoning prediction lives on
            # the count axis where bioS does have a birth year.
            if (
                spec.sweep == "count"
                and self.table is not None
                and spec.related_reasoning_tokens > 0
            ):
                built.append(
                    tasks_module.CompareTask(
                        self.table,
                        self.corpus_schema,
                        self.vocabulary,
                        domain_token="<compare>",
                        probe_ids=self.table.probe_ids,
                        seed=spec.seed + 5,
                        split=split,
                    )
                )
            self.tasks = tuple(built)
            if with_streams:
                self.task_streams = tuple(
                    tasks_module.TaskStream(
                        task, num_tokens=spec.slice_budget(task.name), label=task.name
                    )
                    for task in built
                )
        # The cell's own arithmetic declares which slices it carries, and every token estimate and cost
        # figure is computed from that. Asserting the built set against it means a divergence fails here
        # rather than showing up as a run whose step count nobody can reproduce.
        names = tuple(task.name for task in self.tasks)
        if names != spec.reasoning_slice_names:
            raise OLMoConfigurationError(
                f"cell '{spec.cell_id}' built reasoning slices {names} but its arithmetic declares "
                f"{spec.reasoning_slice_names}. One of the two selection rules has drifted, and the "
                f"cell's token count is computed from the declaration."
            )

    @property
    def fact_tokens(self) -> int:
        """Measured tokens in the fact slice; zero on the reasoning-only control."""
        return 0 if self.stream is None else int(self.stream.num_tokens)

    def summary(self, resolved: cell_module.ResolvedCell) -> Dict[str, Any]:
        """
        Everything worth printing before a run, with the *measured* token count rather than an estimate.

        :param resolved: The resolved cell.

        :returns: A flat mapping.
        """
        out = resolved.summary(0.0 if self.renderer is None else self.renderer.mean_tokens_per_bio)
        out.update(
            {
                "vocab_size": self.vocabulary.size,
                "vocab_size_padded": self.vocabulary.padded_size(),
                "schema_fingerprint": self.corpus_schema.schema.fingerprint()[:16],
                "reasoning_slices": ", ".join(
                    f"{s.task.name}:{s.n_items:,} items/{s.num_tokens:,} tokens"
                    for s in self.task_streams
                )
                or "none",
            }
        )
        if self.renderer is not None and self.stream is not None and self.table is not None:
            out.update(
                {
                    "templates": self.renderer.n_templates,
                    "tokens_per_bio_min": int(self.renderer.template_lengths.min()),
                    "tokens_per_bio_max": self.renderer.max_tokens_per_bio,
                    "bits_per_token": round(self.renderer.bits_per_token, 4),
                    "fact_tokens_measured": self.stream.num_tokens,
                    "table_entities": self.table.n_entities,
                    "stream_fingerprint": self.stream.fingerprint()[:16],
                }
            )
        else:
            out["fact_tokens_measured"] = 0
        # The measured count supersedes the estimate; keep both so a drift is visible.
        out["total_tokens"] = self.fact_tokens + sum(
            stream.num_tokens for stream in self.task_streams
        )
        out["steps"] = out["total_tokens"] // resolved.spec.global_batch_size
        return out
