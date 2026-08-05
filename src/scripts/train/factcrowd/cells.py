"""
A cell: one point of the grid, and everything derivable from it.

One YAML file is one cell, and this module is what a YAML deserialises to. It imports no ``torch``, so
the grid's arithmetic -- which entity count a demand implies, how many tokens that is, how many steps
that is -- is checkable anywhere, including in the ``--dry-run`` that precedes a submission.

**Exactly one of two quantities is stated, never both.** On the count axis a cell states its demand
and ``n_entities`` is derived; on the entropy axis it states ``n_entities`` and ``bits_per_attribute``
and the demand is derived. Stating both is refused rather than reconciled, because a cell whose label
and corpus disagree lands on the trend plot at the wrong x, and that is worse than a missing cell.

**The hyperparameters are part of the cell, not of the launcher.** Learning rate, weight decay, batch
size, sequence length and the schedule are fields here with one value across the whole grid. An
earlier revision left them unspecified, which for an experiment whose result is a two-point trend is
not an implementation detail: cumulative weight decay alone varies 14.6x across a count row, and if
the value itself also drifted between cells there would be nothing left to attribute a difference to.
"""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from olmo_core.exceptions import OLMoConfigurationError

from .corpus import values as corpus_values
from .ladder import rho, sizes

__all__ = [
    "COUNT_AXIS_DEMANDS",
    "ENTROPY_AXIS_BITS",
    "FIRST_RUN_ROWS",
    "CellSpec",
    "ResolvedCell",
    "first_run_cells",
    "entropy_sweep_cells",
    "dilution_ladder_cells",
    "replicate_block",
    "grid_summary",
    "load_cell",
    "load_cells",
    "write_cells",
]


COUNT_AXIS_DEMANDS: Tuple[float, ...] = (0.30, 0.60, 1.20, 2.40, 4.80)
"""
Demanded bits per non-embedding parameter on the count axis.

Reads as :math:`\\rho` = 0.25 / 0.5 / 1 / 2 / 4 at the interpretive constant, but the cell is placed by
the demand, which has no hidden constant in it.
"""

ENTROPY_AXIS_BITS: Tuple[int, ...] = (0, 4, 8, 16, 24, 32)
"""
Bits per attribute on the entropy axis, at fixed entity count and fixed token count.

``b=8`` is four words from pools of four, giving 48 bits/entity against bioS's 47.592 -- so the sweep's
midpoint is the literature's corpus to within 0.9%.
"""

FIRST_RUN_ROWS: Tuple[Tuple[str, Tuple[float, ...]], ...] = (
    ("13M", COUNT_AXIS_DEMANDS),
    ("28M", COUNT_AXIS_DEMANDS),
    ("64M", COUNT_AXIS_DEMANDS[:4]),
)
"""
The first run: 13M and 28M at all five demands, 64M at four.

The 113M row and 64M's highest demand are omitted -- the top row on identification grounds, since
width scaling does not hold reasoning fixed and so that row could not break the size confound it was
added for. Submitted as three jobs, one per row, so the rows run concurrently and a failure in one does
not strand the others.
"""

REASONING_TOKENS: int = 1_000_000_000
"""
The default reasoning budget: 1.0B tokens **per slice**, held constant in *absolute* tokens (PRD 3.4).

Constant absolute rather than constant fraction, because holding the fraction would make reasoning
exposure vary with fact demand -- and reasoning exposure is the thing a reasoning endpoint is most
sensitive to. The cost is that the fraction varies instead: 1.0B is 53% of the smallest cell's tokens
and 2.5% of the largest, since fact tokens span 0.89B to 39.7B across the grid *by construction* --
that span is the axis. Equal absolute exposure is the invariant worth keeping; an equal fraction would
buy nothing and cost the comparison.

Per slice rather than per cell, which is a reading revision 2 did not pin down. A cell's reasoning
total is therefore 1.0B on the reasoning-only control and the entropy axis, both of which carry the
unrelated slice alone, and 2.0B on a count-axis cell that also carries the related one. Splitting a
fixed *total* instead would give the control twice the unrelated-slice exposure of every cell it is
the reference for -- so its score would beat theirs for a reason that has nothing to do with facts.

Whether 1.0B is *enough* at the top of the grid is a live calibration question and not one this
constant settles: M0's G1-G8 bracket the endpoints at both mixture extremes precisely to find out. If
2.5% turns out to be too thin for the task to be learned at all, every cell scores near its floor and
the endpoint is dead on dynamic range -- which is a gate failure, detected before the grid runs, not a
null.
"""

RELATED_REASONING_TOKENS: int = 50_000_000
"""
The related slice's own budget: 50M tokens, an order of magnitude below the unrelated slice's.

Sized on *per-entity coverage*, not on parity with `<mano>`. The related slice draws pairs from the
fixed 25,000-entity probe subset and names two entities per item, so 1.0B tokens would supervise each
entity's birth-year rank **4,211 times against the 200 exposures 3.3 fixes for the fact slice** -- 21x.
Three things follow from that, and all three are bad:

- The birth-year pool is 400 words with no ordinal structure, so the task needs a 400-element arbitrary
  total order plus 25,000 ranks stored in weights. That makes the related slice *parametric* in exactly
  the way 8.3 criticises Mano for being, unacknowledged.
- Roughly 216 kbit of fact demand enters outside the demand axis -- present in every demand cell and
  absent from the control.
- At 4,211 supervisions per entity the slice teaches the rank itself, so "needs two facts to answer" is
  no longer guaranteed and a decline would not be evidence about fact *access*.

50M gives 211 mentions per entity, level with the fact slice. The uncounted demand term does not go
away -- it is constant in absolute bits across cells, so it shifts the x-axis rather than tilting it --
but the slice stops out-supervising the facts it is supposed to depend on.
"""

_DOMAIN_TOKENS: Tuple[str, ...] = ("<facts>", "<mano>", "<compare>")
"""One per corpus slice, and the same tuple ``train_cell`` builds from."""


@dataclass(frozen=True)
class CellSpec:
    """
    One grid cell, as a config file states it.

    :param cell_id: Unique name. Also the checkpoint sub-prefix and the W&B run tag.
    :param row: Ladder row, e.g. ``"28M"``. Resolved through :func:`factcrowd.ladder.sizes.row`.
    :param sweep: ``"count"`` to vary entity count at fixed exposures, ``"entropy"`` to vary value-pool
        entropy at fixed entity count and fixed token count.
    :param demand_bits_per_param: The count axis's stated demand. Must be ``None`` on the entropy axis,
        where demand is derived.
    :param bits_per_attribute: The entropy axis's stated entropy. Must be ``None`` on the count axis.
    :param n_entities: Stated on the entropy axis, derived on the count axis.
    :param exposures: Times each fact appears. 200 everywhere; changing it changes what capacity per
        parameter means, which is why :func:`factcrowd.ladder.rho.resolve_r_e` refuses other values.
    :param sequence_length: Instance length. 512, where attention is 7.6% of FLOPs rather than the 30%
        it is at 2048.
    :param global_batch_size: Tokens per optimizer step. At least 256k, below which the small rows lose
        20-25% of their MFU to batch shape.
    :param learning_rate: Peak learning rate. One value across the grid.
    :param weight_decay: One value across the grid. ``sum(lr * wd)`` is logged per cell because it
        varies 14.6x across a count row and cannot be argued away.
    :param warmup_fraction: WSD warmup, as a fraction of the cell's own step count. A *fraction*
        rather than a step count, because a step count fixed across the grid does the opposite of what
        holding a hyperparameter constant is supposed to do: run length varies 37x across the grid, so
        2,000 steps was 1.4% of the largest cell and 52% of the reasoning-only control. The control
        would have consumed its reasoning tokens at a mean 0.69 of peak learning rate against 0.94 for
        the cell it is compared with -- systematically under-optimised, biasing the crowding
        measurement toward a null. That is the error 3.5 withdraws revision 1 for, reintroduced
        structurally. What has to be constant is the schedule *shape*.
    :param warmup_steps: An explicit warmup in steps, overriding ``warmup_fraction``. For smoke cells,
        which are too short for any fraction to be meaningful. Leave unset on a real cell.
    :param decay_fraction: Fraction of the run spent decaying to zero.
    :param related_reasoning_tokens: The related slice's own absolute budget, sized on per-entity
        coverage rather than on parity with the unrelated slice. See :data:`RELATED_REASONING_TOKENS`.
        Ignored where the cell carries no related slice.
    :param reasoning_tokens: Absolute tokens for the **unrelated** reasoning slice, identical in every
        cell. Per slice rather than a total split between them, because the control
        carries only the unrelated slice and an evenly split total would hand it twice the exposure of
        the cells it is the reference for. So a cell's reasoning total is this times the number of
        slices it carries. Zero marks a facts-only cell; the reasoning-only control states zero fact
        demand instead.
    :param replicate: Which replicate of this cell. Changes the model initialisation and the data
        order and **nothing else** -- the corpus, the reasoning items and their volumes are identical
        across replicates, so a set of replicates is a paired block over one fixed set of facts.

        This is what makes seed replication mean anything here. ``TransformerConfig.init_seed``
        defaults to 0 and was never set, so every cell and every notional replicate initialised the
        same model: varying the cell seed changed the corpus but not the network, and three "seeds"
        would have shared one initialisation. A shared eval set reduces measurement noise; it does not
        create trained-model replicates.
    :param seed: Seeds the entity table, the phrasing choice and the stream order, offset so the three
        stay independent. Held fixed across replicates.
    :param table_entities: Entities to generate in the table, if more than the slice uses. Lets several
        cells share one table on disk.
    :param notes: Free text carried into the run's config record.
    """

    cell_id: str
    row: str
    sweep: str = "count"
    demand_bits_per_param: Optional[float] = None
    bits_per_attribute: Optional[int] = None
    n_entities: Optional[int] = None
    exposures: int = rho.EXPOSURES
    sequence_length: int = 512
    global_batch_size: int = 262_144
    learning_rate: float = 3e-3
    weight_decay: float = 0.1
    warmup_fraction: float = 0.02
    warmup_steps: Optional[int] = None
    decay_fraction: float = 0.1
    reasoning_tokens: int = 0
    related_reasoning_tokens: int = 0
    replicate: int = 0
    seed: int = 1234
    table_entities: Optional[int] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.sweep not in ("count", "entropy"):
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'sweep' must be 'count' or 'entropy', got {self.sweep!r}"
            )
        sizes.row(self.row)  # raises, listing the ladder, if the row is unknown

        if self.sweep == "count":
            if self.demand_bits_per_param is None:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': the count axis is placed by "
                    f"'demand_bits_per_param', which is missing"
                )
            if self.bits_per_attribute is not None:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': 'bits_per_attribute' belongs to the entropy axis; the "
                    f"count axis uses the bioS schema's own 47.592 bits/entity"
                )
            if self.n_entities is not None:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': 'n_entities' is derived from the demand on the count "
                    f"axis, not stated. Stating both is how a cell comes to sit at a demand other "
                    f"than its label."
                )
            if self.demand_bits_per_param < 0:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': 'demand_bits_per_param' must not be negative, got "
                    f"{self.demand_bits_per_param}"
                )
            if self.demand_bits_per_param == 0 and self.reasoning_tokens <= 0:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': demand 0 is the reasoning-only control, so it needs "
                    f"'reasoning_tokens' above zero. As stated it has no facts and no reasoning, "
                    f"which is a run with no data at all."
                )
        else:
            if self.bits_per_attribute is None or self.n_entities is None:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': the entropy axis states both 'bits_per_attribute' and "
                    f"'n_entities' -- the entity count is what it holds fixed while entropy sweeps"
                )
            if self.demand_bits_per_param is not None:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': demand is derived on the entropy axis, not stated"
                )

        for name, value in (
            ("exposures", self.exposures),
            ("sequence_length", self.sequence_length),
            ("global_batch_size", self.global_batch_size),
            ("learning_rate", self.learning_rate),
        ):
            if value <= 0:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': '{name}' must be positive, got {value}"
                )
        if not 0 < self.warmup_fraction < 0.5:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'warmup_fraction' must be in (0, 0.5), got "
                f"{self.warmup_fraction}. Above a half the run is mostly ramp."
            )
        if self.warmup_steps is not None and self.warmup_steps <= 0:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'warmup_steps' must be positive when set, got "
                f"{self.warmup_steps}"
            )
        if not 0 < self.decay_fraction < 1:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'decay_fraction' must be in (0, 1), got "
                f"{self.decay_fraction}"
            )
        if self.replicate < 0:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'replicate' must not be negative, got {self.replicate}"
            )
        if self.related_reasoning_tokens < 0:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'related_reasoning_tokens' must not be negative, got "
                f"{self.related_reasoning_tokens}"
            )
        if self.sweep == "entropy" and (self.n_entities or 0) <= 0:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': the entropy axis needs a positive 'n_entities', got "
                f"{self.n_entities}"
            )
        if self.weight_decay < 0 or self.reasoning_tokens < 0 or self.seed < 0:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'weight_decay', 'reasoning_tokens' and 'seed' must not be "
                f"negative"
            )
        if self.global_batch_size % self.sequence_length:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'global_batch_size' ({self.global_batch_size:,}) must be a "
                f"multiple of 'sequence_length' ({self.sequence_length}), or a step holds a partial "
                f"instance"
            )

    # --- derived quantities ------------------------------------------------------------------------

    @property
    def qualified_id(self) -> str:
        """
        The cell id with its replicate, which is what a filename and a checkpoint prefix must use.

        ``cell_id`` alone omits the replicate, so ``write_cells`` wrote every replicate of a cell to the
        same YAML -- reproduced with r0 and r1 both landing on ``28m_b8.yaml`` -- and two replicates
        submitted from it would share a checkpoint prefix. A design whose whole inferential unit is the
        replicate cannot have the replicate absent from its identity.

        Replicate 0 keeps the bare id, so the committed single-replicate grid is unchanged.
        """
        return self.cell_id if self.replicate == 0 else f"{self.cell_id}_r{self.replicate}"

    @property
    def init_seed(self) -> int:
        """
        Seed for parameter initialisation. Varies with the replicate, never with the corpus.

        A prime stride so a replicate's initialisation cannot coincide with another cell's seed.
        """
        return self.seed + 9_973 * self.replicate

    @property
    def order_seed(self) -> int:
        """Seed for shuffling and batching. Varies with the replicate, like the initialisation."""
        return self.seed + 3 + 9_973 * self.replicate

    @property
    def is_control(self) -> bool:
        """
        Whether this is the reasoning-only control: reasoning data, no facts at all.

        The demand-0 endpoint of every crowding curve, and the only cell where a decline in reasoning
        cannot be attributed to fact load -- so it is what the rest of the row is compared against.
        """
        return self.sweep == "count" and self.demand_bits_per_param == 0

    @property
    def reasoning_slice_names(self) -> Tuple[str, ...]:
        """
        Which reasoning slices this cell carries, and therefore what its reasoning tokens multiply by.

        The single source of truth for that decision: :class:`factcrowd.train_cell.BuiltCorpus` builds
        the slices and then asserts its own set matches this one, so the token arithmetic here and the
        data there cannot disagree about a cell.

        The unrelated slice is everywhere. The related one needs an orderable fact to ask about, so it
        is absent from the reasoning-only control -- which has no facts -- and from the entropy axis,
        whose attributes are positional composites with no ordinal field.

        :returns: The slice names, in build order. Empty on a facts-only cell.
        """
        if self.reasoning_tokens <= 0:
            return ()
        if self.sweep == "count" and not self.is_control and self.related_reasoning_tokens > 0:
            return ("mano", "compare")
        return ("mano",)

    def slice_budget(self, name: str) -> int:
        """
        The absolute token budget for one named slice.

        :param name: A member of :attr:`reasoning_slice_names`.

        :returns: Tokens.

        :raises OLMoConfigurationError: If the cell does not carry that slice.
        """
        if name not in self.reasoning_slice_names:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}' carries {self.reasoning_slice_names}, not {name!r}"
            )
        return self.related_reasoning_tokens if name == "compare" else self.reasoning_tokens

    @property
    def ladder_row(self) -> sizes.LadderRow:
        """The ladder row this cell sits on."""
        return sizes.row(self.row)

    @property
    def non_embedding_params(self) -> int:
        """
        Non-embedding parameters, from the closed form.

        The authoritative count comes off the built model and is asserted against this by
        :func:`factcrowd.ladder.sizes.build`; this is what lets a dry run place the cell without
        ``torch``.
        """
        return sizes.non_embedding_params(self.ladder_row.d_model)

    @property
    def bits_per_entity(self) -> float:
        """
        Attribute bits per entity: the bioS figure on the count axis, ``6b`` on the entropy axis.

        Read from the schema builders rather than restated, so a change to the pools moves this too.
        """
        if self.sweep == "count":
            return corpus_values.bios_bits_per_entity()
        assert self.bits_per_attribute is not None
        return corpus_values.bits_per_entity_for(self.bits_per_attribute)

    def resolve(
        self, *, name_space: Optional[int] = None, vocab_size: Optional[int] = None
    ) -> "ResolvedCell":
        """
        Fill in everything the cell implies, and check the halves agree.

        :param name_space: The name universe, for the demand formula's name term. Defaults to the
            schema's own, which is what a run uses; pass a value only to explore the sensitivity.
        :param vocab_size: Padded vocabulary size, for the *total*-parameter basis (embeddings are tied,
            so one table). Without it both bases report the non-embedding count and 3's dual-basis
            reporting is silently one basis twice. ``train_cell`` passes the built vocabulary's size.

        :returns: The resolved cell.

        :raises OLMoConfigurationError: If the stated demand and the derived entity count disagree by
            more than 1%, or if the cell is unreachable.
        """
        universe = corpus_values.NAME_SPACE if name_space is None else name_space
        params = self.non_embedding_params
        # The total basis needs the embedding table, which needs the vocabulary. Without one the two
        # bases collapse into each other, and 3's argument for reporting both is that they diverge
        # monotonically with model size, so a design that quietly picks one loses the cross-size
        # comparability the size axis exists for. Passing vocab_size is what makes the second basis real
        # rather than a copy of the first. With our closed 3,584-word vocabulary the gap is 1.073x at 13M
        # falling to 1.024x at 113M -- far below the 1.650x/1.217x a tied 32k BPE would give, which is
        # why this is a precaution rather than a correction that changes a conclusion.
        total_params = (
            params if vocab_size is None else params + self.ladder_row.d_model * vocab_size
        )

        if self.sweep == "count" and self.demand_bits_per_param == 0:
            # The reasoning-only control. solve() refuses a zero target and should keep refusing it --
            # the linear path divides by bits_per_entity and the name-term path is non-monotone there,
            # so a solver that answered would be answering by accident. Zero facts is stated, not
            # solved for.
            n_entities = 0
        elif self.sweep == "count":
            assert self.demand_bits_per_param is not None
            size = rho.solve(
                params,
                self.demand_bits_per_param,
                bits_per_entity=self.bits_per_entity,
                name_space=universe,
                exposures=self.exposures,
            )
            n_entities = size.n_entities
            rho.check(
                params,
                self.demand_bits_per_param,
                n_entities,
                bits_per_entity=self.bits_per_entity,
                name_space=universe,
                label=self.cell_id,
            )
        else:
            assert self.n_entities is not None
            n_entities = self.n_entities

        demand = rho.demand(
            n_entities,
            bits_per_entity=self.bits_per_entity,
            non_embedding_params=params,
            total_params=total_params,
            name_space=universe,
        )
        return ResolvedCell(
            spec=self,
            n_entities=n_entities,
            non_embedding_params=params,
            demand_bits=demand.bits,
            demand_per_non_embedding_param=demand.per_non_embedding_param,
            attribute_bits=demand.attribute_bits,
            name_bits=demand.name_bits,
            total_params=total_params,
            demand_per_total_param=demand.per_total_param,
        )

    # --- serialisation -----------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The cell as a plain dictionary, with ``None`` fields dropped for a readable YAML."""
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CellSpec":
        """
        Build a cell from a parsed config, refusing unknown keys.

        :param raw: The parsed mapping.

        :returns: The cell.

        :raises OLMoConfigurationError: If a key is unknown or a required one is missing. Unknown keys
            are refused rather than ignored, because a typo'd override that silently does nothing is
            how a cell comes to run at settings nobody chose.
        """
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise OLMoConfigurationError(
                f"unknown config keys {unknown}; a cell accepts {sorted(known)}"
            )
        missing = sorted({"cell_id", "row"} - set(raw))
        if missing:
            raise OLMoConfigurationError(f"config is missing required keys {missing}")
        return cls(**raw)

    def with_overrides(self, **overrides: Any) -> "CellSpec":
        """
        A copy with fields replaced, re-validated.

        :param overrides: Fields to change.

        :returns: The new cell.
        """
        unknown = sorted(set(overrides) - set(self.__dataclass_fields__))
        if unknown:
            raise OLMoConfigurationError(f"unknown override keys {unknown}")
        return replace(self, **overrides)


@dataclass(frozen=True)
class ResolvedCell:
    """
    A cell with its derived quantities filled in.

    :param spec: The cell as stated.
    :param n_entities: Entities in the fact slice, stated or derived.
    :param non_embedding_params: Closed-form non-embedding parameter count.
    :param demand_bits: Total demanded bits, name term included.
    :param demand_per_non_embedding_param: The experiment's x-axis for this cell.
    :param attribute_bits: The ``N * bits_per_entity`` part.
    :param name_bits: The ``N * log2(N0/N)`` part.
    :param total_params: Non-embedding plus the tied embedding table, when a vocabulary was supplied.
        Equal to ``non_embedding_params`` when it was not.
    :param demand_per_total_param: The second reporting basis of 3. Equal to
        ``demand_per_non_embedding_param`` when no vocabulary was supplied.
    """

    spec: CellSpec
    n_entities: int
    non_embedding_params: int
    demand_bits: float
    demand_per_non_embedding_param: float
    attribute_bits: float
    name_bits: float
    total_params: int = 0
    demand_per_total_param: float = 0.0

    @property
    def rho(self) -> float:
        """The interpretive oversubscription ratio. Presentation only."""
        return rho.rho_from_demand(self.demand_per_non_embedding_param)

    @property
    def n_bios(self) -> int:
        """Biographies in the fact slice."""
        return self.n_entities * self.spec.exposures

    def fact_tokens(self, mean_tokens_per_bio: float) -> int:
        """
        Tokens in the fact slice, given a measured mean biography length.

        Takes the mean rather than assuming one, because the templates were rewritten once and the
        figure moved by a factor of four. The exact count comes from
        :attr:`factcrowd.corpus.stream.BioStream.num_tokens`.

        :param mean_tokens_per_bio: From :attr:`factcrowd.corpus.render.Renderer.mean_tokens_per_bio`.

        :returns: Estimated fact-slice tokens.
        """
        return int(self.n_bios * mean_tokens_per_bio)

    @property
    def reasoning_total(self) -> int:
        """Reasoning tokens across every slice this cell carries, at each slice's own budget."""
        return sum(self.spec.slice_budget(name) for name in self.spec.reasoning_slice_names)

    def total_tokens(self, mean_tokens_per_bio: float) -> int:
        """Fact tokens plus the cell's reasoning tokens, over every slice it carries."""
        return self.fact_tokens(mean_tokens_per_bio) + self.reasoning_total

    def steps(self, mean_tokens_per_bio: float) -> int:
        """
        Optimizer steps at the cell's batch size.

        :param mean_tokens_per_bio: From the renderer.

        :returns: Steps, rounding down; the remainder is under one batch.
        """
        return self.total_tokens(mean_tokens_per_bio) // self.spec.global_batch_size

    def warmup(self, steps: int) -> int:
        """
        WSD warmup in steps: the cell's fraction of its own run, or its explicit override.

        Taking the step count as an argument rather than recomputing it, so the schedule is derived
        from the same number the trainer's ``max_duration`` uses. At least one step, and never the whole
        run -- a cell that warms up for its entire length never reaches peak learning rate.

        :param steps: The cell's total optimizer steps.

        :returns: Warmup steps.

        :raises OLMoConfigurationError: If warmup would consume the whole run.
        """
        warmup = (
            self.spec.warmup_steps
            if self.spec.warmup_steps is not None
            else max(1, round(self.spec.warmup_fraction * steps))
        )
        if warmup >= steps:
            raise OLMoConfigurationError(
                f"cell '{self.spec.cell_id}': warmup of {warmup:,} steps is not below the run's "
                f"{steps:,}, so the learning rate never reaches its peak and the cell is not "
                f"comparable with any other."
            )
        return warmup

    def summary(self, mean_tokens_per_bio: float) -> Dict[str, Any]:
        """
        Everything worth printing in a dry run or logging at startup.

        :param mean_tokens_per_bio: From the renderer.

        :returns: A flat mapping.
        """
        steps = self.steps(mean_tokens_per_bio)
        return {
            "cell_id": self.spec.cell_id,
            "row": self.spec.row,
            "sweep": self.spec.sweep,
            "d_model": self.spec.ladder_row.d_model,
            "non_embedding_params": self.non_embedding_params,
            "bits_per_entity": round(self.spec.bits_per_entity, 4),
            "n_entities": self.n_entities,
            "exposures": self.spec.exposures,
            "demand_bits_per_param": round(self.demand_per_non_embedding_param, 4),
            "demand_bits_per_total_param": round(self.demand_per_total_param, 4),
            "total_params": self.total_params,
            "rho_at_declared_r_e": round(self.rho, 4),
            "attribute_bits": int(self.attribute_bits),
            "name_bits": int(self.name_bits),
            "mean_tokens_per_bio": round(mean_tokens_per_bio, 2),
            "fact_tokens": self.fact_tokens(mean_tokens_per_bio),
            "reasoning_tokens_unrelated": self.spec.reasoning_tokens,
            "reasoning_tokens_related": self.spec.related_reasoning_tokens,
            "reasoning_slices": ",".join(self.spec.reasoning_slice_names) or "none",
            "reasoning_tokens": self.reasoning_total,
            "total_tokens": self.total_tokens(mean_tokens_per_bio),
            "steps": steps,
            "replicate": self.spec.replicate,
            "init_seed": self.spec.init_seed,
            "order_seed": self.spec.order_seed,
            "warmup_steps": self.warmup(steps),
            "warmup_pct": round(100 * self.warmup(steps) / steps, 2),
            "cumulative_lr_times_wd": round(
                steps * self.spec.learning_rate * self.spec.weight_decay, 4
            ),
        }


def _cell_id(row: str, sweep: str, key: Union[float, int]) -> str:
    """A stable, filename-safe cell id."""
    if sweep == "count":
        return f"{row.lower()}_d{str(key).replace('.', 'p')}"
    return f"{row.lower()}_b{int(key)}"


def first_run_cells(**overrides: Any) -> Tuple[CellSpec, ...]:
    """
    The count-axis grid of :data:`FIRST_RUN_ROWS`, plus one reasoning-only control per row.

    :param overrides: Applied to every cell, e.g. ``seed`` or ``learning_rate``.

    :returns: The cells, in the order the three jobs should run them.
    """
    settings: Dict[str, Any] = {
        "reasoning_tokens": REASONING_TOKENS,
        "related_reasoning_tokens": RELATED_REASONING_TOKENS,
        **overrides,
    }
    control: Dict[str, Any] = dict(settings)
    control["notes"] = control.get("notes") or "reasoning-only control: no facts"
    cells: List[CellSpec] = []
    for row, demands in FIRST_RUN_ROWS:
        cells.append(
            CellSpec(
                cell_id=f"{row.lower()}_ctrl",
                row=row,
                sweep="count",
                demand_bits_per_param=0.0,
                **control,
            )
        )
        for demand in demands:
            cells.append(
                CellSpec(
                    cell_id=_cell_id(row, "count", demand),
                    row=row,
                    sweep="count",
                    demand_bits_per_param=demand,
                    **settings,
                )
            )
    return tuple(cells)


def entropy_sweep_cells(row: str = "28M", **overrides: Any) -> Tuple[CellSpec, ...]:
    """
    The iso-token entropy sweep at one row: entity count fixed, entropy swept.

    The entity count is the row's own demand-1.20 count, so ``b=8`` lands on the bioS anchor and the
    sweep straddles it.

    :param row: Which ladder row.
    :param overrides: Applied to every cell.

    :returns: The cells.
    """
    params = sizes.non_embedding_params(sizes.row(row).d_model)
    fixed = rho.solve(
        params,
        1.20,
        bits_per_entity=corpus_values.bios_bits_per_entity(),
        name_space=corpus_values.NAME_SPACE,
    ).n_entities
    settings: Dict[str, Any] = {"reasoning_tokens": REASONING_TOKENS, **overrides}
    return tuple(
        CellSpec(
            cell_id=_cell_id(row, "entropy", bits),
            row=row,
            sweep="entropy",
            bits_per_attribute=bits,
            n_entities=fixed,
            **settings,
        )
        for bits in ENTROPY_AXIS_BITS
    )


def dilution_ladder_cells(
    row: str = "13M", *, demand_bits_per_param: float = 0.0, **overrides: Any
) -> Tuple[CellSpec, ...]:
    """
    G8's calibration ladder: one cell trained on a decreasing share of its reasoning tokens.

    This is the gate that makes a null mean something, and until it runs, :mod:`factcrowd.score_run`
    marks every row non-confirmatory -- correctly, since PRD 8.6 admits a row only on gate evidence. It
    is also the cheapest thing in the design: on the default control at 13M the reference arm is 1.0B
    tokens and the whole ladder is under 1.5 slot-hours on 4xA10G.

    **Why the default is the reasoning-only control.** Diluting reasoning tokens inside a mixture also
    moves the mixture ratio and the total token budget, so a drop there has three candidate causes. With
    no facts present, reasoning exposure is the only thing that varies, which is what a calibration
    instrument wants. The cost is that the ladder then measures the endpoint's dose-response on a model
    trained on reasoning alone, and the confirmatory cells are mixtures -- so pass
    ``demand_bits_per_param`` to build the mixture-matched ladder instead, and read it knowing the dose
    is confounded with the ratio.

    :param row: Which ladder row. Use the row whose slope the ladder is calibrating.
    :param demand_bits_per_param: Facts to carry. ``0.0``, the default, is the reasoning-only control.
    :param overrides: Applied to every arm, e.g. ``seed``.

    :returns: One cell per dose in :data:`factcrowd.measure.gates.DILUTION_DOSES_PCT`, strongest dose
        last.

    :raises OLMoConfigurationError: If an arm would carry no reasoning tokens at all.
    """
    from .measure.gates import (
        DILUTION_DOSES_PCT,  # single source of truth for the doses
    )

    is_control = demand_bits_per_param == 0.0
    settings: Dict[str, Any] = {
        "reasoning_tokens": REASONING_TOKENS,
        # A control carries no orderable fact, so `<compare>` is absent from it whatever this says
        # (see `reasoning_slice_names`). Stated as zero rather than left at a value that is dropped.
        "related_reasoning_tokens": 0 if is_control else RELATED_REASONING_TOKENS,
        **overrides,
    }
    reference = int(settings["reasoning_tokens"])
    related = int(settings["related_reasoning_tokens"])
    cells: List[CellSpec] = []
    for dose in DILUTION_DOSES_PCT:
        scaled = (reference * dose) // 100
        if scaled <= 0:
            raise OLMoConfigurationError(
                f"the {dose}% arm of the dilution ladder would carry {scaled} reasoning tokens. The "
                f"reference arm is {reference:,}; a ladder needs every arm to train on something."
            )
        arm = dict(settings)
        arm["reasoning_tokens"] = scaled
        # The related slice takes the same dose. Holding it fixed would make the ladder a ratio sweep
        # between the two reasoning slices, and G8 reads it as one reasoning-exposure dose.
        arm["related_reasoning_tokens"] = (related * dose) // 100
        arm["notes"] = (
            f"G8 dilution ladder: {dose}% of the reasoning tokens"
            f"{', no facts' if is_control else f', demand {demand_bits_per_param}'}. "
            f"Reference arm is the 100% cell."
        )
        cells.append(
            CellSpec(
                cell_id=f"{row.lower()}_dil{dose}",
                row=row,
                sweep="count",
                demand_bits_per_param=demand_bits_per_param,
                **arm,
            )
        )
    return tuple(cells)


def load_cell(path: Union[str, Path]) -> CellSpec:
    """
    Read one cell from a YAML or JSON file.

    :param path: The config file.

    :returns: The cell.

    :raises OLMoConfigurationError: If the file is unreadable, empty, or holds unknown keys.
    """
    import yaml

    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise OLMoConfigurationError(
            f"{path} must hold a mapping of cell fields, got {type(raw).__name__}"
        )
    try:
        return CellSpec.from_dict(raw)
    except OLMoConfigurationError as error:
        raise OLMoConfigurationError(f"{path}: {error}") from None


def load_cells(directory: Union[str, Path]) -> Tuple[CellSpec, ...]:
    """
    Read every cell in a directory, sorted by filename so a fan-out index is stable.

    A fan-out maps ``AWS_BATCH_JOB_ARRAY_INDEX`` to a cell by position, so the order has to be a
    function of the directory contents and nothing else.

    :param directory: Directory of config files.

    :returns: The cells, sorted by filename.

    :raises OLMoConfigurationError: If the directory holds no configs.
    """
    paths = sorted(p for p in Path(directory).iterdir() if p.suffix in (".yaml", ".yml", ".json"))
    if not paths:
        raise OLMoConfigurationError(f"no cell configs found in {directory}")
    return tuple(load_cell(path) for path in paths)


def write_cells(cells: Iterable[CellSpec], directory: Union[str, Path]) -> List[Path]:
    """
    Write cells out as YAML, one file each.

    Regenerating the grid is a command rather than a hand edit, so the committed configs and the
    generator cannot drift.

    :param cells: The cells.
    :param directory: Destination, created if absent.

    :returns: The paths written.
    """
    import yaml

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    seen: Dict[Path, str] = {}
    for cell in cells:
        # The replicate is part of the filename, and a collision is refused rather than resolved by
        # whichever cell was generated last. Two replicates writing one file is silent data loss in the
        # one place the design cannot afford it: the replicate is the inferential unit.
        path = target / f"{cell.qualified_id}.yaml"
        if path in seen:
            raise OLMoConfigurationError(
                f"cells '{seen[path]}' and '{cell.cell_id}' (replicate {cell.replicate}) both write "
                f"{path.name}. A replicate must be part of a cell's identity or one silently replaces "
                f"the other -- and two runs submitted from them would share a checkpoint prefix."
            )
        seen[path] = cell.cell_id
        path.write_text(yaml.safe_dump(cell.to_dict(), sort_keys=True))
        written.append(path)
    return written


def replicate_block(cells: Iterable[CellSpec], replicates: int) -> Tuple[CellSpec, ...]:
    """
    Expand a grid into paired replicate blocks.

    Each replicate carries the *same* corpus -- same entity table, same reasoning items, same volumes --
    and differs only in initialisation and data order. That is what makes the set a paired block and the
    per-seed slope the right inferential unit (PRD 8.5).

    :param cells: One replicate's worth of cells.
    :param replicates: How many replicates to produce.

    :returns: Every cell at every replicate, grouped by replicate.

    :raises OLMoConfigurationError: If ``replicates`` is below one.
    """
    if replicates < 1:
        raise OLMoConfigurationError(f"'replicates' must be at least 1, got {replicates}")
    base = list(cells)
    out: List[CellSpec] = []
    for index in range(replicates):
        out.extend(replace(cell, replicate=index) for cell in base)
    return tuple(out)


def grid_summary(cells: Iterable[CellSpec], mean_tokens_per_bio: float) -> str:
    """
    A one-line-per-cell table, for a dry run and for the record.

    :param cells: The cells.
    :param mean_tokens_per_bio: From the renderer.

    :returns: The table as text.
    """
    rows = [cell.resolve().summary(mean_tokens_per_bio) for cell in cells]
    keys = (
        "cell_id",
        "row",
        "demand_bits_per_param",
        "rho_at_declared_r_e",
        "n_entities",
        "total_tokens",
        "steps",
    )
    widths = {
        key: max(
            len(key),
            *(len(f"{r[key]:,}" if isinstance(r[key], int) else str(r[key])) for r in rows),
        )
        for key in keys
    }
    lines = ["  ".join(key.rjust(widths[key]) for key in keys)]
    for row in rows:
        lines.append(
            "  ".join(
                (f"{row[key]:,}" if isinstance(row[key], int) else str(row[key])).rjust(widths[key])
                for key in keys
            )
        )
    total = sum(r["total_tokens"] for r in rows)
    lines.append(f"\n{len(rows)} cells, {total:,} tokens total ({total / 1e9:.2f}B)")
    return "\n".join(lines)


def as_json(cells: Iterable[CellSpec], mean_tokens_per_bio: float) -> str:
    """
    The resolved grid as JSON, for the run's own record.

    :param cells: The cells.
    :param mean_tokens_per_bio: From the renderer.

    :returns: JSON text.
    """
    return (
        json.dumps([cell.resolve().summary(mean_tokens_per_bio) for cell in cells], indent=2) + "\n"
    )
