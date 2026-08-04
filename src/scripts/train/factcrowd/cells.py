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

_DOMAIN_TOKENS: Tuple[str, ...] = ("<facts>", "<mano>", "<brevo>", "<related>")
"""One per corpus slice. Mandatory on every segment; there is no flag to disable it."""


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
    :param warmup_steps: WSD warmup. The scheduler itself is OLMo-core's ``WSD``.
    :param decay_fraction: Fraction of the run spent decaying to zero.
    :param reasoning_tokens: Absolute tokens **per reasoning slice**, identical in every cell that
        carries that slice. Per slice rather than a total split between them, because the control
        carries only the unrelated slice and an evenly split total would hand it twice the exposure of
        the cells it is the reference for. So a cell's reasoning total is this times the number of
        slices it carries. Zero marks a facts-only cell; the reasoning-only control states zero fact
        demand instead.
    :param seed: Seeds the entity table, the phrasing choice and the stream order, offset so the three
        stay independent.
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
    warmup_steps: int = 2_000
    decay_fraction: float = 0.1
    reasoning_tokens: int = 0
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
            ("warmup_steps", self.warmup_steps),
        ):
            if value <= 0:
                raise OLMoConfigurationError(
                    f"cell '{self.cell_id}': '{name}' must be positive, got {value}"
                )
        if not 0 < self.decay_fraction < 1:
            raise OLMoConfigurationError(
                f"cell '{self.cell_id}': 'decay_fraction' must be in (0, 1), got "
                f"{self.decay_fraction}"
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
        if self.sweep == "count" and not self.is_control:
            return ("mano", "compare")
        return ("mano",)

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

    def resolve(self, *, name_space: Optional[int] = None) -> "ResolvedCell":
        """
        Fill in everything the cell implies, and check the halves agree.

        :param name_space: The name universe, for the demand formula's name term. Defaults to the
            schema's own, which is what a run uses; pass a value only to explore the sensitivity.

        :returns: The resolved cell.

        :raises OLMoConfigurationError: If the stated demand and the derived entity count disagree by
            more than 1%, or if the cell is unreachable.
        """
        universe = corpus_values.NAME_SPACE if name_space is None else name_space
        params = self.non_embedding_params
        total_params = params + 0  # embeddings are added by the caller that knows the vocabulary

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
            total_params=max(total_params, params),
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
    """

    spec: CellSpec
    n_entities: int
    non_embedding_params: int
    demand_bits: float
    demand_per_non_embedding_param: float
    attribute_bits: float
    name_bits: float

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
        """Reasoning tokens across every slice: the per-slice budget times the number of slices."""
        return self.spec.reasoning_tokens * len(self.spec.reasoning_slice_names)

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
            "rho_at_declared_r_e": round(self.rho, 4),
            "attribute_bits": int(self.attribute_bits),
            "name_bits": int(self.name_bits),
            "mean_tokens_per_bio": round(mean_tokens_per_bio, 2),
            "fact_tokens": self.fact_tokens(mean_tokens_per_bio),
            "reasoning_tokens_per_slice": self.spec.reasoning_tokens,
            "reasoning_slices": ",".join(self.spec.reasoning_slice_names) or "none",
            "reasoning_tokens": self.reasoning_total,
            "total_tokens": self.total_tokens(mean_tokens_per_bio),
            "steps": steps,
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
    settings: Dict[str, Any] = {"reasoning_tokens": REASONING_TOKENS, **overrides}
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
    for cell in cells:
        path = target / f"{cell.cell_id}.yaml"
        path.write_text(yaml.safe_dump(cell.to_dict(), sort_keys=True))
        written.append(path)
    return written


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
