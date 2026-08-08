"""
One tidy row per (cell, replicate, step): the table analysis reads.

Scoring produces objects; analysis wants a rectangle. This module is the join between them, and it keeps
two properties that matter more than convenience:

**Every row carries its own provenance.** The cell id, the demanded bits, the fingerprints and the step
travel with the numbers, so a row is interpretable without the directory it came from. A table whose
rows only make sense next to a filesystem is a table that stops making sense.

**Nothing is recomputed here.** Accuracy, above-floor and achieved bits are all read from the objects
that measured them. A collector that re-derives a quantity is a second implementation of it, and the two
drift.

Output is CSV via the standard library rather than parquet via pandas: the whole first run is 120 rows,
and a dependency that has to be in the training image to write 120 rows is a poor trade.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from olmo_core.exceptions import OLMoConfigurationError

from .bits import AchievedBits
from .checkpoint import CheckpointRef
from .endpoints import EndpointResult

IDENTITY_COLUMNS: Tuple[str, ...] = (
    "cell_id",
    "row",
    "sweep",
    "replicate",
    "step",
    "demand_bits_per_param",
    "bits_per_attribute",
    "n_entities",
    "reasoning_tokens",
    "related_reasoning_tokens",
    "checkpoint_path",
    "confirmatory",
    "admission",
)
"""
What identifies a row, before any measurement.

``replicate`` and ``step`` are here because they are the two axes a trend is taken over, and leaving
either implicit is how a paired design gets analysed as an unpaired one.
"""


@dataclass
class ScoredCheckpoint:
    """
    Everything measured at one checkpoint.

    :param ref: Which checkpoint.
    :param cell: The cell record as saved, so the row needs no second lookup.
    :param resolved: The cell's *resolved* summary, which carries both halves of the identity block.
        The two sweeps each state only half in the cell itself.
    :param endpoints: One result per reasoning endpoint.
    :param achieved: The bit measurement, or ``None`` on the reasoning-only control.
    :param recall: Optional recall figures, keyed by name.
    :param extra: Anything else worth carrying, e.g. the fingerprints.
    """

    ref: CheckpointRef
    cell: Dict[str, Any]
    resolved: Dict[str, Any] = field(default_factory=dict)
    endpoints: Sequence[EndpointResult] = ()
    achieved: Optional[AchievedBits] = None
    recall: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def stated(self, key: str, default: Any = None) -> Any:
        """
        One identity field, read from the resolved record first and the cell second.

        RESOLVED FIRST, THEN THE CELL. The two sweeps state disjoint halves of the identity block: the
        count axis states a demand and derives an entity count, the entropy axis states an entity count
        and derives the demand. :meth:`factcrowd.cells.CellSpec.to_dict` drops ``None``, so reading the
        cell alone leaves ``demand_bits_per_param`` empty on the entropy axis -- and that column *is*
        :attr:`factcrowd.analysis.trend.SeedBlock.demands`, the regressor. The identified axis would have
        reached the analysis with no x.

        A method rather than a closure inside :meth:`rows` because
        :mod:`factcrowd.measure.evidence` needs the same rule, and two copies of a precedence rule is
        one copy that can be updated alone.

        :param key: The field name.
        :param default: Returned when neither record carries it.

        :returns: The value.
        """
        for source in (self.resolved or {}, self.cell):
            value = source.get(key)
            if value is not None:
                return value
        return default

    def rows(self) -> List[Dict[str, Any]]:
        """
        Flatten to one row per endpoint.

        Long rather than wide -- one row per (checkpoint, endpoint) rather than one row with a column per
        endpoint. Long survives adding an endpoint; wide needs a schema change, and PRD 8.3 already names
        two more (Brevo1, Reasoning Core) that are not built.

        A checkpoint with no endpoints still emits one row, so the bit curve is collectable before the
        reasoning half is.

        :returns: The rows.
        """
        # See `stated`: resolved first, then the cell, because the two sweeps state disjoint halves.
        stated = self.stated
        identity = {
            "cell_id": stated("cell_id"),
            "row": stated("row"),
            "sweep": stated("sweep"),
            "replicate": stated("replicate", 0),
            "step": self.ref.step,
            "demand_bits_per_param": stated("demand_bits_per_param"),
            "bits_per_attribute": stated("bits_per_attribute"),
            "n_entities": stated("n_entities"),
            "reasoning_tokens": stated("reasoning_tokens"),
            "related_reasoning_tokens": stated("related_reasoning_tokens"),
            "checkpoint_path": self.ref.path,
        }
        measured: Dict[str, Any] = dict(self.extra)
        if self.achieved is not None:
            measured.update(self.achieved.summary())
        # Used as given. `RecallResult.summary` already namespaces its keys (`template_all_chance`), and
        # prefixing again here -- against a `recall_` prefix `score_run` was separately stripping -- turned
        # every column into `recall_e_all_chance`. Three layers each renaming, agreeing until one of them
        # was renamed. One owner: the result object names its own fields.
        measured.update(self.recall)

        if not self.endpoints:
            return [{**identity, **measured, "storage_row": True}]
        # STORAGE IS PER CHECKPOINT AND THE TABLE IS PER ENDPOINT, SO THE STORAGE FIELDS REPEAT.
        # A cell carrying two endpoints emits two rows with identical `achieved_bits_per_param`, and any
        # analysis that averages or sums that column over rows double-weights exactly those cells -- on
        # the count axis every fact-bearing cell and none of the controls, so the bias is systematic
        # rather than noisy. `storage_row` is True on exactly one row per checkpoint; filter on it before
        # touching a storage column.
        return [
            {**identity, **measured, **result.summary(), "storage_row": index == 0}
            for index, result in enumerate(self.endpoints)
        ]


def select_complete(
    scored: Sequence[ScoredCheckpoint],
) -> Tuple[List[ScoredCheckpoint], List[str]]:
    """
    Keep one checkpoint per ``(cell_id, replicate)`` and only where training actually finished.

    Three things go wrong without this, and all three are quiet.

    **A crashed cell and its re-run are two prefixes and one cell.** ``13m_d0p6`` died at step 3,441 and
    was re-run to 10,732; both wrote checkpoints, both are under prefixes the scorer is pointed at, and
    the table then carries the cell twice with different numbers.

    **``--last-only`` means "the highest checkpoint in this prefix", not "the planned final step".** For a
    crashed run the highest checkpoint is the one before it died, which is a partially-trained model
    presented in the same column as fully-trained ones.

    **A short grid still writes a table.** Nothing compared what was scored against what should exist, so
    an analysis could be run on fifteen cells believing it had eighteen.

    Completion is judged against the cell's own recorded plan -- the last entry of
    ``checkpoint_steps`` -- rather than against a number passed in, because each cell has its own.

    :param scored: Scored checkpoints, possibly several per cell and possibly partial.

    :returns: The kept checkpoints, one per cell, and a list of human-readable notes about what was
        dropped and why. The notes are returned rather than logged so a caller can put them in the record.
    """
    by_cell: Dict[Tuple[str, int], List[ScoredCheckpoint]] = {}
    for entry in scored:
        key = (str(entry.stated("cell_id")), int(entry.stated("replicate", 0)))
        by_cell.setdefault(key, []).append(entry)

    kept: List[ScoredCheckpoint] = []
    notes: List[str] = []
    for (cell_id, replicate), entries in sorted(by_cell.items()):
        planned = 0
        for entry in entries:
            steps = entry.extra.get("checkpoint_steps") or []
            if steps:
                planned = max(planned, int(max(steps)))
        best = max(entries, key=lambda e: e.ref.step)
        if planned and best.ref.step < planned:
            notes.append(
                f"{cell_id} r{replicate}: highest checkpoint is step {best.ref.step:,} but the cell "
                f"planned {planned:,}, so this run never finished and is dropped"
            )
            continue
        if len(entries) > 1:
            notes.append(
                f"{cell_id} r{replicate}: {len(entries)} runs found "
                f"(steps {sorted(e.ref.step for e in entries)}); kept step {best.ref.step:,}"
            )
        kept.append(best)
    return kept, notes


def write_csv(rows: Sequence[Dict[str, Any]], path: Path) -> Path:
    """
    Write rows to CSV with a stable, union-of-keys header.

    The header is the union across rows and is **sorted with the identity columns first**, so a diff
    between two collections is readable and a missing measurement shows as an empty cell rather than
    shifting every column to its right.

    :param rows: The rows.
    :param path: Destination file. Parent directories are created.

    :returns: The path written.

    :raises OLMoConfigurationError: If there are no rows -- an empty table is nearly always a collection
        that pointed at the wrong prefix, and writing a header-only file hides that.
    """
    if not rows:
        raise OLMoConfigurationError(
            "no rows to write; check the checkpoint prefix and that scoring produced results"
        )
    keys: set = set()
    for row in rows:
        keys.update(row)
    ordered = [column for column in IDENTITY_COLUMNS if column in keys]
    ordered += sorted(keys - set(ordered))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})
    return path


def collect(scored: Iterable[ScoredCheckpoint]) -> List[Dict[str, Any]]:
    """
    Flatten many scored checkpoints into rows, sorted so a trend reads in order.

    :param scored: The scored checkpoints, in any order.

    :returns: Rows sorted by cell, replicate, step and endpoint.
    """
    rows: List[Dict[str, Any]] = []
    for item in scored:
        rows.extend(item.rows())
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("cell_id") or ""),
            int(row.get("replicate") or 0),
            int(row.get("step") or 0),
            str(row.get("endpoint") or ""),
        ),
    )
