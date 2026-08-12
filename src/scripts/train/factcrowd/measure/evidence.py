"""
Assemble a gate report from runs that have already been scored.

:mod:`factcrowd.measure.gates` defines the gates and :mod:`factcrowd.score_run` consumes a report to
decide admission, but until this module existed nothing *produced* one. The consequence was concrete
rather than theoretical: every row of a finished confirmatory grid arrived labelled
``confirmatory=False, admission="no gate report"``, and the only way to change that was to hand-write
JSON -- which is to say, to assert the gates had passed rather than to measure it.

What this does is narrow on purpose. It reads scored checkpoints, recognises which of them are playing
a gate role, and hands the numbers to :func:`factcrowd.measure.gates.run_gates`. It does not decide
anything; the gates do. Evidence it cannot find is not filled in with a default -- the gate returns its
own refusal naming the arm still owed, which is the behaviour that makes an early report a useful
checklist instead of a false clean bill.

**Which gates this can currently feed.** Six of seven:

===== ============================================= ==================================================
gate  evidence                                       where it comes from
===== ============================================= ==================================================
G1    accuracy against expression depth              ``*_manoL{depth}`` / ``*_ctxmanoL{depth}``, from
                                                     :func:`factcrowd.cells.mano_calibration_cells`
G2    an untrained checkpoint                        the step-0 checkpoint every run already writes
G4    the achievable ceiling                         the reasoning-only control, ``*_ctrl``
G6    accuracy against parameters at fixed depth     the controls across ladder rows
G7    run-to-run sigma                               replicates of one cell
G8    the reasoning-token dilution ladder            ``*_dil{dose}``, from
                                                     :func:`factcrowd.cells.dilution_ladder_cells`
===== ============================================= ==================================================

**G1 and G2 were the two cheapest gates in the design and both were unreachable for the silliest of
reasons.** ``run_gates`` has accepted ``depth_scores`` and ``random_init_result`` from the beginning and
nothing ever supplied them, so a scored calibration sweep would have left G1 reporting as owed however
much evidence sat in the CSV -- and G2 asks for an untrained checkpoint, which
``CheckpointerCallback.pre_train_checkpoint`` has written at step 0 on every run there has ever been. The
step-0 rows are read from the raw sequence rather than from the per-cell last checkpoint, which is where
they were being discarded.

Only **G3** (the premise-ablated probe) still needs a corpus variant that does not exist. It will report as
owed, and a row cannot be admitted while it does. That is the honest state of the design and not a gap this
module should paper over.
"""

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from olmo_core.exceptions import OLMoConfigurationError

from .collect import ScoredCheckpoint
from .endpoints import EndpointResult
from .gates import GATE_REPORT_VERSION, GateReport, run_gates

__all__ = [
    "DILUTION_CELL_PATTERN",
    "DEPTH_CELL_PATTERN",
    "RoleAssignment",
    "assign_roles",
    "assemble",
]


DILUTION_CELL_PATTERN = re.compile(r"^(?P<row>[^_]+)_dil(?P<dose>\d+)$")
"""
Matches the cell ids :func:`factcrowd.cells.dilution_ladder_cells` writes, e.g. ``13m_dil80``.

Recognising the ladder by name rather than by a flag in the config is deliberate: the ids are generated
by one function and the pattern is pinned by a test, so the two cannot drift. A cell hand-named into
this shape would be picked up, which is why :func:`assign_roles` also checks that what it found is a
complete ladder over one row before handing it to G8.
"""

DEPTH_CELL_PATTERN = re.compile(r"^(?P<row>[^_]+)_(?P<prefix>ctx)?manoL(?P<depth>\d+)$")
"""
Matches the calibration ids :func:`factcrowd.cells.mano_calibration_cells` writes -- ``28m_manoL10`` for
the memorised endpoint and ``28m_ctxmanoL04`` for the in-context one.

``prefix`` is what keeps the two apart, and it has to: they are different instruments with different
floors, so a depth curve mixing ``manoL04`` with ``ctxmanoL04`` would be two curves averaged into one.
:func:`assign_roles` reads only the cells whose prefix matches the endpoint being admitted.
"""

_DEPTH_PREFIX_FOR_ENDPOINT = {"mano": None, "ctxmano": "ctx"}
"""
Which calibration prefix feeds which endpoint's G1.

An endpoint absent from this mapping -- ``compare``, or the ``mano_table`` probe -- has no depth sweep and
G1 reports as owed for it, which is correct rather than a gap: no ladder over its depth exists.
"""

IDENTITY_FIELDS: Tuple[str, ...] = ("row", "sweep", "total_params")
"""
The fields a gate report binds itself to, so it cannot admit a row it says nothing about.

Chosen to be the smallest set that separates measurements which are not evidence about each other:
``row`` and ``total_params`` are the network, and ``sweep`` is the vocabulary -- the count axis builds 3,554
words and the entropy axis 8,000, so the two differ in softmax width and in total parameters even at one
ladder row. The endpoint's own depth is added per endpoint, since ``ctxmano_length`` and ``mano_length``
each matter only to their own endpoint.

Deliberately *not* including the replicate or the demand: a gate report is a statement that the instrument
works at this configuration, and it is meant to admit every arm of the sweep it was gathered for.
"""

_DEPTH_FIELD_FOR_ENDPOINT = {"mano": "mano_length", "ctxmano": "ctxmano_length"}

_CONTROL_SUFFIX = "_ctrl"


def _final(scored: Iterable[ScoredCheckpoint]) -> Dict[Tuple[str, int], ScoredCheckpoint]:
    """
    The last checkpoint of each ``(cell, replicate)``, which is the one a gate reads.

    Intermediate checkpoints are the trajectory, not the result. Averaging them into a gate would mix a
    partially-trained model into a statement about what the design can resolve.
    """
    latest: Dict[Tuple[str, int], ScoredCheckpoint] = {}
    for entry in scored:
        key = (entry.stated("cell_id"), int(entry.stated("replicate", 0)))
        held = latest.get(key)
        if held is None or entry.ref.step > held.ref.step:
            latest[key] = entry
    return latest


def _result(entry: ScoredCheckpoint, endpoint: str) -> Optional[EndpointResult]:
    """The endpoint's full result at one checkpoint, or ``None`` if it was not scored there."""
    for result in entry.endpoints:
        if result.name == endpoint:
            return result
    return None


def _accuracy(entry: ScoredCheckpoint, endpoint: str) -> Optional[float]:
    """
    Just the accuracy, for the gates that take a bare fraction.

    Kept distinct from :func:`_result` so the choice is explicit at each call site: G7 needs the result
    because it caps the unparseable rate, and handing it a float would leave that half of the gate with
    nothing to fail on.
    """
    result = _result(entry, endpoint)
    return None if result is None else result.accuracy


class RoleAssignment:
    """
    Which scored runs play which gate role.

    Kept separate from :func:`assemble` so the assignment can be inspected and reported without running
    the gates -- "G8 saw doses 100/95/90" is the diagnostic a caller needs when a gate refuses, and it
    is not recoverable from the refusal text alone.

    :param endpoint: The endpoint being admitted.
    :param result: Its score at the cell under test.
    :param dilution: G8's ladder, keyed by percent of reasoning tokens retained.
    :param ceiling: G4's achievable ceiling.
    :param by_params: G6's accuracy keyed by non-embedding parameters.
    :param replicates: G7's per-replicate results. **Results, not accuracies**: G7 also caps the
        unparseable rate of its worst replicate, and a bare fraction carries no such count, so half
        the gate would silently have no evidence to fail on.
    :param depths: G1's task-depth sweep, keyed by expression depth.
    :param random_init: G2's untrained checkpoint, the step-0 one every run writes.
    :param under_test: The whole checkpoint :attr:`result` came from, so a report can bind itself to the
        configuration it was built around rather than only to the endpoint's name.
    :param notes: What was recognised and what was skipped, for the caller to log.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        result: Optional[EndpointResult] = None,
        dilution: Optional[Mapping[int, float]] = None,
        ceiling: Optional[float] = None,
        by_params: Optional[Mapping[int, float]] = None,
        replicates: Optional[Sequence[EndpointResult]] = None,
        depths: Optional[Mapping[int, float]] = None,
        random_init: Optional[EndpointResult] = None,
        under_test: Optional[ScoredCheckpoint] = None,
        notes: Sequence[str] = (),
    ) -> None:
        self.endpoint = endpoint
        self.result = result
        self.dilution = dict(dilution) if dilution else None
        self.ceiling = ceiling
        self.by_params = dict(by_params) if by_params else None
        self.replicates = tuple(replicates) if replicates else None
        self.depths = dict(depths) if depths else None
        self.random_init = random_init
        self.under_test = under_test
        self.notes = tuple(notes)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (
            f"RoleAssignment(endpoint={self.endpoint!r}, dilution={self.dilution!r}, "
            f"ceiling={self.ceiling!r}, by_params={self.by_params!r}, "
            f"replicates={self.replicates!r}, depths={self.depths!r}, "
            f"random_init={self.random_init is not None})"
        )


def assign_roles(
    scored: Iterable[ScoredCheckpoint], *, endpoint: str, row: Optional[str] = None
) -> RoleAssignment:
    """
    Recognise the gate roles among a set of scored checkpoints.

    :param scored: Scored checkpoints, from :mod:`factcrowd.score_run`. Intermediate steps are ignored;
        each ``(cell, replicate)`` contributes its last one.
    :param endpoint: Which endpoint's accuracy the gates are about, e.g. ``"mano"``.
    :param row: Restrict the ladder and the ceiling to one ladder row. Defaults to the row the ladder
        was found on, which is the usual case and the one that needs no argument.

    :returns: The assignment.

    :raises OLMoConfigurationError: If a dilution ladder spans more than one row, since two rows'
        ladders interleaved by dose is not a ladder.
    """
    from .gates import DILUTION_DOSES_PCT

    scored_list = list(scored)
    final = _final(scored_list)
    notes: List[str] = []

    # --- G8: the dilution ladder -------------------------------------------------------------------
    ladder: Dict[int, float] = {}
    ladder_rows = set()
    for (cell_id, _), entry in sorted(final.items()):
        match = DILUTION_CELL_PATTERN.match(cell_id)
        if match is None:
            continue
        accuracy = _accuracy(entry, endpoint)
        if accuracy is None:
            notes.append(
                f"{cell_id} looks like a ladder arm but has no {endpoint!r} score; skipped"
            )
            continue
        ladder_rows.add(entry.stated("row"))
        ladder[int(match.group("dose"))] = accuracy
    if len(ladder_rows) > 1:
        raise OLMoConfigurationError(
            f"the dilution ladder spans rows {sorted(ladder_rows)}. G8 reads one dose-response curve; "
            f"arms from two widths interleaved by dose is two curves averaged into one."
        )
    ladder_row = next(iter(ladder_rows), None)
    if row is None:
        row = ladder_row
    if ladder:
        missing = [dose for dose in DILUTION_DOSES_PCT if dose not in ladder]
        notes.append(
            f"G8: ladder on row {ladder_row} with doses {sorted(ladder, reverse=True)}"
            + (f", missing {missing}" if missing else " (complete)")
        )

    # --- G4, G6, G7: the controls ------------------------------------------------------------------
    accuracies_by_params: Dict[int, List[float]] = {}
    ceilings: List[float] = []
    replicates_by_cell: Dict[str, List[EndpointResult]] = {}
    for (cell_id, _), entry in sorted(final.items()):
        result = _result(entry, endpoint)
        if result is None:
            continue
        if cell_id.endswith(_CONTROL_SUFFIX):
            # One point per width, **averaged over that width's replicates**. G6 asks whether accuracy
            # rises with width at fixed depth, and reading one arbitrary seed per width lets seed noise
            # invert the ordering the gate is checking -- which matters here because the sigma block runs
            # three replicates of exactly these cells, so the better estimate is free.
            params = int(entry.stated("non_embedding_params"))
            accuracies_by_params.setdefault(params, []).append(result.accuracy)
            if row is not None and entry.stated("row") == row:
                ceilings.append(result.accuracy)
        # The whole result, not its accuracy: G7 caps the unparseable rate too.
        replicates_by_cell.setdefault(cell_id, []).append(result)
    by_params: Dict[int, float] = {
        params: sum(values) / len(values) for params, values in accuracies_by_params.items()
    }
    ceiling: Optional[float] = sum(ceilings) / len(ceilings) if ceilings else None
    if by_params:
        seen = {params: len(v) for params, v in accuracies_by_params.items()}
        notes.append(
            f"G6: controls at {len(by_params)} width(s) {sorted(by_params)}, "
            f"{'x'.join(str(seen[p]) for p in sorted(seen))} replicate(s) each"
        )
    if ceiling is not None:
        notes.append(
            f"G4: ceiling from the row-{row} control, {100 * ceiling:.2f}% "
            f"(mean of {len(ceilings)})"
        )

    replicated = {cell: scores for cell, scores in replicates_by_cell.items() if len(scores) > 1}
    replicates: Optional[List[EndpointResult]] = None
    if replicated:
        # The most-replicated cell. G7 is a statement about run-to-run sigma at one configuration, so
        # pooling several cells' replicates would confound the seed spread with the treatment.
        cell, scores = max(replicated.items(), key=lambda item: len(item[1]))
        replicates = scores
        notes.append(f"G7: {len(scores)} replicates of {cell}")
        if len(replicated) > 1:
            notes.append(
                f"G7: {sorted(set(replicated) - {cell})} also have replicates and were not used; "
                f"sigma is reported for one configuration at a time"
            )

    # --- G1: the task-depth sweep ------------------------------------------------------------------
    # This is what `configs/cells/calibration` is for, and until this block existed running that sweep
    # would not have closed the gate: `run_gates` has taken `depth_scores` all along and nothing supplied
    # it, so G1 reported as owed however much calibration evidence sat in the CSV.
    wanted_prefix = _DEPTH_PREFIX_FOR_ENDPOINT.get(endpoint, ...)
    depths: Dict[int, float] = {}
    depth_rows = set()
    if wanted_prefix is not ...:
        for (cell_id, _), entry in sorted(final.items()):
            match = DEPTH_CELL_PATTERN.match(cell_id)
            if match is None or match.group("prefix") != wanted_prefix:
                continue
            accuracy = _accuracy(entry, endpoint)
            if accuracy is None:
                notes.append(
                    f"{cell_id} looks like a depth arm but has no {endpoint!r} score; skipped"
                )
                continue
            depth_rows.add(entry.stated("row"))
            depths[int(match.group("depth"))] = accuracy
    if len(depth_rows) > 1:
        raise OLMoConfigurationError(
            f"the depth sweep spans rows {sorted(depth_rows)}. G1 reads one accuracy-against-depth "
            f"curve, and arms from two widths interleaved by depth is two curves averaged into one -- "
            f"which is also what would let a wider row's success stand in for the treatment's."
        )
    if depths:
        notes.append(
            f"G1: depth sweep on row {next(iter(depth_rows))} at depths "
            f"{sorted(depths)} ({endpoint!r})"
        )
    elif wanted_prefix is not ...:
        notes.append(f"G1: no depth sweep found for {endpoint!r}; the gate will report it as owed")

    # --- G2: the untrained checkpoint --------------------------------------------------------------
    # Read from the raw sequence rather than `final`, which keeps the *last* checkpoint per cell and so
    # discards step 0 wherever a run trained at all. Nearly free evidence that was overlooked throughout:
    # `CheckpointerCallback.pre_train_checkpoint` writes step 0 on every run there has ever been.
    random_init: Optional[EndpointResult] = None
    for entry in sorted(scored_list, key=lambda e: (e.stated("cell_id", ""), e.ref.step)):
        if entry.ref.step != 0:
            continue
        candidate = _result(entry, endpoint)
        if candidate is not None:
            random_init = candidate
            notes.append(f"G2: untrained checkpoint from {entry.stated('cell_id')} at step 0")
            break
    if random_init is None:
        notes.append(
            "G2: no step-0 checkpoint carries this endpoint; the gate will report it as owed"
        )

    # --- the cell under test ----------------------------------------------------------------------
    # The highest-demand non-ladder, non-control cell on the row: the gates ask whether the endpoint can
    # resolve an effect where the design most needs it to.
    candidates = [
        entry
        for (cell_id, _), entry in sorted(final.items())
        if not cell_id.endswith(_CONTROL_SUFFIX) and DILUTION_CELL_PATTERN.match(cell_id) is None
        # A calibration arm is evidence *for* the gates, never the cell they admit: it carries no facts,
        # so admitting on one would be admitting a demand-0 run as though it were a treatment.
        and DEPTH_CELL_PATTERN.match(cell_id) is None
        and (row is None or entry.stated("row") == row)
        and _accuracy(entry, endpoint) is not None
    ]
    under_test: Optional[EndpointResult] = None
    chosen: Optional[ScoredCheckpoint] = None
    if candidates:
        chosen = max(candidates, key=lambda e: float(e.stated("demand_bits_per_param", 0.0)))
        under_test = _result(chosen, endpoint)
        notes.append(f"endpoint under test: {chosen.stated('cell_id')} at step {chosen.ref.step}")
    elif ladder:
        # A gate run with no confirmatory cell scored yet: read the ladder's reference arm, which is the
        # same width and the same endpoint. Better than refusing to report at all during M0.
        reference = final.get((f"{str(row).lower()}_dil100", 0))
        if reference is not None:
            chosen = reference
            under_test = _result(reference, endpoint)
            notes.append(
                "endpoint under test: the ladder's 100% arm (no confirmatory cell scored yet)"
            )

    return RoleAssignment(
        endpoint=endpoint,
        result=under_test,
        dilution=ladder or None,
        ceiling=ceiling,
        by_params=by_params or None,
        replicates=replicates,
        depths=depths or None,
        random_init=random_init,
        under_test=chosen,
        notes=notes,
    )


def assemble(
    scored: Iterable[ScoredCheckpoint],
    *,
    endpoint: str,
    row: Optional[str] = None,
    commit: Optional[str] = None,
) -> Tuple[GateReport, RoleAssignment]:
    """
    Score the gates against whatever evidence the runs provide, and return the report.

    :param scored: Scored checkpoints.
    :param endpoint: The endpoint to admit.
    :param row: Restrict to one ladder row; defaults to the ladder's own row.
    :param commit: The commit the evidence was produced at, recorded in the report.

    :returns: The report and the assignment it was built from.

    :raises OLMoConfigurationError: If no checkpoint carries the endpoint at all, since a report about
        an endpoint nothing measured would admit rows on the strength of an empty file.
    """
    scored_rows = list(scored)
    assignment = assign_roles(scored_rows, endpoint=endpoint, row=row)
    if assignment.result is None:
        raise OLMoConfigurationError(
            f"no scored checkpoint carries endpoint {endpoint!r}, so there is nothing to run the gates "
            f"against. Score a run of the cell being admitted, or its dilution ladder, first."
        )
    identity = (
        {}
        if assignment.under_test is None
        else _identity_of(assignment.under_test, endpoint=endpoint)
    )
    results = run_gates(
        assignment.result,
        depth_scores=assignment.depths,
        random_init_result=assignment.random_init,
        achievable_ceiling=assignment.ceiling,
        scores_by_params=assignment.by_params,
        replicates=assignment.replicates,
        dilution_scores=assignment.dilution,
    )
    return (
        GateReport(
            version=GATE_REPORT_VERSION,
            endpoint=endpoint,
            results=results,
            commit=commit or "",
            identity=identity,
        ),
        assignment,
    )


def _identity_of(entry: ScoredCheckpoint, *, endpoint: str) -> Dict[str, str]:
    """
    What the evidence is *about*, for :meth:`factcrowd.measure.gates.GateReport.mismatch`.

    **Read from the cell under test, not from every checkpoint the report saw.** An earlier version took the
    fields all the endpoint's checkpoints agreed on, which sounds safer and is useless: the depth sweep
    varies depth by construction and the width sweep varies the row, so the two fields most worth binding
    are exactly the two that never agree. The report's claim is "this endpoint is a usable instrument at the
    configuration I was built around", and that configuration is one cell.

    :param entry: The checkpoint the gates were run against.
    :param endpoint: The endpoint being admitted, which selects whose depth is bound.

    :returns: The identity fields, as strings. A field the cell does not state is omitted.
    """
    fields = list(IDENTITY_FIELDS)
    depth = _DEPTH_FIELD_FOR_ENDPOINT.get(endpoint)
    if depth is not None:
        fields.append(depth)
    identity: Dict[str, str] = {}
    for key in fields:
        value = entry.stated(key)
        if value is not None:
            identity[key] = str(value)
    return identity
