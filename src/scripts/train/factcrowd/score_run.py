"""
Score a finished run: every checkpoint, every endpoint, one table out.

    python src/scripts/train/factcrowd/score_run.py \\
        --prefix s3://.../runs/$RUN_ID/checkpoints \\
        --out s3://.../runs/$RUN_ID/scores.csv

The second platform entry point, and deliberately not a callback. PRD 8.2: generation re-parallelises the
model and mutates KV-cache state, so recall cannot live inside training; and a bit count over ten
checkpoints of a finished cell costs a fraction of what re-running the cell would. So measurement is a
separate, cheap, single-process job that reads what training wrote.

It is single-process on purpose. ``load_model_and_optim_state`` reshards a checkpoint saved on four ranks
into one unsharded model with no process group, so there is nothing to distribute -- scoring 120
checkpoints is minutes of forward passes and is dominated by pulling shards from S3.

A fan-out run keeps each cell under its own ``cell-N/`` prefix, so ``--prefix`` can name either one cell's
checkpoint directory or the parent of several; :func:`cell_prefixes` finds them either way.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - only when run as a script
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[1]))
    try:
        import olmo_core  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_here.parents[3]))

from factcrowd import provenance as provenance_module  # noqa: E402
from factcrowd.corpus import values as values_module  # noqa: E402
from factcrowd.ladder import rho as rho_module  # noqa: E402
from factcrowd.measure import bits as bits_module  # noqa: E402
from factcrowd.measure import checkpoint as checkpoint_module  # noqa: E402
from factcrowd.measure import collect as collect_module  # noqa: E402
from factcrowd.measure import evidence as evidence_module  # noqa: E402
from factcrowd.measure import gates as gates_module  # noqa: E402
from factcrowd.measure import reasoning as reasoning_module  # noqa: E402
from factcrowd.measure import recall as recall_module  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402

log = logging.getLogger(__name__)


def _write_text(text: str, target: str, work_dir: Path) -> None:
    """
    Write text to a local path or upload it to a URL.

    :param text: What to write.
    :param target: A local path or a URL.
    :param work_dir: Somewhere to stage a remote write from.
    """
    from olmo_core.io import is_url, upload

    if not is_url(target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(text)
        return
    staged = work_dir / Path(target).name
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(text)
    upload(staged, target, save_overwrite=True)


def _torch_dtype(name: str) -> Any:
    """
    Resolve a dtype name to a ``torch.dtype``.

    Imported here rather than at module scope so ``--help`` and the argument parsing still work on a
    machine with no torch, which is what makes a dry read of this program cheap.

    :param name: ``"float32"``, ``"bfloat16"`` or ``"float16"``.

    :returns: The dtype.
    """
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def cell_prefixes(prefix: str) -> Tuple[str, ...]:
    """
    Every checkpoint directory under ``prefix``, whether it is one cell's or a fan-out's parent.

    A single-cell run writes ``<run>/checkpoints/step*``; a fan-out writes
    ``<run>/cell-N/checkpoints/step*``. Rather than make the caller know which, look for checkpoints here
    and fall back to looking one level down.

    :param prefix: A checkpoint directory or the parent of several.

    :returns: The directories that actually contain checkpoints.
    """
    from olmo_core.io import list_directory

    def has_checkpoints(candidate: str) -> bool:
        # A missing directory raises rather than returning empty, and a fan-out is full of siblings that
        # legitimately have no checkpoints subdir -- a cell that died before its first save, `logs/`,
        # `wandb/`. Letting that abort scoring for every *other* cell is the wrong trade.
        try:
            return bool(checkpoint_module.find_checkpoints(candidate))
        except (FileNotFoundError, NotADirectoryError):
            return False

    if has_checkpoints(prefix):
        return (prefix,)
    out: List[str] = []
    try:
        children = list(list_directory(prefix, include_files=False))
    except (FileNotFoundError, NotADirectoryError):
        return ()
    for child in children:
        for candidate in (child, f"{str(child).rstrip('/')}/checkpoints"):
            if has_checkpoints(candidate):
                out.append(candidate)
                break
    return tuple(out)


class Scored(NamedTuple):
    """One checkpoint's scores, plus the corpus so the next checkpoint of the same cell can reuse it."""

    scored: collect_module.ScoredCheckpoint
    corpus: Any


def _slug(prefix: str) -> str:
    """A filesystem-safe directory name for a cell prefix, so cells do not share scratch."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in prefix)[-80:]


def score_checkpoint(
    ref: checkpoint_module.CheckpointRef,
    *,
    dtype: Optional[Any] = None,
    work_dir: Path,
    device: str,
    eval_items: int,
    bit_entities: int,
    bit_offset: int = 0,
    batch_size: int,
    corpus: Any = None,
) -> Scored:
    """
    Load one checkpoint and measure everything on it.

    :param ref: Which checkpoint.
    :param work_dir: Local scratch.
    :param device: Where to run the model.
    :param eval_items: Held-out items per reasoning endpoint.
    :param bit_entities: Entities sampled for the bit count and the recall probe.
    :param batch_size: Sequences per forward pass.

    :returns: The scored checkpoint, ready for :func:`factcrowd.measure.collect.collect`.
    """
    loaded = checkpoint_module.load(
        ref, work_dir=work_dir, device=device, verify=True, corpus=corpus, dtype=dtype
    )
    forward = checkpoint_module.forward_fn(loaded.model, device=device)

    endpoints = [
        reasoning_module.score_reasoning(task, forward, n_items=eval_items, batch_size=batch_size)
        for task in loaded.corpus.tasks
    ]
    achieved = bits_module.score_checkpoint(
        loaded,
        forward,
        n_entities=bit_entities,
        entity_offset=bit_offset,
        batch_size=batch_size,
    )
    warning = None
    if achieved is not None:
        # `achieved <= demanded` is a theorem about the dataset, so a violation is a fault and stops the
        # run. Passing the *published* ~2 bits/param estimate is not: three entropy cells demand more than
        # that, so refusing it would censor the finding. It is recorded on the row instead.
        achieved.check_against_demand(
            rho_module.demanded_bits(
                loaded.resolved.n_entities,
                loaded.cell.bits_per_entity,
                name_space=values_module.NAME_SPACE,
            )
        )
        warning = achieved.capacity_warning()
        if warning:
            log.warning("  %s", warning)

    recalls = recall_module.score_recall(
        loaded,
        forward,
        n_entities=bit_entities,
        entity_offset=bit_offset,
        batch_size=batch_size,
    )
    recall_row: dict = {}
    for result in recalls:
        # Verbatim. Stripping a fixed `recall_` prefix here silently mangled every column once the keys
        # were renamed to `template_*`: `"template_all_chance"[7:]` is `"e_all_chance"`.
        recall_row.update(result.summary())

    return Scored(
        corpus=loaded.corpus,
        scored=collect_module.ScoredCheckpoint(
            ref=ref,
            cell=dict(loaded.record["cell"]),
            resolved=dict(loaded.record.get("resolved") or {}),
            endpoints=endpoints,
            achieved=achieved,
            recall=recall_row,
            extra={
                "schema_fingerprint": (loaded.record.get("fingerprints") or {}).get("schema", "")[
                    :16
                ],
                "eval_items": eval_items,
                "bit_entities": bit_entities,
                "bit_offset": bit_offset,
                "capacity_warning": warning or "",
                # The cell's own plan, so completeness is judged per cell rather than against a number
                # passed in from outside. `select_complete` reads it.
                "checkpoint_steps": list(loaded.record.get("checkpoint_steps") or []),
                # The commit that *trained* these weights, not the one scoring them. A table pooling two
                # revisions is a real risk on a grid this long, and it is invisible without this column.
                "train_commit": str((loaded.record.get("provenance") or {}).get("commit", "")),
                "train_dirty": str((loaded.record.get("provenance") or {}).get("dirty", "")),
            },
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        required=True,
        nargs="+",
        help="One or more checkpoint directories, or fan-out parents. Several because a gate report has "
        "to be assembled from evidence that does not share a parent: the sigma block and the dilution "
        "ladder are separate submissions, so a report built from one of them alone reports G8 missing "
        "while the ladder sits in the next prefix along.",
    )
    parser.add_argument("--out", required=True, help="Where to write the CSV")
    parser.add_argument("--work-dir", default="/tmp/factcrowd-score", help="Local scratch")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="Precision to score in. Named on the command line on purpose: the platform's precision "
        "guard reads the command's text and cannot see a dtype chosen in code, so a submission that "
        "does not say one can be admitted onto a card whose hardware lacks it. Scoring has always run "
        "in float32 -- training's bfloat16 was FSDP's param_dtype, a mixed-precision setting rather "
        "than the dtype the shards hold -- so float32 is the default and the honest word for it.",
    )
    parser.add_argument(
        "--eval-items",
        type=int,
        default=30_000,
        help="Held-out items per endpoint. 30,000 is what PRD 8.5's power calculation assumes; the "
        "earlier default of 2,000 quietly gave the endpoint 3.9x the measurement noise that figure was "
        "computed against.",
    )
    parser.add_argument(
        "--bit-entities",
        type=int,
        default=25_000,
        help="Entities for the bit count and the template probe. 25,000 matches the fixed probe subset "
        "PRD 3.3 holds constant across cells.",
    )
    parser.add_argument(
        "--bit-offset",
        type=int,
        default=0,
        help="First entity the bit and reconstruction cohorts sample. 0 -- the default, and what the "
        "first grid used -- is exactly the <compare> probe window (entities 0..24,999), which receives "
        "direct birth-year supervision: birth_year then reconstructs at 328x chance where the best other "
        "attribute is 1.1x. Pass 25000 for an uncontaminated cohort from the same checkpoints.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated steps to score. Every checkpoint by default.",
    )
    parser.add_argument(
        "--last-only",
        action="store_true",
        help="Score only each cell's final checkpoint. Ten times less work, and enough for every "
        "endpoint number: accuracy, achieved bits and template reconstruction are all read at the end "
        "of training. The trajectory is what the other nine give you, and a first read does not need it. "
        "A gate report only ever reads the last checkpoint anyway, so this changes nothing for one.",
    )
    parser.add_argument(
        "--gate-report",
        default="",
        help="JSON gate report from measure.gates. Without one, every row is written "
        "confirmatory=False: PRD 8.6 requires an endpoint to pass G1-G8 before it can be read, and that "
        "has to be enforced here rather than asserted in a document.",
    )
    parser.add_argument(
        "--write-gate-report",
        default="",
        help="After scoring, assemble a gate report from these runs and write it here. This is how a "
        "report comes to exist: point it at a scored M0 (the dilution ladder plus the controls) and it "
        "runs the gates on what it finds, reporting the rest as owed. Pass the file back as "
        "--gate-report when scoring the confirmatory grid.",
    )
    parser.add_argument(
        "--gate-endpoint",
        default="mano",
        help="Which endpoint --write-gate-report is about.",
    )
    parser.add_argument(
        "--expect-cells",
        type=int,
        default=0,
        help="Refuse to write the table unless exactly this many complete cells were scored. A "
        "confirmatory table is a claim about a whole grid, and nothing else notices a short one.",
    )
    parser.add_argument("--json", action="store_true", help="Also print each row as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    reports = gates_module.load_reports(args.gate_report) if args.gate_report else {}
    if not reports:
        log.warning(
            "no gate report given, so every row is marked confirmatory=False. An endpoint that has not "
            "passed G1-G8 can be plotted but not claimed (PRD 8.6)."
        )

    wanted = {int(step) for step in args.steps.split(",") if step.strip()}
    prefixes: List[str] = []
    for root in args.prefix:
        found = cell_prefixes(root)
        if not found:
            # Named and skipped rather than fatal. With several prefixes, one empty root should not cost
            # the others -- and a silent skip would let a typo read as "that submission scored nothing".
            log.warning("no checkpoints under %s; skipping it", root)
            continue
        prefixes.extend(found)
    if not prefixes:
        raise OLMoConfigurationError(
            f"no checkpoints under any of {list(args.prefix)}. A run that died before its first save "
            f"writes none, and a fan-out keeps them under cell-N/checkpoints -- check one level down."
        )
    log.info("scoring %d cell prefix(es) under %s", len(prefixes), args.prefix)

    scored: List[collect_module.ScoredCheckpoint] = []
    for cell_prefix in prefixes:
        refs = checkpoint_module.find_checkpoints(cell_prefix)
        # ONE CORPUS PER CELL, NOT PER CHECKPOINT. Only the weights differ between a cell's checkpoints,
        # and rebuilding its entity table ten times cost more than the scoring did. A separate work_dir
        # per step also defeated the offset-index cache, so nothing was reused at all.
        corpus = None
        cell_dir = work_dir / _slug(cell_prefix)
        if args.last_only and refs:
            # Per cell, not globally: the cells finish at different steps, so a single --steps list
            # cannot express "the end of each one".
            refs = (max(refs, key=lambda r: r.step),)
        for ref in refs:
            if wanted and ref.step not in wanted:
                continue
            log.info("  %s step %d", cell_prefix, ref.step)
            result = score_checkpoint(
                ref,
                work_dir=cell_dir,
                device=args.device,
                eval_items=args.eval_items,
                bit_entities=args.bit_entities,
                bit_offset=args.bit_offset,
                batch_size=args.batch_size,
                dtype=_torch_dtype(args.dtype),
                corpus=corpus,
            )
            corpus = result.corpus
            scored.append(result.scored)

    # BEFORE THE GATE REPORT, NOT AFTER. A gate must never read a partially-trained model. `assign_roles`
    # keeps the highest step per (cell_id, replicate), which happens to pick a re-run over the crash it
    # replaced -- but only where a re-run exists. A cell that crashed and was never re-run has one
    # checkpoint, that checkpoint is its highest, and it would feed G7's sigma or G4's ceiling as though
    # training had finished.
    scored, completeness = collect_module.select_complete(scored)
    for note in completeness:
        log.warning("  completeness: %s", note)
    # DISTINCT CELLS, NOT CHECKPOINTS. `select_complete` keeps a cell's whole trajectory, so without
    # --last-only `scored` holds about ten entries per cell and comparing its length to a cell count
    # would refuse every correct grid.
    cells_scored = {(str(e.stated("cell_id")), int(e.stated("replicate", 0))) for e in scored}
    if args.expect_cells and len(cells_scored) != args.expect_cells:
        raise OLMoConfigurationError(
            f"{len(cells_scored)} complete cells were scored but --expect-cells said "
            f"{args.expect_cells}. A table is a claim about a grid, so a short one is refused rather "
            f"than written. The notes above name what was dropped."
        )
    log.info("scored %d complete cell(s), %d checkpoint(s)", len(cells_scored), len(scored))

    if args.write_gate_report:
        # Assembled from the runs just scored, so the report cannot claim evidence that was not measured.
        # Written before the rows so a failure here does not leave a CSV that looks admitted.
        gate_report, assignment = evidence_module.assemble(
            scored, endpoint=args.gate_endpoint, commit=provenance_module.commit() or ""
        )
        for note in assignment.notes:
            log.info("  gate evidence: %s", note)
        # THE SAME is_url BRANCH `--out` ALREADY USED, AND THE ABSENCE OF IT HERE COST A REPORT.
        # `Path("s3://b/k").write_text(...)` writes to a local relative directory named `s3:`, so the M0
        # report went to container scratch and vanished while the log said it had been written.
        _write_text(json.dumps(gate_report.as_dict(), indent=2), args.write_gate_report, work_dir)
        verdict = (
            "passes" if gate_report.passed else f"does not pass: {', '.join(gate_report.failures)}"
        )
        log.info("gate report for %r %s -> %s", args.gate_endpoint, verdict, args.write_gate_report)
        # DELIBERATELY NOT FED BACK INTO THIS PASS. A report assembled from these runs must not admit
        # these runs: the gate cells would be admitting themselves, which is what an admission gate is
        # for preventing. Score the confirmatory grid separately and pass the file as --gate-report.
        if not args.gate_report:
            log.info(
                "these rows stay confirmatory=False: a report built from a run cannot admit that same "
                "run. Score the confirmatory grid with --gate-report %s.",
                args.write_gate_report,
            )

    rows = collect_module.collect(scored)
    # Admission is granted per endpoint by a report, or not at all. A row that cannot name a passing
    # report carries the reason, so a plot made from non-confirmatory rows says so on every point.
    for row in rows:
        endpoint = str(row.get("endpoint") or "")
        report = reports.get(endpoint)
        if report is None:
            row["confirmatory"] = False
            row["admission"] = f"no gate report for '{endpoint}'" if endpoint else "no endpoint"
        elif not report.passed:
            row["confirmatory"] = False
            row["admission"] = f"gates not passed: {', '.join(report.failures)}"
        else:
            row["confirmatory"] = True
            row["admission"] = f"{report.version} at {report.commit or 'unknown commit'}"
    if args.json:
        for row in rows:
            print(json.dumps(row))
    # A local target is written straight there; a remote one is staged and uploaded. `io.upload` only
    # handles URL schemes, so branching on `is_url` rather than on whether the paths differ is what keeps
    # `--out /some/path.csv` working -- the integration test found that the hard way.
    from olmo_core.io import is_url

    target = str(args.out)
    if is_url(target):
        from olmo_core.io import upload

        written = collect_module.write_csv(rows, Path(args.work_dir) / "scores.csv")
        upload(written, target, save_overwrite=True)
        log.info("wrote %d rows to %s (staged at %s)", len(rows), target, written)
    else:
        written = collect_module.write_csv(rows, Path(target))
        log.info("wrote %d rows to %s", len(rows), written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
