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
from typing import List, Optional, Sequence, Tuple

if __package__ in (None, ""):  # pragma: no cover - only when run as a script
    _here = Path(__file__).resolve()
    sys.path.insert(0, str(_here.parents[1]))
    try:
        import olmo_core  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(_here.parents[3]))

from factcrowd.measure import bits as bits_module  # noqa: E402
from factcrowd.measure import checkpoint as checkpoint_module  # noqa: E402
from factcrowd.measure import collect as collect_module  # noqa: E402
from factcrowd.measure import reasoning as reasoning_module  # noqa: E402
from factcrowd.measure import recall as recall_module  # noqa: E402

from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402

log = logging.getLogger(__name__)


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


def score_checkpoint(
    ref: checkpoint_module.CheckpointRef,
    *,
    work_dir: Path,
    device: str,
    eval_items: int,
    bit_entities: int,
    batch_size: int,
) -> collect_module.ScoredCheckpoint:
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
    loaded = checkpoint_module.load(ref, work_dir=work_dir, device=device, verify=True)
    forward = checkpoint_module.forward_fn(loaded.model, device=device)

    endpoints = [
        reasoning_module.score_reasoning(task, forward, n_items=eval_items, batch_size=batch_size)
        for task in loaded.corpus.tasks
    ]
    achieved = bits_module.score_checkpoint(
        loaded, forward, n_entities=bit_entities, batch_size=batch_size
    )
    if achieved is not None:
        # A measurement above the capacity ceiling is a fault, not a finding -- fail here rather than
        # let it into the table.
        achieved.check_against_capacity()

    recalls = recall_module.score_recall(
        loaded, forward, n_entities=bit_entities, batch_size=batch_size
    )
    recall_row: dict = {}
    for result in recalls:
        recall_row.update({key[len("recall_") :]: value for key, value in result.summary().items()})

    return collect_module.ScoredCheckpoint(
        ref=ref,
        cell=dict(loaded.record["cell"]),
        resolved=dict(loaded.record.get("resolved") or {}),
        endpoints=endpoints,
        achieved=achieved,
        recall=recall_row,
        extra={
            "schema_fingerprint": (loaded.record.get("fingerprints") or {}).get("schema", "")[:16],
            "eval_items": eval_items,
            "bit_entities": bit_entities,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Entry point.

    :param argv: Argument list, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix", required=True, help="Checkpoint directory, or a fan-out's parent"
    )
    parser.add_argument("--out", required=True, help="Where to write the CSV")
    parser.add_argument("--work-dir", default="/tmp/factcrowd-score", help="Local scratch")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--eval-items", type=int, default=2_000, help="Held-out items per endpoint")
    parser.add_argument(
        "--bit-entities", type=int, default=2_000, help="Entities for the bit count"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated steps to score. Every checkpoint by default.",
    )
    parser.add_argument("--json", action="store_true", help="Also print each row as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    wanted = {int(step) for step in args.steps.split(",") if step.strip()}
    prefixes = cell_prefixes(args.prefix)
    if not prefixes:
        raise OLMoConfigurationError(
            f"no checkpoints under {args.prefix}. A run that died before its first save writes none, "
            f"and a fan-out keeps them under cell-N/checkpoints -- check one level down."
        )
    log.info("scoring %d cell prefix(es) under %s", len(prefixes), args.prefix)

    scored: List[collect_module.ScoredCheckpoint] = []
    for cell_prefix in prefixes:
        refs = checkpoint_module.find_checkpoints(cell_prefix)
        for ref in refs:
            if wanted and ref.step not in wanted:
                continue
            log.info("  %s step %d", cell_prefix, ref.step)
            scored.append(
                score_checkpoint(
                    ref,
                    work_dir=work_dir / f"step{ref.step}",
                    device=args.device,
                    eval_items=args.eval_items,
                    bit_entities=args.bit_entities,
                    batch_size=args.batch_size,
                )
            )

    rows = collect_module.collect(scored)
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
