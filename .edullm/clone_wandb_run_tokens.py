"""Clone a finished W&B run into another project with scaled optimizer-step axes."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from wandb_token_axis import (
    clone_config,
    iter_clone_rows,
    merge_history_rows,
    merge_tags,
    parse_eval_artifact_name,
    parse_train_loss_jsonl,
    row_from_eval_task_loss,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="entity/project/run_id or wandb.ai URL")
    parser.add_argument("--project", required=True, help="destination project")
    parser.add_argument("--entity", default="eduLLM")
    parser.add_argument("--name", required=True, help="destination run name")
    parser.add_argument("--group", default=None)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--job-type", default="cloned-control")
    parser.add_argument("--notes", default=None)
    parser.add_argument(
        "--step-multiplier",
        type=int,
        default=1,
        help="multiply source history steps when logging the clone",
    )
    return parser.parse_args(argv)


def _parse_run_path(run_path: str) -> str:
    run_path = run_path.strip("/")
    if run_path.count("/") == 2:
        return run_path
    if "wandb.ai/" in run_path:
        _, tail = run_path.split("wandb.ai/", 1)
        entity, project, runs, run_id = tail.split("/", 3)
        if runs != "runs":
            raise ValueError(f"unsupported wandb URL: {run_path}")
        return f"{entity}/{project}/{run_id}"
    raise ValueError(f"could not parse run path: {run_path}")


def _config_value(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _rows_from_train_loss_artifact(run: Any, work_dir: Path) -> list[dict[str, Any]]:
    for artifact in run.logged_artifacts():
        if not artifact.name.startswith("train-loss:"):
            continue
        root = Path(artifact.download(root=str(work_dir / "train-loss")))
        rows: list[dict[str, Any]] = []
        for path in root.rglob("train_loss.jsonl"):
            rows.extend(parse_train_loss_jsonl(path.read_text(encoding="utf-8")))
        return rows
    return []


def _rows_from_eval_artifacts(
    run: Any,
    work_dir: Path,
    *,
    global_batch_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in run.logged_artifacts():
        history_step = parse_eval_artifact_name(artifact.name)
        if history_step is None:
            continue
        root = Path(artifact.download(root=str(work_dir / f"eval-{history_step:07d}")))
        for path in root.rglob("*_task_loss.json"):
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
            rows.append(
                row_from_eval_task_loss(
                    payload,
                    history_step=history_step,
                    global_batch_tokens=global_batch_tokens,
                )
            )
    return rows


def collect_source_rows(run: Any, work_dir: Path) -> list[dict[str, Any]]:
    """Merge W&B history, offline train-loss logs, and eval artifacts."""

    global_batch_tokens = int(_config_value(run.config, "global_batch_tokens", 4_194_304))
    train_rows = _rows_from_train_loss_artifact(run, work_dir)
    scan_rows = list(run.scan_history())
    eval_rows = _rows_from_eval_artifacts(
        run,
        work_dir,
        global_batch_tokens=global_batch_tokens,
    )
    return merge_history_rows((train_rows, scan_rows, eval_rows))


def clone_run(
    source_path: str,
    *,
    project: str,
    entity: str,
    name: str,
    group: str | None,
    tags: list[str],
    job_type: str,
    notes: str | None,
    step_multiplier: int,
) -> str:
    import wandb

    api = wandb.Api()
    source = api.run(_parse_run_path(source_path))
    with tempfile.TemporaryDirectory(prefix="wandb-clone-") as temporary:
        rows = collect_source_rows(source, Path(temporary))
        logged = iter_clone_rows(rows, step_multiplier=step_multiplier)
    if not logged:
        raise RuntimeError(f"source run has no step-indexed history: {source.url}")

    destination_tags = merge_tags(source.tags, ["cloned", *tags])
    destination_notes = notes or (
        f"Cloned from {source.url} with optimizer-step x-axis "
        f"(step_multiplier={step_multiplier})."
    )
    with wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=destination_tags,
        notes=destination_notes,
        job_type=job_type,
        config=clone_config(
            dict(source.config),
            cloned_from=source.url,
            step_multiplier=step_multiplier,
        ),
    ) as destination:
        for wandb_step, metrics in logged:
            wandb.log(metrics, step=wandb_step)
        destination_url = destination.url
    return destination_url


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    url = clone_run(
        args.source,
        project=args.project,
        entity=args.entity,
        name=args.name,
        group=args.group,
        tags=args.tag,
        job_type=args.job_type,
        notes=args.notes,
        step_multiplier=args.step_multiplier,
    )
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
