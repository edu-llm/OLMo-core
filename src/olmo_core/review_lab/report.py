from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def _load_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in sorted(run_dir.glob("*/seed_*/summary.json")):
        if path.parts[-3] == "stage1":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["summary_path"] = str(path)
        summaries.append(payload)
    return summaries


def _retention_auc(metrics_path: Path, baseline: Mapping[str, float]) -> float:
    deltas: List[float] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") != "eval" or payload.get("stage") != 2:
                continue
            old_loss = payload["old_loss"]
            deltas.append(_mean(float(old_loss[skill]) - baseline[skill] for skill in baseline))
    return _mean(deltas)


def build_rows(run_dir: str | Path) -> List[Dict[str, Any]]:
    root = Path(run_dir)
    rows: List[Dict[str, Any]] = []
    for summary in _load_summaries(root):
        summary_path = Path(summary.pop("summary_path"))
        mean_final_old = _mean(float(value) for value in summary["final_old_loss"].values())
        mean_new = float(summary["mean_new_loss"])
        rows.append(
            {
                "condition": summary["condition"],
                "seed": int(summary["seed"]),
                "review_events": int(summary["review_events"]),
                "mean_old_loss_delta": float(summary["mean_old_loss_delta"]),
                "mean_buffer_old_loss_delta": float(summary.get("mean_buffer_old_loss_delta", 0.0)),
                "retention_auc_loss_delta": _retention_auc(
                    summary_path.parent / "metrics.jsonl",
                    {key: float(value) for key, value in summary["baseline_old_loss"].items()},
                ),
                "mean_final_old_loss": mean_final_old,
                "mean_new_loss": mean_new,
                "joint_loss": 0.5 * (mean_final_old + mean_new),
                "mastered_facts": int(summary.get("mastered_facts", 0)),
                "exact_regression_rate": float(summary.get("exact_regression_rate", math.nan)),
                "loss_regression_rate": float(summary.get("loss_regression_rate", math.nan)),
                "final_old_exact_match": float(summary.get("final_old_exact_match", math.nan)),
                "final_new_exact_match": float(summary.get("final_new_exact_match", math.nan)),
                "old_content_tokens": int(summary.get("old_content_tokens", 0)),
                "new_content_tokens": int(summary.get("new_content_tokens", 0)),
                "old_stream_digest": str(summary.get("old_stream_digest", "")),
                "new_stream_digest": str(summary.get("new_stream_digest", "")),
                "duration_seconds": float(summary["duration_seconds"]),
                "review_steps": " ".join(str(step) for step in summary["review_steps"]),
            }
        )
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    aggregate: List[Dict[str, Any]] = []
    metrics = (
        "mean_old_loss_delta",
        "mean_buffer_old_loss_delta",
        "retention_auc_loss_delta",
        "mean_final_old_loss",
        "mean_new_loss",
        "joint_loss",
        "duration_seconds",
    )
    optional_metrics = (
        "exact_regression_rate",
        "loss_regression_rate",
        "final_old_exact_match",
        "final_new_exact_match",
    )
    for condition, condition_rows in sorted(grouped.items()):
        item: Dict[str, Any] = {"condition": condition, "seeds": len(condition_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in condition_rows]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        for metric in optional_metrics:
            values = [
                float(row[metric]) for row in condition_rows if not math.isnan(float(row[metric]))
            ]
            if values:
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        item["review_events_mean"] = _mean(float(row["review_events"]) for row in condition_rows)
        aggregate.append(item)
    return aggregate


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, aggregate: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    conditions = [str(row["condition"]) for row in aggregate]
    old_delta = [float(row["mean_old_loss_delta_mean"]) for row in aggregate]
    new_loss = [float(row["mean_new_loss_mean"]) for row in aggregate]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(conditions, old_delta, color="#4C78A8")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Final forgetting (lower is better)")
    axes[0].set_ylabel("Old-skill loss increase")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(conditions, new_loss, color="#F58518")
    axes[1].set_title("New-skill adaptation (lower is better)")
    axes[1].set_ylabel("New-skill answer loss")
    axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_report(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    rows = build_rows(root)
    if not rows:
        raise ValueError(f"No completed condition runs found under {root}")
    aggregate = aggregate_rows(rows)
    _write_csv(root / "results_by_seed.csv", rows)
    _write_csv(root / "results_aggregate.csv", aggregate)
    _write_plot(root / "retention_vs_adaptation.png", aggregate)

    best_joint = min(aggregate, key=lambda row: float(row["joint_loss_mean"]))
    lines = [
        "# OLMo review-scheduler experiment",
        "",
        "Lower values are better. `old Δ` is the increase in held-out old-skill answer loss ",
        "relative to the shared stage-one checkpoint. `trajectory Δ` averages that increase ",
        "through stage two. `joint` gives old and new final loss equal weight.",
        "",
        "`buffer Δ` isolates the change in old-fact loss during the final no-review buffer; ",
        "positive values mean actual delayed forgetting during that period.",
        "",
        "| Condition | Seeds | Reviews | Old Δ | Buffer Δ | Trajectory Δ | New loss | Joint loss | Regression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(aggregate, key=lambda item: float(item["joint_loss_mean"])):
        lines.append(
            "| {condition} | {seeds} | {reviews:.1f} | {old:.4f} ± {old_std:.4f} | "
            "{buffer:.4f} ± {buffer_std:.4f} | "
            "{auc:.4f} ± {auc_std:.4f} | {new:.4f} ± {new_std:.4f} | "
            "{joint:.4f} ± {joint_std:.4f} | {regression} |".format(
                condition=row["condition"],
                seeds=row["seeds"],
                reviews=float(row["review_events_mean"]),
                old=float(row["mean_old_loss_delta_mean"]),
                old_std=float(row["mean_old_loss_delta_std"]),
                buffer=float(row["mean_buffer_old_loss_delta_mean"]),
                buffer_std=float(row["mean_buffer_old_loss_delta_std"]),
                auc=float(row["retention_auc_loss_delta_mean"]),
                auc_std=float(row["retention_auc_loss_delta_std"]),
                new=float(row["mean_new_loss_mean"]),
                new_std=float(row["mean_new_loss_std"]),
                joint=float(row["joint_loss_mean"]),
                joint_std=float(row["joint_loss_std"]),
                regression=(
                    f"{100 * float(row['exact_regression_rate_mean']):.1f}%"
                    if "exact_regression_rate_mean" in row
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            f"Best equal-weight joint result: **{best_joint['condition']}**.",
            "",
            "With three paired seeds, treat this as a directional pilot rather than a decisive ",
            "spacing result. Prioritize buffer Δ and final delayed retention; use trajectory ",
            "AUC only as a secondary measure of usefulness during training.",
        ]
    )
    report_path = root / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
