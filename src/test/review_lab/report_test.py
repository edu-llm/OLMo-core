import json

from olmo_core.review_lab.report import build_rows, write_report


def _write_run(root, condition: str, seed: int, old_delta: float, new_loss: float):
    run = root / condition / f"seed_{seed}"
    run.mkdir(parents=True)
    baseline = {"registry": 1.0, "routing": 1.0}
    final = {skill: loss + old_delta for skill, loss in baseline.items()}
    summary = {
        "condition": condition,
        "seed": seed,
        "review_events": 3,
        "baseline_old_loss": baseline,
        "final_old_loss": final,
        "mean_old_loss_delta": old_delta,
        "final_new_loss": {"registry": new_loss, "routing": new_loss},
        "mean_new_loss": new_loss,
        "duration_seconds": 10.0,
        "review_steps": [1, 4, 7],
    }
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    metrics = [
        {
            "event": "eval",
            "stage": 2,
            "step": 3,
            "old_loss": {skill: loss + old_delta / 2 for skill, loss in baseline.items()},
        },
        {"event": "eval", "stage": 2, "step": 7, "old_loss": final},
    ]
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in metrics), encoding="utf-8"
    )


def test_report_builds_seed_rows_and_deliverables(tmp_path):
    _write_run(tmp_path, "uniform", 17, old_delta=0.2, new_loss=1.1)
    _write_run(tmp_path, "adaptive_due", 17, old_delta=0.1, new_loss=1.0)

    rows = build_rows(tmp_path)
    assert len(rows) == 2
    assert {row["condition"] for row in rows} == {"uniform", "adaptive_due"}
    report_path = write_report(tmp_path)
    assert report_path.exists()
    assert (tmp_path / "results_by_seed.csv").exists()
    assert (tmp_path / "results_aggregate.csv").exists()
    report = report_path.read_text(encoding="utf-8")
    assert "Best equal-weight joint result: **adaptive_due**" in report
