#!/usr/bin/env python
"""Log per-checkpoint eval metrics to Weights & Biases — general, reused for any eval.

Training already streams loss/LR to W&B; the *eval-suite* metrics (KL, retention,
instruction-following, pedagogy) are computed after training and are NOT auto-logged.
This tool pushes them to W&B, keyed by training step, so the whole KL-forgetting /
RL's-Razor comparison lives in one project.

It is deliberately schema-agnostic: give it a JSON list of per-checkpoint points, each
with a ``step`` (or ``point`` like "c16") and any numeric metric fields. Works for our
own eval and, later, for eval numbers handed over by other teams — same call.

    # log a run's curve (resumes the training run if you pass its --run_id)
    python wandb_eval.py --summary out/impl3-a-T2/master_summary.json \
        --project edullm-p7 --run_name impl3-a-T2-eval \
        --figure out/impl3-a-T2/figures/fig_kl_forgetting.png

Two shapes are supported and detected automatically:

* **trajectory** — points carry a ``step`` (or a "c16"-style ``point``): logged as stepped
  metrics so W&B draws a curve, and the last point lands in the run summary.
* **sweep** — points are whole runs ("impl3-a-T4") with no ``step``: logged as a single
  ``wandb.Table`` so the runs stay sortable/comparable. Scraping digits out of the labels
  would invent steps like 332 for "impl3-a-T32", so stepped logging is skipped entirely.

Point schema (extra keys are logged as-is):
    [{"point":"base","step":0,"kl_new":0.0,"acc":0.42,"forget":0.0},
     {"point":"c16","step":16,"kl_new":0.31,"acc":0.33,"forget":0.09,"pedagogy":7.1}, ...]
"""
import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, help="JSON list of per-checkpoint metric dicts.")
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "edullm-p7"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--run_name", default=None, help="W&B run name (default: summary file's dir).")
    p.add_argument("--run_id", default=None, help="Resume this exact training run id (else a new run).")
    p.add_argument("--figure", nargs="*", default=[], help="Optional image(s) to attach.")
    p.add_argument("--step_key", default="step", help="Field to use as the W&B step.")
    return p.parse_args()


def _step_of(pt, step_key):
    if step_key in pt and pt[step_key] is not None:
        return int(pt[step_key])
    label = str(pt.get("point", "0"))
    return 0 if label == "base" else int("".join(c for c in label if c.isdigit()) or 0)


def _has_steps(points, step_key):
    """True only if the points really are a trajectory, not a sweep over separate runs."""
    return any(pt.get(step_key) is not None for pt in points)


def main():
    args = parse_args()
    import wandb

    points = json.load(open(args.summary))
    if not isinstance(points, list):
        raise SystemExit("--summary must be a JSON list of per-checkpoint dicts")
    trajectory = _has_steps(points, args.step_key)
    points = sorted(points, key=(lambda pt: _step_of(pt, args.step_key)) if trajectory
                    else (lambda pt: str(pt.get("point", ""))))

    run_name = args.run_name or os.path.basename(os.path.dirname(os.path.abspath(args.summary))) or "eval"
    run = wandb.init(project=args.project, entity=args.entity, id=args.run_id,
                     name=run_name, resume="allow", job_type="eval")

    metric_keys = set()
    if trajectory:
        for pt in points:
            step = _step_of(pt, args.step_key)
            metrics = {f"eval/{k}": v for k, v in pt.items()
                       if k not in ("point", args.step_key) and isinstance(v, (int, float))}
            metric_keys.update(metrics)
            if metrics:
                wandb.log(metrics, step=step)
        # final-checkpoint values as run summary (handy for the sweep table / parallel-coords)
        last = points[-1]
        for k, v in last.items():
            if k not in ("point", args.step_key) and isinstance(v, (int, float)):
                run.summary[f"final/{k}"] = v
    else:
        columns = ["point"] + sorted({k for pt in points for k, v in pt.items()
                                      if k != "point" and isinstance(v, (int, float))})
        metric_keys.update(columns[1:])
        table = wandb.Table(columns=columns,
                            data=[[pt.get("point")] + [pt.get(c) for c in columns[1:]] for pt in points])
        wandb.log({"eval/sweep": table})
        run.summary["eval/n_points"] = len(points)

    for path in args.figure:
        if os.path.exists(path):
            wandb.log({f"eval/{os.path.splitext(os.path.basename(path))[0]}": wandb.Image(path)})
        else:
            print(f"[warn] figure not found, skipped: {path}")

    print(f"logged {len(points)} points as a {'trajectory' if trajectory else 'sweep table'} "
          f"({sorted(metric_keys)}) to {args.project}/{run_name}"
          + (f" [resumed {args.run_id}]" if args.run_id else ""))
    run.finish()


if __name__ == "__main__":
    main()
