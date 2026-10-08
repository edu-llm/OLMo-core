from __future__ import annotations

import math
from pathlib import Path
import statistics

from .common import read_json, require, write_json


def paired_interval(values, confidence):
    from scipy.stats import t
    require(len(values) >= 2, "At least two complete independent replicate pairs are required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    radius = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * se
    return [mean - radius, mean + radius]


def paired_test(values, margin):
    from scipy.stats import t
    n = len(values)
    require(n >= 2, "At least two paired runs required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    se = sd / math.sqrt(n)
    ci90, ci95 = paired_interval(values, .9), paired_interval(values, .95)
    p = float(2 * t.sf(abs(mean / se), n - 1)) if se else (1.0 if mean == 0 else 0.0)
    return {"n": n, "paired_differences": values, "estimate": mean, "sd": sd, "ci90": ci90,
            "ci95": ci95, "p_two_sided": p,
            "loss_equivalent": ci90[0] > -margin and ci90[1] < margin,
            "useful_direction": "negative" if ci95[1] < -margin else "positive" if ci95[0] > margin else None}


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def trapezoid(points, begin, end):
    values = sorted((x, y) for x, y in points if begin <= x <= end)
    require(values and values[0][0] == begin and values[-1][0] == end and end > begin,
            "AUC requires observed endpoints")
    return sum((xb - xa) * (ya + yb) / 2 for (xa, ya), (xb, yb) in zip(values, values[1:])) / (end - begin)


def power_scenarios(sds, ns=(8, 12, 16), margin=.02, draws=20000, seed=20261007):
    import numpy as np
    from scipy.stats import t
    rng = np.random.default_rng(seed)
    rows = []
    for sd in sds:
        require(sd > 0, "Planning SD must be positive")
        for n in ns:
            for true in (0.0, -margin / 2, margin / 2, -2 * margin, 2 * margin):
                sample = rng.normal(true, sd, (draws, n))
                means = sample.mean(axis=1)
                se = sample.std(axis=1, ddof=1) / math.sqrt(n)
                ci90 = float(t.ppf(.95, n - 1)) * se
                ci95 = float(t.ppf(.975, n - 1)) * se
                rows.append({"sd": sd, "n": n, "true_difference": true, "draws": draws,
                    "tost_power": float(np.mean((means - ci90 > -margin) & (means + ci90 < margin))),
                    "useful_direction_power": float(np.mean((means + ci95 < -margin) | (means - ci95 > margin))),
                    "expected_95ci_halfwidth": float(t.ppf(.975, n - 1)) * sd / math.sqrt(n)})
    return {"seed": seed, "normal_paired_difference_assumption": True, "scenarios": rows}


def summarize_bundles(bundle_paths, output, primary_delay, margin=.02, preregistration=None):
    bundles, costs = [], []
    for path in map(Path, bundle_paths):
        manifest = read_json(path / "prepared" / "manifest.json")
        stage2 = read_json(path / "prepared" / "schedule.json")["stage2_steps"]
        bundle = {"manifest_sha256": manifest["sha256"], "mode": manifest["mode"], "arms": {}}
        for arm in ("NONE", "UNI", "EXP", "MASS", "GEN"):
            directory = path / arm
            matches = sorted((directory / "evaluations").glob("stage2-*.json"))
            evaluations = [read_json(p) for p in matches]
            require(all(e["manifest_sha256"] == manifest["sha256"] for e in evaluations), "Evaluation manifest mismatch")
            endpoint = [e for e in evaluations if e["stage2_step"] == stage2 + primary_delay]
            require(len(endpoint) == 1, f"Missing unique primary endpoint {path}/{arm}")
            aggregates = endpoint[0]["variants"]["canonical"]["aggregate"]
            points = [(e["stage2_step"], e["variants"]["canonical"]["aggregate"]["old"]["loss_event_macro"])
                      for e in evaluations]
            baseline = next(e for e in evaluations if e["stage2_step"] == 0)["variants"]["canonical"]["aggregate"]
            bundle["arms"][arm] = {"primary": aggregates, "stage2_auc": trapezoid(points, 0, stage2),
                "drift_adjusted_change": (aggregates["old"]["loss_event_macro"] - baseline["old"]["loss_event_macro"]) -
                                         (aggregates["control"]["loss_event_macro"] - baseline["control"]["loss_event_macro"])}
        for attempt in path.glob("*/attempt-*.json"):
            costs.append(read_json(attempt))
        bundles.append(bundle)
    require(len({b["mode"] for b in bundles}) == 1, "Never pool development with confirmation")
    require(len({b["manifest_sha256"] for b in bundles}) == len(bundles), "Duplicate replicate bundles")
    if bundles[0]["mode"] == "confirmation":
        require(preregistration is not None, "Confirmation analysis requires frozen preregistration")
        prereg = read_json(preregistration)
        require(len(bundles) == prereg["n"] and {b["manifest_sha256"] for b in bundles} ==
                set(prereg["replicate_manifest_sha256"]), "Incomplete preregistered replicate set")
        require(primary_delay == prereg["primary_delay"] and margin == prereg["loss_margin"], "Analysis changed frozen estimand")
    tests = {}
    for name, a, b in (("H2", "EXP", "UNI"), ("H3", "UNI", "MASS"),
                       ("H1", "UNI", "NONE"), ("HG", "UNI", "GEN")):
        values = [x["arms"][a]["primary"]["old"]["loss_event_macro"] -
                  x["arms"][b]["primary"]["old"]["loss_event_macro"] for x in bundles]
        tests[name] = paired_test(values, margin) if len(values) >= 2 else {
            "n": 1, "paired_differences": values, "estimate": values[0], "inference": "development descriptive only"}
    if len(bundles) >= 2:
        adjusted = holm({key: tests[key]["p_two_sided"] for key in ("H1", "H3", "HG")})
        for key, p in adjusted.items():
            tests[key]["p_holm"] = p
    write_json(output, {"bundles": bundles, "contrasts": tests, "loss_margin": margin,
                        "primary_delay": primary_delay, "cost_attempts": costs,
                        "allocated_gpu_hours_process_lower_bound": sum(c["gpu_count"] * c["wall_seconds"] / 3600 for c in costs),
                        "cost_note": "Add Slurm sacct allocation elapsed, setup jobs, queue time and staging/storage; process time is a lower bound."})
