from __future__ import annotations

import math
import collections
import json
from pathlib import Path
import statistics

from .common import digest, read_json, require, write_json
from .units import LEGACY_POLICY, UNIT_POLICY, LEGACY_METRICS, UNIT_METRICS, GROUNDED_POLICY, GROUNDED_METRICS
from .acquisition import DECLARATION_POLICY


def protected_cluster_loss(rows, components, role="old"):
    """Average probes, units, events within components, then equal clusters."""
    units = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        if row["role"] == role:
            require(math.isfinite(float(row["loss"])), "Nonfinite loss in protected-event sensitivity")
            units[row["event"]][row["unit_id"]].append(float(row["loss"]))
    require(units, "Protected-event sensitivity has no observed role events")
    event_means = {event: statistics.mean(statistics.mean(values) for values in by_unit.values())
                   for event, by_unit in units.items()}
    membership = {}
    for component in components:
        key = tuple(sorted(component))
        require(len(key) >= 2 and len(set(key)) == len(key) and not set(key) & set(membership),
                "Protected-event components overlap or repeat members")
        for event in key:
            membership[event] = key
        observed = set(key) & set(event_means)
        require(not observed or observed == set(key), "Protected-event component crosses roles or has missing observations")
    clusters = collections.defaultdict(list)
    for event, value in event_means.items():
        clusters[membership.get(event, (event,))].append(value)
    return {"loss_equal_cluster_macro": statistics.mean(statistics.mean(values) for values in clusters.values()),
            "events": len(event_means), "clusters": len(clusters),
            "component_event_means": {"|".join(key): statistics.mean(values) for key, values in sorted(clusters.items())}}


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


def validate_analysis_methods(bundles):
    require(len({b["mode"] for b in bundles}) == 1, "Never pool development with confirmation")
    require(len({b["rehearsal_unit_policy"] for b in bundles}) == 1, "Never pool different rehearsal unit policies")
    require(len({b.get("acquisition_policy", DECLARATION_POLICY) for b in bundles}) == 1,
            "Never pool different acquisition policies")


def validate_confirmation_completion(path, manifest, schedule, examples, prereg):
    """Bind complete paired confirmation streams to the actual shared state.

    Poor fixed-dose acquisition remains eligible. Missing/incomplete arms remain
    missing, and cannot quietly turn into a reduced primary-endpoint dataset.
    """
    from .acquisition_grid import checkpoint_digest
    from .final_span import _load_state
    from .schedule import ARMS, stage1_epoch
    from .units import schedule_records
    from .evaluation import aggregate
    path = Path(path)
    shared_path = path / "stage1" / "stage1.pt"
    shared_hash = checkpoint_digest(shared_path)
    shared = _load_state(shared_path, manifest["sha256"])
    acquired = shared["progress"]
    del shared
    require(acquired.get("status") == "stage1_complete" and acquired.get("epoch") == prereg["E"] and
            acquired.get("epoch_cursor") == 0 and acquired.get("confirmation_fixed_dose_complete") is True and
            acquired.get("confirmation_preregistration_sha256") == prereg["sha256"] and
            acquired["global_step"] >= manifest["config"]["warmup_steps"],
            "Confirmation analysis requires actual completed frozen fixed-dose acquisition")
    targets = manifest["acquisition_qa_pool"]["records"]
    require(acquired["acquisition_qa_target_doses"] == {r["id"]: prereg["E"] for r in targets},
            "Shared confirmation state lost or changed source-QA target doses")
    files = [{"path": str(shared_path.resolve()), "sha256": shared_hash}]
    def retain(file):
        files.append({"path": str(file.resolve()), "sha256": checkpoint_digest(file)})
    decision_path = path / "stage1" / "acquisition_decision.json"
    decision = read_json(decision_path)
    require(decision.get("manifest_sha256") == manifest["sha256"] and decision.get("selected_E") == prereg["E"] and
            decision.get("fixed_dose_continuation_authorized") is True and decision.get("outcome_filtering_permitted") is False and
            decision.get("confirmation_preregistration_sha256") == prereg["sha256"] and
            decision.get("acquisition_usable") == acquired["acquisition_usable"] and
            decision.get("observed_acquisition_old_exact_match_event_macro") == acquired["observed_acquisition_old_exact_match_event_macro"],
            "Confirmation decision differs from the actual retained fixed-dose state")
    retain(decision_path)
    acquisition_path = path / "stage1" / "evaluations" / f"acquisition-E{prereg['E']:03}.json"
    acquisition = read_json(acquisition_path)
    require(acquisition["manifest_sha256"] == manifest["sha256"] and acquisition["epoch"] == prereg["E"] and
            acquisition["global_step"] == acquired["global_step"] and acquisition["metric_schema"] == GROUNDED_METRICS and
            acquisition["variants"]["canonical"]["aggregate"]["old"]["exact_match_event_macro"] ==
            acquired["observed_acquisition_old_exact_match_event_macro"], "Actual shared acquisition baseline differs from the fixed-dose decision")
    retain(acquisition_path)
    # Reconstruct Stage 1, so a complete status cannot hide changed content or
    # the training of a control/probe example during the acquisition phase.
    training_records = schedule_records(manifest["split"]["facts"], manifest["unit_registry"],
                                        manifest["qa_teaching_pool"]["records"])
    expected_stage1 = [row for epoch in range(prereg["E"]) for row in
                       stage1_epoch(training_records, epoch, manifest["config"], targets)]
    stage1_log_path = path / "stage1" / "exposures.jsonl"
    stage1_logs = [json.loads(line) for line in stage1_log_path.read_text().splitlines()]
    require(len(stage1_logs) == len(expected_stage1) == acquired["global_step"] == acquired["cursor"],
            "Actual shared acquisition update count differs from its fixed complete stream")
    new_clock, acquisition_clock = 0, 0
    last_update, last_clock = {}, {}
    target_doses, variant_doses, declaration_doses = collections.Counter(), collections.Counter(), collections.Counter()
    for index, (log, row) in enumerate(zip(stage1_logs, expected_stage1)):
        new_clock += sum(examples[key]["loss_tokens"] for key in row if key.startswith(("new/", "qa/")))
        acquisition_keys = [key for key in row if key.startswith("acq/")]
        acquisition_clock += sum(examples[key]["loss_tokens"] for key in acquisition_keys)
        target_doses.update(key.split("/")[1] for key in acquisition_keys)
        variant_doses.update((key.split("/")[1], int(key.split("/")[2])) for key in acquisition_keys)
        declaration_doses.update(key.split("/", 1)[1] for key in row if key.startswith("old/"))
        for key in row:
            if key.startswith("old/"):
                unit = key.split("/", 1)[1]
                last_update[unit], last_clock[unit] = index + 1, new_clock
        require(log["phase"] == "stage1" and log["phase_step"] == index and log["cursor"] == index + 1 and
                log["global_step"] == index + 1 and log["examples"] == row and log["row_sha256"] == digest(row) and
                log["example_content_sha256"] == [examples[key]["content_sha256"] for key in row] and
                log["loss_tokens"] == sum(examples[key]["loss_tokens"] for key in row) and
                log["acquisition_qa_examples"] == acquisition_keys and log["acquisition_qa_tokens_total"] == acquisition_clock and
                log["new_qa_tokens_total"] == new_clock,
                "Actual acquisition content/order/loss masks differ from the fixed prepared stream")
    require(target_doses == collections.Counter(acquired["acquisition_qa_target_doses"]) and
            acquired["acquisition_qa_variant_doses"] == {r["id"]: [variant_doses[(r["id"], v)] for v in range(len(r["questions"]))] for r in targets} and
            acquired["acquisition_qa_tokens_total"] == acquisition_clock and acquired["new_qa_tokens_total"] == new_clock and
            acquired["last_old_update"] == last_update and acquired["last_old_newqa_clock"] == last_clock and
            declaration_doses == collections.Counter({r["id"]: prereg["E"] for r in training_records if r["role"] == "old"}),
            "Shared confirmation acquisition counters differ from actual complete realized content")
    retain(stage1_log_path)
    acquisition_attempt_paths = sorted((path / "stage1").glob("attempt-*.json"))
    acquisition_attempts = [read_json(p) for p in acquisition_attempt_paths]
    require(acquisition_attempts and len(acquisition_attempts) <= prereg["maximum_attempts"] and
            all(a["manifest_sha256"] == manifest["sha256"] for a in acquisition_attempts) and
            acquisition_attempts[-1]["status"] in ("complete", "already_complete") and
            acquisition_attempts[-1]["progress"] == acquired,
            "Confirmation acquisition has no matching complete attempt or exceeds its frozen budget")
    for file in acquisition_attempt_paths:
        retain(file)
    stage2 = schedule["stage2_steps"]
    arm_receipts = {}
    for arm in ARMS:
        directory = path / arm
        terminal_path = directory / "latest.pt"
        state = _load_state(terminal_path, manifest["sha256"])
        terminal = state["progress"]
        require(terminal.get("status") == "complete" and terminal.get("arm") == arm and
                terminal.get("confirmation_fixed_dose_complete") is True and terminal.get("epoch") == prereg["E"] and
                terminal.get("confirmation_preregistration_sha256") == prereg["sha256"] and
                terminal.get("confirmation_shared_checkpoint") == str(shared_path.resolve()) and
                terminal.get("confirmation_shared_checkpoint_sha256") == shared_hash,
                "Confirmation arm lacks a complete actual shared-state fork proof")
        rows = schedule["arms"][arm]
        require(terminal["continuation_cursor"] == len(rows) == terminal["cursor"] and
                terminal["global_step"] == acquired["global_step"] + len(rows), "Confirmation arm buffer or global update clock is incomplete")
        for key in ("acquisition_qa_target_doses", "acquisition_qa_variant_doses", "acquisition_qa_tokens_total"):
            require(terminal[key] == acquired[key], "Confirmation source acquisition QA changed after the shared fork")
        attempt_paths = sorted(directory.glob("attempt-*.json"))
        attempts = [read_json(p) for p in attempt_paths]
        require(attempts and len(attempts) <= prereg["maximum_attempts"] and
                all(a["manifest_sha256"] == manifest["sha256"] for a in attempts) and
                attempts[-1]["status"] == "complete" and attempts[-1]["progress"] == terminal,
                "Confirmation arm has no matching complete attempt or exceeds the frozen attempt budget")
        exposure_path = directory / "exposures.jsonl"
        logs = [json.loads(line) for line in exposure_path.read_text().splitlines()]
        require(len(logs) == len(rows), "Confirmation arm realized stream is incomplete")
        token_clock = acquired["new_qa_tokens_total"]
        last_update, last_clock = dict(acquired["last_old_update"]), dict(acquired["last_old_newqa_clock"])
        clocks = {0: {"updates_since_last_old_exposure": {key: acquired["global_step"] - step for key, step in last_update.items()},
                      "new_qa_tokens_since_last_old_exposure": {key: token_clock - value for key, value in last_clock.items()}}}
        for index, (log, row) in enumerate(zip(logs, rows)):
            token_clock += sum(examples[key]["loss_tokens"] for key in row if key.startswith(("new/", "qa/")))
            for key in row:
                if key.startswith("old/"):
                    unit = key.split("/", 1)[1]
                    last_update[unit], last_clock[unit] = acquired["global_step"] + index + 1, token_clock
            require(log["phase"] == ("stage2" if index < stage2 else "buffer") and log["phase_step"] == index and
                    log["cursor"] == index + 1 and log["global_step"] == acquired["global_step"] + index + 1 and
                    log["examples"] == row and log["row_sha256"] == digest(row) and
                    log["example_content_sha256"] == [examples[key]["content_sha256"] for key in row] and
                    log["loss_tokens"] == sum(examples[key]["loss_tokens"] for key in row) and
                    log["new_qa_tokens_total"] == token_clock and log.get("acquisition_qa_examples") == [] and
                    log["acquisition_qa_tokens_total"] == acquired["acquisition_qa_tokens_total"],
                    "Actual confirmation ordered content/token/exposure stream differs from the frozen schedule")
            clocks[index + 1] = {"updates_since_last_old_exposure": {
                key: acquired["global_step"] + index + 1 - step for key, step in last_update.items()},
                "new_qa_tokens_since_last_old_exposure": {key: token_clock - value for key, value in last_clock.items()}}
        require(terminal["new_qa_tokens_total"] == token_clock and terminal["last_old_update"] == last_update and
                terminal["last_old_newqa_clock"] == last_clock, "Confirmation terminal clocks differ from actual exposures")
        evaluation_paths = sorted((directory / "evaluations").glob("stage2-*.json"))
        evaluations = [read_json(p) for p in evaluation_paths]
        by_step = {e["stage2_step"]: e for e in evaluations}
        required = {0, stage2, len(rows), *(stage2 + delay for delay in manifest["config"]["eval_delays"]),
                    *(round(stage2 * fraction) for fraction in manifest["config"]["stage2_eval_fractions"])}
        require(len(by_step) == len(evaluations) and set(by_step) == required, "Confirmation scheduled evaluation set is incomplete or duplicated")
        for evaluation in evaluations:
            step = evaluation["stage2_step"]
            require(evaluation["manifest_sha256"] == manifest["sha256"] and evaluation["epoch"] == prereg["E"] and
                    evaluation["global_step"] == acquired["global_step"] + step and evaluation["global_buffer_delay"] == step - stage2 and
                    evaluation["metric_schema"] == GROUNDED_METRICS and
                    evaluation["acquisition_qa_tokens_total"] == acquired["acquisition_qa_tokens_total"] and
                    all(evaluation[key] == value for key, value in clocks[step].items()),
                    "Confirmation evaluation dose/update/exposure clocks differ from the actual frozen stream")
            for variant in evaluation["variants"].values():
                require(variant["aggregate"] == aggregate(variant["facts"]), "Confirmation aggregate differs from its actual probe-level outcomes")
        require(by_step[0]["variants"] == acquisition["variants"], "Confirmation arm baseline differs from its actual shared acquisition state")
        arm_receipts[arm] = {"terminal_checkpoint_sha256": checkpoint_digest(terminal_path),
                             "completed_updates": len(rows), "evaluated_steps": sorted(by_step)}
        files.append({"path": str(terminal_path.resolve()), "sha256": arm_receipts[arm]["terminal_checkpoint_sha256"]})
        for file in [exposure_path, *attempt_paths, *evaluation_paths]:
            retain(file)
        del state
    receipt = {"schema": "spacing-confirmation-complete-bundle-v1", "manifest_sha256": manifest["sha256"],
               "preregistration_sha256": prereg["sha256"], "E": prereg["E"],
               "observed_acquisition_old_exact_match_event_macro": acquired["observed_acquisition_old_exact_match_event_macro"],
               "poor_fixed_dose_acquisition_retained": not acquired["acquisition_usable"], "arms": arm_receipts, "files": files}
    receipt["sha256"] = digest(receipt)
    return receipt


def summarize_bundles(bundle_paths, output, primary_delay, margin=.02, preregistration=None):
    bundles, costs = [], []
    prereg = read_json(preregistration) if preregistration is not None else None
    for path in map(Path, bundle_paths):
        manifest = read_json(path / "prepared" / "manifest.json")
        context = None
        if manifest.get("confirmation_protocol_schema"):
            from .prepare import load_prepared
            from .confirmation import validate_confirmation_audit, verify_confirmation_preregistration
            require(prereg is not None, "Confirmation analysis requires frozen preregistration")
            manifest, confirmed_schedule, confirmed_examples = load_prepared(path / "prepared")
            verify_confirmation_preregistration(prereg, manifest)
            context = validate_confirmation_audit(manifest["confirmation_audit"], source_facts=manifest["split"]["facts"])
            completion = validate_confirmation_completion(path, manifest, confirmed_schedule, confirmed_examples, prereg)
        require(not manifest.get("config", {}).get("development_scale_probe"),
                "Stage-1-only scale probes cannot enter spacing-arm analysis")
        stage2 = read_json(path / "prepared" / "schedule.json")["stage2_steps"]
        policy = manifest.get("rehearsal_unit_policy", LEGACY_POLICY)
        expected_metrics = {UNIT_POLICY: UNIT_METRICS, GROUNDED_POLICY: GROUNDED_METRICS,
                            LEGACY_POLICY: LEGACY_METRICS}[policy]
        bundle = {"manifest_sha256": manifest["sha256"], "mode": manifest["mode"], "arms": {},
                  "rehearsal_unit_policy": policy, "metric_schema": expected_metrics,
                  "acquisition_policy": manifest.get("acquisition_policy",
                                                     manifest.get("config", {}).get("acquisition_policy", DECLARATION_POLICY))}
        if context:
            bundle["confirmation_protocol_schema"] = manifest["confirmation_protocol_schema"]
            bundle["same_role_components"] = context["same_role_components"]
            bundle["completion_evidence"] = completion
        for arm in ("NONE", "UNI", "EXP", "MASS", "GEN"):
            directory = path / arm
            matches = sorted((directory / "evaluations").glob("stage2-*.json"))
            evaluations = [read_json(p) for p in matches]
            require(all(e["manifest_sha256"] == manifest["sha256"] for e in evaluations), "Evaluation manifest mismatch")
            require(all(e.get("metric_schema", LEGACY_METRICS) == expected_metrics for e in evaluations),
                    "Evaluation aggregation differs from prepared unit policy")
            endpoint = [e for e in evaluations if e["stage2_step"] == stage2 + primary_delay]
            require(len(endpoint) == 1, f"Missing unique primary endpoint {path}/{arm}")
            aggregates = endpoint[0]["variants"]["canonical"]["aggregate"]
            points = [(e["stage2_step"], e["variants"]["canonical"]["aggregate"]["old"]["loss_event_macro"])
                      for e in evaluations]
            baseline = next(e for e in evaluations if e["stage2_step"] == 0)["variants"]["canonical"]["aggregate"]
            bundle["arms"][arm] = {"primary": aggregates, "stage2_auc": trapezoid(points, 0, stage2),
                "drift_adjusted_change": (aggregates["old"]["loss_event_macro"] - baseline["old"]["loss_event_macro"]) -
                                         (aggregates["control"]["loss_event_macro"] - baseline["control"]["loss_event_macro"])}
            if context:
                rows = endpoint[0]["variants"]["canonical"]["facts"]
                expected = {f["id"]: f for f in manifest["split"]["facts"] if f["role"] in ("old", "new", "control")}
                require(len(rows) == len(expected) and {row["id"] for row in rows} == set(expected) and
                        all(all(row[key] == expected[row["id"]][key] for key in ("event", "role", "unit_id")) for row in rows),
                        "Confirmation endpoint probe membership differs from the frozen manifest")
                bundle["arms"][arm]["protected_event_cluster_sensitivity"] = protected_cluster_loss(rows, context["same_role_components"])
        for attempt in path.glob("*/attempt-*.json"):
            costs.append(read_json(attempt))
        bundles.append(bundle)
    validate_analysis_methods(bundles)
    require(len({b["manifest_sha256"] for b in bundles}) == len(bundles), "Duplicate replicate bundles")
    if bundles[0]["mode"] == "confirmation":
        require(preregistration is not None, "Confirmation analysis requires frozen preregistration")
        if bundles[0].get("confirmation_protocol_schema"):
            require(all(b.get("confirmation_protocol_schema") == prereg.get("confirmation_protocol_schema") and
                        b["rehearsal_unit_policy"] == GROUNDED_POLICY and b["metric_schema"] == GROUNDED_METRICS for b in bundles),
                    "Confirmation protocol/grounded aggregation differs from frozen design")
        else:
            require(all(b["rehearsal_unit_policy"] == UNIT_POLICY for b in bundles) and
                    prereg.get("rehearsal_unit_policy") == UNIT_POLICY and prereg.get("metric_schema") == UNIT_METRICS,
                    "Confirmation requires preregistered supporting-statement units and aggregation")
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
        if bundles[0].get("confirmation_protocol_schema"):
            clustered_values = [x["arms"][a]["protected_event_cluster_sensitivity"]["loss_equal_cluster_macro"] -
                                x["arms"][b]["protected_event_cluster_sensitivity"]["loss_equal_cluster_macro"] for x in bundles]
            tests[name]["protected_event_cluster_sensitivity"] = paired_test(clustered_values, margin)
            if prereg["inference_claim"] == "estimation":
                for result in (tests[name], tests[name]["protected_event_cluster_sensitivity"]):
                    result["loss_equivalent"] = None
                    result["useful_direction"] = None
                    result["inference"] = "Frozen estimation-only fallback; intervals describe uncertainty and do not authorize equivalence or useful-direction claims"
    if len(bundles) >= 2:
        adjusted = holm({key: tests[key]["p_two_sided"] for key in ("H1", "H3", "HG")})
        for key, p in adjusted.items():
            tests[key]["p_holm"] = p
        if bundles[0].get('confirmation_protocol_schema'):
            adjusted_cluster = holm({key: tests[key]['protected_event_cluster_sensitivity']['p_two_sided']
                                     for key in ('H1', 'H3', 'HG')})
            for key, p in adjusted_cluster.items():
                tests[key]['protected_event_cluster_sensitivity']['p_holm'] = p
    write_json(output, {"bundles": bundles, "contrasts": tests, "loss_margin": margin,
                        "inference_claim": prereg.get("inference_claim") if prereg else "development_descriptive",
                        "preregistration_sha256": prereg.get("sha256") if prereg else None,
                        "primary_delay": primary_delay, "cost_attempts": costs,
                        "allocated_gpu_hours_process_lower_bound": sum(c["gpu_count"] * c["wall_seconds"] / 3600 for c in costs),
                        "cost_note": "Add Slurm sacct allocation elapsed, setup jobs, queue time and staging/storage; process time is a lower bound."})
