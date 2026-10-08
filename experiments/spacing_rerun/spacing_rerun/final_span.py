"""Validate the six final-span development bundles and plan confirmation precision.

This report is development evidence. It never authorizes confirmation training.
Run on the machine holding the original prepared, selection and checkpoint files.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import statistics
import sys
import types

from . import SCHEMA
from .acquisition_grid import checkpoint_digest, verify_acquisition_grid_selection
from .acquisition_policy import BINDING_SCHEMA, manifest_policy_trial
from .common import digest, read_json, require, write_json
from .prepare import load_prepared
from .schedule import ARMS
from .units import GROUNDED_METRICS

FROZEN_CORE_SHA256 = "56d0d72fcfbff828dafe27e6713041fc9b0d6ed0ad9de2a799c312083d9975cc"
PLANNING_POLICY = {
    "schema": "spacing-final-span-planning-policy-v1",
    "frozen_utc": "2026-10-08T07:13:53+00:00",
    "acquisition_policy_core_sha256": FROZEN_CORE_SHA256,
    "sensitivity": {"replicates": 6, "span": 252, "primary_delay": 84,
                    "none_loss_rise_min": .10, "none_em_decline_min": .05,
                    "none_minus_uni_min": .02, "gen_minus_uni_min": .02,
                    "loss_one_sided_lower_confidence": .95},
    "margin": .02, "target_power": .80, "planning_sd_upper_confidence": .80,
    "candidate_n": list(range(2, 33)), "simulation_draws": 100000,
    "simulation_seed": 2026100809, "mc_one_sided_lower_confidence": .95,
    "mc_bound_method": "exact_clopper_pearson_binomial",
    "simulation_distribution": "independent_normal_mean_and_chi_square_sample_variance_finite_n",
    "expected_ci_halfwidth_method": "t_0_975_times_sd_times_normal_c4_over_sqrt_n",
    "required_scenarios": [
        {"true_difference": 0., "test": "tost"},
        {"true_difference": -.01, "test": "tost"},
        {"true_difference": .01, "test": "tost"},
        {"true_difference": -.04, "test": "useful_direction"},
        {"true_difference": .04, "test": "useful_direction"}],
    "sd_sensitivities": ["observed", "one_sided_80pct_upper", "1.5_times_upper"],
    "selection": "smallest_n_all_five_mc_lower_bounds_at_least_0_80",
    "fallback": "n32_estimation_claim_with_expected_95pct_ci_halfwidth",
    "normal_paired_difference_assumption": True,
    "normality_established": False,
}
PLANNING_POLICY_SHA256 = digest(PLANNING_POLICY)


def validate_planning_policy():
    path = Path(__file__).resolve().parents[1] / "configs/final-span-planning-policy-20261008.json"
    packet = read_json(path)
    require(packet == dict(PLANNING_POLICY, sha256=PLANNING_POLICY_SHA256),
            "Final-span planning configuration differs from the frozen content hash")
    return packet


def finite_number(value, label, *, probability=False):
    require(type(value) in (float, int) and math.isfinite(value), f"Invalid {label}")
    require(not probability or 0 <= value <= 1, f"Invalid {label} probability")
    return value


def loss_check(values, minimum):
    from scipy.stats import t
    require(len(values) == 6, "Final-span sensitivity requires six paired replicates")
    values = [finite_number(value, "paired sensitivity difference") for value in values]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    lower = mean - float(t.ppf(.95, 5)) * sd / math.sqrt(6)
    return {"paired_differences": values, "mean": mean, "sd": sd,
            "one_sided_95pct_lower": lower, "minimum_mean": minimum,
            "passes": mean >= minimum and lower > 0}


def assay_sensitivity(bundles):
    require(len(bundles) == 6, "Final-span sensitivity requires all six complete bundles")
    rise = [b["arms"]["NONE"]["loss"] - b["acquisition"]["loss"] for b in bundles]
    decline = [b["acquisition"]["exact_match"] - b["arms"]["NONE"]["exact_match"] for b in bundles]
    none_uni = [b["arms"]["NONE"]["loss"] - b["arms"]["UNI"]["loss"] for b in bundles]
    gen_uni = [b["arms"]["GEN"]["loss"] - b["arms"]["UNI"]["loss"] for b in bundles]
    checks = {"none_acquisition_loss_rise": loss_check(rise, .10),
              "none_minus_uni_loss": loss_check(none_uni, .02),
              "gen_minus_uni_loss": loss_check(gen_uni, .02),
              "none_exact_match_decline": {"paired_differences": decline,
                  "mean": statistics.mean(decline), "minimum_mean": .05,
                  "passes": statistics.mean(decline) >= .05}}
    return {"assay_usable": all(check["passes"] for check in checks.values()),
            "checks": checks, "EXP_used_for_eligibility": False}


def _mc_interval(successes, draws):
    from scipy.stats import beta
    # Exact binomial bounds describe simulation error, not uncertainty in the SD.
    return {"power": successes / draws, "successes": successes, "draws": draws,
            "one_sided_95pct_lower": float(beta.ppf(.05, successes, draws - successes + 1)) if successes else 0.,
            "ci95": [float(beta.ppf(.025, successes, draws - successes + 1)) if successes else 0.,
                     float(beta.ppf(.975, successes + 1, draws - successes)) if successes < draws else 1.]}


def plan_precision(paired_differences):
    """Finite-t simulation with the exact normal mean/sample-SD distributions."""
    import numpy as np
    from scipy.special import gammaln
    from scipy.stats import chi2, t
    require(len(paired_differences) == 6, "Planning requires all six independent EXP-minus-UNI pairs")
    values = [finite_number(v, "EXP-minus-UNI difference") for v in paired_differences]
    sd = statistics.stdev(values)
    upper = sd * math.sqrt(5 / float(chi2.ppf(.20, 5)))
    require(sd > 0, "Zero observed SD cannot support a positive variance planning bound; revise planning transparently")
    scenarios, rng = [], np.random.default_rng(PLANNING_POLICY["simulation_seed"])
    draws, margin = PLANNING_POLICY["simulation_draws"], PLANNING_POLICY["margin"]
    for label, planning_sd in (("observed", sd), ("one_sided_80pct_upper", upper), ("1.5_times_upper", 1.5 * upper)):
        for n in PLANNING_POLICY["candidate_n"]:
            # Under normal paired differences the standardized mean and sample
            # variance are independent. This is the full finite-n t experiment.
            z = rng.normal(size=draws)
            sample_sd_ratio = np.sqrt(rng.chisquare(n - 1, size=draws) / (n - 1))
            noise, se = planning_sd * z / math.sqrt(n), planning_sd * sample_sd_ratio / math.sqrt(n)
            r90, r95 = float(t.ppf(.95, n - 1)) * se, float(t.ppf(.975, n - 1)) * se
            # E[s] = c4 * sigma, so this is the expected actual CI half-width.
            c4 = math.sqrt(2 / (n - 1)) * math.exp(float(gammaln(n / 2) - gammaln((n - 1) / 2)))
            halfwidth = float(t.ppf(.975, n - 1)) * planning_sd * c4 / math.sqrt(n)
            for criterion in PLANNING_POLICY["required_scenarios"]:
                true, test = criterion["true_difference"], criterion["test"]
                means = true + noise
                if test == "tost":
                    success = (means - r90 > -margin) & (means + r90 < margin)
                elif true > 0:
                    success = means - r95 > margin
                else:
                    success = means + r95 < -margin
                scenarios.append({"sd_scenario": label, "sd": planning_sd, "n": n,
                    "true_difference": true, "test": test, "expected_95ci_halfwidth": halfwidth,
                    **_mc_interval(int(success.sum()), draws)})
    primary = [row for row in scenarios if row["sd_scenario"] == "one_sided_80pct_upper"]
    eligible = [n for n in PLANNING_POLICY["candidate_n"] if all(
        row["one_sided_95pct_lower"] >= .8 for row in primary if row["n"] == n)]
    chosen = min(eligible) if eligible else 32
    selected = [row for row in primary if row["n"] == chosen]
    return {"paired_differences": values, "observed_mean": statistics.mean(values), "observed_sd": sd,
            "sd_df": 5, "one_sided_80pct_upper_sd": upper,
            "normal_paired_difference_assumption": True, "normality_established": False,
            "chosen_n": chosen, "power_target_met": bool(eligible),
            "claim": "equivalence_and_useful_direction_planning" if eligible else "estimation",
            "expected_95ci_halfwidth": selected[0]["expected_95ci_halfwidth"],
            "chosen_primary_scenarios": selected, "scenarios": scenarios,
            "simulation_error_note": "Binomial intervals cover Monte Carlo error conditional on planning SD. The SD bound assumes independent normal paired differences; six pairs do not establish normality."}


def _load_state(path, manifest_sha):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(state.get("schema") == SCHEMA and state.get("manifest_sha256") == manifest_sha,
            "Full-state checkpoint schema/manifest differs")
    require(all(isinstance(state.get(key), dict) and state[key] for key in ("model", "optimizer", "rng", "progress")),
            "Checkpoint lacks full model, optimizer, RNG or progress state")
    require(set(state["rng"]) >= {"python", "numpy", "torch", "cuda"} and
            set(state["optimizer"]) >= {"state", "param_groups"} and state["optimizer"]["state"],
            "Checkpoint lacks optimizer moments or RNG sources")
    return state


def _old(evaluation):
    old = evaluation["variants"]["canonical"]["aggregate"]["old"]
    return {"loss": finite_number(old["loss_event_macro"], "old answer loss"),
            "exact_match": finite_number(old["exact_match_event_macro"], "old exact match", probability=True)}


def validate_bundle(path, selection_path=None):
    path = Path(path)
    manifest, schedule, examples = load_prepared(path / "prepared")
    selection_path = Path(selection_path) if selection_path else path / "acquisition-selection.json"
    selection = read_json(selection_path)
    selected_checkpoint = Path(selection["selected_checkpoint"])
    selection = verify_acquisition_grid_selection(selection_path, manifest, selected_checkpoint)
    binding = selection["selection_rule"]
    require(binding.get("schema") == BINDING_SCHEMA, "Final span requires the actual two-stage nine-manifest binding")
    core = binding["policy_core"]
    require(core["sha256"] == FROZEN_CORE_SHA256, "Final-span acquisition policy core differs from the frozen amendment")
    trial = manifest_policy_trial(manifest, core)
    require(not trial["development_scale_probe"] and trial["trial_id"].startswith("normal-"),
            "Scale acquisition trajectories cannot enter final-span sensitivity")
    dose = selection["selected_E"]
    state = _load_state(selected_checkpoint, manifest["sha256"])
    acquired = state["progress"]
    del state
    require(acquired["status"] == "stage1_complete" and acquired["epoch"] == dose and
            acquired["epoch_cursor"] == 0 and acquired.get("acquisition_usable") is True,
            "Selected full acquisition state is incomplete or has a different common dose")
    require(set(acquired["acquisition_qa_target_doses"]) == {r["id"] for r in manifest["acquisition_qa_pool"]["records"]} and
            all(value == dose for value in acquired["acquisition_qa_target_doses"].values()),
            "Selected acquisition state has unequal source-QA target doses")
    acquisition_path = selected_checkpoint.parent / "evaluations" / f"acquisition-E{dose:03}.json"
    acquisition = read_json(acquisition_path)
    require(acquisition["manifest_sha256"] == manifest["sha256"] and acquisition["epoch"] == dose and
            acquisition["global_step"] == acquired["global_step"] and acquisition["metric_schema"] == GROUNDED_METRICS,
            "Acquisition baseline does not belong to the selected full state")
    require(.4 <= _old(acquisition)["exact_match"] <= .7, "Selected baseline lies outside the fixed acquisition gate")
    own_decision = next(read_json(row["decision_path"]) for row in selection["grid_inputs"]
                        if read_json(row["decision_path"])["manifest_sha256"] == manifest["sha256"])
    require(_old(acquisition)["exact_match"] == own_decision["grid_results"][str(dose)]["old_exact_match_event_macro"],
            "Acquisition baseline differs from the frozen selected grid score")
    provenance = [{"path": str(selection_path.resolve()), "sha256": checkpoint_digest(selection_path)},
                  {"path": str(selected_checkpoint.resolve()), "sha256": selection["selected_checkpoint_sha256"]},
                  {"path": str(acquisition_path.resolve()), "sha256": checkpoint_digest(acquisition_path)}]
    arms, attempts, raw_endpoints = {}, [], {}
    stage2 = schedule["stage2_steps"]
    for arm in ARMS:
        directory = path / arm
        terminal_path = directory / "latest.pt"
        terminal_state = _load_state(terminal_path, manifest["sha256"])
        terminal = terminal_state["progress"]
        rows = schedule["arms"][arm]
        require(terminal["status"] == "complete" and terminal["arm"] == arm and terminal["epoch"] == dose and
                terminal["continuation_cursor"] == len(rows) and terminal["cursor"] == len(rows) and
                terminal["global_step"] == acquired["global_step"] + len(rows),
                "Missing complete final-span arm checkpoint or incorrect continuation clock")
        for key in ("acquisition_qa_target_doses", "acquisition_qa_variant_doses", "acquisition_qa_tokens_total"):
            require(terminal[key] == acquired[key], "Source acquisition QA changed after the shared fork")
        attempt_paths = sorted(directory.glob("attempt-*.json"))
        require(attempt_paths and len(attempt_paths) <= manifest["config"]["max_attempts"], "Missing or excessive arm attempts")
        arm_attempts = [read_json(p) for p in attempt_paths]
        frozen = datetime.datetime.fromisoformat(PLANNING_POLICY["frozen_utc"])
        starts = [datetime.datetime.fromisoformat(a["started_utc"]) for a in arm_attempts]
        require(all(start.tzinfo is not None and frozen <= start for start in starts),
                "Final-span arm began before the actual prospective planning freeze")
        require(all(a["manifest_sha256"] == manifest["sha256"] for a in arm_attempts) and
                arm_attempts[-1]["status"] == "complete" and arm_attempts[-1]["progress"] == terminal,
                "Final arm attempt and complete checkpoint disagree")
        attempts.extend(arm_attempts)
        exposure_path = directory / "exposures.jsonl"
        logs = [json.loads(line) for line in exposure_path.read_text().splitlines()]
        require(len(logs) == len(rows), "Complete arm exposure stream has missing or extra updates")
        token_clock = acquired["new_qa_tokens_total"]
        last_old_update = dict(acquired["last_old_update"])
        last_old_token_clock = dict(acquired["last_old_newqa_clock"])
        exposure_clocks = {0: {"updates_since_last_old_exposure": {
            key: acquired["global_step"] - step for key, step in last_old_update.items()},
            "new_qa_tokens_since_last_old_exposure": {
                key: token_clock - clock for key, clock in last_old_token_clock.items()}}}
        for i, (log, row) in enumerate(zip(logs, rows)):
            token_clock += sum(examples[key]["loss_tokens"] for key in row if key.startswith(("new/", "qa/")))
            for key in row:
                if key.startswith("old/"):
                    unit = key.split("/", 1)[1]
                    last_old_update[unit] = acquired["global_step"] + i + 1
                    last_old_token_clock[unit] = token_clock
            require(log["cursor"] == i + 1 and log["phase_step"] == i and
                    log["global_step"] == acquired["global_step"] + i + 1 and
                    log["phase"] == ("stage2" if i < stage2 else "buffer") and
                    log["examples"] == row and log["row_sha256"] == digest(row) and
                    log["example_content_sha256"] == [examples[key]["content_sha256"] for key in row] and
                    log["loss_tokens"] == sum(examples[key]["loss_tokens"] for key in row) and
                    log["new_qa_tokens_total"] == token_clock and
                    log.get("acquisition_qa_examples") == [] and
                    log["acquisition_qa_tokens_total"] == acquired["acquisition_qa_tokens_total"],
                    "Realized final-span content, ordered stream, token budget or clocks differ")
            exposure_clocks[i + 1] = {"updates_since_last_old_exposure": {
                key: acquired["global_step"] + i + 1 - step for key, step in last_old_update.items()},
                "new_qa_tokens_since_last_old_exposure": {
                    key: token_clock - clock for key, clock in last_old_token_clock.items()}}
        require(terminal["new_qa_tokens_total"] == token_clock and terminal["last_old_update"] == last_old_update and
                terminal["last_old_newqa_clock"] == last_old_token_clock, "Terminal exposure clocks differ from actual ordered stream")
        evaluation_paths = sorted((directory / "evaluations").glob("stage2-*.json"))
        evaluations = [read_json(p) for p in evaluation_paths]
        require(all(e["manifest_sha256"] == manifest["sha256"] and e["epoch"] == dose and
                    e["metric_schema"] == GROUNDED_METRICS and
                    e["global_step"] == acquired["global_step"] + e["stage2_step"] and
                    e["acquisition_qa_tokens_total"] == acquired["acquisition_qa_tokens_total"] for e in evaluations),
                    "Arm evaluation manifest, dose, metric or clock differs")
        require(all(all(e[key] == value for key, value in exposure_clocks[e["stage2_step"]].items()) for e in evaluations),
                "Evaluation exposure-to-test clocks differ from actual ordered stream")
        by_step = {e["stage2_step"]: e for e in evaluations}
        require(len(by_step) == len(evaluations) and {0, stage2, len(rows), *(stage2 + d for d in (21, 84, 168))} <= set(by_step),
                "Missing or duplicate scheduled final-span evaluations")
        require(by_step[0]["variants"]["canonical"] == acquisition["variants"]["canonical"],
                "Arm acquisition-baseline answers/losses differ from the shared selected state")
        require(by_step[stage2 + 84]["global_buffer_delay"] == 84, "Primary endpoint delay changed")
        arms[arm] = _old(by_step[stage2 + 84])
        endpoint_path = next(p for p, e in zip(evaluation_paths, evaluations) if e["stage2_step"] == stage2 + 84)
        raw_endpoints[arm] = {"path": str(endpoint_path.resolve()), "sha256": checkpoint_digest(endpoint_path),
                              "stage2_step": stage2 + 84, **arms[arm], **exposure_clocks[stage2 + 84]}
        for p in [terminal_path, exposure_path, *attempt_paths, *evaluation_paths]:
            provenance.append({"path": str(p.resolve()), "sha256": checkpoint_digest(p)})
        del terminal_state
    return {"bundle_path": str(path.resolve()), "selection_path": str(selection_path.resolve()),
            "trial_id": trial["trial_id"], "manifest_sha256": manifest["sha256"],
            "selection_sha256": selection["sha256"], "manifest_binding_sha256": binding["sha256"],
            "selected_E": dose, "seeds": {key: manifest["config"][key] for key in ("split_seed", "order_seed", "train_seed", "eval_seed")},
            "stagger_width": schedule["stagger_width"], "stagger_span_ratio": schedule["stagger_span_ratio"],
            "acquisition": _old(acquisition), "arms": arms,
            "raw_acquisition": {"path": str(acquisition_path.resolve()), "sha256": checkpoint_digest(acquisition_path),
                                "epoch": dose, **_old(acquisition)},
            "raw_primary_endpoints": raw_endpoints, "provenance": provenance, "attempts": attempts}


def cost_report(bundles, allocation_accounting=None, chosen_n=None):
    attempts = [a for bundle in bundles for a in bundle["attempts"]]
    process = 0.
    for attempt in attempts:
        seconds = finite_number(attempt["wall_seconds"], "process wall seconds")
        gpus = finite_number(attempt["gpu_count"], "process GPU count")
        require(seconds >= 0 and type(gpus) is int and gpus >= 0, "Invalid process runtime/GPU count")
        process += seconds * gpus / 3600
    packet = {"arm_process_gpu_hours_lower_bound": process,
              "allocation_gpu_hours": None, "accounting_complete": False,
              "scope": "Six development arm bundles including all recorded attempts. Acquisition grids, CPU preparation, queue time, staging and storage are separate.",
              "confirmation_cost_projection": None}
    if allocation_accounting is None:
        return packet
    accounting = read_json(allocation_accounting)
    require(accounting.get("schema") == "spacing-slurm-allocation-accounting-v1", "Unknown allocation accounting schema")
    rows = accounting.get("allocations", [])
    require(len(rows) == len({row["job_id"] for row in rows}), "Duplicate Slurm allocation accounting")
    by_job = {str(row["job_id"]): row for row in rows}
    expected = {str(a["slurm_job_id"]) for a in attempts}
    require("None" not in expected and expected <= set(by_job), "Allocation accounting omits observed arm attempt jobs")
    gpu_hours = {}
    for job in expected:
        row = by_job[job]
        elapsed = finite_number(row["elapsed_seconds"], "allocation elapsed seconds")
        gpus = finite_number(row["allocated_gpu_count"], "allocated GPU count")
        require(elapsed >= 0 and gpus >= 1 and row.get("source") == "sacct" and row.get("state") in
                ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"),
                "Accounting must describe terminal actual sacct GPU allocations")
        gpu_hours[job] = elapsed * gpus / 3600
    allocated = sum(gpu_hours.values())
    require(allocated + 1e-3 >= process, "Allocation accounting is smaller than observed process runtime")
    largest = max(sum(gpu_hours[job] for job in {str(a["slurm_job_id"]) for a in bundle["attempts"]}) for bundle in bundles)
    packet.update(allocation_gpu_hours=allocated, accounting_complete=True,
                  accounting_path=str(Path(allocation_accounting).resolve()),
                  accounting_sha256=checkpoint_digest(allocation_accounting),
                  largest_observed_five_arm_bundle_gpu_hours=largest)
    if chosen_n is not None:
        packet["confirmation_cost_projection"] = {"arm_gpu_hours_at_largest_observed_bundle": chosen_n * largest,
            "n": chosen_n, "basis": "Observed maximum standard-development arm allocation, not a bound on larger confirmation content. Add scaled acquisition, preparation and retry allowance before final budgeting.",
            "full_experiment_budget_ready": False}
    return packet


def assay_evidence(bundles, sensitivity):
    packet = {"schema": "spacing-final-span-assay-evidence-v1", "mode": "development",
              "acquisition_policy_core_sha256": FROZEN_CORE_SHA256,
              "selected_E": bundles[0]["selected_E"],
              "manifest_binding_sha256": bundles[0]["manifest_binding_sha256"],
              "pairs": [{key: bundle[key] for key in
                         ("trial_id", "manifest_sha256", "selection_sha256", "bundle_path", "selection_path",
                          "seeds", "raw_acquisition", "raw_primary_endpoints", "provenance")} for bundle in bundles],
              **sensitivity, "human_review_complete": False, "confirmation_ready": False}
    packet["sha256"] = digest(packet)
    return packet


def evaluate_final_span(bundle_paths, output, *, selection_paths=None, allocation_accounting=None):
    require(not Path(output).exists(), "Final-span report already exists; never overwrite")
    validate_planning_policy()
    require(len(bundle_paths) == 6, "Final span requires all six prespecified normal bundles")
    selections = selection_paths if selection_paths is not None else [None] * 6
    require(len(selections) == 6, "Provide exactly one selection path per bundle")
    bundles = [validate_bundle(path, selection) for path, selection in zip(bundle_paths, selections)]
    require({b["trial_id"] for b in bundles} == {f"normal-{i:02}" for i in range(1, 7)} and
            len({b["manifest_sha256"] for b in bundles}) == 6, "Missing or duplicate frozen normal replicate")
    require(len({b["manifest_binding_sha256"] for b in bundles}) == 1 and
            len({b["selected_E"] for b in bundles}) == 1, "Bundles fork different bindings or common doses")
    bundles.sort(key=lambda b: b["trial_id"])
    sensitivity = assay_sensitivity(bundles)
    differences = [b["arms"]["EXP"]["loss"] - b["arms"]["UNI"]["loss"] for b in bundles]
    precision = plan_precision(differences)
    packet = {"schema": "spacing-final-span-development-report-v1", "mode": "development",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "planning_policy": PLANNING_POLICY, "planning_policy_sha256": PLANNING_POLICY_SHA256,
              "human_review_complete": False, "confirmation_ready": False,
              "selected_E": bundles[0]["selected_E"], "bundles": bundles,
              **sensitivity, "precision_plan": precision,
              "assay_evidence": assay_evidence(bundles, sensitivity),
              "precision_claim_eligible": sensitivity["assay_usable"],
              "cost": cost_report(bundles, allocation_accounting, precision["chosen_n"])}
    packet["sha256"] = digest(packet)
    write_json(output, packet)
    return packet


def _verify_final_span_report_full(report_path):
    """Recompute a saved receipt from its actual immutable input files."""
    validate_planning_policy()
    packet = read_json(report_path)
    require(packet.get("schema") == "spacing-final-span-development-report-v1" and
            packet.get("sha256") == digest({k: v for k, v in packet.items() if k != "sha256"}),
            "Final-span report was modified")
    require(packet["planning_policy"] == PLANNING_POLICY and packet["planning_policy_sha256"] == PLANNING_POLICY_SHA256,
            "Final-span planning policy differs from the prospective policy")
    require(packet.get("mode") == "development" and packet.get("human_review_complete") is False and
            packet.get("confirmation_ready") is False, "Invalid final-span evidence scope")
    originals = packet["bundles"]
    require(len(originals) == 6 and {b["trial_id"] for b in originals} == {f"normal-{i:02}" for i in range(1, 7)} and
            len({b["manifest_sha256"] for b in originals}) == 6 and
            len({b["manifest_binding_sha256"] for b in originals}) == 1 and
            len({b["selected_E"] for b in originals}) == 1, "Incomplete or mismatched frozen final-span bundles")
    actual = [validate_bundle(b["bundle_path"], b["selection_path"]) for b in originals]
    require(actual == originals, "Final-span raw checkpoint/evaluation/stream evidence was modified")
    sensitivity = assay_sensitivity(actual)
    require(all(packet[key] == value for key, value in sensitivity.items()) and
            packet["assay_evidence"] == assay_evidence(actual, sensitivity), "Assay evidence differs from actual paired endpoints")
    precision = plan_precision([b["arms"]["EXP"]["loss"] - b["arms"]["UNI"]["loss"] for b in actual])
    require(precision == packet["precision_plan"] and packet["precision_claim_eligible"] == sensitivity["assay_usable"],
            "Precision plan differs from actual paired EXP-minus-UNI variance or eligibility")
    accounting = packet["cost"].get("accounting_path")
    require(cost_report(actual, accounting, precision["chosen_n"]) == packet["cost"], "Actual allocation/process cost evidence changed")
    require(packet["selected_E"] == actual[0]["selected_E"], "Final-span report common dose changed")
    return packet


PREFLIGHT_SCHEMA = "spacing-final-span-verified-preflight-v1"


def _file_identity(path):
    """ctime catches byte changes even when a writer restores the old mtime."""
    status = Path(path).stat()
    require(Path(path).is_file(), "Preflight evidence must remain a regular file")
    return {"device": status.st_dev, "inode": status.st_ino, "size": status.st_size,
            "mtime_ns": status.st_mtime_ns, "ctime_ns": status.st_ctime_ns,
            "mode": status.st_mode, "uid": status.st_uid, "gid": status.st_gid}


def _runtime_environment():
    # Import the analysis dependencies before indexing their actual loaded code.
    import numpy
    import scipy.stats
    import torch
    code_root = Path(__file__).resolve().parent
    source = {str(p.resolve()): checkpoint_digest(p) for p in sorted(code_root.rglob("*.py"))}
    callables = {}
    proof_modules = ("common", "acquisition_grid", "acquisition_policy", "acquisition", "prepare",
                     "data", "grounding", "units", "schedule", "teaching", "rewrites", "analysis")
    namespaces = {"spacing_rerun.final_span": globals()}
    namespaces.update({"spacing_rerun." + name: vars(importlib.import_module("spacing_rerun." + name))
                       for name in proof_modules})
    for name, namespace in sorted(namespaces.items()):
        for attr, value in sorted(namespace.items()):
            code = getattr(value, "__code__", None)
            if code is not None:
                # Bind the code actually loaded in memory as well as source
                # bytes, so a stale import or runtime monkeypatch invalidates.
                callables[f"{name}.{attr}"] = digest(_code_projection(code))
    versions = {}
    for distribution in ("numpy", "scipy", "torch", "transformers", "ai2-olmo", "tokenizers", "pyarrow"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {"python": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "versions": versions, "package_source_sha256": source,
            "loaded_callable_code_sha256": callables,
            "dependency_roots": {name: str(Path(module.__file__).resolve()) for name, module in
                                 (("numpy", numpy), ("scipy", sys.modules["scipy"]), ("torch", torch))}}


def _code_constant(value):
    # marshal contains interning/reference flags that can vary across calls.
    # Project immutable code fields and constants without those process flags.
    if isinstance(value, types.CodeType):
        return {"code": _code_projection(value)}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return {"tuple": [_code_constant(item) for item in value]}
    if isinstance(value, frozenset):
        items = [_code_constant(item) for item in value]
        return {"frozenset": sorted(items, key=digest)}
    if isinstance(value, (float, complex)):
        return {type(value).__name__: repr(value)}
    require(value is None or type(value) in (bool, int, str), "Unsupported loaded code constant")
    return value


def _code_projection(code):
    return {"bytecode": code.co_code.hex(), "constants": [_code_constant(v) for v in code.co_consts],
            "names": code.co_names, "varnames": code.co_varnames, "freevars": code.co_freevars,
            "cellvars": code.co_cellvars, "argcount": code.co_argcount,
            "posonlyargcount": code.co_posonlyargcount, "kwonlyargcount": code.co_kwonlyargcount,
            "flags": code.co_flags, "filename": code.co_filename, "firstlineno": code.co_firstlineno,
            "linetable": code.co_linetable.hex(), "exceptiontable": code.co_exceptiontable.hex()}


def _runtime_dependency_paths():
    roots = ("numpy", "scipy", "torch", "pyarrow", "tokenizers")
    paths = {Path(sys.executable).resolve()}
    for name, module in sys.modules.copy().items():
        if any(name == root or name.startswith(root + ".") for root in roots):
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).is_file():
                paths.add(Path(filename).resolve())
    return paths


def _preflight_input_paths(report_path, report):
    paths = {Path(report_path).resolve(),
             Path(__file__).resolve().parents[1] / "configs/final-span-planning-policy-20261008.json"}
    prepared_roots = set()
    for bundle in report["bundles"]:
        paths.update(Path(row["path"]).resolve() for row in bundle["provenance"])
        selection_path = Path(bundle["selection_path"]).resolve()
        paths.add(selection_path)
        selection = read_json(selection_path)
        paths.update(Path(row["decision_path"]).resolve() for row in selection["grid_inputs"])
        binding = selection["selection_rule"]
        require(binding["schema"] == BINDING_SCHEMA, "Preflight requires the actual nine-manifest binding")
        for row in binding["trials"]:
            prepared_roots.add(Path(row["prepared_path"]).resolve())
        for arm in ARMS:
            directory = Path(bundle["bundle_path"]) / arm
            paths.update(p.resolve() for p in directory.glob("attempt-*.json"))
            paths.update(p.resolve() for p in (directory / "evaluations").glob("stage2-*.json"))
    # Each of the six selections refers to the same nine prepared directories.
    # Traverse that inventory once, retaining new/removed-input invalidation.
    for prepared in prepared_roots:
        paths.update(p.resolve() for p in prepared.rglob("*") if p.is_file())
    accounting = report["cost"].get("accounting_path")
    if accounting:
        paths.add(Path(accounting).resolve())
    return paths


def create_final_span_preflight(report_path, output):
    """Fully verify once, then freeze the exact local evidence/code identity.

    Bind the returned content SHA into the preregistration before using the
    receipt. It applies only to the same retained files and runtime snapshot.
    Any filesystem identity, code or software change requires a fresh full
    verification. This is a cache of a completed proof, not an approval source.
    """
    require(not Path(output).exists(), "Final-span preflight already exists; never overwrite")
    runtime_before = _runtime_environment()
    candidate = read_json(report_path)
    initial_paths = _preflight_input_paths(report_path, candidate)
    initial_paths.update(Path(path) for path in runtime_before["package_source_sha256"])
    initial_paths.update(_runtime_dependency_paths())
    before = {str(path): _file_identity(path) for path in sorted(initial_paths)}
    report = _verify_final_span_report_full(report_path)
    runtime_after = _runtime_environment()
    require(runtime_before == runtime_after, "Runtime code/software changed during full preflight verification")
    paths = _preflight_input_paths(report_path, report)
    paths.update(Path(path) for path in runtime_after["package_source_sha256"])
    # Full torch deserialization can lazily import additional dependencies.
    dependency_paths = _runtime_dependency_paths()
    paths.update(dependency_paths)
    require(initial_paths <= paths and all(_file_identity(path) == identity for path, identity in before.items()),
            "Evidence changed during full preflight verification")
    known_hashes = dict(runtime_after["package_source_sha256"])
    for bundle in report["bundles"]:
        for row in bundle["provenance"]:
            path = str(Path(row["path"]).resolve())
            require(path not in known_hashes or known_hashes[path] == row["sha256"], "Conflicting verified provenance hashes")
            known_hashes[path] = row["sha256"]
    files = []
    for path in sorted(paths):
        identity = _file_identity(path)
        sha = known_hashes.get(str(path)) or checkpoint_digest(path)
        require(_file_identity(path) == identity, "Evidence changed while indexing full preflight")
        files.append({"path": str(path), "sha256": sha, "identity": identity})
    require(all(_file_identity(row["path"]) == row["identity"] for row in files), "Evidence changed before preflight freeze")
    require(_runtime_environment() == runtime_after, "Runtime changed before preflight freeze")
    packet = {"schema": PREFLIGHT_SCHEMA,
              "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "verification": "complete_raw_proof_chain_and_fixed_power_simulation",
              "report_path": str(Path(report_path).resolve()), "report_sha256": report["sha256"],
              "planning_policy_sha256": PLANNING_POLICY_SHA256,
              "runtime": runtime_after, "dependency_paths": sorted(map(str, dependency_paths)),
              "files": files, "input_index_sha256": digest(files),
              "cache_scope": "Same retained local filesystem objects. Device/inode/size/mtime/ctime/mode and source/software identity must remain unchanged. Bind this receipt SHA in the frozen preregistration.",
              "human_review_complete": False, "confirmation_ready": False}
    packet["sha256"] = digest(packet)
    write_json(output, packet)
    return packet


def _verify_preflight(report_path, preflight_path, expected_sha256):
    require(expected_sha256 is not None, "Cheap preflight verification requires the receipt SHA bound in the frozen preregistration")
    receipt = read_json(preflight_path)
    require(receipt.get("schema") == PREFLIGHT_SCHEMA and receipt.get("sha256") == expected_sha256 ==
            digest({k: v for k, v in receipt.items() if k != "sha256"}), "Verified preflight receipt differs from the frozen content hash")
    require(receipt.get("verification") == "complete_raw_proof_chain_and_fixed_power_simulation" and
            receipt["planning_policy_sha256"] == PLANNING_POLICY_SHA256 and
            receipt["report_path"] == str(Path(report_path).resolve()) and
            receipt.get("human_review_complete") is False and receipt.get("confirmation_ready") is False,
            "Verified preflight scope, report or policy differs")
    validate_planning_policy()
    runtime = _runtime_environment()
    require(runtime == receipt["runtime"], "Runtime code/content/software changed; repeat full preflight verification")
    report = read_json(report_path)
    require(report.get("sha256") == receipt["report_sha256"] == digest({k: v for k, v in report.items() if k != "sha256"}) and
            report.get("planning_policy") == PLANNING_POLICY and report.get("planning_policy_sha256") == PLANNING_POLICY_SHA256,
            "Bound final-span report or planning policy changed")
    files = receipt["files"]
    require(receipt["input_index_sha256"] == digest(files) and len(files) == len({row["path"] for row in files}),
            "Preflight file index was changed or duplicated")
    paths = _preflight_input_paths(report_path, report)
    paths.update(Path(path) for path in runtime["package_source_sha256"])
    paths.update(Path(path) for path in receipt["dependency_paths"])
    require(set(map(str, paths)) == {row["path"] for row in files}, "Preflight input inventory changed; repeat full verification")
    require(all(_file_identity(row["path"]) == row["identity"] for row in files),
            "Verified input bytes/filesystem identity changed; repeat full preflight verification")
    # These small records are rehashed at every launch, in addition to ctime.
    small_paths = {str(Path(report_path).resolve())}
    small_paths.update(str(Path(bundle["selection_path"]).resolve()) for bundle in report["bundles"])
    small_paths.add(str((Path(__file__).resolve().parents[1] / "configs/final-span-planning-policy-20261008.json").resolve()))
    hashes = {row["path"]: row["sha256"] for row in files}
    require(all(checkpoint_digest(path) == hashes[path] for path in small_paths),
            "Bound report, policy or selected-dose proof bytes changed")
    return report


def verify_final_span_report(report_path, *, preflight_path=None, expected_preflight_sha256=None):
    if preflight_path is not None:
        return _verify_preflight(report_path, preflight_path, expected_preflight_sha256)
    require(expected_preflight_sha256 is None, "A preflight SHA requires its actual receipt path")
    return _verify_final_span_report_full(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path, action="append")
    inputs.add_argument("--preflight-report", type=Path, help="Fully verify an existing report and freeze its local immutable input index")
    parser.add_argument("--selection-path", type=Path, action="append")
    parser.add_argument("--allocation-accounting", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.preflight_report is not None:
        require(args.selection_path is None and args.allocation_accounting is None,
                "Preflight consumes the already-bound report and its actual inputs")
        receipt = create_final_span_preflight(args.preflight_report, args.output)
        print(json.dumps({key: receipt[key] for key in ("schema", "sha256", "report_sha256", "confirmation_ready")}))
        return
    report = evaluate_final_span(args.bundle, args.output, selection_paths=args.selection_path,
                                 allocation_accounting=args.allocation_accounting)
    print(json.dumps({key: report[key] for key in ("sha256", "assay_usable", "selected_E", "confirmation_ready")}))
    raise SystemExit(0 if report["assay_usable"] else 2)


if __name__ == "__main__":
    main()
