"""Select a common acquisition dose from nine frozen development trajectories."""
from __future__ import annotations

import datetime
import hashlib
import math
from decimal import Decimal
from pathlib import Path

from .acquisition import GRID_TRAJECTORY_POLICY
from .common import digest, read_json, require, write_json
from .prepare import load_prepared
from .acquisition_policy import BINDING_SCHEMA, resolve_manifest_binding

RULE_POLICY = "all_nine_gate_then_scaled_mean_nearest_0_55_tie_lower_E_v1"


def checkpoint_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def common_grid_dose(rule, decisions):
    if rule.get("schema") == BINDING_SCHEMA:
        rule = resolve_manifest_binding(rule, decisions)
    require(rule.get("schema") == "spacing-acquisition-grid-selection-rule-v1" and
            rule.get("policy") == RULE_POLICY, "Unknown frozen acquisition dose-selection rule")
    require(rule.get("sha256") == digest({k: v for k, v in rule.items() if k != "sha256"}),
            "Acquisition selection rule was modified")
    normal, scaled = rule.get("normal_manifest_sha256", []), rule.get("scaled_manifest_sha256", [])
    require(len(normal) == 6 and len(scaled) == 3 and len(set(normal + scaled)) == 9,
            "Selection requires exactly six normal and three scaled frozen development manifests")
    require(rule.get("candidate_exposures") == [2, 3, 4, 6, 8] and rule.get("target_min") == .4 and
            rule.get("target_max") == .7 and rule.get("scaled_mean_target") == .55,
            "Acquisition dose-selection grid/gates/target differ from the frozen rule")
    frozen = datetime.datetime.fromisoformat(rule["frozen_utc"])
    require(frozen.tzinfo is not None, "Selection rule requires a timezone-aware frozen timestamp")
    by_manifest = {d["manifest_sha256"]: d for d in decisions}
    require(len(decisions) == len(by_manifest) == 9 and set(by_manifest) == set(normal + scaled),
            "Missing, duplicate or additional development acquisition grids")
    for sha, decision in by_manifest.items():
        require(decision.get("mode") == "development" and decision.get("trajectory_policy") == GRID_TRAJECTORY_POLICY and
                decision.get("status") == "acquisition_grid_complete" and decision.get("selected_E") is None and
                decision.get("acquisition_usable") is False, "Only complete unselected development grids are eligible")
        require(decision.get("development_scale_probe", False) == (sha in scaled), "Scale/normal grid assignment differs")
        require(set(decision["grid_results"]) == {str(e) for e in rule["candidate_exposures"]},
                "Incomplete acquisition grid trajectory")
        started = datetime.datetime.fromisoformat(decision["grid_started_utc"])
        require(started.tzinfo is not None and frozen <= started, "Selection rule was frozen after acquisition started")
        for exposure in rule["candidate_exposures"]:
            result = decision["grid_results"][str(exposure)]
            score = result["old_exact_match_event_macro"]
            require(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1,
                    "Invalid acquisition-grid exact-match score")
            require(result["exposures"] == exposure and result["within_acquisition_gate"] == (.4 <= score <= .7),
                    "Acquisition grid dose/gate metadata differs from its score")
    common = [exposure for exposure in rule["candidate_exposures"] if all(
        .4 <= d["grid_results"][str(exposure)]["old_exact_match_event_macro"] <= .7 for d in decisions)]
    exact_means = {str(exposure): sum(Decimal(str(by_manifest[sha]["grid_results"][str(exposure)][
        "old_exact_match_event_macro"])) for sha in scaled) / Decimal(3) for exposure in rule["candidate_exposures"]}
    means = {exposure: float(mean) for exposure, mean in exact_means.items()}
    selected = min(common, key=lambda e: (abs(exact_means[str(e)] - Decimal("0.55")), e)) if common else None
    return {"selected_E": selected, "common_valid_exposures": common, "scaled_means_by_exposure": means}


def normal_manifest_ids(rule):
    if rule.get("schema") == BINDING_SCHEMA:
        normal = {trial["trial_id"] for trial in rule["policy_core"]["trials"] if not trial["development_scale_probe"]}
        return [row["manifest_sha256"] for row in rule["trials"] if row["trial_id"] in normal]
    return rule["normal_manifest_sha256"]


def choose_acquisition_grid_checkpoint(prepared, grid_outputs, selection_rule, output):
    manifest, _, _ = load_prepared(prepared)
    require(manifest["mode"] == "development" and not manifest["config"].get("development_scale_probe") and
            manifest["config"].get("acquisition_trajectory_policy") == GRID_TRAJECTORY_POLICY,
            "Checkpoint selection is only for a normal development acquisition grid")
    output = Path(output)
    require(not output.exists(), "Acquisition selection output already exists; never overwrite")
    rule = read_json(selection_rule)
    inputs = [{"decision_path": str((Path(path) / "acquisition_decision.json").resolve()),
               "decision": read_json(Path(path) / "acquisition_decision.json")} for path in grid_outputs]
    result = common_grid_dose(rule, [item["decision"] for item in inputs])
    require(manifest["sha256"] in normal_manifest_ids(rule), "Normal manifest is not in the frozen rule")
    own = next(item for item in inputs if item["decision"]["manifest_sha256"] == manifest["sha256"])
    selected_path, selected_hash = None, None
    if result["selected_E"] is not None:
        entry = own["decision"]["grid_results"][str(result["selected_E"])]
        selected_path = str((Path(own["decision_path"]).parent / entry["checkpoint"]).resolve())
        selected_hash = checkpoint_digest(selected_path)
        require(selected_hash == entry["checkpoint_sha256"], "Selected full acquisition checkpoint was modified")
    packet = {"schema": "spacing-acquisition-grid-checkpoint-selection-v1", "mode": "development",
              "manifest_sha256": manifest["sha256"], "selection_rule": rule,
              "selection_rule_sha256": rule["sha256"], "policy": RULE_POLICY,
              "grid_inputs": [{"decision_path": item["decision_path"], "decision_sha256": digest(item["decision"])}
                              for item in inputs], **result,
              "selection_usable": result["selected_E"] is not None,
              "selected_checkpoint": selected_path, "selected_checkpoint_sha256": selected_hash,
              "confirmation_ready": False}
    packet["sha256"] = digest(packet)
    write_json(output, packet)
    return packet


def verify_acquisition_grid_selection(selection_path, manifest, checkpoint):
    require(selection_path is not None, "Full-grid continuation requires a frozen common-dose checkpoint-selection record")
    packet = read_json(selection_path)
    require(packet.get("schema") == "spacing-acquisition-grid-checkpoint-selection-v1" and
            packet.get("sha256") == digest({k: v for k, v in packet.items() if k != "sha256"}),
            "Acquisition checkpoint selection was modified")
    require(manifest["mode"] == "development" and not manifest["config"].get("development_scale_probe") and
            packet.get("mode") == "development" and packet.get("confirmation_ready") is False and
            packet.get("policy") == RULE_POLICY and
            packet.get("manifest_sha256") == manifest["sha256"] and packet.get("selection_usable") is True,
            "No valid common acquisition dose for this normal manifest")
    require(packet["selection_rule_sha256"] == packet["selection_rule"]["sha256"], "Selection rule provenance differs")
    decisions = []
    for source in packet["grid_inputs"]:
        decision = read_json(source["decision_path"])
        require(digest(decision) == source["decision_sha256"], "Acquisition selection input grid was modified")
        decisions.append(decision)
    expected = common_grid_dose(packet["selection_rule"], decisions)
    require(all(packet[key] == value for key, value in expected.items()), "Selected dose differs from the frozen common rule")
    require(manifest["sha256"] in normal_manifest_ids(packet["selection_rule"]),
            "Selected manifest is not a normal development grid")
    own = next(source for source, decision in zip(packet["grid_inputs"], decisions)
               if decision["manifest_sha256"] == manifest["sha256"])
    decision = read_json(own["decision_path"])
    entry = decision["grid_results"][str(packet["selected_E"])]
    expected_path = str((Path(own["decision_path"]).parent / entry["checkpoint"]).resolve())
    require(packet["selected_checkpoint"] == expected_path and
            packet["selected_checkpoint_sha256"] == entry["checkpoint_sha256"],
            "Selected checkpoint differs from the grid's frozen dose checkpoint")
    require(str(Path(checkpoint).resolve()) == packet["selected_checkpoint"] and
            checkpoint_digest(checkpoint) == packet["selected_checkpoint_sha256"], "Shared checkpoint differs from selected dose")
    return packet
