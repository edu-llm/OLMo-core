"""Two-stage acquisition freeze: outcome-independent policy, then manifests."""
from __future__ import annotations

import datetime
from collections import Counter
from pathlib import Path

from .acquisition import GRID_TRAJECTORY_POLICY
from .common import digest, require, write_json
from .prepare import load_prepared, validate_config

CORE_SCHEMA = "spacing-acquisition-grid-policy-core-v1"
BINDING_SCHEMA = "spacing-acquisition-grid-manifest-binding-v1"
RULE_POLICY = "all_nine_gate_then_scaled_mean_nearest_0_55_tie_lower_E_v1"
SOURCE_PATH_KEYS = {"training_unit_grounding", "acquisition_qa_bundle", "acquisition_qa_source_input"}


def aware_time(value):
    time = datetime.datetime.fromisoformat(value)
    require(time.tzinfo is not None, "Acquisition provenance requires timezone-aware timestamps")
    return time


def config_frame(config):
    # Paths can change when artifacts are uploaded; content is bound separately.
    return {key: value for key, value in config.items() if key not in SOURCE_PATH_KEYS}


def validate_policy_core(core):
    require(core.get("schema") == CORE_SCHEMA and core.get("policy") == RULE_POLICY,
            "Unknown acquisition policy core")
    require(core.get("sha256") == digest({k: v for k, v in core.items() if k != "sha256"}),
            "Frozen acquisition policy core was modified")
    aware_time(core["frozen_utc"])
    require(core.get("candidate_exposures") == [2, 3, 4, 6, 8] and core.get("target_min") == .4 and
            core.get("target_max") == .7 and core.get("scaled_mean_target") == .55,
            "Policy core differs from the frozen grid/gates/target")
    require(core.get("source_catalog_sha256") and core.get("dataset_revision"),
            "Policy core requires pinned source catalog and dataset provenance")
    trials = core.get("trials", [])
    require(len(trials) == 9 and len({trial["trial_id"] for trial in trials}) == 9 and
            Counter(trial["development_scale_probe"] for trial in trials) == {False: 6, True: 3},
            "Policy core requires six normal and three scaled independent trial frames")
    for trial in trials:
        frame = trial["config_frame"]
        require(not SOURCE_PATH_KEYS & frame.keys(), "Config frame must bind source contents separately from paths")
        validate_config(dict(frame, **{key: "bound-content-path" for key in SOURCE_PATH_KEYS}))
        require(frame.get("acquisition_trajectory_policy") == GRID_TRAJECTORY_POLICY and
                bool(frame.get("development_scale_probe")) == trial["development_scale_probe"] and
                frame["span"] == 252 and frame["primary_delay"] == 84 and frame["buffer_steps"] == 168 and
                frame["eval_delays"] == [21, 84, 168], "Policy trial geometry or final-span settings differ")
        require(trial.get("roles_sha256"), "Policy trial requires frozen source-only role assignment")
        source = trial["source_frame"]
        if trial["development_scale_probe"]:
            require(source == {"source_catalog_sha256": core["source_catalog_sha256"]},
                    "Scale source frame must bind the reviewed content's original source catalog")
        else:
            require(set(source) == {"grounding_bundle_sha256", "acquisition_qa_bundle_sha256",
                                    "acquisition_source_input_sha256"} and all(source.values()),
                    "Normal trial requires the three approved source artifact hashes")
    for scaled, starts, count in ((False, (2701, 2801, 2901), 6), (True, (3701, 3801, 3901), 3)):
        frames = [trial["config_frame"] for trial in trials if trial["development_scale_probe"] == scaled]
        require(sorted(tuple(frame[key] for key in ("order_seed", "train_seed", "eval_seed")) for frame in frames) ==
                [tuple(start + index for start in starts) for index in range(count)],
                "Policy core trial seed tuples differ from the frozen acquisition amendment")
    return core


def manifest_policy_trial(manifest, core):
    validate_policy_core(core)
    require(manifest["mode"] == "development" and manifest["dataset_revision"] == core["dataset_revision"],
            "Grid manifest mode/dataset differ from frozen policy")
    matches = [trial for trial in core["trials"] if trial["config_frame"] == config_frame(manifest["config"])]
    require(len(matches) == 1, "Grid manifest settings/seeds differ from every predeclared policy trial")
    trial = matches[0]
    require(digest(manifest["split"]["roles"]) == trial["roles_sha256"], "Grid roles differ from predeclared policy")
    geometry = {"old": 15, "new": 1, "control": 1, "qa": 3} if trial["development_scale_probe"] else {
        "old": 5, "new": 7, "control": 5, "qa": 3}
    require(Counter(manifest["split"]["roles"].values()) == geometry, "Grid role counts differ from frozen policy")
    audit = manifest["acquisition_qa_audit"]
    source = ({"source_catalog_sha256": audit.get("source_catalog_sha256")}
              if trial["development_scale_probe"] else {
                  "grounding_bundle_sha256": manifest["grounding_audit"]["bundle_sha256"],
                  "acquisition_qa_bundle_sha256": audit["bundle_sha256"],
                  "acquisition_source_input_sha256": audit["source_input_sha256"]})
    require(source == trial["source_frame"], "Grid source provenance differs from frozen trial frame")
    return trial


def validate_manifest_binding(binding):
    require(binding.get("schema") == BINDING_SCHEMA and
            binding.get("sha256") == digest({k: v for k, v in binding.items() if k != "sha256"}),
            "Acquisition manifest binding was modified")
    core = validate_policy_core(binding["policy_core"])
    require(binding["policy_core_sha256"] == core["sha256"], "Acquisition binding refers to a different frozen core")
    frozen, bound = aware_time(core["frozen_utc"]), aware_time(binding["binding_utc"])
    require(frozen <= bound, "Manifest binding precedes the actual policy freeze")
    rows = binding.get("trials", [])
    require(len(rows) == 9 and len({row["trial_id"] for row in rows}) == 9 and
            {row["trial_id"] for row in rows} == {trial["trial_id"] for trial in core["trials"]} and
            len({row["manifest_sha256"] for row in rows}) == 9,
            "Manifest binding must cover all nine unique predeclared trials")
    validated = []
    for row in rows:
        manifest, _, _ = load_prepared(Path(row["prepared_path"]))
        require(manifest["sha256"] == row["manifest_sha256"], "Bound prepared manifest was modified")
        trial = manifest_policy_trial(manifest, core)
        require(trial["trial_id"] == row["trial_id"], "Bound manifest assigned to a different policy trial")
        validated.append((row, trial, manifest))
    return core, validated


def resolve_manifest_binding(binding, decisions):
    core, validated = validate_manifest_binding(binding)
    frozen, bound = aware_time(core["frozen_utc"]), aware_time(binding["binding_utc"])
    by_manifest = {decision["manifest_sha256"]: decision for decision in decisions}
    require(len(decisions) == len(by_manifest) == 9 and set(by_manifest) == {row["manifest_sha256"] for row, _, _ in validated},
            "Grid decisions differ from the actual nine-manifest binding")
    normal, scaled = [], []
    for row, trial, manifest in validated:
        decision = by_manifest[row["manifest_sha256"]]
        require(decision.get("policy_core_sha256") == core["sha256"] and
                decision.get("policy_trial_id") == trial["trial_id"], "Grid did not launch under its frozen policy core")
        started = aware_time(decision["grid_started_utc"])
        require(frozen <= started, "Acquisition grid started before the policy core freeze")
        if trial["development_scale_probe"]:
            require(bound <= started, "Scale grid started before actual manifest binding")
            require(decision.get("policy_manifest_binding_sha256") == binding["sha256"],
                    "Scale grid did not launch under the actual manifest binding")
            scaled.append(manifest["sha256"])
        else:
            normal.append(manifest["sha256"])
    rule = {"schema": "spacing-acquisition-grid-selection-rule-v1", "policy": RULE_POLICY,
            "frozen_utc": core["frozen_utc"], "normal_manifest_sha256": normal, "scaled_manifest_sha256": scaled,
            **{key: core[key] for key in ("candidate_exposures", "target_min", "target_max", "scaled_mean_target")}}
    rule["sha256"] = digest(rule)
    return rule


def bind_policy_manifests(policy_core, prepared_paths, output):
    core = validate_policy_core(policy_core)
    require(not Path(output).exists(), "Manifest binding already exists; never overwrite")
    rows = []
    for prepared in prepared_paths:
        manifest, _, _ = load_prepared(prepared)
        trial = manifest_policy_trial(manifest, core)
        rows.append({"trial_id": trial["trial_id"], "manifest_sha256": manifest["sha256"],
                     "prepared_path": str(Path(prepared).resolve())})
    require(len(rows) == 9 and len({row["trial_id"] for row in rows}) == 9,
            "Binding requires every predeclared normal/scale manifest exactly once")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    require(aware_time(core["frozen_utc"]) <= aware_time(now), "Policy core freeze is in the future")
    packet = {"schema": BINDING_SCHEMA, "policy_core": core, "policy_core_sha256": core["sha256"],
              "binding_utc": now, "trials": rows}
    packet["sha256"] = digest(packet)
    write_json(output, packet)
    return packet
