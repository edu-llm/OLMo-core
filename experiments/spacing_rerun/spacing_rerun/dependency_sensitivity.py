"""Prospective scale-source dependency sensitivity, separate from dose selection.

Run ``python -m spacing_rerun.dependency_sensitivity --help`` for the external
operator interface. This module does not change training or evaluation scores.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import math
import resource
import sys
import time
from pathlib import Path

from .acquisition import GRID_TRAJECTORY_POLICY
from .acquisition_policy import aware_time, validate_manifest_binding
from .common import digest, read_json, require, write_json
from .data import normalize
from .evaluation import aggregate
from .prepare import load_prepared
from .units import GROUNDED_METRICS, GROUNDED_POLICY

POLICY_SCHEMA = "spacing-source-dependency-policy-v1"
BINDING_SCHEMA = "spacing-source-dependency-pregrid-binding-v1"
REPORT_SCHEMA = "spacing-source-dependency-grid-sensitivity-v1"
GRID = [2, 3, 4, 6, 8]
PRIMARY = "mean_probes_within_group_then_groups_within_event_then_events"
SECONDARY = "mean_probes_within_group_then_groups_within_dependency_cluster_then_equal_clusters_within_event_then_events"


def seal(payload):
    return dict(payload, sha256=digest(payload))


def verify_seal(payload, label):
    require(payload.get("sha256") == digest({k: v for k, v in payload.items() if k != "sha256"}),
            f"{label} was modified")


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique(rows, field, label):
    index = {row[field]: row for row in rows}
    require(len(index) == len(rows), f"Duplicate {label}")
    return index


def validate_policy(policy):
    verify_seal(policy, "Dependency policy")
    require(policy.get("schema") == POLICY_SCHEMA and policy.get("candidate_exposures") == GRID,
            "Unknown dependency policy or incomplete exposure grid")
    require(policy.get("primary_aggregation") == PRIMARY and policy.get("secondary_aggregation") == SECONDARY and
            policy.get("diagnostic_only") is True and policy.get("dose_selection_permitted") is False and
            policy.get("arm_eligibility_permitted") is False and policy.get("target_min") == .4 and
            policy.get("target_max") == .7, "Dependency sensitivity changed primary eligibility")
    aware_time(policy["frozen_utc"])
    require(policy.get("scale_outcomes_seen_at_freeze") is False and policy.get("normal_grid_outcomes_seen") is True,
            "Dependency policy must disclose which development outcomes were already observed")
    require(policy.get("source_catalog_sha256") and policy.get("authored_chunk_sha256") and
            policy.get("policy_core_sha256") and policy.get("dataset_revision"), "Missing source/core provenance")
    seen = set()
    clusters = unique(policy["dependency_sets"], "id", "dependency set")
    require(clusters, "No declared dependency sets")
    for cluster in clusters.values():
        groups = unique(cluster["groups"], "id", "dependency group")
        require(len(groups) >= 2 and cluster["role"] == "old", "Dependency set must contain multiple old groups")
        for group in groups.values():
            require(group["id"] not in seen, "Group occurs in multiple dependency sets")
            seen.add(group["id"])
            require(group["event"] == cluster["event"] and group["source_unit_ids"] and
                    group["content_sha256"] == digest({k: group[k] for k in
                        ("id", "event", "source_unit_ids", "statement")}), "Wrong dependency group content or event")
            sources = unique(group["sources"], "unit_id", "dependency source unit")
            require(set(sources) == set(group["source_unit_ids"]), "Dependency source-unit membership differs")
            for source in sources.values():
                require(source["event"] == cluster["event"] and source["source_unit_sha256"] and
                        source["canonical_answer_labels"] and source["source_assertion"] and
                        source["content_sha256"] == digest({k: v for k, v in source.items() if k != "content_sha256"}),
                        "Wrong dependency source content or event")
                if source["status"].startswith("preserve_source"):
                    require(group["source_unit_ids"] == [source["unit_id"]] and
                            group["statement"] == source["source_assertion"], "Preserved source must remain a raw singleton")
    return policy


def validate_membership(policy, manifest, schedule, grounding):
    """Bind raw assertions, declaration groups, probes and event-level old roles."""
    validate_policy(policy)
    verify_seal(manifest, "Prepared manifest")
    verify_seal(schedule, "Prepared schedule")
    verify_seal(grounding, "Grounding declarations")
    require(manifest["mode"] == "development" and manifest["config"].get("development_scale_probe") is True and
            manifest["config"].get("acquisition_trajectory_policy") == GRID_TRAJECTORY_POLICY and
            manifest["rehearsal_unit_policy"] == GROUNDED_POLICY, "Dependency diagnostic requires a grounded scale grid")
    require(manifest["dataset_revision"] == policy["dataset_revision"] and
            manifest["acquisition_qa_audit"]["source_catalog_sha256"] == policy["source_catalog_sha256"] ==
            grounding["source_catalog_sha256"], "Dependency source catalog/revision differs")
    require(manifest["schedule_sha256"] == schedule["sha256"] and
            manifest["grounding_audit"]["bundle_sha256"] == grounding["sha256"], "Prepared source/schedule binding differs")
    units = unique(manifest["unit_registry"]["units"], "id", "grounded unit")
    sources = unique(manifest["source_unit_registry"]["units"], "id", "raw source unit")
    ground_groups = unique(grounding["groups"], "id", "grounding group")
    ground_sources = unique(grounding["source_units"], "unit_id", "grounding source")
    facts = unique(manifest["split"]["facts"], "id", "prepared probe")
    all_members = []
    for unit in units.values():
        members = unit["member_probe_ids"]
        require(members and len(set(members)) == len(members), "Missing or duplicate unit probe membership")
        require(all(pid in facts and all(facts[pid][key] == unit[key] for key in ("event", "role")) and
                    facts[pid]["unit_id"] == unit["id"] for pid in members), "Prepared unit/probe membership differs")
        all_members.extend(members)
    require(collections.Counter(all_members) == collections.Counter(facts.keys()), "Prepared probe coverage differs")
    placements = []
    for cluster in policy["dependency_sets"]:
        require(manifest["split"]["roles"].get(cluster["event"]) == "old", "Dependency event is not entirely old")
        starts = {}
        for group in cluster["groups"]:
            gid = group["id"]
            require(gid in units and gid in ground_groups, "Missing dependency group")
            unit, actual = units[gid], ground_groups[gid]
            for key in ("event", "source_unit_ids", "statement"):
                require(unit[key] == actual[key] == group[key], "Dependency group raw content/membership differs")
            require(unit["role"] == actual["role"] == "old", "Dependency group crossed event/role")
            expected_probes = []
            for source in group["sources"]:
                sid = source["unit_id"]
                require(sid in sources and sid in ground_sources, "Missing dependency source unit")
                raw, prepared = sources[sid], ground_sources[sid]
                require(raw["event"] == prepared["event_id"] == source["event"] and
                        raw["role"] == prepared["role"] == "old" and
                        raw["source_statement"] == prepared["source_statement"] == source["source_assertion"] and
                        prepared["source_unit_sha256"] == source["source_unit_sha256"] and
                        prepared["status"] == source["status"] and prepared["group_id"] == gid,
                        "Dependency source raw content/hash/status/role differs")
                require(raw["member_probe_ids"] == prepared["probe_ids"] and
                        sorted(set(prepared["canonical_answers"])) == sorted(set(source["canonical_answer_labels"])) and
                        all(facts[pid]["source_unit_id"] == sid for pid in raw["member_probe_ids"]),
                        "Dependency source gold or probe membership differs")
                expected_probes.extend(raw["member_probe_ids"])
            require(sorted(expected_probes) == sorted(unit["member_probe_ids"]), "Dependency group lost or gained probes")
            require(gid in schedule["starts"], "Dependency group has no declared schedule placement")
            starts[gid] = schedule["starts"][gid]
        placements.append({"dependency_set": cluster["id"], "event": cluster["event"], "starts": starts,
                           "placement": "shared" if len(set(starts.values())) == 1 else "mixed_schedule",
                           "review_arms_executed": False,
                           "scope": "scale acquisition-only grid; stage2 schedule is compiled but never executed"})
    return placements


def _input(path):
    return {"path": str(Path(path).resolve()), "file_sha256": file_sha256(path)}


def _verify_input(record):
    require(file_sha256(record["path"]) == record["file_sha256"], "Dependency bound input file was modified")


def bind_pregrid(policy_path, prepared, final_source, output):
    """Create an actual UTC external receipt before submitting a scale grid."""
    require(not Path(output).exists(), "Dependency binding already exists; never overwrite")
    policy = validate_policy(read_json(policy_path))
    manifest, schedule, _ = load_prepared(prepared)
    grounding = read_json(Path(prepared) / "grounding-declarations.json")
    placements = validate_membership(policy, manifest, schedule, grounding)
    authored = read_json(final_source)
    verify_seal(authored, "Final source author artifact")
    require(authored["sha256"] == policy["authored_chunk_sha256"] and
            authored["source_catalog_sha256"] == policy["source_catalog_sha256"], "Final authored source binding differs")
    actual_groups = unique(authored["groups"], "id", "authored group")
    actual_sources = unique(authored["source_units"], "unit_id", "authored source")
    for cluster in policy["dependency_sets"]:
        for group in cluster["groups"]:
            require(digest(actual_groups[group["id"]]) == group["authored_group_content_sha256"] and
                    all(actual_groups[group["id"]][key] == group[key] for key in
                        ("id", "event", "statement", "source_unit_ids")), "Final authored group differs")
            for source in group["sources"]:
                require(all(actual_sources[source["unit_id"]][key] == value for key, value in source.items()
                            if key != "content_sha256"), "Final authored source differs")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    require(aware_time(policy["frozen_utc"]) <= aware_time(now), "Dependency policy freeze is in the future")
    require(aware_time(manifest["created_utc"]) <= aware_time(now), "Prepared manifest is from the future")
    packet = seal({"schema": BINDING_SCHEMA, "binding_utc": now, "policy_sha256": policy["sha256"],
                   "manifest_sha256": manifest["sha256"], "prepared_path": str(Path(prepared).resolve()),
                   "placements": placements, "inputs": [_input(path) for path in
                       (policy_path, Path(prepared) / "manifest.json", Path(prepared) / "schedule.json",
                        Path(prepared) / "grounding-declarations.json", final_source)],
                   "diagnostic_only": True, "scale_outcomes_used": False})
    write_json(output, packet)
    return packet


def cluster_metric(rows, dependency_sets, metric):
    """Use all probes; each undeclared group is its own dependency cluster."""
    mapping = {(cluster["event"], group["id"]): "dependency/" + cluster["id"]
               for cluster in dependency_sets for group in cluster["groups"]}
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in rows:
        value = row.get(metric)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
                "Missing or non-finite dependency metric")
        require((metric != "exact_match" or value in (0, 1)) and (metric != "loss" or value >= 0),
                "Invalid dependency metric")
        grouped[row["event"]][row["unit_id"]].append(float(value))
    events, cluster_counts = {}, {}
    for event, units in grouped.items():
        clusters = collections.defaultdict(list)
        for gid, values in units.items():
            clusters[mapping.get((event, gid), "group/" + gid)].append(sum(values) / len(values))
        means = [sum(values) / len(values) for values in clusters.values()]
        events[event] = sum(means) / len(means)
        cluster_counts[event] = len(means)
    require(events, "No old probes for dependency diagnostic")
    return {"event_macro": sum(events.values()) / len(events), "by_event": events,
            "cluster_counts_by_event": cluster_counts}


def process_io():
    path = Path("/proc/self/io")
    if not path.exists():
        return None
    counters = dict(line.split(":", 1) for line in path.read_text().splitlines())
    return {key: int(counters[key]) for key in ("rchar", "read_bytes")}


def checkpoint_progress(path, manifest):
    """Inspect trusted immutable checkpoint metadata without materializing tensor storage."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    require(state.get("schema") == manifest["schema"] and state.get("manifest_sha256") == manifest["sha256"] and
            all(key in state for key in ("model", "optimizer", "rng", "progress")),
            "Dose checkpoint is not the bound full-state schema")
    return state["progress"]


def grid_sensitivity(policy_path, prepared, pregrid_binding, manifest_binding, stage1, output):
    began = time.monotonic()
    io_before = process_io()
    checkpoint_bytes_hashed = 0
    require(not Path(output).exists(), "Dependency diagnostic output already exists; never overwrite")
    policy = validate_policy(read_json(policy_path))
    receipt = read_json(pregrid_binding)
    verify_seal(receipt, "Dependency pre-grid binding")
    require(receipt.get("schema") == BINDING_SCHEMA and receipt.get("diagnostic_only") is True and
            receipt.get("scale_outcomes_used") is False and receipt["policy_sha256"] == policy["sha256"],
            "Invalid dependency pre-grid binding")
    for record in receipt["inputs"]:
        _verify_input(record)
    expected_inputs = [_input(path) for path in (policy_path, Path(prepared) / "manifest.json",
                        Path(prepared) / "schedule.json", Path(prepared) / "grounding-declarations.json")]
    require(len(receipt["inputs"]) == 5 and receipt["inputs"][:4] == expected_inputs and
            receipt["prepared_path"] == str(Path(prepared).resolve()),
            "Dependency receipt must retain exact policy, manifest, schedule, grounding and final-source inputs")
    authored = read_json(receipt["inputs"][4]["path"])
    verify_seal(authored, "Final source author artifact")
    require(authored["sha256"] == policy["authored_chunk_sha256"] and
            authored["source_catalog_sha256"] == policy["source_catalog_sha256"], "Final authored source binding differs")
    manifest, schedule, _ = load_prepared(prepared)
    placements = validate_membership(policy, manifest, schedule, read_json(Path(prepared) / "grounding-declarations.json"))
    require(receipt["manifest_sha256"] == manifest["sha256"] and receipt["placements"] == placements,
            "Dependency receipt refers to another manifest/placement")
    binding = read_json(manifest_binding)
    core, bound = validate_manifest_binding(binding)
    require(core["sha256"] == policy["policy_core_sha256"], "Dependency policy core differs")
    matches = [(row, trial) for row, trial, actual in bound if actual["sha256"] == manifest["sha256"]]
    require(len(matches) == 1 and matches[0][1]["development_scale_probe"] is True,
            "Scale manifest is not in the actual nine-manifest binding")
    decision_path = Path(stage1) / "acquisition_decision.json"
    decision = read_json(decision_path)
    require(decision.get("schema") == manifest["schema"] and decision.get("manifest_sha256") == manifest["sha256"] and
            decision.get("mode") == "development" and decision.get("trajectory_policy") == GRID_TRAJECTORY_POLICY and
            decision.get("development_scale_probe") is True and decision.get("status") == "acquisition_grid_complete" and
            decision.get("selected_E") is None and decision.get("acquisition_usable") is False and
            decision.get("final_exposures") == 8 and set(decision["grid_results"]) == {str(e) for e in GRID},
            "Dependency diagnostic requires a complete unselected scale trajectory")
    started = aware_time(decision["grid_started_utc"])
    require(started <= datetime.datetime.now(datetime.timezone.utc) and
            aware_time(policy["frozen_utc"]) <= aware_time(receipt["binding_utc"]) <= started and
            aware_time(manifest["created_utc"]) <= aware_time(receipt["binding_utc"]) and
            aware_time(binding["binding_utc"]) <= started, "Dependency policy/binding was attached after grid training started")
    require(decision.get("policy_core_sha256") == core["sha256"] and
            decision.get("policy_manifest_binding_sha256") == binding["sha256"] and
            decision.get("policy_trial_id") == matches[0][1]["trial_id"], "Decision differs from actual core/manifest binding")
    facts = {f["id"]: f for f in manifest["split"]["facts"] if f["role"] in ("old", "new", "control")}
    results, inputs = [], [_input(path) for path in (policy_path, pregrid_binding, manifest_binding, decision_path)]
    for exposure in GRID:
        path = Path(stage1) / "evaluations" / f"acquisition-E{exposure:03}.json"
        evaluation = read_json(path)
        require(evaluation.get("schema") == manifest["schema"] and evaluation.get("manifest_sha256") == manifest["sha256"] and
                evaluation.get("metric_schema") == GROUNDED_METRICS and
                evaluation.get("tag") == f"acquisition-E{exposure:03}" and evaluation.get("epoch") == exposure and
                evaluation.get("stage2_step") is None, "Evaluation schema/manifest/exposure differs")
        canonical = evaluation["variants"]["canonical"]
        rows = unique(canonical["facts"], "id", "evaluation probe")
        require(set(rows) == set(facts) and all(all(row[k] == facts[pid][k] for k in
                    ("unit_id", "event", "role")) for pid, row in rows.items()), "Evaluation probe/unit/role membership differs")
        for pid, row in rows.items():
            fact = facts[pid]
            require(isinstance(row.get("prediction"), str) and
                    row.get("exact_match") == int(normalize(row["prediction"]) in {normalize(a) for a in fact["aliases"]}),
                    "Evaluation exact match differs from frozen canonical acceptance")
            tokens, nll = row.get("answer_tokens"), row.get("answer_nll_sum")
            require(isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0 and
                    tokens == len(fact["probe"]["answer_ids"]) and isinstance(nll, (int, float)) and
                    math.isfinite(nll) and nll >= 0 and isinstance(row.get("loss"), (int, float)) and
                    math.isclose(row["loss"], nll / tokens, abs_tol=1e-12, rel_tol=0),
                    "Evaluation loss differs from original gold answer-token NLL")
        old = [row for row in canonical["facts"] if row["role"] == "old"]
        primary = aggregate(old)["old"]
        entry = decision["grid_results"][str(exposure)]
        checkpoint_path = Path(stage1) / f"stage1-E{exposure:03}.pt"
        actual_checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint_bytes_hashed += checkpoint_path.stat().st_size
        require(entry.get("checkpoint") == checkpoint_path.name and
                entry.get("checkpoint_sha256") == actual_checkpoint_sha, "Dose checkpoint file/hash differs")
        progress = checkpoint_progress(checkpoint_path, manifest)
        require(progress["epoch"] == exposure and progress["epoch_cursor"] == 0 and
                progress["status"] == "stage1_complete" and
                progress["acquisition_usable"] == entry["within_acquisition_gate"] and
                progress["global_step"] == evaluation.get("global_step") and
                progress["acquisition_qa_tokens_total"] == evaluation.get("acquisition_qa_tokens_total"),
                "Evaluation dose/step/token clocks differ from actual dose checkpoint")
        targets = {record["id"] for record in manifest["acquisition_qa_pool"]["records"]}
        require(set(progress["acquisition_qa_target_doses"]) == targets and
                all(value == exposure for value in progress["acquisition_qa_target_doses"].values()) and
                set(progress["acquisition_qa_variant_doses"]) == targets and
                all(value == [(exposure + 1) // 2, exposure // 2]
                    for value in progress["acquisition_qa_variant_doses"].values()),
                "Actual checkpoint source-QA exposure/variant clocks differ")
        inputs.append({"path": str(checkpoint_path.resolve()), "file_sha256": actual_checkpoint_sha})
        for metric in ("exact_match", "loss"):
            collapsed = cluster_metric(old, policy["dependency_sets"], metric)
            value = primary[metric + "_event_macro"]
            require(math.isclose(value, canonical["aggregate"]["old"][metric + "_event_macro"], abs_tol=1e-12, rel_tol=0),
                    "Existing primary aggregation differs from original probe scores")
            if metric == "exact_match":
                require(entry["exposures"] == exposure and entry["old_exact_match_event_macro"] == value and
                        entry["within_acquisition_gate"] == (.4 <= value <= .7), "Decision primary score/gate differs")
            results.append({"exposures": exposure, "metric": metric, "primary_event_macro": value,
                            "equal_dependency_cluster_event_macro": collapsed["event_macro"],
                            "secondary_minus_primary": collapsed["event_macro"] - value,
                            "secondary_by_event": collapsed["by_event"],
                            "cluster_counts_by_event": collapsed["cluster_counts_by_event"]})
        inputs.append(_input(path))
    io_after = process_io()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    measurements = {"elapsed_seconds": time.monotonic() - began,
                    "checkpoint_bytes_stream_hashed": checkpoint_bytes_hashed,
                    "host_peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
                    "host_peak_rss_scope": "entire diagnostic process high-water mark",
                    "process_io_delta": {key: io_after[key] - io_before[key] for key in io_before} if io_before else None,
                    "checkpoint_inspection": "sequential CPU mmap metadata; tensor storage not materialized"}
    packet = seal({"schema": REPORT_SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "policy_sha256": policy["sha256"], "manifest_sha256": manifest["sha256"],
                   "policy_manifest_binding_sha256": binding["sha256"], "grid_started_utc": decision["grid_started_utc"],
                   "pregrid_binding_sha256": receipt["sha256"], "diagnostic_only": True,
                   "dose_selection_permitted": False, "arm_eligibility_permitted": False,
                   "primary_aggregation": PRIMARY, "secondary_aggregation": SECONDARY,
                   "all_old_probes_retained": True, "placements": placements, "results": results, "inputs": inputs,
                   "verification_measurements": measurements})
    write_json(output, packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bind = commands.add_parser("bind", help="Bind policy to reviewed prepared inputs before scale submission")
    report = commands.add_parser("report", help="Verify all completed acquisition doses and write sensitivity")
    for command in (bind, report):
        for flag in ("policy", "prepared", "output"):
            command.add_argument("--" + flag, required=True)
    bind.add_argument("--final-source", required=True)
    for flag in ("pregrid-binding", "manifest-binding", "stage1"):
        report.add_argument("--" + flag, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "bind":
        bind_pregrid(args.pop("policy"), args.pop("prepared"), args.pop("final_source"), args.pop("output"))
    else:
        grid_sensitivity(args.pop("policy"), args.pop("prepared"), args.pop("pregrid_binding"),
                         args.pop("manifest_binding"), args.pop("stage1"), args.pop("output"))


if __name__ == "__main__":
    main()
