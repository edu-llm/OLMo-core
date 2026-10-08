"""Reviewed development declarations grounded in short sources and event factsheets."""
from __future__ import annotations

import collections
import copy

from .common import digest, require
from .data import REVISION, normalize
from .units import GROUNDED_POLICY, validate_units

FACTSHEETS_SHA256 = "5c4488db6308ee558f7a4598a977234a993c90caad10a9d97e5b13b19a2f5286"


def group_identity(source_unit_ids):
    return "ground-" + digest(sorted(source_unit_ids))[:24]


def validate_bundle(bundle, facts, source_registry, *, mode, partition_hash, roles, pinned_factsheets=None):
    require(mode == "development", "Grounded assertions and reviewed grouping are development-only")
    require(bundle["schema"] == "spacing-grounded-declarations-v1" and bundle["policy"] == GROUNDED_POLICY,
            "Unknown grounded assertion schema")
    require(digest({k: v for k, v in bundle.items() if k != "sha256"}) == bundle["sha256"],
            "Grounding bundle was modified")
    require(bundle["source_dataset_revision"] == REVISION and bundle["source_factsheets_sha256"] == FACTSHEETS_SHA256,
            "Grounding source revision/hash differs")
    require(bundle["partition_sha256"] == partition_hash and bundle["roles_sha256"] == digest(roles),
            "Grounding event assignment differs")
    require(bundle["independent_agent_review_complete"] is True and bundle.get("independent_agent_reviewer"),
            "Grounding/grouping requires completed independent review")
    require(bundle["authoring_inputs"] == ["source_statement", "canonical_answers", "event_factsheet"] and
            bundle["evaluation_question_text_included"] is False and bundle["model_outcomes_used"] is False,
            "Grounding author inputs changed")
    require(bundle["original_evaluation_probe_count"] == len(facts), "Grounding evaluation probe count differs")
    by_probe = {f["id"]: f for f in facts}
    sources = {u["id"]: u for u in source_registry["units"] if u["role"] in ("old", "new")}
    records = bundle["source_units"]
    require(len(records) == len(sources) == bundle["source_unit_count"] and
            {r["unit_id"] for r in records} == set(sources), "Lost or duplicated original source units")
    require({r["index"] for r in records} == set(range(len(records))), "Source audit indices changed")
    used_events = {u["event"] for u in sources.values()}
    require(set(bundle["factsheets"]) == used_events, "Grounding factsheet coverage differs")
    if pinned_factsheets is not None:
        require(all(bundle["factsheets"][event] == pinned_factsheets[event] for event in used_events),
                "Bundle factsheet text differs from pinned parquet")
    groups = bundle["groups"]
    group_ids = {g["id"] for g in groups}
    require(len(group_ids) == len(groups), "Duplicate grounded group IDs")
    by_group = {g["id"]: g for g in groups}
    seen, texts = [], set()
    heldout = {normalize(f["question"]) for f in facts if f["role"] in ("old", "new", "control")}
    complete_questions = {" ".join(f["question"].strip().casefold().split()) for f in facts
                          if f["role"] in ("old", "new", "control")}
    for group in groups:
        source_ids = group["source_unit_ids"]
        require(source_ids == sorted(set(source_ids)) and source_ids and all(k in sources for k in source_ids),
                "Invalid source-unit group membership")
        require(group["id"] == group_identity(source_ids), "Unstable grounded group identity")
        require(all(sources[k]["event"] == group["event"] and sources[k]["role"] == group["role"] for k in source_ids),
                "Grounding group crossed event roles")
        require(group["review_status"] == "approved_development" and group["grouping_rationale"],
                "Unreviewed assertion group")
        statement = group["statement"]
        require(statement.strip() and "?" not in statement and "Question:" not in statement and "Answer:" not in statement,
                "Grounded training text must be a declaration")
        # A relative clause can share words with a question without training
        # that question. Keep punctuation for substring checks; also reject a
        # whole declaration equal to a question after frozen normalization.
        require(normalize(statement) not in heldout and not any(
                question in " ".join(statement.strip().casefold().split()) for question in complete_questions),
                "Held-out question text entered a grounded declaration")
        signature = (group["event"], normalize(statement))
        require(signature not in texts, "Identical grounded declarations must be explicitly grouped")
        texts.add(signature)
        seen.extend(source_ids)
    require(collections.Counter(seen) == collections.Counter(sources.keys()), "Source units lost or counted in multiple groups")
    for record in records:
        source = sources[record["unit_id"]]
        require(record["event_id"] == source["event"] and record["role"] == source["role"] and
                record["source_statement"] == source["source_statement"] and
                record["probe_ids"] == source["member_probe_ids"], "Grounding source assertion/membership differs")
        require(record["canonical_answers"] == [by_probe[p]["answer"] for p in source["member_probe_ids"]],
                "Grounding changed source answer labels")
        require(record["source_statement_sha256"] == digest(source["source_statement"]) and
                record["source_fields_sha256"] == digest({k: record[k] for k in
                    ("unit_id", "event_id", "role", "source_statement", "canonical_answers", "probe_ids")}),
                "Grounding source-field hashes differ")
        require(record["group_id"] in by_group, "Missing grounded group")
        group = by_group[record["group_id"]]
        require(record["unit_id"] in group["source_unit_ids"] and record["index"] in group["source_indices"] and
                record["trained_declaration"] == group["statement"], "Source/group declaration mapping differs")
        require(record["review_status"] == "approved_development", "Unreviewed source declaration")
        text = bundle["factsheets"][record["event_id"]]
        require(record["factsheet_sha256"] == digest(text) and record["evidence"], "Missing factsheet evidence")
        lines = text.splitlines()
        require(all(0 <= e["line"] < len(lines) and e["quote"] == lines[e["line"]] for e in record["evidence"]),
                "Grounding evidence is not an exact pinned factsheet excerpt")
        if record["status"].startswith("preserve_source"):
            require(group["source_unit_ids"] == [record["unit_id"]] and group["statement"] == record["source_statement"],
                    "Unresolved source conflict must retain its original assertion and unit")
        require(record["nonliteral_answer_labels"] == [a for a in sorted(set(record["canonical_answers"]))
                if normalize(a) not in normalize(group["statement"])], "Grounding answer-label coverage audit differs")
    for group in groups:
        require(sorted(group["source_indices"]) == sorted(r["index"] for r in records if r["group_id"] == group["id"]),
                "Grounding audit-group indices differ")


def build_grounded_registry(facts, source_registry, bundle):
    by_probe = {f["id"]: f for f in facts}
    source_by_id = {u["id"]: u for u in source_registry["units"]}
    units = []
    for source in source_registry["units"]:
        for pid in source["member_probe_ids"]:
            by_probe[pid]["source_unit_id"] = source["id"]
        if source["role"] not in ("old", "new"):
            units.append(dict(source, source_unit_ids=[source["id"]]))
    for group in bundle["groups"]:
        probes = sorted(pid for sid in group["source_unit_ids"] for pid in source_by_id[sid]["member_probe_ids"])
        unit = {"id": group["id"], "event": group["event"], "role": group["role"],
                "source_unit_ids": group["source_unit_ids"], "statement": group["statement"],
                "member_probe_ids": probes, "canonical_probe_id": probes[0],
                "answer_labels": sorted({normalize(by_probe[p]["answer"]) for p in probes}),
                "representation": "reviewed_grounded_assertion"}
        units.append(unit)
        for pid in probes:
            by_probe[pid]["unit_id"] = unit["id"]
            by_probe[pid]["statement"] = unit["statement"]
    registry = {"policy": GROUNDED_POLICY, "units": sorted(units, key=lambda u: u["id"]),
                "source_registry_sha256": source_registry["sha256"], "grounding_bundle_sha256": bundle["sha256"],
                "probe_counts": dict(collections.Counter(f["role"] for f in facts)),
                "unit_counts": dict(collections.Counter(u["role"] for u in units)),
                "aggregation": "mean_probes_within_unit_then_units_within_event_then_events"}
    registry["sha256"] = digest(registry)
    return registry


def validate_grounded_registry(facts, registry, source_registry, bundle):
    sources = {u["id"]: u for u in source_registry["units"]}
    original = []
    for fact in facts:
        require(fact.get("source_unit_id") in sources, "Missing original source-unit provenance")
        unit = sources[fact["source_unit_id"]]
        original.append(dict(fact, unit_id=unit["id"], statement=unit["statement"]))
    validate_units(original, source_registry)
    expected_facts = copy.deepcopy(original)
    expected = build_grounded_registry(expected_facts, source_registry, bundle)
    require(registry == expected, "Grounded registry differs from the reviewed group ledger")
    require(all(all(actual[k] == frozen[k] for k in ("unit_id", "source_unit_id", "statement"))
                for actual, frozen in zip(facts, expected_facts)), "Grounded probe membership/declaration differs")
