"""Separate supporting-statement rehearsal units from their QA probes."""
from __future__ import annotations

import collections

from .common import digest, require
from .data import normalize

UNIT_POLICY = "source_statement_v1"
GROUNDED_POLICY = "grounded_assertion_v1"
LEGACY_POLICY = "question_root_v1"
UNIT_METRICS = "spacing-metrics-v2-unit-macro"
GROUNDED_METRICS = "spacing-metrics-v3-grounded-unit-macro"
LEGACY_METRICS = "spacing-metrics-v1-fact-macro"


def source_statement(fact):
    return fact.get("source_statement", fact["statement"])


def unit_identity(event, statement):
    signature = {"event": event, "normalized_source_statement": normalize(statement)}
    return "unit-" + digest(signature)[:24], signature


def build_units(facts):
    """Retain every probe; select one full declarative representation per unit."""
    groups = collections.defaultdict(list)
    for fact in facts:
        source = source_statement(fact)
        unit_id, signature = unit_identity(fact["event"], source)
        fact["unit_id"] = unit_id
        fact["source_statement"] = source
        fact["candidate_statement"] = fact.get("candidate_statement", fact["statement"])
        groups[unit_id].append(fact)
    units = []
    for unit_id, probes in sorted(groups.items()):
        probes = sorted(probes, key=lambda f: f["id"])
        require(len({f["event"] for f in probes}) == 1 and len({f["role"] for f in probes}) == 1,
                "A supporting statement crossed event roles")
        first = probes[0]
        _, signature = unit_identity(first["event"], first["source_statement"])
        require(all(unit_identity(f["event"], f["source_statement"])[1] == signature for f in probes),
                "Rehearsal unit hash collision")
        answers = sorted({normalize(f["answer"]) for f in probes})
        multiple_answers = len(answers) > 1
        statement = first["source_statement"] if multiple_answers else first["candidate_statement"]
        units.append({"id": unit_id, "event": first["event"], "role": first["role"],
                      "source_signature": signature, "source_statement": first["source_statement"],
                      "statement": statement, "member_probe_ids": [f["id"] for f in probes],
                      "canonical_probe_id": first["id"], "answer_labels": answers,
                      "representation": "source_multiple_answers" if multiple_answers else "canonical_probe_representation"})
    # Distinct source units can acquire the same wording after rewriting. Revert
    # such units to source text rather than introduce duplicate rehearsal doses.
    while True:
        by_text = collections.defaultdict(list)
        for unit in units:
            if unit["role"] in ("old", "new"):
                by_text[(unit["event"], normalize(unit["statement"]))].append(unit)
        collisions = [group for group in by_text.values() if len(group) > 1]
        if not collisions:
            break
        changed = False
        for group in collisions:
            for unit in group:
                if unit["statement"] != unit["source_statement"]:
                    unit["statement"] = unit["source_statement"]
                    unit["representation"] = "source_rewrite_collision"
                    changed = True
        require(changed, "Distinct source rehearsal units have indistinguishable original text")
    by_id = {unit["id"]: unit for unit in units}
    for fact in facts:
        if fact["role"] in ("old", "new"):
            fact["statement"] = by_id[fact["unit_id"]]["statement"]
    registry = {"policy": UNIT_POLICY, "units": units,
                "probe_counts": dict(collections.Counter(f["role"] for f in facts)),
                "unit_counts": dict(collections.Counter(u["role"] for u in units)),
                "representation_counts": dict(collections.Counter(u["representation"] for u in units)),
                "aggregation": "mean_probes_within_unit_then_units_within_event_then_events"}
    registry["sha256"] = digest(registry)
    validate_units(facts, registry)
    return registry


def validate_units(facts, registry):
    require(registry["policy"] == UNIT_POLICY, "Unknown rehearsal unit policy")
    require(digest({k: v for k, v in registry.items() if k != "sha256"}) == registry["sha256"],
            "Rehearsal unit registry was modified")
    by_probe = {f["id"]: f for f in facts}
    require(len(by_probe) == len(facts), "Duplicate probe IDs")
    seen, training_texts = [], set()
    for unit in registry["units"]:
        expected_id, signature = unit_identity(unit["event"], unit["source_statement"])
        require(unit["id"] == expected_id and unit["source_signature"] == signature, "Unstable source unit identity")
        require(unit["member_probe_ids"] == sorted(set(unit["member_probe_ids"])) and unit["member_probe_ids"],
                "Invalid unit probe membership")
        require(unit["canonical_probe_id"] == unit["member_probe_ids"][0], "Unit canonical probe changed")
        require(unit["answer_labels"] == sorted({normalize(by_probe[pid]["answer"])
                for pid in unit["member_probe_ids"] if pid in by_probe}), "Unit answer labels changed")
        if len(unit["answer_labels"]) > 1:
            require(unit["statement"] == unit["source_statement"], "Multiple-answer unit lost original content")
        for pid in unit["member_probe_ids"]:
            require(pid in by_probe, "Unit contains an unknown probe")
            fact = by_probe[pid]
            require(fact["unit_id"] == unit["id"] and fact["event"] == unit["event"] and
                    fact["role"] == unit["role"], "Probe/unit role mismatch")
            require(unit_identity(fact["event"], source_statement(fact))[1] == signature,
                    "Unit merged different supporting statements")
            if fact["role"] in ("old", "new"):
                require(fact["statement"] == unit["statement"], "Probe points to a different trained representation")
            seen.append(pid)
        if unit["role"] in ("old", "new"):
            key = (unit["event"], normalize(unit["statement"]))
            require(key not in training_texts, "Duplicate training statement units")
            training_texts.add(key)
    require(collections.Counter(seen) == collections.Counter(by_probe.keys()), "Lost or duplicated evaluation probes")
    require(registry["probe_counts"] == dict(collections.Counter(f["role"] for f in facts)), "Probe count audit mismatch")
    require(registry["unit_counts"] == dict(collections.Counter(u["role"] for u in registry["units"])),
            "Rehearsal unit count audit mismatch")


def schedule_records(facts, registry=None, qa_teaching_records=None):
    if registry is None:
        return facts if qa_teaching_records is None else [f for f in facts if f["role"] != "qa"] + qa_teaching_records
    by_probe = {f["id"]: f for f in facts}
    records = []
    for unit in registry["units"]:
        if unit["role"] in ("old", "new"):
            record = dict(unit)
            record["statement_ids"] = by_probe[unit["canonical_probe_id"]].get("statement_ids", [])
            records.append(record)
    # QA teaching is a question-answer stream, not repeated old/new statements.
    records.extend(f for f in facts if f["role"] == "control" or (f["role"] == "qa" and qa_teaching_records is None))
    if qa_teaching_records is not None:
        records.extend(qa_teaching_records)
    return records


def training_key(fact):
    return fact.get("unit_id", fact["id"]) if fact["role"] in ("old", "new") else fact["id"]
