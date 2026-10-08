"""Build unreviewed, source-only authoring packets from pinned files.

This utility performs no semantic authoring or approval. Evaluation question text
is used internally only to verify that it never enters an author packet.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import math
from pathlib import Path
import re

from .common import digest, require, write_json
from .data import REVISION, canonicalize, normalize, outer_partition, assign_roles
from .grounding import FACTSHEETS_SHA256
from .units import build_units


QA_SHA256 = "c178467bfd7db74b17c2d57333bb8908f881c0e1abfbe216c7db75f0934765b7"
OUTER_SEED = 20261007
SPLIT_SEEDS = tuple(range(2026100801, 2026100833))
FORBIDDEN_KEYS = {"question", "questions", "paraphrase", "paraphrase_probe", "mcq",
                  "prediction", "predictions", "model_outcomes", "probe", "prompt",
                  "prompt_ids", "question_id", "span_answer"}


def sealed(payload):
    return dict(payload, sha256=digest(payload))


def opaque_probe_id(probe_id):
    return "probe-" + digest([REVISION, str(probe_id)])


def source_texts(payload):
    """Check serialized text fields while deliberately ignoring opaque hashes."""
    if isinstance(payload, dict):
        require(not (set(payload) & FORBIDDEN_KEYS), "Evaluation field entered a source-only packet")
        for key, value in payload.items():
            if key.endswith("sha256") or key.endswith("_id") or key.endswith("_ids"):
                continue
            yield from source_texts(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from source_texts(value)
    elif isinstance(payload, str):
        yield " ".join(payload.casefold().split())


def reject_question_text(payload, forbidden_questions):
    normalized_questions = {normalize(question) for question in forbidden_questions}
    for text in source_texts(payload):
        # Preserve punctuation for substring checks: a source relative clause
        # can share words with a question without containing its complete form.
        require(normalize(text) not in normalized_questions and
                not any(question in text for question in forbidden_questions),
                "Complete source evaluation question text entered a source-only packet")


def evidence_candidates(assertion, answers, factsheet):
    """Literal matching suggests evidence lines; it does not verify semantics."""
    assertion_words = set(re.findall(r"\w+", assertion.casefold()))
    normalized_answers = [normalize(answer) for answer in answers]
    candidates = []
    for index, line in enumerate(factsheet.splitlines()):
        if not line.strip():
            continue
        words = set(re.findall(r"\w+", line.casefold()))
        answer_matches = [answer for answer in normalized_answers if answer and answer in normalize(line)]
        overlap = len(assertion_words & words) / max(len(assertion_words), 1)
        if answer_matches or overlap >= .5:
            candidates.append({"line": index, "quote": line,
                               "literal_answer_matches": answer_matches,
                               "assertion_word_overlap": overlap})
    candidates.sort(key=lambda row: (-bool(row["literal_answer_matches"]),
                                     -row["assertion_word_overlap"], row["line"]))
    return sorted(candidates[:6], key=lambda row: row["line"])


def entity_inventory(factsheet_rows, assertions):
    """Produce candidate entity/fact links using source fields, never QA wording."""
    names = collections.defaultdict(list)
    for row in factsheet_rows:
        for field in ("entities", "locations"):
            for index, line in enumerate(str(row.get(field, "")).splitlines()):
                match = re.match(r"^\s*\d+[.)]\s+(.+?)\s+-\s+", line)
                if match:
                    label = match.group(1).strip()
                    names[normalize(label)].append({"event": str(row["event_id"]), "label": label,
                        "source_field": field, "line": index, "quote": line,
                        "source_field_sha256": digest(str(row[field]))})
    repeated = []
    for label, mentions in sorted(names.items()):
        events = sorted({mention["event"] for mention in mentions})
        if len(events) > 1:
            repeated.append({"candidate_id": "entity-" + digest(label), "normalized_label": label,
                             "events": events, "mentions": mentions, "review_status": "unreviewed",
                             "disposition": "unreviewed_name_only_overlap"})
    by_assertion = collections.defaultdict(list)
    for fact in assertions:
        by_assertion[normalize(fact["statement"])].append({"event": fact["event"],
                                                          "opaque_probe_id": opaque_probe_id(fact["id"])})
    exact_links = [{"candidate_id": "assertion-" + digest(statement), "source_assertion": statement,
                    "events": sorted({member["event"] for member in members}), "members": members,
                    "review_status": "unreviewed"}
                   for statement, members in sorted(by_assertion.items())
                   if len({member["event"] for member in members}) > 1]
    answers = collections.defaultdict(list)
    for fact in assertions:
        answers[normalize(fact["answer"])].append({"event": fact["event"],
            "source_answer_label": fact["answer"], "source_assertion": fact["statement"],
            "opaque_probe_id": opaque_probe_id(fact["id"])})
    answer_links = []
    for answer, mentions in sorted(answers.items()):
        events = sorted({mention["event"] for mention in mentions})
        if len(events) > 1:
            answer_links.append({"candidate_id": "answer-" + digest(answer), "normalized_answer_label": answer,
                                 "events": events, "mentions": mentions,
                                 "generic_numeric_or_short_label": bool(re.fullmatch(r"[\d\W]+", answer)) or len(answer.split()) < 2,
                                 "review_status": "unreviewed", "disposition": "unreviewed_answer_only_overlap",
                                 "requires_event_relation_evidence": True})
    return sealed({"schema": "spacing-source-entity-candidates-v1", "dataset_revision": REVISION,
                   "scope": "all 100 source events; exact labels and exact source assertions only",
                   "inputs": ["source_assertions", "source_answer_labels", "pinned_factsheet_entities_and_locations"],
                   "semantic_audit_complete": False,
                   "interpretation": "Candidate generation only. Shared names do not establish shared facts or leakage.",
                   "label_inventory": [{"normalized_label": label, "mentions": mentions}
                                       for label, mentions in sorted(names.items())],
                   "repeated_entity_candidates": repeated, "exact_assertion_candidates": exact_links,
                   "answer_label_overlap_candidates": answer_links})


def build_catalog(source_directory, output, *, chunk_unit_limit=256, chunk_event_limit=10,
                  scope="confirmation"):
    import pyarrow.parquet as pq

    require(isinstance(chunk_unit_limit, int) and chunk_unit_limit > 0, "Invalid author chunk capacity")
    require(isinstance(chunk_event_limit, int) and chunk_event_limit > 0, "Invalid event chunk capacity")
    require(scope in ("development", "confirmation"), "Unknown source catalog partition")
    source_directory, output = Path(source_directory), Path(output)
    require(not output.exists(), "Catalog output already exists; never overwrite an author packet")
    qa_path, factsheet_path = source_directory / "fict_qa.parquet", source_directory / "fictsheets.parquet"
    checksums = {"fict_qa": hashlib.sha256(qa_path.read_bytes()).hexdigest(),
                 "fictsheets": hashlib.sha256(factsheet_path.read_bytes()).hexdigest()}
    require(checksums == {"fict_qa": QA_SHA256, "fictsheets": FACTSHEETS_SHA256},
            "Source catalog input differs from pinned parquet bytes")
    rows, factsheet_rows = pq.read_table(qa_path).to_pylist(), pq.read_table(factsheet_path).to_pylist()
    facts, audit = canonicalize(rows)
    partition = outer_partition(facts, audit["cross_event_links"], OUTER_SEED)
    require(len(rows) == 7500 and len(facts) == 1797 and len(factsheet_rows) == 100,
            "Pinned source coverage differs")
    factsheets = {str(row["event_id"]): str(row["fictsheet"]) for row in factsheet_rows}
    require(len(factsheets) == 100 and set(factsheets) == {fact["event"] for fact in facts},
            "Missing or duplicated source factsheet events")
    selected = [dict(fact, role="catalog") for fact in facts if fact["event"] in partition[scope]]
    registry = build_units(selected)
    by_probe = {fact["id"]: fact for fact in selected}
    units = []
    private_probe_map = {}
    for unit in registry["units"]:
        probes = [by_probe[probe_id] for probe_id in unit["member_probe_ids"]]
        canonical_probes = []
        for probe in probes:
            opaque_id = opaque_probe_id(probe["id"])
            canonical_probes.append({"opaque_probe_id": opaque_id, "source_answer_label": probe["answer"],
                                    "source_style": probe["style"],
                                    "opaque_duplicate_cluster_members": [opaque_probe_id(member)
                                                                          for member in probe["members"]]})
            private_probe_map[opaque_id] = {"original_probe_id": probe["id"], "source_unit_id": unit["id"],
                                           "event": unit["event"]}
        answers = sorted({probe["answer"] for probe in probes})
        payload = {"unit_id": unit["id"], "event": unit["event"],
                   "source_assertion": unit["source_statement"], "source_signature": unit["source_signature"],
                   "canonical_answer_labels": answers, "canonical_probes": canonical_probes,
                   "factsheet_sha256": digest(factsheets[unit["event"]]),
                   "evidence_candidates_only": True,
                   "evidence_candidates": evidence_candidates(unit["source_statement"], answers,
                                                              factsheets[unit["event"]]),
                   "author_status": "not_authored", "review_status": "unreviewed"}
        units.append(sealed(payload))
    units.sort(key=lambda unit: (unit["event"], unit["unit_id"]))
    source_header = {"dataset_revision": REVISION, "pinned_parquet_sha256": checksums,
                     "outer_seed": OUTER_SEED, "outer_partition_sha256": partition["sha256"],
                     "scope": "80 held-out source events, before semantic grounding and grouping",
                     "role_independent": True, "semantic_audit_complete": False,
                     "human_review_complete": False,
                     "authoring_inputs": ["source_assertions", "source_answer_labels", "event_factsheets"],
                     "excluded_inputs": ["canonical_evaluation_wording", "paraphrase_evaluation_wording",
                                         "MCQ_choices", "model_outputs", "linked_fictions"]}
    if scope == "development":
        source_header.update(catalog_scope="development",
                             scope="20 development source events, before semantic grounding and grouping")
    by_event = collections.defaultdict(list)
    for unit in units:
        by_event[unit["event"]].append(unit)
    event_stats = [{"event": event, "source_units": len(by_event[event]),
                    "canonical_probes": sum(len(unit["canonical_probes"]) for unit in by_event[event]),
                    "source_answer_targets": sum(len({normalize(a) for a in unit["canonical_answer_labels"]})
                                                 for unit in by_event[event]),
                    "source_qa_rows": sum(str(row["event_id"]) == event for row in rows)}
                   for event in sorted(by_event)]
    catalog = sealed({"schema": f"spacing-{scope}-source-catalog-v1", **source_header,
                      "partition": partition, "event_statistics": event_stats, "units": units,
                      "factsheets": {event: {"text": factsheets[event], "sha256": digest(factsheets[event])}
                                     for event in sorted(by_event)},
                      "coverage": {"events": len(by_event), "canonical_probes": len(selected),
                                   "source_units": len(units),
                                   "source_answer_targets": sum(row["source_answer_targets"] for row in event_stats)}})
    expected_coverage = {"confirmation": {"events": 80, "canonical_probes": 1449,
                                          "source_units": 1258, "source_answer_targets": 1327},
                         "development": {"events": 20, "canonical_probes": 348,
                                         "source_units": 301, "source_answer_targets": 314}}
    require(catalog["coverage"] == expected_coverage[scope], "Source partition inventory differs")
    chunks, current, current_count = [], [], 0
    for event in sorted(by_event):
        require(len(by_event[event]) <= chunk_unit_limit, "One event exceeds chunk unit capacity")
        if current and (current_count + len(by_event[event]) > chunk_unit_limit or len(current) >= chunk_event_limit):
            chunks.append(current)
            current, current_count = [], 0
        current.append(event)
        current_count += len(by_event[event])
    if current:
        chunks.append(current)
    author_packets = []
    for index, events in enumerate(chunks, 1):
        packet = sealed({"schema": "source-only-confirmation-authoring-input-v1", **source_header,
                         "source_catalog_sha256": catalog["sha256"], "chunk_index": index,
                         "events": events, "units": [unit for event in events for unit in by_event[event]],
                         "factsheets": {event: catalog["factsheets"][event] for event in events}})
        author_packets.append(packet)
    geometry = []
    for seed in SPLIT_SEEDS:
        split = assign_roles(facts, partition, scope, seed)
        split_registry = build_units(split["facts"])
        old_units = [unit for unit in split_registry["units"] if unit["role"] == "old"]
        targets = sum(len(unit["answer_labels"]) for unit in old_units)
        stagger = (len(old_units) - 1) // 2
        geometry.append({"split_seed": seed, "roles": split["roles"],
                         "style_imbalance": split["style_imbalance"],
                         "canonical_probe_counts": dict(collections.Counter(fact["role"] for fact in split["facts"])),
                         "source_unit_counts": split_registry["unit_counts"],
                         "old_source_answer_targets": targets, "cohort_size": 2, "review_cap": 8,
                         "max_stagger_span_ratio": .75, "stagger_width": stagger,
                         "minimum_span_multiple_21": 21 * math.ceil(stagger / (.75 * 21)),
                         "mixed_stage1_updates_per_cycle_batch16_qa4": math.ceil((len(old_units) + targets) / 12)})
    geometry = sealed({"schema": f"spacing-source-only-{scope}-geometry-v1",
                       "source_catalog_sha256": catalog["sha256"], "seeds": list(SPLIT_SEEDS),
                       "selection_status": "metadata-only planning candidates, not preregistered replicates",
                       "unit_basis": "exact source units, before reviewed semantic grouping",
                       "replicates": geometry,
                       "first32_common_feasible_span": max(row["minimum_span_multiple_21"] for row in geometry)})
    scale_plan = None
    if scope == "development":
        original_roles = assign_roles(facts, partition, "development", 999)["roles"]
        qa_events = sorted(event for event, role in original_roles.items() if role == "qa")
        stats_by_event = {row["event"]: row for row in event_stats}
        eligible = sorted((event for event in by_event if event not in qa_events),
                          key=lambda event: (stats_by_event[event]["source_units"], event))
        roles = {event: "qa" if event in qa_events else "old" for event in by_event}
        roles[eligible[0]], roles[eligible[1]] = "new", "control"
        unit_counts = collections.Counter()
        probe_counts = collections.Counter()
        target_counts = collections.Counter()
        for row in event_stats:
            role = roles[row["event"]]
            unit_counts[role] += row["source_units"]
            probe_counts[role] += row["canonical_probes"]
            target_counts[role] += row["source_answer_targets"]
        old_count = unit_counts["old"]
        scale_plan = sealed({"schema": "spacing-development-scale-source-role-plan-v1",
            "source_catalog_sha256": catalog["sha256"], "outer_partition_sha256": partition["sha256"],
            "mode": "development", "selection_status": "metadata-only proposal; root must freeze before probe outcomes",
            "selection_rule": "Keep QA events from development split999; choose the two smallest remaining source-unit counts, breaking ties by event ID, as new then control; all other events old",
            "base_split_seed": 999, "qa_events": qa_events, "placeholder_new_event": eligible[0],
            "placeholder_control_event": eligible[1], "roles": roles,
            "event_role_counts": dict(collections.Counter(roles.values())),
            "source_unit_counts": dict(unit_counts), "canonical_probe_counts": dict(probe_counts),
            "source_answer_target_counts": dict(target_counts),
            "stagger_width_q2": (old_count - 1) // 2,
            "minimum_span_multiple_21_q2_cap8_ratio075": 21 * math.ceil(((old_count - 1) // 2) / (.75 * 21)),
            "mixed_stage1_updates_per_cycle_batch16_qa4": math.ceil((old_count + target_counts["old"]) / 12),
            "semantic_audit_complete": False})
    entities = entity_inventory(factsheet_rows, facts)
    all_facts = [dict(fact, role="cross_entity_catalog") for fact in facts]
    all_registry = build_units(all_facts)
    all_by_probe = {fact["id"]: fact for fact in all_facts}
    all_by_event = collections.defaultdict(list)
    for unit in all_registry["units"]:
        all_by_event[unit["event"]].append({"unit_id": unit["id"], "source_assertion": unit["source_statement"],
            "canonical_answer_labels": sorted({all_by_probe[pid]["answer"] for pid in unit["member_probe_ids"]}),
            "opaque_probe_ids": [opaque_probe_id(pid) for pid in unit["member_probe_ids"]]})
    factsheet_by_event = {str(row["event_id"]): row for row in factsheet_rows}
    cross_entity_packet = sealed({"schema": "source-only-cross-entity-authoring-input-v1", **source_header,
        "scope": "all 100 source events, including the 20 development events", "partition": partition,
        "source_catalog_sha256": catalog["sha256"], "entity_candidates_sha256": entities["sha256"],
        "events": [{"event": event,
                    "outer_partition": "development" if event in partition["development"] else "confirmation",
                    "factsheet": {"text": factsheets[event], "sha256": digest(factsheets[event])},
                    "factsheet_metadata": {key: str(factsheet_by_event[event][key])
                                            for key in ("entities", "events", "locations", "times", "reasons")},
                    "source_assertions": all_by_event[event]} for event in sorted(all_by_event)],
        "coverage": {"events": len(all_by_event), "source_units": len(all_registry["units"]),
                     "canonical_probes": len(all_facts)},
        "audit_status": "unreviewed; candidate overlaps require contextual independent dispositions"})
    ledger = sealed({"schema": "spacing-unreviewed-source-coverage-ledger-v1",
                     "source_catalog_sha256": catalog["sha256"], "independent_agent_review_complete": False,
                     "human_review_complete": False,
                     "units": [{"unit_id": unit["unit_id"], "event": unit["event"],
                                "source_fields_sha256": unit["sha256"], "author": None, "reviewer": None,
                                "review_status": "unreviewed", "objections": []} for unit in units],
                     "events": [{"event": event, "entity_audit_status": "unreviewed",
                                 "source_conflict_audit_status": "unreviewed"} for event in sorted(by_event)]})
    forbidden_questions = {" ".join(str(row["question"]).casefold().split())
                           for row in rows if str(row["question"]).strip()}
    for payload in [catalog, entities, cross_entity_packet, ledger, *author_packets]:
        reject_question_text(payload, forbidden_questions)
    index = sealed({"schema": "spacing-source-authoring-chunk-index-v1",
                    "source_catalog_sha256": catalog["sha256"], "chunk_unit_limit": chunk_unit_limit,
                    "chunk_event_limit": chunk_event_limit,
                    "chunks": [{"path": f"author-packets/chunk-{i:02}.json", "sha256": packet["sha256"],
                                "events": packet["events"], "source_units": len(packet["units"]),
                                "canonical_probes": sum(len(unit["canonical_probes"]) for unit in packet["units"])}
                               for i, packet in enumerate(author_packets, 1)]})
    output.mkdir(parents=True)
    for name, payload in (("source-catalog.json", catalog), ("author-chunk-index.json", index),
                          ("role-geometry-first32.json", geometry), ("entity-candidates.json", entities),
                          ("cross-entity-source-packet-all100.json", cross_entity_packet),
                          ("review-coverage-ledger.json", ledger),
                          ("private-probe-id-map.json", sealed({"source_catalog_sha256": catalog["sha256"],
                                                               "probes": private_probe_map}))):
        write_json(output / name, payload)
    for i, packet in enumerate(author_packets, 1):
        write_json(output / f"author-packets/chunk-{i:02}.json", packet)
    if scale_plan is not None:
        write_json(output / "development-scale-role-plan.json", scale_plan)
    result = {"output": str(output.resolve()), "catalog_sha256": catalog["sha256"],
            "coverage": catalog["coverage"], "author_chunks": len(author_packets),
            "first32_common_feasible_span": geometry["first32_common_feasible_span"],
            "repeated_entity_candidates": len(entities["repeated_entity_candidates"]),
            "answer_label_overlap_candidates": len(entities["answer_label_overlap_candidates"]),
            "semantic_audit_complete": False, "source_question_forms_checked": len(forbidden_questions)}
    if scale_plan is not None:
        result["development_scale_source_counts"] = scale_plan["source_unit_counts"]
        result["development_scale_source_answer_targets"] = scale_plan["source_answer_target_counts"]
        result["development_scale_qa_events"] = qa_events
        result["development_scale_placeholder_events"] = [eligible[0], eligible[1]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-unit-limit", type=int, default=256)
    parser.add_argument("--chunk-event-limit", type=int, default=10)
    parser.add_argument("--partition", "--scope", dest="scope", choices=("development", "confirmation"),
                        default="confirmation")
    args = parser.parse_args()
    import json
    print(json.dumps(build_catalog(args.source_directory, args.output, chunk_unit_limit=args.chunk_unit_limit,
                                   chunk_event_limit=args.chunk_event_limit, scope=args.scope),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
