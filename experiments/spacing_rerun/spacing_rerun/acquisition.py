"""Development-only source QA for shared acquisition, never for later review."""
from __future__ import annotations

import collections
import re

from .common import digest, require
from .data import REVISION, normalize

DECLARATION_POLICY = "declaration_only_v1"
SOURCE_QA_ACQUISITION_POLICY = "declaration_plus_source_qa_v1"
ADAPTIVE_TRAJECTORY_POLICY = "adaptive_first_usable_v1"
GRID_TRAJECTORY_POLICY = "fixed_grid_no_early_stop_v1"


def acquisition_identity(unit_id, answer):
    return "acq-" + digest([unit_id, normalize(answer)])[:24]


def acquisition_key(record, variant):
    return f"acq/{record['id']}/{variant}"


def acquisition_similarity(facts, records):
    """Frozen lexical overlap diagnostic; never select or filter training probes."""
    threshold = 0.7
    by_unit = collections.defaultdict(list)
    per_record = {r["id"]: {"id": r["id"], "unit_id": r["unit_id"], "max_jaccard": 0.0,
                            "max_jaccard_by_variant": [0.0] * len(r["questions"])} for r in records}
    for record in records:
        for variant, question in enumerate(record["questions"]):
            by_unit[record["unit_id"]].append((record["id"], variant, set(re.findall(r"\w+", question.casefold()))))
    probes = []
    for fact in facts:
        if fact["role"] != "old":
            continue
        require(by_unit[fact["unit_id"]], "No source QA questions for similarity diagnostic unit")
        for field, variant in (("question", "canonical"), ("paraphrase", "paraphrase")):
            if not fact.get(field):
                continue
            words = set(re.findall(r"\w+", fact[field].casefold()))
            scores = []
            for rid, qvariant, source_words in by_unit[fact["unit_id"]]:
                union = words | source_words
                score = len(words & source_words) / len(union) if union else 0.0
                scores.append((score, rid, qvariant))
                per_record[rid]["max_jaccard"] = max(per_record[rid]["max_jaccard"], score)
                per_record[rid]["max_jaccard_by_variant"][qvariant] = max(
                    per_record[rid]["max_jaccard_by_variant"][qvariant], score)
            score, rid, qvariant = max(scores)
            probes.append({"id": fact["id"], "unit_id": fact["unit_id"], "event": fact["event"],
                           "variant": variant, "max_jaccard": score, "nearest_record_id": rid,
                           "nearest_question_variant": qvariant, "at_or_above_threshold": score >= threshold})
    report = {"schema": "spacing-acquisition-lexical-overlap-v1", "threshold": threshold,
              "tokenization": "casefold then Unicode regex word tokens; set Jaccard",
              "scope": "evaluation versus source QA questions within the same old rehearsal unit",
              "selection_policy": "diagnostic_only_no_training_or_primary_evaluation_filter",
              "probes": probes, "records": [per_record[k] for k in sorted(per_record)]}
    report["sha256"] = digest(report)
    return report


def validate_acquisition_bundle(bundle, source_input, facts, registry, grounding_bundle, *, mode, confirmation_audit=None):
    """Bind reviewed questions to source targets and reject held-out probe forms."""
    confirmed = mode == "confirmation" and confirmation_audit is not None
    require((mode == "development" or confirmed) and bundle.get("mode") == mode,
            "Source QA acquisition is development-only until separately audited and frozen")
    if confirmed:
        from .confirmation import validate_confirmation_acquisition_bundle
        validate_confirmation_acquisition_bundle(bundle, source_input, confirmation_audit, {fact["event"]: fact["role"] for fact in facts})
    review_status = "approved_confirmation" if confirmed else "approved_development"
    require(bundle.get("schema") == "spacing-acquisition-qa-v1", "Unknown acquisition QA schema")
    require(digest({k: v for k, v in bundle.items() if k != "sha256"}) == bundle.get("sha256"),
            "Acquisition QA bundle was modified")
    require(source_input.get("schema") == "source-only-acquisition-authoring-input-v1" and
            source_input.get("dataset_revision") == REVISION, "Acquisition source input revision/schema differs")
    require(digest({k: v for k, v in source_input.items() if k != "sha256"}) == source_input.get("sha256"),
            "Acquisition source input was modified")
    require(bundle.get("source_input_sha256") == source_input["sha256"], "Acquisition source input hash differs")
    require(bundle.get("grounding_bundle_sha256") == source_input.get("grounding_bundle_sha256") ==
            grounding_bundle["sha256"], "Acquisition grounding provenance differs")
    require(bundle.get("authoring_inputs") == ["source_assertions", "source_answer_labels", "event_factsheets"] and
            bundle.get("evaluation_question_text_included") is False and bundle.get("model_outcomes_used") is False,
            "Acquisition authoring inputs included evaluation questions or model outcomes")
    require(bundle.get("independent_agent_review_complete") is True and bundle.get("independent_agent_reviewer"),
            "Acquisition QA requires completed independent review")
    old_units = {u["id"]: u for u in registry["units"] if u["role"] == "old"}
    source_units = source_input.get("units", [])
    require(len(source_units) == len(old_units) and {u["unit_id"] for u in source_units} == set(old_units),
            "Acquisition authoring unit membership differs")
    source_by_id = {u["unit_id"]: u for u in source_units}
    old_sources = {r["unit_id"]: r for r in grounding_bundle["source_units"] if r["role"] == "old"}
    for source in source_units:
        unit = old_units[source["unit_id"]]
        require(source["event"] == unit["event"] and source["statement"] == unit["statement"] and
                {normalize(a) for a in source["answer_targets"]} == set(unit["answer_labels"]),
                "Acquisition source unit event/statement/answers differ")
        assertions = source.get("source_assertions", [])
        expected = [old_sources[sid] for sid in unit["source_unit_ids"]]
        expected_payload = [{"statement": r["source_statement"], "answer_targets": sorted(set(r["canonical_answers"])),
                             "status": r["status"], "evidence": r["evidence"]} for r in expected]
        require(collections.Counter(digest(r) for r in assertions) ==
                collections.Counter(digest(r) for r in expected_payload),
                "Acquisition source assertions/evidence differ from grounded source ledger")
    events = {u["event"] for u in old_units.values()}
    require(set(source_input.get("factsheets", {})) == events and all(
            source_input["factsheets"][event] == grounding_bundle["factsheets"][event] for event in events),
            "Acquisition factsheets differ from pinned grounding evidence")
    heldout = {normalize(f[k]) for f in facts if f["role"] in ("old", "new", "control")
               for k in ("question", "paraphrase") if f.get(k)}
    targets = {(unit["id"], answer) for unit in old_units.values() for answer in unit["answer_labels"]}
    records, seen, seen_questions = bundle.get("records", []), [], set()
    require(records, "Empty acquisition QA bundle")
    for record in records:
        unit_id, answer = record.get("unit_id"), normalize(record.get("answer", ""))
        require((unit_id, answer) in targets and record.get("event") == old_units[unit_id]["event"],
                "Acquisition QA target crossed source-unit membership/answer/event roles")
        require(record["id"] == acquisition_identity(unit_id, answer), "Unstable acquisition target identity")
        require(record.get("review_status") == review_status and record.get("rationale"),
                "Unreviewed acquisition QA target")
        questions = record.get("questions", [])
        require(len(questions) == 2 and len({normalize(q) for q in questions}) == 2,
                "Acquisition targets require two distinct source-derived questions")
        for question in questions:
            require(isinstance(question, str) and question.strip() and "\n" not in question and
                    "Question:" not in question and "Answer:" not in question,
                    "Acquisition question contains context or QA markers")
            normalized = normalize(question)
            require(normalized not in heldout, "Held-out evaluation question entered acquisition QA")
            declarations = [old_units[unit_id]["statement"]] + [
                s["statement"] for s in source_by_id[unit_id]["source_assertions"]]
            require(not any(normalize(statement) in normalized for statement in declarations),
                    "Acquisition question embeds its declaration as context")
            require(normalized not in seen_questions, "Duplicate acquisition question across targets")
            seen_questions.add(normalized)
        evidence = record.get("evidence", [])
        require(evidence, "Missing acquisition source evidence")
        lines = source_input["factsheets"][record["event"]].splitlines()
        require(all(isinstance(e.get("line"), int) and 0 <= e["line"] < len(lines) and
                    e.get("quote") == lines[e["line"]] for e in evidence),
                "Acquisition evidence is not an exact pinned source excerpt")
        seen.append((unit_id, answer))
    require(collections.Counter(seen) == collections.Counter(targets), "Acquisition QA lost or duplicated answer targets")
