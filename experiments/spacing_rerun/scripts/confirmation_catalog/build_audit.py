"""Convert the reviewed catalog into the confirmation audit wrapper and validate it.

Run on FarmShare:
    python -I -B build_audit.py CATALOG_ROOT SOURCE_DIR A_DIR OUTPUT
A_DIR is the final Channel A event directory, relative to CATALOG_ROOT (e.g. authored/A-final5).
SOURCE_DIR holds the pinned fict_qa.parquet and fictsheets.parquet.

The conversions follow AUDIT-CONFORMANCE-AMENDMENT.md part 2 and are counted in the wrapper.
Receipts are rebuilt from the per-event decision files (part 1, rule 4). Nothing here makes a
semantic choice: a record without a final APPROVE that postdates its last revision stops the build.
"""

import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.acquisition import acquisition_identity  # noqa: E402
from spacing_rerun.common import digest  # noqa: E402
from spacing_rerun.confirmation import (  # noqa: E402
    AUDIT_CONFORMANCE_AMENDMENT_SHA256, AUDIT_SCHEMA, NARROW_EVALUATION_INPUTS, PROTOCOL,
    evaluation_review_content, source_conflict_status, source_preservation_context, source_review_content,
    validate_confirmation_audit)
from spacing_rerun.data import REVISION, canonicalize, normalize  # noqa: E402
from spacing_rerun.grounding import group_identity  # noqa: E402

root, source_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
a_dir, output = root / sys.argv[3], pathlib.Path(sys.argv[4])
if output.exists():
    sys.exit("output exists; never overwrite")
counts = collections.Counter()
STATUS = {"grounded": "grounded", "assertion_only": "preserve_source_ambiguity",
          "conflict_flagged": "preserve_source_conflict"}
AMBIGUITY = {"grounded": "supported", "preserve_source_ambiguity": "retained_source_ambiguity",
             "preserve_source_conflict": "retained_source_conflict"}
ANTHROPIC = {"provider": "anthropic", "harness": "claude-code-agent-tool"}
MODEL_BASIS = ("Claude Code Agent tool model alias; the run log records the alias, not a dated model id")
OWNER = {"name": "study-owner", "harness": "owner-decision-record", "model": "none (human study owner)",
         "human": True, "basis": ("FINAL-PASS-AMENDMENT.md rule 5 and AUDIT-CONFORMANCE-AMENDMENT.md part 3 rule 4; "
                                  "reviews/owner-decisions*.json")}


def sealed(payload):
    return dict(payload, sha256=digest(payload))


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.load(open(path))


import pyarrow.parquet as pq  # noqa: E402

qa_rows = pq.read_table(source_dir / "fict_qa.parquet").to_pylist()
facts, _ = canonicalize(qa_rows)
fact_by_id = {f["id"]: f for f in facts}
pinned_factsheets = {str(r["event_id"]): str(r["fictsheet"])
                     for r in pq.read_table(source_dir / "fictsheets.parquet").to_pylist()}
catalog = load(root / "packets/source-catalog.json")
probe_map = load(root / "packets/private-probe-id-map.json")
raw = {u["unit_id"]: u for u in catalog["units"]}
lines = {e: f["text"].splitlines() for e, f in catalog["factsheets"].items()}
author_packets = [load(root / f"packets/author-packets/chunk-{n:02d}.json") for n in range(1, 9)]
paraphrase_packets = [load(root / f"paraphrase-packets/chunk-{n:02d}.json") for n in range(1, 9)]
# Later owner files (owner-decisions-r6.json) supersede the original one for the same record.
owner = {}
for owner_path in sorted((root / "reviews").glob("owner-decisions*.json"), key=lambda p: (p.stem != "owner-decisions", p.stem)):
    for d in load(owner_path)["decisions"]:
        owner[(d["channel"], d["id"])] = dict(d, path=owner_path)


def fix_evidence(entries, event):
    """1-indexed stripped quotes become 0-indexed exact raw lines."""
    stripped = [line.strip() for line in lines[event]]
    fixed = []
    for entry in entries:
        quote, line = entry.get("quote", "").strip(), entry.get("line")
        if isinstance(line, int) and 1 <= line <= len(stripped) and stripped[line - 1] == quote:
            index = line - 1
        else:
            matches = [i for i, text in enumerate(stripped) if text == quote]
            if len(matches) != 1:
                raise SystemExit(f"{event}: evidence quote is not one factsheet line: {entry}")
            index = matches[0]
            counts["evidence_renumbered_by_quote"] += 1
        counts["evidence_lines_converted"] += 1
        if lines[event][index] != entry.get("quote"):
            counts["evidence_quotes_restored_to_raw_line"] += 1
        row = {"line": index, "quote": lines[event][index]}
        if row not in fixed:
            fixed.append(row)
    return sorted(fixed, key=lambda row: row["line"])


def final_decisions(channel, event):
    """Latest decision per id across rounds, then owner decisions (FINAL-PASS-AMENDMENT.md rule 5)."""
    final = {}
    for path in sorted((root / "reviews" / channel).glob(f"{event}.r*.json"), key=lambda p: int(p.stem.split(".r")[1])):
        rnd = int(path.stem.split(".r")[1])
        doc = load(path)
        for decision in doc.get("decisions", []):
            final[decision["id"]] = {"decision": decision["decision"], "round": rnd, "path": path,
                                     "reviewer": doc.get("reviewer", "claude-sonnet")}
    for (ch, rid), decision in owner.items():
        if ch == channel and decision["event"] == event:
            final[rid] = {"decision": decision["decision"], "round": "owner", "path": decision["path"],
                          "edit": decision["edit"], "disclose": decision["disclose"]}
    return final


def approval(final, rid, record, what):
    decision = final.get(rid)
    revised = record.get("revision_round", 1)
    if decision is None or decision["decision"] != "APPROVE":
        raise SystemExit(f"{what} {rid}: no final APPROVE")
    if decision["round"] != "owner" and decision["round"] < revised:
        raise SystemExit(f"{what} {rid}: approved in round {decision['round']} but revised in round {revised}")
    return decision


def reviewer_identity(channel, event, decision):
    if decision["round"] == "owner":
        return OWNER
    stage = {4: "final-pass-fresh-reviewer", 5: "repair-reviewer", 6: "repair-revision-reviewer"}.get(
        decision["round"], "reviewer")
    return {"name": f"claude-sonnet/confirmation-{channel}/{event}/round-{decision['round']}", **ANTHROPIC,
            "model": "claude-sonnet", "model_basis": MODEL_BASIS, "stage": stage}


def receipts_for(channel, schema, packet, bindings, approvals):
    """One rebuilt receipt per reviewer agent: decisions grouped by (event, round)."""
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    for field, rows in approvals.items():
        for event, decision, record_id, content in rows:
            key = (event, str(decision["round"]))
            note = (f"Final APPROVE in {decision['path'].relative_to(root)}; receipt rebuilt from that decision file."
                    if decision["round"] != "owner" else
                    f"Study-owner decision: {decision['edit']}. Disclosure: {decision['disclose']}.")
            buckets[key][field].append({"id": record_id, "status": "approved", "reviewed_content_sha256": digest(content),
                                        "notes": note, "_decision": decision})
    receipts = []
    for (event, rnd), fields in sorted(buckets.items()):
        decision = next(iter(fields.values()))[0]["_decision"]
        body = {"schema": schema, "reviewer": reviewer_identity(channel, event, decision),
                "review_packet_sha256": packet["sha256"], **bindings, "event": event,
                "review_round": rnd, "receipts_reconstructed": True,
                "decision_file": {"path": str(decision["path"].relative_to(root)), "file_digest": file_sha(decision["path"])},
                "author_rationales_seen": True, "model_outcomes_seen": False, "human_review_complete": False,
                "visibility_basis": ("The reviewer read the authored event file, including rationales and notes "
                                     "(AUTHORING-SPEC.md review section)."),
                **({"acquisition_content_seen": False} if channel == "B" else {})}
        for field, rows in fields.items():
            body[field] = [{k: v for k, v in row.items() if k != "_decision"} for row in rows]
        receipts.append(sealed(body))
    return receipts


# ---------------------------------------------------------------- Channel A
source_chunks, all_qa_forms = [], set()
revision_authors = {"opus": {"name": "claude-opus/confirmation-A/final-pass", **ANTHROPIC, "model": "claude-opus",
                             "model_basis": MODEL_BASIS, "basis": "FINAL-PASS-AMENDMENT.md"}}
for packet in author_packets:
    n = packet["chunk_index"]
    author = {"name": f"claude-haiku/confirmation-A/chunk-{n:02d}", **ANTHROPIC, "model": "claude-haiku",
              "model_basis": MODEL_BASIS, "basis": "ELECTIONS-AMENDMENT.md review-process rule 1"}
    units_out, groups_out, qa_out, related, disclosed = [], [], [], [], collections.defaultdict(list)
    approvals = {"groups": [], "acquisition_records": []}
    for event in packet["events"]:
        doc = load(a_dir / f"{event}.json")
        final = final_decisions("A", event)
        local_units = {u["unit_id"]: u for u in doc["source_units"]}
        if set(local_units) != {u["unit_id"] for u in packet["units"] if u["event"] == event}:
            raise SystemExit(f"{event}: Channel A units differ from the author packet")
        group_id = {g["id"]: group_identity(sorted(g["source_unit_ids"])) for g in doc["groups"]}
        status = {uid: STATUS[u["status"]] for uid, u in local_units.items()}
        unit_evidence = {uid: fix_evidence(u["evidence"], event) for uid, u in local_units.items()}
        groups_local = {g["id"]: g for g in doc["groups"]}
        for g in doc["groups"]:
            members = sorted(g["source_unit_ids"])
            statement = g["statement"]
            if any(status[s] != "grounded" for s in members) and len(members) == 1:
                assertion = raw[members[0]]["source_assertion"]
                if statement != assertion and normalize(statement) == normalize(assertion):
                    statement = assertion
                    counts["preserved_statement_punctuation_restored"] += 1
            evidence = []
            for s in members:
                evidence += [row for row in unit_evidence[s] if row not in evidence]
            group = {"id": group_id[g["id"]], "event": event, "source_unit_ids": members, "statement": statement,
                     "evidence": sorted(evidence, key=lambda row: row["line"]), "review_status": "unreviewed",
                     "grouping_rationale": g["grouping_rationale"]}
            groups_local[g["id"]] = group
            groups_out.append(group)
            approvals["groups"].append((event, approval(final, g["id"], g, "group"), group["id"],
                                        source_review_content(group, "groups")))
            if g.get("revision_round") == 4:
                disclosed["final_pass_opus_revisions"].append(group["id"])
        for uid, u in local_units.items():
            original = raw[uid]
            group = groups_local[u["group_id"]]
            nonliteral = [a for a in sorted(set(original["canonical_answer_labels"]))
                          if normalize(a) not in normalize(group["statement"])]
            if nonliteral != u["nonliteral_answer_labels"]:
                counts["nonliteral_flags_recomputed"] += 1
            approval(final, uid, u, "source unit")
            units_out.append({"unit_id": uid, "group_id": group["id"], "event": event,
                              "source_assertion": original["source_assertion"], "source_unit_sha256": original["sha256"],
                              "opaque_probe_ids": sorted(p["opaque_probe_id"] for p in original["canonical_probes"]),
                              "review_status": "unreviewed", "evidence": unit_evidence[uid],
                              "nonliteral_answer_labels": nonliteral, "status": status[uid],
                              "conflict_rationale": u.get("note") if status[uid] != "grounded" else None,
                              **({"author_note": u["note"]} if status[uid] == "grounded" and u.get("note") else {})})
        targets = set()
        for r in doc["source_qa"]:
            group = groups_local[r["unit_id"]]
            key = (group["id"], normalize(r["answer"]))
            if key in targets:
                counts["duplicate_target_record_dropped"] += 1
                disclosed["duplicate_target_records_dropped"].append(r["id"])
                continue
            targets.add(key)
            record = {"id": acquisition_identity(group["id"], r["answer"]), "unit_id": group["id"], "event": event,
                      "answer": r["answer"], "questions": r["questions"], "review_status": "unreviewed",
                      "rationale": r["rationale"], "evidence": fix_evidence(r["evidence"], event),
                      "source_conflict_status": source_conflict_status({status[s] for s in group["source_unit_ids"]})}
            qa_out.append(record)
            all_qa_forms.update(normalize(q) for q in r["questions"])
            decision = approval(final, r["id"], r, "source QA")
            approvals["acquisition_records"].append((event, decision, record["id"],
                                                     source_review_content(record, "acquisition_records")))
            if r.get("revision_round") == 4 or r.get("final_pass_author"):
                disclosed["final_pass_opus_revisions"].append(record["id"])
            if decision["round"] == "owner":
                disclosed["owner_decided"].append(record["id"])
        related += [{"event": event, **row} for row in doc.get("related_assertions_not_merged", [])]
    authored = sealed({"schema": "spacing-confirmation-authored-source-chunk-v1",
                       "source_catalog_sha256": catalog["sha256"], "source_author_packet_sha256": packet["sha256"],
                       "events": packet["events"], "chunk_index": n, "author": author,
                       "authoring_inputs": ["source_assertions", "source_answer_labels", "event_factsheets"],
                       "dataset_revision": REVISION, "evaluation_question_text_included": False,
                       "model_outcomes_used": False, "human_review_complete": False,
                       "independent_agent_review_complete": False,
                       "revision_authors_disclosed": [{"identity": revision_authors["opus"],
                                                       "record_ids": sorted(set(disclosed["final_pass_opus_revisions"]))}],
                       "owner_decided_record_ids": disclosed["owner_decided"],
                       "duplicate_target_records_dropped": disclosed["duplicate_target_records_dropped"],
                       "related_assertions_not_merged": related,
                       "source_units": units_out, "groups": groups_out, "acquisition_records": qa_out})
    review_packet = sealed({"schema": "spacing-source-independent-review-packet-v1",
                            "source_catalog_sha256": catalog["sha256"], "source_author_packet_sha256": packet["sha256"],
                            "authored_chunk_sha256": authored["sha256"], "events": packet["events"], "chunk_index": n,
                            "factsheets": packet["factsheets"], "author_rationales_included": False,
                            "reconstructed": True,
                            "reconstruction_note": ("Projection of the approved content in the validator's packet shape. "
                                                    "Reviewers worked from per-event authored files instead."),
                            "source_units": [{k: u[k] for k in ("unit_id", "event", "source_assertion",
                                                                 "canonical_answer_labels", "canonical_probes", "sha256")}
                                             for u in packet["units"]],
                            "source_preservation_protocol": source_preservation_context(authored),
                            "groups": [source_review_content(g, "groups") for g in groups_out],
                            "acquisition_records": [source_review_content(r, "acquisition_records") for r in qa_out]})
    bindings = {"source_catalog_sha256": catalog["sha256"], "source_author_packet_sha256": packet["sha256"],
                "authored_chunk_sha256": authored["sha256"], "chunk_index": n}
    reviews = receipts_for("A", "spacing-source-semantic-review-v1", review_packet, bindings, approvals)
    source_chunks.append({"author_packet": packet, "authored": authored, "review_packet": review_packet,
                          "reviews": reviews})
    counts["source_receipts"] += len(reviews)

# ---------------------------------------------------------------- Channel B
evaluation_chunks = []
units_by_opaque = {p["opaque_probe_id"]: u for chunk in source_chunks for u in chunk["authored"]["source_units"]
                   for p in [{"opaque_probe_id": o} for o in u["opaque_probe_ids"]]}
for packet, actual in zip(author_packets, paraphrase_packets):
    n = packet["chunk_index"]
    if actual["events"] != packet["events"]:
        raise SystemExit(f"chunk {n}: paraphrase packet events differ")
    author = {"name": f"claude-haiku/confirmation-B/chunk-{n:02d}", **ANTHROPIC, "model": "claude-haiku",
              "model_basis": MODEL_BASIS, "basis": "ELECTIONS-AMENDMENT.md review-process rule 1"}
    packet_rows, records, approvals = [], [], {"records": []}
    actual_probes = {p["opaque_probe_id"]: p for p in actual["probes"]}
    for event in packet["events"]:
        doc = load(root / f"authored/B/{event}.json")
        final = final_decisions("B", event)
        for p in doc["paraphrases"]:
            opaque = p["opaque_probe_id"]
            mapped = probe_map["probes"][opaque]
            fact = fact_by_id[mapped["original_probe_id"]]
            unit = units_by_opaque[opaque]
            if p["canonical_question"] != fact["question"] or actual_probes[opaque]["canonical_question"] != fact["question"]:
                raise SystemExit(f"{opaque}: canonical question differs from the pinned source")
            base = {"id": fact["id"], "opaque_probe_id": opaque, "event": event, "source_unit_id": mapped["source_unit_id"],
                    "canonical_question": fact["question"], "canonical_answer": fact["answer"]}
            packet_rows.append(base)
            ambiguity = AMBIGUITY[unit["status"]]
            record = dict(base, original_source_assertion=fact.get("source_statement", fact["statement"]), aliases=[],
                          paraphrase=p["paraphrase"], evidence=unit["evidence"] if ambiguity == "supported" or unit["evidence"] else [],
                          source_ambiguity_status=ambiguity, review_status="unreviewed")
            if normalize(record["paraphrase"]) in all_qa_forms or normalize(record["canonical_question"]) in all_qa_forms:
                raise SystemExit(f"{opaque}: evaluation wording equals an acquisition form")
            if ambiguity == "supported" and not record["evidence"]:
                raise SystemExit(f"{opaque}: supported evaluation record lacks evidence")
            records.append(record)
            counts["evaluation_evidence_attached_from_source_unit"] += bool(record["evidence"])
            decision = approval(final, opaque, p, "paraphrase")
            approvals["records"].append((event, decision, fact["id"], evaluation_review_content(record)))
    author_packet = sealed({"schema": "evaluation-only-confirmation-authoring-input-v1",
                            "source_catalog_sha256": catalog["sha256"], "chunk_index": n, "events": packet["events"],
                            "actual_author_packet_sha256": actual["sha256"], "reconstructed": True,
                            "reconstruction_note": ("Channel B authors received the actual packet (opaque ids, canonical "
                                                    "questions and answers). This projection adds original probe ids and "
                                                    "source unit ids for validation."),
                            "records": packet_rows})
    authored = sealed({"schema": "spacing-confirmation-evaluation-catalog-v1", "source_catalog_sha256": catalog["sha256"],
                       "evaluation_author_packet_sha256": author_packet["sha256"],
                       "outer_partition_sha256": catalog["outer_partition_sha256"], "dataset_revision": REVISION,
                       "chunk_index": n, "events": packet["events"], "author": author,
                       "authoring_inputs": NARROW_EVALUATION_INPUTS, "acquisition_content_seen": False,
                       "model_outcomes_used": False, "human_review_complete": False,
                       "independent_agent_review_complete": False,
                       "evidence_basis": "Mechanically attached from the reviewed Channel A source unit; Channel B authors saw no factsheets.",
                       "records": records})
    review_packet = sealed({"schema": "spacing-evaluation-independent-review-packet-v1",
                            "source_catalog_sha256": catalog["sha256"], "evaluation_catalog_sha256": authored["sha256"],
                            "author_packet_sha256": author_packet["sha256"], "author_rationales_included": False,
                            "acquisition_content_included": False, "model_outcomes_included": False,
                            "reconstructed": True, "records": [evaluation_review_content(r) for r in records]})
    bindings = {"source_catalog_sha256": catalog["sha256"], "evaluation_catalog_sha256": authored["sha256"],
                "author_packet_sha256": author_packet["sha256"]}
    reviews = receipts_for("B", "spacing-evaluation-semantic-review-v1", review_packet, bindings, approvals)
    evaluation_chunks.append({"author_packet": author_packet, "authored": authored, "review_packet": review_packet,
                              "reviews": reviews})
    counts["evaluation_receipts"] += len(reviews)

# ---------------------------------------------------------------- cross-event entity review
cross_packet = load(root / "packets/cross-entity-source-packet-all100.json")
candidates = load(root / "packets/entity-candidates.json")
ledger = load(root / "compiled-final/confirmation-entity-ledger.json")
events = {row["event"]: row for row in cross_packet["events"]}
unit_by_opaque_all = {o: (row["event"], a) for row in cross_packet["events"] for a in row["source_assertions"]
                      for o in a["opaque_probe_ids"]}


def refs_for(candidate, label, event_ids):
    refs = []
    for mention in candidate.get("mentions", []) if candidate else []:
        if "source_field" in mention:
            refs.append({k: mention[k] for k in ("event", "source_field", "source_field_sha256", "line", "quote")})
        elif mention.get("opaque_probe_id") in unit_by_opaque_all:
            event, assertion = unit_by_opaque_all[mention["opaque_probe_id"]]
            refs.append({"event": event, "source_field": "source_assertions", "unit_id": assertion["unit_id"],
                         "source_assertion": assertion["source_assertion"]})
    covered = {ref["event"] for ref in refs}
    for event in event_ids:
        if event in covered:
            continue
        text = events[event]["factsheet"]["text"]
        hits = [i for i, line in enumerate(text.splitlines()) if label.casefold() in line.casefold()]
        if not hits:
            raise SystemExit(f"cross-event label {label!r} not found in {event}")
        refs.append({"event": event, "source_field": "factsheet", "source_field_sha256": digest(text), "line": hits[0],
                     "quote": text.splitlines()[hits[0]]})
        counts["cross_refs_found_by_label_search"] += 1
    return refs


listed = {row["candidate_id"]: row for key in ("repeated_entity_candidates", "answer_label_overlap_candidates",
                                                "exact_assertion_candidates") for row in candidates.get(key, [])}
judgments, extra, limitations = [], [], []
for entry in ledger["entries"]:
    if entry["review_status"] != "APPROVED":
        raise SystemExit(f"cross-event entry {entry['candidate_id']} is not approved")
    judgments.append({"candidate_id": entry["candidate_id"], "events": entry["events"], "label": entry["label"],
                      "disposition": entry["disposition"], "rationale": entry["rationale"],
                      "cross_event_leak_risk": entry["cross_event_leak_risk"], "blocking_objection": False,
                      "evidence_refs": refs_for(listed[entry["candidate_id"]], entry["label"], entry["events"])})
for entry in ledger["uncovered_recurrences"]:
    if entry["review_status"] != "APPROVED":
        raise SystemExit(f"cross-event recurrence {entry['id']} is not approved")
    extra.append({"candidate_id": entry["id"], "events": entry["events"], "label": entry["label"],
                  "disposition": entry["disposition"], "rationale": entry["rationale"],
                  "cross_event_leak_risk": entry["cross_event_leak_risk"], "blocking_objection": False,
                  "evidence_refs": refs_for(None, entry["label"], entry["events"])})
for row in judgments + extra:
    if row["cross_event_leak_risk"]:
        limitations.append({"candidate_id": row["candidate_id"], "events": row["events"], "label": row["label"],
                            "limitation": "shared name across events; tolerated under broad_LEAC with event context"})
review = sealed({"schema": "spacing-cross-entity-semantic-review-v1", "dataset_revision": REVISION,
                 "outer_partition_sha256": catalog["outer_partition_sha256"],
                 "entity_candidates_sha256": candidates["sha256"], "source_packet_sha256": cross_packet["sha256"],
                 "reviewer": {"name": "claude-haiku/confirmation-entity-ledger", **ANTHROPIC, "model": "claude-haiku",
                              "model_basis": MODEL_BASIS, "role": "semantic dispositions (ledger author)"},
                 "second_review": {"reviewer": {"name": "claude-sonnet/confirmation-entity-ledger", **ANTHROPIC,
                                                "model": "claude-sonnet", "model_basis": MODEL_BASIS},
                                   "decision_files": ["reviews/entity-ledger.r1.json", "reviews/entity-ledger.r2.json"],
                                   "outcome": "every entry APPROVED"},
                 "ledger_sha256_reference": {"path": "compiled-final/confirmation-entity-ledger.json",
                                             "canonical": ledger["sha256"]},
                 "human_review_complete": False, "model_outcomes_seen": False, "evaluation_questions_seen": False,
                 "author_judgments_seen": False, "independent_agent_review_complete": True,
                 "semantic_audit_complete": True, "reviewed_event_ids": sorted(events), "reviewed_events": len(events),
                 "candidates": judgments, "extra_candidates": extra,
                 "required_same_role_components": [], "other_same_proposition_components_requiring_protection": [],
                 "unresolved_same_proposition_objections": [], "unresolved_identity_limitations": limitations,
                 "evidence_refs_basis": ("Mechanical: candidate mentions, else the first factsheet line containing "
                                         "the label. The ledger dispositions are unchanged.")})

# ---------------------------------------------------------------- wrapper
artifacts = {name: file_sha(root / "compiled-final" / name) for name in sorted(
    p.name for p in (root / "compiled-final").glob("*.json"))}
audit = sealed({"schema": AUDIT_SCHEMA, "protocol_schema": PROTOCOL, "human_review_complete": False,
                "model_outcomes_used": False, "audit_conformance_amendment_sha256": AUDIT_CONFORMANCE_AMENDMENT_SHA256,
                "receipts_reconstructed": True,
                "governing_documents": {name: file_sha(root / name) for name in (
                    "AUTHORING-SPEC.md", "ELECTIONS-AMENDMENT.md", "ROUND3-AMENDMENT.md", "FINAL-PASS-AMENDMENT.md",
                    "AUDIT-CONFORMANCE-AMENDMENT.md", "REPAIR-SPEC.md", "REPAIR-REVIEW-SPEC.md")},
                "catalog_artifact_file_sha256": artifacts, "channel_a_directory": sys.argv[3],
                "mechanical_conversion_counts": dict(sorted(counts.items())),
                "disclosures": [
                    "Agent review is not human review; human_review_complete is false.",
                    "Receipts, review packets and the Channel B author-packet projection were rebuilt after review "
                    "from the decision files (AUDIT-CONFORMANCE-AMENDMENT.md part 1 rule 4).",
                    "Reviewers saw author rationales and notes; receipts declare author_rationales_seen true.",
                    "Final-pass records were rewritten by Claude Opus. Records still failing after the final pass "
                    "(2) and after repair round 6 (4) were settled by the study owner.",
                    "Model fields record Claude Code Agent tool aliases (claude-haiku, claude-sonnet, claude-opus).",
                    "Cross-event leak-risk names are listed as identity limitations, not same-role components."],
                "source_catalog": catalog, "probe_map": probe_map, "source_chunks": source_chunks,
                "evaluation_chunks": evaluation_chunks,
                "cross": {"source_packet": cross_packet, "entity_candidates": candidates, "review": review}})
context = validate_confirmation_audit(audit, source_facts=facts, pinned_factsheets=pinned_factsheets)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(audit, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
print(json.dumps({"audit_sha256": audit["sha256"], "file_sha256": file_sha(output),
                  "source_catalog_sha256": catalog["sha256"], "sources": len(context["sources"]),
                  "groups": len(context["groups"]), "acquisition_records": len(context["acquisition_records"]),
                  "evaluation_records": len(context["evaluation_records"]),
                  "same_role_components": context["same_role_components"],
                  "identity_limitations": len(context["identity_limitations"]),
                  "counts": dict(sorted(counts.items()))}, indent=1))
