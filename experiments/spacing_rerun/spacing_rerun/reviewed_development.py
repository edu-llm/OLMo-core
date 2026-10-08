"""Mechanically derive development bundles from retained, exact-content reviews.

This module neither authors source content nor supplies review decisions. Receipt
arguments retain every version. Only explicit, validated actual receipt edges
can supersede an exact-content rejection; reviewer identity and order cannot.
Blocked runs retain an audit and report, but never write approved bundles.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path

from .acquisition import acquisition_identity, validate_acquisition_bundle
from .common import digest, require, write_json
from .data import REVISION, assign_development_scale_roles, canonicalize, normalize, outer_partition
from .grounding import (FACTSHEETS_SHA256, build_grounded_registry, group_identity,
                        validate_bundle, validate_grounded_registry)
from .source_catalog import (OUTER_SEED, QA_SHA256, evidence_candidates, opaque_probe_id,
                             reject_question_text, sealed)
from .units import GROUNDED_POLICY, build_units

CATALOG_SHA256 = "aff2b524e93eb0b19d3b0f1ee00d811904e35d5a2ee89c730a63dcb154818d56"
CONFIG_SHA256 = "797ca68e406a6135adc2e85dae6fdcf3cd2667c69e6ec1d61902b7256bd88006"
TOKENIZER_SHA256 = "30d71356c5ba154006df5bbb4a0583fc434525ceeb2f27a7d8a237ce5db26dc6"
AUTHOR_INPUTS = ["source_assertions", "source_answer_labels", "event_factsheets"]
FIELDS = ("groups", "acquisition_records")
OUTPUT_NAMES = ("development-scale-grounded-declarations.json",
                "development-scale-acquisition-source-input.json",
                "development-scale-acquisition-qa.json")
APPROVED = ("approved", "approved_development")
NEGATIVE = ("needs_revision", "coverage_gap", "unreviewed", "rejected")
# This is the declaration in the actual chunk-02-v2 Opus receipt. Do not infer
# alternative projections from a digest mismatch or rewrite a reviewer's hash.
GROUP_SOURCE_HASH_DEFINITION = (
    'groups: sha256(canonicalJSON({"group": packet group object, "source_units": '
    '[packet source unit objects in source_unit_ids order]})); acquisition_records: '
    'sha256(canonicalJSON(packet record object)).')
# The fresh chunk-02-v3 Opus receipt explicitly declares the ordinary packet
# projection. Accept its exact declaration, without inferring a scheme by hash.
PACKET_RECORD_HASH_DEFINITION = (
    'sha256 of canonical JSON (sort_keys, separators (",",":"), ensure_ascii=False, UTF-8) '
    'of the exact packet record object as it appears in chunk-02-v3.json (group object alone; '
    'acquisition record object alone). Top-level sha256 = same canonical digest of this report excluding the sha256 key.')
# Actual R4 adjudication declares the same exact-row projection in these words.
# Retain its declaration and digest rather than rewriting the receipt.
R4_PACKET_RECORD_HASH_DEFINITION = (
    "sha256 of canonical JSON (sort_keys=True, separators (',',':'), ensure_ascii=False, UTF-8) "
    "of the exact current v4 packet row (group object alone; acquisition record object alone). "
    "Top-level sha256 = same canonical digest of this receipt excluding the 'sha256' key. "
    "Packet binding uses the packet's embedded sha256, not the file hash.")
R5_PACKET_RECORD_HASH_DEFINITION = (
    "sha256 of canonical JSON (sort_keys=True, separators (',',':'), ensure_ascii=False, UTF-8) "
    "of the exact v6 packet row. strong_v2_content_sha256 = canonical({group, source_units in "
    "source_unit_ids order}). Top-level sha256 = same canonical digest of this receipt excluding "
    "'sha256'. Packet binding uses the packet's embedded sha256.")
# Actual isolated T3 executions declared by the task owner. These are exact
# artifact bindings, not an identity inference from a generic /root label.
SOURCE_REPAIR_REQUEST = "p4-development-scale-source-repair-chunk02-v3-20261008"
SOURCE_REPAIR_SHA256 = "cfe56a4bafc211cd8ebdfa88201ddb5425763e4ea8b2aaeaf9928d1e9b85fc7e"
SOL_REVIEW_REQUEST = "p4-acquisition-rescue-sol-scale-source-review-chunk02-v2-20261008"
SOL_REVIEW_SHA256 = "896e419e77608e1fe09425858f31d5d559dc2f05e3c5d4375824b618ce64d3f8"
# These retained actual adjudications explicitly disclose a source-only first
# pass followed by reading prior judgments. Accept those exact sealed receipts,
# preserving the true disclosure; a prose claim in a new receipt is insufficient.
SOURCE_FIRST_ADJUDICATION_SHA256 = frozenset((
    "7903feefbb2285c57ffbbbb7df94fe94cb29d46534498d1f1e3ab05697f72e29",
    "988efe2a33ffdd51058db6885dd8edd9c96f0d4729a60f26a6bf4575357395e8",
    "8d8585c0835c63c609a623f26265fff9d89588ee777d8345a9577c3e1c82ced0",
    "b2d9dc90d58614bc13acf91d4219feaf9f8a4697114fccda91fb9628bd25551d",
    # Actual R6 accepts only the disclosed pre-scale diagnostic amendment.
    "a0c94078e8c91eab4251b6d860625a7f7b366b472686563ea3aba8f3ff29dddb",
))


def checked(payload, schema=None):
    require(isinstance(payload, dict) and payload.get("sha256") ==
            digest({k: v for k, v in payload.items() if k != "sha256"}), "Artifact digest differs")
    require(schema is None or payload.get("schema") == schema, "Artifact schema differs")
    return payload


def artifact(path, *, sealed_payload=True):
    path = Path(path).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw)
    if sealed_payload:
        checked(payload)
    return {"path": str(path), "file_sha256": hashlib.sha256(raw).hexdigest(), "payload": payload}


def unique(rows, key, label):
    index = {row[key]: row for row in rows}
    require(len(index) == len(rows), f"Duplicate {label}")
    return index


def person_identity(person):
    """Use an explicit identity, never the serialization of a metadata dictionary."""
    if isinstance(person, dict):
        name = next((person.get(k) for k in ("identity", "agent_id", "id", "name", "model_id", "model")
                     if person.get(k)), None)
    else:
        name = person
    require(isinstance(name, str) and name.strip(), "Missing explicit author/reviewer identity")
    name = " ".join(name.casefold().split())
    # These actual receipts name the same model in short and long forms.
    if name == "opus5.5" or ("claude-opus-5-5" in name and
                              not (isinstance(person, dict) and person.get("agent_id"))):
        return "claude-opus-5-5"
    return name


def reviewer_family(person):
    metadata = person if isinstance(person, dict) else {}
    text = " ".join(str(metadata.get(k, "")) for k in
                    ("identity", "model", "model_id")) + " " + person_identity(person)
    # Actual v5 Sol stores the concrete model name in model_family. Generic
    # family/provider labels still cannot manufacture a model identity.
    named_family = str(metadata.get("model_family", "")).casefold()
    if named_family.startswith(("gpt-", "claude-")):
        text += " " + named_family
    text = text.casefold()
    require(not (("claude" in text or "opus" in text) and "gpt" in text),
            "Reviewer identity names conflicting model families")
    inferred = "anthropic" if "claude" in text or "opus" in text else "openai" if "gpt" in text else None
    labels = [str(metadata[k]).casefold() for k in ("family", "provider") if metadata.get(k)]
    labels = ["openai" if label in ("sol", "astra", "luna", "gpt", "openai") else
              "anthropic" if label in ("opus", "sonnet", "haiku", "claude", "anthropic") else label for label in labels]
    require(not inferred or all(label == inferred for label in labels), "Reviewer family contradicts its named model")
    require(inferred is not None, "Missing actual named reviewer model family")
    return inferred


def review_content(record, field):
    keys = ("id", "event", "source_unit_ids", "statement", "evidence") if field == "groups" else (
        "id", "unit_id", "event", "answer", "questions", "evidence")
    return {key: record[key] for key in keys}


def validate_evidence(entries, event, catalog):
    lines = catalog["factsheets"][event]["text"].splitlines()
    require(entries and all(isinstance(e.get("line"), int) and not isinstance(e["line"], bool) and
                            0 <= e["line"] < len(lines) and e.get("quote") == lines[e["line"]]
                            for e in entries), "Evidence is not an exact pinned factsheet excerpt")


def validate_catalog(catalog, probe_map, facts, partition, factsheets):
    checked(catalog, "spacing-development-source-catalog-v1")
    checked(probe_map)
    require(catalog["sha256"] == CATALOG_SHA256, "Raw development source catalog hash differs")
    require(catalog["dataset_revision"] == REVISION and catalog.get("catalog_scope") == "development" and
            catalog["partition"] == partition and catalog["outer_partition_sha256"] == partition["sha256"] and
            catalog["outer_seed"] == OUTER_SEED and catalog["pinned_parquet_sha256"] ==
            {"fict_qa": QA_SHA256, "fictsheets": FACTSHEETS_SHA256}, "Catalog pinned source/partition differs")
    require(catalog.get("role_independent") is True and catalog.get("semantic_audit_complete") is False and
            catalog.get("human_review_complete") is False, "Raw catalog claimed semantic or human approval")
    selected = [dict(f, role="catalog") for f in facts if f["event"] in partition["development"]]
    registry = build_units(selected)
    by_probe = unique(selected, "id", "canonical probe")
    raw = unique(catalog["units"], "unit_id", "catalog source unit")
    require(len(selected) == 348 and len(raw) == len(registry["units"]) == 301 and
            catalog["coverage"] == {"events": 20, "canonical_probes": 348, "source_units": 301,
                                    "source_answer_targets": 314}, "Catalog source unit/probe/target loss")
    expected_map = {}
    for unit in registry["units"]:
        probes = [by_probe[pid] for pid in unit["member_probe_ids"]]
        canonical = []
        for probe in probes:
            opaque = opaque_probe_id(probe["id"])
            canonical.append({"opaque_probe_id": opaque, "source_answer_label": probe["answer"],
                              "source_style": probe["style"], "opaque_duplicate_cluster_members":
                              [opaque_probe_id(pid) for pid in probe["members"]]})
            expected_map[opaque] = {"original_probe_id": probe["id"], "source_unit_id": unit["id"],
                                   "event": unit["event"]}
        answers = sorted({probe["answer"] for probe in probes})
        expected = sealed({"unit_id": unit["id"], "event": unit["event"],
                           "source_assertion": unit["source_statement"],
                           "source_signature": unit["source_signature"], "canonical_answer_labels": answers,
                           "canonical_probes": canonical, "factsheet_sha256": digest(factsheets[unit["event"]]),
                           "evidence_candidates_only": True, "evidence_candidates": evidence_candidates(
                               unit["source_statement"], answers, factsheets[unit["event"]]),
                           "author_status": "not_authored", "review_status": "unreviewed"})
        require(raw.get(unit["id"]) == expected, "Catalog assertion/gold/probe/source hash differs from pinned source")
    require(probe_map.get("source_catalog_sha256") == CATALOG_SHA256 and probe_map["probes"] == expected_map,
            "Private original probe map differs from pinned canonical probes")
    require(catalog["factsheets"] == {event: {"text": factsheets[event], "sha256": digest(factsheets[event])}
                                      for event in partition["development"]}, "Catalog factsheets differ")


def validate_authored_chunk(authored, author_packet, catalog, *, final=True, final_chunk=None):
    """Validate source content; final_chunk may be a retained flag-only bridge."""
    checked(authored, "spacing-confirmation-authored-source-chunk-v1")
    checked(author_packet, "source-only-confirmation-authoring-input-v1")
    reject_question_text(author_packet, ())
    require(authored["source_catalog_sha256"] == author_packet["source_catalog_sha256"] == catalog["sha256"] and
            authored["source_author_packet_sha256"] == author_packet["sha256"], "Authored source packet provenance differs")
    require(authored.get("dataset_revision") == REVISION and authored.get("authoring_inputs") == AUTHOR_INPUTS and
            authored.get("evaluation_question_text_included") is False and authored.get("model_outcomes_used") is False and
            authored.get("human_review_complete") is False and authored.get("independent_agent_review_complete") is False,
            "Authored source claimed approval or changed blinded inputs")
    person_identity(authored.get("author"))
    require(authored["events"] == author_packet["events"] and authored["chunk_index"] == author_packet["chunk_index"],
            "Authored event/index differs")
    raw = unique(catalog["units"], "unit_id", "catalog source unit")
    units = unique(author_packet["units"], "unit_id", "author packet source unit")
    require(set(units) == {sid for sid, row in raw.items() if row["event"] in authored["events"]} and
            all(row == raw[sid] for sid, row in units.items()), "Author packet source units differ")
    require(author_packet["factsheets"] == {event: catalog["factsheets"][event] for event in authored["events"]} and
            author_packet.get("semantic_audit_complete") is False and author_packet.get("human_review_complete") is False,
            "Author packet factsheets or unreviewed status differs")
    sources = unique(authored["source_units"], "unit_id", "authored source unit")
    require(set(sources) == set(units), "Authored source unit loss")
    groups = unique(authored["groups"], "id", "authored grounded group")
    membership = []
    for group in groups.values():
        ids, text = group["source_unit_ids"], group["statement"]
        require(ids and ids == sorted(set(ids)) and set(ids) <= set(units) and
                group["id"] == group_identity(ids) and all(units[sid]["event"] == group["event"] for sid in ids),
                "Grounded group source membership/identity differs")
        require(group.get("review_status") == "unreviewed" and group.get("grouping_rationale") and text.strip() and
                "?" not in text and "Question:" not in text and "Answer:" not in text, "Invalid or approved authored declaration")
        validate_evidence(group["evidence"], group["event"], catalog)
        membership.extend(ids)
    require(collections.Counter(membership) == collections.Counter(units.keys()), "Group source unit loss or duplicate membership")
    metadata_limitations = []
    for source in sources.values():
        original, group = units[source["unit_id"]], groups.get(source["group_id"])
        require(group and source["unit_id"] in group["source_unit_ids"] and source["event"] == original["event"] and
                source["source_assertion"] == original["source_assertion"] and source["source_unit_sha256"] == original["sha256"] and
                sorted(source["opaque_probe_ids"]) == sorted(p["opaque_probe_id"] for p in original["canonical_probes"]) and
                source.get("review_status") == "unreviewed", "Authored original source/hash/probes changed or claimed approval")
        validate_evidence(source["evidence"], source["event"], catalog)
        expected_flags = [a for a in sorted(set(original["canonical_answer_labels"]))
                          if normalize(a) not in normalize(group["statement"])]
        if source["nonliteral_answer_labels"] != expected_flags:
            require(not final and final_chunk is not None and final_chunk["sha256"] != authored["sha256"],
                    "Nonliteral source target flags differ")
            metadata_limitations.append({"field": "source_units", "id": source["unit_id"],
                                         "metadata_field": "nonliteral_answer_labels",
                                         "retained_value": copy.deepcopy(source["nonliteral_answer_labels"]),
                                         "expected_value": expected_flags,
                                         "authored_chunk_sha256": authored["sha256"],
                                         "final_authored_chunk_sha256": final_chunk["sha256"],
                                         "flag_bridge_authored_chunk_sha256": final_chunk["sha256"],
                                         "reason": "Historical derived nonliteral flag differs; unchanged semantic content only"})
        require(source["status"] in ("grounded", "preserve_source_conflict", "preserve_source_ambiguity"),
                "Unknown source conflict status")
    targets = {(g["id"], normalize(a)) for g in groups.values() for sid in g["source_unit_ids"]
               for a in units[sid]["canonical_answer_labels"]}
    unique(authored["acquisition_records"], "id", "acquisition record")
    seen = []
    for record in authored["acquisition_records"]:
        gid, answer = record["unit_id"], normalize(record["answer"])
        require((gid, answer) in targets and record["id"] == acquisition_identity(gid, answer) and
                record["answer"] in {a for sid in groups[gid]["source_unit_ids"] for a in units[sid]["canonical_answer_labels"]} and
                record["event"] == groups[gid]["event"] and record.get("review_status") == "unreviewed" and record.get("rationale"),
                "Authored QA gold/target/membership changed or claimed approval")
        forms = record["questions"]
        require(len(forms) == 2 and len({normalize(q) for q in forms}) == 2 and all(
            isinstance(q, str) and q.strip() and "\n" not in q and "Question:" not in q and "Answer:" not in q for q in forms),
            "QA needs two distinct closed-book question forms")
        validate_evidence(record["evidence"], record["event"], catalog)
        seen.append((gid, answer))
    require(collections.Counter(seen) == collections.Counter(targets), "QA source answer target loss or duplication")
    if metadata_limitations:
        # Validate the counterpart without the historical exception. Final-only
        # singleton guards are reported separately by the builder as before.
        validate_authored_chunk(final_chunk, author_packet, catalog, final=False)
        require(all(authored[k] == final_chunk[k] for k in (
            "source_catalog_sha256", "source_author_packet_sha256", "events", "chunk_index")) and
            all(authored[field] == final_chunk[field] for field in FIELDS) and
            [{k: v for k, v in s.items() if k != "nonliteral_answer_labels"} for s in authored["source_units"]] ==
            [{k: v for k, v in s.items() if k != "nonliteral_answer_labels"} for s in final_chunk["source_units"]],
            "Historical derived flag exception requires unchanged semantic payload/source context")
    if final:
        blockers = authored_content_blockers([authored])
        require(not blockers, blockers[0]["reason"] if blockers else "")
    return metadata_limitations


def authored_content_blockers(final_chunks):
    """Report final-only guards, preserving strict source-conflict singletons.

    Historical rejected versions are retained without imposing these final
    constraints. Current violations must prevent every approved bundle.
    """
    blockers = []
    for chunk in final_chunks:
        groups, texts = {g["id"]: g for g in chunk["groups"]}, set()
        for group in groups.values():
            signature = group["event"], normalize(group["statement"])
            if signature in texts:
                blockers.append({"field": "groups", "id": group["id"],
                                 "reason": "Identical declarations must be explicitly grouped"})
            texts.add(signature)
        for source in chunk["source_units"]:
            group = groups[source["group_id"]]
            if source["status"].startswith("preserve_source") and not (
                    group["source_unit_ids"] == [source["unit_id"]] and
                    group["statement"] == source["source_assertion"] and source.get("conflict_rationale")):
                blockers.append({"field": "groups", "id": group["id"], "source_unit_id": source["unit_id"],
                                 "reason": "Source conflict singleton was merged, rewritten, or unexplained"})
    return blockers


def validate_review_packet(packet, authored, author_packet):
    checked(packet, "spacing-source-independent-review-packet-v1")
    require(packet["source_catalog_sha256"] == authored["source_catalog_sha256"] and
            packet["source_author_packet_sha256"] == author_packet["sha256"] and
            packet["authored_chunk_sha256"] == authored["sha256"] and packet["events"] == authored["events"] and
            packet["chunk_index"] == authored["chunk_index"] and packet.get("author_rationales_included") is False,
            "Independent review packet provenance/blinding differs")
    expected_sources = [{k: u[k] for k in ("unit_id", "event", "source_assertion", "canonical_answer_labels",
                                          "canonical_probes", "sha256")} for u in author_packet["units"]]
    require(packet["factsheets"] == author_packet["factsheets"] and packet["source_units"] == expected_sources,
            "Review original source/gold/factsheet coverage differs")
    for field in FIELDS:
        require(packet[field] == [review_content(row, field) for row in authored[field]],
                "Stripped review content differs from authored content")


def _blinded_receipt(receipt):
    actual_adjudication = receipt.get("sha256") in SOURCE_FIRST_ADJUDICATION_SHA256
    if actual_adjudication:
        checked(receipt, "spacing-source-semantic-review-v1")
    for field in ("independence", "disclosures"):
        independence = receipt.get(field, {})
        require(isinstance(independence, dict), "Receipt independence must be an object")
        require(all(receipt[k] == v for k, v in independence.items() if k in receipt),
                "Receipt independence claims disagree")
        receipt = {**receipt, **independence}
    aliases = (("author_rationales_seen", "author_rationales_read", "author_rationales_used"),
               ("model_outcomes_seen", "model_outcomes_read", "outcomes_read", "model_outcomes_used", "outcomes_seen"))
    scope = str(receipt.get("scope_note", "")).casefold()
    for keys in aliases:
        present = [receipt[k] for k in keys if k in receipt]
        require((present and all(v is False for v in present)) or
                (not present and "no author rationales" in scope and "outcomes" in scope),
                "Actual receipt does not attest blinded author rationale/outcome inputs")
    for key in ("evaluation_questions_seen", "evaluation_questions_read", "eval_questions_read", "other_reviewer_judgments_seen",
                "other_reviews_read", "human_review", "human_review_complete", "human_reviewed", "human_approved",
                "evaluation_question_text_used", "other_reviews_used", "author_correction_response_used",
                "author_responses_seen", "other_review_decisions_seen", "eval_probe_wording_seen",
                "author_responses_used", "author_responses_read", "correction_responses_read", "correction_responses_seen",
                "other_review_judgments_read", "other_review_decisions_used", "evaluation_question_text_included",
                "author_metadata_seen", "evaluation_gold_seen", "gold_probes_seen"):
        require(key not in receipt or receipt[key] is False or
                (key == "other_reviewer_judgments_seen" and receipt[key] is True and actual_adjudication),
                "Review saw forbidden inputs or claimed human approval")


def _receipt_decisions(receipt, field):
    aliases = ("groups", "group_reviews") if field == "groups" else (
        "acquisition_records", "records", "qa_reviews")
    decisions = {}
    for key in aliases:
        if key not in receipt:
            continue
        rows = receipt[key]
        require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows),
                "Receipt decisions must be lists of objects")
        for rid, row in unique(rows, "id", "receipt decision").items():
            require(rid not in decisions or decisions[rid] == row,
                    "Conflicting duplicate receipt decision aliases")
            decisions[rid] = row
    return list(decisions.values())


def _decision_hash(decision):
    hashes = [decision[k] for k in ("reviewed_content_sha256", "content_sha256") if k in decision]
    require(hashes and len(set(hashes)) == 1, "Actual reviewed content hash is missing or inconsistent")
    return hashes[0]


def _decision_status(decision):
    # Actual partial v6 receipts use verdict=accept and verdict=approve.
    # Preserve that raw decision while giving both explicit approvals one status.
    aliases = {"accept": "approved", "approve": "approved"}
    statuses = [aliases.get(decision[k], decision[k]) for k in ("status", "verdict") if k in decision]
    require(statuses and len(set(statuses)) == 1 and statuses[0] in APPROVED + NEGATIVE,
            "Unknown or conflicting actual review decision")
    return statuses[0]


def _receipt_supersedes(receipt):
    """Project only explicit edges, retaining each edge and its raw parent row."""
    edges = receipt.get("supersedes", [])
    require(isinstance(edges, list), "Actual review supersedes must be a list")
    edges = list(edges)
    for field in FIELDS:
        for row in _receipt_decisions(receipt, field):
            nested = row.get("supersedes", [])
            require(isinstance(nested, list), "Actual row-local review supersedes must be a list")
            for edge in nested:
                require(isinstance(edge, dict) and _decision_status(row) in APPROVED and
                        edge.get("field") == field and edge.get("id") == row["id"] and
                        edge.get("new_reviewed_content_sha256") == _decision_hash(row) and
                        isinstance(edge.get("canonical_content_sha256"), str) and
                        ("old_hash_projection" not in edge or edge["old_hash_projection"] in
                         ("exact_packet_row", "group_plus_ordered_source_units")),
                        "Actual row-local review supersession edge differs from its approved parent/projection")
                edges.append(edge)
    seen = set()
    for edge in edges:
        require(isinstance(edge, dict) and edge.get("field") in FIELDS and
                all(isinstance(edge.get(k), str) and edge[k] for k in
                    ("review_receipt_sha256", "id", "reviewed_content_sha256")) and
                ("new_reviewed_content_sha256" not in edge or
                 isinstance(edge["new_reviewed_content_sha256"], str) and edge["new_reviewed_content_sha256"]),
                "Actual review supersession edge is malformed")
        target = tuple(edge[k] for k in ("review_receipt_sha256", "field", "id", "reviewed_content_sha256"))
        require(target not in seen, "Duplicate or conflicting actual review supersession edge")
        seen.add(target)
    return edges


def _receipt_packet_hash(receipt):
    packet_ref = receipt.get("packet", {})
    require(isinstance(packet_ref, dict), "Receipt packet reference must be an object")
    packet_hash = receipt.get("review_packet_embedded_sha256",
                              receipt.get("review_packet_sha256", receipt.get("packet_sha256", packet_ref.get("sha256"))))
    require("packet_sha256" not in receipt or receipt["packet_sha256"] == packet_hash,
            "Receipt packet digest claims disagree")
    require("sha256" not in packet_ref or packet_ref["sha256"] == packet_hash,
            "Receipt packet digest claims disagree")
    return packet_hash


def _group_source_projection(packet, group):
    sources = unique(packet["source_units"], "unit_id", "packet source unit")
    members = group["source_unit_ids"]
    require(members and members == sorted(set(members)) and set(members) <= set(sources) and
            group["id"] == group_identity(members) and
            all(sources[sid]["event"] == group["event"] for sid in members),
            "Stronger group projection source membership/order/provenance differs")
    return {"group": group, "source_units": [sources[sid] for sid in members]}


def _packet_context(packet):
    return {k: packet[k] for k in ("source_catalog_sha256", "source_author_packet_sha256", "chunk_index",
                                  "events", "source_units", "factsheets")}


def _record_context(packet, field, row):
    group = row if field == "groups" else unique(packet["groups"], "id", "packet group")[row["unit_id"]]
    return {**_group_source_projection(packet, group), "factsheet": packet["factsheets"][row["event"]]}


def _require_acyclic_supersession(graph):
    visiting, visited = set(), set()

    def visit(node):
        require(node not in visiting, "Actual review supersession cycle")
        if node in visited:
            return
        visiting.add(node)
        for prior in graph.get(node, ()):
            visit(prior)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


def _delegated_binding(person, retained, evidence, request, expected_hash):
    require(retained["payload"]["sha256"] == expected_hash,
            "Actual task binding differs from exact retained payload hash")
    matches = [row for row in evidence.get("records", []) if row.get("clientRequestId") == request]
    require(len(matches) == 1, "Missing or ambiguous actual completed task evidence: " + request)
    task = matches[0]
    require(task.get("status") == task.get("latestTerminalStatus") == "completed" and
            task.get("hasPendingChildRuns") is False and task.get("taskId") and task.get("childThreadId") and
            task.get("childRunId") == task.get("latestTerminalRunId") and task.get("childRunId") and
            task.get("providerInstanceId") and task.get("resultContextTransferId") and
            request in task["taskId"] and request in task["childThreadId"] and
            Path(retained["path"]).name in task.get("latestTerminalSummary", "") and
            isinstance(person, dict) and person.get("model") == task.get("model"),
            "Actual task execution/model/result differs from bound artifact")
    return {"channel": "task:" + task["taskId"], "identity": copy.deepcopy(person),
            "payload_sha256": expected_hash, "artifact_path": retained["path"],
            "artifact_file_sha256": retained["file_sha256"], "task_record": copy.deepcopy(task)}


def _native_binding(person, retained, evidence):
    rows = list(evidence.get("native_completed_authors", []))
    if evidence.get("native_completed_author"):
        rows.append(evidence["native_completed_author"])
    require(all(isinstance(row, dict) and isinstance(row.get("actual_result", {}), dict) for row in rows),
            "Actual native completion evidence must be objects")
    matches = [row for row in rows if row.get("artifact_sha256", row.get("actual_result", {}).get("authored_sha256")) ==
               retained["payload"]["sha256"]]
    require(len(matches) == 1, "Missing or ambiguous actual native completion evidence: " + retained["path"])
    task = matches[0]
    text = task.get("observed_result_text", task.get("actual_result_text", ""))
    canonical_name = task.get("canonical_agent_name", task.get("canonical_task_name", ""))
    # The confirmation adapter also supports an explicit structured actual
    # result. Its raw completion text need not repeat the hash verbatim.
    structured = task.get("schema") == "spacing-native-author-task-evidence-v1"
    if structured:
        require(task.get("result_source") and task.get("actual_result") == {
            "authored_sha256": retained["payload"]["sha256"],
            "author_packet_sha256": retained["payload"]["source_author_packet_sha256"],
            "input_channel": "source_only", "identity": person},
            "Actual native structured result differs from exact source-only payload")
    require(canonical_name.startswith("/root/") and canonical_name == person.get("agent_id") and
            ("artifact" not in task and structured or
             Path(task.get("artifact", "")).resolve() == Path(retained["path"]).resolve()) and
            isinstance(text, str) and text.strip() and (structured or (
                retained["payload"]["sha256"] in text and Path(retained["path"]).name in text)) and
            task.get("status", "completed" if not structured else None) == "completed" and
            ("model" not in task or task["model"] == person.get("model")),
            "Actual native completion differs from exact author/artifact/result")
    return {"channel": "native:" + canonical_name, "identity": copy.deepcopy(person),
            "payload_sha256": retained["payload"]["sha256"], "artifact_path": retained["path"],
            "artifact_file_sha256": retained["file_sha256"], "task_record": copy.deepcopy(task)}


def _author_channels(final_chunks, authored_artifacts, task_evidence):
    evidence = task_evidence["payload"] if task_evidence else {}
    if task_evidence:
        require(evidence.get("schema") == "spacing-actual-delegation-status-evidence-v1", "Actual task evidence schema differs")
    files = {a["payload"]["sha256"]: a for a in _authored_history(authored_artifacts or [])}
    channels, raw_names, bindings = set(), set(), []
    for chunk in final_chunks:
        name = person_identity(chunk["author"])
        raw_names.add(name)
        if name in ("/root", "root"):
            require(task_evidence is not None and chunk["sha256"] in files,
                    "Source author cannot supply an independent review without actual task provenance")
            current, visited = chunk, set()
            while current["sha256"] != SOURCE_REPAIR_SHA256:
                require(current["sha256"] not in visited, "Actual author provenance cycle")
                visited.add(current["sha256"])
                revision = current.get("metadata", {}).get("actual_revision_author")
                require(isinstance(revision, dict), "Inherited generic author lacks actual revision provenance")
                binding = _native_binding(revision, files[current["sha256"]], evidence)
                bindings.append(binding)
                channels.add(binding["channel"])
                raw_names.add(person_identity(revision))
                prior = files.get(current.get("metadata", {}).get("prior_authored_revision_sha256"))
                require(prior and all(prior["payload"][k] == current[k] for k in
                        ("author", "source_catalog_sha256", "source_author_packet_sha256", "events", "chunk_index")),
                        "Inherited source-author task changed raw identity or source context")
                current = prior["payload"]
            binding = _delegated_binding(current["author"], files[current["sha256"]], evidence,
                                         SOURCE_REPAIR_REQUEST, SOURCE_REPAIR_SHA256)
            bindings.append(binding)
            channels.add(binding["channel"])
        else:
            channels.add("named:" + name)
            revision = chunk.get("metadata", {}).get("actual_revision_author")
            if revision:
                require(task_evidence is not None and chunk["sha256"] in files,
                        "Revision author lacks actual completed task provenance")
                binding = _native_binding(revision, files[chunk["sha256"]], evidence)
                bindings.append(binding)
                channels.add(binding["channel"])
                raw_names.add(person_identity(revision))
    return channels, raw_names, bindings


def review_coverage(final_chunks, packet_artifacts, receipt_artifacts, *, authored_artifacts=None, task_evidence=None):
    """Retain actual judgments; only explicit actual edges resolve rejections.

    Unchanged content carries forward across packets with identical pinned source
    context. The stronger declared group projection keeps its actual review hash
    separate from the canonical packet record digest used for content coverage.
    """
    packets = unique([a["payload"] for a in packet_artifacts], "sha256", "review packet digest")
    for packet in packets.values():
        checked(packet, "spacing-source-independent-review-packet-v1")
    files = {a["payload"]["sha256"]: a for a in packet_artifacts}
    author_channels, authors, author_bindings = _author_channels(final_chunks, authored_artifacts, task_evidence)
    current_packets = {}
    for chunk in final_chunks:
        matches = [p for p in packets.values() if p["authored_chunk_sha256"] == chunk["sha256"]]
        require(len(matches) == 1, "Missing or ambiguous final exact-content review packet")
        current_packets[chunk["chunk_index"]] = matches[0]
        for field in FIELDS:
            require(matches[0][field] == [review_content(row, field) for row in chunk[field]],
                    "Final review packet content differs from authored content")
    # Carry-forward cannot change the source context behind an identical row.
    for packet in packets.values():
        current = current_packets.get(packet["chunk_index"])
        require(current is not None and _packet_context(packet) == _packet_context(current),
            "Retained packet source/member order/provenance differs from final packet")
    histories = collections.defaultdict(list)
    actual_receipts, actual_decisions = {}, {}
    for order, retained in enumerate(receipt_artifacts):
        receipt = checked(retained["payload"], "spacing-source-semantic-review-v1")
        require(receipt["sha256"] not in actual_receipts, "Duplicate actual review receipt digest")
        actual_receipts[receipt["sha256"]] = receipt
        packet_hash = _receipt_packet_hash(receipt)
        packet_ref = receipt.get("packet", {})
        packet = packets.get(packet_hash)
        require(packet is not None, "Actual receipt references a missing retained review packet")
        if "review_packet_embedded_sha256" in receipt:
            # v5 declares raw/canonical; v6 declares canonical/canonical.
            # Both bind the explicitly declared embedded hash to this packet.
            require(receipt.get("review_packet_sha256") in (packet_hash, files[packet_hash]["file_sha256"]),
                    "Receipt declared packet file digest or canonical alias differs")
        require(receipt.get("review_packet_file_sha256", packet_ref.get("file_sha256", files[packet["sha256"]]["file_sha256"])) ==
                files[packet["sha256"]]["file_sha256"], "Review packet file digest differs")
        require("file_sha256" not in packet_ref or packet_ref["file_sha256"] == files[packet["sha256"]]["file_sha256"],
                "Nested review packet file digest differs")
        for key in ("schema", "authored_chunk_sha256", "source_author_packet_sha256"):
            require(key not in packet_ref or packet_ref[key] == packet[key], "Nested receipt packet provenance differs")
        for key in ("authored_chunk_sha256", "source_author_packet_sha256"):
            require(key not in receipt or receipt[key] == packet[key], "Receipt packet provenance differs")
        require(receipt.get("source_catalog_sha256", packet["source_catalog_sha256"]) == packet["source_catalog_sha256"],
                "Receipt source catalog hash differs")
        require("chunk_index" not in receipt or receipt["chunk_index"] == packet["chunk_index"], "Receipt chunk index differs")
        if "events" in receipt and receipt["events"] != packet["events"]:
            reviewed_events = {row["event"] for field in FIELDS for row in packet[field]
                               if row["id"] in {d["id"] for d in _receipt_decisions(receipt, field)}}
            require(receipt.get("partial") is True and reviewed_events and
                    receipt["events"] == [event for event in packet["events"] if event in reviewed_events],
                    "Receipt events differ from exact partial decision scope")
        _blinded_receipt(receipt)
        reviewer, family = person_identity(receipt.get("reviewer")), reviewer_family(receipt.get("reviewer"))
        reviewer_binding = None
        reviewer_channel = "named:" + reviewer
        if reviewer in ("/root", "root"):
            require(task_evidence is not None, "Source author cannot supply an independent review without actual reviewer task evidence")
            reviewer_binding = _delegated_binding(receipt["reviewer"], retained, task_evidence["payload"],
                                                  SOL_REVIEW_REQUEST, SOL_REVIEW_SHA256)
            reviewer_channel = reviewer_binding["channel"]
        require(reviewer_channel not in author_channels and (reviewer_binding is not None or reviewer not in authors),
                "Source author cannot supply an independent review in the same actual task channel")
        if reviewer_binding:
            require(all(all(reviewer_binding["task_record"][key] != binding["task_record"][key]
                            for key in ("taskId", "childThreadId", "childRunId"))
                        for binding in author_bindings if binding["channel"].startswith("task:")),
                    "Source author and reviewer share an actual task/thread/run channel")
        coverage = receipt.get("coverage", {})
        require(isinstance(coverage, dict), "Receipt coverage must be an object")
        global_gap = receipt.get("coverage_gap") is True or receipt.get("coverage_complete") is False or any(
            coverage.get(k) is False for k in ("complete", "every_source_unit_grouped_exactly_once",
                "all_original_target_labels_preserved", "original_source_units_assigned_exactly_once",
                "original_gold_targets_fully_preserved"))
        definition = receipt.get("reviewed_content_sha256_definition")
        require(definition is None or definition in (GROUP_SOURCE_HASH_DEFINITION, PACKET_RECORD_HASH_DEFINITION,
                                                     R4_PACKET_RECORD_HASH_DEFINITION, R5_PACKET_RECORD_HASH_DEFINITION),
                "Unsupported actual reviewed content hash definition")
        for field in FIELDS:
            original = unique(packet[field], "id", "packet record")
            decisions = unique(_receipt_decisions(receipt, field), "id", "receipt decision")
            declared_count = coverage.get("groups_reviewed" if field == "groups" else "acquisition_records_reviewed")
            gap = global_gap or (declared_count is not None and declared_count != len(decisions))
            for rid, decision in decisions.items():
                stronger = field == "groups" and definition == GROUP_SOURCE_HASH_DEFINITION
                content = _group_source_projection(packet, original[rid]) if stronger and rid in original else original.get(rid)
                require(rid in original and _decision_hash(decision) == digest(content),
                        "Actual reviewed content hash differs from its retained packet")
                require(all(decision[k] == original[rid][k] for k in original[rid] if k in decision),
                        "Receipt exact payload differs from its retained packet")
                if "strong_v2_content_sha256" in decision:
                    require(field == "groups" and decision["strong_v2_content_sha256"] ==
                            digest(_group_source_projection(packet, original[rid])),
                            "Receipt declared stronger auxiliary hash differs from its retained packet")
                if field == "acquisition_records":
                    group = next(g for g in packet["groups"] if g["id"] == original[rid]["unit_id"])
                    sources = {s["unit_id"]: s for s in packet["source_units"]}
                    source_rows = [sources[sid] for sid in group["source_unit_ids"]]
                    context = {"source_unit_ids": group["source_unit_ids"],
                               "source_unit_sha256": [s["sha256"] for s in source_rows],
                               "source_assertions": [s["source_assertion"] for s in source_rows],
                               "source_answer_labels": sorted({a for s in source_rows for a in s["canonical_answer_labels"]}),
                               "group_content_sha256": digest(group),
                               "factsheet_sha256": packet["factsheets"][original[rid]["event"]]["sha256"]}
                    require(all(decision[k] == value for k, value in context.items() if k in decision),
                            "Receipt source/group/factsheet hashes or original target context differs")
                _decision_status(decision)
                require(decision.get("notes"), "Actual review decision lacks retained notes")
            # An explicit coverage gap blocks the matching packet content even
            # when the report omitted a row. An unclaimed partial receipt does not.
            for rid in original if gap else decisions:
                decision = decisions.get(rid)
                stronger = field == "groups" and receipt.get("reviewed_content_sha256_definition") == GROUP_SOURCE_HASH_DEFINITION
                entry = {
                    "reviewer": reviewer_channel if reviewer_binding else reviewer,
                    "reviewer_raw_identity": copy.deepcopy(receipt["reviewer"]),
                    "reviewer_task_provenance": reviewer_binding,
                    "reviewer_family": family, "receipt_path": retained["path"],
                    "receipt_sha256": receipt["sha256"], "receipt_file_sha256": retained["file_sha256"],
                    "review_packet_sha256": packet["sha256"], "input_order": order,
                    "reviewed_content_sha256": _decision_hash(decision) if decision else None,
                    "packet_content_sha256": digest(original[rid]),
                    "packet_content": copy.deepcopy(original[rid]),
                    "hash_scheme": "group_with_ordered_source_units" if stronger else "packet_record",
                    "reviewed_content": copy.deepcopy(_group_source_projection(packet, original[rid]) if stronger else original[rid]),
                    "status": "coverage_gap" if gap else _decision_status(decision), "decision": copy.deepcopy(decision)}
                histories[(field, rid, digest(original[rid]))].append(entry)
                if decision:
                    actual_decisions[(receipt["sha256"], field, rid, _decision_hash(decision))] = entry
    superseded, graph, validated_edges = {}, collections.defaultdict(set), []
    for receipt_sha, receipt in actual_receipts.items():
        for edge in _receipt_supersedes(receipt):
            prior_sha = edge.get("review_receipt_sha256")
            target = edge["field"], edge.get("id"), edge.get("reviewed_content_sha256")
            prior = actual_decisions.get((prior_sha, *target))
            explicit_new_hash = "new_reviewed_content_sha256" in edge
            new_target = (*target[:2], edge["new_reviewed_content_sha256"] if explicit_new_hash else target[2])
            new = actual_decisions.get((receipt_sha, *new_target))
            require(prior_sha in actual_receipts and prior_sha != receipt_sha and prior is not None and new is not None and
                    prior["status"] in NEGATIVE and _decision_status(prior["decision"]) in NEGATIVE and
                    new["status"] in APPROVED and _decision_status(new["decision"]) in APPROVED and
                    prior["packet_content"] == new["packet_content"] and
                    _packet_context(packets[prior["review_packet_sha256"]]) ==
                    _packet_context(packets[new["review_packet_sha256"]]) and
                    _record_context(packets[prior["review_packet_sha256"]], target[0], prior["packet_content"]) ==
                    _record_context(packets[new["review_packet_sha256"]], target[0], new["packet_content"]) and
                    (not explicit_new_hash or edge.get("canonical_content_sha256") == prior["packet_content_sha256"]) and
                    ("canonical_content_sha256" not in edge or
                     edge["canonical_content_sha256"] == prior["packet_content_sha256"] == new["packet_content_sha256"]) and
                    ("old_hash_projection" not in edge or edge["old_hash_projection"] == {
                        "packet_record": "exact_packet_row",
                        "group_with_ordered_source_units": "group_plus_ordered_source_units"}[prior["hash_scheme"]]) and
                    (explicit_new_hash or (prior["hash_scheme"] == new["hash_scheme"] and
                                           prior["reviewed_content"] == new["reviewed_content"])),
                    "Actual review supersession edge does not bind a prior rejection and new approval")
            superseded.setdefault((prior_sha, *target), set()).add(receipt_sha)
            graph[receipt_sha].add(prior_sha)
            validated_edges.append({**copy.deepcopy(edge), "receipt_sha256": receipt_sha})
    _require_acyclic_supersession(graph)
    result, blockers = [], []
    for chunk in final_chunks:
        for field in FIELDS:
            for row in chunk[field]:
                content_hash = digest(review_content(row, field))
                history = histories[(field, row["id"], content_hash)]
                active = []
                for entry in history:
                    resolving = superseded.get((entry["receipt_sha256"], field, row["id"], entry["reviewed_content_sha256"]), ())
                    entry["superseded_by_receipt_sha256"] = sorted(resolving)
                    if not resolving:
                        active.append(entry)
                approvals = [e for e in active if e["status"] in APPROVED]
                negative = [e for e in active if e["status"] not in APPROVED]
                complete = not negative and len({e["reviewer"] for e in approvals}) >= 2 and len(
                    {e["reviewer_family"] for e in approvals}) >= 2
                reasons = []
                if negative:
                    reasons.append("unresolved exact-content needs_revision/rejection/coverage_gap")
                if len({e["reviewer_family"] for e in approvals}) < 2 or len({e["reviewer"] for e in approvals}) < 2:
                    reasons.append("missing two independent exact-content reviewer identities/families")
                record = {"field": field, "id": row["id"], "event": row["event"],
                          "reviewed_content_sha256": content_hash, "complete": complete,
                          "packet_content_sha256": content_hash,
                          "active_decisions": active,
                          # Compatibility name: all unsuperseded judgments, not
                          # one latest judgment per normalized reviewer.
                          "latest_decisions": active, "matching_history": history, "blockers": reasons}
                result.append(record)
                if not complete:
                    blockers.append({k: record[k] for k in ("field", "id", "event", "reviewed_content_sha256", "blockers")})
    return sealed({"schema": "spacing-development-exact-review-coverage-v1", "records": result,
                   "author_task_provenance": author_bindings, "actual_task_evidence": task_evidence,
                   "validated_supersedes": validated_edges,
                   "blockers": blockers, "independent_agent_review_complete": not blockers,
                   "human_review_complete": False})


def derive_bundles(facts, partition, roles, catalog, final_chunks, coverage, audit_hash, token_count):
    """Only reachable after actual receipt coverage passes; preserve authored text."""
    require(coverage["independent_agent_review_complete"] is True and not coverage["blockers"],
            "Cannot derive approved artifacts from blocked actual reviews")
    require(not authored_content_blockers(final_chunks), "Cannot derive approved artifacts from blocked authored content")
    facts = copy.deepcopy(facts)
    frozen = {f["id"]: {k: copy.deepcopy(f[k]) for k in ("id", "event", "question", "answer", "aliases", "members", "style")}
              for f in facts}
    source_registry = build_units(facts)
    original = {r["unit_id"]: r for c in final_chunks for r in c["source_units"]}
    authored_groups = {g["id"]: g for c in final_chunks for g in c["groups"]}
    authored_qa = [r for c in final_chunks for r in c["acquisition_records"]]
    by_probe = {f["id"]: f for f in facts}
    reviewer_names = sorted({e["reviewer"] for r in coverage["records"] for e in r["latest_decisions"]})
    provenance = {"audit_path": "reviewed-development-audit.json", "audit_sha256": audit_hash,
                  "coverage_sha256": coverage["sha256"], "source_catalog_sha256": catalog["sha256"]}
    sources = []
    for unit in source_registry["units"]:
        if unit["role"] not in ("old", "new"):
            continue
        authored = original[unit["id"]]
        row = {"unit_id": unit["id"], "event_id": unit["event"], "role": unit["role"],
               "source_statement": unit["source_statement"], "canonical_answers":
               [by_probe[pid]["answer"] for pid in unit["member_probe_ids"]], "probe_ids": unit["member_probe_ids"]}
        row.update(index=len(sources), source_statement_sha256=digest(unit["source_statement"]),
                   source_fields_sha256=digest(row), source_unit_sha256=next(
                       u["sha256"] for u in catalog["units"] if u["unit_id"] == unit["id"]),
                   group_id=authored["group_id"], trained_declaration=authored_groups[authored["group_id"]]["statement"],
                   factsheet_sha256=catalog["factsheets"][unit["event"]]["sha256"],
                   status=authored["status"], conflict_rationale=authored["conflict_rationale"],
                   evidence=copy.deepcopy(authored["evidence"]), nonliteral_answer_labels=authored["nonliteral_answer_labels"],
                   review_status="approved_development")
        sources.append(row)
    groups = []
    for gid, authored in sorted(authored_groups.items()):
        if roles[authored["event"]] not in ("old", "new"):
            continue
        group = copy.deepcopy(authored)
        group.update(role=roles[group["event"]], source_indices=[r["index"] for r in sources if r["group_id"] == gid],
                     statement_tokens=token_count(group["statement"]), review_status="approved_development")
        groups.append(group)
    grounding = sealed({"schema": "spacing-grounded-declarations-v1", "policy": GROUNDED_POLICY, "mode": "development",
                        "source_dataset_revision": REVISION, "source_factsheets_sha256": FACTSHEETS_SHA256,
                        "source_catalog_sha256": catalog["sha256"], "partition_sha256": partition["sha256"],
                        "roles_sha256": digest(roles), "authoring_inputs": ["source_statement", "canonical_answers", "event_factsheet"],
                        "evaluation_question_text_included": False, "model_outcomes_used": False,
                        "independent_agent_review_complete": True, "independent_agent_reviewer": reviewer_names,
                        "human_review_complete": False, "review_provenance": provenance,
                        "source_unit_count": len(sources), "original_evaluation_probe_count": len(facts),
                        "factsheets": {e: catalog["factsheets"][e]["text"] for e in sorted({r["event_id"] for r in sources})},
                        "source_units": sources, "groups": groups,
                        "max_statement_tokens": max(g["statement_tokens"] for g in groups)})
    validate_bundle(grounding, facts, source_registry, mode="development", partition_hash=partition["sha256"],
                    roles=roles, pinned_factsheets={e: row["text"] for e, row in catalog["factsheets"].items()})
    registry = build_grounded_registry(facts, source_registry, grounding)
    validate_grounded_registry(facts, registry, source_registry, grounding)
    old_sources = {s["unit_id"]: s for s in sources if s["role"] == "old"}
    units = []
    for unit in registry["units"]:
        if unit["role"] != "old":
            continue
        assertions = [old_sources[sid] for sid in unit["source_unit_ids"]]
        units.append({"unit_id": unit["id"], "event": unit["event"], "statement": unit["statement"],
                      "answer_targets": sorted({a for s in assertions for a in s["canonical_answers"]}),
                      "source_assertions": [{"statement": s["source_statement"], "answer_targets": sorted(set(s["canonical_answers"])),
                                             "status": s["status"], "evidence": s["evidence"]} for s in assertions]})
    source_input = sealed({"schema": "source-only-acquisition-authoring-input-v1", "dataset_revision": REVISION,
                           "source_catalog_sha256": catalog["sha256"], "grounding_bundle_sha256": grounding["sha256"],
                           "review_provenance": provenance, "human_review_complete": False,
                           "independent_agent_review_complete": True, "authoring_inputs": AUTHOR_INPUTS,
                           "evaluation_question_text_included": False, "model_outcomes_used": False,
                           "input_scope": "old reviewed units; original assertions, answer labels and pinned factsheets",
                           "units": units, "factsheets": {e: grounding["factsheets"][e] for e in sorted({u["event"] for u in units})}})
    records = [dict(copy.deepcopy(r), review_status="approved_development") for r in authored_qa if roles[r["event"]] == "old"]
    qa = sealed({"schema": "spacing-acquisition-qa-v1", "mode": "development", "source_catalog_sha256": catalog["sha256"],
                 "grounding_bundle_sha256": grounding["sha256"], "source_input_sha256": source_input["sha256"],
                 "authoring_inputs": AUTHOR_INPUTS, "evaluation_question_text_included": False, "model_outcomes_used": False,
                 "independent_agent_review_complete": True, "independent_agent_reviewer": reviewer_names,
                 "human_review_complete": False, "review_provenance": provenance, "records": records})
    validate_acquisition_bundle(qa, source_input, facts, registry, grounding, mode="development")
    require(len(facts) == len(frozen) and all(frozen[f["id"]] == {k: f[k] for k in frozen[f["id"]]} for f in facts),
            "Derivation changed evaluation questions/gold/aliases or lost probes")
    return grounding, source_input, qa


def training_content_blockers(final_chunks, facts, roles, catalog):
    """Report current validators' constraints without repairing authored text.

    Historical packets can contain rejected questions. Duplicate acquisition
    forms matter for the final old-only training stream, not unused QA roles.
    """
    heldout = {normalize(f[k]) for f in facts if f["role"] in ("old", "new", "control")
               for k in ("question", "paraphrase") if f.get(k)}
    complete = {" ".join(f["question"].casefold().split()) for f in facts
                if f["role"] in ("old", "new", "control")}
    raw = {u["unit_id"]: u for u in catalog["units"]}
    groups = {g["id"]: g for c in final_chunks for g in c["groups"]}
    blockers, seen = [], set()
    for group in groups.values():
        if roles[group["event"]] in ("old", "new") and (normalize(group["statement"]) in heldout or
                any(q in " ".join(group["statement"].casefold().split()) for q in complete)):
            blockers.append({"field": "groups", "id": group["id"], "reason": "held-out evaluation question in declaration"})
    for chunk in final_chunks:
        for record in chunk["acquisition_records"]:
            if roles[record["event"]] != "old":
                continue
            group = groups[record["unit_id"]]
            declarations = [group["statement"]] + [raw[sid]["source_assertion"] for sid in group["source_unit_ids"]]
            for question in record["questions"]:
                text = normalize(question)
                reason = ("duplicate acquisition question across targets" if text in seen else
                          "held-out evaluation question in acquisition" if text in heldout else
                          "acquisition question embeds its source declaration" if any(
                              normalize(d) in text for d in declarations) else None)
                if reason:
                    blockers.append({"field": "acquisition_records", "id": record["id"], "reason": reason})
                seen.add(text)
    return blockers


def _retained_paths(root, directory, supplied):
    paths = [Path(p).resolve() for p in supplied]
    require(len(paths) == len(set(paths)), f"Duplicate supplied {directory} file")
    require(set((root / directory).glob("*.json")) <= set(paths), f"Retain all {directory} versions, including blocking receipts")
    return paths


def _flag_bridge_content(chunk):
    return {**{key: chunk[key] for key in ("source_catalog_sha256", "source_author_packet_sha256", "events", "chunk_index")},
            **{field: chunk[field] for field in FIELDS},
            "source_units": [{k: v for k, v in row.items() if k != "nonliteral_answer_labels"}
                             for row in chunk["source_units"]]}


def _flags_correct(chunk, author_packet):
    groups = unique(chunk["groups"], "id", "authored grounded group")
    units = unique(author_packet["units"], "unit_id", "author packet source unit")
    return all(row["nonliteral_answer_labels"] == [a for a in sorted(set(units[row["unit_id"]]["canonical_answer_labels"]))
               if normalize(a) not in normalize(groups[row["group_id"]]["statement"])] for row in chunk["source_units"])


def _authored_history(files):
    """Corrections remain retained artifacts, never masquerade as full chunks."""
    schemas = {"spacing-confirmation-authored-source-chunk-v1", "spacing-source-author-correction-response-v1"}
    require(all(a["payload"].get("schema") in schemas for a in files), "Unknown retained authored artifact schema")
    return [a for a in files if a["payload"]["schema"] == "spacing-confirmation-authored-source-chunk-v1"]


def _validate_retained_authored_versions(history_files, author_packets_by_hash, catalog, final, packet_files, receipts):
    """Retain digest-bound history, disclosing only verified derived flag limits."""
    author_index = unique([a["payload"] for a in _authored_history(history_files)], "sha256", "authored file digest")
    final_by_index = unique(final, "chunk_index", "final authored chunk")
    reviewed_packet_hashes = {_receipt_packet_hash(r["payload"]) for r in receipts}
    bound_authored_hashes = {a["payload"]["authored_chunk_sha256"] for a in packet_files
                            if a["payload"]["sha256"] in reviewed_packet_hashes} | {c["sha256"] for c in final}
    limitations = []
    for authored_hash in sorted(bound_authored_hashes):
        authored = author_index.get(authored_hash)
        require(authored is not None, "Missing retained authored version for review packet")
        packet = author_packets_by_hash.get(authored["source_author_packet_sha256"])
        require(packet is not None, "Missing retained source author packet")
        counterpart = final_by_index.get(authored["chunk_index"])
        require(counterpart is not None, "Retained review has no selected final chunk counterpart")
        bridge = None
        if authored_hash not in {c["sha256"] for c in final} and not _flags_correct(authored, packet):
            candidates = [c for c in author_index.values() if c["sha256"] != authored_hash and
                          _flag_bridge_content(c) == _flag_bridge_content(authored) and _flags_correct(c, packet)]
            require(candidates, "Historical derived flag lacks a retained valid flag-only bridge")
            # Prefer the selected final only when its semantic payload matches.
            immediate = [c for c in candidates if c.get("metadata", {}).get("prior_authored_revision_sha256") == authored_hash]
            bridge = (sorted(immediate, key=lambda c: c["sha256"])[0] if immediate else counterpart
                      if counterpart in candidates else sorted(candidates, key=lambda c: c["sha256"])[0])
        found = validate_authored_chunk(authored, packet, catalog, final=False,
                                        final_chunk=bridge)
        retained = next(a for a in history_files if a["payload"]["sha256"] == authored_hash)
        bridge_file = next((a for a in history_files if bridge and a["payload"]["sha256"] == bridge["sha256"]), None)
        limitations.extend({**row, "authored_path": retained["path"],
                            "authored_file_sha256": retained["file_sha256"],
                            "final_authored_chunk_sha256": counterpart["sha256"],
                            "flag_bridge_path": bridge_file["path"],
                            "flag_bridge_file_sha256": bridge_file["file_sha256"],
                            "author": copy.deepcopy(authored["author"])} for row in found)
    for retained in packet_files:
        packet = retained["payload"]
        authored = author_index.get(packet["authored_chunk_sha256"])
        require(authored is not None, "Missing retained authored version for review packet")
        validate_review_packet(packet, authored, author_packets_by_hash[authored["source_author_packet_sha256"]])
    return bound_authored_hashes, limitations


def build_reviewed_development(source_catalog_root, authored_chunk_paths, review_packet_paths, review_paths,
                               source_directory, config_path, output, *, tokenizer_json=None, task_evidence_path=None):
    root, output = Path(source_catalog_root).resolve(), Path(output).resolve()
    require(not output.exists(), "Output already exists; never overwrite reviewed development artifacts")
    catalog_file, map_file = artifact(root / "source-catalog.json"), artifact(root / "private-probe-id-map.json")
    index_file, plan_file = artifact(root / "author-chunk-index.json"), artifact(root / "development-scale-role-plan.json")
    config_file = artifact(config_path, sealed_payload=False)
    config = config_file["payload"]
    require(digest(config) == CONFIG_SHA256, "Frozen scaled-grid-01 configuration hash differs")
    packet_paths = _retained_paths(root, "review-packets", review_packet_paths)
    receipt_paths = _retained_paths(root, "reviews", review_paths)
    final_paths = [Path(p).resolve() for p in authored_chunk_paths]
    require(final_paths and len(final_paths) == len(set(final_paths)), "Missing or duplicate final authored chunk paths")
    history_paths = sorted(set((root / "authored").glob("*.json")) | set(final_paths))
    history_files = [artifact(p) for p in history_paths]
    chunks = _authored_history(history_files)
    author_index = unique([a["payload"] for a in chunks], "sha256", "authored file digest")
    require(all(any(a["path"] == str(p) for a in chunks) for p in final_paths), "Selected final is not an authored source chunk")
    final = [next(a["payload"] for a in chunks if a["path"] == str(p)) for p in final_paths]
    packet_files, receipts = [artifact(p) for p in packet_paths], [artifact(p) for p in receipt_paths]
    author_packets = [artifact(root / row["path"]) for row in index_file["payload"]["chunks"]]
    author_packets_by_hash = unique([a["payload"] for a in author_packets], "sha256", "author packet digest")
    catalog, probe_map = catalog_file["payload"], map_file["payload"]
    import pyarrow.parquet as pq
    parquet_files = {"fict_qa": Path(source_directory) / "fict_qa.parquet",
                     "fictsheets": Path(source_directory) / "fictsheets.parquet"}
    pinned = {name: {"path": str(path.resolve()), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for name, path in parquet_files.items()}
    require({k: v["file_sha256"] for k, v in pinned.items()} == {"fict_qa": QA_SHA256, "fictsheets": FACTSHEETS_SHA256},
            "Pinned local parquet file hash differs")
    rows = pq.read_table(parquet_files["fict_qa"]).to_pylist()
    sheet_rows = pq.read_table(parquet_files["fictsheets"]).to_pylist()
    facts, canonical_audit = canonicalize(rows)
    require(len(rows) == 7500 and len(facts) == 1797 and len(sheet_rows) == 100, "Pinned canonical source coverage differs")
    sheets = {str(r["event_id"]): str(r["fictsheet"]) for r in sheet_rows}
    require(len(sheets) == 100, "Duplicated pinned factsheet event")
    partition = outer_partition(facts, canonical_audit["cross_event_links"], config["outer_seed"])
    validate_catalog(catalog, probe_map, facts, partition, sheets)
    forbidden_questions = {" ".join(str(row["question"]).casefold().split()) for row in rows if str(row["question"]).strip()}
    for packet in [catalog, *author_packets_by_hash.values()]:
        reject_question_text(packet, forbidden_questions)
    split = assign_development_scale_roles(facts, partition, config["mode"], config["development_role_counts"],
                                           config["development_fixed_qa_events"])
    require(plan_file["payload"]["source_catalog_sha256"] == catalog["sha256"] and
            plan_file["payload"]["roles"] == split["roles"] and
            index_file["payload"]["source_catalog_sha256"] == catalog["sha256"], "Retained role plan/author index differs")
    for entry in index_file["payload"]["chunks"]:
        packet = author_packets_by_hash.get(entry["sha256"])
        require(packet and entry["events"] == packet["events"] and entry["source_units"] == len(packet["units"]) and
                entry["canonical_probes"] == sum(len(u["canonical_probes"]) for u in packet["units"]), "Author chunk index differs")
    # A concurrent unreviewed draft is retained but cannot change eligibility
    # for the explicitly selected final versions. Validate all versions bound
    # by actual receipts, as well as the selected finals.
    bound_authored_hashes, historical_limitations = _validate_retained_authored_versions(
        history_files, author_packets_by_hash, catalog, final, packet_files, receipts)
    source_ids = [s["unit_id"] for c in final for s in c["source_units"]]
    require(collections.Counter(source_ids) == collections.Counter(u["unit_id"] for u in catalog["units"]),
            "Final authored source unit loss or overlap")
    for field in FIELDS:
        unique([r for c in final for r in c[field]], "id", f"final {field}")
    require(all(any(a["payload"]["authored_chunk_sha256"] == c["sha256"] for a in packet_files) for c in final),
            "Missing final exact-content independent review packet")
    task_evidence = artifact(task_evidence_path, sealed_payload=False) if task_evidence_path else None
    coverage = review_coverage(final, packet_files, receipts, authored_artifacts=history_files, task_evidence=task_evidence)
    authored_blockers = authored_content_blockers(final)
    content_blockers = training_content_blockers(final, split["facts"], split["roles"], catalog)
    ready = coverage["independent_agent_review_complete"] and not authored_blockers and not content_blockers
    audit = sealed({"schema": "spacing-reviewed-development-provenance-v1", "source_catalog_sha256": catalog["sha256"],
                    "source_catalog": catalog_file, "private_probe_map": map_file, "author_chunk_index": index_file,
                    "role_plan": plan_file, "frozen_config": config_file, "pinned_local_parquets": pinned,
                    "outer_partition": partition, "roles": split["roles"], "roles_sha256": digest(split["roles"]),
                    "canonical_probes": split["facts"], "canonical_probes_sha256": digest(split["facts"]),
                    "canonical_probe_count": len(split["facts"]),
                    "author_packets": author_packets, "authored_versions": chunks,
                    "correction_responses": [a for a in history_files if a not in chunks],
                    "unbound_authored_version_sha256": sorted(set(author_index) - bound_authored_hashes),
                    "final_authored_paths": list(map(str, final_paths)), "review_packets": packet_files, "actual_receipts": receipts,
                    "coverage": coverage, "authored_content_blockers": authored_blockers,
                    "historical_metadata_validation_limitations": historical_limitations,
                    "training_content_blockers": content_blockers,
                    "independent_agent_review_complete": ready,
                    "human_review_complete": False})
    bundles = None
    if ready:
        require(tokenizer_json is not None, "Approved derivation requires a pinned local --tokenizer-json for token counts")
        tokenizer_path = Path(tokenizer_json).resolve()
        require(hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == TOKENIZER_SHA256, "Pinned tokenizer file hash differs")
        audit = sealed({**{k: v for k, v in audit.items() if k != "sha256"},
                        "tokenizer": {"path": str(tokenizer_path), "file_sha256": TOKENIZER_SHA256}})
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        bundles = derive_bundles(split["facts"], partition, split["roles"], catalog, final, coverage, audit["sha256"],
                                 lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))
    report = sealed({"schema": "spacing-reviewed-development-validation-v1", "status": "approved" if bundles else "blocked",
                     "audit_sha256": audit["sha256"], "source_catalog_sha256": catalog["sha256"],
                     "canonical_probe_count": 348, "source_unit_count": 301, "source_answer_target_count": 314,
                     "event_role_counts": dict(collections.Counter(split["roles"].values())),
                     "records_checked": len(coverage["records"]),
                     "blocking_records": len({(r["field"], r["id"]) for r in coverage["blockers"] + authored_blockers + content_blockers}),
                     "blockers": coverage["blockers"], "authored_content_blockers": authored_blockers,
                     "historical_metadata_validation_limitations": historical_limitations,
                     "training_content_blockers": content_blockers,
                     "independent_agent_review_complete": bool(bundles),
                     "human_review_complete": False, "approved_artifacts": {name: b["sha256"] for name, b in
                                                                               zip(OUTPUT_NAMES, bundles or ())}})
    # All validation precedes the exclusive directory creation. Existing output
    # directories, including outputs from blocked attempts, are never reused.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "reviewed-development-audit.json", audit)
    write_json(output / "validation-report.json", report)
    for name, bundle in zip(OUTPUT_NAMES, bundles or ()):
        write_json(output / name, bundle)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-catalog-root", required=True)
    parser.add_argument("--authored-chunk", action="append", required=True, help="Final authored chunk; repeat for every chunk")
    parser.add_argument("--review-packet", action="append", required=True, help="Retained review packet; include all versions")
    parser.add_argument("--review", action="append", required=True, help="Actual receipt; include all versions, in any order")
    parser.add_argument("--source-directory", required=True, help="Pinned local fict_qa.parquet and fictsheets.parquet")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/scaled-grid-01.json"))
    parser.add_argument("--tokenizer-json", help="Pinned local model tokenizer.json; required only when reviews pass")
    parser.add_argument("--actual-task-evidence", help="Actual completed T3/native task inventory for generic identities")
    parser.add_argument("--output", required=True, help="New directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = build_reviewed_development(args.source_catalog_root, args.authored_chunk, args.review_packet, args.review,
                                            args.source_directory, args.config, args.output, tokenizer_json=args.tokenizer_json,
                                            task_evidence_path=args.actual_task_evidence)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Blocked: {exc}\n")
    print(json.dumps({k: report[k] for k in ("status", "canonical_probe_count", "source_unit_count", "records_checked",
                                            "blocking_records", "approved_artifacts", "human_review_complete")}, indent=2))
    return 0 if report["status"] == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
