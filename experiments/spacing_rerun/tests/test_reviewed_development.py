"""Synthetic fixtures exercise receipt gates without inventing live approvals."""
import copy
import json
from pathlib import Path

import pytest
import spacing_rerun.reviewed_development as development

from spacing_rerun.acquisition import acquisition_identity
from spacing_rerun.common import digest, write_json
from spacing_rerun.data import REVISION
from spacing_rerun.grounding import group_identity
from spacing_rerun.reviewed_development import (
    AUTHOR_INPUTS, CATALOG_SHA256, FIELDS, GROUP_SOURCE_HASH_DEFINITION, PACKET_RECORD_HASH_DEFINITION,
    _authored_history, _delegated_binding, _native_binding,
    _require_acyclic_supersession, _retained_paths, _validate_retained_authored_versions,
    artifact, authored_content_blockers, build_reviewed_development, checked, derive_bundles, person_identity,
    review_content, review_coverage, training_content_blockers,
    validate_authored_chunk, validate_catalog, validate_review_packet,
)
from spacing_rerun.source_catalog import opaque_probe_id, sealed
from spacing_rerun.units import build_units


def rehash(payload):
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})
    return payload


def envelope(payload, name):
    # File hashes in these in-memory test fixtures are deliberately synthetic.
    return {"path": f"/synthetic-fixture/{name}.json", "file_sha256": digest([name, payload]), "payload": payload}


@pytest.fixture
def source_fixture():
    """All names, events, sources and receipts here are invented test data."""
    facts = []
    for i, role in enumerate(("old", "old", "new", "control", "qa")):
        facts.append({"id": f"synthetic-probe-{i}", "event": f"synthetic-event-{i}", "role": role,
                      "statement": f"Synthetic device {i} uses material {i}.", "answer": f"material {i}",
                      "aliases": [f"material {i}"], "question": f"What material does synthetic device {i} use?",
                      "members": [f"synthetic-probe-{i}"], "style": "test"})
    # A second canonical probe with the same source and answer must survive.
    facts.append(dict(copy.deepcopy(facts[0]), id="synthetic-probe-duplicate", members=["synthetic-probe-duplicate"],
                      question="Give the material of synthetic device zero?"))
    registry = build_units(facts)
    units, sheets = [], {}
    by_probe = {f["id"]: f for f in facts}
    for unit in registry["units"]:
        text = unit["source_statement"]
        sheets[unit["event"]] = {"text": text, "sha256": digest(text)}
        units.append(sealed({"unit_id": unit["id"], "event": unit["event"], "source_assertion": text,
                             "canonical_answer_labels": sorted({by_probe[p]["answer"] for p in unit["member_probe_ids"]}),
                             "canonical_probes": [{"opaque_probe_id": opaque_probe_id(p)} for p in unit["member_probe_ids"]]}))
    catalog = sealed({"units": units, "factsheets": sheets})
    author_packet = sealed({"schema": "source-only-confirmation-authoring-input-v1",
                            "source_catalog_sha256": catalog["sha256"], "units": units, "factsheets": sheets,
                            "events": sorted(sheets), "chunk_index": 1,
                            "semantic_audit_complete": False, "human_review_complete": False})
    groups, sources, records = [], [], []
    for i, unit in enumerate(units):
        gid = group_identity([unit["unit_id"]])
        evidence = [{"line": 0, "quote": unit["source_assertion"]}]
        groups.append({"id": gid, "event": unit["event"], "source_unit_ids": [unit["unit_id"]],
                       "statement": unit["source_assertion"], "evidence": evidence,
                       "grouping_rationale": "Synthetic singleton test fixture.", "review_status": "unreviewed"})
        sources.append({"unit_id": unit["unit_id"], "event": unit["event"], "source_assertion": unit["source_assertion"],
                        "source_unit_sha256": unit["sha256"], "opaque_probe_ids": [p["opaque_probe_id"] for p in unit["canonical_probes"]],
                        "group_id": gid, "status": "grounded", "conflict_rationale": "", "nonliteral_answer_labels": [],
                        "evidence": evidence, "review_status": "unreviewed"})
        answer = unit["canonical_answer_labels"][0]
        records.append({"id": acquisition_identity(gid, answer), "unit_id": gid, "event": unit["event"], "answer": answer,
                        "questions": [f"Supply the value of synthetic test slot {i}?", f"Name synthetic slot {i}'s value?"],
                        "evidence": evidence, "rationale": "Synthetic question test fixture.", "review_status": "unreviewed"})
    authored = sealed({"schema": "spacing-confirmation-authored-source-chunk-v1", "dataset_revision": REVISION,
                       "source_catalog_sha256": catalog["sha256"], "source_author_packet_sha256": author_packet["sha256"],
                       "chunk_index": 1, "events": author_packet["events"], "author": {"agent_id": "synthetic-author", "model": "gpt-test"},
                       "authoring_inputs": AUTHOR_INPUTS, "evaluation_question_text_included": False, "model_outcomes_used": False,
                       "independent_agent_review_complete": False, "human_review_complete": False,
                       "source_units": sources, "groups": groups, "acquisition_records": records})
    return facts, catalog, author_packet, authored


def packet_for(authored, author_packet):
    return sealed({"schema": "spacing-source-independent-review-packet-v1", "source_catalog_sha256": authored["source_catalog_sha256"],
                   "source_author_packet_sha256": author_packet["sha256"], "authored_chunk_sha256": authored["sha256"],
                   "events": authored["events"], "chunk_index": 1, "author_rationales_included": False,
                   "factsheets": author_packet["factsheets"], "source_units": author_packet["units"],
                   **{field: [review_content(r, field) for r in authored[field]] for field in FIELDS}})


def receipt_for(packet, reviewer="synthetic-reviewer/gpt-test", *, field=None, ids=None, status="approved"):
    """An in-memory decision on synthetic content, never a live receipt file."""
    payload = {"schema": "spacing-source-semantic-review-v1", "review_packet_sha256": packet["sha256"],
               "reviewer": reviewer, "author_rationales_seen": False, "model_outcomes_seen": False, "human_review_complete": False}
    for key in FIELDS:
        payload[key] = [{"id": r["id"], "status": status, "notes": "Synthetic test decision.", "reviewed_content_sha256": digest(r)}
                        for r in packet[key] if (field is None or field == key) and (ids is None or r["id"] in ids)]
    return sealed(payload)


OPUS = {"identity": "synthetic-independent-reviewer", "model": "claude-test", "provider": "anthropic"}


def coverage_for(authored, packet, receipts):
    return review_coverage([authored], [envelope(packet, "packet")], [envelope(r, f"receipt-{i}") for i, r in enumerate(receipts)])


def test_two_actual_named_families_required_for_every_group_and_qa(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    one = receipt_for(packet)
    assert not coverage_for(authored, packet, [one])["independent_agent_review_complete"]
    result = coverage_for(authored, packet, [one, receipt_for(packet, OPUS)])
    assert result["independent_agent_review_complete"] and not result["blockers"]
    assert len(result["records"]) == len(authored["groups"]) + len(authored["acquisition_records"])
    assert result["human_review_complete"] is False
    assert result["records"][0]["latest_decisions"][0]["receipt_path"].endswith("receipt-0.json")


def test_distinct_identities_from_same_family_cannot_count_twice(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    result = coverage_for(authored, packet, [receipt_for(packet), receipt_for(packet, "another-independent/gpt-test")])
    assert result["blockers"]


def test_unchanged_v1_carries_forward_but_changed_v2_needs_new_exact_receipts(source_fixture):
    _, _, author_packet, authored = source_fixture
    v1 = packet_for(authored, author_packet)
    receipts = [receipt_for(v1), receipt_for(v1, OPUS)]
    final = copy.deepcopy(authored)
    final["acquisition_records"][0]["questions"][0] += " Revised"
    rehash(final)
    v2 = packet_for(final, author_packet)
    packets = [envelope(v1, "v1"), envelope(v2, "v2")]
    result = review_coverage([final], packets, [envelope(r, str(i)) for i, r in enumerate(receipts)])
    rid = final["acquisition_records"][0]["id"]
    assert [b["id"] for b in result["blockers"]] == [rid]
    for reviewer in ("synthetic-reviewer/gpt-test", OPUS):
        receipts.append(receipt_for(v2, reviewer, field="acquisition_records", ids={rid}))
    result = review_coverage([final], packets, [envelope(r, str(i)) for i, r in enumerate(receipts)])
    assert not result["blockers"]


@pytest.mark.parametrize("status", ["needs_revision", "coverage_gap", "unreviewed", "rejected"])
def test_any_matching_negative_blocks_even_with_two_other_approvals(source_fixture, status):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipts = [receipt_for(packet), receipt_for(packet, OPUS), receipt_for(packet, "third-independent/gpt-test", status=status)]
    result = coverage_for(authored, packet, receipts)
    assert result["blockers"] and all(not r["complete"] for r in result["records"])


@pytest.mark.parametrize("reverse", [False, True])
def test_same_reviewer_approval_cannot_erase_prior_negative_without_edge(source_fixture, reverse):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipts = [receipt_for(packet, status="needs_revision"), receipt_for(packet, OPUS), receipt_for(packet)]
    receipts[0]["reviewed_at"] = "2000-01-01"
    receipts[-1]["reviewed_at"] = "2099-01-01"
    for receipt in receipts:
        rehash(receipt)
    if reverse:
        receipts.reverse()
    result = coverage_for(authored, packet, receipts)
    assert result["blockers"] and all(len(r["matching_history"]) == 3 for r in result["records"])
    assert all(len(r["active_decisions"]) == 3 for r in result["records"])
    assert result["validated_supersedes"] == []


def test_normalized_actual_reviewer_alias_cannot_erase_rejection(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, "Opus5.5", status="needs_revision")
    later = receipt_for(packet, {"identity": "claude-opus-5-5 independent review sub-agent"})
    result = coverage_for(authored, packet, [prior, later, receipt_for(packet)])
    assert result["blockers"]
    assert person_identity(prior["reviewer"]) == person_identity(later["reviewer"])


def test_each_actual_rejection_requires_its_own_receipt_edge(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, status="needs_revision")
    another = copy.deepcopy(prior)
    another["review_round"] = "second synthetic actual receipt"
    rehash(another)
    new = superseding_receipt(packet, prior)
    result = coverage_for(authored, packet, [prior, another, new, receipt_for(packet, OPUS)])
    assert result["blockers"]
    assert all(any(e["receipt_sha256"] == another["sha256"] for e in r["active_decisions"]) for r in result["records"])


def superseding_receipt(packet, prior, reviewer="synthetic-reviewer/gpt-test"):
    new = receipt_for(packet, reviewer)
    new["supersedes"] = [{"review_receipt_sha256": prior["sha256"], "field": field,
                          "id": row["id"], "reviewed_content_sha256": row["reviewed_content_sha256"]}
                         for field in FIELDS for row in prior[field]]
    return rehash(new)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("reviewer", ["synthetic-reviewer/gpt-test", "different-actual-reviewer/gpt-test"])
def test_actual_supersedes_edges_resolve_only_bound_rejections(source_fixture, reverse, reviewer):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, status="needs_revision")
    new = superseding_receipt(packet, prior, reviewer)
    before = copy.deepcopy([prior, new])
    receipts = [prior, receipt_for(packet, OPUS), new]
    result = coverage_for(authored, packet, list(reversed(receipts)) if reverse else receipts)
    assert not result["blockers"]
    assert len(result["validated_supersedes"]) == len(result["records"])
    for record in result["records"]:
        assert len(record["matching_history"]) == 3 and len(record["active_decisions"]) == 2
        rejected = next(e for e in record["matching_history"] if e["receipt_sha256"] == prior["sha256"])
        assert rejected["superseded_by_receipt_sha256"] == [new["sha256"]]
        assert rejected["decision"]["status"] == "needs_revision"
    assert [prior, new] == before


@pytest.mark.parametrize("mutation", ["prior_sha", "field", "id", "digest", "prior_approved", "new_rejected", "self", "not_list"])
def test_supersedes_requires_actual_matching_prior_and_new_decisions(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, status="approved" if mutation == "prior_approved" else "needs_revision")
    new = superseding_receipt(packet, prior)
    edge = new["supersedes"][0]
    if mutation == "prior_sha":
        edge["review_receipt_sha256"] = "missing-actual-receipt"
    elif mutation == "field":
        edge["field"] = "acquisition_records"
    elif mutation == "id":
        edge["id"] = "missing-record"
    elif mutation == "digest":
        edge["reviewed_content_sha256"] = "different-content"
    elif mutation == "new_rejected":
        new["groups"][0]["status"] = "needs_revision"
    elif mutation == "self":
        edge["review_receipt_sha256"] = new["sha256"]
    elif mutation == "not_list":
        new["supersedes"] = edge
    rehash(new)
    with pytest.raises(ValueError, match="supersession edge|supersedes must"):
        coverage_for(authored, packet, [prior, new, receipt_for(packet, OPUS)])


def test_supersession_graph_rejects_cross_record_cycles():
    # Sealed reciprocal SHA references cannot be authored without a hash fixed
    # point; exercise cycle detection directly with test-only graph identities.
    with pytest.raises(ValueError, match="supersession cycle"):
        _require_acyclic_supersession({"receipt-a": {"receipt-b"}, "receipt-b": {"receipt-a"}})
    _require_acyclic_supersession({"receipt-c": {"receipt-b"}, "receipt-b": {"receipt-a"}})


def test_actual_receipt_schema_variants_bind_to_the_same_exact_payload(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet, OPUS)
    receipt["packet"] = {"sha256": receipt.pop("review_packet_sha256"),
                         "file_sha256": envelope(packet, "packet")["file_sha256"],
                         "schema": packet["schema"], "authored_chunk_sha256": packet["authored_chunk_sha256"]}
    receipt["author_rationales_used"] = receipt.pop("author_rationales_seen")
    receipt["model_outcomes_used"] = receipt.pop("model_outcomes_seen")
    receipt["records"] = receipt.pop("acquisition_records")
    for row in receipt["records"]:
        row["content_sha256"] = row.pop("reviewed_content_sha256")
        original = next(r for r in packet["acquisition_records"] if r["id"] == row["id"])
        row.update({k: original[k] for k in ("unit_id", "event", "answer", "questions")})
    rehash(receipt)
    assert not coverage_for(authored, packet, [receipt_for(packet), receipt])["blockers"]
    receipt["records"][0]["questions"] = ["A different synthetic question?", "Another different synthetic question?"]
    rehash(receipt)
    with pytest.raises(ValueError, match="Receipt exact payload"):
        coverage_for(authored, packet, [receipt])


def test_actual_explicit_file_and_embedded_hashes_and_nested_independence(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet, {"name": "synthetic named reviewer", "provider": "OpenAI", "model_family": "gpt-test"})
    receipt["review_packet_embedded_sha256"] = receipt["review_packet_sha256"]
    receipt["review_packet_sha256"] = envelope(packet, "packet")["file_sha256"]
    receipt["independence"] = {k: receipt.pop(k) for k in ("author_rationales_seen", "model_outcomes_seen")}
    rehash(receipt)
    assert not coverage_for(authored, packet, [receipt, receipt_for(packet, OPUS)])["blockers"]
    receipt["review_packet_sha256"] = "different-file-hash"
    rehash(receipt)
    with pytest.raises(ValueError, match="declared packet file digest"):
        coverage_for(authored, packet, [receipt])


@pytest.mark.parametrize("raw_pair", [False, True])
def test_declared_packet_aliases_bind_only_actual_canonical_or_file_pair(source_fixture, raw_pair):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet)
    receipt["review_packet_embedded_sha256"] = packet["sha256"]
    receipt["review_packet_sha256"] = envelope(packet, "packet")["file_sha256"] if raw_pair else packet["sha256"]
    receipt["packet"] = {"sha256": packet["sha256"]}
    rehash(receipt)
    assert not coverage_for(authored, packet, [receipt, receipt_for(packet, OPUS)])["blockers"]
    receipt["packet"]["sha256"] = "conflicting-canonical-hash"
    rehash(receipt)
    with pytest.raises(ValueError, match="packet digest claims disagree"):
        coverage_for(authored, packet, [receipt])


@pytest.mark.parametrize("verdict", ["accept", "approve"])
def test_partial_qa_verdict_disclosures_and_packet_alias_preserve_actual_judgment(source_fixture, verdict):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    review = receipt_for(packet, field="acquisition_records", ids={packet["acquisition_records"][0]["id"]})
    review["packet_sha256"] = review.pop("review_packet_sha256")
    review["disclosures"] = {k: review.pop(k) for k in ("author_rationales_seen", "model_outcomes_seen", "human_review_complete")}
    review["acquisition_records"][0]["verdict"] = verdict
    del review["acquisition_records"][0]["status"]
    rehash(review)
    before = copy.deepcopy(review)
    result = coverage_for(authored, packet, [review, receipt_for(packet, OPUS)])
    record = next(r for r in result["records"] if r["id"] == review["acquisition_records"][0]["id"])
    assert record["complete"]
    assert record["matching_history"][0]["decision"] == review["acquisition_records"][0]
    assert review == before
    review["acquisition_records"][0]["status"] = "needs_revision"
    rehash(review)
    with pytest.raises(ValueError, match="conflicting actual review decision"):
        coverage_for(authored, packet, [review])


@pytest.mark.parametrize("mutation", ["packet", "disclosure", "unknown_verdict"])
def test_fresh_receipt_aliases_cannot_hide_conflicting_or_unknown_claims(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    review = receipt_for(packet)
    if mutation == "packet":
        review["packet_sha256"] = "different-canonical-hash"
    elif mutation == "disclosure":
        review["disclosures"] = {"author_rationales_seen": True}
    else:
        review["acquisition_records"][0]["verdict"] = "looks good"
    rehash(review)
    with pytest.raises(ValueError, match="claims disagree|Unknown or conflicting"):
        coverage_for(authored, packet, [review])


def synthetic_task_binding(person, retained, request):
    """In-memory transport evidence for invented artifacts only."""
    return {"clientRequestId": request, "taskId": "synthetic-task:" + request,
            "childThreadId": "synthetic-thread:" + request, "childRunId": "synthetic-run:" + request,
            "latestTerminalRunId": "synthetic-run:" + request, "status": "completed",
            "latestTerminalStatus": "completed", "hasPendingChildRuns": False,
            "providerInstanceId": "synthetic-provider", "model": person["model"],
            "resultContextTransferId": "synthetic-result:" + request,
            "latestTerminalSummary": "Synthetic test artifact " + Path(retained["path"]).name}


@pytest.mark.parametrize("mutation", ["status", "terminal", "pending", "model", "hash", "request", "thread", "run", "summary", "missing"])
def test_task_binding_rejects_incomplete_or_misbound_execution(source_fixture, mutation):
    _, _, _, authored = source_fixture
    retained = envelope(authored, "synthetic-author")
    request = "synthetic-source-author-task"
    row = synthetic_task_binding(authored["author"], retained, request)
    expected_hash = authored["sha256"]
    if mutation == "status":
        row["status"] = "running"
    elif mutation == "terminal":
        row["latestTerminalStatus"] = "failed"
    elif mutation == "pending":
        row["hasPendingChildRuns"] = True
    elif mutation == "model":
        row["model"] = "different-model"
    elif mutation == "hash":
        expected_hash = "different-payload"
    elif mutation == "request":
        row["clientRequestId"] = "different-task"
    elif mutation == "thread":
        row["childThreadId"] = "different-thread"
    elif mutation == "run":
        row["latestTerminalRunId"] = "different-run"
    elif mutation == "summary":
        row["latestTerminalSummary"] = "different artifact"
    evidence = {"records": [] if mutation == "missing" else [row]}
    with pytest.raises(ValueError, match="Actual task|actual completed task"):
        _delegated_binding(authored["author"], retained, evidence, request, expected_hash)


def test_generic_labels_use_actual_separate_channels_and_keep_negative_judgments(source_fixture, monkeypatch):
    _, _, author_packet, authored = source_fixture
    authored["author"] = {"agent_id": "/root", "model": "gpt-test"}
    rehash(authored)
    monkeypatch.setattr(development, "SOURCE_REPAIR_SHA256", authored["sha256"])
    packet = packet_for(authored, author_packet)
    review = receipt_for(packet, dict(authored["author"]), status="needs_revision")
    monkeypatch.setattr(development, "SOL_REVIEW_SHA256", review["sha256"])
    author_file, review_file = envelope(authored, "synthetic-author"), envelope(review, "synthetic-reviewer")
    evidence = envelope({"schema": "spacing-actual-delegation-status-evidence-v1", "records": [
        synthetic_task_binding(authored["author"], author_file, development.SOURCE_REPAIR_REQUEST),
        synthetic_task_binding(review["reviewer"], review_file, development.SOL_REVIEW_REQUEST)]}, "synthetic-evidence")
    before = copy.deepcopy((authored, review, evidence))
    result = review_coverage([authored], [envelope(packet, "packet")], [review_file,
        envelope(receipt_for(packet), "sol"), envelope(receipt_for(packet, OPUS), "opus")],
        authored_artifacts=[author_file], task_evidence=evidence)
    assert len(result["blockers"]) == len(result["records"])
    for record in result["records"]:
        negative = next(r for r in record["active_decisions"] if r["status"] == "needs_revision")
        assert negative["reviewer_raw_identity"] == review["reviewer"]
        assert negative["reviewer"].startswith("task:")
        assert negative["reviewer_task_provenance"]["payload_sha256"] == review["sha256"]
    assert result["actual_task_evidence"] == evidence
    assert result["author_task_provenance"][0]["identity"] == authored["author"]
    assert (authored, review, evidence) == before


@pytest.mark.parametrize("mutation", ["missing", "hash", "identity", "path", "text", "status", "model"])
def test_native_completion_requires_exact_payload_identity_and_observed_result(source_fixture, mutation):
    _, _, _, authored = source_fixture
    person = {"agent_id": "/root/synthetic-author", "model": "gpt-test"}
    retained = envelope(authored, "synthetic-native-authored")
    task = {"canonical_agent_name": person["agent_id"], "artifact": retained["path"],
            "artifact_sha256": authored["sha256"], "observed_result_text":
            Path(retained["path"]).name + " " + authored["sha256"], "status": "completed", "model": "gpt-test"}
    assert _native_binding(person, retained, {"native_completed_author": task})["payload_sha256"] == authored["sha256"]
    if mutation == "hash":
        task["artifact_sha256"] = "different-hash"
    elif mutation == "identity":
        task["canonical_agent_name"] = "/root/different-author"
    elif mutation == "path":
        task["artifact"] = "/synthetic-fixture/different.json"
    elif mutation == "text":
        task["observed_result_text"] = "No exact payload hash"
    elif mutation == "status":
        task["status"] = "running"
    elif mutation == "model":
        task["model"] = "different-model"
    evidence = {} if mutation == "missing" else {"native_completed_author": task}
    with pytest.raises(ValueError, match="actual native completion|Actual native completion"):
        _native_binding(person, retained, evidence)


def test_native_structured_actual_result_binds_hash_without_editing_completion_text(source_fixture):
    _, _, _, authored = source_fixture
    person = {"agent_id": "/root/synthetic-native-author", "model": "gpt-test"}
    retained = envelope(authored, "synthetic-native-artifact")
    task = {"schema": "spacing-native-author-task-evidence-v1", "canonical_task_name": person["agent_id"],
            "status": "completed", "model": "gpt-test", "actual_result_text": "Synthetic native task completed.",
            "result_source": "Synthetic test observation", "actual_result": {
                "authored_sha256": authored["sha256"], "author_packet_sha256": authored["source_author_packet_sha256"],
                "input_channel": "source_only", "identity": person}}
    before = copy.deepcopy(task)
    assert _native_binding(person, retained, {"native_completed_authors": [task]})["task_record"] == before
    assert task == before
    task["actual_result"]["author_packet_sha256"] = "different-author-packet"
    with pytest.raises(ValueError, match="structured result differs"):
        _native_binding(person, retained, {"native_completed_authors": [task]})


@pytest.mark.parametrize("mutation", [None, "missing_completion", "changed_identity", "changed_context", "missing_prior", "cycle"])
def test_inherited_generic_author_binds_each_native_revision_to_exact_retained_payload(source_fixture, monkeypatch, mutation):
    _, _, _, anchor = source_fixture
    anchor["author"] = {"agent_id": "/root", "model": "gpt-test"}
    rehash(anchor)
    monkeypatch.setattr(development, "SOURCE_REPAIR_SHA256", anchor["sha256"])
    revision_person = {"agent_id": "/root/synthetic-native-author", "model": "gpt-test", "role": "source_only"}
    bridge = copy.deepcopy(anchor)
    bridge["metadata"] = {"actual_revision_author": revision_person, "prior_authored_revision_sha256": anchor["sha256"]}
    rehash(bridge)
    final = copy.deepcopy(bridge)
    final["metadata"]["prior_authored_revision_sha256"] = bridge["sha256"]
    final["acquisition_records"][0]["questions"][0] += " Later revision?"
    if mutation == "changed_identity":
        final["author"]["model"] = "different-model"
    elif mutation == "changed_context":
        final["source_author_packet_sha256"] = "different-source-packet"
    elif mutation == "missing_prior":
        final["metadata"]["prior_authored_revision_sha256"] = "absent-payload"
    rehash(final)
    files = [envelope(anchor, "anchor"), envelope(bridge, "bridge"), envelope(final, "final")]
    if mutation == "cycle":
        # A synthetic cycle is tested directly at the provenance layer. It
        # cannot occur in genuine sealed payloads without changing their hash.
        final["metadata"]["prior_authored_revision_sha256"] = final["sha256"]
    evidence = {"schema": "spacing-actual-delegation-status-evidence-v1", "records": [
        synthetic_task_binding(anchor["author"], files[0], development.SOURCE_REPAIR_REQUEST)],
        "native_completed_authors": [{"canonical_agent_name": revision_person["agent_id"], "status": "completed",
            "artifact": f["path"], "artifact_sha256": f["payload"]["sha256"], "observed_result_text":
            Path(f["path"]).name + " " + f["payload"]["sha256"]} for f in files[1:]]}
    if mutation == "missing_completion":
        evidence["native_completed_authors"].pop()
    if mutation is not None:
        with pytest.raises(ValueError, match="actual native completion|Inherited source-author task|Actual author provenance cycle"):
            development._author_channels([final], files, envelope(evidence, "evidence"))
    else:
        channels, names, bindings = development._author_channels([final], files, envelope(evidence, "evidence"))
        assert channels == {"task:synthetic-task:" + development.SOURCE_REPAIR_REQUEST,
                            "native:/root/synthetic-native-author"}
        assert names == {"/root", "/root/synthetic-native-author"}
        assert {r["payload_sha256"] for r in bindings} == {c["payload"]["sha256"] for c in files}


def test_nested_independence_cannot_hide_conflicting_top_level_claim(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet)
    receipt["independence"] = {"author_rationales_seen": True}
    rehash(receipt)
    with pytest.raises(ValueError, match="independence claims disagree"):
        coverage_for(authored, packet, [receipt])


@pytest.mark.parametrize("alias", ["records", "qa_reviews"])
def test_named_decision_aliases_retain_every_decision_and_notes(source_fixture, alias):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet, OPUS)
    receipt["group_reviews"] = receipt.pop("groups")
    receipt[alias] = receipt.pop("acquisition_records")
    rehash(receipt)
    before = copy.deepcopy(receipt)
    result = coverage_for(authored, packet, [receipt_for(packet), receipt])
    assert not result["blockers"]
    for record in result["records"]:
        key = "group_reviews" if record["field"] == "groups" else alias
        decision = next(d for d in receipt[key] if d["id"] == record["id"])
        entry = next(e for e in record["matching_history"] if e["receipt_sha256"] == receipt["sha256"])
        assert entry["decision"] == decision
        assert entry["reviewed_content_sha256"] == decision["reviewed_content_sha256"]
    assert receipt == before


@pytest.mark.parametrize("alias,field", [("group_reviews", "groups"), ("records", "acquisition_records"),
                                        ("qa_reviews", "acquisition_records")])
def test_alias_duplicates_require_identical_actual_decisions(source_fixture, alias, field):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet, OPUS)
    receipt[alias] = copy.deepcopy(receipt[field])
    rehash(receipt)
    result = coverage_for(authored, packet, [receipt_for(packet), receipt])
    assert not result["blockers"]
    assert all(len(r["matching_history"]) == 2 for r in result["records"])
    receipt[alias][0]["notes"] += " Conflicting duplicate."
    rehash(receipt)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        coverage_for(authored, packet, [receipt])


def stronger_receipt(packet):
    receipt = receipt_for(packet, OPUS)
    receipt["reviewed_content_sha256_definition"] = GROUP_SOURCE_HASH_DEFINITION
    sources = {s["unit_id"]: s for s in packet["source_units"]}
    for row, group in zip(receipt["groups"], packet["groups"]):
        row["reviewed_content_sha256"] = digest({"group": group, "source_units":
                                                [sources[sid] for sid in group["source_unit_ids"]]})
    return rehash(receipt)


def projection_fixture(authored, author_packet):
    """Two synthetic member rows make projection order mutations observable."""
    authored, author_packet = copy.deepcopy(authored), copy.deepcopy(author_packet)
    members = author_packet["units"][:2]
    members[1]["event"] = members[0]["event"]
    group = authored["groups"][0]
    group["source_unit_ids"] = sorted(s["unit_id"] for s in members)
    group["id"] = group_identity(group["source_unit_ids"])
    authored["groups"] = [group]
    authored["acquisition_records"] = []
    rehash(authored)
    return authored, packet_for(authored, author_packet)


def test_declared_stronger_projection_preserves_actual_and_canonical_hashes(source_fixture):
    _, _, author_packet, authored = source_fixture
    authored, packet = projection_fixture(authored, author_packet)
    receipt = stronger_receipt(packet)
    before = copy.deepcopy(receipt)
    result = coverage_for(authored, packet, [receipt_for(packet), receipt])
    assert not result["blockers"]
    record = result["records"][0]
    entry = next(e for e in record["matching_history"] if e["receipt_sha256"] == receipt["sha256"])
    assert entry["decision"] == receipt["groups"][0]
    assert entry["reviewed_content_sha256"] == digest(entry["reviewed_content"])
    assert record["packet_content_sha256"] == digest(packet["groups"][0])
    assert entry["packet_content_sha256"] != entry["reviewed_content_sha256"]
    assert receipt == before


@pytest.mark.parametrize("mutation", ["undeclared", "wrong_definition", "ordinary_hash", "member_order",
                                      "source_hash", "source_duplicate", "notes", "top_provenance"])
def test_stronger_adapter_rejects_undeclared_or_changed_projection(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    authored, packet = projection_fixture(authored, author_packet)
    receipt = stronger_receipt(packet)
    group = packet["groups"][0]
    if mutation == "undeclared":
        receipt.pop("reviewed_content_sha256_definition")
    elif mutation == "wrong_definition":
        receipt["reviewed_content_sha256_definition"] += " Additional fields omitted."
    elif mutation == "ordinary_hash":
        receipt["groups"][0]["reviewed_content_sha256"] = digest(group)
    elif mutation == "member_order":
        sources = {s["unit_id"]: s for s in packet["source_units"]}
        receipt["groups"][0]["reviewed_content_sha256"] = digest({"group": group, "source_units":
                                                    [sources[sid] for sid in reversed(group["source_unit_ids"])]})
    elif mutation in ("source_hash", "source_duplicate"):
        if mutation == "source_hash":
            packet["source_units"][0]["sha256"] = "mutated-source-provenance"
        else:
            packet["source_units"].append(copy.deepcopy(packet["source_units"][0]))
        rehash(packet)
        receipt["review_packet_sha256"] = packet["sha256"]
    elif mutation == "notes":
        receipt["groups"][0].pop("notes")
    else:
        receipt["source_author_packet_sha256"] = "different-source-author-packet"
    rehash(receipt)
    with pytest.raises(ValueError, match="content hash|hash definition|Duplicate packet source|retained notes|provenance"):
        coverage_for(authored, packet, [receipt])


def test_unchanged_record_cannot_carry_forward_different_source_provenance(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    final = copy.deepcopy(authored)
    final["groups"][0]["statement"] += " Revised."
    rehash(final)
    newer = packet_for(final, author_packet)
    newer["source_units"] = copy.deepcopy(newer["source_units"])
    newer["source_units"].reverse()
    rehash(newer)
    with pytest.raises(ValueError, match="source/member order/provenance"):
        review_coverage([final], [envelope(packet, "old"), envelope(newer, "new")],
                        [envelope(receipt_for(packet), "receipt")])


def test_stronger_rejection_requires_its_actual_hash_in_supersedes(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = stronger_receipt(packet)
    prior["groups"][0]["status"] = "needs_revision"
    rehash(prior)
    new = stronger_receipt(packet)
    new["supersedes"] = [{"review_receipt_sha256": prior["sha256"], "field": "groups",
                          "id": prior["groups"][0]["id"],
                          "reviewed_content_sha256": prior["groups"][0]["reviewed_content_sha256"]}]
    rehash(new)
    assert not coverage_for(authored, packet, [prior, new, receipt_for(packet)])["blockers"]
    new["supersedes"][0]["reviewed_content_sha256"] = digest(packet["groups"][0])
    rehash(new)
    with pytest.raises(ValueError, match="supersession edge"):
        coverage_for(authored, packet, [prior, new, receipt_for(packet)])


def adjudicating_receipt(packet, prior, *, stronger=False):
    new = stronger_receipt(packet) if stronger else receipt_for(packet, OPUS)
    new["supersedes"] = []
    for field in FIELDS:
        decisions = {r["id"]: r for r in new[field]}
        rows = {r["id"]: r for r in packet[field]}
        for old in prior[field]:
            new["supersedes"].append({"review_receipt_sha256": prior["sha256"], "field": field,
                                      "id": old["id"], "reviewed_content_sha256": old["reviewed_content_sha256"],
                                      "canonical_content_sha256": digest(rows[old["id"]]),
                                      "new_reviewed_content_sha256": decisions[old["id"]]["reviewed_content_sha256"]})
    return rehash(new)


def nest_supersedes(receipt, prior):
    """Move explicit synthetic edges to their exact approved decision rows."""
    rows = {(field, r["id"]): r for field in FIELDS for r in receipt[field]}
    for edge in receipt.pop("supersedes"):
        edge["old_hash_projection"] = ("group_plus_ordered_source_units" if edge["field"] == "groups" and
                                       prior.get("reviewed_content_sha256_definition") == GROUP_SOURCE_HASH_DEFINITION
                                       else "exact_packet_row")
        rows[(edge["field"], edge["id"])].setdefault("supersedes", []).append(edge)
    return rehash(receipt)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("new_stronger", [False, True])
def test_row_local_edges_validate_exact_cross_projection_without_mutating_receipts(source_fixture, reverse, new_stronger):
    _, _, author_packet, authored = source_fixture
    final, old_packet, new_packet, prior, new = cross_projection_fixture(authored, author_packet, new_stronger=new_stronger)
    if not new_stronger:
        new["reviewed_content_sha256_definition"] = development.R4_PACKET_RECORD_HASH_DEFINITION
    nest_supersedes(new, prior)
    receipts = [envelope(r, str(i)) for i, r in enumerate((prior, new, receipt_for(new_packet)))]
    before = copy.deepcopy((prior, new, old_packet, new_packet))
    result = review_coverage([final], [envelope(old_packet, "old"), envelope(new_packet, "new")],
                             list(reversed(receipts)) if reverse else receipts)
    assert not result["blockers"] and len(result["validated_supersedes"]) == len(result["records"])
    for validated in result["validated_supersedes"]:
        edge = {k: v for k, v in validated.items() if k != "receipt_sha256"}
        row = next(r for r in new[edge["field"]] if r["id"] == edge["id"])
        assert edge in row["supersedes"]
    for record in result["records"]:
        prior_entry = next(e for e in record["matching_history"] if e["receipt_sha256"] == prior["sha256"])
        assert prior_entry["decision"]["status"] == "needs_revision"
        assert prior_entry["superseded_by_receipt_sha256"] == [new["sha256"]]
    assert (prior, new, old_packet, new_packet) == before


@pytest.mark.parametrize("mutation", ["field", "id", "attach_other_row", "attach_other_field", "parent_negative",
    "new_hash", "missing_new_hash", "canonical_hash", "missing_canonical_hash", "old_hash", "prior_missing",
    "prior_approved", "projection_wrong", "projection_unknown", "not_list", "not_object",
    "duplicate", "conflicting", "coverage_gap"])
def test_row_local_edges_reject_malformed_misattached_or_conflicting_claims(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, OPUS, status="approved" if mutation == "prior_approved" else "needs_revision")
    new = nest_supersedes(adjudicating_receipt(packet, prior), prior)
    row = new["groups"][0]
    edge = row["supersedes"][0]
    if mutation == "field":
        edge["field"] = "acquisition_records"
    elif mutation == "id":
        edge["id"] = new["groups"][1]["id"]
    elif mutation == "attach_other_row":
        new["groups"][1]["supersedes"].append(row["supersedes"].pop())
    elif mutation == "attach_other_field":
        new["acquisition_records"][0]["supersedes"].append(row["supersedes"].pop())
    elif mutation == "parent_negative":
        row["status"] = "needs_revision"
    elif mutation in ("new_hash", "canonical_hash", "old_hash"):
        key = {"new_hash": "new_reviewed_content_sha256", "canonical_hash": "canonical_content_sha256",
               "old_hash": "reviewed_content_sha256"}[mutation]
        edge[key] = "different actual content"
    elif mutation in ("missing_new_hash", "missing_canonical_hash"):
        edge.pop({"missing_new_hash": "new_reviewed_content_sha256",
                  "missing_canonical_hash": "canonical_content_sha256"}[mutation])
    elif mutation == "prior_missing":
        edge["review_receipt_sha256"] = "missing actual receipt"
    elif mutation == "projection_wrong":
        edge["old_hash_projection"] = "group_plus_ordered_source_units"
    elif mutation == "projection_unknown":
        edge["old_hash_projection"] = "inferred projection"
    elif mutation == "not_list":
        row["supersedes"] = edge
    elif mutation == "not_object":
        row["supersedes"] = ["malformed edge"]
    elif mutation in ("duplicate", "conflicting"):
        new["supersedes"] = [copy.deepcopy(edge)]
        if mutation == "conflicting":
            new["supersedes"][0]["new_reviewed_content_sha256"] = "conflicting new hash"
    elif mutation == "coverage_gap":
        new["coverage_gap"] = True
    rehash(new)
    with pytest.raises(ValueError, match="supersession edge|supersedes must"):
        coverage_for(authored, packet, [prior, new, receipt_for(packet)])


def test_nested_qa_edge_cannot_supersede_changed_parent_group_context(source_fixture):
    _, _, author_packet, authored = source_fixture
    old_packet = packet_for(authored, author_packet)
    prior = receipt_for(old_packet, OPUS, field="acquisition_records", status="needs_revision")
    final = copy.deepcopy(authored)
    final["groups"][0]["statement"] += " Changed dependency context."
    rehash(final)
    new_packet = packet_for(final, author_packet)
    new = nest_supersedes(adjudicating_receipt(new_packet, prior), prior)
    assert old_packet["acquisition_records"] == new_packet["acquisition_records"]
    with pytest.raises(ValueError, match="supersession edge"):
        review_coverage([final], [envelope(old_packet, "old"), envelope(new_packet, "new")],
                        [envelope(prior, "prior"), envelope(new, "new")])


def test_read_disclosure_aliases_cannot_hide_forbidden_inputs(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    new = receipt_for(packet)
    new.pop("model_outcomes_seen")
    new["disclosures"] = {"model_outcomes_read": False, "author_responses_read": False,
                          "evaluation_questions_read": False, "other_review_judgments_read": False}
    rehash(new)
    assert not coverage_for(authored, packet, [new, receipt_for(packet, OPUS)])["blockers"]
    for key in new["disclosures"]:
        altered = copy.deepcopy(new)
        altered["disclosures"][key] = True
        rehash(altered)
        with pytest.raises(ValueError, match="blinded|forbidden"):
            coverage_for(authored, packet, [altered, receipt_for(packet, OPUS)])


@pytest.mark.parametrize("mutation", [None, "extra", "missing", "order", "not_partial"])
def test_partial_receipt_event_scope_matches_exact_decisions(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    ids = [r["id"] for r in packet["groups"][:2]]
    new = receipt_for(packet, OPUS, field="groups", ids=ids)
    events = {r["event"] for r in packet["groups"][:2]}
    new["partial"] = True
    new["events"] = [e for e in packet["events"] if e in events]
    if mutation == "extra":
        new["events"].append("invented event")
    elif mutation == "missing":
        new["events"].pop()
    elif mutation == "order":
        new["events"].reverse()
    elif mutation == "not_partial":
        new["partial"] = False
    rehash(new)
    if mutation:
        with pytest.raises(ValueError, match="Receipt events differ"):
            coverage_for(authored, packet, [new, receipt_for(packet)])
    else:
        result = coverage_for(authored, packet, [new, receipt_for(packet)])
        assert sum(r["complete"] for r in result["records"]) == 2


@pytest.mark.parametrize("mutation", [None, "wrong_hash", "changed_definition"])
def test_preservation_receipt_keeps_declared_auxiliary_projection_and_negatives(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    new = receipt_for(packet, OPUS, field="groups", status="needs_revision")
    new["reviewed_content_sha256_definition"] = development.R5_PACKET_RECORD_HASH_DEFINITION
    for decision, row in zip(new["groups"], packet["groups"]):
        decision["strong_v2_content_sha256"] = digest(development._group_source_projection(packet, row))
    if mutation == "wrong_hash":
        new["groups"][0]["strong_v2_content_sha256"] = "forged auxiliary hash"
    elif mutation == "changed_definition":
        new["reviewed_content_sha256_definition"] += " Undeclared projection."
    rehash(new)
    if mutation:
        with pytest.raises(ValueError, match="stronger auxiliary hash|hash definition"):
            coverage_for(authored, packet, [new, receipt_for(packet), receipt_for(packet, OPUS)])
    else:
        result = coverage_for(authored, packet, [new, receipt_for(packet), receipt_for(packet, OPUS)])
        assert len(result["blockers"]) == len(packet["groups"])
        assert not result["validated_supersedes"]
        assert all(any(e["decision"] == d for e in r["active_decisions"]) for r, d in
                   zip(result["records"][:len(packet["groups"])], new["groups"]))


def test_nested_edges_use_declared_prior_projection_when_edge_label_is_absent(source_fixture):
    _, _, author_packet, authored = source_fixture
    final, old_packet, new_packet, prior, new = cross_projection_fixture(authored, author_packet)
    nest_supersedes(new, prior)
    for field in FIELDS:
        for row in new[field]:
            for edge in row["supersedes"]:
                edge.pop("old_hash_projection")
    rehash(new)
    coverage = review_coverage([final], [envelope(old_packet, "old"), envelope(new_packet, "new")],
        [envelope(r, str(i)) for i, r in enumerate((prior, new, receipt_for(new_packet)))])
    assert not coverage["blockers"] and len(coverage["validated_supersedes"]) == len(coverage["records"])


def test_source_first_prose_cannot_invent_adjudication_independence(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    new = receipt_for(packet)
    new["other_reviewer_judgments_seen"] = True
    new["review_procedure"] = "Independent source-only first pass; afterwards read prior judgments."
    rehash(new)
    with pytest.raises(ValueError, match="forbidden"):
        coverage_for(authored, packet, [new, receipt_for(packet, OPUS)])


def cross_projection_fixture(authored, author_packet, *, new_stronger=False):
    old_packet = packet_for(authored, author_packet)
    final = copy.deepcopy(authored)
    final["author_version"] = "synthetic fresh review version"
    rehash(final)
    new_packet = packet_for(final, author_packet)
    prior = receipt_for(old_packet, OPUS) if new_stronger else stronger_receipt(old_packet)
    for field in FIELDS:
        for row in prior[field]:
            row["status"] = "needs_revision"
    rehash(prior)
    new = adjudicating_receipt(new_packet, prior, stronger=new_stronger)
    canonical = prior if new_stronger else new
    canonical["reviewed_content_sha256_definition"] = PACKET_RECORD_HASH_DEFINITION
    rehash(canonical)
    if new_stronger:
        for edge in new["supersedes"]:
            edge["review_receipt_sha256"] = prior["sha256"]
        rehash(new)
    return final, old_packet, new_packet, prior, new


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("new_stronger", [False, True])
def test_explicit_cross_projection_edges_bind_both_actual_hashes(source_fixture, reverse, new_stronger):
    _, _, author_packet, authored = source_fixture
    final, old_packet, new_packet, prior, new = cross_projection_fixture(authored, author_packet, new_stronger=new_stronger)
    packets = [envelope(old_packet, "prior-packet"), envelope(new_packet, "new-packet")]
    receipts = [envelope(r, str(i)) for i, r in enumerate((prior, new, receipt_for(new_packet)))]
    before = copy.deepcopy((prior, new, old_packet, new_packet))
    result = review_coverage([final], packets, list(reversed(receipts)) if reverse else receipts)
    assert not result["blockers"]
    assert len(result["validated_supersedes"]) == len(result["records"])
    for record in result["records"]:
        rejected = next(e for e in record["matching_history"] if e["receipt_sha256"] == prior["sha256"])
        approved = next(e for e in record["matching_history"] if e["receipt_sha256"] == new["sha256"])
        assert rejected["packet_content"] == approved["packet_content"]
        assert rejected["superseded_by_receipt_sha256"] == [new["sha256"]]
        assert rejected["decision"]["status"] == "needs_revision"
        if record["field"] == "groups":
            assert rejected["hash_scheme"] != approved["hash_scheme"]
            assert rejected["reviewed_content_sha256"] != approved["reviewed_content_sha256"]
    assert (prior, new, old_packet, new_packet) == before


@pytest.mark.parametrize("mutation", ["old_hash", "new_hash", "canonical_hash", "missing_canonical", "missing_new",
                                      "old_scheme", "new_scheme", "forged_definition", "old_approved", "new_negative"])
def test_cross_projection_edges_reject_forged_hashes_schemes_and_status(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    final, old_packet, new_packet, prior, new = cross_projection_fixture(authored, author_packet)
    edge = new["supersedes"][0]
    if mutation in ("old_hash", "new_hash", "canonical_hash"):
        edge[{"old_hash": "reviewed_content_sha256", "new_hash": "new_reviewed_content_sha256",
              "canonical_hash": "canonical_content_sha256"}[mutation]] = "forged"
    elif mutation == "missing_canonical":
        edge.pop("canonical_content_sha256")
    elif mutation == "missing_new":
        edge.pop("new_reviewed_content_sha256")
    elif mutation == "old_scheme":
        prior.pop("reviewed_content_sha256_definition")
        rehash(prior)
        for e in new["supersedes"]:
            e["review_receipt_sha256"] = prior["sha256"]
    elif mutation == "new_scheme":
        new["reviewed_content_sha256_definition"] = GROUP_SOURCE_HASH_DEFINITION
    elif mutation == "forged_definition":
        new["reviewed_content_sha256_definition"] += " Forged projection."
    elif mutation == "old_approved":
        prior["groups"][0]["status"] = "approved"
        rehash(prior)
        for e in new["supersedes"]:
            e["review_receipt_sha256"] = prior["sha256"]
    else:
        new["groups"][0]["status"] = "needs_revision"
    rehash(new)
    with pytest.raises(ValueError, match="supersession edge|content hash"):
        review_coverage([final], [envelope(old_packet, "prior"), envelope(new_packet, "new")],
                        [envelope(prior, "prior"), envelope(new, "new")])


@pytest.mark.parametrize("mutation", ["source_hash", "source_assertion", "gold", "probe", "source_order", "factsheet",
                                      "author_packet", "catalog", "events", "member_order", "payload"])
def test_cross_projection_edges_require_exact_prior_context_and_canonical_row(source_fixture, mutation):
    _, _, author_packet, authored = source_fixture
    authored, old_packet = projection_fixture(authored, author_packet)
    final = copy.deepcopy(authored)
    final["author_version"] = "synthetic fresh review version"
    rehash(final)
    new_packet = copy.deepcopy(old_packet)
    new_packet["authored_chunk_sha256"] = final["sha256"]
    rehash(new_packet)
    if mutation in ("source_hash", "source_assertion", "gold", "probe"):
        old_packet["source_units"] = copy.deepcopy(old_packet["source_units"])
        source = old_packet["source_units"][0]
        key = {"source_hash": "sha256", "source_assertion": "source_assertion", "gold": "canonical_answer_labels",
               "probe": "canonical_probes"}[mutation]
        source[key] = "forged" if mutation in ("source_hash", "source_assertion") else ["forged"]
    elif mutation == "source_order":
        old_packet["source_units"].reverse()
    elif mutation == "factsheet":
        old_packet["factsheets"] = copy.deepcopy(old_packet["factsheets"])
        old_packet["factsheets"][old_packet["events"][0]]["sha256"] = "forged"
    elif mutation in ("author_packet", "catalog"):
        old_packet["source_author_packet_sha256" if mutation == "author_packet" else "source_catalog_sha256"] = "forged"
    elif mutation == "events":
        old_packet["events"] = list(reversed(old_packet["events"]))
    elif mutation == "member_order":
        old_packet["groups"][0]["source_unit_ids"].reverse()
    else:
        old_packet["groups"][0]["statement"] += " Changed retained semantic payload."
    rehash(old_packet)
    prior = stronger_receipt(old_packet)
    prior["groups"][0]["status"] = "needs_revision"
    rehash(prior)
    new = adjudicating_receipt(new_packet, prior)
    with pytest.raises(ValueError, match="source/member order/provenance|source membership/order|supersession edge"):
        review_coverage([final], [envelope(old_packet, "prior"), envelope(new_packet, "new")],
                        [envelope(prior, "prior"), envelope(new, "new")])


def test_cross_projection_approval_without_explicit_edge_keeps_unchanged_rejection(source_fixture):
    _, _, author_packet, authored = source_fixture
    final, old_packet, new_packet, prior, new = cross_projection_fixture(authored, author_packet)
    new.pop("supersedes")
    rehash(new)
    result = review_coverage([final], [envelope(old_packet, "prior"), envelope(new_packet, "new")],
                             [envelope(r, str(i)) for i, r in enumerate((prior, new, receipt_for(new_packet)))])
    assert len(result["blockers"]) == len(result["records"])
    assert result["validated_supersedes"] == []
    assert all(any(e["receipt_sha256"] == prior["sha256"] and not e["superseded_by_receipt_sha256"]
                   for e in r["active_decisions"]) for r in result["records"])


def test_qa_supersession_requires_exact_group_context_even_with_unchanged_canonical_record(source_fixture):
    _, _, author_packet, authored = source_fixture
    old_packet = packet_for(authored, author_packet)
    prior = receipt_for(old_packet, OPUS, field="acquisition_records", status="needs_revision")
    final = copy.deepcopy(authored)
    final["groups"][0]["statement"] += " Changed group context."
    rehash(final)
    new_packet = packet_for(final, author_packet)
    new = adjudicating_receipt(new_packet, prior)
    assert old_packet["acquisition_records"] == new_packet["acquisition_records"]
    with pytest.raises(ValueError, match="supersession edge"):
        review_coverage([final], [envelope(old_packet, "old"), envelope(new_packet, "new")],
                        [envelope(prior, "prior"), envelope(new, "new")])


def test_declared_canonical_hash_on_legacy_edge_must_still_be_exact(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    prior = receipt_for(packet, status="needs_revision")
    new = superseding_receipt(packet, prior)
    new["supersedes"][0]["canonical_content_sha256"] = "forged canonical hash"
    rehash(new)
    with pytest.raises(ValueError, match="supersession edge"):
        coverage_for(authored, packet, [prior, new, receipt_for(packet, OPUS)])


def test_reviewer_dict_cannot_invent_a_second_family(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    forged_family = {"identity": "synthetic-reviewer/gpt-test", "family": "anthropic"}
    with pytest.raises(ValueError, match="family contradicts"):
        coverage_for(authored, packet, [receipt_for(packet, forged_family)])


def test_provider_and_family_labels_do_not_invent_actual_model_family(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    declared = {"identity": "synthetic reviewer", "family": "openai", "provider": "openai"}
    with pytest.raises(ValueError, match="actual named reviewer model family"):
        coverage_for(authored, packet, [receipt_for(packet, declared)])


def test_named_model_subfamily_and_outcomes_seen_schema(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet, {"identity": "synthetic sol reviewer", "family": "sol", "provider": "openai", "model": "gpt-test"})
    receipt["outcomes_seen"] = receipt.pop("model_outcomes_seen")
    rehash(receipt)
    assert not coverage_for(authored, packet, [receipt, receipt_for(packet, OPUS)])["blockers"]


def test_missing_and_false_claimed_coverage_block(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    missing = receipt_for(packet, OPUS)
    missing["groups"].pop()
    missing["coverage"] = {"groups_reviewed": len(packet["groups"])}
    rehash(missing)
    result = coverage_for(authored, packet, [receipt_for(packet), missing])
    assert any(e["status"] == "coverage_gap" for r in result["records"] for e in r["latest_decisions"])


@pytest.mark.parametrize("mutation, message", [
    ("digest", "Artifact digest"), ("packet", "missing retained"), ("hash", "content hash"),
    ("file", "file digest"), ("author", "Source author"), ("blind", "blinded"),
    ("human", "human approval"), ("duplicate", "Duplicate receipt"),
])
def test_receipt_provenance_and_identity_rejections(source_fixture, mutation, message):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    receipt = receipt_for(packet)
    if mutation == "digest":
        receipt["groups"][0]["status"] = "needs_revision"
    elif mutation == "packet":
        receipt["review_packet_sha256"] = "not-retained"
    elif mutation == "hash":
        receipt["groups"][0]["reviewed_content_sha256"] = "stale"
    elif mutation == "file":
        receipt["review_packet_file_sha256"] = "wrong-file"
    elif mutation == "author":
        receipt["reviewer"] = authored["author"]
    elif mutation == "blind":
        receipt["author_rationales_seen"] = True
    elif mutation == "human":
        receipt["human_review_complete"] = True
    else:
        receipt["groups"].append(copy.deepcopy(receipt["groups"][0]))
    if mutation != "digest":
        rehash(receipt)
    with pytest.raises(ValueError, match=message):
        coverage_for(authored, packet, [receipt])


@pytest.mark.parametrize("mutation, message", [
    ("source_loss", "source unit loss"), ("duplicate_group", "Duplicate authored grounded group"),
    ("gold", "gold/target"), ("assertion", "original source/hash"), ("evidence", "exact pinned"),
    ("approval", "claimed approval"), ("conflict", "Source conflict singleton"),
])
def test_original_source_integrity_rejections(source_fixture, mutation, message):
    _, catalog, author_packet, authored = source_fixture
    if mutation == "source_loss":
        authored["source_units"].pop()
    elif mutation == "duplicate_group":
        authored["groups"].append(copy.deepcopy(authored["groups"][0]))
    elif mutation == "gold":
        authored["acquisition_records"][0]["answer"] = "mutated gold"
    elif mutation == "assertion":
        authored["source_units"][0]["source_assertion"] += " mutated"
    elif mutation == "evidence":
        authored["source_units"][0]["evidence"] = [{"line": 0, "quote": "not the pinned excerpt"}]
    elif mutation == "approval":
        authored["source_units"][0]["review_status"] = "approved_development"
    else:
        authored["source_units"][0]["status"] = "preserve_source_conflict"
        authored["source_units"][0]["conflict_rationale"] = "Synthetic source conflict."
        authored["groups"][0]["statement"] += " Rewritten."
    rehash(authored)
    with pytest.raises(ValueError, match=message):
        validate_authored_chunk(authored, author_packet, catalog)


def test_raw_catalog_hash_is_pinned_even_after_valid_reseal(source_fixture):
    _, catalog, _, _ = source_fixture
    catalog["schema"] = "spacing-development-source-catalog-v1"
    rehash(catalog)
    with pytest.raises(ValueError, match="Raw development source catalog hash"):
        validate_catalog(catalog, sealed({"probes": {}}), [], {}, {})
    catalog["sha256"] = CATALOG_SHA256
    with pytest.raises(ValueError, match="Artifact digest"):
        checked(catalog)


def historical_flag_fixture(authored, author_packet):
    history = copy.deepcopy(authored)
    history["source_units"][0]["nonliteral_answer_labels"] = [author_packet["units"][0]["canonical_answer_labels"][0]]
    history["author_version"] = "synthetic historical stale flag version"
    rehash(history)
    final = copy.deepcopy(authored)
    final["author_version"] = "synthetic final corrected flag version"
    rehash(final)
    old_packet, final_packet = packet_for(history, author_packet), packet_for(final, author_packet)
    return history, final, old_packet, final_packet


def validate_history_fixture(catalog, author_packet, history, final, old_packet, final_packet):
    return _validate_retained_authored_versions(
        [envelope(history, "history"), envelope(final, "final")], {author_packet["sha256"]: author_packet}, catalog,
        [final], [envelope(old_packet, "old-packet"), envelope(final_packet, "final-packet")],
        [envelope(receipt_for(old_packet), "old-receipt")])


def test_historical_stale_flag_is_retained_disclosed_and_carries_unchanged_reviews(source_fixture):
    _, catalog, author_packet, authored = source_fixture
    history, final, old_packet, final_packet = historical_flag_fixture(authored, author_packet)
    before = copy.deepcopy((history, final, old_packet, final_packet))
    bound, limitations = validate_history_fixture(catalog, author_packet, history, final, old_packet, final_packet)
    assert bound == {history["sha256"], final["sha256"]}
    assert len(limitations) == 1
    limitation = limitations[0]
    assert limitation["metadata_field"] == "nonliteral_answer_labels"
    assert limitation["retained_value"] == history["source_units"][0]["nonliteral_answer_labels"]
    assert limitation["expected_value"] == final["source_units"][0]["nonliteral_answer_labels"] == []
    assert limitation["authored_chunk_sha256"] == history["sha256"]
    assert limitation["final_authored_chunk_sha256"] == final["sha256"]
    assert limitation["authored_file_sha256"] == envelope(history, "history")["file_sha256"]
    assert limitation["authored_path"] == envelope(history, "history")["path"]
    assert limitation["author"] == history["author"]
    assert old_packet["source_units"] == final_packet["source_units"] == author_packet["units"]
    assert all("nonliteral_answer_labels" not in r for r in old_packet["source_units"])
    receipts = [receipt_for(old_packet), stronger_receipt(old_packet)]
    coverage = review_coverage([final], [envelope(old_packet, "old"), envelope(final_packet, "new")],
                               [envelope(r, str(i)) for i, r in enumerate(receipts)])
    assert not coverage["blockers"]
    assert all(len(r["matching_history"]) == 2 for r in coverage["records"])
    assert (history, final, old_packet, final_packet) == before


def test_retained_flag_only_bridge_remains_valid_after_separate_final_qa_edit(source_fixture):
    _, catalog, author_packet, authored = source_fixture
    history, bridge, old_packet, bridge_packet = historical_flag_fixture(authored, author_packet)
    final = copy.deepcopy(bridge)
    final["acquisition_records"][0]["questions"][0] += " Separate later QA edit?"
    rehash(final)
    final_packet = packet_for(final, author_packet)
    files = [envelope(history, "history"), envelope(bridge, "flag-bridge"), envelope(final, "final")]
    packets = [envelope(old_packet, "old"), envelope(bridge_packet, "bridge"), envelope(final_packet, "final")]
    reviews = [envelope(receipt_for(old_packet, status="needs_revision"), "old-negative"),
               envelope(receipt_for(final_packet), "new-sol"), envelope(receipt_for(final_packet, OPUS), "new-opus")]
    before = copy.deepcopy((files, packets, reviews))
    _, limitations = _validate_retained_authored_versions(files, {author_packet["sha256"]: author_packet}, catalog,
                                                        [final], packets, reviews)
    assert limitations[0]["flag_bridge_authored_chunk_sha256"] == bridge["sha256"]
    assert limitations[0]["flag_bridge_file_sha256"] == files[1]["file_sha256"]
    assert limitations[0]["final_authored_chunk_sha256"] == final["sha256"]
    coverage = review_coverage([final], packets, reviews)
    assert all(not r["complete"] for r in coverage["records"] if r["field"] == "groups")
    changed = next(r for r in coverage["records"] if r["id"] == final["acquisition_records"][0]["id"])
    assert changed["complete"] and len(changed["matching_history"]) == 2
    assert (files, packets, reviews) == before
    with pytest.raises(ValueError, match="retained valid flag-only bridge"):
        _validate_retained_authored_versions([files[0], files[2]], {author_packet["sha256"]: author_packet}, catalog,
                                            [final], [packets[0], packets[2]], reviews)
    final["source_units"][0]["nonliteral_answer_labels"] = ["incorrect current flag"]
    rehash(final)
    with pytest.raises(ValueError, match="Nonliteral source target flags"):
        validate_authored_chunk(final, author_packet, catalog, final=False)


def test_exact_correction_schema_is_retained_without_full_chunk_validation(source_fixture):
    _, catalog, author_packet, authored = source_fixture
    correction = sealed({"schema": "spacing-source-author-correction-response-v1", "authored_chunk_sha256": authored["sha256"]})
    files = [envelope(authored, "authored"), envelope(correction, "correction")]
    before = copy.deepcopy(files)
    assert _authored_history(files) == [files[0]]
    packet = packet_for(authored, author_packet)
    bound, _ = _validate_retained_authored_versions(files, {author_packet["sha256"]: author_packet}, catalog,
                                                   [authored], [envelope(packet, "packet")], [])
    assert bound == {authored["sha256"]} and files == before
    correction["schema"] = "lookalike-correction-schema"
    rehash(correction)
    with pytest.raises(ValueError, match="Unknown retained authored artifact schema"):
        _authored_history([files[0], envelope(correction, "correction")])


@pytest.mark.parametrize("final_mode", [True, False])
def test_stale_flag_without_explicit_valid_final_counterpart_is_rejected(source_fixture, final_mode):
    _, catalog, author_packet, authored = source_fixture
    history, final, _, _ = historical_flag_fixture(authored, author_packet)
    with pytest.raises(ValueError, match="Nonliteral source target flags"):
        validate_authored_chunk(history, author_packet, catalog, final=final_mode)
    if final_mode:
        with pytest.raises(ValueError, match="Nonliteral source target flags"):
            validate_authored_chunk(history, author_packet, catalog, final=True, final_chunk=final)


@pytest.mark.parametrize("mutation,message", [
    ("final_flag", "Nonliteral source target flags"), ("statement", "unchanged semantic"),
    ("questions", "unchanged semantic"), ("assertion", "original source/hash"),
    ("source_hash", "original source/hash"), ("probe", "original source/hash"),
    ("gold", "gold/target"), ("membership", "membership/identity"), ("evidence", "exact pinned"),
    ("source_status", "unchanged semantic"), ("history_digest", "Artifact digest"),
    ("final_digest", "Artifact digest"),
])
def test_historical_flag_exception_cannot_mask_semantic_source_or_digest_corruption(source_fixture, mutation, message):
    _, catalog, author_packet, authored = source_fixture
    history, final, _, _ = historical_flag_fixture(authored, author_packet)
    if mutation == "final_flag":
        final["source_units"][0]["nonliteral_answer_labels"] = ["forged stale flag"]
    elif mutation == "statement":
        final["groups"][0]["statement"] += " Changed semantic assertion."
    elif mutation == "questions":
        final["acquisition_records"][0]["questions"][0] += " Changed wording."
    elif mutation in ("assertion", "source_hash", "probe", "source_status"):
        key = {"assertion": "source_assertion", "source_hash": "source_unit_sha256", "probe": "opaque_probe_ids",
               "source_status": "status"}[mutation]
        history["source_units"][0][key] = ["forged probe"] if mutation == "probe" else (
            "preserve_source_conflict" if mutation == "source_status" else "forged")
    elif mutation == "gold":
        history["acquisition_records"][0]["answer"] = "forged gold"
    elif mutation == "membership":
        history["groups"][0]["source_unit_ids"] = [history["source_units"][1]["unit_id"]]
    elif mutation == "evidence":
        history["source_units"][0]["evidence"][0]["quote"] = "forged evidence"
    elif mutation == "history_digest":
        history["sha256"] = "forged"
    else:
        final["sha256"] = "forged"
    if mutation != "history_digest":
        rehash(history)
    if mutation != "final_digest":
        rehash(final)
    with pytest.raises(ValueError, match=message):
        validate_authored_chunk(history, author_packet, catalog, final=False, final_chunk=final)


@pytest.mark.parametrize("mutation", ["derived_flag", "assertion", "gold", "probes", "order", "factsheet"])
def test_historical_flag_exception_requires_exact_original_raw_packet_context(source_fixture, mutation):
    _, catalog, author_packet, authored = source_fixture
    history, final, old_packet, final_packet = historical_flag_fixture(authored, author_packet)
    old_packet["source_units"] = copy.deepcopy(old_packet["source_units"])
    source = old_packet["source_units"][0]
    if mutation == "derived_flag":
        source["nonliteral_answer_labels"] = history["source_units"][0]["nonliteral_answer_labels"]
    elif mutation in ("assertion", "gold", "probes"):
        key = {"assertion": "source_assertion", "gold": "canonical_answer_labels", "probes": "canonical_probes"}[mutation]
        source[key] = "forged assertion" if mutation == "assertion" else ["forged"]
    elif mutation == "order":
        old_packet["source_units"].reverse()
    else:
        old_packet["factsheets"] = copy.deepcopy(old_packet["factsheets"])
        old_packet["factsheets"][old_packet["events"][0]]["sha256"] = "forged"
    rehash(old_packet)
    with pytest.raises(ValueError, match="Review original source/gold/factsheet"):
        validate_history_fixture(catalog, author_packet, history, final, old_packet, final_packet)


def test_historical_flag_exception_preserves_negative_reviews_and_final_raw_singleton_guard(source_fixture):
    _, catalog, author_packet, authored = source_fixture
    authored["source_units"][0]["status"] = "preserve_source_conflict"
    authored["source_units"][0]["conflict_rationale"] = "Synthetic retained source conflict."
    authored["groups"][0]["statement"] += " Rewritten conflict."
    rehash(authored)
    history, final, old_packet, final_packet = historical_flag_fixture(authored, author_packet)
    _, limitations = validate_history_fixture(catalog, author_packet, history, final, old_packet, final_packet)
    assert limitations and authored_content_blockers([final])
    with pytest.raises(ValueError, match="Source conflict singleton"):
        validate_authored_chunk(final, author_packet, catalog)
    receipts = [receipt_for(old_packet, status="needs_revision"), receipt_for(final_packet), receipt_for(final_packet, OPUS)]
    coverage = review_coverage([final], [envelope(old_packet, "old"), envelope(final_packet, "new")],
                               [envelope(r, str(i)) for i, r in enumerate(receipts)])
    assert len(coverage["blockers"]) == len(coverage["records"])
    assert not coverage["validated_supersedes"]


def test_review_packet_rejects_changed_gold_and_rationales(source_fixture):
    _, _, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    # Supply the real stripped-source shape expected by this validator.
    packet["source_units"] = [{k: u.get(k) for k in ("unit_id", "event", "source_assertion", "canonical_answer_labels",
                                                    "canonical_probes", "sha256")} for u in author_packet["units"]]
    rehash(packet)
    validate_review_packet(packet, authored, author_packet)
    packet["acquisition_records"][0]["answer"] = "mutated gold"
    rehash(packet)
    with pytest.raises(ValueError, match="Stripped review content"):
        validate_review_packet(packet, authored, author_packet)


def test_original_three_schemas_preserve_all_probes_and_bound_targets(source_fixture):
    facts, catalog, author_packet, authored = source_fixture
    before = copy.deepcopy(facts)
    validate_authored_chunk(authored, author_packet, catalog)
    packet = packet_for(authored, author_packet)
    coverage = coverage_for(authored, packet, [receipt_for(packet), receipt_for(packet, OPUS)])
    roles = {f["event"]: f["role"] for f in facts}
    grounding, source_input, qa = derive_bundles(facts, {"sha256": "synthetic-partition"}, roles, catalog, [authored],
                                               coverage, "synthetic-audit", lambda text: len(text.split()))
    assert facts == before
    assert grounding["original_evaluation_probe_count"] == len(facts)
    assert len(grounding["source_units"]) == 3  # Both old source units and the new source unit.
    assert sum(len(s["probe_ids"]) for s in grounding["source_units"]) == 4
    assert all(s["role"] in ("old", "new") for s in grounding["source_units"])
    assert len(qa["records"]) == len(source_input["units"]) == 2
    for bundle in (grounding, source_input, qa):
        checked(bundle)
        assert bundle["source_catalog_sha256"] == catalog["sha256"]
        assert bundle["review_provenance"]["audit_sha256"] == "synthetic-audit"
        assert bundle["human_review_complete"] is False
    assert qa["grounding_bundle_sha256"] == source_input["grounding_bundle_sha256"] == grounding["sha256"]
    assert qa["source_input_sha256"] == source_input["sha256"]
    originals = {r["id"]: r for r in authored["acquisition_records"]}
    assert all(review_content(r, "acquisition_records") == review_content(originals[r["id"]], "acquisition_records") for r in qa["records"])


def test_blocked_coverage_cannot_issue_approved_schemas(source_fixture):
    facts, catalog, author_packet, authored = source_fixture
    packet = packet_for(authored, author_packet)
    coverage = coverage_for(authored, packet, [receipt_for(packet)])
    with pytest.raises(ValueError, match="blocked actual reviews"):
        derive_bundles(facts, {}, {}, catalog, [authored], coverage, "synthetic-audit", lambda text: 1)


@pytest.mark.parametrize("mutation", ["rewrite", "rationale", "merge"])
def test_source_conflict_guard_remains_blocking_even_with_full_reviews(source_fixture, mutation):
    facts, catalog, author_packet, authored = source_fixture
    source = authored["source_units"][0]
    source["status"] = "preserve_source_ambiguity"
    source["conflict_rationale"] = "Synthetic retained source ambiguity."
    group = next(g for g in authored["groups"] if g["id"] == source["group_id"])
    if mutation == "rewrite":
        group["statement"] += " Rewritten."
    elif mutation == "rationale":
        source["conflict_rationale"] = ""
    else:
        group["source_unit_ids"].append(authored["source_units"][1]["unit_id"])
    rehash(authored)
    blockers = authored_content_blockers([authored])
    assert blockers[0]["source_unit_id"] == source["unit_id"]
    if mutation != "merge":
        # Historical versions remain valid unreviewed inputs; final versions
        # still fail the exact same strict singleton policy.
        validate_authored_chunk(authored, author_packet, catalog, final=False)
        with pytest.raises(ValueError, match="Source conflict singleton"):
            validate_authored_chunk(authored, author_packet, catalog)
    packet = packet_for(authored, author_packet)
    coverage = coverage_for(authored, packet, [receipt_for(packet), receipt_for(packet, OPUS)])
    assert not coverage["blockers"]
    with pytest.raises(ValueError, match="blocked authored content"):
        derive_bundles(facts, {}, {}, catalog, [authored], coverage, "synthetic-audit", lambda text: 1)


def test_current_validator_duplicate_forms_are_reported_without_repair(source_fixture):
    facts, catalog, _, authored = source_fixture
    roles = {f["event"]: f["role"] for f in facts}
    old = [r for r in authored["acquisition_records"] if roles[r["event"]] == "old"]
    old[1]["questions"] = list(old[0]["questions"])
    before = copy.deepcopy(authored)
    result = training_content_blockers([authored], facts, roles, catalog)
    assert len(result) == 2 and all("duplicate" in r["reason"] for r in result)
    assert authored == before


def test_no_overwrite_and_no_omitted_retained_versions(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError, match="never overwrite"):
        build_reviewed_development(tmp_path, [], [], [], tmp_path, tmp_path / "config", output)
    assert sentinel.read_text() == "unchanged"
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    (reviews / "old-blocking-receipt.json").write_text("{}")
    with pytest.raises(ValueError, match="Retain all reviews"):
        _retained_paths(tmp_path, "reviews", [])


def test_frozen_config_rejects_role_selection_changes(source_fixture, tmp_path):
    _, catalog, _, _ = source_fixture
    for name, payload in (("source-catalog.json", catalog), ("private-probe-id-map.json", sealed({})),
                          ("author-chunk-index.json", sealed({})), ("development-scale-role-plan.json", sealed({}))):
        write_json(tmp_path / name, payload)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs/scaled-grid-01.json").read_text())
    config["development_fixed_qa_events"] = ["invented-event"] * 3
    write_json(tmp_path / "changed-config.json", config)
    with pytest.raises(ValueError, match="Frozen scaled-grid-01 configuration hash"):
        build_reviewed_development(tmp_path, [], [], [], tmp_path, tmp_path / "changed-config.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reviewer_dictionary_uses_explicit_identity():
    assert person_identity({"identity": "Named test reviewer", "harness": "test"}) == "named test reviewer"
    with pytest.raises(ValueError, match="explicit author/reviewer identity"):
        person_identity({"role": "independent reviewer"})


@pytest.fixture
def real_source_candidate():
    """Read only actual source artifacts, never eval parquet or model outcomes."""
    root = Path("/tmp/p4-development-scale-source-catalog-20261008")
    final_paths = [root / "authored/chunk-01-v7.json", root / "authored/chunk-02-v7.json"]
    if not all(p.exists() for p in final_paths):
        pytest.skip("Exact chunk-01-v7/chunk-02-v7 source candidate is not available")
    catalog = artifact(root / "source-catalog.json")["payload"]
    assert catalog["sha256"] == CATALOG_SHA256
    index = artifact(root / "author-chunk-index.json")["payload"]
    author_packets = [artifact(root / row["path"])["payload"] for row in index["chunks"]]
    histories = [artifact(p) for p in sorted((root / "authored").glob("*.json"))]
    final = [artifact(p)["payload"] for p in final_paths]
    packets = [artifact(p) for p in sorted((root / "review-packets").glob("*.json"))]
    receipts = [artifact(p) for p in sorted((root / "reviews").glob("*.json"))]
    return catalog, {p["sha256"]: p for p in author_packets}, histories, final, packets, receipts


def test_real_historical_flag_bridge_and_current_flags_are_independently_valid(real_source_candidate):
    catalog, authors, histories, final, packets, receipts = real_source_candidate
    before = copy.deepcopy((histories, final, packets, receipts))
    _, limitations = _validate_retained_authored_versions(histories, authors, catalog, final, packets, receipts)
    for chunk in final:
        validate_authored_chunk(chunk, authors[chunk["source_author_packet_sha256"]], catalog, final=False)
    assert not authored_content_blockers(final)
    for limitation in limitations:
        retained = next(a for a in histories if a["payload"]["sha256"] == limitation["authored_chunk_sha256"])
        bridge = next(a for a in histories if a["payload"]["sha256"] == limitation["flag_bridge_authored_chunk_sha256"])
        assert retained["file_sha256"] == limitation["authored_file_sha256"]
        assert bridge["file_sha256"] == limitation["flag_bridge_file_sha256"]
        assert retained["payload"]["author"] == limitation["author"]
        source = next(s for s in retained["payload"]["source_units"] if s["unit_id"] == limitation["id"])
        assert source["nonliteral_answer_labels"] == limitation["retained_value"]
        if limitation["authored_chunk_sha256"] == development.SOURCE_REPAIR_SHA256:
            assert Path(bridge["path"]).name == "chunk-02-v4.json"
            assert Path(retained["path"]).name == "chunk-02-v3.json"
    assert (histories, final, packets, receipts) == before


def test_real_scientific_judgments_remain_exact_bound_raw_artifacts(real_source_candidate):
    _, _, histories, final, packets, receipts = real_source_candidate
    by_hash = {p["payload"]["sha256"]: p for p in packets}
    negative = []
    for retained in receipts:
        receipt = retained["payload"]
        packet_hash = development._receipt_packet_hash(receipt)
        packet = by_hash[packet_hash]["payload"]
        if "review_packet_embedded_sha256" in receipt:
            assert receipt["review_packet_sha256"] in (packet_hash, by_hash[packet_hash]["file_sha256"])
        for field in FIELDS:
            records = {r["id"]: r for r in packet[field]}
            for decision in development._receipt_decisions(receipt, field):
                row = records[decision["id"]]
                projection = development._group_source_projection(packet, row) if field == "groups" and receipt.get(
                    "reviewed_content_sha256_definition") == GROUP_SOURCE_HASH_DEFINITION else row
                assert decision["notes"] and development._decision_hash(decision) == digest(projection)
                if development._decision_status(decision) in development.NEGATIVE:
                    negative.append((retained["path"], field, decision))
    # Historical negative scientific judgments survive all revisions, including
    # when later actual receipts explicitly supersede them.
    assert negative and any(field == "groups" for _, field, _ in negative)
    assert any(field == "acquisition_records" for _, field, _ in negative)
    assert any(a["payload"].get("schema") == "spacing-source-author-correction-response-v1" for a in histories)
    assert final[0]["sha256"] == "549bfe9bd086a507ad983e83deb5d1c6914c1a71a6dc90ed5a1a680931f0a815"
    assert final[1]["sha256"] == "9596bcffaefe08f85ee873c4de8898eb993b94121278a5599e06d8ebfe1a59d5"


def test_real_generic_author_and_reviewer_bind_to_distinct_actual_t3_channels(real_source_candidate):
    _, _, histories, _, _, receipts = real_source_candidate
    path = Path("/tmp/p4-development-actual-task-evidence-v3-20261008.json")
    if not path.exists():
        pytest.skip("Actual completed task inventory is unavailable")
    evidence = artifact(path, sealed_payload=False)
    author_file = next(a for a in histories if a["payload"]["sha256"] == development.SOURCE_REPAIR_SHA256)
    review_file = next(r for r in receipts if r["payload"]["sha256"] == development.SOL_REVIEW_SHA256)
    author = _delegated_binding(author_file["payload"]["author"], author_file, evidence["payload"],
                                development.SOURCE_REPAIR_REQUEST, development.SOURCE_REPAIR_SHA256)
    reviewer = _delegated_binding(review_file["payload"]["reviewer"], review_file, evidence["payload"],
                                  development.SOL_REVIEW_REQUEST, development.SOL_REVIEW_SHA256)
    assert author["identity"]["agent_id"] == reviewer["identity"]["agent_id"] == "/root"
    assert author["channel"] != reviewer["channel"]
    assert all(author["task_record"][k] != reviewer["task_record"][k] for k in ("taskId", "childThreadId", "childRunId"))
    assert any(development._decision_status(row) in development.NEGATIVE for field in FIELDS
               for row in development._receipt_decisions(review_file["payload"], field))


def test_real_final_candidate_retains_actual_receipts_without_evaluation_reads(real_source_candidate):
    _, _, histories, final, packets, receipts = real_source_candidate
    evidence_path = Path("/tmp/p4-development-actual-task-evidence-v3-20261008.json")
    if not evidence_path.exists():
        pytest.skip("Launch blocker: missing actual completed task inventory")
    evidence = artifact(evidence_path, sealed_payload=False)
    native_rows = list(evidence["payload"].get("native_completed_authors", []))
    if evidence["payload"].get("native_completed_author"):
        native_rows.append(evidence["payload"]["native_completed_author"])
    required = [a for a in histories if Path(a["path"]).name in (
        "chunk-01-v7.json", "chunk-02-v4.json", "chunk-02-v5.json", "chunk-02-v6.json", "chunk-02-v7.json")]
    missing = [a["path"] for a in required if not any(
        row.get("artifact_sha256", row.get("actual_result", {}).get("authored_sha256")) == a["payload"]["sha256"]
        for row in native_rows)]
    if missing:
        pytest.skip("Launch blocker: missing actual native completion evidence for " + ", ".join(missing))
    before = copy.deepcopy((histories, final, packets, receipts, evidence))
    coverage = review_coverage(final, packets, receipts, authored_artifacts=histories, task_evidence=evidence)
    actual = {r["payload"]["sha256"]: r["payload"] for r in receipts}
    for record in coverage["records"]:
        for entry in record["matching_history"]:
            assert entry["decision"] is not None and entry["decision"]["notes"]
            assert development._decision_hash(entry["decision"]) == entry["reviewed_content_sha256"]
            assert digest(entry["reviewed_content"]) == entry["reviewed_content_sha256"]
            assert digest(entry["packet_content"]) == entry["packet_content_sha256"]
            assert entry["receipt_sha256"] in actual
            assert entry["reviewer_raw_identity"] == actual[entry["receipt_sha256"]]["reviewer"]
            if entry["status"] in development.NEGATIVE and not entry["superseded_by_receipt_sha256"]:
                assert not record["complete"] and entry in record["active_decisions"]
    for validated in coverage["validated_supersedes"]:
        edge = {k: v for k, v in validated.items() if k != "receipt_sha256"}
        assert edge in development._receipt_supersedes(actual[validated["receipt_sha256"]])
    r4 = next(r["payload"] for r in receipts if Path(r["path"]).name == "chunk-02-v4-opus-adjudication-r4.json")
    raw_r4_edges = [e for field in FIELDS for row in development._receipt_decisions(r4, field)
                    for e in row.get("supersedes", [])]
    assert len(raw_r4_edges) == 81
    assert len([e for e in coverage["validated_supersedes"] if e["receipt_sha256"] == r4["sha256"]]) == 81
    assert len(coverage["validated_supersedes"]) == 99
    r5 = next(r["payload"] for r in receipts if Path(r["path"]).name == "chunk-02-v6-opus-preservation-r5.json")
    blocked_ids = {(field, row["id"]) for field in FIELDS for row in development._receipt_decisions(r5, field)}
    assert len(blocked_ids) == 12
    assert {(r["field"], r["id"]) for r in coverage["blockers"]} == blocked_ids
    for record in coverage["records"]:
        if (record["field"], record["id"]) in blocked_ids:
            judgment = next(e for e in record["active_decisions"] if e["receipt_sha256"] == r5["sha256"])
            assert judgment["status"] == "needs_revision" and not judgment["superseded_by_receipt_sha256"]
    assert sum(e["status"] in development.NEGATIVE for r in coverage["records"]
               for e in r["active_decisions"]) == 54
    bindings = coverage["author_task_provenance"]
    native = [b for b in bindings if b["channel"].startswith("native:")]
    assert {Path(b["artifact_path"]).name for b in native} == {
        "chunk-01-v7.json", "chunk-02-v4.json", "chunk-02-v5.json", "chunk-02-v6.json", "chunk-02-v7.json"}
    for binding in native:
        assert binding["channel"] == "native:/root/blind_acquisition_author"
        assert binding["task_record"]["canonical_agent_name"] == binding["identity"]["agent_id"]
        if Path(binding["artifact_path"]).name.startswith("chunk-02"):
            assert "retrospective" in binding["task_record"]["result_source"]
            assert "not a quotation of earlier completion messages" in binding["task_record"]["observed_result_text"]
    assert coverage["independent_agent_review_complete"] == (not coverage["blockers"])
    assert (histories, final, packets, receipts, evidence) == before


def test_artifact_preserves_named_file_and_byte_digest(tmp_path):
    path = tmp_path / "synthetic-test-artifact.json"
    payload = sealed({"test_only": True})
    write_json(path, payload)
    retained = artifact(path)
    assert retained["path"] == str(path.resolve()) and retained["payload"] == payload
    assert retained["file_sha256"] != payload["sha256"]
