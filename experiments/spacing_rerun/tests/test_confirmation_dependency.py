import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from spacing_rerun.common import digest
from spacing_rerun.confirmation_dependency import (
    acquisition_form_mappings, bind_canonical_records, complete_contrast_report, contrast_report,
    diagnostic_scores, divergence,
    influence_scores, membership_proposal, paired_summary, score, teaching_form_mappings,
    build_final_policy_input, validate_final_policy, freeze_final_policy, verify_frozen_policy,
    FINAL_REVIEW_SCHEMA,
)
from spacing_rerun.evaluation import aggregate


def seal(value):
    value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    return value


def source_fixture():
    sources = [{"unit_id": f"u{i}", "event": "e1" if i < 3 else "e2",
                "source_assertion": f"assertion{i}", "canonical_answer_labels": [f"gold{i}"],
                "canonical_probes": [{"opaque_probe_id": f"p{i}",
                                      "opaque_duplicate_cluster_members": [f"p{i}", f"p{i}-copy"]}],
                "sha256": f"source{i}"} for i in range(4)]
    groups = [{"id": f"g{i}", "event": s["event"], "statement": s["source_assertion"],
               "source_unit_ids": [s["unit_id"]]} for i, s in enumerate(sources)]
    packet = seal({"chunk_index": 1, "authored_chunk_sha256": "authored", "author_rationales_included": False,
                   "groups": groups, "source_units": sources,
                   "source_preservation_protocol": {"source_dispositions": [
                       {"unit_id": "u0", "status": "preserve_source_ambiguity"}]}})
    recs = [{"id": "r01", "event": "e1", "records": [{"group": groups[i], "sources": [sources[i]]}
                                                           for i in [0, 1]]},
            {"id": "r12", "event": "e1", "records": [{"group": groups[i], "sources": [sources[i]]}
                                                           for i in [1, 2]]}]
    return packet, recs


def scoring_fixture():
    mapping, rows = {}, []
    # Seven scored duplicate members must not outweigh a separate root in g1.
    for i in range(7):
        key = f"a{i}"
        rows.append({"id": key, "unit_id": "g1", "event": "e", "role": "old", "exact_match": 1, "loss": 2})
        mapping[key] = {"event": "e", "source_unit_ids": ["u1"], "source_group_id": "g1",
                        "dependency_block_id": "b12", "duplicate_cluster_id": "c1"}
    for key, gid, sid, block, value in [("z", "g1", "u1", "b12", 0),
                                       ("q", "g2", "u2", "b12", 1),
                                       ("r", "g3", "u3", "b3", 0)]:
        rows.append({"id": key, "unit_id": gid, "event": "e", "role": "old",
                     "exact_match": value, "loss": 2 * value})
        mapping[key] = {"event": "e", "source_unit_ids": [sid], "source_group_id": gid,
                        "dependency_block_id": block, "duplicate_cluster_id": "cluster/" + key}
    return rows, mapping


def actual_evidence_fixture(receipt, name):
    """Synthetic completed-task evidence. Never certifies any actual data."""
    text = json.dumps(receipt, ensure_ascii=False)
    status = {"taskId": "synthetic/task/" + name, "childThreadId": "synthetic/thread/" + name,
              "childRunId": "synthetic/run/" + name, "status": "completed", "latestTerminalStatus": "completed",
              "hasPendingChildRuns": False, "model": "fixture-model",
              "latestTerminalSummary": "Synthetic test output " + receipt["sha256"]}
    snapshot = json.dumps({"task_statuses": [{"clientRequestId": name, "result": status}]})
    return {"receipt_sha256": receipt["sha256"], "client_request_id": name, "task_status": status,
            "receipt_file": {"utf8": text, "file_sha256": hashlib.sha256(text.encode()).hexdigest()},
            "task_status_file": {"utf8": snapshot, "file_sha256": hashlib.sha256(snapshot.encode()).hexdigest()}}


def final_fixture(*, rejection=False):
    from experiments.spacing_rerun.tests.test_confirmation import confirmation_fixture
    from spacing_rerun.confirmation import source_preservation_context
    _, audit = confirmation_fixture()
    base, chunks, source_evidence = audit["source_chunks"][0], [], []
    for index in range(8):
        chunk = copy.deepcopy(base)
        events = base["author_packet"]["events"][index * 10:(index + 1) * 10]
        for key in ("author_packet", "authored", "review_packet"):
            chunk[key]["chunk_index"] = index + 1
            chunk[key]["events"] = events
        chunk["author_packet"]["units"] = [u for u in base["author_packet"]["units"] if u["event"] in events]
        chunk["author_packet"]["factsheets"] = {e: base["author_packet"]["factsheets"][e] for e in events}
        seal(chunk["author_packet"])
        chunk["authored"]["source_author_packet_sha256"] = chunk["author_packet"]["sha256"]
        for key in ("source_units", "groups", "acquisition_records"):
            chunk["authored"][key] = [r for r in base["authored"][key] if r["event"] in events]
            chunk["review_packet"][key] = [r for r in base["review_packet"][key] if r["event"] in events]
        seal(chunk["authored"])
        chunk["review_packet"].update(authored_chunk_sha256=chunk["authored"]["sha256"],
                                      source_author_packet_sha256=chunk["author_packet"]["sha256"],
                                      factsheets=chunk["author_packet"]["factsheets"],
                                      source_preservation_protocol=source_preservation_context(chunk["authored"]))
        seal(chunk["review_packet"])
        receipt = chunk["reviews"][0]
        reviewer = "fixture-source-reviewer-" + str(index + 1)
        receipt["reviewer"]["agent_id"] = reviewer
        receipt["review_packet_sha256"] = chunk["review_packet"]["sha256"]
        for key in ("groups", "acquisition_records"):
            ids = {r["id"] for r in chunk["review_packet"][key]}
            receipt[key] = [r for r in receipt[key] if r["id"] in ids]
        if rejection and index == 0:
            receipt["acquisition_records"][0]["status"] = "needs_revision"
        if index == 0:
            receipt["acquisition_records"][0]["conditions"] = "Synthetic dependence annotation and whole-unit flag mask required."
        seal(receipt)
        chunks.append(chunk)
        source_evidence.append(actual_evidence_fixture(receipt, reviewer))
    first = chunks[0]["review_packet"]
    rec = {"id": "synthetic/singleton-dependence-annotation", "event": first["events"][0],
           "records": [{"group": first["groups"][0], "sources": [first["source_units"][0]]}]}
    prepared = []
    for c in chunks:
        for source in c["review_packet"]["source_units"]:
            gid = next(g["id"] for g in c["review_packet"]["groups"] if source["unit_id"] in g["source_unit_ids"])
            prepared.append({"id": "actual/" + source["unit_id"],
                             "opaque_probe_id": source["canonical_probes"][0]["opaque_probe_id"],
                             "event": source["event"], "role": "old", "unit_id": gid})
    condition = {"id": "synthetic/conditional-source-approval", "source_receipt_sha256": chunks[0]["reviews"][0]["sha256"],
                 "acquisition_record_ids": [first["acquisition_records"][0]["id"]],
                 "dependence_recommendation_ids": [rec["id"]],
                 "flagged_source_unit_ids": [first["source_units"][0]["unit_id"]],
                 "notes": "Synthetic retained content requires a dependence annotation and flagged mask."}
    provenance = seal({"schema": "spacing-confirmation-training-source-provenance-v1", "manifest_sha256": "a" * 64,
                       "examples_sha256": "b" * 64, "qa_answer_tokens_exclude_eos_and_filler": True,
                       "generic_source_attribution": "empty", "outcome_based_filters": False,
                       "records": {"old/" + g["id"]: {"source_unit_ids": sorted(g["source_unit_ids"]),
                           "source_group_id": g["id"], "kind": "declaration", "canonical_probe_id": None,
                           "input_tokens": 3, "answer_target_tokens": 0, "loss_tokens": 2,
                           "content_sha256": digest(g["id"])} for c in chunks for g in c["review_packet"]["groups"]}})
    policy = build_final_policy_input(chunks, audit["source_catalog"], [rec], condition["flagged_source_unit_ids"],
                                     [{"replicate_id": "r0", "manifest_sha256": "a" * 64, "records": prepared,
                                       "training_provenance": provenance}],
                                     source_evidence, policy_authors=[{"agent_id": "fixture-policy-author", "provider": "synthetic", "model": "fixture-model"}],
                                     required_source_conditions=[] if rejection else [condition])
    return policy, chunks, audit["source_catalog"], source_evidence


def final_review_fixture(policy):
    membership = policy["membership"]
    def decisions(rows):
        return [{"id": r["id"], "status": "approved", "reviewed_content_sha256": digest(r),
                 "notes": "Synthetic final review, never actual-data approval."} for r in rows]
    receipt = seal({"schema": FINAL_REVIEW_SCHEMA, "policy_sha256": policy["sha256"],
                    "algorithm_sha256": policy["algorithm_sha256"], "membership_sha256": membership["sha256"],
                    "source_bindings_sha256": digest(policy["source_bindings"]),
                    "canonical_bindings_sha256": digest(policy["canonical_bindings"]),
                    "source_review_evidence_sha256": digest(policy["source_review_evidence_bindings"]),
                    "lane_reporting_policy_sha256": policy["lane_reporting_policy"]["sha256"],
                    "exception_protocol_sha256": policy["exception_protocol"]["sha256"] if policy["exception_protocol"] else None,
                    "protocol_coverage_sha256": policy["protocol_coverage"]["sha256"] if policy["protocol_coverage"] else None,
                    "exception_lanes_sha256": digest(policy["exception_lanes"]),
                    "lane_property_review_bindings_sha256": digest(policy["lane_property_review_bindings"]),
                    "lane_reporting_policy_approved": True,
                    "status": "approved", "membership_status": "final_approved", "unresolved_items": [],
                    "all_source_conditions_captured": True,
                    "model_outcomes_seen": False, "evaluation_wording_seen": False, "author_rationales_seen": False,
                    "human_review_complete": False,
                    "reviewer": {"agent_id": "fixture-final-reviewer", "provider": "synthetic", "model": "fixture-model"},
                    "edge_decisions": decisions([dict(e, id=e["recommendation_id"]) for e in membership["edges"]]),
                    "mask_decisions": decisions([{"id": "flagged", "source_unit_ids": membership["flagged_source_unit_ids"]},
                                                 {"id": "preserved", "source_unit_ids": membership["preserved_source_unit_ids"]}]),
                    "condition_decisions": decisions(policy["required_source_conditions"]),
                    "lane_decisions": [dict(d, **{k: r[k] for k in
                        ("source_unit_ids", "acquisition_record_ids", "whole_edge_source_unit_ids", "dependency_block_ids")},
                        narrow_lane_vs_broader_dependence_resolved=True)
                        for d, r in zip(decisions(policy["exception_lanes"]), policy["exception_lanes"])]})
    return receipt, actual_evidence_fixture(receipt, "fixture-final-reviewer")


class ConfirmationDependencyTests(unittest.TestCase):
    @staticmethod
    def rebuild_fixture_policy(policy, chunks, catalog, source_evidence):
        rows = []
        for row in policy["canonical_bindings"]:
            records = [{"id": rid, "opaque_probe_id": meta["opaque_probe_id"], "event": meta["event"],
                        "role": meta["role"], "unit_id": meta["source_unit_ids"][0]
                        if meta["role"] in ("qa", "control") else meta["source_group_id"]}
                       for maps in (row["binding"]["record_mappings"], row["binding"]["teaching_root_mappings"])
                       for rid, meta in maps.items()]
            rows.append(dict(row, records=records))
        return build_final_policy_input(chunks, catalog,
            [edge["original_recommendation"] for edge in policy["membership"]["edges"]],
            policy["membership"]["flagged_source_unit_ids"], rows, source_evidence,
            policy_authors=policy["policy_authors"], required_source_conditions=policy["required_source_conditions"])

    def test_native_policy_author_aliases_reject_source_and_final_review_overlap_at_every_supported_layer(self):
        # Full all-eight validators, no stubs, and genuine synthetic completed
        # receipt/status bindings. No test creates real policy/source approval.
        for scope in ("source", "final"):
            for position in ("direct", "task_record", "result"):
                for key in ("taskId", "childRunId", "childThreadId", "clientRequestId"):
                    policy, chunks, catalog, source_evidence = final_fixture()
                    target = source_evidence[0] if scope == "source" else final_review_fixture(policy)[1]
                    actual = target["client_request_id"] if key == "clientRequestId" else target["task_status"][key]
                    status = dict(taskId="synthetic/policy-task", childRunId="synthetic/policy-run",
                                  childThreadId="synthetic/policy-thread", clientRequestId="synthetic-policy-request",
                                  status="completed", model="fixture-model")
                    status[key] = actual
                    author = policy["policy_authors"][0]
                    if position == "direct": author[key] = actual
                    elif position == "task_record": author["task_record"] = status
                    else: author["task_record"] = {"result": status}
                    policy = self.rebuild_fixture_policy(policy, chunks, catalog, source_evidence)
                    receipt, evidence = final_review_fixture(policy)
                    with self.subTest(scope=scope, position=position, key=key), self.assertRaisesRegex(
                            ValueError, "actual source/policy-author task channel"):
                        validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_policy_author_provenance_rejects_contradictions_and_unsupported_shapes(self):
        from spacing_rerun.confirmation_dependency import policy_author_channels
        base = {"agent_id": "synthetic-policy-author", "provider": "synthetic", "model": "fixture-model"}
        mutations = [
            {"task_id": "one", "taskId": "two"},
            {"childRunId": "one", "task_record": {"child_run_id": "two"}},
            {"clientRequestId": "one", "task_record": {"result": {"client_request_id": "two"}}},
            {"task_record": {"childThreadId": "one", "result": {"child_thread_id": "two"}}},
            {"task_record": {"childRunId": "one", "latestTerminalRunId": "two"}},
            {"taskId": "one", "childNodeId": "two"},
            {"task_record": []}, {"task_record": {}}, {"task_record": {"result": []}},
            {"task_record": {"result": {"result": {"taskId": "ignored"}}}},
            {"task_record": {"taskId": "one", "execution": {"taskId": "hidden"}}},
            {"result": {"taskId": "unsupported-direct-envelope"}},
            {"task_status": {"taskId": "unsupported-container"}},
            {"childRunID": "unsupported-spelling"}, {"taskId": None}, {"taskId": " padded "},
            {"task_record": {"taskId": "one", "status": {"taskId": "hidden"}}},
            {"task_record": {"taskId": "one", "hasPendingChildRuns": "false"}},
            {"task_record": {"taskId": "one", "actual_result": {"taskId": "hidden"}}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "policy-author|Policy-author"):
                policy_author_channels([dict(base, **mutation)])
        agreeing = dict(base, task_id="same-task", status="completed", task_record={"taskId": "same-task",
            "clientRequestId": "same-request", "result": {"taskId": "same-task", "client_request_id": "same-request",
                "childNodeId": "same-task", "childRunId": "same-run", "latestTerminalRunId": "same-run", "childThreadId": "same-thread"}})
        channels = policy_author_channels([agreeing])
        for alias in ("same-task", "same-request", "same-run", "same-thread"):
            self.assertIn("task:" + alias, channels)
        native = dict(base, task_record={"schema": "spacing-native-author-task-evidence-v1",
            "canonical_task_name": "/root/synthetic-native-policy-author", "status": "completed",
            "actual_result": {"policy_input_sha256": "synthetic-payload-only"},
            "actual_result_text": "Synthetic native output, no approval", "result_source": "synthetic-test"})
        self.assertIn("/root/synthetic-native-policy-author", policy_author_channels([native]))

    def test_final_review_all_explicit_names_reject_author_or_contradictory_identity(self):
        from spacing_rerun.confirmation_dependency import _completed_review, review_channels
        for key in ("identity", "review_task_id", "review_name", "name", "client_label", "actual_identity"):
            policy, chunks, catalog, source_evidence = final_fixture()
            receipt, evidence = final_review_fixture(policy)
            receipt["reviewer"][key] = policy["policy_authors"][0]["agent_id"]
            seal(receipt)
            evidence = actual_evidence_fixture(receipt, receipt["reviewer"]["agent_id"])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contradictory explicit|author.*channel"):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)
        policy, _, _, _ = final_fixture()
        receipt, evidence = final_review_fixture(policy)
        # Every agreeing task/request/run/thread spelling remains represented.
        for key, alias in zip(("identity", "review_task_id", "review_name", "name", "client_label", "actual_identity"),
                              [evidence["client_request_id"], evidence["task_status"]["taskId"],
                               evidence["task_status"]["childRunId"], evidence["task_status"]["childThreadId"],
                               evidence["client_request_id"], evidence["task_status"]["taskId"]]):
            receipt["reviewer"][key] = alias
        seal(receipt); evidence = actual_evidence_fixture(receipt, receipt["reviewer"]["agent_id"])
        binding = _completed_review(receipt, evidence)
        self.assertEqual(set(binding["declared_reviewer_identities"]), set(binding[k] for k in
                         ("task_id", "client_request_id", "child_run_id", "child_thread_id")))
        for name in binding["declared_reviewer_identities"]:
            self.assertIn("named:" + name, review_channels(binding))

    def test_source_all_explicit_reviewer_names_reject_current_and_history_authors(self):
        for scope in ("current", "history"):
            for key in ("identity", "review_task_id", "review_name", "name", "client_label", "actual_identity"):
                policy, chunks, catalog, source_evidence = final_fixture()
                chunk = chunks[0]
                if scope == "history":
                    chunk["history"] = [copy.deepcopy(chunk)]
                    target = chunk["history"][0]["authored"]["author"]["agent_id"]
                    old_evidence = copy.deepcopy(source_evidence[0])
                else:
                    target = chunk["authored"]["author"]["agent_id"]
                source = chunk["reviews"][0]
                source["reviewer"][key] = target
                seal(source)
                source_evidence[0] = actual_evidence_fixture(source, source["reviewer"]["agent_id"])
                if scope == "history": source_evidence.append(old_evidence)
                policy["required_source_conditions"][0]["source_receipt_sha256"] = source["sha256"]
                with self.subTest(scope=scope, key=key), self.assertRaisesRegex(ValueError, "Explicit named source"):
                    policy = self.rebuild_fixture_policy(policy, chunks, catalog, source_evidence)
                    receipt, evidence = final_review_fixture(policy)
                    validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_source_author_channels_retain_all_verified_declared_names(self):
        from spacing_rerun.confirmation_dependency import source_author_channels
        _, chunks, _, _ = final_fixture()
        author = chunks[0]["authored"]["author"]
        for key in ("identity", "review_task_id", "review_name", "name", "client_label", "actual_identity"):
            author[key] = "fixture-author-alias-" + key
        seal(chunks[0]["authored"])
        for name in author.values():
            if isinstance(name, str) and name.startswith("fixture-author-alias-"):
                self.assertIn("named:" + name, source_author_channels(chunks))
        author["name"] = []
        with self.assertRaisesRegex(ValueError, "canonical nonempty"):
            source_author_channels(chunks)

    def test_explicit_source_reviewer_declarations_remain_distinct_from_policy_authors(self):
        for secondary_name in (False, True):
            policy, chunks, catalog, source_evidence = final_fixture()
            source = chunks[0]["reviews"][0]
            declared = "fixture-explicit-source-reviewer"
            source["reviewer"]["agent_id"] = "/root" if secondary_name else declared
            if secondary_name: source["reviewer"]["name"] = declared
            seal(source)
            source_evidence[0] = actual_evidence_fixture(source, "fixture-independent-source-execution")
            policy["required_source_conditions"][0]["source_receipt_sha256"] = source["sha256"]
            policy["policy_authors"][0]["agent_id"] = declared
            policy = self.rebuild_fixture_policy(policy, chunks, catalog, source_evidence)
            receipt, evidence = final_review_fixture(policy)
            with self.subTest(secondary_name=secondary_name), self.assertRaisesRegex(ValueError, "Explicit named source"):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_legacy_generic_source_reviewer_label_is_not_an_actual_policy_task_alias(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        source = chunks[0]["reviews"][0]
        old_sha = source["sha256"]
        source["reviewer"]["agent_id"] = "/root"
        seal(source)
        source_evidence[0] = actual_evidence_fixture(source, "fixture-independent-source-review")
        policy["required_source_conditions"][0]["source_receipt_sha256"] = source["sha256"]
        self.assertNotEqual(old_sha, source["sha256"])
        policy["policy_authors"][0]["agent_id"] = "/root"
        policy["policy_authors"][0]["task_id"] = "synthetic/distinct-policy-author-task"
        row = policy["canonical_bindings"][0]
        records = [{"id": rid, "opaque_probe_id": m["opaque_probe_id"], "event": m["event"], "role": m["role"],
                    "unit_id": m["source_group_id"]} for rid, m in row["binding"]["record_mappings"].items()]
        policy = build_final_policy_input(chunks, catalog,
            [e["original_recommendation"] for e in policy["membership"]["edges"]],
            policy["membership"]["flagged_source_unit_ids"], [dict(row, records=records)], source_evidence,
            policy_authors=policy["policy_authors"], required_source_conditions=policy["required_source_conditions"])
        receipt, evidence = final_review_fixture(policy)
        validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_actual_source_reviewer_cannot_share_current_history_or_policy_author_channels(self):
        # Independent R2's full all-eight reproduction, without validator stubs.
        for scope in ("current", "history", "policy"):
            for alias in ("task", "run", "thread", "request"):
                policy, chunks, catalog, source_evidence = final_fixture()
                chunk = chunks[0]
                target = source_evidence[0]["task_status"]
                record = {"task_id": "synthetic/source-author-task", "child_thread_id": "synthetic/source-author-thread",
                          "child_run_id": "synthetic/source-author-run", "client_request_id": "fixture-source-author-channel",
                          "status": "completed", "model": "fixture-model"}
                key, actual = {"task": ("task_id", "taskId"), "run": ("child_run_id", "childRunId"),
                               "thread": ("child_thread_id", "childThreadId"), "request": ("client_request_id", None)}[alias]
                if actual:
                    record[key] = target[actual]
                else:
                    # Keep the author's valid source-author request role, and
                    # bind the completed reviewer to that same actual request.
                    source_evidence[0]["client_request_id"] = record[key]
                    raw = json.dumps({"task_statuses": [{"clientRequestId": record[key], "result": target}]})
                    source_evidence[0]["task_status_file"] = {"utf8": raw, "file_sha256": hashlib.sha256(raw.encode()).hexdigest()}
                provenance = [{"role": "author", "identity": chunk["authored"]["author"],
                    "authored_sha256": chunk["authored"]["sha256"], "author_packet_sha256": chunk["author_packet"]["sha256"],
                    "input_channel": "source_only", "task_id": record["task_id"], "task_record": record}]
                if scope == "current": chunk["author_task_provenance"] = provenance
                if scope == "history":
                    history = copy.deepcopy(chunk)
                    history["author_task_provenance"] = provenance
                    chunk["history"] = [history]
                if scope == "policy": policy["policy_authors"][0]["task_record"] = record
                row = policy["canonical_bindings"][0]
                records = [{"id": rid, "opaque_probe_id": m["opaque_probe_id"], "event": m["event"], "role": m["role"],
                            "unit_id": m["source_unit_ids"][0] if m["role"] in ("qa", "control") else m["source_group_id"]}
                           for rid, m in row["binding"]["record_mappings"].items()]
                policy = build_final_policy_input(chunks, catalog,
                    [e["original_recommendation"] for e in policy["membership"]["edges"]],
                    policy["membership"]["flagged_source_unit_ids"], [dict(row, records=records)], source_evidence,
                    policy_authors=policy["policy_authors"], required_source_conditions=policy["required_source_conditions"])
                receipt, evidence = final_review_fixture(policy)
                with self.subTest(scope=scope, alias=alias), self.assertRaisesRegex(ValueError, "Source semantic reviewer shares an actual"):
                    validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_generic_source_conditions_bind_own_event_source_and_entire_edge_mask(self):
        from spacing_rerun.confirmation_dependency import _conditions_satisfied, _final_membership
        policy, chunks, _, _ = final_fixture()
        packet, condition = chunks[0]["review_packet"], policy["required_source_conditions"][0]
        other_group = packet["groups"][1]
        other_sources = [s for s in packet["source_units"] if s["unit_id"] in other_group["source_unit_ids"]]
        unrelated = {"id": "synthetic/unrelated", "event": other_group["event"],
                     "records": [{"group": other_group, "sources": other_sources}]}
        recommendations = [e["original_recommendation"] for e in policy["membership"]["edges"]] + [unrelated]
        policy["membership"] = _final_membership([c["review_packet"] for c in chunks], recommendations,
            sorted(set(policy["membership"]["flagged_source_unit_ids"]) | set(other_group["source_unit_ids"])))
        changed = copy.deepcopy(policy)
        changed["required_source_conditions"][0].update(dependence_recommendation_ids=[unrelated["id"]],
                                                       flagged_source_unit_ids=other_group["source_unit_ids"])
        with self.assertRaisesRegex(ValueError, "own QA event"): _conditions_satisfied(changed, chunks)
        # Same-event own-source omissions and broader-only mask omissions use
        # genuine exact row/content approval; only condition geometry is bad.
        from experiments.spacing_rerun.tests.test_confirmation_lane import lane_fixture
        _, lane, _, _, membership, lane_chunks, _ = lane_fixture()
        condition = {"id": "synthetic/generic-broad-condition", "source_receipt_sha256": lane_chunks[0]["reviews"][0]["sha256"],
                     "acquisition_record_ids": ["qa0"], "dependence_recommendation_ids": ["r01", "r12"],
                     "flagged_source_unit_ids": ["u0", "u1", "u2"], "notes": "Synthetic actual QA/broader-edge requirement."}
        actual = {"membership": membership, "required_source_conditions": [condition]}
        _conditions_satisfied(actual, lane_chunks)
        for mutation in ("omit_own_source", "omit_broader_mask"):
            changed = copy.deepcopy(actual)
            row = changed["required_source_conditions"][0]
            if mutation == "omit_own_source": row["dependence_recommendation_ids"] = ["r12"]
            if mutation == "omit_broader_mask": row["flagged_source_unit_ids"] = ["u0", "u1"]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "own original QA provenance or whole"):
                _conditions_satisfied(changed, lane_chunks)

    def test_actual_final_reviewer_cannot_share_source_author_task_or_run_alias(self):
        for alias in ("task", "run", "thread"):
            policy, chunks, catalog, source_evidence = final_fixture()
            chunk = chunks[0]
            author = chunk["authored"]["author"]
            record = {"task_id": "synthetic/source-author-task", "child_thread_id": "synthetic/source-thread",
                      "child_run_id": "synthetic/source-run", "client_request_id": "fixture-source-author-channel",
                      "status": "completed", "model": "fixture-model"}
            record[{"task": "task_id", "run": "child_run_id", "thread": "child_thread_id"}[alias]] = {
                "task": "synthetic/task/fixture-final-reviewer", "run": "synthetic/run/fixture-final-reviewer",
                "thread": "synthetic/thread/fixture-final-reviewer"}[alias]
            chunk["author_task_provenance"] = [{"role": "author", "identity": author,
                "authored_sha256": chunk["authored"]["sha256"], "author_packet_sha256": chunk["author_packet"]["sha256"],
                "input_channel": "source_only", "task_id": record["task_id"], "task_record": record}]
            # Rebuild with actual provenance. No source/review validator is stubbed.
            row = policy["canonical_bindings"][0]
            records = [{"id": rid, "opaque_probe_id": meta["opaque_probe_id"], "event": meta["event"],
                        "role": meta["role"], "unit_id": meta["source_group_id"]}
                       for rid, meta in row["binding"]["record_mappings"].items()]
            policy = build_final_policy_input(chunks, catalog,
                [e["original_recommendation"] for e in policy["membership"]["edges"]],
                policy["membership"]["flagged_source_unit_ids"], [dict(row, records=records)], source_evidence,
                policy_authors=policy["policy_authors"], required_source_conditions=policy["required_source_conditions"])
            receipt, evidence = final_review_fixture(policy)
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "actual source/policy-author task channel"):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_concrete_review_models_require_exact_ids_and_disclosed_legacy_aliases(self):
        from spacing_rerun.confirmation_dependency import _completed_review
        for declared, actual in (("gpt-6.1-sol", "gpt-6-sol"), ("claude-opus-5-5", "claude-opus-4-6")):
            receipt = seal({"reviewer": {"agent_id": "synthetic-model-review", "provider": "synthetic", "model": declared}})
            evidence = actual_evidence_fixture(receipt, "synthetic-model-review")
            evidence["task_status"]["model"] = actual
            raw = json.dumps({"task_statuses": [{"clientRequestId": evidence["client_request_id"], "result": evidence["task_status"]}]})
            evidence["task_status_file"] = {"utf8": raw, "file_sha256": hashlib.sha256(raw.encode()).hexdigest()}
            with self.subTest(declared=declared), self.assertRaisesRegex(ValueError, "Concrete completed review model"):
                _completed_review(receipt, evidence)
        receipt = seal({"reviewer": {"agent_id": "synthetic-model-review", "provider": "synthetic", "model": "Claude Opus 5.5"}})
        evidence = actual_evidence_fixture(receipt, "synthetic-model-review")
        evidence["task_status"]["model"] = "claude-opus-5-5"
        raw = json.dumps({"task_statuses": [{"clientRequestId": evidence["client_request_id"], "result": evidence["task_status"]}]})
        evidence["task_status_file"] = {"utf8": raw, "file_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        with self.assertRaises(ValueError): _completed_review(receipt, evidence, require_named_task=False)
        evidence["legacy_model_alias"] = {"receipt_sha256": receipt["sha256"], "declared_model": "Claude Opus 5.5",
                                          "actual_model": "claude-opus-5-5", "notes": "Explicit synthetic legacy spelling only."}
        result = _completed_review(receipt, evidence, require_named_task=False)
        self.assertEqual(result["actual_model"], "claude-opus-5-5")
        self.assertEqual(result["declared_model"], "Claude Opus 5.5")
        with self.assertRaises(ValueError): _completed_review(receipt, evidence)

    def test_original_child_write_trace_can_bind_legacy_abbreviated_summary_only(self):
        from spacing_rerun.confirmation_dependency import _completed_review
        receipt = seal({"reviewer": {"agent_id": "synthetic-write-review", "provider": "synthetic", "model": "fixture-model"}})
        original = actual_evidence_fixture(receipt, "synthetic-write-review")
        status = original["task_status"]
        status["latestTerminalSummary"] = "Legacy abbreviated seal " + receipt["sha256"][:8] + " /synthetic/independent-review-output-long-name.json"
        raw = json.dumps({"task_statuses": [{"clientRequestId": original["client_request_id"], "result": status}]})
        original["task_status_file"] = {"utf8": raw, "file_sha256": hashlib.sha256(raw.encode()).hexdigest()}
        path = "/synthetic/independent-review-output-long-name.json"
        original["receipt_file"]["path"] = path
        item = {"itemId": "synthetic-write-item", "sourceThreadId": status["childThreadId"], "runId": status["childRunId"],
                "type": "command_execution", "status": "completed", "visibility": "local", "textTruncated": False,
                "nextTextOffset": None, "text": f"$ build && chmod 444 {path} && shasum -a 256 {path}\n" +
                receipt["sha256"] + "\n" + original["receipt_file"]["file_sha256"] + "  " + path + "\n" + path[-45:] + " True"}
        trace = {"thread": {"threadId": status["childThreadId"], "status": "completed", "latestRunId": status["childRunId"], "activeRunId": None, "pendingRequestCount": 0, "model": status["model"]}, "items": [item],
                 "recentRuns": [{"runId": status["childRunId"], "status": "completed", "model": status["model"]}]}
        def retain(evidence, value):
            raw = json.dumps({"structuredContent": value, "content": [{"type": "text", "text": json.dumps(value)}], "isError": False})
            evidence["completion_trace_file"] = {"utf8": raw, "file_sha256": hashlib.sha256(raw.encode()).hexdigest(), "trace_json_pointer": "/structuredContent"}
        retain(original, trace)
        self.assertEqual(_completed_review(receipt, original, require_named_task=False)["original_write_trace_binding"]["item_id"],
                         "synthetic-write-item")
        with self.assertRaises(ValueError): _completed_review(receipt, original)
        for mutation in ("raw_bytes", "thread", "run", "truncated", "path", "hash", "seal", "verification", "item"):
            evidence, changed = copy.deepcopy(original), copy.deepcopy(trace)
            current = changed["items"][0]
            if mutation == "thread": changed["thread"]["threadId"] = "another-thread"
            if mutation == "run": current["runId"] = "another-run"
            if mutation == "truncated": current["textTruncated"] = True
            if mutation == "path": current["text"] = current["text"].replace(path, "/wrong/path.json")
            if mutation == "hash": current["text"] = current["text"].replace(evidence["receipt_file"]["file_sha256"], "0" * 64)
            if mutation == "seal": current["text"] = current["text"].replace(receipt["sha256"], receipt["sha256"][:8])
            if mutation == "verification": current["text"] = current["text"].replace(" True", " False")
            if mutation == "item": current.pop("itemId")
            retain(evidence, changed)
            if mutation == "raw_bytes": evidence["completion_trace_file"]["utf8"] += " "
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _completed_review(receipt, evidence, require_named_task=False)

    def test_adopted_protocol_requires_bound_coverage_and_conditional_routing(self):
        from experiments.spacing_rerun.tests.test_confirmation_lane import protocol_fixture
        policy, chunks, catalog, source_evidence = final_fixture()
        row = policy["canonical_bindings"][0]
        prepared = [{"replicate_id": row["replicate_id"], "manifest_sha256": row["manifest_sha256"],
                     "training_provenance": row["training_provenance"], "records": [{
                         "id": rid, "opaque_probe_id": meta["opaque_probe_id"], "event": meta["event"], "role": meta["role"],
                         "unit_id": meta["source_unit_ids"][0] if meta["role"] in ("control", "qa") else meta["source_group_id"]}
                         for maps in (row["binding"]["record_mappings"], row["binding"]["teaching_root_mappings"])
                         for rid, meta in maps.items()]}]
        kwargs = {"policy_authors": policy["policy_authors"], "required_source_conditions": policy["required_source_conditions"],
                  "exception_protocol": protocol_fixture()}
        args = (chunks, catalog, [e["original_recommendation"] for e in policy["membership"]["edges"]],
                policy["membership"]["flagged_source_unit_ids"], prepared, source_evidence)
        with self.assertRaisesRegex(ValueError, "coverage proof"):
            build_final_policy_input(*args, **kwargs)
        ids = sorted({rid for c in policy["required_source_conditions"] for rid in c["acquisition_record_ids"]})
        custody = {"required_surface_correlate_ids_by_record": {r["id"]: [] for c in chunks for r in c["review_packet"]["acquisition_records"]}}
        proof = seal({"conditional_acquisition_record_ids": ids, "synthetic_contract_only": True, "evaluation_custody": custody})
        # This isolates dependency -> coverage routing. The genuine coverage
        # validator has separate receipt/context mutation tests. It supplies
        # no actual semantic approval for the synthetic proof here.
        with patch("spacing_rerun.sourceqa_protocol.verify_protocol_coverage") as verifier:
            candidate = build_final_policy_input(*args, **kwargs, protocol_coverage=proof)
            verifier.assert_called_once()
            context = verifier.call_args.kwargs["dependency_context"]
            self.assertEqual(context["membership_sha256"], policy["membership"]["sha256"])
            self.assertEqual(context["lane_reporting_policy_sha256"], candidate["lane_reporting_policy"]["sha256"])
            self.assertEqual(candidate["protocol_coverage"], proof)
            receipt, evidence = final_review_fixture(candidate)
            validate_final_policy(candidate, chunks, catalog, receipt, evidence, source_evidence)
            changed = copy.deepcopy(proof)
            changed["evaluation_custody"]["required_surface_correlate_ids_by_record"] = {}
            seal(changed)
            with self.assertRaisesRegex(ValueError, "L5 obligations"):
                build_final_policy_input(*args, **kwargs, protocol_coverage=changed)
        proof = seal({"conditional_acquisition_record_ids": []})
        with self.assertRaisesRegex(ValueError, "unconditional record"):
            build_final_policy_input(*args, **kwargs, protocol_coverage=proof)

    def test_final_policy_freeze_is_exclusive_and_guard_rechecks_manifest(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        receipt, evidence = final_review_fixture(policy)
        valid = validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)
        self.assertTrue(valid["source_approval_verified_separately"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            frozen = freeze_final_policy(path, policy, chunks, catalog, receipt, evidence, source_evidence)
            self.assertEqual(verify_frozen_policy(path, frozen["sha256"], chunks, catalog,
                                                 manifest_sha256="a" * 64), frozen)
            with self.assertRaises(FileExistsError):
                freeze_final_policy(path, policy, chunks, catalog, receipt, evidence, source_evidence)
            with self.assertRaisesRegex(ValueError, "Launch manifest"):
                verify_frozen_policy(path, frozen["sha256"], chunks, catalog, manifest_sha256="b" * 64)
            changed = copy.deepcopy(chunks)
            changed[0]["review_packet"]["groups"][0]["statement"] += " Changed."
            seal(changed[0]["review_packet"])
            with self.assertRaisesRegex(ValueError, "source binding changed"):
                verify_frozen_policy(path, frozen["sha256"], changed, catalog)

    def test_policy_approval_never_substitutes_for_missing_source_approval(self):
        policy, chunks, catalog, source_evidence = final_fixture(rejection=True)
        receipt, evidence = final_review_fixture(policy)
        with self.assertRaisesRegex(ValueError, "Unapproved current content"):
            validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_explicit_source_conditions_cannot_be_silently_dropped(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        policy["required_source_conditions"] = []
        seal(policy)
        receipt, evidence = final_review_fixture(policy)
        with self.assertRaisesRegex(ValueError, "Explicit current source approval conditions"):
            validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_final_input_rejects_changed_algorithm_and_provisional_membership(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        for mutation in ("algorithm", "provisional", "edge", "duplicate", "flag_mask", "preserved_mask", "fewer_chunks"):
            value, current = copy.deepcopy(policy), copy.deepcopy(chunks)
            if mutation == "algorithm":
                value["algorithm"]["primary"]["gold_aliases_em_answer_nll"] = "changed gold"
            if mutation == "provisional":
                value["membership"]["schema"] = "spacing-confirmation-dependency-membership-proposal-v1"
            if mutation == "edge":
                value["membership"]["edges"][0]["current_group_ids"] = []
            if mutation == "duplicate":
                value["membership"]["record_mappings"][next(iter(value["membership"]["record_mappings"]))]["canonical_probe_id"] = "invented"
            if mutation == "flag_mask":
                value["membership"]["flagged_source_unit_ids"].append("absent-unit")
            if mutation == "preserved_mask":
                value["membership"]["preserved_source_unit_ids"] = value["membership"]["flagged_source_unit_ids"]
            if mutation == "fewer_chunks":
                current.pop()
            seal(value["membership"])
            seal(value)
            receipt, evidence = final_review_fixture(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_policy(value, current, catalog, receipt, evidence, source_evidence)

    def test_final_review_needs_every_edge_mask_condition_and_separate_identity(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        original, _ = final_review_fixture(policy)
        for mutation in ("pending", "edge", "mask", "condition", "author", "unresolved", "conditions_incomplete", "outcomes"):
            receipt = copy.deepcopy(original)
            if mutation == "pending": receipt["membership_status"] = "provisional"
            if mutation == "edge": receipt["edge_decisions"] = []
            if mutation == "mask": receipt["mask_decisions"] = receipt["mask_decisions"][:1]
            if mutation == "condition": receipt["condition_decisions"][0]["status"] = "needs_revision"
            if mutation == "author": receipt["reviewer"] = policy["policy_authors"][0]
            if mutation == "unresolved": receipt["unresolved_items"] = ["pending-source04"]
            if mutation == "conditions_incomplete": receipt["all_source_conditions_captured"] = False
            if mutation == "outcomes": receipt["model_outcomes_seen"] = True
            seal(receipt)
            evidence = actual_evidence_fixture(receipt, receipt["reviewer"]["agent_id"])
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_completed_receipt_needs_exact_actual_status_and_raw_bytes(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        receipt, original = final_review_fixture(policy)
        for mutation in ("status", "summary", "missing_run", "pending_children", "receipt_bytes", "status_bytes", "model", "identity"):
            evidence = copy.deepcopy(original)
            if mutation == "status": evidence["task_status"]["status"] = "running"
            if mutation == "summary": evidence["task_status"]["latestTerminalSummary"] = "no receipt digest"
            if mutation == "missing_run": evidence["task_status"]["childRunId"] = None
            if mutation == "pending_children": evidence["task_status"]["hasPendingChildRuns"] = True
            if mutation == "model": evidence["task_status"]["model"] = "different-model"
            if mutation == "identity": evidence["client_request_id"] = "someone-else"
            if mutation == "receipt_bytes": evidence["receipt_file"]["utf8"] = "{}"
            if mutation not in ("receipt_bytes", "status_bytes"):
                text = json.dumps({"task_statuses": [{"clientRequestId": evidence["client_request_id"], "result": evidence["task_status"]}]})
                evidence["task_status_file"] = {"utf8": text, "file_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if mutation == "status_bytes": evidence["task_status_file"]["utf8"] = "{}"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, source_evidence)

    def test_source_status_bindings_cannot_be_removed_or_changed_after_review(self):
        policy, chunks, catalog, original = final_fixture()
        receipt, evidence = final_review_fixture(policy)
        for mutation in ("missing", "different_task", "raw_receipt"):
            changed = copy.deepcopy(original)
            if mutation == "missing": changed.pop()
            if mutation == "different_task":
                changed[0]["task_status"]["taskId"] = "different-actual-task"
                text = json.dumps({"task_statuses": [{"clientRequestId": changed[0]["client_request_id"], "result": changed[0]["task_status"]}]})
                changed[0]["task_status_file"] = {"utf8": text, "file_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if mutation == "raw_receipt": changed[0]["receipt_file"]["file_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_policy(policy, chunks, catalog, receipt, evidence, changed)

    def test_final_mapping_rejects_omitted_or_repeated_duplicate_roots(self):
        policy, chunks, catalog, source_evidence = final_fixture()
        for mutation in ("missing", "repeat"):
            changed = copy.deepcopy(policy)
            binding = changed["canonical_bindings"][0]["binding"]
            rid = next(iter(binding["record_mappings"]))
            if mutation == "missing": binding["record_mappings"].pop(rid)
            if mutation == "repeat": binding["record_mappings"]["duplicate-root-row"] = binding["record_mappings"][rid]
            seal(binding)
            seal(changed)
            receipt, evidence = final_review_fixture(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_policy(changed, chunks, catalog, receipt, evidence, source_evidence)

    def test_duplicate_first_mean_changes_secondary_without_changing_primary(self):
        rows, mapping = scoring_fixture()
        secondary = score(rows, mapping, "exact_match")
        self.assertEqual(secondary["group_means"]["e"]["g1"], .5)
        self.assertEqual(secondary["event_macro"], .375)
        self.assertEqual(score(rows, mapping, "loss")["event_macro"], .75)
        expected_primary = aggregate(rows)["old"]["exact_match_event_macro"]
        self.assertEqual(expected_primary, .625)
        self.assertEqual(score(rows, mapping, "exact_match", weighting="primary_formula")["event_macro"], expected_primary)

    def test_more_duplicate_forms_do_not_reweight_the_group(self):
        rows, mapping = scoring_fixture()
        reduced = [row for row in rows if not row["id"].startswith("a") or row["id"] == "a0"]
        self.assertEqual(score(rows, mapping, "exact_match")["event_macro"],
                         score(reduced, mapping, "exact_match")["event_macro"])

    def test_overlapping_sets_form_one_proposed_component_and_keep_provenance(self):
        packet, recs = source_fixture()
        result = membership_proposal([packet], recs, ["u2"])
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(result["group_block"]["g0"], result["group_block"]["g2"])
        self.assertEqual(len(result["proposed_edges"]), 2)
        self.assertFalse(result["final_membership_frozen"])
        self.assertFalse(result["membership_approval_claimed"])
        self.assertEqual(result["preserved_source_unit_ids"], ["u0"])
        self.assertEqual(len(result["record_mappings"]), 8)

    def test_original_unit_projection_survives_approved_source_regrouping(self):
        packet, recs = source_fixture()
        changed = copy.deepcopy(packet)
        changed["groups"] = [{"id": "merged", "event": "e1", "statement": "same approved claim",
                              "source_unit_ids": ["u0", "u1"]}] + changed["groups"][2:]
        result = membership_proposal([seal(changed)], recs)
        self.assertEqual(result["proposed_edges"][0]["current_group_ids"], ["merged"])
        self.assertEqual(result["record_mappings"]["p0"]["source_group_id"], "merged")
        self.assertEqual(result["record_mappings"]["p0"]["source_unit_ids"], ["u0"])

    def test_proposals_reject_stale_labels_overlap_and_cross_event_edges(self):
        packet, recs = source_fixture()
        for mutation in ("label", "overlap", "cross_event", "missing_unit"):
            with self.subTest(mutation=mutation):
                p, rs = copy.deepcopy(packet), copy.deepcopy(recs)
                if mutation == "label": p["source_units"][0]["canonical_answer_labels"] = ["changed gold"]
                if mutation == "overlap": p["source_units"][1]["canonical_probes"][0]["opaque_duplicate_cluster_members"] += ["p0"]
                if mutation == "cross_event": rs[0]["records"].append({"group": p["groups"][3], "sources": [p["source_units"][3]]})
                if mutation == "missing_unit": p["groups"][0]["source_unit_ids"] = []
                with self.assertRaises(ValueError): membership_proposal([seal(p)], rs)

    def test_qa_forms_average_within_target_and_teaching_requires_canonical_root(self):
        packet, recs = source_fixture()
        membership = membership_proposal([packet], recs)
        records = [{"id": "t0", "unit_id": "g0", "event": "e1", "answer": "gold0", "questions": ["f0", "f1"]},
                   {"id": "t1", "unit_id": "g0", "event": "e1", "answer": "gold0", "questions": ["f0", "f1", "f2"]}]
        mapping = acquisition_form_mappings(records, membership)
        rows = [{"id": key, "unit_id": "g0", "event": "e1", "role": "old",
                 "exact_match": int(key != "t0/0")} for key in mapping]
        self.assertEqual(score(rows, mapping, "exact_match")["event_macro"], .75)
        teaching = teaching_form_mappings([{"id": "expanded", "canonical_probe_id": "p0"}], membership)
        self.assertEqual(teaching["expanded"]["source_unit_ids"], ["u0"])
        with self.assertRaises(ValueError):
            teaching_form_mappings([{"id": "expanded", "canonical_probe_id": "p0-copy"}], membership)

    def test_control_primary_raw_units_stay_distinct_from_secondary_groups(self):
        rows = [{"id": str(i), "event": "e", "role": "control", "unit_id": f"raw{i}",
                 "exact_match": value} for i, value in enumerate([0, 0, 1])]
        mapping = {r["id"]: {"event": "e", "source_unit_ids": [r["unit_id"]],
                   "source_group_id": "merged" if i < 2 else "other",
                   "dependency_block_id": "merged" if i < 2 else "other",
                   "duplicate_cluster_id": r["id"]} for i, r in enumerate(rows)}
        self.assertAlmostEqual(score(rows, mapping, "exact_match", weighting="primary_formula")["event_macro"], 1 / 3)
        self.assertEqual(score(rows, mapping, "exact_match")["event_macro"], .5)

    def test_canonical_binding_proves_roots_only_and_preserves_control_raw_unit(self):
        packet, recs = source_fixture()
        membership = membership_proposal([packet], recs)
        records = [{"id": "actual0", "opaque_probe_id": "p0", "event": "e1", "role": "old", "unit_id": "g0"},
                   {"id": "actual1", "opaque_probe_id": "p1", "event": "e1", "role": "control", "unit_id": "u1"}]
        result = bind_canonical_records(records, membership)
        self.assertTrue(result["only_canonical_duplicate_roots_scored"])
        self.assertFalse(result["duplicate_members_separately_scored"])
        self.assertEqual(result["record_mappings"]["actual1"]["source_group_id"], "g1")
        for kind in ("member", "wrong_control_unit", "wrong_event", "duplicate"):
            bad = copy.deepcopy(records)
            if kind == "member": bad[0]["opaque_probe_id"] = "p0-copy"
            if kind == "wrong_control_unit": bad[1]["unit_id"] = "g1"
            if kind == "wrong_event": bad[0]["event"] = "e2"
            if kind == "duplicate": bad.append(bad[0])
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                bind_canonical_records(bad, membership)

    def test_masks_keep_frozen_partition_and_report_empty_events_without_zero(self):
        rows, mapping = scoring_fixture()
        before = copy.deepcopy(mapping)
        masked = score(rows, mapping, "exact_match", excluded_units=["u1"])
        self.assertEqual(masked["event_macro"], .5)
        self.assertEqual(masked["block_counts_by_event"]["e"], 2)
        empty = score(rows, mapping, "exact_match", excluded_units=["u1", "u2", "u3"])
        self.assertIsNone(empty["event_macro"])
        self.assertEqual(empty["empty_event_ids"], ["e"])
        self.assertEqual(mapping, before)
        all_scores = diagnostic_scores(rows, mapping, "exact_match", ["u1"], ["u3"])
        self.assertEqual(len(all_scores), 10)
        self.assertEqual(all_scores["equal_block/flagged_only"]["event_macro"], .5)

    def test_acquisition_qa_mask_is_conservative_over_shared_group_provenance(self):
        rows = [{"id": "qa/0", "unit_id": "g", "event": "e", "role": "old", "exact_match": 1}]
        mapping = {"qa/0": {"event": "e", "source_unit_ids": ["u1", "u2"], "source_group_id": "g",
                             "dependency_block_id": "b", "duplicate_cluster_id": "target"}}
        self.assertIsNone(score(rows, mapping, "exact_match", excluded_units=["u2"])["event_macro"])

    def test_complete_influence_includes_singletons_and_both_event_weightings(self):
        rows, mapping = scoring_fixture()
        cases = influence_scores(rows, mapping, "exact_match")
        self.assertEqual(len(cases), 4)
        self.assertEqual(sum(c["kind"] == "leave_one_block_out" for c in cases), 2)
        self.assertEqual(sum(c["kind"] == "leave_one_event_out" for c in cases), 2)
        self.assertEqual(next(c["score"]["event_macro"] for c in cases if c["item_id"] == "b3"), .75)

    def test_pairs_keep_valid_replicates_with_undefined_secondary_scores(self):
        pairs = [{"replicate_id": "r0", "a": 1, "b": 0},
                 {"replicate_id": "r1", "a": 2, "b": 0},
                 {"replicate_id": "r2", "a": None, "b": 0}]
        summary = paired_summary(pairs)
        self.assertEqual((summary["n_registered"], summary["n_pair"]), (3, 2))
        self.assertEqual(summary["missing_secondary_pair_ids"], ["r2"])
        self.assertEqual(summary["estimate"], 1.5)
        self.assertNotIn("p_two_sided", summary)
        self.assertIsNone(paired_summary(pairs[-1:])["estimate"])
        self.assertIsNone(paired_summary(pairs[:1])["ci95_descriptive"])
        with self.assertRaises(ValueError): paired_summary(pairs + pairs[:1])

    def test_a1_reports_sign_zero_coverage_and_pair_count_divergence_explicitly(self):
        primary = {"estimate": 1, "ci95_descriptive": [.2, 1.8], "n_pair": 3}
        secondary = {"estimate": -.1, "ci95_descriptive": [-.2, 0], "n_pair": 2}
        result = divergence(primary, secondary)
        self.assertTrue(result["mean_sign_differs"])
        self.assertTrue(result["ci95_zero_coverage_differs"])
        self.assertTrue(result["complete_pair_counts_differ"])
        self.assertIn("coverage differs: True", result["statement"])
        undefined = divergence(primary, {"estimate": None, "ci95_descriptive": None, "n_pair": 0})
        self.assertIsNone(undefined["mean_sign_differs"])
        self.assertIn("undefined", undefined["statement"])

    def test_each_secondary_has_divergence_labels_and_cannot_drop_registered_pair(self):
        pairs = [{"replicate_id": str(i), "a": 1, "b": 0} for i in range(3)]
        result = contrast_report(pairs, {"equal_block/all_source": pairs}, contrast="EXP-minus-UNI",
                                 endpoint=84, role="old", variant="canonical", metric="loss")
        self.assertIn("primary_secondary_divergence", result["secondaries"]["equal_block/all_source"])
        self.assertEqual(result["endpoint"], 84)
        self.assertFalse(result["primary_scores_replaced"])
        self.assertFalse(result["acquisition_eligibility_changed"])
        with self.assertRaises(ValueError):
            contrast_report(pairs, {"bad": pairs[:-1]}, contrast="EXP-minus-UNI", endpoint=84,
                            role="old", variant="canonical", metric="loss")

    def test_bad_prediction_metrics_and_mapping_fail_before_reporting(self):
        rows, mapping = scoring_fixture()
        for mutation in ("duplicate_id", "nan", "mixed_roles", "different_event", "conflicting_block"):
            with self.subTest(mutation=mutation):
                r, m = copy.deepcopy(rows), copy.deepcopy(mapping)
                if mutation == "duplicate_id": r += r[:1]
                if mutation == "nan": r[0]["exact_match"] = float("nan")
                if mutation == "mixed_roles": r[0]["role"] = "new"
                if mutation == "different_event": r[0]["event"] = "wrong"
                if mutation == "conflicting_block": m["z"]["dependency_block_id"] = "wrong"
                with self.assertRaises(ValueError): score(r, m, "exact_match")

    def test_cached_influence_matches_full_recomputation_including_empty_events(self):
        rows, mapping = scoring_fixture()
        for case in influence_scores(rows, mapping, "loss"):
            if case["kind"] == "leave_one_block_out":
                expected = score(rows, mapping, "loss", excluded_blocks=[case["item_id"]])
            else:
                expected = score(rows, mapping, "loss", weighting=case["weighting"], excluded_events=[case["item_id"]])
            for field in ("event_macro", "empty_event_ids", "rows_retained", "rows_excluded"):
                self.assertEqual(case["score"][field], expected[field])

    def test_complete_report_emits_every_mask_and_influence_and_binds_saved_primary(self):
        a, mapping = scoring_fixture()
        b = [dict(row, exact_match=0) for row in a]
        reps = [{"replicate_id": str(i), "a_rows": a, "b_rows": b, "mapping": mapping,
                 "primary_a": .625, "primary_b": 0} for i in range(3)]
        report = complete_contrast_report(reps, contrast="EXP-minus-UNI", endpoint=84, role="old",
                                          variant="canonical", metric="exact_match",
                                          flagged_units=["u1"], preserved_units=["u3"])
        self.assertEqual(len(report["secondaries"]), 10)
        self.assertEqual(len(report["influence"]), 4)
        self.assertTrue(all("ci95_descriptive" not in case["paired_summary"] for case in report["influence"]))
        self.assertTrue(all("primary_secondary_divergence" in s for s in report["secondaries"].values()))
        self.assertIsNone(next(c["deletion_minus_full_mean_shift"] for c in report["influence"]
                               if c["kind"] == "leave_one_event_out"))
        bad = copy.deepcopy(reps)
        bad[0]["primary_a"] = .5
        with self.assertRaises(ValueError):
            complete_contrast_report(bad, contrast="EXP-minus-UNI", endpoint=84, role="old",
                                     variant="canonical", metric="exact_match",
                                     flagged_units=[], preserved_units=[])


if __name__ == "__main__":
    unittest.main()
