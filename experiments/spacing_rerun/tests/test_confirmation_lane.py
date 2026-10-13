"""Synthetic lane evidence and descriptive tests confer no real approvals."""
import copy
import hashlib
from pathlib import Path
import unittest

from spacing_rerun.common import digest
from spacing_rerun.confirmation_dependency import _final_membership, _bind_prepared_roots
from spacing_rerun.confirmation_lane import (
    ATTRIBUTES, PROPERTY_REVIEW_SCHEMA, balance_disclosures, build_exception_lane,
    complement_contrast, prediction_tables, reporting_policy, validate_lane_inputs,
    validate_final_lane_decisions,
    reporting_specification,
    required_surface_correlates,
)
from experiments.spacing_rerun.tests.test_confirmation_dependency import (
    actual_evidence_fixture, seal, source_fixture,
)


def protocol_fixture():
    root = Path(__file__).resolve().parents[3]
    return seal({"schema": "spacing-sourceqa-prospective-review-protocol-v1",
                 "status": "operator_adopted_criteria_with_launch_obligations_unfulfilled",
                 "source_approval_created_by_this_document": False, "confirmation_ready": False,
                 "main_launched": False, "required_controls": {k: "synthetic criterion, no approvals" for k in
                    ("K1_probe_balance", "K2_lane_confusion_tables", "K3_surface_correlate_flag",
                     "K4_block_and_mask", "K5_A1_A2", "K6_census", "K7_freeze_order",
                     "K8_independent_final_reviews", "K9_disclosure")},
                 "exception_rules": {k: "synthetic criterion, no approvals" for k in
                    ("LE1_eligibility_any_of", "LE1_exclusions", "LE2_membership", "LE3_certification")},
                 "frozen_mechanical_files": {"experiments/spacing_rerun/spacing_rerun/" + name:
                    hashlib.sha256((root / "experiments/spacing_rerun/spacing_rerun" / name).read_bytes()).hexdigest()
                    for name in ("data.py", "acquisition.py", "encoding.py")}})


def lane_fixture():
    packet, recs = source_fixture()
    # Narrow pair u0/u1; accepted dependence component deliberately includes u2.
    # This simulates D25 geometry, but the labels and receipts are synthetic.
    qa = [{"id": "qa" + str(i), "unit_id": "g" + str(i), "event": "e1",
           "answer": "gold" + str(i), "questions": ["fixture A", "fixture B"]} for i in (0, 1, 2)]
    packet["acquisition_records"] = qa
    seal(packet)
    receipt = seal({"reviewer": {"agent_id": "synthetic-qa-reviewer", "provider": "synthetic", "model": "fixture-model"},
                    "acquisition_records": [{"id": r["id"], "status": "approved", "reviewed_content_sha256": digest(r)} for r in qa]})
    chunks = [{"review_packet": packet, "reviews": [receipt],
               "authored": {"author": {"agent_id": "synthetic-source-author", "provider": "synthetic", "model": "fixture-model"}}}]
    membership = _final_membership([packet], recs, ["u0", "u1", "u2"])
    candidate = {"id": "synthetic-lane", "event": "e1", "sub_kind": "original_label_nonuniqueness", "eligibility": "LE1c",
                 "source_unit_ids": ["u0", "u1"], "acquisition_record_ids": ["qa0", "qa1"],
                 "dependence_recommendation_ids": ["r01", "r12"], "disclosure_notes": "Synthetic narrow pair and broader three-unit component.",
                 "source_record_review_bindings": [{"record_id": r["id"], "source_receipt_sha256": receipt["sha256"],
                                                    "reviewed_content_sha256": digest(r)} for r in qa[:2]]}
    lane = build_exception_lane(candidate, membership, chunks)
    protocol = protocol_fixture()
    property_receipt = seal({"schema": PROPERTY_REVIEW_SCHEMA, "status": "approved", "lane_sha256": digest(lane),
        "protocol_sha256": protocol["sha256"], "eligibility": "LE1c", "sub_kind": "original_label_nonuniqueness",
        "eligibility_notes": "Synthetic certification only.", "ordinary_qa_defects_resolved": True, "unresolved_items": [],
        "attribute_certifications": {k: {"no_faithful_nonleaking_semantic_discriminator": True,
                                         "notes": "Synthetic exact-content certification of " + k} for k in ATTRIBUTES},
        "model_outcomes_seen": False, "evaluation_wording_seen": False, "author_rationales_seen": False,
        "human_review_complete": False, "own_attribution_preserved": True, "nonuniqueness_not_created_by_qa": True,
        "original_labels_only": True, "surface_correlates_assessed": True, "surface_correlate_certifications": [],
        "reviewer": {"agent_id": "synthetic-property-reviewer", "provider": "synthetic", "model": "fixture-model"}})
    reviews = [{"lane_id": lane["id"], "receipt": property_receipt,
                "evidence": actual_evidence_fixture(property_receipt, "synthetic-property-reviewer")}]
    authors = [{"agent_id": "synthetic-policy-author", "provider": "synthetic", "model": "fixture-model"}]
    return candidate, lane, protocol, reviews, membership, chunks, authors


class ConfirmationLaneTests(unittest.TestCase):
    def test_typed_candidate_preserves_actual_nonapproval_and_needs_separate_positive_resolution(self):
        from spacing_rerun.confirmation_lane import build_exception_lane_candidate, validate_lane_source_resolutions
        candidate, _, protocol, reviews, membership, chunks, authors = lane_fixture()
        receipt = chunks[0]["reviews"][0]
        receipt["acquisition_records"][0]["status"] = "conditional"
        receipt["acquisition_records"][1]["status"] = "exception_candidate"
        seal(receipt)
        for ref in candidate["source_record_review_bindings"]: ref["source_receipt_sha256"] = receipt["sha256"]
        lane = build_exception_lane_candidate(candidate, membership, chunks)
        self.assertTrue(lane["candidate_only"])
        self.assertFalse(lane["source_approval_verified"])
        self.assertEqual([r["actual_decision"]["status"] for r in lane["source_decision_states"]], ["conditional", "exception_candidate"])
        with self.assertRaises(ValueError): build_exception_lane(candidate, membership, chunks)
        with self.assertRaises(ValueError): validate_lane_source_resolutions([lane], [], chunks)
        with self.assertRaises(ValueError): validate_lane_source_resolutions([lane], candidate["source_record_review_bindings"], chunks)
        property_receipt = reviews[0]["receipt"]
        property_receipt["lane_sha256"] = digest(lane)
        seal(property_receipt)
        reviews[0]["evidence"] = actual_evidence_fixture(property_receipt, "synthetic-property-reviewer")
        validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)
        positive = copy.deepcopy(receipt)
        for row in positive["acquisition_records"]: row["status"] = "approved"
        seal(positive)
        chunks[0]["reviews"].append(positive)
        resolved = [dict(ref, source_receipt_sha256=positive["sha256"]) for ref in candidate["source_record_review_bindings"]]
        validate_lane_source_resolutions([lane], resolved, chunks)
        self.assertEqual(lane, build_exception_lane_candidate(candidate, membership, chunks))
        validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)

    def test_property_actual_channels_reject_policy_author_task_run_and_thread_aliases(self):
        for key, prefix in (("task_id", "synthetic/task/"), ("child_run_id", "synthetic/run/"),
                            ("child_thread_id", "synthetic/thread/")):
            _, lane, protocol, reviews, membership, chunks, authors = lane_fixture()
            authors[0][key] = prefix + "synthetic-property-reviewer"
            with self.subTest(alias=key), self.assertRaisesRegex(ValueError, "actual source/policy-author task channel"):
                validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)

    def test_property_all_explicit_names_reject_author_or_contradictory_identity(self):
        for key in ("identity", "review_task_id", "review_name", "name", "client_label", "actual_identity"):
            _, lane, protocol, reviews, membership, chunks, authors = lane_fixture()
            receipt = reviews[0]["receipt"]
            receipt["reviewer"][key] = authors[0]["agent_id"]
            seal(receipt)
            reviews[0]["evidence"] = actual_evidence_fixture(receipt, receipt["reviewer"]["agent_id"])
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "contradictory explicit|author.*channel"):
                validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)

    def test_property_review_rejects_native_policy_author_aliases_at_every_supported_layer(self):
        for position in ("direct", "task_record", "result"):
            for key in ("taskId", "childRunId", "childThreadId", "clientRequestId"):
                _, lane, protocol, reviews, membership, chunks, authors = lane_fixture()
                actual = reviews[0]["evidence"]
                alias = actual["client_request_id"] if key == "clientRequestId" else actual["task_status"][key]
                status = dict(taskId="synthetic/policy-task", childRunId="synthetic/policy-run",
                              childThreadId="synthetic/policy-thread", clientRequestId="synthetic-policy-request",
                              status="completed", model="fixture-model")
                status[key] = alias
                if position == "direct": authors[0][key] = alias
                elif position == "task_record": authors[0]["task_record"] = status
                else: authors[0]["task_record"] = {"result": status}
                with self.subTest(position=position, key=key), self.assertRaisesRegex(
                        ValueError, "actual source/policy-author task channel"):
                    validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)

    def test_surface_correlate_obligations_derive_from_actual_bound_property_receipt(self):
        _, lane, protocol, reviews, membership, chunks, authors = lane_fixture()
        receipt = reviews[0]["receipt"]
        receipt["surface_correlate_certifications"] = [{"id": "synthetic-L5-cue", "acquisition_record_ids": ["qa0"],
            "source_unit_ids": ["u0"], "original_source_backed": True, "notes": "Synthetic original-source correlate certification."}]
        seal(receipt)
        reviews[0]["evidence"] = actual_evidence_fixture(receipt, "synthetic-property-reviewer")
        validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)
        required = required_surface_correlates([lane], reviews, ["qa0", "qa1", "qa2"])
        self.assertEqual(required, {"qa0": ["synthetic-L5-cue"], "qa1": [], "qa2": []})
        for mutation in ("wrong_unit", "other_qa", "unbacked", "duplicate", "unassessed"):
            values = copy.deepcopy(reviews)
            actual = values[0]["receipt"]
            certification = actual["surface_correlate_certifications"][0]
            if mutation == "wrong_unit": certification["source_unit_ids"] = ["u1"]
            if mutation == "other_qa": certification["acquisition_record_ids"] = ["qa2"]
            if mutation == "unbacked": certification["original_source_backed"] = False
            if mutation == "duplicate": actual["surface_correlate_certifications"].append(copy.deepcopy(certification))
            if mutation == "unassessed": actual["surface_correlates_assessed"] = False
            seal(actual)
            values[0]["evidence"] = actual_evidence_fixture(actual, "synthetic-property-reviewer")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_lane_inputs([lane], protocol, values, membership, chunks, authors)

    def test_preregistration_requires_canonical_policy_and_all_four_seals(self):
        from spacing_rerun.confirmation_dependency import verify_preregistered_dependencies
        # Isolate digest routing; actual source/coverage validation is separate.
        frozen = {"sha256": "f" * 64, "policy": {"exception_protocol": {"sha256": "p" * 64},
                    "protocol_coverage": {"sha256": "c" * 64}, "lane_reporting_policy": reporting_policy()}}
        prereg = {"confirmation_dependency_policy_path": "/synthetic/reviewed-policy.json",
                  "confirmation_dependency_policy_sha256": frozen["sha256"],
                  "sourceqa_protocol_sha256": "p" * 64, "sourceqa_protocol_coverage_sha256": "c" * 64,
                  "lane_reporting_policy_sha256": reporting_policy()["sha256"],
                  "lane_reporting_specification_sha256": reporting_specification()["sha256"]}
        verify_preregistered_dependencies(prereg, frozen)
        for key in prereg:
            value = dict(prereg)
            value.pop(key)
            with self.subTest(omitted=key), self.assertRaises(ValueError):
                verify_preregistered_dependencies(value, frozen)
        value = dict(prereg, lane_reporting_specification_sha256="x" * 64)
        with self.assertRaises(ValueError):
            verify_preregistered_dependencies(value, frozen)

    def test_final_review_cannot_silently_split_narrow_pair_from_broader_edge(self):
        _, lane, _, _, _, _, _ = lane_fixture()
        decision = {"id": lane["id"], "status": "approved", "reviewed_content_sha256": digest(lane),
                    "notes": "Synthetic final review resolves pair vs broader block only.",
                    "narrow_lane_vs_broader_dependence_resolved": True,
                    **{k: lane[k] for k in ("source_unit_ids", "acquisition_record_ids",
                                           "whole_edge_source_unit_ids", "dependency_block_ids")}}
        original = {"lane_reporting_policy_approved": True, "lane_decisions": [decision]}
        validate_final_lane_decisions(original, [lane])
        for mutation in ("missing", "content", "unresolved", "narrowed_edge", "different_qa", "no_reporting"):
            receipt = copy.deepcopy(original)
            if mutation == "missing": receipt["lane_decisions"] = []
            if mutation == "content": receipt["lane_decisions"][0]["reviewed_content_sha256"] = "0" * 64
            if mutation == "unresolved": receipt["lane_decisions"][0]["narrow_lane_vs_broader_dependence_resolved"] = False
            if mutation == "narrowed_edge": receipt["lane_decisions"][0]["whole_edge_source_unit_ids"] = ["u0", "u1"]
            if mutation == "different_qa": receipt["lane_decisions"][0]["acquisition_record_ids"].append("qa2")
            if mutation == "no_reporting": receipt["lane_reporting_policy_approved"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_final_lane_decisions(receipt, [lane])

    def test_narrow_pair_preserves_broader_edge_and_actual_property_evidence(self):
        _, lane, protocol, reviews, membership, chunks, authors = lane_fixture()
        self.assertEqual(lane["source_unit_ids"], ["u0", "u1"])
        self.assertEqual(lane["whole_edge_source_unit_ids"], ["u0", "u1", "u2"])
        binding = validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)
        self.assertEqual(binding[0]["lane_sha256"], digest(lane))
        self.assertEqual(binding[0]["task_id"], "synthetic/task/synthetic-property-reviewer")

    def test_methods_only_or_missing_attribute_certificate_cannot_approve_lane(self):
        _, lane, protocol, original, membership, chunks, authors = lane_fixture()
        for mutation in ("methods", "attribute", "pending", "content", "outcomes", "author", "ordinary", "source_status", "raw_bytes"):
            reviews = copy.deepcopy(original)
            receipt = reviews[0]["receipt"]
            if mutation == "methods": receipt["schema"] = "spacing-source-label-exception-protocol-science-assessment-v1"
            if mutation == "attribute": receipt["attribute_certifications"].pop("scope")
            if mutation == "pending": receipt["unresolved_items"] = ["membership unresolved"]
            if mutation == "content": receipt["lane_sha256"] = "0" * 64
            if mutation == "outcomes": receipt["model_outcomes_seen"] = True
            if mutation == "author": receipt["reviewer"]["agent_id"] = "synthetic-source-author"
            if mutation == "ordinary": receipt["ordinary_qa_defects_resolved"] = False
            seal(receipt)
            reviews[0]["evidence"] = actual_evidence_fixture(receipt, receipt["reviewer"]["agent_id"])
            if mutation == "source_status": reviews[0]["evidence"]["task_status"]["status"] = "running"
            if mutation == "raw_bytes": reviews[0]["evidence"]["receipt_file"]["utf8"] += "corrupt"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)
        with self.assertRaises(ValueError):
            validate_lane_inputs([lane], protocol, [], membership, chunks, authors)

    def test_lane_cannot_narrow_whole_edge_mask_or_invent_gold_or_row_approval(self):
        original, _, _, _, original_membership, original_chunks, _ = lane_fixture()
        for mutation in ("narrow_mask", "same_label", "qa_status", "row_hash", "missing_qa", "other_unit", "wrong_subkind"):
            candidate, membership, chunks = copy.deepcopy(original), copy.deepcopy(original_membership), copy.deepcopy(original_chunks)
            if mutation == "narrow_mask": membership["flagged_source_unit_ids"] = ["u0", "u1"]
            if mutation == "same_label": membership["sources"]["u1"]["canonical_answer_labels"] = ["GOLD0!"]
            if mutation == "qa_status": chunks[0]["reviews"][0]["acquisition_records"][0]["status"] = "needs_revision"
            if mutation == "row_hash": candidate["source_record_review_bindings"][0]["reviewed_content_sha256"] = "0" * 64
            if mutation == "missing_qa": candidate["acquisition_record_ids"] = ["qa0"]
            if mutation == "other_unit": candidate["source_unit_ids"] = ["u0", "u2"]
            if mutation == "wrong_subkind": candidate["sub_kind"] = "alias"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_exception_lane(candidate, membership, chunks)

    def test_adoption_criteria_and_mechanical_pin_are_required(self):
        _, lane, original, reviews, membership, chunks, authors = lane_fixture()
        for mutation in ("methods", "code", "control"):
            protocol = copy.deepcopy(original)
            if mutation == "methods": protocol["status"] = "methods_assessment_only"
            if mutation == "code": protocol["frozen_mechanical_files"]["experiments/spacing_rerun/spacing_rerun/data.py"] = "0" * 64
            if mutation == "control": protocol["required_controls"].pop("K6_census")
            seal(protocol)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_lane_inputs([lane], protocol, reviews, membership, chunks, authors)

    def test_all_pairs_own_sibling_other_and_missing_predictions_remain_descriptive(self):
        lane = {"id": "set3", "source_unit_ids": ["u0", "u1", "u2"],
                "original_labels": {"u0": ["a"], "u1": ["b"], "u2": ["c"]}}
        mapping = {f"r{i}": {"source_unit_ids": ["u" + str(i)]} for i in range(3)}
        rows = [{"id": "r0", "prediction": " B! "}, {"id": "r1", "prediction": "b"}, {"id": "r2"}]
        golds = {"r0": "a", "r1": "b", "r2": "c"}
        tables = prediction_tables([lane], rows, mapping, golds)
        self.assertEqual(len(tables), 3)
        pair01 = tables[0]
        self.assertEqual(pair01["target_sides"][0]["counts"], {"own_label": 0, "sibling_label": 1, "other": 0})
        self.assertEqual(pair01["target_sides"][1]["frequencies"]["own_label"], 1)
        pair02 = tables[1]
        self.assertEqual(pair02["target_sides"][0]["counts"]["other"], 1)
        self.assertIsNone(pair02["target_sides"][1]["counts"])
        self.assertEqual(pair02["target_sides"][1]["missing_prediction_row_ids"], ["r2"])
        self.assertFalse(pair01["either_label_success_metric_computed"])
        with self.assertRaisesRegex(ValueError, "gold differs"):
            prediction_tables([lane], rows, mapping, dict(golds, r0="changed"))

    def test_balance_counts_roots_not_duplicate_members_or_expanded_forms(self):
        packet, _ = source_fixture()
        packet["groups"][0]["source_unit_ids"] = ["u0", "u1", "u2"]
        packet["groups"] = [packet["groups"][0], packet["groups"][3]]
        # Two canonical roots in u1 and one in u0. Duplicate members don't count.
        packet["source_units"][1]["canonical_probes"].append({"opaque_probe_id": "p1-second",
            "opaque_duplicate_cluster_members": ["p1-second", "p1-second-copy"]})
        seal(packet)
        membership = _final_membership([packet], [], [])
        records = [{"id": pid, "opaque_probe_id": pid, "event": meta["event"], "role": "old",
                    "unit_id": meta["source_group_id"]} for pid, meta in membership["record_mappings"].items()
                   if meta["is_canonical_duplicate_root"]]
        binding = _bind_prepared_roots(records, membership)
        lanes = [{"id": "synthetic", "source_unit_ids": ["u0", "u1"], "source_group_ids": ["g0"]}]
        old = next(d for d in balance_disclosures(lanes, binding, membership) if d["role"] == "old")
        group = old["groups"][0]
        self.assertEqual(group["scored_canonical_roots_per_lane_unit"], {"u0": 1, "u1": 2})
        self.assertEqual(group["all_group_scored_canonical_roots"], 4)
        self.assertEqual(group["unchanged_primary_group_weights"], {"u0": .25, "u1": .5})
        self.assertFalse(group["pair_label_swap_disclosures"][0]["equal_count_label_swap_invariant"])
        self.assertFalse(old["primary_reweighted"])
        # Control primary aggregates by raw unit, even inside a merged source
        # group. Its unequal root counts do not imply unequal unit weights.
        controls = [{**r, "role": "control", "unit_id": membership["record_mappings"][r["opaque_probe_id"]]["source_unit_ids"][0]}
                    for r in records]
        control_binding = _bind_prepared_roots(controls, membership)
        control = next(d for d in balance_disclosures(lanes, control_binding, membership) if d["role"] == "control")
        group = control["groups"][0]
        self.assertEqual(group["primary_aggregation_unit_ids"], {"u0": "u0", "u1": "u1"})
        self.assertEqual(group["unchanged_primary_group_weights"], {"u0": 1, "u1": 1})
        self.assertEqual(group["unchanged_primary_event_weights"], {"u0": 1/3, "u1": 1/3})
        self.assertIsNone(group["pair_label_swap_disclosures"][0]["equal_count_label_swap_invariant"])
        self.assertEqual(group["pair_label_swap_disclosures"][0]["status"], "different_primary_aggregation_units")

    def test_lane_only_complement_uses_narrow_units_and_explicit_A1(self):
        mapping = {"lane": {"source_unit_ids": ["u0"], "event": "e", "source_group_id": "g0",
                            "dependency_block_id": "b", "duplicate_cluster_id": "p0"},
                   "broader": {"source_unit_ids": ["u2"], "event": "e", "source_group_id": "g2",
                               "dependency_block_id": "b", "duplicate_cluster_id": "p2"},
                   "merged": {"source_unit_ids": ["u0", "u2"], "event": "e", "source_group_id": "g0",
                              "dependency_block_id": "b", "duplicate_cluster_id": "t"}}
        def rows(a):
            return [{"id": rid, "unit_id": meta["source_group_id"], "event": "e", "role": "old",
                     "exact_match": value} for rid, meta, value in
                    (("lane", mapping["lane"], a), ("broader", mapping["broader"], 1-a), ("merged", mapping["merged"], 1-a))]
        reps = [{"replicate_id": str(i), "a_rows": rows(1), "b_rows": rows(0), "mapping": mapping,
                 "primary_a": .25, "primary_b": .75} for i in range(2)]
        report = complement_contrast(reps, [{"source_unit_ids": ["u0"]}], contrast="H2", endpoint="final",
                                     role="old", variant="canonical", metric="exact_match")
        self.assertEqual(report["lane_only_complement"]["estimate"], 1)
        self.assertTrue(report["primary_secondary_divergence"]["mean_sign_differs"])
        self.assertEqual(report["replicate_scores"][0]["a"]["rows_excluded"], 2)
        empty = complement_contrast(reps, [], contrast="H2", endpoint="final", role="old", variant="canonical", metric="exact_match")
        self.assertIsNone(empty["lane_only_complement"]["estimate"])
        self.assertEqual(empty["lane_only_complement"]["missing_secondary_pair_ids"], ["0", "1"])

    def test_reporting_policy_is_separately_sealed_without_changing_V3R2(self):
        from spacing_rerun.confirmation_dependency import algorithm_v3, SUPPORTED_ALGORITHM_SHA256
        policy = reporting_policy()
        self.assertEqual(policy["sha256"], digest({k: v for k, v in policy.items() if k != "sha256"}))
        self.assertEqual(digest(algorithm_v3()), SUPPORTED_ALGORITHM_SHA256)
        self.assertFalse(policy["membership_approval_claimed"])


if __name__ == "__main__":
    unittest.main()
