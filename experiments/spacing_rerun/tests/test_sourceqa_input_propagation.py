"""Synthetic engineering integration only; no real source or policy approval.

The existing Source fixtures remain unchanged. Derived fixtures bind their
explicit synthetic dependency context and reuse the original synthetic custody
construction. No represented trace command, Session, model or driver run executes.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from spacing_rerun.common import digest
from spacing_rerun import confirmation_dependency as dependency
from spacing_rerun.confirmation import source_review_content, verify_confirmation_dependency_preregistration
from spacing_rerun.confirmation_lane import reporting_specification, validate_lane_inputs
from spacing_rerun.source_patch import normalize_task, reconstruct_question_patch
from spacing_rerun.sourceqa_protocol import verify_protocol_coverage
from spacing_rerun.sourceqa_input_context import (
    add_sourceqa_input_arguments, checked_input_descriptor, load_sourceqa_inputs,
    sourceqa_descriptors_from_arguments, sourceqa_validator_keywords,
)
from spacing_rerun import sourceqa_input_context
from test_confirmation import confirmation_fixture, seal
from test_confirmation_dependency import final_fixture, final_review_fixture
from test_source_patch import proof_fixture, refresh, native_trace_fixture
from test_sourceqa_protocol import coverage_fixture, trusted_evaluator_fixture


def retained_json(root, name, value):
    path = root / name
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return dict(path=str(path), file_sha256=hashlib.sha256(raw).hexdigest(), content_sha256=value["sha256"])


def separate_custody_fixture(current=None):
    """Derive the unchanged Source test's labelled synthetic contract/helper."""
    evidence = proof_fixture()
    if current is not None:
        # This is a new synthetic scope derivation, not a restamp of any
        # retained original fixture or proof. Keep all real source inputs out.
        base = copy.deepcopy(current["authored"])
        selected = copy.deepcopy(base["acquisition_records"][:1])
        selected[0]["question_indices_to_revise"] = [0]
        packet = evidence["input_packet"]
        packet.update(current_authored_sha256=base["sha256"], selected_acquisition_records=selected,
            **{key: base[key] for key in ("source_catalog_sha256", "source_author_packet_sha256", "dataset_revision",
                                        "chunk_index", "events", "groups", "source_units")})
        seal(packet)
        author = evidence["patch"]["revision_author"]
        revised = copy.deepcopy(packet)
        revised.update(prior_authored_sha256=base["sha256"], revision_author=author)
        revised["selected_acquisition_records"][0]["questions"][0] = "Give the recorded source entry for event zero?"
        seal(revised)
        evidence.update(base_authored=base, patch=revised)
        evidence["response"]["prior_authored_sha256"] = base["sha256"]
        refresh(evidence)
    info, status = evidence["artifact_files"]["patch"], evidence["actual_task_status"]
    helper_text, helper_path = "Synthetic retained helper bytes; not executed.\n", "/tmp/synthetic-original-helper.py"
    contract = seal(dict(schema="synthetic-original-author-contract", patch_output_path=info["path"],
        authorized_offline_packing_script_path=helper_path,
        authorized_offline_packing_script_file_sha256=hashlib.sha256(helper_text.encode()).hexdigest()))
    contract_text, contract_path = json.dumps(contract), "/tmp/synthetic-original-contract.json"
    declared = dict(path=contract_path, sha256=contract["sha256"], file_sha256=hashlib.sha256(contract_text.encode()).hexdigest())
    evidence["response"]["author_output_contract"] = declared
    refresh(evidence)
    info = evidence["artifact_files"]["patch"]
    status["latestTerminalSummary"] = "Completed original patch " + info["path"]
    seal(evidence["actual_task_status_inventory"])
    text = json.dumps(evidence["actual_task_status_inventory"], sort_keys=True, indent=2)
    evidence["artifact_files"]["actual_task_status_inventory"].update(utf8=text,
        file_sha256=hashlib.sha256(text.encode()).hexdigest(), canonical_sha256=evidence["actual_task_status_inventory"]["sha256"])
    evidence["task_record"] = normalize_task(status, evidence["patch"]["revision_author"]["task_id"])
    trace = native_trace_fixture(status, evidence["patch"], info)
    item = trace["items"][0]
    boundary = item["text"].rindex("\n{")
    command = shlex.split(item["text"][:boundary].removeprefix("$ "))[2]
    replacement = ("contract_path=Path(" + repr(contract_path) + ")\ncontract=json.loads(contract_path.read_bytes())\n"
        "helper=Path(contract['authorized_offline_packing_script_path'])\n"
        "assert hashlib.sha256(helper.read_bytes()).hexdigest()==contract['authorized_offline_packing_script_file_sha256']\n"
        "patch_path=Path(contract['patch_output_path'])")
    item["command"] = command.replace("patch_path=Path(" + repr(info["path"]) + ")", replacement)
    item["stdout"] = item["text"][boundary + 1:]
    item.pop("text")
    text = json.dumps(dict(isError=False, structuredContent=trace, content=[dict(type="text", text=json.dumps(trace))]))
    evidence["completion_trace_file"] = dict(utf8=text, file_sha256=hashlib.sha256(text.encode()).hexdigest(), trace_json_pointer="/structuredContent")
    seal(evidence)
    inputs = dict(author_output_contract=dict(path=contract_path, utf8=contract_text,
        file_sha256=declared["file_sha256"], canonical_sha256=declared["sha256"]),
        packing_helper=dict(path=helper_path, utf8=helper_text, file_sha256=contract["authorized_offline_packing_script_file_sha256"]))
    if current is None:
        _, audit = confirmation_fixture()
        current = audit["source_chunks"][0]
    current = copy.deepcopy(current)
    historical = copy.deepcopy(current)
    merged = reconstruct_question_patch(evidence, original_trace_input_files=inputs)
    row = dict(role="revision_author", identity=merged["revision_author"], authored_sha256=merged["sha256"],
        author_packet_sha256=merged["source_author_packet_sha256"], input_channel="source_only",
        task_id=evidence["task_record"]["task_id"], task_record=evidence["task_record"])
    historical.update(authored=merged, question_patch_evidence=evidence, author_task_provenance=[row], reviews=[])
    packet = historical["review_packet"]
    packet["authored_chunk_sha256"] = merged["sha256"]
    packet["acquisition_records"] = [source_review_content(r, "acquisition_records") for r in merged["acquisition_records"]]
    seal(packet)
    current["history"] = [historical]
    return current, historical, evidence, {evidence["sha256"]: inputs}


def adopted_fixture():
    """Only synthetic context placeholders change; original fixtures are retained."""
    base, original_chunks, original_catalog, source_evidence = final_fixture()
    protocol, chunks, catalog, original_proof = coverage_fixture()
    assert all(c["review_packet"] == original["review_packet"] for c, original in zip(chunks, original_chunks))
    assert catalog == original_catalog
    proof = copy.deepcopy(original_proof)
    context = dict(membership_sha256=base["membership"]["sha256"], exception_lanes_sha256=digest([]),
        required_source_conditions_sha256=digest(base["required_source_conditions"]),
        lane_reporting_policy_sha256=base["lane_reporting_policy"]["sha256"], algorithm_sha256=base["algorithm_sha256"])
    proof["dependency_context"] = context
    proof["evaluation_custody"]["dependency_context"] = context
    seal(proof)
    metadata = trusted_evaluator_fixture(chunks, catalog)
    prepared = []
    for row in base["canonical_bindings"]:
        records = [dict(id=rid, opaque_probe_id=meta["opaque_probe_id"], event=meta["event"], role=meta["role"],
            unit_id=meta["source_unit_ids"][0] if meta["role"] in ("qa", "control") else meta["source_group_id"])
            for maps in (row["binding"]["record_mappings"], row["binding"]["teaching_root_mappings"]) for rid, meta in maps.items()]
        prepared.append(dict(replicate_id=row["replicate_id"], manifest_sha256=row["manifest_sha256"],
            records=records, training_provenance=row["training_provenance"]))
    kwargs = dict(policy_authors=base["policy_authors"], required_source_conditions=base["required_source_conditions"],
        exception_protocol=protocol, protocol_coverage=proof)
    args = (chunks, catalog, [e["original_recommendation"] for e in base["membership"]["edges"]],
        base["membership"]["flagged_source_unit_ids"], prepared, source_evidence)
    policy = dependency.build_final_policy_input(*args, **kwargs, expected_evaluator_metadata=metadata)
    return policy, chunks, catalog, source_evidence, metadata, args, kwargs, original_proof


def separate_history_policy_fixture():
    base, chunks, catalog, source_evidence = final_fixture()
    chunks[0], _, evidence, mapping = separate_custody_fixture(chunks[0])
    prepared = []
    for row in base["canonical_bindings"]:
        records = [dict(id=rid, opaque_probe_id=meta["opaque_probe_id"], event=meta["event"], role=meta["role"],
            unit_id=meta["source_unit_ids"][0] if meta["role"] in ("qa", "control") else meta["source_group_id"])
            for maps in (row["binding"]["record_mappings"], row["binding"]["teaching_root_mappings"]) for rid, meta in maps.items()]
        prepared.append(dict(replicate_id=row["replicate_id"], manifest_sha256=row["manifest_sha256"], records=records,
            training_provenance=row["training_provenance"]))
    args = (chunks, catalog, [e["original_recommendation"] for e in base["membership"]["edges"]],
        base["membership"]["flagged_source_unit_ids"], prepared, source_evidence)
    kwargs = dict(policy_authors=base["policy_authors"], required_source_conditions=base["required_source_conditions"])
    policy = dependency.build_final_policy_input(*args, **kwargs, original_trace_input_files_by_proof=mapping)
    return policy, chunks, catalog, source_evidence, evidence, mapping, args, kwargs


class SourceQAInputPropagationTests(unittest.TestCase):
    def test_lane_caller_forwards_current_and_history_original_custody(self):
        current, historical, evidence, mapping = separate_custody_fixture()
        before = copy.deepcopy((current, historical, evidence, mapping))
        authors = [dict(agent_id="synthetic-policy-author", provider="synthetic", model="synthetic-model")]
        for chunk in (current, historical):
            self.assertEqual(validate_lane_inputs([], None, [], {}, [chunk], authors,
                original_trace_input_files_by_proof=mapping), [])
            with self.assertRaisesRegex(ValueError, "custody"):
                validate_lane_inputs([], None, [], {}, [chunk], authors)
        self.assertEqual((current, historical, evidence, mapping), before)

    def test_real_policy_freeze_and_saved_report_retain_separate_history_custody(self):
        policy, chunks, catalog, source_evidence, evidence, mapping, args, kwargs = separate_history_policy_fixture()
        before = copy.deepcopy((chunks, evidence, mapping))
        with self.assertRaisesRegex(ValueError, "custody"):
            dependency.build_final_policy_input(*args, **kwargs)
        receipt, actual_review = final_review_fixture(policy)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic-history-policy.json"
            frozen = dependency.freeze_final_policy(path, policy, chunks, catalog, receipt, actual_review, source_evidence,
                original_trace_input_files_by_proof=mapping)
            retained = path.read_bytes()
            dependency.verify_frozen_policy(path, frozen["sha256"], chunks, catalog,
                original_trace_input_files_by_proof=mapping)
            with self.assertRaisesRegex(ValueError, "custody"):
                dependency.verify_frozen_policy(path, frozen["sha256"], chunks, catalog)
            prereg = seal(dict(n=1, E=8, replicate_manifest_sha256=["a" * 64]))
            audit = seal(dict(schema="synthetic-separate-history-audit", source_chunks=chunks, source_catalog=catalog))
            bundles = [dict(manifest=dict(sha256="a" * 64, confirmation_audit=audit), completion={})]
            with self.assertRaisesRegex(ValueError, "Source-only dependency input seal differs"):
                dependency.build_saved_dependency_report(bundles, frozen, prereg, original_trace_input_files_by_proof=mapping)
            with self.assertRaisesRegex(ValueError, "custody"):
                dependency.build_saved_dependency_report(bundles, frozen, prereg)
            self.assertEqual(path.read_bytes(), retained)
        self.assertEqual((chunks, evidence, mapping), before)

    def test_cli_none_and_complete_separate_descriptors(self):
        parser = argparse.ArgumentParser()
        add_sourceqa_input_arguments(parser)
        self.assertIsNone(sourceqa_descriptors_from_arguments(parser.parse_args([])))
        with self.assertRaises(ValueError):
            sourceqa_descriptors_from_arguments(parser.parse_args(["--expected-evaluator-metadata", "/tmp/synthetic.json"]))
        flags = ["--expected-evaluator-metadata", "/tmp/synthetic.json", "--expected-evaluator-metadata-file-sha256", "a" * 64,
            "--expected-evaluator-metadata-content-sha256", "b" * 64]
        descriptors = sourceqa_descriptors_from_arguments(parser.parse_args(flags))
        self.assertEqual(descriptors["expected_evaluator_metadata"]["content_sha256"], "b" * 64)
        self.assertIsNone(descriptors["original_trace_custody"])
        self.assertEqual(sourceqa_validator_keywords(), {})
        inputs = load_sourceqa_inputs()
        self.assertEqual(inputs, dict(expected_evaluator_metadata=None, original_trace_input_files_by_proof=None))
        objects = dict(expected_evaluator_metadata={}, original_trace_input_files_by_proof={})
        routed = sourceqa_validator_keywords(**objects)
        for key in objects:
            self.assertIs(routed[key], objects[key])

    def test_descriptor_rejects_changed_bytes_seal_schema_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = seal(dict(schema="synthetic-descriptor"))
            ref = retained_json(root, "synthetic.json", value)
            self.assertEqual(checked_input_descriptor(ref), value)
            for changed in (dict(ref, file_sha256="0" * 64), dict(ref, content_sha256="0" * 64), dict(ref, approved=True),
                            dict(ref, path="relative.json")):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    checked_input_descriptor(changed)
            link = root / "symlink.json"
            link.symlink_to(ref["path"])
            with self.assertRaises(ValueError):
                checked_input_descriptor(dict(ref, path=str(link)))
            with self.assertRaises(ValueError):
                load_sourceqa_inputs(dict(expected_evaluator_metadata=ref, original_trace_custody=None))
            Path(ref["path"]).write_text(Path(ref["path"]).read_text() + " ")
            with self.assertRaises(ValueError):
                checked_input_descriptor(ref)

    def test_current_and_history_custody_reach_real_author_validator(self):
        current, historical, evidence, mapping = separate_custody_fixture()
        unchanged = copy.deepcopy((current, historical, evidence, mapping))
        for chunks in ([historical], [current]):
            self.assertTrue(dependency.source_author_channels(chunks, original_trace_input_files_by_proof=mapping))
            for bad in (None, {}, {"0" * 64: mapping[evidence["sha256"]]}):
                with self.subTest(history=bool(chunks[0].get("history")), bad=bad), self.assertRaisesRegex(ValueError, "custody"):
                    dependency.source_author_channels(chunks, original_trace_input_files_by_proof=bad)
            for name in ("author_output_contract", "packing_helper"):
                for change in ("missing", "path", "raw"):
                    bad = copy.deepcopy(mapping)
                    files = bad[evidence["sha256"]]
                    if change == "missing":
                        files.pop(name)
                    elif change == "path":
                        files[name]["path"] += ".changed"
                    else:
                        files[name]["utf8"] += " "
                    with self.subTest(name=name, change=change), self.assertRaises(ValueError):
                        dependency.source_author_channels(chunks, original_trace_input_files_by_proof=bad)
        self.assertEqual((current, historical, evidence, mapping), unchanged)

    def test_loaded_adapter_retains_mapping_and_reports_original_references(self):
        _, _, evidence, mapping = separate_custody_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = seal(dict(schema="spacing-separate-original-trace-caller-custody-v1",
                binding_proof_path="/tmp/synthetic-proof.json", binding_proof_file_sha256="a" * 64, binding_proof_sha256=evidence["sha256"],
                original_response_binding=dict(path="/tmp/synthetic-response.json", file_sha256="b" * 64, sha256="c" * 64),
                original_trace_input_files_by_proof=mapping, original_source_proof_trace_status_contract_helper_restamped=False,
                semantic_source_property_approval_created=False, source_commands_executed=False))
            ref = retained_json(root, "synthetic-custody.json", adapter)
            descriptors = dict(expected_evaluator_metadata=None, original_trace_custody=ref)
            actual_loaded = []
            original_reader = sourceqa_input_context.checked_input_descriptor
            def observe_original_reader(reference):
                value = original_reader(reference)
                actual_loaded.append(value)
                return value
            # Instrument the loader only; actual raw/seal checks and every
            # Source/scientific validator remain unchanged.
            with patch.object(sourceqa_input_context, "checked_input_descriptor", side_effect=observe_original_reader):
                inputs, refs = load_sourceqa_inputs(descriptors, include_references=True)
            self.assertEqual(len(actual_loaded), 1)
            self.assertIs(inputs["original_trace_input_files_by_proof"], actual_loaded[0]["original_trace_input_files_by_proof"])
            self.assertEqual(inputs["original_trace_input_files_by_proof"], mapping)
            routed = sourceqa_validator_keywords(**inputs)
            self.assertIs(routed["original_trace_input_files_by_proof"], inputs["original_trace_input_files_by_proof"])
            self.assertEqual({r["path"] for r in refs}, {ref["path"], adapter["binding_proof_path"],
                adapter["original_response_binding"]["path"], *(r["path"] for r in mapping[evidence["sha256"]].values())})
            self.assertEqual(Path(ref["path"]).read_bytes(), (json.dumps(adapter, sort_keys=True, indent=2) + "\n").encode())
            for flag in ("original_source_proof_trace_status_contract_helper_restamped", "semantic_source_property_approval_created", "source_commands_executed"):
                changed = copy.deepcopy(adapter)
                changed[flag] = True
                seal(changed)
                bad_ref = retained_json(root, flag + ".json", changed)
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    load_sourceqa_inputs(dict(descriptors, original_trace_custody=bad_ref))

    def test_loaded_metadata_is_forwarded_as_the_actual_parsed_object(self):
        _, chunks, catalog, _ = final_fixture()
        metadata = trusted_evaluator_fixture(chunks, catalog)
        with tempfile.TemporaryDirectory() as tmp:
            ref = retained_json(Path(tmp), "synthetic-independent-metadata.json", metadata)
            descriptors = dict(expected_evaluator_metadata=ref, original_trace_custody=None)
            loaded = []
            original_reader = sourceqa_input_context.checked_input_descriptor
            def observe(reference):
                value = original_reader(reference)
                loaded.append(value)
                return value
            with patch.object(sourceqa_input_context, "checked_input_descriptor", side_effect=observe):
                inputs, refs = load_sourceqa_inputs(descriptors, include_references=True)
            self.assertIs(inputs["expected_evaluator_metadata"], loaded[0])
            self.assertIs(sourceqa_validator_keywords(**inputs)["expected_evaluator_metadata"], loaded[0])
            self.assertEqual(refs, [ref, metadata["catalog_binding"], metadata["snapshot_binding"], metadata["provenance_binding"]])

    def test_adopted_policy_builder_requires_independent_metadata_and_exact_context(self):
        policy, chunks, catalog, _, metadata, args, kwargs, original = adopted_fixture()
        proof = kwargs["protocol_coverage"]
        unchanged = copy.deepcopy((args, kwargs, metadata, original))
        verify_protocol_coverage(proof, kwargs["exception_protocol"], chunks, catalog,
            dependency_context=proof["dependency_context"], expected_evaluator_metadata=metadata)
        self.assertEqual(policy["protocol_coverage"]["dependency_context"], proof["dependency_context"])
        with self.assertRaisesRegex(ValueError, "metadata|trusted|independent"):
            dependency.build_final_policy_input(*args, **kwargs)
        for field in ("configuration_sha256", "catalog_binding", "snapshot_binding", "provenance_binding", "scope", "required_surface_correlate_ids_by_record"):
            changed = copy.deepcopy(metadata)
            if field == "configuration_sha256":
                changed[field] = "0" * 64
            elif field.endswith("_binding"):
                changed[field]["file_sha256"] = "0" * 64
            elif field == "scope":
                changed[field]["source_catalog_sha256"] = "0" * 64
            else:
                changed[field][next(iter(changed[field]))] = ["synthetic-missing-correlate"]
            seal(changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                dependency.build_final_policy_input(*args, **kwargs, expected_evaluator_metadata=changed)
        wrong = copy.deepcopy(kwargs)
        wrong["protocol_coverage"]["dependency_context"]["membership_sha256"] = "0" * 64
        seal(wrong["protocol_coverage"])
        with self.assertRaises(ValueError):
            dependency.build_final_policy_input(*args, **wrong, expected_evaluator_metadata=metadata)
        self.assertEqual((args, kwargs, metadata, original), unchanged)

    def test_frozen_policy_and_preregistration_forward_real_coverage_inputs(self):
        policy, chunks, catalog, source_evidence, metadata, *_ = adopted_fixture()
        receipt, review_evidence = final_review_fixture(policy)
        validation = dependency.validate_final_policy(policy, chunks, catalog, receipt, review_evidence, source_evidence,
            expected_evaluator_metadata=metadata)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic-frozen-policy.json"
            frozen = dependency.freeze_final_policy(path, policy, chunks, catalog, receipt, review_evidence, source_evidence,
                expected_evaluator_metadata=metadata)
            before = path.read_bytes()
            actual = dependency.verify_frozen_policy(path, frozen["sha256"], chunks, catalog, expected_evaluator_metadata=metadata)
            self.assertEqual(actual["validation"], validation)
            prereg = dict(confirmation_dependency_policy_path=str(path), confirmation_dependency_policy_sha256=frozen["sha256"],
                sourceqa_protocol_sha256=policy["exception_protocol"]["sha256"],
                sourceqa_protocol_coverage_sha256=policy["protocol_coverage"]["sha256"],
                lane_reporting_policy_sha256=policy["lane_reporting_policy"]["sha256"],
                lane_reporting_specification_sha256=reporting_specification()["sha256"],
                frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            manifest = dict(sha256="a" * 64, confirmation_audit=dict(source_chunks=chunks, source_catalog=catalog), config={})
            verify_confirmation_dependency_preregistration(prereg, manifest, expected_evaluator_metadata=metadata)
            for call in (lambda: dependency.validate_final_policy(policy, chunks, catalog, receipt, review_evidence, source_evidence),
                         lambda: dependency.verify_frozen_policy(path, frozen["sha256"], chunks, catalog),
                         lambda: verify_confirmation_dependency_preregistration(prereg, manifest)):
                with self.assertRaises(ValueError):
                    call()
            self.assertEqual(path.read_bytes(), before)

    def test_saved_report_reaches_real_policy_guard_before_deliberate_completion_rejection(self):
        policy, chunks, catalog, source_evidence, metadata, *_ = adopted_fixture()
        receipt, review_evidence = final_review_fixture(policy)
        with tempfile.TemporaryDirectory() as tmp:
            frozen = dependency.freeze_final_policy(Path(tmp) / "synthetic.json", policy, chunks, catalog,
                receipt, review_evidence, source_evidence, expected_evaluator_metadata=metadata)
            prereg = seal(dict(n=1, E=8, replicate_manifest_sha256=["a" * 64],
                confirmation_dependency_policy_path="/tmp/synthetic-report-policy.json",
                confirmation_dependency_policy_sha256=frozen["sha256"],
                sourceqa_protocol_sha256=policy["exception_protocol"]["sha256"],
                sourceqa_protocol_coverage_sha256=policy["protocol_coverage"]["sha256"],
                lane_reporting_policy_sha256=policy["lane_reporting_policy"]["sha256"],
                lane_reporting_specification_sha256=reporting_specification()["sha256"]))
            audit = seal(dict(schema="synthetic-saved-audit", source_chunks=chunks, source_catalog=catalog))
            # Deliberately unsealed completion. This tests Source routing only,
            # and never creates successful saved-result/report evidence.
            bundles = [dict(manifest=dict(sha256="a" * 64, confirmation_audit=audit), completion={})]
            with self.assertRaisesRegex(ValueError, "Source-only dependency input seal differs"):
                dependency.build_saved_dependency_report(bundles, frozen, prereg, expected_evaluator_metadata=metadata)
            with self.assertRaisesRegex(ValueError, "metadata|trusted|independent"):
                dependency.build_saved_dependency_report(bundles, frozen, prereg)

    def test_driver_workers_reject_bad_external_descriptors_before_work(self):
        path = Path(__file__).resolve().parents[1] / "scripts/confirmation_analysis_performance.py"
        spec = importlib.util.spec_from_file_location("synthetic_sourceqa_driver_input_test", path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        for worker in (driver.worker_point, driver.worker_probe):
            with self.subTest(worker=worker.__name__), self.assertRaisesRegex(ValueError, "routing fields"):
                worker(dict(sourceqa_input_descriptors=dict(candidate_policy_claim=True)))

    def test_nonexecuted_training_prepared_analysis_and_cli_forwarding_structure(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "spacing_rerun/training.py": {"run_acquisition": {"load_prepared", "verify_preregistration", "Session"},
                "run_arm": {"load_prepared", "verify_preregistration", "Session"},
                "verify_preregistration": {"verify_confirmation_preregistration"}, "Session.__init__": {"load_prepared"}},
            "spacing_rerun/prepare.py": {"load_prepared": {"validate_confirmation_audit"}},
            "spacing_rerun/analysis.py": {"summarize_bundles": {"verify_confirmation_preregistration", "load_prepared",
                "validate_confirmation_audit", "verify_frozen_policy", "build_saved_dependency_report", "load_dependency_bundle"}},
            "spacing_rerun/confirmation.py": {"verify_confirmation_preregistration": {"validate_confirmation_audit", "verify_confirmation_dependency_preregistration"}},
            "scripts/confirmation_analysis_performance.py": {"generate_bundle": {"load_real_prepared"},
                "checkpoint_sample_binding": {"load_real_prepared"}, "observed_shape": {"load_real_prepared"},
                "worker_probe": {"load_real_prepared"}, "worker_point": {"build_saved_dependency_report"},
                "preflight": {"verify_frozen_policy", "load_real_prepared", "checkpoint_sample_binding"}},
        }
        def functions(tree, prefix=""):
            result = {}
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    result.update(functions(node, node.name + "."))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[prefix + node.name] = node
            return result
        def call_name(node):
            return node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
        for relative, owners in expected.items():
            nodes = functions(ast.parse((root / relative).read_text()))
            for owner, targets in owners.items():
                calls = [n for n in ast.walk(nodes[owner]) if isinstance(n, ast.Call) and call_name(n) in targets]
                self.assertEqual({call_name(n) for n in calls}, targets)
                for call in calls:
                    routed = [kw.value for kw in call.keywords if kw.arg is None and isinstance(kw.value, ast.Call)
                              and call_name(kw.value) == "sourceqa_validator_keywords"]
                    with self.subTest(file=relative, owner=owner, call=call_name(call)):
                        self.assertTrue(routed)
        cli = ast.parse((root / "spacing_rerun/__main__.py").read_text())
        names = [call_name(n) for n in ast.walk(cli) if isinstance(n, ast.Call)]
        self.assertIn("add_sourceqa_input_arguments", names)
        self.assertIn("load_sourceqa_inputs", names)
        self.assertIn("sourceqa_descriptors_from_arguments", names)


if __name__ == "__main__":
    unittest.main()
