"""Small CPU artifact tests. None creates policy approval or benchmark evidence."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import struct

from spacing_rerun.common import digest
from spacing_rerun.confirmation_dependency import _saved_evaluation
from experiments.spacing_rerun.tests.test_confirmation_dependency_reporting import reporting_fixture
from experiments.spacing_rerun.tests.test_confirmation_dependency import final_fixture, final_review_fixture

DRIVER_PATH = Path(__file__).resolve().parents[1] / "scripts/confirmation_analysis_performance.py"
spec = importlib.util.spec_from_file_location("confirmation_analysis_performance", DRIVER_PATH)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def bpf_decision(machine, syscall, arguments=(), *, before_exec=False):
    program = driver.execution.native_filter_program(machine, before_exec=before_exec)
    memory = {0: syscall, 4: {"x86_64":0xc000003e,"aarch64":0xc00000b7}[machine]}
    for i, value in enumerate(arguments):
        memory[16 + 8 * i] = value & 0xffffffff; memory[20 + 8 * i] = value >> 32
    pc = accumulator = 0
    for _ in range(1000):
        code, jt, jf, value = program[pc]
        if code == 0x20: accumulator = memory.get(value, 0); pc += 1
        elif code == 0x15: pc += 1 + (jt if accumulator == value else jf)
        elif code == 0x45: pc += 1 + (jt if accumulator & value else jf)
        elif code == 0x06: return value
        else: raise AssertionError("Unsupported BPF test instruction")
    raise AssertionError("BPF test did not terminate")


class AnalysisPerformanceArtifactTests(unittest.TestCase):
    def test_synthetic_evaluator_uses_exact_roots_and_real_aggregation(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, _, _, policy, _, _ = reporting_fixture(Path(name))
            binding = policy["canonical_bindings"][0]["binding"]
            first = driver.synthetic_evaluation(manifest, binding, policy["membership"], random.Random(19), 10)
            second = driver.synthetic_evaluation(manifest, binding, policy["membership"], random.Random(19), 10)
            self.assertEqual(first, second)
            self.assertFalse(first["production_approval"])
            self.assertFalse(first["scientific_outcome"])
            sections = _saved_evaluation(first, manifest, binding, policy["membership"])
            self.assertEqual(set(sections), {"transfer/canonical", "transfer/paraphrase", "acquisition_qa"})
            self.assertEqual(driver.count_prediction_rows(first), 10)
            first["variants"]["canonical"]["facts"][0]["loss"] += 1
            with self.assertRaisesRegex(ValueError, "aggregation"):
                _saved_evaluation(first, manifest, binding, policy["membership"])

    def test_loss_only_endpoints_keep_behavior_absent(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, _, _, policy, _, _ = reporting_fixture(Path(name))
            binding = policy["canonical_bindings"][0]["binding"]
            value = driver.synthetic_evaluation(manifest, binding, policy["membership"], random.Random(23), 11, behavior=False)
            self.assertNotIn("acquisition_qa_diagnostic", value)
            for section in value["variants"].values():
                for row in section["facts"]:
                    self.assertNotIn("exact_match", row)
                    self.assertNotIn("prediction", row)
            _saved_evaluation(value, manifest, binding, policy["membership"])

    def test_output_overlap_and_symlink_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            production = root / "production"
            production.mkdir()
            for output in (production / "fake", root):
                with self.assertRaisesRegex(ValueError, "outside"):
                    driver.isolate_output(str(output), [str(production)])
            link = root / "link"
            link.symlink_to(production, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "canonical"):
                driver.isolate_output(str(link / "fake"), [str(production)])
            accepted, _ = driver.isolate_output(str(root / "synthetic"), [str(production)])
            self.assertEqual(accepted.path, root / "synthetic")
            accepted.close()

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            output = root / "synthetic"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "new"):
                driver.isolate_output(str(output), [str(root / "production")])

    def checkpoint_fixture(self, root):
        files = []
        for role in ("stage1", "NONE", "UNI", "EXP", "MASS", "GEN"):
            file = root / (role + ".pt")
            file.write_bytes(("CPU test artifact only " + role).encode())
            files.append(dict(digest_slot=role, sample_stage="small_cpu_test_artifact", bytes=file.stat().st_size,
                              **driver.raw_ref(file)))
        ref = dict(path=str(root), manifest_sha256=digest("test-metadata-only"))
        return [ref], [dict(manifest_sha256=ref["manifest_sha256"], files=files)]

    def test_checkpoint_sample_files_cannot_be_missing_and_slots_must_be_complete(self):
        with tempfile.TemporaryDirectory() as name:
            prepared, checkpoints = self.checkpoint_fixture(Path(name).resolve())
            self.assertEqual(driver.checkpoint_inputs(checkpoints, prepared), checkpoints)
            Path(checkpoints[0]["files"][0]["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "absent"):
                driver.checkpoint_inputs(checkpoints, prepared)
            checkpoints[0]["files"][0] = dict(checkpoints[0]["files"][1])
            with self.assertRaisesRegex(ValueError, "six real digest"):
                driver.checkpoint_inputs(checkpoints, prepared)

    def test_actual_checkpoint_probe_hashes_bytes_and_rejects_change(self):
        with tempfile.TemporaryDirectory() as name:
            prepared, checkpoints = self.checkpoint_fixture(Path(name).resolve())
            payload = dict(kind="checkpoint_digest", prepared=prepared, checkpoints=checkpoints,
                           checkpoint_sample=dict(prepared=prepared[0]))
            value = driver.worker_probe(payload)
            self.assertEqual(value["operation_count"], 6)
            self.assertTrue(all(row["observed_sha256"] == row["file_sha256"] for row in value["observations"]))
            Path(checkpoints[0]["files"][0]["path"]).write_bytes(b"changed CPU test artifact")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes differ|actual checkpoint sample size"):
                driver.worker_probe(payload)

    def test_reused_physical_checkpoint_still_receives_six_actual_digest_calls(self):
        with tempfile.TemporaryDirectory() as name:
            prepared, checkpoints = self.checkpoint_fixture(Path(name).resolve())
            original = checkpoints[0]["files"][0]
            checkpoints[0]["files"] = [dict(original, digest_slot=slot) for slot in
                                        ("stage1", "NONE", "UNI", "EXP", "MASS", "GEN")]
            driver.checkpoint_inputs(checkpoints, prepared)
            value = driver.worker_probe(dict(kind="checkpoint_digest", prepared=prepared, checkpoints=checkpoints,
                                            checkpoint_sample=dict(prepared=prepared[0])))
            self.assertEqual(value["operation_count"], 6)
            self.assertEqual(len({row["path"] for row in value["observations"]}), 1)
            self.assertTrue(all(row["sample_manifest_sha256"] == prepared[0]["manifest_sha256"] for row in value["observations"]))

    def test_archive_shape_reads_directory_without_loading_pickle_or_tensor_values(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / "synthetic-archive-fixture.pt"
            path.write_bytes(b"not a checkpoint")
            with self.assertRaisesRegex(ValueError, "not a torch-save"):
                driver.checkpoint_archive_shape(path)
            with zipfile.ZipFile(path, "w") as archive:
                # Invalid pickle bytes prove the shape reader never deserializes.
                archive.writestr("fixture/data.pkl", b"invalid pickle, CPU archive test only")
                archive.writestr("fixture/version", b"3\n")
                archive.writestr("fixture/data/0", b"small synthetic storage bytes")
            value = driver.checkpoint_archive_shape(path)
            self.assertEqual(value["bytes"], path.stat().st_size)
            self.assertEqual(value["storage_entries"], [dict(archive_member="fixture/data/0", bytes=29, compressed_bytes=29)])

    def test_worker_measurement_retains_actual_logs_and_memory_peaks(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            prepared, checkpoints = self.checkpoint_fixture(root)
            measured, refs = driver.execute_worker(dict(kind="checkpoint_digest", prepared=prepared,
                checkpoints=checkpoints, checkpoint_sample=dict(prepared=prepared[0])), root / "cpu-artifact-probe")
            self.assertFalse(measured["scientific_outcome"])
            self.assertGreater(measured["wall_seconds"], 0)
            self.assertGreater(measured["child_peak_rss_bytes"], 0)
            self.assertEqual(measured["peak_rss_bytes"], measured["child_peak_rss_bytes"] + measured["parent_peak_rss_bytes"])
            self.assertEqual(measured["worker"]["result"]["operation_count"], 6)
            invocation=json.loads((root/"cpu-artifact-probe.invocation.json").read_bytes())
            channels=measured["native_process_observation"]["inherited_channel_identities"]
            self.assertEqual(channels,invocation["inherited_channel_identities"])
            self.assertEqual(set(channels),{"invocation","producer","preexec_receipt","custody_journal"})
            self.assertTrue(all(row["device"]==root.stat().st_dev for row in channels.values()))
            self.assertEqual(len({row["inode"] for row in channels.values()}),4)
            self.assertTrue(all(driver.raw_ref(ref["path"]) == ref for ref in refs))

    def test_fixture_preregistration_never_approves_a_different_n(self):
        refs = [dict(manifest_sha256=digest("one")), dict(manifest_sha256=digest("two"))]
        prereg = driver.sealed(dict(driver.MARKERS, n=2, E=4, primary_delay=84,
            replicate_manifest_sha256=[ref["manifest_sha256"] for ref in refs]))
        driver.check_fixture_preregistration(prereg, refs)
        with self.assertRaisesRegex(ValueError, "exact first-n"):
            driver.check_fixture_preregistration(prereg, refs[:1])
        changed = dict(prereg, production_approval=True)
        changed = driver.sealed({k: v for k, v in changed.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "exact first-n"):
            driver.check_fixture_preregistration(changed, refs)

    def test_completion_boundary_pins_synthetic_files_without_approvals(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            file = root / "synthetic.json"
            driver.write_json(file, dict(driver.MARKERS, predictions="synthetic_seeded_random"))
            prereg = driver.sealed(dict(driver.MARKERS, E=4))
            completion = driver.completion_seal(root, dict(sha256=digest("fixture")), prereg)
            driver.check_seal(completion)
            self.assertFalse(completion["production_approval"])
            self.assertFalse(completion["scientific_outcome"])
            self.assertEqual(completion["files"], [dict(path=str(file), sha256=driver.file_hash(file))])

    def probe_fixture(self, root):
        prepared, checkpoints = self.checkpoint_fixture(root)
        return dict(kind="checkpoint_digest", prepared=prepared, checkpoints=checkpoints,
                    checkpoint_sample=dict(prepared=prepared[0]))

    def test_real_child_result_stdout_and_input_tamper_rejects_with_partial_evidence(self):
        original = driver.execution.launch_native
        for artifact in ("result", "stdout", "input"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve()
                payload = self.probe_fixture(root)
                def altered(invocation, anchor, log_path):
                    producer, native = original(invocation, anchor, log_path)
                    if artifact == "result":
                        path = Path(invocation["payload"]["worker_result_path"])
                        value = json.loads(path.read_bytes()); value["result"]["operation_count"] = 192
                        path.write_text(json.dumps(value))
                    elif artifact == "stdout":
                        Path(log_path).write_text('{"contradictory_stdout":true}\n')
                    else:
                        Path(invocation["input_path"]).write_text('{}\n')
                    return producer, native
                with patch.object(driver.execution, "launch_native", side_effect=altered), self.assertRaises(ValueError):
                    driver.execute_worker(payload, root / "tampered")
                failures = list(root.glob("failure-*.json"))
                self.assertEqual(len(failures), 1)
                failure = json.loads(failures[0].read_bytes())
                self.assertFalse(failure["retained_files_deleted"])
                self.assertTrue((root / "tampered.log").is_file())

    def test_exact_worker_schema_runtime_nonce_count_and_maximum_are_reconciled(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            measured, _ = driver.execute_worker(self.probe_fixture(root), root / "actual")
            invocation = json.loads((root / "actual.invocation.json").read_bytes())
            worker = measured["worker"]
            native = measured["native_process_observation"]
            mutations = [lambda v: v.update(nonce="different-invocation"),
                         lambda v: v.update(deployment_sha256="different-source"),
                         lambda v: v["observed_runtime"].update(node="different-node"),
                         lambda v: v["result"].update(operation_count=192),
                         lambda v: v["result"].update(wall_seconds_per_operation=1e-12),
                         lambda v: v["result"]["observations"][0].update(observed_sha256="wrong-digest"),
                         lambda v: v["loaded_code_identity"]["buffered_package_modules"].update(__main__="wrong-driver"),
                         lambda v: v["loaded_code_identity"]["compiler"].update(optimize=999),
                         lambda v: v["loaded_code_identity"].update(lifetime_runtime_imports=[]),
                         lambda v: v["loaded_code_identity"].update(native_process_enforcement={"install_result":0})]
            for mutate in mutations:
                changed = copy.deepcopy(worker); mutate(changed)
                with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                    driver.validate_worker_result(invocation["payload"], changed, invocation, native)

    def test_parent_and_nested_directory_swap_cannot_redirect_creation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            parent, production = root / "synthetic-parent", root / "production"
            parent.mkdir(); production.mkdir()
            anchor, _ = driver.isolate_output(str(parent / "output"), [str(production)])
            moved = root / "moved-parent"; parent.rename(moved); parent.symlink_to(production, target_is_directory=True)
            try:
                with self.assertRaises((ValueError, OSError)):
                    anchor.activate()
                self.assertFalse((production / "output").exists())
            finally:
                anchor.close()
            nested_root = root / "anchored"; nested_root.mkdir()
            anchor = driver.execution.OutputAnchor(nested_root, new=False)
            nested = nested_root / "nested"; nested.mkdir(); nested.rename(nested_root / "retained-nested")
            nested.symlink_to(production, target_is_directory=True)
            try:
                with self.assertRaises(OSError):
                    anchor.write(nested / "synthetic.json", b"labelled fixture")
                anchor.retain_failure(ValueError("nested fixture replaced"))
                self.assertFalse((production / "synthetic.json").exists())
                self.assertEqual(len(list(nested_root.glob("failure-*.json"))), 1)
            finally:
                anchor.close()

    def test_active_root_swap_retains_failure_in_original_inode(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); output = root / "output"; output.mkdir()
            production = root / "production"; production.mkdir()
            anchor = driver.execution.OutputAnchor(output, new=False)
            retained = root / "retained-output"; output.rename(retained); output.symlink_to(production, target_is_directory=True)
            try:
                with self.assertRaises(OSError):
                    anchor.write(output / "fake.json", b"fixture")
                anchor.retain_failure(ValueError("original root pathname replaced"))
                self.assertFalse(list(production.iterdir()))
                self.assertEqual(len(list(retained.glob("failure-*.json"))), 1)
            finally:
                anchor.close()

    def test_source_mutation_is_rejected_without_editing_owner_source(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); source = root / "driver-copy.py"
            source.write_bytes(DRIVER_PATH.read_bytes())
            copied_spec = importlib.util.spec_from_file_location("copied_driver", source)
            copied = importlib.util.module_from_spec(copied_spec)
            # The copied file needs only its import-time source identity; it is
            # never launched and no package/model validator is substituted.
            copied_spec.loader.exec_module(copied)
            source.write_bytes(source.read_bytes() + b"\n# scratch mutation\n")
            with self.assertRaisesRegex(ValueError, "changed after parent import"):
                copied.execute_worker(self.probe_fixture(root), root / "probe")
            self.assertEqual(len(list(root.glob("failure-*.json"))), 1)

    def test_captured_source_buffer_survives_caller_mutation_and_completion_rejects_it(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); package = root / "spacing_rerun"; package.mkdir()
            module = package / "__init__.py"; module.write_text("VALUE = 17\n")
            driver_copy = root / "driver.py"; driver_copy.write_text("print('CPU source capture fixture')\n")
            deployment = driver.execution.capture_sources(driver_copy, package)
            row = next(r for r in deployment["files"] if r["path"] == str(module))
            module.write_text("VALUE = 99\n")
            namespace = {}; exec(compile(driver.execution.base64.b64decode(row["data"]), str(module), "exec"), namespace)
            self.assertEqual(namespace["VALUE"], 17)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                driver.execution.verify_sources(deployment)

    def test_wait4_peak_belongs_to_this_child_after_an_earlier_larger_child(self):
        # The earlier child is an ordinary byte allocation, never a model.
        subprocess.run([sys.executable, "-I", "-c", "data=bytearray(96*1024*1024);data[0]=1"], check=True)
        previous_peak = driver.execution.rss_bytes(resource.getrusage(resource.RUSAGE_CHILDREN))
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            measured, _ = driver.execute_worker(self.probe_fixture(root), root / "small-child")
            self.assertLess(measured["child_peak_rss_bytes"], previous_peak)
            self.assertEqual(measured["child_peak_rss_bytes"], measured["native_process_observation"]["through_exit_child_peak_rss_bytes"])
            self.assertIn("through_exit", measured["memory_method"])

    def test_shared_json_graphs_are_retained_concurrently_and_not_claimed_as_rss(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); refs = []
            for i in range(3):
                path = root / (str(i) + ".json"); path.write_text(json.dumps({"values": list(range(100 + i))}))
                refs.append(driver.raw_ref(path))
            value = driver.observed_shape([], [], refs)
            self.assertEqual(value["peak_live_shared_json_graphs"], 3)
            self.assertEqual(value["peak_live_shared_serialized_bytes"], sum(Path(r["path"]).stat().st_size for r in refs))
            self.assertEqual(value["cumulative_parsed_json_bytes"], value["peak_live_shared_serialized_bytes"])
            self.assertNotIn("peak_rss_bytes", value)

    def test_native_saved_scalars_generation_mcq_and_prediction_bound(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, _, _, policy, _, _ = reporting_fixture(Path(name))
            envelope = dict(max_answer_tokens=13, maximum_prediction_utf8_bytes=96, token_id_upper_bound=999999)
            value = driver.synthetic_evaluation(manifest, policy["canonical_bindings"][0]["binding"],
                policy["membership"], random.Random(4), 10, envelope=envelope)
            for variant, section in value["variants"].items():
                for row in section["facts"]:
                    self.assertAlmostEqual(row["answer_nll_sum"] / row["answer_tokens"], row["loss"])
                    self.assertAlmostEqual((row["answer_nll_sum"] + row["eos_nll"]) / (row["answer_tokens"] + 1), row["loss_with_eos"])
                    self.assertEqual(len(row["prediction"].encode()), 96)
                    self.assertEqual(len(row["prediction_token_ids"]), 13)
                    self.assertIn("generation_terminated", row); self.assertIn("stop_token", row)
                    self.assertEqual("mcq_accuracy" in row, variant == "canonical")
                    if variant == "canonical":
                        self.assertIn("mcq_accuracy_normalized", row)
                        self.assertIn("candidate_mean_log_likelihoods", row)
            self.assertEqual(value["saved_string_envelope"], envelope)
            self.assertIn("metric_schema", value); self.assertIn("timings", value)
            with self.assertRaisesRegex(ValueError, "smaller"):
                driver.synthetic_metrics(random.Random(1), "long native label", True,
                    envelope=dict(max_answer_tokens=1, maximum_prediction_utf8_bytes=1, token_id_upper_bound=1))

    def test_native_supervisor_rss_includes_allocations_after_a_preexit_sample(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); package = root / "spacing_rerun"; package.mkdir()
            (package / "__init__.py").write_text("# labelled CPU supervision fixture\n")
            script = root / "late-allocation.py"
            script.write_text("import json,os,resource\n"
                "before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
                "later=bytearray(48*1024*1024)\n"
                "later[0]=1\n"
                "data=json.dumps({'preexit_raw_rss':before,'labelled_cpu_artifact_only':True}).encode()\n"
                "os.write(BUFFERED_INVOCATION['producer_channel_fd'],data)\n")
            deployment = driver.execution.capture_sources(script, package)
            anchor = driver.execution.OutputAnchor(root, new=False)
            try:
                retained = driver.deploy_buffers(deployment, anchor)
                invocation = dict(deployment=deployment, deployment_root=str(retained), input_path=str(root / "unused-input.json"))
                producer, native = driver.execution.launch_native(invocation, anchor, root / "late.log")
                before = json.loads(producer)["preexit_raw_rss"] * (1 if sys.platform == "darwin" else 1024)
                self.assertEqual(native["exit_code"], 0)
                self.assertGreater(native["through_exit_child_peak_rss_bytes"], before + 24 * 1024 * 1024)
            finally:
                anchor.close()

    def test_owner_adapter_completed_transport_rejects_stale_missing_and_contradictory_evidence(self):
        policy, _, _, _ = final_fixture()
        receipt, evidence = final_review_fixture(policy)
        evidence["task_status"]["workState"] = "result_available"
        evidence["task_status_file"]["utf8"] = json.dumps({"task_statuses": [
            {"clientRequestId": evidence["client_request_id"], "result": evidence["task_status"]}]})
        evidence["task_status_file"]["file_sha256"] = driver.execution.sha(evidence["task_status_file"]["utf8"].encode())
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); report_path, status_path = root / "review.json", root / "status.json"
            report_path.write_text(evidence["receipt_file"]["utf8"])
            status_path.write_text(evidence["task_status_file"]["utf8"])
            binding = dict(report=dict(driver.raw_ref(report_path), content_sha256=receipt["sha256"]),
                actual_task_status=driver.raw_ref(status_path), completed_review_evidence=evidence)
            completed = driver.verify_adapter_review(binding, receipt)
            self.assertEqual(completed["child_run_id"], evidence["task_status"]["childRunId"])
            missing = dict(binding); missing.pop("completed_review_evidence")
            with self.assertRaisesRegex(ValueError, "transport"):
                driver.verify_adapter_review(missing, receipt)
            for key, value in (("latestTerminalStatus", "failed"), ("hasPendingChildRuns", True),
                               ("latestTerminalRunId", "stale-run"), ("workState", "running"),
                               ("latestTerminalSummary", "parent says review done")):
                changed = copy.deepcopy(binding); status = changed["completed_review_evidence"]["task_status"]
                status[key] = value
                raw = json.dumps({"task_statuses": [{"clientRequestId": evidence["client_request_id"], "result": status}]})
                status_path.write_text(raw)
                changed["actual_task_status"] = driver.raw_ref(status_path)
                changed["completed_review_evidence"]["task_status_file"] = dict(utf8=raw, file_sha256=driver.execution.sha(raw.encode()))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    driver.verify_adapter_review(changed, receipt)
            # Hashes of contradictory current files cannot replace original transport.
            status_path.write_text('{}\n'); altered = dict(binding, actual_task_status=driver.raw_ref(status_path))
            with self.assertRaisesRegex(ValueError, "contradicts"):
                driver.verify_adapter_review(altered, receipt)

    def test_missing_checkpoint_owner_association_is_an_explicit_external_prerequisite(self):
        # Fail before any scientific prepared validator or archive read.
        with self.assertRaisesRegex(ValueError, "producer"):
            driver.checkpoint_sample_binding({}, {}, [])

    def test_captured_adapter_registers_dataclass_module_and_removes_it(self):
        raw = b"from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Row:\n    value:int\ndef cpu_fixture(payload):\n    return {'labelled_cpu_artifact_only':True,'value':Row(payload['value']).value}\n"
        checksum = driver.execution.sha(raw)
        value = driver.call_captured_adapter(raw, Path("/labelled/artifact-only.py"), checksum, {"value": 19}, "cpu_fixture")
        self.assertEqual(value, dict(labelled_cpu_artifact_only=True, value=19))
        self.assertNotIn("p4_exact_owner_adapter_" + checksum[:16], sys.modules)
        with self.assertRaisesRegex(ValueError, "buffer identity"):
            driver.call_captured_adapter(raw + b"# changed", Path("/labelled/artifact-only.py"), checksum, {}, "cpu_fixture")

    def test_native_exposure_envelope_preserves_rows_tokens_and_progress(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            manifest, schedule, examples, _, _, _ = reporting_fixture(root)
            rows = schedule["arms"]["NONE"][:2]
            path = root / "synthetic-exposures.jsonl"
            driver.save_exposures(path, rows, examples, 7, stage2_steps=1, manifest=manifest)
            values = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([v["examples"] for v in values], rows)
            self.assertEqual([v["global_step"] for v in values], [8, 9])
            self.assertEqual([v["cursor"] for v in values], [1, 2])
            self.assertEqual([v["phase"] for v in values], ["stage2", "buffer"])
            for value, row in zip(values, rows):
                self.assertFalse(value["production_approval"])
                self.assertEqual(value["row_sha256"], digest(row))
                self.assertEqual(value["loss_tokens"], sum(examples[k]["loss_tokens"] for k in row))
                self.assertEqual(value["example_content_sha256"], [examples[k]["content_sha256"] for k in row])
                self.assertTrue({"schema", "loss", "gradient_norm", "lr", "loss_weight", "content_tokens",
                                 "generic_filler_tokens", "padding_tokens", "acquisition_qa_tokens_total"}.issubset(value))

    def test_missing_builder_retained_projection_fails_before_source_loading(self):
        for evidence in ({}, {"schema": "p4-final-builder-live-memory-observation-v1", "n": 2,
                              "retained_same_report_object": True, "full_production_validation_projection": True}):
            with self.assertRaisesRegex(ValueError, "retained metadata"):
                driver.validate_builder_memory(evidence, dict(prepared=[{}, {}]), {})

    def test_worker_package_import_cannot_fall_back_to_uncaptured_disk_module(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); package = root / "spacing_rerun"; package.mkdir()
            (package / "__init__.py").write_text("# labelled source closure fixture\n")
            script = root / "import-extra.py"; script.write_text("import spacing_rerun.uncaptured\n")
            deployment = driver.execution.capture_sources(script, package)
            anchor = driver.execution.OutputAnchor(root, new=False)
            try:
                retained = driver.deploy_buffers(deployment, anchor)
                anchor.write(retained / "spacing_rerun/uncaptured.py", b"print('unexpected mutable disk import')\n")
                invocation = dict(deployment=deployment, deployment_root=str(retained), input_path=str(root / "unused-input.json"))
                _, native = driver.execution.launch_native(invocation, anchor, root / "import.log")
                self.assertNotEqual(native["exit_code"], 0)
                log = (root / "import.log").read_text()
                self.assertIn("outside captured source closure", log)
                self.assertNotIn("unexpected mutable disk import", log)
            finally:
                anchor.close()

    def test_nested_inode_move_after_parent_and_after_open_rejects_before_write(self):
        for boundary in ("parent", "opened"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve(); output = root / "output"; output.mkdir()
                production = root / "production"; production.mkdir()
                anchor = driver.execution.OutputAnchor(output, new=False)
                path = output / "nested" / "result.json"
                handle = None
                try:
                    if boundary == "opened": handle = anchor.open_exclusive(path, binary=True)
                    else:
                        parent, _ = anchor.parent(path); os.close(parent)
                    (output / "nested").rename(production / "moved")
                    (output / "nested").mkdir()
                    with self.assertRaisesRegex(ValueError, "nested directory inode chain"):
                        if handle: handle.write(b"must not be written")
                        else: anchor.open_exclusive(path, binary=True)
                    failure = anchor.failure_state(ValueError("fixture inode moved"))
                    custody = failure["retained_directory_custody"]["nested"]
                    self.assertFalse(custody["declared_path_matches_inode"])
                    self.assertEqual(custody["kernel_observed_fd_path"], str(production / "moved"))
                    if handle:
                        self.assertEqual((production / "moved" / "result.json").read_bytes(), b"")
                        self.assertEqual(failure["created_files"][0]["custody"]["kernel_observed_current_path"],
                                         str(production / "moved" / "result.json"))
                finally:
                    if handle: handle.handle.close()
                    anchor.close()

    def test_output_registry_retains_directories_without_one_descriptor_per_file(self):
        with tempfile.TemporaryDirectory() as name:
            anchor = driver.execution.OutputAnchor(Path(name).resolve(), new=False)
            try:
                for i in range(1100): anchor.write(anchor.path / "nested" / f"item-{i}.json", b"{}");
                self.assertEqual(len(anchor.created), 1100)
                self.assertEqual(len(anchor.directory_chains), 1)
                anchor.verify(complete=True)
            finally: anchor.close()

    def test_real_captured_imports_match_all_six_compiled_functions_and_reject_impostor(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); script = root / "identity-probe.py"
            script.write_text('''import base64,json,os,types,sys,performance_driver_runtime as runtime
row=next(r for r in BUFFERED_INVOCATION['deployment']['files'] if r['relative']=='scripts/confirmation_analysis_performance.py')
module=types.ModuleType('identity_review_driver');module.__file__=BUFFERED_INVOCATION['deployment_root']+'/'+row['relative']
exec(compile(base64.b64decode(row['data']),module.__file__,'exec',dont_inherit=True),module.__dict__)
from spacing_rerun import confirmation_dependency as dep,confirmation_lane as lane
expected=module.expected_function_codes(BUFFERED_INVOCATION['deployment'],BUFFERED_INVOCATION['deployment_root'])
actual={name:runtime.code_identity(getattr(lane if name=='complement_contrast' else dep,name).__code__) for name in module.FUNCTIONS}
counts,profile=module.profile_functions();sys.setprofile(profile)
try: dep.influence_scores([],{},'loss')
finally: sys.setprofile(None)
impostor=dep.influence_scores.__code__.replace(co_name='impostor')
out={'expected':expected,'actual':actual,'observations':counts.observations,'impostor_differs':runtime.code_sha(impostor)!=actual['influence_scores']['code_sha256']}
os.write(BUFFERED_INVOCATION['producer_channel_fd'],json.dumps(out).encode())
''')
            deployment = driver.execution.capture_sources(script, DRIVER_PATH.parents[1] / "spacing_rerun", extra=[DRIVER_PATH])
            anchor = driver.execution.OutputAnchor(root, new=False)
            try:
                retained = driver.deploy_buffers(deployment, anchor)
                data, native = driver.execution.launch_native(dict(deployment=deployment, deployment_root=str(retained),
                    input_path=str(root / "unused.json")), anchor, root / "identity.log")
                self.assertEqual(native["exit_code"], 0, (root / "identity.log").read_text())
                value = json.loads(data)
                self.assertEqual(set(value["actual"]), set(driver.FUNCTIONS))
                self.assertEqual(value["actual"], value["expected"])
                self.assertTrue(value["impostor_differs"])
                self.assertEqual(value["observations"], [dict(function="influence_scores", **value["expected"]["influence_scores"])])
            finally: anchor.close()

    def test_unapproved_transient_import_is_rejected_before_body_and_approved_lifetime_survives_deletion(self):
        for approved, direct in ((False, False), (False, True), (True, False)):
            with self.subTest(approved=approved, direct=direct), tempfile.TemporaryDirectory() as name:
                root = Path(name).resolve(); script = root / "transient.py"
                (root / "unapproved_transient.py").write_text("raise RuntimeError('unapproved module body executed')\n")
                script.write_text('''import os,sys,json,performance_driver_runtime as runtime
sys.path.insert(0,os.path.dirname(BUFFERED_INVOCATION['input_path']))
''' + ('''import fractions
del sys.modules['fractions']
out=runtime.loaded_identity(BUFFERED_INVOCATION['deployment'])
os.write(BUFFERED_INVOCATION['producer_channel_fd'],json.dumps(out).encode())
''' if approved else ('''import importlib.util
spec=importlib.util.spec_from_file_location('unapproved_transient',os.path.dirname(BUFFERED_INVOCATION['input_path'])+'/unapproved_transient.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
''' if direct else "import unapproved_transient\n")))
                deployment = driver.execution.capture_sources(script, DRIVER_PATH.parents[1] / "spacing_rerun")
                anchor = driver.execution.OutputAnchor(root, new=False)
                try:
                    retained = driver.deploy_buffers(deployment, anchor)
                    data, native = driver.execution.launch_native(dict(deployment=deployment, deployment_root=str(retained),
                        input_path=str(root / "unused.json")), anchor, root / "transient.log")
                    if approved:
                        self.assertEqual(native["exit_code"], 0, (root / "transient.log").read_text())
                        value = json.loads(data)
                        self.assertNotIn("fractions", [r["module"] for r in value["loaded_runtime_files"]])
                        rows = [r for r in value["lifetime_runtime_imports"] if r["module"] == "fractions"]
                        self.assertEqual([r["phase"] for r in rows], ["before_runtime_module_execution", "completed_runtime_module_execution"])
                        self.assertEqual(value["native_process_enforcement"]["scope"], "artifact_only")
                    else:
                        self.assertNotEqual(native["exit_code"], 0)
                        log = (root / "transient.log").read_text()
                        self.assertIn("outside captured interpreter roots", log)
                        self.assertNotIn("unapproved module body executed", log)
                finally: anchor.close()

    def test_same_physical_checkpoint_all_192_occurrences_are_read_and_bound(self):
        from spacing_rerun.acquisition_grid import checkpoint_digest
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve(); initial, inputs = self.checkpoint_fixture(root)
            pin = inputs[0]["files"][0]
            refs = [dict(initial[0], manifest_sha256=digest(str(i))) for i in range(32)]
            rows = [dict(manifest_sha256=r["manifest_sha256"], files=[dict(pin, digest_slot=slot) for slot in
                    ("stage1", "NONE", "UNI", "EXP", "MASS", "GEN")]) for r in refs]
            producer = dict(physical_files=[dict(pin, native_save_refs=[{"actual_artifact_test": True}],
                                                native_execution_refs=[{"actual_artifact_test": True}])])
            sample = dict(prepared=refs[0], native_producer_validation=producer,
                          operation_associations=driver.reconcile_checkpoint_associations(rows, producer))
            payload = dict(kind="checkpoint_digest", prepared=refs, checkpoints=rows, checkpoint_sample=sample)
            with patch("spacing_rerun.acquisition_grid.checkpoint_digest", wraps=checkpoint_digest) as read:
                result = driver.worker_probe(payload)
            self.assertEqual(read.call_count, 192); self.assertEqual(result["operation_count"], 192)
            self.assertEqual([r["native_operation_association"] for r in result["observations"]], sample["operation_associations"])
            for key, wrong in (("sample_stage", "contradictory"), ("file_sha256", "f" * 64),
                               ("bytes", pin["bytes"] + 1), ("native_save_refs", [{"different": True}])):
                mutated = copy.deepcopy(payload); mutated["checkpoints"][-1]["files"][-1][key] = wrong
                with self.subTest(key=key), self.assertRaises(ValueError): driver.worker_probe(mutated)
            mutated = copy.deepcopy(payload); mutated["checkpoint_sample"]["operation_associations"].reverse()
            with self.assertRaisesRegex(ValueError, "Ordered digest operation map"): driver.worker_probe(mutated)

    def test_exact_native_serializer_bounds_controls_backslashes_unicode_and_decoded_graph(self):
        from spacing_rerun.data import normalize
        envelope = dict(max_answer_tokens=13, maximum_prediction_utf8_bytes=96, token_id_upper_bound=999999)
        row = driver.synthetic_metrics(random.Random(4), "real gold", True, envelope=envelope, choices=4)
        encoded = lambda v: json.dumps(v, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
        self.assertEqual(int(normalize(row["prediction"]) == normalize("real gold")), row["exact_match"])
        for text in ("\\" * 96, "\x00" * 96, "\n" * 96, "\U00010000" * 24):
            native = {k: v for k, v in row.items() if not k.startswith("_synthetic_")}
            native["prediction"] = text
            native["loss"] = -1.2345678901234567e-200
            driver.scalar_serialization_reserve(row)
            self.assertGreaterEqual(len(encoded(row)), len(encoded(native)))
            decoded = json.loads(encoded(row))
            self.assertGreaterEqual(sys.getsizeof(decoded["_synthetic_decoded_prediction_surrogate"]), sys.getsizeof(text))
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / "native.json"; driver.write_json(path, row)
            self.assertEqual(path.read_bytes(), encoded(row))

    def test_metadata_conditional_generic_similarity_and_native_timing_shapes(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, _, _, policy, _, _ = reporting_fixture(Path(name))
            binding = policy["canonical_bindings"][0]["binding"]
            manifest["generic_eval"] = [[1,2,3]]
            old = next(rid for rid, meta in binding["record_mappings"].items() if meta["role"] == "old")
            manifest["acquisition_qa_similarity"] = dict(sha256=digest("test-only metadata similarity"), threshold=.5,
                probes=[dict(id=old, variant="canonical", max_jaccard=.25)])
            value = driver.synthetic_evaluation(manifest, binding, policy["membership"], random.Random(3), 0)
            self.assertIsNotNone(value["generic_loss"])
            self.assertEqual(value["acquisition_qa_similarity_diagnostic"]["low_similarity_probes"], 1)
            self.assertTrue({"canonical_loss_seconds", "paraphrase_loss_seconds", "canonical_generation_seconds",
                "paraphrase_generation_seconds", "mcq_seconds", "generic_seconds", "acquisition_qa_diagnostic_seconds"} <= set(value["timings"]))
            if manifest["config"].get("qa_teaching_diagnostics"): self.assertIn("qa_teaching_diagnostic_seconds", value["timings"])
            value = driver.synthetic_evaluation(manifest, binding, policy["membership"], random.Random(3), 0, behavior=False, generic=False)
            self.assertEqual(set(value["timings"]), {"canonical_loss_seconds", "paraphrase_loss_seconds"})
            self.assertIsNone(value["generic_loss"]); self.assertNotIn("acquisition_qa_similarity_diagnostic", value)

    def test_native_seccomp_program_decisions_without_claiming_kernel_execution(self):
        import errno
        def decision(machine, syscall, arguments=(), arch=None):
            program = driver.execution.native_filter_program(machine)
            memory = {0:syscall,4:arch or {"x86_64":0xc000003e,"aarch64":0xc00000b7}[machine]}
            memory.update({16 + i * 8:v for i,v in enumerate(arguments)})
            pc=0; accumulator=0
            for _ in range(200):
                code,jt,jf,k=program[pc]
                if code==0x20: accumulator=memory.get(k,0);pc+=1
                elif code==0x15: pc+=1+(jt if accumulator==k else jf)
                elif code==0x45: pc+=1+(jt if accumulator & k else jf)
                elif code==0x06: return k
                else: self.fail("unexpected BPF instruction")
            self.fail("filter did not terminate")
        reject=0x00050000|errno.EPERM; allow=0x7fff0000
        for machine, fork, clone, mmap, protect in (("x86_64",57,56,9,10),("aarch64",221,220,222,226)):
            self.assertEqual(decision(machine,fork),reject)
            self.assertEqual(decision(machine,clone,[0]),reject)
            self.assertEqual(decision(machine,clone,[0x10000]),allow)
            self.assertEqual(decision(machine,mmap,[0,4096,4,0x20]),reject)
            self.assertEqual(decision(machine,mmap,[0,4096,4,2]),allow)
            self.assertEqual(decision(machine,protect,[0,4096,4]),reject)
            self.assertEqual(decision(machine,protect,[0,4096,1]),allow)
            self.assertEqual(decision(machine,435),0x00050000|errno.ENOSYS)
            self.assertEqual(decision(machine,0),allow)
            self.assertEqual(decision(machine,0,arch=123),reject)
        self.assertEqual(decision("x86_64",0x40000000|57),reject)
        self.assertEqual(driver.execution.install_native_process_filter(False),
                         dict(scope="artifact_only",kernel_native_process_filter_installed=False))

    def test_landlock_rule_scope_is_exact_and_same_device_isolation_rejects(self):
        policy = dict(runtime=dict(files=[dict(path="/runtime/python",file_sha256="runtime-fixture")]),
            exact_data_files=[dict(path="/application/owner.py",file_sha256="application-fixture")],
            production_roots=["/readonly-data"],output_root="/separate-output")
        rules = driver.execution.native_read_rules(policy,dict(files=[dict(path="/application/driver.py")]),12345)
        self.assertEqual(rules["/runtime/python"],5)
        self.assertEqual(rules["/application/driver.py"],4)
        self.assertEqual(rules["/application/owner.py"],4)
        self.assertEqual(rules["/proc/12345/maps"],4)
        self.assertEqual(rules["/proc"],8)
        self.assertNotIn("/proc/12345/fd",rules)
        self.assertEqual(rules["/readonly-data"],12)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();output=root/"output";output.mkdir();production=root/"production";production.mkdir()
            anchor=driver.execution.OutputAnchor(output,new=False)
            try:
                with patch.object(driver.execution.sys,"platform","linux"), patch.object(driver.execution,"require_immutable_runtime"), \
                     patch.object(Path,"read_text",return_value="1 0 1:1 / / ro,noexec - tmpfs fixture ro\n"):
                    with self.assertRaisesRegex(ValueError,"share a filesystem"):
                        driver.execution.require_native_namespace({},anchor,[str(production)])
            finally:anchor.close()

    def test_kernel_write_and_executable_memory_rules_cover_exact_R3_reproduction(self):
        import errno
        reject = 0x00050000 | errno.EPERM; allow = 0x7fff0000
        for machine, mmap, protect, pkey, personality, native_mutations, execve in (
                ("x86_64",9,10,329,135,[101,311,323,321,298,216,16,425,426,427],59),
                ("aarch64",222,226,288,92,[117,271,282,280,241,234,29,425,426,427],221)):
            for before in (False, True):
                with self.subTest(machine=machine, before_exec=before):
                    self.assertEqual(bpf_decision(machine,mmap,[0,4096,7,2],before_exec=before),reject)
                    self.assertEqual(bpf_decision(machine,mmap,[0,4096,5,2],before_exec=before),allow)
                    self.assertEqual(bpf_decision(machine,mmap,[0,4096,3,2],before_exec=before),allow)
                    self.assertEqual(bpf_decision(machine,mmap,[0,4096,5,0x22],before_exec=before),reject)
                    for call in (protect,pkey):
                        self.assertEqual(bpf_decision(machine,call,[0,4096,5],before_exec=before),reject)
                        self.assertEqual(bpf_decision(machine,call,[0,4096,3],before_exec=before),allow)
                    self.assertEqual(bpf_decision(machine,personality,[0x0400000],before_exec=before),reject)
                    self.assertEqual(bpf_decision(machine,personality,[0xffffffff],before_exec=before),allow)
                    for call in native_mutations: self.assertEqual(bpf_decision(machine,call,before_exec=before),reject)
                    self.assertEqual(bpf_decision(machine,execve,before_exec=before),allow if before else reject)
        self.assertTrue(driver.execution.native_handled_access() & 2)
        self.assertTrue(driver.execution.native_handled_access() & 16384)
        policy = dict(runtime=dict(files=[dict(path="/runtime/python")]), exact_data_files=[],
                      production_roots=["/data"], output_root="/output")
        rules = driver.execution.native_read_rules(policy,dict(files=[]),777)
        self.assertTrue(rules["/output"] & 2)
        self.assertTrue(rules["/output"] & 16384)
        for path in ("/runtime/python","/data","/proc/777/maps","/proc"):
            self.assertFalse(rules[path] & 2)
        self.assertNotIn("/proc/777/mem",rules)

    def test_ELF_metadata_rejects_initial_RWX_executable_and_missing_stack(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name).resolve() / "metadata-only.elf"
            def fixture(load_flags=5, stack_flags=6, stack=True):
                header = bytearray(64); header[:6] = b"\x7fELF\x02\x01"
                struct.pack_into("<HH",header,16,3,62); struct.pack_into("<Q",header,32,64)
                struct.pack_into("<HH",header,54,56,2 if stack else 1)
                first=bytearray(56);struct.pack_into("<II",first,0,1,load_flags)
                second=bytearray(56);struct.pack_into("<II",second,0,0x6474e551,stack_flags)
                path.write_bytes(header + first + (second if stack else b""))
                return [dict(driver.raw_ref(path))]
            valid = driver.execution.elf_execution_policy(fixture(),str(path))
            self.assertEqual(valid["files"][0]["load_flags"],[5])
            self.assertEqual(valid["files"][0]["gnu_stack_flags"],6)
            for kwargs in (dict(load_flags=7),dict(stack_flags=7),dict(stack=False)):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    driver.execution.elf_execution_policy(fixture(**kwargs),str(path))

    def test_child_result_identical_inode_replacement_rejects_with_original_unknown_custody(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();output=root/"output";output.mkdir()
            prepared,checkpoints=self.checkpoint_fixture(root)
            payload=dict(kind="checkpoint_digest",prepared=prepared,checkpoints=checkpoints,checkpoint_sample=dict(prepared=prepared[0]))
            original=driver.execution.launch_native;captured={}
            def replaced(invocation,anchor,log):
                data,native=original(invocation,anchor,log)
                path=Path(invocation["payload"]["worker_result_path"]);raw=path.read_bytes()
                captured.update(original=driver.execution.identity(path.stat()),registry=json.loads(data)["output_registry"])
                path.rename(root/"moved-original.json");path.write_bytes(raw)
                captured["replacement"]=driver.execution.identity(path.stat())
                return data,native
            with patch.object(driver.execution,"launch_native",new=replaced):
                with self.assertRaisesRegex(ValueError,"pathname/inode changed"):
                    driver.execute_worker(payload,output/"probe")
            self.assertNotEqual(captured["original"],captured["replacement"])
            self.assertEqual((root/"moved-original.json").read_bytes(),(output/"probe.result.json").read_bytes())
            state=json.loads(next(output.glob("failure-*.json")).read_bytes())
            row=next(row for row in state["created_files"] if row["relative_path"]=="probe.result.json")
            self.assertEqual({k:row[k] for k in ("device","inode")},captured["original"])
            self.assertFalse(row["custody"]["anchored_name_matches_created_inode"])
            self.assertIsNone(row["custody"]["kernel_observed_current_path"])
            self.assertIn(captured["registry"],state["received_original_registries"])

    def test_real_child_custody_normal_path_and_later_export_replacements(self):
        for suffix in ("result.json","observation.json","native.json"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as name:
                root=Path(name).resolve();output=root/"output";output.mkdir()
                prepared,checkpoints=self.checkpoint_fixture(root)
                payload=dict(kind="checkpoint_digest",prepared=prepared,checkpoints=checkpoints,checkpoint_sample=dict(prepared=prepared[0]))
                measured,refs=driver.execute_worker(payload,output/"probe")
                invocation=json.loads((output/"probe.invocation.json").read_bytes())
                driver.checked_output_custody(invocation,measured["worker"]["output_registry"],str(output/"probe.result.json"))
                registry=measured["output_registry"]
                self.assertTrue({"probe.input.json","probe.invocation.json","probe.log","probe.result.json","probe.native.json","probe.observation.json"} <=
                                {row["relative_path"] for row in registry["entries"]})
                target=output/("probe."+suffix);raw=target.read_bytes();target.rename(root/("original-"+suffix));target.write_bytes(raw)
                probe=dict(raw_log_refs=refs,peak_rss_bytes=measured["peak_rss_bytes"],wall_seconds=measured["worker"]["result"]["wall_seconds_per_operation"],
                           total_worker_wall_seconds=measured["wall_seconds"],operation_count=6)
                with self.assertRaisesRegex(ValueError,"pathname/inode changed"):
                    driver.reconcile_exported_values(dict(points=[],separate_real_probes=dict(digest=probe)),dict(records=[]))

    def fault_deployment(self, root, *, before_work="", after_main=""):
        scripts=root/"fault-copy"/"scripts";scripts.mkdir(parents=True)
        source=DRIVER_PATH.read_text()
        line='            value = worker_point(payload) if payload["kind"] == "point" else worker_probe(payload)\n'
        self.assertEqual(source.count(line),1)
        if before_work: source=source.replace(line,''.join('            '+v+'\n' for v in before_work.splitlines())+line)
        if after_main: source+='\nif __name__ == "__main__":\n'+''.join('    '+v+'\n' for v in after_main.splitlines())
        path=scripts/DRIVER_PATH.name;path.write_text(source)
        (scripts/"performance_driver_runtime.py").write_bytes(Path(driver.execution.__file__).read_bytes())
        return driver.execution.capture_sources(path,DRIVER_PATH.parents[1]/"spacing_rerun")

    def failure_probe(self, root, deployment, *, after_exit=None):
        output=root/"output";output.mkdir();prepared,checkpoints=self.checkpoint_fixture(root)
        payload=dict(kind="checkpoint_digest",prepared=prepared,checkpoints=checkpoints,checkpoint_sample=dict(prepared=prepared[0]))
        original=driver.execution.launch_native;captured={}
        def launched(invocation,anchor,log):
            data,native=original(invocation,anchor,log);captured.update(producer=data,native=native,invocation=dict(invocation))
            if after_exit: after_exit(invocation,anchor,captured)
            return data,native
        with patch.object(driver.execution,"capture_sources",return_value=deployment), patch.object(driver.execution,"launch_native",new=launched):
            with self.assertRaises(ValueError): driver.execute_worker(payload,output/"probe")
        state=json.loads(next(output.glob("failure-*.json")).read_bytes())
        return output,state,captured

    def test_genuine_nonzero_child_after_result_keeps_original_registry_and_raw_channels(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();deployment=self.fault_deployment(root,after_main='raise RuntimeError("labelled after genuine result fault")')
            def moved(invocation,anchor,captured):
                path=Path(invocation["payload"]["worker_result_path"]);raw=path.read_bytes()
                captured["original"]=driver.execution.identity(path.stat());path.rename(root/"original-result.json");path.write_bytes(raw)
            output,state,captured=self.failure_probe(root,deployment,after_exit=moved)
            self.assertEqual(captured["native"]["exit_code"],1)
            row=next(r for r in state["created_files"] if r["relative_path"]=="probe.result.json")
            self.assertEqual(driver.execution.identity_values(row),captured["original"])
            self.assertIsNone(row["custody"]["kernel_observed_current_path"])
            self.assertEqual((output/"probe.producer.bin").read_bytes(),captured["producer"])
            self.assertEqual((output/"probe.custody-journal.bin").read_bytes(),driver.execution.base64.b64decode(captured["native"]["custody_journal_raw_base64"]))
            self.assertFalse(state["original_child_custody_complete"])

    def test_genuine_caught_partial_failure_keeps_moved_directory_originals(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();deployment=self.fault_deployment(root,before_work='ACTIVE_OUTPUT.write(ACTIVE_OUTPUT.path/"partial-only"/"opaque.bin",b"CPU artifact")\nraise RuntimeError("labelled pre-result partial fault")')
            def moved(invocation,anchor,captured):
                path=anchor.path/"partial-only";captured["original"]=driver.execution.identity(path.stat());path.rename(root/"original-partial-directory");path.mkdir()
            output,state,captured=self.failure_probe(root,deployment,after_exit=moved)
            self.assertEqual(captured["native"]["exit_code"],1)
            packet=json.loads(captured["producer"]);self.assertEqual(packet["schema"],"p4-buffered-worker-failure-v1")
            self.assertTrue(any(r["relative_path"]=="partial-only/opaque.bin" for r in state["created_files"]))
            custody=state["retained_directory_custody"]["partial-only"]
            self.assertEqual(custody["retained_inode"],captured["original"]);self.assertIsNone(custody["kernel_observed_fd_path"])
            self.assertEqual(state["worker_custody_transports"][0]["journal_prefix"]["terminal_kind"],"failure")
            self.assertTrue((output/"probe.log").read_bytes())

    def test_genuine_abrupt_partial_exit_keeps_flushed_prefix_without_success_packet(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();deployment=self.fault_deployment(root,before_work='ACTIVE_OUTPUT.write(ACTIVE_OUTPUT.path/"partial-only"/"opaque.bin",b"CPU artifact")\nos._exit(23)')
            output,state,captured=self.failure_probe(root,deployment)
            self.assertEqual(captured["native"]["exit_code"],23);self.assertEqual(captured["producer"],b"")
            self.assertTrue(any(r["relative_path"]=="partial-only/opaque.bin" for r in state["created_files"]))
            prefix=state["worker_custody_transports"][0]["journal_prefix"]
            self.assertIsNone(prefix["terminal_kind"]);self.assertFalse(prefix["complete_registered_journal"])
            self.assertFalse(state["original_child_custody_complete"])
            self.assertEqual((output/"probe.producer.bin").read_bytes(),b"")

    def test_missing_truncated_wrong_binding_and_invalid_producer_transport_is_retained_unverified(self):
        variants={"missing":'os.ftruncate(invocation["custody_journal_fd"],0)',
                  "truncated":'os.write(invocation["custody_journal_fd"],b"{")',
                  "deep-invalid":'os.write(invocation["custody_journal_fd"],b"["*2000+b"0"+b"]"*2000+b"\\n")',
                  "wrong-binding":'os.write(invocation["custody_journal_fd"],execution.encoded(dict(schema="p4-worker-custody-journal-v1",invocation_buffer_sha256="wrong",sequence=3,previous_sha256=ACTIVE_OUTPUT.custody_journal.previous,event="created",value=dict(relative_path="forged.bin",kind="file",device=1,inode=1)))+b"\\n")'}
        for label,damage in variants.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                root=Path(name).resolve();code='ACTIVE_OUTPUT.write(ACTIVE_OUTPUT.path/"partial-only"/"opaque.bin",b"CPU artifact")\n'+damage+'\nos.write(invocation["producer_channel_fd"],b"invalid producer bytes")\nos._exit(24)'
                deployment=self.fault_deployment(root,before_work=code);output,state,captured=self.failure_probe(root,deployment)
                self.assertEqual(captured["native"]["exit_code"],24)
                transport=state["worker_custody_transports"][0];prefix=transport["journal_prefix"]
                self.assertTrue(prefix["issues"]);self.assertFalse(prefix["complete_registered_journal"])
                self.assertFalse(any(r["relative_path"]=="forged.bin" for r in state["created_files"]))
                partial=any(r["relative_path"]=="partial-only/opaque.bin" for r in state["created_files"])
                self.assertEqual(partial,label!="missing")
                self.assertEqual((output/"probe.producer.bin").read_bytes(),b"invalid producer bytes")
                self.assertFalse(state["original_child_custody_complete"])

    def test_exit_zero_with_missing_journal_rejects_and_retains_original_success_packet_claims(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();deployment=self.fault_deployment(root,after_main='os.ftruncate(BUFFERED_INVOCATION["custody_journal_fd"],0)')
            output,state,captured=self.failure_probe(root,deployment)
            self.assertEqual(captured["native"]["exit_code"],0)
            self.assertTrue(any(r["relative_path"]=="probe.result.json" for r in state["created_files"]))
            self.assertEqual((output/"probe.custody-journal.bin").read_bytes(),b"")
            self.assertFalse(state["original_child_custody_complete"])

    def test_child_created_directory_and_unregistered_output_are_in_custody_scope(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();output=root/"output";output.mkdir();script=root/"custody.py"
            script.write_text('''import os,performance_driver_runtime as rt
a=rt.OutputAnchor(BUFFERED_INVOCATION['output_root'],new=False,inherited_fd=BUFFERED_INVOCATION['output_fd'],expected=BUFFERED_INVOCATION['output_identity'])
a.receive_registry(BUFFERED_INVOCATION['output_registry_before_child'])
a.write(a.path/'child-only'/'opaque.json',b'{}')
os.write(BUFFERED_INVOCATION['producer_channel_fd'],rt.encoded({'output_registry':a.registry()}))
''')
            anchor=driver.execution.OutputAnchor(output,new=False)
            deployment=driver.execution.capture_sources(script,DRIVER_PATH.parents[1]/"spacing_rerun")
            try:
                deployed=driver.deploy_buffers(deployment,anchor)
                data,native=driver.execution.launch_native(dict(deployment=deployment,deployment_root=str(deployed),input_path=str(root/"unused"),
                    output_root=str(output),output_fd=anchor.fd,output_identity=anchor.root_identity),anchor,output/"child.log")
                self.assertEqual(native["exit_code"],0,(output/"child.log").read_text())
                registry=json.loads(data)["output_registry"]
                original_inode=driver.execution.identity((output/"child-only").stat())
                (output/"child-only").rename(root/"moved-child-directory");(output/"child-only").mkdir()
                with self.assertRaisesRegex(ValueError,"nested directory inode chain"):
                    anchor.receive_registry(registry)
                state=anchor.failure_state(ValueError("moved child directory"))
                self.assertEqual(state["retained_directory_custody"]["child-only"]["retained_inode"],original_inode)
                self.assertIsNone(state["retained_directory_custody"]["child-only"]["kernel_observed_fd_path"])
            finally:anchor.close()
        with tempfile.TemporaryDirectory() as name:
            anchor=driver.execution.OutputAnchor(Path(name).resolve(),new=False)
            try:
                baseline=anchor.registry();anchor.receive_registry(baseline)
                malformed=dict(baseline,root_identity=dict(device=True,inode=baseline["root_identity"]["inode"]))
                with self.assertRaisesRegex(ValueError,"root/schema"):
                    driver.execution.checked_registry(malformed,anchor.path,dict(device=1,inode=baseline["root_identity"]["inode"]))
                (anchor.path/"unregistered.json").write_bytes(b"{}")
                with self.assertRaisesRegex(ValueError,"unregistered"):
                    anchor.registry()
            finally:anchor.close()


if __name__ == "__main__":
    unittest.main()
