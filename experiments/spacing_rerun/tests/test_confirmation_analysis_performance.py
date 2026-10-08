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
import time
import unittest
from unittest.mock import patch
import zipfile
import struct
import base64
import hashlib
import socket
import contextlib
import signal

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
            native=measured["native_process_observation"]
            self.assertEqual(measured["schema"],"p4-analysis-performance-worker-observation-v6")
            self.assertEqual(measured["worker"]["schema"],"p4-buffered-worker-result-v6")
            self.assertEqual(native["inherited_channel_identities_after"],channels)
            self.assertEqual(driver.execution.original_preexec_receipt(native),b"")
            self.assertIsNone(native["preexec_kernel_observation"])
            self.assertEqual(len(refs),9)
            self.assertEqual((root/"cpu-artifact-probe.preexec-receipt.bin").read_bytes(),b"")
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
        for suffix in ("result.json","observation.json","native.json","preexec-receipt.bin"):
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


    def test_original_preexec_bytes_preserve_noncanonical_json_without_reconstruction(self):
        raw=b' { "labelled_cpu_fixture" : true, "value" : 7 } \n\n'
        identity=dict(device=1,inode=2,mode=3)
        native=dict(driver.execution.observe_preexec_receipt(raw),
                    inherited_channel_identities=dict(preexec_receipt=identity),
                    preexec_receipt_channel_identity=identity)
        self.assertEqual(driver.execution.original_preexec_receipt(native),raw)
        self.assertNotEqual(raw,driver.execution.encoded(native["preexec_kernel_observation"])+b"\n")
        self.assertEqual(native["preexec_receipt_raw_bytes"],len(raw))
        self.assertEqual(native["preexec_receipt_raw_sha256"],driver.execution.sha(raw))

    def test_original_preexec_bytes_reject_length_hash_parsed_and_identity_tampering(self):
        raw=b'{"labelled_cpu_fixture":true}\n'
        identity=dict(device=1,inode=2,mode=3)
        native=dict(driver.execution.observe_preexec_receipt(raw),
                    inherited_channel_identities=dict(preexec_receipt=identity),
                    preexec_receipt_channel_identity=identity)
        changes=[dict(preexec_receipt_raw_bytes=True),dict(preexec_receipt_raw_bytes=len(raw)+1),
                 dict(preexec_receipt_raw_sha256="wrong"),dict(preexec_receipt_raw_base64="!"),
                 dict(preexec_kernel_observation={}),dict(preexec_receipt_parse_error="invented"),
                 dict(preexec_receipt_channel_identity={})]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                driver.execution.original_preexec_receipt(dict(native,**change))

    def receipt_failure_probe(self,root,raw,*,fail_preexec=False):
        """Real CPU child writes labelled bytes; no kernel policy is installed."""
        output=root/"output";output.mkdir()
        payload=self.probe_fixture(root)
        actual_popen=driver.execution.subprocess.Popen
        def injected_popen(*args,**kwargs):
            self.assertIsNone(kwargs["preexec_fn"])
            receipt_fd=kwargs["pass_fds"][2]
            def labelled_cpu_callback():
                if raw: os.write(receipt_fd,raw)
                if fail_preexec: raise RuntimeError("labelled CPU preexec callback fault")
            kwargs["preexec_fn"]=labelled_cpu_callback
            return actual_popen(*args,**kwargs)
        with patch.object(driver.execution.subprocess,"Popen",new=injected_popen):
            with self.assertRaises(driver.execution.NativeLaunchFailure) as caught:
                driver.execute_worker(payload,output/"probe")
        native=caught.exception.native_observation
        state=json.loads(next(output.glob("failure-*.json")).read_bytes())
        self.assertEqual((output/"probe.preexec-receipt.bin").read_bytes(),raw)
        self.assertEqual(driver.execution.original_preexec_receipt(native),raw)
        self.assertEqual(json.loads((output/"probe.native.json").read_bytes()),native)
        self.assertEqual(state["worker_custody_transports"][0]["native_observation"],native)
        self.assertFalse(state["original_child_custody_complete"])
        self.assertFalse(native["original_child_custody_complete"])
        return output,state,native

    def test_genuine_malformed_truncated_and_valid_artifact_receipt_bytes_are_retained_and_rejected(self):
        for raw in (b"{",b"not JSON",b' { "labelled_cpu_fixture" : true } \n',b"["*2000+b"0"):
            with self.subTest(raw_length=len(raw)),tempfile.TemporaryDirectory() as name:
                output,state,native=self.receipt_failure_probe(Path(name).resolve(),raw)
                self.assertTrue(native["wait4_observation_available"])
                self.assertEqual(native["exit_code"],0)
                self.assertGreater(native["pid"],0)
                self.assertTrue((output/"probe.producer.bin").read_bytes())
                if raw.lstrip().startswith(b'{ "labelled'):
                    self.assertIsNone(native["preexec_receipt_parse_error"])
                    self.assertIn("Artifact launch",native["launch_error"])
                else:
                    self.assertTrue(native["preexec_receipt_parse_error"])
                    self.assertIn("parse failed",native["launch_error"])

    def test_genuine_Popen_preexec_failure_retains_available_bytes_without_invented_wait4(self):
        with tempfile.TemporaryDirectory() as name:
            output,state,native=self.receipt_failure_probe(Path(name).resolve(),b'{"labelled_partial":',fail_preexec=True)
            self.assertEqual(native["schema"],"p4-internal-native-launch-failure-v1")
            self.assertEqual(native["launch_error_type"],"SubprocessError")
            self.assertFalse(native["wait4_observation_available"])
            for key in ("pid","exit_code","raw_wait_status","through_exit_child_peak_rss_bytes"):
                self.assertIsNone(native[key])
            self.assertEqual(native["inherited_channel_identities"],native["inherited_channel_identities_after"])
            self.assertEqual((output/"probe.producer.bin").read_bytes(),b"")

    def test_genuine_Popen_preexec_failure_with_absent_receipt_stays_honestly_empty(self):
        with tempfile.TemporaryDirectory() as name:
            _,_,native=self.receipt_failure_probe(Path(name).resolve(),b"",fail_preexec=True)
            self.assertEqual(native["preexec_receipt_raw_bytes"],0)
            self.assertEqual(native["preexec_receipt_raw_sha256"],driver.execution.sha(b""))
            self.assertIsNone(native["preexec_kernel_observation"])
            self.assertIsNone(native["preexec_receipt_parse_error"])

    def test_actual_prebuffer_output_setup_failure_retains_empty_channels_without_journal_claims(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();output=root/"output";output.mkdir()
            payload=self.probe_fixture(root)
            (output/"probe.invocation.json").write_bytes(b"labelled existing output")
            with self.assertRaises(driver.execution.NativeLaunchFailure) as caught:
                driver.execute_worker(payload,output/"probe")
            native=caught.exception.native_observation
            self.assertEqual(native["launch_error_type"],"FileExistsError")
            self.assertIsNone(native["invocation_buffer_sha256"])
            self.assertIsNone(native["pid"])
            self.assertFalse(native["wait4_observation_available"])
            self.assertEqual(json.loads((output/"probe.native.json").read_bytes()),native)
            state=json.loads(next(output.glob("failure-*.json")).read_bytes())
            transport=state["worker_custody_transports"][0]
            self.assertIsNone(transport["journal_prefix"])
            self.assertIn("No invocation buffer",str(transport["issues"]))
            self.assertFalse(transport["original_child_custody_complete"])
            for suffix in ("producer.bin","custody-journal.bin","preexec-receipt.bin"):
                self.assertEqual((output/("probe."+suffix)).read_bytes(),b"")

    def timeout_failure_probe(self,root):
        """External CPU watchdog reaps a real child, then reports its timeout."""
        output=root/"output";output.mkdir();payload=self.probe_fixture(root)
        deployment=self.fault_deployment(root,before_work='os.write(invocation["producer_channel_fd"],b"labelled CPU producer prefix")\nos.write(invocation["custody_journal_fd"],b"{labelled journal truncated")\nprint("labelled CPU stdout prefix",flush=True)\ntime.sleep(60)')
        actual_popen=driver.execution.subprocess.Popen;actual_wait4=driver.execution.os.wait4;watchdog={}
        def injected_popen(*args,**kwargs):
            self.assertIsNone(kwargs["preexec_fn"])
            receipt_fd=kwargs["pass_fds"][2]
            kwargs["preexec_fn"]=lambda:os.write(receipt_fd,b'{"labelled_timeout":')
            return actual_popen(*args,**kwargs)
        def external_watchdog(pid,options):
            self.assertEqual(options,0)
            deadline=time.monotonic()+3
            ready=False
            while time.monotonic()<deadline:
                row=actual_wait4(pid,os.WNOHANG)
                self.assertEqual(row[0],0,"CPU watchdog child exited before timeout")
                if b"labelled CPU stdout prefix" in (output/"probe.log").read_bytes():
                    ready=True;break
                time.sleep(.01)
            os.kill(pid,9);observed_pid,status,usage=actual_wait4(pid,0)
            watchdog.update(pid=observed_pid,status=status,exit_code=os.waitstatus_to_exitcode(status),peak_rss_bytes=driver.execution.rss_bytes(usage),prefix_ready=ready)
            if not ready: raise AssertionError("CPU watchdog prefix did not arrive")
            raise subprocess.TimeoutExpired("labelled external CPU watchdog",3)
        with patch.object(driver.execution,"capture_sources",return_value=deployment),patch.object(driver.execution.subprocess,"Popen",new=injected_popen),patch.object(driver.execution.os,"wait4",new=external_watchdog):
            with self.assertRaises(driver.execution.NativeLaunchFailure) as caught:
                driver.execute_worker(payload,output/"probe")
        native=caught.exception.native_observation;state=json.loads(next(output.glob("failure-*.json")).read_bytes())
        return output,state,native,watchdog

    def test_actual_external_timeout_preserves_raw_prefixes_and_known_Popen_pid(self):
        with tempfile.TemporaryDirectory() as name:
            output,state,native,watchdog=self.timeout_failure_probe(Path(name).resolve())
            self.assertEqual(native["pid"],watchdog["pid"])
            self.assertEqual(watchdog["exit_code"],-9)
            self.assertTrue(watchdog["prefix_ready"])
            self.assertEqual(native["launch_error_type"],"TimeoutExpired")
            self.assertFalse(native["wait4_observation_available"])
            for key in ("exit_code","raw_wait_status","through_exit_child_peak_rss_bytes"):
                self.assertIsNone(native[key])
            self.assertEqual((output/"probe.producer.bin").read_bytes(),b"labelled CPU producer prefix")
            self.assertEqual((output/"probe.preexec-receipt.bin").read_bytes(),b'{"labelled_timeout":')
            self.assertIn(b"labelled CPU stdout prefix",(output/"probe.log").read_bytes())
            self.assertTrue((output/"probe.custody-journal.bin").read_bytes().endswith(b"{labelled journal truncated"))
            self.assertEqual(driver.execution.original_preexec_receipt(native),b'{"labelled_timeout":')
            self.assertEqual(json.loads((output/"probe.native.json").read_bytes()),native)
            transport=state["worker_custody_transports"][0]
            self.assertIsNotNone(transport["journal_prefix"]["original_registry"])
            self.assertIsNone(transport["journal_prefix"]["terminal_kind"])
            self.assertFalse(transport["original_child_custody_complete"])

    def test_parent_preexec_receipt_reconciles_actual_nonce_policy_wait4_and_four_channels(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name).resolve();measured,_=driver.execute_worker(self.probe_fixture(root),root/"probe")
            invocation=json.loads((root/"probe.invocation.json").read_bytes())
            native=measured["native_process_observation"]
            changes=[dict(invocation_nonce="wrong"),dict(policy_sha256="invented"),
                     dict(invocation_buffer_sha256="wrong"),dict(wait4_observation_available=False),
                     dict(inherited_channel_identities_after={}),dict(preexec_receipt_channel_identity={}),
                     dict(driver.execution.observe_preexec_receipt(b'{"labelled_cpu_fixture":true}'))]
            for change in changes:
                with self.subTest(change=change),self.assertRaises(ValueError):
                    driver.validate_worker_result(invocation["payload"],measured["worker"],invocation,dict(native,**change))

    def test_export_preexec_receipt_rejects_missing_duplicate_bytes_and_hash_tampering(self):
        for mutation in ("missing","duplicate","bytes","hash"):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as name:
                root=Path(name).resolve();measured,refs=driver.execute_worker(self.probe_fixture(root),root/"probe")
                receipt_ref=next(r for r in refs if r["path"].endswith(".preexec-receipt.bin"))
                refs=copy.deepcopy(refs)
                if mutation=="missing": refs=[r for r in refs if not r["path"].endswith(".preexec-receipt.bin")]
                elif mutation=="duplicate": refs.append(copy.deepcopy(receipt_ref))
                elif mutation=="bytes": Path(receipt_ref["path"]).write_bytes(b"labelled mutation")
                else: next(r for r in refs if r["path"]==receipt_ref["path"])["file_sha256"]="wrong"
                probe=dict(raw_log_refs=refs,peak_rss_bytes=measured["peak_rss_bytes"],
                    wall_seconds=measured["worker"]["result"]["wall_seconds_per_operation"],
                    total_worker_wall_seconds=measured["wall_seconds"],operation_count=6)
                with self.assertRaises(ValueError):
                    driver.reconcile_exported_values(dict(points=[],separate_real_probes=dict(digest=probe)),dict(records=[]))


OWNER_RETAINER_PATH = Path(os.environ.get("P4_DRIVER_R7_OWNER_ARTIFACT",
    str(DRIVER_PATH.parents[3] / "owner-artifacts/native_channel_retainer.py")))
RETAINER_CHILD = r"""
import json,sys,types
from pathlib import Path
path=Path(sys.argv[1]);ns=dict(__name__='original_fd_cpu_fixture',__file__=str(path))
exec(compile(path.read_bytes(),str(path),'exec',dont_inherit=True),ns)
rt=ns['load_runtime'](sys.argv[2],sys.argv[3])
config=json.loads(Path(sys.argv[5]).read_bytes())
r=ns['OriginalChannelRetainer'](rt,int(sys.argv[4]),sys.argv[6],config['binding'],artifact_only=True)
summary=r.serve(max_seconds=config.get('seconds',5),interval=.05)
print(json.dumps(dict(pid=__import__('os').getpid(),ppid=__import__('os').getppid(),summary=summary)),flush=True)
"""
SUPERVISOR_KILL_CHILD = r"""
import json,sys,types,os,time
from pathlib import Path
p=Path(sys.argv[1]);ns=dict(__name__='kill_cpu_fixture',__file__=str(p))
exec(compile(p.read_bytes(),str(p),'exec',dont_inherit=True),ns)
rt=ns['load_runtime'](sys.argv[2],sys.argv[3]);config=json.loads(Path(sys.argv[4]).read_bytes())
anchor=rt.OutputAnchor(Path(config['output']),new=False,inherited_fd=int(sys.argv[5]),expected=config['identity'])
anchor.receive_registry(config['registry'])
original_popen=rt.subprocess.Popen
original_wait4=rt.os.wait4
# Explicit CPU fault injection writes an actual truncated fixture receipt only.
def popen(*args,**kwargs):
    preexec=kwargs['preexec_fn'];receipt_fd=kwargs['pass_fds'][2]
    def injected():
        preexec();os.write(receipt_fd,b'{"labelled_CPU_receipt_prefix":')
    kwargs['preexec_fn']=injected
    return original_popen(*args,**kwargs)
def reaped(pid,options):
    value=original_wait4(pid,options)
    Path(config['marker']).write_text(json.dumps(dict(pid=value[0],status=value[1],reaped=True)))
    time.sleep(60)
    return value
rt.subprocess.Popen=popen;rt.os.wait4=reaped
rt.launch_native(config['invocation'],anchor,Path(config['output'])/'native.log',channel_observer_fd=int(sys.argv[6]))
"""


class OriginalChannelTransportArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=OWNER_RETAINER_PATH.read_bytes()
        expected=os.environ.get("P4_DRIVER_R7_OWNER_ARTIFACT_SHA256")
        if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError("Owner artifact differs from caller frozen raw pin")
        ns=dict(__name__="p4_owner_artifact_tests",__file__=str(OWNER_RETAINER_PATH))
        exec(compile(raw,str(OWNER_RETAINER_PATH),"exec",dont_inherit=True),ns)
        import types
        cls.owner=types.SimpleNamespace(**ns)
        cls.rt=driver.execution

    @contextlib.contextmanager
    def root(self):
        retained=os.environ.get("P4_DRIVER_R7_TEST_EVIDENCE_ROOT")
        if retained:
            yield Path(tempfile.mkdtemp(prefix=self._testMethodName+"-",dir=retained)).resolve()
        else:
            with tempfile.TemporaryDirectory() as name: yield Path(name).resolve()

    def invocation(self,root,anchor,code):
        package=root/'tiny_cpu_package';package.mkdir()
        (package/'__init__.py').write_text('# labelled CPU artifact only\n')
        script=root/'opaque_cpu.py';script.write_text(code)
        deployment=self.rt.capture_sources(script,package)
        retained=driver.deploy_buffers(deployment,anchor)
        return dict(deployment=deployment,deployment_root=str(retained),input_path=str(root/'unused'),
            nonce='CPU-original-invocation',observer_run_nonce='CPU-run-reservation',observer_attempt_nonce='CPU-attempt-reservation')

    def binding(self,invocation,anchor):
        return dict(invocation_nonce=invocation['nonce'],run_nonce=invocation['observer_run_nonce'],
            attempt_nonce=invocation['observer_attempt_nonce'],deployment_sha256=invocation['deployment']['sha256'],
            policy_sha256=None,output_root_identity=anchor.root_identity,scope='artifact_only_cpu')

    def start_retainer(self,root,invocation,anchor,*,seconds=5):
        outside,supervisor=self.owner.create_observer_bridge(artifact_only=True)
        config=root/'retainer-config.json';config.write_bytes(self.rt.encoded(dict(binding=self.binding(invocation,anchor),seconds=seconds)))
        argv=[sys.executable,'-I','-B','-S','-c',RETAINER_CHILD,str(OWNER_RETAINER_PATH),str(self.rt.__file__),
            hashlib.sha256(Path(self.rt.__file__).read_bytes()).hexdigest(),str(outside.fileno()),str(config),str(root/'retained')]
        stdout=(root/'retainer.stdout.log').open('wb');stderr=(root/'retainer.stderr.log').open('wb')
        child=subprocess.Popen(argv,pass_fds=(outside.fileno(),),stdout=stdout,stderr=stderr)
        outside.close();stdout.close();stderr.close()
        (root/'retainer-process.json').write_bytes(self.rt.encoded(dict(argv=argv,pid=child.pid,parent_pid=os.getpid())))
        return child,supervisor

    def reap(self,child,root):
        try: status=child.wait(timeout=8)
        except subprocess.TimeoutExpired:
            child.kill();child.wait();raise
        (root/'retainer-process-exit.json').write_bytes(self.rt.encoded(dict(pid=child.pid,exit_code=status)))
        self.assertEqual(status,0,(root/'retainer.stderr.log').read_text())
        return json.loads((root/'retained/retainer-summary.json').read_bytes())

    def test_real_rights_preserve_same_original_and_shared_offset(self):
        with self.root() as root, tempfile.TemporaryFile(dir=root) as handle:
            a,b=self.owner.create_observer_bridge(artifact_only=True)
            try:
                handle.write(b'original opaque bytes');handle.seek(7)
                self.rt.observer_send_packet(a,b'raw-message',[handle.fileno()])
                packet=self.rt.observer_receive_packet(b)
                fd=packet['fds'][0]
                try:
                    self.assertEqual(self.rt.identity(os.fstat(fd)),self.rt.identity(os.fstat(handle.fileno())))
                    self.assertEqual(os.fstat(fd).st_nlink,0);self.assertFalse(os.get_inheritable(fd))
                    self.assertEqual(os.pread(fd,8,0),b'original')
                    self.assertEqual(handle.tell(),7)
                    os.lseek(fd,9,os.SEEK_SET);self.assertEqual(handle.tell(),9)
                    self.assertEqual(packet['raw'],b'raw-message')
                finally: os.close(fd)
            finally: a.close();b.close()

    def test_partial_frame_length_and_payload_preserve_original_received_bytes(self):
        for label,raw in [('header',b'\x00\x00'),('oversize',struct.pack('!I',32769)),('payload',struct.pack('!I',9)+b'prefix')]:
            with self.subTest(label=label):
                a,b=self.owner.create_observer_bridge(artifact_only=True)
                try:
                    a.sendall(raw);a.shutdown(socket.SHUT_WR)
                    with self.assertRaises(self.rt.ObserverPacketFailure) as caught:self.rt.observer_receive_packet(b)
                    self.assertEqual(caught.exception.header+ caught.exception.raw,raw)
                finally:a.close();b.close()

    def test_total_frame_deadline_keeps_actual_partial_bytes_and_restores_timeout(self):
        a,b=self.owner.create_observer_bridge(artifact_only=True)
        try:
            raw=struct.pack('!I',9)+b'prefix';a.sendall(raw)
            with self.assertRaises(self.rt.ObserverPacketFailure) as caught:
                self.rt.observer_receive_packet(b,deadline=time.monotonic()+.03)
            self.assertEqual(caught.exception.header+caught.exception.raw,raw)
            self.assertIsNone(b.gettimeout())
        finally:a.close();b.close()

    def test_excess_rights_reject_without_leaking_received_fds(self):
        import array
        a,b=self.owner.create_observer_bridge(artifact_only=True)
        handles=[tempfile.TemporaryFile() for _ in range(8)]
        try:
            a.sendmsg([struct.pack('!I',1)+b'x'],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[h.fileno() for h in handles]))])
            with self.assertRaises(self.rt.ObserverPacketFailure) as caught:self.rt.observer_receive_packet(b)
            self.assertIn('truncated',str(caught.exception))
            for h in handles:self.assertGreaterEqual(os.fstat(h.fileno()).st_ino,1)
        finally:
            for h in handles:h.close()
            a.close();b.close()

    def test_observer_constructor_rejects_without_closing_borrowed_descriptor(self):
        a,b=self.owner.create_observer_bridge(artifact_only=True)
        try:
            with self.assertRaisesRegex(ValueError,'nonce'):self.rt.NativeChannelObserver(a.fileno(),{},dict(device=1,inode=2))
            os.fstat(a.fileno())
            bad=dict(nonce='x',observer_run_nonce='y',observer_attempt_nonce='z',deployment={})
            with self.assertRaises(KeyError):self.rt.NativeChannelObserver(a.fileno(),bad,dict(device=1,inode=2))
            os.fstat(a.fileno());self.assertIsNone(a.gettimeout())
        finally:a.close();b.close()

    def test_production_requires_actual_linux_credentials_without_mocked_identity(self):
        if sys.platform.startswith('linux'):
            self.skipTest('Mac negative-path artifact check; genuine namespace credentials remain external')
        a,b=self.owner.create_observer_bridge(artifact_only=True)
        try:
            invocation=dict(nonce='x',observer_run_nonce='y',observer_attempt_nonce='z',deployment={'sha256':'0'*64},native_namespace={'CPU':True})
            with self.assertRaisesRegex(ValueError,'Linux'):self.rt.NativeChannelObserver(a.fileno(),invocation,dict(device=1,inode=2))
            with self.assertRaisesRegex(ValueError,'Linux'):self.owner.create_observer_bridge()
            os.fstat(a.fileno())
        finally:a.close();b.close()

    def test_actual_invalid_ack_rejects_exact_message_binding(self):
        with self.root() as root:
            anchor,bridge,retainer,observer=self.manual_retainer(root)
            try:
                bogus=self.rt.encoded(dict(schema='p4-original-native-observer-ack-v1',sequence=0,
                    message_raw_sha256='0'*64,binding_sha256=self.rt.sha(self.rt.encoded(observer.binding)),
                    durable_registration=True,receipt_sha256='0'*64))
                self.rt.observer_send_packet(retainer.sock,bogus)
                with self.assertRaisesRegex(ValueError,'exact durable message'):observer.send('CPU-invalid-ACK',{})
                self.assertEqual(observer.sequence,0)
                packet=self.rt.observer_receive_packet(retainer.sock)
                self.assertEqual(json.loads(packet['raw'])['kind'],'CPU-invalid-ACK')
            finally:self.close_manual(anchor,bridge,retainer,observer)

    def test_pinned_runtime_loader_rejects_mutation_and_symlink(self):
        with self.root() as root:
            path=root/'runtime.py';raw=Path(self.rt.__file__).read_bytes();path.write_bytes(raw)
            checksum=self.rt.sha(raw)
            loaded=self.owner.load_runtime(path,checksum)
            self.assertEqual(loaded.BOOTSTRAP,self.rt.BOOTSTRAP)
            link=root/'link.py';link.symlink_to(path)
            with self.assertRaisesRegex(ValueError,'immutable'):self.owner.load_runtime(link,checksum)
            path.write_bytes(raw+b'# labelled CPU mutation\n')
            with self.assertRaisesRegex(ValueError,'immutable'):self.owner.load_runtime(path,checksum)

    def test_real_launch_ack_stages_and_original_native_stream_with_bridge_absent_in_child(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();child=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,"""import os,json
identity=BUFFERED_INVOCATION['original_native_observer']['supervisor_bridge_identity']
present=False
for fd in range(3,256):
    try:s=os.fstat(fd)
    except OSError:continue
    present=present or {'device':s.st_dev,'inode':s.st_ino}==identity
os.write(BUFFERED_INVOCATION['producer_channel_fd'],json.dumps({'labelled_CPU_bridge_present':present}).encode())
print('labelled CPU stdout',flush=True)
""")
                child,bridge=self.start_retainer(root,invocation,anchor)
                producer,native=self.owner.launch_and_retain(self.rt,invocation,anchor,anchor.path/'native.log',root/'returns',channel_observer_fd=bridge.detach())
                summary=self.reap(child,root);child=None
                self.assertFalse(json.loads(producer)['labelled_CPU_bridge_present'])
                self.assertTrue(native['wait4_observation_available']);self.assertEqual(native['exit_code'],0)
                self.assertFalse(summary['writer_quiescence_certified']);self.assertTrue(summary['unread_tail_unavailable'])
                self.assertFalse(summary['actual_kernel_noexec_or_namespace_enforcement'])
                self.assertEqual(set(summary['complete_original_emitted_raw_records']),{'invocation_buffer','native_record'})
                first=json.loads((root/'retained/epochs/0000/index.json').read_bytes())
                self.assertEqual({r['role'] for r in first['rows']},{'invocation','producer','preexec_receipt','custody_journal'})
                self.assertTrue(all(r['bytes']==0 for r in first['rows']))
                second=json.loads((root/'retained/epochs/0001/index.json').read_bytes())
                self.assertTrue(all(r['bytes']==0 for r in second['rows']))
                messages=[json.loads(p.read_bytes()) for p in sorted((root/'retained/messages').glob('*.payload.bin'))]
                self.assertEqual([v['kind'] for v in messages[:2]],['channels_created','named_files_created'])
                original_invocation=(anchor.path/'native.invocation.json').read_bytes().rstrip(b'\n')
                invocation_binding=json.loads(original_invocation)['original_native_observer']
                self.assertEqual(invocation_binding['binding'],messages[0]['binding'])
                self.assertEqual(summary['complete_original_emitted_raw_records']['invocation_buffer']['raw_sha256'],hashlib.sha256(original_invocation).hexdigest())
                self.assertEqual((root/'returns/native.json').read_bytes(),self.rt.encoded(native))
                self.assertEqual((root/'returns/preexec-receipt.bin').read_bytes(),b'')
                self.assertNotEqual(json.loads((root/'retained/retainer-start.json').read_bytes())['process_pid'],os.getpid())
                roles=summary['original_identities']
                for role in native['inherited_channel_identities']:
                    self.assertEqual({k:roles[role][k] for k in ('device','inode')},native['inherited_channel_identities'][role])
            finally:
                if bridge is not None:bridge.close()
                if child is not None:child.kill();child.wait()
                anchor.close()

    def test_wrong_nonce_rejects_before_invocation_write_or_native_popen(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();child=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,"raise RuntimeError('must not execute')\n")
                child,bridge=self.start_retainer(root,invocation,anchor,seconds=.3)
                invocation['observer_run_nonce']='different-reservation'
                with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                    self.rt.launch_native(invocation,anchor,anchor.path/'native.log',channel_observer_fd=bridge.fileno())
                self.assertIsNone(caught.exception.native_observation['argv'])
                self.assertIsNone(caught.exception.native_observation['pid'])
                self.assertFalse((anchor.path/'native.invocation.json').exists())
                summary=self.reap(child,root);child=None
                self.assertEqual(summary['messages'],0);self.assertEqual(summary['original_identities'],{})
                self.assertIn('binding differs',' '.join(summary['errors']))
                self.assertTrue((root/'retained/messages/0000.payload.bin').exists())
            finally:
                if bridge is not None:bridge.close()
                if child is not None:child.kill();child.wait()
                anchor.close()

    def manual_retainer(self,root):
        anchor=self.rt.OutputAnchor(root/'original',new=True).activate()
        invocation=dict(nonce='CPU-invocation',observer_run_nonce='CPU-run',observer_attempt_nonce='CPU-attempt',deployment={'sha256':'0'*64})
        a,b=self.owner.create_observer_bridge(artifact_only=True)
        retainer=self.owner.OriginalChannelRetainer(self.rt,b.detach(),root/'retained',self.binding(invocation,anchor),artifact_only=True)
        observer=self.rt.NativeChannelObserver(a.fileno(),invocation,anchor.root_identity)
        return anchor,a,retainer,observer

    def accept_message(self,retainer,observer,kind,value,roles=(),fds=(),*,sequence=None):
        raw=self.rt.encoded(dict(schema='p4-original-native-observer-message-v1',sequence=retainer.sequence if sequence is None else sequence,
            previous_raw_sha256=retainer.previous,kind=kind,binding=observer.binding,roles=list(roles),value=value))
        packet=dict(header=struct.pack('!I',len(raw)),raw=raw,fds=list(fds),credentials=None,credential_raw_base64=None)
        retainer.accept(packet)
        ack=self.rt.observer_receive_packet(observer.sock)
        self.assertEqual(json.loads(ack['raw'])['message_raw_sha256'],self.rt.sha(raw))

    def close_manual(self,anchor,bridge,retainer,observer):
        observer.detach();bridge.close()
        for fd in retainer.originals.values():os.close(fd)
        retainer.sock.close();retainer.anchor.close();anchor.close()

    def register_manual(self,retainer,observer,anchor,handles):
        import fcntl
        roles=['root_anchor','invocation','producer','preexec_receipt','custody_journal']
        fds=[anchor.fd]+[h.fileno() for h in handles]
        value={role:dict(self.rt.identity(os.fstat(fd)),access_mode=fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE,anonymous=os.fstat(fd).st_nlink==0) for role,fd in zip(roles,fds)}
        self.accept_message(retainer,observer,'channels_created',value,roles,[os.dup(fd) for fd in fds])
        files=[]
        for role in ('invocation_file','stdout_log'):
            h=anchor.open_exclusive(anchor.path/role,binary=True);files.append(h)
        fds=[h.fileno() for h in files];roles=['invocation_file','stdout_log']
        value={role:dict(self.rt.identity(os.fstat(fd)),access_mode=fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE,anonymous=False) for role,fd in zip(roles,fds)}
        self.accept_message(retainer,observer,'named_files_created',value,roles,[os.dup(fd) for fd in fds])
        return files

    def test_retainer_samples_original_after_path_replacement_and_detects_prefix_conflict(self):
        with self.root() as root:
            anchor,bridge,retainer,observer=self.manual_retainer(root);handles=[tempfile.TemporaryFile(dir=anchor.path) for _ in range(4)];files=[]
            try:
                files=self.register_manual(retainer,observer,anchor,handles)
                files[1].write(b'original stdout prefix');os.fsync(files[1].fileno())
                retainer.capture('test-original')
                (anchor.path/'stdout_log').rename(root/'moved-original-log');(anchor.path/'stdout_log').write_bytes(b'replacement log')
                retainer.capture('test-replaced-name')
                self.assertEqual((retainer.anchor.path/'epochs/0003/stdout_log.bin').read_bytes(),b'original stdout prefix')
                os.ftruncate(files[1].fileno(),0);os.pwrite(files[1].fileno(),b'overwrite',0)
                retainer.capture('test-overwrite')
                index=json.loads((retainer.anchor.path/'epochs/0004/index.json').read_bytes())
                self.assertTrue(next(r for r in index['rows'] if r['role']=='stdout_log')['prior_prefix_conflict'])
                self.assertFalse(any(r['writer_quiescence_certified'] for r in index['rows']))
            finally:
                try:
                    for h in files:
                        try:h.close()
                        except ValueError:pass  # Expected original pathname replacement detection.
                finally:
                    for h in handles:h.close()
                    self.close_manual(anchor,bridge,retainer,observer)

    def test_retainer_partial_frame_and_controller_eof_never_certify_quiescence(self):
        with self.root() as root:
            anchor,bridge,retainer,observer=self.manual_retainer(root)
            try:
                raw=struct.pack('!I',9)+b'prefix'
                bridge.sendall(raw);bridge.shutdown(socket.SHUT_WR)
                summary=retainer.serve(max_seconds=.15,interval=.05)
                self.assertTrue(summary['bridge_eof']);self.assertTrue(summary['errors'])
                self.assertFalse(summary['writer_quiescence_certified'])
                self.assertEqual((root/'retained/incoming-error.header.bin').read_bytes()+(root/'retained/incoming-error.payload.bin').read_bytes(),raw)
            finally:observer.detach();bridge.close();anchor.close()
        with self.root() as root:
            anchor,bridge,retainer,observer=self.manual_retainer(root);read,write=os.pipe()
            try:
                os.write(write,b'unverified controller bytes');os.close(write);write=None
                summary=retainer.serve(controller_fd=read,max_seconds=.2,interval=.05)
                self.assertEqual(summary['reason'],'controller_stop_observation_unverified')
                self.assertFalse(summary['writer_quiescence_certified'])
                self.assertEqual((root/'retained/outside-controller-observation.bin').read_bytes(),b'unverified controller bytes')
            finally:
                if write is not None:os.close(write)
                os.close(read);observer.detach();bridge.close();anchor.close()

    def test_retainer_constructor_failure_preserves_unconsumed_descriptor(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'original',new=True).activate()
            a,b=self.owner.create_observer_bridge(artifact_only=True)
            invocation=dict(nonce='x',observer_run_nonce='y',observer_attempt_nonce='z',deployment={'sha256':'0'*64})
            (root/'already-existing').mkdir()
            try:
                with self.assertRaisesRegex(ValueError,'new'):
                    self.owner.OriginalChannelRetainer(self.rt,b.fileno(),root/'already-existing',self.binding(invocation,anchor),artifact_only=True)
                os.fstat(b.fileno());self.assertIsNone(b.gettimeout())
            finally:a.close();b.close();anchor.close()

    def test_retainer_summary_write_failure_keeps_truthful_available_summary(self):
        with self.root() as root:
            anchor,bridge,retainer,observer=self.manual_retainer(root)
            original=retainer.write
            def failed(relative,raw):
                if relative=='retainer-summary.json':raise OSError('labelled CPU summary storage failure')
                return original(relative,raw)
            try:
                bridge.shutdown(socket.SHUT_WR)
                with patch.object(retainer,'write',new=failed):
                    with self.assertRaises(self.owner.RetainerLifecycleFailure) as caught:
                        retainer.serve(max_seconds=.1,interval=.05)
                summary=caught.exception.available_summary
                self.assertTrue(summary['bridge_eof']);self.assertFalse(summary['writer_quiescence_certified'])
                self.assertTrue(summary['unread_tail_unavailable'])
                self.assertFalse((root/'retained/retainer-summary.json').exists())
            finally:observer.detach();bridge.close();anchor.close()

    def test_duplicate_sequence_bad_roles_and_incomplete_or_conflicting_raw_fragments_reject(self):
        cases=['roles','count','duplicate','offset','hash','total','finish']
        for case in cases:
            with self.subTest(case=case),self.root() as root:
                anchor,bridge,retainer,observer=self.manual_retainer(root);handles=[tempfile.TemporaryFile(dir=anchor.path) for _ in range(4)];files=[]
                try:
                    if case=='roles':
                        with self.assertRaisesRegex(ValueError,'five roles'):
                            self.accept_message(retainer,observer,'channels_created',{},['wrong'])
                    elif case=='count':
                        with self.assertRaisesRegex(ValueError,'FD count'):
                            self.accept_message(retainer,observer,'channels_created',{},['root_anchor','invocation','producer','preexec_receipt','custody_journal'])
                    else:
                        files=self.register_manual(retainer,observer,anchor,handles)
                        fragment=dict(record='a'*32,record_kind='native_record',offset=0,total_bytes=6,raw_sha256=self.rt.sha(b'abcdef'),raw_base64=base64.b64encode(b'abc').decode())
                        self.accept_message(retainer,observer,'raw_piece',fragment)
                        if case=='duplicate':
                            with self.assertRaisesRegex(ValueError,'sequence'):self.accept_message(retainer,observer,'raw_piece',fragment,sequence=2)
                        elif case=='finish':
                            with self.assertRaisesRegex(ValueError,'complete'):self.accept_message(retainer,observer,'finished',dict(success=False,actual_wait4_available=False))
                        else:
                            fragment.update(offset=3,raw_base64=base64.b64encode(b'def').decode())
                            if case=='offset':fragment['offset']=2
                            if case=='hash':fragment['raw_base64']=base64.b64encode(b'xxx').decode()
                            if case=='total':fragment['total_bytes']=7
                            with self.assertRaises(ValueError):self.accept_message(retainer,observer,'raw_piece',fragment)
                    self.assertFalse(retainer.finished)
                finally:
                    for h in files+handles:h.close()
                    self.close_manual(anchor,bridge,retainer,observer)

    def test_actual_supervisor_sigkill_retainer_keeps_original_raw_prefixes_without_final_record(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();retainer=None;supervisor=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,"""import os
os.write(BUFFERED_INVOCATION['producer_channel_fd'],b'labelled CPU producer prefix')
os.write(BUFFERED_INVOCATION['custody_journal_fd'],b'{labelled CPU journal prefix')
print('labelled CPU stdout prefix',flush=True)
""")
                retainer,bridge=self.start_retainer(root,invocation,anchor,seconds=1.3)
                config=root/'supervisor-config.json';marker=root/'actual-worker-reaped.json'
                config.write_bytes(self.rt.encoded(dict(output=str(anchor.path),identity=anchor.root_identity,registry=anchor.registry(),invocation=invocation,marker=str(marker))))
                argv=[sys.executable,'-I','-B','-S','-c',SUPERVISOR_KILL_CHILD,str(OWNER_RETAINER_PATH),str(self.rt.__file__),
                    self.rt.sha(Path(self.rt.__file__).read_bytes()),str(config),str(anchor.fd),str(bridge.fileno())]
                with (root/'supervisor.stdout.log').open('wb') as out,(root/'supervisor.stderr.log').open('wb') as err:
                    supervisor=subprocess.Popen(argv,pass_fds=(anchor.fd,bridge.fileno()),stdout=out,stderr=err)
                bridge.close();deadline=time.monotonic()+8
                while not marker.exists() and supervisor.poll() is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(marker.exists(),(root/'supervisor.stderr.log').read_text())
                self.assertTrue(json.loads(marker.read_bytes())['reaped'])
                supervisor.kill();status=supervisor.wait(timeout=5)
                (root/'supervisor-actual-kill-binding.json').write_bytes(self.rt.encoded(dict(argv=argv,pid=supervisor.pid,exit_code=status,signal=signal.SIGKILL,actual_worker_reap=json.loads(marker.read_bytes()))))
                self.assertEqual(status,-signal.SIGKILL);supervisor=None
                summary=self.reap(retainer,root);retainer=None
                self.assertTrue(summary['bridge_eof']);self.assertFalse(summary['launcher_finished'])
                self.assertFalse(summary['native_record_fragment_emitted']);self.assertNotIn('native_record',summary['complete_original_emitted_raw_records'])
                self.assertFalse(summary['writer_quiescence_certified']);self.assertTrue(summary['unread_tail_unavailable'])
                final=root/'retained/epochs'/('%04d'%(summary['epochs']-1))
                self.assertEqual((final/'producer.bin').read_bytes(),b'labelled CPU producer prefix')
                self.assertEqual((final/'preexec_receipt.bin').read_bytes(),b'{"labelled_CPU_receipt_prefix":')
                self.assertEqual((final/'custody_journal.bin').read_bytes(),b'{labelled CPU journal prefix')
                self.assertEqual((final/'stdout_log.bin').read_bytes(),b'labelled CPU stdout prefix\n')
            finally:
                if bridge is not None:bridge.close()
                for child in (supervisor,retainer):
                    if child is not None:child.kill();child.wait()
                anchor.close()

    def test_adapter_preserves_original_failure_and_bytes_when_export_write_fails(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();a,b=self.owner.create_observer_bridge(artifact_only=True)
            native=dict(preexec_receipt_raw_base64='',preexec_receipt_raw_bytes=0,preexec_receipt_raw_sha256=self.rt.sha(b''),
                preexec_kernel_observation=None,preexec_receipt_parse_error=None,preexec_receipt_channel_identity={'device':1,'inode':2},
                inherited_channel_identities={'preexec_receipt':{'device':1,'inode':2}},custody_journal_raw_base64=base64.b64encode(b'opaque journal').decode())
            failure=self.rt.NativeLaunchFailure(ValueError('original launch rejection'),native,b'opaque producer')
            try:
                original_write=self.rt.OutputAnchor.write
                def damaged(instance,path,raw):
                    if Path(path).name=='preexec-receipt.bin':raise OSError('labelled CPU export failure')
                    return original_write(instance,path,raw)
                with patch.object(self.rt,'launch_native',side_effect=failure),patch.object(self.rt.OutputAnchor,'write',new=damaged):
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.owner.launch_and_retain(self.rt,{},anchor,anchor.path/'x.log',root/'returns',channel_observer_fd=a.detach())
                self.assertIs(caught.exception,failure);self.assertEqual(caught.exception.producer_bytes,b'opaque producer')
                self.assertIn('export failure',' '.join(caught.exception.retention_errors))
                self.assertEqual((root/'returns/producer.bin').read_bytes(),b'opaque producer')
                self.assertEqual((root/'returns/native.json').read_bytes(),self.rt.encoded(native))
            finally:a.close();b.close();anchor.close()

    def test_bad_ack_or_closed_retainer_rejects_and_owned_child_reaps_with_actual_wait4(self):
        # An actual child is reaped after observer delivery fails, with no poll/wait substitute.
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();retainer=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,'import time\ntime.sleep(60)\n')
                retainer,bridge=self.start_retainer(root,invocation,anchor)
                original_send=self.rt.NativeChannelObserver.send
                def failed(observer,kind,value,*args,**kwargs):
                    if kind=='popen_returned':raise OSError('labelled CPU observer ACK failure')
                    return original_send(observer,kind,value,*args,**kwargs)
                with patch.object(self.rt.NativeChannelObserver,'send',new=failed):
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.rt.launch_native(invocation,anchor,anchor.path/'native.log',channel_observer_fd=bridge.fileno())
                native=caught.exception.native_observation
                self.assertTrue(native['wait4_observation_available']);self.assertEqual(native['exit_code'],-signal.SIGKILL)
                with self.assertRaises(ChildProcessError):os.wait4(native['pid'],os.WNOHANG)
                summary=self.reap(retainer,root);retainer=None
                self.assertTrue(summary['launcher_finished']);self.assertFalse(summary['writer_quiescence_certified'])
            finally:
                if bridge is not None:bridge.close()
                if retainer is not None:retainer.kill();retainer.wait()
                anchor.close()

    def test_actual_retainer_sigkill_rejects_launch_and_reaps_original_owned_worker(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();retainer=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,'import time\ntime.sleep(60)\n')
                retainer,bridge=self.start_retainer(root,invocation,anchor)
                original=self.rt.subprocess.Popen
                def launched(*args,**kwargs):
                    child=original(*args,**kwargs)
                    retainer.kill();status=retainer.wait(timeout=3)
                    self.assertEqual(status,-signal.SIGKILL)
                    (root/'retainer-actual-sigkill.json').write_bytes(self.rt.encoded(dict(pid=retainer.pid,exit_code=status)))
                    return child
                with patch.object(self.rt.subprocess,'Popen',new=launched):
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.rt.launch_native(invocation,anchor,anchor.path/'native.log',channel_observer_fd=bridge.fileno())
                native=caught.exception.native_observation
                self.assertTrue(native['wait4_observation_available']);self.assertEqual(native['exit_code'],-signal.SIGKILL)
                self.assertIn('observer_delivery_error',native)
                with self.assertRaises(ChildProcessError):os.wait4(native['pid'],os.WNOHANG)
                self.assertFalse((root/'retained/retainer-summary.json').exists())
                self.assertTrue((root/'retained/messages/0001.receipt.json').exists())
                retainer=None
            finally:
                if bridge is not None:bridge.close()
                if retainer is not None:retainer.kill();retainer.wait()
                anchor.close()

    def test_observer_failure_after_child_exit_preserves_exact_wait4_without_Popen_kill_poll(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'output',new=True).activate();retainer=None;bridge=None
            try:
                invocation=self.invocation(root,anchor,"print('labelled CPU worker naturally finished',flush=True)\n")
                retainer,bridge=self.start_retainer(root,invocation,anchor)
                original=self.rt.NativeChannelObserver.send
                def failed(observer,kind,value,*args,**kwargs):
                    if kind=='popen_returned':
                        deadline=time.monotonic()+5
                        while b'naturally finished' not in (anchor.path/'native.log').read_bytes() and time.monotonic()<deadline:time.sleep(.01)
                        time.sleep(.1)
                        raise OSError('labelled CPU observer failure after natural child exit')
                    return original(observer,kind,value,*args,**kwargs)
                with patch.object(self.rt.NativeChannelObserver,'send',new=failed),patch.object(self.rt.subprocess.Popen,'kill',side_effect=AssertionError('Popen kill may poll/reap')):
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.rt.launch_native(invocation,anchor,anchor.path/'native.log',channel_observer_fd=bridge.fileno())
                native=caught.exception.native_observation
                self.assertTrue(native['wait4_observation_available']);self.assertEqual(native['exit_code'],0)
                with self.assertRaises(ChildProcessError):os.wait4(native['pid'],os.WNOHANG)
                self.reap(retainer,root);retainer=None
            finally:
                if bridge is not None:bridge.close()
                if retainer is not None:retainer.kill();retainer.wait()
                anchor.close()


class ObserverRejectionEvidenceArtifactTests(unittest.TestCase):
    setUpClass=OriginalChannelTransportArtifactTests.__dict__['setUpClass']
    root=OriginalChannelTransportArtifactTests.root
    binding=OriginalChannelTransportArtifactTests.binding

    def start_peer(self,root,bridge,mode):
        peer=OWNER_RETAINER_PATH.parent/'observer_rejection_cpu_peer.py'
        expected=os.environ.get('P4_DRIVER_R8_CPU_PEER_SHA256')
        if expected is not None:self.assertEqual(hashlib.sha256(peer.read_bytes()).hexdigest(),expected)
        argv=[sys.executable,'-I','-B','-S',str(peer),str(OWNER_RETAINER_PATH),str(self.rt.__file__),
            self.rt.sha(Path(self.rt.__file__).read_bytes()),str(bridge.fileno()),str(root),mode]
        with (root/'peer.stdout.log').open('wb') as out,(root/'peer.stderr.log').open('wb') as err:
            child=subprocess.Popen(argv,pass_fds=(bridge.fileno(),),stdout=out,stderr=err)
        (root/'actual-peer-process.json').write_bytes(self.rt.encoded(dict(argv=argv,pid=child.pid,ppid=os.getpid())))
        return child

    def reap_peer(self,child,root):
        try:pid,status,usage=os.wait4(child.pid,0)
        except BaseException:child.kill();child.wait();raise
        child.returncode=os.waitstatus_to_exitcode(status)
        (root/'actual-peer-exit.json').write_bytes(self.rt.encoded(dict(pid=pid,ppid=os.getpid(),raw_wait_status=status,
            exit_code=child.returncode,through_exit_rss_bytes=self.rt.rss_bytes(usage),CPU_artifact_only=True)))
        self.assertEqual(child.returncode,0,(root/'peer.stderr.log').read_text())

    def test_semantic_registration_rejection_persists_all_actual_originals_and_bytes(self):
        import fcntl
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'original',new=True).activate()
            invocation=dict(nonce='opaque-invocation',observer_run_nonce='opaque-run',observer_attempt_nonce='opaque-attempt',deployment={'sha256':'0'*64})
            a,b=self.owner.create_observer_bridge(artifact_only=True);child=None;handles=[]
            try:
                binding=self.binding(invocation,anchor)
                (root/'receiver-config.json').write_bytes(self.rt.encoded(dict(binding=binding)))
                child=self.start_peer(root,b,'receiver');b.close()
                handles=[tempfile.TemporaryFile(dir=anchor.path) for _ in range(4)]
                roles=['root_anchor','invocation','producer','preexec_receipt','custody_journal'];fds=[anchor.fd]+[h.fileno() for h in handles]
                value={role:dict(self.rt.identity(os.fstat(fd)),access_mode=fcntl.fcntl(fd,fcntl.F_GETFL)&os.O_ACCMODE,
                    anonymous=os.fstat(fd).st_nlink==0) for role,fd in zip(roles,fds)}
                actual=[self.rt.identity(os.fstat(fd)) for fd in fds]
                value['custody_journal']['inode']+=1
                raw=self.rt.encoded(dict(schema='p4-original-native-observer-message-v1',sequence=0,previous_raw_sha256=None,
                    kind='channels_created',binding=dict(binding,launcher_pid_namespace=os.getpid()),roles=roles,value=value))
                self.rt.observer_send_packet(a,raw,fds)
                self.reap_peer(child,root);child=None
                stored=json.loads((root/'retained/incoming-error.observation.json').read_bytes())
                self.assertEqual(stored['received_unvalidated_fd_identities'],actual)
                self.assertEqual([r['original_fstat_identity'] for r in stored['received_descriptor_observations']],actual)
                self.assertEqual((root/'retained/incoming-error.header.bin').read_bytes(),struct.pack('!I',len(raw)))
                self.assertEqual((root/'retained/incoming-error.payload.bin').read_bytes(),raw)
                self.assertEqual(stored['payload_raw_sha256'],self.rt.sha(raw))
                self.assertTrue(stored['received_unvalidated_fds_closed'])
                self.assertTrue(all(r['close_succeeded'] for r in stored['descriptor_cleanup_observations']))
                self.assertFalse(stored['sender_declarations_used_as_original_authority'])
                self.assertFalse(stored['acceptance_certified']);self.assertFalse(stored['Linux_identity_or_enforcement_certified'])
                self.assertNotEqual(actual[-1]['inode'],value['custody_journal']['inode'])
                summary=json.loads((root/'retained/retainer-summary.json').read_bytes())
                self.assertEqual(summary['messages'],0);self.assertEqual(summary['original_identities'],{})
                self.assertFalse(summary['launcher_finished'])
                diagnostic=json.loads((root/'diagnostic-receiver-observations.json').read_bytes())
                self.assertEqual(stored['received_unvalidated_fd_identities'],diagnostic['identities'])
                self.assertEqual(stored['credential_raw_base64'],diagnostic['credential_raw_base64'])
            finally:
                if child is not None:child.kill();child.wait()
                for h in handles:h.close()
                a.close();b.close();anchor.close()

    def test_official_adapter_retains_original_semantic_and_partial_ACK_rejections(self):
        for mode in ('noncanonical','wrong_binding','invalid_fd','partial_payload','partial_header'):
            with self.subTest(mode=mode),self.root() as root:
                anchor=self.rt.OutputAnchor(root/'original',new=True).activate()
                a,b=self.owner.create_observer_bridge(artifact_only=True);child=None
                try:
                    child=self.start_peer(root,b,mode);b.close()
                    invocation=dict(nonce='opaque-invocation',observer_run_nonce='opaque-run',observer_attempt_nonce='opaque-attempt',deployment={'sha256':'0'*64})
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.owner.launch_and_retain(self.rt,invocation,anchor,anchor.path/'never-started.log',root/'available',channel_observer_fd=a.detach())
                    self.reap_peer(child,root);child=None
                    native=caught.exception.native_observation;observed=native['observer_rejection_observations'][0]
                    self.assertIsNone(native['pid']);self.assertIsNone(native['raw_wait_status']);self.assertIsNone(native['through_exit_child_peak_rss_bytes'])
                    self.assertFalse(native['wait4_observation_available']);self.assertIsNone(native['invocation_buffer_sha256'])
                    self.assertEqual(caught.exception.observer_rejection_observations,native['observer_rejection_observations'])
                    self.assertIsInstance(caught.exception.__cause__,self.rt.ObserverPacketFailure)
                    for kind in ('header','payload'):
                        original=(root/('diagnostic-sent-'+kind+'.bin')).read_bytes()
                        self.assertEqual(base64.b64decode(observed[kind+'_raw_base64'],validate=True),original)
                        self.assertEqual(observed[kind+'_raw_sha256'],self.rt.sha(original))
                        self.assertEqual((root/('available/observer-rejections/0000.'+kind+'.bin')).read_bytes(),original)
                    self.assertEqual((root/'available/native.json').read_bytes(),self.rt.encoded(native))
                    self.assertEqual(json.loads((root/'available/observer-rejections/0000.observation.json').read_bytes()),observed)
                    self.assertEqual((root/'available/producer.bin').read_bytes(),b'')
                    self.assertFalse((anchor.path/'never-started.invocation.json').exists())
                    self.assertFalse(observed['acceptance_certified']);self.assertFalse(observed['sender_declarations_used_as_original_authority'])
                    self.assertFalse(observed['Linux_identity_or_enforcement_certified'])
                    if mode=='invalid_fd':
                        actual=json.loads((root/'diagnostic-ACK-descriptor-identity.json').read_bytes())
                        self.assertEqual(observed['received_unvalidated_fd_identities'],[actual])
                        self.assertTrue(observed['received_unvalidated_fds_closed'])
                        self.assertTrue(observed['descriptor_cleanup_observations'][0]['close_succeeded'])
                    else:self.assertEqual(observed['received_unvalidated_fd_identities'],[])
                finally:
                    if child is not None:child.kill();child.wait()
                    a.close();b.close();anchor.close()

    def test_unavailable_descriptor_is_reported_without_inventing_an_identity_or_close(self):
        with tempfile.TemporaryFile() as handle:fd=os.dup(handle.fileno())
        os.close(fd)
        packet=dict(header=b'\x00\x00',raw=b'actual prefix',fds=[fd],credentials=None,credential_raw_base64=None)
        observed=self.rt.observer_received_packet_observation(packet,phase='labelled_CPU_unavailable_descriptor')
        self.assertEqual(observed['received_unvalidated_fd_identities'],[])
        row=observed['received_descriptor_observations'][0]
        self.assertIsNone(row['original_fstat_identity']);self.assertEqual(row['fstat_error']['errno'],9)
        self.rt.observer_close_received_fds([fd],observed)
        self.assertFalse(observed['received_unvalidated_fds_closed'])
        self.assertFalse(observed['descriptor_cleanup_observations'][0]['close_succeeded'])
        self.assertEqual(observed['descriptor_cleanup_observations'][0]['close_error']['errno'],9)

    def test_inline_original_ACK_survives_an_export_failure(self):
        with self.root() as root:
            anchor=self.rt.OutputAnchor(root/'original',new=True).activate()
            a,b=self.owner.create_observer_bridge(artifact_only=True);child=None
            try:
                child=self.start_peer(root,b,'noncanonical');b.close()
                invocation=dict(nonce='opaque-invocation',observer_run_nonce='opaque-run',observer_attempt_nonce='opaque-attempt',deployment={'sha256':'0'*64})
                original_write=self.rt.OutputAnchor.write
                def failed(instance,path,raw):
                    if Path(path).name=='0000.payload.bin':raise OSError('labelled CPU observer export failure')
                    return original_write(instance,path,raw)
                with patch.object(self.rt.OutputAnchor,'write',new=failed):
                    with self.assertRaises(self.rt.NativeLaunchFailure) as caught:
                        self.owner.launch_and_retain(self.rt,invocation,anchor,anchor.path/'never-started.log',root/'available',channel_observer_fd=a.detach())
                self.reap_peer(child,root);child=None
                observed=caught.exception.observer_rejection_observations[0]
                self.assertEqual(base64.b64decode(observed['payload_raw_base64'],validate=True),(root/'diagnostic-sent-payload.bin').read_bytes())
                self.assertIn('observer export failure',' '.join(caught.exception.retention_errors))
                self.assertEqual((root/'available/native.json').read_bytes(),self.rt.encoded(caught.exception.native_observation))
                self.assertIsInstance(caught.exception.__cause__,self.rt.ObserverPacketFailure)
            finally:
                if child is not None:child.kill();child.wait()
                a.close();b.close();anchor.close()


if __name__ == "__main__":
    unittest.main()
