"""Bounded synthetic corrections; real loader/Source/shape guards are retained."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from spacing_rerun import sourceqa_input_context as ctx
from spacing_rerun import sourceqa_protocol as source
from test_confirmation_dependency import seal
from test_sourceqa_input_propagation import retained_json


class SourceQAInputCorrectionTests(unittest.TestCase):
    def setUp(self):
        retained_root = os.environ.get("P4_SOURCEQA_CORRECTION_FIXTURES")
        if retained_root:
            self.root = Path(retained_root).resolve() / self._testMethodName
            self.root.mkdir(parents=True)
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)

    def test_symlink_substitution_at_actual_open_rejected_for_identical_bytes(self):
        ref = retained_json(self.root, "leaf.json", seal(dict(schema="synthetic-leaf")))
        leaf = Path(ref["path"])
        target = self.root / "identical.json"
        target.write_bytes(leaf.read_bytes())
        original_open = ctx.os.open
        calls = []
        def substitute(path, flags, *args, **kwargs):
            calls.append(flags)
            leaf.unlink()
            leaf.symlink_to(target)
            return original_open(path, flags, *args, **kwargs)
        with patch.object(ctx.os, "open", side_effect=substitute), self.assertRaisesRegex(ValueError, "symlink"):
            ctx.checked_input_descriptor(ref)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0] & os.O_NOFOLLOW)
        self.assertTrue(leaf.is_symlink())
        self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), ref["file_sha256"])

    def test_direct_symlink_directory_and_fifo_rejected_without_blocking(self):
        ref = retained_json(self.root, "regular.json", seal(dict(schema="synthetic")))
        link = self.root / "link.json"
        link.symlink_to(ref["path"])
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        for path in (link, self.root, fifo):
            with self.subTest(kind=str(path)), self.assertRaises(ValueError):
                ctx.checked_input_descriptor(dict(ref, path=str(path)))
        self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))

    def test_same_opened_regular_descriptor_bytes_parse_once_and_mapping_identity(self):
        mapping = {"d" * 64: dict(original_missing_exit_fields_preserved=True)}
        adapter = seal(dict(schema="spacing-separate-original-trace-caller-custody-v1",
            binding_proof_sha256="d" * 64, original_trace_input_files_by_proof=mapping,
            original_source_proof_trace_status_contract_helper_restamped=False,
            semantic_source_property_approval_created=False, source_commands_executed=False))
        ref = retained_json(self.root, "adapter.json", adapter)
        leaf = Path(ref["path"])
        original_bytes = leaf.read_bytes()
        actual_open, actual_stat, actual_fdopen, actual_parse = ctx.os.open, ctx.os.fstat, ctx.os.fdopen, ctx.json.loads
        descriptors, stats, read_bytes, parsed = [], [], [], []
        def open_once(*args, **kwargs):
            fd = actual_open(*args, **kwargs)
            descriptors.append(fd)
            return fd
        def stat_opened(fd):
            stats.append(fd)
            # The regular object is already open. A later path replacement
            # does not change which bytes this same descriptor identifies.
            leaf.rename(self.root / "original-opened.json")
            leaf.write_text('{"substituted":true}\n')
            return actual_stat(fd)
        class Reader:
            def __init__(self, opened):
                self.opened = opened
            def __enter__(self):
                self.opened.__enter__()
                return self
            def __exit__(self, *args):
                return self.opened.__exit__(*args)
            def read(self):
                value = self.opened.read()
                read_bytes.append(value)
                return value
        def fdopen(fd, *args, **kwargs):
            self.assertEqual(fd, descriptors[0])
            return Reader(actual_fdopen(fd, *args, **kwargs))
        def parse(value, *args, **kwargs):
            self.assertIs(value, read_bytes[0])
            result = actual_parse(value, *args, **kwargs)
            parsed.append(result)
            return result
        with patch.object(ctx.os, "open", side_effect=open_once), patch.object(ctx.os, "fstat", side_effect=stat_opened), \
             patch.object(ctx.os, "fdopen", side_effect=fdopen), patch.object(ctx.json, "loads", side_effect=parse):
            inputs = ctx.load_sourceqa_inputs(dict(expected_evaluator_metadata=None, original_trace_custody=ref))
        self.assertEqual(stats, descriptors)
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(read_bytes, [original_bytes])
        self.assertEqual(len(parsed), 1)
        self.assertIs(inputs["original_trace_input_files_by_proof"], parsed[0]["original_trace_input_files_by_proof"])
        self.assertEqual(inputs["original_trace_input_files_by_proof"], mapping)
        with self.assertRaises(OSError):
            actual_stat(descriptors[0])

    def test_wrong_raw_or_canonical_pin_still_rejected_and_fd_closed(self):
        ref = retained_json(self.root, "original.json", seal(dict(schema="synthetic")))
        actual_open = ctx.os.open
        for name in ("file_sha256", "content_sha256"):
            opened = []
            def observe(*args, **kwargs):
                fd = actual_open(*args, **kwargs)
                opened.append(fd)
                return fd
            with self.subTest(pin=name), patch.object(ctx.os, "open", side_effect=observe), self.assertRaises(ValueError):
                ctx.checked_input_descriptor(dict(ref, **{name: "0" * 64}))
            self.assertEqual(len(opened), 1)
            with self.assertRaises(OSError):
                os.fstat(opened[0])

    def accounting_fixture(self, declared):
        helper = self.root / "helper.py"
        helper.write_text("Synthetic retained helper bytes; never executed.\n")
        contract = seal(dict(schema="synthetic-original-contract",
            authorized_offline_packing_script_path=str(helper),
            authorized_offline_packing_script_file_sha256=hashlib.sha256(helper.read_bytes()).hexdigest()))
        contract_ref = retained_json(self.root, "contract.json", contract)
        contract_info = dict(path=contract_ref["path"], file_sha256=contract_ref["file_sha256"],
                             utf8=Path(contract_ref["path"]).read_text())
        if declared:
            contract_info[declared] = contract_ref["content_sha256"]
        response = seal(dict(schema="synthetic-original-response",
            author_output_contract=dict(path=contract_ref["path"], file_sha256=contract_ref["file_sha256"], sha256=contract["sha256"])))
        response_ref = retained_json(self.root, "response.json", response)
        proof = seal(dict(schema="synthetic-original-proof"))
        proof_ref = retained_json(self.root, "proof.json", proof)
        mapping = {proof["sha256"]: dict(author_output_contract=contract_info,
            packing_helper=dict(path=str(helper), file_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(), utf8=helper.read_text()))}
        adapter = seal(dict(schema="spacing-separate-original-trace-caller-custody-v1",
            binding_proof_sha256=proof_ref["content_sha256"], binding_proof_path=proof_ref["path"],
            binding_proof_file_sha256=proof_ref["file_sha256"], original_response_binding=dict(response_ref, sha256=response["sha256"]),
            original_trace_input_files_by_proof=mapping, original_source_proof_trace_status_contract_helper_restamped=False,
            semantic_source_property_approval_created=False, source_commands_executed=False))
        adapter_ref = retained_json(self.root, "adapter.json", adapter)
        evidence = dict(receipt_file=dict(path=response_ref["path"], file_sha256=response_ref["file_sha256"],
                                         utf8=Path(response_ref["path"]).read_text()), original_trace_input_files=mapping[proof["sha256"]])
        return dict(expected_evaluator_metadata=None, original_trace_custody=adapter_ref), mapping, proof, response, evidence, helper, contract_ref

    def check_contract_accounting(self, declared):
        descriptors, mapping, proof, response, evidence, helper, contract = self.accounting_fixture(declared)
        before = copy.deepcopy(mapping)
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("synthetic_contract_accounting_driver", root / "scripts/confirmation_analysis_performance.py")
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        actual_parse = json.loads
        parsed = []
        def observe(value, *args, **kwargs):
            parsed.append(value)
            return actual_parse(value, *args, **kwargs)
        with patch.object(json, "loads", side_effect=observe):
            inputs, refs = ctx.load_sourceqa_inputs(descriptors, include_references=True)
            self.assertEqual(len(parsed), 1, "One actual descriptor decode")
            evidence["original_trace_input_files"] = inputs["original_trace_input_files_by_proof"][proof["sha256"]]
            files = source._trace_virtual_inputs(response, evidence)
            self.assertEqual(len(parsed), 2, "Actual separate embedded contract decode")
            selected = ctx.sourceqa_json_references(refs)
            shape = driver.observed_shape([], [], selected)
        retained = next(r for r in refs if r["path"] == contract["path"])
        self.assertEqual(inputs["original_trace_input_files_by_proof"], before)
        self.assertEqual(mapping, before)
        self.assertIs(evidence["original_trace_input_files"], inputs["original_trace_input_files_by_proof"][proof["sha256"]])
        if declared:
            self.assertEqual(retained[declared], contract["content_sha256"])
            self.assertEqual(retained["content_sha256"], contract["content_sha256"])
        else:
            self.assertNotIn("canonical_sha256", retained)
            self.assertNotIn("sha256", retained)
            self.assertNotIn("content_sha256", retained)
        self.assertIn(contract["path"], {r["path"] for r in selected})
        self.assertNotIn(str(helper), {r["path"] for r in selected})
        self.assertEqual(files[str(helper)], helper.read_bytes())
        self.assertEqual(shape["peak_live_shared_json_graphs"], len(selected))
        self.assertEqual(shape["loaded_json_bytes"], sum(Path(r["path"]).stat().st_size for r in selected))
        self.assertEqual(len(parsed), 2 + len(selected))
        self.assertEqual(retained["accounting_provenance"], "derived_reference_from_unchanged_original_custody")
        # Actual descriptor graph, embedded Source contract graph and shape's
        # separately decoded shared graphs are distinct parses. This is a
        # bounded stage count, not a full Driver/Builder memory attestation.
        print("SOURCEQA_PARSE_ACCOUNTING", dict(descriptor_decodes=1, actual_Source_embedded_contract_decodes=1,
                                              actual_shape_shared_decodes=len(selected), helper_JSON_decodes=0))

    def test_actual_contract_canonical_field_parser_and_shape_accounting(self):
        self.check_contract_accounting("canonical_sha256")

    def test_actual_contract_sha_field_parser_and_shape_accounting(self):
        self.check_contract_accounting("sha256")

    def test_actual_contract_missing_canonical_preserved_and_still_JSON_accounted(self):
        self.check_contract_accounting(None)

    def test_driver_shape_and_builder_reference_routes_use_same_selector(self):
        path = Path(__file__).resolve().parents[1] / "scripts/confirmation_analysis_performance.py"
        tree = ast.parse(path.read_text())
        definitions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name in ("worker_point", "_run"):
            calls = [n for n in ast.walk(definitions[name]) if isinstance(n, ast.Call) and
                     isinstance(n.func, ast.Name) and n.func.id == "sourceqa_json_references"]
            self.assertEqual(len(calls), 1)
        runner = ast.unparse(definitions["_run"])
        self.assertIn("builder_shared_json_refs=spec['builder_shared_json_refs'] + sourceqa_json_refs", runner)
        self.assertIn("sourceqa_json_refs = sourceqa_json_references(sourceqa_refs)", runner)
        worker = ast.unparse(definitions["worker_point"])
        self.assertIn("payload['additional_parse_refs'] + sourceqa_json_references(sourceqa_refs)", worker)


if __name__ == "__main__":
    unittest.main()
