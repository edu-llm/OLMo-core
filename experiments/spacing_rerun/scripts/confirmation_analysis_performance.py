#!/usr/bin/env python3
"""Observe CPU analysis cost from genuine metadata and synthetic saved predictions.

This driver does not run a tokenizer or model. The strict R3 interface requires
separate genuine frozen policies for the first 2, 8 and 32 preparations. The
driver cannot turn a chosen-n approval into approvals for other registrations.
Only the preregistration and completion eligibility boundaries are fixtures.
All content, provenance, aggregate and final-policy validators remain real.
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import types
import zipfile
import uuid

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[1]
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spacing_rerun.common import digest, require
import performance_driver_runtime as execution

CAPTURED_DRIVER_SOURCE = globals().get("CAPTURED_DRIVER_SOURCE", Path(__file__).read_bytes())
ACTIVE_OUTPUT = None
ACTIVE_DEPLOYMENT = None

INPUT_SCHEMA = "p4-analysis-performance-driver-input-v5"
BENCHMARK_SCHEMA = "p4-analysis-benchmark-synthetic-predictions-v1"
POINTS = (2, 8, 32)
FUNCTIONS = ("validate_final_policy", "saved_bundle_diagnostics", "complete_contrast_report",
             "influence_scores", "complement_contrast", "build_saved_dependency_report")
STUBS = ["verify_preregistered_dependencies", "completion_seal"]
SHAPE_KEYS = ("batch_size", "microbatch_size", "context_length", "loss_tokens_per_example",
              "learning_rate", "warmup_steps", "eval_context_length", "eval_batch_size",
              "generation_batch_size", "max_answer_tokens", "generic_eval_sequences",
              "qa_per_step", "review_cap", "cohort_size")
MEMORY_KEYS = ("loaded_json_bytes", "schedule_rows", "parse_records", "registered_replicates",
               "prediction_rows", "source_units")
MARKERS = dict(production_approval=False, scientific_outcome=False,
               test_fixture_preregistration=True)


def sealed(value):
    return dict(value, sha256=digest(value))


def check_seal(value):
    require(value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
            "JSON content seal differs")
    return value


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_path(value):
    require(isinstance(value, str) and value and value == str(Path(value).resolve()),
            "Artifact paths must use canonical absolute literal paths")
    path = Path(value)
    require(path.is_absolute() and str(path.resolve()) == str(path),
            "Artifact paths must be canonical absolute paths")
    return path


def raw_ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), file_sha256=file_hash(path))


def checked_json(ref, *, content_seal=True):
    path = canonical_path(ref["path"])
    require(path.is_file() and not path.is_symlink(), "Retained JSON is absent or a symlink")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == ref["file_sha256"], "Retained JSON bytes differ")
    value = json.loads(raw)
    if content_seal:
        check_seal(value)
        require(value["sha256"] == ref["content_sha256"], "Retained JSON content identity differs")
    return value


def write_json(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n"
    if ACTIVE_OUTPUT is not None:
        ACTIVE_OUTPUT.write(path, data)
    else:
        # Standalone labelled artifact helpers still reject symlink traversal.
        anchor = execution.OutputAnchor(Path(path).parent, new=False)
        try:
            anchor.write(path, data)
        finally:
            anchor.close()


def isolate_output(output, production_roots):
    output = canonical_path(output)
    roots = [canonical_path(root) for root in production_roots]
    require(roots and len(set(roots)) == len(roots), "Declare every production run root once")
    require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in roots),
            "Synthetic output must be outside every declared production run root")
    require(not output.exists(), "Synthetic output directory must be new")
    return execution.OutputAnchor(output), roots


def checkpoint_inputs(rows, prepared):
    from spacing_rerun.schedule import ARMS
    require(len(rows) == len(prepared), "Six retained checkpoint inputs required for every preparation")
    result, physical_claims = [], {}
    for row, ref in zip(rows, prepared):
        require(row["manifest_sha256"] == ref["manifest_sha256"], "Checkpoint manifest order differs")
        pins = row["files"]
        require(len(pins) == 6 and {p["digest_slot"] for p in pins} == {"stage1", *ARMS},
                "Each replicate needs six real digest operations, one per checkpoint slot")
        for pin in pins:
            path = canonical_path(pin["path"])
            require(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
                    "An actual retained checkpoint is absent or empty")
            require(len(pin.get("file_sha256", "")) == 64 and
                    all(c in "0123456789abcdef" for c in pin["file_sha256"]),
                    "Actual checkpoint byte identity is required")
            require(type(pin.get("bytes")) is int and pin["bytes"] == path.stat().st_size and
                    isinstance(pin.get("sample_stage"), str) and pin["sample_stage"] == pin["sample_stage"].strip() and pin["sample_stage"],
                    "Retain the actual checkpoint sample size and source stage")
            actual = path.stat()
            physical_key = (actual.st_dev, actual.st_ino)
            claim = {k: v for k, v in pin.items() if k not in ("digest_slot", "path", "registered_metadata_manifest_sha256")}
            require(physical_key not in physical_claims or physical_claims[physical_key] == claim,
                    "Repeated physical checkpoint has contradictory native producer/bytes/stage provenance")
            physical_claims[physical_key] = claim
        result.append(row)
    return result


def local_git(*args):
    return subprocess.check_output(["git", *args], cwd=REPOSITORY)


def committed_code(spec):
    """Bind all actual package bytes and the driver to the truthful measured HEAD."""
    from spacing_rerun import confirmation_dependency as dependency
    from spacing_rerun.confirmation_lane import reporting_specification
    commit = spec["code_commit"]
    require(len(commit) == 40 and local_git("rev-parse", "HEAD").decode().strip() == commit,
            "Declared measured commit must equal actual Git HEAD")
    algorithm = canonical_path(spec["algorithm_file"]["path"])
    reporting = canonical_path(spec["reporting_file"]["path"])
    algorithm_spec = checked_json(spec["algorithm_file"])
    report_spec = checked_json(spec["reporting_file"])
    require(spec["algorithm_sha256"] == dependency.SUPPORTED_ALGORITHM_SHA256 and
            algorithm_spec.get("algorithm") == dependency.algorithm_v3() and
            report_spec == reporting_specification() and
            spec["reporting_specification_sha256"] == report_spec["sha256"],
            "Use actual final supported algorithm/reporting hashes")
    paths = sorted((EXPERIMENT / "spacing_rerun").rglob("*.py")) + [algorithm, reporting,
            Path(execution.__file__).resolve()]
    rows = []
    for path in paths:
        require(path.is_file() and not path.is_symlink() and path.is_relative_to(REPOSITORY),
                "Actual CPU package source must be retained under its Git repository")
        relative = str(path.relative_to(REPOSITORY))
        sha = file_hash(path)
        require(hashlib.sha256(local_git("show", commit + ":" + relative)).hexdigest() == sha,
                "CPU package differs from its actual committed source: " + relative)
        rows.append(dict(git_path=relative, file_sha256=sha))
    require(len({row["git_path"] for row in rows}) == len(rows), "CPU scope repeats a source file")
    scope = sealed(dict(schema="p4-cpu-performance-package-scope-v1",
                        algorithm_sha256=dependency.SUPPORTED_ALGORITHM_SHA256,
                        files=sorted(rows, key=lambda row: row["git_path"])))
    driver = dict(raw_ref(__file__), git_path=str(Path(__file__).resolve().relative_to(REPOSITORY)))
    require(hashlib.sha256(local_git("show", commit + ":" + driver["git_path"])).hexdigest() == driver["file_sha256"],
            "Measurement driver must be committed before an actual measurement")
    return dict(gpu_count=0, code_commit=commit, package_scope=scope, measurement_driver=driver)


def load_real_prepared(ref, *, confirmation_only=True):
    from spacing_rerun.prepare import load_prepared
    path = canonical_path(ref["path"])
    require(file_hash(path / "manifest.json") == ref["manifest_file_sha256"],
            "Prepared manifest raw bytes differ")
    manifest, schedule, examples = load_prepared(path)
    require(manifest["sha256"] == ref["manifest_sha256"] and manifest.get("test_fixture") is not True and
            (not confirmation_only or manifest.get("mode") == "confirmation"),
            "Only actual validated preparation metadata can supply this sample")
    require(all(k in manifest["config"] and manifest["config"][k] is not None for k in SHAPE_KEYS),
            "Actual numerical metadata has a missing shape")
    return manifest, schedule, examples


def checkpoint_archive_shape(path):
    require(zipfile.is_zipfile(path), "Checkpoint size/type sample is not a torch-save ZIP archive")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(any(entry.filename.endswith("/data.pkl") for entry in entries) and
                any(entry.filename.endswith("/version") for entry in entries),
                "Checkpoint archive lacks actual torch-save metadata members")
        storage = [dict(archive_member=entry.filename, bytes=entry.file_size,
                        compressed_bytes=entry.compress_size)
                   for entry in entries if "/data/" in entry.filename]
        require(storage, "Checkpoint archive has no actual stored tensor payloads")
        return dict(raw_ref(path), bytes=Path(path).stat().st_size,
                    archive_format="torch_save_zip", storage_entries=storage)


def checkpoint_sample_binding(spec, target_model, roots):
    """Validate a real compatible checkpoint size/type sample without loading it.

    Repeated paths are valid digest operations. They supply no independent main
    checkpoints or cold-cache timing evidence. Raw production provenance stays
    separate from synthetic outcomes and remains the owner's responsibility.
    """
    from spacing_rerun import SCHEMA
    from spacing_rerun.confirmation import numerical_sources, NUMERICAL_SCOPE
    sample = spec.get("checkpoint_sample")
    require(isinstance(sample, dict),
            "UNRESOLVED_CHECKPOINT_NATIVE_ASSOCIATION: genuine checkpoint producer sample is required")
    require(sample.get("producer_association_adapter") is not None,
            "UNRESOLVED_CHECKPOINT_NATIVE_ASSOCIATION: genuine per-physical-file native producer validation is required")
    # A final owner adapter must authenticate producer execution and exact saves.
    # Raw-log hashes or copied identity labels never replace that contract.
    producer_validation = execute_exact_adapter(sample["producer_association_adapter"],
        dict(sample=sample, checkpoints=spec["checkpoints"], code_commit=spec["code_commit"]),
        required_function="validate_checkpoint_native_associations")
    ref = sample["prepared"]
    require(any(canonical_path(ref["path"]).is_relative_to(root) for root in roots),
            "Checkpoint sample preparation root must be disclosed")
    manifest, schedule, examples = load_real_prepared(ref, confirmation_only=False)
    model = {k: manifest[k] for k in target_model}
    numerical = {k: manifest["config"][k] for k in SHAPE_KEYS}
    require(model == target_model and sample["checkpoint_schema"] == SCHEMA,
            "Checkpoint digest sample model/revision/tokenizer or checkpoint type differs")
    measured_commit = sample["code_commit"]
    require(len(measured_commit) == 40 and all(c in "0123456789abcdef" for c in measured_commit),
            "Retain the checkpoint producer's actual measured Git commit")
    identities = {}
    for name, functions in NUMERICAL_SCOPE.items():
        relative = "experiments/spacing_rerun/spacing_rerun/" + name
        measured = numerical_sources(local_git("show", measured_commit + ":" + relative).decode(), functions)
        current = numerical_sources(local_git("show", spec["code_commit"] + ":" + relative).decode(), functions)
        require(measured == current, "Actual checkpoint producer numerical code differs: " + name)
        identities[name] = measured
    evidence = sample["raw_log_refs"]
    require(evidence and all(raw_ref(canonical_path(ref["path"])) == ref for ref in evidence),
            "Actual retained checkpoint-producer provenance byte refs are required")
    del manifest, schedule, examples
    paths = [pin["path"] for row in spec["checkpoints"] for pin in row["files"]]
    physical = sorted(set(paths))
    physical_inodes = {(Path(path).stat().st_dev, Path(path).stat().st_ino) for path in physical}
    require(producer_validation.get("native_save_execution_reconciled") is True and
            {r["path"] for r in producer_validation.get("physical_files", [])} == set(physical) and
            producer_validation.get("sample_manifest_sha256") == ref["manifest_sha256"] and
            producer_validation.get("producer_code_commit") == measured_commit,
            "UNRESOLVED_CHECKPOINT_NATIVE_ASSOCIATION: native producer associations do not match actual sample")
    operation_associations = reconcile_checkpoint_associations(spec["checkpoints"], producer_validation)
    file_shapes = [checkpoint_archive_shape(path) for path in physical]
    return sealed(dict(schema="p4-checkpoint-digest-actual-compatible-sample-v1", **MARKERS,
        prepared=ref, model_identity=model, numerical_config=numerical,
        code_commit=measured_commit, target_code_commit=spec["code_commit"],
        checkpoint_schema=sample["checkpoint_schema"], raw_log_refs=evidence,
        native_producer_validation=producer_validation, operation_associations=operation_associations,
        numerical_git_identity=sealed(dict(schema="p4-measurement-numerical-git-identity-v1",
            measured_commit=measured_commit, final_commit=spec["code_commit"],
            numerical_scope={k:list(v) for k,v in NUMERICAL_SCOPE.items()}, function_sha256=identities,
            method="actual_git_function_source_identity")),
        physical_file_count=len(physical_inodes), physical_path_count=len(physical),
        digest_operation_count=len(paths), repeated_paths=len(paths) != len(physical),
        repeated_physical_inodes=len(paths) != len(physical_inodes),
        physical_files=file_shapes, tensor_values_or_pickle_loaded=False,
        independent_main_checkpoints_claimed=False, cache_control="not_evicted_repeated_reads_may_be_warm",
        cold_cache_throughput_measured=False,
        limitation="Digest timing covers actual retained compatible sample bytes with possible warm cache. It does not certify independent main checkpoint existence or cold storage throughput. The explicit memory margin supplies no missing I/O rate."))


def reconcile_checkpoint_associations(checkpoints, producer):
    physical = producer.get("physical_files", [])
    paths = {p["path"] for checkpoint in checkpoints for p in checkpoint["files"]}
    require(len(physical) == len(paths) and {r.get("path") for r in physical} == paths,
            "Native checkpoint associations lack exact unique physical path coverage")
    by_path = {r["path"]: r for r in physical}
    result = []
    for checkpoint in checkpoints:
        for pin in checkpoint["files"]:
            row = by_path[pin["path"]]
            require(all(row.get(k) == pin[k] for k in ("path", "file_sha256", "bytes", "sample_stage")) and
                    row.get("native_save_refs") and row.get("native_execution_refs"),
                    "UNRESOLVED_CHECKPOINT_NATIVE_ASSOCIATION: each repeated operation must bind its actual native producer")
            for key in ("native_save_refs", "native_execution_refs", "producer_code_commit", "producer_manifest_sha256"):
                require(key not in pin or pin[key] == row.get(key), "Repeated checkpoint producer provenance contradicts actual association")
            result.append(dict(row, digest_slot=pin["digest_slot"],
                               registered_metadata_manifest_sha256=checkpoint["manifest_sha256"]))
    return result


def verify_exact_adapter(adapter, *, required_function):
    """Parent verifies the external committed adapter before any worker starts."""
    source = canonical_path(adapter["path"])
    raw = source.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == adapter["file_sha256"], "Final owner adapter source changed")
    require(adapter.get("function") == required_function and isinstance(adapter.get("code_commit"), str) and
            len(adapter["code_commit"]) == 40 and adapter.get("git_path") and adapter.get("independent_review_binding"),
            "UNRESOLVED_OWNER_ADAPTER: exact committed function and genuine external review binding are required")
    require(local_git("show", adapter["code_commit"] + ":" + adapter["git_path"]) == raw,
            "Final owner adapter differs from its actual committed source")
    tree = ast.parse(raw)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == required_function]
    require(len(nodes) == 1, "Final owner adapter lacks one exact exported function")
    lines = raw.splitlines(keepends=True)
    function_source = b"".join(lines[nodes[0].lineno - 1:nodes[0].end_lineno])
    require(execution.sha(function_source) == adapter.get("function_source_sha256"),
            "Final owner adapter function source identity differs")
    binding = adapter["independent_review_binding"]
    report = checked_json(binding["report"])
    completion = verify_adapter_review(binding, report)
    require(report.get("implementation_verdict") == "approved" and
            report.get("approved_implementation_file_sha256") == adapter["file_sha256"] and
            required_function in report.get("approved_exported_functions", []),
            "UNRESOLVED_OWNER_ADAPTER: retained exact review/task binding differs; root must authenticate original custody")
    return dict(adapter=adapter, exact_source_sha256=execution.sha(raw),
                review_seal=report["sha256"], verified_function=required_function,
                completed_review_binding=completion)


def verify_adapter_review(binding, report):
    from spacing_rerun.confirmation_dependency import _completed_review
    evidence = binding.get("completed_review_evidence")
    require(isinstance(evidence, dict),
            "UNRESOLVED_OWNER_ADAPTER: original exact completed review transport is required")
    report_raw = Path(binding["report"]["path"]).read_bytes()
    status_raw = Path(binding["actual_task_status"]["path"]).read_bytes()
    checked_json(binding["actual_task_status"], content_seal=False)
    require(evidence.get("receipt_file", {}).get("utf8", "").encode() == report_raw and
            evidence.get("receipt_file", {}).get("file_sha256") == binding["report"]["file_sha256"] and
            evidence.get("task_status_file", {}).get("utf8", "").encode() == status_raw and
            evidence.get("task_status_file", {}).get("file_sha256") == binding["actual_task_status"]["file_sha256"],
            "Owner adapter original receipt/status transport contradicts exact retained files")
    status = evidence.get("task_status", {})
    require(status.get("workState") == "result_available",
            "Owner adapter latest actual review result is unavailable")
    return _completed_review(report, evidence, require_named_task=True)


def execute_exact_adapter(adapter, payload, *, required_function):
    """Execute captured bytes after parent verification; no Git child in worker."""
    source = canonical_path(adapter["path"])
    invocation = globals().get("BUFFERED_INVOCATION")
    if invocation:
        bindings = invocation.get("verified_owner_adapters", [])
        binding = next((r for r in bindings if r["adapter"] == adapter and
                        r["verified_function"] == required_function), None)
        rows = [r for r in invocation["deployment"]["files"] if r["path"] == str(source)]
        require(binding and len(rows) == 1, "Owner adapter lacks captured parent-verification binding")
        raw = execution.base64.b64decode(rows[0]["data"])
        require(execution.sha(raw) == binding["exact_source_sha256"] == adapter["file_sha256"],
                "Captured owner adapter differs from parent verified exact source")
        require(source.read_bytes() == raw, "Final owner adapter changed from its captured source buffer")
    else:
        verify_exact_adapter(adapter, required_function=required_function)
        raw = source.read_bytes()
    result = call_captured_adapter(raw, source, adapter["file_sha256"], payload, required_function)
    require(source.read_bytes() == raw, "Final owner adapter source changed through completion")
    return result


def call_captured_adapter(raw, source, checksum, payload, required_function):
    """Mechanical module execution only; caller supplies already verified bytes."""
    require(execution.sha(raw) == checksum, "Owner adapter buffer identity differs")
    name = "p4_exact_owner_adapter_" + checksum[:16]
    require(name not in sys.modules, "Owner adapter module identity is already occupied")
    module = types.ModuleType(name)
    module.__file__ = str(source)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(source), "exec", dont_inherit=True), module.__dict__)
        function = getattr(module, required_function, None)
        require(callable(function), "Final owner adapter lacks its required exported function")
        if required_function == "probe_builder_memory":
            result = function(payload["shared_json_refs"], payload["prepared_refs"], payload["report_ref"], payload["metadata"])
        else:
            result = function(payload)
    finally:
        sys.modules.pop(name, None)
    require(isinstance(result, dict), "Owner adapter did not return typed evidence")
    return result


def synthetic_metrics(rng, gold, behavior, *, answer_tokens=1, envelope=None, choices=0, correct_index=0, aliases=None):
    envelope = envelope or dict(max_answer_tokens=8, maximum_prediction_utf8_bytes=64, token_id_upper_bound=99999)
    loss, eos = .01 + rng.random() * 4, .01 + rng.random() * 4
    row = dict(answer_nll_sum=loss * answer_tokens, answer_tokens=answer_tokens, eos_nll=eos,
               loss=loss, loss_with_eos=(loss * answer_tokens + eos) / (answer_tokens + 1))
    if behavior:
        from spacing_rerun.data import normalize
        success = rng.randrange(2)
        size = envelope["maximum_prediction_utf8_bytes"]
        require(len(gold.encode()) <= size, "Saved string envelope is smaller than an actual fixed answer label")
        accepted = {normalize(s) for s in (aliases if aliases is not None else [gold])}
        require(normalize(gold) in accepted, "Native fixed answer is absent from its exact-match aliases")
        miss = next((s for s in ["", *(chr(i) for i in range(33, 127))] if normalize(s) not in accepted and len(s.encode()) <= size), None)
        require(miss is not None, "Synthetic miss cannot preserve actual alias scoring within the reviewed bound")
        stem = gold if success else miss
        require(len(stem.encode()) <= size, "Saved string envelope is smaller than an actual fixed answer label")
        row.update(exact_match=success,
                   prediction=stem + "\x0b" * (size - len(stem.encode())),
                   prediction_token_ids=[int(str(envelope["token_id_upper_bound"])) for _ in range(envelope["max_answer_tokens"])],
                   generation_terminated=0, stop_token=None)
        require(int(normalize(row["prediction"]) in accepted) == success,
                "Synthetic string padding changed native exact-match scoring")
        encoded_prediction = len(json.dumps(row["prediction"], ensure_ascii=False).encode())
        deficit = 6 * size + 2 - encoded_prediction
        row["_synthetic_json_prediction_escape_reserve"] = "\x0b" * ((deficit + 5) // 6)
        token_deficit = max(0, len(str(envelope["token_id_upper_bound"])) - len("null"))
        row["_synthetic_json_stop_token_width_reserve"] = "\x0b" * ((token_deficit + 5) // 6)
        row["_synthetic_decoded_prediction_surrogate"] = "\U00010000" * size
        row["_synthetic_prediction_envelope_method"] = "exact_UTF8_and_JSON_escape_bound_plus_separate_four_byte_decoded_surrogate"
        if choices:
            require(choices > 1 and 0 <= correct_index < choices, "Native MCQ metadata is invalid")
            sums = [-.01 - rng.random() * 10 for _ in range(choices)]
            means = [s / max(1, answer_tokens) for s in sums]
            others = [i for i in range(choices) if i != correct_index]
            row.update(mcq_accuracy=int(sums[correct_index] > max(sums[i] for i in others)),
                       mcq_accuracy_normalized=int(means[correct_index] > max(means[i] for i in others)),
                       mcq_margin=sums[correct_index] - max(sums[i] for i in others),
                       mcq_margin_normalized=means[correct_index] - max(means[i] for i in others),
                       candidate_log_likelihoods=sums, candidate_mean_log_likelihoods=means)
    return row


def synthetic_evaluation(manifest, binding, membership, rng, global_step, *, behavior=True, generic=True, envelope=None):
    """Use opaque root/provenance and target labels; never open real outcome files."""
    from spacing_rerun.evaluation import aggregate
    from spacing_rerun.confirmation_dependency import acquisition_form_mappings
    from spacing_rerun.units import GROUNDED_METRICS, UNIT_METRICS, GROUNDED_POLICY
    from spacing_rerun import SCHEMA
    golds = {fact["id"]: fact["answer"] for fact in manifest["split"]["facts"]}
    facts = {fact["id"]: fact for fact in manifest["split"]["facts"]}
    variants = {}
    for variant in ("canonical", "paraphrase"):
        rows = []
        for rid, meta in binding["record_mappings"].items():
            unit = meta["source_unit_ids"][0] if meta["role"] == "control" else meta["source_group_id"]
            rows.append(dict(id=rid, unit_id=unit, event=meta["event"], role=meta["role"],
                             **synthetic_metrics(rng, golds[rid], behavior, envelope=envelope,
                                answer_tokens=max(1, len(facts[rid].get("probe" if variant == "canonical" else
                                    "paraphrase_probe", {}).get("answer_ids", [0]))),
                                choices=len(facts[rid].get("mcq", {}).get("choices", [0, 1, 2, 3]))
                                        if variant == "canonical" else 0,
                                correct_index=facts[rid].get("mcq", {}).get("correct", 0),
                                aliases=facts[rid].get("aliases", [golds[rid]]))))
        variants[variant] = dict(facts=rows, aggregate=aggregate(rows))
    value = dict(MARKERS, schema=SCHEMA, manifest_sha256=manifest["sha256"], global_step=global_step,
                 rehearsal_unit_policy=manifest.get("rehearsal_unit_policy"),
                 metric_schema=GROUNDED_METRICS if manifest.get("rehearsal_unit_policy") == GROUNDED_POLICY else UNIT_METRICS,
                 variants=variants, generic_loss=None, qa_teaching_diagnostic=None,
                 timings={variant + "_loss_seconds": rng.random() for variant in variants},
                 wall_seconds=rng.random(), predictions="synthetic_seeded_random",
                 saved_string_envelope=envelope or dict(kind="labelled_artifact_test_only", maximum_prediction_utf8_bytes=64))
    if behavior:
        value["timings"].update({variant + "_generation_seconds": rng.random() for variant in variants})
        value["timings"]["mcq_seconds"] = rng.random()
        rows = [dict(id=rid, target_id=meta["target_id"], question_variant=meta["question_variant"],
                     unit_id=meta["source_group_id"], event=meta["event"], role="old",
                     training_exposures=0,
                     **synthetic_metrics(rng, meta["original_answer_label"], True, envelope=envelope))
                for rid, meta in acquisition_form_mappings(manifest["acquisition_qa_pool"]["records"], membership).items()]
        value["acquisition_qa_diagnostic"] = dict(facts=rows, aggregate=aggregate(rows)["old"],
            diagnostic_only=True, acquisition_supervision_pool=True, answer_targets=len(rows) // 2,
            question_variants=len(rows), seen_question_variants=0, unseen_question_variants=len(rows),
            in_sample_acquisition_questions=False,
            acquisition_policy=manifest.get("acquisition_policy"),
            aggregation="mean_variants_and_targets_within_unit_then_units_within_event_then_events",
            interpretation="Supervised acquisition question forms; never substitute for held-out-form acquisition.")
        value["timings"]["acquisition_qa_diagnostic_seconds"] = rng.random()
        value["acquisition_qa_similarity_diagnostic"] = None
        similarity = manifest.get("acquisition_qa_similarity")
        if similarity:
            low_ids = {p["id"] for p in similarity["probes"] if p["variant"] == "canonical" and p["max_jaccard"] < similarity["threshold"]}
            old_rows = [r for r in variants["canonical"]["facts"] if r["role"] == "old"]
            low_rows = [r for r in old_rows if r["id"] in low_ids]
            value["acquisition_qa_similarity_diagnostic"] = dict(diagnostic_only=True,
                similarity_report_sha256=similarity["sha256"], threshold=similarity["threshold"],
                canonical_old_probes=len(old_rows), low_similarity_probes=len(low_rows),
                interpretation="Lexical overlap sensitivity after supervised acquisition; not evidence of novel factual knowledge.",
                aggregate=aggregate(low_rows)["old"] if low_rows else None)
        if manifest["config"].get("qa_teaching_diagnostics", False):
            rows = [dict(id=rid, unit_id=meta["source_unit_ids"][0], event=meta["event"], role="qa",
                         **synthetic_metrics(rng, golds[rid], True, envelope=envelope,
                                             aliases=facts[rid].get("aliases", [golds[rid]])))
                    for rid, meta in binding["teaching_root_mappings"].items()]
            value["qa_teaching_diagnostic"] = dict(facts=rows, aggregate=aggregate(rows)["qa"],
                                                   diagnostic_only=True, in_sample_teaching_questions=True)
            value["timings"]["qa_teaching_diagnostic_seconds"] = rng.random()
    if generic and manifest.get("generic_eval"):
        value["generic_loss"] = .01 + rng.random() * 4
        value["timings"]["generic_seconds"] = rng.random()
    scalar_serialization_reserve(value)
    return value


def scalar_serialization_reserve(value):
    """Native finite floats use at most 24 JSON bytes under reviewed CPython."""
    def deficit(v):
        if type(v) is float: return max(0, 24 - len(json.dumps(v, allow_nan=False)))
        if isinstance(v, dict): return sum(deficit(r) for k, r in v.items() if not k.startswith("_synthetic_"))
        if isinstance(v, list): return sum(deficit(r) for r in v)
        return 0
    value["_synthetic_json_finite_float_width_reserve"] = "\x0b" * ((deficit(value) + 5) // 6)


def save_exposures(path, rows, examples, offset, stage2_steps=None, *, manifest=None, acquisition_tokens=0):
    from spacing_rerun import SCHEMA
    from spacing_rerun.acquisition import SOURCE_QA_ACQUISITION_POLICY
    new_tokens = 0
    anchor = ACTIVE_OUTPUT or execution.OutputAnchor(Path(path).parent, new=False)
    with anchor.open_exclusive(path) as handle:
        for i, row in enumerate(rows):
            new_tokens += sum(examples[key]["loss_tokens"] for key in row if key.startswith(("new/", "qa/")))
            acq = [key for key in row if key.startswith("acq/")]
            acquisition_tokens += sum(examples[key]["loss_tokens"] for key in acq)
            value = dict(MARKERS, schema=SCHEMA, cursor=i + 1, global_step=offset + i + 1, examples=row, row_sha256=digest(row),
                         phase="stage1" if stage2_steps is None else "stage2" if i < stage2_steps else "buffer",
                         phase_step=i, example_content_sha256=[examples[key]["content_sha256"] for key in row],
                         loss_tokens=sum(examples[key]["loss_tokens"] for key in row), loss=1.234567890123,
                         gradient_norm=1.234567890123, lr=(manifest or {}).get("config", {}).get("learning_rate", 0.00003),
                         loss_weight=1.0, new_qa_tokens_total=new_tokens,
                         rehearsal_unit_policy=(manifest or {}).get("rehearsal_unit_policy"),
                         content_tokens=sum(examples[key].get("content_tokens", examples[key]["loss_tokens"]) for key in row),
                         generic_filler_tokens=sum(examples[key].get("generic_filler_tokens", 0) for key in row),
                         padding_tokens=sum(examples[key].get("padding_tokens", 0) for key in row),
                         acquisition_policy=SOURCE_QA_ACQUISITION_POLICY, acquisition_qa_examples=acq,
                         acquisition_qa_tokens_total=acquisition_tokens)
            scalar_serialization_reserve(value)
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    if ACTIVE_OUTPUT is None:
        anchor.close()


def completion_seal(directory, manifest, prereg):
    """Explicit synthetic eligibility boundary; actual byte pins and seals are real."""
    from spacing_rerun.schedule import ARMS
    files = sorted([*directory.rglob("*.json"), *directory.rglob("*.jsonl")])
    return sealed(dict(MARKERS, schema="p4-synthetic-performance-completion-v1",
                       manifest_sha256=manifest["sha256"], preregistration_sha256=prereg["sha256"],
                       E=prereg["E"], arms={arm: dict(MARKERS) for arm in ARMS},
                       files=[dict(path=str(file.resolve()), sha256=file_hash(file)) for file in files]))


def native_evaluation_metadata(value, manifest, schedule, examples, executed, *, tag, epoch, stage2_step=None):
    """Reconstruct native metadata clocks solely from frozen metadata rows."""
    from spacing_rerun.acquisition import SOURCE_QA_ACQUISITION_POLICY
    old_units = sorted(key.split("/", 1)[1] for key in examples if key.startswith("old/"))
    last_update, last_new, new_tokens, acq_tokens, doses = {}, {}, 0, 0, collections.Counter()
    for step, row in enumerate(executed, 1):
        new_tokens += sum(examples[k]["loss_tokens"] for k in row if k.startswith(("new/", "qa/")))
        acq_tokens += sum(examples[k]["loss_tokens"] for k in row if k.startswith("acq/"))
        for key in row:
            if key.startswith("old/"):
                last_update[key.split("/", 1)[1]], last_new[key.split("/", 1)[1]] = step, new_tokens
            elif key.startswith("acq/"):
                doses[key] += 1
    value.update(tag=tag, epoch=epoch, stage2_step=stage2_step, acquisition_policy=SOURCE_QA_ACQUISITION_POLICY,
        acquisition_qa_tokens_total=acq_tokens,
        old_exposure_clock_definition="declaration exposures; acquisition QA doses have separate counters")
    if value.get("acquisition_qa_diagnostic"):
        diagnostic = value["acquisition_qa_diagnostic"]
        for row in diagnostic["facts"]:
            row["training_exposures"] = doses[f"acq/{row['target_id']}/{row['question_variant']}"]
        seen = sum(row["training_exposures"] > 0 for row in diagnostic["facts"])
        diagnostic.update(seen_question_variants=seen, unseen_question_variants=len(diagnostic["facts"]) - seen,
                          in_sample_acquisition_questions=seen == len(diagnostic["facts"]))
    if stage2_step is not None:
        from spacing_rerun.units import training_key
        value.update(global_buffer_delay=stage2_step - schedule["stage2_steps"],
            updates_since_last_old_exposure={key: len(executed) - last_update.get(key, 0) for key in old_units},
            new_qa_tokens_since_last_old_exposure={key: new_tokens - last_new.get(key, 0) for key in old_units},
            old_probe_to_exposure_unit={f["id"]: training_key(f) for f in manifest["split"]["facts"] if f["role"] == "old"})
    scalar_serialization_reserve(value)
    return value


def generate_bundle(directory, ref, policy, prereg, rng, envelope):
    from spacing_rerun.schedule import ARMS, stage1_epoch
    from spacing_rerun.units import schedule_records
    from spacing_rerun.confirmation_dependency import primary_aggregate_binding, _endpoint_labels
    manifest, schedule, examples = load_real_prepared(ref)
    prepared = next(row for row in policy["canonical_bindings"] if row["manifest_sha256"] == manifest["sha256"])
    # Read-only metadata links let the production compact reader stream the
    # original validator. Synthetic outcomes and logs remain under directory.
    ACTIVE_OUTPUT.mkdir(directory)
    ACTIVE_OUTPUT.symlink(directory / "prepared", str(canonical_path(ref["path"])))
    training = schedule_records(manifest["split"]["facts"], manifest["unit_registry"], manifest["qa_teaching_pool"]["records"])
    shared = [row for epoch in range(prereg["E"]) for row in
              stage1_epoch(training, epoch, manifest["config"], manifest["acquisition_qa_pool"]["records"])]
    save_exposures(directory / "stage1/exposures.jsonl", shared, examples, 0, manifest=manifest)
    fixed = synthetic_evaluation(manifest, prepared["binding"], policy["membership"], rng, len(shared), envelope=envelope)
    native_evaluation_metadata(fixed, manifest, schedule, examples, shared, tag=f"acquisition-E{prereg['E']:03}", epoch=prereg["E"])
    write_json(directory / f"stage1/evaluations/acquisition-E{prereg['E']:03}.json", fixed)
    # The builder's actual workload includes the evaluator's stage1 baseline.
    # Preserve and parse that shape separately; do not invent acquisition E=0.
    baseline = synthetic_evaluation(manifest, prepared["binding"], policy["membership"], rng, 0, generic=False, envelope=envelope)
    native_evaluation_metadata(baseline, manifest, schedule, examples, [], tag="baseline", epoch=0)
    write_json(directory / "synthetic-baseline.json", baseline)
    arms = {}
    primary_step = schedule["stage2_steps"] + prereg["primary_delay"]
    for arm in ARMS:
        save_exposures(directory / arm / "exposures.jsonl", schedule["arms"][arm], examples,
                       len(shared), schedule["stage2_steps"], manifest=manifest,
                       acquisition_tokens=sum(examples[k]["loss_tokens"] for row in shared for k in row if k.startswith("acq/")))
        for step in sorted(set(_endpoint_labels(schedule, manifest["config"]).values())):
            value = synthetic_evaluation(manifest, prepared["binding"], policy["membership"], rng,
                                         len(shared) + step, behavior=step == 0 or step >= schedule["stage2_steps"],
                                         generic=step in (0, len(schedule["arms"][arm])), envelope=envelope)
            native_evaluation_metadata(value, manifest, schedule, examples, shared + schedule["arms"][arm][:step],
                tag=f"stage2-{step:06}", epoch=prereg["E"], stage2_step=step)
            write_json(directory / arm / f"evaluations/stage2-{step:06}.json", value)
            if step == primary_step:
                arms[arm] = value["variants"]["canonical"]["aggregate"]
    require(set(arms) == set(ARMS), "Primary delay must be an actual saved schedule endpoint")
    completion = completion_seal(directory, manifest, prereg)
    primary = primary_aggregate_binding(manifest["sha256"], completion, prereg["primary_delay"], primary_step, arms)
    write_json(directory / "synthetic-completion.json", completion)
    write_json(directory / "synthetic-primary-binding.json", sealed(dict(MARKERS,
        schema="p4-synthetic-primary-binding-wrapper-v1", primary_aggregate_binding=primary)))
    # Compact input retains the genuine seal, not the labelled disk wrapper.
    return dict(path=str(directory.resolve()), manifest_sha256=manifest["sha256"],
                confirmation_audit_sha256=manifest["confirmation_audit"]["sha256"],
                completion=completion, primary_aggregate_binding=primary)


def count_prediction_rows(value):
    return sum(len(section["facts"]) for section in value["variants"].values()) + sum(
        len(value[key]["facts"]) for key in ("acquisition_qa_diagnostic", "qa_teaching_diagnostic") if isinstance(value.get(key), dict))


def observed_shape(prepared, bundles, additional_refs):
    """Counts bytes and rows actually parsed in this worker, without tokenization."""
    from spacing_rerun.units import schedule_records
    from spacing_rerun.confirmation_dependency import _saved_evaluation
    shape = dict.fromkeys(MEMORY_KEYS, 0)
    shape.update(cumulative_parsed_json_bytes=0, saved_evaluation_json_bytes=0,
                 exposure_jsonl_bytes=0, peak_live_shared_json_graphs=0,
                 peak_live_shared_serialized_bytes=0)
    shared_graphs = [checked_json(ref, content_seal=False) for ref in additional_refs]
    shape["peak_live_shared_json_graphs"] = len(shared_graphs)
    shape["peak_live_shared_serialized_bytes"] = sum(Path(ref["path"]).stat().st_size for ref in additional_refs)
    shape["loaded_json_bytes"] += shape["peak_live_shared_serialized_bytes"]
    for ref, bundle in zip(prepared, bundles):
        manifest, schedule, examples = load_real_prepared(ref)
        files = sorted(canonical_path(ref["path"]).rglob("*.json"))
        for file in files:
            require(file.is_file(), "Prepared JSON is absent")
            json.loads(file.read_bytes())
            shape["loaded_json_bytes"] += file.stat().st_size
        shape["schedule_rows"] += sum(len(rows) for rows in schedule["arms"].values())
        records = schedule_records(manifest["split"]["facts"], manifest["unit_registry"], manifest["qa_teaching_pool"]["records"])
        shape["parse_records"] += len(manifest["split"]["facts"]) + len(records) + len(manifest["acquisition_qa_pool"]["records"])
        shape["source_units"] += len(manifest["source_unit_registry"]["units"])
        shape["registered_replicates"] += 1
        directory = Path(bundle["path"])
        outcomes = [directory / "synthetic-baseline.json", *sorted((directory / "stage1/evaluations").glob("*.json"))]
        outcomes += [file for arm in schedule["arms"] for file in sorted((directory / arm / "evaluations").glob("*.json"))]
        for file in outcomes:
            value = json.loads(file.read_bytes())
            require(value.get("scientific_outcome") is False and value.get("production_approval") is False,
                    "Synthetic outcome markers are absent")
            shape["prediction_rows"] += count_prediction_rows(value)
            shape["saved_evaluation_json_bytes"] += file.stat().st_size
        shape["exposure_jsonl_bytes"] += sum(file.stat().st_size for file in directory.rglob("*.jsonl"))
        # Validate the additional baseline with the same production aggregate
        # and root validator, since the saved reporter consumes fixed-E only.
        policy = bundle.pop("_policy")
        registered = next(row for row in policy["canonical_bindings"] if row["manifest_sha256"] == manifest["sha256"])
        _saved_evaluation(json.loads((directory / "synthetic-baseline.json").read_bytes()),
                          manifest, registered["binding"], policy["membership"])
        del manifest, schedule, examples, records
    shape["cumulative_parsed_json_bytes"] = shape["loaded_json_bytes"] + shape["saved_evaluation_json_bytes"] + shape["exposure_jsonl_bytes"]
    require(len(shared_graphs) == shape["peak_live_shared_json_graphs"], "Concurrent shared graph lifetime lost")
    return shape


def profile_functions():
    from spacing_rerun import confirmation_dependency as dependency
    from spacing_rerun import confirmation_lane as lane
    functions = {getattr(dependency, name).__code__: name for name in FUNCTIONS if name != "complement_contrast"}
    functions[lane.complement_contrast.__code__] = "complement_contrast"
    counts = collections.Counter()
    counts.observations = []
    def profile(frame, event, argument):
        if event == "call" and frame.f_code in functions:
            counts[functions[frame.f_code]] += 1
            counts.observations.append(dict(function=functions[frame.f_code],
                **execution.code_identity(frame.f_code)))
    return counts, profile


def worker_point(payload):
    from spacing_rerun import confirmation_dependency as dependency
    from unittest.mock import patch
    frozen = checked_json(payload["frozen_policy"])
    prereg = checked_json(payload["preregistration"])
    check_fixture_preregistration(prereg, payload["prepared"])
    bundles = json.loads(Path(payload["bundles_path"]).read_bytes())
    counts, profile = profile_functions()
    sys.setprofile(profile)
    try:
        # No policy/source/audit/content/aggregate validators are replaced.
        with patch.object(dependency, "verify_preregistered_dependencies", new=lambda *args: None):
            report = dependency.build_saved_dependency_report(bundles, frozen, prereg)
        for bundle in bundles:
            bundle["_policy"] = frozen["policy"]
        shape = observed_shape(payload["prepared"], bundles, payload["additional_parse_refs"])
    finally:
        sys.setprofile(None)
    require(set(counts) == set(FUNCTIONS) and all(counts[name] > 0 for name in FUNCTIONS),
            "Full production algorithm did not exercise every required function")
    write_json(payload["report_path"], sealed(dict(MARKERS,
        schema="p4-synthetic-performance-report-wrapper-v1", production_report=report)))
    wrapper_ref = dict(raw_ref(payload["report_path"]),
        content_sha256=json.loads(Path(payload["report_path"]).read_bytes())["sha256"])
    builder_memory = execute_exact_adapter(payload["builder_memory_probe"], dict(
        prepared=payload["prepared"], shared_json_refs=payload["builder_shared_json_refs"],
        point_report_object=report, point_report_ref=wrapper_ref, n=len(payload["prepared"])),
        required_function="probe_analysis_live_memory")
    validate_builder_memory(builder_memory, payload, wrapper_ref)
    return dict(shape=shape, function_call_counts=dict(counts), stubbed=STUBS,
                function_observations=counts.observations,
                builder_memory=builder_memory, report=raw_ref(payload["report_path"]))


def validate_builder_memory(value, payload, report_ref):
    coverage = value.get("coverage", {})
    require(value.get("schema") == "p4-final-builder-live-memory-observation-v1" and
            type(value.get("n")) is int and value["n"] == len(payload["prepared"]) and
            value.get("prepared") == payload["prepared"] and value.get("point_report_ref") == report_ref and
            value.get("retained_same_report_object") is True and
            value.get("full_retained_metadata_validation_projection") is True and
            all(coverage.get(key) is True for key in ("actual_prepared_validator", "actual_dependency_guard",
                "actual_requirement_counter", "actual_metadata_workload_projection")) and
            coverage.get("actual_complete_cost_projection") is value.get("full_production_validation_projection") and
            type(value.get("full_production_validation_projection")) is bool and
            coverage.get("final_selected_n_or_approval_guards_claimed") is False and
            coverage.get("n_context") == "synthetic_analysis_performance_only" and
            all(value.get(k) is False for k in ("scientific_outcome", "production_approval", "preregistration_emitted")),
            "UNRESOLVED_FINAL_BUILDER_MEMORY: actual retained metadata paths/identities/limitations differ")
    scope = check_seal(value.get("builder_function_scope", {}))
    require(scope.get("schema") == "p4-builder-memory-function-scope-v1" and scope.get("functions"),
            "Builder memory lacks exact final function source scope")
    adapter = payload["builder_memory_probe"]
    invocation = globals().get("BUFFERED_INVOCATION", {})
    sources = [r for r in invocation.get("deployment", {}).get("files", []) if r["path"] == adapter["path"]]
    raw = execution.base64.b64decode(sources[0]["data"]) if sources else Path(adapter["path"]).read_bytes()
    require(execution.sha(raw) == adapter["file_sha256"], "Builder scope source is outside exact adapter binding")
    tree = ast.parse(raw)
    definitions = {n.name: hashlib.sha256(ast.get_source_segment(raw.decode(), n).encode()).hexdigest()
                   for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    required = {"probe_analysis_live_memory", "probe_builder_memory", "_probe_builder_memory", "parsed_prepared",
                "metadata_workload_projection", "retained_metadata_validation_projection", "real_requirements",
                "cost_projection", "_build", "Evidence"}
    rows = scope["functions"]
    require(len(rows) == len(required) and {r.get("name") for r in rows} == required and
            all(definitions.get(r["name"]) == r.get("source_sha256") for r in rows),
            "Builder memory function identities differ from exact captured final source")
    observations = value.get("live_graph_observations")
    shared = payload["builder_shared_json_refs"]
    shared = list(shared.values()) if isinstance(shared, dict) else shared
    require(value.get("shared_json_refs") == shared,
            "Builder memory retained another exact shared JSON registration")
    require(isinstance(observations, list) and observations and
            all(type(value.get(k)) is int and value[k] > 0 for k in
                ("cumulative_parsed_json_bytes", "peak_live_serialized_input_bytes", "peak_live_decoded_graph_bytes")) and
            value["peak_live_serialized_input_bytes"] == max(r["live_serialized_input_bytes"] for r in observations) and
            value["peak_live_decoded_graph_bytes"] == max(r["live_decoded_graph_bytes"] for r in observations),
            "Builder memory peaks disagree with actual retained lifetime observations")
    stages = collections.Counter(r.get("stage") for r in observations)
    require(stages["load_shared_json"] == len(shared) and stages["retain_actual_report"] == 1 and
            stages["during_actual_prepared_validation"] == len(payload["prepared"]) and
            stages["during_actual_metadata_projection"] == len(payload["prepared"]),
            "Builder lifetime observations omit actual per-n/shared/report validation paths")
    require(type(value.get("report_live_object_id")) is int and
            value["report_live_object_id"] != value.get("decoded_inner_report_object_id"),
            "Builder probe did not retain original report and independently decoded wrapper graphs")
    return value


def worker_probe(payload):
    from spacing_rerun.acquisition_grid import checkpoint_digest
    observations = []
    if payload["kind"] == "checkpoint_digest":
        checkpoint_inputs(payload["checkpoints"], payload["prepared"])
    associations = checkpoint_operation_associations(payload) if payload["kind"] == "checkpoint_digest" else None
    for ref in payload["prepared"]:
        if payload["kind"] == "load_prepared_and_audit":
            tick = time.monotonic()
            manifest, schedule, examples = load_real_prepared(ref)
            audit_sha = manifest["confirmation_audit"]["sha256"]
            del manifest, schedule, examples
            observations.append(dict(manifest_sha256=ref["manifest_sha256"],
                                     confirmation_audit_sha256=audit_sha,
                                     wall_seconds=time.monotonic() - tick))
        else:
            row = next(row for row in payload["checkpoints"] if row["manifest_sha256"] == ref["manifest_sha256"])
            for pin in row["files"]:
                tick = time.monotonic()
                actual = checkpoint_digest(pin["path"])
                elapsed = time.monotonic() - tick
                require(actual == pin["file_sha256"], "Actual checkpoint bytes differ from retained identity")
                observations.append(dict(pin, registered_metadata_manifest_sha256=ref["manifest_sha256"],
                                         sample_manifest_sha256=payload["checkpoint_sample"]["prepared"]["manifest_sha256"],
                                         observed_sha256=actual, bytes=Path(pin["path"]).stat().st_size,
                                         wall_seconds=elapsed))
                if associations is not None:
                    observations[-1]["native_operation_association"] = associations[len(observations) - 1]
    return dict(observations=observations, operation_count=len(observations),
                # A conservative observed single-operation duration. The builder
                # multiplies this by actual required calls, not a guessed rate.
                wall_seconds_per_operation=max(row["wall_seconds"] for row in observations))


def checkpoint_operation_associations(payload):
    sample = payload["checkpoint_sample"]
    if "operation_associations" not in sample:
        require(not payload.get("immutable_runtime"), "Production digest probe lacks per-operation native associations")
        return None
    associations = reconcile_checkpoint_associations(payload["checkpoints"], sample["native_producer_validation"])
    require(sample["operation_associations"] == associations,
            "Ordered digest operation map contradicts every native checkpoint occurrence")
    return associations


def rss_bytes(usage):
    return int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))


def observed_runtime():
    versions = {}
    for name in ("numpy", "scipy", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=sys.version, executable=sys.executable, platform=sys.platform,
                node=os.uname().nodename, installed_distributions=versions,
                slurm={key: os.environ[key] for key in
                       ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS") if key in os.environ},
                thread_environment={key: os.environ[key] for key in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if key in os.environ})


def validate_worker_result(payload, value, invocation, native):
    require(set(value) == {*MARKERS, "schema", "invocation_sha256", "nonce", "deployment_sha256",
            "result", "child_wall_seconds", "child_preexit_rss_bytes", "descendant_peak_rss_bytes",
            "observed_runtime", "loaded_code_identity", "output_registry", "parent_output_registry_sha256"} and
            value["schema"] == "p4-buffered-worker-result-v5" and all(value.get(k) is v for k, v in MARKERS.items()),
            "Worker result type/markers differ")
    require(value["invocation_sha256"] == invocation["payload_sha256"] and value["nonce"] == invocation["nonce"] and
            value["deployment_sha256"] == invocation["deployment"]["sha256"], "Worker result is from another exact invocation")
    require(native.get("inherited_channel_identities") == invocation.get("inherited_channel_identities") and
            isinstance(invocation.get("inherited_channel_identities"), dict) and
            set(invocation["inherited_channel_identities"]) == {"invocation", "producer", "preexec_receipt", "custody_journal"} and
            all(row["device"] == invocation["output_identity"]["device"] for row in invocation["inherited_channel_identities"].values()),
            "Inherited native channels do not bind the actual output device/invocation")
    journal_raw = execution.base64.b64decode(native["custody_journal_raw_base64"], validate=True)
    journal = execution.read_custody_journal(journal_raw, invocation)
    require(execution.sha(journal_raw) == native["custody_journal_raw_sha256"] and journal["complete_registered_journal"] and
            journal["terminal_kind"] == "success" and journal["original_registry"] == value["output_registry"],
            "Successful worker lacks exact complete original creation journal")
    require(value["parent_output_registry_sha256"] == execution.sha(execution.encoded(invocation["output_registry_before_child"])),
            "Worker output custody does not bind actual parent baseline")
    checked_output_custody(invocation, value["output_registry"], payload["worker_result_path"],
                           payload.get("report_path") if payload["kind"] == "point" else None)
    require(type(value["child_preexit_rss_bytes"]) is int and 0 < value["child_preexit_rss_bytes"] <= native["through_exit_child_peak_rss_bytes"] and
            value["descendant_peak_rss_bytes"] == 0, "Worker RSS contradicts its through-exit OS observation")
    require(isinstance(value["child_wall_seconds"], (int, float)) and 0 < value["child_wall_seconds"] <= native["wall_seconds"] and
            math.isfinite(value["child_wall_seconds"]), "Worker duration contradicts its native process duration")
    runtime = value["observed_runtime"]
    require(runtime["python"] == sys.version and runtime["platform"] == sys.platform and
            runtime["node"] == os.uname().nodename and Path(runtime["executable"]).resolve() == Path(sys.executable).resolve(),
            "Worker runtime contradicts actual invocation")
    require(runtime.get("slurm") == {key: os.environ[key] for key in
                ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_GPUS") if key in os.environ} and
            runtime.get("thread_environment") == {key: os.environ[key] for key in
                ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if key in os.environ},
            "Worker allocation/thread runtime contradicts actual invocation environment")
    loaded = value["loaded_code_identity"]
    require(set(loaded) == {"buffered_package_modules", "loaded_function_code_sha256", "compiler",
                "loaded_runtime_files", "loaded_native_libraries", "interpreter", "lifetime_runtime_imports",
                "native_process_enforcement", "prebootstrap_scope"}, "Loaded code/lifetime identity schema differs")
    require(loaded.get("interpreter") == invocation["deployment"]["interpreter"],
            "Loaded interpreter contradicts captured runtime identity")
    require(loaded.get("compiler") == invocation["deployment"]["compiler"] == execution.compiler_identity(),
            "Loaded compiler differs from exact captured compile settings")
    lifetime = loaded.get("lifetime_runtime_imports")
    require(isinstance(lifetime, list) and lifetime and any(r.get("phase") == "prebootstrap_startup" for r in lifetime),
            "Before-exec startup and lifetime runtime import census absent")
    phases = {"prebootstrap_startup", "reviewed_bootstrap_imports", "before_runtime_module_execution", "completed_runtime_module_execution"}
    before = set()
    approved = {r["path"]: r["file_sha256"] for r in (invocation.get("immutable_runtime") or {}).get("files", [])}
    for row in lifetime:
        require(row.get("phase") in phases and isinstance(row.get("module"), str), "Lifetime import observation schema differs")
        key = (row["module"], row.get("path"), row.get("file_sha256"), row.get("interpreter_sha256"))
        if row["phase"] == "completed_runtime_module_execution":
            require(key in before, "Runtime execution completed without a before-execution census record")
        else: before.add(key)
        if "path" in row:
            require(file_hash(row["path"]) == row["file_sha256"], "Lifetime runtime bytes changed through completion")
            if approved:
                require(approved.get(row["path"]) == row["file_sha256"] and row.get("approved_complete_census_enforced") is True,
                        "Lifetime import escaped approved before-exec runtime census")
            else:
                require(any(Path(row["path"]).is_relative_to(Path(root)) for root in invocation["deployment"]["artifact_runtime_roots"]),
                        "Artifact lifetime import escaped captured interpreter roots")
        else:
            require(row.get("origin") in (None, "built-in", "frozen") and
                    row.get("interpreter_sha256") == loaded["interpreter"]["sha256"], "Built-in/frozen import interpreter binding differs")
    package_sources = {r["relative"]: r["sha256"] for r in invocation["deployment"]["files"]}
    for name, checksum in loaded.get("buffered_package_modules", {}).items():
        if name == "__main__":
            relative = invocation["deployment"]["driver_relative"]
        elif name == "performance_driver_runtime":
            relative = next(r for r in package_sources if r.endswith("performance_driver_runtime.py"))
        else:
            relative = name.replace(".", "/") + ".py"
            if relative not in package_sources:
                relative = name.replace(".", "/") + "/__init__.py"
        require(package_sources.get(relative) == checksum, "Loaded module is outside captured exact source")
    require(loaded.get("buffered_package_modules", {}).get("__main__") ==
            package_sources[invocation["deployment"]["driver_relative"]], "Worker driver loaded code binding absent")
    if invocation.get("immutable_runtime"):
        execution.require_immutable_runtime(invocation["immutable_runtime"])
        receipt = native.get("preexec_kernel_observation")
        rules = execution.native_read_rules(invocation["native_namespace"], invocation["deployment"], native["pid"])
        require(isinstance(receipt, dict) and receipt.get("pid") == native["pid"] and
                receipt.get("invocation_buffer_sha256") == native["invocation_buffer_sha256"] and
                receipt.get("policy_sha256") == execution.sha(execution.encoded(invocation["native_namespace"])) and
                receipt.get("runtime_census_sha256") == invocation["immutable_runtime"]["runtime_census_sha256"] and
                receipt.get("elf_execution_policy_sha256") == execution.sha(execution.encoded(invocation["native_namespace"]["elf_execution_policy"])) and
                receipt.get("restrict_self_result") == 0 and receipt.get("enforcement_before_execve") is True and
                receipt.get("exact_self_maps_read_path") == "/proc/" + str(native["pid"]) + "/maps" and
                receipt.get("exact_rules_sha256") == execution.sha(execution.encoded(rules)) and
                receipt.get("exact_rule_count") == len(rules) and
                receipt.get("handled_access_fs") == execution.native_handled_access() and
                native.get("preexec_receipt_raw_sha256") == execution.sha(execution.encoded(receipt) + b"\n") and
                receipt.get("abi", 0) >= 3,
                "Native preexec runtime census receipt contradicts actual invocation/PID")
        prefilter = receipt.get("preexec_native_filter", {})
        require(prefilter.get("pid") == native["pid"] and prefilter.get("install_result") == 0 and
                prefilter.get("architecture") == os.uname().machine and prefilter.get("stage") == "before_execve" and
                type(prefilter.get("personality_query_result")) is int and prefilter["personality_query_result"] >= 0 and
                not prefilter["personality_query_result"] & 0x0400000 and
                prefilter.get("program_sha256") == execution.sha(execution.encoded(execution.native_filter_program(prefilter["architecture"], before_exec=True))),
                "Native preexec memory/write enforcement contradicts actual PID/program/personality")
        enforcement = loaded.get("native_process_enforcement", {})
        require(enforcement.get("pid") == native["pid"] and enforcement.get("install_result") == 0 and
                enforcement.get("architecture") == os.uname().machine and enforcement.get("stage") == "before_application" and
                type(enforcement.get("personality_query_result")) is int and enforcement["personality_query_result"] >= 0 and
                not enforcement["personality_query_result"] & 0x0400000 and
                enforcement.get("program_sha256") == execution.sha(execution.encoded(execution.native_filter_program(enforcement["architecture"]))) and
                enforcement.get("actual_threads_remain_in_worker_RSS") is True,
                "Native descendant/executable-memory enforcement contradicts actual worker")
    else:
        require(loaded.get("native_process_enforcement") == dict(scope="artifact_only", kernel_native_process_filter_installed=False),
                "Artifact worker cannot claim production native enforcement")
    for row in loaded["loaded_runtime_files"] + loaded["loaded_native_libraries"]:
        require(file_hash(row["path"]) == row["file_sha256"], "Loaded runtime bytes changed through worker completion")
        if invocation.get("immutable_runtime"):
            expected = {r["path"]: r["file_sha256"] for r in invocation["immutable_runtime"]["files"]}
            require(expected.get(row["path"]) == row["file_sha256"], "Loaded runtime file is outside the immutable deployment census")
    result = value["result"]
    if payload["kind"] != "point":
        require(set(result) == {"observations", "operation_count", "wall_seconds_per_operation"}, "Probe result schema differs")
        observations = result["observations"]
        associations = checkpoint_operation_associations(payload) if payload["kind"] == "checkpoint_digest" else None
        expected_rows = []
        for ref in payload["prepared"]:
            if payload["kind"] == "load_prepared_and_audit":
                expected_rows.append(dict(manifest_sha256=ref["manifest_sha256"]))
            else:
                checkpoint = next(r for r in payload["checkpoints"] if r["manifest_sha256"] == ref["manifest_sha256"])
                expected_rows.extend(dict(pin, registered_metadata_manifest_sha256=ref["manifest_sha256"],
                    sample_manifest_sha256=payload["checkpoint_sample"]["prepared"]["manifest_sha256"],
                    observed_sha256=pin["file_sha256"]) for pin in checkpoint["files"])
        if associations is not None:
            for row, association in zip(expected_rows, associations): row["native_operation_association"] = association
        require(type(result["operation_count"]) is int and len(observations) == result["operation_count"] == len(expected_rows) > 0,
                "Probe operation count disagrees with actual invocation/observations")
        for observed, expected in zip(observations, expected_rows):
            require(all(observed.get(k) == v for k, v in expected.items()), "Probe observation disagrees with invoked metadata/digest bytes")
            require(isinstance(observed["wall_seconds"], (int, float)) and math.isfinite(observed["wall_seconds"]) and
                    0 < observed["wall_seconds"] <= native["wall_seconds"], "Probe observation time is invalid")
        require(result["wall_seconds_per_operation"] == max(r["wall_seconds"] for r in observations),
                "Probe maximum time disagrees with actual observation rows")
    else:
        require(set(result) == {"shape", "function_call_counts", "function_observations", "stubbed", "report", "builder_memory"},
                "Point result schema differs")
        require(result["stubbed"] == STUBS and result["shape"]["registered_replicates"] == len(payload["prepared"]),
                "Point stub or registration evidence differs")
        counts = collections.Counter(row["function"] for row in result["function_observations"])
        require(dict(counts) == result["function_call_counts"] and set(counts) == set(FUNCTIONS) and all(counts.values()),
                "Point function counts disagree with actual code-object observation rows")
        expected_codes = expected_function_codes(invocation["deployment"], invocation["deployment_root"])
        require(all({k: row.get(k) for k in expected_codes[row["function"]]} == expected_codes[row["function"]]
                    for row in result["function_observations"]),
                "Point profiled a code object outside its immutable source buffer")
        loaded_functions = loaded.get("loaded_function_code_sha256", {})
        qualified = {("spacing_rerun.confirmation_lane." if name == "complement_contrast" else
                      "spacing_rerun.confirmation_dependency.") + name: row["code_sha256"] for name, row in expected_codes.items()}
        require(loaded_functions == qualified,
                "Loaded six production functions differ from compiled captured source")
        wrapper = checked_json(dict(result["report"], content_sha256=json.loads(Path(result["report"]["path"]).read_bytes())["sha256"]))
        require(all(wrapper.get(k) is v for k, v in MARKERS.items()), "Point report lost synthetic markers")
        report = check_seal(wrapper["production_report"])
        require(report["frozen_policy_sha256"] == payload["frozen_policy"]["content_sha256"] and
                report["preregistration_sha256"] == payload["preregistration"]["content_sha256"] and
                {b["manifest_sha256"] for b in report["bundles"]} == {r["manifest_sha256"] for r in payload["prepared"]} and
                len(report["bundles"]) == len(payload["prepared"]), "Point report registration/policy identity differs")
        require(counts["build_saved_dependency_report"] == 1 and counts["saved_bundle_diagnostics"] == len(payload["prepared"]) and
                counts["complete_contrast_report"] == len(report["contrasts"]) == counts["complement_contrast"] and
                counts["influence_scores"] == 2 * len(payload["prepared"]) * len(report["contrasts"]),
                "Point actual report work and function observations disagree")
        require(len(report["lane_descriptive_reporting"]["lane_only_complement_contrasts"]) == len(report["contrasts"]),
                "Point omitted actual lane work")
        validate_builder_memory(result["builder_memory"], payload, dict(result["report"], content_sha256=wrapper["sha256"]))
    return value


def checked_output_custody(invocation, registry, result_path, report_path=None):
    root = Path(invocation["output_root"])
    baseline = execution.checked_registry(invocation["output_registry_before_child"], root, invocation["output_identity"])
    rows = execution.checked_registry(registry, root, invocation["output_identity"])
    prior = {r["relative_path"]: r for r in baseline}
    current = {r["relative_path"]: r for r in rows}
    require(all(current.get(key) == row for key, row in prior.items()), "Child output registry changed/omitted parent inode custody")
    for path in (result_path, report_path):
        if path is None: continue
        relative = str(Path(path).relative_to(root))
        require(relative not in prior and current.get(relative, {}).get("kind") == "file",
                "Original child output registry omits report/result inode")
    anchor = execution.OutputAnchor(root, new=False)
    try:
        require(anchor.root_identity == invocation["output_identity"], "Actual output root differs from original invocation inode")
        anchor.receive_registry(invocation["output_registry_before_child"])
        anchor.receive_registry(registry)
    finally: anchor.close()
    return registry


def write_registered_json(anchor, path, value, registry_key="output_registry"):
    """Capture this file's original inode before serializing its custody record."""
    with anchor.open_exclusive(path, binary=True) as handle:
        value[registry_key] = anchor.registry()
        handle.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode() + b"\n")
        handle.flush(); os.fsync(handle.fileno())
    anchor.verify(complete=True)
    return value


def expected_function_codes(deployment, root):
    result = {}
    for row in deployment["files"]:
        if row["relative"] not in ("spacing_rerun/confirmation_dependency.py", "spacing_rerun/confirmation_lane.py"):
            continue
        code = compile(execution.base64.b64decode(row["data"]), str(Path(root) / row["relative"]), "exec", dont_inherit=True)
        for item in code.co_consts:
            if hasattr(item, "co_name") and item.co_name in FUNCTIONS:
                result[item.co_name] = execution.code_identity(item)
    require(set(result) == set(FUNCTIONS), "Captured full package lacks a required actual function")
    return result


def deploy_buffers(deployment, anchor):
    root = anchor.path / ("source-deployment-" + deployment["sha256"][:16])
    if not root.exists():
        anchor.mkdir(root)
        for row in deployment["files"]:
            anchor.write(root / row["relative"], execution.base64.b64decode(row["data"]))
    for row in deployment["files"]:
        require(file_hash(root / row["relative"]) == row["sha256"], "Retained deployment changed")
    return root


def verify_deployment_commit(deployment, code):
    """Close the preflight-to-capture gap using the actual captured source bytes."""
    require(local_git("rev-parse", "HEAD").decode().strip() == code["code_commit"],
            "Measured HEAD changed before captured deployment")
    for row in deployment["files"]:
        path = Path(row["path"])
        if path.is_relative_to(REPOSITORY):
            blob = local_git("show", code["code_commit"] + ":" + str(path.relative_to(REPOSITORY)))
            require(execution.sha(blob) == row["sha256"],
                    "Captured source/resource differs from actual measured commit: " + row["path"])
    require(next(r["sha256"] for r in deployment["files"] if r["relative"] == deployment["driver_relative"]) ==
            code["measurement_driver"]["file_sha256"], "Captured driver differs from preflight measurement source")
    return deployment


def preserve_worker_custody(invocation, producer, native, anchor, root):
    """Retain raw transport and validated originals before any exit rejection."""
    transport = dict(schema="p4-worker-original-custody-transport-v1", native_observation=native,
                     producer_raw_base64=execution.base64.b64encode(producer).decode("ascii"),
                     producer_raw_sha256=execution.sha(producer), issues=[], original_child_custody_complete=False)
    anchor.worker_custody_transports.append(transport)
    journal_raw = execution.base64.b64decode(native.get("custody_journal_raw_base64", ""), validate=True)
    transport["custody_journal_raw_base64"] = execution.base64.b64encode(journal_raw).decode("ascii")
    transport["custody_journal_raw_sha256"] = execution.sha(journal_raw)
    require(transport["custody_journal_raw_sha256"] == native.get("custody_journal_raw_sha256"), "Native original journal bytes/hash differ")
    for suffix, data in ((".producer.bin", producer), (".custody-journal.bin", journal_raw)):
        path = root.with_suffix(suffix)
        try:
            anchor.write(path, data)
            transport[suffix + "_ref"] = raw_ref(path)
        except (ValueError, OSError) as error:
            transport["issues"].append("Raw transport remains inline: " + type(error).__name__ + ": " + str(error))
    journal = execution.read_custody_journal(journal_raw, invocation)
    transport["journal_prefix"] = journal
    if journal["original_registry"] is not None:
        anchor.receive_registry(journal["original_registry"], verify_current=False)
    value = None
    try:
        value = json.loads(producer)
        require(isinstance(value, dict) and value.get("schema") in {"p4-buffered-worker-result-v5", "p4-buffered-worker-failure-v1"} and
                all(value.get(k) is v for k, v in MARKERS.items()) and value.get("nonce") == invocation["nonce"] and
                value.get("invocation_sha256") == invocation["payload_sha256"] and
                value.get("deployment_sha256") == invocation["deployment"]["sha256"] and
                value.get("parent_output_registry_sha256") == execution.sha(execution.encoded(invocation["output_registry_before_child"])),
                "Producer output custody belongs to another invocation/type")
        anchor.receive_registry(value["output_registry"], verify_current=False)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
        transport["issues"].append("Producer custody unverified: " + type(error).__name__ + ": " + str(error))
    try:
        anchor.verify(complete=True)
    except (ValueError, OSError) as error:
        transport["issues"].append("Original current custody differs: " + type(error).__name__ + ": " + str(error))
    return value, transport


def execute_worker(payload, output):
    """Captured code, independent producer channel, and exact through-exit wait4."""
    global ACTIVE_OUTPUT
    root = Path(output)
    local_anchor = ACTIVE_OUTPUT is None
    anchor = ACTIVE_OUTPUT or execution.OutputAnchor(root.parent, new=False)
    previous = ACTIVE_OUTPUT
    ACTIVE_OUTPUT = anchor
    input_path, result_path, log_path = root.with_suffix(".input.json"), root.with_suffix(".result.json"), root.with_suffix(".log")
    try:
        require(Path(__file__).read_bytes() == CAPTURED_DRIVER_SOURCE, "Driver source changed after parent import")
        deployment = ACTIVE_DEPLOYMENT or execution.capture_sources(Path(__file__).resolve(), EXPERIMENT / "spacing_rerun")
        execution.verify_sources(deployment)
        deployment_root = deploy_buffers(deployment, anchor)
        worker_payload = dict(payload, worker_result_path=str(result_path))
        write_json(input_path, worker_payload)
        invocation = dict(nonce=uuid.uuid4().hex, payload_sha256=file_hash(input_path),
            input_path=str(input_path), payload=worker_payload, deployment=deployment, deployment_root=str(deployment_root),
            output_root=str(anchor.path), output_fd=anchor.fd, output_identity=anchor.root_identity,
            immutable_runtime=payload.get("immutable_runtime"))
        invocation["verified_owner_adapters"] = [verify_exact_adapter(payload["builder_memory_probe"],
            required_function="probe_analysis_live_memory")] if payload["kind"] == "point" else []
        if payload.get("immutable_runtime"):
            data_files = {row["path"] for row in deployment["files"]}
            def collect(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key != "immutable_runtime": collect(item)
                elif isinstance(value, list):
                    for item in value: collect(item)
                elif isinstance(value, str) and value.startswith("/"):
                    candidate = Path(value)
                    if candidate.is_file() and not candidate.is_relative_to(anchor.path): data_files.add(value)
            collect(worker_payload)
            invocation["native_namespace"] = execution.require_native_namespace(payload["immutable_runtime"],
                anchor, payload["production_run_roots"], exact_data_files=data_files)
        producer, native = execution.launch_native(invocation, anchor, log_path)
        value, custody_transport = preserve_worker_custody(invocation, producer, native, anchor, root)
        write_json(root.with_suffix(".native.json"), native)
        require(native["exit_code"] == 0, "CPU worker failed; retain and inspect " + str(log_path))
        require(producer, "Worker did not emit its independent native producer channel")
        require(value is not None and not custody_transport["issues"], "Worker original custody transport is invalid: " + str(custody_transport["issues"]))
        require(value.get("schema") == "p4-buffered-worker-result-v5" and value.get("nonce") == invocation["nonce"] and
                value.get("invocation_sha256") == invocation["payload_sha256"] and
                value.get("deployment_sha256") == deployment["sha256"], "Producer output custody belongs to another invocation")
        anchor.receive_registry(value["output_registry"])
        require(json.loads(result_path.read_bytes()) == value and log_path.read_bytes().rstrip() == producer.rstrip(),
                "Worker result/stdout contradict the native producer channel")
        require(file_hash(input_path) == invocation["payload_sha256"], "Worker input changed through execution")
        execution.verify_sources(deployment)
        deploy_buffers(deployment, anchor)
        anchor.verify(complete=True)
        validate_worker_result(worker_payload, value, invocation, native)
        peak = native["parent_peak_rss_bytes"] + native["through_exit_child_peak_rss_bytes"]
        measured = dict(MARKERS, schema="p4-analysis-performance-worker-observation-v5",
            wall_seconds=native["wall_seconds"], peak_rss_bytes=peak,
            parent_peak_rss_bytes=native["parent_peak_rss_bytes"], child_peak_rss_bytes=native["through_exit_child_peak_rss_bytes"],
            descendant_peak_rss_bytes=0, memory_method="per_child_wait4_through_exit_plus_distinct_parent_high_water",
            native_process_observation=native, producer_channel_sha256=execution.sha(producer),
            invocation_nonce=invocation["nonce"], payload_sha256=invocation["payload_sha256"],
            deployment_sha256=deployment["sha256"], worker=value)
        observation_path = root.with_suffix(".observation.json")
        write_registered_json(anchor, observation_path, measured)
        return measured, [raw_ref(p) for p in (log_path, result_path, observation_path, input_path,
                          root.with_suffix(".native.json"), root.with_suffix(".invocation.json"),
                          root.with_suffix(".producer.bin"), root.with_suffix(".custody-journal.bin"))]
    except BaseException as error:
        anchor.retain_failure(error)
        raise
    finally:
        ACTIVE_OUTPUT = previous
        if local_anchor:
            anchor.close()


def check_fixture_preregistration(prereg, prepared):
    check_seal(prereg)
    require(all(prereg.get(k) is v for k, v in MARKERS.items()) and
            prereg.get("n") == len(prepared) and prereg.get("replicate_manifest_sha256") ==
            [ref["manifest_sha256"] for ref in prepared],
            "Only explicitly labelled exact first-n fixture preregistration is allowed")
    require(type(prereg.get("E")) is int and prereg["E"] > 0 and
            type(prereg.get("primary_delay")) is int and prereg["primary_delay"] >= 0,
            "Fixture preregistration needs actual fixed E and primary endpoint")


def saved_string_envelope(spec, model, numerical):
    """Require a retained metadata-derived decoding bound, never real outcomes."""
    ref = spec.get("saved_string_envelope")
    require(ref is not None, "UNRESOLVED_SAVED_STRING_ENVELOPE: tokenizer metadata decoding/serialization bound is required")
    envelope = checked_json(ref)
    require(envelope.get("schema") == "p4-metadata-derived-saved-string-envelope-v2" and
            envelope.get("tokenizer_sha256") == model["tokenizer_sha256"] and
            envelope.get("real_predictions_or_outcomes_used") is False and
            envelope.get("tokenizer_or_model_invoked") is False and envelope.get("metadata_evidence_refs") and
            envelope.get("method") == "maximum_vocabulary_piece_utf8_bytes_plus_native_decoder_bound",
            "Saved prediction envelope is not bound to the actual tokenizer metadata/decoder source")
    for key in ("maximum_prediction_utf8_bytes", "token_id_upper_bound", "max_answer_tokens"):
        require(type(envelope.get(key)) is int and envelope[key] > 0, "Saved prediction bound is not a positive integer")
    require(envelope["max_answer_tokens"] == numerical["max_answer_tokens"] and
            envelope["maximum_prediction_utf8_bytes"] >= envelope["max_answer_tokens"] * envelope.get("maximum_piece_utf8_bytes", 0) and
            type(envelope.get("maximum_piece_utf8_bytes")) is int and envelope["maximum_piece_utf8_bytes"] > 0,
            "Saved prediction envelope does not cover the native maximum token count")
    require(envelope.get("saved_json_serializer") == dict(sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False, trailing_newline=True) and
            envelope.get("exposure_json_serializer") == dict(sort_keys=True, ensure_ascii=False, allow_nan=False, trailing_newline=True) and
            envelope.get("maximum_json_prediction_string_bytes") == 6 * envelope["maximum_prediction_utf8_bytes"] + 2 and
            envelope.get("finite_float_maximum_json_bytes") == 24 and
            envelope.get("maximum_token_id_json_bytes") == len(str(envelope["token_id_upper_bound"])) and
            envelope.get("decoded_prediction_maximum_codepoints") == envelope["maximum_prediction_utf8_bytes"] and
            envelope.get("decoded_prediction_surrogate_kind") == "separate_four_byte_unicode_string_preserving_native_prediction_and_scoring" and
            envelope.get("compiler") == execution.compiler_identity(),
            "Saved envelope omits exact serializer, scalar/token widths or decoded graph bounds")
    binding = spec.get("saved_string_envelope_review_binding")
    require(isinstance(binding, dict), "UNRESOLVED_SAVED_STRING_ENVELOPE: actual metadata bound review is required")
    review = checked_json(binding["report"])
    verify_adapter_review(binding, review)
    require(review.get("implementation_verdict") == "approved" and
            review.get("approved_saved_string_envelope_sha256") == envelope["sha256"],
            "Saved string bound lacks exact completed independent metadata/source review")
    for evidence in envelope["metadata_evidence_refs"]:
        require(raw_ref(evidence["path"]) == evidence, "Saved string bound metadata bytes differ")
    return envelope


def reconcile_exported_values(benchmark, telemetry):
    """Replay exact exported values against the retained execution observations."""
    def observation(row):
        refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".observation.json")]
        require(len(refs) == 1, "Export lacks one exact point/probe execution observation")
        value = checked_json(refs[0], content_seal=False)
        native = value["native_process_observation"]
        require(value["peak_rss_bytes"] == native["parent_peak_rss_bytes"] + native["through_exit_child_peak_rss_bytes"] and
                value["wall_seconds"] == native["wall_seconds"] and native["exit_code"] == 0,
                "Export observation contradicts its native per-child process accounting")
        require(row["peak_rss_bytes"] == value["peak_rss_bytes"], "Exported memory contradicts actual observation")
        invocation_refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".invocation.json")]
        require(len(invocation_refs) == 1, "Export lacks exact invocation source/payload evidence")
        invocation = checked_json(invocation_refs[0], content_seal=False)
        custody = execution.OutputAnchor(invocation["output_root"], new=False)
        try:
            custody.receive_registry(invocation["output_registry_before_child"])
            custody.receive_registry(value["worker"]["output_registry"])
            custody.receive_registry(value["output_registry"])
            # A shared output may have later points; their original identities
            # are checked by the live parent and each later observed registry.
            custody.verify(complete=True)
        finally: custody.close()
        require(execution.sha(execution.encoded(invocation)) == native["invocation_buffer_sha256"], "Export invocation differs from actual native input")
        native_refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".native.json")]
        input_refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".input.json")]
        require(len(native_refs) == len(input_refs) == 1 and
                checked_json(native_refs[0], content_seal=False) == native and
                checked_json(input_refs[0], content_seal=False) == invocation["payload"] and
                input_refs[0]["file_sha256"] == invocation["payload_sha256"],
                "Export input/native files contradict captured invocation/observation")
        log = next(r for r in row["raw_log_refs"] if r["path"].endswith(".log"))
        result = next(r for r in row["raw_log_refs"] if r["path"].endswith(".result.json"))
        producer_refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".producer.bin")]
        journal_refs = [r for r in row["raw_log_refs"] if r["path"].endswith(".custody-journal.bin")]
        require(len(producer_refs) == len(journal_refs) == 1 and
                Path(producer_refs[0]["path"]).read_bytes() == Path(log["path"]).read_bytes() and
                Path(journal_refs[0]["path"]).read_bytes() == execution.base64.b64decode(native["custody_journal_raw_base64"], validate=True),
                "Export raw producer/journal channels contradict original native observations")
        for ref in row["raw_log_refs"]:
            require(raw_ref(ref["path"]) == ref, "Exported native evidence changed")
        require(json.loads(Path(log["path"]).read_bytes()) == json.loads(Path(result["path"]).read_bytes()) == value["worker"],
                "Exported stdout/result/worker observation contradict")
        require(value["producer_channel_sha256"] == file_hash(log["path"]) and
                value["invocation_nonce"] == invocation["nonce"] and
                value["payload_sha256"] == invocation["payload_sha256"] and
                value["deployment_sha256"] == invocation["deployment"]["sha256"],
                "Export worker producer/invocation/code binding contradicts original bytes")
        validate_worker_result(invocation["payload"], value["worker"], invocation, native)
        return value
    for point in benchmark["points"]:
        observed = observation(point)
        result = observed["worker"]["result"]
        require(point["wall_seconds"] == observed["wall_seconds"] and point["function_call_counts"] == result["function_call_counts"] and
                point["memory_evidence"]["shape"] == result["shape"] and
                point["memory_evidence"]["builder_memory"] == result["builder_memory"],
                "Exported point values differ from actual function/report/live-memory observations")
    for probe in benchmark["separate_real_probes"].values():
        observed = observation(probe); result = observed["worker"]["result"]
        require(probe["operation_count"] == len(result["observations"]) == result["operation_count"] and
                probe["wall_seconds"] == max(r["wall_seconds"] for r in result["observations"]) and
                probe["total_worker_wall_seconds"] == observed["wall_seconds"], "Exported probe summary contradicts observations")
    largest = benchmark["points"][-1]
    record = telemetry["records"][0]
    require(record["wall_seconds"] == largest["wall_seconds"] and record["peak_rss_bytes"] == largest["peak_rss_bytes"] and
            record["memory_evidence"] == largest["memory_evidence"] and record["raw_log_refs"] == largest["raw_log_refs"],
            "Exported telemetry contradicts the retained largest-n point")


def preflight(spec, output):
    require(spec.get("schema") == INPUT_SCHEMA and all(spec.get(k) is v for k, v in MARKERS.items()),
            "Only the explicitly nonproduction performance input schema is allowed")
    require(type(spec.get("random_seed")) is int, "Synthetic random seed must be an integer")
    require(type(spec.get("cpu_count")) is int and spec["cpu_count"] > 0 and
            os.environ.get("SLURM_JOB_ID", "").isdigit() and
            os.environ.get("SLURM_CPUS_PER_TASK", "").isdigit() and
            spec["cpu_count"] == int(os.environ["SLURM_CPUS_PER_TASK"]),
            "Remote owner must execute inside a truthful Slurm CPU allocation")
    require(spec.get("margin_fraction", 0) > 0 and math.isfinite(spec["margin_fraction"]) and
            type(spec.get("margin_bytes")) is int and spec["margin_bytes"] >= 0,
            "Declare a positive conservative memory margin and nonnegative byte margin")
    output_path = canonical_path(output)
    roots = [canonical_path(root) for root in spec["production_run_roots"]]
    require(roots and len(set(roots)) == len(roots) and
            all(not output_path.is_relative_to(root) and not root.is_relative_to(output_path) for root in roots),
            "Synthetic output must be outside every declared production run root")
    prepared = spec["prepared"]
    require(len(prepared) == 32 and len({ref["manifest_sha256"] for ref in prepared}) == 32,
            "Strict R3 requires 32 distinct actual prepared metadata bindings")
    for ref in prepared:
        path = canonical_path(ref["path"])
        require(any(path.is_relative_to(root) for root in roots),
                "Every preparation's production root must be disclosed")
        require((path / "manifest.json").is_file() and file_hash(path / "manifest.json") == ref["manifest_file_sha256"],
                "Prepared metadata is absent or changed")
    checkpoint_inputs(spec["checkpoints"], prepared)
    require(all(any(canonical_path(pin["path"]).is_relative_to(root) for root in roots)
                for row in spec["checkpoints"] for pin in row["files"]),
            "Every checkpoint's production root must be disclosed")
    require(set(spec["contexts"]) == {str(n) for n in POINTS},
            "Strict R3 needs separately genuine approved frozen policies for n=2,8,32")
    model, numerical, fixed_recipe = None, None, None
    for n in POINTS:
        context = spec["contexts"][str(n)]
        frozen = checked_json(context["frozen_policy"])
        prereg = checked_json(context["preregistration"])
        check_fixture_preregistration(prereg, prepared[:n])
        manifest, schedule, examples = load_real_prepared(prepared[0])
        require(manifest["config"].get("acquisition_exposures") == [prereg["E"]] and
                manifest["config"].get("primary_delay") == prereg["primary_delay"],
                "Synthetic preregistration E/primary delay must match actual frozen preparation")
        from spacing_rerun.confirmation_dependency import verify_frozen_policy
        actual = verify_frozen_policy(context["frozen_policy"]["path"], context["frozen_policy"]["content_sha256"],
                                      manifest["confirmation_audit"]["source_chunks"], manifest["confirmation_audit"]["source_catalog"])
        require(actual == frozen and {row["manifest_sha256"] for row in frozen["policy"]["canonical_bindings"]} ==
                {ref["manifest_sha256"] for ref in prepared[:n]},
                "Per-point genuine policy approval binds another registration; do not subset or restamp it")
        model = {k: manifest[k] for k in ("model", "model_revision", "tokenizer_sha256")}
        numerical = {k: manifest["config"][k] for k in SHAPE_KEYS}
        fixed_recipe = (prereg["E"], prereg["primary_delay"])
        del manifest, schedule, examples
    for ref in prepared:
        manifest, schedule, examples = load_real_prepared(ref)
        require({k: manifest[k] for k in model} == model and
                {k: manifest["config"][k] for k in SHAPE_KEYS} == numerical and
                manifest["config"].get("acquisition_exposures") == [fixed_recipe[0]] and
                manifest["config"].get("primary_delay") == fixed_recipe[1],
                "Actual preparation model or numerical shapes differ")
        del manifest, schedule, examples
    for ref in spec["additional_parse_refs"]:
        checked_json(ref, content_seal=False)
    require(spec.get("builder_memory_probe"),
            "UNRESOLVED_FINAL_BUILDER_MEMORY: exact committed final-builder live-memory probe is required")
    require(spec.get("runtime_deployment"),
            "UNRESOLVED_RUNTIME_CUSTODY: final verified runtime deployment contract is required")
    runtime = execution.require_immutable_runtime(spec["runtime_deployment"])
    envelope = saved_string_envelope(spec, model, numerical)
    code = committed_code(spec)
    sample_binding = checkpoint_sample_binding(spec, model, roots)
    output, _ = isolate_output(str(output_path), spec["production_run_roots"])
    return output, code, model, numerical, sample_binding, runtime, envelope


def _run(spec_path, output):
    global ACTIVE_OUTPUT, ACTIVE_DEPLOYMENT
    spec_path = canonical_path(spec_path)
    input_identity = raw_ref(spec_path)
    spec = check_seal(json.loads(spec_path.read_bytes()))
    anchor, code, model, numerical, checkpoint_sample, runtime, envelope = preflight(spec, output)
    ACTIVE_OUTPUT = anchor.activate()
    output = anchor.path
    ACTIVE_DEPLOYMENT = execution.capture_sources(Path(__file__).resolve(), EXPERIMENT / "spacing_rerun",
        extra=[spec["builder_memory_probe"]["path"], spec["checkpoint_sample"]["producer_association_adapter"]["path"]])
    execution.verify_sources(ACTIVE_DEPLOYMENT)
    verify_deployment_commit(ACTIVE_DEPLOYMENT, code)
    write_json(output / "retained-input.json", spec)
    points = []
    for n in POINTS:
        context = spec["contexts"][str(n)]
        policy = checked_json(context["frozen_policy"])["policy"]
        prereg = checked_json(context["preregistration"])
        rng = random.Random(spec["random_seed"] + n)
        point_root = output / f"n{n:02}"
        refs = [generate_bundle(point_root / f"replicate-{index:02}", ref, policy, prereg, rng, envelope)
                for index, ref in enumerate(spec["prepared"][:n])]
        bundles_path = point_root / "compact-bundles.json"
        write_json(bundles_path, refs)
        # Release policy/preparation graphs before the measured worker starts.
        del policy, prereg, refs
        measured, logs = execute_worker(dict(kind="point", prepared=spec["prepared"][:n],
            frozen_policy=context["frozen_policy"], preregistration=context["preregistration"],
            bundles_path=str(bundles_path), report_path=str(point_root / "synthetic-report.json"),
            additional_parse_refs=spec["additional_parse_refs"], immutable_runtime=runtime,
            production_run_roots=spec["production_run_roots"],
            builder_memory_probe=spec["builder_memory_probe"], builder_shared_json_refs=spec["builder_shared_json_refs"]), point_root / "analysis")
        proof = dict(kind="measured_matched_size_envelope", method=measured["memory_method"],
                     shape=measured["worker"]["result"]["shape"], peak_rss_bytes=measured["peak_rss_bytes"],
                     builder_memory=measured["worker"]["result"]["builder_memory"],
                     margin_fraction=spec["margin_fraction"], margin_bytes=spec["margin_bytes"],
                     measured_cpu_code=code, raw_log_refs=logs)
        points.append(dict(MARKERS, n=n, prepared=spec["prepared"][:n],
            wall_seconds=measured["wall_seconds"], peak_rss_bytes=measured["peak_rss_bytes"],
            memory_evidence=proof, raw_log_refs=logs, policy_context=context,
            function_call_counts=measured["worker"]["result"]["function_call_counts"]))
        print(json.dumps(dict(n=n, wall_seconds=measured["wall_seconds"], peak_rss_bytes=measured["peak_rss_bytes"])), flush=True)
    probes = {}
    for kind in ("load_prepared_and_audit", "checkpoint_digest"):
        measured, logs = execute_worker(dict(kind=kind, prepared=spec["prepared"], checkpoints=spec["checkpoints"],
                                            checkpoint_sample=checkpoint_sample, immutable_runtime=runtime,
                                            production_run_roots=spec["production_run_roots"]), output / kind)
        result = measured["worker"]["result"]
        expected = 32 if kind == "load_prepared_and_audit" else 32 * 6
        require(result["operation_count"] == expected, "Actual separate probe call count differs")
        probes[kind] = dict(MARKERS, wall_seconds=result["wall_seconds_per_operation"],
            wall_seconds_basis="maximum_observed_single_operation", total_worker_wall_seconds=measured["wall_seconds"],
            peak_rss_bytes=measured["peak_rss_bytes"], operation_count=expected, raw_log_refs=logs)
        if kind == "checkpoint_digest":
            probes[kind]["checkpoint_sample"] = checkpoint_sample
    # Include probe process peaks in each actual point's conservative envelope.
    for point in points:
        point["memory_evidence"]["peak_rss_bytes"] = max(point["peak_rss_bytes"], *(p["peak_rss_bytes"] for p in probes.values()))
        point["memory_evidence"]["raw_log_refs"] = list(point["raw_log_refs"]) + [ref for p in probes.values() for ref in p["raw_log_refs"]]
    benchmark = sealed(dict(MARKERS, schema=BENCHMARK_SCHEMA, kind="analysis_performance_sample",
        predictions="synthetic_seeded_random", random_seed=spec["random_seed"],
        point_seed_derivation="random_seed + point_n",
        algorithm_sha256=spec["algorithm_sha256"], reporting_specification_sha256=spec["reporting_specification_sha256"],
        package_scope=code["package_scope"], outside_every_run_root=True, real_metadata=True,
        production_run_roots=spec["production_run_roots"], synthetic_root=str(output),
        stubbed=STUBS, production_functions_exercised=list(FUNCTIONS), points=points,
        separate_real_probes=probes, checkpoint_digests_per_replicate=6, input_reference=input_identity))
    require(raw_ref(spec_path) == input_identity, "Driver input changed during measurement")
    benchmark_path = output / "analysis-benchmark.json"
    write_json(benchmark_path, benchmark)
    point = points[-1]
    membership = checked_json(spec["contexts"]["32"]["frozen_policy"])["policy"]["membership"]
    shape = point["memory_evidence"]["shape"]
    record = dict(MARKERS, **code, component="analysis", backend=spec["hardware"]["backend"],
        gpu=spec["hardware"]["gpu"], software=spec["hardware"]["software"],
        measured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method="actual_observed_wall_and_peak_rss", prepared=spec["prepared"][0],
        manifest_sha256=spec["prepared"][0]["manifest_sha256"], model_identity=model, numerical_config=numerical,
        synthetic_predictions=True, workload=dict(
            prediction_rows_times_influence_cases=shape["prediction_rows"] * (1 + 80 + len(membership["blocks"])),
            source_units=shape["source_units"]), wall_seconds=point["wall_seconds"],
        peak_rss_bytes=point["peak_rss_bytes"], cpu_count=spec["cpu_count"], peak_gpu_bytes=0,
        memory_evidence=point["memory_evidence"], raw_log_refs=point["raw_log_refs"],
        analysis_benchmark=dict(raw_ref(benchmark_path), content_sha256=benchmark["sha256"]))
    telemetry = dict(MARKERS, records=[record])
    reconcile_exported_values(benchmark, telemetry)
    execution.verify_sources(ACTIVE_DEPLOYMENT)
    ACTIVE_OUTPUT.verify(complete=True)
    write_json(output / "analysis-telemetry.json", telemetry)
    custody_path = output / "analysis-output-custody.json"
    write_registered_json(ACTIVE_OUTPUT, custody_path, dict(MARKERS,
        schema="p4-analysis-final-output-custody-v1", benchmark=raw_ref(benchmark_path), telemetry=raw_ref(output / "analysis-telemetry.json")))
    ACTIVE_OUTPUT.verify(complete=True)
    print(json.dumps(dict(benchmark=raw_ref(benchmark_path), telemetry=raw_ref(output / "analysis-telemetry.json"), output_custody=raw_ref(custody_path))), flush=True)


def run(spec_path, output):
    global ACTIVE_OUTPUT, ACTIVE_DEPLOYMENT
    try:
        _run(spec_path, output)
    except BaseException as error:
        if ACTIVE_OUTPUT:
            ACTIVE_OUTPUT.retain_failure(error)
        raise
    finally:
        if ACTIVE_OUTPUT:
            ACTIVE_OUTPUT.close()
        ACTIVE_OUTPUT = ACTIVE_DEPLOYMENT = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Sealed immutable driver input JSON")
    parser.add_argument("--output", help="New absolute synthetic directory outside every production root")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        global ACTIVE_OUTPUT
        invocation = globals().get("BUFFERED_INVOCATION")
        require(invocation is not None, "Workers require the captured native supervisor invocation")
        ACTIVE_OUTPUT = execution.OutputAnchor(invocation["output_root"], new=False,
            inherited_fd=invocation["output_fd"], expected=invocation["output_identity"])
        ACTIVE_OUTPUT.receive_registry(invocation["output_registry_before_child"])
        try:
            ACTIVE_OUTPUT.custody_journal = execution.CustodyJournal(invocation)
            raw = Path(args.worker).read_bytes()
            require(execution.sha(raw) == invocation["payload_sha256"], "Captured worker payload byte identity differs")
            payload = json.loads(raw)
            require(payload == invocation["payload"], "Captured worker payload value differs")
            started = time.monotonic()
            value = worker_point(payload) if payload["kind"] == "point" else worker_probe(payload)
            ACTIVE_OUTPUT.verify(complete=True)
            # Workers have no grandchildren. Fail if future code changes introduce
            # them, since a single RUSAGE_CHILDREN maximum is not their peak sum.
            child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            require(child_usage.ru_maxrss == 0, "Worker spawned descendants; memory method needs revision")
            runtime = observed_runtime()
            loaded = execution.loaded_identity(invocation["deployment"])
            result = dict(MARKERS, schema="p4-buffered-worker-result-v5", result=value,
                          invocation_sha256=invocation["payload_sha256"], nonce=invocation["nonce"],
                          deployment_sha256=invocation["deployment"]["sha256"],
                          child_wall_seconds=time.monotonic() - started,
                          child_preexit_rss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF)), descendant_peak_rss_bytes=0,
                          observed_runtime=runtime, loaded_code_identity=loaded,
                          parent_output_registry_sha256=execution.sha(execution.encoded(invocation["output_registry_before_child"])))
            write_registered_json(ACTIVE_OUTPUT, payload["worker_result_path"], result)
            ACTIVE_OUTPUT.custody_journal.append("success", ACTIVE_OUTPUT.original_registry())
            data = execution.encoded(result) + b"\n"
            fd = invocation["producer_channel_fd"]
            with os.fdopen(os.dup(fd), "wb") as channel:
                channel.write(data); channel.flush(); os.fsync(channel.fileno())
            sys.stdout.buffer.write(data); sys.stdout.buffer.flush()
        except BaseException as error:
            registry = ACTIVE_OUTPUT.original_registry()
            failure = dict(MARKERS, schema="p4-buffered-worker-failure-v1", nonce=invocation["nonce"],
                invocation_sha256=invocation["payload_sha256"], deployment_sha256=invocation["deployment"]["sha256"],
                parent_output_registry_sha256=execution.sha(execution.encoded(invocation["output_registry_before_child"])),
                output_registry=registry, error_type=type(error).__name__, error=str(error),
                original_child_custody_complete=False)
            try:
                if ACTIVE_OUTPUT.custody_journal is not None:
                    ACTIVE_OUTPUT.custody_journal.append("failure", registry)
            except BaseException as transport_error:
                sys.stderr.write("Failure custody journal incomplete: " + repr(transport_error) + "\n")
            try:
                data = execution.encoded(failure) + b"\n"
                with os.fdopen(os.dup(invocation["producer_channel_fd"]), "wb") as channel:
                    channel.write(data); channel.flush(); os.fsync(channel.fileno())
            except BaseException as transport_error:
                sys.stderr.write("Failure producer custody incomplete: " + repr(transport_error) + "\n")
            raise
        finally:
            ACTIVE_OUTPUT.close()
    else:
        require(args.input and args.output, "--input and --output are required")
        run(args.input, args.output)


if __name__ == "__main__":
    main()
