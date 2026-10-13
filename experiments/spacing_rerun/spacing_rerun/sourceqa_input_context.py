"""Explicit caller input custody for Source-QA validation.

Raw/canonical pins bind supplied bytes. They do not authenticate a custodian or
make a candidate's proposed pins trusted. The operator supplies these references
separately from candidate policy/coverage; original provenance remains external.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .common import digest, require


def sourceqa_validator_keywords(*, expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Keep absent-context calls compatible; supplied objects retain identity."""
    return {key: value for key, value in
            (("expected_evaluator_metadata", expected_evaluator_metadata),
             ("original_trace_input_files_by_proof", original_trace_input_files_by_proof)) if value is not None}


def checked_input_descriptor(reference):
    require(isinstance(reference, dict) and set(reference) == {"path", "file_sha256", "content_sha256"},
            "Source-QA input requires explicit path/raw/canonical pins")
    require(isinstance(reference["path"], str) and Path(reference["path"]).is_absolute() and
            all(isinstance(reference[k], str) and re.fullmatch("[0-9a-f]{64}", reference[k])
                for k in ("file_sha256", "content_sha256")), "Malformed Source-QA input descriptor")
    path = Path(reference["path"])
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise ValueError("Source-QA input is absent, a symlink, or unreadable") from error
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), "Source-QA input is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as opened:
            raw = opened.read()
    finally:
        os.close(descriptor)
    require(hashlib.sha256(raw).hexdigest() == reference["file_sha256"], "Source-QA input raw bytes differ")
    value = json.loads(raw)
    require(isinstance(value, dict) and value.get("sha256") == reference["content_sha256"] and
            digest({k: v for k, v in value.items() if k != "sha256"}) == reference["content_sha256"],
            "Source-QA input canonical seal differs")
    return value


def load_sourceqa_inputs(descriptors=None, *, include_references=False):
    """Return the actual loaded metadata and original nested mapping unchanged."""
    if descriptors is None:
        inputs = dict(expected_evaluator_metadata=None, original_trace_input_files_by_proof=None)
        return (inputs, []) if include_references else inputs
    require(isinstance(descriptors, dict) and set(descriptors) ==
            {"expected_evaluator_metadata", "original_trace_custody"}, "Source-QA input routing fields differ")
    metadata_ref = descriptors["expected_evaluator_metadata"]
    custody_ref = descriptors["original_trace_custody"]
    metadata = checked_input_descriptor(metadata_ref) if metadata_ref is not None else None
    if metadata is not None:
        require(metadata.get("schema") == "spacing-sourceqa-current-evaluator-input-metadata-v1" and
                set(metadata) == {"schema", "sha256", "scope", "configuration_sha256", "catalog_binding",
                                  "snapshot_binding", "provenance_binding", "required_surface_correlate_ids_by_record"},
                "Expected evaluator input must contain existing metadata fields only")
    custody = checked_input_descriptor(custody_ref) if custody_ref is not None else None
    if custody is not None:
        require(custody.get("schema") == "spacing-separate-original-trace-caller-custody-v1" and
                custody.get("original_source_proof_trace_status_contract_helper_restamped") is False and
                custody.get("semantic_source_property_approval_created") is False and
                custody.get("source_commands_executed") is False,
                "Original trace custody is missing or claims a restamp/approval/execution")
        mapping = custody.get("original_trace_input_files_by_proof")
        require(isinstance(mapping, dict) and mapping and
                all(isinstance(k, str) and re.fullmatch("[0-9a-f]{64}", k) and isinstance(v, dict)
                    for k, v in mapping.items()) and custody.get("binding_proof_sha256") in mapping,
                "Original trace custody proof keys differ")
    else:
        mapping = None
    inputs = dict(expected_evaluator_metadata=metadata, original_trace_input_files_by_proof=mapping)
    return (inputs, sourceqa_input_references(descriptors, inputs, custody=custody)) if include_references else inputs


def sourceqa_input_references(descriptors, inputs, *, custody):
    """Disclose actual separately supplied input files for read scope/accounting."""
    if descriptors is None:
        return []
    references = [r for r in descriptors.values() if r is not None]
    metadata = inputs["expected_evaluator_metadata"]
    if metadata is not None:
        references.extend(metadata[k] for k in ("catalog_binding", "snapshot_binding", "provenance_binding"))
    custody_ref = descriptors["original_trace_custody"]
    if custody_ref is not None:
        references.append(dict(path=custody["binding_proof_path"], file_sha256=custody["binding_proof_file_sha256"],
                               content_sha256=custody["binding_proof_sha256"]))
        response = custody["original_response_binding"]
        references.append(dict(path=response["path"], file_sha256=response["file_sha256"],
                               content_sha256=response["sha256"]))
        for files in inputs["original_trace_input_files_by_proof"].values():
            for name, reference in files.items():
                retained = dict(path=reference["path"], file_sha256=reference["file_sha256"])
                for field in ("canonical_sha256", "sha256"):
                    if field in reference:
                        retained[field] = reference[field]
                if name == "author_output_contract":
                    # This classifies the JSON parsed from original utf8 by the
                    # unchanged Source guard; it does not restamp that record.
                    retained["sourceqa_json_parse"] = "original_author_output_contract"
                    retained["accounting_provenance"] = "derived_reference_from_unchanged_original_custody"
                    if "canonical_sha256" in reference or "sha256" in reference:
                        retained["content_sha256"] = reference.get("canonical_sha256", reference.get("sha256"))
                references.append(retained)
    return references


def sourceqa_json_references(references):
    """Select declared JSON and original contract JSON without inventing pins."""
    return [reference for reference in references if "content_sha256" in reference or
            reference.get("sourceqa_json_parse") == "original_author_output_contract"]


def add_sourceqa_input_arguments(parser):
    for name in ("expected-evaluator-metadata", "original-trace-custody"):
        parser.add_argument("--" + name, help="Separately authenticated caller input file; hashes bind bytes only")
        parser.add_argument("--" + name + "-file-sha256", help="Independently supplied raw file SHA-256")
        parser.add_argument("--" + name + "-content-sha256", help="Independently supplied canonical seal")


def sourceqa_descriptors_from_arguments(args):
    result = {}
    for name in ("expected_evaluator_metadata", "original_trace_custody"):
        values = [getattr(args, name, None), getattr(args, name + "_file_sha256", None),
                  getattr(args, name + "_content_sha256", None)]
        require(all(v is None for v in values) or all(v is not None for v in values),
                "Source-QA caller input needs its path and both independent pins")
        result[name] = dict(zip(("path", "file_sha256", "content_sha256"), values)) if values[0] is not None else None
    return result if any(v is not None for v in result.values()) else None
