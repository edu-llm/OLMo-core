"""Compare dense and split on the evaluator's paired, family-keyed results.

Every target endpoint is recomputed from paired per-example sufficient statistics.
Token-micro and example-macro NLL/accuracy are reported together so weighting
cannot drift silently. Exact generation remains a family/condition diagnostic;
Metamath validity stays excluded until the versioned sound tri-state integration.
No cross-family weighting is introduced.

It verifies evaluator controls and cohorts. Training-control equality must come
from the saved platform configs and the arm YAML equality test; the old local
``arm_fingerprint.json``/mask-sidecar workflow is not used by platform runs.

    python src/scripts/train/p3_math_split/evals/compare_arms.py \
      --dense results/dense.json --split results/split.json \
      --dense-config runs/dense/config.json \
      --split-config runs/split/config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import expected_diagnostic_cohort_size

RESULT_SCHEMA_VERSION = "p3-eval-v9"
COMPARISON_SCHEMA_VERSION = "p3-comparison-v5"
METAMATH_VERIFIER_SCHEMA_VERSION = "p3-metamath-tristate-v1"
MODEL_EXPORT_SCHEMA_VERSION = "p3-model-export-v1"
FINAL_CHECKPOINT_STEP = 23_166
SAFETENSORS_SHARD = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
SAFETENSORS_INDEX = "model.safetensors.index.json"
INFERENCE_SCOPE = (
    "Descriptive paired estimates conditional on the two seed-42 checkpoints. "
    "There is no equivalence claim and no non-inferiority claim because no margin "
    "has been approved."
)
_MISSING = object()
MATCHED_RESULT_PATHS = (
    ("evaluation_controls", "evaluator_seed"),
    ("evaluation_controls", "conditions"),
    ("evaluation_controls", "condition_cohort_policy"),
    ("evaluation_controls", "do_sample"),
    ("evaluation_controls", "temperature"),
    ("evaluation_controls", "context_length"),
    ("evaluation_controls", "max_new_tokens"),
    ("evaluation_controls", "limit"),
    ("evaluation_controls", "nll_chunk_size"),
    ("evaluation_controls", "nll_context_policy"),
    ("evaluation_controls", "nll_target_policy"),
    ("evaluation_controls", "generation_backend"),
    ("input_provenance", "hash_algorithm"),
    ("input_provenance", "corpus_hash_policy"),
    ("input_provenance", "tokenizer_sha256"),
    ("input_provenance", "corpus_sha256"),
    ("input_provenance", "eval_shard_sha256"),
    ("input_provenance", "heldout_manifest_sha256"),
    ("input_provenance", "train_shard_sha256"),
    ("input_provenance", "evaluator_sha256"),
    ("input_provenance", "model", "checkpoint_step"),
    ("input_provenance", "model", "base_model_id"),
    ("input_provenance", "model", "base_model_revision"),
    ("input_provenance", "model", "initial_weights_sha256"),
    ("input_provenance", "model", "source_commit"),
    ("input_provenance", "model", "model_type"),
    ("input_provenance", "model", "architectures"),
    ("input_provenance", "model", "semantic_config_sha256"),
    ("input_provenance", "model", "export_metadata_schema"),
)
REQUIRED_ONLY_RESULT_PATHS = (
    ("input_provenance", "model", "resolved_path"),
    ("input_provenance", "model", "arm"),
    ("input_provenance", "model", "trained_weight_files"),
    ("input_provenance", "model", "trained_weights_root_sha256"),
    ("input_provenance", "model", "export_metadata"),
)
REQUIRED_FAMILY_KEYS = (
    "source_examples",
    "context_eligible_examples",
    "evaluated_examples",
    "conditions",
)
REQUIRED_CONDITION_KEYS = (
    "target_nll_sum",
    "target_tokens",
    "target_correct",
    "target_token_micro_nll_per_token",
    "target_example_macro_nll_per_token",
    "target_token_micro_accuracy",
    "target_example_macro_accuracy",
    "source_examples",
    "context_eligible_examples",
    "evaluated_examples",
    "generation_attempted_examples",
    "whole_proof_budget_eligible_examples",
    "whole_proof_budget_ineligible_examples",
    "whole_proof_budget_coverage_evaluated",
    "exact_match_count_evaluated",
    "exact_match_rate_evaluated",
    "exact_match_count_budget_eligible",
    "exact_match_rate_budget_eligible",
    "per_example",
)
REQUIRED_ITEM_KEYS = (
    "id",
    "nll_sum",
    "target_tokens",
    "target_correct",
    "target_nll_per_token",
    "target_token_accuracy",
    "exact_match",
    "whole_proof_budget_eligible",
    "generation_attempted",
)
REQUIRED_METAMATH_VERIFICATION_KEYS = (
    "availability",
    "verifier_schema_version",
    "condition_supported",
    "condition_reason",
    "evaluated_count",
    "valid_count",
    "invalid_count",
    "unknown_count",
    "excluded_count",
    "unavailable_count",
    "decided_count",
    "valid_rate_decided",
    "valid_rate_denominator",
)
REQUIRED_METAMATH_ITEM_KEYS = (
    "status",
    "verifier_schema_version",
    "target_label",
    "source_database",
    "reason_code",
    "reason",
)
METAMATH_DECIDED_STATUSES = frozenset(("valid", "invalid"))
METAMATH_EVALUATED_STATUSES = frozenset(("valid", "invalid", "unknown"))
METAMATH_NONDECIDED_STATUSES = frozenset(("excluded", "unavailable"))
EXPECTED_LOADED_METAMATH_DATABASES = ("iset", "nf", "set")
ALLOWED_CONFIG_DIFFERENCES = {
    ("arm",),
    ("platform_run_manifest_id",),
    ("platform_run_manifest_sha256",),
    ("train_module", "arm"),
    ("trainer", "save_folder"),
    ("trainer", "callbacks", "wandb", "name"),
}


def _recompute_metamath_status_counts(per_example):
    counts = {
        "valid_count": 0,
        "invalid_count": 0,
        "unknown_count": 0,
        "excluded_count": 0,
        "unavailable_count": 0,
    }
    for item in per_example:
        metamath = item.get("metamath")
        if not isinstance(metamath, dict):
            counts["unavailable_count"] += 1
            continue
        status = metamath.get("status")
        if status == "valid":
            counts["valid_count"] += 1
        elif status == "invalid":
            counts["invalid_count"] += 1
        elif status == "unknown":
            counts["unknown_count"] += 1
        elif status == "excluded":
            counts["excluded_count"] += 1
        else:
            counts["unavailable_count"] += 1
    return counts


def _validate_metamath_availability(availability, context):
    if not isinstance(availability, dict):
        raise ValueError(f"{context}: metamath availability must be an object")
    status = availability.get("status")
    if status == "available":
        if not availability.get("mm_dir_supplied"):
            raise ValueError(f"{context}: available Metamath validity requires mm_dir_supplied")
        if not availability.get("metamath_sources_verified"):
            raise ValueError(
                f"{context}: available Metamath validity requires metamath_sources_verified"
            )
        loaded = availability.get("loaded_source_databases")
        if loaded != list(EXPECTED_LOADED_METAMATH_DATABASES):
            raise ValueError(
                f"{context}: available Metamath validity requires loaded databases "
                f"{list(EXPECTED_LOADED_METAMATH_DATABASES)!r}, got {loaded!r}"
            )
        if availability.get("required_schema") != METAMATH_VERIFIER_SCHEMA_VERSION:
            raise ValueError(f"{context}: metamath required_schema must match the verifier contract")
        if availability.get("detected_schema") != METAMATH_VERIFIER_SCHEMA_VERSION:
            raise ValueError(f"{context}: metamath detected_schema must match the verifier contract")
        if availability.get("reason") is not None:
            raise ValueError(f"{context}: available Metamath validity must not carry a reason")
    elif status == "unavailable":
        if not availability.get("reason"):
            raise ValueError(f"{context}: unavailable Metamath validity requires a reason")
    else:
        raise ValueError(f"{context}: metamath availability status is invalid")


def _validate_metamath_item(metamath, context):
    if not isinstance(metamath, dict):
        raise ValueError(f"{context}: metamath must be an object")
    for key in REQUIRED_METAMATH_ITEM_KEYS:
        if key not in metamath:
            raise ValueError(f"{context}: metamath key {key!r} is missing")
    status = metamath["status"]
    if status not in METAMATH_DECIDED_STATUSES | METAMATH_EVALUATED_STATUSES | METAMATH_NONDECIDED_STATUSES:
        raise ValueError(f"{context}: metamath status {status!r} is invalid")
    if status in METAMATH_EVALUATED_STATUSES:
        if metamath["verifier_schema_version"] != METAMATH_VERIFIER_SCHEMA_VERSION:
            raise ValueError(
                f"{context}: metamath verifier_schema_version must be "
                f"{METAMATH_VERIFIER_SCHEMA_VERSION!r}"
            )
    if "valid" in metamath:
        raise ValueError(f"{context}: boolean metamath.valid is forbidden")


def _validate_metamath_verification(label, family, condition, value, *, result):
    context = f"{label}/{family}/{condition}"
    verification = value.get("metamath_verification")
    if verification is None:
        raise ValueError(f"{context}: metamath_verification is missing")
    if not isinstance(verification, dict):
        raise ValueError(f"{context}: metamath_verification must be an object")
    for key in REQUIRED_METAMATH_VERIFICATION_KEYS:
        if key not in verification:
            raise ValueError(f"{context}: metamath_verification key {key!r} is missing")
    availability = verification["availability"]
    _validate_metamath_availability(availability, context)
    if verification["valid_rate_denominator"] != "valid_count + invalid_count":
        raise ValueError(f"{context}: metamath valid_rate_denominator is invalid")
    if availability["status"] == "available" and result.get("metamath_sources") is None:
        raise ValueError(f"{context}: metamath_sources is required when validity is available")
    condition_supported = verification["condition_supported"]

    per_example = value["per_example"]
    indexed = _index_unique(per_example, context)
    for example_id, item in indexed.items():
        if item.get("metamath") is None:
            raise ValueError(f"{context}/{example_id}: metamath item metadata is missing")
        metamath = item["metamath"]
        _validate_metamath_item(metamath, f"{context}/{example_id}")
        if availability["status"] != "available" and metamath["status"] in METAMATH_EVALUATED_STATUSES:
            raise ValueError(
                f"{context}/{example_id}: evaluated metamath status is forbidden when "
                "validity is unavailable"
            )
        if not condition_supported and metamath["status"] in METAMATH_EVALUATED_STATUSES:
            raise ValueError(
                f"{context}/{example_id}: evaluated metamath status is forbidden for "
                "unsupported conditions"
            )

    counts = _recompute_metamath_status_counts(per_example)
    decided = counts["valid_count"] + counts["invalid_count"]
    evaluated = counts["valid_count"] + counts["invalid_count"] + counts["unknown_count"]
    if availability["status"] != "available" or not condition_supported:
        expected_counts = {
            "valid_count": 0,
            "invalid_count": 0,
            "unknown_count": 0,
            "decided_count": 0,
            "evaluated_count": 0,
            "valid_rate_decided": None,
        }
    else:
        expected_counts = {
            **counts,
            "decided_count": decided,
            "evaluated_count": evaluated,
            "valid_rate_decided": (counts["valid_count"] / decided if decided else None),
        }

    for key in ("valid_count", "invalid_count", "unknown_count", "excluded_count", "unavailable_count"):
        if verification[key] != counts[key]:
            raise ValueError(
                f"{context}: metamath {key}={verification[key]!r} does not match "
                f"per-example recomputation {counts[key]!r}"
            )
    for key in ("decided_count", "evaluated_count", "valid_rate_decided"):
        actual = verification[key]
        expected = expected_counts[key]
        if key == "valid_rate_decided":
            _assert_close(actual, expected, f"{context}.{key}")
        elif actual != expected:
            raise ValueError(
                f"{context}: metamath {key}={actual!r} does not match recomputation {expected!r}"
            )

    cohort_size = len(per_example)
    if sum(counts.values()) != cohort_size:
        raise ValueError(f"{context}: metamath per-example statuses do not sum to cohort size")
    if verification["verifier_schema_version"] != (
        METAMATH_VERIFIER_SCHEMA_VERSION
        if availability["status"] == "available"
        else availability.get("detected_schema")
    ):
        raise ValueError(f"{context}: metamath verifier_schema_version is inconsistent")


def _display(value) -> str:
    return "<missing>" if value is _MISSING else repr(value)


def _get_path(value, path):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _path_name(path) -> str:
    return ".".join(path)


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(_flatten(child, prefix + (str(key),)))
        return out
    if isinstance(value, list):
        return {prefix + (str(i),): child for i, child in enumerate(value)}
    return {prefix: value}


def validate_training_configs(dense, split):
    if dense.get("arm") != "dense":
        raise ValueError("dense config does not declare arm=dense")
    if split.get("arm") != "split":
        raise ValueError("split config does not declare arm=split")
    if dense.get("train_module", {}).get("arm") != "dense":
        raise ValueError("dense config does not declare train_module.arm=dense")
    if split.get("train_module", {}).get("arm") != "split":
        raise ValueError("split config does not declare train_module.arm=split")
    for config, label in ((dense, "dense"), (split, "split")):
        source_commit = config.get("source_commit")
        if not isinstance(source_commit, str) or not source_commit.strip():
            raise ValueError(f"{label} config source_commit must be nonempty")
    dense_seed = dense.get("init_seed", _MISSING)
    split_seed = split.get("init_seed", _MISSING)
    if dense_seed != 42 or split_seed != 42:
        raise ValueError(
            "both checkpoint configs must declare init_seed seed 42; "
            f"dense={_display(dense_seed)}, split={_display(split_seed)}"
        )
    d = _flatten(dense)
    s = _flatten(split)
    differences = {}
    for path in set(d) | set(s):
        dense_value = d.get(path, _MISSING)
        split_value = s.get(path, _MISSING)
        if path not in ALLOWED_CONFIG_DIFFERENCES and dense_value != split_value:
            differences[path] = (dense_value, split_value)
    if differences:
        sample = ", ".join(
            f"{'.'.join(path)}=({_display(values[0])}, {_display(values[1])})"
            for path, values in sorted(differences.items())[:5]
        )
        raise ValueError(f"training configs differ outside the arm: {sample}")


def _require_path(result, path, label):
    value = _get_path(result, path)
    if value is _MISSING:
        raise ValueError(f"{_path_name(path)} is missing from {label} result")
    return value


def _index_unique(items, context):
    indexed = {}
    for item in items:
        example_id = item.get("id", _MISSING)
        if example_id is _MISSING:
            raise ValueError(f"{context}: item id is missing")
        if example_id in indexed:
            raise ValueError(f"{context}: duplicate example ID {example_id!r}")
        indexed[example_id] = item
    return indexed


def _assert_close(actual, expected, context):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(
                f"{context} is inconsistent: reported={actual!r}, " f"recomputed={expected!r}"
            )
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"{context} is inconsistent: reported={actual!r}, " f"recomputed={expected!r}"
        )


def _nonnegative_integer(value, context):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return value


def _validate_sha256_value(value, context):
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_sha256_value(child, f"{context}.{key}")
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be lowercase 64-hex SHA-256")


def _validate_provenance_digests(value, context):
    if not isinstance(value, dict):
        return
    for key, child in value.items():
        child_context = f"{context}.{key}"
        if key.endswith("_sha256"):
            _validate_sha256_value(child, child_context)
        else:
            _validate_provenance_digests(child, child_context)


def _validated_weight_layout(filenames, context):
    if filenames == {"model.safetensors"}:
        return ["model.safetensors"], None
    if "model.safetensors" in filenames:
        raise ValueError(f"{context}: trained weight file inventory has extra files")
    if SAFETENSORS_INDEX not in filenames:
        raise ValueError(
            f"{context}: trained weight inventory must contain one safetensors file "
            "or exact shards plus the index"
        )
    shard_names = filenames - {SAFETENSORS_INDEX}
    matches = {name: SAFETENSORS_SHARD.fullmatch(name) for name in shard_names}
    if not matches or any(match is None for match in matches.values()):
        raise ValueError(f"{context}: trained weight shard filenames are invalid")
    totals = {int(match.group(2)) for match in matches.values()}
    if len(totals) != 1:
        raise ValueError(f"{context}: trained weight shard counts disagree")
    total = next(iter(totals))
    expected = {f"model-{index:05d}-of-{total:05d}.safetensors" for index in range(1, total + 1)}
    if shard_names != expected:
        raise ValueError(f"{context}: trained weight shard inventory is incomplete")
    return sorted(shard_names), SAFETENSORS_INDEX


def _trained_weight_inventory_root(files, context):
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{context}: trained weight inventory must be nonempty")
    shard_names, index_name = _validated_weight_layout(set(files), context)
    for filename in shard_names + ([index_name] if index_name is not None else []):
        entry = files.get(filename)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "dtype"}:
            raise ValueError(
                f"{context}: trained weight entry {filename!r} must contain "
                "exactly sha256, bytes, and dtype"
            )
        _validate_sha256_value(entry["sha256"], f"{context}.{filename}.sha256")
        size = entry["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{context}: trained weight entry {filename!r} bytes must be positive")
        expected_dtype = "json" if filename == index_name else "BF16"
        if entry["dtype"] != expected_dtype:
            raise ValueError(
                f"{context}: trained weight entry {filename!r} dtype must be " f"{expected_dtype!r}"
            )
    payload = json.dumps(
        files,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _platform_manifest_identity(value, context, *, allow_unavailable=False):
    manifest_id = value.get("platform_run_manifest_id", _MISSING)
    manifest_sha256 = value.get("platform_run_manifest_sha256", _MISSING)
    if manifest_id is _MISSING and manifest_sha256 is _MISSING:
        return None
    if (
        allow_unavailable
        and manifest_id in {_MISSING, ""}
        and manifest_sha256
        in {
            _MISSING,
            "",
        }
    ):
        return None
    if manifest_id is _MISSING or not isinstance(manifest_id, str) or not manifest_id.strip():
        raise ValueError(f"{context} platform run manifest ID must be nonempty")
    manifest_id = manifest_id.strip()
    if manifest_sha256 is _MISSING:
        return manifest_id, None
    if (
        not isinstance(manifest_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None
    ):
        raise ValueError(f"{context} platform run manifest SHA-256 must be lowercase 64-hex")
    return manifest_id, manifest_sha256


def _validate_input_provenance(result, label, evaluated_families):
    provenance = _require_path(result, ("input_provenance",), label)
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} input_provenance must be an object")
    if provenance["hash_algorithm"] != "sha256":
        raise ValueError(f"{label} input_provenance.hash_algorithm must equal 'sha256'")
    _validate_provenance_digests(provenance, f"{label}.input_provenance")
    for map_name in (
        "eval_shard_sha256",
        "heldout_manifest_sha256",
        "train_shard_sha256",
    ):
        digest_map = provenance[map_name]
        if not isinstance(digest_map, dict) or set(digest_map) != evaluated_families:
            raise ValueError(
                f"{label} input_provenance.{map_name} family keys must exactly "
                "match evaluated families"
            )


def _validate_denominator_order(value, context):
    source = _nonnegative_integer(value["source_examples"], f"{context}.source_examples")
    context_eligible = _nonnegative_integer(
        value["context_eligible_examples"],
        f"{context}.context_eligible_examples",
    )
    evaluated = _nonnegative_integer(value["evaluated_examples"], f"{context}.evaluated_examples")
    if not 0 <= evaluated <= context_eligible <= source:
        raise ValueError(
            f"{context}: require 0 <= evaluated_examples <= "
            "context_eligible_examples <= source_examples"
        )
    return source, context_eligible, evaluated


def _validate_model_provenance(result, label):
    model = _require_path(result, ("input_provenance", "model"), label)
    if not isinstance(model, dict):
        raise ValueError(f"{label} input_provenance.model must be an object")
    result_arm = result.get("arm")
    if result_arm not in {"dense", "split"}:
        raise ValueError(f"{label} result arm must be 'dense' or 'split'")
    model_arm = model.get("arm")
    if model_arm != result_arm:
        raise ValueError(
            f"{label} model provenance arm {model_arm!r} differs from result arm {result_arm!r}"
        )
    checkpoint_step = model["checkpoint_step"]
    if (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step <= 0
    ):
        raise ValueError(f"{label} checkpoint_step must be a positive integer")
    if checkpoint_step != FINAL_CHECKPOINT_STEP:
        raise ValueError(
            f"{label} checkpoint_step must be {FINAL_CHECKPOINT_STEP} for "
            "reportable evaluation"
        )
    for key in (
        "base_model_id",
        "base_model_revision",
        "initial_weights_sha256",
        "source_commit",
        "trained_weights_root_sha256",
        "semantic_config_sha256",
        "export_metadata_schema",
    ):
        value = model[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} model provenance {key} must be nonempty")
    if model["export_metadata_schema"] != MODEL_EXPORT_SCHEMA_VERSION:
        raise ValueError(f"{label} export_metadata_schema must be {MODEL_EXPORT_SCHEMA_VERSION!r}")
    digest = model["initial_weights_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} initial_weights_sha256 must be a SHA-256 digest")
    trained_weight_files = model.get("trained_weight_files")
    trained_root = _trained_weight_inventory_root(
        trained_weight_files,
        f"{label} model trained weight inventory",
    )
    if model["trained_weights_root_sha256"] != trained_root:
        raise ValueError(f"{label} model trained weight root differs from its canonical inventory")
    metadata = model["export_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{label} model export_metadata must be an object")
    if metadata.get("schema_version") != MODEL_EXPORT_SCHEMA_VERSION:
        raise ValueError(f"{label} model export_metadata must use {MODEL_EXPORT_SCHEMA_VERSION!r}")
    model_manifest = _platform_manifest_identity(model, f"{label} model")
    metadata_manifest = _platform_manifest_identity(
        metadata,
        f"{label} export metadata",
    )
    if metadata_manifest != model_manifest:
        raise ValueError(f"{label} platform run manifest differs between model and export metadata")
    for key in (
        "checkpoint_step",
        "arm",
        "base_model_id",
        "base_model_revision",
        "initial_weights_sha256",
        "source_commit",
        "trained_weight_files",
        "trained_weights_root_sha256",
    ):
        if metadata.get(key, _MISSING) != model[key]:
            field = key.replace("_", " ") if key.startswith("trained_weight") else key
            raise ValueError(
                f"{label} model export_metadata {field} differs from extracted provenance"
            )


def validate_result_config_binding(result, config, *, expected_arm, label):
    """Bind one result/export identity back to its saved checkpoint config."""
    if expected_arm not in {"dense", "split"}:
        raise ValueError(f"unknown expected arm {expected_arm!r}")
    _validate_model_provenance(result, label)
    model = result["input_provenance"]["model"]
    for actual, context in (
        (result.get("arm"), "result arm"),
        (model.get("arm"), "model arm"),
        (config.get("arm"), "config arm"),
        (config.get("train_module", {}).get("arm"), "config train_module.arm"),
    ):
        if actual != expected_arm:
            raise ValueError(f"{label} {context} must be {expected_arm!r}, got {actual!r}")
    source_commit = config.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise ValueError(f"{label} config source_commit must be nonempty")
    if source_commit.strip() != model["source_commit"]:
        raise ValueError(f"{label} config source_commit differs from evaluated model provenance")
    config_manifest = _platform_manifest_identity(
        config,
        f"{label} config",
        allow_unavailable=True,
    )
    model_manifest = _platform_manifest_identity(model, f"{label} model")
    if config_manifest != model_manifest:
        raise ValueError(
            f"{label} platform run manifest differs between config and model provenance"
        )


def _validate_condition_schema(label, family, condition, value, *, result):
    context = f"{label}/{family}/{condition}"
    if not isinstance(value, dict):
        raise ValueError(f"{context}: condition result must be an object")
    for key in REQUIRED_CONDITION_KEYS:
        if key not in value:
            raise ValueError(f"{context}: required key {key!r} is missing")
    if "exact_match_rate_all" in value:
        raise ValueError(f"{context}: ambiguous deprecated exact_match_rate_all is not accepted")
    if any(key == "metamath_valid" or key.startswith("metamath_valid_") for key in value):
        raise ValueError(f"{context}: boolean Metamath validity fields are not accepted")
    if not isinstance(value["per_example"], list):
        raise ValueError(f"{context}: per_example must be a list")
    _, _, evaluated = _validate_denominator_order(value, context)

    indexed = _index_unique(value["per_example"], context)
    if len(indexed) != evaluated:
        raise ValueError(
            f"{context}: evaluated_examples must equal the unique per-example cohort size"
        )
    for example_id, item in indexed.items():
        for key in REQUIRED_ITEM_KEYS:
            if key not in item:
                raise ValueError(f"{context}/{example_id}: required key {key!r} is missing")
        tokens = item["target_tokens"]
        correct = item["target_correct"]
        nll_sum = item["nll_sum"]
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError(f"{context}/{example_id}: target_tokens must include EOS")
        if not isinstance(correct, int) or not 0 <= correct <= tokens:
            raise ValueError(f"{context}/{example_id}: invalid target_correct")
        if not math.isfinite(float(nll_sum)) or nll_sum < 0:
            raise ValueError(f"{context}/{example_id}: invalid nll_sum")
        for key in (
            "exact_match",
            "whole_proof_budget_eligible",
            "generation_attempted",
        ):
            if type(item[key]) is not bool:
                raise ValueError(f"{context}/{example_id}: {key} must be an actual bool")
        metamath = item.get("metamath")
        if "metamath_valid" in item or (isinstance(metamath, dict) and "valid" in metamath):
            raise ValueError(
                f"{context}/{example_id}: boolean Metamath validity is forbidden "
                "without the sound versioned tri-state API"
            )
        if family == "metamath":
            if metamath is None:
                raise ValueError(f"{context}/{example_id}: metamath item metadata is missing")
            _validate_metamath_item(metamath, f"{context}/{example_id}")
        if item["exact_match"] and not item["generation_attempted"]:
            raise ValueError(f"{context}/{example_id}: exact_match implies generation_attempted")
        if item["whole_proof_budget_eligible"] and not item["generation_attempted"]:
            raise ValueError(
                f"{context}/{example_id}: budget eligible implies generation_attempted"
            )
        _assert_close(
            item["target_nll_per_token"],
            nll_sum / tokens,
            f"{context}/{example_id}.target_nll_per_token",
        )
        _assert_close(
            item["target_token_accuracy"],
            correct / tokens,
            f"{context}/{example_id}.target_token_accuracy",
        )

    attempted = _nonnegative_integer(
        value["generation_attempted_examples"],
        f"{context}.generation_attempted_examples",
    )
    if attempted > evaluated:
        raise ValueError(
            f"{context}: generation_attempted_examples must not exceed evaluated_examples"
        )
    budget_eligible = _nonnegative_integer(
        value["whole_proof_budget_eligible_examples"],
        f"{context}.whole_proof_budget_eligible_examples",
    )
    budget_ineligible = _nonnegative_integer(
        value["whole_proof_budget_ineligible_examples"],
        f"{context}.whole_proof_budget_ineligible_examples",
    )
    if budget_eligible + budget_ineligible != evaluated:
        raise ValueError(f"{context}: budget eligible+ineligible must equal evaluated")
    exact_evaluated = _nonnegative_integer(
        value["exact_match_count_evaluated"],
        f"{context}.exact_match_count_evaluated",
    )
    if exact_evaluated > evaluated:
        raise ValueError(f"{context}: exact evaluated count exceeds evaluated examples")
    if exact_evaluated > attempted:
        raise ValueError(
            f"{context}: exact_match_count_evaluated must not exceed "
            "generation_attempted_examples"
        )
    exact_eligible = _nonnegative_integer(
        value["exact_match_count_budget_eligible"],
        f"{context}.exact_match_count_budget_eligible",
    )
    if exact_eligible > budget_eligible:
        raise ValueError(f"{context}: exact eligible count exceeds eligible examples")

    items = list(indexed.values())
    target_tokens = sum(item["target_tokens"] for item in items)
    target_correct = sum(item["target_correct"] for item in items)
    target_nll_sum = sum(item["nll_sum"] for item in items)
    n_examples = len(items)
    expected = {
        "target_nll_sum": target_nll_sum,
        "target_tokens": target_tokens,
        "target_correct": target_correct,
        "target_token_micro_nll_per_token": (
            target_nll_sum / target_tokens if target_tokens else None
        ),
        "target_token_micro_accuracy": (target_correct / target_tokens if target_tokens else None),
        "target_example_macro_nll_per_token": (
            sum(item["nll_sum"] / item["target_tokens"] for item in items) / n_examples
            if n_examples
            else None
        ),
        "target_example_macro_accuracy": (
            sum(item["target_correct"] / item["target_tokens"] for item in items) / n_examples
            if n_examples
            else None
        ),
        "evaluated_examples": n_examples,
        "generation_attempted_examples": sum(item["generation_attempted"] for item in items),
        "whole_proof_budget_eligible_examples": sum(
            item["whole_proof_budget_eligible"] for item in items
        ),
        "whole_proof_budget_ineligible_examples": sum(
            not item["whole_proof_budget_eligible"] for item in items
        ),
        "exact_match_count_evaluated": sum(item["exact_match"] for item in items),
        "exact_match_count_budget_eligible": sum(
            item["exact_match"] and item["whole_proof_budget_eligible"] for item in items
        ),
    }
    for key, recomputed in expected.items():
        _assert_close(value[key], recomputed, f"{context}.{key}")

    eligible = expected["whole_proof_budget_eligible_examples"]
    rates = {
        "whole_proof_budget_coverage_evaluated": (eligible / n_examples if n_examples else None),
        "exact_match_rate_evaluated": (
            expected["exact_match_count_evaluated"] / n_examples if n_examples else None
        ),
        "exact_match_rate_budget_eligible": (
            expected["exact_match_count_budget_eligible"] / eligible if eligible else None
        ),
    }
    for key, recomputed in rates.items():
        _assert_close(value[key], recomputed, f"{context}.{key}")
    if family == "metamath":
        _validate_metamath_verification(label, family, condition, value, result=result)
    return frozenset(indexed)


def _validate_condition_cohort_membership(
    label,
    family,
    condition,
    condition_ids,
    *,
    family_evaluated,
    present_ids,
):
    context = f"{label}/{family}/{condition}"
    if condition == "facts_present":
        if len(condition_ids) != family_evaluated:
            raise ValueError(
                f"{context}: condition cohort must equal the full family evaluated cohort"
            )
        return
    expected_size = expected_diagnostic_cohort_size(family_evaluated)
    if expected_size == 0:
        if condition_ids:
            raise ValueError(f"{context}: diagnostic cohort must be empty when policy size is 0")
        return
    if len(condition_ids) != expected_size:
        raise ValueError(
            f"{context}: diagnostic cohort size {len(condition_ids)!r} "
            f"does not match policy count {expected_size!r}"
        )
    if not condition_ids:
        raise ValueError(f"{context}: diagnostic cohort must be nonempty")
    if not condition_ids.issubset(present_ids):
        raise ValueError(f"{context}: diagnostic cohort must be a subset of facts_present")


def _validate_result_schema(result, label):
    if result.get("schema_version", _MISSING) != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{label} result schema differs from required {RESULT_SCHEMA_VERSION!r}; "
            "old or incomplete evaluator outputs are not comparable"
        )
    for path in MATCHED_RESULT_PATHS + REQUIRED_ONLY_RESULT_PATHS:
        _require_path(result, path, label)
    families = result.get("families", _MISSING)
    if not isinstance(families, dict):
        raise ValueError(f"families is missing from {label} result")
    _validate_input_provenance(result, label, set(families))
    _validate_model_provenance(result, label)
    requested_conditions = set(_require_path(result, ("evaluation_controls", "conditions"), label))
    for family, family_result in families.items():
        if not isinstance(family_result, dict):
            raise ValueError(f"{label}/{family}: family result must be an object")
        for key in REQUIRED_FAMILY_KEYS:
            if key not in family_result:
                raise ValueError(f"{label}/{family}: required key {key!r} is missing")
        _validate_denominator_order(family_result, f"{label}/{family}")
        conditions = family_result["conditions"]
        if not isinstance(conditions, dict):
            raise ValueError(f"{label}/{family}: conditions must be an object")
        if set(conditions) != requested_conditions:
            raise ValueError(
                f"{label}/{family}: evaluation_controls.conditions differs from "
                "family condition keys"
            )
        cohort_ids_by_condition = {}
        family_evaluated = family_result["evaluated_examples"]
        for condition, value in conditions.items():
            condition_ids = _validate_condition_schema(label, family, condition, value, result=result)
            cohort_ids_by_condition[condition] = condition_ids
            for key in ("source_examples", "context_eligible_examples"):
                _assert_close(
                    value[key],
                    family_result[key],
                    f"{label}/{family}/{condition}.{key}",
                )
            if condition == "facts_present":
                if value["evaluated_examples"] != family_evaluated:
                    raise ValueError(
                        f"{label}/{family}/{condition}: condition cohort must equal "
                        "the full family evaluated cohort"
                    )
            elif value["evaluated_examples"] != len(condition_ids):
                raise ValueError(
                    f"{label}/{family}/{condition}: evaluated_examples must equal "
                    "the diagnostic cohort size"
                )
        present_ids = cohort_ids_by_condition.get("facts_present")
        if present_ids is None:
            raise ValueError(f"{label}/{family}: facts_present cohort is required")
        for condition, condition_ids in cohort_ids_by_condition.items():
            _validate_condition_cohort_membership(
                label,
                family,
                condition,
                condition_ids,
                family_evaluated=family_evaluated,
                present_ids=present_ids,
            )


def validate_eval_compatibility(dense, split):
    _validate_result_schema(dense, "dense")
    _validate_result_schema(split, "split")
    if dense.get("arm") != "dense" or split.get("arm") != "split":
        raise ValueError("results must be dense and split respectively")
    for path in MATCHED_RESULT_PATHS:
        dense_value = _require_path(dense, path, "dense")
        split_value = _require_path(split, path, "split")
        if dense_value != split_value:
            raise ValueError(
                f"{_path_name(path)} differs: dense={dense_value!r}, " f"split={split_value!r}"
            )
    if set(dense["families"]) != set(split["families"]):
        raise ValueError("evaluated family sets differ")
    for family in dense["families"]:
        dense_family = dense["families"][family]
        split_family = split["families"][family]
        for key in (
            "source_examples",
            "context_eligible_examples",
            "evaluated_examples",
        ):
            if dense_family[key] != split_family[key]:
                raise ValueError(
                    f"{family}: {key} differs: dense={dense_family[key]!r}, "
                    f"split={split_family[key]!r}"
                )
        if set(dense_family["conditions"]) != set(split_family["conditions"]):
            raise ValueError(f"{family}: evaluated condition sets differ")
        for condition in dense_family["conditions"]:
            dense_ids = set(
                _index_unique(
                    dense_family["conditions"][condition]["per_example"],
                    f"dense/{family}/{condition}",
                )
            )
            split_ids = set(
                _index_unique(
                    split_family["conditions"][condition]["per_example"],
                    f"split/{family}/{condition}",
                )
            )
            if dense_ids != split_ids:
                raise ValueError(f"{family}/{condition}: cohort IDs differ between arms")
    if "metamath" in dense["families"]:
        if dense.get("metamath_sources") != split.get("metamath_sources"):
            raise ValueError("metamath_sources differs between arms")


def _bootstrap_interval(differences):
    differences.sort()
    n_boot = len(differences)
    low_index = min(n_boot - 1, int(0.025 * n_boot))
    high_index = min(n_boot - 1, max(0, math.ceil(0.975 * n_boot) - 1))
    return differences[low_index], differences[high_index]


def _outcome_endpoint(pairs, n_boot, seed):
    """Paired descriptive rate difference and example-bootstrap interval."""
    if not pairs:
        return None
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        dense_estimate = sum(pairs[index][0] for index in indices) / n
        split_estimate = sum(pairs[index][1] for index in indices) / n
        diffs.append(split_estimate - dense_estimate)
    low, high = _bootstrap_interval(diffs)
    dense_estimate = sum(pair[0] for pair in pairs) / n
    split_estimate = sum(pair[1] for pair in pairs) / n
    return {
        "paired_examples": n,
        "dense_estimate": dense_estimate,
        "split_estimate": split_estimate,
        "difference_split_minus_dense": split_estimate - dense_estimate,
        "paired_bootstrap_ci95_low": low,
        "paired_bootstrap_ci95_high": high,
    }


TARGET_ENDPOINTS = {
    "target_token_micro_nll_per_token": ("nll_sum", False),
    "target_example_macro_nll_per_token": ("nll_sum", True),
    "target_token_micro_accuracy": ("target_correct", False),
    "target_example_macro_accuracy": ("target_correct", True),
}


def _target_estimate(pairs, indices, arm_index, numerator, macro):
    selected = [pairs[index][arm_index] for index in indices]
    if macro:
        return sum(float(item[numerator]) / item["target_tokens"] for item in selected) / len(
            selected
        )
    return sum(float(item[numerator]) for item in selected) / sum(
        item["target_tokens"] for item in selected
    )


def _target_endpoint(pairs, numerator, macro, n_boot, seed):
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    n = len(pairs)
    all_indices = list(range(n))
    dense_estimate = _target_estimate(pairs, all_indices, 0, numerator, macro)
    split_estimate = _target_estimate(pairs, all_indices, 1, numerator, macro)
    rng = random.Random(seed)
    differences = []
    for _ in range(n_boot):
        indices = [rng.randrange(n) for _ in range(n)]
        dense_bootstrap = _target_estimate(pairs, indices, 0, numerator, macro)
        split_bootstrap = _target_estimate(pairs, indices, 1, numerator, macro)
        differences.append(split_bootstrap - dense_bootstrap)
    low, high = _bootstrap_interval(differences)
    return {
        "paired_examples": n,
        "dense_estimate": dense_estimate,
        "split_estimate": split_estimate,
        "difference_split_minus_dense": split_estimate - dense_estimate,
        "paired_bootstrap_ci95_low": low,
        "paired_bootstrap_ci95_high": high,
    }


def _paired_items(dense_condition, split_condition, context):
    dense_by_id = _index_unique(dense_condition["per_example"], f"dense/{context}")
    split_by_id = _index_unique(split_condition["per_example"], f"split/{context}")
    if set(dense_by_id) != set(split_by_id):
        raise ValueError(f"{context}: paired IDs differ between arms")
    pairs = []
    for example_id in sorted(dense_by_id, key=str):
        dense_item = dense_by_id[example_id]
        split_item = split_by_id[example_id]
        for key in (
            "target_tokens",
            "whole_proof_budget_eligible",
            "generation_attempted",
        ):
            if dense_item[key] != split_item[key]:
                raise ValueError(f"{context}/{example_id}: {key} eligibility differs between arms")
        pairs.append((dense_item, split_item))
    return pairs


def _metamath_decided(item) -> bool:
    metamath = item.get("metamath")
    return isinstance(metamath, dict) and metamath.get("status") in METAMATH_DECIDED_STATUSES


def _metamath_valid(item) -> bool:
    metamath = item.get("metamath")
    return isinstance(metamath, dict) and metamath.get("status") == "valid"


def _compare_metamath_validity(pairs, n_boot, seed):
    eligible_pairs = []
    unknown_pairs = 0
    excluded_pairs = 0
    unavailable_pairs = 0
    for dense_item, split_item in pairs:
        dense_status = dense_item.get("metamath", {}).get("status")
        split_status = split_item.get("metamath", {}).get("status")
        statuses = {dense_status, split_status}
        if statuses <= METAMATH_DECIDED_STATUSES:
            eligible_pairs.append((dense_item, split_item))
        elif "unknown" in statuses:
            unknown_pairs += 1
        elif "excluded" in statuses:
            excluded_pairs += 1
        else:
            unavailable_pairs += 1
    if not eligible_pairs:
        return {
            "paired_examples": len(pairs),
            "eligible_paired_examples": 0,
            "unknown_paired_examples": unknown_pairs,
            "excluded_paired_examples": excluded_pairs,
            "unavailable_paired_examples": unavailable_pairs,
            "dense_estimate": None,
            "split_estimate": None,
            "difference_split_minus_dense": None,
            "paired_bootstrap_ci95_low": None,
            "paired_bootstrap_ci95_high": None,
            "valid_rate_denominator": "valid_count + invalid_count",
            "verifier_schema_version": METAMATH_VERIFIER_SCHEMA_VERSION,
        }
    rate_pairs = [
        (float(_metamath_valid(dense_item)), float(_metamath_valid(split_item)))
        for dense_item, split_item in eligible_pairs
    ]
    endpoint = _outcome_endpoint(rate_pairs, n_boot, seed)
    return {
        "paired_examples": len(pairs),
        "eligible_paired_examples": len(eligible_pairs),
        "unknown_paired_examples": unknown_pairs,
        "excluded_paired_examples": excluded_pairs,
        "unavailable_paired_examples": unavailable_pairs,
        "dense_estimate": endpoint["dense_estimate"],
        "split_estimate": endpoint["split_estimate"],
        "difference_split_minus_dense": endpoint["difference_split_minus_dense"],
        "paired_bootstrap_ci95_low": endpoint["paired_bootstrap_ci95_low"],
        "paired_bootstrap_ci95_high": endpoint["paired_bootstrap_ci95_high"],
        "valid_rate_denominator": "valid_count + invalid_count",
        "verifier_schema_version": METAMATH_VERIFIER_SCHEMA_VERSION,
    }


def compare_condition(dense, split, *, family, condition, n_boot, seed):
    dc = dense["families"][family]["conditions"].get(condition)
    sc = split["families"][family]["conditions"].get(condition)
    if dc is None or sc is None:
        return None

    context = f"{family}/{condition}"
    pairs = _paired_items(dc, sc, context)
    if not pairs:
        return None

    target_metrics = {
        name: _target_endpoint(
            pairs,
            numerator=numerator,
            macro=macro,
            n_boot=n_boot,
            seed=seed,
        )
        for name, (numerator, macro) in TARGET_ENDPOINTS.items()
    }
    exact_pairs = [
        (float(bool(dense_item["exact_match"])), float(bool(split_item["exact_match"])))
        for dense_item, split_item in pairs
    ]
    eligible_exact_pairs = [
        (
            float(bool(dense_item["exact_match"])),
            float(bool(split_item["exact_match"])),
        )
        for dense_item, split_item in pairs
        if dense_item["whole_proof_budget_eligible"]
    ]
    outcomes = {
        "exact_match_evaluated": _outcome_endpoint(exact_pairs, n_boot, seed),
        "exact_match_budget_eligible": _outcome_endpoint(eligible_exact_pairs, n_boot, seed),
    }
    if family == "metamath":
        dense_verification = dc.get("metamath_verification")
        split_verification = sc.get("metamath_verification")
        if dense_verification is None or split_verification is None:
            raise ValueError(f"{context}: metamath_verification is missing")
        if not dense_verification.get("condition_supported") or not split_verification.get(
            "condition_supported"
        ):
            outcomes["metamath_validity_decided"] = None
        else:
            outcomes["metamath_validity_decided"] = _compare_metamath_validity(pairs, n_boot, seed)

    return {
        "family": family,
        "condition": condition,
        "paired_examples": len(pairs),
        "target_metrics": target_metrics,
        "outcomes": outcomes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense", required=True, help="results JSON from run_eval.py --arm dense")
    ap.add_argument("--split", required=True, help="results JSON from run_eval.py --arm split")
    ap.add_argument("--dense-config", help="dense checkpoint's ConfigSaver config.json")
    ap.add_argument("--split-config", help="split checkpoint's ConfigSaver config.json")
    ap.add_argument(
        "--skip-training-config-check",
        action="store_true",
        help="debugging only; results are not reportable without config equality",
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument(
        "--seed",
        type=int,
        default=20260801,
        help="paired-bootstrap RNG seed; not an additional model-training seed",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dense = json.load(open(args.dense, encoding="utf-8"))
    split = json.load(open(args.split, encoding="utf-8"))
    training_config_verified = False
    dense_config = None
    split_config = None
    if args.skip_training_config_check:
        print("WARNING: training config check skipped; comparison is not reportable")
    elif not args.dense_config or not args.split_config:
        sys.exit(
            "--dense-config and --split-config are required unless "
            "--skip-training-config-check is set"
        )
    else:
        try:
            dense_config = json.load(open(args.dense_config, encoding="utf-8"))
            split_config = json.load(open(args.split_config, encoding="utf-8"))
            validate_training_configs(dense_config, split_config)
        except ValueError as exc:
            sys.exit(str(exc))
    try:
        validate_eval_compatibility(dense, split)
        if dense_config is not None and split_config is not None:
            validate_result_config_binding(
                dense,
                dense_config,
                expected_arm="dense",
                label="dense",
            )
            validate_result_config_binding(
                split,
                split_config,
                expected_arm="split",
                label="split",
            )
            training_config_verified = True
    except ValueError as exc:
        sys.exit(str(exc))

    controls = dense["evaluation_controls"]
    decoding = "sampled" if controls["do_sample"] else "greedy"
    print(f"\n{INFERENCE_SCOPE}")
    print(
        f"decoding: {decoding}; evaluator seed: {controls['evaluator_seed']}; "
        f"paired bootstrap seed: {args.seed}\n"
    )

    out = []
    for family, family_result in dense["families"].items():
        for condition in family_result["conditions"]:
            r = compare_condition(
                dense,
                split,
                family=family,
                condition=condition,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            if r is None:
                continue
            out.append(r)
            print(f"{family}/{condition} ({r['paired_examples']:,} paired examples)")
            for name, endpoint in r["target_metrics"].items():
                if "accuracy" in name:
                    dense_value = f"{endpoint['dense_estimate']:.2%}"
                    split_value = f"{endpoint['split_estimate']:.2%}"
                    difference = f"{endpoint['difference_split_minus_dense']:+.2%}"
                    interval = (
                        f"[{endpoint['paired_bootstrap_ci95_low']:+.2%}, "
                        f"{endpoint['paired_bootstrap_ci95_high']:+.2%}]"
                    )
                else:
                    dense_value = f"{endpoint['dense_estimate']:.4f}"
                    split_value = f"{endpoint['split_estimate']:.4f}"
                    difference = f"{endpoint['difference_split_minus_dense']:+.4f}"
                    interval = (
                        f"[{endpoint['paired_bootstrap_ci95_low']:+.4f}, "
                        f"{endpoint['paired_bootstrap_ci95_high']:+.4f}]"
                    )
                print(
                    f"  {name}: dense {dense_value}, split {split_value}, "
                    f"split-dense {difference}, paired 95% CI {interval}"
                )
            for name, endpoint in r["outcomes"].items():
                if endpoint is None:
                    print(f"  {name}: no eligible paired examples")
                    continue
                if endpoint.get("dense_estimate") is None or endpoint.get("split_estimate") is None:
                    print(f"  {name}: no eligible paired decided examples")
                    continue
                interval = (
                    f"[{endpoint['paired_bootstrap_ci95_low']:+.2%}, "
                    f"{endpoint['paired_bootstrap_ci95_high']:+.2%}]"
                )
                print(
                    f"  {name}: dense {endpoint['dense_estimate']:.2%}, "
                    f"split {endpoint['split_estimate']:.2%}, split-dense "
                    f"{endpoint['difference_split_minus_dense']:+.2%}, "
                    f"paired 95% CI {interval}"
                )
            print()

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        common_provenance = {
            key: value for key, value in dense["input_provenance"].items() if key != "model"
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "schema_version": COMPARISON_SCHEMA_VERSION,
                    "inference_scope": INFERENCE_SCOPE,
                    "training_config_equality_verified": training_config_verified,
                    "evaluation_controls": controls,
                    "input_provenance": common_provenance,
                    "checkpoint_models": {
                        "dense": dense["input_provenance"]["model"],
                        "split": split["input_provenance"]["model"],
                    },
                    "paired_bootstrap": {
                        "resampling_unit": "example ID within each family/condition",
                        "samples": args.n_boot,
                        "seed": args.seed,
                        "interval": "percentile 95%",
                    },
                    "comparisons": out,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
