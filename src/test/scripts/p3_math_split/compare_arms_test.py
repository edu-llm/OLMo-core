"""Paired comparisons over the evaluator's family-keyed output."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from . import load_project_module

compare = load_project_module("compare_arms")

P3_ROOT = Path("src/scripts/train/p3_math_split")
EVAL_COMPARE_ARMS = P3_ROOT / "evals" / "compare_arms.py"
LEGACY_COMPARE_ARMS = P3_ROOT / "compare_arms.py"


def test_compare_arms_entrypoint_lives_under_evals_subfolder():
    assert EVAL_COMPARE_ARMS.is_file()
    assert not LEGACY_COMPARE_ARMS.exists()

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def trained_weight_identity(arm):
    files = {
        "model.safetensors": {
            "sha256": SHA_B if arm == "dense" else SHA_C,
            "bytes": 100 if arm == "dense" else 101,
            "dtype": "BF16",
        }
    }
    root = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return files, root


def item(
    example_id,
    *,
    tokens,
    correct,
    nll_sum,
    exact,
    budget_eligible=True,
    generation_attempted=True,
):
    return {
        "id": str(example_id),
        "nll_sum": nll_sum,
        "target_tokens": tokens,
        "target_correct": correct,
        "target_nll_per_token": nll_sum / tokens,
        "target_token_accuracy": correct / tokens,
        "exact_match": exact,
        "whole_proof_budget_eligible": budget_eligible,
        "generation_attempted": generation_attempted,
    }


def result(arm, items):
    trained_weight_files, trained_weights_root_sha256 = trained_weight_identity(arm)
    target_tokens = sum(entry["target_tokens"] for entry in items)
    target_correct = sum(entry["target_correct"] for entry in items)
    target_nll_sum = sum(entry["nll_sum"] for entry in items)
    eligible = [entry for entry in items if entry["whole_proof_budget_eligible"]]
    exact = sum(bool(entry["exact_match"]) for entry in items)
    exact_eligible = sum(bool(entry["exact_match"]) for entry in eligible)
    return {
        "schema_version": "p3-eval-v5",
        "arm": arm,
        "evaluation_controls": {
            "evaluator_seed": 20260801,
            "conditions": ["facts_present"],
            "condition_cohort_policy": {
                "facts_present": {
                    "selection": "stable-sha256-stratified-per-family-v1",
                    "numerator": 4,
                    "denominator": 5,
                    "rounding": "ceiling",
                },
                "facts_absent": {
                    "selection": "stable-sha256-stratified-per-family-v1",
                    "numerator": 1,
                    "denominator": 10,
                    "rounding": "ceiling",
                    "subset_of": "facts_present",
                },
                "facts_corrupted": {
                    "selection": "stable-sha256-stratified-per-family-v1",
                    "numerator": 1,
                    "denominator": 10,
                    "rounding": "ceiling",
                    "subset_of": "facts_present",
                },
            },
            "do_sample": False,
            "temperature": 0.7,
            "context_length": 16_384,
            "max_new_tokens": 8_192,
            "limit": None,
            "nll_chunk_size": 256,
            "nll_context_policy": "bounded_sliding_window_preserve_predecessor",
            "nll_target_policy": "combined_prompt_target_suffix_plus_single_eos",
        },
        "input_provenance": {
            "hash_algorithm": "sha256",
            "corpus_hash_policy": "evaluated inputs",
            "tokenizer_sha256": SHA_B,
            "corpus_sha256": SHA_C,
            "eval_shard_sha256": {"mizar": SHA_D},
            "heldout_manifest_sha256": {"mizar": SHA_E},
            "train_shard_sha256": {"mizar": SHA_F},
            "evaluator_sha256": SHA_D,
            "model": {
                "resolved_path": f"/models/{arm}",
                "checkpoint_step": 24_540,
                "arm": arm,
                "base_model_id": "Qwen/Qwen2.5-0.5B",
                "base_model_revision": "revision-abc",
                "initial_weights_sha256": SHA_A,
                "source_commit": "c" * 40,
                "platform_run_manifest_id": f"manifest-{arm}",
                "platform_run_manifest_sha256": SHA_D if arm == "dense" else SHA_E,
                "trained_weight_files": trained_weight_files,
                "trained_weights_root_sha256": trained_weights_root_sha256,
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "semantic_config_sha256": SHA_E,
                "export_metadata_schema": "p3-model-export-v1",
                "export_metadata": {
                    "schema_version": "p3-model-export-v1",
                    "checkpoint_step": 24_540,
                    "arm": arm,
                    "base_model_id": "Qwen/Qwen2.5-0.5B",
                    "base_model_revision": "revision-abc",
                    "initial_weights_sha256": SHA_A,
                    "source_commit": "c" * 40,
                    "platform_run_manifest_id": f"manifest-{arm}",
                    "platform_run_manifest_sha256": SHA_D if arm == "dense" else SHA_E,
                    "trained_weight_files": copy.deepcopy(trained_weight_files),
                    "trained_weights_root_sha256": trained_weights_root_sha256,
                },
            },
        },
        "families": {
            "mizar": {
                "source_examples": len(items),
                "context_eligible_examples": len(items),
                "evaluated_examples": len(items),
                "conditions": {
                    "facts_present": {
                        "target_nll_sum": target_nll_sum,
                        "target_tokens": target_tokens,
                        "target_correct": target_correct,
                        "target_token_micro_nll_per_token": target_nll_sum / target_tokens,
                        "target_example_macro_nll_per_token": sum(
                            entry["target_nll_per_token"] for entry in items
                        )
                        / len(items),
                        "target_token_micro_accuracy": target_correct / target_tokens,
                        "target_example_macro_accuracy": sum(
                            entry["target_token_accuracy"] for entry in items
                        )
                        / len(items),
                        "source_examples": len(items),
                        "context_eligible_examples": len(items),
                        "evaluated_examples": len(items),
                        "generation_attempted_examples": sum(
                            bool(entry["generation_attempted"]) for entry in items
                        ),
                        "whole_proof_budget_eligible_examples": len(eligible),
                        "whole_proof_budget_ineligible_examples": len(items) - len(eligible),
                        "whole_proof_budget_coverage_evaluated": len(eligible) / len(items),
                        "exact_match_count_evaluated": exact,
                        "exact_match_rate_evaluated": exact / len(items),
                        "exact_match_count_budget_eligible": exact_eligible,
                        "exact_match_rate_budget_eligible": exact_eligible / len(eligible)
                        if eligible
                        else None,
                        "per_example": items,
                    }
                },
            }
        },
    }


def test_family_condition_comparison_is_paired_and_reports_descriptive_endpoints():
    dense = result(
        "dense",
        [
            item(0, tokens=2, correct=2, nll_sum=2.0, exact=True),
            item(1, tokens=2, correct=0, nll_sum=4.0, exact=False),
            item(2, tokens=2, correct=2, nll_sum=3.0, exact=True),
        ],
    )
    split = result(
        "split",
        [
            item(0, tokens=2, correct=2, nll_sum=1.0, exact=True),
            item(1, tokens=2, correct=2, nll_sum=2.0, exact=True),
            item(2, tokens=2, correct=0, nll_sum=3.0, exact=False),
        ],
    )

    compare.validate_eval_compatibility(dense, split)
    got = compare.compare_condition(
        dense,
        split,
        family="mizar",
        condition="facts_present",
        n_boot=100,
        seed=1,
    )

    assert got["paired_examples"] == 3
    exact = got["outcomes"]["exact_match_evaluated"]
    assert exact["dense_estimate"] == pytest.approx(2 / 3)
    assert exact["split_estimate"] == pytest.approx(2 / 3)
    assert "paired_bootstrap_ci95_low" in exact
    assert "paired_bootstrap_ci95_high" in exact
    assert "mcnemar_p" not in exact
    assert "verdict" not in got


def test_comparator_accepts_distinct_content_addressed_trained_checkpoints():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))

    assert (
        dense["input_provenance"]["model"]["trained_weights_root_sha256"]
        != split["input_provenance"]["model"]["trained_weights_root_sha256"]
    )
    compare.validate_eval_compatibility(dense, split)


def test_micro_and_macro_accuracy_and_nll_are_bootstrapped_from_sufficient_stats():
    dense = result(
        "dense",
        [
            item("long", tokens=100, correct=90, nll_sum=100.0, exact=False),
            item("short", tokens=1, correct=0, nll_sum=10.0, exact=False),
        ],
    )
    split = result(
        "split",
        [
            item("long", tokens=100, correct=80, nll_sum=120.0, exact=False),
            item("short", tokens=1, correct=1, nll_sum=1.0, exact=False),
        ],
    )

    got = compare.compare_condition(
        dense,
        split,
        family="mizar",
        condition="facts_present",
        n_boot=200,
        seed=1,
    )
    metrics = got["target_metrics"]
    assert metrics["target_token_micro_accuracy"]["difference_split_minus_dense"] < 0
    assert metrics["target_example_macro_accuracy"]["difference_split_minus_dense"] > 0
    assert metrics["target_token_micro_nll_per_token"]["difference_split_minus_dense"] > 0
    assert metrics["target_example_macro_nll_per_token"]["difference_split_minus_dense"] < 0
    for endpoint in metrics.values():
        assert endpoint["paired_examples"] == 2
        assert "paired_bootstrap_ci95_low" in endpoint
        assert "paired_bootstrap_ci95_high" in endpoint


def test_comparison_refuses_different_eval_cohorts():
    dense = result(
        "dense",
        [
            item(0, tokens=2, correct=1, nll_sum=2.0, exact=True),
            item(1, tokens=2, correct=1, nll_sum=2.0, exact=False),
        ],
    )
    split = copy.deepcopy(dense)
    split["arm"] = "split"
    split["input_provenance"]["model"]["resolved_path"] = "/models/split"
    split["families"]["mizar"]["conditions"]["facts_present"]["per_example"][1]["id"] = "other"

    with pytest.raises(ValueError, match="paired IDs differ"):
        compare.compare_condition(
            dense,
            split,
            family="mizar",
            condition="facts_present",
            n_boot=10,
            seed=1,
        )


def test_comparison_rejects_duplicate_ids_before_mapping():
    dense = result(
        "dense",
        [item("same", tokens=2, correct=1, nll_sum=2.0, exact=True)],
    )
    split = result(
        "split",
        [item("same", tokens=2, correct=1, nll_sum=2.0, exact=True)],
    )
    dense_condition = dense["families"]["mizar"]["conditions"]["facts_present"]
    dense_condition["per_example"].append(copy.deepcopy(dense_condition["per_example"][0]))

    with pytest.raises(ValueError, match="duplicate example ID 'same'"):
        compare.compare_condition(
            dense,
            split,
            family="mizar",
            condition="facts_present",
            n_boot=10,
            seed=1,
        )


def test_comparator_requires_condition_subsets_and_pairing_across_arms():
    entries = [
        item("first", tokens=2, correct=1, nll_sum=2.0, exact=False),
        item("second", tokens=2, correct=1, nll_sum=2.0, exact=False),
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["evaluation_controls"]["conditions"].append("facts_absent")
        arm_result["families"]["mizar"]["conditions"]["facts_absent"] = copy.deepcopy(
            arm_result["families"]["mizar"]["conditions"]["facts_present"]
        )

    dense["families"]["mizar"]["conditions"]["facts_absent"]["per_example"][1]["id"] = "different"
    with pytest.raises(ValueError, match="not a subset of facts_present"):
        compare.validate_eval_compatibility(dense, split)

    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    split["families"]["mizar"]["conditions"]["facts_present"]["per_example"][1]["id"] = "different"
    with pytest.raises(ValueError, match="cohort IDs differ between arms"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_accepts_paired_absent_subset_of_present_cohort():
    entries = [
        item("first", tokens=2, correct=1, nll_sum=2.0, exact=False),
        item("second", tokens=2, correct=1, nll_sum=2.0, exact=False),
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["evaluation_controls"]["conditions"].append("facts_absent")
        absent = result(arm_result["arm"], [copy.deepcopy(entries[0])])["families"]["mizar"][
            "conditions"
        ]["facts_present"]
        absent["source_examples"] = len(entries)
        absent["context_eligible_examples"] = len(entries)
        arm_result["families"]["mizar"]["conditions"]["facts_absent"] = absent

    compare.validate_eval_compatibility(dense, split)
    got = compare.compare_condition(
        dense,
        split,
        family="mizar",
        condition="facts_absent",
        n_boot=10,
        seed=1,
    )
    assert got["paired_examples"] == 1


def test_comparator_accepts_sampled_present_cohort():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(4)
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        family = arm_result["families"]["mizar"]
        family["source_examples"] = 5
        family["context_eligible_examples"] = 5
        family["evaluated_examples"] = 5
        present = family["conditions"]["facts_present"]
        present["source_examples"] = 5
        present["context_eligible_examples"] = 5

    compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_impossible_family_denominator_ordering():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        family = arm_result["families"]["mizar"]
        family["source_examples"] = 0
        family["conditions"]["facts_present"]["source_examples"] = 0

    with pytest.raises(
        ValueError,
        match=r"0 <= evaluated_examples <= context_eligible_examples <= source_examples",
    ):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_attempted_examples", 2, "generation_attempted_examples"),
        ("whole_proof_budget_ineligible_examples", 1, "budget eligible\\+ineligible"),
        ("exact_match_count_budget_eligible", 2, "exact eligible"),
    ],
)
def test_comparator_rejects_impossible_condition_denominators(field, value, message):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=True)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["families"]["mizar"]["conditions"]["facts_present"][field] = value

    with pytest.raises(ValueError, match=message):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_requires_budget_eligible_items_to_be_attempted():
    entries = [
        item(
            "one",
            tokens=2,
            correct=1,
            nll_sum=2.0,
            exact=False,
            budget_eligible=True,
            generation_attempted=False,
        )
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))

    with pytest.raises(ValueError, match="budget eligible.*generation_attempted"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("checkpoint_step",), 0, "positive integer"),
        (("base_model_revision",), "", "base_model_revision"),
        (("initial_weights_sha256",), "", "initial_weights_sha256"),
        (("export_metadata_schema",), None, "export_metadata_schema"),
    ],
)
def test_comparator_rejects_equally_unknown_model_provenance(path, replacement, message):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        model = arm_result["input_provenance"]["model"]
        model[path[0]] = replacement

    with pytest.raises(ValueError, match=message):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("target", "replacement"),
    [
        ("model", "split"),
        ("export_metadata", "split"),
    ],
)
def test_comparator_cross_checks_result_and_exported_model_arm(target, replacement):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    model = dense["input_provenance"]["model"]
    if target == "model":
        model["arm"] = replacement
    else:
        model["export_metadata"]["arm"] = replacement

    with pytest.raises(ValueError, match="dense.*arm"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    "tamper",
    ["model_root", "metadata_root", "model_inventory", "metadata_inventory"],
)
def test_comparator_requires_each_trained_weight_root_and_inventory_to_be_consistent(tamper):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    model = dense["input_provenance"]["model"]
    if tamper == "model_root":
        model["trained_weights_root_sha256"] = "0" * 64
    elif tamper == "metadata_root":
        model["export_metadata"]["trained_weights_root_sha256"] = "0" * 64
    elif tamper == "model_inventory":
        model["trained_weight_files"]["model.safetensors"]["sha256"] = SHA_D
    else:
        model["export_metadata"]["trained_weight_files"]["model.safetensors"]["sha256"] = SHA_D

    with pytest.raises(ValueError, match="trained weight"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_reports_missing_trained_weight_root_as_schema_error():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    del dense["input_provenance"]["model"]["trained_weights_root_sha256"]

    with pytest.raises(ValueError, match="trained_weights_root_sha256.*missing"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_requires_nonempty_matching_source_commit():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        model = arm_result["input_provenance"]["model"]
        model["source_commit"] = ""
        model["export_metadata"]["source_commit"] = ""

    with pytest.raises(ValueError, match="source_commit"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_binds_available_platform_manifest_to_export_metadata():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    dense["input_provenance"]["model"]["export_metadata"]["platform_run_manifest_sha256"] = SHA_F

    with pytest.raises(ValueError, match="platform run manifest"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_requires_sha256_algorithm_even_when_arms_agree():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["input_provenance"]["hash_algorithm"] = "SHA-256"

    with pytest.raises(ValueError, match="hash_algorithm.*sha256"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("tokenizer_sha256",), "abc"),
        (("corpus_sha256",), "g" * 64),
        (("evaluator_sha256",), "A" * 64),
        (("model", "semantic_config_sha256"), "short"),
    ],
)
def test_comparator_rejects_malformed_scalar_hashes_even_when_arms_agree(path, replacement):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        node = arm_result["input_provenance"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = replacement

    with pytest.raises(ValueError, match="lowercase 64-hex"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_malformed_future_provenance_digests():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["input_provenance"]["future"] = {
            "artifact_sha256": "Z" * 64,
        }

    with pytest.raises(ValueError, match=r"future\.artifact_sha256.*lowercase 64-hex"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("map_name", "replacement"),
    [
        ("eval_shard_sha256", "short"),
        ("heldout_manifest_sha256", "g" * 64),
        ("train_shard_sha256", "A" * 64),
    ],
)
def test_comparator_rejects_malformed_family_hash_map_values(map_name, replacement):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["input_provenance"][map_name]["mizar"] = replacement

    with pytest.raises(ValueError, match="lowercase 64-hex"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("map_name", "replacement"),
    [
        ("eval_shard_sha256", {}),
        ("eval_shard_sha256", {"mizar": SHA_D, "extra": SHA_A}),
        ("heldout_manifest_sha256", {}),
        ("heldout_manifest_sha256", {"mizar": SHA_E, "extra": SHA_A}),
        ("train_shard_sha256", {}),
        ("train_shard_sha256", {"mizar": SHA_F, "extra": SHA_A}),
    ],
)
def test_comparator_requires_exact_family_hash_map_keys(map_name, replacement):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        arm_result["input_provenance"][map_name] = replacement

    with pytest.raises(ValueError, match=rf"{map_name}.*family keys"):
        compare.validate_eval_compatibility(dense, split)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("exact_match", 1),
        ("whole_proof_budget_eligible", "yes"),
        ("generation_attempted", 1),
    ],
)
def test_comparator_requires_actual_boolean_per_example_logic(field, replacement):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    entries[0][field] = replacement
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))

    with pytest.raises(ValueError, match=rf"{field}.*bool"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_requires_exact_matches_to_have_generation_attempted():
    entries = [
        item(
            "one",
            tokens=2,
            correct=1,
            nll_sum=2.0,
            exact=True,
            budget_eligible=False,
            generation_attempted=False,
        )
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))

    with pytest.raises(ValueError, match="exact_match implies generation_attempted"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_exact_aggregate_above_attempted_aggregate():
    entries = [
        item(
            "one",
            tokens=2,
            correct=1,
            nll_sum=2.0,
            exact=False,
            budget_eligible=False,
            generation_attempted=False,
        )
    ]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        condition = arm_result["families"]["mizar"]["conditions"]["facts_present"]
        condition["exact_match_count_evaluated"] = 1
        condition["exact_match_rate_evaluated"] = 1.0

    with pytest.raises(
        ValueError,
        match="exact_match_count_evaluated.*generation_attempted_examples",
    ):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_boolean_metamath_validity_from_incomplete_api():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    entries[0]["metamath"] = {"status": "evaluated", "valid": True}
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))

    with pytest.raises(ValueError, match="boolean Metamath validity"):
        compare.validate_eval_compatibility(dense, split)


def test_training_config_check_allows_only_arm_and_output_identity():
    dense = {
        "arm": "dense",
        "source_commit": "c" * 40,
        "platform_run_manifest_id": "manifest-dense",
        "platform_run_manifest_sha256": SHA_D,
        "init_seed": 42,
        "train_module": {"arm": "dense", "optim": {"lr": 2e-5}},
        "trainer": {
            "save_folder": "s3://dense",
            "callbacks": {"wandb": {"name": "dense-run"}},
        },
    }
    split = {
        "arm": "split",
        "source_commit": "c" * 40,
        "platform_run_manifest_id": "manifest-split",
        "platform_run_manifest_sha256": SHA_E,
        "init_seed": 42,
        "train_module": {"arm": "split", "optim": {"lr": 2e-5}},
        "trainer": {
            "save_folder": "s3://split",
            "callbacks": {"wandb": {"name": "split-run"}},
        },
    }
    compare.validate_training_configs(dense, split)

    split["train_module"]["optim"]["lr"] = 3e-5
    with pytest.raises(ValueError, match="training configs differ"):
        compare.validate_training_configs(dense, split)


@pytest.mark.parametrize(
    ("side", "path", "replacement"),
    [
        ("dense", ("arm",), "split"),
        ("split", ("arm",), "dense"),
        ("dense", ("train_module", "arm"), "split"),
        ("split", ("train_module", "arm"), "dense"),
    ],
)
def test_training_config_check_requires_both_saved_arm_fields(side, path, replacement):
    dense = {
        "arm": "dense",
        "source_commit": "c" * 40,
        "init_seed": 42,
        "train_module": {"arm": "dense"},
    }
    split = {
        "arm": "split",
        "source_commit": "c" * 40,
        "init_seed": 42,
        "train_module": {"arm": "split"},
    }
    target = dense if side == "dense" else split
    if len(path) == 1:
        target[path[0]] = replacement
    else:
        target[path[0]][path[1]] = replacement

    with pytest.raises(ValueError, match=rf"{side} config.*arm"):
        compare.validate_training_configs(dense, split)


def test_training_config_check_distinguishes_missing_from_explicit_null():
    dense = {
        "arm": "dense",
        "source_commit": "c" * 40,
        "init_seed": 42,
        "train_module": {"arm": "dense"},
        "optional_control": None,
    }
    split = {
        "arm": "split",
        "source_commit": "c" * 40,
        "init_seed": 42,
        "train_module": {"arm": "split"},
    }

    with pytest.raises(ValueError, match=r"optional_control=.*missing"):
        compare.validate_training_configs(dense, split)


def test_training_config_check_enforces_the_seed_42_checkpoint_contract():
    dense = {
        "arm": "dense",
        "source_commit": "c" * 40,
        "init_seed": 42,
        "train_module": {"arm": "dense"},
    }
    split = {
        "arm": "split",
        "source_commit": "c" * 40,
        "init_seed": 43,
        "train_module": {"arm": "split"},
    }

    with pytest.raises(ValueError, match="seed 42"):
        compare.validate_training_configs(dense, split)


def test_training_config_check_requires_source_commit():
    dense = {
        "arm": "dense",
        "source_commit": "",
        "init_seed": 42,
        "train_module": {"arm": "dense"},
    }
    split = {
        "arm": "split",
        "source_commit": "",
        "init_seed": 42,
        "train_module": {"arm": "split"},
    }

    with pytest.raises(ValueError, match="source_commit"):
        compare.validate_training_configs(dense, split)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("source_commit", "different-commit", "source_commit"),
        ("platform_run_manifest_id", "different-manifest", "platform run manifest"),
        ("platform_run_manifest_sha256", SHA_F, "platform run manifest"),
    ],
)
def test_comparator_cross_checks_saved_config_source_and_manifest_identity(
    field, replacement, message
):
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense_result = result("dense", copy.deepcopy(entries))
    dense_config = {
        "arm": "dense",
        "train_module": {"arm": "dense"},
        "source_commit": "c" * 40,
        "platform_run_manifest_id": "manifest-dense",
        "platform_run_manifest_sha256": SHA_D,
    }

    compare.validate_result_config_binding(
        dense_result,
        dense_config,
        expected_arm="dense",
        label="dense",
    )
    dense_config[field] = replacement
    with pytest.raises(ValueError, match=message):
        compare.validate_result_config_binding(
            dense_result,
            dense_config,
            expected_arm="dense",
            label="dense",
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("evaluation_controls", "evaluator_seed"), 7),
        (("evaluation_controls", "conditions"), ["facts_absent"]),
        (("evaluation_controls", "do_sample"), True),
        (("evaluation_controls", "temperature"), 0.9),
        (("evaluation_controls", "context_length"), 8_192),
        (("evaluation_controls", "max_new_tokens"), 4_096),
        (("evaluation_controls", "limit"), 3),
        (("evaluation_controls", "nll_chunk_size"), 128),
        (("evaluation_controls", "nll_context_policy"), "other"),
        (("evaluation_controls", "nll_target_policy"), "content_without_eos"),
        (("input_provenance", "tokenizer_sha256"), SHA_C),
        (("input_provenance", "corpus_sha256"), SHA_D),
        (("input_provenance", "eval_shard_sha256"), {"mizar": SHA_E}),
        (("input_provenance", "model", "checkpoint_step"), 22_000),
        (("input_provenance", "model", "base_model_id"), "other-model"),
        (("input_provenance", "model", "base_model_revision"), "other-revision"),
        (("input_provenance", "model", "initial_weights_sha256"), SHA_B),
        (("input_provenance", "model", "source_commit"), "different-commit"),
        (("input_provenance", "model", "semantic_config_sha256"), SHA_F),
    ],
)
def test_eval_compatibility_requires_every_control_and_provenance_match(path, replacement):
    entries = [item(0, tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    node = split
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement

    with pytest.raises(ValueError, match="differs"):
        compare.validate_eval_compatibility(dense, split)


def test_eval_compatibility_distinguishes_missing_from_explicit_null_limit():
    entries = [item(0, tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    compare.validate_eval_compatibility(dense, split)
    del split["evaluation_controls"]["limit"]

    with pytest.raises(ValueError, match=r"limit.*missing"):
        compare.validate_eval_compatibility(dense, split)


def test_comparison_refuses_old_or_incomplete_result_schema():
    entries = [item(0, tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    del dense["families"]["mizar"]["conditions"]["facts_present"]["per_example"][0]["nll_sum"]

    with pytest.raises(ValueError, match=r"nll_sum.*missing"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_has_no_equivalence_or_non_inferiority_verdict():
    assert not hasattr(compare, "verdict")
    scope = compare.INFERENCE_SCOPE.lower()
    assert "two seed-42 checkpoints" in scope
    assert "no equivalence" in scope
    assert "no non-inferiority" in scope
