"""Paired comparisons over the evaluator's family-keyed output."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from . import load_project_module

compare = load_project_module("compare_arms")
run_eval = load_project_module("run_eval")

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


ALL_CONTEXT_COHORT_POLICY = run_eval.CONDITION_COHORT_POLICY


def diagnostic_items(items, *, family, condition, seed=20260801):
    rows = [{"id": entry["id"]} for entry in items]
    selected_ids = {
        row["id"]
        for row in run_eval.rows_for_condition(
            rows,
            family=family,
            condition=condition,
            seed=seed,
        )
    }
    return [copy.deepcopy(entry) for entry in items if entry["id"] in selected_ids]


def condition_block(items, *, family="mizar", condition="facts_present", seed=20260801):
    cohort_items = (
        copy.deepcopy(items)
        if condition == "facts_present"
        else diagnostic_items(items, family=family, condition=condition, seed=seed)
    )
    target_tokens = sum(entry["target_tokens"] for entry in cohort_items)
    target_correct = sum(entry["target_correct"] for entry in cohort_items)
    target_nll_sum = sum(entry["nll_sum"] for entry in cohort_items)
    eligible = [entry for entry in cohort_items if entry["whole_proof_budget_eligible"]]
    exact = sum(bool(entry["exact_match"]) for entry in cohort_items)
    exact_eligible = sum(bool(entry["exact_match"]) for entry in eligible)
    evaluated = len(cohort_items)
    return {
        "target_nll_sum": target_nll_sum,
        "target_tokens": target_tokens,
        "target_correct": target_correct,
        "target_token_micro_nll_per_token": target_nll_sum / target_tokens,
        "target_example_macro_nll_per_token": sum(
            entry["target_nll_per_token"] for entry in cohort_items
        )
        / evaluated,
        "target_token_micro_accuracy": target_correct / target_tokens,
        "target_example_macro_accuracy": sum(
            entry["target_token_accuracy"] for entry in cohort_items
        )
        / evaluated,
        "source_examples": len(items),
        "context_eligible_examples": len(items),
        "evaluated_examples": evaluated,
        "generation_attempted_examples": sum(
            bool(entry["generation_attempted"]) for entry in cohort_items
        ),
        "whole_proof_budget_eligible_examples": len(eligible),
        "whole_proof_budget_ineligible_examples": evaluated - len(eligible),
        "whole_proof_budget_coverage_evaluated": len(eligible) / evaluated,
        "exact_match_count_evaluated": exact,
        "exact_match_rate_evaluated": exact / evaluated,
        "exact_match_count_budget_eligible": exact_eligible,
        "exact_match_rate_budget_eligible": exact_eligible / len(eligible) if eligible else None,
        "per_example": cohort_items,
    }


def result(arm, items, *, checkpoint_step=None, family="mizar", conditions=None):
    if checkpoint_step is None:
        checkpoint_step = compare.FINAL_CHECKPOINT_STEP
    if conditions is None:
        conditions = ["facts_present"]
    trained_weight_files, trained_weights_root_sha256 = trained_weight_identity(arm)
    family_items = copy.deepcopy(items)
    condition_results = {
        condition: condition_block(
            family_items,
            family=family,
            condition=condition,
        )
        for condition in conditions
    }
    return {
        "schema_version": "p3-eval-v9",
        "arm": arm,
        "evaluation_controls": {
            "evaluator_seed": 20260801,
            "conditions": conditions,
            "condition_cohort_policy": ALL_CONTEXT_COHORT_POLICY,
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
                "checkpoint_step": checkpoint_step,
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
                    "checkpoint_step": checkpoint_step,
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
            family: {
                "source_examples": len(family_items),
                "context_eligible_examples": len(family_items),
                "evaluated_examples": len(family_items),
                "conditions": condition_results,
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


def test_comparator_accepts_diagnostic_subsets_with_policy_counts_and_paired_arms():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    conditions = ["facts_present", "facts_absent", "facts_corrupted"]
    dense = result("dense", copy.deepcopy(entries), conditions=conditions)
    split = result("split", copy.deepcopy(entries), conditions=conditions)

    compare.validate_eval_compatibility(dense, split)

    present_ids = {
        entry["id"]
        for entry in dense["families"]["mizar"]["conditions"]["facts_present"]["per_example"]
    }
    absent_ids = {
        entry["id"]
        for entry in dense["families"]["mizar"]["conditions"]["facts_absent"]["per_example"]
    }
    corrupted_ids = {
        entry["id"]
        for entry in dense["families"]["mizar"]["conditions"]["facts_corrupted"]["per_example"]
    }
    assert len(present_ids) == 100
    assert len(absent_ids) == run_eval.expected_diagnostic_cohort_size(100)
    assert len(corrupted_ids) == run_eval.expected_diagnostic_cohort_size(100)
    assert absent_ids != corrupted_ids
    assert absent_ids.issubset(present_ids)
    assert corrupted_ids.issubset(present_ids)


def test_comparator_allows_different_diagnostic_ids_across_conditions():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    conditions = ["facts_present", "facts_absent", "facts_corrupted"]
    dense = result("dense", copy.deepcopy(entries), conditions=conditions)
    split = result("split", copy.deepcopy(entries), conditions=conditions)

    compare.validate_eval_compatibility(dense, split)


def _rebuild_condition_aggregates(condition):
    items = condition["per_example"]
    target_tokens = sum(entry["target_tokens"] for entry in items)
    target_correct = sum(entry["target_correct"] for entry in items)
    target_nll_sum = sum(entry["nll_sum"] for entry in items)
    eligible = [entry for entry in items if entry["whole_proof_budget_eligible"]]
    exact = sum(bool(entry["exact_match"]) for entry in items)
    exact_eligible = sum(bool(entry["exact_match"]) for entry in eligible)
    evaluated = len(items)
    condition.update(
        {
            "target_nll_sum": target_nll_sum,
            "target_tokens": target_tokens,
            "target_correct": target_correct,
            "target_token_micro_nll_per_token": (
                target_nll_sum / target_tokens if target_tokens else None
            ),
            "target_example_macro_nll_per_token": (
                sum(entry["target_nll_per_token"] for entry in items) / evaluated
                if evaluated
                else None
            ),
            "target_token_micro_accuracy": (
                target_correct / target_tokens if target_tokens else None
            ),
            "target_example_macro_accuracy": (
                sum(entry["target_token_accuracy"] for entry in items) / evaluated
                if evaluated
                else None
            ),
            "evaluated_examples": evaluated,
            "generation_attempted_examples": sum(
                bool(entry["generation_attempted"]) for entry in items
            ),
            "whole_proof_budget_eligible_examples": len(eligible),
            "whole_proof_budget_ineligible_examples": evaluated - len(eligible),
            "whole_proof_budget_coverage_evaluated": (
                len(eligible) / evaluated if evaluated else None
            ),
            "exact_match_count_evaluated": exact,
            "exact_match_rate_evaluated": exact / evaluated if evaluated else None,
            "exact_match_count_budget_eligible": exact_eligible,
            "exact_match_rate_budget_eligible": exact_eligible / len(eligible) if eligible else None,
        }
    )


def test_comparator_rejects_cross_arm_diagnostic_cohort_mismatch():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    conditions = ["facts_present", "facts_absent"]
    dense = result("dense", copy.deepcopy(entries), conditions=conditions)
    split = result("split", copy.deepcopy(entries), conditions=conditions)
    dense_absent_ids = {
        entry["id"]
        for entry in dense["families"]["mizar"]["conditions"]["facts_absent"]["per_example"]
    }
    replacement_id = next(
        entry["id"]
        for entry in dense["families"]["mizar"]["conditions"]["facts_present"]["per_example"]
        if entry["id"] not in dense_absent_ids
    )
    split["families"]["mizar"]["conditions"]["facts_absent"]["per_example"][0]["id"] = (
        replacement_id
    )

    with pytest.raises(ValueError, match="cohort IDs differ between arms"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_diagnostic_cohort_not_subset_of_present():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    dense = result("dense", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    split = result("split", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    for arm_result in (dense, split):
        absent = arm_result["families"]["mizar"]["conditions"]["facts_absent"]
        absent["per_example"][0]["id"] = "foreign"

    with pytest.raises(ValueError, match="subset of facts_present"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_wrong_size_diagnostic_cohort():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    dense = result("dense", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    split = result("split", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    absent = dense["families"]["mizar"]["conditions"]["facts_absent"]
    absent["per_example"] = absent["per_example"][:-1]
    _rebuild_condition_aggregates(absent)
    split["families"]["mizar"]["conditions"]["facts_absent"] = copy.deepcopy(absent)

    with pytest.raises(ValueError, match="policy count"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_empty_diagnostic_cohort_when_policy_requires_nonempty():
    entries = [
        item(str(index), tokens=2, correct=1, nll_sum=2.0, exact=False)
        for index in range(100)
    ]
    dense = result("dense", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    split = result("split", copy.deepcopy(entries), conditions=["facts_present", "facts_absent"])
    for arm_result in (dense, split):
        absent = arm_result["families"]["mizar"]["conditions"]["facts_absent"]
        absent["per_example"] = []
        _rebuild_condition_aggregates(absent)

    with pytest.raises(ValueError, match="policy count"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_non_final_checkpoint_step():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries), checkpoint_step=24_540)
    split = result("split", copy.deepcopy(entries), checkpoint_step=24_540)

    with pytest.raises(ValueError, match=str(compare.FINAL_CHECKPOINT_STEP)):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_partial_present_cohort():
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

    with pytest.raises(ValueError, match="full family evaluated cohort"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_sampled_present_cohort():
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

    with pytest.raises(ValueError, match="full family evaluated cohort"):
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


METAMATH_SOURCES = {
    "commit": "abc123",
    "files": {
        "set.mm": {"sha256": SHA_D},
        "iset.mm": {"sha256": SHA_E},
        "nf.mm": {"sha256": SHA_F},
    },
}


def _attach_metamath_sources(payload):
    payload["metamath_sources"] = copy.deepcopy(METAMATH_SOURCES)
    return payload


def metamath_item(
    example_id,
    *,
    tokens,
    correct,
    nll_sum,
    exact,
    metamath_status,
    budget_eligible=True,
    generation_attempted=True,
    verifier_schema_version="p3-metamath-tristate-v1",
):
    entry = item(
        example_id,
        tokens=tokens,
        correct=correct,
        nll_sum=nll_sum,
        exact=exact,
        budget_eligible=budget_eligible,
        generation_attempted=generation_attempted,
    )
    entry["metamath"] = {
        "status": metamath_status,
        "verifier_schema_version": (
            verifier_schema_version
            if metamath_status in {"valid", "invalid", "unknown"}
            else None
        ),
        "target_label": "target",
        "source_database": "set",
        "reason_code": "",
        "reason": "",
    }
    return entry


def metamath_verification_block(items, *, condition_supported=True, condition_reason=None):
    counts = {
        "valid_count": 0,
        "invalid_count": 0,
        "unknown_count": 0,
        "excluded_count": 0,
        "unavailable_count": 0,
    }
    for entry in items:
        status = entry["metamath"]["status"]
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
    decided = counts["valid_count"] + counts["invalid_count"]
    return {
        "availability": {
            "status": "available",
            "required_schema": "p3-metamath-tristate-v1",
            "detected_schema": "p3-metamath-tristate-v1",
            "reason": None,
            "mm_dir_supplied": True,
            "metamath_sources_verified": True,
            "loaded_source_databases": ["iset", "nf", "set"],
        },
        "verifier_schema_version": "p3-metamath-tristate-v1",
        "condition_supported": condition_supported,
        "condition_reason": condition_reason,
        "evaluated_count": decided + counts["unknown_count"] if condition_supported else 0,
        **counts,
        "decided_count": decided if condition_supported else 0,
        "valid_rate_decided": (
            counts["valid_count"] / decided if condition_supported and decided else None
        ),
        "valid_rate_denominator": "valid_count + invalid_count",
    }


def metamath_result(arm, items, *, condition="facts_present", condition_supported=True):
    payload = result(arm, items)
    family = payload["families"]["metamath"] = copy.deepcopy(payload["families"]["mizar"])
    del payload["families"]["mizar"]
    family["heldout_manifest"] = "metamath"
    condition_block = family["conditions"]["facts_present"]
    condition_block["per_example"] = items
    condition_block["metamath_verification"] = metamath_verification_block(
        items,
        condition_supported=condition_supported,
        condition_reason=(
            run_eval.CORRUPTED_METAMATH_REASON if not condition_supported else None
        ),
    )
    family["conditions"] = {condition: condition_block}
    payload["evaluation_controls"]["conditions"] = [condition]
    payload["input_provenance"]["eval_shard_sha256"] = {"metamath": SHA_D}
    payload["input_provenance"]["heldout_manifest_sha256"] = {"metamath": SHA_E}
    payload["input_provenance"]["train_shard_sha256"] = {"metamath": SHA_F}
    return _attach_metamath_sources(payload)


def test_comparator_rejects_old_eval_schema_version():
    entries = [item("one", tokens=2, correct=1, nll_sum=2.0, exact=False)]
    dense = result("dense", copy.deepcopy(entries))
    split = result("split", copy.deepcopy(entries))
    dense["schema_version"] = "p3-eval-v8"

    with pytest.raises(ValueError, match="p3-eval-v9"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_pairs_metamath_validity_without_counting_unknown_as_invalid():
    entries = [
        metamath_item("a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"),
        metamath_item(
            "b", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="invalid"
        ),
        metamath_item(
            "c", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="unknown"
        ),
    ]
    dense = metamath_result(
        "dense",
        copy.deepcopy(entries),
    )
    split = metamath_result(
        "split",
        [
            metamath_item(
                "a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"
            ),
            metamath_item(
                "b", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"
            ),
            metamath_item(
                "c", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="unknown"
            ),
        ],
    )
    compare.validate_eval_compatibility(dense, split)
    got = compare.compare_condition(
        dense,
        split,
        family="metamath",
        condition="facts_present",
        n_boot=100,
        seed=1,
    )
    endpoint = got["outcomes"]["metamath_validity_decided"]
    assert endpoint["paired_examples"] == 3
    assert endpoint["eligible_paired_examples"] == 2
    assert endpoint["unknown_paired_examples"] == 1
    assert endpoint["dense_estimate"] == 0.5
    assert endpoint["split_estimate"] == 1.0
    assert endpoint["difference_split_minus_dense"] == 0.5
    assert endpoint["valid_rate_denominator"] == "valid_count + invalid_count"
    assert "dense_valid_rate_decided" not in endpoint


def test_comparator_main_prints_and_writes_metamath_validity(tmp_path, monkeypatch, capsys):
    entries = [
        metamath_item("a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"),
        metamath_item(
            "b", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="invalid"
        ),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result(
        "split",
        [
            metamath_item(
                "a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"
            ),
            metamath_item(
                "b", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"
            ),
        ],
    )
    dense_path = tmp_path / "dense.json"
    split_path = tmp_path / "split.json"
    out_path = tmp_path / "comparison.json"
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_arms.py",
            "--dense",
            str(dense_path),
            "--split",
            str(split_path),
            "--skip-training-config-check",
            "--n-boot",
            "20",
            "--seed",
            "1",
            "--out",
            str(out_path),
        ],
    )
    compare.main()
    captured = capsys.readouterr()
    assert "metamath_validity_decided" in captured.out
    assert "dense 50.00%" in captured.out or "dense 50%" in captured.out
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    endpoint = payload["comparisons"][0]["outcomes"]["metamath_validity_decided"]
    assert endpoint["dense_estimate"] == 0.5
    assert endpoint["split_estimate"] == 1.0


def test_comparator_rejects_available_availability_without_runtime_gates():
    entries = [
        metamath_item("a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        verification = arm_result["families"]["metamath"]["conditions"]["facts_present"][
            "metamath_verification"
        ]
        verification["availability"]["mm_dir_supplied"] = False
        verification["availability"]["metamath_sources_verified"] = False
        verification["availability"]["loaded_source_databases"] = []

    with pytest.raises(ValueError, match="mm_dir_supplied"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_mismatched_metamath_sources_between_arms():
    entries = [
        metamath_item("a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result("split", copy.deepcopy(entries))
    split["metamath_sources"] = copy.deepcopy(METAMATH_SOURCES)
    split["metamath_sources"]["commit"] = "different"

    with pytest.raises(ValueError, match="metamath_sources"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_inconsistent_metamath_aggregate_counts():
    entries = [
        metamath_item("a", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="valid"),
        metamath_item(
            "b", tokens=2, correct=1, nll_sum=2.0, exact=False, metamath_status="invalid"
        ),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result("split", copy.deepcopy(entries))
    verification = dense["families"]["metamath"]["conditions"]["facts_present"][
        "metamath_verification"
    ]
    verification["valid_count"] = 2
    split["families"]["metamath"]["conditions"]["facts_present"]["metamath_verification"] = (
        copy.deepcopy(verification)
    )

    with pytest.raises(ValueError, match="valid_count"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_unknown_without_versioned_schema():
    entries = [
        metamath_item(
            "a",
            tokens=2,
            correct=1,
            nll_sum=2.0,
            exact=False,
            metamath_status="unknown",
            verifier_schema_version=None,
        ),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result("split", copy.deepcopy(entries))

    with pytest.raises(ValueError, match="verifier_schema_version"):
        compare.validate_eval_compatibility(dense, split)


def test_comparator_rejects_unavailable_metamath_block_with_reportable_rates():
    entries = [
        metamath_item(
            "a",
            tokens=2,
            correct=1,
            nll_sum=2.0,
            exact=False,
            metamath_status="unavailable",
            verifier_schema_version=None,
        ),
    ]
    dense = metamath_result("dense", copy.deepcopy(entries))
    split = metamath_result("split", copy.deepcopy(entries))
    for arm_result in (dense, split):
        verification = arm_result["families"]["metamath"]["conditions"]["facts_present"][
            "metamath_verification"
        ]
        verification["availability"]["status"] = "unavailable"
        verification["availability"]["reason"] = "no mm dir"
        verification["availability"]["mm_dir_supplied"] = False
        verification["availability"]["metamath_sources_verified"] = False
        verification["availability"]["loaded_source_databases"] = []
        verification["evaluated_count"] = 1
        verification["valid_count"] = 1
        del arm_result["metamath_sources"]

    with pytest.raises(ValueError, match="valid_count"):
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
