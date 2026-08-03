"""Paired comparisons over the evaluator's family-keyed output."""

import pytest

from . import load_project_module


compare = load_project_module("compare_arms")


def result(arm, outcomes, nll):
    return {
        "arm": arm,
        "greedy": True,
        "context_length": 16_384,
        "max_new_tokens": 8_192,
        "families": {
            "mizar": {
                "conditions": {
                    "facts_present": {
                        "target_nll_per_token": nll,
                        "per_example": [
                            {
                                "id": str(i),
                                "exact_match": outcome,
                                "target_token_accuracy": float(outcome),
                            }
                            for i, outcome in enumerate(outcomes)
                        ],
                    }
                }
            }
        },
    }


def test_family_condition_comparison_is_paired_and_reports_nll():
    dense = result("dense", [True, False, True], 1.5)
    split = result("split", [True, True, False], 1.25)

    compare.validate_eval_compatibility(dense, split)
    got = compare.compare_condition(
        dense,
        split,
        family="mizar",
        condition="facts_present",
        metric="exact_match",
        n_boot=100,
        seed=1,
    )

    assert got["n"] == 3
    assert got["dense_only_wins"] == 1
    assert got["split_only_wins"] == 1
    assert got["dense_nll"] == 1.5
    assert got["split_nll"] == 1.25
    assert got["nll_difference"] == pytest.approx(-0.25)


def test_token_match_compares_paired_per_example_accuracy():
    dense = result("dense", [True, False], 1.5)
    split = result("split", [True, False], 1.4)
    dense_items = dense["families"]["mizar"]["conditions"]["facts_present"][
        "per_example"
    ]
    split_items = split["families"]["mizar"]["conditions"]["facts_present"][
        "per_example"
    ]
    dense_items[0]["target_token_accuracy"] = 0.25
    dense_items[1]["target_token_accuracy"] = 0.75
    split_items[0]["target_token_accuracy"] = 0.50
    split_items[1]["target_token_accuracy"] = 1.00

    got = compare.compare_condition(
        dense,
        split,
        family="mizar",
        condition="facts_present",
        metric="token_match",
        n_boot=100,
        seed=1,
    )
    assert got["dense_rate"] == pytest.approx(0.5)
    assert got["split_rate"] == pytest.approx(0.75)
    assert got["difference"] == pytest.approx(0.25)
    assert got["mcnemar_p"] is None


def test_comparison_refuses_different_eval_cohorts():
    dense = result("dense", [True, False], 1.5)
    split = result("split", [True, False], 1.4)
    split["families"]["mizar"]["conditions"]["facts_present"]["per_example"][1][
        "id"
    ] = "other"

    with pytest.raises(ValueError, match="paired IDs differ"):
        compare.compare_condition(
            dense,
            split,
            family="mizar",
            condition="facts_present",
            metric="exact_match",
            n_boot=10,
            seed=1,
        )


def test_training_config_check_allows_only_arm_and_output_identity():
    dense = {
        "train_module": {"arm": "dense", "optim": {"lr": 2e-5}},
        "trainer": {
            "save_folder": "s3://dense",
            "callbacks": {"wandb": {"name": "dense-run"}},
        },
    }
    split = {
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
