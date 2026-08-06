"""Prove that sharded evaluation reproduces the unsharded result exactly.

Two independent things have to hold, and only one of them is arithmetic.

The arithmetic half is that merged aggregates equal unsharded aggregates. The
reference values here are written out longhand rather than borrowed from
``merge_shards``, so agreement means two implementations concur instead of one
implementation agreeing with itself.

The other half is the one that is easy to get wrong. ``facts_corrupted`` draws
replacement statements from a single RNG stream advanced once per row, so a
shard that iterates only its own rows lands at a different point in that stream
and silently scores different prompts. The test pins that by checking a shard's
prompts against the unsharded prompts position by position, and by confirming
that the naive slice-first implementation actually fails the same check --
otherwise the guard would pass for the wrong reason.

Runs without torch, transformers, vLLM or a GPU.

    python src/scripts/train/p3_math_split/evals/shard_merge_test.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge_shards import merge_results
from run_eval import materialize_condition, shard_positions

SEED = 20260801
FAMILY = "metamath"


def _rows(count: int) -> list[dict]:
    return [
        {
            "id": f"ex{index:04d}",
            "theorem": f"set:thm{index}",
            "goal": f"|- goal {index}",
            "cited": [f"fact{index}a"],
            "facts": {
                f"fact{index}a": f"|- statement {index} alpha",
                f"fact{index}b": f"|- statement {index} beta",
            },
            "target": f"proof {index}",
        }
        for index in range(count)
    ]


def _corrupt_pool(rows) -> list[str]:
    return sorted({value for row in rows for value in row["facts"].values()})


# --------------------------------------------------------------------------
# Cohort slicing
# --------------------------------------------------------------------------


def test_shard_positions_partition():
    for total in (0, 1, 2, 7, 40, 41, 494):
        for count in (1, 2, 3, 4, 8):
            slices = [
                shard_positions(total, shard_index=index, shard_count=count)
                for index in range(count)
            ]
            union = set()
            for index, positions in enumerate(slices):
                assert not (union & positions), f"shard {index} overlaps an earlier shard"
                union |= positions
            assert union == set(range(total)), (total, count)
            # No shard may be starved while another holds two more; that is what
            # keeps per-shard wall clock predictable when sizing the fleet.
            sizes = [len(positions) for positions in slices]
            assert max(sizes) - min(sizes) <= 1, sizes


def test_shard_positions_rejects_bad_arguments():
    for bad in ({"shard_index": 0, "shard_count": 0}, {"shard_index": 2, "shard_count": 2}):
        try:
            shard_positions(10, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# --------------------------------------------------------------------------
# Prompt identity under sharding
# --------------------------------------------------------------------------


def _unsharded_prompts(rows, condition):
    rng = random.Random(SEED)
    pool = _corrupt_pool(rows)
    return [materialize_condition(row, condition, rng, pool).prompt for row in rows]


def _sharded_prompts(rows, condition, *, shard_index, shard_count, advance_skipped):
    """Reproduce run_eval's shard loop, optionally with the naive bug."""
    positions = shard_positions(
        len(rows), shard_index=shard_index, shard_count=shard_count
    )
    rng = random.Random(SEED)
    pool = _corrupt_pool(rows)
    out = {}
    for index, row in enumerate(rows):
        if index not in positions and not advance_skipped:
            continue
        prompt = materialize_condition(row, condition, rng, pool).prompt
        if index in positions:
            out[index] = prompt
    return out


def test_sharded_prompts_match_unsharded():
    rows = _rows(40)
    for condition in ("facts_present", "facts_absent", "facts_corrupted"):
        expected = _unsharded_prompts(rows, condition)
        for shard_count in (2, 4):
            for shard_index in range(shard_count):
                actual = _sharded_prompts(
                    rows,
                    condition,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    advance_skipped=True,
                )
                for position, prompt in actual.items():
                    assert prompt == expected[position], (
                        f"{condition} shard {shard_index}/{shard_count} "
                        f"position {position} differs from the unsharded prompt"
                    )


def test_naive_slicing_would_corrupt_the_corrupted_condition():
    """The guard above must be load-bearing, not vacuously satisfied."""
    rows = _rows(40)
    expected = _unsharded_prompts(rows, "facts_corrupted")
    naive = _sharded_prompts(
        rows, "facts_corrupted", shard_index=1, shard_count=2, advance_skipped=False
    )
    assert any(
        naive[position] != expected[position] for position in naive
    ), "slice-first sharding was expected to desynchronize the corruption RNG"

    # Conditions that never touch the RNG are unaffected either way.
    for condition in ("facts_present", "facts_absent"):
        expected_clean = _unsharded_prompts(rows, condition)
        naive_clean = _sharded_prompts(
            rows, condition, shard_index=1, shard_count=2, advance_skipped=False
        )
        assert all(
            naive_clean[position] == expected_clean[position] for position in naive_clean
        )


# --------------------------------------------------------------------------
# Merge arithmetic
# --------------------------------------------------------------------------


def _item(index: int) -> dict:
    """A per-example record with the fields the merge recomputes from."""
    tokens = 7 + (index % 5)
    correct = index % (tokens + 1)
    nll_sum = 1.5 * tokens + 0.125 * index
    attempted = index % 9 != 0
    eligible = attempted and index % 3 != 0
    return {
        "id": f"ex{index:04d}",
        "theorem": f"set:thm{index}",
        "cited": [f"fact{index}a"],
        "nll_sum": nll_sum,
        "target_tokens": tokens,
        "target_correct": correct,
        "target_nll_per_token": nll_sum / tokens,
        "target_token_accuracy": correct / tokens,
        "exact_match": attempted and index % 7 == 0,
        "whole_proof_budget_eligible": eligible,
        "generation_attempted": attempted,
        "generation_budget": 8192 if attempted else 0,
        "metamath": {
            "status": ("valid", "invalid", "unknown", "excluded")[index % 4],
            "verifier_schema_version": "p3-metamath-tristate-v1",
            "target_label": f"thm{index}",
            "source_database": "set",
            "reason_code": "",
            "reason": "",
        },
    }


def _reference_condition(items, *, source_examples, context_eligible):
    """Longhand expected aggregates, deliberately not sharing merge's code."""
    n = len(items)
    tokens = sum(item["target_tokens"] for item in items)
    correct = sum(item["target_correct"] for item in items)
    nll_sum = sum(item["nll_sum"] for item in items)
    attempted = sum(1 for item in items if item["generation_attempted"])
    eligible = [item for item in items if item["whole_proof_budget_eligible"]]
    exact_all = sum(1 for item in items if item["exact_match"])
    exact_eligible = sum(1 for item in eligible if item["exact_match"])
    return {
        "target_nll_sum": nll_sum,
        "target_tokens": tokens,
        "target_correct": correct,
        "target_token_micro_nll_per_token": nll_sum / tokens,
        "target_token_micro_accuracy": correct / tokens,
        "target_example_macro_nll_per_token": sum(
            item["nll_sum"] / item["target_tokens"] for item in items
        )
        / n,
        "target_example_macro_accuracy": sum(
            item["target_correct"] / item["target_tokens"] for item in items
        )
        / n,
        "source_examples": source_examples,
        "context_eligible_examples": context_eligible,
        "evaluated_examples": n,
        "generation_attempted_examples": attempted,
        "whole_proof_budget_eligible_examples": len(eligible),
        "whole_proof_budget_ineligible_examples": n - len(eligible),
        "whole_proof_budget_coverage_evaluated": len(eligible) / n,
        "exact_match_count_evaluated": exact_all,
        "exact_match_rate_evaluated": exact_all / n,
        "exact_match_count_budget_eligible": exact_eligible,
        "exact_match_rate_budget_eligible": exact_eligible / len(eligible),
    }


AVAILABILITY = {
    "status": "available",
    "required_schema": "p3-metamath-tristate-v1",
    "detected_schema": "p3-metamath-tristate-v1",
    "reason": None,
    "mm_dir_supplied": True,
    "metamath_sources_verified": True,
    "loaded_source_databases": ["iset", "nf", "set"],
}


def _shard_result(items, *, shard_index, shard_count, source_examples, context_eligible):
    condition = {
        "nll_context_length": 16_384,
        "nll_chunk_size": 256,
        "nll_context_policy": "bounded_sliding_window_preserve_predecessor",
        "nll_target_policy": "combined_prompt_target_suffix_plus_single_eos",
        "nll_sliding_window_examples": shard_index,
        "max_prompt_plus_target_plus_eos_tokens": 1_000 + shard_index,
        "source_examples": source_examples,
        "context_eligible_examples": context_eligible,
        "per_example": items,
        "metamath_verification": {
            "availability": AVAILABILITY,
            "condition_supported": True,
            "condition_reason": None,
        },
    }
    result = {
        "schema_version": "p3-eval-v9",
        "evaluation_controls": {"evaluator_seed": SEED, "limit": None},
        "input_provenance": {"hash_algorithm": "sha256"},
        "arm": "dense",
        "families": {
            FAMILY: {
                "source_examples": source_examples,
                "context_eligible_examples": context_eligible,
                "evaluated_examples": context_eligible,
                "excluded_over_context_examples": 0,
                "excluded_over_context_items": [],
                "heldout_manifest": "metamath",
                "conditions": {"facts_present": condition},
            }
        },
    }
    if shard_count > 1:
        result["shard"] = {"index": shard_index, "count": shard_count}
    return result


def test_merged_aggregates_match_unsharded():
    total = 41  # deliberately not divisible by any shard count under test
    items = [_item(index) for index in range(total)]
    expected = _reference_condition(items, source_examples=494, context_eligible=total)

    for shard_count in (2, 3, 4):
        shards = [
            _shard_result(
                items[shard_index::shard_count],
                shard_index=shard_index,
                shard_count=shard_count,
                source_examples=494,
                context_eligible=total,
            )
            for shard_index in range(shard_count)
        ]
        merged = merge_results(shards)
        condition = merged["families"][FAMILY]["conditions"]["facts_present"]

        assert "shard" not in merged, "merged output must not advertise a shard marker"
        assert [item["id"] for item in condition["per_example"]] == [
            item["id"] for item in items
        ], f"{shard_count} shards: merged per-example order differs from the cohort order"

        for key, want in expected.items():
            got = condition[key]
            if isinstance(want, float):
                assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-15), (
                    f"{shard_count} shards: {key} merged to {got!r}, expected {want!r}"
                )
            else:
                assert got == want, (
                    f"{shard_count} shards: {key} merged to {got!r}, expected {want!r}"
                )

        assert condition["nll_sliding_window_examples"] == sum(range(shard_count))
        assert condition["max_prompt_plus_target_plus_eos_tokens"] == 1_000 + shard_count - 1

        verification = condition["metamath_verification"]
        assert verification["valid_count"] == sum(
            1 for item in items if item["metamath"]["status"] == "valid"
        )
        assert verification["invalid_count"] == sum(
            1 for item in items if item["metamath"]["status"] == "invalid"
        )
        assert verification["decided_count"] == (
            verification["valid_count"] + verification["invalid_count"]
        )


def test_merge_rejects_incomplete_and_malformed_input():
    total = 12
    items = [_item(index) for index in range(total)]

    def shards(count):
        return [
            _shard_result(
                items[index::count],
                shard_index=index,
                shard_count=count,
                source_examples=494,
                context_eligible=total,
            )
            for index in range(count)
        ]

    cases = {
        "missing shard": shards(3)[:2],
        "duplicated shard": [shards(3)[0], shards(3)[0], shards(3)[2]],
        "unsharded input": [_shard_result(items, shard_index=0, shard_count=1,
                                          source_examples=494, context_eligible=total)],
    }
    for label, bad in cases.items():
        try:
            merge_results(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {label}")

    # A shard whose slice is the wrong size means the shards came from cohorts
    # that were not identical; silently merging them would fabricate a result.
    truncated = shards(2)
    truncated[1]["families"][FAMILY]["conditions"]["facts_present"]["per_example"] = truncated[1][
        "families"
    ][FAMILY]["conditions"]["facts_present"]["per_example"][:-1]
    try:
        merge_results(truncated)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a short shard")

    # Shards from different runs must not merge, however well the numbers line up.
    mismatched = shards(2)
    mismatched[1]["input_provenance"] = {"hash_algorithm": "sha256", "corpus_sha256": "ff" * 32}
    try:
        merge_results(mismatched)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for divergent input provenance")


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
