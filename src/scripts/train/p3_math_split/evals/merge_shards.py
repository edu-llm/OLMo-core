"""Recombine sharded ``run_eval.py`` outputs into one unsharded result file.

Sharding exists because the generation pass is embarrassingly parallel across
examples while the evaluator is a single process. Each shard scores a strided
slice of every family/condition cohort and writes a partial result; this tool
puts them back together.

Every aggregate is recomputed from the per-example sufficient statistics rather
than averaged across shards. Averaging would be wrong for the token-micro
endpoints, whose denominators are token counts that differ per shard, and it
would hide a missing shard behind a plausible-looking number. Recomputing means
the merged file is derived from exactly the same arithmetic ``run_eval.py``
would have applied to the union of the cohorts.

The merged output carries no shard marker, so it is byte-comparable against an
unsharded run and is accepted by ``compare_arms.py`` unchanged.

    python src/scripts/train/p3_math_split/evals/merge_shards.py \
      --shards results/dense.shard*.json --out results/dense.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_eval import (
    expected_diagnostic_cohort_size,
    summarize_generation,
    summarize_metamath_validity,
)

#: Merged verbatim after asserting every shard agrees. These describe the run,
#: not the cohort, so a disagreement means the shards are not the same run.
IDENTICAL_TOP_LEVEL_KEYS = (
    "schema_version",
    "evaluation_controls",
    "input_provenance",
    "arm",
    "metamath_sources",
)
IDENTICAL_FAMILY_KEYS = (
    "source_examples",
    "context_eligible_examples",
    "evaluated_examples",
    "excluded_over_context_examples",
    "excluded_over_context_items",
    "heldout_manifest",
)
IDENTICAL_CONDITION_KEYS = (
    "nll_context_length",
    "nll_chunk_size",
    "nll_context_policy",
    "nll_target_policy",
    "source_examples",
    "context_eligible_examples",
)


def _require_equal(values, context):
    first = values[0]
    for index, value in enumerate(values[1:], 1):
        if value != first:
            raise ValueError(
                f"{context} differs between shard 0 and shard {index}: "
                f"{first!r} != {value!r}"
            )
    return first


def _expected_shard_size(total: int, shard_index: int, shard_count: int) -> int:
    """Size of the strided slice ``range(shard_index, total, shard_count)``."""
    if total <= shard_index:
        return 0
    return (total - shard_index + shard_count - 1) // shard_count


def _ordered_per_example(shard_items, shard_count):
    """Restore the original cohort order from strided shard slices.

    Shard ``i`` holds cohort positions ``i, i+n, i+2n, ...``, so cohort position
    ``p`` is item ``p // n`` of shard ``p % n``. Reconstructing that order makes
    the merged per-example list identical to the unsharded one rather than
    merely equivalent as a set, which is what lets the merge be checked by
    direct comparison.
    """
    total = sum(len(items) for items in shard_items)
    for shard_index, items in enumerate(shard_items):
        expected = _expected_shard_size(total, shard_index, shard_count)
        if len(items) != expected:
            raise ValueError(
                f"shard {shard_index} holds {len(items)} examples but a strided "
                f"split of {total} across {shard_count} shards requires {expected}; "
                "the shards are not slices of the same cohort"
            )
    return [shard_items[p % shard_count][p // shard_count] for p in range(total)]


def _expected_cohort_size(condition: str, family_result: dict) -> int:
    """Cohort size the policy requires, independent of how it was sharded.

    The strided-size check alone cannot catch a truncated final shard, because
    dropping the last example of the last shard is indistinguishable from a
    cohort that was one smaller. The family-level evaluated count is not
    sharded, so it anchors the total to something no shard can influence.
    """
    evaluated = family_result["evaluated_examples"]
    if condition == "facts_present":
        return evaluated
    return expected_diagnostic_cohort_size(evaluated)


def merge_condition(condition_shards, *, condition, family_result, shard_count, context):
    """Rebuild one family/condition result from its per-shard partials."""
    merged = {}
    for key in IDENTICAL_CONDITION_KEYS:
        merged[key] = _require_equal(
            [shard[key] for shard in condition_shards], f"{context}.{key}"
        )

    per_example = _ordered_per_example(
        [shard["per_example"] for shard in condition_shards], shard_count
    )
    expected_total = _expected_cohort_size(condition, family_result)
    if len(per_example) != expected_total:
        raise ValueError(
            f"{context}: merged cohort has {len(per_example)} examples but the "
            f"family's evaluated count requires {expected_total}; a shard is "
            "truncated or the shards came from different cohorts"
        )

    seen = set()
    for item in per_example:
        if item["id"] in seen:
            raise ValueError(f"{context}: duplicate example ID {item['id']!r} across shards")
        seen.add(item["id"])

    target_tokens = sum(item["target_tokens"] for item in per_example)
    target_correct = sum(item["target_correct"] for item in per_example)
    target_nll_sum = sum(item["nll_sum"] for item in per_example)
    n_examples = len(per_example)
    merged.update(
        {
            "target_nll_sum": target_nll_sum,
            "target_tokens": target_tokens,
            "target_correct": target_correct,
            "target_token_micro_nll_per_token": (
                target_nll_sum / target_tokens if target_tokens else None
            ),
            "target_token_micro_accuracy": (
                target_correct / target_tokens if target_tokens else None
            ),
            "target_example_macro_nll_per_token": (
                sum(item["target_nll_per_token"] for item in per_example) / n_examples
                if n_examples
                else None
            ),
            "target_example_macro_accuracy": (
                sum(item["target_token_accuracy"] for item in per_example) / n_examples
                if n_examples
                else None
            ),
            # Counts of examples, so they add; the token ceiling is a maximum.
            "nll_sliding_window_examples": sum(
                shard["nll_sliding_window_examples"] for shard in condition_shards
            ),
            "max_prompt_plus_target_plus_eos_tokens": max(
                shard["max_prompt_plus_target_plus_eos_tokens"] for shard in condition_shards
            ),
        }
    )
    merged.update(
        summarize_generation(
            per_example,
            source_examples=family_result["source_examples"],
            context_eligible_examples=family_result["context_eligible_examples"],
        )
    )
    merged["per_example"] = per_example

    if "metamath_verification" in condition_shards[0]:
        verifications = [shard["metamath_verification"] for shard in condition_shards]
        merged["metamath_verification"] = summarize_metamath_validity(
            per_example,
            availability=_require_equal(
                [item["availability"] for item in verifications],
                f"{context}.metamath_verification.availability",
            ),
            condition_supported=_require_equal(
                [item["condition_supported"] for item in verifications],
                f"{context}.metamath_verification.condition_supported",
            ),
            condition_reason=_require_equal(
                [item["condition_reason"] for item in verifications],
                f"{context}.metamath_verification.condition_reason",
            ),
        )
    return merged


def merge_results(shards: list[dict]) -> dict:
    """Merge complete strided shards of one arm into an unsharded result."""
    if not shards:
        raise ValueError("no shard results supplied")

    declared = []
    for position, shard in enumerate(shards):
        marker = shard.get("shard")
        if marker is None:
            raise ValueError(
                f"input {position} carries no shard marker; it is an unsharded "
                "result and must not be merged"
            )
        declared.append((marker["index"], marker["count"]))
    shard_count = _require_equal([count for _, count in declared], "shard.count")
    indices = sorted(index for index, _ in declared)
    if indices != list(range(shard_count)):
        missing = sorted(set(range(shard_count)) - set(indices))
        raise ValueError(
            f"expected shards 0..{shard_count - 1}; missing {missing}, got {indices}"
        )
    ordered = [None] * shard_count
    for shard, (index, _) in zip(shards, declared):
        if ordered[index] is not None:
            raise ValueError(f"shard index {index} was supplied more than once")
        ordered[index] = shard
    shards = ordered

    merged = {}
    for key in IDENTICAL_TOP_LEVEL_KEYS:
        if key not in shards[0]:
            continue
        merged[key] = _require_equal([shard.get(key) for shard in shards], key)

    family_names = _require_equal([sorted(shard["families"]) for shard in shards], "family set")
    merged["families"] = {}
    for family in family_names:
        family_shards = [shard["families"][family] for shard in shards]
        if any("probe" in shard for shard in family_shards):
            raise ValueError(
                f"{family}: --probe results are not sharded and cannot be merged; "
                "run the probe once as a separate unsharded invocation"
            )
        family_result = {
            key: _require_equal([shard[key] for shard in family_shards], f"{family}.{key}")
            for key in IDENTICAL_FAMILY_KEYS
        }
        condition_names = _require_equal(
            [sorted(shard["conditions"]) for shard in family_shards],
            f"{family} condition set",
        )
        family_result["conditions"] = {
            condition: merge_condition(
                [shard["conditions"][condition] for shard in family_shards],
                condition=condition,
                family_result=family_result,
                shard_count=shard_count,
                context=f"{family}/{condition}",
            )
            for condition in condition_names
        }
        merged["families"][family] = family_result

    # Key order matters only for byte-comparison against an unsharded run, which
    # writes families last.
    return {**{k: v for k, v in merged.items() if k != "families"}, "families": merged["families"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", nargs="+", required=True, help="every shard result JSON for one arm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.shards]
    try:
        merged = merge_results(shards)
    except (ValueError, KeyError) as error:
        raise SystemExit(f"merge failed: {error}") from error

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    examples = sum(
        len(condition["per_example"])
        for family in merged["families"].values()
        for condition in family["conditions"].values()
    )
    print(f"merged {len(shards)} shards, {examples:,} examples -> {args.out}")


if __name__ == "__main__":
    main()
