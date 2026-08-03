"""Compare dense and split on the evaluator's paired, family-keyed results.

The evaluator reports target-token NLL, per-example next-token match, and
whole-output outcomes for all six families. This script keeps outcomes paired by
example ID, bootstraps their difference, uses McNemar for binary outcomes, and
reports the aggregate NLL difference.

It verifies evaluator controls and cohorts. Training-control equality must come
from the saved platform configs and the arm YAML equality test; the old local
``arm_fingerprint.json``/mask-sidecar workflow is not used by platform runs.

    python src/scripts/train/p3_math_split/compare_arms.py --dense results/dense.json --split results/split.json --dense-config runs/dense/config.json --split-config runs/split/config.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

EVAL_CONTROLS = ("greedy", "context_length", "max_new_tokens")
ALLOWED_CONFIG_DIFFERENCES = {
    ("train_module", "arm"),
    ("trainer", "save_folder"),
    ("trainer", "callbacks", "wandb", "name"),
}


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
    if dense.get("train_module", {}).get("arm") != "dense":
        raise ValueError("dense config does not declare train_module.arm=dense")
    if split.get("train_module", {}).get("arm") != "split":
        raise ValueError("split config does not declare train_module.arm=split")
    d = _flatten(dense)
    s = _flatten(split)
    differences = {
        path: (d.get(path), s.get(path))
        for path in set(d) | set(s)
        if path not in ALLOWED_CONFIG_DIFFERENCES and d.get(path) != s.get(path)
    }
    if differences:
        sample = ", ".join(
            f"{'.'.join(path)}={values!r}"
            for path, values in sorted(differences.items())[:5]
        )
        raise ValueError(f"training configs differ outside the arm: {sample}")


def validate_eval_compatibility(dense, split):
    if dense.get("arm") != "dense" or split.get("arm") != "split":
        raise ValueError("results must be dense and split respectively")
    for key in EVAL_CONTROLS:
        if dense.get(key) != split.get(key):
            raise ValueError(
                f"evaluator control {key!r} differs: "
                f"dense={dense.get(key)!r}, split={split.get(key)!r}"
            )
    if set(dense.get("families", {})) != set(split.get("families", {})):
        raise ValueError("evaluated family sets differ")
    for family in dense["families"]:
        dc = set(dense["families"][family].get("conditions", {}))
        sc = set(split["families"][family].get("conditions", {}))
        if dc != sc:
            raise ValueError(f"{family}: evaluated condition sets differ")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b = dense-only wins, c = split-only wins.

    Exact binomial rather than the chi-square approximation: discordant counts here can
    be small, and the approximation is unreliable below ~25.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap(pairs, n_boot, seed):
    """CI on (split rate - dense rate), resampling examples, not outcomes."""
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(pairs[i][0] for i in idx)
        s = sum(pairs[i][1] for i in idx)
        diffs.append((s - d) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot) - 1]
    return lo, hi


def _outcome(item, metric):
    if metric == "token_match":
        return item.get("target_token_accuracy")
    if metric == "exact_match":
        return item.get("exact_match")
    if metric == "metamath_valid":
        return item.get("metamath", {}).get("valid")
    raise ValueError(metric)


def compare_condition(dense, split, *, family, condition, metric, n_boot, seed):
    dc = dense["families"][family]["conditions"].get(condition)
    sc = split["families"][family]["conditions"].get(condition)
    if dc is None or sc is None:
        return None

    d_by_id = {e["id"]: e for e in dc["per_example"]}
    s_by_id = {e["id"]: e for e in sc["per_example"]}
    if set(d_by_id) != set(s_by_id):
        raise ValueError(f"{family}/{condition}: paired IDs differ between arms")
    ids = sorted(d_by_id)
    if not ids:
        return None

    pairs = []
    for i in ids:
        d = _outcome(d_by_id[i], metric)
        s = _outcome(s_by_id[i], metric)
        if (d is None) != (s is None):
            raise ValueError(f"{family}/{condition}/{i}: metric eligibility differs")
        if d is not None:
            if metric == "token_match":
                pairs.append((float(d), float(s)))
            else:
                pairs.append((int(bool(d)), int(bool(s))))
    if not pairs:
        return None
    d_rate = sum(p[0] for p in pairs) / len(pairs)
    s_rate = sum(p[1] for p in pairs) / len(pairs)
    if metric == "token_match":
        b = c = None
        p = None
    else:
        b = sum(1 for d, s in pairs if d and not s)  # dense only
        c = sum(1 for d, s in pairs if s and not d)  # split only
        p = mcnemar_exact(b, c)
    lo, hi = paired_bootstrap(pairs, n_boot, seed)

    return {
        "family": family,
        "condition": condition,
        "metric": metric,
        "n": len(pairs),
        "dense_rate": d_rate,
        "split_rate": s_rate,
        "difference": s_rate - d_rate,
        "dense_only_wins": b,
        "split_only_wins": c,
        "mcnemar_p": p,
        "ci95_low": lo,
        "ci95_high": hi,
        "dense_nll": dc["target_nll_per_token"],
        "split_nll": sc["target_nll_per_token"],
        "nll_difference": sc["target_nll_per_token"] - dc["target_nll_per_token"],
    }


def verdict(r, alpha=0.05):
    """State the outcome in the terms the question was asked in.

    'Match or outperform' is a one-sided claim about non-inferiority, so a
    non-significant difference is a real answer here, not a failed experiment. The
    margin is the CI, and it is reported rather than hidden behind the p-value.
    """
    if r["mcnemar_p"] is None:
        if r["ci95_low"] > 0:
            return "split BETTER"
        if r["ci95_high"] < 0:
            return "split WORSE"
    elif r["mcnemar_p"] < alpha:
        return "split BETTER" if r["difference"] > 0 else "split WORSE"
    if r["ci95_low"] > -0.02:
        return "match (equivalent within 2pp)"
    return "inconclusive (CI too wide to rule out a real loss)"


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
    ap.add_argument(
        "--metric",
        default="token_match",
        choices=("token_match", "exact_match", "metamath_valid"),
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dense = json.load(open(args.dense, encoding="utf-8"))
    split = json.load(open(args.split, encoding="utf-8"))
    if args.skip_training_config_check:
        print("WARNING: training config check skipped; comparison is not reportable")
    elif not args.dense_config or not args.split_config:
        sys.exit(
            "--dense-config and --split-config are required unless "
            "--skip-training-config-check is set"
        )
    else:
        try:
            validate_training_configs(
                json.load(open(args.dense_config, encoding="utf-8")),
                json.load(open(args.split_config, encoding="utf-8")),
            )
        except ValueError as exc:
            sys.exit(str(exc))
    try:
        validate_eval_compatibility(dense, split)
    except ValueError as exc:
        sys.exit(str(exc))

    print(
        f"\nmetric: {args.metric}   "
        f"decoding: {'greedy' if dense.get('greedy') else 'sampled'}\n"
    )
    header = (
        f"{'family/condition':<34}{'dense':>8}{'split':>8}{'diff':>9}"
        f"{'NLL Δ':>10}{'95% CI':>18}{'p':>9}  verdict"
    )
    print(header)
    print("-" * len(header))

    out = []
    for family, family_result in dense["families"].items():
        if args.metric == "metamath_valid" and family != "metamath":
            continue
        for condition in family_result["conditions"]:
            r = compare_condition(
                dense,
                split,
                family=family,
                condition=condition,
                metric=args.metric,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            if r is None:
                continue
            out.append(r)
            ci = f"[{r['ci95_low']:+.1%}, {r['ci95_high']:+.1%}]"
            label = f"{family}/{condition}"
            p_display = "-" if r["mcnemar_p"] is None else f"{r['mcnemar_p']:.3g}"
            print(
                f"{label:<34}{r['dense_rate']:>7.1%}{r['split_rate']:>8.1%}"
                f"{r['difference']:>+9.1%}{r['nll_difference']:>+10.3f}"
                f"{ci:>18}{p_display:>9}  {verdict(r)}"
            )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metric": args.metric,
                    "eval_controls": {key: dense.get(key) for key in EVAL_CONTROLS},
                    "comparisons": out,
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
