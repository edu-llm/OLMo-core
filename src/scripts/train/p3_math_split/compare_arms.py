"""Compare the two arms, and refuse to do it if the controls did not actually hold.

Two jobs, in this order:

1. Check the fingerprints. Both runs must agree on seed, tokenizer, architecture,
   sequence length, batch size, optimizer, schedule, step count, total input tokens,
   and the sha256 of the token array. They must differ on exactly one thing: the label
   mask. If anything else differs the comparison is not the experiment that was
   designed, and this script exits non-zero instead of printing a number someone will
   quote later.

2. Paired statistics. The arms are evaluated on the same examples, so the comparison is
   paired: McNemar's exact test on the discordant pairs, plus a paired bootstrap CI on
   the difference in valid-proof rate. An unpaired t-test here would throw away most of
   the power and is the usual way this kind of result gets called a wash.

    python src/scripts/train/p3_math_split/compare_arms.py \\
        --dense results/dense_retrieval.json --split results/split_retrieval.json \\
        --dense-run runs/dense --split-run runs/split
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

# Everything that must match between arms. `arm`, `label_mask_path`,
# `label_mask_sha256`, and `supervised_tokens_this_arm` are the sanctioned differences.
CONTROLLED = [
    "seed",
    "sequence_length",
    "global_batch_size_sequences",
    "global_batch_size_tokens",
    "rank_microbatch_size_tokens",
    "grad_accum_steps",
    "world_size",
    "max_steps",
    "total_input_tokens",
    "n_instances",
    "learning_rate",
    "warmup_steps",
    "weight_decay",
    "betas",
    "eps",
    "max_grad_norm",
    "lr_alpha_f",
    "tie_embeddings",
    "tokens_sha256",
]


def check_fingerprints(dense_run, split_run):
    paths = {
        a: os.path.join(p, "arm_fingerprint.json")
        for a, p in (("dense", dense_run), ("split", split_run))
    }
    for arm, p in paths.items():
        if not os.path.exists(p):
            sys.exit(
                f"{p} not found — {arm} did not finish, or was run by something "
                f"other than src/scripts/train/p3_math_split/train.py"
            )
    fp = {arm: json.load(open(p, encoding="utf-8")) for arm, p in paths.items()}

    problems = []
    for key in CONTROLLED:
        d, s = fp["dense"].get(key), fp["split"].get(key)
        if d != s:
            problems.append(f"  {key}: dense={d!r} split={s!r}")

    if fp["dense"].get("label_mask_sha256") == fp["split"].get("label_mask_sha256"):
        problems.append(
            "  label_mask_sha256 is IDENTICAL — both arms trained with the "
            "same mask, so there is no experiment here"
        )

    if problems:
        print("controls did not hold:\n" + "\n".join(problems))
        sys.exit(1)

    print("controls verified: arms differ only in the label mask")
    print(
        f"  seed {fp['dense']['seed']}  steps {fp['dense']['max_steps']:,}  "
        f"input tokens {fp['dense']['total_input_tokens']:,}  "
        f"lr {fp['dense']['learning_rate']}"
    )
    print(
        f"  supervised tokens: dense {fp['dense']['supervised_tokens_this_arm']:,}  "
        f"split {fp['split']['supervised_tokens_this_arm']:,}  "
        f"(this difference IS the manipulation)"
    )
    return fp


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


def compare_condition(dense, split, condition, metric, n_boot, seed):
    dc = dense["conditions"].get(condition)
    sc = split["conditions"].get(condition)
    if dc is None or sc is None:
        return None

    d_by_id = {e["id"]: e for e in dc["per_example"]}
    s_by_id = {e["id"]: e for e in sc["per_example"]}
    ids = sorted(set(d_by_id) & set(s_by_id))
    if len(ids) != len(d_by_id) or len(ids) != len(s_by_id):
        print(
            f"  [{condition}] WARNING: only {len(ids):,} of "
            f"{len(d_by_id):,}/{len(s_by_id):,} examples are shared; pairing on those"
        )
    if not ids:
        return None

    pairs = [(int(bool(d_by_id[i][metric])), int(bool(s_by_id[i][metric]))) for i in ids]
    d_rate = sum(p[0] for p in pairs) / len(pairs)
    s_rate = sum(p[1] for p in pairs) / len(pairs)
    b = sum(1 for d, s in pairs if d and not s)  # dense only
    c = sum(1 for d, s in pairs if s and not d)  # split only
    p = mcnemar_exact(b, c)
    lo, hi = paired_bootstrap(pairs, n_boot, seed)

    return {
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
    }


def verdict(r, alpha=0.05):
    """State the outcome in the terms the question was asked in.

    'Match or outperform' is a one-sided claim about non-inferiority, so a
    non-significant difference is a real answer here, not a failed experiment. The
    margin is the CI, and it is reported rather than hidden behind the p-value.
    """
    if r["mcnemar_p"] < alpha:
        return "split BETTER" if r["difference"] > 0 else "split WORSE"
    if r["ci95_low"] > -0.02:
        return "match (equivalent within 2pp)"
    return "inconclusive (CI too wide to rule out a real loss)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dense", required=True, help="results JSON from run_eval.py --arm dense")
    ap.add_argument("--split", required=True, help="results JSON from run_eval.py --arm split")
    ap.add_argument("--dense-run", default="runs/dense")
    ap.add_argument("--split-run", default="runs/split")
    ap.add_argument(
        "--metric",
        default="valid",
        choices=("valid", "goal_reached", "exact_match", "all_grounded"),
    )
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument(
        "--skip-fingerprint-check",
        action="store_true",
        help="compare anyway; only for debugging, never for a reported result",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.skip_fingerprint_check:
        check_fingerprints(args.dense_run, args.split_run)
    else:
        print("WARNING: fingerprint check skipped — this result is not publishable")

    dense = json.load(open(args.dense, encoding="utf-8"))
    split = json.load(open(args.split, encoding="utf-8"))
    if dense["split"] != split["split"]:
        sys.exit(f"eval splits differ: {dense['split']} vs {split['split']}")

    print(
        f"\neval split: {dense['split']}   metric: {args.metric}   "
        f"decoding: {'greedy' if dense.get('greedy') else 'sampled'}\n"
    )
    header = f"{'condition':<18}{'dense':>8}{'split':>8}{'diff':>9}{'95% CI':>18}{'p':>9}  verdict"
    print(header)
    print("-" * len(header))

    out = []
    for condition in dense["conditions"]:
        r = compare_condition(dense, split, condition, args.metric, args.n_boot, args.seed)
        if r is None:
            continue
        out.append(r)
        ci = f"[{r['ci95_low']:+.1%}, {r['ci95_high']:+.1%}]"
        print(
            f"{r['condition']:<18}{r['dense_rate']:>7.1%}{r['split_rate']:>8.1%}"
            f"{r['difference']:>+9.1%}{ci:>18}{r['mcnemar_p']:>9.3g}  {verdict(r)}"
        )

    if "probe" in dense and "probe" in split:
        print("\nfact-recall probe (state a fact given only its name)")
        for arm, res in (("dense", dense["probe"]), ("split", split["probe"])):
            t, h = res["train_facts"], res["heldout_facts"]
            print(
                f"  {arm:<6} train-visible facts {t['exact_rate']:>6.1%} (n={t['n']})   "
                f"held-out facts {h['exact_rate']:>6.1%} (n={h['n']})"
            )
        print("  dense should lead on train-visible facts; both should be near zero on")
        print("  held-out. If dense does NOT lead, the mask did not change what was stored.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"metric": args.metric, "comparisons": out}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
