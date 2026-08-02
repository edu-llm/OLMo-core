"""Build the real training corpus from set.mm: train / eval / held-out-fact partition.

`build_metamath_sample.py` produces a format demonstration and says so — it caps at
--limit, keeps only 3-10 step proofs, and ships no held-out split. This is the
production job: all of set.mm, a real partition, and a held-out fact manifest that
`src/test/scripts/p3_math_split/corpus_invariants_test.py` can actually check against.

Three differences from the sample that matter:

  no alphabetical prefix bias
      The sample does `sorted(labels)` then breaks at --limit, so a 500-example shard
      is the alphabetically-first 500 theorems. Here every theorem is expanded and the
      *split* is seeded-random, so train and eval are drawn from the same distribution.

  wider step band
      3-10 steps was chosen to keep the demo small. That filter is what pushes the
      sample's masked fraction to ~46%: short proofs mean the fact block dominates the
      text. Allowing longer proofs brings it into the 17-30% design band on its own.

  held-out facts are held out on both leak paths
      A fact leaks if a training example cites it, and *also* if a training example is
      its proof (96.4% of set.mm facts are proved in-corpus, so the goal line is a
      second door). Both are closed here, and I2 in the invariant suite re-checks it.

Usage:
    python src/scripts/train/p3_math_split/build_corpus.py --db data/set.mm --out corpus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from mm_expand import MM, expand  # noqa: E402
except ImportError:  # pragma: no cover
    sys.exit(
        "src/scripts/train/p3_math_split/mm_expand.py not found.\n"
        "Run the fetch step in README.md once first — it writes the Metamath toolchain and\n"
        "downloads data/set.mm. This script deliberately reuses that expander rather\n"
        "than reimplementing the proof stack machine."
    )

HDR = "I know these mathematical statements:"
SEP = "---"


def render_fact(label, kind, data):
    """`hypotheses => conclusion`, so inference rules are self-contained.

    Same rendering as the sample builder, and it must stay the same: the eval prompts
    are built from this, so a divergence would silently change what the model sees at
    test time relative to training.
    """
    concl = " ".join(data[0])
    hyps = [" ".join(h[2]) for h in (data[1] if len(data) > 1 else []) if h[0] == "$e"]
    return f"{' & '.join(hyps)} => {concl}" if hyps else concl


def render_example(facts, goal, target):
    block = HDR + "\n" + "\n".join(f"{n} : {s}" for n, s in facts.items())
    return f"{block}\n{SEP}\nGOAL {goal}\n{target}", len(block)


def extract(mm, args):
    """Expand every $p theorem into a candidate example. Returns rows + a reject tally."""
    logical = {
        lb
        for lb, (k, d) in mm.labels.items()
        if k in ("$a", "$p") and d and d[0] and d[0][0] == "|-"
    }
    provable = sorted(lb for lb, (k, _) in mm.labels.items() if k == "$p")

    if args.max_theorems:
        # Sample across the whole database rather than truncating. `sorted()` above puts
        # set.mm in alphabetical order, so taking a prefix would give an
        # alphabetically-biased slice of mathematics -- the exact bug the sample builder
        # has. Only for smoke runs; a real corpus uses all of it.
        provable = sorted(
            random.Random(args.seed).sample(provable, min(args.max_theorems, len(provable)))
        )
        print(f"  --max-theorems: sampled {len(provable):,} theorems (SMOKE RUN, not a corpus)")

    rows: list = []
    tally: Counter = Counter()
    for lbl in provable:
        try:
            expr, _mand, refs, trace = expand(mm, lbl)
        except Exception:
            tally["expand_failed"] += 1
            continue

        steps = [(lb, " ".join(e)) for (lb, e, _) in trace if e and e[0] == "|-"]
        if not (args.min_steps <= len(steps) <= args.max_steps):
            tally["step_count_out_of_band"] += 1
            continue

        # The soundness gate: the proof must reduce to the statement it claims.
        if steps[-1][1] != " ".join(expr):
            tally["failed_to_reduce"] += 1
            continue

        used = [r for r in dict.fromkeys(refs) if r in logical]
        if not (args.min_facts <= len(used) <= args.max_facts):
            tally["fact_count_out_of_band"] += 1
            continue

        eid = hashlib.md5(lbl.encode()).hexdigest()[:12]
        order = list(used)
        random.Random(eid).shuffle(order)  # block order must not leak step order (I10)
        facts = {r: render_fact(r, *mm.labels[r]) for r in order}

        goal = " ".join(expr)
        target = "\n".join(f"{i + 1:>3}  {lb:<12} {e}" for i, (lb, e) in enumerate(steps))
        text, block_len = render_example(facts, goal, target)

        if len(text) > args.max_chars:
            tally["too_long"] += 1
            continue

        rows.append(
            {
                "id": eid,
                "theorem": lbl,
                "facts": facts,
                "cited": used,
                "goal": goal,
                "target": target,
                "text": text,
                "mask_start": 0,
                "mask_end": block_len,
                "n_steps": len(steps),
            }
        )
        tally["kept"] += 1
    return rows, tally


def choose_heldout(rows, args):
    """Pick facts to withhold from supervision entirely — from a frequency *band*.

    Both ends of the band matter:

      too rare      withholding a fact cited twice gives an eval set of two. Not
                    measurable, and it makes the retrieval split noise.

      too common    this is the trap. In set.mm the citation distribution is brutally
                    skewed: on a 1.2k-example sample, `eqid` is cited by 12% of
                    examples and `a1i` by 5%. Holding out one workhorse rule deletes a
                    large slice of the training set AND removes the most common
                    inference patterns, so the arms end up trained on a different
                    (and much smaller) distribution than intended. That has nothing to
                    do with the loss mask, but it would show up in the result.

    Sampling is seeded and sorted-first so the manifest reproduces across machines,
    which is what I11 (shared-manifest sha256) exists to enforce.
    """
    freq = Counter(f for r in rows for f in r["cited"])
    eligible = sorted(
        f for f, c in freq.items() if args.heldout_min_freq <= c <= args.heldout_max_freq
    )
    if len(eligible) < args.n_heldout:
        raise SystemExit(
            f"only {len(eligible)} facts are cited between {args.heldout_min_freq} and "
            f"{args.heldout_max_freq} times; cannot hold out {args.n_heldout}. Lower "
            f"--n-heldout, or widen --heldout-min-freq/--heldout-max-freq (raising the "
            f"max risks withholding a workhorse rule -- check the reported eval share)."
        )
    chosen = sorted(random.Random(args.seed).sample(eligible, args.n_heldout))
    covered = sum(freq[f] for f in chosen)
    print(
        f"  held-out facts: {len(chosen)} from {len(eligible)} eligible "
        f"(cited {args.heldout_min_freq}-{args.heldout_max_freq}x); "
        f"{covered:,} citations, max single-fact freq "
        f"{max(freq[f] for f in chosen)}"
    )
    return chosen


def partition(rows, heldout, args):
    """Split into train / eval_retrieval / eval_iid with no leaks in either direction.

    eval_retrieval is the measurement that answers the research question: every example
    cites at least one fact the model was never supervised on, so producing a correct
    proof requires reading the fact out of context rather than recalling it.

    eval_iid is the control: same distribution, but every cited fact was visible during
    training. If the split arm wins on eval_retrieval and ties on eval_iid, the effect
    is about retrieval. If it wins on both, something more general is going on.
    """
    held = set(heldout)
    rng = random.Random(args.seed)

    eval_retrieval, clean = [], []
    for r in rows:
        # A training example may neither cite a held-out fact nor *be* its proof.
        if set(r["cited"]) & held:
            eval_retrieval.append(r)
        elif r["theorem"] in held:
            continue  # goal-line leak; drop entirely rather than put it in eval
        else:
            clean.append(r)

    rng.shuffle(clean)
    n_iid = min(args.n_eval_iid, len(clean) // 10)
    eval_iid, train = clean[:n_iid], clean[n_iid:]

    if args.max_eval_retrieval and len(eval_retrieval) > args.max_eval_retrieval:
        rng.shuffle(eval_retrieval)
        eval_retrieval = eval_retrieval[: args.max_eval_retrieval]

    # I7: no theorem may be proved in both train and eval, even by a different route.
    train_thms = {r["theorem"] for r in train}
    eval_retrieval = [r for r in eval_retrieval if r["theorem"] not in train_thms]
    eval_iid = [r for r in eval_iid if r["theorem"] not in train_thms]

    # Training order is fixed here, once, and both arms consume this exact file.
    # "Identical input documents and order" is a property of the artifact, not of
    # two independently-seeded data loaders that we hope agree.
    train.sort(key=lambda r: r["id"])
    rng.shuffle(train)
    return train, eval_retrieval, eval_iid


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def describe(name, rows):
    if not rows:
        print(f"  {name:<18} 0 examples")
        return
    mf = sum((r["mask_end"] - r["mask_start"]) / len(r["text"]) for r in rows) / len(rows)
    chars = sum(len(r["text"]) for r in rows)
    steps = sum(r["n_steps"] for r in rows) / len(rows)
    facts = sum(len(r["facts"]) for r in rows) / len(rows)
    print(
        f"  {name:<18} {len(rows):>7,} ex  {chars / 1e6:>6.1f} MB  "
        f"mask {mf:>5.1%}  steps {steps:>4.1f}  facts {facts:>4.1f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/set.mm")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=40,
        help="the sample's cap of 10 is what inflates its masked fraction to 46%%",
    )
    ap.add_argument("--min-facts", type=int, default=2)
    ap.add_argument("--max-facts", type=int, default=8)
    ap.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="drop examples that will not fit the training sequence length",
    )
    ap.add_argument("--n-heldout", type=int, default=500)
    ap.add_argument(
        "--heldout-min-freq",
        type=int,
        default=5,
        help="a fact cited fewer times than this gives an unmeasurably small eval",
    )
    ap.add_argument(
        "--heldout-max-freq",
        type=int,
        default=50,
        help="a fact cited more times than this is a workhorse; withholding it "
        "reshapes the training distribution",
    )
    ap.add_argument(
        "--max-eval-share",
        type=float,
        default=0.30,
        help="abort if this fraction of examples ends up citing a held-out fact",
    )
    ap.add_argument("--n-eval-iid", type=int, default=1000)
    ap.add_argument("--max-eval-retrieval", type=int, default=2000)
    ap.add_argument(
        "--max-theorems",
        type=int,
        default=None,
        help="seeded random subsample of theorems, for smoke runs only",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if not os.path.exists(args.db):
        sys.exit(f"{args.db} not found — run the fetch step in README.md to download set.mm")

    print(f"parsing {args.db}")
    mm = MM().parse(args.db)

    print("expanding proofs (this takes a few minutes over full set.mm)")
    rows, tally = extract(mm, args)
    print("  " + "  ".join(f"{k}={v:,}" for k, v in sorted(tally.items())))
    if not rows:
        sys.exit("no examples survived the filters")

    heldout = choose_heldout(rows, args)
    train, eval_retrieval, eval_iid = partition(rows, heldout, args)

    # A held-out set that swallows the corpus is the failure mode choose_heldout's
    # frequency cap is there to prevent. Check the outcome, not just the intent.
    eval_share = len(eval_retrieval) / max(len(rows), 1)
    if eval_share > args.max_eval_share:
        raise SystemExit(
            f"{eval_share:.0%} of examples cite a held-out fact (limit "
            f"{args.max_eval_share:.0%}), leaving only {len(train):,} for training. "
            f"Lower --heldout-max-freq or --n-heldout: some withheld fact is a "
            f"workhorse rule and removing it is reshaping the training distribution."
        )

    manifest = {"facts": heldout}
    manifest_sha = hashlib.sha256(json.dumps(sorted(heldout)).encode()).hexdigest()
    with open(os.path.join(args.out, "heldout.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "eval_retrieval.jsonl"), eval_retrieval)
    write_jsonl(os.path.join(args.out, "eval_iid.jsonl"), eval_iid)
    # I7 looks for <shard>_eval.jsonl beside the shard; give it the union so the
    # train/eval overlap check actually runs instead of skipping.
    write_jsonl(os.path.join(args.out, "train_eval.jsonl"), eval_retrieval + eval_iid)

    print()
    describe("train", train)
    describe("eval_retrieval", eval_retrieval)
    describe("eval_iid", eval_iid)
    print()
    print(f"  held-out facts     {len(heldout):,}")
    print(f"  HELDOUT_SHA256     {manifest_sha}")
    print()
    print("  gate the corpus before training on it:")
    print(f"    SHARD_PATH={args.out}/train.jsonl \\")
    print(f"    HELDOUT_PATH={args.out}/heldout.json \\")
    print(f"    HELDOUT_SHA256={manifest_sha} \\")
    print("      python3 -m pytest src/test/scripts/p3_math_split/corpus_invariants_test.py -q")


if __name__ == "__main__":
    main()
