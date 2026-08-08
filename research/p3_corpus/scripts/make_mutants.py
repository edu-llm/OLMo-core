"""Corrupt a shard six ways, so the invariant suite can be proven to bite.

A green test run is meaningless until the test has been watched to fail. Each mutant
injects exactly one defect that a real extraction bug would produce, and the suite
must go red on every one.

Note on m3: mutate a fact that occurs in MANY examples. Altering a fact that appears
once creates no clash, the suite stays green, and it looks like a weak test when it
is really a weak mutation. That trap cost a debugging cycle during development.

Usage:
    python scripts/make_mutants.py --shard shards/mizar.jsonl --out shards/mutants
"""

import argparse
import json
import os
from collections import Counter


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--heldout", default=None,
                    help="heldout.json; defaults to a sibling of --shard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    with open(a.shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= a.rows:
                break
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) < 100:
        raise SystemExit(f"shard too small to mutate: {len(rows)} rows")

    hp = a.heldout or os.path.join(os.path.dirname(a.shard), "heldout.json")
    if os.path.exists(hp):
        held = json.load(open(hp))["facts"]
    else:
        # A format-demo shard carries no held-out split. Mint a stub of SYNTHETIC
        # names that appear nowhere in the shard: the real shard must still pass
        # against this manifest, while m2 and m6 inject the probes and so fail I2.
        # Drawing the stub from real shard facts would make the real shard fail —
        # a false alarm on good data.
        held = ["__HELDOUT_PROBE_1__", "__HELDOUT_PROBE_2__"]
        json.dump({"facts": held, "seed": None,
                   "policy": "STUB from make_mutants — synthetic probes, not a real "
                             "held-out set"},
                  open(hp, "w"), indent=1)
        print(f"no held-out set at {hp}; wrote a synthetic 2-probe stub")

    freq = Counter(n for r in rows for n in r["facts"])
    common, ncommon = freq.most_common(1)[0]

    def clone():
        return [dict(r) for r in rows]

    written = 0
    skipped = []

    # m1 — a fact with a name but no statement
    m = clone()
    m[10] = dict(m[10]); m[10]["facts"] = dict(m[10]["facts"])
    m[10]["facts"][list(m[10]["facts"])[0]] = ""
    write(f"{a.out}/m1_empty_stmt.jsonl", m)
    written += 1

    # m2 — a training example citing a held-out fact
    if held:
        m = clone()
        m[20] = dict(m[20])
        m[20]["cited"] = list(m[20]["cited"]) + [held[0]]
        m[20]["facts"] = dict(m[20]["facts"]); m[20]["facts"][held[0]] = "leaked stmt"
        write(f"{a.out}/m2_heldout_cited.jsonl", m)
        written += 1
    else:
        skipped.append("m2_heldout_cited")

    # m3 — one name, two statements (mutate a FREQUENT fact)
    m = clone()
    hits = 0
    for i, r in enumerate(m):
        if common in r["facts"] and hits < 3:
            m[i] = dict(r); m[i]["facts"] = dict(r["facts"])
            m[i]["facts"][common] = f"variant {i}"
            hits += 1
    write(f"{a.out}/m3_name_clash.jsonl", m)
    written += 1

    # m4 — target identical to the goal
    m = clone()
    m[40] = dict(m[40]); m[40]["target"] = m[40]["goal"]
    write(f"{a.out}/m4_degenerate_target.jsonl", m)
    written += 1

    # m5 — mask truncated mid-block
    m = clone()
    m[50] = dict(m[50]); m[50]["mask_end"] = 30
    write(f"{a.out}/m5_bad_mask.jsonl", m)
    written += 1

    # m6 — the proof OF a held-out fact left in training
    if held:
        m = clone()
        m[60] = dict(m[60])
        m[60]["theorem"] = held[1] if len(held) > 1 else held[0]
        write(f"{a.out}/m6_heldout_proof.jsonl", m)
        written += 1
    else:
        skipped.append("m6_heldout_proof")

    print(f"wrote {written} mutants of {len(rows):,} rows to {a.out}")
    if skipped:
        print(
            f"  skipped {len(skipped)} heldout-dependent mutants because heldout facts "
            f"are empty: {', '.join(skipped)}"
        )
    print(f"  m3 mutates {common} ({ncommon} occurrences) — frequent enough to clash")
    print("  every mutant MUST make tests/test_corpus_invariants.py fail")


if __name__ == "__main__":
    main()
