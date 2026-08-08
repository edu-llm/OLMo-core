"""Build a small Metamath shard in the target format — a concrete spec artifact.

The architecture team needs a real file to shape their loader against, not a schema
description. This produces N examples in exactly the format the four production jobs
emit, including the verification step Machine A requires.

Differences from the full Machine A job: it caps at --limit examples and skips the
held-out split, since the point is the shape rather than the corpus.

Usage:
    python scripts/build_metamath_sample.py --out /tmp/dscount/shards --limit 500
"""

import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, "scripts")
from mm_expand import MM, expand  # noqa: E402

HDR = "I know these mathematical statements:"
SEP = "---"


def render_fact(label, kind, data):
    """`name : hypotheses => conclusion`, so inference rules are self-contained.

    Printing only the conclusion makes `syl` read `|- ( ph -> ch )`, which says
    nothing — 57.0% of cited Metamath facts carry $e hypotheses.
    """
    concl = " ".join(data[0])
    hyps = [" ".join(h[2]) for h in (data[1] if len(data) > 1 else [])
            if h[0] == "$e"]
    return f"{' & '.join(hyps)} => {concl}" if hyps else concl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/dscount/mm/set.mm")
    ap.add_argument("--out", default="/tmp/dscount/shards")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    mm = MM().parse(a.db)
    logical = {l for l, (k, d) in mm.labels.items()
               if k in ("$a", "$p") and d and d[0] and d[0][0] == "|-"}
    prov = sorted(l for l, (k, _) in mm.labels.items() if k == "$p")

    kept = verified = failed = 0
    rows = []
    for lbl in prov:
        if kept >= a.limit:
            break
        try:
            expr, mand, refs, trace = expand(mm, lbl)
        except Exception:
            continue
        steps = [(l, " ".join(e)) for (l, e, _) in trace if e and e[0] == "|-"]
        if not (3 <= len(steps) <= 10):
            continue

        # Machine A's required check: the proof must reduce to its own statement
        if steps[-1][1] != " ".join(expr):
            failed += 1
            continue
        verified += 1

        used = [r for r in dict.fromkeys(refs) if r in logical]
        if not (2 <= len(used) <= 6):
            continue

        eid = hashlib.md5(lbl.encode()).hexdigest()[:12]
        order = list(used)
        random.Random(eid).shuffle(order)          # block order must not leak step order
        facts = {r: render_fact(r, *mm.labels[r]) for r in order}

        goal = " ".join(expr)
        target = "\n".join(f"{i+1:>3}  {l:<12} {e}"
                           for i, (l, e) in enumerate(steps))
        block = HDR + "\n" + "\n".join(f"{n} : {s}" for n, s in facts.items())
        text = f"{block}\n{SEP}\nGOAL {goal}\n{target}"

        rows.append({"id": eid, "theorem": lbl, "facts": facts, "cited": used,
                     "goal": goal, "target": target, "text": text,
                     "mask_start": 0, "mask_end": len(block)})
        kept += 1

    path = os.path.join(a.out, "metamath_sample.jsonl")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    b = sum(len(r["text"].encode()) for r in rows)
    mf = sum((r["mask_end"] - r["mask_start"]) / len(r["text"]) for r in rows)
    print(f"wrote {path}")
    print(f"  examples        {len(rows):,}")
    print(f"  verified        {verified:,}   failed to reduce {failed:,}")
    print(f"  text bytes      {b/1e6:.2f} MB   file {os.path.getsize(path)/1e6:.2f} MB")
    print(f"  facts/example   {sum(len(r['facts']) for r in rows)/len(rows):.2f}")
    print(f"  masked fraction {mf/len(rows):.1%}")


if __name__ == "__main__":
    main()
