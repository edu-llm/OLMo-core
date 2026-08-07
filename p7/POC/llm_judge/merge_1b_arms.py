#!/usr/bin/env python
"""Merge the already-generated 1B arms into the 7B results, to make ONE judge batch.

Rubric scores are only comparable inside a judge batch. ``check_scale_anchor.py`` explains
why and ``impl3x5_klw/EVAL_HANDOFF.md`` §5 measures the damage: across the n=16 and n=100
batches the two shared arms moved in *opposite* directions (D4 −0.010, A1 +0.033), so no
single offset maps one batch onto the other. Reading a fresh 7B summary against the committed
1B table is therefore not a weak comparison, it is an unavailable one.

The fix is to judge them together, and here it is nearly free: the 1B family was generated on
**this same 300-item problem set**, in the same condition (+SI), with the same greedy decode.
So the 1B text already exists and merging costs no GPU — only a wider grouping.

    python merge_1b_arms.py --sevenb arms_multiturn_7b.jsonl --out arms_multiturn_14.jsonl

**What the merged batch can and cannot say.** The seven 7B arms are a controlled family: one
dataset, one seed, one hyperparameter set, differing only in the loss reweighting. The 1B
arms are *not* a scaled-down copy of them — they differ in base model generation (OLMo-2 vs
Olmo-3), LoRA capacity (r=16/α=32 vs r=32/α=64) and peak LR (2e-4 vs 1e-4) as well as in
parameter count. So an absolute 1B-vs-7B gap confounds four things at once and is not a
scaling result. What does survive is the **shape** of each family — whether the ordering
across reweighting strengths reproduces at the other scale — and the ``SFT``/``SFT_seed2``
pair, which is the only seed replicate anywhere in either family and therefore the only
estimate of how large a gap has to be before it means anything.

``B_raw_SI`` is deliberately **not** merged. Both files carry that key and they are different
models — Olmo-3-7B base in one, OLMo-2-1B base in the other. Silently unioning them would put
two models under one name. The 7B one is kept, because it anchors the scale for the arms the
run is actually about and it is the baseline ``judge_pedagogy.py`` auto-detects. Pass
``--with-1b-base`` to bring the 1B base in as ``B_raw_SI_1b``: that buys a within-scale
normalisation (each family's lift over *its own* base), which partly defuses the recipe
confound above, at the cost of one more candidate in every judge call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 1B arm key -> key in the merged file. The six that have a 7B counterpart, plus the seed
#: replicate. ``ssd*`` arms are Impl 5 / Impl 3x5 work on a different data mix, so they are
#: not part of this family and are left out.
MERGE_1B = {
    "SFT": "SFT_1b",
    "SFT_seed2": "SFT_seed2_1b",
    "density_A_T8": "density_A_T8_1b",
    "density_B_T0.5": "density_B_T0.5_1b",
    "density_B_T1": "density_B_T1_1b",
    "density_B_T2": "density_B_T2_1b",
}
BASE_1B_KEY = "B_raw_SI"
BASE_1B_AS = "B_raw_SI_1b"


def load(path: Path) -> dict[tuple, dict]:
    rows = [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]
    keyed = {(r.get("dialogue_id"), r.get("turn")): r for r in rows}
    if len(keyed) != len(rows):
        raise SystemExit(f"{path}: {len(rows)} rows collapse to {len(keyed)} "
                         "(dialogue_id, turn) keys — duplicates would break pairing")
    return keyed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sevenb", default=str(HERE / "arms_multiturn_7b.jsonl"),
                    help="Output of run_gen7b.py.")
    ap.add_argument("--oneb", default=str(HERE / "arms_multiturn.jsonl"),
                    help="The committed 1B run, already generated on this problem set.")
    ap.add_argument("--out", default=str(HERE / "arms_multiturn_14.jsonl"))
    ap.add_argument("--with-1b-base", action="store_true",
                    help=f"Also merge the 1B base cell as {BASE_1B_AS}.")
    args = ap.parse_args()

    seven, one = load(Path(args.sevenb)), load(Path(args.oneb))

    # Pairing is the whole point: an item present in one file and not the other would drop out
    # of every paired contrast silently rather than erroring.
    only7, only1 = sorted(set(seven) - set(one)), sorted(set(one) - set(seven))
    if only7 or only1:
        raise SystemExit(
            f"problem sets differ — {len(only7)} only in 7B, {len(only1)} only in 1B. "
            f"first few: {only7[:3]} / {only1[:3]}. Both must be the same 300-item set.")

    wanted = dict(MERGE_1B)
    if args.with_1b_base:
        wanted[BASE_1B_KEY] = BASE_1B_AS

    missing = [k for k in wanted if k not in next(iter(one.values()))["outputs"]]
    if missing:
        raise SystemExit(f"{args.oneb} has no arm(s) {missing}; present: "
                         f"{sorted(next(iter(one.values()))['outputs'])}")

    merged = []
    for key, r in seven.items():
        r = {**r, "outputs": dict(r["outputs"])}
        src = one[key]["outputs"]
        for old, new in wanted.items():
            if new in r["outputs"]:
                raise SystemExit(f"key collision on {new} — the 7B file already has it")
            r["outputs"][new] = src[old]
        # Provenance per arm, because the merged file mixes two generation runs on two base
        # models and the summary alone would not record which text came from where.
        r["provenance"] = {
            **{k: {"base_model": (r.get("gen_meta") or {}).get("base_model"),
                   "source": Path(args.sevenb).name}
               for k in seven[key]["outputs"]},
            **{new: {"base_model": (one[key].get("gen_meta") or {}).get("base_model"),
                     "source": Path(args.oneb).name, "orig_key": old}
               for old, new in wanted.items()},
        }
        merged.append(r)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        for r in merged:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    setups = sorted(merged[0]["outputs"])
    print(f"wrote {len(merged)} rows x {len(setups)} setups -> {out}")
    for s in setups:
        p = merged[0]["provenance"][s]
        print(f"  {s:<20} {p['base_model']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
