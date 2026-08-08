# DP2-KDA strategic review — preserved evidence

Raw artifacts behind the 2026-08-01 first-principles review that recommended **stopping the
strict-beta DP2 program as scoped**. Pulled off FarmShare scratch (`/scratch/users/ericrcwu/
agent-runs/review-sigma`), which is purgeable, so these numbers were one cleanup away from
being unreproducible assertions.

## Contents

`review-sigma/` — 158 JSON, 9 Python, 18 logs, 4 TSV, 1 sbatch. The scripts and job script are
included deliberately: without them the JSONs are numbers with no provenance.

- **155 per-run records** — one per (arm, bundle, geometry, step-count). Keys include `arm`,
  `bundle_id`, `accuracy_by_length`, `beta_regime`, `allow_neg_eigval`, `param_ledger`.
- **3 pairwise aggregates** — `s4000_pairs.json`, `s4000_small_pairs.json`, `s16000_pairs.json`,
  `hard_pairs.json`. Keys are `armA|armB|length`, values `{n, mA, mB, sA, ...}`.

## The load-bearing numbers

The pre-registered contrast (DP2-strict vs **R1-P**, the capacity-matched primary comparator),
from `s4000_pairs.json`:

| length | n | DP2-strict | R1-P | diff |
|---|---|---|---|---|
| 64 | 12 | 59.782 | 56.519 | **+3.26pp** |

Against the program's own **+5pp** relevance floor (runbook §5.8.1 condition 3), this is a
**FAIL** — real and significant, but 1.5-4x below the threshold that was pre-registered as
"worth it".

**The decisive finding is the sign flip.** At the smaller geometry (~505K non-embedding params,
`s4000_small_pairs.json`) the same contrast is **-1.92pp at L64 and -3.58pp at L40**. Same code,
same seeds, opposite sign. The setting that selects between these is the difficulty grid, which
runbook §5.5 requires be enumerated and hashed before the first calibration job and **which does
not exist**. Independently reproduced: see the `## Verification` note below.

Also here: `R1-P|Reflection|64` = 56.519 vs **95.864**, i.e. reflection beta (in (0,2)) beats
every strict arm by ~39pp at n=12. That gap, not the strict-vs-strict contrast, is where the
measured signal lives.

## Verification status

Re-derived independently from these files, stratified by geometry and step count: small-geometry
**-3.58pp (L40) / -1.92pp (L64)** matches the review to the decimal. The large-geometry figure
differs from the review's +3.26pp depending on how `R1-P`'s three distinct `ffn_dim` values
(174 / 177 / 1343 -- 1343 is *not* param-matched) are filtered; pooling across geometries or
step counts inflates it badly (a naive group-by-arm gives +16.4pp). **Stratify before
differencing.**

Caveat on provenance: these runs predate `04a25e7`, so some records carry
`probe_source_revision: "unknown"`. Treat them as review evidence, not as manifest-grade
pre-registered results -- no pre-registration artifact ever existed for this program.
