# Phase 2 — what phase 1 established, and what to run next

Phase 1 trained 32 cells and produced one usable scientific result, one decisive negative result, and a set
of design defects that had to be found by scoring a finished grid. This file says what to do about that. It
replaces `SUBMIT.md`, `ROUND2.md` and `SCORE.md`, which were operational notes for submissions that have
now all run; `PRD.md` §16 keeps the incident history.

Token budgets below are exact. **Throughput on A100/H100 is not measured**, and every hour or dollar figure
for those shapes is an extrapolation, labelled as one.

---

## 1. What phase 1 established

### The dependent variable does not exist

`<mano>` accuracy at expression length 10 is a **constant function** at every width and every fact load:

| | |
|---|---|
| cells scored | 18 (12 count, 6 entropy; 13M and 28M) |
| accuracy range | 4.127% – 4.610% |
| best-constant floor | 4.695% |
| cells at or above floor | **0 of 18** |

Eleven of those cells scored *exactly* 1342/30000, which is the number of eval items answered `<n0>`; one
scored 1339, the count for `<n12>`. They are constant functions. `degenerate_rate` reported 0.0 for all of
them because it matched only `<n16>`, the *best* constant — and the one cell it did flag, at 99.86%, had
consequently the highest accuracy in the table.

So the count and entropy axes measured noise about a constant. **Crowding is untested, and three seeds
would not have changed that**: an endpoint with no dynamic range has none at any replication.

### The storage measurement works, at low demand

Storage and template reconstruction agree cell by cell on the count axis:

| demand (b/param) | 13M stored/prior | 28M stored/prior | reconstruction |
|---|---|---|---|
| 0.3 | 45.5% | 51.7% | 17–20× chance |
| 0.6 | 17.6% | 18.8% | 6.5–8× |
| 1.2 | 10.4% | 4.0% | 1.6–3.4× |
| 2.4 | 15.1% | 0.01% | 1.0–6.7× |
| 4.8 | 0.4% | 0% | ~1× |

Stored bits decompose additively over attribute spans, so `birth_year` — the attribute `<compare>`
supervises — contributes at most its own prior of 8.644 bits. That bounds the contamination:

- **demand 0.3: 13.0 bits (13M) and 15.9 bits (28M) of genuine multi-attribute storage** beyond
  `birth_year`;
- **demand ≥ 0.6: the entire signal is consistent with `birth_year` alone**, i.e. with `<compare>`'s
  supervision rather than biography storage.

A real one-seed descriptive result at demand 0.3, and nothing above it.

### b32 is a real, unresolved storage-like signal — an earlier draft of this file had it backwards

`28m_b32` claims 67.56 bits/entity, 35.2% of prior, three orders of magnitude above every other entropy
cell. An earlier revision called that an artefact because b32 reconstructs 0.097% of attributes while b4
reconstructs 6.2%, and read the two as anti-correlated.

**That comparison was invalid.** Those are raw rates, and the chance levels differ by eight orders of
magnitude:

| b | pool per attribute | chance | reconstruction | × chance |
|---|---|---|---|---|
| 4 | 16 | 6.25e-02 | 6.26% | **1.0×** |
| 8 | 256 | 3.91e-03 | 0.448% | 1.1× |
| 32 | 4.29e9 | 2.33e-10 | 0.056% | **2,400,000×** |

b4 is *at* chance. b32 is 2.4 million times chance. The reconstruction **corroborates** the storage
estimate rather than contradicting it. The scorer's `round(chance, 9)` stored b32's chance as literal `0.0`,
which is how the error survived; chance is no longer rounded and a
`template_<attr>_generation_over_chance` column is published beside it.

There is also a structural reason "artefact" was never a coherent story: probability mass on inactive
softmax rows can only *suppress* a CE-based storage estimate, never manufacture a positive 67.56-bit
reduction.

**Correct statement:** b32 shows a large, late-emerging teacher-forced value-information signal. It is
neither proven memorisation nor proven artefact, and it is single-seed and post-hoc. Do not discard it —
run §5.2 first.

---

## 2. Defects found by scoring, and now fixed

| defect | consequence | fix |
|---|---|---|
| `MANO_LENGTH` was a module constant | no depth sweep was expressible, so unlearnability was found by training 18 cells rather than by a 2-hour calibration | `CellSpec.mano_length`; the task fingerprint moves with it |
| `<compare>`'s answer is the birth-*year value* | `year(E) = max` of the answers over E's items. **97.1% of years recovered from 400k items; 99.7% eval accuracy with no biographies.** Not fixable by reducing mentions — at 1.6 mentions 58% still leak | `EntityTable.probe_ids_for(split)` gives train and eval **disjoint** pools. Verified: 0 of 30,000 eval items share an entity with training supervision |
| bit cohort was entities 0–24,999 | exactly `<compare>`'s probe window: `birth_year` reconstructs at 328× chance where the best other attribute is 1.1× | `--bit-offset` |
| `degenerate_rate` matched one constant | eleven collapsed cells read as non-degenerate | `modal_rate` |
| gate reports written with `Path.write_text()` | an `s3://` URI creates a *local directory named* `s3:`; the first report never existed | both directions go through `olmo_core.io` |
| admission failed open four ways | a lone passing `G1`, an invented gate, a repeated gate, and `"passed": "false"` all admitted rows | coverage checked against `GATES`; non-boolean verdicts refused |
| partial runs entered the table | `--last-only` means "highest in this prefix", which for a crashed run is a half-trained model | `select_complete` picks one **run** per cell and keeps that run's whole trajectory; `--expect-cells` counts distinct cells |
| storage fields repeated per endpoint row | averaging that column double-weights fact-bearing cells and no controls | `storage_row` |
| `save_async` opened a second process group | five of six run failures died 0–15 steps after a checkpoint | `save_async=False` |

---

## 3. What to keep, what to discard

**Keep and re-use.** The 32 trained cells' checkpoints remain valid artefacts. The `schema` fingerprint is
unchanged by every fix above, and phase-1 checkpoints recorded no `reasoning_structure`, so
`verify_fingerprints` still admits them — **storage and template reconstruction can be re-scored from the
existing checkpoints without retraining.**

**Discard as scientific results; keep as incident history.**

- Every `<mano>` number from phase 1. The endpoint was a constant function.
- Every `<compare>` number from phase 1. The task was invalid by design and has now changed, so old values
  are not comparable — even though no phase-1 model actually exploited the leak (they scored 0.5–1.0%
  against a 0.605% floor, not the ~99.7% the leak permits).
- **Not** `28m_b32`'s storage figure. An earlier revision of this file listed it for discard on a
  comparison that ignored chance levels; it is the most interesting unresolved result here.

**Nothing needs retraining to be *correct*.** It needs a working endpoint to be *meaningful*.

---

## 4. The plan

Ordered so the cheapest thing that can invalidate the next thing runs first. This is the ordering phase 1
got wrong.

### A. Mano calibration — gates everything

Reasoning-only, so each cell is 1.0B tokens rather than up to 35B, and the sweep doubles as **G1's
task-depth evidence**.

**Calibrate at the treatment architecture, which the committed configs do not.** They are 13M and 113M on
the *count* vocabulary (3,584 padded rows). The primary treatment is 28M on the *entropy union* vocabulary
(8,064 rows) — a different softmax width and a different total parameter count, 31.43M against 29.71M. An
exit rule reading "at least one row passes" could therefore be satisfied by 113M while the 28M treatment
stays unlearnable. **Calibrate 28M in the exact entropy architecture first; use 13M and 113M afterwards as
width-response evidence for G6, once a length is frozen.**

**Content-disjoint splits are now in force, and they change what short lengths mean.** The expression space
is `23**L * 2**(L-1)`: 1,058 at L2, and a 1.0B-token budget buys 125M items, so every expression appeared
about 118,000 times and 100% of the eval set was trained on verbatim. Measured from a 60,000-item sample:
100% overlap at L2, 72% at L3, and L4 is exhausted at the full stream too (37 items per expression). Those
lengths were measuring lookup, not computation.

`ManoTask` now assigns an expression to the train or eval half by hashing **its content**, redrawing when a
draw lands in the wrong half. Verified at **0.00% overlap at every length**, and the answer distribution is
undisturbed — entropy 4.52 bits of a 4.524-bit maximum, floors within sampling noise of their old values.
Short lengths are now honest tests of computation, which is what makes them usable as calibration points
rather than only as pipeline controls.

G4 requires floor-to-ceiling ≥ 10 pp. Measured floors, which shift because at short lengths a *copy* policy
beats a constant:

| L | floor | winning policy | ceiling needed |
|---|---|---|---|
| 2 | 6.220% | copy@2 | 16.2% |
| 3 | 5.310% | copy@2 | 15.3% |
| 4 | 4.610% | constant `<n21>` | 14.6% |
| 5 | 4.625% | constant `<n11>` | 14.6% |
| 6 | 4.635% | constant `<n3>` | 14.6% |
| 8 | 4.680% | constant `<n2>` | 14.7% |
| 10 | 4.695% | constant `<n16>` | 14.7% |

**Exit condition, and floor + 10 pp is not it.** That threshold is only G4's *range* requirement. The
binding constraint is G1, whose in-band lower bound is 20% of the floor-to-100 range: at a ~4.65% floor
that is **≥ 23.8% absolute accuracy**, not 14.7%. G1 additionally wants a ≥15 pp spread across depths, and
G3 a ≥15 pp ablation drop. So the rule is:

> Select the **hardest content-disjoint length** at 28M in the entropy architecture that reaches
> **≥ 23.8% absolute**, shows a **≥ 15 pp spread** across the swept depths, and has `modal_rate` bounded
> away from 1. Confirm the selected length on independent seeds before freezing it.

`modal_rate` is a diagnostic, not yet a gate input — no gate consumes it, and treating it as an exit
criterion without a pre-registered threshold would be inventing one. Pre-register the bound or drop it from
the rule.

If nothing clears 23.8%, §6 applies instead of §4.B.

Submit with `--config-dir …/configs/cells/calibration` and `fanout_size: 14`. The index maps by *filename*,
and `113m` sorts before `13m` because `"113" < "13"` as strings — so **0–6 are the 113M cells and 7–13 are
the 13M ones**, lengths ascending within each. Verified against the CLI; index 14 is refused.

### B. Entropy sweep × 3 seeds — the primary result

18 cells, **108.0B tokens**. 28M, b ∈ {0, 4, 8, 16, 24, 32}, at the calibrated length.

First because it is the **identified** axis: iso-token by construction, so entity count, token budget and
mixture are held and only entropy varies. Half the cost of the count axis at the same row, and three seeds
make it the first block in this project capable of an equivalence or non-inferiority statement —
`analysis/trend.py` refuses a single seed block, correctly, and every phase-1 treatment was `replicate: 0`.

### C. Count axis × 3 seeds — the confounded axis

18 cells, **214.6B tokens** at 28M; or reduced to demands {0, 0.6, 2.4}, 9 cells, **71.8B tokens**.

**B − C is not a decomposition, and earlier revisions of this file said it was.** The two axes differ in
schema and templates, vocabulary, active support and target frequency, entity count, total tokens and
steps, mixture ratio and optimiser history, demand range, and in `<compare>` being present only in positive
count cells. Subtracting their slopes does not isolate "tokens and steps". Treat C as **descriptive
sensitivity and external validity**, run it after the primary block, and keep subtraction language out of
the write-up.

Run the reduced form first: three levels still give a slope.

### D. Scale

64M is **469.0B tokens** for six levels × 3 seeds; 113M is **832.0B**. Neither is worth committing before
B and C show an effect to scale. The 113M rung is defined in `ladder/sizes.py` and has never been trained.

### E. Close the remaining gates

- **G2** — an untrained checkpoint. `CheckpointerCallback.pre_train_checkpoint` writes step 0; score it.
  Nearly free, and it closes a gate that has been open throughout.
- **G3** — the premise-ablated probe, which needs a corpus variant.
- **G8** — the dilution ladder re-run at the calibrated length. 5 cells, 4.25B tokens.

---

## 5. Diagnostics on existing checkpoints — no training

1. **Re-score with `--bit-offset 25000`.** An uncontaminated storage cohort, turning the `birth_year` bound
   into a direct measurement.
2. **Every b24/b32 checkpoint, not just the last**: full-softmax value CE, active-pool-renormalised CE,
   inactive/padded probability mass, and original vs name-permuted value CE. Omit `--last-only`, which
   cannot locate a transition.

   A first draft of `select_complete` would have made this impossible: it kept the single highest-stepped
   checkpoint per cell, which deduplicated a crashed run against its re-run correctly and destroyed every
   trajectory in the process. The end-to-end test asked for three steps, got one, and that is how it was
   found. The unit of choice is the run; a cell's ten checkpoints are its result, and only its duplicate
   *runs* are redundancy.
3. **`modal_rate` across all 18 phase-1 cells**, confirming the constant-collapse count directly rather
   than by inference from the 1342 coincidence.

---

## 6. If calibration fails

If no (row, length) clears floor + 10 pp, `<mano>` is not the endpoint at this scale, and the design needs a
different dependent variable before any grid is worth buying:

1. **Fixed `<compare>` as primary.** The leak is closed and the floor is 0.605%, so its achievable range is
   far wider. It is a *related*-fact composition rather than unrelated reasoning, which changes what a
   crowding result would mean — say so rather than substituting quietly.
2. **Retrieval-then-compute**: ask for an attribute of whichever entity satisfies a comparison. Answer from
   a 263-pool, floor ~0.4%, and the supervision reveals a composition rather than a value.
3. **The in-context endpoints** `PRD.md` §8.3 names, trading a memorised task for read-and-answer.

---

## 7. Cost

| block | cells | tokens |
|---|---|---|
| A calibration | 14 | 14.0B |
| B entropy × 3 | 18 | 108.0B |
| C count × 3, reduced | 9 | 71.8B |
| C′ count × 3, full | 18 | 214.6B |
| D 64M × 3 | 18 | 469.0B |
| D′ 113M × 3 | 18 | 832.0B |

**Measured throughput, 4×A10G:** 13M **886k tok/s**, 28M **484k tok/s**.

**The finding that governs hardware choice:** 8×A10G gave **1.06×** of 4×A10G at 28M. Doubling the devices
bought 6%. These sizes are communication- and launch-bound rather than FLOP-bound, so a faster card will not
return its FLOP ratio, and an A100 cluster is not automatically the right answer. What uses a big card is a
*bigger model* — an argument for the 113M rung, not for putting 28M on an H100.

At the measured 28M rate on 4×A10G:

| block | slot-hours | rate used |
|---|---|---|
| A | **13.5–18.3** | 13M measured (886k) + 113M fitted (172k)/FLOP-linear (121k) |
| B | 62 | measured, 28M |
| C reduced | 41 | measured, 28M |
| C′ full | 123 | measured, 28M |
| D 64M × 3 | ~492 | **fitted**, 265k tok/s |
| D′ 113M × 3 | ~1,341 | **fitted**, 172k tok/s |

The two measured points give tok/s ∝ P^−0.746. Because that exponent is **below 1**, larger models are
*more* efficient per parameter on this hardware — which is the quantitative form of "a big card wants a big
model", and the reason D/D′ are the rows worth putting on an A100 rather than 28M. A strictly FLOP-linear
fit would give 215k and 121k tok/s instead, so treat the D rows as a range and not a number.

**Before committing to A100/H100, run A on 4×A10G and one B cell on 8×A100 as a paired measurement.** That
is the only way to get a real scaling factor for this workload. Phase 1 has already shown once what an
unmeasured factor costs: the README's 1.9× guess for eight devices was really 1.06×, a job set its runtime
bound from it, and it was killed at 72% after 13.8 hours.

Read `cost` and `approval_class` out of `edullm check --json`. Do not take a price from this file.

---

## 8. What would make this publishable

Phase 1 supports a negative-feasibility report plus a one-seed descriptive storage curve at demand 0.3.
That is worth writing, and it is not a crowding result.

A crowding result needs, and phase 1 had none of:

1. a **pre-calibrated** endpoint with a measured ≥10 pp achievable range (§4.A);
2. at least **three complete replicates** per treatment level (§4.B, §4.C);
3. the **storage measure named for what it computes** — teacher-forced attribute-value CE reduction on a
   fixed cohort, not Allen-Zhu R(F), which would need the name term and an untaught prefix;
4. **G1–G3 closed**, which §4.E finishes.
