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
| best-constant floor | 4.695% as measured then; **4.348% is the analytic value** (§2) |
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

### The degenerate-baseline estimator reported a maximum, not a rate

Found by building `InContextManoTask`, whose floor came back at **11.1%** against an analytic 10.45%.
`degenerate_baseline` searched every prompt offset for a copy policy and reported the best one's rate **on
the sample that selected it**, so what it returned was a maximum over W offsets — biased upward by roughly
`2.5 * sqrt(p(1-p)/n)`. At the ~20 offsets every task had until now that is a rounding error, which is why
a flat three-standard-error bar sufficed. At 240 it is 0.6pp, and worse, **it scaled with
`tokens_per_item`**, so a wide endpoint and a narrow one could not have their floors compared — which is
exactly what the in-context and memorised variants need from each other.

Both families now select on one half of the sample and score on the other, `degenerate_answer` included so
the invariant between them holds. It also moved `mano`'s L10 floor from 4.695% to 4.277%, against the
analytic `1/23 = 4.348%` — so the old figure was inflated by ~0.35pp too, and every floor quoted in
earlier revisions of this file was slightly high. The cost is variance, since the held-out half is half the
sample, so the production floor sample is now 60,000 (`measure.reasoning.FLOOR_SAMPLE`); it is computed
once per run rather than per checkpoint.

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

G4 requires floor-to-ceiling ≥ 10 pp. Floors measured at 60,000 items with the corrected estimator
(§2), for **both** endpoints the plan now carries:

| L | `ctxmano` floor | width (padded) | `mano` floor | width |
|---|---|---|---|---|
| 2 | 10.02% | 250 → **256** | 6.09% | 8 |
| 3 | 10.05% | 252 → **256** | 5.97% | 10 |
| 4 | 10.69% | 254 → **256** | 4.41% | 12 |
| 5 | 9.98% | 256 → **256** | 4.25% | 14 |
| 6 | — | 258, unalignable | 4.32% | 16 |
| 8 | — | 262, unalignable | 4.50% | 20 |
| 10 | — | 266, unalignable | 3.89% | 24 |

Both have **analytic floors to check the estimates against**, which is the useful part. `mano` answers are
uniform over 23 residues, so the constant floor is `1/23 = 4.348%`. `ctxmano` answers are uniform over k,
but its best fixed-offset copy is right whenever the cell it reads equals the answer and once in `2k**2`
items that offset *is* the answer's own cell: `1/(2k**2) + (1 - 1/(2k**2))/k = 10.450%` at k=10. The
estimates above bracket both. **A floor that drifts away from its analytic value is a bug in the
estimator, not a property of the length** — that is how the selection bias in §2 was found.

**The in-context ladder stops at length 5, and the reason is packing rather than difficulty.** The trainer
concatenates a slice and chunks it into 512-token instances, so an item of width `w` is cut by a boundary
unless `w` divides 512. At `<mano>`'s 24 tokens that is **3.1%** of items — the figure already recorded on
`TaskStream.num_tokens`. At an in-context item's 266 it is **52%**: half of them truncated, most losing the
answer or part of the table they were meant to read. That is not a small tax, it is a broken endpoint.

`InContextManoTask` therefore takes `pad_to`, and every config sets it to **256**. At k=10 the natural
widths at lengths 2–5 are 250, 252, 254 and 256, so one aligned width covers the whole ladder for **at most
2.3% of tokens** and takes the cut rate to zero. Length 6 is 258 and is **refused at construction** rather
than run unaligned. Padding goes after the `eos` so the answer's offset is unchanged, uses `PAD` rather
than `EOS` so the document-boundary count is not inflated, and does not move the measured floor.

Four depths is enough for a depth response but tight for G1's ≥ 15 pp spread. The fallback, if the spread
over 2–5 comes back flat, is a **smaller alphabet**: k=6 gives width `102 + 2L`, so `pad_to=128` aligns
depths **2 through 13** — twelve depths, and deeper composition than the memorised task ever reached — at a
17.8% floor instead of 10.45%, which raises G1's absolute bar from 28.4% to 34.3%. See §6.

**Exit condition, and floor + 10 pp is not it.** That threshold is only G4's *range* requirement. The
binding constraint is G1, whose in-band lower bound is 20% of the floor-to-100 range:

| endpoint | floor | G1 needs |
|---|---|---|
| `ctxmano` | ~10.45% | **≥ 28.4% absolute** |
| `mano` | ~4.35% | **≥ 23.5% absolute** |

G1 additionally wants a ≥ 15 pp spread across depths, and G3 a ≥ 15 pp ablation drop. So the rule is:

> Select the **hardest length** at 28M in the entropy architecture that reaches its endpoint's absolute
> bound above, shows a **≥ 15 pp spread** across the swept depths, and has `modal_rate` bounded away
> from 1. Confirm the selected length on independent seeds before freezing it.

`ctxmano` is the confirmatory endpoint, so **its** sweep is the one that gates. `mano` is calibrated in the
same submission and reported as a secondary; if it fails to clear 23.5% at every length, the secondary
reading is simply unavailable and the confirmatory result stands on its own.

`modal_rate` is a diagnostic, not yet a gate input — no gate consumes it, and treating it as an exit
criterion without a pre-registered threshold would be inventing one. Pre-register the bound or drop it from
the rule.

**The committed configs are now 28M in the entropy architecture, which is where the treatment lives.**
`configs/cells/calibration/` holds **11 cells, `fanout_size: 11`**, and the index maps by *filename*:

| index | cell | endpoint |
|---|---|---|
| 0–3 | `p2_28m_ctxmanoL02` … `L05` | in-context, confirmatory |
| 4–10 | `p2_28m_manoL02` … `L10` | memorised, secondary |

`p2_28m_ctxmanoL*` sorts before `p2_28m_manoL*`, so the in-context block comes first. **Verify this against
the CLI before submitting** — that ordering is filename-derived, and the phase-1 directory had `113m`
sorting before `13m` for the same reason, which is the sort of thing that trains the wrong cell silently.

The 13M and 113M width-response sweep for G6 moved to `configs/cells/calibration_width/` (8 cells,
in-context only) and runs after a length is frozen, not before. An entropy-axis reasoning-only cell carries
`bits_per_attribute: 0` and **two** entities — the axis requires a positive count and the name space refuses
anything below two, so 16,800 tokens against 1.0B of reasoning is the smallest fact slice the pipeline will
build. The demand is nil and the vocabulary is the treatment's, which is the point.

**Verified by `--dry-run`, not by inspection.** Index 0 resolves to `28m_ctxmanoL02` at 3,906,250 items —
1.0B ÷ 256 exactly, so the padded width is live — index 4 to `28m_manoL02` at 125,000,000, index 10 to
`28m_manoL10` at 41,666,666, index 11 is refused, and all eleven arms come out at **3,814 steps on 1.000B
tokens and 31,426,944 parameters**, which is the entropy architecture rather than the count axis's. The
sweep is iso-token, so a depth response cannot be a training-length response. The first draft of these
configs carried `n_entities: 1`, which passed every structural check and died in `BioStream`; a test now
builds all eleven **with streams**.

If neither endpoint clears its bound, §6 applies instead of §4.B.

#### `<mano>` is partly a memory task, and the probe says how much

The operator tables have to live in the weights. Two tables of 23x23 entries plus a 23-entry unary map is
1,058 values at log2(23) bits: **4,786 bits, or 4.8 kbit**, against 114.3 Mbit of demanded fact content at
b=32 -- **0.0042%** of it. Ordinarily that is too small to argue about. But the design deliberately runs the
model about 4x oversubscribed, and the marginal thing is exactly what gets evicted there. So if `<mano>`
falls under fact load, "facts crowded out the arithmetic tables" fits as well as "facts crowded out
reasoning" -- and the first is knowledge-versus-knowledge, which Physics 3.3 established years ago.

`measure/reasoning.score_table_probe` scores a **length-2 task on the train split** beside the endpoint at
every checkpoint, reported as `mano_table`. One operation with a memorised expression is a table lookup and
nothing else, so:

| `mano_table` (L2, train) | `mano` (L10, eval) | reading |
|---|---|---|
| flat | falls | composition broke while lookup survived -- **reasoning** |
| falls | falls | the tables were evicted -- **knowledge vs knowledge**, not this paper's claim |
| flat | flat | no effect to explain (P3) |
| falls | flat | incoherent; suspect the harness |

It is a diagnostic, not a gate: no gate consumes it, and adding an unregistered threshold would be
inventing one. It costs one extra 4,000-item eval pass per checkpoint and it converts the confound into a
row of the results table. **The train split and length 2 are both deliberate** -- this probe wants the
memorised case, which is why it does not go through the eval-split guard the endpoint does.

**What the probe cannot do is make `<mano>` the published task.** It is not the task from the phi-3
claim, it did not work in phase 1, and calibration is what decides whether it works at all. If a length
clears 23.5% and the probe then shows both curves falling together, the honest write-up says
knowledge-versus-knowledge and the memorised reading is simply unavailable — which is why §9.1 made the
in-context variant the confirmatory endpoint rather than leaving the probe to carry the argument alone.

#### Submitting it

`.edullm/run.yaml` currently holds the M0 gate-report re-score, so it has to be swapped before this block
goes out, and the swap is a commit on an `edullm/` branch because **the platform builds from the commit, not
the working tree**. The shape:

```yaml
suggested_compute: gpu-4xa10g        # 4x was proven available; 8x returned 1.06x of it
fanout: {size: 11, index_parameter: cell}
command: >-
  bash -lc 'python src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID"
  --config-dir src/scripts/train/factcrowd/configs/cells/calibration
  --cell-index "$AWS_BATCH_JOB_ARRAY_INDEX"
  --save-folder "${EDULLM_OUTPUT_PREFIX}ckpt"
  train_module.dp_config.param_dtype=bfloat16'
```

**Name the dtype in the command text.** The precision guard reads the words of the command and cannot see a
dtype the program sets in code, so a command that omits it is accepted onto a card with no bfloat16 in
hardware and dies on the first kernel that needs the format — after the machine has been billed.

Then `edullm check --json` before `submit`, and **read `cost` and `approval_class` out of that output**. Do
not take a price from this file; those live in reviewed configuration that changes without anybody being
told. Match refusals on `code`, never on their prose.

### B. Entropy sweep × 3 seeds — the primary result

18 cells, 28M, b ∈ {0, 4, 8, 16, 24, 32}, at the calibrated lengths, every cell at
`mano_variant: both` so one run yields the confirmatory in-context endpoint, the secondary memorised one and
the table probe. Token totals are recomputed in §7 for the extra reasoning slice.

**Its configs are deliberately not committed, and that is the whole lesson of phase 1.** The block reads
`ctxmano_length` and `mano_length` from §4.A's result, so writing them now would mean choosing the lengths
before measuring them — which is precisely how eighteen cells came to be trained at a depth where the
endpoint had no dynamic range. One line generates them once a length is frozen:

```python
cells.write_cells(
    cells.replicate_block(
        cells.entropy_sweep_cells(row="28M", mano_variant="both",
                                  ctxmano_length=<calibrated>, mano_length=<calibrated>, phase="p2"),
        3,
    ),
    ".../configs/cells/entropy_p2",
)
```

Verified to produce 18 cells with distinct `qualified_id`s, and replicates that differ in **initialisation
and data order only** — `seed` is held at 1234 across all three so the corpus is identical, while
`init_seed` moves 1234 → 11207 → 21180. That is what makes the set a paired block and the per-seed slope
the right inferential unit rather than a cell-level standard error 2.83× too small.

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

- **G1 and G2 are now closed by evidence that already existed**, and both were unreachable for the same
  silly reason: `run_gates` has accepted `depth_scores` and `random_init_result` since it was written and
  nothing ever supplied them. `measure/evidence.py` now recognises `*_manoL{depth}` and
  `*_ctxmanoL{depth}` as G1's depth curve — filtered to the endpoint's own variant, since the two have
  different floors and one curve built from both would measure the difference between the *tasks* — and
  reads G2's untrained model from the **step-0 checkpoint every run has always written**, taken off the raw
  sequence rather than the per-cell last checkpoint, which is where it was being discarded.

  So G1 closes when §4.A is scored, and G2 costs nothing at all. Verified end to end: on a synthetic
  scored sweep both gates return verdicts **on the merits** rather than `no evidence`, and G3 — which
  genuinely has none — still says so.
- **G3** — the premise-ablated probe, the one remaining gate with no evidence path. It needs a corpus
  variant that does not exist, and a row cannot be admitted while it is owed.
- **G8** — the dilution ladder re-run at the calibrated length, now **iso-token**. 5 cells, 5.0B tokens.

  The phase-1 ladder was not comparable across its own arms. Cutting reasoning from 1.0B to 0.6B tokens at
  demand 0 cut total tokens and steps by the same 40% — 2,288 steps at the 60% dose against 3,814 at the
  0% one — so the dose moved training length in the *opposite* direction to the thing the gate holds
  constant, and a decline could be read either way. `dilution_ladder_cells(..., iso_token=True)` backfills
  exactly what the dose removes with **zero-bit biographies** (`b=0`, so schema-shaped tokens carrying no
  identifying content: 47,619 filler entities at the 60% dose, 11,905 at 90%). Every arm now trains 3,814
  steps on ~1.000B tokens and only reasoning share moves. `configs/cells/gates/` is regenerated.

  `iso_token=True` with a nonzero `demand_bits_per_param` is **refused**, not silently reconciled: the
  filler *is* the fact content under iso-token, so a requested demand has nowhere to go. Pass
  `iso_token=False` for a mixture-matched ladder and read its dose as share *and* length, which is the
  honest reading of the phase-1 arms.

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

Two distinct failures now, because there are two endpoints, and they want different responses.

### The in-context ladder is flat over depths 2–5

The most likely failure, and it is a *range* problem rather than a learnability one: four depths is a short
lever for G1's ≥ 15 pp spread. The response is a **smaller alphabet**, not a different task.

| k | floor | G1 needs | width | aligned depths at `pad_to` | padding worst case |
|---|---|---|---|---|---|
| 6 | 17.82% | 34.3% | `102 + 2L` | **2–13** at 128 | 17% (at L=2) |
| 9 | 11.66% | 29.3% | `204 + 2L` | 2–26 at 256 | 18.8% |
| **10** | **10.45%** | **28.4%** | `246 + 2L` | **2–5** at 256 | 2.3% |
| 12 | 8.65% | 26.9% | `342 + 2L` | 2–85 at 512 | 32.4% |

k=6 is the useful fallback: twelve depths, and composition deeper than the memorised task ever reached,
for a floor 7.4 pp higher and up to 17% of tokens in padding. k=9 and k=12 buy depth at more padding than
they are worth. The trade is depth range against floor, and if the k=10 ladder is flat, depth range is
exactly what is missing.

### Neither endpoint clears its absolute bound

Then reasoning-by-composition is not learnable at this scale and the design needs a different dependent
variable before any grid is worth buying:

1. **Fixed `<compare>` as primary.** The leak is closed and the floor is ~1%, so its achievable range is
   far wider. It is a *related*-fact composition rather than unrelated reasoning, which changes what a
   crowding result would mean — say so rather than substituting quietly.
2. **Retrieval-then-compute**: ask for an attribute of whichever entity satisfies a comparison. Answer from
   a 263-pool, floor ~0.4%, and the supervision reveals a composition rather than a value.

Note that option 3 in earlier revisions of this list — "the in-context endpoints `PRD.md` §8.3 names" — is
no longer a fallback. It is §4.A's confirmatory endpoint, built and calibrated.

---

## 7. Cost

Every treatment cell now carries `mano_variant: both`, which adds one reasoning slice at
`REASONING_TOKENS` — exactly **+1.0B per cell**, since the budget is per slice rather than a split of one
total. The right-hand column is the figure to use; the middle one is what the same block cost with a single
endpoint, kept so the delta is visible.

| block | cells | one endpoint | **both endpoints** |
|---|---|---|---|
| A calibration (4 ctx + 7 mano) | 11 | 11.0B | **11.0B** — one variant per cell by construction |
| A′ width response, ctx only | 8 | 8.0B | **8.0B** |
| B entropy × 3 | 18 | 108.0B | **126.0B** |
| C count × 3, reduced | 9 | 71.8B | **80.8B** |
| C′ count × 3, full | 18 | 214.6B | **232.6B** |
| D 64M × 3 | 18 | 469.0B | **487.0B** |
| D′ 113M × 3 | 18 | 832.0B | **850.0B** |
| E G8 iso-token ladder | 5 | 5.0B | **5.0B** |

The extra slice is 17% of block B and 4% of block D′ — it buys the confound-free confirmatory endpoint, the
memorised secondary and the table probe from one run instead of two blocks, so it is the cheapest thing in
this table per unit of what it settles.

**Measured throughput, 4×A10G:** 13M **886k tok/s**, 28M **484k tok/s**.

**The finding that governs hardware choice:** 8×A10G gave **1.06×** of 4×A10G at 28M. Doubling the devices
bought 6%. These sizes are communication- and launch-bound rather than FLOP-bound, so a faster card will not
return its FLOP ratio, and an A100 cluster is not automatically the right answer. What uses a big card is a
*bigger model* — an argument for the 113M rung, not for putting 28M on an H100.

At the measured 28M rate on 4×A10G:

| block | slot-hours | rate used |
|---|---|---|
| A calibration, 28M | **6.3** | measured, 28M (484k) |
| A′ width, 13M + 113M | **4.6–7.1** | 13M measured (886k) + 113M fitted (172k)/FLOP-linear (121k) |
| B, both endpoints | **72** | measured, 28M |
| C reduced, both | 46 | measured, 28M |
| C′ full, both | 133 | measured, 28M |
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

## 9. Two decisions, now taken

### 9.1 `<mano>` keeps its tables in the weights as a *secondary*; the confirmatory endpoint puts them in the prompt

**Decision: run both.** `InContextManoTask` is the confirmatory endpoint and `ManoTask` the secondary, both
carried by every treatment cell at `mano_variant: both`.

§4.A's probe *measures* the table confound. It does not remove it. Removal is what the in-context variant
is: the operator tables are stated in the prompt and **redrawn every item**, so nothing about the mapping is
memorisable — a model that had stored every table it ever saw still cannot answer, because this item's
table is new. A decline under fact load therefore cannot be table eviction.

The honest caveat is that the construct changes: it becomes **read-then-compute** rather than
compute-from-memory, so it says nothing about stored procedural knowledge. That is a real narrowing, and it
is also much closer to the chained-inference-over-context claim that motivated the project. Carrying both
means the narrowing costs nothing — the memorised endpoint is still there, with its probe, for the other
reading.

Both live in one cell rather than two blocks, which is what makes this affordable: one extra reasoning
slice at `REASONING_TOKENS` rather than a second treatment sweep. The **budget is per slice, not a split of
one total** — the same rule that keeps the reasoning-only control comparable to the arms it references —
so `both` doubles unrelated-reasoning tokens from 1.0B to 2.0B per cell. Against a treatment cell's fact
budget that is small; against a *control* cell it is the whole cell, so §7's control figures move and are
recomputed there.

**k=10, and the prompt turns out not to bind.** Stating one operator row-major with a row label costs
`k*(2+k)` tokens, so an item is `2*(1 + k*(2+k))*... ` → **266 tokens at k=10 and L=10**, against a
512-token instance. An earlier revision of this section estimated ~214 at L=4 only and assumed depth would
be sacrificed; it is not. The whole L2–L10 calibration ladder fits at k=10, and k=12 fits too at 362:

| k | floor | table tokens | width at L=10 | fits 512 | *aligns* to 512 |
|---|---|---|---|---|---|
| 8 | 13.18% | 162 | 186 | yes | at 256, depths 2–45 |
| **10** | **10.45%** | **242** | **266** | **yes** | **at 256, depths 2–5** |
| 12 | 8.65% | 338 | 362 | yes | at 512, depths 2–85 |
| 16 | 6.43% | 578 | 602 | **no** — refused | — |

**"Fits" is the weaker criterion, and it was the only one this section had in its first revision.** An item
that fits a 512-token instance but does not *divide* it is cut by a boundary — 52% of the time at width 266
— so the column that matters is the last one. See §4.A; the fallback ordering is in §6.

Ten over twelve because 10.45% already leaves ~90 points of range, more than any gate asks for, and the
extra 1.8 points is not worth 96 tokens of context per item plus a jump to `pad_to=512`. The refusal at
k=16 is a raised `OLMoConfigurationError`, not a truncation.

One thing to keep in view: `<ctxmano>` items are **254–266 tokens against `<mano>`'s 8–24**, so at equal
token budget the in-context endpoint sees roughly 11× fewer items. That is not a confound between the fact
arms — it is identical across them — but it does mean the two endpoints are not equally *trained*, and the
confirmatory sweep's learnability question is being asked of a much smaller item count. §4.A calibrates it
at the real budget, which is the only way to answer that.

### 9.2 Three seeds, and the margin is computed rather than claimed

**Decision: three replicates**, as §4.B has it. What changes is what gets reported.

The margin is **computed from the sigma the block measures**, via
`analysis/trend.achievable_margin(slope_sd, n_blocks)`. `2pp` is a *comparator* — the size of the one
published effect, 2.09 pp at Pythia-410M — and not a negligibility threshold. Nobody has one of those for
this question, and asserting one would be inventing it.

At three seeds that is a demanding place to stand, and the write-up should say so:

| between-seed slope SD | margin supportable at **3 seeds** | at 5 seeds, for reference |
|---|---|---|
| 0.3 pp | 0.98 pp | 0.50 pp |
| 0.6 pp | 1.96 pp | 1.01 pp |
| 1.0 pp | 3.26 pp | 1.68 pp |
| 1.5 pp | 4.90 pp | 2.52 pp |
| 2.0 pp | 6.53 pp | 3.36 pp |

**df=2 is expensive and the cost goes in the paper, not only in this file.** One-sided `t(0.95, 2) = 2.920`
against `1.796` at `df=11` — a **63% penalty** on every interval — and the exact-power MDE multiplier is
`3.264*sd` at n=3 against `1.682*sd` at n=5. Read in reverse: a 2pp claim at three seeds *requires* the
per-seed slope SD to come in at **≤ 0.613 pp**, and phase 1 never measured that quantity on a treatment.

So the reporting rule, which is the actual content of this decision:

> **Lead with the interval, not the verdict.** `[−1.2, +0.4]pp` is informative at any sigma. "Equivalent at
> 2pp" is a claim whose content depends entirely on a sigma the reader cannot see. If the measured sigma
> supports only a 4.9pp margin, the finding is "declines larger than 4.9 pp are excluded" — a weaker claim
> that is true — and the 2pp comparator is quoted beside it as the effect size that would have been needed
> to match the literature, not as a bar that was cleared.

If sigma comes back small enough that 2pp is supportable, say that and show the arithmetic. Adding seeds
later is a second submission on a later commit, so it is a fallback rather than the plan.

### 9.3 A related limit worth stating rather than fixing

**The sigma measured on the sigma cells does not transfer to the count axis.** All nine sit at demand 0. One
control sigma cannot set one MDE for a sweep whose arms differ in entity count, token budget, steps, mixture
and optimiser history, because between-seed variance is not guaranteed constant across them — and on the
count axis it is precisely the arms with more entities that have more to be variable about. The entropy axis
is iso-token by construction and much closer to safe. For §4.C, either measure sigma at a positive demand or
state the assumption; do not quote a demand-0 MDE as though it were the axis's.
