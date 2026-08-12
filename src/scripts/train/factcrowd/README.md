# factcrowd — how to run it

`PRD.md` is why. This is how.

Does storing facts consume capacity that would otherwise serve reasoning? We train small models on
synthetic biographies, sweep the fact bits they demand per parameter, and measure reasoning on the
same checkpoints.

> **PHASE 1 IS COMPLETE AND ITS ENDPOINT FAILED.** All 32 cells trained; scoring found `<mano>` accuracy to
> be a *constant function* in all 18 confirmatory cells — 4.13%–4.61% against a 4.695% best-constant floor,
> zero cells above it — so the count and entropy axes measured noise about a constant and crowding is
> untested rather than refuted. The storage half produced a real one-seed result at demand 0.3.
> **Read `PHASE2.md` before running anything**: the first job there is a 14-cell, 14B-token `<mano>`
> calibration that gates every other spend, and `<compare>` has been redesigned because its answer leaked
> the value it asked for.

## Check it locally first

Nothing here needs a GPU, and `--dry-run` catches every config and arithmetic error:

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/train_cell.py local \
    --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml \
    --dry-run
```

It resolves the cell, generates the entity table, builds the vocabulary and the renderer, builds the
token-offset index, and prints the plan with a sample biography. Add `--json` for a machine-readable
version. The worst case in the grid — `64m_d2p4`, 2.79M entities — takes about two minutes and 8.6 MB of
scratch.

`--build-only` goes one step further: it builds the model, the mixture and the trainer, prints the
settings the platform enforces, and exits without training. That is where the checkpoint schedule, the
mixture targets and the parallelism config are decided, so it is the check worth running against a real
cell before spending a queue slot:

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/train_cell.py preflight \
    --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml \
    --save-folder /tmp/preflight --build-only
```

Run the tests the same way. The suite is torch-free apart from the end-to-end smoke runs, which are
marked `slow` and take a few minutes each because they train a real model in a subprocess:

```bash
python3 -m pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd -m 'not slow'
python3 -m pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd -m slow
```

## Submit a cell

One cell is one run. Push to a branch named `edullm/…` (nothing else gets an image), wait 8–11
minutes for the build and its scan, then [submit a
run](https://github.com/edu-llm/platform/actions/workflows/submit-run.yml):

| Field | Value |
|---|---|
| `repository` | `OLMo-core` |
| `commit_sha` | full SHA of a built commit |
| `workload_profile` | `olmo-core-check` for a smoke, `olmo-core-train` for a cell |
| `compute_profile` | `gpu-1xa10g` for a smoke, `gpu-8xh100` or `gpu-8xa100` for a cell |
| `team` | `memory-split` |
| `dataset_release` | **`none`** — the corpus is generated in-process |
| `wandb_project` | yours |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --cell src/scripts/train/factcrowd/configs/cells/count/28m_d1p2.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

One line, deliberately. The newlines an earlier draft used sit *inside* the single quotes, so `bash -lc`
reads them as statement separators and tries to run `src/scripts/...` as its own command.

`bash -lc` is required — the container runs the command directly, so without it `$EDULLM_RUN_ID`
arrives as fourteen literal characters. `--save-folder` must be on the line even though the program
defaults to it: the platform reads the command text to check that a run promising a checkpoint writes
one, and cannot see inside the process.

## The three jobs

The first run is 17 cells — five demands per row, four on 64M, plus a reasoning-only control on each.
Submit it as **three fan-outs, one per row**, so the rows run concurrently:

| `--row` | `fanout_size` | tokens | wall clock on 8×H100 | cost |
|---|---|---|---|---|
| `13M` | 6 | 34.7B | 1.3 h | $73 |
| `28M` | 6 | 71.5B | 4.6 h | $252 |
| `64M` | 5 | 76.6B | 8.8 h | $487 |

```
bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --row 28M --save-folder "$EDULLM_CHECKPOINT_DIR"'
```

with `fanout_size: 6` and `fanout_index_parameter: cell`. Each cell reads

> **`fanout_size` and `fanout_index_parameter` are not fields of `.edullm/run.yaml`.** `edullm check` refuses both with `Extra inputs are not permitted`. They are properties of the *submission record* (`schemas/submission-inputs.schema.json`), which the CLI derives; pass the fan-out to the verb and confirm the spelling with `edullm check --help`. The mentions below are stale and are kept only because the surrounding arithmetic is not.

`AWS_BATCH_JOB_ARRAY_INDEX` and gets its own checkpoint prefix. Sequential the grid is 14.7 h and about
$812; three jobs bring the wall clock to 8.8 h, and a failure in one row does not strand the others.

Those hours are `PRD.md` §10's FLOP estimate at 12/16/20% MFU per row, scaled to the current token
counts — **not measured**. Read the real MFU off the first cell's first 50 steps and re-budget before
submitting the 64M row, which is three-fifths of the cost.

The index maps to a cell by *position in the sorted directory*, and `ctrl` sorts before `d0p3`, so the
control is always index 0. Verify the mapping before submitting — a `fanout_size` from an older
directory runs a different cell under the name that was approved:

```
[0] 13m_ctrl  [1] 13m_d0p3  [2] 13m_d0p6  [3] 13m_d1p2  [4] 13m_d2p4  [5] 13m_d4p8
[0] 28m_ctrl  [1] 28m_d0p3  [2] 28m_d0p6  [3] 28m_d1p2  [4] 28m_d2p4  [5] 28m_d4p8
[0] 64m_ctrl  [1] 64m_d0p3  [2] 64m_d0p6  [3] 64m_d1p2  [4] 64m_d2p4
```

The entropy sweep is a fourth job, `--sweep entropy --row 28M`, `fanout_size: 6`, 36.0B tokens,
2.3 h and about $127.
It is the identified axis (`PRD.md` §3.1) and the cheapest thing here. Its indices sort alphabetically,
so `[0] b0 [1] b16 [2] b24 [3] b32 [4] b4 [5] b8` — every cell is self-labelled, so the order is
harmless, but it is not numeric.

### Pre-flight, all verified on this commit

| | |
|---|---|
| worst-case startup | 120 s and 8.6 MB of scratch for `64m_d2p4`, the 2.86M-entity cell |
| `max_checkpoints` | `None`, so no prune deletes a key the workload role may not delete |
| `ephemeral_save_interval` | `None` |
| evaluators | none attached |
| `max_duration` | explicit, in steps |
| W&B | attached, enabled only when `EDULLM_WANDB_PROJECT` is set |
| every checkpoint | carries the cell config and the schema, vocabulary, stream and task fingerprints |

That last row is what makes it safe to train before the scoring half exists. The corpus is generated
rather than stored, so a checkpoint is only scoreable if the run recorded how to rebuild it — the cell
config replays the vocabulary, entity table and reasoning items exactly, and the fingerprints let a
scorer prove it rebuilt the right ones rather than assume. Nothing about scoring needs to be decided
before these jobs run.



## Measured throughput, and the A10G route

`13m_ctrl` ran end to end on **4×A10G in 19 min 15 s** — 1.0B tokens, 3,814 steps, ten checkpoints,
`865,801 tokens/s` including setup and every save. That is the number the table below is built from, not
an estimate.

Two notes on reading the run's own metrics. OLMo-core's speed monitor reports `MFU 7.3%` against a
declared device peak of 312 TFLOP/s — that is the A100 figure; A10G's bf16 peak is 125, so the real MFU
is about **18%**, healthy for a 13M model. And that cell is 100% `<mano>`, the *slowest* generator here
(0.59M tok/s single-threaded against the bio renderer's 10.1M), yet data loading was 0.06% of step time
— so the fact cells, which use the fast renderer, are not data-bound either.

**Both columns below are now measured**, over twelve cells of the first count grid:

| row | world size | aggregate tok/s | n |
|---|---|---|---|
| 13M | 4 | 886,128 | 6 |
| 28M | 4 | 484,214 | 5 |
| 28M | 8 | 514,203 | 1 |

**Eight devices bought 6%, not 90%.** An earlier revision of this table took eight as 1.9× four and
flagged it as unmeasured; the real figure is 1.06×, so g5's missing NVLink costs more than the optimistic
end allowed for. Two corrections follow, and the second one killed a run:

- Hours do **not** scale linearly with non-embedding parameters. 28M runs at 484k tok/s where the
  parameter ratio predicts 390k, because the bigger model uses the device better (MFU 8.1% against 7.2%).
  Every 28M figure in the old table was ~26% pessimistic.
- `28m_d4p8` at eight devices needs **19.0 h**, not the 13.4 h the old table printed. A submission that
  set its runtime bound from that number was killed at 13.80 h and step 97,450 of 134,480, with CE, grad
  norm and throughput all steady to the last point — a wall-clock kill wearing a crash's clothes.

| cell | tokens | 4×A10G | 8×A10G |
|---|---|---|---|
| `13m_ctrl` … `13m_d4p8` | 1.0–15.9B | 0.3–5.0 h | — |
| `28m_ctrl` … `28m_d1p2` | 1.0–9.3B | 0.6–5.3 h | — |
| `28m_d2p4` | 17.8B | 10.2 h | 9.6 h |
| `28m_d4p8` | 35.3B | **20.2 h — inside the cap** | 19.0 h |

**So `28m_d4p8` does not need eight devices after all**, and four is better on every axis at once:
inside the 24 h ceiling, $115 of compute against $310, two concurrent slots instead of one, and it puts
the whole 28M row at one world size. The old advice to use eight was built on the 1.9× guess.

**64M still cannot run the fact cells on A10G.** Extrapolating the measured 13M→28M scaling gives ~265k
tok/s, so `64m_d2p4` needs ~41 h and `64m_d1p2` ~21 h. That row belongs on the P pool. The 64M
*control* is 1.0B tokens and fits in ~1.1 h, which is why the σ block can include it.

### One world size per confirmatory row

A row's slope is fitted across its demand levels. If world size changes *within* the row it becomes a
second variable, and on this plan it would not vary randomly — only the top cell needs eight devices, so
world size would correlate almost perfectly with demand and any effect of it would land directly on the
slope. Direction unknown, which is the problem: it could manufacture the result or mask it.

FSDP holds `global_batch_size` fixed, so the expected update is world-size-invariant and a 2-rank loss
curve reproduces a 1-rank one exactly on this code. That is evidence, not a guarantee: bf16 reduction
order differs with rank count and divergence compounds over tens of thousands of steps. It is not worth
resting the primary result on.

So: **every cell in a confirmatory row runs at one world size.** Once the throughput was measured this
stopped being a trade at all — the three options are no longer close:

| 28M row | makespan | cost | confound |
|---|---|---|---|
| **all six on 4×A10G** | **20.5 h** (two slots) | **~$232** | **none** |
| all six on 8×A10G | 38.6 h (one slot) | ~$629 | none |
| five on 4×A10G + `d4p8` on 8×A10G | ~19.0 h | ~$442 | world size aliases demand |

**Take the first**: cheapest, unconfounded, and within an hour of the fastest. The middle row is what an
earlier revision recommended, on the belief that eight devices were 1.9× four and that `d4p8` could not
fit on four; both were wrong.

The first grid ran as the third row, so `28m_d4p8` is currently at world size 8 while its five siblings
are at 4 — and only the top cell differs, which is exactly the correlation with demand that makes it a
second treatment. That partial run is **descriptive only**. Re-running it on four devices is $115 and
puts the row back on one world size, and phase 1 did exactly that.

The 13M row never had the problem: all six cells fit on 4×A10G, `13m_d4p8` at 5.0 h. 64M's fact cells go
to the P pool, where one world size covers the row anyway.

### Where the first grid actually got to

The count grid ran first, ahead of M0 and the entropy sweep, which is the ordering the next section argues
against. As of 2026-08-05 it stands at ten of twelve cells complete, with two to re-run:

| | |
|---|---|
| complete | both controls, `13m_d0p3/d1p2/d2p4/d4p8`, `28m_d0p3/d0p6/d1p2/d2p4` |
| `13m_d0p6` | crashed at step 3,441 / 10,732, seven checkpoints written |
| `28m_d4p8` | killed at step 97,450 / 134,480 after 13.80 h, last checkpoint step 67,239 |

235 GPU-hours and about $399 spent. Nothing is scored yet and no gate report exists, so every row of it is
`confirmatory=False` — correctly, and it stayed that way: G4 refused the endpoint. **`PHASE2.md` is the
live plan**; the operational notes for phase 1's submissions have been removed now that they have all run,
and `PRD.md` §16 keeps the incident history.

One reading to avoid: `train/CE loss` falls monotonically with demand across this grid, from 1.689 at both
controls to 0.79 at `28m_d2p4`. That is not the model getting better at reasoning. Fact share rises from
45% to 97% across the grid and correlates with final CE at **r = −0.899**; templated biographies are
highly predictable and drag the average down. The endpoint is Mano accuracy on the frozen 30,000-item eval
set, and only `score_run` produces it.

### The sequence, and why the count grid is not first

`score_run` marks every row `confirmatory=False` without a gate report, and admission is per endpoint.
Running the count grid first therefore produces 170 checkpoints of correctly-labelled *descriptive* data.
The cheap runs that change that come first:

| # | milestone | what it buys | profile | cells | makespan | cost |
|---|---|---|---|---|---|---|
| A | **M0 σ** — controls, 3 widths × 3 replicates | run-to-run σ (G7), accuracy vs width (G6), the ceiling (G4). σ is what keys the seed count for everything after. | `gpu-4xa10g` | 9 | ~4.0 h | ~$45 |
| B | **M0 G8** — the dilution ladder | the one gate that makes a null mean something: a treatment known to hurt, calibrated in reasoning-exposure units | `gpu-4xa10g` | 5 | ~0.7 h | ~$8 |
| C | **M2** — the entropy sweep at 28M | *the identified axis and the primary result* | `gpu-4xa10g` | 6 | ~13.0 h | ~$147 |
| D | **M3** — the count grid, 13M | the confounded axis, cheap row first | `gpu-4xa10g` | 6 | ~5.6 h | ~$63 |
| E | **M3** — the count grid, 28M | the row the M3 − M2 subtraction needs | `gpu-8xa10g` | 6 | ~27.2 h | ~$442 |

The table totals $705 and 50.5 h. A and B are $53 and 4.7 h of that — 7.5% of the cost, 9% of the time —
and until they run, nothing downstream can be claimed rather than plotted. C is the primary result and
costs a third of E. That ordering also means the first thing you learn is whether the design is powered at
all, which is the cheapest possible way to find out it is not.

64M stays on the P pool and is not in this table.

### Submitting it

Ceilings are `rate × nodes × maximum_runtime_hours × maximum_attempts × cells`. **Every figure below
assumes `maximum_attempts=1`**, which the `gh` commands pass explicitly — the platform's default is
higher, and at 2 attempts **E1 ($652) and E3 ($554)** cross the $500 routine bound and need admin
release. The rest stay under it, C and D only barely at $476.
Runtime limits are set per submission rather than stretched to cover the longest cell, because the cell
count multiplies into the ceiling.

| # | `compute_profile` | `nproc` | selector | `fanout_size` | `max_runtime_hours` | ceiling |
|---|---|---|---|---|---|---|
| A | `gpu-4xa10g` | 4 | `--config-dir …/configs/cells/sigma` | 9 | 3 | $153 |
| B | `gpu-4xa10g` | 4 | `--config-dir …/configs/cells/gates` | 5 | 2 | $57 |
| C | `gpu-4xa10g` | 4 | `--row 28M --sweep entropy` | 6 | 7 | $238 |
| D | `gpu-4xa10g` | 4 | `--row 13M` | 6 | 7 | $238 |
| E1 | `gpu-8xa10g` | 8 | `--row 28M` (all six, short limit) | 4 | 5 | $326 |
| E2 | `gpu-8xa10g` | 8 | `--cell …/count/28m_d2p4.yaml` | — | 9 | $147 |
| E3 | `gpu-8xa10g` | 8 | `--cell …/count/28m_d4p8.yaml` | — | 17 | $277 |

`--config-dir` fans out over a whole directory ordered by filename; `--row` filters one axis directory to
a ladder row. A and B use the first because neither set is a row: `sigma/` is three widths × three
replicates and `gates/` is one row's five dilution arms, so `--row` would select everything anyway and
only look like a constraint. **E1's `--row 28M` selects all six cells**, so its index range must be
capped by `fanout_size: 4` — filenames sort by ascending demand (`ctrl, d0p3, d0p6, d1p2, d2p4, d4p8`),
which puts the two long cells last and lets E2/E3 pick them up by name with limits that fit them.

`sigma/` holds all three replicates including r0, which duplicates the three controls in `count/` — about
2.7 slot-hours and $15 of repeated work. Deliberate: it makes A self-contained, so the gate report can be
assembled from A and B alone, before any confirmatory cell has run. That is the entire reason for running
M0 first, and $15 is a cheap price for not coupling the report to the grid it admits.

E is split three ways for the ceiling, not for speed: one submission of six cells at a 17 h limit would
price at $1,466 and need admin. Split, every line is routine. The cells still run sequentially in the
single 8×A10G slot, so the makespan is unchanged at ~27.2 h.

Whether D can overlap E depends on one platform fact worth confirming before you plan around it: if the
G pool's `MaxvCpus` is a single shared budget, then one 8×A10G job consumes what two 4×A10G jobs would
and the two cannot run at once — makespan becomes the sum, ~33 h, rather than the max. If the profiles
draw on separate budgets they overlap and it is ~27 h. The per-submission ceilings above are unaffected
either way.

### Producing the gate report

Two passes, and the second cannot admit the first:

```bash
# 1. Score M0 (A + B) and assemble the report from what those runs measured.
python src/scripts/train/factcrowd/score_run.py \
  --prefix s3://…/factcrowd-m0 --out m0-scores.csv \
  --write-gate-report gates-mano.json --gate-endpoint mano

# 2. Score the confirmatory grid, admitting rows against that report.
python src/scripts/train/factcrowd/score_run.py \
  --prefix s3://…/factcrowd-m3 --out m3-scores.csv \
  --gate-report gates-mano.json
```

Pass 1 deliberately does **not** admit its own rows: a report assembled from a run cannot admit that run,
or the gate cells would be admitting themselves. It logs what it recognised — which doses it saw, which
control became the ceiling, how many replicates fed σ — so a refusal is diagnosable.

Four gates are feedable from configs that exist (G4, G6, G7, G8). **G1, G2 and G3 need task and corpus
variants that are not built**, so they come back refused and no row is admitted yet. That is the honest
state of the design: the report is a checklist of what is owed, and it is not going to quietly pass.

## Before you submit: what an expert review changed

An independent review returned a stop-ship verdict on the commit before this one. Four defects would
have invalidated the runs, and all four are fixed with regression tests. Two are worth knowing about
because they change what you should expect to see:

- **Eight GPUs used to train eight independent models.** No `dp_config` meant FSDP was never applied,
  gradients were never reduced, and each rank saw an eighth of the corpus — ~25 exposures instead of
  200. Now FSDP with bf16 parameters, fp32 reductions, clipping at 1.0 and compilation on GPU. Runtime
  and cost estimates in this file describe *that* configuration; the older ones described fp32 without
  compilation, so treat both as unmeasured until the 13M row reports its MFU.
- **The entropy sweep varied the model.** Its vocabulary grew with entropy — 1,920 padded tokens at b=0
  against 8,064 at b=32, i.e. 8.1% more parameters and a 4.2× wider softmax on the high-entropy arm.
  Every cell now shares one union vocabulary. If you are running job 4, use a commit at or after this
  one; earlier entropy results are not comparable across cells.

The other two were latent: the reasoning eval set was 100% leaked from training by its keying, and
`init_seed` was never set so replicates would have shared one initialisation. Neither had run yet.

`PRD.md` §16.5 has the full account, including four places I judged differently from the review and the
list of accepted-but-unbuilt work.

## Configs

One YAML per cell under `configs/cells/`, one directory per submittable set, because a fan-out maps an
array index to a cell by position — its size has to be what `ls` says, and one shared directory would let
`--row 28M` pick up cells the submission never asked for.

| directory | cells | what submits it |
|---|---|---|
| `count/` | 17 | the confounded count axis, three rows plus a control each |
| `entropy/` | 6 | the iso-token entropy axis at 28M — the identified one |
| `gates/` | 5 | G8's dilution ladder, the cheapest run in the design |
| `sigma/` | 9 | M0's σ block: the three controls × three replicates |
| `calibration/` | 14 | `<mano>` depth sweep, 13M and 113M × 7 lengths — **run this before any grid** |
| `round2/` | 3 | the three cells that hit the checkpoint-save defect, kept for provenance |
| `smoke/` | 4 | local end-to-end tests, never submitted |

A cell states **either** its demand or its entity count, never both — the other is derived, so a cell's
label and its corpus cannot disagree. Regenerate them rather than hand-editing the set:

```python
from factcrowd import cells
cells.write_cells(cells.first_run_cells(), "src/scripts/train/factcrowd/configs/cells/count")
cells.write_cells(cells.entropy_sweep_cells("28M"), "src/scripts/train/factcrowd/configs/cells/entropy")
cells.write_cells(cells.dilution_ladder_cells("13M"), "src/scripts/train/factcrowd/configs/cells/gates")
controls = [c for c in cells.first_run_cells() if c.is_control]
cells.write_cells(cells.replicate_block(controls, 3), "src/scripts/train/factcrowd/configs/cells/sigma")
```

`write_cells` refuses a filename collision rather than resolving it: replicates are named by
`qualified_id`, so two of them cannot silently overwrite each other — which they did, since `cell_id`
omits the replicate and the replicate is this design's inferential unit.

Editing one field of one cell by hand is fine; a test resolves every committed config, so a cell that
does not add up fails locally rather than on a GPU.

## What is here, and what is OLMo-core's

Ours is the corpus and the arithmetic that places a cell on the demand axis:

| | |
|---|---|
| `ladder/rho.py` | demand ↔ entity count, both parameter bases, the name term |
| `ladder/sizes.py` | the four widths at depth 12, with the built parameter count asserted |
| `corpus/entities.py` | the entity table; exact bits, unique names, a prefix property across cells |
| `corpus/values.py` | attribute values as words; the bioS and entropy schemas |
| `corpus/vocab.py` | the closed word-level vocabulary |
| `corpus/render.py` | 32 templates over four length bands; exact value spans |
| `corpus/stream.py` | stream order, the token-offset index, token assembly |
| `corpus/tasks.py` | the reasoning slices — Mano at L=10, and the related comparison task |
| `corpus/source.py` | a 90-line `TokenSource` adapter |
| `cells.py` | one cell, and everything derivable from it |
| `train_cell.py` | assembles OLMo-core's configs and calls `fit()` |
| `measure/gates.py` | G1–G8, and the versioned report that is the only thing granting admission |
| `measure/evidence.py` | assembles that report from scored runs; feeds G4, G6, G7, G8 |
| `score_run.py` | checkpoints in, one CSV row per (cell, replicate, step, endpoint) out |

Everything else is OLMo-core's and is **not** reimplemented: `TransformerConfig.llama_like` for the
model, `AdamWConfig` and `WSD` for optimisation, `ConcatAndChunkInstanceSource` for packing,
`ComposableDataLoaderConfig` for shuffling and batching, `ListCheckpointerCallback` for the log-spaced
snapshots, and the trainer for checkpointing and resumption.

The split between `corpus/stream.py` and `corpus/source.py` is deliberate. All the logic is on the
torch-free side and tested; the adapter that needs the full install is four methods. Earlier in this
branch a module put its only real logic behind a torch import, its tests skipped, and a call that
raised `TypeError` for every input passed both review and type-checking.

## The reasoning slices

Set `reasoning_tokens` on a cell and it trains on facts *and* reasoning, mixed through
`MixingInstanceSource` at **fixed absolute token counts** — not fixed ratios. Every cell that carries a
slice sees the same number of that slice's tokens, so a difference in reasoning score cannot be a
difference in reasoning exposure. A bare `CellSpec` defaults to `0`, which trains on facts alone; the
grid generators default to 1.0B.

The budget is **per slice**, not a total split between them. That distinction is load-bearing: the
control carries only `<mano>`, so splitting a fixed total would have handed it twice the `<mano>`
exposure of every cell it is the reference for, and its score would have beaten theirs for a reason
that has nothing to do with facts.

The two slices carry deliberately *different* budgets — `reasoning_tokens` 1.0B for `<mano>`,
`related_reasoning_tokens` 50M for `<compare>`. The related one is sized on **per-entity coverage**, not
on parity: it names two of the fixed 25,000 probe entities per item, so at 1.0B each entity's birth-year
rank would be supervised 4,211 times against the 200 exposures the fact slice gets. At that rate the
slice teaches the ranks outright, "needs two facts" stops being true, and a decline would say nothing
about fact access. 50M gives 211 mentions per entity, level with the facts.

Two slices, each prefixed with a domain token so the endpoints stay separable:

| | | |
|---|---|---|
| `<mano>` | mod-23 mental arithmetic, 10 operands, no chain of thought | unrelated to the facts |
| `<compare>` | the earlier of two people's birth years | needs two facts to answer |

`<mano>` is Physics 4.1's task at the length that paper found the transition at. It is the primary
endpoint because it shares no entity with the corpus: if it degrades as fact demand rises, the cost is
capacity, not fact access. `<compare>` is the contrast — it reads two biographies, so it should
degrade earlier and harder if the mechanism is retrieval instead.

Both are generated per example from a seed rather than drawn from a fixed set, so neither slice holds an
item worth memorising. Both report a **measured** degenerate floor, searched over two policy families —
the best constant answer *and* the best copy of a fixed span of the prompt: **4.64%** for Mano against a
4.35% uniform baseline, **0.70%** for compare.

Measuring the floors changed both tasks, which is the argument for measuring them.

- Mano: with a free choice of operand, `× 0` is absorbing and the floor came out at **8.34%** — above
  the 6.80% the paper reports for the length this design *rejected*. Zero is now excluded as a
  multiplicand.
- `<compare>` originally answered with the earlier person's **name**, which is a span of its own prompt,
  so "always name the first person" scored **50.2%** while the best constant name managed 0.02%. A binary
  endpoint with a 50% floor has half its range available to a policy that reads no facts, and any score
  below 50% would be under its own floor. It now answers with the earlier **birth year**, a word that
  never appears in the prompt, so no copy policy can reach it.

`<compare>` is skipped where there is nothing to compare: on the entropy axis, whose attributes are six
positional four-word composites with no birth year, and on the control, which has no facts at all. Both
carry `<mano>` alone. `CellSpec.reasoning_slice_names` is the single source of truth for that choice and
`BuiltCorpus` asserts its own built set against it, so the token arithmetic and the data cannot disagree
about a cell.

### The reasoning-only control

One per row — `13m_ctrl`, `28m_ctrl`, `64m_ctrl` — stating `demand_bits_per_param: 0.0`. Zero facts:
no entity table, no renderer, no fact stream. It keeps the row's schema and vocabulary so the model is
architecturally identical to the cells it anchors, and it is the demand-0 endpoint of every crowding
curve — the one cell where a low reasoning score cannot be blamed on fact load.

Zero entities is *stated*, never solved for: `rho.solve` refuses a zero target and should keep refusing
it, since its linear path divides by bits-per-entity and its name-term path is non-monotone there. A
cell with demand 0 and no reasoning tokens is refused outright, being a run with no data at all.

## Scoring a finished run

Training writes checkpoints; `score_run.py` reads them and produces one CSV you can analyse.

```bash
PYTHONPATH=src python3 src/scripts/train/factcrowd/score_run.py \
    --prefix s3://.../runs/$RUN_ID/checkpoints \
    --out    s3://.../runs/$RUN_ID/scores.csv
```

`--prefix` takes either one cell's checkpoint directory or a fan-out's parent — it finds the `cell-N/`
segments itself. Add `--steps 0,1220,3814` to score a subset.

**It is a separate single-process job, not a callback.** Recall needs free-running generation, which
`TransformerGenerationModule` cannot provide from inside a trainer without re-parallelising the model
(`PRD.md` §8.2). And `load_model_and_optim_state` reshards a checkpoint saved on four ranks into one
unsharded model with no process group, so there is nothing to distribute. Scoring the whole first run is
minutes of forward passes on one small GPU, dominated by pulling shards from S3.

### What each checkpoint yields

| | |
|---|---|
| **reasoning** | per endpoint: three counts, accuracy, answer-token CE in bits, and the **measured** floor |
| **achieved bits** | Allen-Zhu's estimator over the value spans the renderer returns, with the per-entity distribution |
| **template reconstruction** | `template_<attr>_generation` and `_recognition`, each with its own `_chance` — *not* closed-book recall, which is unbuilt (PRD §8.2) |

Both endpoints render a **single-token answer at a known position**, so grading is a teacher-forced argmax
— identical to what a greedy decode would produce, with no continuation to truncate and no string to
parse. That is what removes the failure mode `PRD.md` §1 catalogues four times: an eval whose score is
bounded by its parser rather than by the model. `n_unparseable` is reported anyway, because §8.6's G7
requires it under 5% and a future multi-token endpoint could break the property.

### The layering

| | |
|---|---|
| `measure/spans.py` | which loss position pays for which token — one rule, one place |
| `measure/endpoints.py` | `EndpointResult`: the shape every endpoint reports in |
| `measure/checkpoint.py` | find a checkpoint, rebuild its corpus, verify fingerprints, load weights |
| `measure/reasoning.py` | the dependent variable |
| `measure/bits.py` | achieved bits against demanded |
| `measure/evidence.py` | assembles a gate report from scored runs — the only route to `confirmatory=True` |
| `measure/recall.py` | generation and recognition |
| `measure/gates.py` | `PRD.md` §8.6's G1–G8 |
| `measure/collect.py` | one tidy row per (cell, replicate, step, endpoint) |
| `analysis/trend.py` | per-seed slopes, TOST equivalence, non-inferiority, MDE |

`spans.py` exists because an off-by-one there is invisible: `ce_loss[t]` scores `input_ids[t+1]`, so the
cost of token *p* is `ce_loss[p-1]`. Get it backwards and a bit count charges a value token's cost to the
literal before it, and an endpoint grades the token before the answer. Both produce plausible numbers. It
is checked against a manually computed cross-entropy from a real model.

Every scorer takes a `forward` callable rather than a model, so the logic is testable on a stub that
returns chosen logits — including a model that answers one position early, which is the only way to catch
the off-by-one from the scorer's side.

### Two honesty properties worth knowing before you read a number

**Achieved bits are an upper bound while intra-document masking is off.** A packed biography can attend to
its neighbour, so some of what looks stored was read from context. Every row carries
`bits_is_upper_bound`, and `check_against_capacity()` refuses a figure above Physics 3.3's ~2 bits/param
ceiling — such a figure is a broken measurement, not a striking result.

**The rebuild is verified, not assumed.** The corpus is generated rather than stored, so scoring replays
the cell recorded beside the weights and checks the schema and vocabulary fingerprints against what
training wrote. This has already caught a real case: a checkpoint from before the entropy axis moved to a
union vocabulary rebuilds to a different schema today, and scoring it would have produced entirely
reasonable-looking numbers about a corpus the model never saw.

### Analysis

`analysis/trend.py` takes the **per-seed slope** as the inferential unit, not the individual cell.
`PRD.md` §8.5 asked for one regression over all cells; that treats correlated observations as independent,
and on the planned 3×6 design the naive cell-level standard error comes out **2.83× smaller** — its 90%
interval declares equivalence where the blocked interval cannot, on the same data. `pooled_trend` is
therefore per ladder row, and both `tost` (genuine ±2pp equivalence) and `non_inferiority` (the one-sided
rule, correctly named) are provided so a null can say which one it is.

## Still to build

The gates' *evidence*, and the endpoints that need building to satisfy them. `measure/gates.py`
implements G1–G8, but several need runs that do not exist yet — a label-permuted control, a
premise-ablated probe, a reasoning-token dilution ladder — and a gate whose evidence is missing returns
**false**, never a silent pass. `PRD.md` §8.3 also still wants Brevo1 and Reasoning Core: until an
in-context endpoint lands, a `<mano>` decline is ambiguous between "reasoning crowded out" and "mod-23
tables crowded out".

Intra-document masking (`PRD.md` §7.3) and padding reasoning items to 32 tokens so chunking stops cutting ~3% of them are both specified and unbuilt.
