# Round two, 2026-08-07 — what is left, and why three cells died

Jobs 2–6 of `SUBMIT.md` all ran. Two estimates in that file were worth the arithmetic: `28m_d4p8` finished
in **20.14 h** against a 20.2 h projection, and `13m_d0p6` in **0.89 h** against 0.88 h.

> **ALL TRAINING IS COMPLETE as of 2026-08-07 19:12 UTC.** count 12/12, entropy 6/6, sigma 9/9,
> ladder 5/5 — 32 cells. The three round-2 cells finished on **4 physical A10Gs** (indices 0-3) at their
> exact targets in ~19 min each, which also confirms `save_async=False`: they are the same cells that
> crashed at step 19 three times, their checkpoint schedule still starts at 19, and they no longer die.
>
> **4xA10G is available and has been throughout** — 33 of 39 runs used it, three of them today. Nothing
> needs migrating. The only work left is scoring, and scoring is single-process on `gpu-1xa10g`, the
> shape with the best availability on the platform. Jump to **`SCORE.md`**.

## Where the design stands

| block | cells | state |
|---|---|---|
| count grid | 12 / 12 | **complete** — `13m_d0p6` and `28m_d4p8` both re-ran clean |
| entropy sweep (M2) | 6 / 6 | **complete** — the identified axis, and the primary result |
| σ block (M0, G4/G6/G7) | 7 / 9 | `13m_ctrl` r0 and r1 crashed at step 19 |
| dilution ladder (M0, G8) | 4 / 5 | `13m_dil95` crashed at step 589 |

`28m_d4p8` re-ran on **four** devices, so the 28M row is now at one world size throughout and its
confirmatory slope is unconfounded. The 8-GPU partial from 2026-08-05 stays descriptive.

## The defect that ate five runs

Six runs have failed across the whole project. **Five died within 0–15 steps of a planned checkpoint**:

| cell | died at step | last planned checkpoint | Δ |
|---|---|---|---|
| `13m_ctrl` (pilot) | 19 | 19 | 0 |
| `13m_ctrl` r0 | 19 | 19 | 0 |
| `13m_ctrl` r1 | 19 | 19 | 0 |
| `13m_dil95` | 589 | 579 | 10 |
| `13m_d0p6` | 3,441 | 3,434 | 7 |

The sixth, `28m_d4p8` at step 97,450 with its last checkpoint 30,211 steps behind, is the unrelated
wall-clock kill.

**Cause.** `CheckpointerCallback.save_async` defaults to `None`, and `pre_train` resolves that to
`backend_supports_cpu()` — true on this backend. It then runs:

```python
self.checkpointer.process_group = dist.new_group(timeout=timedelta(minutes=30))
```

A second process group. That is precisely the construct `async_bookkeeping` used, and turning *that* off
is already recorded in `train_cell.py` as having cost this project a run. The same thing came back through
a different field and kept eating runs for three more days, because nothing in the config said the word
`async` — the default resolved to it at runtime.

Fixed by passing `save_async=False`. Synchronous saving costs ~2 s per save, about 30 s over a run's ten.
A test asserts it on a *built trainer* rather than on the config literal, since a config literal is exactly
what failed to say it.

All 13M, which looks like a model-size effect and is more likely an exposure one: 13M cells step ~0.3 s,
so an async save spanning 2 s overlaps ~7 steps, against ~4 at 28M. Nothing here proves the mechanism —
what it proves is that saves are where runs die, and that the async path is the only thing special
about them.

## 1. Re-run the three crashed cells

All 13M, all tiny. Together ~0.9 h of GPU time. **Submit against a commit that carries `save_async=False`**
or they will crash the same way.

Ceiling **$34.03** = 5.672 × 1 × 2 × 1 × 3.

```bash
gh workflow run submit-run.yml -R edu-llm/platform \
  -f repository=OLMo-core \
  -f commit_sha=edullm/fact-crowding \
  -f workload_profile=olmo-core-train \
  -f compute_profile=gpu-4xa10g \
  -f dataset_release=none \
  -f team=memory-split \
  -f experiment=fact-crowding \
  -f wandb_project=fact-crowding \
  -f maximum_runtime_hours=2 \
  -f maximum_attempts=1 \
  -f fanout_size=3 \
  -f fanout_index_parameter=cell \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --config-dir src/scripts/train/factcrowd/configs/cells/round2 --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

`configs/cells/round2/` holds exactly the three: `[0] 13m_ctrl [1] 13m_ctrl_r1 [2] 13m_dil95`. Indices 0
and 1 both log the line `factcrowd cell '13m_ctrl'` — `cell_id` omits the replicate and only
`qualified_id` carries it — so check `replicate` in the run record, not the banner, to tell them apart. A separate
directory rather than re-running the whole σ block, because six of its nine cells are already done and a
fan-out index maps to a cell by position in a sorted directory — pointing at `sigma/` would retrain all
nine and give the survivors new checkpoint prefixes.

## 2. Score everything

### Why not a bigger GPU

`gpu-1xa100` and `gpu-4xa100` do not exist — the profile list is `cpu-32vcpu`, `gpu-1x{a10g,l4,l40s,t4}`,
`gpu-4x{a10g,l4,l40s,t4}`, `gpu-8x{a100,a10g,l4,l40s,t4}`. The only A100 shape is `gpu-8xa100`, and it is
the wrong tool three times over: `require_a_process_for_every_device` refuses a single-process command on
an 8-GPU profile; the rate is $21.96/h against $1.006/h; and it places `after_a_wait` with a **61-minute
median** queue (worst 12.6 h) where `gpu-1xa10g` places *reliably*.

Scoring is not GPU-bound anyway. Corpus building is numpy over up to 2.47M entities, checkpoint loads are
network, and the forwards are a 13–64M model on 24-token sequences.

### What actually makes it faster

Two structural levers, and one that measured smaller than expected:

| lever | factor | basis |
|---|---|---|
| `--last-only` | ~10× | one checkpoint per cell instead of ten |
| fan-out over prefixes | ~6× wall clock | six independent 1×A10G jobs, same GPU-hours |
| `--batch-size 512` | **1.17×** | measured, 989 s → 846 s on a real 13M checkpoint |

The batch size was expected to dominate and does not. On CPU it bought 17%, and there is no GPU in the
development environment to test whether launch overhead makes it matter more there — so it is set to 512
because it costs nothing and is proven not to change any number (bit-identical CSVs across all fourteen
rows at 32 and at 512, with a fast test pinning the invariant), **not** because it is load-bearing.

The two structural levers are hardware-independent and together turn ~4 h into minutes.

### The command

Six prefixes, one per fan-out index. `--prefix` still walks a fan-out parent, so each index covers a whole
submission.

Ceiling **$24.14** = 1.006 × 1 × 4 × 1 × 6.

```bash
gh workflow run submit-run.yml -R edu-llm/platform \
  -f repository=OLMo-core \
  -f commit_sha=edullm/fact-crowding \
  -f workload_profile=olmo-core-check \
  -f compute_profile=gpu-1xa10g \
  -f dataset_release=none \
  -f team=memory-split \
  -f experiment=fact-crowding \
  -f wandb_project=fact-crowding \
  -f maximum_runtime_hours=4 \
  -f maximum_attempts=1 \
  -f fanout_size=6 \
  -f fanout_index_parameter=cell \
  -f command='bash -lc '"'"'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs; P=(run_019fcfc6-c6cd-70be-aec5-e3e10d9fc2c4/ run_019fcfc8-14dd-706c-a2fa-ab2492af0bbb/ run_019fcfc8-acca-70c6-9345-f3674a37f8b0/checkpoints/ run_019fd3c2-ebb6-7038-85b6-eb508275feb4/checkpoints/ run_019fd3c2-9cb5-70d1-8aa4-7412df51bc6c/checkpoints/ run_019fd3c1-1d5f-7071-9171-a0c93f7dd0cc/); N=(count-13m count-28m-short count-28m-d2p4 count-28m-d4p8 count-13m-d0p6 entropy-28m); I=${AWS_BATCH_JOB_ARRAY_INDEX:-0}; python src/scripts/train/factcrowd/score_run.py --prefix $B/${P[$I]} --out ${EDULLM_OUTPUT_PREFIX}${N[$I]}.csv --work-dir /tmp/s --device cuda --batch-size 512 --last-only'"'"''
```

Drop `--last-only` for the full ten-checkpoint trajectory — the bit curve over training needs it, a first
endpoint read does not. Each cell writes under its own `cell-N/` output prefix, so collect six CSVs rather
than one.

Ceiling for the full-trajectory version, at 12 h: $72.43. Still routine.

### The old single-job form

Ceiling **$12.07** = 1.006 × 1 × 12 × 1 × 1. Expect ~4 h for ~270 checkpoints; 12 h is insurance. Keep this
if you would rather have one CSV than six.

```bash
gh workflow run submit-run.yml -R edu-llm/platform \
  -f repository=OLMo-core \
  -f commit_sha=edullm/fact-crowding \
  -f workload_profile=olmo-core-check \
  -f compute_profile=gpu-1xa10g \
  -f dataset_release=none \
  -f team=memory-split \
  -f experiment=fact-crowding \
  -f wandb_project=fact-crowding \
  -f maximum_runtime_hours=12 \
  -f maximum_attempts=1 \
  -f command='bash -lc '"'"'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs; S=src/scripts/train/factcrowd/score_run.py; O="$EDULLM_OUTPUT_PREFIX"; python $S --prefix $B/run_019fcfc6-c6cd-70be-aec5-e3e10d9fc2c4/ --out ${O}count-13m.csv --work-dir /tmp/s1 --device cuda; python $S --prefix $B/run_019fcfc8-14dd-706c-a2fa-ab2492af0bbb/ --out ${O}count-28m-short.csv --work-dir /tmp/s2 --device cuda; python $S --prefix $B/run_019fcfc8-acca-70c6-9345-f3674a37f8b0/checkpoints/ --out ${O}count-28m-d2p4.csv --work-dir /tmp/s3 --device cuda; python $S --prefix $B/run_019fd3c2-ebb6-7038-85b6-eb508275feb4/checkpoints/ --out ${O}count-28m-d4p8.csv --work-dir /tmp/s4 --device cuda; python $S --prefix $B/run_019fd3c2-9cb5-70d1-8aa4-7412df51bc6c/checkpoints/ --out ${O}count-13m-d0p6.csv --work-dir /tmp/s5 --device cuda; python $S --prefix $B/run_019fd3c1-1d5f-7071-9171-a0c93f7dd0cc/ --out ${O}entropy-28m.csv --work-dir /tmp/s6 --device cuda'"'"''
```

The 13M count prefix still contains the **crashed** `13m_d0p6` at `cell-2` alongside the good cells; its
seven checkpoints will be scored and are worth keeping as a partial trajectory. The completed re-run is a
separate prefix (`…c2-9cb5…`), and the two are told apart by `step` — the re-run reaches 10,732.

## 3. Build the gate report, then admit the grid

Only after job 1 lands, since G7 needs three replicates of `13m_ctrl` and two of them are the crashed ones.
Two passes, because a report assembled from a run must not admit that run.

```bash
gh workflow run submit-run.yml -R edu-llm/platform \
  -f repository=OLMo-core \
  -f commit_sha=edullm/fact-crowding \
  -f workload_profile=olmo-core-check \
  -f compute_profile=gpu-1xa10g \
  -f dataset_release=none \
  -f team=memory-split \
  -f experiment=fact-crowding \
  -f wandb_project=fact-crowding \
  -f maximum_runtime_hours=6 \
  -f maximum_attempts=1 \
  -f command='bash -lc '"'"'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs; S=src/scripts/train/factcrowd/score_run.py; O="$EDULLM_OUTPUT_PREFIX"; python $S --prefix $B/run_019fd3c0-8d2c-70ce-873e-4c2e333856b6/ $B/run_019fd3bf-ece6-708d-9706-08967ddbd557/ $ROUND2_PREFIX --out ${O}m0.csv --work-dir /tmp/g1 --device cuda --write-gate-report ${O}gates-mano.json --gate-endpoint mano'"'"''
```

**This will not pass, and that is the correct answer.** G4, G6, G7 and G8 are feedable; G1 (task-depth
sweep), G2 (untrained checkpoint) and G3 (premise-ablated probe) need task and corpus variants that are not
built, so they report as owed and no row is admitted. The report is a checklist of what is missing, not a
verdict that the design is broken.

`--prefix` takes several roots, which is what makes one report possible: the σ block, the ladder and the
round-1 re-runs are three separate submissions, and a report assembled from any one of them alone would
report the other gates missing while their evidence sat in the next prefix along. Set `ROUND2_PREFIX` to
`$B/<the round-2 run id>/` once job 1 lands, or drop it and accept G7 running on one replicate.

A root with no checkpoints is warned about and skipped rather than fatal, so a wrong id costs that
submission and not the report.

## What the training curves already say, and what they cannot

`train/CE` on the **count** axis is not a result: fact share rises 45%→97% across the grid and correlates
with final CE at r = −0.899, so the fall from 1.689 to 0.79 is mixture composition.

The **entropy** axis is different, and this is the one worth looking at, because it is iso-token by
construction — same entity count, same token budget, same mixture at every cell. Composition is held
fixed, so CE differences are real:

| cell | demand b/param | final CE |
|---|---|---|
| `28m_b0` | 0.200 | 0.7196 |
| `28m_b4` | 0.704 | 1.0424 |
| `28m_b8` | 1.209 | 1.2722 |
| `28m_b16` | 2.217 | 2.0128 |
| `28m_b24` | 3.226 | 2.6581 |
| `28m_b32` | 4.235 | **2.3951** |

Monotone rising, as it should be — more bits per attribute is less compressible — **except that b32 sits
below b24**.

**The corpus is not the cause.** Each biography carries 6 value spans of 4 tokens, and the distinct tokens
observed at each position within a span, over 60,000 entities, are exactly the pool size: 2 / 4 / 16 / 64 /
256 for b4 / b8 / b16 / b24 / b32. Realised entropy equals stated entropy — 24, 48, 96, 144, 192 bits per
entity — at every level, and `mean_tokens_per_bio` is 42.0 across the whole sweep. The axis is built
correctly.

So compare each cell against the CE it would pay if it memorised *nothing* and simply predicted value
tokens at their pool entropy — `0.833 × (bits_per_entity / 42) × ln2 + 0.167 × 1.689`, the second term
being the reasoning slice at the control's CE:

| cell | no-memorisation floor | observed | observed − floor |
|---|---|---|---|
| `28m_b0` | 0.282 | 0.671 | +0.389 |
| `28m_b4` | 0.612 | 1.010 | +0.399 |
| `28m_b8` | 0.942 | 1.252 | +0.310 |
| `28m_b16` | 1.602 | 2.030 | +0.428 |
| `28m_b24` | 2.262 | 2.708 | +0.446 |
| `28m_b32` | 2.922 | 2.422 | **−0.500** |

Five cells sit a consistent 0.31–0.45 nats *above* their floor. b32 is the only one below it, and going
below is only reachable by memorising. Its trajectory says the same thing: b16 and b24 are flat from 10%
to 100% (2.041→2.030, 2.672→2.708 — b24 actually *rises*), while b32 is flat to 75% and then falls hard,
3.20 → 2.90 → 2.42 through the cooldown.

So the highest-demand cell on the axis appears to have memorised more than the cell below it, which is
backwards for a capacity account. Three readings are open and this data cannot separate them: a real
late-training phase change, an artefact of the WSD cooldown interacting with a loss that has not yet
started descending, or something specific to a 256-option pool. **It is not explained, and b32 is the
highest-leverage point on the primary axis**, so it should be resolved before the entropy slope is
published. The cheapest discriminator is the scored endpoint: if b32's Mano accuracy and its achieved-bits
curve agree with its CE drop, the phase change is real; if they do not, the CE drop is about the training
mixture and not about storage.

None of this is the endpoint. Mano accuracy on the frozen 30,000-item eval set is, and only `score_run`
produces it.
