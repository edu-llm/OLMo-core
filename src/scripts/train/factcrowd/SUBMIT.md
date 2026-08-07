# Submission commands, 2026-08-05

Every value here was read off the platform's own config rather than remembered: input names and choice
options from `edu-llm/platform/.github/workflows/submit-run.yml`, hourly rates and bounds from
`config/workload-catalog.yaml`, the `$500` routine ceiling and the 24 h routine runtime from
`config/policy.yaml`. Ceilings below are `rate × nodes × runtime × attempts × cells`, the formula in
`contracts/workload.py`.

**Prerequisite: the branch must be on origin.** These reference `configs/cells/{sigma,gates}/` and
`--config-dir`, which exist only on `edullm/fact-crowding`. The platform builds an image from a pushed
commit, so nothing below runs until:

```bash
git push origin edullm/fact-crowding      # fast-forward, nothing to clobber
```

Then wait 8–11 minutes for the image build and its scan before submitting.

The commands pass `commit_sha=edullm/fact-crowding` rather than a literal SHA. The field takes "Branch, tag
or commit", the platform resolves it and seals the real SHA into the lineage record, and a file living in
the repository it names cannot pin a SHA that stays correct past the commit which writes it. Substitute an
explicit SHA if you need two submissions to be provably the same code.

> **SUPERSEDED 2026-08-07 for jobs 2–6 — they ran.** The count grid is complete (12/12), the entropy
> sweep is complete (6/6), and both re-runs landed: `28m_d4p8` finished at 134,479/134,480 in 20.14 h
> against a 20.2 h estimate, and `13m_d0p6` in 0.89 h. Three cells still owe a re-run because they hit the
> checkpoint-save defect described below. **`ROUND2.md` has the remaining commands.** Job 1's scoring
> prefixes are stale — it predates the two re-runs.

## State this replaces

| cell | outcome |
|---|---|
| 10 cells across both rows | complete, all 10 checkpoints + final |
| `13m_d0p6` | crashed at step 3,441 / 10,732 (32%), 7 checkpoints written |
| `28m_d4p8` | failed at step 97,450 / 134,480 (72.5%) after 13.80 h, last checkpoint step 67,239 |

Measured throughput, which every runtime below is derived from rather than estimated:

| row | world size | aggregate tok/s |
|---|---|---|
| 13M | 4 | 886,128 (n=6) |
| 28M | 4 | 484,214 (n=5) |
| 28M | 8 | 514,203 (n=1) — **1.06× for double the devices** |

## Common fields

```
-R edu-llm/platform submit-run.yml
-f repository=OLMo-core
-f commit_sha=edullm/fact-crowding
-f dataset_release=none
-f team=memory-split
-f experiment=fact-crowding
-f wandb_project=fact-crowding
```

`image_digest` is left unset on purpose — the image comes from the commit.

---

## 1. Score everything that exists — do this first

No new training. `--prefix` takes a fan-out's parent and finds the `cell-N/` children, so four prefixes
cover all twelve cells including the two partials. `gpu-1xa10g` because scoring is one process:
`require_a_process_for_every_device` refuses a multi-GPU profile whose command has no
`torch.distributed.run`, and scoring needs neither.

`olmo-core-check` (no checkpoint contract — scoring writes a CSV, not checkpoints) defaults to a 1 h
bound, so the override matters. Expect ~2 h for ~120 checkpoints; 8 h is insurance.

Ceiling **$8.05** = 1.006 × 1 × 8 × 1 × 1.

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
  -f maximum_runtime_hours=8 \
  -f maximum_attempts=1 \
  -f command='bash -lc '"'"'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs; S=src/scripts/train/factcrowd/score_run.py; O="$EDULLM_OUTPUT_PREFIX"; python $S --prefix $B/run_019fcfc6-c6cd-70be-aec5-e3e10d9fc2c4/ --out ${O}scores-count-13m.csv --work-dir /tmp/sc1 --device cuda; python $S --prefix $B/run_019fcfc8-14dd-706c-a2fa-ab2492af0bbb/ --out ${O}scores-count-28m-short.csv --work-dir /tmp/sc2 --device cuda; python $S --prefix $B/run_019fcfc8-acca-70c6-9345-f3674a37f8b0/checkpoints/ --out ${O}scores-count-28m-d2p4.csv --work-dir /tmp/sc3 --device cuda; python $S --prefix $B/run_019fcfca-270a-70dc-bc37-a4d7865db0c5/checkpoints/ --out ${O}scores-count-28m-d4p8-partial.csv --work-dir /tmp/sc4 --device cuda'"'"''
```

Four invocations rather than a loop, separated by `;` rather than `&&`, so one bad prefix does not cost
the other three. Each writes its CSV to `$EDULLM_OUTPUT_PREFIX` — a full `s3://…/teams/memory-split/runs/<this run>/`
URL that the workload role can write to.

Rows will all read `confirmatory=False, admission="no gate report for 'mano'"`. That is correct: jobs 2
and 3 are what produce the report.

## 2. G8 dilution ladder — the cheapest gate

5 cells, 1.0B tokens or less each, longest ~0.31 h at 13M. Fan-out over a directory because the ladder is
not a ladder *row* — `--row` would select all five anyway and only look like a constraint.

Ceiling **$56.72** = 5.672 × 1 × 2 × 1 × 5.

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
  -f fanout_size=5 \
  -f fanout_index_parameter=cell \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --config-dir src/scripts/train/factcrowd/configs/cells/gates --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

Index order is by filename, so `[0] 13m_dil100 [1] 13m_dil60 [2] 13m_dil80 [3] 13m_dil90 [4] 13m_dil95`
— not dose order, and harmless because every cell is self-labelled.

## 3. M0 σ block — three widths × three replicates

9 cells. This is what keys the seed count for the whole design, and it feeds G4 (ceiling), G6 (accuracy
against width) and G7 (run-to-run σ) as well. Longest cell is `64m_ctrl` at 1.0B tokens; **64M has never
run here**, so its throughput is extrapolated (~265k tok/s → ~1.05 h) and 3 h is deliberate margin.

Ceiling **$153.14** = 5.672 × 1 × 3 × 1 × 9.

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
  -f maximum_runtime_hours=3 \
  -f maximum_attempts=1 \
  -f fanout_size=9 \
  -f fanout_index_parameter=cell \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --config-dir src/scripts/train/factcrowd/configs/cells/sigma --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

`[0] 13m_ctrl [1] 13m_ctrl_r1 [2] 13m_ctrl_r2 [3] 28m_ctrl [4] 28m_ctrl_r1 [5] 28m_ctrl_r2 [6] 64m_ctrl [7] 64m_ctrl_r1 [8] 64m_ctrl_r2`

## 4. M2 entropy sweep — the identified axis, and the primary result

6 cells, 6.00B tokens each, 3.44 h each at the measured 28M rate. Two 4×A10G slots → ~10.3 h makespan.

Ceiling **$204.19** = 5.672 × 1 × 6 × 1 × 6.

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
  -f maximum_runtime_hours=6 \
  -f maximum_attempts=1 \
  -f fanout_size=6 \
  -f fanout_index_parameter=cell \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --row 28M --sweep entropy --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

`[0] 28m_b0 [1] 28m_b16 [2] 28m_b24 [3] 28m_b32 [4] 28m_b4 [5] 28m_b8` — alphabetical, not numeric.

## 5. Re-run `13m_d0p6`

2.81B tokens, ~0.88 h at the measured 13M rate. Restores the pre-registered six-level 13M grid; without
it `check_blocks(required_levels=6)` will refuse to report that row as the pre-registered design.

Ceiling **$17.02** = 5.672 × 1 × 3 × 1 × 1.

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
  -f maximum_runtime_hours=3 \
  -f maximum_attempts=1 \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --cell src/scripts/train/factcrowd/configs/cells/count/13m_d0p6.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

## 6. Re-run `28m_d4p8` — on four devices, not eight

35.25B tokens. At the measured 4-GPU rate that is **20.2 h**, inside the 24 h bound; at 8 GPUs it is
19.0 h, because doubling devices bought 6%. Four devices is therefore both **cheaper** ($115 of compute
against $310) and the thing that makes the 28M row unconfounded — every cell in it at one world size.

A fresh prefix, deliberately: re-submitting against the old folder would auto-resume from step 67,239 at
world size 8 and keep the confound.

Ceiling **$260.91** = 5.672 × 1 × 23 × 2 × 1. Two attempts here, unlike the short jobs, because a
transient failure 18 h in should resume rather than discard the run.

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
  -f maximum_runtime_hours=23 \
  -f maximum_attempts=2 \
  -f command='bash -lc '"'"'python -m torch.distributed.run --nproc-per-node=4 --standalone src/scripts/train/factcrowd/train_cell.py "$EDULLM_RUN_ID" --cell src/scripts/train/factcrowd/configs/cells/count/28m_d4p8.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'"'"''
```

20.2 h against a 23 h bound is 14% margin. If it fails again, the honest read is that this cell does not
fit an A10G at all and belongs on the P pool — `gpu-8xa100` at 21.96/h would do it in roughly 4 h.

## 7. Score again, once 2–6 land

Same shape as job 1 but pointed at the new run ids, and with the gate report written from M0 and then fed
back to admit the grid. Two passes, because a report assembled from a run must not admit that run:

```bash
# pass 1: build the report from the sigma block and the ladder
python src/scripts/train/factcrowd/score_run.py --prefix <sigma-run>/ --out m0-sigma.csv \
  --write-gate-report gates-mano.json --gate-endpoint mano --device cuda
# pass 2: admit the confirmatory grid against it
python src/scripts/train/factcrowd/score_run.py --prefix <count-run>/ --out count.csv \
  --gate-report gates-mano.json --device cuda
```

G1, G2 and G3 will still report as owed — they need task and corpus variants that are not built — so no
row is admitted yet. That is the design's real state, not a bug in the report.

## Totals

| job | cells | wall clock | ceiling |
|---|---|---|---|
| 1 score | 12 | ~2 h | $8 |
| 2 ladder | 5 | ~0.4 h | $57 |
| 3 σ block | 9 | ~4 h | $153 |
| 4 entropy | 6 | ~10.3 h | $204 |
| 5 `13m_d0p6` | 1 | ~0.9 h | $17 |
| 6 `28m_d4p8` | 1 | ~20.2 h | $261 |

Every ceiling is under `routine_maximum_cost_usd` ($500) and every runtime under
`routine_maximum_runtime_hours` (24), so all six are lead-approvable with no admin exception. Expected
*actual* spend is far below the ceilings — about $310 of compute in total.

Two 4×A10G slots serve jobs 2–6, so they queue rather than run together; submit in the order above.
`config/capacity.yaml` currently classes four-card G shapes as `after_a_wait`, so expect queue time.
