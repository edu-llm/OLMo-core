# Scoring — the last two jobs, 2026-08-07

**All 32 cells have trained.** count 12/12, entropy 6/6, σ 9/9, ladder 5/5. Nothing else needs a GPU for
training; these two jobs turn checkpoints into the table you analyse.

## On migrating off 4×A10G

Not needed, and the evidence is direct rather than inferred. Of 39 runs, **33 ran on four physical A10Gs**
— system metrics report GPU indices `0,1,2,3`, not a derived number — and three of those finished today at
19:12 UTC. `config/capacity.yaml` classes `gpu-4xa10g` as `after_a_wait`: nineteen nodes arrived between
01 and 05 August, all eleven runs that queued for them started, median wait 7 minutes, worst 1.5 hours,
and *none was ever cancelled for want of capacity*.

It is also moot: **scoring is single-process**, so it does not want four devices at all.
`require_a_process_for_every_device` would refuse a one-process command on any multi-GPU shape.

For the record, the A100 route is not available even if wanted: there is no `gpu-1xa100` or `gpu-4xa100`
in the profile list, only `gpu-8xa100` — which refuses single-process commands, costs $21.96/h against
$1.006/h, and queues a 61-minute median against `gpu-1xa10g`'s reliable placement.

`gpu-1xa10g`, `gpu-1xl4` and `gpu-1xt4` all place **reliably**. If A10G is ever contended, `gpu-1xl4`
($0.8048/h) is a drop-in substitute — change one field.

## 1. Score the confirmatory data

Eighteen cells over six prefixes, one per fan-out index. `--last-only` scores each cell's final checkpoint,
which is where every endpoint number is read; drop it when you want the bit curve over training.

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

| index | prefix | cells |
|---|---|---|
| 0 | `…fcfc6-c6cd…` | 13M count: ctrl, d0p3, d1p2, d2p4, d4p8 (plus the crashed d0p6, harmless) |
| 1 | `…fcfc8-14dd…` | 28M count: ctrl, d0p3, d0p6, d1p2 |
| 2 | `…fcfc8-acca…` | `28m_d2p4` |
| 3 | `…fd3c2-ebb6…` | `28m_d4p8`, the clean 4-device re-run |
| 4 | `…fd3c2-9cb5…` | `13m_d0p6`, the clean re-run |
| 5 | `…fd3c1-1d5f…` | the whole entropy sweep |

Each cell writes under its own `cell-N/` output prefix, so collect six CSVs.

## 2. Build the gate report

**One job, three prefixes, single process** — a gate report is assembled from one `score_run` invocation,
and its evidence is spread across three submissions:

- σ block `…fd3c0-8d2c…` — G4 ceiling, G6 width sweep, and `13m_ctrl` replicate 2
- ladder `…fd3bf-ece6…` — G8 doses 100 / 90 / 80 / 60
- round two `…fdd84-9e11…` — `13m_ctrl` replicates 0 and 1, and the missing dose 95

Neither of the first two alone yields a usable report: the σ block has no ladder, the ladder has one
replicate, and G7 needs three. Together they supply all four feedable gates.

The crashed originals sit inside those same prefixes and are handled rather than avoided — `assign_roles`
keys on `(cell_id, replicate)` and keeps the **highest step**, so `13m_ctrl` r0 resolves to the 3,814-step
re-run rather than the 19-step corpse, and `13m_dil95` to 3,623 rather than 589.

Ceiling **$4.02** = 1.006 × 1 × 4 × 1 × 1.

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
  -f command='bash -lc '"'"'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs; O="$EDULLM_OUTPUT_PREFIX"; python src/scripts/train/factcrowd/score_run.py --prefix $B/run_019fd3c0-8d2c-70ce-873e-4c2e333856b6/ $B/run_019fd3bf-ece6-708d-9706-08967ddbd557/ $B/run_019fdd84-9e11-707b-bcd3-adbadb4468ea/ --out ${O}m0.csv --work-dir /tmp/g --device cuda --batch-size 512 --last-only --write-gate-report ${O}gates-mano.json --gate-endpoint mano'"'"''
```

Expect the report **not to pass**. G4, G6, G7 and G8 now have their evidence; G1 (task-depth sweep), G2
(untrained checkpoint) and G3 (premise-ablated probe) need task and corpus variants that are not built, so
they report as owed and no row is admitted. The report is the checklist, not the verdict.

Run both together — they are independent, both on a reliably-placing shape, ~$28 all in.

## Then

Feed `gates-mano.json` back as `--gate-report` when you want rows marked confirmatory, and read the CSVs.
The first question to put to them is the one the training curves could not answer: on the entropy axis,
`28m_b32`'s CE sits 0.50 nats *below* its no-memorisation floor while every other cell sits 0.31–0.45
*above* theirs. If b32's Mano accuracy and achieved-bits curve track that drop, it is a real late-training
phase change; if they do not, the CE drop is about the training mixture and not about storage.
