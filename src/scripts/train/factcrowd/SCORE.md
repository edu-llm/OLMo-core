# Scoring — the last two submissions, 2026-08-08

**All 32 cells have trained.** count 12/12, entropy 6/6, σ 9/9, ladder 5/5. These two submissions turn
checkpoints into the table you analyse.

## Corrections to earlier revisions of this file

Earlier versions of this document, and of `SUBMIT.md` and `ROUND2.md`, gave three instructions that
`CLAUDE.md` and `AGENTS.md` forbid. They are wrong and the commands below replace them:

- **`gh workflow run` is not the submission path.** `edullm` is. What a run does travels in
  `.edullm/run.yaml`; the compute profile, experiment and dataset are supplied at submit time.
- **Never read a price, a runtime bound, a cost ceiling or an approval class out of a document —
  including this one.** Every `$` figure in the earlier revisions was my own arithmetic over
  configuration that changes without notice. Run `edullm check --json` and read `cost` and
  `approval_class`.
- **Do not call AWS.** A Colab plan that set `AWS_ACCESS_KEY_ID` and ran `aws s3 ls` was drafted here and
  has been deleted. For most people it fails for want of a role; for the few it works for, it leaves no run
  anybody can cite. The sanctioned "my own machine" route is `edullm run` / `edullm shell`, which ships this
  tree to a machine — ungated, and likewise not citable, so use it to explore and not to produce the result.

A fourth correction, about hardware: I offered `gpu-1xt4` as a cheap substitute. For **training** that is
dangerous — a T4 is Turing and has no bfloat16 in hardware, `torch.cuda.is_bf16_supported()` lies about it,
and training sets bfloat16 in code where the platform's guard cannot see it. For **scoring** it is fine,
because scoring runs in float32; `--dtype float32` now says so in the command, which is what lets the guard
do its job either way.

## Is the queue actually the problem?

A fan-out's **parent job runs no container.** It has no attempts, no log stream and no `startedAt`, and
Batch marks it `SUCCEEDED` when its children succeed — which is exactly what a report reading
`Status: SUCCEEDED`, `Attempts 0 of 1`, `Log stream: not yet assigned` looks like. Read a *child* before
concluding anything:

```bash
edullm status --json <run-id>      # free, answers from GitHub, may be polled
edullm logs <run-id>               # slow by construction; not for a loop
```

If the children have run, the six CSVs are already under that run's output prefix and nothing below is
needed.

## 1. Score the confirmatory data

`.edullm/run.yaml` in this commit already holds this submission: `workload_profile: olmo-core-check`
(scoring writes a CSV and no checkpoint, so no checkpoint contract), a six-way fan-out on
`index_parameter: cell`, and one prefix per index.

```bash
# Commit and push first -- the platform builds the image from the last commit on an edullm/* branch.
git push origin edullm/fact-crowding

edullm check --json --experiment fact-crowding --dataset none --compute gpu-1xa10g
# exit 0 stands; 1 is refused on the merits, match on `code`; 2 the command or install is wrong;
# 3 the platform could not be asked, and is the only one worth retrying.
# Read `cost` and `approval_class` out of that JSON. Do not take them from here.

edullm submit --experiment fact-crowding --dataset none --compute gpu-1xa10g
```

`--dataset none` is deliberate and is a statement, not an omission: scoring reads existing checkpoints and
no published corpus.

| index | prefix | cells |
|---|---|---|
| 0 | `…fcfc6-c6cd…` | 13M count: ctrl, d0p3, d1p2, d2p4, d4p8 (plus the crashed d0p6, harmless) |
| 1 | `…fcfc8-14dd…` | 28M count: ctrl, d0p3, d0p6, d1p2 |
| 2 | `…fcfc8-acca…` | `28m_d2p4` |
| 3 | `…fd3c2-ebb6…` | `28m_d4p8`, the clean four-device re-run |
| 4 | `…fd3c2-9cb5…` | `13m_d0p6`, the clean re-run |
| 5 | `…fd3c1-1d5f…` | the whole entropy sweep |

## 2. Build the gate report

Edit `.edullm/run.yaml`: drop the `fanout` block, and replace the command with the one below. One file holds
one command, so a second submission is a second commit — which is also what makes each one citable.

```
command: >-
  bash -lc 'B=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/runs;
  python src/scripts/train/factcrowd/score_run.py --prefix
  $B/run_019fd3c0-8d2c-70ce-873e-4c2e333856b6/
  $B/run_019fd3bf-ece6-708d-9706-08967ddbd557/
  $B/run_019fdd84-9e11-707b-bcd3-adbadb4468ea/
  --out ${EDULLM_OUTPUT_PREFIX}m0.csv --work-dir /tmp/g
  --device cuda --dtype float32 --batch-size 512 --last-only
  --write-gate-report ${EDULLM_OUTPUT_PREFIX}gates-mano.json --gate-endpoint mano'
```

All three prefixes are needed: the σ block has no dilution ladder, the ladder has one replicate, and G7
needs three. Together they supply every gate that can currently be fed.

The crashed originals sit inside those same prefixes and are handled rather than avoided —
`assign_roles` keys on `(cell_id, replicate)` and keeps the **highest step**, so `13m_ctrl` r0 resolves to
the 3,814-step re-run rather than the 19-step corpse, and `13m_dil95` to 3,623 rather than 589.

Expect the report **not** to pass. G4, G6, G7 and G8 have their evidence now; G1 (task-depth sweep), G2
(untrained checkpoint) and G3 (premise-ablated probe) need task and corpus variants that are not built, so
they report as owed and no row is admitted. That is the checklist, not the verdict.

## Then

Feed `gates-mano.json` back as `--gate-report` to mark rows confirmatory, and read the CSVs. The first
question to put to them is the one the training curves could not answer: on the entropy axis, `28m_b32`'s CE
sits 0.50 nats *below* its no-memorisation floor while every other cell sits 0.31–0.45 *above* theirs. If
b32's Mano accuracy and achieved-bits curve track that drop, it is a real late-training phase change; if
they do not, the drop is about the training mixture and not about storage.
