# Runbook: what we submit, in order

## Iteration is local and near-instant

`edullm check` is a **local lint**: it "reaches no network", runs in ~0.2 s, queues
nothing and costs nothing. So the edit loop never touches hardware:

| step | where | time | cost |
|---|---|---|---|
| test suite (94 tests) | CPU | 13 s | 0 |
| calibration gate | CPU | ~20 s | 0 |
| corpus build, 3M-token smoke | CPU | ~60 s | 0 |
| `edullm check --json` | local | 0.2 s | 0 |
| corpus build, real (~500M tokens) | CPU | ~1-2 h, once | 0 |

Only three checks are deferred to submit time, all registry-related:
`no_published_image`, `image_is_ambiguous`, `image_scan_findings_unreviewed`. So a
clean `check` is not a promise, but everything about the *spec* is caught locally.

**The slow loop is the image build: 8-11 min per push** to an `edullm/**` branch.
So batch Dockerfile/dependency changes; don't iterate through them.

## Walltime is a cap, billed for actual time

`maximum_runtime_hours` becomes Batch's `attemptDurationSeconds` -- a timeout that
kills the attempt, not a reservation. The platform's own notification format shows
it: *"$0.02 spent, $2.01 authorised - ran 1m on gpu-1xa10g."*

So a 50-step smoke job costs **cents**, not an hour. One caveat: **approval
routing uses the worst case** (rate x hours x attempts x cells), so a generous
`--hours` can push a submission past the $500 auto-approve bound into a lead's
queue even though you will never be billed it. Keep `--hours` tight while
iterating.

## Prerequisites (one-time, and the real blocker)

None of this is optional and the order matters.

1. **`memsplit-hop` must exist as a GitHub repo under `edu-llm`.** It currently has
   no remote. Create and push it first.
2. **In this repo**: `.edullm/Dockerfile` (done -- must build `FROM ${BASE_IMAGE}`,
   verified pre-build), a build-caller workflow triggered on `edullm/**` *and*
   `workflow_dispatch`, and `AWS_ECR_PUBLISHER_ROLE_ARN` set as a **repository
   variable by hand** (`gh variable set`). There is no org-level variable; missing
   it yields `publisher_role_arn_is_empty`, and this step silently published
   nothing for another repo for days.
3. **Register the repo** -- do this *after* step 2, because a registration PR
   against a repo with no Dockerfile declares a path pointing at nothing:

   ```bash
   gh workflow run register-repository.yml --repo edu-llm/platform \
     -f repository=memsplit-hop \
     -f github_repository_id=$(gh api repos/edu-llm/memsplit-hop --jq .id) \
     -f reason="n-hop fact-externalisation training; custom entrypoint, single GPU"
   ```

   It edits five platform files, verifies, and pushes a branch -- **it does not
   open the PR**; it prints a compare URL and a body to paste. Then merge, then a
   role-holder must **deploy**. Nothing is registered until both.

4. **Hand-edit the generated workload profile before opening that PR.** The tool
   hardcodes 1 hour / 1 attempt and names it `<repo>-check`; `--hours 9` against a
   1 h profile is refused with `runtime_above_the_workload_bound`. Add to
   `config/workload-catalog.yaml`:

   ```yaml
     - name: memsplit-hop-train
       repository: memsplit-hop
       maximum_runtime_hours: "9"
       maximum_attempts: 1      # all compute environments are EC2, not SPOT, so
                                # retries only fire for a lost host
       checkpoint: null
   ```

   Precedent: `nested-learning-train` is 12 h while its auto-generated comment
   still says "one hour, one attempt" -- it was hand-adjusted in the PR. `config/**`
   is owned by nine people, so any admin or lead can review it.

## The corpus: upload once, read from all 12 runs

There is **no `EDULLM_INPUT_PREFIX`**. The sanctioned writable location outside a
run's own output is the team dataset prefix:

```bash
python scripts/build_corpus.py --out data/nhop_v1 \
    --n-entities 10000 --total-tokens 500_000_000 \
    --train-depths 1 2 3 --eval-depths 4 5 --n-eval 1000

P=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/datasets/memsplit-hop-v1
aws s3 cp data/nhop_v1/tokens.bin           $P/
aws s3 cp data/nhop_v1/weights.dense.bin    $P/
aws s3 cp data/nhop_v1/weights.split.bin    $P/
aws s3 cp data/nhop_v1/weights.random_contig.bin  $P/
aws s3 cp data/nhop_v1/weights.random_scatter.bin $P/
aws s3 cp data/nhop_v1/organizer.jsonl      $P/
aws s3 cp data/nhop_v1/report.json          $P/
aws s3 cp --recursive data/nhop_v1/eval/    $P/eval/
```

Submit with `dataset_release=none`. Each run stages only `tokens.bin` plus **its
own** sidecar (`checkpoint_io.stage_files`), so ~1.5 GB per run rather than 3 GB,
and re-staging is skipped by byte-count on a resumed attempt.

If you have no local AWS session (about 15 of 35 people don't), do the upload from
inside the lane: `edullm run --project memsplit -- bash upload.sh`.

## The jobs

### Job 0 -- smoke, ~5 minutes, ~$0.10

Proves the submission path end to end: custom non-OLMo entrypoint, corpus staging
from S3, S3 checkpoint write, and resume across attempts. Worth doing on its own
because **every command in the platform's recorded evidence is either `python -c`
or `python -m olmo_core.train`** -- we are the first custom entrypoint to run.

```bash
git checkout -b edullm/memsplit-smoke && git push -u origin edullm/memsplit-smoke
# wait 8-11 min for the image
edullm check --json --team memory-split --experiment smoke --dataset none
MEMSPLIT_CONDITION=dense MEMSPLIT_SEED=0 \
  edullm submit --team memory-split --experiment smoke --dataset none \
                --compute gpu-1xa10g --hours 1
edullm status && edullm logs <run-id>
```

with `total_tokens` overridden to ~50 steps. What to read off it: the `tok_s` in
`log.jsonl` (replaces the throughput estimate below), that `attempt 1` appears in
stdout, and that `ckpt.pt` lands in `$EDULLM_CHECKPOINT_DIR`. Then resubmit the
same run id once to confirm it prints `resumed at step N` and `attempt 2`.

### Jobs 1-12 -- the depth matrix

4 conditions x 3 seeds, one `edullm submit` each. **Twelve separate submissions,
not a fan-out**: a fan-out always routes to a team lead regardless of cost, while
each single run self-approves under $500.

```bash
for cond in dense split random_contig random_scatter; do
  for seed in 0 1 2; do
    MEMSPLIT_CONDITION=$cond MEMSPLIT_SEED=$seed \
    edullm submit --team memory-split --experiment depth-v1 --dataset none \
                  --compute gpu-1xa10g --hours 9
  done
done
```

Concurrency is not a constraint: `gpu-1xa10g` has its own stack at `MaxvCpus: 384`,
i.e. **96 concurrent single-GPU jobs**, and 12 runs is 48 of 768 account vCPU. G
and P quotas are separate pools.

One hazard: per-environment ceilings deliberately exceed the account quota, and an
unplaceable job sits in `RUNNABLE` under a state the submitter cannot read, with a
**1800-second auto-cancel**. If a submission vanishes after ~30 minutes, that is
why.

### Jobs 13+ -- evaluation

Eval is generation over ~15 strata x 1000 items and does not need 9 hours. Run it
as a separate short submission per checkpoint, or locally if you pull snapshots
down -- it is single-pass inference, not training.

## Cost and time

`gpu-1xa10g` at ~$1.01/hr. A10G has 600 GB/s bandwidth against the L40S's 864, and
these models are bandwidth-bound at this scale, so scaling the measured L40S
`d40m` figure of 184,671 tok/s gives **~128,000 tok/s (estimated)**:

| | per run (2B tokens) | 12 runs | cost |
|---|---:|---:|---:|
| gpu-1xa10g (est.) | ~4.3 h | ~52 h wall-parallel | **~$52** |
| worst-case authorised | 9 h x 12 | — | $109 |

4.3 h fits comfortably inside a 9 h cap, so **one attempt per run** and resume is
insurance rather than the plan. **Replace the estimate with Job 0's measured
`tok_s` before submitting the matrix.**

## Order of operations

1. `pytest` + `scripts/calibrate_nhop.py --n-entities 900 --n-items 1000` locally.
   The gate must print `usable` with no power warnings.
2. Build the real corpus; check all 11 gates pass in `report.json`.
3. Prereqs 1-4 above (repo, Dockerfile, publisher role, registration + deploy).
   **This is the long pole -- it is a PR plus a manual deploy, not a command.**
4. Upload the corpus.
5. Job 0. Read `tok_s`. Re-cost.
6. Jobs 1-12.
