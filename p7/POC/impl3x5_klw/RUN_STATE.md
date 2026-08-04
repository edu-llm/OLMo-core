# Run state — 2026-08-04

Superseded in part by [`RESULTS.md`](RESULTS.md), which holds the measured numbers. This file
is the operational log: what ran, what broke, and the traps worth knowing next time.

## Jobs, all resolved except one

| run | profile | outcome |
|---|---|---|
| `run_019fca79-…` | gpu-4xl40s | gate — every stage passed, then exit 1 on the appended `aws s3 cp`: **the AWS CLI is not in the image** (the driver uses boto3, which is why its own uploads worked) |
| `run_019fcaae-…` | gpu-4xl40s | **training, SUCCEEDED** — 4 arms, 22 ckpts each, ped_nll; checkpoints + results in S3 |
| `run_019fcd96-…` | gpu-4xl40s | math axis — **cancelled**, `InsufficientInstanceCapacity` for `g6e.12xlarge` in all four AZs the ASG can use |
| `run_019fcdcc-…` | **gpu-4xa10g** | math axis, retry on a plentiful instance family — Batch job `b4234f07-11b5-421d-8f4e-a78bd3759586` |

Judge ran locally, n=100, and its artifacts are in S3 at
`runs/run_019fcaae-…/impl3x5_judge_n100.tar.gz`.

## Capacity: g6e is the constrained family

`g6e.*` (L40S) is offered in **four** AZs — 1a, 1b, 1c, 1d. `g5.*` (A10G), `g6.*` (L4) and
`g4dn.*` (T4) get **five**, including `us-east-1f`. When `g6e.12xlarge` ran dry it failed in
every AZ available to it, rotating through them every ~8 minutes indefinitely while the Batch
job sat `RUNNABLE` with no explanation on the job record.

**`gpu-1xl40s` is NOT a fallback for `gpu-4xl40s`** — same g6e family, same four AZs. That was
my first suggestion and it was wrong. Go to `gpu-4xa10g` / `gpu-4xl4` / `gpu-4xt4` instead.
Diagnose with `autoscaling describe-scaling-activities`; Batch surfaces none of this.

## Checking on them

```bash
for RID in run_019fca79-3155-7019-b03d-e72422a7f31e run_019fcaae-3e79-70f8-908a-952c04a4d459; do
  aws batch list-jobs --job-queue sbsandbox-intern-edullm-gpu-4xl40s \
    --filters name=JOB_NAME,values=$RID \
    --query 'jobSummaryList[].{id:jobId,status:status}' --output table
  aws s3 ls s3://sbsandbox-intern-edullm-outputs/teams/post-training/runs/$RID/
done
```

Two traps worth remembering:

- plain `batch list-jobs` defaults to `RUNNING` and returns `[]` for a queued job — always pass
  `--filters name=JOB_NAME`.
- the log group is **`/aws/batch/sbsandbox-intern-edullm-gpu`**, not the `/aws/batch/job` that
  `container.logStreamName` implies. `get-log-events` on the latter raises
  `ResourceNotFoundException`.

## Still open

- **The math / forgetting axis** — in flight on `gpu-4xa10g`. Do not also request `bridge` in a
  math invocation: `bridge.py` skips checkpoint dirs that already exist, so an unfiltered bridge
  first would expose all 22 steps and the `--steps` filter would silently do nothing.
- **The 2x2.** These four arms answer "does James's reweighting add anything on top of Impl 5's
  targets", baselined against D4. They cannot separate that from "reweighting would have helped
  on gold too" — that needs his three configs re-run on *gold* through this pipeline, 3 runs.
- **A kappa-validated judge.** One judge family so far; `tutor-eval-suite` is still the missing
  piece.

## Done, with numbers in RESULTS.md

Training (4 arms, 22 ckpts each), ped_nll, and the blind pedagogy judge at n=100. The control
`bT451` matched D4 to 1.2e-4 on ped_nll and byte-identically on judge generation. All three
conditions are null against D4 on judged pedagogy, so **matched pedagogy quality is established
with CIs** — which is what makes a forgetting difference readable as a result.

## Judge harness

`llm_judge/build_problem_set.py` + `run_judge_impl3x5.py`, neither committed here (`llm_judge/`
is not on this branch). Both are inside
`s3://…/runs/run_019fcaae-…/impl3x5_judge_n100.tar.gz` along with the results.

**The problem-set rule is the part to not get wrong.** `generate_test_results.py` slices the
first N *rows* of the test split, but that split holds 1,743 dialogues over only 513 problems —
the first 16 rows are just **4** distinct problems, the first 100 rows only 29. Greedy decoding
makes repeats byte-identical, so judging them separately inflates n by 3-4x and shrinks every
CI. `build_problem_set.py` takes the first N *distinct* problems and verifies the result nests
the published 16 exactly.
