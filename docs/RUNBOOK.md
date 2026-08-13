# Runbook: the jobs, what each tests, and the command

## Shapes: what is actually available

Read from `config/workload-catalog.yaml`. Only `provisioned: true` can run.

| shape | $/hr | GPUs | d40m tok/s | provisioned |
|---|---:|---:|---:|:--:|
| **`gpu-8xa100`** | 21.9576 | 8 | ~332,000 (est.) | yes |
| **`gpu-1xl40s`** | 1.8610 | 1 | **184,671 (measured)** | yes |
| `gpu-1xa10g` | 1.0060 | 1 | ~128,000 (est.) | yes |
| `gpu-1xh100` | 6.8800 | 1 | — | **no** |
| `gpu-8xh100` | 55.0400 | 8 | — | **no** |

Both H100 shapes are in the submission form's dropdown but their routing rows were
withdrawn from `config/execution-targets.yaml` on 2026-08-04, so selecting one fails
at admission. There is **no single-GPU A100** — `gpu-8xa100` is the only A100 shape.

`gpu-1xl40s` is the honest fallback rather than the cheaper `gpu-1xa10g`, because the
L40S is the card the `d40m` throughput figure was actually measured on. The A10G and
A100 numbers are bandwidth-scaled estimates (600 / 1555 GB/s against the L40S's 864).

## Why A100 for wall-clock, and the shape of the win

Because there is no 1×A100, using A100s means **one independent arm per GPU on a
`gpu-8xa100` node** — 8 cells finishing in the wall-clock time of one.
`.edullm/train.py` spawns those processes itself, which requires
`EDULLM_LAUNCH_CHECK=waived` (GPUs > 1 with no recognised launcher is otherwise
refused as `process_per_device`; the guard reads command text and cannot see
processes a program starts).

| plan | wall-clock for 12 cells | cost |
|---|---:|---:|
| **2 × `gpu-8xa100`, concurrent** | **~1.7 h** | ~$75 |
| 12 × `gpu-1xl40s`, concurrent | ~3.0 h | ~$67 |
| 12 × `gpu-1xa10g`, concurrent | ~4.3 h | ~$52 |
| Colab A100, sequential | ~26 h | subscription |

Note the last row: **Colab is the worst option for wall-clock**, not the best. One
interactive session runs 12 arms in series; the platform runs them in parallel. Per-GPU
speed is not the binding constraint — concurrency is, and the platform has plenty
(8 concurrent `gpu-8xa100`, 24 for `gpu-1xl40s`, 96 for `gpu-1xa10g`; 12 cells is
well inside all three).

## Iteration is local; the only slow step is the image build

`edullm check` is a **local lint** — "reaches no network", ~0.2 s, no cost, no queue.

| step | where | time |
|---|---|---|
| 101 tests | CPU | 28 s |
| calibration gate | CPU | ~20 s |
| corpus build, 3M-token smoke | CPU | ~60 s |
| `edullm check --json` | local | 0.2 s |
| **image build on push to `edullm/**`** | CI | **8–11 min** |

Batch Dockerfile and dependency changes; don't iterate through them.

Walltime is a **cap billed for actual time**, not a reservation — the platform's own
notification reads *"$0.02 spent, $2.01 authorised · ran 1m."* But **approval routing
uses the worst case** (rate × hours × attempts × cells), so on `gpu-8xa100` a generous
`--hours` climbs fast: 9 h × $21.96 = $198, still under the $500 auto-approve bound,
but 24 h would not be. Keep `--hours` tight.

---

# The jobs

## Prerequisite P — RESOLVED: an independent branch of OLMo-core

**This is the operative path.** A repository of its own cannot be registered without
widening the ECR publisher role, whose IAM stack is applied from a laptop and whose
file is owned by the two platform admins. OLMo-core is already registered, so its ECR
repository, publisher-role trust, and `AWS_ECR_PUBLISHER_ROLE_ARN` all already exist.

It is also the established convention rather than a workaround: OLMo-core carries ~148
branches, dozens under `edullm/**`, and this project line is already among them --
`edullm/fact-crowding`, `edullm/memory-split-135m`, `edullm/p3-math-split`.

```bash
scripts/sync_to_olmo_core.sh            # dry run: runs the tests, prints the push
scripts/sync_to_olmo_core.sh --confirm  # pushes to edullm/memsplit-hop on OLMo-core
```

The branch carries **this** repository's history and nothing of OLMo-core's, so it can
never be fast-forwarded into their main by accident, and GitHub runs workflows from the
pushed branch's own tree -- so only our build-caller runs on it and OLMo-core's CI does
not. The branch name must match `edullm/**` or the build never fires.

Nothing needs registering, and **there is no platform pull request at all**, because
both workload profiles already exist:

| profile | hours | attempts | checkpoint | used for |
|---|---:|---:|---|---|
| `olmo-core-check` | 1 | 1 | none | Job 1 smoke |
| `olmo-core-train` | 24 | 2 | `resume_required: true` | Jobs 2-4 |

`olmo-core-train` declares `resume_required: true`, and we satisfy it for real --
`checkpoint_io` writes to `s3://` and `ResumeGuard` raises rather than silently
restarting. That is the contract a sibling repo declared and did not honour, gating its
load on `os.path.exists()` against an `s3://` URI so every retry repeated the run at
full price. With 2 attempts permitted, this matters.

**Submit against `OLMo-core`**, not `memsplit-hop` — that is what is registered and
what both profiles name.

<details>
<summary>The two paths that did not work, kept for the record</summary>

## Rejected: two onboarding paths, and P2 is much cheaper

The platform frames this choice itself. Both `edullm add --reason` and the
registration workflow's `reason` input ask for *"why this needs a repository of its
own **rather than a workload in an existing one**."* Those are the two paths.

### P2 (recommended): a workload in an already-registered repository

Cheapest by a wide margin, because it touches `config/**` only -- owned by **nine**
people (2 admins + 7 team leads) -- and needs **no ECR repository, no publisher-role
widening, no laptop deploy, and no image build of our own.** We ride the host repo's
existing image.

The registered repositories are `edullm-alt-cl`, `edullm-data`, `edullm-p1`,
`nested-learning`, `OLMo-core`, `olmo-eval-full`, `open-instruct-scored-rewards`.
`edullm-p1` is the natural host: it already runs a **custom non-OLMo entrypoint**
that spawns its own launcher, which is exactly our shape.

`--spec` makes this work without disturbing the host: multiple run specs per repo are
supported (`find_spec` walks upward from cwd, and `--spec <path>` overrides). So our
spec lives at `experiments/memsplit-hop/.edullm/run.yaml` inside the host repo and is
selected explicitly.

**Minimal version -- zero platform pull requests.** `edullm-p1-train` already allows
10 h with one attempt, which covers our ~1.7-4.3 h runs. Point our spec at it:

```yaml
schema_version: 1
workload_profile: edullm-p1-train      # already exists; no catalog change needed
suggested_compute: gpu-8xa100
command: >-
  bash -lc 'EDULLM_LAUNCH_CHECK=waived python experiments/memsplit-hop/.edullm/train.py
  "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR" --data-root <s3 prefix>'
```

**If a dedicated profile is wanted**, it is one row in `config/workload-catalog.yaml`:

```yaml
  - name: edullm-p1-memsplit-hop-train
    repository: edullm-p1
    maximum_runtime_hours: "9"      # string: a Decimal read from base-ten text
    maximum_attempts: 1
    checkpoint:
      interval_minutes: 20
      destination_prefix: "s3://sbsandbox-intern-edullm-outputs/teams/"
      resume_required: false
```

**The one thing to check first:** the host image must already carry our dependencies
(`torch`, `numpy`, `PyYAML`, `tiktoken`, `boto3`). If it does not, adding them means a
Dockerfile change in **someone else's repository**, which triggers a rebuild there --
so ask before doing it, and prefer vendoring nothing that needs a new dependency.

### P1: a repository of its own (what we attempted)

Two halves, and the second is the long pole. In order:

1. `memsplit-hop` must exist as a GitHub repo under `edu-llm` — it currently has no
   remote.
2. In this repo: `.edullm/Dockerfile` (done), a build-caller workflow on `edullm/**`
   plus `workflow_dispatch`, and `AWS_ECR_PUBLISHER_ROLE_ARN` set **by hand** as a
   repository variable (`gh variable set`). There is no org-level variable; missing it
   yields `publisher_role_arn_is_empty`, which silently published nothing for another
   repo for days.
3. Register:
   ```bash
   gh workflow run register-repository.yml --repo edu-llm/platform \
     -f repository=memsplit-hop \
     -f github_repository_id=$(gh api repos/edu-llm/memsplit-hop --jq .id) \
     -f reason="n-hop fact-externalisation training; custom entrypoint, 1 arm per GPU"
   ```
   It edits five platform files and pushes a branch but **does not open the PR** — it
   prints a compare URL and a body to paste. Then merge, then a role-holder must
   **deploy**.
   Then set the publisher role variable. **This ARN is not something you create** --
   it is the pre-existing role `sbsandbox-intern-edullm-ecr-publisher`, and the
   registration skill says so: *"The ARN is the one `infra/README.md` records for
   `sbsandbox-intern-edullm-ecr-publisher` … registering a repository does not create
   it."* Derive it with an AWS session:

   ```bash
   ACCT=$(aws sts get-caller-identity --query Account --output text)
   gh variable set AWS_ECR_PUBLISHER_ROLE_ARN --repo edu-llm/memsplit-hop \
     --body "arn:aws:iam::$ACCT:role/sbsandbox-intern-edullm-ecr-publisher"
   ```

   or read the `RoleArn` output of stack `sbsandbox-intern-edullm-ecr-publisher-iam`.

   **And this is why P2 is cheaper.** The publisher role's policy enumerates each ECR
   repository explicitly, so registering ours means editing
   `infra/iam/ecr-publisher-role.yaml` -- which is owned by the two admins only, and
   whose stack is *"Applied from: laptop"*. The workflow states it plainly: *"the
   publisher role widening is an IAM stack nobody can apply from CI at all."* So P1
   ends in a human with AWS credentials running a deploy, not in a merge.

4. **Hand-edit the generated catalog entry before opening the PR.** The tool hardcodes
   1 hour / 1 attempt and names it `<repo>-check`; `--hours 9` against a 1 h profile is
   refused with `runtime_above_the_workload_bound`. Add:
   ```yaml
     - name: memsplit-hop-train
       repository: memsplit-hop
       maximum_runtime_hours: "9"
       maximum_attempts: 1      # all compute environments are EC2, not SPOT
       checkpoint: null
   ```
   `config/**` is owned by nine people, so any admin or lead can review it. Precedent:
   `nested-learning-train` is 12 h while its generated comment still says one hour.

</details>

## Job 0 — corpus build (local, CPU, no platform)

**Tests:** that the corpus satisfies all 11 integrity gates before a GPU is touched.

```bash
MEMSPLIT_TOKENIZER=byte python scripts/calibrate_nhop.py \
    --depths 1 2 3 4 5 --n-entities 900 --n-items 1000     # must print "usable"

python scripts/build_corpus.py --out data/nhop_v1 \
    --n-entities 10000 --total-tokens 500_000_000 \
    --train-depths 1 2 3 --eval-depths 4 5 --n-eval 1000
```

Gates: populations disjoint · lane shares within 1% · all trained depths present ·
**no held-out depth in training** · compose depths balanced · equal mass for both
controls · dense supervises all · novel names *and values* disjoint · organizer covers
every eval hop · skip rate < 25%. The build aborts on any failure.

Then upload once, read from every run:

```bash
P=s3://sbsandbox-intern-edullm-outputs/teams/memory-split/datasets/memsplit-hop-v1
aws s3 cp data/nhop_v1/tokens.bin $P/
for c in dense split random_contig random_scatter; do
  aws s3 cp data/nhop_v1/weights.$c.bin $P/
done
aws s3 cp data/nhop_v1/organizer.jsonl $P/
aws s3 cp data/nhop_v1/organizer_novel.jsonl $P/
aws s3 cp data/nhop_v1/report.json $P/
aws s3 cp --recursive data/nhop_v1/eval/ $P/eval/
```

There is no `EDULLM_INPUT_PREFIX`; the team dataset prefix is the one location outside
a run's own output that a workload role may write. Submit everything with
`dataset_release=none`. Each run stages `tokens.bin` plus only the sidecars its cells
need, skipped by byte-count on a resumed attempt.

## Job 1 — smoke, single GPU, ~5 min, ~$0.15

**Tests:** the submission path end to end — custom non-OLMo entrypoint, S3 corpus
staging, S3 checkpoint write, and **resume across attempts**. Worth its own job
because every command in the platform's recorded evidence is `python -c` or
`python -m olmo_core.train`; we are the first custom entrypoint to run.

```bash
scripts/sync_to_olmo_core.sh --confirm   # pushes to OLMo-core's edullm/memsplit-hop
# wait 8-11 min for the image, published to OLMo-core's ECR repository

edullm check --json --team memory-split --experiment smoke --dataset none

MEMSPLIT_CONFIG=configs/smoke_d40m.yaml MEMSPLIT_CONDITION=dense MEMSPLIT_SEED=0 \
edullm submit --spec .edullm/run.single.yaml \
              --team memory-split --experiment smoke --dataset none \
              --compute gpu-1xl40s --hours 1
# repository resolves to OLMo-core, from the workload profile the spec names

edullm status && edullm logs <run-id>
```

**Read off it:** `tok_s` from `log.jsonl` — this replaces the estimate and re-costs
everything below. Also confirm `attempt 1` in stdout and `ckpt.pt` in
`$EDULLM_CHECKPOINT_DIR`. Then **resubmit the identical command** and confirm it prints
`resumed at step N` and `attempt 2`. If it prints `attempt 2, starting at step 0` the
resume path is broken and the ResumeGuard should have raised — investigate before
spending anything larger.

## Job 2 — 8-cell A100 smoke, ~10 min, ~$4

**Tests:** the multi-cell path specifically — that 8 arms land on 8 distinct GPUs, that
each writes to its own `cell-<i>/` checkpoint prefix, that the corpus is staged **once**
by the parent rather than 8 racing downloads, and that the launcher waiver is accepted.

```bash
MEMSPLIT_CONFIG=configs/smoke_d40m.yaml \
MEMSPLIT_CELLS="dense:0,split:0,random_contig:0,random_scatter:0,dense:1,split:1,random_contig:1,random_scatter:1" \
edullm submit --team memory-split --experiment smoke8 --dataset none \
              --compute gpu-8xa100 --hours 1
```

**Read off it:** eight `[cell i] gpu=i ...` lines, one `staged [...]` line (not eight),
eight `cell-*/ckpt.pt` objects, and `all 8 cells complete`. If any cell fails the job
exits non-zero by design — a matrix with a missing cell is not a matrix.

## Jobs 3–4 — the depth matrix, 2 × 8-GPU A100, ~1.7 h wall, ~$75

**Tests the actual hypothesis.** 4 conditions × 3 seeds over trained depths {1,2,3},
evaluated at held-out depths {4,5}.

- `dense` vs `split` — the treatment.
- `random_contig` and `random_scatter` — the two equal-mass controls. Both mask exactly
  as many tokens as `split` does, on non-fact spans. Contiguous matches span structure
  and under-matches difficulty by 23–35%; scattered matches difficulty to 5–10% and
  gives up contiguity. If the effect survives **both**, it is not a masking artifact.
- 3 seeds, each randomising init *and* data order.

```bash
CONDS="dense split random_contig random_scatter"

# Job 3: seeds 0 and 1  (8 cells)
MEMSPLIT_CELLS=$(for s in 0 1; do for c in $CONDS; do printf "%s:%s," $c $s; done; done | sed 's/,$//') \
edullm submit --team memory-split --experiment depth-v1 --dataset none \
              --compute gpu-8xa100 --hours 4

# Job 4: seed 2  (4 cells, 4 GPUs idle)
MEMSPLIT_CELLS=$(for c in $CONDS; do printf "%s:2," $c; done | sed 's/,$//') \
edullm submit --team memory-split --experiment depth-v1 --dataset none \
              --compute gpu-8xa100 --hours 4
```

Both submit at once and run concurrently (8 concurrent `gpu-8xa100` permitted), so
wall-clock is one run's duration. Job 4 leaves 4 GPUs idle — that is ~$18 of waste,
which is cheaper than the alternative of a 12-cell fan-out, because **a fan-out always
routes to a team lead regardless of cost** while each of these self-approves under $500.

**Safety fallback**, if `gpu-8xa100` is unobtainable — same 12 cells, 12 single-GPU
jobs, ~3.0 h wall-clock on measured throughput:

```bash
for c in dense split random_contig random_scatter; do for s in 0 1 2; do
  MEMSPLIT_CONDITION=$c MEMSPLIT_SEED=$s \
  edullm submit --spec .edullm/run.single.yaml \
                --team memory-split --experiment depth-v1 --dataset none \
                --compute gpu-1xl40s --hours 5
done; done
```

## Job 5 — evaluation, single GPU, short

**Tests:** the depth curve against the *pⁿ* null, which is the whole point. Evaluation
is single-pass generation over ~15 strata × 1000 items, so it needs neither 8 GPUs nor
9 hours. Run it per snapshot, or pull snapshots down and run locally.

The eval driver is **not yet written** — it is the one remaining gap. What it must
report, per arm per depth: end-to-end accuracy, **per-hop conditional accuracy**, the
*pⁿ* prediction from each arm's measured per-hop reliability, and the store counters
(`n_query_spans` separately from `n_lookups`, so store-detached addressing stays
visible).

---

## Order of operations

1. Job 0 locally — calibration `usable`, corpus gates all OK, upload.
2. Prerequisite P — repo, Dockerfile, publisher role, registration PR, **deploy**.
3. Job 1. Read `tok_s`. Re-cost. Confirm resume on a second submission.
4. Job 2. Confirm 8 cells, 8 GPUs, one staging, per-cell checkpoints.
5. Jobs 3–4 together.
6. Job 5 once the eval driver exists.

Steps 1 and 2 are independent and can proceed in parallel — the registration PR does
not need the corpus, and the corpus does not need the platform.
