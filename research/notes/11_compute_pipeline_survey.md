# 11. Compute pipeline survey of the `edu-llm` GitHub org

Surveyed 2026-10-04, read-only and remote only (`gh api`, nothing cloned or downloaded). Scope: every repo in `edu-llm` except `platform` and `OLMo-core`. Branches other than the default were read where they held the relevant material (noted inline).

## TL;DR

- **The org ran at least five different compute paths, not one.** (1) the AWS `platform` (`edullm` CLI, GitHub `workflow_dispatch` to Step Functions admission to AWS Batch, plus a separate "Block:" lane for EC2 capacity-block fleets); (2) MIT ORCD Engaging via SLURM (an earlier GitHub-Issue-queue "operator" system, plus plain `sbatch`); (3) Stanford FarmShare via SLURM over an SSH control socket; (4) RunPod 8xA100 pods; (5) hand-launched EC2 (capacity-block `p6-b200`/`p5`, `g6.xlarge` via SSM, a shared p6 box). AI2 Beaker configs exist only as upstream leftovers in forks.
- **The platform contract is concrete and visible from consumer repos.** A registered repo ships `.edullm/Dockerfile` (`ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`, no default), a caller workflow that `uses: edu-llm/platform/.github/workflows/build-research-image.yml@main` on pushes to `edullm/**` or `main`, and `.edullm/run.yaml` (`schema_version`, `workload_profile`, `suggested_compute`, `command`). The container gets `EDULLM_RUN_ID`, `EDULLM_CHECKPOINT_DIR`, `EDULLM_OUTPUT_PREFIX` (S3 under `sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/`), and the checkpoint path must appear literally in the command text.
- **The platform is retired in practice.** `edullm-p1` commits on 2026-09-22 say "the platform is no longer used" and "drop the retired platform's agent layer", and the 2026-09-23/24 commits strip AWS credential helpers, S3 transfer workers and the RunPod paths. P1's final reported runs were re-run on FarmShare 4-8x L40S with W&B as the artifact store.
- **Data access standard: `edullm-data`.** Datasets are addressed as `<family>/<name>/<version>` in `s3://edullm-data/`, written only through an airlock validator (AWS Batch), and read with `edullm_data.read.dataset_paths()` / `resolve_latest()` / `build_mixture()`. Its own HANDOFF says no platform training job ever read it (IAM grant missing). With AWS frozen, `s3://edullm-data` reachability is an open question.
- **ORCD is MIT's Office of Research Computing and Data (Engaging cluster, `orcd-login.mit.edu`, partitions `mit_normal_gpu`, `mit_preemptable`).** The ORCD operator code survives only in `educompute` archive branches and in `p7stuff`; Amy Lin (`alsy7009`) is listed as an assignment-enabled operator. `grpo-tutor` ran on ORCD H100s; `p7stuff` SFT ran on ORCD L40S.
- **W&B is the one constant across every path.** Entity `eduLLM`; projects seen: `test`, `pretraining`, `posttraining`, `evaluation`, `data-pipeline`, `grpo_tutor`, `skillit`, `token-selection`, `nested-learning-block`.
- **Not found:** no `devops-dashboard` repo is visible to this account (404), only a pointer to it in `repo-template/README.md`. No decommission MANIFEST, `plans/edullm-decommission-plan.md`, TASK-127, Max McCorkel mention, or named replacement pipeline turned up in any in-scope repo. `educompute` is readable but its `main` holds only a README saying EduCompute "was designed but never implemented" and its planning docs are "local-only".

## Repo table

| Repo | Purpose | Compute-relevant? | Launch mechanism |
| --- | --- | --- | --- |
| `educompute` (private, readable) | Archive of ORCD operator code; EduCompute planning placeholder | Yes (historical) | ORCD SLURM via `edullm run` operator CLI, GitHub Issue queue (archive branches only) |
| `repo-template` (private) | Secret-leak guardrails + sample OIDC deploy workflow | Partly | GitHub OIDC to `sbsandbox` CloudFormation deploy role; documents AWS SCP/boundary rules incl. Capacity Blocks |
| `edullm-p1` (public, "pending transfer") | P1 dataset pipelines + token-selection and Skill-DAG experiments | Yes, heavily | FarmShare SLURM (`submit_from_laptop.sh`), RunPod (`runpod/launch.sh`), formerly platform image; dataset builds via `.sbatch` |
| `edullm-data` | Dataset standard: publish/validate/read via landing-to-data airlock | Yes | Validator/fsck run as AWS Batch jobs fired by EventBridge; platform image build caller; Python reader library |
| `olmo-eval-full` (fork of `allenai/olmo-eval`) | Evaluation suite + eduLLM checkpoint-eval flows | Yes | Upstream Beaker/Modal launcher; platform-registered image; plan for EC2+SSM `g6.xlarge` retroactive checkpoint evals |
| `grpo-tutor` | GRPO teacher-tutoring stack | Yes | ORCD SLURM `sbatch` (`mit_preemptable`, 1x H100, `--requeue`), W&B |
| `open-instruct-scored-rewards` | open-instruct + scored rewards for GRPO | Yes | Platform `.edullm/run.yaml` (`gpu-8xa100`, `accelerate launch` SFT then DPO); upstream Beaker/mason configs unused |
| `open-instruct` (fork) | Upstream AI2 post-training code | Marginal | Upstream Beaker/`mason.py`; branch `edullm/add-research-image` suggests platform image work |
| `edullm-alt-cl` | OLMo2-370M math front-load + anneal + SFT experiment | Yes | Platform-ready (`.edullm/` + `guides/platform-submit.md`), never registered; local `launch/launch_arm.sh` |
| `memsplit-hop` | MemorySplit methodology reset | Yes | Platform `.edullm/run.yaml` (`olmo-core-train`, `gpu-8xa100`), image built under OLMo-core's registration |
| `Memory-Split-P3` | MemorySplit results/code package | Yes | FarmShare SLURM (160M tier); EC2 Capacity Block `p6-b200.48xlarge` via `LAUNCH.sh` + SSM (1B tier) |
| `nested-learning` | Nested Learning / CMS on OLMoE-1B-7B | Yes | Platform "Block:" workflows on an 8x `p5.48xlarge` capacity block; generic SLURM arrays |
| `p3math` | Lean 4 facts-vs-reasoning loss-mask experiment | Yes | Manual run on a shared p6 B200 EC2 box via `aws ssm start-session` |
| `p7stuff` | OLMo-core copy with P7 tutor-layer POC | Yes | ORCD operator system (same as `educompute` archive) + direct ORCD `sbatch` |
| `tokenizer-flores-validation` | Multilingual tokenizer efficiency benchmark | Minor | EC2 `r7i.2xlarge` via PowerShell launcher; Docker image documented as an AWS Batch entrypoint |
| `edullm-token-selection` | Earlier token-selection code | Minor | Direct `torchrun` on a host with an idle GPU; reads `s3://edullm-dataset-regmix/...` |
| `sbsandbox-oidc-smoke` (private) | Smoke test of GitHub OIDC to `sbsandbox` | Infra only | GitHub Actions assumes `InternGitHubActionsDeploy-sbsandbox` |
| `dolma` (fork) | Upstream data toolkit | No | n/a |
| `p1-qa-results` | P1 QA augmentation results package | No | n/a (results only) |
| `p1-scaffolding-results` (private) | P1 worked-examples results package | No | n/a (results only) |
| `tutor-review` | Static human-review app for `grpo_tutor` dialogues | No | GitHub Pages |
| `demo-repository` (private) | GitHub demo boilerplate | No | n/a |

## Detailed findings

### 1. The AWS platform contract, as seen from consumer repos

`platform` itself was out of scope, but five repos carry its contract.

**The rule it distributes.** `olmo-eval-full/AGENTS.md` (also merged into `edullm-data` and `edullm-alt-cl` via `edullm/agent-layer` on 2026-08-06) carries a block "Managed by edu-llm/platform":

> "## Running anything on a GPU: use `edullm`, never AWS ... **Do not write a script that calls AWS.** No `boto3`, no `aws` CLI ..."

Install: `uv tool install --force git+https://github.com/edu-llm/platform`. Verbs: `edullm check` (prices a submission, lists refusals, no network), `submit`, `status`, `logs`, `cancel`, `add`, `ask`, and `run`/`shell` ("Ungated, and no run anybody can cite"). Other rules in that block: "The platform takes a commit, not a working tree"; the image is built by pushing a branch named `edullm/<something>`; pass `--dataset none` when no corpus is read; write the dtype into the command text because a guard reads it (`bfloat16_not_in_the_hardware`).

**Image.** Every compliant `.edullm/Dockerfile` (`edullm-data`, `olmo-eval-full`, `edullm-alt-cl`, `open-instruct-scored-rewards`, `memsplit-hop`, `nested-learning`, and the vendored copy in `edullm-p1/experiments/token-selection/olmo_core_token_selection/Dockerfile`) starts with:

```
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
```

and says the platform's `verify_dockerfile_base` gate "refuses this file unless BASE_IMAGE is declared in the global scope with no default". The registered base is a bare Python image, so each repo pins torch etc. itself (e.g. `edullm-alt-cl`: `torch==2.9.0`, OLMo-core at commit `99e0009e...`, `edullm-data` tag `v0.6.3`, `boto3`, `botocore[crt]`). No `ENTRYPOINT`, because the run's `command` is passed as a container override.

**Build trigger.** `.github/workflows/edullm-platform-build.yml` (or `build-image.yml` / `build-research-image.yml`): on push to `edullm/**` and `main`, `uses: edu-llm/platform/.github/workflows/build-research-image.yml@main` with `repository`, `publisher_role_arn: ${{ vars.AWS_ECR_PUBLISHER_ROLE_ARN }}`, and a `test_command`. Must reference `@main` (OIDC trust policy) and the caller must grant `id-token: write`. `edullm-alt-cl/guides/platform-submit.md` adds: "Wait for the ECR vulnerability scan (~10 minutes) before submitting."

**Run spec.** `.edullm/run.yaml` (written by `edullm check` from `config/workload-catalog.yaml`):

```yaml
schema_version: 1
workload_profile: olmo-core-train          # memsplit-hop
suggested_compute: gpu-8xa100
command: >-
  bash -lc 'EDULLM_LAUNCH_CHECK=waived python .edullm/train.py "$EDULLM_RUN_ID"
  --save-folder "$EDULLM_CHECKPOINT_DIR" --data-root s3://.../memsplit-hop-v1/'
```

**Container env and outputs** (`edullm-alt-cl/guides/platform-submit.md`):

| Variable | Meaning |
| --- | --- |
| `EDULLM_RUN_ID` | Stable run id (Batch retries reuse it) |
| `EDULLM_CHECKPOINT_DIR` | `s3://sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/checkpoints/` |
| `EDULLM_OUTPUT_PREFIX` | `s3://sbsandbox-intern-edullm-outputs/teams/{team}/runs/{run_id}/` |

Plus an `EDULLM_DATASET_*` triple when a `dataset_release` is chosen. "Checkpoints **must** go to `"$EDULLM_CHECKPOINT_DIR"` on the command line. The platform checkpoint guard reads the command text." Configs disable checkpoint pruning "so OLMo-core does not prune objects the workload role cannot delete". Submission form fields: `repository`, `commit_sha`, `dataset_release`, `team`, `experiment`, `wandb_project`, `command`. Unregistered repos are refused as `unregistered_repository`; `gpu-8xa100` is "an **admin exception** (>$20/hr)". `open-instruct-scored-rewards/.edullm/run.yaml` notes the platform "exposes the last fifty lines the container printed and nothing above them", so they redirect each stage to a log and `tail -40` it on failure.

**Capacity limits.** `edullm-data/docs/PLATFORM-INTEGRATION.md` (2026-07-30): the only provisioned Batch GPU was "a single A10G", `gpu-8xa100` raised `UnprovisionedComputeProfileError`; the GPU workload role could not read `s3://edullm-data`. `open-instruct-scored-rewards` later cites real 8xA100 Batch runs (`run_019fe311-ed20`), so 8xA100 was provisioned at some point after that.

**Block lane (capacity blocks).** `nested-learning/GITHUB_BLOCK_12H_RUN.md` lists `platform` workflows `block-launch-fleet.yml`, `block-status.yml`, `block-run.yml`, `block-run-distributed.yml`, `block-logs.yml`, `block-drain.yml`, `block-release.yml`. Fleet inputs: `reservation_id: cr-...`, `instance_type: p5.48xlarge`, `instance_count: 8`, `region: us-east-2`, `outputs_bucket: edullm-block-outputs-us-east-2`, `data_bucket: edullm-data-us-east-2`, `image_tag: <approved OLMo-core image tag>`. `block-run.yml` "clones a public repository and branch onto the node"; `block-run-distributed.yml` "appends `torchrun`" and "expects a Python training entrypoint, not a shell helper".

### 2. `educompute` and the ORCD operator system (MIT Engaging)

`educompute` `main` is one README: "EduCompute was designed but never implemented ... there is no EduCompute runtime code". Future integration would be "an optional adapter that submits through the platform's canonical run manifest". Planning docs "are kept local-only and are **not** committed here". So neither the EduCompute plan nor the decommission plan is reachable here.

The two archive branches (`archive/edullm-operator-0ee6359`, `archive/edullm-operator-wandb-upload-wip`) preserve the "eduLLM/ORCD **operator**" code that lived in `edu-llm/OLMo-core` and was "being retired". The same code is in `p7stuff` `main`. Key files: `src/edullm/{cli,jobs,slurm,ssh}.py`, `config/edullm/{entrypoints,policy,operators}.yaml`, `docs/edullm-team-workflow.md`, `docs/source/guides/edullm_engaging.rst`, `docs/superpowers/specs/2026-07-22-orcd-job-pool-design.md`.

Flow (`docs/edullm-team-workflow.md`): researcher pushes a clean non-main commit to `edu-llm/OLMo-core`, runs the `/submit-edullm-job` skill, which opens a validated GitHub Issue; Actions assign an operator and ping Slack; the operator runs `edullm run`, which SSHes (ControlMaster) to Engaging and runs exactly one `sbatch`. Policy (`config/edullm/policy.yaml`): `slurm_partition: mit_normal_gpu`, `max_runtime_minutes: 360`, `max_gpu_count: 2`, `allowed_gpu_preferences: [any, l40s, h100, h200]`, `scratch_root: $HOME/orcd/scratch/edullm`, W&B entity `eduLLM` with projects `test`, `pretraining`, `posttraining`, `evaluation`, `data-pipeline`. Entrypoints are whitelisted in `entrypoints.yaml` (e.g. `generic-smoke` = `src/examples/llm/train.py` under `torchrun`; `worked-examples-cpt` reads `/orcd/pool/edullm/data/...` and `/orcd/pool/edullm/checkpoints/...`). The generated job script (`src/edullm/slurm.py`) activates `$HOME/venvs/edullm`, checks out the exact SHA into scratch, and exports `EDULLM_REQUEST_ID`, `EDULLM_COMMIT_SHA`, `EDULLM_SEED`, `EDULLM_DATA_MANIFEST(_SHA256)`, `WANDB_RUN_ID` etc. No container: Apptainer and S3 were listed under "Deferred". `edullm_engaging.rst` names "Meric (`meric233`) and Amy Lin (`alsy7009`)" as assignment-enabled operators.

### 3. `repo-template`: what every repo must ship

Only guardrails, not a compute contract: `.gitignore`, `.gitleaks.toml`, `.pre-commit-config.yaml`, `.github/workflows/gitleaks.yml`, a sample `.github/workflows/deploy.yml` (GitHub OIDC to `arn:aws:iam::056956104102:role/InternGitHubActionsDeploy-sbsandbox`, CloudFormation/CDK/SST only, `us-east-1`/`us-east-2`), and `GUARDRAILS.md`. `GUARDRAILS.md` records that GPU `RunInstances` was exempted from the org SCP on 2026-07-23 and that "Capacity Blocks for ML are allowed (`ec2:PurchaseCapacityBlock`), they are PREPAID". The README points to "`plans/intern-github-org-bootstrap-runbook.md` in the devops-dashboard repo", which returns 404 for this account. The platform-ready files (`.edullm/*`, build caller) are not in the template; they were added per repo.

### 4. `edullm-p1` (ML experiments; pending transfer)

Two experiment families plus dataset builders; the most active repo (commits through 2026-10-04 by `nzhao721`).

- **Token selection** (`experiments/token-selection/olmo_core_token_selection/`, vendored byte-identical from `edu-llm/OLMo-core@53daffdf` `.edullm/`, see `PROVENANCE.md`). Two launch adapters:
  - FarmShare: `farmshare/submit_from_laptop.sh` syncs code, pushes a `wandb-session.env`, then over SSH socket to `login.farmshare.stanford.edu` submits `stage_job.sbatch` then `train_job.sbatch` (`--partition=gpu --qos=gpu --gpus-per-node=8 --mem=384G --time=72:00:00`). `config.env` holds `DATASET_ID="pretrain/regmix-10b"`, `DATASET_VERSION="v1"`. `common.sh` loads `python/3.12.3` and `cuda/12.4.0`, forces `OLMO_ATTN_BACKEND=torch`, and moves every cache to scratch because "$HOME is quota-limited". The reported FarmShare runs used 4x L40S (`TRAIN_GPUS=4`).
  - RunPod: `runpod/README.md`, 8x A100-80GB with image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `stage_inputs.py` pulls the sealed corpus from `s3://edullm-data/` and reference checkpoints from `s3://edullm-checkpoints/...`, writes `ready.json`, then `ARM=attention bash .edullm/runpod/launch.sh`.
  - The entrypoint reads `EDULLM_CHECKPOINT_DIR`, `EDULLM_DATASET_VERSION`, `EDULLM_RUNPOD_INPUT_MANIFEST`, `EDULLM_RESUME_CHECKPOINT`, `WANDB_RESUME_ARTIFACT`; production mode asserts world size 8 and online W&B. `production_contract/wandb_artifacts.py` logs checkpoints as W&B `model` artifacts and evals as `eval` artifacts ("checkpoint durability is fail-closed").
  - Arm YAMLs (`*/configs/run_*_10b.yaml`) specify `data.dataset_id`, `olmo_core.repository` + `revision`, `seed`, `train.*`; the README notes the pinned revision is stale and `PROVENANCE.md` is authoritative.
- **Skill-DAG / Skill-It** (`experiments/skill-dag/skillit/olmo_core_skillit/farmshare/`): `launch_no_aws.sh` + `train_no_aws.sbatch` at 4x L40S, "FarmShare's per-user GPU quota caps at 4 concurrent GPUs"; reads a local `ready.json` with no cloud access. `HANDOFF.md` explains running on a second student's FarmShare account against data on a collaborator's scratch.
- **Datasets** (`datasets/{olmo,olmohq,refhq,refhq_new,regmix}/`): SLURM `.sbatch` download/tokenize/publish jobs that publish to `edullm-data`; `datasets/manifests/*` allow byte-for-byte rebuilds from Hugging Face.
- **Platform retired here.** Commit messages: "Drop the eduLLM-platform image contract test; the platform is no longer used" and "drop the retired platform's agent layer" (2026-09-22); "Remove the AWS-only plumbing: credential helpers and S3 transfer workers" (2026-09-23); "Drop the RunPod MixLaw launch path" (2026-09-24); "rewrite the repo docs around the seven final all-L40S runs" (2026-10-01).

### 5. `edullm-data`: dataset standard and how jobs read data

`README.md`, `HANDOFF.md` (2026-07-30), `docs/CONSUMER-CONTRACT.md`, `docs/PLATFORM-INTEGRATION.md`. Producers `publish()` to `s3://edullm-landing/`; an EventBridge rule launches a Batch validator (`edullm-validator:5`, plus `edullm-fsck:4`) that runs gates and promotes into `s3://edullm-data/<family>/<name>/<version>/`. A bucket policy (`edullm-data-airlock-v1`) lets only the validator write. Families: `curriculum`, `eval`, `pretrain`, `probe`, `sft`, `tokenizer`, `vendor`. Published: `pretrain/olmo-150b-dolma2/v1` (157.5B dolma2 tokens, 6,851 train + 60 val shards) and `tokenizer/dolma2-bpe/v1`; P1 also used `pretrain/regmix-10b/v1` and `pretrain/refhq-instruct/v3`.

Read side:

```python
from edullm_data.read import dataset_paths, resolve_latest, build_mixture
r = dataset_paths("pretrain/olmo-150b-dolma2", "v1")   # trainable splits only
r.dtype  # "uint32": pass explicitly, OLMo-core defaults to uint16
```

`build_mixture(dataset, version, sources, ratios, total, seed)` gives a reproducible shard subset. Consumer rules: use plain `NumpyFSLDatasetConfig` with explicit dtype, not `SourceMixtureDatasetConfig`; credentials come from the ambient AWS identity. HANDOFF: "no training run has read it, and the four things blocking that live in `edu-llm/platform`". Publishing must run in-region on Batch ("Ran the publish on the LAPTOP ... a 9-day ETA").

### 6. `olmo-eval-full`: evaluation submission

Fork of `allenai/olmo-eval`; `main` is "upstream sync base + eduLLM platform registration" (`.edullm/Dockerfile`, build caller). The launcher code is upstream Beaker (`src/olmo_eval/launch/beaker/`, `examples/beaker/configs/*.yaml`) and Modal (`launch/modal/`), which the team cannot use. On branch `CheckpointFlows`, `AWS_CHECKPOINT_EVAL_PLAN.md` (decision 2026-08-05): "evals run retroactively, in a batch, not during training. Training's only job is to write checkpoints to S3." The plan uses self-terminating `g6.xlarge` (1x L4) EC2 workers from DLAMI `ami-0b6f2229ad14c9323`, instance profile `EswManagedInstance`, driven over SSM, at most 3 GPU boxes, reading `s3://<bucket>/checkpoints/<run>/step_N/` and writing `s3://<bucket>/olmo-eval-results/<run>/step_N/`; credentials from the `sb-aws-creds` broker. It names a team runbook (`AWS run book.md`, `eduLLM-Evals/scripts/aws/RUNBOOK_rung1_smoketest.md`) that is not in any in-scope repo. `edullm-alt-cl` instead submits `olmo-eval` against the registered `olmo-eval-full` image with `--output-dir "$EDULLM_OUTPUT_PREFIX/eval"`.

### 7. `grpo-tutor` and `open-instruct-scored-rewards`: GPU runs for RL

- `grpo-tutor`: plain SLURM on ORCD Engaging. `scripts/train_v3.sbatch`: `--partition=mit_preemptable --gres=gpu:h100:1 --time=6:00:00 --mem=128G --requeue --open-mode=append`, then `source scripts/env.sh` (activates `~/venv_test`, routes HF/vLLM/Triton/W&B caches to `$SCRATCH`, `WANDB_ENTITY=eduLLM`) and `python -u src/train_h100.py ... --save-every 10 --seed 0 --wandb`. README: "Runs on 1x H100 with vLLM", "preemption-safe checkpoint/resume, and a wandb run that stays continuous across Slurm requeues." Checkpoints stay in `checkpoints/` on the cluster. No platform dependency.
- `open-instruct-scored-rewards`: platform-registered (`.edullm/run.yaml`, `workload_profile: open-instruct-scored-rewards-train`, `suggested_compute: gpu-8xa100`). Command: `accelerate launch --mixed_precision bf16 --num_processes 8 --config_file configs/ds_configs/deepspeed_zero3.yaml open_instruct/finetune.py ...` then `open_instruct/dpo_tune_cache.py`, with `--push_to_hub false --try_launch_beaker_eval_jobs false` and `REFERENCE_LOGPROBS_CACHE_PATH` moved off the AI2 `/weka` default. Deps from `projects/pedagogy_rm/scripts/requirements-cluster.txt`. Flags are statically checked by `.edullm/validate_smoke_flags.py`.

### 8. Other compute-relevant repos

- `memsplit-hop`: `.edullm/run.yaml` + `run.single.yaml`, `.edullm/train.py`; build caller passes `repository: OLMo-core` ("Ride OLMo-core's registration instead of our own"); data under `s3://sbsandbox-intern-edullm-outputs/teams/memory-split/datasets/memsplit-hop-v1/`.
- `Memory-Split-P3`: `160m-farmshare/cluster/` (SSH ControlMaster to `rice-04.farmshare.stanford.edu`, `slurm/train_single.sbatch` with `--partition=gpu --qos=gpu --gres=gpu:1 --requeue`, `scripts/run_train.py --resume auto`); `1b-b200/launch/LAUNCH.sh` does `aws ec2 run-instances --instance-type p6-b200.48xlarge --instance-market-options MarketType=capacity-block` then verifies the instance consumed the reservation "rather than silently falling through to on-demand at ~$99/hr"; staging and an S3 checkpoint sync daemon run over SSM.
- `nested-learning`: platform Block lane (above), plus generic `scripts/run_*_ablation_array.slurm` (`--gres=gpu:8 --array=0-35`), cluster unnamed.
- `p3math`: `scripts/smoke_gpu.sh`, "RUN THIS ON THE SHARED p6 GPU BOX ONLY (via `aws ssm start-session ...`)", picks idle GPU indices by hand, follows an unlisted `training_info.md`.
- `p7stuff`: `p7/POC/ORCD-SFT/run_sft.sbatch` (`-p mit_normal_gpu -G l40s:1 -t 04:00:00`, conda from `$HOME/miniforge3`), `ORCD_RUN.md` (login `orcd-login.mit.edu`, Kerberos + Duo). `p7/POC/EDULLM-NEW-WORKSPACE-SETUP.md` references workspace-only skills `edullm-aws-training` and `sb-aws-readonly`.
- `tokenizer-flores-validation`: `scripts/launch_plan_a_ec2.ps1` (`r7i.2xlarge`, profile `edullm-downloader`, `s3://edullm-datasets/_scratch/...`); `docker/tokenizer-benchmark/Dockerfile` described as "the AWS Batch entrypoint".
- `edullm-token-selection`: direct `python -m torch.distributed.run --standalone ... --launch` on a host with an idle GPU; reads `s3://edullm-dataset-regmix/regmix-10b/tokenized`.

## What a new experiment must provide to be launchable

Inferred from what the platform, ORCD operator, FarmShare and RunPod paths all needed. Meeting these keeps a P4 experiment portable to whichever pipeline the team ends up with.

1. **One non-interactive entrypoint.** `python <entry>.py <run_id> --config <file>` (or `torchrun ... <entry>.py`), with all science in a committed config. No notebook, no shell wrapper hiding the launcher; the platform's launch guard and the Block distributed lane both need the launcher visible in the command.
2. **A committed config file per arm** (YAML/JSON) holding seed, data id + version, model size, token budget, schedule, eval cadence. Hardware knobs (GPU count, microbatch size) stay overridable and out of the experiment fingerprint, as P1's token-selection YAMLs do.
3. **Exact code identity.** Run from a pushed commit SHA (all pipelines required a clean pushed commit; the platform builds from a commit on an `edullm/**` branch). Pin the OLMo-core revision.
4. **Pinned dependencies and an optional container.** `pyproject.toml`/`requirements.txt` with exact versions (torch, OLMo-core SHA, `edullm-data` tag). If containerized, a `.edullm/Dockerfile` with bare `ARG BASE_IMAGE` / `FROM ${BASE_IMAGE}`, no `ENTRYPOINT`, and its own torch install.
5. **Data by id, not path.** Read through `edullm_data.read.dataset_paths()` / `build_mixture()` with `(dataset, version, sources, ratios, total, seed)`, always passing `r.dtype`. Also support a local `ready.json` manifest path (the FarmShare `--no-aws` pattern) so the job runs where S3 is unreachable.
6. **Outputs through env vars.** Take run id, checkpoint dir and output prefix from env (`EDULLM_RUN_ID`, `EDULLM_CHECKPOINT_DIR`, `EDULLM_OUTPUT_PREFIX`), with the checkpoint dir passed literally on the command line; accept both `s3://` and local paths. Keep checkpoint pruning off unless the role can delete.
7. **Resumable and preemption-safe.** `--resume auto` from the latest checkpoint; tolerate `--requeue` (ORCD `mit_preemptable`, FarmShare) and Batch retries reusing the same run id.
8. **W&B wiring in code.** Entity `eduLLM`, a named project, group = study, deterministic run id, the scientific metric emitted by the training code. Optionally log checkpoints and evals as W&B artifacts (P1's fallback when S3 was gone).
9. **Explicit precision, caches on scratch, readable failure output.** Put `bfloat16` in the command; route HF/Triton/Inductor/W&B/vLLM caches to scratch; on failure print the trainer's own last lines (the platform only kept 50).
10. **A dry-run / smoke mode.** A `--dry-run` that prints the resolved plan and a tiny-step smoke config (the `generic-smoke`, `hypothesis-smoke` and `--dry-run` patterns), so a run can be checked before it costs GPU time.
11. **No secrets in git.** Keep `repo-template`'s gitleaks hooks; credentials come from the environment (W&B key, AWS session) and are never committed.

## Open questions for the team

1. With the AWS platform frozen since 2026-08-18, where should P4 run: ORCD Engaging (`mit_normal_gpu` caps at 2 GPUs / 6 h; `mit_preemptable` H100s), Stanford FarmShare (4-GPU per-user cap), RunPod, or a new AWS season?
2. Is `s3://edullm-data` still readable after the freeze, and from where (only `sbsandbox` roles, or exported to an ORCD/FarmShare filesystem)? Will `edullm-data` stay the dataset standard?
3. Where do `plans/edullm-decommission-plan.md`, TASK-127 and the decommission MANIFEST live? They are not in any in-scope repo, and `devops-dashboard` is not visible to this account. Can Max McCorkel grant read access or share them?
4. Is there a named replacement pipeline? `educompute` says EduCompute was never implemented, and its planning docs are local-only.
5. Is the ORCD operator flow (GitHub Issue queue + `edullm run`) retired with OLMo-core's copy, or could it come back? Amy is still listed as an operator in the archived roster.
6. When `edullm-p1` moves into the org, will its FarmShare + W&B-artifact path become the reference for small (370M, L40S) runs?
7. Should evaluation follow `olmo-eval-full`'s retroactive flow (checkpoints first, batch eval later) and on which hardware, now that its EC2+SSM plan depends on the frozen `sbsandbox` account?
8. Which W&B project should P4 use under the `eduLLM` entity, and who controls access?
