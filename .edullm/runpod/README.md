# MixLaw on RunPod (8 × A100)

This directory is an additive RunPod adapter. Training uses the MixLaw
OLMo-core entrypoint with local paths substituted for the same sealed S3
objects.

## Pod

Create an 8 × `NVIDIA A100-SXM4-80GB` pod with the existing
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image. Attach at least 250 GB
of persistent `/workspace` storage. Staging selects a deterministic shard
prefix for each domain from the arm's requested token budget and weight, with
10% headroom, instead of copying the full 506 GB corpus. Checkpoints remain on
the volume. Do not put AWS or W&B
credentials in the pod template, API arguments, environment, or start command.

Bootstrap contains no credentials:

```bash
curl -fsSL https://raw.githubusercontent.com/edu-llm/OLMo-core/edullm/mixlaw-validation-370m/.edullm/runpod/bootstrap.sh |
  bash
```

Set `OLMO_CORE_COMMIT_SHA` before that command to require an exact commit.

## One bounded S3 staging phase

Mint a temporary `sbsandbox` session on the engineer laptop. Copy only that
file over SSH, then immediately remove the laptop copy:

```powershell
& C:\alpha_ai\edullm\scripts\farmshare\mint_aws_session_local.ps1 `
  -Profile sbsandbox -OutputPath $env:TEMP\aws-session-runpod.env
scp -P <ssh-port> $env:TEMP\aws-session-runpod.env root@<pod-host>:/workspace/aws-session.env
Remove-Item -Force $env:TEMP\aws-session-runpod.env
```

On the pod:

```bash
chmod 600 /workspace/aws-session.env
cd /workspace/OLMo-core
PYTHONPATH="$PWD/src:$PWD/.edullm" python3 .edullm/runpod/stage_inputs.py \
  --credentials-file /workspace/aws-session.env \
  --arm-index 0
```

The staging process reads only `s3://edullm-data/pretrain/olmo-127b/v1/`,
checks every selected object size, writes `ready.json` last, removes the pod
credential file in a `finally` block, and unsets its process-local AWS
variables. It does not write to S3. Pass the same `--arm-index` to staging and
training; add `--length-tokens` to both for a shortened run. If staging cannot
finish within one session lifetime, stop and increase transfer throughput or
reduce the bounded staging plan; do not refresh credentials during the job.

## W&B and launch

Copy a separate mode-0600 `/workspace/wandb-session.env` over SSH:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='eduLLM'
```

Then launch arm 0. Omit `LENGTH_TOKENS` for the full 10B-token run. Set
`EDULLM_WANDB_PROJECT` to override the default `mixlaw` project:

```bash
cd /workspace/OLMo-core
ARM_INDEX=0 EDULLM_WANDB_PROJECT=mixlaw-1 bash .edullm/runpod/launch.sh
```

`launch.sh` refuses to start if the AWS file or any AWS credential environment
variable remains. Valid arm indices are documented in `.edullm/MIXLAW.md`.
Use `RECOVERY_MODE=resume` to resume that arm's persistent local checkpoint and
the same W&B run identity. If startup failed before step 0 was checkpointed,
use `RECOVERY_MODE=retry-start`; it reuses-or-creates the same W&B ID but
refuses when any step checkpoint exists. Fresh launches refuse existing state.

Every permanent checkpoint (step 0, each 125-step interval, and the final
step) runs the complete 20-task OLMES BPB suite. Every metric and result JSON
is uploaded to W&B. Intermediate checkpoints remain local; only the final
checkpoint is uploaded as a W&B model artifact.
