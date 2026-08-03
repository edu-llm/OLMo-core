# Skill-It on FarmShare (8 × L40S)

Additive FarmShare bootstrap for the Skill-It OLMo-core branch. It mirrors the
RunPod adapter: stage sealed `s3://edullm-data/` inputs with a temporary AWS
session, delete credentials, then train with PyTorch SDPA (no FlashAttention).
The stager selects deterministic per-domain shard prefixes from the initial
weights with 25% headroom; later adaptive weight shifts sample from those
bounded local pools.

## Quick start (engineer laptop + WSL)

```bash
cd /mnt/c/alpha_ai/OLMo-core-skillit-370m
ARM_INDEX=0 bash .edullm/farmshare/submit_from_laptop.sh
```

Probe is `ARM_INDEX=0`, derivative is `ARM_INDEX=1`.

Prerequisites:

- FarmShare control socket (`/tmp/farmshare-nzhao2.sock`)
- `edullm` repo at `/mnt/c/alpha_ai/edullm` for `push_aws_session_to_farmshare.sh`
  and `push_wandb_session_to_farmshare.sh`
- W&B key file at `/mnt/c/Users/natha/.wandb_api_key` (or `WANDB_API_KEY` in env)

## Resource defaults

Override before submit:

```bash
export TRAIN_GPUS=8 TRAIN_CPUS=64 TRAIN_MEM=384G TRAIN_TIME=72:00:00
export STAGE_CPUS=8 STAGE_MEM=32G STAGE_TIME=06:00:00
```

Jobs always exclude `wheat-01`.

## Recovery

Re-submit with the same `RUN_DIR` and `RECOVERY_MODE=resume` (or `retry-start`
before the first checkpoint). Push a fresh `aws-session.env` only when restaging.

## W&B policy

Every permanent checkpoint uploads full 20-task eval metrics to W&B. Only the
final checkpoint is uploaded as a model artifact (branch code).
