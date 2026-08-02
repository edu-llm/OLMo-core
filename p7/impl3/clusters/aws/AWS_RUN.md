# Running P7 post-training on AWS

Same recipe as ORCD, on an AWS GPU box reached over SSM, with checkpoints streamed to
S3 so a run survives a box teardown and resumes elsewhere.

> **Approvals & guardrails (edu-llm skill).** Never act on AWS or push to GitHub
> without the training lead's explicit approval. The sandbox account is
> `056956104102` (`sbsandbox`); region-locked to **us-east-1/us-east-2** (default
> **us-east-2**). The permissions boundary **denies launching GPU instances** — GPU
> compute is provisioned through a separate DevOps channel; confirm the box with the
> lead. This runbook assumes a box already exists.

## Required tags (SEC-05)
Every AWS resource you create must carry all four tags or it is flagged for deletion:
`Project=edullm`, `Environment=ephemeral`, `ManagedBy=manual`, `Owner=<your-email>`.
S3 objects inherit the bucket; tag any new bucket at creation.

## S3 layout
- `s3://alphaai-edullm-data`         — datasets (from the data team; blank for us now).
- `s3://alphaai-edullm-checkpoints`  — model outputs. This project writes under
  `s3://alphaai-edullm-checkpoints/p7/<user>/<run>/`.

## Step 1 — Connect (laptop)
```bash
export AWS_PROFILE=sbsandbox                       # LAPTOP ONLY
aws ssm start-session --target <instance-id> --region us-east-2
# on the box, become the training user, then:
unset AWS_PROFILE                                  # box uses its EC2 instance role
aws sts get-caller-identity                        # should show an assumed-role ARN, not a profile error
```

## Step 2 — Get the code + env (on the box)
```bash
# copy the post-training/ folder up (rsync over SSM tunnel, or git clone your branch), then:
cd ~/post-training
bash clusters/aws/setup_aws_env.sh                 # venv + deps; uninstalls torchao
```

## Step 3 — Train (inside tmux for long runs)
```bash
tmux new -s p7
source ~/p7post-venv/bin/activate
export S3_DEST=s3://alphaai-edullm-checkpoints/p7/$USER
export WANDB_API_KEY=<your key>          # W&B logging is ON by default (report_to: wandb)
# optional: export WANDB_PROJECT=edullm-p7  WANDB_ENTITY=<team>   (or --no_wandb to disable)

# Impl 2 baseline
export OUT_DIR=out/impl2-sft
export CMD="python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml --output_dir $OUT_DIR --resume auto"
bash clusters/aws/run_aws.sh

# Impl 3 one (variant, T)
export OUT_DIR=out/impl3-a-T2
export CMD="python impl3_kl_reweighted_sft/train_kl_sft.py --variant a --temperature 2 --output_dir $OUT_DIR --config impl3_kl_reweighted_sft/config.yaml --resume auto"
bash clusters/aws/run_aws.sh
```
Detach with `Ctrl-b d`; the run continues. `run_aws.sh` pulls existing checkpoints
from S3 first (so `--resume auto` works across boxes) and syncs back at the end.

## Multi-GPU
A 1B LoRA fits on one GPU, so single-GPU (`GPUS=0`) is the default and simplest. To use
several GPUs on one box, launch the entrypoint under `accelerate launch` /
`torchrun --nproc-per-node N` (the checkpoint-step auto-sizing reads `WORLD_SIZE`).
Only use idle GPU indices on a shared box (see the pretraining runbook's Phase 2).

## Stopping cleanly
Press `Ctrl-C` **once** and wait — the HF Trainer saves a checkpoint on SIGINT. Hitting
it twice can kill a checkpoint mid-write. `run_aws.sh` then syncs the saved state to S3.
