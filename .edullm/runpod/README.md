# Curriculum on RunPod (8 × A100)

This additive adapter preserves the ten-arm curriculum implementation.
It resolves and downloads one arm's immutable parent/order inputs before
training, then substitutes those local files through a wrapper.

Use an 8 × `NVIDIA A100-SXM4-80GB` pod with
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and at least 250 GB of
persistent `/workspace` storage. Do not put AWS or W&B secrets in the RunPod
template, API arguments, environment fields, or start command.

## Bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/edu-llm/OLMo-core/edullm/curriculum-370m/.edullm/runpod/bootstrap.sh |
  bash
```

Set `OLMO_CORE_COMMIT_SHA` first if the checkout must match an exact commit.

## Stage one arm

Mint a temporary `sbsandbox` session on the engineer laptop, copy it over SSH,
then immediately delete the laptop copy:

```powershell
& C:\alpha_ai\edullm\scripts\farmshare\mint_aws_session_local.ps1 `
  -Profile sbsandbox -OutputPath $env:TEMP\aws-session-runpod.env
scp -P <ssh-port> $env:TEMP\aws-session-runpod.env root@<pod-host>:/workspace/aws-session.env
Remove-Item -Force $env:TEMP\aws-session-runpod.env
```

On the pod (`linear10-flesch` is arm 0):

```bash
chmod 600 /workspace/aws-session.env
cd /workspace/OLMo-core
PYTHONPATH="$PWD/src:$PWD/.edullm" python3 .edullm/runpod/stage_inputs.py \
  --credentials-file /workspace/aws-session.env --arm-index 0
```

For a paced arm, add its pinned `--curriculum-version`. Staging reads only
`s3://edullm-data/pretrain/regmix-10b/v1/` and, for paced arms, the selected
sealed `s3://edullm-data/curriculum/regmix-370m/<version>/` order. Control
(`--arm-index 5`) stages the parent only. It publishes `ready.json` only after
all size checks pass, then removes and unsets the AWS session. Restage when
changing arms.

Copy `/workspace/wandb-session.env` separately over SSH with mode 0600:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='eduLLM'
```

## Launch

```bash
cd /workspace/OLMo-core
ARM_INDEX=0 bash .edullm/runpod/launch.sh
```

For a short benchmark, set `LENGTH_TOKENS` to a valid global-batch multiple.
For recovery, use
`RECOVERY_MODE=resume LOAD_PATH=/workspace/edullm-runs/curriculum/<arm>/checkpoints`.
If startup failed before the first step checkpoint, use
`RECOVERY_MODE=retry-start` to preserve the W&B identity without pretending a
checkpoint exists.
The launcher refuses to train while any AWS credential remains and preserves
the eight-rank production task-loss contract.

Every permanent checkpoint's complete 20-task evaluation is uploaded to W&B
as metrics and result artifacts. Intermediate checkpoints stay on local
persistent storage; only the final checkpoint is uploaded as a W&B model
artifact.
