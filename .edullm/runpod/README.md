# Skill-It on RunPod (8 × A100)

These additive files run the Skill-It implementation with sealed inputs copied
to RunPod-local storage before training. Use an 8 ×
`NVIDIA A100-SXM4-80GB` pod, the
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image, and at least 250 GB of
persistent `/workspace` storage. Staging estimates each domain's unique-token
need from the initial Skill-It weights and adds 25% headroom for later updates,
instead of copying the full 506 GB corpus.

Never put AWS or W&B credentials in a RunPod template, API argument,
environment field, or start command.

## Bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/edu-llm/OLMo-core/edullm/skillit-370m/.edullm/runpod/bootstrap.sh |
  bash
```

Optionally set `OLMO_CORE_COMMIT_SHA` to require an exact branch commit.

## Stage the sealed corpus once

On the engineer laptop, mint a temporary `sbsandbox` session, copy it over SSH,
and immediately delete the laptop copy:

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
  --credentials-file /workspace/aws-session.env
```

This bounded phase reads only the labeled
`s3://edullm-data/pretrain/olmo-127b/v1/` objects. It checks each local size,
publishes `ready.json` last, and removes/unsets the temporary AWS session in a
`finally` block. Training performs no AWS access.

Copy `/workspace/wandb-session.env` separately over SSH with mode 0600:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='eduLLM'
```

## Launch

Probe:

```bash
cd /workspace/OLMo-core
ARM_INDEX=0 bash .edullm/runpod/launch.sh
```

Derivative:

```bash
ARM_INDEX=1 bash .edullm/runpod/launch.sh
```

Set `RECOVERY_MODE=resume` to resume from that arm's persistent local
checkpoint directory. The launcher refuses any remaining AWS credential file
or environment variable and keeps the production eight-rank and task-loss
contracts intact. Use `RECOVERY_MODE=retry-start` only when startup failed
before the first step checkpoint; it preserves the W&B ID and refuses once a
step checkpoint exists.

Every permanent checkpoint's complete 20-task evaluation is uploaded to W&B
as metrics and result artifacts. Intermediate checkpoints stay on local
persistent storage; only the final checkpoint is uploaded as a W&B model
artifact.
