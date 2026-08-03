# Token selection on RunPod (8 × A100)

This additive adapter supports the five approved token-selection arms. It stages
the selected sealed corpus and, when required, RefHQ inputs before training.
Frozen RefHQ DistCP checkpoints are materialized locally; the late reference
uses the source methodology's float32 accumulation over steps 1000, 1125, and
1315 followed by a cast to the source parameter dtype.

Use an 8 × `NVIDIA A100-SXM4-80GB` pod with
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` and at least 300 GB of
persistent `/workspace` storage. Do not put AWS or W&B secrets in the RunPod
template, API arguments, environment fields, or start command.

## Bootstrap

```bash
curl -fsSL https://raw.githubusercontent.com/edu-llm/OLMo-core/edullm/token-selection-370m/.edullm/runpod/bootstrap.sh |
  bash
```

Set `OLMO_CORE_COMMIT_SHA` first to require an exact commit.

## Stage one arm

Mint a temporary `sbsandbox` session on the engineer laptop, copy it over SSH,
and immediately delete the laptop copy:

```powershell
& C:\alpha_ai\edullm\scripts\farmshare\mint_aws_session_local.ps1 `
  -Profile sbsandbox -OutputPath $env:TEMP\aws-session-runpod.env
scp -P <ssh-port> $env:TEMP\aws-session-runpod.env root@<pod-host>:/workspace/aws-session.env
Remove-Item -Force $env:TEMP\aws-session-runpod.env
```

Attention example:

```bash
chmod 600 /workspace/aws-session.env
cd /workspace/OLMo-core
PYTHONPATH="$PWD/src:$PWD/.edullm" python3 .edullm/runpod/stage_inputs.py \
  --credentials-file /workspace/aws-session.env \
  --arm attention --dataset-version v1
```

For BLADE, also pass the pinned `--refhq-version`. Reference-dependent arms
automatically download only the required prefixes beneath:

```text
s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/
```

The staging process also reads the selected sealed corpus under
`s3://edullm-data/`. It verifies local object sizes, writes `ready.json` last,
and removes/unsets the temporary AWS session in a `finally` block. Restage
before switching arms.

Copy a separate mode-0600 `/workspace/wandb-session.env` over SSH:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='eduLLM'
```

## Launch

```bash
cd /workspace/OLMo-core
ARM=attention bash .edullm/runpod/launch.sh
```

Set `RECOVERY_MODE=resume` to restore that arm's local checkpoint and W&B run
identity.
Use `RECOVERY_MODE=retry-start` only for a failure before the first step
checkpoint; it removes only pre-checkpoint identity sidecars and preserves the
same W&B ID.
The wrapper preserves the original S3 URI list in the scientific fingerprint
while the loader reads byte-identical local files. The launcher refuses any
remaining AWS credential file or environment variable.

Every permanent checkpoint's complete 20-task evaluation is uploaded to W&B
as metrics and result artifacts. Intermediate checkpoints stay on local
persistent storage; only the final checkpoint is uploaded as a W&B model
artifact.
