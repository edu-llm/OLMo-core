# Three-arm HPO on RunPod (8 GPUs)

This is a local, uncommitted adapter for the three-arm HPO study on
`edullm/hpo-complex`. It stages the sealed `regmix-10b-v1` corpus while a
short-lived AWS session exists, deletes that session, and then runs the unchanged
HPO controller against local files and persistent `/workspace` storage.

Use an **8 × A100 80 GB** pod with at least 250 GB of persistent `/workspace`
storage. The bootstrap checks that exactly eight BF16-capable GPUs are visible.
Do not put AWS, W&B, or model-provider secrets in the RunPod template, API
arguments, environment fields, or start command.

## 1. Transfer this local adapter

These files intentionally are not committed. Copy the entire local directory to
the pod:

```powershell
scp -P <ssh-port> -r `
  C:\alpha_ai\OLMo-core-hpo-complex\.edullm\runpod `
  root@<pod-host>:/workspace/hpo-runpod-adapter
```

The bootstrap clones the pinned committed HPO implementation and overlays this
local adapter:

```bash
bash /workspace/hpo-runpod-adapter/bootstrap.sh
```

The default pin is `4f385fe54918b96756042a89d504ac19b928e1b4`. Set
`OLMO_CORE_COMMIT_SHA` only when this adapter has been reviewed against another
commit.

## 2. Stage the sealed corpus

Mint a temporary `sbsandbox` session on the engineer laptop, copy it over SSH,
then immediately delete the laptop copy:

```powershell
& C:\alpha_ai\edullm\scripts\farmshare\mint_aws_session_local.ps1 `
  -Profile sbsandbox -OutputPath $env:TEMP\aws-session-runpod.env
scp -P <ssh-port> $env:TEMP\aws-session-runpod.env `
  root@<pod-host>:/workspace/aws-session.env
Remove-Item -Force $env:TEMP\aws-session-runpod.env
```

On the pod:

```bash
chmod 600 /workspace/aws-session.env
cd /workspace/OLMo-core
PYTHONPATH="$PWD/src:$PWD/.edullm" \
  python3 .edullm/runpod/stage_inputs.py \
    --credentials-file /workspace/aws-session.env
```

The stager accepts only objects under `s3://edullm-data/`, writes
`/workspace/edullm-inputs/hpo-probe/ready.json` after all size checks pass, and
deletes `/workspace/aws-session.env` even on failure. Training refuses any
remaining AWS credential file or environment variable.

## 3. Install runtime secrets

Copy the same mode-0600 `wandb-session.env` file used by the curriculum RunPod
adapter to `/workspace/wandb-session.env`:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='...'
# Required only for full_acronym_soup and no_proxy:
export OPENAI_API_KEY='...'
# Optional override; defaults to https://gateway.truefoundry.ai/v1
# export OPENAI_BASE_URL='https://gateway.truefoundry.ai/v1'
```

Locally, the controller also reads `~/.wandb_api_key` and `~/.openai_api_key`
when the corresponding environment variables are unset (same pattern as
`.edullm/_tmp_clone_wandb_run.py`).

`WANDB_API_KEY` is required for every mode. `OPENAI_API_KEY` is not used by the
proxy cohort or `no_centaur`.

Centaur’s logical model remains the Brainlift id `gpt-5.6-sol`. The default
transport routes it through TrueFoundry as `openai-group/gpt-5.6-sol` on
`https://gateway.truefoundry.ai/v1`. Override with `OPENAI_BASE_URL` /
`advisor_kwargs.route_model` only when needed. The advisor fails closed if the
model or structured-output contract is unavailable.

```bash
chmod 600 /workspace/wandb-session.env
```

## 4. Launch

The launcher enforces one total four-hour wall-time limit, including final W&B
publication. Expected training wall time is about 1–1.5 hours.

First run the paired proxy cohort required by Arms 1–2:

```bash
cd /workspace/OLMo-core
MODE=proxy-cohort bash .edullm/runpod/launch.sh
```

Then run each arm in a separate slot:

```bash
MODE=full_acronym_soup RUN_SLOT=main bash .edullm/runpod/launch.sh
MODE=no_centaur        RUN_SLOT=main bash .edullm/runpod/launch.sh
MODE=no_proxy          RUN_SLOT=main bash .edullm/runpod/launch.sh
```

`no_proxy` can run before the cohort because it does not use proxy rankings.
`full_acronym_soup` and `no_centaur` refuse to start until shared
`proxy-evidence.json` exists.

## Recovery

Every job has persistent state under:

```text
/workspace/edullm-runs/hpo-probe/<mode>/<slot>/
```

Resume the same controller/checkpoint lineage and W&B identity with:

```bash
MODE=no_proxy RUN_SLOT=main RECOVERY_MODE=resume \
  bash .edullm/runpod/launch.sh
```

Fresh mode refuses existing state. Use a new `RUN_SLOT` for an independent run.
The proxy cohort can also be resumed; completed first-rung checkpoints are
reused by the segment workers.

## Artifacts

- Trial and exact-retrain checkpoints remain on persistent `/workspace`.
- Controller state, segment specs, study result, proxy evidence, and scalar
  statistics are mirrored to W&B project `hpo-probe`.
- After a successful arm, the final selected checkpoint and local run log are
  uploaded as W&B artifacts.
- The shared proxy evidence lives at
  `/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json`.

No production job is launched by bootstrap or staging. Only `launch.sh` starts
GPU training.
