# HPO on RunPod (8 GPUs)

This adapter supports the historical three-arm HPO study and the separate
`curriculum_quadratic_mtld` extension on `edullm/hpo-complex`. It stages the
sealed inputs while a short-lived AWS session exists, deletes that session, and
then runs the unchanged HPO controller against local files and persistent
`/workspace` storage.

Use an **8 × A100 80 GB** pod with at least 500 GB of persistent `/workspace`
storage. Earlier 250 GB HPO pods filled during checkpoint writes after retries left multiple
slots behind. The launcher now requires 300 GiB free after staging by default and refuses before
training if that headroom is unavailable. The bootstrap checks that exactly eight BF16-capable
GPUs are visible.
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

## 2. Stage the sealed releases

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
    --credentials-file /workspace/aws-session.env \
    --release-set curriculum
```

For either new curriculum arm, `--release-set curriculum` resolves one schema-v2 local manifest
containing only:

- `pretrain/opt-with-synthetic-10b/v1` group `tokens`, including train and
  validation objects for the curriculum arm;
- `curriculum/opt-with-synthetic-10b/v1` group `mtld`.

Use `--release-set legacy` for historical RegMix-only modes, or omit the option only when one pod
must support both families. Avoiding RegMix on curriculum-only pods preserves checkpoint
headroom.

These are existing immutable releases; staging does not rebuild or publish a
dataset. The stager accepts only objects under `s3://edullm-data/`, writes
`/workspace/edullm-inputs/hpo-probe/ready.json` after all size checks pass, and
deletes `/workspace/aws-session.env` even on failure. Training refuses any
remaining AWS credential file or environment variable.

## 3. Install runtime secrets

Copy the same mode-0600 `wandb-session.env` file used by the curriculum RunPod
adapter to `/workspace/wandb-session.env`:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='...'
# Required for full_acronym_soup, no_proxy, and curriculum_quadratic_mtld:
export OPENAI_API_KEY='...'
# Optional override; defaults to https://gateway.truefoundry.ai/v1
# export OPENAI_BASE_URL='https://gateway.truefoundry.ai/v1'
```

Locally, the controller also reads `~/.wandb_api_key` and `~/.openai_api_key`
when the corresponding environment variables are unset (same pattern as
`.edullm/_tmp_clone_wandb_run.py`).

`WANDB_API_KEY` is required for every mode. `OPENAI_API_KEY` is not used by the
proxy cohort, `no_centaur`, or `curriculum_quadratic_mtld_no_centaur`.

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

First run the paired proxy cohort required by the two historical proxy arms:

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

The two curriculum extensions are independent of that cohort. Their coordinated
specs are `.edullm/hpo-curriculum-quadratic-mtld.json` (30% Centaur) and
`.edullm/hpo-curriculum-quadratic-mtld-no-centaur.json` (Centaur disabled).
Preflight the Centaur launch without creating a run identity or starting training:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main DRY_RUN=1 \
  bash .edullm/runpod/launch.sh
```

Then launch it:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main \
  bash .edullm/runpod/launch.sh
```

Launch the matched no-Centaur ablation in a separate mode and slot:

```bash
MODE=curriculum_quadratic_mtld_no_centaur RUN_SLOT=main \
  bash .edullm/runpod/launch.sh
```

For both curriculum modes, concurrent population trials use one GPU each. When only one resumed
winner remains, the launcher switches that continuation to an 8-rank job across GPUs 0–7 without
changing its global batch size.

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

For either curriculum arm, use its same mode and slot on resume:

```bash
MODE=curriculum_quadratic_mtld RUN_SLOT=main RECOVERY_MODE=resume \
  bash .edullm/runpod/launch.sh
```

The slot has separate controller state, checkpoints, logs, and `run.env`.
`run.env` preserves the original `EDULLM_RUN_ID` and `WANDB_RUN_ID`; do not copy
it to a different slot.

## Artifacts

- Trial and exact-retrain checkpoints remain on persistent `/workspace`.
- Controller state, segment specs, study result, proxy evidence, and scalar
  statistics are mirrored to W&B project `hpo-probe`.
- After a successful arm, the final selected checkpoint and local run log are
  uploaded as W&B artifacts.
- Curriculum runs use the same W&B project, `hpo-probe`, and publish the
  final-evaluation-selected checkpoint as an `hpo-winner-checkpoint` artifact.
  Its metadata plus `study-result.json` are the handoff to a separately reviewed
  long-run recipe.
- The shared proxy evidence lives at
  `/workspace/edullm-runs/hpo-probe/shared/proxy-evidence.json`.

No production job is launched by bootstrap or staging. Only `launch.sh` starts
GPU training.
