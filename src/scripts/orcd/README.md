# ORCD smoke jobs

## Verify checkpoint and resume

Run both submissions from the root of the same clean checkout. Set `EDULLM_REPO_ROOT`
to that checkout and set `EDULLM_SCRATCH` to an explicit ORCD Scratch path before
continuing. Keep `EDULLM_COMMIT_SHA`, `RUN_NAME`, `WANDB_RUN_ID`, and `SAVE_FOLDER`
unchanged between submissions.

```bash
: "${EDULLM_REPO_ROOT:?set EDULLM_REPO_ROOT to the clean checkout}"
: "${EDULLM_SCRATCH:?set EDULLM_SCRATCH to an ORCD Scratch path}"

# Initial run
export EDULLM_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"
export RUN_NAME=orcd-bootstrap
export WANDB_RUN_ID=orcd-bootstrap
export SAVE_FOLDER="$EDULLM_SCRATCH/runs/$RUN_NAME"
export HARD_STOP_STEPS=20
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",SAVE_FOLDER="$SAVE_FOLDER",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch

# Resume after the first job reaches a checkpoint
export HARD_STOP_STEPS=25
mkdir -p "$EDULLM_SCRATCH/logs"
sbatch --output="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --error="$EDULLM_SCRATCH/logs/%x-%j.log" \
  --export=EDULLM_REPO_ROOT="$EDULLM_REPO_ROOT",EDULLM_COMMIT_SHA="$EDULLM_COMMIT_SHA",EDULLM_SCRATCH="$EDULLM_SCRATCH",RUN_NAME="$RUN_NAME",WANDB_RUN_ID="$WANDB_RUN_ID",SAVE_FOLDER="$SAVE_FOLDER",HARD_STOP_STEPS="$HARD_STOP_STEPS" \
  src/scripts/orcd/generic_smoke.sbatch
```

The first job writes durable checkpoints at steps 10 and 20. The second job uses
the training script's automatic checkpoint loading and advances the same run to
step 25.
