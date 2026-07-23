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

## Pilot a bounded S3 transfer

A live transfer is conditional on separate, explicit approval and does not block
Plan 1 exit. The sandbox bucket is pilot-only; measure AWS transfer charges before
scaling this workflow.

Use only a bounded token shard that is public or research-cleared and has a known
SHA-256 digest. Keep the short-lived presigned GET and PUT URLs in a temporary JSON
file outside the repository with mode `0600`. Never place either URL in Git, command
arguments, Slurm logs, W&B, reports, or other durable output. The JSON object has
`download_url` and `upload_url` keys, but documentation and committed fixtures must
not contain URL or signature values.

After approval, run the transfer tool from a clean checkout. Pass the private JSON
file with `--url-file`, the known shard digest with `--expected-sha256`, a strict
byte bound with `--max-download-bytes`, and an explicit Engaging Scratch directory
with `--work-dir`. Pass `step20/config.json` or a deliberately selected checkpoint
archive from the smoke run with `--result-file`; do not upload a synthetic digest
payload.

Copy the verified downloaded shard into the generic smoke data root, generate the
local validation shard there, and submit the staged-data training run with both
`EDULLM_SCRATCH` and `OLMO_DATA_ROOT` set to explicit Engaging Scratch paths. The
tool reports byte counts, SHA-256, elapsed time, and throughput without including
credentials or presigned URLs.
