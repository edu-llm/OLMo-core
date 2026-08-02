#!/bin/bash
# ============================================================================
# Per-CHECKPOINT eval over the Impl-3 sweep -> the data for Figure 3 of RL's Razor.
#
# WHY. Every eval so far scored only each run's FINAL checkpoint, giving one point per
# (variant, T). Figure 3 wants the mid-training checkpoints too, so that each configuration
# contributes a whole trajectory and can be given its own fit. The checkpoints already exist
# (12 per complete run, log-spaced: 1 2 3 4 8 16 32 64 128 256 512 923) — only the eval is
# missing. This job fills it in: ~120 checkpoints from the 10 runs that finished their epoch.
#
# WHY IT FITS IN ONE JOB. A naive loop over eval/generate_eval.py + run_kl_curve.py would take
# well over a day, because it reloads the base model, re-generates the base's answers, and
# re-generates the base's KL continuations for EVERY checkpoint — all three are identical at
# every point — and it generates one prompt at a time. sweep_ckpt_eval.py hoists all of that
# out of the loop and batches generation, which is the difference between ~8 min and ~2 min
# per checkpoint.
#
# PREEMPTION IS SAFE. Results are appended per checkpoint and finished ones are skipped on
# restart, so if this is preempted just resubmit the identical script and it continues.
#
#     sbatch clusters/orcd/ckpt_sweep_eval.sbatch      # from post-training/
#     squeue --me  ;  tail -f logs/p7-ckpteval_<jobid>.out
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -J p7-ckpteval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
# Named for the protocol, not a version number. "v2" collided with a killed job's partial output
# and the protocol guard (correctly) refused to append; a name that changes whenever the
# measurement changes makes that failure impossible instead of merely detected.
OUT="${OUT:-out/ckpt_sweep_bare_hint250.jsonl}"
POC_CKPT="${POC_CKPT:-}"
cd "$PROJECT"

die() { echo "FATAL: $*" >&2; echo "Fix: run 'bash clusters/orcd/setup_orcd_env.sh' on a login node once, then resubmit." >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
python -c "import torch, transformers, peft" 2>/dev/null || die "env '$ENV_NAME' missing torch/transformers/peft."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not visible to torch."
# These are the pieces added for this job; a stale checkout would fail deep into the run.
python -c "from common.kl import base_continuations, mean_kl_cached, pedagogy_contexts" 2>/dev/null \
    || die "common/kl.py lacks base_continuations -> stale code on ORCD. rsync the project up and resubmit."
python -c "import sys; sys.path.insert(0,'eval/math_eval'); import math_scoring" 2>/dev/null \
    || die "eval/math_eval/math_scoring.py missing -> stale code on ORCD. rsync and resubmit."
python -c "
import ast,sys
src=open('eval/sweep_ckpt_eval.py').read()
need=['math_stats','parse_extras']
missing=[n for n in need if ('def %s'%n) not in src]
sys.exit('missing '+','.join(missing) if missing else 0)" \
    || die "eval/sweep_ckpt_eval.py predates the bare+hint rewrite -> stale code on ORCD. rsync and resubmit."
NMATH=$(wc -l < eval/math_eval/math_logic_prompts.jsonl)
[ "$NMATH" -ge 250 ] || die "math set has $NMATH items, expected 250 -> stale prompts on ORCD. rsync and resubmit."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
echo "preflight OK: $(python -c 'import torch;print("torch",torch.__version__)') | HF_HOME=$HF_HOME"

[ -f data/socrateach_sft_val.jsonl ] || python snapshot_hf_dataset.py || die "no held-out data."
nvidia-smi || true

echo "== per-checkpoint eval =="
# --require_epoch 0.99 drops the two runs killed mid-epoch (a-T16, a-T32): a partial run's
# checkpoints would sit at artificially low KL and bend every fit they appear in.
#
# The prior-task probe is 250 GSM8K items in two conditions (bare / boxed-answer hint), and
# nothing else. IFEval is off: at 34 prompts it never separated any two configurations (pooled
# R^2 = 0.16 against KL). BBH and AIME are gone: base scores 6.7% and 0.0% on them, at or below
# chance, so they cannot show forgetting. All of that budget went into GSM8K, the one probe with
# headroom (base 60%), which takes the resolvable accuracy gap from ~22 points to ~12.
# The POC anchor is no longer scored by default. It was included once to settle whether the POC
# lineage and impl2-rerun are the same model: on 250 items they agree to within 0.004 accuracy and
# 0.028 KL, so it now only double-plots the vanilla endpoint. Set POC_CKPT=<dir> to re-enable.
EXTRA=()
[ -n "${POC_CKPT:-}" ] && [ -d "$POC_CKPT" ] && EXTRA=(--extra_ckpt "poc-c923=$POC_CKPT")
python eval/sweep_ckpt_eval.py \
    --base_model "$BASE_MODEL" \
    --runs "${RUNS:-out/*}" \
    --out "$OUT" \
    --batch 32 --gen_max 512 --n_kl 64 --n_nll 128 \
    --require_epoch 0.99 \
    ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?

N=$(wc -l < "$OUT" 2>/dev/null || echo 0)
echo "== $N checkpoint rows in $OUT (exit $STATUS) =="

if [ "$STATUS" -ne 0 ]; then
    echo "eval exited non-zero — resubmit this same script; finished checkpoints are skipped."
    exit "$STATUS"
fi

echo "== figures =="
python eval/plot_figure3.py --data "$OUT" --out_dir out/figures || echo "[warn] plotting failed"

echo "== DONE =="
echo "Pull down:  rsync -az orcd-login:~/post-training/$OUT orcd-login:~/post-training/out/figures ./post-training/out/"
