#!/bin/bash
# ============================================================================
# T=451 limit check + in-harness vanilla SFT control — 2 runs, 1xH200, BATCH mode.
#
# WHY THIS JOB. The 2026-07-30 sweep found that no Impl-3 run gets near vanilla SFT on
# pedagogy (best 0.50 vs impl2's 0.65). But the theory says T -> inf makes every pedagogy
# multiplier -> 1, i.e. the objective becomes plain SFT, so *some* high T must reproduce
# impl2. The two runs that would have shown that (a-T16, a-T32) crashed mid-epoch, so the
# limit was never actually tested. This job tests it directly with two runs:
#
#   1. impl3-b-T451  — variant b at T=451. z-scores are O(1), so exp(-z/451) ~ 1 +- 0.2%:
#                      the weights are uniform to within a rounding error. This is the
#                      T -> inf limit reached through the REAL reweighting code path.
#   2. impl2-rerun   — vanilla SFT (no weighting at all) through the SAME config, so the
#                      only difference from run 1 is the weighting code itself.
#
# Read it as a 2x2 collapse: if the two land on the same pedagogy score, the harness is
# consistent and Impl-3's failure is a real property of the objective. If they differ, the
# weighting path has a bug and the whole sweep needs re-reading. Either answer is decisive.
#
# Both use the Impl-3 config (log-spaced checkpoints, 1 epoch, LR 2e-4, LoRA r16) so they
# are directly comparable to the 12 runs already in out/.
#
# Submit from the project root (post-training/):
#     sbatch clusters/orcd/t451_control.sbatch
#
# Monitor:  squeue --me    then    tail -f logs/p7-t451_<jobid>.out
#
# Nothing here overwrites the 2026-07-30 results: the KL file, the pedagogy generations and
# the judge batches all go to NEW paths (out/kl_t451.json, eval/llm_judge/t451/).
# ============================================================================
#SBATCH -p mit_preemptable       # Standing choice: preemptable gets a GPU far sooner, and every
                                 # run here uses --resume auto + log-spaced checkpoints, so being
                                 # preempted costs the steps since the last checkpoint, not the run.
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J p7-t451
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"   # forward-KL reference for variant b
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
CFG="impl3_kl_reweighted_sft/config.yaml"
TEMP="${TEMP:-451}"
cd "$PROJECT"

TAG_KL="impl3-b-T${TEMP}"
TAG_SFT="impl2-rerun"

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT | runs: $TAG_KL, $TAG_SFT"

# ---- 0. PREFLIGHT: fail loudly rather than reporting a hollow "Completed" ------------------
die() { echo "FATAL: $*" >&2; echo "Fix: run 'bash clusters/orcd/setup_orcd_env.sh' on a login node once, then resubmit." >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH (Miniforge not installed)."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
python -c "import torch, transformers, peft" 2>/dev/null || die "env '$ENV_NAME' missing torch/transformers/peft."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not visible to torch."
python -c "from common.modeling import load_tokenizer_for" 2>/dev/null \
    || die "common/modeling.py has no load_tokenizer_for -> stale code on ORCD. rsync the project up and resubmit."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "preflight OK: $(python -c 'import torch;print("torch",torch.__version__)') | HF_HOME=$HF_HOME"
[ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] && echo "WARNING: no WANDB_API_KEY and no ~/.netrc -> W&B offline."

# variant b's forward-KL signal is measured against the vanilla adapter — without it run 1 is
# impossible, and a T-limit check with only the control half is pointless. Stop instead.
[ -e "$VANILLA_SFT/adapter_config.json" ] || [ -e "$VANILLA_SFT/adapter_model.safetensors" ] \
    || die "vanilla adapter '$VANILLA_SFT' not found — variant b needs it. rsync checkpoint-923 up first."

nvidia-smi || true

# ---- 0b. chain the per-checkpoint eval so this run produces CURVES, not two points -----------
# Every eval axis is measured at every log-spaced checkpoint (that is what the Figure-3 style
# plots need), which is a separate GPU job because it takes hours on its own. Queue it now with
# a dependency instead of at the end, so it is submitted even if this job is preempted.
# 'afterany' not 'afterok': a preempted training job still leaves usable finished runs behind,
# and sweep_ckpt_eval.py refuses to grade any run that did not complete its epoch.
if [ "${CHAIN_EVAL:-1}" = "1" ]; then
    sbatch --dependency=afterany:"$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
        clusters/orcd/ckpt_sweep_eval.sbatch \
        && echo "queued per-checkpoint eval after job $SLURM_JOB_ID" \
        || echo "[warn] could not queue the follow-up eval; submit ckpt_sweep_eval.sbatch by hand"
fi

# Same H200 knobs as the sweep: effective batch 32, no grad checkpointing, resume-safe.
COMMON="--config $CFG --per_device_batch 32 --grad_accum 1 --no_grad_checkpointing --resume auto"

# ---- 1. precompute the forward-KL signal ---------------------------------------------------
# The cache key is (data, variant, base, sft_model) with NO temperature in it, so if the sweep's
# weights/ dir is still here this is a cache hit and costs seconds instead of ~40 min.
echo "== precompute (variant b) =="
python impl3_kl_reweighted_sft/precompute_weights.py \
    --variant b --sft_model_id "$VANILLA_SFT" --config "$CFG" || die "precompute failed."

# ---- 2. train both runs sequentially on the one GPU ----------------------------------------
echo "== train 1/2: $TAG_KL (variant b, T=$TEMP) =="
# shellcheck disable=SC2086
python impl3_kl_reweighted_sft/train_kl_sft.py \
    --variant b --temperature "$TEMP" --sft_model_id "$VANILLA_SFT" $COMMON \
    || echo "[warn] $TAG_KL failed"

echo "== train 2/2: $TAG_SFT (vanilla SFT, no weighting) =="
# Same config file as above => identical data, LR, LoRA, epoch count and log-spaced checkpoints.
# train_sft.py passes attach_weights=None, so the ONLY difference is the reweighting.
# shellcheck disable=SC2086
python impl1_2_prompting_sft/train_sft.py \
    --output_dir "out/$TAG_SFT" --run_name "$TAG_SFT" $COMMON \
    || echo "[warn] $TAG_SFT failed"

# ---- 2b. refuse to evaluate a run that did not finish its epoch -----------------------------
# The 2026-07-30 job silently graded two checkpoints from runs killed at epoch ~0.38, which made
# them look impressively low-KL when they had simply stopped early. Eval now only sees runs that
# actually completed, and anything short is reported loudly.
echo "== completeness check =="
complete_epoch() {
    python - "$1" <<'PY'
import glob, json, re, sys
d = sys.argv[1]
cks = glob.glob(f"{d}/checkpoint-*/trainer_state.json")
if not cks:
    print(f"  {d}: NO CHECKPOINTS"); sys.exit(1)
latest = max(cks, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
ep = json.load(open(latest)).get("epoch") or 0.0
print(f"  {d}: epoch={ep:.3f}")
sys.exit(0 if ep >= 0.99 else 1)
PY
}

EVAL_TAGS=()
SKIPPED=""
for tag in "$TAG_KL" "$TAG_SFT"; do
    if complete_epoch "out/$tag"; then EVAL_TAGS+=("$tag"); else SKIPPED="$SKIPPED $tag"; fi
done
[ -n "$SKIPPED" ] && echo "WARNING: incomplete, excluded from eval:$SKIPPED (resubmit — --resume auto will continue them)"
[ ${#EVAL_TAGS[@]} -eq 0 ] && die "neither run completed an epoch — nothing worth evaluating."
echo "evaluating: ${EVAL_TAGS[*]}"

# ---- 3. eval — same suite as the sweep, written to NEW paths --------------------------------
echo "== eval generation =="
if [ ! -f data/socrateach_sft_val.jsonl ]; then
  python snapshot_hf_dataset.py || die "no data/socrateach_sft_val.jsonl (prestage on the login node)."
fi

final_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

CKPTS=""
for tag in "${EVAL_TAGS[@]}"; do
    step=$(final_ckpt "out/$tag"); [ -n "$step" ] || continue
    CKPTS="$CKPTS $tag=out/$tag/checkpoint-$step"
done
echo "final checkpoints:$CKPTS"

# (a) KL axis. impl2 is included again so this file is self-contained and directly comparable
# to out/kl_by_checkpoint.json without having to merge anything.
# shellcheck disable=SC2086
python run_kl_curve.py --base_model "$BASE_MODEL" \
    --ckpt $CKPTS impl2="$VANILLA_SFT" \
    --pedagogy_file data/socrateach_sft_val.jsonl --n_prompts 64 \
    --out out/kl_t451.json || echo "[warn] KL curve failed"

# (b) math retention — deterministic, no judge. New tags, so no results are clobbered. IFEval ran
# here too and was dropped: 34 prompts never separated any two configurations.
FAILED=""
for tag in "${EVAL_TAGS[@]}"; do
    step=$(final_ckpt "out/$tag"); [ -n "$step" ] || continue
    python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl \
        --adapter "out/$tag/checkpoint-$step" --out "eval/math_eval/results_$tag.jsonl" \
        || { echo "[warn] math gen $tag failed"; FAILED="$FAILED math:$tag"; }
done

# (d) pedagogy. Same test file, same --n_dialogues 40 and same seed as the sweep, so the judged
# contexts are the SAME 40 dialogues and the scores are comparable to the 2026-07-30 table.
# base and impl2 are regenerated here as in-batch anchors: the judge scores each batch on its own,
# so without them there is nothing in this batch to calibrate the two new runs against.
mkdir -p eval/llm_judge/t451
PED_CANDS="base= impl2=$VANILLA_SFT"
for tag in "${EVAL_TAGS[@]}"; do
    step=$(final_ckpt "out/$tag"); [ -n "$step" ] || continue
    PED_CANDS="$PED_CANDS $tag=out/$tag/checkpoint-$step"
done
echo "pedagogy candidates:$PED_CANDS"
# shellcheck disable=SC2086
python eval/gen_pedagogy.py --base_model "$BASE_MODEL" \
    --test_file data/socrateach_sft_val.jsonl --candidates $PED_CANDS \
    --n_dialogues 40 --out eval/llm_judge/t451/test_results.jsonl \
    && ( cd eval/llm_judge && python build_batches.py t451/test_results.jsonl t451 2 ) \
    || { echo "[warn] pedagogy generation / judge-batch build failed"; FAILED="$FAILED pedagogy"; }

echo "== DONE =="
[ -n "$FAILED" ] && echo "FAILED points:$FAILED"
if [ -n "$SKIPPED" ]; then
    echo "INCOMPLETE RUNS:$SKIPPED — resubmit to finish them before trusting any of this."
    exit 1
fi
echo "Next (off-GPU): score the new tags with eval/math_eval/score_results.py;"
echo "                judge eval/llm_judge/t451/judge_batch_{0,1}.json into judge_out_{0,1}.json there,"
echo "                then: cd eval/llm_judge/t451 && python ../aggregate.py   (reads/writes in t451/)"
echo "Key question: does $TAG_KL land on the same pedagogy score as $TAG_SFT?"
