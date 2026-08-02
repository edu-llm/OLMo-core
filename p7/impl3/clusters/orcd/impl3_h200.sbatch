#!/bin/bash
# ============================================================================
# Impl-3 (KL-reweighted SFT) — full sweep on 2xH200, BATCH mode.
#
# Unlike `salloc` (interactive, holds your terminal), `sbatch` queues this job and
# runs it AUTOMATICALLY the moment the scheduler grants the GPUs — you can log out.
# It does the whole IMPL3_H200_RUNBOOK end to end: precompute -> train (variant a on
# GPU0, variant b on GPU1, in parallel) -> deterministic eval generation.
#
# Submit from the project root (post-training/) so relative paths resolve:
#     export WANDB_API_KEY=...            # W&B on by default
#     export HF_HOME=/orcd/pool/<you>/hf_cache
#     sbatch clusters/orcd/impl3_h200.sbatch
#
# Monitor:   squeue --me         then   tail -f logs/p7-impl3_<jobid>.out
# Skip variant b (no dep on the vanilla adapter):  RUN_VARIANT_B=0 sbatch ... (via --export)
# ============================================================================
#SBATCH -p mit_preemptable       # Standing choice: preemptable gets a GPU far sooner. --resume auto
                                 # plus log-spaced checkpoints means preemption costs the steps since
                                 # the last checkpoint, not the run.
#SBATCH -N 1                     # single node (no multi-node split)
#SBATCH --gpus-per-node=h200:1   # ONE H200 is plenty for a 1B LoRA (~4GB, <40% util). 1-GPU jobs
                                 # schedule fastest and avoid the multi-node / idle-GPU waste we hit
                                 # with 2 GPUs. Runs go sequentially; the whole sweep still fits 4h.
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J p7-impl3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"   # variant-b reference adapter (Impl-2)
RUN_VARIANT_A="${RUN_VARIANT_A:-1}"            # set 0 to run only variant b (a already trained -> don't retrain)
RUN_VARIANT_B="${RUN_VARIANT_B:-1}"            # set 0 to run only variant a (~half the work)
CFG="impl3_kl_reweighted_sft/config.yaml"
# T ladder shifted UP on 2026-07-29: low T over-concentrates the reweighting onto a few
# high-signal tokens; T=0.5 and T=1 were unstable / behaved badly in the variant-a runs, so the
# ladder is now 2..32 (higher T = gentler, closer to vanilla). Keep this in sync with submit_sweep.sh.
TEMPS=(2 4 8 16 32)

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT | variant_a=$RUN_VARIANT_A variant_b=$RUN_VARIANT_B"
cd "$PROJECT"

# ---- 0. PREFLIGHT: fail loudly if the env isn't set up (don't fake a "Completed" job) ------
die() { echo "FATAL: $*" >&2; echo "Fix: run 'bash clusters/orcd/setup_orcd_env.sh' on a login node once, then resubmit." >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH (Miniforge not installed)."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
command -v python >/dev/null 2>&1 || die "python not on PATH after activating '$ENV_NAME'."
python -c "import torch, transformers, peft" 2>/dev/null \
    || die "env '$ENV_NAME' is missing torch/transformers/peft (setup didn't finish)."
# Advisory only. Eval used to die on "... sentencepiece or tiktoken" — that was really
# common.modeling loading the tokenizer from a LoRA-only checkpoint dir (fixed: it now falls back to
# the base model's tokenizer). These stay useful for odd tokenizers, so warn but never block.
python -c "import sentencepiece, tiktoken" 2>/dev/null \
    || echo "NOTE: sentencepiece/tiktoken not installed (fine for OLMo-2; 'pip install sentencepiece tiktoken' if a custom tokenizer needs conversion)."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || die "CUDA not visible to torch on this node (no GPU / wrong node)."
echo "preflight OK: $(python -V 2>&1) | $(python -c 'import torch;print("torch",torch.__version__,"cuda",torch.cuda.is_available())')"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "HF_HOME=$HF_HOME"
[ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] && echo "WARNING: no WANDB_API_KEY and no ~/.netrc -> W&B offline. Run 'wandb login' or add --no_wandb."

# variant b needs the vanilla Impl-2 adapter; if it isn't here, skip b instead of failing 5 runs.
if [ "$RUN_VARIANT_B" = "1" ] && [ ! -e "$VANILLA_SFT/adapter_config.json" ] && [ ! -e "$VANILLA_SFT/adapter_model.safetensors" ]; then
    echo "WARNING: vanilla adapter '$VANILLA_SFT' not found -> running variant a only (set VANILLA_SFT or rsync checkpoint-923)."
    RUN_VARIANT_B=0
fi
echo "effective variant_a=$RUN_VARIANT_A variant_b=$RUN_VARIANT_B"
# b-only reuses variant a's already-trained checkpoints (out/impl3-a-*/) for the eval stage; a is not retrained.
[ "$RUN_VARIANT_A" = "0" ] && [ "$RUN_VARIANT_B" = "0" ] && die "both variants disabled (RUN_VARIANT_A=0 and RUN_VARIANT_B=0) — nothing to train."
# How many GPUs did we actually get? We now request ONE, but the script still adapts to more.
NGPU=$(python -c "import torch;print(max(1,torch.cuda.device_count()))")
echo "visible GPUs on $(hostname): $NGPU"
# Put variant b's precompute on a 2nd GPU only if we truly have one AND a also runs; else GPU0
# (CUDA index 1 is an invalid ordinal on a 1-GPU node).
B_GPU=0; [ "$NGPU" -gt 1 ] && [ "$RUN_VARIANT_A" = "1" ] && B_GPU=1
nvidia-smi || true

# ---- chain the per-checkpoint eval so the sweep produces CURVES, not one point per run --------
# All eval axes are measured at every log-spaced checkpoint. That cannot run inline here — a
# 10-run sweep is already over this job's time limit before any eval — so it goes out as a
# dependent job. 'afterany' because a preempted sweep still leaves finished runs worth grading,
# and sweep_ckpt_eval.py skips any run that did not complete its epoch.
if [ "${CHAIN_EVAL:-1}" = "1" ]; then
    sbatch --dependency=afterany:"$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
        clusters/orcd/ckpt_sweep_eval.sbatch \
        && echo "queued per-checkpoint eval after job $SLURM_JOB_ID" \
        || echo "[warn] could not queue the follow-up eval; submit ckpt_sweep_eval.sbatch by hand"
fi

# H200 speed knobs: EFFECTIVE batch = 32 (per_device 32 x accum 1 x 1 GPU) so #steps and the
# recipe match the POC; drop grad checkpointing (141GB fits). --resume auto survives preemption.
COMMON="--config $CFG --per_device_batch 32 --grad_accum 1 --no_grad_checkpointing --resume auto"

# ---- 1. precompute weight signals (variant a on GPU0, variant b on GPU1) ------------------
echo "== precompute =="
if [ "$RUN_VARIANT_A" = "1" ]; then
  CUDA_VISIBLE_DEVICES=0 python impl3_kl_reweighted_sft/precompute_weights.py \
      --variant a --config "$CFG" &
fi
if [ "$RUN_VARIANT_B" = "1" ]; then
  CUDA_VISIBLE_DEVICES=$B_GPU python impl3_kl_reweighted_sft/precompute_weights.py \
      --variant b --sft_model_id "$VANILLA_SFT" --config "$CFG" &
fi
wait
echo "== precompute done =="

# ---- 2. train the sweep — GPU pool: spread ALL runs across every visible GPU -------------
# Build one flat work list of every (variant,T) run and hand them out round-robin to each visible
# GPU; each GPU drains its slice sequentially. With the default 1-GPU request all runs go to GPU0
# in sequence; if a job ever gets >1 GPU (on one node) they run in parallel automatically.
echo "== train =="
JOBS=()
[ "$RUN_VARIANT_A" = "1" ] && for T in "${TEMPS[@]}"; do JOBS+=("a $T"); done
[ "$RUN_VARIANT_B" = "1" ] && for T in "${TEMPS[@]}"; do JOBS+=("b $T"); done
echo "scheduling ${#JOBS[@]} training runs across $NGPU GPU(s)"

for g in $(seq 0 $((NGPU-1))); do
  (
    idx=$g
    while [ "$idx" -lt "${#JOBS[@]}" ]; do
      # shellcheck disable=SC2086
      set -- ${JOBS[$idx]}; v=$1; T=$2
      extra=""; [ "$v" = "b" ] && extra="--sft_model_id $VANILLA_SFT"
      echo "[gpu$g] start variant=$v T=$T"
      # shellcheck disable=SC2086
      CUDA_VISIBLE_DEVICES=$g python impl3_kl_reweighted_sft/train_kl_sft.py \
          --variant "$v" --temperature "$T" $extra $COMMON \
          || echo "[warn] variant $v T=$T failed"
      idx=$((idx + NGPU))
    done
  ) > "logs/train_impl3_gpu${g}_$SLURM_JOB_ID.txt" 2>&1 &
done
wait
echo "== train done; checkpoints under out/impl3-*/ =="

# ---- 3. deterministic eval generation (final checkpoint of each run) ----------------------
echo "== eval generation =="
# Held-out pedagogy split for the KL axis. Prefer a prestaged local snapshot (data/); only try
# the Hub if it's absent (compute nodes may be offline — prestage on the login node to be safe).
if [ ! -f data/socrateach_sft_val.jsonl ]; then
  python snapshot_hf_dataset.py || echo "[warn] could not fetch data/ (prestage it on the login node: python snapshot_hf_dataset.py)"
fi

# pick the highest-numbered checkpoint in each run dir (robust to exact step count)
final_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

CKPTS=""
for d in out/impl3-*/; do
  [ -d "$d" ] || continue
  step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  CKPTS="$CKPTS $(basename "$d")=$d/checkpoint-$step"
done
echo "final checkpoints:$CKPTS"
[ -z "$CKPTS" ] && echo "[warn] no checkpoints found under out/impl3-*/ — did training run? check logs/train_impl3_*.txt"

# (a) KL axis vs base (one shared base load); include the vanilla Impl-2 point
python run_kl_curve.py --base_model allenai/OLMo-2-0425-1B-Instruct \
    --ckpt $CKPTS impl2="$VANILLA_SFT" \
    --pedagogy_file data/socrateach_sft_val.jsonl --n_prompts 64 \
    --out out/kl_by_checkpoint.json || echo "[warn] KL curve failed"

# (b) math retention + (c) IFEval — both deterministic (no subagents)
for d in out/impl3-*/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d"); step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  python eval/generate_eval.py --prompts eval/math_eval/math_logic_prompts.jsonl \
      --adapter "$d/checkpoint-$step" --out eval/math_eval/results_$tag.jsonl \
      || echo "[warn] math gen $tag failed"
  python eval/generate_eval.py --prompts eval/general_eval/ifeval_prompts.jsonl \
      --adapter "$d/checkpoint-$step" --out eval/general_eval/results_$tag.jsonl \
      || echo "[warn] ifeval gen $tag failed"
done

# (d) pedagogy generation (NEW-task quality — the judge-based y-axis). Generate one tutor turn per
# held-out context for base + vanilla Impl-2 + each final checkpoint, then blind them into judge
# batches. This is the ONLY judge-based (subagent) axis; scoring happens off-GPU afterward.
PED_CANDS="base= impl2=$VANILLA_SFT"
for d in out/impl3-*/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d"); step=$(final_ckpt "$d"); [ -n "$step" ] || continue
  PED_CANDS="$PED_CANDS $tag=$d/checkpoint-$step"
done
echo "pedagogy candidates:$PED_CANDS"
python eval/gen_pedagogy.py --base_model allenai/OLMo-2-0425-1B-Instruct \
    --test_file data/socrateach_sft_val.jsonl --candidates $PED_CANDS \
    --n_dialogues 40 --out eval/llm_judge/test_results_instruct.jsonl \
    && ( cd eval/llm_judge && python build_batches.py test_results_instruct.jsonl ) \
    || echo "[warn] pedagogy generation / judge-batch build failed"

echo "== ALL DONE. Deterministic: grade_math_logic.py / grade_ifeval.py, then plot + wandb_eval.py."
echo "   Pedagogy: judge eval/llm_judge/judge_batch_*.json (frontier judge) -> judge_out_*.json -> aggregate.py =="
