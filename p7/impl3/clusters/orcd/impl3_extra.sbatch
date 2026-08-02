#!/bin/bash
# ============================================================================
# Four more Impl-3 runs to fill the gaps in the sweep — 1xH200, BATCH mode.
#
# WHAT AND WHY.
#   b-T0.5, b-T1   The cold end of variant b, never tried. b-T2 is currently the ONLY config
#                  with a Pareto win over vanilla SFT at matched pedagogy NLL (+13 pts hinted
#                  GSM8K), and it sits at the coldest T we have run. Either the advantage keeps
#                  growing as T drops or it turns over — both answers are worth one GPU-hour.
#                  Caveat worth stating: config.yaml deliberately excludes T<=1 as "unstable",
#                  and variant a's T=0.5/T=1 runs are the evidence (their pedagogy NLL ended
#                  WORSE than base: 2.74 and 2.14 vs 1.42). These two may fail the same way.
#   a-T16, a-T32   Reruns. Both died at step 256 (epoch 0.28) in the 2026-07-30 sweep and have
#                  been excluded from every figure since. --resume auto continues them from
#                  their last checkpoint, so only the remaining ~72% is paid for.
#
# ORDER. The two b runs go FIRST even though the a reruns are cheaper. On mit_preemptable this
# job can lose its GPU at any moment, so the untested-and-informative half runs before the
# fill-in-the-table half.
#
# EVERY RUN IS RESTART-SAFE: log-spaced checkpoints + --resume auto. If preempted, resubmit the
# identical script and each run continues from its last checkpoint.
#
# All four use the same config as the 12 runs already in out/ (1 epoch, LR 2e-4, LoRA r16,
# log-spaced checkpoints) and variant b uses the SAME forward-KL reference (checkpoint-923),
# which is what keeps the precompute cache valid and the numbers comparable.
#
#     sbatch clusters/orcd/impl3_extra.sbatch      # from post-training/
#     squeue --me  ;  tail -f logs/p7-impl3x_<jobid>.out
#
# To finish only part of the set, override RUNS and give the pedagogy batches their own directory
# so the earlier runs' unjudged generations survive; CHAIN_EVAL=0 when an eval job is already up,
# since two of them appending to one jsonl would interleave rows:
#
#     RUNS="a:16 a:32" PED_DIR=extra_a CHAIN_EVAL=0 sbatch clusters/orcd/impl3_extra.sbatch
#
# The chained per-checkpoint eval appends into the SAME file the current figures are built from
# (out/ckpt_sweep_bare_hint250.jsonl), so these runs drop straight onto the existing graph.
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -J p7-impl3x
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"   # forward-KL reference for variant b — MUST match
                                               # the existing b runs or the cache key changes
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
CFG="impl3_kl_reweighted_sft/config.yaml"
cd "$PROJECT"

# "variant:temperature", in execution order (see ORDER above).
RUNS="${RUNS:-b:0.5 b:1 a:16 a:32}"

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT"
echo "planned runs: $RUNS"

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
    || die "common/modeling.py has no load_tokenizer_for -> stale code on ORCD. rsync up and resubmit."
# Assert resume actually works, by replaying both things that broke it: the torch<2.6 version gate
# on optimizer.pt, and weights_only unpickling of rng_state.pth (numpy's MT19937 key). Two attempts
# died seconds after launch on these, one guard each, so exercise a REAL checkpoint file here —
# a preflight failure costs nothing next to a wasted allocation.
python - <<'PY' || die "resume is still blocked -> stale common/sft_train.py on ORCD. rsync up and resubmit."
import glob, sys, torch
import transformers.trainer as t
from common.sft_train import _allow_resume_from_our_own_checkpoints as unlock
unlock()
try:
    getattr(t, "check_torch_load_is_safe", lambda: None)()  # absent on versions without the gate
    rng = sorted(glob.glob("out/*/checkpoint-*/rng_state.pth"))
    if rng:
        torch.load(rng[-1], weights_only=True)
        print("resume preflight: rng_state loads ->", rng[-1])
    else:
        print("resume preflight: no existing rng_state.pth to test (fresh runs only)")
except Exception as e:
    print("resume would still fail:", type(e).__name__, e)
    sys.exit(1)
PY
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$HOME/.cache/matplotlib}"
echo "preflight OK: $(python -c 'import torch;print("torch",torch.__version__)') | HF_HOME=$HF_HOME"
[ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ] && echo "WARNING: no WANDB_API_KEY and no ~/.netrc -> W&B offline."

# Variant b measures its per-token signal against this adapter; without it the two b runs cannot
# start at all, and the cache key would silently change if a different one were substituted.
[ -e "$VANILLA_SFT/adapter_config.json" ] || [ -e "$VANILLA_SFT/adapter_model.safetensors" ] \
    || die "vanilla adapter '$VANILLA_SFT' not found — variant b needs it. rsync checkpoint-923 up first."

nvidia-smi || true

# ---- 0b. chain the per-checkpoint eval BEFORE training, so preemption cannot skip it --------
# 'afterany': a preempted training job still leaves completed runs behind, and sweep_ckpt_eval.py
# refuses to grade any run short of a full epoch, so partial work is filtered rather than trusted.
# No OUT override -> it appends to out/ckpt_sweep_bare_hint250.jsonl, whose protocol stamp
# (bare+hint@250 GSM8K + first-turn KL) matches, so the 145 existing rows are skipped and only the
# new checkpoints are scored. A protocol mismatch would abort instead of silently mixing probes.
if [ "${CHAIN_EVAL:-1}" = "1" ]; then
    sbatch --dependency=afterany:"$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
        clusters/orcd/ckpt_sweep_eval.sbatch \
        && echo "queued per-checkpoint eval after job $SLURM_JOB_ID" \
        || echo "[warn] could not queue the follow-up eval; submit ckpt_sweep_eval.sbatch by hand"
fi

[ -f data/socrateach_sft_val.jsonl ] || python snapshot_hf_dataset.py || die "no held-out data."

# Same H200 knobs as every other run in out/: effective batch 32, no grad checkpointing.
COMMON="--config $CFG --per_device_batch 32 --grad_accum 1 --no_grad_checkpointing --resume auto"

# ---- 1. precompute both signals (expected: two cache hits) ---------------------------------
# Cache key is (data, variant, base_model, sft_model) with NO temperature in it, so all the
# temperatures below share one precompute per variant. weights/signal_{a,b}_*.pt already exist
# from the earlier sweeps; this should cost seconds.
for v in a b; do
    echo "== precompute (variant $v) =="
    if [ "$v" = "b" ]; then
        python impl3_kl_reweighted_sft/precompute_weights.py \
            --variant b --sft_model_id "$VANILLA_SFT" --config "$CFG" || die "precompute b failed."
    else
        python impl3_kl_reweighted_sft/precompute_weights.py \
            --variant a --config "$CFG" || die "precompute a failed."
    fi
done

# ---- 2. train the four runs sequentially on the one GPU ------------------------------------
i=0
NRUNS=$(echo "$RUNS" | wc -w)
for spec in $RUNS; do
    v="${spec%%:*}"; T="${spec##*:}"
    i=$((i + 1))
    tag="impl3-${v}-T${T}"
    echo ""
    echo "== train $i/$NRUNS: $tag =="
    if [ -d "out/$tag" ]; then
        have=$(ls -d "out/$tag"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
        echo "   existing checkpoints up to step ${have:-none} -> --resume auto continues from there"
    fi
    if [ "$v" = "b" ]; then
        # shellcheck disable=SC2086
        python impl3_kl_reweighted_sft/train_kl_sft.py \
            --variant b --temperature "$T" --sft_model_id "$VANILLA_SFT" $COMMON \
            || echo "[warn] $tag failed"
    else
        # shellcheck disable=SC2086
        python impl3_kl_reweighted_sft/train_kl_sft.py \
            --variant a --temperature "$T" $COMMON \
            || echo "[warn] $tag failed"
    fi
done

# ---- 3. completeness check — never let a partial run reach a figure ------------------------
# The 2026-07-30 sweep graded two runs killed at epoch 0.28, which made them look impressively
# low-KL when they had merely stopped early. Anything short is reported loudly and excluded.
echo ""
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
for spec in $RUNS; do
    tag="impl3-${spec%%:*}-T${spec##*:}"
    if complete_epoch "out/$tag"; then EVAL_TAGS+=("$tag"); else SKIPPED="$SKIPPED $tag"; fi
done
[ -n "$SKIPPED" ] && echo "WARNING: incomplete, excluded from eval:$SKIPPED (resubmit — --resume auto continues them)"

# ---- 4. pedagogy generations for the judge -------------------------------------------------
# The deterministic axes (KL, bare+hint GSM8K, pedagogy NLL) are all produced by the chained
# per-checkpoint job, so the only thing left to generate here is the pedagogy text a judge needs.
# Same test file / --n_dialogues 40 / seed as every previous batch, so the judged contexts are the
# same 40 dialogues. base and impl2 ride along as in-batch anchors: the judge scores each batch
# independently, so without them there is nothing to calibrate the new runs against.
if [ ${#EVAL_TAGS[@]} -gt 0 ] && [ "${GEN_PEDAGOGY:-1}" = "1" ]; then
    echo ""
    echo "== pedagogy generation =="
    # PED_DIR must be unique per invocation: a partial resubmit (RUNS="a:16 a:32") writing into the
    # default 'extra' would clobber the b-T0.5/b-T1 generations a previous submit left there, and
    # those are judged by hand later, so overwriting them silently loses that work.
    PED_DIR="${PED_DIR:-extra}"
    mkdir -p "eval/llm_judge/$PED_DIR"
    PED_CANDS="base= impl2=$VANILLA_SFT"
    for tag in "${EVAL_TAGS[@]}"; do
        step=$(ls -d "out/$tag"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1)
        [ -n "$step" ] && PED_CANDS="$PED_CANDS $tag=out/$tag/checkpoint-$step"
    done
    echo "pedagogy candidates:$PED_CANDS"
    # shellcheck disable=SC2086
    python eval/gen_pedagogy.py --base_model "$BASE_MODEL" \
        --test_file data/socrateach_sft_val.jsonl --candidates $PED_CANDS \
        --n_dialogues 40 --out "eval/llm_judge/$PED_DIR/test_results.jsonl" \
        && ( cd eval/llm_judge && python build_batches.py "$PED_DIR/test_results.jsonl" "$PED_DIR" 2 ) \
        || echo "[warn] pedagogy generation / judge-batch build failed"
fi

echo ""
echo "== DONE =="
echo "completed runs: ${EVAL_TAGS[*]:-none}"
if [ -n "$SKIPPED" ]; then
    echo "INCOMPLETE:$SKIPPED — resubmit this script to finish them."
    exit 1
fi
echo "The chained eval job appends to out/ckpt_sweep_bare_hint250.jsonl; once it finishes:"
echo "  python eval/plot_figure3.py --data out/ckpt_sweep_bare_hint250.jsonl --out_dir out/figures"
echo "  python eval/plot_figure3.py --data out/ckpt_sweep_bare_hint250.jsonl --out_dir out/figures \\"
echo "         --prior_key math_bare --suffix _bare"
