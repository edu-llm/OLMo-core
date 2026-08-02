#!/bin/bash
# ============================================================================
# Generate tutor responses for the whole P7 fleet with the eval team's tutor_cat harness.
#
# 18 models x 1304 scenarios (TutorBench 662 + Bridge 642) = ~23.5k generations. Each model is
# loaded ONCE and answers both benchmarks, which is the whole point of the --benchmarks registry:
# a 1B model generates far faster than vLLM loads it, so per-benchmark runs would pay the load
# cost twice for nothing.
#
# GENERATION ONLY. No judge and no ability estimate come out of this; it writes response shards
# that `tutor-cat run` later grades. The judge server is a separate job and is NOT required here,
# which is why this can start before Prometheus is up.
#
# Resume is the harness default (--no-resume disables it): finished (benchmark, model) shards are
# skipped on restart. That is what makes mit_preemptable safe here — a preemption costs at most
# the model in flight, so just resubmit.
#
#     sbatch clusters/orcd/tutorcat_generate.sbatch      # from post-training/
#
# Useful overrides:
#     ONLY=TutorBench   sbatch ...   # one benchmark
#     LIMIT=20          sbatch ...   # cap scenarios/model, smoke only
#     MODEL=<id>        sbatch ...   # single model from the manifest
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00
#SBATCH -J p7-tcgen
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-tutorcat}"
EVAL_REPO="${EVAL_REPO:-$HOME/olmo-eval-full/eduLLM-Evals}"
MANIFEST="${MANIFEST:-$HOME/tutorcat_models.yaml}"
OUT_DIR="${OUT_DIR:-$HOME/tutorcat_runs/responses}"
ONLY="${ONLY:-TutorBench,Bridge}"
cd "$PROJECT" || exit 1

die() { echo "FATAL: $*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
# Every check below has already cost us a wasted queue slot at least once.
CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' missing — run clusters/orcd/setup_tutorcat.sh"

# absl-py is an undeclared dep of their ifeval verifier; without it EVERY subcommand dies on import.
python -c "import torch, vllm, absl, tutor_cat; from tutor_cat.cli import main" 2>/dev/null \
    || die "$ENV_NAME incomplete (need torch+vllm+absl+tutor_cat). See TUTOR_CAT_EVAL.md section 8."

[ -f "$MANIFEST" ]   || die "manifest $MANIFEST missing — run clusters/orcd/merge_tutorcat.sbatch"
[ -d "$EVAL_REPO" ]  || die "eval repo $EVAL_REPO missing"
cd "$EVAL_REPO"      || die "cannot cd $EVAL_REPO"
[ -f benchmarks.yaml ] || die "benchmarks.yaml missing in $EVAL_REPO"

nvidia-smi -L || die "no GPU visible"

# Guard the config schema. If a merge ran under transformers 5.x, config.json carries `dtype` and a
# nested `rope_parameters` block instead of `torch_dtype` and a top-level `rope_theta`. vLLM 0.10
# reads the top-level key, and on a miss silently substitutes rope_theta=10000 for this model's
# 500000 — wrong positional encoding at every position, fluent word salad, exit code 0. It cost us
# a full submission once; it is a two-second check.
python - "${MODEL_DIR:-$HOME/tutorcat_models}" <<'PY' || exit 1
import json, glob, sys
paths = sorted(glob.glob(f"{sys.argv[1]}/*/config.json"))
if not paths:
    sys.exit(f"FATAL: no merged models under {sys.argv[1]}")
bad = [p.split("/")[-2] for p in paths
       if (lambda c: c.get("rope_theta") != 500000 or "rope_parameters" in c or "torch_dtype" not in c)(json.load(open(p)))]
if bad:
    sys.exit("FATAL: %d/%d merged model(s) have transformers-5.x config schema: %s\n"
             "Restore the base config.json (see eval/merge_for_tutorcat.py: copy_base_files)."
             % (len(bad), len(paths), ", ".join(bad)))
print(f"config schema OK ({len(paths)} models, top-level rope_theta=500000)")
PY

# Compute nodes have no outbound network. Everything must already be on disk: merged models are
# local paths, and the base model has to be in the HF cache or vLLM hangs retrying the hub.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
[ -d "$HF_HOME/hub/models--allenai--OLMo-2-0425-1B-Instruct" ] \
    || die "base model not in HF cache and node is offline — pre-fetch it on the login node."

# --- smoke test ------------------------------------------------------------
# 8 scenarios on one model. Catches a bad manifest, a missing chat template or an OOM in ~2 min
# instead of after an hour of queue plus an hour of generation.
echo "== smoke test =="
SMOKE_DIR="$HOME/tutorcat_runs/_smoke_${SLURM_JOB_ID:-manual}"
# Pin to ONE model: --limit caps scenarios per model but still walks all 18, and 18 model loads is
# not a smoke test. It must be a MERGED model, not the base — the base id contains "Instruct", so
# it passes the chat-template heuristic on its own and would prove nothing about the pinning fix.
SMOKE_MODEL="${SMOKE_MODEL:-$HOME/tutorcat_models/impl2-rerun}"
[ -d "$SMOKE_MODEL" ] || die "smoke model $SMOKE_MODEL missing"
timeout 1800 tutor-cat generate \
    --models "$MANIFEST" --benchmarks benchmarks.yaml --only TutorBench \
    --out-dir "$SMOKE_DIR" --gpus 1 --limit 8 --no-resume --model "$SMOKE_MODEL"
SMOKE_STATUS=$?
SMOKE_ROWS=$(cat "$SMOKE_DIR"/TutorBench/*.jsonl 2>/dev/null | wc -l)
echo "smoke: exit=$SMOKE_STATUS rows=$SMOKE_ROWS"
[ "$SMOKE_STATUS" -eq 0 ] && [ "$SMOKE_ROWS" -gt 0 ] || die "smoke test produced nothing — not burning the allocation on the full fleet."

# Inspect the rows, not just the exit code. Keys are the PRD's Title-Case contract
# (tutor_cat/respgen/records.py) — "Output", not "response".
#
# "Chat Template Applied" is the one that matters: their registry infers it from substrings of the
# model id, our ids are local paths with no 'instruct' marker, and a False here means every
# chat-tuned model was prompted as a raw base LM. That produces fluent, plausible, meaningless
# text and raises nothing. The manifest pins it true; this confirms it survived to runtime.
python - "$SMOKE_DIR" <<'PY'
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/**/*.jsonl", recursive=True) for l in open(f)]
if not rows:
    sys.exit("FATAL: smoke wrote no rows.")
empty  = sum(1 for r in rows if not (r.get("Output") or "").strip())
notmpl = sum(1 for r in rows if not int(r.get("Chat Template Applied", 0)))
issues = sum(1 for r in rows if int(r.get("Issue", 0)))
trunc  = sum(1 for r in rows if int(r.get("Truncated", 0)))
print(f"smoke rows={len(rows)} empty={empty} no_chat_template={notmpl} issues={issues} truncated={trunc}")
if notmpl:
    sys.exit(f"FATAL: {notmpl}/{len(rows)} rows generated WITHOUT a chat template.")
if empty == len(rows):
    sys.exit("FATAL: every smoke output is empty — check max_model_len / generation params.")
sample = next(r["Output"] for r in rows if (r.get("Output") or "").strip())
print("sample output:", sample[:300].replace("\n", " "))
PY
[ $? -eq 0 ] || exit 1
rm -rf "$SMOKE_DIR"

# --- full fleet ------------------------------------------------------------
echo
echo "== full run: benchmarks=$ONLY =="
mkdir -p "$OUT_DIR"
EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${MODEL:-}" ] && EXTRA+=(--model "$MODEL")

date +"start %F %T"
tutor-cat generate \
    --models "$MANIFEST" --benchmarks benchmarks.yaml --only "$ONLY" \
    --out-dir "$OUT_DIR" --gpus 1 \
    ${EXTRA[@]+"${EXTRA[@]}"}
STATUS=$?
date +"end   %F %T"

echo
echo "== shards written =="
for d in "$OUT_DIR"/*/; do
    [ -d "$d" ] || continue
    echo "$(basename "$d"): $(ls "$d" | wc -l)/18 models, $(cat "$d"/*.jsonl 2>/dev/null | wc -l) rows"
done
echo "exit=$STATUS  (non-zero or short counts: just resubmit, finished shards are skipped)"
exit "$STATUS"
