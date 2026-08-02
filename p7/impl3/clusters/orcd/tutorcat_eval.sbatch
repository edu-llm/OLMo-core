#!/bin/bash
# ============================================================================
# The actual evaluation: judge our 18 models' TutorBench responses and estimate ability.
#
# Two things run here, on one node:
#   1. Prometheus 2 7B served by vLLM on localhost:8000 (the judge).
#   2. 18 concurrent `tutor-cat run` processes, one per model, driving the CAT engine.
#
# Their scripts assume the judge runs on a GPU node and the pipeline runs on your laptop over an
# SSH tunnel. Co-locating both removes the tunnel entirely — nothing to keep alive, nothing to
# break mid-run.
#
# WHY CONCURRENT: the engine judges criteria sequentially (`verdict = judge.evaluate(...)` per
# criterion), and TutorBench averages 9.8 criteria per scenario at up to 50 scenarios per model —
# ~8.8k judge calls. Serially that is a >10 h job. One vLLM server with 18 clients batches them,
# which is what the GPU is good at. run_id embeds the tutor name, so the runs cannot collide.
#
# The models themselves never run: every response is a cache hit pre-populated by
# eval/tutorcat_build_cache.py, so the tutor API clients are constructed but never called.
#
# Resume: a model whose run dir already holds final_result.json is skipped, so a preemption only
# costs the models still in flight.
#
#     sbatch clusters/orcd/tutorcat_eval.sbatch      # from post-training/
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -J p7-tceval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

ENV_NAME="${ENV_NAME:-tutorcat}"
EVAL_REPO="${EVAL_REPO:-$HOME/olmo-eval-full/eduLLM-Evals}"
CONFIG="${CONFIG:-$HOME/tutorcat_p7_config.yaml}"
JUDGE_MODEL="${JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PORT="${PORT:-8000}"
MODE="${MODE:-cat}"

die() { echo "FATAL: $*" >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' missing"
cd "$EVAL_REPO" || die "cannot cd $EVAL_REPO"

[ -f "$CONFIG" ] || die "config $CONFIG missing — run eval/tutorcat_build_cache.py first"
nvidia-smi -L || die "no GPU visible"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1          # compute nodes are offline; weights are pre-fetched
export TOKENIZERS_PARALLELISM=false
export JUDGE_API_KEY="not-used"  # OpenAI() refuses to construct without a key; never sent
# Compute nodes have the driver but no CUDA toolkit, and FlashInfer's sampler JIT-compiles at
# startup -> "Could not find nvcc". Their serve script hits this too.
export VLLM_USE_FLASHINFER_SAMPLER=0

[ -d "$HF_HOME/hub/models--prometheus-eval--prometheus-7b-v2.0" ] \
    || die "Prometheus weights not in HF cache and the node is offline — pre-fetch on the login node."

# Every scenario must be a cache hit. If the cache were missing, the tutors would silently try to
# reach an OpenAI endpoint with local filesystem paths as model names.
CACHE_DIRS=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['cache_dir'])" "$CONFIG")
[ -d "$CACHE_DIRS" ] || die "response cache $CACHE_DIRS missing"
echo "cache: $(ls "$CACHE_DIRS" | wc -l) model dirs, $(find "$CACHE_DIRS" -name '*.json' | wc -l) responses"

# --- judge server ----------------------------------------------------------
echo "== starting Prometheus judge on :$PORT =="
vllm serve "$JUDGE_MODEL" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 16384 --gpu-memory-utilization 0.90 \
    > "$SLURM_SUBMIT_DIR/logs/judge_${SLURM_JOB_ID}.log" 2>&1 &
JUDGE_PID=$!
trap 'kill $JUDGE_PID 2>/dev/null' EXIT

for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { echo "judge ready after ${i}0s"; break; }
    kill -0 $JUDGE_PID 2>/dev/null || die "judge died on startup — see logs/judge_${SLURM_JOB_ID}.log"
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 \
    || die "judge did not become ready within 20 min"

# --- one CAT run per model, in parallel ------------------------------------
mapfile -t TUTORS < <(python -c "
import yaml, sys
print('\n'.join(t['name'] for t in yaml.safe_load(open(sys.argv[1]))['tutors']))" "$CONFIG")
echo "== ${#TUTORS[@]} tutors, mode=$MODE =="

mkdir -p runs
CAT_PIDS=()
for t in "${TUTORS[@]}"; do
    if compgen -G "runs/*_${t}_${MODE}/final_result.json" > /dev/null; then
        echo "skip $t (already complete)"
        continue
    fi
    ( tutor-cat run --config "$CONFIG" --tutor "$t" --mode "$MODE" \
        > "$SLURM_SUBMIT_DIR/logs/cat_${SLURM_JOB_ID}_${t}.log" 2>&1
      echo "done $t (exit $?)" ) &
    CAT_PIDS+=($!)
done
# Wait on the CAT runs SPECIFICALLY. A bare `wait` also waits on the judge server, which serves
# until killed — the job then hangs holding a GPU until the time limit even though every result is
# already written. Cost us an idle GPU once.
[ ${#CAT_PIDS[@]} -gt 0 ] && wait "${CAT_PIDS[@]}"
echo "== all CAT runs returned =="
kill $JUDGE_PID 2>/dev/null && echo "judge stopped"

# --- summary ---------------------------------------------------------------
python - <<'PY'
import glob, json, os

SKILLS = ("content", "diagnosis", "scaffolding")
fmt = lambda d: " ".join("%+.2f" % d.get(k, float("nan")) for k in SKILLS)

rows = []
for f in sorted(glob.glob("runs/*/final_result.json")):
    d = json.load(open(f))
    mpath = os.path.join(os.path.dirname(f), "manifest.json")
    name = json.load(open(mpath))["tutor_name"] if os.path.exists(mpath) else d["run_id"]
    rows.append((name, d))

if not rows:
    print("NO RESULTS — check logs/cat_*.log")
else:
    hdr = "%-16s %20s %20s %4s %5s %4s  %s" % (
        "tutor", "theta c/d/s", "SE c/d/s", "n", "prec", "crit", "stop")
    print(hdr); print("-" * len(hdr))
    for name, d in sorted(rows):
        print("%-16s %20s %20s %4d %5s %4d  %s" % (
            name, fmt(d["theta"]), fmt(d["se"]), d["scenarios_administered"],
            "yes" if d["precision_reached"] else "NO",
            d["critical_failure_count"], d["stop_reason"]))
    n_prec = sum(1 for _, d in rows if d["precision_reached"])
    print(f"\n{len(rows)}/18 complete, {n_prec} reached target precision")
PY
