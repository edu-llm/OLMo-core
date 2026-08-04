#!/bin/bash
# ============================================================================
# Pedagogy generation in the +SI condition — the 2x2's cells B and D.
#
# WHY THIS JOB. Every Impl-3 pedagogy number we have was generated with NO system
# instruction: eval/gen_pedagogy.py strips the row's system message and, until now, had
# no way to put one back. So the judged scores (base 0.386, vanilla SFT 0.653, best
# Impl-3 0.500) are all cell C of the POC's 2x2, not the headline cell D.
#
# That matters twice over:
#   1. The POC's reported SFT number, 0.84, IS cell D. Comparing an Impl-3 no-SI score
#      against it is comparing different conditions.
#   2. Figure 3's y-axis, ped_nll, is computed WITH the per-dialogue SI in context
#      (sweep_ckpt_eval.py tokenizes the raw val rows, all 1,724 of which start with a
#      system message). So NLL and the judge disagree partly because one condition has
#      the SI and the other does not — a-T8 reads as 81% of vanilla's learning by NLL
#      and 9% by the judge.
#
# This job regenerates held-out pedagogy turns with the canonical eval SI in context, for a
# SHORT candidate list. base and the vanilla SFT are always included as in-batch anchors: the
# judge scores each batch independently, and their SI-vs-noSI deltas are what tell us whether
# we reproduced the POC's A->B (0.38->0.71) and C->D (0.52->0.84) effects.
#
# It also fixes an n-inflation bug. The held-out file stores several dialogues per problem, so
# the first 40 rows are only 12 DISTINCT problems; with greedy decoding the copies produce
# byte-identical responses, so the old runs weighted 9 problems 4-5x and reported n=40 for a
# sample of 12. gen_pedagogy.py now dedupes on (problem, context) and --n_dialogues counts
# distinct problems. Deduping the existing no-SI labels moves every setup by <=0.021 and
# changes no ranking, so the conclusions stand — but the n does not.
#
# Submit from the project root (post-training/):
#     sbatch clusters/orcd/ped_si_regen.sbatch
#     TAGS="impl3-a-T8 impl3-b-T1" N_DIALOGUES=60 sbatch clusters/orcd/ped_si_regen.sbatch
#
# Nothing existing is overwritten: output goes to eval/llm_judge/si/.
# ============================================================================
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH --gpus-per-node=h200:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J p7-ped-si
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -uo pipefail
mkdir -p logs

PROJECT="${PROJECT:-$SLURM_SUBMIT_DIR}"
ENV_NAME="${ENV_NAME:-p7post}"
VANILLA_SFT="${VANILLA_SFT:-checkpoint-923}"
BASE_MODEL="${BASE_MODEL:-allenai/OLMo-2-0425-1B-Instruct}"
OUT_DIR="${OUT_DIR:-eval/llm_judge/si}"
N_BATCHES="${N_BATCHES:-4}"
N_DIALOGUES="${N_DIALOGUES:-100}"   # DISTINCT problems now that generation dedupes
# The runs worth judging in the +SI condition, not the whole sweep. Fewer candidates buys more
# problems at the same GPU cost, and n is what the old batch was short of.
#   a-T8   best math retention of any run, but 0/40 step-level guidance without an SI
#   b-T1   never judged at all (it finished after the sweep's judging batch)
#   b-T2   good pedagogy NLL, judged BELOW base without an SI
#   b-T32  the best no-SI Impl-3 score (0.500)
TAGS="${TAGS:-impl3-a-T8 impl3-b-T1 impl3-b-T2 impl3-b-T32 impl2-rerun}"
cd "$PROJECT"

echo "Job $SLURM_JOB_ID on $(hostname) | PROJECT=$PROJECT"

# ---- 0. PREFLIGHT ---------------------------------------------------------------------------
die() { echo "FATAL: $*" >&2; exit 1; }

CONDA_SH="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
[ -f "$CONDA_SH" ] || die "conda not found at $CONDA_SH. Run clusters/orcd/setup_orcd_env.sh once."
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME" || die "conda env '$ENV_NAME' does not exist."
python -c "import torch, transformers, peft" 2>/dev/null || die "env '$ENV_NAME' missing torch/transformers/peft."
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || die "CUDA not visible to torch."

# The whole point of this job is the --si flag. An older copy of the script on ORCD would
# silently regenerate the no-SI condition again and look like it succeeded.
python - <<'PY' || die "eval/gen_pedagogy.py is stale on ORCD (needs --si and dedupe). rsync the project up and resubmit."
import sys
src = open("eval/gen_pedagogy.py", encoding="utf-8").read()
sys.exit(0 if all(s in src for s in ('"--si"', "CANONICAL_SI", "--no_dedupe")) else 1)
PY
python -c "from common.system_instructions import CANONICAL_SI; assert len(CANONICAL_SI) > 200" \
    || die "CANONICAL_SI missing or truncated."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
echo "preflight OK | HF_HOME=$HF_HOME"
nvidia-smi || true

[ -f data/socrateach_sft_val.jsonl ] || python snapshot_hf_dataset.py \
    || die "no data/socrateach_sft_val.jsonl (prestage on the login node)."
[ -e "$VANILLA_SFT/adapter_config.json" ] || [ -e "$VANILLA_SFT/adapter_model.safetensors" ] \
    || die "vanilla adapter '$VANILLA_SFT' not found — it is the in-batch anchor. rsync it up first."

# ---- 1. candidate list: base + vanilla + every run that finished its epoch --------------------
final_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sed 's#.*checkpoint-##' | sort -n | tail -1; }

complete_epoch() {
python - "$1" <<'PY'
import glob, json, re, sys
d = sys.argv[1]
cks = glob.glob(f"{d}/checkpoint-*/trainer_state.json")
if not cks:
    sys.exit(1)
latest = max(cks, key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)))
sys.exit(0 if (json.load(open(latest)).get("epoch") or 0.0) >= 0.99 else 1)
PY
}

PED_CANDS="base= impl2=$VANILLA_SFT"
SKIPPED=""
for tag in $TAGS; do
    d="out/$tag"
    [ -d "$d" ] || { echo "[warn] no such run: $d"; SKIPPED="$SKIPPED $tag"; continue; }
    if ! complete_epoch "$d"; then SKIPPED="$SKIPPED $tag"; continue; fi
    step=$(final_ckpt "$d"); [ -n "$step" ] || continue
    PED_CANDS="$PED_CANDS $tag=$d/checkpoint-$step"
done
[ -n "$SKIPPED" ] && echo "WARNING: missing or incomplete, excluded:$SKIPPED"
echo "candidates:$PED_CANDS"

# ---- 2. generate with the canonical SI, then blind into judge batches -------------------------
mkdir -p "$OUT_DIR"
# shellcheck disable=SC2086
python eval/gen_pedagogy.py --base_model "$BASE_MODEL" \
    --test_file data/socrateach_sft_val.jsonl --candidates $PED_CANDS \
    --si canonical --n_dialogues "$N_DIALOGUES" --out "$OUT_DIR/test_results.jsonl" \
    || die "pedagogy generation failed"

( cd eval/llm_judge && python build_batches.py si/test_results.jsonl si "$N_BATCHES" ) \
    || die "judge-batch build failed"

# Cheap sanity check that the SI actually took effect: in the no-SI batch the base model
# ended a reply with a question 0 times out of 40. If that is still ~0 here, the SI never
# reached the prompt and the whole job is a repeat of the condition we already have.
python - "$OUT_DIR/test_results.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
assert rows and rows[0].get("si") == "canonical", "rows are not tagged si=canonical"
assert any(m["role"] == "system" for m in rows[0]["context"]), "no system message in context"
uniq = {(r["problem"], json.dumps(r["context"], sort_keys=True)) for r in rows}
assert len(uniq) == len(rows), f"dedupe failed: {len(rows)} rows but {len(uniq)} distinct"
print(f"\n{len(rows)} rows, all distinct, si={rows[0]['si']}, {len(rows[0]['outputs'])} candidates")
print(f'{"tag":<16}{"ends with ?":>12}{"has boxed":>11}{"mean chars":>12}')
for tag in rows[0]["outputs"]:
    t = [r["outputs"][tag] for r in rows]
    q = sum(x.rstrip().endswith("?") for x in t)
    b = sum("\\boxed" in x for x in t)
    print(f"{tag:<16}{q:>12}{b:>11}{sum(map(len, t)) / len(t):>12.0f}")
PY

echo "== DONE =="
echo "Next (off-GPU): judge $OUT_DIR/judge_batch_*.json -> judge_out_*.json,"
echo "                then: cd $OUT_DIR && python ../aggregate.py"
echo "Key question: does vanilla SFT reproduce the POC's cell-D 0.84, and do a-T8 / b-T2"
echo "              stay near base once the SI is present?"
