#!/bin/bash
# Submit a whole sweep as separate SLURM jobs (one GPU each). Run from the PROJECT root
# (the post-training/ dir) so relative paths resolve:
#
#     bash clusters/orcd/submit_sweep.sh impl2                 # vanilla Impl-2 baseline
#     bash clusters/orcd/submit_sweep.sh impl3 a               # Impl-3 variant a, all T
#     bash clusters/orcd/submit_sweep.sh impl3 b <vanilla_sft> # Impl-3 variant b (needs a vanilla SFT)
#     bash clusters/orcd/submit_sweep.sh impl4                 # Impl-4 distilled-fraction sweep
set -euo pipefail

KIND="${1:?usage: submit_sweep.sh impl2|impl3|impl4 [args]}"
SBATCH="clusters/orcd/run.sbatch"
PROJECT="$(pwd)"
export PROJECT

submit() {  # $1 = command string, $2 = job name
    echo "sbatch -> $2"
    sbatch -J "$2" --export=ALL,PROJECT="$PROJECT",CMD="$1" "$SBATCH"
}

case "$KIND" in
  impl2)
    submit "python impl1_2_prompting_sft/train_sft.py --config impl1_2_prompting_sft/config.yaml --output_dir out/impl2-sft --resume auto" "p7-impl2"
    ;;
  impl3)
    VARIANT="${2:?impl3 needs a variant: a or b}"
    EXTRA=""
    [ "$VARIANT" = "b" ] && EXTRA="--sft_model_id ${3:?variant b needs a vanilla SFT path}"
    for T in 2 4 8 16 32; do   # ladder shifted up 2026-07-29 (low T unstable); keep in sync with impl3_h200.sbatch
      submit "python impl3_kl_reweighted_sft/train_kl_sft.py --variant $VARIANT --temperature $T $EXTRA --config impl3_kl_reweighted_sft/config.yaml --resume auto" "p7-impl3-$VARIANT-T$T"
    done
    ;;
  impl4)
    for f in 0 0.25 0.5 0.75 1.0; do
      submit "python impl4_sdft/train_sdft.py --distilled_frac $f --gold_dir data --distilled_pedagogy impl4_sdft/distilled/pedagogy_rewrite.jsonl --distilled_general impl4_sdft/distilled/general_domains.jsonl --config impl4_sdft/config.yaml --resume auto" "p7-impl4-f$f"
    done
    ;;
  *)
    echo "unknown sweep kind: $KIND"; exit 1;;
esac
echo "Submitted. Monitor: squeue -u \$USER ; tail -f logs/*.out"
