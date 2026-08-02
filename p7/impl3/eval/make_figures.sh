#!/usr/bin/env bash
# Regenerate every Figure-3 variant from one sweep file.
#
# Two axes are deliberately crossed, giving four figures:
#
#   KL condition   no SI (the default, unsuffixed) / with SI. The prior-task probes never carry a
#                  pedagogy system instruction, so the no-SI KL is measured in the condition the
#                  eval actually runs in, and it is the one that collapses both variants onto a
#                  single curve — which is the invariance RL's Razor claims, and the reason it is
#                  the default. It is NOT uniformly better: within variant b alone the with-SI KL
#                  predicts slightly better. The with-SI figures are kept because the gap between
#                  the conditions is itself the finding — variant a disagrees by 7-20x, i.e. it
#                  learned a policy gated on the SI rather than an unconditional one.
#   math prompt    hinted ("put your final answer in \boxed{}") / bare. The hint collides with the
#                  tutor persona and triggers Socratic refusal, so the two measure different
#                  things: hinted mixes refusal with skill loss, bare isolates skill.
#
# Usage:  bash eval/make_figures.sh [sweep.jsonl] [out_dir] [python]
set -euo pipefail

DATA="${1:-out/ckpt_sweep_bare_hint250.jsonl}"
OUT="${2:-out/figures}"
PY="${3:-python}"

[ -f "$DATA" ] || { echo "no sweep file at $DATA" >&2; exit 1; }
mkdir -p "$OUT"

# The unsuffixed figure is the primary one: no-SI KL, hinted GSM8K.
# suffix : prior_key : kl_key : kl_alt
for spec in \
    ":math_hint:kl_ped_noSI:kl_new_SI" \
    "_bare:math_bare:kl_ped_noSI:kl_new_SI" \
    "_withSIkl:math_hint:kl_new_SI:kl_ped_noSI" \
    "_withSIkl_bare:math_bare:kl_new_SI:kl_ped_noSI" ; do
    IFS=: read -r suffix prior kl klalt <<<"$spec"
    echo "=== ${suffix:-(default)}  prior=$prior  kl=$kl ==="
    "$PY" eval/plot_figure3.py --data "$DATA" --out_dir "$OUT" \
        --prior_key "$prior" --kl_key "$kl" --kl_alt "$klalt" --suffix "$suffix" \
        | grep -E "POOLED|saved" || true
done

echo
echo "figures in $OUT:"
ls -1 "$OUT"/fig3_*.png
