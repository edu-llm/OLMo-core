#!/bin/bash
# STANDING PRE-SUBMISSION CHECK. Run this against the FINAL sha before every submission.
#
# WHY THIS EXISTS
# ---------------
# The Exp-2 submission command named `run_exp2.py`, a file that has never existed. It was listed
# as an open item and then written into the command as though it were done.
#
# THE PLATFORM CANNOT CATCH THIS. The offline compiler validates the SHAPE of a submission --
# profile, cost, launcher, checkpoint contract -- not whether command[2] names a file that is
# actually in the image. That submission would have compiled clean, SELF-RELEASED in the AUTOMATIC
# band (no human in the loop at all), pulled a 4.4 GB image, and died on "No such file or
# directory". The cost is the whole run and the wait, for a typo a `git ls-tree` answers free.
#
# THE GENERALISATION, and it is the same shape as the hardcoded-path bug that killed job 1676377:
#
#     A CLAIM ABOUT AN ARTIFACT MUST BE CHECKED AGAINST THE ARTIFACT.
#
# `run_exp2.py` in a command string is a claim about the image. `mqar_data.py` on an absolute
# laptop path is a claim about the filesystem. Neither is verified by re-reading the config that
# makes it -- that is an empty comparison set (EXP2-DESIGN.md Sec 12.4), which always reports
# success. The check has to consult the artifact.
#
# Usage:  ./check_submission.sh [sha]     (defaults to origin/edullm/dynconv-exp2)

set -uo pipefail

REPO_DIR="${REPO_DIR:-/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/claude-01--liv-short-conv-mixer}"
BRANCH="${BRANCH:-origin/edullm/dynconv-exp2}"
PREFIX="experiments/dynconv/exp2"

# Every file an Exp-2 submission command may name. Add to this list, never remove silently.
ENTRY_POINTS=(
  "mqar_harness.py"    # the real entry point -- pilot AND sweep both use it
  "calibration.py"     # baseline-only difficulty recalibration
  "preflight.py"       # the assertion suite; must run before any sweep
  "timing_guard.py"    # physical-impossibility checks on the timing path
)

# Inputs the harness needs at runtime. Absent -> the job dies at import, as job 1676377 did.
RUNTIME_INPUTS=(
  "mqar_data.py"              # the Zoology-faithful generator -- DEFINES the task
  "mqar_calibration.json"     # recorded calibration (FarmShare 1670987)
  "mqar_positive_control.json"
  "arms.py"
  "dynamic_conv.py"
  "sigma.py"
)

cd "$REPO_DIR" || { echo "cannot cd to $REPO_DIR"; exit 2; }
git fetch -q origin "${BRANCH#origin/}" 2>/dev/null
SHA="${1:-$(git rev-parse "$BRANCH" 2>/dev/null)}"
[ -n "$SHA" ] || { echo "cannot resolve a sha"; exit 2; }

echo "=============================================================================="
echo "PRE-SUBMISSION ARTIFACT CHECK"
echo "  sha    : $SHA"
echo "  prefix : $PREFIX"
echo "=============================================================================="

TREE="$(git ls-tree -r "$SHA" --name-only 2>/dev/null)" || { echo "sha not fetched"; exit 2; }
fail=0

echo
echo "-- entry points a command may name --"
for f in "${ENTRY_POINTS[@]}"; do
  if grep -qx "$PREFIX/$f" <<<"$TREE"; then echo "  PRESENT : $f"
  else echo "  ABSENT  : $f   <-- a command naming this dies at exec"; fail=1; fi
done

echo
echo "-- runtime inputs --"
for f in "${RUNTIME_INPUTS[@]}"; do
  if grep -qx "$PREFIX/$f" <<<"$TREE"; then echo "  PRESENT : $f"
  else echo "  ABSENT  : $f   <-- job dies at import"; fail=1; fi
done

# The specific ghost, asserted by name so it can never come back.
echo
if grep -qx "$PREFIX/run_exp2.py" <<<"$TREE"; then
  echo "  NOTE: run_exp2.py now EXISTS -- update the commands deliberately, do not assume."
else
  echo "  CONFIRMED ABSENT: run_exp2.py (the ghost entry point). Commands must name mqar_harness.py."
fi

echo
echo "  files under $PREFIX at this sha: $(grep -c "^$PREFIX/" <<<"$TREE")"
echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS -- every named artifact exists at this sha."
else
  echo "RESULT: FAIL -- do not submit. Fix the command or push the missing file."
fi
exit "$fail"
