#!/usr/bin/env bash
# Mutation test for the gated-convolution suite.
#
# WHY THIS EXISTS
#   A green suite proves nothing about whether the tests could ever go red. This project has
#   shipped four vacuous greens in a single build -- an identity init, an expectation read from the
#   mutated field, an empty getattr walk, and a CPU-unreachable fused branch -- each of which
#   passed while asserting nothing. So every mutation below breaks the source in a way that ought
#   to matter, and the suite must catch it. A SURVIVING mutation is an unguarded defect.
#
#   Each mutation names the test that should catch it, so a survivor points at the specific gap
#   rather than just at "something is wrong".
#
# USAGE
#   bash scripts/mutate_gated_conv.sh
#
# Restores the files on every exit path, including a failure or an interrupt.

set -uo pipefail
cd "$(dirname "$0")/.."

SRC=src/olmo_core/nn/gated_convolution.py
REC=src/olmo_core/nn/attention/recurrent.py
TESTS=src/test/nn/gated_convolution_test.py
PY="${PY:-/Users/ericwu/Developer/Capstone_LLM/OLMo-core/.venv/bin/python}"

cp "$SRC" "$SRC.orig"
cp "$REC" "$REC.orig"
restore() { mv -f "$SRC.orig" "$SRC" 2>/dev/null; mv -f "$REC.orig" "$REC" 2>/dev/null; }
trap restore EXIT INT TERM

caught=0
survived=0
noop=0

# A MUTATION THAT DOES NOT APPLY IS THE WORST OUTCOME, AND IT LOOKS LIKE A RESULT.
#
# M12's perl regex stopped matching after the source was reworded, so the "mutation" was a no-op
# and the run reported SURVIVED -- which reads as an unguarded defect that was not there. Reword it
# the other way and the same no-op reports "caught", i.e. a green from a test that was never
# challenged. Either way the number is fiction, and nothing in the output said so.
#
# So run_case now diffs the source against the pristine copy FIRST. No change means NO-OP, which is
# a hard failure, not a verdict.
run_case() {
  local name="$1" expect="$2"
  local out
  if cmp -s "$SRC" "$SRC.orig" && cmp -s "$REC" "$REC.orig"; then
    echo "  NO-OP     $name"
    echo "            neither source file changed, so the pattern matched nothing."
    echo "            A verdict here would be fiction whichever way it fell -- fix the pattern."
    noop=$((noop + 1))
    restore
    cp "$SRC" "$SRC.orig"
    cp "$REC" "$REC.orig"
    return
  fi
  out=$(PYTHONNOUSERSITE=1 PYTHONPATH=src "$PY" -m pytest -q "$TESTS" -p no:randomly 2>&1)
  if grep -qE "^[0-9]+ passed" <<<"$out" && ! grep -q "failed" <<<"$out"; then
    echo "  SURVIVED  $name"
    echo "            expected '$expect' to catch it -- UNGUARDED DEFECT"
    survived=$((survived + 1))
  else
    local hit
    hit=$(grep -oE "FAILED [^ ]+::[A-Za-z0-9_]+" <<<"$out" | sed 's/.*:://' | sort -u | tr '\n' ' ')
    echo "  caught    $name"
    echo "            by: ${hit:-<unnamed>}"
    if [ -n "$expect" ] && ! grep -q "$expect" <<<"$out"; then
      echo "            NOTE: '$expect' did not fire; a different test caught it"
    fi
    caught=$((caught + 1))
  fi
  restore
  cp "$SRC" "$SRC.orig"
  cp "$REC" "$REC.orig"
}

# A MUTATION RESULT IS ONLY VALID FOR THE HOST IT RAN ON.
#
# M12 was CAUGHT on a laptop without 'fla' and SURVIVED on a GPU host with it -- the most
# misleading shape a result can take, because the laptop reported a pass. Cause: without 'fla',
# KimiDeltaAttention's 'assert has_fla()' fires before the check under test, so the laptop failed
# for an unrelated reason. So the environment is printed with the verdicts, and every claim about
# this script's output must name the host it came from.
echo "Environment (a mutation verdict does not transfer between these):"
PYTHONNOUSERSITE=1 PYTHONPATH=src "$PY" - <<'PYEOF'
import torch
try:
    import fla
    fla_v = fla.__version__
except ImportError:
    fla_v = None
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}  fla={fla_v}")
if fla_v is None:
    print("  WARNING: fla is absent, so the 7 fla-gated tests SKIP and any mutation they")
    print("           would have caught will report as SURVIVED-or-caught by accident.")
PYEOF
echo

echo "Baseline (must be fully green before any mutation means anything):"
base=$(PYTHONNOUSERSITE=1 PYTHONPATH=src "$PY" -m pytest -q "$TESTS" -p no:randomly 2>&1 | tail -1)
echo "  $base"
if grep -q "failed" <<<"$base"; then
  echo "ABORT: the baseline is already red, so no mutation result is interpretable."
  exit 1
fi
echo

# M1: make the gate a CONSTANT per-channel rescale, i.e. absorbable into the depthwise taps.
# This is the mutation that matters most: it is the scientifically vacuous version of the module,
# and it trains stably. If it survives, the whole experiment can be vacuous undetected.
echo "M1  gate becomes a constant (absorbable) rescale"
perl -0pi -e 's/2\.0 \* torch\.sigmoid\(pre_scale \* u\),\n(\s+)2\.0 \* torch\.sigmoid\(post_scale \* u\),/2.0 * torch.sigmoid(pre_scale).expand_as(u),\n${1}2.0 * torch.sigmoid(post_scale).expand_as(u),/' "$SRC"
run_case "constant gate" "test_the_gate_is_not_a_constant_rescale"

# M2: zero BOTH lowrank factors -- the dead-branch bug this suite already caught once for real.
echo "M2  lowrank zeroes both factors (dead branch)"
perl -0pi -e 's/init_linear\(self\.gate_down, std=std, generator=generator\)/_apply_init(_zero, self.gate_down.weight)/' "$SRC"
run_case "dead lowrank branch" "test_gate_gradient_is_alive_at_init"

# M3: gate no longer starts at 1.0, so the two arms do not start from the same function.
echo "M3  gate does not start neutral"
perl -0pi -e 's/return 2\.0 \* torch\.sigmoid/return 1.9 * torch.sigmoid/; s/2\.0 \* torch\.sigmoid\(pre_scale \* u\)/1.9 * torch.sigmoid(pre_scale * u)/' "$SRC"
run_case "non-neutral init" "test_at_init_the_gated_module_equals_the_plain_convolution"

# M4: apply the activation BEFORE the convolution instead of after. A different operator, and one
# that trains -- the exact class of silent error short_conv.py's docstring warns about.
echo "M4  activation moves before the convolution"
perl -0pi -e 's/        z = z\.transpose\(-1, -2\)\n        if self\.activation in \("silu", "swish"\):\n            z = F\.silu\(z\)/        z = z.transpose(-1, -2)\n        if False:\n            z = F.silu(z)/' "$SRC"
run_case "activation dropped" "test_activation_is_applied_after_the_convolution"

# M5: ignore cu_seqlens, letting a k=4 filter read across a document boundary.
echo "M5  cu_seqlens ignored (filter crosses documents)"
perl -0pi -e 's/        if cu_seqlens is not None:\n            if u\.shape\[0\] != 1:/        if False:\n            if u.shape[0] != 1:/' "$SRC"
run_case "document boundary crossed" "test_cu_seqlens_stops_the_filter_at_a_document_boundary"

# M6: break causality by padding on both sides, so the filter reads the future.
echo "M6  convolution is no longer causal"
perl -0pi -e 's/\)\[\.\.\., :seq_len\]\n        z = z\.transpose/)[..., (self.kernel_size - 1) : (self.kernel_size - 1 + seq_len)]\n        z = z.transpose/' "$SRC"
run_case "non-causal filter" "test_output_is_causal"

# M7: gate parameter arithmetic off by the number of gates -- a plausible slip that silently
# unbalances a parameter-matched arm ledger.
echo "M7  gate_param_count off by a factor of 2"
perl -0pi -e 's/        return 2 \* hidden_size\n    if structure == "lowrank":/        return hidden_size\n    if structure == "lowrank":/' "$SRC"
run_case "wrong depthwise param count" "test_depthwise_gate_param_count_is_two_per_channel"

# M8: gate_activation_bytes forgets that autograd retains two tensors per gate, halving the memory
# figure a run would be sized from.
echo "M8  memory estimate halved"
perl -0pi -e 's/    return 2 \* tensors_per_gate \* per_tensor/    return tensors_per_gate * per_tensor/' "$SRC"
run_case "halved memory estimate" "test_gate_activation_bytes_matches_the_documented_kda_geometry"

# M9a: the CONSTRUCTOR default flips to gated. Note this is a different surface from the config
# field below -- M9a survived the first run of this script, because every test went through the
# config. Anything instantiating the module directly would have got the treatment arm silently.
echo "M9a constructor default flips to True"
perl -0pi -e 's/        gated_conv: bool = False,/        gated_conv: bool = True,/' "$REC"
run_case "constructor default changed" "test_kda_constructor_default_is_also_ungated"

# M9b: the CONFIG field flips to gated, silently invalidating every existing KDA measurement.
echo "M9b config default flips to True"
perl -0pi -e 's/^    gated_conv: bool = False$/    gated_conv: bool = True/m' "$REC"
run_case "config default changed" "test_the_default_is_the_shipped_operator"

# M10: the gated arm gets silu even when configured activation-free, collapsing two of the three
# arms onto each other.
echo "M10 gated arm silently keeps silu"
perl -0pi -e 's/            activation=self\.gated_conv_activation,  # type: ignore\[arg-type\]/            activation="silu",/' "$REC"
run_case "arms collapse" "test_kda_build_conv_honours_the_configured_activation"

# M10b: gate_input is never threaded, so a lowrank arm raises on step 1 -- or worse, is passed to a
# plain CausalConv1d that does not accept it.
echo "M10b gate_input never threaded"
perl -0pi -e 's/        if self\.gated_conv and self\.gate_structure == "lowrank":\n            return \{"gate_input": x\}/        if False:\n            return {"gate_input": x}/' "$REC"
run_case "gate_input dropped" "test_kda_conv_kwargs_thread_gate_input_only_for_lowrank"

# M11: num_params stops counting the gate, so an arm ledger is wrong by 12,288/layer.
echo "M11 num_params forgets the gate"
perl -0pi -e 's/        params \+= self\.gate_params\(d_model\)/        params += 0/' "$REC"
run_case "gate omitted from num_params" "test_depthwise_gate_costs_12288_params_per_layer"

# M12: a lowrank config without a rank is accepted instead of refused, so it builds something
# unintended rather than failing.
# M12: the config stops refusing a lowrank arm with no rank. Matched on the CONDITION line alone,
# not on the multi-line raise, because the message text is the thing most likely to be reworded --
# and a pattern that stops matching turns this into a no-op with a fictional verdict.
echo "M12 lowrank without a rank is accepted"
perl -pi -e 's/^        if self\.gate_structure == "lowrank" and self\.gate_rank is None:$/        if False:/' "$REC"
run_case "missing rank accepted" "test_lowrank_without_a_rank_is_refused"

# M13: the "gate options set but gated_conv is False" refusal is removed, so a config that reads as
# a treatment arm in a YAML diff trains as the control.
echo "M13 gate options without gated_conv are silently ignored"
perl -pi -e 's/^            if self\.gate_rank is not None:$/            if False:/' "$REC"
run_case "gate options ignored" "test_gate_options_without_gated_conv_are_refused"

# M14: the plain convolution's activation is hard-coded again, so the arm that ISOLATES the gate
# becomes unbuildable and the arm set silently drops to three cells that vary three things at once.
echo "M14 plain conv activation hard-coded to silu"
perl -pi -e 's/^                activation=self\.conv_activation,  # type: ignore\[arg-type\]$/                activation="silu",/' "$REC"
run_case "no-activation arm unbuildable" "test_the_no_activation_plain_arm_is_buildable"

# M15: an unrecognised activation string is accepted. fla's kernel matches 'silu'/'swish' EXACTLY
# and otherwise runs activation-free without error, so "Silu" would silently be the None arm.
echo "M15 unrecognised activation accepted at construction"
perl -0pi -e 's/        if activation not in \(None, "silu", "swish"\):/        if False:/' "$SRC"
run_case "bad activation accepted" "test_an_unrecognised_activation_is_refused_at_construction"

# M16: reset_parameters stops zeroing the gates, so anything that builds the module without calling
# a mixer's init_weights runs on uninitialized memory -- which is often all zeros, making a broken
# module look merely inert.
echo "M16 reset_parameters leaves the gates uninitialised"
perl -0pi -e 's/            self\.pre_scale\.zero_\(\)\n            self\.post_scale\.zero_\(\)/            pass/' "$SRC"
run_case "gates left uninitialised" "test_reset_parameters_leaves_the_gate_neutral"

# M17: gate_structure is no longer checked against gated_conv=False, so a config reading as a
# lowrank treatment arm trains as the plain control.
echo "M17 gate_structure ignored when gated_conv is False"
perl -0pi -e 's/            if self\.gate_structure != "depthwise":/            if False:/' "$REC"
run_case "lowrank-labelled control" "test_gate_options_without_gated_conv_are_refused"

echo
echo "=========================================="
echo "caught   $caught"
echo "SURVIVED $survived"
echo "NO-OP    $noop"
if [ "$noop" -gt 0 ]; then
  echo
  echo "$noop mutation(s) did not apply. Their verdicts are fiction, so the totals above do not"
  echo "mean what they say. Fix the patterns before reading anything else here."
  exit 1
fi
if [ "$survived" -gt 0 ]; then
  echo
  echo "A surviving mutation is an unguarded defect. Add the assertion before trusting the suite."
  exit 1
fi
echo "Every mutation applied, and every one was caught."
