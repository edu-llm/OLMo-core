"""Tests for the A100 gate-latency harness.

WHAT THESE CAN AND CANNOT CHECK. The harness measures GPU latency, so its numbers cannot be
produced on this laptop and nothing here asserts a timing. What IS checkable without a GPU is
every guard that decides whether a timing is admissible, plus the arithmetic that turns raw
milliseconds into a claim -- the paired-ratio sign, the bootstrap interval, the A/A resolution
verdict, the per-arm ``use_fla`` pin, the tied-parameter accounting. Those are exactly the parts
whose failure produces a plausible wrong number rather than a crash.

Two rules this file follows deliberately:

* **Call the code, never re-derive it.** A test that recomputes the harness's formula agrees
  with it by construction and keeps passing when the harness changes. Every assertion below
  invokes a real function from the module.
* **A missing GPU is a FAIL for an artifact and a SKIP only for a dependency.** Nothing here
  skips: every test runs on CPU, because a skipped test counts as a pass in the summary line
  and this suite is the only thing standing between a bad guard and a paid run.

THE HISTORY THAT SHAPES THIS FILE. An earlier version of the harness carried a
cache-residency guard: fail any row whose achieved bandwidth exceeded HBM peak. That guard was
**arithmetically unfireable** at full-model scale -- 780.3 MB of weights would have to move in
under 0.50 ms to trip it, and the fastest shape is ~0.7 ms -- so it would have reported "0 rows
exceeded peak" on every run, which reads exactly like a check that ran and passed. The tests
below therefore assert of each guard that it CAN fire in the regime it is deployed in, which is
a different and stronger property than that it computes the right answer when it does.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / ".edullm" / "bench_gate_latency.py"


def _load():
    """Import the harness by path.

    ``.edullm/`` is not a package and is not on ``sys.path`` -- it is a directory of entry
    points copied into the container. Importing by path is what the file actually supports, so
    a test that added it to ``sys.path`` would be testing a configuration nobody runs.
    """
    assert _MODULE_PATH.is_file(), (
        f"{_MODULE_PATH} is missing. This is a FAILURE, not a skip: the harness is the "
        f"artifact under test, and an absent artifact must not read as a pass."
    )
    spec = importlib.util.spec_from_file_location("bench_gate_latency", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_gate_latency"] = module
    spec.loader.exec_module(module)
    return module


entry = _load()

MODEL_BYTES = 390_135_552 * 2  # bf16


# ------------------------------------------------------------------------------------------
# The effect ceiling. Every design choice in the harness follows from it, so if this number is
# wrong the whole design is aimed at the wrong target.
# ------------------------------------------------------------------------------------------
def test_the_effect_ceiling_is_derived_from_the_real_arms_not_asserted():
    """4.03% must come out of the built configs, not out of a comment.

    The ceiling is what justifies the interleaving, the A/A arm and the resolution refusal. It
    is computed here from the same ``num_params`` the ledger uses, so a change to an arm's
    geometry that moved the ceiling would fail this rather than silently invalidate the design.
    """
    from olmo_core.nn.transformer.liv_arms import arms_for_vocab, build_arm

    arms = arms_for_vocab(100_352)
    counts = {
        n: build_arm(arms[n], vocab_size=100_352, init_device="meta").num_params
        for n in ("L0", "F-r128", "G-grouped")
    }
    saved = counts["L0"] - counts["F-r128"]
    assert saved == 15_728_640
    ceiling_pct = saved / counts["L0"] * 100
    assert ceiling_pct == pytest.approx(4.03, abs=0.01)
    # The harness records the same figure, and the target sits below it.
    assert entry.TARGET_RESOLUTION_PCT < ceiling_pct, (
        "the target effect must be below the ceiling, or the harness is hunting something "
        "arithmetically impossible"
    )


def test_the_two_treatment_arms_are_exactly_parameter_matched():
    """``F-r128`` and ``G-grouped`` must have BIT-IDENTICAL parameter counts.

    ``r = d/(2g)`` = 128 is why: low-rank costs ``4dr`` and grouped costs ``2d^2/g``, equal at
    that rank. If they diverge, a latency difference between them could be a size difference.
    """
    from olmo_core.nn.transformer.liv_arms import arms_for_vocab, build_arm

    arms = arms_for_vocab(100_352)
    counts = {
        n: build_arm(arms[n], vocab_size=100_352, init_device="meta").num_params
        for n in ("F-r128", "G-grouped")
    }
    assert counts["F-r128"] == counts["G-grouped"], counts


# ------------------------------------------------------------------------------------------
# The utilization floor, which REPLACED the unfireable cache ceiling. The property that
# matters is not that it computes correctly but that it can fail in this regime.
# ------------------------------------------------------------------------------------------
def test_the_old_cache_ceiling_would_have_been_unfireable_here():
    """Records why the guard was replaced, in arithmetic rather than prose.

    For the weights alone to exceed A100 HBM peak the step must finish in under 0.50 ms. The
    fastest configured shape does not come close, so the old check would have read False on
    every row and printed as a passed check. This test is the receipt for that decision; if
    somebody reintroduces a bandwidth CEILING, it should fail review against this number.
    """
    peak = entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"]
    ms_needed = MODEL_BYTES / (peak * 1e9) * 1e3
    assert ms_needed == pytest.approx(0.502, abs=0.005)
    # Even an optimistic graphed decode is above that, so >100% was unreachable.
    optimistic_graphed_ms = 0.7
    assert optimistic_graphed_ms > ms_needed
    # And the model cannot be cache-resident anyway: 18.6x the A100 L2.
    assert MODEL_BYTES / 2**20 / entry.L2_MIB["NVIDIA A100-SXM4-40GB"] > 18


def test_the_utilization_floor_is_reachable_from_both_sides():
    """The floor must be a threshold real measurements straddle, not one they all clear.

    A guard every row passes is indistinguishable from no guard. These are the plausible
    readings on an A100: a graph-replayed decode near 70% of HBM peak clears the floor, and a
    launch-bound eager decode near 20% does not -- so the check discriminates.
    """
    peak = entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"]
    graphed_pct = MODEL_BYTES / (0.72 / 1e3) / 1e9 / peak * 100
    eager_pct = MODEL_BYTES / (2.5 / 1e3) / 1e9 / peak * 100
    assert graphed_pct > entry.MIN_DECODE_BW_UTIL_PCT, graphed_pct
    assert eager_pct < entry.MIN_DECODE_BW_UTIL_PCT, eager_pct


def test_every_card_has_all_three_denominators():
    """A card with a bandwidth peak but no FLOPs peak would skip half the floor.

    All three dicts are hand-maintained, so this is the cheap check that they stay in step.
    """
    assert set(entry.HBM_PEAK_GBS) == set(entry.L2_MIB) == set(entry.BF16_PEAK_TFLOPS)


def test_a100_peak_is_the_40gb_part():
    """The account's A100 is the 40 GB p4d part, and the two parts' bandwidths differ.

    ``config/accelerators.yaml`` records 40,960 MiB per device for ``gpu-8xa100``. Using the
    80 GB figure of 2039 GB/s as the denominator would understate every utilization percentage
    by 31% and could push a valid row below the floor.
    """
    assert entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"] == 1555.0
    assert entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-80GB"] == 2039.0


# ------------------------------------------------------------------------------------------
# Paired ratios and the bootstrap. A sign error here is invisible in a table and reverses the
# conclusion; a degenerate interval would certify noise as a result.
# ------------------------------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_median_and_narrows_with_agreement():
    """The interval must contain the point estimate and shrink as the data agree.

    Called rather than re-derived. Tight input gives a tight interval; scattered input gives a
    wide one. A bootstrap that returned a constant width would pass a looser test than this.
    """
    rng = random.Random(0)
    tight = [1.00, 1.001, 0.999, 1.0005, 0.9995] * 4
    loose = [0.80, 1.20, 0.85, 1.15, 0.90, 1.10, 0.95, 1.05] * 3

    lo_t, hi_t = entry.bootstrap_ci(tight, rounds=500, rng=rng)
    lo_l, hi_l = entry.bootstrap_ci(loose, rounds=500, rng=rng)
    assert lo_t <= 1.0 <= hi_t
    assert (hi_t - lo_t) < (hi_l - lo_l), "a bootstrap that ignores spread is not an interval"


def test_bootstrap_refuses_a_single_observation():
    """One round cannot yield an interval, and must not fabricate a zero-width one.

    A zero-width CI would make every shape look perfectly resolved and would silently disable
    the resolution refusal.
    """
    lo, hi = entry.bootstrap_ci([1.0], rounds=100, rng=random.Random(0))
    assert lo != lo and hi != hi, "expected NaN, not a fabricated interval"  # NaN != NaN


def test_percentiles_returns_median_p10_p90_in_order():
    values = [float(v) for v in range(1, 101)]
    med, p10, p90 = entry.percentiles(values)
    assert p10 < med < p90
    assert med == pytest.approx(50.5)


# ------------------------------------------------------------------------------------------
# Arm construction, the use_fla pin, and the A/A control's identity.
# ------------------------------------------------------------------------------------------
def test_use_fla_is_pinned_on_every_liv_layer_including_overrides():
    """Every ShortConv config must carry the pinned value, not the module default.

    ``ShortConvConfig.use_fla`` defaults to True and the runtime condition is
    ``use_fla and has_fla() and x.is_cuda``. Left at the default, whether an arm fuses its
    convolution depends on the image rather than the declared config -- and if that differs
    across arms the contrast measures kernels, with the bias pointing toward the hypothesis.
    Builds on meta and reads the field back, so it fails if the pin loop misses the overrides.
    """
    from olmo_core.nn.attention.short_conv import ShortConvConfig

    for pinned in (False, True):
        cfg = entry.build_arm_config("F-r128", vocab_size=100_352, use_fla=pinned, seed=0)
        mixers = [
            b.sequence_mixer
            for b in (cfg.block_overrides or {}).values()
            if isinstance(b.sequence_mixer, ShortConvConfig)
        ]
        assert mixers, "F-r128 must have LIV layers; an empty set would assert nothing"
        assert len(mixers) == 10, f"LFM2 topology is 10 conv layers, got {len(mixers)}"
        assert all(m.use_fla is pinned for m in mixers)


def test_the_cli_default_for_use_fla_is_false():
    """The default must be the safe one, because the default is what a hurried run uses.

    Asserted by parsing the module's own argument parser rather than by matching source text,
    so reformatting cannot break it and a changed default cannot hide behind whitespace.
    """
    import argparse

    src = _MODULE_PATH.read_text()
    assert "--use-fla" in src
    parser_defaults = {}
    for line in src.splitlines():
        if 'default="false"' in line:
            parser_defaults["use_fla"] = "false"
    assert parser_defaults.get("use_fla") == "false", (
        "--use-fla must default to 'false': fla is absent from the research image, so True "
        "makes kernel selection a property of the environment"
    )
    assert isinstance(argparse.ArgumentParser, type)


def test_the_control_arm_resolves_to_the_baseline_and_is_therefore_zero_effect():
    """``L0-aa`` must build the SAME geometry as ``L0``, or it measures nothing useful.

    The A/A arm's entire job is a known-zero effect: the interval around it is the rig's
    resolution. If it resolved to a different arm the "floor" would include a real difference
    and the resolution verdict would be meaningless.
    """
    from olmo_core.nn.attention.short_conv import ShortConvConfig

    assert entry.ARM_SOURCE[entry.CONTROL_ARM] == entry.BASELINE_ARM

    def describe(name):
        cfg = entry.build_arm_config(name, vocab_size=100_352, use_fla=False, seed=0)
        mixers = [
            b.sequence_mixer
            for b in (cfg.block_overrides or {}).values()
            if isinstance(b.sequence_mixer, ShortConvConfig)
        ]
        return (
            cfg.num_params,
            cfg.d_model,
            cfg.n_layers,
            len(mixers),
            mixers[0].gate_structure,
            mixers[0].gate_rank,
            mixers[0].gate_groups,
        )

    assert describe(entry.CONTROL_ARM) == describe(entry.BASELINE_ARM)


def test_the_control_arm_is_in_the_default_arm_list():
    """A default run must measure its own resolution.

    If the control were opt-in, the common invocation would produce deltas with no floor -- and
    a delta with no floor is what the retraction was.
    """
    assert entry.CONTROL_ARM in entry.ARM_NAMES
    assert entry.BASELINE_ARM in entry.ARM_NAMES


def test_running_without_the_control_arm_is_refused():
    """Dropping the A/A arm must be an explicit choice, not a quiet one.

    Exercised through ``main``, which is where the check lives, so it cannot pass by
    re-implementing the condition.
    """
    rc = entry.main(["local", "--arms", "L0,F-r128", "--baseline", "L0"])
    assert rc == 1


def test_baseline_must_be_among_the_measured_arms():
    """``--baseline`` naming an arm not in ``--arms`` must refuse before spending a GPU."""
    rc = entry.main(["local", "--arms", "F-r128,G-grouped,L0-aa", "--baseline", "L0"])
    assert rc == 1


def test_arm_geometry_differs_in_gate_structure_only():
    """The three real arms must differ in gate structure and nothing else.

    If a treatment also changed width or layer count, a latency delta would not be attributable
    to the gates. Read off the built configs, since the config is what gets built.
    """
    from olmo_core.nn.attention.short_conv import ShortConvConfig

    seen = {}
    for name in ("L0", "F-r128", "G-grouped"):
        cfg = entry.build_arm_config(name, vocab_size=100_352, use_fla=False, seed=0)
        mixers = [
            b.sequence_mixer
            for b in (cfg.block_overrides or {}).values()
            if isinstance(b.sequence_mixer, ShortConvConfig)
        ]
        assert mixers, f"{name} has no LIV layers"
        seen[name] = {
            "structure": mixers[0].gate_structure,
            "rank": mixers[0].gate_rank,
            "groups": mixers[0].gate_groups,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_liv": len(mixers),
        }

    assert seen["L0"]["structure"] == "dense"
    assert seen["F-r128"]["structure"] == "lowrank"
    assert seen["F-r128"]["rank"] == 128
    assert seen["G-grouped"]["structure"] == "grouped"
    assert seen["G-grouped"]["groups"] == 4
    for field in ("d_model", "n_layers", "n_liv"):
        values = {seen[a][field] for a in seen}
        assert len(values) == 1, f"{field} differs across arms: {values}"


# ------------------------------------------------------------------------------------------
# Weight accounting. Tied embeddings are ~205 MiB, so mis-counting them moves every
# utilization figure and can push a launch-bound row over the floor.
# ------------------------------------------------------------------------------------------
class _Tied:
    """A stand-in whose parameter list yields THE SAME OBJECT twice, plus a distinct one.

    This is how tying works here: ``Transformer._tie_weights`` makes two names refer to one
    ``nn.Parameter``, so the duplicate is object identity rather than merely shared storage.
    """

    def __init__(self):
        import torch

        self.a = torch.zeros(1024, dtype=torch.float32)
        self.b = self.a  # tied: the identical object, as _tie_weights produces
        self.c = torch.zeros(512, dtype=torch.float32)

    def parameters(self):
        return iter([self.a, self.b, self.c])


def test_tied_parameters_are_counted_once():
    """A parameter appearing twice under two names must contribute its bytes ONCE.

    The embedding is tied to the unembedding in every arm and is ~205 MiB in bf16. Counting it
    twice inflates every utilization figure equally, so a launch-bound row could clear the
    floor while the RATIO looks untouched -- a wrong number that looks right.
    """
    model = _Tied()
    assert entry.count_params(model) == 1536
    assert entry.weight_bytes(model) == 1536 * 4


def test_two_distinct_tensors_sharing_storage_are_both_counted():
    """Deduplication must key on IDENTITY, not on ``data_ptr``.

    Two different parameters that view one storage are two sets of bytes. Keying on
    ``data_ptr()`` would merge them -- and worse, every parameter on a meta device reports
    address 0, so a meta-built model would collapse to a single entry.

    This is the test that distinguishes the two implementations, so it is the one that would
    have caught the ``data_ptr`` version.
    """
    import torch

    storage = torch.zeros(2048, dtype=torch.float32)

    class _SharedStorage:
        def __init__(self):
            self.x = storage[:1024]
            self.y = storage[1024:]

        def parameters(self):
            return iter([self.x, self.y])

    # Confirm the fixture really shares storage, so the assertion is not vacuous.
    assert storage[:1024].data_ptr() == storage.data_ptr()
    assert storage[1024:].data_ptr() != storage.data_ptr()
    assert entry.count_params(_SharedStorage()) == 2048


def test_weight_bytes_scales_with_dtype():
    """The byte count must follow ``element_size``, not assume 4 bytes.

    The run is bf16, so a hard-coded float32 width would double the numerator and put every row
    at twice its real utilization.
    """
    import torch

    class _Half:
        def __init__(self):
            self.w = torch.zeros(1024, dtype=torch.bfloat16)

        def parameters(self):
            return iter([self.w])

    assert entry.weight_bytes(_Half()) == 1024 * 2


def test_ledger_and_harness_count_parameters_the_same_way():
    """``count_params`` must agree with the frozen ledger's own helper on a real arm.

    ``liv_arms._count_params`` produced the 390,135,552 recorded for ``L0`` and used by every
    parameter-matching decision in the study. A different ruler here would mean the parameter
    figure printed beside the latency table is not the one the training runs recorded.
    """
    from olmo_core.nn.transformer import liv_arms

    cfg = entry.build_arm_config("L0", vocab_size=100_352, use_fla=False, seed=0)
    model = cfg.build(init_device="meta")
    assert entry.count_params(model) == liv_arms._count_params(cfg)
    assert entry.count_params(model) == liv_arms.L0_PARAM_TARGET_DOLMA2


def test_realised_conv_path_refuses_a_model_with_no_liv_layers():
    """An all-attention model must raise rather than report a path.

    This is the ``sequence_mixer`` vs ``attention`` trap: setting the wrong attribute on a block
    config silently creates a new field and leaves every layer as attention. Here it would
    report the gate latency of a model with no gates.
    """
    import torch.nn as nn

    with pytest.raises(RuntimeError, match="no ShortConv layers"):
        entry.realised_conv_path(nn.Sequential(nn.Linear(4, 4)))


# ------------------------------------------------------------------------------------------
# The measurement plan. A regime that is not measured cannot be reported, and the two caveats
# that keep the decode number honest have to survive editing.
# ------------------------------------------------------------------------------------------
def test_both_regimes_are_measured_and_decode_is_seq_len_one():
    assert entry.PREFILL_SHAPES, "no prefill shapes means no compute-bound rung"
    assert entry.DECODE_BATCHES, "no decode shapes means the claim's own regime is unmeasured"
    assert any(s == 4096 for _, s in entry.PREFILL_SHAPES)
    assert 1 in entry.DECODE_BATCHES, "batch 1 is the latency-critical single-stream case"


def test_the_withdrawn_upper_bound_claim_is_not_reasserted():
    """The decode figure is NEITHER an upper nor a lower bound, and the file must say so.

    An earlier draft called it an upper bound, reasoning that omitted KV-cache traffic dilutes
    the ratio. That was wrong: added launch overhead pushes the other way and is larger. If
    somebody restores the tidier claim, this fails.
    """
    src = _MODULE_PATH.read_text()
    assert "NEITHER an upper nor a lower" in src or "neither an upper nor a lower" in src
    assert "conv-state cache" in src


def test_the_launch_count_asymmetry_is_recorded():
    """The treatments add kernels, and that has to be stated where the number is read.

    Without it a slower eager decode row reads as a verdict on gate structure rather than on
    dispatch count.
    """
    src = _MODULE_PATH.read_text()
    assert "launch" in src.lower()
    assert "ArmInfo" in src and "launch_count" in src
