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
def test_no_bandwidth_ceiling_has_been_reintroduced():
    """A bandwidth CEILING must not come back, and this detects it rather than describing it.

    An earlier version of this test only recomputed the arithmetic below and claimed in its
    docstring that a reintroduced ceiling "should fail review against this number". Nothing in
    it detected one. This does: it greps the harness for the field and flag the old guard used.

    The arithmetic is kept because it is the reason, and it is worth being able to cite:
    """
    src = _MODULE_PATH.read_text()
    assert "cache_resident" not in src, (
        "a cache-residency ceiling has been reintroduced. It cannot fire at full-model scale "
        "(see the arithmetic below) and will print as a passed check."
    )
    assert "fail-on-cache-resident" not in src

    # Why: tripping >100% of A100 HBM peak needs the whole model to move in under 0.502 ms.
    peak = entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"]
    ms_needed = MODEL_BYTES / (peak * 1e9) * 1e3
    assert ms_needed == pytest.approx(0.502, abs=0.005)
    # The fastest configured shape is well above that, so the ceiling was unreachable.
    assert 0.7 > ms_needed
    # And the model cannot be cache-resident anyway: 18.6x the A100 L2.
    assert MODEL_BYTES / 2**20 / entry.L2_MIB["NVIDIA A100-SXM4-40GB"] > 18


A100_GBS = 1555.0
A100_TFLOPS = 312.0


def _util(**kw):
    """Call the real ``utilization`` with A100 denominators and sensible defaults."""
    args = dict(
        regime="decode",
        mode="graphed",
        median_ms=0.72,
        weight_bytes_=MODEL_BYTES,
        flops_per_token=0,
        batch_size=1,
        seq_len=1,
        peak_gbs=A100_GBS,
        peak_tflops=A100_TFLOPS,
    )
    args.update(kw)
    return entry.utilization(**args)


def test_the_utilization_floor_is_reachable_from_both_sides():
    """The floor must be a threshold real measurements straddle, not one they all clear.

    A guard every row passes is indistinguishable from no guard. Calls the real function on the
    plausible A100 readings: a graph-replayed decode near 70% of HBM peak clears it, a
    launch-bound eager-speed decode near 20% does not.

    This calls ``utilization`` rather than re-deriving the formula, which is the difference that
    matters. The previous version of this test recomputed the arithmetic on literals, so
    inverting the comparison, swapping the two peaks, or deleting ``utilization_ok`` entirely
    all left it green.
    """
    fast = _util(median_ms=0.72)
    assert fast["pct_of_hbm_peak"] > entry.MIN_DECODE_BW_UTIL_PCT
    assert fast["utilization_ok"] is True

    slow = _util(median_ms=2.5)
    assert slow["pct_of_hbm_peak"] < entry.MIN_DECODE_BW_UTIL_PCT
    assert slow["utilization_ok"] is False
    assert "launch-bound" in slow["utilization_note"]


def test_eager_decode_is_exempt_from_the_floor_and_says_so():
    """Eager decode is EXPECTED launch-bound, so failing it would fire on designed behaviour.

    A guard that false-alarms gets an --allow flag passed to silence it, and then it protects
    nothing. The row still reports its bandwidth; it just is not judged on it.
    """
    row = _util(mode="eager", median_ms=2.5)
    assert row["utilization_ok"] is None
    assert "exempt" in row["utilization_note"]
    assert row["pct_of_hbm_peak"] is not None, "the number is still reported, just not judged"


#: Forward-only FLOPs per token for the real 390M model, derived rather than invented:
#: ``num_flops_per_token`` returns ``6 * params`` (training), and inference is the forward third.
#: Using a made-up figure here is how the first version of these tests came to assert that a
#: perfectly healthy prefill row was below the floor.
REAL_FWD_FLOPS_PER_TOKEN = int(6 * 390_135_552 / 3)


def test_prefill_is_judged_on_flops_and_never_on_weight_bandwidth():
    """Prefill must not carry a weights-only bandwidth ratio.

    At batch 4 x 4096 the logits tensor alone is 3.29 GB, four times the weight footprint, and
    real traffic is 15-20 GB against the 0.78 GB a weights-only ratio counts -- a ~20x
    understatement. So the bandwidth fields stay None and the FLOPs fields carry the verdict.
    """
    row = _util(
        regime="prefill",
        mode="eager",
        median_ms=100.0,
        flops_per_token=REAL_FWD_FLOPS_PER_TOKEN,
        batch_size=4,
        seq_len=4096,
    )
    assert row["pct_of_hbm_peak"] is None
    assert row["achieved_gbs"] is None
    assert row["pct_of_flops_peak"] is not None
    assert row["utilization_ok"] is not None


def test_the_prefill_floor_passes_a_realistic_row_and_fails_a_starved_one():
    """The floor must clear a plausible prefill and reject a starved one.

    Both sides matter. A floor that fails healthy measurements gets an --allow flag passed to
    silence it; a floor nothing can fail is decoration. The healthy figure is derived from the
    real model (12.8 TFLOP at b=4 s=4096, so ~41% of A100 peak at 100 ms) rather than chosen to
    make the assertion pass.
    """
    healthy = _util(
        regime="prefill",
        mode="eager",
        median_ms=100.0,
        flops_per_token=REAL_FWD_FLOPS_PER_TOKEN,
        batch_size=4,
        seq_len=4096,
    )
    assert healthy["pct_of_flops_peak"] == pytest.approx(41.0, abs=1.0)
    assert healthy["utilization_ok"] is True

    # Same work, 2.5x slower: 16% of peak, which is not a compute-bound region.
    starved = _util(
        regime="prefill",
        mode="eager",
        median_ms=250.0,
        flops_per_token=REAL_FWD_FLOPS_PER_TOKEN,
        batch_size=4,
        seq_len=4096,
    )
    assert starved["utilization_ok"] is False
    assert "compute-bound" in starved["utilization_note"]


def test_a_nonpositive_timing_is_rejected_rather_than_dividing():
    """A zero or negative elapsed time must fail, not raise or produce infinity."""
    row = _util(median_ms=0.0)
    assert row["utilization_ok"] is False
    assert "not usable" in row["utilization_note"]


def test_forward_only_flops_fraction_is_one_third():
    """The codebase's ``num_flops_per_token`` is the 6x TRAINING convention.

    ``short_conv.py``'s own docstring warns that a mixer using the 2x convention "would report a
    third of its true cost". This harness measures inference, so it must take the forward third.
    Leaving the 6x figure would make the denominator 3x too large, drive every prefill row to a
    third of its real utilization, and fail the floor on a healthy measurement.
    """
    assert entry.FORWARD_ONLY_FLOPS_FRACTION == pytest.approx(1 / 3)


def test_lastlogit_rows_are_not_charged_the_full_lm_head():
    """``logits_to_keep=1`` removes ~23% of prefill work, so it must not be billed for it.

    Charging the full-head figure would overstate those rows' utilization by ~1.3x and let them
    clear the floor for free. Calls the real re-pricing helper.
    """
    head = 205_000_000
    info = entry.ArmInfo(
        arm="L0",
        source_arm="L0",
        params_total=390_135_552,
        weight_bytes=MODEL_BYTES,
        conv_path="nn.Conv1d",
        gate_structure="dense",
        flops_per_token_4096=880_000_000,
    )
    full = entry.arm_flops_per_token(info, regime="prefill", seq_len=4096, lm_head_flops=head)
    trimmed = entry.arm_flops_per_token(
        info, regime="prefill-lastlogit", seq_len=4096, lm_head_flops=head
    )
    assert trimmed < full, "the lastlogit row must be charged less, not the same"
    # And both are forward-only, i.e. a third of the recorded 6x figure.
    assert full == pytest.approx(880_000_000 / 3, rel=1e-6)


# ------------------------------------------------------------------------------------------
# The clock-spread guard. It was gated on a value that is always None in the research image,
# which made it unfireable -- the same disease as the cache ceiling it sits beside.
# ------------------------------------------------------------------------------------------
def test_clock_spread_distinguishes_unmeasured_from_agreeing():
    """``None`` must mean "not measured", never "the clocks agree".

    ``pynvml`` is absent from the research image, so every clock reads None and the spread is
    None. If the guard treated that as 0% it would pass silently on every run -- which is
    exactly what it did before the refusal was added.
    """
    assert entry.clock_spread_pct([None, None, None]) is None
    assert entry.clock_spread_pct([1410.0]) is None, "one arm cannot have a spread"
    assert entry.clock_spread_pct([1410.0, 1410.0]) == pytest.approx(0.0)


def test_clock_spread_exceeds_the_limit_on_a_realistic_drift():
    """A 3% swing must trip the 2% limit; a 0.5% swing must not.

    An A100-SXM4's SM clock moves 4-10% between idle and sustained bf16 load, so this is the
    magnitude the guard exists for -- and it is larger than the 1.8% effect being hunted.
    """
    tripping = entry.clock_spread_pct([1410.0, 1410.0, 1368.0])
    assert tripping is not None and tripping > entry.MAX_CLOCK_SPREAD_PCT, tripping
    fine = entry.clock_spread_pct([1410.0, 1410.0, 1403.0])
    assert fine is not None and fine < entry.MAX_CLOCK_SPREAD_PCT, fine


def test_missing_nvml_is_refused_by_default():
    """The CLI must not proceed without a drift receipt unless told to in as many words."""
    src = _MODULE_PATH.read_text()
    assert "--allow-unmeasured-clocks" in src
    assert "allow_unmeasured_clocks" in src


# ------------------------------------------------------------------------------------------
# The A/A resolution verdict. Width alone is not enough: a narrow interval that excludes zero
# describes a rig with a systematic bias between two byte-identical models.
# ------------------------------------------------------------------------------------------
def _cell(**kw):
    args = dict(
        arm=entry.CONTROL_ARM,
        regime="decode",
        mode="graphed",
        batch_size=1,
        seq_len=1,
        rounds=20,
        median_ms=1.0,
        p10_ms=0.99,
        p90_ms=1.01,
    )
    args.update(kw)
    return entry.Cell(**args)


def test_a_tight_interval_that_excludes_zero_is_still_unresolvable():
    """THE finding this function was extracted for.

    An A/A control reading +3.0% with CI [+2.8, +3.3] has a width of 0.5pp, so a width-only
    check certifies it as able to resolve 1.8%. But the control's true effect is EXACTLY zero,
    so that interval says the rig has a 3pp systematic bias between two identical models. The
    guard the design calls "the one that can fail for the right reason" was passing for the
    wrong one.
    """
    biased = _cell(vs_baseline_pct=3.0, ci_low_pct=2.8, ci_high_pct=3.3)
    lines, unresolvable = entry.resolution_verdict([biased])
    assert unresolvable, "a 3pp bias between identical models must not be called resolvable"
    assert any("excludes zero" in line for line in lines)


def test_a_wide_interval_bracketing_zero_is_also_unresolvable():
    """The width condition must still hold: noisier than the effect means it cannot be seen."""
    noisy = _cell(vs_baseline_pct=0.1, ci_low_pct=-4.0, ci_high_pct=4.2)
    _lines, unresolvable = entry.resolution_verdict([noisy])
    assert unresolvable


def test_a_tight_interval_bracketing_zero_passes():
    """The one case that should pass: narrow AND centred on zero."""
    good = _cell(vs_baseline_pct=0.05, ci_low_pct=-0.4, ci_high_pct=0.5)
    lines, unresolvable = entry.resolution_verdict([good])
    assert unresolvable == []
    assert any("OK" in line for line in lines)


def test_no_control_cell_is_a_failure_not_an_absence():
    """A run with no A/A measurement must not report a reassuring blank.

    Reachable in practice with ``--baseline L0-aa``, which makes the control the baseline so it
    gets no ratio. Before this, that path printed "NOT MEASURED" and exited zero.
    """
    lines, unresolvable = entry.resolution_verdict([])
    assert unresolvable, "an unmeasured floor must appear in the failure list"
    assert any("NOT MEASURED" in line for line in lines)


def test_the_verdict_ignores_cells_from_other_arms():
    """Only the control arm's cells define the floor.

    A treatment arm's tight interval must not be mistaken for evidence about the rig.
    """
    treatment = _cell(arm="F-r128", vs_baseline_pct=1.9, ci_low_pct=1.5, ci_high_pct=2.3)
    lines, unresolvable = entry.resolution_verdict([treatment])
    assert any(
        "NOT MEASURED" in line for line in lines
    ), "a treatment cell must not be read as the control"
    assert unresolvable


# ------------------------------------------------------------------------------------------
# The launch counter, whose profiler field was renamed between torch versions.
# ------------------------------------------------------------------------------------------
class _Ev:
    def __init__(self, key, count, device_time=None, cuda_time=None):
        self.key = key
        self.count = count
        if device_time is not None:
            self.device_time_total = device_time
        if cuda_time is not None:
            self.cuda_time_total = cuda_time


def test_launch_counter_reads_the_current_field_name():
    """torch 2.9 exposes ``device_time_total``; the old spelling was ``cuda_time_total``.

    The stub exposes ONLY the modern field, which is what makes this discriminating: a version
    reading only ``cuda_time_total`` finds nothing here and must return ``(None, None)``.

    An earlier version of this test used a stub that set the field it was checking for AND kept
    rows exposing neither, so switching the harness to the legacy spelling alone left all 46
    tests green. A mutation run caught that; this is the repair.
    """
    total, copies = entry.summarise_profile(
        [
            _Ev("aten::addmm", 10, device_time=500),
            _Ev("aten::copy_", 4, device_time=20),
        ]
    )
    assert total == 14
    assert copies == 4


def test_launch_counter_returns_none_when_no_event_exposes_a_known_field():
    """A third rename must surface as "unknown", never as zero kernels.

    Zero reads as a measurement; None becomes a refusal in ``main``. This is the case the old
    leniency swallowed -- it kept unrecognised rows and counted them, so a renamed field produced
    a confident total assembled from rows whose device time was never read.
    """

    class _Renamed:
        def __init__(self):
            self.key = "aten::addmm"
            self.count = 10
            self.gpu_time_total = 500  # neither spelling the harness knows

    assert entry.summarise_profile([_Renamed()]) == (None, None)


def test_launch_counter_still_reads_the_legacy_field_name():
    """An older torch must not silently report zero kernels."""
    total, copies = entry.summarise_profile(
        [_Ev("aten::addmm", 7, cuda_time=100), _Ev("aten::contiguous", 3, cuda_time=5)]
    )
    assert total == 10
    assert copies == 3


def test_launch_counter_returns_none_on_an_empty_profile():
    """No events must be None, so the caller can refuse rather than print 0."""
    assert entry.summarise_profile([]) == (None, None)


def test_missing_launch_counts_are_refused_by_default():
    src = _MODULE_PATH.read_text()
    assert "--allow-missing-launch-counts" in src
    assert "allow_missing_launch_counts" in src


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

    Asserted by PARSING the module's own parser, which is what the previous version of this test
    only claimed to do -- it scanned source lines for the literal ``default="false"`` and set a
    flag on any match, never tying it to this argument. Flipping this default to ``"true"``
    passed so long as some other option still carried ``"false"``.
    """
    opts = entry.build_parser().parse_args([])
    assert opts.use_fla == "false", (
        "--use-fla must default to 'false': fla is absent from the research image, so True "
        "makes kernel selection a property of the environment rather than of the config"
    )


def test_the_guard_bypasses_all_default_to_off():
    """Every --allow-* escape hatch must be off by default.

    Each one disables a check that exists because a previous version of this harness shipped a
    number it should not have. A default-on bypass is the same as no check.
    """
    opts = entry.build_parser().parse_args([])
    assert opts.allow_unresolvable is False
    assert opts.allow_unmeasured_clocks is False
    assert opts.allow_missing_launch_counts is False
    assert (
        opts.skip_graphed is False
    ), "graphed decode must be on by default; eager alone ranks arms by launch count"


def test_the_default_arm_list_includes_the_control():
    """Parsed from the CLI, not read off the constant, since the CLI is what a run uses."""
    opts = entry.build_parser().parse_args([])
    arms = opts.arms.split(",")
    assert entry.CONTROL_ARM in arms
    assert entry.BASELINE_ARM in arms
    assert "F-r128" in arms and "G-grouped" in arms


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


def test_running_without_the_control_arm_is_refused(capsys):
    """Dropping the A/A arm must be an explicit choice, not a quiet one.

    **Asserts the MESSAGE, not just the exit code.** An earlier version checked only
    ``rc == 1``, and on any CPU host ``main`` returns 1 from the ``torch.cuda.is_available()``
    check long before reaching this guard -- so deleting the guard entirely left the test green.
    It was passing on the wrong return. The guard has since been moved above the torch import
    (it is pure string work), and this now proves which refusal fired.
    """
    rc = entry.main(["local", "--arms", "L0,F-r128", "--baseline", "L0"])
    assert rc == 1
    assert "A/A control arm" in capsys.readouterr().err


def test_the_control_arm_may_be_dropped_deliberately(capsys):
    """With ``--allow-unresolvable`` the run must get PAST the control check.

    This is the negative control for the test above: it proves the refusal is conditional on the
    flag rather than unconditional. It still exits 1 on a CPU host, but for the CUDA reason, and
    asserting which reason is the whole point.
    """
    rc = entry.main(["local", "--arms", "L0,F-r128", "--baseline", "L0", "--allow-unresolvable"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "A/A control arm" not in err, "the flag must suppress this refusal"
    assert "no CUDA device" in err, "and the run must then stop for the honest reason"


def test_baseline_must_be_among_the_measured_arms(capsys):
    """``--baseline`` naming an arm not in ``--arms`` must refuse before spending a GPU.

    Same history as the control-arm test: this used to pass on the CUDA return.
    """
    rc = entry.main(["local", "--arms", "F-r128,G-grouped,L0-aa", "--baseline", "L0"])
    assert rc == 1
    assert "is not in --arms" in capsys.readouterr().err


def test_argv_validation_happens_before_the_torch_import(capsys):
    """The ordering is load-bearing and is asserted rather than trusted.

    A guard that only runs on a GPU host cannot protect you from misconfiguring a GPU run. Both
    argv checks are pure string work, so they belong first -- and the proof is that on this
    CPU-only host the argv message appears INSTEAD of the CUDA message.
    """
    entry.main(["local", "--arms", "F-r128", "--baseline", "L0"])
    err = capsys.readouterr().err
    assert "is not in --arms" in err
    assert "no CUDA device" not in err, (
        "the argv guard must fire first; if the CUDA message appears the ordering regressed and "
        "these guards are unreachable on every CPU host"
    )


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


def test_two_distinct_tensors_at_one_address_are_both_counted():
    """Deduplication must key on IDENTITY, not on ``data_ptr``.

    The fixture holds two DIFFERENT objects that report the SAME address, which is the only
    configuration that separates the two implementations: id-keyed counts both (4096), ptr-keyed
    merges them (2048).

    An earlier version of this test used ``storage[:1024]`` and ``storage[1024:]`` and proved in
    its own body that their addresses DIFFER -- so a ptr-keyed implementation passed it too, and
    it was vacuous with respect to the thing it claimed to catch.
    """
    import torch

    storage = torch.zeros(2048, dtype=torch.float32)
    view_a = storage.view(2048)
    view_b = storage[:]

    # The premise, asserted rather than assumed: distinct objects, one address.
    assert view_a is not view_b
    assert view_a.data_ptr() == view_b.data_ptr() == storage.data_ptr()

    class _Aliased:
        def parameters(self):
            return iter([view_a, view_b])

    assert entry.count_params(_Aliased()) == 4096, (
        "id-keyed dedup must count both; a data_ptr-keyed version returns 2048 here, which is "
        "exactly the bug this test exists to catch"
    )


def test_meta_device_parameters_do_not_collapse():
    """The other half of the ``data_ptr`` bug, and the more dangerous one in practice.

    Every tensor on a meta device reports address 0, so a ptr-keyed dedup collapses an entire
    model to one parameter. ``build_arm_config`` builds on meta, so this is the path the harness
    actually takes.
    """
    import torch

    with torch.device("meta"):
        a = torch.zeros(1024, dtype=torch.float32)
        b = torch.zeros(512, dtype=torch.float32)

    assert a.data_ptr() == b.data_ptr() == 0, "premise: meta tensors share address 0"

    class _Meta:
        def parameters(self):
            return iter([a, b])

    assert entry.count_params(_Meta()) == 1536, (
        "a data_ptr-keyed dedup would return 1024 here, silently reporting one parameter's "
        "worth of working set for a whole model"
    )


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
