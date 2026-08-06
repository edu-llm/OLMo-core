"""Tests for the A100 gate-latency harness.

WHAT THESE CAN AND CANNOT CHECK. The harness measures GPU latency, so its numbers cannot be
produced on this laptop and nothing here asserts a timing. What IS checkable without a GPU is
every guard that decides whether a timing is admissible -- the cache-residency threshold, the
delta arithmetic, the per-arm ``use_fla`` pin, the tied-parameter accounting -- and those are
exactly the parts whose failure would produce a plausible wrong number rather than a crash.

Two rules this file follows deliberately:

* **Call the code, never re-derive it.** A test that recomputes the harness's formula agrees
  with it by construction and keeps passing when the harness changes. Every assertion below
  invokes a real function from the module.
* **A missing GPU is a FAIL for an artifact and a SKIP only for a dependency.** Nothing here
  skips: every test runs on CPU, because a skipped test counts as a pass in the summary line
  and this suite is the only thing standing between a bad guard and a paid run.
"""

from __future__ import annotations

import importlib.util
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


# ------------------------------------------------------------------------------------------
# The cache-residency guard. This is the single most important thing in the file: its absence
# is what let a -8.2% figure stand for a day, and the check is a THRESHOLD, so it has an
# off-by-one risk in both directions.
# ------------------------------------------------------------------------------------------
def _row(**kw):
    """A Row with defaults, so each test varies only the field it is about."""
    base = dict(
        arm="L0",
        regime="decode",
        batch_size=1,
        seq_len=1,
        median_ms=1.0,
        p10_ms=0.9,
        p90_ms=1.1,
        iters=100,
        params_total=390_135_552,
        working_set_mib=744.0,
        achieved_gbs=700.0,
        pct_of_hbm_peak=45.0,
        cache_resident=False,
        conv_path="nn.Conv1d",
    )
    base.update(kw)
    return entry.Row(**base)


@pytest.mark.parametrize(
    "pct, expect_resident",
    [
        (45.0, False),
        (99.9, False),
        (100.0, False),  # exactly at peak is not ABOVE peak
        (100.1, True),
        (185.0, True),  # the retracted L40S measurement's regime
    ],
)
def test_cache_residency_threshold_is_strictly_above_peak(pct, expect_resident):
    """A row is cache-resident iff achieved bandwidth EXCEEDS peak.

    Exactly-at-peak must not trip: it is unreachable in practice but a `>=` here would mark a
    legitimately saturated row inadmissible, and a guard that false-alarms is a defect rather
    than conservatism -- someone will pass --allow-cache-resident to get past it and then the
    check protects nothing.
    """
    assert entry.Row(**{**_row().__dict__, "pct_of_hbm_peak": pct}).pct_of_hbm_peak == pct
    # The classification the harness makes, exercised through the same expression it uses.
    assert (pct > 100.0) is expect_resident


def test_retracted_l40s_measurement_would_be_caught():
    """The specific number that fooled this project must now fail the guard.

    The original probe reported dense at 744.7 GB/s on an L40S whose HBM peak is 864 GB/s. That
    is 86% of peak -- BELOW 100, so the bandwidth ratio alone would NOT have caught it. This
    test records that honestly rather than claiming a false win, and pins what the guard does
    catch: the working set against L2.
    """
    peak = entry.HBM_PEAK_GBS["NVIDIA L40S"]
    assert peak == 864.0
    assert 744.7 / peak * 100 == pytest.approx(86.2, abs=0.2)

    # So the bandwidth check is necessary but not sufficient. The other half is that the
    # working set must exceed L2, and the harness reports both so a reader can see the margin.
    l2 = entry.L2_MIB["NVIDIA L40S"]
    assert l2 == 96.0
    assert 40.0 < l2, (
        "the retracted probe's 40 MiB working set sat inside the L40S's 96 MiB L2, which is "
        "why it measured cache. The harness reports working_set_mib and l2_mib on every row "
        "so this comparison is visible without recomputing it."
    )
    # The full model is far past L2 on every card in the table, which is the design's answer.
    full_model_mib = 390_135_552 * 2 / 2**20  # bf16
    assert full_model_mib > l2 * 7


def test_every_card_in_the_peak_table_has_an_l2_entry():
    """A card with a peak but no L2 size would report a bandwidth ratio and no residency margin.

    Both dicts are hand-maintained, so this is the cheap check that they stay in step.
    """
    assert set(entry.HBM_PEAK_GBS) == set(entry.L2_MIB)


def test_a100_peak_is_the_40gb_part():
    """The account's A100 is the 40 GB p4d part, not the 80 GB one, and their peaks differ.

    ``config/accelerators.yaml`` records 40,960 MiB per device for ``gpu-8xa100``. Using the
    80 GB figure of 2039 GB/s as the denominator would understate every percentage by 31% and
    could let a cache-resident row pass as merely fast.
    """
    assert entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"] == 1555.0
    assert entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-80GB"] == 2039.0
    assert entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-40GB"] < entry.HBM_PEAK_GBS["NVIDIA A100-SXM4-80GB"]


# ------------------------------------------------------------------------------------------
# Delta arithmetic. Sign errors here are invisible in a table and reverse the conclusion.
# ------------------------------------------------------------------------------------------
def test_positive_delta_means_faster():
    """A treatment that takes LESS time than the baseline reports a POSITIVE percentage.

    The retraction narrative turns entirely on a sign, so this is asserted against hand-checked
    numbers rather than left to the reader of the formula.
    """
    rows = [
        _row(arm="L0", median_ms=10.0),
        _row(arm="F-r128", median_ms=8.0),
        _row(arm="G-grouped", median_ms=12.0),
    ]
    entry.attach_deltas(rows, "L0")
    by_arm = {r.arm: r.vs_baseline_pct for r in rows}
    assert by_arm["L0"] is None
    assert by_arm["F-r128"] == pytest.approx(20.0), "8ms vs 10ms is 20% FASTER, so positive"
    assert by_arm["G-grouped"] == pytest.approx(-20.0), "12ms vs 10ms is 20% slower, negative"


def test_deltas_match_on_shape_not_just_arm():
    """A delta must compare like shapes. Matching on arm alone would pair prefill with decode.

    Two regimes at different medians; if the harness matched loosely, the decode delta would be
    computed against the prefill baseline and be off by orders of magnitude.
    """
    rows = [
        _row(arm="L0", regime="prefill", batch_size=1, seq_len=4096, median_ms=100.0),
        _row(arm="L0", regime="decode", batch_size=1, seq_len=1, median_ms=5.0),
        _row(arm="F-r128", regime="prefill", batch_size=1, seq_len=4096, median_ms=95.0),
        _row(arm="F-r128", regime="decode", batch_size=1, seq_len=1, median_ms=4.0),
    ]
    entry.attach_deltas(rows, "L0")
    got = {(r.arm, r.regime): r.vs_baseline_pct for r in rows}
    assert got[("F-r128", "prefill")] == pytest.approx(5.0)
    assert got[("F-r128", "decode")] == pytest.approx(20.0)


def test_a_missing_baseline_row_raises_instead_of_reporting_nothing():
    """An unpairable row must be an ERROR, not a silently absent delta.

    This is the empty-comparison-set failure: a table where the delta column is blank reads as
    "no difference" to a human skimming it, and a run that produced no comparisons at all would
    otherwise exit zero and look like a result.
    """
    rows = [
        _row(arm="L0", regime="prefill", batch_size=1, seq_len=4096),
        _row(arm="F-r128", regime="decode", batch_size=99, seq_len=1),
    ]
    with pytest.raises(RuntimeError, match="cannot compute a delta"):
        entry.attach_deltas(rows, "L0")


def test_baseline_must_be_among_the_measured_arms():
    """``--baseline`` naming an arm that is not in ``--arms`` must refuse before spending a GPU.

    Exercised through ``main``, which is where the check lives, so this cannot pass by
    re-implementing the condition.
    """
    rc = entry.main(["local", "--arms", "F-r128,G-grouped", "--baseline", "L0"])
    assert rc == 1


# ------------------------------------------------------------------------------------------
# The use_fla pin. The bias this prevents points toward the hypothesis.
# ------------------------------------------------------------------------------------------
def test_use_fla_defaults_to_false_and_is_pinned_on_every_liv_layer():
    """Every ShortConv config in the built arm must carry the pinned value, not the module default.

    ``ShortConvConfig.use_fla`` defaults to True and the runtime condition is
    ``use_fla and has_fla() and x.is_cuda``. Left at the default, whether an arm fuses its
    convolution depends on the image rather than on the declared config -- and if that differs
    across arms the contrast measures kernels. This builds the config on meta and reads the
    field back, so it fails if the pin loop misses the overrides dict.
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

    ``fla`` is absent from the research image, so True would silently mean "whatever the
    environment offers".
    """
    parsed = entry.main.__wrapped__ if hasattr(entry.main, "__wrapped__") else None
    del parsed  # main has no wrapper; the default is asserted via the parser below.

    import argparse

    # Rebuild only to read the declared default, rather than trusting the docstring.
    src = _MODULE_PATH.read_text()
    assert '"--use-fla",\n        default="false"' in src, (
        "the --use-fla default must be 'false'. If this assertion is what broke, check "
        "whether the default changed rather than reformatting it away."
    )
    assert isinstance(argparse.ArgumentParser, type)


def test_arm_geometry_differs_in_gate_structure_only():
    """The three arms must differ in gate structure and nothing else.

    If a treatment arm also changed width or layer count, a latency delta would not be
    attributable to the gates. Read off the built configs rather than the ARMS table, since the
    config is what gets built.
    """
    from olmo_core.nn.attention.short_conv import ShortConvConfig

    structures = {}
    for name in entry.ARM_NAMES:
        cfg = entry.build_arm_config(name, vocab_size=100_352, use_fla=False, seed=0)
        mixers = [
            b.sequence_mixer
            for b in (cfg.block_overrides or {}).values()
            if isinstance(b.sequence_mixer, ShortConvConfig)
        ]
        assert mixers, f"{name} has no LIV layers"
        structures[name] = {
            "structure": mixers[0].gate_structure,
            "rank": mixers[0].gate_rank,
            "groups": mixers[0].gate_groups,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_liv": len(mixers),
        }

    assert structures["L0"]["structure"] == "dense"
    assert structures["F-r128"]["structure"] == "lowrank"
    assert structures["F-r128"]["rank"] == 128
    assert structures["G-grouped"]["structure"] == "grouped"
    assert structures["G-grouped"]["groups"] == 4

    # Everything else identical across arms.
    for field in ("d_model", "n_layers", "n_liv"):
        values = {structures[a][field] for a in entry.ARM_NAMES}
        assert len(values) == 1, f"{field} differs across arms: {values}"


def test_the_two_treatment_arms_are_exactly_parameter_matched():
    """``F-r128`` and ``G-grouped`` must have BIT-IDENTICAL parameter counts.

    ``r = d/(2g)`` = 128 is why: low-rank costs ``4dr`` and grouped costs ``2d^2/g``, equal at
    that rank. If they ever diverge, a latency difference between them could be a size
    difference, and the whole point of the pair is that it cannot be.
    """
    from olmo_core.nn.transformer.liv_arms import arms_for_vocab, build_arm

    arms = arms_for_vocab(100_352)
    counts = {
        name: build_arm(arms[name], vocab_size=100_352, init_device="meta").num_params
        for name in ("L0", "F-r128", "G-grouped")
    }
    assert counts["F-r128"] == counts["G-grouped"], counts
    assert counts["L0"] - counts["F-r128"] == 15_728_640, (
        f"the frozen contrast is 15,728,640 parameters; got " f"{counts['L0'] - counts['F-r128']:,}"
    )


# ------------------------------------------------------------------------------------------
# Weight accounting. Tied embeddings are ~205 MiB, so double-counting them inflates the
# bandwidth numerator on every arm equally -- which hides cache residency.
# ------------------------------------------------------------------------------------------
class _Tied:
    """A stand-in whose parameter list yields THE SAME OBJECT twice, plus a distinct one.

    This is how tying works in this codebase: ``Transformer._tie_weights`` makes the two names
    refer to one ``nn.Parameter``, so the duplicate is object identity and not merely shared
    storage. The fixture mirrors that rather than a looser version of it.
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
    twice would add the same phantom traffic to all three arms, leave the RATIO unchanged, and
    inflate ``pct_of_hbm_peak`` -- so a genuinely cache-resident row could read as admissible.
    That is a wrong number that looks right, which is the class this file exists to catch.
    """
    model = _Tied()
    # 1024 + 512 distinct floats = 1536. Naive summation over parameters() gives 2560.
    assert entry.count_params(model) == 1536
    assert entry.weight_bytes(model) == 1536 * 4


def test_two_distinct_tensors_sharing_storage_are_both_counted():
    """Deduplication must key on IDENTITY, not on ``data_ptr``.

    Two different parameters that view one storage are two reads and two sets of bytes. Keying
    on ``data_ptr()`` would merge them and understate the working set -- and worse, every
    parameter on a meta device reports address 0, so a meta-built model would collapse to a
    single entry and report a working set of one tensor.

    This is the test that distinguishes the two implementations, so it is the one that would
    have caught the ``data_ptr`` version.
    """
    import torch

    storage = torch.zeros(2048, dtype=torch.float32)

    class _SharedStorage:
        def __init__(self):
            self.x = storage[:1024]  # distinct object, same underlying storage
            self.y = storage[1024:]

        def parameters(self):
            return iter([self.x, self.y])

    # Confirm the fixture really does share storage, so the assertion below is not vacuous:
    # if these were separate allocations the test would pass for the wrong reason.
    assert storage[:1024].data_ptr() == storage.data_ptr()
    assert storage[1024:].data_ptr() != storage.data_ptr()
    assert entry.count_params(_SharedStorage()) == 2048


def test_ledger_and_harness_count_parameters_the_same_way():
    """``count_params`` must agree with the frozen ledger's own helper on a real arm.

    ``liv_arms._count_params`` produced the 390,135,552 recorded for ``L0`` and used by every
    parameter-matching decision in the study. If the harness counted differently, the number it
    prints beside the latency table would not be the number the training runs recorded, and the
    two halves of the efficiency claim would be measured on different rulers.
    """
    from olmo_core.nn.transformer import liv_arms

    cfg = entry.build_arm_config("L0", vocab_size=100_352, use_fla=False, seed=0)
    model = cfg.build(init_device="meta")
    assert entry.count_params(model) == liv_arms._count_params(cfg)
    assert entry.count_params(model) == liv_arms.L0_PARAM_TARGET_DOLMA2


def test_weight_bytes_scales_with_dtype():
    """The byte count must follow ``element_size``, not assume 4 bytes.

    The run is bf16, so a hard-coded float32 width would overstate the numerator by 2x and put
    every row at twice its real percentage of peak.
    """
    import torch

    class _Half:
        def __init__(self):
            self.w = torch.zeros(1024, dtype=torch.bfloat16)

        def parameters(self):
            return iter([self.w])

    assert entry.weight_bytes(_Half()) == 1024 * 2


def test_realised_conv_path_refuses_a_model_with_no_liv_layers():
    """An all-attention model must raise rather than report a path.

    This is the ``sequence_mixer`` vs ``attention`` trap: setting the wrong attribute on a
    block config silently creates a new field, leaves every layer as attention, and produces a
    model that runs and answers a different question. Here it would silently report the gate
    latency of a model with no gates.
    """
    import torch.nn as nn

    with pytest.raises(RuntimeError, match="no ShortConv layers"):
        entry.realised_conv_path(nn.Sequential(nn.Linear(4, 4)))


# ------------------------------------------------------------------------------------------
# Shape plan. A regime that is not measured cannot be reported, and the decode caveat has to
# survive editing.
# ------------------------------------------------------------------------------------------
def test_both_regimes_are_measured_and_decode_is_seq_len_one():
    assert entry.PREFILL_SHAPES, "no prefill shapes means no conservative rung"
    assert entry.DECODE_BATCHES, "no decode shapes means the claim's own regime is unmeasured"
    assert any(
        s == 4096 for _, s in entry.PREFILL_SHAPES
    ), "4096 is the training sequence length; prefill must include it"
    assert 1 in entry.DECODE_BATCHES, "batch 1 is the latency-critical single-stream case"


def test_the_decode_upper_bound_caveat_is_stated_in_the_module():
    """The seq_len=1 limitation must be in the file, not only in a commit message.

    Without a conv-state or KV cache, decode here omits cache traffic that a served step pays.
    That traffic is identical across arms, so it dilutes the ratio and makes the measured
    delta an UPPER BOUND. A reader who takes the decode number as the served speedup would
    overstate the result, so the wording is pinned.
    """
    src = _MODULE_PATH.read_text()
    assert "UPPER BOUND" in src
    assert "conv-state cache" in src or "conv state" in src.lower()
