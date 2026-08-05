"""Tests for mqar_harness.py and calibration.py.

The load-bearing ones assert against MEASURED numbers read out of the recorded JSONs
(``mqar_calibration.json`` FarmShare 1670987, ``mqar_positive_control.json`` 1670928), so a drift in
either the pinned table or the source file breaks a test. The ``test_negative_control_*`` tests prove
each guard can fail.

Budget note: every test here runs with tiny ``steps`` under the explicit smoke flag. The refusal
guard itself is tested separately, without training.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibration as C  # noqa: E402
import mqar_harness as H  # noqa: E402
import sigma as S  # noqa: E402

SMOKE_STEPS = 3
SMOKE_BATCH = 4
SMOKE_EVAL = S.MIN_EVAL_ITEMS  # never reduced: the >=1000-item rule is not a budget knob


@pytest.fixture(scope="module")
def cfg_small():
    """Cheapest well-posed config: 3*D <= seq_len, vocab 256."""
    return H.MQARConfig(seq_len=64, num_pairs=4, vocab_size=H.CALIBRATED_VOCAB)


# ======================================================================================
# Geometry and the spec's declared constants
# ======================================================================================


def test_geometry_matches_spec_section_7():
    """d_model=128, n_layers=6, vocab 256, R=16, hybrid has 2 of 6 attention, allliv has 0."""
    assert H.D_MODEL == 128
    assert H.N_LAYERS == 6
    assert H.CALIBRATED_VOCAB == 256, "vocab 256, NOT Zoology's 8192"
    assert H.GENERATOR_RANK == 16, "R=16 at d=128, R/d = 1/8, a PRE-REGISTERED deviation"
    assert len(H.HYBRID_ATTENTION_LAYERS) == 2
    assert all(0 <= i < H.N_LAYERS for i in H.HYBRID_ATTENTION_LAYERS)
    assert H.ALLLIV_ATTENTION_LAYERS == ()
    assert len(H.ARMS) == 4
    assert H.ARM_CODES == {"static": "S1", "permuted": "S2", "dynqkv": "S3", "dynamic": "S4"}


def test_calibrated_budget_is_512000_examples():
    """SPEC Sec 4.7: 8000 x 64 = 512,000. The budget is part of the calibration, not a free knob."""
    assert H.CALIBRATED_STEPS == 8000
    assert H.CALIBRATED_BATCH_SIZE == 64
    assert H.CALIBRATED_EXAMPLES == 512_000
    assert H.CALIBRATED_LR == 3e-3


def test_s3_is_undefined_in_allliv_and_is_not_silently_substituted():
    """SPEC Sec 1.2: report N/A; do NOT substitute S1."""
    assert ("dynqkv", "allliv") in H.UNDEFINED_CELLS
    assert ("dynqkv", "hybrid") not in H.UNDEFINED_CELLS
    with pytest.raises(ValueError, match="UNDEFINED"):
        H.stub_build_model(
            arm="dynqkv", topology="allliv", kernel_size=3, vocab_size=256,
            d_model=128, n_layers=6, seed=0,
        )


# ======================================================================================
# PAIRING -- data order shared, init per-arm. R3 F3.
# ======================================================================================


def test_data_seed_is_identical_across_arms_and_init_seed_is_not():
    """
    The pairing contract. Same pair => byte-identical data seed for every arm; init seeds differ,
    because init-seed pairing is mechanically impossible when arms have different tensors.
    """
    for pair in (0, 1, 7, 42):
        seeds = {H.data_seed_for(pair) for _ in H.ARMS}
        assert len(seeds) == 1, "data seed must not depend on the arm"
        inits = {H.derive_init_seed(pair, a) for a in H.ARMS}
        assert len(inits) == len(H.ARMS), "init seeds must be distinct per arm"


def test_data_order_is_bit_identical_across_arms_at_the_same_pair():
    """
    THE PAIRING, verified on the actual token stream: two arms at the same ``seed_pair`` must draw
    ``torch.equal`` batches in the same order. This is the only thing the power analysis may assume.
    """
    cfg = H.MQARConfig(seq_len=64, num_pairs=4, vocab_size=256)
    streams = {}
    for arm in ("static", "dynamic"):
        g = torch.Generator().manual_seed(H.data_seed_for(3))
        streams[arm] = [H.make_mqar_batch(cfg, 4, g) for _ in range(5)]
    for (ta, la), (tb, lb) in zip(streams["static"], streams["dynamic"]):
        assert torch.equal(ta, tb), "training batches must be bit-identical across arms"
        assert torch.equal(la, lb)


def test_negative_control_different_pairs_give_different_data():
    """If two pairs produced the same data, 'paired seeds' would be one seed run twice."""
    cfg = H.MQARConfig(seq_len=64, num_pairs=4, vocab_size=256)
    a = H.make_mqar_batch(cfg, 4, torch.Generator().manual_seed(H.data_seed_for(0)))[0]
    b = H.make_mqar_batch(cfg, 4, torch.Generator().manual_seed(H.data_seed_for(1)))[0]
    assert not torch.equal(a, b)


def test_derive_init_seed_is_stable_and_insensitive_to_adding_arms():
    """
    A hash, not an index arithmetic. Adding an arm must not shift any existing arm's seed, which
    would silently invalidate a partially-complete sweep.
    """
    assert H.derive_init_seed(0, "static") == H.derive_init_seed(0, "static")
    assert H.derive_init_seed(0, "static") != H.derive_init_seed(1, "static")
    assert 0 <= H.derive_init_seed(99, "dynamic") < 2**31
    # A hash of the arm NAME cannot depend on how many arms exist.
    assert H.derive_init_seed(5, "dynamic") == int.from_bytes(
        __import__("hashlib").sha256(b"exp2|pair=5|arm=dynamic").digest()[:4], "big"
    ) % (2**31)


def test_stub_builder_is_deterministic_given_the_seed():
    kw = dict(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
              d_model=32, n_layers=2, seed=11)
    m1, m2 = H.stub_build_model(**kw), H.stub_build_model(**kw)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "same seed must give bit-identical params (torch.equal)"
    m3 = H.stub_build_model(**{**kw, "seed": 12})
    assert not all(
        torch.equal(a, b) for a, b in zip(m1.parameters(), m3.parameters())
    ), "a different seed must actually change the init"


# ======================================================================================
# The two Zoology gotchas
# ======================================================================================


def test_random_non_queries_is_pinned_false(cfg_small):
    """
    Gotcha 1: the configs set ``False`` while the CLASS DEFAULT IS ``True``. With ``True`` the filler
    is random tokens that can collide with keys -- a harder, different task.
    """
    assert cfg_small.random_non_queries is False
    H.assert_zoology_gotchas(cfg_small)  # must not raise


def test_negative_control_random_non_queries_true_is_rejected():
    """Proof the guard can fail: flip the gotcha and the harness must refuse."""
    bad = H.MQARConfig(seq_len=64, num_pairs=4, vocab_size=256, random_non_queries=True)
    with pytest.raises(AssertionError, match="GOTCHA 1"):
        H.assert_zoology_gotchas(bad)


def test_state_mixer_identity_must_be_DECLARED_not_assumed():
    """
    Gotcha 2. A model that does not declare the property is REFUSED. Per ``green that means
    nothing``, a default-True lookup would make this check unable to fail.
    """
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=0)
    H.assert_state_mixer_identity(m)  # declares True

    class Undeclared(torch.nn.Module):
        pass

    with pytest.raises(AssertionError, match="GOTCHA 2"):
        H.assert_state_mixer_identity(Undeclared())

    m.state_mixer_is_identity = False
    with pytest.raises(AssertionError, match="GOTCHA 2"):
        H.assert_state_mixer_identity(m)


def test_conv_path_has_no_activation():
    """
    SPEC Sec 5 trap 3: ``CausalConv1d`` defaults to ``activation="silu"``; real LFM2 has none. The
    stub must model that, or a harness smoke test would validate the wrong operator.
    """
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=0)
    convs = [x for x in m.mixers if isinstance(x, H._StubShortConv)]
    assert len(convs) == 2
    for c in convs:
        assert c.activation is None


# ======================================================================================
# Budget refusal
# ======================================================================================


def test_refuses_under_budget_outside_smoke():
    """
    Job 1670963's exact failure: 3000 x 32 = 96,000 examples, a 5.3x shortfall, produced a table
    that read as "too hard" instead of "under-trained".
    """
    with pytest.raises(RuntimeError, match="REFUSING TO RUN UNDER-BUDGET"):
        H.check_budget(3000, 32, smoke=False)
    assert 3000 * 32 == 96_000
    assert H.CALIBRATED_EXAMPLES / 96_000 == pytest.approx(5.333, abs=0.01)
    # The calibrated budget itself passes, and smoke permits anything.
    H.check_budget(H.CALIBRATED_STEPS, H.CALIBRATED_BATCH_SIZE, smoke=False)
    H.check_budget(3, 4, smoke=True)


def test_run_cell_refuses_under_budget(cfg_small):
    with pytest.raises(RuntimeError, match="UNDER-BUDGET"):
        H.run_cell(
            arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
            build_model=H.stub_build_model, steps=5, batch_size=4, smoke=False, verbose=False,
        )


def test_calibrate_refuses_under_budget():
    with pytest.raises(RuntimeError, match="UNDER-BUDGET"):
        C.calibrate(build_model=H.stub_build_model, steps=10, batch_size=4, smoke=False)


# ======================================================================================
# Eval: >=1000 items, clustered SEs, both endpoints
# ======================================================================================


def test_eval_refuses_fewer_than_1000_items(cfg_small):
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=0)
    with pytest.raises(ValueError, match="MIN_EVAL_ITEMS"):
        H.evaluate(m, cfg_small, n_items=999)


def test_eval_reports_both_endpoints_and_a_clustered_se(cfg_small):
    """
    Accuracy AND query NLL, with clustered SEs. At init the NLL must be ~ln(vocab) and the accuracy
    near the 1/vocab guess level -- magnitudes, not existence.
    """
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=0)
    ev = H.evaluate(m, cfg_small, n_items=1000, batch_size=100)
    assert ev.n_items == 1000
    assert ev.n_query_tokens == 1000 * cfg_small.num_pairs
    assert ev.floor == pytest.approx(0.25)
    # An untrained model: NLL near ln(256) = 5.545, accuracy far below the 1/D floor.
    assert 5.0 < ev.nll_query < 6.5, ev.nll_query
    assert ev.accuracy < 0.10
    assert ev.acc_se_clustered > 0
    assert ev.nll_se_clustered > 0
    assert ev.acc_design_effect > 0
    assert len(ev.per_item_accuracy) == 1000


def test_design_effect_is_near_one_for_an_untrained_model(cfg_small):
    """
    MEASURED, and worth stating because it bounds the claim: on an UNTRAINED model the query NLLs
    within a sequence are nearly independent (the model ignores the key-value table entirely), so the
    design effect is ~1 and clustering costs nothing. The 3x inflation the literature warns about is
    a property of a model that has LEARNED something sequence-specific, not of the estimator.

    So this test asserts the honest thing -- the design effect is computed and lands near 1 here --
    rather than asserting an inflation that this fixture cannot produce.
    """
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=3)
    ev = H.evaluate(m, cfg_small, n_items=1000, batch_size=100)
    assert 0.5 < ev.nll_design_effect < 2.0, ev.nll_design_effect
    assert 0.5 < ev.acc_design_effect < 2.0, ev.acc_design_effect


def test_clustering_bites_when_outcomes_are_sequence_correlated(cfg_small):
    """
    The estimator must inflate the SE when clustering IS present -- otherwise it is decoration. This
    is the direct check on the estimator (a synthetic all-right/all-wrong pattern at D=4), paired
    with the test above showing the untrained model is not such a case.
    """
    k = 1000
    d = cfg_small.num_pairs
    perfectly_clustered = S.clustered_mean(
        [float(d) if i % 2 == 0 else 0.0 for i in range(k)], [d] * k
    )
    assert perfectly_clustered.design_effect == pytest.approx(d, rel=0.05)
    assert perfectly_clustered.se_clustered > perfectly_clustered.se_naive


# ======================================================================================
# run_cell end to end
# ======================================================================================


def test_run_cell_end_to_end_smoke(cfg_small):
    rec = H.run_cell(
        arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
        build_model=H.stub_build_model, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH,
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    assert rec.arm == "static"
    assert rec.topology == "allliv"
    assert rec.kernel_size == 3
    assert rec.config == "N64_D4"
    assert rec.num_pairs == 4
    assert rec.floor == pytest.approx(0.25)
    assert rec.data_seed == H.data_seed_for(0)
    assert rec.init_seed == H.derive_init_seed(0, "static")
    assert rec.extra["arm_code"] == "S1"
    assert rec.extra["smoke"] is True
    assert rec.n_eval_items == SMOKE_EVAL
    assert rec.n_params > 0
    assert 0.0 <= rec.accuracy <= 1.0
    assert rec.nll_query > 0


def test_init_loss_is_in_the_ln_vocab_band(cfg_small):
    """
    SPEC Sec 6 check 8: at vocab 256 the init loss must be in [5.5452, 5.7952]. Assert magnitudes,
    not existence -- this check has caught uninitialized weights ~4x in this repo.
    """
    ln_v = math.log(256)
    assert ln_v == pytest.approx(5.5452, abs=1e-3)
    rec = H.run_cell(
        arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=1,
        build_model=H.stub_build_model, steps=1, batch_size=SMOKE_BATCH,
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    got = rec.extra["init_loss"]
    assert ln_v - 0.05 <= got <= ln_v + 0.25, got
    assert rec.extra["init_loss_band"] == pytest.approx([ln_v - 0.05, ln_v + 0.25])


def test_init_loss_band_is_measured_over_enough_tokens_to_be_meaningful(cfg_small):
    """
    The band is +-0.25 nats wide and the per-token NLL has an SD of ~1 nat, so the check is only
    meaningful over enough tokens. MEASURED: the same healthy stub reads 5.32-5.95 over 16 query
    tokens (one batch of 4) but 5.65-5.75 over 4,096 -- i.e. a per-batch reading would fire this
    guard on a correctly initialized model. This test pins the token floor so that regression cannot
    silently return.
    """
    assert H.INIT_LOSS_MIN_TOKENS >= 2048
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=H.D_MODEL, n_layers=H.N_LAYERS, seed=0)
    # Every arm x seed combination must sit inside the band when measured properly.
    for arm in ("static", "permuted", "dynamic"):
        for p in range(3):
            mm = H.stub_build_model(
                arm=arm, topology="allliv", kernel_size=3, vocab_size=256,
                d_model=H.D_MODEL, n_layers=H.N_LAYERS, seed=H.derive_init_seed(p, arm),
            )
            got = H.check_init_loss(mm, cfg_small)  # raises if outside
            assert math.log(256) - 0.05 <= got <= math.log(256) + 0.25
    assert H.check_init_loss(m, cfg_small) > 0


def test_negative_control_a_broken_init_is_caught_by_the_loss_band(cfg_small):
    """
    Proof the loss-band guard can fail. A model whose head is scaled up by 100x has garbage logits,
    so its step-0 loss leaves the band -- and the harness must refuse rather than train happily.
    """

    def broken_builder(**kw):
        m = H.stub_build_model(**kw)
        with torch.no_grad():
            m.head.weight.mul_(100.0)
        return m

    with pytest.raises(AssertionError, match="INIT LOSS OUT OF BAND"):
        H.run_cell(
            arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
            build_model=broken_builder, steps=1, batch_size=SMOKE_BATCH,
            eval_items=SMOKE_EVAL, smoke=True, verbose=False,
        )


def test_paired_arms_on_the_stub_tie_because_the_stub_has_no_mechanism(cfg_small):
    """
    A negative check on the HARNESS: the stub is identical across arms, so a paired sweep must show
    zero difference. If a difference appeared it would be manufactured by the harness (a data-order
    leak or an arm-dependent init reaching the data stream).
    """
    recs = {}
    for arm in ("static", "dynamic"):
        recs[arm] = [
            H.run_cell(
                arm=arm, topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=p,
                build_model=lambda **kw: H.stub_build_model(**{**kw, "seed": 1234}),
                steps=SMOKE_STEPS, batch_size=SMOKE_BATCH, eval_items=SMOKE_EVAL,
                smoke=True, verbose=False,
            )
            for p in range(3)
        ]
    # Same data order AND a forced-identical init => bit-identical results.
    for a, b in zip(recs["static"], recs["dynamic"]):
        assert a.accuracy == pytest.approx(b.accuracy, abs=1e-12)
        assert a.nll_query == pytest.approx(b.nll_query, abs=1e-12)


# ======================================================================================
# Incremental persistence
# ======================================================================================


def test_records_are_written_incrementally_and_reload(tmp_path, cfg_small):
    out = tmp_path / "r.jsonl"
    rec = H.run_cell(
        arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
        build_model=H.stub_build_model, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH,
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    H.append_record(out, rec)
    H.append_record(out, rec)
    assert len(out.read_text().strip().splitlines()) == 2
    back = H.load_records(out)
    assert len(back) == 2
    assert back[0].accuracy == pytest.approx(rec.accuracy)
    assert back[0].arm == "static"
    assert H.completed_keys(out) == {H.cell_key("static", "allliv", 3, "N64_D4", 0)}


def test_a_truncated_final_line_does_not_lose_the_rest(tmp_path, cfg_small):
    """This machine has died mid-run. A truncated JSONL must lose at most its last line."""
    out = tmp_path / "r.jsonl"
    rec = H.run_cell(
        arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
        build_model=H.stub_build_model, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH,
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    H.append_record(out, rec)
    with out.open("a") as fh:
        fh.write('{"arm": "static", "topol')  # simulate a kill mid-write
    assert len(H.load_records(out)) == 1


def test_sweep_resumes_and_skips_completed_cells(tmp_path, cfg_small):
    out = tmp_path / "r.jsonl"
    kw = dict(
        arms=["static"], topologies=["allliv"], kernel_sizes=[3], configs=[cfg_small],
        out_path=out, build_model=H.stub_build_model, steps=SMOKE_STEPS,
        batch_size=SMOKE_BATCH, eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    H.run_sweep(seed_pairs=[0, 1], **kw)
    assert len(H.load_records(out)) == 2
    H.run_sweep(seed_pairs=[0, 1], **kw)  # rerun: must add nothing
    assert len(H.load_records(out)) == 2
    H.run_sweep(seed_pairs=[0, 1, 2], **kw)  # extend: must add exactly one
    assert len(H.load_records(out)) == 3


def test_sweep_skips_undefined_cells_without_substituting(tmp_path, cfg_small):
    out = tmp_path / "r.jsonl"
    recs = H.run_sweep(
        arms=["static", "dynqkv"], topologies=["allliv"], kernel_sizes=[3], configs=[cfg_small],
        seed_pairs=[0], out_path=out, build_model=H.stub_build_model, steps=SMOKE_STEPS,
        batch_size=SMOKE_BATCH, eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    assert {r.arm for r in recs} == {"static"}, "S3 must be absent, not substituted"


# ======================================================================================
# CALIBRATION -- the recorded evidence
# ======================================================================================


def test_recorded_calibration_numbers_reproduce_from_the_json():
    """
    The pinned :data:`calibration.RECORDED` table must re-derive from
    ``mqar_calibration.json`` (FarmShare 1670987). If the file moves, this reports "could not
    verify" rather than silently passing.
    """
    v = C.verify_recorded_numbers()
    assert v["ok"], v["mismatches"]
    assert v["checked"] == 8, f"expected 8 configs, checked {v['checked']}"
    assert v["source"] and Path(v["source"]).is_file()


def test_recorded_positive_control_reproduces_and_the_empty_gap_holds():
    """
    Re-measures the [0.30, 0.80] empty gap that justifies ``SOLVE_THRESHOLD = 0.80``. The threshold's
    justification is a measurement, so it has to be re-measurable.
    """
    v = C.verify_recorded_control()
    assert v["ok"], v["mismatches"]
    assert v["n_trials"] == 12
    assert v["empty_gap_confirmed"] is True
    assert not [a for a in v["accuracies_sorted"] if 0.30 < a < 0.80]


def test_the_ceiling_configs_are_dropped_with_measured_reasons():
    """
    SPEC Sec 4.5: drop the ceiling-saturated configs. ``N128_D8`` (10/10 at 1.0000) and ``N256_D16``
    (5/5 at 1.0000) have sd = 0.00 pp -- they cannot discriminate arms at ANY n.
    """
    assert "N128_D8" in C.DROPPED_CONFIGS
    assert "N256_D16" in C.DROPPED_CONFIGS
    for name in ("N128_D8", "N256_D16"):
        r = next(x for x in C.RECORDED if x.config == name)
        assert r.verdict == "DROP"
        assert r.success_rate == 1.0
        assert r.sigma_pp == 0.0
        assert set(r.per_seed) == {1.0}
        assert "CEILING" in C.DROPPED_CONFIGS[name]


def test_a_saturated_config_has_undefined_required_n_not_a_large_one():
    """
    Why a ceiling is fatal rather than merely expensive: s_delta = 0 makes required-n UNDEFINED. A
    function that returned a small n here would say a ceiling is perfectly powered.
    """
    res = S.required_n(0.0, 10.0)
    assert res["n"] is None
    assert "UNDEFINED" in res["method"]
    assert S.paired_power(10, 0.0, 10.0) != S.paired_power(10, 0.0, 10.0)  # NaN


def test_primary_is_N512_D64_and_secondary_is_N512_D8():
    """
    The README's recommendation, NOT the script's own auto-pick. ``mqar_calibrate.py`` picks
    ``N512_D8`` by proximity to 50 % success; the README recommends against that.
    """
    assert C.PRIMARY_CONFIG.label == "N512_D64"
    assert C.PRIMARY_CONFIG.num_pairs == 64
    assert S.degenerate_floor(C.PRIMARY_CONFIG.num_pairs) == pytest.approx(0.015625)
    assert C.SECONDARY_CONFIG.label == "N512_D8"
    assert C.SECONDARY_CONFIG.seq_len == C.PRIMARY_CONFIG.seq_len, (
        "the pair must share seq_len so it separates CAPACITY from DISTANCE"
    )
    prim = next(r for r in C.RECORDED if r.config == "N512_D64")
    sec = next(r for r in C.RECORDED if r.config == "N512_D8")
    assert prim.verdict == "PRIMARY"
    assert sec.verdict == "SECONDARY"
    # The primary is the graded one even though its success rate is LOWER.
    assert prim.success_rate < sec.success_rate
    assert sum(1 for a in prim.per_seed if 0.2 < a < 0.8) == 2
    assert sum(1 for a in sec.per_seed if 0.2 < a < 0.8) == 0


def test_the_R3_F8_target_band_is_reported_as_unachievable():
    """
    The honest answer. R3 F8 asks for baseline 30-70 % with sigma < 15 pp; every off-ceiling recorded
    config has sigma 19-47 pp, and the lowest belongs to a FLOOR-pinned config. So the band is
    reported as not achievable, with the ANOVA reason (94.1 % of variance is seed, not load).
    """
    assert C.TARGET_ACC_LO == 0.30 and C.TARGET_ACC_HI == 0.70
    assert C.TARGET_SIGMA_MAX_PP == 15.0
    assessments = C.assess_recorded()
    meeting = [a for a in assessments if a.in_target_band and a.sigma_ok]
    assert meeting == [], f"unexpectedly found a config in the band: {meeting}"
    off_ceiling = [a for a in assessments if not a.at_ceiling]
    assert off_ceiling, "some configs must be off ceiling, or the calibration failed entirely"
    assert min(a.sigma_pp for a in off_ceiling) > C.TARGET_SIGMA_MAX_PP
    assert "94.1%" in C.RECORDED_BAND_VERDICT
    assert "NOT ACHIEVABLE" in C.RECORDED_BAND_VERDICT


def test_assess_target_band_flags_ceiling_floor_and_graded():
    ceiling = C.assess_target_band([1.0] * 5, config="x", num_pairs=8)
    assert ceiling.at_ceiling and not ceiling.usable and "CEILING" in ceiling.verdict

    floor = C.assess_target_band([0.13, 0.14, 0.15, 0.15, 0.14], config="x", num_pairs=8)
    assert floor.at_floor and not floor.usable and "FLOOR" in floor.verdict

    graded = C.assess_target_band(
        [0.0515, 0.0853, 0.2043, 0.5584, 0.9825], config="N512_D64", num_pairs=64
    )
    assert graded.usable
    assert not graded.at_ceiling and not graded.at_floor
    assert graded.sigma_pp == pytest.approx(39.39, abs=0.01)
    assert not graded.sigma_ok

    # A hypothetical config that DOES meet the band must be recognised -- otherwise the check is
    # unfalsifiable and would "pass" by always saying no.
    good = C.assess_target_band([0.45, 0.48, 0.50, 0.52, 0.55], config="y", num_pairs=64)
    assert good.in_target_band and good.sigma_ok
    assert "MEETS R3 F8's BAND" in good.verdict


def test_exp2_grid_drops_ceiling_configs_but_re_enters_them_for_allliv():
    """
    ``include_easier=False`` gives only the recorded survivors. The default re-enters N128_D8 and
    N256_D16 because they were at ceiling *with 2 of 4 layers global*, and ``allliv`` has none --
    R5 F5(i) says the recorded cliff is not a receptive-field limit, so removing attention should
    move it.
    """
    survivors = [c.label for c in C.exp2_grid(include_easier=False)]
    assert survivors == ["N512_D64", "N512_D8"]
    full = [c.label for c in C.exp2_grid()]
    assert full == ["N128_D8", "N256_D16", "N512_D64", "N512_D8"]
    assert all(c.vocab_size == 256 for c in C.exp2_grid())


def test_calibration_is_baseline_only():
    """
    Rule 1: no arm flag, ever. Calibrating while looking at a treatment arm is choosing the test
    until it gives the answer you want.
    """
    assert C.BASELINE_ARM == "static"
    import inspect

    sig = inspect.signature(C.calibrate)
    assert "arm" not in sig.parameters and "arms" not in sig.parameters
    src = inspect.getsource(C.calibrate)
    for treatment in ("dynamic", "permuted", "dynqkv"):
        assert treatment not in src, f"calibrate() must not reference the {treatment!r} arm"


def test_positive_control_runs_the_easiest_rung_and_reports_pass_or_fail(cfg_small, tmp_path):
    """
    The control must actually run and must return a boolean verdict. On the untrained stub with 3
    steps it will FAIL, which is the correct output -- and :func:`calibration.main` aborts on it.
    """
    res = C.positive_control(
        build_model=H.stub_build_model, topology="allliv", steps=SMOKE_STEPS,
        batch_size=SMOKE_BATCH, lrs=(3e-3,), out_path=tmp_path / "c.jsonl",
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    assert isinstance(res["passed"], bool)
    assert res["passed"] is False, "3 steps cannot solve MQAR; a True here would be alarming"
    assert "DO NOT run a difficulty sweep" in res["verdict"]
    assert res["trials"][0].extra["role"] == "positive_control"
    assert (tmp_path / "c.jsonl").is_file()


def test_main_aborts_the_sweep_when_the_positive_control_fails(tmp_path, capsys):
    """
    Rule 2, enforced end to end: job 1670922's sweep returned 0.000 everywhere. A sweep whose
    easiest rung scores zero cannot separate "hard task" from "broken setup".
    """
    rc = C.main([
        "--stub", "--smoke", "--steps", "2", "--batch-size", "4",
        "--topologies", "allliv", "--seeds", "1",
        "--out", str(tmp_path / "c.jsonl"),
    ])
    assert rc == 3, "must abort with a distinct non-zero code, not proceed"
    out = capsys.readouterr()
    assert "ABORTING the difficulty sweep" in out.err


def test_evidence_only_mode_needs_no_compute(capsys):
    rc = C.main(["--evidence-only"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "RECORDED EVIDENCE" in out
    assert "N512_D64" in out
    assert "NOT ACHIEVABLE" in out
    # The measured numbers must appear, not a paraphrase.
    assert "0.9825" in out
    assert "8.3178" in out or "8.3178" in out


def test_ln_plateau_ladder_matches_the_measured_losses():
    """
    The loss ladder that makes a wrong algorithm legible (README's table, re-derived):
    ``ln(vocab/2) = 8.3178`` at vocab 8192 ("it's a value token"), ``ln(D) = 1.386`` at D=4
    ("it's one of these D values"), 0 = actually bound. The recorded control measured 8.25-8.34 and
    1.40-1.76 against those predictions.
    """
    assert math.log(4096) == pytest.approx(8.3178, abs=1e-3)
    assert math.log(4) == pytest.approx(1.3863, abs=1e-3)
    assert C.RECORDED_CONTROL["ln_4096"] == pytest.approx(8.3178, abs=1e-3)
    # And vocab 8192 really did fail: best of 6 was 0.214, with 2 exact zeros.
    assert C.RECORDED_CONTROL["vocab_8192_best_accuracy"] == pytest.approx(0.2139, abs=1e-3)
    assert C.RECORDED_CONTROL["vocab_8192_n_exact_zero"] == 2


def test_calibrate_end_to_end_smoke_writes_and_assesses(tmp_path, cfg_small):
    res = C.calibrate(
        build_model=H.stub_build_model, topologies=["allliv"], configs=[cfg_small],
        seeds=2, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH, out_path=tmp_path / "c.jsonl",
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    assert len(res["records"]) == 2
    assert len(res["assessments"]) == 1
    assert isinstance(res["recommendation"], str) and res["recommendation"]
    txt = C.report(res["assessments"], recommendation=res["recommendation"])
    assert "BASELINE (S1) ONLY" in txt


# ======================================================================================
# The sigma report
# ======================================================================================


def test_sigma_report_renders_and_does_not_contain_a_verdict(tmp_path, cfg_small):
    """
    Exp-2's deliverable is a measured sigma and a required n. The report must not print PASS/FAIL.
    """
    recs = []
    for arm in ("static", "dynamic"):
        for p in range(3):
            recs.append(
                H.run_cell(
                    arm=arm, topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=p,
                    build_model=H.stub_build_model, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH,
                    eval_items=SMOKE_EVAL, smoke=True, verbose=False,
                )
            )
    txt = S.sigma_report(recs)
    assert "MEASURED SIGMA AND REQUIRED N" in txt
    assert "POOLED WITHIN-CELL SIGMA" in txt
    assert "48.4" in txt, "the repo anchor must be printed for comparison"
    assert "PASS" not in txt and "FAIL" not in txt


def test_harness_main_runs_end_to_end(tmp_path, capsys):
    rc = H.main([
        "--stub", "--smoke", "--steps", "2", "--batch-size", "4", "--seeds", "2",
        "--seq-len", "64", "--num-pairs", "4", "--arms", "static", "dynamic",
        "--topologies", "allliv", "--out", str(tmp_path / "r.jsonl"),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SMOKE" in out
    assert "MEASURED SIGMA AND REQUIRED N" in out
    assert len(H.load_records(tmp_path / "r.jsonl")) == 4


# ======================================================================================
# REQUIREMENT (a) -- absolute loss / layout gate, from the Exp-0 missing-BOS scar
# ======================================================================================
#
# NOTE ON EXECUTION: these were authored 2026-08-05 under a hard no-local-execution constraint
# (the user's machine is failing). They are syntax-checked but **HAVE NOT BEEN RUN**. Run them on
# FarmShare before trusting them:  pytest test_harness.py -q


def test_this_generator_has_no_bos_and_that_is_asserted(cfg_small):
    """
    The Exp-0 scar is a MISSING BOS costing 2.4-3.8 nats -- ~100x the effect being chased, failing
    silently. The MQAR analogue must be asserted against the layout this generator actually emits.

    `mqar_data.py` is Zoology-faithful and emits **NO** BOS/separator: the layout is
    `k1 v1 ... kD vD <filler> q1 ... qD`. The OLDER in-tree design `probes/mqar_patch.py` DID use
    `MQAR_SEP = 0` at position `length - D - 1`. Asserting a BOS this generator never emits would
    fail every batch, so the gate asserts the invariants that DO hold.
    """
    g = torch.Generator().manual_seed(0)
    tokens, labels = H.make_mqar_batch(cfg_small, 8, g)
    info = H.assert_sequence_layout(cfg_small, tokens, labels)
    assert info["has_bos"] is False
    assert info["query_start"] == cfg_small.seq_len - cfg_small.num_pairs
    assert info["n_labelled_per_row"] == cfg_small.num_pairs


def test_negative_control_layout_gate_catches_shifted_labels(cfg_small):
    """
    Proof the layout gate can fail: roll the labels by one position -- the exact off-by-one that
    `olmo-core-reads-raw-not-npy` records as training happily while being silently wrong.
    """
    g = torch.Generator().manual_seed(0)
    tokens, labels = H.make_mqar_batch(cfg_small, 8, g)
    with pytest.raises(AssertionError, match="LAYOUT"):
        H.assert_sequence_layout(cfg_small, tokens, torch.roll(labels, shifts=1, dims=1))


def test_negative_control_layout_gate_catches_dropped_and_extra_labels(cfg_small):
    g = torch.Generator().manual_seed(0)
    tokens, labels = H.make_mqar_batch(cfg_small, 8, g)

    dropped = labels.clone()
    dropped[0, -1] = H.IGNORE_INDEX
    with pytest.raises(AssertionError, match="exactly D"):
        H.assert_sequence_layout(cfg_small, tokens, dropped)

    extra = labels.clone()
    extra[0, 0] = 200  # a label outside the query span
    with pytest.raises(AssertionError, match="LAYOUT"):
        H.assert_sequence_layout(cfg_small, tokens, extra)


def test_negative_control_layout_gate_catches_a_key_half_label(cfg_small):
    """A label in the KEY half means the key/value split broke -- a large absolute-loss change."""
    g = torch.Generator().manual_seed(0)
    tokens, labels = H.make_mqar_batch(cfg_small, 8, g)
    bad = labels.clone()
    bad[0, -1] = 3  # key half
    with pytest.raises(AssertionError, match="VALUE half"):
        H.assert_sequence_layout(cfg_small, tokens, bad)


def test_init_loss_gate_is_hard_and_blocks_the_run_not_just_warns(cfg_small):
    """
    Requirement (a): the absolute number gates the delta. `check_init_loss` must RAISE, and
    `run_cell` must therefore never return a record whose absolute loss is out of band.
    """
    import inspect

    src = inspect.getsource(H.check_init_loss)
    assert "raise AssertionError" in src, "the band check must raise, not warn"
    assert "warn" not in src.lower() or "raise AssertionError" in src
    # And run_cell calls it BEFORE constructing the optimizer, i.e. before any training.
    rc = inspect.getsource(H.run_cell)
    assert rc.index("check_init_loss") < rc.index("torch.optim.AdamW")
    # Any record that exists proves the gate passed.
    assert "init_loss_in_band" in rc


# ======================================================================================
# REQUIREMENT (b) -- identical kernel path across arms
# ======================================================================================


def test_kernel_path_is_resolved_and_logged():
    """
    `short_conv.py:185` defaults `use_fla=True` while `fla` is absent in many environments, so
    `has_fla()` is False and you silently get plain `nn.Conv1d`. The realised path must be LOGGED,
    per arm, not assumed.
    """
    m = H.stub_build_model(arm="static", topology="allliv", kernel_size=3, vocab_size=256,
                           d_model=32, n_layers=2, seed=0)
    path = H.resolve_kernel_path(m)
    assert path["family"], "a realised backend family must be reported"
    assert sum(path["backends"].values()) >= 1


def test_same_kernel_family_across_arms_passes_and_a_mismatch_raises():
    """
    A fused treatment against an unfused baseline is a confounded contrast that biases toward the
    hypothesis. The gate must fire on a mismatch.
    """
    paths = {}
    for arm in ("static", "dynamic"):
        m = H.stub_build_model(arm=arm, topology="allliv", kernel_size=3, vocab_size=256,
                               d_model=32, n_layers=2, seed=0)
        paths[arm] = H.resolve_kernel_path(m)
    H.assert_same_kernel_family(paths)  # identical stubs: must not raise

    paths["dynamic"] = {"family": "fla.fused x4", "backends": {"fla.fused": 4}, "use_fla_flags": []}
    with pytest.raises(AssertionError, match="KERNEL PATH MISMATCH"):
        H.assert_same_kernel_family(paths)


def test_record_carries_device_dtype_and_kernel_path(cfg_small):
    """Every reported number must be labelled with device and dtype."""
    rec = H.run_cell(
        arm="static", topology="allliv", kernel_size=3, cfg=cfg_small, seed_pair=0,
        build_model=H.stub_build_model, steps=SMOKE_STEPS, batch_size=SMOKE_BATCH,
        eval_items=SMOKE_EVAL, smoke=True, verbose=False,
    )
    for key in ("device", "dtype", "torch_version", "kernel_path", "layout",
                "init_loss", "init_loss_band", "init_loss_in_band"):
        assert key in rec.extra, key
    assert rec.extra["dtype"] == "torch.float32"


# ======================================================================================
# Device resolution + cost planning
# ======================================================================================


def test_resolve_device_refuses_a_silent_cpu_fallback():
    """Falling back to CPU on a GPU allocation wastes it and mis-attributes every timing."""
    assert H.resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cuda requested"):
            H.resolve_device("cuda")
        assert H.resolve_device("auto").type == "cpu"


def test_plan_grid_does_not_overcount_the_undefined_S3_cells():
    """
    S3 is N/A in allliv, so a naive arms x topologies product OVERCOUNTS. The full grid is 560
    cells, not 640; the 80 difference is exactly S3-in-allliv.
    """
    p = H.plan_grid()
    assert p["n_cells"] == 560
    assert p["n_undefined_skipped"] == 80
    assert p["per_topology"] == {"allliv": 240, "hybrid": 320}
    assert 4 * 2 * 4 * 2 * 10 == 640
    assert p["n_cells"] + p["n_undefined_skipped"] == 640
    # And no planned cell is an undefined one.
    assert not [c for c in p["cells"] if (c[0], c[1]) in H.UNDEFINED_CELLS]


def test_seconds_per_cell_is_OUTSTANDING_and_cost_refuses_to_guess():
    """
    The measurement must come from FarmShare. `cost_estimate` has no default and refuses a missing
    or non-positive number, so a plausible-looking constant cannot silently become "the
    measurement".
    """
    assert H.SECONDS_PER_CELL_MEASURED is None, (
        "If this is no longer None, a real FarmShare measurement must accompany it."
    )
    with pytest.raises(ValueError, match="MEASURED"):
        H.cost_estimate(0.0, n_cells=560)
    with pytest.raises(ValueError, match="MEASURED"):
        H.cost_estimate(None, n_cells=560)  # type: ignore[arg-type]
    # The recorded 4-layer CUDA numbers are a LOWER BOUND, and must be labelled as such.
    assert H.RECORDED_SECONDS_PER_CELL_4LAYER_CUDA["N512_D64"] == 592.1
    got = H.cost_estimate(592.1, n_cells=560, n_parallel=1)
    assert got["sequential_gpu_hours"] == pytest.approx(560 * 592.1 / 3600, rel=1e-9)
    assert got["fits_095h_autoapprove"] is False
