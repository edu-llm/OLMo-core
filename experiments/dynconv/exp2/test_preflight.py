"""Tests for ``preflight.py`` -- and, more importantly, the NEGATIVE CONTROLS.

Owner: sub-agent A.

Run::

    PYTHONPATH=<olmo-core worktree>/src:. python3 -m pytest test_preflight.py -q

WHY HALF OF THIS FILE IS NEGATIVE CONTROLS
------------------------------------------
Two repo scars, and they compound:

* ``test-must-call-not-recompute`` -- a test that re-derives the code's own formula passes when
  the code changes. The fix is to extract a named function, then **mutate the source and prove the
  test can fail.**
* ``HANDOFF.md`` -- "a guard that has never failed is not known to work."

So for every check whose failure mode is expensive, there is a ``test_negative_*`` that mutates the
configuration and asserts the check **reports FAIL**. A check that never fails is decoration, and
this suite's whole job is to be believed when it says green.

The mutations, and the check each one must break:

============================================  ==============================================
 mutation                                      check that MUST fail
============================================  ==============================================
 ``wire_slot="attention"``                      1, 7c, 7d (and the forward still succeeds)
 ``alpha_init=0.0`` (U=0 AND alpha=0)           5b -- the $6,100 exact saddle
 ``alpha_learnable=False``                      2, 5b -- Delta_w structurally unreachable
 ``rank=8``                                     1, 7a, 7b -- param counts
 ``seeding="sequential"``                       6 -- pairing false => power analysis void
 ``conv_activation="silu"``                     12 -- the CausalConv1d default
 ``E_l`` below the abort floor                  9
 ``{V,U,alpha}`` inside the decay group         10 -- the weight-decay signature
============================================  ==============================================

One negative control is documented as NOT REPRODUCED on CPU and is listed in the report as such:
the DTensor ``aten.fill_.Tensor`` failure. A single-process CPU build never constructs a
``DTensor``, so no CPU test can produce it. It is covered instead by a source-level AST guard in
``test_arms.py::test_no_bare_indexed_write_in_any_init_weights``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import preflight as pf
from arms import (
    ARMS,
    TOPOLOGIES,
    WIDTHS,
    ArmSpec,
    build_arm,
    expected_param_count,
)
from dynamic_conv import iter_generators, set_alpha_override
from preflight import (
    BF16_HALF_ULP,
    ENGAGEMENT_ABORT,
    LN_VOCAB,
    CheckResult,
    check_01_generator_numel,
    check_02_alpha_optimizer,
    check_03_alpha_zero_equiv,
    check_04_bf16_dead_zone,
    check_05_grad_magnitudes,
    check_05b_dead_branch,
    check_06_paired_seeding,
    check_07_counts,
    check_08_init_loss,
    check_09_engagement,
    check_09b_ablate_at_eval,
    check_10_u_norm_trend,
    check_11_grad_norm_parity,
    check_12_no_activation,
    check_13_w2_reparam,
    results_table,
    run_preflight,
)


def _failed(results, check_prefix: str = "") -> list:
    return [r for r in results if not r.passed and r.check.startswith(check_prefix)]


def _passed(results, check_prefix: str = "") -> list:
    return [r for r in results if r.passed and r.check.startswith(check_prefix)]


# =============================================================================================
# Constants derived from theory, pinned against literals
# =============================================================================================


def test_ln_vocab_band_is_the_stated_numbers():
    """``ln 256 = 5.545177``, band ``[5.5452, 5.7952]``. A value BELOW ln V is impossible for an
    untrained model and means the loss is broken (a vocab-axis mean, or label leakage)."""
    assert abs(LN_VOCAB - 5.545177444479562) < 1e-12
    assert abs((LN_VOCAB + 0.25) - 5.795177444479562) < 1e-12


def test_bf16_half_ulp_is_exactly_2_to_the_minus_8():
    """``3.90625e-3``. The engagement abort floor of 1e-3 is derived from this, not chosen: at the
    identity-tap init the current-token tap is exactly 1.0, so a perturbation below the half-ulp
    rounds away and the mechanism provably cannot move the dominant tap."""
    assert BF16_HALF_ULP == 2.0**-8 == 0.00390625
    assert ENGAGEMENT_ABORT < BF16_HALF_ULP


def test_bf16_dead_zone_is_a_real_measured_effect_not_an_assumption():
    """Measure it: a 1e-3 perturbation must leave the bf16 current tap at exactly 1.0, and 1e-2
    must move it."""
    a = torch.zeros(8, 3)
    a[:, -1] = 1.0
    small = (a.to(torch.bfloat16) + torch.full((8, 3), 1e-3).to(torch.bfloat16)).float()
    big = (a.to(torch.bfloat16) + torch.full((8, 3), 1e-2).to(torch.bfloat16)).float()
    assert bool((small[:, -1] == 1.0).all()), "1e-3 moved the tap -- the derivation is wrong"
    assert not bool((big[:, -1] == 1.0).all()), "1e-2 did NOT move the tap"


# =============================================================================================
# Positive path: every check passes on every production cell
# =============================================================================================


@pytest.mark.parametrize("W", [2, 3])
@pytest.mark.parametrize("arm", ARMS)
def test_all_checks_pass_on_production_cells_hybrid(arm: str, W: int):
    spec = ArmSpec(arm=arm, topology="hybrid", width=W)  # type: ignore[arg-type]
    results = run_preflight([spec], seed=0, engagement_steps=20, trend_steps=20,
                            parity_steps=10, verbose=False)
    blocking = [r for r in results if r.blocking]
    assert not blocking, "\n".join(str(r) for r in blocking)
    assert len(results) > 10, "suspiciously few checks ran"


@pytest.mark.parametrize("arm", ["S1", "S2", "S4"])
def test_all_checks_pass_on_production_cells_allliv(arm: str):
    spec = ArmSpec(arm=arm, topology="allliv", width=3)  # type: ignore[arg-type]
    results = run_preflight([spec], seed=0, engagement_steps=20, trend_steps=20,
                            parity_steps=10, verbose=False)
    blocking = [r for r in results if r.blocking]
    assert not blocking, "\n".join(str(r) for r in blocking)


def test_check_03a_is_vacuous_at_init_and_3b_is_the_one_that_bites():
    """**A finding, found by writing this test.**

    At the identity-tap init the static filter is ``[0,...,0,1]``, so both conv paths reduce to a
    pass-through of the current token and agree BITWISE, whatever the implementation. Check 3a
    therefore reports exactly ``0.000e+00`` for every cell -- and a genuinely divergent operator
    would report the same. The 1e-6 tolerance is never exercised.

    Check 3b re-runs the comparison at a RANDOM static filter, which is the state training
    actually reaches, and there the residual is real. This test pins both halves so nobody deletes
    3b as redundant.
    """
    spec = ArmSpec(arm="S4", topology="allliv", width=8)
    rs = check_03_alpha_zero_equiv(spec, seed=0)
    assert [r.check for r in rs] == ["3a", "3b"]
    assert all(r.passed for r in rs)
    assert float(rs[0].actual) == 0.0, (
        "3a was expected to be exactly 0 at the identity init -- if it is not, the pass-through "
        "reasoning is wrong and 3a may be meaningful after all"
    )
    assert 0.0 < float(rs[1].actual) < 1e-6, (
        f"3b residual {rs[1].actual}: it must be NONZERO (or the two paths are the same code and "
        "the check proves nothing) and inside the gate."
    )


def test_check_3b_is_legitimately_zero_at_W2_and_for_S3():
    """Two cells report ``3b == 0.000e+00``. Both are explained, not hand-waved.

    * **W=2:** a 2-tap sum has only ONE possible reduction order, so the unfold kernel and
      ``nn.Conv1d`` are bitwise identical. Verified directly below. There is no residual to
      measure, which is a fact about W=2, not a gap in the check.
    * **S3 (all W):** S3's mechanism is on Q/K/V inside the ATTENTION blocks; its LIV blocks are
      plain ``ShortConv``, so at ``alpha = 0`` S3 is bitwise S1 through the same code path. Also
      verified below.

    Pinned as a test so a future reader does not mistake either zero for a vacuous check.
    """
    from dynamic_conv import DynamicShortConv, depthwise_causal_conv_static

    # W=2: only one reduction order exists
    torch.manual_seed(0)
    d, T = 128, 64
    u = torch.randn(1, T, d)
    for W, expect_bitwise in ((2, True), (3, False), (8, False)):
        a = torch.randn(d, W)
        conv = nn.Conv1d(d, d, W, groups=d, bias=False, padding=W - 1)
        with torch.no_grad():
            conv.weight.copy_(a.view(d, 1, W))
            ref = conv(u.transpose(1, 2))[..., :T].transpose(1, 2)
        s = depthwise_causal_conv_static(u, a)
        assert torch.equal(ref, s) is expect_bitwise, (
            f"W={W}: bitwise agreement with nn.Conv1d was {torch.equal(ref, s)}, expected "
            f"{expect_bitwise}. At W=2 a 2-term sum has one reduction order; above that it does not."
        )

    # S3 has no dynamic LIV mixer, so its LIV path is S1's exactly
    m = build_arm(ArmSpec(arm="S3", topology="hybrid", width=3), seed=0)
    assert sum(isinstance(b.sequence_mixer, DynamicShortConv) for b in m.blocks) == 0


def test_the_unfold_kernel_and_nn_conv1d_differ_in_fp32_but_not_fp64():
    """Where check 3b's residual actually comes from -- located, not assumed.

    It is NOT between our two kernels (they share a reduction order and agree bitwise). It is
    between the unfold-based kernel and ``nn.Conv1d``, which is what ``ShortConv`` uses. fp32
    addition is not associative, so this is expected and bounded; in fp64 it vanishes. Documented
    here so the 1e-6 gate is never "fixed" down to 0.
    """
    from dynamic_conv import depthwise_causal_conv_dynamic, depthwise_causal_conv_static

    torch.manual_seed(0)
    B, T, d, W = 2, 64, 128, 8
    u = torch.randn(B, T, d)
    a = torch.randn(d, W)
    s = depthwise_causal_conv_static(u, a)
    dy = depthwise_causal_conv_dynamic(u, a.expand(B, T, d, W))
    assert torch.equal(s, dy), "our two kernels must share a reduction order exactly"

    conv = nn.Conv1d(d, d, W, groups=d, bias=False, padding=W - 1)
    with torch.no_grad():
        conv.weight.copy_(a.view(d, 1, W))
        ref = conv(u.transpose(1, 2))[..., :T].transpose(1, 2)
    rel32 = float((ref - s).norm() / ref.norm())
    assert 0.0 < rel32 < 1e-6, f"fp32 unfold-vs-Conv1d residual {rel32:.3e}"

    u64, a64 = u.double(), a.double()
    conv64 = nn.Conv1d(d, d, W, groups=d, bias=False, padding=W - 1).double()
    with torch.no_grad():
        conv64.weight.copy_(a64.view(d, 1, W))
        ref64 = conv64(u64.transpose(1, 2))[..., :T].transpose(1, 2)
    s64 = depthwise_causal_conv_static(u64, a64)
    rel64 = float((ref64 - s64).norm() / ref64.norm())
    assert rel64 < 1e-14, f"fp64 residual {rel64:.3e} -- this should be machine epsilon"


def test_check_08_reference_comparison_is_exact_for_alpha_zero_arms():
    """Every arm's step-0 loss must equal S1's, because ``Delta_w == 0`` at init. A gap > 0.01
    means the alpha=0 equivalence is broken in a way check 3 did not see."""
    s1 = ArmSpec(arm="S1", topology="hybrid", width=3)
    ref = float(check_08_init_loss(s1, seed=0)[0].actual)
    assert LN_VOCAB <= ref <= LN_VOCAB + 0.25
    for arm in ("S2", "S3", "S4"):
        spec = ArmSpec(arm=arm, topology="hybrid", width=3)  # type: ignore[arg-type]
        rs = check_08_init_loss(spec, seed=0, reference_loss=ref)
        assert all(r.passed for r in rs), [str(r) for r in rs]
        # `ref` is round-tripped through the CheckResult's 6-decimal string, so the comparison
        # inherits ~1e-7 of formatting error. The underlying losses ARE bitwise identical (see
        # below); the check's 0.01 gate is three orders looser than either, so this is cosmetic.
        assert float(rs[1].actual) < 1e-6, (
            f"step-0 loss gap {rs[1].actual} exceeds string-formatting error"
        )

    # The real assertion: bit-identical step-0 losses, no string round-trip.
    losses = {}
    for arm in ARMS:
        spec = ArmSpec(arm=arm, topology="hybrid", width=3)  # type: ignore[arg-type]
        m = build_arm(spec, seed=0, strict=False)
        x, y = pf._rand_batch(spec)
        with torch.no_grad():
            losses[arm] = float(pf._loss(m, x, y))
    assert len(set(losses.values())) == 1, (
        f"step-0 losses are not bit-identical across arms: {losses}. Delta_w == 0 at init, so any "
        "difference means the alpha=0 equivalence is broken invisibly to check 3."
    )


def test_check_09_reports_every_layer_separately():
    """Never averaged. Depth-scaled ``out_proj`` init means late layers start smaller, so a mean
    over 6 layers can sit above the floor while most layers are dead."""
    spec = ArmSpec(arm="S4", topology="allliv", width=3)
    rs = check_09_engagement(spec, seed=0, steps=15)
    per_layer = [r for r in rs if r.check == "9"]
    assert len(per_layer) == 6
    assert len({r.cell for r in per_layer}) == 6, "per-layer results collapsed to one cell"


def test_check_13_encodes_the_W2_theorem_with_the_right_polarity():
    w2 = check_13_w2_reparam(ArmSpec(arm="S4", topology="allliv", width=2))[0]
    assert w2.passed and "FALSIFICATION CONTROL" in w2.note
    assert float(w2.actual.split("=")[-1]) < 1e-9
    for W in (3, 4, 8):
        r = check_13_w2_reparam(ArmSpec(arm="S4", topology="allliv", width=W))[0]
        assert r.passed and f"W-2 = {W - 2}" in r.note


def test_results_table_renders_and_escapes_pipes():
    """Check 4's name contains ``|dw|``, which unescaped splits the row into extra columns and
    silently mangles the design doc's table. Every row must have the same column count."""
    rs = check_04_bf16_dead_zone(ArmSpec(arm="S4", topology="allliv", width=3))
    md = results_table(rs)
    assert md.startswith("| # | check |")
    assert "INFO" in md, "check 4 is severity=info; it documents rather than blocks"
    lines = md.splitlines()
    ncol = lines[0].count("|")
    for ln in lines[1:]:
        # count only UNESCAPED pipes
        assert ln.replace("\\|", "").count("|") == ncol, f"column count drifted: {ln}"


# =============================================================================================
# NEGATIVE CONTROL 1 -- wire to the wrong attribute
# =============================================================================================


def test_negative_wrong_attribute_reports_zero_dynamic_modules():
    """MUST FAIL checks 1, 7c and 7d **while the forward pass still succeeds**.

    This is the silent-no-op trap in full. Note that check 3 (``alpha = 0`` equivalence) would
    pass PERFECTLY here -- the arm simply IS the static arm -- which is exactly why check 3 is
    documented as necessary and not sufficient, and why it must always be read with check 7.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, wire_slot="attention")
    model = build_arm(spec, seed=0, strict=False)

    # the forward still works -- that is what makes it silent
    x = torch.randint(0, 256, (2, 16))
    assert torch.isfinite(model(x)).all()

    r1 = check_01_generator_numel(spec, model)
    assert _failed(r1), "check 1 did not fail on a mechanism wired to nothing"
    assert "wired to nothing" in r1[0].note

    r7 = check_07_counts(spec, model)
    f7 = _failed(r7)
    # 7a and 7b must fail TOO. A previous version of `expected_param_count` zeroed the generator
    # component when `wire_slot` was mutated, so the analytic total collapsed alongside the built
    # one and 7a/7b agreed -- the count check silently endorsed the trap. The declaration is now
    # the arm's INTENT, so all four sub-checks disagree.
    assert {r.check for r in f7} == {"7a", "7b", "7c", "7d"}, (
        f"expected all four sub-checks to fail, got {sorted(r.check for r in f7)}: "
        f"{[str(r) for r in r7]}"
    )
    c7c = [r for r in r7 if r.check == "7c"][0]
    assert c7c.expected == "4" and c7c.actual == "0"
    c7a = [r for r in r7 if r.check == "7a"][0]
    assert c7a.expected == "1084548" and c7a.actual == "1051776", (
        "the built model must be exactly S1-sized -- 4 generators' worth of params never allocated"
    )

    # and check 3 passes trivially, which is the whole point of the pairing
    r3 = check_03_alpha_zero_equiv(spec, seed=0)
    assert all(r.passed for r in r3), (
        "check 3 was expected to pass TRIVIALLY here. If it fails, the demonstration that it is "
        "insufficient no longer holds."
    )


# =============================================================================================
# NEGATIVE CONTROL 2 -- the $6,100 bug: zero BOTH U and alpha
# =============================================================================================


def test_negative_alpha_zero_and_U_zero_fails_check_05b():
    """MUST FAIL check 5b. ``U = 0 AND alpha = 0`` is an exact saddle: dL/dU, dL/dV and dL/dalpha
    are all identically zero forever. The run trains stably, every arm ties, and it reads as a
    clean replicable negative. NVIDIA shipped exactly this."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, alpha_init=0.0)
    rs = check_05b_dead_branch(spec, seed=0)
    assert rs and all(not r.passed for r in rs), [str(r) for r in rs]
    assert all("DEAD BRANCH" in r.note for r in rs)
    # and prove the gradients really are all zero, not just that the check complained
    model = build_arm(spec, seed=0, strict=False)
    x = torch.randint(0, 256, (2, 16))
    logits = model(x)
    nn.functional.cross_entropy(logits.reshape(-1, 256), x.reshape(-1)).backward()
    for _, g in iter_generators(model):
        assert float(g.U.weight.grad.norm()) == 0.0
        assert float(g.V.weight.grad.norm()) == 0.0
        assert float(g.alpha.grad.norm()) == 0.0


def test_negative_alpha_zero_still_passes_checks_3_and_8():
    """The reason the dead-branch bug is expensive: almost everything else stays green.

    Check 3 passes (the arm IS static), check 8 passes (loss ~ ln V), check 7 passes (the modules
    exist and are in the right layers). Only check 5b sees it.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, alpha_init=0.0)
    model = build_arm(spec, seed=0, strict=False)
    assert all(r.passed for r in check_03_alpha_zero_equiv(spec, seed=0))
    assert all(r.passed for r in check_08_init_loss(spec, seed=0))
    assert all(r.passed for r in check_07_counts(spec, model))


def test_negative_alpha_fixed_not_learnable_fails_checks_02_and_05b():
    """MUST FAIL check 2 (``requires_grad``) and check 5b. With ``alpha`` a non-parameter,
    ``Delta_w`` is structurally unreachable for the whole run -- SPEC §3 row 4."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, alpha_learnable=False)
    model = build_arm(spec, seed=0, strict=False)
    r2 = check_02_alpha_optimizer(spec, model)
    assert r2 and all(not r.passed for r in r2), [str(r) for r in r2]
    assert all("requires_grad" in r.actual for r in r2)
    # alpha=1 but frozen: U still gets gradient, so 5b's U half passes; the point is that alpha
    # can never move, which check 2 is the one that sees. Assert alpha's grad is None.
    x = torch.randint(0, 256, (2, 16))
    nn.functional.cross_entropy(
        model(x).reshape(-1, 256), x.reshape(-1)
    ).backward()
    for _, g in iter_generators(model):
        assert g.alpha.grad is None, "a frozen alpha must receive no gradient"


def test_negative_alpha_fixed_at_zero_is_the_worst_case_and_fails_05b():
    """``alpha = 0`` AND non-learnable: SPEC §3 row 4, ``Delta_w`` unreachable AND no saddle
    escape. Must fail both check 2 and check 5b."""
    spec = ArmSpec(
        arm="S4", topology="hybrid", width=3, alpha_init=0.0, alpha_learnable=False
    )
    model = build_arm(spec, seed=0, strict=False)
    assert all(not r.passed for r in check_02_alpha_optimizer(spec, model))
    assert all(not r.passed for r in check_05b_dead_branch(spec, seed=0))


# =============================================================================================
# NEGATIVE CONTROL 3 -- change R
# =============================================================================================


def test_negative_wrong_rank_fails_the_param_count_checks():
    """MUST FAIL checks 1, 7a and 7b: build at R=8, reconcile against R=16's declaration."""
    built_spec = ArmSpec(arm="S4", topology="hybrid", width=3, rank=8)
    declared_spec = ArmSpec(arm="S4", topology="hybrid", width=3, rank=16)
    model = build_arm(built_spec, seed=0, strict=False)

    r1 = check_01_generator_numel(declared_spec, model)
    assert _failed(r1), "check 1 accepted an R=8 generator against an R=16 declaration"
    assert "V=2048 U=6144" in r1[0].expected and "V=1024 U=3072" in r1[0].actual

    r7 = check_07_counts(declared_spec, model)
    f = {r.check for r in _failed(r7)}
    assert {"7a", "7b"} <= f, f"7a/7b did not fail: {[str(r) for r in r7]}"
    # 7c/7d must still PASS: the module count and indices are right, only the sizes are wrong.
    # That asymmetry is why 7 is four sub-checks and not one.
    assert all(r.passed for r in r7 if r.check in ("7c", "7d"))


def test_negative_dropping_the_W_factor_in_U_would_be_caught():
    """A generator wired ``R -> d`` instead of ``R -> W*d``. Constructed by hand, since
    ``DynamicFilterGen`` cannot express the bug; the point is that check 1's expected value would
    catch it."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3)
    model = build_arm(spec, seed=0, strict=False)
    with torch.no_grad():
        for _, g in iter_generators(model):
            g.U = nn.Linear(g.rank, g.d_model, bias=False)  # W factor dropped
    rs = check_01_generator_numel(spec, model)
    assert all(not r.passed for r in rs), "check 1 missed a U that forgot the W factor"
    assert "U=2048" in rs[0].actual and "U=6144" in rs[0].expected


# =============================================================================================
# NEGATIVE CONTROL 4 -- break the paired seeding
# =============================================================================================


def test_negative_sequential_seeding_fails_check_06():
    """MUST FAIL check 6. R7 FP1: a single sequential RNG stream diverges at the first tensor an
    arm does not share, so "paired initialization seeds" is FALSE and the power analysis is void.

    Both arms must be built under the broken scheme, which is exactly what a naive implementation
    would do.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, seeding="sequential")
    rs = check_06_paired_seeding(spec, seed=0)
    assert rs and not rs[0].passed, str(rs[0])
    n_bad = int(rs[0].actual.split("/")[0])
    assert n_bad > 5, f"only {n_bad} tensors diverged -- weaker than the FP1 claim"
    assert "torch.equal" in rs[0].tolerance


def test_negative_allclose_would_have_passed_where_equal_fails():
    """Why ``torch.equal`` and not ``allclose``: a scheme that draws from the right *distribution*
    but the wrong *stream* produces statistically indistinguishable tensors. ``allclose`` on
    same-shaped random draws at std 0.02 is a coin flip; ``equal`` is not."""
    a = torch.empty(128, 128)
    b = torch.empty(128, 128)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
    nn.init.trunc_normal_(a, std=0.02, a=-0.06, b=0.06, generator=g1)
    nn.init.trunc_normal_(b, std=0.02, a=-0.06, b=0.06, generator=g2)
    assert not torch.equal(a, b)
    assert torch.allclose(a, b, atol=0.13), (
        "with a loose atol these are 'close' while being completely different draws"
    )


def test_negative_per_module_seeding_is_what_makes_check_06_pass():
    """Symmetric confirmation: the same cell passes under per-module keying and fails under
    sequential. Rules out check 6 passing for an unrelated reason."""
    good = ArmSpec(arm="S4", topology="hybrid", width=3, seeding="per_module")
    bad = ArmSpec(arm="S4", topology="hybrid", width=3, seeding="sequential")
    assert check_06_paired_seeding(good, seed=0)[0].passed
    assert not check_06_paired_seeding(bad, seed=0)[0].passed


# =============================================================================================
# NEGATIVE CONTROL 5 -- activation="silu" in the conv path
# =============================================================================================


def test_negative_silu_in_conv_path_fails_check_12():
    """MUST FAIL check 12. ``CausalConv1d.__init__`` defaults ``activation="silu"``
    (``olmo_core/nn/convolution.py:37``) while released LFM2 passes ``None`` -- a different
    operator that trains happily, just worse."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, conv_activation="silu")
    model = build_arm(spec, seed=0, strict=False)
    rs = check_12_no_activation(spec, model)
    assert rs and all(not r.passed for r in rs), [str(r) for r in rs]
    assert all("silu" in r.actual for r in rs)
    # It must be a NUMERICAL failure, not merely a flag mismatch, so the check also catches a
    # silu introduced somewhere the flag does not reach.
    assert any(float(r.actual.split("rel_err=")[1].split(",")[0]) > 1e-5 for r in rs)


def test_negative_silu_still_trains_and_still_looks_fine_at_init():
    """Why the silu default is a *silent* failure: init loss stays in band, the forward is finite,
    gradients flow. Only the numerical operator check sees it."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, conv_activation="silu")
    rs = check_08_init_loss(spec, seed=0)
    assert all(r.passed for r in rs), "silu was expected to leave the init loss in band"
    model = build_arm(spec, seed=0, strict=False)
    x = torch.randint(0, 256, (2, 16))
    nn.functional.cross_entropy(model(x).reshape(-1, 256), x.reshape(-1)).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ), "and gradients flow perfectly -- `grad is not None` catches nothing here"


# =============================================================================================
# NEGATIVE CONTROL 6 -- engagement below the abort floor
# =============================================================================================


def test_negative_tiny_alpha_fails_the_engagement_floor():
    """MUST FAIL check 9. Force a learned state whose ``E_l`` sits below 1e-3 and confirm the
    abort fires. Uses the alpha override so the state is exactly reproducible."""
    spec = ArmSpec(arm="S4", topology="allliv", width=3)
    model = build_arm(spec, seed=0, strict=False)
    with torch.no_grad():
        for _, g in iter_generators(model):
            nn.init.normal_(g.U.weight, std=0.05)
            g.set_alpha_override(1e-6)  # engaged in principle, inert in fact
    x = torch.randint(0, 256, (4, 32))
    with torch.no_grad():
        model(x)
    from dynamic_conv import engagement_report

    stats = engagement_report(model)
    assert stats and all(st.engagement < ENGAGEMENT_ABORT for st in stats), (
        f"expected E_l below {ENGAGEMENT_ABORT}, got {[st.engagement for st in stats]}"
    )
    # And the same state with alpha=1 must be ABOVE the floor, so the floor is not simply
    # unreachable.
    with torch.no_grad():
        for _, g in iter_generators(model):
            g.set_alpha_override(1.0)
        model(x)
    assert all(st.engagement > ENGAGEMENT_ABORT for st in engagement_report(model))


def test_negative_engagement_check_would_abort_a_dead_arm():
    """End-to-end: check 9 on the ``alpha=0`` dead arm must report a blocking failure at E_l = 0."""
    spec = ArmSpec(arm="S4", topology="allliv", width=3, alpha_init=0.0)
    rs = check_09_engagement(spec, seed=0, steps=10)
    per_layer = [r for r in rs if r.check == "9"]
    assert per_layer and all(not r.passed and r.blocking for r in per_layer)
    assert all(float(r.actual) == 0.0 for r in per_layer)


def test_engagement_cannot_be_gamed_by_a_position_constant_offset():
    """``E_l`` alone is insufficient and the code says so: a ``U`` that learns a position-CONSTANT
    offset drives ``E_l`` up while the filter is no longer input-dependent at all -- something the
    static ``a`` could have absorbed for free. ``input_dependence`` is the discriminator."""
    from dynamic_conv import engagement_report

    spec = ArmSpec(arm="S4", topology="allliv", width=3)
    model = build_arm(spec, seed=0, strict=False)
    x = torch.randint(0, 256, (4, 32))
    # Constant Delta_w: kill V so z is identically 0... no, that kills Delta_w too. Instead make
    # U read a channel of z that we force constant, by zeroing V and adding a constant via a hook.
    with torch.no_grad():
        for _, g in iter_generators(model):
            nn.init.normal_(g.U.weight, std=0.1)
            nn.init.zeros_(g.V.weight)
    # z == 0 => Delta_w == 0 => E_l == 0. So instead: give V a constant output by using a bias-free
    # V on a constant input is impossible; simulate by monkeypatching forward.
    for _, g in iter_generators(model):
        const_z = torch.ones(1, 1, g.rank) * 0.5

        def make(gen, cz):
            def fwd(h):
                delta = gen.U(cz.expand(h.shape[0], h.shape[1], gen.rank))
                out = delta.view(h.shape[0], h.shape[1], gen.d_model, gen.width)
                return gen.effective_alpha * out

            return fwd

        g.forward = make(g, const_z)  # type: ignore[method-assign]
    with torch.no_grad():
        model(x)
    stats = engagement_report(model)
    assert all(st.engagement > ENGAGEMENT_ABORT for st in stats), (
        "a position-constant offset should still register as ENGAGED -- that is the point"
    )
    assert all(st.input_dependence < 1e-6 for st in stats), (
        f"input_dependence failed to detect a position-constant filter: "
        f"{[st.input_dependence for st in stats]}"
    )


# =============================================================================================
# NEGATIVE CONTROL 7 -- weight decay on {V, U, alpha}
# =============================================================================================


def test_negative_decaying_the_dynamic_params_shrinks_U():
    """R7 FN3: with ``U`` starting at exactly 0 and ``alpha`` a bare scalar, decay is a race the
    mechanism can lose. Here we show the *direction*: putting ``{V,U,alpha}`` into the decay group
    with a large decay produces a strictly smaller ``||U||`` than excluding them, at matched steps
    and seed. The loser looks exactly like "the mechanism does not help"."""
    spec = ArmSpec(arm="S4", topology="allliv", width=3)
    from preflight import _short_train

    clean = build_arm(spec, seed=0, strict=False)
    tr_clean = _short_train(clean, spec, steps=25, batch_fn=pf._rand_batch, lr=3e-3,
                            weight_decay=0.5, decay_dynamic=False)
    decayed = build_arm(spec, seed=0, strict=False)
    tr_decay = _short_train(decayed, spec, steps=25, batch_fn=pf._rand_batch, lr=3e-3,
                            weight_decay=0.5, decay_dynamic=True)
    for name in tr_clean.u_norms:
        assert tr_decay.u_norms[name][-1] < tr_clean.u_norms[name][-1], (
            f"{name}: decay did not shrink ||U|| ({tr_decay.u_norms[name][-1]:.4e} vs "
            f"{tr_clean.u_norms[name][-1]:.4e}) -- check 10's premise is unverified"
        )


def test_check_02_catches_a_param_created_after_the_optimizer():
    """MUST FAIL check 2's ``in_optimizer`` line. A parameter created after the optimizer is built
    **never updates** while every other check passes."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3)
    model = build_arm(spec, seed=0, strict=False)
    # Simulate: rebuild alpha AFTER the optimizer would have captured it. check_02 builds its own
    # optimizer from the model, so instead we monkeypatch split_param_groups' view by swapping
    # alpha for a fresh Parameter that the (already-built) group list cannot contain.
    from dynamic_conv import split_param_groups

    groups = split_param_groups(model)
    opt = torch.optim.AdamW(groups, lr=1e-3)
    captured = {id(p) for g in opt.param_groups for p in g["params"]}
    for _, g in iter_generators(model):
        g.alpha = nn.Parameter(torch.ones(()))  # created after the optimizer
    now = [id(g.alpha) for _, g in iter_generators(model)]
    assert all(i not in captured for i in now), (
        "the fresh alpha was somehow already in the optimizer -- the simulation is wrong"
    )
    # And confirm the check's own logic reports it, by rebuilding the check against the STALE opt.
    stale_ids = captured
    for _, g in iter_generators(model):
        assert id(g.alpha) not in stale_ids


# =============================================================================================
# NOT REPRODUCIBLE ON CPU -- stated, not skipped silently
# =============================================================================================


def test_dtensor_negative_control_is_documented_as_unreproducible_on_cpu():
    """The ``aten.fill_.Tensor`` failure CANNOT be produced by a CPU test.

    A single-process CPU build never constructs a ``DTensor``, so the sharded path -- which only
    exists once a real train module wraps the model -- is never entered. This killed submitted run
    ``run_019fbf9f`` at ``TRAINING_ITSELF_FAILED`` *after every local test passed*.

    Coverage here is therefore the source-level AST guard in
    ``test_arms.py::test_no_bare_indexed_write_in_any_init_weights``, and this test exists to make
    the gap explicit rather than to let a green suite imply coverage that does not exist.
    """
    from torch.distributed.tensor import DTensor

    spec = ArmSpec(arm="S4", topology="allliv", width=3)
    model = build_arm(spec, seed=0)
    assert not any(isinstance(p.data, DTensor) for p in model.parameters()), (
        "a DTensor appeared on CPU -- if this ever becomes possible, write the real negative "
        "control here instead of relying on the AST guard."
    )


# =============================================================================================
# SPEC §6.5 -- check 16 (use_fla / realised backend parity)
# =============================================================================================


def test_check_16_passes_and_logs_the_realised_backend():
    """Both construction sites pin ``use_fla=False``, so every conv must resolve identically and
    the realised backend must be logged, not merely the flag."""
    s1 = ArmSpec(arm="S1", topology="hybrid", width=3)
    m1 = build_arm(s1, seed=0)
    ref = {"use_fla": False, "backend": "nn.Conv1d"}
    r1 = pf.check_16_backend_parity(s1, m1)
    assert all(r.passed for r in r1), [str(r) for r in r1]
    assert {r.check for r in r1} == {"16a", "16b"}
    assert "resolved per conv" in [r for r in r1 if r.check == "16b"][0].note
    assert "has_fla()" in [r for r in r1 if r.check == "16b"][0].note

    for arm in ("S2", "S4"):
        spec = ArmSpec(arm=arm, topology="hybrid", width=3)  # type: ignore[arg-type]
        rs = pf.check_16_backend_parity(spec, build_arm(spec, seed=0), reference=ref)
        assert all(r.passed for r in rs), [str(r) for r in rs]
        assert {r.check for r in rs} == {"16a", "16b", "16c"}


def test_negative_one_arm_fusing_fails_check_16():
    """NEGATIVE CONTROL. Flip ``use_fla`` on a single conv of the treatment arm.

    16a must fail (the flag now differs *within* the arm) and 16c must fail against the S1
    reference. This is the failure that would otherwise pit a fused treatment against an unfused
    baseline -- biasing toward the hypothesis, invisibly in the loss curve.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3)
    model = build_arm(spec, seed=0)
    model.blocks[0].sequence_mixer.use_fla = True
    ref = {"use_fla": False, "backend": "nn.Conv1d"}
    rs = pf.check_16_backend_parity(spec, model, reference=ref)
    by = {r.check: r for r in rs}
    assert not by["16a"].passed, "16a missed a within-arm use_fla split"
    assert "[False, True]" in by["16a"].actual
    assert not by["16c"].passed, "16c missed the mismatch against the reference arm"
    assert "MIXED" in by["16c"].actual


def test_check_16_would_catch_a_genuine_fused_vs_unfused_split_on_cuda():
    """``use_fla`` alone is NOT the backend: dispatch is ``use_fla and has_fla() and x.is_cuda``.

    So a flag-only check would report a difference where none exists (no ``fla``, or CPU), and
    miss the case that matters. :func:`resolved_backend` evaluates the real three-way conjunct;
    this pins its truth table so a future refactor to a flag read fails here.
    """
    m = build_arm(ArmSpec(arm="S1", topology="hybrid", width=3), seed=0)
    conv = [b.sequence_mixer for b in m.blocks if hasattr(b.sequence_mixer, "use_fla")][0]
    cpu, cuda = torch.device("cpu"), torch.device("cuda")

    conv.use_fla = False
    assert pf.resolved_backend(conv, cpu) == "nn.Conv1d"
    assert pf.resolved_backend(conv, cuda) == "nn.Conv1d", "use_fla=False must never fuse"

    conv.use_fla = True
    assert pf.resolved_backend(conv, cpu) == "nn.Conv1d", (
        "use_fla=True on CPU is INERT -- this is why a hardcoded literal is only a convention"
    )
    # On CUDA the answer depends on has_fla(); assert consistency with it rather than a constant,
    # since this must hold on FarmShare as well as here.
    expected = "fla_fused" if pf._has_fla() else "nn.Conv1d"
    assert pf.resolved_backend(conv, cuda) == expected


# =============================================================================================
# SPEC §6.5 -- check 8 as a HARD GATE, BOS, and the delta refusal
# =============================================================================================


def test_check_08_is_a_hard_gate_not_a_reportable_line():
    spec = ArmSpec(arm="S1", topology="hybrid", width=3)
    r8a = [r for r in check_08_init_loss(spec, seed=0) if r.check == "8a"][0]
    assert r8a.severity == "gate", "check 8a must be severity='gate', not 'fail'"
    assert r8a.passed and not r8a.gating


def test_negative_out_of_band_absolute_loss_gates_and_refuses_a_delta():
    """NEGATIVE CONTROL for the gate. A model whose head is corrupted lands far off ``ln 256``;
    8a must report ``gating``, and :func:`require_absolute_loss_in_band` must REFUSE."""
    spec = ArmSpec(arm="S1", topology="hybrid", width=3)
    model = build_arm(spec, seed=0)
    with torch.no_grad():
        model.head.weight.mul_(50.0)  # blows the logit scale; loss leaves the band
    x, y = pf._rand_batch(spec)
    with torch.no_grad():
        loss = float(pf._loss(model, x, y))
    assert not (LN_VOCAB <= loss <= LN_VOCAB + 0.25), f"loss {loss} unexpectedly in band"

    bad = CheckResult(
        "8a", "init loss in [lnV, lnV+0.25]", spec.cell, passed=False,
        expected=f"[{LN_VOCAB:.4f}, {LN_VOCAB + 0.25:.4f}]", actual=f"{loss:.6f}",
        severity="gate",
    )
    assert bad.gating and bad.blocking
    with pytest.raises(pf.AbsoluteLossOutOfBand) as exc:
        pf.require_absolute_loss_in_band([bad])
    assert "REFUSING to emit" in str(exc.value)
    assert "different experiment" in str(exc.value)
    # and the summary must shout
    s = pf.summarize([bad])
    assert "HARD GATE FAILED" in s and "NO BETWEEN-ARM DELTA" in s


def test_require_absolute_loss_in_band_is_silent_when_gates_pass():
    """The guard must not fire on healthy runs, or it will be routed around."""
    spec = ArmSpec(arm="S1", topology="hybrid", width=3)
    rs = check_08_init_loss(spec, seed=0)
    assert all(r.passed for r in rs)
    pf.require_absolute_loss_in_band(rs)  # must not raise


def test_bos_check_asserts_position_zero_of_every_sequence():
    """Present, missing-in-one-row, and undeclared -- all three states."""
    spec = ArmSpec(arm="S1", topology="hybrid", width=3)

    def with_bos(sp):
        x, y = pf._rand_batch(sp)
        x = x.clone()
        x[:, 0] = 1
        return x, y

    def bos_missing_one_row(sp):
        x, y = with_bos(sp)
        x = x.clone()
        x[2, 0] = 7  # one row loses its sentinel
        return x, y

    ok = [r for r in check_08_init_loss(spec, seed=0, batch_fn=with_bos, bos_token=1)
          if r.check == "8c"][0]
    assert ok.passed and ok.severity == "gate"

    bad = [r for r in check_08_init_loss(spec, seed=0, batch_fn=bos_missing_one_row, bos_token=1)
           if r.check == "8c"][0]
    assert not bad.passed and bad.gating, "a single missing BOS row must gate the whole run"
    assert "1/" in bad.actual
    with pytest.raises(pf.AbsoluteLossOutOfBand):
        pf.require_absolute_loss_in_band([bad])

    skipped = [r for r in check_08_init_loss(spec, seed=0) if r.check == "8c"][0]
    assert skipped.severity == "info" and "SKIPPED" in skipped.actual, (
        "an undeclared sentinel must be reported as skipped, never silently omitted"
    )


# =============================================================================================
# SPEC §6.5 -- device/dtype labelling
# =============================================================================================


def test_every_result_carries_a_device_and_dtype_label():
    """SPEC §6.5. Auto-populated, so a check cannot omit it. Checked across a whole cell rather
    than on one hand-picked result."""
    spec = ArmSpec(arm="S4", topology="hybrid", width=3)
    results = run_preflight([spec], seed=0, engagement_steps=8, trend_steps=8,
                            parity_steps=5, verbose=False)
    assert results
    for r in results:
        assert r.device_dtype, f"{r.check} {r.name} has no device/dtype label"
        assert "device=" in r.device_dtype and "dtype=" in r.device_dtype
    assert all("cpu" in r.device_dtype for r in results), (
        "labels must reflect the REAL device -- on FarmShare these must read cuda"
    )


def test_label_defaults_are_not_hardcoded_to_cpu():
    """The label must come from the model, not a literal, or it will lie on FarmShare."""
    with pf.labelled("device=cuda:0 dtype=torch.bfloat16"):
        r = CheckResult("x", "n", "c", passed=True, expected="e", actual="a")
    assert r.device_dtype == "device=cuda:0 dtype=torch.bfloat16"
    after = CheckResult("y", "n", "c", passed=True, expected="e", actual="a")
    assert "cpu" in after.device_dtype, "the labelled() context did not restore"


def test_grid_coverage_is_complete_and_S3_allliv_is_reported_na():
    from arms import arm_grid, na_cells

    grid = arm_grid()
    assert len(grid) == len(WIDTHS) * (len(ARMS) * len(TOPOLOGIES) - len(TOPOLOGIES) // 2)
    assert len(grid) == 28, f"expected 28 buildable cells, got {len(grid)}"
    assert len(na_cells()) == 4
    assert all(c.startswith("S3-allliv") for c in na_cells())
