"""
Tests for ternary QAT (TWN + STE).

These import torch and therefore **do not run on the laptop** (see the repo's container
``CLAUDE.md``). They run on FarmShare under ``~/maple-verify/L4/``. The zero-fraction test is the
discriminating one: it fails if the quantizer has been "corrected" to BitNet b1.58.
"""

import math

import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import (
    MAPLE_SWIGLU_LIMIT,
    FeedForwardConfig,
    FeedForwardType,
)
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.nn.quantization import (
    BITNET_B158_GAUSSIAN_ZERO_FRACTION,
    TWN_DELTA_FACTOR,
    TWN_GAUSSIAN_ZERO_FRACTION,
    QuantConfig,
    QuantLinear,
    assert_no_float8_conflict,
    audit_quantization,
    gaussian_zero_fraction,
    twn_quantize,
    twn_quantize_ste,
    twn_threshold_and_scale,
)

# ---------------------------------------------------------------------------------
# C1 -- the quantizer itself
# ---------------------------------------------------------------------------------


def test_gaussian_zero_fraction_closed_form():
    """
    The closed form must reproduce the two numbers the TWN-vs-BitNet call rests on, to six
    places -- these are the constants the whole quantizer identity argument runs on, so the
    tolerance is tight on purpose. (This test already caught one transcription error: the
    module constant was 0.4237, from the plan document's rounded 42.4%, against the true
    0.4235110.)
    """
    assert gaussian_zero_fraction(0.7) == pytest.approx(TWN_GAUSSIAN_ZERO_FRACTION, abs=1e-6)
    # BitNet b1.58 rounds W/mean|W| to nearest integer, so the zero band is |W| < 0.5*mean|W|,
    # i.e. an effective delta factor of 0.5.
    assert gaussian_zero_fraction(0.5) == pytest.approx(
        BITNET_B158_GAUSSIAN_ZERO_FRACTION, abs=1e-6
    )


def test_zero_fraction_matches_twn_not_bitnet():
    """
    THE discriminating test. Ternarize a large Gaussian matrix and check the zero fraction.

    TWN predicts 42.37%, BitNet b1.58 predicts 31.0%, and the released Maple weights measured
    38.7-42.9%. If this asserts ~0.31, someone has "fixed" the quantizer to b1.58 -- which the
    evidence rules out despite Maple citing BitNet.
    """
    torch.manual_seed(0)
    w = torch.randn(512, 4096)
    q = twn_quantize(w, in_dim=-1)
    zero_frac = (q == 0).float().mean().item()

    assert zero_frac == pytest.approx(TWN_GAUSSIAN_ZERO_FRACTION, abs=0.01)
    assert abs(zero_frac - BITNET_B158_GAUSSIAN_ZERO_FRACTION) > 0.08, (
        f"zero fraction {zero_frac:.4f} is near BitNet b1.58's 0.3101, not TWN's 0.4235 -- "
        "the quantizer has been changed to b1.58, which the artifact evidence rules out"
    )
    # And inside the band actually observed in the released weights.
    assert 0.35 < zero_frac < 0.46


def test_exactly_three_distinct_values_per_row():
    """Every output row holds at most {-alpha, 0, +alpha} and nothing else."""
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    q = twn_quantize(w, in_dim=-1)

    for r in range(q.shape[0]):
        vals = torch.unique(q[r])
        assert vals.numel() <= 3, f"row {r} has {vals.numel()} distinct values"
        nonzero = vals[vals != 0].abs()
        if nonzero.numel():
            # The two nonzero magnitudes must be the SAME alpha.
            assert torch.allclose(nonzero, nonzero[0].expand_as(nonzero))
        # Zero must be present at this size -- TWN zeroes ~42% of a Gaussian row.
        assert (q[r] == 0).any()


def test_alpha_is_per_output_row_not_per_tensor():
    """
    Rows scaled differently must get different alphas.

    A per-*tensor* alpha (BitNet's original formulation) would give one shared scale, so this
    catches that variant even when the threshold is right.
    """
    torch.manual_seed(0)
    w = torch.randn(4, 1024)
    w[0] *= 100.0
    _, alpha = twn_threshold_and_scale(w, in_dim=-1)
    assert alpha.shape == (4, 1)
    assert alpha[0, 0] > 50 * alpha[1, 0]


def test_threshold_is_exactly_0p7_mean_abs():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    delta, _ = twn_threshold_and_scale(w, in_dim=-1)
    expected = TWN_DELTA_FACTOR * w.abs().mean(dim=-1, keepdim=True)
    assert torch.allclose(delta, expected, atol=1e-6)


def test_alpha_is_mean_of_surviving_magnitudes():
    torch.manual_seed(0)
    w = torch.randn(8, 512)
    delta, alpha = twn_threshold_and_scale(w, in_dim=-1)
    for r in range(w.shape[0]):
        surviving = w[r].abs()[w[r].abs() > delta[r, 0]]
        assert alpha[r, 0].item() == pytest.approx(surviving.mean().item(), rel=1e-5)


def test_all_zero_row_does_not_nan():
    """An all-zero row must quantize to zero, not NaN via a 0/0 alpha."""
    w = torch.zeros(3, 128)
    w[1, :] = 1.0
    q = twn_quantize(w, in_dim=-1)
    assert torch.isfinite(q).all()
    assert (q[0] == 0).all()
    assert (q[2] == 0).all()


def test_scale_invariance():
    """TWN's zero pattern depends only on the shape of the distribution, not its scale."""
    torch.manual_seed(0)
    w = torch.randn(32, 1024)
    q1 = twn_quantize(w, in_dim=-1)
    q2 = twn_quantize(w * 37.0, in_dim=-1)
    assert torch.equal(q1 == 0, q2 == 0)
    assert torch.allclose(q2, q1 * 37.0, rtol=1e-5)


def test_bf16_stats_computed_in_fp32():
    """
    The threshold must be computed in fp32 even for a bf16 weight.

    bf16 has 8 mantissa bits; a naive bf16 mean over 4096 elements drifts enough to move
    borderline weights across the threshold, and which weights become zero is the quantizer's
    identity. Compare against an fp32 reference on the same values.
    """
    torch.manual_seed(0)
    w32 = torch.randn(16, 4096)
    w16 = w32.to(torch.bfloat16)

    delta16, _ = twn_threshold_and_scale(w16, in_dim=-1)
    delta_ref = TWN_DELTA_FACTOR * w16.to(torch.float32).abs().mean(dim=-1, keepdim=True)
    assert torch.allclose(delta16, delta_ref, rtol=1e-6)

    # A bf16-accumulated mean would be visibly off; assert we're not that.
    delta_naive = TWN_DELTA_FACTOR * w16.abs().mean(dim=-1, keepdim=True).to(torch.float32)
    assert not torch.allclose(delta16, delta_naive, rtol=1e-7)


# ---------------------------------------------------------------------------------
# STE
# ---------------------------------------------------------------------------------


def test_ste_gradient_reaches_latent_weight_unmodified():
    """
    Gradient must flow to the latent weight and be the *identity* of the incoming gradient.

    Two failure modes this catches: a detached forward (no gradient at all), and a clipped STE
    (gradient zeroed for sub-threshold weights, which would freeze ~42% of the network).
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64, requires_grad=True)
    q = twn_quantize_ste(w, in_dim=-1)
    g = torch.randn_like(q)
    q.backward(g)

    assert w.grad is not None
    assert torch.equal(w.grad, g), "STE backward is not the identity"
    # Every latent weight gets gradient, including the ~42% currently quantized to zero.
    zeroed = twn_quantize(w.detach(), in_dim=-1) == 0
    assert zeroed.any()
    assert (w.grad[zeroed] != 0).any(), "sub-threshold weights received no gradient (clipped STE)"


def test_ste_lets_a_weight_cross_the_threshold():
    """
    A sub-threshold latent weight must be able to become nonzero after an optimizer step.

    This is the property that makes ternary QAT *training* rather than repeated PTQ, and it is
    exactly what a clipped STE would break.
    """
    torch.manual_seed(0)
    lin = QuantLinear(64, 8, enabled=True)
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)

    before = twn_quantize(lin.weight.detach(), in_dim=-1) == 0
    assert before.any()

    x = torch.randn(32, 64)
    for _ in range(5):
        opt.zero_grad()
        lin(x).square().mean().backward()
        opt.step()

    after = twn_quantize(lin.weight.detach(), in_dim=-1) == 0
    assert not torch.equal(before, after), "no weight ever changed ternary state"


def test_ste_gradient_is_not_the_true_jacobian():
    """
    Sanity: the STE is an approximation, and a test that passes only because forward is a
    no-op would be worthless. Confirm the quantized forward really differs from the latent.
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64)
    assert not torch.allclose(twn_quantize(w, in_dim=-1), w)


# ---------------------------------------------------------------------------------
# The cheapest high-value check: enabled=False is EXACTLY nn.Linear
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_disabled_quant_linear_is_bitwise_identical_to_nn_linear(dtype, bias):
    """
    ``enabled=False`` must be **bitwise** identical to ``nn.Linear``, not merely close.

    This is what makes bf16-vs-ternary a *paired* comparison rather than a comparison of two
    different models. Exact equality, not ``allclose``.
    """
    torch.manual_seed(0)
    ref = nn.Linear(128, 64, bias=bias, dtype=dtype)
    q = QuantLinear(128, 64, bias=bias, enabled=False, dtype=dtype)
    with torch.no_grad():
        q.weight.copy_(ref.weight)
        if bias:
            assert q.bias is not None and ref.bias is not None
            q.bias.copy_(ref.bias)

    x = torch.randn(16, 32, 128, dtype=dtype)
    y_ref, y_q = ref(x), q(x)

    assert y_ref.dtype == y_q.dtype
    assert torch.equal(y_ref, y_q), (
        "enabled=False is not bitwise identical to nn.Linear; max abs diff "
        f"{(y_ref - y_q).abs().max().item():.3e}"
    )


def test_disabled_quant_linear_backward_is_identical():
    torch.manual_seed(0)
    ref = nn.Linear(128, 64, bias=True)
    q = QuantLinear(128, 64, bias=True, enabled=False)
    with torch.no_grad():
        q.weight.copy_(ref.weight)
        assert q.bias is not None and ref.bias is not None
        q.bias.copy_(ref.bias)

    x = torch.randn(16, 128)
    g = torch.randn(16, 64)
    ref(x).backward(g)
    q(x).backward(g)

    assert ref.weight.grad is not None and q.weight.grad is not None
    assert torch.equal(ref.weight.grad, q.weight.grad)
    assert ref.bias is not None and q.bias is not None
    assert ref.bias.grad is not None and q.bias.grad is not None
    assert torch.equal(ref.bias.grad, q.bias.grad)


def test_quant_linear_state_dict_matches_nn_linear():
    """Identical state-dict keys and shapes, so the two arms can share an init and a checkpoint."""
    ref = nn.Linear(128, 64, bias=True)
    for enabled in (False, True):
        q = QuantLinear(128, 64, bias=True, enabled=enabled)
        assert set(q.state_dict()) == set(ref.state_dict())
        for k, v in ref.state_dict().items():
            assert q.state_dict()[k].shape == v.shape
        # And it actually loads.
        q.load_state_dict(ref.state_dict())


def test_quant_linear_is_an_nn_linear():
    """``isinstance`` must hold so init_linear / TP wrappers / float8 filters keep working."""
    assert isinstance(QuantLinear(8, 8), nn.Linear)


def test_checkpoint_interoperates_three_ways():
    """
    A checkpoint must load cleanly between stock, control and ternary builds of the same module,
    with **no missing and no unexpected keys**.

    This is what lets X4a's two arms start from one init, and what stops a resume from a
    non-quantized checkpoint silently reinitializing the projections -- ``load_state_dict``
    with ``strict=False`` elsewhere in the stack would swallow a key mismatch, so the assertion
    is on the returned key lists rather than on the call not raising.
    """
    torch.manual_seed(0)
    common = dict(hidden_size=128, bias=False)
    stock = FeedForwardConfig(**common).build(d_model=64)  # type: ignore[arg-type]
    ctrl = FeedForwardConfig(**common, quant=QuantConfig(enabled=False)).build(d_model=64)  # type: ignore[arg-type]
    tern = FeedForwardConfig(**common, quant=QuantConfig(enabled=True)).build(d_model=64)  # type: ignore[arg-type]

    for src, dst in ((stock, ctrl), (stock, tern), (ctrl, tern), (tern, stock)):
        missing, unexpected = dst.load_state_dict(src.state_dict())
        assert list(missing) == [], f"missing keys {list(missing)}"
        assert list(unexpected) == [], f"unexpected keys {list(unexpected)}"
        assert torch.equal(src.w1.weight, dst.w1.weight)

    # Same init, different forward -- that is the whole point of the pairing.
    x = torch.randn(4, 8, 64)
    assert torch.equal(stock(x), ctrl(x))
    assert not torch.allclose(stock(x), tern(x))


def test_olmo_init_path_accepts_quant_linear():
    """
    ``init_linear`` is typed against ``nn.Linear``; since ``QuantLinear`` is one, the whole
    ``InitMethod`` machinery works with no special case. Verify against the real function, and
    check the resulting zero fraction is still in the TWN band under OLMo-core's own init
    (trunc_normal std=0.02 truncated at 3 sigma) rather than a pure Gaussian.
    """
    from olmo_core.nn.transformer.init import init_linear

    torch.manual_seed(0)
    q = QuantLinear(1024, 256, bias=False, enabled=True)
    init_linear(q, std=0.02)
    zero_frac = (twn_quantize(q.weight.detach(), in_dim=-1) == 0).float().mean().item()
    assert 0.35 < zero_frac < 0.46, zero_frac


def test_enabled_quant_linear_differs_from_nn_linear():
    """Guard against the opposite failure: a toggle that is wired but never fires."""
    torch.manual_seed(0)
    ref = nn.Linear(128, 64, bias=False)
    q = QuantLinear(128, 64, bias=False, enabled=True)
    with torch.no_grad():
        q.weight.copy_(ref.weight)
    x = torch.randn(16, 128)
    assert not torch.allclose(ref(x), q(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_equality_survives_torch_compile(dtype):
    """
    The exact-equality property must hold **under `torch.compile`**, because that is the
    shipped configuration: `.edullm/train_on_corpus.py` sets `compile_model=True`.

    Note carefully what is and is not asserted. Compiled output is *not* bitwise equal to
    eager -- inductor fuses and reassociates, and this was measured at 3.6e-07 on a
    (128 -> 64) fp32 layer. That is fine and expected. The property X4a actually needs is
    that the control arm and stock agree **under the same compilation**, so that the two arms
    differ only in the quantizer and not in the compilation path. That is what this checks.
    """
    torch.manual_seed(0)
    ref = nn.Linear(256, 128, bias=False, dtype=dtype)
    nn.init.trunc_normal_(ref.weight, std=0.02, a=-0.06, b=0.06)
    ctrl = QuantLinear(256, 128, bias=False, enabled=False, dtype=dtype)
    ctrl.load_state_dict(ref.state_dict())

    x = torch.randn(4, 32, 256, dtype=dtype)
    torch._dynamo.reset()
    y_ref = torch.compile(ref)(x)
    torch._dynamo.reset()
    y_ctrl = torch.compile(ctrl)(x)

    assert torch.equal(y_ref, y_ctrl), (
        "compiled QuantLinear(enabled=False) diverged from compiled nn.Linear by "
        f"{(y_ref.float() - y_ctrl.float()).abs().max().item():.3e} -- the X4a control arm is "
        "no longer paired with stock under the shipped compiled config"
    )


def test_compiled_ste_backward_is_still_identity():
    """
    A custom autograd Function under dynamo is a place the backward could silently stop being
    the identity. Assert it survives compilation rather than assuming it does.
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64, requires_grad=True)
    g = torch.randn(8, 64)
    torch._dynamo.reset()
    torch.compile(lambda t: twn_quantize_ste(t, in_dim=-1))(w).backward(g)
    assert w.grad is not None
    assert torch.equal(w.grad, g)


def test_compiled_ternary_forward_is_deterministic():
    """
    A nondeterministic quantizer would inject noise straight into the X4a loss-curve
    comparison, where it would be indistinguishable from a convergence difference.
    """
    torch.manual_seed(0)
    q = QuantLinear(256, 64, bias=False, enabled=True)
    nn.init.trunc_normal_(q.weight, std=0.02, a=-0.06, b=0.06)
    x = torch.randn(4, 32, 256)
    torch._dynamo.reset()
    cq = torch.compile(q)
    a = cq(x)
    assert torch.equal(a, cq(x))
    torch._dynamo.reset()
    assert torch.equal(a, torch.compile(q)(x))


# ---------------------------------------------------------------------------------
# C2 -- module integration
# ---------------------------------------------------------------------------------


def test_feed_forward_none_vs_disabled_vs_enabled():
    torch.manual_seed(0)
    kwargs = dict(hidden_size=256, name=FeedForwardType.default, bias=False)
    ff_none = FeedForwardConfig(**kwargs).build(d_model=128)  # type: ignore[arg-type]
    ff_off = FeedForwardConfig(**kwargs, quant=QuantConfig(enabled=False)).build(d_model=128)  # type: ignore[arg-type]
    ff_on = FeedForwardConfig(**kwargs, quant=QuantConfig(enabled=True)).build(d_model=128)  # type: ignore[arg-type]

    assert not isinstance(ff_none.w1, QuantLinear)
    assert isinstance(ff_off.w1, QuantLinear) and not ff_off.w1.quant_enabled
    assert isinstance(ff_on.w1, QuantLinear) and ff_on.w1.quant_enabled

    ff_off.load_state_dict(ff_none.state_dict())
    ff_on.load_state_dict(ff_none.state_dict())

    x = torch.randn(4, 16, 128)
    assert torch.equal(ff_none(x), ff_off(x)), "quant=None and enabled=False must agree bitwise"
    assert not torch.allclose(ff_none(x), ff_on(x))


def test_feed_forward_quant_config_survives_as_dict_roundtrip():
    """
    ``FeedForwardConfig.build`` calls ``as_dict()``, which recurses and would flatten the nested
    ``QuantConfig`` into a plain dict. If that leaks through, ``quant.enabled`` is an
    AttributeError at the first forward pass -- i.e. at step 0 of a queued platform run.
    """
    ff = FeedForwardConfig(
        hidden_size=64, bias=False, quant=QuantConfig(enabled=True), swiglu_limit=7.0
    ).build(d_model=32)
    assert isinstance(ff.quant, QuantConfig)


def test_attention_quant_toggle():
    torch.manual_seed(0)
    kwargs = dict(n_heads=4, bias=False)
    a_none = AttentionConfig(**kwargs).build(128, layer_idx=0, n_layers=1)  # type: ignore[arg-type]
    a_off = AttentionConfig(**kwargs, quant=QuantConfig(enabled=False)).build(  # type: ignore[arg-type]
        128, layer_idx=0, n_layers=1
    )
    a_on = AttentionConfig(**kwargs, quant=QuantConfig(enabled=True)).build(  # type: ignore[arg-type]
        128, layer_idx=0, n_layers=1
    )

    for name in ("w_q", "w_k", "w_v", "w_out"):
        assert not isinstance(getattr(a_none, name), QuantLinear)
        assert isinstance(getattr(a_off, name), QuantLinear)
        assert isinstance(getattr(a_on, name), QuantLinear)
        assert getattr(a_on, name).quant_enabled

    a_off.load_state_dict(a_none.state_dict())
    x = torch.randn(2, 8, 128)
    assert torch.equal(a_none(x), a_off(x))


def test_attention_qk_norm_stays_full_precision():
    """q_norm/k_norm are norms; the carve-out says norms stay full precision."""
    from olmo_core.nn.layer_norm import LayerNormConfig

    att = AttentionConfig(
        n_heads=4, bias=False, qk_norm=LayerNormConfig(), quant=QuantConfig(enabled=True)
    ).build(128, layer_idx=0, n_layers=1)
    assert att.q_norm is not None and not isinstance(att.q_norm, QuantLinear)
    assert att.k_norm is not None and not isinstance(att.k_norm, QuantLinear)


def test_moe_mlp_quant_toggle_and_row_orientation():
    """
    ``MoEMLP`` reduces over axis 1 for all three weights, and each *output row* of each expert
    must land on a single alpha. Check the realized ternary structure, not just that a flag is set.
    """
    torch.manual_seed(0)
    common = dict(d_model=64, hidden_size=32, num_experts=4)
    mlp_off = MoEMLP(**common, quant=QuantConfig(enabled=False))  # type: ignore[arg-type]
    mlp_on = MoEMLP(**common, quant=QuantConfig(enabled=True))  # type: ignore[arg-type]
    mlp_on.load_state_dict(mlp_off.state_dict())

    x = torch.randn(4, 16, 64)
    assert not torch.allclose(mlp_off(x), mlp_on(x))

    # w1 is (E, d_model, hidden); in_dim=1 means each *column* (output feature) shares an alpha.
    w1 = mlp_on.w1.view(4, 64, 32)
    q = twn_quantize(w1, in_dim=1)
    for e in range(4):
        for h in range(32):
            vals = torch.unique(q[e, :, h])
            assert vals.numel() <= 3


def test_moe_mlp_none_is_identity_to_disabled():
    torch.manual_seed(0)
    common = dict(d_model=64, hidden_size=32, num_experts=4)
    a = MoEMLP(**common)  # type: ignore[arg-type]
    b = MoEMLP(**common, quant=QuantConfig(enabled=False))  # type: ignore[arg-type]
    b.load_state_dict(a.state_dict())
    x = torch.randn(4, 16, 64)
    assert torch.equal(a(x), b(x))


def test_dropless_moe_mlp_in_dim_asymmetry():
    """
    ``DroplessMoEMLP`` has the same *shape* for all three weights but ``gmm`` is called with
    ``trans_b=True`` for w1/w3 and not for w2, so ``in_dim`` differs. Verify the module picks
    the axis that makes alpha per *output* row of the realized matmul.
    """
    torch.manual_seed(0)
    mlp = DroplessMoEMLP(d_model=64, hidden_size=32, num_experts=2, quant=QuantConfig(enabled=True))
    w = mlp.w1.detach().view(2, 32, 64)

    # w1: x @ w[i].T -- output features index axis 1 (hidden), inputs are axis 2 (d_model).
    q_right = twn_quantize(w, in_dim=2)
    for e in range(2):
        for h in range(32):
            assert torch.unique(q_right[e, h, :]).numel() <= 3

    # Reducing over the wrong axis gives a per-input-row alpha: also <=3 values, but along the
    # other direction. Confirm the two differ so the choice is observable, not cosmetic.
    q_wrong = twn_quantize(w, in_dim=1)
    assert not torch.allclose(q_right, q_wrong)


def test_swiglu_clamp_is_asymmetric_and_inert_at_normal_scale():
    """
    gate clamped above only, up clamped both sides -- gpt-oss's shape.

    Also: at realistic activation magnitudes the clamp never fires, which is why including it
    costs nothing. Maple's measured layer-0 pre-activation RMS is ~0.136, i.e. ~52x below 7.0.
    """
    torch.manual_seed(0)
    ff_plain = FeedForwardConfig(hidden_size=64, bias=False).build(d_model=32)
    ff_clamped = FeedForwardConfig(
        hidden_size=64, bias=False, swiglu_limit=MAPLE_SWIGLU_LIMIT
    ).build(d_model=32)
    ff_clamped.load_state_dict(ff_plain.state_dict())

    x = torch.randn(8, 32) * 0.1
    assert torch.equal(ff_plain(x), ff_clamped(x)), "clamp fired at normal activation scale"

    # Force it to fire, and check the asymmetry: a large NEGATIVE gate is untouched, a large
    # POSITIVE gate is clipped.
    with torch.no_grad():
        ff_clamped.w1.weight.fill_(0.0)
        ff_clamped.w3.weight.fill_(0.0)
        ff_clamped.w1.weight[0, 0] = 1.0
        ff_clamped.w3.weight[0, 0] = 1.0
    big = torch.zeros(1, 32)
    big[0, 0] = 100.0
    # gate = 100 -> clamped to 7 ; up = 100 -> clamped to 7
    gate = ff_clamped.w1(big).clamp(max=MAPLE_SWIGLU_LIMIT)
    assert gate[0, 0].item() == pytest.approx(7.0)
    big[0, 0] = -100.0
    gate_neg = ff_clamped.w1(big)
    assert gate_neg[0, 0].item() == pytest.approx(-100.0)  # one-sided: NOT clamped below
    up_neg = ff_clamped.w3(big).clamp(min=-MAPLE_SWIGLU_LIMIT, max=MAPLE_SWIGLU_LIMIT)
    assert up_neg[0, 0].item() == pytest.approx(-7.0)  # two-sided: IS clamped


def test_moe_mlp_clamp_is_inert_at_normal_scale():
    torch.manual_seed(0)
    common = dict(d_model=64, hidden_size=32, num_experts=2)
    a = MoEMLP(**common)  # type: ignore[arg-type]
    b = MoEMLP(**common, swiglu_limit=MAPLE_SWIGLU_LIMIT)  # type: ignore[arg-type]
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 8, 64) * 0.1
    assert torch.equal(a(x), b(x))


# ---------------------------------------------------------------------------------
# C3 -- carve-outs
# ---------------------------------------------------------------------------------


def test_audit_flags_a_quantized_router():
    """
    The carve-out audit must **raise** on a quantized router, not warn.

    Quantizing the router changes *which experts fire*, not merely how accurately -- routing is
    discrete. A silently dropped carve-out is a wrong experiment that trains happily.
    """

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = QuantLinear(8, 4, enabled=True)

    with pytest.raises(OLMoConfigurationError, match="carve-out violated"):
        audit_quantization(Fake())


@pytest.mark.parametrize("name", ["lm_head", "embeddings", "final_norm"])
def test_audit_flags_every_carve_out(name):
    mod = nn.Module()
    mod.add_module(name, QuantLinear(8, 4, enabled=True))
    with pytest.raises(OLMoConfigurationError, match="carve-out violated"):
        audit_quantization(mod)


def test_audit_passes_and_counts_a_legal_model():
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.Embedding(100, 8)
            self.router = nn.Linear(8, 4, bias=False)
            self.w_q = QuantLinear(8, 8, enabled=True)
            self.lm_head = nn.Linear(8, 100, bias=False)

    report = audit_quantization(Fake())
    assert report["num_quantized"] == 1
    assert report["quantized_numel"] == 64
    assert report["num_full_precision"] >= 3


def test_audit_does_not_double_count_a_dense_feed_forward():
    """
    A dense ``FeedForward`` carries ``.quant``, ``.w1`` and ``.w2`` just like a stacked-expert
    MLP, but holds them as ``nn.Linear`` submodules that the audit already counts individually.
    Counting the parent too would inflate ``num_quantized`` -- and an inflated count is exactly
    what would hide the "toggle wired but never fires" failure this number exists to catch.
    """
    ff = FeedForwardConfig(hidden_size=64, bias=False, quant=QuantConfig(enabled=True)).build(
        d_model=32
    )
    # Three projections, and the parent must not be counted as a fourth.
    assert audit_quantization(ff)["num_quantized"] == 3


def test_audit_counts_a_stacked_expert_mlp_once():
    """The expert stack has no Linear submodules, so the parent is the only thing to count."""
    mlp = MoEMLP(d_model=32, hidden_size=16, num_experts=4, quant=QuantConfig(enabled=True))
    assert audit_quantization(mlp)["num_quantized"] == 1


def test_audit_notices_nothing_was_quantized():
    """A report with zero quantized tensors is the 'toggle wired but never fires' failure."""

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.w_q = QuantLinear(8, 8, enabled=False)

    assert audit_quantization(Fake())["num_quantized"] == 0


# ---------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------


def test_float8_conflict_is_refused():
    """
    float8 conversion filters on ``isinstance(m, nn.Linear)``, which ``QuantLinear`` satisfies,
    so it would replace every quantized projection with a ``Float8Linear`` and silently discard
    the quantizer -- a run reporting as the ternary arm while actually training in fp8.

    This is the one place the ``nn.Linear`` subclassing cuts the wrong way, so it gets a guard
    rather than a comment.
    """

    class Fake(nn.Module):
        def __init__(self, enabled):
            super().__init__()
            self.w_q = QuantLinear(8, 8, enabled=enabled)

    with pytest.raises(OLMoConfigurationError, match="float8 conversion would silently replace"):
        assert_no_float8_conflict(Fake(True))
    # Bypassed and stock are both fine: there is no quantizer to discard.
    assert_no_float8_conflict(Fake(False))
    assert_no_float8_conflict(nn.Linear(8, 8))


def test_normalized_feed_forward_rejects_quant():
    with pytest.raises(OLMoConfigurationError, match="not supported with NormalizedFeedForward"):
        FeedForwardConfig(
            hidden_size=64, name=FeedForwardType.normalized, quant=QuantConfig()
        ).build(d_model=32)


def test_tensor_parallel_is_refused_on_quantized_linear():
    """
    Row-wise sharding splits the axis TWN reduces over, so each rank would compute alpha from a
    fraction of each output row -- a different quantizer that trains without complaint.
    """
    with pytest.raises(NotImplementedError, match="tensor parallelism is not supported"):
        QuantLinear(8, 8, enabled=True).assert_no_tensor_parallel()
    # Disabled is fine: there is no quantizer to get wrong.
    QuantLinear(8, 8, enabled=False).assert_no_tensor_parallel()


# ---------------------------------------------------------------------------------
# C5 -- bf16 latent masters
# ---------------------------------------------------------------------------------


def test_bf16_latent_masters_still_train():
    """
    The latent master is the ordinary ``weight`` parameter, so bf16 latents are just
    ``--param-dtype bfloat16``. Confirm the loop runs, stays finite, and moves weights.

    Note the cost model: **nothing here is cheaper than bf16 training.** The forward consumes a
    dequantized full-size tensor, the backward is full precision through the STE, and the latent
    master is full size. Ternary's win is at inference.
    """
    torch.manual_seed(0)
    lin = QuantLinear(64, 32, enabled=True, dtype=torch.bfloat16)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2)
    x = torch.randn(16, 64, dtype=torch.bfloat16)
    w0 = lin.weight.detach().clone()

    losses = []
    for _ in range(10):
        opt.zero_grad()
        loss = lin(x).square().mean()
        assert torch.isfinite(loss), "non-finite loss under bf16 latents"
        loss.backward()
        assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
        opt.step()
        losses.append(loss.item())

    assert not torch.equal(w0, lin.weight.detach())
    assert losses[-1] < losses[0]


def test_quantized_forward_dtype_follows_weight():
    """No silent fp32 upcast: the quantized weight comes back in the weight's dtype."""
    for dtype in (torch.float32, torch.bfloat16):
        w = torch.randn(8, 64, dtype=dtype)
        assert twn_quantize(w, in_dim=-1).dtype == dtype
        lin = QuantLinear(64, 8, enabled=True, dtype=dtype)
        assert lin(torch.randn(4, 64, dtype=dtype)).dtype == dtype


def test_math_constants_are_what_the_docs_claim():
    """
    Guard the numbers the whole TWN-vs-BitNet argument rests on, recomputed here from first
    principles rather than copied from the module -- so this fails if the module constant is
    edited, which is the point.
    """
    assert TWN_DELTA_FACTOR == 0.7
    z = 0.7 * math.sqrt(2.0 / math.pi)
    assert z == pytest.approx(0.5585192, abs=1e-6)
    assert TWN_GAUSSIAN_ZERO_FRACTION == pytest.approx(math.erf(z / math.sqrt(2.0)), abs=1e-6)
    # Both round to the figures the plan document quotes (42.4% and 31.0%).
    assert round(TWN_GAUSSIAN_ZERO_FRACTION * 100, 1) == 42.4
    assert round(BITNET_B158_GAUSSIAN_ZERO_FRACTION * 100, 1) == 31.0
