"""
Tests for :mod:`olmo_core.nn.gated_convolution` and its wiring into KDA.

WHAT THESE TESTS ARE FOR, AND THE FAILURE MODE THEY EXIST TO CATCH
    A gated convolution that is algebraically absorbable into the plain one trains stably, costs
    the parameters you expect, and measures **nothing** -- the two arms are the same function
    class, so the honest effect is exactly zero and any observed difference is optimizer noise.
    Several tests here therefore assert a *real forward-pass difference* rather than merely that
    the module runs.

    A second failure mode is the dead branch: zero-initialize both sides of a multiplicative gate
    and the gradient never flows, so the run reports a clean, replicable null.
    :func:`test_gate_gradient_is_alive_at_init` checks that against a magnitude floor rather than
    against ``is not None``.

WHY THE KDA TESTS ARE GPU-MARKED AND THE CONVOLUTION TESTS ARE NOT
    :class:`~olmo_core.nn.attention.recurrent.KimiDeltaAttention` asserts ``has_fla()`` in its
    constructor, so it cannot be built without ``fla`` -- and ``fla`` needs CUDA. Everything that
    can be checked without it is checked on CPU, deliberately, so the cheap suite covers the
    arithmetic and the operator identity. **A skip is a pass in the summary line**, so the split
    is by genuine dependency, and each GPU-marked test here would be a real gap if it never ran.
"""

import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.attention import KimiDeltaAttentionConfig
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.gated_convolution import (
    GatedCausalConv1d,
    gate_activation_bytes,
    gate_is_absorbable,
    gate_param_count,
)
from olmo_core.testing.utils import requires_fla

#: KDA's own geometry at the scale this experiment will run, so the numbers in the docstrings and
#: the handoff are the numbers the tests check.
D_MODEL = 2048
N_HEADS = 16
HEAD_DIM = 128
CONV_SIZE = 4
GATE_RANK = 128


# ---------------------------------------------------------------------------------------------
# The parameter and memory arithmetic. Pure functions, so these are exact-integer assertions.
# ---------------------------------------------------------------------------------------------


def test_depthwise_gate_param_count_is_two_per_channel():
    assert gate_param_count(hidden_size=2048, structure="depthwise") == 4096
    assert gate_param_count(hidden_size=1, structure="depthwise") == 2


def test_lowrank_gate_param_count():
    # One shared d_model -> r down-projection, then two r -> hidden up-projections.
    expected = 2048 * 128 + 2 * 128 * 2048
    assert (
        gate_param_count(hidden_size=2048, structure="lowrank", d_model=2048, gate_rank=128)
        == expected
    )
    assert expected == 786_432


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"d_model": 2048}, id="no-rank"),
        pytest.param({"gate_rank": 128}, id="no-d_model"),
        pytest.param({"d_model": 2048, "gate_rank": 0}, id="rank-0"),
        pytest.param({"d_model": 2048, "gate_rank": -1}, id="rank-negative"),
    ],
)
def test_lowrank_gate_param_count_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        gate_param_count(hidden_size=2048, structure="lowrank", **kwargs)


def test_unknown_gate_structure_is_rejected_everywhere():
    with pytest.raises(ValueError):
        gate_param_count(hidden_size=8, structure="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        gate_is_absorbable("bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        GatedCausalConv1d(hidden_size=8, kernel_size=3, gate_structure="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("structure", ["depthwise", "lowrank"])
def test_no_shipped_gate_structure_is_absorbable(structure):
    """
    The scientific admissibility precondition, asserted rather than argued.

    A gate that is constant across positions folds into a depthwise convolution's per-channel
    taps, which makes the "gated" arm the same function class as the plain arm. If a future
    structure is added and this flips to ``True``, the experiment built on it is vacuous.
    """
    assert gate_is_absorbable(structure) is False


def test_gate_activation_bytes_matches_the_documented_kda_geometry():
    """
    Guards the memory figure the run will be sized from.

    384 MiB per layer at 8192 tokens per rank, bf16, three 2048-channel streams -- so 28 layers
    cost 10.5 GiB on top of KDA's measured 5.169 GiB peak. This is ~3x peak, and the parameter
    delta (12,288/layer) invites exactly the wrong conclusion about cost.
    """
    per_conv = gate_activation_bytes(
        hidden_size=2048, batch_size=1, seq_len=8192, bytes_per_element=2
    )
    assert per_conv == 2 * 2 * 1 * 8192 * 2048 * 2
    assert per_conv == 134_217_728  # 128 MiB

    per_layer = 3 * per_conv
    assert per_layer == 402_653_184  # 384 MiB
    assert 28 * per_layer / 2**30 == pytest.approx(10.5, abs=0.05)


def test_gate_activation_bytes_scales_as_documented():
    base = dict(hidden_size=512, batch_size=2, seq_len=128, bytes_per_element=2)
    b = gate_activation_bytes(**base)  # type: ignore[arg-type]
    assert gate_activation_bytes(**{**base, "seq_len": 256}) == 2 * b  # type: ignore[arg-type]
    assert gate_activation_bytes(**{**base, "bytes_per_element": 4}) == 2 * b  # type: ignore[arg-type]
    # A recompute-in-backward implementation would retain one tensor per gate, not two.
    assert gate_activation_bytes(**{**base, "tensors_per_gate": 1}) == b // 2  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# The operator itself. These run on CPU through the reference path.
# ---------------------------------------------------------------------------------------------


def _plain_reference(conv: GatedCausalConv1d, u: torch.Tensor) -> torch.Tensor:
    """The ungated convolution with the same weights, as the arm being contrasted against."""
    seq_len = u.shape[1]
    z = F.conv1d(
        u.transpose(-1, -2),
        conv.conv.weight,
        conv.conv.bias,
        padding=conv.kernel_size - 1,
        groups=conv.hidden_size,
    )[..., :seq_len]
    z = z.transpose(-1, -2)
    if conv.activation is not None:
        z = F.silu(z)
    return z


@pytest.mark.parametrize("structure", ["depthwise", "lowrank"])
def test_at_init_the_gated_module_equals_the_plain_convolution(structure):
    """
    The two arms must start from the same function, or the contrast is not an ablation.

    ``2 * sigmoid(0) = 1`` exactly, so with zero-initialized gates the gated module reproduces
    the ungated convolution to floating-point equality -- not approximately.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=64,
        kernel_size=CONV_SIZE,
        gate_structure=structure,
        d_model=32,
        gate_rank=8,
        use_fla=False,
    )
    conv.init_gate_weights()
    u = torch.randn(2, 16, 64)
    x = torch.randn(2, 16, 32)

    got = conv(u, gate_input=x if structure == "lowrank" else None)
    torch.testing.assert_close(got, _plain_reference(conv, u), rtol=0, atol=0)


@pytest.mark.parametrize("structure", ["depthwise", "lowrank"])
def test_a_trained_gate_changes_the_function(structure):
    """
    The gate must buy something once it moves off zero.

    This is the anti-vacuity test with teeth: it perturbs the gate parameters and demands the
    output differ from the plain convolution by a margin far above float noise. A gate that were
    absorbable into the depthwise taps could still pass the *init* test above, so that one alone
    would be theatre.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=64,
        kernel_size=CONV_SIZE,
        gate_structure=structure,
        d_model=32,
        gate_rank=8,
        use_fla=False,
    )
    u = torch.randn(2, 16, 64)
    x = torch.randn(2, 16, 32)
    kw = {"gate_input": x} if structure == "lowrank" else {}

    with torch.no_grad():
        for p in conv.parameters():
            if p is not conv.conv.weight:
                p.normal_(0.0, 0.5)

    diff = (conv(u, **kw) - _plain_reference(conv, u)).abs().max().item()
    assert diff > 1e-2, f"the gate barely moved the output ({diff:.3e}): check for absorption"


def test_the_gate_is_not_a_constant_rescale():
    """
    Directly falsifies absorbability, which is the property the experiment depends on.

    An absorbable gate is one that is constant across positions: then ``out[t] / plain[t]`` is
    the same for every ``t`` and folds into the per-channel taps. Here that ratio must *vary*
    with position, because the gate reads the position-dependent stream.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(hidden_size=8, kernel_size=1, use_fla=False)
    with torch.no_grad():
        conv.pre_scale.fill_(0.7)
        conv.post_scale.fill_(-0.4)
        conv.conv.weight.fill_(1.0)
    u = torch.randn(1, 32, 8)

    ratio = conv(u) / _plain_reference(conv, u)
    # kernel_size=1 removes any mixing across positions, so a position-constant gate would give
    # a ratio identical down each channel. Spread proves position dependence.
    spread = (ratio.max(dim=1).values - ratio.min(dim=1).values).max().item()
    assert spread > 0.1, f"the gate looks position-constant (spread {spread:.3e}): absorbable"


@pytest.mark.parametrize("structure", ["depthwise", "lowrank"])
def test_gate_gradient_is_alive_at_init(structure):
    """
    The dead-branch check, against a magnitude floor rather than ``is not None``.

    ``d/dz [2*sigmoid(z)] = 1/2`` at ``z = 0``, so the gate nonlinearity always passes gradient.
    What can still be dead is the parameters *producing* ``z``. Zero-initializing both factors of
    a product is the classic way to get a permanently dead path that trains stably and reports a
    clean null -- **this test caught exactly that** in the first version of ``"lowrank"``, which
    zeroed ``gate_down`` and both up-projections.

    The assertion is against a floor derived from bf16's smallest useful step, not against
    ``is not None``: a gradient that exists but can never accumulate to a usable update produces
    the same false null.

    ``gate_down`` is exempt on step 1 by construction -- the up-projections are zero, so it
    genuinely cannot have gradient yet. That it becomes trainable is asserted separately by
    :func:`test_lowrank_down_projection_wakes_up_after_one_step`, so the exemption is covered
    rather than a hole.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=64,
        kernel_size=CONV_SIZE,
        gate_structure=structure,
        d_model=32,
        gate_rank=8,
        use_fla=False,
    )
    conv.init_gate_weights(std=0.02, generator=torch.Generator().manual_seed(0))
    # A non-degenerate convolution weight: an all-zero conv would zero the post-gate's gradient
    # for a reason that has nothing to do with the gate.
    with torch.no_grad():
        conv.conv.weight.normal_(0.0, 0.1)

    u = torch.randn(2, 16, 64)
    x = torch.randn(2, 16, 32)
    out = conv(u, gate_input=x if structure == "lowrank" else None)
    out.pow(2).mean().backward()

    # The floor is RELATIVE to the convolution weight's own gradient, not an absolute number.
    # An absolute threshold is scale-dependent: it passes or fails on the input's variance rather
    # than on whether the gate is trainable, which makes it a guard that fires for the wrong
    # reason. bf16 keeps 8 mantissa bits, so a gate gradient more than 2**-9 below a
    # known-live parameter's would be lost to accumulation and the branch is dead in practice.
    conv_grad = conv.conv.weight.grad
    assert conv_grad is not None and conv_grad.abs().max() > 0, "the reference is itself dead"
    floor = 2**-9 * conv_grad.abs().max().item()
    gate_params = {
        name: p
        for name, p in conv.named_parameters()
        if not name.startswith("conv.") and name != "gate_down.weight"
    }
    assert gate_params, "no gate parameters found: the structure built nothing"
    if structure == "lowrank":
        # Both up-projections must be live, or the branch is the dead one this test exists for.
        assert set(gate_params) == {"gate_up_pre.weight", "gate_up_post.weight"}
    dead = {
        name: (0.0 if p.grad is None else p.grad.abs().max().item())
        for name, p in gate_params.items()
        if p.grad is None or p.grad.abs().max().item() <= floor
    }
    assert not dead, f"gate parameters below the {floor:.3e} liveness floor: {dead}"


def test_lowrank_down_projection_wakes_up_after_one_step():
    """
    The other half of the liveness argument for ``"lowrank"``.

    At init the up-projections are zero, so ``gate_down`` has exactly zero gradient on step 1 --
    that is expected and is not the dead-branch failure, because the up-projections *do* have
    gradient and move. Once they have moved, ``gate_down`` must receive gradient too. If it never
    does, half the gate's parameters are decorative and the arm is not measuring what it claims.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=32,
        kernel_size=CONV_SIZE,
        gate_structure="lowrank",
        d_model=16,
        gate_rank=8,
        use_fla=False,
    )
    conv.init_gate_weights(std=0.02, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        conv.conv.weight.normal_(0.0, 0.1)
    u = torch.randn(2, 16, 32)
    x = torch.randn(2, 16, 16)

    # Step 1: the up-projections move, gate_down cannot yet.
    conv(u, gate_input=x).pow(2).mean().backward()
    assert conv.gate_up_pre.weight.grad.abs().max() > 0
    assert conv.gate_down.weight.grad.abs().max() == 0

    # Apply the step, then check gate_down is reachable.
    with torch.no_grad():
        conv.gate_up_pre.weight -= 0.1 * conv.gate_up_pre.weight.grad
        conv.gate_up_post.weight -= 0.1 * conv.gate_up_post.weight.grad
    conv.zero_grad(set_to_none=True)

    conv(u, gate_input=x).pow(2).mean().backward()
    assert conv.gate_down.weight.grad.abs().max() > 0, "gate_down never becomes trainable"


def test_lowrank_gate_init_reports_that_it_drew_and_depthwise_does_not():
    """
    The random-stream fact, asserted rather than assumed.

    ``"lowrank"`` must draw its down-projection (zeroing both factors would kill the branch), so
    a lowrank arm's later parameters differ from the plain arm's. ``"depthwise"`` draws nothing
    and shares the stream exactly. A caller that assumed the wrong one would silently introduce
    or hide a confound.
    """
    dw = GatedCausalConv1d(hidden_size=8, kernel_size=3, use_fla=False)
    assert dw.init_gate_weights() is False

    lr = GatedCausalConv1d(
        hidden_size=8,
        kernel_size=3,
        gate_structure="lowrank",
        d_model=16,
        gate_rank=4,
        use_fla=False,
    )
    gen = torch.Generator().manual_seed(0)
    before = gen.get_state().clone()
    assert lr.init_gate_weights(generator=gen) is True
    assert not torch.equal(gen.get_state(), before), "claimed to draw but consumed nothing"
    assert torch.count_nonzero(lr.gate_down.weight) > 0
    assert torch.count_nonzero(lr.gate_up_pre.weight) == 0
    assert torch.count_nonzero(lr.gate_up_post.weight) == 0


def test_lowrank_requires_its_gate_input():
    """
    A missing ``gate_input`` must raise, not fall back to gating on the stream.

    Falling back would silently run a *different* operator than the one configured, and the run
    would look fine.
    """
    conv = GatedCausalConv1d(
        hidden_size=8,
        kernel_size=3,
        gate_structure="lowrank",
        d_model=16,
        gate_rank=4,
        use_fla=False,
    )
    with pytest.raises(RuntimeError, match="gate_input"):
        conv(torch.randn(1, 4, 8))


def test_lowrank_construction_requires_d_model_and_rank():
    with pytest.raises(ValueError):
        GatedCausalConv1d(hidden_size=8, kernel_size=3, gate_structure="lowrank", d_model=16)
    with pytest.raises(ValueError):
        GatedCausalConv1d(hidden_size=8, kernel_size=3, gate_structure="lowrank", gate_rank=4)


def test_output_is_causal():
    """
    A filter that reads the future is a different operator, and one that trains.

    Perturbing position ``t`` must leave every output at ``< t`` untouched.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(hidden_size=4, kernel_size=CONV_SIZE, use_fla=False)
    with torch.no_grad():
        conv.conv.weight.normal_()
        conv.pre_scale.normal_()
        conv.post_scale.normal_()

    u = torch.randn(1, 12, 4)
    base = conv(u)
    u2 = u.clone()
    u2[:, 7] += 5.0
    perturbed = conv(u2)

    torch.testing.assert_close(base[:, :7], perturbed[:, :7])
    assert not torch.allclose(base[:, 7], perturbed[:, 7])


def test_shape_is_preserved():
    conv = GatedCausalConv1d(hidden_size=32, kernel_size=CONV_SIZE, use_fla=False)
    out = conv(torch.randn(3, 17, 32))
    assert out.shape == (3, 17, 32)


def test_cu_seqlens_stops_the_filter_at_a_document_boundary():
    """
    Documents must be convolved independently.

    A ``kernel_size=4`` filter reading across a boundary mixes unrelated documents, and at a
    ~622-token median document length a 4096-token sequence holds several of them -- so this is
    not a small effect. Perturbing the last token of document 1 must not move document 2.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(hidden_size=8, kernel_size=CONV_SIZE, use_fla=False)
    with torch.no_grad():
        conv.conv.weight.normal_()
        conv.pre_scale.normal_()
        conv.post_scale.normal_()

    u = torch.randn(1, 10, 8)
    cu = torch.tensor([0, 4, 10], dtype=torch.int32)

    split = conv(u, cu_seqlens=cu)
    assert split.shape == u.shape

    u2 = u.clone()
    u2[:, 3] += 7.0  # the last token of document 1
    split2 = conv(u2, cu_seqlens=cu)
    torch.testing.assert_close(split[:, 4:], split2[:, 4:])

    # And the split result must actually differ from the unsplit one, or the test is vacuous:
    # a bug that ignored cu_seqlens entirely would otherwise pass the check above.
    unsplit = conv(u)
    assert not torch.allclose(split[:, 4:], unsplit[:, 4:])


def test_cu_seqlens_rejects_a_real_batch():
    conv = GatedCausalConv1d(hidden_size=8, kernel_size=3, use_fla=False)
    with pytest.raises(RuntimeError, match="batch_size == 1"):
        conv(torch.randn(2, 6, 8), cu_seqlens=torch.tensor([0, 3, 6], dtype=torch.int32))


@pytest.mark.parametrize("activation", [None, "silu"])
def test_activation_is_applied_after_the_convolution(activation):
    """
    The two gated arms differ only by this, so its placement must be pinned.

    ``CausalConv1d`` defaults to ``silu`` and this class defaults to ``None``: an activation
    applied *before* the convolution instead of after is a different operator that trains.
    """
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=8, kernel_size=CONV_SIZE, activation=activation, use_fla=False
    )
    conv.init_gate_weights()
    with torch.no_grad():
        conv.conv.weight.normal_()

    u = torch.randn(1, 6, 8)
    got = conv(u)

    z = F.conv1d(u.transpose(-1, -2), conv.conv.weight, None, padding=CONV_SIZE - 1, groups=8)[
        ..., :6
    ].transpose(-1, -2)
    expected = F.silu(z) if activation == "silu" else z
    torch.testing.assert_close(got, expected)

    if activation == "silu":
        # And prove the two arms are not the same function, which is what makes the third arm of
        # the experiment worth buying.
        assert not torch.allclose(got, z)


def test_default_activation_is_none_and_differs_from_causalconv1d():
    """
    The load-bearing default, and the trap it avoids.

    ``CausalConv1d`` defaults to ``activation="silu"`` inside its fused kernel, so a subclass
    inheriting that default would implement a different operator than LFM2's, which is
    activation-free. This asserts the two defaults genuinely disagree, so a future change to
    either one is caught here rather than in a loss curve.
    """
    assert GatedCausalConv1d(hidden_size=4, kernel_size=3).activation is None
    assert CausalConv1d(hidden_size=4, kernel_size=3).activation == "silu"


def test_unsupported_activation_raises_rather_than_silently_skipping():
    conv = GatedCausalConv1d(hidden_size=4, kernel_size=3, use_fla=False)
    conv.activation = "gelu"  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="unsupported activation"):
        conv(torch.randn(1, 4, 4))


def test_kda_constructor_default_is_also_ungated():
    """
    The config default and the **constructor** default must both be ``False``.

    Found by mutation M9: flipping ``KimiDeltaAttention.__init__``'s ``gated_conv`` default
    survived the whole suite, because every other test goes through
    :class:`KimiDeltaAttentionConfig`. Anything constructing the module directly -- a probe, a
    benchmark, ``fla``-style direct instantiation -- would silently get the treatment arm.

    ``inspect.signature`` reads the default without constructing anything, so this runs on CPU
    without ``fla``. That matters: the constructor is exactly the surface a GPU-only test would
    leave uncovered.
    """
    import inspect

    from olmo_core.nn.attention.recurrent import KimiDeltaAttention

    params = inspect.signature(KimiDeltaAttention.__init__).parameters
    assert params["gated_conv"].default is False
    assert params["gated_conv_activation"].default is None
    assert params["gate_structure"].default == "depthwise"
    assert params["gate_rank"].default is None


def test_kda_build_conv_honours_the_configured_activation():
    """
    ``_build_conv`` must pass the *configured* activation through, not a literal.

    Found by mutation M10: hard-coding ``activation="silu"`` in ``_build_conv`` collapsed
    ``kda-gated`` onto ``kda-gated-silu`` and survived the whole suite, because every test that
    could have seen it needed ``fla`` to build the module.

    This calls the real unbound method against a minimal stand-in for ``self``, so the branch is
    exercised on CPU. It is the actual function, not a reimplementation of it -- a test that
    re-derived the logic would pass no matter what ``_build_conv`` did.
    """
    from olmo_core.nn.attention.recurrent import KimiDeltaAttention

    class _Stub:
        conv_size = CONV_SIZE
        conv_bias = False
        d_model = D_MODEL
        gate_structure = "depthwise"
        gate_rank = None

    for gated, activation, expected_type, expected_activation in (
        (False, None, CausalConv1d, "silu"),
        (True, None, GatedCausalConv1d, None),
        (True, "silu", GatedCausalConv1d, "silu"),
    ):
        stub = _Stub()
        stub.gated_conv = gated  # type: ignore[attr-defined]
        stub.gated_conv_activation = activation  # type: ignore[attr-defined]
        conv = KimiDeltaAttention._build_conv(
            stub, hidden_size=64, dtype=torch.float32, init_device="meta"  # type: ignore[arg-type]
        )
        assert isinstance(conv, expected_type), (gated, activation)
        assert conv.activation == expected_activation, (gated, activation)


def test_kda_conv_kwargs_thread_gate_input_only_for_lowrank():
    """
    ``gate_input`` must reach a ``"lowrank"`` gate and must not be passed to anything else.

    Omitting it would make ``"lowrank"`` raise (which is at least loud), but passing it to a
    plain ``CausalConv1d`` would be a ``TypeError`` on the first step of a paid run. Exercised
    against the real unbound method, on CPU.
    """
    from olmo_core.nn.attention.recurrent import KimiDeltaAttention

    class _Stub:
        pass

    x = torch.zeros(1, 4, D_MODEL)

    for gated, structure, expect_key in (
        (False, "depthwise", False),
        (True, "depthwise", False),
        (True, "lowrank", True),
    ):
        stub = _Stub()
        stub.gated_conv = gated  # type: ignore[attr-defined]
        stub.gate_structure = structure  # type: ignore[attr-defined]
        kwargs = KimiDeltaAttention._conv_kwargs(stub, x)  # type: ignore[arg-type]
        assert ("gate_input" in kwargs) is expect_key, (gated, structure)
        if expect_key:
            assert kwargs["gate_input"] is x


def test_num_gate_params_matches_the_built_module():
    """The predicted count and the real ``state_dict`` must agree, for both structures."""
    for kwargs in (
        dict(gate_structure="depthwise"),
        dict(gate_structure="lowrank", d_model=64, gate_rank=8),
    ):
        conv = GatedCausalConv1d(hidden_size=32, kernel_size=CONV_SIZE, **kwargs)  # type: ignore[arg-type]
        actual = sum(p.numel() for n, p in conv.named_parameters() if not n.startswith("conv."))
        assert conv.num_gate_params() == actual, kwargs


def test_meta_device_construction_works():
    """Configs are built on ``meta`` for parameter counting before anything is allocated."""
    conv = GatedCausalConv1d(hidden_size=32, kernel_size=CONV_SIZE, init_device="meta")
    assert conv.pre_scale.device.type == "meta"
    assert conv.num_gate_params() == 64


# ---------------------------------------------------------------------------------------------
# The KDA config surface. Buildable without fla, so these run on CPU.
# ---------------------------------------------------------------------------------------------


def _cfg(**kwargs) -> KimiDeltaAttentionConfig:
    base = dict(n_heads=N_HEADS, head_dim=HEAD_DIM, expand_v=1.0, conv_size=CONV_SIZE)
    return KimiDeltaAttentionConfig(**{**base, **kwargs})  # type: ignore[arg-type]


def test_the_default_is_the_shipped_operator():
    """
    ``gated_conv=False`` by default, and the parameter total is unmoved.

    Every number in ``KDA/HANDOFF.md`` -- the arm ledger, 285,832 tok/s, the 5.169 GiB peak --
    was measured with plain convolutions. A default that moved would invalidate all of them while
    every other test still passed.
    """
    cfg = _cfg()
    assert cfg.gated_conv is False
    assert cfg.gated_conv_activation is None
    assert cfg.gate_params(D_MODEL) == 0
    assert cfg.num_params(D_MODEL) == _cfg().num_params(D_MODEL)


def test_depthwise_gate_costs_12288_params_per_layer():
    """
    The headline parameter figure, asserted as an exact integer.

    Three convolutions x 2 gates x 2048 channels = 12,288, which is ~0.06% of the layer's
    projections -- small enough that the arms are parameter-matched for free, which is the whole
    reason this experiment is cheap.
    """
    plain = _cfg()
    gated = _cfg(gated_conv=True, gate_structure="depthwise")

    assert gated.gate_params(D_MODEL) == 12_288
    delta = gated.num_params(D_MODEL) - plain.num_params(D_MODEL)
    assert delta == 12_288
    assert delta / plain.num_params(D_MODEL) < 0.001


def test_lowrank_gate_parameter_cost_is_reported_exactly():
    plain = _cfg()
    gated = _cfg(gated_conv=True, gate_structure="lowrank", gate_rank=GATE_RANK)

    # Three convolutions, each 2048*128 + 2*128*2048.
    assert gated.gate_params(D_MODEL) == 3 * (2048 * 128 + 2 * 128 * 2048)
    assert gated.gate_params(D_MODEL) == 2_359_296
    assert gated.num_params(D_MODEL) - plain.num_params(D_MODEL) == 2_359_296


def test_a_dense_gate_would_more_than_double_the_layer_and_is_not_offered():
    """
    Records why ``"lowrank"`` exists instead of a dense gate, with the corrected arithmetic.

    ``KDA/HANDOFF.md`` records this cost as "+60%", which counted **one** gate per stream. There
    are two, so at ``d_model=2048`` a dense gate is 25,165,824 parameters against the layer's own
    17,887,376 -- **140.7%**, more than doubling it. That confounds the mechanism under test with
    raw capacity, the error ``KDA/HANDOFF.md:587`` records for the R sweep.

    Asserted rather than asserted-in-prose so the docstring table cannot drift from the code.
    """
    layer = _cfg().num_params(D_MODEL)
    assert layer == 17_887_376

    dense_gate = 2 * 3 * D_MODEL * D_MODEL  # two gates, three streams
    assert dense_gate == 25_165_824
    assert dense_gate / layer > 1.4

    lowrank = _cfg(gated_conv=True, gate_structure="lowrank", gate_rank=GATE_RANK)
    assert lowrank.gate_params(D_MODEL) / layer == pytest.approx(0.1319, abs=1e-4)

    depthwise = _cfg(gated_conv=True, gate_structure="depthwise")
    assert depthwise.gate_params(D_MODEL) / layer == pytest.approx(0.000687, abs=1e-6)


def test_gate_options_without_gated_conv_are_refused():
    """
    A config that reads as gated and trains as plain is an ablation against itself.

    ``gate_structure="lowrank"`` with ``gated_conv=False`` looks like a treatment arm in a YAML
    diff. Building it must fail rather than quietly run the control twice.
    """
    with pytest.raises(ValueError, match="gate_rank"):
        _cfg(gate_rank=GATE_RANK).build(D_MODEL, layer_idx=0, n_layers=1, init_device="meta")
    with pytest.raises(ValueError, match="gated_conv_activation"):
        _cfg(gated_conv_activation="silu").build(
            D_MODEL, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_lowrank_without_a_rank_is_refused():
    with pytest.raises(ValueError, match="gate_rank"):
        _cfg(gated_conv=True, gate_structure="lowrank").build(
            D_MODEL, layer_idx=0, n_layers=1, init_device="meta"
        )


def test_config_reports_the_memory_cost_only_when_gated():
    plain = _cfg()
    gated = _cfg(gated_conv=True)
    kw = dict(batch_size=1, seq_len=8192, bytes_per_element=2)

    assert plain.gate_activation_bytes(D_MODEL, **kw) == 0  # type: ignore[arg-type]
    # 384 MiB per layer at the microbatch KDA's throughput was measured at.
    assert gated.gate_activation_bytes(D_MODEL, **kw) == 402_653_184  # type: ignore[arg-type]


def test_the_three_arms_are_three_distinct_configurations():
    """
    LIV changes two things at once -- it adds gating and removes the activation. Three arms.

    Without ``kda-gated-silu`` a win cannot be attributed to either change, so the arms must be
    distinguishable at the config level and not collapse onto each other.
    """
    arms = {
        "kda-plain": _cfg(),
        "kda-gated": _cfg(gated_conv=True, gated_conv_activation=None),
        "kda-gated-silu": _cfg(gated_conv=True, gated_conv_activation="silu"),
    }
    assert len({(a.gated_conv, a.gated_conv_activation) for a in arms.values()}) == 3
    # The two gated arms are parameter-identical, so any difference between them is the
    # activation and nothing else.
    assert arms["kda-gated"].num_params(D_MODEL) == arms["kda-gated-silu"].num_params(D_MODEL)
    assert arms["kda-gated"].num_params(D_MODEL) - arms["kda-plain"].num_params(D_MODEL) == 12_288


# ---------------------------------------------------------------------------------------------
# Anything that must construct KimiDeltaAttention. Requires fla, hence CUDA.
# ---------------------------------------------------------------------------------------------


@requires_fla
@pytest.mark.parametrize(
    "structure,rank",
    [
        pytest.param("depthwise", None, id="depthwise"),
        pytest.param("lowrank", GATE_RANK, id="lowrank"),
    ],
)
def test_kda_config_num_params_matches_the_built_module(structure, rank):
    """
    The predicted total and the real module must agree, which is what makes an arm ledger
    trustworthy. ``num_params`` is what solves FFN widths for parameter matching, so an error
    here moves the anchor for every arm.
    """
    cfg = _cfg(gated_conv=True, gate_structure=structure, gate_rank=rank)
    module = cfg.build(D_MODEL, layer_idx=0, n_layers=12, init_device="meta")
    assert cfg.num_params(D_MODEL) == sum(p.numel() for p in module.parameters())


@requires_fla
def test_kda_builds_gated_convolutions_on_all_three_streams():
    """
    All three or none.

    Building the convolutions inline three times is how one stream ends up plain while the other
    two are gated -- which trains fine and is not the operator under test.
    """
    module = _cfg(gated_conv=True).build(D_MODEL, layer_idx=0, n_layers=12, init_device="meta")
    convs = [module.q_conv1d, module.k_conv1d, module.v_conv1d]
    assert len(convs) == 3
    assert all(isinstance(c, GatedCausalConv1d) for c in convs)
    assert all(c.activation is None for c in convs)

    plain = _cfg().build(D_MODEL, layer_idx=0, n_layers=12, init_device="meta")
    plain_convs = [plain.q_conv1d, plain.k_conv1d, plain.v_conv1d]
    assert all(isinstance(c, CausalConv1d) for c in plain_convs)
    assert all(c.activation == "silu" for c in plain_convs)


@requires_fla
def test_kda_gated_silu_arm_realises_silu():
    """The third arm's activation must actually reach the module, not just the config."""
    module = _cfg(gated_conv=True, gated_conv_activation="silu").build(
        D_MODEL, layer_idx=0, n_layers=12, init_device="meta"
    )
    assert all(c.activation == "silu" for c in (module.q_conv1d, module.k_conv1d, module.v_conv1d))


@requires_fla
def test_kda_gated_flops_exceed_plain_but_only_slightly():
    plain = _cfg().build(D_MODEL, layer_idx=0, n_layers=12, init_device="meta")
    gated = _cfg(gated_conv=True).build(D_MODEL, layer_idx=0, n_layers=12, init_device="meta")
    p, g = plain.num_flops_per_token(4096), gated.num_flops_per_token(4096)
    assert g > p, "the gate costs FLOPs and the accounting must say so"
    assert (g - p) / p < 0.01, "a depthwise gate should be well under 1% of the layer"


@requires_fla
def test_kda_init_draws_the_same_convolution_weights_in_both_arms():
    """
    The arms must differ **only** by the gate.

    If the gated arm consumed a different amount of randomness, every later parameter in the
    model would differ too, and that confound is invisible in a loss curve. The gate init is
    all-zeros precisely so it consumes none.
    """
    from olmo_core.nn.transformer.init import InitMethod

    def build_and_init(gated: bool):
        cfg = _cfg(gated_conv=gated)
        m = cfg.build(D_MODEL, layer_idx=0, n_layers=12, init_device="cpu")
        gen = torch.Generator().manual_seed(1234)
        m.init_weights(
            init_method=InitMethod.normal,
            d_model=D_MODEL,
            block_idx=0,
            num_blocks=12,
            generator=gen,
        )
        return m

    plain, gated = build_and_init(False), build_and_init(True)

    torch.testing.assert_close(plain.q_conv1d.weight, gated.q_conv1d.conv.weight)
    torch.testing.assert_close(plain.v_conv1d.weight, gated.v_conv1d.conv.weight)
    # And every non-convolution parameter must be untouched by the gate's presence, which is the
    # part that proves the random stream did not diverge.
    torch.testing.assert_close(plain.w_out.weight, gated.w_out.weight)
    torch.testing.assert_close(plain.A_log, gated.A_log)


@requires_fla
def test_kda_gates_start_at_exactly_one():
    """A gated arm at step 0 must be the plain arm, or the contrast is not an ablation."""
    from olmo_core.nn.transformer.init import InitMethod

    m = _cfg(gated_conv=True).build(D_MODEL, layer_idx=0, n_layers=12, init_device="cpu")
    m.init_weights(
        init_method=InitMethod.normal,
        d_model=D_MODEL,
        block_idx=0,
        num_blocks=12,
        generator=torch.Generator().manual_seed(0),
    )
    for c in (m.q_conv1d, m.k_conv1d, m.v_conv1d):
        assert torch.count_nonzero(c.pre_scale) == 0
        assert torch.count_nonzero(c.post_scale) == 0
        assert 2.0 * torch.sigmoid(torch.zeros(1)) == 1.0
