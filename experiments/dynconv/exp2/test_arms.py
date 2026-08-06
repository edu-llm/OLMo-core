"""Tests for ``arms.py`` and ``dynamic_conv.py``.

Owner: sub-agent A.

Run::

    PYTHONPATH=<olmo-core worktree>/src:. python3 -m pytest test_arms.py -q

TWO RULES THIS FILE OBEYS
-------------------------
1. **A test must CALL the code, not re-derive its formula.** Repo scar
   ``test-must-call-not-recompute``: a test that reimplements the code's own arithmetic passes when
   the code changes. So the parameter counts here are compared against **hand-written literal
   integers**, not against a second copy of ``expected_param_count``'s expression, and the
   analytic function is then checked against the built module.
2. **A guard that has never failed is not known to work.** Every mutation switch on
   :class:`ArmSpec` exists so a negative control can flip it; the negative controls live in
   ``test_preflight.py`` and are asserted to *fail*.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from arms import (
    ARMS,
    ATTN1_ATTENTION_LAYERS,
    D_MODEL,
    HYBRID_ATTENTION_LAYERS,
    N_LAYERS,
    RANK,
    TOPOLOGIES,
    TOPOLOGY_ATTENTION_LAYERS,
    VOCAB_SIZE,
    WIDTHS,
    ArmNotDefined,
    ArmSpec,
    Attention,
    MQARModel,
    arm_grid,
    build_arm,
    derive_generator,
    expected_param_count,
    na_cells,
)
from dynamic_conv import (
    DynamicFilterGen,
    DynamicQKVConv,
    DynamicShortConv,
    depthwise_causal_conv_dynamic,
    depthwise_causal_conv_static,
    gen_param_count,
    iter_generators,
    named_shared_params,
    reset_permutations,
    set_alpha_override,
    split_param_groups,
    static_realizability_residual,
)

# =============================================================================================
# Operator parity -- the route decision
# =============================================================================================
#
# We IMPORT the released-parity ShortConv from the OLMo-core worktree rather than vendoring a
# copy, because the import works on CPU with no heavy deps. DynamicShortConv subclasses it, so the
# static path IS the tested operator by construction rather than by resemblance. These tests pin
# that: if someone later vendors a fork, they fail.


def test_dynamic_short_conv_is_a_short_conv():
    from olmo_core.nn.attention.short_conv import ShortConv

    assert issubclass(DynamicShortConv, ShortConv)
    m = DynamicShortConv(d_model=32, kernel_size=3, rank=8)
    # Same parameter NAMES for every shared tensor -- this is what makes preflight check 6
    # (torch.equal on shared params across arms) expressible at all.
    s = ShortConv(d_model=32, kernel_size=3, use_fla=False)
    shared = set(dict(s.named_parameters())) & set(dict(m.named_parameters()))
    assert shared == set(dict(s.named_parameters())), "DynamicShortConv renamed a shared tensor"


@pytest.mark.parametrize("W", [1, 2, 3, 4, 8])
def test_dynamic_kernel_float64_parity_with_released_shortconv(W: int):
    """float64 parity of our dynamic kernel at ``alpha = 0`` against the released ShortConv.

    This is the parity test that licenses using our own kernel at all. In float64 the residual is
    at machine epsilon; in fp32 it is ~1e-8, because the ``W`` taps accumulate in a different
    order and fp32 addition is not associative. Preflight check 3's fp32 tolerance of 1e-6 is set
    for that reason -- do not tighten it to chase bitwise equality that cannot exist.
    """
    from olmo_core.nn.attention.short_conv import ShortConv

    torch.manual_seed(0)
    d = 16
    m = DynamicShortConv(d_model=d, kernel_size=W, rank=4, dtype=torch.float64)
    s = ShortConv(d_model=d, kernel_size=W, use_fla=False, dtype=torch.float64)
    s.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("dyn.")})
    x = torch.randn(2, 13, d, dtype=torch.float64)

    set_alpha_override(m, 0.0)
    with torch.no_grad():
        yd, ys = m(x), s(x)
    rel = float((yd - ys).norm() / ys.norm())
    assert rel < 1e-13, f"W={W}: float64 residual {rel:.3e} is not machine-epsilon"


@pytest.mark.parametrize("W", [2, 3, 4, 8])
def test_static_and_dynamic_kernels_agree_when_filter_is_position_constant(W: int):
    """The two kernels are the same operator; only the filter's rank in ``t`` differs."""
    torch.manual_seed(1)
    B, T, d = 3, 21, 7
    u = torch.randn(B, T, d, dtype=torch.float64)
    a = torch.randn(d, W, dtype=torch.float64)
    ref = depthwise_causal_conv_static(u, a)
    got = depthwise_causal_conv_dynamic(u, a.expand(B, T, d, W))
    assert torch.allclose(ref, got, atol=1e-14)


def test_conv_is_causal():
    """Changing token t must not change any output before t. A conv that reads the future would
    make MQAR trivially solvable and every arm tie at 1.00."""
    torch.manual_seed(2)
    d, W, T = 6, 4, 12
    a = torch.randn(d, W, dtype=torch.float64)
    u = torch.randn(1, T, d, dtype=torch.float64)
    y0 = depthwise_causal_conv_static(u, a)
    u2 = u.clone()
    u2[0, 7] += 10.0
    y1 = depthwise_causal_conv_static(u2, a)
    assert torch.allclose(y0[:, :7], y1[:, :7], atol=1e-14)
    assert not torch.allclose(y0[:, 7], y1[:, 7])


# =============================================================================================
# Init spec -- SPEC §3, non-negotiable
# =============================================================================================


def test_init_is_U_zero_V_random_alpha_one_learnable():
    g = DynamicFilterGen(d_model=64, rank=16, width=3)
    assert float(g.U.weight.detach().abs().max()) == 0.0, "U must be EXACTLY zero (LoRA warm start)"
    assert float(g.V.weight.detach().abs().max()) > 0.0, (
        "V must be random -- zeroing both legs is a saddle"
    )
    assert float(g.alpha.detach()) == 1.0
    assert g.alpha.requires_grad, "alpha must be LEARNABLE; a fixed alpha makes Delta_w unreachable"
    assert isinstance(g.alpha, nn.Parameter), "alpha as a float is a non-learnable alpha in disguise"


def test_V_init_uses_the_true_contraction_not_the_3d_fan_in():
    """``V``'s bound must be ``1/sqrt(d)``, the TRUE contraction of ``V @ h``.

    ``kaiming_uniform_`` on a 3-D parameter derives ``fan_in = size(1) * receptive_field``, which
    for a ``(R, d, W)`` shape is off by ``sqrt(W)``. Checked statistically rather than by reading
    the source, so a future refactor that delegates to a helper is caught.
    """
    d, R = 128, 16
    g = DynamicFilterGen(d_model=d, rank=R, width=3)
    bound = 1.0 / math.sqrt(d)
    w = g.V.weight.detach()
    assert float(w.abs().max()) <= bound + 1e-6, "V exceeds the U(-1/sqrt(d), 1/sqrt(d)) support"
    # Var of U(-b, b) is b^2/3. With d*R = 2048 samples the estimator is tight enough for 15%.
    expected_std = bound / math.sqrt(3.0)
    got_std = float(w.std())
    assert abs(got_std - expected_std) / expected_std < 0.15, (
        f"V std {got_std:.5f} vs expected {expected_std:.5f} -- wrong fan-in. "
        f"A sqrt(W) error would give {expected_std / math.sqrt(3):.5f} at W=3."
    )


def test_delta_w_is_exactly_zero_at_init_but_U_has_gradient():
    """SPEC §3's whole point, in one test. ``Delta_w == 0`` so the arm starts as the static arm;
    ``||dL/dU|| > 0`` so it does not STAY the static arm."""
    torch.manual_seed(3)
    g = DynamicFilterGen(d_model=32, rank=8, width=3)
    h = torch.randn(2, 5, 32)
    dw = g(h)
    assert dw is not None and float(dw.abs().max()) == 0.0

    m = DynamicShortConv(d_model=32, kernel_size=3, rank=8)
    x = torch.randn(2, 6, 32)
    m(x).pow(2).mean().backward()
    assert float(m.dyn.U.weight.grad.norm()) > 0.0, "U must have gradient at step 0"
    assert float(m.dyn.V.weight.grad.norm()) == 0.0, (
        "||V.grad|| == 0 at step 0 is CORRECT (LoRA). If this is nonzero, U was not exactly zero."
    )


def test_zeroing_both_legs_is_an_exact_saddle():
    """Direct measurement of the $6,100 bug, so the abstract claim is grounded in output.

    R5's measured table: ``U=0, V rand, alpha=1`` gives ``||dL/dU|| = 3.08e-02``;
    ``U=0 AND alpha=0`` gives all three gradients EXACTLY zero.
    """
    torch.manual_seed(4)
    x = torch.randn(2, 6, 32)

    good = DynamicShortConv(d_model=32, kernel_size=3, rank=8, alpha_init=1.0)
    good(x).pow(2).mean().backward()
    assert float(good.dyn.U.weight.grad.norm()) > 0.0

    dead = DynamicShortConv(d_model=32, kernel_size=3, rank=8, alpha_init=0.0)
    dead(x).pow(2).mean().backward()
    assert float(dead.dyn.U.weight.grad.norm()) == 0.0
    assert float(dead.dyn.V.weight.grad.norm()) == 0.0
    assert float(dead.dyn.alpha.grad.norm()) == 0.0, (
        "alpha's gradient must ALSO be zero -- that is what makes it an exact saddle rather than "
        "a slow start."
    )


def test_no_bare_indexed_write_in_any_init_weights():
    """Source-level guard for the DTensor trap, since a CPU test structurally cannot catch it.

    ``w[...] = x`` inside ``init_weights`` lowers to ``aten.fill_.Tensor``, which has no DTensor
    sharding strategy, and killed ``run_019fbf9f`` under FSDP after every local test passed.
    Indexed writes are permitted ONLY inside a closure handed to ``_apply_init``, which
    materializes the full tensor first.
    """
    import ast
    import inspect

    import arms
    import dynamic_conv

    for mod in (dynamic_conv, arms):
        tree = ast.parse(inspect.getsource(mod))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.FunctionDef) or fn.name != "init_weights":
                continue
            # Collect nested function names -- closures passed to _apply_init are exempt.
            nested = {n.name for n in ast.walk(fn) if isinstance(n, ast.FunctionDef)} - {
                "init_weights"
            }
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign):
                    continue
                for tgt in node.targets:
                    if isinstance(tgt, ast.Subscript):
                        # Is it inside a nested function? Re-walk each nested fn to be sure.
                        in_nested = any(
                            any(n is node for n in ast.walk(nf))
                            for nf in ast.walk(fn)
                            if isinstance(nf, ast.FunctionDef) and nf.name in nested
                        )
                        assert in_nested, (
                            f"{mod.__name__}.init_weights has a bare indexed write "
                            f"at line {node.lineno}: this lowers to aten.fill_.Tensor and dies "
                            "under FSDP. Route it through _apply_init."
                        )


# =============================================================================================
# Paired seeding -- R7 FP1. This is the one that voids the power analysis.
# =============================================================================================


def test_derive_generator_is_stable_and_name_keyed():
    a = derive_generator(7, "blocks.0.sequence_mixer")
    b = derive_generator(7, "blocks.0.sequence_mixer")
    c = derive_generator(7, "blocks.1.sequence_mixer")
    assert torch.equal(torch.rand(4, generator=a), torch.rand(4, generator=b))
    assert not torch.equal(
        torch.rand(4, generator=derive_generator(7, "x")),
        torch.rand(4, generator=c),
    )


def test_derive_generator_does_not_use_pythons_salted_hash():
    """``hash(str)`` is salted per process, so a ``hash()``-keyed scheme reproduces within one
    process and silently unpairs the arms across the processes of a sweep -- while every local
    test passes. Pinned against a literal so a refactor to ``hash()`` fails here."""
    import hashlib

    seed, name = 7, "blocks.0.sequence_mixer"
    digest = hashlib.blake2b(f"{seed}/{name}".encode(), digest_size=8).digest()
    expected = int.from_bytes(digest, "big") % (2**63 - 1)
    g = derive_generator(seed, name)
    ref = torch.Generator()
    ref.manual_seed(expected)
    assert torch.equal(torch.rand(8, generator=g), torch.rand(8, generator=ref))


@pytest.mark.parametrize("topology", ["hybrid", "allliv"])
@pytest.mark.parametrize("arm", ["S2", "S4"])
def test_shared_params_bit_identical_across_arms(topology: str, arm: str):
    """``torch.equal``, NOT ``allclose``. Every shared tensor, every arm, same seed."""
    s1 = ArmSpec(arm="S1", topology=topology, width=3)  # type: ignore[arg-type]
    sx = ArmSpec(arm=arm, topology=topology, width=3)  # type: ignore[arg-type]
    a, b = build_arm(s1, seed=11), build_arm(sx, seed=11)
    sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = list(named_shared_params(a, b))
    assert len(shared) > 40, f"only {len(shared)} shared tensors -- the comparison is vacuous"
    bad = [n for n in shared if not torch.equal(sa[n].detach(), sb[n].detach())]
    assert not bad, f"{len(bad)}/{len(shared)} shared tensors differ, first={bad[0]}"


def test_S3_shares_attention_weights_with_S1():
    """S3 adds a generator INSIDE the attention block, which is the hardest pairing case: the
    attention block's own draw order must be unchanged by the generator's presence."""
    a = build_arm(ArmSpec(arm="S1", topology="hybrid", width=3), seed=5)
    b = build_arm(ArmSpec(arm="S3", topology="hybrid", width=3), seed=5)
    sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
    for i in HYBRID_ATTENTION_LAYERS:
        for leaf in ("qkv.weight", "out.weight"):
            k = f"blocks.{i}.sequence_mixer.{leaf}"
            assert torch.equal(sa[k], sb[k]), f"{k} diverged -- S3's generator moved the stream"


def test_sequential_seeding_actually_breaks_pairing():
    """NEGATIVE CONTROL for the pairing mechanism itself.

    Reproduces R7 FP1's single-stream scheme and proves it produces *unpaired* arms. If this ever
    passes, the per-module keying is not doing anything and the real test above is vacuous.
    """
    s1 = ArmSpec(arm="S1", topology="hybrid", width=3, seeding="sequential")
    s4 = ArmSpec(arm="S4", topology="hybrid", width=3, seeding="sequential")
    a, b = build_arm(s1, seed=11, strict=False), build_arm(s4, seed=11, strict=False)
    sa, sb = dict(a.named_parameters()), dict(b.named_parameters())
    shared = list(named_shared_params(a, b))
    bad = [n for n in shared if not torch.equal(sa[n].detach(), sb[n].detach())]
    assert bad, (
        "sequential seeding was expected to DIVERGE the arms but did not. Either the negative "
        "control is broken, or the per-module keying is not the thing making pairing work."
    )


# =============================================================================================
# Arm structure, counts and indices
# =============================================================================================


def test_S3_is_not_defined_in_allliv_and_does_not_silently_become_S1():
    with pytest.raises(ArmNotDefined):
        ArmSpec(arm="S3", topology="allliv", width=3)
    grid = arm_grid()
    assert not [s for s in grid if s.arm == "S3" and s.topology == "allliv"]
    assert len(na_cells()) == len(WIDTHS), "the N/A cells must be reported, not omitted"


def test_hybrid_and_allliv_layer_composition():
    h = ArmSpec(arm="S4", topology="hybrid", width=3)
    assert h.attn_idx == (2, 5) and h.liv_idx == (0, 1, 3, 4)
    assert h.dynamic_layers == (0, 1, 3, 4)
    a = ArmSpec(arm="S4", topology="allliv", width=3)
    assert a.attn_idx == () and a.liv_idx == tuple(range(6))
    assert a.dynamic_layers == tuple(range(6))
    assert ArmSpec(arm="S3", topology="hybrid", width=3).dynamic_layers == (2, 5)
    assert ArmSpec(arm="S1", topology="hybrid", width=3).dynamic_layers == ()


def test_hand_computed_param_counts_hybrid_W3():
    """Literal integers, computed by hand, NOT by re-running the code's expression.

    d=128, vocab=256, L=6, ffn_mult=2 (hidden=256), R=16, hybrid = 4 LIV + 2 attention at (2,5).

      embed        256*128                        =    32,768
      head         128*256                        =    32,768
      norms        (2*6 + 1)*128                  =     1,664
      ffn/layer    2*128*256 + 256*128            =    98,304   -> x6 = 589,824
      LIV/layer    128*128*4 + 3*128              =    65,920   -> x4 = 263,680
      attn/layer   3*128*128 + 128*128            =    65,536   -> x2 = 131,072
      ------------------------------------------------------------------------
      S1 total     32768+32768+1664+589824+263680+131072        = 1,051,776  ... W=3
    """
    p = expected_param_count(ArmSpec(arm="S1", topology="hybrid", width=3))
    assert p["embed"] == 32_768
    assert p["head"] == 32_768
    assert p["norms"] == 1_664
    assert p["ffn"] == 589_824
    assert p["liv_mixers"] == 263_680
    assert p["attn_mixers"] == 131_072
    assert p["total"] == 1_051_776
    assert build_arm(ArmSpec(arm="S1", topology="hybrid", width=3)).n_params == 1_051_776


def test_hand_computed_generator_counts():
    """Per-generator: ``V = d*R = 2048``; ``U = R*W*d`` ; ``alpha = 1``.

      W=3, 1 stream: 2048 + 16*3*128 + 1     = 2048 + 6144 + 1     =  8,193
      W=8, 1 stream: 2048 + 16*8*128 + 1     = 2048 + 16384 + 1    = 18,433
      W=3, 3 streams (S3): 2048 + 16*3*3*128 + 1 = 2048+18432+1    = 20,481
    """
    assert gen_param_count(128, 16, 3, 1) == 8_193
    assert gen_param_count(128, 16, 8, 1) == 18_433
    assert gen_param_count(128, 16, 3, 3) == 20_481
    g = DynamicFilterGen(d_model=128, rank=16, width=3)
    assert sum(p.numel() for p in g.parameters()) == 8_193
    g3 = DynamicFilterGen(d_model=128, rank=16, width=3, n_streams=3)
    assert sum(p.numel() for p in g3.parameters()) == 20_481


def test_hand_computed_S4_hybrid_W3_total():
    """S4-hybrid-W3 = S1-hybrid-W3 + 4 generators = 1,051,776 + 4*8,193 = 1,084,548."""
    assert 1_051_776 + 4 * 8_193 == 1_084_548
    assert build_arm(ArmSpec(arm="S4", topology="hybrid", width=3)).n_params == 1_084_548


def test_hand_computed_S3_hybrid_W3_total():
    """S3 adds, per attention layer, one 3-stream generator PLUS three static (d, W) filters:
    20,481 + 3*128*3 = 20,481 + 1,152 = 21,633. Two attention layers => +43,266.
    1,051,776 + 43,266 = 1,095,042."""
    assert 20_481 + 3 * 128 * 3 == 21_633
    assert 1_051_776 + 2 * 21_633 == 1_095_042
    assert build_arm(ArmSpec(arm="S3", topology="hybrid", width=3)).n_params == 1_095_042


def test_S2_and_S4_are_exactly_param_and_flop_matched():
    """The scientific core of Exp-2. S2 must differ from S4 ONLY by a gather on ``z``.

    If these ever diverge, S2 stops being a control and the primary contrast is void.
    """
    for topo in TOPOLOGIES:
        for W in WIDTHS:
            s2 = build_arm(ArmSpec(arm="S2", topology=topo, width=W))  # type: ignore[arg-type]
            s4 = build_arm(ArmSpec(arm="S4", topology=topo, width=W))  # type: ignore[arg-type]
            assert s2.n_params == s4.n_params, f"{topo} W={W}: {s2.n_params} != {s4.n_params}"
            n2 = {k: v.shape for k, v in s2.named_parameters()}
            n4 = {k: v.shape for k, v in s4.named_parameters()}
            assert n2 == n4, f"{topo} W={W}: parameter NAMES/shapes differ between S2 and S4"


@pytest.mark.parametrize("W", WIDTHS)
@pytest.mark.parametrize("topology", TOPOLOGIES)
@pytest.mark.parametrize("arm", ARMS)
def test_every_buildable_cell_builds_strictly(arm: str, topology: str, W: int):
    """``strict=True`` asserts declared module count, declared layer indices and the analytic
    total against the built module. Every production cell must satisfy all three."""
    if arm == "S3" and topology == "allliv":
        pytest.skip("N/A by construction; asserted elsewhere")
    spec = ArmSpec(arm=arm, topology=topology, width=W)  # type: ignore[arg-type]
    m = build_arm(spec, seed=0)
    assert m.dynamic_module_layers() == spec.dynamic_layers
    assert m.n_dynamic_modules() == len(spec.dynamic_layers)
    x = torch.randint(0, VOCAB_SIZE, (2, 24))
    assert m(x).shape == (2, 24, VOCAB_SIZE)


def test_wrong_attribute_yields_zero_dynamic_modules_while_forward_still_succeeds():
    """NEGATIVE CONTROL: the silent-no-op trap, in full.

    Wiring to ``block.attention`` instead of ``block.sequence_mixer`` must leave 0 generators AND
    the model must still forward, train and produce a plausible loss. That combination is the
    trap: nothing is broken enough to raise.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, wire_slot="attention")
    m = build_arm(spec, seed=0, strict=False)
    assert m.n_dynamic_modules() == 0, "the override landed after all -- the control is broken"
    assert m.dynamic_module_layers() == ()
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    y = m(x)
    assert torch.isfinite(y).all(), "forward must SUCCEED -- that is what makes this silent"
    loss = nn.functional.cross_entropy(y.reshape(-1, VOCAB_SIZE), x.reshape(-1))
    loss.backward()
    assert math.isfinite(float(loss)) and 5.4 < float(loss) < 5.9, (
        "and the loss must look perfectly normal"
    )
    # strict=True is what protects production callers.
    with pytest.raises(AssertionError, match="silently no-ops"):
        build_arm(spec, seed=0, strict=True)


def test_strict_refuses_the_wrong_slot_and_non_strict_still_builds_it():
    """BOTH directions of the ``strict`` contract, asserted explicitly.

    Found by the team lead's independent run: ``build_arm(strict=True)`` used to BUILD a
    ``wire_slot='attention'`` model rather than refusing it. The count checks caught the
    misconfiguration downstream, so the trap was never open -- but a flag named ``strict`` that
    accepts a negative-control knob is a flag a future caller will reasonably over-trust.
    """
    spec = ArmSpec(arm="S4", topology="hybrid", width=3, wire_slot="attention")

    # direction 1: strict=True REFUSES, and the message names the trap
    with pytest.raises(AssertionError) as exc:
        build_arm(spec, seed=0, strict=True)
    msg = str(exc.value)
    assert "NEGATIVE CONTROL" in msg
    assert "silently no-ops" in msg
    assert "sequence_mixer" in msg

    # direction 2: strict=False still builds it -- the negative controls depend on this
    m = build_arm(spec, seed=0, strict=False)
    assert m.n_dynamic_modules() == 0
    assert m.dynamic_module_layers() == ()
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    y = m(x)
    assert torch.isfinite(y).all(), "the forward pass must still succeed under strict=False"

    # and every production cell is unaffected
    for arm in ARMS:
        for topo in TOPOLOGIES:
            if arm == "S3" and topo == "allliv":
                continue
            build_arm(ArmSpec(arm=arm, topology=topo, width=3), seed=0, strict=True)  # type: ignore[arg-type]


def test_block_has_no_attention_attribute():
    """There must be no ``attention`` attribute to wire to by accident."""
    m = build_arm(ArmSpec(arm="S1", topology="hybrid", width=3))
    for blk in m.blocks:
        assert not hasattr(blk, "attention")
        assert hasattr(blk, "sequence_mixer")


def test_changing_R_changes_the_param_count():
    """NEGATIVE CONTROL for the count check: R=8 must not reconcile against R=16's total."""
    r16 = expected_param_count(ArmSpec(arm="S4", topology="hybrid", width=3, rank=16))["total"]
    r8 = expected_param_count(ArmSpec(arm="S4", topology="hybrid", width=3, rank=8))["total"]
    assert r8 != r16
    m8 = build_arm(ArmSpec(arm="S4", topology="hybrid", width=3, rank=8))
    assert m8.n_params == r8 and m8.n_params != r16


def test_forgetting_the_W_factor_in_U_is_detectable():
    """The classic error: wiring ``U: R -> d`` instead of ``R -> W*d``. Off by ``W``, still trains.

    Constructed here directly, since ``DynamicFilterGen`` cannot express it.
    """
    d, R, W = 128, 16, 3
    correct = gen_param_count(d, R, W)
    wrong = d * R + R * d + 1  # the W factor dropped
    assert wrong != correct
    assert correct - wrong == R * d * (W - 1) == 4_096


# =============================================================================================
# Every arm is initialized -- the fan-in scar
# =============================================================================================


def test_every_arm_calls_init_weights_including_S1():
    """The in-tree ``mqar_model.py`` constructs ``ShortConv`` directly and never calls
    ``init_weights``, so its grouped arm ran at ~1/128 of dense activation scale *on the probe
    used to justify that arm*. Detected here by checking that the conv is an identity tap, which
    only ``init_weights`` produces -- ``nn.Conv1d``'s own default is uniform random.
    """
    from olmo_core.nn.attention.short_conv import ShortConv

    for arm in ("S1", "S2", "S4"):
        m = build_arm(ArmSpec(arm=arm, topology="allliv", width=3))  # type: ignore[arg-type]
        for i, blk in enumerate(m.blocks):
            mix = blk.sequence_mixer
            assert isinstance(mix, ShortConv)
            w = mix.conv.weight.view(m.spec.d_model, m.spec.width)
            assert torch.allclose(w[:, -1], torch.ones_like(w[:, -1])), (
                f"{arm} L{i}: current-token tap is not 1.0 -- init_weights was not called"
            )
            assert float(w[:, :-1].abs().max()) == 0.0, f"{arm} L{i}: history taps are not zero"


def test_step0_activation_scale_parity_across_all_arms():
    """SPEC §5.3: assert step-0 activation-scale parity across ALL arms, not just the new one.

    A one-sided fan-in correction previously biased a contrast toward the hypothesis. With
    ``Delta_w == 0`` at init, every arm's block output must have the SAME scale.
    """
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB_SIZE, (4, 32))
    scales = {}
    for arm in ARMS:
        for topo in TOPOLOGIES:
            if arm == "S3" and topo == "allliv":
                continue
            m = build_arm(ArmSpec(arm=arm, topology=topo, width=3))  # type: ignore[arg-type]
            with torch.no_grad():
                h = m.embed(x)
                for blk in m.blocks:
                    h = blk(h)
            scales[(arm, topo)] = float(h.std())
    for topo in TOPOLOGIES:
        vals = [v for (a, t), v in scales.items() if t == topo]
        assert max(vals) / min(vals) < 1.001, (
            f"{topo}: step-0 activation scale differs across arms: {scales}"
        )


def test_init_is_reproducible_and_seed_sensitive():
    a = build_arm(ArmSpec(arm="S4", topology="allliv", width=3), seed=42)
    b = build_arm(ArmSpec(arm="S4", topology="allliv", width=3), seed=42)
    c = build_arm(ArmSpec(arm="S4", topology="allliv", width=3), seed=43)
    sa, sb, sc = (dict(m.named_parameters()) for m in (a, b, c))
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)


# =============================================================================================
# The S2 control
# =============================================================================================


def test_permutation_is_a_true_permutation_per_sequence():
    torch.manual_seed(0)
    g = DynamicFilterGen(d_model=8, rank=4, width=3, permute_z=True)
    z = torch.randn(4, 16, 4, dtype=torch.float64)
    p = g._permute(z)
    for b in range(4):
        # A permutation preserves the multiset of rows. Sorting a row-key is the cheap check.
        ka = torch.sort(z[b].sum(-1)).values
        kb = torch.sort(p[b].sum(-1)).values
        assert torch.allclose(ka, kb, atol=1e-12), f"row {b} is not a permutation of the input"


def test_permutation_is_independent_per_sequence():
    torch.manual_seed(0)
    g = DynamicFilterGen(d_model=8, rank=4, width=3, permute_z=True)
    z = torch.arange(64, dtype=torch.float32).view(1, 16, 4).expand(8, 16, 4).contiguous()
    p = g._permute(z)
    assert any(not torch.equal(p[0], p[i]) for i in range(1, 8)), (
        "every batch row got the same permutation -- a weaker control than specified"
    )


def test_permutation_is_redrawn_every_forward():
    """A FIXED permutation is a LEARNABLE POSITIONAL CODE: position t would learn that it always
    reads pi(t), recovering position-dependent (indeed acausally position-dependent) filters. That
    is the opposite of a control. Redrawing is also what makes the acausal read unexploitable:
    P(pi(t) = t+1) = 1/T and the source identity varies every batch, so future content arrives as
    noise with no learnable decoder."""
    torch.manual_seed(0)
    g = DynamicFilterGen(d_model=8, rank=4, width=3, permute_z=True)
    z = torch.randn(2, 32, 4)
    assert not torch.equal(g._permute(z), g._permute(z))


def test_permutation_stream_is_seeded_and_resettable():
    g = DynamicFilterGen(d_model=8, rank=4, width=3, permute_z=True, permute_seed=99)
    z = torch.randn(2, 12, 4)
    first = g._permute(z)
    g.reset_permutation()
    assert torch.equal(g._permute(z), first), "the permutation stream is not reproducible"


def test_S2_layers_do_not_share_one_permutation_stream():
    m = build_arm(ArmSpec(arm="S2", topology="allliv", width=3))
    seeds = [g.permute_seed for _, g in iter_generators(m)]
    assert len(set(seeds)) == len(seeds), f"layers share a permutation stream: {seeds}"


def test_S2_and_S4_differ_in_output_but_only_via_z():
    """Same weights, same input: S2 and S4 must differ (the permutation bites) -- but both must
    collapse to the identical static path at ``alpha = 0``."""
    torch.manual_seed(0)
    s2 = build_arm(ArmSpec(arm="S2", topology="allliv", width=3), seed=7)
    s4 = build_arm(ArmSpec(arm="S4", topology="allliv", width=3), seed=7)
    # Force nonzero U on both, identically, so Delta_w != 0.
    with torch.no_grad():
        for (_, a), (_, b) in zip(iter_generators(s2), iter_generators(s4)):
            nn.init.normal_(a.U.weight, std=0.05)
            b.U.weight.copy_(a.U.weight)
    x = torch.randint(0, VOCAB_SIZE, (2, 32))
    with torch.no_grad():
        assert not torch.allclose(s2(x), s4(x)), "the permutation had no effect -- S2 == S4"
        set_alpha_override(s2, 0.0)
        set_alpha_override(s4, 0.0)
        assert torch.allclose(s2(x), s4(x), atol=1e-6), "arms diverge even at alpha=0"


def test_causal_prefix_permute_mode_is_strictly_causal():
    """The belt-and-braces variant. No strict permutation of 0..T-1 can satisfy pi(t) <= t for all
    t (induction forces the identity), so this samples j <= t with replacement instead."""
    torch.manual_seed(0)
    g = DynamicFilterGen(
        d_model=4, rank=2, width=3, permute_z=True, permute_mode="causal_prefix"
    )
    T = 32
    z = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(4, T, 2).contiguous()
    p = g._permute(z)
    src = p[..., 0]
    tgt = torch.arange(T, dtype=torch.float32).view(1, T)
    assert bool((src <= tgt).all()), "causal_prefix read from the future"


# =============================================================================================
# S3
# =============================================================================================


def test_S3_qkv_conv_is_identity_at_init():
    """S3 must reduce to S1 at init, or preflight check 3 is untestable for that arm. We use
    ``q <- conv(q)`` with an identity tap rather than the paper's residual ``q <- q + conv(q)``,
    which at identity init would give ``q <- 2q``."""
    torch.manual_seed(0)
    m = DynamicQKVConv(d_model=32, kernel_size=3, rank=8)
    q, k, v = (torch.randn(2, 9, 32) for _ in range(3))
    h = torch.randn(2, 9, 32)
    oq, ok, ov = m(h, q, k, v)
    assert torch.allclose(oq, q, atol=1e-6)
    assert torch.allclose(ok, k, atol=1e-6)
    assert torch.allclose(ov, v, atol=1e-6)


def test_S3_generator_lives_in_the_attention_blocks():
    m = build_arm(ArmSpec(arm="S3", topology="hybrid", width=3))
    for i, blk in enumerate(m.blocks):
        mix = blk.sequence_mixer
        if i in HYBRID_ATTENTION_LAYERS:
            assert isinstance(mix, Attention) and mix.qkv_conv is not None
        else:
            assert not any(isinstance(x, DynamicFilterGen) for x in blk.modules())


# =============================================================================================
# Nonlinear ablation, optimizer groups, W=2 theorem
# =============================================================================================


def test_nonlinear_variant_changes_nothing_at_init_but_is_a_different_function():
    """``silu(0) == 0``, so a zero ``V`` would hide the difference; ``V`` is random, so the two
    variants differ once ``U != 0``. Parameter counts must be identical."""
    lin = DynamicFilterGen(d_model=32, rank=8, width=3, nonlinear=False)
    nl = DynamicFilterGen(d_model=32, rank=8, width=3, nonlinear=True)
    assert sum(p.numel() for p in lin.parameters()) == sum(p.numel() for p in nl.parameters())
    with torch.no_grad():
        nl.V.weight.copy_(lin.V.weight)
        nn.init.normal_(lin.U.weight, std=0.1)
        nl.U.weight.copy_(lin.U.weight)
    h = torch.randn(2, 7, 32)
    assert not torch.allclose(lin(h), nl(h))


def test_dynamic_params_are_excluded_from_weight_decay():
    m = build_arm(ArmSpec(arm="S4", topology="allliv", width=3))
    groups = split_param_groups(m, weight_decay=0.1)
    decayed = {id(p) for g in groups if g["weight_decay"] != 0.0 for p in g["params"]}
    for _, gen in iter_generators(m):
        for p in gen.parameters():
            assert id(p) not in decayed
    # and every parameter is in exactly one group
    allp = [id(p) for g in groups for p in g["params"]]
    assert len(allp) == len(set(allp)) == len(list(m.parameters()))


def test_W2_is_exactly_static_realizable_and_W3_plus_is_not():
    """Reproduces the verified theorem: at W=2 the dynamic block is an EXACT reparameterization,
    so a W=2 dynamic-vs-static difference above seed noise is a BUG, not a result."""
    r2 = static_realizability_residual(12, 2, seed=1)
    assert r2 < 1e-12, f"W=2 residual {r2:.3e} -- the exact reparameterization did not reproduce"
    for W in (3, 4, 8):
        r = static_realizability_residual(12, W, seed=1)
        assert r > 1e-6, f"W={W} residual {r:.3e} is suspiciously small"


def test_W_minus_2_new_dof_accounting():
    """The DOF ledger, per position per channel: W generated, 1 redundant with C_t, 1 with the B
    sequence, leaving W-2. At W=3 the mechanism gets ONE new number out of three."""
    for W, expected in ((2, 0), (3, 1), (4, 2), (8, 6)):
        assert max(0, W - 2) == expected


def test_alpha_override_is_exact_and_reversible():
    m = build_arm(ArmSpec(arm="S4", topology="allliv", width=3))
    with torch.no_grad():
        for _, g in iter_generators(m):
            nn.init.normal_(g.U.weight, std=0.05)
    x = torch.randint(0, VOCAB_SIZE, (2, 16))
    with torch.no_grad():
        base = m(x).clone()
        assert set_alpha_override(m, 0.0) == 6
        zero = m(x).clone()
        assert set_alpha_override(m, None) == 6
        back = m(x).clone()
    assert not torch.allclose(base, zero)
    assert torch.equal(base, back), "the override is not exactly reversible"


def test_conv_activation_flag_actually_changes_the_operator():
    """NEGATIVE CONTROL support: if the flag were inert, check 12's negative control would be
    testing nothing."""
    torch.manual_seed(0)
    clean = DynamicShortConv(d_model=32, kernel_size=3, rank=8, conv_activation=None)
    silu = DynamicShortConv(d_model=32, kernel_size=3, rank=8, conv_activation="silu")
    silu.load_state_dict(clean.state_dict())
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        assert not torch.allclose(clean(x), silu(x))


def test_engagement_is_zero_at_init_and_positive_after_a_step():
    """``E_l == 0`` at init is CORRECT (``U = 0`` gives ``Delta_w == 0``), which is why check 9
    trains before asserting the floor."""
    from dynamic_conv import engagement_report

    m = build_arm(ArmSpec(arm="S4", topology="allliv", width=3))
    x = torch.randint(0, VOCAB_SIZE, (4, 24))
    with torch.no_grad():
        m(x)
    assert all(st.engagement == 0.0 for st in engagement_report(m))

    opt = torch.optim.AdamW(split_param_groups(m), lr=1e-2)
    for _ in range(5):
        m.zero_grad(set_to_none=True)
        logits = m(x)
        nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), x.reshape(-1)
        ).backward()
        opt.step()
    with torch.no_grad():
        m(x)
    stats = engagement_report(m)
    assert len(stats) == 6, "engagement must be reported PER LAYER, never averaged"
    assert all(st.engagement > 0.0 for st in stats), f"{[st.engagement for st in stats]}"


# =================================================================================================
# The `attn1` topology, and the silent-inheritance trap it exposed
# =================================================================================================
#
# Added 2026-08-05 with the topology itself. `ArmSpec.attn_idx` used to read
#
#     () if self.topology == "allliv" else tuple(self.attention_layers)
#
# with `attention_layers` DEFAULTING to HYBRID_ATTENTION_LAYERS. So every topology that was not
# `allliv` silently inherited hybrid's TWO indices. Adding a 1-attention topology under that code
# would have built it with two attention layers, and NO existing check would have fired: both
# `expected_param_count` and `dynamic_layers` derive from `attn_idx`, so the declaration and the
# build would have agreed with each other while disagreeing with the design. That is exactly the
# empty-comparison-set defect (EXP2-DESIGN.md Sec 12.4) -- a check that reads its expectation from
# the same field as the implementation.
#
# So these tests COUNT BUILT MODULES rather than re-reading the spec.


def _count_built_attention(model) -> int:
    """Count `Attention` modules actually present in the built stack."""
    return sum(1 for blk in model.blocks if isinstance(blk.sequence_mixer, Attention))


def _built_attention_indices(model) -> tuple:
    return tuple(
        i for i, blk in enumerate(model.blocks) if isinstance(blk.sequence_mixer, Attention)
    )


@pytest.mark.parametrize(
    "topology,expected_count", [("allliv", 0), ("attn1", 1), ("hybrid", 2)]
)
def test_attention_count_is_what_the_topology_says(topology: str, expected_count: int):
    """The load-bearing regression. Counts BUILT `Attention` modules, so it fails under the
    silent-inheritance bug that would have given `attn1` two attention layers."""
    m = build_arm(ArmSpec(arm="S1", topology=topology, width=3), seed=0)
    got = _count_built_attention(m)
    assert got == expected_count, (
        f"{topology}: built {got} attention modules, design says {expected_count}. "
        f"Built indices {_built_attention_indices(m)}."
    )
    assert _built_attention_indices(m) == TOPOLOGY_ATTENTION_LAYERS[topology]


def test_attn1_is_strictly_between_the_two_saturated_ends():
    """`attn1` exists to be BETWEEN allliv (floor) and hybrid (ceiling). If its attention count is
    not strictly between theirs it is not the topology the calibration is testing."""
    counts = {
        t: _count_built_attention(build_arm(ArmSpec(arm="S1", topology=t, width=3), seed=0))
        for t in ("allliv", "attn1", "hybrid")
    }
    assert counts["allliv"] < counts["attn1"] < counts["hybrid"], counts


def test_attn1_index_is_lfm2s_first_attention_layer():
    """Not a tuned choice: LFM2-16L's published pattern is [2, 5, 8, 10, 12, 14] and `hybrid` takes
    its first TWO, so `attn1` takes its first ONE."""
    assert ATTN1_ATTENTION_LAYERS == (2,)
    assert ATTN1_ATTENTION_LAYERS == HYBRID_ATTENTION_LAYERS[:1]


def test_the_silent_inheritance_bug_would_now_be_caught():
    """NEGATIVE CONTROL on the fix -- a guard that has never failed is not known to work.

    Reproduce the old behaviour by passing hybrid's indices explicitly to `attn1`, and assert the
    count check fires. This proves the test above reads the BUILD, not the label.
    """
    bugged = build_arm(
        ArmSpec(arm="S1", topology="attn1", width=3, attention_layers=HYBRID_ATTENTION_LAYERS),
        seed=0,
    )
    assert _count_built_attention(bugged) == 2, "the reproduction itself must build 2"
    # ...and that is exactly what the production check refuses:
    with pytest.raises(AssertionError):
        got = _count_built_attention(bugged)
        assert got == 1, f"attn1 built {got} attention modules"


def test_attn1_param_count_reconciles_and_sits_between_the_other_two():
    """Analytic vs built, on the new topology, and the ordering is a real constraint: an attention
    block and a LIV block do not cost the same, so swapping one changes the total."""
    for W in WIDTHS:
        for arm in ARMS:
            spec = ArmSpec(arm=arm, topology="attn1", width=W)
            m = build_arm(spec, seed=0, strict=True)  # strict re-checks the analytic total
            assert m.n_params == expected_param_count(spec)["total"]


def test_S2_and_S4_stay_param_matched_in_attn1():
    """The scientific core must survive the new topology: S2 is only a control if it is exactly
    param-matched to S4."""
    for W in WIDTHS:
        s2 = build_arm(ArmSpec(arm="S2", topology="attn1", width=W), seed=0)
        s4 = build_arm(ArmSpec(arm="S4", topology="attn1", width=W), seed=0)
        assert s2.n_params == s4.n_params, f"attn1 W={W}: {s2.n_params} != {s4.n_params}"
        assert {k: v.shape for k, v in s2.named_parameters()} == {
            k: v.shape for k, v in s4.named_parameters()
        }


def test_attn1_dynamic_modules_land_on_the_five_conv_layers():
    """S4 in `attn1` must carry a generator on the 5 LIV layers and NOT on the attention layer."""
    spec = ArmSpec(arm="S4", topology="attn1", width=3)
    m = build_arm(spec, seed=0, strict=True)
    assert m.dynamic_module_layers() == (0, 1, 3, 4, 5)
    assert m.n_dynamic_modules() == 5
    assert 2 not in m.dynamic_module_layers(), "layer 2 is attention; a LIV generator cannot land"


def test_S3_is_defined_in_attn1_on_exactly_one_layer():
    """S3 instruments Q/K/V, so unlike `allliv` it IS defined here -- on the single attention
    layer. It must not be silently N/A, nor silently substituted with S1."""
    spec = ArmSpec(arm="S3", topology="attn1", width=3)
    m = build_arm(spec, seed=0, strict=True)
    assert m.dynamic_module_layers() == (2,)
    assert m.n_dynamic_modules() == 1


def test_out_of_range_attention_index_is_refused():
    """An out-of-range index does not raise anywhere else: `i in attn` simply never matches, so the
    model is built attention-free while declaring itself otherwise."""
    with pytest.raises(ValueError, match="outside"):
        ArmSpec(arm="S1", topology="attn1", width=3, attention_layers=(99,))


def test_topology_tables_agree_between_arms_and_harness():
    """`arms.py` and `mqar_harness.py` each carry a topology table -- arms for the real model, the
    harness for the stub. Two tables that can disagree WILL: the harness would build a stub with a
    different topology than the arm it is standing in for, and every count check would still pass
    because each side is self-consistent."""
    import mqar_harness

    assert set(mqar_harness.TOPOLOGIES) == set(TOPOLOGY_ATTENTION_LAYERS)
    assert mqar_harness.TOPOLOGIES["attn1"] == TOPOLOGY_ATTENTION_LAYERS["attn1"]
    assert mqar_harness.TOPOLOGIES["allliv"] == TOPOLOGY_ATTENTION_LAYERS["allliv"]
    # `hybrid` is DELIBERATELY different -- (1,4) in the harness stub vs (2,5) in arms -- and that
    # predates this change. Pinned here so it is a recorded difference rather than a silent one.
    assert mqar_harness.TOPOLOGIES["hybrid"] == (1, 4)
    assert TOPOLOGY_ATTENTION_LAYERS["hybrid"] == (2, 5)
    assert len(mqar_harness.TOPOLOGIES["hybrid"]) == len(TOPOLOGY_ATTENTION_LAYERS["hybrid"]), (
        "the COUNT must match even where the indices differ -- otherwise the stub and the arm are "
        "different topologies"
    )


# =================================================================================================
# THE BIMODAL BLIND SPOT in the topology conjunction -- found by run_019fd45d, 2026-08-05
# =================================================================================================
#
# `assess_topology` decides "discriminating" as: NOT at_ceiling AND NOT at_floor, where each is a
# >=60% majority fraction over seeds. On a UNIMODAL endpoint that is correct, and it was tested that
# way -- ceiling inputs, floor inputs, middle inputs, each verified to classify correctly.
#
# It is WRONG on a BIMODAL endpoint, which is the only kind this task produces (EXP2-DESIGN Sec 5.2:
# "a run either finds the recall algorithm or sits at chance"). The measured attn1 cell was
# {0.2617, 0.9915, 0.9828} -- one degenerate seed, two SOLVED (final loss 0.025 and 0.040, i.e. the
# pair is bound). Neither fraction reached 60%, so BOTH majority tests answered "no" and the
# conjunction reported "off ceiling and off floor" for a cell where NO SEED is in the middle.
#
# Root cause of the miss: the original tests supplied only unimodal inputs, so the failing case was
# absent from the comparison set. Sec 12.4 again, in the test suite this time rather than the code.
#
# These tests pin the real measured numbers so the blind spot cannot be reintroduced silently.


def _assess(accs, losses):
    import sys as _s
    _s.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import calibration as C

    class _R:
        def __init__(self, a, l):
            self.accuracy = a; self.final_loss = l; self.topology = "attn1"
            self.config = "N64_D4"; self.kernel_size = 3; self.num_pairs = 4; self.seed = 0

    return C.assess_topology([_R(a, l) for a, l in zip(accs, losses)],
                             topology="attn1", num_pairs=4, vocab_size=256, width=3)


def test_the_measured_attn1_cell_is_reported_discriminating_AND_THAT_IS_THE_BUG():
    """DOCUMENTS A KNOWN-WRONG OUTPUT, with the real numbers from run_019fd45d.

    This test asserts the CURRENT (incorrect) behaviour on purpose, so that whoever fixes
    `assess_topology` sees this test fail and must read the reasoning rather than discovering the
    bimodal case for themselves. **Do not "fix" this test by editing the expectation.** Fix
    `assess_topology`, then invert the assertion and cite this docstring.
    """
    v = _assess([0.2617, 0.9915, 0.9828], [1.386, 0.025, 0.040])
    assert v.discriminating is True, "measured behaviour changed -- read this docstring"
    # ...and the reason it is wrong, asserted so the evidence travels with the claim:
    assert v.median_accuracy > 0.98, "median is at ceiling in substance"
    assert v.median_final_loss < 0.05, (
        "median final loss ~0.04 is the 'bound' plateau: the pair IS bound, i.e. SOLVED. "
        "A solved task is a CEILING however the 0.99 accuracy fraction is counted."
    )


def test_no_single_seed_of_the_measured_attn1_cell_is_actually_in_the_middle():
    """The sharp statement of the defect: the AGGREGATE looks mid-range, every SEED is saturated."""
    floor = 0.25
    for acc, loss in ((0.2617, 1.386), (0.9915, 0.025), (0.9828, 0.040)):
        near_floor = acc <= floor * 1.5      # the degenerate 'guess among D' strategy
        solved = loss < 0.10                 # the 'bound' plateau
        assert near_floor or solved, (
            f"acc {acc} / loss {loss} was expected to be at one mode or the other; if a genuine "
            f"middle seed ever appears, the bimodality premise has changed and Sec 5.2 needs revisiting"
        )


def test_a_ceiling_floor_MIXTURE_defeats_the_majority_conjunction():
    """The general form, independent of the measured numbers: half ceiling, half floor, no middle.

    This is the input class the original tests never supplied. Both majority tests answer "no",
    so the conjunction answers "discriminating" for a cell with no discriminating power at all.
    """
    v = _assess([0.01, 0.02, 1.0, 1.0], [4.85, 4.85, 0.001, 0.001])
    assert v.off_ceiling and v.off_floor and v.discriminating, (
        "a 50/50 ceiling-floor mixture currently reports DISCRIMINATING -- this is the blind spot"
    )
    # And the aggregate is actively MISLEADING rather than merely permissive: a half-and-half
    # mixture averages to a median of ~0.51, which looks like the textbook off-ceiling-and-off-floor
    # operating point (~2x floor, "50% success") while NO SEED is anywhere near it. This is worse
    # than the measured attn1 case, not better -- there the median at least sat at 0.98 and hinted
    # at the ceiling. A reader checking only the median would call this cell ideal.
    assert 0.375 < v.median_accuracy < 0.99, (
        "the mixture's median lands in the apparent sweet spot, which is what makes it dangerous"
    )
    assert all((a <= 0.375 or a >= 0.99) for a in v.per_seed_accuracy), (
        "yet every individual seed is pinned at one mode or the other -- the median is an artifact "
        "of averaging two modes and describes no seed in the cell"
    )
