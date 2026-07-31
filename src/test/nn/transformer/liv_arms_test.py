import pytest

from olmo_core.nn.transformer.liv_arms import (
    ARMS,
    ATTENTION_LAYERS,
    D_MODEL,
    L0_PARAM_TARGET,
    N_LAYERS,
    _count_params,
    arm_report,
    build_arm,
    solve_d_model,
    solve_swiglu_width,
)


def test_l0_hits_the_exact_frozen_parameter_target():
    """
    ``L0`` must equal 354,483,968 exactly -- the released-scale ledger from the frozen protocol.

    This single assertion validates the whole geometry at once: tied embeddings, SwiGLU width
    4,608, per-head QK-norm, the 10/6 layer split, and the mixer formula. Two omissions were
    caught by exactly this check:

    * untied embeddings, which added a second 65,536 x 1,024 tensor (+67,108,864, ~19%);
    * missing per-head QK-norm, which LFM2 has as ``q_layernorm``/``k_layernorm`` of size
      ``head_dim`` on each of 6 attention layers (6 x 2 x 64 = 768).
    """
    assert _count_params(build_arm("L0")) == L0_PARAM_TARGET


def test_l0_ledger_reconciles_component_by_component():
    """
    An exact total can still hide two offsetting errors, so check the components independently.
    """
    d, vocab, k = D_MODEL, 65536, 3
    n_attn = len(ATTENTION_LAYERS)
    n_liv = N_LAYERS - n_attn

    embeddings = vocab * d
    attn_mixer = d * (16 * 64) + 2 * d * (8 * 64) + (16 * 64) * d
    liv_mixer = 4 * d * d + k * d  # the brainlift's 4d^2 + kd
    mlp = 3 * d * 4608
    block_norms = 2 * d
    qk_norms = 2 * 64  # per-head q_layernorm + k_layernorm, attention layers only

    total = (
        embeddings
        + n_attn * (attn_mixer + qk_norms)
        + n_liv * liv_mixer
        + N_LAYERS * (mlp + block_norms)
        + d  # final norm
    )
    assert total == L0_PARAM_TARGET


def test_every_arm_places_mixers_where_declared():
    """
    The declaration is only meaningful if the built model matches it.

    Guards the trap that per-layer overrides go through ``block.sequence_mixer``, not
    ``block.attention`` -- setting the wrong field silently yields an all-attention model that
    trains fine and answers a different question.
    """
    for name, arm in ARMS.items():
        model = build_arm(name).build(init_device="meta")
        kinds = [type(b.attention).__name__ for b in model.blocks.values()]
        got_attn = {i for i, kind in enumerate(kinds) if kind == "Attention"}
        assert got_attn == set(arm.attention_layers), f"{name}: attention at {sorted(got_attn)}"
        assert kinds.count("ShortConv") == arm.n_liv_layers, name


def test_kernel_width_arms_differ_only_in_kernel_width():
    """
    The P3 width arms must be otherwise identical to ``L0``, or the comparison is confounded.
    A k-tap change costs exactly ``(k - 3) * d`` per LIV layer and nothing else.
    """
    base = _count_params(build_arm("L0"))
    for name, k in (("W-k5", 5), ("W-k9", 9), ("W-k15", 15)):
        arm = ARMS[name]
        assert arm.gate_structure == ARMS["L0"].gate_structure
        assert arm.attention_layers == ARMS["L0"].attention_layers
        assert arm.d_model == ARMS["L0"].d_model
        expected = base + (k - 3) * D_MODEL * arm.n_liv_layers
        assert _count_params(build_arm(name)) == expected, name


def test_matched_cost_pair_is_exactly_matched():
    """
    ``F-r128`` and ``G-grouped`` must cost *identically*, which is what makes the low-rank vs
    block-diagonal comparison a clean quality question. They are not nested -- block-diagonal
    is full-rank without cross-block mixing, low-rank mixes all channels through a
    128-dimensional bottleneck -- so neither dominates by construction.
    """
    assert _count_params(build_arm("F-r128")) == _count_params(build_arm("G-grouped"))


def test_narrow_control_is_solved_against_the_arm_it_controls():
    """
    ``N-narrow`` exists to answer "why not just build a narrower model?", so it must match
    ``F-r128``'s parameter count closely or it is not a control at all. Tolerance is 0.05%;
    the committed values land at 0.0145%.
    """
    target = _count_params(build_arm("F-r128"))
    got = _count_params(build_arm("N-narrow"))
    assert abs(got - target) / target < 0.0005, f"{got:,} vs {target:,}"


def test_all_attention_control_is_parameter_matched_to_l0():
    """``A16-P``'s SwiGLU width is solved, not chosen. Tolerance 0.05%."""
    got = _count_params(build_arm("A16-P"))
    assert abs(got - L0_PARAM_TARGET) / L0_PARAM_TARGET < 0.0005, f"{got:,}"


def test_parameter_matching_is_not_compute_matching():
    """
    The load-bearing methodological point, asserted so it cannot be forgotten.

    ``A16-P`` is parameter-matched to ``L0`` within 0.03% yet uses ~1.94x the FLOPs per token
    at 32K, because attention's score term grows with context while a convolution's does not.
    Any compute-controlled comparison must match on ``num_flops_per_token``.
    """
    l0 = build_arm("L0").build(init_device="meta")
    a16 = build_arm("A16-P").build(init_device="meta")

    params_ratio = _count_params(build_arm("A16-P")) / L0_PARAM_TARGET
    assert 0.995 < params_ratio < 1.005  # parameter-matched

    ratio_4k = a16.num_flops_per_token(4096) / l0.num_flops_per_token(4096)
    ratio_32k = a16.num_flops_per_token(32768) / l0.num_flops_per_token(32768)
    assert ratio_4k > 1.2, ratio_4k
    assert ratio_32k > 1.8, ratio_32k
    assert ratio_32k > ratio_4k  # the gap widens with context


def test_fewer_attention_layers_cuts_long_context_compute_most():
    """
    ``A-fewer3`` is P2's strongest competitor precisely because halving attention layers cuts
    read bandwidth *and* compute -- something cross-layer KV sharing structurally cannot do.
    """
    l0 = build_arm("L0").build(init_device="meta")
    few = build_arm("A-fewer3").build(init_device="meta")
    assert few.num_flops_per_token(4096) / l0.num_flops_per_token(4096) < 0.95
    assert few.num_flops_per_token(32768) / l0.num_flops_per_token(32768) < 0.80


def test_solvers_reproduce_the_committed_widths():
    """
    The committed widths must be what the solvers produce, or the declarations have drifted
    from the derivation that justified them.
    """
    width, _ = solve_swiglu_width("A16-P")
    assert width == ARMS["A16-P"].swiglu_width

    d_model, _ = solve_d_model("N-narrow", target_params=_count_params(build_arm("F-r128")))
    assert d_model == ARMS["N-narrow"].d_model


def test_mqa_arm_reduces_kv_heads_only():
    arm = ARMS["Q-mqa"]
    assert arm.n_kv_heads == 1
    assert arm.attention_layers == ARMS["L0"].attention_layers
    assert arm.d_model == ARMS["L0"].d_model
    assert _count_params(build_arm("Q-mqa")) < L0_PARAM_TARGET


@pytest.mark.parametrize("name", list(ARMS))
def test_every_arm_runs_forward_and_backward(name: str):
    """Each arm must be trainable, with every parameter receiving gradient."""
    import torch

    cfg = build_arm(name, vocab_size=256, init_device="cpu")
    model = cfg.build(init_device="cpu")
    out = model(torch.randint(0, 256, (2, 12)))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert not [n for n, p in model.named_parameters() if p.grad is None]


def test_arm_report_covers_every_arm():
    report = arm_report()
    for name in ARMS:
        assert name in report
    assert "flops@32K" in report
