import pytest

from olmo_core.nn.transformer.liv_arms import (
    ARMS,
    ATTENTION_LAYERS,
    D_MODEL,
    DOLMA2_VOCAB_SIZE,
    HEAD_DIM,
    KERNEL_SIZE,
    L0_PARAM_TARGET,
    L0_PARAM_TARGET_DOLMA2,
    N_HEADS,
    N_KV_HEADS,
    N_LAYERS,
    SOLVED_WIDTHS,
    SWIGLU_WIDTH,
    VOCAB_SIZE,
    _count_params,
    arm_report,
    arms_for_vocab,
    build_arm,
    solve_d_model,
    solve_swiglu_width,
)


def test_l0_hits_the_exact_frozen_parameter_target():
    """
    ``L0`` must equal :data:`L0_PARAM_TARGET` exactly -- 338,886,400 at vocab 50,304.

    This single assertion validates the whole geometry at once: tied embeddings, SwiGLU width
    4,608, per-head QK-norm, the 10/6 layer split, and the mixer formula. Two omissions were
    caught by exactly this check:

    * untied embeddings, which added a second ``vocab x d_model`` tensor (~19% of the model);
    * missing per-head QK-norm, which LFM2 has as ``q_layernorm``/``k_layernorm`` of size
      ``head_dim`` on each of 6 attention layers (6 x 2 x 64 = 768).

    The target is no longer LFM2's released 354,483,968 -- see :data:`VOCAB_SIZE` for why that
    number turned out to be a padding artifact rather than a reproducible shape. What the study
    actually depends on is arm *differences*, which are bit-identical across vocab sizes.
    """
    assert _count_params(build_arm("L0")) == L0_PARAM_TARGET


def test_l0_ledger_reconciles_component_by_component():
    """
    An exact total can still hide two offsetting errors, so check the components independently.

    Derives every term from the module's own constants rather than hardcoding them -- an earlier
    version pinned ``vocab = 65536`` inline and broke the moment the vocabulary was redeclared,
    which is a test failing for the wrong reason.
    """
    d, vocab, k = D_MODEL, VOCAB_SIZE, KERNEL_SIZE
    n_attn = len(ATTENTION_LAYERS)
    n_liv = N_LAYERS - n_attn

    embeddings = vocab * d  # tied: the LM head reuses this tensor
    attn_mixer = (
        d * (N_HEADS * HEAD_DIM) + 2 * d * (N_KV_HEADS * HEAD_DIM) + (N_HEADS * HEAD_DIM) * d
    )
    liv_mixer = 4 * d * d + k * d  # the brainlift's 4d^2 + kd
    mlp = 3 * d * SWIGLU_WIDTH
    block_norms = 2 * d
    qk_norms = 2 * HEAD_DIM  # per-head q_layernorm + k_layernorm, attention layers only

    total = (
        embeddings
        + n_attn * (attn_mixer + qk_norms)
        + n_liv * liv_mixer
        + N_LAYERS * (mlp + block_norms)
        + d  # final norm
    )
    assert total == L0_PARAM_TARGET


def test_vocab_covers_every_token_in_the_corpus():
    """
    ``VOCAB_SIZE`` must exceed the largest token id in the training data, or the embedding lookup
    indexes out of bounds and training dies on the first batch.

    The corpus is GPT-2 tokenized, whose EOS is **50,256** and appears at every document
    boundary -- so 50,257 is a hard floor. A request for a round 50,000 would have crashed
    immediately (64,472 of the first 50M tokens are >= 50,000).
    """
    GPT2_MAX_TOKEN_ID = 50_256
    assert VOCAB_SIZE > GPT2_MAX_TOKEN_ID
    assert VOCAB_SIZE % 128 == 0, "pad to a multiple of 128 for tensor-core alignment"


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
    the committed values land at 0.0095% (vocab 50,304).
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

    # Measured 1.207x at 4K and 1.886x at 32K under the 6x-params (fwd+bwd) convention that
    # Attention uses. Thresholds sit clear of those values rather than hugging them: an earlier
    # `> 1.2` passed at 1.207 with a 0.007 margin, which would have flipped on any small change
    # without anyone noticing the ratio had moved. The claim under test is "the gap is large and
    # widens with context", so guard that, loosely but meaningfully.
    ratio_4k = a16.num_flops_per_token(4096) / l0.num_flops_per_token(4096)
    ratio_32k = a16.num_flops_per_token(32768) / l0.num_flops_per_token(32768)
    assert 1.10 < ratio_4k < 1.35, ratio_4k
    assert 1.70 < ratio_32k < 2.10, ratio_32k
    assert ratio_32k > ratio_4k * 1.3  # the gap widens substantially with context


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


def test_dolma2_vocab_matches_olmo_cores_own_padding():
    """
    :data:`DOLMA2_VOCAB_SIZE` must equal what OLMo-core derives, not a number typed by hand.

    The same check on the GPT-2 side caught nothing only because it was already right; the point
    is that the corpus and the embedding table cannot silently disagree.
    """
    from olmo_core.data import TokenizerConfig

    assert TokenizerConfig.dolma2().padded_vocab_size() == DOLMA2_VOCAB_SIZE
    assert DOLMA2_VOCAB_SIZE % 128 == 0


@pytest.mark.parametrize("vocab", [VOCAB_SIZE, DOLMA2_VOCAB_SIZE])
def test_l0_hits_its_exact_target_at_both_vocabularies(vocab: int):
    target = L0_PARAM_TARGET if vocab == VOCAB_SIZE else L0_PARAM_TARGET_DOLMA2
    arms = arms_for_vocab(vocab)
    assert _count_params(build_arm(arms["L0"], vocab_size=vocab)) == target


@pytest.mark.parametrize("vocab", [VOCAB_SIZE, DOLMA2_VOCAB_SIZE])
def test_the_arm_contrast_is_vocabulary_independent(vocab: int):
    """
    The load-bearing invariant of the whole study, asserted at both vocabularies.

    Doubling the vocabulary adds 51,249,152 tied-embedding parameters (+15.1% of ``L0``), which
    would look alarming if the arms were compared on absolute size. They are not: every arm shares
    one embedding table and the arms differ only in the mixer, so the vocabulary shifts all of them
    by the same constant and cancels out of every reported contrast. If this ever fails, the
    vocabulary has stopped being a shared constant and no cross-vocabulary claim survives.
    """
    arms = arms_for_vocab(vocab)
    l0 = _count_params(build_arm(arms["L0"], vocab_size=vocab))
    f = _count_params(build_arm(arms["F-r128"], vocab_size=vocab))
    g = _count_params(build_arm(arms["G-grouped"], vocab_size=vocab))

    assert l0 - f == 15_728_640
    assert f == g, "the cost-matched pair must be bit-identical, not merely close"


@pytest.mark.parametrize("vocab", [VOCAB_SIZE, DOLMA2_VOCAB_SIZE])
def test_solved_arms_stay_matched_at_both_vocabularies(vocab: int):
    """
    ``A16-P`` and ``N-narrow`` are solved against targets that MOVE with the vocabulary, so a
    vocabulary change silently un-matches them unless the widths are re-solved. That is the
    failure this test exists to catch: a capacity control still labelled "same size as F-r128"
    while actually carrying a different budget.
    """
    arms = arms_for_vocab(vocab)
    l0 = _count_params(build_arm(arms["L0"], vocab_size=vocab))
    f = _count_params(build_arm(arms["F-r128"], vocab_size=vocab))

    narrow = _count_params(build_arm(arms["N-narrow"], vocab_size=vocab))
    assert abs(narrow - f) / f < 0.0005, f"N-narrow {narrow:,} vs F-r128 {f:,}"

    a16 = _count_params(build_arm(arms["A16-P"], vocab_size=vocab))
    assert abs(a16 - l0) / l0 < 0.0005, f"A16-P {a16:,} vs L0 {l0:,}"


@pytest.mark.parametrize("vocab", [VOCAB_SIZE, DOLMA2_VOCAB_SIZE])
def test_solvers_reproduce_the_table_at_both_vocabularies(vocab: int):
    """A drift between SOLVED_WIDTHS and the derivation that justified it must fail loudly."""
    target = L0_PARAM_TARGET if vocab == VOCAB_SIZE else L0_PARAM_TARGET_DOLMA2
    arms = arms_for_vocab(vocab)

    width, _ = solve_swiglu_width(arms["A16-P"], target_params=target, vocab_size=vocab)
    assert width == SOLVED_WIDTHS[vocab].a16p_swiglu

    f = _count_params(build_arm(arms["F-r128"], vocab_size=vocab))
    d_model, _ = solve_d_model(arms["N-narrow"], target_params=f, vocab_size=vocab)
    assert d_model == SOLVED_WIDTHS[vocab].narrow_d_model


def test_unknown_vocab_raises_instead_of_silently_mismatching():
    """
    Defaulting would be the dangerous behaviour: the arms would build fine, train fine, and be
    matched against the wrong target with nothing to indicate it.
    """
    with pytest.raises(KeyError, match="no solved widths"):
        arms_for_vocab(65_536)


def test_arms_for_vocab_does_not_mutate_the_module_level_declarations():
    """``ARMS`` is shared global state; returning a mutated view would corrupt later callers."""
    before = (ARMS["A16-P"].swiglu_width, ARMS["N-narrow"].d_model, ARMS["N-narrow"].swiglu_width)
    arms_for_vocab(DOLMA2_VOCAB_SIZE)
    after = (ARMS["A16-P"].swiglu_width, ARMS["N-narrow"].d_model, ARMS["N-narrow"].swiglu_width)
    assert before == after
