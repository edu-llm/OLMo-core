import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.mamba3 import DEFAULT_D_STATE, Mamba3, Mamba3Config
from olmo_core.nn.mamba3.mixer import Mamba3Mixer, Mamba3MixerConfig
from olmo_core.nn.transformer import Transformer


def _tiny_hybrid_config(**kwargs) -> Mamba3Config:
    return Mamba3Config.mamba3_hybrid_like(
        d_model=64,
        vocab_size=128,
        n_layers=4,
        n_heads=4,
        intermediate_size=128,
        mamba_n_heads=4,
        mamba_head_dim=16,
        d_state=16,
        n_groups=1,
        mimo_rank=2,
        **kwargs,
    )


def test_mamba3_hybrid_block_pattern_and_mixer_types():
    config = _tiny_hybrid_config()

    # 1:3 attention-to-Mamba-3 ratio.
    assert config.block_pattern == ["mamba3", "mamba3", "mamba3", "attn"]
    assert isinstance(config.block, dict)
    assert set(config.block.keys()) == {"mamba3", "attn"}

    resolved = config.resolved_block_configs
    assert len(resolved) == 4
    mixer_types = [type(b.sequence_mixer).__name__ for b in resolved]
    assert mixer_types == [
        "Mamba3MixerConfig",
        "Mamba3MixerConfig",
        "Mamba3MixerConfig",
        "AttentionConfig",
    ]

    # Attention layers are NoPE by default.
    attn_block = resolved[3]
    assert isinstance(attn_block.sequence_mixer, AttentionConfig)
    assert attn_block.sequence_mixer.rope is None


def test_mamba3_hybrid_use_rope_flag():
    config = _tiny_hybrid_config(use_rope=True)
    attn_block = config.resolved_block_configs[3]
    assert isinstance(attn_block.sequence_mixer, AttentionConfig)
    assert attn_block.sequence_mixer.rope is not None


def _mamba_mixer(config: Mamba3Config) -> Mamba3MixerConfig:
    """Pull the Mamba-3 mixer out of a hybrid config's block dict, narrowed for the type checker."""
    assert isinstance(config.block, dict)
    mixer = config.block["mamba3"].sequence_mixer
    assert isinstance(mixer, Mamba3MixerConfig)
    return mixer


def test_mamba3_olmo3_370M_is_architecturally_equivalent_to_olmo3_370M():
    """
    The preset must be OLMo-3-370M with Mamba-3 substituted for the sliding-window layers.

    OLMo-3's attention pattern is ``[4096, 4096, 4096, -1]``: three sliding-window layers then
    one full-attention layer. The Mamba-3 hybrid's ``["mamba3", "mamba3", "mamba3", "attn"]`` is
    the same 3:1 shape, so the swap is exactly "replace the sub-quadratic layer" and the
    ablation isolates SSM-versus-sliding-window rather than a diffuse architecture change.
    Everything else -- width, depth, head count, feed-forward size, RoPE, QK-norm, norm epsilon
    -- has to match, or the comparison is confounded.
    """
    from olmo_core.nn.transformer import TransformerConfig

    vocab_size = 1024
    reference = TransformerConfig.olmo3_370M(vocab_size=vocab_size)
    config = Mamba3Config.mamba3_olmo3_370M(vocab_size=vocab_size)

    assert config.d_model == reference.d_model
    assert config.n_layers == reference.n_layers

    ref_attn = reference.resolved_block_configs[0].sequence_mixer
    resolved = config.resolved_block_configs
    assert len(resolved) == reference.n_layers

    # 12 Mamba-3 + 4 full-attention, in the same 3:1 period as OLMo-3's [4096, 4096, 4096, -1].
    kinds = [type(b.sequence_mixer).__name__ for b in resolved]
    assert kinds == (["Mamba3MixerConfig"] * 3 + ["AttentionConfig"]) * (len(resolved) // 4)
    assert kinds.count("Mamba3MixerConfig") == 12
    assert kinds.count("AttentionConfig") == 4

    attn = resolved[3].sequence_mixer
    assert isinstance(attn, AttentionConfig)
    assert attn.n_heads == ref_attn.n_heads
    assert attn.rope is not None and ref_attn.rope is not None
    assert attn.rope.theta == ref_attn.rope.theta
    assert (attn.qk_norm is None) == (ref_attn.qk_norm is None)
    feed_forward = resolved[3].feed_forward
    ref_feed_forward = reference.resolved_block_configs[0].feed_forward
    assert feed_forward is not None and ref_feed_forward is not None
    assert feed_forward.hidden_size == ref_feed_forward.hidden_size

    # The surviving attention layers are the *full-attention* ones. Carrying a sliding window
    # here would window the very layers OLMo-3 leaves global.
    assert getattr(attn, "sliding_window", None) is None


def test_mamba3_olmo3_370M_is_parameter_matched():
    """
    Parameter count must stay close to the reference, on the right metric.

    Both configs leave the LM head untied, so ``olmo3_370M`` is 474M total and only the repo's
    own ``num_active_non_embedding_params`` recovers the 371M the name refers to. Matching on
    the label instead of this metric silently produces models ~100M apart.

    The tolerance is 3%, not 1%. The preset is deliberately SISO with ``n_groups=1`` so that
    ``rotation_block_size`` is the *only* thing that changes between the TC^0 baseline and an
    NC^1 arm, and that costs 2.2%: ``mimo_rank`` only widens ``in_B``/``in_C``
    (``bc_out = n_groups * mimo_rank * d_state``), so dropping 4 -> 1 removes 787k parameters
    per Mamba layer, 9.45M over the twelve of them. Buying it back with ``n_groups=4`` would
    match to 0.95% but add a second axis of difference to every later comparison.
    """
    from olmo_core.nn.transformer import TransformerConfig

    vocab_size = 100352
    reference = TransformerConfig.olmo3_370M(vocab_size=vocab_size)
    config = Mamba3Config.mamba3_olmo3_370M(vocab_size=vocab_size)

    ref_params = reference.num_active_non_embedding_params
    got_params = config.num_active_non_embedding_params
    relative = abs(got_params - ref_params) / ref_params
    assert relative < 0.03, (
        f"parameter mismatch: {got_params / 1e6:.1f}M vs reference {ref_params / 1e6:.1f}M "
        f"({relative:.2%})"
    )


def test_mamba3_olmo3_370M_defaults_are_a_clean_tc0_baseline():
    """
    The preset must default to the abelian baseline, with nothing else in the way.

    ``rotation_block_size=2`` is TC^0 (a cumulative *sum* of angles). SISO with a single
    ``(B, C)`` group keeps every other axis fixed, so a later NC^1 or PNC^1 arm differs from
    this run in exactly one config field. ``a_log_init_max`` is lowered from the library
    default of 16 because at 16 the decay horizon is 11-44 tokens, against the 4096-token
    sliding window these layers replace.
    """
    mixer = _mamba_mixer(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))
    assert mixer.rotation_block_size == 2
    assert mixer.mimo_rank == 1
    assert mixer.n_groups == 1
    assert mixer.a_log_init_max == 0.1


@pytest.mark.parametrize("rotation_block_size", [2, 3])
def test_mamba3_olmo3_370M_switches_block_size_without_touching_anything_else(
    rotation_block_size: int,
):
    """
    Flipping ``rotation_block_size`` must be the only difference between the arms.

    If any other field moved with it, the NC^1 comparison would confound the transition
    algebra with whatever else changed.
    """
    baseline = _mamba_mixer(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))
    switched = _mamba_mixer(
        Mamba3Config.mamba3_olmo3_370M(vocab_size=1024, rotation_block_size=rotation_block_size)
    )
    differing = {
        field for field in vars(baseline) if getattr(baseline, field) != getattr(switched, field)
    }
    assert differing <= {"rotation_block_size"}, f"unexpected fields also changed: {differing}"


def test_mamba3_olmo3_370M_supports_block_size_3_with_no_other_change():
    """
    The NC^1 arm must be reachable by flipping one field and nothing else.

    This previously pinned the opposite -- that the default ``d_state`` could *not* express
    ``b=3``, so an NC^1 arm needed a second override. The default now admits 2, 3 and 4 alike,
    which is the property the ablation actually depends on, so the assertion is inverted rather
    than dropped: it still fails the moment someone picks a ``d_state`` that reintroduces the
    trap.
    """
    baseline = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    nc1 = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024, rotation_block_size=3)

    # Builds without a d_state override -- the whole point.
    nc1.build(init_device="meta")

    base_mixer, nc1_mixer = _mamba_mixer(baseline), _mamba_mixer(nc1)
    assert base_mixer.d_state == nc1_mixer.d_state
    assert nc1_mixer.d_state % 3 == 0
    differing = {f for f in vars(base_mixer) if getattr(base_mixer, f) != getattr(nc1_mixer, f)}
    assert differing == {"rotation_block_size"}, f"unexpected fields also changed: {differing}"


def test_mamba3_is_transformer_subclass():
    # The transformer train module relies on `isinstance(model, Transformer)`.
    config = _tiny_hybrid_config()
    model = config.build(init_device="meta")
    assert isinstance(model, Mamba3)
    assert isinstance(model, Transformer)


def test_mamba3_hybrid_num_params_matches_built():
    config = _tiny_hybrid_config()
    model = config.build(init_device="meta")
    num_actual = sum(p.numel() for p in model.parameters())
    assert config.num_params == num_actual


def test_mamba3_hybrid_forward_backward():
    torch.manual_seed(0)
    config = _tiny_hybrid_config()
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    # Layers should be 3 Mamba-3 mixers followed by 1 attention.
    mixers = [type(b.attention).__name__ for b in model.blocks.values()]
    assert mixers == ["Mamba3Mixer", "Mamba3Mixer", "Mamba3Mixer", "Attention"]
    assert isinstance(model.blocks["0"].attention, Mamba3Mixer)

    input_ids = torch.randint(0, 128, (2, 16))
    labels = torch.randint(0, 128, (2, 16))
    output = model(input_ids, labels=labels)
    loss = output.loss
    assert torch.isfinite(loss).all()
    loss.sum().backward()


def test_mamba3_hybrid_presets():
    for preset in (Mamba3Config.mamba3_hybrid_190M, Mamba3Config.mamba3_hybrid_1B):
        config = preset(vocab_size=1024)
        assert config.block_pattern == ["mamba3", "mamba3", "mamba3", "attn"]
        assert config.n_layers % 4 == 0
        # num_params is computable and positive.
        assert config.num_params > 0


@pytest.mark.parametrize("preset", [Mamba3Config.mamba3_hybrid_190M, Mamba3Config.mamba3_hybrid_1B])
def test_mamba3_hybrid_presets_support_non_solvable_rotation(preset):
    """
    The presets must be usable with ``rotation_block_size=3`` directly.

    ``b >= 3`` is the whole reason the rotation block size exists -- it is what makes the
    transition monoid non-solvable -- and the presets are the documented entry point for smoke
    runs. This used to require a ``d_state`` override because the default was not divisible by
    3; it no longer does, so the override is gone and the test asserts the plain call works.
    """
    config = preset(vocab_size=1024, rotation_block_size=3)
    mamba_mixers = [
        m
        for b in config.resolved_block_configs
        if isinstance(m := b.sequence_mixer, Mamba3MixerConfig)
    ]
    assert mamba_mixers
    assert all(m.d_state == DEFAULT_D_STATE and m.rotation_block_size == 3 for m in mamba_mixers)

    model = config.build(init_device="meta")
    assert config.num_params == sum(p.numel() for p in model.parameters())


def test_mamba3_hybrid_attn_backend_threads_to_attention_blocks():
    """``attn_backend`` must reach the attention layers without disturbing the Mamba-3 ones."""
    backend = AttentionBackendName.flash_2

    for config in (
        _tiny_hybrid_config(attn_backend=backend),
        Mamba3Config.mamba3_hybrid_190M(vocab_size=1024, attn_backend=backend),
        Mamba3Config.mamba3_hybrid_1B(vocab_size=1024, attn_backend=backend),
    ):
        mixers = [b.sequence_mixer for b in config.resolved_block_configs]
        attn_mixers = [m for m in mixers if isinstance(m, AttentionConfig)]
        mamba_mixers = [m for m in mixers if isinstance(m, Mamba3MixerConfig)]

        assert attn_mixers, "expected at least one attention layer in the hybrid"
        assert mamba_mixers, "expected at least one Mamba-3 layer in the hybrid"
        assert all(m.backend == backend for m in attn_mixers)
        # The Mamba-3 mixer has no attention backend to speak of.
        assert all(not hasattr(m, "backend") for m in mamba_mixers)


def test_mamba3_hybrid_omitting_attn_backend_is_unchanged():
    """Omitting ``attn_backend`` must reproduce the pre-existing config exactly."""
    default = Mamba3Config.mamba3_hybrid_190M(vocab_size=1024)

    # Recorded values. These track `DEFAULT_D_STATE`, so they move whenever it does -- that is
    # expected and is not what this test is about. The load-bearing assertion is the
    # `with_backend == default` comparison below, which is independent of the state size.
    assert default.num_params == 120_998_808
    assert default.num_non_embedding_params == 120_212_376
    assert all(
        m.backend is None
        for b in default.resolved_block_configs
        if isinstance(m := b.sequence_mixer, AttentionConfig)
    )

    # Passing a backend must change the attention backend and nothing else.
    with_backend = Mamba3Config.mamba3_hybrid_190M(
        vocab_size=1024, attn_backend=AttentionBackendName.flash_2
    )
    assert with_backend.num_params == default.num_params
    for block in with_backend.resolved_block_configs:
        if isinstance(block.sequence_mixer, AttentionConfig):
            block.sequence_mixer.backend = None
    assert with_backend.as_config_dict() == default.as_config_dict()


def test_mamba3_bad_block_pattern_raises():
    with pytest.raises(OLMoConfigurationError):
        _tiny_hybrid_config(block_pattern=["mamba3", "does_not_exist"])


def test_mamba3_config_round_trip():
    """The hybrid config (nested Mamba3BlockConfig + mixer/attention configs) must round-trip."""
    config = _tiny_hybrid_config()
    rebuilt = Mamba3Config.from_dict(config.as_config_dict())

    assert rebuilt.block_pattern == config.block_pattern
    assert rebuilt.num_params == config.num_params
    orig_types = [type(b.sequence_mixer).__name__ for b in config.resolved_block_configs]
    rebuilt_types = [type(b.sequence_mixer).__name__ for b in rebuilt.resolved_block_configs]
    assert (
        rebuilt_types
        == orig_types
        == [
            "Mamba3MixerConfig",
            "Mamba3MixerConfig",
            "Mamba3MixerConfig",
            "AttentionConfig",
        ]
    )
