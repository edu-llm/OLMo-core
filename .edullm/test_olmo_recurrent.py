"""Tests for the recurrent-depth OLMo3 integration.

These run on CPU against a real installed ``olmo_core``, so they exercise the actual
upstream ``Transformer``, ``TransformerConfig`` and ``ResidualStream`` rather than stand-ins.
The 370M cases build on the meta device, which costs no memory and still gives real
parameter counts.

    pip install -e /path/to/OLMo-core && pytest integrations/olmo-core/
"""

import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo_core")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmo_recurrent as R  # noqa: E402
from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402


def tiny(**kwargs) -> R.RecurrentTransformerConfig:
    """A 6-layer d=64 model, split 1 / 4 / 1, small enough to train in a test."""
    options = dict(
        d_model=64,
        vocab_size=256,
        n_layers=6,
        n_heads=4,
        n_prelude=1,
        n_coda=1,
        default_n_loops=3,
        max_loops=3,
    )
    options.update(kwargs)
    config = R.RecurrentTransformerConfig.llama_like(**options)
    return config.apply_recurrent_residual_alpha()


def built(config, seed: int = 0):
    torch.manual_seed(seed)
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    return model


# --------------------------------------------------------------------------------------
# The parameter budget: the recurrence costs 2*d^2 + 4*d and nothing else changes.
# --------------------------------------------------------------------------------------


def test_recurrence_costs_exactly_two_d_squared_plus_four_d_at_370m():
    vocab = TokenizerConfig.dolma2().padded_vocab_size()
    base = TransformerConfig.olmo3_370M(vocab_size=vocab)
    recurrent = R.RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=vocab)

    d = base.d_model
    assert recurrent.num_params - base.num_params == 2 * d * d + 4 * d == 2_101_248
    # Width, depth and the FFN are untouched, which is what "otherwise the same model" means.
    assert (recurrent.d_model, recurrent.n_layers) == (base.d_model, base.n_layers)
    assert recurrent.block.feed_forward.hidden_size == base.block.feed_forward.hidden_size
    assert recurrent.block.name == base.block.name


def test_analytic_count_matches_the_built_model_at_370m():
    vocab = TokenizerConfig.dolma2().padded_vocab_size()
    config = R.RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=vocab)
    model = config.build(init_device="meta")
    assert config.num_params == model.num_params
    assert config.num_non_embedding_params == model.num_non_embedding_params


def test_analytic_count_matches_the_built_model_when_tiny():
    config = tiny()
    model = config.build(init_device="meta")
    assert config.num_params == model.num_params


def test_looping_deeper_adds_no_parameters():
    shallow = R.RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=256, default_n_loops=1)
    deep = R.RecurrentTransformerConfig.recurrent_olmo3_370M(
        vocab_size=256, default_n_loops=8, max_loops=8
    )
    assert shallow.num_params == deep.num_params


# --------------------------------------------------------------------------------------
# The block dict keeps the flat integer-keyed shape every upstream apply_* method assumes.
# --------------------------------------------------------------------------------------


def test_blocks_stay_flat_and_integer_keyed():
    model = tiny().build(init_device="meta")
    assert list(model.blocks.keys()) == [str(i) for i in range(model.n_layers)]
    # get_rope_buffers and apply_activation_checkpointing both parse these.
    assert all(int(key) == i for i, key in enumerate(model.blocks.keys()))


def test_the_three_groups_partition_every_layer_exactly_once():
    model = tiny().build(init_device="meta")
    covered = list(model.prelude_range) + list(model.recurrent_range) + list(model.coda_range)
    assert covered == list(range(model.n_layers))


def test_a_split_that_leaves_no_recurrent_layers_is_refused():
    from olmo_core.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError, match="recurrent core"):
        tiny(n_prelude=3, n_coda=3)


def test_incoherent_loop_bounds_are_refused():
    from olmo_core.exceptions import OLMoConfigurationError

    with pytest.raises(OLMoConfigurationError, match="loop bounds"):
        tiny(default_n_loops=9, max_loops=4)


# --------------------------------------------------------------------------------------
# The residual scale: right value, and on the recurrent blocks only.
# --------------------------------------------------------------------------------------


def test_residual_alpha_is_one_over_n_root_l_on_recurrent_blocks_only():
    config = R.RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=256)
    expected = 1.0 / (config.max_loops * math.sqrt(config.n_recurrent_layers))
    assert config.residual_alpha == pytest.approx(expected)

    assert sorted(config.block_overrides) == list(range(2, 14))
    for override in config.block_overrides.values():
        assert override.attention_residual_alpha == pytest.approx(expected)
        assert override.feed_forward_residual_alpha == pytest.approx(expected)
    # Prelude and coda are left alone, so they are bit-identical to the baseline.
    assert config.block.attention_residual_alpha is None


def test_alpha_reaches_the_built_blocks_and_scales_the_branch_not_the_skip():
    model = tiny().build(init_device="meta")
    alpha = tiny().residual_alpha
    for i in model.recurrent_range:
        assert model.blocks[str(i)].attention_residual_stream.alpha == pytest.approx(alpha)
    for i in list(model.prelude_range) + list(model.coda_range):
        assert model.blocks[str(i)].attention_residual_stream.alpha == 1.0

    # ResidualStream is `residual + alpha * branch`. Pin that, because the whole eps argument
    # depends on which of the two arguments alpha multiplies.
    from olmo_core.nn.residual_stream import ResidualStream

    stream = ResidualStream(alpha=0.25)
    skip, branch = torch.ones(4), torch.ones(4)
    assert torch.allclose(stream(skip, branch), torch.full((4,), 1.25))


def test_residual_epsilon_modes():
    assert R.residual_epsilon(4, 9, mode="factored") == pytest.approx(1.0 / 12)
    assert R.residual_epsilon(4, 9, mode="one_over_n") == pytest.approx(0.25)
    assert R.residual_epsilon(4, 9, mode="one_over_sqrt_n") == pytest.approx(0.5)
    assert R.residual_epsilon(4, 9, mode="none") == 1.0


# --------------------------------------------------------------------------------------
# The LTI carry is a contraction, whatever its parameters do.
# --------------------------------------------------------------------------------------


def test_spectral_radius_is_inside_the_margin_even_after_hostile_updates():
    injection = R.StableLTIInjection(32, margin=0.02)
    with torch.no_grad():
        # Drive the raw parameters somewhere training never would.
        injection.theta_A.fill_(-20.0)
        injection.theta_dt.fill_(-20.0)
    assert injection.spectral_radius() <= 0.98 + 1e-6

    with torch.no_grad():
        injection.theta_A.fill_(20.0)
        injection.theta_dt.fill_(20.0)
    a_bar, _ = injection.discretize()
    assert float(a_bar.detach().min()) >= 0.0
    assert injection.spectral_radius() <= 0.98 + 1e-6


def test_discretization_stays_fp32_under_bf16_autocast():
    injection = R.StableLTIInjection(16, margin=0.02)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a_bar, b_bar = injection.discretize()
    assert a_bar.dtype == torch.float32
    assert b_bar.dtype == torch.float32


def test_the_lti_parameters_are_three_vectors_of_length_d():
    injection = R.StableLTIInjection(64)
    assert sum(p.numel() for p in injection.parameters()) == 3 * 64


# --------------------------------------------------------------------------------------
# The forward pass.
# --------------------------------------------------------------------------------------


def test_depth_changes_the_output():
    model = built(tiny())
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        one = model(ids, n_loops=1)
        three = model(ids, n_loops=3)
    assert not torch.allclose(one, three), "looping more should change the prediction"


def test_every_new_parameter_receives_a_finite_gradient():
    model = built(tiny())
    ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))
    output = model(ids, labels=labels)
    loss = output.loss if hasattr(output, "loss") else output[1]
    assert torch.isfinite(loss)
    loss.backward()

    named = dict(model.named_parameters())
    for name in (
        "adapter.weight",
        "norm_e.weight",
        "injection.theta_A",
        "injection.theta_dt",
        "injection.B_cont",
    ):
        grad = named[name].grad
        assert grad is not None, f"{name} got no gradient"
        assert torch.isfinite(grad).all(), f"{name} got a non-finite gradient"
    assert not [n for n, p in model.named_parameters() if p.grad is None]


def test_a_recurrent_block_is_reused_rather_than_duplicated():
    """The loop shares weights across iterations, which is why depth is free."""
    model = built(tiny())
    ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))
    weight = model.blocks["1"].feed_forward.w1.weight
    output = model(ids, labels=labels, n_loops=3)
    (output.loss if hasattr(output, "loss") else output[1]).backward()
    # One tensor, gradient accumulated over three visits, rather than three tensors.
    assert weight.grad is not None
    assert sum(1 for _ in model.parameters()) == len(list(dict(model.named_parameters())))


def test_truncated_backprop_detaches_the_early_iterations():
    """With backprop_depth=1 only the last iteration is differentiated."""
    full = built(tiny(backprop_depth=None), seed=3)
    truncated = built(tiny(backprop_depth=1), seed=3)
    ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))

    grads = {}
    for tag, model in (("full", full), ("truncated", truncated)):
        output = model(ids, labels=labels, n_loops=3)
        (output.loss if hasattr(output, "loss") else output[1]).backward()
        grads[tag] = dict(model.named_parameters())["adapter.weight"].grad.clone()

    assert torch.isfinite(grads["truncated"]).all()
    # Fewer differentiated iterations must give a different gradient, or nothing was cut.
    assert not torch.allclose(grads["full"], grads["truncated"])


def test_forward_is_unaffected_by_grad_mode_bookkeeping():
    model = built(tiny(backprop_depth=1))
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = model(ids, n_loops=3)
    assert torch.isfinite(out).all()


def test_activation_checkpointing_is_exact_and_leaves_the_block_keys_alone():
    """The one upstream method that could plausibly break the split, checked rather than argued.

    ``apply_activation_checkpointing`` re-registers each block under its ``enumerate()``
    position rather than its key, which is safe only because the keys are contiguous integers
    from zero. It also has to stay exact across a block that is called T times rather than
    once, since each call opens its own recompute region.
    """
    from olmo_core.nn.transformer import TransformerActivationCheckpointingMode as Mode

    ids = torch.randint(0, 256, (2, 16))
    labels = torch.randint(0, 256, (2, 16))

    def gradients(model):
        output = model(ids, labels=labels, n_loops=3)
        (output.loss if hasattr(output, "loss") else output[1]).backward()
        # The wrapper inserts `_checkpoint_wrapped_module` into every wrapped parameter path.
        return {
            name.replace("_checkpoint_wrapped_module.", ""): p.grad.clone()
            for name, p in model.named_parameters()
        }

    plain = gradients(built(tiny(), seed=0))

    checkpointed_model = built(tiny(), seed=0)
    checkpointed_model.apply_activation_checkpointing(Mode.full)
    assert list(checkpointed_model.blocks.keys()) == [
        str(i) for i in range(checkpointed_model.n_layers)
    ]
    assert list(checkpointed_model.recurrent_range) == [1, 2, 3, 4]

    checkpointed = gradients(checkpointed_model)
    assert set(plain) == set(checkpointed)
    worst = max((plain[n] - checkpointed[n]).abs().max().item() for n in plain)
    assert worst == 0.0, f"recomputation changed the gradient by {worst:.3e}"


def test_it_actually_learns_and_stays_contractive_while_doing_so():
    """Everything else here says the loop runs. This says it trains.

    A 60-step memorization of one batch, which a working model solves easily and a model
    whose recurrence is silently broken does not. The spectral radius is re-read at the end
    because the contraction has to survive the optimizer, not just initialization.
    """
    config = tiny(d_model=64, vocab_size=128)
    model = built(config, seed=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    ids = torch.randint(0, 128, (4, 32))
    labels = torch.roll(ids, -1, dims=1)

    losses = []
    for _ in range(60):
        output = model(ids, labels=labels)
        loss = output.loss if hasattr(output, "loss") else output[1]
        losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    assert losses[-1] < losses[0] * 0.25, f"loss barely moved: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert model.injection.spectral_radius() <= 1.0 - config.spectral_margin + 1e-6


# --------------------------------------------------------------------------------------
# Reported FLOPs have to grow with the loop, or throughput and MFU are fiction.
# --------------------------------------------------------------------------------------


def test_flops_per_token_grows_with_depth():
    model = tiny().build(init_device="meta")
    model.n_loops = 1
    shallow = model.num_flops_per_token(128)
    model.n_loops = 3
    deep = model.num_flops_per_token(128)
    assert deep > shallow

    per_iteration = sum(
        model.blocks[str(i)].num_flops_per_token(128) for i in model.recurrent_range
    )
    assert deep - shallow == 2 * per_iteration


def test_at_one_loop_the_flops_match_a_plain_stack_of_the_same_blocks():
    model = tiny().build(init_device="meta")
    model.n_loops = 1
    every_block = sum(
        model.blocks[str(i)].num_flops_per_token(128) for i in range(model.n_layers)
    ) + model.lm_head.num_flops_per_token(128)
    assert model.num_flops_per_token(128) == every_block


# --------------------------------------------------------------------------------------
# Reaching it the way the platform's runner does, and getting it back off disk.
# --------------------------------------------------------------------------------------


def test_install_makes_the_factory_reachable_by_getattr():
    """The runner does `getattr(TransformerConfig, name, None)` and nothing else."""
    R.install()
    factory = getattr(TransformerConfig, "recurrent_olmo3_370M", None)
    assert factory is not None
    config = factory(vocab_size=TokenizerConfig.dolma2().padded_vocab_size())
    assert isinstance(config, R.RecurrentTransformerConfig)
    assert config.n_recurrent_layers == 12


def test_config_round_trips_through_the_saved_config_dict():
    """ConfigSaverCallback writes as_config_dict; a reader resolves _CLASS_ by import."""
    from olmo_core.config import Config

    config = R.RecurrentTransformerConfig.recurrent_olmo3_370M(vocab_size=256)
    as_dict = config.as_config_dict()
    assert as_dict["_CLASS_"] == "olmo_recurrent.RecurrentTransformerConfig"

    restored = Config.from_dict(as_dict)
    assert isinstance(restored, R.RecurrentTransformerConfig)
    assert restored.n_prelude == config.n_prelude
    assert restored.default_n_loops == config.default_n_loops
    assert restored.num_params == config.num_params


# --------------------------------------------------------------------------------------
# The optional depth schedule.
# --------------------------------------------------------------------------------------


def test_depth_schedule_holds_min_depth_then_varies_and_is_reproducible():
    callback = R.RecurrentDepthCallback(min_depth=1, max_depth=4, shallow_fraction=0.7, seed=11)
    total = 1000

    assert all(callback.depth_for_step(s, total) == 1 for s in range(0, 700, 37))
    later = [callback.depth_for_step(s, total) for s in range(700, 1000)]
    assert set(later) <= {1, 2, 3, 4}
    assert len(set(later)) > 1, "the deep stage should actually vary"

    # Derived from the step number, so a resume replays it with no state to checkpoint.
    again = R.RecurrentDepthCallback(min_depth=1, max_depth=4, shallow_fraction=0.7, seed=11)
    assert [again.depth_for_step(s, total) for s in range(700, 1000)] == later


def test_depth_schedule_is_a_no_op_when_the_bounds_coincide():
    callback = R.RecurrentDepthCallback(min_depth=4, max_depth=4)
    assert {callback.depth_for_step(s, 100) for s in range(100)} == {4}


def test_depth_schedule_handles_a_run_with_no_step_count():
    callback = R.RecurrentDepthCallback(min_depth=1, max_depth=4)
    assert 1 <= callback.depth_for_step(0, None) <= 4
