"""
Tests for MoE support (``olmo_core.latentcot.moe``) and the arms on an MoE base.

Split deliberately in two:

- **CPU tests** cover the arithmetic and the plumbing. ``count_forwards`` is the divisor that
  keeps the routers' auxiliary losses comparable between a one-forward arm and a ``K+2``-forward
  one, and ``normalized_aux_losses`` must restore the weights it scales even when a step raises —
  both checkable without ever running an expert.
- **GPU tests** (``@requires_gpu``) cover a real MoE model, because every MoE path in this
  repository routes through ``olmo_core.kernels.moe``, which is Triton and therefore CUDA-only —
  ``import triton`` fails outright on macOS. The repo's own MoE tests are marked the same way.
"""

import json

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.loss import arm_loss
from olmo_core.latentcot.moe import (
    collect_router_metrics,
    count_forwards,
    describe_moe,
    finish_step,
    is_moe_model,
    normalized_aux_losses,
    reset_router_state,
)
from olmo_core.latentcot.train_driver import train_arm
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.testing import requires_gpu

from .test_train_driver import _tiny_model

D_MODEL = 128
K = 2


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture
def dataset(tok, tmp_path):
    path = tmp_path / "conversations" / "train-00000.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("w") as f:
        for s in range(6):
            ex = generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2))
            f.write(json.dumps(to_sft_record(ex)) + "\n")
    return LatentCotDataset(path, num_continuous_thoughts=K)


def _moe_config():
    """A tiny MoE rung. Needs CUDA to run a forward, but builds anywhere."""
    return TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        n_layers=2,
        n_heads=4,
        vocab_size=T.PADDED_VOCAB_SIZE,
        num_experts=4,
        top_k=2,
        expert_hidden_size=256,
        capacity_factor=2.0,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
    )


# --------------------------------------------------------------------------------------
# is_moe_model / describe_moe — must not require a forward, so these run on CPU
# --------------------------------------------------------------------------------------


def test_dense_model_is_not_moe():
    model = _tiny_model()
    assert is_moe_model(model) is False
    assert describe_moe(model) is None


def test_moe_model_is_detected_and_described():
    model = _moe_config().build(init_device="cpu")
    assert is_moe_model(model) is True
    described = describe_moe(model)
    assert described is not None
    assert described["num_experts"] == 4
    assert described["top_k"] == 2
    assert described["num_moe_blocks"] == 2
    assert described["lb_loss_weight"] is not None


def test_moe_helpers_are_noops_on_a_dense_model():
    """A dense base must be entirely unaffected — no attribute errors, no metrics."""
    model = _tiny_model()
    reset_router_state(model)  # must not raise
    finish_step(model)  # must not raise
    assert collect_router_metrics(model) == {}


# --------------------------------------------------------------------------------------
# count_forwards — the divisor that removes the arm-dependent confound
# --------------------------------------------------------------------------------------


def test_count_forwards_is_one_per_example_for_the_anchor_arms(dataset):
    examples = [dataset[i] for i in range(3)]
    assert count_forwards(examples, mode="explicit_cot") == 3
    assert count_forwards(examples, mode="no_cot") == 3


def test_count_forwards_counts_the_whole_codi_chain(dataset):
    """One teacher branch + K thought steps + one assembled student forward, per example."""
    examples = [dataset[i] for i in range(3)]
    assert count_forwards(examples, mode="codi") == 3 * (K + 2)


def test_codi_does_far_more_forwards_than_the_anchors(dataset):
    """
    The confound this divisor exists to remove: uncorrected, A2 applies K+2 times the balancing
    pressure A0 does, on exactly the A2-vs-A0 comparison gate A is defined on.
    """
    examples = [dataset[i] for i in range(3)]
    assert count_forwards(examples, mode="codi") == (K + 2) * count_forwards(
        examples, mode="explicit_cot"
    )


def test_count_forwards_is_never_zero():
    assert count_forwards([], mode="codi") == 1  # it is used as a divisor


def test_count_forwards_rejects_an_unknown_mode(dataset):
    with pytest.raises(ValueError, match="unknown arm mode"):
        count_forwards([dataset[0]], mode="latent_cot")


# --------------------------------------------------------------------------------------
# normalized_aux_losses — scales only the router weights, and always restores them
# --------------------------------------------------------------------------------------


def test_normalized_aux_losses_scales_then_restores_router_weights():
    model = _moe_config().build(init_device="cpu")
    routers = [b.feed_forward_moe.router for b in model.blocks.values()]
    before = [(r.lb_loss_weight, r.z_loss_weight) for r in routers]
    assert all(lb is not None for lb, _ in before)

    with normalized_aux_losses(model, 12):
        for router, (lb, z) in zip(routers, before):
            assert router.lb_loss_weight == pytest.approx(lb / 12)
            assert router.z_loss_weight == pytest.approx(z / 12)

    assert [(r.lb_loss_weight, r.z_loss_weight) for r in routers] == before


def test_normalized_aux_losses_restores_weights_even_if_the_body_raises():
    """A failed step must not leave the model permanently detuned."""
    model = _moe_config().build(init_device="cpu")
    routers = [b.feed_forward_moe.router for b in model.blocks.values()]
    before = [(r.lb_loss_weight, r.z_loss_weight) for r in routers]
    with pytest.raises(RuntimeError, match="boom"):
        with normalized_aux_losses(model, 12):
            raise RuntimeError("boom")
    assert [(r.lb_loss_weight, r.z_loss_weight) for r in routers] == before


@pytest.mark.parametrize("num_forwards", [0, 1])
def test_normalized_aux_losses_is_a_noop_for_a_single_forward(num_forwards):
    model = _moe_config().build(init_device="cpu")
    routers = [b.feed_forward_moe.router for b in model.blocks.values()]
    before = [(r.lb_loss_weight, r.z_loss_weight) for r in routers]
    with normalized_aux_losses(model, num_forwards):
        assert [(r.lb_loss_weight, r.z_loss_weight) for r in routers] == before


def test_normalized_aux_losses_leaves_a_disabled_term_disabled():
    """`None` means that loss is off; scaling must not switch it on."""
    cfg = _moe_config()
    model = cfg.build(init_device="cpu")
    for block in model.blocks.values():
        block.feed_forward_moe.router.z_loss_weight = None
    with normalized_aux_losses(model, 12):
        assert all(b.feed_forward_moe.router.z_loss_weight is None for b in model.blocks.values())


def test_normalized_aux_losses_is_a_noop_on_a_dense_model():
    model = _tiny_model()
    with normalized_aux_losses(model, 12):
        pass  # must not raise


# --------------------------------------------------------------------------------------
# The real thing. CUDA only.
# --------------------------------------------------------------------------------------


@requires_gpu
def test_continuous_thoughts_run_on_an_moe_model(dataset):
    from olmo_core.latentcot.cot import embed_tokens, run_continuous_thoughts

    model = _moe_config().build(init_device="cuda")
    model.train()
    ex = dataset[0]
    ids = torch.tensor([ex["input_ids"][: ex["bot_pos"] + 1]], dtype=torch.long, device="cuda")
    thoughts, embeds = run_continuous_thoughts(model, embed_tokens(model, ids), K)
    assert thoughts.shape == (1, K, D_MODEL)
    assert embeds.shape[1] == ids.shape[1] + K
    # The hook reads MoETransformerBlock's output, which is a plain tensor like the dense one.
    assert torch.isfinite(thoughts).all()


@requires_gpu
@pytest.mark.parametrize("arm_key", ["A0", "A1", "A2", "A3", "A4"])
def test_every_arm_trains_on_an_moe_model(dataset, arm_key):
    torch.manual_seed(0)
    model = _moe_config().build(init_device="cuda")
    history = train_arm(
        model, ARMS[arm_key], dataset, steps=3, batch_size=2, warmup_steps=1, log_every=1
    )
    assert len(history) == 3
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in history)


@requires_gpu
def test_moe_router_metrics_reach_the_history(dataset):
    """The expert-balance series that show routing collapse must be in the logged entry."""
    torch.manual_seed(0)
    model = _moe_config().build(init_device="cuda")
    history = train_arm(
        model, ARMS["A2"], dataset, steps=2, batch_size=2, warmup_steps=1, log_every=1
    )
    assert any(key.startswith("moe/") for key in history[0]), sorted(history[0])
    # Per-block series are dropped on purpose; only the totals are logged.
    assert not any("block" in key for key in history[0])


@requires_gpu
def test_normalized_aux_losses_shrinks_the_accumulated_balancing_loss(dataset):
    """
    The mechanism itself, on real experts: dividing the router weights must reduce the aux loss
    accumulated over a step. That is what makes the pressure comparable across arms.
    """
    examples = [dataset[i] for i in range(2)]
    cfg = _moe_config()

    def accumulated(num_forwards):
        torch.manual_seed(0)
        model = cfg.build(init_device="cuda")
        model.train()
        reset_router_state(model)
        with normalized_aux_losses(model, num_forwards):
            arm_loss(model, examples, mode="codi", distill_weight=1.0)
        return collect_router_metrics(model, reset=False).get("moe/load_balancing_loss")

    uncorrected, corrected = accumulated(1), accumulated(K + 2)
    assert uncorrected is not None and corrected is not None
    assert corrected < uncorrected


@requires_gpu
def test_moe_does_not_change_the_ce_loss(dataset):
    """
    The trap this design avoids: `loss_div_factor` would have normalized the aux loss AND the
    cross-entropy, silently rescaling the LM objective. Scaling router weights must leave the
    reported CE untouched.
    """
    examples = [dataset[i] for i in range(2)]
    cfg = _moe_config()

    def ce(num_forwards):
        torch.manual_seed(0)
        model = cfg.build(init_device="cuda")
        model.train()
        with normalized_aux_losses(model, num_forwards):
            _, metrics = arm_loss(model, examples, mode="codi", distill_weight=1.0)
        return metrics["ce_student"]

    assert ce(1) == pytest.approx(ce(K + 2), rel=1e-6)
