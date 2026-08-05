"""Tests for the CODI loss and train-module integration (PRD Phase 4)."""

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.loss import codi_loss, vocab_manifold_reg
from olmo_core.nn.transformer import TransformerConfig

D_MODEL = 128


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:  # ImportError or download failure
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_model():
    try:
        cfg = TransformerConfig.llama_like(
            d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
        )
        return cfg.build(init_device="cpu")
    except Exception as e:  # pragma: no cover
        pytest.skip(f"could not build tiny transformer: {e}")


def _examples(k, n=2):
    return [
        encode_example(
            generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2)), k
        )
        for s in range(n)
    ]


def _grad_norm(model):
    return (
        sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)
        .sqrt()
        .item()
    )


def test_codi_loss_returns_metrics_and_grad_flows(tok, tiny_model):
    """Regression guard: optimize `.loss` (not the detached `.ce_loss`) so grads flow."""
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    loss, metrics = codi_loss(
        tiny_model, _examples(2), distill_weight=1.0, vocab_reg="R1", vocab_reg_weight=0.01
    )
    assert torch.isfinite(loss)
    assert set(metrics) == {
        "ce_teacher",
        "ce_student",
        "distill",
        "vocab_reg",
        "thought_rms",  # diagnostic only, not part of the objective
    }
    assert metrics["thought_rms"] > 0
    loss.backward()
    # a substantial gradient must reach the parameters (catches the ce_loss-detached bug)
    assert _grad_norm(tiny_model) > 0.5


def test_codi_training_reduces_ce_student(tok, tiny_model):
    torch.manual_seed(0)
    model = tiny_model
    model.train()
    examples = _examples(2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    first = last = None
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        loss, metrics = codi_loss(
            model, examples, distill_weight=1.0, vocab_reg="R1", vocab_reg_weight=0.01
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # deep K-step graph needs clipping
        opt.step()
        first = first or metrics
        last = metrics

    assert last["ce_student"] < first["ce_student"] - 1.0  # clear decrease
    assert last["ce_student"] < 2.0  # actually memorizes the answer


def test_vocab_reg_variants(tok, tiny_model):
    thoughts = torch.randn(2, 3, D_MODEL)
    assert float(vocab_manifold_reg(tiny_model, thoughts, "none").detach()) == 0.0
    r1 = float(vocab_manifold_reg(tiny_model, thoughts, "R1").detach())
    r2 = float(vocab_manifold_reg(tiny_model, thoughts, "R2").detach())
    l2 = float(vocab_manifold_reg(tiny_model, thoughts, "L2").detach())
    assert r1 > 0 and r2 > 0 and l2 > 0
    # the three penalties are distinct measures
    assert len({round(r1, 6), round(r2, 6), round(l2, 6)}) == 3


def test_r1_entropy_floor_adds_anticollapse_penalty(tok, tiny_model):
    torch.manual_seed(0)
    thoughts = torch.randn(2, 3, D_MODEL)
    base = float(vocab_manifold_reg(tiny_model, thoughts, "R1", entropy_floor=0.0).detach())
    # floor=0 is a no-op — identical to the default (unfloored) R1.
    assert base == float(vocab_manifold_reg(tiny_model, thoughts, "R1").detach())
    # a floor above the achievable entropy (max is log(vocab)) is never satisfied, so the hinge
    # adds a strictly positive anti-collapse penalty on top of the base pull.
    high = float(vocab_manifold_reg(tiny_model, thoughts, "R1", entropy_floor=1e6).detach())
    assert high > base
    # the floor only touches R1's mixture target; L2/R2 return before the entropy term.
    assert float(
        vocab_manifold_reg(tiny_model, thoughts, "L2", entropy_floor=1e6).detach()
    ) == float(vocab_manifold_reg(tiny_model, thoughts, "L2").detach())


def test_config_build_returns_codi_module(tiny_model):
    from olmo_core.latentcot.train_module import (
        CodiTransformerTrainModule,
        CodiTransformerTrainModuleConfig,
    )
    from olmo_core.optim import AdamWConfig

    cfg = CodiTransformerTrainModuleConfig(
        rank_microbatch_size=256,
        max_sequence_length=256,
        optim=AdamWConfig(lr=3e-4),
        num_continuous_thoughts=6,
        distill_weight=1.0,
        vocab_reg="R1",
        vocab_reg_weight=0.01,
    )
    module = cfg.build(tiny_model)
    assert isinstance(module, CodiTransformerTrainModule)
    assert module.num_continuous_thoughts == 6
    assert module.vocab_reg == "R1"
    assert module.vocab_reg_weight == 0.01
