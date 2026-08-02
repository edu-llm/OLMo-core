"""Tests for the continuous-thought forward path (PRD Phase 3)."""

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.cot import (
    embed_tokens,
    run_continuous_thoughts,
    student_forward,
)
from olmo_core.nn.transformer import TransformerConfig

D_MODEL = 128


@pytest.fixture(scope="module")
def tiny_model():
    """A small CPU transformer (2 layers) sized to the padded dolma2 vocab."""
    try:
        cfg = TransformerConfig.llama_like(
            d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
        )
        return cfg.build(init_device="cpu")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"could not build tiny transformer on CPU: {e}")


@pytest.mark.parametrize("k", [1, 2, 4])
def test_thought_shapes(tiny_model, k):
    batch, prefix_len = 2, 6
    prefix_ids = torch.randint(0, 1000, (batch, prefix_len))
    prefix_embeds = embed_tokens(tiny_model, prefix_ids)
    thoughts, embeds = run_continuous_thoughts(tiny_model, prefix_embeds, k)
    assert thoughts.shape == (batch, k, D_MODEL)
    assert embeds.shape == (batch, prefix_len + k, D_MODEL)
    # the running embeds is exactly the prefix followed by the K thoughts
    assert torch.equal(embeds[:, :prefix_len], prefix_embeds)


def test_run_continuous_thoughts_requires_positive_k(tiny_model):
    prefix_embeds = embed_tokens(tiny_model, torch.randint(0, 1000, (1, 4)))
    with pytest.raises(ValueError):
        run_continuous_thoughts(tiny_model, prefix_embeds, 0)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_grad_flows_through_thought_chain(tiny_model, k):
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    batch, prefix_len, suffix_len = 2, 5, 3
    prefix_ids = torch.randint(0, 1000, (batch, prefix_len))  # question <bot>
    suffix_ids = torch.randint(0, 1000, (batch, suffix_len))  # <eot> <distill> answer

    prefix_embeds = embed_tokens(tiny_model, prefix_ids)
    thoughts, embeds = run_continuous_thoughts(tiny_model, prefix_embeds, k)
    embeds.retain_grad()  # inspect gradient at the continuous-thought positions
    full = torch.cat([embeds, embed_tokens(tiny_model, suffix_ids)], dim=1)

    dummy_ids = torch.zeros(full.shape[:2], dtype=torch.long)
    labels = torch.full((batch, full.shape[1]), -100, dtype=torch.long)
    labels[:, -2] = suffix_ids[:, -1]  # supervise predicting the answer token
    out = tiny_model(dummy_ids, input_embeddings=full, labels=labels)
    out.loss.backward()

    assert thoughts.shape == (batch, k, D_MODEL)
    assert torch.isfinite(out.loss)
    # gradient reached the K continuous-thought positions -> it flows through the chain
    thought_grad = embeds.grad[:, prefix_len : prefix_len + k, :]
    assert thought_grad.abs().sum() > 0
    # ...and back to the prompt embedding table
    assert tiny_model.embeddings.weight.grad is not None


def test_student_forward_returns_loss_and_thoughts(tiny_model):
    batch, k = 2, 3
    prefix_ids = torch.randint(0, 1000, (batch, 5))
    suffix_ids = torch.randint(0, 1000, (batch, 3))
    full_len = 5 + k + 3
    labels = torch.full((batch, full_len), -100, dtype=torch.long)
    labels[:, -2] = suffix_ids[:, -1]
    out, thoughts = student_forward(tiny_model, prefix_ids, k, suffix_ids, labels=labels)
    assert thoughts.shape == (batch, k, D_MODEL)
    assert torch.isfinite(out.loss)

    # without labels the model returns logits over the full assembled sequence
    logits, thoughts2 = student_forward(tiny_model, prefix_ids, k, suffix_ids, labels=None)
    assert thoughts2.shape == (batch, k, D_MODEL)
    assert logits.shape == (batch, full_len, T.PADDED_VOCAB_SIZE)
