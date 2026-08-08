"""Tests for the continuous-thought forward path (PRD Phase 3)."""

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.cot import (
    embed_tokens,
    final_norm,
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


@pytest.fixture(scope="module")
def deep_model():
    """
    A deeper CPU transformer (8 layers) for scale tests.

    Residual-stream magnitude grows with depth, so the 2-layer ``tiny_model`` cannot show
    thought-magnitude drift at all — see ``test_thought_scale_does_not_compound_over_k``.
    """
    try:
        cfg = TransformerConfig.llama_like(
            d_model=D_MODEL, n_layers=8, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
        )
        return cfg.build(init_device="cpu")
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"could not build deep transformer on CPU: {e}")


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


def test_final_norm_applies_the_lm_head_norm(tiny_model):
    hidden = torch.randn(2, 3, D_MODEL) * 7.0
    assert torch.equal(final_norm(tiny_model, hidden), tiny_model.lm_head.norm(hidden))


def test_final_norm_passes_through_without_a_norm(tiny_model):
    """No LM head (pipeline parallelism) or a head built with layer_norm=None -> identity."""

    class _NoHead:
        lm_head = None

    class _HeadWithoutNorm:
        class lm_head:  # noqa: N801 - stand-in for e.g. NormalizedLMHead
            norm = None

    hidden = torch.randn(1, 2, D_MODEL)
    assert torch.equal(final_norm(_NoHead(), hidden), hidden)
    assert torch.equal(final_norm(_HeadWithoutNorm(), hidden), hidden)


def test_thought_scale_does_not_compound_over_k(deep_model):
    """
    Regression guard: thoughts must stay at token-embedding scale for every step of K.

    The last block's output is the PRE-final-norm residual stream, whose
    magnitude grows with depth and then compounds each time it is fed back (measured on the
    370M rung at K=10: RMS 5.8 -> 52 vs an embedding RMS of 1.0). ``final_norm`` inside
    ``run_continuous_thoughts`` is what keeps it flat, which matters because the A3/A4
    regularizers would otherwise mask the drift while unregularized A2 would not — an
    arm-dependent artifact. Needs a deep model and a real K: at 2 layers / K=2 the drift is
    too small to see.
    """
    k = 10
    prefix_ids = torch.randint(0, 1000, (1, 6))
    prefix_embeds = embed_tokens(deep_model, prefix_ids)
    thoughts, _ = run_continuous_thoughts(deep_model, prefix_embeds, k)

    rms = [float(thoughts[:, i].detach().float().pow(2).mean().sqrt()) for i in range(k)]
    embed_rms = float(prefix_embeds.detach().float().pow(2).mean().sqrt())

    # Flat in K (unnormalized, this ratio is ~5x at 8 layers and ~9x at 16).
    assert rms[-1] / rms[0] < 1.5, f"thought magnitude compounds over K: {rms}"
    # And in the same range as the embeddings they are spliced next to.
    assert all(0.2 * embed_rms < v < 5.0 * embed_rms for v in rms), f"{rms} vs {embed_rms}"


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
