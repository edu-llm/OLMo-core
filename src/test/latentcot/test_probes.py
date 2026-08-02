"""Tests for the probing utilities (PRD Phase 6.3)."""

import pytest
import torch

from olmo_core.latentcot import probes as P
from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.cot import embed_tokens, run_continuous_thoughts
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.evaluate import codi_answer_margin_fn, node_token_id
from olmo_core.nn.transformer import TransformerConfig

D_MODEL = 128


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_model():
    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
    )
    return cfg.build(init_device="cpu")


def _thoughts(model, ex, k):
    prefix = torch.tensor([ex["input_ids"][: ex["bot_pos"] + 1]])
    return run_continuous_thoughts(model, embed_tokens(model, prefix), k)[0]


def test_logit_lens_is_a_distribution(tiny_model):
    hidden = torch.randn(2, 3, D_MODEL)
    probs = P.logit_lens(tiny_model, hidden)
    assert probs.shape == (2, 3, T.PADDED_VOCAB_SIZE)
    assert torch.allclose(probs.sum(-1), torch.ones(2, 3), atol=1e-4)


def test_decodability_in_unit_range(tiny_model):
    d = P.decodability(tiny_model, torch.randn(2, 4, D_MODEL))
    assert 0.0 <= d <= 1.0


def test_superposition_mass_shape_and_range(tok, tiny_model):
    ex = encode_example(generate(num_nodes=12, branching=2, depth=3, seed=1, reachable=True), 3)
    thoughts = _thoughts(tiny_model, ex, 3)
    frontier_ids = [node_token_id(n) for n in ex["frontiers"][1]]
    mass = P.superposition_mass(tiny_model, thoughts, frontier_ids)
    assert mass.shape == thoughts.shape[:-1]
    assert (mass >= 0).all() and (mass <= 1.0 + 1e-4).all()


def test_linear_probe_beats_shuffled_control():
    # 3 well-separated Gaussian clusters -> label is linearly decodable.
    g = torch.Generator().manual_seed(0)
    means = torch.tensor([[4.0, 0, 0], [0, 4, 0], [0, 0, 4]])
    x = torch.cat([means[i] + 0.3 * torch.randn(50, 3, generator=g) for i in range(3)])
    y = torch.tensor([0] * 50 + [1] * 50 + [2] * 50)
    acc = P.linear_probe_accuracy(x, y, seed=0)
    shuffled = P.linear_probe_accuracy(x, y, shuffle_labels=True, seed=0)
    assert acc > 0.85
    assert acc > shuffled + 0.3  # real structure, not chance


def test_causal_ablation_returns_nonnegative(tok, tiny_model):
    ex = encode_example(generate(num_nodes=12, branching=2, depth=2, seed=2, reachable=True), 3)
    thoughts = _thoughts(tiny_model, ex, 3)
    margin_fn = codi_answer_margin_fn(tiny_model, ex)
    effect = P.causal_ablation_margin_change(margin_fn, thoughts, torch.randn(D_MODEL))
    assert effect >= 0.0
