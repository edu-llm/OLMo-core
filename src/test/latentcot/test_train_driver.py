"""Tests for the Phase 8 training-driver core (olmo_core.latentcot.train_driver)."""

import json

import pytest

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.train_driver import iter_batches, train_arm
from olmo_core.nn.transformer import TransformerConfig

D_MODEL = 128


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
    return LatentCotDataset(path, num_continuous_thoughts=2)


def _tiny_model():
    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
    )
    return cfg.build(init_device="cpu")


def test_iter_batches_shapes(dataset):
    batches = list(iter_batches(dataset, batch_size=3, steps=4, seed=0))
    assert len(batches) == 4
    assert all(len(b["examples"]) == 3 for b in batches)


def test_train_arm_reduces_loss(dataset):
    import torch

    torch.manual_seed(0)
    model = _tiny_model()
    history = train_arm(
        model, ARMS["A2"], dataset, steps=120, batch_size=3, lr=3e-4, seed=0, log_every=20
    )
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]  # the arm is learning


def test_train_arm_runs_for_all_modes(dataset):
    # one short run per arm mode confirms the driver drives every arm
    for key in ("A0", "A2", "A4"):  # explicit_cot, codi, codi+L2
        model = _tiny_model()
        history = train_arm(model, ARMS[key], dataset, steps=8, batch_size=2, seed=0, log_every=4)
        assert history and all("loss" in h for h in history)
