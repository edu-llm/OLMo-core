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
        model,
        ARMS["A2"],
        dataset,
        steps=120,
        batch_size=3,
        lr=3e-4,
        warmup_steps=10,  # short warmup so most of the run is at peak LR
        seed=0,
        log_every=20,
    )
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]  # the arm is learning


def test_train_arm_logs_drift_tripwires(dataset):
    """History carries grad_norm and thought_rms — the early warnings for latent-path drift."""
    model = _tiny_model()
    history = train_arm(
        model, ARMS["A2"], dataset, steps=6, batch_size=2, warmup_steps=2, seed=0, log_every=1
    )
    assert all("grad_norm" in h and "thought_rms" in h for h in history)
    assert all(h["grad_norm"] >= 0 and h["thought_rms"] > 0 for h in history)


def test_train_arm_applies_warmup_schedule(dataset):
    """The fine-tune LR follows WSD: a linear warmup ramp, capped at the peak, recorded per step."""
    model = _tiny_model()
    peak = 1e-3
    history = train_arm(
        model, ARMS["A2"], dataset, steps=60, batch_size=2, lr=peak, warmup_steps=20, log_every=1
    )
    lrs = [h["lr"] for h in history]
    # linear warmup: rises from ~0 and is strictly increasing across the warmup window
    assert lrs[0] < lrs[5] < lrs[15]
    assert lrs[0] < peak  # first step is well below the peak (no full-LR step into the base)
    assert max(lrs) <= peak + 1e-9  # never overshoots the peak
    assert any(abs(lr - peak) < 1e-5 for lr in lrs)  # reaches the peak after warmup


def test_train_arm_checkpoint_policy(dataset, tmp_path):
    """Rolling checkpoints keep only the last `keep_last`; a best.pt/best.json is written."""
    model = _tiny_model()
    save_dir = tmp_path / "run"
    val = [dataset[i] for i in range(len(dataset))]
    train_arm(
        model,
        ARMS["A2"],
        dataset,
        steps=25,
        batch_size=2,
        lr=1e-3,
        warmup_steps=5,
        seed=0,
        log_every=100,
        save_dir=save_dir,
        save_every=5,
        keep_last=2,
        val_examples=val,
    )
    rolling = sorted(save_dir.glob("step*.pt"))
    assert len(rolling) == 2  # rolling window drops the oldest, keeps the 2 most recent
    assert (save_dir / "best.pt").exists()
    best = json.loads((save_dir / "best.json").read_text())
    assert set(best) == {"step", "val_acc"} and 0.0 <= best["val_acc"] <= 1.0


def test_train_arm_no_checkpoints_without_save_dir(dataset, tmp_path):
    """Default (no save_dir) writes no checkpoint artifacts — the smoke/unit path is unchanged."""
    model = _tiny_model()
    train_arm(model, ARMS["A2"], dataset, steps=6, batch_size=2, seed=0, log_every=100)
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("best.json"))


def test_train_arm_runs_for_all_modes(dataset):
    # one short run per arm mode confirms the driver drives every arm
    for key in ("A0", "A2", "A4"):  # explicit_cot, codi, codi+L2
        model = _tiny_model()
        history = train_arm(model, ARMS[key], dataset, steps=8, batch_size=2, seed=0, log_every=4)
        assert history and all("loss" in h for h in history)
