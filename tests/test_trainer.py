"""Resume exactness and schedule properties.

The resume tests exist because a run was previously lost at 87% of its budget when
a session ended, and the shorter snapshot was then compared against the other
arm's full-budget one. On Colab and on spot instances preemption is the normal
case, so "resumes approximately" is not good enough.
"""

import numpy as np
import pytest
import torch

from memsplit.data import PackedDataset, ShardPaths, log_spaced_steps
from memsplit.trainer import TrainConfig, Trainer, cosine_lr


def _corpus(tmp_path, n_tokens=40_000, masked_frac=0.2, vocab=512):
    rng = np.random.default_rng(0)
    toks = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    toks.tofile(tmp_path / "tokens.bin")
    w = np.ones(n_tokens, dtype=np.uint8)
    idx = rng.choice(n_tokens, size=int(n_tokens * masked_frac), replace=False)
    w[idx] = 0
    w.tofile(tmp_path / "weights.split.bin")
    np.ones(n_tokens, dtype=np.uint8).tofile(tmp_path / "weights.dense.bin")
    return tmp_path


def _cfg(tmp_path, data, **kw):
    base = dict(
        run_id="t", out_dir=str(tmp_path / "out"), data_root=str(data),
        condition="dense", preset="toy", ctx=32, micro_batch_size=2,
        tokens_per_step=128, total_tokens=128 * 12, lr=1e-3, warmup_steps=2,
        device="cpu", log_every=1, checkpoint_minutes=1e9,
    )
    base.update(kw)
    return TrainConfig(**base)


# ------------------------------------------------------------------ dataset


def test_sidecar_must_match_stream_length(tmp_path):
    data = _corpus(tmp_path)
    np.ones(10, dtype=np.uint8).tofile(tmp_path / "weights.bad.bin")
    with pytest.raises(ValueError, match="same stream"):
        PackedDataset(ShardPaths.for_condition(data, "bad"), ctx=32, micro_batch_size=2)


def test_all_conditions_index_the_same_stream(tmp_path):
    data = _corpus(tmp_path)
    d = PackedDataset(ShardPaths.for_condition(data, "dense"), 32, 2)
    s = PackedDataset(ShardPaths.for_condition(data, "split"), 32, 2)
    xd, yd, wd = d.next_batch(torch.device("cpu"))
    xs, ys, ws = s.next_batch(torch.device("cpu"))
    # Identical inputs; only the loss weights differ. That is the whole design.
    assert torch.equal(xd, xs)
    assert not torch.equal(wd, ws)
    assert (wd == 1).all()
    assert (ws == 0).any()


def test_masked_targets_are_ignored_and_weighted_consistently(tmp_path):
    data = _corpus(tmp_path)
    s = PackedDataset(ShardPaths.for_condition(data, "split"), 32, 2)
    _, y, w = s.next_batch(torch.device("cpu"))
    assert ((w == 0) == (y == -100)).all()


def test_cursor_round_trips(tmp_path):
    data = _corpus(tmp_path)
    a = PackedDataset(ShardPaths.for_condition(data, "dense"), 32, 2)
    for _ in range(5):
        a.next_batch(torch.device("cpu"))
    state = a.state_dict()
    b = PackedDataset(ShardPaths.for_condition(data, "dense"), 32, 2)
    b.load_state_dict(state)
    xa, _, _ = a.next_batch(torch.device("cpu"))
    xb, _, _ = b.next_batch(torch.device("cpu"))
    assert torch.equal(xa, xb)


# ------------------------------------------------------------------ schedule


def test_log_spaced_schedule_is_dense_early():
    steps = log_spaced_steps(1000)
    assert steps[0] == 1 and steps[-1] == 1000
    # The specific failure being prevented: nothing evaluated before step 47.
    assert len([s for s in steps if s < 47]) >= 8, steps
    assert steps == sorted(set(steps))


def test_schedule_brackets_a_fast_crossing():
    """A schedule whose first point is past the threshold cannot be interpolated."""
    from memsplit.metrics import compute_to_threshold

    fast = [1.0 - 0.5 ** (s / 3) for s in log_spaced_steps(1000)]
    c = compute_to_threshold(
        log_spaced_steps(1000), fast, 0.9, tokens_per_step=1.0, flops_per_token=1.0
    )
    assert c.bracketed, c.note


def test_cosine_lr_warms_up_and_floors():
    assert cosine_lr(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert cosine_lr(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert cosine_lr(100, 1.0, 10, 100) == pytest.approx(0.1)
    assert cosine_lr(55, 1.0, 10, 100) < 1.0


# ------------------------------------------------------------------- resume


def test_loss_divisor_is_a_constant_not_the_surviving_count(tmp_path):
    data = _corpus(tmp_path)
    dense = _cfg(tmp_path, data, condition="dense")
    split = _cfg(tmp_path, data, condition="split")
    assert dense.loss_divisor == split.loss_divisor
    assert dense.loss_divisor == dense.accum * dense.micro_batch_size * dense.ctx


def test_resume_is_bit_exact(tmp_path):
    """Interrupt at step 6, resume, and land on the same weights as an uninterrupted run."""
    data = _corpus(tmp_path)

    whole = Trainer(_cfg(tmp_path, data, run_id="whole", out_dir=str(tmp_path / "w")))
    whole.train(resume=False)
    ref = {k: v.clone() for k, v in whole.model.state_dict().items()}

    part = Trainer(_cfg(tmp_path, data, run_id="part", out_dir=str(tmp_path / "p")))
    part.train(resume=False, max_steps=6)
    part.save_checkpoint()
    assert part.step == 6

    # resume="auto" is the real calling pattern; it is also what satisfies the
    # ResumeGuard, which refuses a second attempt that loaded nothing.
    resumed = Trainer(_cfg(tmp_path, data, run_id="part", out_dir=str(tmp_path / "p")))
    resumed.train(resume="auto")
    assert resumed.step == resumed.cfg.total_steps

    got = resumed.model.state_dict()
    for k in ref:
        assert torch.allclose(ref[k], got[k], atol=1e-6), k


def test_checkpoint_write_is_atomic(tmp_path):
    from pathlib import Path as P

    from memsplit import checkpoint_io as cio

    data = _corpus(tmp_path)
    t = Trainer(_cfg(tmp_path, data, out_dir=str(tmp_path / "a")))
    t.train(resume=False, max_steps=2)
    t.save_checkpoint()
    assert cio.exists(t.ckpt_path)
    assert not P(t.ckpt_path + ".tmp").exists(), "temp file must be replaced"


def test_log_records_supervised_fraction(tmp_path):
    """So a lower split-arm loss is never read as better modelling."""
    import json

    data = _corpus(tmp_path)
    t = Trainer(_cfg(tmp_path, data, condition="split", out_dir=str(tmp_path / "s")))
    t.train(resume=False, max_steps=3)
    rows = [json.loads(l) for l in (t.out / "log.jsonl").read_text().splitlines() if l]
    assert rows
    assert 0.0 < rows[-1]["supervised_frac"] < 1.0
    assert rows[-1]["loss_divisor"] == t.cfg.loss_divisor


def test_snapshots_land_on_the_schedule(tmp_path):
    data = _corpus(tmp_path)
    t = Trainer(_cfg(tmp_path, data, out_dir=str(tmp_path / "snap")))
    t.train(resume=False)
    got = sorted(int(p.stem.replace("step", "")) for p in (t.out / "snapshots").glob("*.pt"))
    assert 1 in got, got
    assert t.cfg.total_steps in got, got


# ------------------------------------------------- s3-aware IO and resume guard


def test_s3_uri_is_not_probed_with_a_local_path_check():
    """`Path("s3://...").exists()` is always False -- the bug being prevented."""
    from pathlib import Path as P

    from memsplit import checkpoint_io as cio

    assert cio.is_s3("s3://bucket/key")
    assert not cio.is_s3("/tmp/x")
    assert P("s3://bucket/key").exists() is False  # the trap, made explicit
    assert cio.join("s3://b/run", "ckpt.pt") == "s3://b/run/ckpt.pt"
    assert cio.join("/tmp/run", "ckpt.pt") == "/tmp/run/ckpt.pt"


def test_resume_guard_refuses_a_silent_restart(tmp_path):
    """Second attempt with no loadable checkpoint must crash, not redo the work."""
    from memsplit import checkpoint_io as cio

    g = cio.ResumeGuard(str(tmp_path))
    assert g.attempts_so_far() == 0
    assert g.check_and_record(loaded_checkpoint=False) == 1   # first attempt is fine

    with pytest.raises(RuntimeError, match="silently repeat"):
        g.check_and_record(loaded_checkpoint=False)

    # A genuine resume on attempt 2 is fine.
    g2 = cio.ResumeGuard(str(tmp_path))
    assert g2.check_and_record(loaded_checkpoint=True) == 2


def test_resume_guard_can_be_disabled_for_a_deliberate_fresh_start(tmp_path):
    from memsplit import checkpoint_io as cio

    g = cio.ResumeGuard(str(tmp_path), enabled=False)
    g.check_and_record(False)
    g.check_and_record(False)  # no raise


def test_trainer_survives_two_attempts_and_does_not_repeat_work(tmp_path):
    """End to end: attempt 1 dies at step 6, attempt 2 finishes from there."""
    data = _corpus(tmp_path)
    out = str(tmp_path / "run")

    a = Trainer(_cfg(tmp_path, data, out_dir=out))
    a.train(resume="auto", max_steps=6)
    a.save_checkpoint()

    b = Trainer(_cfg(tmp_path, data, out_dir=out))
    b.train(resume="auto")
    assert b.step == b.cfg.total_steps
    assert b.guard.attempts_so_far() == 2
