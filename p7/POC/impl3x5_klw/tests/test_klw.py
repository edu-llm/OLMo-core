"""Unit tests for the weighting algebra and the pieces the acceptance checks build on.

The acceptance checks are the real gate — they run the actual Trainer against the actual mix.
These are the fast, no-model tests that pin the behaviours those checks depend on, plus the
couplings to Impl 5 and Impl 4 that would otherwise break silently when those trees change
(``klw/_impl5.py`` explains why that coupling is deliberate).

    python -m pytest tests/test_klw.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from klw import weighting                                          # noqa: E402
from klw.config_klw import (                                       # noqa: E402
    ALL_ARMS,
    CONDITION_ARMS,
    CONTROL_ARMS,
    GRAD_ACCUM,
    PER_DEVICE_BATCH,
    resolve_arm,
    variants_needed,
)

IGNORE = -100


# --------------------------------------------------------------------------- #
# the multiplier algebra
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [0.5, 1.0, 2.0, 8.0, 451.0])
def test_mean_is_exactly_one(T):
    """Without this the pedagogy:general ratio and the effective LR move with T."""
    rng = np.random.default_rng(0)
    s = rng.lognormal(size=20_000)
    med, scale = weighting.robust_stats(s)
    m = weighting.multipliers(weighting.robust_z(s, med, scale), T)
    assert abs(m.mean() - 1.0) < 1e-12


def test_multipliers_are_shift_invariant():
    """A constant added to the signal must not change any multiplier.

    The robust z-score subtracts the median, so this is really a test that the standardisation
    is applied before the softmax and not bypassed somewhere.
    """
    rng = np.random.default_rng(1)
    s = rng.normal(3.0, 2.0, 5_000)
    a = weighting.multipliers(weighting.robust_z(s, *weighting.robust_stats(s)), 2.0)
    s2 = s + 17.0
    b = weighting.multipliers(weighting.robust_z(s2, *weighting.robust_stats(s2)), 2.0)
    assert np.allclose(a, b, atol=1e-12)


def test_low_signal_gets_more_weight():
    """Low T concentrates weight on tokens the base finds *easy* — the sign matters.

    Getting this backwards would push the model *away* from the base while looking like a
    working implementation, so it is asserted rather than assumed from the minus sign.
    """
    s = np.array([0.0, 1.0, 2.0, 3.0, 4.0] * 200, dtype=float)
    m = weighting.multipliers(weighting.robust_z(s, *weighting.robust_stats(s)), 1.0)
    per = {v: m[s == v][0] for v in (0.0, 4.0)}
    assert per[0.0] > per[4.0]
    assert np.all(np.diff([m[s == v][0] for v in (0.0, 1.0, 2.0, 3.0, 4.0)]) < 0)


def test_temperature_limit_flattens():
    rng = np.random.default_rng(2)
    s = rng.lognormal(size=10_000)
    z = weighting.robust_z(s, *weighting.robust_stats(s))
    devs = [np.abs(weighting.multipliers(z, T) - 1).mean() for T in (1.0, 10.0, 100.0, 1000.0)]
    assert all(devs[i] > devs[i + 1] for i in range(len(devs) - 1))
    assert devs[-1] < 1e-2


def test_degenerate_signal_raises_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="MAD is 0"):
        weighting.robust_stats(np.ones(1000))
    with pytest.raises(ValueError, match="non-finite"):
        weighting.robust_stats(np.array([1.0, 2.0, np.nan, 4.0]))
    with pytest.raises(ValueError, match="empty"):
        weighting.robust_stats(np.zeros(0))


@pytest.mark.parametrize("T", [-1.0, 0.0, float("inf"), float("nan")])
def test_bad_temperature_raises(T):
    with pytest.raises(ValueError, match="temperature"):
        weighting.multipliers(np.zeros(10), T)


def test_ess_is_one_for_uniform_and_small_when_concentrated():
    assert weighting.describe(np.ones(1000))["ess"] == pytest.approx(1.0)
    spike = np.zeros(1000)
    spike[0] = 1000.0
    assert weighting.describe(spike)["ess"] < 0.002


# --------------------------------------------------------------------------- #
# scatter: multipliers back onto the label axis
# --------------------------------------------------------------------------- #
def test_scatter_puts_multipliers_on_unmasked_positions_in_order():
    labels = [IGNORE, IGNORE, 7, 8, IGNORE, 9]
    out = weighting.scatter_to_labels(labels, np.array([0.5, 1.5, 2.5], dtype=np.float32))
    assert out == [0.0, 0.0, 0.5, 1.5, 0.0, 2.5]


def test_scatter_general_rows_get_exactly_one():
    """Replay tokens always get 1.0 (IMPL3_HANDOFF §4.1) — never a pedagogy multiplier."""
    labels = [IGNORE, 4, 5, IGNORE]
    out = weighting.scatter_to_labels(labels, np.zeros(0, np.float32), general=True)
    assert out == [0.0, 1.0, 1.0, 0.0]


def test_scatter_length_mismatch_raises():
    labels = [IGNORE, 1, 2, 3]
    with pytest.raises(ValueError):
        weighting.scatter_to_labels(labels, np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        weighting.scatter_to_labels(labels, np.array([1.0] * 9, dtype=np.float32))


# --------------------------------------------------------------------------- #
# cache round-trip and digests
# --------------------------------------------------------------------------- #
def test_signal_cache_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    vals = rng.normal(size=25).astype(np.float32)
    offsets = np.array([0, 10, 10, 25], dtype=np.int64)      # row 1 is a general row
    cache = weighting.SignalCache(
        variant="b", values=vals, offsets=offsets,
        row_hash=np.array([1, 2, 3], dtype=np.uint64),
        is_pedagogy=np.array([True, False, True]), meta={"k": "v"})
    p = tmp_path / "c.npz"
    cache.save(p)
    back = weighting.SignalCache.load(p)
    assert back.variant == "b" and back.n_rows == 3 and back.meta == {"k": "v"}
    assert np.array_equal(back.values, vals)
    assert np.array_equal(back.row(0), vals[:10])
    assert back.row(1).size == 0
    assert back.is_pedagogy.tolist() == [True, False, True]


def test_row_digest_is_sensitive_to_ids_and_to_the_mask():
    ids = [1, 2, 3, 4, 5]
    labels = [IGNORE, IGNORE, 3, 4, 5]
    d = weighting.row_digest(ids, labels)
    assert d == weighting.row_digest(ids, labels)                       # stable
    assert d != weighting.row_digest([1, 2, 3, 4, 6], labels)           # ids matter
    assert d != weighting.row_digest(ids, [IGNORE, 2, 3, 4, 5])         # the mask matters
    # The label *values* are the ids at unmasked positions, so a digest over the mask alone is
    # the right sensitivity: same mask, same ids => same row.
    assert d == weighting.row_digest(tuple(ids), tuple(labels))


def test_content_key_excludes_nothing_it_is_given_and_changes_with_each_part():
    a = weighting.content_key("data", "model", "ref", "1024")
    assert a != weighting.content_key("data2", "model", "ref", "1024")
    assert a != weighting.content_key("data", "model", "ref2", "1024")
    assert a == weighting.content_key("data", "model", "ref", "1024")


def _fake_adapter(root: Path, payload: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_model.safetensors").write_bytes(payload)
    return root


def test_signal_key_ignores_the_reference_for_variant_a(tmp_path):
    """Regression. This exact asymmetry was a live bug that smoke_klw.py caught.

    The precompute folded the reference digest into one key shared by both variants while the
    trainer left it out for variant a, so a variant-a arm looked up a key nothing had written
    and reported "missing signal cache" for a cache that existed. Variant a's signal is
    ``−log π₀(y_t|ctx)`` and genuinely does not depend on π_SFT, so its key must not include it.
    """
    train = tmp_path / "train.jsonl"
    train.write_bytes(b'{"messages": []}\n')
    r1 = _fake_adapter(tmp_path / "ref1", b"aaaa")
    r2 = _fake_adapter(tmp_path / "ref2", b"bbbb")

    # variant a: reference is irrelevant, including absent-vs-present.
    ka = weighting.signal_key("a", train, "base", r1, 1024)
    assert ka == weighting.signal_key("a", train, "base", r2, 1024)
    assert ka == weighting.signal_key("a", train, "base", None, 1024)

    # variant b: reference is part of the key (§4.1 "keep this fixed").
    kb1 = weighting.signal_key("b", train, "base", r1, 1024)
    assert kb1 != weighting.signal_key("b", train, "base", r2, 1024)
    assert kb1 != weighting.signal_key("b", train, "base", None, 1024)

    # the two variants never collide, and data / model / max_len all still matter
    assert ka != kb1
    train.write_bytes(b'{"messages": [1]}\n')
    assert ka != weighting.signal_key("a", train, "base", r1, 1024)
    assert weighting.signal_key("a", train, "base", r1, 1024) != \
        weighting.signal_key("a", train, "other", r1, 1024)
    assert weighting.signal_key("a", train, "base", r1, 1024) != \
        weighting.signal_key("a", train, "base", r1, 512)


def test_signal_key_is_temperature_free_by_construction(tmp_path):
    """§4.1: one precompute serves a whole temperature sweep, so bT1/bT2/bT451 share a cache."""
    train = tmp_path / "t.jsonl"
    train.write_bytes(b"x\n")
    ref = _fake_adapter(tmp_path / "r", b"z")
    keys = {resolve_arm(a).name: weighting.signal_key(
        resolve_arm(a).variant, train, "base", ref, 1024)
        for a in ("bT1", "bT2", "bT451")}
    assert len(set(keys.values())) == 1, keys
    with pytest.raises(ValueError, match="variant"):
        weighting.signal_key("c", train, "base", ref, 1024)


def test_build_row_multipliers_normalises_over_the_whole_corpus_not_per_row():
    """The softmax is global (§4.1). Two rows with different signal levels must not each
    normalise to mean 1 — only the corpus as a whole does."""
    values = np.concatenate([np.full(50, 1.0), np.full(50, 5.0)]).astype(np.float32)
    cache = weighting.SignalCache(
        variant="a", values=values, offsets=np.array([0, 50, 100], dtype=np.int64),
        row_hash=np.zeros(2, dtype=np.uint64), is_pedagogy=np.ones(2, bool), meta={})
    rows, diag = weighting.build_row_multipliers(cache, 1.0)
    assert diag["multiplier"]["mean"] == pytest.approx(1.0, abs=1e-12)
    assert rows[0].mean() > 1.0 > rows[1].mean()       # low signal up, high signal down
    assert np.concatenate(rows).mean() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# the run matrix
# --------------------------------------------------------------------------- #
def test_arms_are_the_three_judge_winners_plus_the_control():
    assert CONDITION_ARMS == ("bT1", "bT2", "aT8")
    assert CONTROL_ARMS == ("bT451",)
    assert set(ALL_ARMS) == set(CONDITION_ARMS) | set(CONTROL_ARMS)
    assert [resolve_arm(a).impl3_name for a in ALL_ARMS] == \
        ["b-T1", "b-T2", "a-T8", "b-T451"]
    assert resolve_arm("bT451").role == "control"
    assert [resolve_arm(a).needs_reference for a in ALL_ARMS] == [True, True, False, True]
    assert variants_needed(ALL_ARMS) == ("a", "b")
    assert variants_needed(("aT8",)) == ("a",)


def test_batch_shape_is_d4s_and_is_not_silently_changed():
    """8 x 4 is A1's and D4's. See klw/config_klw.py for why it is not a tuning knob."""
    assert (PER_DEVICE_BATCH, GRAD_ACCUM) == (8, 4)
    assert PER_DEVICE_BATCH * GRAD_ACCUM == 32


def test_arm_validation_rejects_nonsense():
    from klw.config_klw import ArmKLW
    with pytest.raises(ValueError, match="variant"):
        ArmKLW("x", "c", 1.0)
    with pytest.raises(ValueError, match="temperature"):
        ArmKLW("x", "a", 0.0)
    with pytest.raises(KeyError):
        resolve_arm("nope")


# --------------------------------------------------------------------------- #
# the couplings that must not drift
# --------------------------------------------------------------------------- #
def test_inherits_impl5s_layout_rather_than_restating_it():
    from klw import config_klw
    from klw._impl5 import config5
    for name in ("N_BLOCKS", "N_PED", "N_GEN", "N_TRAIN", "PED_PER_BLOCK", "GEN_PER_BLOCK",
                 "CKPT_GRID", "SEED", "MAX_LEN", "BASE_MODEL"):
        assert getattr(config_klw, name) == getattr(config5, name), name
    assert config_klw.N_PED == 22_152 and config_klw.N_GEN == 7_384
    assert config_klw.N_TRAIN == 29_536 == config_klw.N_BLOCKS * 32
    assert len(config_klw.CKPT_GRID) == 22


def test_data_and_reference_arms_are_d4():
    """Every arm trains on D4's file, and variant b's pi_SFT is D4's own ckpt-923.

    Not impl4-A1: §1's definition is "a vanilla SFT run on identical data", and on Impl 5's mix
    that is D4. A gold-SFT reference would measure how far gold-SFT moved on contexts it never
    saw.
    """
    from klw import config_klw
    assert config_klw.DATA_ARM == "D4"
    assert config_klw.REFERENCE_ADAPTER_ARM == "D4"
    assert config_klw.REFERENCE_ADAPTER_STEP == 923


def test_train_file_is_shared_not_copied_per_arm():
    from klw import paths_klw
    assert paths_klw.train_file("D4").name == "socrateach_sft_train.jsonl"
    assert paths_klw.train_file("D4").parent.name == "D4"
    # ... and it lives under impl5_ssd, so no arm can train on a divergent copy.
    assert "impl5_ssd" in str(paths_klw.train_file("D4"))
    assert paths_klw.reference_adapter("D4", 923).name == "ckpt-923"
