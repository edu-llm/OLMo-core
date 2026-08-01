"""Tests for the MQAR generator.

A silently-wrong generator is the worst failure mode available here: the model still trains, the
loss still falls, and the calibration produces a confident number for the wrong task. These
assert the properties the task depends on rather than that it runs.

    pytest mqar_data_test.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from mqar_data import (  # noqa: E402
    CALIBRATION_GRID,
    DISTANCE_SWEEP,
    IGNORE_INDEX,
    MQARConfig,
    make_mqar_batch,
    mqar_accuracy,
)

GRID = CALIBRATION_GRID + DISTANCE_SWEEP


@pytest.mark.parametrize("cfg", GRID, ids=lambda c: c.label)
def test_every_query_resolves_to_its_bound_value(cfg: MQARConfig):
    """
    THE correctness property: at each query position, the label must be the value that followed
    that key earlier in the sequence. If this is wrong the task is unlearnable-by-construction
    and no amount of training reveals it.
    """
    tokens, labels = make_mqar_batch(cfg, 8, torch.Generator().manual_seed(0))
    for b in range(tokens.shape[0]):
        qpos = (labels[b] != IGNORE_INDEX).nonzero().flatten().tolist()
        assert len(qpos) == cfg.num_pairs
        for p in qpos:
            key, want = tokens[b, p].item(), labels[b, p].item()
            body = tokens[b, : cfg.seq_len - cfg.num_pairs]
            hits = (body == key).nonzero().flatten().tolist()
            assert hits, f"query key {key} never appears in the body"
            assert tokens[b, hits[0] + 1].item() == want


@pytest.mark.parametrize("cfg", GRID, ids=lambda c: c.label)
def test_queries_occupy_exactly_the_final_positions(cfg: MQARConfig):
    """Answers must come strictly after every pair, or the task is not recall over a distance."""
    _, labels = make_mqar_batch(cfg, 4, torch.Generator().manual_seed(1))
    expected = set(range(cfg.seq_len - cfg.num_pairs, cfg.seq_len))
    for b in range(labels.shape[0]):
        assert set((labels[b] != IGNORE_INDEX).nonzero().flatten().tolist()) == expected


@pytest.mark.parametrize("cfg", GRID, ids=lambda c: c.label)
def test_keys_are_sampled_without_replacement(cfg: MQARConfig):
    """A repeated key makes its mapping ambiguous and silently caps achievable accuracy."""
    tokens, labels = make_mqar_batch(cfg, 8, torch.Generator().manual_seed(2))
    for b in range(tokens.shape[0]):
        qpos = (labels[b] != IGNORE_INDEX).nonzero().flatten()
        keys = tokens[b, qpos].tolist()
        assert len(set(keys)) == len(keys)


@pytest.mark.parametrize("cfg", GRID, ids=lambda c: c.label)
def test_keys_and_values_come_from_disjoint_halves(cfg: MQARConfig):
    """
    Overlapping key/value vocabularies let the model exploit token identity instead of learning
    the binding, which inflates accuracy for the wrong reason.
    """
    half = cfg.vocab_size // 2
    tokens, labels = make_mqar_batch(cfg, 8, torch.Generator().manual_seed(3))
    for b in range(tokens.shape[0]):
        qpos = (labels[b] != IGNORE_INDEX).nonzero().flatten()
        assert (tokens[b, qpos] < half).all(), "keys must be in the low half"
        assert (labels[b, qpos] >= half).all(), "values must be in the high half"


def test_query_order_is_shuffled_relative_to_pair_order():
    """
    If queries were asked in presentation order, position alone would predict the answer and the
    model could score well without doing recall at all.
    """
    cfg = MQARConfig(seq_len=256, num_pairs=16)
    tokens, labels = make_mqar_batch(cfg, 32, torch.Generator().manual_seed(4))
    in_order = 0
    for b in range(tokens.shape[0]):
        qpos = (labels[b] != IGNORE_INDEX).nonzero().flatten()
        query_keys = tokens[b, qpos].tolist()
        body = tokens[b, : cfg.seq_len - cfg.num_pairs].tolist()
        first_seen = [body.index(k) for k in query_keys]
        if first_seen == sorted(first_seen):
            in_order += 1
    assert in_order < 4, f"{in_order}/32 batches asked queries in presentation order"


def test_filler_token_is_never_a_real_value():
    """The filler must not collide with an answer, or some queries become unanswerable."""
    cfg = MQARConfig(seq_len=512, num_pairs=8)
    _, labels = make_mqar_batch(cfg, 16, torch.Generator().manual_seed(5))
    assert (labels[labels != IGNORE_INDEX] != cfg.vocab_size - 1).all()


def test_random_non_queries_flag_changes_the_filler():
    """
    Zoology's class default is True but its published configs set False, and the two are
    different tasks -- random filler can collide with keys. Pinned False in MQARConfig; this
    asserts the flag actually does something so the pin is meaningful.
    """
    base = MQARConfig(seq_len=128, num_pairs=8)
    rand = MQARConfig(seq_len=128, num_pairs=8, random_non_queries=True)
    t_pad, _ = make_mqar_batch(base, 4, torch.Generator().manual_seed(6))
    t_rnd, _ = make_mqar_batch(rand, 4, torch.Generator().manual_seed(6))
    filler = base.vocab_size - 1
    assert (t_pad == filler).sum() > 0
    assert (t_rnd == filler).sum() < (t_pad == filler).sum()


def test_distance_sweep_holds_capacity_fixed():
    """
    The whole point of the distance sweep: pair count constant, distance growing. If capacity
    moved too, a degradation could not be attributed to retention distance.
    """
    assert len({c.num_pairs for c in DISTANCE_SWEEP}) == 1
    lens = [c.seq_len for c in DISTANCE_SWEEP]
    assert lens == sorted(lens) and lens[-1] >= 8 * lens[0]


def test_generator_is_deterministic_given_a_seed():
    """Paired-seed designs require this; without it, arm differences include RNG differences."""
    cfg = MQARConfig(seq_len=128, num_pairs=8)
    a = make_mqar_batch(cfg, 4, torch.Generator().manual_seed(7))
    b = make_mqar_batch(cfg, 4, torch.Generator().manual_seed(7))
    c = make_mqar_batch(cfg, 4, torch.Generator().manual_seed(8))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_accuracy_ignores_non_query_positions():
    """
    Unmasked scoring would count the ignored positions as wrong and report near-zero for every
    model, which would look like "the task is too hard" rather than "the metric is broken".
    """
    cfg = MQARConfig(seq_len=64, num_pairs=4)
    tokens, labels = make_mqar_batch(cfg, 2, torch.Generator().manual_seed(9))
    logits = torch.zeros(2, cfg.seq_len, cfg.vocab_size)
    qpos = labels != IGNORE_INDEX
    logits.scatter_(2, labels.clamp_min(0).unsqueeze(-1), 10.0)  # predict the label everywhere
    assert mqar_accuracy(logits, labels) == pytest.approx(1.0)

    logits2 = torch.zeros(2, cfg.seq_len, cfg.vocab_size)
    logits2[..., 0] = 10.0  # always predict token 0, which is never a value
    assert mqar_accuracy(logits2, labels) == pytest.approx(0.0)
    assert qpos.sum() == 2 * cfg.num_pairs


def test_impossible_configs_are_rejected():
    with pytest.raises(ValueError, match="seq_len"):
        MQARConfig(seq_len=16, num_pairs=8)  # needs >= 24 positions
    with pytest.raises(ValueError, match="key vocabulary"):
        MQARConfig(seq_len=512, num_pairs=64, vocab_size=64)
