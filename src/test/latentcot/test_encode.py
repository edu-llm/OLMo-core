"""Tests for latent-CoT tokenization / encoding (PRD Phase 2)."""

import json

import pytest

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.data.encode import (
    encode_example,
    render_answer,
    render_cot,
    render_question,
)
from olmo_core.latentcot.data.graph_gen import generate


@pytest.fixture(scope="module")
def tok():
    """The dolma2 tokenizer; skip the whole module's tokenizer tests if unavailable."""
    try:
        return T.load_tokenizer()
    except Exception as e:  # ImportError or network/download failure
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


def test_specials_never_collide_with_real_tokens():
    # No tokenizer needed: the four control ids live in the unused padded region.
    ids = list(T.SPECIAL_TOKENS.values())
    assert len(set(ids)) == len(ids), "special ids must be distinct"
    assert all(T.VOCAB_SIZE <= i < T.PADDED_VOCAB_SIZE for i in ids)


def test_roundtrip_decode(tok):
    for seed in range(6):
        ex = generate(num_nodes=18, branching=3, depth=3, seed=seed, reachable=bool(seed % 2))
        for render in (render_question, render_cot, render_answer):
            s = render(ex)
            assert T.decode(T.encode(s)) == s, (render.__name__, repr(s))


def test_student_structure_and_answer_only_mask(tok):
    ex = generate(num_nodes=24, branching=3, depth=4, seed=1, reachable=True)
    k = 5
    d = encode_example(ex, k)
    sid = d["input_ids"]
    assert sid[d["bot_pos"]] == T.BOT
    assert sid[d["bot_pos"] + 1 : d["bot_pos"] + 1 + k] == [T.THOUGHT] * k
    assert sid[d["bot_pos"] + 1 + k] == T.EOT
    assert sid[d["distill_pos"]] == T.DISTILL
    # supervised positions == exactly the answer span, and the first is right after <distill>
    true_pos = [i for i, m in enumerate(d["label_mask"]) if m]
    assert true_pos == list(range(len(sid) - d["answer_len"], len(sid)))
    assert min(true_pos) == d["distill_pos"] + 1  # answer is predicted from the <distill> state


def test_teacher_structure_and_cot_plus_answer_mask(tok):
    ex = generate(num_nodes=24, branching=3, depth=4, seed=2, reachable=False)
    d = encode_example(ex, 5)
    tid = d["teacher_input_ids"]
    assert tid[d["teacher_bot_pos"]] == T.BOT
    assert tid[d["teacher_distill_pos"]] == T.DISTILL
    ttrue = [i for i, m in enumerate(d["teacher_label_mask"]) if m]
    cot_positions = list(range(d["teacher_bot_pos"] + 1, d["teacher_distill_pos"] - 1))
    ans_positions = list(range(len(tid) - d["answer_len"], len(tid)))
    assert ttrue == cot_positions + ans_positions
    # a control token is never a supervised target
    assert all(tid[i] < T.VOCAB_SIZE for i in ttrue)


def test_teacher_and_student_share_question_and_suffix(tok):
    ex = generate(num_nodes=18, branching=3, depth=3, seed=3, reachable=True)
    d = encode_example(ex, 4)
    # same question prefix (bot_pos is the question length) and same trailing answer
    assert d["bot_pos"] == d["teacher_bot_pos"]
    assert d["input_ids"][: d["bot_pos"]] == d["teacher_input_ids"][: d["teacher_bot_pos"]]
    assert d["input_ids"][-d["answer_len"] :] == d["teacher_input_ids"][-d["answer_len"] :]


def test_encode_requires_positive_k():
    ex = generate(num_nodes=12, branching=2, depth=2, seed=0, reachable=True)
    with pytest.raises(ValueError):
        encode_example(ex, 0)


def test_dataset_and_collate(tok, tmp_path):
    import torch

    from olmo_core.latentcot.data.dataset import LatentCotDataset, collate

    path = tmp_path / "d.jsonl"
    with path.open("w") as f:
        for seed in range(6):
            ex = generate(num_nodes=18, branching=3, depth=3, seed=seed, reachable=bool(seed % 2))
            f.write(json.dumps(ex.to_dict()) + "\n")

    ds = LatentCotDataset(path, num_continuous_thoughts=4)
    assert len(ds) == 6

    items = [ds[i] for i in range(4)]
    batch = collate(items, pad_id=T.TOKENIZER_CONFIG.pad_token_id)
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].dtype == torch.long
    assert batch["label_mask"].dtype == torch.bool
    # right-padded to the batch's max length
    assert batch["input_ids"].shape[1] == max(len(it["input_ids"]) for it in items)
    assert batch["label_mask"].shape == batch["input_ids"].shape
    # padding positions are never supervised
    for row, it in zip(batch["label_mask"], items):
        assert row[: len(it["input_ids"])].sum().item() == sum(it["label_mask"])
        assert row[len(it["input_ids"]) :].sum().item() == 0
