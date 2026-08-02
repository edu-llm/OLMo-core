"""Tests that generated rows satisfy the sft-conversations/v1 contract (eduLLM platform)."""

import fnmatch
import hashlib

from olmo_core.latentcot.data.encode import to_sft_record
from olmo_core.latentcot.data.graph_gen import Example, generate


def _ex(seed, reachable):
    return generate(num_nodes=12, branching=2, depth=2, seed=seed, reachable=reachable)


def _messages_wellformed(messages) -> bool:
    """Mirror the validator's check_messages_wellformed (sft_conversations_v1.py)."""
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if not isinstance(m.get("role"), str) or not m.get("role"):
            return False
        if "content" not in m:
            return False
    return True


def _dedup_key(record) -> str:
    """The validator's default sft leakage key: sha256 over each message's role + content."""
    parts = [f"{m['role']}\x1f{m['content']}" for m in record["messages"]]
    return hashlib.sha256("\x1e".join(parts).encode("utf-8")).hexdigest()


def test_sft_record_messages_are_wellformed():
    for seed in range(6):
        record = to_sft_record(_ex(seed, bool(seed % 2)))
        assert _messages_wellformed(record["messages"])
        assert [m["role"] for m in record["messages"]] == ["user", "assistant"]
        assert all(isinstance(m["content"], str) and m["content"] for m in record["messages"])


def test_record_carries_example_and_roundtrips():
    ex = _ex(1, True)
    record = to_sft_record(ex)
    assert "messages" in record
    # the extra 'messages' key is ignored by Example.from_dict; the Example is preserved
    assert Example.from_dict(record) == ex


def test_zero_train_heldout_leakage():
    # disjoint seed ranges (as gen_graph_data.py uses) -> disjoint conversations -> 0 leakage
    train = [to_sft_record(_ex(s, bool(s % 2))) for s in range(0, 20)]
    heldout = [to_sft_record(_ex(s, bool(s % 2))) for s in range(10_000_000, 10_000_020)]
    train_keys = {_dedup_key(r) for r in train}
    heldout_keys = {_dedup_key(r) for r in heldout}
    assert not (train_keys & heldout_keys)


def test_partition_globs_select_the_right_splits():
    assert fnmatch.fnmatch("train-00000.jsonl", "train-*.jsonl")
    assert not fnmatch.fnmatch("train-00000.jsonl", "heldout-*.jsonl")
    assert fnmatch.fnmatch("heldout-00000.jsonl", "heldout-*.jsonl")
    assert not fnmatch.fnmatch("heldout-00000.jsonl", "train-*.jsonl")
