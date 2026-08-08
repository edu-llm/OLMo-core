"""Pure tests for frontload-cl path grouping (no OLMo / AWS deps)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[3] / ".edullm"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from frontload_cl.corpus import Refusal, group_paths_by_source, source_name_from_path  # noqa: E402
from frontload_cl import constants as C  # noqa: E402
from frontload_cl.sft_tokenize import tokenize_messages  # noqa: E402


def test_source_name_from_s3_path():
    path = (
        "s3://edullm-data/pretrain/frontload-cl-10b/v1/tokens/"
        "fineweb-edu-main/train-00000.u32le.bin"
    )
    assert source_name_from_path(path) == "fineweb-edu-main"


def test_group_paths_by_source():
    paths = [
        "s3://edullm-data/pretrain/frontload-cl-10b/v1/tokens/finewiki/train-00000.u32le.bin",
        "s3://edullm-data/pretrain/frontload-cl-10b/v1/tokens/fineweb-edu-main/train-00001.u32le.bin",
        "s3://edullm-data/pretrain/frontload-cl-10b/v1/tokens/finewiki/train-00001.u32le.bin",
    ]
    grouped = group_paths_by_source(paths)
    assert set(grouped) == {"finewiki", "fineweb-edu-main"}
    assert len(grouped["finewiki"]) == 2


def test_missing_tokens_segment_refuses():
    with pytest.raises(Refusal):
        source_name_from_path("s3://edullm-data/pretrain/frontload-cl-10b/v1/train-00000.u32le.bin")


def test_ladder_budgets():
    assert C.GLOBAL_BATCH_SIZE == 192 * 4096
    assert C.TOTAL_STEPS == C.TOTAL_TRAIN_TOKENS // C.GLOBAL_BATCH_SIZE
    assert C.WARMUP_TOKENS == C.WARMUP_STEPS * C.GLOBAL_BATCH_SIZE
    assert C.HQ_TOTAL + C.SFT_LIKE_TOTAL == C.TOTAL_TRAIN_TOKENS
    assert C.PRIMER_BLOCK + C.PRIMER_DISPERSED == C.SFT_LIKE_TOTAL


def test_experiment_hq_and_sft_like_breakdown():
    """Budgets match EXPERIMENT-early-behavior-primer.md tables."""
    assert C.HQ_FINEWEB_MAIN == 8_360_000_000
    assert C.HQ_FINEWEB_ANNEAL == 950_000_000
    assert C.HQ_FINEWIKI_TOTAL == 490_000_000
    assert C.HQ_FINEWIKI_MAIN == 440_000_000
    assert C.HQ_FINEWIKI_ANNEAL == 50_000_000
    assert C.SFT_LIKE_COSMOPEDIA == 80_000_000
    assert C.SFT_LIKE_FINEMATH == 60_000_000
    assert C.SFT_LIKE_OPENHERMES == 30_000_000
    assert C.SFT_LIKE_NATURAL_REASONING == 30_000_000
    assert abs(sum(C.SFT_LIKE_RATIOS.values()) - 1.0) < 1e-12
    assert C.HQ_FINEWIKI_RATIO == 0.05
    assert C.PEAK_LR == 7.8e-4
    assert C.WARMUP_STEPS == 472
    assert C.TOTAL_STEPS == 12_715


def test_warmup_tokens_divisible_by_global_batch():
    """LR warmup window is an integer number of optimizer steps."""
    assert C.WARMUP_TOKENS % C.GLOBAL_BATCH_SIZE == 0
    assert C.WARMUP_TOKENS // C.GLOBAL_BATCH_SIZE == C.WARMUP_STEPS


def test_primer_block_is_half_of_sft_like():
    assert C.PRIMER_BLOCK == C.SFT_LIKE_TOTAL // 2
    assert C.PRIMER_BLOCK / C.SFT_LIKE_TOTAL == 0.5


def test_dataset_ids_match_design():
    assert C.DATASET_ID == "pretrain/frontload-cl-10b"
    assert C.SFT_DATASET_ID == "sft/frontload-cl-chat-sft"
    assert C.TOKENIZER_ID == "tokenizer/dolma2-bpe"


def test_milestone_steps_primer_and_control():
    """Nominal boundaries: warmup 472, primer~599, anneal~11444 — all far from 1000s."""
    assert C.STEPS_AFTER_WARMUP == 472
    assert C.STEPS_AFTER_PRIMER_BLOCK == 472 + C.PRIMER_BLOCK // C.GLOBAL_BATCH_SIZE
    assert C.STEPS_AT_ANNEAL_START == (C.HQ_PRE_ANNEAL + C.SFT_LIKE_TOTAL) // C.GLOBAL_BATCH_SIZE

    primer = C.milestone_checkpoint_steps("primer")
    control = C.milestone_checkpoint_steps("control")
    assert primer == sorted(
        [C.STEPS_AFTER_WARMUP, C.STEPS_AFTER_PRIMER_BLOCK, C.STEPS_AT_ANNEAL_START]
    )
    assert control == sorted([C.STEPS_AFTER_WARMUP, C.STEPS_AT_ANNEAL_START])
    assert C.STEPS_AFTER_PRIMER_BLOCK not in control


def test_milestone_proximity_skips_near_periodic():
    assert C.milestone_checkpoint_steps(
        "control", milestones=[1000, 11444], proximity=100
    ) == [11444]
    assert C.milestone_checkpoint_steps(
        "control", milestones=[1950, 11444], proximity=100
    ) == [11444]
    assert 1850 in C.milestone_checkpoint_steps(
        "control", milestones=[1850, 11444], proximity=100
    )


def test_a100_hardware_defaults():
    assert C.DEFAULT_RANK_MICROBATCH_SIZE == 24 * C.SEQ_LENGTH
    assert C.DEFAULT_ATTN_BACKEND == "flash_2"
    assert C.SMOKE_STEPS == 20
    # 8 GPUs × 24 sequences = global batch 192
    assert C.GLOBAL_BATCH_SEQUENCES % 8 == 0
    assert C.DEFAULT_RANK_MICROBATCH_SIZE * 8 == C.GLOBAL_BATCH_SIZE


def test_smoke_override_logic():
    """Mirrors apply_smoke_overrides without importing the train script (heavy deps)."""
    class Opts:
        smoke = True
        steps = C.TOTAL_STEPS
        save_interval = C.DEFAULT_SAVE_INTERVAL
        rank_microbatch_size = C.DEFAULT_RANK_MICROBATCH_SIZE
        attn_backend = C.DEFAULT_ATTN_BACKEND

    opts = Opts()
    if opts.smoke and opts.steps == C.TOTAL_STEPS:
        opts.steps = C.SMOKE_STEPS
    if opts.smoke and opts.save_interval == C.DEFAULT_SAVE_INTERVAL:
        opts.save_interval = max(opts.steps + 1, C.SMOKE_STEPS + 1)
    assert opts.steps == 20
    assert opts.save_interval > opts.steps


def test_sft_hparams():
    assert C.SFT_EPOCHS == 1
    assert C.SFT_SEQ_LENGTH == C.SEQ_LENGTH
    assert C.SFT_GLOBAL_BATCH_SIZE == 64 * 4096
    assert C.SFT_PEAK_LR == 8e-5
    assert C.SFT_WEIGHT_DECAY == 0.0
    assert "<|user|>" in C.SFT_CHAT_TEMPLATE
    assert "<|assistant|>" in C.SFT_CHAT_TEMPLATE


class _FakeTok:
    """Minimal encoder: each distinct string maps to a fixed id sequence."""

    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    vocab_size = 100278

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        # Stable per-string ids so assistant spans are identifiable.
        return [abs(hash(text)) % 10_000 + 1]


def test_tokenize_messages_masks_only_assistant():
    tok = _FakeTok()
    ids, mask = tokenize_messages(
        tok,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    assert ids
    assert len(ids) == len(mask)
    # BOS + user turn are False; assistant body + closing EOS are True.
    assert mask[0] is False
    assert any(mask)
    assert mask.count(True) >= 2  # body token + eos token at minimum


def test_tokenize_messages_truncates():
    tok = _FakeTok()
    ids, mask = tokenize_messages(
        tok,
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ],
        max_seq_length=4,  # bos + user + assistant header + body
    )
    assert len(ids) == 4
    assert len(mask) == 4
    assert mask[-1] is True  # assistant body kept; closing eos truncated


def test_tokenize_messages_matches_olmo2_multiturn_boundaries():
    tok = _FakeTok()
    ids, mask = tokenize_messages(
        tok,
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ],
    )
    assert len(ids) == len(mask) == 10
    # The official template inserts one untrained newline after a non-final
    # assistant EOS, before the next user header.
    assert mask == [False, False, False, True, True, False, False, False, True, True]


def test_tokenize_conversations_writes_npy_shards(tmp_path, monkeypatch):
    import gzip
    import json

    from frontload_cl import sft_tokenize as st

    conv = tmp_path / "train-00000.jsonl.gz"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "source": "unit",
        }
    ]
    with gzip.open(conv, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    monkeypatch.setattr(st, "load_hf_tokenizer", lambda _name=None: _FakeTok())
    out = tmp_path / "tokens"
    stats = st.tokenize_conversations_to_dir(
        [str(conv)],
        out,
        max_seq_length=64,
        tokens_per_shard=1_000_000,
        seed=0,
    )
    assert stats["num_conversations"] == 1
    assert stats["num_shards"] == 1
    token_paths, mask_paths = st.find_tokenized_shards(out)
    assert len(token_paths) == 1 and len(mask_paths) == 1
    import numpy as np

    tokens = np.fromfile(token_paths[0], dtype=np.uint32)
    masks = np.fromfile(mask_paths[0], dtype=np.bool_)
    assert len(tokens) == len(masks)
    assert masks.dtype == np.bool_
    assert masks.any()
    assert Path(token_paths[0]).stat().st_size == len(tokens) * np.dtype(np.uint32).itemsize
    assert not Path(token_paths[0]).read_bytes().startswith(b"\x93NUMPY")

    # Exercise the reader used by train_sft.py, not np.load (which would accept
    # the header-bearing format that OLMo's raw memmap reader rejects).
    if sys.platform == "win32":
        import multiprocessing as mp
        import multiprocessing.context as mp_context

        original_get_context = mp.get_context
        monkeypatch.setattr(mp_context, "ForkProcess", mp_context.SpawnProcess, raising=False)
        monkeypatch.setattr(
            mp,
            "get_context",
            lambda method=None: original_get_context("spawn" if method == "fork" else method),
        )
    from olmo_core.data import (
        NumpyDatasetDType,
        NumpyFSLDatasetConfig,
        TokenizerConfig,
    )

    dataset = NumpyFSLDatasetConfig(
        paths=token_paths,
        label_mask_paths=mask_paths,
        tokenizer=TokenizerConfig.dolma2(),
        dtype=NumpyDatasetDType.uint32,
        sequence_length=4,
    ).build()
    item = dataset[0]
    assert item["input_ids"].shape == item["label_mask"].shape == (4,)

    # A second call with the same input contract reuses the committed shards.
    stats2 = st.tokenize_conversations_to_dir(
        [str(conv)],
        out,
        max_seq_length=64,
        tokens_per_shard=1_000_000,
        seed=0,
    )
    assert stats2["reused"] is True


def test_tokenize_limit_does_not_materialize_the_input(tmp_path, monkeypatch):
    from frontload_cl import sft_tokenize as st

    def rows(_paths):
        for i in range(2):
            yield {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
        raise AssertionError("tokenizer consumed beyond --tokenize-limit")

    monkeypatch.setattr(st, "iter_conversation_rows", rows)
    monkeypatch.setattr(st, "load_hf_tokenizer", lambda _name=None: _FakeTok())
    stats = st.tokenize_conversations_to_dir(
        ["unused.jsonl"],
        tmp_path / "tokens",
        limit=2,
        tokens_per_shard=1_000_000,
    )
    assert stats["num_input_conversations"] == 2
    assert stats["num_conversations"] == 2


def test_incomplete_tokenization_is_rebuilt(tmp_path, monkeypatch):
    import numpy as np
    from frontload_cl import sft_tokenize as st

    out = tmp_path / "tokens"
    out.mkdir()
    np.save(out / "token_ids_part_0000.npy", np.asarray([123], dtype=np.uint32))
    np.save(out / "labels_mask_part_0000.npy", np.asarray([True], dtype=np.bool_))

    monkeypatch.setattr(
        st,
        "iter_conversation_rows",
        lambda _paths: iter(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        ),
    )
    monkeypatch.setattr(st, "load_hf_tokenizer", lambda _name=None: _FakeTok())
    stats = st.tokenize_conversations_to_dir(
        ["unused.jsonl"],
        out,
        tokens_per_shard=1_000_000,
    )
    assert stats["reused"] is False
    tokens = np.fromfile(stats["token_paths"][0], dtype=np.uint32)
    assert tokens.tolist() != [123]


def test_changed_tokenization_contract_rebuilds(tmp_path, monkeypatch):
    from frontload_cl import sft_tokenize as st

    monkeypatch.setattr(
        st,
        "iter_conversation_rows",
        lambda _paths: iter(
            [
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ]
        ),
    )
    monkeypatch.setattr(st, "load_hf_tokenizer", lambda _name=None: _FakeTok())
    out = tmp_path / "tokens"
    first = st.tokenize_conversations_to_dir(
        ["unused.jsonl"],
        out,
        max_seq_length=64,
        tokens_per_shard=1_000_000,
    )
    second = st.tokenize_conversations_to_dir(
        ["unused.jsonl"],
        out,
        max_seq_length=32,
        tokens_per_shard=1_000_000,
    )
    assert first["reused"] is False
    assert second["reused"] is False
    assert second["max_seq_length"] == 32
