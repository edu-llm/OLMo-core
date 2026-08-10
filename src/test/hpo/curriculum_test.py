from __future__ import annotations

import sys
import types
import json
from types import SimpleNamespace

import numpy as np
import pytest

# bettermap currently imports a Python <=3.13-only multiprocessing symbol on Windows.
if sys.version_info >= (3, 14):
    sys.modules.setdefault(
        "bettermap",
        SimpleNamespace(
            ordered_map_per_thread=lambda function, values, **kwargs: map(function, values)
        ),
    )

from olmo_core.hpo.curriculum import (
    ARM9_BOUNDARY_NUMERATORS,
    ARM9_PACING_ID,
    CURRICULUM_MANIFEST_SHA256,
    PARENT_MANIFEST_SHA256,
    CurriculumDataError,
    CurriculumDataLoader,
    CurriculumInputIdentity,
    ParentChunkDataset,
    build_curriculum_hpo_experiment,
    curriculum_corpus_from_reads,
    curriculum_pool_for_tokens,
    token_phase_boundaries,
    validate_complete_permutation,
)

PARENT_HASH = PARENT_MANIFEST_SHA256
ORDER_HASH = CURRICULUM_MANIFEST_SHA256


def _identity(*, order: bool = False, manifest_hash: str | None = None):
    return CurriculumInputIdentity(
        dataset_id=(
            "curriculum/opt-with-synthetic-10b" if order else "pretrain/opt-with-synthetic-10b"
        ),
        version="v1",
        group="mtld" if order else "tokens",
        profile="token-order/v1" if order else "pretrain-tokens/v1",
        manifest_sha256=manifest_hash or (ORDER_HASH if order else PARENT_HASH),
        source_ids=() if order else ("source-a", "source-b"),
    )


def _parent(tmp_path, *, chunks: int = 100) -> ParentChunkDataset:
    path = tmp_path / "source-a" / "train-00000.u32le.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.arange(chunks * 4 + 1, dtype="<u4").tofile(path)
    return ParentChunkDataset([path], sequence_length=4, dtype="<u4")


def _loader(
    tmp_path,
    *,
    global_batch_size: int = 8,
    parent_identity: CurriculumInputIdentity | None = None,
    order_identity: CurriculumInputIdentity | None = None,
) -> CurriculumDataLoader:
    parent = _parent(tmp_path)
    return CurriculumDataLoader(
        parent,
        ranked_chunk_indices=np.arange(len(parent), dtype=np.int64),
        pacing=ARM9_PACING_ID,
        difficulty_metric="mtld",
        seed=42,
        target_tokens=2_384,
        global_batch_size=global_batch_size,
        work_dir=tmp_path / "loader",
        parent_identity=parent_identity or _identity(),
        order_identity=order_identity or _identity(order=True),
        pad_token_id=0,
        vocab_size=1_000,
    )


def test_complete_permutation_rejects_missing_duplicate_and_out_of_range_indices():
    validate_complete_permutation(np.array([2, 0, 1]), 3)
    for invalid in (
        np.array([0, 1]),
        np.array([0, 1, 1]),
        np.array([0, 1, 3]),
        np.array([0.0, 1.0, 2.0]),
    ):
        with pytest.raises(CurriculumDataError):
            validate_complete_permutation(invalid, 3)


def test_parent_chunks_are_shard_local_and_reserve_a_next_token(tmp_path):
    paths = []
    for source, start in (("a", 0), ("b", 100)):
        path = tmp_path / source / "train-00000.u32le.bin"
        path.parent.mkdir()
        np.arange(start, start + 9, dtype="<u4").tofile(path)
        paths.append(path)

    dataset = ParentChunkDataset(paths, sequence_length=4, dtype="<u4")

    assert len(dataset) == 4
    assert dataset.source_ids == ("a", "b")
    assert dataset[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert dataset[2]["input_ids"].tolist() == [100, 101, 102, 103]
    assert dataset[-1]["input_ids"].tolist() == [104, 105, 106, 107]


def test_arm9_boundaries_use_exact_integer_token_fractions_for_every_batch_size():
    target_tokens = 503_316_480
    boundaries = token_phase_boundaries(target_tokens)

    assert ARM9_BOUNDARY_NUMERATORS == (0, 18, 54, 109, 182, 273, 382, 509, 654, 818, 1000)
    assert boundaries[0] == 0
    assert boundaries[-1] == (1000 * target_tokens + 2383) // 2384
    for bucket, boundary in enumerate(boundaries[1:-1], start=1):
        assert curriculum_pool_for_tokens(boundary - 1, 100, target_tokens)[0] == (bucket - 1) * 10
        assert curriculum_pool_for_tokens(boundary, 100, target_tokens)[0] == bucket * 10

    for batch_size in (256 * 1024, 512 * 1024, 1024 * 1024):
        for tokens_seen in range(0, target_tokens, batch_size):
            start, end = curriculum_pool_for_tokens(tokens_seen, 100, target_tokens)
            assert start % 10 == 0
            assert end - start in (10, 100)


def test_sampling_is_deterministic_and_resume_continues_at_the_next_token_batch(tmp_path):
    loader = _loader(tmp_path)
    loader.reshuffle()
    first = loader.global_indices_for_tokens(0)
    assert np.array_equal(first, loader.global_indices_for_tokens(0))
    assert set(first) <= set(range(10))

    iterator = iter(loader)
    next(iterator)
    next(iterator)
    state = loader.state_dict()
    expected = loader.global_indices_for_tokens(16)

    resumed = _loader(tmp_path)
    resumed.load_state_dict(state)
    assert resumed.batches_processed == 2
    assert resumed.global_train_tokens_seen == 16
    assert np.array_equal(resumed.global_indices_for_tokens(16), expected)
    assert next(iter(resumed))["index"].tolist() == expected.tolist()


def test_resume_refuses_changed_identity_or_inconsistent_token_progress(tmp_path):
    loader = _loader(tmp_path)
    loader.reshuffle()
    next(iter(loader))
    state = loader.state_dict()

    changed = _loader(
        tmp_path,
        order_identity=_identity(order=True, manifest_hash="c" * 64),
    )
    with pytest.raises(CurriculumDataError, match="identity"):
        changed.load_state_dict(state)

    inconsistent = dict(state)
    inconsistent["global_train_tokens_seen"] += 1
    with pytest.raises(CurriculumDataError, match="next batch"):
        _loader(tmp_path).load_state_dict(inconsistent)


def test_read_contract_uses_existing_parent_train_val_and_pinned_mtld_order():
    parent = SimpleNamespace(
        paths=["/data/tokens/source-a/train-00000.bin"],
        val=["/data/tokens/source-a/val-00000.bin"],
        dtype="uint32",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=PARENT_HASH,
    )
    order = SimpleNamespace(
        paths=["/data/mtld/train-00000.bin"],
        dtype="uint32",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=ORDER_HASH,
    )

    corpus = curriculum_corpus_from_reads(parent, order)

    assert corpus.train_paths == tuple(parent.paths)
    assert corpus.val_paths == tuple(parent.val)
    assert corpus.order_paths == tuple(order.paths)
    assert corpus.parent_identity.manifest_sha256 == PARENT_HASH
    assert corpus.order_identity.manifest_sha256 == ORDER_HASH
    assert corpus.parent_identity.source_ids == ("source-a",)

    order.manifest_sha256 = "c" * 64
    with pytest.raises(CurriculumDataError, match="manifest"):
        curriculum_corpus_from_reads(parent, order)


def test_read_contract_keeps_token_and_permutation_dtypes_independent():
    parent = SimpleNamespace(
        paths=["/data/tokens/source-a/train-00000.bin"],
        val=["/data/tokens/source-a/val-00000.bin"],
        dtype="uint32",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=PARENT_HASH,
    )
    order = SimpleNamespace(
        paths=["/data/mtld/train-00000.bin"],
        dtype="uint64",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=ORDER_HASH,
    )

    corpus = curriculum_corpus_from_reads(parent, order)

    assert corpus.dtype.value == "uint32"
    assert corpus.order_dtype.value == "uint64"


def test_factory_requests_exact_parent_and_order_groups(monkeypatch):
    parent = SimpleNamespace(
        paths=["/data/tokens/source-a/train-00000.bin"],
        train=["/data/tokens/source-a/train-00000.bin"],
        val=["/data/tokens/source-a/val-00000.bin"],
        dtype="uint32",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=PARENT_HASH,
    )
    order = SimpleNamespace(
        paths=["/data/mtld/train-00000.bin"],
        train=["/data/mtld/train-00000.bin"],
        val=None,
        dtype="uint64",
        byte_order=None,
        header_bytes=0,
        manifest_sha256=ORDER_HASH,
    )
    calls = []

    def dataset_paths(dataset_id, version, *, s3, group=None, **kwargs):
        calls.append((dataset_id, version, group, kwargs))
        return order if dataset_id.startswith("curriculum/") else parent

    fake_package = types.ModuleType("edullm_data")
    fake_read = types.ModuleType("edullm_data.read")
    fake_s3 = types.ModuleType("edullm_data.s3")
    fake_read.dataset_paths = dataset_paths

    class Boto3S3:
        @classmethod
        def default(cls):
            return object()

    fake_s3.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", fake_package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", fake_read)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", fake_s3)
    monkeypatch.setenv("EDULLM_DATASET_ID", "pretrain/opt-with-synthetic-10b")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "/tmp/checkpoints")

    config = build_curriculum_hpo_experiment()

    assert config.data_loader.order_dtype.value == "uint64"
    assert [(dataset_id, group) for dataset_id, _, group, _ in calls] == [
        ("pretrain/opt-with-synthetic-10b", "tokens"),
        ("pretrain/opt-with-synthetic-10b", "tokens"),
        ("curriculum/opt-with-synthetic-10b", "mtld"),
    ]


def test_synthetic_factory_loader_checkpoint_resume_end_to_end(monkeypatch, tmp_path):
    source_ids = (
        "algebraic-stack",
        "arxiv",
        "dclm",
        "nemotron-hqdqa",
        "open-web-math",
        "pes2o",
        "starcoder",
        "wiki",
    )
    train_paths = []
    val_paths = []
    for source_index, source_id in enumerate(source_ids):
        source_dir = tmp_path / source_id
        source_dir.mkdir()
        train_path = source_dir / "train.bin"
        val_path = source_dir / "val.bin"
        np.arange(source_index * 10, source_index * 10 + 9, dtype="<u4").tofile(train_path)
        np.arange(9, dtype="<u4").tofile(val_path)
        train_paths.append(str(train_path))
        val_paths.append(str(val_path))
    order_path = tmp_path / "mtld-order.u64.bin"
    np.arange(16, dtype="<u8").tofile(order_path)

    parent = SimpleNamespace(
        paths=train_paths,
        train=train_paths,
        val=val_paths,
        dtype="uint32",
        byte_order=sys.byteorder,
        header_bytes=0,
        manifest_sha256=PARENT_HASH,
    )
    order = SimpleNamespace(
        paths=[str(order_path)],
        train=[str(order_path)],
        val=None,
        dtype="uint64",
        byte_order=sys.byteorder,
        header_bytes=0,
        manifest_sha256=ORDER_HASH,
    )

    def dataset_paths(dataset_id, version, *, s3, **kwargs):
        del version, s3, kwargs
        return order if dataset_id.startswith("curriculum/") else parent

    fake_package = types.ModuleType("edullm_data")
    fake_read = types.ModuleType("edullm_data.read")
    fake_s3 = types.ModuleType("edullm_data.s3")
    fake_read.dataset_paths = dataset_paths

    class Boto3S3:
        @classmethod
        def default(cls):
            return object()

    fake_s3.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", fake_package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", fake_read)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", fake_s3)
    monkeypatch.setenv("EDULLM_DATASET_ID", "pretrain/opt-with-synthetic-10b")
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setenv("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe")
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))

    config = build_curriculum_hpo_experiment(
        sequence_length=4,
        global_batch_size=256 * 1024,
        rank_microbatch_size=4096,
        work_dir=str(tmp_path / "work"),
    )
    dataset = config.dataset.build()
    loader = config.data_loader.build(dataset)
    loader.reshuffle()
    first_batch = next(iter(loader))
    checkpoint_path = tmp_path / "loader-state.json"
    checkpoint_path.write_text(json.dumps(loader.state_dict()), encoding="utf-8")

    resumed = config.data_loader.build(config.dataset.build())
    resumed.load_state_dict(json.loads(checkpoint_path.read_text(encoding="utf-8")))
    expected_indices = resumed.global_indices_for_tokens(256 * 1024)
    resumed_batch = next(iter(resumed))

    assert first_batch["input_ids"].numel() == 256 * 1024
    assert config.curriculum_identity["parent"]["source_ids"] == list(source_ids)
    assert resumed.global_train_tokens_seen == 2 * 256 * 1024
    assert resumed_batch["index"].tolist() == expected_indices.tolist()
