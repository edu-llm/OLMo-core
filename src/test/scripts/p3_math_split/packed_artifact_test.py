"""Integration checks for the published raw-shard format.

Set ``TOKENIZED_DIR`` to the resumable working payload directory containing
``train_meta.json`` / ``val_meta.json``. The tests skip in the container source
tree, where those local artifacts intentionally do not exist.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

from train_module import DerivedMaskTrainModule  # noqa: E402

TOKENIZED = Path(os.environ.get("TOKENIZED_DIR", "artifacts/public"))


@pytest.fixture(scope="module", params=("train", "val"))
def artifact(request):
    meta_path = TOKENIZED / f"{request.param}_meta.json"
    if not meta_path.exists():
        pytest.skip(f"tokenized artifacts unavailable: {meta_path}")
    return TOKENIZED, json.loads(meta_path.read_text()), request.param


def test_manifest_arithmetic_and_files(artifact):
    root, meta, split = artifact
    assert meta["tokens_dtype"] == "uint32"
    assert meta["byte_order"] == "little"
    assert meta["sequence_length"] == 16_384
    assert meta["packed"] is True
    assert meta["separator_ids"] == [10952, 15513, 969]
    assert sum(g["instances"] for g in meta["groups"].values()) == meta["instances"]
    assert sum(g["real_tokens"] for g in meta["groups"].values()) == meta["real_tokens"]
    assert (
        sum(g["dropped_over_length"] for g in meta["groups"].values())
        == meta["dropped_over_length"]
    )
    for group in meta["groups"].values():
        for shard in group["shards"]:
            path = root / shard["path"]
            assert path.name.startswith(f"{split}-")
            assert path.name.endswith(".u32le.bin")
            assert path.stat().st_size == shard["bytes"] == shard["tokens"] * 4
            assert shard["tokens"] % meta["sequence_length"] == 0


def test_sampled_rows_decode_to_in_range_ids_and_have_one_or_more_documents(artifact):
    root, meta, _ = artifact
    eos = meta["eos_token_id"]
    for group in meta["groups"].values():
        for shard in group["shards"]:
            a = np.memmap(root / shard["path"], mode="r", dtype="<u4")
            rows = a.reshape(-1, meta["sequence_length"])
            for i in sorted({0, len(rows) // 2, len(rows) - 1}):
                row = rows[i]
                assert int(row.max()) < 151_936
                assert np.count_nonzero(row == eos) >= 1


def test_real_derived_mask_on_sampled_packed_rows(artifact):
    root, meta, _ = artifact
    eos = meta["eos_token_id"]
    model = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    model._sep = torch.tensor(meta["separator_ids"], dtype=torch.long)
    model.eos_token_id = eos
    model.pad_token_id = meta["pad_token_id"]

    masked = real = checked = 0
    for group in meta["groups"].values():
        shard = group["shards"][0]
        rows = np.memmap(root / shard["path"], mode="r", dtype="<u4").reshape(
            -1, meta["sequence_length"]
        )
        for i in sorted({0, len(rows) // 2, len(rows) - 1}):
            ids = torch.from_numpy(np.asarray(rows[i], dtype=np.int64).copy()).unsqueeze(0)
            supervised = model.supervised_mask(ids)
            padding = model.padding_mask(ids)
            assert not torch.any(supervised & padding)
            # Every packed row has at least one fact block and one target.
            assert torch.any(supervised)
            assert torch.any(~supervised & ~padding)
            # Real EOS is supervised; only repeated-EOS tail padding is not.
            eos_positions = torch.nonzero(ids[0] == eos).flatten()
            assert torch.any(supervised[0, eos_positions] & ~padding[0, eos_positions])
            masked += int((~supervised & ~padding).sum())
            real += int((~padding).sum())
            checked += 1
    fraction = masked / real
    assert checked == 18
    assert 0.05 < fraction < 0.60, f"fact mask fraction {fraction:.2%} is implausible"

