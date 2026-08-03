"""Compatibility guards for the derived-mask artifact design.

The original tests loaded `tokens.npy` and parallel boolean mask arrays. Published
data now consists only of raw packed uint32 token shards; both arms read identical
bytes and `DerivedMaskTrainModule` locates every fact boundary at runtime. Detailed
packed-byte checks live in `packed_artifact_test.py`; this file prevents regression
to sidecars or a second arm-specific token stream.
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


@pytest.fixture(scope="module")
def train_meta():
    path = TOKENIZED / "train_meta.json"
    if not path.exists():
        pytest.skip(f"tokenized artifacts unavailable: {path}")
    return json.loads(path.read_text())


def test_no_published_mask_sidecars(train_meta):
    """The split/dense difference belongs to code, never parallel mutable bytes."""
    assert train_meta["packed"] is True
    assert not list(TOKENIZED.rglob("*label_mask*"))
    assert not list(TOKENIZED.rglob("*.npy"))


def test_both_arms_read_one_shared_token_inventory(train_meta):
    paths = [
        shard["path"]
        for group in train_meta["groups"].values()
        for shard in group["shards"]
    ]
    assert len(paths) == 6
    assert all("/train-" in path for path in paths)
    assert not any("dense" in path or "split" in path for path in paths)


def test_runtime_mask_boundary_on_real_packed_bytes(train_meta):
    group = train_meta["groups"]["metamath"]
    shard = TOKENIZED / group["shards"][0]["path"]
    rows = np.memmap(shard, mode="r", dtype="<u4").reshape(
        -1, train_meta["sequence_length"]
    )
    ids = torch.from_numpy(np.asarray(rows[0], dtype=np.int64).copy()).unsqueeze(0)

    module = DerivedMaskTrainModule.__new__(DerivedMaskTrainModule)
    module._sep = torch.tensor(train_meta["separator_ids"], dtype=torch.long)
    module.eos_token_id = train_meta["eos_token_id"]
    module.pad_token_id = train_meta["pad_token_id"]
    module.arm = "split"

    supervised = module.supervised_mask(ids)[0]
    padding = module.padding_mask(ids)[0]
    labels_live = module.label_supervision_mask(ids)[0]
    assert torch.any(supervised)
    assert torch.any(~supervised & ~padding)
    assert not torch.any(supervised & padding)

    sep = module._sep
    starts = torch.nonzero(
        (ids.unfold(1, len(sep), 1) == sep).all(dim=-1)[0]
    ).flatten()
    assert len(starts) >= 1
    for start in starts:
        assert not supervised[start : start + len(sep)].any()
        assert supervised[start + len(sep)]
        # labels[i] predicts input_ids[i+1], so score the first goal token from
        # the label position at the separator's final token.
        assert labels_live[start + len(sep) - 1]
        assert not labels_live[start + len(sep) - 2]

