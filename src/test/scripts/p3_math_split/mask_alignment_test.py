"""The label mask must cover the fact block and nothing else, in token space.

``corpus_invariants_test.py`` checks the mask in *character* space. By the time OLMo-core
sees it, the mask is a per-token boolean, and the conversion is where a silent
off-by-one lives: get the direction of the shift wrong and the split arm trains on the
last fact token while skipping the first proof token, which is exactly the sort of thing
that produces a small, plausible, wrong result.

Run:
    TOKENIZED_DIR=tokenized pytest -v src/test/scripts/p3_math_split/mask_alignment_test.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

TOKENIZED = os.environ.get("TOKENIZED_DIR", "tokenized")
CORPUS = os.environ.get("CORPUS_DIR", "corpus")
SPLIT = os.environ.get("SPLIT_NAME", "train")
HDR = "I know these mathematical statements:"
SEP = "---"


def _path(suffix):
    return os.path.join(TOKENIZED, f"{SPLIT}_{suffix}")


@pytest.fixture(scope="module")
def meta():
    p = _path("meta.json")
    if not os.path.exists(p):
        pytest.skip(f"not tokenized yet: {p}")
    return json.load(open(p, encoding="utf-8"))


@pytest.fixture(scope="module")
def arrays(meta):
    return {
        "tokens": np.load(_path("tokens.npy")),
        "dense": np.load(_path("label_mask_dense.npy")),
        "split": np.load(_path("label_mask_split.npy")),
        "index": json.load(open(_path("index.json"), encoding="utf-8")),
    }


def test_all_three_arrays_are_the_same_length(arrays, meta):
    n = meta["n_instances"] * meta["sequence_length"]
    assert arrays["tokens"].shape == (n,)
    assert arrays["dense"].shape == (n,)
    assert arrays["split"].shape == (n,)


def test_masks_are_bool_and_tokens_are_uint32(arrays):
    """OLMo-core reads label masks with dtype=np.bool_ (numpy_dataset.py:607)."""
    assert arrays["dense"].dtype == np.bool_
    assert arrays["split"].dtype == np.bool_
    assert arrays["tokens"].dtype == np.uint32, "vocab 151936 does not fit in uint16"


def test_split_mask_is_a_strict_subset_of_dense(arrays):
    """Split supervises a subset of dense: the fact tokens, and only those, differ."""
    dense, split = arrays["dense"], arrays["split"]
    assert np.all(dense | split == dense), "split supervises a token dense does not"
    assert split.sum() < dense.sum(), "the masks are identical — there is no experiment"


def test_the_two_arms_differ_only_inside_fact_blocks(arrays, meta):
    """Every position where the masks disagree must be a real (non-pad) token."""
    dense, split = arrays["dense"], arrays["split"]
    differ = dense & ~split
    assert differ.sum() > 0
    # Differences must never fall on padding, which is unsupervised in both arms.
    assert np.all(dense[differ]), "a masked-only position is not supervised in dense either"
    S = meta["sequence_length"]
    for entry in arrays["index"][:200]:
        lo = entry["instance"] * S
        n_tok = entry["n_tokens"]
        assert not differ[
            lo + n_tok : lo + S
        ].any(), f"instance {entry['instance']}: arms differ on padding"
        assert differ[lo : lo + n_tok].sum() == entry["n_fact_tokens"]


def test_padding_is_unsupervised_in_both_arms(arrays, meta):
    S = meta["sequence_length"]
    for entry in arrays["index"][:500]:
        lo = entry["instance"] * S
        hi = lo + entry["n_tokens"]
        end = lo + S
        assert not arrays["dense"][hi:end].any(), "dense supervises padding"
        assert not arrays["split"][hi:end].any(), "split supervises padding"


def test_dense_supervises_every_real_token(arrays, meta):
    S = meta["sequence_length"]
    for entry in arrays["index"][:500]:
        lo = entry["instance"] * S
        hi = lo + entry["n_tokens"]
        assert arrays["dense"][lo:hi].all(), "dense skips a real token"


def test_split_mask_boundary_lands_on_the_separator():
    """Decode the first supervised token of each split instance and check where it is.

    This is the assertion that would fail on an off-by-one: the first token the split
    arm scores must be at or after the separator, never inside the last fact.
    """
    transformers = pytest.importorskip("transformers")
    meta_path = _path("meta.json")
    if not os.path.exists(meta_path):
        pytest.skip("not tokenized yet")
    meta = json.load(open(meta_path, encoding="utf-8"))
    tokens = np.load(_path("tokens.npy"))
    split = np.load(_path("label_mask_split.npy"))
    index = json.load(open(_path("index.json"), encoding="utf-8"))

    tok = transformers.AutoTokenizer.from_pretrained(meta["tokenizer"])
    S = meta["sequence_length"]
    checked = 0
    for entry in index[:50]:
        lo = entry["instance"] * S
        hi = lo + entry["n_tokens"]
        local = np.flatnonzero(split[lo:hi])
        assert local.size, f"instance {entry['instance']} has no supervised tokens"
        first = int(local[0])

        prefix = tok.decode(tokens[lo : lo + first].tolist())
        assert HDR in prefix, "the fact-block header is not in the unsupervised prefix"
        assert SEP not in prefix, (
            f"instance {entry['instance']}: the separator is inside the masked prefix, "
            f"so the split arm is not being scored on the separator it must emit"
        )
        # Everything from the first supervised token on must contain the goal.
        rest = tok.decode(tokens[lo + first : hi].tolist())
        assert "GOAL" in rest, "the GOAL line is not in the supervised region"
        checked += 1
    assert checked, "no instances checked"


def test_fact_fraction_is_in_the_design_band(meta):
    frac = meta["fact_token_fraction"]
    assert 0.05 < frac < 0.60, (
        f"fact tokens are {frac:.1%} of supervised tokens; outside the 5-60% band. "
        f"Too low and the arms barely differ; too high and the split arm has little "
        f"signal. 17-30% is the design target."
    )
