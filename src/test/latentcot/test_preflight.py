"""Tests for the matched-budget dry-run / integrity checks (PRD Phase 7)."""

from dataclasses import replace

import pytest

from olmo_core.latentcot.arms import ARMS, build_arm_config
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.preflight import (
    assert_disjoint_seeds,
    assert_same_base_checkpoint,
    checkpoint_fingerprint,
    per_arm_compute,
    preflight,
)
from olmo_core.latentcot.train_module import CodiTransformerTrainModuleConfig
from olmo_core.optim import AdamWConfig

K = 4


@pytest.fixture(scope="module")
def tok():
    from olmo_core.latentcot import tokens as T

    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


def _base():
    return CodiTransformerTrainModuleConfig(
        rank_microbatch_size=256,
        max_sequence_length=256,
        optim=AdamWConfig(lr=3e-4),
        num_continuous_thoughts=K,
    )


def _examples(seeds, tok_needed=True):
    return [
        encode_example(
            generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2)), K
        )
        for s in seeds
    ]


def test_checkpoint_fingerprint_stable_and_sensitive(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model.pt").write_bytes(b"x" * 100)
    fp1 = checkpoint_fingerprint(d)
    assert fp1 == checkpoint_fingerprint(d)  # stable
    (d / "model.pt").write_bytes(b"x" * 101)  # size change
    assert checkpoint_fingerprint(d) != fp1  # sensitive
    with pytest.raises(FileNotFoundError):
        checkpoint_fingerprint(tmp_path / "does-not-exist")


def test_assert_same_base_checkpoint():
    assert_same_base_checkpoint({"A0": "h", "A2": "h", "A3": "h"})  # ok
    with pytest.raises(AssertionError):
        assert_same_base_checkpoint({"A0": "h", "A2": "different"})


def test_assert_disjoint_seeds(tok):
    train = _examples(range(4))
    test = _examples(range(100, 104))
    assert_disjoint_seeds(train, test)  # ok
    with pytest.raises(AssertionError):
        assert_disjoint_seeds(train, _examples(range(2, 6)))  # 2,3 overlap


def test_per_arm_compute_counts_k(tok):
    train = _examples(range(4))
    report = per_arm_compute(train, ["A1", "A2"])
    # CODI (A2) processes the K continuous-thought passes -> strictly more compute than no-CoT (A1)
    assert report["A2"]["forward_token_cost"] > report["A1"]["forward_token_cost"]


def test_preflight_passes_on_matched_arms(tok):
    arm_configs = {name: build_arm_config(_base(), arm) for name, arm in ARMS.items()}
    train, test = _examples(range(4)), _examples(range(100, 104))
    fps = {name: "same-base" for name in arm_configs}
    report = preflight(arm_configs, train, test, fps)
    assert report["matched_config"] and report["same_base_checkpoint"] and report["disjoint_seeds"]
    assert set(report["per_arm_compute"]) == set(ARMS)


def test_preflight_fails_on_out_of_whitelist_confound(tok):
    arm_configs = {name: build_arm_config(_base(), arm) for name, arm in ARMS.items()}
    arm_configs["A2"] = replace(arm_configs["A2"], optim=AdamWConfig(lr=1e-2))  # confound
    train, test = _examples(range(4)), _examples(range(100, 104))
    fps = {name: "same-base" for name in arm_configs}
    with pytest.raises(AssertionError):
        preflight(arm_configs, train, test, fps)
