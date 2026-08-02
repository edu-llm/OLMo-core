"""The two arm configs must differ in exactly one setting.

Cheap insurance against the most likely way this experiment quietly stops being an
experiment: someone tunes the learning rate while debugging one arm and does not mirror
it into the other. compare_arms.py catches that after the fact from the run
fingerprints; this catches it before the GPUs are booked.

    pytest -v src/test/scripts/p3_math_split/configs_test.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

CONFIGS = Path("src/scripts/train/p3_math_split/configs")


@pytest.fixture(scope="module")
def arms():
    return {
        name: yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
        for name in ("dense", "split")
    }


def test_arm_field_matches_the_filename(arms):
    assert arms["dense"]["arm"] == "dense"
    assert arms["split"]["arm"] == "split"


def test_shared_blocks_are_identical(arms):
    dense, split = arms["dense"]["shared"], arms["split"]["shared"]
    differing = {
        k: (dense.get(k), split.get(k))
        for k in set(dense) | set(split)
        if dense.get(k) != split.get(k)
    }
    assert not differing, (
        "the arms' shared training settings differ, so any result would confound the "
        f"loss mask with these: {differing}"
    )


def test_arm_is_the_only_top_level_difference(arms):
    dense, split = arms["dense"], arms["split"]
    assert set(dense) == set(split)
    assert [k for k in dense if dense[k] != split[k]] == ["arm"]


@pytest.mark.parametrize(
    "key",
    [
        "seed",
        "sequence_length",
        "global_batch_size_sequences",
        "rank_microbatch_size_sequences",
        "learning_rate",
        "warmup_steps",
        "lr_alpha_f",
        "weight_decay",
        "betas",
        "eps",
        "max_grad_norm",
        "tie_embeddings",
    ],
)
def test_every_control_is_specified(arms, key):
    """No control may be left to a default that could change under a version bump."""
    for name, cfg in arms.items():
        assert key in cfg["shared"], f"{name}.yaml does not pin '{key}'"


def test_batching_divides_evenly(arms):
    s = arms["dense"]["shared"]
    assert s["global_batch_size_sequences"] % s["rank_microbatch_size_sequences"] == 0, (
        "global batch is not a multiple of the microbatch; gradient accumulation would "
        "differ between arms if either is retuned"
    )


def test_step_count_is_derivable(arms):
    s = arms["dense"]["shared"]
    assert s.get("max_steps") is not None or s.get("epochs") is not None, (
        "set either max_steps or epochs; total steps and total input tokens are "
        "controls and must not fall back to a library default"
    )


def test_platform_run_loads_pretrained_qwen_before_wrapping_the_model():
    """The experiment is continual pretraining, not a random-init 0.5B run.

    The platform reference builds a fresh model. Our copy must preserve the Qwen-
    specific exception: build the exact architecture, load released HF weights
    strictly, and only then hand it to the train module/FSDP setup. Omitting the
    load still trains and produces plausible curves, so pin the call and its order.
    """
    source = Path("src/scripts/train/p3_math_split/train_platform.py").read_text()
    built = source.index('config.model.build(init_device="cpu")')
    stripped = source.index("strip_attn_out_bias(model)")
    loaded = source.index("load_hf_weights(model)")
    wrapped = source.index("config.train_module.build(model)")
    assert built < stripped < loaded < wrapped


def test_platform_reader_selects_train_partition_explicitly():
    """v0.2.0 returns every manifest entry when split=None.

    A release contains train and validation shards in the same group, so relying
    on the copied reference script's "default is trainable" comment leaks eval
    into training without raising.
    """
    source = Path("src/scripts/train/p3_math_split/train_platform.py").read_text()
    assert 'dataset_paths(dataset_id, version, split="train", s3=s3)' in source
