"""What the wrapper does to the config, with the platform's corpus reader stubbed out.

``resolve_corpus`` reaches for ``edullm_data``, which only exists in the training image, so it
is replaced here with a manifest naming one shard. Everything after that point is the code
that actually runs on the platform.

Run with ``pytest -v .edullm/test_train_hyper_connections.py``.
"""

import math
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms  # noqa: E402
import hyper_connection_arms as arms  # noqa: E402
import train_hyper_connections as entry  # noqa: E402
import train_on_corpus  # noqa: E402

from olmo_core.data import NumpyDatasetDType, TokenizerConfig  # noqa: E402
from olmo_core.nn.residual_stream import (  # noqa: E402
    HC_DYNAMIC_PARAM_GLOB,
    HC_STATIC_PARAM_GLOB,
)
from olmo_core.nn.transformer import TransformerBlockType  # noqa: E402

SHARD_COUNT = 6


@pytest.fixture(autouse=True)
def stub_corpus(monkeypatch, tmp_path):
    # Several shards, because the real corpus has 41 and the held-out split needs somewhere to
    # take them from. Deliberately built out of order so the split has to sort them itself.
    shards = [tmp_path / f"part-0-{i:05d}.npy" for i in reversed(range(SHARD_COUNT))]
    for shard in shards:
        shard.write_bytes(b"")

    def resolve_corpus(*, dataset_id, version, tokenizer_id):
        del tokenizer_id
        return train_on_corpus.Corpus(
            dataset_id=dataset_id,
            version=version,
            paths=[str(s) for s in shards],
            dtype=NumpyDatasetDType.uint32,
            tokenizer=TokenizerConfig.dolma2(),
            rows=None,
        )

    monkeypatch.setattr(train_on_corpus, "resolve_corpus", resolve_corpus)


def build(argv):
    opts, overrides = entry.build_parser().parse_known_args(argv)
    return entry.build_config(opts, overrides), opts


BASE_ARGV = [
    "run",
    "--dataset-id",
    "pretrain/dolma2-10b",
    "--dataset-version",
    "v1",
    "--dataset-tokenizer",
    "tokenizer/dolma2-bpe",
    "--save-folder",
    "/tmp/checkpoints",
]


def test_defaults_are_the_370m_run_not_the_platforms_190m():
    """
    The platform's default model factory is olmo2_190M. An arm that quietly trained that
    would be hard to spot from a loss curve and impossible to compare against anything.
    """
    config, opts = build(BASE_ARGV + ["--arm", "baseline"])
    assert opts.model_factory == "hc_370M"
    assert config.model.d_model == 1024
    assert config.model.n_layers == 16
    assert opts.sequence_length == 4096
    assert opts.global_batch_size * opts.steps == pytest.approx(10e9, rel=0.05)


@pytest.mark.parametrize("name", sorted(arms.ARMS))
def test_every_arm_builds_through_the_platform_path(name: str):
    config, _ = build(BASE_ARGV + ["--arm", name])
    arm = arms.ARMS[name]

    assert not isinstance(config.model.block, dict)
    if arm.hyper_connections is None:
        assert config.model.block.name == TransformerBlockType.reordered_norm
        assert config.model.block.hyper_connections is None
    else:
        assert config.model.block.name == TransformerBlockType.hyper_connection_reordered_norm
        assert config.model.block.hyper_connections == arm.hyper_connections

    if arm.reuse_factor is None:
        assert config.model.block_reuse is None
    else:
        assert config.model.block_reuse is not None


@pytest.mark.parametrize("name", sorted(arms.ARMS))
def test_the_weight_decay_split_is_on_everywhere_except_the_arm_that_tests_it(name: str):
    config, _ = build(BASE_ARGV + ["--arm", name])
    patterns = [
        pattern
        for override in (config.train_module.optim.group_overrides or [])
        for pattern in override.params
    ]

    arm = arms.ARMS[name]
    expected = arm.hyper_connections is not None and name != "decay-everything"
    assert (HC_STATIC_PARAM_GLOB in patterns) is expected
    assert (HC_DYNAMIC_PARAM_GLOB in patterns) is expected

    # The embeddings override the platform installs must survive either way.
    assert "embeddings.weight" in patterns


def test_decay_everything_differs_from_faithful_only_in_the_optimizer():
    faithful, _ = build(BASE_ARGV + ["--arm", "faithful"])
    decayed, _ = build(BASE_ARGV + ["--arm", "decay-everything"])

    assert faithful.model.as_config_dict() == decayed.model.as_config_dict()
    assert faithful.train_module.optim.group_overrides != decayed.train_module.optim.group_overrides


def test_seed_moves_initialization_shuffle_and_the_global_rng_together():
    """
    A "seed" that only reshuffles the data measures less variance than the run really has, and
    the noise floor everything is compared against would come out too small.
    """
    zero, opts_zero = build(BASE_ARGV + ["--arm", "baseline", "--seed", "0"])
    two, opts_two = build(BASE_ARGV + ["--arm", "baseline", "--seed", "2"])

    assert two.init_seed == zero.init_seed + 2
    assert two.model.init_seed == zero.model.init_seed + 2
    assert opts_two.data_seed == opts_zero.data_seed + 2


def test_weight_decay_reaches_both_the_model_and_the_dynamic_group():
    config, _ = build(BASE_ARGV + ["--arm", "faithful", "--weight-decay", "0.1"])
    assert config.train_module.optim.weight_decay == 0.1

    dynamic = [
        override
        for override in config.train_module.optim.group_overrides
        if HC_DYNAMIC_PARAM_GLOB in override.params
    ]
    assert dynamic[0].opts == dict(weight_decay=0.1)


def test_dotted_overrides_still_reach_the_config():
    config, _ = build(BASE_ARGV + ["--arm", "faithful", "train_module.max_grad_norm=0.5"])
    assert config.train_module.max_grad_norm == 0.5


def test_held_out_shards_come_out_of_training_and_are_the_same_for_every_arm():
    """
    regmix-10b declares no validation split, so the eval set is carved from the training
    shards. It has to be the *same* carve on every arm and every seed -- arms evaluated on
    different data are not comparable, which is the one way this could quietly go wrong.
    """
    carves = []
    for argv in (
        ["--arm", "baseline"],
        ["--arm", "faithful"],
        ["--arm", "mhc", "--seed", "2"],
    ):
        config, _ = build(BASE_ARGV + argv)
        held = config.trainer.callbacks["held_out"]
        assert len(config.dataset.paths) == SHARD_COUNT - 2
        assert len(held.eval_dataset.paths) == 2
        assert not set(config.dataset.paths) & set(held.eval_dataset.paths)
        carves.append(tuple(held.eval_dataset.paths))

    assert len(set(carves)) == 1, "the held-out set moved between arms"


def test_held_out_can_be_turned_off():
    config, _ = build(BASE_ARGV + ["--arm", "baseline", "--held-out-shards", "0"])
    assert "held_out" not in config.trainer.callbacks
    assert len(config.dataset.paths) == SHARD_COUNT
    # BPB still attaches, so a run without a held-out set still reports it for training loss.
    assert "bits_per_byte" in config.trainer.callbacks


def test_bpb_is_written_beside_every_ce_loss():
    config, _ = build(BASE_ARGV + ["--arm", "faithful", "--bytes-per-token", "4.0"])
    callback = config.trainer.callbacks["bits_per_byte"]

    metrics = {
        "train/CE loss": 4.0 * math.log(2),
        "eval/held_out/CE loss": 8.0 * math.log(2),
        "throughput/device/TPS": 1234.0,
    }
    callback.pre_log_metrics(0, metrics)

    assert metrics["train/BPB"] == pytest.approx(1.0)
    assert metrics["eval/held_out/BPB"] == pytest.approx(2.0)
    assert "throughput/device/BPB" not in metrics


def test_bpb_leaves_a_non_finite_loss_alone():
    callback = hyper_connection_arms.BitsPerByteCallback()
    metrics = {"train/CE loss": float("nan")}
    callback.pre_log_metrics(0, metrics)
    assert "train/BPB" not in metrics


def test_split_refuses_to_leave_no_training_data():
    with pytest.raises(ValueError, match="still have a training set"):
        hyper_connection_arms.split_held_out(["a", "b"], 2)


def test_the_platform_attention_backend_is_used_everywhere():
    """
    olmo3_370M asks for flash-2 and the image has no flash-attn; the first rehearsal died on
    exactly that. Both factories have to come back with a backend that exists.
    """
    from olmo_core.nn.attention import AttentionBackendName

    for argv in (["--arm", "baseline"], ["--arm", "faithful", "--model-factory", "hc_rehearsal"]):
        config, opts = build(BASE_ARGV + argv)
        assert not isinstance(config.model.block, dict)
        mixer = config.model.block.sequence_mixer
        assert mixer.backend == AttentionBackendName.torch, opts.model_factory
        # The window is inert at seq 4096 and forces SDPA off its fused causal path.
        assert mixer.sliding_window is None, opts.model_factory


def test_an_unknown_arm_is_refused_at_parse_time():
    with pytest.raises(SystemExit):
        build(BASE_ARGV + ["--arm", "no-such-arm"])


def test_the_monitor_is_attached_only_for_hyper_connection_arms(monkeypatch):
    seen = {}

    def fake_train(config, opts=None):
        seen["callbacks"] = set(config.trainer.callbacks)

    monkeypatch.setattr(entry, "_train", fake_train)

    for name in ("faithful", "mhc", "tied-faithful"):
        config, opts = build(BASE_ARGV + ["--arm", name])
        entry.train(config, opts)
        assert "hyper_connections" in seen["callbacks"], name

    for name in ("baseline", "tied-baseline"):
        config, opts = build(BASE_ARGV + ["--arm", name])
        entry.train(config, opts)
        assert "hyper_connections" not in seen["callbacks"], name


def test_fail_closed_reaches_the_monitor(monkeypatch):
    seen = {}

    def fake_train(config, opts=None):
        seen["callback"] = config.trainer.callbacks["hyper_connections"]

    monkeypatch.setattr(entry, "_train", fake_train)
    config, opts = build(BASE_ARGV + ["--arm", "faithful", "--fail-closed-by-step", "150"])
    entry.train(config, opts)

    assert seen["callback"].fail_closed_by_step == 150
