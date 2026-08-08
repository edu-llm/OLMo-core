"""What the wrapper does to the config, with the platform's corpus reader stubbed out.

``resolve_corpus`` reaches for ``edullm_data``, which only exists in the training image, so it
is replaced here with a manifest naming one shard. Everything after that point is the code
that actually runs on the platform.

Run with ``pytest -v .edullm/test_train_hyper_connections.py``.
"""

import math
import os
import pathlib
import shlex
import sys

import pytest
import yaml

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

    # By default the corpus declares its own validation split, which is what regmix-10b does:
    # seven shards, one per source. Tests that want the carve fallback override this.
    val = [str(tmp_path / src / "val-00000.npy") for src in ("arxiv", "dclm", "wiki")]
    monkeypatch.setattr(entry, "declared_validation_paths", lambda config: sorted(val))


@pytest.fixture(autouse=True)
def not_a_fanout_cell(monkeypatch):
    """
    Nothing here is a fan-out unless the test says so. Without this a developer machine that
    happened to export either variable would change what every other test in the file means.
    """
    monkeypatch.delenv(entry.FANOUT_INDEX_VARIABLE, raising=False)
    monkeypatch.delenv(entry.FANOUT_PARAMETER_VARIABLE, raising=False)


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


def test_the_default_horizon_is_the_tranche_and_not_the_full_10b():
    """
    The two horizons are both real and they are different numbers, so the default has to say
    which one it is. 12,715 steps is the 10B the experiment wants and 37.7 hours, which does
    not fit the workload's 24-hour attempt; 6,000 is what this tranche runs. The default comes
    from the arm table so that the cost model and the parser cannot drift apart.
    """
    _, opts = build(BASE_ARGV + ["--arm", "baseline"])
    assert opts.steps == arms.TRANCHE_STEPS == 6_000
    assert opts.global_batch_size * opts.steps == pytest.approx(4.72e9, rel=0.01)
    assert arms.FULL_HORIZON_STEPS * opts.global_batch_size == pytest.approx(10e9, rel=0.05)
    # Warmup follows the horizon rather than being pinned to the number it was at 12,715.
    assert opts.warmup_steps == round(arms.TRANCHE_STEPS * 0.02) == 120


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


def fanout_env(index, parameter="seed"):
    return {
        entry.FANOUT_INDEX_VARIABLE: str(index),
        entry.FANOUT_PARAMETER_VARIABLE: parameter,
    }


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a_fanout_cell_takes_its_replicate_number_from_the_array_index(index: int):
    """
    THE WHOLE POINT OF THE FAN-OUT. One submission, one command, three cells, three different
    replicates. Batch is the only thing that knows which cell a process is, and it says so in
    AWS_BATCH_JOB_ARRAY_INDEX and nowhere else.
    """
    seed, provenance = entry.resolve_seed(None, fanout_env(index))
    assert seed == index
    assert entry.FANOUT_INDEX_VARIABLE in provenance


def test_the_three_cells_of_a_fanout_are_actually_three_different_models(monkeypatch):
    """
    A test that reads the seed back is not enough: the failure this guards is three cells
    training the same model, so what has to differ is the three numbers the model, the global
    RNG and the shuffle are actually drawn from.
    """
    seen = set()
    for index in (0, 1, 2):
        for name, value in fanout_env(index).items():
            monkeypatch.setenv(name, value)
        config, opts = build(BASE_ARGV + ["--arm", "faithful"])
        seen.add((config.init_seed, config.model.init_seed, opts.data_seed))

    assert len(seen) == 3, f"cells collided onto the same seeds: {seen}"


def test_an_explicit_seed_inside_a_fanout_is_refused_rather_than_honoured(monkeypatch):
    """
    THE WORST FAILURE AVAILABLE HERE AND THE ONLY ONE THAT PRODUCES NO SYMPTOM. Every cell of
    a fan-out is handed the same command, so honouring a --seed written into run.yaml would
    run one replicate three times: three bit-identical models, a measured noise floor of
    zero, and nine curves lying exactly on top of each other in W&B, which reads as a very
    clean experiment rather than as a broken one.
    """
    with pytest.raises(train_on_corpus.Refusal) as refused:
        entry.resolve_seed(1, fanout_env(2))

    assert refused.value.stage == train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD
    assert "noise floor of zero" in refused.value.explanation

    for name, value in fanout_env(2).items():
        monkeypatch.setenv(name, value)
    with pytest.raises(train_on_corpus.Refusal):
        build(BASE_ARGV + ["--arm", "baseline", "--seed", "1"])


def test_an_index_that_does_not_mean_the_seed_is_refused():
    """
    The index is one integer and the label beside it is the only thing that says what it
    counts. A checkpoint sweep's cell number read as a replicate number is the same accident
    as the one above, arriving under a different name.
    """
    with pytest.raises(train_on_corpus.Refusal, match="only reads a fan-out index"):
        entry.resolve_seed(None, fanout_env(2, parameter="checkpoint"))
    with pytest.raises(train_on_corpus.Refusal, match="only reads a fan-out index"):
        entry.resolve_seed(None, {entry.FANOUT_INDEX_VARIABLE: "2"})
    with pytest.raises(train_on_corpus.Refusal, match="not an integer"):
        entry.resolve_seed(None, fanout_env("two"))


def test_outside_a_fanout_the_command_line_wins_and_the_default_is_zero():
    assert entry.resolve_seed(None, {})[0] == 0
    assert entry.resolve_seed(2, {})[0] == 2
    assert "--seed 2" in entry.resolve_seed(2, {})[1]


def test_the_resolved_seed_is_visible_before_anything_expensive_happens(capsys):
    """
    A silent wrong seed is unrecoverable from the run's own output, so the resolution is
    printed by build_config -- which runs on the platform and under --preflight alike, before
    the process group, the model or the corpus.
    """
    build(BASE_ARGV + ["--arm", "baseline", "--seed", "2"])
    printed = capsys.readouterr().out
    assert "seed 2" in printed
    assert entry._SEED_PROVENANCE and "seed 2" in entry._SEED_PROVENANCE[0]


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


def test_the_declared_validation_split_is_used_and_costs_no_training_data():
    """
    regmix-10b publishes seven val shards, one per source, which train_on_corpus's own comment
    says it does not. Using them keeps the full token budget, makes the split the publisher's
    rather than an arbitrary slice of ours, and arrives stratified. The eval set also has to be
    identical across arms and seeds -- arms scored on different data are not comparable, which
    is the one way this goes quietly wrong.
    """
    seen = []
    for argv in (["--arm", "baseline"], ["--arm", "faithful"], ["--arm", "mhc", "--seed", "2"]):
        config, _ = build(BASE_ARGV + argv)
        held = config.trainer.callbacks["held_out"]
        assert len(config.dataset.paths) == SHARD_COUNT, "training data was carved into"
        assert not set(config.dataset.paths) & set(held.eval_dataset.paths)
        seen.append(tuple(held.eval_dataset.paths))

    assert len(set(seen)) == 1, "the held-out set moved between arms"


def test_every_held_out_shard_is_labelled_with_its_source():
    """
    The evaluator names each metric after the shard's label and raises on a shard without one,
    which is what killed the third submission. Labelling by source is also what turns one
    pooled bits-per-byte into one per source, and a single average over arxiv, code, web text
    and Wikipedia hides the effect it is supposed to measure.
    """
    config, _ = build(BASE_ARGV + ["--arm", "faithful"])
    dataset = config.trainer.callbacks["held_out"].eval_dataset

    labels = [m["label"] for m in dataset.metadata]
    assert len(labels) == len(dataset.paths)
    assert sorted(labels) == ["arxiv", "dclm", "wiki"]


def test_a_corpus_with_no_declared_split_falls_back_to_carving(monkeypatch):
    monkeypatch.setattr(entry, "declared_validation_paths", lambda config: [])

    config, _ = build(BASE_ARGV + ["--arm", "baseline"])
    held = config.trainer.callbacks["held_out"]
    assert len(config.dataset.paths) == SHARD_COUNT - 2
    assert len(held.eval_dataset.paths) == 2
    assert all("label" in m for m in held.eval_dataset.metadata)


def test_the_held_out_dataset_is_the_shape_the_evaluator_demands():
    """
    LMEvaluator refuses anything but a padded FSL dataset, and it refuses it at build time --
    which on this platform means eighteen minutes into a run, not at submission. The first
    attempt at the held-out set died exactly there.
    """
    from olmo_core.data import NumpyPaddedFSLDatasetConfig

    config, _ = build(BASE_ARGV + ["--arm", "baseline"])
    eval_dataset = config.trainer.callbacks["held_out"].eval_dataset

    assert isinstance(eval_dataset, NumpyPaddedFSLDatasetConfig)
    assert eval_dataset.sequence_length == config.dataset.sequence_length
    assert eval_dataset.tokenizer == config.dataset.tokenizer
    assert eval_dataset.dtype == config.dataset.dtype


def test_held_out_can_be_turned_off():
    config, _ = build(BASE_ARGV + ["--arm", "baseline", "--eval-interval", "0"])
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


@pytest.mark.parametrize("factor", [1.0, 0.75, 0.5, 0.25, 0.0])
def test_partial_rotary_is_a_free_axis_across_the_whole_sweep(factor: float):
    """
    Nobody has measured in-distribution BPB against the fraction of head channels carrying
    RoPE, at any scale, with a noise floor. It costs nothing to ask: RoPE has no parameters and
    is not counted in num_flops_per_token, so an arm at 0.25 is iso-everything with one at 1.0.
    """
    baseline, _ = build(BASE_ARGV + ["--arm", "baseline"])
    swept, _ = build(BASE_ARGV + ["--arm", "baseline", "--partial-rotary-factor", str(factor)])

    assert swept.model.block.sequence_mixer.rope.partial_rotary_factor == factor
    assert swept.model.num_params == baseline.model.num_params
    assert swept.model.build(init_device="meta").num_flops_per_token(4096) == baseline.model.build(
        init_device="meta"
    ).num_flops_per_token(4096)


def test_partial_rotary_composes_with_a_hyper_connection_arm():
    config, _ = build(BASE_ARGV + ["--arm", "faithful", "--partial-rotary-factor", "0.5"])
    assert config.model.block.sequence_mixer.rope.partial_rotary_factor == 0.5
    assert config.model.block.hyper_connections is not None


def test_partial_rotary_is_untouched_unless_asked_for():
    config, _ = build(BASE_ARGV + ["--arm", "baseline"])
    assert config.model.block.sequence_mixer.rope.partial_rotary_factor == 1.0


def test_partial_rotary_refuses_a_cut_that_splits_a_rotation_pair():
    """
    RoPE rotates channels in pairs. An odd count silently leaves one channel out of the
    rotation it was supposed to be in, which is a quietly different model rather than an error.
    """
    with pytest.raises(ValueError, match="which is odd"):
        build(BASE_ARGV + ["--arm", "baseline", "--partial-rotary-factor", "0.03"])


def test_partial_rotary_refuses_an_out_of_range_factor():
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        build(BASE_ARGV + ["--arm", "baseline", "--partial-rotary-factor", "1.5"])


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


def committed_command() -> list:
    """
    The words the platform will exec, from the spec it reads them out of.

    :returns: The tokens inside the ``bash -lc`` wrapper.
    """
    spec = yaml.safe_load((pathlib.Path(_HERE) / "run.yaml").read_text())
    wrapper = shlex.split(spec["command"])
    assert wrapper[:2] == ["bash", "-lc"], wrapper[:2]
    return shlex.split(wrapper[2])


def committed_argv() -> list:
    """
    Everything the committed command passes to this program, launcher stripped.

    :returns: The argv the entry point will see.
    """
    tokens = committed_command()
    for i, token in enumerate(tokens):
        if token.endswith("train_hyper_connections.py"):
            return tokens[i + 1 :]
    raise AssertionError(f"the committed command runs no known entry point: {tokens}")


def test_the_committed_command_carries_no_word_the_parser_does_not_know():
    """
    ``main`` reads argv with ``parse_known_args`` and hands the leftovers to the config layer
    as dotted overrides. So a flag this parser has never heard of is not an error: it is read
    as an override, fails to resolve against any config field, and takes the run down inside a
    container -- or worse, resolves against something and quietly trains a different model.
    A typo in run.yaml has to be caught here, on a laptop, because nothing downstream will.
    """
    _, leftovers = entry.build_parser().parse_known_args(committed_argv())
    assert leftovers == []


def test_the_committed_microbatch_divides_the_rank_batch_exactly():
    """
    Gradient accumulation splits each rank's share of the global batch into whole microbatches,
    so a rank batch that is not a multiple of the microbatch is either a refusal at startup or
    a silently different batch size. Sequence length has to divide it for the same reason.
    """
    tokens = committed_command()
    ranks = [t for t in tokens if t.startswith("--nproc-per-node")]
    assert len(ranks) == 1, tokens
    nproc = int(ranks[0].split("=")[1])

    opts, _ = entry.build_parser().parse_known_args(committed_argv())
    rank_batch, remainder = divmod(opts.global_batch_size, nproc)
    assert remainder == 0
    assert rank_batch % opts.rank_microbatch_size == 0
    assert opts.rank_microbatch_size % opts.sequence_length == 0


def test_the_default_save_interval_is_priced_in_wall_clock():
    """
    Pins the checkpoint count an arm writes, which is stated nowhere else. It is the number
    the interval was chosen for, and the thing that changes if somebody moves the interval or
    the step count without pricing the loss a lost host would take.
    """
    _, opts = build(BASE_ARGV + ["--arm", "baseline"])
    assert opts.save_interval == arms.TRANCHE_SAVE_INTERVAL == 500
    assert opts.steps // opts.save_interval + 1 == 13
    exposure_hours = opts.save_interval * arms.MEASURED_SECONDS_PER_STEP / 3600
    assert exposure_hours == pytest.approx(1.43, abs=0.05)


def test_the_committed_command_carries_no_seed():
    """
    THE ONE WORD THAT MUST NOT BE IN run.yaml. This tranche submits three seeds as one
    three-cell fan-out, and every cell is handed this exact command. A --seed here would run
    one replicate three times and report a noise floor of zero. `resolve_seed` refuses that
    combination at run time; this catches it on a laptop, before a queue wait.
    """
    assert "--seed" not in committed_argv()

    opts, _ = entry.build_parser().parse_known_args(committed_argv())
    assert opts.seed is None


def test_the_committed_command_is_the_tranche_the_arm_table_priced():
    """
    The arm table's cost model and the submitted command are two statements of the same run,
    and nothing at submission compares them: `edullm check` prices a ceiling out of the
    workload profile and never reads either. So they are compared here.
    """
    opts, _ = entry.build_parser().parse_known_args(committed_argv())

    assert opts.arm in arms.FUNDED, "run.yaml runs an arm the tranche does not fund"
    assert opts.steps == arms.TRANCHE_STEPS
    assert opts.save_interval == arms.TRANCHE_SAVE_INTERVAL
    assert opts.eval_interval == arms.TRANCHE_EVAL_INTERVAL
    assert opts.monitor_interval == arms.TRANCHE_MONITOR_INTERVAL
    assert opts.global_batch_size == arms.TRANCHE_TOKENS_PER_STEP
    assert opts.model_factory == "hc_370M"
    # The platform reads command words to decide what a shape can do and cannot see a dtype
    # set in code, so the dtype has to be in the text or a T4 would accept this.
    assert "--param-dtype" in committed_argv()


def test_the_committed_run_fits_one_attempt_of_the_bound_it_will_be_submitted_under():
    """
    A cell killed by the attempt timeout is not reliably retried -- the platform's retry table
    grants a second attempt for a lost host and reaches a timeout only through a
    no-exit-code fall-through that torchrun's non-zero exit on SIGTERM races. So the committed
    command has to finish inside one attempt, with room for eighteen hours of drift.
    """
    opts, _ = entry.build_parser().parse_known_args(committed_argv())
    hours = arms.arm_seconds(arms.ARMS[opts.arm], opts.steps) / 3600
    assert hours < 0.9 * 21.0, f"{opts.arm} needs {hours:.1f}h against a 21h attempt"
