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


def spec_of(name: str = "run.yaml") -> dict:
    """
    One submission spec, parsed.

    :param name: The file, relative to ``.edullm/``.
    """
    return yaml.safe_load((pathlib.Path(_HERE) / name).read_text())


def committed_command(name: str = "run.yaml") -> list:
    """
    The words the platform will exec, from the spec it reads them out of.

    :param name: The spec file, relative to ``.edullm/``.

    :returns: The tokens inside the ``bash -lc`` wrapper.
    """
    wrapper = shlex.split(spec_of(name)["command"])
    assert wrapper[:2] == ["bash", "-lc"], wrapper[:2]
    return shlex.split(wrapper[2])


def committed_argv(name: str = "run.yaml") -> list:
    """
    Everything the committed command passes to this program, launcher stripped.

    :param name: The spec file, relative to ``.edullm/``.

    :returns: The argv the entry point will see.
    """
    tokens = committed_command(name)
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


def test_the_committed_command_is_a_shape_this_table_prices():
    """
    The arm table's cost model and the submitted command are two statements of the same run,
    and nothing at submission compares them: `edullm check` prices a ceiling out of the
    workload profile and never reads either. So they are compared here.

    TWO SHAPES ARE ALLOWED HERE AND NOT ONE, because run.yaml legitimately holds either the
    tranche or a throughput probe -- see ``PROBE_STEPS`` in hyper_connection_arms.py for why
    the probe exists. Which one it is, is read off the step count and then every other
    interval is pinned against that choice, so a file half-edited from one into the other is
    a failure here rather than a run that saves on the wrong interval. What both share is
    checked once, below the branch.
    """
    opts, _ = entry.build_parser().parse_known_args(committed_argv())

    if opts.steps == arms.TRANCHE_STEPS:
        assert opts.save_interval == arms.TRANCHE_SAVE_INTERVAL
        assert opts.eval_interval == arms.TRANCHE_EVAL_INTERVAL
    elif opts.steps == arms.PROBE_STEPS:
        assert opts.save_interval == arms.PROBE_SAVE_INTERVAL
        assert opts.eval_interval == arms.PROBE_EVAL_INTERVAL
        assert opts.warmup_steps == arms.PROBE_WARMUP_STEPS
    else:
        raise AssertionError(
            f"run.yaml runs {opts.steps} steps, which is neither the tranche's "
            f"{arms.TRANCHE_STEPS} nor the probe's {arms.PROBE_STEPS}, so nothing has priced it"
        )

    assert opts.arm in arms.FUNDED, "run.yaml runs an arm the tranche does not fund"
    assert opts.monitor_interval == arms.TRANCHE_MONITOR_INTERVAL
    assert opts.global_batch_size == arms.TRANCHE_TOKENS_PER_STEP
    assert opts.model_factory == "hc_370M"
    # The platform reads command words to decide what a shape can do and cannot see a dtype
    # set in code, so the dtype has to be in the text or a T4 would accept this.
    assert "--param-dtype" in committed_argv()


def test_the_committed_command_starts_one_process_per_device():
    """
    THE REFUSAL THIS CATCHES HAS ALREADY COST A SUBMISSION. `edullm check` refused the 4-rank
    command against gpu-8xa100 with ``process_per_device``.

    The platform's ``launchers.require_a_process_for_every_device`` reads
    ``CONTAINER_SHAPES[profile].gpus`` and compares it against the count it parses out of the
    command -- and it parses it the way a shell would, opening the ``bash -lc`` wrapper,
    finding ``python`` in command position, recognising ``-m torch.distributed.run``, then
    scanning for ``--nproc-per-node`` or ``--nproc_per_node`` in either spelling. Fewer
    processes than devices bills cards that idle; more puts two ranks on one card.

    So this reads the same two facts out of the same file. ``suggested_compute`` is what the
    submission defaults to, and a command whose rank count was left behind by a change of
    shape is exactly the file this catches.
    """
    spec = yaml.safe_load((pathlib.Path(_HERE) / "run.yaml").read_text())
    profile = spec["suggested_compute"]
    assert profile in arms.GPUS_PER_COMPUTE_PROFILE, (
        f"run.yaml suggests {profile!r}, whose device count is not recorded in "
        "GPUS_PER_COMPUTE_PROFILE, so nothing here can check the rank count against it"
    )

    tokens = committed_command()
    assert "--standalone" in tokens, "a single-host launcher needs no rendezvous endpoint"
    ranks = [t for t in tokens if t.startswith("--nproc-per-node")]
    assert len(ranks) == 1, tokens
    assert int(ranks[0].split("=")[1]) == arms.GPUS_PER_COMPUTE_PROFILE[profile]


def test_the_committed_run_fits_one_attempt_of_the_bound_it_will_be_submitted_under():
    """
    A cell killed by the attempt timeout is not reliably retried -- the platform's retry table
    grants a second attempt for a lost host and reaches a timeout only through a
    no-exit-code fall-through that torchrun's non-zero exit on SIGTERM races. So the committed
    command has to finish inside one attempt, with room for eighteen hours of drift.

    Priced at the L40S step time whatever shape the command names, which is conservative in
    the direction that matters: the probe exists precisely because nobody knows the A100
    number yet, and assuming a faster card before measuring one is the assumption this whole
    branch is trying to avoid making.
    """
    opts, _ = entry.build_parser().parse_known_args(committed_argv())
    hours = arms.arm_seconds(arms.ARMS[opts.arm], opts.steps) / 3600
    assert hours < 0.9 * 21.0, f"{opts.arm} needs {hours:.1f}h against a 21h attempt"


STAGED = sorted(arms.STAGED_TRANCHES)


@pytest.mark.parametrize("index", range(len(arms.TRANCHE_CELLS)))
def test_every_cell_of_the_tranche_resolves_to_its_own_arm_and_seed(index: int):
    """
    THE WHOLE POINT OF THE arm-and-seed FAN-OUT, AND THE FAILURE IT RULES OUT IS SILENT.
    One submission, one command, one cell per run. Batch is the only thing that knows which
    cell a process is and it says so in AWS_BATCH_JOB_ARRAY_INDEX; everything else about the
    cell is derived from the arm table here.

    THE COUNT COMES FROM THE ARM TABLE AND IS NOT WRITTEN DOWN. It was nine when this was
    written and is fifteen now, and nothing in this file was edited to follow it -- which is
    the property the derivation exists for.
    """
    name, seed = arms.cell(index)
    assert (name, seed) == arms.TRANCHE_CELLS[index]
    assert name in arms.FUNDED
    assert 0 <= seed < arms.ARMS[name].seeds

    resolved_arm, resolved_seed, provenance = entry.resolve_cell(
        None, None, fanout_env(index, parameter=entry.FANOUT_INDEX_PARAMETER_CELL)
    )
    assert (resolved_arm, resolved_seed) == (name, seed)
    assert entry.FANOUT_INDEX_VARIABLE in provenance


def test_the_cell_table_is_the_tranche_and_covers_each_arm_at_each_seed():
    assert len(arms.TRANCHE_CELLS) == arms.total_runs() == 20
    assert arms.TRANCHE_CELLS[0] == ("baseline", 0)
    # The last cell follows the arm table's order, which is the pre-registration's numbering, so
    # funding arm 9 moved the tail from arm 3 to arm 9 without the list being edited.
    assert arms.TRANCHE_CELLS[-1] == ("mhc", arms.ARMS["mhc"].seeds - 1)
    assert len(set(arms.TRANCHE_CELLS)) == len(arms.TRANCHE_CELLS)
    for name in arms.FUNDED:
        seeds = sorted(s for a, s in arms.TRANCHE_CELLS if a == name)
        assert seeds == list(range(arms.ARMS[name].seeds))
        # Contiguous from zero, because Batch requires that of an array index and the table is
        # what the index is read against.
        assert seeds == list(range(len(seeds)))


def test_the_cells_are_actually_that_many_different_models(monkeypatch):
    """
    Reading the pair back is not enough: the failure this guards is cells that train the same
    thing, so what has to differ is the arm together with the three numbers the model, the
    global RNG and the shuffle are drawn from. Two cells landing on one seed report a noise
    floor smaller than the one there is, and every contrast in the analysis plan is divided
    by it.
    """
    monkeypatch.setenv(entry.FANOUT_PARAMETER_VARIABLE, entry.FANOUT_INDEX_PARAMETER_CELL)
    seen = set()
    for index in range(len(arms.TRANCHE_CELLS)):
        monkeypatch.setenv(entry.FANOUT_INDEX_VARIABLE, str(index))
        config, opts = build(BASE_ARGV)
        seen.add((opts.arm, config.init_seed, config.model.init_seed, opts.data_seed))

    assert len(seen) == len(arms.TRANCHE_CELLS), f"cells collided: {sorted(seen)}"
    assert {arm for arm, *_ in seen} == set(arms.FUNDED)


def test_a_cell_outside_the_tranche_is_refused_rather_than_wrapped():
    """
    An index past the end means --fanout-size and the arm table disagree about how many runs
    there are, which is a submission that runs the wrong experiment rather than a crash.
    """
    beyond = len(arms.TRANCHE_CELLS)
    with pytest.raises(train_on_corpus.Refusal, match=f"--fanout-size {beyond}"):
        entry.resolve_cell(
            None, None, fanout_env(beyond, parameter=entry.FANOUT_INDEX_PARAMETER_CELL)
        )


@pytest.mark.parametrize("flag,value", [("arm", "faithful"), ("seed", 1)])
def test_a_flag_the_cell_index_owns_is_refused_inside_the_arm_and_seed_fanout(flag, value):
    """
    Every cell of a fan-out is handed the same command, so an --arm written into the spec runs
    one arm in every cell and a --seed runs one replicate in every cell. Neither raises,
    neither bends a curve, and the tranche reports a noise floor it did not measure.
    """
    kwargs = {"explicit_arm": None, "explicit_seed": None, f"explicit_{flag}": value}
    with pytest.raises(train_on_corpus.Refusal, match="noise floor"):
        entry.resolve_cell(
            environ=fanout_env(0, parameter=entry.FANOUT_INDEX_PARAMETER_CELL), **kwargs
        )


def test_no_arm_and_no_cell_index_is_refused_rather_than_defaulted():
    """
    There is no arm this experiment can silently mean, and the parser can no longer require
    one because the arm-and-seed command deliberately carries none.
    """
    with pytest.raises(train_on_corpus.Refusal, match="nothing says which arm"):
        entry.resolve_cell(None, None, {})


def test_the_seed_fanout_still_works_beside_the_arm_and_seed_one():
    """
    THIS IS THE PATH ALL FIFTEEN CELLS ACTUALLY TOOK, so it has to keep working rather than
    merely keep parsing. The tranche was split into three per-arm submissions for the
    pre-registration's reason, stage 1 has already run through `resolve_seed`, and both stage-2
    specs use the same call so that one mechanism assigned every replicate in the module.
    """
    name, seed, provenance = entry.resolve_cell("faithful", None, fanout_env(2))
    assert (name, seed) == ("faithful", 2)
    assert entry.FANOUT_INDEX_VARIABLE in provenance


@pytest.mark.parametrize("shape", STAGED)
def test_both_staged_tranches_are_the_shape_this_table_prices(shape: str):
    """
    THE FILE THIS CATCHES IS A HALF-EDITED ONE, and it catches it on a laptop rather than at
    admission or eighteen hours into nine machines. Two variants are staged because which one
    runs is decided by a probe that is still going, and the one that loses is edited by nobody
    and read by nobody until the day it is needed -- which is exactly the file that drifts.

    Everything here is read out of ``STAGED_TRANCHES``, so the cost model, the submitted
    command and this test are one statement rather than three. Nothing at submission compares
    them: ``edullm check`` prices a ceiling out of the workload profile and never reads either.
    """
    staged = arms.STAGED_TRANCHES[shape]
    spec = spec_of(staged.spec)
    opts, leftovers = entry.build_parser().parse_known_args(committed_argv(staged.spec))

    assert spec["schema_version"] == 1
    assert spec["workload_profile"] == "olmo-core-train"
    assert spec["suggested_compute"] == staged.compute_profile == shape

    # A flag the parser has never heard of is not an error: parse_known_args hands it on as a
    # dotted override, where it either fails inside a container or resolves against something
    # and quietly trains a different model.
    assert leftovers == []

    assert opts.steps == staged.steps
    assert opts.warmup_steps == staged.warmup_steps
    assert opts.rank_microbatch_size == staged.rank_microbatch_size
    assert opts.save_interval == arms.TRANCHE_SAVE_INTERVAL
    assert opts.eval_interval == arms.TRANCHE_EVAL_INTERVAL
    assert opts.monitor_interval == arms.TRANCHE_MONITOR_INTERVAL
    assert opts.fail_closed_by_step == arms.TRANCHE_FAIL_CLOSED_BY_STEP
    assert opts.global_batch_size == arms.TRANCHE_TOKENS_PER_STEP
    assert opts.sequence_length == 4096
    assert opts.model_factory == "hc_370M"

    # Defaults, written into the command anyway: a default is a number a later commit may
    # move, and nine runs compared against each other have to have been compared at the same
    # settings. What is in the command text is what the lineage record seals.
    argv = committed_argv(staged.spec)
    for flag in (
        "--weight-decay",
        "--z-loss-multiplier",
        "--held-out-shards",
        "--bytes-per-token",
        "--eval-interval",
        "--monitor-interval",
        # The platform reads command words to decide what a shape can do and cannot see a
        # dtype set in code, so a command that does not name it is accepted onto a T4.
        "--param-dtype",
    ):
        assert flag in argv, flag
    assert opts.weight_decay == entry.DEFAULT_WEIGHT_DECAY
    assert opts.z_loss_multiplier == 1e-5
    assert opts.held_out_shards == arms.HELD_OUT_SHARDS
    assert opts.bytes_per_token == arms.DOLMA2_BYTES_PER_TOKEN

    # Unset means ordinary RoPE. Setting it would put an unmeasured second change inside the
    # contrast H1 and H2a are about.
    assert opts.partial_rotary_factor is None

    # $EDULLM_CHECKPOINT_DIR has to be expanded by a shell for the platform's
    # require_a_save_folder_a_retry_can_find to see it, and a retry needs the same folder.
    assert "$EDULLM_CHECKPOINT_DIR" in spec["command"]


@pytest.mark.parametrize("shape", STAGED)
def test_neither_staged_tranche_carries_the_flags_the_cell_index_owns(shape: str):
    """
    THE TWO WORDS THAT MUST NOT BE IN EITHER FILE. Nine cells are handed one command, so an
    --arm here runs one arm nine times and a --seed runs one replicate nine times: no error,
    no visibly wrong curve, nine lines lying on top of each other that read as a very clean
    experiment. ``resolve_cell`` refuses both at run time; this catches them before a queue
    wait.
    """
    argv = committed_argv(arms.STAGED_TRANCHES[shape].spec)
    assert "--arm" not in argv
    assert "--seed" not in argv

    opts, _ = entry.build_parser().parse_known_args(argv)
    assert opts.arm is None
    assert opts.seed is None


@pytest.mark.parametrize("shape", STAGED)
def test_both_staged_tranches_start_one_process_per_device(shape: str):
    """
    The refusal this catches has already cost a submission: ``edullm check`` refused a 4-rank
    command against gpu-8xa100 with ``process_per_device``. The platform's
    ``require_a_process_for_every_device`` reads the device count out of its own
    CONTAINER_SHAPES and compares it against the count it parses out of the command, so a rank
    count left behind by a change of shape is exactly the file this catches -- and the two
    staged variants differ in precisely that.
    """
    staged = arms.STAGED_TRANCHES[shape]
    tokens = committed_command(staged.spec)
    assert "--standalone" in tokens, "a single-host launcher needs no rendezvous endpoint"

    ranks = [t for t in tokens if t.startswith("--nproc-per-node")]
    assert len(ranks) == 1, tokens
    nproc = int(ranks[0].split("=")[1])
    assert nproc == arms.GPUS_PER_COMPUTE_PROFILE[staged.compute_profile]

    # Gradient accumulation splits each rank's share into whole microbatches, so a rank batch
    # that is not a multiple of the microbatch is a refusal at startup or a silently different
    # batch size.
    opts, _ = entry.build_parser().parse_known_args(committed_argv(staged.spec))
    rank_batch, remainder = divmod(opts.global_batch_size, nproc)
    assert remainder == 0
    assert rank_batch % opts.rank_microbatch_size == 0
    assert opts.rank_microbatch_size % opts.sequence_length == 0


@pytest.mark.parametrize("shape", STAGED)
def test_each_staged_tranche_fits_one_attempt_of_the_bound_it_is_submitted_under(shape: str):
    """
    A cell killed by the attempt timeout is not reliably retried: the platform's retry table
    grants a second attempt for a lost host and reaches a timeout only through a
    no-exit-code fall-through that torchrun's non-zero exit on SIGTERM races. So each variant
    has to finish inside one attempt of its own ``--hours``, at the step time it is priced
    against -- the measurement for the L40S, and for the A100 the threshold the probe has to
    clear, which is the slowest step that would send the tranche there at all.
    """
    staged = arms.STAGED_TRANCHES[shape]
    assert (
        staged.hours_per_run < staged.hours
    ), f"{shape} needs {staged.hours_per_run:.1f}h against a {staged.hours:.0f}h attempt"
    # And the bound is one the platform will accept: --hours may only lower the workload's own.
    assert staged.hours <= 24.0


@pytest.mark.parametrize("shape", STAGED)
def test_a_retry_of_either_variant_loses_a_bounded_and_stated_amount_of_work(shape: str):
    """
    The resume is sound -- ``Trainer.fit`` loads from the save folder with load_trainer_state
    and load_optim_state hard-coded true, and each cell's prefix is re-derived identically by
    a retry of the same child -- so what a lost host costs is one save interval and no more.

    What the resume does NOT restore is ``max_steps``: ``Trainer.state_dict`` writes it and
    ``load_state_dict`` never reads it back, and ``Scheduler.set_lr`` reads it live on every
    step. So every attempt has to be launched with the same explicit ``--steps``, which is
    what pins it into the command text here rather than leaving it to the parser default.
    """
    staged = arms.STAGED_TRANCHES[shape]
    assert staged.checkpoint_exposure_hours < 1.5
    assert "--steps" in committed_argv(staged.spec)

    checkpoints = staged.steps // arms.TRANCHE_SAVE_INTERVAL + 1
    assert checkpoints == (13 if staged.steps == arms.TRANCHE_STEPS else 26)


def test_the_step_time_that_would_buy_the_full_horizon_is_written_down_in_advance():
    """
    A threshold chosen after the measurement is a threshold chosen to agree with it.

    This pins the number the probe will be read against: the slowest A100 step that still
    lands a 12,715-step arm inside 24 hours with a tenth of the bound unspent. It is not
    ``24 * 3600 * 0.9 / 12715``, which would be 6.11 s and wrong by the 1.24 hours the 27
    evaluations, 26 checkpoints and 254 monitor firings cost whatever card they run on.
    """
    assert arms.seconds_per_step_to_fit(24.0) == pytest.approx(
        arms.A100_STEP_SECONDS_FOR_FULL_HORIZON, abs=0.01
    )
    # The measured L40S step is the reason this branch exists at all: it does not clear the
    # threshold, so the tranche cannot reach the full horizon on that shape.
    assert arms.MEASURED_SECONDS_PER_STEP > arms.A100_STEP_SECONDS_FOR_FULL_HORIZON


# ---------------------------------------------------------------------------------------
# THE THREE STAGE SPECS, AND THE ONE TEST THAT MAKES A SPLIT TRANCHE AS SAFE AS ONE
# ---------------------------------------------------------------------------------------
#
# The tranche went out as three five-cell submissions rather than one fifteen-cell one,
# because the pre-registration forbids submitting a treatment arm before the noise floor has
# numbers in it. That split gives up a guarantee the single submission had for free: in one
# submission every cell is the same command at the same commit, so there is nothing for a
# setting to drift between. Three submissions at three commits is three chances for one, and
# the symptom of one is a plausible loss curve.
#
# So the guarantee is rebuilt here. These tests are the only thing in the pipeline that
# compares the arms' commands to each other: `edullm check` prices a ceiling out of the
# workload profile and never reads a hyperparameter, and the analysis reads what the runs
# report rather than what they were asked for.

STAGES = list(arms.STAGE_SPECS)


def resolved(name: str) -> dict:
    """
    Every option one stage's committed command resolves to, through the real parser.

    Resolved rather than textual, because the failure is a *value* that differs and the two
    ways a value gets set -- written in the command, or left to a default -- are exactly what
    the stages disagree about. Comparing the command text would report stage 1 and stage 2 as
    different on nine flags that resolve to the same nine numbers, which is a test nobody
    would keep.

    :param name: The spec file, relative to ``.edullm/``.
    """
    opts, leftovers = entry.build_parser().parse_known_args(committed_argv(name))
    assert leftovers == [], f"{name} carries words the parser does not know: {leftovers}"
    return vars(opts)


def test_the_stage_specs_differ_in_the_arm_and_in_nothing_else():
    """
    THE TEST THE SPLIT TRANCHE EXISTS ON, AND THE ONE FAILURE IT CATCHES IS INVISIBLE
    EVERYWHERE ELSE.

    Twenty runs in four submissions are only an experiment if all twenty were trained at
    the same settings. A hand-edit to one arm's command -- a microbatch nudged to fit a
    memory scare, a horizon shortened to save an hour, a learning rate copied from another
    file -- makes the arms incomparable and produces loss curves that look precisely like
    loss curves. Nothing downstream disagrees: the platform prices a ceiling from the
    workload profile and never reads these words, and the analysis reads the runs' own
    reports rather than the commands that made them.

    THIS TEST WAS CALLED ``test_the_three_stage_specs_differ_in_the_arm_and_in_nothing_else``
    and the headers of run.faithful-stage.yaml and run.output-only-stage.yaml still cite it
    under that name. Those two files are the text two admitted submissions were built from and
    are not edited after the fact; the count in the old name is what went stale when ``mhc``
    was funded, and this line is where a reader who greps the old name lands.

    THE COMPARISON IS OVER THE WHOLE RESOLVED OPTION SET AND NOT OVER A LIST OF THINGS TO
    CHECK, which is the difference between a test that holds and a test that held. A checked
    list silently permits every option not on it, and the option that breaks this experiment
    is the one a later commit adds -- which is on nobody's list by construction. The
    allowlist is inverted instead: ``STAGE_CONTRAST_EXEMPT`` names what may differ and
    carries the reason it may, so a new option is caught by default and a new exemption is a
    reviewable line in the arm table.
    """
    options = {name: resolved(arms.STAGE_SPECS[name].spec) for name in STAGES}

    keys = {frozenset(o) for o in options.values()}
    assert len(keys) == 1, "the specs parse to different option sets"

    differ = {key for key in next(iter(keys)) if len({repr(o[key]) for o in options.values()}) > 1}
    assert differ == set(arms.STAGE_CONTRAST_EXEMPT), (
        "the stages differ in "
        + ", ".join(sorted(differ))
        + " and the only differences the arm table permits are "
        + ", ".join(sorted(arms.STAGE_CONTRAST_EXEMPT))
        + ". Every option outside that set has to be identical across all four, or the "
        "twenty runs were not trained at the same settings and the contrast is not one."
    )

    # And the difference in the one option the experiment is made of is the four arms
    # themselves, rather than two specs that happen to name the same one.
    assert {o["arm"] for o in options.values()} == set(arms.FUNDED)
    for name, opts in options.items():
        assert opts["arm"] == name == arms.STAGE_SPECS[name].arm


@pytest.mark.parametrize("name", STAGES)
def test_every_stage_is_pinned_to_what_the_running_baseline_resolved_to(name: str):
    """
    The diff above catches a stage that disagrees with the other two TODAY. This catches the
    other half: a default that has moved since stage 1 ran.

    Stage 1 is `run_019fe279-4ef0` and its five cells were built from commit 38b665919. They
    trained at whatever the parser resolved to THEN, and no later commit can change that --
    but a later commit can change what stage 2 resolves to, and re-parsing stage 1's command
    on a laptop today reports the new default rather than the one it ran under. So the values
    those cells actually used are frozen in ``STAGE_PINNED`` as literals, and all three stages
    are held against them.

    A failure here does not mean "fix this number". It means stage 1 and stage 2 were trained
    at different settings, and the only repair is five more baseline cells.
    """
    options = resolved(arms.STAGE_SPECS[name].spec)
    for option, value in arms.STAGE_PINNED.items():
        assert options[option] == value, (
            f"{name} resolves {option} to {options[option]!r}, and the five baseline cells "
            f"of run_019fe279-4ef0 trained at {value!r}"
        )


def test_the_pinned_table_still_agrees_with_the_design_constants_it_froze():
    """
    ``STAGE_PINNED`` is literals on purpose -- a reference to ``DEFAULT_WEIGHT_DECAY`` would
    move with the default it exists to catch -- and the cost of literals is that they can
    fall out of step with the design silently. This is the cross-check, and it is the one
    place the two are compared.

    It fails if somebody changes the tranche's horizon, its intervals or its corpus constants
    while stage 2 is still unsubmitted, which is a real thing to want and is not free: it
    makes the running baseline incomparable to whatever stage 2 becomes.
    """
    assert arms.STAGE_PINNED["steps"] == arms.TRANCHE_STEPS
    assert arms.STAGE_PINNED["save_interval"] == arms.TRANCHE_SAVE_INTERVAL
    assert arms.STAGE_PINNED["eval_interval"] == arms.TRANCHE_EVAL_INTERVAL
    assert arms.STAGE_PINNED["monitor_interval"] == arms.TRANCHE_MONITOR_INTERVAL
    assert arms.STAGE_PINNED["global_batch_size"] == arms.TRANCHE_TOKENS_PER_STEP
    assert arms.STAGE_PINNED["warmup_steps"] == round(
        arms.TRANCHE_STEPS * arms.TRANCHE_WARMUP_FRACTION
    )
    assert arms.STAGE_PINNED["held_out_shards"] == arms.HELD_OUT_SHARDS
    assert arms.STAGE_PINNED["bytes_per_token"] == arms.DOLMA2_BYTES_PER_TOKEN
    assert arms.STAGE_PINNED["weight_decay"] == entry.DEFAULT_WEIGHT_DECAY
    assert arms.STAGE_PINNED["learning_rate"] == entry.DEFAULT_LEARNING_RATE

    # Nothing that the fan-out index owns, and nothing the platform supplies, is in here: a
    # pinned seed would be five cells of one replicate and a pinned checkpoint directory would
    # be five cells writing over each other.
    assert not {"arm", "seed", "data_seed", "save_folder", "run_name"} & set(arms.STAGE_PINNED)


def test_the_two_stages_that_omit_the_lane_guard_omit_it_for_the_reasons_claimed(monkeypatch):
    """
    ``fail_closed_by_step`` is exempt from the diff, and this is the proof rather than the
    claim. TWO OF THE FOUR STAGES OMIT IT AND THE TWO REASONS ARE NOT THE SAME ONE, which is
    exactly why an exemption granted in prose needs a test underneath it.

    On BASELINE the option is UNREACHABLE: ``train`` attaches
    ``HyperConnectionMonitorCallback`` where ``arm.hyper_connections is not None`` and the
    baseline's is ``None``, so stage 1's omitting it and stage 2's setting it to 400 describe
    the same run. If a later commit attaches that monitor unconditionally -- for lane norms on
    an ordinary residual stream, say, which is a reasonable thing to want -- the exemption stops
    being sound and this fails.

    On MHC the option is REACHABLE AND MISCALIBRATED, which is the opposite situation and the
    more dangerous one: setting it would abort all five cells at step 400, because the Sinkhorn
    projection compresses the lane dispersion the guard reads.
    ``test_the_mhc_arm_is_the_only_stage_the_lane_gate_would_refuse`` in
    ``test_hyper_connection_arms.py`` measures that; what this asserts is that the arm which
    would be refused is the arm whose command leaves the flag out.
    """
    seen: set = set()

    def fake_train(config, opts=None):
        seen.clear()
        seen.update(config.trainer.callbacks)

    monkeypatch.setattr(entry, "_train", fake_train)

    for name in STAGES:
        stage = arms.STAGE_SPECS[name]
        argv = committed_argv(stage.spec) + BASE_ARGV[1:]
        opts, overrides = entry.build_parser().parse_known_args(argv)
        entry.train(entry.build_config(opts, overrides), opts)

        reachable = "hyper_connections" in seen
        if name == "baseline":
            assert not reachable, "the baseline now runs the monitor, so the exemption is unsound"
            assert opts.fail_closed_by_step is None
        elif name == "mhc":
            # Reachable, and left unset anyway. The monitor is still attached and still records
            # the radius H5 is about; it is the abort threshold that is absent.
            assert reachable, "mhc without the monitor would leave H5 without its instrument"
            assert opts.fail_closed_by_step is None, (
                "run.mhc-stage.yaml now sets --fail-closed-by-step, which would abort all five "
                "cells at step 400; see MHC_LANE_DISPERSION_AT_GATE"
            )
        else:
            assert reachable, name
            assert opts.fail_closed_by_step == arms.TRANCHE_FAIL_CLOSED_BY_STEP == 400


@pytest.mark.parametrize("name", STAGES)
def test_every_stage_carries_its_arm_and_not_the_replicate_the_index_owns(name: str):
    """
    THE ASYMMETRY IS LOAD-BEARING IN BOTH DIRECTIONS. Under
    ``--fanout-index-parameter seed`` all five cells are handed one command, so an explicit
    --seed runs one replicate five times: five identical curves, a measured noise floor of
    zero, and every later contrast significant against it. And nothing but the command
    supplies the arm on this path, so --arm has to be there or the run is refused.
    """
    argv = committed_argv(arms.STAGE_SPECS[name].spec)
    assert "--seed" not in argv
    assert "--arm" in argv

    options = resolved(arms.STAGE_SPECS[name].spec)
    assert options["seed"] is None
    assert options["arm"] == arms.STAGE_SPECS[name].arm

    # Which is the combination resolve_seed resolves rather than refuses, at every cell.
    for index in range(arms.STAGE_SPECS[name].cells):
        seed, provenance = entry.resolve_seed(None, fanout_env(index))
        assert seed == index
        assert entry.FANOUT_INDEX_VARIABLE in provenance


@pytest.mark.parametrize("name", STAGES)
def test_every_stage_starts_one_process_per_device(name: str):
    """
    The refusal this catches has already cost a submission: ``edullm check`` refused a 4-rank
    command against gpu-8xa100 with ``process_per_device``. Fewer processes than devices bills
    cards that idle, and more puts two ranks on one card.
    """
    spec = spec_of(arms.STAGE_SPECS[name].spec)
    assert spec["schema_version"] == 1
    assert spec["workload_profile"] == "olmo-core-train"

    profile = spec["suggested_compute"]
    assert profile in arms.GPUS_PER_COMPUTE_PROFILE, profile

    tokens = committed_command(arms.STAGE_SPECS[name].spec)
    assert "--standalone" in tokens, "a single-host launcher needs no rendezvous endpoint"
    ranks = [t for t in tokens if t.startswith("--nproc-per-node")]
    assert len(ranks) == 1, tokens
    nproc = int(ranks[0].split("=")[1])
    assert nproc == arms.GPUS_PER_COMPUTE_PROFILE[profile]

    # Gradient accumulation splits each rank's share into whole microbatches.
    options = resolved(arms.STAGE_SPECS[name].spec)
    rank_batch, remainder = divmod(options["global_batch_size"], nproc)
    assert remainder == 0
    assert rank_batch % options["rank_microbatch_size"] == 0

    # A retry has to find the same folder, and only a shell can expand this.
    assert "$EDULLM_CHECKPOINT_DIR" in spec["command"]


@pytest.mark.parametrize("name", STAGES)
def test_every_stage_fits_one_attempt_of_the_bound_it_is_submitted_under(name: str):
    """
    A cell killed by the attempt timeout is not reliably retried: the platform grants a second
    attempt for a lost host and reaches a timeout only through a no-exit-code fall-through
    that torchrun's non-zero exit on SIGTERM races. So a cell has to finish inside one
    attempt of ``--hours``, with room for eighteen hours of step-time drift.

    ``--steps`` is in the command text rather than left to the default for the reason a retry
    exposes: ``Trainer.state_dict`` writes ``max_steps`` and ``load_state_dict`` never reads
    it back, while ``Scheduler.set_lr`` reads it live on every step, so a second attempt
    launched under a different horizon resumes with a different schedule.
    """
    stage = arms.STAGE_SPECS[name]
    spare = 1.0 - stage.hours_per_cell / arms.STAGE_HOURS
    assert spare > 0.05, (
        f"{name} needs {stage.hours_per_cell:.1f}h against a {arms.STAGE_HOURS:.0f}h attempt, "
        f"which leaves {spare:.1%} of the bound for step-time drift"
    )
    # And 19 rather than the profile's 24, which is not caution being traded for money for its
    # own sake: --hours is also the hours factor of the approved ceiling, where it is
    # multiplied by two attempts and by five cells, so each hour of headroom costs ten
    # cell-hours of ceiling per stage. 6% of eighteen hours is about an hour of drift, and the
    # step time it is drifting from is measured rather than assumed.
    assert spare < 0.15, "more headroom than the drift needs, bought with ceiling"
    assert arms.STAGE_HOURS <= 24.0, "--hours may only lower the workload's own bound"
    assert "--steps" in committed_argv(stage.spec)

    # What a lost host throws away, which is one save interval and no more.
    exposure = arms.TRANCHE_SAVE_INTERVAL * arms.MEASURED_SECONDS_PER_STEP / 3600.0
    assert exposure < 1.5


def test_the_stages_are_the_whole_tranche_and_no_arm_is_left_out():
    """
    Twenty cells in four submissions have to be the same twenty the arm table prices, or the
    budget is for one design and the runs are another. The count is not written down here: it
    is the arm table's, so a seed added to an arm and not to a stage fails.
    """
    assert sorted(arms.STAGE_SPECS) == sorted(arms.FUNDED)
    assert sum(stage.cells for stage in arms.STAGE_SPECS.values()) == arms.total_runs() == 20
    assert {stage.cells for stage in arms.STAGE_SPECS.values()} == {arms.STAGE_CELLS}

    # Balanced, because an unbalanced contrast carries SE = sigma*sqrt(1/n_a + 1/n_b) and pays
    # for the smaller arm twice. mhc joining at five seeds rather than at three is what keeps
    # H5 as powerful as H1 and H2a.
    assert len({arms.ARMS[name].seeds for name in arms.FUNDED}) == 1

    # Stage 1 is the noise floor and it is the one that has run. A treatment arm submitted
    # first would have nothing to be compared against that was not estimated from itself. The
    # later stage numbers are not an ordering constraint -- 3 records that mhc was funded by a
    # grant after stage 2 was written, and either may go out first.
    assert arms.STAGE_SPECS["baseline"].stage == 1
    assert arms.STAGE_SPECS["baseline"].run_id
    assert all(arms.STAGE_SPECS[name].stage == 2 for name in ("faithful", "output-only"))
    assert arms.STAGE_SPECS["mhc"].stage == 3
    assert all(stage.stage > 1 for name, stage in arms.STAGE_SPECS.items() if name != "baseline")


@pytest.mark.parametrize("shape", STAGED)
def test_the_superseded_whole_tranche_specs_say_so_where_a_submit_line_is_read(shape: str):
    """
    THESE TWO FILES CARRY A SUBMIT LINE THAT WOULD NOW BE REFUSED OR WOULD OVERSPEND, AND THEY
    ARE KEPT ANYWAY, so the thing to catch is somebody copying one out of them.

    Both say `--fanout-size 9` and the tranche is twenty cells, because the seed count moved and
    then an arm was funded and ``TRANCHE_CELLS`` followed both. The L40S variant at twenty
    cells, 21 hours and two attempts prices a ceiling near $8,800; the A100 variant is
    superseded outright, because measured seed sigma makes horizon the wrong thing to buy at
    any step time and its shape question was therefore never load-bearing.

    What is worth keeping is the shape: one submission, one approval, one commit for every
    cell, expressed through the ``arm-and-seed`` fan-out, which is the right answer wherever a
    pre-registration does not force the arms apart. So the files stay, the tests above keep
    them coherent, and this one keeps them from being submitted.
    """
    header = (pathlib.Path(_HERE) / arms.STAGED_TRANCHES[shape].spec).read_text()
    header = header.split("schema_version:")[0]

    assert "SUPERSEDED" in header
    assert "DO NOT SUBMIT" in header
    # Naming what replaced it, because "superseded" without a successor sends the reader back
    # to the git log to find out by what.
    for stage in arms.STAGE_SPECS.values():
        assert stage.spec in header, stage.spec

    # And the count it disagrees with is stated rather than left for the reader to discover at
    # admission, where the disagreement is between --fanout-size and a table. Written as digits
    # rather than as a word, because the count has now moved twice and a spelt-out number in a
    # banner is the thing that gets left behind.
    assert str(arms.total_runs()) in header
