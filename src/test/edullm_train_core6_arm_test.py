"""What the bake-off runner MEASURES, executed rather than grepped for.

``src/test/static/mamba_comparison_contract_test.py`` asserts that strings like
``'"throughput_tok_s_steady"'`` and ``"WARMUP_STEPS_EXCLUDED = 50"`` appear in
``.edullm/train_core6_arm.py``. That catches a field being renamed or deleted and catches
nothing else: every endpoint in this file could compute the wrong number, off by one step or
zero instead of null, and every one of those assertions would still pass. The numbers here are
what six arms get ranked against, so the arithmetic wants running and not reading.

The entry point is not importable as a package -- it sits in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path, exactly as
``edullm_train_on_corpus_test.py`` loads its own.

EVERYTHING BELOW RUNS ON A CPU WITH NO KERNEL PACKAGES INSTALLED. The accelerated backends
(FlashRNN, the ahead-of-time Flash-PD CUDA extension, ``fla``'s GDN2, ``mamba_ssm``'s Mamba-3)
are present only in the built image, so the preflight tests supply fakes and monkeypatch
``torch.cuda`` rather than skipping. A preflight that is only exercised on the machine it is
meant to protect is a preflight nobody has tested.
"""

import importlib.util
import sys
import types
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from olmo_core.train.callbacks import ConfigSaverCallback, GPUMemoryMonitorCallback
from olmo_core.train.trainer import Trainer

EDULLM = Path(__file__).parent.parent.parent / ".edullm"


def _load(name: str, filename: str):
    path = EDULLM / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load("edullm_train_core6_arm", "train_core6_arm.py")
arms = _load("model_arch_tests", "model_arch_tests.py")


class FakeTrainer:
    """The two counters ``LossWatcher.post_step`` reads, and nothing else."""

    def __init__(self, step: int = 0, tokens_seen: int = 0) -> None:
        self.global_step = step
        self.global_train_tokens_seen = tokens_seen


class FakeClock:
    """A ``perf_counter`` that only moves when a test moves it.

    Deliberately not a generator of pre-baked values: pytest and the logging module both call
    ``perf_counter`` at unpredictable moments, and a clock that advanced on every read would
    make the recorded step durations depend on who else looked at it.
    """

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def watcher(monkeypatch):
    """A :class:`LossWatcher` wired to a fake trainer and a hand-cranked clock.

    Returns the watcher, its trainer and its clock. ``end_step`` closes one training step out
    the way the trainer does: advance the clock, move the counters, call ``post_step``.
    """
    clock = FakeClock()
    monkeypatch.setattr(entry.time, "perf_counter", clock)
    losses = entry.LossWatcher()
    trainer = FakeTrainer()
    losses._trainer = trainer

    def end_step(*, seconds: float, tokens: int) -> None:
        clock.advance(seconds)
        trainer.global_step += 1
        trainer.global_train_tokens_seen += tokens
        losses.post_step()

    return types.SimpleNamespace(losses=losses, trainer=trainer, clock=clock, end_step=end_step)


# --- what a step sample is a sample OF ---------------------------------------------------


def test_the_first_post_step_starts_the_clock_and_records_no_step(watcher):
    watcher.clock.advance(300.0)  # process start, dataset open, FSDP wrap, the first compile.
    watcher.trainer.global_step = 1
    watcher.trainer.global_train_tokens_seen = 524_288
    watcher.losses.post_step()

    assert watcher.losses.steps == []


def test_each_sample_is_labelled_with_the_step_it_timed_and_the_tokens_that_step_moved(
    watcher,
):
    watcher.clock.advance(300.0)
    watcher.end_step(seconds=0.0, tokens=524_288)  # step 1: starts the clock, records nothing.
    watcher.end_step(seconds=2.0, tokens=524_288)
    watcher.end_step(seconds=3.0, tokens=524_288)

    assert watcher.losses.steps == [
        entry.StepSample(step=2, seconds=2.0, tokens=524_288),
        entry.StepSample(step=3, seconds=3.0, tokens=524_288),
    ]
    # The 300 seconds before step 1 belong to startup and are in no sample.
    assert sum(sample.seconds for sample in watcher.losses.steps) == 5.0


def test_tokens_are_differenced_from_the_trainers_total_not_recomputed_from_a_batch_size(
    watcher,
):
    watcher.end_step(seconds=1.0, tokens=524_288)
    watcher.end_step(seconds=1.0, tokens=524_288)
    # A short final batch: the trainer's total moves by less than a full global batch.
    watcher.end_step(seconds=1.0, tokens=131_072)

    assert [sample.tokens for sample in watcher.losses.steps] == [524_288, 131_072]


def test_a_token_total_that_goes_backwards_is_clamped_to_zero_rather_than_negative(watcher):
    watcher.end_step(seconds=1.0, tokens=524_288)
    watcher.end_step(seconds=1.0, tokens=-1_000)

    assert [sample.tokens for sample in watcher.losses.steps] == [0]


def test_the_watcher_forwards_the_speed_monitors_per_device_averages_and_both_losses(watcher):
    watcher.losses.log_metrics(
        5,
        {
            "train/CE loss": 11.0,
            "throughput/device/TPS (actual avg)": 41_000.0,
            "throughput/device/TPS": 43_000.0,
        },
    )
    watcher.losses.log_metrics(10, {"train/CE loss": 9.5})

    assert watcher.losses.first == 11.0
    assert watcher.losses.last == 9.5
    assert watcher.losses.tps_device_avg == 41_000.0
    assert watcher.losses.tps_device_last == 43_000.0


# --- the warmup boundary -----------------------------------------------------------------


def test_the_frozen_warmup_cutoff_is_fifty_steps():
    assert entry.WARMUP_STEPS_EXCLUDED == 50


@pytest.mark.parametrize("step, kept", [(1, False), (49, False), (50, False), (51, True)])
def test_the_cutoff_is_strict_so_step_fifty_is_discarded_and_step_fiftyone_is_kept(step, kept):
    samples = [entry.StepSample(step=step, seconds=1.0, tokens=10)]

    assert bool(entry.steps_after_warmup(samples)) is kept


def test_exactly_the_warmup_count_of_steps_is_discarded_from_a_run_that_started_at_step_one():
    samples = [entry.StepSample(step=n, seconds=1.0, tokens=10) for n in range(1, 201)]

    kept = entry.steps_after_warmup(samples)

    assert len(samples) - len(kept) == entry.WARMUP_STEPS_EXCLUDED
    assert kept[0].step == entry.WARMUP_STEPS_EXCLUDED + 1


def test_a_resumed_attempt_keeps_every_sample_because_the_cutoff_reads_the_step_index():
    # A second Batch attempt picks up at step 1,201. Dropping "the first 50 entries" would
    # throw away 50 steady-state steps and exclude no startup at all.
    samples = [entry.StepSample(step=n, seconds=1.0, tokens=10) for n in range(1201, 1251)]

    assert entry.steps_after_warmup(samples) == samples


# --- throughput: null, never zero ---------------------------------------------------------


def test_throughput_is_total_tokens_over_total_seconds_and_not_the_mean_of_the_rates():
    samples = [
        entry.StepSample(step=51, seconds=1.0, tokens=100),
        entry.StepSample(step=52, seconds=9.0, tokens=100),
    ]

    # The mean of the two per-step rates would be 55.6, which the run never achieved.
    assert entry.throughput_tokens_per_second(samples) == 20.0


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param([], id="no steps at all"),
        pytest.param(
            [entry.StepSample(step=51, seconds=1.0, tokens=0)], id="loader reported no tokens"
        ),
        pytest.param([entry.StepSample(step=51, seconds=0.0, tokens=100)], id="no time elapsed"),
    ],
)
def test_an_unmeasurable_throughput_is_null_and_never_zero(samples):
    assert entry.throughput_tokens_per_second(samples) is None


def test_the_speed_report_nulls_every_derived_figure_when_no_tokens_moved():
    losses = entry.LossWatcher()
    losses.steps = [entry.StepSample(step=n, seconds=1.0, tokens=0) for n in range(1, 101)]

    report = entry.throughput_report(
        losses, world_size=8, wall_clock_seconds=120.0, flops_per_token=2_000_000_000
    )

    for field in (
        "throughput_tok_s_steady",
        "throughput_tok_s_steady_per_device",
        "throughput_tok_s_whole_run",
        "throughput_tok_s_whole_run_per_device",
        "throughput_tok_s_all_steps",
        "mfu_pct",
    ):
        assert report[field] is None, field
    # A null with no reason beside it is indistinguishable from a bug in the report.
    assert report["mfu_basis"]


def test_the_speed_report_measures_the_steady_window_and_names_the_divisor_it_used():
    losses = entry.LossWatcher()
    # 50 warmup steps at 10s, then 50 steady steps at 1s, all moving the same tokens.
    losses.steps = [
        entry.StepSample(step=n, seconds=10.0 if n <= 50 else 1.0, tokens=1_000)
        for n in range(1, 101)
    ]

    report = entry.throughput_report(losses, world_size=8, wall_clock_seconds=1_000.0)

    assert report["steps_measured"] == 100
    assert report["steady_state_steps"] == 50
    assert report["warmup_steps_excluded"] == entry.WARMUP_STEPS_EXCLUDED
    assert report["tokens_in_steady_window"] == 50_000
    assert report["steady_window_seconds"] == 50.0
    # Steady excludes the slow warmup; whole-run and all-steps do not, and they differ from
    # each other because wall clock includes what happened before the first post_step.
    assert report["throughput_tok_s_steady"] == 1_000.0
    assert report["throughput_tok_s_steady_per_device"] == 125.0
    assert report["throughput_tok_s_all_steps"] == pytest.approx(100_000 / 550.0)
    assert report["throughput_tok_s_whole_run"] == 100.0


def test_mfu_is_null_when_the_card_is_not_in_the_peak_table_and_says_so():
    losses = entry.LossWatcher()
    losses.steps = [entry.StepSample(step=n, seconds=1.0, tokens=1_000) for n in range(1, 101)]

    report = entry.throughput_report(
        losses,
        world_size=1,
        wall_clock_seconds=100.0,
        flops_per_token=2_000_000_000,
        device_name="NVIDIA GeForce RTX 4090",
    )

    assert report["device_peak_bf16_flops"] is None
    assert report["mfu_pct"] is None
    assert "RTX 4090" in report["mfu_basis"]


# --- quantiles ----------------------------------------------------------------------------


def test_a_quantile_of_nothing_is_null_rather_than_a_zero_second_step():
    assert entry.quantile_nearest_rank([], 0.5) is None


@pytest.mark.parametrize("q", [0.0, -0.1, 1.5])
def test_a_quantile_outside_the_unit_interval_is_rejected(q):
    with pytest.raises(ValueError):
        entry.quantile_nearest_rank([1.0, 2.0], q)


@pytest.mark.parametrize(
    "values, q, expected",
    [
        pytest.param([4.0, 1.0, 3.0, 2.0], 0.5, 2.0, id="even n takes the lower middle"),
        pytest.param([3.0, 1.0, 2.0], 0.5, 2.0, id="odd n takes the middle"),
        pytest.param([float(n) for n in range(1, 11)], 0.9, 9.0, id="p90 of ten"),
        pytest.param([float(n) for n in range(1, 11)], 1.0, 10.0, id="q=1 is the max"),
        pytest.param([7.0], 0.9, 7.0, id="a single sample is every quantile"),
    ],
)
def test_the_quantile_is_taken_by_nearest_rank(values, q, expected):
    assert entry.quantile_nearest_rank(values, q) == expected


@pytest.mark.parametrize("q", [0.5, 0.9, 1.0])
def test_every_quantile_is_a_step_time_that_was_actually_observed(q):
    values = [0.31, 2.75, 0.29, 0.30, 11.4, 0.33, 0.28]

    assert entry.quantile_nearest_rank(values, q) in values


# --- peak memory: a running maximum, taken before the monitor resets it ---------------------


@pytest.fixture
def cuda_memory(monkeypatch):
    """Pretend CUDA is present and hand back scripted peak-memory readings."""
    readings = {"allocated": 0, "reserved": 0}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: readings["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: readings["reserved"])
    return readings


def test_peak_memory_is_a_running_max_over_steps_and_not_the_last_step_read(watcher, cuda_memory):
    for allocated, reserved in ((10, 20), (30, 44), (5, 6)):
        cuda_memory["allocated"] = allocated * 1024**3
        cuda_memory["reserved"] = reserved * 1024**3
        watcher.end_step(seconds=1.0, tokens=1_000)

    # The transient peak in the middle step survives the later, smaller readings.
    assert watcher.losses.peak_allocated_bytes == 30 * 1024**3
    assert watcher.losses.peak_reserved_bytes == 44 * 1024**3
    assert watcher.losses.memory_samples == 3


def test_the_memory_report_labels_the_running_max_as_such(watcher, cuda_memory):
    cuda_memory["allocated"] = 30 * 1024**3
    cuda_memory["reserved"] = 44 * 1024**3
    watcher.end_step(seconds=1.0, tokens=1_000)

    report = entry.memory_report(watcher.losses)

    assert report["peak_memory_source"] == "per_step_running_max"
    assert report["peak_memory_gib"] == 30.0
    assert report["peak_memory_reserved_gib"] == 44.0
    assert report["peak_memory_samples"] == 1


def test_a_post_fit_read_with_no_samples_is_labelled_a_last_step_lower_bound(cuda_memory):
    cuda_memory["allocated"] = 7 * 1024**3
    cuda_memory["reserved"] = 9 * 1024**3

    report = entry.memory_report(entry.LossWatcher())

    assert report["peak_memory_source"] == "final_step_only"
    assert report["peak_memory_samples"] == 0
    assert report["peak_memory_gib"] == 7.0


def test_a_run_without_cuda_reports_null_memory_rather_than_zero_gib(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = entry.memory_report(entry.LossWatcher())

    assert report["peak_memory_source"] == "unavailable"
    assert report["peak_memory_gib"] is None
    assert report["peak_memory_reserved_gib"] is None


# --- callback ordering, which is what makes the running max possible ------------------------


def test_the_watcher_samples_memory_before_the_monitor_resets_the_counters():
    # GPUMemoryMonitorCallback.post_step calls reset_peak_memory_stats(). If it ran first the
    # running max above would be a max over one step's worth of allocation.
    assert entry.LossWatcher.priority > GPUMemoryMonitorCallback.priority

    holder = types.SimpleNamespace(
        callbacks=OrderedDict(
            # The registration order build_config and train() actually use.
            [
                ("gpu_monitor", GPUMemoryMonitorCallback()),
                ("config_saver", ConfigSaverCallback()),
                ("edullm_losses", entry.LossWatcher()),
            ]
        )
    )
    # The trainer's own sort, run over the trainer's own registry, rather than the rule
    # restated here. It reads nothing off `self` but `callbacks`.
    Trainer._sort_callbacks(cast(Trainer, holder))
    order = list(holder.callbacks)

    assert order.index("edullm_losses") < order.index("gpu_monitor")


# --- the preflight, per arm -----------------------------------------------------------------


def _opts(arm: str, **overrides):
    values = {
        "arm": arm,
        "rank_microbatch_size": 8192,
        "sequence_length": 4096,
        # The only choice --param-dtype accepts, and what the preflight asks the card about.
        "param_dtype": "bfloat16",
        # A real cell's seed, because the preflight reads the model this run is about to
        # build to find out which kernel it has to compile, and that model is built from
        # the arm and the init seed together.
        "init_seed": arms.INIT_SEEDS_BY_ARM[arm][arms.DATA_SEEDS[0]],
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _realised_slstm(arm: str = "xlstm"):
    """The model the runner would train, and the sLSTM mixers inside it.

    Read back off that model rather than repeated as literals here: a test that names the
    kernel dtype, the head count or the batch a second time passes for exactly as long as
    the two copies agree, which is the failure it is supposed to catch.
    """
    from olmo_core.nn.xlstm import SLSTMMixerConfig

    model = arms.build_model_config(arm, arms.INIT_SEEDS_BY_ARM[arm][arms.DATA_SEEDS[0]])
    return model, [
        block.sequence_mixer
        for block in model.resolved_block_configs
        if isinstance(block.sequence_mixer, SLSTMMixerConfig)
    ]


def _realised_slstm_kernel_dtypes(arm: str = "xlstm") -> set[str]:
    return {mixer.kernel_dtype for mixer in _realised_slstm(arm)[1]}


def _move_slstm_layers(monkeypatch, mover) -> None:
    """Rebuild the xLSTM arm with each of its sLSTM mixers passed through ``mover``.

    ``build_model_config`` asserts the arm's exact frozen parameter count, and head count and
    batch are geometry -- eight heads is a different model from four -- so the mutations the
    prewarm has to follow cannot go through that assertion. The widths are the real arm's,
    solved before the move, so everything the preflight reads other than the moved field is
    the model this run would train.
    """
    from olmo_core.nn.xlstm import SLSTMMixerConfig

    widths = arms.solve_widths("xlstm")
    treatment = arms._treatment_mixer

    def moved(arm: str, layer_index: int):
        mixer = treatment(arm, layer_index)
        if isinstance(mixer, SLSTMMixerConfig):
            return mover(layer_index, mixer)
        return mixer

    monkeypatch.setattr(arms, "_treatment_mixer", moved)
    monkeypatch.setattr(
        arms,
        "build_model_config",
        lambda arm, init_seed: arms._model_for_widths(arm, widths, init_seed),
    )


@pytest.fixture
def cuda_a100(monkeypatch):
    """An sm_80 device on rank-local ordinal 3, so a device-index bug cannot hide behind 0.

    ``is_bf16_supported`` is answered here too, and truthfully for this card: torch's own
    implementation reads the real driver, so on the CPU box these tests run on it would either
    contradict the sm_80 this fixture is pretending to be or raise while trying to look.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *_, **__: True)


@pytest.fixture
def installed(monkeypatch):
    """Report chosen package versions to ``importlib.metadata`` and hide everything else."""
    import importlib.metadata

    def install(**versions):
        def version(package):
            try:
                return versions[package]
            except KeyError:
                raise importlib.metadata.PackageNotFoundError(package) from None

        monkeypatch.setattr(importlib.metadata, "version", version)

    return install


@pytest.fixture
def flashrnn(monkeypatch):
    """Stand in for the two FlashRNN entry points, recording what the prewarm was handed."""
    import olmo_core.nn.xlstm as xlstm

    calls = types.SimpleNamespace(preflighted=0, prewarm_kwargs=None)

    def preflight():
        calls.preflighted += 1

    def prewarm(**kwargs):
        calls.prewarm_kwargs = kwargs

    monkeypatch.setattr(xlstm, "_preflight_flashrnn", preflight, raising=True)
    monkeypatch.setattr(xlstm, "_prewarm_flashrnn", prewarm, raising=True)
    return calls


XLSTM_PINS = {"xlstm": "2.0.5", "mlstm-kernels": "2.0.4", "flashrnn": "1.0.6"}


@pytest.fixture
def native_extension(monkeypatch):
    """Install a fake ahead-of-time Flash-PD extension exporting the named symbols."""
    import olmo_core.nn.flash_pd_native.cuda as native_cuda

    def install(*symbols):
        extension = types.SimpleNamespace(**{name: (lambda *_, **__: None) for name in symbols})
        monkeypatch.setattr(native_cuda, "_EXTENSION", extension)
        return extension

    return install


@pytest.fixture
def fla_module(monkeypatch):
    """Install a fake ``fla`` package, optionally carrying ``fla.ops.gdn2.chunk_gdn2``."""
    from olmo_core.nn.attention import flash_linear_attn_api

    def install(*, gdn2=True):
        fla = types.ModuleType("fla")
        ops = types.ModuleType("fla.ops")
        gdn2_module = types.ModuleType("fla.ops.gdn2")
        if gdn2:
            setattr(gdn2_module, "chunk_gdn2", lambda *_, **__: None)
        setattr(ops, "gdn2", gdn2_module)
        setattr(fla, "ops", ops)
        for name, module in (
            ("fla", fla),
            ("fla.ops", ops),
            ("fla.ops.gdn2", gdn2_module),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(flash_linear_attn_api, "fla", fla)
        return fla

    return install


@pytest.fixture
def no_kernels(monkeypatch, installed):
    """An image carrying none of the accelerated kernels. Says nothing about the device.

    Spelled out rather than left to whatever the developer's machine happens to lack: this
    repository's own test environment ships an unpinned ``fla``, so a contract test that
    passed because of what was installed on one afternoon would not be a contract.

    THE CARD IS A SEPARATE AXIS and every test below pairs this with one. A missing kernel
    proved on a host that also has no GPU proves nothing about which of the two was refused,
    and the order of those two checks is exactly what is under test here.
    """
    import olmo_core.nn.flash_pd_native.cuda as native_cuda
    from olmo_core.nn.attention import flash_linear_attn_api
    from olmo_core.nn.mamba3 import mamba3_ssd_api

    installed()
    monkeypatch.setattr(native_cuda, "_EXTENSION", None)
    monkeypatch.setattr(flash_linear_attn_api, "fla", None)
    monkeypatch.setattr(mamba3_ssd_api, "has_mamba3", lambda: False)


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_every_runnable_arm_is_refused_on_a_host_that_has_none_of_its_kernels(
    arm, cuda_a100, no_kernels
):
    # Every arm in the comparison names a kernel and none of them falls back, so every one of
    # them has to say so here rather than thirty seconds into a billed machine -- and it has to
    # say so as a staged Refusal, since the exit code is the only channel out of the container.
    # An arm added to RUNNABLE_ARMS with no preflight branch fails this and nothing else.
    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert refused.value.explanation


def test_xlstm_prewarms_flashrnn_on_this_ranks_own_device(cuda_a100, installed, flashrnn):
    installed(**XLSTM_PINS)

    entry.preflight_accelerated_arm(_opts("xlstm"))

    assert flashrnn.preflighted == 1
    device = flashrnn.prewarm_kwargs["device"]
    # torch.cuda.current_device(), not 0: on an 8-GPU node every rank but one would prewarm
    # and then measure a card belonging to somebody else.
    assert device == torch.device("cuda", 3)
    assert flashrnn.prewarm_kwargs["batch_size"] == 8192 // 4096
    assert flashrnn.prewarm_kwargs["seq_len"] == 4096


def test_the_prewarm_compiles_the_kernel_dtype_the_slstm_layers_are_built_with(
    cuda_a100, installed, flashrnn
):
    # The prewarm exists to pay the FlashRNN JIT before the measured steps. A dtype it does
    # not share with the arm buys nothing: the kernel training calls compiles inside the
    # first step anyway, and the throughput this study ranks arms on carries that compile.
    installed(**XLSTM_PINS)

    entry.preflight_accelerated_arm(_opts("xlstm"))

    assert _realised_slstm_kernel_dtypes() == {flashrnn.prewarm_kwargs["kernel_dtype"]}


def test_the_prewarm_dtype_follows_the_arm_rather_than_being_named_a_second_time(
    cuda_a100, installed, flashrnn, monkeypatch
):
    # Move the ledger's sLSTM layers to a THIRD dtype, so neither the arm's current pin nor
    # the float32 this call used to ask for can pass by coincidence. This is the drift the
    # runner already suffered once: the arm went to bfloat16 and the preflight did not.
    moved = replace(arms._slstm_mixer(), kernel_dtype="float16")
    monkeypatch.setattr(arms, "_slstm_mixer", lambda: moved)
    installed(**XLSTM_PINS)

    entry.preflight_accelerated_arm(_opts("xlstm"))

    assert (
        _realised_slstm_kernel_dtypes() == {"float16"} == {flashrnn.prewarm_kwargs["kernel_dtype"]}
    )


def test_the_prewarm_head_geometry_follows_the_arm_rather_than_being_named_a_second_time(
    cuda_a100, installed, flashrnn, monkeypatch
):
    # Move the ledger's sLSTM layers to a head count this file has never been told about, so
    # neither the four the call used to name nor the 1024 // 4 beside it can pass by
    # coincidence. Both numbers are in FlashRNN's compile cache key, so a prewarm that keeps
    # its own copy of them compiles an artifact no step calls and leaves the kernel the arm
    # does call to compile inside the first measured step.
    _move_slstm_layers(monkeypatch, lambda _, mixer: replace(mixer, n_heads=8))
    installed(**XLSTM_PINS)

    entry.preflight_accelerated_arm(_opts("xlstm"))

    model, mixers = _realised_slstm()
    assert {mixer.n_heads for mixer in mixers} == {8}
    assert flashrnn.prewarm_kwargs["n_heads"] == 8
    # The head dimension is the layer's own -- the model width over its heads, which is what
    # xlstm's sLSTMLayerConfig computes -- and not a constant that happens to divide it.
    assert flashrnn.prewarm_kwargs["head_dim"] == model.d_model // 8 == 128


def test_the_prewarm_batch_is_the_one_the_slstm_layers_were_compiled_to_accept(
    cuda_a100, installed, flashrnn, monkeypatch
):
    # The persistent FlashRNN layer is built for one batch and refuses every other at its
    # first forward, so the batch that gets warmed has to be the layer's own rather than the
    # ledger's 2 written down here a second time.
    _move_slstm_layers(monkeypatch, lambda _, mixer: replace(mixer, batch_size=4))
    installed(**XLSTM_PINS)

    entry.preflight_accelerated_arm(_opts("xlstm", rank_microbatch_size=4 * 4096))

    assert {mixer.batch_size for mixer in _realised_slstm()[1]} == {4}
    assert flashrnn.prewarm_kwargs["batch_size"] == 4


@pytest.mark.parametrize(
    "rank_microbatch_size",
    [
        # Four sequences a rank against layers built for two: the trainer splits a rank's
        # microbatch into rank_microbatch_size // sequence_length instances, so this is the
        # batch the layer is handed and it is not the batch it was compiled for.
        pytest.param(4 * 4096, id="feeds four sequences into a two-sequence layer"),
        # Not a whole number of sequences at all, so the floor that used to be handed to the
        # prewarm was not the batch anything would run.
        pytest.param(8192 + 1000, id="not a whole number of sequences"),
    ],
)
def test_a_rank_microbatch_that_does_not_feed_the_layers_batch_is_refused_before_the_prewarm(
    rank_microbatch_size, cuda_a100, installed, flashrnn
):
    installed(**XLSTM_PINS)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("xlstm", rank_microbatch_size=rank_microbatch_size))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    # Both halves of the disagreement, because "wrong batch" without the two numbers sends
    # the reader back to the command line to work out which of them to move.
    assert str(rank_microbatch_size) in refused.value.explanation
    assert str(_realised_slstm()[1][0].batch_size) in refused.value.explanation
    assert flashrnn.prewarm_kwargs is None


@pytest.mark.parametrize(
    "moved",
    [
        pytest.param({"kernel_dtype": "float32"}, id="dtype"),
        pytest.param({"n_heads": 8}, id="head count"),
        pytest.param({"batch_size": 4}, id="batch"),
    ],
)
def test_slstm_layers_that_disagree_on_the_prewarm_contract_are_refused_rather_than_half_warmed(
    moved, cuda_a100, installed, flashrnn, monkeypatch
):
    # One prewarm compiles one shape at one dtype, so an arm whose two sLSTM layers ask for
    # different ones cannot be warmed by it. Picking either would leave the other compiling
    # in the measured window, which is the same wrong number this fix removes, arrived at
    # more quietly.
    _move_slstm_layers(
        monkeypatch,
        lambda index, mixer: replace(mixer, **moved)
        if index == arms.XLSTM_SLSTM_LAYERS[0]
        else mixer,
    )
    installed(**XLSTM_PINS)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("xlstm"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert flashrnn.prewarm_kwargs is None


@pytest.mark.parametrize("package", sorted(XLSTM_PINS))
def test_xlstm_refuses_a_kernel_package_that_moved_off_its_pin(
    package, cuda_a100, installed, flashrnn
):
    installed(**{**XLSTM_PINS, package: "9.9.9"})

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("xlstm"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert package in refused.value.explanation
    assert flashrnn.preflighted == 0


def test_xlstm_refuses_a_card_that_is_not_sm80(installed, flashrnn, monkeypatch):
    installed(**XLSTM_PINS)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (8, 9))

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("xlstm"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert flashrnn.preflighted == 0


@pytest.mark.parametrize(
    "arm, symbols",
    [
        ("native-pd", ("forward", "paper_backward")),
        ("mamba3-siso-pd", ("mamba3_forward", "paper_backward")),
    ],
)
def test_a_flash_pd_arm_passes_when_the_extension_exports_the_entry_points_it_calls(
    arm, symbols, cuda_a100, native_extension
):
    native_extension(*symbols)

    entry.preflight_accelerated_arm(_opts(arm))


@pytest.mark.parametrize(
    "arm, present, missing",
    [
        # Each arm's own entry points, and nothing else, is what makes its build usable: the
        # native-pd mixer runs forward/paper_backward and the SISO mixer runs
        # mamba3_forward/paper_backward, so a build carrying one arm's symbols is no evidence
        # for the other's.
        ("native-pd", ("forward",), "paper_backward"),
        ("native-pd", ("paper_backward",), "forward"),
        ("mamba3-siso-pd", ("forward", "paper_backward"), "mamba3_forward"),
        ("mamba3-siso-pd", ("mamba3_forward",), "paper_backward"),
    ],
)
def test_a_flash_pd_arm_names_the_extension_symbol_its_build_is_missing(
    arm, present, missing, cuda_a100, native_extension
):
    native_extension(*present)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert missing in refused.value.explanation


@pytest.mark.parametrize("arm", ["native-pd", "mamba3-siso-pd"])
def test_a_flash_pd_arm_refuses_a_host_with_no_cuda_device(arm, native_extension, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    native_extension("forward", "backward", "mamba3_forward", "paper_backward")

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START


GDN2_PINS = {"flash-linear-attention": "0.5.1", "fla-core": "0.5.1"}


def test_gdn_passes_on_the_pinned_fla_that_ships_chunk_gdn2(cuda_a100, fla_module, installed):
    fla_module()
    installed(**GDN2_PINS)

    entry.preflight_accelerated_arm(_opts("gdn"))


@pytest.mark.parametrize("package", sorted(GDN2_PINS))
def test_gdn_refuses_when_either_half_of_fla_moved_off_the_pin(
    package, cuda_a100, fla_module, installed
):
    # fla and fla-core are two distributions of one kernel library and the GDN2 op moved
    # between them; a run that read one pin and not the other would measure whichever
    # implementation happened to win the import.
    fla_module()
    installed(**{**GDN2_PINS, package: "0.5.0"})

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("gdn"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert package in refused.value.explanation


def test_gdn_refuses_a_pinned_fla_whose_build_has_no_chunk_gdn2(cuda_a100, fla_module, installed):
    fla_module(gdn2=False)
    installed(**GDN2_PINS)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("gdn"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert "chunk_gdn2" in refused.value.explanation


def test_mamba_b3_passes_when_the_official_mamba3_kernel_imports(cuda_a100, monkeypatch):
    from olmo_core.nn.mamba3 import mamba3_ssd_api

    monkeypatch.setattr(mamba3_ssd_api, "has_mamba3", lambda: True)

    entry.preflight_accelerated_arm(_opts("mamba-b3"))


def test_mamba_b3_refuses_when_the_official_mamba3_kernel_is_absent(cuda_a100, monkeypatch):
    from olmo_core.nn.mamba3 import mamba3_ssd_api

    monkeypatch.setattr(mamba3_ssd_api, "has_mamba3", lambda: False)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts("mamba-b3"))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START


# --- the card, which every arm shares and only xLSTM used to ask about -----------------------


@pytest.fixture
def every_kernel(monkeypatch, installed, flashrnn, native_extension, fla_module):
    """An image carrying all five arms' kernels, so the device is the only thing left to refuse.

    The mirror of ``no_kernels``: with every package present and pinned, a refusal below can
    only have come from the device, which is what makes these tests statements about the order
    of the two checks rather than about either one on its own.
    """
    from olmo_core.nn.mamba3 import mamba3_ssd_api

    installed(**XLSTM_PINS, **GDN2_PINS)
    native_extension("forward", "paper_backward", "mamba3_forward")
    fla_module()
    monkeypatch.setattr(mamba3_ssd_api, "has_mamba3", lambda: True)


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_every_runnable_arm_is_refused_on_a_host_with_no_cuda_device_at_all(
    arm, every_kernel, monkeypatch
):
    # A wheel's symbols are in the wheel whatever machine it was unpacked on, so four of the
    # five arms used to pass their own probe on a CPU-only host and die inside the first step.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    assert refused.value.explanation


@pytest.mark.parametrize(
    "capability",
    [
        # An L4/4090-shaped card: newer than sm_80, and every kernel here is written for sm_80
        # rather than for "at least sm_80".
        pytest.param((8, 9), id="sm_89"),
        # A T4. Turing, and the shape the platform's own precision guard exists for.
        pytest.param((7, 5), id="sm_75"),
    ],
)
@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_every_runnable_arm_is_refused_on_a_card_that_is_not_sm80(
    arm, capability, every_kernel, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START
    # Both shapes, because "wrong card" without the two numbers is a sentence that sends the
    # reader back to the Batch console to find out what they were given.
    assert f"sm_{capability[0]}{capability[1]}" in refused.value.explanation
    assert "sm_80" in refused.value.explanation


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_every_runnable_arm_is_refused_when_the_card_cannot_do_the_requested_precision(
    arm, cuda_a100, every_kernel, monkeypatch
):
    # DEFENSE TWO. Defense one is the platform's guard, which reads --param-dtype bfloat16 out
    # of the command text and refuses the shape before a machine is billed; it can only see
    # words. This asks the card that actually turned up, and it is what stands between a run
    # and the first kernel that needs a format the hardware does not have.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *_, **__: False)

    with pytest.raises(entry.Refusal) as refused:
        entry.preflight_accelerated_arm(_opts(arm))

    assert refused.value.stage is entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION
    assert "bfloat16" in refused.value.explanation


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_every_runnable_arm_passes_on_the_a100_the_comparison_is_defined_on(
    arm, cuda_a100, every_kernel
):
    # The other half of the contract. A guard that refused the machine the study runs on would
    # be caught by nothing above, since every test up there is asserting a refusal.
    entry.preflight_accelerated_arm(_opts(arm))


def test_the_precision_refusal_is_its_own_exit_code_and_not_a_held_out_one():
    precision = entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION

    # 73 and 74 are the held-out endpoint. Reusing either would tell every reader of the exit
    # code, and every grep of this file, that a corpus declared no held-out split -- about a
    # failure that never opened the corpus.
    assert int(precision) == 75
    assert precision not in (
        entry.Stage.THE_CORPUS_DECLARES_NO_HELD_OUT_SPLIT,
        entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
        entry.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
    )
    # An IntEnum silently ALIASES a repeated value, so a number already spent would not be a
    # new stage at all -- it would be one of the above under a second name.
    numbers = [member.value for member in entry.Stage.__members__.values()]
    assert len(numbers) == len(set(numbers))


def test_the_precision_refusal_reaches_the_container_as_that_exit_code(monkeypatch):
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)

    def refuse() -> None:
        raise entry.Refusal(
            entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, "no bfloat16 here"
        )

    monkeypatch.setattr(entry, "main", refuse)

    assert entry.cli() == int(entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION)


# --- what a dry run is allowed to need, which is no GPU at all -------------------------------


@pytest.fixture
def run_main(monkeypatch):
    """Drive ``main()`` over stubbed collaborators, recording the order it called them in."""
    called: list[str] = []

    def record(name: str, result=None):
        def stub(*_, **__):
            called.append(name)
            return result

        return stub

    for variable, value in (
        ("EDULLM_DATASET_ID", "pretrain/regmix-10b"),
        ("EDULLM_DATASET_VERSION", "v1"),
        ("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe"),
        ("EDULLM_CHECKPOINT_DIR", "s3://bucket/teams/t/runs/r/checkpoints/"),
    ):
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(entry, "build_config", record("build_config", object()))
    monkeypatch.setattr(entry, "show", record("show"))
    monkeypatch.setattr(entry, "prepare_training_environment", record("prepare"))
    monkeypatch.setattr(entry, "preflight_accelerated_arm", record("preflight"))
    monkeypatch.setattr(entry, "train", record("train"))
    monkeypatch.setattr(entry, "teardown_training_environment", record("teardown"))
    # The machine these tests run on, said out loud: a dry run has to survive it.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    data_seed = arms.DATA_SEEDS[0]

    def run(*extra: str) -> list[str]:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train_core6_arm.py",
                "a-run-id",
                "--arm",
                "mamba-b3",
                "--data-seed",
                str(data_seed),
                "--init-seed",
                str(arms.INIT_SEEDS_BY_ARM["mamba-b3"][data_seed]),
                *extra,
            ],
        )
        entry.main()
        return called

    return types.SimpleNamespace(run=run, called=called)


def test_a_dry_run_resolves_and_prints_without_asking_for_a_gpu(run_main):
    # --dry-run answers "what would this run be", which is a question about the config and the
    # corpus and not about the card. Forcing the A100 guard onto it would make the one command
    # a researcher can run on a laptop refuse on the laptop.
    assert run_main.run("--dry-run") == ["build_config", "show"]


def test_a_real_run_reaches_the_preflight_once_the_training_environment_is_up(run_main):
    # And the same file on the training path is the opposite: the device is checked, after the
    # process group exists so the check is of THIS rank's card, and before a step is taken.
    assert run_main.run() == ["build_config", "prepare", "preflight", "train", "teardown"]


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_the_runner_exempts_the_same_timescale_parameters_the_ledger_declares(arm, monkeypatch):
    """The runner is what every spec dispatches, so its optimizer is the one that trains.

    The ledger beside it already declares which parameters each arm keeps out of weight
    decay. This asserts the runner asks the ledger rather than carrying a second, shorter
    list -- the shape of drift that decays ``A_log``/``dt_bias``/``D`` for eleven hours while
    every printed field still reads correctly.
    """
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig

    corpus = entry.Corpus(
        dataset_id="pretrain/reservoir-dolma2",
        version="v1",
        paths=["s3://reader-owned/train-00000.u32le.bin"],
        dtype=NumpyDatasetDType.uint32,
        tokenizer=TokenizerConfig.dolma2(),
        rows=250_242_924_544,
        val_paths=["s3://reader-owned/val-00000.u32le.bin"],
        val_rows=975_077_376,
    )
    monkeypatch.setattr(entry, "resolve_corpus", lambda **_: corpus)

    data_seed = arms.DATA_SEEDS[0]
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--arm",
            arm,
            "--data-seed",
            str(data_seed),
            "--init-seed",
            str(arms.INIT_SEEDS_BY_ARM[arm][data_seed]),
            "--save-folder",
            "s3://bucket/checkpoints/",
            "--dataset-id",
            "pretrain/reservoir-dolma2",
            "--dataset-version",
            "v1",
            "--dataset-tokenizer",
            "tokenizer/dolma2-bpe",
        ]
    )

    config = entry.build_config(opts, overrides)

    assert config.train_module.optim.group_overrides == arms.weight_decay_group_overrides(arm)


def test_a_refused_preflight_still_tears_the_training_environment_down(run_main, monkeypatch):
    def refuse(*_) -> None:
        raise entry.Refusal(
            entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, "no bfloat16 here"
        )

    monkeypatch.setattr(entry, "preflight_accelerated_arm", refuse)

    with pytest.raises(entry.Refusal) as refused:
        run_main.run()

    assert refused.value.stage is entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION
    # A refusal that leaves the process group up hangs the container instead of exiting with
    # the number the refusal exists to report.
    assert run_main.called == ["build_config", "prepare", "teardown"]


# --- the decode geometry, which is this study's and not the previous bake-off's --------------


def _recurrent_mixers(arm: str) -> list:
    """The mixer configuration the ledger actually builds into each of an arm's non-attention
    layers, read off the built model rather than described a second time here."""
    model = arms.build_model_config(arm, arms.INIT_SEEDS_BY_ARM[arm][arms.DATA_SEEDS[0]])
    return [
        block.sequence_mixer
        for index, block in enumerate(model.resolved_block_configs)
        if index not in set(arms.ATTENTION_LAYERS)
    ]


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_the_decode_geometry_reads_this_studys_layers_off_the_ledger(arm):
    """The decode probe's geometry is the ledger's, and the ledger is the only one in the tree.

    It used to be read out of an ``olmo_core.nn.transformer.core6_arms`` registry belonging to
    the previous mixer bake-off, which describes two mixer slots of sixteen and does not exist
    here at all. Every committed spec passes ``--no-decode-probe``, so the import error was
    swallowed into ``decode_fast_path_taken: false`` and cost nothing -- until the first person
    who turns the probe on and is handed a missing measurement with a plausible-looking reason.
    """
    assert importlib.util.find_spec("olmo_core.nn.transformer.core6_arms") is None

    geometry = entry._decode_geometry(arm)

    assert geometry["mixer_layers"] == len(arms.RECURRENT_LAYERS) == 12
    assert geometry["attention_layers"] == len(arms.ATTENTION_LAYERS) == 4
    assert geometry["total_layers"] == arms.N_LAYERS == 16
    # Every recurrence the arm carries is named. ``xlstm`` runs two of them, and one of the two
    # quietly standing in for both is how a decode figure comes to describe a model nobody ran.
    for mixer in _recurrent_mixers(arm):
        assert type(mixer).__name__ in geometry["config_class"]


@pytest.mark.parametrize("arm", arms.RUNNABLE_ARMS)
def test_the_decode_probe_reports_the_ledger_geometry_on_a_host_with_no_device(arm, monkeypatch):
    """No GPU means no latency, and it does not mean no geometry.

    The footprint and the KV-cache contrast are arithmetic on the arm's own layers, which is
    why they are emitted with no device present. The probe still records nothing measured, and
    it must not be recording that because it could not find out what the arm is.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    probe = entry.decode_probe(arm_name=arm)

    assert "geometry could not be read" not in probe["decode_basis"]
    assert probe["decode_fast_path_taken"] is False
    assert probe["decode_mixer_layers"] == len(arms.RECURRENT_LAYERS)
    assert probe["decode_attention_layers"] == len(arms.ATTENTION_LAYERS)
    assert probe["decode_total_layers"] == arms.N_LAYERS
    assert probe["decode_kv_bytes_per_token"] > 0


@pytest.mark.parametrize("arm", ("gdn", "mamba-b3"))
def test_the_decode_footprint_is_the_head_geometry_the_arm_itself_declares(arm, monkeypatch):
    """Over twelve layers, not two, and with the state axis each mixer actually accumulates on.

    Mamba-3 accumulates over ``d_state`` per head where the delta-rule control accumulates over
    its head dimension, so a single literal here would report one of the two arms wrong.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    shapes = {
        (
            mixer.n_heads,
            getattr(mixer, "d_state", None) or mixer.head_dim,
            int(mixer.head_dim * getattr(mixer, "expand_v", 1.0)),
        )
        for mixer in _recurrent_mixers(arm)
    }
    assert len(shapes) == 1
    heads, head_k, head_v = shapes.pop()

    probe = entry.decode_probe(arm_name=arm)

    assert probe["decode_head_k_dim"] == head_k
    assert probe["decode_head_v_dim"] == head_v
    assert probe["decode_state_elems_per_layer"] == heads * head_k * head_v
    # fp32 state over every layer that carries the recurrence -- the field a serving fleet is
    # sized with, and the one the bake-off's two slots would have reported at a sixth of.
    assert probe["decode_state_bytes_per_seq"] == (
        heads * head_k * head_v * len(arms.RECURRENT_LAYERS) * 4
    )


@pytest.mark.parametrize("arm", ("xlstm", "native-pd", "mamba3-siso-pd"))
def test_an_arm_whose_mixer_declares_no_head_dimension_reports_no_footprint(arm, monkeypatch):
    """A null footprint with a stated cause, never ``d_model // n_heads`` standing in for one."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # Read off the ledger rather than asserted about it: these mixers declare heads and, where
    # they have one, a state width, and no head dimension at all. Dividing d_model by the head
    # count would guess the layout of a state this file has never seen -- wrong in the field
    # that decides serving batch size, and entirely reasonable-looking while it is wrong.
    assert not any(getattr(mixer, "head_dim", None) for mixer in _recurrent_mixers(arm))

    probe = entry.decode_probe(arm_name=arm)

    assert probe["decode_state_elems_per_layer"] is None
    assert probe["decode_state_bytes_per_seq"] is None
    assert probe["decode_mixer_layers"] == len(arms.RECURRENT_LAYERS)
    assert "geometry could not be read" not in probe["decode_basis"]
