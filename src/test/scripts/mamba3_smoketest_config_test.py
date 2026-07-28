"""
Tests for the mamba3-test.py smoke-test config.

These assert the configuration is safe to run on a single GPU outside Beaker, which is the
only way it is ever actually run. Every check here guards a failure that is silent or
near-silent: a data root pointing at storage the machine cannot reach, a data-parallel mesh
that degenerates at world size 1, and a run with no instrumentation to notice either.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.experiment import CliContext, SubCmd

SCRIPT = Path("src/scripts/train/smoketests/mamba3-test.py")


@pytest.fixture(scope="module")
def smoketest() -> ModuleType:
    """
    Load the hyphenated script as a module.

    The directory goes on ``sys.path`` because that is what happens when the script is run
    directly (``python src/scripts/train/smoketests/mamba3-test.py``), and the script imports
    its sibling sentinel module.
    """
    assert SCRIPT.exists(), f"{SCRIPT} not found; run pytest from the repo root"
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("mamba3_smoketest", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _ctx(*overrides: str) -> CliContext:
    return CliContext(
        script=str(SCRIPT),
        cmd=SubCmd.train,
        run_name="test-run",
        cluster="local",
        overrides=list(overrides),
    )


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch):
    """No test may inherit a data root from the developer's shell."""
    for var in ("MAMBA3_DATA_ROOT", "MAMBA3_SAVE_FOLDER", "MAMBA3_WORK_DIR"):
        monkeypatch.delenv(var, raising=False)


def test_missing_data_root_fails_loudly_off_beaker(smoketest, monkeypatch: pytest.MonkeyPatch):
    """
    Without Beaker credentials and without an explicit data root, building the config must
    raise and name the variable to set.

    The failure this guards is the expensive kind. Falling back to ``gs://ai2-llm`` on a
    machine with no GCS credentials does not fail at startup; it fails whenever the loader
    first reaches for a shard, potentially minutes into a run that has already claimed a GPU.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    with pytest.raises(ValueError, match="MAMBA3_DATA_ROOT"):
        smoketest.build_experiment_config(_ctx())


def test_explicit_data_root_reaches_both_dataset_configs(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    ``mix_base_dir`` is set in two places -- the training dataset and the LM evaluator's eval
    dataset. Both must follow the single data root, or an override silently fixes training
    while leaving evaluation pointed at unreachable storage.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))

    config = smoketest.build_experiment_config(_ctx())

    assert str(config.dataset.mix_base_dir) == str(tmp_path)
    eval_dataset = config.trainer.callbacks["lm_evaluator"].eval_dataset
    assert str(eval_dataset.mix_base_dir) == str(tmp_path)
    for name, value in (
        ("dataset", config.dataset.mix_base_dir),
        ("lm_evaluator", eval_dataset.mix_base_dir),
    ):
        assert not str(value).startswith("gs://"), f"{name} still points at GCS"


def test_data_parallel_is_not_degenerate_hsdp(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    HSDP's replicate dimension is the node count, which is 1 for this single-node smoke test,
    so HSDP can only ever build a degenerate mesh here. It does not error -- it just adds a
    second mesh dimension of size 1 and the wrapping cost that comes with it.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))

    dp_config = smoketest.build_experiment_config(_ctx()).train_module.dp_config
    assert dp_config is not None
    assert dp_config.name != DataParallelType.hsdp


def test_sentinel_callback_is_attached(smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    The run must carry the silent-failure sentinel.

    Without it nothing detects a non-finite gradient norm (the trainer's finiteness check reads
    only the CE loss), a stalled skip-step optimizer, or a decay horizon shorter than the
    context -- all of which leave the GPU busy and the loss curve looking plausible.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))

    callbacks = smoketest.build_experiment_config(_ctx()).trainer.callbacks
    assert "mamba3_sentinel" in callbacks, f"got callbacks: {sorted(callbacks)}"
    sentinel = callbacks["mamba3_sentinel"]
    assert sentinel.sequence_length == smoketest.SEQ_LENGTH


def test_behaviour_flags_default_to_the_multi_gpu_recipe(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    The flags that change *what is tested* must default off, so an unflagged run is the real
    B200 recipe: data-parallel on, no activation checkpointing, no autocast standing in for
    FSDP's cast. These used to be environment variables, where an exported value from an
    earlier single-GPU debugging session would silently follow you into the next run.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))

    assert smoketest.ACTIVATION_CHECKPOINTING is False
    assert smoketest.DISABLE_DP is False

    train_module = smoketest.build_experiment_config(_ctx()).train_module
    assert train_module.ac_config is None
    assert train_module.dp_config is not None
    assert train_module.autocast_precision is None


def test_disable_dp_drops_the_mesh_and_restores_the_cast(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    ``--disable-dp`` has to do both halves. Dropping ``dp_config`` alone would leave the model
    in fp32, so the SSD input would not be reduced precision and dispatch would quietly fall
    through to the chunked path -- a single-GPU run that no longer tests the fused kernel.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))
    monkeypatch.setattr(smoketest, "DISABLE_DP", True)

    train_module = smoketest.build_experiment_config(_ctx()).train_module
    assert train_module.dp_config is None
    assert train_module.autocast_precision is not None


def test_activation_checkpointing_flag_reaches_ac_config(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """``ac_config`` defaults to ``None``, so a dotted override cannot create it; the flag must."""
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))
    monkeypatch.setattr(smoketest, "ACTIVATION_CHECKPOINTING", True)

    assert smoketest.build_experiment_config(_ctx()).train_module.ac_config is not None


def test_requiring_the_fast_kernel_under_checkpointing_is_rejected(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    The official kernel's ``autograd.Function`` cannot run under non-reentrant checkpointing.
    Caught at build time, this is a one-line usage error; uncaught it is a ``CheckpointError``
    partway through the first backward.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))
    monkeypatch.setattr(smoketest, "ACTIVATION_CHECKPOINTING", True)
    monkeypatch.setattr(smoketest, "REQUIRE_FAST_KERNEL", True)

    with pytest.raises(SystemExit, match="--require-fast-kernel"):
        smoketest.build_experiment_config(_ctx())


def test_memory_horizon_covers_the_sequence_length(
    smoketest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """
    ``a_log_init_max`` must be set so the state can retain a signal across the whole context.

    At the library default of 16, ``alpha ~= 0.92`` and the longest-memory head of a built
    190M model holds a signal for ~114 steps -- against a 512-token context. Such a run trains
    happily and learns nothing at range, which is indistinguishable from success on the loss
    curve alone.

    The bound is derived rather than hard-coded, and it uses the *median* head. A best-case
    bound (smallest decay paired with the smallest step size) gives ~863 steps at
    ``a_log_init_max=16`` and would pass, but that corner is not drawn with only 12 heads: the
    measured horizon on a built 190M model at that setting is 22-114 steps. The median tracks
    the measurement, so that is what is asserted.
    """
    monkeypatch.setattr(smoketest, "get_beaker_username", lambda: None)
    monkeypatch.setenv("MAMBA3_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("MAMBA3_SAVE_FOLDER", str(tmp_path / "save"))

    mixer = smoketest.build_experiment_config(_ctx()).model.block["mamba3"].sequence_mixer
    assert mixer.a_log_init_max < 16.0, "still at the library default; horizon will be ~114 steps"

    # Median head: |A| ~ Uniform(0, a_log_init_max) has median a_log_init_max / 2, and dt is
    # drawn log-uniformly on [0.001, 0.1] with geometric mean 0.01.
    alpha_median = math.exp(-0.01 * mixer.a_log_init_max / 2)
    horizon = math.log(1e-6) / math.log(alpha_median)
    assert horizon > smoketest.SEQ_LENGTH, (
        f"median-head horizon {horizon:.0f} steps does not cover SEQ_LENGTH="
        f"{smoketest.SEQ_LENGTH}"
    )
