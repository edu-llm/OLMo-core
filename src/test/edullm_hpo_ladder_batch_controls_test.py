"""Contracts for the 370M RegMix batch-ablation arms logged on a 256 Ki W&B axis."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "hpo_ladder_batch_controls.py"
SPECS = {
    "ladder-768ki": EDULLM_DIR / "run-olmo-ladder-768ki.yaml",
    "ladder-4mi": EDULLM_DIR / "run-olmo-ladder-4mi.yaml",
    "mixlaw-regmix-4mi": EDULLM_DIR / "run-mixlaw-regmix-4mi.yaml",
}
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EDULLM_DIR))

if sys.platform == "win32" and not hasattr(mp.context, "ForkProcess"):
    mp.context.ForkProcess = mp.context.SpawnProcess  # type: ignore[attr-defined]
    _get_context = mp.get_context
    mp.get_context = lambda method=None: _get_context("spawn" if method == "fork" else method)

import olmo_core.train.callbacks.wandb as wandb_mod  # noqa: E402
from olmo_core.data import NumpyDatasetDType  # noqa: E402


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("hpo_ladder_batch_controls", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controls = _load_entrypoint()


def _environment() -> dict[str, str]:
    return {
        "EDULLM_DATASET_ID": "pretrain/regmix-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
        "EDULLM_CHECKPOINT_DIR": "/workspace/run/checkpoints",
        "EDULLM_WANDB_PROJECT": "hpo-ladder",
        "EDULLM_RUN_ID": "run-123",
    }


def _corpus():
    return controls.ResolvedRegMix(
        paths=("/workspace/regmix/train.bin",),
        dtype=NumpyDatasetDType.uint32,
    )


def _config(arm_name: str, length_steps: int | None = None):
    arm = controls.ARMS[arm_name]
    return controls.build_experiment_config(
        arm,
        _corpus(),
        save_folder="/workspace/run/checkpoints",
        length_steps=length_steps,
        environ=_environment(),
    )


@pytest.mark.parametrize(
    "arm_name, batch, multiplier, native_steps",
    [
        ("ladder-768ki", 768 * 1_024, 3, 12_715),
        ("ladder-4mi", 4 * 1_024 * 1_024, 16, 2_384),
        ("mixlaw-regmix-4mi", 4 * 1_024 * 1_024, 16, 2_384),
    ],
)
def test_arms_use_token_aligned_wandb_axis(arm_name, batch, multiplier, native_steps) -> None:
    arm = controls.ARMS[arm_name]
    config = _config(arm_name)
    identity = controls.scientific_identity(arm, config)
    wandb = config.trainer.callbacks["wandb"]
    eval_callback = config.trainer.callbacks["task_loss_eval"]

    assert config.data_loader.global_batch_size == batch
    assert config.train_module.rank_microbatch_size == 16_384
    assert arm.total_steps == native_steps
    assert arm.wandb_step_multiplier == multiplier
    assert identity["wandb_axis_tokens"] == 262_144
    assert identity["wandb_logged_steps"] == native_steps * multiplier
    assert 38_000 <= identity["wandb_logged_steps"] <= 38_200
    assert wandb.step_multiplier == multiplier
    assert eval_callback.wandb_step_multiplier == multiplier
    assert identity["dataset_id"] == "pretrain/regmix-10b"


def test_ladder_768ki_keeps_olmo_ladder_optimizer() -> None:
    config = _config("ladder-768ki")
    optim = config.train_module.optim
    scheduler = config.train_module.scheduler

    assert optim.__class__.__name__ == "AdamWConfig"
    assert optim.lr == pytest.approx(7.78548e-4)
    assert optim.betas == (0.9, 0.95)
    assert scheduler.warmup_fraction == 0.005
    assert scheduler.alpha_f == 0.1
    assert config.data_loader.seed == 12_536
    assert config.init_seed == 12_536


def test_ladder_4mi_matches_768ki_except_batch() -> None:
    small = _config("ladder-768ki")
    large = _config("ladder-4mi")
    small_sched = small.train_module.scheduler
    large_sched = large.train_module.scheduler

    assert large.train_module.optim.__class__.__name__ == "AdamWConfig"
    assert large.train_module.optim.lr == small.train_module.optim.lr
    assert large_sched.warmup_fraction == small_sched.warmup_fraction
    assert large.data_loader.global_batch_size == 4_194_304
    assert large.data_loader.seed == small.data_loader.seed


def test_mixlaw_regmix_uses_library_defaults_on_sealed_corpus() -> None:
    config = _config("mixlaw-regmix-4mi")
    identity = controls.scientific_identity(controls.ARMS["mixlaw-regmix-4mi"], config)
    optim = config.train_module.optim
    scheduler = config.train_module.scheduler

    assert config.dataset.paths == ["/workspace/regmix/train.bin"]
    assert optim.__class__.__name__ == "SkipStepAdamWConfig"
    assert optim.lr == pytest.approx(4e-4)
    assert optim.betas == (0.9, 0.95)
    assert scheduler.warmup == 24
    assert scheduler.warmup_fraction is None
    assert scheduler.alpha_f == 0.1
    assert config.data_loader.seed == 6_199
    assert identity["optimizer"]["name"] == "SkipStepAdamW"


def test_scaled_wandb_callback_multiplies_logged_steps() -> None:
    logged: list[tuple[int, dict[str, float]]] = []

    class _FakeWandB:
        def log(self, metrics, step):
            logged.append((step, metrics))

    callback = controls.ScaledWandBCallback(step_multiplier=16, enabled=True)
    callback._wandb = _FakeWandB()
    original = wandb_mod.get_rank
    wandb_mod.get_rank = lambda: 0
    try:
        callback.log_metrics(125, {"checkpoint/step": 125.0, "train/loss": 2.5})
    finally:
        wandb_mod.get_rank = original

    assert logged == [(2000, {"checkpoint/step": 2000.0, "train/loss": 2.5})]


def test_platform_specs_target_hpo_ladder_and_eight_a100s() -> None:
    for arm_name, path in SPECS.items():
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = controls.torchrun_command(arm_name)
        assert spec["suggested_compute"] == "gpu-8xa100"
        assert "EDULLM_WANDB_PROJECT=hpo-ladder" in spec["command"]
        assert f"--arm {arm_name}" in spec["command"]
        assert "--nproc-per-node=8" in command
        assert arm_name in command
