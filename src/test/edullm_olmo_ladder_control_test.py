"""Contracts for the fixed no-curriculum OLMo-ladder 370M control arm."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "olmo_ladder_control.py"
RUN_SPEC = EDULLM_DIR / "run-olmo-ladder-control.yaml"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EDULLM_DIR))

if sys.platform == "win32" and not hasattr(mp.context, "ForkProcess"):
    mp.context.ForkProcess = mp.context.SpawnProcess  # type: ignore[attr-defined]
    _get_context = mp.get_context
    mp.get_context = lambda method=None: _get_context("spawn" if method == "fork" else method)

from olmo_core.data import NumpyDatasetDType  # noqa: E402


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("olmo_ladder_control", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


control = _load_entrypoint()


def _environment() -> dict[str, str]:
    return {
        "EDULLM_DATASET_ID": "pretrain/regmix-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
        "EDULLM_CHECKPOINT_DIR": "/workspace/run/checkpoints",
        "EDULLM_WANDB_PROJECT": "hpo-validation",
        "EDULLM_RUN_ID": "run-123",
    }


def _corpus():
    return control.ResolvedRegMix(
        paths=("/workspace/regmix/train.bin",),
        dtype=NumpyDatasetDType.uint32,
    )


def _config(length_steps: int = control.TOTAL_STEPS):
    return control.build_experiment_config(
        _corpus(),
        save_folder="/workspace/run/checkpoints",
        length_steps=length_steps,
        environ=_environment(),
    )


def test_control_uses_requested_model_data_and_eight_a100_batch() -> None:
    config = _config()

    assert config.model.d_model == 1_024
    assert config.model.n_layers == 16
    assert config.model.block.sequence_mixer.n_heads == 16
    assert config.dataset.paths == ["/workspace/regmix/train.bin"]
    assert config.dataset.sequence_length == 2_048
    assert config.data_loader.global_batch_size == 262_144
    assert config.train_module.rank_microbatch_size == 32_768
    assert config.train_module.max_sequence_length == 2_048
    assert str(config.train_module.dp_config.name) == "hsdp"
    assert str(config.train_module.dp_config.param_dtype) == "bfloat16"
    assert str(config.train_module.dp_config.reduce_dtype) == "float32"
    assert config.train_module.compile_model is True
    assert config.train_module.float8_config.enabled is False
    assert config.train_module.max_grad_norm == 1.0


def test_control_uses_requested_adamw_and_cosine_schedule() -> None:
    config = _config()
    optim = config.train_module.optim
    scheduler = config.train_module.scheduler

    assert optim.__class__.__name__ == "AdamWConfig"
    assert optim.lr == pytest.approx(7.78548e-4)
    assert optim.betas == (0.9, 0.95)
    assert optim.eps == 1e-8
    assert optim.weight_decay == 0.1
    assert optim.fused is True
    assert optim.group_overrides[0].params == ["embeddings.weight"]
    assert optim.group_overrides[0].opts == {"weight_decay": 0.0}
    assert scheduler.__class__.__name__ == "CosWithWarmup"
    assert scheduler.warmup_fraction == 0.005
    assert scheduler.alpha_f == 0.1
    assert scheduler.get_lr(optim.lr, control.TOTAL_STEPS, control.TOTAL_STEPS) == pytest.approx(
        7.78548e-5
    )


def test_control_trains_largest_whole_batch_below_10b_and_records_identity() -> None:
    config = _config()
    identity = control.scientific_identity(config)

    assert control.TOTAL_STEPS == 38_146
    assert control.TRAIN_TOKENS == 9_999_745_024
    assert round(control.TOTAL_STEPS * control.WARMUP_FRACTION) == 191
    assert config.trainer.max_duration.value == control.TOTAL_STEPS
    assert identity["control"] == "no_curriculum"
    assert identity["dataset_id"] == "pretrain/regmix-10b"
    assert identity["train_tokens"] == control.TRAIN_TOKENS
    assert identity["scheduler"]["terminal_lr"] == pytest.approx(7.78548e-5)
    assert len(identity["checkpoint_steps"]) == 21
    assert identity["checkpoint_steps"][0] == 0
    assert identity["checkpoint_steps"][-1] == control.TOTAL_STEPS


def test_platform_spec_and_torchrun_require_eight_a100s() -> None:
    spec = yaml.safe_load(RUN_SPEC.read_text(encoding="utf-8"))
    command = control.torchrun_command()

    assert spec["suggested_compute"] == "gpu-8xa100"
    assert "EDULLM_WANDB_PROJECT=hpo-validation" in spec["command"]
    assert "olmo_ladder_control.py" in spec["command"]
    assert "--nproc-per-node=8" in command
    assert control.platform_values(_environment()) == (
        "/workspace/run/checkpoints",
        "run-123",
    )


def test_smoke_duration_must_still_support_the_evaluation_ladder() -> None:
    with pytest.raises(control.FinalValidationConfigError, match="at least 20"):
        _config(length_steps=19)
