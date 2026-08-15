"""Contracts for the OLMo-ladder 370M OPT+synthetic shuffle and MTLD arms."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "olmo_ladder_opt_synthetic.py"
SPECS = {
    "shuffle": EDULLM_DIR / "run-olmo-ladder-opt-synthetic-shuffle.yaml",
    "warmup-quadratic": EDULLM_DIR / "run-olmo-ladder-opt-synthetic-warmup-quadratic.yaml",
}
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EDULLM_DIR))

if sys.platform == "win32" and not hasattr(mp.context, "ForkProcess"):
    mp.context.ForkProcess = mp.context.SpawnProcess  # type: ignore[attr-defined]
    _get_context = mp.get_context
    mp.get_context = lambda method=None: _get_context("spawn" if method == "fork" else method)

from olmo_core.data import NumpyDatasetDType  # noqa: E402
from olmo_core.hpo.curriculum import (  # noqa: E402
    ARM9_PACING_ID,
    CurriculumCorpus,
    CurriculumInputIdentity,
    token_phase_boundaries,
)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("olmo_ladder_opt_synthetic", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arm = _load_entrypoint()


def _environment() -> dict[str, str]:
    return {
        "EDULLM_DATASET_ID": "pretrain/opt-with-synthetic-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
        "EDULLM_CHECKPOINT_DIR": "/workspace/run/checkpoints",
        "EDULLM_WANDB_PROJECT": "hpo-cl",
        "EDULLM_RUN_ID": "run-123",
        "WANDB_RUN_GROUP": "hpo-cl-olmo-ladder",
    }


def _shuffle_corpus():
    return arm.ResolvedRegMix(
        paths=("/workspace/opt-synthetic/train.bin",),
        dtype=NumpyDatasetDType.uint32,
    )


def _curriculum_corpus():
    parent = CurriculumInputIdentity(
        dataset_id="pretrain/opt-with-synthetic-10b",
        version="v1",
        group="tokens",
        profile="pretrain-tokens/v1",
        manifest_sha256="a" * 64,
        source_ids=("opt-with-synthetic",),
    )
    order = CurriculumInputIdentity(
        dataset_id="curriculum/opt-with-synthetic-10b",
        version="v1",
        group="mtld",
        profile="token-order/v1",
        manifest_sha256="b" * 64,
    )
    return CurriculumCorpus(
        train_paths=("/workspace/opt-synthetic/train.bin",),
        val_paths=(),
        order_paths=("/workspace/curriculum/mtld.bin",),
        dtype=NumpyDatasetDType.uint32,
        order_dtype=NumpyDatasetDType.uint32,
        parent_identity=parent,
        order_identity=order,
    )


def _config(arm_name: str, length_steps: int = arm.TOTAL_STEPS):
    selected = arm.ARMS[arm_name]
    corpus = _curriculum_corpus() if selected.curriculum else _shuffle_corpus()
    return arm.build_experiment_config(
        selected,
        corpus,
        save_folder="/workspace/run/checkpoints",
        length_steps=length_steps,
        environ=_environment(),
    )


@pytest.mark.parametrize("arm_name", ["shuffle", "warmup-quadratic"])
def test_arms_match_ladder_control_model_batch_and_eight_a100s(arm_name) -> None:
    config = _config(arm_name)

    assert config.model.d_model == 1_024
    assert config.model.n_layers == 16
    assert config.model.block.sequence_mixer.n_heads == 16
    assert config.dataset.sequence_length == 2_048
    assert config.data_loader.global_batch_size == 262_144
    assert config.train_module.rank_microbatch_size == 16_384
    assert config.train_module.max_sequence_length == 2_048
    assert str(config.train_module.dp_config.name) == "hsdp"
    assert str(config.train_module.dp_config.param_dtype) == "bfloat16"
    assert str(config.train_module.dp_config.reduce_dtype) == "float32"
    assert config.train_module.compile_model is True
    assert config.train_module.float8_config.enabled is False
    assert config.train_module.max_grad_norm == 1.0
    if arm_name == "shuffle":
        assert config.dataset.paths == ["/workspace/opt-synthetic/train.bin"]
    else:
        assert config.dataset.paths == ["/workspace/opt-synthetic/train.bin"]


@pytest.mark.parametrize("arm_name", ["shuffle", "warmup-quadratic"])
def test_arms_use_ladder_adamw_and_cosine_schedule(arm_name) -> None:
    config = _config(arm_name)
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
    assert scheduler.get_lr(optim.lr, arm.TOTAL_STEPS, arm.TOTAL_STEPS) == pytest.approx(7.78548e-5)


def test_shuffle_arm_records_no_curriculum_identity() -> None:
    config = _config("shuffle")
    identity = arm.scientific_identity(arm.ARMS["shuffle"], config)

    assert arm.TOTAL_STEPS == 38_146
    assert arm.TRAIN_TOKENS == 9_999_745_024
    assert round(arm.TOTAL_STEPS * arm.WARMUP_FRACTION) == 191
    assert config.trainer.max_duration.value == arm.TOTAL_STEPS
    assert config.data_loader.seed == arm.SEED
    assert config.init_seed == arm.SEED
    assert identity["control"] == "no_curriculum"
    assert identity["curriculum_learning"] is False
    assert identity["data_ordering"] == "deterministic_no_replacement_shuffle"
    assert identity["dataset_id"] == "pretrain/opt-with-synthetic-10b"
    assert identity["dataset_group"] == "tokens"
    assert identity["train_tokens"] == arm.TRAIN_TOKENS
    assert identity["scheduler"]["terminal_lr"] == pytest.approx(7.78548e-5)
    assert len(identity["checkpoint_steps"]) == 21
    assert identity["checkpoint_steps"][0] == 0
    assert identity["checkpoint_steps"][-1] == arm.TOTAL_STEPS


def test_curriculum_arm_uses_token_progress_warmup_quadratic_mtld() -> None:
    config = _config("warmup-quadratic")
    identity = arm.scientific_identity(arm.ARMS["warmup-quadratic"], config)
    boundaries = token_phase_boundaries(arm.TRAIN_TOKENS)

    assert config.data_loader.pacing == ARM9_PACING_ID
    assert config.data_loader.difficulty_metric == "mtld"
    assert config.data_loader.seed == arm.SEED
    assert config.data_loader.target_tokens == arm.TRAIN_TOKENS
    assert config.data_loader.order_paths == ["/workspace/curriculum/mtld.bin"]
    assert identity["control"] == "warmup_quadratic_mtld"
    assert identity["curriculum_learning"] is True
    assert identity["dataset_id"] == "pretrain/opt-with-synthetic-10b"
    assert identity["curriculum_dataset_id"] == "curriculum/opt-with-synthetic-10b"
    assert identity["curriculum_order_group"] == "mtld"
    assert identity["pacing"] == ARM9_PACING_ID
    assert identity["curriculum_identity"]["token_phase_boundaries"] == list(boundaries)
    assert boundaries[-1] == (1000 * arm.TRAIN_TOKENS + 2383) // 2384
    assert config.trainer.max_duration.value == arm.TOTAL_STEPS
    assert len(identity["checkpoint_steps"]) == 21
    assert identity["checkpoint_steps"][-1] == arm.TOTAL_STEPS


def test_worker_manifest_refuses_remote_or_missing_inputs(tmp_path) -> None:
    corpus = _curriculum_corpus()
    manifest = arm.write_corpus_manifest(corpus, tmp_path / "corpus.json")

    with pytest.raises(arm.FinalValidationConfigError, match="not fully staged"):
        arm.load_corpus_manifest(manifest)


def test_platform_specs_and_torchrun_require_eight_a100s() -> None:
    for arm_name, path in SPECS.items():
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        command = arm.torchrun_command(arm_name)

        assert spec["suggested_compute"] == "gpu-8xa100"
        assert "EDULLM_DATASET_ID=pretrain/opt-with-synthetic-10b" in spec["command"]
        assert "EDULLM_DATASET_VERSION=v1" in spec["command"]
        assert "EDULLM_DATASET_TOKENIZER=tokenizer/dolma2-bpe" in spec["command"]
        assert "EDULLM_WANDB_PROJECT=hpo-cl" in spec["command"]
        assert "WANDB_RUN_GROUP=hpo-cl-olmo-ladder" in spec["command"]
        assert "olmo_ladder_opt_synthetic.py" in spec["command"]
        assert f"--arm {arm_name}" in spec["command"]
        assert "--nproc-per-node=8" in command
        assert arm_name in command

    assert arm.platform_values(_environment()) == (
        "/workspace/run/checkpoints",
        "run-123",
    )
    with pytest.raises(arm.FinalValidationConfigError, match="platform dataset must be"):
        arm.platform_values({**_environment(), "EDULLM_DATASET_ID": "pretrain/regmix-10b"})


@pytest.mark.parametrize("arm_name", ["shuffle", "warmup-quadratic"])
def test_smoke_duration_must_still_support_the_evaluation_ladder(arm_name) -> None:
    with pytest.raises(arm.FinalValidationConfigError, match="at least 20"):
        _config(arm_name, length_steps=19)
