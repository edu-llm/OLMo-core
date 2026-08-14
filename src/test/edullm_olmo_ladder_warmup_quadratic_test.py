"""Contracts for the OLMo-ladder 370M warmup-quadratic MTLD curriculum arm."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "olmo_ladder_warmup_quadratic.py"
RUN_SPEC = EDULLM_DIR / "run-olmo-ladder-warmup-quadratic.yaml"
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
    spec = importlib.util.spec_from_file_location("olmo_ladder_warmup_quadratic", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arm = _load_entrypoint()


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
    parent = CurriculumInputIdentity(
        dataset_id="pretrain/regmix-10b",
        version="v1",
        group="tokens",
        profile="pretrain-tokens/v1",
        manifest_sha256="a" * 64,
        source_ids=("regmix",),
    )
    order = CurriculumInputIdentity(
        dataset_id="curriculum/regmix-370m",
        version="v1",
        group="mtld",
        profile="token-order/v1",
        manifest_sha256="b" * 64,
    )
    return CurriculumCorpus(
        train_paths=("/workspace/regmix/train.bin",),
        val_paths=(),
        order_paths=("/workspace/curriculum/mtld.bin",),
        dtype=NumpyDatasetDType.uint32,
        order_dtype=NumpyDatasetDType.uint32,
        parent_identity=parent,
        order_identity=order,
    )


def _config(length_steps: int = arm.TOTAL_STEPS):
    return arm.build_experiment_config(
        _corpus(),
        save_folder="/workspace/run/checkpoints",
        length_steps=length_steps,
        environ=_environment(),
    )


def test_inputs_are_staged_once_and_workers_require_local_files(tmp_path) -> None:
    objects = {
        ("sealed-data", "regmix/train.bin"): bytes(range(64)),
        ("sealed-data", "orders/mtld.bin"): bytes(range(32)),
    }

    class FakeS3:
        def __init__(self) -> None:
            self.downloads: list[tuple[str, str]] = []

        def head_object(self, *, Bucket, Key):
            return {"ContentLength": len(objects[(Bucket, Key)])}

        def download_file(self, bucket, key, filename, *, Config):
            del Config
            self.downloads.append((bucket, key))
            Path(filename).write_bytes(objects[(bucket, key)])

    original = _corpus()
    remote = CurriculumCorpus(
        train_paths=("s3://sealed-data/regmix/train.bin",),
        val_paths=(),
        order_paths=("s3://sealed-data/orders/mtld.bin",),
        dtype=original.dtype,
        order_dtype=original.order_dtype,
        parent_identity=original.parent_identity,
        order_identity=original.order_identity,
    )
    client = FakeS3()
    staged = arm.stage_curriculum(
        remote,
        cache_dir=tmp_path / "cache",
        s3_client=client,
        transfer_config=object(),
    )

    assert set(client.downloads) == {
        ("sealed-data", "regmix/train.bin"),
        ("sealed-data", "orders/mtld.bin"),
    }
    assert all("://" not in path for path in (*staged.train_paths, *staged.order_paths))
    assert Path(staged.train_paths[0]).read_bytes() == objects[("sealed-data", "regmix/train.bin")]
    assert Path(staged.order_paths[0]).read_bytes() == objects[("sealed-data", "orders/mtld.bin")]

    # A retry reuses complete immutable objects instead of downloading them again.
    restaged = arm.stage_curriculum(
        remote,
        cache_dir=tmp_path / "cache",
        s3_client=client,
        transfer_config=object(),
    )
    assert restaged == staged
    assert len(client.downloads) == 2

    manifest = arm.write_corpus_manifest(staged, tmp_path / "cache" / "corpus.json")
    assert arm.load_corpus_manifest(manifest) == staged


def test_worker_manifest_refuses_remote_or_missing_inputs(tmp_path) -> None:
    corpus = _corpus()
    manifest = arm.write_corpus_manifest(corpus, tmp_path / "corpus.json")

    with pytest.raises(arm.FinalValidationConfigError, match="not fully staged"):
        arm.load_corpus_manifest(manifest)


def test_arm_matches_ladder_control_model_batch_and_eight_a100s() -> None:
    config = _config()

    assert config.model.d_model == 1_024
    assert config.model.n_layers == 16
    assert config.model.block.sequence_mixer.n_heads == 16
    assert config.dataset.paths == ["/workspace/regmix/train.bin"]
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


def test_arm_uses_ladder_adamw_and_cosine_schedule() -> None:
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
    assert scheduler.get_lr(optim.lr, arm.TOTAL_STEPS, arm.TOTAL_STEPS) == pytest.approx(7.78548e-5)


def test_arm_uses_token_progress_warmup_quadratic_mtld() -> None:
    config = _config()
    identity = arm.scientific_identity(config)
    boundaries = token_phase_boundaries(arm.TRAIN_TOKENS)

    assert config.data_loader.pacing == ARM9_PACING_ID
    assert config.data_loader.difficulty_metric == "mtld"
    assert config.data_loader.seed == arm.SEED
    assert config.data_loader.target_tokens == arm.TRAIN_TOKENS
    assert config.data_loader.order_paths == ["/workspace/curriculum/mtld.bin"]
    assert identity["control"] == "warmup_quadratic_mtld"
    assert identity["curriculum_learning"] is True
    assert identity["dataset_id"] == "pretrain/regmix-10b"
    assert identity["curriculum_dataset_id"] == "curriculum/regmix-370m"
    assert identity["pacing"] == ARM9_PACING_ID
    assert identity["curriculum_identity"]["token_phase_boundaries"] == list(boundaries)
    assert boundaries[-1] == (1000 * arm.TRAIN_TOKENS + 2383) // 2384
    assert arm.TOTAL_STEPS == 38_146
    assert arm.TRAIN_TOKENS == 9_999_745_024
    assert round(arm.TOTAL_STEPS * arm.WARMUP_FRACTION) == 191
    assert config.trainer.max_duration.value == arm.TOTAL_STEPS
    assert len(identity["checkpoint_steps"]) == 21
    assert identity["checkpoint_steps"][0] == 0
    assert identity["checkpoint_steps"][-1] == arm.TOTAL_STEPS


def test_platform_spec_and_torchrun_require_eight_a100s() -> None:
    spec = yaml.safe_load(RUN_SPEC.read_text(encoding="utf-8"))
    command = arm.torchrun_command()

    assert spec["suggested_compute"] == "gpu-8xa100"
    assert "EDULLM_WANDB_PROJECT=hpo-ladder" in spec["command"]
    assert "olmo_ladder_warmup_quadratic.py" in spec["command"]
    assert "--nproc-per-node=8" in command
    assert arm.platform_values(_environment()) == (
        "/workspace/run/checkpoints",
        "run-123",
    )


def test_smoke_duration_must_still_support_the_evaluation_ladder() -> None:
    with pytest.raises(arm.FinalValidationConfigError, match="at least 20"):
        _config(length_steps=19)
