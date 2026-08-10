"""Contracts for stock OLMo2-370M final validation of probe winners."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "final_validation.py"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EDULLM_DIR))

if sys.platform == "win32" and not hasattr(mp.context, "ForkProcess"):
    mp.context.ForkProcess = mp.context.SpawnProcess  # type: ignore[attr-defined]
    _get_context = mp.get_context
    mp.get_context = lambda method=None: _get_context("spawn" if method == "fork" else method)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("olmo_core_final_validation", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


final_validation = _load_entrypoint()
import final_validation_wandb as wandb_policy  # noqa: E402


def _environment() -> dict[str, str]:
    return {
        "EDULLM_DATASET_ID": "pretrain/regmix-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
        "EDULLM_CHECKPOINT_DIR": "/workspace/run/checkpoints",
        "EDULLM_WANDB_PROJECT": "hpo-final-validation",
        "EDULLM_RUN_ID": "run-123",
        "WANDB_RUN_GROUP": "hpo-final-validation-370m-10b",
    }


def _corpus():
    return final_validation.ResolvedRegMix(
        paths=("/workspace/regmix/train.bin",),
        dtype=final_validation.NumpyDatasetDType.uint32,
    )


def _config(name: str = "no-proxy-winner", length_tokens: int | None = None):
    vector = final_validation.load_vectors()[name]
    return vector, final_validation.build_experiment_config(
        vector,
        _corpus(),
        save_folder="/workspace/run/checkpoints",
        length_tokens=length_tokens,
        environ=_environment(),
    )


def test_preregistered_vectors_are_the_two_probe_winners() -> None:
    vectors = final_validation.load_vectors()
    assert set(vectors) == {"no-proxy-winner", "no-centaur-winner"}
    assert vectors["no-proxy-winner"].probe_run_id == "904ea39d368dfe412048a6063c1600df"
    assert vectors["no-proxy-winner"].probe_trial_id == "t9_0"
    assert vectors["no-proxy-winner"].hps == {
        "lr": 0.0004125460019173203,
        "weight_decay": 0.01473432082609167,
        "beta2_gap": 0.0014689794923786166,
        "eps": 1.4798352708540092e-12,
        "warmup_fraction": 0.01131976840436488,
        "decay_fraction": 0.05,
        "terminal_lr_ratio": 0.021488995515927797,
        "global_batch_mult": 0.5,
        "max_grad_norm": 0.3,
    }
    assert vectors["no-centaur-winner"].probe_run_id == "06e12699f744b8d2e562e78afa003b7f"
    assert vectors["no-centaur-winner"].probe_trial_id == "t8_0"
    assert vectors["no-centaur-winner"].hps == {
        "lr": 0.00030060095254686933,
        "weight_decay": 0.01,
        "beta2_gap": 0.001,
        "eps": 1e-12,
        "warmup_fraction": 0.007007066567546487,
        "decay_fraction": 0.05,
        "terminal_lr_ratio": 0.0,
        "global_batch_mult": 0.5,
        "max_grad_norm": 0.34844106730841967,
    }
    for vector in vectors.values():
        assert set(vector.hps) == final_validation.OPTIMIZED_FIELDS
        assert vector.global_batch_tokens == 16_384


def test_global_batch_override_rebinds_only_the_batch_hyperparameter() -> None:
    original = final_validation.load_vectors()["no-proxy-winner"]
    overridden = final_validation.with_global_batch_tokens(original, 262_144)

    assert original.global_batch_tokens == 16_384
    assert overridden.global_batch_tokens == 262_144
    assert overridden.hps == {
        **original.hps,
        "global_batch_mult": 8.0,
    }
    config = final_validation.build_experiment_config(
        overridden,
        _corpus(),
        save_folder="/workspace/run/checkpoints",
        length_tokens=262_144 * 100,
        environ=_environment(),
    )
    assert config.data_loader.global_batch_size == 262_144
    assert config.train_module.rank_microbatch_size == 32_768
    assert final_validation.scientific_identity(overridden, config)["optimized_hps"][
        "global_batch_mult"
    ] == 8.0


@pytest.mark.parametrize("name", ["no-proxy-winner", "no-centaur-winner"])
def test_stock_olmo2_370m_contract_changes_only_optimized_fields(name: str) -> None:
    vector, config = _config(name)
    hps = vector.hps
    assert config.model.d_model == 1_024
    assert config.model.n_layers == 16
    assert config.model.block.sequence_mixer.n_heads == 16
    assert config.model.block.feed_forward.hidden_size == 4_096
    assert config.model.init_method == "normal"
    assert config.model.tie_word_embeddings is False
    assert config.dataset.paths == ["/workspace/regmix/train.bin"]
    assert config.dataset.sequence_length == 2_048
    assert str(config.dataset.dtype) == "uint32"
    assert config.data_loader.global_batch_size == 16_384
    assert config.data_loader.seed == 12_536
    assert config.train_module.rank_microbatch_size == 2_048
    assert config.train_module.max_sequence_length == 2_048
    assert config.train_module.compile_model is True
    assert str(config.train_module.dp_config.name) == "hsdp"
    assert str(config.train_module.dp_config.param_dtype) == "bfloat16"
    assert str(config.train_module.dp_config.reduce_dtype) == "float32"
    assert config.train_module.z_loss_multiplier == 1e-5
    assert config.train_module.max_grad_norm == hps["max_grad_norm"]
    optim = config.train_module.optim
    assert optim.__class__.__name__ == "SkipStepAdamWConfig"
    assert optim.lr == hps["lr"]
    assert optim.betas == (0.9, 1.0 - hps["beta2_gap"])
    assert optim.eps == hps["eps"]
    assert optim.weight_decay == hps["weight_decay"]
    assert optim.group_overrides[0].params == ["embeddings.weight"]
    assert optim.group_overrides[0].opts == {"weight_decay": 0.0}
    scheduler = config.train_module.scheduler
    assert str(scheduler.units) == "tokens"
    assert scheduler.warmup_fraction == hps["warmup_fraction"]
    assert scheduler.decay_fraction == hps["decay_fraction"]
    assert scheduler.decay_min_lr == hps["terminal_lr_ratio"] * hps["lr"]
    assert config.init_seed == 12_536


def test_10b_duration_and_twenty_interval_ladder_are_exact() -> None:
    vector, config = _config()
    assert vector.total_steps == 610_351
    assert vector.train_tokens == 9_999_990_784
    assert config.trainer.max_duration.value == vector.total_steps
    ladder = final_validation.validation_steps(vector.total_steps)
    assert len(ladder) == 21
    assert ladder[0] == 0
    assert ladder[-1] == vector.total_steps
    gaps = [right - left for left, right in zip(ladder, ladder[1:])]
    assert max(gaps) - min(gaps) <= 1
    checkpointer = config.trainer.callbacks["checkpointer"]
    assert checkpointer.save_interval is None
    assert checkpointer.fixed_steps == ladder[1:]
    assert checkpointer.pre_train_checkpoint is True
    evaluator = config.trainer.callbacks["task_loss_eval"]
    assert evaluator.checkpoint_steps == tuple(ladder)
    assert evaluator.nproc == 8


def test_short_run_must_preserve_winner_batch_and_still_has_21_points() -> None:
    vector = final_validation.load_vectors()["no-proxy-winner"]
    length = vector.global_batch_tokens * 100
    _, config = _config(length_tokens=length)
    assert config.trainer.max_duration.value == 100
    assert final_validation.validation_steps(100) == [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ]
    with pytest.raises(final_validation.FinalValidationConfigError, match="positive multiple"):
        final_validation.build_experiment_config(
            vector,
            _corpus(),
            save_folder="/workspace/run/checkpoints",
            length_tokens=length + 1,
            environ=_environment(),
        )


def test_scientific_identity_names_only_probe_overrides() -> None:
    vector, config = _config()
    identity = final_validation.scientific_identity(vector, config)
    assert identity["model"] == "olmo2_370M"
    assert identity["dataset_id"] == "pretrain/regmix-10b"
    assert identity["budget_tokens_requested"] == 10_000_000_000
    assert identity["train_tokens"] == 9_999_990_784
    assert identity["optimized_hps"] == dict(vector.hps)
    assert set(identity["optimized_hps"]) == final_validation.OPTIMIZED_FIELDS


def test_torchrun_and_platform_contract() -> None:
    command = final_validation.torchrun_command("no-proxy-winner", None, 262_144)
    assert "--nproc-per-node=8" in command
    assert command[-4:] == ["--vector", "no-proxy-winner", "--global-batch-tokens", "262144"]
    assert final_validation.platform_values(_environment()) == (
        "/workspace/run/checkpoints",
        "run-123",
    )
    bad = {**_environment(), "EDULLM_DATASET_ID": "pretrain/not-regmix"}
    with pytest.raises(final_validation.FinalValidationConfigError, match="platform dataset"):
        final_validation.platform_values(bad)


def test_task_loss_result_requires_all_20_labels(tmp_path: Path) -> None:
    result = tmp_path / "eval.json"
    labels = {label: index / 10 for index, label in enumerate(wandb_policy.TASK_LABELS)}
    result.write_text(
        __import__("json").dumps({"labels": labels}),
        encoding="utf-8",
    )
    payload = wandb_policy.validate_eval(result)
    metrics = wandb_policy.eval_metrics(payload)
    assert len([key for key in metrics if key.startswith("eval/bpb/")]) == 20
    assert "eval/macro_bpb" in metrics
    labels.pop(wandb_policy.TASK_LABELS[-1])
    result.write_text(__import__("json").dumps({"labels": labels}), encoding="utf-8")
    with pytest.raises(wandb_policy.FinalValidationContractError, match="missing 1"):
        wandb_policy.validate_eval(result)
