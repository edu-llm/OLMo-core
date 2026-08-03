"""The single OLMo2-370M recipe; arm selection changes loaders and loss policy only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from production_contract.checkpoint import (
    checkpointer_kwargs_for_ladder,
    make_run_fingerprint,
    write_run_fingerprint,
)

from .arms import ArmSpec

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_TOKENS = 4_194_304
RANK_MICROBATCH_TOKENS = 32_768
PEAK_LR = 4e-4
WARMUP_STEPS = 24
ALPHA_F = 0.1
Z_LOSS = 1e-5
MAX_GRAD_NORM = 1.0
INIT_SEED = 6198
DATA_SEED = 42
PRODUCTION_WORLD_SIZE = 8
CUSTOM_LOSS_METHODS = frozenset(
    {"random", "rho_excess", "rel_ema", "middle_ppl", "learnability", "attention_topk"}
)


def total_steps(max_tokens: int) -> int:
    return int(max_tokens) // GLOBAL_BATCH_TOKENS


def _reference_digest(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"materialized reference does not exist: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def immutable_corpus_binding(dataset_id: str, corpus: Any) -> dict[str, Any]:
    """Compactly bind a resolved, seal-verified corpus into checkpoint identity."""
    paths = [str(path) for path in corpus.paths]
    version = str(corpus.version)
    if not version or version == "latest":
        raise ValueError(f"{dataset_id} did not resolve to an immutable version")
    dtype = getattr(corpus.dtype, "value", corpus.dtype)
    return {
        "dataset_id": dataset_id,
        "version": version,
        "path_count": len(paths),
        "paths_sha256": hashlib.sha256(
            json.dumps(paths, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "dtype": str(dtype),
        "rows": None if corpus.rows is None else int(corpus.rows),
    }


def scientific_identity(
    arm: ArmSpec,
    *,
    dataset_binding: Mapping[str, Any],
    refhq_binding: Optional[Mapping[str, Any]],
    max_tokens: int,
    reference_path: Optional[str],
    early_reference_path: Optional[str],
    late_reference_path: Optional[str],
    passive_reference_path: Optional[str] = None,
    precomputed_selection_binding: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    main_binding = dict(dataset_binding)
    if main_binding.get("dataset_id") != arm.dataset_id:
        raise ValueError("resolved main corpus does not match the selected arm")
    if arm.requires_refhq_stream != (refhq_binding is not None):
        raise ValueError("BLADE RefHQ binding is missing or attached to a non-BLADE arm")
    if precomputed_selection_binding is not None and arm.method != "middle_ppl":
        raise ValueError("precomputed token masks are only supported for middle-PPL")
    return {
        "arm": arm.name,
        "run_id": arm.run_id,
        "method": arm.method,
        "dataset_id": arm.dataset_id,
        "dataset_version": main_binding["version"],
        "dataset_binding": main_binding,
        "refhq_binding": dict(refhq_binding) if refhq_binding is not None else None,
        "tokenizer": "tokenizer/dolma2-bpe",
        "model": "TransformerConfig.olmo2_370M",
        "init_seed": INIT_SEED,
        "data_seed": DATA_SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "global_batch_tokens": GLOBAL_BATCH_TOKENS,
        "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
        "max_tokens": int(max_tokens),
        "total_steps": total_steps(max_tokens),
        "peak_lr": PEAK_LR,
        "warmup_steps": WARMUP_STEPS,
        "alpha_f": ALPHA_F,
        "z_loss_multiplier": Z_LOSS,
        "keep_fraction": arm.keep_fraction,
        "ema_seed": arm.ema_seed,
        "ema_alpha": arm.ema_alpha,
        "ema_tau": arm.ema_tau,
        "reference_contract": arm.reference_contract,
        "early_reference_contract": arm.early_reference_contract,
        "late_reference_contract": arm.late_reference_contract,
        "reference_sha256": _reference_digest(reference_path),
        "early_reference_sha256": _reference_digest(early_reference_path),
        "late_reference_sha256": _reference_digest(late_reference_path),
        "passive_reference_sha256": _reference_digest(passive_reference_path),
        "precomputed_selection_binding": (
            dict(precomputed_selection_binding)
            if precomputed_selection_binding is not None
            else None
        ),
        "wandb_project": arm.wandb_project,
        "checkpoint_contract": "schema-v2-ladder125-task-loss20-wandb",
    }


def _loader(
    corpus,
    *,
    work_dir: Path,
    seed: int,
    process_group=None,
    label_mask_paths: Optional[Sequence[str]] = None,
):
    from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig

    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        label_mask_paths=list(label_mask_paths) if label_mask_paths is not None else None,
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=corpus.tokenizer,
        dtype=corpus.dtype,
        work_dir=str(work_dir),
    ).build()
    return NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_TOKENS,
        seed=seed,
        num_workers=int(os.environ.get("EDULLM_NUM_WORKERS", "4")),
    ).build(dataset, dp_process_group=process_group)


def _train_module_config():
    from olmo_core.config import DType
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.distributed.utils import is_distributed
    from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
    from olmo_core.train.train_module import (
        TransformerDataParallelConfig,
        TransformerTrainModuleConfig,
    )

    dp_config = (
        TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        )
        if is_distributed()
        else None
    )
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_TOKENS,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=PEAK_LR,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS, alpha_f=ALPHA_F),
        compile_model=True,
        dp_config=dp_config,
        z_loss_multiplier=Z_LOSS,
        max_grad_norm=MAX_GRAD_NORM,
        state_dict_save_opts={
            "full_state_dict": False,
            "cpu_offload": True,
            "flatten_optimizer_state_dict": True,
        },
        state_dict_load_opts={
            "full_state_dict": False,
            "strict": True,
            "flatten_optimizer_state_dict": True,
        },
    )


def _custom_module(model, config, selection):
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    from .train_module import TokenWeightedTrainModule

    return TokenWeightedTrainModule(
        model=model,
        optim=config.optim,
        rank_microbatch_size=config.rank_microbatch_size,
        max_sequence_length=config.max_sequence_length,
        compile_model=config.compile_model,
        dp_config=config.dp_config,
        z_loss_multiplier=config.z_loss_multiplier,
        max_grad_norm=config.max_grad_norm,
        scheduler=config.scheduler,
        label_ignore_index=config.label_ignore_index,
        state_dict_save_opts=StateDictOptions(**(config.state_dict_save_opts or {})),
        state_dict_load_opts=StateDictOptions(**(config.state_dict_load_opts or {})),
        selection_config=selection,
    )


def build_trainer(
    arm: ArmSpec,
    corpus,
    *,
    refhq_corpus=None,
    max_tokens: int,
    save_folder: Path,
    work_dir: Path,
    progress_dir: Path,
    task_loss_script: Path,
    reference_path: Optional[str] = None,
    early_reference_path: Optional[str] = None,
    late_reference_path: Optional[str] = None,
    passive_reference_path: Optional[str] = None,
    precomputed_label_mask_paths: Optional[Sequence[str]] = None,
    precomputed_selection_binding: Optional[Mapping[str, Any]] = None,
    resume: bool = False,
    production: bool = True,
):
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import Duration, LoadStrategy, TrainerConfig
    from olmo_core.train.callbacks import (
        CheckpointerCallback,
        ConfigSaverCallback,
        GPUMemoryMonitorCallback,
        WandBCallback,
    )

    from production_contract.task_loss import TaskLossEvalCallback

    from .blade import BladeCallback, ResumableBatchStream
    from .train_module import (
        TokenSelectionConfig,
        TokenSelectionStateCallback,
        TokenWeightedTrainModule,
    )

    steps = total_steps(max_tokens)
    model_config = TransformerConfig.olmo2_370M(
        vocab_size=corpus.tokenizer.padded_vocab_size(),
        init_seed=INIT_SEED,
    )
    model = model_config.build(init_device="meta")
    module_config = _train_module_config()
    selection_config = TokenSelectionConfig(
        method="full" if arm.method == "blade" else arm.method,
        keep_fraction=arm.keep_fraction,
        total_steps=steps,
        seed=DATA_SEED,
        reference_path=late_reference_path if arm.method == "middle_ppl" else reference_path,
        early_reference_path=early_reference_path,
        late_reference_path=None if arm.method == "middle_ppl" else late_reference_path,
        passive_reference_path=passive_reference_path,
        ema_seed=arm.ema_seed,
        ema_alpha=arm.ema_alpha,
        ema_tau=arm.ema_tau,
    )
    precomputed_middle_ppl = precomputed_label_mask_paths is not None
    if precomputed_middle_ppl and arm.method != "middle_ppl":
        raise ValueError("precomputed label masks are only supported for middle-PPL")
    if precomputed_middle_ppl != (precomputed_selection_binding is not None):
        raise ValueError("precomputed label masks and their binding must be supplied together")
    if precomputed_middle_ppl and passive_reference_path:
        raise ValueError("passive online scoring is incompatible with precomputed middle-PPL")
    if (arm.method in CUSTOM_LOSS_METHODS and not precomputed_middle_ppl) or passive_reference_path:
        train_module = _custom_module(model, module_config, selection_config)
    else:
        train_module = module_config.build(model)

    main_loader = _loader(
        corpus,
        work_dir=work_dir / "main",
        seed=DATA_SEED,
        process_group=train_module.dp_process_group,
        label_mask_paths=precomputed_label_mask_paths,
    )
    checkpoint_kwargs = checkpointer_kwargs_for_ladder(steps, 125, save_async=False)
    checkpoint_kwargs["pre_train_checkpoint"] = not resume
    # TaskLossEvalCallback finalizes in post_step, before CheckpointerCallback's
    # post_train fallback. Save the true final step in post_train_batch so the
    # synchronous evaluator never waits for a checkpoint that cannot exist yet.
    checkpoint_kwargs["fixed_steps"] = [
        *checkpoint_kwargs["fixed_steps"],
        steps,
    ]
    trainer_config = (
        TrainerConfig(
            save_folder=str(save_folder),
            load_strategy=LoadStrategy.if_available if resume else LoadStrategy.never,
            load_trainer_state=resume,
            load_optim_state=resume,
            max_duration=Duration.tokens(max_tokens),
        )
        .with_callback("checkpointer", CheckpointerCallback(**checkpoint_kwargs))
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=arm.run_id,
                project=arm.wandb_project,
                group=arm.name,
                tags=["token-selection", arm.name, arm.method],
                enabled=production or bool(os.environ.get("WANDB_API_KEY")),
                config={
                    "arm": arm.name,
                    "method": arm.method,
                    "dataset_id": arm.dataset_id,
                    "max_tokens": max_tokens,
                    "precomputed_selection": precomputed_middle_ppl,
                    "precomputed_selection_binding": (
                        dict(precomputed_selection_binding)
                        if precomputed_selection_binding is not None
                        else None
                    ),
                },
            ),
        )
    )
    if production or task_loss_script.is_file():
        trainer_config = trainer_config.with_callback(
            "task_loss",
            TaskLossEvalCallback(
                total_steps=steps,
                save_folder=save_folder,
                run_name=arm.run_id,
                results_dir=progress_dir / "task_loss",
                eval_script=task_loss_script,
                arm=arm.name,
                progress_dir=progress_dir,
                method=arm.method,
                task_loss_nproc=PRODUCTION_WORLD_SIZE if production else None,
                production=production,
                wandb_mode=os.environ.get("WANDB_MODE", "online"),
            ),
        )
    if isinstance(train_module, TokenWeightedTrainModule):
        trainer_config = trainer_config.with_callback(
            "token_selection_state", TokenSelectionStateCallback()
        )
    if arm.method == "blade":
        if refhq_corpus is None:
            raise ValueError("BLADE requires the immutable RefHQ corpus as its second stream")
        reference_train_loader = _loader(
            corpus,
            work_dir=work_dir / "blade-reference-train",
            seed=DATA_SEED + 17,
            process_group=train_module.dp_process_group,
        )
        refhq_loader = _loader(
            refhq_corpus,
            work_dir=work_dir / "blade-refhq",
            seed=DATA_SEED + 101,
            process_group=train_module.dp_process_group,
        )

        def reference_factory():
            reference = model_config.build(init_device=str(train_module.device))
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            return reference

        trainer_config = trainer_config.with_callback(
            "blade_state",
            BladeCallback(
                total_steps=steps,
                reference_factory=reference_factory,
                reference_train_stream=ResumableBatchStream(reference_train_loader),
                refhq_stream=ResumableBatchStream(refhq_loader),
            ),
        )

    trainer = trainer_config.build(train_module, main_loader)
    trainer.callbacks["config_saver"].config = {
        "recipe": "unchanged-olmo2-370m-v1",
        "arm": arm.name,
        "method": arm.method,
        "precomputed_selection": precomputed_middle_ppl,
        "precomputed_selection_binding": (
            dict(precomputed_selection_binding)
            if precomputed_selection_binding is not None
            else None
        ),
        "scientific_constants": {
            "sequence_length": SEQUENCE_LENGTH,
            "global_batch_tokens": GLOBAL_BATCH_TOKENS,
            "rank_microbatch_tokens": RANK_MICROBATCH_TOKENS,
            "peak_lr": PEAK_LR,
            "warmup_steps": WARMUP_STEPS,
            "alpha_f": ALPHA_F,
            "z_loss_multiplier": Z_LOSS,
            "max_grad_norm": MAX_GRAD_NORM,
        },
    }
    return trainer


def write_identity(save_folder: Path, progress_dir: Path, identity: dict[str, Any]) -> None:
    fingerprint = write_run_fingerprint(save_folder, identity)
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "run_identity.json").write_text(
        json.dumps(make_run_fingerprint(identity), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if fingerprint.name != "run_fingerprint.json":
        raise RuntimeError("unexpected fingerprint path")
