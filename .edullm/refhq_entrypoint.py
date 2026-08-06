#!/usr/bin/env python3
"""Plain full-CE OLMo2-370M reference training on RefHQ Instruct v3.

This deliberately reuses the fixed MixLaw 370M optimization, checkpoint, task-loss,
and W&B contracts while replacing its source-mixture dataset with one ordinary
``NumpyFSLDatasetConfig`` over the single published RefHQ token partition.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.float8 import Float8Config
from olmo_core.nn.transformer import (
    TransformerConfig,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.script_utils import ExperimentConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from mixlaw_wandb_policy import MixLawWandBEvalCallback

DATASET_ID = "pretrain/refhq-instruct"
DATASET_VERSION = "v3"
TOKENIZER_ID = "tokenizer/dolma2-bpe"
TRAIN_ROWS = 3_942_810_012
GPU_RANKS = 8
SEQUENCE_LENGTH = 2_048
GLOBAL_BATCH_TOKENS = 4_194_304
RANK_MICROBATCH_TOKENS = 32_768
TOTAL_STEPS = 940
TRAIN_TOKENS = TOTAL_STEPS * GLOBAL_BATCH_TOKENS
SEED = 12_536
SAVE_INTERVAL = 125
EVAL_SCRIPT = Path(__file__).with_name("eval_task_loss_olmo_core.py")


class RefHQConfigError(RuntimeError):
    """The staged data or launch environment violates the fixed run contract."""


@dataclass(frozen=True)
class StagedCorpus:
    paths: tuple[str, ...]
    rows: int


def staged_corpus() -> StagedCorpus:
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/refhq-instruct/ready.json",
        )
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefHQConfigError(f"invalid staged input manifest: {manifest_path}") from exc

    expected = {
        "schema_version": 1,
        "family": "refhq-instruct",
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "dtype": "uint32",
        "byte_order": "little",
        "header_bytes": 0,
        "rows": TRAIN_ROWS,
    }
    changed = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if changed:
        raise RefHQConfigError(f"invalid staged RefHQ manifest fields: {changed}")

    paths: list[str] = []
    for record in payload.get("objects", []):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["size"]):
            raise RefHQConfigError(f"staged object is missing or changed: {path}")
        paths.append(str(path))
    if not paths:
        raise RefHQConfigError("staged RefHQ manifest contains no token objects")
    if TRAIN_TOKENS > TRAIN_ROWS or TRAIN_ROWS >= (TOTAL_STEPS + 1) * GLOBAL_BATCH_TOKENS:
        raise RefHQConfigError(
            f"fixed 940-step budget does not consume exactly one batch-aligned pass: "
            f"rows={TRAIN_ROWS} train_tokens={TRAIN_TOKENS}"
        )
    return StagedCorpus(paths=tuple(paths), rows=int(payload["rows"]))


def build_experiment_config(
    corpus: StagedCorpus,
    *,
    save_folder: str,
    work_dir: str = "/workspace/edullm-runs/refhq-instruct/work",
) -> ExperimentConfig:
    rank_microbatch_tokens = int(
        os.environ.get("EDULLM_RANK_MICROBATCH_TOKENS", str(RANK_MICROBATCH_TOKENS))
    )
    if (
        rank_microbatch_tokens <= 0
        or rank_microbatch_tokens % SEQUENCE_LENGTH
        or GLOBAL_BATCH_TOKENS % (rank_microbatch_tokens * GPU_RANKS)
    ):
        raise RefHQConfigError(
            "EDULLM_RANK_MICROBATCH_TOKENS must be a positive sequence-length "
            "multiple that evenly divides the per-step global batch"
        )

    tokenizer = TokenizerConfig.dolma2()
    dataset = NumpyFSLDatasetConfig(
        paths=list(corpus.paths),
        sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint32,
        work_dir=work_dir,
    )
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=rank_microbatch_tokens,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=4e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            group_overrides=[
                OptimGroupOverride(
                    params=["embeddings.weight"], opts={"weight_decay": 0.0}
                )
            ],
        ),
        scheduler=CosWithWarmup(warmup=24, alpha_f=0.1),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        float8_config=Float8Config(enabled=False),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    )
    # ``retry-start`` uses WANDB_RESUME=allow but still needs a fresh step-0
    # checkpoint for the mandatory pre-train task-loss evaluation. Only a real
    # checkpoint resume should suppress that save.
    skip_pre_train = os.environ.get("WANDB_RESUME", "").lower() == "must"
    run_name = (
        os.environ.get("WANDB_NAME")
        or os.environ.get("EDULLM_RUN_ID")
        or "refhq-instruct-ce-370m"
    )
    trainer = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            work_dir=work_dir,
            max_duration=Duration.steps(TOTAL_STEPS),
            metrics_collect_interval=5,
            cancel_check_interval=10,
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=SAVE_INTERVAL,
                ephemeral_save_interval=None,
                fixed_steps=[TOTAL_STEPS],
                pre_train_checkpoint=not skip_pre_train,
                save_async=True,
                max_checkpoints=None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                project=os.environ.get("EDULLM_WANDB_PROJECT"),
                group=os.environ.get("WANDB_RUN_GROUP"),
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "task_loss_eval",
            MixLawWandBEvalCallback(
                arm="refhq-instruct-full-ce",
                total_steps=TOTAL_STEPS,
                save_folder=save_folder,
                run_name=run_name,
                work_dir=os.environ.get(
                    "EDULLM_EVAL_WORK_DIR",
                    str(Path(work_dir) / "task-loss-eval"),
                ),
                eval_script=EVAL_SCRIPT,
                interval=SAVE_INTERVAL,
                nproc=GPU_RANKS,
            ),
        )
    )
    return ExperimentConfig(
        model=TransformerConfig.olmo2_370M(
            vocab_size=tokenizer.padded_vocab_size()
        ),
        dataset=dataset,
        data_loader=NumpyDataLoaderConfig(
            global_batch_size=GLOBAL_BATCH_TOKENS,
            seed=SEED,
            num_workers=4,
        ),
        train_module=train_module,
        trainer=trainer,
        init_seed=SEED,
    )


def refuse_aws_credentials() -> None:
    for key in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    ):
        if os.environ.get(key):
            raise RefHQConfigError(f"refusing to train while {key} is present")


def main() -> int:
    try:
        refuse_aws_credentials()
        if int(os.environ.get("WORLD_SIZE", "0")) != GPU_RANKS:
            raise RefHQConfigError(f"worker requires WORLD_SIZE={GPU_RANKS}")
        save_folder = os.environ.get("EDULLM_CHECKPOINT_DIR", "")
        project = os.environ.get("EDULLM_WANDB_PROJECT", "")
        if not save_folder:
            raise RefHQConfigError("EDULLM_CHECKPOINT_DIR is required")
        if project != "token-selection":
            raise RefHQConfigError(
                f"EDULLM_WANDB_PROJECT must be token-selection, got {project!r}"
            )

        config = build_experiment_config(
            staged_corpus(),
            save_folder=save_folder,
            work_dir=os.environ.get(
                "EDULLM_WORK_DIR",
                "/workspace/edullm-runs/refhq-instruct/work",
            ),
        )
        prepare_training_environment(seed=config.init_seed, shared_filesystem=False)
        try:
            model = config.model.build(init_device="meta")
            train_module = config.train_module.build(model)
            dataset = config.dataset.build()
            data_loader = config.data_loader.build(
                dataset, dp_process_group=train_module.dp_process_group
            )
            trainer = config.trainer.build(train_module, data_loader)
            config_saver = trainer.callbacks["config_saver"]
            assert isinstance(config_saver, ConfigSaverCallback)
            config_saver.config = config.as_config_dict()
            trainer.maybe_load_checkpoint()
            trainer.fit()
        finally:
            teardown_training_environment()
    except RefHQConfigError as exc:
        print(f"[refhq-instruct] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
