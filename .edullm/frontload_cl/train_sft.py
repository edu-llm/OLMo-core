"""
Shared post-PT SFT for both frontload-cl arms (one epoch).

Data: ``sft/frontload-cl-chat-sft`` (conversation JSONL on ``edullm-data``).
This script tokenizes with Dolma2 + the OLMo 2 chat template into local
``.npy`` token/mask shards, then runs OLMo2-370M SFT from a pretrain checkpoint.

Both PT arms share this mix and these hparams; only ``--checkpoint`` differs.

Examples::

    # Resolve conversations + tokenize plan (no fit)
    python .edullm/frontload_cl/train_sft.py "$EDULLM_RUN_ID" \\
      --dry-run --save-folder "$EDULLM_CHECKPOINT_DIR"

    # Tokenize only (writes under --tokens-dir)
    python .edullm/frontload_cl/train_sft.py "$EDULLM_RUN_ID" \\
      --tokenize-only --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --tokens-dir /tmp/frontload-cl-sft-tokens

    # Train one epoch from a pretrain arm checkpoint (8 GPUs)
    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
      .edullm/frontload_cl/train_sft.py "$EDULLM_RUN_ID" \\
      --checkpoint "$PT_CKPT" --save-folder "$EDULLM_CHECKPOINT_DIR"'

Platform note: the submission form may only offer pretrain corpora. Pass
``dataset_release: none`` and keep ``--dataset-id sft/frontload-cl-chat-sft``
(or set ``EDULLM_DATASET_ID``) so this script resolves conversations itself.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, cast

import rich
import torch

_DIR = Path(__file__).resolve().parent
_PARENT = str(_DIR.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from frontload_cl import constants as C  # noqa: E402
from frontload_cl.attn import resolve_attn_backend  # noqa: E402
from frontload_cl.corpus import (  # noqa: E402
    Refusal,
    Stage,
    default_run_name,
)
from frontload_cl.sft_tokenize import (  # noqa: E402
    find_tokenized_shards,
    resolve_conversation_paths,
    tokenize_conversations_to_dir,
)

from olmo_core.config import Config, DType  # noqa: E402
from olmo_core.data import (  # noqa: E402
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.types import LongDocStrategy  # noqa: E402
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.utils import barrier, get_rank  # noqa: E402
from olmo_core.io import clear_directory, list_directory, normalize_path  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.optim import LinearWithWarmup, SkipStepAdamWConfig  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
    LoadStrategy,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (  # noqa: E402
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer  # noqa: E402
from olmo_core.train.train_module import (  # noqa: E402
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all  # noqa: E402

log = logging.getLogger(__name__)

STEP_DIRECTORY = re.compile(r"^step(\d+)$")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyPackedFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    dataset_id: str = ""
    dataset_version: str = ""
    checkpoint: str = ""
    init_seed: int = 12536


@contextlib.contextmanager
def during(stage: Stage):
    try:
        yield
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(stage, f"{type(exc).__name__}: {exc}") from exc


def torn_step_directories(save_folder: str) -> List[str]:
    try:
        children = list(list_directory(save_folder, include_files=False))
    except FileNotFoundError:
        return []
    torn = [
        path
        for path in children
        if STEP_DIRECTORY.match(os.path.basename(normalize_path(path))) is not None
        and not Checkpointer.dir_is_checkpoint(path)
    ]
    return sorted(torn)


def remove_torn_checkpoints(save_folder: str) -> List[str]:
    removed: List[str] = []
    if get_rank() == 0:
        for path in torn_step_directories(save_folder):
            log.warning("clearing torn checkpoint %s", path)
            clear_directory(path)
            removed.append(path)
    barrier()
    return removed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frontload_cl.train_sft",
        description="OLMo2-370M shared SFT (one epoch) for frontload-cl arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_name", nargs="?", default=default_run_name())
    p.add_argument(
        "--dataset-id",
        default=os.environ.get("EDULLM_DATASET_ID", C.SFT_DATASET_ID),
    )
    p.add_argument(
        "--dataset-version",
        default=os.environ.get("EDULLM_DATASET_VERSION", "latest"),
    )
    p.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        help="Must expand $EDULLM_CHECKPOINT_DIR on the platform command line.",
    )
    p.add_argument(
        "--checkpoint",
        default=os.environ.get("EDULLM_PRETRAIN_CHECKPOINT", ""),
        help="Pretrain checkpoint URI (primer or control). Required unless "
        "--dry-run / --tokenize-only.",
    )
    p.add_argument(
        "--tokens-dir",
        default="/tmp/frontload-cl-sft-tokens",
        help="Directory for token_ids_part_*.npy + labels_mask_part_*.npy. "
        "Reused if shards already exist.",
    )
    p.add_argument("--work-dir", default="/tmp/frontload-cl-sft-cache")
    p.add_argument("--sequence-length", type=int, default=C.SFT_SEQ_LENGTH)
    p.add_argument("--learning-rate", type=float, default=C.SFT_PEAK_LR)
    p.add_argument("--global-batch-size", type=int, default=C.SFT_GLOBAL_BATCH_SIZE)
    p.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=8 * C.SFT_SEQ_LENGTH,
        help="Tokens per microbatch per rank. Default fills 8-GPU share of SFT "
        f"global batch ({8 * C.SFT_SEQ_LENGTH} = 8×{C.SFT_SEQ_LENGTH}).",
    )
    p.add_argument(
        "--attn-backend",
        default=C.DEFAULT_ATTN_BACKEND,
        choices=("flash_2", "flash_3", "flash_4", "torch", "te"),
        help="Attention backend. flash_2 needs flash-attn in the image (A100+).",
    )
    p.add_argument("--epochs", type=int, default=C.SFT_EPOCHS)
    p.add_argument("--save-interval", type=int, default=C.SFT_SAVE_INTERVAL)
    p.add_argument("--data-seed", type=int, default=C.DATA_SEED)
    p.add_argument("--model-factory", default=C.MODEL_FACTORY)
    p.add_argument(
        "--hf-tokenizer",
        default=C.SFT_HF_TOKENIZER,
        help="HF id / path used for chat-template tokenization.",
    )
    p.add_argument(
        "--tokenize-limit",
        type=int,
        default=None,
        help="Optional cap on conversations (debug).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve conversations, ensure/report tokens, print config, exit.",
    )
    p.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Tokenize conversations into --tokens-dir and exit.",
    )
    return p


def require_sft_env(opts) -> None:
    if not opts.save_folder:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "EDULLM_CHECKPOINT_DIR / --save-folder is required",
        )
    if not opts.dry_run and not opts.tokenize_only and not opts.checkpoint:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "pass --checkpoint (pretrain arm URI) or set EDULLM_PRETRAIN_CHECKPOINT",
        )


def ensure_tokenized(opts) -> Dict:
    """Resolve conversation shards and tokenize into ``opts.tokens_dir`` if needed."""
    version, paths = resolve_conversation_paths(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        split="train",
    )
    opts._dataset_version = version  # type: ignore[attr-defined]
    opts._conversation_paths = paths  # type: ignore[attr-defined]
    log.info(
        "resolved %s/%s train → %d conversation shards",
        opts.dataset_id,
        version,
        len(paths),
    )

    # Rank 0 tokenizes; others wait. Shards are local to the node work dir.
    stats: Optional[Dict] = None
    if get_rank() == 0:
        stats = tokenize_conversations_to_dir(
            paths,
            opts.tokens_dir,
            tokenizer_name=opts.hf_tokenizer,
            max_seq_length=opts.sequence_length,
            seed=opts.data_seed,
            limit=opts.tokenize_limit,
        )
    barrier()
    token_paths, mask_paths = find_tokenized_shards(opts.tokens_dir)
    if not token_paths:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"no tokenized shards under {opts.tokens_dir} after tokenize",
        )
    if stats is None:
        stats = {
            "output_dir": opts.tokens_dir,
            "num_shards": len(token_paths),
            "token_paths": token_paths,
            "mask_paths": mask_paths,
            "reused": True,
        }
    opts._token_paths = token_paths  # type: ignore[attr-defined]
    opts._mask_paths = mask_paths  # type: ignore[attr-defined]
    return stats


def build_config(opts, overrides: List[str], *, token_paths: List[str], mask_paths: List[str]) -> ExperimentConfig:
    tokenizer = TokenizerConfig.dolma2()
    factory = getattr(TransformerConfig, opts.model_factory, None)
    if factory is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {opts.model_factory}"
        )
    attn_backend = resolve_attn_backend(opts.attn_backend)
    model_config = factory(
        vocab_size=tokenizer.padded_vocab_size(),
        attn_backend=attn_backend,
    )

    dataset_config = NumpyPackedFSLDatasetConfig(
        paths=token_paths,
        label_mask_paths=mask_paths,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint32,
        sequence_length=opts.sequence_length,
        work_dir=opts.work_dir,
        generate_doc_lengths=True,
        long_doc_strategy=LongDocStrategy.truncate,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=opts.learning_rate,
            weight_decay=C.SFT_WEIGHT_DECAY,
            betas=C.ADAM_BETAS,
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=C.GRAD_CLIP,
        scheduler=LinearWithWarmup(
            warmup_fraction=C.SFT_WARMUP_FRACTION,
            alpha_f=0.0,
        ),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            load_strategy=LoadStrategy.never,  # load PT ckpt / resume explicitly in train()
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.epochs(opts.epochs),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=None,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=f"{opts.run_name}-sft",
                project=os.environ.get("EDULLM_WANDB_PROJECT")
                or os.environ.get("WANDB_PROJECT"),
                cancel_check_interval=10,
                enabled=bool(
                    os.environ.get("EDULLM_WANDB_PROJECT") or os.environ.get("WANDB_PROJECT")
                ),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=opts.dataset_id,
        dataset_version=getattr(opts, "_dataset_version", opts.dataset_version),
        checkpoint=opts.checkpoint,
    )
    return config.merge(overrides)


class LossWatcher(Callback):
    def __init__(self) -> None:
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.wandb_url = ""

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        del step
        if not self.wandb_url:
            with contextlib.suppress(Exception):
                import wandb

                self.wandb_url = getattr(wandb.run, "url", "") or ""
        loss = metrics.get("train/CE loss")
        if loss is None:
            return
        if self.first is None:
            self.first = float(loss)
        self.last = float(loss)


def train(config: ExperimentConfig, opts) -> None:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    losses = LossWatcher()
    trainer.add_callback("edullm_losses", losses)

    remove_torn_checkpoints(trainer.save_folder)
    if not trainer.maybe_load_checkpoint(trainer.save_folder):
        log.info("loading pretrain weights from %s (no trainer state)", opts.checkpoint)
        trainer.load_checkpoint(opts.checkpoint, load_trainer_state=False)

    started = time.monotonic()
    trainer.fit()
    if get_rank() == 0:
        print(
            json.dumps(
                {
                    "run_id": opts.run_name,
                    "stage": "sft",
                    "dataset_id": config.dataset_id,
                    "dataset_version": config.dataset_version,
                    "checkpoint": opts.checkpoint,
                    "steps": trainer.global_step,
                    "epoch": getattr(trainer, "epoch", None),
                    "first_loss": losses.first,
                    "last_loss": losses.last,
                    "seconds": time.monotonic() - started,
                    "checkpoint_uri": opts.save_folder,
                    "wandb_url": losses.wandb_url,
                    "tokens_dir": opts.tokens_dir,
                },
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()
    require_sft_env(opts)

    # Tokenize before prepare_training_environment so --tokenize-only / --dry-run
    # work on CPU check profiles without initializing the distributed stack.
    # For multi-GPU train we still need a barrier; if not yet distributed, get_rank()==0.
    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        # Dry-run / tokenize-only: single-process path (no barrier across ranks).
        if opts.dry_run or opts.tokenize_only:
            version, paths = resolve_conversation_paths(
                dataset_id=opts.dataset_id,
                version=opts.dataset_version,
                split="train",
            )
            opts._dataset_version = version  # type: ignore[attr-defined]
            log.info(
                "resolved %s/%s train → %d conversation shards",
                opts.dataset_id,
                version,
                len(paths),
            )
            stats = tokenize_conversations_to_dir(
                paths,
                opts.tokens_dir,
                tokenizer_name=opts.hf_tokenizer,
                max_seq_length=opts.sequence_length,
                seed=opts.data_seed,
                limit=opts.tokenize_limit,
            )
            rich.print(stats)
            if opts.tokenize_only:
                return
            config = build_config(
                opts,
                overrides,
                token_paths=stats["token_paths"],
                mask_paths=stats["mask_paths"],
            )
            rich.print(config)
            return

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        prepare_training_environment()
    try:
        with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
            stats = ensure_tokenized(opts)
            config = build_config(
                opts,
                overrides,
                token_paths=stats["token_paths"],
                mask_paths=stats["mask_paths"],
            )
            if get_rank() == 0:
                rich.print(config)
        with during(Stage.TRAINING_ITSELF_FAILED):
            train(config, opts)
    finally:
        teardown_training_environment()


def cli() -> int:
    try:
        main()
    except Refusal as refusal:
        print(refusal.explanation, file=sys.stderr)
        print(f"edullm-stage: {refusal.stage.name} exit={int(refusal.stage)}", file=sys.stderr)
        if refusal.__cause__ is not None:
            traceback.print_exception(
                type(refusal.__cause__), refusal.__cause__, refusal.__cause__.__traceback__
            )
        return int(refusal.stage)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
