#!/usr/bin/env python3
"""Worked-examples MetaMath CPT for one arm (additive; not a policy entrypoint yet).

Launch (example)::

    torchrun --standalone --nproc-per-node=1 \\
      src/scripts/hypothesis/worked_examples/train_cpt_arm.py \\
      we-fade-ordered \\
      --arm fade_ordered \\
      --pack-dir /orcd/pool/edullm/data/worked-examples-metamath-v0 \\
      --load-path /path/to/converted/OLMo-Ladder-760M-0.5xC \\
      --token-budget 200000000 \\
      --wandb-project pretraining \\
      --wandb-group worked-examples-faded-scaffolds

Requires tokenized shard + label_mask from ``tokenize_arms.py``.
Fade arms mask loss before ``loss_start_char`` via ``label_mask-00000.npy``.

W&B entity is always ``eduLLM``. Scientific metrics ``eval/pass_at_n`` /
``eval/pass_ratio_at_n`` are recorded by ``PassNEvalCallback`` when prediction
files are provided (see ``holdout_passn.py``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

ARMS = ("bare", "complete", "fade_ordered", "fade_shuffled")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    init_seed: int = 0
    load_path: Optional[str] = None
    load_trainer_state: bool = False


def train(config: ExperimentConfig) -> None:
    if get_rank() == 0:
        rich.print(config)

    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info("Loading CPT init checkpoint from %s", config.load_path)
        trainer.load_checkpoint(config.load_path, load_trainer_state=config.load_trainer_state)

    trainer.fit()


def build_config(opts: argparse.Namespace, overrides: List[str]) -> ExperimentConfig:
    pack = Path(opts.pack_dir)
    arm = opts.arm
    if arm not in ARMS:
        raise SystemExit(f"--arm must be one of {ARMS}")

    shard = pack / "tokenized" / arm / "shard-00000.npy"
    mask = pack / "tokenized" / arm / "label_mask-00000.npy"
    if not shard.exists():
        raise SystemExit(f"missing shard {shard}; run tokenize_arms.py")
    if not mask.exists():
        raise SystemExit(
            f"missing label mask {mask}; re-run tokenize_arms.py "
            "(writes label_mask-00000.npy including fade loss_start_char masks)"
        )

    save_folder = opts.save_folder or f"/tmp/{opts.run_name}"
    work_dir = opts.work_dir or f"{save_folder}/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()
    model_config = TransformerConfig.olmo2_760M(
        vocab_size=tokenizer_config.padded_vocab_size(),
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=[str(shard)],
        label_mask_paths=[str(mask)],
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer_config,
        work_dir=work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
        num_workers=opts.num_workers,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.lr,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=opts.compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=opts.warmup_steps),
    )

    # Import callback from this directory (script launch).
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from passn_eval_callback import PassNEvalCallback  # noqa: WPS433

    holdout = pack / "eval" / "holdout_bare.jsonl"
    trainer_config = (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(opts.token_budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                entity="eduLLM",
                project=opts.wandb_project,
                group=opts.wandb_group,
                tags=["orcd", "worked-examples-cpt", "olmo2-760m", arm],
                cancel_check_interval=10,
                enabled=opts.wandb,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "passn_eval",
            PassNEvalCallback(
                holdout_path=str(holdout) if holdout.exists() else None,
                predictions_path=opts.predictions_path,
                n_samples=opts.pass_n,
                max_items=opts.eval_max_items,
                eval_every_steps=opts.eval_every_steps,
                wandb_entity="eduLLM",
                wandb_project=opts.wandb_project,
                wandb_group=opts.wandb_group,
                enabled=True,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=opts.seed,
        load_path=opts.load_path,
        load_trainer_state=False,
    )
    return config.merge(overrides)


def parser_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_name", type=str)
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--pack-dir", type=str, required=True)
    p.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Converted OLMo-core checkpoint (e.g. Ladder 760M-0.5xC via convert_checkpoint_from_hf)",
    )
    p.add_argument("--token-budget", type=int, required=True, help="Matched token budget across arms")
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument("--global-batch-size", type=int, default=524288, help="Tokens")
    p.add_argument("--rank-microbatch-size", type=int, default=8192, help="Tokens")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-folder", type=str, default=None)
    p.add_argument("--work-dir", type=str, default=None)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-project", type=str, default="pretraining")
    p.add_argument("--wandb-group", type=str, default="worked-examples-faded-scaffolds")
    p.add_argument("--pass-n", type=int, default=8, help="N for Pass@N / PassRatio@N")
    p.add_argument("--predictions-path", type=str, default=None)
    p.add_argument("--eval-max-items", type=int, default=0)
    p.add_argument("--eval-every-steps", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    opts, overrides = p.parse_known_args()
    return opts, overrides


def main() -> None:
    opts, overrides = parser_args()
    config = build_config(opts, overrides)
    if opts.dry_run:
        if get_rank() == 0:
            rich.print(config)
        return
    train(config)


if __name__ == "__main__":
    try:
        prepare_training_environment()
        main()
    finally:
        teardown_training_environment()
