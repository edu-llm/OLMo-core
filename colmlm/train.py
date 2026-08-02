"""Train the ``base`` or ``split`` SmolLM2-135M model for the memory-split experiment.

* ``base``  -- standard next-token-prediction loss over every token (the model memorizes facts).
* ``split`` -- identical, except fact-span tokens are excluded from the loss via the corpus's
  ``label_mask``. This isolates reasoning from fact memorization, with **no** query/contrastive
  objective and **no** vocab change. ``base`` and ``split`` see the exact same token stream; only
  the mask differs.

Reads the tokenized corpus produced by ``colmlm/prepare_data.py`` (token + ``label_mask`` memmaps
and a ``manifest.json``). Follows the eduLLM ``olmo-core.md`` required settings: checkpoints go to
``--save-folder`` (kept on the command line), ``max_checkpoints=None``, no LM/downstream
evaluators, and an explicit ``max_duration``.

Single GPU / CPU:
    python -m colmlm.train "$RUN" --mode split --data-dir data/tokenized \\
        --save-folder "$EDULLM_CHECKPOINT_DIR" --steps 4000
Multi-GPU:
    python -m torch.distributed.run --nproc-per-node=N --standalone -m colmlm.train "$RUN" \\
        --mode base --data-dir data/tokenized --save-folder "$EDULLM_CHECKPOINT_DIR" --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import rich

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetDType, NumpyFSLDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
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

from colmlm.tokenizer import smollm2_tokenizer_config


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    mode: str = "base"
    data_dir: str = ""
    init_seed: int = 12536


def _shard_paths(data_dir: str, manifest: dict) -> tuple[List[str], List[str]]:
    """Reconstruct token and mask shard paths from the manifest's worker list, under ``data_dir``."""
    tokens, masks = [], []
    for shard in manifest["shards"]:
        w = shard["worker"]
        tokens.append(f"{data_dir}/tokens/train-{w:05d}.bin")
        masks.append(f"{data_dir}/masks/train-{w:05d}.mask.bin")
    return tokens, masks


def build_config(opts) -> ExperimentConfig:
    if opts.mode not in ("base", "split"):
        raise SystemExit(f"--mode must be 'base' or 'split', got {opts.mode!r}")

    manifest = json.loads((Path(opts.data_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dtype") != "uint16":
        raise SystemExit(f"expected uint16 corpus, manifest says {manifest.get('dtype')!r}")
    token_paths, mask_paths = _shard_paths(opts.data_dir, manifest)

    tokenizer = smollm2_tokenizer_config()
    model_config = TransformerConfig.smollm2_135M(vocab_size=tokenizer.vocab_size)

    dataset_config = NumpyFSLDatasetConfig(
        paths=token_paths,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint16,  # explicit; never inferred (see olmo-core.md)
        work_dir=opts.work_dir,
        # The one line that distinguishes split from base: fact tokens -> -100 -> excluded from NTP.
        label_mask_paths=mask_paths if opts.mode == "split" else None,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=opts.warmup_steps),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=None,  # role can't prune; None keeps all (see olmo-core.md)
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                project=os.environ.get("WANDB_PROJECT") or os.environ.get("EDULLM_WANDB_PROJECT"),
                cancel_check_interval=10,
                enabled=bool(os.environ.get("WANDB_PROJECT") or os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )
    # No lm_evaluator / downstream_evaluator: both fail at trainer construction on this platform.

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        mode=opts.mode,
        data_dir=opts.data_dir,
    )
    return config.merge(opts.overrides)


def train(config: ExperimentConfig) -> None:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    from typing import cast

    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    trainer.maybe_load_checkpoint()
    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colmlm.train", description="Train base/split SmolLM2-135M.")
    p.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    p.add_argument("--mode", choices=["base", "split"], required=True)
    p.add_argument("--data-dir", default=os.environ.get("COLMLM_DATA_DIR", "data/tokenized"))
    p.add_argument("--save-folder", default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""))
    p.add_argument("--work-dir", default="/tmp/colmlm-dataset-cache")
    p.add_argument("--sequence-length", type=int, default=4096)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--save-interval", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--global-batch-size", type=int, default=128 * 4096)
    p.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (needs a C compiler).")
    p.add_argument("--dry-run", action="store_true", help="Build and print the config, train nothing.")
    return p


def main() -> None:
    opts, overrides = build_parser().parse_known_args()
    opts.overrides = overrides
    config = build_config(opts)
    if opts.dry_run:
        rich.print(config)
        rich.print(
            f"[bold green]mode={config.mode}[/] "
            f"params={config.model.num_params:,} "
            f"label_mask={'on' if config.dataset.label_mask_paths else 'off'} "
            f"shards={len(config.dataset.paths)}"
        )
        return
    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
