"""
Controlled comparison: **plain linear attention vs Gated DeltaNet (GDN)** at the
370M OLMo-3 scale, on the SAME data mix, SAME recipe, and the SAME ``fla``
chunked-scan Triton kernel family.

This reuses the proven 370M dolma2 recipe (LR / global-batch / schedule from the
OLMo ladder formulas) and only changes the sequence mixer of every block:

  * ``--mixer attention`` -> the stock ``olmo3_370M`` sliding-window softmax attention (baseline);
  * ``--mixer gdn``       -> :class:`olmo_core.nn.attention.recurrent.GatedDeltaNet`;
  * ``--mixer linear``    -> :class:`olmo_linear_attn.LinearAttention` (a gate/delta ablation of
                             GatedDeltaNet: identical projections/conv/L2-norm/head layout, only the
                             gated delta recurrence removed).

The GDN and linear mixers use IDENTICAL head geometry (``n_heads``, ``head_dim``,
``expand_v``, ``n_v_heads``, ``conv_size``) so the ONLY architectural difference is
the gated delta mechanism itself (GDN additionally carries the gate projections
``w_a`` / ``w_b`` / ``w_g`` and ``A_log`` / ``dt_bias`` -- reported below so the
parameter delta is transparent).

Run under torchrun (one process per GPU). Pin to a single B200, e.g.::

    CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc-per-node=1 \
        experiments/linear-attn-vs-gdn/train_mixer.py linear-attn-370m-10b --mixer linear \
        --save-folder s3://edullm-olmo-370m-ckpts/linear-attn-vs-gdn/linear \
        --work-dir /mnt/nvme/olmo-work

Validate on CPU first (no GPU, no training)::

    python experiments/linear-attn-vs-gdn/train_mixer.py test --mixer linear --dry-run
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

# Make sibling `olmo_linear_attn` importable both when run as a script and when
# workers/checkpoint-resume re-import by the serialized class path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Additive I/O-robustness shim: makes the one-time S3 shard-size fingerprint burst
# reliable (throttles HeadObject fan-out + jittered retry on transient 403s). Does
# NOT modify OLMo-core source or any training numerics. Must import before the
# dataset is built. See s3_io_robustness.py for details.
import s3_io_robustness  # noqa: E402,F401

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.source_mixture import SourceMixtureDatasetConfig, SourceMixtureList
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from olmo_linear_attn import LinearAttentionConfig  # noqa: E402  (needs sys.path insert above)

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198
    load_path: Optional[str] = None


# --- Defaults (the proven 370M ladder recipe at seq-len 4096) --------------------------------
DEFAULT_DATA_CONFIG = "s3://edullm-datasets/olmo-150b-dolma2/configs/10b-config.yaml"
DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_GLOBAL_BATCH_SIZE = 192 * DEFAULT_SEQUENCE_LENGTH  # 786,432 tokens (192 sequences)
DEFAULT_RANK_MICROBATCH_SIZE = 8 * DEFAULT_SEQUENCE_LENGTH  # 32,768 tokens/rank on a single B200
DEFAULT_TOKEN_BUDGET = 10_000_000_000  # 10B-token water-fill mix (~1.35x Chinchilla for 370M)
DEFAULT_CHECKPOINT_TOKENS = 2_500_000_000  # ~4 checkpoints over a 10B run
DEFAULT_WARMUP_STEPS = 500
DEFAULT_SEED = 6198


def ladder_lr(model_params: int) -> float:
    """OLMo ladder LR formula, with the /4 seq-4096 adjustment used by the 370M ladder run."""
    base = 0.0047 * (model_params / 108_000_000) ** (-1 / 3)
    return base / 4  # seq-len 4096 adjustment (matches allenai/OLMo-ladder 370M)


def _make_mixer_config(opts):
    """Build the sequence-mixer config for the chosen recurrence, with matched head geometry."""
    common = dict(
        n_heads=opts.n_heads,
        n_v_heads=opts.n_v_heads,
        head_dim=opts.head_dim,
        expand_v=opts.expand_v,
        conv_size=opts.conv_size,
    )
    if opts.mixer == "gdn":
        return GatedDeltaNetConfig(allow_neg_eigval=True, **common)
    elif opts.mixer == "linear":
        return LinearAttentionConfig(qk_l2norm=True, normalize=opts.linear_normalize, **common)
    else:
        raise ValueError(f"_make_mixer_config called for non-recurrent mixer: {opts.mixer}")


def _swap_sequence_mixer(model_config: TransformerConfig, mixer_config) -> None:
    """Replace the sequence mixer on every block (single block, dict of blocks, or overrides)."""

    def apply(block):
        if isinstance(block, TransformerBlockConfig):
            block.sequence_mixer = mixer_config

    block = model_config.block
    if isinstance(block, dict):
        for b in block.values():
            apply(b)
    else:
        apply(block)
    if getattr(model_config, "block_overrides", None):
        for b in model_config.block_overrides.values():
            apply(b)


def build_config(opts, overrides: List[str]):
    tokenizer_config = TokenizerConfig.dolma2()

    # --- Model -------------------------------------------------------------------------------
    try:
        factory = getattr(TransformerConfig, opts.model_factory)
    except AttributeError:
        raise ValueError(f"Unknown model factory: {opts.model_factory}")
    model_kwargs = {}
    if opts.attn_backend is not None:
        model_kwargs["attn_backend"] = AttentionBackendName(opts.attn_backend)
    model_config = factory(vocab_size=tokenizer_config.padded_vocab_size(), **model_kwargs)

    # Swap the sequence mixer for the recurrent variants. `attention` leaves the stock
    # sliding-window softmax attention in place (baseline).
    if opts.mixer in ("gdn", "linear"):
        _swap_sequence_mixer(model_config, _make_mixer_config(opts))

    # --- Data: the dolma2 source mixture -----------------------------------------------------
    source_list = SourceMixtureList.from_yaml(opts.data_config)
    src_mix = SourceMixtureDatasetConfig(
        source_list=source_list,
        requested_tokens=opts.token_budget,
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
    )
    dataset_config = NumpyFSLDatasetConfig.from_src_mix(
        src_mix,
        tokenizer=tokenizer_config,
        sequence_length=opts.sequence_length,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.seed,
        num_workers=8,
    )

    # --- Optimization ------------------------------------------------------------------------
    lr = opts.lr if opts.lr is not None else ladder_lr(model_config.num_non_embedding_params)
    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=lr,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType(opts.dp_type),
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=opts.warmup_steps),
    )

    # --- Trainer -----------------------------------------------------------------------------
    tokens_per_step = opts.global_batch_size
    save_interval_steps = max(1, round(opts.checkpoint_tokens / tokens_per_step))
    ephemeral_save_interval = max(1, save_interval_steps // 4)

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=10,
            max_duration=Duration.tokens(opts.token_budget),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=save_interval_steps,
                ephemeral_save_interval=ephemeral_save_interval,
                save_async=False,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                entity="eduLLM",
                project="pretraining",
                enabled=opts.wandb,
                cancel_check_interval=10,
            ),
        )
    )

    # --- Comparable evals --------------------------------------------------------------------
    # Both runs must be judged on the SAME yardstick. Pass a common held-out set via --eval-data
    # and/or downstream tasks via --eval-tasks (use the SAME values for the linear and gdn runs).
    if opts.eval_data:
        eval_paths = [p.strip() for p in opts.eval_data.split(",") if p.strip()]
        trainer_config = trainer_config.with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=eval_paths,
                    metadata=[{"label": "heldout-val"} for _ in eval_paths],
                    sequence_length=opts.sequence_length,
                    tokenizer=tokenizer_config,
                    work_dir=opts.work_dir,
                ),
                eval_interval=opts.eval_interval,
                eval_on_finish=True,
            ),
        )
    if opts.eval_tasks:
        tasks = [t.strip() for t in opts.eval_tasks.split(",") if t.strip()]
        trainer_config = trainer_config.with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks,
                tokenizer=tokenizer_config,
                eval_interval=opts.eval_interval,
                eval_on_finish=True,
            ),
        )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        init_seed=opts.seed,
        load_path=opts.load_path,
    ).merge(overrides)

    return config


def train(config):
    if get_rank() == 0:
        rich.print(config)
        n_params = config.model.num_params
        n_non_emb = config.model.num_non_embedding_params
        log.info(f"Model: {n_params:,} params ({n_non_emb:,} non-embedding)")

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    if not trainer.no_checkpoints and not trainer.maybe_load_checkpoint() and config.load_path:
        log.info(f"No checkpoint in save folder; loading from {config.load_path}")
        trainer.load_checkpoint(config.load_path, load_trainer_state=False)

    trainer.fit()


def parse_args():
    parser = argparse.ArgumentParser(
        prog=sys.argv[0],
        description="Compare plain linear attention vs Gated DeltaNet at 370M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", type=str, help="Name of the run (used for W&B + checkpoint dir).")
    parser.add_argument("--mixer", type=str, default="gdn", choices=["attention", "gdn", "linear"],
                        help="Sequence mixer: stock softmax attention, Gated DeltaNet, or plain linear attention.")
    parser.add_argument("--model-factory", type=str, default="olmo3_370M")
    parser.add_argument("--attn-backend", type=str, default=None,
                        help="Override the base attention backend (only relevant for --mixer attention).")
    # Matched head geometry for the recurrent mixers (defaults follow d_model=1024, 16 heads).
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-v-heads", type=int, default=16,
                        help="Value heads for the recurrent mixers (state = n_v_heads*head_dim*head_v_dim).")
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--expand-v", type=float, default=1.0,
                        help="Value expansion for the recurrent mixers. Default 1.0 keeps the model "
                             "at the ~370M rung (value_dim=d_model); GDN canonical is 2.0.")
    parser.add_argument("--conv-size", type=int, default=4)
    parser.add_argument("--dp-type", type=str, default="fsdp", choices=["fsdp", "hsdp", "ddp"],
                        help="Data-parallel strategy. Default fsdp (works at world-size 1 = one GPU/run).")
    parser.add_argument("--linear-normalize", action="store_true", default=False,
                        help="Use the linear-attention denominator normalization (default: off = pure "
                             "ungated cumulative sum, the honest GDN gate/delta ablation).")
    parser.add_argument("--data-config", type=str, default=DEFAULT_DATA_CONFIG,
                        help="SourceMixtureList YAML (local path or s3://). Default: the 10B water-fill mix.")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--global-batch-size", type=int, default=DEFAULT_GLOBAL_BATCH_SIZE,
                        help="Global batch size IN TOKENS (must be a multiple of sequence-length).")
    parser.add_argument("--rank-microbatch-size", type=int, default=DEFAULT_RANK_MICROBATCH_SIZE,
                        help="Per-rank microbatch IN TOKENS.")
    parser.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                        help="Total tokens to train on (also caps how much of the mix is drawn).")
    parser.add_argument("--checkpoint-tokens", type=int, default=DEFAULT_CHECKPOINT_TOKENS)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override LR. Default: OLMo ladder formula (~7.8e-4 for 370M @ seq4096).")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-data", type=str, default=None,
                        help="Comma-separated held-out .npy paths/globs for a COMMON validation set "
                             "(use the SAME value for both runs to compare fairly).")
    parser.add_argument("--eval-tasks", type=str, default=None,
                        help="Comma-separated downstream task names (e.g. hellaswag,arc_easy).")
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-folder", type=str, default=None,
                        help="Where to write checkpoints. Must be s3://... for real runs (box is ephemeral).")
    parser.add_argument("--allow-local-save", action="store_true",
                        help="Permit a non-remote --save-folder (NOT durable; discouraged).")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Local scratch dir for dataset index/cache (e.g. /mnt/nvme/olmo-work).")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--compile", dest="compile", action="store_true", default=True,
                        help="torch.compile the model (default: on).")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile (use if a recurrent kernel won't compile).")
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Build and print the config, then exit.")
    opts, overrides = parser.parse_known_args()

    if opts.work_dir is None:
        opts.work_dir = "/tmp/olmo-dataset-cache"
    if opts.global_batch_size % opts.sequence_length != 0:
        parser.error("--global-batch-size must be a multiple of --sequence-length")

    remote_prefixes = ("s3://", "gs://", "http://", "https://")
    if opts.save_folder is None:
        opts.save_folder = f"/tmp/{opts.run_name}"
    if (
        not opts.dry_run
        and not opts.allow_local_save
        and not str(opts.save_folder).startswith(remote_prefixes)
    ):
        parser.error(
            "checkpoints must be saved to S3: pass --save-folder s3://<bucket>/<run> "
            "(or --allow-local-save to override for a throwaway local run)."
        )
    return opts, overrides


def main():
    opts, overrides = parse_args()

    if opts.dry_run:
        config = build_config(opts, overrides)
        rich.print(config)
        print(
            f"\nmixer={opts.mixer}  params={config.model.num_params:,}  "
            f"non_embedding={config.model.num_non_embedding_params:,}"
        )
        print("Dry run OK — config built. Remove --dry-run and launch under torchrun to train.")
        return

    prepare_training_environment()
    try:
        config = build_config(opts, overrides)
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
