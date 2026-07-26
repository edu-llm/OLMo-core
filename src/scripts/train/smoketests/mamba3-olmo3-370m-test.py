"""
Smoke test: the ``mamba3_olmo3_370M`` ablation, end-to-end.

Trains :meth:`~olmo_core.nn.mamba3.config.Mamba3Config.mamba3_olmo3_370M` -- OLMo-3-370M with
Mamba-3 in place of the sliding-window attention layers, parameter-matched to
``TransformerConfig.olmo3_370M`` (372.4M vs 371.3M active non-embedding) -- for 20 steps to
confirm the ablation config trains end to end inside the stock
:class:`~olmo_core.train.train_module.TransformerTrainModule`.

This is the sibling of ``mamba3-test.py`` (which exercises the 190M hybrid). It is derived from
the same base (``src/scripts/train/OLMo3/OLMo-3-190M.py``) and reuses that script's guards; the
only substantive change is the model preset. The point is to catch anything specific to the
370M ablation -- the wider ``d_model=1024``/16-layer shape, ``use_rope=True`` on the attention
layers, and the SISO (``mimo_rank=1``) default -- that the 190M smoke test would not.

Success:

- All 20 steps complete with no kernel, shape, or dtype errors out of the Mamba-3 mixer.
- ``train/CE loss`` stays finite and trends downward.
- The LM evaluator reports CE loss and PPL, confirming eval mode runs.
- The Mamba-3 sentinel raises no critical alert (no non-finite grad norm, no stalled
  skip-step optimizer, and the decay horizon covers the context).

What the preset fixes, and why it is not overridden here:
    ``mamba3_olmo3_370M`` already defaults to ``mimo_rank=1`` (so the run is eligible for the
    official ``mamba-ssm`` SISO Triton kernel), ``n_groups=1``, ``a_log_init_max=0.1`` (so the
    decay horizon covers a long context rather than the ~114-step horizon the library default of
    16 gives), and ``rotation_block_size=2`` (the TC^0 baseline). Those are the ablation's real
    settings, so the smoke test uses them as-is.

``MAMBA3_ROTATION_BLOCK_SIZE`` selects the transition-block size ``b``: ``2`` is the TC^0
    baseline (default); ``3`` is the smallest non-solvable (NC^1) block (``A_5 subset SO(3)``,
    and ``SO(3)`` is simple so it cannot collapse to an abelian subgroup); ``4`` also works
    (``SO(4) contains SO(3)``) but is *fragile* -- ``SO(4)`` has an abelian ``SO(2)xSO(2)``
    maximal torus the model can settle into, silently reverting to b=2 behavior. The preset's
    default ``d_state`` (``DEFAULT_D_STATE``, 192) admits all of 2, 3 and 4, so switching arms is
    a single ``MAMBA3_ROTATION_BLOCK_SIZE`` change; ``MAMBA3_D_STATE`` overrides the state size
    only if you want a different one. Set here rather than via a ``--model...`` override so the
    sentinel's expected value stays in lockstep.

Running outside Ai2 (no Beaker token, no GCS) uses the same environment overrides as
``mamba3-test.py``: ``MAMBA3_DATA_ROOT`` (the data root, consumed by *both* the training and
eval datasets), ``MAMBA3_SAVE_FOLDER``, ``MAMBA3_WORK_DIR``, ``MAMBA3_ACTIVATION_CHECKPOINTING``,
and ``MAMBA3_DISABLE_DP``. See ``mamba3-test.py`` for the full description of each.

Examples:
    Dry run (renders the config; needs a data root set even though it reads no data):
        MAMBA3_DATA_ROOT=/tmp/ai2-llm MAMBA3_SAVE_FOLDER=/tmp/m3-370m \
            python src/scripts/train/smoketests/mamba3-olmo3-370m-test.py \
            dry_run test-mamba3-370m local

    Single node (e.g. one AWS B200 box), 8 GPUs:
        export MAMBA3_DATA_ROOT=/data/ai2-llm
        export MAMBA3_SAVE_FOLDER=/data/checkpoints/mamba3-370m
        torchrun --nproc-per-node=8 \
            src/scripts/train/smoketests/mamba3-olmo3-370m-test.py \
            train test-mamba3-370m local

    NC^1 arm (b=3, the non-fragile block):
        MAMBA3_ROTATION_BLOCK_SIZE=3 MAMBA3_DATA_ROOT=/data/ai2-llm \
            MAMBA3_SAVE_FOLDER=/data/checkpoints/mamba3-370m-nc1 \
            torchrun --nproc-per-node=8 \
            src/scripts/train/smoketests/mamba3-olmo3-370m-test.py \
            train test-mamba3-370m-nc1 local
"""

import os
from datetime import datetime
from typing import Optional

from olmo_core.config import DType
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
    get_work_dir,
)
from olmo_core.internal.cookbook import configure_required_callbacks
from olmo_core.internal.experiment import CliContext, ExperimentConfig, main
from olmo_core.launch.beaker import BeakerLaunchConfig
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.transformer import (
    TransformerActivationCheckpointingMode,
    TransformerDataParallelWrappingStrategy,
)
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import LMEvaluatorCallbackConfig, WandBCallback
from olmo_core.train.train_module import (
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

# Sibling module, resolved via the script's own directory on sys.path.
from mamba3_sentinel import Mamba3SentinelCallback  # isort: skip

SEQ_LENGTH = 512


def _rotation_block_size(model_config: Mamba3Config) -> int:
    """
    Read the SSM rotation block size out of a named-block hybrid config.

    :param model_config: A hybrid config built by ``Mamba3Config.mamba3_*``.
    :returns: The ``rotation_block_size`` of the Mamba-3 blocks.
    """
    blocks = model_config.block
    assert isinstance(blocks, dict), "hybrid Mamba-3 configs use named blocks"
    return blocks["mamba3"].sequence_mixer.rotation_block_size


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the ``mamba3_olmo3_370M`` smoke-test experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :returns: The experiment config, with any CLI overrides merged in.
    """
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )

    # Both `get_root_dir()` and `build_launch_config()` need a Beaker token. Without one, skip the
    # launch config so the script still renders and can be driven directly by torchrun.
    beaker_launch_config: Optional[BeakerLaunchConfig] = None
    has_beaker = get_beaker_username() is not None
    if has_beaker:
        root_dir = get_root_dir(cli_context.cluster)
        beaker_launch_config = build_launch_config(
            name=cli_context.run_name,
            cmd=cli_context.remote_cmd,
            cluster=cli_context.cluster,
            root_dir=root_dir,
            workspace="ai2/OLMo_3",
            num_nodes=1,
            nccl_debug=False,
        )
    else:
        root_dir = "gs://ai2-llm"

    # Single source of truth for the data root; feeds both the training and eval `mix_base_dir`.
    data_root = os.environ.get("MAMBA3_DATA_ROOT")
    if data_root is None:
        if not has_beaker:
            # Defaulting to `gs://ai2-llm` on a machine with no GCS credentials fails minutes
            # into a run that has already taken a GPU, not at startup. Refuse to start instead.
            raise ValueError(
                "No Beaker credentials, so there is no default data root. Set MAMBA3_DATA_ROOT "
                "to a local corpus directory (or an s3:// prefix this machine can read), e.g. "
                "MAMBA3_DATA_ROOT=/mnt/nvme/corpus."
            )
        data_root = root_dir

    if (save_dir := os.environ.get("MAMBA3_SAVE_FOLDER")) is None:
        save_dir = (
            f"{root_dir}/checkpoints/{cli_context.run_name}"
            if has_beaker
            else f"./checkpoints/{cli_context.run_name}"
        )

    if (work_dir := os.environ.get("MAMBA3_WORK_DIR")) is None:
        work_dir = get_work_dir(root_dir) if has_beaker else f"{save_dir}/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    # The ablation preset: OLMo-3-370M with Mamba-3 replacing the sliding-window layers. Its
    # defaults (mimo_rank=1, n_groups=1, a_log_init_max=0.1, use_rope=True, rotation_block_size=2)
    # are the real ablation settings, so they are used as-is. `d_state` is left to the preset's
    # default (`DEFAULT_D_STATE`, 192), which admits b=2, 3 and 4 -- so the TC^0 baseline (b=2)
    # and the non-solvable NC^1 arms (b=3, the smallest/non-fragile block; b=4) are all a single
    # `MAMBA3_ROTATION_BLOCK_SIZE` change with no d_state juggling. `MAMBA3_D_STATE` overrides it
    # only if you deliberately want a different state size. Both are set here, not via a
    # `--model...` override, so the sentinel's expected block size stays in lockstep (a CLI
    # override merges in after the sentinel is built).
    rotation_block_size = int(os.environ.get("MAMBA3_ROTATION_BLOCK_SIZE", "2"))
    d_state_override = os.environ.get("MAMBA3_D_STATE")
    d_state_kwargs = {} if d_state_override is None else {"d_state": int(d_state_override)}
    model_config = Mamba3Config.mamba3_olmo3_370M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        rotation_block_size=rotation_block_size,
        **d_state_kwargs,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=SEQ_LENGTH * 2,
        max_sequence_length=SEQ_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=3e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=False,
        # FSDP, not HSDP: at world size 1 HSDP only builds a degenerate size-1 replicate mesh.
        # MAMBA3_DISABLE_DP=1 drops data parallelism entirely for single-GPU local runs, where
        # FSDP2's reduce-scatter can fail on the degenerate 1-rank NCCL path. Leave unset (FSDP)
        # for the real multi-GPU B200 run.
        dp_config=(
            None
            if os.environ.get("MAMBA3_DISABLE_DP") == "1"
            else TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
                wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
            )
        ),
        # Without FSDP nothing casts the model, so the SSD input would be fp32 and the official
        # SISO kernel would not arm; autocast restores reduced precision on the single-device run.
        autocast_precision=(DType.bfloat16 if os.environ.get("MAMBA3_DISABLE_DP") == "1" else None),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=2),
        ac_config=(
            TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.full
            )
            if os.environ.get("MAMBA3_ACTIVATION_CHECKPOINTING") == "1"
            else None
        ),
    )

    # A small eval dataset doubles as training data -- this is only a smoke test.
    dataset_config = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=data_root,
        work_dir=work_dir,
        tokenizer=tokenizer_config,
        sequence_length=SEQ_LENGTH,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=SEQ_LENGTH * 16, seed=34521, num_workers=4
    )

    cancel_check_interval = 5
    trainer_config = (
        TrainerConfig(
            save_folder=save_dir,
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=cancel_check_interval,
            max_duration=Duration.steps(20),
        )
        .with_callbacks(configure_required_callbacks(run_name_with_ts))
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name_with_ts,
                group=cli_context.run_name,
                entity="ai2-llm",
                project="olmo3",
                enabled=False,
                cancel_check_interval=cancel_check_interval,
            ),
        )
        .with_callback(
            "mamba3_sentinel",
            Mamba3SentinelCallback(
                run_dir=save_dir,
                sequence_length=SEQ_LENGTH,
                expected_rotation_block_size=_rotation_block_size(model_config),
            ),
        )
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=data_root,
                    sequence_length=SEQ_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=10,
                eval_duration=Duration.steps(4),
            ),
        )
    )

    experiment_config = ExperimentConfig(
        run_name=cli_context.run_name,
        launch=beaker_launch_config,
        model=model_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        init_seed=1337,
    )
    experiment_config = experiment_config.merge(cli_context.overrides)
    return experiment_config


if __name__ == "__main__":
    main(config_builder=build_experiment_config)
