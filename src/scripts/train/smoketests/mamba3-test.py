"""
Smoke test: Mamba-3 hybrid end-to-end training.

Trains the ``mamba3_hybrid_190M`` preset from :class:`~olmo_core.nn.mamba3.config.Mamba3Config`
(12 layers = 9 Mamba-3 + 3 attention, ~270M params) for 20 steps to verify that the Mamba-3
mixer trains end to end inside the stock :class:`TransformerTrainModule`.

Success:

- All 20 steps complete with no kernel, shape, or dtype errors out of the Mamba-3 mixer.
- ``train/CE loss`` stays finite (no NaN/Inf) and trends downward.
- The LM evaluator reports CE loss and PPL at steps 10 and 20, confirming the mixer also runs
  in eval mode (``torch.no_grad``, no gating on the training graph).

Reference kernel, not the fast kernel:
    The fast ``mamba-ssm`` Mamba-3 kernel is NOT installed in this environment, so
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.dispatch_mamba3_ssd` falls through to the
    pure-PyTorch :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.mamba3_ssd_reference`. The
    :func:`~olmo_core.nn.mamba3.mamba3_ssd_api.has_mamba3` probe imports
    ``mamba_ssm.modules.mamba3``, which only exists on ``mamba-ssm`` ``main`` and requires a
    source build. What this run validates is therefore the fallback path.

Where this deviates from the 190M base config, and why:
    ``SEQ_LENGTH`` is 512 rather than the base config's 8192. The reference SSD kernel is a
    sequential Python loop over timesteps, and each step leaves a ``(batch, n_heads, mimo_rank,
    d_state, head_dim)`` tensor alive for autograd -- on the order of 3 MiB per sequence per
    timestep at this model size. At 8192 timesteps that is tens of GiB per Mamba-3 layer per
    sequence and hundreds of GiB across the nine Mamba-3 layers, which OOMs on any single GPU.
    At 512 the whole 20-step run fits in roughly 30 GiB while still exercising every code path.
    Raise it once the fast kernel is wired up.

    ``compile_model`` is off. ``torch.compile`` would unroll the reference kernel's per-timestep
    Python loop into a graph with thousands of nodes per Mamba-3 layer.

    The scheduler warms up over 2 steps instead of 2000, so that the learning rate is actually
    non-negligible within a 20-step run and "loss decreases" is a meaningful signal.

Running outside Ai2 (no Beaker token, no GCS):
    Without a Beaker token there is no launch config, so :class:`BeakerCallback` self-skips and
    plain ``torchrun`` is the intended entry point. Two things still have to be redirected away
    from ``gs://ai2-llm``: the data root and the checkpoint save folder. Both have environment
    variable overrides, which is the recommended way to set them because **the data root is
    consumed in two independent places** - the training dataset and the LM evaluator's dataset -
    and setting only one of them is the classic way to get a confusing failure halfway through
    a run.

    ``MAMBA3_DATA_ROOT``
        Base dir for the ``v3_small_ppl_validation`` mix, used for *both* datasets. Accepts a
        local path, ``s3://...``, or ``gs://...``. The mix resolves to
        ``<root>/eval-data/perplexity/v3_small_dolma2-tokenizer/*/val/part-0-00000.npy``
        (11 files), so whatever you point this at must reproduce that relative layout.
        Defaults to the Beaker-derived root, or ``gs://ai2-llm`` when there is no token.

    ``MAMBA3_SAVE_FOLDER``
        Checkpoint/metrics output dir. Defaults to the Beaker-derived root when a token is
        present, otherwise to the local ``./checkpoints/<run-name>`` - never ``gs://`` on a
        machine that has no Beaker credentials.

    ``MAMBA3_WORK_DIR``
        Dataset cache dir (tokenized-instance indices). Defaults to a ``dataset-cache``
        subdirectory of the save folder when running outside Beaker.

    ``MAMBA3_ACTIVATION_CHECKPOINTING``
        Set to ``1`` to checkpoint every block. Off by default. This is the escape hatch for the
        reference SSD kernel's ``O(T)`` activation memory, and it cannot be reached with a
        ``--train_module.ac_config.mode=...`` override because ``ac_config`` defaults to ``None``
        and the override merge cannot create a config node from nothing.

    The equivalent ``--`` overrides, if you would rather be explicit than use the environment,
    are listed under "Examples" below. Note again that the data root needs *two* of them.

Examples:
    Dry run:
        python src/scripts/train/smoketests/mamba3-test.py \
            dry_run test-mamba3 ai2/jupiter

    Launch:
        python src/scripts/train/smoketests/mamba3-test.py \
            launch test-mamba3 ai2/jupiter \
            --launch.priority=low \
            --launch.follow=false

    Single node outside Beaker (e.g. one AWS B200 box), 8 GPUs, via the environment:
        export MAMBA3_DATA_ROOT=/data/ai2-llm
        export MAMBA3_SAVE_FOLDER=/data/checkpoints/mamba3-test
        torchrun --nproc-per-node=8 src/scripts/train/smoketests/mamba3-test.py \
            train test-mamba3 local

    The same thing spelled out as explicit overrides. Both ``mix_base_dir`` settings are
    required; they are separate config nodes:
        torchrun --nproc-per-node=8 src/scripts/train/smoketests/mamba3-test.py \
            train test-mamba3 local \
            --trainer.save_folder=/data/checkpoints/mamba3-test \
            --dataset.mix_base_dir=/data/ai2-llm \
            --trainer.callbacks.lm_evaluator.eval_dataset.mix_base_dir=/data/ai2-llm
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


def _rotation_block_size(model_config: Mamba3Config) -> int:
    """
    Read the SSM rotation block size out of a named-block hybrid config.

    :param model_config: A hybrid config built by ``Mamba3Config.mamba3_hybrid_*``.

    :returns: The ``rotation_block_size`` of the Mamba-3 blocks.
    """
    blocks = model_config.block
    assert isinstance(blocks, dict), "hybrid Mamba-3 configs use named blocks"
    return blocks["mamba3"].sequence_mixer.rotation_block_size


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


def build_experiment_config(cli_context: CliContext) -> ExperimentConfig:
    """
    Build the Mamba-3 hybrid smoke-test experiment config.

    :param cli_context: The CLI context supplied by :func:`olmo_core.internal.experiment.main`.
    :returns: The experiment config, with any CLI overrides merged in.
    """
    run_name_with_ts = (
        f"{cli_context.run_name}-{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
    )

    # Both `get_root_dir()` and `build_launch_config()` need a Beaker token. Without one, fall
    # back to the same root dir `get_root_dir()` uses for non-Beaker clusters and skip the launch
    # config, so the script still renders and can be driven directly by torchrun.
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

    # Single source of truth for the data root. It feeds two independent `mix_base_dir` settings
    # below (training dataset + LM evaluator dataset); they must not be allowed to drift.
    data_root = os.environ.get("MAMBA3_DATA_ROOT")
    if data_root is None:
        if not has_beaker:
            # Defaulting to `root_dir` here would silently point the loader at `gs://ai2-llm` on
            # a machine with no GCS credentials. That does not fail at startup -- it fails when
            # the loader first reaches for a shard, minutes into a run that has already taken a
            # GPU. Refuse to start instead.
            raise ValueError(
                "No Beaker credentials, so there is no default data root. Set MAMBA3_DATA_ROOT "
                "to a local corpus directory (or an s3:// prefix this machine can read), e.g. "
                "MAMBA3_DATA_ROOT=/mnt/nvme/corpus."
            )
        data_root = root_dir

    if (save_dir := os.environ.get("MAMBA3_SAVE_FOLDER")) is None:
        # Off Beaker there are no GCS credentials, so a `gs://` save folder would fail on the
        # first checkpoint rather than at startup. Default somewhere writable instead.
        save_dir = (
            f"{root_dir}/checkpoints/{cli_context.run_name}"
            if has_beaker
            else f"./checkpoints/{cli_context.run_name}"
        )

    if (work_dir := os.environ.get("MAMBA3_WORK_DIR")) is None:
        work_dir = get_work_dir(root_dir) if has_beaker else f"{save_dir}/dataset-cache"

    tokenizer_config = TokenizerConfig.dolma2()

    # Unlike `TransformerConfig.olmo3_190M`, this preset takes no `attn_backend`; its three
    # attention layers use the default backend and NoPE, since the Mamba-3 layers carry position.
    #
    # `a_log_init_max` is lowered from the library default of 16. At 16 the decay is
    # `alpha ~= 0.92`, and the measured horizon on this exact preset is 22-114 steps against a
    # 512-token context -- the model would train happily while most of its context stayed out of
    # reach, which looks identical to success on the loss curve. At 1.0 the median head reaches
    # ~2800 steps.
    # `mimo_rank=1` is what makes the run eligible for the official `mamba-ssm` SISO Triton
    # kernel: `dispatch_mamba3_ssd` requires rank 1, so at the library default of 4 the smoke
    # test silently took the chunked PyTorch path no matter what was installed -- meaning it
    # exercised a kernel the real runs will not use, which is the opposite of what a smoke test
    # is for. MIMO also buys no state-tracking power (it widens the read/write rank, not the
    # transition monoid) while multiplying the cost of applying the rotation.
    #
    # `MAMBA3_ROTATION_BLOCK_SIZE` selects the transition-block size b: 2 is the TC^0 baseline
    # (default), 3 is the smallest non-solvable (NC^1) block. Set here rather than via a
    # `--model...` override on purpose -- the sentinel below reads its expected value from this
    # same `model_config`, so threading it here keeps the two in lockstep, whereas a CLI override
    # merges in *after* the sentinel is built and would trip its rotation-block-size mismatch
    # alarm. `DEFAULT_D_STATE` (192) admits both 2 and 3, so no `d_state` change is needed.
    rotation_block_size = int(os.environ.get("MAMBA3_ROTATION_BLOCK_SIZE", "2"))
    # Main runs use the fast official kernel (the default, `prefer_official_kernel=None`, which
    # arms it whenever eligible). Activation checkpointing is the *only* reason to deviate: the
    # official kernel's autograd.Function is incompatible with non-reentrant checkpointing, so an
    # AC run must take the chunked path. Tie the two together here rather than letting a user set
    # AC=1 and hit a mid-backward CheckpointError. AC is the extreme-sequence-length escape hatch;
    # at the smoke config's seq 512 it is off and the fast kernel runs.
    activation_checkpointing = os.environ.get("MAMBA3_ACTIVATION_CHECKPOINTING") == "1"
    model_config = Mamba3Config.mamba3_hybrid_190M(
        vocab_size=tokenizer_config.padded_vocab_size(),
        a_log_init_max=1.0,
        mimo_rank=1,
        rotation_block_size=rotation_block_size,
        prefer_official_kernel=False if activation_checkpointing else None,
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
        # FSDP, not HSDP: HSDP's replicate dimension is the node count, which is 1 for this
        # single-node smoke test, so HSDP can only build a degenerate mesh with a size-1
        # dimension. It does not error, it just adds a mesh dimension and wrapping cost for
        # nothing. FSDP is correct at both 1 and 8 GPUs on one node.
        #
        # `MAMBA3_DISABLE_DP=1` drops data parallelism entirely (`dp_config=None`). This exists
        # for single-GPU local smoke runs: FSDP2's reduce-scatter is a NCCL collective, and on a
        # single consumer GPU (e.g. an RTX 50-series laptop card) that collective can fail with
        # "CUDA driver error: device not ready" in `foreach_reduce` -- an issue in the degenerate
        # 1-rank NCCL path, not in the model. With no data-parallel wrapping there are no
        # collectives and the run exercises the full model, optimizer, checkpointing and eval on
        # one device. Leave it unset (FSDP) for the real multi-GPU B200 run.
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
        # With FSDP the bf16 cast comes from `param_dtype`; without it (single-device local run)
        # nothing casts the model, so the SSD input would be fp32 and `dispatch_mamba3_ssd` would
        # fall through to the chunked path -- defeating the point of exercising the official
        # kernel. Autocast restores reduced precision so the official SISO kernel still arms.
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

    # Use a small eval dataset as training data — this is just a smoke test.
    # NOTE: `mix_base_dir` is set here *and* on the LM evaluator's dataset below. Override both,
    # or set MAMBA3_DATA_ROOT once.
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
                # The second of the two `mix_base_dir` settings — see the note on the training
                # dataset above.
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=data_root,
                    sequence_length=SEQ_LENGTH,
                    tokenizer=tokenizer_config,
                    work_dir=work_dir,
                ),
                eval_interval=10,
                # Cap the eval loop: a full epoch over the validation mix would dominate the
                # run time given the sequential reference kernel.
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
