"""Finish an interrupted training run, from its own saved config, on the platform.

    python .edullm/resume_run.py "$EDULLM_RUN_ID" \
        --from-checkpoint s3://.../stage-gdn-halfkv-resume/checkpoints/step3179 \
        --save-folder "$EDULLM_CHECKPOINT_DIR" --rank-microbatch-size 16384

WHAT THIS IS FOR. `gdn-halfkv` stopped at step3179 of 12716 -- 2.5B of a 10B-token run -- because
the box training it moved on. Its sibling `linear-halfkv` finished, so the half-KV pair is one arm
short, and the pair is what isolates state capacity from the gating mechanism.

THE CONFIG IS READ OUT OF THE CHECKPOINT, NOT RETYPED, AND THAT IS THE WHOLE DESIGN. A resumed run
has to match the original in the optimizer (`skip_step_adamw`, not the platform default `adamw`),
the learning rate (7.965e-4, from a ladder formula over the non-embedding count), the z-loss
multiplier, the global batch, the warmup, the mixer geometry and the seed. Retyping nine fields
from a launch script is nine chances to change the experiment while believing it continued;
`config.json` beside the weights already holds every one of them, exactly as they ran.

A TRUE RESUME, NOT A WARM START, AND THE DIFFERENCE IS NOT COSMETIC. `Trainer.maybe_load_checkpoint`
looks in the save folder and restores the step counter, the LR-schedule position and the
data-loader position. `--load-path` -- what `train_mixer.py` offers -- calls
`load_checkpoint(..., load_trainer_state=False)`, which takes the weights and optimizer and
restarts everything else at step 0: the cosine schedule would run a second time over the
remaining 7.5B tokens, and the model would be trained on an LR trajectory no run ever had. So
this stages the checkpoint INTO the save folder and lets `maybe_load_checkpoint` do the work.

WHY STAGING IS NEEDED AT ALL. The GPU workload role reads `edullm-data` and
`sbsandbox-intern-edullm-outputs/teams/*/runs/*` and nothing else, so a checkpoint in
`edullm-checkpoints` is unreadable from a run. It gets copied into an outputs prefix first, and
this copies from there into `$EDULLM_CHECKPOINT_DIR`, which the role may write.

TWO OVERRIDES ARE APPLIED AND BOTH ARE DELIBERATE.

  * `rank_microbatch_size`. The original ran 32,768 on a B200. On A100-40GB that is an OOM before
    the first step: at N=32,768 against a 100,352 vocab the loss path's five (N,V) fp32 tensors
    come to roughly 67 GiB against ~38 GiB usable (A100-MFU-PLAYBOOK.md B3/B4), and an OOM gets
    no Batch retry. Gradient accumulation is mathematically equivalent, so this changes step time
    and not the model.
  * `save_overwrite=False`. The original set True, which clears the target step directory before
    every save. That is precisely wrong for a run whose second attempt is meant to resume the
    first, and `train_on_corpus.py` sets False for exactly this reason.

Everything else is left as the checkpoint recorded it.
"""

import argparse
import json
import logging
import os
import sys
from typing import cast

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "linear-attn-vs-gdn"
    ),
)

import olmo_linear_attn  # noqa: E402,F401  (registers linear_attention, for a linear resume)

from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig  # noqa: E402
from olmo_core.distributed.utils import get_rank  # noqa: E402
from olmo_core.io import copy_dir, get_bytes_range, get_file_size  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.train import (  # noqa: E402
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback  # noqa: E402
from olmo_core.train.train_module import TransformerTrainModuleConfig  # noqa: E402
from olmo_core.utils import seed_all  # noqa: E402

log = logging.getLogger("resume_run")


def read_json(uri: str):
    return json.loads(get_bytes_range(uri, 0, get_file_size(uri)).decode("utf-8"))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Resume an interrupted run from its saved config.")
    p.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    p.add_argument("--from-checkpoint", required=True, help="Staged stepNNNNN dir to resume from.")
    p.add_argument("--save-folder", default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""))
    p.add_argument("--rank-microbatch-size", type=int, default=16384)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--dry-run", action="store_true")
    opts, overrides = p.parse_known_args()

    if not opts.save_folder:
        raise SystemExit("--save-folder (or EDULLM_CHECKPOINT_DIR) is required")

    src = opts.from_checkpoint.rstrip("/")
    step_name = src.rsplit("/", 1)[-1]
    saved = read_json(f"{src}/config.json")
    log.info(f"read config from {src}/config.json")

    model_cfg = TransformerConfig.from_dict(saved["model"])
    dataset_cfg = NumpyFSLDatasetConfig.from_dict(saved["dataset"])
    loader_cfg = NumpyDataLoaderConfig.from_dict(saved["data_loader"])
    tm_cfg = cast(
        TransformerTrainModuleConfig, TransformerTrainModuleConfig.from_dict(saved["train_module"])
    )
    trainer_cfg = cast(TrainerConfig, TrainerConfig.from_dict(saved["trainer"]))

    mixer = getattr(model_cfg.block, "sequence_mixer", None)
    log.info(
        f"resuming {type(mixer).__name__} d_model={model_cfg.d_model} "
        f"L={model_cfg.n_layers} params={model_cfg.num_params:,}"
    )
    log.info(
        f"recipe as recorded: optim={type(tm_cfg.optim).__name__} lr={tm_cfg.optim.lr} "
        f"z_loss={tm_cfg.z_loss_multiplier} gbs={loader_cfg.global_batch_size} "
        f"seq={tm_cfg.max_sequence_length} max_duration={trainer_cfg.max_duration}"
    )

    # The two overrides, announced rather than silent.
    old_mb = tm_cfg.rank_microbatch_size
    tm_cfg.rank_microbatch_size = opts.rank_microbatch_size
    log.info(
        f"OVERRIDE rank_microbatch_size {old_mb} -> {tm_cfg.rank_microbatch_size} (A100 memory)"
    )
    trainer_cfg.save_folder = opts.save_folder
    trainer_cfg.save_overwrite = False
    log.info(f"OVERRIDE save_overwrite -> False; save_folder -> {opts.save_folder}")
    if "checkpointer" in trainer_cfg.callbacks:
        ck = trainer_cfg.callbacks["checkpointer"]
        ck.save_interval = opts.save_interval  # type: ignore[attr-defined]
        ck.ephemeral_save_interval = None  # type: ignore[attr-defined]
        # max_checkpoints=None or the prune deletes a .metadata.json the workload role may not
        # delete, and the run dies partway through -- see train_on_corpus.py.
        ck.max_checkpoints = None  # type: ignore[attr-defined]
    for name in ("lm_evaluator", "downstream_evaluator"):
        if name in trainer_cfg.callbacks:
            trainer_cfg.callbacks[name].enabled = False  # type: ignore[attr-defined]
            log.info(f"disabled {name}: it fails at trainer construction on this platform")

    dest = f"{opts.save_folder.rstrip('/')}/{step_name}"
    if opts.dry_run:
        log.info(f"DRY RUN: would copy {src} -> {dest}, then resume to {trainer_cfg.max_duration}")
        return 0

    # STAGE INTO THE SAVE FOLDER so maybe_load_checkpoint finds it. Rank 0 only, then a barrier:
    # every rank must see the copy finished before the loader looks for it.
    prepare_training_environment()
    try:
        from olmo_core.distributed.utils import barrier

        if get_rank() == 0:
            log.info(f"staging {src} -> {dest}")
            copy_dir(src, dest, save_overwrite=True)
            log.info("staged")
        barrier()

        seed_all(saved.get("init_seed", 6198))
        model = model_cfg.build(init_device="meta")
        train_module = tm_cfg.build(model)
        dataset = dataset_cfg.build()
        data_loader = loader_cfg.build(dataset, dp_process_group=train_module.dp_process_group)
        trainer = trainer_cfg.build(train_module, data_loader)
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = saved

        # THE ASSERTION THIS FILE EXISTS FOR. If the staged checkpoint is not found, Trainer
        # starts from scratch at step 0 and trains a fresh model for 10B tokens while every log
        # line looks ordinary -- the expensive silent failure, not a crash.
        if not trainer.maybe_load_checkpoint():
            raise RuntimeError(
                f"no checkpoint found in {opts.save_folder} after staging {step_name}. Refusing to "
                "start from scratch: that would spend a full run's compute on a model nobody asked "
                "for and report success."
            )
        log.info(f"resumed at step {trainer.global_step} of {trainer_cfg.max_duration}")
        trainer.fit()
        log.info(f"finished at step {trainer.global_step}")
        return 0
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    sys.exit(main())
