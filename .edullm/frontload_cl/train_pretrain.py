"""
Frontload-cl pretrain: OLMo2-370M, 10B tokens, primer vs control schedule.

Platform path (``frontload-cl-10b-v1`` is registered; push this ``edullm/**`` branch so
the image builds, then check / submit via the CLI — do not call AWS from a laptop)::

    git push -u origin edullm/frontload-cl

    edullm check  --json --team pre-training --experiment frontload-cl \\
      --dataset frontload-cl-10b-v1 --compute gpu-8xa100
    edullm submit --team pre-training --experiment frontload-cl \\
      --dataset frontload-cl-10b-v1 --compute gpu-8xa100

Committed specs under ``.edullm/`` (command text must name dtype and checkpoint dir)::

    .edullm/run.yaml          — full primer arm (default)
    .edullm/run-smoke.yaml    — 20-step GPU smoke (``edullm check --spec …``)
    .edullm/run-control.yaml  — full control arm

GPU smoke (same 370M / microbatch / flash_2 as the real run, 20 steps) — run this
before the full arms on the target 8×A100 shape::

    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
      .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \\
      --arm primer --smoke --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --param-dtype bfloat16'

Primer arm (8 GPUs)::

    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
      .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \\
      --arm primer --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --param-dtype bfloat16'

Control arm::

    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
      .edullm/frontload_cl/train_pretrain.py "$EDULLM_RUN_ID" \\
      --arm control --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --param-dtype bfloat16'

Dry-run (resolve corpus + print curriculum, no fit) — use ``olmo-core-check`` on CPU
and waive checkpoint if you want::

    bash -lc 'EDULLM_CHECKPOINT_CHECK=waived python .edullm/frontload_cl/train_pretrain.py \\
      "$EDULLM_RUN_ID" --arm primer --dry-run --save-folder "$EDULLM_CHECKPOINT_DIR" \\
      --param-dtype bfloat16'

Notes
-----
* Defaults: FlashAttention-2 (``--attn-backend flash_2``), ``torch.compile``, HSDP,
  rank microbatch ``24×4096`` (fills 8-GPU share of global batch 192 with no grad accum).
  Fall back with ``--attn-backend torch`` or lower ``--rank-microbatch-size`` if needed.
* 10B @ 370M is ~12,715 steps. The routine ``olmo-core-train`` ceiling is 24h / 2 attempts.
  If a shape cannot finish in 24h, ask for a runtime exception or chain via resume
  (same run id / checkpoint dir on Batch retry only — a new submission is a new run id).
* ``pretrain/frontload-cl-10b/v1`` is on ``s3://edullm-data`` and registered as
  ``frontload-cl-10b-v1`` (``edullm data frontload-cl-10b-v1``).
* Put ``--param-dtype bfloat16`` in the command: the platform guard reads command words and
  cannot see a dtype set only in code. A T4 has no bfloat16 in hardware.
* Curriculum (``schedule.py``): both arms share an HQ-main warmup (~371M, no SFT-like);
  then primer does a 100M SFT-like block + mixed rest, control flat-mixes remaining HQ +
  all 200M SFT-like; same HQ anneal. Mix targets floor to seq length; tiny
  ``max_repetition_factor`` so pool-edge packing remainders do not refuse a complete corpus.
  A short corpus fails at mix build or at the steps×batch cover check — it does not under-train.
* Checkpoints go to ``$EDULLM_CHECKPOINT_DIR`` →
  ``s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/<run_id>/checkpoints/``.
* SFT is a separate script (``.edullm/frontload_cl/train_sft.py``); both PT arms share it.
  Full design / status: ``.edullm/frontload_cl/DESIGN.md``.
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

# Allow ``python .edullm/frontload_cl/train_pretrain.py`` (not installed as a package).
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
    require_platform_env,
    resolve_corpus,
)
from frontload_cl.schedule import build_phases  # noqa: E402

from olmo_core.config import Config, DType  # noqa: E402
from olmo_core.data.composable import (  # noqa: E402
    ComposableDataLoaderConfig,
    ShuffleStrategy,
)
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.utils import barrier, get_rank  # noqa: E402
from olmo_core.exceptions import OLMoConfigurationError  # noqa: E402
from olmo_core.io import clear_directory, list_directory, normalize_path  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
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
    validate_precision_support,
)
from olmo_core.utils import seed_all  # noqa: E402

log = logging.getLogger(__name__)

STEP_DIRECTORY = re.compile(r"^step(\d+)$")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    data_loader: ComposableDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    arm: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
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


def apply_smoke_overrides(opts) -> None:
    """Shrink duration / checkpointing for a GPU smoke while keeping memory shape."""
    if not opts.smoke:
        return
    if opts.steps == C.TOTAL_STEPS:
        opts.steps = C.SMOKE_STEPS
    # Skip mid-run periodic saves; smoke still exercises compile + a few steady steps.
    if opts.save_interval == C.DEFAULT_SAVE_INTERVAL:
        opts.save_interval = max(opts.steps + 1, C.SMOKE_STEPS + 1)
    log.info(
        "smoke mode: steps=%d save_interval=%d microbatch=%d attn=%s",
        opts.steps,
        opts.save_interval,
        opts.rank_microbatch_size,
        opts.attn_backend,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frontload_cl.train_pretrain",
        description="OLMo2-370M frontload-cl pretrain (primer|control).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_name", nargs="?", default=default_run_name())
    p.add_argument("--arm", choices=("primer", "control"), required=True)
    p.add_argument("--dataset-id", default=os.environ.get("EDULLM_DATASET_ID", C.DATASET_ID))
    p.add_argument(
        "--dataset-version", default=os.environ.get("EDULLM_DATASET_VERSION", "")
    )
    p.add_argument(
        "--dataset-tokenizer",
        default=os.environ.get("EDULLM_DATASET_TOKENIZER", C.TOKENIZER_ID),
    )
    p.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        help="Must expand $EDULLM_CHECKPOINT_DIR on the platform command line.",
    )
    p.add_argument("--work-dir", default="/tmp/frontload-cl-cache")
    p.add_argument("--sequence-length", type=int, default=C.SEQ_LENGTH)
    p.add_argument("--steps", type=int, default=C.TOTAL_STEPS)
    p.add_argument("--warmup-steps", type=int, default=C.WARMUP_STEPS)
    p.add_argument("--learning-rate", type=float, default=C.PEAK_LR)
    p.add_argument("--global-batch-size", type=int, default=C.GLOBAL_BATCH_SIZE)
    p.add_argument(
        "--rank-microbatch-size",
        type=int,
        default=C.DEFAULT_RANK_MICROBATCH_SIZE,
        help="Tokens per microbatch per rank. Default fills 8-GPU share of global batch "
        f"({C.DEFAULT_RANK_MICROBATCH_SIZE} = 24×{C.SEQ_LENGTH}). Lower if OOM.",
    )
    p.add_argument(
        "--attn-backend",
        default=C.DEFAULT_ATTN_BACKEND,
        choices=("flash_2", "flash_3", "flash_4", "torch", "te"),
        help="Attention backend. flash_2 needs flash-attn in the image (A100+).",
    )
    p.add_argument(
        "--save-interval",
        type=int,
        default=C.DEFAULT_SAVE_INTERVAL,
        help="Periodic checkpoint every N steps. Also saves at curriculum milestones "
        "(warmup / primer / anneal) unless within proximity of a periodic save.",
    )
    p.add_argument(
        "--checkpoint-milestone-proximity",
        type=int,
        default=C.CHECKPOINT_MILESTONE_PROXIMITY,
        help="Skip a curriculum milestone checkpoint if it is within this many steps "
        "of a periodic save_interval multiple.",
    )
    p.add_argument("--data-seed", type=int, default=C.DATA_SEED)
    p.add_argument("--model-factory", default=C.MODEL_FACTORY)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve corpus, build curriculum, visualize, exit.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"GPU smoke: {C.SMOKE_STEPS} steps (unless --steps is set), same model/"
            "microbatch/attn as a full run, skip mid-run checkpoints. Use on the "
            "target 8×A100 shape to catch OOM / flash / compile failures before a 10B run."
        ),
    )
    p.add_argument(
        "--param-dtype",
        default=DType.bfloat16.value,
        choices=[DType.bfloat16.value, DType.float16.value, DType.float32.value],
        help="Parameter dtype HSDP holds and computes in. THE DEFAULT IS THE DTYPE THIS "
        "FILE ALWAYS USED. Name it on the platform command so the submission guard can "
        "see it (code defaults are invisible to that check). float32 on a T4.",
    )
    return p


def build_config(opts, overrides: List[str]) -> ExperimentConfig:
    corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version or "latest",
        tokenizer_id=opts.dataset_tokenizer,
    )
    log.info(
        "%s/%s arm=%s sources=%s",
        corpus.dataset_id,
        corpus.version,
        opts.arm,
        {k: len(v) for k, v in corpus.paths_by_source.items()},
    )

    factory = getattr(TransformerConfig, opts.model_factory, None)
    if factory is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {opts.model_factory}"
        )
    attn_backend = resolve_attn_backend(opts.attn_backend)
    model_config = factory(
        vocab_size=corpus.tokenizer.padded_vocab_size(),
        attn_backend=attn_backend,
    )
    log.info("attention backend: %s", attn_backend)

    data_loader_config = ComposableDataLoaderConfig(
        tokenizer=corpus.tokenizer,
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=8,
        shuffle=True,
        shuffle_strategy=ShuffleStrategy.intra_source,
        work_dir=opts.work_dir,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=SkipStepAdamWConfig(
            lr=opts.learning_rate,
            weight_decay=C.WEIGHT_DECAY,
            betas=C.ADAM_BETAS,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        # param_dtype from --param-dtype so the choice appears in the command text; the
        # platform reads command words and cannot see a dtype set only in code.
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.hsdp,
            param_dtype=DType(opts.param_dtype),
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=C.GRAD_CLIP,
        scheduler=CosWithWarmup(warmup=opts.warmup_steps),
    )

    if opts.smoke:
        milestone_steps: list[int] = []
    else:
        milestone_steps = C.milestone_checkpoint_steps(
            opts.arm,
            save_interval=opts.save_interval,
            proximity=opts.checkpoint_milestone_proximity,
            total_steps=opts.steps,
        )
    log.info(
        "checkpoints: every %d steps + milestones %s (proximity=%d)",
        opts.save_interval,
        milestone_steps,
        opts.checkpoint_milestone_proximity,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            metrics_collect_interval=10 if not opts.smoke else 1,
            cancel_check_interval=10 if not opts.smoke else 1,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                ephemeral_save_interval=None,
                max_checkpoints=None,
                save_async=True,
                fixed_steps=milestone_steps or None,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=f"{opts.run_name}-{opts.arm}" + ("-smoke" if opts.smoke else ""),
                project=os.environ.get("EDULLM_WANDB_PROJECT"),
                # No `group`: the platform puts the experiment in WANDB_RUN_GROUP.
                cancel_check_interval=10 if not opts.smoke else 1,
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    config = ExperimentConfig(
        model=model_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        arm=opts.arm,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
    )
    # Stash corpus on the opts object for train(); avoids a second resolve.
    opts._corpus = corpus  # type: ignore[attr-defined]
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


def _assert_curriculum_covers_steps(phases, opts) -> None:
    """Refuse early if the curriculum cannot supply Duration.steps(opts.steps)."""
    needed = opts.steps * opts.global_batch_size
    have = sum(phase.num_tokens for phase in phases)
    if have < needed:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"curriculum has {have:,} tokens but {opts.steps} steps × "
            f"{opts.global_batch_size} needs {needed:,}. "
            "Usually the published corpus is short of a source budget "
            "(incomplete FineWeb build is the common case).",
        )


def train(config: ExperimentConfig, opts) -> None:
    corpus = opts._corpus
    phases = build_phases(
        opts.arm,
        corpus,
        sequence_length=opts.sequence_length,
        work_dir=opts.work_dir,
    )
    _assert_curriculum_covers_steps(phases, opts)
    if get_rank() == 0:
        rich.print(config)
        for phase in phases:
            phase.visualize()

    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    data_loader = config.data_loader.build(
        *phases,
        work_dir=opts.work_dir,
        dp_process_group=train_module.dp_process_group,
    )
    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    losses = LossWatcher()
    trainer.add_callback("edullm_losses", losses)

    remove_torn_checkpoints(trainer.save_folder)
    trainer.maybe_load_checkpoint()
    started = time.monotonic()
    trainer.fit()
    if get_rank() == 0:
        print(
            json.dumps(
                {
                    "run_id": opts.run_name,
                    "arm": opts.arm,
                    "dataset_id": config.dataset_id,
                    "dataset_version": config.dataset_version,
                    "steps": trainer.global_step,
                    "first_loss": losses.first,
                    "last_loss": losses.last,
                    "seconds": time.monotonic() - started,
                    "checkpoint_uri": opts.save_folder,
                    "wandb_url": losses.wandb_url,
                },
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()
    apply_smoke_overrides(opts)
    require_platform_env(opts)

    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        config = build_config(opts, overrides)

    # Same early refusal as train_on_corpus: before the process group, exit 73 when the
    # merged config asks for bfloat16 on silicon that has none. Library build() also checks.
    try:
        validate_precision_support(config)
    except OLMoConfigurationError as unusable:
        if opts.dry_run:
            log.warning("%s", unusable)
        else:
            raise Refusal(
                Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, str(unusable)
            ) from None

    if opts.dry_run:
        phases = build_phases(
            opts.arm,
            opts._corpus,
            sequence_length=opts.sequence_length,
            work_dir=opts.work_dir,
        )
        _assert_curriculum_covers_steps(phases, opts)
        rich.print(config)
        for phase in phases:
            phase.visualize()
        return

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        prepare_training_environment()
    try:
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
