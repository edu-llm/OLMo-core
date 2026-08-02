"""Train the ``base`` or ``split`` SmolLM2-135M model for the memory-split experiment.

* ``base``  -- standard next-token-prediction loss over every token (the model memorizes facts).
* ``split`` -- identical, except fact-span tokens are excluded from the loss via the corpus's
  ``label_mask``. This isolates reasoning from fact memorization, with **no** query/contrastive
  objective and **no** vocab change. ``base`` and ``split`` see the exact same token stream; only
  the mask differs.

Hyperparameters follow the **control** arm of smollm2-135m-control-vs-colmlm-archive.md (used for
*both* models here): 750M-token FineWeb-Edu prefix, seq len 2,048, peak LR 3e-4, AdamW
betas (0.9, 0.95), weight decay 0.1, cosine schedule with 2%-of-steps warmup, global batch 65,536
tokens/step, seed 42, and a 20B-token budget (~305k steps).

Reads the tokenized corpus either as ``colmlm/prepare_data.py`` writes it (token + ``label_mask``
memmaps under a root ``manifest.json``) or as the eduLLM dataset library publishes it (a sealed
group manifest under ``tokens/``, which carries no masks unless a mask profile was published).
Which layout is present is detected; ``--mask-dir`` is where the masks are when they are not
beside the tokens. Follows the eduLLM ``olmo-core.md`` required settings: checkpoints go to
``--save-folder`` (kept on the command line), ``max_checkpoints=None``, no LM/downstream
evaluators, and an explicit ``max_duration``.

Single GPU / CPU:
    python -m colmlm.train "$RUN" --mode split --data-dir data/tokenized-750m \\
        --save-folder "$EDULLM_CHECKPOINT_DIR"
Multi-GPU:
    python -m torch.distributed.run --nproc-per-node=N --standalone -m colmlm.train "$RUN" \\
        --mode base --data-dir data/tokenized-750m --save-folder "$EDULLM_CHECKPOINT_DIR"
Against a promoted corpus, with masks held elsewhere:
    python -m colmlm.train "$RUN" --mode split \\
        --data-dir s3://edullm-data/pretrain/fineweb-edu-750m/v2 \\
        --mask-dir s3://.../masks --save-folder "$EDULLM_CHECKPOINT_DIR"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, cast

import rich

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyDatasetDType, NumpyFSLDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.io import file_exists, get_bytes_range, get_file_size, list_directory, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
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
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

from colmlm.tokenizer import smollm2_tokenizer_config

# Control-arm defaults (smollm2-135m-control-vs-colmlm-archive.md), used for both base and split.
CONTROL_SEED = 42

# Reasoning eval suite (5-shot rank-classification, accuracy + BPB, every 250M training tokens):
# the archive's HellaSwag / PIQA / OpenBookQA plus CommonsenseQA, Social IQa, and ARC-Easy.
CONTROL_EVAL_TASKS = [
    "arc_easy_test_rc_5shot",
    "csqa_val_rc_5shot",
    "hellaswag_rc_5shot",
    "openbookqa_test_rc_5shot",
    "piqa_val_rc_5shot",
    "socialiqa_val_rc_5shot",
]


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    mode: str = "base"
    data_dir: str = ""
    mask_dir: str = ""
    init_seed: int = CONTROL_SEED


#: The manifest each layout is known by, relative to ``--data-dir``. ``prepare_data.py`` writes one
#: at the corpus root listing an entry per annotate worker; a corpus promoted into the eduLLM
#: dataset library writes a sealed group manifest beside the objects it describes.
NATIVE_MANIFEST = "manifest.json"
EDULLM_DATA_MANIFEST = "tokens/manifest.json"
MASK_SUFFIX = ".mask.bin"


@dataclass
class Corpus:
    """What a manifest says about the token shards, before any of it has been trusted."""

    token_paths: List[str]
    dtype: Optional[str]
    byte_order: Optional[str]
    header_bytes: Optional[int]
    total_tokens: Optional[int]
    manifest: str


def _under(prefix: str, *parts: str) -> str:
    """Join under a local directory or an ``s3://`` prefix, which ``Path`` would mangle."""
    return "/".join([normalize_path(prefix), *parts])


def _read_manifest(path: str) -> Optional[dict]:
    if not file_exists(path):
        return None
    return json.loads(get_bytes_range(path, 0, get_file_size(path)).decode("utf-8"))


def read_corpus(data_dir: str) -> Corpus:
    """Resolve the token shards under ``data_dir``, in whichever of the two layouts wrote it.

    Detected rather than flagged. The two manifests sit at different paths and carry different
    keys, so there is nothing here for a person to remember on the one run where forgetting it
    costs a GPU-day. ``data_dir`` may be a local directory or an ``s3://`` prefix.
    """
    native = _read_manifest(_under(data_dir, NATIVE_MANIFEST))
    if native is not None and "shards" in native:
        return _corpus_from_native(data_dir, native)
    if native is not None and "entries" in native:
        # --data-dir pointed at the group rather than the corpus, which is the easy mistake: the
        # group manifest is called manifest.json too, and its entry paths are written relative to
        # the dataset root, so reading it from here would resolve every shard one level too deep.
        raise SystemExit(
            f"{data_dir} is the '{native.get('group', '?')}' group of an edullm-data corpus and "
            "not the corpus; --data-dir takes the dataset root that its entry paths are relative to"
        )

    group = _read_manifest(_under(data_dir, EDULLM_DATA_MANIFEST))
    if group is not None and "entries" in group:
        return _corpus_from_edullm_data(data_dir, group)

    raise SystemExit(
        f"no corpus manifest under {data_dir}: neither {NATIVE_MANIFEST} with 'shards' "
        f"(prepare_data.py) nor {EDULLM_DATA_MANIFEST} with 'entries' (edullm-data) is there"
    )


def _corpus_from_native(data_dir: str, manifest: dict) -> Corpus:
    """``prepare_data.py``'s own layout: one shard per annotate worker, every one trainable."""
    return Corpus(
        token_paths=[
            _under(data_dir, "tokens", f"train-{shard['worker']:05d}.bin")
            for shard in manifest["shards"]
        ],
        dtype=manifest.get("dtype"),
        byte_order=manifest.get("byte_order"),
        header_bytes=manifest.get("header_bytes"),
        total_tokens=manifest.get("total_tokens"),
        manifest=NATIVE_MANIFEST,
    )


def _corpus_from_edullm_data(data_dir: str, manifest: dict) -> Corpus:
    """The group manifest edullm-data seals: paths as published, and a split per entry.

    Paths are taken as they were sealed rather than rebuilt from an index. The promoted shards are
    named ``train-00000.u16le.bin``, so reconstructing a filename means already knowing the width
    that the file it points at is what declares.
    """
    entries = [entry for entry in manifest["entries"] if entry.get("split") == "train"]
    if not entries:
        splits = sorted({str(entry.get("split")) for entry in manifest["entries"]})
        raise SystemExit(
            f"{EDULLM_DATA_MANIFEST} under {data_dir} has no 'train' entries (splits: "
            f"{', '.join(splits)}); held-out shards are not training data"
        )

    formats = {json.dumps(entry.get("format") or {}, sort_keys=True) for entry in entries}
    if len(formats) > 1:
        raise SystemExit(
            f"{EDULLM_DATA_MANIFEST} under {data_dir} declares {len(formats)} different shard "
            "formats; one corpus is read at one width, byte order and offset"
        )
    fmt = entries[0].get("format") or {}

    counts = [entry.get("count") or {} for entry in entries]
    return Corpus(
        token_paths=[_under(data_dir, entry["path"]) for entry in entries],
        dtype=fmt.get("dtype"),
        byte_order=fmt.get("byte_order"),
        header_bytes=fmt.get("header_bytes"),
        total_tokens=(
            sum(count["value"] for count in counts)
            if all(count.get("unit") == "tokens" for count in counts)
            else None
        ),
        manifest=EDULLM_DATA_MANIFEST,
    )


def _mask_path(mask_dir: str, token_path: str) -> str:
    """The mask that pairs with one token shard, named after the shard rather than its position.

    ``train-00000.bin`` and ``train-00000.u16le.bin`` are the same shard under the two layouts, so
    the name is taken up to its first suffix and ``prepare_data.py``'s ``.mask.bin`` put back.
    """
    stem = normalize_path(token_path).rsplit("/", 1)[-1].split(".", 1)[0]
    return _under(mask_dir, stem + MASK_SUFFIX)


def resolve_masks(mask_dir: str, token_paths: List[str]) -> List[str]:
    """One mask per token shard, or refuse to start.

    A ``split`` run that cannot find its masks is not a degraded run. Every fact token stays in the
    loss, the arm becomes a second ``base``, and the only trace is that the two curves agree --
    which is a result the experiment would go on to report. A promoted corpus publishes no
    ``masks/`` group at all, so arriving here with none is the ordinary case rather than a corner.
    """
    try:
        published = [
            path
            for path in list_directory(mask_dir, include_dirs=False)
            if path.endswith(MASK_SUFFIX)
        ]
    except FileNotFoundError:
        published = []
    if not published:
        raise SystemExit(
            f"--mode split needs one {MASK_SUFFIX} per token shard and {mask_dir} holds none. "
            "A corpus promoted into the eduLLM dataset library carries tokens only; point "
            "--mask-dir at the masks prepare_data.py wrote for the same build."
        )
    if len(published) != len(token_paths):
        raise SystemExit(
            f"{mask_dir} holds {len(published)} masks and this corpus has {len(token_paths)} token "
            f"shards. Masks from another build mask the wrong spans, which reads as a slightly "
            "worse split arm and not as an error."
        )

    paths = [_mask_path(mask_dir, token_path) for token_path in token_paths]
    missing = [path for path in paths if not file_exists(path)]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(paths)} masks are not under {mask_dir}, starting with "
            f"{missing[0]}; the {len(published)} masks there are named for other shards"
        )
    return paths


def build_config(opts) -> ExperimentConfig:
    if opts.mode not in ("base", "split"):
        raise SystemExit(f"--mode must be 'base' or 'split', got {opts.mode!r}")

    corpus = read_corpus(opts.data_dir)
    if corpus.dtype != "uint16":
        raise SystemExit(f"expected uint16 corpus, manifest says {corpus.dtype!r}")
    # The other two ways a corpus decodes into in-range ids and a merely-worse loss curve. Both
    # manifests declare them; see .edullm/train_on_corpus.py for what each costs.
    if corpus.header_bytes:
        raise SystemExit(
            f"{corpus.manifest} declares {corpus.header_bytes} header bytes and OLMo-core memmaps "
            "from offset zero, so the header would be read as tokens"
        )
    if corpus.byte_order is not None and corpus.byte_order != sys.byteorder:
        raise SystemExit(
            f"{corpus.manifest} says {corpus.byte_order}-endian and this host is {sys.byteorder}-"
            "endian; numpy would read every token to a different, in-range-looking id"
        )

    mask_dir = opts.mask_dir or _under(opts.data_dir, "masks")
    mask_paths = resolve_masks(mask_dir, corpus.token_paths) if opts.mode == "split" else None

    # Resolve the training-step budget from tokens / epochs / steps (control cap: 20B tokens).
    gbs = opts.global_batch_size
    if opts.steps is not None:
        total_steps = opts.steps
    elif opts.epochs is not None:
        if corpus.total_tokens is None:
            raise SystemExit(f"--epochs needs a token count and {corpus.manifest} declares none")
        total_steps = opts.epochs * math.ceil(corpus.total_tokens / gbs)
    else:
        total_steps = math.ceil(opts.budget_tokens / gbs)
    warmup = opts.warmup_steps if opts.warmup_steps is not None else round(opts.warmup_ratio * total_steps)
    eval_interval_steps = max(1, round(opts.eval_interval_tokens / gbs))  # archive: every 250M tokens
    wandb_project = (
        opts.wandb_project or os.environ.get("WANDB_PROJECT") or os.environ.get("EDULLM_WANDB_PROJECT")
    )

    tokenizer = smollm2_tokenizer_config()
    model_config = TransformerConfig.smollm2_135M(vocab_size=tokenizer.vocab_size)

    dataset_config = NumpyFSLDatasetConfig(
        paths=corpus.token_paths,
        sequence_length=opts.sequence_length,
        tokenizer=tokenizer,
        dtype=NumpyDatasetDType.uint16,  # explicit; never inferred (see olmo-core.md)
        work_dir=opts.work_dir,
        # The one line that distinguishes split from base: fact tokens -> -100 -> excluded from NTP.
        # None for base; for split, resolve_masks has already refused an incomplete mask set.
        label_mask_paths=mask_paths,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(lr=opts.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1),
        compile_model=opts.compile,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=warmup),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(total_steps),
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
                project=wandb_project,
                entity=opts.wandb_entity or os.environ.get("WANDB_ENTITY"),
                # None -> the W&B client reads WANDB_RUN_GROUP (set by the eduLLM platform) itself.
                group=opts.wandb_group,
                tags=["smollm2-135m", f"mode:{opts.mode}", *opts.wandb_tags],
                config={
                    "mode": opts.mode,
                    "learning_rate": opts.learning_rate,
                    "sequence_length": opts.sequence_length,
                    "global_batch_tokens": gbs,
                    "budget_tokens": opts.budget_tokens,
                    "total_steps": total_steps,
                    "warmup_steps": warmup,
                    "weight_decay": 0.1,
                    "corpus": opts.data_dir,
                    "corpus_manifest": corpus.manifest,
                    "masks": mask_dir if opts.mode == "split" else None,
                    "eval_tasks": list(opts.eval_tasks) if opts.eval else [],
                },
                cancel_check_interval=10,
                enabled=(wandb_project is not None) and not opts.no_wandb,
            ),
        )
        .with_callback(
            "downstream_evaluator",
            # In-loop eval matching the archive. Requires ai2-olmo-core[eval] (ai2-olmo-eval);
            # the stock eduLLM platform image does NOT install it, so pass --no-eval there.
            DownstreamEvaluatorCallbackConfig(
                tasks=list(opts.eval_tasks),
                tokenizer=tokenizer,
                eval_interval=eval_interval_steps,
                eval_on_finish=True,
                enabled=opts.eval,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )
    # No lm_evaluator: its C4 validation shard index 404s on this platform (see olmo-core.md).

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        mode=opts.mode,
        data_dir=opts.data_dir,
        mask_dir=mask_dir if opts.mode == "split" else "",
    )
    return config.merge(opts.overrides)


def train(config: ExperimentConfig) -> None:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    trainer.maybe_load_checkpoint()
    trainer.fit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colmlm.train", description="Train base/split SmolLM2-135M.")
    p.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    p.add_argument("--mode", choices=["base", "split"], required=True)
    p.add_argument("--data-dir", default=os.environ.get("COLMLM_DATA_DIR", "data/tokenized-750m"))
    p.add_argument(
        "--mask-dir", default=os.environ.get("COLMLM_MASK_DIR"),
        help="Where --mode split reads the label masks; local path or s3:// prefix. Defaults to "
        "{data-dir}/masks. A corpus promoted into the eduLLM dataset library publishes tokens "
        "only, so a split run against one has to be told where its masks are.",
    )
    p.add_argument("--save-folder", default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""))
    p.add_argument("--work-dir", default="/tmp/colmlm-dataset-cache")
    # Control hyperparameters (both models); see smollm2-135m-control-vs-colmlm-archive.md.
    p.add_argument("--sequence-length", type=int, default=2048)
    p.add_argument(
        "--budget-tokens", type=int, default=20_000_000_000,
        help="Training-token budget (control cap: 20B ~= 305k steps ~= 26.6 epochs of 750M).",
    )
    p.add_argument("--epochs", type=int, default=None, help="Train N epochs over the corpus (overrides --budget-tokens).")
    p.add_argument("--steps", type=int, default=None, help="Absolute step budget (overrides tokens/epochs).")
    p.add_argument("--save-interval", type=int, default=3814, help="~250M tokens at 65,536/step.")
    p.add_argument("--warmup-ratio", type=float, default=0.02, help="Warmup as a fraction of --steps.")
    p.add_argument("--warmup-steps", type=int, default=None, help="Absolute warmup; overrides ratio.")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--global-batch-size", type=int, default=32 * 2048, help="Tokens/step (32 x 2048).")
    p.add_argument("--rank-microbatch-size", type=int, default=8 * 2048, help="Tokens/microbatch/rank.")
    p.add_argument("--data-seed", type=int, default=CONTROL_SEED)
    p.add_argument(
        "--eval", action=argparse.BooleanOptionalAction, default=True,
        help="In-loop downstream eval (HellaSwag/PIQA/OpenBookQA); needs ai2-olmo-core[eval]. "
        "Use --no-eval on the stock eduLLM image, which lacks ai2-olmo-eval.",
    )
    p.add_argument("--eval-tasks", nargs="*", default=CONTROL_EVAL_TASKS,
                   help="olmo_eval task ids (5-shot RC; report accuracy + BPB).")
    p.add_argument("--eval-interval-tokens", type=int, default=250_000_000,
                   help="Run eval every N training tokens (archive: 250M).")
    # Weights & Biases (AWS monitoring). Project/entity/group also read from WANDB_* env vars.
    p.add_argument("--wandb-project", default=None, help="W&B project (else $WANDB_PROJECT/$EDULLM_WANDB_PROJECT).")
    p.add_argument("--wandb-entity", default=None, help="W&B entity (else $WANDB_ENTITY).")
    p.add_argument("--wandb-group", default=None, help="W&B group (else $WANDB_RUN_GROUP; e.g. memory-split-135m).")
    p.add_argument("--wandb-tags", nargs="*", default=[], help="Extra W&B tags (mode is added automatically).")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile (needs a C compiler).")
    p.add_argument("--dry-run", action="store_true", help="Build and print the config, train nothing.")
    return p


def main() -> None:
    opts, overrides = build_parser().parse_known_args()
    opts.overrides = overrides
    config = build_config(opts)
    if opts.dry_run:
        rich.print(config)
        md = config.trainer.max_duration
        ev = config.trainer.callbacks.get("downstream_evaluator")
        rich.print(
            f"[bold green]mode={config.mode}[/] "
            f"params={config.model.num_params:,} "
            f"label_mask={'on' if config.dataset.label_mask_paths else 'off'} "
            f"shards={len(config.dataset.paths)} "
            f"max_duration={md.value:,} {md.unit} "
            f"eval={'on' if (ev is not None and ev.enabled) else 'off'} "
            f"eval_interval={getattr(ev, 'eval_interval', None)} tasks={getattr(ev, 'tasks', None)}"
        )
        return
    prepare_training_environment()
    try:
        train(config)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
