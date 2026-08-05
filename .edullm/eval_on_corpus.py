"""Evaluate a checkpoint on the held-out split of a published eduLLM corpus.

    python .edullm/eval_on_corpus.py "$EDULLM_RUN_ID" \\
        --checkpoint s3://.../step7629/ \\
        --model-factory olmo2_100M \\
        --save-folder "$EDULLM_CHECKPOINT_DIR"

Loads the sealed ``val`` shards (never train), runs CE/PPL via OLMo-core's
``LMEvaluator``, then writes ``eval_summary.json`` under ``$EDULLM_OUTPUT_PREFIX``
(falling back to ``--save-folder`` / cwd). Cancels after the first eval so no
training steps run.

Per-language labels are taken from the path segment under ``tokens/``
(e.g. ``.../tokens/eng_Latn/val-00000.u32le.bin`` → ``eng_Latn``).

BPB here is bits-per-token (``CE / ln(2)``), not bits-per-byte: the shards are
pre-tokenized and decoding every id for UTF-8 byte length is a separate path.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, cast
from urllib.parse import urlparse

# Sibling helpers from the training entrypoint (same directory in the image).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_on_corpus import (  # noqa: E402
    ExperimentConfig,
    Refusal,
    Stage,
    during,
    leave_the_reason_in_wandb,
    read_failure,
    resolve_corpus,
)

from olmo_core.config import DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.callbacks.evaluator_callback import LMEvaluatorCallbackConfig
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


def lang_label(path: str) -> str:
    """``.../tokens/<lang>/val-*.u32le.bin`` → ``<lang>``."""
    parts = path.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if part == "tokens" and i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def resolve_val_corpus(*, dataset_id: str, version: str, tokenizer_id: str):
    """Like ``resolve_corpus`` but only the held-out split."""
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3
    from train_on_corpus import corpus_from_manifest

    s3 = Boto3S3.default()
    if version in ("", "latest"):
        try:
            resolved = resolve_latest(dataset_id, s3=s3)
        except Refusal:
            raise
        except BaseException as exc:
            raise Refusal(read_failure(exc), f"{type(exc).__name__}: {exc}") from exc
        if resolved is None:
            raise Refusal(
                Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS,
                f"no published version of {dataset_id}",
            )
        version = resolved

    try:
        read = dataset_paths(dataset_id, version, s3=s3, split="val")
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(
            read_failure(exc),
            f"reading val of {dataset_id}/{version}: {type(exc).__name__}: {exc}",
        ) from exc

    if not read.paths:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} has no val shards",
        )
    return corpus_from_manifest(
        read, dataset_id=dataset_id, version=version, tokenizer_id=tokenizer_id
    )


def output_dir(opts) -> str:
    prefix = os.environ.get("EDULLM_OUTPUT_PREFIX", "").rstrip("/")
    if prefix:
        return prefix
    if opts.save_folder:
        return opts.save_folder.rstrip("/")
    return os.getcwd()


def write_json(uri_or_path: str, payload: Dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if uri_or_path.startswith("s3://"):
        import boto3

        parsed = urlparse(uri_or_path)
        boto3.client("s3").put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=text.encode("utf-8"),
            ContentType="application/json",
        )
    else:
        os.makedirs(os.path.dirname(uri_or_path) or ".", exist_ok=True)
        with open(uri_or_path, "w", encoding="utf-8") as fh:
            fh.write(text)


def metrics_from_evaluators(trainer) -> Dict[str, float]:
    """Read CE/PPL left on the LM evaluators after ``perform_eval``."""
    out: Dict[str, float] = {}
    cb = trainer.callbacks.get("lm_evaluator")
    if cb is None:
        return out
    for evaluator in getattr(cb, "evaluators", []):
        try:
            metrics = evaluator.compute_metrics()
        except Exception as exc:  # noqa: BLE001
            log.warning("compute_metrics failed: %s", exc)
            continue
        for name, value in metrics.items():
            try:
                out[f"eval/lm/{name}"] = float(value.item() if hasattr(value, "item") else value)
            except (TypeError, ValueError):
                continue
    return out


def summarise_metrics(
    *,
    raw: Dict[str, float],
    dataset_id: str,
    dataset_version: str,
    tokenizer_id: str,
    checkpoint: str,
    model_factory: str,
    seconds: float,
) -> Dict[str, Any]:
    """Turn ``eval/lm/<lang>/CE loss`` keys into a stable summary schema."""
    by_lang: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        # eval/lm/eng_Latn/CE loss  or  eval/lm/eng_Latn/PPL
        parts = key.split("/")
        if len(parts) < 4:
            continue
        lang, metric = parts[2], parts[3]
        slot = by_lang.setdefault(lang, {})
        if metric == "CE loss":
            slot["ce_nats"] = value
            slot["bits_per_token"] = value / math.log(2.0)
        elif metric == "PPL":
            slot["ppl"] = value

    # Micro-average over languages that have CE (equal weight per lang for the table).
    ces = [v["ce_nats"] for v in by_lang.values() if "ce_nats" in v]
    micro = {
        "ce_nats": sum(ces) / len(ces) if ces else float("nan"),
        "bits_per_token": (sum(ces) / len(ces) / math.log(2.0)) if ces else float("nan"),
        "ppl": math.exp(sum(ces) / len(ces)) if ces else float("nan"),
        "n_languages": len(ces),
    }

    return {
        "schema_version": 1,
        "kind": "phase0_val_lm_eval",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "tokenizer_id": tokenizer_id,
        "checkpoint": checkpoint,
        "model_factory": model_factory,
        "seconds": seconds,
        "languages": {k: by_lang[k] for k in sorted(by_lang)},
        "micro_average": micro,
        "raw_metrics": raw,
        "notes": (
            "bits_per_token is CE/ln(2) (nats→bits). Not bits-per-byte; shards are "
            "pre-tokenized without paired UTF-8 lengths in this path."
        ),
    }


def build_config(opts, overrides: List[str]):
    train_corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    val_corpus = resolve_val_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    log.info(
        "eval %s/%s: %d train shards (unused for steps), %d val shards, dtype %s",
        val_corpus.dataset_id,
        val_corpus.version,
        len(train_corpus.paths),
        len(val_corpus.paths),
        val_corpus.dtype,
    )

    factory = getattr(TransformerConfig, opts.model_factory, None)
    if factory is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {opts.model_factory}"
        )

    model_config = factory(vocab_size=val_corpus.tokenizer.padded_vocab_size())

    # Trainer still needs a train dataset to construct; we cancel before any step.
    dataset_config = NumpyFSLDatasetConfig(
        paths=train_corpus.paths,
        sequence_length=opts.sequence_length,
        tokenizer=val_corpus.tokenizer,
        dtype=train_corpus.dtype,
        work_dir=opts.work_dir,
    )

    eval_metadata = [{"label": lang_label(p)} for p in val_corpus.paths]
    eval_dataset = NumpyPaddedFSLDatasetConfig(
        paths=val_corpus.paths,
        metadata=eval_metadata,
        sequence_length=opts.sequence_length,
        tokenizer=val_corpus.tokenizer,
        dtype=val_corpus.dtype,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=2,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=1e-3,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=1),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            save_overwrite=False,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            # Safety: even if cancel fails, do not train the full corpus.
            max_duration=Duration.steps(1),
            load_path=opts.checkpoint,
            load_optim_state=False,
            load_trainer_state=False,
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=eval_dataset,
                eval_interval=None,
                eval_on_startup=True,
                cancel_after_first_eval=True,
                eval_duration=Duration.epochs(1),
                log_interval=10,
            ),
        )
    )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=val_corpus.dataset_id,
        dataset_version=val_corpus.version,
    )
    return config.merge(overrides)


def run_eval(config: ExperimentConfig, opts) -> Dict[str, Any]:
    seed_all(config.init_seed)
    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    if "config_saver" in trainer.callbacks:
        cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = (
            config.as_config_dict()
        )

    started = time.monotonic()
    # load_path on TrainerConfig loads during fit startup; explicit call covers both orders.
    if opts.checkpoint:
        trainer.maybe_load_checkpoint(opts.checkpoint)
    trainer.fit()
    seconds = time.monotonic() - started
    barrier()

    raw = metrics_from_evaluators(trainer)
    if not raw:
        log.warning("no eval metrics on evaluators after fit(); summary will be empty")

    return summarise_metrics(
        raw=raw,
        dataset_id=opts.dataset_id,
        dataset_version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
        checkpoint=opts.checkpoint,
        model_factory=opts.model_factory,
        seconds=seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_on_corpus",
        description="Evaluate a checkpoint on sealed val shards of a published corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    parser.add_argument("--dataset-id", default=os.environ.get("EDULLM_DATASET_ID", ""))
    parser.add_argument("--dataset-version", default=os.environ.get("EDULLM_DATASET_VERSION", ""))
    parser.add_argument(
        "--dataset-tokenizer", default=os.environ.get("EDULLM_DATASET_TOKENIZER", "")
    )
    parser.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        help="Scratch prefix for trainer bookkeeping (not the checkpoint under test).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="s3:// or local path to a complete stepN/ checkpoint directory.",
    )
    parser.add_argument("--work-dir", default="/tmp/dataset-cache")
    parser.add_argument("--model-factory", default="olmo2_100M")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=256 * 1024)
    parser.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    parser.add_argument("--data-seed", type=int, default=0)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()

    missing = [
        name
        for name, value in (
            ("EDULLM_DATASET_ID", opts.dataset_id),
            ("EDULLM_DATASET_VERSION", opts.dataset_version),
            ("EDULLM_DATASET_TOKENIZER", opts.dataset_tokenizer),
            ("EDULLM_CHECKPOINT_DIR / --save-folder", opts.save_folder),
            ("--checkpoint", opts.checkpoint),
        )
        if not value
    ]
    if missing:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "unset: " + ", ".join(missing),
        )

    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        config = build_config(opts, overrides)

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        prepare_training_environment()
    try:
        with during(Stage.TRAINING_ITSELF_FAILED):
            summary = run_eval(config, opts)
        if get_rank() == 0:
            dest = f"{output_dir(opts).rstrip('/')}/eval_summary.json"
            write_json(dest, summary)
            log.info("wrote %s", dest)
            print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        teardown_training_environment()


def cli() -> int:
    try:
        main()
        return 0
    except Refusal as refusal:
        print(f"REFUSED [{refusal.stage.name}]: {refusal.explanation}", file=sys.stderr)
        leave_the_reason_in_wandb(
            run_name=os.environ.get("EDULLM_RUN_ID", "local"),
            stage=refusal.stage,
            explanation=refusal.explanation,
        )
        return int(refusal.stage)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return code if code is not None else 1
    except BaseException as exc:  # noqa: BLE001
        print(f"UNCAUGHT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return int(Stage.TRAINING_ITSELF_FAILED)


if __name__ == "__main__":
    raise SystemExit(cli())
