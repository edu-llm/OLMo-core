"""Evaluate a checkpoint for true UTF-8 bits-per-byte on FLORES-200 devtest.

Plan B languages only. Downloads the ungated FLORES-200 tarball, loads the
published gigatoken ``tokenizer.json`` from ``s3://edullm-data``, encodes each
language into a contiguous ``.u32le.bin`` shard (no EOS, matching Plan B packing),
runs OLMo-core ``LMEvaluator``, then reports per-language and micro-average BPB:

    bits_per_byte = bits_per_token / (utf8_bytes / n_tokens)

where ``bits_per_token = CE / ln(2)`` and ``utf8_bytes`` / ``n_tokens`` are measured
on the same FLORES text that was encoded.

    torchrun --nproc-per-node=8 .edullm/eval_flores_bpb.py "$EDULLM_RUN_ID" \\
        --checkpoint s3://.../step80200/ \\
        --model-factory olmo2_1B_v2 \\
        --model.n_layers=12 \\
        --save-folder "$EDULLM_CHECKPOINT_DIR"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_on_corpus import (  # noqa: E402
    lang_label,
    metrics_from_evaluators,
    output_dir,
    write_chunked_instance_indices,
    write_json,
)
from train_on_corpus import (  # noqa: E402
    ExperimentConfig,
    Refusal,
    Stage,
    during,
    leave_the_reason_in_wandb,
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
from olmo_core.io import copy_file
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

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
PLAN_B_LANGS = (
    "eng_Latn",
    "hun_Latn",
    "zho_Hans",
    "hin_Deva",
    "swh_Latn",
    "hat_Latn",
)
TOKENIZER_URI = {
    "tokenizer/gigatoken-bpe": "s3://edullm-data/tokenizer/gigatoken-bpe/v1/tokenizer.json",
    "tokenizer/gigatoken-superbpe": "s3://edullm-data/tokenizer/gigatoken-superbpe/v1/tokenizer.json",
}


def ensure_flores(root: Path) -> Path:
    """Download and extract FLORES-200 if ``root/devtest`` is missing."""
    split_dir = root / "devtest"
    if split_dir.is_dir() and any(split_dir.iterdir()):
        return root
    root.mkdir(parents=True, exist_ok=True)
    tarball = root / "flores200_dataset.tar.gz"
    if not tarball.is_file():
        log.info("downloading FLORES-200 from %s", FLORES_URL)
        urllib.request.urlretrieve(FLORES_URL, tarball)
    log.info("extracting %s", tarball)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(path=root.parent)
    # Tarball extracts to flores200_dataset/ beside or under root.
    candidates = [
        root,
        root.parent / "flores200_dataset",
        Path("flores200_dataset"),
    ]
    for cand in candidates:
        if (cand / "devtest").is_dir():
            return cand
    raise Refusal(
        Stage.THE_CONFIG_WOULD_NOT_BUILD,
        f"FLORES extract finished but no devtest/ under {root}",
    )


def load_flores_lines(flores_root: Path, lang: str, split: str = "devtest") -> List[str]:
    path = flores_root / split / f"{lang}.{split}"
    if not path.is_file():
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"missing FLORES file {path}",
        )
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, f"empty FLORES file {path}")
    return lines


def fetch_tokenizer_json(tokenizer_id: str, dest: Path) -> Path:
    uri = TOKENIZER_URI.get(tokenizer_id)
    if uri is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"no FLORES tokenizer URI mapped for {tokenizer_id!r}",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        log.info("copying %s → %s", uri, dest)
        copy_file(uri, str(dest), save_overwrite=True, quiet=True)
    return dest


def encode_language(
    *,
    lines: Sequence[str],
    encode,
) -> Tuple[np.ndarray, int, int]:
    """Return uint32 ids, utf-8 byte count, and token count for one language."""
    ids: List[int] = []
    utf8_bytes = 0
    for line in lines:
        raw = line.encode("utf-8")
        utf8_bytes += len(raw)
        # Sentence-separated by newline so boundaries match FLORES lines.
        piece = encode(line)
        if not piece:
            continue
        ids.extend(piece)
    if not ids:
        raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, "encoder produced zero tokens")
    arr = np.asarray(ids, dtype=np.uint32)
    return arr, utf8_bytes, int(arr.size)


def materialize_flores_shards(
    *,
    work_dir: str,
    tokenizer_id: str,
    langs: Sequence[str],
) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Write per-language FLORES shards; return local paths and byte/token stats."""
    work = Path(work_dir)
    flores_root = ensure_flores(work / "flores200_dataset")
    tok_path = fetch_tokenizer_json(tokenizer_id, work / "tokenizer" / "tokenizer.json")

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tok_path))

    def encode(text: str) -> List[int]:
        return list(tok.encode(text).ids)

    paths: List[str] = []
    stats: Dict[str, Dict[str, int]] = {}
    for lang in langs:
        lines = load_flores_lines(flores_root, lang)
        arr, utf8_bytes, n_tokens = encode_language(lines=lines, encode=encode)
        out_dir = work / "flores_shards" / "tokens" / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "val-00000.u32le.bin"
        arr.tofile(out_path)
        paths.append(str(out_path))
        stats[lang] = {
            "utf8_bytes": utf8_bytes,
            "n_tokens": n_tokens,
            "n_lines": len(lines),
        }
        log.info(
            "FLORES %s: %d lines, %d tokens, %d utf-8 bytes (%.3f BPT)",
            lang,
            len(lines),
            n_tokens,
            utf8_bytes,
            utf8_bytes / n_tokens,
        )
    return paths, stats


def summarise_flores(
    *,
    raw: Dict[str, float],
    byte_stats: Dict[str, Dict[str, int]],
    dataset_id: str,
    dataset_version: str,
    tokenizer_id: str,
    checkpoint: str,
    model_factory: str,
    seconds: float,
    langs: Sequence[str],
) -> Dict[str, Any]:
    by_lang: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
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

    for lang, st in byte_stats.items():
        slot = by_lang.setdefault(lang, {})
        slot["utf8_bytes"] = float(st["utf8_bytes"])
        slot["n_tokens"] = float(st["n_tokens"])
        slot["n_lines"] = float(st["n_lines"])
        slot["bytes_per_token"] = st["utf8_bytes"] / st["n_tokens"]
        if "bits_per_token" in slot:
            slot["bits_per_byte"] = slot["bits_per_token"] / slot["bytes_per_token"]

    # Equal-weight micro-average over Plan B langs that have BPB.
    bpbs = [by_lang[l]["bits_per_byte"] for l in langs if "bits_per_byte" in by_lang.get(l, {})]
    non_cjk = [
        by_lang[l]["bits_per_byte"]
        for l in langs
        if l != "zho_Hans" and "bits_per_byte" in by_lang.get(l, {})
    ]
    micro = {
        "bits_per_byte": sum(bpbs) / len(bpbs) if bpbs else float("nan"),
        "bits_per_byte_non_cjk": sum(non_cjk) / len(non_cjk) if non_cjk else float("nan"),
        "n_languages": len(bpbs),
    }
    return {
        "schema_version": 1,
        "kind": "flores200_devtest_lm_bpb",
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "tokenizer_id": tokenizer_id,
        "checkpoint": checkpoint,
        "model_factory": model_factory,
        "seconds": seconds,
        "languages": {k: by_lang[k] for k in sorted(by_lang)},
        "micro_average": micro,
        "raw_metrics": raw,
        "byte_stats": byte_stats,
        "notes": (
            "True UTF-8 bits-per-byte on FLORES-200 devtest: "
            "BPB = (CE/ln2) / (utf8_bytes/n_tokens), with utf8_bytes and n_tokens "
            "measured on the encoded FLORES lines. Val packing matches Plan B "
            "(contiguous ids, no EOS). CE is mean over non-pad next-token losses."
        ),
    }


def build_config(opts, overrides: List[str], flores_paths: List[str]):
    train_corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    # Tokenizer / dtype come from the sealed training corpus; eval paths are FLORES.
    factory = getattr(TransformerConfig, opts.model_factory, None)
    if factory is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {opts.model_factory}"
        )
    model_config = factory(vocab_size=train_corpus.tokenizer.padded_vocab_size())

    dataset_config = NumpyFSLDatasetConfig(
        paths=train_corpus.paths[:1],  # unused; cancel before steps
        sequence_length=opts.sequence_length,
        tokenizer=train_corpus.tokenizer,
        dtype=train_corpus.dtype,
        work_dir=opts.work_dir,
    )

    eval_metadata = [{"label": lang_label(p)} for p in flores_paths]
    write_chunked_instance_indices(
        flores_paths,
        work_dir=opts.work_dir,
        sequence_length=opts.sequence_length,
        dtype=train_corpus.dtype,
    )
    eval_dataset = NumpyPaddedFSLDatasetConfig(
        paths=flores_paths,
        metadata=eval_metadata,
        sequence_length=opts.sequence_length,
        tokenizer=train_corpus.tokenizer,
        dtype=train_corpus.dtype,
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
        dataset_id=train_corpus.dataset_id,
        dataset_version=train_corpus.version,
    )
    return config.merge(overrides)


def run_eval(config: ExperimentConfig, opts, byte_stats: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
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
    if opts.checkpoint:
        trainer.maybe_load_checkpoint(opts.checkpoint)
    trainer.fit()
    seconds = time.monotonic() - started
    barrier()

    raw = metrics_from_evaluators(trainer)
    if not raw:
        log.warning("no eval metrics on evaluators after fit(); summary will be empty")

    return summarise_flores(
        raw=raw,
        byte_stats=byte_stats,
        dataset_id=opts.dataset_id,
        dataset_version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
        checkpoint=opts.checkpoint,
        model_factory=opts.model_factory,
        seconds=seconds,
        langs=PLAN_B_LANGS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_flores_bpb",
        description="FLORES-200 devtest LM bits-per-byte for Plan B checkpoints.",
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
    parser.add_argument("--model-factory", default="olmo2_1B_v2")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=16384)
    parser.add_argument("--rank-microbatch-size", type=int, default=2048)
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

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        prepare_training_environment()
    try:
        with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
            # Single-node 8×GPU: shared local disk; every rank can read what rank 0 writes.
            if get_rank() == 0:
                flores_paths, byte_stats = materialize_flores_shards(
                    work_dir=opts.work_dir,
                    tokenizer_id=opts.dataset_tokenizer,
                    langs=PLAN_B_LANGS,
                )
                write_json(
                    f"{opts.work_dir.rstrip('/')}/flores_byte_stats.json",
                    {"paths": flores_paths, "byte_stats": byte_stats},
                )
            barrier()
            meta = json.loads(
                Path(opts.work_dir).joinpath("flores_byte_stats.json").read_text(encoding="utf-8")
            )
            flores_paths = list(meta["paths"])
            byte_stats = dict(meta["byte_stats"])
            config = build_config(opts, overrides, flores_paths)

        with during(Stage.TRAINING_ITSELF_FAILED):
            summary = run_eval(config, opts, byte_stats)
        if get_rank() == 0:
            dest = f"{output_dir(opts).rstrip('/')}/flores_bpb_summary.json"
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
