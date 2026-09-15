#!/usr/bin/env python3
"""Dump the eval item text ai2-olmo actually scores (methodology v4 section 4a).

The contamination scan has to search the corpus for the same strings the
evaluator scores. Re-deriving them from HuggingFace fields does not work:
ai2-olmo *assembles* what it scores. HellaSwag's context is activity label +
ctx_a + capitalized ctx_b; MMLU's four subject groups are cut from the `all`
config so positional order is ai2-olmo's rather than HF row order; and for
WinoGrande it is genuinely unclear whether the scored continuation is the
option word or the sentence suffix. An index built from the wrong strings
produces a plausible-looking contamination rate for an item set the endpoint
does not use.

So this reads the strings out of ai2-olmo's own task objects, via the same
`build_evaluator` path the real evaluator uses. No model weights are needed --
`build_evaluator` takes (train_config, evaluator_config, tokenizer, device).

Two independent renderings of each gold are emitted and compared:

  doc_to_continuations(doc)[doc_to_label(doc)]   the task's own string
  tokenizer.decode(sample["continuation"])       what the model is scored on

They should agree after normalization. Where they do not, that is the
WinoGrande/HellaSwag ambiguity showing itself, and the disagreement is
reported per label rather than silently resolved in favor of one.

Output: gzipped JSONL, one row per (label, doc_id), with sha256 of the
normalized context and gold so section 4b can assert the training-time
evaluator saw the same items.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

os.environ.setdefault("WANDB_DISABLED", "1")
os.environ.setdefault("WANDB_MODE", "disabled")
# The HF cache on scratch is fully populated; never reach the network.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from olmo.config import (  # noqa: E402
    EvaluatorConfig,
    EvaluatorType,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
)
from olmo.eval import build_evaluator  # noqa: E402
from olmo.tokenizer import Tokenizer  # noqa: E402

_EDULLM = Path(__file__).resolve().parents[1]
if str(_EDULLM) not in sys.path:
    sys.path.insert(0, str(_EDULLM))
from production_contract.task_loss import TASK_LOSS_RAW_LABELS  # noqa: E402

log = logging.getLogger("dump_eval_items")

# Must match the scanner's normalizer exactly. Hashed into the run
# fingerprint, per v4 section 6.2 -- a divergence here silently zeroes the
# whole measurement, so it lives in one place.
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_config(tokenizer_id: str) -> TrainConfig:
    """Minimal TrainConfig. Only the tokenizer and sequence length matter here.

    max_sequence_length is pinned to 2048 rather than left to ai2-olmo's
    default of 1024, which silently truncates 5-shot context at half the
    model's trained length (audit finding 1e). It affects which items get
    truncated, so it must match the real evaluator.
    """
    config = TrainConfig.new()
    config.model = ModelConfig(max_sequence_length=2048)
    config.tokenizer = TokenizerConfig(identifier=tokenizer_id)
    config.device_eval_batch_size = 4
    config.seed = 42
    config.evaluators = []
    return config


def dump_label(config: TrainConfig, tokenizer: Tokenizer, label: str) -> tuple[list[dict], dict]:
    """Return one row per doc_id for `label`, plus a per-label summary."""
    evaluator = build_evaluator(
        config,
        EvaluatorConfig(label=label, type=EvaluatorType.downstream, device_eval_batch_size=4),
        tokenizer,
        torch.device("cpu"),
    )
    task = evaluator.eval_loader.dataset
    docs = task.dataset

    # The gold continuation per doc, as the task itself renders it.
    rows: list[dict] = []
    decode_mismatch = 0
    label_id_missing = 0

    # sample["continuation"] is token ids; group samples by doc_id so the
    # gold can be cross-checked against the tokenized form the model scores.
    by_doc: dict[int, dict[int, list[int]]] = {}
    for sample in task.samples:
        by_doc.setdefault(int(sample["doc_id"]), {})[int(sample["cont_id"])] = sample[
            "continuation"
        ]

    for doc_id, doc in enumerate(docs):
        try:
            context = task.doc_to_text(doc)
            conts = task.doc_to_continuations(doc)
            gold_idx = task.doc_to_label(doc)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{label}: doc {doc_id} could not be rendered: {exc}") from exc
        if not isinstance(gold_idx, int) or not 0 <= gold_idx < len(conts):
            label_id_missing += 1
            continue
        gold = conts[gold_idx]

        norm_ctx = normalize(str(context))
        norm_gold = normalize(str(gold))

        # Cross-check against the tokenized continuation actually scored.
        decoded = None
        cont_ids = by_doc.get(doc_id, {}).get(gold_idx)
        if cont_ids is not None:
            decoded = normalize(tokenizer.decode(list(cont_ids)))
            if decoded != norm_gold:
                decode_mismatch += 1

        rows.append(
            {
                "label": label,
                "doc_id": doc_id,
                "context": norm_ctx,
                "gold": norm_gold,
                "context_sha256": sha256(norm_ctx),
                "gold_sha256": sha256(norm_gold),
                "context_words": len(norm_ctx.split()),
                "gold_words": len(norm_gold.split()),
                "n_candidates": len(conts),
                "gold_index": gold_idx,
                "gold_decode_matches": decoded is None or decoded == norm_gold,
            }
        )

    if not rows:
        raise RuntimeError(f"{label}: produced no rows")

    summary = {
        "label": label,
        "n_docs": len(rows),
        "n_samples": len(task.samples),
        "decode_mismatch": decode_mismatch,
        "label_id_missing": label_id_missing,
        "mean_context_words": sum(r["context_words"] for r in rows) / len(rows),
        "median_gold_words": sorted(r["gold_words"] for r in rows)[len(rows) // 2],
        "gold_under_8_words": sum(1 for r in rows if r["gold_words"] < 8),
        "context_under_8_words": sum(1 for r in rows if r["context_words"] < 8),
    }
    return rows, summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output .jsonl.gz path")
    parser.add_argument("--summary", required=True, help="output summary .json path")
    parser.add_argument("--tokenizer", default="allenai/dolma2-tokenizer")
    parser.add_argument(
        "--labels",
        default="",
        help="comma-separated subset for a fast probe; default is all 20",
    )
    args = parser.parse_args()

    labels = (
        [x.strip() for x in args.labels.split(",") if x.strip()]
        if args.labels
        else list(TASK_LOSS_RAW_LABELS)
    )

    config = build_config(args.tokenizer)
    tokenizer = Tokenizer.from_train_config(config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    total = 0
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        for label in labels:
            rows, summary = dump_label(config, tokenizer, label)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
            summaries.append(summary)
            total += len(rows)
            log.info(
                "%-46s docs=%6d samples=%7d decode_mismatch=%5d "
                "gold<8w=%6d ctx<8w=%5d",
                label,
                summary["n_docs"],
                summary["n_samples"],
                summary["decode_mismatch"],
                summary["gold_under_8_words"],
                summary["context_under_8_words"],
            )

    Path(args.summary).write_text(
        json.dumps({"total_rows": total, "labels": summaries}, indent=1), encoding="utf-8"
    )
    log.info("")
    log.info("total rows: %d across %d labels", total, len(summaries))
    log.info("wrote %s", out)
    log.info("wrote %s", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
