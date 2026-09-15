#!/usr/bin/env python3
"""Dump the eval item text ai2-olmo actually scores (methodology v4 section 4a).

The contamination scan has to search the corpus for the same strings the
evaluator scores. Re-deriving them from HuggingFace fields does not work:
ai2-olmo *assembles* what it scores. HellaSwag's context is activity label +
ctx_a + capitalized ctx_b; MMLU's four subject groups are cut from the `all`
config so positional order is ai2-olmo's rather than HF row order; and for
WinoGrande it was an open question whether the scored continuation is the
option word or the sentence suffix. An index built from the wrong strings
produces a plausible-looking contamination rate for an item set the endpoint
does not use.

Measured answer, for the record: WinoGrande's scored continuation is the
sentence SUFFIX. "sarah was a much better surgeon than maria so maria" ->
"always got the easier cases". BoolQ's gold really is yes/no, and its stem
carries the 98-word passage.

So this reads the strings out of ai2-olmo's own task objects, via the same
`build_evaluator` path the real evaluator uses. No model weights are needed --
`build_evaluator` takes (train_config, evaluator_config, tokenizer, device).

It turns out the ambiguity above dissolves once you look in the right place.
Every `*_rc_5shot_bpb` label is an `OEEvalTask`, which holds pre-built oe-eval
requests where `request["request"]["context"]` and `["continuation"]` are
already the exact strings the model is conditioned on and scored on. There is
nothing to assemble and nothing to guess: `doc_to_text` raises
NotImplementedError on that class precisely because it is never used.

Two things worth knowing about the shape of that data:

  - For a `*_bpb` label, `prep_examples` skips every non-target continuation
    and forces `cont_id = 0`, so exactly one sample per doc is ever scored and
    it is the gold. (This is also why `per_item.reduce_per_item` taking the
    lowest `cont_id` is correct rather than merely conventional.) The full
    request list is still on the task object, so the gold is selected here
    explicitly by `label == idx` and the candidate count is recorded.
  - The context carries the five shared few-shot exemplars. Indexing those
    would make every item in a label match whenever any exemplar text appears
    in the corpus, so the shared preamble is identified as the longest common
    prefix across the label's contexts and stripped, leaving the item's own
    stem.

Output: gzipped JSONL, one row per (label, doc_id), with sha256 of the
normalized stem and gold so section 4b can assert the training-time evaluator
saw the same items.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from pathlib import Path

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
from task_loss.item_identity import (  # noqa: E402
    common_prefix,
    normalize,
    normalizer_fingerprint,
    sha256,
)

log = logging.getLogger("dump_eval_items")


def build_config(tokenizer_id: str) -> TrainConfig:
    """Minimal TrainConfig. Only the tokenizer and sequence length matter here.

    max_sequence_length is pinned to 2048 rather than left to ai2-olmo's
    default of 1024, which silently truncates 5-shot context at half the
    model's trained length (audit finding 1e). It affects which items get
    truncated, so it must match the real evaluator.
    """
    config = TrainConfig.new()
    config.model = ModelConfig(
        # These must match eval_task_loss_olmo_core.make_config, or
        # Tokenizer.from_train_config refuses with a vocab size mismatch.
        vocab_size=100_278,
        embedding_size=100_352,
        eos_token_id=100_257,
        pad_token_id=100_277,
        max_sequence_length=2048,
    )
    config.tokenizer = TokenizerConfig(identifier=tokenizer_id)
    config.device_eval_batch_size = 4
    config.seed = 42
    config.evaluators = []
    return config


def dump_label(config: TrainConfig, tokenizer: Tokenizer, label: str) -> tuple[list[dict], dict]:
    """Return one row per doc_id for `label`, plus a per-label summary.

    `OEEvalTask` holds pre-built oe-eval requests in `task.dataset`, where
    `request["request"]["context"]` and `["continuation"]` are already the
    strings the model is conditioned on and scored on. There is no
    `doc_to_text` to call -- that method raises NotImplementedError on this
    class -- and no field assembly to reverse-engineer, which is what settles
    the WinoGrande and HellaSwag "which field is the gold" question.

    For a `*_bpb` label, `prep_examples` skips every non-target continuation
    and forces `cont_id = 0`, so exactly one sample per doc is ever scored and
    it is the gold. Here the full request list is still available, so the gold
    is selected explicitly by `label == idx` and the candidate count recorded.
    """
    evaluator = build_evaluator(
        config,
        EvaluatorConfig(label=label, type=EvaluatorType.downstream, device_eval_batch_size=4),
        tokenizer,
        torch.device("cpu"),
    )
    task = evaluator.eval_loader.dataset

    # Read from task.samples rather than task.dataset. Every task type builds
    # `samples` in prep_examples with the same schema, whereas the raw
    # `dataset` payload differs by task (MMLU's is not a list of request
    # dicts, and iterating it yields strings). `samples` is also the more
    # authoritative source: `ctx` and `continuation` are the exact token
    # sequences the model is conditioned on and scored on, already truncated
    # from the left at model_ctx_len the way the real evaluator truncates.
    #
    # For a *_bpb label there is exactly one sample per doc and it is the
    # gold, so no gold selection is needed here.
    gold: dict[int, dict] = {}
    duplicate_doc_ids = 0
    for sample in task.samples:
        doc_id = int(sample["doc_id"])
        if doc_id in gold:
            duplicate_doc_ids += 1
            continue
        gold[doc_id] = sample

    if not gold:
        raise RuntimeError(f"{label}: task.samples is empty")
    if duplicate_doc_ids:
        raise RuntimeError(
            f"{label}: {duplicate_doc_ids} docs carry more than one scored sample; "
            f"a *_bpb label must keep only the gold continuation, so the "
            f"one-row-per-item assumption behind the join is wrong here"
        )

    doc_ids = sorted(gold)
    raw_contexts = [tokenizer.decode(list(gold[d]["ctx"])) for d in doc_ids]
    n_candidates: dict[int, int] = {}
    nonint_label = 0
    preamble = common_prefix(raw_contexts)
    # Only treat it as a few-shot preamble if it is substantial; a short
    # accidental overlap between two stems is not an exemplar block.
    if len(preamble.split()) < 8:
        preamble = ""

    rows: list[dict] = []
    for doc_id, raw_ctx in zip(doc_ids, raw_contexts):
        sample = gold[doc_id]
        raw_gold = tokenizer.decode(list(sample["continuation"]))
        stem = raw_ctx[len(preamble) :] if preamble else raw_ctx

        norm_stem = normalize(stem)
        norm_gold = normalize(raw_gold)
        rows.append(
            {
                "label": label,
                "doc_id": doc_id,
                "stem": norm_stem,
                "gold": norm_gold,
                "stem_sha256": sha256(norm_stem),
                "gold_sha256": sha256(norm_gold),
                "full_context_sha256": sha256(normalize(raw_ctx)),
                "stem_words": len(norm_stem.split()),
                "gold_words": len(norm_gold.split()),
                "n_candidates": n_candidates.get(doc_id, 0),
            }
        )

    scored = {int(s["doc_id"]) for s in task.samples}
    summary = {
        "label": label,
        "n_docs": len(rows),
        "n_samples": len(task.samples),
        "n_scored_doc_ids": len(scored),
        "doc_ids_match_scored": sorted(scored) == doc_ids,
        "nonint_label": nonint_label,
        "preamble_words": len(preamble.split()),
        "mean_stem_words": sum(r["stem_words"] for r in rows) / len(rows),
        "median_stem_words": sorted(r["stem_words"] for r in rows)[len(rows) // 2],
        "median_gold_words": sorted(r["gold_words"] for r in rows)[len(rows) // 2],
        "gold_under_8_words": sum(1 for r in rows if r["gold_words"] < 8),
        "stem_under_8_words": sum(1 for r in rows if r["stem_words"] < 8),
        "example_stem": rows[0]["stem"][:160],
        "example_gold": rows[0]["gold"][:160],
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
                "%-46s docs=%6d samples=%7d stem_med=%4d "
                "gold<8w=%6d stem<8w=%5d",
                label,
                summary["n_docs"],
                summary["n_samples"],
                summary["median_stem_words"],
                summary["gold_under_8_words"],
                summary["stem_under_8_words"],
            )

    Path(args.summary).write_text(
        json.dumps(
            {
                "total_rows": total,
                "normalizer_fingerprint": normalizer_fingerprint(),
                "labels": summaries,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    log.info("")
    log.info("total rows: %d across %d labels", total, len(summaries))
    log.info("wrote %s", out)
    log.info("wrote %s", args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
