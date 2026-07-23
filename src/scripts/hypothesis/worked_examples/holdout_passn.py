#!/usr/bin/env python3
"""Holdout Pass@N / PassRatio@N for worked-examples CPT (W&B-safe metric names).

Reads ``eval/holdout_bare.jsonl`` (bare problem prompts + ``final_answer``).
Logs validator-safe names (no whitespace):

- ``eval/pass_at_n``
- ``eval/pass_ratio_at_n``

Can run standalone after a checkpoint, or be imported by the train callback.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

_NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    text = text.strip()
    if "####" in text:
        text = text.split("####")[-1].strip()
    if "Answer:" in text:
        text = text.split("Answer:")[-1].strip()
    text = text.replace(",", "")
    m = _NUM.findall(text)
    if m:
        return m[-1].lstrip("+")
    return text.lower()


def is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


@dataclass
class HoldoutItem:
    prompt: str
    final_answer: str
    family_id: Optional[str] = None


def load_holdout(path: Path, *, max_items: int = 0) -> List[HoldoutItem]:
    items: List[HoldoutItem] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = row.get("question") or row.get("problem") or row.get("prompt")
            if q is None and "text" in row:
                # fall back: strip Answer: if present
                q = str(row["text"]).split("Answer:")[0].strip()
            ans = row.get("final_answer") or row.get("answer")
            if q is None or ans is None:
                raise ValueError(f"holdout row missing question/answer keys: {row.keys()}")
            prompt = q if str(q).startswith("Problem:") else f"Problem: {q}\n"
            if not prompt.endswith("\n"):
                prompt += "\n"
            items.append(
                HoldoutItem(
                    prompt=prompt,
                    final_answer=str(ans),
                    family_id=row.get("family_id") or row.get("original_question"),
                )
            )
            if max_items and len(items) >= max_items:
                break
    return items


def pass_at_n_and_ratio(correct_flags: Sequence[bool]) -> Tuple[float, float]:
    """Per-item: Pass@N = any correct; PassRatio@N = mean correct."""
    if not correct_flags:
        return 0.0, 0.0
    any_ok = 1.0 if any(correct_flags) else 0.0
    ratio = sum(1.0 for c in correct_flags if c) / len(correct_flags)
    return any_ok, ratio


def aggregate_pass_metrics(
    per_item_correct: Iterable[Sequence[bool]],
) -> dict[str, float]:
    pass_scores: List[float] = []
    ratio_scores: List[float] = []
    for flags in per_item_correct:
        p, r = pass_at_n_and_ratio(flags)
        pass_scores.append(p)
        ratio_scores.append(r)
    n = max(len(pass_scores), 1)
    return {
        "eval/pass_at_n": sum(pass_scores) / n,
        "eval/pass_ratio_at_n": sum(ratio_scores) / n,
        "eval/n_items": float(len(pass_scores)),
    }


def score_generations(
    items: Sequence[HoldoutItem],
    generations: Sequence[Sequence[str]],
) -> dict[str, float]:
    if len(items) != len(generations):
        raise ValueError("items and generations length mismatch")
    per_item = [
        [is_correct(gen, item.final_answer) for gen in gens]
        for item, gens in zip(items, generations)
    ]
    return aggregate_pass_metrics(per_item)


def log_metrics_to_wandb(
    metrics: dict[str, float],
    *,
    entity: str = "eduLLM",
    project: str = "pretraining",
    group: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    step: Optional[int] = None,
) -> None:
    """Log Pass@N metrics to W&B (optional; requires WANDB_API_KEY)."""
    import os

    import wandb

    if "WANDB_API_KEY" not in os.environ:
        log.warning("WANDB_API_KEY missing; skipping W&B log")
        return
    run = wandb.run
    created = False
    if run is None:
        wandb.init(entity=entity, project=project, group=group, name=name, tags=tags or [])
        created = True
        run = wandb.run
    assert run is not None
    payload = {k: float(v) for k, v in metrics.items()}
    if step is None:
        run.log(payload)
    else:
        run.log(payload, step=step)
    if created:
        wandb.finish()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", type=Path, required=True, help="Path to holdout_bare.jsonl")
    ap.add_argument("--predictions", type=Path, required=True, help="JSONL: {gens: [str,...]}")
    ap.add_argument("--max-items", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="eduLLM")
    ap.add_argument("--wandb-project", default="pretraining")
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-name", default=None)
    args = ap.parse_args()

    items = load_holdout(args.holdout, max_items=args.max_items)
    gens: List[List[str]] = []
    with args.predictions.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gens.append([str(x) for x in row["gens"]])
    if len(gens) != len(items):
        raise SystemExit(f"prediction rows {len(gens)} != holdout items {len(items)}")

    metrics = score_generations(items, gens)
    print(json.dumps(metrics, indent=2))
    if args.wandb:
        log_metrics_to_wandb(
            metrics,
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
        )


if __name__ == "__main__":
    main()
