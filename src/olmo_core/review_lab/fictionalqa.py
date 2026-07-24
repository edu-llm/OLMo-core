from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import requests

from .config import DataConfig
from .micro_world import MicroWorldRecord, write_micro_world


DATASETS_SERVER = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api/datasets"
STYLES = ("blog", "corporate", "encyclopedia", "news", "social")
_STYLE_PATTERN = re.compile(r"_style_([a-z]+)_num_")


def _get_json(url: str, *, params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object from {response.url}")
    return payload


def fetch_fictionalqa_rows(config: DataConfig) -> List[Dict[str, Any]]:
    """Fetch the small FictionalQA QA table through the public Dataset Viewer API."""

    metadata = _get_json(f"{HUB_API}/{config.source_dataset}")
    actual_revision = str(metadata.get("sha", ""))
    if config.source_revision and actual_revision != config.source_revision:
        raise ValueError(
            "FictionalQA revision changed: "
            f"expected {config.source_revision}, got {actual_revision or 'unknown'}"
        )

    split_payload = _get_json(
        f"{DATASETS_SERVER}/splits", params={"dataset": config.source_dataset}
    )
    matching = [
        split
        for split in split_payload.get("splits", [])
        if split.get("config") == config.source_config and split.get("split") == "train"
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Could not uniquely locate {config.source_dataset}/{config.source_config}/train"
        )
    size_payload = _get_json(
        f"{DATASETS_SERVER}/size", params={"dataset": config.source_dataset}
    )
    sizes = [
        split
        for split in size_payload.get("size", {}).get("splits", [])
        if split.get("config") == config.source_config and split.get("split") == "train"
    ]
    if len(sizes) != 1:
        raise ValueError(
            f"Could not determine the size of {config.source_dataset}/{config.source_config}/train"
        )
    total = int(sizes[0]["num_rows"])

    rows: List[Dict[str, Any]] = []
    page_size = 100
    for offset in range(0, total, page_size):
        payload = _get_json(
            f"{DATASETS_SERVER}/rows",
            params={
                "dataset": config.source_dataset,
                "config": config.source_config,
                "split": "train",
                "offset": offset,
                "length": min(page_size, total - offset),
            },
        )
        rows.extend(dict(item["row"]) for item in payload.get("rows", []))
    if len(rows) != total:
        raise ValueError(f"Expected {total} FictionalQA rows, downloaded {len(rows)}")
    return rows


def _style_from_fiction_id(fiction_id: str) -> str:
    match = _STYLE_PATTERN.search(fiction_id)
    if not match or match.group(1) not in STYLES:
        raise ValueError(f"Could not recover a FictionalQA style from {fiction_id!r}")
    return match.group(1)


def _event_split(rows: Sequence[Mapping[str, Any]], config: DataConfig) -> Dict[str, str]:
    events = sorted({str(row["event_id"]) for row in rows})
    needed = config.old_events + config.new_events
    if needed > len(events):
        raise ValueError(f"Requested {needed} events but FictionalQA only has {len(events)}")
    rng = random.Random(config.seed)
    rng.shuffle(events)
    return {
        **{event_id: "old" for event_id in events[: config.old_events]},
        **{
            event_id: "new"
            for event_id in events[config.old_events : config.old_events + config.new_events]
        },
    }


def records_from_fictionalqa_rows(
    rows: Sequence[Mapping[str, Any]], config: DataConfig
) -> List[MicroWorldRecord]:
    """Create document-learning and closed-book QA records from FictionalQA.

    Duplicate question clusters are reduced to their canonical root. The model trains only on the
    dataset's declarative ``fict`` statement and is evaluated on the differently worded question
    plus its short ``natural_answer``. Event-level old/new splitting prevents fact leakage.
    """

    event_stages = _event_split(rows, config)
    canonical: Dict[str, Mapping[str, Any]] = {}
    for row in rows:
        event_id = str(row["event_id"])
        if event_id not in event_stages:
            continue
        question_id = str(row["question_id"])
        root = str(row.get("duplicate_root") or question_id)
        if question_id == root:
            canonical[root] = row

    records: List[MicroWorldRecord] = []
    for question_id, row in sorted(canonical.items()):
        event_id = str(row["event_id"])
        stage = event_stages[event_id]
        skill = _style_from_fiction_id(str(row["fiction_id"]))
        fact = str(row["fict"]).strip()
        question = str(row["question"]).strip()
        answer = str(row["natural_answer"]).strip()
        if not fact or not question or not answer:
            continue
        records.append(
            MicroWorldRecord(
                example_id=f"{question_id}-train",
                fact_id=question_id,
                stage=stage,
                split="train",
                skill=skill,
                entity=event_id,
                value=answer,
                prompt=f"Fictional fact: {fact}",
                answer="",
            )
        )
        records.append(
            MicroWorldRecord(
                example_id=f"{question_id}-eval",
                fact_id=question_id,
                stage=stage,
                split="eval",
                skill=skill,
                entity=event_id,
                value=answer,
                prompt=f"Question: {question}\nAnswer:",
                answer=answer,
            )
        )

    required = {
        (stage, split, skill)
        for stage in ("old", "new")
        for split in ("train", "eval")
        for skill in STYLES
    }
    observed = {(record.stage, record.split, record.skill) for record in records}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"FictionalQA conversion produced empty groups: {missing}")
    return records


def records_from_fictionalqa_interference_rows(
    rows: Sequence[Mapping[str, Any]], config: DataConfig
) -> List[MicroWorldRecord]:
    """Create paired Archive-A/Archive-B facts that compete without contradiction.

    Archive A contains the original FictionalQA answer. Archive B reuses the same
    question with a deterministic answer drawn from another fact in the same
    document style. Explicit archive labels let both answers remain logically valid
    while making the entity/relation surface form maximally interfering.
    """

    events = sorted({str(row["event_id"]) for row in rows})
    if config.old_events > len(events):
        raise ValueError(
            f"Requested {config.old_events} old events but FictionalQA only has {len(events)}"
        )
    rng = random.Random(config.seed)
    rng.shuffle(events)
    selected_events = set(events[: config.old_events])

    canonical: List[Mapping[str, Any]] = []
    for row in rows:
        if str(row["event_id"]) not in selected_events:
            continue
        question_id = str(row["question_id"])
        if question_id == str(row.get("duplicate_root") or question_id):
            canonical.append(row)
    canonical.sort(key=lambda row: str(row["question_id"]))

    by_style: Dict[str, List[Mapping[str, Any]]] = {style: [] for style in STYLES}
    for row in canonical:
        by_style[_style_from_fiction_id(str(row["fiction_id"]))].append(row)

    distractors: Dict[str, str] = {}
    for style, style_rows in by_style.items():
        if not style_rows:
            raise ValueError(f"FictionalQA interference conversion has no {style} facts")
        answers = [str(row["natural_answer"]).strip() for row in style_rows]
        for index, row in enumerate(style_rows):
            original = answers[index]
            distractor = ""
            for offset in range(1, len(style_rows) + 1):
                candidate = answers[(index + offset) % len(style_rows)]
                if candidate and candidate.casefold() != original.casefold():
                    distractor = candidate
                    break
            if not distractor:
                raise ValueError(f"Could not find a distinct Archive-B answer for style {style}")
            distractors[str(row["question_id"])] = distractor

    records: List[MicroWorldRecord] = []
    for row in canonical:
        question_id = str(row["question_id"])
        event_id = str(row["event_id"])
        style = _style_from_fiction_id(str(row["fiction_id"]))
        fact = str(row["fict"]).strip()
        question = str(row["question"]).strip()
        old_answer = str(row["natural_answer"]).strip()
        new_answer = distractors[question_id]
        if not fact or not question or not old_answer or not new_answer:
            continue

        records.extend(
            [
                MicroWorldRecord(
                    example_id=f"{question_id}-archive-a-train",
                    fact_id=question_id,
                    stage="old",
                    split="train",
                    skill=style,
                    entity=event_id,
                    value=old_answer,
                    prompt=(
                        "Archive A fictional record: "
                        f'For the question "{question}", the answer is {old_answer}.'
                    ),
                    answer="",
                ),
                MicroWorldRecord(
                    example_id=f"{question_id}-archive-a-eval",
                    fact_id=question_id,
                    stage="old",
                    split="eval",
                    skill=style,
                    entity=event_id,
                    value=old_answer,
                    prompt=(
                        "According to Archive A, answer the fictional question.\n"
                        f"Question: {question}\nAnswer:"
                    ),
                    answer=old_answer,
                ),
                MicroWorldRecord(
                    example_id=f"{question_id}-archive-b-train",
                    fact_id=f"archive-b-{question_id}",
                    stage="new",
                    split="train",
                    skill=style,
                    entity=event_id,
                    value=new_answer,
                    prompt=(
                        "Archive B fictional update: "
                        f'For the question "{question}", the answer is {new_answer}.'
                    ),
                    answer="",
                ),
                MicroWorldRecord(
                    example_id=f"{question_id}-archive-b-eval",
                    fact_id=f"archive-b-{question_id}",
                    stage="new",
                    split="eval",
                    skill=style,
                    entity=event_id,
                    value=new_answer,
                    prompt=(
                        "According to Archive B, answer the fictional question.\n"
                        f"Question: {question}\nAnswer:"
                    ),
                    answer=new_answer,
                ),
            ]
        )
    return records


def _write_metadata(config: DataConfig, records: Iterable[MicroWorldRecord]) -> None:
    records = list(records)
    path = Path(config.path).with_suffix(".metadata.json")
    counts: Dict[str, int] = {}
    for record in records:
        key = f"{record.stage}/{record.split}/{record.skill}"
        counts[key] = counts.get(key, 0) + 1
    metadata = {
        "dataset": config.source_dataset,
        "revision": config.source_revision,
        "config": config.source_config,
        "seed": config.seed,
        "old_events": config.old_events,
        "new_events": config.new_events,
        "records": len(records),
        "counts": dict(sorted(counts.items())),
        "training_field": "fict",
        "evaluation_fields": ["question", "natural_answer"],
        "deduplication": "question_id == duplicate_root",
        "interference_design": (
            "paired_archive_answers"
            if config.dataset == "fictionalqa_interference"
            else "event_disjoint"
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_fictionalqa(config: DataConfig, *, force: bool = False) -> Path:
    output = Path(config.path)
    if output.exists() and not force:
        return output
    rows = fetch_fictionalqa_rows(config)
    records = (
        records_from_fictionalqa_interference_rows(rows, config)
        if config.dataset == "fictionalqa_interference"
        else records_from_fictionalqa_rows(rows, config)
    )
    result = write_micro_world(records, output, force=True)
    _write_metadata(config, records)
    return result
