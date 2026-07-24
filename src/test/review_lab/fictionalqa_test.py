from olmo_core.review_lab.config import DataConfig
from olmo_core.review_lab.fictionalqa import (
    records_from_fictionalqa_interference_rows,
    records_from_fictionalqa_rows,
)
from olmo_core.review_lab.micro_world import group_records


def _row(event: int, style: str, question: int, *, duplicate_root: str | None = None):
    question_id = f"event_{event:03d}_style_{style}_num_000_question_{question:03d}"
    return {
        "event_id": f"event_{event:03d}",
        "fiction_id": f"event_{event:03d}_style_{style}_num_000",
        "question_id": question_id,
        "duplicate_root": duplicate_root or question_id,
        "fict": f"Fictional statement {event} {style} {question}",
        "question": f"What happened in {event} {style} {question}?",
        "natural_answer": f"answer-{event}-{style}-{question}",
    }


def _rows():
    return [
        _row(event, style, question)
        for event in range(4)
        for style in ("blog", "corporate", "encyclopedia", "news", "social")
        for question in range(2)
    ]


def test_fictionalqa_conversion_is_event_disjoint_and_pairs_fact_with_question():
    config = DataConfig(dataset="fictionalqa", old_events=2, new_events=2, seed=7)
    records = records_from_fictionalqa_rows(_rows(), config)

    old_events = {record.entity for record in records if record.stage == "old"}
    new_events = {record.entity for record in records if record.stage == "new"}
    assert old_events.isdisjoint(new_events)
    assert len(records) == len(_rows()) * 2
    assert {record.skill for record in records} == {
        "blog",
        "corporate",
        "encyclopedia",
        "news",
        "social",
    }

    by_fact = {}
    for record in records:
        by_fact.setdefault(record.fact_id, []).append(record)
    assert all({item.split for item in pair} == {"train", "eval"} for pair in by_fact.values())
    assert all(
        next(item for item in pair if item.split == "train").answer == ""
        for pair in by_fact.values()
    )
    assert all(
        next(item for item in pair if item.split == "eval").answer
        for pair in by_fact.values()
    )


def test_fictionalqa_conversion_drops_non_root_duplicates():
    rows = _rows()
    duplicate = dict(rows[0])
    duplicate["question_id"] += "_duplicate"
    duplicate["duplicate_root"] = rows[0]["question_id"]
    records = records_from_fictionalqa_rows(rows + [duplicate], DataConfig(
        dataset="fictionalqa", old_events=2, new_events=2, seed=7
    ))
    assert len(records) == len(rows) * 2


def test_dynamic_grouping_works_for_fictionalqa_styles():
    records = records_from_fictionalqa_rows(
        _rows(), DataConfig(dataset="fictionalqa", old_events=2, new_events=2, seed=7)
    )
    grouped = group_records(records, stage="old", split="train")
    assert set(grouped) == {"blog", "corporate", "encyclopedia", "news", "social"}


def test_interference_conversion_pairs_archive_answers_without_contradiction():
    config = DataConfig(
        dataset="fictionalqa_interference", old_events=2, new_events=0, seed=7
    )
    records = records_from_fictionalqa_interference_rows(_rows(), config)

    old_eval = {
        record.fact_id: record
        for record in records
        if record.stage == "old" and record.split == "eval"
    }
    new_eval = {
        record.fact_id.removeprefix("archive-b-"): record
        for record in records
        if record.stage == "new" and record.split == "eval"
    }
    assert old_eval.keys() == new_eval.keys()
    assert len(old_eval) == 20
    assert all(old_eval[fact_id].value != new_eval[fact_id].value for fact_id in old_eval)
    assert all("Archive A" in record.prompt for record in old_eval.values())
    assert all("Archive B" in record.prompt for record in new_eval.values())
