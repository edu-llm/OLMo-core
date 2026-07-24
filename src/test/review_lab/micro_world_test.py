from dataclasses import replace

from olmo_core.review_lab.config import DataConfig
from olmo_core.review_lab.micro_world import SKILLS, generate_micro_world, group_records


def test_micro_world_is_deterministic_and_has_expected_size():
    config = DataConfig(
        facts_per_skill=8,
        new_facts_per_skill=12,
        train_paraphrases_per_fact=2,
        eval_paraphrases_per_fact=1,
        eval_facts_per_skill=4,
    )
    first = generate_micro_world(config)
    second = generate_micro_world(config)
    changed = generate_micro_world(replace(config, seed=config.seed + 1))

    expected_per_skill = (8 * 2 + 4 * 1) + (12 * 2 + 4 * 1)
    assert len(first) == len(SKILLS) * expected_per_skill
    assert first == second
    assert first != changed


def test_eval_prompts_do_not_contain_the_answer():
    records = generate_micro_world(DataConfig(facts_per_skill=4, new_facts_per_skill=4))
    eval_records = [record for record in records if record.split == "eval"]
    assert eval_records
    assert all(record.answer for record in eval_records)
    assert all(record.answer.lower() not in record.prompt.lower() for record in eval_records)


def test_grouping_keeps_stage_split_and_skills_separate():
    records = generate_micro_world(DataConfig(facts_per_skill=4, new_facts_per_skill=5))
    grouped = group_records(records, stage="old", split="train")
    assert set(grouped) == set(SKILLS)
    assert all(record.stage == "old" for values in grouped.values() for record in values)
    assert all(record.split == "train" for values in grouped.values() for record in values)
