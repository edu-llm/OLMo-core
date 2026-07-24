#!/usr/bin/env python3
"""Generate and validate the tiny P4 blocked/interleaved dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Sequence


Digits = tuple[int, ...]
BATCH_SIZE = 32
TRAIN_PER_SKILL = 4096
TEST_PER_SKILL = 256
SKILLS = ("A", "B", "C", "D")
ORDERS = tuple(
    tuple(SKILLS[(start + offset) % len(SKILLS)] for offset in range(len(SKILLS)))
    for start in range(len(SKILLS))
)
DELIMITERS = {
    "A": ("<", ">"),
    "B": ("[", "]"),
    "C": ("{", "}"),
    "D": ("(", ")"),
}


def rotate_left(xs: Digits) -> Digits:
    return xs[1:] + xs[:1]


def rotate_right(xs: Digits) -> Digits:
    return xs[-1:] + xs[:-1]


def reverse(xs: Digits) -> Digits:
    return xs[::-1]


def swap_pairs(xs: Digits) -> Digits:
    return tuple(value for i in range(0, len(xs), 2) for value in (xs[i + 1], xs[i]))


TRANSFORMS: dict[str, Callable[[Digits], Digits]] = {
    "A": rotate_left,
    "B": rotate_right,
    "C": reverse,
    "D": swap_pairs,
}


def skeleton_key(xs: Sequence[int]) -> str:
    return "".join(str(x) for x in xs)


def valid_skeleton(xs: Digits) -> bool:
    outputs = {TRANSFORMS[skill](xs) for skill in SKILLS}
    return len(outputs) == len(SKILLS)


def draw_skeletons(count: int, rng: random.Random, used: set[Digits]) -> list[Digits]:
    result: list[Digits] = []
    while len(result) < count:
        xs = tuple(rng.randrange(10) for _ in range(6))
        if xs in used or not valid_skeleton(xs):
            continue
        used.add(xs)
        result.append(xs)
    return result


def make_record(split: str, skill: str, xs: Digits) -> dict[str, object]:
    left, right = DELIMITERS[skill]
    answer = TRANSFORMS[skill](xs)
    skeleton = skeleton_key(xs)
    record_id = hashlib.sha256(f"{split}:{skill}:{skeleton}".encode("ascii")).hexdigest()[:20]
    prompt = f"Input: {left} {' '.join(map(str, xs))} {right}\nOutput: "
    target = f"{' '.join(map(str, answer))}\n"
    return {
        "id": record_id,
        "split": split,
        "skill": skill,
        "skeleton": skeleton,
        "input_digits": list(xs),
        "output_digits": list(answer),
        "prompt": prompt,
        "target": target,
        "text": prompt + target,
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_schedules(
    train_by_skill: dict[str, list[dict[str, object]]],
    order: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blocked: list[dict[str, object]] = []
    for skill in order:
        blocked.extend(train_by_skill[skill])

    interleaved: list[dict[str, object]] = []
    batches_per_skill = TRAIN_PER_SKILL // BATCH_SIZE
    for batch_index in range(batches_per_skill):
        start = batch_index * BATCH_SIZE
        stop = start + BATCH_SIZE
        for skill in order:
            interleaved.extend(train_by_skill[skill][start:stop])
    return blocked, interleaved


def batch_skills(rows: Sequence[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    for start in range(0, len(rows), BATCH_SIZE):
        skills = {str(row["skill"]) for row in rows[start : start + BATCH_SIZE]}
        if len(skills) != 1:
            raise AssertionError(f"batch at record {start} is not skill-homogeneous: {skills}")
        labels.append(next(iter(skills)))
    return labels


def validate(
    train_by_skill: dict[str, list[dict[str, object]]],
    test_rows: Sequence[dict[str, object]],
    schedules: dict[int, tuple[Sequence[dict[str, object]], Sequence[dict[str, object]]]],
) -> dict[str, object]:
    expected_records = TRAIN_PER_SKILL * len(SKILLS)

    order_validations: list[dict[str, object]] = []
    reference_ids: Counter[str] | None = None
    reference_blocked: Sequence[dict[str, object]] | None = None
    for order_index, order in enumerate(ORDERS):
        blocked, interleaved = schedules[order_index]
        assert len(blocked) == expected_records
        assert len(interleaved) == expected_records
        blocked_ids = Counter(str(row["id"]) for row in blocked)
        interleaved_ids = Counter(str(row["id"]) for row in interleaved)
        assert blocked_ids == interleaved_ids
        assert len(blocked_ids) == expected_records
        if reference_ids is None:
            reference_ids = blocked_ids
            reference_blocked = blocked
        else:
            assert blocked_ids == reference_ids

        blocked_labels = batch_skills(blocked)
        interleaved_labels = batch_skills(interleaved)
        expected_blocked = [
            skill
            for skill in order
            for _ in range(TRAIN_PER_SKILL // BATCH_SIZE)
        ]
        expected_interleaved = list(order) * (TRAIN_PER_SKILL // BATCH_SIZE)
        assert blocked_labels == expected_blocked
        assert interleaved_labels == expected_interleaved

        for skill in SKILLS:
            order_blocked_ids = [row["id"] for row in blocked if row["skill"] == skill]
            order_interleaved_ids = [
                row["id"] for row in interleaved if row["skill"] == skill
            ]
            original_ids = [row["id"] for row in train_by_skill[skill]]
            assert order_blocked_ids == original_ids == order_interleaved_ids

        order_validations.append(
            {
                "blocked_batch_labels_sha256": hashlib.sha256(
                    "".join(blocked_labels).encode("ascii")
                ).hexdigest(),
                "blocked_steps": len(blocked_labels),
                "interleaved_batch_labels_sha256": hashlib.sha256(
                    "".join(interleaved_labels).encode("ascii")
                ).hexdigest(),
                "interleaved_steps": len(interleaved_labels),
                "order": list(order),
                "order_index": order_index,
            }
        )

    assert reference_blocked is not None

    train_skeletons = {str(row["skeleton"]) for row in reference_blocked}
    test_skeletons = {str(row["skeleton"]) for row in test_rows}
    assert train_skeletons.isdisjoint(test_skeletons)
    assert Counter(row["skill"] for row in test_rows) == Counter({skill: TEST_PER_SKILL for skill in SKILLS})

    for row in list(reference_blocked) + list(test_rows):
        xs = tuple(int(x) for x in row["input_digits"])
        expected = TRANSFORMS[str(row["skill"])](xs)
        assert tuple(row["output_digits"]) == expected
        assert valid_skeleton(xs)

    return {
        "batch_size": BATCH_SIZE,
        "orders": order_validations,
        "test_examples": len(test_rows),
        "test_examples_per_skill": dict(Counter(str(row["skill"]) for row in test_rows)),
        "train_examples": expected_records,
        "train_examples_per_skill": dict(
            Counter(str(row["skill"]) for row in reference_blocked)
        ),
        "train_test_skeleton_overlap": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    used: set[Digits] = set()
    train_skeletons = draw_skeletons(TRAIN_PER_SKILL, rng, used)
    test_skeletons = draw_skeletons(TEST_PER_SKILL, rng, used)

    train_by_skill: dict[str, list[dict[str, object]]] = {}
    for skill_index, skill in enumerate(SKILLS):
        rows = [make_record("train", skill, xs) for xs in train_skeletons]
        random.Random(args.seed + 1000 + skill_index).shuffle(rows)
        train_by_skill[skill] = rows

    test_rows = [
        make_record("test", skill, xs)
        for skill in SKILLS
        for xs in test_skeletons
    ]
    schedules = {
        order_index: build_schedules(train_by_skill, order)
        for order_index, order in enumerate(ORDERS)
    }
    validation = validate(train_by_skill, test_rows, schedules)

    train_rows = [row for skill in SKILLS for row in train_by_skill[skill]]
    paths = {
        "train": args.output_dir / "train.jsonl",
        "test": args.output_dir / "test.jsonl",
        "blocked": args.output_dir / "blocked_order.jsonl",
        "interleaved": args.output_dir / "interleaved_order.jsonl",
    }
    for order_index in schedules:
        paths[f"blocked_order_{order_index}"] = (
            args.output_dir / f"blocked_order_{order_index}.jsonl"
        )
        paths[f"interleaved_order_{order_index}"] = (
            args.output_dir / f"interleaved_order_{order_index}.jsonl"
        )
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["test"], test_rows)
    write_jsonl(paths["blocked"], schedules[0][0])
    write_jsonl(paths["interleaved"], schedules[0][1])
    for order_index, (blocked, interleaved) in schedules.items():
        write_jsonl(paths[f"blocked_order_{order_index}"], blocked)
        write_jsonl(paths[f"interleaved_order_{order_index}"], interleaved)

    manifest = {
        "generator_seed": args.seed,
        "rules": {
            "A": "rotate_left",
            "B": "rotate_right",
            "C": "reverse",
            "D": "swap_adjacent_pairs",
        },
        "files": {
            name: {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
        "validation": validation,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
