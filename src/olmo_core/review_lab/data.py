from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import DataConfig
from .micro_world import MicroWorldRecord, build_micro_world, load_micro_world


def build_review_data(config: DataConfig, *, force: bool = False) -> Path:
    if config.dataset == "micro_world":
        return build_micro_world(config, force=force)
    if config.dataset in {"fictionalqa", "fictionalqa_interference"}:
        from .fictionalqa import build_fictionalqa

        return build_fictionalqa(config, force=force)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def load_review_data(config: DataConfig) -> List[MicroWorldRecord]:
    return load_micro_world(config.path)


def review_skills(records: List[MicroWorldRecord]) -> Tuple[str, ...]:
    skills = tuple(sorted({record.skill for record in records}))
    if not skills:
        raise ValueError("Review dataset contains no skills")
    return skills
