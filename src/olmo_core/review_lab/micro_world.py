from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .config import DataConfig


SKILLS = ("registry", "routing", "codebook", "precedence")

_VALUES = (
    "amber",
    "birch",
    "cobalt",
    "dawn",
    "ember",
    "frost",
    "garnet",
    "harbor",
    "indigo",
    "juniper",
    "kelp",
    "lilac",
    "marble",
    "north",
    "onyx",
    "pearl",
    "quartz",
    "reed",
    "sable",
    "thistle",
    "umber",
    "violet",
    "willow",
    "yarrow",
)

_TRAIN_TEMPLATES: Mapping[str, Sequence[str]] = {
    "registry": (
        "In the Luma registry, {entity} is assigned to guild {value}.",
        "The official Luma ledger records {entity} under the {value} guild.",
        "Registry fact: the guild for {entity} is {value}.",
    ),
    "routing": (
        "On the Vela transit map, gate {entity} routes to station {value}.",
        "Travel through Vela gate {entity} and the destination is {value} station.",
        "Routing fact: {entity} has terminal station {value}.",
    ),
    "codebook": (
        "The Orin codebook maps signal {entity} to token {value}.",
        "Under Orin protocol, {entity} is decoded as {value}.",
        "Codebook fact: the token paired with {entity} is {value}.",
    ),
    "precedence": (
        "In the Sora ritual, action {entity} must be followed by {value}.",
        "The Sora sequence places {value} immediately after {entity}.",
        "Sequence fact: after {entity}, perform {value}.",
    ),
}

_EVAL_TEMPLATES: Mapping[str, Sequence[str]] = {
    "registry": (
        "Question: Which guild contains {entity} in the Luma registry?\nAnswer:",
        "Complete the ledger entry. {entity} belongs to guild",
        "Luma lookup for {entity}: guild =",
    ),
    "routing": (
        "Question: Where does Vela gate {entity} lead?\nAnswer:",
        "Complete the route. Gate {entity} terminates at station",
        "Vela destination for gate {entity}:",
    ),
    "codebook": (
        "Question: How is signal {entity} decoded in the Orin codebook?\nAnswer:",
        "Complete the mapping. Orin signal {entity} maps to token",
        "Orin decode for {entity}:",
    ),
    "precedence": (
        "Question: What immediately follows {entity} in the Sora ritual?\nAnswer:",
        "Complete the sequence. After {entity}, perform",
        "Sora successor for {entity}:",
    ),
}


@dataclass(frozen=True)
class MicroWorldRecord:
    example_id: str
    fact_id: str
    stage: str
    split: str
    skill: str
    entity: str
    value: str
    prompt: str
    answer: str


def _stable_word(seed: int, skill: str, stage: str, index: int) -> str:
    payload = f"{seed}:{skill}:{stage}:{index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=5).hexdigest()
    return f"{skill[:2]}-{digest}"


def _fact_value(rng: random.Random, used: set[str]) -> str:
    choices = [value for value in _VALUES if value not in used]
    if not choices:
        used.clear()
        choices = list(_VALUES)
    value = rng.choice(choices)
    used.add(value)
    return value


def generate_micro_world(config: DataConfig) -> List[MicroWorldRecord]:
    """Generate counterfactual facts with held-out prompt paraphrases.

    The same underlying facts appear in train and eval records, but their surface forms do not.
    That makes evaluation a test of transferring a learned relation to a new phrasing instead of
    merely continuing a memorized training sentence.
    """

    rng = random.Random(config.seed)
    records: List[MicroWorldRecord] = []
    stage_sizes = {"old": config.facts_per_skill, "new": config.new_facts_per_skill}

    for stage, fact_count in stage_sizes.items():
        for skill in SKILLS:
            used_values: set[str] = set()
            for fact_index in range(fact_count):
                entity = _stable_word(config.seed, skill, stage, fact_index)
                value = _fact_value(rng, used_values)
                fact_id = f"{stage}-{skill}-{fact_index:04d}"

                train_templates = list(_TRAIN_TEMPLATES[skill])
                rng.shuffle(train_templates)
                for template_index, template in enumerate(
                    train_templates[: config.train_paraphrases_per_fact]
                ):
                    records.append(
                        MicroWorldRecord(
                            example_id=f"{fact_id}-train-{template_index}",
                            fact_id=fact_id,
                            stage=stage,
                            split="train",
                            skill=skill,
                            entity=entity,
                            value=value,
                            prompt=template.format(entity=entity, value=value),
                            answer="",
                        )
                    )

                if fact_index >= config.eval_facts_per_skill:
                    continue
                eval_templates = list(_EVAL_TEMPLATES[skill])
                rng.shuffle(eval_templates)
                for template_index, template in enumerate(
                    eval_templates[: config.eval_paraphrases_per_fact]
                ):
                    records.append(
                        MicroWorldRecord(
                            example_id=f"{fact_id}-eval-{template_index}",
                            fact_id=fact_id,
                            stage=stage,
                            split="eval",
                            skill=skill,
                            entity=entity,
                            value=value,
                            prompt=template.format(entity=entity),
                            answer=value,
                        )
                    )

    return records


def write_micro_world(
    records: Iterable[MicroWorldRecord], path: str | Path, *, force: bool = False
) -> Path:
    output = Path(path)
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return output


def build_micro_world(config: DataConfig, *, force: bool = False) -> Path:
    return write_micro_world(generate_micro_world(config), config.path, force=force)


def load_micro_world(path: str | Path) -> List[MicroWorldRecord]:
    records: List[MicroWorldRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                records.append(MicroWorldRecord(**raw))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Invalid record at {path}:{line_number}") from exc
    return records


def group_records(
    records: Iterable[MicroWorldRecord], *, stage: str, split: str
) -> Dict[str, List[MicroWorldRecord]]:
    matching = [
        record for record in records if record.stage == stage and record.split == split
    ]
    grouped: Dict[str, List[MicroWorldRecord]] = {
        skill: [] for skill in sorted({record.skill for record in matching})
    }
    for record in matching:
        grouped[record.skill].append(record)
    if not grouped:
        raise ValueError(f"No {stage}/{split} examples")
    return grouped
