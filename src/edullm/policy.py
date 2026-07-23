"""
Load trusted eduLLM queue policy and operator configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from edullm.models import Operator


@dataclass(frozen=True)
class Policy:
    """Trusted limits, allowlists, and entrypoint profiles for queue requests."""

    wandb_entity: str
    allowed_wandb_projects: tuple[str, ...]
    max_runtime_minutes: int = 360
    max_gpu_count: int = 2
    allowed_gpu_preferences: tuple[str, ...] = ("any", "l40s", "h100", "h200")
    required_checks: tuple[str, ...] = (
        "Lint",
        "Test",
        "Test checkpoint",
        "Test transformer",
        "Test attention",
        "Test examples",
        "Test scripts",
        "Integration tests",
        "Test olmo3 ladder",
        "Type check",
        "Build",
        "Style",
        "Docs",
    )
    entrypoints: dict[str, dict] = field(default_factory=dict)
    reminder_after_minutes: int = 15
    reassign_after_minutes: int = 30


def load_policy(path: Path, entrypoints_path: Path | None = None) -> Policy:
    """
    Load queue limits and the protected entrypoint allowlist.

    :param path: The policy YAML path.
    :param entrypoints_path: An optional entrypoint YAML path. By default the
        sibling ``entrypoints.yaml`` file is used.

    :returns: The loaded immutable policy record.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    entrypoints_path = entrypoints_path or path.with_name("entrypoints.yaml")
    entrypoints = yaml.safe_load(entrypoints_path.read_text(encoding="utf-8"))["entrypoints"]
    return Policy(
        wandb_entity=data["wandb_entity"],
        allowed_wandb_projects=tuple(data["allowed_wandb_projects"]),
        max_runtime_minutes=int(data.get("max_runtime_minutes", 360)),
        max_gpu_count=int(data.get("max_gpu_count", 2)),
        allowed_gpu_preferences=tuple(
            data.get("allowed_gpu_preferences", ["any", "l40s", "h100", "h200"])
        ),
        required_checks=tuple(data["required_checks"]),
        entrypoints=entrypoints,
        reminder_after_minutes=int(data.get("reminder_after_minutes", 15)),
        reassign_after_minutes=int(data.get("reassign_after_minutes", 30)),
    )


def load_operators(path: Path) -> tuple[Operator, ...]:
    """
    Load the reviewed operator roster.

    :param path: The operator YAML path.

    :returns: Operators in their configured order.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return tuple(
        Operator(
            github=row["github"],
            slack_user_id=row["slack_user_id"],
            rotation_order=int(row["rotation_order"]),
            enabled=bool(row.get("enabled", True)),
            apptainer_path=row.get("apptainer_path"),
            apptainer_sha256=row.get("apptainer_sha256"),
        )
        for row in data["operators"]
    )
