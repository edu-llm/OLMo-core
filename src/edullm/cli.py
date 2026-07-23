"""
Internal module CLI used by the hard-disabled validation workflow.

The user-facing ``edullm`` console entry point remains owned by Task 6.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from edullm.automation import AutomationResult, load_team_leads, validate_issue
from edullm.github import GitHubClient, GitHubError
from edullm.policy import load_policy

ValidationRunner = Callable[..., AutomationResult]


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if str(parsed) != value or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m edullm.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    automation = commands.add_parser("automation")
    automation_commands = automation.add_subparsers(
        dest="automation_command",
        required=True,
    )
    validate = automation_commands.add_parser("validate")
    validate.add_argument("--issue", type=_positive_integer, required=True)
    return parser


def automation_validate(
    issue_number: int,
    *,
    token: str,
    repository: str,
    root: Path,
) -> AutomationResult:
    """
    Load tracked controls and validate one GitHub Issue.

    :param issue_number: The positive Issue number.
    :param token: The workflow-provided GitHub token.
    :param repository: The workflow-provided ``owner/name`` repository.
    :param root: The checked-out repository root.

    :returns: The validation automation result.
    """
    config = root / "config/edullm"
    policy = load_policy(
        config / "policy.yaml",
        config / "entrypoints.yaml",
    )
    reviewers = load_team_leads(config / "team-leads.yaml")
    github = GitHubClient(token, repository)
    validated_at = datetime.now(timezone.utc).replace(microsecond=0)
    return validate_issue(
        issue_number,
        github=github,
        policy=policy,
        allowed_reviewers=reviewers,
        validated_at=validated_at,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    validation_runner: ValidationRunner = automation_validate,
) -> int:
    """
    Run the internal ``automation validate`` module command.

    :param argv: Optional command arguments without the module name.
    :param environ: Optional environment mapping for tests.
    :param validation_runner: Optional validation dependency for tests.

    :returns: A process exit status.
    """
    arguments = _parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    token = environment.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2
    repository = environment.get("GITHUB_REPOSITORY")
    if not repository:
        print("GITHUB_REPOSITORY is required", file=sys.stderr)
        return 2

    try:
        result = validation_runner(
            arguments.issue,
            token=token,
            repository=repository,
            root=Path.cwd(),
        )
    except (GitHubError, OSError, ValueError):
        print(
            "eduLLM validation configuration or GitHub access failed",
            file=sys.stderr,
        )
        return 1

    if result.operational_error:
        print(result.errors[0], file=sys.stderr)
        return 1
    print(f"eduLLM Issue #{arguments.issue}: {result.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
