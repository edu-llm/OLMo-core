"""
Fail-closed validation orchestration for eduLLM Issue requests.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Set
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from edullm.github import (
    GitHubError,
    GitHubValidationError,
    IssueComment,
    normalize_actor_login,
)
from edullm.models import JobRequest, JobStatus
from edullm.policy import Policy
from edullm.request_parser import IssueParseError, parse_issue
from edullm.validation import (
    STATUS_MARKER,
    StatusCommentError,
    build_status_comment,
    validate_request,
)

VALIDATION_MARKER = "<!-- edullm-validation:v1 -->"

_TEAM_LEADS_FIELD = "team_leads"
_MANAGED_STATUS_LABELS = frozenset(f"status:{status.value}" for status in JobStatus)
_SAFE_REVIEW_REASONS = frozenset(
    {
        "malformed GitHub pull evidence",
        "no qualifying pull request has the requested SHA as its head",
        "malformed GitHub review evidence",
        "an authorized reviewer currently requests changes",
        "no authorized reviewer approved the requested SHA",
        "malformed GitHub check evidence",
        "malformed GitHub contents evidence",
        "script does not exist at the requested SHA",
    }
)


class AutomationStateError(RuntimeError):
    """A sanitized fail-closed machine-comment state error."""


@dataclass(frozen=True)
class ValidationDecision:
    """Immutable result of local policy and reviewed-commit validation."""

    status: str
    errors: tuple[str, ...]


@dataclass(frozen=True)
class AutomationResult:
    """Immutable result returned by one Issue validation attempt."""

    status: str
    errors: tuple[str, ...]
    operational_error: bool


def load_team_leads(path: Path) -> frozenset[str]:
    """
    Load the protected, explicit team-lead login allowlist.

    The exact schema is ``team_leads: [<login>, ...]``. An empty list is a
    valid disabled configuration; validation will not become ready until at
    least one protected login is present.

    :param path: The tracked team-lead YAML path.

    :returns: Case-normalized immutable GitHub actor logins.

    :raises ValueError: If YAML, schema, a login, or uniqueness is invalid.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("team-leads: invalid YAML") from None
    except OSError:
        raise ValueError("team-leads: configuration cannot be read") from None
    if type(document) is not dict or set(document) != {_TEAM_LEADS_FIELD}:
        raise ValueError("team-leads: expected only the team_leads field")
    values = cast(dict[object, object], document)[_TEAM_LEADS_FIELD]
    if type(values) is not list:
        raise ValueError("team-leads: team_leads must be a list")

    normalized: set[str] = set()
    for index, value in enumerate(cast(list[object], values)):
        try:
            login = normalize_actor_login(value)
        except GitHubValidationError:
            raise ValueError(f"team-leads: team_leads[{index}] is invalid") from None
        if login in normalized:
            raise ValueError("team-leads: team_leads contains duplicate logins")
        normalized.add(login)
    return frozenset(normalized)


def validation_decision(
    request: JobRequest,
    *,
    policy: Policy,
    github: Any,
    allowed_reviewers: Set[str],
) -> ValidationDecision:
    """
    Validate local request safety before consulting reviewed-commit evidence.

    :param request: The parsed immutable request.
    :param policy: The trusted queue policy.
    :param github: A :class:`~edullm.github.GitHubClient`-compatible client.
    :param allowed_reviewers: Protected explicit team-lead logins.

    :returns: A deterministic immutable ready/requested decision.
    """
    local_errors = validate_request(request, policy)
    if local_errors:
        return ValidationDecision("requested", tuple(local_errors))

    try:
        reviewers = frozenset(normalize_actor_login(value) for value in allowed_reviewers)
    except (GitHubValidationError, TypeError):
        return ValidationDecision(
            "requested",
            ("protected team lead configuration is invalid",),
        )
    if not reviewers:
        return ValidationDecision(
            "requested",
            ("validation is disabled until protected team leads are configured",),
        )

    review = github.reviewed_commit(
        request.commit_sha,
        script_path=request.script_path,
        allowed_reviewers=set(reviewers),
        required_checks=set(policy.required_checks),
    )
    if review.approved:
        return ValidationDecision("ready", ())
    return ValidationDecision("requested", (_sanitized_review_reason(review.reason),))


def validate_issue(
    issue_number: int,
    *,
    github: Any,
    policy: Policy,
    allowed_reviewers: Set[str],
    validated_at: datetime,
) -> AutomationResult:
    """
    Validate one current Issue and persist its fail-closed machine state.

    ``status:ready`` is invalidated before parsing or remote review. A canonical
    status comment is persisted only after the current Issue is re-fetched and
    re-parsed to the same digest, and the ready label is written only after that
    comment succeeds.

    :param issue_number: The trusted positive Issue number.
    :param github: A :class:`~edullm.github.GitHubClient`-compatible client.
    :param policy: The trusted queue policy.
    :param allowed_reviewers: Protected explicit team-lead logins.
    :param validated_at: The canonical validation timestamp.

    :returns: The immutable validation attempt result.
    """
    try:
        initial = github.fetch_issue(issue_number)
        requested_labels = _transition_labels(initial.labels, JobStatus.REQUESTED)
        github.replace_issue_labels(issue_number, requested_labels)

        try:
            request = parse_issue(
                initial.body,
                issue_number=issue_number,
                requester=initial.requester,
            )
        except IssueParseError as error:
            errors = tuple(error.errors)
            _persist_validation_errors(github, issue_number, errors)
            return AutomationResult("requested", errors, False)

        decision = validation_decision(
            request,
            policy=policy,
            github=github,
            allowed_reviewers=allowed_reviewers,
        )
        if decision.errors:
            _persist_validation_errors(github, issue_number, decision.errors)
            return AutomationResult("requested", decision.errors, False)

        status_body = build_status_comment(request, validated_at=validated_at)
        current = github.fetch_issue(issue_number)
        try:
            current_request = parse_issue(
                current.body,
                issue_number=issue_number,
                requester=current.requester,
            )
        except IssueParseError:
            current_request = None
        if (
            current_request is None
            or current_request.digest != request.digest
            or current_request.canonical_json() != request.canonical_json()
        ):
            errors = ("Issue changed during validation; submit or save the current request again",)
            _persist_validation_errors(github, issue_number, errors)
            return AutomationResult("requested", errors, False)

        _persist_machine_comment(
            github,
            issue_number,
            marker=STATUS_MARKER,
            kind="status",
            body=status_body,
        )
        ready_labels = _transition_labels(current.labels, JobStatus.READY)
        github.replace_issue_labels(issue_number, ready_labels)
        return AutomationResult("ready", (), False)
    except AutomationStateError as error:
        return AutomationResult("requested", (str(error),), True)
    except (GitHubError, StatusCommentError):
        return AutomationResult(
            "requested",
            ("GitHub validation operation failed",),
            True,
        )


def _sanitized_review_reason(reason: object) -> str:
    if type(reason) is not str:
        return "commit review evidence was not accepted"
    if reason in _SAFE_REVIEW_REASONS or reason.startswith(
        "required checks are not uniquely successful: "
    ):
        return reason
    return "commit review evidence was not accepted"


def _transition_labels(labels: Iterable[str], status: JobStatus) -> tuple[str, ...]:
    preserved = {label for label in labels if label not in _MANAGED_STATUS_LABELS}
    preserved.add(f"status:{status.value}")
    return tuple(sorted(preserved))


def _persist_validation_errors(
    github: Any,
    issue_number: int,
    errors: tuple[str, ...],
) -> None:
    escaped = tuple(html.escape(error, quote=False) for error in errors)
    body = (
        f"{VALIDATION_MARKER}\n"
        "### eduLLM validation\n\n"
        "This request is not ready:\n" + "\n".join(f"- {error}" for error in escaped)
    )
    _persist_machine_comment(
        github,
        issue_number,
        marker=VALIDATION_MARKER,
        kind="validation",
        body=body,
    )


def _persist_machine_comment(
    github: Any,
    issue_number: int,
    *,
    marker: str,
    kind: str,
    body: str,
) -> None:
    comments = github.list_issue_comments(issue_number)
    existing = _find_machine_comment(comments, marker=marker, kind=kind)
    if existing is None:
        persisted = github.create_issue_comment(issue_number, body)
    else:
        persisted = github.update_issue_comment(existing.id, body)
    if not persisted.author_is_bot or persisted.body != body:
        raise AutomationStateError(f"persisted eduLLM {kind} comment is invalid")


def _find_machine_comment(
    comments: Iterable[IssueComment],
    *,
    marker: str,
    kind: str,
) -> IssueComment | None:
    marker_line = re.compile(rf"^{re.escape(marker)}$", re.MULTILINE)
    matches: list[IssueComment] = []
    for comment in comments:
        count = len(marker_line.findall(comment.body))
        if count > 1:
            raise AutomationStateError(f"multiple eduLLM {kind} markers were found")
        if count == 1:
            matches.append(comment)
    if len(matches) > 1:
        raise AutomationStateError(f"multiple eduLLM {kind} comments were found")
    if not matches:
        return None
    if not matches[0].author_is_bot:
        raise AutomationStateError(f"eduLLM {kind} marker is not bot-authored")
    return matches[0]
