from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from edullm.automation import (
    VALIDATION_MARKER,
    AutomationResult,
    ValidationDecision,
    load_team_leads,
    validate_issue,
    validation_decision,
)
from edullm.github import GitHubAPIError, GitHubIssue, IssueComment, ReviewResult
from edullm.validation import STATUS_MARKER, parse_status_comment

VALIDATED_AT = datetime(2026, 7, 23, 6, 0, 0, tzinfo=timezone.utc)
BODY = Path("src/test/edullm/fixtures/valid_issue.md").read_text(encoding="utf-8")


class RecordingReviewGitHub:
    def __init__(self, result=None):
        self.result = result or ReviewResult(True, 7, "approved")
        self.calls = []

    def reviewed_commit(
        self,
        sha,
        *,
        script_path,
        allowed_reviewers,
        required_checks,
    ):
        self.calls.append((sha, script_path, allowed_reviewers, required_checks))
        return self.result


class AutomationGitHub(RecordingReviewGitHub):
    def __init__(
        self,
        issues,
        *,
        comments=(),
        result=None,
        review_error=None,
        create_error=None,
        update_error=None,
        labels_error=None,
    ):
        super().__init__(result)
        self.issues = list(issues)
        self.comments = list(comments)
        self.review_error = review_error
        self.create_error = create_error
        self.update_error = update_error
        self.labels_error = labels_error
        self.events = []

    def fetch_issue(self, issue_number):
        self.events.append(("fetch", issue_number))
        if len(self.issues) > 1:
            return self.issues.pop(0)
        return self.issues[0]

    def replace_issue_labels(self, issue_number, labels):
        self.events.append(("labels", issue_number, tuple(labels)))
        if self.labels_error is not None:
            raise self.labels_error
        return tuple(labels)

    def reviewed_commit(self, *args, **kwargs):
        self.events.append(("review", args[0]))
        if self.review_error is not None:
            raise self.review_error
        return super().reviewed_commit(*args, **kwargs)

    def list_issue_comments(self, issue_number):
        self.events.append(("comments", issue_number))
        return tuple(self.comments)

    def create_issue_comment(self, issue_number, body):
        self.events.append(("create", issue_number, body))
        if self.create_error is not None:
            raise self.create_error
        comment = IssueComment(
            id=99,
            body=body,
            author="github-actions[bot]",
            author_is_bot=True,
        )
        self.comments.append(comment)
        return comment

    def update_issue_comment(self, comment_id, body):
        self.events.append(("update", comment_id, body))
        if self.update_error is not None:
            raise self.update_error
        comment = IssueComment(
            id=comment_id,
            body=body,
            author="github-actions[bot]",
            author_is_bot=True,
        )
        self.comments = [
            comment if existing.id == comment_id else existing for existing in self.comments
        ]
        return comment


def _issue(
    *,
    body=BODY,
    requester="student",
    labels=("edullm-job", "status:ready", "research"),
):
    return GitHubIssue(
        number=42,
        body=body,
        requester=requester,
        labels=labels,
    )


def _labels_events(github):
    return [event for event in github.events if event[0] == "labels"]


def test_load_team_leads_normalizes_users_and_supported_bots(tmp_path):
    path = tmp_path / "team-leads.yaml"
    path.write_text(
        "team_leads:\n  - Team-Lead\n  - Review-App[bot]\n",
        encoding="utf-8",
    )

    assert load_team_leads(path) == frozenset({"team-lead", "review-app[bot]"})


def test_empty_production_team_lead_allowlist_is_valid_but_disabled():
    assert load_team_leads(Path("config/edullm/team-leads.yaml")) == frozenset()


@pytest.mark.parametrize(
    "document",
    [
        "",
        "[]\n",
        "unknown: []\n",
        "team_leads: {}\n",
        "team_leads: [operator]\nextra: true\n",
        "team_leads: ['']\n",
        "team_leads: [' operator']\n",
        "team_leads: ['operator/other']\n",
        "team_leads: ['[bot]']\n",
        "team_leads: [operator, Operator]\n",
        "team_leads: [review-app[bot], Review-App[BOT]]\n",
        "team_leads: [7]\n",
    ],
)
def test_load_team_leads_rejects_malformed_or_duplicate_configuration(tmp_path, document):
    path = tmp_path / "team-leads.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="team-leads"):
        load_team_leads(path)


def test_load_team_leads_redacts_yaml_parser_details(tmp_path):
    secret = "ghp_DO_NOT_ECHO_THIS_SECRET"
    path = tmp_path / "team-leads.yaml"
    path.write_text(f"team_leads: [operator\n{secret}", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        load_team_leads(path)

    assert secret not in str(raised.value)


def test_valid_request_uses_task_3_review_contract(valid_request, policy, reviewed_github):
    decision = validation_decision(
        valid_request,
        policy=policy,
        github=reviewed_github,
        allowed_reviewers={"operator"},
    )

    assert decision == ValidationDecision(status="ready", errors=())


def test_validation_decision_passes_exact_script_reviewers_and_policy_checks(valid_request, policy):
    github = RecordingReviewGitHub()

    validation_decision(
        valid_request,
        policy=policy,
        github=github,
        allowed_reviewers={"team-lead"},
    )

    assert github.calls == [
        (
            valid_request.commit_sha,
            valid_request.script_path,
            {"team-lead"},
            set(policy.required_checks),
        )
    ]


def test_local_errors_short_circuit_github_review(valid_request, policy):
    github = RecordingReviewGitHub()
    request = replace(valid_request, commit_sha="main")

    decision = validation_decision(
        request,
        policy=policy,
        github=github,
        allowed_reviewers={"team-lead"},
    )

    assert decision.status == "requested"
    assert decision.errors == ("commit SHA must be 40 lowercase hexadecimal characters",)
    assert github.calls == []


def test_restricted_request_is_rejected_before_github_review(valid_request, policy):
    github = RecordingReviewGitHub()
    request = replace(valid_request, data_classification="restricted")

    decision = validation_decision(
        request,
        policy=policy,
        github=github,
        allowed_reviewers={"team-lead"},
    )

    assert decision.errors == ("restricted data is not accepted by the public pilot queue",)
    assert github.calls == []


def test_empty_protected_reviewer_set_fails_closed_without_review_call(valid_request, policy):
    github = RecordingReviewGitHub()

    decision = validation_decision(
        valid_request,
        policy=policy,
        github=github,
        allowed_reviewers=frozenset(),
    )

    assert decision == ValidationDecision(
        "requested",
        ("validation is disabled until protected team leads are configured",),
    )
    assert github.calls == []


def test_review_rejection_reason_is_returned_without_request_data(valid_request, policy):
    github = RecordingReviewGitHub(
        ReviewResult(False, None, "no authorized reviewer approved the requested SHA")
    )

    decision = validation_decision(
        valid_request,
        policy=policy,
        github=github,
        allowed_reviewers={"team-lead"},
    )

    assert decision.errors == ("no authorized reviewer approved the requested SHA",)
    assert valid_request.purpose not in "\n".join(decision.errors)
    assert not any(argument in "\n".join(decision.errors) for argument in valid_request.argv)


def test_decision_and_automation_results_are_immutable():
    decision = ValidationDecision("requested", ("not ready",))
    result = AutomationResult("requested", ("not ready",), False)

    with pytest.raises(FrozenInstanceError):
        decision.status = "ready"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = "ready"  # type: ignore[misc]


def test_invalid_issue_is_requested_and_updates_one_sanitized_validation_comment(
    policy,
):
    existing = IssueComment(
        id=5,
        body=f"{VALIDATION_MARKER}\nold",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))],
        comments=[existing],
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is False
    assert github.calls == []
    assert _labels_events(github) == [
        ("labels", 42, ("edullm-job", "research", "status:requested"))
    ]
    updates = [event for event in github.events if event[0] == "update"]
    assert len(updates) == 1
    assert updates[0][1] == 5
    assert updates[0][2].startswith(VALIDATION_MARKER + "\n")
    assert "missing heading: Purpose" in updates[0][2]
    assert BODY not in updates[0][2]


def test_valid_issue_persists_canonical_status_before_ready_and_preserves_labels(
    policy,
):
    github = AutomationGitHub([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult("ready", (), False)
    creates = [event for event in github.events if event[0] == "create"]
    assert len(creates) == 1
    status = parse_status_comment(creates[0][2])
    assert status.request.requester == "student"
    assert status.validated_at == VALIDATED_AT
    assert github.events.index(("review", "a" * 40)) > github.events.index(
        ("labels", 42, ("edullm-job", "research", "status:requested"))
    )
    assert github.events.index(creates[0]) < github.events.index(
        ("labels", 42, ("edullm-job", "research", "status:ready"))
    )


@pytest.mark.parametrize(
    "second_issue",
    [
        _issue(body=BODY.replace("Skill-DAG smoke", "edited purpose", 1)),
        _issue(requester="different-user"),
    ],
)
def test_issue_edit_or_requester_race_fails_closed(policy, second_issue):
    github = AutomationGitHub([_issue(), second_issue])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.errors == (
        "Issue changed during validation; submit or save the current request again",
    )
    assert not any(event[0] == "create" and STATUS_MARKER in event[2] for event in github.events)
    assert _labels_events(github) == [
        ("labels", 42, ("edullm-job", "research", "status:requested"))
    ]


def test_duplicate_status_comments_are_rejected_without_selecting_one(policy, valid_request):
    comments = [
        IssueComment(
            id=index,
            body=f"{STATUS_MARKER}\n{{}}",
            author="github-actions[bot]",
            author_is_bot=True,
        )
        for index in (5, 6)
    ]
    github = AutomationGitHub([_issue(), _issue()], comments=comments)

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM status comments were found",)
    assert not any(event[0] in {"create", "update"} for event in github.events)
    assert all("status:ready" not in event[2] for event in _labels_events(github))


def test_duplicate_status_markers_in_one_comment_are_rejected(policy):
    comment = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}\n{STATUS_MARKER}",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[comment])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM status markers were found",)


def test_human_authored_machine_marker_is_rejected(policy):
    comment = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}",
        author="student",
        author_is_bot=False,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[comment])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("eduLLM status marker is not bot-authored",)


def test_idempotent_retry_updates_existing_status_comment_without_duplication(
    policy,
):
    existing = IssueComment(
        id=5,
        body=f"{STATUS_MARKER}\n{{}}",
        author="github-actions[bot]",
        author_is_bot=True,
    )
    github = AutomationGitHub([_issue(), _issue()], comments=[existing])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.status == "ready"
    assert len([event for event in github.events if event[0] == "update"]) == 1
    assert not any(event[0] == "create" for event in github.events)
    assert len(github.comments) == 1
    parse_status_comment(github.comments[0].body)


def test_duplicate_validation_comments_fail_closed_on_invalid_request(policy):
    comments = [
        IssueComment(
            id=index,
            body=f"{VALIDATION_MARKER}\nold",
            author="github-actions[bot]",
            author_is_bot=True,
        )
        for index in (5, 6)
    ]
    github = AutomationGitHub(
        [_issue(body=BODY.replace("### Purpose", "## Purpose", 1))],
        comments=comments,
    )

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.operational_error is True
    assert result.errors == ("multiple eduLLM validation comments were found",)
    assert not any(event[0] in {"create", "update"} for event in github.events)


@pytest.mark.parametrize(
    "failure_field",
    ["review_error", "create_error", "update_error"],
)
def test_github_failures_never_transition_to_ready(policy, failure_field):
    kwargs = {failure_field: GitHubAPIError("GitHub API request failed")}
    comments = ()
    if failure_field == "update_error":
        comments = (
            IssueComment(
                id=5,
                body=f"{STATUS_MARKER}\n{{}}",
                author="github-actions[bot]",
                author_is_bot=True,
            ),
        )
    github = AutomationGitHub([_issue(), _issue()], comments=comments, **kwargs)

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result.status == "requested"
    assert result.operational_error is True
    assert result.errors == ("GitHub validation operation failed",)
    assert all("status:ready" not in event[2] for event in _labels_events(github))


def test_ready_label_failure_reports_operational_error_after_status_persistence(
    policy,
):
    class FailSecondLabelUpdate(AutomationGitHub):
        def replace_issue_labels(self, issue_number, labels):
            if any(label == "status:ready" for label in labels):
                raise GitHubAPIError("sensitive API response")
            return super().replace_issue_labels(issue_number, labels)

    github = FailSecondLabelUpdate([_issue(), _issue()])

    result = validate_issue(
        42,
        github=github,
        policy=policy,
        allowed_reviewers={"team-lead"},
        validated_at=VALIDATED_AT,
    )

    assert result == AutomationResult(
        "requested",
        ("GitHub validation operation failed",),
        True,
    )
    assert any(event[0] == "create" and STATUS_MARKER in event[2] for event in github.events)
    assert _labels_events(github) == [
        ("labels", 42, ("edullm-job", "research", "status:requested"))
    ]
