from pathlib import Path

import pytest

import edullm.cli as cli
from edullm.assignment import AssignmentResult
from edullm.automation import AutomationResult


class RestrictedEnvironment(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        if key not in {"GITHUB_TOKEN", "GITHUB_REPOSITORY"}:
            raise AssertionError(f"unexpected environment read: {key}")
        return super().get(key, default)


def test_module_cli_accepts_automation_validate_and_positive_issue(tmp_path, monkeypatch, capsys):
    environment = RestrictedEnvironment(
        {
            "GITHUB_TOKEN": "secret-token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        }
    )
    calls = []

    def runner(issue_number, *, token, repository, root):
        calls.append((issue_number, token, repository, root))
        return AutomationResult("ready", (), False)

    monkeypatch.chdir(tmp_path)
    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ=environment,
        validation_runner=runner,
    )

    assert exit_code == 0
    assert calls == [(42, "secret-token", "edu-llm/OLMo-core", tmp_path)]
    assert environment.reads == ["GITHUB_TOKEN", "GITHUB_REPOSITORY"]
    output = capsys.readouterr()
    assert output.out == "eduLLM Issue #42: ready\n"
    assert "secret-token" not in output.out + output.err


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["validate", "--issue", "42"],
        ["automation", "validate"],
        ["automation", "validate", "--issue", "0"],
        ["automation", "validate", "--issue", "-1"],
        ["automation", "validate", "--issue", "1.5"],
        ["automation", "submit", "--issue", "42"],
    ],
)
def test_module_cli_rejects_missing_commands_or_nonpositive_issue(arguments):
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments, environ={})

    assert raised.value.code == 2


@pytest.mark.parametrize(
    "environment,message",
    [
        (
            {"GITHUB_REPOSITORY": "edu-llm/OLMo-core"},
            "GITHUB_TOKEN is required",
        ),
        (
            {"GITHUB_TOKEN": "token"},
            "GITHUB_REPOSITORY is required",
        ),
    ],
)
def test_cli_requires_only_expected_github_environment(environment, message, capsys):
    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ=RestrictedEnvironment(environment),
    )

    assert exit_code == 2
    output = capsys.readouterr()
    assert message in output.err


def test_cli_operational_failure_is_sanitized_and_nonzero(capsys):
    token = "top-secret-token"

    def runner(issue_number, *, token, repository, root):  # noqa: ARG001
        return AutomationResult(
            "requested",
            ("GitHub validation operation failed",),
            True,
        )

    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ={
            "GITHUB_TOKEN": token,
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        validation_runner=runner,
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert "GitHub validation operation failed" in output.err
    assert token not in output.out + output.err


def test_invalid_request_is_a_handled_requested_result(capsys):
    def runner(issue_number, *, token, repository, root):  # noqa: ARG001
        return AutomationResult(
            "requested",
            ("missing heading: Purpose",),
            False,
        )

    exit_code = cli.main(
        ["automation", "validate", "--issue", "42"],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        validation_runner=runner,
    )

    assert exit_code == 0
    output = capsys.readouterr()
    assert output.out == "eduLLM Issue #42: requested\n"


def test_default_runner_loads_only_tracked_configuration(tmp_path, monkeypatch):
    loaded = []
    fake_policy = object()
    fake_reviewers = frozenset({"lead"})
    fake_client = object()
    expected_result = AutomationResult("ready", (), False)

    def fake_load_policy(policy_path, entrypoints_path=None):
        loaded.append((policy_path, entrypoints_path))
        return fake_policy

    def fake_load_team_leads(path):
        loaded.append(path)
        return fake_reviewers

    def fake_client_factory(token, repository):
        assert token == "token"
        assert repository == "edu-llm/OLMo-core"
        return fake_client

    def fake_validate_issue(
        issue_number,
        *,
        github,
        policy,
        allowed_reviewers,
        validated_at,
    ):
        assert issue_number == 42
        assert github is fake_client
        assert policy is fake_policy
        assert allowed_reviewers is fake_reviewers
        assert validated_at.tzinfo is not None
        return expected_result

    monkeypatch.setattr(cli, "load_policy", fake_load_policy)
    monkeypatch.setattr(cli, "load_team_leads", fake_load_team_leads)
    monkeypatch.setattr(cli, "GitHubClient", fake_client_factory)
    monkeypatch.setattr(cli, "validate_issue", fake_validate_issue)

    result = cli.automation_validate(
        42,
        token="token",
        repository="edu-llm/OLMo-core",
        root=tmp_path,
    )

    assert result is expected_result
    assert loaded == [
        (
            tmp_path / "config/edullm/policy.yaml",
            tmp_path / "config/edullm/entrypoints.yaml",
        ),
        tmp_path / "config/edullm/team-leads.yaml",
    ]


def test_package_does_not_register_user_facing_edullm_console_entrypoint():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" not in pyproject
    assert "edullm =" not in pyproject


@pytest.mark.parametrize(
    "command,runner_name,expected_output",
    [
        (
            "assign",
            "assignment_runner",
            "eduLLM assignment scan: 1 result(s)\n",
        ),
        (
            "reminders",
            "reminder_runner",
            "eduLLM reminder scan: 1 result(s)\n",
        ),
    ],
)
def test_internal_cli_exposes_only_hard_disabled_assignment_automation(
    tmp_path,
    monkeypatch,
    capsys,
    command,
    runner_name,
    expected_output,
):
    calls = []

    def runner(*, token, repository, webhook, root):
        calls.append((token, repository, webhook, root))
        return (AssignmentResult(42, "assigned", "alice", False),)

    kwargs = {
        "assignment_runner": lambda **unused: (),
        "reminder_runner": lambda **unused: (),
        runner_name: runner,
    }
    monkeypatch.chdir(tmp_path)

    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/T1/B1/secret",
        },
        **kwargs,
    )

    assert exit_code == 0
    assert calls == [
        (
            "token",
            "edu-llm/OLMo-core",
            "https://hooks.slack.com/services/T1/B1/secret",
            tmp_path,
        )
    ]
    output = capsys.readouterr()
    assert output.out == expected_output
    assert "hooks.slack.com" not in output.out + output.err


@pytest.mark.parametrize("command", ["assign", "reminders"])
def test_assignment_cli_allows_absent_webhook_for_closed_operator_roster(command, capsys):
    def runner(*, token, repository, webhook, root):  # noqa: ARG001
        assert webhook is None
        return ()

    kwargs = {
        "assignment_runner": runner,
        "reminder_runner": runner,
    }
    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        **kwargs,
    )

    assert exit_code == 0
    assert "0 result(s)" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["assign", "reminders"])
def test_assignment_cli_returns_nonzero_for_sanitized_operational_result(command, capsys):
    def runner(**unused):
        return (AssignmentResult(42, "error", None, True),)

    exit_code = cli.main(
        ["automation", command],
        environ={
            "GITHUB_TOKEN": "token",
            "GITHUB_REPOSITORY": "edu-llm/OLMo-core",
        },
        assignment_runner=runner,
        reminder_runner=runner,
    )

    assert exit_code == 1
    output = capsys.readouterr()
    assert "automation operation failed" in output.err
