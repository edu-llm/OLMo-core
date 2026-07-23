import json
import re
from pathlib import Path

import yaml

from edullm.policy import load_policy
from edullm.request_parser import ISSUE_HEADINGS, fields_from_markdown, parse_issue
from edullm.validation import validate_request

ISSUE_FORM = Path(".github/ISSUE_TEMPLATE/edullm-job-request.yml")
ISSUE_CONFIG = Path(".github/ISSUE_TEMPLATE/config.yml")
WORKFLOW = Path(".github/workflows/edullm-validate.yml")
MAIN_WORKFLOW = Path(".github/workflows/main.yml")
CODEOWNERS = Path(".github/CODEOWNERS")
RULESET = Path("config/edullm/main-ruleset.json")
TEAM_LEADS = Path("config/edullm/team-leads.yaml")
FIXTURE = Path("src/test/edullm/fixtures/valid_issue.md")


def _load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _workflow_trigger(document):
    # PyYAML's YAML 1.1 resolver treats the plain key "on" as boolean true.
    return document.get("on", document.get(True))


def test_main_ci_has_one_required_edullm_core_check():
    workflow = _load_yaml(MAIN_WORKFLOW)
    names = [row["name"] for row in workflow["jobs"]["checks"]["strategy"]["matrix"]["task"]]
    assert names.count("Test eduLLM core") == 1

    policy = load_policy(Path("config/edullm/policy.yaml"))
    assert policy.required_checks.count("Test eduLLM core") == 1


def test_issue_form_exactly_matches_parser_headings_and_production_validation():
    form = _load_yaml(ISSUE_FORM)
    fields = [item for item in form["body"] if item["type"] != "markdown"]
    headings = tuple(item["attributes"]["label"] for item in fields)

    assert headings == ISSUE_HEADINGS
    assert "Requester" not in headings
    assert all(item["validations"] == {"required": True} for item in fields)

    fixture_fields = fields_from_markdown(FIXTURE.read_text(encoding="utf-8"))
    rendered = "\n\n".join(
        f"### {item['attributes']['label']}\n" f"{fixture_fields[item['attributes']['label']]}"
        for item in fields
    )
    request = parse_issue(rendered, issue_number=42, requester="student")
    policy = load_policy(Path("config/edullm/policy.yaml"))

    assert validate_request(request, policy) == []


def test_issue_form_has_exact_labels_and_required_dropdown_contracts():
    form = _load_yaml(ISSUE_FORM)
    fields = {
        item["attributes"]["label"]: item for item in form["body"] if item["type"] != "markdown"
    }

    assert form["labels"] == ["edullm-job", "status:requested"]
    assert fields["Launcher"]["type"] == "dropdown"
    assert fields["Launcher"]["attributes"]["options"] == [
        "python",
        "torchrun",
        "bash",
    ]
    assert fields["GPU count"]["type"] == "dropdown"
    assert fields["GPU count"]["attributes"]["options"] == ["1", "2"]
    assert fields["GPU preference"]["type"] == "dropdown"
    assert fields["GPU preference"]["attributes"]["options"] == [
        "any",
        "l40s",
        "h100",
        "h200",
    ]
    assert fields["W&B project"]["type"] == "dropdown"
    assert fields["W&B project"]["attributes"]["options"] == [
        "test",
        "pretraining",
        "posttraining",
        "evaluation",
        "data-pipeline",
    ]
    assert fields["Data classification"]["type"] == "dropdown"
    assert fields["Data classification"]["attributes"]["options"] == [
        "public",
        "research-cleared",
    ]
    arguments = fields["Arguments JSON"]
    assert arguments["type"] == "textarea"
    assert "ordered JSON string array" in arguments["attributes"]["description"]
    assert arguments["attributes"]["placeholder"].startswith("[")


def test_issue_template_config_disables_blank_issues_without_invented_links():
    config = _load_yaml(ISSUE_CONFIG)

    assert config == {
        "blank_issues_enabled": False,
        "contact_links": [],
    }


def test_workflow_is_hard_disabled_with_exact_triggers_and_issue_filter():
    text = WORKFLOW.read_text(encoding="utf-8")
    workflow = _load_yaml(WORKFLOW)
    trigger = _workflow_trigger(workflow)
    job = workflow["jobs"]["validate"]

    assert trigger == {"issues": {"types": ["opened", "edited", "reopened"]}}
    assert "false" in job["if"]
    assert "contains(github.event.issue.labels.*.name, 'edullm-job')" in job["if"]
    assert "vars." not in job["if"]
    assert "secrets." not in job["if"]
    assert "${{ false &&" in text


def test_workflow_has_per_issue_concurrency_and_exact_least_privilege():
    workflow = _load_yaml(WORKFLOW)

    assert "github.event.issue.number" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] is True
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "write",
        "pull-requests": "read",
        "checks": "read",
    }


def test_workflow_pins_actions_and_passes_only_quoted_issue_environment():
    workflow = _load_yaml(WORKFLOW)
    steps = workflow["jobs"]["validate"]["steps"]
    checkout = steps[0]
    setup_python = steps[1]
    validation = steps[-1]

    checkout_action, checkout_sha = checkout["uses"].split("@", 1)
    setup_action, setup_sha = setup_python["uses"].split("@", 1)
    assert checkout_action == "actions/checkout"
    assert setup_action == "actions/setup-python"
    assert re.fullmatch(r"[0-9a-f]{40}", checkout_sha)
    assert re.fullmatch(r"[0-9a-f]{40}", setup_sha)
    assert checkout["with"]["persist-credentials"] is False
    assert setup_python["with"]["python-version"] == "3.11"

    assert "${{" not in validation["run"]
    assert '--issue "$ISSUE_NUMBER"' in validation["run"]
    assert validation["env"]["ISSUE_NUMBER"] == "${{ github.event.issue.number }}"
    assert validation["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert validation["env"]["GITHUB_REPOSITORY"] == "${{ github.repository }}"
    assert not any("gh " in step.get("run", "") for step in steps)
    assert "id-token" not in workflow["permissions"]
    assert "actions" not in workflow["permissions"]
    assert "packages" not in workflow["permissions"]


def test_codeowners_protects_all_queue_controls_with_team_leads():
    lines = {
        line
        for line in CODEOWNERS.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert lines == {
        "/config/edullm/ @edu-llm/team-leads",
        "/.github/CODEOWNERS @edu-llm/team-leads",
        "/.github/ISSUE_TEMPLATE/ @edu-llm/team-leads",
        "/.github/workflows/edullm-* @edu-llm/team-leads",
        "/src/edullm/ @edu-llm/team-leads",
        "/.cursor/skills/submit-edullm-job/ @edu-llm/team-leads",
    }
    assert "compute-operators" not in "\n".join(lines)


def test_ruleset_matches_policy_checks_and_has_no_bypass():
    ruleset = json.loads(RULESET.read_text(encoding="utf-8"))
    policy = load_policy(Path("config/edullm/policy.yaml"))

    assert ruleset["name"] == "Protect main and queue controls"
    assert ruleset["target"] == "branch"
    assert ruleset["enforcement"] == "active"
    assert ruleset["bypass_actors"] == []
    assert ruleset["conditions"] == {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}}

    rules = {rule["type"]: rule for rule in ruleset["rules"]}
    assert set(rules) == {
        "deletion",
        "non_fast_forward",
        "pull_request",
        "required_status_checks",
    }
    assert rules["deletion"] == {"type": "deletion"}
    assert rules["non_fast_forward"] == {"type": "non_fast_forward"}
    assert rules["pull_request"]["parameters"] == {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": True,
        "required_review_thread_resolution": True,
        "require_code_owner_review": True,
        "require_last_push_approval": True,
    }

    status_parameters = rules["required_status_checks"]["parameters"]
    assert status_parameters["strict_required_status_checks_policy"] is True
    assert status_parameters["do_not_enforce_on_create"] is False
    contexts = [item["context"] for item in status_parameters["required_status_checks"]]
    assert contexts == list(policy.required_checks)
    assert "Test edullm queue" not in contexts


def test_production_team_leads_file_is_explicitly_empty():
    assert _load_yaml(TEAM_LEADS) == {"team_leads": []}
