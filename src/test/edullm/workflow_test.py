from pathlib import Path

import yaml

WORKFLOWS = Path(".github/workflows")


def test_every_edullm_workflow_remains_literally_disabled_and_sha_pinned():
    paths = sorted(WORKFLOWS.glob("edullm-*.yml"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "${{ false &&" in text
        assert "persist-credentials: false" in text
        for line in text.splitlines():
            if line.strip().startswith("- uses:"):
                reference = line.split("@", 1)[1]
                assert len(reference) == 40
                assert all(character in "0123456789abcdef" for character in reference)


def test_terminal_notification_workflow_has_least_privilege_and_scoped_secret():
    path = WORKFLOWS / "edullm-terminal-notify.yml"
    text = path.read_text(encoding="utf-8")
    document = yaml.load(text, Loader=yaml.BaseLoader)

    assert document["permissions"] == {"contents": "read", "issues": "write"}
    assert document["concurrency"] == {
        "group": "edullm-terminal-notification",
        "cancel-in-progress": "false",
    }
    steps = document["jobs"]["notify"]["steps"]
    secret_steps = [step for step in steps if "SLACK_WEBHOOK_URL" in step.get("env", {})]
    assert len(secret_steps) == 1
    assert secret_steps[0]["run"] == "python -m edullm.cli automation terminal"
    assert "SLACK_WEBHOOK_URL" not in "\n".join(step.get("run", "") for step in steps[:-1])
