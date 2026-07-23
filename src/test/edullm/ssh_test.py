import os
import stat
import subprocess

import pytest

import edullm.ssh as ssh


class RecordingRunner:
    def __init__(self, results=None):
        self.calls = []
        self.results = list(results or [])

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_control_block_uses_project_alias_and_one_hour_persist():
    block = ssh.control_block("philote")

    assert block == "\n".join(
        [
            "Host orcd-login",
            "    Hostname orcd-login.mit.edu",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/edullm-%C",
            "    ControlPersist 1h",
            "    User philote",
        ]
    )


@pytest.mark.parametrize("username", ["", "two words", "name;touch /tmp/x", "-option"])
def test_control_block_rejects_unsafe_usernames(username):
    with pytest.raises(ValueError, match="username"):
        ssh.control_block(username)


def test_remote_argv_is_shell_quoted_once_and_has_finite_timeout():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.run_remote(["python", "-c", "print('two words; $HOME')"])

    argv, options = runner.calls[0]
    assert argv == [
        "ssh",
        "orcd-login",
        "python -c 'print('\"'\"'two words; $HOME'\"'\"')'",
    ]
    assert options["timeout"] == ssh.COMMAND_TIMEOUT_SECONDS
    assert options["capture_output"] is True
    assert options["text"] is True
    assert options["check"] is False


def test_direct_check_does_not_depend_on_project_alias():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.run_direct("philote", ["hostname"])

    argv, _ = runner.calls[0]
    assert argv == [
        "ssh",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-l",
        "philote",
        "orcd-login.mit.edu",
        "hostname",
    ]
    assert "orcd-login" not in argv[:-1]


def test_write_remote_sends_file_content_only_via_stdin():
    secret = "literal-super-secret"
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    client.write_remote("~/.config/edullm/wandb.key", secret + "\n")

    argv, options = runner.calls[0]
    assert secret not in repr(argv)
    assert options["input"] == secret + "\n"
    assert options["timeout"] == ssh.COMMAND_TIMEOUT_SECONDS
    assert argv == [
        "ssh",
        "orcd-login",
        "umask 077 && mkdir -p ~/.config/edullm && cat > "
        "~/.config/edullm/wandb.key && chmod 600 ~/.config/edullm/wandb.key",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["ssh", "orcd-login"], 30, output="private-output"),
        OSError("private local path"),
    ],
)
def test_subprocess_failures_are_sanitized(failure):
    runner = RecordingRunner([failure])
    client = ssh.SSHClient(runner=runner)

    with pytest.raises(ssh.SSHError, match="SSH command failed") as raised:
        client.run_remote(["printf", "private-argument"])

    text = str(raised.value)
    assert "private-output" not in text
    assert "private local path" not in text
    assert "private-argument" not in text


def test_nonzero_remote_failure_is_sanitized():
    result = subprocess.CompletedProcess(
        ["ssh"], 23, "captured-private-stdout", "captured-private-stderr"
    )
    client = ssh.SSHClient(runner=RecordingRunner([result]))

    with pytest.raises(ssh.SSHError, match="SSH command failed") as raised:
        client.run_remote(["false"])

    assert "private" not in str(raised.value)


def test_close_master_treats_missing_control_socket_as_already_closed():
    result = subprocess.CompletedProcess(
        ["ssh"], 255, "", "Control socket connect(/tmp/socket): No such file or directory"
    )
    runner = RecordingRunner([result])

    assert ssh.SSHClient(runner=runner).close_master() is False
    assert runner.calls[0][0] == ["ssh", "-O", "exit", "orcd-login"]


def test_close_master_does_not_hide_other_errors():
    result = subprocess.CompletedProcess(["ssh"], 255, "", "Permission denied")

    with pytest.raises(ssh.SSHError, match="could not close"):
        ssh.SSHClient(runner=RecordingRunner([result])).close_master()


def test_plan_adds_control_block_without_changing_unrelated_bytes():
    original = b"# personal config\nHost github.com\n  User git\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.changed is True
    assert plan.proposed.startswith(original)
    assert plan.proposed[len(original) :] == b"\n" + ssh.control_block("philote").encode() + b"\n"
    assert "--- ~/.ssh/config" in plan.redacted_diff
    assert "+++ ~/.ssh/config (proposed)" in plan.redacted_diff


def test_plan_replaces_one_exact_alias_and_preserves_neighbor_blocks():
    before = (
        b"Host first\n  User one\n\n"
        b"Host orcd-login\n  Hostname old.example\n  User old\n\n"
        b"Host last\n  User three\n"
    )

    plan = ssh.plan_control_config(before, "new-user")

    assert plan.proposed.startswith(b"Host first\n  User one\n\n")
    assert plan.proposed.endswith(b"\nHost last\n  User three\n")
    assert plan.proposed.count(b"Host orcd-login") == 1
    assert b"old.example" not in plan.proposed
    assert b"User new-user" in plan.proposed


def test_plan_recognizes_tab_separated_host_directive_without_duplicate():
    original = b"Host\torcd-login\n\tHostname old.example\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.count(b"Host orcd-login") == 1
    assert b"Host\torcd-login" not in plan.proposed


def test_plan_is_idempotent_for_existing_managed_block():
    original = (ssh.control_block("philote") + "\n").encode()

    plan = ssh.plan_control_config(original, "philote")

    assert plan.changed is False
    assert plan.proposed == original
    assert plan.redacted_diff == ""


@pytest.mark.parametrize(
    "original",
    [
        b"Host orcd-login\n  User one\nHost orcd-login\n  User two\n",
        b"Host orcd-login other\n  User one\n",
        b"Host orcd-*\n  User one\n",
        b"Host *\n  ServerAliveInterval 60\n",
        b"ControlMaster no\nHost github.com\n  User git\n",
        b"Match all\n  ControlPersist no\n",
    ],
)
def test_plan_rejects_duplicate_or_ambiguous_matching_host_blocks(original):
    with pytest.raises(ssh.SSHConfigError, match="safely"):
        ssh.plan_control_config(original, "philote")


def test_plan_allows_unrelated_multi_pattern_blocks():
    original = b"Host github.com gitlab.com\n  User git\n"

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.startswith(original)


def test_plan_rejects_include_that_could_hide_conflicting_alias():
    original = b"Include ~/.ssh/config.d/*\nHost github.com\n  User git\n"

    with pytest.raises(ssh.SSHConfigError, match="safely"):
        ssh.plan_control_config(original, "philote")


def test_diff_redacts_sensitive_values_from_replaced_alias():
    original = (
        b"Host orcd-login\n"
        b"  Hostname old.example\n"
        b"  IdentityFile /Users/operator/private-key\n"
        b"  ProxyCommand secret-command --token value\n"
    )

    plan = ssh.plan_control_config(original, "philote")

    assert "private-key" not in plan.redacted_diff
    assert "--token" not in plan.redacted_diff
    assert "IdentityFile <redacted>" in plan.redacted_diff
    assert "ProxyCommand <redacted>" in plan.redacted_diff


def test_secure_apply_creates_backup_and_mode_0600(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n  User git\n"
    config.write_bytes(original)
    config.chmod(0o644)
    plan = ssh.plan_control_config(original, "philote")

    backup = ssh.apply_control_config(config, plan)

    assert backup is not None
    assert backup.read_bytes() == original
    assert config.read_bytes() == plan.proposed
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
    assert stat.S_IMODE(config.parent.stat().st_mode) == 0o700
    assert not list(config.parent.glob(".config.edullm-*"))


def test_secure_apply_is_noop_when_plan_is_unchanged(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = (ssh.control_block("philote") + "\n").encode()
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")

    assert ssh.apply_control_config(config, plan) is None
    assert list(config.parent.iterdir()) == [config]


def test_secure_apply_rejects_symlink_without_touching_target(tmp_path):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    target = tmp_path / "target"
    target.write_bytes(b"untouched")
    config = ssh_dir / "config"
    config.symlink_to(target)
    plan = ssh.plan_control_config(b"", "philote")

    with pytest.raises(ssh.SSHConfigError, match="symbolic link"):
        ssh.apply_control_config(config, plan)

    assert target.read_bytes() == b"untouched"


def test_secure_apply_detects_stale_plan(tmp_path):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    config.write_bytes(b"Host first\n")
    plan = ssh.plan_control_config(config.read_bytes(), "philote")
    config.write_bytes(b"Host changed\n")

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == b"Host changed\n"


def test_failed_atomic_replace_leaves_original_and_secure_backup(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o666)
    plan = ssh.plan_control_config(original, "philote")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ssh.os, "replace", fail_replace)

    with pytest.raises(ssh.SSHConfigError, match="could not update"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == original
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    backups = list(config.parent.glob("config.edullm-backup*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert not list(config.parent.glob(".config.edullm-*"))


def test_read_control_config_rejects_unsafe_parent_symlink(tmp_path):
    real_dir = tmp_path / "real-ssh"
    real_dir.mkdir()
    linked_dir = tmp_path / ".ssh"
    linked_dir.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ssh.SSHConfigError, match="symbolic link"):
        ssh.read_control_config(linked_dir / "config")


def test_created_config_and_backup_never_have_group_or_other_permissions(tmp_path):
    config = tmp_path / ".ssh" / "config"
    plan = ssh.plan_control_config(b"", "philote")
    previous_umask = os.umask(0)
    try:
        backup = ssh.apply_control_config(config, plan)
    finally:
        os.umask(previous_umask)

    assert backup is not None
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert stat.S_IMODE(backup.stat().st_mode) == 0o600
