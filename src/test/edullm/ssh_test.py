import io
import os
import shlex
import stat
import subprocess
from pathlib import Path

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
    assert argv[:2] == ["ssh", "orcd-login"]
    assert shlex.split(argv[2]) == [
        "$HOME/venvs/edullm/bin/python",
        "-m",
        "edullm.ssh_helper",
        "--target",
        "wandb.key",
    ]
    assert "cat >" not in argv[2]


def test_write_remote_rejects_paths_outside_fixed_private_targets():
    runner = RecordingRunner()
    client = ssh.SSHClient(runner=runner)

    with pytest.raises(ValueError, match="remote path"):
        client.write_remote("~/.ssh/authorized_keys", "content")

    assert runner.calls == []


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
    assert "personal config" not in plan.redacted_diff
    assert "github.com" not in plan.redacted_diff
    assert "User git" not in plan.redacted_diff


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
    assert "old.example" in plan.redacted_diff
    assert "User old" in plan.redacted_diff
    assert "Host first" not in plan.redacted_diff
    assert "Host last" not in plan.redacted_diff


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


@pytest.mark.parametrize(
    ("unsafe_line", "secret"),
    [
        ("  SetEnv AWS_SECRET_ACCESS_KEY=aws-secret", "aws-secret"),
        ("  SetEnv GITHUB_TOKEN=github-secret", "github-secret"),
        ("  SetEnv WANDB_API_KEY=wandb-secret", "wandb-secret"),
        ("  UnknownDirective arbitrary-secret", "arbitrary-secret"),
        ("  ProxyCommand sh -c 'send proxy-secret'", "proxy-secret"),
        ("  ProxyCommand=proxy-secret", "proxy-secret"),
    ],
)
def test_plan_rejects_unsafe_target_lines_without_displaying_secrets(unsafe_line, secret):
    original = f"Host orcd-login\n  Hostname old.example\n{unsafe_line}\n".encode()

    with pytest.raises(ssh.SSHConfigError, match="safely") as raised:
        ssh.plan_control_config(original, "philote")

    assert secret not in str(raised.value)


def test_plan_preserves_neighboring_comments_without_displaying_them():
    original = (
        b"Host orcd-login\n"
        b"  Hostname old.example\n"
        b"  User old\n"
        b"\n"
        b"# next-token=neighbor-secret\n"
        b"Host next\n"
        b"  User next\n"
    )

    plan = ssh.plan_control_config(original, "philote")

    assert b"# next-token=neighbor-secret\nHost next\n" in plan.proposed
    assert "neighbor-secret" not in plan.redacted_diff
    assert "Host next" not in plan.redacted_diff


def test_plan_preserves_trailing_comments_at_eof_without_displaying_them():
    original = (
        b"Host orcd-login\n"
        b"  Hostname old.example\n"
        b"  User old\n"
        b"\n"
        b"# eof-token=neighbor-secret\n"
    )

    plan = ssh.plan_control_config(original, "philote")

    assert plan.proposed.endswith(b"\n# eof-token=neighbor-secret\n")
    assert "neighbor-secret" not in plan.redacted_diff


def test_plan_rejects_comment_inside_target_block_before_safe_directive():
    original = b"Host orcd-login\n" b"  # token=inside-secret\n" b"  Hostname old.example\n"

    with pytest.raises(ssh.SSHConfigError, match="safely") as raised:
        ssh.plan_control_config(original, "philote")

    assert "inside-secret" not in str(raised.value)


def test_secure_apply_creates_backup_and_preserves_safe_original_mode(tmp_path):
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
    assert stat.S_IMODE(config.stat().st_mode) == 0o644
    assert stat.S_IMODE(backup.stat().st_mode) == 0o644
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
    config.chmod(0o640)
    plan = ssh.plan_control_config(original, "philote")

    def fail_replace(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(ssh.os, "replace", fail_replace)

    with pytest.raises(ssh.SSHConfigError, match="could not update"):
        ssh.apply_control_config(config, plan)

    assert config.read_bytes() == original
    assert stat.S_IMODE(config.stat().st_mode) == 0o640
    backups = list(config.parent.glob("config.edullm-backup*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o640
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


def test_secure_apply_commits_with_same_open_directory_fd(tmp_path, monkeypatch):
    config = tmp_path / ".ssh" / "config"
    config.parent.mkdir(mode=0o700)
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    real_replace = ssh.os.replace
    calls = []

    def record_replace(*args, **kwargs):
        calls.append((args, kwargs))
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(ssh.os, "replace", record_replace)

    ssh.apply_control_config(config, plan)

    assert len(calls) == 1
    _, options = calls[0]
    assert options["src_dir_fd"] == options["dst_dir_fd"]


def test_secure_apply_rejects_parent_swap_and_cleans_old_directory_temp(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    config = ssh_dir / "config"
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    plan = ssh.plan_control_config(original, "philote")
    moved = tmp_path / ".ssh-moved"
    real_write = ssh._write_all
    swapped = False

    def swap_parent_after_write(descriptor, content):
        nonlocal swapped
        real_write(descriptor, content)
        if not swapped and content == plan.proposed:
            swapped = True
            ssh_dir.rename(moved)
            ssh_dir.mkdir(mode=0o700)

    monkeypatch.setattr(ssh, "_write_all", swap_parent_after_write)

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert (moved / "config").read_bytes() == original
    assert not list(moved.glob(".config.edullm-*"))
    assert not (ssh_dir / "config").exists()


def test_secure_apply_rejects_target_swap_after_validation(tmp_path, monkeypatch):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir(mode=0o700)
    config = ssh_dir / "config"
    original = b"Host github.com\n"
    config.write_bytes(original)
    config.chmod(0o600)
    destination = tmp_path / "destination"
    destination.write_bytes(b"untouched")
    plan = ssh.plan_control_config(original, "philote")
    real_write = ssh._write_all
    swapped = False

    def swap_target_after_write(descriptor, content):
        nonlocal swapped
        real_write(descriptor, content)
        if not swapped and content == plan.proposed:
            swapped = True
            config.unlink()
            config.symlink_to(destination)

    monkeypatch.setattr(ssh, "_write_all", swap_target_after_write)

    with pytest.raises(ssh.SSHConfigError, match="changed"):
        ssh.apply_control_config(config, plan)

    assert destination.read_bytes() == b"untouched"
    assert not list(ssh_dir.glob(".config.edullm-*"))


def _private_parent(home: Path) -> Path:
    return home / ".config" / "edullm"


@pytest.mark.parametrize("target_name", ["wandb.key", "wandb.env"])
def test_remote_helper_atomically_creates_mode_0600_file(tmp_path, target_name):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    ssh_helper.atomic_write_private(home, target_name, io.BytesIO(b"private-content\n"))

    target = _private_parent(home) / target_name
    assert target.read_bytes() == b"private-content\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert not list(target.parent.glob(".edullm-write-*"))


def test_remote_helper_rejects_existing_0644_before_reading_input(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o644)

    class Unreadable:
        @staticmethod
        def read(size=-1):
            raise AssertionError("input must not be read")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", Unreadable())

    assert target.read_bytes() == b"existing"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_rejects_target_symlink_without_touching_destination(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    destination = tmp_path / "destination"
    destination.write_bytes(b"untouched")
    (parent / "wandb.key").symlink_to(destination)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"replacement"))

    assert destination.read_bytes() == b"untouched"
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_rejects_parent_symlink(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()
    real_config = tmp_path / "real-config"
    real_config.mkdir()
    (home / ".config").symlink_to(real_config, target_is_directory=True)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"private-content"))

    assert not (real_config / "edullm").exists()


def test_remote_helper_input_failure_is_sanitized_and_cleans_temp(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    class FailedInput:
        @staticmethod
        def read(size=-1):
            assert list(_private_parent(home).glob(".edullm-write-*"))
            raise OSError("private-content")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed") as raised:
        ssh_helper.atomic_write_private(home, "wandb.key", FailedInput())

    assert "private-content" not in str(raised.value)
    assert not list(_private_parent(home).glob(".edullm-write-*"))
    assert not (_private_parent(home) / "wandb.key").exists()


def test_remote_helper_sanitizes_non_os_input_failure(tmp_path):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()

    class FailedInput:
        @staticmethod
        def read(size=-1):
            raise RuntimeError("private-content")

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed") as raised:
        ssh_helper.atomic_write_private(home, "wandb.key", FailedInput())

    assert "private-content" not in str(raised.value)
    assert not list(_private_parent(home).glob(".edullm-write-*"))


def test_remote_helper_handles_short_writes(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    home.mkdir()
    real_write = ssh_helper.os.write
    calls = 0

    def short_write(descriptor, content):
        nonlocal calls
        calls += 1
        return real_write(descriptor, content[:1])

    monkeypatch.setattr(ssh_helper.os, "write", short_write)

    ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"complete"))

    assert calls > 1
    assert (_private_parent(home) / "wandb.key").read_bytes() == b"complete"


def test_remote_helper_replace_failure_cleans_temp_and_preserves_target(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(ssh_helper.os, "replace", fail_replace)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"replacement"))

    assert target.read_bytes() == b"existing"
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_detects_target_edit_before_commit(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    target = parent / "wandb.key"
    target.write_bytes(b"existing")
    target.chmod(0o600)
    real_validate = ssh_helper._validate_target_snapshot
    calls = 0

    def edit_before_validate(directory_fd, target_name, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.write_bytes(b"concurrent")
            target.chmod(0o600)
        return real_validate(directory_fd, target_name, expected)

    monkeypatch.setattr(ssh_helper, "_validate_target_snapshot", edit_before_validate)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"replacement"))

    assert target.read_bytes() == b"concurrent"
    assert not list(parent.glob(".edullm-write-*"))


def test_remote_helper_detects_parent_swap_before_commit(tmp_path, monkeypatch):
    from edullm import ssh_helper

    home = tmp_path / "home"
    parent = _private_parent(home)
    parent.mkdir(parents=True)
    real_validate = ssh_helper._validate_directory_identity
    calls = 0

    def swap_before_validate(path, directory_fd, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            moved = path.with_name("edullm-moved")
            path.rename(moved)
            path.mkdir()
        return real_validate(path, directory_fd, expected)

    monkeypatch.setattr(ssh_helper, "_validate_directory_identity", swap_before_validate)

    with pytest.raises(ssh_helper.PrivateWriteError, match="private write failed"):
        ssh_helper.atomic_write_private(home, "wandb.key", io.BytesIO(b"private-content"))

    assert not (_private_parent(home) / "wandb.key").exists()
    assert not list((_private_parent(home).with_name("edullm-moved")).glob(".edullm-write-*"))
