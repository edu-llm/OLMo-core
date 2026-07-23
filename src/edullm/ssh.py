"""
Safe SSH and OpenSSH configuration boundaries for eduLLM operators.
"""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
import secrets
import shlex
import stat
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

COMMAND_TIMEOUT_SECONDS = 30.0
_ALIAS = "orcd-login"
_HOSTNAME = "orcd-login.mit.edu"
_USERNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SENSITIVE_DIRECTIVE = re.compile(
    r"(?i)^([+-]?\s*)(IdentityFile|CertificateFile|ProxyCommand|LocalCommand|RemoteCommand)\b"
)
_REMOTE_PATH = re.compile(r"(?:~|/)[A-Za-z0-9._~/-]+\Z")


class SSHError(RuntimeError):
    """A sanitized SSH operation failure."""


class SSHConfigError(RuntimeError):
    """An unsafe or failed OpenSSH configuration operation."""


@dataclass(frozen=True)
class SSHConfigPlan:
    """A proposed byte-preserving OpenSSH configuration update."""

    original: bytes
    proposed: bytes
    redacted_diff: str

    @property
    def changed(self) -> bool:
        """Return whether this plan changes the configuration."""
        return self.original != self.proposed


class SSHClient:
    """Injectable subprocess boundary for project SSH operations."""

    def __init__(self, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run):
        self._runner = runner

    def _run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
        operation: str = "SSH command failed",
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(
                list(argv),
                input=input_text,
                check=False,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            raise SSHError(operation) from None
        if check and result.returncode != 0:
            raise SSHError(operation)
        return result

    def run_remote(
        self,
        argv: Sequence[str],
        *,
        check: bool = True,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        """Run one argv-safe command through the project alias."""
        command = _remote_command(argv)
        return self._run(
            ["ssh", _ALIAS, command],
            check=check,
            timeout=timeout,
        )

    def run_direct(
        self,
        username: str,
        argv: Sequence[str],
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> subprocess.CompletedProcess[str]:
        """Run one argv-safe command against Engaging without the alias."""
        _validate_username(username)
        command = _remote_command(argv)
        return self._run(
            [
                "ssh",
                "-o",
                "ControlMaster=no",
                "-o",
                "ControlPath=none",
                "-l",
                username,
                _HOSTNAME,
                command,
            ],
            timeout=timeout,
        )

    def write_remote(
        self,
        path: str,
        content: str,
        *,
        timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        """Write private content remotely using standard input."""
        if _REMOTE_PATH.fullmatch(path) is None or ".." in path.split("/"):
            raise ValueError("remote path is unsafe")
        parent = path.rsplit("/", maxsplit=1)[0]
        command = f"umask 077 && mkdir -p {parent} && cat > {path} && chmod 600 {path}"
        self._run(
            ["ssh", _ALIAS, command],
            input_text=content,
            timeout=timeout,
        )

    def close_master(self) -> bool:
        """
        Close only the project ControlMaster.

        :returns: ``True`` when a master was closed and ``False`` when it was
            already absent.
        """
        result = self._run(
            ["ssh", "-O", "exit", _ALIAS],
            operation="could not close eduLLM SSH session",
            check=False,
        )
        if result.returncode == 0:
            return True
        stderr = result.stderr.lower()
        already_closed = "control socket" in stderr and (
            "no such file" in stderr or "does not exist" in stderr
        )
        if already_closed:
            return False
        raise SSHError("could not close eduLLM SSH session")


def control_block(username: str) -> str:
    """Render the managed ``Host orcd-login`` block."""
    _validate_username(username)
    return "\n".join(
        [
            f"Host {_ALIAS}",
            f"    Hostname {_HOSTNAME}",
            "    ControlMaster auto",
            "    ControlPath ~/.ssh/edullm-%C",
            "    ControlPersist 1h",
            f"    User {username}",
        ]
    )


def plan_control_config(original: bytes, username: str) -> SSHConfigPlan:
    """Plan a safe managed-block addition or replacement."""
    block = control_block(username)
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        raise SSHConfigError("SSH config cannot be updated safely") from None

    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    starts: list[tuple[int, list[str]]] = []
    block_started = False
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        directive = stripped.rstrip("\r\n").split(None, maxsplit=1)
        keyword = directive[0]
        normalized_keyword = keyword.casefold()
        if normalized_keyword in {"include", "match"}:
            raise SSHConfigError("SSH config cannot be updated safely")
        if not block_started and normalized_keyword in {
            "hostname",
            "user",
            "controlmaster",
            "controlpath",
            "controlpersist",
        }:
            raise SSHConfigError("SSH config cannot be updated safely")
        if normalized_keyword != "host":
            continue
        block_started = True
        remainder = directive[1] if len(directive) == 2 else ""
        try:
            patterns = shlex.split(remainder, comments=True, posix=True)
        except ValueError:
            raise SSHConfigError("SSH config cannot be updated safely") from None
        if not patterns:
            raise SSHConfigError("SSH config cannot be updated safely")
        starts.append((index, patterns))

    exact_starts: list[int] = []
    for start, patterns in starts:
        exact = len(patterns) == 1 and patterns[0].casefold() == _ALIAS
        if exact:
            exact_starts.append(start)
            continue
        if any(_host_pattern_matches(pattern, _ALIAS) for pattern in patterns):
            raise SSHConfigError("SSH config cannot be updated safely")
    if len(exact_starts) > 1:
        raise SSHConfigError("SSH config cannot be updated safely")

    rendered = block.replace("\n", newline) + newline
    if exact_starts:
        start = exact_starts[0]
        end = len(lines)
        for index in range(start + 1, len(lines)):
            stripped = lines[index].lstrip()
            directive = stripped.rstrip("\r\n").split(None, maxsplit=1)
            keyword = directive[0].casefold() if directive else ""
            if keyword in {"host", "match"}:
                end = index
                break
        proposed_text = "".join(lines[:start]) + rendered + "".join(lines[end:])
    else:
        separator = newline if text else ""
        proposed_text = text + separator + rendered

    proposed = proposed_text.encode("utf-8")
    return SSHConfigPlan(
        original=original,
        proposed=proposed,
        redacted_diff=_redacted_diff(text, proposed_text) if proposed != original else "",
    )


def read_control_config(path: Path) -> bytes:
    """Read an OpenSSH configuration without following unsafe links."""
    if not _inspect_private_directory(path.parent):
        return b""
    try:
        path_status = path.lstat()
    except FileNotFoundError:
        return b""
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    if stat.S_ISLNK(path_status.st_mode):
        raise SSHConfigError("SSH config must not be a symbolic link")
    if not stat.S_ISREG(path_status.st_mode) or path_status.st_uid != os.getuid():
        raise SSHConfigError("SSH config cannot be read safely")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        raise SSHConfigError("SSH config cannot be read safely") from None
    try:
        opened_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_status.st_mode)
            or opened_status.st_uid != os.getuid()
            or (opened_status.st_dev, opened_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise SSHConfigError("SSH config changed while it was being read")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def apply_control_config(path: Path, plan: SSHConfigPlan) -> Path | None:
    """Apply a confirmed plan atomically and return its secure backup."""
    current = read_control_config(path)
    if current != plan.original:
        raise SSHConfigError("SSH config changed after the proposed diff")
    if not plan.changed:
        if path.exists():
            _ensure_private_directory(path.parent)
            _secure_existing_file(path)
        return None

    _ensure_private_directory(path.parent)
    if path.exists():
        _secure_existing_file(path)
    backup = _create_backup(path, current)
    temporary = path.with_name(f".{path.name}.edullm-{secrets.token_hex(8)}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        _write_all(descriptor, plan.proposed)
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        if read_control_config(path) != plan.original:
            raise SSHConfigError("SSH config changed after the proposed diff")
        os.replace(temporary, path)
        _secure_existing_file(path)
        _fsync_directory(path.parent)
    except SSHConfigError:
        raise
    except OSError:
        raise SSHConfigError("SSH config could not update safely") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
    return backup


def run_remote(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run one command through the project SSH alias."""
    return SSHClient().run_remote(argv)


def write_remote(path: str, content: str) -> None:
    """Write content remotely through standard input."""
    SSHClient().write_remote(path, content)


def close_master() -> bool:
    """Close only the project ControlMaster."""
    return SSHClient().close_master()


def _validate_username(username: str) -> None:
    if _USERNAME.fullmatch(username) is None or username.startswith("-"):
        raise ValueError("Engaging username is invalid")


def _remote_command(argv: Sequence[str]) -> str:
    if not argv or any(not isinstance(argument, str) or "\0" in argument for argument in argv):
        raise ValueError("remote argv is invalid")
    return shlex.join(argv)


def _host_pattern_matches(pattern: str, hostname: str) -> bool:
    if pattern.startswith("!"):
        pattern = pattern[1:]
    return fnmatch.fnmatchcase(hostname.casefold(), pattern.casefold())


def _redacted_diff(original: str, proposed: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(),
        proposed.splitlines(),
        fromfile="~/.ssh/config",
        tofile="~/.ssh/config (proposed)",
        n=0,
        lineterm="",
    )
    redacted = []
    for line in lines:
        match = _SENSITIVE_DIRECTIVE.match(line)
        if match:
            redacted.append(f"{match.group(1)}{match.group(2)} <redacted>")
        else:
            redacted.append(line)
    return "\n".join(redacted) + "\n"


def _ensure_private_directory(path: Path) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700, parents=True)
            status = path.lstat()
        except OSError:
            raise SSHConfigError("SSH directory cannot be created safely") from None
    except OSError:
        raise SSHConfigError("SSH directory cannot be inspected safely") from None
    if stat.S_ISLNK(status.st_mode):
        raise SSHConfigError("SSH directory must not be a symbolic link")
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise SSHConfigError("SSH directory is unsafe")
    try:
        path.chmod(0o700)
    except OSError:
        raise SSHConfigError("SSH directory permissions cannot be secured") from None


def _inspect_private_directory(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        raise SSHConfigError("SSH directory cannot be inspected safely") from None
    if stat.S_ISLNK(status.st_mode):
        raise SSHConfigError("SSH directory must not be a symbolic link")
    if not stat.S_ISDIR(status.st_mode) or status.st_uid != os.getuid():
        raise SSHConfigError("SSH directory is unsafe")
    return True


def _create_backup(path: Path, content: bytes) -> Path:
    index = 0
    while True:
        suffix = "" if index == 0 else f".{index}"
        candidate = path.with_name(f"{path.name}.edullm-backup{suffix}")
        try:
            descriptor = os.open(
                candidate,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            index += 1
            continue
        except OSError:
            raise SSHConfigError("SSH config backup could not be created safely") from None
        try:
            _write_all(descriptor, content)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        except OSError:
            try:
                candidate.unlink()
            except OSError:
                pass
            raise SSHConfigError("SSH config backup could not be created safely") from None
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
        return candidate


def _secure_existing_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_uid != os.getuid():
            raise SSHConfigError("SSH config is unsafe")
        os.fchmod(descriptor, 0o600)
    except SSHConfigError:
        raise
    except OSError:
        raise SSHConfigError("SSH config permissions cannot be secured") from None
    finally:
        if "descriptor" in locals():
            os.close(descriptor)


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        raise SSHConfigError("SSH directory could not be synchronized") from None
    finally:
        if "descriptor" in locals():
            os.close(descriptor)
