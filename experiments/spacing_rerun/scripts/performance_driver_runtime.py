"""CPU driver containment and native process observations. No model operations."""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import uuid
import types
import sysconfig
import io
import stat
import struct


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def identity(stat):
    return dict(device=stat.st_dev, inode=stat.st_ino)


def semantic_code(value):
    """Canonical values without marshal reference flags or object interning."""
    if isinstance(value, types.CodeType):
        fields = ("co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags",
                  "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_filename", "co_name", "co_qualname",
                  "co_firstlineno", "co_code", "co_linetable", "co_exceptiontable", "co_consts")
        return {key: semantic_code(getattr(value, key)) for key in fields}
    if value is None or value is Ellipsis:
        return {"singleton": "None" if value is None else "Ellipsis"}
    if isinstance(value, bytes): return {"bytes_hex": value.hex()}
    if isinstance(value, tuple): return {"tuple": [semantic_code(v) for v in value]}
    if isinstance(value, frozenset):
        return {"frozenset": sorted((semantic_code(v) for v in value), key=lambda v: encoded(v))}
    if isinstance(value, (str, int, bool)): return {"type": type(value).__name__, "value": value}
    if isinstance(value, (float, complex)): return {"type": type(value).__name__, "repr": repr(value)}
    raise ValueError("Unsupported compiled-code constant: " + type(value).__name__)


def compiler_identity():
    return dict(python=sys.version, cache_tag=sys.implementation.cache_tag,
                implementation=sys.implementation.name, optimize=sys.flags.optimize,
                compile_mode="exec", dont_inherit=True)


def code_sha(code):
    return sha(encoded(dict(schema="p4-recursive-semantic-code-v1", compiler=compiler_identity(), code=semantic_code(code))))


def code_identity(code):
    return dict(code_sha256=code_sha(code), filename=code.co_filename, qualname=code.co_qualname,
                flags=code.co_flags, compiler=compiler_identity())


def open_directory(path):
    """Walk from / using descriptors, rejecting every symlink component."""
    path = Path(path)
    require(path.is_absolute(), "Output path must be absolute")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


class OutputAnchor:
    """Retain a directory inode and anchor every creation relative to it.

    Pathname verification detects moves; descriptor writes cannot follow a
    replacement parent symlink. Failure records retain inode and partial refs.
    """
    def __init__(self, path, *, new=True, inherited_fd=None, expected=None):
        self.path = Path(path)
        require(self.path.is_absolute() and str(self.path.resolve()) == str(self.path),
                "Output root must use its original canonical literal path")
        self.parent_fd = None
        self.fd = None
        self.created = []
        self.directory_chains = {}
        self.directory_expected = {}
        self.files_by_path = {}
        self.received_registries = []
        self.custody_journal = None
        self.worker_custody_transports = []
        self.new = new
        if inherited_fd is not None:
            self.fd = os.dup(inherited_fd)
            self.root_identity = identity(os.fstat(self.fd))
            require(self.root_identity == expected, "Inherited output root identity differs")
        elif new:
            self.parent_fd = open_directory(self.path.parent)
            self.parent_identity = identity(os.fstat(self.parent_fd))
            require(not self.path.exists(), "Synthetic output directory must be new")
            self.root_identity = None
        else:
            self.fd = open_directory(self.path)
            self.root_identity = identity(os.fstat(self.fd))
        self.verify()

    def activate(self):
        require(self.new and self.fd is None, "Output root is already active")
        self.verify()
        os.mkdir(self.path.name, mode=0o700, dir_fd=self.parent_fd)
        self.fd = os.open(self.path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.parent_fd)
        self.root_identity = identity(os.fstat(self.fd))
        self.verify()
        return self

    def verify(self, *, complete=False):
        target = self.path if self.fd is not None else self.path.parent
        expected = self.root_identity if self.fd is not None else self.parent_identity
        current = open_directory(target)
        try:
            require(identity(os.fstat(current)) == expected, "Output root/parent was moved or replaced")
        finally:
            os.close(current)
        if complete:
            for relative in self.directory_expected:
                self.verify_path(self.path / relative / "chain-check", include_file=False)
            for row in self.created:
                self.verify_path(self.path / row["relative_path"])

    def verify_path(self, path, *, include_file=True):
        self.verify()
        relative_path = self._relative(path)
        for index in range(1, len(relative_path.parts)):
            relative = str(Path(*relative_path.parts[:index]))
            expected = self.directory_expected.get(relative)
            if expected is None:
                continue
            current = open_directory(self.path / relative)
            try:
                require(identity(os.fstat(current)) == expected,
                        "Output nested directory inode chain was moved or replaced: " + relative)
            finally:
                os.close(current)
        row = self.files_by_path.get(str(relative_path)) if include_file else None
        if row is not None:
            actual = os.stat(path, follow_symlinks=False)
            require(identity(actual) == {k: row[k] for k in ("device", "inode")},
                    "Created output file pathname/inode changed: " + row["relative_path"])

    def _relative(self, path):
        path = Path(path)
        require(path.is_absolute() and path.is_relative_to(self.path), "Write escapes anchored output")
        relative = path.relative_to(self.path)
        require(relative.parts and all(p not in ("", ".", "..") for p in relative.parts), "Invalid anchored relative path")
        return relative

    def parent(self, path, *, create=True):
        self.verify()
        relative = self._relative(path)
        fd = os.dup(self.fd)
        try:
            for index, part in enumerate(relative.parts[:-1], 1):
                self.verify_path(path, include_file=False)
                prefix = str(Path(*relative.parts[:index]))
                created = False
                if create:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        created = True
                    except FileExistsError:
                        require(self.custody_journal is None or prefix in self.directory_expected,
                                "Worker cannot adopt an unregistered existing directory inode")
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = child
                # Retain each full prefix, not merely its final descriptor.
                if prefix not in self.directory_chains:
                    self.directory_chains[prefix] = os.dup(fd)
                observed = identity(os.fstat(fd))
                require(prefix not in self.directory_expected or self.directory_expected[prefix] == observed,
                        "Output opened directory contradicts original inode claim")
                self.directory_expected[prefix] = observed
                if created and self.custody_journal is not None:
                    self.custody_journal.append("created", dict(relative_path=prefix, kind="directory", **observed))
                self.verify_path(path, include_file=False)
            self.verify_path(path)
            return fd, relative.name
        except BaseException:
            os.close(fd)
            raise

    def open_exclusive(self, path, *, binary=False):
        parent, name = self.parent(path)
        try:
            self.verify_path(path)
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        finally:
            os.close(parent)
        row = dict(relative_path=str(self._relative(path)), kind="file", **identity(os.fstat(fd)))
        self.created.append(row)
        self.files_by_path[row["relative_path"]] = row
        try:
            if self.custody_journal is not None: self.custody_journal.append("created", row)
            self.verify_path(path)
        except BaseException:
            os.close(fd)
            raise
        unbuffered = os.fdopen(fd, "wb", buffering=0)
        handle = unbuffered if binary else io.TextIOWrapper(unbuffered, encoding="utf-8", write_through=True)
        return AnchoredFile(self, path, handle)

    def write(self, path, data):
        with self.open_exclusive(path, binary=True) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.verify_path(path)

    def mkdir(self, path):
        parent, name = self.parent(path)
        try:
            self.verify_path(path)
            os.mkdir(name, mode=0o700, dir_fd=parent)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            self.directory_chains[str(self._relative(path))] = child
            self.directory_expected[str(self._relative(path))] = identity(os.fstat(child))
            if self.custody_journal is not None:
                self.custody_journal.append("created", dict(relative_path=str(self._relative(path)), kind="directory", **identity(os.fstat(child))))
        finally:
            os.close(parent)
        self.verify_path(path / "chain-check", include_file=False)

    def symlink(self, path, target):
        parent, name = self.parent(path)
        try:
            self.verify_path(path)
            os.symlink(target, name, dir_fd=parent)
            row = dict(relative_path=str(self._relative(path)), kind="symlink", **identity(os.stat(name, dir_fd=parent, follow_symlinks=False)))
            self.created.append(row); self.files_by_path[row["relative_path"]] = row
            if self.custody_journal is not None: self.custody_journal.append("created", row)
        finally:
            os.close(parent)
        self.verify_path(path, include_file=False)

    def file_custody(self, row):
        relative = Path(row["relative_path"])
        parent_name = str(relative.parent)
        parent_fd = self.directory_chains.get(parent_name, self.fd if parent_name == "." else None)
        temporary = None
        if parent_fd is None:
            try:
                temporary = open_directory(self.path / relative.parent)
                if identity(os.fstat(temporary)) == self.directory_expected.get(parent_name): parent_fd = temporary
            except OSError: pass
        parent = self.directory_custody(parent_name)
        try:
            current = identity(os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)) if parent_fd is not None else None
        except OSError:
            current = None
        finally:
            if temporary is not None: os.close(temporary)
        expected = {k: row[k] for k in ("device", "inode")}
        matches = current == expected
        actual_parent = parent["kernel_observed_fd_path"]
        return dict(retained_inode=expected, anchored_parent_custody=parent,
                    anchored_name_matches_created_inode=matches,
                    kernel_observed_current_path=str(Path(actual_parent) / relative.name) if matches and actual_parent else None,
                    moved_file_current_path_available=matches and actual_parent is not None,
                    external_rename_prevention_claimed=False)

    def directory_custody(self, relative):
        expected = self.root_identity if relative == "." else self.directory_expected[relative]
        retained = self.fd if relative == "." else self.directory_chains.get(relative)
        if retained is not None: return fd_custody(retained, self.path / relative)
        try:
            fd = open_directory(self.path / relative)
            try:
                if identity(os.fstat(fd)) == expected: return fd_custody(fd, self.path / relative)
            finally: os.close(fd)
        except OSError: pass
        return dict(retained_inode=expected, declared_path=str(self.path / relative), declared_path_matches_inode=False,
                    kernel_observed_fd_path=None, path_observation_method="original_reported_inode_current_path_unknown",
                    external_rename_prevention_claimed=False)

    def registry(self):
        """Complete expected custody, without retaining one descriptor per file."""
        self.verify(complete=True)
        actual = {}
        for parent, dirs, files in os.walk(self.path, followlinks=False):
            for name in sorted([*dirs, *files]):
                path = Path(parent) / name; info = path.lstat()
                kind = "directory" if stat.S_ISDIR(info.st_mode) else "symlink" if stat.S_ISLNK(info.st_mode) else "file" if stat.S_ISREG(info.st_mode) else None
                require(kind is not None, "Unregistered special output entry")
                relative = str(path.relative_to(self.path))
                actual[relative] = dict(relative_path=relative, kind=kind, **identity(info))
        expected = {relative: dict(relative_path=relative, kind="directory", **value) for relative, value in self.directory_expected.items()}
        expected.update({relative: dict(row) for relative, row in self.files_by_path.items()})
        # Initial existing scratch entries become a captured parent baseline only.
        if not self.received_registries:
            for relative, row in actual.items():
                if relative not in expected:
                    if row["kind"] == "directory": self.directory_expected[relative] = identity_values(row)
                    else: self.created.append(row); self.files_by_path[relative] = row
                    expected[relative] = row
        require(actual == expected, "Output inventory has an unregistered, removed or replaced inode")
        return self.original_registry()

    def original_registry(self):
        """Original claims only, never a post-failure pathname inventory."""
        expected = {relative: dict(relative_path=relative, kind="directory", **value) for relative, value in self.directory_expected.items()}
        expected.update({relative: dict(row) for relative, row in self.files_by_path.items()})
        return dict(schema="p4-output-inode-registry-v1", output_root=str(self.path), root_identity=self.root_identity,
                    entries=[expected[key] for key in sorted(expected)])

    def receive_registry(self, registry, *, verify_current=True):
        """Register original producer claims before checking current pathnames."""
        rows = checked_registry(registry, self.path, self.root_identity)
        self.received_registries.append(registry)
        for row in rows:
            relative = row["relative_path"]
            if row["kind"] == "directory":
                previous = self.directory_expected.get(relative)
                require(previous is None or previous == identity_values(row), "Received directory contradicts original parent inode")
                self.directory_expected[relative] = identity_values(row)
            else:
                previous = self.files_by_path.get(relative)
                require(previous is None or previous == row, "Received file contradicts original parent inode")
                if previous is None: self.created.append(dict(row)); self.files_by_path[relative] = dict(row)
        if verify_current: self.verify(complete=True)
        return registry

    def failure_state(self, error):
        return dict(schema="p4-performance-partial-output-v1", original_path=str(self.path),
                    root_identity=self.root_identity, created_files=[dict(row, custody=self.file_custody(row)) for row in self.created],
                    retained_directory_custody={relative: self.directory_custody(relative) for relative in self.directory_expected},
                    received_original_registries=self.received_registries,
                    worker_custody_transports=self.worker_custody_transports,
                    original_child_custody_complete=False,
                    original_child_custody_limit="Retained validated producer/journal claims only; failed process, invalid/missing transport or syscall-to-journal interruption can leave custody incomplete. No current replacement inode is adopted.",
                    error_type=type(error).__name__, error=str(error), outside_production_verified=False,
                    retained_files_deleted=False)

    def retain_failure(self, error):
        # Writes only to the pinned inode even if its pathname has been moved.
        if self.fd is None:
            return
        name = "failure-" + uuid.uuid4().hex + ".json"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded(self.failure_state(error)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        for fd in self.directory_chains.values():
            os.close(fd)
        self.directory_chains.clear(); self.directory_expected.clear(); self.files_by_path.clear()
        for name in ("fd", "parent_fd"):
            value = getattr(self, name, None)
            if value is not None:
                os.close(value)
                setattr(self, name, None)


def identity_values(row):
    return {key: row[key] for key in ("device", "inode")}


def checked_inode_entry(row):
    require(isinstance(row, dict) and set(row) == {"relative_path", "kind", "device", "inode"} and
            isinstance(row["relative_path"], str) and isinstance(row["kind"], str) and row["kind"] in {"directory", "file", "symlink"} and
            all(type(row[k]) is int and row[k] >= 0 for k in ("device", "inode")), "Output registry entry schema differs")
    relative = Path(row["relative_path"])
    require(not relative.is_absolute() and relative.parts and str(relative) == row["relative_path"] and
            all(v not in ("", ".", "..") for v in relative.parts), "Output registry path escapes")
    return relative


def checked_registry(registry, root, expected):
    require(isinstance(registry, dict) and set(registry) == {"schema", "output_root", "root_identity", "entries"} and
            registry["schema"] == "p4-output-inode-registry-v1" and registry["output_root"] == str(root) and
            isinstance(registry["root_identity"], dict) and set(registry["root_identity"]) == {"device", "inode"} and
            all(type(registry["root_identity"][k]) is int and registry["root_identity"][k] >= 0 for k in ("device", "inode")) and
            registry["root_identity"] == expected and isinstance(registry["entries"], list), "Output registry root/schema differs")
    seen = {}
    for row in registry["entries"]:
        checked_inode_entry(row)
        require(row["relative_path"] not in seen, "Output registry path repeats")
        seen[row["relative_path"]] = row
    require([row["relative_path"] for row in registry["entries"]] == sorted(seen), "Output registry order differs")
    for relative in seen:
        for parent in Path(relative).parents:
            if str(parent) != ".": require(seen.get(str(parent), {}).get("kind") == "directory", "Output registry omits an original directory chain")
    return registry["entries"]


class CustodyJournal:
    """Flush original creation claims before later work, including abrupt exits."""
    def __init__(self, invocation):
        self.fd = invocation["custody_journal_fd"]
        self.binding = sha(encoded(invocation))
        self.sequence = 0
        self.previous = None
        self.append("start", invocation["output_registry_before_child"])

    def append(self, event, value):
        row = dict(schema="p4-worker-custody-journal-v1", invocation_buffer_sha256=self.binding,
                   sequence=self.sequence, previous_sha256=self.previous, event=event, value=value)
        data = encoded(row) + b"\n"
        pending = memoryview(data)
        while pending:
            written = os.write(self.fd, pending)
            require(written > 0, "Custody journal write made no progress")
            pending = pending[written:]
        os.fsync(self.fd)
        self.previous = sha(encoded(row)); self.sequence += 1


def read_custody_journal(raw, invocation):
    """Retain only a structurally valid bound prefix, without pathname reads."""
    baseline = invocation["output_registry_before_child"]
    root, expected = Path(invocation["output_root"]), invocation["output_identity"]
    checked_registry(baseline, root, expected)
    claims = {row["relative_path"]: row for row in baseline["entries"]}
    previous = None; accepted = 0; terminal = None; issues = []
    binding = sha(encoded(invocation))
    for line in raw.splitlines(keepends=True):
        try:
            require(terminal is None, "Custody journal has records after terminal")
            row = json.loads(line)
            require(isinstance(row, dict) and set(row) == {"schema", "invocation_buffer_sha256", "sequence", "previous_sha256", "event", "value"} and
                    row["schema"] == "p4-worker-custody-journal-v1" and type(row["sequence"]) is int and row["sequence"] == accepted and
                    row["previous_sha256"] == previous and row["invocation_buffer_sha256"] == binding and
                    line == encoded(row) + b"\n", "Custody journal binding/order/raw framing differs")
            event, value = row["event"], row["value"]
            require(isinstance(event, str), "Custody journal event is malformed")
            if accepted == 0:
                require(event == "start" and value == baseline, "Custody journal omits exact original parent baseline")
            elif event == "created":
                relative = checked_inode_entry(value)
                require(value["relative_path"] not in claims,
                        "Custody journal creation repeats an original path")
                require(all(str(parent) == "." or claims.get(str(parent), {}).get("kind") == "directory" for parent in relative.parents),
                        "Custody journal omits original directory creation chain")
                claims[value["relative_path"]] = value
            else:
                require(event in {"success", "failure"}, "Custody journal event is unknown")
                checked_registry(value, root, expected)
                require(value["entries"] == [claims[key] for key in sorted(claims)], "Custody terminal contradicts original creation prefix")
                terminal = event
            previous = sha(encoded(row)); accepted += 1
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as error:
            issues.append(type(error).__name__ + ": " + str(error)); break
    if not raw: issues.append("Custody journal unavailable")
    if terminal is None: issues.append("Custody journal terminal unavailable; syscall-to-journal interruption may omit claims")
    registry = dict(baseline, entries=[claims[key] for key in sorted(claims)]) if accepted else None
    return dict(schema="p4-worker-custody-journal-prefix-observation-v1", raw_sha256=sha(raw), raw_bytes=len(raw),
                validated_record_count=accepted, original_registry=registry, terminal_kind=terminal, issues=issues,
                complete_registered_journal=bool(terminal and not issues), all_original_artifact_custody_certified=False)


class AnchoredFile:
    """Check the full current chain at open, write, flush and close boundaries."""
    def __init__(self, anchor, path, handle):
        self.anchor, self.path, self.handle = anchor, path, handle
    def __getattr__(self, name): return getattr(self.handle, name)
    def __enter__(self): return self
    def write(self, data):
        self.anchor.verify_path(self.path)
        value = self.handle.write(data)
        self.anchor.verify_path(self.path)
        return value
    def flush(self):
        self.anchor.verify_path(self.path)
        self.handle.flush()
        self.anchor.verify_path(self.path)
    def close(self):
        try:
            self.anchor.verify_path(self.path)
        finally:
            self.handle.close()
        self.anchor.verify_path(self.path)
    def __exit__(self, kind, value, traceback):
        if kind is None: self.close()
        else: self.handle.close()


def fd_custody(fd, declared):
    retained = identity(os.fstat(fd))
    try:
        pathname_matches = identity(os.stat(declared, follow_symlinks=False)) == retained
    except OSError:
        pathname_matches = False
    actual = None
    try:
        if sys.platform.startswith("linux"):
            actual = os.readlink("/proc/self/fd/" + str(fd))
        elif sys.platform == "darwin":
            import fcntl
            actual = fcntl.fcntl(fd, 50, b"\0" * 1024).split(b"\0", 1)[0].decode()
    except (OSError, UnicodeError):
        pass
    return dict(retained_inode=retained, declared_path=str(declared), declared_path_matches_inode=pathname_matches,
                kernel_observed_fd_path=actual, path_observation_method="kernel_fd_path_or_unavailable",
                external_rename_prevention_claimed=False)


def capture_sources(driver_path, package_root, *, extra=()):
    """Capture source once; workers compile these bytes, never caller paths."""
    driver_path, package_root = Path(driver_path), Path(package_root)
    files = sorted(package_root.rglob("*.py")) + sorted(package_root.parent.joinpath("configs").glob("*.json"))
    files += sorted(package_root.parent.joinpath("plans").glob("*.json"))
    files += [driver_path, Path(__file__), *map(Path, extra)]
    files = list(dict.fromkeys(files))
    rows = []
    for path in files:
        require(path.is_file() and not path.is_symlink(), "Deployment source is absent or a symlink")
        data = path.read_bytes()
        relative = str(path.relative_to(package_root.parent)) if path.is_relative_to(package_root.parent) else "external/" + path.name
        rows.append(dict(path=str(path), relative=relative, sha256=sha(data), bytes=len(data),
                         data=base64.b64encode(data).decode()))
    require(len({r["relative"] for r in rows}) == len(rows), "Captured deployment has a relative-path collision")
    value = dict(schema="p4-cpu-buffered-deployment-v1", files=rows,
                 driver_relative=next(r["relative"] for r in rows if r["path"] == str(driver_path)),
                 interpreter=dict(path=str(Path(sys.executable).resolve()), sha256=sha(Path(sys.executable).read_bytes()),
                                  version=sys.version),
                 runtime_paths=list(dict.fromkeys(str(Path(p).resolve()) for p in
                     [*sys.path, *sysconfig.get_paths().values()] if p and Path(p).is_absolute() and
                     any(Path(p).resolve().is_relative_to(Path(prefix).resolve()) for prefix in (sys.base_prefix, sys.prefix)))),
                 artifact_runtime_roots=list(dict.fromkeys(str(Path(p).resolve()) for p in (sys.base_prefix, sys.prefix))),
                 compiler=compiler_identity())
    value["sha256"] = sha(encoded(value))
    return value


def verify_sources(deployment):
    require(deployment["sha256"] == sha(encoded({k: v for k, v in deployment.items() if k != "sha256"})),
            "Buffered deployment identity differs")
    for row in deployment["files"]:
        require(sha(base64.b64decode(row["data"])) == row["sha256"], "Captured source buffer differs")
        require(sha(Path(row["path"]).read_bytes()) == row["sha256"], "Source changed during CPU measurement: " + row["path"])
    require(sha(Path(deployment["interpreter"]["path"]).read_bytes()) == deployment["interpreter"]["sha256"],
            "Executed interpreter bytes changed")


def require_immutable_runtime(contract):
    """Production requires an actual read-only runtime mount, not a JSON claim."""
    require(sys.platform.startswith("linux"), "UNRESOLVED_IMMUTABLE_RUNTIME: production CPU runtime requires Linux mount evidence")
    process_status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    require(all(int(process_status[key].strip(), 16) == 0 for key in ("CapEff", "CapPrm")) and
            process_status.get("NoNewPrivs", "").strip() == "1",
            "UNRESOLVED_IMMUTABLE_RUNTIME: effective/permitted capability or privilege reacquisition remains enabled")
    root = Path(contract["root"])
    require(root.is_absolute() and root.resolve() == root, "Immutable runtime root is noncanonical")
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    def readonly_mount(target):
        covering = []
        for line in mounts:
            fields = line.split()
            mount = Path(fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\"))
            if target.is_relative_to(mount):
                covering.append((len(mount.parts), mount, fields[5].split(","), line))
        require(covering, "UNRESOLVED_IMMUTABLE_RUNTIME: runtime mount absent")
        _, mount, options, line = max(covering, key=lambda row: row[0])
        require("ro" in options, "UNRESOLVED_IMMUTABLE_RUNTIME: runtime file has a writable covering mount")
        return mount, line
    mount, line = readonly_mount(root)
    files = contract["files"]
    require(files and len({r["path"] for r in files}) == len(files), "Runtime file census absent or repeated")
    expected = {r["path"]: r["file_sha256"] for r in files}
    for path, checksum in expected.items():
        target = Path(path)
        require(target.is_relative_to(root) and target.resolve() == target and sha(target.read_bytes()) == checksum,
                "Immutable runtime file is outside its image or changed")
        readonly_mount(target)
    exe = str(Path(sys.executable).resolve())
    require(exe in expected, "UNRESOLVED_IMMUTABLE_RUNTIME: actual interpreter absent from runtime census")
    return dict(root=str(root), mount=str(mount), raw_mountinfo=line, files=files,
                interpreter=exe, runtime_census_sha256=sha(encoded(files)),
                namespace_privilege_status={key: process_status[key].strip() for key in ("CapEff", "CapPrm", "NoNewPrivs")})


def require_native_namespace(runtime, anchor, production_roots, *, exact_data_files=()):
    """Actual covering mounts and opened devices, not an approval boolean."""
    require(sys.platform.startswith("linux"), "UNRESOLVED_NATIVE_CENSUS: Linux kernel enforcement is required")
    require_immutable_runtime(runtime)
    lines = Path("/proc/self/mountinfo").read_text().splitlines()
    mounts = []
    for line in lines:
        fields = line.split(); mounts.append((Path(fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")), set(fields[5].split(",")), line, fields[2]))
    def covering(path):
        rows = [r for r in mounts if path.is_relative_to(r[0])]
        require(rows, "Native namespace path has no actual covering mount")
        return max(rows, key=lambda r: len(r[0].parts))
    anchor.verify()
    output_device = os.fstat(anchor.fd).st_dev
    observed = []
    for root in map(Path, production_roots):
        fd = open_directory(root)
        try:
            st = os.fstat(fd)
            require(st.st_dev != output_device,
                    "UNRESOLVED_WRITE_ISOLATION: output and production share a filesystem; external rename remains possible")
            descendants = [r for r in mounts if r[0].is_relative_to(root)]
            devices = {str(os.major(st.st_dev)) + ":" + str(os.minor(st.st_dev)), *(r[3] for r in descendants)}
            relevant = [covering(root), *descendants, *(r for r in mounts if r[3] in devices)]
            require(all({"ro", "noexec"} <= r[1] for r in relevant),
                    "UNRESOLVED_NATIVE_CENSUS: every production covering/descendant/device-alias mount must be ro,noexec")
            require(all(r[3] != str(os.major(output_device)) + ":" + str(os.minor(output_device)) for r in descendants),
                    "UNRESOLVED_WRITE_ISOLATION: a production descendant aliases the actual output filesystem")
            observed.append(dict(path=str(root), descriptor_identity=identity(st), covering_mounts=[r[2] for r in relevant]))
        finally:
            os.close(fd)
    output_mounts = [covering(anchor.path), *(r for r in mounts if r[0].is_relative_to(anchor.path))]
    output_devices = {str(os.major(output_device)) + ":" + str(os.minor(output_device)), *(r[3] for r in output_mounts)}
    output_mounts += [r for r in mounts if r[3] in output_devices]
    require(all("noexec" in r[1] for r in output_mounts),
            "UNRESOLVED_NATIVE_CENSUS: writable output/data requires actual noexec mounts")
    data_files = []
    for path in sorted(set(exact_data_files)):
        target = Path(path)
        require(target.is_absolute() and target.resolve() == target and target.is_file(), "Native exact data read path is absent/noncanonical")
        device = target.stat().st_dev
        aliases = [covering(target), *(r for r in mounts if r[3] == str(os.major(device)) + ":" + str(os.minor(device)))]
        require(all({"ro", "noexec"} <= r[1] for r in aliases),
                "UNRESOLVED_NATIVE_CENSUS: exact application/data read files and every device alias require ro,noexec")
        data_files.append(dict(path=str(target), file_sha256=sha(target.read_bytes()), device_alias_mounts=[r[2] for r in aliases]))
    return dict(schema="p4-native-namespace-enforcement-input-v1", runtime=runtime,
                elf_execution_policy=elf_execution_policy(runtime["files"], runtime["interpreter"]),
                production_roots=production_roots, output_root=str(anchor.path),
                output_descriptor_identity=identity(os.fstat(anchor.fd)), production_descriptors=observed,
                output_covering_mounts=[r[2] for r in output_mounts],
                exact_data_files=data_files,
                mechanism="landlock_before_exec_exact_read_census_plus_noexec_data_and_distinct_output_device")


def elf_execution_policy(files, interpreter):
    """Reject kernel-created initial RWX segments or an executable ELF stack."""
    rows = []
    for row in files:
        path = Path(row["path"])
        with path.open("rb") as stream:
            header = stream.read(64)
            if not header.startswith(b"\x7fELF"): continue
            require(len(header) == 64 and header[4:6] == b"\x02\x01", "Native ELF class/endianness unsupported")
            kind, machine = struct.unpack_from("<HH", header, 16)
            if kind not in (2, 3): continue
            phoff = struct.unpack_from("<Q", header, 32)[0]
            phsize, phcount = struct.unpack_from("<HH", header, 54)
            require(phsize == 56 and 0 < phcount < 65535 and phoff + phsize * phcount <= path.stat().st_size,
                    "Native ELF program-header extent is invalid")
            stream.seek(phoff); raw = stream.read(phsize * phcount)
        segments = [struct.unpack_from("<II", raw, i * phsize) for i in range(phcount)]
        require(all(not (tag == 1 and flags & 3 == 3) for tag, flags in segments), "Native ELF contains an initial writable executable load segment")
        stacks = [flags for tag, flags in segments if tag == 0x6474e551]
        require(len(stacks) == 1 and not stacks[0] & 1, "Native ELF executable/missing GNU_STACK policy rejected")
        rows.append(dict(path=str(path), file_sha256=row["file_sha256"], machine=machine,
                         program_header_sha256=sha(header + raw), load_flags=[flags for tag, flags in segments if tag == 1],
                         gnu_stack_flags=stacks[0]))
    require(any(row["path"] == interpreter for row in rows), "Actual interpreter lacks reviewed ELF stack/load metadata")
    return dict(schema="p4-native-ELF-execution-permissions-v1", files=rows, interpreter=interpreter,
                kernel_initial_writable_executable_segments_allowed=False, executable_stack_allowed=False)


def native_read_rules(policy, deployment, pid):
    read_file, read_dir, execute = 4, 8, 1
    rules = {}
    def add(path, access): rules[path] = rules.get(path, 0) | access
    for row in policy["runtime"]["files"]: add(row["path"], read_file | execute)
    for row in deployment["files"]: add(row["path"], read_file)
    for row in policy["exact_data_files"]: add(row["path"], read_file)
    add("/proc/" + str(pid) + "/maps", read_file)
    for root in [*policy["production_roots"], policy["output_root"]]: add(root, read_file | read_dir)
    # Newly opened writable files and filesystem mutations are output-only.
    add(policy["output_root"], 2 | 128 | 256 | 4096 | 16384)
    for path in list(rules):
        for parent in Path(path).parents: add(str(parent), read_dir)
    return rules


def native_handled_access():
    # Landlock ABI3 supports all filesystem rights through TRUNCATE.
    return (1 << 15) - 1


def landlock_before_exec(policy, deployment, receipt_fd, invocation_hash):
    """Kernel denies uncensused file reads before the dynamic linker/Python starts."""
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = int(libc.syscall(444, ctypes.c_void_p(), ctypes.c_size_t(0), ctypes.c_uint(1)))
    require(abi >= 3, "UNRESOLVED_NATIVE_CENSUS: Landlock ABI3 write/truncate scope unavailable")
    class Ruleset(ctypes.Structure): _fields_ = [("handled_access_fs", ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]
    read_file, read_dir, execute = 4, 8, 1
    attributes = Ruleset(native_handled_access())
    ruleset = int(libc.syscall(444, ctypes.byref(attributes), ctypes.sizeof(attributes), 0))
    require(ruleset >= 0, "Landlock ruleset creation failed: errno=" + str(ctypes.get_errno()))
    rules = native_read_rules(policy, deployment, os.getpid())
    maps_path = "/proc/" + str(os.getpid()) + "/maps"
    for path, access in sorted(rules.items()):
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = PathRule(access, fd)
            result = int(libc.syscall(445, ruleset, 1, ctypes.byref(rule), 0))
            require(result == 0, "Landlock exact path rule failed: " + path + " errno=" + str(ctypes.get_errno()))
        finally: os.close(fd)
    result = int(libc.syscall(446, ruleset, 0))
    os.close(ruleset)
    require(result == 0, "Landlock restriction failed: errno=" + str(ctypes.get_errno()))
    preexec_filter = install_native_process_filter(True, before_exec=True)
    record = dict(schema="p4-preexec-landlock-observation-v1", pid=os.getpid(), abi=abi,
        invocation_buffer_sha256=invocation_hash, policy_sha256=sha(encoded(policy)),
        runtime_census_sha256=policy["runtime"]["runtime_census_sha256"],
        elf_execution_policy_sha256=sha(encoded(policy["elf_execution_policy"])),
        exact_rules_sha256=sha(encoded(rules)), exact_rule_count=len(rules),
        handled_access_fs=native_handled_access(), preexec_native_filter=preexec_filter,
        restrict_self_result=result, enforcement_before_execve=True, exact_self_maps_read_path=maps_path)
    os.write(receipt_fd, encoded(record) + b"\n")
    os.close(receipt_fd)


def native_filter_program(machine, *, before_exec=False):
    import errno
    nr = {"x86_64": (0xc000003e, [57,58,319,30,101,311,323,321,298,216,16,425,426,427], [59,322],56,9,10,329,135),
          "aarch64":(0xc00000b7,[279,196,117,271,282,280,241,234,29,425,426,427], [221,281],220,222,226,288,92)}
    require(machine in nr, "Native process seccomp architecture unsupported")
    arch, denied, exec_calls, clone, mmap, mprotect, pkey_mprotect, personality = nr[machine]
    if not before_exec: denied = denied + exec_calls
    allow, reject = 0x7fff0000, 0x00050000 | errno.EPERM
    instructions=[(0x20,0,0,4),(0x15,1,0,arch),(0x06,0,0,reject),(0x20,0,0,0),
                  (0x45,0,1,0x40000000),(0x06,0,0,reject)]
    for syscall in denied: instructions += [(0x15,0,1,syscall),(0x06,0,0,reject)]
    def syscall_block(number, block):
        instructions.extend([(0x15,0,len(block),number), *block, (0x20,0,0,0)])
    # Reject every initial writable executable mapping, including private files.
    syscall_block(mmap, [(0x20,0,0,32),(0x45,0,5,4),(0x45,0,1,2),(0x06,0,0,reject),
                         (0x20,0,0,40),(0x45,0,1,0x20),(0x06,0,0,reject)])
    for syscall in (mprotect,pkey_mprotect):
        syscall_block(syscall, [(0x20,0,0,32),(0x45,0,1,4),(0x06,0,0,reject)])
    # Query is allowed; READ_IMPLIES_EXEC and other personality changes are not.
    syscall_block(personality, [(0x20,0,0,16),(0x15,1,0,0xffffffff),(0x06,0,0,reject)])
    instructions += [(0x15,0,1,435),(0x06,0,0,0x00050000 | errno.ENOSYS),
        (0x15,0,3,clone),(0x20,0,0,16),(0x45,1,0,0x00010000),(0x06,0,0,reject),(0x06,0,0,allow)]
    return instructions


def install_native_process_filter(production, *, before_exec=False):
    if not production:
        return dict(scope="artifact_only", kernel_native_process_filter_installed=False)
    import ctypes, platform, errno
    libc = ctypes.CDLL(None, use_errno=True)
    machine = platform.machine()
    personality_number = {"x86_64":135,"aarch64":92}.get(machine)
    require(personality_number is not None, "Native personality architecture unsupported")
    libc.syscall.restype = ctypes.c_long
    personality = int(libc.syscall(personality_number, ctypes.c_uint(0xffffffff)))
    require(personality >= 0, "Native personality query failed")
    if before_exec and personality & 0x0400000:
        cleared = int(libc.syscall(personality_number, ctypes.c_uint(personality & ~0x0400000)))
        require(cleared >= 0, "Cannot clear inherited READ_IMPLIES_EXEC")
        personality = int(libc.syscall(personality_number, ctypes.c_uint(0xffffffff)))
    require(personality >= 0 and not personality & 0x0400000,
            "Native implicit executable-read personality remains enabled")
    class Filter(ctypes.Structure): _fields_=[("code",ctypes.c_ushort),("jt",ctypes.c_ubyte),("jf",ctypes.c_ubyte),("k",ctypes.c_uint32)]
    class Program(ctypes.Structure): _fields_=[("len",ctypes.c_ushort),("filter",ctypes.POINTER(Filter))]
    instructions=native_filter_program(machine, before_exec=before_exec)
    filters=(Filter*len(instructions))(*(Filter(*r) for r in instructions)); program=Program(len(filters),filters)
    result=int(libc.prctl(22,2,ctypes.byref(program),0,0))
    require(result == 0, "Native process seccomp filter failed: errno=" + str(ctypes.get_errno()))
    return dict(schema="p4-native-process-filter-observation-v2", pid=os.getpid(), architecture=machine,
                program_sha256=sha(encoded(instructions)), install_result=result,
                stage="before_execve" if before_exec else "before_application", personality_query_result=personality,
                denied_operations="writable_exec_mmap,anonymous_exec_mmap,exec_mprotect,personality_changes,ptrace,process_vm_writev,userfaultfd,io_uring,bpf,perf,remap_file_pages,ioctl,fork,memfd,shmat,nonthread_clone;exec_denied_after_bootstrap;clone3_ENOSYS",
                actual_threads_remain_in_worker_RSS=True)


BOOTSTRAP = r'''
import sys
prebootstrap_start_modules = sorted(sys.modules)
import base64, hashlib, importlib.abc, importlib.util, importlib.machinery, json, os, sys, types
fd = int(sys.argv[1]); raw = b""
while True:
    chunk = os.read(fd, 1048576)
    if not chunk: break
    raw += chunk
os.close(fd)
data = json.loads(raw); deployment = data["deployment"]
expected = sys.argv[2]
assert hashlib.sha256(raw).hexdigest() == expected
sys.path[:] = deployment["runtime_paths"]
os.environ["SPACING_EXPERIMENT_ROOT"] = data["deployment_root"]
def prohibit_descendants(event, arguments):
    if event in {"subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec"}:
        raise RuntimeError("CPU worker descendants are outside this measured process scope: " + event)
sys.addaudithook(prohibit_descendants)
approved = {r["path"]:r["file_sha256"] for r in (data.get("immutable_runtime") or {}).get("files", [])}
artifact_roots = tuple(deployment.get("artifact_runtime_roots", []))
lifetime = []
sys._p4_runtime_lifetime_observations = lifetime
def admitted(name, origin, phase):
    if origin in (None, "built-in", "frozen"):
        row = {"module":name,"origin":origin,"phase":phase,"interpreter_sha256":deployment["interpreter"]["sha256"]}
    else:
        actual = os.path.realpath(origin)
        if approved:
            if actual not in approved: raise ImportError("Runtime module is outside approved before-exec census: " + actual)
        elif not any(actual == root or actual.startswith(root + "/") for root in artifact_roots):
            raise ImportError("Artifact runtime import is outside captured interpreter roots: " + actual)
        with open(actual,"rb") as stream: checksum = hashlib.sha256(stream.read()).hexdigest()
        if approved and checksum != approved[actual]: raise ImportError("Runtime source changed before execution: " + actual)
        row = {"module":name,"path":actual,"file_sha256":checksum,"phase":phase,
               "approved_complete_census_enforced":bool(approved)}
    lifetime.append(row)
    return row
for name, module in sorted(sys.modules.items()):
    spec = getattr(module,"__spec__",None)
    admitted(name, getattr(spec,"origin",None) or getattr(module,"__file__",None),
             "prebootstrap_startup" if name in prebootstrap_start_modules else "reviewed_bootstrap_imports")
class CensusLoader:
    def __init__(self, original, row): self.original,self.row=original,row
    def create_module(self,spec):
        return self.original.create_module(spec) if hasattr(self.original,"create_module") else None
    def exec_module(self,module):
        if self.row.get("path","").endswith(".py"):
            # Execute verified source, never an unbound bytecode cache.
            raw = open(self.row["path"],"rb").read()
            if hashlib.sha256(raw).hexdigest()!=self.row["file_sha256"]: raise ImportError("Runtime source changed at execution")
            exec(compile(raw,self.row["path"],"exec",dont_inherit=True),module.__dict__)
        else: self.original.exec_module(module)
        lifetime.append(dict(self.row,phase="completed_runtime_module_execution"))
    def __getattr__(self,name): return getattr(self.original,name)
original_finders = tuple(sys.meta_path)
class CensusFinder:
    def find_spec(self,fullname,path=None,target=None):
        for finder in original_finders:
            spec = finder.find_spec(fullname,path,target)
            if spec is not None:
                row = admitted(fullname,spec.origin,"before_runtime_module_execution")
                if spec.loader is not None: spec.loader=CensusLoader(spec.loader,row)
                return spec
        return None
sys.meta_path.insert(0,CensusFinder())
# Direct public loader use has no meta_path event. Guard its execution too.
def guarded_direct_exec(original):
    def run(loader,module):
        spec=getattr(module,"__spec__",None)
        origin=getattr(spec,"origin",None) or getattr(loader,"path",None)
        row=admitted(module.__name__,origin,"before_runtime_module_execution")
        if row.get("path","").endswith(".py"):
            raw=open(row["path"],"rb").read()
            if hashlib.sha256(raw).hexdigest()!=row["file_sha256"]: raise ImportError("Direct runtime source changed at execution")
            exec(compile(raw,row["path"],"exec",dont_inherit=True),module.__dict__)
        else: original(loader,module)
        lifetime.append(dict(row,phase="completed_runtime_module_execution"))
    return run
for loader_type in (importlib.machinery.SourceFileLoader,importlib.machinery.SourcelessFileLoader,importlib.machinery.ExtensionFileLoader):
    loader_type.exec_module=guarded_direct_exec(loader_type.exec_module)
original_extension_create=importlib.machinery.ExtensionFileLoader.create_module
def guarded_extension_create(loader,spec):
    admitted(spec.name,spec.origin,"before_runtime_module_execution")
    return original_extension_create(loader,spec)
importlib.machinery.ExtensionFileLoader.create_module=guarded_extension_create
sources = {}
for row in deployment["files"]:
    content = base64.b64decode(row["data"])
    assert hashlib.sha256(content).hexdigest() == row["sha256"]
    relative = row["relative"]
    name = None
    if relative.startswith("spacing_rerun/") and relative.endswith(".py"):
        name = relative[:-3].replace("/", ".")
        if name.endswith(".__init__"): name = name[:-9]
    elif relative.endswith("performance_driver_runtime.py"):
        name = "performance_driver_runtime"
    if name: sources[name] = (row, content)
class Loader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in sources:
            row, content = sources[fullname]
            return importlib.util.spec_from_loader(fullname, self, is_package=row["relative"].endswith("/__init__.py"))
        if fullname == "spacing_rerun" or fullname.startswith("spacing_rerun."):
            raise ImportError("Package module is outside captured source closure: " + fullname)
    def create_module(self, spec): return None
    def exec_module(self, module):
        row, content = sources[module.__name__]
        module.__file__ = data["deployment_root"] + "/" + row["relative"]
        if row["relative"].endswith("/__init__.py"): module.__path__ = [os.path.dirname(module.__file__)]
        exec(compile(content, module.__file__, "exec", dont_inherit=True), module.__dict__)
        module.__captured_source_sha256__ = row["sha256"]
sys.meta_path.insert(0, Loader())
import performance_driver_runtime as captured_runtime
sys._p4_native_process_enforcement = captured_runtime.install_native_process_filter(bool(data.get("native_namespace")))
driver = next(r for r in deployment["files"] if r["relative"] == deployment["driver_relative"])
namespace = {"__name__": "__main__", "__file__": data["deployment_root"] + "/" + driver["relative"],
             "CAPTURED_DRIVER_SOURCE": base64.b64decode(driver["data"]), "BUFFERED_INVOCATION": data,
             "__captured_source_sha256__": driver["sha256"]}
main_module = types.ModuleType("__main__")
main_module.__dict__.update(namespace)
sys.modules["__main__"] = main_module
sys.argv[:] = [namespace["__file__"], "--worker", data["input_path"]]
exec(compile(namespace["CAPTURED_DRIVER_SOURCE"], namespace["__file__"], "exec", dont_inherit=True), main_module.__dict__)
'''


def loaded_identity(deployment):
    files = {row["sha256"] for row in deployment["files"]}
    package = {}
    runtime = []
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        captured = getattr(module, "__captured_source_sha256__", None)
        if captured:
            require(captured in files, "Loaded package code is outside captured deployment")
            package[name] = captured
        elif path and Path(path).is_file():
            runtime.append(dict(module=name, path=str(Path(path).resolve()), file_sha256=sha(Path(path).read_bytes())))
    code_objects = {}
    for module_name in ("spacing_rerun.confirmation_dependency", "spacing_rerun.confirmation_lane"):
        module = sys.modules.get(module_name)
        if module:
            for name in ("validate_final_policy", "saved_bundle_diagnostics", "complete_contrast_report", "influence_scores",
                         "complement_contrast", "build_saved_dependency_report"):
                function = getattr(module, name, None)
                if function and hasattr(function, "__code__"):
                    code_objects[module_name + "." + name] = code_sha(function.__code__)
    native = []
    if sys.platform.startswith("linux"):
        paths = set()
        mapped = Path("/proc/" + str(os.getpid()) + "/maps").read_text()
        for line in mapped.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                require(not fields[5].endswith(" (deleted)"), "Loaded native runtime file was deleted")
                paths.add(str(Path(fields[5]).resolve()))
        for path in sorted(paths):
            native.append(dict(path=path, file_sha256=sha(Path(path).read_bytes())))
    return dict(buffered_package_modules=package, loaded_function_code_sha256=code_objects, compiler=compiler_identity(),
                loaded_runtime_files=runtime, loaded_native_libraries=native, interpreter=deployment["interpreter"],
                lifetime_runtime_imports=list(getattr(sys,"_p4_runtime_lifetime_observations", [])),
                native_process_enforcement=getattr(sys,"_p4_native_process_enforcement", None),
                prebootstrap_scope="startup and reviewed bootstrap reads are constrained before execve by production Landlock; artifact mode has no such production proof")


def rss_bytes(usage):
    return int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024))


OBSERVER_FRAME_LIMIT = 32768
OBSERVER_FD_LIMIT = 7


class ObserverPacketFailure(ValueError):
    def __init__(self, error, header=b"", raw=b"", credentials=None, credential_raw=None, received_fd_identities=(), *, observation=None):
        super().__init__(str(error))
        self.header = header
        self.raw = raw
        self.credentials = credentials
        self.credential_raw_base64 = base64.b64encode(credential_raw).decode("ascii") if credential_raw is not None else None
        self.received_fd_identities = list(received_fd_identities)
        self.packet_observation = observation


def observer_received_packet_observation(packet, *, phase):
    """Capture receiver bytes and every available fstat before validation/close."""
    descriptors = []; identities = []
    for index, fd in enumerate(packet.get("fds", ())):
        observed = None; error = None
        try:
            observed = identity(os.fstat(fd)); identities.append(observed)
        except OSError as exception:
            error = dict(type=type(exception).__name__, errno=exception.errno, message=str(exception))
        descriptors.append(dict(index=index, receiver_namespace_fd=fd, original_fstat_identity=observed,
                                fstat_error=error, validation_status="unvalidated"))
    header = packet.get("header", b""); raw = packet.get("raw", b"")
    return dict(schema="p4-original-observer-rejection-observation-v1", phase=phase,
        received_packet=packet.get("received_packet", True),
        header_raw_base64=base64.b64encode(header).decode("ascii"), header_raw_bytes=len(header), header_raw_sha256=sha(header),
        payload_raw_base64=base64.b64encode(raw).decode("ascii"), payload_raw_bytes=len(raw), payload_raw_sha256=sha(raw),
        credentials_receiver_namespace=packet.get("credentials"), credential_raw_base64=packet.get("credential_raw_base64"),
        credential_raw_fragments_base64=list(packet.get("credential_raw_fragments_base64", [])),
        received_unvalidated_fd_identities=identities, received_descriptor_observations=descriptors,
        descriptor_cleanup_observations=[], received_unvalidated_fds_closed=False,
        sender_declarations_used_as_original_authority=False, acceptance_certified=False,
        Linux_identity_or_enforcement_certified=False)


def observer_close_received_fds(fds, observation):
    """Retain each actual cleanup result without erasing the original error."""
    cleanup = []
    for index, fd in enumerate(fds):
        error = None
        try: os.close(fd)
        except OSError as exception:
            error = dict(type=type(exception).__name__, errno=exception.errno, message=str(exception))
        cleanup.append(dict(index=index, receiver_namespace_fd=fd, close_succeeded=error is None, close_error=error))
    observation["descriptor_cleanup_observations"] = cleanup
    observation["received_unvalidated_fds_closed"] = all(row["close_succeeded"] for row in cleanup)


def observer_packet_failure(error, packet, observation):
    """Typed byte observation shared by framing and semantic rejection paths."""
    credential_raw = packet.get("credential_raw_base64")
    return ObserverPacketFailure(error, packet.get("header", b""), packet.get("raw", b""), packet.get("credentials"),
        base64.b64decode(credential_raw, validate=True) if credential_raw is not None else None,
        observation["received_unvalidated_fd_identities"], observation=observation)


def observer_send_packet(sock, raw, fds=()):
    import array
    import socket
    require(0 < len(raw) <= OBSERVER_FRAME_LIMIT and len(fds) <= OBSERVER_FD_LIMIT,
            "Observer frame or descriptor count exceeds exact bound")
    frame = struct.pack("!I", len(raw)) + raw
    ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", fds))] if fds else []
    sent = sock.sendmsg([frame], ancillary)
    require(sent > 0, "Observer send made no progress")
    if sent < len(frame): sock.sendall(frame[sent:])


def observer_receive_packet(sock, *, deadline=None):
    """Bounded stream frame, retaining actual partial bytes on failure."""
    import array
    import socket
    header = b""; raw = b""; fds = []; credentials = None; credential_bytes = None
    credential_raw_fragments = []
    original_timeout = sock.gettimeout()
    def receive(size, target):
        nonlocal credentials, credential_bytes, header, raw
        if deadline is not None:
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Observer total frame deadline elapsed")
            sock.settimeout(min(original_timeout, remaining) if original_timeout is not None else remaining)
        data, ancillary, flags, _ = sock.recvmsg(size, socket.CMSG_SPACE(OBSERVER_FD_LIMIT*array.array("i").itemsize)+socket.CMSG_SPACE(12))
        if target == "header": header += data
        else: raw += data
        # Adopt all actual received rights before any ancillary validation fails.
        for level, kind, value in ancillary:
            if level == socket.SOL_SOCKET and kind == getattr(socket, "SCM_CREDENTIALS", -1):
                credential_bytes = value
                credential_raw_fragments.append(base64.b64encode(value).decode("ascii"))
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                received = array.array("i")
                received.frombytes(value[:len(value)-len(value)%received.itemsize])
                fds.extend(received)
                for fd in received: os.set_inheritable(fd,False)
        for level, kind, value in ancillary:
            require(level == socket.SOL_SOCKET, "Unexpected observer ancillary level")
            if kind == socket.SCM_RIGHTS:
                received = array.array("i")
                require(len(value) % received.itemsize == 0, "Observer rights bytes truncated")
            elif kind == getattr(socket, "SCM_CREDENTIALS", -1):
                require(len(value) == 12, "Observer credential bytes differ")
                pid, uid, gid = struct.unpack("iII", value)
                actual = dict(pid=pid, uid=uid, gid=gid)
                require(credentials is None or credentials == actual, "Observer sender changed within frame")
                credentials = actual
            else: raise ValueError("Unexpected observer ancillary type")
        require(not flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) and len(fds) <= OBSERVER_FD_LIMIT,
                "Observer ancillary/message truncated or too many descriptors")
        return data
    try:
        while len(header) < 4:
            data = receive(4-len(header), "header")
            if not data:
                require(not header and not fds, "Observer partial header ended")
                return None
        length = struct.unpack("!I", header)[0]
        require(0 < length <= OBSERVER_FRAME_LIMIT, "Observer frame length exceeds bound")
        while len(raw) < length:
            data = receive(length-len(raw), "payload")
            require(data, "Observer partial payload ended")
        return dict(header=header, raw=raw, fds=fds, credentials=credentials,
                    credential_raw_base64=base64.b64encode(credential_bytes).decode("ascii") if credential_bytes is not None else None,
                    credential_raw_fragments_base64=credential_raw_fragments)
    except Exception as error:
        packet = dict(header=header, raw=raw, fds=fds, credentials=credentials,
            credential_raw_base64=base64.b64encode(credential_bytes).decode("ascii") if credential_bytes is not None else None,
            credential_raw_fragments_base64=credential_raw_fragments)
        observation = observer_received_packet_observation(packet, phase="framing_rejection")
        observer_close_received_fds(fds, observation)
        raise observer_packet_failure(error, packet, observation) from error
    finally:
        if deadline is not None: sock.settimeout(original_timeout)


class NativeChannelObserver:
    """Borrowed supervisor-only bridge; RW duplicates are held by retainer."""
    def __init__(self, fd, invocation, root_identity):
        import socket
        require(type(fd) is int and fd >= 3, "Observer needs one explicit supervisor descriptor")
        self.production = bool(invocation.get("native_namespace"))
        require(not self.production or (sys.platform.startswith("linux") and hasattr(socket, "SO_PASSCRED")),
                "Production observer needs actual Linux SCM_CREDENTIALS")
        for key in ("nonce", "observer_run_nonce", "observer_attempt_nonce"):
            require(isinstance(invocation.get(key), str) and invocation[key], "Observer requires original fresh nonce binding")
        self.fd = fd; self.identity = identity(os.fstat(fd))
        self.sock = socket.socket(fileno=fd)
        self.original_timeout = self.sock.gettimeout()
        try:
            self.sock.settimeout(5)
            require(self.sock.family == socket.AF_UNIX and self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM,
                    "Observer must be a connected Unix stream bridge")
            self.sock.getpeername()
            if hasattr(socket, "SO_PASSCRED"): self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self.binding = dict(invocation_nonce=invocation["nonce"], run_nonce=invocation["observer_run_nonce"],
                attempt_nonce=invocation["observer_attempt_nonce"], deployment_sha256=invocation["deployment"]["sha256"],
                policy_sha256=sha(encoded(invocation["native_namespace"])) if invocation.get("native_namespace") is not None else None,
                output_root_identity=root_identity, scope="production_credentials_required" if self.production else "artifact_only_cpu",
                launcher_pid_namespace=os.getpid())
        except BaseException:
            self.sock.settimeout(self.original_timeout)
            self.sock.detach()
            raise
        self.sequence = 0; self.previous = None; self.registrations = []
        self.receiver_credentials = None
        self.rejection_observations = []

    def send(self, kind, value, roles=(), fds=()):
        raw = encoded(dict(schema="p4-original-native-observer-message-v1", sequence=self.sequence,
            previous_raw_sha256=self.previous, kind=kind, binding=self.binding, roles=list(roles), value=value))
        observer_send_packet(self.sock, raw, fds)
        try:
            packet = observer_receive_packet(self.sock, deadline=time.monotonic()+5)
        except ObserverPacketFailure as error:
            error.packet_observation["expected_ACK_sequence"] = self.sequence
            error.packet_observation["sent_message_raw_sha256"] = sha(raw)
            self.rejection_observations.append(error.packet_observation)
            raise
        if packet is None:
            packet = dict(header=b"", raw=b"", fds=[], credentials=None, credential_raw_base64=None, received_packet=False)
        observation = observer_received_packet_observation(packet, phase="ACK_semantic_rejection")
        observation.update(expected_ACK_sequence=self.sequence, sent_message_raw_sha256=sha(raw))
        try:
            require(observation["received_packet"], "Observer ACK absent")
            require(not packet["fds"] and (not self.production or packet["credentials"] is not None),
                    "Observer ACK descriptors/actual credentials differ")
            if self.production:
                require(packet["credentials"]["pid"] > 0 and
                        (self.receiver_credentials is None or self.receiver_credentials == packet["credentials"]),
                        "Observer ACK receiver credentials changed")
                self.receiver_credentials = packet["credentials"]
            ack = json.loads(packet["raw"])
            require(encoded(ack) == packet["raw"] and set(ack) == {"schema", "sequence", "message_raw_sha256", "binding_sha256", "durable_registration", "receipt_sha256"} and
                    ack["schema"] == "p4-original-native-observer-ack-v1" and type(ack["sequence"]) is int and ack["sequence"] == self.sequence and
                    ack["message_raw_sha256"] == sha(raw) and ack["binding_sha256"] == sha(encoded(self.binding)) and
                    ack["durable_registration"] is True and isinstance(ack["receipt_sha256"], str) and len(ack["receipt_sha256"]) == 64 and
                    all(c in "0123456789abcdef" for c in ack["receipt_sha256"]),
                    "Observer ACK does not bind exact durable message")
        except Exception as error:
            observer_close_received_fds(packet["fds"], observation)
            self.rejection_observations.append(observation)
            raise observer_packet_failure(error, packet, observation) from error
        self.previous = sha(raw); self.sequence += 1
        return dict(message_raw_sha256=sha(raw), ack_raw_sha256=sha(packet["raw"]),
                    receipt_sha256=ack["receipt_sha256"], receiver_credentials=packet["credentials"])

    def register(self, kind, handles):
        import fcntl
        roles = [name for name, fd in handles]
        value = {name: dict(identity(os.fstat(fd)), access_mode=fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE,
                           anonymous=os.fstat(fd).st_nlink == 0) for name, fd in handles}
        receipt = self.send(kind, value, roles, [fd for name, fd in handles])
        self.registrations.append(dict(kind=kind, identities=value, **receipt))

    def raw_value(self, kind, raw):
        require(len(raw) <= 2*1024*1024, "Observer raw native/invocation record exceeds bound")
        record = uuid.uuid4().hex
        for offset in range(0, max(1,len(raw)), 8192):
            self.send("raw_piece", dict(record=record, record_kind=kind, offset=offset,
                total_bytes=len(raw), raw_sha256=sha(raw), raw_base64=base64.b64encode(raw[offset:offset+8192]).decode("ascii")))

    def public_binding(self):
        return dict(schema="p4-original-native-observer-binding-v1", binding=self.binding,
                    registrations=self.registrations, supervisor_bridge_identity=self.identity,
                    outside_RW_duplicate_authority_disclosed=True, kernel_readonly_observer_claimed=False)

    def close_in_child(self):
        try:
            observed = identity(os.fstat(self.fd))
        except OSError as error:
            import errno
            require(error.errno == errno.EBADF, "Observer child bridge closure is unknown")
            return
        require(observed == self.identity, "Observer child descriptor was replaced before closure")
        os.close(self.fd)

    def detach(self):
        self.sock.settimeout(self.original_timeout)
        self.sock.detach()


class NativeLaunchFailure(ValueError):
    """Available actual channel bytes survive a rejected native launch."""
    def __init__(self, error, native, producer):
        super().__init__(str(error))
        self.native_observation = native
        self.producer_bytes = producer
        if "observer_rejection_observations" in native:
            self.observer_rejection_observations = native["observer_rejection_observations"]


def original_observer_rejection_bytes(native):
    """Decode/check retained originals, never recreate a received ACK as JSON."""
    records = []
    for observed in native.get("observer_rejection_observations", []):
        require(observed.get("schema") == "p4-original-observer-rejection-observation-v1",
                "Original observer rejection schema differs")
        raw_values = {}
        for kind in ("header", "payload"):
            raw = base64.b64decode(observed[kind+"_raw_base64"], validate=True)
            require(type(observed[kind+"_raw_bytes"]) is int and len(raw) == observed[kind+"_raw_bytes"] and
                    sha(raw) == observed[kind+"_raw_sha256"], "Original observer rejection bytes/length/hash differ")
            raw_values[kind] = raw
        records.append((observed, raw_values))
    return records


def observe_preexec_receipt(raw):
    parsed = None; error = None
    if raw:
        try:
            parsed = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as exception:
            error = type(exception).__name__ + ": " + str(exception)
    return dict(preexec_receipt_raw_base64=base64.b64encode(raw).decode("ascii"),
                preexec_receipt_raw_bytes=len(raw), preexec_receipt_raw_sha256=sha(raw),
                preexec_kernel_observation=parsed, preexec_receipt_parse_error=error)


def original_preexec_receipt(native):
    """Verify actual retained bytes, never reconstruct an original from JSON."""
    raw = base64.b64decode(native["preexec_receipt_raw_base64"], validate=True)
    observed = observe_preexec_receipt(raw)
    require(type(native.get("preexec_receipt_raw_bytes")) is int and
            all(native.get(key) == value for key, value in observed.items()),
            "Original preexec receipt bytes/length/hash/parse observation differ")
    require(native.get("preexec_receipt_channel_identity") == native["inherited_channel_identities"]["preexec_receipt"],
            "Original preexec receipt channel identity differs")
    return raw


def launch_native(invocation, anchor, log_path, *, channel_observer_fd=None):
    """One exact wait4 observation and original channel bytes through exit."""
    anchor.verify()
    with tempfile.TemporaryFile(dir=anchor.path) as source, tempfile.TemporaryFile(dir=anchor.path) as channel, tempfile.TemporaryFile(dir=anchor.path) as enforcement, tempfile.TemporaryFile(dir=anchor.path) as journal:
        anchor.verify()
        handles = (("invocation", source), ("producer", channel), ("preexec_receipt", enforcement), ("custody_journal", journal))
        inherited_channels = {name: identity(os.fstat(handle.fileno())) for name, handle in handles}
        require(all(row["device"] == anchor.root_identity["device"] for row in inherited_channels.values()),
                "Inherited native channels are outside the anchored noexec output device")
        invocation["inherited_channel_identities"] = inherited_channels
        invocation["producer_channel_fd"] = channel.fileno()
        invocation["custody_journal_fd"] = journal.fileno()
        policy = invocation.get("native_namespace")
        if policy:
            require(len(list(Path("/proc/self/task").iterdir())) == 1,
                    "Native preexec supervisor requires a single actual parent thread")
        started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        started = time.monotonic()
        pid = status = usage = child = None; launch_error = None; argv = None; data = None; observer = None
        try:
            if channel_observer_fd is not None:
                require(channel_observer_fd not in {fd.fileno() for _, fd in handles} | {anchor.fd},
                        "Observer bridge aliases a native channel/root descriptor")
                observer = NativeChannelObserver(channel_observer_fd, invocation, anchor.root_identity)
                observer.register("channels_created", [("root_anchor", anchor.fd)]+[(name, fd.fileno()) for name, fd in handles])
            with anchor.open_exclusive(Path(log_path).with_suffix(".invocation.json"), binary=True) as invocation_file, anchor.open_exclusive(log_path) as log:
                if observer:
                    observer.register("named_files_created", [("invocation_file", invocation_file.fileno()), ("stdout_log", log.fileno())])
                    invocation["original_native_observer"] = observer.public_binding()
                invocation["output_registry_before_child"] = anchor.registry()
                data = encoded(invocation)
                source.write(data); source.flush(); source.seek(0)
                invocation_file.write(data + b"\n"); invocation_file.flush(); os.fsync(invocation_file.fileno())
                if observer:
                    observer.raw_value("invocation_buffer", data)
                argv = [sys.executable, "-I", "-B", "-S", "-c", BOOTSTRAP, str(source.fileno()), sha(data)]
                preexec = (lambda: landlock_before_exec(policy, invocation["deployment"], enforcement.fileno(), sha(data))) if policy else None
                if observer:
                    def observed_preexec():
                        observer.close_in_child()
                        if policy: landlock_before_exec(policy, invocation["deployment"], enforcement.fileno(), sha(data))
                    preexec = observed_preexec
                child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(source.fileno(), channel.fileno(), enforcement.fileno(), journal.fileno(), anchor.fd),
                                         preexec_fn=preexec,
                                         env=dict({k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}, CUDA_VISIBLE_DEVICES=""),
                                         cwd=str(anchor.path))
                if observer: observer.send("popen_returned", dict(pid=child.pid, pid_scope="launcher_namespace"))
                pid, status, usage = os.wait4(child.pid, 0)
                child.returncode = os.waitstatus_to_exitcode(status)
                if observer: observer.send("wait4_observed", dict(pid=pid, raw_wait_status=status, exit_code=child.returncode,
                    through_exit_child_peak_rss_bytes=rss_bytes(usage), pid_scope="launcher_namespace"))
        except Exception as error:
            launch_error = error
            if child is not None and pid is None:
                pid = child.pid
            if child is not None and usage is None and child.returncode is None and observer is not None:
                try:
                    import signal
                    try: os.kill(child.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    pid, status, usage = os.wait4(child.pid, 0)
                    child.returncode = os.waitstatus_to_exitcode(status)
                except Exception as reap_error:
                    launch_error = RuntimeError(str(error)+"; owned-child reaping unavailable: "+str(reap_error))
        elapsed = time.monotonic() - started
        channel.seek(0); producer_bytes = channel.read()
        enforcement.seek(0); enforcement_bytes = enforcement.read()
        journal.seek(0); journal_bytes = journal.read()
        receipt = observe_preexec_receipt(enforcement_bytes)
        channels_after = {name: identity(os.fstat(handle.fileno())) for name, handle in handles}
        native = dict(schema="p4-internal-native-process-observation-v1" if usage is not None else "p4-internal-native-launch-failure-v1",
            pid=pid, argv=argv, cwd=str(anchor.path), invocation_buffer_sha256=sha(data) if data is not None else None,
            invocation_nonce=invocation.get("nonce"), policy_sha256=sha(encoded(policy)) if policy is not None else None,
            inherited_channel_identities=inherited_channels, inherited_channel_identities_after=channels_after,
            preexec_receipt_channel_identity=inherited_channels["preexec_receipt"],
            custody_journal_raw_base64=base64.b64encode(journal_bytes).decode("ascii"), custody_journal_raw_sha256=sha(journal_bytes),
            started_utc=started_utc, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), wall_seconds=elapsed,
            exit_code=child.returncode if usage is not None else None, raw_wait_status=status,
            through_exit_child_peak_rss_bytes=rss_bytes(usage) if usage is not None else None,
            parent_peak_rss_bytes=rss_bytes(resource.getrusage(resource.RUSAGE_SELF)),
            wait4_observation_available=usage is not None, original_child_custody_complete=False,
            native_loading_completeness_scope="production kernel denies uncensused file reads before execve; data noexec; artifact mode supplies no native-loading completeness proof",
            descendant_scope="production_seccomp_prohibits_native_descendants;artifact_only_CPython_audit_and_RUSAGE_CHILDREN_no_native_completeness_proof",
            root_scheduler_or_terminal_authenticity_claimed=False, **receipt)
        if launch_error is None:
            try:
                require(inherited_channels == channels_after,
                        "Inherited native channel descriptor identities changed through exit")
                require(receipt["preexec_receipt_parse_error"] is None, "Original preexec receipt parse failed: " + str(receipt["preexec_receipt_parse_error"]))
                kernel_receipt = receipt["preexec_kernel_observation"]
                if policy:
                    require(isinstance(kernel_receipt, dict) and kernel_receipt.get("pid") == pid and
                            kernel_receipt.get("invocation_buffer_sha256") == sha(data) and
                            kernel_receipt.get("policy_sha256") == sha(encoded(policy)) and
                            kernel_receipt.get("exact_self_maps_read_path") == "/proc/" + str(pid) + "/maps" and
                            kernel_receipt.get("handled_access_fs") == native_handled_access() and
                            kernel_receipt.get("restrict_self_result") == 0 and kernel_receipt.get("abi", 0) >= 3,
                            "Actual preexec kernel receipt is missing or contradicts native invocation")
                else:
                    require(not enforcement_bytes and kernel_receipt is None,
                            "Artifact launch cannot claim a nonempty kernel receipt")
            except Exception as error:
                launch_error = error
        if launch_error is not None:
            native["launch_error_type"] = type(launch_error).__name__
            native["launch_error"] = str(launch_error)
        if observer is not None:
            native["original_native_observer"] = observer.public_binding()
            if observer.rejection_observations:
                native["observer_rejection_observations"] = list(observer.rejection_observations)
            try:
                observer.raw_value("native_record", encoded(native))
                observer.send("finished", dict(success=launch_error is None, actual_wait4_available=usage is not None))
            except Exception as observer_error:
                native["observer_delivery_error"] = type(observer_error).__name__+": "+str(observer_error)
                if launch_error is None:
                    launch_error = observer_error
                    native["launch_error_type"] = type(launch_error).__name__
                    native["launch_error"] = str(launch_error)
            finally:
                if observer.rejection_observations:
                    native["observer_rejection_observations"] = list(observer.rejection_observations)
                observer.detach()
        if launch_error is not None:
            raise NativeLaunchFailure(launch_error, native, producer_bytes) from launch_error
        return producer_bytes, native
