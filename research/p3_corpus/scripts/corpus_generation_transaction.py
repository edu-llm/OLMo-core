"""Versioned, occurrence-routed, atomic corpus generation transactions."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import inspect
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

API_VERSION = 2
MANIFEST_FILENAME = "MANIFEST.json"
ROUTES_FILENAME = "ROUTES.jsonl"
CURRENT_FILENAME = "CURRENT"
QUARANTINE_FILENAME = "QUARANTINE.json"
TRANSACTION_STATE_SCHEMA_VERSION = "generation-transaction-state/v1"
MANIFEST_SCHEMA_VERSION = "corpus-generation-manifest/v2"
CURRENT_SCHEMA_VERSION = "corpus-generation-current/v2"
ROUTES_SCHEMA_VERSION = "physical-occurrence-routes/v2"
ACCOUNTING_SCHEME = "physical-occurrence-routes/v2"
PLAN_SCHEMA_VERSION = "corpus-generation-plan/v2"
LOGICAL_ROOT_SCHEMA_VERSION = "logical-generation-root/v1"
PHYSICAL_ID_POLICY = "caller-supplied-immutable-id/v1"
COMMIT_POINT = "successful atomic CURRENT replacement"

_GENERATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SIBLING_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DROP_TYPE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_VALIDATOR_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}/v[1-9][0-9]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RESERVED_LINK_FIELDS = {
    "schema_version",
    "generation_id",
    "source_generation_id",
    "plan_root_sha256",
}
_MANIFEST_KEYS = {
    "accounting",
    "api_version",
    "directories",
    "generation_id",
    "logical_root_sha256",
    "manifest_root_sha256",
    "outputs",
    "physical_generation_id_policy",
    "plan_root_sha256",
    "requested_siblings",
    "routes",
    "schema_version",
    "source_generation_id",
}
_OUTPUT_METADATA_KEYS = {
    "bytes",
    "drop_types",
    "generation_id",
    "logical_sha256",
    "path",
    "role",
    "rows",
    "schema",
    "sha256",
    "sibling",
    "source_generation_id",
    "validator",
}
_ROUTES_METADATA_KEYS = {
    "bytes",
    "path",
    "root_sha256",
    "rows",
    "schema_version",
    "sha256",
}
_ROUTE_KEYS = {
    "destination_path",
    "destination_row",
    "disposition",
    "drop_type",
    "occurrence_id",
    "plan_root_sha256",
    "raw_path",
    "raw_row",
    "raw_sha256",
    "sibling",
}
_DROP_RECORD_KEYS = {
    "details",
    "drop_type",
    "generation_id",
    "occurrence_id",
    "plan_root_sha256",
    "raw_path",
    "raw_row",
    "raw_sha256",
    "schema_version",
    "sibling",
    "source_generation_id",
}
_QUARANTINE_KEYS = {
    "entry",
    "generation_id",
    "kind",
    "original_name",
    "reason",
    "schema_version",
}
_TRANSACTION_STATE_KEYS = {
    "generation_id",
    "logical_root_sha256",
    "manifest_sha256",
    "schema_version",
    "state",
}
_REQUIRED_DIR_FD_FUNCTIONS = {
    "mkdir": os.mkdir,
    "open": os.open,
    "rename": os.rename,
    "stat": os.stat,
    "unlink": os.unlink,
}
_REQUIRED_DIR_FD_KEYWORDS = {
    "mkdir": ("dir_fd",),
    "open": ("dir_fd",),
    "rename": ("src_dir_fd", "dst_dir_fd"),
    "stat": ("dir_fd", "follow_symlinks"),
    "unlink": ("dir_fd",),
}


class GenerationError(RuntimeError):
    """Base class for generation transaction failures."""


class InventoryError(GenerationError):
    """The declared or physical inventory is not exact."""


class AccountingError(GenerationError):
    """Physical raw occurrences do not have exactly one valid route."""


class ValidationError(GenerationError):
    """A schema, seal, route, mode, or digest failed validation."""


class UnsafePathAPIError(GenerationError):
    """A removed pathname adapter was called."""


class PlatformCapabilityError(GenerationError):
    """The host cannot provide required symlink-safe filesystem operations."""


class GenerationExistsError(GenerationError):
    """An immutable physical generation ID already exists."""


class CommitUncertainError(GenerationError):
    """CURRENT was replaced but its parent-directory fsync was not confirmed."""

    def __init__(
        self,
        generation_id: str,
        logical_root_sha256: str,
        cause: BaseException,
    ):
        super().__init__(
            f"commit state is uncertain for {generation_id}: "
            f"{type(cause).__name__}: {cause}"
        )
        self.generation_id = generation_id
        self.logical_root_sha256 = logical_root_sha256
        self.cause = cause

    def resolve(self, coordinator: GenerationCoordinator) -> PublishedGeneration:
        """Adjudicate the visible CURRENT seal after an uncertain commit."""

        resolved = coordinator.resolve_current()
        if (
            resolved.generation_id != self.generation_id
            or resolved.logical_root_sha256 != self.logical_root_sha256
        ):
            raise ValidationError(
                "commit-uncertain CURRENT does not select the attempted generation"
            )
        coordinator._recover_visible_transaction(
            self.generation_id,
            self.logical_root_sha256,
        )
        recovered = coordinator.resolve_current()
        return replace(recovered, commit_state="durable_recovered")


class OutputRole(str, Enum):
    """Semantic role of a declared generation output."""

    RAW = "raw"
    TRAIN = "train"
    EVAL = "eval"
    HELDOUT = "heldout"
    SIDECAR = "sidecar"


class PublishPhase(str, Enum):
    """Fault-injection boundaries, including low-level commit operations."""

    STAGING_CREATED = "staging_created"
    OUTPUTS_WRITTEN = "outputs_written"
    MANIFEST_WRITTEN = "manifest_written"
    VALIDATED = "validated"
    STAGING_SYNCED = "staging_synced"
    GENERATION_RENAME_BEFORE = "generation_rename_before"
    GENERATION_RENAME_AFTER = "generation_rename_after"
    GENERATIONS_FSYNC_BEFORE = "generations_fsync_before"
    GENERATIONS_FSYNC_AFTER = "generations_fsync_after"
    CURRENT_TEMP_WRITE_BEFORE = "current_temp_write_before"
    CURRENT_TEMP_WRITE_AFTER = "current_temp_write_after"
    CURRENT_TEMP_PARENT_FSYNC_BEFORE = "current_temp_parent_fsync_before"
    CURRENT_TEMP_PARENT_FSYNC_AFTER = "current_temp_parent_fsync_after"
    CURRENT_REPLACE_BEFORE = "current_replace_before"
    CURRENT_REPLACE_AFTER = "current_replace_after"
    ROOT_FSYNC_BEFORE = "root_fsync_before"
    ROOT_FSYNC_AFTER = "root_fsync_after"


@dataclass(frozen=True, slots=True)
class ValidationContext:
    """Generation links required by structured output validators."""

    generation_id: str
    source_generation_id: str
    plan_root_sha256: str


@dataclass(frozen=True, slots=True)
class JsonlValidator:
    """Concrete reconstructible JSONL object-schema validator."""

    schema_version: str
    required_fields: tuple[str, ...]
    allow_empty: bool = False
    require_generation_links: bool = False
    validator_version: int = 1

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if type(self.allow_empty) is not bool:
            raise InventoryError("JSONL validator allow_empty must be a boolean")
        if type(self.require_generation_links) is not bool:
            raise InventoryError(
                "JSONL validator require_generation_links must be a boolean"
            )
        if type(self.validator_version) is not int or self.validator_version != 1:
            raise InventoryError("unsupported JSONL validator version")
        if not isinstance(self.required_fields, (tuple, list)):
            raise InventoryError("JSONL validator required fields must be a sequence")
        fields = tuple(self.required_fields)
        if not fields or any(not isinstance(name, str) or not name for name in fields):
            raise InventoryError("JSONL validator requires named fields")
        if len(fields) != len(set(fields)):
            raise InventoryError("JSONL validator required fields must be unique")
        object.__setattr__(self, "required_fields", tuple(sorted(fields)))

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "allow_empty": self.allow_empty,
            "kind": "jsonl-object",
            "require_generation_links": self.require_generation_links,
            "required_fields": list(self.required_fields),
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
        }

    def validate(self, path: Path, context: ValidationContext) -> int:
        records = _read_jsonl_objects(path)
        if not records and not self.allow_empty:
            raise ValidationError(f"{path}: structured JSONL output is empty")
        for row, record in enumerate(records, start=1):
            _validate_structured_record(
                record,
                path=path,
                row=row,
                schema_version=self.schema_version,
                required_fields=self.required_fields,
                require_generation_links=self.require_generation_links,
                context=context,
            )
        return len(records)

    def logical_sha256(self, path: Path, context: ValidationContext) -> str:
        if not self.require_generation_links:
            return _sha256_file(path)
        records = _read_jsonl_objects(path)
        normalized = b"".join(
            _canonical_json_bytes(_normalize_physical_links(record))
            for record in records
        )
        return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True, slots=True)
class JsonObjectValidator:
    """Concrete reconstructible JSON object-schema validator."""

    schema_version: str
    required_fields: tuple[str, ...]
    require_generation_links: bool = False
    validator_version: int = 1

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if type(self.require_generation_links) is not bool:
            raise InventoryError(
                "JSON object validator require_generation_links must be a boolean"
            )
        if type(self.validator_version) is not int or self.validator_version != 1:
            raise InventoryError("unsupported JSON object validator version")
        if not isinstance(self.required_fields, (tuple, list)):
            raise InventoryError("JSON object validator fields must be a sequence")
        fields = tuple(self.required_fields)
        if not fields or any(not isinstance(name, str) or not name for name in fields):
            raise InventoryError("JSON object validator requires named fields")
        if len(fields) != len(set(fields)):
            raise InventoryError("JSON object validator fields must be unique")
        object.__setattr__(self, "required_fields", tuple(sorted(fields)))

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "kind": "json-object",
            "require_generation_links": self.require_generation_links,
            "required_fields": list(self.required_fields),
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
        }

    def validate(self, path: Path, context: ValidationContext) -> int:
        record = _read_json_object(path, str(path))
        _validate_structured_record(
            record,
            path=path,
            row=None,
            schema_version=self.schema_version,
            required_fields=self.required_fields,
            require_generation_links=self.require_generation_links,
            context=context,
        )
        return 1

    def logical_sha256(self, path: Path, context: ValidationContext) -> str:
        if not self.require_generation_links:
            return _sha256_file(path)
        record = _read_json_object(path, str(path))
        return hashlib.sha256(
            _canonical_json_bytes(_normalize_physical_links(record))
        ).hexdigest()


BinaryValidation = Callable[[Path, ValidationContext], int]


@dataclass(frozen=True, slots=True)
class BinaryValidator:
    """Explicit versioned validator for non-structured binary outputs."""

    schema_version: str
    validator_id: str
    validate: BinaryValidation = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if not isinstance(self.validator_id, str) or not _VALIDATOR_ID_RE.fullmatch(
            self.validator_id
        ):
            raise InventoryError(
                "binary validator ID must be concrete and versioned, e.g. name/v1"
            )
        if not callable(self.validate):
            raise InventoryError("binary validator must provide a callable")

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "kind": "binary",
            "schema_version": self.schema_version,
            "validator_id": self.validator_id,
            "validator_version": 1,
        }

    def logical_sha256(self, path: Path, context: ValidationContext) -> str:
        return _sha256_file(path)


OutputValidator = JsonlValidator | JsonObjectValidator | BinaryValidator
Producer = Callable[["GenerationWriter"], None]
FaultInjector = Callable[[PublishPhase, Path], None]


def _validate_schema_version(schema: str) -> None:
    if (
        not isinstance(schema, str)
        or not schema
        or not re.search(r"(?:/|-)v[1-9][0-9]*\Z", schema)
    ):
        raise InventoryError(
            f"schema must have an explicit /vN or -vN version: {schema!r}"
        )


def _validate_logical_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise InventoryError("output path must be a non-empty string")
    if "\\" in path:
        raise InventoryError(f"output path must use POSIX separators: {path!r}")
    logical = PurePosixPath(path)
    if logical.is_absolute() or str(logical) != path:
        raise InventoryError(f"output path is not canonical and relative: {path!r}")
    if any(part in {"", ".", ".."} for part in logical.parts):
        raise InventoryError(f"output path escapes its generation: {path!r}")
    if path in {MANIFEST_FILENAME, ROUTES_FILENAME, CURRENT_FILENAME}:
        raise InventoryError(f"output path is reserved: {path!r}")
    if logical.parts[0] in {".staging", "generations", "quarantine"}:
        raise InventoryError(f"output path uses a reserved directory: {path!r}")


def _validate_generation_id(generation_id: str) -> None:
    if not isinstance(generation_id, str) or not _GENERATION_ID_RE.fullmatch(
        generation_id
    ):
        raise InventoryError(
            "generation ID must be a safe caller-supplied physical ID: "
            f"{generation_id!r}"
        )


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """One exact output with a mandatory concrete schema validator."""

    path: str
    role: OutputRole
    schema: str
    validator: OutputValidator
    sibling: str | None = None
    drop_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_logical_path(self.path)
        if not isinstance(self.role, OutputRole):
            raise InventoryError(f"invalid output role for {self.path}: {self.role!r}")
        _validate_schema_version(self.schema)
        if not isinstance(
            self.validator,
            (JsonlValidator, JsonObjectValidator, BinaryValidator),
        ):
            raise InventoryError(
                f"concrete versioned validator required for {self.path}"
            )
        if self.validator.schema_version != self.schema:
            raise InventoryError(
                f"schema {self.schema!r} does not match validator "
                f"{self.validator.schema_version!r} for {self.path}"
            )
        if isinstance(self.validator, JsonlValidator) and self.validator.allow_empty:
            raise InventoryError(f"empty structured outputs are forbidden: {self.path}")
        if self.sibling is not None and (
            not isinstance(self.sibling, str) or not _SIBLING_RE.fullmatch(self.sibling)
        ):
            raise InventoryError(
                f"invalid sibling name for {self.path}: {self.sibling!r}"
            )
        drop_types = tuple(self.drop_types)
        if len(drop_types) != len(set(drop_types)):
            raise InventoryError(f"duplicate drop type declared for {self.path}")
        for drop_type in drop_types:
            if not isinstance(drop_type, str) or not _DROP_TYPE_RE.fullmatch(drop_type):
                raise InventoryError(
                    f"invalid declared drop type for {self.path}: {drop_type!r}"
                )
        if drop_types and self.role is not OutputRole.SIDECAR:
            raise InventoryError(
                f"drop types are only valid on sidecar outputs: {self.path}"
            )
        if self.role in {OutputRole.RAW, OutputRole.TRAIN, OutputRole.EVAL} and not (
            isinstance(self.validator, JsonlValidator) and self.path.endswith(".jsonl")
        ):
            raise InventoryError(
                f"accounted {self.role.value} output must use a JSONL validator: "
                f"{self.path}"
            )
        if self.role is OutputRole.HELDOUT and (
            not isinstance(self.validator, (JsonlValidator, JsonObjectValidator))
            or not self.validator.require_generation_links
        ):
            raise InventoryError(
                f"heldout output must validate generation/root links: {self.path}"
            )
        if (
            self.role is OutputRole.SIDECAR
            and isinstance(self.validator, (JsonlValidator, JsonObjectValidator))
            and not self.validator.require_generation_links
        ):
            raise InventoryError(
                f"structured sidecar must validate generation/root links: {self.path}"
            )
        if drop_types and (
            not isinstance(self.validator, JsonlValidator)
            or not self.validator.require_generation_links
        ):
            raise InventoryError(
                f"typed drop sidecar must validate generation/root links: {self.path}"
            )
        object.__setattr__(self, "drop_types", tuple(sorted(drop_types)))


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Complete declaration for one physical generation."""

    generation_id: str
    source_generation_id: str
    requested_siblings: tuple[str, ...]
    outputs: tuple[OutputSpec, ...]
    plan_root_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_generation_id(self.generation_id)
        if (
            not isinstance(self.source_generation_id, str)
            or not self.source_generation_id.strip()
            or any(ord(char) < 32 for char in self.source_generation_id)
        ):
            raise InventoryError("source generation ID must be a non-empty string")
        siblings = tuple(self.requested_siblings)
        if not siblings:
            raise InventoryError("at least one requested sibling is required")
        if len(siblings) != len(set(siblings)):
            raise InventoryError("requested sibling names must be unique")
        for sibling in siblings:
            if not isinstance(sibling, str) or not _SIBLING_RE.fullmatch(sibling):
                raise InventoryError(f"invalid requested sibling: {sibling!r}")
        object.__setattr__(self, "requested_siblings", siblings)

        outputs = tuple(self.outputs)
        if not outputs:
            raise InventoryError("output inventory must not be empty")
        object.__setattr__(self, "outputs", outputs)
        paths = [spec.path for spec in outputs]
        duplicates = sorted(path for path, count in Counter(paths).items() if count > 1)
        if duplicates:
            raise InventoryError(f"duplicate output path(s): {', '.join(duplicates)}")

        requested = set(siblings)
        for spec in outputs:
            if spec.sibling is not None and spec.sibling not in requested:
                raise InventoryError(
                    f"output {spec.path} names unrequested sibling {spec.sibling!r}"
                )
            if (
                spec.role in {OutputRole.RAW, OutputRole.TRAIN, OutputRole.EVAL}
                and spec.sibling is None
            ):
                raise InventoryError(
                    f"{spec.role.value} output requires a sibling: {spec.path}"
                )
        if not any(spec.role is OutputRole.HELDOUT for spec in outputs):
            raise InventoryError("inventory must declare at least one heldout file")
        for sibling in siblings:
            counts = {
                role: sum(
                    spec.sibling == sibling and spec.role is role for spec in outputs
                )
                for role in (OutputRole.RAW, OutputRole.TRAIN, OutputRole.EVAL)
            }
            if any(counts[role] != 1 for role in counts):
                raise InventoryError(
                    f"requested sibling {sibling} must declare exactly one raw, "
                    "train, eval output; got "
                    f"raw={counts[OutputRole.RAW]}, "
                    f"train={counts[OutputRole.TRAIN]}, "
                    f"eval={counts[OutputRole.EVAL]}"
                )
            if not any(
                spec.sibling == sibling
                and spec.role is OutputRole.SIDECAR
                and spec.drop_types
                for spec in outputs
            ):
                raise InventoryError(
                    f"requested sibling {sibling} must declare a typed drop sidecar"
                )

        plan_body = {
            "outputs": [
                _spec_descriptor(spec)
                for spec in sorted(outputs, key=lambda item: item.path)
            ],
            "requested_siblings": list(siblings),
            "schema_version": PLAN_SCHEMA_VERSION,
            "source_generation_id": self.source_generation_id,
        }
        object.__setattr__(
            self,
            "plan_root_sha256",
            hashlib.sha256(_canonical_json_bytes(plan_body)).hexdigest(),
        )


@dataclass(frozen=True, slots=True)
class RawOccurrence:
    """One stable physical occurrence of a native raw JSONL row."""

    occurrence_id: str
    sibling: str
    logical_source: str
    raw_row: int
    raw_sha256: str
    raw_bytes: bytes = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class DropRecord:
    """One typed drop route for a known raw occurrence."""

    occurrence_id: str
    drop_type: str
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PublishedGeneration:
    """A resolved generation and its commit state."""

    generation_id: str
    path: Path
    manifest_sha256: str
    logical_root_sha256: str
    manifest: Mapping[str, Any]
    commit_state: str = "durable"
    post_commit_warnings: tuple[str, ...] = ()


class GenerationWriter:
    """Staging-only writer that records physical occurrence routes."""

    def __init__(self, plan: GenerationPlan, staging_directory: Path):
        self.plan = plan
        self._staging_directory = staging_directory
        self._staging_descriptor = _open_directory_nofollow(
            staging_directory,
            error_type=InventoryError,
        )
        self._context = ValidationContext(
            generation_id=plan.generation_id,
            source_generation_id=plan.source_generation_id,
            plan_root_sha256=plan.plan_root_sha256,
        )
        self._specs = {spec.path: spec for spec in plan.outputs}
        self._registered: dict[str, tuple[int, int, int, int]] = {}
        self._raw_occurrences: dict[str, tuple[RawOccurrence, ...]] = {}
        self._occurrences_by_id: dict[str, RawOccurrence] = {}
        self._routes: list[dict[str, Any]] = []
        self._destination_rows: Counter[str] = Counter()
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise GenerationError("generation writer is already closed")

    def _spec(self, logical_path: str) -> OutputSpec:
        self._require_open()
        try:
            return self._specs[logical_path]
        except KeyError as error:
            raise InventoryError(
                f"output is not declared in the exact inventory: {logical_path}"
            ) from error

    def output_path(self, logical_path: str) -> Path:
        """Reject the removed writable-path adapter."""

        raise UnsafePathAPIError(
            "pathname staging access is unsafe; use a secure open_output/"
            "open_routed_output callback or copy_file"
        )

    def _private_path(self, logical_path: str) -> Path:
        return self._staging_directory.joinpath(*PurePosixPath(logical_path).parts)

    def _assert_staging_current(self) -> None:
        current = _open_directory_nofollow(
            self._staging_directory,
            error_type=InventoryError,
        )
        try:
            expected = os.fstat(self._staging_descriptor)
            actual = os.fstat(current)
            if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
                raise InventoryError("staging directory was replaced during generation")
        finally:
            os.close(current)

    def register_existing(self, logical_path: str) -> None:
        """Reject the removed register-after-path-write adapter."""

        raise UnsafePathAPIError(
            "pathname registration is unsafe; use a secure open_output callback "
            "or copy_file"
        )

    @contextmanager
    def _open_output_callback(
        self,
        logical_path: str,
        spec: OutputSpec,
    ) -> Iterator[BinaryIO]:
        if logical_path in self._registered:
            raise InventoryError(f"duplicate output write: {logical_path}")
        self._assert_staging_current()
        parent, leaf, descriptor = _open_exclusive_relative_output(
            self._staging_descriptor,
            logical_path,
            error_type=InventoryError,
        )
        opened = os.fstat(descriptor)
        handle = os.fdopen(descriptor, "wb", closefd=False)
        accepted = False
        try:
            yield handle
            handle.flush()
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            _validate_regular_metadata(metadata, logical_path, InventoryError)
            identity = _metadata_identity(metadata)
            self._assert_staging_current()
            current_identity = _safe_output_identity(
                self._staging_descriptor,
                logical_path,
                error_type=InventoryError,
            )
            if current_identity != identity:
                raise InventoryError(
                    f"output was replaced during secure callback: {logical_path}"
                )
            _validate_output(self._private_path(logical_path), spec, self._context)
            self._registered[logical_path] = identity
            os.fsync(parent)
            accepted = True
        finally:
            try:
                try:
                    handle.close()
                finally:
                    if not accepted:
                        _unlink_opened_output(
                            parent,
                            leaf,
                            opened,
                            error_type=InventoryError,
                        )
            finally:
                os.close(descriptor)
                os.close(parent)

    @contextmanager
    def open_output(self, logical_path: str) -> Iterator[BinaryIO]:
        """
        Yield an already-open, no-follow binary output stream.

        The stream is fsynced, schema-validated, and registered on successful
        context exit. Pathname access is never exposed.
        """

        spec = self._spec(logical_path)
        if spec.role in {OutputRole.TRAIN, OutputRole.EVAL}:
            raise InventoryError(
                f"{logical_path} requires open_routed_output with occurrence IDs"
            )
        with self._open_output_callback(logical_path, spec) as handle:
            yield handle

    def write_stream(self, logical_path: str, chunks: Iterable[bytes]) -> None:
        """Write bytes unchanged and fsync the completed staging file."""

        spec = self._spec(logical_path)
        if spec.role in {OutputRole.TRAIN, OutputRole.EVAL}:
            raise InventoryError(
                f"{logical_path} requires write_routed_jsonl with occurrence IDs"
            )
        if logical_path in self._registered:
            raise InventoryError(f"duplicate output write: {logical_path}")
        with self.open_output(logical_path) as handle:
            for chunk in chunks:
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError(f"output chunks must be bytes for {logical_path}")
                handle.write(chunk)

    def write_bytes(self, logical_path: str, payload: bytes) -> None:
        """Write one declared non-routed output without rewriting bytes."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(f"output payload must be bytes for {logical_path}")
        self.write_stream(logical_path, (payload,))

    def copy_file(
        self,
        logical_path: str,
        source: str | Path,
        *,
        chunk_bytes: int = 1024 * 1024,
        occurrence_ids: Sequence[str] | None = None,
    ) -> None:
        """
        Securely copy bytes from an external trusted temporary file.

        Tools that require a real pathname must finish their output outside the
        transaction staging tree, then pass that trusted file here.
        """

        spec = self._spec(logical_path)
        if spec.role in {OutputRole.TRAIN, OutputRole.EVAL}:
            if occurrence_ids is None:
                raise InventoryError(
                    f"{logical_path} requires occurrence_ids for secure routed copy"
                )
            output_context = self.open_routed_output(
                logical_path,
                occurrence_ids,
            )
        else:
            if occurrence_ids is not None:
                raise InventoryError(
                    f"{logical_path} is not routed and forbids occurrence_ids"
                )
            output_context = self.open_output(logical_path)

        source_path = Path(source)
        descriptor = _open_regular_file_nofollow(
            source_path,
            error_type=InventoryError,
        )
        try:
            with (
                os.fdopen(descriptor, "rb", closefd=False) as handle,
                output_context as output,
            ):
                while chunk := handle.read(chunk_bytes):
                    output.write(chunk)
        finally:
            os.close(descriptor)

    def raw_occurrences(self, logical_path: str) -> tuple[RawOccurrence, ...]:
        """Index and return stable physical occurrences from one raw output."""

        spec = self._spec(logical_path)
        if spec.role is not OutputRole.RAW or spec.sibling is None:
            raise AccountingError(f"not a declared raw sibling output: {logical_path}")
        if logical_path not in self._registered:
            raise InventoryError(f"raw output is not registered: {logical_path}")
        if logical_path in self._raw_occurrences:
            return self._raw_occurrences[logical_path]
        self._assert_staging_current()
        current_identity = _safe_output_identity(
            self._staging_descriptor,
            logical_path,
            error_type=InventoryError,
        )
        if current_identity != self._registered[logical_path]:
            raise InventoryError(f"raw output was replaced: {logical_path}")
        path = self._private_path(logical_path)
        _validate_output(path, spec, self._context)
        occurrences = _raw_occurrences(path, spec)
        for occurrence in occurrences:
            if occurrence.occurrence_id in self._occurrences_by_id:
                raise AccountingError(
                    f"duplicate physical occurrence ID: {occurrence.occurrence_id}"
                )
            self._occurrences_by_id[occurrence.occurrence_id] = occurrence
        result = tuple(occurrences)
        self._raw_occurrences[logical_path] = result
        return result

    def write_routed_jsonl(
        self,
        logical_path: str,
        occurrences: Iterable[RawOccurrence],
    ) -> None:
        """Write train/eval rows from exact raw occurrences."""

        spec = self._spec(logical_path)
        if spec.role not in {OutputRole.TRAIN, OutputRole.EVAL}:
            raise AccountingError(
                f"output is not train/eval routed JSONL: {logical_path}"
            )
        routed = tuple(occurrences)
        self._validate_route_occurrences(spec, routed)
        with self.open_routed_output(
            logical_path,
            [occurrence.occurrence_id for occurrence in routed],
        ) as handle:
            for occurrence in routed:
                handle.write(occurrence.raw_bytes)

    @contextmanager
    def open_routed_output(
        self,
        logical_path: str,
        occurrence_ids: Sequence[str],
    ) -> Iterator[BinaryIO]:
        """
        Yield a secure native train/eval stream linked to raw occurrences.

        Native bytes are validated against the exact occurrence sequence and
        routes are registered only after successful context exit.
        """

        spec = self._spec(logical_path)
        if spec.role not in {OutputRole.TRAIN, OutputRole.EVAL}:
            raise AccountingError(
                f"output is not train/eval routed JSONL: {logical_path}"
            )
        occurrences = tuple(self._lookup_occurrence(item) for item in occurrence_ids)
        self._validate_route_occurrences(spec, occurrences)
        with self._open_output_callback(logical_path, spec) as handle:
            yield handle
        lines = _read_relative_binary_lines(
            self._staging_descriptor,
            logical_path,
            error_type=InventoryError,
        )
        if len(lines) != len(occurrences):
            raise AccountingError(
                f"{logical_path}: route count does not match native output rows"
            )
        for row, (line, occurrence) in enumerate(zip(lines, occurrences), start=1):
            if line != occurrence.raw_bytes:
                raise AccountingError(
                    f"{logical_path}:{row}: native bytes do not match occurrence"
                )
            self._append_route(
                spec,
                occurrence,
                destination_row=row,
                drop_type=None,
            )

    def register_routed_existing(
        self,
        logical_path: str,
        occurrence_ids: Sequence[str],
    ) -> None:
        """Reject the removed register-after-path-write adapter."""

        raise UnsafePathAPIError(
            "pathname registration is unsafe; use a secure open_routed_output callback"
        )

    def _validate_route_occurrences(
        self,
        spec: OutputSpec,
        occurrences: Sequence[RawOccurrence],
    ) -> None:
        for occurrence in occurrences:
            known = self._occurrences_by_id.get(occurrence.occurrence_id)
            if known != occurrence:
                raise AccountingError(
                    f"unknown raw occurrence: {occurrence.occurrence_id}"
                )
            if known.sibling != spec.sibling:
                raise AccountingError(
                    f"cross-sibling occurrence route: {known.sibling} -> {spec.sibling}"
                )

    def _lookup_occurrence(self, occurrence_id: str) -> RawOccurrence:
        try:
            return self._occurrences_by_id[occurrence_id]
        except KeyError as error:
            raise AccountingError(f"unknown raw occurrence: {occurrence_id}") from error

    def _append_route(
        self,
        spec: OutputSpec,
        occurrence: RawOccurrence,
        *,
        destination_row: int,
        drop_type: str | None,
    ) -> None:
        disposition = "drop" if spec.role is OutputRole.SIDECAR else spec.role.value
        self._routes.append(
            {
                "destination_path": spec.path,
                "destination_row": destination_row,
                "disposition": disposition,
                "drop_type": drop_type,
                "occurrence_id": occurrence.occurrence_id,
                "plan_root_sha256": self.plan.plan_root_sha256,
                "raw_path": occurrence.logical_source,
                "raw_row": occurrence.raw_row,
                "raw_sha256": occurrence.raw_sha256,
                "sibling": occurrence.sibling,
            }
        )
        self._destination_rows[spec.path] += 1

    def write_drop_sidecar(
        self,
        logical_path: str,
        records: Iterable[DropRecord],
    ) -> None:
        """Write canonical typed drop records linked to raw occurrences."""

        spec = self._spec(logical_path)
        if spec.role is not OutputRole.SIDECAR or not spec.drop_types:
            raise AccountingError(
                f"output is not a declared typed drop sidecar: {logical_path}"
            )
        records_tuple = tuple(records)
        encoded: list[bytes] = []
        routed: list[tuple[RawOccurrence, str]] = []
        for record in records_tuple:
            if not isinstance(record, DropRecord):
                raise AccountingError(f"non-DropRecord in {logical_path}")
            occurrence = self._lookup_occurrence(record.occurrence_id)
            if occurrence.sibling != spec.sibling:
                raise AccountingError(
                    f"cross-sibling occurrence route: "
                    f"{occurrence.sibling} -> {spec.sibling}"
                )
            if (
                not isinstance(record.drop_type, str)
                or record.drop_type not in spec.drop_types
            ):
                raise AccountingError(
                    f"drop type must be named and declared for {logical_path}: "
                    f"{record.drop_type!r}"
                )
            details = {} if record.details is None else record.details
            if not isinstance(details, Mapping):
                raise AccountingError("drop record details must be a mapping")
            payload = {
                "details": dict(details),
                "drop_type": record.drop_type,
                "generation_id": self.plan.generation_id,
                "occurrence_id": occurrence.occurrence_id,
                "plan_root_sha256": self.plan.plan_root_sha256,
                "raw_path": occurrence.logical_source,
                "raw_row": occurrence.raw_row,
                "raw_sha256": occurrence.raw_sha256,
                "schema_version": spec.schema,
                "sibling": occurrence.sibling,
                "source_generation_id": self.plan.source_generation_id,
            }
            try:
                encoded.append(_canonical_json_bytes(payload))
            except (TypeError, ValueError) as error:
                raise AccountingError(
                    f"drop record details are not JSON serializable: {error}"
                ) from error
            routed.append((occurrence, record.drop_type))
        self.write_stream(logical_path, encoded)
        for destination_row, (occurrence, drop_type) in enumerate(routed, start=1):
            self._append_route(
                spec,
                occurrence,
                destination_row=destination_row,
                drop_type=drop_type,
            )

    def write_linked_json(
        self,
        logical_path: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Write a structured JSON object with generation and plan-root links."""

        spec = self._spec(logical_path)
        if not isinstance(spec.validator, JsonObjectValidator):
            raise ValidationError(f"{logical_path} is not a JSON object output")
        if not spec.validator.require_generation_links:
            raise ValidationError(f"{logical_path} does not require generation links")
        if not isinstance(payload, Mapping):
            raise ValidationError("linked JSON payload must be a mapping")
        reserved = _RESERVED_LINK_FIELDS & set(payload)
        if reserved:
            raise ValidationError(
                f"linked JSON payload sets reserved fields: {sorted(reserved)}"
            )
        linked = {
            **dict(payload),
            "generation_id": self.plan.generation_id,
            "plan_root_sha256": self.plan.plan_root_sha256,
            "schema_version": spec.schema,
            "source_generation_id": self.plan.source_generation_id,
        }
        self.write_bytes(logical_path, _canonical_json_bytes(linked))

    def _write_routes(self) -> None:
        routes = sorted(
            self._routes,
            key=lambda item: (
                item["sibling"],
                item["raw_path"],
                item["raw_row"],
                item["destination_path"],
                item["destination_row"],
            ),
        )
        self._assert_staging_current()
        _write_relative_chunks(
            self._staging_descriptor,
            ROUTES_FILENAME,
            (_canonical_json_bytes(route) for route in routes),
            error_type=InventoryError,
        )

    def _close(self) -> None:
        self._closed = True

    def _release(self) -> None:
        if self._staging_descriptor >= 0:
            os.close(self._staging_descriptor)
            self._staging_descriptor = -1


class GenerationCoordinator:
    """Coordinate validated staging, sealing, and one explicit commit point."""

    commit_point = COMMIT_POINT

    def __init__(
        self,
        root: str | Path,
        *,
        binary_validators: Iterable[BinaryValidator] = (),
    ):
        self.root = Path(root)
        self._binary_validators = {}
        for validator in binary_validators:
            key = (validator.schema_version, validator.validator_id)
            if key in self._binary_validators:
                raise InventoryError(f"duplicate binary validator: {key}")
            self._binary_validators[key] = validator

    @property
    def generations_directory(self) -> Path:
        return self.root / "generations"

    @property
    def staging_directory(self) -> Path:
        return self.root / ".staging"

    @property
    def quarantine_directory(self) -> Path:
        return self.root / "quarantine"

    @property
    def transactions_directory(self) -> Path:
        return self.root / "transactions"

    def publish(
        self,
        plan: GenerationPlan,
        producer: Producer,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> PublishedGeneration:
        """Publish at the successful atomic replacement of CURRENT."""

        if not isinstance(plan, GenerationPlan):
            raise TypeError("plan must be a GenerationPlan")
        if not callable(producer):
            raise TypeError("producer must be callable")
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault injector must be callable")
        self._ensure_layout()
        with self._exclusive_lock():
            self._quarantine_stale_staging()
            self._quarantine_stale_pointer_temps()
            self._quarantine_stale_transaction_temps()
            self._reconcile_transaction_state()
            immutable = self.generations_directory / plan.generation_id
            pending_state = self._pending_transaction_path(plan.generation_id)
            committed_state = self._committed_transaction_path(plan.generation_id)
            if (
                immutable.exists()
                or immutable.is_symlink()
                or pending_state.exists()
                or committed_state.exists()
            ):
                raise GenerationExistsError(
                    f"immutable generation ID already exists: {plan.generation_id}"
                )
            staging = _safe_create_child_directory(
                self.staging_directory,
                f"{plan.generation_id}.staging",
                mode=0o700,
                error_type=GenerationError,
            )
            try:
                writer = GenerationWriter(plan, staging)
            except BaseException as error:
                if staging.exists() or staging.is_symlink():
                    self._quarantine_path(
                        staging,
                        reason=f"unsafe staging initialization: {error}",
                        generation_id=plan.generation_id,
                    )
                raise
            pointer_temporary = self.root / f".CURRENT.{plan.generation_id}.tmp"
            committed = False
            durable = False
            immutable_created = False
            manifest: dict[str, Any] | None = None
            manifest_sha256 = ""
            logical_root_sha256 = ""
            transaction_record: dict[str, Any] | None = None

            try:
                _inject(fault_injector, PublishPhase.STAGING_CREATED, staging)
                producer(writer)
                writer._close()
                _inject(fault_injector, PublishPhase.OUTPUTS_WRITTEN, staging)

                writer._write_routes()
                manifest = self._build_manifest(plan, writer)
                writer._assert_staging_current()
                _write_relative_chunks(
                    writer._staging_descriptor,
                    MANIFEST_FILENAME,
                    (_canonical_json_bytes(manifest),),
                    error_type=InventoryError,
                )
                manifest_sha256 = _sha256_file(staging / MANIFEST_FILENAME)
                logical_root_sha256 = str(manifest["logical_root_sha256"])
                _inject(fault_injector, PublishPhase.MANIFEST_WRITTEN, staging)
                validated = self._validate_generation_directory(
                    staging,
                    expected_plan=plan,
                    require_sealed=False,
                )
                _inject(fault_injector, PublishPhase.VALIDATED, staging)

                _fsync_tree(staging)
                _inject(fault_injector, PublishPhase.STAGING_SYNCED, staging)
                transaction_record = {
                    "generation_id": plan.generation_id,
                    "logical_root_sha256": logical_root_sha256,
                    "manifest_sha256": manifest_sha256,
                    "schema_version": TRANSACTION_STATE_SCHEMA_VERSION,
                    "state": "pending",
                }
                _write_atomic_control_file(
                    pending_state,
                    _canonical_json_bytes(transaction_record),
                )

                _inject(
                    fault_injector,
                    PublishPhase.GENERATION_RENAME_BEFORE,
                    staging,
                )
                _safe_rename_child(
                    self.staging_directory,
                    staging.name,
                    self.generations_directory,
                    immutable.name,
                    error_type=GenerationError,
                )
                immutable_created = True
                _inject(
                    fault_injector,
                    PublishPhase.GENERATION_RENAME_AFTER,
                    immutable,
                )
                _seal_tree(immutable)
                _fsync_tree(immutable)
                _inject(
                    fault_injector,
                    PublishPhase.GENERATIONS_FSYNC_BEFORE,
                    immutable,
                )
                _fsync_directory(self.staging_directory)
                _fsync_directory(self.generations_directory)
                _inject(
                    fault_injector,
                    PublishPhase.GENERATIONS_FSYNC_AFTER,
                    immutable,
                )
                self._validate_generation_directory(
                    immutable,
                    expected_plan=plan,
                    expected_generation_id=plan.generation_id,
                    require_sealed=True,
                )

                seal = {
                    "generation_id": plan.generation_id,
                    "logical_root_sha256": logical_root_sha256,
                    "manifest_sha256": manifest_sha256,
                    "schema_version": CURRENT_SCHEMA_VERSION,
                }
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_TEMP_WRITE_BEFORE,
                    immutable,
                )
                _write_durable_file(
                    pointer_temporary,
                    _canonical_json_bytes(seal),
                    exclusive=True,
                )
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_TEMP_WRITE_AFTER,
                    immutable,
                )
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_TEMP_PARENT_FSYNC_BEFORE,
                    immutable,
                )
                _fsync_directory(self.root)
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_TEMP_PARENT_FSYNC_AFTER,
                    immutable,
                )
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_REPLACE_BEFORE,
                    immutable,
                )
                _safe_rename_child(
                    self.root,
                    pointer_temporary.name,
                    self.root,
                    CURRENT_FILENAME,
                    error_type=GenerationError,
                )
                committed = True
                _inject(
                    fault_injector,
                    PublishPhase.CURRENT_REPLACE_AFTER,
                    immutable,
                )
                _inject(
                    fault_injector,
                    PublishPhase.ROOT_FSYNC_BEFORE,
                    immutable,
                )
                _fsync_directory(self.root)
                durable = True
                warnings: list[str] = []
                try:
                    _inject(
                        fault_injector,
                        PublishPhase.ROOT_FSYNC_AFTER,
                        immutable,
                    )
                except Exception as warning:  # noqa: BLE001 - commit is already durable.
                    warnings.append(f"{type(warning).__name__}: {warning}")
                try:
                    self._promote_transaction_state(
                        pending_state,
                        committed_state,
                        transaction_record,
                    )
                except Exception as warning:  # noqa: BLE001 - commit is already durable.
                    warnings.append(f"{type(warning).__name__}: {warning}")
                writer._release()
                return PublishedGeneration(
                    generation_id=plan.generation_id,
                    path=immutable,
                    manifest_sha256=manifest_sha256,
                    logical_root_sha256=logical_root_sha256,
                    manifest=validated,
                    post_commit_warnings=tuple(warnings),
                )
            except BaseException as error:
                writer._close()
                writer._release()
                if (
                    not committed
                    and manifest_sha256
                    and logical_root_sha256
                    and self._current_matches(
                        plan.generation_id,
                        manifest_sha256,
                        logical_root_sha256,
                    )
                ):
                    committed = True
                if committed:
                    if durable and manifest is not None:
                        return PublishedGeneration(
                            generation_id=plan.generation_id,
                            path=immutable,
                            manifest_sha256=manifest_sha256,
                            logical_root_sha256=logical_root_sha256,
                            manifest=manifest,
                            post_commit_warnings=(f"{type(error).__name__}: {error}",),
                        )
                    raise CommitUncertainError(
                        plan.generation_id,
                        logical_root_sha256,
                        error,
                    ) from error

                if immutable_created and immutable.exists():
                    self._quarantine_path(
                        immutable,
                        reason=f"{type(error).__name__}: {error}",
                        generation_id=plan.generation_id,
                    )
                elif staging.exists() or staging.is_symlink():
                    self._quarantine_path(
                        staging,
                        reason=f"{type(error).__name__}: {error}",
                        generation_id=plan.generation_id,
                    )
                if pointer_temporary.exists() or pointer_temporary.is_symlink():
                    self._quarantine_path(
                        pointer_temporary,
                        reason=f"unpublished pointer: {type(error).__name__}: {error}",
                        generation_id=plan.generation_id,
                    )
                if pending_state.exists() or pending_state.is_symlink():
                    self._quarantine_path(
                        pending_state,
                        reason=f"uncommitted transaction: {type(error).__name__}: {error}",
                        generation_id=plan.generation_id,
                    )
                raise

    def resolve_current(
        self,
        *,
        expected_plan: GenerationPlan | None = None,
        required_siblings: Sequence[str] = (),
    ) -> PublishedGeneration:
        """Resolve only a sealed, read-only generation selected by CURRENT."""

        _require_secure_filesystem_capabilities()
        _require_real_directory(self.root, "generation root")
        pointer_path = self.root / CURRENT_FILENAME
        _require_regular_file(pointer_path, CURRENT_FILENAME, ValidationError)
        pointer = _read_json_object(pointer_path, "CURRENT seal")
        expected_keys = {
            "generation_id",
            "logical_root_sha256",
            "manifest_sha256",
            "schema_version",
        }
        if set(pointer) != expected_keys:
            raise ValidationError("CURRENT seal fields are not exact")
        if pointer["schema_version"] != CURRENT_SCHEMA_VERSION:
            raise ValidationError("unsupported CURRENT schema")
        generation_id = pointer["generation_id"]
        try:
            _validate_generation_id(generation_id)
        except InventoryError as error:
            raise ValidationError(f"invalid CURRENT generation ID: {error}") from error
        for name in ("manifest_sha256", "logical_root_sha256"):
            if not isinstance(pointer[name], str) or not _SHA256_RE.fullmatch(
                pointer[name]
            ):
                raise ValidationError(f"CURRENT {name} is malformed")
        if expected_plan is not None and expected_plan.generation_id != generation_id:
            raise ValidationError(
                f"CURRENT selects {generation_id}, expected "
                f"{expected_plan.generation_id}"
            )
        _require_real_directory(
            self.generations_directory,
            "generations control path",
        )
        generation = self.generations_directory / generation_id
        if not generation.is_dir() or generation.is_symlink():
            raise ValidationError(
                "CURRENT generation directory is missing or a symlink"
            )
        manifest_path = generation / MANIFEST_FILENAME
        _require_regular_file(manifest_path, MANIFEST_FILENAME, ValidationError)
        actual_manifest_sha256 = _sha256_file(manifest_path)
        if actual_manifest_sha256 != pointer["manifest_sha256"]:
            raise ValidationError("CURRENT manifest SHA-256 mismatch")
        manifest = self._validate_generation_directory(
            generation,
            expected_plan=expected_plan,
            expected_generation_id=generation_id,
            require_sealed=True,
        )
        if manifest["logical_root_sha256"] != pointer["logical_root_sha256"]:
            raise ValidationError("CURRENT logical root mismatch")
        required = set(required_siblings)
        actual = set(manifest["requested_siblings"])
        missing = sorted(required - actual)
        if missing:
            raise ValidationError(
                f"required sibling family is missing: {', '.join(missing)}"
            )
        commit_state = self._transaction_commit_state(pointer)
        return PublishedGeneration(
            generation_id=generation_id,
            path=generation,
            manifest_sha256=actual_manifest_sha256,
            logical_root_sha256=manifest["logical_root_sha256"],
            manifest=manifest,
            commit_state=commit_state,
        )

    def quarantine_inventory(self) -> list[dict[str, Any]]:
        """Return validated quarantine entries; reject untracked material."""

        _require_secure_filesystem_capabilities()
        if not self.quarantine_directory.is_dir():
            return []
        records = []
        for entry in sorted(
            self.quarantine_directory.iterdir(), key=lambda item: item.name
        ):
            if not entry.is_dir() or entry.is_symlink():
                raise ValidationError(f"uninventoried quarantine path: {entry.name}")
            marker = entry / QUARANTINE_FILENAME
            if not marker.is_file() or marker.is_symlink():
                raise ValidationError(f"uninventoried quarantine entry: {entry.name}")
            children = {child.name for child in entry.iterdir()}
            if children != {QUARANTINE_FILENAME, "payload"}:
                raise ValidationError(
                    f"extra uninventoried quarantine material: {entry.name}"
                )
            record = _read_json_object(marker, "quarantine marker")
            if set(record) != _QUARANTINE_KEYS or record.get("entry") != entry.name:
                raise ValidationError(f"invalid quarantine inventory: {entry.name}")
            if record.get("schema_version") != "generation-quarantine/v1":
                raise ValidationError(f"invalid quarantine schema: {entry.name}")
            if record.get("kind") not in {"file", "directory"}:
                raise ValidationError(f"invalid quarantine kind: {entry.name}")
            if (
                not isinstance(record.get("reason"), str)
                or not record["reason"]
                or not isinstance(record.get("original_name"), str)
                or not record["original_name"]
                or "/" in record["original_name"]
                or "\\" in record["original_name"]
            ):
                raise ValidationError(f"invalid quarantine metadata: {entry.name}")
            generation_id = record.get("generation_id")
            if generation_id is not None:
                try:
                    _validate_generation_id(generation_id)
                except InventoryError as error:
                    raise ValidationError(
                        f"invalid quarantine generation ID: {entry.name}"
                    ) from error
            payload = entry / "payload"
            if not payload.exists() and not payload.is_symlink():
                raise ValidationError(f"quarantine payload is missing: {entry.name}")
            payload_metadata = payload.lstat()
            if stat.S_ISLNK(payload_metadata.st_mode):
                raise ValidationError(f"quarantine payload is a symlink: {entry.name}")
            actual_kind = (
                "directory"
                if stat.S_ISDIR(payload_metadata.st_mode)
                else "file"
                if stat.S_ISREG(payload_metadata.st_mode)
                else "invalid"
            )
            if actual_kind != record["kind"]:
                raise ValidationError(f"quarantine payload kind mismatch: {entry.name}")
            records.append(record)
        return records

    def _ensure_layout(self) -> None:
        _require_secure_filesystem_capabilities()
        _mkdir_durable(self.root)
        if not self.root.is_dir() or self.root.is_symlink():
            raise GenerationError(
                f"generation root is not a real directory: {self.root}"
            )
        for directory in (
            self.generations_directory,
            self.staging_directory,
            self.quarantine_directory,
            self.transactions_directory,
        ):
            if directory.exists() and (
                not directory.is_dir() or directory.is_symlink()
            ):
                raise GenerationError(f"control path is not a directory: {directory}")
            _mkdir_durable(directory)
        _fsync_directory(self.generations_directory)
        _fsync_directory(self.staging_directory)
        _fsync_directory(self.quarantine_directory)
        _fsync_directory(self.transactions_directory)
        _fsync_directory(self.root)

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        lock_path = self.root / ".generation.lock"
        flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
        parent = _open_directory_nofollow(self.root, error_type=GenerationError)
        try:
            descriptor = os.open(
                lock_path.name,
                flags,
                0o600,
                dir_fd=parent,
            )
            os.fsync(parent)
        finally:
            os.close(parent)
        try:
            _validate_regular_metadata(
                os.fstat(descriptor),
                str(lock_path),
                GenerationError,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _pending_transaction_path(self, generation_id: str) -> Path:
        return self.transactions_directory / f"{generation_id}.pending.json"

    def _committed_transaction_path(self, generation_id: str) -> Path:
        return self.transactions_directory / f"{generation_id}.committed.json"

    def _promote_transaction_state(
        self,
        pending: Path,
        committed: Path,
        record: Mapping[str, Any] | None,
    ) -> None:
        if record is None:
            raise ValidationError("transaction record is missing")
        _validate_transaction_record(record, expected_state="pending")
        committed_record = {**dict(record), "state": "committed"}
        if committed.exists() or committed.is_symlink():
            existing = _read_json_object(committed, "committed transaction state")
            _validate_transaction_record(existing, expected_state="committed")
            if existing != committed_record:
                raise ValidationError("committed transaction state conflicts")
        else:
            _write_atomic_control_file(
                committed,
                _canonical_json_bytes(committed_record),
            )
        if pending.exists():
            _safe_unlink_child(
                self.transactions_directory,
                pending.name,
                error_type=ValidationError,
            )

    def _transaction_commit_state(self, pointer: Mapping[str, Any]) -> str:
        _require_real_directory(
            self.transactions_directory,
            "transactions control path",
        )
        generation_id = pointer["generation_id"]
        committed = self._committed_transaction_path(generation_id)
        pending = self._pending_transaction_path(generation_id)
        if committed.is_file() and not committed.is_symlink():
            record = _read_json_object(committed, "committed transaction state")
            _validate_transaction_record(record, expected_state="committed")
            _validate_transaction_matches_pointer(record, pointer)
            return "durable"
        if pending.is_file() and not pending.is_symlink():
            record = _read_json_object(pending, "pending transaction state")
            _validate_transaction_record(record, expected_state="pending")
            _validate_transaction_matches_pointer(record, pointer)
            return "visible_not_durable"
        raise ValidationError(
            f"CURRENT transaction state is missing for {generation_id}"
        )

    def _current_matches(
        self,
        generation_id: str,
        manifest_sha256: str,
        logical_root_sha256: str,
    ) -> bool:
        pointer_path = self.root / CURRENT_FILENAME
        try:
            metadata = pointer_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                return False
            pointer = _read_json_object(pointer_path, "CURRENT seal")
        except (FileNotFoundError, ValidationError):
            return False
        return pointer == {
            "generation_id": generation_id,
            "logical_root_sha256": logical_root_sha256,
            "manifest_sha256": manifest_sha256,
            "schema_version": CURRENT_SCHEMA_VERSION,
        }

    def _recover_visible_transaction(
        self,
        generation_id: str,
        logical_root_sha256: str,
    ) -> None:
        with self._exclusive_lock():
            pointer = _read_json_object(
                self.root / CURRENT_FILENAME,
                "CURRENT seal",
            )
            if (
                pointer.get("generation_id") != generation_id
                or pointer.get("logical_root_sha256") != logical_root_sha256
            ):
                raise ValidationError(
                    "commit-uncertain CURRENT changed before recovery"
                )
            _fsync_directory(self.root)
            pending = self._pending_transaction_path(generation_id)
            committed = self._committed_transaction_path(generation_id)
            if pending.is_file() and not pending.is_symlink():
                record = _read_json_object(pending, "pending transaction state")
                self._promote_transaction_state(pending, committed, record)
            elif not committed.is_file() or committed.is_symlink():
                raise ValidationError(
                    f"transaction state is missing for {generation_id}"
                )

    def _reconcile_transaction_state(self) -> None:
        current: dict[str, Any] | None = None
        current_path = self.root / CURRENT_FILENAME
        if current_path.exists() or current_path.is_symlink():
            _require_regular_file(current_path, CURRENT_FILENAME, ValidationError)
            current = _read_json_object(current_path, "CURRENT seal")

        known_names: set[str] = set()
        committed_ids: set[str] = set()
        pending_records: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted(self.transactions_directory.iterdir()):
            if (
                not path.is_file()
                or path.is_symlink()
                or not (
                    path.name.endswith(".pending.json")
                    or path.name.endswith(".committed.json")
                )
            ):
                raise ValidationError(f"uninventoried transaction state: {path.name}")
            record = _read_json_object(path, "transaction state")
            expected_state = (
                "pending" if path.name.endswith(".pending.json") else "committed"
            )
            _validate_transaction_record(record, expected_state=expected_state)
            expected_name = f"{record['generation_id']}.{expected_state}.json"
            if path.name != expected_name:
                raise ValidationError(
                    f"transaction state filename mismatch: {path.name}"
                )
            if path.name in known_names:
                raise ValidationError(f"duplicate transaction state: {path.name}")
            known_names.add(path.name)
            if expected_state == "committed":
                committed_ids.add(record["generation_id"])
            else:
                pending_records.append((path, record))

        current_id = current.get("generation_id") if current is not None else None
        for pending, record in pending_records:
            generation_id = record["generation_id"]
            committed = self._committed_transaction_path(generation_id)
            if generation_id in committed_ids:
                committed_record = _read_json_object(
                    committed,
                    "committed transaction state",
                )
                expected = {**record, "state": "committed"}
                if committed_record != expected:
                    raise ValidationError(
                        f"conflicting transaction states for {generation_id}"
                    )
                _safe_unlink_child(
                    self.transactions_directory,
                    pending.name,
                    error_type=ValidationError,
                )
                continue
            if current_id == generation_id and current is not None:
                _validate_transaction_matches_pointer(record, current)
                _fsync_directory(self.root)
                self._promote_transaction_state(pending, committed, record)
                committed_ids.add(generation_id)
                continue
            generation = self.generations_directory / generation_id
            if generation.exists() or generation.is_symlink():
                self._quarantine_path(
                    generation,
                    reason="uncommitted generation recovered after interruption",
                    generation_id=generation_id,
                )
            self._quarantine_path(
                pending,
                reason="uncommitted transaction recovered after interruption",
                generation_id=generation_id,
            )

        for generation in sorted(self.generations_directory.iterdir()):
            if generation.name not in committed_ids:
                if current_id == generation.name:
                    raise ValidationError(
                        f"CURRENT generation lacks transaction state: {generation.name}"
                    )
                self._quarantine_path(
                    generation,
                    reason="generation has no committed transaction state",
                    generation_id=(
                        generation.name
                        if _GENERATION_ID_RE.fullmatch(generation.name)
                        else None
                    ),
                )
        for generation_id in committed_ids:
            generation = self.generations_directory / generation_id
            if not generation.is_dir() or generation.is_symlink():
                raise ValidationError(
                    f"committed generation is missing: {generation_id}"
                )

    def _quarantine_stale_staging(self) -> None:
        for path in sorted(
            self.staging_directory.iterdir(), key=lambda item: item.name
        ):
            generation_id = path.name.removesuffix(".staging")
            self._quarantine_path(
                path,
                reason="stale or malformed staging generation",
                generation_id=(
                    generation_id
                    if _GENERATION_ID_RE.fullmatch(generation_id)
                    else None
                ),
            )

    def _quarantine_stale_pointer_temps(self) -> None:
        for path in sorted(
            self.root.glob(".CURRENT.*.tmp"), key=lambda item: item.name
        ):
            self._quarantine_path(
                path,
                reason="stale unpublished CURRENT temporary",
                generation_id=None,
            )

    def _quarantine_stale_transaction_temps(self) -> None:
        for path in sorted(
            self.transactions_directory.glob(".*.tmp"),
            key=lambda item: item.name,
        ):
            self._quarantine_path(
                path,
                reason="stale transaction-state temporary",
                generation_id=None,
            )

    def _quarantine_path(
        self,
        path: Path,
        *,
        reason: str,
        generation_id: str | None,
    ) -> Path:
        _make_tree_writable(path)
        base = generation_id or _safe_quarantine_name(path.name)
        ordinal = 1
        while True:
            entry_name = f"{base}.quarantine-{ordinal:04d}"
            destination = self.quarantine_directory / entry_name
            if not destination.exists():
                break
            ordinal += 1
        destination.mkdir(mode=0o700)
        payload = destination / "payload"
        os.rename(path, payload)
        kind = "directory" if payload.is_dir() else "file"
        record = {
            "entry": entry_name,
            "generation_id": generation_id,
            "kind": kind,
            "original_name": path.name,
            "reason": reason,
            "schema_version": "generation-quarantine/v1",
        }
        _write_durable_file(
            destination / QUARANTINE_FILENAME,
            _canonical_json_bytes(record),
            exclusive=True,
        )
        _fsync_directory(destination)
        _fsync_directory(path.parent)
        _fsync_directory(self.quarantine_directory)
        return destination

    def _build_manifest(
        self,
        plan: GenerationPlan,
        writer: GenerationWriter,
    ) -> dict[str, Any]:
        declared = {spec.path for spec in plan.outputs}
        expected = declared | {ROUTES_FILENAME}
        expected_directories = _expected_directory_inventory(expected)
        writer._assert_staging_current()
        actual_files, actual_directories = _physical_inventory(
            writer._staging_directory,
            error_type=InventoryError,
        )
        missing = sorted(expected - actual_files)
        extra = sorted(actual_files - expected)
        if missing:
            raise InventoryError(f"missing declared output files: {', '.join(missing)}")
        if extra:
            raise InventoryError(f"extra unexpected output files: {', '.join(extra)}")
        missing_directories = sorted(expected_directories - actual_directories)
        extra_directories = sorted(actual_directories - expected_directories)
        if missing_directories:
            raise InventoryError(
                "missing declared directories: " + ", ".join(missing_directories)
            )
        if extra_directories:
            raise InventoryError(
                "undeclared directories: " + ", ".join(extra_directories)
            )
        unregistered = sorted(declared - set(writer._registered))
        if unregistered:
            raise InventoryError(
                "unregistered declared output files: " + ", ".join(unregistered)
            )
        for logical_path, identity in writer._registered.items():
            current_identity = _safe_output_identity(
                writer._staging_descriptor,
                logical_path,
                error_type=InventoryError,
            )
            if current_identity != identity:
                raise InventoryError(
                    f"unregistered or replaced output after registration: {logical_path}"
                )

        metadata = [
            _output_metadata(
                writer._staging_directory,
                spec,
                context=writer._context,
            )
            for spec in sorted(plan.outputs, key=lambda item: item.path)
        ]
        accounting = _validate_accounting(writer._staging_directory, plan)
        routes_path = writer._staging_directory / ROUTES_FILENAME
        routes_rows = len(_read_jsonl_objects(routes_path))
        routes_sha256 = _sha256_file(routes_path)
        routes = {
            "bytes": routes_path.stat().st_size,
            "path": ROUTES_FILENAME,
            "root_sha256": routes_sha256,
            "rows": routes_rows,
            "schema_version": ROUTES_SCHEMA_VERSION,
            "sha256": routes_sha256,
        }
        logical_root = _logical_root(plan, metadata, routes)
        body = {
            "accounting": accounting,
            "api_version": API_VERSION,
            "directories": sorted(expected_directories),
            "generation_id": plan.generation_id,
            "logical_root_sha256": logical_root,
            "outputs": metadata,
            "physical_generation_id_policy": PHYSICAL_ID_POLICY,
            "plan_root_sha256": plan.plan_root_sha256,
            "requested_siblings": list(plan.requested_siblings),
            "routes": routes,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_generation_id": plan.source_generation_id,
        }
        return {
            **body,
            "manifest_root_sha256": hashlib.sha256(
                _canonical_json_bytes(body)
            ).hexdigest(),
        }

    def _validate_generation_directory(
        self,
        generation: Path,
        *,
        expected_plan: GenerationPlan | None,
        expected_generation_id: str | None = None,
        require_sealed: bool,
    ) -> dict[str, Any]:
        if not generation.is_dir() or generation.is_symlink():
            raise ValidationError("generation is not a real directory")
        manifest_path = generation / MANIFEST_FILENAME
        _require_regular_file(manifest_path, MANIFEST_FILENAME, ValidationError)
        manifest = _read_json_object(manifest_path, "generation manifest")
        if set(manifest) != _MANIFEST_KEYS:
            raise ValidationError("generation manifest fields are not exact")
        if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValidationError("unsupported generation manifest schema")
        if (
            type(manifest["api_version"]) is not int
            or manifest["api_version"] != API_VERSION
        ):
            raise ValidationError("unsupported generation API version")
        if manifest["physical_generation_id_policy"] != PHYSICAL_ID_POLICY:
            raise ValidationError("physical generation ID policy mismatch")
        body = dict(manifest)
        manifest_root = body.pop("manifest_root_sha256")
        if (
            not isinstance(manifest_root, str)
            or manifest_root != hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        ):
            raise ValidationError("generation manifest root is invalid")

        generation_id = manifest["generation_id"]
        try:
            _validate_generation_id(generation_id)
        except InventoryError as error:
            raise ValidationError(f"invalid manifest generation ID: {error}") from error
        if (
            expected_generation_id is not None
            and generation_id != expected_generation_id
        ):
            raise ValidationError("manifest generation ID does not match directory")
        source_generation_id = manifest["source_generation_id"]
        requested_siblings = manifest["requested_siblings"]
        if not isinstance(source_generation_id, str) or not source_generation_id:
            raise ValidationError("manifest source generation ID is missing")
        if not isinstance(requested_siblings, list):
            raise ValidationError("manifest requested siblings must be a list")
        if not isinstance(manifest["outputs"], list):
            raise ValidationError("manifest outputs must be a list")
        if not isinstance(manifest["directories"], list) or not all(
            isinstance(directory, str) for directory in manifest["directories"]
        ):
            raise ValidationError("manifest directories must be a JSON list of strings")
        if len(manifest["directories"]) != len(set(manifest["directories"])):
            raise ValidationError("manifest directories contain duplicates")

        reconstructed_specs = []
        metadata_by_path = {}
        for item in manifest["outputs"]:
            if not isinstance(item, dict) or set(item) != _OUTPUT_METADATA_KEYS:
                raise ValidationError("output metadata fields are not exact")
            path = item.get("path")
            if path in metadata_by_path:
                raise ValidationError(f"duplicate manifest output path: {path}")
            if not isinstance(item["drop_types"], list) or not all(
                isinstance(drop_type, str) for drop_type in item["drop_types"]
            ):
                raise ValidationError(
                    f"{path}: drop_types must be a JSON list of strings"
                )
            validator = _validator_from_descriptor(
                item["validator"],
                binary_validators=self._binary_validators,
            )
            try:
                spec = OutputSpec(
                    path=path,
                    role=OutputRole(item["role"]),
                    schema=item["schema"],
                    validator=validator,
                    sibling=item["sibling"],
                    drop_types=tuple(item["drop_types"]),
                )
            except (InventoryError, TypeError, ValueError) as error:
                raise ValidationError(
                    f"invalid output metadata for {path!r}: {error}"
                ) from error
            if item["generation_id"] != generation_id:
                raise ValidationError(f"cross-file generation ID mismatch for {path}")
            if item["source_generation_id"] != source_generation_id:
                raise ValidationError(
                    f"cross-file source generation ID mismatch for {path}"
                )
            for digest_name in ("sha256", "logical_sha256"):
                if not isinstance(item[digest_name], str) or not _SHA256_RE.fullmatch(
                    item[digest_name]
                ):
                    raise ValidationError(f"{path}: malformed {digest_name}")
            for integer_name in ("bytes", "rows"):
                value = item[integer_name]
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValidationError(f"{path}: invalid {integer_name}")
            reconstructed_specs.append(spec)
            metadata_by_path[path] = item
        try:
            reconstructed = GenerationPlan(
                generation_id=generation_id,
                source_generation_id=source_generation_id,
                requested_siblings=tuple(requested_siblings),
                outputs=tuple(reconstructed_specs),
            )
        except InventoryError as error:
            raise ValidationError(f"manifest inventory is invalid: {error}") from error
        if reconstructed.plan_root_sha256 != manifest["plan_root_sha256"]:
            raise ValidationError("manifest plan root is invalid")
        validation_plan = reconstructed
        if expected_plan is not None:
            if (
                expected_plan.generation_id != generation_id
                or expected_plan.source_generation_id != source_generation_id
                or expected_plan.requested_siblings != tuple(requested_siblings)
                or expected_plan.plan_root_sha256 != manifest["plan_root_sha256"]
            ):
                raise ValidationError(
                    "manifest plan identity does not match declaration"
                )
            expected_specs = {spec.path: spec for spec in expected_plan.outputs}
            if set(expected_specs) != set(metadata_by_path):
                raise ValidationError(
                    "manifest output inventory does not match declaration"
                )
            for path, spec in expected_specs.items():
                if _spec_descriptor(spec) != {
                    key: metadata_by_path[path][key]
                    for key in (
                        "drop_types",
                        "path",
                        "role",
                        "schema",
                        "sibling",
                        "validator",
                    )
                }:
                    raise ValidationError(
                        f"{path}: manifest schema declaration changed"
                    )
            validation_plan = expected_plan

        routes = manifest["routes"]
        if not isinstance(routes, dict) or set(routes) != _ROUTES_METADATA_KEYS:
            raise ValidationError("routes metadata fields are not exact")
        if (
            routes["path"] != ROUTES_FILENAME
            or routes["schema_version"] != ROUTES_SCHEMA_VERSION
        ):
            raise ValidationError("routes metadata schema/path is invalid")
        for field_name in ("bytes", "rows"):
            value = routes[field_name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(
                    f"routes {field_name} must be a non-negative integer"
                )
        for field_name in ("sha256", "root_sha256"):
            value = routes[field_name]
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValidationError(f"routes {field_name} is malformed")
        declared = set(metadata_by_path)
        expected_files = declared | {ROUTES_FILENAME, MANIFEST_FILENAME}
        expected_directories = _expected_directory_inventory(expected_files)
        if manifest["directories"] != sorted(expected_directories):
            raise ValidationError(
                "manifest directories do not match the canonical file inventory"
            )
        actual_files, actual_directories = _physical_inventory(
            generation,
            error_type=ValidationError,
        )
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        if missing:
            raise ValidationError(f"missing generation files: {', '.join(missing)}")
        if extra:
            raise ValidationError(f"extra generation files: {', '.join(extra)}")
        missing_directories = sorted(expected_directories - actual_directories)
        extra_directories = sorted(actual_directories - expected_directories)
        if missing_directories:
            raise ValidationError(
                "missing declared directories: " + ", ".join(missing_directories)
            )
        if extra_directories:
            raise ValidationError(
                "undeclared directories: " + ", ".join(extra_directories)
            )
        if require_sealed:
            _validate_read_only_tree(generation)

        context = ValidationContext(
            generation_id=generation_id,
            source_generation_id=source_generation_id,
            plan_root_sha256=validation_plan.plan_root_sha256,
        )
        specs = {spec.path: spec for spec in validation_plan.outputs}
        for path in sorted(metadata_by_path):
            item = metadata_by_path[path]
            output = generation.joinpath(*PurePosixPath(path).parts)
            _require_regular_file(output, path, ValidationError)
            actual_sha = _sha256_file(output)
            if actual_sha != item["sha256"]:
                raise ValidationError(
                    f"{path}: SHA-256 mismatch; expected {item['sha256']}, "
                    f"got {actual_sha}"
                )
            if output.stat().st_size != item["bytes"]:
                raise ValidationError(f"{path}: byte-size mismatch")
            rows = _validate_output(output, specs[path], context)
            if rows != item["rows"]:
                raise ValidationError(f"{path}: row-count mismatch")
            logical_sha = specs[path].validator.logical_sha256(output, context)
            if logical_sha != item["logical_sha256"]:
                raise ValidationError(f"{path}: logical SHA-256 mismatch")

        routes_path = generation / ROUTES_FILENAME
        _require_regular_file(routes_path, ROUTES_FILENAME, ValidationError)
        if (
            routes_path.stat().st_size != routes["bytes"]
            or _sha256_file(routes_path) != routes["sha256"]
            or routes["root_sha256"] != routes["sha256"]
            or len(_read_jsonl_objects(routes_path)) != routes["rows"]
        ):
            raise ValidationError("routes metadata does not match physical routes")
        accounting = _validate_accounting(generation, validation_plan)
        if accounting != manifest["accounting"]:
            raise ValidationError("manifest accounting summary is stale")
        logical_root = _logical_root(validation_plan, manifest["outputs"], routes)
        if logical_root != manifest["logical_root_sha256"]:
            raise ValidationError("logical generation root mismatch")
        return manifest


def _spec_descriptor(spec: OutputSpec) -> dict[str, Any]:
    return {
        "drop_types": list(spec.drop_types),
        "path": spec.path,
        "role": spec.role.value,
        "schema": spec.schema,
        "sibling": spec.sibling,
        "validator": spec.validator.descriptor,
    }


def _validator_from_descriptor(
    descriptor: Any,
    *,
    binary_validators: Mapping[tuple[str, str], BinaryValidator],
) -> OutputValidator:
    if not isinstance(descriptor, dict):
        raise ValidationError("validator descriptor must be an object")
    kind = descriptor.get("kind")
    try:
        if kind == "jsonl-object":
            expected = {
                "allow_empty",
                "kind",
                "require_generation_links",
                "required_fields",
                "schema_version",
                "validator_version",
            }
            if set(descriptor) != expected:
                raise ValidationError("JSONL validator descriptor is not exact")
            if (
                type(descriptor["allow_empty"]) is not bool
                or type(descriptor["require_generation_links"]) is not bool
            ):
                raise ValidationError("validator boolean fields are not booleans")
            if (
                type(descriptor["validator_version"]) is not int
                or descriptor["validator_version"] != 1
                or not isinstance(descriptor["required_fields"], list)
                or not all(
                    isinstance(name, str) for name in descriptor["required_fields"]
                )
            ):
                raise ValidationError("JSONL validator fields are not strictly typed")
            return JsonlValidator(
                schema_version=descriptor["schema_version"],
                required_fields=tuple(descriptor["required_fields"]),
                allow_empty=descriptor["allow_empty"],
                require_generation_links=descriptor["require_generation_links"],
                validator_version=descriptor["validator_version"],
            )
        if kind == "json-object":
            expected = {
                "kind",
                "require_generation_links",
                "required_fields",
                "schema_version",
                "validator_version",
            }
            if set(descriptor) != expected:
                raise ValidationError("JSON validator descriptor is not exact")
            if type(descriptor["require_generation_links"]) is not bool:
                raise ValidationError("validator boolean fields are not booleans")
            if (
                type(descriptor["validator_version"]) is not int
                or descriptor["validator_version"] != 1
                or not isinstance(descriptor["required_fields"], list)
                or not all(
                    isinstance(name, str) for name in descriptor["required_fields"]
                )
            ):
                raise ValidationError("JSON validator fields are not strictly typed")
            return JsonObjectValidator(
                schema_version=descriptor["schema_version"],
                required_fields=tuple(descriptor["required_fields"]),
                require_generation_links=descriptor["require_generation_links"],
                validator_version=descriptor["validator_version"],
            )
        if kind == "binary":
            expected = {
                "kind",
                "schema_version",
                "validator_id",
                "validator_version",
            }
            if (
                set(descriptor) != expected
                or type(descriptor["validator_version"]) is not int
                or descriptor["validator_version"] != 1
                or not isinstance(descriptor["schema_version"], str)
                or not isinstance(descriptor["validator_id"], str)
            ):
                raise ValidationError("binary validator descriptor is not exact")
            key = (descriptor["schema_version"], descriptor["validator_id"])
            try:
                return binary_validators[key]
            except KeyError as error:
                raise ValidationError(
                    f"binary validator is not registered: {key}"
                ) from error
    except (InventoryError, TypeError, ValueError) as error:
        raise ValidationError(f"invalid validator descriptor: {error}") from error
    raise ValidationError(f"unknown validator kind: {kind!r}")


def _validate_structured_record(
    record: Mapping[str, Any],
    *,
    path: Path,
    row: int | None,
    schema_version: str,
    required_fields: Sequence[str],
    require_generation_links: bool,
    context: ValidationContext,
) -> None:
    location = f"{path}:{row}" if row is not None else str(path)
    if record.get("schema_version") != schema_version:
        raise ValidationError(f"{location}: schema version mismatch")
    missing = sorted(set(required_fields) - set(record))
    if missing:
        raise ValidationError(f"{location}: missing required fields {missing}")
    if require_generation_links:
        if record.get("generation_id") != context.generation_id:
            raise ValidationError(f"{location}: stale generation link")
        if record.get("source_generation_id") != context.source_generation_id:
            raise ValidationError(f"{location}: stale source generation link")
        if record.get("plan_root_sha256") != context.plan_root_sha256:
            raise ValidationError(f"{location}: stale root link")


def _normalize_physical_links(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "generation_id" in normalized:
        normalized["generation_id"] = "<physical-generation-id>"
    return normalized


def _validate_output(
    path: Path,
    spec: OutputSpec,
    context: ValidationContext,
) -> int:
    try:
        rows = spec.validator.validate(path, context)
    except GenerationError:
        raise
    except BaseException as error:
        raise ValidationError(
            f"{spec.path}: validator failed: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
        raise ValidationError(
            f"{spec.path}: validator returned an empty or invalid row count"
        )
    return rows


def _raw_occurrences(path: Path, spec: OutputSpec) -> list[RawOccurrence]:
    assert spec.sibling is not None
    occurrences = []
    for row, raw_bytes in enumerate(_read_binary_lines(path), start=1):
        digest = hashlib.sha256(raw_bytes).hexdigest()
        occurrence_id = f"occurrence/v1:{spec.sibling}:{spec.path}:{row}:{digest}"
        occurrences.append(
            RawOccurrence(
                occurrence_id=occurrence_id,
                sibling=spec.sibling,
                logical_source=spec.path,
                raw_row=row,
                raw_sha256=digest,
                raw_bytes=raw_bytes,
            )
        )
    return occurrences


def _output_metadata(
    generation: Path,
    spec: OutputSpec,
    *,
    context: ValidationContext,
) -> dict[str, Any]:
    path = generation.joinpath(*PurePosixPath(spec.path).parts)
    _require_regular_file(path, spec.path, ValidationError)
    rows = _validate_output(path, spec, context)
    return {
        "bytes": path.stat().st_size,
        "drop_types": list(spec.drop_types),
        "generation_id": context.generation_id,
        "logical_sha256": spec.validator.logical_sha256(path, context),
        "path": spec.path,
        "role": spec.role.value,
        "rows": rows,
        "schema": spec.schema,
        "sha256": _sha256_file(path),
        "sibling": spec.sibling,
        "source_generation_id": context.source_generation_id,
        "validator": spec.validator.descriptor,
    }


def _logical_root(
    plan: GenerationPlan,
    metadata: Sequence[Mapping[str, Any]],
    routes: Mapping[str, Any],
) -> str:
    outputs = [
        {
            "drop_types": item["drop_types"],
            "logical_sha256": item["logical_sha256"],
            "path": item["path"],
            "role": item["role"],
            "rows": item["rows"],
            "schema": item["schema"],
            "sibling": item["sibling"],
            "validator": item["validator"],
        }
        for item in sorted(metadata, key=lambda value: value["path"])
    ]
    body = {
        "outputs": outputs,
        "plan_root_sha256": plan.plan_root_sha256,
        "requested_siblings": list(plan.requested_siblings),
        "routes_root_sha256": routes["root_sha256"],
        "schema_version": LOGICAL_ROOT_SCHEMA_VERSION,
        "source_generation_id": plan.source_generation_id,
    }
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _validate_accounting(
    generation: Path,
    plan: GenerationPlan,
) -> dict[str, Any]:
    specs = {spec.path: spec for spec in plan.outputs}
    context = ValidationContext(
        generation_id=plan.generation_id,
        source_generation_id=plan.source_generation_id,
        plan_root_sha256=plan.plan_root_sha256,
    )
    occurrences: dict[str, RawOccurrence] = {}
    occurrences_by_sibling: dict[str, set[str]] = {}
    for sibling in plan.requested_siblings:
        raw_spec = next(
            spec
            for spec in plan.outputs
            if spec.sibling == sibling and spec.role is OutputRole.RAW
        )
        raw_path = generation.joinpath(*PurePosixPath(raw_spec.path).parts)
        sibling_occurrences = _raw_occurrences(raw_path, raw_spec)
        occurrences_by_sibling[sibling] = {
            occurrence.occurrence_id for occurrence in sibling_occurrences
        }
        for occurrence in sibling_occurrences:
            if occurrence.occurrence_id in occurrences:
                raise AccountingError(
                    f"duplicate occurrence ID: {occurrence.occurrence_id}"
                )
            occurrences[occurrence.occurrence_id] = occurrence

    routes_path = generation / ROUTES_FILENAME
    route_records = _read_jsonl_objects(routes_path)
    assigned: dict[str, Mapping[str, Any]] = {}
    destination_seen: set[tuple[str, int]] = set()
    destination_lines = {
        spec.path: _read_binary_lines(
            generation.joinpath(*PurePosixPath(spec.path).parts)
        )
        for spec in plan.outputs
        if spec.role in {OutputRole.TRAIN, OutputRole.EVAL}
        or (spec.role is OutputRole.SIDECAR and spec.drop_types)
    }
    summaries = {
        sibling: {
            "drop_rows": 0,
            "drop_types": Counter(),
            "eval_rows": 0,
            "raw_rows": len(occurrences_by_sibling[sibling]),
            "train_rows": 0,
        }
        for sibling in plan.requested_siblings
    }

    for route_number, route in enumerate(route_records, start=1):
        if set(route) != _ROUTE_KEYS:
            raise AccountingError(f"route {route_number}: fields are not exact")
        occurrence_id = route["occurrence_id"]
        if occurrence_id in assigned:
            raise AccountingError(
                f"occurrence assigned more than once: {occurrence_id}"
            )
        try:
            occurrence = occurrences[occurrence_id]
        except KeyError as error:
            raise AccountingError(
                f"route references unknown occurrence: {occurrence_id}"
            ) from error
        if (
            route["sibling"] != occurrence.sibling
            or route["raw_path"] != occurrence.logical_source
            or route["raw_row"] != occurrence.raw_row
            or route["raw_sha256"] != occurrence.raw_sha256
        ):
            raise AccountingError(
                f"route physical occurrence fields mismatch: {occurrence_id}"
            )
        if route["plan_root_sha256"] != plan.plan_root_sha256:
            raise AccountingError(f"route root link mismatch: {occurrence_id}")
        destination_path = route["destination_path"]
        try:
            destination_spec = specs[destination_path]
        except KeyError as error:
            raise AccountingError(
                f"route destination is not inventoried: {destination_path}"
            ) from error
        if destination_spec.sibling != occurrence.sibling:
            raise AccountingError(
                f"cross-sibling occurrence route: "
                f"{occurrence.sibling} -> {destination_spec.sibling}"
            )
        disposition = route["disposition"]
        expected_role = {
            "train": OutputRole.TRAIN,
            "eval": OutputRole.EVAL,
            "drop": OutputRole.SIDECAR,
        }.get(disposition)
        if expected_role is None or destination_spec.role is not expected_role:
            raise AccountingError(
                f"route disposition/destination mismatch: {occurrence_id}"
            )
        destination_row = route["destination_row"]
        lines = destination_lines[destination_path]
        if (
            not isinstance(destination_row, int)
            or isinstance(destination_row, bool)
            or not 1 <= destination_row <= len(lines)
        ):
            raise AccountingError(f"route destination row is invalid: {occurrence_id}")
        destination_key = (destination_path, destination_row)
        if destination_key in destination_seen:
            raise AccountingError(
                f"destination row has multiple routes: {destination_path}:{destination_row}"
            )
        destination_seen.add(destination_key)
        if disposition in {"train", "eval"}:
            if lines[destination_row - 1] != occurrence.raw_bytes:
                raise AccountingError(
                    f"routed native bytes changed: {destination_path}:{destination_row}"
                )
            if route["drop_type"] is not None:
                raise AccountingError("non-drop route has a drop type")
            summaries[occurrence.sibling][f"{disposition}_rows"] += 1
        else:
            try:
                drop = json.loads(lines[destination_row - 1])
            except (UnicodeError, json.JSONDecodeError) as error:
                raise AccountingError("drop route is not valid JSON") from error
            if not isinstance(drop, dict) or set(drop) != _DROP_RECORD_KEYS:
                raise AccountingError("drop sidecar fields are not exact")
            if (
                drop["occurrence_id"] != occurrence_id
                or drop["sibling"] != occurrence.sibling
                or drop["raw_path"] != occurrence.logical_source
                or drop["raw_row"] != occurrence.raw_row
                or drop["raw_sha256"] != occurrence.raw_sha256
            ):
                raise AccountingError("drop route does not match physical occurrence")
            _validate_structured_record(
                drop,
                path=generation / destination_path,
                row=destination_row,
                schema_version=destination_spec.schema,
                required_fields=(
                    "details",
                    "drop_type",
                    "occurrence_id",
                    "raw_path",
                    "raw_row",
                    "raw_sha256",
                    "sibling",
                ),
                require_generation_links=True,
                context=context,
            )
            drop_type = route["drop_type"]
            if (
                drop_type != drop["drop_type"]
                or drop_type not in destination_spec.drop_types
            ):
                raise AccountingError("drop type is not declared or route-linked")
            summaries[occurrence.sibling]["drop_rows"] += 1
            summaries[occurrence.sibling]["drop_types"][drop_type] += 1
        assigned[occurrence_id] = route

    unassigned = sorted(set(occurrences) - set(assigned))
    if unassigned:
        raise AccountingError(
            f"unassigned raw occurrences: {len(unassigned)}; first {unassigned[0]}"
        )
    for path, lines in destination_lines.items():
        routed_rows = sum(route["destination_path"] == path for route in route_records)
        if routed_rows != len(lines):
            raise AccountingError(f"{path}: output rows and occurrence routes differ")
    normalized = {}
    for sibling, summary in summaries.items():
        normalized[sibling] = {
            **summary,
            "drop_types": dict(sorted(summary["drop_types"].items())),
        }
    return {"scheme": ACCOUNTING_SCHEME, "siblings": normalized}


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records = []
    for row, line in enumerate(_read_binary_lines(path), start=1):
        try:
            record = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValidationError(f"{path}:{row}: invalid JSONL: {error}") from error
        if not isinstance(record, dict):
            raise ValidationError(f"{path}:{row}: JSONL row must be an object")
        records.append(record)
    return records


def _read_binary_lines(path: Path) -> list[bytes]:
    lines = []
    with path.open("rb") as handle:
        for row, line in enumerate(handle, start=1):
            if not line.endswith(b"\n"):
                raise ValidationError(
                    f"{path}:{row}: JSONL row is not newline terminated"
                )
            lines.append(line)
    return lines


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return payload


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _validate_transaction_record(
    record: Mapping[str, Any],
    *,
    expected_state: str,
) -> None:
    if set(record) != _TRANSACTION_STATE_KEYS:
        raise ValidationError("transaction state fields are not exact")
    if record.get("schema_version") != TRANSACTION_STATE_SCHEMA_VERSION:
        raise ValidationError("transaction state schema is invalid")
    if record.get("state") != expected_state:
        raise ValidationError("transaction state value is invalid")
    try:
        _validate_generation_id(record.get("generation_id"))
    except InventoryError as error:
        raise ValidationError("transaction generation ID is invalid") from error
    for field_name in ("manifest_sha256", "logical_root_sha256"):
        value = record.get(field_name)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValidationError(f"transaction {field_name} is malformed")


def _validate_transaction_matches_pointer(
    record: Mapping[str, Any],
    pointer: Mapping[str, Any],
) -> None:
    for field_name in (
        "generation_id",
        "manifest_sha256",
        "logical_root_sha256",
    ):
        if record.get(field_name) != pointer.get(field_name):
            raise ValidationError(
                f"transaction state does not match CURRENT {field_name}"
            )


def _require_real_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValidationError(f"{label} is missing") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValidationError(f"{label} is a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValidationError(f"{label} is not a directory")


def _require_secure_filesystem_capabilities() -> None:
    missing = []
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or nofollow <= 0:
        missing.append("O_NOFOLLOW")
    if type(directory) is not int or directory <= 0:
        missing.append("O_DIRECTORY")
    supports_dir_fd = getattr(os, "supports_dir_fd", frozenset())
    for name, function in _REQUIRED_DIR_FD_FUNCTIONS.items():
        current = getattr(os, name, None)
        supports_keywords = False
        if callable(current):
            try:
                parameters = inspect.signature(current).parameters
            except (TypeError, ValueError):
                parameters = {}
            supports_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ) or all(
                keyword in parameters for keyword in _REQUIRED_DIR_FD_KEYWORDS[name]
            )
        if function not in supports_dir_fd or not supports_keywords:
            missing.append(f"dir_fd:{name}")
    for name in ("close", "dup", "fdopen", "fstat", "fsync"):
        if not callable(getattr(os, name, None)):
            missing.append(f"fd:{name}")
    if missing:
        raise PlatformCapabilityError(
            "secure filesystem capabilities unavailable: " + ", ".join(missing)
        )


def _mkdir_durable(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY
    flags |= os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(absolute.anchor, flags)
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            child_path = current / part
            created = False
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, flags, dir_fd=descriptor)
                os.fsync(child)
                created = True
            except OSError as error:
                _raise_component_error(
                    error,
                    f"directory component {child_path}",
                    GenerationError,
                )
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise GenerationError(
                    f"directory component is not a real directory: {child_path}"
                )
            os.close(descriptor)
            descriptor = child
            current = child_path
            if created:
                # The fd fsyncs above are authoritative; these path fsyncs also
                # preserve the existing observability contract for durability tests.
                _fsync_directory(current.parent)
                _fsync_directory(current)
    finally:
        os.close(descriptor)


def _directory_open_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _raise_component_error(
    error: OSError,
    label: str,
    error_type: type[GenerationError],
) -> None:
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        raise error_type(
            f"symlink or non-directory component is forbidden: {label}"
        ) from error
    raise error_type(f"unsafe filesystem component {label}: {error}") from error


def _open_directory_nofollow(
    path: Path,
    *,
    error_type: type[GenerationError],
) -> int:
    absolute = Path(os.path.abspath(path))
    flags = _directory_open_flags()
    descriptor = os.open(absolute.anchor, flags)
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current /= part
            try:
                child = os.open(part, flags, dir_fd=descriptor)
            except OSError as error:
                _raise_component_error(error, str(current), error_type)
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise error_type(f"not a real directory: {current}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _safe_create_child_directory(
    parent: Path,
    name: str,
    *,
    mode: int,
    error_type: type[GenerationError],
) -> Path:
    if not name or "/" in name or name in {".", ".."}:
        raise error_type(f"invalid directory child name: {name!r}")
    parent_descriptor = _open_directory_nofollow(parent, error_type=error_type)
    try:
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_descriptor)
        except OSError as error:
            _raise_component_error(error, str(parent / name), error_type)
        os.fsync(parent_descriptor)
        child = os.open(name, _directory_open_flags(), dir_fd=parent_descriptor)
        try:
            os.fsync(child)
        finally:
            os.close(child)
    finally:
        os.close(parent_descriptor)
    return parent / name


def _safe_rename_child(
    source_parent: Path,
    source_name: str,
    destination_parent: Path,
    destination_name: str,
    *,
    error_type: type[GenerationError],
) -> None:
    source = _open_directory_nofollow(source_parent, error_type=error_type)
    destination = _open_directory_nofollow(
        destination_parent,
        error_type=error_type,
    )
    try:
        os.rename(
            source_name,
            destination_name,
            src_dir_fd=source,
            dst_dir_fd=destination,
        )
        os.fsync(source)
        os.fsync(destination)
    except OSError as error:
        _raise_component_error(
            error,
            f"{source_parent / source_name} -> {destination_parent / destination_name}",
            error_type,
        )
    finally:
        os.close(source)
        os.close(destination)


def _safe_unlink_child(
    parent: Path,
    name: str,
    *,
    error_type: type[GenerationError],
) -> None:
    descriptor = _open_directory_nofollow(parent, error_type=error_type)
    try:
        try:
            os.unlink(name, dir_fd=descriptor)
        except OSError as error:
            _raise_component_error(error, str(parent / name), error_type)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_relative_parent(
    root_descriptor: int,
    logical_path: str,
    *,
    create: bool,
    error_type: type[GenerationError],
) -> tuple[int, str]:
    parts = PurePosixPath(logical_path).parts
    if not parts:
        raise error_type("relative output path is empty")
    descriptor = os.dup(root_descriptor)
    traversed = []
    try:
        for part in parts[:-1]:
            traversed.append(part)
            try:
                child = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise error_type(f"missing output directory: {'/'.join(traversed)}")
                os.mkdir(part, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    part,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
                os.fsync(child)
            except OSError as error:
                _raise_component_error(
                    error,
                    "/".join(traversed),
                    error_type,
                )
            metadata = os.fstat(child)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise error_type(
                    f"output component is not a directory: {'/'.join(traversed)}"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor, parts[-1]
    except BaseException:
        os.close(descriptor)
        raise


def _safe_output_identity(
    root_descriptor: int,
    logical_path: str,
    *,
    error_type: type[GenerationError],
) -> tuple[int, int, int, int]:
    parent, leaf = _open_relative_parent(
        root_descriptor,
        logical_path,
        create=False,
        error_type=error_type,
    )
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(leaf, flags, dir_fd=parent)
        except OSError as error:
            _raise_component_error(error, logical_path, error_type)
        try:
            metadata = os.fstat(descriptor)
            _validate_regular_metadata(metadata, logical_path, error_type)
            return (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _open_exclusive_relative_output(
    root_descriptor: int,
    logical_path: str,
    *,
    error_type: type[GenerationError],
) -> tuple[int, str, int]:
    parent, leaf = _open_relative_parent(
        root_descriptor,
        logical_path,
        create=True,
        error_type=error_type,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(
                leaf,
                flags,
                0o600,
                dir_fd=parent,
            )
        except FileExistsError as error:
            try:
                metadata = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
            except OSError:
                metadata = None
            if metadata is not None and stat.S_ISLNK(metadata.st_mode):
                raise error_type(f"output leaf is a symlink: {logical_path}") from error
            raise error_type(f"duplicate physical output: {logical_path}") from error
        except OSError as error:
            _raise_component_error(error, logical_path, error_type)
        _validate_regular_metadata(
            os.fstat(descriptor),
            logical_path,
            error_type,
        )
        return parent, leaf, descriptor
    except BaseException:
        os.close(parent)
        raise


def _unlink_opened_output(
    parent_descriptor: int,
    leaf: str,
    opened: os.stat_result,
    *,
    error_type: type[GenerationError],
) -> None:
    try:
        current = os.stat(
            leaf,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError as error:
        _raise_component_error(error, leaf, error_type)
    if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
        os.unlink(leaf, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)


def _read_relative_binary_lines(
    root_descriptor: int,
    logical_path: str,
    *,
    error_type: type[GenerationError],
) -> list[bytes]:
    parent, leaf = _open_relative_parent(
        root_descriptor,
        logical_path,
        create=False,
        error_type=error_type,
    )
    try:
        try:
            descriptor = os.open(
                leaf,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            _raise_component_error(error, logical_path, error_type)
        try:
            _validate_regular_metadata(
                os.fstat(descriptor),
                logical_path,
                error_type,
            )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return list(handle)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _open_regular_file_nofollow(
    path: Path,
    *,
    error_type: type[GenerationError],
) -> int:
    parent = _open_directory_nofollow(path.parent, error_type=error_type)
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=parent,
            )
        except OSError as error:
            _raise_component_error(error, str(path), error_type)
        try:
            _validate_regular_metadata(
                os.fstat(descriptor),
                str(path),
                error_type,
            )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        os.close(parent)


def _write_relative_chunks(
    root_descriptor: int,
    logical_path: str,
    chunks: Iterable[bytes],
    *,
    error_type: type[GenerationError],
) -> None:
    parent, _, descriptor = _open_exclusive_relative_output(
        root_descriptor,
        logical_path,
        error_type=error_type,
    )
    try:
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                for chunk in chunks:
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise TypeError(
                            f"output chunks must be bytes for {logical_path}"
                        )
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_durable_file(path: Path, payload: bytes, *, exclusive: bool) -> None:
    parent = _open_directory_nofollow(
        path.parent,
        error_type=ValidationError,
    )
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_TRUNC
    flags |= os.O_NOFOLLOW
    try:
        try:
            descriptor = os.open(path.name, flags, 0o600, dir_fd=parent)
        except OSError as error:
            _raise_component_error(error, str(path), ValidationError)
        try:
            metadata = os.fstat(descriptor)
            _validate_regular_metadata(metadata, str(path), ValidationError)
            with os.fdopen(descriptor, "wb", closefd=False) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(parent)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _write_atomic_control_file(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise GenerationExistsError(f"control state already exists: {path.name}")
    temporary = path.parent / f".{path.name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        raise ValidationError(f"stale control temporary exists: {temporary.name}")
    _write_durable_file(temporary, payload, exclusive=True)
    _safe_rename_child(
        path.parent,
        temporary.name,
        path.parent,
        path.name,
        error_type=ValidationError,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_nofollow(
        path,
        error_type=ValidationError,
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories.append(current_path)
        for name in directory_names:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValidationError(f"symlink directory in generation: {candidate}")
        for name in file_names:
            candidate = current_path / name
            _require_regular_file(
                candidate,
                str(candidate.relative_to(root)),
                ValidationError,
            )
            descriptor = os.open(candidate, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in reversed(directories):
        _fsync_directory(directory)


def _seal_tree(root: Path) -> None:
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            _require_regular_file(path, str(path.relative_to(root)), ValidationError)
            path.chmod(0o444)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                raise ValidationError(f"cannot seal symlink directory: {path}")
    directories = [
        Path(current)
        for current, _, _ in os.walk(root, topdown=False, followlinks=False)
    ]
    for directory in directories:
        directory.chmod(0o555)


def _make_tree_writable(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        if not path.is_symlink():
            path.chmod(0o600)
        return
    path.chmod(0o755)
    for current, directory_names, file_names in os.walk(path, followlinks=False):
        current_path = Path(current)
        current_path.chmod(0o755)
        for name in directory_names:
            candidate = current_path / name
            if not candidate.is_symlink():
                candidate.chmod(0o755)
        for name in file_names:
            candidate = current_path / name
            if not candidate.is_symlink():
                candidate.chmod(0o644)


def _validate_read_only_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValidationError(f"sealed generation contains symlink: {path}")
        if metadata.st_mode & 0o222:
            raise ValidationError(f"sealed generation path is writable: {path}")


def _require_regular_file(
    path: Path,
    label: str,
    error_type: type[GenerationError],
) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise error_type(f"missing file: {label}") from error
    _validate_regular_metadata(metadata, label, error_type)


def _validate_regular_metadata(
    metadata: os.stat_result,
    label: str,
    error_type: type[GenerationError],
) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise error_type(f"symlink is forbidden: {label}")
    if not stat.S_ISREG(metadata.st_mode):
        raise error_type(f"special node is forbidden: {label}")
    if metadata.st_nlink != 1:
        raise error_type(f"hard link is forbidden: {label}")


def _expected_directory_inventory(files: Iterable[str]) -> set[str]:
    directories = set()
    for logical_path in files:
        parent = PurePosixPath(logical_path).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _physical_inventory(
    root: Path,
    *,
    error_type: type[GenerationError],
) -> tuple[set[str], set[str]]:
    files = set()
    directories = set()
    stack = [(root, PurePosixPath("."))]
    while stack:
        current_path, current_logical = stack.pop()
        try:
            entries = list(os.scandir(current_path))
        except OSError as error:
            raise error_type(
                f"cannot scan generation inventory at {current_logical}: {error}"
            ) from error
        for entry in entries:
            logical = (
                PurePosixPath(entry.name)
                if current_logical == PurePosixPath(".")
                else current_logical / entry.name
            )
            relative = logical.as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise error_type(f"symlink is forbidden in inventory: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.add(relative)
                stack.append((Path(entry.path), logical))
            elif stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise error_type(f"hard link is forbidden in inventory: {relative}")
                files.add(relative)
            else:
                raise error_type(f"special node is forbidden in inventory: {relative}")
    return files, directories


def _safe_quarantine_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return normalized or "unknown"


def _inject(
    fault_injector: FaultInjector | None,
    phase: PublishPhase,
    path: Path,
) -> None:
    if fault_injector is not None:
        fault_injector(phase, path)
