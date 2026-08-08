"""Build and verify the immutable, transaction-bound P3 evaluator release."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import split_mml_semantic_holdout as mml_holdout
from corpus_generation_transaction import (
    API_VERSION as CORPUS_TRANSACTION_API_VERSION,
)
from corpus_generation_transaction import (
    CURRENT_SCHEMA_VERSION as CORPUS_CURRENT_SCHEMA,
)
from corpus_generation_transaction import (
    MANIFEST_FILENAME as CORPUS_MANIFEST_NAME,
)
from corpus_generation_transaction import (
    MANIFEST_SCHEMA_VERSION as CORPUS_MANIFEST_SCHEMA,
)
from corpus_generation_transaction import (
    BinaryValidator,
    DropRecord,
    GenerationCoordinator,
    GenerationError,
    GenerationPlan,
    JsonlValidator,
    JsonObjectValidator,
    OutputRole,
    OutputSpec,
    PublishPhase,
)

PROFILE_NAME = "p3-evaluator-release-v1"
PLATFORM_PROFILE_REGISTERED = False
APPROVED_DATASET_ID = "eval/formal-proof-premises-500m"
UNPUBLISHED_VERSION = "__UNPUBLISHED__"
UNPUBLISHED_PLATFORM_MANIFEST_SHA256 = "__PUBLISHER_GROUP_MANIFEST_SHA256__"

MANIFEST_SCHEMA = "p3-evaluator-release-manifest/v2"
COMPLETION_SCHEMA = "p3-evaluator-release-completion/v2"
TRAIN_VISIBILITY_SCHEMA = "p3-train-visibility/v2"
CORPUS_UNION_VISIBILITY_SCHEMA = "p3-corpus-union-visibility/v2"
DROP_LEDGER_SCHEMA = "p3-corpus-drop/v2"
COHORT_LEDGER_SCHEMA = "p3-eval-cohort-ledger/v1"
TOKENIZER_SEAL_SCHEMA = "p3-tokenizer-seal/v1"

MANIFEST_NAME = "evaluator/release-manifest.json"
COMPLETION_SEAL_NAME = "evaluator/completion.json"
TRANSACTION_PAYLOAD_SCHEMA = "p3-evaluator-transaction-payload/v1"
TRANSACTION_PAYLOAD_VALIDATOR_ID = "p3-evaluator-transaction-payload/v1"

FAMILIES = ("enigma", "isabelle", "metamath", "mizar", "prf2", "thproofs")
MML_NATIVE_FAMILIES = ("mizar", "thproofs", "prf2", "enigma")
MML_PROJECTIONS = ("mizar", "atp")
MML_PROJECTION_SHARDS = {
    "atp": ("prf2", "enigma"),
    "mizar": ("mizar", "thproofs"),
}
MML_MAPPED_CLASS_RE = re.compile(
    r"^mml:v1:(theorem|definition):([A-Z][A-Z0-9_]*):([1-9]\d*)$"
)
MML_SOURCE_IDENTITY_POLICY_SCHEMA = "mml-source-identity-policy-v1"
HELDOUT_NAMES = ("atp", "isabelle", "metamath", "mizar", "mml")
SEMANTIC_SIDECARS = ("drop_reasons", "eval_exposure")
SOURCE_MANIFEST_NAMES = FAMILIES

FAMILY_ROW_SCHEMAS = {
    "enigma": "p3-atp-proof-row/v2",
    "isabelle": "p3-isabelle-transition/v2",
    "metamath": "p3-metamath-proof-row/v1",
    "mizar": "p3-mizar-proof-row/v1",
    "prf2": "p3-atp-proof-row/v2",
    "thproofs": "p3-mizar-proof-row/v1",
}
SOURCE_MANIFEST_SCHEMAS = {
    name: f"p3-{name}-sources/v1" for name in SOURCE_MANIFEST_NAMES
}
EXPECTED_FAMILY_SOURCE_MANIFESTS = {
    "enigma": ("enigma",),
    "isabelle": ("isabelle",),
    "metamath": ("metamath",),
    "mizar": ("mizar",),
    "prf2": ("prf2",),
    "thproofs": ("mizar", "thproofs"),
}
VERIFIER_MANIFEST_SCHEMAS = {"metamath": "p3-metamath-verifier/v1"}
HELDOUT_SCHEMAS = {
    "atp": "p3-mml-holdout-projection/v2",
    "isabelle": "p3-isabelle-heldout/v2",
    "metamath": "p3-metamath-heldout/v1",
    "mizar": "p3-mml-holdout-projection/v2",
    "mml": "p3-mml-holdout-manifest/v2",
}

EVAL_PATHS = {family: f"evaluator/eval-{family}.jsonl" for family in FAMILIES}
HELDOUT_PATHS = {name: f"evaluator/heldout-{name}.json" for name in HELDOUT_NAMES}
VISIBILITY_PATHS = {
    family: f"evaluator/visibility-{family}.jsonl" for family in FAMILIES
}
UNION_VISIBILITY_PATH = "evaluator/visibility-corpus-union.jsonl"
DROP_PATHS = {family: f"evaluator/sidecar-drop-{family}.jsonl" for family in FAMILIES}
COHORT_PATHS = {
    family: f"evaluator/sidecar-cohort-{family}.jsonl" for family in FAMILIES
}
SEMANTIC_SIDECAR_PATHS = {
    "drop_reasons": "evaluator/sidecar-mml-drop-reasons.jsonl",
    "eval_exposure": "evaluator/sidecar-mml-exposure.jsonl",
}
SOURCE_PATHS = {
    name: f"evaluator/provenance-source-{name}.json" for name in SOURCE_MANIFEST_NAMES
}
TOKENIZER_PATH = "evaluator/provenance-tokenizer.json"
CORPUS_MANIFEST_PATH = "evaluator/provenance-corpus-manifest.json"
CORPUS_CURRENT_PATH = "evaluator/provenance-corpus-current.json"
VERIFIER_PATHS = {
    name: f"evaluator/sidecar-verifier-{name}.json"
    for name in VERIFIER_MANIFEST_SCHEMAS
}

MANIFEST_KEYS = {
    "dataset_id",
    "families",
    "generation_id",
    "inventory",
    "inventory_root_sha256",
    "manifest_root_sha256",
    "profile",
    "provenance",
    "roles",
    "schema_version",
    "version",
}
COMPLETION_KEYS = {
    "dataset_id",
    "generation_id",
    "inventory_entries",
    "inventory_root_sha256",
    "manifest_file_sha256",
    "manifest_path",
    "manifest_root_sha256",
    "profile",
    "schema_version",
    "status",
    "version",
}
INVENTORY_BASE_KEYS = {
    "bindings",
    "bytes",
    "generation_id",
    "logical_path",
    "role",
    "rows",
    "schema",
    "sha256",
}


def _canonical_path_contract(
    oracle_schemas: Mapping[str, str] | None = None,
) -> dict[str, tuple[str, str]]:
    contract: dict[str, tuple[str, str]] = {}
    for family in FAMILIES:
        contract[EVAL_PATHS[family]] = (f"eval:{family}", FAMILY_ROW_SCHEMAS[family])
        contract[VISIBILITY_PATHS[family]] = (
            f"train-visibility:{family}",
            TRAIN_VISIBILITY_SCHEMA,
        )
        contract[DROP_PATHS[family]] = (f"drop-ledger:{family}", DROP_LEDGER_SCHEMA)
        contract[COHORT_PATHS[family]] = (
            f"cohort-ledger:{family}",
            COHORT_LEDGER_SCHEMA,
        )
    for name in HELDOUT_NAMES:
        contract[HELDOUT_PATHS[name]] = (f"heldout:{name}", HELDOUT_SCHEMAS[name])
    contract[UNION_VISIBILITY_PATH] = (
        "corpus-union-visibility",
        CORPUS_UNION_VISIBILITY_SCHEMA,
    )
    contract[SEMANTIC_SIDECAR_PATHS["drop_reasons"]] = (
        "semantic-sidecar:drop_reasons",
        "p3-mml-drop-reasons/v1",
    )
    contract[SEMANTIC_SIDECAR_PATHS["eval_exposure"]] = (
        "semantic-sidecar:eval_exposure",
        "p3-mml-eval-exposure/v1",
    )
    for name in SOURCE_MANIFEST_NAMES:
        contract[SOURCE_PATHS[name]] = (
            f"source-manifest:{name}",
            SOURCE_MANIFEST_SCHEMAS[name],
        )
    contract[TOKENIZER_PATH] = ("tokenizer-seal", TOKENIZER_SEAL_SCHEMA)
    contract[CORPUS_MANIFEST_PATH] = (
        "corpus-generation-manifest",
        CORPUS_MANIFEST_SCHEMA,
    )
    contract[CORPUS_CURRENT_PATH] = (
        "corpus-generation-current",
        CORPUS_CURRENT_SCHEMA,
    )
    for name, path in VERIFIER_PATHS.items():
        contract[path] = (f"verifier-manifest:{name}", VERIFIER_MANIFEST_SCHEMAS[name])
    for name, schema in sorted((oracle_schemas or {}).items()):
        contract[f"evaluator/sidecar-oracle-{name}.json"] = (
            f"oracle-manifest:{name}",
            schema,
        )
    return dict(sorted(contract.items()))


CANONICAL_PATH_CONTRACT = _canonical_path_contract()


def _semantic_transaction_schemas() -> dict[str, str]:
    result = {
        "heldout/mml.json": "p3-mml-holdout-manifest/v2",
        "heldout/mizar.json": "p3-mml-holdout-projection/v2",
        "heldout/atp.json": "p3-mml-holdout-projection/v2",
        "sidecars/eval_exposure.jsonl": "p3-mml-eval-exposure/v1",
        "sidecars/drop_reasons.jsonl": "p3-mml-drop-reasons/v1",
    }
    for family in MML_NATIVE_FAMILIES:
        native_version = "v2" if family in {"prf2", "enigma"} else "v1"
        for directory in ("shards", "eval", "dropped"):
            result[f"{directory}/{family}.jsonl"] = (
                f"p3-mml-{directory}-artifact/{native_version}"
            )
    return result


SEMANTIC_TRANSACTION_SCHEMAS = _semantic_transaction_schemas()

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_NAME_RE = re.compile(r"[a-z0-9][a-z0-9_-]*\Z")
_VERSION_RE = re.compile(r"v[1-9][0-9]*\Z")


class EvaluatorReleaseError(RuntimeError):
    """The evaluator release is incomplete, inconsistent, or has drifted."""


@dataclass(frozen=True)
class InputArtifact:
    """One source selected by its exact corpus-transaction inventory path."""

    path: Path
    transaction_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(
            self,
            "transaction_path",
            _safe_relative_path(self.transaction_path, max_labels=None),
        )


@dataclass(frozen=True)
class GenerationBinding:
    """Validated immutable corpus transaction identity and exact inventory."""

    generation_id: str
    path: Path
    logical_root_sha256: str
    transaction_inventory_sha256: str
    manifest_file_sha256: str
    current_seal_sha256: str
    manifest_path: Path
    current_path: Path
    manifest: Mapping[str, Any]
    inventory: Mapping[str, Mapping[str, Any]]
    coordinator: GenerationCoordinator
    validated: bool = True


@dataclass(frozen=True)
class EvaluatorReleaseSpec:
    """All evaluator inputs, each selected from one validated generation."""

    dataset_id: str
    version: str
    generation: GenerationBinding
    eval_files: Mapping[str, InputArtifact]
    train_files: Mapping[str, InputArtifact]
    semantic_contract_root: Path
    heldout_manifests: Mapping[str, InputArtifact]
    tokenizer_seal: InputArtifact
    source_manifests: Mapping[str, InputArtifact]
    family_source_manifests: Mapping[str, Sequence[str]]
    drop_ledgers: Mapping[str, InputArtifact]
    cohort_ledgers: Mapping[str, InputArtifact]
    verifier_manifests: Mapping[str, InputArtifact]
    oracle_manifests: Mapping[str, InputArtifact]
    expected_semantic_holdout_root_sha256: str
    expected_tokenizer_seal_sha256: str
    expected_source_manifests_root_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_contract_root", Path(self.semantic_contract_root))


@dataclass(frozen=True)
class PlannedPayload:
    """One native or generated file in the evaluator release."""

    logical_path: str
    schema: str
    generation_id: str
    rows: int | None
    bytes_count: int
    sha256: str
    role: str
    bindings: Mapping[str, str]
    source_binding: Mapping[str, Any] | None = None
    derived_from_transaction_paths: tuple[str, ...] = ()
    source_path: Path | None = None
    data: bytes | None = None

    def read_bytes(self) -> bytes:
        """Return the planned bytes without reserializing native inputs."""

        if self.data is not None:
            return self.data
        if self.source_path is None:
            raise EvaluatorReleaseError(f"{self.logical_path}: payload has no source")
        return self.source_path.read_bytes()

    def inventory_entry(self) -> dict[str, Any]:
        """Return the exact manifest inventory record."""

        result: dict[str, Any] = {
            "bindings": dict(self.bindings),
            "bytes": self.bytes_count,
            "generation_id": self.generation_id,
            "logical_path": self.logical_path,
            "role": self.role,
            "rows": self.rows,
            "schema": self.schema,
            "sha256": self.sha256,
        }
        if self.source_binding is not None:
            result["source_binding"] = dict(self.source_binding)
        else:
            result["derived_from_transaction_paths"] = list(
                self.derived_from_transaction_paths
            )
        return result


@dataclass(frozen=True)
class ReleasePlan:
    """Deterministic evaluator payload and control bytes."""

    dataset_id: str
    generation_id: str
    payloads: tuple[PlannedPayload, ...]
    manifest: Mapping[str, Any]
    seal: Mapping[str, Any]
    manifest_bytes: bytes
    completion_seal_bytes: bytes
    manifest_root_sha256: str
    seal_sha256: str


@dataclass(frozen=True)
class EvaluatorRelease:
    """A fully verified evaluator release resolved to local paths."""

    root: Path
    manifest_path: Path
    completion_seal_path: Path
    manifest: Mapping[str, Any]
    seal: Mapping[str, Any]
    manifest_root_sha256: str
    seal_sha256: str
    family_paths: Mapping[str, Path]
    heldout_paths: Mapping[str, Path]
    class_manifest_paths: Mapping[str, Path]
    semantic_manifest_path: Path
    semantic_projection_paths: Mapping[str, Path]
    semantic_sidecar_paths: Mapping[str, Path]
    train_visibility_paths: Mapping[str, Path]
    corpus_union_visibility_path: Path
    source_manifest_paths: Mapping[str, Path]
    drop_ledger_paths: Mapping[str, Path]
    cohort_ledger_paths: Mapping[str, Path]
    tokenizer_seal_path: Path
    verifier_manifest_paths: Mapping[str, Path]
    oracle_manifest_paths: Mapping[str, Path]
    provenance: Mapping[str, Any]

    @property
    def path(self) -> Path:
        """Compatibility name used by transaction publication results."""

        return self.root


FaultInjector = Callable[[PublishPhase, Path], None]


def canonical_json_bytes(value: Any, *, newline: bool = False) -> bytes:
    """Serialize stable canonical UTF-8 JSON."""

    result = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return result + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Stream one file into SHA-256."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvaluatorReleaseError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_release_version(version: Any) -> str:
    if version == UNPUBLISHED_VERSION:
        return version
    if isinstance(version, str) and _VERSION_RE.fullmatch(version):
        return version
    raise EvaluatorReleaseError(
        "version must be __UNPUBLISHED__ for local use or explicit vN for production"
    )


def _safe_relative_path(value: str, *, max_labels: int | None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvaluatorReleaseError(f"unsafe logical path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise EvaluatorReleaseError(f"unsafe logical path {value!r}")
    if max_labels is not None and len(path.parts) != max_labels:
        raise EvaluatorReleaseError(
            f"platform evaluator paths require exactly {max_labels} labels: {value}"
        )
    return value


def _safe_name(value: str, context: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME_RE.fullmatch(value) is None:
        raise EvaluatorReleaseError(f"{context} has unsafe name {value!r}")
    return value


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluatorReleaseError(f"{context} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise EvaluatorReleaseError(f"{context} must be a JSON object")
    return value


def _exact_keys(
    values: Mapping[str, Any],
    expected: Sequence[str],
    context: str,
) -> None:
    actual = set(values)
    wanted = set(expected)
    if actual != wanted:
        raise EvaluatorReleaseError(
            f"{context} must contain exact families/keys {sorted(wanted)}; "
            f"missing={sorted(wanted - actual)}, extra={sorted(actual - wanted)}"
        )


def _manifest_root(document: Mapping[str, Any], context: str) -> str:
    declared = _require_sha256(
        document.get("manifest_root_sha256"),
        f"{context} manifest root",
    )
    body = dict(document)
    body.pop("manifest_root_sha256", None)
    actual = canonical_sha256(body)
    if declared != actual:
        raise EvaluatorReleaseError(f"{context} manifest root mismatch")
    return actual


def _transaction_inventory(
    published: Any,
) -> dict[str, Mapping[str, Any]]:
    manifest = published.manifest
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("api_version") != CORPUS_TRANSACTION_API_VERSION
        or manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA
        or manifest.get("generation_id") != published.generation_id
        or manifest.get("logical_root_sha256") != published.logical_root_sha256
    ):
        raise EvaluatorReleaseError("corpus transaction is not the current validated v2 contract")
    raw_outputs = manifest.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise EvaluatorReleaseError("corpus transaction output inventory is empty")
    inventory: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(raw_outputs):
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise EvaluatorReleaseError(
                f"corpus transaction inventory entry {index} is malformed"
            )
        path = _safe_relative_path(record["path"], max_labels=None)
        if path in inventory:
            raise EvaluatorReleaseError(f"corpus transaction inventory duplicates {path}")
        if record.get("generation_id") != published.generation_id:
            raise EvaluatorReleaseError(f"{path}: transaction generation ID mismatch")
        if (
            isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
        ):
            raise EvaluatorReleaseError(f"{path}: invalid transaction bytes")
        if (
            isinstance(record.get("rows"), bool)
            or not isinstance(record.get("rows"), int)
            or record["rows"] <= 0
        ):
            raise EvaluatorReleaseError(f"{path}: invalid transaction rows")
        _require_sha256(record.get("sha256"), f"{path} transaction SHA-256")
        if not isinstance(record.get("schema"), str) or not record["schema"]:
            raise EvaluatorReleaseError(f"{path}: transaction schema is missing")
        if record.get("role") not in {"raw", "train", "eval", "heldout", "sidecar"}:
            raise EvaluatorReleaseError(f"{path}: transaction role is invalid")
        physical = Path(published.path) / path
        if not physical.is_file() or physical.is_symlink():
            raise EvaluatorReleaseError(f"{path}: transaction output is not regular")
        if (
            physical.stat().st_size != record["bytes"]
            or sha256_file(physical) != record["sha256"]
        ):
            raise EvaluatorReleaseError(f"{path}: transaction output drift")
        inventory[path] = dict(record)
    return inventory


def generation_binding_from_transaction(
    transaction_root: str | os.PathLike[str],
    current_seal: str | os.PathLike[str],
    *,
    binary_validators: Iterable[BinaryValidator] = (),
) -> GenerationBinding:
    """Resolve and bind a corpus transaction through its strict v2 coordinator."""

    coordinator = GenerationCoordinator(
        Path(transaction_root),
        binary_validators=binary_validators,
    )
    current_path = Path(current_seal)
    expected_current = coordinator.root / "CURRENT"
    try:
        if current_path.resolve(strict=True) != expected_current.resolve(strict=True):
            raise EvaluatorReleaseError(
                "transaction CURRENT must be the coordinator's canonical pointer"
            )
        published = coordinator.resolve_current(required_siblings=FAMILIES)
    except (GenerationError, OSError) as error:
        raise EvaluatorReleaseError(f"strict corpus transaction resolution failed: {error}") from error
    if published.commit_state not in {"durable", "durable_recovered"}:
        raise EvaluatorReleaseError("corpus transaction is not durably committed")
    generation_id = published.generation_id
    path = published.path
    manifest_path = path / CORPUS_MANIFEST_NAME
    manifest_file_sha256 = published.manifest_sha256
    logical_root = published.logical_root_sha256
    inventory = _transaction_inventory(published)
    transaction_inventory_sha256 = canonical_sha256(published.manifest["outputs"])
    return GenerationBinding(
        generation_id=generation_id,
        path=path,
        logical_root_sha256=logical_root,
        transaction_inventory_sha256=transaction_inventory_sha256,
        manifest_file_sha256=manifest_file_sha256,
        current_seal_sha256=sha256_file(current_path),
        manifest_path=manifest_path,
        current_path=current_path,
        manifest=dict(published.manifest),
        inventory=inventory,
        coordinator=coordinator,
    )


def _validate_generation(binding: GenerationBinding) -> None:
    if not isinstance(binding, GenerationBinding) or not binding.validated:
        raise EvaluatorReleaseError("a validated corpus generation contract is required")
    _require_sha256(binding.logical_root_sha256, "corpus logical root")
    _require_sha256(binding.manifest_file_sha256, "corpus manifest file SHA-256")
    _require_sha256(binding.current_seal_sha256, "corpus CURRENT seal SHA-256")
    try:
        published = binding.coordinator.resolve_current(required_siblings=FAMILIES)
    except GenerationError as error:
        raise EvaluatorReleaseError(f"strict corpus transaction revalidation failed: {error}") from error
    if (
        published.generation_id != binding.generation_id
        or published.path != binding.path
        or published.manifest_sha256 != binding.manifest_file_sha256
        or published.logical_root_sha256 != binding.logical_root_sha256
        or canonical_sha256(published.manifest["outputs"])
        != binding.transaction_inventory_sha256
        or published.manifest != binding.manifest
        or sha256_file(binding.current_path) != binding.current_seal_sha256
    ):
        raise EvaluatorReleaseError("corpus transaction logical root or binding drift")
    rebound = _transaction_inventory(published)
    if rebound != binding.inventory:
        raise EvaluatorReleaseError("corpus transaction inventory binding drift")


def _bound_artifact(
    binding: GenerationBinding,
    artifact: InputArtifact,
    *,
    context: str,
    expected_path: str | None = None,
    expected_role: str,
    expected_schema: str,
    expected_sibling: str | None,
) -> Mapping[str, Any]:
    if not isinstance(artifact, InputArtifact):
        raise EvaluatorReleaseError(f"{context} is not an InputArtifact")
    transaction_path = artifact.transaction_path
    if expected_path is not None and transaction_path != expected_path:
        raise EvaluatorReleaseError(
            f"{context}: transaction path must be {expected_path}, got {transaction_path}"
        )
    try:
        record = binding.inventory[transaction_path]
    except KeyError as error:
        raise EvaluatorReleaseError(
            f"{context}: file is absent from approved transaction inventory"
        ) from error
    approved = binding.path / transaction_path
    try:
        same_file = artifact.path.samefile(approved)
    except OSError:
        same_file = False
    if not same_file:
        raise EvaluatorReleaseError(
            f"{context}: source is not the approved transaction output"
        )
    if (
        record.get("role") != expected_role
        or record.get("schema") != expected_schema
        or record.get("sibling") != expected_sibling
    ):
        raise EvaluatorReleaseError(
            f"{context}: transaction role/schema/sibling binding mismatch"
        )
    return record


def named_artifact_root(artifacts: Mapping[str, InputArtifact]) -> str:
    """Hash canonical source names and transaction-selected bytes."""

    return canonical_sha256(
        {
            name: {
                "sha256": sha256_file(artifact.path),
                "transaction_path": artifact.transaction_path,
            }
            for name, artifact in sorted(artifacts.items())
        }
    )


def _read_jsonl(
    path: Path,
    *,
    context: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.endswith(b"\n") or not line.strip():
                    raise EvaluatorReleaseError(
                        f"{context}:{line_number}: malformed JSONL framing"
                    )
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EvaluatorReleaseError(
                        f"{context}:{line_number}: invalid JSONL"
                    ) from error
                if not isinstance(record, dict):
                    raise EvaluatorReleaseError(
                        f"{context}:{line_number}: row must be an object"
                    )
                records.append(record)
    except OSError as error:
        raise EvaluatorReleaseError(f"{context}: could not read JSONL") from error
    if not records:
        raise EvaluatorReleaseError(f"{context}: JSONL file has zero rows")
    return records


def _validate_family_rows(
    path: Path,
    family: str,
    *,
    evaluation: bool,
) -> list[dict[str, Any]]:
    records = _read_jsonl(path, context=f"{family} {'eval' if evaluation else 'train'}")
    ids: set[str] = set()
    required = {"facts", "id", "schema_version", "text", "theorem"}
    if evaluation:
        required |= {"cited", "goal", "mask_end", "mask_start", "target"}
    for line_number, record in enumerate(records, 1):
        missing = required - set(record)
        if missing:
            raise EvaluatorReleaseError(
                f"{family}:{line_number}: malformed row missing {sorted(missing)}"
            )
        if record.get("schema_version") != FAMILY_ROW_SCHEMAS[family]:
            raise EvaluatorReleaseError(f"{family}:{line_number}: exact row schema mismatch")
        row_id = record["id"]
        if not isinstance(row_id, str) or not row_id or row_id in ids:
            raise EvaluatorReleaseError(f"{family}:{line_number}: invalid or duplicate row id")
        ids.add(row_id)
        if not all(isinstance(record[field], str) for field in ("text", "theorem")):
            raise EvaluatorReleaseError(f"{family}:{line_number}: malformed text/theorem")
        facts = record["facts"]
        if not isinstance(facts, dict) or not facts:
            raise EvaluatorReleaseError(f"{family}:{line_number}: facts must be nonempty")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(statement, str)
            for name, statement in facts.items()
        ):
            raise EvaluatorReleaseError(f"{family}:{line_number}: malformed fact mapping")
        if family in {"prf2", "enigma"} and not record["theorem"].startswith(f"{family}:"):
            raise EvaluatorReleaseError(
                f"{family}:{line_number}: ATP theorem is bound to the wrong family"
            )
        if evaluation and (
            not isinstance(record["goal"], str)
            or not isinstance(record["target"], str)
            or not isinstance(record["cited"], list)
            or not all(isinstance(item, str) for item in record["cited"])
            or isinstance(record["mask_start"], bool)
            or not isinstance(record["mask_start"], int)
            or isinstance(record["mask_end"], bool)
            or not isinstance(record["mask_end"], int)
        ):
            raise EvaluatorReleaseError(f"{family}:{line_number}: malformed eval row")
    return records


def _representation(family: str) -> str:
    if family in {"mizar", "thproofs"}:
        return "mizar"
    if family in {"prf2", "enigma"}:
        return "atp"
    return family


def _generic_class_id(representation: str, native_name: str) -> str:
    digest = hashlib.sha256(
        f"p3-visible-class/v1\0{representation}\0{native_name}".encode()
    ).hexdigest()
    return f"p3:v1:{representation}:{digest}"


def _generic_statement_digest(representation: str, statement: str) -> str:
    return hashlib.sha256(
        f"p3-visible-statement/v1\0{representation}\0{statement}".encode()
    ).hexdigest()


def _mapped_aliases(class_id: str, representation: str, native_name: str):
    match = re.fullmatch(r"mml:v1:(theorem|definition):([^:]+):([0-9]+)", class_id)
    if match is None:
        return {representation: {native_name}}
    kind, article, number = match.groups()
    atp_prefix = "t" if kind == "theorem" else "d"
    mizar_suffix = number if kind == "theorem" else f"def_{number}"
    return {
        "atp": {f"{atp_prefix}{number}_{article.lower()}"},
        "mizar": {f"{article}:{mizar_suffix}"},
    }


def _visibility_accumulator(
    train_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    classes: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        representation = _representation(family)
        for row in train_rows[family]:
            for raw_name, statement in row["facts"].items():
                if representation in {"mizar", "atp"}:
                    identity = mml_holdout.semantic_identity(
                        raw_name,
                        representation=representation,
                    )
                    class_id = identity.class_id
                    native_name = identity.native_name
                    statement_hash = mml_holdout.statement_digest(
                        representation,
                        statement,
                    )
                else:
                    native_name = raw_name.strip()
                    class_id = _generic_class_id(representation, native_name)
                    statement_hash = _generic_statement_digest(
                        representation,
                        statement,
                    )
                record = classes.setdefault(
                    class_id,
                    {
                        "aliases": {},
                        "families": set(),
                        "members": {},
                        "statement_hashes": {},
                    },
                )
                record["families"].add(family)
                aliases = _mapped_aliases(class_id, representation, native_name)
                for alias_representation, values in aliases.items():
                    record["aliases"].setdefault(alias_representation, set()).update(values)
                member_key = (family, representation, native_name)
                record["members"].setdefault(member_key, set()).add(raw_name)
                record["statement_hashes"].setdefault(representation, set()).add(
                    statement_hash
                )
    return classes


def _visibility_record(
    class_id: str,
    value: Mapping[str, Any],
    *,
    schema: str,
    family_filter: str | None,
) -> dict[str, Any]:
    families = sorted(value["families"])
    if family_filter is not None:
        families = [family_filter]
    members = [
        {
            "family": family,
            "native_name": native_name,
            "raw_names": sorted(raw_names),
            "representation": representation,
        }
        for (family, representation, native_name), raw_names in sorted(
            value["members"].items()
        )
        if family_filter is None or family == family_filter
    ]
    representations = {member["representation"] for member in members}
    return {
        "aliases_by_representation": {
            representation: sorted(aliases)
            for representation, aliases in sorted(value["aliases"].items())
        },
        "class_id": class_id,
        "native_members": members,
        "schema_version": schema,
        "statement_hashes_by_representation": {
            representation: sorted(value["statement_hashes"][representation])
            for representation in sorted(representations)
        },
        "visible_in_families": families,
    }


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record, newline=True) for record in records)


def _source_binding(
    binding: GenerationBinding,
    transaction_path: str,
) -> dict[str, Any]:
    record = binding.inventory[transaction_path]
    return {
        "logical_root_sha256": binding.logical_root_sha256,
        "transaction_inventory_sha256": binding.transaction_inventory_sha256,
        "role": record["role"],
        "schema": record["schema"],
        "sibling": record.get("sibling"),
        "transaction_path": transaction_path,
    }


def _native_payload(
    logical_path: str,
    *,
    artifact: InputArtifact,
    record: Mapping[str, Any],
    role: str,
    bindings: Mapping[str, str],
    generation: GenerationBinding,
) -> PlannedPayload:
    return PlannedPayload(
        logical_path=_safe_relative_path(logical_path, max_labels=2),
        schema=str(record["schema"]),
        generation_id=generation.generation_id,
        rows=int(record["rows"]),
        bytes_count=int(record["bytes"]),
        sha256=str(record["sha256"]),
        role=role,
        bindings=bindings,
        source_binding=_source_binding(generation, artifact.transaction_path),
        source_path=artifact.path,
    )


def _control_payload(
    logical_path: str,
    *,
    source_path: Path,
    schema: str,
    role: str,
    rows: int | None,
    bindings: Mapping[str, str],
    generation: GenerationBinding,
    source_binding: Mapping[str, Any],
) -> PlannedPayload:
    return PlannedPayload(
        logical_path=_safe_relative_path(logical_path, max_labels=2),
        schema=schema,
        generation_id=generation.generation_id,
        rows=rows,
        bytes_count=source_path.stat().st_size,
        sha256=sha256_file(source_path),
        role=role,
        bindings=bindings,
        source_binding=source_binding,
        source_path=source_path,
    )


def _generated_payload(
    logical_path: str,
    records: Sequence[Mapping[str, Any]],
    *,
    schema: str,
    role: str,
    bindings: Mapping[str, str],
    generation: GenerationBinding,
    derived_from: Sequence[str],
) -> PlannedPayload:
    data = _jsonl_bytes(records)
    return PlannedPayload(
        logical_path=_safe_relative_path(logical_path, max_labels=2),
        schema=schema,
        generation_id=generation.generation_id,
        rows=len(records),
        bytes_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        role=role,
        bindings=bindings,
        derived_from_transaction_paths=tuple(sorted(derived_from)),
        data=data,
    )


def _generated_json_payload(
    logical_path: str,
    value: Mapping[str, Any],
    *,
    schema: str,
    role: str,
    bindings: Mapping[str, str],
    generation: GenerationBinding,
    derived_from: Sequence[str],
) -> PlannedPayload:
    data = canonical_json_bytes(value, newline=True)
    return PlannedPayload(
        logical_path=_safe_relative_path(logical_path, max_labels=2),
        schema=schema,
        generation_id=generation.generation_id,
        rows=1,
        bytes_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        role=role,
        bindings=bindings,
        derived_from_transaction_paths=tuple(sorted(derived_from)),
        data=data,
    )


def _projection_alias(class_id: str, family: str) -> str | None:
    match = MML_MAPPED_CLASS_RE.fullmatch(class_id)
    if match is None:
        return None
    kind, article, number = match.groups()
    if family == "mizar":
        infix = "" if kind == "theorem" else "def_"
        return f"{article}:{infix}{number}"
    prefix = "t" if kind == "theorem" else "d"
    return f"{prefix}{number}_{article.lower()}"


def canonical_mml_projections(
    authoritative: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Derive deterministic projections solely from a persisted MML manifest."""

    try:
        authoritative_root = str(authoritative["manifest_root_sha256"])
        class_records = authoritative["class_records"]
        if not isinstance(class_records, list):
            raise TypeError("class_records")
        ordered_classes = sorted(class_records, key=lambda item: str(item["class_id"]))
        projections: dict[str, dict[str, Any]] = {}
        for family in MML_PROJECTIONS:
            shards = MML_PROJECTION_SHARDS[family]
            facts: set[str] = set()
            statement_hashes: set[str] = set()
            projected_classes = []
            for record in ordered_classes:
                names = {
                    str(member["native_name"])
                    for shard in sorted(record["native_members_by_shard"])
                    if shard in shards
                    for member in record["native_members_by_shard"][shard]
                }
                alias = _projection_alias(str(record["class_id"]), family)
                if alias is not None:
                    names.add(alias)
                hashes = sorted(
                    str(value)
                    for value in record["statement_digests_by_representation"].get(
                        family, ()
                    )
                )
                facts.update(names)
                statement_hashes.update(hashes)
                projected_classes.append(
                    {
                        "class_id": record["class_id"],
                        "kind": record["kind"],
                        "native_names": sorted(names),
                        "statement_hashes": hashes,
                        "route_totals": record["route_totals"],
                        "route_root_sha256": record["route_root_sha256"],
                    }
                )
            body = {
                "schema_version": HELDOUT_SCHEMAS[family],
                "family": family,
                "facts": sorted(facts),
                "shards": list(shards),
                "classes": projected_classes,
                "statement_hashes": sorted(statement_hashes),
                "canonicalization": authoritative["statement_canonicalization"][family],
                "mapping": {
                    "version": authoritative["mapping_version"],
                    "sha256": authoritative["mapping_sha256"],
                },
                "source_root_sha256": authoritative["source_root_sha256"],
                "tokenizer_root_sha256": authoritative["tokenizer_root_sha256"],
                "route_totals": authoritative["partition_projections"]["totals"],
                "route_plan_root_sha256": authoritative["route_plan_root_sha256"],
                "derived_from_selected_classes": len(class_records),
                "authoritative_manifest_root_sha256": authoritative_root,
            }
            projections[family] = {
                **body,
                "projection_root_sha256": canonical_sha256(body),
            }
        return projections
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluatorReleaseError(
            "authoritative MML manifest cannot derive canonical projections"
        ) from error


def _validate_spec(
    spec: EvaluatorReleaseSpec,
) -> tuple[mml_holdout.ValidatedHoldoutContract, str, str]:
    if spec.dataset_id != APPROVED_DATASET_ID:
        raise EvaluatorReleaseError(
            f"dataset_id must be the approved evaluator ID {APPROVED_DATASET_ID}"
        )
    _validate_release_version(spec.version)
    _validate_generation(spec.generation)
    for name, values in (
        ("eval_files", spec.eval_files),
        ("train_files", spec.train_files),
        ("drop_ledgers", spec.drop_ledgers),
        ("cohort_ledgers", spec.cohort_ledgers),
        ("family_source_manifests", spec.family_source_manifests),
    ):
        _exact_keys(values, FAMILIES, name)
    _exact_keys(spec.heldout_manifests, ("isabelle", "metamath"), "heldout_manifests")
    _exact_keys(spec.source_manifests, SOURCE_MANIFEST_NAMES, "source_manifests")
    if {
        family: tuple(names)
        for family, names in spec.family_source_manifests.items()
    } != EXPECTED_FAMILY_SOURCE_MANIFESTS:
        raise EvaluatorReleaseError("family-to-source manifest binding is not canonical")

    try:
        semantic = mml_holdout.load_holdout_contract(
            spec.semantic_contract_root,
            production=True,
        )
    except mml_holdout.HoldoutError as error:
        raise EvaluatorReleaseError(f"production semantic holdout rejected: {error}") from error
    if semantic.authoritative_root != _require_sha256(
        spec.expected_semantic_holdout_root_sha256,
        "expected semantic holdout root",
    ):
        raise EvaluatorReleaseError("semantic holdout root mismatch")

    generation = spec.generation
    for family in FAMILIES:
        _bound_artifact(
            generation,
            spec.eval_files[family],
            context=f"{family} eval",
            expected_path=f"eval/{family}.jsonl",
            expected_role="eval",
            expected_schema=FAMILY_ROW_SCHEMAS[family],
            expected_sibling=family,
        )
        _bound_artifact(
            generation,
            spec.train_files[family],
            context=f"{family} train",
            expected_path=f"train/{family}.jsonl",
            expected_role="train",
            expected_schema=FAMILY_ROW_SCHEMAS[family],
            expected_sibling=family,
        )
        _bound_artifact(
            generation,
            spec.drop_ledgers[family],
            context=f"{family} drop ledger",
            expected_path=f"drops/{family}.jsonl",
            expected_role="sidecar",
            expected_schema=DROP_LEDGER_SCHEMA,
            expected_sibling=family,
        )
        _bound_artifact(
            generation,
            spec.cohort_ledgers[family],
            context=f"{family} cohort ledger",
            expected_path=f"cohorts/{family}.jsonl",
            expected_role="sidecar",
            expected_schema=COHORT_LEDGER_SCHEMA,
            expected_sibling=family,
        )
    for family, artifact in spec.heldout_manifests.items():
        _bound_artifact(
            generation,
            artifact,
            context=f"{family} heldout",
            expected_path=f"heldout/{family}.json",
            expected_role="heldout",
            expected_schema=(
                "p3-isabelle-heldout/v2"
                if family == "isabelle"
                else "p3-metamath-heldout/v1"
            ),
            expected_sibling=family,
        )
    _bound_artifact(
        generation,
        spec.tokenizer_seal,
        context="tokenizer seal",
        expected_path="provenance/tokenizer.json",
        expected_role="sidecar",
        expected_schema=TOKENIZER_SEAL_SCHEMA,
        expected_sibling=None,
    )
    for name, artifact in spec.source_manifests.items():
        _bound_artifact(
            generation,
            artifact,
            context=f"{name} source manifest",
            expected_path=f"sources/{name}.json",
            expected_role="sidecar",
            expected_schema=SOURCE_MANIFEST_SCHEMAS[name],
            expected_sibling=name,
        )
    for name, artifact in spec.verifier_manifests.items():
        _safe_name(name, "verifier")
        try:
            schema = VERIFIER_MANIFEST_SCHEMAS[name]
        except KeyError as error:
            raise EvaluatorReleaseError(f"unsupported verifier manifest {name}") from error
        _bound_artifact(
            generation,
            artifact,
            context=f"{name} verifier",
            expected_path=f"verifiers/{name}.json",
            expected_role="sidecar",
            expected_schema=schema,
            expected_sibling=name,
        )
    for name, artifact in spec.oracle_manifests.items():
        _safe_name(name, "oracle")
        _bound_artifact(
            generation,
            artifact,
            context=f"{name} oracle",
            expected_path=f"oracles/{name}.json",
            expected_role="sidecar",
            expected_schema=f"p3-{name}-oracle/v1",
            expected_sibling=name if name in FAMILIES else None,
        )

    for relative, semantic_artifact in semantic.artifacts.items():
        transaction_path = f"semantic/{relative}"
        artifact = InputArtifact(semantic_artifact.path, transaction_path)
        _bound_artifact(
            generation,
            artifact,
            context=f"MML semantic artifact {relative}",
            expected_path=transaction_path,
            expected_role="sidecar",
            expected_schema=SEMANTIC_TRANSACTION_SCHEMAS[relative],
            expected_sibling=None,
        )
    tokenizer_sha = sha256_file(spec.tokenizer_seal.path)
    if tokenizer_sha != _require_sha256(
        spec.expected_tokenizer_seal_sha256,
        "expected tokenizer seal",
    ):
        raise EvaluatorReleaseError("tokenizer seal SHA-256 mismatch")
    source_root = named_artifact_root(spec.source_manifests)
    if source_root != _require_sha256(
        spec.expected_source_manifests_root_sha256,
        "expected source manifests root",
    ):
        raise EvaluatorReleaseError("source manifests root mismatch")
    return semantic, tokenizer_sha, source_root


def _canonical_roles(
    *,
    oracle_names: Iterable[str] = (),
    verifier_names: Iterable[str] = VERIFIER_MANIFEST_SCHEMAS,
) -> dict[str, Any]:
    return {
        "class_manifest_by_family": {
            "enigma": HELDOUT_PATHS["atp"],
            "isabelle": HELDOUT_PATHS["isabelle"],
            "metamath": HELDOUT_PATHS["metamath"],
            "mizar": HELDOUT_PATHS["mizar"],
            "prf2": HELDOUT_PATHS["atp"],
            "thproofs": HELDOUT_PATHS["mizar"],
        },
        "cohort_ledgers": COHORT_PATHS,
        "corpus_generation_current": CORPUS_CURRENT_PATH,
        "corpus_generation_manifest": CORPUS_MANIFEST_PATH,
        "corpus_union_visibility": UNION_VISIBILITY_PATH,
        "drop_ledgers": DROP_PATHS,
        "eval": EVAL_PATHS,
        "heldout": HELDOUT_PATHS,
        "oracle_manifests": {
            name: f"evaluator/sidecar-oracle-{name}.json"
            for name in sorted(oracle_names)
        },
        "semantic_sidecars": SEMANTIC_SIDECAR_PATHS,
        "source_manifests": SOURCE_PATHS,
        "tokenizer_seal": TOKENIZER_PATH,
        "train_visibility": VISIBILITY_PATHS,
        "verifier_manifests": {
            name: VERIFIER_PATHS[name] for name in sorted(verifier_names)
        },
    }


def _sha256_json_schema() -> dict[str, Any]:
    return {"pattern": "^[0-9a-f]{64}$", "type": "string"}


def _version_json_schema() -> dict[str, Any]:
    return {
        "oneOf": [
            {"const": UNPUBLISHED_VERSION},
            {"pattern": "^v[1-9][0-9]*$", "type": "string"},
        ]
    }


def manifest_json_schema(
    oracle_schemas: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Generate the manifest schema from the canonical runtime contract."""

    path_contract = _canonical_path_contract(oracle_schemas)
    sha = {"$ref": "#/$defs/sha256"}
    bindings_properties = {
        "corpus_logical_root_sha256": sha,
        "corpus_transaction_inventory_sha256": sha,
        "semantic_holdout_root_sha256": sha,
        "source_manifests_root_sha256": sha,
        "tokenizer_seal_sha256": sha,
    }
    source_binding_properties = {
        "logical_root_sha256": sha,
        "role": {
            "enum": [
                "eval",
                "heldout",
                "sidecar",
                "train",
                "transaction-current",
                "transaction-manifest",
            ]
        },
        "schema": {"pattern": "(?:/|-)v[1-9][0-9]*$", "type": "string"},
        "sibling": {"type": ["string", "null"]},
        "transaction_inventory_sha256": sha,
        "transaction_path": {"$ref": "#/$defs/transactionPath"},
    }
    inventory_properties = {
        "bindings": {"$ref": "#/$defs/bindings"},
        "bytes": {"minimum": 0, "type": "integer"},
        "derived_from_transaction_paths": {
            "items": {"$ref": "#/$defs/transactionPath"},
            "minItems": 1,
            "type": "array",
            "uniqueItems": True,
        },
        "generation_id": {"minLength": 1, "type": "string"},
        "logical_path": {"type": "string"},
        "role": {"type": "string"},
        "rows": {
            "oneOf": [
                {"minimum": 1, "type": "integer"},
                {"type": "null"},
            ]
        },
        "schema": {"type": "string"},
        "sha256": sha,
        "source_binding": {"$ref": "#/$defs/sourceBinding"},
    }
    inventory_base = {
        "additionalProperties": False,
        "oneOf": [
            {"required": ["source_binding"]},
            {"required": ["derived_from_transaction_paths"]},
        ],
        "properties": inventory_properties,
        "required": sorted(INVENTORY_BASE_KEYS),
        "type": "object",
    }
    path_branches = [
        {
            "properties": {
                "logical_path": {"const": path},
                "role": {"const": role},
                "schema": {"const": schema},
            },
            "required": ["logical_path", "role", "schema"],
        }
        for path, (role, schema) in path_contract.items()
    ]
    family_sha_object = {
        "additionalProperties": False,
        "properties": {family: sha for family in FAMILIES},
        "required": list(FAMILIES),
        "type": "object",
    }
    provenance = {
        "additionalProperties": False,
        "properties": {
            "corpus_generation": {
                "additionalProperties": False,
                "properties": {
                    "current_seal_sha256": sha,
                    "generation_id": {"minLength": 1, "type": "string"},
                    "logical_root_sha256": sha,
                    "manifest_file_sha256": sha,
                    "transaction_inventory_sha256": sha,
                },
                "required": [
                    "current_seal_sha256",
                    "generation_id",
                    "logical_root_sha256",
                    "manifest_file_sha256",
                    "transaction_inventory_sha256",
                ],
                "type": "object",
            },
            "semantic_holdout": {
                "additionalProperties": False,
                "properties": {
                    "artifact_inventory_root_sha256": sha,
                    "manifest_root_sha256": sha,
                    "production": {"const": True},
                    "source_policy_id": {"minLength": 1, "type": "string"},
                    "source_policy_sha256": sha,
                    "source_root_sha256": sha,
                    "tokenizer_root_sha256": sha,
                },
                "required": [
                    "artifact_inventory_root_sha256",
                    "manifest_root_sha256",
                    "production",
                    "source_policy_id",
                    "source_policy_sha256",
                    "source_root_sha256",
                    "tokenizer_root_sha256",
                ],
                "type": "object",
            },
            "source_manifests": {
                "additionalProperties": False,
                "properties": {
                    "family_bindings": {
                        "const": {
                            family: list(EXPECTED_FAMILY_SOURCE_MANIFESTS[family])
                            for family in FAMILIES
                        }
                    },
                    "root_sha256": sha,
                },
                "required": ["family_bindings", "root_sha256"],
                "type": "object",
            },
            "tokenizer": {
                "additionalProperties": False,
                "properties": {"seal_sha256": sha},
                "required": ["seal_sha256"],
                "type": "object",
            },
            "train_source_sha256": family_sha_object,
            "visibility_index_sha256": family_sha_object,
        },
        "required": [
            "corpus_generation",
            "semantic_holdout",
            "source_manifests",
            "tokenizer",
            "train_source_sha256",
            "visibility_index_sha256",
        ],
        "type": "object",
    }
    properties = {
        "dataset_id": {"const": APPROVED_DATASET_ID},
        "families": {"const": list(FAMILIES)},
        "generation_id": {"minLength": 1, "type": "string"},
        "inventory": {
            "items": {
                "allOf": [
                    {"$ref": "#/$defs/inventoryBase"},
                    {"oneOf": path_branches},
                ]
            },
            "minItems": len(path_contract),
            "type": "array",
        },
        "inventory_root_sha256": sha,
        "manifest_root_sha256": sha,
        "profile": {"const": PROFILE_NAME},
        "provenance": {"$ref": "#/$defs/provenance"},
        "roles": {"const": _canonical_roles(oracle_names=(oracle_schemas or {}))},
        "schema_version": {"const": MANIFEST_SCHEMA},
        "version": _version_json_schema(),
    }
    return {
        "$id": "https://edu-llm.invalid/schemas/p3-evaluator-release-v1.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(MANIFEST_KEYS),
        "title": "P3 evaluator release manifest",
        "type": "object",
        "$defs": {
            "bindings": {
                "additionalProperties": False,
                "properties": bindings_properties,
                "required": sorted(bindings_properties),
                "type": "object",
            },
            "inventoryBase": inventory_base,
            "provenance": provenance,
            "sha256": _sha256_json_schema(),
            "sourceBinding": {
                "additionalProperties": False,
                "properties": source_binding_properties,
                "required": sorted(source_binding_properties),
                "type": "object",
            },
            "transactionPath": {
                "pattern": "^(?!/)(?!.*(?:^|/)\\.\\.(?:/|$)).+$",
                "type": "string",
            },
        },
    }


def completion_json_schema() -> dict[str, Any]:
    """Generate the completion schema from the runtime constants."""

    properties = {
        "dataset_id": {"const": APPROVED_DATASET_ID},
        "generation_id": {"minLength": 1, "type": "string"},
        "inventory_entries": {"minimum": 1, "type": "integer"},
        "inventory_root_sha256": {"$ref": "#/$defs/sha256"},
        "manifest_file_sha256": {"$ref": "#/$defs/sha256"},
        "manifest_path": {"const": MANIFEST_NAME},
        "manifest_root_sha256": {"$ref": "#/$defs/sha256"},
        "profile": {"const": PROFILE_NAME},
        "schema_version": {"const": COMPLETION_SCHEMA},
        "status": {"const": "complete"},
        "version": _version_json_schema(),
    }
    return {
        "$id": "https://edu-llm.invalid/schemas/p3-evaluator-release-completion-v1.json",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "additionalProperties": False,
        "properties": properties,
        "required": sorted(COMPLETION_KEYS),
        "title": "P3 evaluator release completion seal",
        "type": "object",
        "$defs": {"sha256": _sha256_json_schema()},
    }


def write_json_schemas(directory: str | os.PathLike[str]) -> tuple[Path, Path]:
    """Write deterministic documentation schemas from the runtime constants."""

    root = Path(directory)
    manifest_path = root / "p3-evaluator-release-v1.schema.json"
    completion_path = root / "p3-evaluator-release-completion-v1.schema.json"
    manifest_path.write_text(
        json.dumps(manifest_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion_path.write_text(
        json.dumps(completion_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, completion_path


def plan_release(spec: EvaluatorReleaseSpec) -> ReleasePlan:
    """Validate all contracts and return deterministic release bytes."""

    semantic, tokenizer_sha, source_root = _validate_spec(spec)
    generation = spec.generation
    train_rows: dict[str, list[dict[str, Any]]] = {}
    train_source_sha: dict[str, str] = {}
    for family in FAMILIES:
        _validate_family_rows(spec.eval_files[family].path, family, evaluation=True)
        train_rows[family] = _validate_family_rows(
            spec.train_files[family].path,
            family,
            evaluation=False,
        )
        train_source_sha[family] = sha256_file(spec.train_files[family].path)

    bindings = {
        "corpus_logical_root_sha256": generation.logical_root_sha256,
        "corpus_transaction_inventory_sha256": generation.transaction_inventory_sha256,
        "semantic_holdout_root_sha256": semantic.authoritative_root,
        "source_manifests_root_sha256": source_root,
        "tokenizer_seal_sha256": tokenizer_sha,
    }
    roles = _canonical_roles(
        oracle_names=spec.oracle_manifests,
        verifier_names=spec.verifier_manifests,
    )

    payloads: list[PlannedPayload] = []
    for family in FAMILIES:
        for category, artifacts, output_role, logical in (
            (
                "eval",
                spec.eval_files,
                f"eval:{family}",
                roles["eval"][family],
            ),
            (
                "drop",
                spec.drop_ledgers,
                f"drop-ledger:{family}",
                roles["drop_ledgers"][family],
            ),
            (
                "cohort",
                spec.cohort_ledgers,
                f"cohort-ledger:{family}",
                roles["cohort_ledgers"][family],
            ),
        ):
            artifact = artifacts[family]
            record = generation.inventory[artifact.transaction_path]
            payloads.append(
                _native_payload(
                    logical,
                    artifact=artifact,
                    record=record,
                    role=output_role,
                    bindings=bindings,
                    generation=generation,
                )
            )

    semantic_sources = {
        "heldout/mml.json": HELDOUT_PATHS["mml"],
        "sidecars/eval_exposure.jsonl": SEMANTIC_SIDECAR_PATHS["eval_exposure"],
        "sidecars/drop_reasons.jsonl": SEMANTIC_SIDECAR_PATHS["drop_reasons"],
    }
    for relative, logical in semantic_sources.items():
        source = InputArtifact(semantic.artifacts[relative].path, f"semantic/{relative}")
        record = generation.inventory[source.transaction_path]
        role = (
            f"heldout:{Path(relative).stem}"
            if relative.startswith("heldout/")
            else f"semantic-sidecar:{Path(relative).stem}"
        )
        payloads.append(
            _native_payload(
                logical,
                artifact=source,
                record=record,
                role=role,
                bindings=bindings,
                generation=generation,
            )
        )
    canonical_projections = canonical_mml_projections(semantic.manifest)
    for family in MML_PROJECTIONS:
        payloads.append(
            _generated_json_payload(
                HELDOUT_PATHS[family],
                canonical_projections[family],
                schema=HELDOUT_SCHEMAS[family],
                role=f"heldout:{family}",
                bindings=bindings,
                generation=generation,
                derived_from=("semantic/heldout/mml.json",),
            )
        )
    for family, artifact in spec.heldout_manifests.items():
        record = generation.inventory[artifact.transaction_path]
        payloads.append(
            _native_payload(
                roles["heldout"][family],
                artifact=artifact,
                record=record,
                role=f"heldout:{family}",
                bindings=bindings,
                generation=generation,
            )
        )

    manifest_source_binding = {
        "logical_root_sha256": generation.logical_root_sha256,
        "transaction_inventory_sha256": generation.transaction_inventory_sha256,
        "role": "transaction-manifest",
        "schema": CORPUS_MANIFEST_SCHEMA,
        "sibling": None,
        "transaction_path": CORPUS_MANIFEST_NAME,
    }
    current_source_binding = {
        "logical_root_sha256": generation.logical_root_sha256,
        "transaction_inventory_sha256": generation.transaction_inventory_sha256,
        "role": "transaction-current",
        "schema": CORPUS_CURRENT_SCHEMA,
        "sibling": None,
        "transaction_path": "CURRENT",
    }
    payloads.extend(
        (
            _control_payload(
                roles["corpus_generation_manifest"],
                source_path=generation.manifest_path,
                schema=CORPUS_MANIFEST_SCHEMA,
                role="corpus-generation-manifest",
                rows=1,
                bindings=bindings,
                generation=generation,
                source_binding=manifest_source_binding,
            ),
            _control_payload(
                roles["corpus_generation_current"],
                source_path=generation.current_path,
                schema=CORPUS_CURRENT_SCHEMA,
                role="corpus-generation-current",
                rows=1,
                bindings=bindings,
                generation=generation,
                source_binding=current_source_binding,
            ),
        )
    )
    tokenizer_record = generation.inventory[spec.tokenizer_seal.transaction_path]
    payloads.append(
        _native_payload(
            roles["tokenizer_seal"],
            artifact=spec.tokenizer_seal,
            record=tokenizer_record,
            role="tokenizer-seal",
            bindings=bindings,
            generation=generation,
        )
    )
    for name, artifact in spec.source_manifests.items():
        payloads.append(
            _native_payload(
                roles["source_manifests"][name],
                artifact=artifact,
                record=generation.inventory[artifact.transaction_path],
                role=f"source-manifest:{name}",
                bindings=bindings,
                generation=generation,
            )
        )
    for category, artifacts in (
        ("verifier", spec.verifier_manifests),
        ("oracle", spec.oracle_manifests),
    ):
        role_map = roles[f"{category}_manifests"]
        for name, artifact in sorted(artifacts.items()):
            payloads.append(
                _native_payload(
                    role_map[name],
                    artifact=artifact,
                    record=generation.inventory[artifact.transaction_path],
                    role=f"{category}-manifest:{name}",
                    bindings=bindings,
                    generation=generation,
                )
            )

    classes = _visibility_accumulator(train_rows)
    family_visibility_sha: dict[str, str] = {}
    for family in FAMILIES:
        records = [
            _visibility_record(
                class_id,
                value,
                schema=TRAIN_VISIBILITY_SCHEMA,
                family_filter=family,
            )
            for class_id, value in sorted(classes.items())
            if family in value["families"]
        ]
        payload = _generated_payload(
            roles["train_visibility"][family],
            records,
            schema=TRAIN_VISIBILITY_SCHEMA,
            role=f"train-visibility:{family}",
            bindings=bindings,
            generation=generation,
            derived_from=(spec.train_files[family].transaction_path,),
        )
        payloads.append(payload)
        family_visibility_sha[family] = payload.sha256
    union_records = [
        _visibility_record(
            class_id,
            value,
            schema=CORPUS_UNION_VISIBILITY_SCHEMA,
            family_filter=None,
        )
        for class_id, value in sorted(classes.items())
    ]
    payloads.append(
        _generated_payload(
            roles["corpus_union_visibility"],
            union_records,
            schema=CORPUS_UNION_VISIBILITY_SCHEMA,
            role="corpus-union-visibility",
            bindings=bindings,
            generation=generation,
            derived_from=tuple(
                spec.train_files[family].transaction_path for family in FAMILIES
            ),
        )
    )

    payloads.sort(key=lambda item: item.logical_path)
    paths = [item.logical_path for item in payloads]
    release_roles = [item.role for item in payloads]
    if len(paths) != len(set(paths)) or len(release_roles) != len(set(release_roles)):
        raise EvaluatorReleaseError("release payload paths and roles must be unique")
    oracle_schemas = {
        name: str(generation.inventory[artifact.transaction_path]["schema"])
        for name, artifact in spec.oracle_manifests.items()
    }
    path_contract = _canonical_path_contract(oracle_schemas)
    if set(paths) != set(path_contract):
        raise EvaluatorReleaseError("release paths do not match the canonical evaluator group")
    for payload in payloads:
        expected_role, expected_schema = path_contract[payload.logical_path]
        if (payload.role, payload.schema) != (expected_role, expected_schema):
            raise EvaluatorReleaseError(
                f"{payload.logical_path}: role/schema is not canonical"
            )
    inventory = [item.inventory_entry() for item in payloads]
    inventory_root = canonical_sha256(inventory)
    provenance = {
        "corpus_generation": {
            "current_seal_sha256": generation.current_seal_sha256,
            "generation_id": generation.generation_id,
            "logical_root_sha256": generation.logical_root_sha256,
            "manifest_file_sha256": generation.manifest_file_sha256,
            "transaction_inventory_sha256": generation.transaction_inventory_sha256,
        },
        "semantic_holdout": {
            "artifact_inventory_root_sha256": semantic.manifest[
                "artifact_inventory_root_sha256"
            ],
            "manifest_root_sha256": semantic.authoritative_root,
            "production": True,
            "source_policy_id": semantic.manifest["source_identity_policy"]["policy_id"],
            "source_policy_sha256": semantic.manifest["source_identity_policy"][
                "policy_sha256"
            ],
            "source_root_sha256": semantic.manifest["source_root_sha256"],
            "tokenizer_root_sha256": semantic.manifest["tokenizer_root_sha256"],
        },
        "source_manifests": {
            "family_bindings": {
                family: list(EXPECTED_FAMILY_SOURCE_MANIFESTS[family])
                for family in FAMILIES
            },
            "root_sha256": source_root,
        },
        "tokenizer": {"seal_sha256": tokenizer_sha},
        "train_source_sha256": train_source_sha,
        "visibility_index_sha256": family_visibility_sha,
    }
    body = {
        "dataset_id": spec.dataset_id,
        "families": list(FAMILIES),
        "generation_id": generation.generation_id,
        "inventory": inventory,
        "inventory_root_sha256": inventory_root,
        "profile": PROFILE_NAME,
        "provenance": provenance,
        "roles": roles,
        "schema_version": MANIFEST_SCHEMA,
        "version": spec.version,
    }
    root = canonical_sha256(body)
    manifest = {**body, "manifest_root_sha256": root}
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    seal = {
        "dataset_id": spec.dataset_id,
        "generation_id": generation.generation_id,
        "inventory_entries": len(inventory),
        "inventory_root_sha256": inventory_root,
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "manifest_path": MANIFEST_NAME,
        "manifest_root_sha256": root,
        "profile": PROFILE_NAME,
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "version": spec.version,
    }
    seal_bytes = canonical_json_bytes(seal, newline=True)
    return ReleasePlan(
        dataset_id=spec.dataset_id,
        generation_id=generation.generation_id,
        payloads=tuple(payloads),
        manifest=manifest,
        seal=seal,
        manifest_bytes=manifest_bytes,
        completion_seal_bytes=seal_bytes,
        manifest_root_sha256=root,
        seal_sha256=hashlib.sha256(seal_bytes).hexdigest(),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_payload(payload: PlannedPayload, destination: Path) -> None:
    data = payload.read_bytes()
    if len(data) != payload.bytes_count or hashlib.sha256(data).hexdigest() != payload.sha256:
        raise EvaluatorReleaseError(f"{payload.logical_path}: source changed during staging")
    _write_file(destination, data)


def _materialize_into(plan: ReleasePlan, root: Path) -> None:
    for payload in plan.payloads:
        _write_payload(payload, root / payload.logical_path)
    _write_file(root / MANIFEST_NAME, plan.manifest_bytes)
    _write_file(root / COMPLETION_SEAL_NAME, plan.completion_seal_bytes)
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(directory)
    _fsync_directory(root)
    load_release(root, expected_version=str(plan.manifest["version"]))


def materialize_release(
    plan: ReleasePlan,
    destination: str | os.PathLike[str],
) -> EvaluatorRelease:
    """Materialize one local immutable release for tests or pre-publication review."""

    destination_path = Path(destination)
    if destination_path.exists() or destination_path.is_symlink():
        raise EvaluatorReleaseError(f"immutable release already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.{plan.generation_id}.staging.",
            dir=destination_path.parent,
        )
    )
    try:
        _materialize_into(plan, staging)
        os.rename(staging, destination_path)
        _fsync_directory(destination_path.parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return load_release(
        destination_path,
        expected_version=str(plan.manifest["version"]),
    )


def build_release(
    spec: EvaluatorReleaseSpec,
    destination: str | os.PathLike[str],
) -> EvaluatorRelease:
    """Plan and materialize one local evaluator release."""

    return materialize_release(plan_release(spec), destination)


class EvaluatorReleaseCoordinator:
    """Evaluator adapter that delegates publication to transaction-v2."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        expected_version: str = UNPUBLISHED_VERSION,
    ):
        self.root = Path(root)
        self.expected_version = _validate_release_version(expected_version)
        self._transaction_coordinator: GenerationCoordinator | None = None

    @property
    def transaction_coordinator(self) -> GenerationCoordinator:
        """Return the accepted transaction coordinator used by this adapter."""

        if self._transaction_coordinator is None:
            raise EvaluatorReleaseError("evaluator transaction has not been initialized")
        return self._transaction_coordinator

    @staticmethod
    def _binary_rows(path: Path, _context: Any) -> int:
        if path.suffix == ".json":
            return 1
        return max(1, len(path.read_bytes().splitlines()))

    def _transaction_plan(
        self,
        plan: ReleasePlan,
    ) -> tuple[GenerationPlan, tuple[BinaryValidator, ...], dict[str, bytes]]:
        transport_schema = "p3-evaluator-transport-row/v1"
        transport_drop_schema = "p3-evaluator-transport-drop/v1"
        transport_heldout_schema = "p3-evaluator-transport-heldout/v1"
        transport_paths = {
            "raw": "_transport/raw.jsonl",
            "train": "_transport/train.jsonl",
            "eval": "_transport/eval.jsonl",
            "drop": "_transport/drop.jsonl",
            "heldout": "_transport/heldout.json",
        }
        transport_validator = JsonlValidator(
            schema_version=transport_schema,
            required_fields=("payload",),
        )
        drop_validator = JsonlValidator(
            schema_version=transport_drop_schema,
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
        )
        heldout_validator = JsonObjectValidator(
            schema_version=transport_heldout_schema,
            required_fields=("release_manifest_root_sha256",),
            require_generation_links=True,
        )
        outputs = [
            OutputSpec(
                path=transport_paths[role],
                role=OutputRole(role),
                schema=transport_schema,
                sibling="evaluator-release",
                validator=transport_validator,
            )
            for role in ("raw", "train", "eval")
        ]
        outputs.extend(
            (
                OutputSpec(
                    path=transport_paths["drop"],
                    role=OutputRole.SIDECAR,
                    schema=transport_drop_schema,
                    sibling="evaluator-release",
                    drop_types=("transport",),
                    validator=drop_validator,
                ),
                OutputSpec(
                    path=transport_paths["heldout"],
                    role=OutputRole.HELDOUT,
                    schema=transport_heldout_schema,
                    sibling="evaluator-release",
                    validator=heldout_validator,
                ),
            )
        )
        payload_bytes = {
            payload.logical_path: payload.read_bytes() for payload in plan.payloads
        }
        payload_bytes[MANIFEST_NAME] = plan.manifest_bytes
        payload_bytes[COMPLETION_SEAL_NAME] = plan.completion_seal_bytes
        payload_validator = BinaryValidator(
            schema_version=TRANSACTION_PAYLOAD_SCHEMA,
            validator_id=TRANSACTION_PAYLOAD_VALIDATOR_ID,
            validate=self._binary_rows,
        )
        for path in sorted(payload_bytes):
            outputs.append(
                OutputSpec(
                    path=path,
                    role=OutputRole.SIDECAR,
                    schema=TRANSACTION_PAYLOAD_SCHEMA,
                    validator=payload_validator,
                )
            )
        transaction_plan = GenerationPlan(
            generation_id=plan.generation_id,
            source_generation_id=str(
                plan.manifest["provenance"]["corpus_generation"][
                    "logical_root_sha256"
                ]
            ),
            requested_siblings=("evaluator-release",),
            outputs=tuple(outputs),
        )
        return transaction_plan, (payload_validator,), payload_bytes

    def publish(
        self,
        plan: ReleasePlan,
        *,
        fault_injector: FaultInjector | None = None,
    ) -> EvaluatorRelease:
        """Publish only through the accepted secure transaction-v2 writer."""

        if plan.manifest["version"] != self.expected_version:
            raise EvaluatorReleaseError("evaluator publication version does not match caller")
        transaction_plan, validators, payload_bytes = self._transaction_plan(plan)
        coordinator = GenerationCoordinator(self.root, binary_validators=validators)
        self._transaction_coordinator = coordinator

        def producer(writer: Any) -> None:
            raw_rows = [
                {"payload": disposition, "schema_version": "p3-evaluator-transport-row/v1"}
                for disposition in ("train", "eval", "drop")
            ]
            writer.write_bytes(
                "_transport/raw.jsonl",
                b"".join(canonical_json_bytes(row, newline=True) for row in raw_rows),
            )
            occurrences = writer.raw_occurrences("_transport/raw.jsonl")
            writer.write_routed_jsonl("_transport/train.jsonl", (occurrences[0],))
            writer.write_routed_jsonl("_transport/eval.jsonl", (occurrences[1],))
            writer.write_drop_sidecar(
                "_transport/drop.jsonl",
                (
                    DropRecord(
                        occurrence_id=occurrences[2].occurrence_id,
                        drop_type="transport",
                        details={"reason": "transaction accounting carrier"},
                    ),
                ),
            )
            writer.write_linked_json(
                "_transport/heldout.json",
                {"release_manifest_root_sha256": plan.manifest_root_sha256},
            )
            for path, data in sorted(payload_bytes.items()):
                writer.write_bytes(path, data)

        coordinator.publish(
            transaction_plan,
            producer,
            fault_injector=fault_injector,
        )
        return self.resolve_current()

    def resolve_current(self) -> EvaluatorRelease:
        """Resolve through transaction-v2, then verify the evaluator group."""

        payload_validator = BinaryValidator(
            schema_version=TRANSACTION_PAYLOAD_SCHEMA,
            validator_id=TRANSACTION_PAYLOAD_VALIDATOR_ID,
            validate=self._binary_rows,
        )
        coordinator = GenerationCoordinator(
            self.root,
            binary_validators=(payload_validator,),
        )
        self._transaction_coordinator = coordinator
        published = coordinator.resolve_current(
            required_siblings=("evaluator-release",),
        )
        return load_release(
            published.path,
            expected_version=self.expected_version,
            transaction_envelope=True,
        )


def _inventory_by_path(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = manifest.get("inventory")
    if not isinstance(raw, list) or not raw:
        raise EvaluatorReleaseError("manifest inventory must be nonempty")
    if manifest.get("inventory_root_sha256") != canonical_sha256(raw):
        raise EvaluatorReleaseError("manifest inventory root mismatch")
    result: dict[str, Mapping[str, Any]] = {}
    previous = ""
    roles: set[str] = set()
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise EvaluatorReleaseError(f"inventory entry {index} is malformed")
        required = {
            "bindings",
            "bytes",
            "generation_id",
            "logical_path",
            "role",
            "rows",
            "schema",
            "sha256",
        }
        lineage = {"source_binding", "derived_from_transaction_paths"} & set(entry)
        if len(lineage) != 1 or set(entry) != required | lineage:
            raise EvaluatorReleaseError(f"inventory entry {index} fields are not exact")
        path = _safe_relative_path(entry["logical_path"], max_labels=2)
        role = entry["role"]
        if path <= previous or path in result:
            raise EvaluatorReleaseError("inventory paths are not canonical and unique")
        if not isinstance(role, str) or not role or role in roles:
            raise EvaluatorReleaseError("inventory roles must be nonempty and unique")
        previous = path
        roles.add(role)
        result[path] = entry
    return result


def _verify_inventory_file(
    root: Path,
    path: str,
    entry: Mapping[str, Any],
    generation_id: str,
) -> None:
    if entry.get("generation_id") != generation_id:
        raise EvaluatorReleaseError(f"{path}: inventory generation mismatch")
    physical = root / path
    if not physical.is_file() or physical.is_symlink():
        raise EvaluatorReleaseError(f"{path}: inventory payload is missing")
    if (
        physical.stat().st_size != entry.get("bytes")
        or sha256_file(physical) != entry.get("sha256")
    ):
        raise EvaluatorReleaseError(f"{path}: sha256 or byte count mismatch")
    rows = entry.get("rows")
    if rows is not None:
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise EvaluatorReleaseError(f"{path}: rows must be positive or null")
        actual = len(physical.read_bytes().splitlines())
        if path.endswith(".json") and actual != 1:
            actual = 1
        if actual != rows:
            raise EvaluatorReleaseError(f"{path}: row count mismatch")


def _role_map(
    roles: Mapping[str, Any],
    name: str,
    expected: Sequence[str] | None = None,
) -> dict[str, str]:
    value = roles.get(name)
    if not isinstance(value, dict):
        raise EvaluatorReleaseError(f"roles.{name} must be an object")
    if expected is not None:
        _exact_keys(value, expected, f"roles.{name}")
    result = {}
    for key, path in value.items():
        if not isinstance(path, str):
            raise EvaluatorReleaseError(f"roles.{name}.{key} is not a path")
        result[key] = _safe_relative_path(path, max_labels=2)
    return result


def _verify_roles(
    manifest: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        raise EvaluatorReleaseError("manifest roles are missing")
    expected_keys = {
        "class_manifest_by_family",
        "cohort_ledgers",
        "corpus_generation_current",
        "corpus_generation_manifest",
        "corpus_union_visibility",
        "drop_ledgers",
        "eval",
        "heldout",
        "oracle_manifests",
        "semantic_sidecars",
        "source_manifests",
        "tokenizer_seal",
        "train_visibility",
        "verifier_manifests",
    }
    if set(roles) != expected_keys:
        raise EvaluatorReleaseError("manifest role categories are not exact")
    exact_maps = {
        "eval": EVAL_PATHS,
        "heldout": HELDOUT_PATHS,
        "train_visibility": VISIBILITY_PATHS,
        "drop_ledgers": DROP_PATHS,
        "cohort_ledgers": COHORT_PATHS,
        "source_manifests": SOURCE_PATHS,
        "semantic_sidecars": SEMANTIC_SIDECAR_PATHS,
        "class_manifest_by_family": {
            "enigma": HELDOUT_PATHS["atp"],
            "isabelle": HELDOUT_PATHS["isabelle"],
            "metamath": HELDOUT_PATHS["metamath"],
            "mizar": HELDOUT_PATHS["mizar"],
            "prf2": HELDOUT_PATHS["atp"],
            "thproofs": HELDOUT_PATHS["mizar"],
        },
    }
    for name, expected in exact_maps.items():
        if _role_map(roles, name, tuple(expected)) != expected:
            raise EvaluatorReleaseError(f"roles.{name} is not canonical")
    singles = {
        "corpus_generation_current": CORPUS_CURRENT_PATH,
        "corpus_generation_manifest": CORPUS_MANIFEST_PATH,
        "corpus_union_visibility": UNION_VISIBILITY_PATH,
        "tokenizer_seal": TOKENIZER_PATH,
    }
    for name, expected in singles.items():
        if roles.get(name) != expected:
            raise EvaluatorReleaseError(f"roles.{name} is not canonical")
    referenced = set(singles.values())
    for value in exact_maps.values():
        referenced.update(value.values())
    for name in ("verifier_manifests", "oracle_manifests"):
        referenced.update(_role_map(roles, name).values())
    if referenced != set(entries):
        raise EvaluatorReleaseError("roles do not cover the exact inventory once")


def _verify_canonical_inventory(
    manifest: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    roles = manifest["roles"]
    verifier_roles = _role_map(roles, "verifier_manifests")
    if verifier_roles != VERIFIER_PATHS:
        raise EvaluatorReleaseError("verifier manifest roles are not canonical")
    oracle_roles = _role_map(roles, "oracle_manifests")
    oracle_schemas: dict[str, str] = {}
    for name, path in oracle_roles.items():
        expected_path = f"evaluator/sidecar-oracle-{name}.json"
        if path != expected_path or path not in entries:
            raise EvaluatorReleaseError("oracle manifest roles are not canonical")
        oracle_schemas[name] = str(entries[path].get("schema"))
    contract = _canonical_path_contract(oracle_schemas)
    if set(entries) != set(contract):
        raise EvaluatorReleaseError("inventory does not match the canonical evaluator group")
    for path, entry in entries.items():
        if (entry.get("role"), entry.get("schema")) != contract[path]:
            raise EvaluatorReleaseError(f"{path}: canonical role/schema mismatch")

    expected_derived = {
        VISIBILITY_PATHS[family]: [f"train/{family}.jsonl"] for family in FAMILIES
    }
    expected_derived[UNION_VISIBILITY_PATH] = [
        f"train/{family}.jsonl" for family in FAMILIES
    ]
    for family in MML_PROJECTIONS:
        expected_derived[HELDOUT_PATHS[family]] = ["semantic/heldout/mml.json"]
    for path, derived in expected_derived.items():
        if entries[path].get("derived_from_transaction_paths") != derived:
            raise EvaluatorReleaseError(f"{path}: canonical derivation lineage mismatch")

    expected_sources = {
        EVAL_PATHS[family]: f"eval/{family}.jsonl" for family in FAMILIES
    }
    expected_sources.update(
        {DROP_PATHS[family]: f"drops/{family}.jsonl" for family in FAMILIES}
    )
    expected_sources.update(
        {COHORT_PATHS[family]: f"cohorts/{family}.jsonl" for family in FAMILIES}
    )
    expected_sources.update(
        {SOURCE_PATHS[name]: f"sources/{name}.json" for name in SOURCE_MANIFEST_NAMES}
    )
    expected_sources.update(
        {
            HELDOUT_PATHS["isabelle"]: "heldout/isabelle.json",
            HELDOUT_PATHS["metamath"]: "heldout/metamath.json",
            HELDOUT_PATHS["mml"]: "semantic/heldout/mml.json",
            SEMANTIC_SIDECAR_PATHS[
                "eval_exposure"
            ]: "semantic/sidecars/eval_exposure.jsonl",
            SEMANTIC_SIDECAR_PATHS[
                "drop_reasons"
            ]: "semantic/sidecars/drop_reasons.jsonl",
            TOKENIZER_PATH: "provenance/tokenizer.json",
            CORPUS_MANIFEST_PATH: CORPUS_MANIFEST_NAME,
            CORPUS_CURRENT_PATH: "CURRENT",
            VERIFIER_PATHS["metamath"]: "verifiers/metamath.json",
        }
    )
    for path, transaction_path in expected_sources.items():
        binding = entries[path].get("source_binding")
        if not isinstance(binding, dict) or binding.get("transaction_path") != transaction_path:
            raise EvaluatorReleaseError(f"{path}: canonical source lineage mismatch")


def _verify_packaged_transaction(
    root: Path,
    manifest: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    provenance = manifest["provenance"]["corpus_generation"]
    transaction_manifest_path = root / CORPUS_MANIFEST_PATH
    transaction_current_path = root / CORPUS_CURRENT_PATH
    transaction_manifest = _read_json(
        transaction_manifest_path,
        "packaged corpus transaction manifest",
    )
    current = _read_json(transaction_current_path, "packaged corpus CURRENT")
    if (
        transaction_manifest.get("schema_version") != CORPUS_MANIFEST_SCHEMA
        or transaction_manifest.get("api_version") != CORPUS_TRANSACTION_API_VERSION
        or transaction_manifest.get("generation_id") != manifest["generation_id"]
        or transaction_manifest.get("logical_root_sha256")
        != provenance.get("logical_root_sha256")
        or sha256_file(transaction_manifest_path)
        != provenance.get("manifest_file_sha256")
        or current.get("schema_version") != CORPUS_CURRENT_SCHEMA
        or current.get("generation_id") != manifest["generation_id"]
        or current.get("logical_root_sha256") != provenance.get("logical_root_sha256")
        or current.get("manifest_sha256") != provenance.get("manifest_file_sha256")
        or sha256_file(transaction_current_path)
        != provenance.get("current_seal_sha256")
    ):
        raise EvaluatorReleaseError("packaged corpus transaction provenance drift")
    transaction_outputs = {
        item["path"]: item for item in transaction_manifest.get("outputs", ())
    }
    for path, entry in entries.items():
        binding = entry.get("source_binding")
        if binding is None:
            derived = entry.get("derived_from_transaction_paths")
            if not isinstance(derived, list) or not derived:
                raise EvaluatorReleaseError(f"{path}: generated binding is missing")
            if any(item not in transaction_outputs for item in derived):
                raise EvaluatorReleaseError(f"{path}: generated binding leaves inventory")
            continue
        if (
            not isinstance(binding, dict)
            or binding.get("logical_root_sha256")
            != provenance.get("logical_root_sha256")
            or binding.get("transaction_inventory_sha256")
            != provenance.get("transaction_inventory_sha256")
        ):
            raise EvaluatorReleaseError(f"{path}: transaction source binding is invalid")
        transaction_path = binding.get("transaction_path")
        if transaction_path in {CORPUS_MANIFEST_NAME, "CURRENT"}:
            continue
        transaction_record = transaction_outputs.get(transaction_path)
        if transaction_record is None or any(
            binding.get(field) != transaction_record.get(field)
            for field in ("role", "schema", "sibling")
        ):
            raise EvaluatorReleaseError(f"{path}: transaction source binding drift")
        if (
            entry.get("sha256") != transaction_record.get("sha256")
            or entry.get("bytes") != transaction_record.get("bytes")
            or entry.get("rows") != transaction_record.get("rows")
        ):
            raise EvaluatorReleaseError(f"{path}: packaged source differs from transaction")


def _verify_packaged_semantic(root: Path, manifest: Mapping[str, Any]) -> None:
    authoritative = _read_json(root / HELDOUT_PATHS["mml"], "MML authoritative holdout")
    semantic_provenance = manifest["provenance"]["semantic_holdout"]
    authoritative_root = _manifest_root(authoritative, "MML authoritative holdout")
    if authoritative_root != semantic_provenance["manifest_root_sha256"]:
        raise EvaluatorReleaseError("packaged semantic holdout root drift")
    policy = authoritative.get("source_identity_policy")
    loader = authoritative.get("loader_contract")
    ordered_inputs = authoritative.get("ordered_inputs")
    if (
        not isinstance(policy, dict)
        or not isinstance(loader, dict)
        or not isinstance(ordered_inputs, list)
        or [record.get("shard") for record in ordered_inputs]
        != list(MML_NATIVE_FAMILIES)
        or policy.get("test_only") is not False
        or policy.get("injected_test_seams") is not False
        or loader.get("publication_mode") != "production"
    ):
        raise EvaluatorReleaseError("packaged semantic holdout is not production")
    try:
        policy_payload = {
            "version": MML_SOURCE_IDENTITY_POLICY_SCHEMA,
            "policy_id": policy["policy_id"],
            "test_only": False,
            "shards": {
                record["shard"]: {
                    "input_sha256": record["sha256"],
                    "source_snapshots": record["source_snapshots"],
                    "source_manifest_root_sha256": record[
                        "source_manifest_root_sha256"
                    ],
                    "quality_filter_root_sha256": record[
                        "quality_filter_root_sha256"
                    ],
                    "schema_generation_root_sha256": record[
                        "schema_generation_root_sha256"
                    ],
                }
                for record in ordered_inputs
            },
        }
    except (KeyError, TypeError) as error:
        raise EvaluatorReleaseError(
            "packaged semantic source identity is incomplete"
        ) from error
    source_root = canonical_sha256(ordered_inputs)
    policy_sha = canonical_sha256(policy_payload)
    if (
        authoritative.get("source_root_sha256") != source_root
        or policy.get("policy_sha256") != policy_sha
        or semantic_provenance.get("source_root_sha256") != source_root
        or semantic_provenance.get("source_policy_id") != policy.get("policy_id")
        or semantic_provenance.get("source_policy_sha256") != policy_sha
        or semantic_provenance.get("tokenizer_root_sha256")
        != authoritative.get("tokenizer_root_sha256")
        or semantic_provenance.get("artifact_inventory_root_sha256")
        != canonical_sha256(authoritative.get("artifact_inventory"))
    ):
        raise EvaluatorReleaseError("packaged semantic source roots drift")
    expected = canonical_mml_projections(authoritative)
    for family in MML_PROJECTIONS:
        projection_path = root / HELDOUT_PATHS[family]
        actual = _read_json(projection_path, f"{family} projection")
        if (
            actual != expected[family]
            or projection_path.read_bytes()
            != canonical_json_bytes(expected[family], newline=True)
        ):
            raise EvaluatorReleaseError(f"{family} semantic projection drift")
    records = {
        item["path"]: item for item in authoritative["artifact_inventory"]
    }
    for relative, packaged in (
        ("sidecars/eval_exposure.jsonl", SEMANTIC_SIDECAR_PATHS["eval_exposure"]),
        ("sidecars/drop_reasons.jsonl", SEMANTIC_SIDECAR_PATHS["drop_reasons"]),
    ):
        if sha256_file(root / packaged) != records[relative]["sha256"]:
            raise EvaluatorReleaseError(f"{relative}: semantic sidecar drift")


def _verify_visibility_file(
    path: Path,
    *,
    schema: str,
    family: str | None,
) -> None:
    records = _read_jsonl(path, context=str(path))
    class_ids = []
    for record in records:
        if record.get("schema_version") != schema:
            raise EvaluatorReleaseError(f"{path}: visibility schema mismatch")
        class_id = record.get("class_id")
        if not isinstance(class_id, str) or not class_id:
            raise EvaluatorReleaseError(f"{path}: visibility class ID is invalid")
        class_ids.append(class_id)
        families = record.get("visible_in_families")
        if (
            not isinstance(families, list)
            or families != sorted(set(families))
            or (family is not None and families != [family])
            or not isinstance(record.get("native_members"), list)
            or not isinstance(record.get("aliases_by_representation"), dict)
            or not isinstance(record.get("statement_hashes_by_representation"), dict)
        ):
            raise EvaluatorReleaseError(f"{path}: visibility record is malformed")
    if class_ids != sorted(set(class_ids)):
        raise EvaluatorReleaseError(f"{path}: visibility classes are not canonical")


def _verify_internal_file_contracts(
    root: Path,
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    for family in FAMILIES:
        for path, schema, family_field in (
            (DROP_PATHS[family], DROP_LEDGER_SCHEMA, "sibling"),
            (COHORT_PATHS[family], COHORT_LEDGER_SCHEMA, "family"),
        ):
            for row in _read_jsonl(root / path, context=path):
                if (
                    row.get("schema_version") != schema
                    or row.get(family_field) != family
                ):
                    raise EvaluatorReleaseError(f"{path}: internal schema/family mismatch")
    for name in ("isabelle", "metamath"):
        record = _read_json(root / HELDOUT_PATHS[name], f"{name} heldout")
        if (
            record.get("schema_version") != HELDOUT_SCHEMAS[name]
            or record.get("family") != name
        ):
            raise EvaluatorReleaseError(
                f"{HELDOUT_PATHS[name]}: internal schema/family mismatch"
            )
    for name in SOURCE_MANIFEST_NAMES:
        record = _read_json(root / SOURCE_PATHS[name], f"{name} source manifest")
        if (
            record.get("schema_version") != SOURCE_MANIFEST_SCHEMAS[name]
            or record.get("source") != name
        ):
            raise EvaluatorReleaseError(
                f"{SOURCE_PATHS[name]}: internal schema/source mismatch"
            )
    tokenizer = _read_json(root / TOKENIZER_PATH, "tokenizer seal")
    if tokenizer.get("schema_version") != TOKENIZER_SEAL_SCHEMA:
        raise EvaluatorReleaseError(f"{TOKENIZER_PATH}: internal schema mismatch")
    for name, path in VERIFIER_PATHS.items():
        record = _read_json(root / path, f"{name} verifier manifest")
        if record.get("schema_version") != VERIFIER_MANIFEST_SCHEMAS[name]:
            raise EvaluatorReleaseError(f"{path}: internal schema mismatch")
    for path, entry in entries.items():
        if not path.startswith("evaluator/"):
            raise EvaluatorReleaseError(f"{path}: payload leaves evaluator group")
        if entry.get("logical_path") != path:
            raise EvaluatorReleaseError(f"{path}: inventory path mismatch")


def load_release(
    root: str | os.PathLike[str],
    *,
    expected_version: str,
    transaction_envelope: bool = False,
) -> EvaluatorRelease:
    """Verify completion, inventory, transaction, semantic, and role contracts."""

    root_path = Path(root)
    if not root_path.is_dir() or root_path.is_symlink():
        raise EvaluatorReleaseError(f"evaluator release directory is missing: {root_path}")
    manifest_path = root_path / MANIFEST_NAME
    seal_path = root_path / COMPLETION_SEAL_NAME
    if not manifest_path.is_file() or not seal_path.is_file():
        raise EvaluatorReleaseError("completion seal or evaluator manifest is missing")
    manifest = _read_json(manifest_path, "evaluator manifest")
    seal = _read_json(seal_path, "evaluator completion seal")
    expected_version = _validate_release_version(expected_version)
    if manifest.get("dataset_id") != APPROVED_DATASET_ID:
        raise EvaluatorReleaseError("evaluator dataset ID is not approved")
    if manifest.get("version") != expected_version:
        raise EvaluatorReleaseError("evaluator release version does not match caller")
    if (
        set(manifest) != MANIFEST_KEYS
        or set(seal) != COMPLETION_KEYS
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("profile") != PROFILE_NAME
        or manifest.get("families") != list(FAMILIES)
        or seal.get("schema_version") != COMPLETION_SCHEMA
        or seal.get("profile") != PROFILE_NAME
        or seal.get("status") != "complete"
    ):
        raise EvaluatorReleaseError("evaluator manifest/completion contract is invalid")
    manifest_root = _manifest_root(manifest, "evaluator release")
    for field in ("dataset_id", "generation_id", "profile", "version"):
        if seal.get(field) != manifest.get(field):
            raise EvaluatorReleaseError(f"completion seal {field} mismatch")
    if (
        seal.get("manifest_root_sha256") != manifest_root
        or seal.get("manifest_file_sha256") != sha256_file(manifest_path)
        or seal.get("manifest_path") != MANIFEST_NAME
        or seal.get("inventory_root_sha256") != manifest.get("inventory_root_sha256")
    ):
        raise EvaluatorReleaseError("completion seal does not bind the evaluator manifest")
    generation_id = manifest.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise EvaluatorReleaseError("evaluator generation ID is missing")
    entries = _inventory_by_path(manifest)
    if seal.get("inventory_entries") != len(entries):
        raise EvaluatorReleaseError("completion inventory count mismatch")
    actual_files = set()
    for path in root_path.rglob("*"):
        if path.is_symlink():
            raise EvaluatorReleaseError(f"release contains a forbidden symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root_path).as_posix())
    expected_files = set(entries) | {MANIFEST_NAME, COMPLETION_SEAL_NAME}
    compared_files = (
        {path for path in actual_files if path.startswith("evaluator/")}
        if transaction_envelope
        else actual_files
    )
    if compared_files != expected_files:
        raise EvaluatorReleaseError(
            f"exact inventory mismatch: missing={sorted(expected_files - compared_files)}, "
            f"extra={sorted(compared_files - expected_files)}"
        )
    expected_bindings = {
        "corpus_logical_root_sha256": manifest["provenance"]["corpus_generation"][
            "logical_root_sha256"
        ],
        "corpus_transaction_inventory_sha256": manifest["provenance"][
            "corpus_generation"
        ]["transaction_inventory_sha256"],
        "semantic_holdout_root_sha256": manifest["provenance"]["semantic_holdout"][
            "manifest_root_sha256"
        ],
        "source_manifests_root_sha256": manifest["provenance"]["source_manifests"][
            "root_sha256"
        ],
        "tokenizer_seal_sha256": manifest["provenance"]["tokenizer"]["seal_sha256"],
    }
    for path, entry in entries.items():
        if entry.get("bindings") != expected_bindings:
            raise EvaluatorReleaseError(f"{path}: root bindings are not exact")
        _verify_inventory_file(root_path, path, entry, generation_id)
    _verify_roles(manifest, entries)
    _verify_canonical_inventory(manifest, entries)
    _verify_packaged_transaction(root_path, manifest, entries)
    _verify_packaged_semantic(root_path, manifest)
    _verify_internal_file_contracts(root_path, entries)

    roles = manifest["roles"]
    eval_paths = _role_map(roles, "eval", FAMILIES)
    for family, logical in eval_paths.items():
        _validate_family_rows(root_path / logical, family, evaluation=True)
    visibility_paths = _role_map(roles, "train_visibility", FAMILIES)
    for family, logical in visibility_paths.items():
        _verify_visibility_file(
            root_path / logical,
            schema=TRAIN_VISIBILITY_SCHEMA,
            family=family,
        )
    _verify_visibility_file(
        root_path / roles["corpus_union_visibility"],
        schema=CORPUS_UNION_VISIBILITY_SCHEMA,
        family=None,
    )
    heldout_paths = _role_map(roles, "heldout", HELDOUT_NAMES)
    source_paths = _role_map(roles, "source_manifests", SOURCE_MANIFEST_NAMES)
    drop_paths = _role_map(roles, "drop_ledgers", FAMILIES)
    cohort_paths = _role_map(roles, "cohort_ledgers", FAMILIES)
    verifier_paths = _role_map(roles, "verifier_manifests")
    oracle_paths = _role_map(roles, "oracle_manifests")
    sidecar_paths = _role_map(roles, "semantic_sidecars", SEMANTIC_SIDECARS)
    class_paths = _role_map(roles, "class_manifest_by_family", FAMILIES)
    return EvaluatorRelease(
        root=root_path,
        manifest_path=manifest_path,
        completion_seal_path=seal_path,
        manifest=manifest,
        seal=seal,
        manifest_root_sha256=manifest_root,
        seal_sha256=sha256_file(seal_path),
        family_paths={family: root_path / path for family, path in eval_paths.items()},
        heldout_paths={name: root_path / path for name, path in heldout_paths.items()},
        class_manifest_paths={
            family: root_path / path for family, path in class_paths.items()
        },
        semantic_manifest_path=root_path / heldout_paths["mml"],
        semantic_projection_paths={
            family: root_path / heldout_paths[family] for family in MML_PROJECTIONS
        },
        semantic_sidecar_paths={
            name: root_path / path for name, path in sidecar_paths.items()
        },
        train_visibility_paths={
            family: root_path / path for family, path in visibility_paths.items()
        },
        corpus_union_visibility_path=root_path / roles["corpus_union_visibility"],
        source_manifest_paths={
            name: root_path / path for name, path in source_paths.items()
        },
        drop_ledger_paths={
            family: root_path / path for family, path in drop_paths.items()
        },
        cohort_ledger_paths={
            family: root_path / path for family, path in cohort_paths.items()
        },
        tokenizer_seal_path=root_path / roles["tokenizer_seal"],
        verifier_manifest_paths={
            name: root_path / path for name, path in verifier_paths.items()
        },
        oracle_manifest_paths={
            name: root_path / path for name, path in oracle_paths.items()
        },
        provenance=manifest["provenance"],
    )


def dependency_template(release: EvaluatorRelease) -> dict[str, str]:
    """Return an honest, unusable pre-publication dependency template."""

    return {
        "dataset_id": str(release.manifest["dataset_id"]),
        "evaluator_manifest_root_sha256": release.manifest_root_sha256,
        "evaluator_seal_sha256": release.seal_sha256,
        "manifest_sha256": UNPUBLISHED_PLATFORM_MANIFEST_SHA256,
        "role": "evaluator",
        "version": UNPUBLISHED_VERSION,
    }


def bind_published_dependency(
    release: EvaluatorRelease,
    *,
    version: str,
    platform_group_manifest_sha256: str,
) -> dict[str, str]:
    """Bind the exact promoted platform group and local evaluator seals."""

    if not isinstance(version, str) or _VERSION_RE.fullmatch(version) is None:
        raise EvaluatorReleaseError(f"published version must be explicit vN: {version!r}")
    if release.manifest.get("version") != version:
        raise EvaluatorReleaseError(
            "published dependency version does not match the loaded evaluator release"
        )
    _require_sha256(
        platform_group_manifest_sha256,
        "platform group manifest SHA-256",
    )
    return {
        "dataset_id": str(release.manifest["dataset_id"]),
        "evaluator_manifest_root_sha256": release.manifest_root_sha256,
        "evaluator_seal_sha256": release.seal_sha256,
        "manifest_sha256": platform_group_manifest_sha256,
        "role": "evaluator",
        "version": version,
    }


def require_token_dependency(
    dependency: Mapping[str, Any] | None,
    release: EvaluatorRelease,
    *,
    expected_dataset_id: str,
    expected_version: str,
    expected_platform_group_manifest_sha256: str,
    expected_evaluator_manifest_root_sha256: str,
    expected_evaluator_seal_sha256: str,
) -> dict[str, str]:
    """Require all caller-owned evaluator dependency pins with no defaults."""

    if dependency is None:
        raise EvaluatorReleaseError("evaluator dependency is required for token release")
    if not isinstance(dependency, Mapping):
        raise EvaluatorReleaseError("evaluator dependency must be an object")
    _require_sha256(
        expected_platform_group_manifest_sha256,
        "expected platform group manifest SHA-256",
    )
    _require_sha256(
        expected_evaluator_manifest_root_sha256,
        "expected evaluator manifest root",
    )
    _require_sha256(expected_evaluator_seal_sha256, "expected evaluator seal")
    if expected_dataset_id != APPROVED_DATASET_ID:
        raise EvaluatorReleaseError("expected evaluator dataset ID is not approved")
    if _VERSION_RE.fullmatch(expected_version) is None:
        raise EvaluatorReleaseError("expected evaluator version must be explicit vN")
    expected = {
        "dataset_id": expected_dataset_id,
        "evaluator_manifest_root_sha256": expected_evaluator_manifest_root_sha256,
        "evaluator_seal_sha256": expected_evaluator_seal_sha256,
        "manifest_sha256": expected_platform_group_manifest_sha256,
        "role": "evaluator",
        "version": expected_version,
    }
    if set(dependency) != set(expected):
        raise EvaluatorReleaseError("evaluator dependency fields are not exact")
    for field, value in expected.items():
        if dependency.get(field) != value:
            raise EvaluatorReleaseError(
                f"evaluator dependency drift for {field}: "
                f"expected {value!r}, got {dependency.get(field)!r}"
            )
    if (
        expected_dataset_id != release.manifest["dataset_id"]
        or expected_version != release.manifest["version"]
        or expected_evaluator_manifest_root_sha256 != release.manifest_root_sha256
        or expected_evaluator_seal_sha256 != release.seal_sha256
    ):
        raise EvaluatorReleaseError(
            "caller dependency pins do not identify the loaded evaluator release"
        )
    return {field: str(dependency[field]) for field in expected}


__all__ = [
    "APPROVED_DATASET_ID",
    "CANONICAL_PATH_CONTRACT",
    "COHORT_LEDGER_SCHEMA",
    "COMPLETION_SEAL_NAME",
    "CORPUS_UNION_VISIBILITY_SCHEMA",
    "DROP_LEDGER_SCHEMA",
    "EXPECTED_FAMILY_SOURCE_MANIFESTS",
    "FAMILIES",
    "FAMILY_ROW_SCHEMAS",
    "MANIFEST_NAME",
    "PLATFORM_PROFILE_REGISTERED",
    "PROFILE_NAME",
    "SEMANTIC_TRANSACTION_SCHEMAS",
    "SOURCE_MANIFEST_NAMES",
    "SOURCE_MANIFEST_SCHEMAS",
    "TOKENIZER_SEAL_SCHEMA",
    "TRAIN_VISIBILITY_SCHEMA",
    "UNPUBLISHED_PLATFORM_MANIFEST_SHA256",
    "UNPUBLISHED_VERSION",
    "VERIFIER_MANIFEST_SCHEMAS",
    "EvaluatorRelease",
    "EvaluatorReleaseCoordinator",
    "EvaluatorReleaseError",
    "EvaluatorReleaseSpec",
    "GenerationBinding",
    "InputArtifact",
    "PlannedPayload",
    "ReleasePlan",
    "bind_published_dependency",
    "build_release",
    "canonical_json_bytes",
    "canonical_sha256",
    "completion_json_schema",
    "dependency_template",
    "generation_binding_from_transaction",
    "load_release",
    "manifest_json_schema",
    "materialize_release",
    "named_artifact_root",
    "plan_release",
    "require_token_dependency",
    "sha256_file",
    "write_json_schemas",
]
