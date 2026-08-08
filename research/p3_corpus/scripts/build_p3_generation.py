"""Build one immutable, occurrence-routed P3 corpus generation.

Production invocation (fail-closed on unfinished technical roots/contracts)::

    python scripts/build_p3_generation.py --corpus-root /tmp/p3-generation-root --work-root /tmp/p3-generation-work --generation-id <ID> --tokenizer-seal <TOKENIZER-SEAL.json> --tokenizer-path <VENDORED-TOKENIZER> --metamath-drop-ledger <METAMATH-OVERLENGTH.json> --policies <POLICIES.json> --source-manifest metamath=<METAMATH.json> --source-manifest mizar=<MIZAR.json> --source-manifest thproofs=<THPROOFS.json> --source-manifest prf2=<PRF2.json> --source-manifest enigma=<ENIGMA.json> --source-manifest isabelle=<ISABELLE.json> --mizar-semantic-index <MIZAR.sqlite> --mml-contract-root <PERSISTED-MML-V7>

Builders never receive a path in the transaction root. They write only beneath
an external trusted work directory; this module then streams exact bytes through
the descriptor-safe transaction writer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

try:  # Support both ``python scripts/...`` and ``from scripts import ...``.
    from . import build_metamath_shard as metamath_builder
    from . import split_mml_semantic_holdout as mml_holdout
    from .build_atp_shard import (
        ProofStep,
        is_refutation_formula,
        render_target,
        source_dependencies,
    )
    from .corpus_generation_transaction import (
        DropRecord,
        GenerationCoordinator,
        GenerationPlan,
        JsonlValidator,
        JsonObjectValidator,
        OutputRole,
        OutputSpec,
        PublishedGeneration,
    )
except ImportError:  # pragma: no cover - exercised by the production CLI.
    import build_metamath_shard as metamath_builder
    import split_mml_semantic_holdout as mml_holdout
    from build_atp_shard import (
        ProofStep,
        is_refutation_formula,
        render_target,
        source_dependencies,
    )
    from corpus_generation_transaction import (
        DropRecord,
        GenerationCoordinator,
        GenerationPlan,
        JsonlValidator,
        JsonObjectValidator,
        OutputRole,
        OutputSpec,
        PublishedGeneration,
    )


EXACT_SIBLINGS = (
    "metamath",
    "mizar",
    "thproofs",
    "prf2",
    "enigma",
    "isabelle",
)
MML_SIBLINGS = ("mizar", "thproofs", "prf2", "enigma")
ROW_SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}
DROP_SCHEMA = "p3-typed-drop/v2"
SOURCE_LINK_SCHEMA = "p3-source-manifest-link/v2"
TOKENIZER_LINK_SCHEMA = "p3-tokenizer-link/v2"
POLICY_LINK_SCHEMA = "p3-policy-link/v2"
SCHEMA_LINK_SCHEMA = "p3-schema-roots-link/v2"
OCCURRENCE_LINK_SCHEMA = "p3-source-occurrences-link/v2"
PRECHECK_LINK_SCHEMA = "p3-deep-precheck-link/v2"
MML_HELDOUT_LINK_SCHEMA = "p3-mml-heldout-link/v2"
FAMILY_HELDOUT_LINK_SCHEMA = "p3-family-heldout-link/v2"
SOURCE_MANIFEST_SCHEMA = "p3-family-source-manifest/v2"
POLICY_SCHEMA = "p3-generation-policies/v2"
BUILDER_QUARANTINE_SCHEMA = "p3-builder-quarantine/v1"
PREFLIGHT_SCHEMA = "p3-generation-preflight/v2"
ENIGMA_LOW_TIER_INPUT_BINDING_SCHEMA = "p3-enigma-low-tier-input-binding/v1"
METAMATH_HELDOUT_SCHEMA = "metamath-heldout-v2"
ISABELLE_HELDOUT_SCHEMA = "isabelle-transition-v2"
METAMATH_HELDOUT_FIELDS = (
    "facts",
    "family",
    "local_assumptions",
    "mode",
    "requested_heldout",
)
ISABELLE_HELDOUT_FIELDS = (
    "facts",
    "family",
    "mode",
    "requested_heldout",
    "statements",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TEMPORARY_NAME_RE = re.compile(
    r"(?:^\.|~$|\.tmp$|\.temp$|\.partial$|\.swp$|\.bak$)",
    re.IGNORECASE,
)
BYPASS_TOKEN_RE = re.compile(
    r"(?:^|[-_/])(?:test(?:-only)?|allow-unsealed|bypass(?:-[a-z0-9]+)*|"
    r"skip-check|legacy-production)(?:$|[-_/])",
    re.IGNORECASE,
)
NO_EXCLUSION_PATH = os.devnull

PRODUCTION_BUILDER_SCRIPTS = {
    "metamath": "build_metamath_shard.py",
    "mizar": "build_mizar_human_shard.py",
    "thproofs": "build_thproofs_shard.py",
    "prf2": "build_atp_shard.py",
    "enigma": "build_atp_shard.py",
    "isabelle": "build_isabelle_shard.py",
}

PRODUCTION_BUILDER_OPTIONS = {
    "metamath": {
        "--mm-dir": "one",
        "--heldout": "one",
        "--seed": "one",
        "--tokenizer": "one",
    },
    "mizar": {
        "--mml-root": "one",
        "--html-root": "one",
        "--thproofs-root": "one",
        "--semantic-index": "one",
        "--semantic-index-sha256": "one",
        "--source-manifest": "one",
        "--mizar-archive": "one",
        "--html-archive": "one",
        "--thproofs-archive": "one",
        "--tokenizer-path": "one",
        "--name": "one",
        "--heldout": "one",
        "--seed": "one",
    },
    "thproofs": {
        "--src": "one",
        "--semantic-index": "one",
        "--source-manifest": "one",
        "--mml-root": "one",
        "--html-root": "one",
        "--mizar-archive": "one",
        "--html-archive": "one",
        "--thproofs-archive": "one",
        "--exclude": "one",
        "--name": "one",
        "--heldout": "one",
        "--seed": "one",
    },
    "prf2": {
        "--src": "many",
        "--name": "one",
        "--fenced": "zero",
        "--heldout": "one",
        "--min-steps": "one",
        "--dedup": "zero",
        "--jaccard": "one",
        "--seed": "one",
    },
    "enigma": {
        "--src": "many",
        "--name": "one",
        "--fenced": "zero",
        "--heldout": "one",
        "--min-steps": "one",
        "--dedup": "zero",
        "--jaccard": "one",
        "--seed": "one",
        "--enigma-low-tier-base": "one",
        "--tokenizer-json": "one",
    },
    "isabelle": {
        "--src": "one",
        "--name": "one",
        "--heldout": "one",
        "--seed": "one",
        "--tokenizer-path": "one",
    },
}

COMMON_REQUIRED_FIELDS = (
    "cited",
    "facts",
    "goal",
    "id",
    "mask_end",
    "mask_start",
    "schema_version",
    "source_metadata",
    "target",
    "text",
    "theorem",
)
DROP_TYPES = {
    "metamath": (
        "heldout_citation",
        "heldout_own_proof",
        "overlength",
        "duplicate",
        "incomplete",
    ),
    "mizar": (
        "class_member_disagreement",
        "exact_atp_duplicate",
        "overlength",
        "statement_disagreement",
    ),
    "thproofs": (
        "class_member_disagreement",
        "direct_mizar_trajectory_duplicate",
        "exact_atp_duplicate",
        "overlength",
        "statement_disagreement",
    ),
    "prf2": (
        "class_member_disagreement",
        "exact_atp_duplicate",
        "overlength",
        "statement_disagreement",
    ),
    "enigma": (
        "class_member_disagreement",
        "exact_atp_duplicate",
        "overlength",
        "statement_disagreement",
    ),
    "isabelle": (
        "heldout_local_statement",
        "heldout_own_proof",
        "heldout_target_state",
        "heldout_trajectory_sibling",
    ),
}


class IntegrationError(RuntimeError):
    """A production-integration gate failed before publication."""


def _trim_unused_heap() -> None:
    """Return large temporary validation buffers to glibc when available."""

    gc.collect()
    try:
        import ctypes

        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


@dataclass(frozen=True)
class _LineOccurrence:
    line_number: int
    byte_start: int
    byte_end: int
    raw_bytes: bytes
    raw_sha256: str
    record: dict[str, Any]


@dataclass(frozen=True)
class _NativeDrop:
    raw_row: int
    drop_type: str
    details: Mapping[str, Any]


@dataclass(frozen=True)
class _MetamathIsolationContext:
    held_fact_names: frozenset[str]
    held_statement_identities: frozenset[str]
    held_fact_names_by_statement: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _MetamathRouteClassification:
    disposition: str
    drop_type: str | None
    drop_details: Mapping[str, Any]
    detail: str


@dataclass(frozen=True)
class _FamilyPackage:
    family: str
    raw: Path
    train: Path
    eval: Path
    drops: tuple[_NativeDrop, ...]
    heldout: Mapping[str, Any]
    source_accounting: Mapping[str, Any] | None = None
    overlength_drop_ledger: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class SyntheticBuildResult:
    """Result and external builder roots for the deterministic dry fixture."""

    published: PublishedGeneration
    builder_output_roots: tuple[Path, ...]


@dataclass(frozen=True)
class ProductionBuildResult:
    """Published production generation and its external builder roots."""

    published: PublishedGeneration
    builder_output_roots: tuple[Path, ...]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IntegrationError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise IntegrationError(f"{label} must be a JSON object: {path}")
    return value


def _require_json_contract(
    value: Mapping[str, Any],
    *,
    schema: str,
    required_fields: Sequence[str],
    label: str,
) -> None:
    if value.get("schema_version") != schema:
        raise IntegrationError(
            f"{label} internal schema_version must be exactly {schema!r}"
        )
    missing = sorted(field for field in required_fields if field not in value)
    if missing:
        raise IntegrationError(f"{label} is missing required contract fields: {missing}")


def _validate_family_heldout_contract(
    value: Mapping[str, Any],
    *,
    family: str,
) -> None:
    if family == "metamath":
        schema = METAMATH_HELDOUT_SCHEMA
        fields = METAMATH_HELDOUT_FIELDS
    elif family == "isabelle":
        schema = ISABELLE_HELDOUT_SCHEMA
        fields = ISABELLE_HELDOUT_FIELDS
    else:
        raise IntegrationError(f"{family}: no family-local heldout contract")
    _require_json_contract(
        value,
        schema=schema,
        required_fields=fields,
        label=f"{family.capitalize()} heldout contract",
    )
    if value.get("family") != family or value.get("mode") != "family_local_heldout":
        raise IntegrationError(f"{family.capitalize()} heldout family/mode is invalid")


def _require_digest(value: Any, label: str, *, production: bool = False) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise IntegrationError(f"{label} must be a SHA-256 digest")
    if production and len(set(value)) == 1:
        raise IntegrationError(f"{label} is an unfinished placeholder digest")
    return value


def _iter_digest_fields(
    value: Any,
    *,
    prefix: str,
) -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(key, str) and key.endswith("sha256"):
                yield field, _require_digest(item, field)
            else:
                yield from _iter_digest_fields(item, prefix=field)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_digest_fields(item, prefix=f"{prefix}[{index}]")


def _validate_status_metadata(
    value: Any,
    *,
    label: str,
    decision_field: str,
) -> None:
    if not isinstance(value, Mapping):
        raise IntegrationError(f"{label} must be an object")
    decision = value.get(decision_field)
    status = value.get("status")
    if type(decision) is not bool or not isinstance(status, str) or not status.strip():
        raise IntegrationError(
            f"{label} must preserve boolean {decision_field!r} and honest status text"
        )
    identifier = value.get("identifier")
    if identifier is not None and (
        not isinstance(identifier, str) or not identifier.strip()
    ):
        raise IntegrationError(f"{label} identifier must be nonempty text when present")


def _source_manifest_root(manifest: Mapping[str, Any]) -> str:
    def without_recursive_roots(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_recursive_roots(item)
                for key, item in value.items()
                if key not in {
                    "manifest_root_sha256",
                    "source_manifest_root_sha256",
                }
            }
        if isinstance(value, list):
            return [without_recursive_roots(item) for item in value]
        return value

    return _canonical_sha256(without_recursive_roots(manifest))


def _make_source_manifest(family: str, *, test_only: bool) -> dict[str, Any]:
    if family not in EXACT_SIBLINGS:
        raise IntegrationError(f"unknown family {family!r}")
    source_digest = hashlib.sha256(f"synthetic-source:{family}".encode()).hexdigest()
    quality_digest = hashlib.sha256(f"synthetic-quality:{family}".encode()).hexdigest()
    schema_digest = hashlib.sha256(
        f"synthetic-schema:{family}:{ROW_SCHEMAS[family]}".encode()
    ).hexdigest()
    index_roots = (
        {
            "semantic_index_schema": "mizar-semantic-index-v1",
            "semantic_index_sha256": hashlib.sha256(
                b"synthetic-current-mizar-index"
            ).hexdigest(),
        }
        if family in {"mizar", "thproofs"}
        else {}
    )
    metadata = {
        "schema_version": f"{family}-build-source-v2",
        "source_roots": {
            "synthetic": {
                "reference": f"synthetic://{family}",
                "sha256": source_digest,
            }
        },
        "index_roots": index_roots,
        "quality_filter_root_sha256": quality_digest,
        "schema_generation_root_sha256": schema_digest,
    }
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "family": family,
        "row_schema_version": ROW_SCHEMAS[family],
        "row_source_metadata": metadata,
        "source_snapshots": [
            {
                "reference": f"synthetic://{family}",
                "sha256": source_digest,
            }
        ],
        "builder": {
            "driver": "synthetic-six-family-v2",
            "partition_mode": (
                "pooled-mml-1000-v1"
                if family in MML_SIBLINGS
                else "family-local-heldout-v2"
            ),
        },
        "license": {
            "approved": False,
            "identifier": "synthetic-test-only",
            "status": "test fixture; not redistributable production data",
        },
        "source_verifier_acceptance": {
            "accepted": family != "metamath",
            "status": "synthetic-test-only",
        },
        "test_only": test_only,
    }
    root = _source_manifest_root(manifest)
    metadata["source_manifest_root_sha256"] = root
    manifest["manifest_root_sha256"] = root
    return manifest


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    family: str,
    production: bool,
) -> dict[str, Any]:
    expected_fields = {
        "builder",
        "family",
        "license",
        "manifest_root_sha256",
        "row_schema_version",
        "row_source_metadata",
        "schema_version",
        "source_snapshots",
        "source_verifier_acceptance",
        "test_only",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_fields:
        raise IntegrationError(f"{family}: source manifest fields are not exact")
    result = json.loads(json.dumps(manifest))
    if result["schema_version"] != SOURCE_MANIFEST_SCHEMA:
        raise IntegrationError(f"{family}: source manifest schema is not v2")
    if result["family"] != family:
        raise IntegrationError(f"{family}: source manifest family mismatch")
    if result["row_schema_version"] != ROW_SCHEMAS[family]:
        raise IntegrationError(f"{family}: row schema policy mismatch")
    actual_root = _source_manifest_root(result)
    if result["manifest_root_sha256"] != actual_root:
        raise IntegrationError(f"{family}: source-manifest root is invalid")
    metadata = result["row_source_metadata"]
    required_metadata = {
        "index_roots",
        "quality_filter_root_sha256",
        "schema_generation_root_sha256",
        "schema_version",
        "source_manifest_root_sha256",
        "source_roots",
    }
    if not isinstance(metadata, dict) or not required_metadata <= set(metadata):
        raise IntegrationError(f"{family}: row source metadata fields are not exact")
    _require_digest(
        metadata["source_manifest_root_sha256"],
        f"{family} source-manifest root",
        production=production,
    )
    if not isinstance(metadata["source_roots"], dict) or not metadata["source_roots"]:
        raise IntegrationError(f"{family}: source roots are missing")
    if not isinstance(metadata["index_roots"], dict):
        raise IntegrationError(f"{family}: index roots must be an object")
    if family in {"mizar", "thproofs"}:
        if set(metadata["index_roots"]) != {
            "semantic_index_schema",
            "semantic_index_sha256",
        }:
            raise IntegrationError(f"{family}: current Mizar index roots are missing")
        if (
            metadata["index_roots"]["semantic_index_schema"]
            != "mizar-semantic-index-v1"
        ):
            raise IntegrationError(f"{family}: current Mizar index schema drift")
        _require_digest(
            metadata["index_roots"]["semantic_index_sha256"],
            f"{family} semantic index root",
            production=production,
        )
    elif metadata["index_roots"]:
        raise IntegrationError(f"{family}: unexpected index roots")
    _require_digest(
        metadata["quality_filter_root_sha256"],
        f"{family} quality-filter root",
        production=production,
    )
    _require_digest(
        metadata["schema_generation_root_sha256"],
        f"{family} schema-generation root",
        production=production,
    )
    snapshots = result["source_snapshots"]
    if not isinstance(snapshots, list) or not snapshots:
        raise IntegrationError(f"{family}: exact source snapshots are missing")
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, dict) or not snapshot:
            raise IntegrationError(f"{family}: malformed source snapshot")
        reference = snapshot.get("reference")
        if reference is not None and (
            not isinstance(reference, str) or not reference.strip()
        ):
            raise IntegrationError(f"{family}: unnamed source snapshot")
        digests = list(
            _iter_digest_fields(
                snapshot,
                prefix=f"{family} source snapshot {index}",
            )
        )
        if not digests:
            raise IntegrationError(
                f"{family}: source snapshot lacks a native hash or tree root"
            )
        for label, digest in digests:
            _require_digest(digest, label, production=production)
    _validate_status_metadata(
        result["license"],
        label=f"{family} source license metadata",
        decision_field="approved",
    )
    _validate_status_metadata(
        result["source_verifier_acceptance"],
        label=f"{family} source verifier metadata",
        decision_field="accepted",
    )
    if production and result["test_only"] is not False:
        raise IntegrationError(f"{family}: production refuses test-only sources")
    return result


def _synthetic_policies() -> dict[str, Any]:
    body = {
        "schema_version": POLICY_SCHEMA,
        "families": list(EXACT_SIBLINGS),
        "mml": {
            "classes": 1_000,
            "seed": 20_260_801,
            "mode": "pooled_semantic_1000",
            "policy_pins": {
                "policy_sha256": mml_holdout.current_policy_pins().policy_sha256,
                "mapping_sha256": mml_holdout.current_policy_pins().mapping_sha256,
                "atp_deduplication_sha256": (
                    mml_holdout.current_policy_pins().atp_deduplication_sha256
                ),
            },
        },
        "metamath": {
            "heldout_facts": 500,
            "local_assumptions": True,
            "mode": "family_local_heldout",
        },
        "isabelle": {
            "heldout_facts": 500,
            "row_schema": "isabelle-transition-v2",
            "trajectory_drops": True,
            "mode": "family_local_heldout",
        },
        "test_only": True,
    }
    return {**body, "policy_root_sha256": _canonical_sha256(body)}


def _validate_policies(policies: Mapping[str, Any], *, production: bool) -> dict[str, Any]:
    expected = {
        "families",
        "isabelle",
        "metamath",
        "mml",
        "policy_root_sha256",
        "schema_version",
        "test_only",
    }
    if not isinstance(policies, Mapping) or set(policies) != expected:
        raise IntegrationError("generation policy manifest fields are not exact")
    result = json.loads(json.dumps(policies))
    body = dict(result)
    root = body.pop("policy_root_sha256")
    if root != _canonical_sha256(body):
        raise IntegrationError("generation policy root is invalid")
    if result["schema_version"] != POLICY_SCHEMA:
        raise IntegrationError("generation policy schema is not v2")
    if result["families"] != list(EXACT_SIBLINGS):
        raise IntegrationError("generation policies do not name exact six families")
    if (
        result["mml"].get("classes") != 1_000
        or result["mml"].get("seed") != 20_260_801
        or result["mml"].get("mode") != "pooled_semantic_1000"
    ):
        raise IntegrationError("MML policy is not the approved pooled 1,000-class route")
    if result["metamath"].get("heldout_facts") != 500:
        raise IntegrationError("Metamath policy must hold exactly 500 facts")
    if (
        result["isabelle"].get("heldout_facts") != 500
        or result["isabelle"].get("row_schema") != "isabelle-transition-v2"
        or result["isabelle"].get("trajectory_drops") is not True
    ):
        raise IntegrationError("Isabelle policy is not the accepted 500-fact trajectory split")
    if production and result["test_only"] is not False:
        raise IntegrationError("production refuses test-only generation policies")
    return result


def _validate_tokenizer_seal(seal: Mapping[str, Any]) -> dict[str, Any]:
    expected = mml_holdout.approved_tokenizer_seal()
    if not isinstance(seal, Mapping) or dict(seal) != expected:
        raise IntegrationError("tokenizer seal is not the approved exact Qwen seal")
    return dict(expected)


def _validate_metamath_drop_ledger(
    drop_ledger: Mapping[str, Any],
    *,
    tokenizer_seal: Mapping[str, Any],
) -> dict[str, Any]:
    tokenizer = _validate_tokenizer_seal(tokenizer_seal)
    if not isinstance(drop_ledger, Mapping):
        raise IntegrationError("Metamath overlength drop ledger is missing")
    ledger = _json_copy(drop_ledger)
    try:
        metamath_builder.validate_drop_ledger(ledger)
    except (TypeError, ValueError) as error:
        raise IntegrationError(
            f"Metamath overlength drop ledger validation failed: {error}"
        ) from error
    if ledger.get("tokenizer_seal") != tokenizer:
        raise IntegrationError("Metamath drop ledger tokenizer seal is stale")
    return ledger


def _generic_record(
    *,
    family: str,
    row_id: str,
    theorem: str,
    facts: Mapping[str, str],
    cited: Sequence[str],
    goal: str,
    target: str,
    source_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    return {
        "schema_version": ROW_SCHEMAS[family],
        "id": row_id,
        "theorem": theorem,
        "facts": dict(facts),
        "cited": list(cited),
        "goal": goal,
        "target": target,
        "text": f"{block}\n---\nGOAL {goal}\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
        "source_metadata": _json_copy(source_metadata),
    }


def _atp_record(
    *,
    family: str,
    row_id: str,
    theorem: str,
    fact_name: str,
    fact_statement: str,
    goal: str,
    source_metadata: Mapping[str, Any],
    marker: str = "",
) -> dict[str, Any]:
    goal_name = f"goal_{row_id.replace('-', '_')}"
    step = ProofStep(
        name=f"step_{row_id.replace('-', '_')}",
        role="plain",
        formula="$false",
        rule="resolution",
        parents=[fact_name, goal_name],
        parent_sources=[fact_name, goal_name],
        source=f"inference(resolution,[status(thm)],[{fact_name},{goal_name}])",
    )
    target = render_target([step])
    block = f"I know these mathematical statements:\n{fact_name} : {fact_statement}"
    text = f"{block}\n---\nGOAL {goal}{marker}\n{target}"
    return {
        "schema_version": "atp-v2",
        "id": row_id,
        "theorem": theorem,
        "facts": {fact_name: fact_statement},
        "cited": [fact_name],
        "local_inputs": {},
        "goal_name": goal_name,
        "goal": f"{goal}{marker}",
        "proof_steps": [
            {
                "name": step.name,
                "role": step.role,
                "formula": step.formula,
                "rule": step.rule,
                "parents": step.parents,
                "parent_sources": step.parent_sources,
                "source": step.source,
            }
        ],
        "target": target,
        "text": text,
        "mask_start": 0,
        "mask_end": len(block),
        "source_metadata": _json_copy(source_metadata),
    }


def _metamath_record(
    row_id: str,
    source_metadata: Mapping[str, Any],
    *,
    fact_name: str = "mp",
    fact_statement: str = "|- ph => |- ph",
    theorem: str | None = None,
) -> dict[str, Any]:
    facts = {fact_name: fact_statement}
    local = {"fixture.1": "|- ph"}
    target = f"  1  {fact_name:<14} |- ph"
    block = (
        "I know these mathematical statements:\n"
        f"{fact_name} : {fact_statement}\n"
        "Local assumptions:\n"
        "fixture.1 : |- ph"
    )
    return {
        "schema_version": "metamath-proof-v2",
        "id": row_id,
        "theorem": theorem or f"set:{row_id}",
        "facts": facts,
        "cited": [fact_name],
        "local_assumptions": local,
        "goal": "|- ph",
        "target": target,
        "text": f"{block}\n---\nGOAL |- ph\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
        "source_metadata": _json_copy(source_metadata),
    }


def _isabelle_record(
    row_id: str,
    source_metadata: Mapping[str, Any],
    *,
    fact_name: str = "Global.fact",
    fact_statement: str = "global statement",
    trajectory_key: str | None = None,
) -> dict[str, Any]:
    facts = {fact_name: fact_statement}
    aliases = {"g": fact_name}
    state_before = f"state before {row_id}"
    state_after = f"state after {row_id}"
    tactic = "by exact"
    theorem_statement = f"shows synthetic_{row_id}"
    goal = f"THEOREM\n{theorem_statement}\nSTATE_BEFORE\n{state_before}"
    target = f"TACTIC\n{tactic}\nSTATE_AFTER\n{state_after}"
    block = f"I know these mathematical statements:\ng [{fact_name}] : {fact_statement}"
    return {
        "schema_version": "isabelle-transition-v2",
        "id": hashlib.sha256(f"row:{row_id}".encode()).hexdigest(),
        "trajectory_id": hashlib.sha256(
            f"trajectory:{trajectory_key or row_id}".encode()
        ).hexdigest(),
        "transition_index": 0,
        "theorem": f"Synthetic.{row_id}",
        "theory": "Synthetic",
        "theorem_statement": theorem_statement,
        "facts": facts,
        "cited": [fact_name],
        "premise_aliases": aliases,
        "local_assumptions": {},
        "local_names": {},
        "state_before": state_before,
        "tactic": tactic,
        "state_after": state_after,
        "goal": goal,
        "target": target,
        "text": f"{block}\n---\nGOAL\n{goal}\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
        "source_metadata": _json_copy(source_metadata),
    }


def synthetic_family_record(
    family: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one valid current-schema row and its exact synthetic source manifest."""

    manifest = _make_source_manifest(family, test_only=True)
    metadata = manifest["row_source_metadata"]
    if family == "metamath":
        record = _metamath_record("metamath-fixture", metadata)
    elif family in {"mizar", "thproofs"}:
        record = _generic_record(
            family=family,
            row_id=f"{family}-fixture",
            theorem="SAFE:2",
            facts={"SAFE:1": "for x being set holds x = x"},
            cited=("SAFE:1",),
            goal="for x being set holds x = x",
            target="thus thesis by SAFE:1;",
            source_metadata=metadata,
        )
    elif family in {"prf2", "enigma"}:
        record = _atp_record(
            family=family,
            row_id=f"{family}-fixture",
            theorem=f"{family}:t2_safe",
            fact_name="t1_safe",
            fact_statement="p(a)",
            goal="q(a)",
            source_metadata=metadata,
        )
    elif family == "isabelle":
        record = _isabelle_record("isabelle-fixture", metadata)
    else:
        raise IntegrationError(f"unknown family {family!r}")
    return record, manifest


def _require_text_mapping(value: Any, label: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, dict) or (not value and not allow_empty):
        raise IntegrationError(f"{label} must be a {'possibly empty ' if allow_empty else ''}map")
    if not all(
        isinstance(name, str)
        and name
        and isinstance(statement, str)
        and statement.strip()
        for name, statement in value.items()
    ):
        raise IntegrationError(f"{label} contains malformed text")


def _validate_row_source_metadata(
    record: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    location: str,
) -> None:
    actual = record.get("source_metadata")
    expected = source_manifest["row_source_metadata"]
    if not isinstance(actual, dict):
        raise IntegrationError(f"{location}: source metadata is missing")
    if actual.get("source_manifest_root_sha256") != expected.get(
        "source_manifest_root_sha256"
    ):
        raise IntegrationError(f"{location}: source-manifest root mismatch")
    if actual.get("index_roots") != expected.get("index_roots"):
        raise IntegrationError(f"{location}: index root mismatch")
    if actual.get("schema_generation_root_sha256") != expected.get(
        "schema_generation_root_sha256"
    ):
        raise IntegrationError(f"{location}: schema-generation root mismatch")
    if actual.get("quality_filter_root_sha256") != expected.get(
        "quality_filter_root_sha256"
    ):
        raise IntegrationError(f"{location}: quality-filter root mismatch")
    if actual.get("source_roots") != expected.get("source_roots"):
        raise IntegrationError(f"{location}: source root mismatch")
    if actual != expected:
        raise IntegrationError(f"{location}: row source metadata is not exact")


def _validate_atp_record(record: Mapping[str, Any], location: str) -> None:
    local_inputs = record.get("local_inputs")
    _require_text_mapping(local_inputs, f"{location}: local_inputs", allow_empty=True)
    goal_name = record.get("goal_name")
    if not isinstance(goal_name, str) or not goal_name:
        raise IntegrationError(f"{location}: ATP goal_name is missing")
    raw_steps = record.get("proof_steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise IntegrationError(f"{location}: ATP proof steps are missing")
    steps = []
    for index, item in enumerate(raw_steps, 1):
        required = {
            "formula",
            "name",
            "parent_sources",
            "parents",
            "role",
            "rule",
            "source",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise IntegrationError(f"{location}: malformed ATP step {index}")
        parsed = source_dependencies(item["source"])
        if parsed is None:
            raise IntegrationError(f"{location}: ATP step source is not reconstructible")
        rule, parent_sources, parents = parsed
        if (
            rule != item["rule"]
            or parent_sources != item["parent_sources"]
            or parents != item["parents"]
        ):
            raise IntegrationError(f"{location}: ATP parent/reference mismatch")
        steps.append(
            ProofStep(
                name=item["name"],
                role=item["role"],
                formula=item["formula"],
                rule=item["rule"],
                parents=item["parents"],
                parent_sources=item["parent_sources"],
                source=item["source"],
            )
        )
    supplied = set(record["facts"]) | set(local_inputs) | {goal_name}
    seen = set()
    for step in steps:
        for parent in step.parents:
            if parent not in supplied and parent not in seen:
                raise IntegrationError(
                    f"{location}: unresolved ATP parent/reference {parent!r}"
                )
        if step.name in seen:
            raise IntegrationError(f"{location}: duplicate ATP step {step.name!r}")
        seen.add(step.name)
    if not is_refutation_formula(steps[-1].formula):
        raise IntegrationError(f"{location}: ATP target does not end in refutation")
    if render_target(steps) != record["target"]:
        raise IntegrationError(f"{location}: ATP target is not reconstructible")


def validate_family_record(
    record: Mapping[str, Any],
    *,
    family: str,
    source_manifest: Mapping[str, Any],
    location: str,
) -> None:
    """Validate one complete family row, including source roots and rendering."""

    if family not in EXACT_SIBLINGS:
        raise IntegrationError(f"{location}: unknown family {family!r}")
    source_manifest = _validate_source_manifest(
        source_manifest,
        family=family,
        production=False,
    )
    if not isinstance(record, Mapping):
        raise IntegrationError(f"{location}: row is not an object")
    missing = [field for field in COMMON_REQUIRED_FIELDS if field not in record]
    if missing:
        raise IntegrationError(f"{location}: missing row fields {missing}")
    if record["schema_version"] != ROW_SCHEMAS[family]:
        raise IntegrationError(f"{location}: malformed family schema")
    for field in ("id", "theorem", "goal", "target", "text"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise IntegrationError(f"{location}: {field} must be nonempty text")
    _require_text_mapping(record["facts"], f"{location}: facts")
    cited = record["cited"]
    if not isinstance(cited, list) or not all(isinstance(item, str) for item in cited):
        raise IntegrationError(f"{location}: cited must be a string list")
    missing_facts = set(cited) - set(record["facts"])
    if missing_facts:
        raise IntegrationError(
            f"{location}: cited names lack fact statements: {sorted(missing_facts)[:3]}"
        )
    _validate_row_source_metadata(record, source_manifest, location=location)

    if family == "metamath":
        local = record.get("local_assumptions")
        _require_text_mapping(
            local,
            f"{location}: local assumptions",
            allow_empty=True,
        )
        block = "I know these mathematical statements:\n" + "\n".join(
            f"{name} : {statement}" for name, statement in record["facts"].items()
        )
        block += "\nLocal assumptions:"
        if local:
            block += "\n" + "\n".join(
                f"{name} : {statement}" for name, statement in local.items()
            )
        for target_line in record["target"].splitlines():
            match = re.match(r"^\s*\d+\s+(\S+)\s+(.+)$", target_line)
            if match is None:
                raise IntegrationError(f"{location}: malformed Metamath target")
            label = match.group(1)
            if label == "(reuse)" or label in local or label not in record["facts"]:
                raise IntegrationError(
                    f"{location}: target label is not a supplied global fact"
                )
        expected = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    elif family in {"prf2", "enigma"}:
        _validate_atp_record(record, location)
        block = "I know these mathematical statements:\n" + "\n".join(
            f"{name} : {statement}" for name, statement in record["facts"].items()
        )
        if record["local_inputs"]:
            block += "\nLocal ATP inputs:\n" + "\n".join(
                f"{name} : {statement}"
                for name, statement in record["local_inputs"].items()
            )
        expected = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    elif family in {"mizar", "thproofs"}:
        block = "I know these mathematical statements:\n" + "\n".join(
            f"{name} : {statement}" for name, statement in record["facts"].items()
        )
        expected = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    else:
        required = {
            "local_assumptions",
            "local_names",
            "premise_aliases",
            "state_after",
            "state_before",
            "tactic",
            "theorem_statement",
            "trajectory_id",
            "transition_index",
        }
        missing_isabelle = sorted(required - set(record))
        if missing_isabelle:
            raise IntegrationError(
                f"{location}: missing Isabelle transition fields {missing_isabelle}"
            )
        aliases = record["premise_aliases"]
        local = record["local_assumptions"]
        local_names = record["local_names"]
        if not all(isinstance(value, dict) for value in (aliases, local, local_names)):
            raise IntegrationError(f"{location}: malformed Isabelle premise maps")
        lines = ["I know these mathematical statements:"]
        for alias in sorted(aliases):
            qualified = aliases[alias]
            if qualified not in record["facts"]:
                raise IntegrationError(f"{location}: Isabelle alias lacks global fact")
            lines.append(f"{alias} [{qualified}] : {record['facts'][qualified]}")
        if local:
            lines.append("Local assumptions:")
            for alias in sorted(local):
                if alias not in local_names:
                    raise IntegrationError(
                        f"{location}: Isabelle local assumption lacks source name"
                    )
                lines.append(f"{alias} [{local_names[alias]}] : {local[alias]}")
        block = "\n".join(lines)
        goal = (
            f"THEOREM\n{record['theorem_statement']}\n"
            f"STATE_BEFORE\n{record['state_before']}"
        )
        target = (
            f"TACTIC\n{record['tactic']}\n"
            f"STATE_AFTER\n{record['state_after']}"
        )
        if record["goal"] != goal or record["target"] != target:
            raise IntegrationError(
                f"{location}: Isabelle target/state_after is not reconstructible"
            )
        expected = f"{block}\n---\nGOAL\n{goal}\n{target}"
    if (
        record["mask_start"] != 0
        or record["mask_end"] != len(block)
        or record["text"][: record["mask_end"]] != block
    ):
        raise IntegrationError(f"{location}: mask span is not reconstructible")
    if record["text"] != expected:
        raise IntegrationError(f"{location}: text is not reconstructible")


def _read_occurrences(path: Path, *, label: str) -> tuple[_LineOccurrence, ...]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IntegrationError(f"{label}: output is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise IntegrationError(f"{label}: output must be a real regular file")
    occurrences = []
    byte_start = 0
    with path.open("rb") as source:
        for line_number, raw_bytes in enumerate(source, 1):
            if not raw_bytes.endswith(b"\n"):
                raise IntegrationError(
                    f"{label}:{line_number}: row is not newline terminated"
                )
            try:
                record = json.loads(raw_bytes)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise IntegrationError(
                    f"{label}:{line_number}: invalid JSONL"
                ) from error
            if not isinstance(record, dict):
                raise IntegrationError(f"{label}:{line_number}: row is not an object")
            byte_end = byte_start + len(raw_bytes)
            occurrences.append(
                _LineOccurrence(
                    line_number=line_number,
                    byte_start=byte_start,
                    byte_end=byte_end,
                    raw_bytes=raw_bytes,
                    raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                    record=record,
                )
            )
            byte_start = byte_end
    if not occurrences:
        raise IntegrationError(f"{label}: structured output is empty")
    return tuple(occurrences)


def _validate_rows(
    path: Path,
    *,
    family: str,
    source_manifest: Mapping[str, Any],
    label: str,
) -> tuple[_LineOccurrence, ...]:
    occurrences = _read_occurrences(path, label=label)
    for occurrence in occurrences:
        validate_family_record(
            occurrence.record,
            family=family,
            source_manifest=source_manifest,
            location=f"{label}:{occurrence.line_number}",
        )
    return occurrences


def _mizar_row(
    *,
    family: str,
    row_id: str,
    theorem: str,
    facts: Mapping[str, str],
    cited: Sequence[str],
    source_metadata: Mapping[str, Any],
    marker: str = "",
) -> dict[str, Any]:
    target_name = cited[0]
    return _generic_record(
        family=family,
        row_id=row_id,
        theorem=theorem,
        facts=facts,
        cited=cited,
        goal=f"synthetic goal {row_id}{marker}",
        target=f"thus thesis by {target_name};",
        source_metadata=source_metadata,
    )


def _synthetic_mml_rows(
    family: str,
    source_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    metadata = source_manifest["row_source_metadata"]
    safe_mizar = ("SAFE:999", "safe statement")
    safe_atp = ("t999_safe", "safe(statement)")
    if family == "mizar":
        pool_facts = {
            f"ART{index}:1": f"statement {index}" for index in range(1_000)
        }
        rows = [
            _mizar_row(
                family=family,
                row_id="mizar-pool",
                theorem="POOL:1",
                facts=pool_facts,
                cited=list(pool_facts),
                source_metadata=metadata,
            )
        ]
    elif family == "thproofs":
        rows = [
            _mizar_row(
                family=family,
                row_id="thproofs-exposure",
                theorem="THSAFE:1",
                facts={"ART0:1": "statement 0"},
                cited=("ART0:1",),
                source_metadata=metadata,
            )
        ]
    elif family == "prf2":
        rows = [
            _atp_record(
                family=family,
                row_id="prf2-exposure",
                theorem="prf2:t50_prfonly",
                fact_name="t1_art1",
                fact_statement="p1(a)",
                goal="prf_goal(a)",
                source_metadata=metadata,
            )
        ]
    elif family == "enigma":
        rows = [
            _atp_record(
                family=family,
                row_id="enigma-exposure",
                theorem="enigma:t51_enigmaonly",
                fact_name="t1_art2",
                fact_statement="p2(a)",
                goal="enigma_goal(a)",
                source_metadata=metadata,
            )
        ]
    else:
        raise IntegrationError(f"{family}: not an MML sibling")
    for index in range(3):
        if family in {"mizar", "thproofs"}:
            rows.append(
                _mizar_row(
                    family=family,
                    row_id=f"{family}-train-{index}",
                    theorem=f"{'MZSAFE' if family == 'mizar' else 'THSAFE'}:{100 + index}",
                    facts={safe_mizar[0]: safe_mizar[1]},
                    cited=(safe_mizar[0],),
                    source_metadata=metadata,
                )
            )
        else:
            rows.append(
                _atp_record(
                    family=family,
                    row_id=f"{family}-train-{index}",
                    theorem=f"{family}:t{100 + index}_safe",
                    fact_name=safe_atp[0],
                    fact_statement=safe_atp[1],
                    goal=f"safe_goal_{index}(a)",
                    source_metadata=metadata,
                )
            )
    if family in {"mizar", "thproofs"}:
        rows.append(
            _mizar_row(
                family=family,
                row_id=f"{family}-overlength",
                theorem="LONG:1",
                facts={safe_mizar[0]: safe_mizar[1]},
                cited=(safe_mizar[0],),
                source_metadata=metadata,
                marker=" OVERLENGTH",
            )
        )
    else:
        rows.append(
            _atp_record(
                family=family,
                row_id=f"{family}-overlength",
                theorem=f"{family}:t900_long",
                fact_name=safe_atp[0],
                fact_statement=safe_atp[1],
                goal="long_goal(a)",
                source_metadata=metadata,
                marker=" OVERLENGTH",
            )
        )
    return rows


def _make_memory_sources(
    raw_paths: Mapping[str, Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, mml_holdout.MemoryShardSource],
    mml_holdout.SourceIdentityPolicy,
]:
    sources = {}
    approved = {}
    for family in MML_SIBLINGS:
        lines = tuple(_read_occurrences(raw_paths[family], label=family))
        native_lines = tuple(item.raw_bytes for item in lines)
        digest = hashlib.sha256(b"".join(native_lines)).hexdigest()
        manifest = source_manifests[family]
        snapshots = tuple(
            mml_holdout.SourceSnapshot(
                reference=item["reference"],
                sha256=item["sha256"],
            )
            for item in manifest["source_snapshots"]
        )
        metadata = manifest["row_source_metadata"]
        sources[family] = mml_holdout.MemoryShardSource(
            name=family,
            logical_path=f"raw/{family}.jsonl",
            lines=native_lines,
            expected_input_sha256=digest,
            source_snapshots=snapshots,
            source_manifest_root_sha256=metadata[
                "source_manifest_root_sha256"
            ],
            quality_filter_root_sha256=metadata["quality_filter_root_sha256"],
            schema_generation_root_sha256=metadata[
                "schema_generation_root_sha256"
            ],
        )
        approved[family] = mml_holdout.ApprovedShardSource(
            input_sha256=digest,
            source_snapshots=snapshots,
            source_manifest_root_sha256=metadata[
                "source_manifest_root_sha256"
            ],
            quality_filter_root_sha256=metadata["quality_filter_root_sha256"],
            schema_generation_root_sha256=metadata[
                "schema_generation_root_sha256"
            ],
        )
    return (
        sources,
        mml_holdout.SourceIdentityPolicy(
            policy_id="synthetic-six-family-integration-v1",
            shards=approved,
            test_only=True,
        ),
    )


def _build_synthetic_mml_contract(
    raw_paths: Mapping[str, Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> mml_holdout.ValidatedHoldoutContract:
    sources, source_policy = _make_memory_sources(raw_paths, source_manifests)
    tokenizer = mml_holdout.TokenizerSeam(
        seal=mml_holdout.approved_tokenizer_seal(),
        count_text_plus_eos=lambda text: (
            16_385 if "OVERLENGTH" in text else max(1, len(text.split())) + 1
        ),
    )
    plan = mml_holdout.plan_semantic_holdout(
        sources,
        tokenizer=tokenizer,
        policy_pins=mml_holdout.current_policy_pins(),
        source_policy=source_policy,
    )
    mml_holdout.write_partition_atomically(plan, sources=sources, output=output)
    return mml_holdout.load_holdout_contract(output, production=False)


def _build_family_local_package(
    family: str,
    output: Path,
    source_manifest: Mapping[str, Any],
    *,
    duplicate_id: bool,
) -> _FamilyPackage:
    metadata = source_manifest["row_source_metadata"]
    if family == "metamath":
        rows = [
            _metamath_record(
                "metamath-train",
                metadata,
                fact_name="safe",
                fact_statement="|- ps => |- ps",
            ),
            _metamath_record("metamath-eval", metadata),
            _metamath_record(
                "metamath-drop",
                metadata,
                fact_name="safe",
                fact_statement="|- ps => |- ps",
                theorem="set:mp",
            ),
        ]
        heldout = {
            "schema_version": "metamath-heldout-v2",
            "family": family,
            "mode": "family_local_heldout",
            "facts": ["mp"],
            "requested_heldout": 1,
            "local_assumptions": True,
        }
        drop_type = "heldout_own_proof"
    elif family == "isabelle":
        rows = [
            _isabelle_record(
                "isabelle-train",
                metadata,
                fact_name="Safe.fact",
                fact_statement="safe statement",
            ),
            _isabelle_record(
                "isabelle-eval",
                metadata,
                trajectory_key="held-trajectory",
            ),
            _isabelle_record(
                "isabelle-drop",
                metadata,
                fact_name="Safe.fact",
                fact_statement="safe statement",
                trajectory_key="held-trajectory",
            ),
        ]
        heldout = {
            "schema_version": "isabelle-transition-v2",
            "family": family,
            "mode": "family_local_heldout",
            "facts": ["Global.fact"],
            "statements": {"Global.fact": "global statement"},
            "requested_heldout": 1,
            "trajectory_drops": True,
        }
        drop_type = "heldout_trajectory_sibling"
    else:
        raise IntegrationError(f"{family}: not a family-local builder")
    if duplicate_id:
        rows[1]["id"] = rows[0]["id"]
    raw = output / "raw.jsonl"
    train = output / "train.jsonl"
    eval_path = output / "eval.jsonl"
    _write_jsonl(raw, rows)
    _write_jsonl(train, rows[:1])
    _write_jsonl(eval_path, rows[1:2])
    _write_json(output / "heldout.json", heldout)
    return _FamilyPackage(
        family=family,
        raw=raw,
        train=train,
        eval=eval_path,
        drops=(
            _NativeDrop(
                raw_row=3,
                drop_type=drop_type,
                details={"reason": "synthetic accepted trajectory exclusion"},
            ),
        ),
        heldout=heldout,
    )


def _mml_link_payload(contract: mml_holdout.ValidatedHoldoutContract) -> dict[str, Any]:
    def read_sidecar(relative: str) -> list[dict[str, Any]]:
        path = contract.root / relative
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    return {
        "family": "mml",
        "mode": "pooled_semantic_1000",
        "selected_classes": len(contract.selected_class_ids),
        "authoritative_manifest_root_sha256": contract.authoritative_root,
        "contract_manifest": contract.manifest,
        "projections": {
            family: contract.projections[family] for family in ("mizar", "atp")
        },
        "eval_exposure": read_sidecar("sidecars/eval_exposure.jsonl"),
        "drop_reasons": read_sidecar("sidecars/drop_reasons.jsonl"),
    }


def _row_validator(family: str) -> JsonlValidator:
    required = list(COMMON_REQUIRED_FIELDS)
    if family == "metamath":
        required.append("local_assumptions")
    elif family in {"prf2", "enigma"}:
        required.extend(("goal_name", "local_inputs", "proof_steps"))
    elif family == "isabelle":
        required.extend(
            (
                "local_assumptions",
                "local_names",
                "premise_aliases",
                "state_after",
                "state_before",
                "tactic",
                "theorem_statement",
                "trajectory_id",
                "transition_index",
            )
        )
    return JsonlValidator(
        schema_version=ROW_SCHEMAS[family],
        required_fields=tuple(required),
        allow_empty=False,
    )


def _linked_validator(schema: str, fields: Sequence[str]) -> JsonObjectValidator:
    return JsonObjectValidator(
        schema_version=schema,
        required_fields=tuple(fields),
        require_generation_links=True,
    )


def make_generation_plan(
    *,
    generation_id: str,
    source_generation_id: str,
) -> GenerationPlan:
    """Declare the one exact production inventory for all six siblings."""

    outputs: list[OutputSpec] = []
    for family in EXACT_SIBLINGS:
        validator = _row_validator(family)
        outputs.extend(
            (
                OutputSpec(
                    path=f"raw/{family}.jsonl",
                    role=OutputRole.RAW,
                    schema=ROW_SCHEMAS[family],
                    sibling=family,
                    validator=validator,
                ),
                OutputSpec(
                    path=f"shards/{family}.jsonl",
                    role=OutputRole.TRAIN,
                    schema=ROW_SCHEMAS[family],
                    sibling=family,
                    validator=validator,
                ),
                OutputSpec(
                    path=f"eval/{family}.jsonl",
                    role=OutputRole.EVAL,
                    schema=ROW_SCHEMAS[family],
                    sibling=family,
                    validator=validator,
                ),
                OutputSpec(
                    path=f"sidecars/drops/{family}.jsonl",
                    role=OutputRole.SIDECAR,
                    schema=DROP_SCHEMA,
                    sibling=family,
                    drop_types=DROP_TYPES[family],
                    validator=JsonlValidator(
                        schema_version=DROP_SCHEMA,
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
                    ),
                ),
                OutputSpec(
                    path=f"sidecars/sources/{family}.json",
                    role=OutputRole.SIDECAR,
                    schema=SOURCE_LINK_SCHEMA,
                    sibling=family,
                    validator=_linked_validator(
                        SOURCE_LINK_SCHEMA,
                        ("family", "manifest", "manifest_root_sha256"),
                    ),
                ),
            )
        )
    outputs.extend(
        (
            OutputSpec(
                path="heldout/mml.json",
                role=OutputRole.HELDOUT,
                schema=MML_HELDOUT_LINK_SCHEMA,
                validator=_linked_validator(
                    MML_HELDOUT_LINK_SCHEMA,
                    (
                        "authoritative_manifest_root_sha256",
                        "contract_manifest",
                        "drop_reasons",
                        "eval_exposure",
                        "family",
                        "mode",
                        "projections",
                        "selected_classes",
                    ),
                ),
            ),
            OutputSpec(
                path="heldout/metamath.json",
                role=OutputRole.HELDOUT,
                schema=FAMILY_HELDOUT_LINK_SCHEMA,
                sibling="metamath",
                validator=_linked_validator(
                    FAMILY_HELDOUT_LINK_SCHEMA,
                    (
                        "contract",
                        "family",
                        "mode",
                        "source_manifest_root_sha256",
                    ),
                ),
            ),
            OutputSpec(
                path="heldout/isabelle.json",
                role=OutputRole.HELDOUT,
                schema=FAMILY_HELDOUT_LINK_SCHEMA,
                sibling="isabelle",
                validator=_linked_validator(
                    FAMILY_HELDOUT_LINK_SCHEMA,
                    (
                        "contract",
                        "family",
                        "mode",
                        "source_manifest_root_sha256",
                    ),
                ),
            ),
            OutputSpec(
                path="sidecars/tokenizer.json",
                role=OutputRole.SIDECAR,
                schema=TOKENIZER_LINK_SCHEMA,
                validator=_linked_validator(
                    TOKENIZER_LINK_SCHEMA,
                    ("seal", "tokenizer_root_sha256"),
                ),
            ),
            OutputSpec(
                path="sidecars/policies.json",
                role=OutputRole.SIDECAR,
                schema=POLICY_LINK_SCHEMA,
                validator=_linked_validator(
                    POLICY_LINK_SCHEMA,
                    ("policies", "policy_root_sha256"),
                ),
            ),
            OutputSpec(
                path="sidecars/schemas.json",
                role=OutputRole.SIDECAR,
                schema=SCHEMA_LINK_SCHEMA,
                validator=_linked_validator(
                    SCHEMA_LINK_SCHEMA,
                    ("drop_schema", "row_schemas", "source_manifest_roots"),
                ),
            ),
            OutputSpec(
                path="sidecars/occurrences.json",
                role=OutputRole.SIDECAR,
                schema=OCCURRENCE_LINK_SCHEMA,
                validator=_linked_validator(
                    OCCURRENCE_LINK_SCHEMA,
                    ("families", "occurrences"),
                ),
            ),
            OutputSpec(
                path="sidecars/precheck.json",
                role=OutputRole.SIDECAR,
                schema=PRECHECK_LINK_SCHEMA,
                validator=_linked_validator(
                    PRECHECK_LINK_SCHEMA,
                    ("counts", "families", "status", "validators"),
                ),
            ),
        )
    )
    return GenerationPlan(
        generation_id=generation_id,
        source_generation_id=source_generation_id,
        requested_siblings=EXACT_SIBLINGS,
        outputs=tuple(outputs),
    )


def _expected_output_paths() -> set[str]:
    return {
        spec.path
        for spec in make_generation_plan(
            generation_id="inventory",
            source_generation_id="inventory",
        ).outputs
    }


def _reject_legacy_layout(root: Path) -> None:
    legacy = sorted(
        name
        for name in ("raw", "shards", "eval", "heldout", "sidecars", "artifacts")
        if (root / name).exists() or (root / name).is_symlink()
    )
    if legacy:
        raise IntegrationError(
            "legacy corpus directories are forbidden at the transaction root: "
            + ", ".join(legacy)
        )


def _source_generation_id(
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer: Mapping[str, Any],
    policies: Mapping[str, Any],
    *,
    mml_contract_root_sha256: str | None = None,
) -> str:
    payload = {
        "families": [
            {
                "family": family,
                "root": source_manifests[family]["manifest_root_sha256"],
            }
            for family in EXACT_SIBLINGS
        ],
        "tokenizer": _canonical_sha256(tokenizer),
        "policies": policies["policy_root_sha256"],
    }
    if mml_contract_root_sha256 is not None:
        _require_digest(
            mml_contract_root_sha256,
            "authoritative MML contract root",
            production=True,
        )
        payload["mml_contract_root_sha256"] = mml_contract_root_sha256
    root = _canonical_sha256(payload)
    return f"p3-sources-v2-{root}"


def _route_occurrence_ids(
    path: Path,
    raw: tuple[_LineOccurrence, ...],
    *,
    family: str,
    label: str,
) -> tuple[int, ...]:
    by_bytes: dict[bytes, list[int]] = {}
    for item in raw:
        by_bytes.setdefault(item.raw_bytes, []).append(item.line_number)
    used = Counter()
    rows = []
    for item in _read_occurrences(path, label=label):
        candidates = by_bytes.get(item.raw_bytes, [])
        index = used[item.raw_bytes]
        if index >= len(candidates):
            raise IntegrationError(
                f"{family}: routed row is not an exact unused raw occurrence"
            )
        rows.append(candidates[index])
        used[item.raw_bytes] += 1
    return tuple(rows)


def _validate_global_ids(
    raw_by_family: Mapping[str, tuple[_LineOccurrence, ...]],
) -> None:
    seen = {}
    for family in EXACT_SIBLINGS:
        for item in raw_by_family[family]:
            row_id = item.record["id"]
            previous = seen.get(row_id)
            if previous is not None:
                raise IntegrationError(
                    f"duplicate raw row id {row_id!r}: "
                    f"{previous} and {family}:{item.line_number}"
                )
            seen[row_id] = f"{family}:{item.line_number}"


def _reload_mml_contract_at_generation_boundary(
    contract: mml_holdout.ValidatedHoldoutContract,
    *,
    production: bool,
) -> mml_holdout.ValidatedHoldoutContract:
    if not isinstance(contract, mml_holdout.ValidatedHoldoutContract):
        raise IntegrationError("MML generation requires the authoritative typed contract")
    if contract.production is not production:
        raise IntegrationError("MML typed contract publication mode is stale")
    try:
        authoritative = mml_holdout.load_holdout_contract(
            contract.root,
            production=production,
        )
        _trim_unused_heap()
    except (
        mml_holdout.HoldoutError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise IntegrationError(
            f"MML holdout contract tuple validation failed: {error}"
        ) from error
    if contract != authoritative:
        raise IntegrationError("MML typed contract is stale at the generation boundary")
    return authoritative


def _write_transaction_payload(
    writer,
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer: Mapping[str, Any],
    policies: Mapping[str, Any],
    packages: Mapping[str, _FamilyPackage],
    mml_contract: mml_holdout.ValidatedHoldoutContract,
    fault_injector=None,
) -> None:
    def before_copy(logical_path: str) -> None:
        if fault_injector is not None:
            fault_injector(f"final_copy:{logical_path}")

    test_only = policies.get("test_only")
    if not isinstance(test_only, bool):
        raise IntegrationError("generation policy publication mode is missing")
    mml_contract = _reload_mml_contract_at_generation_boundary(
        mml_contract,
        production=not test_only,
    )
    metamath_source_accounting = None
    metamath_drop_ledger = None
    package = packages["metamath"]
    has_metamath_source_binding = (
        package.source_accounting is not None
        or package.overlength_drop_ledger is not None
    )
    if not test_only or has_metamath_source_binding:
        if (
            package.source_accounting is None
            or package.overlength_drop_ledger is None
        ):
            raise IntegrationError(
                "production Metamath package lacks native source/drop accounting"
            )
        metamath_drop_ledger = _validate_metamath_drop_ledger(
            package.overlength_drop_ledger,
            tokenizer_seal=tokenizer,
        )

    transaction_occurrences = {}
    occurrence_sidecar = []
    raw_counts = {}
    seen_ids = {}
    for family in EXACT_SIBLINGS:
        native_rows = _validate_rows(
            packages[family].raw,
            family=family,
            source_manifest=source_manifests[family],
            label=f"raw/{family}.jsonl",
        )
        for item in native_rows:
            row_id = item.record["id"]
            previous = seen_ids.get(row_id)
            if previous is not None:
                raise IntegrationError(
                    f"duplicate raw row id {row_id!r}: "
                    f"{previous} and {family}:{item.line_number}"
                )
            seen_ids[row_id] = f"{family}:{item.line_number}"
        if family == "metamath" and metamath_drop_ledger is not None:
            metamath_source_accounting = _validate_metamath_source_accounting(
                package.source_accounting,
                raw=native_rows,
                train=_read_occurrences(
                    package.train,
                    label="normalized/train/metamath.jsonl",
                ),
                evaluation=_read_occurrences(
                    package.eval,
                    label="normalized/eval/metamath.jsonl",
                ),
                drops=package.drops,
                drop_ledger=metamath_drop_ledger,
                tokenizer_seal=tokenizer,
            )
        logical = f"raw/{family}.jsonl"
        before_copy(logical)
        writer.copy_file(logical, packages[family].raw)
        indexed = writer.raw_occurrences(logical)
        if len(indexed) != len(native_rows):
            raise IntegrationError(f"{family}: transaction raw indexing changed row count")
        raw_counts[family] = len(native_rows)
        transaction_occurrences[family] = indexed
        for native, routed in zip(native_rows, indexed, strict=True):
            if native.raw_sha256 != routed.raw_sha256:
                raise IntegrationError(f"{family}: raw occurrence digest changed")
            occurrence_sidecar.append(
                {
                    "family": family,
                    "row_id": native.record["id"],
                    "raw_path": logical,
                    "source_line": native.line_number,
                    "byte_start": native.byte_start,
                    "byte_end": native.byte_end,
                    "raw_sha256": native.raw_sha256,
                    "occurrence_id": routed.occurrence_id,
                }
            )
        del native_rows
        gc.collect()

    mml_routes = mml_contract.manifest["row_routes"]
    for family in MML_SIBLINGS:
        for disposition, path in (
            ("train", mml_contract.family_paths[family].train),
            ("eval", mml_contract.family_paths[family].eval),
        ):
            row_numbers = tuple(
                route["line_number"]
                for route in mml_routes[family]
                if route["disposition"] == disposition
            )
            with path.open("rb") as routed_file:
                for number in row_numbers:
                    expected = transaction_occurrences[family][number - 1].raw_bytes
                    if routed_file.read(len(expected)) != expected:
                        raise IntegrationError(
                            f"{family}: MML {disposition} bytes disagree with route plan"
                        )
                if routed_file.read(1):
                    raise IntegrationError(
                        f"{family}: MML {disposition} contains trailing bytes"
                    )
            logical = (
                f"{'shards' if disposition == 'train' else 'eval'}/{family}.jsonl"
            )
            before_copy(logical)
            writer.copy_file(
                logical,
                path,
                occurrence_ids=[
                    transaction_occurrences[family][number - 1].occurrence_id
                    for number in row_numbers
                ],
            )
        drops = []
        for route in mml_routes[family]:
            if route["disposition"] != "drop":
                continue
            drops.append(
                DropRecord(
                    occurrence_id=transaction_occurrences[family][
                        route["line_number"] - 1
                    ].occurrence_id,
                    drop_type=route["drop_reason"],
                    details={
                        "mml_manifest_root_sha256": mml_contract.authoritative_root,
                        "native_row_sha256": route["native_row_sha256"],
                        "row_id": route["row_id"],
                        "text_plus_eos_tokens": route["text_plus_eos_tokens"],
                    },
                )
            )
        logical = f"sidecars/drops/{family}.jsonl"
        before_copy(logical)
        writer.write_drop_sidecar(logical, drops)

    for family in ("metamath", "isabelle"):
        package = packages[family]
        raw_by_bytes: dict[bytes, list[int]] = defaultdict(list)
        for occurrence in transaction_occurrences[family]:
            raw_by_bytes[occurrence.raw_bytes].append(occurrence.raw_row)

        def routed_rows(path: Path, *, label: str) -> tuple[int, ...]:
            used: Counter[bytes] = Counter()
            rows = []
            for item in _read_occurrences(path, label=label):
                available = raw_by_bytes.get(item.raw_bytes, ())
                offset = used[item.raw_bytes]
                if offset >= len(available):
                    raise IntegrationError(
                        f"{family}: routed row is not an exact unused raw occurrence"
                    )
                rows.append(available[offset])
                used[item.raw_bytes] += 1
            return tuple(rows)

        train_rows = routed_rows(
            package.train,
            label=f"builder/train/{family}.jsonl",
        )
        eval_rows = routed_rows(
            package.eval,
            label=f"builder/eval/{family}.jsonl",
        )
        logical = f"shards/{family}.jsonl"
        before_copy(logical)
        writer.copy_file(
            logical,
            package.train,
            occurrence_ids=[
                transaction_occurrences[family][number - 1].occurrence_id
                for number in train_rows
            ],
        )
        logical = f"eval/{family}.jsonl"
        before_copy(logical)
        writer.copy_file(
            logical,
            package.eval,
            occurrence_ids=[
                transaction_occurrences[family][number - 1].occurrence_id
                for number in eval_rows
            ],
        )
        logical = f"sidecars/drops/{family}.jsonl"
        before_copy(logical)
        writer.write_drop_sidecar(
            logical,
            [
                DropRecord(
                    occurrence_id=transaction_occurrences[family][
                        drop.raw_row - 1
                    ].occurrence_id,
                    drop_type=drop.drop_type,
                    details=drop.details,
                )
                for drop in package.drops
            ],
        )

    before_copy("heldout/mml.json")
    writer.write_linked_json("heldout/mml.json", _mml_link_payload(mml_contract))
    for family in ("metamath", "isabelle"):
        logical = f"heldout/{family}.json"
        before_copy(logical)
        payload = {
            "family": family,
            "mode": "family_local_heldout",
            "contract": dict(packages[family].heldout),
            "source_manifest_root_sha256": source_manifests[family][
                "manifest_root_sha256"
            ],
        }
        if family == "metamath" and metamath_source_accounting is not None:
            payload.update(
                {
                    "overlength_drop_ledger": metamath_drop_ledger,
                    "source_accounting": metamath_source_accounting,
                }
            )
        writer.write_linked_json(
            logical,
            payload,
        )
    for family in EXACT_SIBLINGS:
        logical = f"sidecars/sources/{family}.json"
        before_copy(logical)
        writer.write_linked_json(
            logical,
            {
                "family": family,
                "manifest": dict(source_manifests[family]),
                "manifest_root_sha256": source_manifests[family][
                    "manifest_root_sha256"
                ],
            },
        )
    before_copy("sidecars/tokenizer.json")
    writer.write_linked_json(
        "sidecars/tokenizer.json",
        {
            "seal": dict(tokenizer),
            "tokenizer_root_sha256": _canonical_sha256(tokenizer),
        },
    )
    before_copy("sidecars/policies.json")
    writer.write_linked_json(
        "sidecars/policies.json",
        {
            "policies": dict(policies),
            "policy_root_sha256": policies["policy_root_sha256"],
        },
    )
    before_copy("sidecars/schemas.json")
    writer.write_linked_json(
        "sidecars/schemas.json",
        {
            "row_schemas": ROW_SCHEMAS,
            "drop_schema": DROP_SCHEMA,
            "source_manifest_roots": {
                family: source_manifests[family]["manifest_root_sha256"]
                for family in EXACT_SIBLINGS
            },
        },
    )
    before_copy("sidecars/occurrences.json")
    occurrence_payload = {
        "families": list(EXACT_SIBLINGS),
        "occurrences": occurrence_sidecar,
    }
    if metamath_source_accounting is not None:
        occurrence_payload["source_accounting"] = {
            "metamath": metamath_source_accounting
        }
    writer.write_linked_json("sidecars/occurrences.json", occurrence_payload)
    before_copy("sidecars/precheck.json")
    precheck_payload = {
        "status": "clean",
        "families": list(EXACT_SIBLINGS),
        "counts": {
            family: {
                "raw": raw_counts[family],
                "train": sum(
                    1
                    for route in (
                        mml_routes[family] if family in MML_SIBLINGS else ()
                    )
                    if route["disposition"] == "train"
                )
                if family in MML_SIBLINGS
                else len(_read_occurrences(packages[family].train, label=family)),
                "eval": sum(
                    1
                    for route in (
                        mml_routes[family] if family in MML_SIBLINGS else ()
                    )
                    if route["disposition"] == "eval"
                )
                if family in MML_SIBLINGS
                else len(_read_occurrences(packages[family].eval, label=family)),
            }
            for family in EXACT_SIBLINGS
        },
        "validators": {
            "rows": "family-deep-reconstruction-v2",
            "mml": "ValidatedHoldoutContract",
            "routes": "physical-occurrence-routes/v2",
        },
    }
    if metamath_source_accounting is not None:
        precheck_payload["source_accounting"] = {
            "metamath": metamath_source_accounting
        }
        precheck_payload["validators"]["metamath_source_occurrences"] = (
            "metamath-source-occurrences-v1"
        )
    writer.write_linked_json("sidecars/precheck.json", precheck_payload)


def _builder_quarantine(
    run_root: Path,
    *,
    work_root: Path,
    generation_id: str,
    error: BaseException,
) -> None:
    if not run_root.exists():
        return
    quarantine = work_root / "quarantine"
    quarantine.mkdir(exist_ok=True)
    suffix = 0
    while True:
        name = f"{generation_id}.{suffix:03d}"
        destination = quarantine / name
        if not destination.exists():
            break
        suffix += 1
    run_root.rename(destination)
    _write_json(
        destination / "BUILDER_QUARANTINE.json",
        {
            "schema_version": BUILDER_QUARANTINE_SCHEMA,
            "generation_id": generation_id,
            "error_type": type(error).__name__,
            "reason": str(error),
        },
    )


def builder_quarantine_inventory(work_root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read the external builder quarantine without inspecting corpus generations."""

    root = Path(work_root) / "quarantine"
    if not root.exists():
        return []
    records = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.is_symlink():
            raise IntegrationError(f"invalid builder quarantine entry: {entry}")
        marker = entry / "BUILDER_QUARANTINE.json"
        record = _read_json(marker, "builder quarantine marker")
        if record.get("schema_version") != BUILDER_QUARANTINE_SCHEMA:
            raise IntegrationError(f"invalid builder quarantine schema: {entry.name}")
        records.append(record)
    return records


def synthetic_fault_points() -> tuple[str, ...]:
    """Return every direct-unit synthetic integration fault boundary."""

    return (
        *(f"raw_builder:{family}" for family in EXACT_SIBLINGS),
        *(f"builder_complete:{family}" for family in EXACT_SIBLINGS),
        "split_builder:start:metamath",
        "split_builder:complete:metamath",
        "split_builder:start:isabelle",
        "split_builder:complete:isabelle",
        "split_builder:start:mml",
        "split_builder:complete:mml",
        *(f"family_split:complete:{family}" for family in EXACT_SIBLINGS),
        "normalization:metamath",
        "normalization:isabelle",
        "partition:metamath",
        "partition:isabelle",
        "mml_partition",
        "precheck",
        *(f"final_copy:{path}" for path in sorted(_expected_output_paths())),
    )


def build_synthetic_generation(
    *,
    corpus_root: str | os.PathLike[str],
    work_root: str | os.PathLike[str],
    generation_id: str,
    forbidden_legacy_paths: Sequence[str | os.PathLike[str]] = (),
    fail_family: str | None = None,
    duplicate_id_family: str | None = None,
    fault_point: str | None = None,
    metamath_drop_ledger: Mapping[str, Any] | None = None,
) -> SyntheticBuildResult:
    """Build the deterministic, small six-family end-to-end fixture."""

    root = Path(corpus_root)
    work = Path(work_root)
    _reject_legacy_layout(root)
    if not work.is_absolute():
        raise IntegrationError("trusted work root must be absolute")
    if not work.is_dir() or work.is_symlink():
        raise IntegrationError("trusted work root must be an existing real directory")
    resolved_root = root.resolve(strict=False)
    resolved_work = work.resolve()
    if resolved_work == resolved_root or resolved_work.is_relative_to(resolved_root):
        raise IntegrationError("builder work root must be external to transaction root")
    forbidden = tuple(Path(path).resolve() for path in forbidden_legacy_paths)
    if any(
        resolved_work == path or resolved_work.is_relative_to(path) for path in forbidden
    ):
        raise IntegrationError("builder work root overlaps a forbidden legacy corpus")
    if fail_family is not None and fail_family not in EXACT_SIBLINGS:
        raise IntegrationError(f"unknown crash family {fail_family!r}")
    if duplicate_id_family is not None and duplicate_id_family not in EXACT_SIBLINGS:
        raise IntegrationError(f"unknown duplicate-ID family {duplicate_id_family!r}")
    if fault_point is not None and fault_point not in synthetic_fault_points():
        raise IntegrationError(f"unknown synthetic fault point {fault_point!r}")

    def inject(point: str) -> None:
        if fault_point == point:
            raise IntegrationError(f"injected fault at {point}")

    tokenizer = _validate_tokenizer_seal(mml_holdout.approved_tokenizer_seal())
    source_manifests = {
        family: _validate_source_manifest(
            _make_source_manifest(family, test_only=True),
            family=family,
            production=False,
        )
        for family in EXACT_SIBLINGS
    }
    synthetic_ledger = None
    if metamath_drop_ledger is not None:
        synthetic_ledger = _validate_metamath_drop_ledger(
            metamath_drop_ledger,
            tokenizer_seal=tokenizer,
        )
        native_manifest = {
            "repository": "synthetic://metamath",
            "commit": "fixture",
            "files": {
                "synthetic.mm": {
                    "sha256": hashlib.sha256(b"synthetic-metamath").hexdigest()
                }
            },
        }
        manifest = _make_source_manifest("metamath", test_only=True)
        manifest["row_source_metadata"] = metamath_builder.build_source_metadata(
            native_manifest,
            {},
            drop_ledger=synthetic_ledger,
            tokenizer_seal=tokenizer,
        )
        manifest["source_snapshots"] = [native_manifest]
        manifest["manifest_root_sha256"] = _source_manifest_root(manifest)
        source_manifests["metamath"] = _validate_source_manifest(
            manifest,
            family="metamath",
            production=False,
        )
    policies = _validate_policies(_synthetic_policies(), production=False)
    source_generation_id = _source_generation_id(
        source_manifests,
        tokenizer,
        policies,
    )
    plan = make_generation_plan(
        generation_id=generation_id,
        source_generation_id=source_generation_id,
    )
    run_root = Path(
        tempfile.mkdtemp(prefix=f"p3-{generation_id}.", dir=str(resolved_work))
    )
    builder_roots: list[Path] = []

    def producer(writer) -> None:
        raw_paths = {}
        packages: dict[str, _FamilyPackage] = {}
        for family in EXACT_SIBLINGS:
            output = run_root / "builders" / family
            output.mkdir(parents=True)
            builder_roots.append(output)
            if fail_family == family:
                raise IntegrationError(f"{family}: injected builder crash")
            inject(f"raw_builder:{family}")
            if family in MML_SIBLINGS:
                rows = _synthetic_mml_rows(family, source_manifests[family])
                if duplicate_id_family == family:
                    rows[1]["id"] = rows[0]["id"]
                raw = output / "raw.jsonl"
                _write_jsonl(raw, rows)
                raw_paths[family] = raw
                inject(f"builder_complete:{family}")
            else:
                inject(f"builder_complete:{family}")
                inject(f"split_builder:start:{family}")
                packages[family] = _build_family_local_package(
                    family,
                    output,
                    source_manifests[family],
                    duplicate_id=duplicate_id_family == family,
                )
                if family == "metamath" and synthetic_ledger is not None:
                    package = packages[family]
                    raw = _validate_rows(
                        package.raw,
                        family="metamath",
                        source_manifest=source_manifests["metamath"],
                        label="synthetic/raw/metamath.jsonl",
                    )
                    source_accounting = _validate_metamath_source_accounting(
                        None,
                        raw=raw,
                        train=_read_occurrences(
                            package.train,
                            label="synthetic/train/metamath.jsonl",
                        ),
                        evaluation=_read_occurrences(
                            package.eval,
                            label="synthetic/eval/metamath.jsonl",
                        ),
                        drops=package.drops,
                        drop_ledger=synthetic_ledger,
                        tokenizer_seal=tokenizer,
                    )
                    packages[family] = _FamilyPackage(
                        family=package.family,
                        raw=package.raw,
                        train=package.train,
                        eval=package.eval,
                        drops=package.drops,
                        heldout=package.heldout,
                        source_accounting=source_accounting,
                        overlength_drop_ledger=synthetic_ledger,
                    )
                inject(f"split_builder:complete:{family}")
            if family in {"metamath", "isabelle"}:
                inject(f"normalization:{family}")
                inject(f"partition:{family}")
                inject(f"family_split:complete:{family}")
        inject("mml_partition")
        inject("split_builder:start:mml")
        contract_root = run_root / "mml-contract"
        builder_roots.append(contract_root)
        try:
            contract = _build_synthetic_mml_contract(
                raw_paths,
                source_manifests,
                contract_root,
            )
        except (
            mml_holdout.HoldoutError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise IntegrationError(str(error)) from error
        inject("split_builder:complete:mml")
        for family in MML_SIBLINGS:
            packages[family] = _FamilyPackage(
                family=family,
                raw=raw_paths[family],
                train=contract.family_paths[family].train,
                eval=contract.family_paths[family].eval,
                drops=(),
                heldout=contract.manifest,
            )
            inject(f"family_split:complete:{family}")
        inject("precheck")
        _write_transaction_payload(
            writer,
            source_manifests=source_manifests,
            tokenizer=tokenizer,
            policies=policies,
            packages=packages,
            mml_contract=contract,
            fault_injector=inject,
        )

    try:
        published = GenerationCoordinator(root).publish(plan, producer)
    except BaseException as error:
        _builder_quarantine(
            run_root,
            work_root=resolved_work,
            generation_id=generation_id,
            error=error,
        )
        if isinstance(error, IntegrationError):
            raise
        raise IntegrationError(
            f"six-family generation {generation_id} failed: {error}"
        ) from error
    return SyntheticBuildResult(
        published=published,
        builder_output_roots=tuple(builder_roots),
    )


def _safe_builder_output_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrationError(f"{label}: builder output path is missing")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise IntegrationError(f"{label}: builder output path is not safe and relative")
    if str(path) != value:
        raise IntegrationError(f"{label}: builder output path is not canonical")
    return value


def _builder_option_value(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def validate_production_builder_command(
    *,
    family: str,
    stage: str,
    argv: Any,
) -> list[str]:
    """Validate a command against the accepted builder's closed argparse surface."""

    if family not in PRODUCTION_BUILDER_SCRIPTS:
        raise IntegrationError(f"{family}/{stage}: unknown production builder family")
    if stage not in {"raw", "split"}:
        raise IntegrationError(f"{family}/{stage}: unknown production builder stage")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or not all(isinstance(value, str) and value and "\0" not in value for value in argv)
    ):
        raise IntegrationError(f"{family}/{stage}: builder argv must be a string list")
    for value in argv:
        normalized = value.replace("\\", "/")
        if BYPASS_TOKEN_RE.search(normalized):
            raise IntegrationError(
                f"{family}/{stage}: production bypass/test token is forbidden: {value}"
            )
    executable = Path(argv[0]).name
    if executable not in {"python", "python3", Path(sys.executable).name}:
        raise IntegrationError(f"{family}/{stage}: builder must use the current Python")
    expected_script = (SCRIPT_DIRECTORY / PRODUCTION_BUILDER_SCRIPTS[family]).resolve()
    supplied_script = Path(argv[1])
    if not supplied_script.is_absolute():
        supplied_script = SCRIPT_DIRECTORY.parent / supplied_script
    if supplied_script.resolve(strict=False) != expected_script:
        raise IntegrationError(
            f"{family}/{stage}: command is not the accepted {expected_script.name} builder"
        )

    allowed = PRODUCTION_BUILDER_OPTIONS[family]
    parsed: dict[str, list[str]] = {}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if not flag.startswith("--") or "=" in flag:
            raise IntegrationError(
                f"{family}/{stage}: unknown positional or combined argument {flag!r}"
            )
        arity = allowed.get(flag)
        if arity is None:
            raise IntegrationError(f"{family}/{stage}: unknown builder flag {flag}")
        if flag in parsed:
            raise IntegrationError(f"{family}/{stage}: duplicate builder flag {flag}")
        if arity == "zero":
            parsed[flag] = []
            index += 1
            continue
        values = []
        index += 1
        while index < len(argv) and not argv[index].startswith("--"):
            values.append(argv[index])
            index += 1
            if arity == "one":
                break
        if not values:
            raise IntegrationError(f"{family}/{stage}: {flag} requires a value")
        if index < len(argv) and not argv[index].startswith("--"):
            raise IntegrationError(f"{family}/{stage}: {flag} has too many values")
        parsed[flag] = values

    low_tier_flags = {"--enigma-low-tier-base", "--tokenizer-json"}
    supplied_low_tier_flags = low_tier_flags.intersection(parsed)
    if supplied_low_tier_flags:
        if family != "enigma" or stage != "raw":
            raise IntegrationError(
                f"{family}/{stage}: ENIGMA low-tier flags are valid only in enigma/raw"
            )
        if supplied_low_tier_flags != low_tier_flags:
            raise IntegrationError(
                f"{family}/{stage}: ENIGMA low-tier flags must appear together"
            )

    required = {
        "metamath": {"--mm-dir", "--heldout", "--seed", "--tokenizer"},
        "mizar": {
            "--mml-root",
            "--html-root",
            "--thproofs-root",
            "--semantic-index",
            "--semantic-index-sha256",
            "--source-manifest",
            "--mizar-archive",
            "--html-archive",
            "--thproofs-archive",
            "--tokenizer-path",
            "--name",
            "--heldout",
            "--seed",
        },
        "thproofs": {
            "--src",
            "--semantic-index",
            "--source-manifest",
            "--mml-root",
            "--html-root",
            "--mizar-archive",
            "--html-archive",
            "--thproofs-archive",
            "--name",
            "--heldout",
            "--seed",
        },
        "prf2": {"--src", "--name", "--heldout", "--min-steps", "--seed"},
        "enigma": {
            "--src",
            "--name",
            "--fenced",
            "--heldout",
            "--min-steps",
            "--dedup",
            "--jaccard",
            "--seed",
            "--enigma-low-tier-base",
            "--tokenizer-json",
        },
        "isabelle": {
            "--src",
            "--name",
            "--heldout",
            "--seed",
            "--tokenizer-path",
        },
    }[family]
    missing = sorted(required - set(parsed))
    if missing:
        raise IntegrationError(
            f"{family}/{stage}: required builder flags are missing: {missing}"
        )
    if "--name" in allowed and parsed.get("--name") != [family]:
        raise IntegrationError(f"{family}/{stage}: builder --name must be {family}")
    expected_heldout = "500" if stage == "split" else "0"
    if parsed["--heldout"] != [expected_heldout]:
        raise IntegrationError(
            f"{family}/{stage}: builder --heldout must be {expected_heldout}"
        )
    normalized_argv = list(argv)
    if family == "thproofs":
        exclusion = parsed.get("--exclude")
        if exclusion is None:
            normalized_argv.extend(("--exclude", NO_EXCLUSION_PATH))
        elif exclusion != [NO_EXCLUSION_PATH]:
            raise IntegrationError(
                f"{family}/{stage}: pooled raw build must use no exclusion"
            )
    return normalized_argv


def _validate_builder_argv(
    argv: Any,
    *,
    family: str,
    stage: str,
    corpus_root: Path,
    forbidden_legacy_paths: Sequence[Path],
) -> list[str]:
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(value, str) and value and "\0" not in value for value in argv)
    ):
        raise IntegrationError(f"{family}/{stage}: builder argv must be a string list")
    forbidden_flags = {"--out", "--output", "--output-dir", "-o"}
    if any(
        value in forbidden_flags
        or any(value.startswith(flag + "=") for flag in forbidden_flags)
        for value in argv
    ):
        raise IntegrationError(
            f"{family}/{stage}: builder argv must not choose its output path"
        )
    protected = (
        corpus_root.resolve(strict=False),
        (SCRIPT_DIRECTORY.parent / "corpus").resolve(strict=False),
        (SCRIPT_DIRECTORY.parent / "artifacts").resolve(strict=False),
        *forbidden_legacy_paths,
    )
    for value in argv[1:]:
        candidate = Path(value).expanduser()
        resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (SCRIPT_DIRECTORY.parent / candidate).resolve(strict=False)
        )
        if any(
            resolved == root or resolved.is_relative_to(root)
            for root in protected
        ):
            raise IntegrationError(
                f"{family}/{stage}: builder argv references a protected corpus path"
            )
    return list(argv)


def _validate_stage_inventory_declaration(
    inventory: Any,
    *,
    family: str,
    stage: str,
) -> dict[str, Mapping[str, Any]]:
    if (
        not isinstance(inventory, Sequence)
        or isinstance(inventory, (str, bytes))
        or not inventory
    ):
        raise IntegrationError(f"{family}/{stage}: exact stage inventory is missing")
    declarations = {}
    for entry in inventory:
        if not isinstance(entry, Mapping):
            raise IntegrationError(f"{family}/{stage}: malformed inventory declaration")
        kind = entry.get("kind")
        file_format = entry.get("format")
        expected_fields = (
            {"path", "kind"}
            if kind == "directory"
            else {
                "path",
                "kind",
                "format",
                "schema",
                *(("required_fields",) if file_format == "json" else ()),
            }
        )
        optional = (
            {"allow_empty", "source_manifest_root_sha256"}
            if kind == "file"
            else set()
        )
        if set(entry) - optional != expected_fields or kind not in {"file", "directory"}:
            raise IntegrationError(
                f"{family}/{stage}: inventory declaration fields are not exact"
            )
        relative = _safe_builder_output_path(
            entry.get("path"),
            label=f"{family}/{stage}/inventory",
        )
        if relative in declarations:
            raise IntegrationError(
                f"{family}/{stage}: duplicate inventory path {relative}"
            )
        if kind == "file":
            if entry.get("format") not in {"json", "jsonl", "binary"}:
                raise IntegrationError(
                    f"{family}/{stage}: unsupported inventory format"
                )
            schema = entry.get("schema")
            if not isinstance(schema, str) or not re.search(r"(?:/|-)v[1-9][0-9]*\Z", schema):
                raise IntegrationError(
                    f"{family}/{stage}: inventory schema is not versioned"
                )
            if file_format == "json":
                required_fields = entry.get("required_fields")
                if (
                    not isinstance(required_fields, list)
                    or not required_fields
                    or len(required_fields) != len(set(required_fields))
                    or not all(
                        isinstance(field, str)
                        and field
                        and field != "schema_version"
                        for field in required_fields
                    )
                ):
                    raise IntegrationError(
                        f"{family}/{stage}: JSON required_fields are malformed"
                    )
            expected_root = entry.get("source_manifest_root_sha256")
            if expected_root is not None and not SHA256_RE.fullmatch(expected_root):
                raise IntegrationError(
                    f"{family}/{stage}: inventory source root is malformed"
                )
            allow_empty = entry.get("allow_empty", False)
            if type(allow_empty) is not bool:
                raise IntegrationError(
                    f"{family}/{stage}: inventory allow_empty must be a boolean"
                )
            if allow_empty and (stage != "raw" or file_format != "jsonl"):
                raise IntegrationError(
                    f"{family}/{stage}: only auxiliary raw JSONL may be empty"
                )
        declarations[relative] = dict(entry)
    for relative in declarations:
        parent = Path(relative).parent
        while parent != Path("."):
            parent_name = parent.as_posix()
            parent_entry = declarations.get(parent_name)
            if parent_entry is None or parent_entry["kind"] != "directory":
                raise IntegrationError(
                    f"{family}/{stage}: parent directory {parent_name!r} "
                    "must be declared"
                )
            parent = parent.parent
    return declarations


def _validate_inventory_file(
    path: Path,
    declaration: Mapping[str, Any],
    *,
    family: str,
    stage: str,
) -> None:
    file_format = declaration["format"]
    if path.stat().st_size == 0 and declaration.get("allow_empty") is True:
        return
    if file_format == "binary":
        if path.stat().st_size == 0:
            raise IntegrationError(f"{family}/{stage}: empty binary inventory output")
        return
    if file_format == "json":
        values = [_read_json(path, f"{family}/{stage}/{declaration['path']}")]
    else:
        occurrences = _read_occurrences(
            path,
            label=f"{family}/{stage}/{declaration['path']}",
        )
        if not occurrences:
            raise IntegrationError(f"{family}/{stage}: empty JSONL inventory output")
        values = [item.record for item in occurrences]
    expected_schema = declaration["schema"]
    expected_root = declaration.get("source_manifest_root_sha256")
    for value in values:
        actual_schema = value.get("schema_version")
        if actual_schema != expected_schema:
            raise IntegrationError(
                f"{family}/{stage}: inventory schema mismatch for {declaration['path']}"
            )
        if file_format == "json":
            missing = sorted(
                field
                for field in declaration["required_fields"]
                if field not in value
            )
            if missing:
                raise IntegrationError(
                    f"{family}/{stage}: inventory JSON is missing required "
                    f"contract fields: {missing}"
                )
        if expected_root is not None:
            metadata = value.get("source_metadata")
            if (
                not isinstance(metadata, Mapping)
                or metadata.get("source_manifest_root_sha256") != expected_root
            ):
                raise IntegrationError(
                    f"{family}/{stage}: inventory source root mismatch"
                )


def _validate_exact_stage_inventory(
    output_root: Path,
    inventory: Any,
    *,
    family: str,
    stage: str,
) -> dict[str, Path]:
    declarations = _validate_stage_inventory_declaration(
        inventory,
        family=family,
        stage=stage,
    )
    actual = {}
    for path in sorted(output_root.rglob("*")):
        relative = path.relative_to(output_root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise IntegrationError(f"{family}/{stage}: symlink in builder inventory")
        kind = (
            "directory"
            if stat.S_ISDIR(metadata.st_mode)
            else "file"
            if stat.S_ISREG(metadata.st_mode)
            else "special"
        )
        if kind == "special":
            raise IntegrationError(f"{family}/{stage}: special staging output is forbidden")
        if TEMPORARY_NAME_RE.search(path.name):
            raise IntegrationError(f"{family}/{stage}: temporary staging output is forbidden")
        actual[relative] = kind
    declared_kinds = {path: entry["kind"] for path, entry in declarations.items()}
    if actual != declared_kinds:
        undeclared = sorted(set(actual) - set(declared_kinds))
        missing = sorted(set(declared_kinds) - set(actual))
        wrong_kind = sorted(
            path
            for path in set(actual) & set(declared_kinds)
            if actual[path] != declared_kinds[path]
        )
        raise IntegrationError(
            f"{family}/{stage}: exact recursive inventory mismatch; "
            f"undeclared={undeclared}, missing={missing}, wrong_kind={wrong_kind}"
        )
    resolved = {}
    for relative, declaration in declarations.items():
        path = output_root.joinpath(*Path(relative).parts)
        if declaration["kind"] == "file":
            _validate_inventory_file(
                path,
                declaration,
                family=family,
                stage=stage,
            )
            resolved[relative] = path
    return resolved


def _run_builder_callback_stage(
    *,
    family: str,
    stage: str,
    output_root: Path,
    inventory: list[Mapping[str, Any]],
    callback,
) -> dict[str, Path]:
    """Direct unit-only callback seam; production config cannot encode this."""

    output_root.mkdir(parents=True)
    callback(output_root)
    return _validate_exact_stage_inventory(
        output_root,
        inventory,
        family=family,
        stage=stage,
    )


def _run_builder_stage(
    *,
    family: str,
    stage: str,
    specification: Mapping[str, Any],
    output_root: Path,
    corpus_root: Path,
    forbidden_legacy_paths: Sequence[Path],
) -> dict[str, Path]:
    if not isinstance(specification, Mapping) or set(specification) != {
        "argv",
        "inventory",
        "outputs",
    }:
        raise IntegrationError(
            f"{family}/{stage}: builder command fields must be argv, inventory, and outputs"
        )
    argv = _validate_builder_argv(
        specification["argv"],
        family=family,
        stage=stage,
        corpus_root=corpus_root,
        forbidden_legacy_paths=forbidden_legacy_paths,
    )
    outputs = specification["outputs"]
    if not isinstance(outputs, Mapping) or not outputs:
        raise IntegrationError(f"{family}/{stage}: expected outputs are missing")
    relative_outputs = {
        str(name): _safe_builder_output_path(
            value,
            label=f"{family}/{stage}/{name}",
        )
        for name, value in outputs.items()
        if isinstance(name, str) and name
    }
    if len(relative_outputs) != len(outputs):
        raise IntegrationError(f"{family}/{stage}: malformed expected output name")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if output_root.exists() or output_root.is_symlink():
        raise IntegrationError(
            f"{family}/{stage}: builder output root must not already exist"
        )
    log = output_root.parent / f"{output_root.name}.builder.log"
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    scripts_path = str(SCRIPT_DIRECTORY)
    environment["PYTHONPATH"] = (
        scripts_path
        if not environment.get("PYTHONPATH")
        else scripts_path + os.pathsep + environment["PYTHONPATH"]
    )
    with log.open("wb") as log_file:
        result = subprocess.run(
            [*argv, "--out", str(output_root)],
            cwd=SCRIPT_DIRECTORY.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise IntegrationError(
            f"{family}: {stage} builder crashed with exit code {result.returncode}"
        )
    inventory_files = _validate_exact_stage_inventory(
        output_root,
        specification["inventory"],
        family=family,
        stage=stage,
    )
    resolved_outputs = {}
    for name, relative in relative_outputs.items():
        path = inventory_files.get(relative)
        if path is None:
            raise IntegrationError(
                f"{family}/{stage}: {name} output is not a declared inventory file"
            )
        resolved_outputs[name] = path
    return resolved_outputs


def _validate_production_builder_config(
    manifest: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    builder = manifest.get("builder")
    expected = (
        {"driver", "partition_mode", "raw"}
        if family in MML_SIBLINGS
        else {"driver", "partition_mode", "raw", "split"}
    )
    if not isinstance(builder, dict) or set(builder) != expected:
        raise IntegrationError(f"{family}: production builder config is not exact")
    if builder["driver"] != "external-command-v2":
        raise IntegrationError(f"{family}: unsupported production builder driver")
    expected_mode = (
        "pooled-mml-1000-v1"
        if family in MML_SIBLINGS
        else "family-local-heldout-v2"
    )
    if builder["partition_mode"] != expected_mode:
        raise IntegrationError(f"{family}: wrong builder partition mode")
    builder = _json_copy(builder)
    for stage in ("raw",) if family in MML_SIBLINGS else ("raw", "split"):
        specification = builder[stage]
        if not isinstance(specification, dict) or set(specification) != {
            "argv",
            "inventory",
            "outputs",
        }:
            raise IntegrationError(
                f"{family}/{stage}: production command fields are not exact"
            )
        specification["argv"] = validate_production_builder_command(
            family=family,
            stage=stage,
            argv=specification["argv"],
        )
        declarations = _validate_stage_inventory_declaration(
            specification["inventory"],
            family=family,
            stage=stage,
        )
        outputs = specification["outputs"]
        if not isinstance(outputs, Mapping) or not outputs:
            raise IntegrationError(f"{family}/{stage}: output map is missing")
        output_paths = {
            _safe_builder_output_path(
                relative,
                label=f"{family}/{stage}/{name}",
            )
            for name, relative in outputs.items()
            if isinstance(name, str) and name
        }
        if len(output_paths) != len(outputs) or any(
            declarations.get(path, {}).get("kind") != "file"
            for path in output_paths
        ):
            raise IntegrationError(
                f"{family}/{stage}: outputs must reference declared inventory files"
            )
        expected_output_names = (
            {"raw"} if stage == "raw" else {"train", "eval", "heldout"}
        )
        if set(outputs) != expected_output_names:
            raise IntegrationError(
                f"{family}/{stage}: logical output inventory is not exact"
            )
        for name, relative in outputs.items():
            declaration = declarations[relative]
            if name in {"raw", "train", "eval"}:
                if (
                    declaration.get("format") != "jsonl"
                    or declaration.get("schema") != ROW_SCHEMAS[family]
                    or declaration.get("source_manifest_root_sha256")
                        != manifest["row_source_metadata"][
                            "source_manifest_root_sha256"
                        ]
                        or declaration.get("allow_empty") is True
                ):
                    raise IntegrationError(
                        f"{family}/{stage}: {name} schema/source root is not exact"
                    )
            else:
                expected_heldout_schema = (
                    METAMATH_HELDOUT_SCHEMA
                    if family == "metamath"
                    else ISABELLE_HELDOUT_SCHEMA
                )
                expected_heldout_fields = (
                    METAMATH_HELDOUT_FIELDS
                    if family == "metamath"
                    else ISABELLE_HELDOUT_FIELDS
                )
                if (
                    declaration.get("format") != "json"
                    or declaration.get("schema") != expected_heldout_schema
                    or set(declaration.get("required_fields", ()))
                    != set(expected_heldout_fields)
                ):
                    raise IntegrationError(
                        f"{family}/{stage}: heldout schema is not exact"
                    )
    return builder


def _exact_split_rows(
    path: Path,
    *,
    raw: tuple[_LineOccurrence, ...],
    family: str,
    label: str,
) -> tuple[_LineOccurrence, ...]:
    raw_by_bytes: dict[bytes, list[_LineOccurrence]] = defaultdict(list)
    for item in raw:
        raw_by_bytes[item.raw_bytes].append(item)
    seen: Counter[bytes] = Counter()
    result = []
    for item in _read_occurrences(path, label=label):
        available = raw_by_bytes.get(item.raw_bytes, ())
        offset = seen[item.raw_bytes]
        if offset >= len(available):
            raise IntegrationError(
                f"{family}: split output contains a non-raw or duplicate occurrence"
            )
        result.append(available[offset])
        seen[item.raw_bytes] += 1
    return tuple(result)


def _metamath_isolation_context(
    raw: tuple[_LineOccurrence, ...],
    held_facts: Iterable[str],
) -> _MetamathIsolationContext:
    held_fact_names = frozenset(held_facts)
    if not held_fact_names:
        raise IntegrationError("Metamath held-fact contract is empty")
    held_fact_names_by_statement: dict[str, set[str]] = {}
    for item in raw:
        for name, statement in item.record["facts"].items():
            if name not in held_fact_names:
                continue
            statement_identities = metamath_builder.normalized_statements((statement,))
            statement_identity = next(iter(statement_identities))
            held_fact_names_by_statement.setdefault(statement_identity, set()).add(name)
    if not held_fact_names_by_statement:
        raise IntegrationError("Metamath held statements cannot be reconstructed")
    frozen_names_by_statement = {
        statement: tuple(sorted(names))
        for statement, names in held_fact_names_by_statement.items()
    }
    return _MetamathIsolationContext(
        held_fact_names=held_fact_names,
        held_statement_identities=frozenset(frozen_names_by_statement),
        held_fact_names_by_statement=frozen_names_by_statement,
    )


def _classify_metamath_route(
    record: Mapping[str, Any],
    context: _MetamathIsolationContext,
) -> _MetamathRouteClassification:
    theorem = record["theorem"].split(":", 1)[-1]
    target_steps = []
    for line in record["target"].splitlines():
        match = re.match(r"^\s*\d+\s+(\S+)\s+(.+)$", line)
        if match is None:
            raise IntegrationError("Metamath target is malformed during isolation check")
        target_steps.append((match.group(1), match.group(2)))
    target_labels = {label for label, _ in target_steps}
    held_name_exposure = bool(
        set(record["cited"]) & context.held_fact_names
        or set(record["facts"]) & context.held_fact_names
        or target_labels & context.held_fact_names
    )
    local = record.get("local_assumptions", {})
    goal_statement_identities = metamath_builder.normalized_statements((record["goal"],))
    visible_statement_identities = metamath_builder.normalized_statements(
        (
            *record["facts"].values(),
            *(local.values() if isinstance(local, Mapping) else ()),
            *(expression for _, expression in target_steps),
        )
    )
    if theorem in context.held_fact_names:
        return _MetamathRouteClassification(
            disposition="drop",
            drop_type="heldout_own_proof",
            drop_details={"held_fact": theorem},
            detail="Metamath held own-proof",
        )
    held_goal_identities = (
        goal_statement_identities & context.held_statement_identities
    )
    if held_goal_identities:
        held_statement = min(held_goal_identities)
        return _MetamathRouteClassification(
            disposition="drop",
            drop_type="heldout_own_proof",
            drop_details={
                "held_fact": context.held_fact_names_by_statement[held_statement][0]
            },
            detail="Metamath held own-proof",
        )
    if held_name_exposure or (
        visible_statement_identities & context.held_statement_identities
    ):
        return _MetamathRouteClassification(
            disposition="eval",
            drop_type=None,
            drop_details={},
            detail="Metamath held name/statement/local/target",
        )
    return _MetamathRouteClassification(
        disposition="train",
        drop_type=None,
        drop_details={},
        detail="Metamath held isolation",
    )


def _metamath_source_occurrence_binding(
    raw: tuple[_LineOccurrence, ...],
    *,
    drop_ledger: Mapping[str, Any],
    tokenizer_seal: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = _validate_metamath_drop_ledger(
        drop_ledger,
        tokenizer_seal=tokenizer_seal,
    )
    accounting = ledger["accounting"]
    if len(raw) != accounting["eligible_rows"]:
        raise IntegrationError(
            "Metamath eligible source occurrence count disagrees with drop ledger"
        )

    eligible_entries = []
    eligible_ids = set()
    eligible_theorems = set()
    expected_drop_binding = {
        "schema_version": ledger["schema_version"],
        "canonical_root_sha256": ledger["canonical_root_sha256"],
        "entries_root_sha256": ledger["entries_root_sha256"],
        "accounting": dict(accounting),
    }
    for item in raw:
        record = item.record
        row_id = record["id"]
        theorem = record["theorem"]
        if row_id in eligible_ids or theorem in eligible_theorems:
            raise IntegrationError(
                "Metamath eligible source occurrences contain duplicate identities"
            )
        eligible_ids.add(row_id)
        eligible_theorems.add(theorem)
        metadata = record.get("source_metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("drop_ledger") != expected_drop_binding
            or metadata.get("tokenizer_seal") != dict(tokenizer_seal)
            or metadata.get("tokenizer_root_sha256")
            != ledger["tokenizer_root_sha256"]
        ):
            raise IntegrationError(
                "Metamath row source metadata lacks the exact ledger/tokenizer binding"
            )
        eligible_entries.append(
            {
                "id": row_id,
                "theorem": theorem,
                "native_row_sha256": metamath_builder.native_row_sha256(record),
            }
        )

    dropped_ids = {entry["id"] for entry in ledger["entries"]}
    dropped_theorems = {entry["theorem"] for entry in ledger["entries"]}
    if eligible_ids & dropped_ids or eligible_theorems & dropped_theorems:
        raise IntegrationError(
            "Metamath overlength ledger entry was also emitted as eligible"
        )
    if len(eligible_entries) + len(ledger["entries"]) != accounting["source_rows"]:
        raise IntegrationError("Metamath source occurrence accounting is incomplete")

    eligible_entries.sort(key=lambda entry: (entry["id"], entry["theorem"]))
    native_occurrences = sorted(
        [
            *(
                {"source_disposition": "eligible", **entry}
                for entry in eligible_entries
            ),
            *(
                {
                    "source_disposition": "overlength",
                    "id": entry["id"],
                    "theorem": entry["theorem"],
                    "native_row_sha256": entry["native_row_sha256"],
                }
                for entry in ledger["entries"]
            ),
        ],
        key=lambda entry: (entry["id"], entry["theorem"]),
    )
    body = {
        "schema_version": "metamath-source-occurrences-v1",
        "ordering": "id-then-theorem-v1",
        "source_rows": accounting["source_rows"],
        "eligible_rows": accounting["eligible_rows"],
        "overlength_rows": accounting["dropped_rows"],
        "eligible_entries_root_sha256": _canonical_sha256(eligible_entries),
        "overlength_entries_root_sha256": ledger["entries_root_sha256"],
        "native_occurrences_root_sha256": _canonical_sha256(native_occurrences),
        "drop_ledger_root_sha256": ledger["canonical_root_sha256"],
        "tokenizer_root_sha256": ledger["tokenizer_root_sha256"],
    }
    return {**body, "source_occurrence_binding_root_sha256": _canonical_sha256(body)}


def _validate_metamath_source_accounting(
    source_accounting: Mapping[str, Any] | None,
    *,
    raw: tuple[_LineOccurrence, ...],
    train: Sequence[_LineOccurrence],
    evaluation: Sequence[_LineOccurrence],
    drops: Iterable[_NativeDrop | Mapping[str, Any]],
    drop_ledger: Mapping[str, Any],
    tokenizer_seal: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = _validate_metamath_drop_ledger(
        drop_ledger,
        tokenizer_seal=tokenizer_seal,
    )
    source_occurrences = _metamath_source_occurrence_binding(
        raw,
        drop_ledger=ledger,
        tokenizer_seal=tokenizer_seal,
    )
    drop_types = Counter()
    native_drop_count = 0
    for drop in drops:
        drop_type = (
            drop.drop_type if isinstance(drop, _NativeDrop) else drop.get("drop_type")
        )
        if not isinstance(drop_type, str) or not drop_type:
            raise IntegrationError("Metamath final typed drop is malformed")
        drop_types[drop_type] += 1
        native_drop_count += 1
    drop_types["overlength"] = ledger["accounting"]["dropped_rows"]
    final_accounted = (
        len(train)
        + len(evaluation)
        + native_drop_count
        + ledger["accounting"]["dropped_rows"]
    )
    if len(train) + len(evaluation) + native_drop_count != len(raw):
        raise IntegrationError("Metamath eligible route accounting is incomplete")
    if final_accounted != ledger["accounting"]["source_rows"]:
        raise IntegrationError("Metamath final source routing is incomplete")
    body = {
        **source_occurrences,
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "drop_types": dict(sorted(drop_types.items())),
        "accounted_rows": final_accounted,
    }
    expected = {**body, "accounting_root_sha256": _canonical_sha256(body)}
    if source_accounting is not None and dict(source_accounting) != expected:
        raise IntegrationError(
            "Metamath persisted source accounting or occurrence root is stale"
        )
    return expected


def _normalize_metamath_package(
    *,
    raw_output: Path,
    split_outputs: Mapping[str, Path],
    destination: Path,
    source_manifest: Mapping[str, Any],
    drop_ledger: Mapping[str, Any] | None = None,
    tokenizer_seal: Mapping[str, Any] | None = None,
) -> _FamilyPackage:
    required = {"train", "eval", "heldout"}
    if set(split_outputs) != required:
        raise IntegrationError("metamath split outputs are not exact")
    raw = _validate_rows(
        raw_output,
        family="metamath",
        source_manifest=source_manifest,
        label="builder/raw/metamath.jsonl",
    )
    heldout = _read_json(split_outputs["heldout"], "Metamath heldout manifest")
    _validate_family_heldout_contract(heldout, family="metamath")
    facts = heldout.get("facts")
    if (
        not isinstance(facts, list)
        or len(facts) != 500
        or heldout.get("requested_heldout") != 500
        or heldout.get("local_assumptions") is not True
    ):
        raise IntegrationError("Metamath builder did not select exactly 500 held facts")
    production = source_manifest.get("test_only") is False
    ledger = None
    source_occurrences = None
    if production:
        if drop_ledger is None or tokenizer_seal is None:
            raise IntegrationError(
                "Metamath production normalization requires the exact drop ledger "
                "and fixed tokenizer seal"
            )
        ledger = _validate_metamath_drop_ledger(
            drop_ledger,
            tokenizer_seal=tokenizer_seal,
        )
        source_occurrences = _metamath_source_occurrence_binding(
            raw,
            drop_ledger=ledger,
            tokenizer_seal=tokenizer_seal,
        )
        heldout_body = dict(heldout)
        heldout_root = heldout_body.pop("manifest_root_sha256", None)
        if heldout_root != metamath_builder.canonical_sha256(heldout_body):
            raise IntegrationError("Metamath heldout manifest root is stale")
        eligibility = heldout.get("eligibility")
        expected_eligibility = {
            "accounting": ledger["accounting"],
            "drop_ledger": {
                "path": "drops/metamath-overlength.json",
                "schema_version": ledger["schema_version"],
                "canonical_root_sha256": ledger["canonical_root_sha256"],
                "entries_root_sha256": ledger["entries_root_sha256"],
            },
            "max_text_plus_eos_tokens": metamath_builder.MAX_TEXT_PLUS_EOS_TOKENS,
            "tokenizer_seal": dict(tokenizer_seal),
            "tokenizer_root_sha256": ledger["tokenizer_root_sha256"],
        }
        if not isinstance(eligibility, Mapping) or any(
            eligibility.get(key) != value
            for key, value in expected_eligibility.items()
        ):
            raise IntegrationError(
                "Metamath heldout eligibility disagrees with the exact drop ledger"
            )
    held = set(facts)
    isolation_context = _metamath_isolation_context(raw, held)
    train_rows = []
    eval_rows = []
    drops = []
    split_eval_rows = []
    for item in raw:
        classification = _classify_metamath_route(item.record, isolation_context)
        if classification.disposition == "drop":
            if classification.drop_type is None:
                raise IntegrationError("Metamath drop classification lacks a type")
            drops.append(
                _NativeDrop(
                    raw_row=item.line_number,
                    drop_type=classification.drop_type,
                    details=classification.drop_details,
                )
            )
            split_eval_rows.append(item)
        elif classification.disposition == "eval":
            eval_rows.append(item)
            split_eval_rows.append(item)
        else:
            train_rows.append(item)
    if not train_rows or not eval_rows or not drops:
        raise IntegrationError("Metamath production routes must include train/eval/drop")
    builder_train = _exact_split_rows(
        split_outputs["train"],
        raw=raw,
        family="metamath",
        label="builder/split/train/metamath.jsonl",
    )
    builder_eval = _exact_split_rows(
        split_outputs["eval"],
        raw=raw,
        family="metamath",
        label="builder/split/eval/metamath.jsonl",
    )
    if [item.line_number for item in builder_train] != [
        item.line_number for item in train_rows
    ] or [item.line_number for item in builder_eval] != [
        item.line_number for item in split_eval_rows
    ]:
        raise IntegrationError("Metamath builder split disagrees with held-fact routes")
    if len(train_rows) + len(eval_rows) + len(drops) != len(raw):
        raise IntegrationError("Metamath eligible route accounting is incomplete")

    source_accounting = None
    if production:
        assert ledger is not None
        assert source_occurrences is not None
        partition = heldout.get("partition_accounting")
        expected_partition_rows = {
            "source_rows": ledger["accounting"]["source_rows"],
            "train_rows": len(builder_train),
            "eval_rows": len(builder_eval),
            "drop_rows": ledger["accounting"]["dropped_rows"],
            "drop_text_plus_eos_tokens": ledger["accounting"][
                "dropped_text_plus_eos_tokens"
            ],
            "source_text_plus_eos_tokens": ledger["accounting"][
                "source_text_plus_eos_tokens"
            ],
        }
        if not isinstance(partition, Mapping) or any(
            partition.get(key) != value
            for key, value in expected_partition_rows.items()
        ):
            raise IntegrationError(
                "Metamath heldout partition accounting disagrees with source routes"
            )
        train_tokens = partition.get("train_text_plus_eos_tokens")
        eval_tokens = partition.get("eval_text_plus_eos_tokens")
        if (
            not isinstance(train_tokens, int)
            or isinstance(train_tokens, bool)
            or not isinstance(eval_tokens, int)
            or isinstance(eval_tokens, bool)
            or train_tokens + eval_tokens
            != ledger["accounting"]["eligible_text_plus_eos_tokens"]
        ):
            raise IntegrationError("Metamath eligible token accounting is stale")
        source_accounting = _validate_metamath_source_accounting(
            None,
            raw=raw,
            train=train_rows,
            evaluation=eval_rows,
            drops=drops,
            drop_ledger=ledger,
            tokenizer_seal=tokenizer_seal,
        )
    destination.mkdir(parents=True)
    train = destination / "train.jsonl"
    evaluation = destination / "eval.jsonl"
    train.write_bytes(b"".join(item.raw_bytes for item in train_rows))
    evaluation.write_bytes(b"".join(item.raw_bytes for item in eval_rows))
    return _FamilyPackage(
        family="metamath",
        raw=raw_output,
        train=train,
        eval=evaluation,
        drops=tuple(drops),
        heldout=heldout,
        source_accounting=source_accounting,
        overlength_drop_ledger=ledger,
    )


def _normalize_isabelle_package(
    *,
    raw_output: Path,
    split_outputs: Mapping[str, Path],
    destination: Path,
    source_manifest: Mapping[str, Any],
) -> _FamilyPackage:
    required = {"train", "eval", "heldout"}
    if set(split_outputs) != required:
        raise IntegrationError("isabelle split outputs are not exact")
    raw = _validate_rows(
        raw_output,
        family="isabelle",
        source_manifest=source_manifest,
        label="builder/raw/isabelle.jsonl",
    )
    heldout = _read_json(split_outputs["heldout"], "Isabelle heldout manifest")
    _validate_family_heldout_contract(heldout, family="isabelle")
    facts = heldout.get("facts")
    statements = heldout.get("statements")
    if (
        not isinstance(facts, list)
        or len(facts) != 500
        or heldout.get("requested_heldout") != 500
        or not isinstance(statements, dict)
        or set(statements) != set(facts)
    ):
        raise IntegrationError(
            "Isabelle builder did not produce the accepted 500-fact transition split"
        )
    train_rows = _exact_split_rows(
        split_outputs["train"],
        raw=raw,
        family="isabelle",
        label="builder/split/train/isabelle.jsonl",
    )
    eval_rows = _exact_split_rows(
        split_outputs["eval"],
        raw=raw,
        family="isabelle",
        label="builder/split/eval/isabelle.jsonl",
    )
    train_numbers = {item.line_number for item in train_rows}
    eval_numbers = {item.line_number for item in eval_rows}
    if train_numbers & eval_numbers:
        raise IntegrationError("Isabelle builder routed one occurrence twice")
    direct_trajectories = {item.record["trajectory_id"] for item in eval_rows}

    try:
        from . import build_isabelle_shard as isabelle_builder
    except ImportError:  # pragma: no cover
        import build_isabelle_shard as isabelle_builder
    held_names_by_statement: defaultdict[str, set[str]] = defaultdict(set)
    held_by_anchor: defaultdict[str, list[str]] = defaultdict(list)
    for name, statement in statements.items():
        normalized = isabelle_builder.normalize_layout(statement)
        held_names_by_statement[normalized].add(name)
        anchor = isabelle_builder._statement_anchor(normalized)
        if anchor:
            held_by_anchor[anchor].append(normalized)
    trajectory_exposures: defaultdict[str, set[str]] = defaultdict(set)
    for item in raw:
        trajectory_exposures[item.record["trajectory_id"]].update(
            isabelle_builder._heldout_exposure_types(
                item.record,
                held_names=set(facts),
                held_names_by_statement=held_names_by_statement,
                held_by_anchor=held_by_anchor,
            )
        )
    type_map = {
        "own_proof_declaration": "heldout_own_proof",
        "local_statement": "heldout_local_statement",
        "target_state": "heldout_target_state",
    }
    priority = ("own_proof_declaration", "local_statement", "target_state")
    drops = []
    for item in raw:
        if item.line_number in train_numbers or item.line_number in eval_numbers:
            continue
        trajectory = item.record["trajectory_id"]
        if trajectory in direct_trajectories:
            drop_type = "heldout_trajectory_sibling"
            exposure = "direct-eval trajectory sibling"
        else:
            exposures = trajectory_exposures[trajectory]
            selected = next((kind for kind in priority if kind in exposures), None)
            if selected is None:
                raise IntegrationError(
                    "Isabelle split omitted a row without a typed trajectory exposure"
                )
            drop_type = type_map[selected]
            exposure = selected
        drops.append(
            _NativeDrop(
                raw_row=item.line_number,
                drop_type=drop_type,
                details={
                    "trajectory_id": trajectory,
                    "exposure": exposure,
                },
            )
        )
    if not train_rows or not eval_rows or not drops:
        raise IntegrationError("Isabelle production routes must include train/eval/drop")
    if len(train_rows) + len(eval_rows) + len(drops) != len(raw):
        raise IntegrationError("Isabelle production route accounting is incomplete")
    destination.mkdir(parents=True)
    train = destination / "train.jsonl"
    evaluation = destination / "eval.jsonl"
    train.write_bytes(b"".join(item.raw_bytes for item in train_rows))
    evaluation.write_bytes(b"".join(item.raw_bytes for item in eval_rows))
    return _FamilyPackage(
        family="isabelle",
        raw=raw_output,
        train=train,
        eval=evaluation,
        drops=tuple(drops),
        heldout=heldout,
    )


def _mml_contract_preflight_roots(
    contract: mml_holdout.ValidatedHoldoutContract,
) -> dict[str, str]:
    manifest = contract.manifest
    return {
        "manifest": contract.authoritative_root,
        "artifact_inventory": manifest["artifact_inventory_root_sha256"],
        "routes": manifest["route_plan_root_sha256"],
        "source": contract.source_root_sha256,
        "quality": manifest["quality_filter_root_sha256"],
        "schema": manifest["schema_generation_root_sha256"],
        "deduplication": manifest["deduplication_root_sha256"],
        "acceptance": manifest["acceptance_root_sha256"],
        "tokenizer": contract.tokenizer_root_sha256,
    }


def _validate_production_mml_contract_roots(
    contract: mml_holdout.ValidatedHoldoutContract,
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer_seal: Mapping[str, Any],
    policies: Mapping[str, Any],
) -> None:
    if (
        not isinstance(contract, mml_holdout.ValidatedHoldoutContract)
        or not contract.production
        or contract.test_only
    ):
        raise IntegrationError("persisted MML contract is not production-authoritative")
    manifest = contract.manifest
    if manifest.get("manifest_root_sha256") != contract.authoritative_root:
        raise IntegrationError("persisted MML authoritative manifest root mismatch")
    ordered_inputs = manifest.get("ordered_inputs")
    if not isinstance(ordered_inputs, list) or [
        record.get("shard") for record in ordered_inputs
    ] != list(MML_SIBLINGS):
        raise IntegrationError("persisted MML contract inputs are not exact and ordered")
    if not set(MML_SIBLINGS) <= set(source_manifests):
        raise IntegrationError("persisted MML contract lacks supplied raw-family manifests")

    approved = mml_holdout.production_source_policy()
    table = mml_holdout.PRODUCTION_SOURCE_IDENTITY_TABLE
    if (
        approved.test_only
        or set(approved.shards) != set(MML_SIBLINGS)
        or approved.deduplication_roots is None
    ):
        raise IntegrationError("approved production MML source policy is not exact")
    supplied_pins = policies.get("mml", {}).get("policy_pins")
    current_pins = mml_holdout.current_policy_pins()
    expected_pins = {
        "policy_sha256": current_pins.policy_sha256,
        "mapping_sha256": current_pins.mapping_sha256,
        "atp_deduplication_sha256": current_pins.atp_deduplication_sha256,
    }
    if supplied_pins != expected_pins:
        raise IntegrationError("persisted MML contract policy pins are stale")

    for record in ordered_inputs:
        family = record["shard"]
        approved_shard = approved.shards[family]
        expected_rows = table[family].get("input_rows")
        if record.get("logical_path") != f"raw/{family}.jsonl":
            raise IntegrationError(f"{family}: persisted MML raw path is substituted")
        if record.get("rows") != expected_rows:
            raise IntegrationError(
                f"{family}: persisted MML row count is stale; "
                f"expected {expected_rows:,}, got {record.get('rows')!r}"
            )
        if record.get("sha256") != approved_shard.input_sha256:
            raise IntegrationError(f"{family}: persisted MML raw input root mismatch")

        source_metadata = source_manifests[family].get("row_source_metadata")
        if not isinstance(source_metadata, Mapping):
            raise IntegrationError(f"{family}: supplied row source metadata is missing")
        expected_roots = {
            "source_manifest_root_sha256": approved_shard.source_manifest_root_sha256,
            "quality_filter_root_sha256": approved_shard.quality_filter_root_sha256,
            "schema_generation_root_sha256": (
                approved_shard.schema_generation_root_sha256
            ),
            "deduplication_root_sha256": approved.deduplication_roots[family],
            "acceptance_roots": dict(approved_shard.acceptance_roots),
        }
        labels = {
            "source_manifest_root_sha256": "source-manifest",
            "quality_filter_root_sha256": "quality-filter",
            "schema_generation_root_sha256": "schema-generation",
            "deduplication_root_sha256": "deduplication",
            "acceptance_roots": "acceptance",
        }
        for field, expected in expected_roots.items():
            if record.get(field) != expected:
                raise IntegrationError(
                    f"{family}: persisted MML {labels[field]} root mismatch"
                )
        for field in (
            "source_manifest_root_sha256",
            "quality_filter_root_sha256",
            "schema_generation_root_sha256",
        ):
            if source_metadata.get(field) != expected_roots[field]:
                raise IntegrationError(
                    f"{family}: supplied {labels[field]} root disagrees with "
                    "persisted MML contract"
                )
        expected_snapshots = [
            {"reference": item.reference, "sha256": item.sha256}
            for item in approved_shard.source_snapshots
        ]
        if record.get("source_snapshots") != expected_snapshots:
            raise IntegrationError(
                f"{family}: persisted MML source snapshot inventory mismatch"
            )
        if (
            contract.quality_filter_roots_by_shard.get(family)
            != expected_roots["quality_filter_root_sha256"]
            or contract.schema_generation_roots_by_shard.get(family)
            != expected_roots["schema_generation_root_sha256"]
            or contract.deduplication_roots_by_shard.get(family)
            != expected_roots["deduplication_root_sha256"]
            or contract.acceptance_roots_by_shard.get(family)
            != expected_roots["acceptance_roots"]
        ):
            raise IntegrationError(f"{family}: persisted typed MML roots are stale")

    if contract.source_root_sha256 != mml_holdout.source_root(ordered_inputs):
        raise IntegrationError("persisted MML source root mismatch")
    aggregate_specs = {
        "quality_filter_root_sha256": "quality_filter_root_sha256",
        "schema_generation_root_sha256": "schema_generation_root_sha256",
        "deduplication_root_sha256": "deduplication_root_sha256",
    }
    for manifest_field, record_field in aggregate_specs.items():
        expected = mml_holdout._json_sha256(
            [
                {"shard": record["shard"], record_field: record[record_field]}
                for record in ordered_inputs
            ]
        )
        if manifest.get(manifest_field) != expected:
            raise IntegrationError(f"persisted MML aggregate {record_field} mismatch")
    expected_acceptance = mml_holdout._json_sha256(
        [
            {
                "shard": record["shard"],
                "acceptance_roots": record["acceptance_roots"],
            }
            for record in ordered_inputs
        ]
    )
    if manifest.get("acceptance_root_sha256") != expected_acceptance:
        raise IntegrationError("persisted MML aggregate acceptance root mismatch")
    tokenizer_root = mml_holdout._json_sha256(tokenizer_seal)
    if (
        contract.tokenizer_root_sha256 != tokenizer_root
        or manifest.get("tokenizer_root_sha256") != tokenizer_root
    ):
        raise IntegrationError("persisted MML tokenizer root mismatch")
    for label, root in _mml_contract_preflight_roots(contract).items():
        _require_digest(root, f"persisted MML {label} root", production=True)


def _load_production_mml_contract(
    contract_root: str | os.PathLike[str],
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer_seal: Mapping[str, Any],
    policies: Mapping[str, Any],
    raw_paths: Mapping[str, Path] | None = None,
) -> mml_holdout.ValidatedHoldoutContract:
    requested = _require_readable_path(
        Path(contract_root),
        label="persisted authoritative MML contract",
        regular_file=False,
    )
    if not requested.is_dir():
        raise IntegrationError("persisted authoritative MML contract must be a directory")
    try:
        contract = mml_holdout.load_holdout_contract(requested, production=True)
    except (
        mml_holdout.HoldoutError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise IntegrationError(
            f"persisted authoritative MML contract validation failed: {error}"
        ) from error
    if contract.root.resolve() != requested:
        raise IntegrationError("persisted authoritative MML contract path was substituted")
    _validate_production_mml_contract_roots(
        contract,
        source_manifests=source_manifests,
        tokenizer_seal=tokenizer_seal,
        policies=policies,
    )
    if raw_paths is None:
        _trim_unused_heap()
        return contract
    if set(raw_paths) != set(MML_SIBLINGS):
        raise IntegrationError("persisted MML validation requires exact raw family inputs")
    records = {
        record["shard"]: record for record in contract.manifest["ordered_inputs"]
    }
    artifact_inodes = set()
    for artifact in contract.artifacts.values():
        try:
            metadata = artifact.path.stat(follow_symlinks=False)
        except OSError as error:
            raise IntegrationError(
                f"persisted MML artifact disappeared during raw validation: {artifact.path}"
            ) from error
        artifact_inodes.add((metadata.st_dev, metadata.st_ino))
    for family in MML_SIBLINGS:
        path = Path(raw_paths[family])
        resolved = _require_readable_path(
            path,
            label=f"{family} raw MML input",
            regular_file=True,
        )
        if resolved.is_relative_to(requested):
            raise IntegrationError(f"{family}: raw MML path substitutes a contract artifact")
        metadata = resolved.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) in artifact_inodes:
            raise IntegrationError(f"{family}: raw MML input hard-links a contract artifact")
        actual_sha256, actual_bytes, actual_rows = _sha256_jsonl_nofollow(
            resolved,
            label=f"{family} raw MML input",
        )
        expected = records[family]
        if actual_sha256 != expected["sha256"]:
            raise IntegrationError(f"{family}: raw MML input root mismatch")
        if actual_rows != expected["rows"]:
            raise IntegrationError(
                f"{family}: raw MML row count mismatch; "
                f"expected {expected['rows']:,}, got {actual_rows:,}"
            )
        inventory_bytes = sum(
            artifact.bytes
            for relative, artifact in contract.artifacts.items()
            if relative in {
                f"shards/{family}.jsonl",
                f"eval/{family}.jsonl",
                f"dropped/{family}.jsonl",
            }
        )
        if actual_bytes != inventory_bytes:
            raise IntegrationError(
                f"{family}: raw MML byte count disagrees with partition inventory"
            )
    _trim_unused_heap()
    return contract


def _validate_enigma_low_tier_input_binding(
    argv: Sequence[str],
    *,
    tokenizer_path: Path,
    tokenizer_seal: Mapping[str, Any],
    mml_contract: mml_holdout.ValidatedHoldoutContract,
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = _preflight_enigma_low_tier_paths(argv)
    if set(paths) != {"enigma_low_tier_base", "tokenizer_json"}:
        raise IntegrationError("production ENIGMA requires both approved low-tier inputs")
    try:
        try:
            from . import build_atp_shard as atp_builder
        except ImportError:  # pragma: no cover
            import build_atp_shard as atp_builder

        expected_base = dict(
            atp_builder.ENIGMA_LOW_TIER_SOURCE_CONTRACT["accepted_base"]
        )
        expected_tokenizer = dict(
            atp_builder.ENIGMA_LOW_TIER_SOURCE_CONTRACT["tokenizer"]
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise IntegrationError("approved ENIGMA low-tier source contract is missing") from error

    base_shards = _require_readable_path(
        paths["enigma_low_tier_base"] / "shards",
        label="ENIGMA accepted base shard directory",
        regular_file=False,
    )
    if not base_shards.is_dir():
        raise IntegrationError("ENIGMA accepted base shard directory is invalid")
    base_shard = base_shards / "enigma.jsonl"
    base_sha256, base_bytes, base_rows = _sha256_jsonl_nofollow(
        base_shard,
        label="ENIGMA accepted base shard",
    )
    actual_base = {
        "bytes": base_bytes,
        "rows": base_rows,
        "sha256": base_sha256,
    }
    if actual_base != expected_base:
        raise IntegrationError(
            "ENIGMA accepted base contract mismatch: "
            f"expected {expected_base}, got {actual_base}"
        )

    requested_tokenizer = paths["tokenizer_json"]
    canonical_tokenizer = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    )
    canonical_tokenizer = _require_readable_path(
        canonical_tokenizer,
        label="canonical ENIGMA low-tier tokenizer JSON",
        regular_file=True,
    )
    if requested_tokenizer != canonical_tokenizer:
        raise IntegrationError(
            "ENIGMA low-tier tokenizer path substitutes the canonical tokenizer input"
        )
    tokenizer_json_sha256 = _sha256_regular_file_nofollow(
        requested_tokenizer,
        label="ENIGMA low-tier tokenizer JSON",
    )
    tokenizer_config = _require_readable_path(
        requested_tokenizer.parent / "tokenizer_config.json",
        label="ENIGMA low-tier tokenizer config",
        regular_file=True,
    )
    tokenizer_config_sha256 = _sha256_regular_file_nofollow(
        tokenizer_config,
        label="ENIGMA low-tier tokenizer config",
    )
    if (
        tokenizer_json_sha256 != tokenizer_seal.get("tokenizer_json_sha256")
        or tokenizer_config_sha256 != tokenizer_seal.get("tokenizer_config_sha256")
        or tokenizer_json_sha256 != expected_tokenizer.get("tokenizer_json_sha256")
        or tokenizer_config_sha256
        != expected_tokenizer.get("tokenizer_config_sha256")
    ):
        raise IntegrationError("ENIGMA low-tier tokenizer SHA-256 contract mismatch")

    enigma_input = next(
        record
        for record in mml_contract.manifest["ordered_inputs"]
        if record["shard"] == "enigma"
    )
    expected_final_rows = mml_holdout.PRODUCTION_SOURCE_IDENTITY_TABLE["enigma"][
        "input_rows"
    ]
    if enigma_input.get("rows") != expected_final_rows:
        raise IntegrationError(
            "ENIGMA persisted contract is base-only or stale; "
            f"expected {expected_final_rows:,} rows"
        )
    acceptance_roots = dict(mml_contract.acceptance_roots_by_shard["enigma"])
    acceptance_root = _canonical_sha256(acceptance_roots)
    accepted_base_root = _canonical_sha256(actual_base)
    body = {
        "schema_version": ENIGMA_LOW_TIER_INPUT_BINDING_SCHEMA,
        "accepted_base_path": str(paths["enigma_low_tier_base"]),
        "accepted_base": actual_base,
        "accepted_base_root_sha256": accepted_base_root,
        "final_enigma_rows": expected_final_rows,
        "final_enigma_sha256": enigma_input["sha256"],
        "acceptance_roots": acceptance_roots,
        "acceptance_root_sha256": acceptance_root,
        "contract_acceptance_root_sha256": mml_contract.manifest[
            "acceptance_root_sha256"
        ],
        "tokenizer_path": str(requested_tokenizer),
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "tokenizer_root_sha256": mml_contract.tokenizer_root_sha256,
    }
    binding = {**body, "binding_root_sha256": _canonical_sha256(body)}
    roots = {
        "accepted_base_sha256": base_sha256,
        "accepted_base_root_sha256": accepted_base_root,
        "acceptance_root_sha256": mml_contract.manifest["acceptance_root_sha256"],
        "enigma_acceptance_root_sha256": acceptance_root,
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "tokenizer_root_sha256": mml_contract.tokenizer_root_sha256,
        "binding_root_sha256": binding["binding_root_sha256"],
    }
    return paths, roots, binding


def _bind_enigma_low_tier_source_manifest(
    manifest: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    body = dict(binding)
    root = body.pop("binding_root_sha256", None)
    if (
        body.get("schema_version") != ENIGMA_LOW_TIER_INPUT_BINDING_SCHEMA
        or root != _canonical_sha256(body)
    ):
        raise IntegrationError("ENIGMA low-tier generation input binding is invalid")
    result = _json_copy(manifest)
    acceptance = result.get("source_verifier_acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("accepted") is not True:
        raise IntegrationError("ENIGMA source verifier acceptance is missing")
    acceptance["generation_input_binding"] = dict(binding)
    result["manifest_root_sha256"] = _source_manifest_root(result)
    return result


def _build_production_mml_contract(
    *,
    raw_paths: Mapping[str, Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer_seal: Mapping[str, Any],
    tokenizer_path: Path,
    policies: Mapping[str, Any],
    output: Path,
) -> mml_holdout.ValidatedHoldoutContract:
    approved = mml_holdout.production_source_policy()
    if set(approved.shards) != set(MML_SIBLINGS) or approved.test_only:
        raise IntegrationError("approved production MML source policy is not exact")
    sources = {}
    for family in MML_SIBLINGS:
        manifest = source_manifests[family]
        metadata = manifest["row_source_metadata"]
        approved_shard = approved.shards[family]
        if (
            metadata["source_manifest_root_sha256"]
            != approved_shard.source_manifest_root_sha256
            or metadata["quality_filter_root_sha256"]
            != approved_shard.quality_filter_root_sha256
            or metadata["schema_generation_root_sha256"]
            != approved_shard.schema_generation_root_sha256
        ):
            raise IntegrationError(
                f"{family}: supplied source roots disagree with approved MML policy"
            )
        sources[family] = mml_holdout.PathShardSource(
            name=family,
            logical_path=f"raw/{family}.jsonl",
            path=raw_paths[family],
            expected_input_sha256=approved_shard.input_sha256,
            source_snapshots=approved_shard.source_snapshots,
            source_manifest_root_sha256=approved_shard.source_manifest_root_sha256,
            quality_filter_root_sha256=approved_shard.quality_filter_root_sha256,
            schema_generation_root_sha256=approved_shard.schema_generation_root_sha256,
        )
    try:
        try:
            from .build_isabelle_shard import (
                _tokenizer_metadata,
                load_vendored_tokenizer,
            )
        except ImportError:  # pragma: no cover
            from build_isabelle_shard import (
                _tokenizer_metadata,
                load_vendored_tokenizer,
            )

        backend = load_vendored_tokenizer(tokenizer_path)
        actual_seal = _tokenizer_metadata(backend)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise IntegrationError(f"sealed tokenizer load failed: {error}") from error
    if dict(actual_seal) != dict(tokenizer_seal):
        raise IntegrationError("loaded tokenizer bytes disagree with tokenizer seal")
    tokenizer = mml_holdout.TokenizerSeam(
        seal=actual_seal,
        count_text_plus_eos=lambda text: (
            len(backend.encode(text, add_special_tokens=False).ids) + 1
        ),
    )
    pins_raw = policies["mml"]["policy_pins"]
    pins = mml_holdout.PolicyPins(
        policy_sha256=pins_raw["policy_sha256"],
        mapping_sha256=pins_raw["mapping_sha256"],
        atp_deduplication_sha256=pins_raw["atp_deduplication_sha256"],
    )
    plan = mml_holdout.plan_semantic_holdout(
        sources,
        tokenizer=tokenizer,
        policy_pins=pins,
        source_policy=approved,
    )
    mml_holdout.write_partition_atomically(plan, sources=sources, output=output)
    return mml_holdout.load_holdout_contract(output, production=True)


def _verify_external_mizar_rows(
    index_path: Path,
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    raw_paths: Mapping[str, Path],
) -> None:
    actual = hashlib.sha256(index_path.read_bytes()).hexdigest()
    for family in ("mizar", "thproofs"):
        expected = source_manifests[family]["row_source_metadata"]["index_roots"][
            "semantic_index_sha256"
        ]
        if actual != expected:
            raise IntegrationError(f"{family}: current Mizar index root is stale")
    try:
        try:
            from .build_mizar_human_shard import resolve_global_citations
            from .build_thproofs_shard import resolve_index_references
            from .mizar_current_index import MizarIndex
        except ImportError:  # pragma: no cover
            from build_mizar_human_shard import resolve_global_citations
            from build_thproofs_shard import resolve_index_references
            from mizar_current_index import MizarIndex

        with MizarIndex(index_path) as index:
            statements = index.statement_map()
            for family in ("mizar", "thproofs"):
                rows = _read_occurrences(
                    raw_paths[family],
                    label=f"builder/raw/{family}.jsonl",
                )
                for item in rows:
                    record = item.record
                    if statements.get(record["theorem"]) != record["goal"]:
                        raise IntegrationError(
                            f"raw/{family}:{item.line_number}: "
                            "current Mizar goal/index mismatch"
                        )
                    for name, statement in record["facts"].items():
                        if statements.get(name) != statement:
                            raise IntegrationError(
                                f"raw/{family}:{item.line_number}: "
                                f"current Mizar fact/index mismatch {name}"
                            )
                    if family == "mizar":
                        resolution = resolve_global_citations(
                            record["target"],
                            index,
                            theorem=record["theorem"],
                        )
                        references = list(resolution.references)
                        unresolved = list(resolution.unresolved)
                        citations_disagree = references != record["cited"]
                    else:
                        references, unresolved = resolve_index_references(
                            record["target"],
                            index,
                            statements,
                            theorem=record["theorem"],
                        )
                        citations_disagree = set(references) != set(record["cited"])
                    if unresolved or citations_disagree:
                        raise IntegrationError(
                            f"raw/{family}:{item.line_number}: "
                            "current Mizar target/reference mismatch"
                        )
    except IntegrationError:
        raise
    except Exception as error:
        raise IntegrationError(
            f"current Mizar semantic index validation failed: {error}"
        ) from error


def build_production_generation(
    *,
    corpus_root: str | os.PathLike[str],
    work_root: str | os.PathLike[str],
    generation_id: str,
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer_seal: Mapping[str, Any],
    tokenizer_path: str | os.PathLike[str],
    metamath_drop_ledger: Mapping[str, Any],
    policies: Mapping[str, Any],
    mizar_semantic_index: str | os.PathLike[str],
    mml_contract_root: str | os.PathLike[str],
    forbidden_legacy_paths: Sequence[str | os.PathLike[str]] = (),
    _fault_point: str | None = None,
) -> ProductionBuildResult:
    """Run accepted builders and ingest one persisted production MML contract."""

    root = Path(corpus_root)
    work = Path(work_root)
    if _fault_point is not None and _fault_point not in synthetic_fault_points():
        raise IntegrationError(f"unknown production fault point {_fault_point!r}")

    def inject(point: str) -> None:
        if _fault_point == point:
            raise IntegrationError(f"injected fault at {point}")

    _reject_legacy_layout(root)
    _resolved_root, resolved_work = _validate_secure_production_roots(
        corpus_root=root,
        work_root=work,
    )
    forbidden = tuple(Path(path).resolve() for path in forbidden_legacy_paths)
    if any(
        resolved_work == path or resolved_work.is_relative_to(path)
        for path in forbidden
    ):
        raise IntegrationError("production work root overlaps a legacy corpus")
    if set(source_manifests) != set(EXACT_SIBLINGS):
        raise IntegrationError("production requires exact six-family source manifests")
    validated_sources = {
        family: _validate_source_manifest(
            source_manifests[family],
            family=family,
            production=True,
        )
        for family in EXACT_SIBLINGS
    }
    blockers = production_blockers(validated_sources)
    if blockers:
        raise IntegrationError("real full build blocked: " + "; ".join(blockers))
    builder_configs = {
        family: _validate_production_builder_config(
            validated_sources[family],
            family=family,
        )
        for family in EXACT_SIBLINGS
    }
    tokenizer = _validate_tokenizer_seal(tokenizer_seal)
    validated_policies = _validate_policies(policies, production=True)
    index_path = Path(mizar_semantic_index)
    if not index_path.is_file() or index_path.is_symlink():
        raise IntegrationError("current Mizar semantic index is missing or unsafe")
    tokenizer_artifact = Path(tokenizer_path)
    if not tokenizer_artifact.exists() or tokenizer_artifact.is_symlink():
        raise IntegrationError("sealed tokenizer path is missing or unsafe")
    ledger = _validate_metamath_drop_ledger(
        metamath_drop_ledger,
        tokenizer_seal=tokenizer,
    )
    _validate_builder_native_source_metadata(
        validated_sources["metamath"],
        family="metamath",
        validated_paths={},
        validated_roots={},
        tokenizer_metadata=tokenizer,
        tokenizer_path=tokenizer_artifact,
        metamath_drop_ledger=ledger,
    )
    persisted_contract = _load_production_mml_contract(
        mml_contract_root,
        source_manifests=validated_sources,
        tokenizer_seal=tokenizer,
        policies=validated_policies,
    )
    persisted_contract_root = persisted_contract.authoritative_root
    _validate_external_mml_contract_location(
        persisted_contract.root,
        corpus_root=root,
        work_root=resolved_work,
    )
    _enigma_paths, _enigma_roots, enigma_binding = (
        _validate_enigma_low_tier_input_binding(
            builder_configs["enigma"]["raw"]["argv"],
            tokenizer_path=tokenizer_artifact,
            tokenizer_seal=tokenizer,
            mml_contract=persisted_contract,
        )
    )
    validated_sources = dict(validated_sources)
    validated_sources["enigma"] = _validate_source_manifest(
        _bind_enigma_low_tier_source_manifest(
            validated_sources["enigma"],
            enigma_binding,
        ),
        family="enigma",
        production=True,
    )
    source_generation_id = _source_generation_id(
        validated_sources,
        tokenizer,
        validated_policies,
        mml_contract_root_sha256=persisted_contract_root,
    )
    plan = make_generation_plan(
        generation_id=generation_id,
        source_generation_id=source_generation_id,
    )
    # The production contract is several GiB when materialized. Release the
    # preflight copy before builders run; the boundary reload below revalidates
    # the persisted bytes against their freshly rebuilt raw inputs.
    del persisted_contract
    _trim_unused_heap()
    run_root = Path(
        tempfile.mkdtemp(prefix=f"p3-{generation_id}.", dir=str(resolved_work))
    )
    builder_roots: list[Path] = []

    def producer(writer) -> None:
        packages: dict[str, _FamilyPackage] = {}
        raw_paths = {}
        for family in EXACT_SIBLINGS:
            family_root = run_root / "builders" / family
            family_root.mkdir(parents=True)
            builder_roots.append(family_root)
            inject(f"raw_builder:{family}")
            raw_outputs = _run_builder_stage(
                family=family,
                stage="raw",
                specification=builder_configs[family]["raw"],
                output_root=family_root / "raw-build",
                corpus_root=root,
                forbidden_legacy_paths=forbidden,
            )
            inject(f"builder_complete:{family}")
            if set(raw_outputs) != {"raw"}:
                raise IntegrationError(f"{family}: raw builder output must be exact")
            raw_paths[family] = raw_outputs["raw"]
            if family in MML_SIBLINGS:
                continue
            inject(f"split_builder:start:{family}")
            split_outputs = _run_builder_stage(
                family=family,
                stage="split",
                specification=builder_configs[family]["split"],
                output_root=family_root / "split-build",
                corpus_root=root,
                forbidden_legacy_paths=forbidden,
            )
            inject(f"split_builder:complete:{family}")
            if family == "metamath":
                inject("normalization:metamath")
                packages[family] = _normalize_metamath_package(
                    raw_output=raw_outputs["raw"],
                    split_outputs=split_outputs,
                    destination=family_root / "normalized",
                    source_manifest=validated_sources[family],
                    drop_ledger=ledger,
                    tokenizer_seal=tokenizer,
                )
                inject("partition:metamath")
                inject("family_split:complete:metamath")
            else:
                inject("normalization:isabelle")
                packages[family] = _normalize_isabelle_package(
                    raw_output=raw_outputs["raw"],
                    split_outputs=split_outputs,
                    destination=family_root / "normalized",
                    source_manifest=validated_sources[family],
                )
                inject("partition:isabelle")
                inject("family_split:complete:isabelle")
        _verify_external_mizar_rows(
            index_path,
            source_manifests=validated_sources,
            raw_paths=raw_paths,
        )
        inject("mml_partition")
        inject("split_builder:start:mml")
        contract = _load_production_mml_contract(
            mml_contract_root,
            source_manifests=validated_sources,
            tokenizer_seal=tokenizer,
            policies=validated_policies,
            raw_paths={family: raw_paths[family] for family in MML_SIBLINGS},
        )
        if contract.authoritative_root != persisted_contract_root:
            raise IntegrationError(
                "persisted authoritative MML contract changed during generation"
            )
        inject("split_builder:complete:mml")
        for family in MML_SIBLINGS:
            packages[family] = _FamilyPackage(
                family=family,
                raw=raw_paths[family],
                train=contract.family_paths[family].train,
                eval=contract.family_paths[family].eval,
                drops=(),
                heldout=contract.manifest,
            )
            inject(f"family_split:complete:{family}")
        inject("precheck")
        _write_transaction_payload(
            writer,
            source_manifests=validated_sources,
            tokenizer=tokenizer,
            policies=validated_policies,
            packages=packages,
            mml_contract=contract,
            fault_injector=inject,
        )

    try:
        published = GenerationCoordinator(root).publish(plan, producer)
    except BaseException as error:
        _builder_quarantine(
            run_root,
            work_root=resolved_work,
            generation_id=generation_id,
            error=error,
        )
        if isinstance(error, IntegrationError):
            raise
        raise IntegrationError(
            f"production six-family generation {generation_id} failed: {error}"
        ) from error
    return ProductionBuildResult(
        published=published,
        builder_output_roots=tuple(builder_roots),
    )


def _materialize_mml_contract(
    generation_path: Path,
    wrapper: Mapping[str, Any],
    destination: Path,
) -> None:
    for directory in ("shards", "eval", "dropped", "heldout", "sidecars"):
        (destination / directory).mkdir(parents=True, exist_ok=True)
    manifest = wrapper["contract_manifest"]
    routes = manifest["row_routes"]
    for family in MML_SIBLINGS:
        raw_lines = (generation_path / "raw" / f"{family}.jsonl").read_bytes().splitlines(
            keepends=True
        )
        for disposition, directory in (
            ("train", "shards"),
            ("eval", "eval"),
            ("drop", "dropped"),
        ):
            payload = b"".join(
                raw_lines[route["line_number"] - 1]
                for route in routes[family]
                if route["disposition"] == disposition
            )
            (destination / directory / f"{family}.jsonl").write_bytes(payload)
    def pretty(value: Any) -> bytes:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode()
    (destination / "heldout" / "mml.json").write_bytes(pretty(manifest))
    for family in ("mizar", "atp"):
        (destination / "heldout" / f"{family}.json").write_bytes(
            pretty(wrapper["projections"][family])
        )
    for key, filename in (
        ("eval_exposure", "eval_exposure.jsonl"),
        ("drop_reasons", "drop_reasons.jsonl"),
    ):
        (destination / "sidecars" / filename).write_bytes(
            b"".join(_canonical_json_bytes(record) for record in wrapper[key])
        )


def _verify_current_mizar_index(
    path: Path,
    *,
    source_manifests: Mapping[str, Mapping[str, Any]],
    generation_path: Path,
) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    for family in ("mizar", "thproofs"):
        expected = source_manifests[family]["row_source_metadata"]["index_roots"][
            "semantic_index_sha256"
        ]
        if actual != expected:
            raise IntegrationError(f"{family}: supplied current Mizar index root is stale")
    try:
        try:
            from .build_mizar_human_shard import resolve_global_citations
            from .build_thproofs_shard import resolve_index_references
            from .mizar_current_index import MizarIndex
        except ImportError:  # pragma: no cover
            from build_mizar_human_shard import resolve_global_citations
            from build_thproofs_shard import resolve_index_references
            from mizar_current_index import MizarIndex

        with MizarIndex(path) as index:
            statements = index.statement_map()
            for family in ("mizar", "thproofs"):
                for split in ("shards", "eval"):
                    rows = _read_occurrences(
                        generation_path / split / f"{family}.jsonl",
                        label=f"{split}/{family}.jsonl",
                    )
                    for item in rows:
                        record = item.record
                        if statements.get(record["theorem"]) != record["goal"]:
                            raise IntegrationError(
                                f"{split}/{family}:{item.line_number}: "
                                "current Mizar goal/index mismatch"
                            )
                        for name, statement in record["facts"].items():
                            if statements.get(name) != statement:
                                raise IntegrationError(
                                    f"{split}/{family}:{item.line_number}: "
                                    f"current Mizar fact/index mismatch {name}"
                                )
                        if family == "mizar":
                            resolution = resolve_global_citations(
                                record["target"],
                                index,
                                theorem=record["theorem"],
                            )
                            references = list(resolution.references)
                            unresolved = list(resolution.unresolved)
                            citations_disagree = references != record["cited"]
                        else:
                            references, unresolved = resolve_index_references(
                                record["target"],
                                index,
                                statements,
                                theorem=record["theorem"],
                            )
                            citations_disagree = set(references) != set(record["cited"])
                        if unresolved or citations_disagree:
                            raise IntegrationError(
                                f"{split}/{family}:{item.line_number}: "
                                "current Mizar target/reference mismatch"
                            )
    except IntegrationError:
        raise
    except Exception as error:  # noqa: BLE001 - normalize index backend failures.
        raise IntegrationError(f"current Mizar semantic index validation failed: {error}")


def _read_route_ledger(generation_path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    routes = {family: {} for family in EXACT_SIBLINGS}
    for item in _read_occurrences(
        generation_path / "ROUTES.jsonl",
        label="transaction route ledger",
    ):
        route = item.record
        family = route.get("sibling")
        raw_row = route.get("raw_row")
        if family not in routes or not isinstance(raw_row, int) or raw_row < 1:
            raise IntegrationError("transaction route ledger has an invalid raw identity")
        if raw_row in routes[family]:
            raise IntegrationError(
                f"{family}: duplicate occurrence in transaction route ledger"
            )
        routes[family][raw_row] = route
    return routes


def _read_drop_ledger(
    generation_path: Path,
    family: str,
) -> dict[int, dict[str, Any]]:
    drops = {}
    for item in _read_occurrences(
        generation_path / "sidecars" / "drops" / f"{family}.jsonl",
        label=f"{family} typed drop ledger",
    ):
        record = item.record
        raw_row = record.get("raw_row")
        if record.get("schema_version") != DROP_SCHEMA:
            raise IntegrationError(f"{family}: typed drop schema is stale")
        if not isinstance(raw_row, int) or raw_row < 1 or raw_row in drops:
            raise IntegrationError(f"{family}: typed drop raw identity is invalid")
        drops[raw_row] = record
    return drops


def _assert_semantic_route(
    *,
    family: str,
    raw_row: int,
    expected_disposition: str,
    expected_drop_type: str | None,
    route: Mapping[str, Any],
    drops: Mapping[int, Mapping[str, Any]],
    detail: str,
) -> None:
    if route.get("disposition") != expected_disposition:
        raise IntegrationError(
            f"{family}: {detail} route mismatch at raw row {raw_row}; "
            f"expected {expected_disposition}"
        )
    if expected_disposition == "drop":
        drop = drops.get(raw_row)
        if (
            route.get("drop_type") != expected_drop_type
            or drop is None
            or drop.get("drop_type") != expected_drop_type
            or drop.get("occurrence_id") != route.get("occurrence_id")
        ):
            raise IntegrationError(
                f"{family}: {detail} typed drop mismatch at raw row {raw_row}"
            )
    elif raw_row in drops:
        raise IntegrationError(
            f"{family}: non-drop route has a typed drop at raw row {raw_row}"
        )


def _verify_metamath_isolation(
    *,
    raw: tuple[_LineOccurrence, ...],
    train: tuple[_LineOccurrence, ...],
    evaluation: tuple[_LineOccurrence, ...],
    heldout: Mapping[str, Any],
    routes: Mapping[int, Mapping[str, Any]],
    drops: Mapping[int, Mapping[str, Any]],
) -> None:
    contract = heldout.get("contract")
    held = set(contract.get("facts", ())) if isinstance(contract, Mapping) else set()
    isolation_context = _metamath_isolation_context(raw, held)
    if set(routes) != set(range(1, len(raw) + 1)):
        raise IntegrationError("Metamath route inventory is incomplete")

    for item in raw:
        classification = _classify_metamath_route(item.record, isolation_context)
        _assert_semantic_route(
            family="metamath",
            raw_row=item.line_number,
            expected_disposition=classification.disposition,
            expected_drop_type=classification.drop_type,
            route=routes[item.line_number],
            drops=drops,
            detail=classification.detail,
        )
    for split, records in (("train", train), ("eval", evaluation)):
        for item in records:
            classification = _classify_metamath_route(item.record, isolation_context)
            if classification.disposition != split:
                raise IntegrationError(
                    f"Metamath {classification.detail} violation in actual {split} row "
                    f"{item.line_number}"
                )


def _verify_isabelle_isolation(
    *,
    raw: tuple[_LineOccurrence, ...],
    train: tuple[_LineOccurrence, ...],
    evaluation: tuple[_LineOccurrence, ...],
    heldout: Mapping[str, Any],
    routes: Mapping[int, Mapping[str, Any]],
    drops: Mapping[int, Mapping[str, Any]],
) -> None:
    contract = heldout.get("contract")
    held = set(contract.get("facts", ())) if isinstance(contract, Mapping) else set()
    if not held:
        raise IntegrationError("Isabelle held-fact contract is empty")
    try:
        from . import build_isabelle_shard as isabelle_builder
    except ImportError:  # pragma: no cover
        import build_isabelle_shard as isabelle_builder

    statements = {
        name: statement
        for item in raw
        for name, statement in item.record["facts"].items()
        if name in held
    }
    if set(statements) != held:
        raise IntegrationError("Isabelle held statements cannot be reconstructed")
    held_names_by_statement: defaultdict[str, set[str]] = defaultdict(set)
    held_by_anchor: defaultdict[str, list[str]] = defaultdict(list)
    for name, statement in statements.items():
        normalized = isabelle_builder.normalize_layout(statement)
        held_names_by_statement[normalized].add(name)
        anchor = isabelle_builder._statement_anchor(normalized)
        if anchor:
            held_by_anchor[anchor].append(normalized)
    direct_rows = {
        item.line_number
        for item in raw
        if set(item.record["facts"]) & held or set(item.record["cited"]) & held
    }
    direct_trajectories = {
        item.record["trajectory_id"]
        for item in raw
        if item.line_number in direct_rows
    }
    trajectory_exposures: defaultdict[str, set[str]] = defaultdict(set)
    for item in raw:
        trajectory_exposures[item.record["trajectory_id"]].update(
            isabelle_builder._heldout_exposure_types(
                item.record,
                held_names=held,
                held_names_by_statement=held_names_by_statement,
                held_by_anchor=held_by_anchor,
            )
        )
    type_map = {
        "own_proof_declaration": "heldout_own_proof",
        "local_statement": "heldout_local_statement",
        "target_state": "heldout_target_state",
    }
    if set(routes) != set(range(1, len(raw) + 1)):
        raise IntegrationError("Isabelle route inventory is incomplete")
    for item in raw:
        trajectory = item.record["trajectory_id"]
        if item.line_number in direct_rows:
            disposition, drop_type = "eval", None
            detail = "Isabelle direct held fact"
        elif trajectory in direct_trajectories:
            disposition, drop_type = "drop", "heldout_trajectory_sibling"
            detail = "Isabelle held trajectory sibling"
        else:
            exposure = next(
                (
                    value
                    for value in (
                        "own_proof_declaration",
                        "local_statement",
                        "target_state",
                    )
                    if value in trajectory_exposures[trajectory]
                ),
                None,
            )
            if exposure is None:
                disposition, drop_type = "train", None
                detail = "Isabelle held isolation"
            else:
                disposition, drop_type = "drop", type_map[exposure]
                detail = f"Isabelle {exposure}"
        _assert_semantic_route(
            family="isabelle",
            raw_row=item.line_number,
            expected_disposition=disposition,
            expected_drop_type=drop_type,
            route=routes[item.line_number],
            drops=drops,
            detail=detail,
        )
    eval_trajectories = {item.record["trajectory_id"] for item in evaluation}
    for item in evaluation:
        if not (
            set(item.record["facts"]) & held or set(item.record["cited"]) & held
        ):
            raise IntegrationError(
                f"Isabelle eval row {item.line_number} lacks a direct held fact"
            )
    for item in train:
        trajectory = item.record["trajectory_id"]
        exposures = isabelle_builder._heldout_exposure_types(
            item.record,
            held_names=held,
            held_names_by_statement=held_names_by_statement,
            held_by_anchor=held_by_anchor,
        )
        if (
            set(item.record["facts"]) & held
            or set(item.record["cited"]) & held
            or trajectory in direct_trajectories
            or trajectory in eval_trajectories
            or exposures
        ):
            raise IntegrationError(
                f"Isabelle held trajectory/statement isolation violation in train "
                f"row {item.line_number}"
            )


def _validate_declared_json_outputs(generation_path: Path) -> None:
    plan = make_generation_plan(
        generation_id="json-contract-verification",
        source_generation_id="json-contract-verification",
    )
    for specification in plan.outputs:
        if not specification.path.endswith(".json"):
            continue
        if not isinstance(specification.validator, JsonObjectValidator):
            raise IntegrationError(
                f"{specification.path}: declared JSON output lacks an object validator"
            )
        payload = _read_json(
            generation_path / specification.path,
            f"declared JSON output {specification.path}",
        )
        _require_json_contract(
            payload,
            schema=specification.schema,
            required_fields=specification.validator.required_fields,
            label=specification.path,
        )

    for family in ("metamath", "isabelle"):
        wrapper = _read_json(
            generation_path / "heldout" / f"{family}.json",
            f"{family} heldout wrapper",
        )
        contract = wrapper.get("contract")
        if not isinstance(contract, Mapping):
            raise IntegrationError(
                f"{family.capitalize()} heldout wrapper lacks required contract"
            )
        _validate_family_heldout_contract(contract, family=family)


def _independent_verify_generation_path(
    generation_path: Path,
    *,
    production: bool,
) -> dict[str, Any]:
    """Recompute row, sidecar, route, and family isolation claims from bytes."""

    _validate_declared_json_outputs(generation_path)
    source_manifests = {}
    for family in EXACT_SIBLINGS:
        source_link = _read_json(
            generation_path / "sidecars" / "sources" / f"{family}.json",
            f"{family} linked source manifest",
        )
        source_manifests[family] = _validate_source_manifest(
            source_link.get("manifest"),
            family=family,
            production=production,
        )
    rows = {}
    for family in EXACT_SIBLINGS:
        rows[family] = {}
        for split in ("raw", "shards", "eval"):
            rows[family][split] = _validate_rows(
                generation_path / split / f"{family}.jsonl",
                family=family,
                source_manifest=source_manifests[family],
                label=f"{split}/{family}.jsonl",
            )

    def one_schema(values: Iterable[str], *, label: str) -> str:
        schemas = set(values)
        if len(schemas) != 1:
            raise IntegrationError(f"{label} has mixed or missing schemas")
        return next(iter(schemas))

    schema_link = _read_json(
        generation_path / "sidecars" / "schemas.json",
        "schema sidecar",
    )
    expected_schema_payload = {
        "row_schemas": {
            family: one_schema(
                (
                    item.record["schema_version"]
                    for split in ("raw", "shards", "eval")
                    for item in rows[family][split]
                ),
                label=f"{family} rows",
            )
            for family in EXACT_SIBLINGS
        },
        "drop_schema": one_schema(
            (
                    item.record["schema_version"]
                    for family in EXACT_SIBLINGS
                    for item in _read_occurrences(
                        generation_path
                        / "sidecars"
                        / "drops"
                        / f"{family}.jsonl",
                        label=f"{family} drop sidecar",
                    )
            ),
            label="typed drop files",
        ),
        "source_manifest_roots": {
            family: source_manifests[family]["manifest_root_sha256"]
            for family in EXACT_SIBLINGS
        },
    }
    if any(
        schema_link.get(key) != value for key, value in expected_schema_payload.items()
    ):
        raise IntegrationError("schema sidecar disagrees with recomputed row/file schemas")

    expected_precheck = {
        "status": "clean",
        "families": list(EXACT_SIBLINGS),
        "counts": {
            family: {
                "raw": len(rows[family]["raw"]),
                "train": len(rows[family]["shards"]),
                "eval": len(rows[family]["eval"]),
            }
            for family in EXACT_SIBLINGS
        },
        "validators": {
            "rows": "family-deep-reconstruction-v2",
            "mml": "ValidatedHoldoutContract",
            "routes": "physical-occurrence-routes/v2",
        },
    }
    precheck = _read_json(
        generation_path / "sidecars" / "precheck.json",
        "precheck sidecar",
    )
    occurrence_link = _read_json(
        generation_path / "sidecars" / "occurrences.json",
        "source occurrence sidecar",
    )

    routes = _read_route_ledger(generation_path)
    drops = {
        family: _read_drop_ledger(generation_path, family)
        for family in EXACT_SIBLINGS
    }
    for family in EXACT_SIBLINGS:
        if len(routes[family]) != len(rows[family]["raw"]):
            raise IntegrationError(f"{family}: route accounting is incomplete")
        routed_drops = {
            raw_row: route
            for raw_row, route in routes[family].items()
            if route.get("disposition") == "drop"
        }
        if set(routed_drops) != set(drops[family]):
            raise IntegrationError(f"{family}: typed drop ledger is stale")
        for raw_row, route in routed_drops.items():
            if route.get("drop_type") != drops[family][raw_row].get("drop_type"):
                raise IntegrationError(f"{family}: typed drop reason is stale")

    metamath_heldout = _read_json(
        generation_path / "heldout" / "metamath.json",
        "Metamath heldout contract",
    )
    persisted_ledger = metamath_heldout.get("overlength_drop_ledger")
    persisted_accounting = metamath_heldout.get("source_accounting")
    if production and (
        not isinstance(persisted_ledger, Mapping)
        or not isinstance(persisted_accounting, Mapping)
    ):
        raise IntegrationError(
            "production Metamath heldout lacks exact source/drop accounting"
        )
    if isinstance(persisted_ledger, Mapping) or isinstance(
        persisted_accounting,
        Mapping,
    ):
        if not isinstance(persisted_ledger, Mapping) or not isinstance(
            persisted_accounting,
            Mapping,
        ):
            raise IntegrationError(
                "Metamath source accounting and overlength ledger must appear together"
            )
        source_tokenizer = source_manifests["metamath"]["row_source_metadata"].get(
            "tokenizer_seal"
        )
        if not isinstance(source_tokenizer, Mapping):
            raise IntegrationError(
                "Metamath source metadata lacks the fixed tokenizer seal"
            )
        expected_source_accounting = _validate_metamath_source_accounting(
            persisted_accounting,
            raw=rows["metamath"]["raw"],
            train=rows["metamath"]["shards"],
            evaluation=rows["metamath"]["eval"],
            drops=drops["metamath"].values(),
            drop_ledger=persisted_ledger,
            tokenizer_seal=source_tokenizer,
        )
        expected_source_link = {"metamath": expected_source_accounting}
        if occurrence_link.get("source_accounting") != expected_source_link:
            raise IntegrationError(
                "source occurrence inventory lacks exact Metamath accounting roots"
            )
        expected_precheck["source_accounting"] = expected_source_link
        expected_precheck["validators"]["metamath_source_occurrences"] = (
            "metamath-source-occurrences-v1"
        )
    elif occurrence_link.get("source_accounting") is not None:
        raise IntegrationError("unexpected Metamath source accounting inventory")
    if any(precheck.get(key) != value for key, value in expected_precheck.items()):
        raise IntegrationError("precheck sidecar disagrees with recomputed validators/counts")
    _verify_metamath_isolation(
        raw=rows["metamath"]["raw"],
        train=rows["metamath"]["shards"],
        evaluation=rows["metamath"]["eval"],
        heldout=metamath_heldout,
        routes=routes["metamath"],
        drops=drops["metamath"],
    )
    isabelle_heldout = _read_json(
        generation_path / "heldout" / "isabelle.json",
        "Isabelle heldout contract",
    )
    _verify_isabelle_isolation(
        raw=rows["isabelle"]["raw"],
        train=rows["isabelle"]["shards"],
        evaluation=rows["isabelle"]["eval"],
        heldout=isabelle_heldout,
        routes=routes["isabelle"],
        drops=drops["isabelle"],
    )
    return {
        "schemas": expected_schema_payload,
        "precheck": expected_precheck,
    }


def verify_generation(
    corpus_root: str | os.PathLike[str],
    *,
    production: bool = True,
    mizar_semantic_index: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Deep-verify the transaction selected by ``CURRENT``."""

    root = Path(corpus_root)
    _reject_legacy_layout(root)
    resolved = GenerationCoordinator(root).resolve_current(
        required_siblings=EXACT_SIBLINGS,
    )
    if tuple(resolved.manifest["requested_siblings"]) != EXACT_SIBLINGS:
        raise IntegrationError("CURRENT does not select the exact six-family plan")
    actual_outputs = {item["path"] for item in resolved.manifest["outputs"]}
    if actual_outputs != _expected_output_paths():
        raise IntegrationError("generation output inventory is not exact")
    generation_path = resolved.path

    source_manifests = {}
    for family in EXACT_SIBLINGS:
        source_link = _read_json(
            generation_path / "sidecars" / "sources" / f"{family}.json",
            f"{family} linked source manifest",
        )
        if source_link.get("family") != family:
            raise IntegrationError(f"{family}: source sidecar family mismatch")
        manifest = _validate_source_manifest(
            source_link.get("manifest"),
            family=family,
            production=production,
        )
        if source_link.get("manifest_root_sha256") != manifest["manifest_root_sha256"]:
            raise IntegrationError(f"{family}: stale linked source manifest")
        source_manifests[family] = manifest

    raw_by_family = {}
    seen_ids = {}
    for family in EXACT_SIBLINGS:
        for split in ("raw", "shards", "eval"):
            rows = _validate_rows(
                generation_path / split / f"{family}.jsonl",
                family=family,
                source_manifest=source_manifests[family],
                label=f"{split}/{family}.jsonl",
            )
            if split == "raw":
                raw_by_family[family] = rows
                for item in rows:
                    previous = seen_ids.get(item.record["id"])
                    if previous is not None:
                        raise IntegrationError(
                            f"duplicate raw row id {item.record['id']!r}: "
                            f"{previous} and {family}:{item.line_number}"
                        )
                    seen_ids[item.record["id"]] = f"{family}:{item.line_number}"

    occurrence_link = _read_json(
        generation_path / "sidecars" / "occurrences.json",
        "source occurrence sidecar",
    )
    if occurrence_link.get("families") != list(EXACT_SIBLINGS):
        raise IntegrationError("source occurrence family inventory is not exact")
    expected_occurrences = []
    for family in EXACT_SIBLINGS:
        for item in raw_by_family[family]:
            expected_occurrences.append(
                {
                    "family": family,
                    "row_id": item.record["id"],
                    "raw_path": f"raw/{family}.jsonl",
                    "source_line": item.line_number,
                    "byte_start": item.byte_start,
                    "byte_end": item.byte_end,
                    "raw_sha256": item.raw_sha256,
                }
            )
    actual_occurrences = occurrence_link.get("occurrences")
    if not isinstance(actual_occurrences, list) or len(actual_occurrences) != len(
        expected_occurrences
    ):
        raise IntegrationError("source occurrence inventory row count is stale")
    for expected, actual in zip(expected_occurrences, actual_occurrences, strict=True):
        if not isinstance(actual, dict) or any(
            actual.get(key) != value for key, value in expected.items()
        ):
            raise IntegrationError("source line/byte occurrence identity is stale")
        if not isinstance(actual.get("occurrence_id"), str):
            raise IntegrationError("source occurrence lacks transaction occurrence ID")

    tokenizer_link = _read_json(
        generation_path / "sidecars" / "tokenizer.json",
        "tokenizer sidecar",
    )
    tokenizer = _validate_tokenizer_seal(tokenizer_link.get("seal"))
    if tokenizer_link.get("tokenizer_root_sha256") != _canonical_sha256(tokenizer):
        raise IntegrationError("tokenizer sidecar root is stale")
    policy_link = _read_json(
        generation_path / "sidecars" / "policies.json",
        "policy sidecar",
    )
    policies = _validate_policies(
        policy_link.get("policies"),
        production=production,
    )
    if policy_link.get("policy_root_sha256") != policies["policy_root_sha256"]:
        raise IntegrationError("policy sidecar root is stale")

    mml_link = _read_json(
        generation_path / "heldout" / "mml.json",
        "MML heldout sidecar",
    )
    if (
        mml_link.get("mode") != "pooled_semantic_1000"
        or mml_link.get("selected_classes") != 1_000
    ):
        raise IntegrationError("MML heldout is not the pooled 1,000-class contract")
    with tempfile.TemporaryDirectory(prefix="p3-mml-verify.", dir="/tmp") as temporary:
        contract_path = Path(temporary)
        _materialize_mml_contract(generation_path, mml_link, contract_path)
        try:
            contract = mml_holdout.load_holdout_contract(
                contract_path,
                production=production,
            )
        except (
            mml_holdout.HoldoutError,
            OSError,
            TypeError,
            ValueError,
        ) as error:
            raise IntegrationError(f"MML holdout contract validation failed: {error}")
        if contract.authoritative_root != mml_link.get(
            "authoritative_manifest_root_sha256"
        ):
            raise IntegrationError("MML authoritative root link is stale")
        for family in MML_SIBLINGS:
            if contract.family_paths[family].train.read_bytes() != (
                generation_path / "shards" / f"{family}.jsonl"
            ).read_bytes():
                raise IntegrationError(f"{family}: MML train route bytes are stale")
            if contract.family_paths[family].eval.read_bytes() != (
                generation_path / "eval" / f"{family}.jsonl"
            ).read_bytes():
                raise IntegrationError(f"{family}: MML eval route bytes are stale")

    for family in ("metamath", "isabelle"):
        heldout = _read_json(
            generation_path / "heldout" / f"{family}.json",
            f"{family} heldout sidecar",
        )
        if (
            heldout.get("family") != family
            or heldout.get("mode") != "family_local_heldout"
            or heldout.get("source_manifest_root_sha256")
            != source_manifests[family]["manifest_root_sha256"]
        ):
            raise IntegrationError(f"{family}: heldout mode/root is stale")
        contract_payload = heldout.get("contract")
        if not isinstance(contract_payload, dict):
            raise IntegrationError(f"{family}: heldout contract is malformed")
        _validate_family_heldout_contract(contract_payload, family=family)
        facts = contract_payload.get("facts")
        if not isinstance(facts, list) or not facts:
            raise IntegrationError(f"{family}: heldout facts are missing")
        if production and len(facts) != 500:
            raise IntegrationError(f"{family}: production requires exactly 500 held facts")
        if family == "isabelle" and (
            contract_payload.get("schema_version") != "isabelle-transition-v2"
            or contract_payload.get("trajectory_drops") is not True
        ):
            raise IntegrationError("Isabelle heldout is not transition-v2 trajectory mode")

    _independent_verify_generation_path(
        generation_path,
        production=production,
    )

    if production:
        if mizar_semantic_index is None:
            raise IntegrationError(
                "production verification requires the current Mizar semantic index"
            )
        _verify_current_mizar_index(
            Path(mizar_semantic_index),
            source_manifests=source_manifests,
            generation_path=generation_path,
        )

    return {
        "status": "clean",
        "generation_id": resolved.generation_id,
        "logical_root_sha256": resolved.logical_root_sha256,
        "families": list(EXACT_SIBLINGS),
        "mml_selected_classes": len(contract.selected_class_ids),
        "modes": {
            "production": production,
            "mml": "pooled_semantic_1000",
            "metamath": "family_local_heldout",
            "isabelle": "family_local_heldout",
        },
    }


def production_blockers(
    source_manifests: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Return unresolved technical generation gates without legal/evaluator overreach."""

    blockers = []
    try:
        mml_holdout.production_source_policy()
    except mml_holdout.HoldoutError as error:
        blockers.append(f"MML source policy roots are unfinished: {error}")
    if source_manifests is None:
        blockers.append("exact six-family source manifests were not supplied")
        return blockers
    if set(source_manifests) != set(EXACT_SIBLINGS):
        blockers.append("exact six-family source manifests were not supplied")
        return blockers
    for family in EXACT_SIBLINGS:
        manifest = source_manifests[family]
        builder = manifest.get("builder") if isinstance(manifest, Mapping) else None
        if not isinstance(builder, Mapping) or builder.get("driver") != (
            "external-command-v2"
        ):
            blockers.append(f"{family} accepted production builder is not supplied")
    return blockers


def build_preflight_report(
    *,
    validated_paths: Mapping[str, str],
    validated_roots: Mapping[str, str],
    validated_commands: Mapping[str, Sequence[str]],
    blockers: Sequence[str] = (),
    errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a content-addressed production preflight report."""

    if errors:
        status = "invalid"
    elif blockers:
        status = "blocked"
    else:
        status = "ready"
    body = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": status,
        "validated_paths": dict(sorted(validated_paths.items())),
        "validated_roots": dict(sorted(validated_roots.items())),
        "validated_commands": {
            key: list(value) for key, value in sorted(validated_commands.items())
        },
        "blockers": list(blockers),
        "errors": list(errors),
    }
    return {**body, "preflight_root_sha256": _canonical_sha256(body)}


def _validate_secure_production_roots(
    *,
    corpus_root: Path,
    work_root: Path,
) -> tuple[Path, Path]:
    temporary_root = Path("/tmp").resolve()
    trusted_roots = [temporary_root]
    configured_root = os.environ.get("P3_TRUSTED_GENERATION_ROOT")
    if configured_root:
        requested_root = Path(configured_root).expanduser()
        if (
            not requested_root.is_absolute()
            or not requested_root.is_dir()
            or requested_root.is_symlink()
        ):
            raise IntegrationError(
                "P3_TRUSTED_GENERATION_ROOT must be an existing absolute real directory"
            )
        trusted_roots.append(requested_root.resolve())
    if not work_root.is_absolute() or not work_root.is_dir() or work_root.is_symlink():
        raise IntegrationError(
            "trusted production work root must be an existing absolute real directory"
        )
    resolved_work = work_root.resolve()
    resolved_corpus = corpus_root.resolve(strict=False)
    work_trust_root = next(
        (root for root in trusted_roots if resolved_work.is_relative_to(root)),
        None,
    )
    corpus_trust_root = next(
        (root for root in trusted_roots if resolved_corpus.is_relative_to(root)),
        None,
    )
    if work_trust_root is None:
        raise IntegrationError("production work root is outside every trusted root")
    if corpus_trust_root is None:
        raise IntegrationError("production transaction root is outside every trusted root")
    if work_trust_root != corpus_trust_root:
        raise IntegrationError(
            "production work and transaction roots must share one trusted root"
        )
    if resolved_work == resolved_corpus or resolved_work.is_relative_to(resolved_corpus):
        raise IntegrationError("production builder work must be external to transaction root")
    if resolved_corpus == resolved_work or resolved_corpus.is_relative_to(resolved_work):
        raise IntegrationError("production transaction root must be external to builder work")
    if corpus_root.exists() and corpus_root.is_symlink():
        raise IntegrationError("production transaction root must not be a symlink")
    parent = resolved_corpus.parent
    if not parent.is_dir() or parent.is_symlink():
        raise IntegrationError("production transaction parent must be an existing real directory")
    return resolved_corpus, resolved_work


def _validate_external_mml_contract_location(
    contract_root: Path,
    *,
    corpus_root: Path,
    work_root: Path,
) -> None:
    resolved_contract = contract_root.resolve()
    for label, managed_root in (
        ("transaction", corpus_root.resolve(strict=False)),
        ("builder work", work_root.resolve()),
    ):
        if (
            resolved_contract == managed_root
            or resolved_contract.is_relative_to(managed_root)
            or managed_root.is_relative_to(resolved_contract)
        ):
            raise IntegrationError(
                f"persisted MML contract must be external to production {label} root"
            )


def _require_readable_path(path: Path, *, label: str, regular_file: bool) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise IntegrationError(f"{label} is missing or unreadable: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise IntegrationError(f"{label} must not be a symlink: {path}")
    if regular_file and not stat.S_ISREG(metadata.st_mode):
        raise IntegrationError(f"{label} must be a regular file: {path}")
    if not regular_file and not (
        stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
    ):
        raise IntegrationError(f"{label} must be a regular file or directory: {path}")
    if not os.access(path, os.R_OK):
        raise IntegrationError(f"{label} is not readable: {path}")
    return path.resolve()


def _sha256_regular_file_nofollow(path: Path, *, label: str) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrationError(f"{label} cannot be opened safely: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise IntegrationError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_metamath_drop_ledger_file(
    path: Path,
    *,
    supplied: Mapping[str, Any],
    tokenizer_seal: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    resolved = _require_readable_path(
        path,
        label="Metamath overlength drop ledger",
        regular_file=True,
    )
    before = _sha256_regular_file_nofollow(
        resolved,
        label="Metamath overlength drop ledger",
    )
    on_disk = _read_json(resolved, "Metamath overlength drop ledger")
    after = _sha256_regular_file_nofollow(
        resolved,
        label="Metamath overlength drop ledger",
    )
    if before != after or on_disk != dict(supplied):
        raise IntegrationError(
            "Metamath overlength drop ledger bytes changed or were substituted"
        )
    ledger = _validate_metamath_drop_ledger(
        on_disk,
        tokenizer_seal=tokenizer_seal,
    )
    return resolved, ledger, before


def _sha256_jsonl_nofollow(
    path: Path,
    *,
    label: str,
) -> tuple[str, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise IntegrationError(f"{label} cannot be opened safely: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise IntegrationError(f"{label} is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        rows = 0
        last_byte = None
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            rows += chunk.count(b"\n")
            last_byte = chunk[-1]
        if size and last_byte != ord("\n"):
            raise IntegrationError(f"{label} lacks a final JSONL newline")
        return digest.hexdigest(), size, rows
    finally:
        os.close(descriptor)


def _resolve_command_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIRECTORY.parent / path
    return path


def _builder_option_values(
    argv: Sequence[str],
    *,
    family: str,
    name: str,
) -> list[str]:
    arity = PRODUCTION_BUILDER_OPTIONS[family].get(name)
    if arity is None:
        return []
    for index, value in enumerate(argv[2:], start=2):
        if value != name:
            continue
        if arity == "zero":
            return []
        values = []
        for candidate in argv[index + 1 :]:
            if candidate.startswith("--"):
                break
            values.append(candidate)
            if arity == "one":
                break
        return values
    return []


def _required_builder_option_values(
    argv: Sequence[str],
    *,
    family: str,
    name: str,
) -> list[str]:
    values = _builder_option_values(argv, family=family, name=name)
    if not values:
        raise IntegrationError(f"{family}: production source command lacks {name}")
    return values


def _record_native_source_roots(
    metadata: Mapping[str, Any],
    *,
    family: str,
    validated_roots: dict[str, str],
) -> None:
    for label, digest in _iter_digest_fields(
        metadata,
        prefix=f"builder_source/{family}",
    ):
        key = re.sub(r"[^A-Za-z0-9_.\-/]+", "_", label)
        validated_roots[key] = digest


def _validate_builder_native_source_metadata(
    manifest: Mapping[str, Any],
    *,
    family: str,
    validated_paths: dict[str, str],
    validated_roots: dict[str, str],
    tokenizer_metadata: Mapping[str, Any] | None = None,
    tokenizer_path: Path | None = None,
    metamath_drop_ledger: Mapping[str, Any] | None = None,
) -> None:
    """Recompute each builder's native source identity from its real inputs."""

    builder = manifest.get("builder")
    raw = builder.get("raw") if isinstance(builder, Mapping) else None
    argv = raw.get("argv") if isinstance(raw, Mapping) else None
    if not isinstance(argv, list):
        raise IntegrationError(f"{family}: production raw builder command is missing")
    actual = manifest.get("row_source_metadata")
    if not isinstance(actual, Mapping):
        raise IntegrationError(f"{family}: row source metadata is missing")

    source_paths: list[Path] = []
    try:
        if family == "metamath":
            if tokenizer_metadata is None or tokenizer_path is None:
                raise IntegrationError(
                    "metamath: fixed tokenizer seal and path are required"
                )
            if metamath_drop_ledger is None:
                raise IntegrationError(
                    "metamath: exact overlength drop ledger is required"
                )
            tokenizer_seal = _validate_tokenizer_seal(tokenizer_metadata)
            try:
                ledger = _json_copy(metamath_drop_ledger)
                metamath_builder.validate_drop_ledger(ledger)
            except (TypeError, ValueError) as error:
                raise IntegrationError(
                    f"metamath: overlength drop ledger validation failed: {error}"
                ) from error
            if ledger.get("tokenizer_seal") != tokenizer_seal:
                raise IntegrationError(
                    "metamath: drop ledger tokenizer seal is stale"
                )
            resolved_tokenizer = _require_readable_path(
                tokenizer_path,
                label="Metamath production tokenizer",
                regular_file=False,
            )
            commands = {"raw": argv}
            split = builder.get("split")
            if isinstance(split, Mapping) and isinstance(split.get("argv"), list):
                commands["split"] = split["argv"]
            for stage, command in commands.items():
                command_tokenizer = _resolve_command_path(
                    _required_builder_option_values(
                        command,
                        family=family,
                        name="--tokenizer",
                    )[0]
                )
                resolved_command_tokenizer = _require_readable_path(
                    command_tokenizer,
                    label=f"Metamath {stage} builder tokenizer",
                    regular_file=False,
                )
                if resolved_command_tokenizer != resolved_tokenizer:
                    raise IntegrationError(
                        "metamath: builder tokenizer path is substituted"
                    )
                if (
                    _load_actual_tokenizer_seal(resolved_command_tokenizer)
                    != tokenizer_seal
                ):
                    raise IntegrationError(
                        "metamath: builder tokenizer bytes disagree with fixed seal"
                    )
            mm_dir = _resolve_command_path(
                _required_builder_option_values(
                    argv,
                    family=family,
                    name="--mm-dir",
                )[0]
            )
            native_manifest = metamath_builder.source_manifest(str(mm_dir))
            _databases, _statements, conflict_map = (
                metamath_builder.load_pinned_databases(str(mm_dir))
            )
            try:
                metamath_builder.verify_pinned_oversized_population(
                    native_manifest,
                    ledger,
                )
            except (RuntimeError, TypeError, ValueError) as error:
                raise IntegrationError(
                    f"metamath: pinned overlength population is stale: {error}"
                ) from error
            expected = metamath_builder.build_source_metadata(
                native_manifest,
                conflict_map,
                drop_ledger=ledger,
                tokenizer_seal=tokenizer_seal,
            )
            native_occurrences = [
                {
                    "id": entry["id"],
                    "theorem": entry["theorem"],
                    "native_row_sha256": entry["native_row_sha256"],
                }
                for entry in ledger["entries"]
            ]
            validated_roots.update(
                {
                    "metamath_drop_ledger/canonical": ledger[
                        "canonical_root_sha256"
                    ],
                    "metamath_drop_ledger/entries": ledger[
                        "entries_root_sha256"
                    ],
                    "metamath_drop_ledger/native_occurrences": _canonical_sha256(
                        native_occurrences
                    ),
                    "metamath_drop_ledger/tokenizer": ledger[
                        "tokenizer_root_sha256"
                    ],
                }
            )
            source_paths.extend(mm_dir / f"{database}.mm" for database in ("set", "iset", "nf"))
        elif family in {"prf2", "enigma"}:
            try:
                from . import build_atp_shard as atp_builder
            except ImportError:  # pragma: no cover
                import build_atp_shard as atp_builder

            source_values = _required_builder_option_values(
                argv,
                family=family,
                name="--src",
            )
            source_files: list[str] = []
            source_of: dict[str, str] = {}
            for source_value in source_values:
                resolved_source = str(_resolve_command_path(source_value))
                files = atp_builder._files_for_source(resolved_source)
                if not files:
                    raise IntegrationError(
                        f"{family}: source matched no builder-native proof files"
                    )
                source_paths.append(Path(resolved_source))
                for path in files:
                    source_files.append(path)
                    source_of[path] = resolved_source
            expected = atp_builder._build_source_metadata(
                [str(path) for path in source_paths],
                source_files,
                source_of,
                SimpleNamespace(
                    dedup="--dedup" in argv,
                    fenced="--fenced" in argv,
                    jaccard=float(
                        _builder_option_value(argv, "--jaccard") or 0.5
                    ),
                    min_steps=int(
                        _required_builder_option_values(
                            argv,
                            family=family,
                            name="--min-steps",
                        )[0]
                    ),
                ),
            )
        elif family == "isabelle":
            try:
                from . import build_isabelle_shard as isabelle_builder
            except ImportError:  # pragma: no cover
                import build_isabelle_shard as isabelle_builder

            source_path = _resolve_command_path(
                _required_builder_option_values(
                    argv,
                    family=family,
                    name="--src",
                )[0]
            )
            source_paths.append(source_path)
            expected = isabelle_builder._build_source_metadata(
                isabelle_builder.verify_source_file(source_path)
            )
        elif family in {"mizar", "thproofs"}:
            try:
                from . import build_mizar_human_shard as direct_mizar_builder
                from . import build_thproofs_shard as thproofs_builder
                from . import mizar_current_index
            except ImportError:  # pragma: no cover
                import build_mizar_human_shard as direct_mizar_builder
                import build_thproofs_shard as thproofs_builder
                import mizar_current_index

            tree_flags = {
                "mml": "--mml-root",
                "html": "--html-root",
                "thproofs": "--thproofs-root" if family == "mizar" else "--src",
            }
            archive_flags = {
                "mml": "--mizar-archive",
                "html": "--html-archive",
                "thproofs": "--thproofs-archive",
            }
            roots = {
                name: _resolve_command_path(
                    _required_builder_option_values(
                        argv,
                        family=family,
                        name=flag,
                    )[0]
                )
                for name, flag in tree_flags.items()
            }
            archives = {
                name: _resolve_command_path(
                    _required_builder_option_values(
                        argv,
                        family=family,
                        name=flag,
                    )[0]
                )
                for name, flag in archive_flags.items()
            }
            source_manifest_path = _resolve_command_path(
                _required_builder_option_values(
                    argv,
                    family=family,
                    name="--source-manifest",
                )[0]
            )
            index_path = _resolve_command_path(
                _required_builder_option_values(
                    argv,
                    family=family,
                    name="--semantic-index",
                )[0]
            )
            upstream = mizar_current_index.verify_source_manifest(
                source_manifest_path,
                roots,
                archive_paths=archives,
            )
            supplied_index_root = _sha256_regular_file_nofollow(
                index_path,
                label=f"{family} semantic index",
            )
            if family == "mizar":
                declared_index_root = _required_builder_option_values(
                    argv,
                    family=family,
                    name="--semantic-index-sha256",
                )[0]
                if supplied_index_root != declared_index_root:
                    raise IntegrationError(
                        "mizar: semantic index command root is stale"
                    )
                effective_tokenizer = (
                    dict(tokenizer_metadata)
                    if tokenizer_metadata is not None
                    else dict(actual.get("source_roots", {}).get("tokenizer", {}))
                )
                config = SimpleNamespace(
                    production=True,
                    replay_sample_size=direct_mizar_builder.REPLAY_SAMPLE_SIZE,
                    semantic_index_sha256=supplied_index_root,
                )
                expected = direct_mizar_builder._source_metadata(
                    config,
                    upstream,
                    effective_tokenizer,
                )
                expected["source_manifest_root_sha256"] = actual.get(
                    "source_manifest_root_sha256"
                )
            else:
                with mizar_current_index.MizarIndex(index_path) as index:
                    expected = thproofs_builder._verified_source_metadata(
                        index,
                        index_path,
                        source_manifest_path,
                        upstream,
                        roots,
                    )
            source_paths.extend(
                [*roots.values(), *archives.values(), source_manifest_path, index_path]
            )
        else:  # pragma: no cover - caller is closed over EXACT_SIBLINGS.
            raise IntegrationError(f"unknown builder-native source family {family}")
    except IntegrationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError, SystemExit) as error:
        raise IntegrationError(
            f"{family}: builder-native source metadata verification failed: {error}"
        ) from error

    if dict(actual) != dict(expected):
        raise IntegrationError(
            f"{family}: builder-native source metadata disagrees with real inputs"
        )
    _record_native_source_roots(
        actual,
        family=family,
        validated_roots=validated_roots,
    )
    for index, path in enumerate(source_paths):
        resolved = _require_readable_path(
            path,
            label=f"{family} builder-native source",
            regular_file=False,
        )
        validated_paths[f"builder_source/{family}/{index}"] = str(resolved)


def _validate_source_snapshot_paths(
    manifest: Mapping[str, Any],
    *,
    family: str,
    validated_paths: dict[str, str],
    validated_roots: dict[str, str],
) -> None:
    snapshots = manifest.get("source_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise IntegrationError(f"{family}: exact source snapshots are missing")

    native_digests = {
        digest
        for section in ("source_roots", "source_trees", "source_archives")
        for _label, digest in _iter_digest_fields(
            manifest["row_source_metadata"].get(section, {}),
            prefix=section,
        )
    }
    command_files_by_digest: dict[str, Path] = {}
    builder = manifest.get("builder")
    raw = builder.get("raw") if isinstance(builder, Mapping) else None
    argv = raw.get("argv") if isinstance(raw, Mapping) else ()
    if isinstance(argv, Sequence) and not isinstance(argv, (str, bytes)):
        for value in argv[2:]:
            if not isinstance(value, str) or value.startswith("--"):
                continue
            candidate = _resolve_command_path(value)
            try:
                metadata = candidate.lstat()
            except OSError:
                continue
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                continue
            digest = _sha256_regular_file_nofollow(
                candidate,
                label=f"{family} command source",
            )
            command_files_by_digest.setdefault(digest, candidate.resolve())

    declared_snapshots = set()
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise IntegrationError(f"{family}: malformed source snapshot")
        snapshot_key = _canonical_sha256(snapshot)
        if snapshot_key in declared_snapshots:
            raise IntegrationError(f"{family}: duplicate source snapshot declaration")
        declared_snapshots.add(snapshot_key)
        digests = list(
            _iter_digest_fields(
                snapshot,
                prefix=f"{family} source snapshot {index}",
            )
        )
        if not digests:
            raise IntegrationError(f"{family}: source snapshot has no verifiable root")
        reference = snapshot.get("reference")
        direct_path = None
        direct_digest = None
        if (
            isinstance(reference, str)
            and "://" not in reference
            and Path(reference).is_absolute()
        ):
            direct_path = _require_readable_path(
                Path(reference),
                label=f"{family} source snapshot",
                regular_file=False,
            )
            if direct_path.is_file():
                direct_digest = _sha256_regular_file_nofollow(
                    direct_path,
                    label=f"{family} source snapshot",
                )
                if direct_digest not in {digest for _label, digest in digests}:
                    raise IntegrationError(
                        f"{family}: source snapshot hash mismatch for {reference}"
                    )
                command_files_by_digest.setdefault(direct_digest, direct_path)

        for digest_index, (label, digest) in enumerate(digests):
            matched_path = command_files_by_digest.get(digest)
            if digest not in native_digests and matched_path is None:
                raise IntegrationError(
                    f"{family}: source snapshot root is not bound to builder inputs: {label}"
                )
            key = f"source_snapshot/{family}/{index}"
            if len(digests) > 1:
                key += f"/{digest_index}"
            path_value = matched_path or direct_path
            if path_value is not None:
                validated_paths[key] = str(path_value)
            elif isinstance(reference, str) and reference:
                validated_paths[key] = reference
            else:
                validated_paths[key] = f"builder-native://{family}/{index}"
            validated_roots[key] = digest


def _load_actual_tokenizer_seal(tokenizer_path: Path) -> dict[str, Any]:
    try:
        try:
            from .build_isabelle_shard import (
                _tokenizer_metadata,
                load_vendored_tokenizer,
            )
        except ImportError:  # pragma: no cover
            from build_isabelle_shard import (
                _tokenizer_metadata,
                load_vendored_tokenizer,
            )

        metadata = dict(_tokenizer_metadata(load_vendored_tokenizer(tokenizer_path)))
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise IntegrationError(f"sealed tokenizer load failed: {error}") from error
    loaded_path = metadata.pop("path", None)
    expected_path = (
        tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
    ).resolve()
    if not isinstance(loaded_path, str) or Path(loaded_path).resolve() != expected_path:
        raise IntegrationError("loaded tokenizer path disagrees with requested tokenizer")
    return metadata


def _preflight_enigma_low_tier_paths(argv: Sequence[str]) -> dict[str, Path]:
    base_value = _builder_option_value(argv, "--enigma-low-tier-base")
    tokenizer_value = _builder_option_value(argv, "--tokenizer-json")
    if base_value is None and tokenizer_value is None:
        return {}
    if base_value is None or tokenizer_value is None:
        raise IntegrationError("ENIGMA low-tier command paths must appear together")

    base_path = _require_readable_path(
        _resolve_command_path(base_value),
        label="ENIGMA low-tier accepted base",
        regular_file=False,
    )
    if not base_path.is_dir():
        raise IntegrationError("ENIGMA low-tier accepted base must be a directory")
    tokenizer_path = _require_readable_path(
        _resolve_command_path(tokenizer_value),
        label="ENIGMA low-tier tokenizer JSON",
        regular_file=True,
    )
    return {
        "enigma_low_tier_base": base_path,
        "tokenizer_json": tokenizer_path,
    }


def preflight_production_inputs(
    *,
    corpus_root: str | os.PathLike[str],
    work_root: str | os.PathLike[str],
    source_manifest_paths: Mapping[str, Path],
    source_manifests: Mapping[str, Mapping[str, Any]],
    tokenizer_seal_path: Path,
    tokenizer_seal: Mapping[str, Any],
    tokenizer_path: Path,
    metamath_drop_ledger_path: Path,
    metamath_drop_ledger: Mapping[str, Any],
    policies_path: Path,
    policies: Mapping[str, Any],
    mizar_semantic_index: Path,
    mml_contract_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Fully validate production inputs without creating or running anything."""

    resolved_corpus, resolved_work = _validate_secure_production_roots(
        corpus_root=Path(corpus_root),
        work_root=Path(work_root),
    )
    if set(source_manifest_paths) != set(EXACT_SIBLINGS) or set(source_manifests) != set(
        EXACT_SIBLINGS
    ):
        raise IntegrationError("preflight requires exact six-family source manifests")
    validated_paths = {
        "corpus_root": str(Path(corpus_root).resolve(strict=False)),
        "work_root": str(Path(work_root).resolve()),
    }
    validated_roots = {}
    validated_sources = {}
    for family in EXACT_SIBLINGS:
        manifest_path = _require_readable_path(
            source_manifest_paths[family],
            label=f"{family} source manifest",
            regular_file=True,
        )
        validated_paths[f"source_manifest/{family}"] = str(manifest_path)
        manifest = _validate_source_manifest(
            source_manifests[family],
            family=family,
            production=True,
        )
        validated_sources[family] = manifest
        validated_roots[f"source_manifest/{family}"] = manifest[
            "manifest_root_sha256"
        ]

    seal_path = _require_readable_path(
        tokenizer_seal_path,
        label="tokenizer seal",
        regular_file=True,
    )
    policies_file = _require_readable_path(
        policies_path,
        label="generation policies",
        regular_file=True,
    )
    tokenizer_artifact = _require_readable_path(
        tokenizer_path,
        label="vendored tokenizer",
        regular_file=False,
    )
    index_path = _require_readable_path(
        mizar_semantic_index,
        label="current Mizar semantic index",
        regular_file=True,
    )
    validated_paths.update(
        {
            "tokenizer_seal": str(seal_path),
            "tokenizer": str(tokenizer_artifact),
            "policies": str(policies_file),
            "mizar_semantic_index": str(index_path),
        }
    )
    tokenizer = _validate_tokenizer_seal(tokenizer_seal)
    actual_tokenizer = _load_actual_tokenizer_seal(tokenizer_artifact)
    if actual_tokenizer != tokenizer:
        raise IntegrationError("loaded tokenizer bytes disagree with tokenizer seal")
    ledger_path, ledger, ledger_file_root = _validate_metamath_drop_ledger_file(
        metamath_drop_ledger_path,
        supplied=metamath_drop_ledger,
        tokenizer_seal=tokenizer,
    )
    validated_paths["metamath_drop_ledger"] = str(ledger_path)
    validated_policies = _validate_policies(policies, production=True)
    validated_roots["tokenizer"] = _canonical_sha256(tokenizer)
    validated_roots.update(
        {
            "metamath_drop_ledger/canonical": ledger["canonical_root_sha256"],
            "metamath_drop_ledger/entries": ledger["entries_root_sha256"],
            "metamath_drop_ledger/file": ledger_file_root,
        }
    )
    validated_roots["policies"] = validated_policies["policy_root_sha256"]
    actual_index_root = hashlib.sha256(index_path.read_bytes()).hexdigest()
    for family in ("mizar", "thproofs"):
        expected = validated_sources[family]["row_source_metadata"]["index_roots"][
            "semantic_index_sha256"
        ]
        if actual_index_root != expected:
            raise IntegrationError(
                f"{family}: current Mizar semantic index hash does not match manifest"
            )
    validated_roots["mizar_semantic_index"] = actual_index_root

    builder_configs = {
        family: _validate_production_builder_config(
            validated_sources[family],
            family=family,
        )
        for family in EXACT_SIBLINGS
    }
    mml_contract = _load_production_mml_contract(
        mml_contract_root,
        source_manifests=validated_sources,
        tokenizer_seal=tokenizer,
        policies=validated_policies,
    )
    _validate_external_mml_contract_location(
        mml_contract.root,
        corpus_root=resolved_corpus,
        work_root=resolved_work,
    )
    validated_paths["mml_contract"] = str(mml_contract.root.resolve())
    validated_roots.update(
        {
            f"mml_contract/{name}": root
            for name, root in _mml_contract_preflight_roots(mml_contract).items()
        }
    )
    enigma_input_paths, enigma_input_roots, enigma_binding = (
        _validate_enigma_low_tier_input_binding(
            builder_configs["enigma"]["raw"]["argv"],
            tokenizer_path=tokenizer_artifact,
            tokenizer_seal=tokenizer,
            mml_contract=mml_contract,
        )
    )
    validated_sources["enigma"] = _validate_source_manifest(
        _bind_enigma_low_tier_source_manifest(
            validated_sources["enigma"],
            enigma_binding,
        ),
        family="enigma",
        production=True,
    )
    validated_paths.update(
        {
            f"command_input/enigma/raw/{name}": str(path)
            for name, path in enigma_input_paths.items()
        }
    )
    validated_roots.update(
        {
            f"command_input/enigma/raw/{name}": root
            for name, root in enigma_input_roots.items()
        }
    )
    validated_roots["effective_source_manifest/enigma"] = validated_sources[
        "enigma"
    ]["manifest_root_sha256"]
    for family in EXACT_SIBLINGS:
        _validate_builder_native_source_metadata(
            validated_sources[family],
            family=family,
            validated_paths=validated_paths,
            validated_roots=validated_roots,
            tokenizer_metadata=actual_tokenizer,
            tokenizer_path=tokenizer_artifact,
            metamath_drop_ledger=ledger,
        )
        _validate_source_snapshot_paths(
            validated_sources[family],
            family=family,
            validated_paths=validated_paths,
            validated_roots=validated_roots,
        )

    commands = {}
    for family in EXACT_SIBLINGS:
        config = builder_configs[family]
        stages = ("raw",) if family in MML_SIBLINGS else ("raw", "split")
        for stage in stages:
            argv = config[stage]["argv"]
            commands[f"{family}/{stage}"] = list(argv)
            explicit_values = {
                value
                for flag in ("--enigma-low-tier-base", "--tokenizer-json")
                if (value := _builder_option_value(argv, flag)) is not None
            }
            for value in argv[2:]:
                if value in explicit_values:
                    continue
                if not value.startswith("/"):
                    continue
                if value == NO_EXCLUSION_PATH:
                    metadata = Path(value).lstat()
                    if not stat.S_ISCHR(metadata.st_mode):
                        raise IntegrationError(
                            "thproofs no-exclusion device is not a character device"
                        )
                    key = f"command_input/{family}/{stage}/no_exclusion"
                    validated_paths[key] = value
                    validated_roots[key] = hashlib.sha256(b"").hexdigest()
                    continue
                candidate = _require_readable_path(
                    Path(value),
                    label=f"{family}/{stage} command input",
                    regular_file=False,
                )
                validated_paths[
                    f"command_input/{family}/{stage}/{len(validated_paths)}"
                ] = str(candidate)
    blockers = production_blockers(validated_sources)
    report = build_preflight_report(
        validated_paths=validated_paths,
        validated_roots=validated_roots,
        validated_commands=commands,
        blockers=blockers,
    )
    return report, validated_sources, tokenizer, validated_policies


def _parse_source_assignments(values: Sequence[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or family not in EXACT_SIBLINGS or not raw_path:
            raise IntegrationError(
                "--source-manifest must be one of exact family=/path assignments"
            )
        if family in result:
            raise IntegrationError(f"duplicate source manifest for {family}")
        result[family] = Path(raw_path)
    if set(result) != set(EXACT_SIBLINGS):
        missing = sorted(set(EXACT_SIBLINGS) - set(result))
        extra = sorted(set(result) - set(EXACT_SIBLINGS))
        raise IntegrationError(
            f"exact six source manifests required; missing={missing}, extra={extra}"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Validate production inputs and fail closed until all roots are accepted."""

    parser = argparse.ArgumentParser(
        description="Build one immutable six-family P3 generation",
        epilog=(
            "Real full build requires exact source-backed six-family manifests, "
            "builder-native roots, and the finalized production source policy."
        ),
    )
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument("--tokenizer-seal", required=True, type=Path)
    parser.add_argument(
        "--metamath-drop-ledger",
        action="append",
        default=[],
        type=Path,
        help="exact builder-authored Metamath overlength ledger; supply once",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="vendored tokenizer directory; required once production gates are complete",
    )
    parser.add_argument("--policies", required=True, type=Path)
    parser.add_argument("--source-manifest", action="append", default=[])
    parser.add_argument("--mizar-semantic-index", type=Path)
    parser.add_argument(
        "--mml-contract-root",
        type=Path,
        help="independently persisted authoritative MML semantic-holdout v7 root",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate supplied roots only; never invoke builders or publish CURRENT",
    )
    args = parser.parse_args(argv)
    partial_paths: dict[str, str] = {}
    try:
        if len(args.metamath_drop_ledger) != 1:
            raise IntegrationError(
                "exactly one Metamath drop ledger must be supplied"
            )
        metamath_drop_ledger_path = args.metamath_drop_ledger[0]
        partial_paths["metamath_drop_ledger"] = str(
            metamath_drop_ledger_path.resolve(strict=False)
        )
        assignments = _parse_source_assignments(args.source_manifest)
        partial_paths.update(
            {
                f"source_manifest/{family}": str(path.resolve(strict=False))
                for family, path in assignments.items()
            }
        )
        source_manifests = {
            family: _read_json(path, f"{family} source manifest")
            for family, path in assignments.items()
        }
        if args.mizar_semantic_index is None:
            raise IntegrationError("current Mizar semantic index is required")
        if args.tokenizer_path is None:
            raise IntegrationError("vendored tokenizer path is required")
        if args.mml_contract_root is None:
            raise IntegrationError(
                "persisted authoritative MML contract root is required"
            )
        tokenizer_seal = _read_json(args.tokenizer_seal, "tokenizer seal")
        metamath_drop_ledger = _read_json(
            metamath_drop_ledger_path,
            "Metamath overlength drop ledger",
        )
        policy_payload = _read_json(args.policies, "generation policies")
        report, validated_sources, tokenizer, policies = preflight_production_inputs(
            corpus_root=args.corpus_root,
            work_root=args.work_root,
            source_manifest_paths=assignments,
            source_manifests=source_manifests,
            tokenizer_seal_path=args.tokenizer_seal,
            tokenizer_seal=tokenizer_seal,
            tokenizer_path=args.tokenizer_path,
            metamath_drop_ledger_path=metamath_drop_ledger_path,
            metamath_drop_ledger=metamath_drop_ledger,
            policies_path=args.policies,
            policies=policy_payload,
            mizar_semantic_index=args.mizar_semantic_index,
            mml_contract_root=args.mml_contract_root,
        )
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        if report["status"] != "ready":
            print(
                "P3 generation refused: real full build blocked:\n  - "
                + "\n  - ".join(report["blockers"]),
                file=sys.stderr,
            )
            return 2
        if args.dry_run:
            return 0
        result = build_production_generation(
            corpus_root=args.corpus_root,
            work_root=args.work_root,
            generation_id=args.generation_id,
            source_manifests=validated_sources,
            tokenizer_seal=tokenizer,
            tokenizer_path=args.tokenizer_path,
            metamath_drop_ledger=metamath_drop_ledger,
            policies=policies,
            mizar_semantic_index=args.mizar_semantic_index,
            mml_contract_root=args.mml_contract_root,
        )
        print(
            f"published {result.published.generation_id} "
            f"{result.published.logical_root_sha256}"
        )
        return 0
    except (IntegrationError, OSError, ValueError, TypeError) as error:
        if args.dry_run:
            report = build_preflight_report(
                validated_paths=partial_paths,
                validated_roots={},
                validated_commands={},
                errors=[str(error)],
            )
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        print(f"P3 generation refused: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
