"""Hostile contract tests for the separately published evaluator release."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_evaluator_release as release
import split_mml_semantic_holdout as holdout
from corpus_generation_transaction import (
    CURRENT_SCHEMA_VERSION,
    BinaryValidator,
    CommitUncertainError,
    DropRecord,
    GenerationCoordinator,
    GenerationPlan,
    JsonlValidator,
    JsonObjectValidator,
    OutputRole,
    OutputSpec,
    PublishPhase,
    ValidationError,
)

FAMILIES = ("enigma", "isabelle", "metamath", "mizar", "prf2", "thproofs")
GENERATION_ID = "corpus-gen-0001"
SOURCE_GENERATION_ID = "source-snapshot-0001"
DROP_SCHEMA = "p3-corpus-drop/v2"
HELDOUT_SCHEMAS = {
    "isabelle": "p3-isabelle-heldout/v2",
    "metamath": "p3-metamath-heldout/v1",
}


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return release.canonical_json_bytes(value, newline=True)


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))
    return path


def _common_row(
    row_id: str,
    theorem: str,
    facts: dict[str, str],
    *,
    schema_version: str | None = None,
) -> dict[str, Any]:
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    goal = f"goal {row_id}"
    target = f"proof {row_id}"
    row: dict[str, Any] = {
        "id": row_id,
        "theorem": theorem,
        "facts": facts,
        "cited": list(facts),
        "goal": goal,
        "target": target,
        "text": f"{block}\n---\nGOAL {goal}\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
    }
    if schema_version is not None:
        row["schema_version"] = schema_version
    return row


def _mizar_source_row(row_id: str, theorem: str, facts: dict[str, str]) -> dict[str, Any]:
    return _common_row(row_id, theorem, facts)


def _line(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _mml_sources(rows_by_shard: dict[str, list[dict[str, Any]]]):
    sources = {}
    for shard in holdout.SHARD_ORDER:
        lines = tuple(_line(row) for row in rows_by_shard.get(shard, ()))
        sources[shard] = holdout.MemoryShardSource(
            name=shard,
            logical_path=f"raw/{shard}.jsonl",
            lines=lines,
            expected_input_sha256=holdout.digest_lines(lines),
            source_snapshots=(
                holdout.SourceSnapshot(
                    reference=f"{shard}-source@production-test",
                    sha256=_sha(f"{shard}:snapshot"),
                ),
            ),
            source_manifest_root_sha256=_sha(f"{shard}:source-manifest"),
            quality_filter_root_sha256=_sha(f"{shard}:quality"),
            schema_generation_root_sha256=_sha(f"{shard}:schema-generation"),
        )
    return sources


def _production_mml_contract(tmp_path: Path, monkeypatch):
    pool = _mizar_source_row(
        "mizar-eval-1",
        "OTHER:1",
        {
            f"ARTICLE_{index}:1": f"selected statement {index}"
            for index in range(1000)
        },
    )
    train_rows = [
        _mizar_source_row(
            f"mizar-train-{index}",
            f"SAFE:{index + 2}",
            {"ARTICLE:7": "shared mapped statement"},
        )
        for index in range(3)
    ]
    dropped = _mizar_source_row("mizar-drop-1", "DROP:2", {"DROP:1": "drop"})
    rows = {"mizar": [pool, *train_rows, dropped]}
    sources = _mml_sources(rows)
    policy = holdout.SourceIdentityPolicy(
        policy_id="production-mml-source-policy-v1",
        test_only=False,
        shards={
            shard: holdout.ApprovedShardSource(
                input_sha256=source.expected_input_sha256,
                source_snapshots=source.source_snapshots,
                source_manifest_root_sha256=source.source_manifest_root_sha256,
                quality_filter_root_sha256=source.quality_filter_root_sha256,
                schema_generation_root_sha256=source.schema_generation_root_sha256,
            )
            for shard, source in sources.items()
        },
    )
    monkeypatch.setattr(holdout, "production_source_policy", lambda: policy)
    tokenizer = holdout.TokenizerSeam(
        seal=holdout.approved_tokenizer_seal(),
        count_text_plus_eos=lambda text: 16_385 if text == dropped["text"] else 32,
    )
    plan = holdout.plan_semantic_holdout(
        sources,
        tokenizer=tokenizer,
        policy_pins=holdout.current_policy_pins(),
        source_policy=policy,
    )
    root = tmp_path / "mml-contract"
    holdout.write_partition_atomically(plan, sources=sources, output=root)
    contract = holdout.load_holdout_contract(root, production=True)
    return root, contract, policy


def _family_rows(family: str) -> list[dict[str, Any]]:
    schema = release.FAMILY_ROW_SCHEMAS[family]
    if family == "mizar":
        mapped_name = "ARTICLE:7"
    elif family == "prf2":
        mapped_name = "t7_article"
    else:
        mapped_name = f"{family}_visible"
    return [
        {
            **_common_row(
                f"{family}-train-1",
                f"{family}:train",
                {mapped_name: "shared mapped statement"},
                schema_version=schema,
            ),
        },
        {
            **_common_row(
                f"{family}-eval-1",
                f"{family}:eval",
                {f"{family}_held": f"{family} held statement"},
                schema_version=schema,
            ),
        },
        {
            **_common_row(
                f"{family}-drop-1",
                f"{family}:drop",
                {f"{family}_drop": f"{family} drop statement"},
                schema_version=schema,
            ),
        },
    ]


def _binary_rows(path: Path, _context) -> int:
    if path.suffix == ".json":
        return 1
    return max(1, len(path.read_bytes().splitlines()))


@dataclass(frozen=True)
class Fixture:
    spec: release.EvaluatorReleaseSpec
    published: Any
    coordinator: GenerationCoordinator
    binary_validators: tuple[BinaryValidator, ...]
    current: Path
    semantic_contract: Any
    production_policy: Any


def _build_fixture(tmp_path: Path, monkeypatch) -> Fixture:
    _semantic_root, semantic_contract, production_policy = _production_mml_contract(
        tmp_path,
        monkeypatch,
    )
    source_root = tmp_path / "transaction-inputs"
    tokenizer = _write_json(
        source_root / "tokenizer.json",
        {
            "schema_version": release.TOKENIZER_SEAL_SCHEMA,
            "identity": "tokenizer/qwen25-vendored/v1",
            "manifest_sha256": _sha("tokenizer-manifest"),
            "behavior_digest": _sha("tokenizer-behavior"),
        },
    )
    source_manifests = {
        name: _write_json(
            source_root / "sources" / f"{name}.json",
            {
                "schema_version": release.SOURCE_MANIFEST_SCHEMAS[name],
                "source": name,
                "revision": f"{name}-revision",
            },
        )
        for name in release.SOURCE_MANIFEST_NAMES
    }
    cohorts = {
        family: _write_jsonl(
            source_root / "cohorts" / f"{family}.jsonl",
            [
                {
                    "schema_version": release.COHORT_LEDGER_SCHEMA,
                    "family": family,
                    "row_id": f"{family}-eval-1",
                    "cohort": "context_eligible",
                }
            ],
        )
        for family in FAMILIES
    }
    verifier = _write_json(
        source_root / "verifiers" / "metamath.json",
        {"schema_version": "p3-metamath-verifier/v1", "status": "tri-state"},
    )

    binary_schemas: dict[str, BinaryValidator] = {}

    def binary(schema: str) -> BinaryValidator:
        validator = binary_schemas.get(schema)
        if validator is None:
            validator = BinaryValidator(
                schema_version=schema,
                validator_id=f"p3-evaluator-source-{len(binary_schemas) + 1}/v1",
                validate=_binary_rows,
            )
            binary_schemas[schema] = validator
        return validator

    row_validators = {
        family: JsonlValidator(
            schema_version=release.FAMILY_ROW_SCHEMAS[family],
            required_fields=("facts", "id", "text", "theorem"),
        )
        for family in FAMILIES
    }
    drop_validator = JsonlValidator(
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
    )
    outputs: list[OutputSpec] = []
    for family in FAMILIES:
        schema = release.FAMILY_ROW_SCHEMAS[family]
        outputs.extend(
            (
                OutputSpec(
                    path=f"raw/{family}.jsonl",
                    role=OutputRole.RAW,
                    schema=schema,
                    sibling=family,
                    validator=row_validators[family],
                ),
                OutputSpec(
                    path=f"train/{family}.jsonl",
                    role=OutputRole.TRAIN,
                    schema=schema,
                    sibling=family,
                    validator=row_validators[family],
                ),
                OutputSpec(
                    path=f"eval/{family}.jsonl",
                    role=OutputRole.EVAL,
                    schema=schema,
                    sibling=family,
                    validator=row_validators[family],
                ),
                OutputSpec(
                    path=f"drops/{family}.jsonl",
                    role=OutputRole.SIDECAR,
                    schema=DROP_SCHEMA,
                    sibling=family,
                    drop_types=("over_context",),
                    validator=drop_validator,
                ),
                OutputSpec(
                    path=f"cohorts/{family}.jsonl",
                    role=OutputRole.SIDECAR,
                    schema=release.COHORT_LEDGER_SCHEMA,
                    sibling=family,
                    validator=binary(release.COHORT_LEDGER_SCHEMA),
                ),
            )
        )
    for name, schema in HELDOUT_SCHEMAS.items():
        outputs.append(
            OutputSpec(
                path=f"heldout/{name}.json",
                role=OutputRole.HELDOUT,
                schema=schema,
                sibling=name,
                validator=JsonObjectValidator(
                    schema_version=schema,
                    required_fields=("facts", "family"),
                    require_generation_links=True,
                ),
            )
        )
    outputs.append(
        OutputSpec(
            path="provenance/tokenizer.json",
            role=OutputRole.SIDECAR,
            schema=release.TOKENIZER_SEAL_SCHEMA,
            validator=binary(release.TOKENIZER_SEAL_SCHEMA),
        )
    )
    for name in release.SOURCE_MANIFEST_NAMES:
        schema = release.SOURCE_MANIFEST_SCHEMAS[name]
        outputs.append(
            OutputSpec(
                path=f"sources/{name}.json",
                role=OutputRole.SIDECAR,
                schema=schema,
                sibling=name if name in FAMILIES else None,
                validator=binary(schema),
            )
        )
    outputs.append(
        OutputSpec(
            path="verifiers/metamath.json",
            role=OutputRole.SIDECAR,
            schema=release.VERIFIER_MANIFEST_SCHEMAS["metamath"],
            sibling="metamath",
            validator=binary(release.VERIFIER_MANIFEST_SCHEMAS["metamath"]),
        )
    )
    for relative, artifact in semantic_contract.artifacts.items():
        schema = release.SEMANTIC_TRANSACTION_SCHEMAS[relative]
        outputs.append(
            OutputSpec(
                path=f"semantic/{relative}",
                role=OutputRole.SIDECAR,
                schema=schema,
                validator=binary(schema),
            )
        )

    plan = GenerationPlan(
        generation_id=GENERATION_ID,
        source_generation_id=SOURCE_GENERATION_ID,
        requested_siblings=FAMILIES,
        outputs=tuple(outputs),
    )
    validators = tuple(binary_schemas.values())
    coordinator = GenerationCoordinator(
        tmp_path / "corpus-transaction",
        binary_validators=validators,
    )

    def producer(writer):
        for family in FAMILIES:
            rows = _family_rows(family)
            writer.write_bytes(
                f"raw/{family}.jsonl",
                b"".join(_canonical_bytes(row) for row in rows),
            )
            occurrences = writer.raw_occurrences(f"raw/{family}.jsonl")
            writer.write_routed_jsonl(f"train/{family}.jsonl", [occurrences[0]])
            writer.write_routed_jsonl(f"eval/{family}.jsonl", [occurrences[1]])
            writer.write_drop_sidecar(
                f"drops/{family}.jsonl",
                [
                    DropRecord(
                        occurrence_id=occurrences[2].occurrence_id,
                        drop_type="over_context",
                        details={"token_count": 16_385},
                    )
                ],
            )
            writer.copy_file(f"cohorts/{family}.jsonl", cohorts[family])
        for family in tuple(HELDOUT_SCHEMAS):
            writer.write_linked_json(
                f"heldout/{family}.json",
                {"family": family, "facts": [f"{family}_held"]},
            )
        writer.copy_file("provenance/tokenizer.json", tokenizer)
        for name, path in source_manifests.items():
            writer.copy_file(f"sources/{name}.json", path)
        writer.copy_file("verifiers/metamath.json", verifier)
        for relative, artifact in semantic_contract.artifacts.items():
            writer.copy_file(f"semantic/{relative}", artifact.path)

    published = coordinator.publish(plan, producer)
    current = coordinator.root / "CURRENT"
    binding = release.generation_binding_from_transaction(
        coordinator.root,
        current,
        binary_validators=validators,
    )

    def artifact(transaction_path: str) -> release.InputArtifact:
        return release.InputArtifact(
            path=published.path / transaction_path,
            transaction_path=transaction_path,
        )

    semantic_transaction_root = published.path / "semantic"
    spec = release.EvaluatorReleaseSpec(
        dataset_id=release.APPROVED_DATASET_ID,
        version=release.UNPUBLISHED_VERSION,
        generation=binding,
        eval_files={
            family: artifact(f"eval/{family}.jsonl") for family in FAMILIES
        },
        train_files={
            family: artifact(f"train/{family}.jsonl") for family in FAMILIES
        },
        semantic_contract_root=semantic_transaction_root,
        heldout_manifests={
            family: artifact(f"heldout/{family}.json")
            for family in HELDOUT_SCHEMAS
        },
        tokenizer_seal=artifact("provenance/tokenizer.json"),
        source_manifests={
            name: artifact(f"sources/{name}.json")
            for name in release.SOURCE_MANIFEST_NAMES
        },
        family_source_manifests=release.EXPECTED_FAMILY_SOURCE_MANIFESTS,
        drop_ledgers={
            family: artifact(f"drops/{family}.jsonl") for family in FAMILIES
        },
        cohort_ledgers={
            family: artifact(f"cohorts/{family}.jsonl") for family in FAMILIES
        },
        verifier_manifests={"metamath": artifact("verifiers/metamath.json")},
        oracle_manifests={},
        expected_semantic_holdout_root_sha256=semantic_contract.authoritative_root,
        expected_tokenizer_seal_sha256=release.sha256_file(tokenizer),
        expected_source_manifests_root_sha256=release.named_artifact_root(
            {
                name: artifact(f"sources/{name}.json")
                for name in release.SOURCE_MANIFEST_NAMES
            }
        ),
    )
    return Fixture(
        spec=spec,
        published=published,
        coordinator=coordinator,
        binary_validators=validators,
        current=current,
        semantic_contract=semantic_contract,
        production_policy=production_policy,
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch) -> Fixture:
    return _build_fixture(tmp_path, monkeypatch)


def _resign_release(root: Path, mutate) -> None:
    manifest_path = root / release.MANIFEST_NAME
    seal_path = root / release.COMPLETION_SEAL_NAME
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest["inventory_root_sha256"] = release.canonical_sha256(manifest["inventory"])
    body = dict(manifest)
    body.pop("manifest_root_sha256")
    manifest["manifest_root_sha256"] = release.canonical_sha256(body)
    manifest_bytes = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    seal = json.loads(seal_path.read_text())
    seal["manifest_root_sha256"] = manifest["manifest_root_sha256"]
    seal["manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    seal["inventory_root_sha256"] = manifest["inventory_root_sha256"]
    seal_path.write_bytes(_canonical_bytes(seal))


def test_transaction_adapter_uses_valid_v2_logical_root(fixture):
    binding = fixture.spec.generation
    assert fixture.published.logical_root_sha256 != fixture.published.manifest_sha256
    assert binding.logical_root_sha256 == fixture.published.logical_root_sha256
    assert binding.manifest_file_sha256 == fixture.published.manifest_sha256
    assert binding.inventory
    plan = release.plan_release(fixture.spec)
    assert plan.manifest["provenance"]["corpus_generation"][
        "logical_root_sha256"
    ] == fixture.published.logical_root_sha256
    assert "manifest_root_sha256" not in plan.manifest["provenance"]["corpus_generation"]
    with pytest.raises((TypeError, release.EvaluatorReleaseError)):
        release.generation_binding_from_transaction(fixture.published, fixture.current)


def test_transaction_adapter_rejects_resigned_counterfeit_inventory(fixture, tmp_path):
    copied_root = tmp_path / "counterfeit-transaction"
    shutil.copytree(fixture.coordinator.root, copied_root)
    for path in [copied_root, *copied_root.rglob("*")]:
        path.chmod(path.stat().st_mode | 0o200)
    generation = copied_root / "generations" / GENERATION_ID
    manifest_path = generation / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = [
        item for item in manifest["outputs"] if item["path"] != "eval/prf2.jsonl"
    ]
    body = dict(manifest)
    body.pop("manifest_root_sha256")
    manifest["manifest_root_sha256"] = release.canonical_sha256(body)
    manifest_bytes = _canonical_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    current_path = copied_root / "CURRENT"
    current = json.loads(current_path.read_text())
    current["manifest_sha256"] = manifest_sha
    current_path.write_bytes(_canonical_bytes(current))
    committed_path = copied_root / "transactions" / f"{GENERATION_ID}.committed.json"
    committed = json.loads(committed_path.read_text())
    committed["manifest_sha256"] = manifest_sha
    committed_path.write_bytes(_canonical_bytes(committed))
    for path in [generation, *generation.rglob("*")]:
        path.chmod(path.stat().st_mode & ~0o222)
    coordinator = GenerationCoordinator(
        copied_root,
        binary_validators=fixture.binary_validators,
    )
    with pytest.raises((release.EvaluatorReleaseError, ValidationError)):
        release.generation_binding_from_transaction(
            coordinator.root,
            current_path,
            binary_validators=fixture.binary_validators,
        )


def test_empty_or_unvalidated_generation_contract_is_rejected(fixture):
    empty_manifest = {**fixture.published.manifest, "outputs": []}
    empty = replace(fixture.published, manifest=empty_manifest)
    with pytest.raises((TypeError, release.EvaluatorReleaseError)):
        release.generation_binding_from_transaction(empty, fixture.current)

    binding = replace(fixture.spec.generation, validated=False)
    with pytest.raises(release.EvaluatorReleaseError, match="validated"):
        release.plan_release(replace(fixture.spec, generation=binding))


def test_every_input_is_bound_to_transaction_path_role_schema_and_sibling(
    fixture,
    tmp_path,
):
    plan = release.plan_release(fixture.spec)
    eval_entry = next(
        item for item in plan.manifest["inventory"] if item["role"] == "eval:prf2"
    )
    assert eval_entry["source_binding"] == {
        "transaction_path": "eval/prf2.jsonl",
        "role": "eval",
        "schema": release.FAMILY_ROW_SCHEMAS["prf2"],
        "sibling": "prf2",
        "logical_root_sha256": fixture.published.logical_root_sha256,
            "transaction_inventory_sha256": fixture.spec.generation.transaction_inventory_sha256,
    }

    swapped = dict(fixture.spec.eval_files)
    swapped["prf2"] = fixture.spec.eval_files["enigma"]
    with pytest.raises(release.EvaluatorReleaseError, match="prf2.*transaction|sibling|path"):
        release.plan_release(replace(fixture.spec, eval_files=swapped))

    copied = tmp_path / "unbound-copy.jsonl"
    shutil.copy2(fixture.spec.eval_files["prf2"].path, copied)
    unbound = dict(fixture.spec.eval_files)
    unbound["prf2"] = replace(unbound["prf2"], path=copied)
    with pytest.raises(release.EvaluatorReleaseError, match="approved transaction"):
        release.plan_release(replace(fixture.spec, eval_files=unbound))


def test_real_production_mml_contract_and_exact_projections_are_used(fixture):
    plan = release.plan_release(fixture.spec)
    assert fixture.semantic_contract.production
    assert not fixture.semantic_contract.test_only
    assert plan.manifest["roles"]["heldout"] == {
        "atp": "evaluator/heldout-atp.json",
        "isabelle": "evaluator/heldout-isabelle.json",
        "metamath": "evaluator/heldout-metamath.json",
        "mizar": "evaluator/heldout-mizar.json",
        "mml": "evaluator/heldout-mml.json",
    }
    assert plan.manifest["roles"]["class_manifest_by_family"] == {
        "enigma": "evaluator/heldout-atp.json",
        "isabelle": "evaluator/heldout-isabelle.json",
        "metamath": "evaluator/heldout-metamath.json",
        "mizar": "evaluator/heldout-mizar.json",
        "prf2": "evaluator/heldout-atp.json",
        "thproofs": "evaluator/heldout-mizar.json",
    }
    assert "thproofs" not in fixture.semantic_contract.projections


def test_test_only_or_unrelated_semantic_contract_is_rejected(
    fixture,
    tmp_path,
    monkeypatch,
):
    row = _mizar_source_row(
        "synthetic-eval",
        "OTHER:1",
        {f"TEST_{index}:1": f"statement {index}" for index in range(1000)},
    )
    sources = _mml_sources({"mizar": [row]})
    policy = holdout.SourceIdentityPolicy(
        policy_id="synthetic-test-policy-v1",
        test_only=True,
        shards={
            shard: holdout.ApprovedShardSource(
                input_sha256=source.expected_input_sha256,
                source_snapshots=source.source_snapshots,
                source_manifest_root_sha256=source.source_manifest_root_sha256,
                quality_filter_root_sha256=source.quality_filter_root_sha256,
                schema_generation_root_sha256=source.schema_generation_root_sha256,
            )
            for shard, source in sources.items()
        },
    )
    test_plan = holdout.plan_semantic_holdout(
        sources,
        tokenizer=holdout.TokenizerSeam(
            seal=holdout.approved_tokenizer_seal(),
            count_text_plus_eos=lambda _text: 32,
        ),
        policy_pins=holdout.current_policy_pins(),
        source_policy=policy,
    )
    test_root = tmp_path / "test-only-contract"
    holdout.write_partition_atomically(test_plan, sources=sources, output=test_root)
    with pytest.raises((release.EvaluatorReleaseError, holdout.HoldoutError), match="test-only"):
        release.plan_release(
            replace(
                fixture.spec,
                semantic_contract_root=test_root,
                expected_semantic_holdout_root_sha256=test_plan.sealed_manifest_root_sha256,
            )
        )

    unrelated = tmp_path / "unrelated-production-contract"
    shutil.copytree(fixture.spec.semantic_contract_root, unrelated)
    monkeypatch.setattr(
        holdout,
        "production_source_policy",
        lambda: fixture.production_policy,
    )
    with pytest.raises(release.EvaluatorReleaseError, match="approved transaction"):
        release.plan_release(replace(fixture.spec, semantic_contract_root=unrelated))


def test_semantic_visibility_merges_mizar_atp_aliases_and_scoped_hashes(
    fixture,
    tmp_path,
):
    loaded = release.build_release(fixture.spec, tmp_path / "release")
    records = [
        json.loads(line)
        for line in loaded.corpus_union_visibility_path.read_text().splitlines()
    ]
    mapped = next(
        item for item in records if item["class_id"] == "mml:v1:theorem:ARTICLE:7"
    )
    assert mapped["aliases_by_representation"] == {
        "atp": ["t7_article"],
        "mizar": ["ARTICLE:7"],
    }
    assert mapped["visible_in_families"] == ["mizar", "prf2"]
    assert set(mapped["statement_hashes_by_representation"]) == {"atp", "mizar"}
    assert {
        (member["family"], member["native_name"])
        for member in mapped["native_members"]
    } == {("mizar", "ARTICLE:7"), ("prf2", "t7_article")}


def test_exact_flat_routes_schemas_and_unique_roles(fixture):
    plan = release.plan_release(fixture.spec)
    assert set(plan.manifest) == release.MANIFEST_KEYS
    roles = [entry["role"] for entry in plan.manifest["inventory"]]
    assert len(roles) == len(set(roles))
    for entry in plan.manifest["inventory"]:
        assert len(Path(entry["logical_path"]).parts) == 2
        assert Path(entry["logical_path"]).parts[0] == "evaluator"
        expected_role, expected_schema = release.CANONICAL_PATH_CONTRACT[
            entry["logical_path"]
        ]
        assert (entry["role"], entry["schema"]) == (expected_role, expected_schema)
        lineage_key = (
            "source_binding"
            if "source_binding" in entry
            else "derived_from_transaction_paths"
        )
        assert set(entry) == release.INVENTORY_BASE_KEYS | {lineage_key}
        assert entry["schema"]
        assert entry["generation_id"] == GENERATION_ID
        assert entry["bindings"]["corpus_logical_root_sha256"] == (
            fixture.published.logical_root_sha256
        )
    assert plan.manifest["roles"]["eval"] == {
        family: f"evaluator/eval-{family}.jsonl" for family in FAMILIES
    }
    assert plan.manifest["roles"]["train_visibility"] == {
        family: f"evaluator/visibility-{family}.jsonl" for family in FAMILIES
    }


def test_family_swap_duplicate_role_and_arbitrary_schema_are_rejected(
    fixture,
    tmp_path,
):
    loaded = release.build_release(fixture.spec, tmp_path / "release")

    def duplicate_role(manifest):
        manifest["roles"]["eval"]["prf2"] = manifest["roles"]["eval"]["enigma"]

    _resign_release(loaded.root, duplicate_role)
    with pytest.raises(release.EvaluatorReleaseError, match="canonical|duplicate|role"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )

    loaded = release.build_release(fixture.spec, tmp_path / "release-role")

    def swap_family_contract(manifest):
        entries = {item["logical_path"]: item for item in manifest["inventory"]}
        left = entries["evaluator/eval-enigma.jsonl"]
        right = entries["evaluator/eval-prf2.jsonl"]
        left["role"], right["role"] = right["role"], left["role"]

    _resign_release(loaded.root, swap_family_contract)
    with pytest.raises(release.EvaluatorReleaseError, match="canonical|role"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )

    loaded = release.build_release(fixture.spec, tmp_path / "release-lineage")

    def wrong_visibility_lineage(manifest):
        entry = next(
            item
            for item in manifest["inventory"]
            if item["logical_path"] == "evaluator/visibility-prf2.jsonl"
        )
        entry["derived_from_transaction_paths"] = ["train/enigma.jsonl"]

    _resign_release(loaded.root, wrong_visibility_lineage)
    with pytest.raises(release.EvaluatorReleaseError, match="lineage|derived|canonical"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )

    loaded = release.build_release(fixture.spec, tmp_path / "release-extra-key")

    def add_inventory_key(manifest):
        manifest["inventory"][0]["unexpected"] = True

    _resign_release(loaded.root, add_inventory_key)
    with pytest.raises(release.EvaluatorReleaseError, match="fields|exact"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )

    binding = fixture.spec.generation
    changed_inventory = {
        **binding.inventory,
        "eval/prf2.jsonl": {
            **binding.inventory["eval/prf2.jsonl"],
            "schema": "arbitrary-eval/v99",
        },
    }
    with pytest.raises(release.EvaluatorReleaseError, match="inventory|schema"):
        release.plan_release(
            replace(
                fixture.spec,
                generation=replace(binding, inventory=changed_inventory),
            )
        )


@pytest.mark.parametrize("logical_path", tuple(release.CANONICAL_PATH_CONTRACT))
@pytest.mark.parametrize("field", ("role", "schema"))
def test_resigned_mutation_matrix_matches_every_canonical_contract(
    fixture,
    tmp_path,
    logical_path,
    field,
):
    loaded = release.build_release(
        fixture.spec,
        tmp_path / f"release-{Path(logical_path).name}",
    )

    def mutate_contract(manifest):
        entry = next(
            item for item in manifest["inventory"] if item["logical_path"] == logical_path
        )
        entry[field] = f"counterfeit-{field}/v99"

    _resign_release(loaded.root, mutate_contract)
    with pytest.raises(release.EvaluatorReleaseError, match="canonical|role|schema"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )


def test_malformed_eval_row_is_rejected_before_packaging(fixture, tmp_path):
    copied = tmp_path / "malformed.jsonl"
    row = _family_rows("isabelle")[1]
    row.pop("target")
    copied.write_bytes(_canonical_bytes(row))
    artifact = release.InputArtifact(
        path=copied,
        transaction_path="eval/isabelle.jsonl",
    )
    eval_files = {**fixture.spec.eval_files, "isabelle": artifact}
    with pytest.raises(release.EvaluatorReleaseError, match="approved transaction|target"):
        release.plan_release(replace(fixture.spec, eval_files=eval_files))


def test_same_size_eval_mutation_and_stale_visibility_fail_load(fixture, tmp_path):
    loaded = release.build_release(fixture.spec, tmp_path / "release")
    eval_path = loaded.family_paths["metamath"]
    original = eval_path.read_bytes()
    eval_path.write_bytes(original.replace(b"held statement", b"told statement", 1))
    assert eval_path.stat().st_size == len(original)
    with pytest.raises(release.EvaluatorReleaseError, match="sha256"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )

    loaded = release.build_release(fixture.spec, tmp_path / "release-2")
    visibility = loaded.train_visibility_paths["prf2"]
    original = visibility.read_bytes()
    visibility.write_bytes(original.replace(b"t7_article", b"x7_article", 1))
    assert visibility.stat().st_size == len(original)
    with pytest.raises(release.EvaluatorReleaseError, match="sha256"):
        release.load_release(
            loaded.root, expected_version=release.UNPUBLISHED_VERSION
        )


@pytest.mark.parametrize(
    "field",
    [
        "expected_semantic_holdout_root_sha256",
        "expected_tokenizer_seal_sha256",
        "expected_source_manifests_root_sha256",
    ],
)
def test_wrong_semantic_tokenizer_or_source_root_is_rejected(fixture, field):
    with pytest.raises(release.EvaluatorReleaseError, match="root|seal"):
        release.plan_release(replace(fixture.spec, **{field: _sha("wrong")}))


def test_partial_write_missing_extra_family_and_generation_drift_reject(
    fixture,
    tmp_path,
):
    missing = dict(fixture.spec.eval_files)
    missing.pop("isabelle")
    with pytest.raises(release.EvaluatorReleaseError, match="exact families"):
        release.plan_release(replace(fixture.spec, eval_files=missing))
    extra = {**fixture.spec.eval_files, "lean": fixture.spec.eval_files["isabelle"]}
    with pytest.raises(release.EvaluatorReleaseError, match="exact families"):
        release.plan_release(replace(fixture.spec, eval_files=extra))
    with pytest.raises(release.EvaluatorReleaseError, match="logical root"):
        release.plan_release(
            replace(
                fixture.spec,
                generation=replace(
                    fixture.spec.generation,
                    logical_root_sha256=_sha("stale-generation"),
                ),
            )
        )

    complete = release.build_release(fixture.spec, tmp_path / "complete")
    partial = tmp_path / "partial"
    (partial / "evaluator").mkdir(parents=True)
    shutil.copy2(complete.manifest_path, partial / release.MANIFEST_NAME)
    with pytest.raises(release.EvaluatorReleaseError, match="completion"):
        release.load_release(
            partial, expected_version=release.UNPUBLISHED_VERSION
        )


def test_typed_evaluator_transaction_is_immutable_atomic_and_fault_aware(
    fixture,
    tmp_path,
):
    plan = release.plan_release(fixture.spec)
    coordinator = release.EvaluatorReleaseCoordinator(tmp_path / "evaluator-transaction")
    published = coordinator.publish(plan)
    assert published.root == coordinator.root / "generations" / GENERATION_ID
    assert coordinator.resolve_current().manifest_root_sha256 == plan.manifest_root_sha256
    assert json.loads((coordinator.root / "CURRENT").read_text())["schema_version"] == (
        CURRENT_SCHEMA_VERSION
    )

    faulted = release.EvaluatorReleaseCoordinator(tmp_path / "faulted")

    def fail(phase, _path):
        if phase is PublishPhase.CURRENT_REPLACE_BEFORE:
            raise RuntimeError("injected")

    with pytest.raises(RuntimeError, match="injected"):
        faulted.publish(plan, fault_injector=fail)
    assert not (faulted.root / "CURRENT").exists()

    uncertain = release.EvaluatorReleaseCoordinator(tmp_path / "uncertain")

    def fail_after_commit(phase, _path):
        if phase is PublishPhase.ROOT_FSYNC_BEFORE:
            raise RuntimeError("post-commit injected")

    with pytest.raises(CommitUncertainError) as error:
        uncertain.publish(plan, fault_injector=fail_after_commit)
    assert (uncertain.root / "CURRENT").exists()
    recovered = error.value.resolve(uncertain.transaction_coordinator)
    assert recovered.commit_state == "durable_recovered"
    assert uncertain.resolve_current().manifest_root_sha256 == plan.manifest_root_sha256


def test_reserved_platform_group_manifest_path_never_collides(fixture, tmp_path):
    loaded = release.build_release(fixture.spec, tmp_path / "release")
    assert release.MANIFEST_NAME == "evaluator/release-manifest.json"
    assert loaded.seal["manifest_path"] == release.MANIFEST_NAME
    reserved = loaded.root / "evaluator/manifest.json"
    reserved.write_bytes(b'{"platform":"generated"}\n')
    with pytest.raises(release.EvaluatorReleaseError, match="extra|inventory"):
        release.load_release(
            loaded.root,
            expected_version=release.UNPUBLISHED_VERSION,
        )


def test_fresh_coordinator_resolves_durable_and_commit_uncertain_release(
    fixture,
    tmp_path,
):
    plan = release.plan_release(fixture.spec)
    root = tmp_path / "durable"
    release.EvaluatorReleaseCoordinator(root).publish(plan)
    restarted = release.EvaluatorReleaseCoordinator(root)
    assert restarted.resolve_current().manifest_root_sha256 == plan.manifest_root_sha256
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    policy_path = tmp_path / "production-policy.pkl"
    policy_path.write_bytes(pickle.dumps(fixture.production_policy))
    process_code = (
        "import pickle,sys;"
        f"sys.path.insert(0, {str(scripts)!r});"
        "import build_evaluator_release as release;"
        "import split_mml_semantic_holdout as holdout;"
        f"policy=pickle.loads(open({str(policy_path)!r},'rb').read());"
        "holdout.production_source_policy=lambda:policy;"
        f"resolved=release.EvaluatorReleaseCoordinator({str(root)!r}).resolve_current();"
        "print(resolved.manifest_root_sha256)"
    )
    assert (
        subprocess.check_output(
            [sys.executable, "-c", process_code],
            text=True,
        ).strip()
        == plan.manifest_root_sha256
    )

    uncertain_root = tmp_path / "uncertain-restart"

    def fail_after_commit(phase, _path):
        if phase is PublishPhase.ROOT_FSYNC_BEFORE:
            raise RuntimeError("uncertain restart")

    with pytest.raises(CommitUncertainError):
        release.EvaluatorReleaseCoordinator(uncertain_root).publish(
            plan,
            fault_injector=fail_after_commit,
        )
    restarted = release.EvaluatorReleaseCoordinator(uncertain_root)
    assert restarted.resolve_current().manifest_root_sha256 == plan.manifest_root_sha256


def test_fresh_loader_uses_persisted_semantic_contract_not_mutable_workspace(
    fixture,
    monkeypatch,
    tmp_path,
):
    plan = release.plan_release(fixture.spec)
    root = tmp_path / "persisted-contract"
    release.EvaluatorReleaseCoordinator(root).publish(plan)

    def changed_policy():
        raise holdout.HoldoutError("workspace policy changed after publication")

    def changed_projection(_manifest):
        return {
            family: {"family": family, "projection_root_sha256": _sha(f"changed-{family}")}
            for family in release.MML_PROJECTIONS
        }

    monkeypatch.setattr(holdout, "production_source_policy", changed_policy)
    monkeypatch.setattr(holdout, "derive_compatibility_projections", changed_projection)
    loaded = release.EvaluatorReleaseCoordinator(root).resolve_current()

    assert loaded.manifest_root_sha256 == plan.manifest_root_sha256
    assert loaded.provenance["semantic_holdout"]["manifest_root_sha256"] == (
        fixture.semantic_contract.authoritative_root
    )


def _reordered_mapping(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: _reordered_mapping(item, rng) for key, item in items}
    if isinstance(value, list):
        return [_reordered_mapping(item, rng) for item in value]
    return value


def test_projection_canonicalization_ignores_mapping_insertion_order(fixture):
    authoritative = fixture.semantic_contract.manifest
    expected = release.canonical_mml_projections(authoritative)
    for seed in range(50):
        reordered = _reordered_mapping(authoritative, random.Random(seed))
        assert release.canonical_mml_projections(reordered) == expected


def test_fifty_parallel_fresh_process_cycles_are_tree_deterministic(
    fixture,
    tmp_path,
):
    plan = release.plan_release(fixture.spec)
    roots = [tmp_path / f"cycle-{cycle:02d}" for cycle in range(50)]

    def publish(root: Path) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        loaded = release.EvaluatorReleaseCoordinator(root).publish(plan)
        tree = tuple(
            sorted(
                (
                    entry["logical_path"],
                    entry["sha256"],
                )
                for entry in loaded.manifest["inventory"]
            )
        )
        return loaded.manifest_root_sha256, loaded.seal_sha256, tree

    with ThreadPoolExecutor(max_workers=8) as executor:
        published = list(executor.map(publish, roots))
    assert len(set(published)) == 1

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    process_code = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(scripts)!r})",
            "import build_evaluator_release as release",
            "import split_mml_semantic_holdout as holdout",
            "original = holdout.derive_compatibility_projections",
            "def reordered(manifest):",
            "    projections = original(manifest)",
            "    for projection in projections.values():",
            "        projection['classes'] = list(reversed(projection['classes']))",
            "    return projections",
            "holdout.derive_compatibility_projections = reordered",
            (
                "holdout.production_source_policy = lambda: (_ for _ in ()).throw("
                "holdout.HoldoutError('workspace source policy changed'))"
            ),
            "loaded = release.EvaluatorReleaseCoordinator(sys.argv[1]).resolve_current()",
            "print(loaded.manifest_root_sha256, loaded.seal_sha256)",
        )
    )

    def reload_in_fresh_process(item: tuple[int, Path]) -> str:
        seed, root = item
        environment = {**os.environ, "PYTHONHASHSEED": str(seed)}
        return subprocess.check_output(
            [sys.executable, "-c", process_code, str(root)],
            env=environment,
            text=True,
        ).strip()

    with ThreadPoolExecutor(max_workers=8) as executor:
        reloaded = list(executor.map(reload_in_fresh_process, enumerate(roots)))
    expected = (
        f"{plan.manifest_root_sha256} "
        f"{hashlib.sha256(plan.completion_seal_bytes).hexdigest()}"
    )
    assert reloaded == [expected] * 50


@pytest.mark.parametrize("phase", tuple(PublishPhase))
def test_evaluator_publication_inherits_complete_transaction_fault_matrix(
    fixture,
    tmp_path,
    phase,
):
    plan = release.plan_release(fixture.spec)
    coordinator = release.EvaluatorReleaseCoordinator(tmp_path / f"fault-{phase.value}")

    def fail(actual, _path):
        if actual is phase:
            raise RuntimeError(f"injected {phase.value}")

    if phase is PublishPhase.ROOT_FSYNC_AFTER:
        loaded = coordinator.publish(plan, fault_injector=fail)
        assert loaded.manifest_root_sha256 == plan.manifest_root_sha256
        assert (coordinator.root / "CURRENT").exists()
    elif phase in {
        PublishPhase.CURRENT_REPLACE_AFTER,
        PublishPhase.ROOT_FSYNC_BEFORE,
    }:
        with pytest.raises(CommitUncertainError):
            coordinator.publish(plan, fault_injector=fail)
        assert (coordinator.root / "CURRENT").exists()
    else:
        with pytest.raises(RuntimeError, match="injected"):
            coordinator.publish(plan, fault_injector=fail)
        assert not (coordinator.root / "CURRENT").exists()


def test_deterministic_rerun_has_identical_logical_release(fixture, tmp_path):
    first_plan = release.plan_release(fixture.spec)
    second_plan = release.plan_release(fixture.spec)
    assert first_plan.manifest_bytes == second_plan.manifest_bytes
    assert first_plan.completion_seal_bytes == second_plan.completion_seal_bytes
    first = release.EvaluatorReleaseCoordinator(tmp_path / "first").publish(first_plan)
    second = release.EvaluatorReleaseCoordinator(tmp_path / "second").publish(second_plan)
    assert first.manifest_root_sha256 == second.manifest_root_sha256
    assert first.seal_sha256 == second.seal_sha256


def test_token_dependency_requires_all_caller_pins_and_rejects_drift(
    fixture,
    tmp_path,
):
    loaded = release.build_release(
        replace(fixture.spec, version="v7"),
        tmp_path / "release",
    )
    dependency = release.bind_published_dependency(
        loaded,
        version="v7",
        platform_group_manifest_sha256=_sha("platform-group"),
    )
    expected = {
        "expected_dataset_id": release.APPROVED_DATASET_ID,
        "expected_version": "v7",
        "expected_platform_group_manifest_sha256": _sha("platform-group"),
        "expected_evaluator_manifest_root_sha256": loaded.manifest_root_sha256,
        "expected_evaluator_seal_sha256": loaded.seal_sha256,
    }
    assert release.require_token_dependency(dependency, loaded, **expected) == dependency
    with pytest.raises(TypeError):
        release.require_token_dependency(dependency, loaded)
    with pytest.raises(release.EvaluatorReleaseError, match="required"):
        release.require_token_dependency(None, loaded, **expected)
    for field in (
        "dataset_id",
        "version",
        "manifest_sha256",
        "evaluator_manifest_root_sha256",
        "evaluator_seal_sha256",
    ):
        drifted = {**dependency, field: "v8" if field == "version" else _sha(field)}
        with pytest.raises(release.EvaluatorReleaseError, match="dependency"):
            release.require_token_dependency(drifted, loaded, **expected)


def test_manifest_and_completion_schemas_validate(fixture, tmp_path):
    loaded = release.build_release(fixture.spec, tmp_path / "release")
    docs = Path(__file__).resolve().parents[1] / "docs"
    manifest_schema = json.loads(
        (docs / "p3-evaluator-release-v1.schema.json").read_text()
    )
    completion_schema = json.loads(
        (docs / "p3-evaluator-release-completion-v1.schema.json").read_text()
    )
    generated_manifest_schema = release.manifest_json_schema()
    generated_completion_schema = release.completion_json_schema()
    assert manifest_schema == generated_manifest_schema
    assert completion_schema == generated_completion_schema
    Draft202012Validator.check_schema(manifest_schema)
    Draft202012Validator.check_schema(completion_schema)
    Draft202012Validator(manifest_schema).validate(loaded.manifest)
    Draft202012Validator(completion_schema).validate(loaded.seal)

    for entry in loaded.manifest["inventory"]:
        for field in ("role", "schema"):
            mutated = deepcopy(loaded.manifest)
            target = next(
                item
                for item in mutated["inventory"]
                if item["logical_path"] == entry["logical_path"]
            )
            target[field] = f"counterfeit-{field}/v99"
            with pytest.raises(JsonSchemaValidationError):
                Draft202012Validator(manifest_schema).validate(mutated)


def test_exact_dataset_and_explicit_version_contract(fixture, tmp_path):
    with pytest.raises(release.EvaluatorReleaseError, match="dataset"):
        release.plan_release(
            replace(fixture.spec, dataset_id="eval/counterfeit-release")
        )
    with pytest.raises(release.EvaluatorReleaseError, match="version"):
        release.plan_release(replace(fixture.spec, version="latest"))

    local = release.build_release(fixture.spec, tmp_path / "local")
    with pytest.raises(release.EvaluatorReleaseError, match="version"):
        release.load_release(local.root, expected_version="v7")
    assert release.load_release(
        local.root,
        expected_version=release.UNPUBLISHED_VERSION,
    ).manifest["version"] == release.UNPUBLISHED_VERSION

    schema = release.manifest_json_schema()
    for field, value in (
        ("dataset_id", "eval/counterfeit-release"),
        ("version", "latest"),
    ):
        mutated = deepcopy(local.manifest)
        mutated[field] = value
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(schema).validate(mutated)

        resigned = release.build_release(
            fixture.spec,
            tmp_path / f"resigned-{field}",
        )
        _resign_release(
            resigned.root,
            lambda manifest, field=field, value=value: manifest.__setitem__(
                field, value
            ),
        )
        with pytest.raises(release.EvaluatorReleaseError, match="dataset|version"):
            release.load_release(
                resigned.root,
                expected_version=release.UNPUBLISHED_VERSION,
            )

    production_spec = replace(fixture.spec, version="v7")
    production = release.build_release(production_spec, tmp_path / "production")
    assert release.load_release(
        production.root,
        expected_version="v7",
    ).manifest["version"] == "v7"


def test_platform_docs_define_self_registering_profile_and_no_new_ui():
    docs = (
        Path(__file__).resolve().parents[1] / "docs" / "p3-evaluator-release-v1.md"
    ).read_text()
    assert "five shipped profiles" in docs
    assert "no `text-corpus/v1`" in docs
    assert "registry.register(sys.modules[__name__])" in docs
    assert "No evaluator selector" in docs
    assert "two path labels" in docs
    assert "group manifest SHA-256" in docs
