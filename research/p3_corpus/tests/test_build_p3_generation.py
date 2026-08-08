"""Production-integration tests for the six-family P3 generation transaction."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_p3_generation as generation
from scripts.corpus_generation_transaction import GenerationCoordinator

atp_builder = importlib.import_module("scripts.build_atp_shard")
direct_mizar = importlib.import_module("scripts.build_mizar_human_shard")
isabelle_builder = importlib.import_module("scripts.build_isabelle_shard")
metamath_builder = importlib.import_module("scripts.build_metamath_shard")
thproofs = importlib.import_module("scripts.build_thproofs_shard")
mizar_current_index = importlib.import_module("scripts.mizar_current_index")

FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXED_QWEN_TOKENIZER = REPOSITORY_ROOT / "tokenizers" / "qwen25-vendored"
METAMATH_16K_CANDIDATE = (
    REPOSITORY_ROOT / ".p3-work" / "full13" / "metamath-16k-v1"
)
JSON_OUTPUT_CASES = tuple(
    (
        spec.path,
        spec.schema,
        tuple(spec.validator.required_fields),
    )
    for spec in generation.make_generation_plan(
        generation_id="json-contract-cases",
        source_generation_id="json-contract-cases",
    ).outputs
    if spec.path.endswith(".json")
)


def _make_metamath_drop_ledger(
    entries,
    *,
    eligible_rows,
    eligible_tokens=100,
    tokenizer_seal=None,
):
    tokenizer_seal = (
        generation.mml_holdout.approved_tokenizer_seal()
        if tokenizer_seal is None
        else tokenizer_seal
    )
    dropped_tokens = sum(entry["text_plus_eos_tokens"] for entry in entries)
    accounting = {
        "source_rows": eligible_rows + len(entries),
        "eligible_rows": eligible_rows,
        "dropped_rows": len(entries),
        "source_text_plus_eos_tokens": eligible_tokens + dropped_tokens,
        "eligible_text_plus_eos_tokens": eligible_tokens,
        "dropped_text_plus_eos_tokens": dropped_tokens,
        "dropped_excess_tokens": sum(
            entry["text_plus_eos_tokens"]
            - metamath_builder.MAX_TEXT_PLUS_EOS_TOKENS
            for entry in entries
        ),
    }
    ledger = metamath_builder._drop_ledger_body(
        list(entries),
        accounting,
        tokenizer_seal,
        metamath_builder.MAX_TEXT_PLUS_EOS_TOKENS,
    )
    ledger["canonical_root_sha256"] = metamath_builder.canonical_sha256(ledger)
    metamath_builder.validate_drop_ledger(ledger)
    return ledger


def _metamath_drop_entry(row_id="overlength", theorem="set:overlength"):
    return {
        "schema_version": metamath_builder.DROP_ENTRY_SCHEMA,
        "id": row_id,
        "theorem": theorem,
        "text_plus_eos_tokens": 16_385,
        "native_row_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
        "reason_schema_version": metamath_builder.DROP_REASON_SCHEMA,
        "reason": metamath_builder.OVERLENGTH_DROP_REASON,
    }


def _resign_metamath_drop_ledger(ledger):
    entries = ledger["entries"]
    accounting = ledger["accounting"]
    dropped_tokens = sum(entry["text_plus_eos_tokens"] for entry in entries)
    accounting["dropped_rows"] = len(entries)
    accounting["dropped_text_plus_eos_tokens"] = dropped_tokens
    accounting["dropped_excess_tokens"] = sum(
        entry["text_plus_eos_tokens"]
        - metamath_builder.MAX_TEXT_PLUS_EOS_TOKENS
        for entry in entries
    )
    accounting["source_rows"] = accounting["eligible_rows"] + len(entries)
    accounting["source_text_plus_eos_tokens"] = (
        accounting["eligible_text_plus_eos_tokens"] + dropped_tokens
    )
    ledger["tokenizer_root_sha256"] = metamath_builder.canonical_sha256(
        ledger["tokenizer_seal"]
    )
    ledger["entries_root_sha256"] = metamath_builder.canonical_sha256(entries)
    ledger.pop("canonical_root_sha256", None)
    ledger["canonical_root_sha256"] = metamath_builder.canonical_sha256(ledger)
    return ledger


def _tree_payload(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


@pytest.fixture(scope="module")
def repository_tokenizer_seal():
    seal = generation.mml_holdout.approved_tokenizer_seal()
    assert (
        hashlib.sha256(
            (FIXED_QWEN_TOKENIZER / "tokenizer.json").read_bytes()
        ).hexdigest()
        == seal["tokenizer_json_sha256"]
    )
    assert (
        hashlib.sha256(
            (FIXED_QWEN_TOKENIZER / "tokenizer_config.json").read_bytes()
        ).hexdigest()
        == seal["tokenizer_config_sha256"]
    )
    return seal


@pytest.fixture
def prepared_tokenizer_seal(repository_tokenizer_seal):
    return dict(repository_tokenizer_seal)


def test_synthetic_six_family_generation_is_reproducible_and_deep_clean(tmp_path):
    root = tmp_path / "transaction"
    work = tmp_path / "trusted-work"
    work.mkdir()
    legacy = tmp_path / "legacy-corpus"
    (legacy / "shards").mkdir(parents=True)
    sentinel = legacy / "shards" / "metamath.jsonl"
    sentinel.write_bytes(b"legacy bytes must remain untouched\n")
    legacy_before = _tree_payload(legacy)

    first = generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="synthetic-a",
        forbidden_legacy_paths=(legacy,),
    )
    second = generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="synthetic-b",
        forbidden_legacy_paths=(legacy,),
    )

    resolved = GenerationCoordinator(root).resolve_current(
        required_siblings=FAMILIES,
    )
    report = generation.verify_generation(root, production=False)

    assert resolved.generation_id == "synthetic-b"
    assert tuple(resolved.manifest["requested_siblings"]) == FAMILIES
    assert first.published.logical_root_sha256 == second.published.logical_root_sha256
    assert report["status"] == "clean"
    assert report["families"] == list(FAMILIES)
    assert report["mml_selected_classes"] == 1_000
    assert report["modes"] == {
        "isabelle": "family_local_heldout",
        "metamath": "family_local_heldout",
        "mml": "pooled_semantic_1000",
        "production": False,
    }
    assert _tree_payload(legacy) == legacy_before
    assert first.builder_output_roots
    assert second.builder_output_roots
    for output_root in (*first.builder_output_roots, *second.builder_output_roots):
        assert output_root.is_relative_to(work)
        assert not output_root.is_relative_to(legacy)

    occurrence_index = json.loads(
        (resolved.path / "sidecars" / "occurrences.json").read_text()
    )
    assert occurrence_index["families"] == list(FAMILIES)
    assert occurrence_index["occurrences"]
    for occurrence in occurrence_index["occurrences"]:
        assert occurrence["source_line"] >= 1
        assert occurrence["byte_start"] >= 0
        assert occurrence["byte_end"] > occurrence["byte_start"]
        assert len(occurrence["raw_sha256"]) == 64


@pytest.mark.parametrize(
    ("relative", "mutate", "message"),
    (
        (
            "heldout/metamath.json",
            lambda payload: payload["overlength_drop_ledger"]["entries"][0].__setitem__(
                "native_row_sha256",
                "0" * 64,
            ),
            "drop ledger|native",
        ),
        (
            "sidecars/occurrences.json",
            lambda payload: payload["source_accounting"]["metamath"].__setitem__(
                "native_occurrences_root_sha256",
                "0" * 64,
            ),
            "occurrence|accounting",
        ),
        (
            "sidecars/precheck.json",
            lambda payload: payload["source_accounting"]["metamath"].__setitem__(
                "accounting_root_sha256",
                "0" * 64,
            ),
            "precheck|accounting",
        ),
    ),
)
def test_generation_persists_recomputable_metamath_source_accounting(
    tmp_path,
    relative,
    mutate,
    message,
):
    ledger = _make_metamath_drop_ledger(
        [_metamath_drop_entry()],
        eligible_rows=3,
    )
    result = generation.build_synthetic_generation(
        corpus_root=tmp_path / "transaction",
        work_root=tmp_path,
        generation_id="metamath-source-accounting",
        metamath_drop_ledger=ledger,
    )
    generation._independent_verify_generation_path(
        result.published.path,
        production=False,
    )
    heldout = json.loads(
        (result.published.path / "heldout" / "metamath.json").read_text()
    )
    accounting = heldout["source_accounting"]
    assert accounting["source_rows"] == 4
    assert accounting["train_rows"] == 1
    assert accounting["eval_rows"] == 1
    assert accounting["drop_types"] == {
        "heldout_own_proof": 1,
        "overlength": 1,
    }
    path = result.published.path / relative
    path.chmod(0o644)
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(generation.IntegrationError, match=message):
        generation._independent_verify_generation_path(
            result.published.path,
            production=False,
        )


@pytest.mark.parametrize("family", FAMILIES)
def test_builder_crash_at_each_family_preserves_current_and_quarantines(
    tmp_path,
    family,
):
    root = tmp_path / "transaction"
    work = tmp_path / "trusted-work"
    work.mkdir()
    baseline = generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="baseline",
    )
    old_tree = _tree_payload(baseline.published.path)

    with pytest.raises(generation.IntegrationError, match=family):
        generation.build_synthetic_generation(
            corpus_root=root,
            work_root=work,
            generation_id=f"crash-{family}",
            fail_family=family,
        )

    coordinator = GenerationCoordinator(root)
    assert coordinator.resolve_current().generation_id == "baseline"
    assert _tree_payload(baseline.published.path) == old_tree
    assert coordinator.quarantine_inventory()
    assert generation.builder_quarantine_inventory(work)


def test_legacy_layout_at_transaction_root_is_rejected_before_builders_run(tmp_path):
    root = tmp_path / "transaction"
    (root / "shards").mkdir(parents=True)
    (root / "shards" / "metamath.jsonl").write_text("{}\n")
    work = tmp_path / "trusted-work"
    work.mkdir()

    with pytest.raises(generation.IntegrationError, match="legacy"):
        generation.build_synthetic_generation(
            corpus_root=root,
            work_root=work,
            generation_id="must-not-run",
        )

    assert not (root / "CURRENT").exists()
    assert not list(work.iterdir())


def test_duplicate_raw_row_id_is_rejected_before_current_changes(tmp_path):
    root = tmp_path / "transaction"
    work = tmp_path / "trusted-work"
    work.mkdir()
    generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="old",
    )

    with pytest.raises(generation.IntegrationError, match="duplicate raw row id"):
        generation.build_synthetic_generation(
            corpus_root=root,
            work_root=work,
            generation_id="duplicate-id",
            duplicate_id_family="mizar",
        )

    assert GenerationCoordinator(root).resolve_current().generation_id == "old"


@pytest.mark.parametrize(
    ("family", "mutation", "message"),
    [
        (
            "metamath",
            lambda row: row["facts"].pop(row["cited"][0]),
            "cited|fact",
        ),
        (
            "mizar",
            lambda row: row.__setitem__("mask_end", row["mask_end"] - 1),
            "mask|reconstruct",
        ),
        (
            "thproofs",
            lambda row: row.__setitem__("text", row["text"] + " drift"),
            "text|reconstruct",
        ),
        (
            "prf2",
            lambda row: row["proof_steps"][0]["parents"].append("missing-parent"),
            "parent|reference",
        ),
        (
            "enigma",
            lambda row: row.__setitem__("schema_version", "atp-v1"),
            "schema",
        ),
        (
            "isabelle",
            lambda row: row.__setitem__("state_after", "stale state"),
            "target|state_after|reconstruct",
        ),
    ],
)
def test_every_current_family_schema_mutant_is_rejected(family, mutation, message):
    row, source_manifest = generation.synthetic_family_record(family)
    mutation(row)

    with pytest.raises(generation.IntegrationError, match=message):
        generation.validate_family_record(
            row,
            family=family,
            source_manifest=source_manifest,
            location=f"raw/{family}.jsonl:1",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["source_metadata"].__setitem__(
                "source_manifest_root_sha256",
                "0" * 64,
            ),
            "source.manifest root",
        ),
        (
            lambda row: row["source_metadata"]["index_roots"].__setitem__(
                "semantic_index_sha256",
                "0" * 64,
            ),
            "index root",
        ),
        (
            lambda row: row["source_metadata"].__setitem__(
                "schema_generation_root_sha256",
                "0" * 64,
            ),
            "schema.generation root",
        ),
    ],
)
def test_stale_source_index_and_schema_roots_are_rejected(mutation, message):
    row, source_manifest = generation.synthetic_family_record("thproofs")
    mutation(row)

    with pytest.raises(generation.IntegrationError, match=message):
        generation.validate_family_record(
            row,
            family="thproofs",
            source_manifest=source_manifest,
            location="raw/thproofs.jsonl:1",
        )


def test_production_readiness_reports_only_missing_technical_inputs():
    blockers = generation.production_blockers()

    assert any("six-family source manifests" in blocker for blocker in blockers)
    assert not any("license" in blocker.lower() for blocker in blockers)
    assert not any("Metamath" in blocker for blocker in blockers)


def test_thproofs_declares_direct_mizar_duplicate_drop():
    assert (
        "direct_mizar_trajectory_duplicate"
        in generation.DROP_TYPES["thproofs"]
    )


def test_cli_refuses_real_build_before_creating_output(tmp_path, monkeypatch):
    output = tmp_path / "transaction"
    work = tmp_path / "trusted-work"
    work.mkdir()
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}")
    policies = tmp_path / "policies.json"
    policies.write_text("{}")
    manifests = []
    for family in FAMILIES:
        path = tmp_path / f"{family}.json"
        path.write_text("{}")
        manifests.extend(["--source-manifest", f"{family}={path}"])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_p3_generation.py",
            "--corpus-root",
            str(output),
            "--work-root",
            str(work),
            "--generation-id",
            "production-v2",
            "--tokenizer-seal",
            str(tokenizer),
            "--policies",
            str(policies),
            *manifests,
        ],
    )

    assert generation.main() == 2
    assert not output.exists()
    assert not list(work.iterdir())


def test_occurrence_index_hashes_exact_raw_byte_ranges(tmp_path):
    result = generation.build_synthetic_generation(
        corpus_root=tmp_path / "transaction",
        work_root=tmp_path,
        generation_id="byte-identities",
    )
    payload = json.loads(
        (result.published.path / "sidecars" / "occurrences.json").read_text()
    )

    for occurrence in payload["occurrences"]:
        raw = result.published.path / occurrence["raw_path"]
        data = raw.read_bytes()
        exact = data[occurrence["byte_start"] : occurrence["byte_end"]]
        assert hashlib.sha256(exact).hexdigest() == occurrence["raw_sha256"]
        assert exact.endswith(b"\n")


def _stage_declaration(family):
    row, _ = generation.synthetic_family_record(family)
    return (
        {
            "path": "raw.jsonl",
            "kind": "file",
            "format": "jsonl",
            "schema": generation.ROW_SCHEMAS[family],
            "source_manifest_root_sha256": row["source_metadata"][
                "source_manifest_root_sha256"
            ],
        },
    )


def _stage_callback(family, mutation):
    row, _ = generation.synthetic_family_record(family)

    def callback(root):
        raw = root / "raw.jsonl"
        raw.write_text(json.dumps(row) + "\n", encoding="utf-8")
        if mutation == "extra-file":
            (root / "surprise.txt").write_text("extra")
        elif mutation == "extra-directory":
            (root / "nested").mkdir()
        elif mutation == "nested-extra":
            (root / "nested").mkdir()
            (root / "nested" / "surprise.json").write_text("{}")
        elif mutation == "symlink":
            (root / "link").symlink_to(raw)
        elif mutation == "temp":
            (root / ".raw.jsonl.tmp").write_text("temporary")
        elif mutation == "missing":
            raw.unlink()
        elif mutation == "wrong-schema":
            payload = json.loads(raw.read_text())
            payload["schema_version"] = "wrong-v9"
            raw.write_text(json.dumps(payload) + "\n")
        elif mutation == "wrong-root":
            payload = json.loads(raw.read_text())
            payload["source_metadata"]["source_manifest_root_sha256"] = "0" * 64
            raw.write_text(json.dumps(payload) + "\n")

    return callback


@pytest.mark.parametrize("family", FAMILIES)
def test_builder_callback_stage_accepts_only_exact_declared_inventory(
    tmp_path,
    family,
):
    result = generation._run_builder_callback_stage(
        family=family,
        stage="raw",
        output_root=tmp_path / family,
        inventory=_stage_declaration(family),
        callback=_stage_callback(family, "clean"),
    )

    assert result == {"raw.jsonl": tmp_path / family / "raw.jsonl"}


def test_external_builder_receives_a_fresh_output_root(tmp_path, monkeypatch):
    row, source_manifest = generation.synthetic_family_record("metamath")
    source_root = source_manifest["row_source_metadata"][
        "source_manifest_root_sha256"
    ]
    output_root = tmp_path / "work" / "metamath" / "raw-build"

    def run(argv, **kwargs):
        del kwargs
        out = Path(argv[argv.index("--out") + 1])
        assert out == output_root
        assert not out.exists()
        out.mkdir()
        generation._write_jsonl(out / "raw.jsonl", [row])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(generation.subprocess, "run", run)
    result = generation._run_builder_stage(
        family="metamath",
        stage="raw",
        specification={
            "argv": [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "build_metamath_shard.py"),
                "--mm-dir",
                str(tmp_path / "mm"),
                "--heldout",
                "0",
                "--seed",
                "20260801",
                "--tokenizer",
                str(FIXED_QWEN_TOKENIZER),
            ],
            "inventory": [
                {
                    "path": "raw.jsonl",
                    "kind": "file",
                    "format": "jsonl",
                    "schema": generation.ROW_SCHEMAS["metamath"],
                    "source_manifest_root_sha256": source_root,
                }
            ],
            "outputs": {"raw": "raw.jsonl"},
        },
        output_root=output_root,
        corpus_root=tmp_path / "transaction",
        forbidden_legacy_paths=(),
    )

    assert result == {"raw": output_root / "raw.jsonl"}


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "mutation",
    [
        "extra-file",
        "extra-directory",
        "nested-extra",
        "symlink",
        "temp",
        "missing",
        "wrong-schema",
        "wrong-root",
    ],
)
def test_every_builder_rejects_hostile_recursive_stage_inventory(
    tmp_path,
    family,
    mutation,
):
    with pytest.raises(
        generation.IntegrationError,
        match="inventory|undeclared|missing|symlink|schema|root|temporary",
    ):
        generation._run_builder_callback_stage(
            family=family,
            stage="raw",
            output_root=tmp_path / f"{family}-{mutation}",
            inventory=_stage_declaration(family),
            callback=_stage_callback(family, mutation),
        )


def _builder_heldout_payload(family):
    if family == "metamath":
        return {
            "schema_version": "metamath-heldout-v2",
            "family": "metamath",
            "mode": "family_local_heldout",
            "facts": ["mp"],
            "requested_heldout": 1,
            "local_assumptions": True,
        }
    return {
        "schema_version": "isabelle-transition-v2",
        "family": "isabelle",
        "mode": "family_local_heldout",
        "facts": ["Global.fact"],
        "statements": {"Global.fact": "global statement"},
        "requested_heldout": 1,
        "trajectory_drops": True,
    }


@pytest.mark.parametrize("family", ("metamath", "isabelle"))
@pytest.mark.parametrize("mutation", ("missing-schema", "wrong-schema", "missing-field"))
def test_structured_builder_json_requires_internal_schema_and_contract_fields(
    tmp_path,
    family,
    mutation,
):
    payload = _builder_heldout_payload(family)
    required = tuple(key for key in payload if key != "schema_version")
    expected_schema = payload["schema_version"]
    if mutation == "missing-schema":
        payload.pop("schema_version")
    elif mutation == "wrong-schema":
        payload["schema_version"] = "wrong-heldout-v9"
    else:
        payload.pop(required[0])

    def callback(root):
        (root / "heldout.json").write_text(json.dumps(payload) + "\n")

    with pytest.raises(generation.IntegrationError, match="schema|required"):
        generation._run_builder_callback_stage(
            family=family,
            stage="split",
            output_root=tmp_path / f"{family}-{mutation}",
            inventory=[
                {
                    "path": "heldout.json",
                    "kind": "file",
                    "format": "json",
                    "schema": expected_schema,
                    "required_fields": list(required),
                }
            ],
            callback=callback,
        )


def _metamath_normalization_fixture(tmp_path):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    metadata = source_manifest["row_source_metadata"]
    rows = [
        generation._metamath_record(
            "train",
            metadata,
            fact_name="safe",
            fact_statement="|- ps => |- ps",
        ),
        generation._metamath_record("eval", metadata),
        generation._metamath_record(
            "drop",
            metadata,
            fact_name="safe",
            fact_statement="|- ps => |- ps",
            theorem="set:mp",
        ),
    ]
    raw = tmp_path / "raw.jsonl"
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    generation._write_jsonl(raw, rows)
    generation._write_jsonl(train, rows[:1])
    generation._write_jsonl(evaluation, rows[1:])
    heldout = tmp_path / "heldout.json"
    return source_manifest, raw, train, evaluation, heldout


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-schema",
        "wrong-schema",
        "missing-family",
        "missing-mode",
        "missing-requested",
        "missing-local-contract",
    ),
)
def test_metamath_normalization_rejects_hostile_heldout_v2_contract(
    tmp_path,
    mutation,
):
    source_manifest, raw, train, evaluation, heldout = (
        _metamath_normalization_fixture(tmp_path)
    )
    payload = {
        "schema_version": "metamath-heldout-v2",
        "family": "metamath",
        "mode": "family_local_heldout",
        "facts": ["mp", *(f"unused-{index}" for index in range(499))],
        "requested_heldout": 500,
        "local_assumptions": True,
    }
    field = {
        "missing-schema": "schema_version",
        "missing-family": "family",
        "missing-mode": "mode",
        "missing-requested": "requested_heldout",
        "missing-local-contract": "local_assumptions",
    }.get(mutation)
    if field is not None:
        payload.pop(field)
    else:
        payload["schema_version"] = "wrong-heldout-v9"
    heldout.write_text(json.dumps(payload) + "\n")

    with pytest.raises(generation.IntegrationError, match="Metamath.*heldout|schema"):
        generation._normalize_metamath_package(
            raw_output=raw,
            split_outputs={
                "train": train,
                "eval": evaluation,
                "heldout": heldout,
            },
            destination=tmp_path / "normalized",
            source_manifest=source_manifest,
        )


@pytest.mark.parametrize(
    "token",
    [
        "--test",
        "--test-only",
        "--allow-unsealed",
        "--bypass-source-check",
        "--skip-check",
        "--legacy-production",
        "/tmp/test/source",
    ],
)
def test_production_builder_argv_rejects_every_bypass_token(tmp_path, token):
    source = tmp_path / "source.json"
    source.write_text("{}")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    argv = [
        sys.executable,
        "scripts/build_isabelle_shard.py",
        "--src",
        str(source),
        "--name",
        "isabelle",
        "--heldout",
        "500",
        "--seed",
        "20260801",
        "--tokenizer-path",
        str(tokenizer),
        token,
    ]

    with pytest.raises(generation.IntegrationError, match="bypass|test|unknown"):
        generation.validate_production_builder_command(
            family="isabelle",
            stage="split",
            argv=argv,
        )


def test_production_builder_command_rejects_unknown_flag_and_heldout_drift(tmp_path):
    del tmp_path
    source = Path("/tmp/p3-production-proof-source")
    base = [
        sys.executable,
        "scripts/build_atp_shard.py",
        "--src",
        str(source),
        "--name",
        "prf2",
        "--heldout",
        "0",
        "--min-steps",
        "4",
        "--seed",
        "20260801",
    ]
    with pytest.raises(generation.IntegrationError, match="unknown"):
        generation.validate_production_builder_command(
            family="prf2",
            stage="raw",
            argv=[*base, "--mystery"],
        )
    drifted = list(base)
    drifted[drifted.index("0")] = "500"
    with pytest.raises(generation.IntegrationError, match="heldout"):
        generation.validate_production_builder_command(
            family="prf2",
            stage="raw",
            argv=drifted,
        )


def _enigma_low_tier_command(tmp_path_factory):
    root = tmp_path_factory.mktemp("p3-enigma-command")
    source = root / "mzr01"
    source.mkdir()
    accepted_base = root / "enigma-accepted-base"
    accepted_base.mkdir()
    argv = [
        sys.executable,
        "scripts/build_atp_shard.py",
        "--src",
        str(source),
        "--name",
        "enigma",
        "--fenced",
        "--heldout",
        "0",
        "--min-steps",
        "4",
        "--dedup",
        "--jaccard",
        "0.5",
        "--seed",
        "20260801",
        "--enigma-low-tier-base",
        str(accepted_base),
        "--tokenizer-json",
        str(FIXED_QWEN_TOKENIZER / "tokenizer.json"),
    ]
    return argv, accepted_base


def test_enigma_low_tier_builder_command_is_exact_and_raw_only(tmp_path_factory):
    argv, _ = _enigma_low_tier_command(tmp_path_factory)

    assert (
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=argv,
        )
        == argv
    )
    split_argv = list(argv)
    split_argv[split_argv.index("0")] = "500"
    with pytest.raises(generation.IntegrationError, match="raw"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="split",
            argv=split_argv,
        )


@pytest.mark.parametrize(
    "flag",
    ("--enigma-low-tier-base", "--tokenizer-json"),
)
@pytest.mark.parametrize(
    "family", tuple(family for family in FAMILIES if family != "enigma")
)
def test_enigma_low_tier_flags_are_rejected_for_every_other_family(
    tmp_path_factory,
    family,
    flag,
):
    root = tmp_path_factory.mktemp("p3-enigma-other-family")
    value = (
        root / "enigma-accepted-base"
        if flag == "--enigma-low-tier-base"
        else FIXED_QWEN_TOKENIZER / "tokenizer.json"
    )
    argv = [
        sys.executable,
        f"scripts/{generation.PRODUCTION_BUILDER_SCRIPTS[family]}",
        flag,
        str(value),
    ]

    with pytest.raises(generation.IntegrationError, match="unknown builder flag"):
        generation.validate_production_builder_command(
            family=family,
            stage="raw",
            argv=argv,
        )


@pytest.mark.parametrize(
    "flag",
    ("--enigma-low-tier-base", "--tokenizer-json"),
)
def test_enigma_low_tier_flags_reject_duplicates_and_non_single_values(
    tmp_path_factory,
    flag,
):
    argv, _ = _enigma_low_tier_command(tmp_path_factory)
    index = argv.index(flag)
    value = argv[index + 1]

    duplicate = [*argv, flag, value]
    with pytest.raises(generation.IntegrationError, match="duplicate"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=duplicate,
        )

    missing_value = [*argv[:index], *argv[index + 2 :], flag]
    with pytest.raises(generation.IntegrationError, match="requires a value"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=missing_value,
        )

    extra_value = [*argv[: index + 2], "extra-value", *argv[index + 2 :]]
    with pytest.raises(generation.IntegrationError, match="too many values"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=extra_value,
        )


@pytest.mark.parametrize(
    "missing_flag",
    ("--enigma-low-tier-base", "--tokenizer-json"),
)
def test_enigma_low_tier_command_requires_both_flags(tmp_path_factory, missing_flag):
    argv, _ = _enigma_low_tier_command(tmp_path_factory)
    index = argv.index(missing_flag)
    del argv[index : index + 2]

    with pytest.raises(generation.IntegrationError, match="must appear together"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=argv,
        )


@pytest.mark.parametrize(
    "flag",
    ("--enigma-low-tier-base", "--tokenizer-json"),
)
def test_enigma_low_tier_preflight_requires_readable_paths(tmp_path_factory, flag):
    argv, accepted_base = _enigma_low_tier_command(tmp_path_factory)
    expected = generation._preflight_enigma_low_tier_paths(argv)

    assert expected == {
        "enigma_low_tier_base": accepted_base.resolve(),
        "tokenizer_json": (FIXED_QWEN_TOKENIZER / "tokenizer.json").resolve(),
    }

    index = argv.index(flag)
    argv[index + 1] = str(accepted_base.parent / "missing")
    with pytest.raises(generation.IntegrationError, match="missing or unreadable"):
        generation._preflight_enigma_low_tier_paths(argv)


def test_canonical_metamath_raw_adapter_needs_no_name_flag():
    argv = [
        sys.executable,
        "scripts/build_metamath_shard.py",
        "--mm-dir",
        "/tmp/p3-production-metamath",
        "--heldout",
        "0",
        "--seed",
        "20260801",
        "--tokenizer",
        str(FIXED_QWEN_TOKENIZER),
    ]

    assert generation.validate_production_builder_command(
        family="metamath",
        stage="raw",
        argv=argv,
    ) == argv
    assert "--name" not in argv


def test_canonical_thproofs_raw_adapter_uses_no_exclusion():
    root = Path("/tmp/p3-production-mizar")
    argv = [
        sys.executable,
        "scripts/build_thproofs_shard.py",
        "--src",
        str(root / "thproofs"),
        "--semantic-index",
        str(root / "semantic.sqlite"),
        "--source-manifest",
        str(root / "sources.json"),
        "--mml-root",
        str(root / "mml"),
        "--html-root",
        str(root / "html"),
        "--mizar-archive",
        str(root / "mizar.tar"),
        "--html-archive",
        str(root / "html.tar"),
        "--thproofs-archive",
        str(root / "thproofs.tar"),
        "--name",
        "thproofs",
        "--heldout",
        "0",
        "--seed",
        "20260801",
    ]

    normalized = generation.validate_production_builder_command(
        family="thproofs",
        stage="raw",
        argv=argv,
    )

    assert normalized == [*argv, "--exclude", os.devnull]
    assert "" not in normalized


def test_direct_mizar_production_command_is_registered_and_canonical() -> None:
    root = Path("/tmp/p3-direct-mizar-input")
    argv = [
        sys.executable,
        "scripts/build_mizar_human_shard.py",
        "--mml-root",
        str(root / "mml"),
        "--html-root",
        str(root / "html"),
        "--thproofs-root",
        str(root / "thproofs"),
        "--semantic-index",
        str(root / "semantic.sqlite"),
        "--semantic-index-sha256",
        "a" * 64,
        "--source-manifest",
        str(root / "upstream.json"),
        "--mizar-archive",
        str(root / "mizar.tar"),
        "--html-archive",
        str(root / "html.tar"),
        "--thproofs-archive",
        str(root / "thproofs.tar"),
        "--tokenizer-path",
        str(root / "tokenizer"),
        "--name",
        "mizar",
        "--heldout",
        "0",
        "--seed",
        "20260801",
    ]

    assert generation.validate_production_builder_command(
        family="mizar",
        stage="raw",
        argv=argv,
    ) == argv
    drifted = list(argv)
    drifted[drifted.index("mizar")] = "mizar_human"
    with pytest.raises(generation.IntegrationError, match="name"):
        generation.validate_production_builder_command(
            family="mizar",
            stage="raw",
            argv=drifted,
        )


_POST_GENERATION_STATEMENTS = {
    "EXT:1": "external theorem",
    "TARSKI:def_3": "external definition",
    "ORDINAL1:sch_1": "external scheme",
    "SAMPLE:2": "article-level Lm2",
    "SAMPLE:3": "contextual local at theorem 20",
    "SAMPLE:4": "contextual local at theorem 21",
    "SAMPLE:20": "sample goal 20",
    "SAMPLE:21": "sample goal 21",
}


class _PostGenerationDispatchIndex:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback

    def statement_map(self):
        return dict(_POST_GENERATION_STATEMENTS)

    def article_local_label_maps(self):
        return {
            "SAMPLE": {
                "Lm2": ("SAMPLE:2",),
                "Ctx": ("SAMPLE:3", "SAMPLE:4"),
            }
        }

    def resolve_local_label(self, article, label, *, at_identity):
        if article == "SAMPLE" and label == "Lm2":
            return "SAMPLE:2"
        contextual = {
            ("SAMPLE", "Ctx", "SAMPLE:20"): "SAMPLE:3",
            ("SAMPLE", "Ctx", "SAMPLE:21"): "SAMPLE:4",
        }
        try:
            return contextual[(article, label, at_identity)]
        except KeyError as error:
            raise KeyError(label) from error


def _post_generation_mizar_row():
    cited = ["EXT:1", "SAMPLE:3", "TARSKI:def_3", "ORDINAL1:sch_1"]
    return {
        "theorem": "SAMPLE:20",
        "goal": _POST_GENERATION_STATEMENTS["SAMPLE:20"],
        "facts": {name: _POST_GENERATION_STATEMENTS[name] for name in cited},
        "cited": cited,
        "target": (
            "Lm2: P by EXT:1;\n"
            "thus thesis by Lm2, Ctx, TARSKI:def 3, ORDINAL1:sch 1;"
        ),
    }


def _post_generation_thproof_row():
    return {
        "theorem": "SAMPLE:20",
        "goal": _POST_GENERATION_STATEMENTS["SAMPLE:20"],
        "facts": {
            "EXT:1": _POST_GENERATION_STATEMENTS["EXT:1"],
            "SAMPLE:3": _POST_GENERATION_STATEMENTS["SAMPLE:3"],
        },
        "cited": ["SAMPLE:3", "EXT:1"],
        "target": "thus thesis by EXT:1, Ctx;",
    }


def _write_post_generation_dispatch_case(
    tmp_path,
    monkeypatch,
    *,
    mizar_row=None,
    thproof_row=None,
):
    index_path = tmp_path / "semantic.sqlite"
    index_path.write_bytes(b"minimized current Mizar semantic index")
    monkeypatch.setattr(
        mizar_current_index,
        "MizarIndex",
        lambda path: _PostGenerationDispatchIndex(),
    )
    generation_path = tmp_path / "generation"
    for split in ("shards", "eval"):
        (generation_path / split).mkdir(parents=True, exist_ok=True)
        generation._write_jsonl(
            generation_path / split / "mizar.jsonl",
            [mizar_row or _post_generation_mizar_row()],
        )
        generation._write_jsonl(
            generation_path / split / "thproofs.jsonl",
            [thproof_row or _post_generation_thproof_row()],
        )
    index_root = hashlib.sha256(index_path.read_bytes()).hexdigest()
    source_manifests = {
        family: {
            "row_source_metadata": {
                "index_roots": {"semantic_index_sha256": index_root}
            }
        }
        for family in ("mizar", "thproofs")
    }
    return index_path, source_manifests, generation_path


def test_post_generation_direct_mizar_dispatch_handles_proof_local_definition_scheme_and_context(
    tmp_path,
    monkeypatch,
):
    index_path, source_manifests, generation_path = (
        _write_post_generation_dispatch_case(tmp_path, monkeypatch)
    )

    generation._verify_current_mizar_index(
        index_path,
        source_manifests=source_manifests,
        generation_path=generation_path,
    )


def test_post_generation_thproofs_keeps_native_resolver_and_set_semantics(
    tmp_path,
    monkeypatch,
):
    index_path, source_manifests, generation_path = (
        _write_post_generation_dispatch_case(tmp_path, monkeypatch)
    )
    native_direct_resolver = direct_mizar.resolve_global_citations
    direct_targets = []

    def track_direct_dispatch(body, index, *, theorem):
        direct_targets.append(body)
        return native_direct_resolver(body, index, theorem=theorem)

    monkeypatch.setattr(direct_mizar, "resolve_global_citations", track_direct_dispatch)

    generation._verify_current_mizar_index(
        index_path,
        source_manifests=source_manifests,
        generation_path=generation_path,
    )
    assert direct_targets == [_post_generation_mizar_row()["target"]] * 2


def test_post_generation_rejects_direct_mizar_resolver_swap(tmp_path, monkeypatch):
    index_path, source_manifests, generation_path = (
        _write_post_generation_dispatch_case(tmp_path, monkeypatch)
    )

    def swapped_resolver(body, index, *, theorem):
        references, unresolved = thproofs.resolve_index_references(
            body,
            index,
            index.statement_map(),
            theorem=theorem,
        )
        return direct_mizar.CitationResolution(
            references=tuple(references),
            unresolved=tuple(unresolved),
            proof_local_labels=(),
        )

    monkeypatch.setattr(direct_mizar, "resolve_global_citations", swapped_resolver)
    with pytest.raises(generation.IntegrationError, match="target/reference mismatch"):
        generation._verify_current_mizar_index(
            index_path,
            source_manifests=source_manifests,
            generation_path=generation_path,
        )


@pytest.mark.parametrize("mutation", ("unresolved-global", "wrong-theorem-context"))
def test_post_generation_rejects_invalid_direct_mizar_reference_semantics(
    tmp_path,
    monkeypatch,
    mutation,
):
    row = _post_generation_mizar_row()
    if mutation == "unresolved-global":
        row["target"] += "\nthus thesis by MISSING:1;"
    else:
        row["theorem"] = "SAMPLE:21"
        row["goal"] = _POST_GENERATION_STATEMENTS["SAMPLE:21"]
    index_path, source_manifests, generation_path = (
        _write_post_generation_dispatch_case(
            tmp_path,
            monkeypatch,
            mizar_row=row,
        )
    )

    with pytest.raises(generation.IntegrationError, match="target/reference mismatch"):
        generation._verify_current_mizar_index(
            index_path,
            source_manifests=source_manifests,
            generation_path=generation_path,
        )


@pytest.mark.parametrize("mutation", ("source-root", "indexed-statement"))
def test_post_generation_rejects_current_mizar_source_index_drift(
    tmp_path,
    monkeypatch,
    mutation,
):
    row = _post_generation_mizar_row()
    if mutation == "indexed-statement":
        row["facts"]["SAMPLE:3"] = "drifted contextual statement"
    index_path, source_manifests, generation_path = (
        _write_post_generation_dispatch_case(
            tmp_path,
            monkeypatch,
            mizar_row=row,
        )
    )
    if mutation == "source-root":
        source_manifests["mizar"]["row_source_metadata"]["index_roots"][
            "semantic_index_sha256"
        ] = "0" * 64

    with pytest.raises(
        generation.IntegrationError,
        match="index root is stale|fact/index mismatch",
    ):
        generation._verify_current_mizar_index(
            index_path,
            source_manifests=source_manifests,
            generation_path=generation_path,
        )


def test_production_config_cannot_encode_callback_or_test_seam():
    manifest = generation._make_source_manifest("isabelle", test_only=False)
    manifest["builder"] = {
        "driver": "external-command-v2",
        "partition_mode": "family-local-heldout-v2",
        "raw": {"argv": [], "outputs": {}, "inventory": []},
        "split": {"argv": [], "outputs": {}, "inventory": []},
        "callback": "unit-test-only",
    }

    with pytest.raises(generation.IntegrationError, match="exact|callback"):
        generation._validate_production_builder_config(
            manifest,
            family="isabelle",
        )


def test_dry_run_rejects_nonexistent_inputs_before_reporting_blockers(
    tmp_path,
    monkeypatch,
    capsys,
):
    work = tmp_path / "work"
    work.mkdir()
    missing = tmp_path / "missing.json"
    argv = [
        "build_p3_generation.py",
        "--dry-run",
        "--corpus-root",
        str(tmp_path / "corpus"),
        "--work-root",
        str(work),
        "--generation-id",
        "preflight",
        "--tokenizer-seal",
        str(missing),
        "--metamath-drop-ledger",
        str(missing),
        "--tokenizer-path",
        str(tmp_path / "missing-tokenizer"),
        "--policies",
        str(missing),
        "--mizar-semantic-index",
        str(tmp_path / "missing-index"),
    ]
    for family in FAMILIES:
        argv.extend(["--source-manifest", f"{family}={missing}"])
    monkeypatch.setattr(sys, "argv", argv)

    assert generation.main() == 2
    error = capsys.readouterr().err
    assert "not valid JSON" in error or "missing" in error
    assert "real full build blocked" not in error


def test_preflight_accepts_exact_fixed_qwen_identity_without_local_path(
    prepared_tokenizer_seal,
):
    seal = generation._validate_tokenizer_seal(prepared_tokenizer_seal)
    backend = isabelle_builder.load_vendored_tokenizer(FIXED_QWEN_TOKENIZER)
    config = json.loads(
        (FIXED_QWEN_TOKENIZER / "tokenizer_config.json").read_text(encoding="utf-8")
    )

    assert backend.backend.token_to_id(config["pad_token"]) == seal["eos_token_id"] == 151_643
    assert backend.encode("---\nGOAL", add_special_tokens=False).ids == [10_952, 15_513, 969]
    assert generation._load_actual_tokenizer_seal(FIXED_QWEN_TOKENIZER) == seal


@pytest.mark.parametrize(
    "field",
    [
        "identity",
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "behavior_digest",
        "tokenizers_version",
        "eos_token_id",
        "max_text_plus_eos_tokens",
    ],
)
def test_preflight_rejects_every_one_field_tokenizer_identity_mutation(
    prepared_tokenizer_seal,
    field,
):
    seal = prepared_tokenizer_seal
    value = seal[field]
    seal[field] = value + 1 if isinstance(value, int) else f"{value}-mutated"

    with pytest.raises(generation.IntegrationError, match="approved exact Qwen seal"):
        generation._validate_tokenizer_seal(seal)


def test_preflight_accepts_relocated_exact_tokenizer_bytes_but_rejects_substitution(
    tmp_path,
    prepared_tokenizer_seal,
):
    relocated = tmp_path / "qwen25-vendored"
    relocated.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (relocated / name).write_bytes((FIXED_QWEN_TOKENIZER / name).read_bytes())

    assert generation._load_actual_tokenizer_seal(relocated) == prepared_tokenizer_seal

    tokenizer_json = relocated / "tokenizer.json"
    payload = tokenizer_json.read_bytes()
    tokenizer_json.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(generation.IntegrationError, match="tokenizer.*SHA-256|sealed tokenizer"):
        generation._load_actual_tokenizer_seal(relocated)


def test_preflight_rejects_runtime_tokenizer_path_substitution(tmp_path, monkeypatch):
    real_metadata = isabelle_builder._tokenizer_metadata

    def substituted_path_metadata(tokenizer):
        metadata = real_metadata(tokenizer)
        metadata["path"] = str(tmp_path / "substituted" / "tokenizer.json")
        return metadata

    monkeypatch.setattr(
        isabelle_builder,
        "_tokenizer_metadata",
        substituted_path_metadata,
    )

    with pytest.raises(generation.IntegrationError, match="tokenizer.*path"):
        generation._load_actual_tokenizer_seal(FIXED_QWEN_TOKENIZER)


def test_production_metamath_preflight_binds_verified_16k_candidate_ledger(
    tmp_path,
    monkeypatch,
    repository_tokenizer_seal,
):
    ledger_path = METAMATH_16K_CANDIDATE / "drops" / "metamath-overlength.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    native_manifest = json.loads(
        (METAMATH_16K_CANDIDATE / "metamath_sources.json").read_text(
            encoding="utf-8"
        )
    )
    conflict_map = {"fixture-conflict": {"set": "|- ph", "iset": "|- ps"}}
    expected_metadata = metamath_builder.build_source_metadata(
        native_manifest,
        conflict_map,
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    for database in ("set", "iset", "nf"):
        (mm_dir / f"{database}.mm").write_text(f"{database}\n")
    manifest = {
        "builder": {
            "raw": {
                "argv": [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "build_metamath_shard.py"),
                    "--mm-dir",
                    str(mm_dir),
                    "--heldout",
                    "0",
                    "--seed",
                    "20260801",
                    "--tokenizer",
                    str(FIXED_QWEN_TOKENIZER),
                ]
            }
        },
        "row_source_metadata": expected_metadata,
    }
    monkeypatch.setattr(
        metamath_builder,
        "source_manifest",
        lambda _mm_dir: native_manifest,
    )
    monkeypatch.setattr(
        metamath_builder,
        "load_pinned_databases",
        lambda _mm_dir: (None, None, conflict_map),
    )
    paths = {}
    roots = {}

    generation._validate_builder_native_source_metadata(
        manifest,
        family="metamath",
        validated_paths=paths,
        validated_roots=roots,
        tokenizer_metadata=repository_tokenizer_seal,
        tokenizer_path=FIXED_QWEN_TOKENIZER,
        metamath_drop_ledger=ledger,
    )

    assert ledger["accounting"] == {
        "source_rows": 67_034,
        "eligible_rows": 66_074,
        "dropped_rows": 960,
        "source_text_plus_eos_tokens": 141_077_915,
        "eligible_text_plus_eos_tokens": 113_086_656,
        "dropped_text_plus_eos_tokens": 27_991_259,
        "dropped_excess_tokens": 12_262_619,
    }
    assert ledger["canonical_root_sha256"] in roots.values()
    assert ledger["entries_root_sha256"] in roots.values()
    assert ledger["tokenizer_root_sha256"] in roots.values()
    assert expected_metadata["drop_ledger"]["accounting"]["source_rows"] == 67_034


def test_production_metamath_preflight_has_no_aggregate_only_metadata_fallback(
    tmp_path,
    monkeypatch,
    repository_tokenizer_seal,
):
    ledger = json.loads(
        (
            METAMATH_16K_CANDIDATE
            / "drops"
            / "metamath-overlength.json"
        ).read_text(encoding="utf-8")
    )
    native_manifest = json.loads(
        (METAMATH_16K_CANDIDATE / "metamath_sources.json").read_text(
            encoding="utf-8"
        )
    )
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    for database in ("set", "iset", "nf"):
        (mm_dir / f"{database}.mm").write_text(f"{database}\n")
    aggregate_only = generation._make_source_manifest(
        "metamath",
        test_only=False,
    )["row_source_metadata"]
    manifest = {
        "builder": {
            "raw": {
                "argv": [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "build_metamath_shard.py"),
                    "--mm-dir",
                    str(mm_dir),
                    "--heldout",
                    "0",
                    "--seed",
                    "20260801",
                    "--tokenizer",
                    str(FIXED_QWEN_TOKENIZER),
                ]
            }
        },
        "row_source_metadata": aggregate_only,
    }
    monkeypatch.setattr(
        metamath_builder,
        "source_manifest",
        lambda _mm_dir: native_manifest,
    )
    monkeypatch.setattr(
        metamath_builder,
        "load_pinned_databases",
        lambda _mm_dir: (None, None, {}),
    )

    with pytest.raises(generation.IntegrationError, match="drop ledger.*required"):
        generation._validate_builder_native_source_metadata(
            manifest,
            family="metamath",
            validated_paths={},
            validated_roots={},
            tokenizer_metadata=repository_tokenizer_seal,
            tokenizer_path=FIXED_QWEN_TOKENIZER,
        )
    with pytest.raises(generation.IntegrationError, match="source metadata|real inputs"):
        generation._validate_builder_native_source_metadata(
            manifest,
            family="metamath",
            validated_paths={},
            validated_roots={},
            tokenizer_metadata=repository_tokenizer_seal,
            tokenizer_path=FIXED_QWEN_TOKENIZER,
            metamath_drop_ledger=ledger,
        )


def test_production_metamath_preflight_rejects_split_tokenizer_path_substitution(
    tmp_path,
    monkeypatch,
    repository_tokenizer_seal,
):
    ledger = json.loads(
        (
            METAMATH_16K_CANDIDATE
            / "drops"
            / "metamath-overlength.json"
        ).read_text(encoding="utf-8")
    )
    native_manifest = json.loads(
        (METAMATH_16K_CANDIDATE / "metamath_sources.json").read_text(
            encoding="utf-8"
        )
    )
    conflict_map = {}
    metadata = metamath_builder.build_source_metadata(
        native_manifest,
        conflict_map,
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    for database in ("set", "iset", "nf"):
        (mm_dir / f"{database}.mm").write_text(f"{database}\n")
    substituted = tmp_path / "substituted-tokenizer"
    substituted.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (substituted / name).write_bytes((FIXED_QWEN_TOKENIZER / name).read_bytes())

    def command(heldout, tokenizer):
        return {
            "argv": [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "build_metamath_shard.py"),
                "--mm-dir",
                str(mm_dir),
                "--heldout",
                heldout,
                "--seed",
                "20260801",
                "--tokenizer",
                str(tokenizer),
            ]
        }

    manifest = {
        "builder": {
            "raw": command("0", FIXED_QWEN_TOKENIZER),
            "split": command("500", substituted),
        },
        "row_source_metadata": metadata,
    }
    monkeypatch.setattr(
        metamath_builder,
        "source_manifest",
        lambda _mm_dir: native_manifest,
    )
    monkeypatch.setattr(
        metamath_builder,
        "load_pinned_databases",
        lambda _mm_dir: (None, None, conflict_map),
    )

    with pytest.raises(generation.IntegrationError, match="tokenizer path.*substitut"):
        generation._validate_builder_native_source_metadata(
            manifest,
            family="metamath",
            validated_paths={},
            validated_roots={},
            tokenizer_metadata=repository_tokenizer_seal,
            tokenizer_path=FIXED_QWEN_TOKENIZER,
            metamath_drop_ledger=ledger,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "id",
        "theorem",
        "native-sha",
        "token-length",
    ),
)
def test_production_metamath_preflight_rejects_resigned_candidate_ledger_mutation(
    tmp_path,
    monkeypatch,
    repository_tokenizer_seal,
    mutation,
):
    original = json.loads(
        (
            METAMATH_16K_CANDIDATE
            / "drops"
            / "metamath-overlength.json"
        ).read_text(encoding="utf-8")
    )
    native_manifest = json.loads(
        (METAMATH_16K_CANDIDATE / "metamath_sources.json").read_text(
            encoding="utf-8"
        )
    )
    conflict_map = {"fixture-conflict": {"set": "|- ph", "iset": "|- ps"}}
    expected_metadata = metamath_builder.build_source_metadata(
        native_manifest,
        conflict_map,
        drop_ledger=original,
        tokenizer_seal=repository_tokenizer_seal,
    )
    mutated = json.loads(json.dumps(original))
    if mutation == "missing":
        mutated["entries"].pop()
    elif mutation == "extra":
        mutated["entries"].append(
            _metamath_drop_entry("ffffffffffff", "set:extra-overlength")
        )
        mutated["entries"].sort(key=lambda entry: (entry["id"], entry["theorem"]))
    elif mutation == "id":
        mutated["entries"][0]["id"] = "000000000000"
        mutated["entries"].sort(key=lambda entry: (entry["id"], entry["theorem"]))
    elif mutation == "theorem":
        mutated["entries"][0]["theorem"] += "-mutated"
        mutated["entries"].sort(key=lambda entry: (entry["id"], entry["theorem"]))
    elif mutation == "native-sha":
        mutated["entries"][0]["native_row_sha256"] = "f" * 64
    else:
        mutated["entries"][0]["text_plus_eos_tokens"] += 1
    _resign_metamath_drop_ledger(mutated)
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    for database in ("set", "iset", "nf"):
        (mm_dir / f"{database}.mm").write_text(f"{database}\n")
    manifest = {
        "builder": {
            "raw": {
                "argv": [
                    sys.executable,
                    str(REPOSITORY_ROOT / "scripts" / "build_metamath_shard.py"),
                    "--mm-dir",
                    str(mm_dir),
                    "--heldout",
                    "0",
                    "--seed",
                    "20260801",
                    "--tokenizer",
                    str(FIXED_QWEN_TOKENIZER),
                ]
            }
        },
        "row_source_metadata": expected_metadata,
    }
    monkeypatch.setattr(
        metamath_builder,
        "source_manifest",
        lambda _mm_dir: native_manifest,
    )
    monkeypatch.setattr(
        metamath_builder,
        "load_pinned_databases",
        lambda _mm_dir: (None, None, conflict_map),
    )

    with pytest.raises(
        generation.IntegrationError,
        match="overlength|source metadata|real inputs",
    ):
        generation._validate_builder_native_source_metadata(
            manifest,
            family="metamath",
            validated_paths={},
            validated_roots={},
            tokenizer_metadata=repository_tokenizer_seal,
            tokenizer_path=FIXED_QWEN_TOKENIZER,
            metamath_drop_ledger=mutated,
        )


@pytest.mark.parametrize("mutation", ("duplicate", "ordering", "stale-root", "tokenizer"))
def test_metamath_ledger_authoritative_api_rejects_intrinsic_mutation(
    repository_tokenizer_seal,
    mutation,
):
    ledger = _make_metamath_drop_ledger(
        [
            _metamath_drop_entry("drop-a", "set:drop-a"),
            _metamath_drop_entry("drop-b", "set:drop-b"),
        ],
        eligible_rows=3,
    )
    if mutation == "duplicate":
        ledger["entries"].append(dict(ledger["entries"][0]))
        _resign_metamath_drop_ledger(ledger)
    elif mutation == "ordering":
        ledger["entries"].reverse()
        _resign_metamath_drop_ledger(ledger)
    elif mutation == "stale-root":
        ledger["entries"][0]["native_row_sha256"] = "f" * 64
    else:
        ledger["tokenizer_seal"]["identity"] = "substituted/tokenizer"
        _resign_metamath_drop_ledger(ledger)

    with pytest.raises(generation.IntegrationError, match="drop ledger|ledger"):
        generation._validate_metamath_drop_ledger(
            ledger,
            tokenizer_seal=repository_tokenizer_seal,
        )


def test_metamath_preflight_ledger_file_rejects_missing_and_substituted_bytes(
    tmp_path,
    repository_tokenizer_seal,
):
    supplied = _make_metamath_drop_ledger(
        [_metamath_drop_entry()],
        eligible_rows=3,
    )
    path = tmp_path / "metamath-overlength.json"
    path.write_text(json.dumps(supplied) + "\n")
    resolved, validated, digest = (
        generation._validate_metamath_drop_ledger_file(
            path,
            supplied=supplied,
            tokenizer_seal=repository_tokenizer_seal,
        )
    )
    assert resolved == path.resolve()
    assert validated == supplied
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    substituted = _make_metamath_drop_ledger(
        [_metamath_drop_entry("different", "set:different")],
        eligible_rows=3,
    )
    path.write_text(json.dumps(substituted) + "\n")
    with pytest.raises(generation.IntegrationError, match="substitut|changed"):
        generation._validate_metamath_drop_ledger_file(
            path,
            supplied=supplied,
            tokenizer_seal=repository_tokenizer_seal,
        )
    with pytest.raises(generation.IntegrationError, match="missing|unreadable"):
        generation._validate_metamath_drop_ledger_file(
            tmp_path / "missing.json",
            supplied=supplied,
            tokenizer_seal=repository_tokenizer_seal,
        )


def test_metamath_source_accounting_rejects_ledger_row_also_emitted(
    tmp_path,
    repository_tokenizer_seal,
):
    entry = _metamath_drop_entry("also-emitted", "set:also-emitted")
    ledger = _make_metamath_drop_ledger([entry], eligible_rows=1)
    native_manifest = {
        "repository": "synthetic://metamath",
        "commit": "fixture",
        "files": {"set.mm": {"sha256": hashlib.sha256(b"set").hexdigest()}},
    }
    metadata = metamath_builder.build_source_metadata(
        native_manifest,
        {},
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )
    row = generation._metamath_record(
        "also-emitted",
        metadata,
        theorem="set:also-emitted",
    )
    raw_path = tmp_path / "raw.jsonl"
    generation._write_jsonl(raw_path, [row])

    with pytest.raises(generation.IntegrationError, match="also emitted"):
        generation._metamath_source_occurrence_binding(
            generation._read_occurrences(raw_path, label="overlap/raw"),
            drop_ledger=ledger,
            tokenizer_seal=repository_tokenizer_seal,
        )


def test_metamath_normalizer_accounts_for_eligible_routes_and_native_overlength(
    tmp_path,
    repository_tokenizer_seal,
):
    ledger = _make_metamath_drop_ledger(
        [_metamath_drop_entry()],
        eligible_rows=3,
    )
    native_manifest = {
        "repository": "synthetic://metamath",
        "commit": "fixture",
        "files": {"set.mm": {"sha256": hashlib.sha256(b"set").hexdigest()}},
    }
    source_metadata = metamath_builder.build_source_metadata(
        native_manifest,
        {},
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )
    source_manifest = generation._make_source_manifest("metamath", test_only=False)
    source_manifest["row_source_metadata"] = source_metadata
    source_manifest["source_snapshots"] = [native_manifest]
    source_manifest["manifest_root_sha256"] = generation._source_manifest_root(
        source_manifest
    )
    rows = [
        generation._metamath_record(
            "train",
            source_metadata,
            fact_name="safe",
            fact_statement="|- ps => |- ps",
        ),
        generation._metamath_record("eval", source_metadata),
        generation._metamath_record(
            "held-own-proof",
            source_metadata,
            fact_name="safe",
            fact_statement="|- ps => |- ps",
            theorem="set:mp",
        ),
    ]
    raw = tmp_path / "raw.jsonl"
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    heldout = tmp_path / "heldout.json"
    generation._write_jsonl(raw, rows)
    generation._write_jsonl(train, rows[:1])
    generation._write_jsonl(evaluation, rows[1:])
    heldout_payload = {
        "schema_version": "metamath-heldout-v2",
        "family": "metamath",
        "mode": "family_local_heldout",
        "facts": ["mp", *(f"unused-{index}" for index in range(499))],
        "requested_heldout": 500,
        "actual_heldout": 500,
        "local_assumptions": True,
        "eligibility": {
            "schema_version": "metamath-text-plus-eos-eligibility-v1",
            "max_text_plus_eos_tokens": 16_384,
            "tokenizer_seal": repository_tokenizer_seal,
            "tokenizer_root_sha256": ledger["tokenizer_root_sha256"],
            "drop_ledger": {
                "path": "drops/metamath-overlength.json",
                "schema_version": ledger["schema_version"],
                "canonical_root_sha256": ledger["canonical_root_sha256"],
                "entries_root_sha256": ledger["entries_root_sha256"],
            },
            "accounting": ledger["accounting"],
        },
        "partition_accounting": {
            "source_rows": 4,
            "source_text_plus_eos_tokens": ledger["accounting"][
                "source_text_plus_eos_tokens"
            ],
            "train_rows": 1,
            "train_text_plus_eos_tokens": 40,
            "eval_rows": 2,
            "eval_text_plus_eos_tokens": 60,
            "drop_rows": 1,
            "drop_text_plus_eos_tokens": ledger["accounting"][
                "dropped_text_plus_eos_tokens"
            ],
        },
    }
    heldout_payload["manifest_root_sha256"] = metamath_builder.canonical_sha256(
        heldout_payload
    )
    heldout.write_text(json.dumps(heldout_payload) + "\n")

    package = generation._normalize_metamath_package(
        raw_output=raw,
        split_outputs={
            "train": train,
            "eval": evaluation,
            "heldout": heldout,
        },
        destination=tmp_path / "normalized",
        source_manifest=source_manifest,
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )

    assert package.source_accounting["source_rows"] == 4
    assert package.source_accounting["train_rows"] == 1
    assert package.source_accounting["eval_rows"] == 1
    assert package.source_accounting["drop_types"] == {
        "heldout_own_proof": 1,
        "overlength": 1,
    }
    assert package.source_accounting["accounted_rows"] == 4
    assert package.overlength_drop_ledger == ledger
    generation._validate_metamath_source_accounting(
        package.source_accounting,
        raw=generation._read_occurrences(raw, label="fixture/raw"),
        train=generation._read_occurrences(package.train, label="fixture/train"),
        evaluation=generation._read_occurrences(
            package.eval,
            label="fixture/eval",
        ),
        drops=package.drops,
        drop_ledger=ledger,
        tokenizer_seal=repository_tokenizer_seal,
    )
    mutated = json.loads(json.dumps(package.source_accounting))
    mutated["native_occurrences_root_sha256"] = "0" * 64
    with pytest.raises(generation.IntegrationError, match="source accounting|occurrence"):
        generation._validate_metamath_source_accounting(
            mutated,
            raw=generation._read_occurrences(raw, label="fixture/raw"),
            train=generation._read_occurrences(package.train, label="fixture/train"),
            evaluation=generation._read_occurrences(
                package.eval,
                label="fixture/eval",
            ),
            drops=package.drops,
            drop_ledger=ledger,
            tokenizer_seal=repository_tokenizer_seal,
        )
    old_artifact_substitution = json.loads(
        json.dumps(package.source_accounting)
    )
    old_artifact_substitution.update(
        {
            "source_rows": 67_034,
            "eligible_rows": 67_034,
            "overlength_rows": 0,
            "train_rows": 66_074,
            # The old builder's 960 eval rows decomposed into 497 final eval
            # plus 463 heldout-own-proof drops.
            "eval_rows": 497,
            "drop_types": {"heldout_own_proof": 463, "overlength": 0},
            "accounted_rows": 67_034,
        }
    )
    old_body = dict(old_artifact_substitution)
    old_body.pop("accounting_root_sha256")
    old_artifact_substitution["accounting_root_sha256"] = (
        generation._canonical_sha256(old_body)
    )
    with pytest.raises(generation.IntegrationError, match="source accounting|route"):
        generation._validate_metamath_source_accounting(
            old_artifact_substitution,
            raw=generation._read_occurrences(raw, label="fixture/raw"),
            train=generation._read_occurrences(package.train, label="fixture/train"),
            evaluation=generation._read_occurrences(
                package.eval,
                label="fixture/eval",
            ),
            drops=package.drops,
            drop_ledger=ledger,
            tokenizer_seal=repository_tokenizer_seal,
        )


def test_signed_preflight_report_binds_paths_roots_and_commands():
    report = generation.build_preflight_report(
        validated_paths={
            "tokenizer": "/tmp/tokenizer",
            "mizar_index": "/tmp/mizar.sqlite",
        },
        validated_roots={
            "tokenizer": "a" * 64,
            "mizar_index": "b" * 64,
        },
        validated_commands={
            "isabelle/raw": ["python", "scripts/build_isabelle_shard.py"],
        },
        blockers=["MML roots unfinished"],
    )
    body = dict(report)
    signature = body.pop("preflight_root_sha256")

    assert report["status"] == "blocked"
    assert report["schema_version"] == "p3-generation-preflight/v2"
    assert signature == generation._canonical_sha256(body)
    assert report["validated_paths"]["tokenizer"] == "/tmp/tokenizer"
    assert report["validated_commands"]["isabelle/raw"][1].endswith(
        "build_isabelle_shard.py"
    )


def test_preflight_recomputes_every_local_source_snapshot_hash(tmp_path):
    source = tmp_path / "source.tar"
    source.write_bytes(b"actual source archive")
    manifest = generation._make_source_manifest("prf2", test_only=False)
    manifest["source_snapshots"] = [
        {
            "reference": str(source),
            "sha256": "0" * 64,
        }
    ]

    with pytest.raises(generation.IntegrationError, match="snapshot hash"):
        generation._validate_source_snapshot_paths(
            manifest,
            family="prf2",
            validated_paths={},
            validated_roots={},
        )


def test_uri_snapshots_are_verified_against_local_builder_archives(tmp_path):
    archive = tmp_path / "html.tar"
    archive.write_bytes(b"verified source archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    reference = "https://sources.invalid/html.tar"
    manifest = {
        "builder": {
            "raw": {
                "argv": [
                    sys.executable,
                    "scripts/build_mizar_human_shard.py",
                    "--html-archive",
                    str(archive),
                ]
            }
        },
        "row_source_metadata": {
            "source_roots": {
                "html": {
                    "archive_sha256": digest,
                    "file_count": 1,
                    "reference": reference,
                    "tree_sha256": "a" * 64,
                }
            }
        },
        "source_snapshots": [{"reference": reference, "sha256": digest}],
    }
    paths = {}
    roots = {}

    generation._validate_source_snapshot_paths(
        manifest,
        family="mizar",
        validated_paths=paths,
        validated_roots=roots,
    )

    assert paths["source_snapshot/mizar/0"] == str(archive.resolve())
    assert roots["source_snapshot/mizar/0"] == digest


@pytest.mark.parametrize(
    ("family", "snapshot"),
    [
        (
            "metamath",
            {
                "commit": "82830c78861b96e906d9868c30c35dbd98be5db5",
                "files": {
                    "set.mm": {
                        "sha256": hashlib.sha256(b"set.mm snapshot").hexdigest()
                    }
                },
                "repository": "https://github.com/metamath/set.mm",
            },
        ),
        (
            "prf2",
            {
                "files": 25_060,
                "source": "prf2",
                "tree_sha256": hashlib.sha256(b"prf2 source tree").hexdigest(),
            },
        ),
        (
            "isabelle",
            {
                "dataset": "Simontwice/nlproofs",
                "file": "raw_data/human_data/all_data.json",
                "revision": "f947ccc827ccd236464e19cd4cc23dfda7fc5575",
                "sha256": hashlib.sha256(b"Isabelle source").hexdigest(),
                "size_bytes": 2_327_313_460,
            },
        ),
    ],
)
def test_source_manifest_accepts_builder_native_snapshot_records(family, snapshot):
    manifest = generation._make_source_manifest(family, test_only=False)
    manifest["source_snapshots"] = [snapshot]
    manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)

    assert generation._validate_source_manifest(
        manifest,
        family=family,
        production=True,
    )["source_snapshots"] == [snapshot]


def test_preflight_recomputes_builder_native_atp_tree_metadata(tmp_path):
    source = tmp_path / "prf2"
    source.mkdir()
    proof = source / "proof.tstp"
    proof.write_bytes(b"first source proof")
    args = SimpleNamespace(dedup=False, fenced=False, jaccard=0.5, min_steps=4)
    source_path = str(source)
    proof_path = str(proof)
    metadata = atp_builder._build_source_metadata(
        [source_path],
        [proof_path],
        {proof_path: source_path},
        args,
    )
    manifest = generation._make_source_manifest("prf2", test_only=False)
    manifest["row_source_metadata"] = metadata
    manifest["source_snapshots"] = [
        {
            "files": metadata["source_roots"]["source_1"]["files"],
            "reference": source_path,
            "tree_sha256": metadata["source_roots"]["source_1"]["tree_sha256"],
        }
    ]
    manifest["builder"] = {
        "driver": "external-command-v2",
        "partition_mode": "pooled-mml-1000-v1",
        "raw": {
            "argv": [
                sys.executable,
                "scripts/build_atp_shard.py",
                "--src",
                source_path,
                "--name",
                "prf2",
                "--heldout",
                "0",
                "--min-steps",
                "4",
                "--seed",
                "20260801",
            ],
            "inventory": [],
            "outputs": {},
        },
    }
    manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)

    generation._validate_builder_native_source_metadata(
        manifest,
        family="prf2",
        validated_paths={},
        validated_roots={},
    )

    proof.write_bytes(b"drifted source proof")
    with pytest.raises(generation.IntegrationError, match="source metadata|tree"):
        generation._validate_builder_native_source_metadata(
            manifest,
            family="prf2",
            validated_paths={},
            validated_roots={},
        )


def test_row_build_source_root_may_differ_from_family_manifest_root():
    source_root = hashlib.sha256(b"builder-native-source-root").hexdigest()
    manifest = generation._make_source_manifest("prf2", test_only=False)
    manifest["row_source_metadata"]["source_manifest_root_sha256"] = source_root
    manifest["builder"] = {
        "driver": "external-command-v2",
        "partition_mode": "pooled-mml-1000-v1",
        "raw": {
            "argv": [
                sys.executable,
                "scripts/build_atp_shard.py",
                "--src",
                "/tmp/p3-production-prf2",
                "--name",
                "prf2",
                "--heldout",
                "0",
                "--min-steps",
                "4",
                "--seed",
                "20260801",
            ],
            "inventory": [
                {
                    "path": "shards/prf2.jsonl",
                    "kind": "file",
                    "format": "jsonl",
                    "schema": generation.ROW_SCHEMAS["prf2"],
                    "source_manifest_root_sha256": source_root,
                },
                {"path": "shards", "kind": "directory"},
            ],
            "outputs": {"raw": "shards/prf2.jsonl"},
        },
    }
    manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)

    assert manifest["manifest_root_sha256"] != source_root
    generation._validate_production_builder_config(manifest, family="prf2")


def test_token_data_preflight_keeps_unapproved_license_as_external_status():
    manifest = generation._make_source_manifest("mizar", test_only=False)

    validated = generation._validate_source_manifest(
        manifest,
        family="mizar",
        production=True,
    )

    assert validated["license"]["approved"] is False
    assert "not redistributable" in validated["license"]["status"]


def test_token_data_preflight_does_not_require_metamath_validity_acceptance():
    manifest = generation._make_source_manifest("metamath", test_only=False)

    validated = generation._validate_source_manifest(
        manifest,
        family="metamath",
        production=True,
    )

    assert validated["source_verifier_acceptance"]["accepted"] is False


def test_production_blockers_clear_when_supplied_inputs_satisfy_contracts(monkeypatch):
    monkeypatch.setattr(
        generation.mml_holdout,
        "production_source_policy",
        lambda: object(),
    )
    manifests = {}
    for family in FAMILIES:
        manifest = generation._make_source_manifest(family, test_only=False)
        manifest["builder"] = {"driver": "external-command-v2"}
        manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)
        manifests[family] = manifest

    assert generation.production_blockers(manifests) == []


def test_raw_builder_modes_emit_one_nonempty_raw_output(tmp_path):
    row, source_manifest = generation.synthetic_family_record("prf2")
    source_root = source_manifest["row_source_metadata"][
        "source_manifest_root_sha256"
    ]

    def callback(root):
        generation._write_jsonl(root / "raw.jsonl", [row])
        (root / "eval.jsonl").write_bytes(b"")

    outputs = generation._run_builder_callback_stage(
        family="prf2",
        stage="raw",
        output_root=tmp_path / "raw-build",
        inventory=[
            {
                "path": "raw.jsonl",
                "kind": "file",
                "format": "jsonl",
                "schema": generation.ROW_SCHEMAS["prf2"],
                "source_manifest_root_sha256": source_root,
            },
            {
                "path": "eval.jsonl",
                "kind": "file",
                "format": "jsonl",
                "schema": generation.ROW_SCHEMAS["prf2"],
                "source_manifest_root_sha256": source_root,
                "allow_empty": True,
            },
        ],
        callback=callback,
    )

    assert outputs["raw.jsonl"].stat().st_size > 0
    assert outputs["eval.jsonl"].read_bytes() == b""


def _mutable_generation_file(result, relative):
    path = result.published.path / relative
    path.chmod(0o644)
    return path


@pytest.fixture(scope="module")
def structured_generation_path(tmp_path_factory):
    root = tmp_path_factory.mktemp("structured-generation") / "transaction"
    work = tmp_path_factory.mktemp("structured-work")
    return generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="structured-json-contracts",
    ).published.path


@pytest.mark.parametrize(
    ("relative", "expected_schema", "required_fields"),
    JSON_OUTPUT_CASES,
)
@pytest.mark.parametrize(
    "mutation",
    ("missing-schema", "wrong-schema", "missing-required"),
)
def test_every_declared_json_role_rejects_internal_contract_mutation(
    structured_generation_path,
    relative,
    expected_schema,
    required_fields,
    mutation,
):
    path = structured_generation_path / relative
    original = path.read_bytes()
    path.chmod(0o644)
    payload = json.loads(original)
    if mutation == "missing-schema":
        payload.pop("schema_version")
    elif mutation == "wrong-schema":
        payload["schema_version"] = "wrong-linked-v9"
    else:
        payload.pop(required_fields[0])
    path.write_text(json.dumps(payload) + "\n")
    try:
        with pytest.raises(generation.IntegrationError, match="schema|required"):
            generation._validate_declared_json_outputs(structured_generation_path)
    finally:
        path.write_bytes(original)
        path.chmod(0o444)


@pytest.mark.parametrize(
    "mutation",
    ("missing-schema", "wrong-schema", "missing-required"),
)
def test_production_verifier_rejects_hostile_nested_metamath_heldout_contract(
    structured_generation_path,
    mutation,
):
    path = structured_generation_path / "heldout/metamath.json"
    original = path.read_bytes()
    path.chmod(0o644)
    payload = json.loads(original)
    contract = payload["contract"]
    if mutation == "missing-schema":
        contract.pop("schema_version")
    elif mutation == "wrong-schema":
        contract["schema_version"] = "wrong-heldout-v9"
    else:
        contract.pop("local_assumptions")
    path.write_text(json.dumps(payload) + "\n")
    try:
        with pytest.raises(
            generation.IntegrationError,
            match="Metamath.*heldout|schema|required",
        ):
            generation._independent_verify_generation_path(
                structured_generation_path,
                production=False,
            )
    finally:
        path.write_bytes(original)
        path.chmod(0o444)


def test_independent_verifier_recomputes_schema_and_precheck_sidecars(tmp_path):
    result = generation.build_synthetic_generation(
        corpus_root=tmp_path / "transaction",
        work_root=tmp_path,
        generation_id="independent-sidecars",
    )
    schema_path = _mutable_generation_file(result, "sidecars/schemas.json")
    schema = json.loads(schema_path.read_text())
    schema["row_schemas"]["metamath"] = "wrong-v9"
    schema_path.write_text(json.dumps(schema) + "\n")

    with pytest.raises(generation.IntegrationError, match="schema sidecar"):
        generation._independent_verify_generation_path(
            result.published.path,
            production=False,
        )

    schema_path.write_bytes(
        json.dumps(
            {
                **schema,
                "row_schemas": generation.ROW_SCHEMAS,
            }
        ).encode()
        + b"\n"
    )
    precheck_path = _mutable_generation_file(result, "sidecars/precheck.json")
    precheck = json.loads(precheck_path.read_text())
    precheck["counts"]["metamath"]["train"] += 1
    precheck_path.write_text(json.dumps(precheck) + "\n")
    with pytest.raises(generation.IntegrationError, match="precheck sidecar"):
        generation._independent_verify_generation_path(
            result.published.path,
            production=False,
        )


def _rerender_metamath_record(row):
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in row["facts"].items()
    )
    block += "\nLocal assumptions:"
    if row["local_assumptions"]:
        block += "\n" + "\n".join(
            f"{name} : {statement}"
            for name, statement in row["local_assumptions"].items()
        )
    row["text"] = f"{block}\n---\nGOAL {row['goal']}\n{row['target']}"
    row["mask_end"] = len(block)
    return row


def _metamath_statement_route_record(source_metadata, row_id, route, statement):
    row = generation._metamath_record(
        row_id,
        source_metadata,
        fact_name="safe",
        fact_statement="|- safe",
    )
    if route == "facts.values":
        row["facts"] = {"renamed-held": statement}
        row["cited"] = ["renamed-held"]
        row["target"] = "  1  renamed-held   |- safe"
    elif route == "goal":
        row["goal"] = statement
    elif route == "target-expression":
        row["target"] = f"  1  safe           {statement}"
    elif route == "local-assumption":
        row["local_assumptions"] = {"local.1": statement}
    else:  # pragma: no cover - the parametrization is closed.
        raise AssertionError(f"unknown Metamath statement route: {route}")
    return _rerender_metamath_record(row)


def _metamath_production_route_rows(source_metadata):
    near_miss = generation._metamath_record(
        "near-misses",
        source_metadata,
        fact_name="renamed-near",
        fact_statement="|- held suffix",
    )
    near_miss["goal"] = "|- held suffix"
    near_miss["target"] = "  1  renamed-near   |- held suffix"
    near_miss["local_assumptions"] = {"local.near": "|- held suffix"}
    _rerender_metamath_record(near_miss)
    return [
        (
            "held-cited-name",
            "eval",
            generation._metamath_record(
                "held-cited-name",
                source_metadata,
                fact_name="held",
                fact_statement="|- held",
            ),
        ),
        ("near-misses", "train", near_miss),
        (
            "theorem-name",
            "drop",
            generation._metamath_record(
                "theorem-name",
                source_metadata,
                fact_name="safe",
                fact_statement="|- safe",
                theorem="set:held",
            ),
        ),
        (
            "facts.values",
            "eval",
            _metamath_statement_route_record(
                source_metadata,
                "facts-values",
                "facts.values",
                "  |-   held  ",
            ),
        ),
        (
            "goal",
            "drop",
            _metamath_statement_route_record(
                source_metadata,
                "goal-alias",
                "goal",
                "  |-   held  ",
            ),
        ),
        (
            "target-expression",
            "eval",
            _metamath_statement_route_record(
                source_metadata,
                "target-expression",
                "target-expression",
                "  |-   held  ",
            ),
        ),
        (
            "local-assumption",
            "eval",
            _metamath_statement_route_record(
                source_metadata,
                "local-assumption",
                "local-assumption",
                "  |-   held  ",
            ),
        ),
    ]


def _write_metamath_normalization_case(
    tmp_path,
    source_manifest,
    entries,
    *,
    force_train=None,
):
    raw = tmp_path / "raw.jsonl"
    train = tmp_path / "train.jsonl"
    evaluation = tmp_path / "eval.jsonl"
    heldout = tmp_path / "heldout.json"
    generation._write_jsonl(raw, [row for _, _, row in entries])
    generation._write_jsonl(
        train,
        [
            row
            for route, disposition, row in entries
            if disposition == "train" or route == force_train
        ],
    )
    generation._write_jsonl(
        evaluation,
        [
            row
            for route, disposition, row in entries
            if disposition in {"eval", "drop"} and route != force_train
        ],
    )
    heldout.write_text(
        json.dumps(
            {
                "schema_version": "metamath-heldout-v2",
                "family": "metamath",
                "mode": "family_local_heldout",
                "facts": ["held", *(f"unused-{index}" for index in range(499))],
                "requested_heldout": 500,
                "local_assumptions": True,
            }
        )
        + "\n"
    )
    return raw, train, evaluation, heldout


def test_metamath_normalizer_uses_statement_aware_production_routes(tmp_path):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    entries = _metamath_production_route_rows(source_manifest["row_source_metadata"])
    raw, train, evaluation, heldout = _write_metamath_normalization_case(
        tmp_path,
        source_manifest,
        entries,
    )

    package = generation._normalize_metamath_package(
        raw_output=raw,
        split_outputs={
            "train": train,
            "eval": evaluation,
            "heldout": heldout,
        },
        destination=tmp_path / "normalized",
        source_manifest=source_manifest,
    )

    assert [
        item.record["id"]
        for item in generation._read_occurrences(package.train, label="normalized train")
    ] == [row["id"] for _, disposition, row in entries if disposition == "train"]
    assert [
        item.record["id"]
        for item in generation._read_occurrences(package.eval, label="normalized eval")
    ] == [row["id"] for _, disposition, row in entries if disposition == "eval"]
    assert [(drop.raw_row, drop.drop_type) for drop in package.drops] == [
        (raw_row, "heldout_own_proof")
        for raw_row, (_, disposition, _) in enumerate(entries, 1)
        if disposition == "drop"
    ]


def test_metamath_normalizer_accepts_builder_drop_ledger_path(tmp_path):
    tokenizer = generation.mml_holdout.approved_tokenizer_seal()
    ledger = _make_metamath_drop_ledger(
        [_metamath_drop_entry()],
        eligible_rows=7,
    )
    source_manifest = generation._make_source_manifest("metamath", test_only=False)
    source_metadata = source_manifest["row_source_metadata"]
    source_metadata.update(
        {
            "drop_ledger": {
                "schema_version": ledger["schema_version"],
                "canonical_root_sha256": ledger["canonical_root_sha256"],
                "entries_root_sha256": ledger["entries_root_sha256"],
                "accounting": dict(ledger["accounting"]),
            },
            "tokenizer_seal": dict(tokenizer),
            "tokenizer_root_sha256": ledger["tokenizer_root_sha256"],
        }
    )
    source_manifest["manifest_root_sha256"] = generation._source_manifest_root(
        source_manifest
    )
    entries = _metamath_production_route_rows(source_metadata)
    raw, train, evaluation, heldout = _write_metamath_normalization_case(
        tmp_path,
        source_manifest,
        entries,
    )
    heldout_payload = json.loads(heldout.read_text())
    heldout_payload["eligibility"] = {
        "accounting": dict(ledger["accounting"]),
        "drop_ledger": {
            "path": "drops/metamath-overlength.json",
            "schema_version": ledger["schema_version"],
            "canonical_root_sha256": ledger["canonical_root_sha256"],
            "entries_root_sha256": ledger["entries_root_sha256"],
        },
        "max_text_plus_eos_tokens": metamath_builder.MAX_TEXT_PLUS_EOS_TOKENS,
        "tokenizer_seal": dict(tokenizer),
        "tokenizer_root_sha256": ledger["tokenizer_root_sha256"],
    }
    heldout_payload["partition_accounting"] = {
        "source_rows": ledger["accounting"]["source_rows"],
        "source_text_plus_eos_tokens": ledger["accounting"][
            "source_text_plus_eos_tokens"
        ],
        "train_rows": 1,
        "eval_rows": 6,
        "train_text_plus_eos_tokens": 40,
        "eval_text_plus_eos_tokens": 60,
        "drop_rows": ledger["accounting"]["dropped_rows"],
        "drop_text_plus_eos_tokens": ledger["accounting"][
            "dropped_text_plus_eos_tokens"
        ],
    }
    heldout_payload["manifest_root_sha256"] = metamath_builder.canonical_sha256(
        heldout_payload
    )
    heldout.write_text(json.dumps(heldout_payload) + "\n")

    package = generation._normalize_metamath_package(
        raw_output=raw,
        split_outputs={
            "train": train,
            "eval": evaluation,
            "heldout": heldout,
        },
        destination=tmp_path / "normalized",
        source_manifest=source_manifest,
        drop_ledger=ledger,
        tokenizer_seal=tokenizer,
    )

    assert package.source_accounting["source_rows"] == 8


def _metamath_exposure_candidate(source_metadata, route):
    if route == "held-cited-name":
        row = generation._metamath_record(
            "candidate-held-cited-name",
            source_metadata,
            fact_name="held",
            fact_statement="|- held",
        )
        return "eval", row
    if route == "theorem-name":
        row = generation._metamath_record(
            "candidate-theorem-name",
            source_metadata,
            fact_name="safe",
            fact_statement="|- safe",
            theorem="set:held",
        )
        return "drop", row
    disposition = "drop" if route == "goal" else "eval"
    return disposition, _metamath_statement_route_record(
        source_metadata,
        f"candidate-{route}",
        route,
        "  |-   held  ",
    )


@pytest.mark.parametrize(
    "visible_route",
    (
        "theorem-name",
        "goal",
        "facts.values",
        "target-expression",
        "local-assumption",
        "held-cited-name",
    ),
)
def test_metamath_normalizer_rejects_each_exposure_forced_into_train(
    tmp_path,
    visible_route,
):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    source_metadata = source_manifest["row_source_metadata"]
    candidate_disposition, candidate = _metamath_exposure_candidate(
        source_metadata,
        visible_route,
    )
    entries = [
        (
            "baseline-held",
            "eval",
            generation._metamath_record(
                "baseline-held",
                source_metadata,
                fact_name="held",
                fact_statement="|- held",
            ),
        ),
        (
            "baseline-train",
            "train",
            generation._metamath_record(
                "baseline-train",
                source_metadata,
                fact_name="safe",
                fact_statement="|- safe",
            ),
        ),
        (
            "baseline-drop",
            "drop",
            generation._metamath_record(
                "baseline-drop",
                source_metadata,
                fact_name="safe",
                fact_statement="|- safe",
                theorem="set:held",
            ),
        ),
        (visible_route, candidate_disposition, candidate),
    ]
    raw, train, evaluation, heldout = _write_metamath_normalization_case(
        tmp_path,
        source_manifest,
        entries,
        force_train=visible_route,
    )

    with pytest.raises(generation.IntegrationError, match="builder split disagrees"):
        generation._normalize_metamath_package(
            raw_output=raw,
            split_outputs={
                "train": train,
                "eval": evaluation,
                "heldout": heldout,
            },
            destination=tmp_path / "normalized",
            source_manifest=source_manifest,
        )


def test_metamath_normalizer_rejects_malformed_target_before_routing(tmp_path):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    entries = _metamath_production_route_rows(source_manifest["row_source_metadata"])
    target_row = next(row for route, _, row in entries if route == "target-expression")
    target_row["target"] = "malformed target"
    _rerender_metamath_record(target_row)
    raw, train, evaluation, heldout = _write_metamath_normalization_case(
        tmp_path,
        source_manifest,
        entries,
    )

    with pytest.raises(generation.IntegrationError, match="malformed Metamath target"):
        generation._normalize_metamath_package(
            raw_output=raw,
            split_outputs={
                "train": train,
                "eval": evaluation,
                "heldout": heldout,
            },
            destination=tmp_path / "normalized",
            source_manifest=source_manifest,
        )


def test_metamath_normalizer_requires_reconstructable_held_statement(tmp_path):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    entries = [
        entry
        for entry in _metamath_production_route_rows(
            source_manifest["row_source_metadata"]
        )
        if entry[0] != "held-cited-name"
    ]
    raw, train, evaluation, heldout = _write_metamath_normalization_case(
        tmp_path,
        source_manifest,
        entries,
    )

    with pytest.raises(
        generation.IntegrationError,
        match="held statements cannot be reconstructed",
    ):
        generation._normalize_metamath_package(
            raw_output=raw,
            split_outputs={
                "train": train,
                "eval": evaluation,
                "heldout": heldout,
            },
            destination=tmp_path / "normalized",
            source_manifest=source_manifest,
        )


@pytest.mark.parametrize("drop_mutation", ("missing", "wrong-type"))
def test_metamath_isolation_verifier_rejects_invalid_own_proof_drop(
    tmp_path,
    drop_mutation,
):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    source_metadata = source_manifest["row_source_metadata"]
    rows = [
        generation._metamath_record(
            "held-anchor",
            source_metadata,
            fact_name="held",
            fact_statement="|- held",
        ),
        generation._metamath_record(
            "safe-train",
            source_metadata,
            fact_name="safe",
            fact_statement="|- safe",
        ),
        _metamath_statement_route_record(
            source_metadata,
            "goal-own-proof",
            "goal",
            "  |-   held  ",
        ),
    ]
    raw_path = tmp_path / "metamath.jsonl"
    generation._write_jsonl(raw_path, rows)
    raw = generation._validate_rows(
        raw_path,
        family="metamath",
        source_manifest=source_manifest,
        label="minimized/metamath.jsonl",
    )
    drops = {}
    if drop_mutation == "wrong-type":
        drops[3] = {
            "drop_type": "heldout_citation",
            "occurrence_id": "metamath:3",
        }

    with pytest.raises(generation.IntegrationError, match="typed drop mismatch"):
        generation._verify_metamath_isolation(
            raw=raw,
            train=(raw[1],),
            evaluation=(raw[0],),
            heldout={"contract": {"facts": ["held"]}},
            routes={
                1: {"disposition": "eval", "occurrence_id": "metamath:1"},
                2: {"disposition": "train", "occurrence_id": "metamath:2"},
                3: {
                    "disposition": "drop",
                    "drop_type": "heldout_own_proof",
                    "occurrence_id": "metamath:3",
                },
            },
            drops=drops,
        )


@pytest.mark.parametrize(
    ("visible_route", "expected_disposition", "expected_drop_type"),
    (
        ("facts.values", "eval", None),
        ("goal", "drop", "heldout_own_proof"),
        ("target-expression", "eval", None),
        ("local-assumption", "eval", None),
    ),
)
def test_metamath_isolation_verifier_routes_canonical_visible_statements(
    tmp_path,
    visible_route,
    expected_disposition,
    expected_drop_type,
):
    source_manifest = generation._make_source_manifest("metamath", test_only=True)
    source_metadata = source_manifest["row_source_metadata"]
    rows = [
        generation._metamath_record(
            "held-anchor",
            source_metadata,
            fact_name="held",
            fact_statement="|- held",
        ),
        _metamath_statement_route_record(
            source_metadata,
            f"{visible_route}-exposed",
            visible_route,
            "  |-   held  ",
        ),
        _metamath_statement_route_record(
            source_metadata,
            f"{visible_route}-near-miss",
            visible_route,
            "|- held suffix",
        ),
    ]
    raw_path = tmp_path / "metamath.jsonl"
    generation._write_jsonl(raw_path, rows)
    raw = generation._validate_rows(
        raw_path,
        family="metamath",
        source_manifest=source_manifest,
        label="minimized/metamath.jsonl",
    )

    routes = {
        1: {"disposition": "eval", "occurrence_id": "metamath:1"},
        2: {
            "disposition": expected_disposition,
            "drop_type": expected_drop_type,
            "occurrence_id": "metamath:2",
        },
        3: {"disposition": "train", "occurrence_id": "metamath:3"},
    }
    drops = (
        {
            2: {
                "drop_type": expected_drop_type,
                "occurrence_id": "metamath:2",
            }
        }
        if expected_drop_type
        else {}
    )
    assert [routes[index]["disposition"] for index in routes] == [
        "eval",
        expected_disposition,
        "train",
    ]
    assert set(drops) == ({2} if expected_drop_type else set())

    generation._verify_metamath_isolation(
        raw=raw,
        train=(raw[2],),
        evaluation=(raw[0], raw[1]) if expected_disposition == "eval" else (raw[0],),
        heldout={"contract": {"facts": ["held"]}},
        routes=routes,
        drops=drops,
    )


def test_independent_verifier_recomputes_metamath_and_isabelle_isolation(tmp_path):
    result = generation.build_synthetic_generation(
        corpus_root=tmp_path / "transaction",
        work_root=tmp_path,
        generation_id="independent-isolation",
    )
    metamath = _mutable_generation_file(result, "shards/metamath.jsonl")
    row = json.loads(metamath.read_text())
    row["theorem"] = "set:mp"
    metamath.write_text(json.dumps(row) + "\n")
    with pytest.raises(generation.IntegrationError, match="Metamath.*held|own-proof"):
        generation._independent_verify_generation_path(
            result.published.path,
            production=False,
        )

    metamath.write_bytes(
        (result.published.path / "raw/metamath.jsonl").read_bytes().splitlines(
            keepends=True
        )[0]
    )
    isabelle = _mutable_generation_file(result, "shards/isabelle.jsonl")
    eval_row = json.loads(
        (result.published.path / "eval/isabelle.jsonl").read_text()
    )
    train_row = json.loads(isabelle.read_text())
    train_row["trajectory_id"] = eval_row["trajectory_id"]
    isabelle.write_text(json.dumps(train_row) + "\n")
    with pytest.raises(generation.IntegrationError, match="Isabelle.*trajectory"):
        generation._independent_verify_generation_path(
            result.published.path,
            production=False,
        )


def test_fault_matrix_preserves_prior_current_and_exact_quarantines(tmp_path):
    root = tmp_path / "transaction"
    work = tmp_path / "work"
    work.mkdir()
    baseline = generation.build_synthetic_generation(
        corpus_root=root,
        work_root=work,
        generation_id="fault-baseline",
    )
    old_tree = _tree_payload(baseline.published.path)
    points = generation.synthetic_fault_points()
    expected_prefixes = {
        *(f"raw_builder:{family}" for family in FAMILIES),
        *(f"builder_complete:{family}" for family in FAMILIES),
        "split_builder:start:metamath",
        "split_builder:complete:metamath",
        "split_builder:start:isabelle",
        "split_builder:complete:isabelle",
        "split_builder:start:mml",
        "split_builder:complete:mml",
        *(f"family_split:complete:{family}" for family in FAMILIES),
        "normalization:metamath",
        "normalization:isabelle",
        "partition:metamath",
        "partition:isabelle",
        "mml_partition",
        "precheck",
    }
    assert expected_prefixes <= set(points)
    assert {
        f"final_copy:{path}"
        for path in generation._expected_output_paths()
    } <= set(points)

    for index, point in enumerate(points):
        with pytest.raises(generation.IntegrationError, match="injected fault"):
            generation.build_synthetic_generation(
                corpus_root=root,
                work_root=work,
                generation_id=f"fault-{index}",
                fault_point=point,
            )
        coordinator = GenerationCoordinator(root)
        assert coordinator.resolve_current().generation_id == "fault-baseline"
        assert _tree_payload(baseline.published.path) == old_tree
        assert not (coordinator.generations_directory / f"fault-{index}").exists()
        if point.startswith(("split_builder:", "family_split:")):
            matching = [
                entry
                for entry in coordinator.quarantine_directory.iterdir()
                if json.loads((entry / "QUARANTINE.json").read_text()).get(
                    "generation_id"
                )
                == f"fault-{index}"
            ]
            assert len(matching) == 1
            payload = matching[0] / "payload"
            assert not any((payload / path).exists() for path in generation._expected_output_paths())

    transaction_quarantine = GenerationCoordinator(root).quarantine_inventory()
    builder_quarantine = generation.builder_quarantine_inventory(work)
    assert len(transaction_quarantine) == len(points)
    assert len(builder_quarantine) == len(points)
    assert {record["reason"] for record in builder_quarantine} == {
        f"injected fault at {point}" for point in points
    }


def _materialized_mml_contract(generation_path, tmp_path, suffix):
    wrapper = json.loads((generation_path / "heldout" / "mml.json").read_text())
    contract_root = tmp_path / f"mml-contract-{suffix}"
    generation._materialize_mml_contract(generation_path, wrapper, contract_root)
    contract = generation.mml_holdout.load_holdout_contract(
        contract_root,
        production=False,
    )
    return wrapper, contract


def _write_pretty_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _resign_mml_manifest(contract, manifest):
    body = dict(manifest)
    body.pop("manifest_root_sha256")
    manifest["manifest_root_sha256"] = generation.mml_holdout._manifest_root(body)
    projections = generation.mml_holdout.derive_compatibility_projections(manifest)
    _write_pretty_json(contract.root / "heldout" / "mml.json", manifest)
    for family, projection in projections.items():
        _write_pretty_json(
            contract.root / "heldout" / f"{family}.json",
            projection,
        )
    return replace(
        contract,
        authoritative_root=manifest["manifest_root_sha256"],
        manifest=manifest,
        projections=projections,
    )


def _resigned_mml_tuple_mutant(generation_path, tmp_path, component):
    _, contract = _materialized_mml_contract(
        generation_path,
        tmp_path,
        component,
    )
    manifest = json.loads(json.dumps(contract.manifest))
    contract_tuple = manifest["contract_tuple"]
    if component == "edullm_data_commit":
        contract_tuple["edullm_data_commit"] = "0" * 40
    elif component == "contract_tuple_schema":
        contract_tuple["schema_version"] = "mml-semantic-holdout-contract-tuple-v1"
    else:
        contract_tuple["components"][component] = {
            "version": "downgraded-or-arbitrary",
            "sha256": "0" * 64,
        }
    manifest["contract_tuple_sha256"] = generation.mml_holdout._json_sha256(
        contract_tuple
    )
    return _resign_mml_manifest(contract, manifest)


class _RoutePlanningForbidden(dict):
    def __getitem__(self, key):
        raise AssertionError(f"route planning started for {key}")


class _PublicationForbidden:
    def __getattr__(self, name):
        raise AssertionError(f"publication writer was touched through {name}")


def _write_with_forbidden_routes(contract, *, production=False):
    generation._write_transaction_payload(
        _PublicationForbidden(),
        source_manifests=_RoutePlanningForbidden(),
        tokenizer={},
        policies={"test_only": not production},
        packages=_RoutePlanningForbidden(),
        mml_contract=contract,
    )


@pytest.mark.parametrize(
    "component",
    [
        "manifest",
        "loader",
        "compatibility",
        "policy",
        "mapping",
        "statement_hash",
        "atp_deduplication",
        "source_policy",
        "canonicalization",
        "edullm_data_commit",
        "contract_tuple_schema",
    ],
)
def test_generation_boundary_rejects_resigned_noncanonical_mml_tuple_before_routes(
    structured_generation_path,
    tmp_path,
    component,
):
    hostile_contract = _resigned_mml_tuple_mutant(
        structured_generation_path,
        tmp_path,
        component,
    )

    with pytest.raises(generation.IntegrationError, match="MML.*contract tuple"):
        _write_with_forbidden_routes(hostile_contract)


def test_generation_persists_exact_mml_tuple_inventory_and_roots(
    structured_generation_path,
    tmp_path,
):
    wrapper, contract = _materialized_mml_contract(
        structured_generation_path,
        tmp_path,
        "persisted",
    )
    manifest = contract.manifest
    expected_tuple = generation.mml_holdout.canonical_contract_tuple(
        manifest["source_identity_policy"]
    )

    assert manifest["contract_tuple"] == expected_tuple
    assert manifest["contract_tuple_sha256"] == generation.mml_holdout._json_sha256(
        expected_tuple
    )
    assert (
        expected_tuple["edullm_data_commit"]
        == generation.mml_holdout.EDULLM_DATA_COMMIT
    )
    assert (
        wrapper["authoritative_manifest_root_sha256"]
        == manifest["manifest_root_sha256"]
        == contract.authoritative_root
    )
    assert manifest[
        "artifact_inventory_root_sha256"
    ] == generation.mml_holdout.artifact_inventory_root(manifest["artifact_inventory"])
    assert set(contract.artifacts) == {
        record["path"] for record in manifest["artifact_inventory"]
    }
    assert wrapper["contract_manifest"] == manifest
    assert wrapper["projections"] == contract.projections


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("stale-projection", "compatibility projection"),
        ("missing-artifact", "inventory"),
        ("extra-artifact", "inventory"),
    ],
)
def test_generation_boundary_inherits_exact_projection_and_inventory_checks(
    structured_generation_path,
    tmp_path,
    mutation,
    message,
):
    _, contract = _materialized_mml_contract(
        structured_generation_path,
        tmp_path,
        mutation,
    )
    if mutation == "stale-projection":
        projection_path = contract.root / "heldout" / "mizar.json"
        projection = json.loads(projection_path.read_text())
        projection["facts"].append("DRIFT:1")
        projection_body = dict(projection)
        projection_body.pop("projection_root_sha256")
        projection["projection_root_sha256"] = generation.mml_holdout._json_sha256(
            projection_body
        )
        _write_pretty_json(projection_path, projection)
    elif mutation == "missing-artifact":
        (contract.root / "eval" / "enigma.jsonl").unlink()
    else:
        (contract.root / "sidecars" / "extra.jsonl").write_bytes(b"")

    with pytest.raises(generation.IntegrationError, match=message):
        _write_with_forbidden_routes(contract)


def test_generation_boundary_refuses_test_only_contract_in_production(
    structured_generation_path,
    tmp_path,
):
    _, contract = _materialized_mml_contract(
        structured_generation_path,
        tmp_path,
        "test-only-production",
    )

    with pytest.raises(generation.IntegrationError, match="test-only"):
        _write_with_forbidden_routes(
            replace(contract, production=True),
            production=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-step-field", "missing ATP step fields"),
        ("malformed-delimiters", "malformed TPTP delimiters"),
    ],
)
def test_generation_boundary_rejects_resigned_malformed_atp_artifacts(
    structured_generation_path,
    tmp_path,
    mutation,
    message,
):
    _, contract = _materialized_mml_contract(
        structured_generation_path,
        tmp_path,
        mutation,
    )
    artifact_path = contract.root / "shards" / "prf2.jsonl"
    lines = artifact_path.read_bytes().splitlines(keepends=True)
    row = json.loads(lines[0])
    if mutation == "missing-step-field":
        row["proof_steps"][0].pop("source")
    else:
        row["facts"][next(iter(row["facts"]))] = "([{]})"
    lines[0] = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    payload = b"".join(lines)
    artifact_path.write_bytes(payload)

    manifest = json.loads(json.dumps(contract.manifest))
    route = next(
        record
        for record in manifest["row_routes"]["prf2"]
        if record["disposition"] == "train"
    )
    route["native_row_sha256"] = hashlib.sha256(lines[0]).hexdigest()
    manifest["route_plan_root_sha256"] = generation.mml_holdout.route_plan_root(
        manifest["row_routes"]
    )
    manifest["partition_projections"]["route_plan_root_sha256"] = manifest[
        "route_plan_root_sha256"
    ]
    manifest["partition_projections"]["by_shard"]["prf2"]["route_root_sha256"] = (
        generation.mml_holdout._json_sha256(manifest["row_routes"]["prf2"])
    )
    inventory_record = next(
        record
        for record in manifest["artifact_inventory"]
        if record["path"] == "shards/prf2.jsonl"
    )
    inventory_record["sha256"] = hashlib.sha256(payload).hexdigest()
    inventory_record["bytes"] = len(payload)
    inventory_record["rows"] = len(lines)
    manifest["artifact_inventory_root_sha256"] = (
        generation.mml_holdout.artifact_inventory_root(manifest["artifact_inventory"])
    )
    hostile_contract = _resign_mml_manifest(contract, manifest)

    with pytest.raises(generation.IntegrationError, match=message):
        _write_with_forbidden_routes(hostile_contract)


def _current_production_mml_sources():
    policy = generation.mml_holdout.production_source_policy()
    return {
        family: {
            "row_source_metadata": {
                "source_manifest_root_sha256": (
                    policy.shards[family].source_manifest_root_sha256
                ),
                "quality_filter_root_sha256": (
                    policy.shards[family].quality_filter_root_sha256
                ),
                "schema_generation_root_sha256": (
                    policy.shards[family].schema_generation_root_sha256
                ),
            },
            "source_verifier_acceptance": {
                "accepted": True,
                "status": "fixture source replay clean",
            },
            "manifest_root_sha256": hashlib.sha256(
                f"outer-source-manifest:{family}".encode()
            ).hexdigest(),
        }
        for family in generation.MML_SIBLINGS
    }


def _current_production_policies():
    pins = generation.mml_holdout.current_policy_pins()
    return {
        "mml": {
            "policy_pins": {
                "policy_sha256": pins.policy_sha256,
                "mapping_sha256": pins.mapping_sha256,
                "atp_deduplication_sha256": pins.atp_deduplication_sha256,
            }
        }
    }


def _current_production_mml_contract(tmp_path, *, root=None):
    root = root or (tmp_path / "persisted-mml-v7")
    root.mkdir(parents=True, exist_ok=True)
    policy = generation.mml_holdout.production_source_policy()
    table = generation.mml_holdout.PRODUCTION_SOURCE_IDENTITY_TABLE
    ordered_inputs = []
    family_paths = {}
    for family in generation.MML_SIBLINGS:
        approved = policy.shards[family]
        ordered_inputs.append(
            {
                "shard": family,
                "logical_path": f"raw/{family}.jsonl",
                "sha256": approved.input_sha256,
                "rows": table[family]["input_rows"],
                "source_snapshots": [
                    {
                        "reference": snapshot.reference,
                        "sha256": snapshot.sha256,
                    }
                    for snapshot in approved.source_snapshots
                ],
                "source_manifest_root_sha256": (
                    approved.source_manifest_root_sha256
                ),
                "quality_filter_root_sha256": approved.quality_filter_root_sha256,
                "schema_generation_root_sha256": (
                    approved.schema_generation_root_sha256
                ),
                "deduplication_root_sha256": policy.deduplication_roots[family],
                "acceptance_roots": dict(approved.acceptance_roots),
            }
        )
        family_root = root / family
        family_root.mkdir(exist_ok=True)
        paths = {}
        for disposition in ("train", "eval", "dropped"):
            path = family_root / f"{disposition}.jsonl"
            path.write_bytes(b"")
            paths[disposition] = path
        family_paths[family] = generation.mml_holdout.FamilyPaths(
            train=paths["train"],
            eval=paths["eval"],
            dropped=paths["dropped"],
        )

    manifest = {
        "schema_version": generation.mml_holdout.MANIFEST_SCHEMA_VERSION,
        "selected_classes": 1_000,
        "seed": 20_260_801,
        "ordered_inputs": ordered_inputs,
        "source_root_sha256": generation.mml_holdout.source_root(ordered_inputs),
        "quality_filter_root_sha256": generation.mml_holdout._json_sha256(
            [
                {
                    "shard": record["shard"],
                    "quality_filter_root_sha256": record[
                        "quality_filter_root_sha256"
                    ],
                }
                for record in ordered_inputs
            ]
        ),
        "schema_generation_root_sha256": generation.mml_holdout._json_sha256(
            [
                {
                    "shard": record["shard"],
                    "schema_generation_root_sha256": record[
                        "schema_generation_root_sha256"
                    ],
                }
                for record in ordered_inputs
            ]
        ),
        "deduplication_root_sha256": generation.mml_holdout._json_sha256(
            [
                {
                    "shard": record["shard"],
                    "deduplication_root_sha256": record[
                        "deduplication_root_sha256"
                    ],
                }
                for record in ordered_inputs
            ]
        ),
        "acceptance_root_sha256": generation.mml_holdout._json_sha256(
            [
                {
                    "shard": record["shard"],
                    "acceptance_roots": record["acceptance_roots"],
                }
                for record in ordered_inputs
            ]
        ),
        "tokenizer_root_sha256": generation.mml_holdout._json_sha256(
            generation.mml_holdout.approved_tokenizer_seal()
        ),
        "route_plan_root_sha256": hashlib.sha256(b"current route plan").hexdigest(),
        "artifact_inventory_root_sha256": hashlib.sha256(
            b"current artifact inventory"
        ).hexdigest(),
    }
    manifest["manifest_root_sha256"] = hashlib.sha256(
        b"authoritative mml v7"
    ).hexdigest()
    return generation.mml_holdout.ValidatedHoldoutContract(
        root=root,
        production=True,
        test_only=False,
        authoritative_root=manifest["manifest_root_sha256"],
        manifest=manifest,
        projections={},
        artifacts={
            "heldout/mml.json": generation.mml_holdout.PublishedArtifact(
                path=root / "heldout" / "mml.json",
                sha256=hashlib.sha256(b"manifest").hexdigest(),
                bytes=1,
                rows=1,
                schema=generation.mml_holdout.MANIFEST_SCHEMA_VERSION,
            )
        },
        family_paths=family_paths,
        exposure_index={},
        tokenizer_root_sha256=manifest["tokenizer_root_sha256"],
        source_root_sha256=manifest["source_root_sha256"],
        quality_filter_roots_by_shard={
            record["shard"]: record["quality_filter_root_sha256"]
            for record in ordered_inputs
        },
        schema_generation_roots_by_shard={
            record["shard"]: record["schema_generation_root_sha256"]
            for record in ordered_inputs
        },
        deduplication_roots_by_shard={
            record["shard"]: record["deduplication_root_sha256"]
            for record in ordered_inputs
        },
        acceptance_roots_by_shard={
            record["shard"]: record["acceptance_roots"]
            for record in ordered_inputs
        },
    )


def test_production_contract_loader_uses_authoritative_v7_validator_and_current_shapes(
    tmp_path,
    monkeypatch,
):
    contract = _current_production_mml_contract(tmp_path)
    calls = []

    def load(root, *, production):
        calls.append((Path(root), production))
        return contract

    monkeypatch.setattr(generation.mml_holdout, "load_holdout_contract", load)
    loaded = generation._load_production_mml_contract(
        contract.root,
        source_manifests=_current_production_mml_sources(),
        tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
        policies=_current_production_policies(),
    )

    assert loaded is contract
    assert calls == [(contract.root.resolve(), True)]
    inputs = {record["shard"]: record for record in loaded.manifest["ordered_inputs"]}
    assert inputs["mizar"]["rows"] == 55_353
    assert inputs["enigma"]["rows"] == 29_166


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing|unreadable"),
        ("wrong-schema", "authoritative|manifest-v7"),
    ],
)
def test_missing_or_invalid_persisted_production_contract_fails_closed(
    tmp_path,
    mutation,
    message,
):
    root = tmp_path / "persisted-mml-v7"
    if mutation == "wrong-schema":
        (root / "heldout").mkdir(parents=True)
        (root / "heldout" / "mml.json").write_text(
            json.dumps({"schema_version": "mml-semantic-holdout-manifest-v6"})
        )

    with pytest.raises(generation.IntegrationError, match=message):
        generation._load_production_mml_contract(
            root,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
        )


def test_mutated_persisted_production_contract_artifact_fails_closed(
    structured_generation_path,
    tmp_path,
    monkeypatch,
):
    wrapper = json.loads((structured_generation_path / "heldout" / "mml.json").read_text())
    root = tmp_path / "mutated-persisted-contract"
    generation._materialize_mml_contract(structured_generation_path, wrapper, root)
    artifact = root / "shards" / "mizar.jsonl"
    payload = bytearray(artifact.read_bytes())
    payload[len(payload) // 2] ^= 1
    artifact.write_bytes(payload)
    monkeypatch.setattr(
        generation.mml_holdout,
        "_validate_publication_mode",
        lambda manifest, *, production: False,
    )

    with pytest.raises(generation.IntegrationError, match="artifact|SHA-256"):
        generation._load_production_mml_contract(
            root,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
        )


@pytest.mark.parametrize(
    ("family", "rows", "message"),
    [
        ("mizar", 50_114, "55,353|row count|stale"),
        ("enigma", 27_079, "29,166|row count|base-only"),
    ],
)
def test_persisted_contract_rejects_stale_mizar_and_base_only_enigma(
    tmp_path,
    monkeypatch,
    family,
    rows,
    message,
):
    contract = _current_production_mml_contract(tmp_path)
    manifest = json.loads(json.dumps(contract.manifest))
    next(
        record for record in manifest["ordered_inputs"] if record["shard"] == family
    )["rows"] = rows
    hostile = replace(contract, manifest=manifest)
    monkeypatch.setattr(
        generation.mml_holdout,
        "load_holdout_contract",
        lambda root, *, production: hostile,
    )

    with pytest.raises(generation.IntegrationError, match=message):
        generation._load_production_mml_contract(
            contract.root,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_manifest_root_sha256", "source.manifest"),
        ("quality_filter_root_sha256", "quality"),
        ("schema_generation_root_sha256", "schema"),
        ("deduplication_root_sha256", "deduplication"),
        ("acceptance_roots", "acceptance"),
    ],
)
def test_persisted_contract_rejects_every_source_and_acceptance_root_mismatch(
    tmp_path,
    monkeypatch,
    field,
    message,
):
    contract = _current_production_mml_contract(tmp_path)
    manifest = json.loads(json.dumps(contract.manifest))
    mizar = manifest["ordered_inputs"][0]
    mizar[field] = {} if field == "acceptance_roots" else "0" * 64
    hostile = replace(contract, manifest=manifest)
    monkeypatch.setattr(
        generation.mml_holdout,
        "load_holdout_contract",
        lambda root, *, production: hostile,
    )

    with pytest.raises(generation.IntegrationError, match=message):
        generation._load_production_mml_contract(
            contract.root,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
        )


def test_persisted_contract_loader_rejects_path_substitution(tmp_path, monkeypatch):
    requested = tmp_path / "requested-contract"
    requested.mkdir()
    substituted = _current_production_mml_contract(
        tmp_path,
        root=tmp_path / "substituted-contract",
    )
    monkeypatch.setattr(
        generation.mml_holdout,
        "load_holdout_contract",
        lambda root, *, production: substituted,
    )

    with pytest.raises(generation.IntegrationError, match="path|substitut"):
        generation._load_production_mml_contract(
            requested,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
        )


def test_production_generation_ingests_persisted_mml_v7_without_replanning(
    tmp_path,
    monkeypatch,
):
    corpus_root = tmp_path / "corpus"
    work_root = tmp_path / "work"
    work_root.mkdir()
    index = tmp_path / "mizar.sqlite"
    index.write_bytes(b"current semantic index")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    contract = _current_production_mml_contract(tmp_path)
    source_manifests = {
        family: {
            **_current_production_mml_sources().get(family, {}),
            "manifest_root_sha256": hashlib.sha256(family.encode()).hexdigest(),
            "source_verifier_acceptance": {
                "accepted": True,
                "status": "fixture",
            },
        }
        for family in FAMILIES
    }
    calls = []

    def load_contract(
        root,
        *,
        source_manifests,
        tokenizer_seal,
        policies,
        raw_paths=None,
    ):
        del source_manifests, tokenizer_seal, policies
        calls.append((Path(root), None if raw_paths is None else set(raw_paths)))
        return contract

    def run_stage(*, family, stage, output_root, **kwargs):
        del kwargs
        output_root.mkdir(parents=True)
        if stage == "raw":
            path = output_root / f"{family}.jsonl"
            path.write_text("{}\n")
            return {"raw": path}
        train = output_root / "train.jsonl"
        evaluation = output_root / "eval.jsonl"
        heldout = output_root / "heldout.json"
        train.write_text("{}\n")
        evaluation.write_text("{}\n")
        heldout.write_text("{}\n")
        return {"train": train, "eval": evaluation, "heldout": heldout}

    def normalize(*, raw_output, destination, **kwargs):
        del kwargs
        destination.mkdir()
        train = destination / "train.jsonl"
        evaluation = destination / "eval.jsonl"
        train.write_text("{}\n")
        evaluation.write_text("{}\n")
        return generation._FamilyPackage(
            family="local",
            raw=raw_output,
            train=train,
            eval=evaluation,
            drops=(),
            heldout={},
        )

    class Coordinator:
        def __init__(self, root):
            self.root = root

        def publish(self, plan, producer):
            producer(object())
            return SimpleNamespace(
                generation_id=plan.generation_id,
                logical_root_sha256="f" * 64,
                path=Path(self.root) / "generations" / plan.generation_id,
            )

    monkeypatch.setattr(generation, "_load_production_mml_contract", load_contract)
    monkeypatch.setattr(
        generation,
        "_build_production_mml_contract",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("production MML replanning was invoked")
        ),
    )
    monkeypatch.setattr(
        generation,
        "_validate_source_manifest",
        lambda manifest, **kwargs: manifest,
    )
    monkeypatch.setattr(generation, "production_blockers", lambda manifests: [])
    monkeypatch.setattr(
        generation,
        "_validate_production_builder_config",
        lambda manifest, *, family: {
            "raw": {"argv": []},
            **(
                {}
                if family in generation.MML_SIBLINGS
                else {"split": {"argv": []}}
            ),
        },
    )
    monkeypatch.setattr(
        generation,
        "_validate_tokenizer_seal",
        lambda seal: dict(seal),
    )
    monkeypatch.setattr(
        generation,
        "_validate_metamath_drop_ledger",
        lambda ledger, **kwargs: dict(ledger),
    )
    monkeypatch.setattr(
        generation,
        "_validate_builder_native_source_metadata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation,
        "_validate_policies",
        lambda policies, *, production: policies,
    )
    monkeypatch.setattr(
        generation,
        "_validate_enigma_low_tier_input_binding",
        lambda *args, **kwargs: ({}, {}, {"schema_version": "fixture-v1"}),
    )
    monkeypatch.setattr(
        generation,
        "_bind_enigma_low_tier_source_manifest",
        lambda manifest, binding: manifest,
    )
    monkeypatch.setattr(generation, "_source_generation_id", lambda *args, **kwargs: "src")
    monkeypatch.setattr(generation, "_run_builder_stage", run_stage)
    monkeypatch.setattr(generation, "_normalize_metamath_package", normalize)
    monkeypatch.setattr(generation, "_normalize_isabelle_package", normalize)
    monkeypatch.setattr(generation, "_verify_external_mizar_rows", lambda *args, **kwargs: None)
    published_contracts = []
    monkeypatch.setattr(
        generation,
        "_write_transaction_payload",
        lambda writer, **kwargs: published_contracts.append(kwargs["mml_contract"]),
    )
    monkeypatch.setattr(generation, "GenerationCoordinator", Coordinator)

    result = generation.build_production_generation(
        corpus_root=corpus_root,
        work_root=work_root,
        generation_id="persisted-mml",
        source_manifests=source_manifests,
        tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
        tokenizer_path=tokenizer_path,
        metamath_drop_ledger={},
        policies={"test_only": False, "policy_root_sha256": "e" * 64},
        mizar_semantic_index=index,
        mml_contract_root=contract.root,
    )

    assert result.published.generation_id == "persisted-mml"
    assert calls == [
        (contract.root, None),
        (contract.root, set(generation.MML_SIBLINGS)),
    ]
    assert published_contracts == [contract]


def test_production_enigma_raw_command_requires_approved_low_tier_pair(
    tmp_path_factory,
):
    argv, _ = _enigma_low_tier_command(tmp_path_factory)
    for flag in ("--enigma-low-tier-base", "--tokenizer-json"):
        index = argv.index(flag)
        del argv[index : index + 2]

    with pytest.raises(generation.IntegrationError, match="low-tier|required|missing"):
        generation.validate_production_builder_command(
            family="enigma",
            stage="raw",
            argv=argv,
        )


def _small_enigma_low_tier_inputs(tmp_path, monkeypatch):
    base = tmp_path / "enigma-accepted-base"
    (base / "shards").mkdir(parents=True)
    (base / "eval").mkdir()
    (base / "heldout").mkdir()
    shard = base / "shards" / "enigma.jsonl"
    shard.write_bytes(b'{"id":"base"}\n')
    (base / "eval" / "enigma.jsonl").write_bytes(b"")
    (base / "heldout" / "enigma.json").write_text("{}\n")
    expected = {
        "bytes": shard.stat().st_size,
        "rows": 1,
        "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
    }
    source_contract = json.loads(
        json.dumps(atp_builder.ENIGMA_LOW_TIER_SOURCE_CONTRACT)
    )
    source_contract["accepted_base"] = expected
    monkeypatch.setattr(
        atp_builder,
        "ENIGMA_LOW_TIER_SOURCE_CONTRACT",
        source_contract,
    )
    source = tmp_path / "mzr01"
    source.mkdir()
    argv = [
        sys.executable,
        "scripts/build_atp_shard.py",
        "--src",
        str(source),
        "--name",
        "enigma",
        "--fenced",
        "--heldout",
        "0",
        "--min-steps",
        "4",
        "--dedup",
        "--jaccard",
        "0.5",
        "--seed",
        "20260801",
        "--enigma-low-tier-base",
        str(base),
        "--tokenizer-json",
        str(FIXED_QWEN_TOKENIZER / "tokenizer.json"),
    ]
    return argv, base, expected


def test_preflight_binds_enigma_acceptance_and_tokenizer_roots_into_source_metadata(
    tmp_path,
    monkeypatch,
):
    argv, base, expected_base = _small_enigma_low_tier_inputs(tmp_path, monkeypatch)
    contract = _current_production_mml_contract(tmp_path)
    paths, roots, binding = generation._validate_enigma_low_tier_input_binding(
        argv,
        tokenizer_path=FIXED_QWEN_TOKENIZER,
        tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
        mml_contract=contract,
    )
    source_manifest = {
        "source_verifier_acceptance": {
            "accepted": True,
            "status": "accepted low-tier fixture",
        },
        "row_source_metadata": {
            "source_manifest_root_sha256": contract.manifest["ordered_inputs"][3][
                "source_manifest_root_sha256"
            ]
        },
        "manifest_root_sha256": "0" * 64,
    }
    bound = generation._bind_enigma_low_tier_source_manifest(
        source_manifest,
        binding,
    )

    assert paths == {
        "enigma_low_tier_base": base.resolve(),
        "tokenizer_json": (FIXED_QWEN_TOKENIZER / "tokenizer.json").resolve(),
    }
    assert roots["accepted_base_sha256"] == expected_base["sha256"]
    assert roots["acceptance_root_sha256"] == contract.manifest[
        "acceptance_root_sha256"
    ]
    assert roots["tokenizer_root_sha256"] == contract.tokenizer_root_sha256
    assert binding["final_enigma_rows"] == 29_166
    assert binding["acceptance_roots"] == contract.acceptance_roots_by_shard[
        "enigma"
    ]
    assert (
        bound["source_verifier_acceptance"]["generation_input_binding"] == binding
    )
    assert bound["row_source_metadata"] == source_manifest["row_source_metadata"]
    assert bound["manifest_root_sha256"] == generation._source_manifest_root(bound)


@pytest.mark.parametrize("mutation", ("accepted-base", "tokenizer", "path-substitution"))
def test_enigma_low_tier_preflight_detects_mutation_and_path_substitution(
    tmp_path,
    monkeypatch,
    mutation,
):
    argv, base, _ = _small_enigma_low_tier_inputs(tmp_path, monkeypatch)
    tokenizer_path = FIXED_QWEN_TOKENIZER
    if mutation == "accepted-base":
        (base / "shards" / "enigma.jsonl").write_bytes(b'{"id":"mutated"}\n')
    else:
        relocated = tmp_path / "relocated-tokenizer"
        relocated.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (relocated / name).write_bytes((FIXED_QWEN_TOKENIZER / name).read_bytes())
        argv[argv.index("--tokenizer-json") + 1] = str(relocated / "tokenizer.json")
        if mutation == "tokenizer":
            tokenizer_path = relocated
            tokenizer = relocated / "tokenizer.json"
            payload = tokenizer.read_bytes()
            tokenizer.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])

    with pytest.raises(
        generation.IntegrationError,
        match="accepted base|tokenizer|path|SHA-256",
    ):
        generation._validate_enigma_low_tier_input_binding(
            argv,
            tokenizer_path=tokenizer_path,
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            mml_contract=_current_production_mml_contract(tmp_path),
        )


def _raw_mml_paths(tmp_path):
    paths = {}
    for family in generation.MML_SIBLINGS:
        path = tmp_path / "raw-builders" / family / f"{family}.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
        paths[family] = path
    return paths


def test_persisted_contract_verifies_exact_supplied_raw_mml_roots(
    tmp_path,
    monkeypatch,
):
    contract = replace(_current_production_mml_contract(tmp_path), artifacts={})
    records = {
        record["shard"]: record for record in contract.manifest["ordered_inputs"]
    }
    raw_paths = _raw_mml_paths(tmp_path)
    monkeypatch.setattr(
        generation.mml_holdout,
        "load_holdout_contract",
        lambda root, *, production: contract,
    )
    monkeypatch.setattr(
        generation,
        "_sha256_jsonl_nofollow",
        lambda path, *, label: (
            records[Path(path).stem]["sha256"],
            0,
            records[Path(path).stem]["rows"],
        ),
    )

    loaded = generation._load_production_mml_contract(
        contract.root,
        source_manifests=_current_production_mml_sources(),
        tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
        policies=_current_production_policies(),
        raw_paths=raw_paths,
    )

    assert loaded is contract
    assert records["mizar"]["rows"] == 55_353
    assert records["enigma"]["rows"] == 29_166


@pytest.mark.parametrize("mutation", ("hash", "rows", "path-substitution"))
def test_persisted_contract_rejects_supplied_raw_mml_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    contract = replace(_current_production_mml_contract(tmp_path), artifacts={})
    records = {
        record["shard"]: record for record in contract.manifest["ordered_inputs"]
    }
    raw_paths = _raw_mml_paths(tmp_path)
    if mutation == "path-substitution":
        substituted = contract.root / "raw" / "mizar.jsonl"
        substituted.parent.mkdir()
        substituted.write_text("{}\n")
        raw_paths["mizar"] = substituted

    def metrics(path, *, label):
        del label
        family = Path(path).stem
        record = records[family]
        if family == "mizar" and mutation == "hash":
            return "0" * 64, 0, record["rows"]
        if family == "mizar" and mutation == "rows":
            return record["sha256"], 0, 50_114
        return record["sha256"], 0, record["rows"]

    monkeypatch.setattr(
        generation.mml_holdout,
        "load_holdout_contract",
        lambda root, *, production: contract,
    )
    monkeypatch.setattr(generation, "_sha256_jsonl_nofollow", metrics)

    with pytest.raises(
        generation.IntegrationError,
        match="raw MML.*root|row count|substitut",
    ):
        generation._load_production_mml_contract(
            contract.root,
            source_manifests=_current_production_mml_sources(),
            tokenizer_seal=generation.mml_holdout.approved_tokenizer_seal(),
            policies=_current_production_policies(),
            raw_paths=raw_paths,
        )


def test_cli_forwards_persisted_mml_contract_root(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    work = tmp_path / "work"
    work.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    index = tmp_path / "mizar.sqlite"
    index.write_bytes(b"index")
    contract = tmp_path / "persisted-mml-v7"
    contract.mkdir()
    seal = tmp_path / "tokenizer-seal.json"
    policies = tmp_path / "policies.json"
    drop_ledger = tmp_path / "metamath-overlength.json"
    seal.write_text("{}")
    policies.write_text("{}")
    drop_ledger.write_text("{}")
    assignments = []
    for family in FAMILIES:
        path = tmp_path / f"{family}.json"
        path.write_text("{}")
        assignments.extend(("--source-manifest", f"{family}={path}"))

    forwarded = []

    def preflight(**kwargs):
        forwarded.append(
            (
                "preflight",
                kwargs["mml_contract_root"],
                kwargs["metamath_drop_ledger_path"],
                kwargs["metamath_drop_ledger"],
            )
        )
        return (
            {"status": "ready", "blockers": []},
            {family: {} for family in FAMILIES},
            {},
            {},
        )

    def build(**kwargs):
        forwarded.append(
            (
                "build",
                kwargs["mml_contract_root"],
                kwargs["metamath_drop_ledger"],
            )
        )
        return SimpleNamespace(
            published=SimpleNamespace(
                generation_id="cli-contract",
                logical_root_sha256="f" * 64,
            )
        )

    monkeypatch.setattr(generation, "preflight_production_inputs", preflight)
    monkeypatch.setattr(generation, "build_production_generation", build)

    assert (
        generation.main(
            [
                "--corpus-root",
                str(corpus),
                "--work-root",
                str(work),
                "--generation-id",
                "cli-contract",
                "--tokenizer-seal",
                str(seal),
                "--tokenizer-path",
                str(tokenizer),
                "--policies",
                str(policies),
                "--mizar-semantic-index",
                str(index),
                "--mml-contract-root",
                str(contract),
                "--metamath-drop-ledger",
                str(drop_ledger),
                *assignments,
            ]
        )
        == 0
    )
    assert forwarded == [
        ("preflight", contract, drop_ledger, {}),
        ("build", contract, {}),
    ]


@pytest.mark.parametrize("ledger_count", (0, 2))
def test_cli_requires_exactly_one_metamath_drop_ledger(
    tmp_path,
    monkeypatch,
    capsys,
    ledger_count,
):
    corpus = tmp_path / "corpus"
    work = tmp_path / "work"
    work.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    index = tmp_path / "mizar.sqlite"
    index.write_bytes(b"index")
    contract = tmp_path / "persisted-mml-v7"
    contract.mkdir()
    seal = tmp_path / "tokenizer-seal.json"
    policies = tmp_path / "policies.json"
    ledger = tmp_path / "metamath-overlength.json"
    seal.write_text("{}")
    policies.write_text("{}")
    ledger.write_text("{}")
    assignments = []
    for family in FAMILIES:
        path = tmp_path / f"{family}.json"
        path.write_text("{}")
        assignments.extend(("--source-manifest", f"{family}={path}"))
    ledger_args = [
        value
        for _ in range(ledger_count)
        for value in ("--metamath-drop-ledger", str(ledger))
    ]
    monkeypatch.setattr(
        generation,
        "preflight_production_inputs",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid CLI reached production preflight")
        ),
    )

    result = generation.main(
        [
            "--dry-run",
            "--corpus-root",
            str(corpus),
            "--work-root",
            str(work),
            "--generation-id",
            "cli-ledger-cardinality",
            "--tokenizer-seal",
            str(seal),
            "--tokenizer-path",
            str(tokenizer),
            "--policies",
            str(policies),
            "--mizar-semantic-index",
            str(index),
            "--mml-contract-root",
            str(contract),
            *ledger_args,
            *assignments,
        ]
    )

    assert result == 2
    assert "exactly one Metamath drop ledger" in capsys.readouterr().err
