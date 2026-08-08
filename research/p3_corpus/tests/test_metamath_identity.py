"""Regression coverage for cross-database Metamath fact identity."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import build_metamath_shard as builder
from scripts.mm_expand import Incomplete

PINNED_MM_DIR = Path(os.environ.get("P3_PINNED_METAMATH_DIR", "/tmp/p3-audit-mm"))


def _mini_database(database: str, conflict_statement: str) -> str:
    return f"""
$c |- A B C $.
shared $a |- C $.
conflict $a {conflict_statement} $.
{database}-shared-th $p |- C $= ( shared ) A $.
{database}-conflict-th $p {conflict_statement} $= ( conflict ) A $.
"""


def _write_mini_databases(tmp_path: Path) -> Path:
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    sources = {
        "set": _mini_database("set", "|- A"),
        "iset": _mini_database("iset", "|- B"),
        "nf": _mini_database("nf", "|- A"),
    }
    for database, source in sources.items():
        (mm_dir / f"{database}.mm").write_text(source, encoding="utf-8")
    return mm_dir


def _run_mini_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    heldout: int,
    seed: int = 1,
) -> tuple[list[dict], list[dict], dict]:
    mm_dir = _write_mini_databases(tmp_path)
    source_hashes = {
        database: hashlib.sha256((mm_dir / f"{database}.mm").read_bytes()).hexdigest()
        for database in builder.DBS
    }
    expected_conflicts = {
        "conflict": {
            "set": "|- A",
            "iset": "|- B",
            "nf": "|- A",
        }
    }
    monkeypatch.setattr(builder, "SOURCE_SHA256", source_hashes)
    monkeypatch.setattr(builder, "EXPECTED_CONFLICT_COUNT", 1, raising=False)
    monkeypatch.setattr(
        builder,
        "EXPECTED_CONFLICT_MAP_SHA256",
        builder.canonical_sha256(expected_conflicts),
        raising=False,
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_metamath_shard.py",
            "--mm-dir",
            str(mm_dir),
            "--out",
            str(output),
            "--heldout",
            str(heldout),
            "--seed",
            str(seed),
        ],
    )

    builder.main()

    train = [
        json.loads(line)
        for line in (output / "shards" / "metamath.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in (output / "eval" / "metamath.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = json.loads(
        (output / "heldout" / "metamath.json").read_text(encoding="utf-8")
    )
    return train, evaluation, manifest


def _run_single_database_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_text: str,
    *,
    heldout: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    mm_dir = tmp_path / "single-mm"
    mm_dir.mkdir()
    source = mm_dir / "set.mm"
    source.write_text(source_text, encoding="utf-8")
    monkeypatch.setattr(builder, "DBS", ("set",))
    monkeypatch.setattr(
        builder,
        "SOURCE_SHA256",
        {"set": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    output = tmp_path / "single-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_metamath_shard.py",
            "--mm-dir",
            str(mm_dir),
            "--out",
            str(output),
            "--heldout",
            str(heldout),
            "--seed",
            str(seed),
        ],
    )

    builder.main()

    train = [
        json.loads(line)
        for line in (output / "shards" / "metamath.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    evaluation = [
        json.loads(line)
        for line in (output / "eval" / "metamath.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifest = json.loads(
        (output / "heldout" / "metamath.json").read_text(encoding="utf-8")
    )
    return train, evaluation, manifest


def test_conflict_map_and_fact_identity_qualify_only_conflicts() -> None:
    statements = {
        "set": {"19.2": "set statement", "syl": "shared statement"},
        "iset": {"19.2": "iset statement", "syl": "shared statement"},
        "nf": {"19.2": "set statement", "syl": "shared statement"},
    }

    conflicts = builder.compute_conflict_map(statements)

    assert conflicts == {
        "19.2": {
            "set": "set statement",
            "iset": "iset statement",
            "nf": "set statement",
        }
    }
    assert builder.fact_identity("set", "19.2", conflicts) == "set:19.2"
    assert builder.fact_identity("iset", "19.2", conflicts) == "iset:19.2"
    assert builder.fact_identity("nf", "19.2", conflicts) == "nf:19.2"
    assert builder.fact_identity("set", "syl", conflicts) == "syl"
    assert builder.fact_identity("iset", "syl", conflicts) == "syl"


def test_builder_renders_each_database_statement_and_only_assertion_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, evaluation, _ = _run_mini_build(
        tmp_path,
        monkeypatch,
        heldout=0,
    )
    rows = train + evaluation
    conflict_rows = {
        row["theorem"].split(":", 1)[0]: row
        for row in rows
        if row["theorem"].endswith("-conflict-th")
    }

    assert set(conflict_rows) == {"set", "iset", "nf"}
    for database, expected_statement in {
        "set": "|- A",
        "iset": "|- B",
        "nf": "|- A",
    }.items():
        identity = f"{database}:conflict"
        row = conflict_rows[database]
        assert row["facts"] == {identity: expected_statement}
        assert row["cited"] == [identity]
        assert identity in row["target"]
        assert row["target"].endswith(expected_statement)

    conflict_map = {"conflict": {"set": "|- A", "iset": "|- B"}}
    assert (
        builder.render_trace_label(
            "iset",
            "conflict",
            assertion_labels={"conflict"},
            conflict_map=conflict_map,
        )
        == "iset:conflict"
    )
    assert (
        builder.render_trace_label(
            "iset",
            "th.1",
            assertion_labels={"conflict"},
            conflict_map=conflict_map,
        )
        == "th.1"
    )
    assert (
        builder.render_trace_label(
            "iset",
            "(reuse)",
            assertion_labels={"conflict"},
            conflict_map=conflict_map,
        )
        == "(reuse)"
    )


def test_heldout_class_frequency_pools_aliases_and_separates_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eligible = ["|- A", "|- B", "|- C"]
    seed = next(
        candidate
        for candidate in range(100)
        if random.Random(candidate).sample(eligible, 1) == ["|- A"]
    )

    train, evaluation, manifest = _run_mini_build(
        tmp_path,
        monkeypatch,
        heldout=1,
        seed=seed,
    )

    assert manifest["facts"] == ["nf:conflict"]
    assert manifest["statement_aliases"] == [
        "nf-conflict-th",
        "set-conflict-th",
        "set:conflict",
    ]
    assert all("shared" not in identity for identity in manifest["facts"])
    assert any(row["cited"] == ["shared"] for row in train)
    assert {
        row["theorem"].split(":", 1)[0]
        for row in evaluation
        if row["cited"][0].endswith(":conflict")
    } == {"set", "nf"}


COMMON_ALIAS_MM = """
$c |- A B = $.
a1i $a |- A = A $.
alias-1 $a |- A = A $.
alias-2 $a |- A = A $.
rare-a1i $a |- A = A $.
other $a |- B $.
common-1 $p |- A = A $= ( a1i ) A $.
common-2 $p |- A = A $= ( alias-1 ) A $.
common-3 $p |- A = A $= ( alias-2 ) A $.
rare-use $p |- A = A $= ( rare-a1i ) A $.
other-use $p |- B $= ( other ) A $.
"""


class _ConditionalLengthTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> _LengthEncoding:
        assert add_special_tokens is False
        return _LengthEncoding(16_384 if "\nGOAL |- B\n" in text else 10)


def test_main_filters_before_selection_and_binds_exact_drop_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        builder,
        "load_fixed_qwen_tokenizer",
        lambda _path: (
            _ConditionalLengthTokenizer(),
            builder.FIXED_QWEN_TOKENIZER_SEAL,
        ),
    )

    train, evaluation, manifest = _run_single_database_build(
        tmp_path,
        monkeypatch,
        COMMON_ALIAS_MM,
        heldout=0,
        seed=20260801,
    )
    ledger = json.loads(
        (tmp_path / "single-output" / "drops" / "metamath-overlength.json").read_text(
            encoding="utf-8"
        )
    )

    builder.validate_drop_ledger(ledger)
    assert [entry["theorem"] for entry in ledger["entries"]] == ["set:other-use"]
    assert ledger["entries"][0]["text_plus_eos_tokens"] == 16_385
    assert len(train) == 4
    assert evaluation == []
    assert all(row["theorem"] != "set:other-use" for row in train)
    assert manifest["selected_statement_classes"] == []
    assert manifest["partition_accounting"] == {
        "source_rows": 5,
        "source_text_plus_eos_tokens": 16_429,
        "train_rows": 4,
        "train_text_plus_eos_tokens": 44,
        "eval_rows": 0,
        "eval_text_plus_eos_tokens": 0,
        "drop_rows": 1,
        "drop_text_plus_eos_tokens": 16_385,
    }
    manifest_body = dict(manifest)
    manifest_root = manifest_body.pop("manifest_root_sha256")
    assert manifest_root == builder.canonical_sha256(manifest_body)
    assert {
        row["source_metadata"]["drop_ledger"]["canonical_root_sha256"]
        for row in train
    } == {ledger["canonical_root_sha256"]}


def test_common_statement_class_is_ineligible_despite_rare_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_tail = ["a1i", "alias-1", "alias-2", "other", "rare-a1i"]
    seed = next(
        candidate
        for candidate in range(100)
        if random.Random(candidate).sample(legacy_tail, 1) == ["rare-a1i"]
    )

    train, evaluation, manifest = _run_single_database_build(
        tmp_path,
        monkeypatch,
        COMMON_ALIAS_MM,
        heldout=1,
        seed=seed,
    )

    assert manifest["schema_version"] == "metamath-heldout-v2"
    assert manifest["facts"] == ["other"]
    assert manifest["statement_aliases"] == ["other-use"]
    assert {row["theorem"] for row in evaluation} == {"set:other-use"}
    assert {row["theorem"] for row in train} == {
        "set:common-1",
        "set:common-2",
        "set:common-3",
        "set:rare-use",
    }

    from scripts import build_p3_generation as generation

    rows = train + evaluation
    occurrences = []
    for line_number, row in enumerate(rows, 1):
        raw = (json.dumps(row) + "\n").encode()
        occurrences.append(
            generation._LineOccurrence(
                line_number=line_number,
                byte_start=0,
                byte_end=len(raw),
                raw_bytes=raw,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                record=row,
            )
        )
    context = generation._metamath_isolation_context(
        tuple(occurrences),
        manifest["facts"],
    )
    assert {
        row["theorem"]: generation._classify_metamath_route(row, context).disposition
        for row in train
    } == {row["theorem"]: "train" for row in train}
    assert generation._classify_metamath_route(evaluation[0], context).disposition == (
        "drop"
    )


NAME_ONLY_ALIAS_MM = """
$c |- H C $.
base $a |- C $.
${
  held.1 $e |- H $.
  held $a |- C $.
$}
${
  alias.1 $e |- H $.
  alias $p |- C $= ( base ) B $.
$}
${
  use.1 $e |- H $.
  use $p |- C $= ( held ) AB $.
$}
"""


def test_selection_excludes_name_only_class_the_v2_classifier_cannot_represent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    eligible = ["|- C", "|- H => |- C"]
    seed = next(
        candidate
        for candidate in range(100)
        if random.Random(candidate).sample(eligible, 1) == ["|- H => |- C"]
    )
    train, evaluation, manifest = _run_single_database_build(
        tmp_path,
        monkeypatch,
        NAME_ONLY_ALIAS_MM,
        heldout=1,
        seed=seed,
    )

    assert not train
    assert manifest["facts"] == ["base"]
    assert manifest["statement_aliases"] == []
    assert {row["theorem"] for row in evaluation} == {"set:alias", "set:use"}

    from scripts import build_p3_generation as generation

    rows = []
    for line_number, row in enumerate(evaluation, 1):
        raw = (json.dumps(row) + "\n").encode()
        rows.append(
            generation._LineOccurrence(
                line_number=line_number,
                byte_start=0,
                byte_end=len(raw),
                raw_bytes=raw,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                record=row,
            )
        )
    context = generation._metamath_isolation_context(tuple(rows), manifest["facts"])
    assert all(
        generation._classify_metamath_route(row, context).disposition == "drop"
        for row in evaluation
    )


def test_selects_exactly_500_tail_statement_classes_deterministically() -> None:
    identity_statements = {}
    named_fact_identities = set()
    exposure_counts = Counter()
    for index in range(620):
        statement = f"|- class {index:03}"
        primary = f"fact-{index:03}"
        alias = f"alias-{index:03}"
        identity_statements[primary] = statement
        identity_statements[alias] = f"  {statement}  "
        named_fact_identities.add(primary)
        exposure_counts[statement] = 1 + index % 2 if index < 600 else 3
    identity_statements["expression-only"] = "|- expression only"
    exposure_counts["|- expression only"] = 1

    identities_by_class = builder.statement_identities_by_class(identity_statements)
    first = builder.select_heldout_statement_classes(
        exposure_counts,
        identities_by_class,
        named_fact_identities=named_fact_identities,
        requested=500,
        seed=20260801,
    )
    second = builder.select_heldout_statement_classes(
        exposure_counts,
        identities_by_class,
        named_fact_identities=named_fact_identities,
        requested=500,
        seed=20260801,
    )
    other_seed = builder.select_heldout_statement_classes(
        exposure_counts,
        identities_by_class,
        named_fact_identities=named_fact_identities,
        requested=500,
        seed=20260802,
    )

    assert first == second
    assert first["selected_classes"] != other_seed["selected_classes"]
    assert len(first["tail_classes"]) == 600
    assert len(first["selected_classes"]) == 500
    assert len(first["representatives"]) == 500
    assert len(first["statement_aliases"]) == 500
    assert all(
        exposure_counts[statement_class] in (1, 2)
        for statement_class in first["selected_classes"]
    )
    assert "|- expression only" not in first["tail_classes"]
    assert not any(
        exposure_counts[statement_class] > 2
        for statement_class in first["selected_classes"]
    )


class _LengthEncoding:
    def __init__(self, length: int) -> None:
        self.ids = [0] * length


class _LengthTokenizer:
    def __init__(self, lengths: dict[str, int]) -> None:
        self.lengths = lengths

    def encode(self, text: str, *, add_special_tokens: bool) -> _LengthEncoding:
        assert add_special_tokens is False
        return _LengthEncoding(self.lengths[text])


def _length_record(row_id: str, text: str, statement_class: str) -> tuple[dict, frozenset]:
    return (
        {
            "schema_version": builder.ROW_SCHEMA,
            "id": row_id,
            "theorem": f"set:{row_id}",
            "text": text,
            "source_metadata": {"excluded": "from-native-row-identity"},
        },
        frozenset({statement_class}),
    )


def test_text_plus_eos_boundary_filters_whole_rows_without_truncation() -> None:
    exact = _length_record("exact", "exact-text", "|- exact")
    over = _length_record("over", "over-text", "|- over")
    original_exact = deepcopy(exact[0])
    original_over = deepcopy(over[0])
    tokenizer = _LengthTokenizer({"exact-text": 16_383, "over-text": 16_384})

    eligible, ledger = builder.partition_records_by_text_plus_eos(
        [exact, over],
        tokenizer=tokenizer,
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )

    assert eligible == [(exact[0], exact[1], 16_384)]
    assert exact[0] == original_exact
    assert over[0] == original_over
    assert all(item[0]["id"] != "over" for item in eligible)
    assert ledger["schema_version"] == "metamath-overlength-drop-ledger-v1"
    assert ledger["entries"] == [
        {
            "schema_version": "metamath-overlength-drop-v1",
            "id": "over",
            "theorem": "set:over",
            "text_plus_eos_tokens": 16_385,
            "native_row_sha256": builder.native_row_sha256(over[0]),
            "reason_schema_version": "metamath-row-eligibility-reason-v1",
            "reason": "text_plus_eos_exceeds_maximum",
        }
    ]
    assert ledger["accounting"] == {
        "source_rows": 2,
        "eligible_rows": 1,
        "dropped_rows": 1,
        "source_text_plus_eos_tokens": 32_769,
        "eligible_text_plus_eos_tokens": 16_384,
        "dropped_text_plus_eos_tokens": 16_385,
        "dropped_excess_tokens": 1,
    }
    assert ledger["entries_root_sha256"] == builder.canonical_sha256(
        ledger["entries"]
    )
    root = ledger["canonical_root_sha256"]
    body = dict(ledger)
    del body["canonical_root_sha256"]
    assert root == builder.canonical_sha256(body)
    builder.validate_drop_ledger(ledger)


def test_drop_ledger_is_sorted_deterministic_and_mutation_evident() -> None:
    records = [
        _length_record("z-drop", "z-text", "|- z"),
        _length_record("kept", "kept-text", "|- kept"),
        _length_record("a-drop", "a-text", "|- a"),
    ]
    tokenizer = _LengthTokenizer(
        {"z-text": 20_000, "kept-text": 3, "a-text": 17_000}
    )
    first_eligible, first = builder.partition_records_by_text_plus_eos(
        records,
        tokenizer=tokenizer,
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )
    second_eligible, second = builder.partition_records_by_text_plus_eos(
        list(reversed(records)),
        tokenizer=tokenizer,
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )

    assert [entry["id"] for entry in first["entries"]] == ["a-drop", "z-drop"]
    assert first == second
    assert first_eligible[0][0]["id"] == second_eligible[0][0]["id"] == "kept"

    changed_identity = deepcopy(first)
    changed_identity["entries"][0]["id"] = "substituted"
    with pytest.raises(ValueError, match="entries root|canonical root"):
        builder.validate_drop_ledger(changed_identity)

    aggregate_only = deepcopy(first)
    aggregate_only["entries"] = []
    with pytest.raises(ValueError, match="entries|accounting|root"):
        builder.validate_drop_ledger(aggregate_only)

    path_substitution = deepcopy(first)
    path_substitution["canonical_root_sha256"] = "drops/metamath-overlength.json"
    with pytest.raises(ValueError, match="canonical root"):
        builder.validate_drop_ledger(path_substitution)


def test_pinned_source_oversized_population_gate_recomputes_identity_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _length_record("audited-drop", "audited-text", "|- audited")
    _, ledger = builder.partition_records_by_text_plus_eos(
        [record],
        tokenizer=_LengthTokenizer({"audited-text": 16_384}),
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )
    manifest = {
        "repository": "https://github.com/metamath/set.mm",
        "commit": builder.SOURCE_COMMIT,
        "files": {
            "set.mm": {
                "sha256": (
                    "7695d59e1c5c9182231e002425c82c86569bc044f30770bb32c276f7bafbf644"
                )
            },
            "iset.mm": {
                "sha256": (
                    "2851ed617e011b08b4d61c8312f34183aaf4da6b06b19512dac1e397ce709e4f"
                )
            },
            "nf.mm": {
                "sha256": (
                    "727a3707545e13ec53f03502eb07dc4635a8c176f275d4014a17fbd823e66083"
                )
            },
        },
    }
    expected_id_root = hashlib.sha256(b"audited-drop").hexdigest()
    monkeypatch.setattr(builder, "EXPECTED_PINNED_OVERSIZED_ROWS", 1)
    monkeypatch.setattr(
        builder,
        "EXPECTED_PINNED_OVERSIZED_IDS_SHA256",
        expected_id_root,
    )
    monkeypatch.setattr(builder, "EXPECTED_PINNED_OVERSIZED_TOKENS", 16_385)

    audit = builder.verify_pinned_oversized_population(manifest, ledger)

    assert audit == {
        "schema_version": "metamath-pinned-overlength-reproduction-v1",
        "source_rows": 1,
        "text_plus_eos_tokens": 16_385,
        "sorted_id_set_sha256": expected_id_root,
    }

    mutated = deepcopy(ledger)
    mutated["entries"][0]["id"] = "different-drop"
    mutated["entries_root_sha256"] = builder.canonical_sha256(mutated["entries"])
    mutated_body = dict(mutated)
    del mutated_body["canonical_root_sha256"]
    mutated["canonical_root_sha256"] = builder.canonical_sha256(mutated_body)
    with pytest.raises(RuntimeError, match="pinned oversized population"):
        builder.verify_pinned_oversized_population(manifest, mutated)


def test_overlength_filter_precedes_tail_exposure_and_keeps_500_represented() -> None:
    records = []
    identity_statements = {}
    named_fact_identities = set()
    lengths = {}
    for index in range(500):
        statement = f"|- eligible {index:03}"
        identity = f"eligible-{index:03}"
        text = f"text-{index:03}"
        records.append(_length_record(identity, text, statement))
        identity_statements[identity] = statement
        named_fact_identities.add(identity)
        lengths[text] = 1
    dropped_identity = "dropped-tail"
    dropped_statement = "|- dropped tail"
    records.append(
        _length_record(dropped_identity, "overlength-text", dropped_statement)
    )
    identity_statements[dropped_identity] = dropped_statement
    named_fact_identities.add(dropped_identity)
    lengths["overlength-text"] = 16_384
    for index in range(3):
        records.append(
            _length_record(
                f"common-{index}",
                f"common-text-{index}",
                "|- common",
            )
        )
        lengths[f"common-text-{index}"] = 1
    identity_statements["common"] = "|- common"
    named_fact_identities.add("common")

    eligible, ledger = builder.partition_records_by_text_plus_eos(
        records,
        tokenizer=_LengthTokenizer(lengths),
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )
    counts = builder.statement_class_exposure_counts(
        statement_classes for _, statement_classes, _ in eligible
    )
    selection = builder.select_heldout_statement_classes(
        counts,
        builder.statement_identities_by_class(identity_statements),
        named_fact_identities=named_fact_identities,
        requested=500,
        seed=20260801,
    )
    held_classes = frozenset(selection["selected_classes"])
    evaluation = [
        record
        for record, statement_classes, _ in eligible
        if statement_classes & held_classes
    ]
    train = [
        (record, statement_classes)
        for record, statement_classes, _ in eligible
        if not statement_classes & held_classes
    ]
    represented = set().union(
        *(
            statement_classes & held_classes
            for _, statement_classes, _ in eligible
            if statement_classes & held_classes
        )
    )

    assert ledger["accounting"]["dropped_rows"] == 1
    assert ledger["entries"][0]["id"] == dropped_identity
    assert dropped_statement not in counts
    assert dropped_statement not in held_classes
    assert len(held_classes) == 500
    assert len(evaluation) == 500
    assert represented == set(held_classes)
    assert all(not statement_classes & held_classes for _, statement_classes in train)


def test_drop_root_is_bound_into_source_quality_and_schema_roots() -> None:
    record = _length_record("drop", "drop-text", "|- drop")
    _, ledger = builder.partition_records_by_text_plus_eos(
        [record],
        tokenizer=_LengthTokenizer({"drop-text": 16_384}),
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )
    manifest = {
        "files": {
            "set.mm": {"sha256": "1" * 64},
        }
    }
    conflict_map = {}
    first = builder.build_source_metadata(
        manifest,
        conflict_map,
        drop_ledger=ledger,
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )

    assert first["drop_ledger"] == {
        "schema_version": ledger["schema_version"],
        "canonical_root_sha256": ledger["canonical_root_sha256"],
        "entries_root_sha256": ledger["entries_root_sha256"],
        "accounting": ledger["accounting"],
    }
    assert first["quality_filter"]["drop_ledger_root_sha256"] == (
        ledger["canonical_root_sha256"]
    )
    assert first["schema_generation"]["drop_ledger_root_sha256"] == (
        ledger["canonical_root_sha256"]
    )
    assert first["quality_filter_root_sha256"] == builder.canonical_sha256(
        first["quality_filter"]
    )
    assert first["schema_generation_root_sha256"] == builder.canonical_sha256(
        first["schema_generation"]
    )

    changed = deepcopy(ledger)
    changed["entries"][0]["id"] = "same-aggregate-different-row"
    changed["entries_root_sha256"] = builder.canonical_sha256(changed["entries"])
    changed_body = dict(changed)
    del changed_body["canonical_root_sha256"]
    changed["canonical_root_sha256"] = builder.canonical_sha256(changed_body)
    second = builder.build_source_metadata(
        manifest,
        conflict_map,
        drop_ledger=changed,
        tokenizer_seal=builder.FIXED_QWEN_TOKENIZER_SEAL,
    )

    assert first["drop_ledger"]["accounting"] == second["drop_ledger"]["accounting"]
    assert first["quality_filter_root_sha256"] != second["quality_filter_root_sha256"]
    assert (
        first["schema_generation_root_sha256"]
        != second["schema_generation_root_sha256"]
    )


def test_visible_statement_classes_count_each_class_once_per_row() -> None:
    identity_statements = {
        "held": "|- held",
        "held-alias": "  |-   held  ",
        "held-theorem": "|- held",
        "safe": "|- safe",
        "near": "|- held suffix",
    }
    record = {
        "facts": {
            "held-alias": "  |-   held  ",
            "safe": "|- safe",
        },
        "cited": ["held-alias", "safe"],
        "local_assumptions": {
            "local.1": " |- held ",
            "local.near": "|- held suffix",
        },
        "goal": "  |- held ",
        "target": "  1  held-alias    |- held\n  2  safe   |- safe",
    }

    visible = builder.visible_statement_classes(
        record,
        identity_statements,
        theorem_identity="held-theorem",
    )
    near_only = builder.visible_statement_classes(
        {
            "facts": {"near": "|- held suffix"},
            "cited": ["near"],
            "local_assumptions": {},
            "goal": "|- held suffix",
            "target": "  1  near  |- held suffix",
        },
        identity_statements,
        theorem_identity="near",
    )
    counts = builder.statement_class_exposure_counts(
        [visible, visible, near_only],
    )

    assert "|- held" in visible
    assert "|- held" not in near_only
    assert counts["|- held"] == 2
    assert counts["|- held suffix"] == 3


def test_statement_alias_helper_returns_rendered_identities_for_later_integration() -> (
    None
):
    statements = {
        "set": {"19.2": "classical", "syl": "shared"},
        "iset": {"19.2": "intuitionistic", "syl": "shared"},
        "nf": {"19.2": "classical", "syl": "shared"},
    }
    conflicts = builder.compute_conflict_map(statements)
    rendered = builder.rendered_fact_statements(statements, conflicts)

    assert rendered == {
        "set:19.2": "classical",
        "iset:19.2": "intuitionistic",
        "nf:19.2": "classical",
        "syl": "shared",
    }
    assert builder.statement_aliases_for({"set:19.2"}, rendered) == {"nf:19.2"}


def test_statement_exposure_helper_scans_fact_values_and_target_expressions() -> None:
    held_statements = builder.normalized_statements({"  |-   held  "})
    assert held_statements == frozenset({"|- held"})
    record = {
        "facts": {
            "safe": "|- safe",
            "renamed-held": "  |-   held  ",
        },
        "goal": "  |- held ",
        "local_assumptions": {"local.1": " |- held "},
        "target": (
            "  1  safe           |- safe\n"
            "  2  unrelated     |- held\n"
            "  3  other          |- other"
        ),
    }

    exposure = builder.statement_alias_exposure(record, held_statements)

    assert exposure == {
        "fact_identities": ("renamed-held",),
        "goal_expressions": ("|- held",),
        "local_assumption_values": ("|- held",),
        "target_expressions": ("|- held",),
    }
    assert builder.statement_alias_exposure(
        {
            "facts": {"safe": "|- safe"},
            "goal": "|- safe",
            "local_assumptions": {"local.near": "|- held suffix"},
            "target": "  1  safe  |- safe",
        },
        held_statements,
    ) == {
        "fact_identities": (),
        "goal_expressions": (),
        "local_assumption_values": (),
        "target_expressions": (),
    }


@pytest.mark.parametrize("theorem", ("set:rpnnen1lem6", "set:rpnnen1"))
def test_reported_qq_local_assumption_is_a_full_statement_exposure(
    theorem: str,
) -> None:
    held_statements = builder.normalized_statements({"|- QQ e. _V"})
    exposure = builder.statement_alias_exposure(
        {
            "theorem": theorem,
            "facts": {"safe": "|- safe"},
            "goal": "|- safe",
            "local_assumptions": {f"{theorem}.q": "  |-   QQ e. _V  "},
            "target": "  1  safe  |- safe",
        },
        held_statements,
    )

    assert exposure["local_assumption_values"] == ("|- QQ e. _V",)


@pytest.mark.skipif(
    not all((PINNED_MM_DIR / f"{database}.mm").exists() for database in builder.DBS),
    reason="pinned set/iset/nf sources are unavailable",
)
def test_real_pinned_conflicts_and_legacy_affected_proof_counts() -> None:
    databases, statements, conflicts = builder.load_pinned_databases(PINNED_MM_DIR)

    assert len(conflicts) == 394
    assert builder.canonical_sha256(conflicts) == (
        "900adee09e42be5d7dda266a61e3095991825e9cf2612675e5589af932edc0c2"
    )
    assert conflicts["19.2"] == {
        "set": "|- ( A. x ph -> E. x ph )",
        "iset": "|- ( A. x ph -> E. y ph )",
        "nf": "|- ( A. x ph -> E. x ph )",
    }
    assert conflicts["2a1i"] == {
        "set": "|- ph => |- ( ps -> ( ch -> ph ) )",
        "iset": "|- ch => |- ( ph -> ( ps -> ch ) )",
        "nf": "|- ch => |- ( ph -> ( ps -> ch ) )",
    }
    assert all(
        builder.fact_identity(database, label, conflicts) == f"{database}:{label}"
        for label, per_database in conflicts.items()
        for database in per_database
    )
    assert all(
        builder.fact_identity(database, "syl", conflicts) == "syl"
        for database in builder.DBS
    )

    legacy_statements: dict[str, str] = {}
    for database in builder.DBS:
        for label, statement in statements[database].items():
            if label not in legacy_statements:
                legacy_statements[label] = statement

    affected = Counter()
    for database, mm in databases.items():
        logical = set(statements[database])
        for label, (kind, _) in mm.labels.items():
            if kind != "$p":
                continue
            try:
                expression, mandatory, references, trace = builder.expand(mm, label)
            except Incomplete:
                continue
            logical_trace = [
                (step_label, " ".join(step_expression))
                for step_label, step_expression, _ in trace
                if step_expression and step_expression[0] == "|-"
            ]
            if not logical_trace or logical_trace[-1][1] != " ".join(expression):
                continue
            _, steps = builder.split_model_trace(mandatory, trace)
            if not steps:
                continue
            used = [
                reference
                for reference in dict.fromkeys(references)
                if reference in logical
            ]
            if not used:
                continue
            if any(
                legacy_statements[reference] != statements[database][reference]
                for reference in used
            ):
                affected[database] += 1

    assert affected == {"iset": 771, "nf": 1_084}
