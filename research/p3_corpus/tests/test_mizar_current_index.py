"""Contract tests for the isolated current-MML semantic index."""

import hashlib
import importlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

build_thproofs_shard = importlib.import_module("build_thproofs_shard")

from mizar_current_index import (
    FLAT_TREE_HASH_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    DuplicateIdentityError,
    MalformedSemanticHtml,
    MizarIndex,
    SourceVerificationError,
    _iter_miz_theorem_goals,
    build_index,
    hash_flat_tree,
    parse_html_article,
    parse_thproof_file,
    summarize_thproofs,
    theorem_identity,
    verify_source_manifest,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mizar_current" / "sample.html"
CURRENT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "manifests"
    / "mizar-8.1.15_5.94.1493.json"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_manifest_and_builder_pin_verified_completion_denominators():
    manifest = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["expected"]["thproof_files"] == 76_696
    assert manifest["expected"]["thproof_join_count"] == 76_696
    assert manifest["expected"]["explicit_proof_bearing_extracts"] == 69_698
    assert manifest["expected"]["complete_explicit_proofs"] == 58_658
    assert manifest["expected"]["sample_candidate_count"] == 240
    assert manifest["expected"]["sample_thproof_join_count"] == 239
    assert manifest["expected"]["sample_source_goal_match_count"] == 239
    assert manifest["expected"]["sample_generated_identity_count"] == 1
    assert manifest["expected"]["sample_agreement_count"] == 240
    assert manifest["expected"]["sample_mismatch_count"] == 0
    assert build_thproofs_shard.HARD_MIN_EXPLICIT_PROOFS == 65_000
    assert build_thproofs_shard.HARD_MIN_COMPLETE_PROOFS == 55_000
    assert build_thproofs_shard.HARD_MIN_COMPLETION_RATE == 0.80
    assert build_thproofs_shard.HARD_MIN_ACCEPTED_PROOFS == 45_000
    assert build_thproofs_shard.HARD_MIN_ACCEPTED_RATE == 0.80


def _write_sources(tmp_path):
    html = tmp_path / "html"
    mml = tmp_path / "mml"
    thproofs = tmp_path / "thproofs"
    html.mkdir()
    mml.mkdir()
    thproofs.mkdir()
    shutil.copyfile(FIXTURE, html / "sample.html")
    (mml / "sample.miz").write_text(
        """\
reserve x for set;

theorem Th1:
  x = x & y <= z by SAMPLE:2;
""",
        encoding="utf-8",
    )
    (thproofs / "t1_sample").write_text(
        """\
reserve x for set;
theorem Th1:
  x = x & y <= z
proof
  thus thesis by SAMPLE:2;
end;
""",
        encoding="utf-8",
    )
    # SAMPLE:2 is deliberately absent as a literal theorem in sample.miz. The
    # semantic HTML identity remains authoritative, as it is for generated T
    # identities in the real release.
    (thproofs / "t2_sample").write_text(
        "theorem for q being set holds q = q by SAMPLE:1;\n",
        encoding="utf-8",
    )
    return {"html": html, "mml": mml, "thproofs": thproofs}


def _write_manifest(tmp_path, roots, *, expected=None):
    sources = {}
    specs = {
        "mml": ("*.miz", "0" * 64),
        "html": ("*.html", "1" * 64),
        "thproofs": ("*", "2" * 64),
    }
    for name, (file_glob, archive_sha256) in specs.items():
        tree = hash_flat_tree(roots[name], file_glob=file_glob)
        sources[name] = {
            "archive_url": f"https://invalid.example/{name}.tar",
            "archive_sha256": archive_sha256,
            "file_glob": file_glob,
            "file_count": tree.file_count,
            "tree_sha256": tree.sha256,
        }
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "release": {
            "mizar_version": "8.1.15",
            "mml_version": "5.94.1493-test",
        },
        "tree_hash": {
            "schema": FLAT_TREE_HASH_SCHEMA,
            "description": "basename UTF-8 + NUL + raw SHA-256(file), sorted",
        },
        "sources": sources,
        "expected": expected or {},
        "proof_policy": {
            "completion_denominator": "explicit_proof_bearing_extracts",
            "minimum_explicit_completion_rate": 0.5,
        },
        "licensing": {
            "redistribution_rights_asserted": False,
            "status": "uncertain; legal review required",
        },
    }
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _run_current_thproof_builder(
    monkeypatch,
    *,
    roots,
    manifest,
    semantic_index,
    output,
    html2=None,
):
    for name in (
        "HARD_MIN_SOURCE_FILES",
        "HARD_MIN_NAME_MATCHES",
        "HARD_MIN_EXPLICIT_PROOFS",
        "HARD_MIN_COMPLETE_PROOFS",
        "HARD_MIN_ACCEPTED_PROOFS",
    ):
        monkeypatch.setattr(build_thproofs_shard, name, 1)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_NAME_JOIN_RATE", 0.5)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_COMPLETION_RATE", 0.5)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_ACCEPTED_RATE", 0.5)
    argv = [
        "build_thproofs_shard.py",
        "--src",
        str(roots["thproofs"]),
        "--semantic-index",
        str(semantic_index),
        "--source-manifest",
        str(manifest),
        "--mml-root",
        str(roots["mml"]),
        "--html-root",
        str(roots["html"]),
        "--exclude",
        "",
        "--out",
        str(output),
        "--heldout",
        "0",
    ]
    if html2 is not None:
        argv.extend(["--html2", str(html2)])
    monkeypatch.setattr(sys, "argv", argv)
    return build_thproofs_shard.main()


def test_streaming_parser_preserves_expanded_semantics_and_local_labels():
    records = {record.identity: record for record in parse_html_article(FIXTURE)}

    assert set(records) == {
        "SAMPLE:1",
        "SAMPLE:2",
        "SAMPLE:def_1",
        "SAMPLE:sch_1",
    }
    theorem = records["SAMPLE:1"]
    assert theorem.kind == "theorem"
    assert theorem.local_label == "Th1"
    assert theorem.statement == "for x being set holds x = x & y <= z"
    assert "source comment" not in theorem.statement
    assert "proof payload" not in theorem.statement
    assert "&amp;" in theorem.statement_html
    assert "<font" in theorem.statement_html
    assert theorem.provenance.html_file == "sample.html"
    assert theorem.provenance.html_anchor == "T1"
    assert theorem.provenance.identity_text == "SAMPLE:1"
    assert theorem.provenance.html_line > 0

    definition = records["SAMPLE:def_1"]
    assert definition.kind == "definition"
    assert definition.local_label == "DefThing"
    assert definition.statement.startswith("for x being set holds")
    assert "&" in definition.statement

    scheme = records["SAMPLE:sch_1"]
    assert scheme.kind == "scheme"
    assert scheme.local_label == "SampleScheme"
    assert "SampleScheme" in scheme.statement
    assert "F1()" in scheme.statement
    assert "provided A1:" in scheme.statement
    assert "ignored" not in scheme.statement


def test_parser_rejects_duplicate_ids_and_unclosed_semantic_records(tmp_path):
    duplicate = tmp_path / "dup.html"
    duplicate.write_text(
        """\
<div about="#T1" typeof="oo:Theorem"><a name="T1">:: DUP:1</a>
<div class="add">P</div></div>
<div about="#T1" typeof="oo:Theorem"><a name="T1">:: DUP:1</a>
<div class="add">Q</div></div>
""",
        encoding="utf-8",
    )
    with pytest.raises(DuplicateIdentityError, match="DUP:1"):
        list(parse_html_article(duplicate))

    malformed = tmp_path / "malformed.html"
    malformed.write_text(
        """\
<div about="#T1" typeof="oo:Theorem"><a name="T1">:: BAD:1</a>
<div class="add">P</div>
""",
        encoding="utf-8",
    )
    with pytest.raises(MalformedSemanticHtml, match="T1"):
        list(parse_html_article(malformed))


def test_proof_categories_use_only_explicit_proof_bearing_denominator(tmp_path):
    chunks = {
        "t1_sample": """\
theorem P
proof
  thus thesis;
end;
""",
        "t2_sample": "theorem Q by SAMPLE:1;\n",
        "t3_sample": "theorem R proof thus thesis;\n",
        "t4_sample": "theorem canceled;\n",
        "t5_sample": "theorem S by Lm6\n",
        "not_a_thproof": "reserve x for set;\n",
    }
    records = []
    for name, text in chunks.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        records.append(parse_thproof_file(path))

    categories = {record.file_name: record.category for record in records}
    assert categories == {
        "not_a_thproof": "invalid_name",
        "t1_sample": "complete_explicit_proof",
        "t2_sample": "inline_justification",
        "t3_sample": "malformed_explicit_proof",
        "t4_sample": "canceled",
        "t5_sample": "malformed_declaration",
    }
    assert next(
        record for record in records if record.file_name == "t5_sample"
    ).source_goal == "S"
    metrics = summarize_thproofs(records)
    assert metrics["complete_explicit_proofs"] == 1
    assert metrics["explicit_proof_bearing_extracts"] == 2
    assert metrics["explicit_completion_rate"] == 0.5
    assert metrics["all_file_completion_rate"] == pytest.approx(1 / 6)


def test_miz_goal_scanner_stops_before_a_later_local_lemma_proof(tmp_path):
    source = tmp_path / "sample.miz"
    source.write_text(
        """\
theorem Th1:
  P by EXT:1;

Lm1: Q
proof
  thus thesis;
end;

theorem Th2:
  R
proof
  thus thesis;
end;
""",
        encoding="utf-8",
    )

    assert list(_iter_miz_theorem_goals(source)) == ["P", "R"]


def test_manifest_verification_detects_hash_drift_and_tree_order_is_stable(tmp_path):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)

    verified = verify_source_manifest(manifest, roots)
    assert verified["release"]["mizar_version"] == "8.1.15"
    before = hash_flat_tree(roots["thproofs"], file_glob="*")

    (roots["thproofs"] / "t1_sample").write_text("changed\n", encoding="utf-8")
    with pytest.raises(SourceVerificationError, match="thproofs"):
        verify_source_manifest(manifest, roots)

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b").write_bytes(b"two")
    (first / "a").write_bytes(b"one")
    (second / "a").write_bytes(b"one")
    (second / "b").write_bytes(b"two")
    assert hash_flat_tree(first).sha256 == hash_flat_tree(second).sha256
    assert hash_flat_tree(roots["thproofs"]).sha256 != before.sha256


def test_index_rejects_manifest_change_during_build(tmp_path, monkeypatch):
    import mizar_current_index

    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)
    original_align = mizar_current_index._align_mml

    def align_then_drift(connection, mml_root):
        result = original_align(connection, mml_root)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["release"]["mml_version"] = "changed-mid-build"
        manifest.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(mizar_current_index, "_align_mml", align_then_drift)
    with pytest.raises(SourceVerificationError, match="changed during build"):
        build_index(
            manifest_path=manifest,
            roots=roots,
            sqlite_path=tmp_path / "index.sqlite",
            jsonl_path=tmp_path / "index.jsonl",
        )
    assert not (tmp_path / "index.sqlite").exists()
    assert not (tmp_path / "index.jsonl").exists()


def test_index_is_byte_deterministic_and_exposes_builder_apis(tmp_path):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)
    outputs = []
    reports = []
    for ordinal in (1, 2):
        sqlite_path = tmp_path / f"index-{ordinal}.sqlite"
        jsonl_path = tmp_path / f"index-{ordinal}.jsonl"
        reports.append(
            build_index(
                manifest_path=manifest,
                roots=roots,
                sqlite_path=sqlite_path,
                jsonl_path=jsonl_path,
            )
        )
        outputs.append((sqlite_path, jsonl_path))

    assert _sha256(outputs[0][0]) == _sha256(outputs[1][0])
    assert _sha256(outputs[0][1]) == _sha256(outputs[1][1])
    assert reports[0]["content"] == reports[1]["content"]
    assert reports[0]["content"]["statement_count"] == 4
    assert reports[0]["content"]["thproof_join_count"] == 2
    assert reports[0]["content"]["mml_alignment"] == {
        "generated_or_unmatched": 1,
        "literal_goal_match": 1,
    }

    with MizarIndex(outputs[0][0]) as index:
        statements = index.statement_map()
        local = index.article_local_label_maps()
        assert statements["SAMPLE:1"] == (
            "for x being set holds x = x & y <= z"
        )
        assert local["SAMPLE"] == {
            "DefThing": ("SAMPLE:def_1",),
            "SampleScheme": ("SAMPLE:sch_1",),
            "Th1": ("SAMPLE:1",),
        }
        assert index.theorem_identity("t1_sample") == "SAMPLE:1"
        assert index.source_goal("SAMPLE:1") == "x = x & y <= z"
        assert index.source_goal("SAMPLE:def_1") is None
        assert index.metadata()["schema_version"]

    assert theorem_identity("/some/path/t36_partpr_1") == "PARTPR_1:36"


def test_sample_crosscheck_accounts_for_nonliteral_generated_identity(tmp_path):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["crosscheck"] = {
        "sample_articles": ["SAMPLE"],
        "generated_identities": ["SAMPLE:2"],
    }
    manifest.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=tmp_path / "index.sqlite",
        jsonl_path=tmp_path / "index.jsonl",
    )

    assert report["content"]["sample_candidate_count"] == 2
    assert report["content"]["sample_thproof_join_count"] == 2
    assert report["content"]["sample_source_goal_match_count"] == 1
    assert report["content"]["sample_generated_identity_count"] == 1
    assert report["content"]["sample_agreement_count"] == 2
    assert report["content"]["sample_mismatch_count"] == 0


def test_active_thproof_builder_requires_verified_semantic_index(
    tmp_path, monkeypatch
):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)
    sqlite_path = tmp_path / "index.sqlite"
    build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=sqlite_path,
        jsonl_path=tmp_path / "index.jsonl",
    )
    output = tmp_path / "output"

    assert _run_current_thproof_builder(
        monkeypatch,
        roots=roots,
        manifest=manifest,
        semantic_index=sqlite_path,
        output=output,
    ) == 0
    record = json.loads(
        (output / "shards" / "thproofs.jsonl").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == "mizar-proof-v2"
    assert record["goal"] == "for x being set holds x = x & y <= z"
    assert record["source_metadata"]["schema_version"] == (
        "mizar-thproof-build-source-v1"
    )
    assert record["source_metadata"]["index_roots"] == {
        "semantic_index_schema": "mizar-semantic-index-v1",
        "semantic_index_sha256": _sha256(sqlite_path),
    }
    assert record["source_metadata"]["semantic_index_sha256"] == _sha256(
        sqlite_path
    )
    assert record["source_metadata"]["source_manifest_sha256"] == _sha256(
        manifest
    )
    assert record["source_metadata"]["source_trees"] == (
        MizarIndex(sqlite_path).metadata()["source_trees"]
    )


def test_active_builder_rejects_source_index_drift_and_wrong_association(
    tmp_path, monkeypatch
):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(tmp_path, roots)
    sqlite_path = tmp_path / "index.sqlite"
    build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=sqlite_path,
        jsonl_path=tmp_path / "index.jsonl",
    )

    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "UPDATE thproofs SET file_name = ? WHERE identity = ?",
            ("t99_wrong", "SAMPLE:1"),
        )
    assert _run_current_thproof_builder(
        monkeypatch,
        roots=roots,
        manifest=manifest,
        semantic_index=sqlite_path,
        output=tmp_path / "wrong-association",
    ) != 0

    drift_root = tmp_path / "drift"
    drift_root.mkdir()
    roots = _write_sources(drift_root)
    manifest = _write_manifest(drift_root, roots)
    sqlite_path = tmp_path / "drift" / "index.sqlite"
    build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=sqlite_path,
        jsonl_path=tmp_path / "drift" / "index.jsonl",
    )
    (roots["mml"] / "sample.miz").write_text("changed\n", encoding="utf-8")
    assert _run_current_thproof_builder(
        monkeypatch,
        roots=roots,
        manifest=manifest,
        semantic_index=sqlite_path,
        output=tmp_path / "drift-output",
    ) != 0


def test_active_builder_records_numeric_provenance_as_diagnostic(
    tmp_path, monkeypatch
):
    roots = _write_sources(tmp_path)
    proof = roots["thproofs"] / "t1_sample"
    proof.write_text(
        proof.read_text(encoding="utf-8").replace(
            "theorem Th1:",
            "theorem Th1: :: OTHER:2",
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(tmp_path, roots)
    sqlite_path = tmp_path / "index.sqlite"
    build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=sqlite_path,
        jsonl_path=tmp_path / "index.jsonl",
    )
    output = tmp_path / "output"

    assert _run_current_thproof_builder(
        monkeypatch,
        roots=roots,
        manifest=manifest,
        semantic_index=sqlite_path,
        output=output,
    ) == 0
    record = json.loads(
        (output / "shards" / "thproofs.jsonl").read_text(encoding="utf-8")
    )
    assert record["source_diagnostics"]["numeric_provenance"] == "mismatch"
    assert record["goal"] == "for x being set holds x = x & y <= z"


def test_legacy_html2_cannot_produce_active_thproof_output(tmp_path, monkeypatch):
    roots = _write_sources(tmp_path)
    output = tmp_path / "output"
    stale = output / "shards" / "thproofs.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_thproofs_shard.py",
            "--src",
            str(roots["thproofs"]),
            "--html2",
            str(roots["html"]),
            "--out",
            str(output),
        ],
    )

    assert build_thproofs_shard.main() != 0
    assert not stale.exists()


def test_reused_local_labels_keep_order_and_resolve_at_theorem_scope(tmp_path):
    roots = _write_sources(tmp_path)
    article = roots["html"] / "sample.html"
    text = article.read_text(encoding="utf-8")
    text = text.replace(
        "</body>",
        """\
<div about="#T3" typeof="oo:Theorem">
<span class="kw">theorem </span><span class="lab">Th1</span>:
<a name="T3"><span class="comment">:: SAMPLE:3</span><br></a>
<div class="add">for z being set holds z = z;</div>
</div>
<div about="#T4" typeof="oo:Theorem">
<span class="kw">theorem </span>
<a name="T4"><span class="comment">:: SAMPLE:4</span><br></a>
<div class="add">for w being set holds w = w;</div>
</div>
</body>
""",
    )
    article.write_text(text, encoding="utf-8")
    manifest = _write_manifest(tmp_path, roots)
    sqlite_path = tmp_path / "index.sqlite"
    build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=sqlite_path,
        jsonl_path=tmp_path / "index.jsonl",
    )

    with MizarIndex(sqlite_path) as index:
        labels = index.article_local_label_maps()
        assert labels["SAMPLE"]["Th1"] == ("SAMPLE:1", "SAMPLE:3")
        assert index.resolve_local_label(
            "SAMPLE", "Th1", at_identity="SAMPLE:2"
        ) == "SAMPLE:1"
        assert index.resolve_local_label(
            "SAMPLE", "Th1", at_identity="SAMPLE:3"
        ) == "SAMPLE:1"
        assert index.resolve_local_label(
            "SAMPLE", "Th1", at_identity="SAMPLE:4"
        ) == "SAMPLE:3"
        statements = index.statement_map()
        refs, missing = build_thproofs_shard.resolve_index_references(
            "thus thesis by Th1;",
            index,
            statements,
            theorem="SAMPLE:4",
        )
        assert refs == ["SAMPLE:3"]
        assert missing == []


def test_non_utf8_miz_comments_use_reported_lossless_fallback(tmp_path):
    roots = _write_sources(tmp_path)
    mml = roots["mml"] / "sample.miz"
    mml.write_bytes(mml.read_bytes() + b"\n:: legacy byte \xa7\n")
    manifest = _write_manifest(tmp_path, roots)

    report = build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=tmp_path / "index.sqlite",
        jsonl_path=tmp_path / "index.jsonl",
    )

    assert report["content"]["mml_non_utf8_files"] == 1
    assert report["content"]["mml_alignment"]["literal_goal_match"] == 1


def test_expected_counts_are_hard_gates_but_generated_identities_are_not(tmp_path):
    roots = _write_sources(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        roots,
        expected={
            "html_article_files": 1,
            "statement_count": 4,
            "theorem_count": 2,
            "thproof_files": 2,
            "thproof_join_count": 2,
            "duplicate_identities": 0,
            "missing_thproof_identities": 0,
        },
    )
    report = build_index(
        manifest_path=manifest,
        roots=roots,
        sqlite_path=tmp_path / "index.sqlite",
        jsonl_path=tmp_path / "index.jsonl",
    )

    assert report["content"]["mml_alignment"]["generated_or_unmatched"] == 1

    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["expected"]["thproof_join_count"] = 3
    manifest.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceVerificationError, match="thproof_join_count"):
        build_index(
            manifest_path=manifest,
            roots=roots,
            sqlite_path=tmp_path / "bad.sqlite",
            jsonl_path=tmp_path / "bad.jsonl",
        )
