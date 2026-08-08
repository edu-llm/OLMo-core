"""Focused tests for the deep corpus schema verifier."""

import copy
import hashlib
import importlib
import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

split_heldout = importlib.import_module("split_heldout")
verify_corpus = importlib.import_module("verify_corpus")
generation = importlib.import_module("build_p3_generation")
direct_mizar = importlib.import_module("build_mizar_human_shard")

REPO_ROOT = Path(__file__).resolve().parents[1]
DIRECT_MIZAR_REPLAY_ROOT_ENV = "P3_DIRECT_MIZAR_REPLAY_ROOT"
DIRECT_MIZAR_SOURCE_ROOT_ENV = "P3_DIRECT_MIZAR_SOURCE_ROOT"
DIRECT_MIZAR_APPROVED = {
    "rows": 55_353,
    "tokens": 42_851_393,
    "token_sequence_sha256": (
        "ea246e12e76ec67827c91bd919bc6271abf39ff05906d4812454b1255b44ea01"
    ),
    "raw_sha256": "54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f",
    "manifest_sha256": (
        "dfda6cfb3815f8032044450b0d8378b1da8efd2ec0e793e05add13159ea7f551"
    ),
    "manifest_root_sha256": (
        "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"
    ),
    "quality_filter_root_sha256": (
        "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d"
    ),
    "schema_generation_root_sha256": (
        "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242"
    ),
    "semantic_index_sha256": (
        "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835"
    ),
    "mml_tree_sha256": (
        "3d1af5b3e840aca5631541b42510b35c1b15dfa988af70ce463f58c899e88714"
    ),
    "fact_frequencies_sha256": (
        "d214c7e60492a2664fa9b96e83c304ba1fe62fec1ce8ad621cbd1fd9a2b3e8e0"
    ),
    "primary_rows": 50_114,
    "primary_bytes": 353_408_320,
    "primary_sha256": (
        "0d563be6fae81cd21b551c422378792ef4daad454cab9ed86bb751f127daefd1"
    ),
    "recovery": {
        "rows": 5_239,
        "tokens_with_eos": 9_946_266,
        "identity_source_order_sha256": (
            "6d113a43ff0b0af8aae13325908d2507b9b63aadcc01d50c37d73e29549396fa"
        ),
        "identity_set_sha256": (
            "048f47cf87e6eaeccf87f3aafb202236373dea000719ae221c5ee33896dad8cd"
        ),
        "source_binding_sha256": (
            "790c86db30604c5836be70e28df527bb6c1a41b30620cfaf327122db047be65c"
        ),
        "token_sequence_sha256": (
            "3391cb491f1e7e8ec23b7725d27ceb95b4d5d51bd5856a8e46372507102d5ca4"
        ),
        "text_hash_sequence_sha256": (
            "e116af514ee5cc7fc3415d01a68ec42037206849f3b62e6bdddbfefe4637659f"
        ),
    },
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joined_sha256(values):
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def direct_mizar_real_artifact():
    artifact_root = (
        Path(
            os.environ.get(
                DIRECT_MIZAR_REPLAY_ROOT_ENV,
                REPO_ROOT / ".p3-work" / "full13" / "mizar",
            )
        )
        .expanduser()
        .resolve()
    )
    source_root = (
        Path(
            os.environ.get(
                DIRECT_MIZAR_SOURCE_ROOT_ENV,
                artifact_root.parents[1] / "sources",
            )
        )
        .expanduser()
        .resolve()
    )
    paths = {
        "root": artifact_root,
        "raw": artifact_root / "raw" / "mizar.jsonl",
        "manifest": artifact_root / "manifests" / "mizar.json",
        "report": artifact_root / "reports" / "mizar.build.json",
        "frequencies": artifact_root / "reports" / "mizar.fact_frequencies.json",
        "checksums": artifact_root / "checksums" / "mizar.json",
        "index": source_root / "mizar-current-8.1.15-final2.sqlite",
        "mml": source_root / "p3-source-audit" / "extract-mizar" / "mml",
    }
    missing = [
        name for name, path in paths.items() if name != "root" and not path.exists()
    ]
    if missing:
        pytest.skip(
            "optional direct-Mizar real artifact is unavailable "
            f"({', '.join(missing)} missing); set {DIRECT_MIZAR_REPLAY_ROOT_ENV} "
            f"and {DIRECT_MIZAR_SOURCE_ROOT_ENV}"
        )
    return paths


class _ResolverDispatchIndex:
    def __init__(self):
        self._statements = {
            "EXT:1": "external theorem",
            "TARSKI:def_3": "external definition",
            "ORDINAL1:sch_1": "external scheme",
            "SAMPLE:2": "article-level Lm2",
        }

    def statement_map(self):
        return dict(self._statements)

    def article_local_label_maps(self):
        return {"SAMPLE": {"Lm2": ("SAMPLE:2",)}}

    def resolve_local_label(self, article, label, *, at_identity):
        if (article, label, at_identity) == ("SAMPLE", "Lm2", "SAMPLE:20"):
            return "SAMPLE:2"
        raise KeyError(label)


def _mizar_record(
    *,
    facts,
    cited,
    goal,
    target,
    theorem="SAMPLE:1",
    source_metadata=None,
):
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    record = {
        "id": "mizar-example",
        "theorem": theorem,
        "facts": facts,
        "cited": cited,
        "goal": goal,
        "target": target,
        "text": f"{block}\n---\nGOAL {goal}\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
    }
    if source_metadata is not None:
        record["source_metadata"] = source_metadata
    return record


def _write_single_mizar_corpus(tmp_path, record):
    corpus = tmp_path / "corpus"
    for name in ("raw", "shards", "eval", "heldout"):
        (corpus / name).mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    (corpus / "raw" / "mizar.jsonl").write_text(line)
    (corpus / "shards" / "mizar.jsonl").write_text(line)
    (corpus / "eval" / "mizar.jsonl").write_text("")
    (corpus / "heldout" / "mizar.json").write_text(
        json.dumps({"facts": [], "shards": ["mizar"]})
    )
    return corpus


def _run_verifier(monkeypatch, corpus, *extra):
    del monkeypatch
    return verify_corpus.legacy_audit(
        ["--corpus", str(corpus), *extra],
    )


def _write_semantic_index(tmp_path):
    path = tmp_path / "semantic.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE statements (
                identity TEXT PRIMARY KEY,
                article TEXT,
                kind TEXT,
                number INTEGER,
                local_label TEXT,
                statement TEXT,
                statement_html TEXT,
                statement_sha256 TEXT,
                html_file TEXT,
                html_anchor TEXT,
                html_line INTEGER,
                identity_text TEXT
            );
            CREATE TABLE local_labels (
                article TEXT,
                label TEXT,
                identity TEXT
            );
            """)
        statements = {
            "SAMPLE:1": "for x being set holds x = x",
            "SAMPLE:2": "for y being set holds y = y",
        }
        for number, (identity, statement) in enumerate(statements.items(), 1):
            connection.execute(
                "INSERT INTO statements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity,
                    "SAMPLE",
                    "theorem",
                    number,
                    None,
                    statement,
                    statement,
                    hashlib.sha256(statement.encode()).hexdigest(),
                    "sample.html",
                    f"T{number}",
                    number,
                    identity,
                ),
            )
        metadata = {
            "schema_version": "mizar-semantic-index-v1",
            "source_manifest_sha256": "a" * 64,
            "source_trees": {
                "html": {"file_count": 1, "tree_sha256": "b" * 64},
                "mml": {"file_count": 1, "tree_sha256": "c" * 64},
                "thproofs": {"file_count": 2, "tree_sha256": "d" * 64},
            },
            "release": {"mizar_version": "8.1.15", "mml_version": "5.94.1493"},
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )
    source_metadata = {
        "schema_version": "mizar-thproof-build-source-v1",
        "semantic_index_schema": "mizar-semantic-index-v1",
        "semantic_index_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_manifest_schema": "mizar-current-sources-v1",
        "source_manifest_sha256": "a" * 64,
        "release": metadata["release"],
        "source_roots": {
            "html": "/verified/html",
            "mml": "/verified/mml",
            "thproofs": "/verified/thproofs",
        },
        "source_trees": metadata["source_trees"],
        "source_archives": {},
        "licensing": {"redistribution_rights_asserted": False},
    }
    return path, source_metadata


def _write_secondary_mizar_fixture(tmp_path):
    mml_root = tmp_path / "mml-secondary"
    mml_root.mkdir()
    source_text = """\
theorem Left: LeftGoal proof thus thesis by EXT:1; end;
theorem Recovered: RecoveryGoal proof thus thesis by EXT:1; end;
theorem Right: RightGoal proof thus thesis by EXT:1; end;
"""
    source_path = mml_root / "sample.miz"
    source_path.write_text(source_text, encoding="utf-8")
    declarations = direct_mizar.parse_miz_article(
        source_text,
        article="SAMPLE",
        source_file=source_path.name,
    )
    proof_sha256 = hashlib.sha256(b"thus thesis by EXT:1;").hexdigest()

    index_path = tmp_path / "secondary.sqlite"
    with sqlite3.connect(index_path) as connection:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE statements (
                identity TEXT PRIMARY KEY,
                article TEXT,
                kind TEXT,
                number INTEGER,
                local_label TEXT,
                statement TEXT,
                statement_html TEXT,
                statement_sha256 TEXT,
                html_file TEXT,
                html_anchor TEXT,
                html_line INTEGER,
                identity_text TEXT
            );
            CREATE TABLE local_labels (
                article TEXT,
                label TEXT,
                identity TEXT
            );
            CREATE TABLE thproofs (
                identity TEXT PRIMARY KEY,
                source_goal TEXT,
                mml_alignment TEXT,
                category TEXT,
                proof_sha256 TEXT
            );
            """)
        statement_rows = (
            ("SAMPLE:1", "SAMPLE", 1, "Left", "LeftGoal", 10),
            ("SAMPLE:2", "SAMPLE", 2, "Recovered", "RecoveryGoal", 20),
            ("SAMPLE:3", "SAMPLE", 3, "Right", "RightGoal", 30),
            ("EXT:1", "EXT", 1, None, "External fact", 1),
        )
        for identity, article, number, label, statement, html_line in statement_rows:
            connection.execute(
                "INSERT INTO statements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    identity,
                    article,
                    "theorem",
                    number,
                    label,
                    statement,
                    statement,
                    hashlib.sha256(statement.encode()).hexdigest(),
                    f"{article.lower()}.html",
                    f"T{number}",
                    html_line,
                    identity,
                ),
            )
            if label is not None:
                connection.execute(
                    "INSERT INTO local_labels VALUES (?, ?, ?)",
                    (article, label, identity),
                )
        connection.executemany(
            "INSERT INTO thproofs VALUES (?, ?, ?, ?, ?)",
            (
                (
                    "SAMPLE:1",
                    "LeftGoal",
                    "literal_goal_match",
                    "complete_explicit_proof",
                    proof_sha256,
                ),
                (
                    "SAMPLE:2",
                    "RecoveryGoal",
                    "literal_goal_match",
                    "malformed_explicit_proof",
                    None,
                ),
                (
                    "SAMPLE:3",
                    "RightGoal",
                    "literal_goal_match",
                    "complete_explicit_proof",
                    proof_sha256,
                ),
            ),
        )
        metadata = {
            "schema_version": "mizar-semantic-index-v1",
            "source_manifest_sha256": "a" * 64,
            "source_trees": {
                "html": {"file_count": 1, "tree_sha256": "b" * 64},
                "mml": {"file_count": 1, "tree_sha256": "c" * 64},
                "thproofs": {"file_count": 3, "tree_sha256": "d" * 64},
            },
            "release": {
                "mizar_version": "8.1.15",
                "mml_version": "5.94.1493",
            },
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            [(key, json.dumps(value)) for key, value in metadata.items()],
        )

    manifest = generation._make_source_manifest("mizar", test_only=True)
    manifest["row_source_metadata"]["index_roots"]["semantic_index_sha256"] = (
        hashlib.sha256(index_path.read_bytes()).hexdigest()
    )
    manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)
    manifest["row_source_metadata"]["source_manifest_root_sha256"] = manifest[
        "manifest_root_sha256"
    ]

    with importlib.import_module("mizar_current_index").MizarIndex(index_path) as index:
        anchors = direct_mizar._load_anchors(index, "SAMPLE")
        primary, unanchored = direct_mizar._strict_complete_alignment(
            declarations,
            anchors,
        )
        assert unanchored == 1
        recovered = direct_mizar._secondary_unique_label_alignment(
            declarations,
            anchors,
            primary,
        )
        assert len(recovered) == 1
        resolution = direct_mizar.resolve_global_citations(
            recovered[0].source.target,
            index,
            theorem=recovered[0].identity,
        )
        facts = {"EXT:1": index.statement_map()["EXT:1"]}
    text, mask_start, mask_end = direct_mizar.render_training_text(
        facts,
        recovered[0].anchor.statement,
        recovered[0].source.target,
    )
    row = direct_mizar._record(
        recovered[0],
        source_file_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        source_encoding="utf-8",
        resolution=resolution,
        facts=facts,
        text=text,
        mask_start=mask_start,
        mask_end=mask_end,
        token_length=len(text.encode("utf-8")) + 1,
        source_metadata=manifest["row_source_metadata"],
        seed=20260801,
    )
    return index_path, mml_root, manifest, row


def test_current_mizar_v2_verifier_replays_source_index_and_references(tmp_path):
    index_path, _ = _write_semantic_index(tmp_path)
    mml_root = tmp_path / "mml"
    mml_root.mkdir()
    source_text = """\
theorem Direct:
  for x being set holds x = x
proof
  thus thesis by SAMPLE:2;
end;
"""
    source_path = mml_root / "sample.miz"
    source_path.write_text(source_text, encoding="utf-8")
    declaration = direct_mizar.parse_miz_article(
        source_text,
        article="SAMPLE",
        source_file=source_path.name,
    )[0]
    manifest = generation._make_source_manifest("mizar", test_only=True)
    manifest["row_source_metadata"]["index_roots"]["semantic_index_sha256"] = (
        hashlib.sha256(index_path.read_bytes()).hexdigest()
    )
    manifest["manifest_root_sha256"] = generation._source_manifest_root(manifest)
    manifest["row_source_metadata"]["source_manifest_root_sha256"] = manifest[
        "manifest_root_sha256"
    ]
    facts = {"SAMPLE:2": "for y being set holds y = y"}
    target = declaration.target
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    row = {
        "schema_version": "mizar-proof-v2",
        "id": hashlib.sha256(
            (
                f"mizar-proof-v2\0SAMPLE:1\0{source_path.name}\0"
                f"1\0{declaration.target_sha256}"
            ).encode()
        ).hexdigest(),
        "theorem": "SAMPLE:1",
        "facts": facts,
        "cited": ["SAMPLE:2"],
        "local_assumptions": {},
        "goal": "for x being set holds x = x",
        "target": target,
        "text": f"{block}\n---\nGOAL for x being set holds x = x\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
        "source": {
            "article": "SAMPLE",
            "file": source_path.name,
            "encoding": "utf-8",
            "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "declaration_ordinal": 1,
            "target_start": declaration.target_start,
            "target_end": declaration.target_end,
            "target_sha256": declaration.target_sha256,
        },
        "source_metadata": manifest["row_source_metadata"],
    }

    assert (
        verify_corpus.direct_mizar_record_errors(
            row,
            semantic_index=index_path,
            mml_root=mml_root,
            source_manifest=manifest,
        )
        == []
    )
    drifted = json.loads(json.dumps(row))
    drifted["source"]["target_sha256"] = "0" * 64
    assert any(
        kind == "mizar_source_target_mismatch"
        for kind, _ in verify_corpus.direct_mizar_record_errors(
            drifted,
            semantic_index=index_path,
            mml_root=mml_root,
            source_manifest=manifest,
        )
    )


def test_verifier_independently_replays_secondary_source_binding(tmp_path):
    index_path, mml_root, manifest, row = _write_secondary_mizar_fixture(tmp_path)

    assert (
        verify_corpus.direct_mizar_record_errors(
            row,
            semantic_index=index_path,
            mml_root=mml_root,
            source_manifest=manifest,
        )
        == []
    )

    missing = copy.deepcopy(row)
    missing.pop("source_index_binding")
    assert any(
        kind == "mizar_source_binding_missing"
        for kind, _ in verify_corpus.direct_mizar_record_errors(
            missing,
            semantic_index=index_path,
            mml_root=mml_root,
            source_manifest=manifest,
        )
    )

    mutated = copy.deepcopy(row)
    mutated["source_index_binding"]["previous_proof_hash_anchor"]["index_number"] = 99
    assert any(
        kind == "mizar_source_binding_mismatch"
        for kind, _ in verify_corpus.direct_mizar_record_errors(
            mutated,
            semantic_index=index_path,
            mml_root=mml_root,
            source_manifest=manifest,
        )
    )


def test_current_index_dispatch_rejects_thproof_resolver_for_direct_mizar():
    index = _ResolverDispatchIndex()
    target = """\
Lm2: P by EXT:1;
thus thesis by Lm2, TARSKI:def 3, ORDINAL1:sch 1;
"""

    references, unresolved = verify_corpus.resolve_current_index_references(
        "mizar",
        target,
        index,
        index.statement_map(),
        theorem="SAMPLE:20",
    )

    assert references == ["EXT:1", "TARSKI:def_3", "ORDINAL1:sch_1"]
    assert unresolved == []
    wrong_references, wrong_unresolved = importlib.import_module(
        "build_thproofs_shard"
    ).resolve_index_references(
        target,
        index,
        index.statement_map(),
        theorem="SAMPLE:20",
    )
    assert "SAMPLE:2" in wrong_references or "Lm2" in wrong_unresolved


def test_current_index_dispatch_keeps_thproofs_on_its_native_resolver(monkeypatch):
    calls = []

    def direct(*args, **kwargs):
        calls.append(("mizar", args, kwargs))
        raise AssertionError("direct resolver must not handle thproofs")

    def thproofs(*args, **kwargs):
        calls.append(("thproofs", args, kwargs))
        return ["SAMPLE:2"], []

    monkeypatch.setattr(direct_mizar, "resolve_global_citations", direct)
    monkeypatch.setattr(
        importlib.import_module("build_thproofs_shard"),
        "resolve_index_references",
        thproofs,
    )

    assert verify_corpus.resolve_current_index_references(
        "thproofs",
        "thus thesis by Lm2;",
        object(),
        {"SAMPLE:2": "statement"},
        theorem="SAMPLE:20",
    ) == (["SAMPLE:2"], [])
    assert [call[0] for call in calls] == ["thproofs"]


def test_production_dispatch_replays_approved_recovered_direct_mizar_candidate(
    direct_mizar_real_artifact,
):
    paths = direct_mizar_real_artifact
    raw_path = paths["raw"]
    index_path = paths["index"]
    mml_root = paths["mml"]
    mizar_index = importlib.import_module("mizar_current_index")
    approved = DIRECT_MIZAR_APPROVED
    manifest_bytes = paths["manifest"].read_bytes()
    manifest = generation._validate_source_manifest(
        json.loads(manifest_bytes),
        family="mizar",
        production=False,
    )
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    checksums = json.loads(paths["checksums"].read_text(encoding="utf-8"))["files"]
    expected_metadata = manifest["row_source_metadata"]

    assert hashlib.sha256(manifest_bytes).hexdigest() == approved["manifest_sha256"]
    assert manifest["manifest_root_sha256"] == approved["manifest_root_sha256"]
    assert (
        expected_metadata["source_manifest_root_sha256"]
        == approved["manifest_root_sha256"]
    )
    assert (
        expected_metadata["quality_filter_root_sha256"]
        == approved["quality_filter_root_sha256"]
    )
    assert (
        expected_metadata["schema_generation_root_sha256"]
        == approved["schema_generation_root_sha256"]
    )
    assert (
        expected_metadata["index_roots"]["semantic_index_sha256"]
        == approved["semantic_index_sha256"]
    )
    assert (
        expected_metadata["source_roots"]["mml"]["tree_sha256"]
        == approved["mml_tree_sha256"]
    )
    assert _sha256_file(index_path) == approved["semantic_index_sha256"]
    mml_tree = mizar_index.hash_flat_tree(mml_root, file_glob="*.miz")
    assert (mml_tree.file_count, mml_tree.sha256) == (
        1_500,
        approved["mml_tree_sha256"],
    )

    assert report["counters"]["accepted_rows"] == approved["rows"]
    assert report["counters"]["recovered_unique_label"] == approved["recovery"]["rows"]
    assert report["deep_self_check"] == {
        "index_rows_checked": approved["rows"],
        "reconstruction_rows_checked": approved["rows"],
        "reference_rows_checked": approved["rows"],
        "rows_checked": approved["rows"],
        "source_rows_checked": approved["rows"],
        "status": "clean",
    }
    assert report["output_hashes"]["raw_jsonl_sha256"] == approved["raw_sha256"]
    assert report["output_hashes"]["manifest_sha256"] == approved["manifest_sha256"]
    assert (
        report["output_hashes"]["fact_frequencies_sha256"]
        == approved["fact_frequencies_sha256"]
    )
    report_recovery = report["direct_mizar_recovery"]
    assert {key: report_recovery[key] for key in approved["recovery"]} == approved[
        "recovery"
    ]
    assert report_recovery["duplicate_checks"] == {
        "accepted_or_internal_text": "clean",
        "accepted_thproof_trajectory": "clean",
    }
    assert checksums["raw/mizar.jsonl"] == approved["raw_sha256"]
    assert checksums["manifests/mizar.json"] == approved["manifest_sha256"]
    assert (
        checksums["reports/mizar.fact_frequencies.json"]
        == approved["fact_frequencies_sha256"]
    )
    assert _sha256_file(paths["report"]) == checksums["reports/mizar.build.json"]

    raw_digest = hashlib.sha256()
    primary_digest = hashlib.sha256()
    primary_bytes = 0
    all_token_lengths = []
    recovery_identities = []
    recovery_text_hashes = []
    recovery_token_lengths = []
    recovery_source_bindings = []
    primary_ids = set()
    primary_identities = set()
    primary_texts = set()
    recovery_ids = set()
    recovery_identity_set = set()
    recovery_texts = set()
    seen_ids = set()
    seen_texts = set()
    fact_frequencies = Counter()
    source_hashes = {}
    current_source_file = None
    source_text = None
    declarations = None
    source_encoding = None
    rows = 0

    with mizar_index.MizarIndex(index_path) as index:
        statements = index.statement_map()
        with raw_path.open("rb") as source:
            for rows, line in enumerate(source, 1):
                assert line.endswith(b"\n"), f"row {rows} is not newline terminated"
                raw_digest.update(line)
                if rows <= approved["primary_rows"]:
                    primary_digest.update(line)
                    primary_bytes += len(line)
                record = json.loads(line)
                where = f"row {rows} ({record.get('theorem')})"
                assert record["schema_version"] == direct_mizar.ROW_SCHEMA, where
                assert record["family"] == "mizar", where
                assert record["split"] == "raw", where
                assert record["heldout"] == 0, where
                assert record["source_metadata"] == expected_metadata, where

                row_id = record["id"]
                text_hash = hashlib.sha256(record["text"].encode()).hexdigest()
                assert row_id not in seen_ids, f"{where}: duplicate id"
                assert text_hash not in seen_texts, f"{where}: duplicate text"
                seen_ids.add(row_id)
                seen_texts.add(text_hash)
                all_token_lengths.append(record["token_length_with_eos"])

                source_record = record["source"]
                source_file = source_record["file"]
                if source_file != current_source_file:
                    assert Path(source_file).name == source_file, where
                    source_path = mml_root / source_file
                    source_text, source_encoding = direct_mizar._read_miz(source_path)
                    declarations = direct_mizar.parse_miz_article(
                        source_text,
                        article=source_path.stem.upper(),
                        source_file=source_file,
                    )
                    current_source_file = source_file
                    index._direct_mizar_binding_cache = {}
                    source_hashes.setdefault(source_file, _sha256_file(source_path))

                ordinal = int(source_record["declaration_ordinal"])
                declaration = declarations[ordinal - 1]
                expected_source = {
                    "article": declaration.article,
                    "file": declaration.source_file,
                    "encoding": source_encoding,
                    "file_sha256": source_hashes[source_file],
                    "declaration_ordinal": declaration.ordinal,
                    "label": declaration.label,
                    "source_goal": declaration.source_goal,
                    "index_compatible_source_goal": declaration.index_source_goal,
                    "line_start": declaration.line_start,
                    "line_end": declaration.line_end,
                    "declaration_start": declaration.declaration_start,
                    "declaration_end": declaration.declaration_end,
                    "target_start": declaration.target_start,
                    "target_end": declaration.target_end,
                    "declaration_sha256": hashlib.sha256(
                        declaration.source_declaration.encode()
                    ).hexdigest(),
                    "target_sha256": declaration.target_sha256,
                }
                assert declaration.category == "complete_explicit_proof", where
                assert source_record == expected_source, f"{where}: source replay"
                assert declaration.target == record["target"], f"{where}: target"
                assert (
                    source_text[declaration.target_start : declaration.target_end]
                    == record["target"]
                ), f"{where}: target offsets"
                assert record["local_assumptions"] == declaration.local_assumptions, (
                    f"{where}: local assumptions"
                )
                proof_local_labels = list(
                    dict.fromkeys(
                        label
                        for _, label in direct_mizar._proof_local_labels(
                            record["target"]
                        )
                    )
                )
                assert record["proof_local_labels"] == proof_local_labels, (
                    f"{where}: proof-local labels"
                )

                anchor_row = index.connection.execute(
                    """
                    SELECT
                        s.number, s.statement, s.statement_sha256, s.local_label,
                        s.html_file, s.html_anchor, s.html_line,
                        t.source_goal, t.mml_alignment, t.category, t.proof_sha256
                    FROM statements AS s
                    LEFT JOIN thproofs AS t ON t.identity = s.identity
                    WHERE s.identity = ? AND s.kind = 'theorem'
                    """,
                    (record["theorem"],),
                ).fetchone()
                assert anchor_row is not None, f"{where}: missing index theorem"
                expected_index = {
                    "identity": record["theorem"],
                    "number": int(anchor_row[0]),
                    "local_label": anchor_row[3],
                    "source_goal": anchor_row[7],
                    "mml_alignment": anchor_row[8],
                    "proof_category": anchor_row[9],
                    "proof_sha256": anchor_row[10],
                    "statement_sha256": anchor_row[2],
                    "html_file": anchor_row[4],
                    "html_anchor": anchor_row[5],
                    "html_line": int(anchor_row[6]),
                }
                assert record["goal"] == anchor_row[1], f"{where}: goal replay"
                assert record["index"] == expected_index, f"{where}: index replay"

                references, unresolved = verify_corpus.resolve_current_index_references(
                    "mizar",
                    record["target"],
                    index,
                    statements,
                    theorem=record["theorem"],
                )
                assert unresolved == [], f"{where}: unresolved {unresolved[:3]}"
                assert references == record["cited"], f"{where}: citation replay"
                expected_fact_order = direct_mizar.deterministic_fact_order(
                    references,
                    row_key=record["theorem"],
                    seed=20260801,
                )
                assert list(record["facts"]) == expected_fact_order, (
                    f"{where}: fact order"
                )
                assert all(
                    statements.get(name) == statement
                    for name, statement in record["facts"].items()
                ), f"{where}: fact values"
                fact_frequencies.update(record["cited"])

                assert (
                    verify_corpus._direct_mizar_source_binding_errors(
                        record,
                        declaration=declaration,
                        declarations=declarations,
                        index=index,
                    )
                    == []
                ), f"{where}: source/index binding"
                rendered, mask_start, mask_end = direct_mizar.render_training_text(
                    record["facts"],
                    record["goal"],
                    record["target"],
                )
                assert record["text"] == rendered, f"{where}: rendering"
                assert record["mask_start"] == mask_start, f"{where}: mask start"
                assert record["mask_end"] == mask_end, f"{where}: mask end"
                assert record["mask"] == {
                    "schema_version": direct_mizar.MASK_SCHEMA,
                    "start": mask_start,
                    "end": mask_end,
                }, f"{where}: mask"
                expected_id = hashlib.sha256(
                    "\0".join(
                        (
                            direct_mizar.ROW_SCHEMA,
                            record["theorem"],
                            source_file,
                            str(ordinal),
                            declaration.target_sha256,
                        )
                    ).encode()
                ).hexdigest()
                assert row_id == expected_id, f"{where}: deterministic id"

                if rows <= approved["primary_rows"]:
                    assert "source_index_binding" not in record, where
                    primary_ids.add(row_id)
                    primary_identities.add(record["theorem"])
                    primary_texts.add(text_hash)
                else:
                    assert "source_index_binding" in record, where
                    assert row_id not in primary_ids, f"{where}: primary id overlap"
                    assert record["theorem"] not in primary_identities, (
                        f"{where}: primary identity overlap"
                    )
                    assert text_hash not in primary_texts, (
                        f"{where}: primary text overlap"
                    )
                    recovery_ids.add(row_id)
                    recovery_identity_set.add(record["theorem"])
                    recovery_texts.add(text_hash)
                    recovery_identities.append(record["theorem"])
                    recovery_text_hashes.append(text_hash)
                    recovery_token_lengths.append(record["token_length_with_eos"])
                    recovery_source_bindings.append(
                        "\0".join(
                            (
                                source_record["article"],
                                str(source_record["declaration_ordinal"]),
                                record["theorem"],
                                str(source_record["label"]),
                                source_record["target_sha256"],
                                text_hash,
                            )
                        )
                    )

    assert rows == approved["rows"]
    assert raw_digest.hexdigest() == approved["raw_sha256"]
    assert primary_bytes == approved["primary_bytes"]
    assert primary_digest.hexdigest() == approved["primary_sha256"]
    assert len(primary_ids) == approved["primary_rows"]
    assert len(recovery_ids) == approved["recovery"]["rows"]
    assert len(recovery_identity_set) == approved["recovery"]["rows"]
    assert len(recovery_texts) == approved["recovery"]["rows"]
    assert primary_ids.isdisjoint(recovery_ids)
    assert primary_identities.isdisjoint(recovery_identity_set)
    assert primary_texts.isdisjoint(recovery_texts)
    assert len(seen_ids) == len(seen_texts) == approved["rows"]
    assert sum(all_token_lengths) == approved["tokens"]
    assert _joined_sha256(all_token_lengths) == approved["token_sequence_sha256"]

    replayed_recovery = {
        "rows": len(recovery_identities),
        "tokens_with_eos": sum(recovery_token_lengths),
        "identity_source_order_sha256": _joined_sha256(recovery_identities),
        "identity_set_sha256": _joined_sha256(sorted(recovery_identities)),
        "source_binding_sha256": _joined_sha256(recovery_source_bindings),
        "token_sequence_sha256": _joined_sha256(recovery_token_lengths),
        "text_hash_sequence_sha256": _joined_sha256(recovery_text_hashes),
    }
    assert replayed_recovery == approved["recovery"]
    expected_frequencies = json.loads(paths["frequencies"].read_text(encoding="utf-8"))
    assert dict(sorted(fact_frequencies.items())) == expected_frequencies
    assert _sha256_file(paths["frequencies"]) == approved["fact_frequencies_sha256"]


def test_documented_recovered_production_contract():
    docs = (REPO_ROOT / "docs" / "mizar-human-shard.md").read_text(encoding="utf-8")
    assert "55,353 accepted raw rows, including 5,239" in docs
    assert DIRECT_MIZAR_APPROVED["raw_sha256"] in docs
    assert DIRECT_MIZAR_APPROVED["manifest_sha256"] in docs
    assert DIRECT_MIZAR_APPROVED["manifest_root_sha256"] in docs


def test_verifier_reconstructs_metamath_local_assumptions_inside_mask(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    for name in ("raw", "shards", "eval", "heldout"):
        (corpus / name).mkdir(parents=True, exist_ok=True)

    facts = {"ext": "|- ph => |- ph"}
    local_assumptions = {"th.1": "|- ph"}
    target = "  1  ext            |- ph"
    block = (
        "I know these mathematical statements:\n"
        "ext : |- ph => |- ph\n"
        "Local assumptions:\n"
        "th.1 : |- ph"
    )
    record = {
        "id": "example",
        "theorem": "set:th",
        "facts": facts,
        "cited": ["ext"],
        "local_assumptions": local_assumptions,
        "goal": "|- ph",
        "target": target,
        "text": f"{block}\n---\nGOAL |- ph\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
    }
    line = json.dumps(record) + "\n"
    (corpus / "raw" / "metamath.jsonl").write_text(line)
    (corpus / "shards" / "metamath.jsonl").write_text(line)
    (corpus / "eval" / "metamath.jsonl").write_text("")
    (corpus / "heldout" / "metamath.json").write_text(
        json.dumps({"facts": [], "shards": ["metamath"]})
    )

    del monkeypatch
    assert verify_corpus.legacy_audit(["--corpus", str(corpus)]) == 0


def test_verifier_rejects_canceled_mizar_facts(tmp_path, monkeypatch):
    record = _mizar_record(
        facts={"SAMPLE:1": "canceled;"},
        cited=["SAMPLE:1"],
        goal="for x being set holds x = x",
        target="thus thesis by SAMPLE:1;",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)

    assert _run_verifier(monkeypatch, corpus) == 1


def test_verifier_rejects_qualified_target_reference_missing_from_prompt(
    tmp_path, monkeypatch
):
    record = _mizar_record(
        facts={"SAMPLE:2": "for x being set holds x = x"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = x",
        target="thus thesis from ORDINAL1:sch_1(A1);",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)

    assert _run_verifier(monkeypatch, corpus) == 1


def test_verifier_uses_semantic_index_for_mizar_values_and_metadata(
    tmp_path, monkeypatch
):
    semantic_index, source_metadata = _write_semantic_index(tmp_path)
    record = _mizar_record(
        facts={"SAMPLE:2": "for y being set holds y = y"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = x",
        target="thus thesis by SAMPLE:2;",
        source_metadata=source_metadata,
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)
    args = ("--mizar-semantic-index", str(semantic_index))

    assert _run_verifier(monkeypatch, corpus, *args) == 0

    record["facts"]["SAMPLE:2"] = "corrupted statement"
    block = "I know these mathematical statements:\nSAMPLE:2 : corrupted statement"
    record["mask_end"] = len(block)
    record["text"] = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    corpus = _write_single_mizar_corpus(tmp_path, record)
    assert _run_verifier(monkeypatch, corpus, *args) == 1


def test_verifier_rejects_missing_or_drifted_mizar_source_metadata(
    tmp_path, monkeypatch
):
    semantic_index, source_metadata = _write_semantic_index(tmp_path)
    record = _mizar_record(
        facts={"SAMPLE:2": "for y being set holds y = y"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = x",
        target="thus thesis by SAMPLE:2;",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)
    args = ("--mizar-semantic-index", str(semantic_index))
    assert _run_verifier(monkeypatch, corpus, *args) == 1

    source_metadata["semantic_index_sha256"] = "0" * 64
    record["source_metadata"] = source_metadata
    corpus = _write_single_mizar_corpus(tmp_path, record)
    assert _run_verifier(monkeypatch, corpus, *args) == 1


def test_verifier_checks_mizar_gold_identity_against_html2_source(
    tmp_path, monkeypatch
):
    html2 = tmp_path / "html2"
    html2.mkdir()
    (html2 / "sample.xml1.txt").write_text("""\
theorem Th1: :: SAMPLE:1
for x being set holds x = x
proof
  thus thesis by SAMPLE:2;
end;
theorem :: SAMPLE:2
for x being set holds x = x by SAMPLE:1;
""")
    record = _mizar_record(
        facts={"SAMPLE:2": "for x being set holds x = x"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = {}",
        target="thus thesis by SAMPLE:2;",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)

    assert _run_verifier(monkeypatch, corpus, "--mizar-html2", str(html2)) == 1

    record["goal"] = "for x being set holds x = x"
    block = record["text"][: record["mask_end"]]
    record["text"] = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    corpus = _write_single_mizar_corpus(tmp_path, record)
    assert _run_verifier(monkeypatch, corpus, "--mizar-html2", str(html2)) == 0


def test_verifier_checks_every_mizar_fact_value_against_html2_source(
    tmp_path, monkeypatch
):
    html2 = tmp_path / "html2"
    html2.mkdir()
    (html2 / "sample.xml1.txt").write_text("""\
theorem Th1: :: SAMPLE:1
for x being set holds x = x
proof
  thus thesis by SAMPLE:2;
end;
theorem :: SAMPLE:2
for x being set holds x = x by SAMPLE:1;
""")
    record = _mizar_record(
        facts={"SAMPLE:2": "for x being set holds x = {}"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = x",
        target="thus thesis by SAMPLE:2;",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)

    assert _run_verifier(monkeypatch, corpus, "--mizar-html2", str(html2)) == 1

    record["facts"]["SAMPLE:2"] = "for x being set holds x = x"
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in record["facts"].items()
    )
    record["mask_end"] = len(block)
    record["text"] = f"{block}\n---\nGOAL {record['goal']}\n{record['target']}"
    corpus = _write_single_mizar_corpus(tmp_path, record)
    assert _run_verifier(monkeypatch, corpus, "--mizar-html2", str(html2)) == 0


def _write_metamath_holdout_corpus(tmp_path, *, local_assumptions, target_expression):
    corpus = tmp_path / "corpus"
    for name in ("raw", "shards", "eval", "heldout"):
        (corpus / name).mkdir(parents=True, exist_ok=True)

    facts = {"ext": "|- ps => |- ch"}
    target = f"  1  ext            {target_expression}"
    block = (
        "I know these mathematical statements:\n"
        "ext : |- ps => |- ch\n"
        "Local assumptions:"
    )
    if local_assumptions:
        block += "\n" + "\n".join(
            f"{name} : {statement}" for name, statement in local_assumptions.items()
        )
    record = {
        "id": "metamath-held-exposure",
        "theorem": "set:safe",
        "facts": facts,
        "cited": ["ext"],
        "local_assumptions": local_assumptions,
        "goal": "|- ch",
        "target": target,
        "text": f"{block}\n---\nGOAL |- ch\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
    }
    line = json.dumps(record) + "\n"
    (corpus / "raw" / "metamath.jsonl").write_text(line)
    (corpus / "shards" / "metamath.jsonl").write_text(line)
    (corpus / "eval" / "metamath.jsonl").write_text("")
    (corpus / "heldout" / "metamath.json").write_text(
        json.dumps(
            {
                "facts": ["held"],
                "family": "metamath",
                "shards": ["metamath"],
                "statement_hashes": [
                    split_heldout.statement_hash("|- ph", family="metamath")
                ],
                "canonicalization": {
                    "family": "metamath",
                    "scheme": "metamath-token-v2",
                    "version": 2,
                },
            }
        )
    )
    return corpus


def test_verifier_rejects_held_statement_in_metamath_local_assumption(
    tmp_path, monkeypatch
):
    corpus = _write_metamath_holdout_corpus(
        tmp_path,
        local_assumptions={"safe.1": "|- ph"},
        target_expression="|- ch",
    )

    assert _run_verifier(monkeypatch, corpus) == 1


def test_verifier_rejects_held_statement_in_metamath_target_expression(
    tmp_path, monkeypatch
):
    corpus = _write_metamath_holdout_corpus(
        tmp_path,
        local_assumptions={},
        target_expression="|- ph",
    )

    assert _run_verifier(monkeypatch, corpus) == 1


def test_default_mode_requires_transaction_current_and_rejects_legacy_tree(
    tmp_path,
    monkeypatch,
    capsys,
):
    corpus = tmp_path / "legacy"
    (corpus / "raw").mkdir(parents=True)
    (corpus / "raw" / "metamath.jsonl").write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        ["verify_corpus.py", "--corpus", str(corpus)],
    )

    assert verify_corpus.main() == 1
    output = capsys.readouterr()
    assert "CURRENT" in output.err
    assert "legacy" in output.err.lower()


def test_legacy_audit_is_explicit_read_only_and_never_production_clean(
    tmp_path,
    monkeypatch,
    capsys,
):
    record = _mizar_record(
        facts={"SAMPLE:2": "for y being set holds y = y"},
        cited=["SAMPLE:2"],
        goal="for x being set holds x = x",
        target="thus thesis by SAMPLE:2;",
    )
    corpus = _write_single_mizar_corpus(tmp_path, record)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_corpus.py",
            "--corpus",
            str(corpus),
            "--legacy-audit",
        ],
    )

    assert verify_corpus.main() == 2
    output = capsys.readouterr()
    assert "LEGACY AUDIT" in output.out
    assert "PRODUCTION VERIFY CLEAN" not in output.out


def test_production_mode_delegates_to_transaction_deep_verifier(
    tmp_path,
    monkeypatch,
    capsys,
):
    calls = []

    def fake_verify(root, *, production, mizar_semantic_index):
        calls.append((Path(root), production, Path(mizar_semantic_index)))
        return {
            "status": "clean",
            "generation_id": "production-v2",
            "logical_root_sha256": "a" * 64,
            "families": [
                "metamath",
                "mizar",
                "thproofs",
                "prf2",
                "enigma",
                "isabelle",
            ],
            "mml_selected_classes": 1_000,
            "modes": {},
        }

    index = tmp_path / "mizar.sqlite"
    index.write_bytes(b"index")
    transaction = tmp_path / "transaction"
    transaction.mkdir()
    (transaction / "CURRENT").write_text("{}\n")
    monkeypatch.setattr(
        "scripts.build_p3_generation.verify_generation",
        fake_verify,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_corpus.py",
            "--corpus",
            str(transaction),
            "--mizar-semantic-index",
            str(index),
        ],
    )

    assert verify_corpus.main() == 0
    assert calls == [(transaction, True, index)]
    assert "PRODUCTION VERIFY CLEAN" in capsys.readouterr().out


def test_legacy_html2_flag_is_forbidden_in_production_mode(tmp_path, monkeypatch):
    html2 = tmp_path / "html2"
    html2.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_corpus.py",
            "--corpus",
            str(tmp_path / "transaction"),
            "--mizar-html2",
            str(html2),
        ],
    )

    with pytest.raises(SystemExit):
        verify_corpus.main()
