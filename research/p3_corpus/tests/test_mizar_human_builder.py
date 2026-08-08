"""TDD contract for the direct official-source human Mizar builder."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

import build_mizar_human_shard as builder
import build_p3_generation as generation
from mizar_current_index import (
    FLAT_TREE_HASH_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    build_index,
    hash_flat_tree,
)

SHAPES = Path(__file__).parent / "fixtures" / "mizar_human" / "shapes.miz"


class TinyTokenizer:
    """Deterministic test tokenizer with the production metadata surface."""

    identity = builder.QWEN_TOKENIZER_ID
    tokenizer_json_sha256 = "a" * 64
    tokenizer_config_sha256 = "b" * 64
    behavior_digest = "c" * 64
    tokenizers_version = "test"
    eos_token_id = builder.QWEN_EOS_TOKEN_ID
    path = "/test/tokenizer.json"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode("utf-8"))


class FakeIndex:
    """Small index facade that records contextual local-label lookups."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        names = (
            "EXT:1",
            "EXT:2",
            "TARSKI:def_3",
            "ORDINAL1:sch_1",
            "SAMPLE:def_5",
            "SAMPLE:9",
            "OTHER:19",
        )
        self._statements = {name: f"statement for {name}" for name in names}

    def statement_map(self) -> dict[str, str]:
        return dict(self._statements)

    def article_local_label_maps(self) -> dict[str, dict[str, tuple[str, ...]]]:
        return {
            "SAMPLE": {
                "Def5a": ("SAMPLE:def_5",),
                "Th1": ("SAMPLE:1", "SAMPLE:9"),
            },
            "OTHER": {
                "Lm4": ("OTHER:19",),
                "Reused": ("OTHER:2", "OTHER:7"),
            },
        }

    def resolve_local_label(self, article: str, label: str, *, at_identity: str) -> str:
        self.calls.append((article, label, at_identity))
        resolved = {
            ("SAMPLE", "Def5a", "SAMPLE:20"): "SAMPLE:def_5",
            ("SAMPLE", "Th1", "SAMPLE:20"): "SAMPLE:9",
        }
        try:
            return resolved[(article, label, at_identity)]
        except KeyError as error:
            raise KeyError(label) from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_html(records: list[tuple[int, str, str]]) -> str:
    blocks = ["<html><body>"]
    for number, label, statement in records:
        label_html = f'<span class="lab">{label}</span>:' if label else ""
        blocks.append(f"""\
<div about="#T{number}" typeof="oo:Theorem">
<span class="kw">theorem </span>{label_html}
<a name="T{number}"><span class="comment">:: SAMPLE:{number}</span><br></a>
<div class="add">{statement}</div>
</div>""")
    blocks.append("</body></html>")
    return "\n".join(blocks)


def _thproof(label: str, source_goal: str, target: str) -> str:
    return f"theorem {label}:\n" f"  {source_goal}\n" "proof\n" f"{target}\n" "end;\n"


def _write_release(tmp_path: Path) -> tuple[builder.BuildConfig, TinyTokenizer]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots = {
        "mml": tmp_path / "mml",
        "html": tmp_path / "html",
        "thproofs": tmp_path / "thproofs",
    }
    for root in roots.values():
        root.mkdir()

    long_comment = "x" * 17_000
    source = f"""\
begin

theorem Base:
  BaseGoal
proof
  thus thesis;
end;

theorem Def5a:
  SharedGoal
proof
  assume A1: BaseGoal;
  thus thesis by Base, A1;
end;

theorem Duplicate:
  SharedGoal
proof
  assume A1: BaseGoal;
  thus thesis by Base, A1;
end;

theorem LongOne:
  LongGoal
proof
  thus thesis by Base;
  :: {long_comment}
end;

theorem InlineOne:
  InlineGoal by Base;

theorem BareOne:
  BareGoal;

theorem Broken:
  BrokenGoal
proof
  now
    thus thesis by Base;
  end;

theorem Unresolved:
  UnresolvedGoal
proof
  thus thesis by MissingGlobal;
end;

theorem Final:
  FinalGoal
proof
  thus thesis by Base;
end;

theorem Gone:
  canceled;
"""
    (roots["mml"] / "sample.miz").write_text(source, encoding="utf-8")

    semantic = [
        (1, "Base", "BaseGoal"),
        (2, "Def5a", "SharedGoal"),
        (3, "Duplicate", "SharedGoal"),
        (4, "LongOne", "LongGoal"),
        (5, "InlineOne", "InlineGoal"),
        (6, "BareOne", "BareGoal"),
        (7, "Broken", "BrokenGoal"),
        (8, "Unresolved", "UnresolvedGoal"),
        (9, "Final", "FinalGoal"),
    ]
    (roots["html"] / "sample.html").write_text(
        _semantic_html(semantic),
        encoding="utf-8",
    )
    thproof_targets = {
        1: "  thus thesis;",
        2: "  thus thesis by Base;",
        3: "  thus thesis by Base;",
        4: "  thus thesis by Base;",
        5: "  thus thesis by Base;",
        6: "  thus thesis by Base;",
        7: "  thus thesis by Base;",
        8: "  thus thesis by Base;",
        9: "  thus thesis by Base;",
    }
    for number, label, source_goal in semantic:
        (roots["thproofs"] / f"t{number}_sample").write_text(
            _thproof(label, source_goal, thproof_targets[number]),
            encoding="utf-8",
        )

    archives = {}
    sources = {}
    for ordinal, (name, root) in enumerate(roots.items()):
        archive = tmp_path / f"{name}.archive"
        archive.write_bytes(f"archive-{name}".encode("ascii"))
        archives[name] = archive
        file_glob = "*.miz" if name == "mml" else "*.html" if name == "html" else "*"
        tree = hash_flat_tree(root, file_glob=file_glob)
        sources[name] = {
            "archive_url": f"https://invalid.example/{name}",
            "archive_sha256": _sha256(archive),
            "file_glob": file_glob,
            "file_count": tree.file_count,
            "tree_sha256": tree.sha256,
        }

    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "release": {
            "mizar_version": builder.MIZAR_VERSION,
            "mml_version": builder.MML_VERSION,
        },
        "tree_hash": {
            "schema": FLAT_TREE_HASH_SCHEMA,
            "description": "test",
        },
        "sources": sources,
        "expected": {},
        "proof_policy": {
            "completion_denominator": "explicit_proof_bearing_extracts",
            "minimum_explicit_completion_rate": 0,
        },
        "licensing": {
            "redistribution_rights_asserted": False,
            "status": "test fixture; no redistribution",
        },
    }
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    index_path = tmp_path / "semantic.sqlite"
    build_index(
        manifest_path=manifest_path,
        roots=roots,
        sqlite_path=index_path,
        jsonl_path=tmp_path / "semantic.jsonl",
        archive_paths=archives,
    )
    config = builder.BuildConfig(
        mml_root=roots["mml"],
        html_root=roots["html"],
        thproofs_root=roots["thproofs"],
        semantic_index=index_path,
        semantic_index_sha256=_sha256(index_path),
        source_manifest=manifest_path,
        mizar_archive=archives["mml"],
        html_archive=archives["html"],
        thproofs_archive=archives["thproofs"],
        out=tmp_path / "output",
        name="mizar",
        heldout=0,
        seed=20260801,
        production=False,
    )
    return config, TinyTokenizer()


def test_parser_balances_nested_human_proofs_and_never_crosses_declarations() -> None:
    text = SHAPES.read_text(encoding="utf-8")

    declarations = builder.parse_miz_article(
        text,
        article="SHAPES",
        source_file=SHAPES.name,
    )

    assert [declaration.category for declaration in declarations] == [
        "complete_explicit_proof",
        "inline_justification",
        "canceled",
        "no_explicit_proof",
        "malformed_explicit_proof",
        "complete_explicit_proof",
    ]
    first = declarations[0]
    assert first.label == "Def5a"
    assert "per cases;" in first.target
    assert "suppose A1: C;" in first.target
    assert "case B1;" in first.target
    assert "hereby" in first.target
    assert text[first.target_start : first.target_end] == first.target
    assert first.local_assumptions == {
        "A0": "H",
        "A1": "C",
        "A2": "not C",
    }
    malformed = declarations[-2]
    assert malformed.target is None
    assert "theorem Good" not in malformed.source_declaration
    assert declarations[-1].target == "thus thesis by EXT:1;"


def test_literal_identity_alignment_uses_anchors_and_rejects_drift() -> None:
    source = builder.parse_miz_article(
        """\
theorem First: P proof thus thesis; end;
theorem Second: Q proof thus thesis; end;
""",
        article="SAMPLE",
        source_file="sample.miz",
    )
    anchors = [
        builder.SemanticAnchor(
            identity="SAMPLE:1",
            article="SAMPLE",
            number=1,
            local_label="Generated",
            source_goal="P",
            mml_alignment="generated_or_unmatched",
        ),
        builder.SemanticAnchor(
            identity="SAMPLE:3",
            article="SAMPLE",
            number=3,
            local_label="First",
            source_goal="P",
            mml_alignment="literal_goal_match",
        ),
        builder.SemanticAnchor(
            identity="SAMPLE:8",
            article="SAMPLE",
            number=8,
            local_label="Second",
            source_goal="Q",
            mml_alignment="literal_goal_match",
        ),
    ]

    aligned = builder.align_article_declarations(source, anchors)

    assert [item.identity for item in aligned] == ["SAMPLE:3", "SAMPLE:8"]
    assert [item.source.ordinal for item in aligned] == [1, 2]

    with pytest.raises(builder.SourceIndexMismatch, match="order|anchor"):
        builder.align_article_declarations(source, [anchors[2], anchors[1]])
    with pytest.raises(builder.SourceIndexMismatch, match="count|unmapped"):
        builder.align_article_declarations(source, anchors[:-1])
    changed = [
        anchors[1],
        builder.SemanticAnchor(**{**anchors[2].__dict__, "source_goal": "R"}),
    ]
    with pytest.raises(builder.SourceIndexMismatch, match="source goal"):
        builder.align_article_declarations(source, changed)


def test_production_alignment_disambiguates_repeated_proofs_by_global_order() -> None:
    source = builder.parse_miz_article(
        """\
theorem P proof thus thesis; end;
theorem P proof thus thesis; end;
""",
        article="SAMPLE",
        source_file="sample.miz",
    )
    proof_sha256 = hashlib.sha256(b"thus thesis;").hexdigest()
    anchors = [
        builder.SemanticAnchor(
            identity=f"SAMPLE:{number}",
            article="SAMPLE",
            number=number,
            local_label=None,
            source_goal="P",
            mml_alignment="literal_goal_match",
            proof_category="complete_explicit_proof",
            proof_sha256=proof_sha256,
        )
        for number in (66, 74)
    ]

    aligned, unanchored = builder._strict_complete_alignment(source, anchors)

    assert [item.identity for item in aligned] == ["SAMPLE:66", "SAMPLE:74"]
    assert [item.source.ordinal for item in aligned] == [1, 2]
    assert unanchored == 0


def test_production_alignment_skips_generated_goal_match_false_positives() -> None:
    source = builder.parse_miz_article(
        "theorem P proof thus thesis; end;\n",
        article="SAMPLE",
        source_file="sample.miz",
    )
    source_hash = hashlib.sha256(b"thus thesis;").hexdigest()
    generated_hash = hashlib.sha256(b"generated proof").hexdigest()
    anchors = [
        builder.SemanticAnchor(
            identity=f"SAMPLE:{number}",
            article="SAMPLE",
            number=number,
            local_label=None,
            source_goal="P",
            mml_alignment="literal_goal_match",
            proof_category="complete_explicit_proof",
            proof_sha256=generated_hash,
        )
        for number in range(1, 5)
    ]
    anchors.append(
        builder.SemanticAnchor(
            identity="SAMPLE:5",
            article="SAMPLE",
            number=5,
            local_label=None,
            source_goal="P",
            mml_alignment="literal_goal_match",
            proof_category="complete_explicit_proof",
            proof_sha256=source_hash,
        )
    )

    aligned, unanchored = builder._strict_complete_alignment(source, anchors)

    assert [item.identity for item in aligned] == ["SAMPLE:5"]
    assert unanchored == 0


def _secondary_alignment_fixture():
    declarations = builder.parse_miz_article(
        """\
theorem Left: LeftGoal proof thus thesis by EXT:1; end;
theorem Recovered: RecoveryGoal proof thus thesis by EXT:1; end;
theorem Right: RightGoal proof thus thesis by EXT:1; end;
""",
        article="SAMPLE",
        source_file="sample.miz",
    )
    proof_sha256 = hashlib.sha256(b"thus thesis by EXT:1;").hexdigest()
    anchors = [
        builder.SemanticAnchor(
            identity="SAMPLE:1",
            article="SAMPLE",
            number=1,
            local_label="Left",
            source_goal="LeftGoal",
            mml_alignment="literal_goal_match",
            statement="LeftGoal",
            proof_category="complete_explicit_proof",
            proof_sha256=proof_sha256,
        ),
        builder.SemanticAnchor(
            identity="SAMPLE:2",
            article="SAMPLE",
            number=2,
            local_label="Recovered",
            source_goal="RecoveryGoal",
            mml_alignment="literal_goal_match",
            statement="RecoveryGoal",
            proof_category="malformed_explicit_proof",
            proof_sha256=None,
        ),
        builder.SemanticAnchor(
            identity="SAMPLE:3",
            article="SAMPLE",
            number=3,
            local_label="Right",
            source_goal="RightGoal",
            mml_alignment="literal_goal_match",
            statement="RightGoal",
            proof_category="complete_explicit_proof",
            proof_sha256=proof_sha256,
        ),
    ]
    primary, unanchored = builder._strict_complete_alignment(declarations, anchors)
    assert [item.identity for item in primary] == ["SAMPLE:1", "SAMPLE:3"]
    assert unanchored == 1
    return declarations, anchors, primary


def test_secondary_alignment_recovers_complete_unique_labeled_malformed_index_proof() -> (
    None
):
    declarations, anchors, primary = _secondary_alignment_fixture()

    recovered = builder._secondary_unique_label_alignment(
        declarations,
        anchors,
        primary,
    )

    assert [item.identity for item in recovered] == ["SAMPLE:2"]
    assert recovered[0].source.ordinal == 2
    assert recovered[0].source_index_binding == {
        "schema_version": builder.SOURCE_INDEX_BINDING_SCHEMA,
        "method": builder.SECONDARY_ALIGNMENT_METHOD,
        "source_label_occurrences": 1,
        "index_label_occurrences": 1,
        "normalized_goal_sha256": hashlib.sha256(b"RecoveryGoal").hexdigest(),
        "previous_proof_hash_anchor": {
            "source_ordinal": 1,
            "identity": "SAMPLE:1",
            "index_number": 1,
            "proof_sha256": hashlib.sha256(b"thus thesis by EXT:1;").hexdigest(),
        },
        "next_proof_hash_anchor": {
            "source_ordinal": 3,
            "identity": "SAMPLE:3",
            "index_number": 3,
            "proof_sha256": hashlib.sha256(b"thus thesis by EXT:1;").hexdigest(),
        },
    }


@pytest.mark.parametrize(
    "unsafe_case",
    (
        "unlabeled",
        "reused_source_label",
        "reused_index_label",
        "generated_same_goal",
        "available_mismatched_proof_hash",
        "ambiguous_candidates",
        "outside_neighbor_bounds",
        "already_used_anchor",
    ),
)
def test_secondary_alignment_fails_closed_on_unsafe_binding(unsafe_case: str) -> None:
    declarations, anchors, primary = _secondary_alignment_fixture()
    declarations = list(declarations)
    anchors = list(anchors)

    if unsafe_case == "unlabeled":
        declarations[1] = replace(declarations[1], label=None)
    elif unsafe_case == "reused_source_label":
        declarations[0] = replace(declarations[0], label="Recovered")
    elif unsafe_case == "reused_index_label":
        anchors.append(
            replace(
                anchors[1],
                identity="SAMPLE:9",
                number=9,
            )
        )
    elif unsafe_case == "generated_same_goal":
        anchors[1] = replace(
            anchors[1],
            mml_alignment="generated_or_unmatched",
        )
    elif unsafe_case == "available_mismatched_proof_hash":
        anchors[1] = replace(
            anchors[1],
            proof_category="complete_explicit_proof",
            proof_sha256=hashlib.sha256(b"different proof").hexdigest(),
        )
    elif unsafe_case == "ambiguous_candidates":
        anchors.append(
            replace(
                anchors[1],
                identity="SAMPLE:4",
                number=2,
            )
        )
    elif unsafe_case == "outside_neighbor_bounds":
        anchors[1] = replace(anchors[1], number=4)
    elif unsafe_case == "already_used_anchor":
        anchors[1] = replace(anchors[1], identity="SAMPLE:1")
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(unsafe_case)

    assert (
        builder._secondary_unique_label_alignment(
            declarations,
            anchors,
            primary,
        )
        == []
    )


def test_citations_resolve_nonstandard_reused_and_qualified_labels() -> None:
    index = FakeIndex()
    body = """\
A1: P by EXT:1,2;
thus thesis by A1, TARSKI:def 3, Def5a, Th1, OTHER:Lm4;
thus thesis from ORDINAL1:sch 1(A1);
"""

    resolution = builder.resolve_global_citations(
        body,
        index,
        theorem="SAMPLE:20",
    )

    assert resolution.references == (
        "EXT:1",
        "EXT:2",
        "TARSKI:def_3",
        "SAMPLE:def_5",
        "SAMPLE:9",
        "OTHER:19",
        "ORDINAL1:sch_1",
    )
    assert resolution.unresolved == ()
    assert "A1" in resolution.proof_local_labels
    assert ("SAMPLE", "Def5a", "SAMPLE:20") in index.calls
    assert ("SAMPLE", "Th1", "SAMPLE:20") in index.calls


def test_proof_local_labels_are_not_global_and_unresolved_rows_are_explicit() -> None:
    index = FakeIndex()
    body = """\
A1: P;
thus thesis by A1, MissingLabel, OTHER:Reused;
"""

    resolution = builder.resolve_global_citations(
        body,
        index,
        theorem="SAMPLE:20",
    )

    assert resolution.references == ()
    assert resolution.unresolved == ("MissingLabel", "OTHER:Reused")
    assert resolution.proof_local_labels == ("A1",)


def test_proof_local_labels_cover_same_line_and_assume_that_forms() -> None:
    index = FakeIndex()
    body = """\
A: P; then B: Q;
assume that C: R and D: S;
1_G: T;
thus thesis by A, B, C, D, 1_G, EXT:1;
"""

    resolution = builder.resolve_global_citations(
        body,
        index,
        theorem="SAMPLE:20",
    )

    assert resolution.references == ("EXT:1",)
    assert resolution.unresolved == ()
    assert resolution.proof_local_labels == ("A", "B", "C", "D", "1_G")


def test_citation_clause_stops_before_chained_calculation_text() -> None:
    index = FakeIndex()
    body = """\
A1: P;
A2: Q;
x = x by A1
  .= y by A2, EXT:1;
"""

    resolution = builder.resolve_global_citations(
        body,
        index,
        theorem="SAMPLE:20",
    )

    assert resolution.references == ("EXT:1",)
    assert resolution.unresolved == ()
    assert resolution.proof_local_labels == ("A1", "A2")


def test_fact_shuffle_is_deterministic_and_independent_of_citation_order() -> None:
    forward = ["A:1", "B:1", "C:1", "A:1"]
    backward = list(reversed(forward))
    expected = sorted(
        set(forward),
        key=lambda name: (
            hashlib.sha256(
                f"mizar-human-proof-v1\0{20260801}\0SAMPLE:20\0{name}".encode()
            ).hexdigest(),
            name,
        ),
    )

    observed = builder.deterministic_fact_order(
        forward,
        row_key="SAMPLE:20",
        seed=20260801,
    )
    assert observed == expected
    assert observed == builder.deterministic_fact_order(
        backward,
        row_key="SAMPLE:20",
        seed=20260801,
    )


def test_full_staging_build_is_deterministic_typed_and_deep_replayable(
    tmp_path: Path,
) -> None:
    config, tokenizer = _write_release(tmp_path)

    report = builder.build_corpus(config, tokenizer)

    counters = report["counters"]
    assert counters["declarations_total"] == 10
    assert counters["accepted_rows"] == 2
    assert counters["dropped_canceled"] == 1
    assert counters["dropped_inline_justification"] == 1
    assert counters["dropped_no_explicit_proof"] == 1
    assert counters["dropped_malformed_explicit_proof"] == 1
    assert counters["dropped_no_global_citation"] == 1
    assert counters["dropped_unresolved_reference"] == 1
    assert counters["dropped_duplicate"] == 1
    assert counters["dropped_overlength"] == 1

    raw_path = config.out / "raw" / "mizar.jsonl"
    rows = [
        json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["theorem"] for row in rows] == ["SAMPLE:2", "SAMPLE:9"]
    first = rows[0]
    assert first["schema_version"] == builder.ROW_SCHEMA
    assert first["split"] == "raw"
    assert first["heldout"] == 0
    assert first["goal"] == "SharedGoal"
    assert first["facts"] == {"SAMPLE:1": "BaseGoal"}
    assert first["cited"] == ["SAMPLE:1"]
    assert first["local_assumptions"] == {"A1": "BaseGoal"}
    assert first["target"] == ("assume A1: BaseGoal;\n  thus thesis by Base, A1;")
    assert (
        first["text"]
        == builder.render_training_text(
            first["facts"],
            first["goal"],
            first["target"],
        )[0]
    )
    assert first["token_length_with_eos"] == len(tokenizer.encode(first["text"])) + 1
    assert (
        first["source"]["target_sha256"]
        == hashlib.sha256(first["target"].encode("utf-8")).hexdigest()
    )
    assert (
        first["index"]["statement_sha256"]
        == hashlib.sha256(first["goal"].encode("utf-8")).hexdigest()
    )

    assert report["deep_self_check"]["rows_checked"] == 2
    assert report["raw_replay"]["rows_checked"] == 2
    assert report["context_eligibility"]["eligible_rows"] == 2
    assert report["context_eligibility"]["overlength_rows"] == 1
    assert report["fact_frequencies"] == {"SAMPLE:1": 2}
    assert report["output_hashes"]["raw_jsonl_sha256"] == _sha256(raw_path)
    assert not (config.out / "shards").exists()
    assert not (config.out / "eval").exists()

    second = builder.BuildConfig(
        **{**config.__dict__, "out": tmp_path / "output-second"}
    )
    second_report = builder.build_corpus(second, tokenizer)
    assert raw_path.read_bytes() == (second.out / "raw" / "mizar.jsonl").read_bytes()
    assert (config.out / "manifests" / "mizar.json").read_bytes() == (
        second.out / "manifests" / "mizar.json"
    ).read_bytes()
    assert (
        report["output_hashes"]["raw_jsonl_sha256"]
        == second_report["output_hashes"]["raw_jsonl_sha256"]
    )


def test_real_builder_fixture_satisfies_canonical_six_family_adapter(
    tmp_path: Path,
) -> None:
    config, tokenizer = _write_release(tmp_path)

    report = builder.build_corpus(config, tokenizer)
    manifest = json.loads(
        (config.out / "manifests" / "mizar.json").read_text(encoding="utf-8")
    )
    row = json.loads(
        (config.out / "raw" / "mizar.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )

    assert row["schema_version"] == "mizar-proof-v2"
    assert set(manifest) == {
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
    assert manifest["schema_version"] == "p3-family-source-manifest/v2"
    assert manifest["family"] == "mizar"
    assert manifest["row_schema_version"] == "mizar-proof-v2"
    assert report["logical_raw_path"] == "raw/mizar.jsonl"
    generation._validate_source_manifest(manifest, family="mizar", production=False)
    generation.validate_family_record(
        row,
        family="mizar",
        source_manifest=manifest,
        location="raw/mizar.jsonl:1",
    )
    raw_spec = manifest["builder"]["raw"]
    inventory = generation._validate_exact_stage_inventory(
        config.out,
        raw_spec["inventory"],
        family="mizar",
        stage="raw",
    )
    assert inventory[raw_spec["outputs"]["raw"]] == config.out / "raw" / "mizar.jsonl"


def test_source_and_index_hash_drift_fail_closed(tmp_path: Path) -> None:
    config, tokenizer = _write_release(tmp_path)
    source = config.mml_root / "sample.miz"
    source.write_text(source.read_text(encoding="utf-8") + "\n:: drift\n")

    with pytest.raises(builder.BuildError, match="source drift"):
        builder.build_corpus(config, tokenizer)
    assert not config.out.exists()

    clean_config, clean_tokenizer = _write_release(tmp_path / "clean")
    wrong_hash = builder.BuildConfig(
        **{**clean_config.__dict__, "semantic_index_sha256": "0" * 64}
    )
    with pytest.raises(builder.BuildError, match="index SHA-256"):
        builder.build_corpus(wrong_hash, clean_tokenizer)
    assert not wrong_hash.out.exists()


def test_builder_api_is_raw_only_and_legacy_html2_is_not_an_option(
    tmp_path: Path,
) -> None:
    config, tokenizer = _write_release(tmp_path)
    heldout = builder.BuildConfig(**{**config.__dict__, "heldout": 1})

    with pytest.raises(builder.BuildError, match="heldout=0"):
        builder.build_corpus(heldout, tokenizer)

    parser = builder.create_argument_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--html2" not in options
    assert {
        "--semantic-index",
        "--semantic-index-sha256",
        "--source-manifest",
        "--mml-root",
        "--html-root",
        "--thproofs-root",
        "--heldout",
        "--tokenizer-path",
    } <= options


def test_production_counts_are_hard_release_gates() -> None:
    counters = Counter(builder.EXPECTED_PRODUCTION_COUNTERS)

    builder._validate_production_counts(counters)

    counters["mapped_complete_declarations"] -= 1
    counters["dropped_source_index_unanchored"] += 1
    with pytest.raises(builder.BuildError, match="production count mismatch"):
        builder._validate_production_counts(counters)


def test_exact_vendored_qwen_tokenizer_is_accepted() -> None:
    tokenizer_path = (
        Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
    )

    tokenizer = builder.load_vendored_tokenizer(tokenizer_path)

    assert tokenizer.identity == builder.QWEN_TOKENIZER_ID
    assert tokenizer.eos_token_id == builder.QWEN_EOS_TOKEN_ID
    assert tokenizer.behavior_digest == builder.APPROVED_TOKENIZER_BEHAVIOR_SHA256
