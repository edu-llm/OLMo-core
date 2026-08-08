"""Coverage regressions for the Mizar parsers.

The shard invariants check that whatever we emit is well formed. They cannot see
what we never emitted, which is where the expensive bugs were: two regexes
required syntax that only LABELLED theorems have, so half the MML was invisible
and 24,176 examples were dropped as "unresolvable citations" when the citations
were fine and the dictionary was short.

These tests pin the shapes that were being missed. They are written against the
real MML files rather than fixtures, because the bug was a mismatch with the real
syntax and a fixture would just re-encode my wrong assumption.
"""

import glob
import hashlib
import json
import os
import re
import sys

import pytest

sys.path.insert(0, "scripts")

MIZAR = os.environ.get("MIZAR_HTML2", "/tmp/dscount/mizar/html2")
AFINSQ = os.path.join(MIZAR, "afinsq_1.xml1.txt")
SCRATCH_HTML2 = "/tmp/memorysplit-mizar-tdd-sol/sources/html2/html2"
needs_html2 = pytest.mark.skipif(
    not os.path.exists(AFINSQ), reason="MML html2 not present"
)
needs_scratch_html2 = pytest.mark.skipif(
    not os.path.exists(os.path.join(SCRATCH_HTML2, "afinsq_1.xml1.txt")),
    reason="scratch MML html2 not present",
)


def test_build_plan_does_not_pair_incompatible_mizar_sources():
    plan = (
        os.path.join(os.path.dirname(__file__), "..", "CORPUS_BUILD_PLAN.md")
    )
    with open(plan, encoding="utf-8") as plan_file:
        text = plan_file.read()

    assert "nn_conj20 html2 is incompatible with MML 8.1.15 thproofs" in text
    assert "must not be combined" in text
    assert "Current MML 8.1.15 semantic HTML and plain sources are recovered" in text
    assert "mizar-semantic-index-v1" in text
    assert "mizar-current-sources-v1" in text
    assert "--semantic-index" in text
    assert "redistribution rights remain unresolved" in text
    assert "pending source recovery" not in text


@needs_html2
def test_unlabelled_theorems_are_parsed():
    """`theorem :: AFINSQ_1:1` has no label and so no colon before the `::`."""
    from build_mizar_shard import parse_article
    stmt, _ = parse_article(AFINSQ)
    assert "AFINSQ_1:1" in stmt, "unlabelled theorem missing from dictionary"
    assert "AFINSQ_1:2" in stmt, "labelled theorem missing from dictionary"


@needs_html2
def test_dictionary_covers_most_theorem_headers():
    """Every `:: ARTICLE:N` header in a file should reach the dictionary."""
    from build_mizar_shard import parse_article
    stmt, _ = parse_article(AFINSQ)
    with open(AFINSQ, errors="replace") as article_file:
        article = article_file.read()
    headers = {m.group(1) for m in
               re.finditer(r"^theorem.*?::\s*([A-Z_0-9]+:\d+)\s*$",
                           article, re.MULTILINE)}
    got = headers & set(stmt)
    assert len(got) >= 0.9 * len(headers), (
        f"only {len(got)}/{len(headers)} theorem headers parsed")


@needs_html2
def test_unlabelled_theorems_yield_proof_examples():
    """The example extractor must see unlabelled theorems too, not just the
    dictionary — this is a separate regex with the same defect."""
    from build_mizar_shard import iter_theorem_proofs
    with open(AFINSQ, errors="replace") as article_file:
        txt = article_file.read()
    names = {name for name, _, _ in iter_theorem_proofs(txt)}
    labelled = {m.group(1) for m in re.finditer(
        r"^theorem\s+\w+\s*:\s*::\s*([A-Z_0-9]+:\d+)", txt, re.MULTILINE)}
    assert names - labelled, "extractor found only labelled theorems"


def test_inline_proof_keyword_is_split():
    """In thproofs the `proof` keyword often sits at the end of the statement
    line rather than alone on its own line."""
    from build_thproofs_shard import split_proof
    body = split_proof('for x holds P x proof let x; thus P x; end;')
    assert body is not None and body.strip().startswith("let x")


def test_bare_citation_is_not_a_proof():
    """`theorem X by Th99;` has no derivation — the target would be the name of
    the one fact already sitting in the block."""
    from build_thproofs_shard import split_proof
    assert split_proof("vars Non a = vars a by Th99;") is None


REAL_SHAPE_ARTICLE = """\
theorem Th1: :: SAMPLE:1
for x being set holds x = x
proof
  thus thesis;
end;

Lm2: for x being set holds x = x
proof
  thus thesis;
end;

theorem :: SAMPLE:2
for x being set holds x = x by SAMPLE:1;
theorem Lm3: :: SAMPLE:3
for x being set holds x = x
definition
let x be set;
end;
proof
  this proof belongs to no theorem;
end;

:: deftheorem Def7 defines sample SAMPLE:def_7_:_
for x being set holds x = x;

scheme :: ORDINAL1:sch 1
OrdinalMin{ P1[ Ordinal] } :
 ex A being Ordinal st P1[A]
provided
A1: ex A being Ordinal st P1[A]
proof
  thus thesis by A1;
end;

theorem :: SAMPLE:4
for x being set holds x = x proof
  thus thesis by SAMPLE:1;
end;

theorem Gone: :: SAMPLE:5
canceled;
theorem :: SAMPLE:6
for x being set holds x = x
proof
  now
    thus x = x;
  end;
  thus thesis by SAMPLE:1;
end;
"""


def test_sequential_parser_bounds_proofs_to_the_immediately_preceding_theorem():
    from build_mizar_shard import iter_theorem_proofs

    proofs = list(iter_theorem_proofs(REAL_SHAPE_ARTICLE))

    assert [name for name, _, _ in proofs] == [
        "SAMPLE:1",
        "SAMPLE:4",
        "SAMPLE:6",
    ]
    assert all("theorem ::" not in goal for _, goal, _ in proofs)
    assert "this proof belongs to no theorem" not in "\n".join(
        body for _, _, body in proofs
    )
    assert "Lm2:" not in proofs[0][2]
    assert "  end;\n  thus thesis" in proofs[-1][2]


def test_article_dictionary_drops_canceled_statements_and_keeps_real_shapes(tmp_path):
    from build_mizar_shard import parse_article

    article = tmp_path / "sample.xml1.txt"
    article.write_text(REAL_SHAPE_ARTICLE)
    statements, local = parse_article(article)

    assert set(statements) == {
        "SAMPLE:1",
        "SAMPLE:2",
        "SAMPLE:3",
        "SAMPLE:4",
        "SAMPLE:6",
        "SAMPLE:def_7",
        "SAMPLE:Lm2",
        "ORDINAL1:sch_1",
    }
    assert local["Th1"] == "SAMPLE:1"
    assert local["Lm2"] == "SAMPLE:Lm2"
    assert local["Lm3"] == "SAMPLE:3"
    assert local["Def7"] == "SAMPLE:def_7"
    assert "Gone" not in local
    assert all(
        statement.strip().lower() != "canceled;"
        for statement in statements.values()
    )


def test_thproofs_resolves_schemes_definitions_and_exact_article_local_labels():
    from build_thproofs_shard import resolve_references

    body = """\
A1: x = x by EXT:1,2, TARSKI:def 4,def 5;
thus thesis by A1,Th7,Lm2,Def3;
thus thesis from ORDINAL1:sch_1(A1);
"""
    local = {
        "Th7": "ARTICLE:12",
        "Lm2": "ARTICLE:9",
        "Def3": "ARTICLE:def_8",
    }
    expected = [
        "EXT:1",
        "EXT:2",
        "TARSKI:def_4",
        "TARSKI:def_5",
        "ARTICLE:12",
        "ARTICLE:9",
        "ARTICLE:def_8",
        "ORDINAL1:sch_1",
    ]
    statements = {name: f"statement for {name}" for name in expected}

    refs, missing = resolve_references(body, local, statements)

    assert refs == expected
    assert missing == []

    refs, missing = resolve_references(
        body,
        {name: value for name, value in local.items() if name != "Lm2"},
        statements,
    )
    assert refs == [name for name in expected if name != "ARTICLE:9"]
    assert missing == ["Lm2"]

    refs, missing = resolve_references(
        body, local, {name: value for name, value in statements.items()
                      if name != "ORDINAL1:sch_1"}
    )
    assert refs == expected
    assert missing == ["ORDINAL1:sch_1"]


def test_thproofs_resolves_all_article_labels_not_proof_labels():
    from build_mizar_shard import resolve_references

    local = {
        "Def5a": "ARTICLE:def_9",
        "Auxiliary": "ARTICLE:17",
        "Th7b": "ARTICLE:22",
    }
    statements = {
        "ARTICLE:def_9": "definition statement",
        "ARTICLE:17": "auxiliary statement",
        "ARTICLE:22": "theorem statement",
    }
    body = """\
A1: x = x by Def5a;
thus thesis by A1, Auxiliary, Th7b;
"""

    refs, missing = resolve_references(body, local, statements)

    assert refs == ["ARTICLE:def_9", "ARTICLE:17", "ARTICLE:22"]
    assert missing == []


def test_reference_resolution_uses_the_exact_parse_article_label_map(tmp_path):
    from build_mizar_shard import parse_article, resolve_references

    article = tmp_path / "article.xml1.txt"
    article.write_text(
        """\
theorem Def5a: :: ARTICLE:1
for x being set holds x = x by ARTICLE:2;
theorem Auxiliary: :: ARTICLE:2
for x being set holds x = x by ARTICLE:1;
"""
    )
    statements, local = parse_article(article)

    refs, missing = resolve_references(
        "thus thesis by Def5a, Auxiliary, A1;",
        local,
        statements,
    )

    assert local == {"Def5a": "ARTICLE:1", "Auxiliary": "ARTICLE:2"}
    assert refs == ["ARTICLE:1", "ARTICLE:2"]
    assert missing == []


def test_qualified_local_labels_resolve_only_against_the_named_article():
    from build_mizar_shard import resolve_references

    current = {"Lm4": "CURRENT:4", "Def5a": "CURRENT:def_5"}
    all_locals = {
        "CURRENT": current,
        "OTHER": {
            "Lm4": "OTHER:19",
            "Def5a": "OTHER:def_8",
            "Auxiliary": "OTHER:23",
        },
    }
    statements = {
        "CURRENT:4": "current lemma",
        "CURRENT:def_5": "current definition",
        "OTHER:19": "other lemma",
        "OTHER:def_8": "other definition",
        "OTHER:23": "other auxiliary",
    }

    refs, missing = resolve_references(
        "thus thesis by Lm4, OTHER:Lm4, OTHER:Def5a, OTHER:Auxiliary;",
        current,
        statements,
        local_by_article=all_locals,
    )

    assert refs == [
        "CURRENT:4",
        "OTHER:19",
        "OTHER:def_8",
        "OTHER:23",
    ]
    assert missing == []

    refs, missing = resolve_references(
        "thus thesis by OTHER:Lm4, MISSING:Def5a;",
        current,
        statements,
        local_by_article={"CURRENT": current},
    )
    assert refs == ["OTHER:Lm4", "MISSING:Def5a"]
    assert missing == ["OTHER:Lm4", "MISSING:Def5a"]


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (
            """theorem OddLabel:
for x being set holds x = x
proof
  now
    thus x = x;
  end;
  thus thesis;
end;
""",
            "now\n    thus x = x;\n  end;\n  thus thesis;",
        ),
        (
            """theorem WeirdLabel: for x being set holds x = "proof end;"
proof
  :: proof and end; in a source comment are inert
  thus thesis;
end;
""",
            ':: proof and end; in a source comment are inert\n  thus thesis;',
        ),
    ],
)
def test_thproofs_accepts_only_balanced_outer_proof_completion(chunk, expected):
    from build_thproofs_shard import split_proof

    assert split_proof(chunk) == expected


@pytest.mark.parametrize(
    "chunk",
    [
        "theorem T: P proof now thus P; end;",
        "theorem T: P proof thus P; end; garbage",
        "theorem T: P proof now thus P; end; end; end;",
        "theorem T: P proof now thus P; end garbage; end;",
    ],
)
def test_thproofs_rejects_incomplete_or_malformed_proof_blocks(chunk):
    from build_thproofs_shard import split_proof

    assert split_proof(chunk) is None


HTML2_POST_PROOF_FIXTURE = """\
theorem :: SAMPLE:1
for p being Proof of S holds p " {0} = p " {0}
proof
  now__::_thesis:_p_=_p
  percasesthen (p = p or p <> p);
  supposeA1: p = p;
    thus thesis;
  end;
  suppose p <> p;
    thus thesis;
  end;
  end;
  end;
end;

begin
"""


def test_html2_declarations_bound_exporter_blocks_before_proof_validation():
    from build_mizar_shard import iter_theorem_proofs

    proofs = list(iter_theorem_proofs(HTML2_POST_PROOF_FIXTURE))

    assert len(proofs) == 1
    _, _, body = proofs[0]
    assert "percasesthen" in body
    assert body.endswith("end;")
    assert "begin" not in body


@pytest.mark.parametrize(
    "suffix",
    [
        "\n\nBogusLabel: P by SAMPLE:2;",
        "\n\nBogusLabel: P proof thus thesis;",
        "\n\nP by SAMPLE:2;",
        "\n\nP from SampleScheme(SAMPLE:2);",
        "\n\nend;",
        "\n\ngarbage;",
    ],
)
def test_html2_parser_rejects_unaccounted_post_proof_text(suffix):
    from build_mizar_shard import iter_theorem_proofs

    article = (
        "theorem :: SAMPLE:1\n"
        "P proof thus thesis by SAMPLE:2; end;"
        f"{suffix}\n"
        "theorem :: SAMPLE:2\n"
        "Q proof thus thesis; end;\n"
    )
    assert {name for name, _, _ in iter_theorem_proofs(article)} == {"SAMPLE:2"}


@pytest.mark.parametrize(
    "declaration",
    [
        "begin",
        "set X = {};",
        "defpred P[set] means $1 = $1;",
        "deffunc F(set) -> set = $1;",
        "reconsider X = {} as set;",
        "consider X being set such that\nA1: X = X;",
        "scheme :: SAMPLE:sch 1",
        "AuxiliaryLabel: Q\nproof\n  thus thesis;\nend;",
    ],
)
def test_html2_parser_recognizes_structural_top_level_boundaries(declaration):
    from build_mizar_shard import iter_theorem_proofs

    article = (
        "theorem :: SAMPLE:1\n"
        "P proof thus thesis; end;\n\n"
        f"{declaration}\n\n"
        "theorem :: SAMPLE:2\n"
        "Q proof thus thesis; end;\n"
    )
    assert {name for name, _, _ in iter_theorem_proofs(article)} == {
        "SAMPLE:1",
        "SAMPLE:2",
    }


@needs_scratch_html2
@pytest.mark.parametrize(
    ("file_name", "identity"),
    [
        ("arytm_2.xml1.txt", "ARYTM_2:13"),
        ("euler_1.xml1.txt", "EULER_1:17"),
        ("graph_2.xml1.txt", "GRAPH_2:19"),
        ("jordan1c.xml1.txt", "JORDAN1C:2"),
        ("measure7.xml1.txt", "MEASURE7:11"),
        ("scm_comp.xml1.txt", "SCM_COMP:2"),
        ("taylor_1.xml1.txt", "TAYLOR_1:11"),
    ],
)
def test_legacy_parser_accepts_seven_structural_continuations(
    file_name, identity
):
    from build_mizar_shard import iter_theorem_proofs

    with open(
        os.path.join(SCRATCH_HTML2, file_name),
        encoding="utf-8",
        errors="replace",
    ) as article_file:
        names = {name for name, _, _ in iter_theorem_proofs(article_file.read())}
    assert identity in names


@needs_scratch_html2
def test_full_scratch_html2_declaration_classification_preserves_baseline():
    from build_mizar_shard import _declaration_sections, split_outer_proof

    declarations = with_proof = accepted = 0
    for path in sorted(glob.glob(os.path.join(SCRATCH_HTML2, "*.txt"))):
        with open(path, encoding="utf-8", errors="replace") as article_file:
            text = article_file.read()
        for kind, _, content in _declaration_sections(text):
            if kind != "theorem":
                continue
            declarations += 1
            proof_start, body = split_outer_proof(content)
            if proof_start is not None:
                with_proof += 1
            if body is not None:
                accepted += 1

    assert declarations == 52_179
    assert with_proof == 47_921
    assert accepted == 47_920


def test_thproofs_gold_goal_must_match_the_same_html2_theorem():
    from build_thproofs_shard import goal_diagnostic

    assert goal_diagnostic(
        "NonNumericLabel: for X being set holds {[X,union G], y} = Z",
        "for X being set holds {[X,(union G)],y} = Z",
    ) == "rendering-difference"
    assert goal_diagnostic(
        "NonNumericLabel: for X being set holds X = X :: source comment",
        "for X being set holds X = X",
    ) == "match"
    assert goal_diagnostic(
        "NonNumericLabel: for X being set holds X = {}",
        "for X being set holds X = {}",
    ) == "match"
    assert goal_diagnostic(
        "NonNumericLabel: for X being set holds X = X",
        "for X being set holds X = {}",
    ) == "different"


def _tree_digest(root):
    digest = hashlib.sha256()
    files = sorted(path for path in root.iterdir() if path.is_file())
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _write_thproof_source_pair(
    tmp_path,
    *,
    provenance_identity="SAMPLE:1",
    raw_goal="for x being set holds {[x,union G], y} = Z",
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    html2 = tmp_path / "html2"
    thproofs = tmp_path / "thproofs"
    html2.mkdir()
    thproofs.mkdir()
    (html2 / "sample.xml1.txt").write_text(
        """\
theorem SourceLabel: :: SAMPLE:1
for x being set holds {[x,(union G)],y} = Z
proof
  thus thesis by SAMPLE:2;
end;
theorem :: SAMPLE:2
for x being set holds x = x by SAMPLE:1;
"""
    )
    (thproofs / "t1_sample").write_text(
        f"""\
theorem ExportLabel: :: {provenance_identity}
{raw_goal}
proof
  thus thesis by SAMPLE:2;
end;
"""
    )
    manifest = {
        "schema_version": "mizar-thproof-sources-v1",
        "mml_version": "8.1.15-test",
        "html2": {
            "version": "html2-test",
            "file_count": 1,
            "tree_sha256": _tree_digest(html2),
        },
        "thproofs": {
            "version": "thproofs-test",
            "file_count": 1,
            "tree_sha256": _tree_digest(thproofs),
        },
        "coverage": {
            "minimum_source_files": 1,
            "minimum_name_matches": 1,
            "minimum_name_join_rate": 1.0,
            "minimum_complete_proofs": 1,
            "minimum_completion_rate": 1.0,
            "minimum_accepted_proofs": 1,
            "minimum_accepted_rate": 1.0,
        },
    }
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_text(json.dumps(manifest))
    return html2, thproofs, manifest_path


def _run_thproof_builder(monkeypatch, html2, thproofs, manifest, output):
    import build_thproofs_shard

    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_SOURCE_FILES", 1)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_NAME_MATCHES", 1)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_COMPLETE_PROOFS", 1)
    monkeypatch.setattr(build_thproofs_shard, "HARD_MIN_ACCEPTED_PROOFS", 1)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_thproofs_shard.py",
            "--src",
            str(thproofs),
            "--html2",
            str(html2),
            "--source-manifest",
            str(manifest),
            "--exclude",
            "",
            "--out",
            str(output),
            "--heldout",
            "0",
        ],
    )
    return build_thproofs_shard.main()


def test_production_thproof_floors_require_a_full_release():
    import build_thproofs_shard

    floors = build_thproofs_shard.effective_coverage_floors(
        {
            "minimum_source_files": 0,
            "minimum_name_matches": 0,
            "minimum_name_join_rate": 0.0,
            "minimum_complete_proofs": 0,
            "minimum_completion_rate": 0.0,
            "minimum_accepted_proofs": 0,
            "minimum_accepted_rate": 0.0,
        }
    )
    assert floors["minimum_source_files"] >= 70_000
    assert floors["minimum_name_matches"] >= 60_000
    assert floors["minimum_explicit_proofs"] >= 65_000
    assert floors["minimum_complete_proofs"] >= 55_000
    assert floors["minimum_completion_rate"] >= 0.80


def test_legacy_html2_manifest_cannot_authorize_thproof_output(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest = _write_thproof_source_pair(tmp_path)
    output = tmp_path / "output"

    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest, output
    ) != 0
    assert not (output / "shards" / "thproofs.jsonl").exists()


def test_legacy_html2_numeric_provenance_cannot_bypass_semantic_index(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest = _write_thproof_source_pair(
        tmp_path, provenance_identity="OTHER:2"
    )
    output = tmp_path / "output"

    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest, output
    ) != 0
    assert not (output / "shards" / "thproofs.jsonl").exists()


def test_thproofs_builder_rejects_genuinely_different_raw_goal(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest = _write_thproof_source_pair(
        tmp_path,
        raw_goal="for x being set holds x = {}",
    )

    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest, tmp_path / "output"
    ) != 0


def test_thproofs_builder_refuses_missing_manifest_and_invalidates_stale_outputs(
    tmp_path, monkeypatch
):
    html2, thproofs, _ = _write_thproof_source_pair(tmp_path)
    output = tmp_path / "output"
    stale = [
        output / "shards" / "thproofs.jsonl",
        output / "eval" / "thproofs.jsonl",
        output / "heldout" / "thproofs.json",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale")

    assert _run_thproof_builder(
        monkeypatch,
        html2,
        thproofs,
        tmp_path / "absent.json",
        output,
    ) != 0
    assert all(not path.exists() for path in stale)
    assert all(list(path.parent.glob(path.name + ".stale*")) for path in stale)


def test_thproofs_builder_rejects_hash_drift_and_failed_join_gate(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest_path = _write_thproof_source_pair(tmp_path)
    output = tmp_path / "output"
    (thproofs / "t1_sample").write_text("changed")
    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest_path, output
    ) != 0

    html2, thproofs, manifest_path = _write_thproof_source_pair(
        tmp_path / "second"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["coverage"]["minimum_name_matches"] = 2
    manifest_path.write_text(json.dumps(manifest))
    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest_path, tmp_path / "second-output"
    ) != 0


def test_code_floors_reject_sparse_join_even_when_manifest_requests_zero(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest_path = _write_thproof_source_pair(tmp_path)
    for number in range(2, 22):
        (thproofs / f"t{number}_missing").write_text(
            "theorem T: P proof thus thesis; end;"
        )
    manifest = json.loads(manifest_path.read_text())
    manifest["thproofs"]["file_count"] = 21
    manifest["thproofs"]["tree_sha256"] = _tree_digest(thproofs)
    manifest["coverage"] = {
        "minimum_source_files": 0,
        "minimum_name_matches": 0,
        "minimum_name_join_rate": 0.0,
        "minimum_complete_proofs": 0,
        "minimum_completion_rate": 0.0,
        "minimum_accepted_proofs": 0,
        "minimum_accepted_rate": 0.0,
    }
    manifest_path.write_text(json.dumps(manifest))

    assert _run_thproof_builder(
        monkeypatch, html2, thproofs, manifest_path, tmp_path / "output"
    ) != 0


def test_malformed_exclude_quarantines_all_hostile_stale_outputs(
    tmp_path, monkeypatch
):
    html2, thproofs, manifest = _write_thproof_source_pair(tmp_path)
    output = tmp_path / "output"
    stale = [
        output / "shards" / "thproofs.jsonl",
        output / "eval" / "thproofs.jsonl",
        output / "heldout" / "thproofs.json",
    ]
    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"looks": "fresh"}\n')
    exclude = tmp_path / "exclude.jsonl"
    exclude.write_text('{"theorem": "SAMPLE:1"}\nnot-json\n')

    import build_thproofs_shard

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_thproofs_shard.py",
            "--src",
            str(thproofs),
            "--html2",
            str(html2),
            "--source-manifest",
            str(manifest),
            "--exclude",
            str(exclude),
            "--out",
            str(output),
        ],
    )

    assert build_thproofs_shard.main() != 0
    assert all(not path.exists() for path in stale)
    assert all(list(path.parent.glob(path.name + ".stale*")) for path in stale)


@needs_html2
def test_article_dictionary_size():
    """Whole-library smoke test: the MML has well over 40k theorems."""
    from build_mizar_shard import parse_article
    n = 0
    for p in sorted(glob.glob(os.path.join(MIZAR, "*.txt")))[:200]:
        s, _ = parse_article(p)
        n += len(s)
    assert n > 8000, f"only {n} statements from 200 articles"
