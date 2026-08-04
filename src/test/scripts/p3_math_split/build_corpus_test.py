"""Metamath corpus rendering regressions for the P3 experiment."""

from types import SimpleNamespace

from . import load_project_module

build_corpus = load_project_module("build_corpus")
mm_expand = load_project_module("mm_expand")
mm_verify = load_project_module("mm_verify")


MINI_MM = r"""
$c |- wff $.
$v ph ps $.
wph $f wff ph $.
wps $f wff ps $.
${
  ext.1 $e |- ph $.
  ext $a |- ph $.
$}
${
  dup.1 $e |- ph $.
  dup.2 $e |- ph $.
  dup $a |- ph $.
$}
${
  th.1 $e |- ph $.
  th.unused $e |- ps $.
  th $p |- ph $= ( ext dup ) AACEZGF $.
$}
"""


def test_extract_renders_only_used_local_e_hypotheses_and_removes_their_pushes(
    tmp_path,
):
    source = tmp_path / "mini.mm"
    source.write_text(MINI_MM, encoding="utf-8")
    mm = mm_expand.MM().parse(source)
    expr, _, _, source_trace = mm_expand.expand(mm, "th")
    assert expr == ["|-", "ph"]
    assert "(reuse)" in [label for label, _, _ in source_trace]
    args = SimpleNamespace(
        max_theorems=None,
        seed=1,
        min_steps=1,
        max_steps=40,
        min_facts=1,
        max_facts=8,
        max_chars=6000,
    )

    rows, tally = build_corpus.extract(mm, args)

    assert tally["kept"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert "(reuse)" not in row["target"]
    assert row["local_assumptions"] == {"th.1": "|- ph"}
    assert "th.unused" not in row["text"]
    assert row["facts"] == {
        "ext": "|- ph => |- ph",
        "dup": "|- ph & |- ph => |- ph",
    }
    assert row["cited"] == ["ext", "dup"]
    assert "th.1" not in row["target"]
    assert "ext" in row["target"]
    assert "dup" in row["target"]
    assert row["text"].index("Local assumptions:") < row["text"].index("\n---\nGOAL ")
    assert row["mask_end"] == row["text"].index("\n---\n")

    verified = mm_verify.verify_proof(
        mm,
        row["target"],
        row["goal"],
        row["facts"],
        local_assumptions=row["local_assumptions"],
        target_label="th",
    )
    assert verified.status is mm_verify.VerificationStatus.VALID, verified.reason
