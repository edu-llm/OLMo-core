"""Regression coverage for theorem-local Metamath essential hypotheses."""

import hashlib
import json
import sys

from scripts import build_metamath_shard as builder

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


def test_builder_uses_only_local_e_pushes_and_keeps_external_fact_logic(
    tmp_path, monkeypatch
):
    """``mand`` is broad; the decoded trace decides which local givens are relevant."""
    mm_dir = tmp_path / "mm"
    mm_dir.mkdir()
    source = mm_dir / "set.mm"
    source.write_text(MINI_MM, encoding="utf-8")

    mm = builder.MM().parse(source)
    expr, mand, refs, trace = builder.expand(mm, "th")
    assert expr == ["|-", "ph"], "saved-backreference replay must still reduce"
    assert [label for kind, label, _ in mand if kind == "$e"] == [
        "th.1",
        "th.unused",
    ]
    assert "th.1" in [label for label, _, _ in trace]
    assert "th.unused" not in [label for label, _, _ in trace]
    assert "(reuse)" in [label for label, _, _ in trace]
    assert refs == ["ext", "dup"]

    monkeypatch.setattr(builder, "DBS", ("set",))
    monkeypatch.setattr(
        builder,
        "SOURCE_SHA256",
        {"set": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    out = tmp_path / "corpus"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_metamath_shard.py",
            "--mm-dir",
            str(mm_dir),
            "--out",
            str(out),
            "--heldout",
            "2",
        ],
    )

    builder.main()

    record = json.loads((out / "eval" / "metamath.jsonl").read_text())
    heldout = json.loads((out / "heldout" / "metamath.json").read_text())
    drop_ledger = json.loads(
        (out / "drops" / "metamath-overlength.json").read_text()
    )

    assert "(reuse)" not in record["target"]
    assert record["schema_version"] == "metamath-proof-v2"
    assert record["source_metadata"]["schema_version"] == (
        "metamath-build-source-v3"
    )
    builder.validate_drop_ledger(drop_ledger)
    assert drop_ledger["accounting"]["dropped_rows"] == 0
    assert record["source_metadata"]["drop_ledger"]["canonical_root_sha256"] == (
        drop_ledger["canonical_root_sha256"]
    )
    assert record["source_metadata"]["quality_filter"][
        "drop_ledger_root_sha256"
    ] == drop_ledger["canonical_root_sha256"]
    assert record["source_metadata"]["schema_generation"][
        "drop_ledger_root_sha256"
    ] == drop_ledger["canonical_root_sha256"]
    assert heldout["eligibility"]["drop_ledger"]["canonical_root_sha256"] == (
        drop_ledger["canonical_root_sha256"]
    )
    assert record["source_metadata"]["source_manifest_root_sha256"] == (
        hashlib.sha256(
            json.dumps(
                builder.source_manifest(mm_dir),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    assert record["source_metadata"]["source_roots"]
    assert record["source_metadata"]["index_roots"] == {}
    assert record["local_assumptions"] == {"th.1": "|- ph"}
    assert "th.unused" not in record["text"]
    assert record["facts"] == {
        "ext": "|- ph => |- ph",
        "dup": "|- ph & |- ph => |- ph",
    }
    assert record["cited"] == ["ext", "dup"]
    assert heldout["facts"] == ["dup", "ext"]

    assert "th.1" not in record["target"]
    assert "th.unused" not in record["target"]
    assert "ext" in record["target"]
    assert "dup" in record["target"]

    separator = "\n---\nGOAL "
    assert record["text"].count(separator) == 1
    assert record["text"].index("Local assumptions:") < record["text"].index(separator)
    assert record["mask_end"] == record["text"].index("\n---\n")
    assert record["text"][: record["mask_end"]].endswith(
        "Local assumptions:\nth.1 : |- ph"
    )
