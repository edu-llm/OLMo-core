"""Soundness regressions for compressed Metamath source-proof replay."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mm_expand import Incomplete, MM, expand  # noqa: E402, I001


FIXTURES = Path(__file__).parent / "fixtures" / "metamath"


def test_parser_preserves_active_and_mandatory_disjoint_pairs(tmp_path: Path) -> None:
    source = tmp_path / "frame.mm"
    source.write_text(
        """
$c setvar |- P $.
$v x y z $.
vx $f setvar x $.
vy $f setvar y $.
vz $f setvar z $.
${
  $d x y z $.
  pair $a |- P x y $.
$}
""",
        encoding="utf-8",
    )

    mm = MM().parse(source)
    frame = mm.assertion_frames["pair"]

    assert frame.active_disjoint == {
        frozenset(("x", "y")),
        frozenset(("x", "z")),
        frozenset(("y", "z")),
    }
    assert frame.mandatory_disjoint == {frozenset(("x", "y"))}
    assert {"x", "y", "z"} <= mm.variables


def test_expander_rejects_substituted_essential_hypothesis_mismatch() -> None:
    mm = MM().parse(FIXTURES / "invalid_essential.mm")

    with pytest.raises(Incomplete, match="essential hypothesis mismatch"):
        expand(mm, "bad")


def test_expander_rejects_disjoint_substitutions_with_same_variable() -> None:
    mm = MM().parse(FIXTURES / "invalid_disjoint.mm")

    with pytest.raises(Incomplete, match="disjoint variable violation"):
        expand(mm, "bad")


def test_expander_accepts_authorized_disjoint_substitutions(tmp_path: Path) -> None:
    source = tmp_path / "valid-disjoint.mm"
    source.write_text(
        """
$c setvar |- P $.
$v x y z w $.
vx $f setvar x $.
vy $f setvar y $.
vz $f setvar z $.
vw $f setvar w $.
${
  $d x y $.
  pair $a |- P x y $.
$}
${
  $d z w $.
  good $p |- P z w $= ( pair ) ABC $.
$}
""",
        encoding="utf-8",
    )

    mm = MM().parse(source)
    expr, _, _, trace = expand(mm, "good")

    assert expr == ["|-", "P", "z", "w"]
    assert trace[-1][1] == expr


@pytest.mark.parametrize(
    ("proof", "reason"),
    [
        ("( ) Z", "save with empty stack"),
        ("( missing ) AB", "unknown proof label"),
        ("?", "proof contains \\?"),
        ("wph", "unsupported uncompressed"),
    ],
)
def test_expander_rejects_malformed_or_unsupported_proofs(
    tmp_path: Path, proof: str, reason: str
) -> None:
    source = tmp_path / "malformed.mm"
    source.write_text(
        f"""
$c wff |- $.
$v ph $.
wph $f wff ph $.
bad $p |- ph $= {proof} $.
""",
        encoding="utf-8",
    )

    mm = MM().parse(source)
    with pytest.raises(Incomplete, match=reason):
        expand(mm, "bad")


def test_expander_rejects_stack_underflow_and_final_expression_mismatch(
    tmp_path: Path,
) -> None:
    underflow = tmp_path / "underflow.mm"
    underflow.write_text(
        """
$c wff |- T $.
$v ph $.
wph $f wff ph $.
uses-ph $a |- T ph $.
bad $p |- T $= ( uses-ph ) A $.
""",
        encoding="utf-8",
    )
    mismatch = tmp_path / "mismatch.mm"
    mismatch.write_text(
        """
$c wff |- $.
$v ph $.
wph $f wff ph $.
bad $p |- ph $= ( ) A $.
""",
        encoding="utf-8",
    )

    with pytest.raises(Incomplete, match="stack underflow"):
        expand(MM().parse(underflow), "bad")
    with pytest.raises(Incomplete, match="final expression mismatch"):
        expand(MM().parse(mismatch), "bad")
