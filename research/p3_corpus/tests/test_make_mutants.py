"""Regression tests for the corpus mutant generator."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts import build_p3_generation as generation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "make_mutants.py"
ALL_MUTANTS = {
    "m1_empty_stmt.jsonl",
    "m2_heldout_cited.jsonl",
    "m3_name_clash.jsonl",
    "m4_degenerate_target.jsonl",
    "m5_bad_mask.jsonl",
    "m6_heldout_proof.jsonl",
}
EMPTY_HELDOUT_MUTANTS = ALL_MUTANTS - {
    "m2_heldout_cited.jsonl",
    "m6_heldout_proof.jsonl",
}


def _write_accepted_enigma_shard(tmp_path: Path) -> Path:
    template, source_manifest = generation.synthetic_family_record("enigma")
    rows = []
    for index in range(100):
        row = copy.deepcopy(template)
        row["id"] = f"enigma-fixture-{index:03d}"
        generation.validate_family_record(
            row,
            family="enigma",
            source_manifest=source_manifest,
            location=f"enigma.jsonl:{index + 1}",
        )
        rows.append(row)

    shard = tmp_path / "enigma.jsonl"
    shard.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return shard


def _run_mutants(shard: Path, heldout: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--shard",
            str(shard),
            "--heldout",
            str(heldout),
            "--out",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _mutant_bytes(output: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(output.glob("*.jsonl"))}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_empty_heldout_skips_only_dependent_mutants_deterministically(tmp_path: Path) -> None:
    shard = _write_accepted_enigma_shard(tmp_path)
    heldout = tmp_path / "heldout.json"
    heldout.write_text(
        json.dumps(
            {
                "facts": [],
                "seed": 20260801,
                "corpus": "enigma",
                "family": "atp",
                "shards": ["enigma"],
                "statement_hashes": [],
                "canonicalization": {
                    "family": "atp",
                    "scheme": "tptp-layout-v2",
                    "version": 2,
                },
                "policy": (
                    "premises cited 1-2x; citing proofs, all alternate proofs, and exact "
                    "statement aliases removed"
                ),
            }
        ),
        encoding="utf-8",
    )
    original_heldout = heldout.read_bytes()

    first_output = tmp_path / "mutants-first"
    first = _run_mutants(shard, heldout, first_output)

    assert first.returncode == 0, first.stderr
    assert set(_mutant_bytes(first_output)) == EMPTY_HELDOUT_MUTANTS
    assert heldout.read_bytes() == original_heldout
    assert "wrote 4 mutants of 100 rows" in first.stdout
    assert (
        "skipped 2 heldout-dependent mutants because heldout facts are empty: "
        "m2_heldout_cited, m6_heldout_proof"
    ) in first.stdout

    second_output = tmp_path / "mutants-second"
    second = _run_mutants(shard, heldout, second_output)

    assert second.returncode == 0, second.stderr
    assert _mutant_bytes(second_output) == _mutant_bytes(first_output)


def test_nonempty_heldout_still_writes_all_six_mutants(tmp_path: Path) -> None:
    shard = _write_accepted_enigma_shard(tmp_path)
    heldout = tmp_path / "heldout.json"
    heldout_facts = ["heldout:first", "heldout:second"]
    heldout.write_text(json.dumps({"facts": heldout_facts}), encoding="utf-8")
    output = tmp_path / "mutants"

    result = _run_mutants(shard, heldout, output)

    assert result.returncode == 0, result.stderr
    assert set(_mutant_bytes(output)) == ALL_MUTANTS
    assert f"wrote 6 mutants of 100 rows to {output}" in result.stdout
    assert "skipped" not in result.stdout

    cited_mutant = _read_jsonl(output / "m2_heldout_cited.jsonl")
    assert cited_mutant[20]["cited"][-1] == heldout_facts[0]
    assert cited_mutant[20]["facts"][heldout_facts[0]] == "leaked stmt"

    proof_mutant = _read_jsonl(output / "m6_heldout_proof.jsonl")
    assert proof_mutant[60]["theorem"] == heldout_facts[1]
