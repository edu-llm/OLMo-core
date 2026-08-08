"""Fail-closed contract for the canonical v3 evaluator-corpus projection."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASSEMBLER_PATH = ROOT / "scripts" / "assemble_v3_evaluator_root.py"
FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}


def _load_assembler():
    assert ASSEMBLER_PATH.is_file(), f"missing production assembler: {ASSEMBLER_PATH}"
    spec = importlib.util.spec_from_file_location("p3_v3_evaluator_assembler", ASSEMBLER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_row(path: Path, *, family: str, role: str) -> None:
    separator = "---\nGOAL"
    text = f"I know these mathematical statements:\nf : statement\n{separator} goal\nproof"
    _write_json(
        path,
        {
            "schema_version": SCHEMAS[family],
            "id": f"{family}-{role}",
            "facts": {"f": "statement"},
            "goal": "goal",
            "target": "proof",
            "text": text,
            "mask_start": 0,
            "mask_end": text.index("---"),
        },
    )


def _source_entry(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": 1,
    }


@pytest.fixture
def sealed_sources(tmp_path):
    sources = tmp_path / "sources"
    mml = sources / "mml-semantic-holdout-v7"
    builders = sources / "generation-work" / "builders"
    paths: dict[str, dict[str, Path]] = {}

    for family in ("mizar", "thproofs", "prf2", "enigma"):
        paths[family] = {
            "train": mml / "shards" / f"{family}.jsonl",
            "eval": mml / "eval" / f"{family}.jsonl",
        }
    for family in ("metamath", "isabelle"):
        family_root = builders / family
        paths[family] = {
            "train": family_root / "normalized-resume" / "train.jsonl",
            "eval": family_root / "normalized-resume" / "eval.jsonl",
        }

    for family in FAMILIES:
        for role, path in paths[family].items():
            _write_row(path, family=family, role=role)

    for name in ("atp", "mizar"):
        _write_json(mml / "heldout" / f"{name}.json", {"facts": [f"{name}-held"]})
    for family in ("metamath", "isabelle"):
        _write_json(
            builders / family / "split-build" / "heldout" / f"{family}.json",
            {"facts": [f"{family}-held"]},
        )
    _write_json(
        builders / "metamath" / "split-build" / "metamath_sources.json",
        {"schema_version": "metamath-source-manifest-v1", "sources": {}},
    )

    families = {
        family: {
            "schema": SCHEMAS[family],
            "train": _source_entry(paths[family]["train"]),
            "eval": _source_entry(paths[family]["eval"]),
        }
        for family in FAMILIES
    }
    body = {"schema_version": "p3-sealed-corpus-manifest-v1", "families": families}
    seal = {
        **body,
        "manifest_root_sha256": _fingerprint(body),
        "total_train_rows": len(FAMILIES),
        "total_eval_rows": len(FAMILIES),
        "tokenizer": {"path": "/sealed/tokenizer", "files": {}},
    }
    seal_path = sources / "sealed-corpus-manifest.json"
    _write_json(seal_path, seal)
    return seal_path, paths


def test_assembles_exact_hardlinked_run_eval_layout(sealed_sources, tmp_path):
    assembler = _load_assembler()
    seal_path, source_paths = sealed_sources
    output = tmp_path / "corpus-v3"

    report = assembler.assemble_evaluator_root(
        seal_path,
        output,
        expected_train_rows=len(FAMILIES),
        expected_eval_rows=len(FAMILIES),
    )

    expected_files = {
        *(f"shards/{family}.jsonl" for family in FAMILIES),
        *(f"eval/{family}.jsonl" for family in FAMILIES),
        "heldout/atp.json",
        "heldout/isabelle.json",
        "heldout/metamath.json",
        "heldout/mizar.json",
        "metamath_sources.json",
        "evaluator_manifest.json",
        "README.md",
    }
    assert {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    } == expected_files
    for family in FAMILIES:
        assert os.path.samefile(output / "shards" / f"{family}.jsonl", source_paths[family]["train"])
        assert os.path.samefile(output / "eval" / f"{family}.jsonl", source_paths[family]["eval"])

    assert report["total_train_rows"] == len(FAMILIES)
    assert report["total_eval_rows"] == len(FAMILIES)
    assert report["families"] == list(FAMILIES)
    validated = assembler.validate_evaluator_root(
        output,
        expected_train_rows=len(FAMILIES),
        expected_eval_rows=len(FAMILIES),
    )
    assert validated["evaluator_root_sha256"] == report["evaluator_root_sha256"]
    readme = (output / "README.md").read_text(encoding="utf-8")
    run_eval_path = (
        "../eduLLM/OLMo-core/src/scripts/train/p3_math_split/evals/run_eval.py"
    )
    assert "--corpus corpus-v3" in readme
    assert "assemble_v3_evaluator_root.py --out corpus-v3 --check-only" in readme
    assert run_eval_path in readme
    assert "p3_math_split/run_eval.py" not in readme
    assert "verify_corpus.py" not in readme
    assert "4,191" not in readme  # synthetic tests must not fabricate production counts
    for block in re.findall(r"```bash\n(.*?)```", readme, flags=re.S):
        check = subprocess.run(
            ["bash", "-n", "-c", block],
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stderr


def test_refuses_existing_output_without_modifying_it(sealed_sources, tmp_path):
    assembler = _load_assembler()
    seal_path, _ = sealed_sources
    output = tmp_path / "corpus-v3"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must not already exist"):
        assembler.assemble_evaluator_root(
            seal_path,
            output,
            expected_train_rows=len(FAMILIES),
            expected_eval_rows=len(FAMILIES),
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_refuses_source_drift_before_creating_output(sealed_sources, tmp_path):
    assembler = _load_assembler()
    seal_path, source_paths = sealed_sources
    source_paths["mizar"]["eval"].write_text("{}\n", encoding="utf-8")
    output = tmp_path / "corpus-v3"

    with pytest.raises(RuntimeError, match="drift"):
        assembler.assemble_evaluator_root(
            seal_path,
            output,
            expected_train_rows=len(FAMILIES),
            expected_eval_rows=len(FAMILIES),
        )

    assert not output.exists()
