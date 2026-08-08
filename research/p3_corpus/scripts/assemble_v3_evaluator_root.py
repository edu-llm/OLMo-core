#!/usr/bin/env python3
"""Create a canonical run_eval.py view over the exact sealed v3 JSONLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEAL = (
    REPO_ROOT / ".p3-work/full13/sealed-corpus-v3/sealed-corpus-manifest.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "corpus-v3"
FAMILIES = ("metamath", "mizar", "thproofs", "prf2", "enigma", "isabelle")
SCHEMAS = {
    "metamath": "metamath-proof-v2",
    "mizar": "mizar-proof-v2",
    "thproofs": "mizar-proof-v2",
    "prf2": "atp-v2",
    "enigma": "atp-v2",
    "isabelle": "isabelle-transition-v2",
}
SEAL_SCHEMA = "p3-sealed-corpus-manifest-v1"
EVAL_SCHEMA = "p3-evaluator-corpus-v1"
EXPECTED_TRAIN_ROWS = 181_652
EXPECTED_EVAL_ROWS = 4_191
SEPARATOR = "---\nGOAL"
SIDECARS = (
    "heldout/atp.json",
    "heldout/isabelle.json",
    "heldout/metamath.json",
    "heldout/mizar.json",
    "metamath_sources.json",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_dict(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_object(path: Path, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def scan_jsonl(path: Path, *, family: str, role: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{family}/{role} is missing or unsafe: {path}")
    digest = hashlib.sha256()
    ids: set[str] = set()
    rows = 0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{family}/{role}:{line_number} is invalid JSON"
                ) from error
            if not isinstance(row, dict) or row.get("schema_version") != SCHEMAS[family]:
                raise RuntimeError(f"{family}/{role}:{line_number} schema drift")
            missing = [
                key
                for key in ("id", "facts", "goal", "target", "text", "mask_start", "mask_end")
                if key not in row
            ]
            if missing:
                raise RuntimeError(f"{family}/{role}:{line_number} lacks {missing}")
            row_id = row["id"]
            if not isinstance(row_id, str) or not row_id or row_id in ids:
                raise RuntimeError(f"{family}/{role}:{line_number} invalid/duplicate id")
            ids.add(row_id)
            if not isinstance(row["facts"], dict) or not row["facts"]:
                raise RuntimeError(f"{family}/{role}:{line_number} has no facts")
            if not isinstance(row["text"], str) or row["text"].count(SEPARATOR) != 1:
                raise RuntimeError(f"{family}/{role}:{line_number} separator drift")
            rows += 1
    if not rows:
        raise RuntimeError(f"{family}/{role} is empty")
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size, "rows": rows}


def _source(record: Mapping, *, family: str, role: str) -> tuple[Path, dict]:
    path = Path(str(record.get("path", ""))).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"sealed {family}/{role} path is not absolute")
    observed = scan_jsonl(path, family=family, role=role)
    expected = {
        "sha256": record.get("sha256"),
        "bytes": record.get("bytes"),
        "rows": record.get("rows"),
    }
    if observed != expected:
        raise RuntimeError(
            f"sealed {family}/{role} source drift: {expected} != {observed}"
        )
    return path.resolve(strict=True), observed


def load_sources(
    seal_path: Path, *, expected_train_rows: int, expected_eval_rows: int
) -> tuple[dict, dict[str, dict[str, tuple[Path, dict]]]]:
    seal = read_object(seal_path, "sealed corpus manifest")
    families = seal.get("families")
    if seal.get("schema_version") != SEAL_SCHEMA:
        raise RuntimeError("sealed corpus manifest schema is not recognized")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise RuntimeError("sealed manifest must contain exactly six P3 families")
    body = {"schema_version": SEAL_SCHEMA, "families": families}
    if fingerprint_dict(body) != seal.get("manifest_root_sha256"):
        raise RuntimeError("sealed manifest root mismatch")

    sources: dict[str, dict[str, tuple[Path, dict]]] = {}
    for family in FAMILIES:
        record = families[family]
        if not isinstance(record, dict) or record.get("schema") != SCHEMAS[family]:
            raise RuntimeError(f"sealed {family} schema binding is invalid")
        sources[family] = {
            role: _source(record.get(role, {}), family=family, role=role)
            for role in ("train", "eval")
        }
    train_rows = sum(sources[family]["train"][1]["rows"] for family in FAMILIES)
    eval_rows = sum(sources[family]["eval"][1]["rows"] for family in FAMILIES)
    if train_rows != expected_train_rows or seal.get("total_train_rows") != train_rows:
        raise RuntimeError(f"sealed train row total drift: {train_rows}")
    if eval_rows != expected_eval_rows or seal.get("total_eval_rows") != eval_rows:
        raise RuntimeError(f"sealed eval row total drift: {eval_rows}")
    return seal, sources


def sidecar_sources(
    sources: Mapping[str, Mapping[str, tuple[Path, dict]]],
) -> dict[str, Path]:
    mml = sources["mizar"]["eval"][0].parent.parent
    metamath = sources["metamath"]["eval"][0].parent.parent
    isabelle = sources["isabelle"]["eval"][0].parent.parent
    return {
        "heldout/atp.json": mml / "heldout/atp.json",
        "heldout/mizar.json": mml / "heldout/mizar.json",
        "heldout/metamath.json": metamath / "split-build/heldout/metamath.json",
        "heldout/isabelle.json": isabelle / "split-build/heldout/isabelle.json",
        "metamath_sources.json": metamath / "split-build/metamath_sources.json",
    }


def scan_sidecar(path: Path, relative: str) -> dict:
    value = read_object(path, relative)
    if relative.startswith("heldout/"):
        facts = value.get("facts")
        if (
            not isinstance(facts, list)
            or not facts
            or any(not isinstance(fact, str) or not fact for fact in facts)
            or len(set(facts)) != len(facts)
        ):
            raise RuntimeError(f"{relative} must contain unique held-out facts")
    return {"sha256": file_sha256(path), "bytes": path.stat().st_size}


def family_records(sources: Mapping[str, Mapping[str, tuple[Path, dict]]]) -> dict:
    return {
        family: {
            "schema": SCHEMAS[family],
            "train": {
                "path": f"shards/{family}.jsonl",
                **sources[family]["train"][1],
            },
            "eval": {
                "path": f"eval/{family}.jsonl",
                **sources[family]["eval"][1],
            },
        }
        for family in FAMILIES
    }


def manifest_body(
    *,
    source_seal: dict,
    families: dict,
    sidecars: dict,
    total_train_rows: int,
    total_eval_rows: int,
) -> dict:
    return {
        "schema_version": EVAL_SCHEMA,
        "source_seal": source_seal,
        "families": families,
        "sidecars": sidecars,
        "total_train_rows": total_train_rows,
        "total_eval_rows": total_eval_rows,
    }


def expected_inventory() -> set[str]:
    return {
        *(f"shards/{family}.jsonl" for family in FAMILIES),
        *(f"eval/{family}.jsonl" for family in FAMILIES),
        *SIDECARS,
        "evaluator_manifest.json",
        "README.md",
    }


def write_readme(output: Path, manifest: Mapping) -> None:
    output.joinpath("README.md").write_text(
        "\n".join(
            (
                "# Formal Proof Premises v3 evaluator corpus",
                "",
                "Hardlink projection for `run_eval.py`; no row is transformed.",
                "",
                f"- sealed root: `{manifest['source_seal']['manifest_root_sha256']}`",
                f"- train rows: {int(manifest['total_train_rows']):,}",
                f"- eval rows: {int(manifest['total_eval_rows']):,}",
                "",
                "```bash",
                "python scripts/assemble_v3_evaluator_root.py "
                "--out corpus-v3 --check-only",
                "HF_DIR=/path/to/exported/hf",
                "ARM=dense",
                "SMOKE_JSON=/path/to/smoke.json",
                "python ../eduLLM/OLMo-core/src/scripts/train/p3_math_split/evals/run_eval.py "
                '--model "$HF_DIR" --arm "$ARM" --corpus corpus-v3 '
                '--conditions facts_present --limit 1 --out "$SMOKE_JSON"',
                "```",
                "",
                "Never use the legacy `corpus/` root for v3 evaluation.",
                "",
            )
        ),
        encoding="utf-8",
    )


def report_from_manifest(manifest: Mapping) -> dict:
    return {
        "schema_version": manifest["schema_version"],
        "families": list(FAMILIES),
        "total_train_rows": manifest["total_train_rows"],
        "total_eval_rows": manifest["total_eval_rows"],
        "evaluator_root_sha256": manifest["evaluator_root_sha256"],
    }


def validate_evaluator_root(
    output: str | Path,
    *,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_eval_rows: int = EXPECTED_EVAL_ROWS,
) -> dict:
    root = Path(output)
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"evaluator root is missing or unsafe: {root}")
    actual_paths = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"evaluator root contains a symlink: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(root).as_posix())
    if actual_paths != expected_inventory():
        raise RuntimeError(
            "evaluator inventory drift: "
            f"missing={sorted(expected_inventory() - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_inventory())}"
        )

    manifest = read_object(root / "evaluator_manifest.json", "evaluator manifest")
    if manifest.get("schema_version") != EVAL_SCHEMA:
        raise RuntimeError("evaluator manifest schema is not recognized")
    families = manifest.get("families")
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise RuntimeError("evaluator manifest family inventory drift")
    for family in FAMILIES:
        if families[family].get("schema") != SCHEMAS[family]:
            raise RuntimeError(f"{family} evaluator schema drift")
        for role, directory in (("train", "shards"), ("eval", "eval")):
            path = root / directory / f"{family}.jsonl"
            observed = scan_jsonl(path, family=family, role=role)
            expected = {
                key: families[family][role].get(key)
                for key in ("sha256", "bytes", "rows")
            }
            if observed != expected:
                raise RuntimeError(f"{family}/{role} evaluator manifest drift")
    sidecars = manifest.get("sidecars")
    if not isinstance(sidecars, dict) or set(sidecars) != set(SIDECARS):
        raise RuntimeError("evaluator sidecar inventory drift")
    for relative in SIDECARS:
        if scan_sidecar(root / relative, relative) != sidecars[relative]:
            raise RuntimeError(f"{relative} evaluator manifest drift")

    train_rows = sum(families[family]["train"]["rows"] for family in FAMILIES)
    eval_rows = sum(families[family]["eval"]["rows"] for family in FAMILIES)
    if train_rows != expected_train_rows or manifest.get("total_train_rows") != train_rows:
        raise RuntimeError(f"evaluator train row total drift: {train_rows}")
    if eval_rows != expected_eval_rows or manifest.get("total_eval_rows") != eval_rows:
        raise RuntimeError(f"evaluator eval row total drift: {eval_rows}")
    body = {key: value for key, value in manifest.items() if key != "evaluator_root_sha256"}
    if fingerprint_dict(body) != manifest.get("evaluator_root_sha256"):
        raise RuntimeError("evaluator root SHA-256 mismatch")
    return report_from_manifest(manifest)


def assemble_evaluator_root(
    seal_path: str | Path,
    output: str | Path,
    *,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_eval_rows: int = EXPECTED_EVAL_ROWS,
) -> dict:
    seal_path = Path(seal_path).expanduser().resolve(strict=True)
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"output must not already exist: {output}")

    seal, sources = load_sources(
        seal_path,
        expected_train_rows=expected_train_rows,
        expected_eval_rows=expected_eval_rows,
    )
    sidecar_paths = sidecar_sources(sources)
    sidecar_records = {
        relative: scan_sidecar(path, relative)
        for relative, path in sidecar_paths.items()
    }
    families = family_records(sources)
    source_seal = {
        "schema_version": seal["schema_version"],
        "manifest_root_sha256": seal["manifest_root_sha256"],
        "manifest_file_sha256": file_sha256(seal_path),
    }
    body = manifest_body(
        source_seal=source_seal,
        families=families,
        sidecars=sidecar_records,
        total_train_rows=expected_train_rows,
        total_eval_rows=expected_eval_rows,
    )
    manifest = {**body, "evaluator_root_sha256": fingerprint_dict(body)}

    output.parent.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(parents=True)
    (output / "eval").mkdir()
    (output / "heldout").mkdir()
    for family in FAMILIES:
        os.link(sources[family]["train"][0], output / "shards" / f"{family}.jsonl")
        os.link(sources[family]["eval"][0], output / "eval" / f"{family}.jsonl")
    for relative, source in sidecar_paths.items():
        os.link(source, output / relative)
    (output / "evaluator_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_readme(output, manifest)
    return validate_evaluator_root(
        output,
        expected_train_rows=expected_train_rows,
        expected_eval_rows=expected_eval_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed-corpus-manifest", default=str(DEFAULT_SEAL))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate an existing --out root without creating or modifying it",
    )
    args = parser.parse_args()
    if args.check_only:
        report = validate_evaluator_root(args.out)
    else:
        report = assemble_evaluator_root(args.sealed_corpus_manifest, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
