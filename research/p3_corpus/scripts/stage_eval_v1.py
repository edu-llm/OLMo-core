#!/usr/bin/env python3
"""Stage the sealed corpus-v3 evaluator tree for edullm-data publish().

Extracts ``corpus-v3.zip`` byte-for-byte (no row transforms) into
``<stage-root>/evaluator/``, the group prefix expected by ``publish_eval_v1.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ZIP = REPO_ROOT / "corpus-v3.zip"
DEFAULT_STAGE = REPO_ROOT / ".p3-work" / "full13" / "publish-stage-eval-v1"

EXPECTED_ZIP_BYTES = 300_229_499
EXPECTED_ZIP_SHA256 = "79e79b2f9bd12fbd425926fb376ab86ebb4decf6d5ae2527e0794c4f95d28b2e"
EXPECTED_TRAIN_ROWS = 181_652
EXPECTED_EVAL_ROWS = 4_191
EXPECTED_EVALUATOR_ROOT = "e54954386d85094fc8198c6530f094db505fc88025bfd779f95b419cc74442ba"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip(zip_path: Path) -> None:
    size = zip_path.stat().st_size
    if size != EXPECTED_ZIP_BYTES:
        sys.exit(f"ABORT: {zip_path} size {size} != expected {EXPECTED_ZIP_BYTES}")
    digest = file_sha256(zip_path)
    if digest != EXPECTED_ZIP_SHA256:
        sys.exit(f"ABORT: {zip_path} sha256 {digest} != expected {EXPECTED_ZIP_SHA256}")


def _load_validator():
    import importlib.util

    assembler = REPO_ROOT / "scripts" / "assemble_v3_evaluator_root.py"
    spec = importlib.util.spec_from_file_location("p3_v3_evaluator_assembler", assembler)
    if spec is None or spec.loader is None:
        sys.exit(f"ABORT: cannot load validator from {assembler}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_from_zip(zip_path: Path, stage_root: Path, *, force: bool) -> dict:
    verify_zip(zip_path)
    evaluator_root = stage_root / "evaluator"
    if evaluator_root.exists():
        if not force:
            sys.exit(f"ABORT: stage root already exists: {evaluator_root} (pass --force to replace)")
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="corpus-v3-extract-") as tmp:
        extract_dir = Path(tmp)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
        source_root = extract_dir / "corpus-v3"
        if not source_root.is_dir():
            sys.exit(f"ABORT: zip missing corpus-v3/ root; found: {list(extract_dir.iterdir())}")

        validator = _load_validator()
        report = validator.validate_evaluator_root(
            source_root,
            expected_train_rows=EXPECTED_TRAIN_ROWS,
            expected_eval_rows=EXPECTED_EVAL_ROWS,
        )
        if report["evaluator_root_sha256"] != EXPECTED_EVALUATOR_ROOT:
            sys.exit(
                "ABORT: evaluator root mismatch: "
                f"{report['evaluator_root_sha256']} != {EXPECTED_EVALUATOR_ROOT}"
            )

        shutil.copytree(source_root, evaluator_root, symlinks=False, dirs_exist_ok=False)

    manifest = json.loads((evaluator_root / "evaluator_manifest.json").read_text(encoding="utf-8"))
    return {
        "stage_root": str(stage_root.resolve()),
        "evaluator_root_sha256": manifest["evaluator_root_sha256"],
        "source_seal_root_sha256": manifest["source_seal"]["manifest_root_sha256"],
        "total_train_rows": manifest["total_train_rows"],
        "total_eval_rows": manifest["total_eval_rows"],
        "payload_files": sum(1 for _ in evaluator_root.rglob("*") if _.is_file()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE)
    ap.add_argument("--force", action="store_true", help="replace an existing stage root")
    args = ap.parse_args()
    report = stage_from_zip(args.zip.resolve(), args.stage_root.resolve(), force=args.force)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("STAGE_OK")


if __name__ == "__main__":
    main()
