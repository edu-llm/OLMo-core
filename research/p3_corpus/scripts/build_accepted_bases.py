#!/usr/bin/env python3
"""Rebuild the derived ATP accepted bases that generation consumes as input.

The ENIGMA low-tier builder reads a prebuilt accepted base shard and refuses to
run unless its bytes match ``ENIGMA_LOW_TIER_SOURCE_CONTRACT["accepted_base"]``.
That shard is ~1.5 GiB, so it is rebuilt here from the archives pinned in
``source-lock.json`` rather than tracked in git.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ARCHIVE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from build_atp_shard import (  # noqa: E402
    ENIGMA_LOW_TIER_POLICY,
    ENIGMA_LOW_TIER_SOURCE_CONTRACT,
)

ENIGMA_BASE_DIRNAME = "enigma-accepted-base-v1"
READ_CHUNK_BYTES = 1 << 20


def measure_shard(path: Path) -> dict[str, object]:
    """Return the bytes/rows/sha256 triple the low-tier contract is stated in.

    :param path: The accepted base shard to measure.

    :returns: A mapping comparable to the pinned ``accepted_base`` contract.
    """
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
            rows += chunk.count(b"\n")
    return {"bytes": path.stat().st_size, "rows": rows, "sha256": digest.hexdigest()}


def enigma_source_dirs(sources_root: Path) -> list[Path]:
    """Resolve the extracted ENIGMA run directories in contract order.

    :param sources_root: Root written by ``bootstrap_sources.py``.

    :returns: One directory per run, ordered as the accepted base requires.

    :raises SystemExit: If any run directory is missing.
    """
    runs = ENIGMA_LOW_TIER_SOURCE_CONTRACT["source_order"]
    dirs = [sources_root / "extracted" / run for run in runs]
    missing = [str(path) for path in dirs if not path.is_dir()]
    if missing:
        raise SystemExit(
            "cannot reproduce the accepted ENIGMA base without every run; "
            f"missing {missing}. Re-run bootstrap_sources.py to fetch them."
        )
    return dirs


def enigma_command(*, sources_root: Path, out_dir: Path, python: str) -> list[str]:
    """Build the legacy ENIGMA argv that produced the accepted base.

    :param sources_root: Root written by ``bootstrap_sources.py``.
    :param out_dir: Directory the shard/eval/heldout tree is written under.
    :param python: Python executable used to run the builder.

    :returns: The argv, without the low-tier flags that consume this output.
    """
    return [
        python,
        str(SCRIPTS_ROOT / "build_atp_shard.py"),
        "--src",
        *[str(path) for path in enigma_source_dirs(sources_root)],
        "--name",
        "enigma",
        "--fenced",
        "--heldout",
        "0",
        "--min-steps",
        str(ENIGMA_LOW_TIER_POLICY["legacy_min_steps"]),
        "--dedup",
        "--jaccard",
        str(ENIGMA_LOW_TIER_POLICY["legacy_redundancy_jaccard"]),
        "--seed",
        str(ENIGMA_LOW_TIER_POLICY["seed"]),
        "--out",
        str(out_dir),
    ]


def build_enigma_base(
    *,
    sources_root: Path,
    work_root: Path,
    python: str,
    force: bool,
) -> Path:
    """Materialize the accepted ENIGMA base, verifying it against its pin.

    :param sources_root: Root written by ``bootstrap_sources.py``.
    :param work_root: Persistent work root holding ``atp/``.
    :param python: Python executable used to run the builder.
    :param force: Rebuild even when a matching base is already present.

    :returns: The verified accepted base directory.

    :raises SystemExit: If the rebuilt shard does not match the pinned contract.
    """
    expected = ENIGMA_LOW_TIER_SOURCE_CONTRACT["accepted_base"]
    out_dir = work_root / "atp" / ENIGMA_BASE_DIRNAME
    shard = out_dir / "shards" / "enigma.jsonl"

    if shard.is_file() and not force:
        actual = measure_shard(shard)
        if actual == expected:
            print(f"accepted ENIGMA base already matches its pin: {out_dir}")
            return out_dir
        raise SystemExit(
            f"existing accepted ENIGMA base does not match its pin: {shard}\n"
            f"  expected {expected}\n"
            f"  got      {actual}\n"
            "Pass --force to discard and rebuild it."
        )

    staging = out_dir.parent / f".{ENIGMA_BASE_DIRNAME}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SCRIPTS_ROOT}{os.pathsep}{inherited}" if inherited else str(SCRIPTS_ROOT)
    )
    cmd = enigma_command(sources_root=sources_root, out_dir=staging, python=python)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ARCHIVE_ROOT, env=env)

    staged_shard = staging / "shards" / "enigma.jsonl"
    if not staged_shard.is_file():
        raise SystemExit(f"ENIGMA base build wrote no shard: {staged_shard}")
    actual = measure_shard(staged_shard)
    if actual != expected:
        raise SystemExit(
            "rebuilt accepted ENIGMA base does not match its pin; "
            f"left in place for inspection at {staging}\n"
            f"  expected {expected}\n"
            f"  got      {actual}"
        )

    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(staging, out_dir)
    print(f"verified accepted ENIGMA base: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=Path(os.environ.get("P3_SOURCES_ROOT", "/tmp/p3-sources")),
        help="Verified upstream sources root from bootstrap_sources.py",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("P3_WORK_ROOT", "/tmp/p3-rebuild-work")),
        help="Persistent work root (ATP derived artifacts, generation work)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run build_atp_shard.py",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when a matching accepted base is already present",
    )
    args = parser.parse_args()

    build_enigma_base(
        sources_root=args.sources_root.resolve(),
        work_root=args.work_root.resolve(),
        python=args.python,
        force=args.force,
    )
    print("BUILD_ACCEPTED_BASES_OK")


if __name__ == "__main__":
    main()
