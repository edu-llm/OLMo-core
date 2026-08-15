#!/usr/bin/env python3
"""Seal a rebuilt corpus and project it into the v3 evaluator root.

The evaluator JSONL payload is not published to S3 and is not carried in git, so
after a rebuild it exists only if this stage runs. Two existing scripts do the
work; this one sequences them and checks the result against the identity pinned
in ``expected-release-v3.json``:

1. ``seal_v3_corpus.py``           final splits -> sealed-corpus-manifest.json
2. ``assemble_v3_evaluator_root.py`` sealed manifest -> corpus-v3/ hardlink view

A rebuilt corpus whose sealed root differs from the pin is not the v3 corpus, so
this refuses rather than emitting a look-alike evaluator root.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = ARCHIVE_ROOT / "scripts"
EXPECTED_RELEASE = ARCHIVE_ROOT / "expected-release-v3.json"


def expected_seal() -> dict[str, object]:
    """Read the sealed-corpus identity this rebuild must reproduce.

    :returns: The ``sealed_jsonl`` block of ``expected-release-v3.json``.
    """
    payload = json.loads(EXPECTED_RELEASE.read_text(encoding="utf-8"))
    return payload["sealed_jsonl"]


def run(cmd: list[str], *, env: dict[str, str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ARCHIVE_ROOT, env=env)


def seal_corpus(
    *,
    work_root: Path,
    builders_root: Path,
    out_manifest: Path,
    python: str,
    env: dict[str, str],
) -> None:
    """Freeze the rebuilt splits into a sealed corpus manifest."""
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            python,
            str(SCRIPTS_ROOT / "seal_v3_corpus.py"),
            "--work-root",
            str(work_root),
            "--builders-root",
            str(builders_root),
            "--out",
            str(out_manifest),
            "--tokenizer-dir",
            str(ARCHIVE_ROOT / "tokenizers/qwen25-vendored"),
        ],
        env=env,
    )


def check_seal_matches_pin(manifest_path: Path) -> None:
    """Refuse a sealed manifest that is not the pinned v3 identity.

    :raises SystemExit: On any drift in root digest or row totals.
    """
    expected = expected_seal()
    actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for key in ("manifest_root_sha256", "total_train_rows", "total_eval_rows"):
        if actual.get(key) != expected.get(key):
            problems.append(f"  {key}: expected {expected.get(key)!r}, got {actual.get(key)!r}")
    for family, spec in sorted(expected.get("families", {}).items()):
        got = actual.get("families", {}).get(family, {})
        if got.get("train", {}).get("sha256") != spec.get("train_sha256"):
            problems.append(
                f"  {family}.train.sha256: expected {spec.get('train_sha256')!r}, "
                f"got {got.get('train', {}).get('sha256')!r}"
            )
    if problems:
        raise SystemExit(
            "rebuilt corpus does not match the pinned v3 seal:\n" + "\n".join(problems)
        )
    print(f"sealed root matches expected-release-v3.json: {expected['manifest_root_sha256']}")


def assemble_root(
    *, manifest_path: Path, out_dir: Path, python: str, env: dict[str, str]
) -> None:
    """Project the sealed splits into the evaluator root, then re-check it."""
    run(
        [
            python,
            str(SCRIPTS_ROOT / "assemble_v3_evaluator_root.py"),
            "--sealed-corpus-manifest",
            str(manifest_path),
            "--out",
            str(out_dir),
        ],
        env=env,
    )
    run(
        [
            python,
            str(SCRIPTS_ROOT / "assemble_v3_evaluator_root.py"),
            "--sealed-corpus-manifest",
            str(manifest_path),
            "--out",
            str(out_dir),
            "--check-only",
        ],
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("P3_WORK_ROOT", "/tmp/p3-rebuild-work")),
        help="Rebuild work root holding mml-semantic-holdout-v7/ and generation work",
    )
    parser.add_argument(
        "--builders-root",
        type=Path,
        default=None,
        help=(
            "Directory holding <family>/normalized-resume/ for metamath and isabelle; "
            "defaults to <work-root>/generation-work/<generation-id>/builders"
        ),
    )
    parser.add_argument(
        "--generation-id",
        default="p3-full13-rebuild",
        help="Generation id used by the rebuild, for locating the builders root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Evaluator root to create (defaults to <work-root>/corpus-v3)",
    )
    parser.add_argument(
        "--sealed-corpus-manifest",
        type=Path,
        default=None,
        help="Sealed manifest path (defaults to <work-root>/sealed-corpus-v3/...)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the underlying scripts",
    )
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    builders_root = (
        args.builders_root.resolve()
        if args.builders_root
        else work_root / "generation-work" / args.generation_id / "builders"
    )
    out_dir = (args.out or work_root / "corpus-v3").resolve()
    manifest_path = (
        args.sealed_corpus_manifest
        or work_root / "sealed-corpus-v3" / "sealed-corpus-manifest.json"
    ).resolve()

    if not builders_root.is_dir():
        raise SystemExit(
            f"builders root not found: {builders_root}\n"
            "metamath and isabelle are sealed from <builders>/<family>/normalized-resume/. "
            "Pass --builders-root explicitly if the rebuild laid them out elsewhere."
        )

    env = os.environ.copy()
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SCRIPTS_ROOT}{os.pathsep}{inherited}" if inherited else str(SCRIPTS_ROOT)
    )

    seal_corpus(
        work_root=work_root,
        builders_root=builders_root,
        out_manifest=manifest_path,
        python=args.python,
        env=env,
    )
    check_seal_matches_pin(manifest_path)
    assemble_root(
        manifest_path=manifest_path, out_dir=out_dir, python=args.python, env=env
    )
    print(f"evaluator root ready: {out_dir}")
    print("ASSEMBLE_EVALUATOR_CORPUS_OK")


if __name__ == "__main__":
    main()
