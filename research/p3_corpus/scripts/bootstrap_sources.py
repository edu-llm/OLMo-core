#!/usr/bin/env python3
"""Download and verify every upstream source pinned in source-lock.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ARCHIVE_ROOT / "source-lock.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def verify_archive(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def extract_tar(archive: Path, dest: Path, gzip: bool) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    mode = "r:gz" if gzip else "r:"
    with tarfile.open(archive, mode) as tar:
        tar.extractall(dest, filter="data")


def fetch_hf_file(dataset: str, revision: str, hf_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{hf_path}"
    download(url, dest)


def bootstrap_metamath(root: Path, spec: dict, *, download_only: bool) -> None:
    for item in spec["files"]:
        dest = root / item["relative_path"]
        if dest.exists() and sha256_file(dest) == item["sha256"]:
            print(f"OK existing {dest.relative_to(root)}")
            continue
        print(f"fetch {item['url']}")
        download(item["url"], dest)
        verify_archive(dest, item["sha256"])


def bootstrap_archives(root: Path, archives: list[dict], *, download_only: bool) -> None:
    for item in archives:
        if item.get("optional"):
            continue
        dest = root / item["relative_path"]
        if dest.exists() and sha256_file(dest) == item["sha256"]:
            print(f"OK existing {dest.relative_to(root)}")
        else:
            print(f"fetch {item['url']}")
            download(item["url"], dest)
            verify_archive(dest, item["sha256"])
        if download_only:
            continue
        extract = item.get("extract")
        if not extract:
            continue
        extract_dest = root / extract["relative_path"]
        if extract_dest.exists():
            print(f"skip extract {extract_dest.relative_to(root)}")
            continue
        gzip = extract["kind"] == "tar.gz"
        print(f"extract {dest.name} -> {extract_dest.relative_to(root)}")
        extract_tar(dest, extract_dest, gzip)


def bootstrap_isabelle(root: Path, spec: dict, *, download_only: bool) -> None:
    item = spec["files"][0]
    dest = root / item["relative_path"]
    if dest.exists() and sha256_file(dest) == item["sha256"]:
        print(f"OK existing {dest.relative_to(root)}")
        return
    print(f"fetch HF {spec['dataset']}@{spec['revision']}:{item['hf_path']}")
    fetch_hf_file(spec["dataset"], spec["revision"], item["hf_path"], dest)
    verify_archive(dest, item["sha256"])


def build_mizar_index(root: Path, lock: dict) -> None:
    spec = lock["sources"]["mizar_current"]
    derived = spec["derived"]["semantic_index_sqlite"]
    sqlite_path = root / derived["relative_path"]
    if sqlite_path.exists() and sha256_file(sqlite_path) == derived["sha256"]:
        print(f"OK existing {sqlite_path.relative_to(root)}")
        return
    mml = root / "extract-mizar/mml"
    html = root / "extract-html-current/html"
    thproofs = root / "extract-thproofs/thproofs"
    for path in (mml, html, thproofs):
        if not path.exists():
            raise SystemExit(f"missing extracted tree: {path}")
    manifest = ARCHIVE_ROOT / spec["manifest"]
    script = ARCHIVE_ROOT / "scripts/mizar_current_index.py"
    cmd = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest),
        "--mml",
        str(mml),
        "--html",
        str(html),
        "--thproofs",
        str(thproofs),
        "--sqlite",
        str(sqlite_path),
        "--jsonl",
        str(sqlite_path.with_suffix(".jsonl")),
        "--report",
        str(sqlite_path.with_suffix(".report.json")),
        "--mizar-archive",
        str(root / "archives/mizar-8.1.15_5.94.1493-i386-linux.tar"),
        "--html-archive",
        str(root / "archives/html-abstr-8.1.15_5.94.1493.tar.gz"),
        "--thproofs-archive",
        str(root / "archives/thproofs-8.1.15_5.94.1493.tar.gz"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARCHIVE_ROOT / "scripts")
    print("run", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ARCHIVE_ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("P3_SOURCES_ROOT", "/tmp/p3-sources-bootstrap")),
        help="Directory to populate with verified upstream bytes",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=DEFAULT_LOCK,
        help="Path to source-lock.json",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Fetch and hash-check archives without extraction or index build",
    )
    parser.add_argument(
        "--build-mizar-index",
        action="store_true",
        help="After extraction, run mizar_current_index.py when trees are present",
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"bootstrap root: {root}")

    bootstrap_metamath(root, lock["sources"]["metamath"], download_only=args.download_only)
    bootstrap_archives(
        root,
        lock["sources"]["mizar_current"]["archives"],
        download_only=args.download_only,
    )
    bootstrap_archives(
        root, lock["sources"]["prf2"]["archives"], download_only=args.download_only
    )
    bootstrap_archives(
        root, lock["sources"]["enigma"]["archives"], download_only=args.download_only
    )
    bootstrap_isabelle(root, lock["sources"]["isabelle"], download_only=args.download_only)

    if args.build_mizar_index and not args.download_only:
        build_mizar_index(root, lock)

    print("BOOTSTRAP_OK")


if __name__ == "__main__":
    main()
