"""Build the official Metamath verifier from an immutable pinned source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("metamath_verifier_manifest.json")


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path, checkout: Path) -> Path:
    """Build and hash-check the pinned executable.

    :param output: Destination for the verified executable.
    :param checkout: Persistent source checkout/build directory.

    :returns: The output path.
    """

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    source = manifest["source"]
    build_config = manifest["build"]
    expected_binary = manifest["binary"]

    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--no-checkout",
                source["repository"],
                str(checkout),
            ]
        )
    _run(
        [
            "git",
            "-C",
            str(checkout),
            "checkout",
            "--detach",
            source["commit"],
        ]
    )
    actual_commit = _run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
    ).stdout.strip()
    if actual_commit != source["commit"]:
        raise RuntimeError(
            f"source commit mismatch: expected {source['commit']}, got {actual_commit}"
        )

    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("the pinned build requires gcc")
    compiler_identity = _run(
        [compiler, "--version"],
        capture_output=True,
    ).stdout.splitlines()[0]
    if compiler_identity != build_config["compiler"]:
        raise RuntimeError(
            "compiler identity mismatch: expected "
            f"{build_config['compiler']!r}, got {compiler_identity!r}"
        )

    sources = sorted((checkout / "src").glob("m*.c"))
    if not sources:
        raise RuntimeError("pinned checkout has no src/m*.c files")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    flags = [
        flag.replace("<checkout>", str(checkout))
        for flag in build_config["flags"]
    ]
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = str(build_config["source_date_epoch"])
    _run(
        [
            compiler,
            *flags,
            *(str(path.relative_to(checkout)) for path in sources),
            "-o",
            str(temporary),
        ],
        env=environment,
        cwd=checkout,
    )

    digest = _sha256(temporary)
    if digest != expected_binary["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "official verifier binary hash mismatch: expected "
            f"{expected_binary['sha256']}, got {digest}"
        )
    temporary.replace(output)
    return output


def main() -> int:
    """Build the pinned verifier from the command line."""

    parser = argparse.ArgumentParser()
    default_root = Path("/tmp/p3-metamath-official")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root / "metamath",
    )
    parser.add_argument(
        "--checkout",
        type=Path,
        default=default_root / "metamath-exe",
    )
    args = parser.parse_args()
    print(build(args.output, args.checkout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
