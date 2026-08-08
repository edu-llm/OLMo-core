"""
Pinned upstream provenance for the HPO controller.

Every externally sourced artifact the controller depends on is pinned here so a run is
reproducible and the FT-PFN weight file can be checksum-verified before a trusted load.
This module is deliberately dependency-free (standard library only).

Provenance
----------
- **ifBO** (FT-PFN surrogate + MFPI-random reference): MIT licensed, pinned to release
  ``v0.4.1`` at commit :data:`IFBO_COMMIT`. The FT-PFN posterior code is used directly;
  the MFPI-random controller logic is re-implemented (see :mod:`olmo_core.hpo.ifbo`).
- **FT-PFN weights** (``bopfn_broken_unisep_1000curves_10params_2M``): distributed under
  CC BY 4.0 via Figshare. Attribution and any modification must be recorded when the
  artifact is redistributed.
- **NePS** (reference for MFPI-random / fantasization): Apache-2.0; only algorithm logic
  is adapted, with NOTICE/modification attribution, not imported wholesale.
- **BTTackler**, **IPBT**, **Centaur**: see the corresponding submodules for their
  provenance and clean-room/reuse boundaries.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO
from urllib.request import urlopen

__all__ = [
    "IFBO_PACKAGE_VERSION",
    "IFBO_COMMIT",
    "IFBO_CODE_LICENSE",
    "FTPFN_MODEL_VERSION",
    "FTPFN_ARCHIVE_URL",
    "FTPFN_ARCHIVE_FILENAME",
    "FTPFN_ARTIFACT_FILENAME",
    "FTPFN_ARCHIVE_MD5",
    "FTPFN_ARCHIVE_SHA256",
    "FTPFN_ARTIFACT_MD5",
    "FTPFN_ARTIFACT_SHA256",
    "FTPFN_ARTIFACT_LICENSE",
    "FTPFN_ARTIFACT_SOURCE",
    "FTPFN_MAX_HP_DIMS",
    "FTPFN_MAX_CONTEXT_CURVES",
    "default_ftpfn_cache_dir",
    "ensure_ftpfn_artifact",
]

# --- ifBO package (code) ---
IFBO_PACKAGE_VERSION: str = "0.4.1"
"""The exact ``ifBO`` PyPI release the controller is validated against."""

IFBO_COMMIT: str = "8ddcef0ed1ca88f2992108d39876e926aa58b0f2"
"""The upstream commit that ``v0.4.1`` points to."""

IFBO_CODE_LICENSE: str = "MIT"
"""License of the ``ifBO`` source code."""

# --- FT-PFN weight artifact (data) ---
FTPFN_MODEL_VERSION: str = "0.0.1"
"""The FT-PFN surrogate weight version. Distinct from the package version."""

FTPFN_ARCHIVE_URL: str = "https://api.figshare.com/v2/file/download/61709839"
"""Official public download URL in ifBO's current model registry."""

FTPFN_ARCHIVE_FILENAME: str = f"ftpfnv{FTPFN_MODEL_VERSION}.tar.gz"
"""Published Figshare archive filename."""

FTPFN_ARTIFACT_FILENAME: str = "bopfn_broken_unisep_1000curves_10params_2M.pt"
"""Checkpoint filename inside the published archive."""

FTPFN_ARCHIVE_MD5: str = "eb7567eaae91f2a958bf81083655f97b"
"""Figshare's published MD5 for file 61709839 (``ftpfnv0.0.1.tar.gz``)."""

FTPFN_ARCHIVE_SHA256: str = "989bc724e832b272f2608c0204cc0ed4f2728dfa835a2525b5eed275236c12d4"
"""Locally verified SHA-256 of the published archive."""

FTPFN_ARTIFACT_MD5: str = "d857292ca08c31fa18805e66e83e3437"
"""Locally verified MD5 of ``bopfn_broken_unisep_1000curves_10params_2M.pt``."""

FTPFN_ARTIFACT_SHA256: str = "2626a7955f6c607008e979dcf8bf4cd524c0b6dc696de7e415f58d616c814c69"
"""Locally verified SHA-256 of the extracted checkpoint."""

FTPFN_ARTIFACT_LICENSE: str = "CC-BY-4.0"
"""License of the downloaded FT-PFN weights."""

FTPFN_ARTIFACT_SOURCE: str = (
    "https://figshare.com/articles/dataset/IfBO_surrogate-compressed/31286173"
)
"""Where the FT-PFN weights are published."""

# --- FT-PFN v0.0.1 input contract ---
FTPFN_MAX_HP_DIMS: int = 10
"""Maximum number of non-fidelity hyperparameter dimensions FT-PFN v0.0.1 accepts."""

FTPFN_MAX_CONTEXT_CURVES: int = 1000
"""Maximum number of distinct configurations FT-PFN v0.0.1 can hold in context."""


def _digests(stream: BinaryIO) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b""):
        md5.update(chunk)
        sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _verify_file(
    path: Path,
    *,
    expected_md5: str,
    expected_sha256: str,
    label: str,
) -> None:
    with path.open("rb") as stream:
        md5, sha256 = _digests(stream)
    if md5 != expected_md5 or sha256 != expected_sha256:
        raise ValueError(
            f"{label} checksum mismatch: MD5 {md5} "
            f"(expected {expected_md5}), SHA-256 {sha256} "
            f"(expected {expected_sha256})"
        )


def default_ftpfn_cache_dir() -> Path:
    """Return the user-overridable cache for the pinned public FT-PFN model."""
    override = os.getenv("OLMO_CORE_HPO_ARTIFACT_CACHE")
    if override:
        root = Path(override).expanduser()
    else:
        root = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
        root = root / "olmo_core" / "hpo"
    return root / f"ftpfn-v{FTPFN_MODEL_VERSION}"


def ensure_ftpfn_artifact(
    cache_dir: str | Path | None = None,
) -> Path:
    """Provision and verify the official public FT-PFN checkpoint.

    Existing cache entries are trusted only after both pinned checksums pass. A
    missing artifact is streamed atomically from the official Figshare URL and
    only the expected regular checkpoint member is extracted from the archive.
    """
    cache = default_ftpfn_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    artifact_path = cache / FTPFN_ARTIFACT_FILENAME
    archive_path = cache / FTPFN_ARCHIVE_FILENAME

    if artifact_path.is_file():
        try:
            _verify_file(
                artifact_path,
                expected_md5=FTPFN_ARTIFACT_MD5,
                expected_sha256=FTPFN_ARTIFACT_SHA256,
                label="FT-PFN artifact",
            )
            return artifact_path
        except ValueError:
            artifact_path.unlink()

    archive_valid = False
    if archive_path.is_file():
        try:
            _verify_file(
                archive_path,
                expected_md5=FTPFN_ARCHIVE_MD5,
                expected_sha256=FTPFN_ARCHIVE_SHA256,
                label="FT-PFN archive",
            )
            archive_valid = True
        except ValueError:
            archive_path.unlink()

    if not archive_valid:
        with tempfile.NamedTemporaryFile(
            dir=cache,
            prefix=f".{FTPFN_ARCHIVE_FILENAME}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with urlopen(FTPFN_ARCHIVE_URL, timeout=60) as response:
                    shutil.copyfileobj(response, temporary, length=1 << 20)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        try:
            _verify_file(
                temporary_path,
                expected_md5=FTPFN_ARCHIVE_MD5,
                expected_sha256=FTPFN_ARCHIVE_SHA256,
                label="FT-PFN archive",
            )
            os.replace(temporary_path, archive_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    with tarfile.open(archive_path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name == FTPFN_ARTIFACT_FILENAME
        ]
        if len(members) != 1:
            raise ValueError("FT-PFN archive must contain exactly one expected checkpoint")
        source = archive.extractfile(members[0])
        if source is None:
            raise ValueError("FT-PFN checkpoint could not be read from archive")
        with source, tempfile.NamedTemporaryFile(
            dir=cache,
            prefix=f".{FTPFN_ARTIFACT_FILENAME}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_artifact = Path(temporary.name)
            while chunk := source.read(1 << 20):
                temporary.write(chunk)

    try:
        _verify_file(
            temporary_artifact,
            expected_md5=FTPFN_ARTIFACT_MD5,
            expected_sha256=FTPFN_ARTIFACT_SHA256,
            label="FT-PFN artifact",
        )
        os.replace(temporary_artifact, artifact_path)
    except BaseException:
        temporary_artifact.unlink(missing_ok=True)
        raise
    return artifact_path
