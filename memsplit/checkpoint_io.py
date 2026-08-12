"""Checkpoint IO that works for a local path or an `s3://` URI, and fails loudly.

## The bug this exists to avoid

A sibling repo on the same platform declares `resume_required: true` and then gates
its checkpoint load on `os.path.exists()` against an `s3://` URI. `Path("s3://b/k")`
never exists on a local filesystem, so the check is **always false**: every retry
silently started from step 0 and repeated the previous attempt in full, at full
price, while reporting success.

Two defences here:

1. `exists()` / `load()` / `save()` understand the `s3://` scheme, so the check
   actually asks the right storage system.
2. `ResumeGuard` writes an attempt marker *before* training. If a marker from a
   previous attempt is present but no checkpoint can be loaded, that is a lost
   checkpoint, not a fresh start, and it raises. Silently redoing work is the
   expensive failure; crashing is the cheap one.

Platform jobs cap walltime well below what a full run needs, so resume is the
normal path rather than an error path, and it has to be exact. This is also what
makes a preempted Colab session survivable -- the previous generation lost a
split-arm run at 0.87B of 1.0B tokens and then reported the 0.797B snapshot
against the other arm's 0.996B one.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch


def is_s3(uri: str | Path) -> bool:
    return str(uri).startswith("s3://")


def _split_s3(uri: str) -> tuple[str, str]:
    p = urlparse(str(uri))
    return p.netloc, p.path.lstrip("/")


def _client():
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "an s3:// checkpoint path was given but boto3 is not installed; "
            "install boto3 or use a local path"
        ) from exc
    return boto3.client("s3")


def join(root: str | Path, *parts: str) -> str:
    if is_s3(root):
        return "/".join([str(root).rstrip("/"), *parts])
    return str(Path(root).joinpath(*parts))


def exists(uri: str | Path) -> bool:
    if is_s3(uri):
        bucket, key = _split_s3(str(uri))
        try:
            _client().head_object(Bucket=bucket, Key=key)
            return True
        except Exception:
            return False
    return Path(uri).exists()


def save_obj(obj, uri: str | Path) -> None:
    """Write atomically: temp then replace (local) or single PUT (S3)."""
    if is_s3(uri):
        buf = io.BytesIO()
        torch.save(obj, buf)
        buf.seek(0)
        bucket, key = _split_s3(str(uri))
        # A PUT is atomic from a reader's point of view, so no temp key needed.
        _client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_obj(uri: str | Path, map_location="cpu"):
    if is_s3(uri):
        bucket, key = _split_s3(str(uri))
        body = _client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return torch.load(io.BytesIO(body), map_location=map_location, weights_only=False)
    return torch.load(Path(uri), map_location=map_location, weights_only=False)


def write_text(uri: str | Path, text: str) -> None:
    if is_s3(uri):
        bucket, key = _split_s3(str(uri))
        _client().put_object(Bucket=bucket, Key=key, Body=text.encode())
        return
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(uri: str | Path) -> str:
    if is_s3(uri):
        bucket, key = _split_s3(str(uri))
        return _client().get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return Path(uri).read_text(encoding="utf-8")


@dataclass
class ResumeGuard:
    """Refuse to silently restart when a previous attempt left no checkpoint."""

    root: str
    marker_name: str = "attempt.txt"
    enabled: bool = True

    @property
    def marker(self) -> str:
        return join(self.root, self.marker_name)

    def attempts_so_far(self) -> int:
        if not exists(self.marker):
            return 0
        try:
            return int(read_text(self.marker).strip() or 0)
        except ValueError:
            return 0

    def check_and_record(self, loaded_checkpoint: bool) -> int:
        """Call once at startup, after attempting a load. Returns the attempt no.

        Raises when a prior attempt is recorded but nothing was loaded -- the
        situation in which the sibling repo quietly repeated a full run.
        """
        prior = self.attempts_so_far()
        if self.enabled and prior > 0 and not loaded_checkpoint:
            raise RuntimeError(
                f"attempt {prior + 1} at {self.root!r} found no loadable "
                "checkpoint although a previous attempt is recorded. Starting "
                "from scratch here would silently repeat that work at full cost. "
                "Check the checkpoint path and credentials, or pass "
                "resume_required=False if a fresh start is genuinely intended."
            )
        write_text(self.marker, str(prior + 1))
        return prior + 1
