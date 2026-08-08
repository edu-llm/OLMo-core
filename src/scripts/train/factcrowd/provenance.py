"""
What code and what environment produced an artefact.

Every fingerprint in this project answers "is this the same corpus?". None of them answers "is this the
same *code*?", and that is the question a reader of a finished table actually has: two checkpoints with
identical schema, vocabulary and renderer digests can still come from revisions that scored the endpoint
differently, and nothing in the record would say so.

Small on purpose. The commit is the useful 90% -- the image digest and run id are recorded when the
platform exports them and left absent otherwise, rather than being guessed at.

A helper here rather than in :mod:`factcrowd.train_cell` and :mod:`factcrowd.score_run` separately: the
second copy of a "read the commit" function is the one that keeps working after the first is fixed.
"""

import os
import subprocess
from typing import Dict, Optional

__all__ = ["commit", "is_dirty", "record", "ENVIRONMENT_KEYS"]


ENVIRONMENT_KEYS: Dict[str, str] = {
    "EDULLM_RUN_ID": "run_id",
    "EDULLM_IMAGE_DIGEST": "image_digest",
    "AWS_BATCH_JOB_ID": "batch_job_id",
    "AWS_BATCH_JOB_ARRAY_INDEX": "fanout_index",
}
"""
Platform variables worth keeping, mapped to the names they get in the record.

The fan-out index is here because it is the only thing distinguishing two cells of one submission in the
platform's own logs, and a run that turns out to have trained the wrong cell is diagnosed by comparing it
against the config directory's ordering.
"""


def _git(*args: str) -> Optional[str]:
    """
    Run a git command, returning ``None`` on any failure.

    Failure is normal, not exceptional: the training image clones without history depth in some
    configurations, and scoring may run from a directory that is not a checkout at all. A missing commit
    is worth recording as missing; it is not worth an exception on a path that is otherwise pure
    bookkeeping.
    """
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def commit() -> Optional[str]:
    """
    The commit this code came from, or ``None`` if nothing can say.

    **The platform is asked before git, and that ordering is the fix.** The runtime image excludes
    ``.git``, so inside a run every git command fails and the field came back empty on every checkpoint
    this project has written -- which is exactly the case the field exists for. The platform injects
    ``EDULLM_COMMIT_SHA``, which is authoritative there: it is the commit the image was built from, and it
    is what the lineage record already seals.

    Git remains the fallback for a laptop, where the environment variable is absent and the working tree
    is the truth.

    :returns: The hash, full from the platform and short from git.
    """
    from_platform = os.environ.get("EDULLM_COMMIT_SHA")
    if from_platform:
        return from_platform
    return _git("rev-parse", "--short", "HEAD") or None


def is_dirty() -> Optional[bool]:
    """
    Whether the working tree has uncommitted changes, or ``None`` if that cannot be determined.

    Worth recording separately from the commit, because a dirty tree means the commit does **not**
    identify the code that ran -- and a launch path that allows ``allow_dirty=True`` makes that reachable
    rather than hypothetical.

    :returns: True when dirty.
    """
    if os.environ.get("EDULLM_COMMIT_SHA"):
        # On the platform the tree is a fresh clone of that commit, so it is clean by construction --
        # and git cannot be asked anyway, because the image carries no `.git`.
        return False
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


def record() -> Dict[str, object]:
    """
    Everything known about how this process was produced.

    :returns: A JSON-serialisable record. Keys whose value is unknown are omitted rather than set to
        ``None``, so a reader can distinguish "not recorded" from "recorded as absent" -- and so an older
        record does not gain empty columns when a newer key is added.
    """
    out: Dict[str, object] = {}
    head = commit()
    if head is not None:
        out["commit"] = head
    dirty = is_dirty()
    if dirty is not None:
        out["dirty"] = dirty
    for variable, key in ENVIRONMENT_KEYS.items():
        value = os.environ.get(variable)
        if value:
            out[key] = value
    return out
