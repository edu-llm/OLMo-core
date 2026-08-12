"""Keep a run's whole output, because `edullm logs` returns only the last fifty lines.

Import and call :func:`survive` first thing in an entry point. Everything the process writes to
stdout and stderr afterwards is duplicated to a local file, and that file is uploaded to
``$EDULLM_OUTPUT_PREFIX`` when the process exits -- on success, on an exception, and on a
staged refusal.

WHY THIS EXISTS. Four runs in this experiment could not be diagnosed from their logs, each for a
different reason and all the same underlying one:

  * a benchmark reported OOM for every cell including L=1024 and had discarded the exception
    message, so the reason was gone rather than merely out of view;
  * a torchrun failure printed `exitcode: 1, error_file: <N/A>` and filled the window with its
    own wrapper output, pushing the child's traceback above it;
  * an eval exited 1 with the last fifty lines ending at a model build;
  * another eval's last fifty lines were blank.

The full stream lives in CloudWatch, which nothing on the platform side but `cancel-run.yml` may
read, and that verb returns fifty lines. So the only way to keep a diagnosis is for the run to
write one somewhere the workload role can put it -- and the role's one write is under
``teams/*/runs/*``, which is exactly where ``$EDULLM_OUTPUT_PREFIX`` points.

The upload is best-effort and never raises: a run that finished its work should not fail because
its log could not be filed, and a run that already failed should not have that failure replaced.
"""

import atexit
import os
import sys
from typing import Optional, TextIO


class _Tee:
    """Write to two streams. Line-buffered on the file so a kill still leaves the tail."""

    def __init__(self, primary: TextIO, mirror: TextIO) -> None:
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        try:
            self._mirror.write(data)
            self._mirror.flush()
        except Exception:  # noqa: BLE001 -- mirroring must never break the real stream
            pass
        return n

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._mirror.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name):
        return getattr(self._primary, name)


def survive(name: str, local_dir: str = "/tmp", prefix: Optional[str] = None) -> Optional[str]:
    """Mirror stdout/stderr to a file and upload it at exit.

    :param name: used in the object key, so several runs' logs do not collide.
    :param local_dir: where the mirror file is written.
    :param prefix: an ``s3://`` prefix to upload to. Defaults to ``$EDULLM_OUTPUT_PREFIX``.

    :returns: the local path being written, or ``None`` if mirroring could not be set up.
    """
    # RANK-AWARE, because eight ranks writing one key would race and leave whichever finished
    # last. Only rank 0's stream is uploaded; the others still mirror locally, which is enough
    # to tell a per-rank failure from a global one if the machine is ever reachable.
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    path = os.path.join(local_dir, f"{name}.rank{rank}.log")
    try:
        handle = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None

    sys.stdout = _Tee(sys.stdout, handle)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.stderr, handle)  # type: ignore[assignment]

    target_prefix = prefix or os.environ.get("EDULLM_OUTPUT_PREFIX")

    def _upload() -> None:
        try:
            handle.flush()
        except Exception:  # noqa: BLE001
            pass
        if not target_prefix or rank not in ("0", 0):
            return
        try:
            from olmo_core.io import upload

            upload(path, f"{target_prefix.rstrip('/')}/{name}.log", save_overwrite=True)
        except Exception:  # noqa: BLE001 -- see the module docstring: never replace a real failure
            pass

    atexit.register(_upload)
    print(f"[SURVIVE] mirroring output to {path}; upload target {target_prefix}", flush=True)
    return path
