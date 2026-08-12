"""
I/O-robustness shim -- ADDITIVE, imported for side effects. Does **NOT** modify
any OLMo-core source file; it only rebinds two callables at runtime.

Why this exists
---------------
Building the 10B water-fill mixture requires a content *fingerprint*
(:pyattr:`olmo_core.data.numpy_dataset.NumpyDatasetBase.fingerprint`) that hashes
every candidate shard's basename **and byte size**. Deriving the fingerprint
therefore HeadObjects all ~6.9k shards on *every* cold start (it must run before
the cached index can even be located). That sizing is driven by
``NumpyDatasetBase.map`` which uses ``ThreadPoolExecutor(max_workers=None)`` ->
32 threads on this 192-vCPU box.

Under that 32-wide burst, S3/STS intermittently returns a transient ``403`` while
the instance-role credentials refresh. ``_s3_file_size``'s retry budget can be
exhausted across many simultaneous in-flight requests, aborting startup with
``OLMoNetworkError`` (observed: 17 shards failing every attempt at once). A
sequential re-check of all 6.9k shards finds **zero** genuinely-forbidden objects,
confirming the failure is load/timing transient, not a permission problem.

What this changes (and what it deliberately does NOT)
-----------------------------------------------------
* Caps the *default* concurrency of ``NumpyDatasetBase.map`` so the sizing burst
  is modest -- shrinking the simultaneous-403 blast radius.
* Wraps ``get_file_size`` (as referenced inside ``numpy_dataset``) with an extra
  jittered exponential-backoff retry on transient S3 errors, so any refresh
  window is ridden out rather than fatal.

It touches **no training numerics** whatsoever -- only the parallelism and retry
policy of S3 object-size lookups performed at dataset-index build time. Training
data reads (data-loader workers) and checkpoint I/O are untouched.
"""

import functools
import logging
import random
import time

import olmo_core.data.numpy_dataset as _nd

log = logging.getLogger(__name__)

# Modest fan-out for the one-time sizing burst. 8 threads size ~6.9k shards in a
# few tens of seconds while keeping the concurrent-403 blast radius small.
_MAX_SIZE_WORKERS = 8
# Extra retry envelope layered *under* OLMo's own retriable, with real backoff.
_MAX_SIZE_RETRIES = 8
_BACKOFF_CAP_S = 8.0


def _install() -> None:
    # 1) Throttle the default map() fan-out (used by NumpyDatasetBase.file_sizes).
    _orig_map = _nd.NumpyDatasetBase.map

    @functools.wraps(_orig_map)
    def _throttled_map(self, func, *, max_workers=None, method="threads", _paths=None):
        if max_workers is None:
            max_workers = _MAX_SIZE_WORKERS
        return _orig_map(self, func, max_workers=max_workers, method=method, _paths=_paths)

    _nd.NumpyDatasetBase.map = _throttled_map  # type: ignore[method-assign]

    # 2) Resilient object-size lookup: retry transient S3/STS errors (e.g. the 403
    #    seen during instance-role credential refresh) with jittered backoff.
    #    FileNotFoundError (a genuine 404) is never retried.
    _orig_get_file_size = _nd.get_file_size

    @functools.wraps(_orig_get_file_size)
    def _resilient_get_file_size(path, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(_MAX_SIZE_RETRIES):
            try:
                return _orig_get_file_size(path, *args, **kwargs)
            except FileNotFoundError:
                raise
            except Exception as exc:  # transient 403 / throttling / conn reset
                last_exc = exc
                delay = min(_BACKOFF_CAP_S, 0.25 * (2**attempt)) * (0.5 + random.random())
                log.warning(
                    "get_file_size('%s') transient error (attempt %d/%d): %s; retrying in %.2fs",
                    path,
                    attempt + 1,
                    _MAX_SIZE_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    _nd.get_file_size = _resilient_get_file_size  # type: ignore[assignment]

    log.info(
        "s3_io_robustness installed: map() default max_workers=%d, get_file_size retries=%d",
        _MAX_SIZE_WORKERS,
        _MAX_SIZE_RETRIES,
    )


_install()
