"""Local Windows/Python 3.14 compatibility for OLMo-core's bettermap dependency."""

import multiprocessing
import multiprocessing.context

if not hasattr(multiprocessing.context, "ForkProcess"):
    multiprocessing.context.ForkProcess = multiprocessing.context.SpawnProcess  # type: ignore[attr-defined]

_original_get_context = multiprocessing.get_context


def _get_context(method=None):
    return _original_get_context("spawn" if method == "fork" else method)


multiprocessing.get_context = _get_context
