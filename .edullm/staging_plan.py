"""Platform-neutral planning for bounded local staging of token shards."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence


def select_paths(
    paths: Sequence[str],
    *,
    required_tokens: float,
    headroom: float,
    size_of: Callable[[str], int],
    max_files: int | None = None,
) -> tuple[tuple[str, ...], int]:
    """Select the smallest deterministic shard prefix satisfying weighted demand."""
    if required_tokens < 0:
        raise ValueError("required_tokens must be non-negative")
    if not math.isfinite(headroom) or headroom < 1.0:
        raise ValueError("headroom must be finite and at least 1.0")
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")

    target_tokens = max(1, math.ceil(required_tokens * headroom))
    selected: list[str] = []
    selected_tokens = 0
    for uri in paths:
        size = int(size_of(uri))
        if size <= 0 or size % 4:
            raise RuntimeError(f"published uint32 shard has invalid size {size}: {uri}")
        selected.append(uri)
        selected_tokens += size // 4
        if max_files is not None:
            if len(selected) >= max_files:
                break
        elif selected_tokens >= target_tokens:
            break
    if not selected:
        raise RuntimeError("published domain resolved no shards")
    return tuple(selected), selected_tokens
