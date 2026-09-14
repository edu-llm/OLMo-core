"""Per-pool no-replacement sampling, factored out of the trainer-facing loader.

Depends only on numpy, so the sampling mechanism -- the direct fix for audit
finding 1a (curriculum arms saw as little as 58% of the corpus, repeated up
to 1.7x, while control saw about 100%) -- is unit-testable without pulling in
torch or olmo_core. `curriculum_loader.CurriculumDataLoader` delegates to
`PoolSampler` for every draw.
"""

from __future__ import annotations

import numpy as np


class PoolSampler:
    """Caches and draws from per-pool no-replacement permutations.

    A "pool" is identified purely by its (start, end) range. The cache is
    keyed by (start, end, cycle), so it is a pure memoization of
    `np.random.default_rng((seed, start, end, cycle)).permutation(width)` --
    querying the same key always returns the same permutation regardless of
    call order, which is what makes every draw a pure function of
    (start, end, prior_draws, take) and therefore safe for out-of-order or
    random access, not just sequential iteration.
    """

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _permutation(self, start: int, end: int, cycle: int) -> np.ndarray:
        key = (start, end, cycle)
        cached = self._cache.get(key)
        if cached is None:
            width = end - start
            rng = np.random.default_rng((self.seed, start, end, cycle))
            cached = rng.permutation(width)
            self._cache[key] = cached
        return cached

    def draw(self, start: int, end: int, prior_draws: int, take: int) -> np.ndarray:
        """Draw `take` absolute positions from [start, end).

        Resumes a no-replacement walk at cumulative offset `prior_draws`,
        reshuffling (an independent fresh permutation) only once the current
        pass through the pool is exhausted -- so repeats occur only after
        every element of the pool has already been drawn once.
        """
        if end <= start:
            raise ValueError("invalid pool range")
        if take <= 0:
            raise ValueError("take must be positive")
        if prior_draws < 0:
            raise ValueError("prior_draws must be non-negative")
        width = end - start
        result = np.empty(take, dtype=np.int64)
        filled = 0
        cursor = prior_draws
        while filled < take:
            cycle, offset = divmod(cursor, width)
            perm = self._permutation(start, end, cycle)
            chunk = min(width - offset, take - filled)
            result[filled : filled + chunk] = perm[offset : offset + chunk]
            filled += chunk
            cursor += chunk
        return start + result

    def cache_size(self) -> int:
        """Number of distinct (pool, cycle) permutations materialized so far."""
        return len(self._cache)
