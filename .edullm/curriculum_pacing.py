"""Pure, step-addressed curriculum pacing semantics for `curriculum-new`.

Rewritten from scratch (no code copied from the edullm repo) after a
five-reviewer audit found the sampling regime, decile grid, and warmup arm in
the prior implementation were three independent, compounding defects. See
plan `wild-swinging-aurora.md` sections 1 and 6 for the full audit trail.

Every function here is a pure function of (step, size, pacing, take) with no
hidden state -- the loader is free to call these in any order, which is what
makes every arm truly resumable and randomly-addressable rather than only
resumable from a checkpoint.

`pool_for_step` returns a `PoolSpec`: a contiguous half-open range
`[start, end)` of positions in a difficulty-ranked array of size `n`,
plus `prior_draws` -- the exact number of chunks the loader must already
have drawn from that SAME pool at all earlier steps of this pacing, computed
in closed form. The loader uses `prior_draws` to resume a no-replacement
walk through the pool (reshuffling only once the pool is fully exhausted),
which is the direct fix for the largest audit finding (1a): under the
prior implementation, every non-control arm resampled its pool with
replacement on every step, so control saw about 100% of the corpus once
while curriculum arms saw as little as 58% of it, repeated up to 1.7x. This
module makes no sampling decisions itself; it only says which pool is active
and how much of it has already been consumed.

A useful emergent property: bucket boundaries (`split_equal_mass`) are a
pure function of the pool size alone, so the range for decile 3 is
IDENTICAL across every pacing that visits it. Combined with the loader using
a permutation keyed by (seed, start, end, cycle), two different arms
drawing from decile 3 at the same cumulative offset draw the SAME chunks --
common random numbers across arms sharing a seed, which is audit finding
1g, achieved without any special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

N_BUCKETS = 10

# The curriculum phase length is corpus-attrition-dependent (see plan 6a);
# the three supported values keep both N_BUCKETS and the interleave
# sub-bucket count exact:
#   2300 -> 230/decile, 23/sub-bucket   (<=3.4% gate attrition)
#   2200 -> 220/decile, 22/sub-bucket   (<=7.6%)
#   2100 -> 210/decile, 21/sub-bucket   (<=11.8%)
# CURRICULUM_STEPS is set once the corpus rebuild measures actual gate
# attrition. ANNEAL_STEPS is always chosen so CURRICULUM_STEPS + ANNEAL_STEPS
# == TOTAL_STEPS == 2500 (= 20 x 125, so the eval cadence divides the run
# exactly regardless of which branch is taken).
CURRICULUM_STEPS = 2300
TOTAL_STEPS = 2500
ANNEAL_STEPS = TOTAL_STEPS - CURRICULUM_STEPS
if ANNEAL_STEPS <= 0:
    raise AssertionError("CURRICULUM_STEPS must leave a positive anneal phase")
if CURRICULUM_STEPS % N_BUCKETS:
    raise AssertionError("CURRICULUM_STEPS must be an exact multiple of N_BUCKETS")

# K = warmup window length in steps, for warmup_ramp_1000 and
# warmup_quadratic_n10_1000. Independent of CURRICULUM_STEPS by design (plan
# 6b): both warmup arms hand off to uniform full-pool sampling at K=1000
# regardless of which attrition branch (2300/2200/2100) is active.
WARMUP_K = 1000
if not 0 < WARMUP_K < CURRICULUM_STEPS:
    raise AssertionError("WARMUP_K must fall strictly inside the curriculum phase")

CURRICULUM_DATASET_ID = "curriculum/opt-with-synthetic-10b"

# Three metrics survive the corpus rebuild (plan section 4); learnability is
# dropped entirely, and compression's sort direction is fixed at its one
# declaration site here rather than left to disagree with a stale SCHEMA
# string as it did before (plan 4c/4d).
DIFFICULTY_METRICS = ("compression_ratio", "mtld", "flesch")

PACING_NAMES = (
    "control",
    "linear_n10",
    "anti_linear_n10",
    "quadratic_n10",
    "warmup_ramp_1000",
    "warmup_quadratic_n10_1000",
    "interleave_i10_linear",
)


@dataclass(frozen=True)
class PoolSpec:
    """A contiguous half-open range [start, end) plus its draw history.

    prior_draws is the number of chunks already drawn from this EXACT pool
    (same start, same end) at all steps earlier than the one this PoolSpec
    was resolved for, within the same pacing run. It may exceed
    end - start: several arms (quadratic, warmup-quadratic) deliberately
    over-sample a decile relative to its size, and once cumulative draws
    exceed the pool width the loader reshuffles and starts a fresh
    no-replacement pass rather than sampling with replacement.
    """

    start: int
    end: int
    prior_draws: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid pool range")
        if self.prior_draws < 0:
            raise ValueError("prior_draws must be non-negative")

    @property
    def width(self) -> int:
        return self.end - self.start


def split_evenly(total: int, n: int) -> list[int]:
    """Largest-remainder split of `total` into `n` near-equal integer parts.

    Used for the equal-exposure arms (linear_n10, anti_linear_n10, and
    interleave_i10_linear's sub-buckets). Every supported CURRICULUM_STEPS
    value is an exact multiple of N_BUCKETS, so in practice every part is
    exactly total // n with zero remainder to distribute -- this is the
    direct fix for the finding that the prior grid's final decile got 134
    steps against 250 for every other decile (a 46% shortfall).
    """
    if total < 0 or n <= 0:
        raise ValueError("total must be non-negative and n must be positive")
    width, remainder = divmod(int(total), int(n))
    return [width + 1 if index < remainder else width for index in range(n)]


def split_weighted(total: int, weights: Sequence[float]) -> list[int]:
    """Largest-remainder split of `total` proportional to `weights`.

    Verified by hand to reproduce both the quadratic_n10 boundaries at
    CURRICULUM_STEPS=2300 (weights 1..10 -> sizes
    42,84,125,167,209,251,293,335,376,418) and the warmup_quadratic_n10_1000
    boundaries at K=1000 (same weights -> sizes
    18,36,55,73,91,109,127,145,164,182) exactly.
    """
    weights = [float(w) for w in weights]
    if total < 0 or not weights or any(w < 0 for w in weights):
        raise ValueError("total must be non-negative and weights must be non-negative")
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    raw = [w * total / total_weight for w in weights]
    floors = [int(value) for value in raw]
    remainder = int(total) - sum(floors)
    if remainder < 0:
        raise AssertionError("unreachable: floor sum cannot exceed total")
    order = sorted(range(len(weights)), key=lambda i: raw[i] - floors[i], reverse=True)
    for index in order[:remainder]:
        floors[index] += 1
    if sum(floors) != int(total):
        raise AssertionError("unreachable: largest-remainder split must sum to total")
    return floors


def boundaries_from_sizes(sizes: Sequence[int]) -> tuple[int, ...]:
    """Cumulative boundaries (0, s0, s0+s1, ..., sum(sizes)) from segment sizes."""
    cumulative = [0]
    for size in sizes:
        cumulative.append(cumulative[-1] + int(size))
    return tuple(cumulative)


LINEAR_SEGMENT_BOUNDARIES = boundaries_from_sizes(split_evenly(CURRICULUM_STEPS, N_BUCKETS))
QUADRATIC_SEGMENT_BOUNDARIES = boundaries_from_sizes(
    split_weighted(CURRICULUM_STEPS, range(1, N_BUCKETS + 1))
)
WARMUP_QUADRATIC_SEGMENT_BOUNDARIES = boundaries_from_sizes(
    split_weighted(WARMUP_K, range(1, N_BUCKETS + 1))
)
INTERLEAVE_SUBBUCKET_SIZES = tuple(split_evenly(CURRICULUM_STEPS // N_BUCKETS, N_BUCKETS))

# Every pacing except the decile-ordered ones spends its tail (and the
# post-curriculum anneal, which is out of the curriculum-phase step range
# but shares the same full-pool identity) uniform over the whole array.
# This is the step at which that full-pool phase begins for each pacing;
# anti_linear, linear, quadratic and interleave never touch the full pool
# until the anneal boundary, while the warmup arms hand off to it at K.
_FULL_POOL_START_STEP: dict[str, int] = {
    "control": 0,
    "linear_n10": CURRICULUM_STEPS,
    "anti_linear_n10": CURRICULUM_STEPS,
    "quadratic_n10": CURRICULUM_STEPS,
    "interleave_i10_linear": CURRICULUM_STEPS,
    "warmup_ramp_1000": WARMUP_K,
    "warmup_quadratic_n10_1000": WARMUP_K,
}


def segment_index(step: int, boundaries: Sequence[int]) -> int:
    step = int(step)
    if step < 0:
        raise ValueError("step must be non-negative")
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= step < end:
            return index
    if step == boundaries[-1]:
        return len(boundaries) - 2
    raise ValueError("step is outside the boundary range")


def split_equal_mass(size: int, buckets: int = N_BUCKETS) -> list[tuple[int, int]]:
    """Equal-mass [start, end) ranges over a difficulty-ranked array of `size`.

    A pure function of `size` alone, so bucket i's range is identical
    across every pacing that visits it -- see the module docstring.
    """
    if size < 0 or buckets <= 0:
        raise ValueError("size must be non-negative and buckets must be positive")
    width, remainder = divmod(int(size), int(buckets))
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(buckets):
        end = start + width + (1 if index < remainder else 0)
        result.append((start, end))
        start = end
    return result


def interleave_subbucket_index(step: int) -> int:
    """Which of the 10 mini-curriculum sub-buckets `step` falls into.

    Each of the 10 top-level segments replays a full 10-bucket mini
    curriculum internally, at CURRICULUM_STEPS/100 steps per sub-bucket --
    exact for every supported CURRICULUM_STEPS value, and identical every
    top-level segment (the same bucket order 0..9 repeats each time), which
    is what makes `_interleave_prior_draws` below a closed form.
    """
    segment = segment_index(step, LINEAR_SEGMENT_BOUNDARIES)
    segment_start = LINEAR_SEGMENT_BOUNDARIES[segment]
    local_step = int(step) - segment_start
    elapsed = 0
    for index, duration in enumerate(INTERLEAVE_SUBBUCKET_SIZES):
        elapsed += duration
        if local_step < elapsed:
            return index
    return N_BUCKETS - 1


def _segment_prior_draws(step: int, boundaries: Sequence[int], take: int) -> int:
    """Chunks already drawn from the CURRENT segment's pool.

    Valid whenever each segment's pool identity is unique to that segment --
    true for linear_n10, anti_linear_n10, quadratic_n10, and
    warmup_quadratic_n10_1000's phase-1 segments, none of which ever revisit
    an earlier segment's decile.
    """
    segment = segment_index(step, boundaries)
    local_offset = int(step) - boundaries[segment]
    return local_offset * int(take)


def _full_pool_prior_draws(step: int, start_step: int, take: int) -> int:
    """Chunks already drawn from a full pool active continuously since `start_step`."""
    elapsed = int(step) - int(start_step)
    if elapsed < 0:
        raise ValueError("step precedes the full-pool phase for this pacing")
    return elapsed * int(take)


def _interleave_prior_draws(step: int, take: int) -> int:
    """Chunks already drawn from the CURRENT sub-bucket's pool, across all visits.

    Sums one complete visit's worth of draws for every earlier top-level
    segment (bucket b appears exactly once per segment, always at the same
    relative position), plus however much of the current visit has elapsed.
    """
    segment = segment_index(step, LINEAR_SEGMENT_BOUNDARIES)
    bucket = interleave_subbucket_index(step)
    subbucket_start_local = sum(INTERLEAVE_SUBBUCKET_SIZES[:bucket])
    occurrence_start = LINEAR_SEGMENT_BOUNDARIES[segment] + subbucket_start_local
    local_offset = int(step) - occurrence_start
    prior_occurrences = segment
    return (prior_occurrences * INTERLEAVE_SUBBUCKET_SIZES[bucket] + local_offset) * int(take)


def pool_for_step(step: int, size: int, pacing: str, take: int) -> PoolSpec:
    """Resolve the active difficulty-ranked pool for `step` under `pacing`.

    `step` ranges over the WHOLE run, [0, TOTAL_STEPS) -- both the
    curriculum phase and the post-curriculum anneal, since the anneal is
    uniform over the full pool for every pacing and that is expressed here
    as a trivial fallback rather than handled ad hoc by the loader.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if take <= 0:
        raise ValueError("take must be positive")
    if pacing not in PACING_NAMES:
        raise ValueError(f"unknown pacing {pacing!r}")
    step = int(step)
    if not 0 <= step < TOTAL_STEPS:
        raise ValueError("step is outside [0, TOTAL_STEPS)")

    start_of_full_pool = _FULL_POOL_START_STEP[pacing]
    if step >= start_of_full_pool:
        return PoolSpec(0, size, _full_pool_prior_draws(step, start_of_full_pool, take))

    # Every remaining branch is strictly inside the curriculum phase and
    # strictly before this pacing's full-pool handoff.
    buckets = split_equal_mass(size)

    if pacing == "linear_n10":
        segment = segment_index(step, LINEAR_SEGMENT_BOUNDARIES)
        start, end = buckets[segment]
        prior = _segment_prior_draws(step, LINEAR_SEGMENT_BOUNDARIES, take)
        return PoolSpec(start, max(start + 1, end), prior)

    if pacing == "anti_linear_n10":
        # The only difference from linear_n10: present bucket
        # (N_BUCKETS-1-segment) instead of bucket[segment], so the hardest
        # decile is trained first. Exposure schedule (which segment gets
        # which step range) is identical to linear_n10; only the mapping
        # from segment to decile is reversed.
        segment = segment_index(step, LINEAR_SEGMENT_BOUNDARIES)
        start, end = buckets[N_BUCKETS - 1 - segment]
        prior = _segment_prior_draws(step, LINEAR_SEGMENT_BOUNDARIES, take)
        return PoolSpec(start, max(start + 1, end), prior)

    if pacing == "quadratic_n10":
        segment = segment_index(step, QUADRATIC_SEGMENT_BOUNDARIES)
        start, end = buckets[segment]
        prior = _segment_prior_draws(step, QUADRATIC_SEGMENT_BOUNDARIES, take)
        return PoolSpec(start, max(start + 1, end), prior)

    if pacing == "warmup_ramp_1000":
        # Linear 0->100% ramp across the FULL difficulty axis over the first
        # WARMUP_K steps: at step t the active band is [t*n/K, (t+1)*n/K).
        # This is the corrected replacement for the prior warmup_1000, whose
        # fixed one-global-batch stride reached only 42% of the axis (the
        # single most consequential planted defect in the audit). Consecutive
        # bands never overlap (hi(t) == lo(t+1)), so each band is used
        # exactly once, ever -- prior_draws is always 0.
        lo = (step * size) // WARMUP_K
        hi = ((step + 1) * size) // WARMUP_K
        return PoolSpec(lo, max(lo + 1, hi), 0)

    if pacing == "warmup_quadratic_n10_1000":
        segment = segment_index(step, WARMUP_QUADRATIC_SEGMENT_BOUNDARIES)
        start, end = buckets[segment]
        prior = _segment_prior_draws(step, WARMUP_QUADRATIC_SEGMENT_BOUNDARIES, take)
        return PoolSpec(start, max(start + 1, end), prior)

    if pacing == "interleave_i10_linear":
        bucket = interleave_subbucket_index(step)
        start, end = buckets[bucket]
        prior = _interleave_prior_draws(step, take)
        return PoolSpec(start, max(start + 1, end), prior)

    raise AssertionError("unreachable")
