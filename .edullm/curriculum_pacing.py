"""Pure, step-addressed curriculum pacing semantics.

The ``step`` accepted by every function is the zero-based batch index.  OLMo's
trainer increments ``global_step`` only after a loader yields, so a resumable
loader must use its own ``batches_processed`` value when calling this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

TOTAL_STEPS = 2384
N_BUCKETS = 10
SEGMENT_BOUNDARIES = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2384)
# Weights 1..10 scaled to TOTAL_STEPS via largest-remainder (unit ≈ 43.345).
QUADRATIC_SEGMENT_BOUNDARIES = (0, 43, 130, 260, 433, 650, 910, 1213, 1560, 1950, 2384)
CURRICULUM_DATASET_ID = "curriculum/regmix-370m"
PACING_NAMES = (
    "control",
    "linear_n10",
    "quadratic_n10",
    "expanding_25_1000",
    "warmup_1000",
    "interleave_i10_linear",
)
DIFFICULTY_METRICS = ("compression_ratio", "flesch", "mtld", "learnability")
ORDER_GROUPS = {
    "compression_ratio": "compression",
    "flesch": "flesch",
    "mtld": "mtld",
    "learnability": "learnability",
}
CURRICULUM_ORDER_GROUP_FOR_METRIC = ORDER_GROUPS


@dataclass(frozen=True)
class PoolSpec:
    start: int
    end: int
    ordered: bool = False
    ordered_step: int | None = None


def segment_index(step: int, boundaries: Sequence[int] = SEGMENT_BOUNDARIES) -> int:
    step = int(step)
    if step < 0:
        raise ValueError("step must be non-negative")
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start <= step < end:
            return index
    if step == boundaries[-1]:
        return len(boundaries) - 2
    raise ValueError(f"step {step} is outside [0, {boundaries[-1]}]")


def segment_range(segment: int, boundaries: Sequence[int] = SEGMENT_BOUNDARIES) -> tuple[int, int]:
    if not 0 <= int(segment) < len(boundaries) - 1:
        raise ValueError(f"segment {segment} is out of range")
    return int(boundaries[int(segment)]), int(boundaries[int(segment) + 1])


def curriculum_order_group(metric: str) -> str:
    try:
        return ORDER_GROUPS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown difficulty metric {metric!r}") from exc


def split_equal_mass(size: int, buckets: int = N_BUCKETS) -> list[tuple[int, int]]:
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


def expanding_eligible_fraction(step: int) -> float:
    step = max(0, int(step))
    return 1.0 if step >= 1000 else 0.25 + 0.75 * (step / 1000.0)


def interleave_subbucket_durations(segment_steps: int) -> list[int]:
    if segment_steps <= 0:
        raise ValueError("segment_steps must be positive")
    width, remainder = divmod(int(segment_steps), N_BUCKETS)
    durations = [width] * N_BUCKETS
    durations[-1] += remainder
    return durations


def interleave_subbucket_index(step: int) -> int:
    segment = segment_index(step)
    start, end = SEGMENT_BOUNDARIES[segment : segment + 2]
    local_step = int(step) - start
    elapsed = 0
    for index, duration in enumerate(interleave_subbucket_durations(end - start)):
        elapsed += duration
        if local_step < elapsed:
            return index
    return N_BUCKETS - 1


def pool_for_step(step: int, size: int, pacing: str) -> PoolSpec:
    if pacing not in PACING_NAMES:
        raise ValueError(f"unknown pacing {pacing!r}")
    if size <= 0:
        raise ValueError("size must be positive")
    if pacing == "control":
        return PoolSpec(0, size)

    buckets = split_equal_mass(size)
    if pacing == "linear_n10":
        start, end = buckets[segment_index(step)]
        return PoolSpec(start, max(start + 1, end) if start == end else end)
    if pacing == "quadratic_n10":
        start, end = buckets[segment_index(step, QUADRATIC_SEGMENT_BOUNDARIES)]
        return PoolSpec(start, max(start + 1, end) if start == end else end)
    if pacing == "expanding_25_1000":
        end = min(size, max(1, round(expanding_eligible_fraction(step) * size)))
        return PoolSpec(0, end)
    if pacing == "warmup_1000":
        if int(step) < 1000:
            return PoolSpec(0, size, ordered=True, ordered_step=int(step))
        return PoolSpec(0, size)
    if pacing == "interleave_i10_linear":
        start, end = buckets[interleave_subbucket_index(step)]
        return PoolSpec(start, max(start + 1, end) if start == end else end)
    raise AssertionError("unreachable")
