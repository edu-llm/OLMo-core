"""Arm-correctness harness (plan section 8) -- runs before any GPU hour.

Pure Python and numpy: no torch, no olmo_core, no network, no FarmShare
access. Executes the real `curriculum_pacing` and `curriculum_sampling`
code at production geometry (the real per-metric corpus sizes measured from
the materialized manifests, and the real 2048-sequence global batch) to
catch exactly the class of defect that the five-reviewer audit found: an
arm whose exposure, coverage, or sampling regime silently does not do what
its name says.

Items below are numbered to match plan section 8. Items 4, 7 and 10
(realized domain mixture, chunk purity, decile-1 content) need the actual
materialized corpus and cannot run here; they were run on FarmShare against
all three metrics as jobs 1721421, 1721422, 1723433 (mixture and purity)
and 1721465, 1723432 (decile-1 content). Item 1 (sort direction) needs real
metric scores judged against an external anchor and is deferred to the
reference-CE check in plan 4h, which runs on the Phase 0 control checkpoint;
that is safe to defer because Phase 0 is control-only and a uniform shuffle
is independent of sort direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import curriculum_pacing as cp
from curriculum_sampling import PoolSampler

# Production geometry, measured from the materialized manifests rather than
# projected. 4,875,093 -- the number this constant held until the corpus was
# actually built -- is the PRE-gate chunk count; the validity gate removed
# 53,954 documents and re-chunking the remainder yields the counts below.
# The three metrics differ by at most 2 chunks, from per-shard remainder
# truncation under different sort orders, so no single constant covers all
# three and the invariants are checked against each.
PRODUCTION_SIZES = {
    "mtld": 4_872_915,
    "flesch": 4_872_916,
    "compression_ratio": 4_872_917,
}
# Default for the bulk of the assertions below. Smallest of the three, so a
# pool that fits here fits under every metric.
PRODUCTION_SIZE = PRODUCTION_SIZES["mtld"]
TAKE = 2048  # global_sequences_per_batch at 4,194,304-token global batch / 2048 seq len

NON_CONTROL_PACINGS = [p for p in cp.PACING_NAMES if p != "control"]
CURRICULUM_ORDERED_PACINGS = [
    "linear_n10",
    "anti_linear_n10",
    "quadratic_n10",
    "warmup_quadratic_n10_1000",
    "interleave_i10_linear",
]


# --- Helpers -----------------------------------------------------------


def _decode_ranked_rank(position: int) -> int:
    """A trivial 'ranked array' where position IS the difficulty rank.

    Real ranked arrays are a permutation of chunk ids sorted by difficulty;
    for pure pacing/sampling tests we only care about POSITIONS in that
    array (rank order), so using the identity permutation is equivalent and
    keeps the harness free of any corpus dependency.
    """
    return position


def _simulate_run(pacing: str, size: int, take: int, seed: int = 0) -> dict[int, list[int]]:
    """Run every step of `pacing` end to end and record which absolute
    positions (in the ranked/full array) were drawn at each step.

    Returns {step: [positions...]}.
    """
    sampler = PoolSampler(seed)
    draws: dict[int, list[int]] = {}
    for step in range(cp.TOTAL_STEPS):
        pool = cp.pool_for_step(step, size, pacing, take)
        positions = sampler.draw(pool.start, pool.end, pool.prior_draws, take)
        draws[step] = positions.tolist()
    return draws


# --- 1. Sort direction is out of scope here (requires real metric scores on
#        real documents) -- covered once the corpus rebuild has scored
#        documents; see plan 4c/4d. Nothing to test with synthetic data.


# --- 2. Exposure equality -----------------------------------------------


class TestExposureEquality:
    def test_linear_gives_every_decile_exactly_one_equal_segment(self) -> None:
        counts = {}
        for step in range(cp.CURRICULUM_STEPS):
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, "linear_n10", TAKE)
            counts[(pool.start, pool.end)] = counts.get((pool.start, pool.end), 0) + 1
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        assert len(counts) == cp.N_BUCKETS
        expected_steps_per_decile = cp.CURRICULUM_STEPS // cp.N_BUCKETS
        for bucket in buckets:
            key = (bucket[0], max(bucket[0] + 1, bucket[1]))
            assert counts[key] == expected_steps_per_decile, (
                f"decile {bucket} got {counts[key]} steps, expected "
                f"{expected_steps_per_decile} -- this is exactly the class of bug "
                f"that gave the old grid's final decile 134 steps against 250"
            )

    def test_anti_linear_has_identical_exposure_to_linear_reversed_only(self) -> None:
        linear_boundaries = []
        anti_boundaries = []
        for step in range(cp.CURRICULUM_STEPS):
            linear_boundaries.append(
                (cp.pool_for_step(step, PRODUCTION_SIZE, "linear_n10", TAKE).start,
                 cp.pool_for_step(step, PRODUCTION_SIZE, "linear_n10", TAKE).end)
            )
            anti_boundaries.append(
                (cp.pool_for_step(step, PRODUCTION_SIZE, "anti_linear_n10", TAKE).start,
                 cp.pool_for_step(step, PRODUCTION_SIZE, "anti_linear_n10", TAKE).end)
            )
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        n = cp.N_BUCKETS
        for step in range(cp.CURRICULUM_STEPS):
            segment = cp.segment_index(step, cp.LINEAR_SEGMENT_BOUNDARIES)
            expect_linear = buckets[segment]
            expect_anti = buckets[n - 1 - segment]
            assert linear_boundaries[step][0] == expect_linear[0]
            assert anti_boundaries[step][0] == expect_anti[0]
        # Same set of (segment_start -> decile) exposure durations, just
        # assigned to the mirrored decile.
        assert linear_boundaries == [
            (buckets[cp.segment_index(s, cp.LINEAR_SEGMENT_BOUNDARIES)][0],
             max(buckets[cp.segment_index(s, cp.LINEAR_SEGMENT_BOUNDARIES)][0] + 1,
                 buckets[cp.segment_index(s, cp.LINEAR_SEGMENT_BOUNDARIES)][1]))
            for s in range(cp.CURRICULUM_STEPS)
        ]

    def test_quadratic_exposure_is_proportional_to_bucket_index(self) -> None:
        sizes = cp.split_weighted(cp.CURRICULUM_STEPS, range(1, cp.N_BUCKETS + 1))
        boundaries = cp.boundaries_from_sizes(sizes)
        assert boundaries == cp.QUADRATIC_SEGMENT_BOUNDARIES
        # Decile 10 (index 9) must get the most steps; decile 1 the fewest.
        assert sizes == sorted(sizes)
        assert sizes[-1] > sizes[0] * 9  # roughly 10x by construction (weights 1..10)

    def test_warmup_ramp_is_flat_across_deciles(self) -> None:
        draws_per_decile = {i: 0 for i in range(cp.N_BUCKETS)}
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        for step in range(cp.WARMUP_K):
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, "warmup_ramp_1000", TAKE)
            # Attribute the band to whichever decile contains its midpoint.
            midpoint = (pool.start + pool.end) // 2
            for index, (start, end) in enumerate(buckets):
                if start <= midpoint < end:
                    draws_per_decile[index] += TAKE
                    break
        total = sum(draws_per_decile.values())
        per_decile_share = [count / total for count in draws_per_decile.values()]
        # Flat to within a small tolerance -- not exact because deciles and
        # the K=1000 band grid do not align perfectly, but no decile should
        # dominate the way the old 42%-of-axis bug did (deciles 6-10 at 0%).
        assert min(per_decile_share) > 0.05, (
            f"decile shares {per_decile_share} -- a near-zero share reproduces "
            f"the old bug where deciles 6-10 were never seen during warmup"
        )
        assert max(per_decile_share) < 0.15


# --- 3. Unique-data parity (the audit's largest finding, 1a) -----------


class TestUniqueDataParity:
    @pytest.mark.parametrize("pacing", ["linear_n10", "anti_linear_n10", "interleave_i10_linear"])
    def test_equal_exposure_arms_never_repeat_within_their_decile_budget(self, pacing: str) -> None:
        """For arms where a decile's total draws stay within its pool size,
        the no-replacement mechanism must produce ZERO repeats -- this is
        the regression test for finding 1a. The old with-replacement
        sampler would have repeats here immediately.
        """
        draws = _simulate_run(pacing, PRODUCTION_SIZE, TAKE)
        per_decile_positions: dict[int, list[int]] = {}
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)

        def decile_of(position: int) -> int:
            for index, (start, end) in enumerate(buckets):
                if start <= position < end:
                    return index
            raise AssertionError("position outside every bucket")

        for step in range(cp.CURRICULUM_STEPS):
            for position in draws[step]:
                per_decile_positions.setdefault(decile_of(position), []).append(position)

        expected_draws_per_decile = cp.CURRICULUM_STEPS // cp.N_BUCKETS * TAKE
        for decile, positions in per_decile_positions.items():
            assert len(positions) == expected_draws_per_decile
            unique = len(set(positions))
            assert unique == len(positions), (
                f"{pacing} decile {decile}: {len(positions)} draws but only "
                f"{unique} unique positions -- repeats occurred even though "
                f"draws ({len(positions)}) did not exceed the decile pool size"
            )

    def test_control_covers_the_whole_corpus_with_zero_repeats_for_one_pass(self) -> None:
        sampler = PoolSampler(seed=0)
        seen: set[int] = set()
        step = 0
        total_draws = 0
        # One full pass through the corpus should be entirely unique.
        while total_draws + TAKE <= PRODUCTION_SIZE:
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, "control", TAKE)
            positions = sampler.draw(pool.start, pool.end, pool.prior_draws, TAKE)
            for position in positions.tolist():
                assert position not in seen, "control repeated a chunk before exhausting the pool"
                seen.add(position)
            total_draws += TAKE
            step += 1
        assert len(seen) == total_draws

    def test_quadratic_hardest_decile_legitimately_repeats_but_evenly(self) -> None:
        """quadratic_n10's decile 10 gets more draws than its pool size, by
        design (ordering + reweighting, plan 6b) -- it MUST see repeats, but
        every element should be drawn either floor(ratio) or ceil(ratio)
        times, never wildly uneven (which would indicate a broken cursor).
        """
        draws = _simulate_run("quadratic_n10", PRODUCTION_SIZE, TAKE)
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        hardest_start, hardest_end = buckets[-1]
        hardest_positions: list[int] = []
        for step in range(cp.CURRICULUM_STEPS):
            for position in draws[step]:
                if hardest_start <= position < hardest_end:
                    hardest_positions.append(position)
        width = hardest_end - hardest_start
        assert len(hardest_positions) > width, "expected decile 10 to legitimately oversample"
        counts = np.bincount(np.array(hardest_positions) - hardest_start, minlength=width)
        assert counts.min() >= counts.max() - 1, (
            f"uneven repeat counts (min={counts.min()}, max={counts.max()}) -- "
            f"the no-replacement cursor should distribute repeats almost evenly"
        )

    def test_common_random_numbers_across_arms_sharing_a_seed(self) -> None:
        """Two different pacings drawing from the SAME decile at the SAME
        cumulative offset, with the same seed, must draw identical chunks
        (finding 1g) -- an emergent property of sharing the pool-boundary
        function and the (seed, start, end, cycle) cache key.
        """
        seed = 7
        sampler_a = PoolSampler(seed)
        sampler_b = PoolSampler(seed)
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        start, end = buckets[4]  # decile 5, arbitrary
        draw_a = sampler_a.draw(start, end, prior_draws=0, take=TAKE)
        draw_b = sampler_b.draw(start, end, prior_draws=0, take=TAKE)
        assert np.array_equal(draw_a, draw_b)


# --- 4. Realized domain mixture is out of scope here (requires the real
#        corpus's domain layout); see plan section 4/6b for the "natural for
#        the linear family" claim, verified once the corpus is materialized.


# --- 5. Axis coverage ----------------------------------------------------


class TestAxisCoverage:
    @pytest.mark.parametrize("pacing", CURRICULUM_ORDERED_PACINGS)
    def test_every_decile_is_reached_during_the_curriculum_phase(self, pacing: str) -> None:
        buckets = cp.split_equal_mass(PRODUCTION_SIZE)
        touched: set[int] = set()

        def decile_of(position: int) -> int | None:
            for index, (start, end) in enumerate(buckets):
                if start <= position < end:
                    return index
            return None

        for step in range(cp.CURRICULUM_STEPS):
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, pacing, TAKE)
            midpoint = (pool.start + pool.end) // 2
            decile = decile_of(midpoint)
            if decile is not None:
                touched.add(decile)
        assert touched == set(range(cp.N_BUCKETS)), (
            f"{pacing} never touched deciles {set(range(cp.N_BUCKETS)) - touched} "
            f"during the curriculum phase -- reproduces the old warmup bug that "
            f"reached only 42% of the axis"
        )

    def test_warmup_ramp_reaches_100_percent_of_the_axis(self) -> None:
        max_hi = 0
        for step in range(cp.WARMUP_K):
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, "warmup_ramp_1000", TAKE)
            max_hi = max(max_hi, pool.end)
        coverage = max_hi / PRODUCTION_SIZE
        assert coverage > 0.999, (
            f"warmup_ramp_1000 reached only {coverage:.4%} of the axis -- the old "
            f"warmup_1000 reached only 41.99%, this must reach essentially 100%"
        )


# --- 6. Non-monotonicity within a bucket --------------------------------


class TestNonMonotonicity:
    @pytest.mark.parametrize("pacing", NON_CONTROL_PACINGS)
    def test_no_bucket_emits_a_strictly_monotonic_difficulty_sequence(self, pacing: str) -> None:
        """Sorting documents by difficulty must not turn any arm into a naive
        strict march through the sorted array -- draws within an active pool
        must be shuffled, not sequential.
        """
        # Concatenate every step's draw within the first decile visit only
        # (enough to detect a monotonic-walk bug without full-run cost).
        flat: list[int] = []
        seen_pool: tuple[int, int] | None = None
        for step in range(cp.TOTAL_STEPS):
            pool = cp.pool_for_step(step, PRODUCTION_SIZE, pacing, TAKE)
            key = (pool.start, pool.end)
            if seen_pool is None:
                seen_pool = key
            if key != seen_pool:
                break
            positions = PoolSampler(seed=0).draw(pool.start, pool.end, pool.prior_draws, TAKE)
            flat.extend(positions.tolist())
            if len(flat) >= 4096:
                break
        assert len(flat) > 1
        is_strictly_increasing = all(b > a for a, b in zip(flat, flat[1:]))
        assert not is_strictly_increasing, (
            f"{pacing} emitted a strictly increasing difficulty sequence -- "
            f"this indicates a naive sorted walk rather than a shuffled draw"
        )


# --- 8. Replicate validity ----------------------------------------------


class TestReplicateValidity:
    @pytest.mark.parametrize("pacing", cp.PACING_NAMES)
    def test_seed_changes_the_data_order(self, pacing: str) -> None:
        sampler_a = PoolSampler(seed=1)
        sampler_b = PoolSampler(seed=2)
        size = PRODUCTION_SIZE
        step = 0 if pacing != "control" else 5  # arbitrary in-range step
        pool = cp.pool_for_step(step, size, pacing, TAKE)
        draw_a = sampler_a.draw(pool.start, pool.end, pool.prior_draws, TAKE)
        draw_b = sampler_b.draw(pool.start, pool.end, pool.prior_draws, TAKE)
        assert not np.array_equal(draw_a, draw_b), (
            f"{pacing}: changing the seed did not change the draw -- this arm "
            f"would contribute zero seed variance, biasing any noise-floor estimate"
        )


# --- 9. Epoch accounting -------------------------------------------------


class TestEpochAccounting:
    def test_total_steps_is_2500_regardless_of_curriculum_split(self) -> None:
        assert cp.TOTAL_STEPS == 2500
        assert cp.TOTAL_STEPS % 125 == 0, "eval cadence must divide the run exactly"

    def test_curriculum_phase_is_exact_multiple_of_buckets_and_subbuckets(self) -> None:
        assert cp.CURRICULUM_STEPS % cp.N_BUCKETS == 0
        assert (cp.CURRICULUM_STEPS // cp.N_BUCKETS) % cp.N_BUCKETS == 0

    def test_linear_total_draws_do_not_exceed_one_epoch(self) -> None:
        total_draws = cp.CURRICULUM_STEPS * TAKE
        assert total_draws <= PRODUCTION_SIZE, (
            f"{total_draws} draws vs {PRODUCTION_SIZE} chunks -- linear_n10 should "
            f"stay within one epoch at the current geometry"
        )
        epochs = total_draws / PRODUCTION_SIZE
        assert 0.9 < epochs <= 1.0


# --- 11. LR continuity is tested at the entrypoint level, not here (this
#         harness has no LR schedule to check yet).


# --- Sanity: closed-form prior_draws matches a brute-force simulation ---


class TestPriorDrawsClosedForm:
    """The closed-form `prior_draws` in `pool_for_step` must exactly equal a
    brute-force count of how many chunks were drawn from the identical pool
    at all earlier steps. This is the correctness property the whole
    no-replacement mechanism depends on.
    """

    @pytest.mark.parametrize("pacing", cp.PACING_NAMES)
    def test_prior_draws_matches_brute_force_count(self, pacing: str) -> None:
        size = 50_000  # smaller than production for a fast brute-force check
        take = 128
        seen_draws: dict[tuple[int, int], int] = {}
        # Only check as many steps as fit within `size` at least twice, or
        # TOTAL_STEPS, whichever is smaller -- brute-force is O(steps).
        limit = min(cp.TOTAL_STEPS, 400)
        for step in range(limit):
            pool = cp.pool_for_step(step, size, pacing, take)
            key = (pool.start, pool.end)
            expected_prior = seen_draws.get(key, 0)
            assert pool.prior_draws == expected_prior, (
                f"{pacing} step {step}: pool {key} claims prior_draws="
                f"{pool.prior_draws}, brute force says {expected_prior}"
            )
            seen_draws[key] = expected_prior + take


# --- Every metric's real materialized geometry, not just the default -----


class TestAllMetricGeometries:
    """The three metrics produce chunk counts differing by up to 2, and the
    decile boundaries are a frozen artifact derived from that count. A
    geometry that is only ever checked at one of the three sizes is not
    checked at production geometry for the other two.
    """

    @pytest.mark.parametrize("metric,size", sorted(PRODUCTION_SIZES.items()))
    def test_deciles_tile_the_corpus_exactly(self, metric: str, size: int) -> None:
        buckets = cp.split_equal_mass(size)
        assert len(buckets) == cp.N_BUCKETS
        assert buckets[0][0] == 0, f"{metric}: first bucket must start at 0"
        assert buckets[-1][1] == size, f"{metric}: last bucket must end at {size}"
        for (_, prev_end), (next_start, _) in zip(buckets, buckets[1:]):
            assert prev_end == next_start, f"{metric}: buckets must be contiguous"
        widths = [end - start for start, end in buckets]
        assert max(widths) - min(widths) <= 1, (
            f"{metric}: largest-remainder split must not differ by more than one "
            f"chunk across deciles, got widths {widths}"
        )

    @pytest.mark.parametrize("metric,size", sorted(PRODUCTION_SIZES.items()))
    def test_linear_spends_equal_steps_per_decile(self, metric: str, size: int) -> None:
        counts: dict[tuple[int, int], int] = {}
        for step in range(cp.CURRICULUM_STEPS):
            pool = cp.pool_for_step(step, size, "linear_n10", TAKE)
            key = (pool.start, pool.end)
            counts[key] = counts.get(key, 0) + 1
        assert len(counts) == cp.N_BUCKETS
        assert set(counts.values()) == {cp.CURRICULUM_STEPS // cp.N_BUCKETS}, (
            f"{metric}: expected {cp.CURRICULUM_STEPS // cp.N_BUCKETS} steps in every "
            f"decile, got {sorted(counts.values())}"
        )

    @pytest.mark.parametrize("metric,size", sorted(PRODUCTION_SIZES.items()))
    @pytest.mark.parametrize("pacing", NON_CONTROL_PACINGS)
    def test_pools_stay_inside_the_corpus(self, metric: str, size: int, pacing: str) -> None:
        for step in range(cp.CURRICULUM_STEPS):
            pool = cp.pool_for_step(step, size, pacing, TAKE)
            assert 0 <= pool.start < pool.end <= size, (
                f"{metric}/{pacing} step {step}: pool [{pool.start}, {pool.end}) "
                f"escapes [0, {size})"
            )
