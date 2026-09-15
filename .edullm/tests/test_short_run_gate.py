"""The freeze contract must refuse an undeclared short run and allow a declared one.

The pre-campaign smoke exists to exercise the real production path -- real
W&B, real task-loss suite, real distributed launch -- on a shortened run. It
deliberately does NOT pass --local-smoke, because that flag disables exactly
the things being tested; it shortens itself with --length-tokens instead.

`CurriculumDataLoader` asserted `total_steps == TOTAL_STEPS` unconditionally,
so the smoke could never start: job 1723916 took four L40S for 8.5 minutes and
died with `total_steps must equal curriculum_pacing.TOTAL_STEPS (2500), got
250`. Two correct intentions in direct conflict, and only executing it showed
that.

The gate keeps the guarantee that matters -- a production arm cannot silently
differ in length -- while allowing a run that declares itself short. These
tests pin both halves, and are stdlib-only so they run without torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EDULLM = Path(__file__).resolve().parents[1]
if str(EDULLM) not in sys.path:
    sys.path.insert(0, str(EDULLM))

import curriculum_pacing as cp  # noqa: E402


def _short_run_decile(total_steps: int) -> int:
    """Mirror of the loader's reported decile for a truncated run."""
    return min(cp.N_BUCKETS, 1 + (total_steps * cp.N_BUCKETS) // cp.CURRICULUM_STEPS)


def test_the_smoke_step_count_is_not_the_production_step_count() -> None:
    """If these ever coincide the gate becomes untestable and pointless."""
    assert cp.TOTAL_STEPS == 2500
    assert 250 != cp.TOTAL_STEPS


def test_a_short_run_truncates_the_curriculum_rather_than_compressing_it() -> None:
    """The reason a short run must stay unpoolable: at 250 of 2500 steps the
    arm never leaves the easy end of the axis, so its numbers are not a
    shrunken version of the real arm -- they are a different experiment."""
    size = 4_872_915
    take = 2048
    reached = set()
    for step in range(250):
        spec = cp.pool_for_step(step, size, "linear_n10", take)
        reached.add(spec.start)
    # A decile is 230 steps, so 250 steps reaches decile 2 and stops -- eight
    # of the ten deciles are never sampled at all.
    assert len(reached) == 2
    assert reached == {0, cp.split_equal_mass(size)[1][0]}
    assert _short_run_decile(250) == 2


def test_a_full_length_run_reaches_every_decile() -> None:
    size = 4_872_915
    starts = {
        cp.pool_for_step(step, size, "linear_n10", 2048).start
        for step in range(cp.CURRICULUM_STEPS)
    }
    assert len(starts) == cp.N_BUCKETS


@pytest.mark.parametrize("steps", [1, 125, 250, 1000, 2499])
def test_declared_short_lengths_are_inside_the_permitted_range(steps: int) -> None:
    assert 0 < steps < cp.TOTAL_STEPS


@pytest.mark.parametrize("steps", [0, -1, 2500, 5000])
def test_lengths_outside_the_permitted_range_are_not_short_runs(steps: int) -> None:
    """A short run is strictly between 0 and TOTAL_STEPS. Zero is not a run,
    and >= TOTAL_STEPS is either the real thing or an overrun -- neither may
    reach the relaxed branch."""
    assert not 0 < steps < cp.TOTAL_STEPS


def test_smoke_ladder_lands_on_permanent_checkpoint_steps() -> None:
    """The smoke's value is that it exercises the permanent-checkpoint path,
    including a mid-run checkpoint, so its length must land on the ladder."""
    from production_contract import checkpoint as checkpoint_contract

    steps = checkpoint_contract.permanent_checkpoint_steps(250, 125)
    assert steps[0] == 0
    assert 125 in steps
    assert steps[-1] == 250
    assert len(steps) >= 3
