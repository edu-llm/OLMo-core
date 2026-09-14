"""Harness item 11: the curriculum -> anneal LR handoff must be continuous.

Plan sections 6c and 8.11. The run is one optimization with two phases:
CURRICULUM_STEPS at the condition's curriculum-phase LR, then ANNEAL_STEPS
of 1-sqrt decay to zero on uniformly sampled data. If the LR *jumps* at the
boundary -- most plausibly by resetting to peak, which is what a
mis-specified `SequentialScheduler` would do -- then the anneal is not a
continuation of the run but a second warm restart, and every arm silently
acquires a third experimental factor that is not in the design.

`SequentialScheduler` documents that it seeds each stage's initial LR from
the previous stage's final LR, so continuity *should* hold by construction.
That is exactly why it is worth asserting: the failure mode is silent, the
guarantee is someone else's docstring, and the cost of being wrong is every
run in the campaign.

These tests import `curriculum_entrypoint`, which needs torch and
olmo_core. olmo_core does not import on Windows (`bettermap` wants
`multiprocessing.context.ForkProcess`), so they skip there and run on
FarmShare and any Linux checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

try:
    import curriculum_entrypoint as ce
except ImportError as exc:  # pragma: no cover - environment-dependent
    # Not `pytest.importorskip`: that re-raises when the failure comes from a
    # nested import rather than the named module being absent, and this one
    # is nested (olmo_core -> ... -> bettermap -> multiprocessing.ForkProcess,
    # which does not exist on Windows).
    pytest.skip(
        f"curriculum_entrypoint needs torch + olmo_core, unavailable here: {exc}",
        allow_module_level=True,
    )

CONDITIONS = ("constant", "cosine")


def _curve(lr_schedule: str) -> list[float]:
    """LR at every step of the run, as the trainer would see it."""
    scheduler = ce.build_scheduler(lr_schedule)
    return [
        float(scheduler.get_lr(ce.PEAK_LR, step, ce.TOTAL_STEPS))
        for step in range(ce.TOTAL_STEPS + 1)
    ]


@pytest.mark.parametrize("lr_schedule", CONDITIONS)
def test_no_upward_jump_after_warmup(lr_schedule: str) -> None:
    curve = _curve(lr_schedule)
    tolerance = 1e-12
    for step in range(ce.WARMUP_STEPS, ce.TOTAL_STEPS):
        assert curve[step + 1] <= curve[step] + tolerance, (
            f"{lr_schedule}: LR rose from {curve[step]:.8g} at step {step} to "
            f"{curve[step + 1]:.8g} at step {step + 1} -- a warm restart, not a schedule"
        )


@pytest.mark.parametrize("lr_schedule", CONDITIONS)
def test_boundary_step_is_no_larger_than_any_in_phase_step(lr_schedule: str) -> None:
    """The curriculum/anneal seam must not be bigger than the schedule's own
    largest ordinary step. This is the actual 'no discontinuity' test: it
    does not assume the seam is exactly zero (the 1-sqrt anneal is steep by
    design just *after* the seam), only that nothing special happens *at* it.
    """
    curve = _curve(lr_schedule)
    boundary = ce.CURRICULUM_STEPS
    seam = abs(curve[boundary] - curve[boundary - 1])
    in_phase = max(
        abs(curve[step + 1] - curve[step])
        for step in range(ce.WARMUP_STEPS, boundary - 1)
    )
    assert seam <= in_phase + 1e-12, (
        f"{lr_schedule}: LR moved {seam:.8g} across the curriculum/anneal boundary "
        f"but at most {in_phase:.8g} at any earlier step -- the seam is a jump"
    )


@pytest.mark.parametrize(
    "lr_schedule,expected",
    [("constant", 1.0), ("cosine", 0.1)],
)
def test_phase_one_ends_where_the_condition_says(lr_schedule: str, expected: float) -> None:
    """Condition A holds peak through the curriculum phase; condition B is
    cosine to alpha_f=0.1. The anneal then starts from that value, so B's
    anneal is an order of magnitude shallower in absolute terms -- inherent
    to the comparison, and checked here so it stays intentional."""
    curve = _curve(lr_schedule)
    assert curve[ce.CURRICULUM_STEPS] == pytest.approx(
        ce.PEAK_LR * expected, rel=1e-6
    ), (
        f"{lr_schedule}: curriculum phase ended at {curve[ce.CURRICULUM_STEPS]:.8g}, "
        f"expected {ce.PEAK_LR * expected:.8g}"
    )


@pytest.mark.parametrize("lr_schedule", CONDITIONS)
def test_anneal_reaches_exactly_zero(lr_schedule: str) -> None:
    curve = _curve(lr_schedule)
    assert curve[ce.TOTAL_STEPS] == pytest.approx(0.0, abs=1e-12), (
        f"{lr_schedule}: final LR is {curve[ce.TOTAL_STEPS]:.8g}, not 0 -- the "
        f"reported checkpoint is then not a fully annealed model"
    )


@pytest.mark.parametrize("lr_schedule", CONDITIONS)
def test_anneal_follows_one_minus_sqrt(lr_schedule: str) -> None:
    """Luo et al.'s WSD decay shape, which is the whole reason the anneal is
    1-sqrt rather than linear or cosine. Checked at the anneal midpoint,
    where 1-sqrt and a linear decay differ by ~21% of the range and so
    cannot be confused."""
    curve = _curve(lr_schedule)
    start = curve[ce.CURRICULUM_STEPS]
    midpoint = ce.CURRICULUM_STEPS + ce.ANNEAL_STEPS // 2
    expected = start * (1.0 - (0.5 ** 0.5))
    assert curve[midpoint] == pytest.approx(expected, rel=1e-6), (
        f"{lr_schedule}: anneal midpoint LR {curve[midpoint]:.8g} != 1-sqrt "
        f"prediction {expected:.8g} (linear would give {start * 0.5:.8g})"
    )
