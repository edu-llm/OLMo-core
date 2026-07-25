"""
State-tracking capability tests for the Mamba-3 mixer (design note section 6.3).

The claim under test: widening the transition block from 2x2 to b x b makes the mixer's
transition monoid non-solvable, so it can track the A_5 word problem at lengths it never saw in
training, while ``b=2`` -- whose cumulative rotation is a cumulative *sum* of angles, and so is
in TC^0 -- cannot.

The real comparison needs a GPU and takes minutes, so it is gated twice: on a GPU being present
and on ``OLMO_RUN_SLOW_STATE_TRACKING=1``. The CPU test below covers the plumbing, so a broken
harness is caught without waiting for the capability run.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from olmo_core.testing import requires_gpu

HARNESS_PATH = Path("src/scripts/train/smoketests/a5_harness.py")

RUN_SLOW = os.environ.get("OLMO_RUN_SLOW_STATE_TRACKING") == "1"
requires_slow = pytest.mark.skipif(
    not RUN_SLOW,
    reason="minutes-scale capability run; set OLMO_RUN_SLOW_STATE_TRACKING=1 to enable",
)


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    if not HARNESS_PATH.exists():
        pytest.skip(f"{HARNESS_PATH} not found; run pytest from the repo root")
    sys.path.insert(0, str(HARNESS_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("a5_harness", HARNESS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("rotation_block_size", [2, 3])
def test_a5_comparison_harness_runs_both_arms(harness, rotation_block_size: int):
    """
    Both arms of the comparison must train and evaluate end to end on CPU at toy scale.

    This is deliberately not a capability assertion -- a model this small trained for five steps
    learns nothing. It exists so that a broken arm (a shape error at ``b=3``, an untrainable
    configuration, a harness that cannot build one of the two models) surfaces in seconds rather
    than after a multi-minute GPU run, and so that a failure there can be attributed to the
    algebra rather than to the plumbing.
    """
    model = harness.build_a5_model(
        rotation_block_size=rotation_block_size,
        n_layers=1,
        d_model=32,
        n_heads=2,
        d_state=12,
        seed=0,
    )
    losses = harness.train_a5(model, train_len=8, steps=5, batch_size=4, lr=1e-3, seed=0)
    assert len(losses) == 5
    assert all(loss == loss for loss in losses), "NaN loss"

    accuracy = harness.evaluate_a5(model, seq_len=8, batch_size=8, seed=0)
    assert 0.0 <= accuracy <= 1.0


def test_untrained_accuracy_is_at_chance_for_every_block_size(harness):
    """
    The negative control's control, across the axis under test.

    If widening the block somehow leaked the label -- say through a shape bug that let the
    readout see the current token -- an untrained ``b=3`` model would score above chance and the
    headline result would be an artifact. Chance is 1/60.
    """
    for rotation_block_size in (2, 3, 4):
        model = harness.build_a5_model(
            rotation_block_size=rotation_block_size,
            n_layers=1,
            d_model=32,
            n_heads=2,
            d_state=12,
            seed=0,
        )
        accuracy = harness.evaluate_a5(model, seq_len=16, batch_size=16, seed=0)
        assert accuracy <= 0.15, (
            f"untrained b={rotation_block_size} scores {accuracy:.1%}, far above the 1.7% "
            f"chance rate -- the harness is leaking"
        )


@requires_gpu
@requires_slow
@pytest.mark.parametrize("rotation_block_size", [2, 3])
def test_a5_word_problem_length_extrapolation(harness, rotation_block_size: int):
    """
    Train at length <= 40, evaluate at 64/128/256.

    The gate from the design note: ``b >= 3`` reaches >= 90% at length 256 while ``b = 2`` stays
    at or below 40%. PD-SSM reports 15.5% for a two-layer complex-diagonal model on this task,
    so a ``b=2`` arm scoring highly means the harness is leaking rather than that the abelian
    model has succeeded -- which is why the control is asserted as an upper bound, not skipped.

    Configuration notes, all of which matter: ``mimo_rank=1`` because MIMO adds no expressivity;
    ``n_groups=n_heads`` so each head has its own rotation schedule; and a small
    ``a_log_init_max`` so the decay horizon actually covers 256 steps. Getting the last one
    wrong makes this fail for optimization reasons that have nothing to do with the algebra.
    """
    import torch

    device = torch.device("cuda")
    model = harness.build_a5_model(
        rotation_block_size=rotation_block_size,
        n_layers=1,
        d_model=128,
        n_heads=4,
        d_state=48,
        mimo_rank=1,
        a_log_init_max=0.1,
        seed=0,
    )
    harness.train_a5(model, train_len=40, steps=4000, batch_size=64, lr=3e-4, seed=0, device=device)

    accuracy_256 = harness.evaluate_a5(model, seq_len=256, batch_size=256, seed=1, device=device)
    if rotation_block_size == 2:
        assert accuracy_256 <= 0.40, (
            f"abelian b=2 reached {accuracy_256:.1%} at length 256. A cumulative sum of angles "
            f"is in TC^0 and cannot track A_5 at length, so this indicates a leaking harness "
            f"rather than a capable model."
        )
    else:
        assert accuracy_256 >= 0.90, (
            f"b={rotation_block_size} only reached {accuracy_256:.1%} at length 256. Before "
            f"concluding the algebra is insufficient, check the decay horizon "
            f"(a_log_init_max) and n_groups -- both fail this test for optimization reasons."
        )
