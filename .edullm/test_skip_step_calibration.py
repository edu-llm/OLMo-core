"""What the replay says, against planted truths rather than against W&B.

The calibration in ``skip_step_calibration.py`` is what the amendment's choice of
``--skip-step-sigma-factor`` rests on, so it needs the same treatment ``noise_floor.py`` gets:
an estimator with a test against a truth somebody planted, so that a wrong answer is a failing
test on a laptop rather than a paragraph in a pre-registration.

Run with ``pytest -v .edullm/test_skip_step_calibration.py``.
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import skip_step_calibration as calib  # noqa: E402

ROLLING = 16


def series(n: int = 200, loss: float = 2.5, grad_norm: float = 0.15) -> list:
    """
    A quiet run with just enough jitter for the rule to have a standard deviation to work
    with. A perfectly flat series has none, and ``get_step_factor`` declines nothing on it.
    """
    return [
        {
            calib.STEP_KEY: i,
            calib.CE_LOSS_KEY: loss + 0.001 * (i % 3),
            calib.GRAD_NORM_KEY: grad_norm + 0.001 * (i % 3),
        }
        for i in range(n)
    ]


def test_the_self_test_passes():
    """The tool's own planted truth, so ``--self-test`` cannot rot unnoticed."""
    calib.self_test()


def test_a_quiet_run_is_left_alone():
    out = calib.replay(series(), rolling_interval_length=ROLLING)
    assert out.count == 0
    assert out.largest_trigger == 0.0
    assert out.steps == 200


def test_a_gradient_excursion_is_declined_and_a_loss_excursion_is_too():
    """
    The rule declines when *either* signal fires. This is the property the whole amendment
    depends on: measured on stage 1, the loss channel was blind at both spike onsets and the
    gradient-norm channel caught both, so a loss-only rule would have changed nothing.
    """
    grad = series()
    grad[100][calib.GRAD_NORM_KEY] = 30.0
    grad_out = calib.replay(grad, rolling_interval_length=ROLLING)
    assert [s.step for s in grad_out.skips] == [100]
    assert grad_out.skips[0].fired_on == "gradient norm"

    loss = series()
    loss[100][calib.CE_LOSS_KEY] = 40.0
    loss_out = calib.replay(loss, rolling_interval_length=ROLLING)
    assert [s.step for s in loss_out.skips] == [100]
    assert "loss" in loss_out.skips[0].fired_on


def test_nothing_is_declined_until_the_window_has_half_filled():
    """
    ``get_step_factor`` returns 1.0 while it has fewer than ``rolling_interval_length // 2``
    observations, however large the excursion. At the tranche's 128 that is step 64, which is
    what puts it clear of the 120-step learning-rate warmup.
    """
    short = series(n=ROLLING)
    short[2][calib.GRAD_NORM_KEY] = 500.0
    assert calib.replay(short, rolling_interval_length=ROLLING).count == 0


@pytest.mark.parametrize("sigma", [4, 5, 6, 8, 10])
def test_a_raised_threshold_never_declines_more(sigma: int):
    """
    Monotone in the threshold, which is what makes the sensitivity sweep in the amendment
    readable as a sweep rather than as five unrelated numbers.
    """
    noisy = series()
    for step in (40, 90, 140):
        noisy[step][calib.GRAD_NORM_KEY] = 0.15 + step / 100.0
    counts = [
        calib.replay(noisy, sigma_factor=s, rolling_interval_length=ROLLING).count
        for s in (sigma, sigma + 2)
    ]
    assert counts[0] >= counts[1]


def test_the_largest_trigger_is_the_largest_and_not_the_latest():
    """
    The discriminator between a benign skip and a spike onset is magnitude, not count, so the
    running maximum must survive a later, smaller trigger.
    """
    noisy = series()
    noisy[100][calib.GRAD_NORM_KEY] = 20.45
    noisy[160][calib.GRAD_NORM_KEY] = 0.31
    out = calib.replay(noisy, rolling_interval_length=ROLLING)
    assert out.largest_trigger == pytest.approx(20.45)


def test_rows_missing_a_signal_are_not_counted_as_steps():
    """
    An evaluation row carries no gradient norm. A step the optimizer never saw is not a step
    it declined, and counting it would dilute the rate the amendment quotes.
    """
    rows = series(n=20)
    rows.append({calib.STEP_KEY: 20, calib.CE_LOSS_KEY: 2.5})
    rows.append({calib.STEP_KEY: 21, calib.GRAD_NORM_KEY: 0.15})
    assert calib.replay(rows, rolling_interval_length=ROLLING).steps == 20


def test_history_out_of_order_is_replayed_in_order():
    """
    ``scan_history`` is not guaranteed to arrive sorted, and a rolling window fed out of order
    is a different rule.
    """
    rows = series()
    rows[100][calib.GRAD_NORM_KEY] = 30.0
    shuffled = list(reversed(rows))
    assert [s.step for s in calib.replay(shuffled, rolling_interval_length=ROLLING).skips] == [100]


def test_the_table_names_whether_the_onset_was_caught():
    caught = calib.Replay(seed=0, steps=6000, skips=[_skip(1374), _skip(1380)])
    missed = calib.Replay(seed=1, steps=6000, skips=[_skip(5000)])

    table = calib.render([caught, missed], episodes={0: (1376, 1418), 1: (1726, 1773)})
    assert "first declined step 1374" in table
    assert "NOT CAUGHT" in table


def _skip(step: int) -> calib.Skip:
    return calib.Skip(step=step, loss=2.5, grad_norm=9.3, loss_z=0.4, grad_norm_z=10.7)
