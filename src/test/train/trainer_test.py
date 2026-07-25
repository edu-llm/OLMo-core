"""Tests for trainer-level metric validation."""

import pytest

from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC
from olmo_core.train.trainer import _raise_on_nonfinite_metrics

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NON_FINITE)
def test_raises_on_nonfinite_ce_loss(value: float):
    with pytest.raises(RuntimeError, match=TRAIN_CE_LOSS_METRIC):
        _raise_on_nonfinite_metrics(7, {TRAIN_CE_LOSS_METRIC: value})


@pytest.mark.parametrize("value", NON_FINITE)
def test_raises_on_nonfinite_grad_norm(value: float):
    """
    A non-finite gradient norm must fail as loudly as a non-finite loss.

    It previously did not, and the consequences were entirely silent. Nothing in the trainer
    inspected this metric, and a NaN gradient norm also poisons
    :meth:`SkipStepOptimizer.get_step_factor`'s rolling statistics, so the run would keep
    stepping with the weights frozen and the loss curve looking healthy.
    """
    with pytest.raises(RuntimeError, match=OPTIM_GRAD_NORM_METRIC):
        _raise_on_nonfinite_metrics(7, {OPTIM_GRAD_NORM_METRIC: value})


def test_error_names_the_metric_and_step():
    """The message has to be actionable: which metric, which step, what value."""
    with pytest.raises(RuntimeError) as exc_info:
        _raise_on_nonfinite_metrics(42, {OPTIM_GRAD_NORM_METRIC: float("nan")})
    message = str(exc_info.value)
    assert OPTIM_GRAD_NORM_METRIC in message
    assert "42" in message
    assert "nan" in message.lower()


def test_allows_finite_metrics():
    _raise_on_nonfinite_metrics(0, {TRAIN_CE_LOSS_METRIC: 3.2, OPTIM_GRAD_NORM_METRIC: 0.9})


def test_ignores_absent_metrics():
    """Not every step reports every metric; absence must not be mistaken for a bad value."""
    _raise_on_nonfinite_metrics(0, {})
    _raise_on_nonfinite_metrics(0, {"some/other metric": float("nan")})
