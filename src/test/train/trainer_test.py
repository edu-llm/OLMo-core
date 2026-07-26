"""Tests for trainer-level metric validation."""

import dataclasses

import pytest

from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC
from olmo_core.train.trainer import Trainer, _raise_on_nonfinite_metrics

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NON_FINITE)
def test_raises_on_nonfinite_ce_loss(value: float):
    """The loss check is upstream behavior and is always on, regardless of the grad-norm gate."""
    with pytest.raises(RuntimeError, match=TRAIN_CE_LOSS_METRIC):
        _raise_on_nonfinite_metrics(7, {TRAIN_CE_LOSS_METRIC: value})
    with pytest.raises(RuntimeError, match=TRAIN_CE_LOSS_METRIC):
        _raise_on_nonfinite_metrics(7, {TRAIN_CE_LOSS_METRIC: value}, check_grad_norm=True)


@pytest.mark.parametrize("value", NON_FINITE)
def test_grad_norm_is_not_checked_by_default(value: float):
    """
    The gradient-norm check must be opt-in so the default matches upstream OLMo-core.

    Raising on a non-finite grad norm is a new failure mode this fork adds; a base model
    (OLMo-2/OLMo-3) run through this trainer must not start hard-failing on a transient NaN grad
    norm it previously absorbed. So with the default flag off, a non-finite grad norm passes the
    metric check untouched (the loss check remains the only always-on guard).
    """
    _raise_on_nonfinite_metrics(7, {OPTIM_GRAD_NORM_METRIC: value})


@pytest.mark.parametrize("value", NON_FINITE)
def test_raises_on_nonfinite_grad_norm_when_enabled(value: float):
    """
    With the gate on, a non-finite gradient norm fails as loudly as a non-finite loss.

    Nothing in upstream inspected this metric, and a NaN grad norm also poisons
    :meth:`SkipStepOptimizer.get_step_factor`'s rolling statistics, so an ungated run would keep
    stepping with the weights frozen and the loss curve looking healthy. Opting in restores loud
    failure for runs that want it.
    """
    with pytest.raises(RuntimeError, match=OPTIM_GRAD_NORM_METRIC):
        _raise_on_nonfinite_metrics(7, {OPTIM_GRAD_NORM_METRIC: value}, check_grad_norm=True)


def test_error_names_the_metric_and_step():
    """The message has to be actionable: which metric, which step, what value."""
    with pytest.raises(RuntimeError) as exc_info:
        _raise_on_nonfinite_metrics(
            42, {OPTIM_GRAD_NORM_METRIC: float("nan")}, check_grad_norm=True
        )
    message = str(exc_info.value)
    assert OPTIM_GRAD_NORM_METRIC in message
    assert "42" in message
    assert "nan" in message.lower()


def test_allows_finite_metrics():
    _raise_on_nonfinite_metrics(
        0, {TRAIN_CE_LOSS_METRIC: 3.2, OPTIM_GRAD_NORM_METRIC: 0.9}, check_grad_norm=True
    )


def test_ignores_absent_metrics():
    """Not every step reports every metric; absence must not be mistaken for a bad value."""
    _raise_on_nonfinite_metrics(0, {})
    _raise_on_nonfinite_metrics(0, {"some/other metric": float("nan")}, check_grad_norm=True)


def test_trainer_defaults_grad_norm_check_off():
    """The Trainer dataclass field must exist and default to upstream behavior (off)."""
    field = {f.name: f for f in dataclasses.fields(Trainer)}["raise_on_nonfinite_grad_norm"]
    assert field.default is False


def test_trainer_config_exposes_grad_norm_gate_and_forwards_it():
    """
    The gate must be settable from ``TrainerConfig``, not just on the ``Trainer`` dataclass.

    ``TrainerConfig.build`` forwards its fields to ``Trainer`` by name via ``**kwargs``, so the
    config field must (a) exist and default to upstream-off and (b) share the Trainer field's
    name -- otherwise the flag is unreachable from a config-driven run, which is how every real
    experiment is launched.
    """
    from olmo_core.train import TrainerConfig

    config_field = {f.name: f for f in dataclasses.fields(TrainerConfig)}.get(
        "raise_on_nonfinite_grad_norm"
    )
    assert config_field is not None, "TrainerConfig must expose the gate"
    assert config_field.default is False

    trainer_fields = {f.name for f in dataclasses.fields(Trainer)}
    assert (
        "raise_on_nonfinite_grad_norm" in trainer_fields
    ), "name parity is what makes **kwargs forward"
