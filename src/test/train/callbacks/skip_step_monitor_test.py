from unittest.mock import Mock

import pytest
import torch

from olmo_core.optim import AdamWConfig, SkipStepAdamWConfig
from olmo_core.train.callbacks import SkipStepMonitorCallback
from olmo_core.train.callbacks.skip_step_monitor import (
    MAX_SKIP_GRAD_NORM_METRIC,
    SKIP_GRAD_NORM_METRIC,
    STEPS_SKIPPED_METRIC,
)
from olmo_core.train.common import OPTIM_GRAD_NORM_METRIC, OPTIM_STEP_SKIPPED_METRIC


def test_the_metric_names_are_pinned_to_the_strings_their_consumers_use():
    """
    ``OPTIM_STEP_SKIPPED_METRIC`` replaced a ``record_metric("step skipped",
    namespace="optim")`` call, and the two forms only agree because ``record_metric`` joins a
    namespace with exactly one slash. Consumers key on the joined string and are outside this
    repository as often as inside it -- W&B panels, saved dashboards, and the default filter
    list in ``ConsoleLoggerCallback``, which still spells both of these out. So the value is
    pinned rather than left to that join, and renaming one becomes a visible break.
    """
    assert OPTIM_STEP_SKIPPED_METRIC == "optim/step skipped"
    assert OPTIM_GRAD_NORM_METRIC == "optim/total grad norm"

    # And the constant is what the old two-argument form produced, so no run that has not
    # asked for any of this sees a key change. This is the joining rule out of
    # `Trainer.record_metric`, which is the only thing the two forms agree through.
    namespace, name = "optim", "step skipped"
    assert f"{namespace.rstrip('/')}/{name.lstrip('/')}" == OPTIM_STEP_SKIPPED_METRIC


def test_disabling_it_records_nothing_at_all():
    """
    ``enabled=False`` has to leave the metric dict untouched rather than write zeroes, or a
    disabled monitor still publishes a stability series that reads as a clean run.
    """
    callback = attached(enabled=False)
    callback.post_attach()

    metrics = {OPTIM_STEP_SKIPPED_METRIC: 1.0, OPTIM_GRAD_NORM_METRIC: 9.3}
    callback.pre_log_metrics(11, metrics)

    assert metrics == {OPTIM_STEP_SKIPPED_METRIC: 1.0, OPTIM_GRAD_NORM_METRIC: 9.3}
    assert callback.steps_skipped == 0


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.fc(x)


def attached(train_module=None, **kwargs) -> SkipStepMonitorCallback:
    callback = SkipStepMonitorCallback(**kwargs)
    trainer = Mock()
    if train_module is not None:
        trainer.train_module = train_module
    callback._trainer = trainer
    return callback


def train_module_with(optim) -> Mock:
    """A train module carrying one optimizer, and nothing else the callback may reach for."""
    return Mock(optim=optim, optimizers=None, spec=["optim", "optimizers"])


def step(callback, step_no: int, *, skipped: bool, grad_norm: float) -> dict:
    metrics = {
        OPTIM_STEP_SKIPPED_METRIC: 1.0 if skipped else 0.0,
        OPTIM_GRAD_NORM_METRIC: grad_norm,
    }
    callback.pre_log_metrics(step_no, metrics)
    return metrics


def test_a_skipped_step_records_its_own_step_number_and_trigger():
    callback = attached()

    clean = step(callback, 10, skipped=False, grad_norm=0.15)
    assert SKIP_GRAD_NORM_METRIC not in clean, "an unskipped step must not claim a trigger"
    assert clean[STEPS_SKIPPED_METRIC] == 0.0

    spiked = step(callback, 11, skipped=True, grad_norm=9.30)
    assert spiked[SKIP_GRAD_NORM_METRIC] == pytest.approx(9.30)
    assert spiked[STEPS_SKIPPED_METRIC] == 1.0
    assert spiked[MAX_SKIP_GRAD_NORM_METRIC] == pytest.approx(9.30)

    assert callback._skipped_steps == [11]
    assert callback.steps_skipped == 1


def test_the_running_maximum_holds_the_largest_trigger_and_not_the_latest():
    """
    The count alone does not separate a run that skipped a dozen unremarkable steps from one
    that skipped the onset of a spike. The largest trigger does, so it must not be overwritten
    by a later, smaller one.
    """
    callback = attached()
    step(callback, 1, skipped=True, grad_norm=20.45)
    last = step(callback, 2, skipped=True, grad_norm=0.31)

    assert last[SKIP_GRAD_NORM_METRIC] == pytest.approx(0.31)
    assert last[MAX_SKIP_GRAD_NORM_METRIC] == pytest.approx(20.45)


def test_the_count_survives_a_resume():
    """
    A second attempt resumes from the last checkpoint. A count that restarted at zero there
    would under-report exactly the cells that lost a host, and nothing else would say so.
    """
    first = attached()
    step(first, 100, skipped=True, grad_norm=9.3)
    step(first, 101, skipped=False, grad_norm=0.14)

    second = attached()
    second.load_state_dict(first.state_dict())
    resumed = step(second, 200, skipped=True, grad_norm=0.4)

    assert second._skipped_steps == [100, 200]
    assert resumed[STEPS_SKIPPED_METRIC] == 2.0
    assert resumed[MAX_SKIP_GRAD_NORM_METRIC] == pytest.approx(9.3)


def test_a_step_that_recorded_no_optimizer_metrics_is_not_counted_as_a_step():
    """
    The evaluations at step 0 reach the metric flush without an optimizer step behind them.
    """
    callback = attached()
    metrics = {"eval/lm/wiki/CE loss": 3.0}
    callback.pre_log_metrics(0, metrics)

    assert metrics == {"eval/lm/wiki/CE loss": 3.0}
    assert callback.steps_skipped == 0


def test_a_grad_norm_that_was_never_recorded_still_counts_the_skip():
    """
    ``optim/total grad norm`` only exists when the run clips. The count is the primary outcome
    and must not be lost because the trigger cannot be named.
    """
    callback = attached()
    metrics = {OPTIM_STEP_SKIPPED_METRIC: 1.0}
    callback.pre_log_metrics(7, metrics)

    assert callback.steps_skipped == 1
    assert metrics[STEPS_SKIPPED_METRIC] == 1.0
    assert SKIP_GRAD_NORM_METRIC not in metrics


def test_it_refuses_a_run_whose_optimizer_can_never_skip():
    """
    Zero skipped steps reads as a perfectly stable arm. An arm that lost the optimizer setting
    has to fail at start-up instead, or the stability outcome is a null that was never measured.
    """
    model = MyModel()
    callback = attached(train_module_with(AdamWConfig().build(model)))

    with pytest.raises(RuntimeError, match="not a SkipStepOptimizer"):
        callback.post_attach()


def test_it_attaches_to_a_run_whose_optimizer_can_skip():
    model = MyModel()
    callback = attached(train_module_with(SkipStepAdamWConfig().build(model)))

    callback.post_attach()


def test_the_summary_names_the_count_the_largest_trigger_and_the_steps():
    callback = attached()
    assert callback.summary() == "no steps skipped"

    step(callback, 1374, skipped=True, grad_norm=0.282)
    step(callback, 1376, skipped=True, grad_norm=9.30)
    summary = callback.summary()

    assert "2 step(s) skipped" in summary
    assert "9.3" in summary
    assert "1374, 1376" in summary
