import pytest

from olmo_core.hpo.objective import EvaluatorGate
from olmo_core.hpo.worker import WorkerConfig


def _wc(**over):
    base = dict(
        trial_id="t0_0",
        gpu=0,
        target_tokens=8192,
        quantum=2048,
        global_batch_size=1024,
        realized_hps={
            "lr": 1e-3,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
        },
        checkpoint_root="/tmp/run",
        evaluator_gate=EvaluatorGate(search_validation="val", untouched="holdout"),
    )
    base.update(over)
    return WorkerConfig(**base)


def test_segment_boundary_requires_search_validation():
    wc = _wc()
    wc.assert_evaluator_ready({"val", "downstream"})  # present -> ok
    with pytest.raises(RuntimeError):
        wc.assert_evaluator_ready({"downstream"})  # missing -> fail closed


def test_untouched_split_is_not_a_substitute_for_search_validation():
    wc = _wc()
    with pytest.raises(RuntimeError):
        wc.assert_evaluator_ready({"holdout"})  # only the untouched split present
