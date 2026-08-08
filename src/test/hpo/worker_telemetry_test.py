from olmo_core.hpo.worker import HpoDiagnosticsCallback
from olmo_core.train.callbacks import Callback


class _FakeTrainer:
    def __init__(self, step, tokens):
        self.global_step = step
        self.global_train_tokens_seen = tokens


def test_callback_is_an_olmo_callback():
    assert issubclass(HpoDiagnosticsCallback, Callback)


def test_callback_records_ce_grad_and_token_progress():
    cb = HpoDiagnosticsCallback(heldout_metric="eval/val CE")
    cb.trainer = _FakeTrainer(step=10, tokens=2048)
    cb.log_metrics(
        10,
        {
            "eval/val CE": 3.5,
            "optim/total grad norm": 1.2,
            "train/CE loss": 3.9,
            "hpo/activation effective support": 0.42,
        },
    )
    snap = cb.snapshot()
    assert snap["heldout_ce"] == 3.5
    assert snap["train_ce"] == 3.9
    assert snap["tokens_seen"] == 2048
    assert snap["grad_norm_history"][-1] == 1.2
    assert snap["activation_ratio"] == 0.42


def test_default_heldout_metric_matches_olmo_evaluator_output():
    cb = HpoDiagnosticsCallback()
    cb.trainer = _FakeTrainer(step=10, tokens=2048)
    cb.log_metrics(10, {"eval/search_validation/val/CE loss": 3.5})
    assert cb.snapshot()["heldout_ce"] == 3.5


def test_numeric_failure_flag_and_bounded_history():
    cb = HpoDiagnosticsCallback(heldout_metric="eval/val CE", max_history=3)
    cb.trainer = _FakeTrainer(step=1, tokens=128)
    for i in range(10):
        cb.trainer = _FakeTrainer(step=i, tokens=128 * (i + 1))
        cb.log_metrics(
            i, {"train/CE loss": float("nan") if i == 5 else 3.0, "optim/total grad norm": float(i)}
        )
    assert cb.snapshot()["numeric_failure"] is True
    # History is bounded to max_history.
    assert len(cb.snapshot()["grad_norm_history"]) == 3


def test_fresh_nonfinite_heldout_clears_restored_finite_value():
    cb = HpoDiagnosticsCallback()
    cb.heldout_ce = 3.5
    cb.heldout_tokens_seen = 1024
    cb.trainer = _FakeTrainer(step=2, tokens=2048)
    cb.log_metrics(2, {cb.heldout_metric: float("nan")})
    snapshot = cb.snapshot()
    assert snapshot["numeric_failure"] is True
    assert snapshot["heldout_ce"] is None
    assert snapshot["heldout_tokens_seen"] == 2048


def test_observation_hash_is_stable_and_content_bound():
    cb = HpoDiagnosticsCallback(heldout_metric="eval/val CE")
    cb.trainer = _FakeTrainer(step=10, tokens=2048)
    cb.log_metrics(10, {"eval/val CE": 3.5})
    h1 = cb.observation_hash()
    h2 = cb.observation_hash()
    assert h1 == h2 and len(h1) == 64
    cb.trainer = _FakeTrainer(step=11, tokens=4096)
    cb.log_metrics(11, {"eval/val CE": 3.4})
    assert cb.observation_hash() != h1  # new evidence -> new hash


def test_post_save_completion_is_not_claimed_inside_checkpointed_callback_state():
    cb = HpoDiagnosticsCallback()
    cb.post_checkpoint_saved("/ckpt/step1")
    assert cb.snapshot()["checkpoint_saved"] is True
    assert "checkpoint_saved" not in cb.state_dict()
    restored = HpoDiagnosticsCallback()
    restored.load_state_dict(cb.state_dict())
    assert restored.snapshot()["checkpoint_saved"] is False
