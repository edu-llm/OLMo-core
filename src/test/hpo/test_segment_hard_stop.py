from types import SimpleNamespace

import pytest
import torch

from olmo_core.hpo.worker import (
    HpoDiagnosticsCallback,
    SegmentComplete,
    SegmentSpec,
    execute_segment,
    next_absolute_hard_stop,
)
from olmo_core.train.common import Duration, DurationUnit


def test_resume_refreshes_absolute_hard_stop_and_trains():
    hs = next_absolute_hard_stop(loaded_tokens=1024, target_tokens=8192, quantum=2048)
    assert isinstance(hs, Duration)
    assert hs.unit is DurationUnit.tokens
    assert hs == Duration.tokens(3072)
    # Strictly above the loaded token count so the segment is guaranteed non-empty.
    assert hs.value > 1024


def test_hard_stop_caps_at_target():
    assert next_absolute_hard_stop(7000, 8192, 2048) == Duration.tokens(8192)


def test_already_complete_lineage_fails_closed():
    with pytest.raises(SegmentComplete):
        next_absolute_hard_stop(8192, 8192, 2048)
    with pytest.raises(SegmentComplete):
        next_absolute_hard_stop(9000, 8192, 2048)


def test_execute_segment_loads_refreshes_stop_saves_and_returns_diagnostics():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 3072
    diagnostics.train_ce = 3.7
    diagnostics.train_ce_history = [4.0, 3.7]
    diagnostics.grad_norm_history = [1.0, 1.2]
    diagnostics.tokens_seen = 3072

    class FakeTrainer:
        def __init__(self):
            self.global_train_tokens_seen = 0
            self.hard_stop = None
            self.loaded = False
            self.fit_called = False

        def maybe_load_checkpoint(self):
            self.loaded = True
            self.global_train_tokens_seen = 1024
            return True

        def fit(self):
            assert self.hard_stop == Duration.tokens(3072)
            self.fit_called = True
            self.global_train_tokens_seen = 3072

        def save_checkpoint(self):
            return "/ckpt/trials/t0/step3"

    trainer = FakeTrainer()
    result = execute_segment(
        trainer,
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=8192,
            hard_stop_tokens=3072,
            lineage_global_batch_size=1024,
        ),
        actual_global_batch_size=1024,
    )
    assert trainer.loaded and trainer.fit_called
    assert result.tokens == 3072
    assert result.heldout_ce == 3.5
    assert result.checkpoint_ref == "/ckpt/trials/t0/step3"
    assert result.train_ce_history == (4.0, 3.7)


def test_execute_segment_does_not_resave_post_train_checkpoint():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 1024
    diagnostics.tokens_seen = 1024

    class FakeTrainer:
        def __init__(self):
            self.global_train_tokens_seen = 0
            self.hard_stop = None
            self.callbacks = {
                "checkpointer": SimpleNamespace(_latest_checkpoint_path="/ckpt/trials/t0/step1")
            }

        def maybe_load_checkpoint(self, *args, **kwargs):
            return False

        def fit(self):
            self.global_train_tokens_seen = 1024

        def save_checkpoint(self):
            raise AssertionError("post_train checkpoint must not be saved twice")

    result = execute_segment(
        FakeTrainer(),
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=2048,
            hard_stop_tokens=1024,
            lineage_global_batch_size=1024,
        ),
        actual_global_batch_size=1024,
    )
    assert result.checkpoint_ref == "/ckpt/trials/t0/step1"


def test_execute_segment_loads_donor_before_validating_hard_stop():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 3072
    diagnostics.tokens_seen = 3072

    class FakeTrainer:
        load_path = "/donor/step2"
        load_trainer_state = False
        load_optim_state = False

        def __init__(self):
            self.global_train_tokens_seen = 0
            self.hard_stop = None
            self.callbacks = {}
            self.loaded = []

        def maybe_load_checkpoint(self, path=None, **kwargs):
            self.loaded.append(path)
            if path == self.load_path:
                self.global_train_tokens_seen = 2048
                return True
            return False

        def fit(self):
            assert self.hard_stop == Duration.tokens(3072)
            self.global_train_tokens_seen = 3072

        def save_checkpoint(self):
            return "/ckpt/t0/step3"

    trainer = FakeTrainer()
    execute_segment(
        trainer,
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=8192,
            hard_stop_tokens=3072,
            lineage_global_batch_size=1024,
        ),
        actual_global_batch_size=1024,
    )
    assert trainer.loaded == [None, "/donor/step2"]


def test_resume_without_new_eval_rejects_stale_heldout_ce():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 1024

    class FakeTrainer:
        load_path = None

        def __init__(self):
            self.global_train_tokens_seen = 1024
            self.hard_stop = None
            self.callbacks = {}

        def maybe_load_checkpoint(self, *args, **kwargs):
            return True

        def fit(self):
            self.global_train_tokens_seen = 2048

        def save_checkpoint(self):
            return "/ckpt/t0/step2"

    with pytest.raises(RuntimeError, match="fresh search-validation"):
        execute_segment(
            FakeTrainer(),
            diagnostics=diagnostics,
            spec=SegmentSpec(
                trial_id="t0",
                target_tokens=4096,
                hard_stop_tokens=2048,
                lineage_global_batch_size=1024,
            ),
            actual_global_batch_size=1024,
        )


def test_shrink_perturb_transition_changes_loaded_donor_weights():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 3072
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)

    class FakeTrainer:
        load_path = "/donor"
        load_trainer_state = True
        load_optim_state = False

        def __init__(self):
            self.train_module = SimpleNamespace(model=model)
            self.global_train_tokens_seen = 0
            self.hard_stop = None
            self.callbacks = {}

        def maybe_load_checkpoint(self, path=None, **kwargs):
            if path == self.load_path:
                with torch.no_grad():
                    model.weight.fill_(10.0)
                self.global_train_tokens_seen = 2048
                return True
            return False

        def fit(self):
            assert model.weight.item() == pytest.approx(5.2)
            self.global_train_tokens_seen = 3072

        def save_checkpoint(self):
            return "/ckpt/t0/step3"

    execute_segment(
        FakeTrainer(),
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=8192,
            hard_stop_tokens=3072,
            lineage_global_batch_size=1024,
            transition={"weight_policy": "shrink_perturb", "weight_scale": 0.4},
        ),
        actual_global_batch_size=1024,
    )


def test_ipbt_batch_size_mutation_authorizes_loader_rebase_before_checkpoint_load():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 4096
    diagnostics.tokens_seen = 4096
    model = torch.nn.Linear(1, 1, bias=False)

    class FakeLoader:
        global_train_tokens_seen = 0
        rebase_authorized = False

        def allow_batch_size_rebase(self):
            self.rebase_authorized = True

    class FakeTrainer:
        load_path = "/donor"
        load_trainer_state = True
        load_optim_state = False

        def __init__(self):
            self.data_loader = FakeLoader()
            self.train_module = SimpleNamespace(model=model)
            self.global_train_tokens_seen = 0
            self.hard_stop = None
            self.callbacks = {}

        def maybe_load_checkpoint(self, *args, **kwargs):
            assert self.data_loader.rebase_authorized
            self.global_train_tokens_seen = 2048
            self.data_loader.global_train_tokens_seen = 2048
            return True

        def fit(self):
            self.global_train_tokens_seen = 4096
            self.data_loader.global_train_tokens_seen = 4096

        def save_checkpoint(self):
            return "/ckpt/t0/step4"

    execute_segment(
        FakeTrainer(),
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=8192,
            hard_stop_tokens=4096,
            lineage_global_batch_size=2048,
            transition={
                "transition_kind": "generation",
                "parent_trial_id": "donor",
                "weight_policy": "shrink_perturb",
                "weight_scale": 0.4,
                "optimizer_reset": True,
            },
        ),
        actual_global_batch_size=2048,
    )


def test_completed_boundary_checkpoint_recovers_observation_without_retraining():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 1024
    diagnostics.train_ce_history = [3.8, 3.5]
    diagnostics.grad_norm_history = [1.0]

    class FakeTrainer:
        load_path = None

        def __init__(self):
            self.global_train_tokens_seen = 0
            self.callbacks = {
                "checkpointer": SimpleNamespace(_latest_checkpoint_path="/ckpt/t0/step1")
            }

        def maybe_load_checkpoint(self, *args, **kwargs):
            self.global_train_tokens_seen = 1024
            return True

        def fit(self):
            raise AssertionError("completed allocation must not retrain")

    result = execute_segment(
        FakeTrainer(),
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=2048,
            hard_stop_tokens=1024,
            lineage_global_batch_size=1024,
        ),
        actual_global_batch_size=1024,
    )
    assert result.tokens == 1024
    assert result.checkpoint_ref == "/ckpt/t0/step1"


def test_completed_redispatch_reports_checkpoint_loaded_from_trial_folder():
    diagnostics = HpoDiagnosticsCallback()
    diagnostics.heldout_ce = 3.5
    diagnostics.heldout_tokens_seen = 1024

    class FakeTrainer:
        load_path = "/donor/t0/step0"
        save_folder = "/ckpt/t0"

        def __init__(self):
            self.global_train_tokens_seen = 0
            self.callbacks = {"checkpointer": SimpleNamespace(_latest_checkpoint_path="")}
            self.checkpointer = SimpleNamespace(
                latest_checkpoint=lambda path: f"{path}/step1",
            )

        def maybe_load_checkpoint(self, *args, **kwargs):
            self.global_train_tokens_seen = 1024
            return True

        def fit(self):
            raise AssertionError("completed allocation must not retrain")

    result = execute_segment(
        FakeTrainer(),
        diagnostics=diagnostics,
        spec=SegmentSpec(
            trial_id="t0",
            target_tokens=2048,
            hard_stop_tokens=1024,
            lineage_global_batch_size=1024,
        ),
        actual_global_batch_size=1024,
    )
    assert result.checkpoint_ref == "/ckpt/t0/step1"
