import sys
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 14):
    sys.modules.setdefault(
        "bettermap",
        SimpleNamespace(
            ordered_map_per_thread=lambda function, values, **kwargs: map(function, values)
        ),
    )

from olmo_core.hpo.objective import EvaluatorGate
from olmo_core.hpo.worker import (
    TrialConfigArtifact,
    WorkerConfig,
    assert_data_loader_token_progress,
    reconstruct_scheduler,
)
from olmo_core.optim.scheduler import SchedulerUnits


def _artifact():
    return TrialConfigArtifact(
        payload={
            "realized_hps": {
                "lr": 1e-3,
                "warmup_fraction": 0.02,
                "decay_fraction": 0.2,
                "terminal_lr_ratio": 0.1,
            },
            "global_batch_size": 1024,
            "target_tokens": 8192,
        }
    )


def test_resume_uses_hashed_checkpoint_config():
    art = _artifact()
    h = art.content_hash
    # Reconstruction succeeds only when the immutable artifact hash matches.
    sch = reconstruct_scheduler(art, expected_hash=h)
    assert sch.units is SchedulerUnits.tokens
    assert sch.warmup_fraction == pytest.approx(0.02)


def test_reconstruction_fails_closed_on_hash_mismatch():
    art = _artifact()
    with pytest.raises(ValueError):
        reconstruct_scheduler(art, expected_hash="deadbeef")


def test_content_hash_is_stable_across_key_order():
    a = TrialConfigArtifact(payload={"a": 1, "b": {"x": 2, "y": 3}})
    b = TrialConfigArtifact(payload={"b": {"y": 3, "x": 2}, "a": 1})
    assert a.content_hash == b.content_hash


def test_worker_trial_artifact_binds_curriculum_identity():
    identity = {
        "parent": {"manifest_sha256": "a" * 64},
        "order": {"manifest_sha256": "b" * 64},
        "pacing": "arm9_warmup_quadratic_n10_token_fraction_v1",
    }
    worker = WorkerConfig(
        trial_id="trial",
        gpu=0,
        target_tokens=100,
        quantum=10,
        global_batch_size=10,
        realized_hps={"lr": 1e-3},
        checkpoint_root="/ckpt",
        evaluator_gate=EvaluatorGate(search_validation="search", untouched="final"),
        curriculum_identity=identity,
    )

    artifact = worker.config_artifact()

    assert artifact.payload["curriculum_identity"] == identity
    changed = TrialConfigArtifact(
        payload={
            **artifact.payload,
            "curriculum_identity": {
                **identity,
                "order": {"manifest_sha256": "c" * 64},
            },
        }
    )
    assert changed.content_hash != artifact.content_hash


def test_curriculum_resume_requires_loader_and_trainer_token_progress_to_match():
    matching = SimpleNamespace(
        global_train_tokens_seen=20,
        data_loader=SimpleNamespace(global_train_tokens_seen=20),
    )
    assert_data_loader_token_progress(matching)
    assert_data_loader_token_progress(
        SimpleNamespace(global_train_tokens_seen=20, data_loader=object())
    )
    assert_data_loader_token_progress(SimpleNamespace(global_train_tokens_seen=20))

    mismatched = SimpleNamespace(
        global_train_tokens_seen=20,
        data_loader=SimpleNamespace(global_train_tokens_seen=10),
    )
    with pytest.raises(RuntimeError, match="token progress"):
        assert_data_loader_token_progress(mismatched)
