import pytest

from olmo_core.hpo.worker import TrialConfigArtifact, reconstruct_scheduler
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
