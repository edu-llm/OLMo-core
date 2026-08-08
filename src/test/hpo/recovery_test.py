import os

import olmo_core.hpo.worker as worker
from olmo_core.hpo.worker import ReconcileResult, reconcile_trial, trial_checkpoint_dir


def _sentinel(root, trial, step):
    d = trial_checkpoint_dir(root, trial, step)
    os.makedirs(d, exist_ok=True)
    return d


def _tokens_for(step):  # 1024 tokens per step
    return 1024 * step


def test_torn_latest_checkpoint_is_dropped_and_rolls_back(tmp_path):
    root = str(tmp_path)
    complete = _sentinel(root, "t0_0", 100)
    torn = _sentinel(root, "t0_0", 200)  # newest but incomplete

    def is_complete(path):
        return path == complete  # step200 never finished writing

    res = reconcile_trial(
        root,
        "t0_0",
        recorded_tokens=_tokens_for(200),
        step_to_tokens=_tokens_for,
        is_complete=is_complete,
    )
    assert isinstance(res, ReconcileResult)
    assert res.dropped_incomplete is True
    assert res.resume_tokens == _tokens_for(100)
    assert res.resume_dir == complete
    assert torn  # exists on disk but is not selected


def test_recorded_matches_latest_complete(tmp_path):
    root = str(tmp_path)
    d = _sentinel(root, "t0_0", 100)
    res = reconcile_trial(
        root,
        "t0_0",
        recorded_tokens=_tokens_for(100),
        step_to_tokens=_tokens_for,
        is_complete=lambda p: True,
    )
    assert res.dropped_incomplete is False
    assert res.resume_dir == d
    assert res.resume_tokens == _tokens_for(100)


def test_no_checkpoints_starts_from_scratch(tmp_path):
    res = reconcile_trial(
        str(tmp_path),
        "t0_0",
        recorded_tokens=0,
        step_to_tokens=_tokens_for,
        is_complete=lambda p: True,
    )
    assert res.resume_dir is None
    assert res.resume_tokens == 0
    assert res.dropped_incomplete is False


def test_only_torn_checkpoint_marks_allocation_dropped(tmp_path):
    root = str(tmp_path)
    _sentinel(root, "t0_0", 100)
    result = reconcile_trial(
        root,
        "t0_0",
        recorded_tokens=_tokens_for(100),
        step_to_tokens=_tokens_for,
        is_complete=lambda path: False,
    )
    assert result.resume_dir is None
    assert result.resume_tokens == 0
    assert result.dropped_incomplete is True


def test_reconcile_trial_supports_platform_checkpoint_uri(monkeypatch):
    children = [
        "s3://bucket/run/trials/t0_0/step100",
        "s3://bucket/run/trials/t0_0/step200",
    ]
    monkeypatch.setattr(worker, "list_directory", lambda *args, **kwargs: iter(children))
    result = reconcile_trial(
        "s3://bucket/run",
        "t0_0",
        recorded_tokens=_tokens_for(200),
        step_to_tokens=_tokens_for,
        is_complete=lambda path: path.endswith("step100"),
    )
    assert result.resume_dir == children[0]
    assert result.resume_tokens == _tokens_for(100)
    assert result.dropped_incomplete is True
