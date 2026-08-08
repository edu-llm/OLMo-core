import os

from olmo_core.hpo.worker import (
    controller_dir,
    latest_step_dir,
    trial_checkpoint_dir,
    trial_namespace,
)


def _norm(p: str) -> str:
    return p.replace(os.sep, "/")


def test_two_trials_do_not_share_step_directories(tmp_path):
    root = str(tmp_path)
    a = trial_checkpoint_dir(root, "t0_0", 100)
    b = trial_checkpoint_dir(root, "t1_0", 100)
    assert a != b
    assert _norm(a).endswith("trials/t0_0/step100")
    assert _norm(b).endswith("trials/t1_0/step100")
    assert _norm(controller_dir(root)).endswith("/controller")


def test_latest_checkpoint_is_scoped_to_one_trial_namespace(tmp_path):
    root = str(tmp_path)
    os.makedirs(trial_checkpoint_dir(root, "t0_0", 100))
    os.makedirs(trial_checkpoint_dir(root, "t0_0", 200))
    os.makedirs(trial_checkpoint_dir(root, "t1_0", 300))  # different trial, higher step
    latest = latest_step_dir(root, "t0_0")
    # Must ignore the shared run root / other trials entirely.
    assert _norm(latest).endswith("trials/t0_0/step200")
    assert _norm(trial_namespace(root, "t0_0")).endswith("trials/t0_0")


def test_latest_step_dir_none_when_empty(tmp_path):
    assert latest_step_dir(str(tmp_path), "nope") is None
