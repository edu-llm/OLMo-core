from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

from production_contract import checkpoint  # noqa: E402
from production_contract import task_loss  # noqa: E402
from production_contract import wandb_artifacts as artifacts  # noqa: E402


def _labels() -> dict[str, float]:
    return {label: float(index + 1) for index, label in enumerate(task_loss.TASK_LOSS_RAW_LABELS)}


def test_permanent_ladder_and_checkpointer_contract() -> None:
    steps = checkpoint.permanent_checkpoint_steps(2360, 125)
    assert steps[0] == 0
    assert steps[-1] == 2360
    assert 2125 in steps
    assert 2250 not in steps

    kwargs = checkpoint.checkpointer_kwargs_for_ladder(2360, 125)
    assert kwargs["save_interval"] is None
    assert kwargs["ephemeral_save_interval"] is None
    assert kwargs["pre_train_checkpoint"] is True
    assert kwargs["save_async"] is False
    assert kwargs["max_checkpoints"] is None
    assert 0 not in kwargs["fixed_steps"]
    assert 2360 not in kwargs["fixed_steps"]


def test_task_loss_callback_uploads_only_the_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_folder = tmp_path / "checkpoints"
    for step in (125, 250):
        checkpoint_dir = save_folder / f"step{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "state.pt").write_bytes(b"state")

    uploads: list[tuple[int, bool]] = []

    def finalize(**kwargs):
        uploads.append((int(kwargs["step"]), bool(kwargs["upload_checkpoint"])))

    monkeypatch.setattr(task_loss, "finalize_permanent_checkpoint", finalize)
    monkeypatch.setattr(task_loss, "_HAS_OLMO_CORE", True)
    callback = task_loss.TaskLossEvalCallback(
        total_steps=250,
        save_folder=save_folder,
        run_name="unit",
        results_dir=tmp_path / "task-loss",
        eval_script=tmp_path / "eval.py",
        interval=125,
    )
    callback.trainer = type("Trainer", (), {"callbacks": {}})()
    callback._maybe_finalize(125)
    callback._maybe_finalize(250)

    assert uploads == [(125, False), (250, True)]


def test_task_loss_callback_skips_already_durable_result_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress_dir = tmp_path / "progress"
    results_dir = progress_dir / "task_loss"
    results_dir.mkdir(parents=True)
    (results_dir / "step0_task_loss.json").write_text(
        json.dumps({"task_loss_bpb": _labels()}),
        encoding="utf-8",
    )
    (progress_dir / "last_durable_step.json").write_text(
        json.dumps({"last_durable_step": 250, "task_loss_complete": True}),
        encoding="utf-8",
    )
    finalized: list[int] = []
    monkeypatch.setattr(
        task_loss,
        "finalize_permanent_checkpoint",
        lambda **kwargs: finalized.append(int(kwargs["step"])),
    )
    monkeypatch.setattr(task_loss, "_HAS_OLMO_CORE", True)
    callback = task_loss.TaskLossEvalCallback(
        total_steps=250,
        save_folder=tmp_path / "checkpoints",
        run_name="unit",
        results_dir=results_dir,
        eval_script=tmp_path / "eval.py",
        interval=125,
        progress_dir=progress_dir,
    )
    callback.trainer = type("Trainer", (), {"callbacks": {}})()

    callback._maybe_finalize(0)

    assert finalized == []
    assert callback._completed == {0}


def test_finalize_skips_nonfinal_checkpoint_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_types: list[str] = []

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name
            self.type = type

        def add_dir(self, _path):
            pass

        def add_file(self, _path, name=None):
            pass

    class Uploaded:
        def wait(self):
            pass

    class FakeRun:
        name = "unit"
        project = "unit-project"
        entity = None

        def log(self, *_args, **_kwargs):
            pass

        def log_artifact(self, artifact, aliases=None):
            artifact_types.append(artifact.type)
            return Uploaded()

    def evaluator(_checkpoint, *, out_path, **_kwargs):
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"labels": _labels()}), encoding="utf-8")

    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": FakeArtifact})())
    (tmp_path / "progress").mkdir()
    for step, upload_checkpoint in ((125, False), (250, True)):
        checkpoint_dir = tmp_path / "checkpoints" / f"step{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "state.pt").write_bytes(b"state")
        checkpoint.finalize_permanent_checkpoint(
            arm="probe",
            checkpoint_dir=checkpoint_dir,
            step=step,
            run_name="unit",
            task_loss_dir=tmp_path / "task-loss",
            task_loss_enabled=True,
            progress_dir=tmp_path / "progress",
            wandb_run=FakeRun(),
            wandb_mode="online",
            production=True,
            upload_checkpoint=upload_checkpoint,
            run_evaluator=evaluator,
        )

    assert artifact_types[:3] == ["eval", "metrics", "eval"]
    assert artifact_types[3:] == ["model", "eval", "metrics", "eval"]


def test_fingerprint_refuses_changed_scientific_identity(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    step = root / "step125"
    step.mkdir(parents=True)
    identity = {"arm": "control", "seed": 42, "total_steps": 2360}
    fingerprint = checkpoint.write_run_fingerprint(root, identity)
    checkpoint.copy_fingerprint_into_checkpoint(fingerprint, step)
    checkpoint.assert_resume_fingerprint(step, identity)

    with pytest.raises(checkpoint.CheckpointContractError, match="seed"):
        checkpoint.assert_resume_fingerprint(step, {**identity, "seed": 7})


def test_task_loss_requires_exact_twenty_raw_labels(tmp_path: Path) -> None:
    complete = {"labels": _labels()}
    assert task_loss.task_loss_payload_complete(complete)
    assert task_loss.task_loss_metrics(complete)["eval/macro_bpb"] == 10.5

    partial = {"labels": dict(list(_labels().items())[:19])}
    assert not task_loss.task_loss_payload_complete(partial)
    result = tmp_path / "partial.json"
    result.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(task_loss.TaskLossContractError, match="partial"):
        task_loss.validate_task_loss_result(result)


def test_pause_eval_reload_restores_before_strict_failure() -> None:
    events: list[str] = []

    class FakeDistributed:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def barrier() -> None:
            events.append("barrier")

    def evaluate(_checkpoint: Path, _out: Path, _run_name: str):
        events.append("eval")
        raise RuntimeError("evaluation failed")

    with pytest.raises(task_loss.TaskLossContractError, match="state was restored"):
        task_loss.pause_eval_reload_distributed(
            "step125",
            "result.json",
            "unit",
            evaluate=evaluate,
            release_train_state=lambda: events.append("release"),
            reload_train_state=lambda: events.append("reload") or object(),
            dist_module=FakeDistributed(),
            empty_device_cache=lambda: events.append("empty_cache"),
            strict=True,
        )
    assert events.index("release") < events.index("eval") < events.index("reload")


def test_pause_eval_reload_propagates_remote_rank_failure() -> None:
    events: list[str] = []

    class FakeDistributed:
        @staticmethod
        def is_initialized() -> bool:
            return True

        @staticmethod
        def barrier() -> None:
            pass

        @staticmethod
        def get_world_size() -> int:
            return 2

        @staticmethod
        def all_gather_object(output, local_error) -> None:
            output[:] = [local_error, "rank 1 failed"]

    with pytest.raises(task_loss.TaskLossContractError, match="state was restored"):
        task_loss.pause_eval_reload_distributed(
            "step125",
            "result.json",
            "unit",
            evaluate=lambda *_args: {"labels": _labels()},
            release_train_state=None,
            reload_train_state=lambda: events.append("reload") or object(),
            dist_module=FakeDistributed(),
            strict=True,
        )
    assert events == ["reload"]


def test_strict_wandb_upload_waits_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "step125"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "state.pt").write_bytes(b"state")

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name

        def add_dir(self, path):
            self.path = path

    class FailedUpload:
        def wait(self):
            raise RuntimeError("upload failed")

    class FakeRun:
        name = "unit"

        def log_artifact(self, _artifact, aliases=None):
            assert aliases == ["latest", "step-0000125"]
            return FailedUpload()

    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": FakeArtifact})())
    with pytest.raises(artifacts.WandbArtifactError, match="did not complete"):
        artifacts.wandb_log_checkpoint(FakeRun(), checkpoint_dir, step=125, strict=True)


def test_durable_marker_advances_only_after_required_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "step125"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.pt").write_bytes(b"state")
    task_loss_dir = tmp_path / "task-loss"
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()

    def evaluator(_checkpoint, *, out_path, **_kwargs):
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"labels": _labels()}), encoding="utf-8")

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.name = name

        def add_dir(self, path):
            self.path = path

        def add_file(self, path, name=None):
            self.path = path

    class FailedUpload:
        def wait(self):
            raise RuntimeError("upload failed")

    class FakeRun:
        name = "unit"
        project = "skillit-probe"
        entity = None

        def log(self, *_args, **_kwargs):
            pass

        def log_artifact(self, _artifact, aliases=None):
            return FailedUpload()

    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": FakeArtifact})())
    with pytest.raises(artifacts.WandbArtifactError):
        checkpoint.finalize_permanent_checkpoint(
            arm="probe",
            checkpoint_dir=checkpoint_dir,
            step=125,
            run_name="probe-unit",
            task_loss_dir=task_loss_dir,
            task_loss_enabled=True,
            progress_dir=progress_dir,
            wandb_run=FakeRun(),
            wandb_mode="online",
            production=True,
            upload_checkpoint=False,
            run_evaluator=evaluator,
        )
    assert checkpoint.read_last_durable_step(progress_dir) is None
