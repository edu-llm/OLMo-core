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


def test_task_loss_callback_never_uploads_a_model_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Campaign policy: model weights never leave FarmShare scratch.

    This previously asserted the final checkpoint WAS uploaded. A 370M
    checkpoint is multi-GB, W&B stages a second local copy under $HOME, and
    FarmShare home has a 48 GB quota -- so the final step must be False too,
    not just the intermediate ones.
    """
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

    assert uploads == [(125, False), (250, False)]


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
    for step, upload_checkpoint in ((125, False), (250, False)):
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

    # Eval and metrics artifacts still upload at every permanent checkpoint;
    # no "model" artifact appears at any step, including the final one.
    assert artifact_types == ["eval", "metrics", "eval", "eval", "metrics", "eval"]
    assert "model" not in artifact_types


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
    # Exercise the upload mechanics, which the campaign policy otherwise
    # blocks outright (see test_model_artifact_upload_is_refused_by_policy).
    monkeypatch.setattr(artifacts, "ALLOW_MODEL_ARTIFACT_UPLOAD", True)
    with pytest.raises(artifacts.WandbArtifactError, match="did not complete"):
        artifacts.wandb_log_checkpoint(FakeRun(), checkpoint_dir, step=125, strict=True)


def test_model_artifact_upload_is_refused_by_policy(tmp_path: Path) -> None:
    """The refusal lives at the upload function, not only at its callers.

    A call site can be edited back by accident; the cost of that mistake is
    a full 48 GB home quota discovered hours into a run, so the block is
    placed where the bytes would actually move.
    """
    checkpoint_dir = tmp_path / "step2500"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "state.pt").write_bytes(b"state")

    assert artifacts.ALLOW_MODEL_ARTIFACT_UPLOAD is False
    with pytest.raises(artifacts.WandbArtifactError, match="uploads are disabled"):
        artifacts.wandb_log_checkpoint(
            object(), checkpoint_dir, step=2500, strict=False
        )


def test_small_artifacts_still_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The policy blocks model weights only. Eval results, metrics and other
    small directory artifacts are exactly what W&B is still for here."""
    logged: list[str] = []

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            self.type = type

        def add_dir(self, path):
            pass

        def add_file(self, path, name=None):
            pass

    class FakeRun:
        name = "unit"

        def log_artifact(self, artifact, aliases=None):
            logged.append(artifact.type)
            return type("Done", (), {"wait": lambda self: None})()

    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": FakeArtifact})())
    metrics_dir = tmp_path / "progress"
    metrics_dir.mkdir()
    (metrics_dir / "progress.json").write_text("{}", encoding="utf-8")

    artifacts.wandb_log_directory_artifact(
        FakeRun(), metrics_dir, name="unit-progress", artifact_type="metrics", strict=True
    )
    assert logged == ["metrics"]


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


def test_prune_keeps_only_latest_durable_checkpoint(tmp_path: Path) -> None:
    save_folder = tmp_path / "checkpoints"
    for step in (0, 125, 250):
        checkpoint_dir = save_folder / f"step{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "state.pt").write_bytes(b"state")
        (checkpoint_dir / "model_eval.pt").write_bytes(b"eval")

    assert checkpoint.discard_regenerable_eval_weights(save_folder / "step250") is True
    assert not (save_folder / "step250" / "model_eval.pt").is_file()

    removed = checkpoint.prune_older_permanent_checkpoints(save_folder, keep_step=250)
    assert sorted(path.name for path in removed) == ["step0", "step125"]
    assert [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)] == [
        "step250"
    ]
    assert (save_folder / "step250" / "state.pt").is_file()


def test_prune_leaves_newer_incomplete_checkpoints_alone(tmp_path: Path) -> None:
    save_folder = tmp_path / "checkpoints"
    for step in (0, 125, 250):
        checkpoint_dir = save_folder / f"step{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "state.pt").write_bytes(b"state")

    removed = checkpoint.prune_older_permanent_checkpoints(save_folder, keep_step=125)
    assert [path.name for path in removed] == ["step0"]
    assert [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)] == [
        "step125",
        "step250",
    ]


def test_prune_refuses_to_delete_missing_keep_step(tmp_path: Path) -> None:
    save_folder = tmp_path / "checkpoints"
    checkpoint_dir = save_folder / "step125"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.pt").write_bytes(b"state")
    with pytest.raises(checkpoint.CheckpointContractError, match="missing kept checkpoint"):
        checkpoint.prune_older_permanent_checkpoints(save_folder, keep_step=250)


def test_finalize_prunes_older_checkpoints_after_durable_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_folder = tmp_path / "checkpoints"
    progress_dir = tmp_path / "progress"
    progress_dir.mkdir()
    for step in (0, 125):
        checkpoint_dir = save_folder / f"step{step}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "state.pt").write_bytes(b"state")
        (checkpoint_dir / "model_eval.pt").write_bytes(b"eval-weights")

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
            return Uploaded()

    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": FakeArtifact})())
    checkpoint.finalize_permanent_checkpoint(
        arm="probe",
        checkpoint_dir=save_folder / "step125",
        step=125,
        run_name="unit",
        task_loss_dir=tmp_path / "task-loss",
        task_loss_enabled=True,
        progress_dir=progress_dir,
        wandb_run=FakeRun(),
        wandb_mode="online",
        production=True,
        upload_checkpoint=False,
        run_evaluator=evaluator,
    )

    durable = checkpoint.read_last_durable_step(progress_dir)
    assert durable is not None
    assert durable["last_durable_step"] == 125
    assert [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)] == [
        "step125"
    ]
    assert not (save_folder / "step125" / "model_eval.pt").is_file()
    assert (save_folder / "step125" / "state.pt").is_file()


def test_directory_artifact_refuses_a_multi_gb_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size cap is the backstop for the other route to W&B.

    `wandb_log_directory_artifact` takes an arbitrary path, so pointing it at
    a checkpoint directory would bypass the model-artifact block entirely.
    """
    big = tmp_path / "checkpoints"
    big.mkdir()
    (big / "state.pt").write_bytes(b"\0" * 2048)
    monkeypatch.setattr(artifacts, "MAX_DIRECTORY_ARTIFACT_BYTES", 1024)
    monkeypatch.setattr(artifacts, "_wandb", type("Wandb", (), {"Artifact": object})())

    class FakeRun:
        name = "unit"

    with pytest.raises(artifacts.WandbArtifactError, match="exceeds the"):
        artifacts.wandb_log_directory_artifact(
            FakeRun(), big, name="unit-ckpt", artifact_type="model", strict=True
        )


def test_resume_never_depends_on_a_wandb_artifact(tmp_path: Path) -> None:
    """Resume must work from FarmShare scratch alone.

    Nothing uploads a model artifact any more, so the durable marker records
    no artifact reference and must not claim a W&B replica. Resume reads the
    local save_folder; if this ever regressed, a run would be unrecoverable
    after the first hard-stop/resume cycle.
    """
    progress = tmp_path / "progress"
    progress.mkdir()
    checkpoint.write_last_durable_step(progress, 250, checkpoint_artifact=None)

    marker = checkpoint.read_last_durable_step(progress)
    assert marker is not None
    assert marker["last_durable_step"] == 250
    assert marker["durability"] == "local_scratch"
    assert "checkpoint_artifact" not in marker


def test_prune_keeps_the_step_resume_needs(tmp_path: Path) -> None:
    """The unconditional post-marker prune must leave the durable step intact.

    Disabling uploads removed the pre-staging prune, so this is now the only
    prune on the path; if it dropped keep_step there would be nothing to
    resume from.
    """
    save_folder = tmp_path / "checkpoints"
    for step in (125, 250):
        directory = save_folder / f"step{step}"
        directory.mkdir(parents=True)
        (directory / "state.pt").write_bytes(b"state")

    checkpoint.prune_older_permanent_checkpoints(save_folder, keep_step=250)
    remaining = [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)]
    assert remaining == ["step250"]


def test_finalize_can_keep_the_whole_ladder(tmp_path: Path) -> None:
    """prune_older=False must leave every earlier checkpoint in place."""
    save_folder = tmp_path / "checkpoints"
    for step in (125, 250):
        directory = save_folder / f"step{step}"
        directory.mkdir(parents=True)
        (directory / "state.pt").write_bytes(b"state")

    checkpoint.finalize_permanent_checkpoint(
        arm="probe",
        checkpoint_dir=save_folder / "step250",
        step=250,
        run_name="unit",
        task_loss_dir=tmp_path / "task-loss",
        task_loss_enabled=False,
        progress_dir=None,
        wandb_run=None,
        wandb_mode="disabled",
        production=False,
        upload_checkpoint=False,
        prune_older=False,
    )
    remaining = [path.name for _, path in checkpoint.list_step_checkpoint_dirs(save_folder)]
    assert remaining == ["step125", "step250"]
