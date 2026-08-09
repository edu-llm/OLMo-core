"""Tests for ephemeral HPO artifact mirroring to W&B."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_core.hpo.wandb_probe import (
    HPO_PROBE_PROJECT,
    HpoProbeSession,
    collect_controller_metrics,
    durable_storage_roots,
    storage_path_is_durable,
    study_result_summary,
)


def test_storage_path_is_durable_for_urls_and_checkpoint_roots(tmp_path):
    durable_root = tmp_path / "checkpoints"
    durable_root.mkdir()
    durable_file = durable_root / "result.json"
    durable_file.write_text("{}")
    ephemeral = tmp_path / "tmp-result.json"
    ephemeral.write_text("{}")

    roots = (str(durable_root),)
    assert storage_path_is_durable("s3://bucket/key.json", durable_roots=roots)
    assert storage_path_is_durable(durable_file, durable_roots=roots)
    assert not storage_path_is_durable(ephemeral, durable_roots=roots)


def test_durable_storage_roots_include_checkpoint_dir_and_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "/mnt/checkpoints")
    roots = durable_storage_roots(checkpoint_dir=str(tmp_path / "override"))
    assert "/mnt/checkpoints" in roots
    assert str(tmp_path / "override") in roots


def test_collect_controller_metrics_reports_budget_and_best_ce():
    controller = SimpleNamespace(
        state=lambda: SimpleNamespace(tokens_charged=100, accelerator_seconds_charged=12.5, trials={"a": 1}),
        log=[1, 2, 3],
        top_candidates=lambda limit: [("trial-a", (0.1,), 2.5)],
    )
    metrics = collect_controller_metrics(controller, step=4)
    assert metrics["hpo/step"] == 4
    assert metrics["hpo/tokens_charged"] == 100
    assert metrics["hpo/best_search_validation_ce"] == 2.5


def test_study_result_summary_flattens_winner_hyperparameters():
    payload = {
        "arm": "no_proxy",
        "winner": {
            "trial_id": "trial-0",
            "hyperparameters": {"lr": 0.001},
            "search_validation_ce": 2.0,
        },
        "total_a100_hours": 1.5,
    }
    summary = study_result_summary(payload)
    assert summary["arm"] == "no_proxy"
    assert summary["winner/trial_id"] == "trial-0"
    assert summary["winner/hyperparameters/lr"] == 0.001
    assert summary["total_a100_hours"] == 1.5


def test_hpo_probe_session_mirrors_only_ephemeral_files(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    durable_root = tmp_path / "durable"
    durable_root.mkdir()
    ephemeral = tmp_path / "study-result.json"
    ephemeral.write_text(json.dumps({"arm": "no_proxy"}))
    durable = durable_root / "proxy-evidence.json"
    durable.write_text("{}")

    logged_artifacts = []

    class FakeArtifact:
        def __init__(self, name, type):
            self.name = name
            self.type = type
            self.files = []
            self.dirs = []

        def add_file(self, path):
            self.files.append(path)

        def add_dir(self, path):
            self.dirs.append(path)

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log_artifact(self, artifact):
            logged_artifacts.append(artifact)

    class FakeWandb:
        run = None

        def init(self, **kwargs):
            assert kwargs["project"] == HPO_PROBE_PROJECT
            FakeWandb.run = FakeRun()
            return FakeWandb.run

        def log(self, metrics, step):
            del metrics, step

        def finish(self, exit_code=0, quiet=True):
            del exit_code, quiet

        Artifact = FakeArtifact

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())

    session = HpoProbeSession.open(
        run_id="run-1",
        job_type="controller",
        durable_roots=(str(durable_root),),
        arm="no_proxy",
    )
    session.record_study_result({"arm": "no_proxy"}, ephemeral)
    session.record_proxy_cohort({"decision": "prune_promote", "metrics": {}}, output_path=durable)
    session.close()

    assert len(logged_artifacts) == 1
    assert logged_artifacts[0].name == "study-result"
    assert logged_artifacts[0].files == [str(ephemeral.resolve())]


def test_hpo_probe_session_requires_api_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        HpoProbeSession.open(
            run_id="run-1",
            job_type="controller",
            durable_roots=(),
        )
