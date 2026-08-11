"""What ``.edullm/hpo_on_corpus.py`` refuses, and how it lays out a segment launch.

Like ``train_on_corpus.py``, the HPO entry point sits in ``.edullm/`` because that is what the
platform image copies and runs, so it is loaded by path here. Its heavy imports are lazy, so this
module is importable without ``edullm_data`` and the topology guards are reachable directly.
"""

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from olmo_core.hpo.centaur import AdvisorResponse
from olmo_core.hpo.types import WorkerObservation


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "hpo_on_corpus.py"
    spec = importlib.util.spec_from_file_location("edullm_hpo_on_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()


def _probe_stub():
    return SimpleNamespace(
        log_heartbeat=lambda: None,
        log_controller=lambda controller: None,
        mirror_ephemeral_path=lambda *args, **kwargs: None,
        mirror_ephemeral_directory=lambda *args, **kwargs: None,
        close=lambda **kwargs: None,
    )


def test_controller_must_be_cpu_only_and_not_under_torchrun():
    entry.assert_controller_is_cpu_only({"WORLD_SIZE": "1", "RANK": "0"})  # ok
    entry.assert_controller_is_cpu_only({})  # ok (nothing inherited)
    with pytest.raises(RuntimeError):
        entry.assert_controller_is_cpu_only({"WORLD_SIZE": "8"})
    with pytest.raises(RuntimeError):
        entry.assert_controller_is_cpu_only({"WORLD_SIZE": "1", "TORCHELASTIC_RUN_ID": "abc"})


def test_plan_segment_builds_world_size_one_launch():
    plan = entry.plan_segment(
        gpu=2,
        master_port=29511,
        trial_id="t3_0",
        checkpoint_root="/run/ckpt",
        loaded_tokens=1024,
        target_tokens=8192,
        quantum=2048,
        base_env={"PATH": "/usr/bin"},
    )
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2"
    assert plan.env["WORLD_SIZE"] == "1"
    assert plan.env["MASTER_PORT"] == "29511"
    # OLMo's save_folder is the trial namespace; Trainer creates step{global_step} below it.
    assert plan.checkpoint_dir.replace("\\", "/").endswith("trials/t3_0")
    assert "/step1024" not in plan.checkpoint_dir.replace("\\", "/")
    # Hard stop is a fresh absolute token ceiling strictly above what was loaded.
    assert plan.hard_stop_tokens == 3072


def test_build_worker_argv_names_dtype_and_trial():
    argv = entry.build_worker_argv(
        run_id="run-abc",
        trial_id="t3_0",
        param_dtype="bfloat16",
        checkpoint_dir="/run/ckpt/trials/t3_0",
        hard_stop_tokens=3072,
    )
    joined = " ".join(argv)
    assert "t3_0" in joined
    assert "bfloat16" in joined  # dtype is explicit in the command
    assert any("hpo_on_corpus" in a or "train_on_corpus" in a for a in argv)
    parsed = entry._parse_args(argv[2:])
    assert parsed.run_segment is True
    assert parsed.run_id == "run-abc"
    assert parsed.trial_id == "t3_0"


def test_main_refuses_under_torchrun(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError):
        entry.main(["--dry-run"])


def test_segment_refuses_non_single_process_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError):
        entry.main(["run-abc", "--run-segment", "--dry-run"])


def test_committed_run_command_parses():
    run_yaml = Path(__file__).parent.parent.parent / ".edullm" / "run.yaml"
    command = yaml.safe_load(run_yaml.read_text())["command"]
    argv = shlex.split(command.replace('"$EDULLM_RUN_ID"', "run-abc"))
    assert argv[:2] == ["bash", "-lc"]
    assert "python .edullm/hpo_on_corpus.py run-abc" in argv[2]
    assert "--controller-spec" in argv[2]
    assert "--param-dtype bfloat16" in argv[2]


def test_non_dry_run_modes_reach_their_runners(monkeypatch):
    calls = []
    monkeypatch.setattr(
        entry, "run_controller", lambda args: calls.append(("controller", args)) or 0
    )
    monkeypatch.setattr(entry, "run_segment", lambda args: calls.append(("segment", args)) or 0)

    assert entry.main(["run-abc", "--param-dtype", "bfloat16"]) == 0
    assert (
        entry.main(
            [
                "run-abc",
                "--run-segment",
                "--trial-id",
                "t0",
                "--checkpoint-dir",
                "/ckpt/t0",
                "--hard-stop-tokens",
                "1024",
            ]
        )
        == 0
    )
    assert [kind for kind, _ in calls] == ["controller", "segment"]


def test_run_segment_loads_factory_executes_and_emits_observation(monkeypatch, tmp_path, capsys):
    segment_spec = {
        "experiment_factory": "tests.fake:build",
        "factory_kwargs": {"value": 1},
        "target_tokens": 8192,
        "global_batch_size": 1024,
        "realized_hps": {
            "lr": 1e-3,
            "weight_decay": 0.1,
            "beta2_gap": 0.01,
            "eps": 1e-8,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        "checkpoint_root": "/run/ckpt",
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "checkpoint_ref": "/run/ckpt/donor/step2",
        "transition": None,
    }
    path = tmp_path / "segment.json"
    path.write_text(json.dumps(segment_spec))
    config = SimpleNamespace(trainer=SimpleNamespace())
    monkeypatch.setattr(entry, "_load_object", lambda ref: lambda **kwargs: config)
    monkeypatch.setattr(
        entry,
        "_run_configured_segment",
        lambda **kwargs: WorkerObservation(
            trial_id="t0",
            tokens=3072,
            heldout_ce=3.5,
            train_ce_history=(3.8, 3.6),
            grad_norm_history=(1.0, 1.1),
            activation_ratio=0.4,
            numeric_failure=False,
            checkpoint_ref="/run/ckpt/trials/t0/step3",
        ),
    )
    args = entry._parse_args(
        [
            "run-abc",
            "--run-segment",
            "--trial-id",
            "t0",
            "--checkpoint-dir",
            str(Path("/run/ckpt") / "trials" / "t0"),
            "--hard-stop-tokens",
            "3072",
            "--segment-spec",
            str(path),
        ]
    )
    assert entry.run_segment(args) == 0
    assert config.trainer.load_path == "/run/ckpt/donor/step2"
    assert config.trainer.load_optim_state is True
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["trial_id"] == "t0"
    assert emitted["checkpoint_ref"].endswith("/step3")


def test_run_controller_restores_dispatches_ingests_and_persists(monkeypatch, tmp_path):
    spec_path = tmp_path / "controller.json"
    spec_path.write_text(json.dumps({"max_rounds": 3}))
    state_path = tmp_path / "controller.jsonl"
    state_path.write_text("prior-state")
    calls = []

    class FakeLog:
        def to_jsonl(self):
            return "new-state"

    class FakeController:
        def __init__(self):
            self.log = FakeLog()
            self.round = 0

        def restore_log(self, text):
            calls.append(("restore", text))

        def propose_round(self):
            self.round += 1
            return [SimpleNamespace(trial_id="t0")] if self.round == 1 else []

        def ingest(self, results):
            calls.append(("ingest", results))

    controller = FakeController()
    monkeypatch.setattr(entry, "_build_controller_from_spec", lambda spec: controller)
    monkeypatch.setattr(entry, "_open_hpo_probe_session", lambda **kwargs: _probe_stub())
    monkeypatch.setattr(
        entry,
        "_dispatch_allocations",
        lambda **kwargs: [
            WorkerObservation(
                trial_id="t0",
                tokens=1024,
                heldout_ce=3.5,
                train_ce_history=(3.8, 3.5),
                grad_norm_history=(1.0,),
                activation_ratio=0.4,
                numeric_failure=False,
                checkpoint_ref="/ckpt/t0/step1",
            )
        ],
    )
    monkeypatch.setattr(
        entry,
        "_persist_controller_log",
        lambda controller, path, remote_root=None: calls.append(
            ("persist", path, controller.log.to_jsonl())
        ),
    )
    args = entry._parse_args(
        [
            "run-abc",
            "--controller-spec",
            str(spec_path),
            "--controller-state",
            str(state_path),
        ]
    )
    assert entry.run_controller(args) == 0
    assert ("restore", "prior-state") in calls
    assert any(call[0] == "ingest" for call in calls)
    assert calls[-1] == ("persist", str(state_path), "new-state")


def test_controller_persists_event_log_when_proposal_raises(monkeypatch, tmp_path):
    spec_path = tmp_path / "controller.json"
    spec_path.write_text(json.dumps({"max_rounds": 1}))
    persisted = []

    class FakeLog:
        def to_jsonl(self):
            return "failure-event"

    class FakeController:
        log = FakeLog()

        def pending_allocations(self):
            return []

        def propose_round(self):
            raise RuntimeError("advisor failed")

    monkeypatch.setattr(entry, "_build_controller_from_spec", lambda spec: FakeController())
    monkeypatch.setattr(entry, "_open_hpo_probe_session", lambda **kwargs: _probe_stub())
    monkeypatch.setattr(
        entry,
        "_persist_controller_log",
        lambda controller, path, remote_root=None: persisted.append(controller.log.to_jsonl()),
    )
    args = entry._parse_args(["run", "--controller-spec", str(spec_path)])
    with pytest.raises(RuntimeError, match="advisor failed"):
        entry.run_controller(args)
    assert persisted == ["failure-event"]


def test_restored_pending_allocations_are_redispatched_first(monkeypatch, tmp_path):
    spec_path = tmp_path / "controller.json"
    spec_path.write_text(json.dumps({"max_rounds": 1}))
    state_path = tmp_path / "controller.jsonl"
    state_path.write_text("prior-state")
    pending = SimpleNamespace(trial_id="pending")
    calls = []

    class FakeLog:
        def to_jsonl(self):
            return "state"

    class FakeController:
        log = FakeLog()

        def restore_log(self, text):
            pass

        def pending_allocations(self):
            return [pending]

        def propose_round(self):
            calls.append("propose")
            return []

        def ingest(self, results):
            calls.append(("ingest", results[0].trial_id))

    monkeypatch.setattr(entry, "_build_controller_from_spec", lambda spec: FakeController())
    monkeypatch.setattr(entry, "_open_hpo_probe_session", lambda **kwargs: _probe_stub())

    def dispatch(**kwargs):
        calls.append(("dispatch", kwargs["allocations"][0].trial_id))
        return [
            WorkerObservation(
                trial_id="pending",
                tokens=1024,
                heldout_ce=3.5,
                train_ce_history=(3.5,),
                grad_norm_history=(1.0,),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/pending/step1",
            )
        ]

    monkeypatch.setattr(entry, "_dispatch_allocations", dispatch)
    monkeypatch.setattr(entry, "_persist_controller_log", lambda *args, **kwargs: None)
    args = entry._parse_args(
        [
            "run-abc",
            "--controller-spec",
            str(spec_path),
            "--controller-state",
            str(state_path),
        ]
    )
    entry.run_controller(args)
    assert calls[:2] == [("dispatch", "pending"), ("ingest", "pending")]
    assert calls[2] == "propose"


def test_segment_payload_contains_worker_config_hash():
    allocation = SimpleNamespace(
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.1,
            "beta2_gap": 0.01,
            "eps": 1e-8,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
            "global_batch_mult": 1.0,
        },
        checkpoint_ref=None,
        transition=None,
    )
    spec = {
        "base_global_batch_size": 1024,
        "experiment_factory": "module:factory",
        "controller": {"target_tokens": 8192, "checkpoint_root": "/ckpt"},
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
    }
    payload = entry._segment_payload(allocation, controller_spec=spec)
    assert len(payload["config_hash"]) == 64
    frozen = entry._segment_payload(
        allocation,
        controller_spec={
            **spec,
            "fidelity": {"kind": "frozen_layer", "train_last_k": 4},
        },
    )
    assert frozen["config_hash"] != payload["config_hash"]


def test_curriculum_identity_is_in_segment_artifact_and_config_hash():
    allocation = SimpleNamespace(
        realized_hps={"global_batch_mult": 1.0},
        checkpoint_ref="/ckpt/donor",
        transition={"donor_trial_id": "parent"},
    )
    identity = {
        "parent": {"manifest_sha256": "a" * 64},
        "order": {"manifest_sha256": "b" * 64},
        "pacing": "arm9_warmup_quadratic_n10_token_fraction_v1",
    }
    spec = {
        "base_global_batch_size": 524_288,
        "experiment_factory": "olmo_core.hpo.curriculum:build_curriculum_hpo_experiment",
        "controller": {
            "target_tokens": 503_316_480,
            "quantum": 50_331_648,
            "checkpoint_root": "/ckpt",
        },
        "factory_kwargs": {"rank_microbatch_size": 4096},
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/lm/opt-with-synthetic-10b-val/CE loss",
        "curriculum_identity": identity,
    }

    payload = entry._segment_payload(allocation, controller_spec=spec)

    assert payload["curriculum_identity"] == identity
    changed = {
        **payload,
        "curriculum_identity": {
            **identity,
            "order": {"manifest_sha256": "c" * 64},
        },
    }
    assert entry._segment_config_hash(changed) != payload["config_hash"]


def test_segment_refuses_factory_with_different_curriculum_identity(monkeypatch, tmp_path):
    expected = {
        "parent": {"manifest_sha256": "a" * 64},
        "order": {"manifest_sha256": "b" * 64},
        "pacing": "arm9_warmup_quadratic_n10_token_fraction_v1",
    }
    segment_spec = {
        "experiment_factory": "module:factory",
        "factory_kwargs": {},
        "curriculum_identity": expected,
    }
    path = tmp_path / "segment.json"
    path.write_text(json.dumps(segment_spec))
    monkeypatch.setattr(
        entry,
        "_load_object",
        lambda ref: lambda **kwargs: SimpleNamespace(
            curriculum_identity={
                **expected,
                "order": {"manifest_sha256": "c" * 64},
            }
        ),
    )
    args = entry._parse_args(
        [
            "run",
            "--run-segment",
            "--trial-id",
            "trial",
            "--checkpoint-dir",
            "/ckpt/trial",
            "--hard-stop-tokens",
            "10",
            "--segment-spec",
            str(path),
        ]
    )

    with pytest.raises(ValueError, match="curriculum identity"):
        entry.run_segment(args)


def test_hpo_probe_receives_curriculum_identity_and_heldout_metric(monkeypatch):
    import olmo_core.hpo.wandb_probe as probe_module

    captured = {}
    monkeypatch.setattr(
        probe_module.HpoProbeSession,
        "open",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    identity = {
        "parent": {"manifest_sha256": "a" * 64},
        "order": {"manifest_sha256": "b" * 64},
        "pacing": "arm9_warmup_quadratic_n10_token_fraction_v1",
        "token_phase_boundaries": [0, 10, 20],
        "source_ids": ["a", "b"],
    }
    spec = {
        "arm": "curriculum_quadratic_mtld",
        "algorithm": "brainlift",
        "controller": {"checkpoint_root": "/ckpt"},
        "model_parameterization": {"kind": "standard"},
        "fidelity": {"kind": "exact"},
        "curriculum_identity": identity,
        "heldout_metric": "eval/lm/opt-with-synthetic-10b-val/CE loss",
    }

    entry._open_hpo_probe_session(
        run_id="run",
        job_type="controller",
        spec=spec,
        checkpoint_root="/ckpt",
    )

    assert captured["config"]["curriculum_identity"] == identity
    assert captured["config"]["heldout_metric"] == spec["heldout_metric"]


def test_segment_payload_merges_fixed_and_searched_hps():
    allocation = SimpleNamespace(
        realized_hps={"lr": 2e-3, "max_grad_norm": 0.8},
        checkpoint_ref=None,
        transition=None,
    )
    payload = entry._segment_payload(
        allocation,
        controller_spec={
            "base_global_batch_size": 32768,
            "fixed_hps": {
                "lr": 1e-3,
                "weight_decay": 0.1,
                "beta2_gap": 0.05,
                "eps": 1e-8,
                "warmup_fraction": 0.05,
                "decay_fraction": 0.2,
                "terminal_lr_ratio": 0.1,
                "global_batch_mult": 1.0,
                "max_grad_norm": 1.0,
            },
            "experiment_factory": "module:factory",
            "controller": {
                "target_tokens": 327680,
                "checkpoint_root": "/ckpt",
            },
            "search_validation_callback": "search_validation",
            "untouched_evaluator": "final_evaluation",
            "heldout_metric": "eval/lm/CE loss",
        },
    )
    assert payload["realized_hps"]["lr"] == 2e-3
    assert payload["realized_hps"]["max_grad_norm"] == 0.8
    assert payload["realized_hps"]["weight_decay"] == 0.1
    assert set(payload["realized_hps"]) == {
        "lr",
        "weight_decay",
        "beta2_gap",
        "eps",
        "warmup_fraction",
        "decay_fraction",
        "terminal_lr_ratio",
        "global_batch_mult",
        "max_grad_norm",
    }


def test_controller_spec_expands_environment_without_placeholders(
    monkeypatch,
):
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", "s3://checkpoints/run-1")
    expanded = entry._expand_environment(
        {
            "root": "${EDULLM_CHECKPOINT_DIR}/hybrid",
            "nested": ["${EDULLM_CHECKPOINT_DIR}", 3],
        }
    )
    assert expanded == {
        "root": "s3://checkpoints/run-1/hybrid",
        "nested": ["s3://checkpoints/run-1", 3],
    }
    with pytest.raises(ValueError, match="unresolved"):
        entry._expand_environment("${MISSING_COMPARISON_ENV}")


def test_controller_log_load_falls_back_to_latest_remote_snapshot(monkeypatch, tmp_path):
    local = tmp_path / "missing.jsonl"
    monkeypatch.setattr(
        entry,
        "_remote_controller_snapshots",
        lambda root: [
            (2, "s3://bucket/controller/decisions-00000002-a.jsonl"),
            (10, "s3://bucket/controller/decisions-00000010-b.jsonl"),
        ],
    )
    monkeypatch.setattr(
        entry,
        "_read_controller_text",
        lambda path: {"s3://bucket/controller/decisions-00000010-b.jsonl": "latest"}[path],
    )
    assert entry._load_controller_log(str(local), "s3://bucket") == "latest"


def test_remote_controller_snapshot_skips_existing_immutable_object(monkeypatch, tmp_path):
    import olmo_core.io as olmo_io

    class Log:
        def __len__(self):
            return 3

        def to_jsonl(self):
            return "same-state"

    controller = SimpleNamespace(log=Log())
    monkeypatch.setattr(olmo_io, "file_exists", lambda path: True)
    monkeypatch.setattr(
        olmo_io,
        "upload",
        lambda *args, **kwargs: pytest.fail("duplicate upload attempted"),
    )
    entry._persist_controller_log(
        controller,
        str(tmp_path / "controller.jsonl"),
        "s3://bucket/run",
    )


def test_worker_oom_returns_typed_fatal_observation(monkeypatch, tmp_path):
    allocation = SimpleNamespace(
        trial_id="t0",
        decision_id=0,
        target_fidelity=1024,
        realized_hps={
            "lr": 1e-3,
            "weight_decay": 0.1,
            "beta2_gap": 0.01,
            "eps": 1e-8,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        checkpoint_ref=None,
        transition=None,
    )
    spec = {
        "base_global_batch_size": 1024,
        "experiment_factory": "module:factory",
        "controller": {
            "target_tokens": 8192,
            "checkpoint_root": "/ckpt",
            "worker_count": 1,
        },
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "gpu_ids": [0],
        "segment_spec_dir": str(tmp_path),
    }
    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="torch.OutOfMemoryError: CUDA out of memory",
            stdout="",
        ),
    )
    result = entry._dispatch_allocations(
        allocations=[allocation],
        controller_spec=spec,
        run_id="run",
        param_dtype="bfloat16",
    )[0]
    assert result.trial_id == "t0"
    assert result.numeric_failure is True
    assert result.tokens == 1024
    assert result.heldout_ce != result.heldout_ce


def test_search_segment_worker_world_size_uses_torchrun_and_charges_all_gpus(monkeypatch, tmp_path):
    allocation = SimpleNamespace(
        trial_id="search-trial",
        decision_id=42,
        kind="start",
        target_fidelity=49_807_360,
        realized_hps={"lr": 1e-3},
        checkpoint_ref=None,
        transition=None,
    )
    spec = {
        "base_global_batch_size": 262_144,
        "experiment_factory": "module:factory",
        "factory_kwargs": {
            "sequence_length": 2_048,
            "rank_microbatch_size": 32_768,
        },
        "fixed_hps": {
            "weight_decay": 0.1,
            "beta2_gap": 0.01,
            "eps": 1e-8,
            "warmup_fraction": 0.02,
            "decay_fraction": 0.2,
            "terminal_lr_ratio": 0.1,
            "max_grad_norm": 1.0,
        },
        "controller": {
            "target_tokens": 499_808_608,
            "quantum": 49_807_360,
            "checkpoint_root": "/ckpt",
            "worker_count": 1,
        },
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "gpu_ids": list(range(8)),
        "segment_spec_dir": str(tmp_path),
        "worker_world_size": 8,
    }
    captured = {}

    def run(argv, *, env, **kwargs):
        captured.update(argv=argv, env=env, kwargs=kwargs)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "trial_id": "search-trial",
                    "tokens": 49_807_360,
                    "heldout_ce": 3.25,
                    "train_ce_history": [3.4, 3.3],
                    "grad_norm_history": [0.9],
                    "activation_ratio": None,
                    "numeric_failure": False,
                    "checkpoint_ref": "/ckpt/trials/search-trial/step190",
                }
            ),
        )

    ticks = iter((100.0, 103.0))
    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.time, "perf_counter", lambda: next(ticks))

    result = entry._dispatch_allocations(
        allocations=[allocation],
        controller_spec=spec,
        run_id="same-run",
        param_dtype="bfloat16",
    )[0]

    assert captured["argv"][1:3] == ["-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in captured["argv"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert captured["env"]["EDULLM_WORKER_WORLD_SIZE"] == "8"
    assert "EDULLM_FINALIST_CONTINUATION" not in captured["env"]
    assert "WORLD_SIZE" not in captured["env"]  # torchrun owns rank metadata
    payload = json.loads((tmp_path / "decision-42.json").read_text())
    assert payload["factory_kwargs"]["rank_microbatch_size"] == 32_768
    assert "finalist_continuation" not in payload
    assert result.checkpoint_ref.endswith("step190")
    assert result.accelerator_seconds == pytest.approx(24.0)


def test_olmo2_190m_search_defaults_to_original_single_gpu_launch(monkeypatch, tmp_path):
    allocation = SimpleNamespace(
        trial_id="dense-probe",
        decision_id=43,
        target_fidelity=32_768,
        realized_hps={"global_batch_mult": 1.0},
        checkpoint_ref=None,
        transition=None,
    )
    spec = {
        "base_global_batch_size": 32_768,
        "experiment_factory": "olmo_core.hpo.comparison:build_comparison_experiment",
        "factory_kwargs": {
            "sequence_length": 2_048,
            "global_batch_size": 32_768,
            "rank_microbatch_size": 4_096,
        },
        "model_parameterization": {
            "kind": "standard",
            "architecture": "olmo2_190M",
            "depth": 12,
            "backend": "none",
        },
        "controller": {
            "target_tokens": 327_680,
            "quantum": 32_768,
            "checkpoint_root": "/ckpt",
            "worker_count": 8,
        },
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "gpu_ids": list(range(8)),
        "segment_spec_dir": str(tmp_path),
    }
    captured = {}

    def run(argv, *, env, **kwargs):
        captured.update(argv=argv, env=env)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "trial_id": "dense-probe",
                    "tokens": 32_768,
                    "heldout_ce": 4.0,
                    "train_ce_history": [4.1],
                    "grad_norm_history": [1.0],
                    "activation_ratio": None,
                    "numeric_failure": False,
                    "checkpoint_ref": "/ckpt/trials/dense-probe/step1",
                }
            ),
        )

    monkeypatch.setattr(entry.subprocess, "run", run)
    result = entry._dispatch_allocations(
        allocations=[allocation],
        controller_spec=spec,
        run_id="dense-run",
        param_dtype="bfloat16",
    )[0]

    assert captured["argv"][1] != "-m"
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert captured["env"]["WORLD_SIZE"] == "1"
    assert "EDULLM_WORKER_WORLD_SIZE" not in captured["env"]
    payload = json.loads((tmp_path / "decision-43.json").read_text())
    assert payload["experiment_factory"] == spec["experiment_factory"]
    assert payload["model_parameterization"]["architecture"] == "olmo2_190M"
    assert payload["factory_kwargs"] == spec["factory_kwargs"]
    assert result.trial_id == "dense-probe"


def test_single_resume_opt_in_uses_eight_rank_finalist_and_charges_all_gpus(monkeypatch, tmp_path):
    allocation = SimpleNamespace(
        trial_id="winner",
        decision_id=41,
        kind="resume",
        target_fidelity=250_019_840,
        realized_hps={"global_batch_mult": 0.5},
        checkpoint_ref="/ckpt/trials/winner/step12208",
        transition={"full_fidelity_rescue": True},
    )
    spec = {
        "base_global_batch_size": 32_768,
        "experiment_factory": "module:factory",
        "factory_kwargs": {
            "sequence_length": 2_048,
            "rank_microbatch_size": 4_096,
        },
        "controller": {
            "target_tokens": 500_039_680,
            "quantum": 50_003_968,
            "checkpoint_root": "/ckpt",
            "worker_count": 8,
        },
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "gpu_ids": list(range(8)),
        "segment_spec_dir": str(tmp_path),
        "finalist_continuation": {
            "enabled": True,
            "world_size": 8,
            "rank_microbatch_size": 2_048,
        },
    }
    captured = {}

    def run(argv, *, env, **kwargs):
        captured.update(argv=argv, env=env, kwargs=kwargs)
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps(
                {
                    "trial_id": "winner",
                    "tokens": 250_019_840,
                    "heldout_ce": 3.25,
                    "train_ce_history": [3.4, 3.3],
                    "grad_norm_history": [0.9],
                    "activation_ratio": None,
                    "numeric_failure": False,
                    "checkpoint_ref": "/ckpt/trials/winner/step15260",
                }
            ),
        )

    ticks = iter((100.0, 103.0))
    monkeypatch.setattr(entry.subprocess, "run", run)
    monkeypatch.setattr(entry.time, "perf_counter", lambda: next(ticks))
    result = entry._dispatch_allocations(
        allocations=[allocation],
        controller_spec=spec,
        run_id="same-run",
        param_dtype="bfloat16",
    )[0]

    assert captured["argv"][1:3] == ["-m", "torch.distributed.run"]
    assert "--nproc-per-node=8" in captured["argv"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert captured["env"]["EDULLM_FINALIST_CONTINUATION"] == "1"
    assert "WORLD_SIZE" not in captured["env"]  # torchrun owns rank metadata
    payload = json.loads((tmp_path / "decision-41.json").read_text())
    assert payload["global_batch_size"] == 16_384
    assert payload["factory_kwargs"]["rank_microbatch_size"] == 2_048
    assert payload["finalist_continuation"]["world_size"] == 8
    assert result.checkpoint_ref.endswith("step15260")
    assert result.accelerator_seconds == pytest.approx(24.0)


def test_finalist_opt_in_never_collapses_a_multi_allocation_batch():
    allocations = [
        SimpleNamespace(
            kind="resume",
            checkpoint_ref=f"/ckpt/t{index}",
            transition=None,
        )
        for index in range(2)
    ]
    assert (
        entry._eligible_finalist_continuation(
            allocations,
            controller_spec={
                "finalist_continuation": {
                    "enabled": True,
                    "world_size": 8,
                    "rank_microbatch_size": 2_048,
                }
            },
            gpu_ids=list(range(8)),
        )
        is None
    )


def test_finalist_opt_in_preserves_an_indivisible_lineage_on_one_gpu():
    allocation = SimpleNamespace(
        kind="resume",
        checkpoint_ref="/ckpt/t0",
        transition=None,
        realized_hps={"global_batch_mult": 0.875},
    )
    assert (
        entry._eligible_finalist_continuation(
            [allocation],
            controller_spec={
                "base_global_batch_size": 32_768,
                "factory_kwargs": {"rank_microbatch_size": 4_096},
                "controller": {"quantum": 50_003_968, "target_tokens": 500_039_680},
                "finalist_continuation": {
                    "enabled": True,
                    "world_size": 8,
                    "rank_microbatch_size": 2_048,
                },
            },
            gpu_ids=list(range(8)),
        )
        is None
    )


def test_controller_runs_untouched_evaluator_once(monkeypatch):
    calls = []

    class Controller:
        completed = False

        def final_evaluation_completed(self):
            return self.completed

        def best(self):
            return ("winner", (0.1, 0.2), 2.5)

        def state(self):
            return SimpleNamespace(
                trials={"winner": SimpleNamespace(latest_checkpoint_ref="/ckpt/winner/step10")}
            )

        def record_final_evaluation(self, payload):
            calls.append(payload)
            self.completed = True

    def evaluator(**kwargs):
        return {"untouched_ce": 2.6, **kwargs}

    monkeypatch.setattr(entry, "_load_object", lambda ref: evaluator)
    controller = Controller()
    spec = {"final_evaluator_factory": "module:evaluate"}
    entry._run_untouched_evaluation(controller, spec)
    entry._run_untouched_evaluation(controller, spec)
    assert len(calls) == 1
    assert calls[0]["trial_id"] == "winner"
    assert calls[0]["checkpoint_ref"] == "/ckpt/winner/step10"


def test_required_final_winner_fails_closed():
    controller = SimpleNamespace(final_evaluation_completed=lambda: False)
    with pytest.raises(RuntimeError, match="winner"):
        entry._enforce_required_winner(controller, {"require_final_winner": True})
    entry._enforce_required_winner(controller, {"require_final_winner": False})


def test_proxy_winner_is_retrained_from_scratch_on_exact_model(
    monkeypatch,
):
    controller = SimpleNamespace(
        best=lambda: ("winner", (0.1, 0.2), 2.5),
        search_space=SimpleNamespace(from_unit=lambda unit: {"lr": 1e-3}),
        config=SimpleNamespace(target_tokens=100),
    )
    captured = {}

    def dispatch(**kwargs):
        captured.update(kwargs)
        allocation = kwargs["allocations"][0]
        assert allocation.checkpoint_ref is None
        assert allocation.current_fidelity == 0
        assert kwargs["controller_spec"]["fidelity"]["kind"] == "exact"
        return [
            WorkerObservation(
                trial_id=allocation.trial_id,
                tokens=100,
                heldout_ce=2.4,
                train_ce_history=(2.8, 2.4),
                grad_norm_history=(1.0,),
                activation_ratio=0.5,
                numeric_failure=False,
                checkpoint_ref="/ckpt/winner-exact/step10",
            )
        ]

    monkeypatch.setattr(entry, "_dispatch_allocations", dispatch)
    result = entry._run_exact_retrain(
        controller,
        {
            "fidelity": {"kind": "frozen_layer", "train_last_k": 4},
            "exact_experiment_factory": "module:exact",
            "exact_factory_kwargs": {},
            "experiment_factory": "module:proxy",
            "controller": {
                "target_tokens": 100,
                "checkpoint_root": "/ckpt",
                "worker_count": 1,
            },
            "base_global_batch_size": 1024,
            "search_validation_callback": "search_validation",
            "untouched_evaluator": "final",
            "heldout_metric": "eval/search_validation/val/CE loss",
        },
        run_id="run",
        param_dtype="bfloat16",
    )
    assert result.checkpoint_ref == "/ckpt/winner-exact/step10"


def test_frozen_evidence_is_reporting_only_until_common_cohort_passes():
    with pytest.raises(ValueError, match="cohort"):
        entry._validate_evidence_gates(
            {
                "fidelity": {"kind": "frozen_layer", "train_last_k": 8},
                "controller": {"quantum": 50},
            }
        )
    pending = {
        "fidelity": {"kind": "frozen_layer", "train_last_k": 8},
        "controller": {"quantum": 50},
        "proxy_evidence_contract": {
            "cohort_id": "v1",
            "config_ids": ["a", "b", "c"],
            "first_rung_tokens": 50,
            "proxy_arm": "full_acronym_soup",
            "reference_arm": "no_proxy",
            "top_k": 2,
            "search_dimensions": 9,
            "seed": 20260808,
        },
        "proxy_admission": {
            "min_rank_corr": 0.7,
            "min_top_k_recall": 0.6,
            "min_samples": 3,
        },
        "frozen_ranking_policy": "reporting_only_until_admitted",
    }
    assert entry._validate_evidence_gates(pending).value == "reporting_only"

    contract = pending["proxy_evidence_contract"]
    proxy_observations = {
        key: {"tokens": 50, "ce": value, "accelerator_seconds": 8.0}
        for key, value in {"a": 1.0, "b": 2.0, "c": 3.0}.items()
    }
    reference_observations = {
        key: {"tokens": 50, "ce": value, "accelerator_seconds": 10.0}
        for key, value in {"a": 1.1, "b": 2.1, "c": 3.1}.items()
    }
    admitted = {
        **pending,
        "proxy_evidence": {
            "schema_version": 1,
            "contract": contract,
            "admission": pending["proxy_admission"],
            "proxy_observations": proxy_observations,
            "reference_observations": reference_observations,
            "decision": "prune_promote",
        },
    }
    assert entry._validate_evidence_gates(admitted).value == "prune_promote"
    tampered = {
        **admitted,
        "proxy_evidence": {
            **admitted["proxy_evidence"],
            "contract": {**contract, "seed": 9},
        },
    }
    with pytest.raises(ValueError, match="contract"):
        entry._validate_evidence_gates(tampered)


@pytest.mark.parametrize(
    ("proxy_seconds", "expected_decision"),
    [(8.0, "prune_promote"), (12.0, "reporting_only")],
)
def test_proxy_cohort_executes_paired_first_rung_and_persists_admission(
    monkeypatch, tmp_path, capsys, proxy_seconds, expected_decision
):
    root = Path(__file__).resolve().parents[2] / ".edullm"
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("EDULLM_RUN_ID", "cohort-test")
    import olmo_core.hpo.umup as umup

    monkeypatch.setattr(umup, "require_official_umup_forward", lambda: None)
    calls = []

    def fake_dispatch(*, allocations, controller_spec, run_id, param_dtype):
        del controller_spec, run_id, param_dtype
        calls.append(tuple(allocation.trial_id for allocation in allocations))
        results = []
        for allocation in allocations:
            reference = "-reference-" in allocation.trial_id
            results.append(
                WorkerObservation(
                    trial_id=allocation.trial_id,
                    tokens=allocation.target_fidelity,
                    heldout_ce=float(allocation.decision_id + 1),
                    train_ce_history=(),
                    grad_norm_history=(),
                    activation_ratio=None,
                    numeric_failure=False,
                    checkpoint_ref=f"/ckpt/{allocation.trial_id}",
                    accelerator_seconds=10.0 if reference else proxy_seconds,
                )
            )
        return results

    monkeypatch.setattr(entry, "_dispatch_allocations", fake_dispatch)
    output = tmp_path / "proxy-evidence.json"
    args = entry._parse_args(
        [
            "cohort-test",
            "--run-proxy-cohort",
            "--proxy-spec",
            str(root / "hpo-full-acronym-soup.json"),
            "--reference-spec",
            str(root / "hpo-no-proxy.json"),
            "--cohort-output",
            str(output),
        ]
    )
    assert entry.run_proxy_cohort(args) == 0
    artifact = json.loads(output.read_text())
    assert artifact["decision"] == expected_decision
    assert set(artifact["proxy_observations"]) == set(artifact["reference_observations"])
    assert len(artifact["proxy_observations"]) == 16
    assert artifact["metrics"]["proxy_kind"] == "umup_frozen_layer"
    assert artifact["metrics"]["net_compute_savings"] == pytest.approx(1.0 - proxy_seconds / 10.0)
    assert len(calls) == 4  # two 8-worker batches for each side
    assert json.loads(capsys.readouterr().out)["output_path"] == str(output)

    monkeypatch.setattr(
        entry,
        "_dispatch_allocations",
        lambda **kwargs: pytest.fail(f"existing evidence was recomputed: {kwargs}"),
    )
    assert entry.run_proxy_cohort(args) == 0
    reused = json.loads(capsys.readouterr().out)
    assert reused["decision"] == expected_decision
    assert reused["reused_existing_evidence"] is True


def test_proxy_cohort_fails_before_dispatch_when_official_umup_is_unavailable(
    monkeypatch, tmp_path
):
    root = Path(__file__).resolve().parents[2] / ".edullm"
    monkeypatch.setenv("EDULLM_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("EDULLM_RUN_ID", "cohort-blocked")
    import olmo_core.hpo.umup as umup

    def unavailable():
        raise RuntimeError("unit-scaling lacks required public operations")

    monkeypatch.setattr(umup, "require_official_umup_forward", unavailable)
    monkeypatch.setattr(
        entry,
        "_dispatch_allocations",
        lambda **kwargs: pytest.fail(f"unexpected dispatch: {kwargs}"),
    )
    args = entry._parse_args(
        [
            "cohort-blocked",
            "--run-proxy-cohort",
            "--proxy-spec",
            str(root / "hpo-full-acronym-soup.json"),
            "--reference-spec",
            str(root / "hpo-no-proxy.json"),
        ]
    )
    with pytest.raises(RuntimeError, match="lacks required public operations"):
        entry.run_proxy_cohort(args)


def test_brainlift_builder_uses_ifbo_and_rejects_cma(monkeypatch):
    class Advisor:
        def advise(self, state):
            return AdvisorResponse(
                action=state["default_action"],
                raw_text="{}",
                model="gpt-5.6-sol",
                version="v1",
                latency_ms=1.0,
            )

    monkeypatch.setattr(entry, "_load_object", lambda ref: lambda **kwargs: Advisor())
    base = {
        "algorithm": "brainlift_test",
        "seed": 7,
        "search_space": [
            {"name": "x", "low": 0.0, "high": 1.0},
            {"name": "y", "low": 0.0, "high": 1.0},
        ],
        "normalizer": {"ce_at_zero": 6.0, "ce_at_one": 2.0},
        "posterior": {"kind": "synthetic", "optimum": [0.5, 0.5]},
        "proposer": {"kind": "ifbo"},
        "btt": {},
        "ipbt": {
            "population_size": 2,
            "top_quantile": 0.5,
            "bottom_quantile": 0.5,
            "update_interval_init": 2,
        },
        "centaur": {
            "scope": "multi_action",
            "model": "gpt-5.6-sol",
            "ratio": 0.3,
            "warmup": 0,
            "advisor_factory": "module:factory",
        },
        "controller": {
            "target_tokens": 8,
            "quantum": 2,
            "n_fidelity_bins": 4,
            "worker_count": 2,
            "budget_tokens": 32,
            "checkpoint_root": "/tmp/hpo",
        },
    }
    controller = entry._build_controller_from_spec(base)
    assert controller.proposer.proposal_source.value == "ifbo"
    assert controller.ipbt_meta_proposer.proposal_source.value == "ifbo"
    assert controller.config.restart_mode.value == "btt_aggregate"
    assert controller._ask_ledger is None
    assert {allocation.source.value for allocation in controller.propose_round()} == {"ifbo"}
    deterministic = entry._build_controller_from_spec(
        {
            **base,
            "arm": "no_centaur",
            "centaur": None,
            "model_parameterization": {
                "kind": "umup",
                "source_architecture": "olmo2_370M",
                "depth": 16,
            },
            "fidelity": {"kind": "frozen_layer", "train_last_k": 8},
        }
    )
    assert deterministic.action_centaur is None
    assert deterministic.action_advisor is None
    olmoe_parameterization = {
        "kind": "standard",
        "architecture": "olmoe_1B_7B",
        "depth": 16,
        "backend": "none",
    }
    olmoe_no_proxy = entry._build_controller_from_spec(
        {
            **base,
            "arm": "olmoe_no_proxy",
            "model_parameterization": olmoe_parameterization,
            "fidelity": {"kind": "exact"},
        }
    )
    olmoe_no_centaur = entry._build_controller_from_spec(
        {
            **base,
            "arm": "olmoe_no_centaur",
            "centaur": None,
            "model_parameterization": olmoe_parameterization,
            "fidelity": {"kind": "exact"},
        }
    )
    assert olmoe_no_proxy.action_centaur.ratio == 0.3
    assert olmoe_no_centaur.action_centaur is None
    assert olmoe_no_centaur.action_advisor is None
    with pytest.raises(ValueError, match="CMA-ES"):
        entry._build_controller_from_spec(
            {**base, "proposer": {"kind": "cma", "population_size": 4}}
        )
    cma_control = entry._build_controller_from_spec(
        {
            **base,
            "algorithm": "cma_control",
            "proposer": {"kind": "cma", "population_size": 4},
            "ipbt": None,
            "centaur": None,
            "controller": {
                **base["controller"],
                "restart_mode": "ipbt_reference",
            },
        }
    )
    assert cma_control._is_cma is True
    with pytest.raises(ValueError, match="real FT-PFN"):
        entry._build_controller_from_spec({**base, "algorithm": "brainlift"})


def test_production_brainlift_provisions_public_ftpfn_artifact(monkeypatch, tmp_path):
    import olmo_core.hpo.artifacts as hpo_artifacts
    import olmo_core.hpo.ftpfn as ftpfn

    artifact = tmp_path / "public-ftpfn.pt"
    calls = {}

    class FakePosterior:
        def __init__(self, artifact_path, *, device):
            calls["artifact_path"] = artifact_path
            calls["device"] = device

    class Advisor:
        def advise(self, state):
            raise AssertionError("not called during construction")

    monkeypatch.setattr(
        hpo_artifacts,
        "ensure_ftpfn_artifact",
        lambda cache_dir=None: (
            calls.setdefault("cache_dir", cache_dir),
            artifact,
        )[1],
    )
    monkeypatch.setattr(ftpfn, "TrustedFTPFN", FakePosterior)
    monkeypatch.setattr(entry, "_load_object", lambda ref: lambda **kwargs: Advisor())

    entry._build_controller_from_spec(
        {
            "algorithm": "brainlift",
            "seed": 7,
            "search_space": [{"name": "x", "low": 0.0, "high": 1.0}],
            "normalizer": {"ce_at_zero": 6.0, "ce_at_one": 2.0},
            "posterior": {
                "kind": "ftpfn",
                "cache_dir": str(tmp_path / "cache"),
                "device": "cpu",
            },
            "proposer": {"kind": "ifbo"},
            "btt": {},
            "ipbt": {
                "population_size": 2,
                "top_quantile": 0.5,
                "bottom_quantile": 0.5,
                "update_interval_init": 2,
            },
            "centaur": {
                "scope": "multi_action",
                "model": "gpt-5.6-sol",
                "ratio": 0.3,
                "warmup": 0,
                "advisor_factory": "module:factory",
            },
            "controller": {
                "target_tokens": 8,
                "quantum": 2,
                "n_fidelity_bins": 4,
                "worker_count": 2,
                "budget_tokens": 32,
                "checkpoint_root": "/tmp/hpo",
            },
        }
    )
    assert calls == {
        "cache_dir": str(tmp_path / "cache"),
        "artifact_path": str(artifact),
        "device": "cpu",
    }


def test_brainlift_builder_requires_sol_multi_action_at_paper_ratio(monkeypatch):
    class Advisor:
        def advise(self, state):
            raise AssertionError("not invoked during construction")

    monkeypatch.setattr(entry, "_load_object", lambda ref: lambda **kwargs: Advisor())
    spec = {
        "algorithm": "brainlift_test",
        "seed": 7,
        "search_space": [{"name": "x", "low": 0.0, "high": 1.0}],
        "normalizer": {"ce_at_zero": 6.0, "ce_at_one": 2.0},
        "posterior": {"kind": "synthetic", "optimum": [0.5]},
        "proposer": {"kind": "ifbo"},
        "btt": {},
        "ipbt": {
            "population_size": 2,
            "top_quantile": 0.5,
            "bottom_quantile": 0.5,
            "update_interval_init": 2,
        },
        "centaur": {
            "scope": "multi_action",
            "model": "gpt-5.6-sol",
            "ratio": 0.3,
            "warmup": 0,
            "advisor_factory": "module:factory",
        },
        "controller": {
            "target_tokens": 8,
            "quantum": 2,
            "n_fidelity_bins": 4,
            "worker_count": 2,
            "budget_tokens": 32,
            "checkpoint_root": "/tmp/hpo",
        },
    }
    controller = entry._build_controller_from_spec(spec)
    assert controller.centaur is None
    assert controller.action_centaur.ratio == 0.3
    assert controller.action_advisor.required_model == "gpt-5.6-sol"

    for bad_centaur in (
        {**spec["centaur"], "scope": "config_only"},
        {**spec["centaur"], "model": "claude-opus-4.6"},
        {**spec["centaur"], "ratio": 0.5},
        {**spec["centaur"], "warmup": 1},
    ):
        with pytest.raises(ValueError):
            entry._build_controller_from_spec({**spec, "centaur": bad_centaur})


def test_reference_ipbt_rejects_brainlift_restart_trigger():
    spec = {
        "algorithm": "ipbt_reference",
        "seed": 7,
        "search_space": [{"name": "x", "low": 0.0, "high": 1.0}],
        "normalizer": {"ce_at_zero": 6.0, "ce_at_one": 2.0},
        "posterior": {"kind": "synthetic", "optimum": [0.5]},
        "proposer": {"kind": "random"},
        "btt": {},
        "ipbt": {
            "population_size": 2,
            "top_quantile": 0.5,
            "bottom_quantile": 0.5,
            "update_interval_init": 2,
        },
        "controller": {
            "target_tokens": 8,
            "quantum": 2,
            "n_fidelity_bins": 4,
            "worker_count": 2,
            "budget_tokens": 32,
            "checkpoint_root": "/tmp/hpo",
            "restart_mode": "btt_aggregate",
        },
    }
    with pytest.raises(ValueError, match="reference"):
        entry._build_controller_from_spec(spec)


def test_run_yaml_exists_with_explicit_dtype_token():
    run_yaml = Path(__file__).parent.parent.parent / ".edullm" / "run.yaml"
    assert run_yaml.exists(), "expected a committed .edullm/run.yaml"
    text = run_yaml.read_text()
    assert "hpo_on_corpus.py" in text
    # The platform's precision guard reads the command text; a dtype token must be present.
    assert ("bfloat16" in text) or ("float32" in text)


def test_three_arm_specs_have_shared_2b_contract_and_only_declared_ablations():
    root = Path(__file__).resolve().parents[2] / ".edullm"
    specs = {
        name: json.loads((root / f"hpo-{name.replace('_', '-')}.json").read_text())
        for name in ("full_acronym_soup", "no_centaur", "no_proxy")
    }
    assert {spec["arm"] for spec in specs.values()} == set(specs)
    for spec in specs.values():
        controller = spec["controller"]
        assert controller["budget_tokens"] == 2_000_158_720
        assert controller["target_tokens"] == 10 * controller["quantum"] == 500_039_680
        assert controller["n_fidelity_bins"] == 10
        assert controller["worker_count"] == 8
        assert spec["ipbt"]["population_size"] == 16
        assert len(spec["search_space"]) == 9

    full = specs["full_acronym_soup"]
    no_centaur = specs["no_centaur"]
    no_proxy = specs["no_proxy"]
    assert full["centaur"] == no_proxy["centaur"]
    assert full["centaur"] == {
        "scope": "multi_action",
        "model": "gpt-5.6-sol",
        "ratio": 0.3,
        "warmup": 0,
        "advisor_factory": "olmo_core.hpo.openai_advisor:build_openai_advisor",
    }
    assert no_centaur["centaur"] is None
    assert (
        full["fidelity"]
        == no_centaur["fidelity"]
        == {
            "kind": "frozen_layer",
            "train_last_k": 8,
        }
    )
    assert no_proxy["fidelity"] == {"kind": "exact"}
    assert full["model_parameterization"]["kind"] == "umup"
    assert no_centaur["model_parameterization"]["depth"] == 16
    assert no_proxy["model_parameterization"] == {
        "kind": "standard",
        "architecture": "olmo2_190M",
        "depth": 12,
        "backend": "none",
    }
    assert full["experiment_factory"].endswith(":build_umup_hpo_experiment")
    assert no_proxy["experiment_factory"].endswith(":build_comparison_experiment")
    assert full["proxy_evidence_contract"] == no_proxy["proxy_evidence_contract"]
    assert "proxy_evidence_path" not in no_proxy
    assert "frozen_ranking_policy" not in no_proxy

    def without(spec, *keys):
        return {key: value for key, value in spec.items() if key not in keys}

    assert without(full, "arm", "centaur") == without(no_centaur, "arm", "centaur")
    for key in (
        "seed",
        "search_space",
        "normalizer",
        "posterior",
        "proposer",
        "btt",
        "ipbt",
        "controller",
        "factory_kwargs",
        "base_global_batch_size",
        "heldout_metric",
    ):
        assert full[key] == no_proxy[key]
    assert entry._validate_evidence_gates(full).value == "reporting_only"
    assert entry._validate_evidence_gates(no_centaur).value == "reporting_only"
    assert entry._validate_evidence_gates(no_proxy).value == "prune_promote"


def test_curriculum_spec_matches_no_proxy_except_approved_data_and_budget_contract():
    root = Path(__file__).resolve().parents[2] / ".edullm"
    no_proxy = json.loads((root / "hpo-no-proxy.json").read_text())
    curriculum = json.loads((root / "hpo-curriculum-quadratic-mtld.json").read_text())

    assert curriculum["arm"] == "curriculum_quadratic_mtld"
    assert curriculum["experiment_factory"].endswith(":build_curriculum_hpo_experiment")
    assert curriculum["model_parameterization"] == no_proxy["model_parameterization"]
    assert curriculum["fidelity"] == no_proxy["fidelity"] == {"kind": "exact"}
    assert "proxy_evidence_contract" not in curriculum
    assert curriculum["search_space"] == no_proxy["search_space"]
    assert curriculum["centaur"] == no_proxy["centaur"]
    assert curriculum["posterior"] == no_proxy["posterior"]
    assert curriculum["proposer"] == no_proxy["proposer"]
    assert curriculum["controller"]["quantum"] == 50_331_648
    assert curriculum["controller"]["target_tokens"] == 503_316_480
    assert curriculum["controller"]["budget_tokens"] == 2_013_265_920
    assert curriculum["btt"]["min_fidelity"] == 50_331_648
    assert curriculum["ipbt"]["update_interval_init"] == 50_331_648
    assert curriculum["base_global_batch_size"] == 524_288
    assert curriculum["factory_kwargs"]["global_batch_size"] == 524_288
    assert curriculum["heldout_metric"] == "eval/lm/opt-with-synthetic-10b-val/CE loss"
    assert curriculum["curriculum_identity"]["parent"] == {
        "dataset_id": "pretrain/opt-with-synthetic-10b",
        "version": "v1",
        "group": "tokens",
        "profile": "pretrain-tokens/v1",
        "manifest_sha256": "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797",
        "source_ids": [
            "algebraic-stack",
            "arxiv",
            "dclm",
            "nemotron-hqdqa",
            "open-web-math",
            "pes2o",
            "starcoder",
            "wiki",
        ],
    }
    assert curriculum["curriculum_identity"]["order"] == {
        "dataset_id": "curriculum/opt-with-synthetic-10b",
        "version": "v1",
        "group": "mtld",
        "profile": "token-order/v1",
        "manifest_sha256": "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d",
        "source_ids": [],
    }
    assert curriculum["curriculum_identity"]["token_phase_boundaries"][-1] == 211_122_685
    for batch_size in (256 * 1024, 512 * 1024, 1024 * 1024):
        assert curriculum["controller"]["quantum"] % batch_size == 0
        assert curriculum["controller"]["target_tokens"] % batch_size == 0


def test_curriculum_no_centaur_spec_is_a_clean_controller_ablation():
    from olmo_core.hpo.arms import Arm

    root = Path(__file__).resolve().parents[2] / ".edullm"
    curriculum = json.loads((root / "hpo-curriculum-quadratic-mtld.json").read_text())
    no_centaur = json.loads((root / "hpo-curriculum-quadratic-mtld-no-centaur.json").read_text())

    assert Arm.CURRICULUM_QUADRATIC_MTLD_NO_CENTAUR.value == no_centaur["arm"]
    assert no_centaur["centaur"] is None
    assert no_centaur["fidelity"] == {"kind": "exact"}
    assert entry._validate_evidence_gates(no_centaur).value == "prune_promote"
    assert no_centaur["experiment_factory"].endswith(":build_curriculum_hpo_experiment")
    assert no_centaur["curriculum_identity"] == curriculum["curriculum_identity"]
    assert no_centaur["heldout_metric"] == curriculum["heldout_metric"]

    ignored = {
        "arm",
        "centaur",
        "controller",
        "controller_state_path",
        "controller_snapshot_root",
        "study_result_path",
    }
    assert {key: value for key, value in no_centaur.items() if key not in ignored} == {
        key: value for key, value in curriculum.items() if key not in ignored
    }
    assert no_centaur["controller"] == {
        **curriculum["controller"],
        "checkpoint_root": (
            "${EDULLM_CHECKPOINT_DIR}/curriculum-quadratic-mtld-no-centaur/${EDULLM_RUN_ID}"
        ),
    }
    assert no_centaur["controller_state_path"].startswith(
        "/tmp/hpo-curriculum-quadratic-mtld-no-centaur-"
    )
    assert no_centaur["controller_snapshot_root"].startswith(
        "${EDULLM_CHECKPOINT_DIR}/curriculum-quadratic-mtld-no-centaur/"
    )
    assert no_centaur["study_result_path"].startswith(
        "/tmp/hpo-curriculum-quadratic-mtld-no-centaur-"
    )


@pytest.mark.parametrize(
    "spec_name",
    (
        "hpo-curriculum-quadratic-mtld.json",
        "hpo-curriculum-quadratic-mtld-no-centaur.json",
    ),
)
@pytest.mark.parametrize("global_batch_mult", (0.5, 1.0, 2.0))
def test_curriculum_finalist_continuation_uses_all_eight_gpus(spec_name, global_batch_mult):
    root = Path(__file__).resolve().parents[2] / ".edullm"
    spec = json.loads((root / spec_name).read_text())
    allocation = SimpleNamespace(
        kind="resume",
        checkpoint_ref="/checkpoints/winner",
        transition=None,
        realized_hps={"global_batch_mult": global_batch_mult},
    )

    assert entry._eligible_finalist_continuation(
        [allocation],
        controller_spec=spec,
        gpu_ids=spec["gpu_ids"],
    ) == {
        "enabled": True,
        "world_size": 8,
        "rank_microbatch_size": 4_096,
    }


def test_posthoc_no_centaur_variant_removes_the_failed_proxy():
    from olmo_core.hpo.arms import Arm

    contract = entry._arm_contract(
        Arm.NO_CENTAUR,
        "proxy_removed_after_failed_admission",
    )
    assert contract["model_parameterization"] == "stock_olmo2_190m"
    assert contract["freeze_first_n_blocks"] == 0
    assert contract["llm_ratio"] == 0.0
    assert contract["llm_scope"] == "none"


def test_global_batch_search_is_aligned_to_every_fidelity_rung():
    quantum = 50_003_968
    for requested in (16_384.0, 30_000.0, 32_768.0, 60_000.0, 65_536.0):
        realized = entry._aligned_global_batch_size(
            requested,
            quantum=quantum,
            multiple=4_096,
        )
        assert quantum % realized == 0
        assert realized % 4_096 == 0


def test_study_result_persists_winner_top_five_and_a100_seconds(tmp_path):
    candidates = [(f"trial-{index}", (index / 10,), 2.0 + index / 10) for index in range(5)]
    controller = SimpleNamespace(
        top_candidates=lambda limit: candidates[:limit],
        search_space=SimpleNamespace(from_unit=lambda unit: {"lr": unit[0]}),
        state=lambda: SimpleNamespace(
            accelerator_seconds_charged=7200.0,
            tokens_charged=2_000_158_720,
            final_evaluations=[],
        ),
    )
    result_path = tmp_path / "result.json"
    exact_result = WorkerObservation(
        trial_id="trial-0-exact-retrain",
        tokens=500_039_680,
        heldout_ce=2.0,
        train_ce_history=(),
        grad_norm_history=(),
        activation_ratio=None,
        numeric_failure=False,
        checkpoint_ref="/ckpt/trial-0-exact-retrain",
        accelerator_seconds=1800.0,
    )
    entry._persist_study_result(
        controller,
        {"arm": "no_proxy", "study_result_path": str(result_path)},
        SimpleNamespace(value="prune_promote"),
        exact_result=exact_result,
    )
    payload = json.loads(result_path.read_text())
    assert payload["winner"]["trial_id"] == "trial-0"
    assert len(payload["top_five_full_fidelity"]) == 5
    assert payload["search_a100_seconds"] == 7200.0
    assert payload["exact_retrain_a100_seconds"] == 1800.0
    assert payload["total_a100_seconds"] == 9000.0
    assert payload["total_a100_hours"] == 2.5
    assert payload["total_tokens_charged"] == 2_500_198_400
    assert payload["trusted_for_selection"] is True
