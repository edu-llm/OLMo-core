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
    assert argv[:2] == ["python", ".edullm/hpo_on_corpus.py"]
    parsed = entry._parse_args(argv[2:])
    assert parsed.run_id == "run-abc"
    assert parsed.param_dtype == "bfloat16"


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
    }
    path = tmp_path / "segment.json"
    path.write_text(json.dumps(segment_spec))
    config = object()
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
            "/run/ckpt/trials/t0",
            "--hard-stop-tokens",
            "3072",
            "--segment-spec",
            str(path),
        ]
    )
    assert entry.run_segment(args) == 0
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


def test_controller_evidence_gate_rejects_unvalidated_proxy_and_unequal_budget():
    with pytest.raises(ValueError):
        entry._validate_evidence_gates({"fidelity": {"kind": "frozen_layer", "train_last_k": 4}})
    valid_proxy = {
        "admission": {
            "min_rank_corr": 0.7,
            "min_top_k_recall": 0.6,
        },
        "metrics": {
            "rank_corr_mean": 0.9,
            "rank_corr_std": 0.01,
            "top_k_recall": 0.9,
            "top_k_recall_std": 0.01,
            "n": 20,
            "net_compute_savings": 0.2,
            "beats_exact_at_equal_budget": True,
            "proxy_kind": "frozen_layer",
        },
    }
    with pytest.raises(ValueError, match="budget"):
        entry._validate_evidence_gates(
            {
                "fidelity": {"kind": "frozen_layer", "train_last_k": 4},
                "proxy_evidence": valid_proxy,
            }
        )
    with pytest.raises(ValueError, match="kind"):
        entry._validate_evidence_gates(
            {
                "fidelity": {"kind": "umup", "width_factor": 0.5},
                "proxy_evidence": valid_proxy,
                "budget_comparison": {},
            }
        )
    with pytest.raises(ValueError):
        entry._validate_evidence_gates(
            {
                "proxy_evidence": {
                    "admission": {
                        "min_rank_corr": 0.7,
                        "min_top_k_recall": 0.6,
                    },
                    "metrics": {
                        "rank_corr_mean": 0.5,
                        "rank_corr_std": 0.1,
                        "top_k_recall": 0.5,
                        "top_k_recall_std": 0.1,
                        "n": 20,
                        "net_compute_savings": 0.2,
                        "beats_exact_at_equal_budget": True,
                    },
                }
            }
        )

    categories = [
        "main",
        "failed",
        "screened",
        "reset",
        "retrained",
        "llm_controller",
        "proxy_selection",
    ]
    with pytest.raises(ValueError):
        entry._validate_evidence_gates(
            {
                "budget_comparison": {
                    "candidate": {
                        "accelerator_seconds": {name: 1.0 for name in categories},
                        "tokens": {name: 1 for name in categories},
                    },
                    "control": {
                        "accelerator_seconds": {name: 1.0 for name in categories},
                        "tokens": {name: 1000 for name in categories},
                    },
                }
            }
        )


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
