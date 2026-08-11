import base64
import importlib.util
import json
import os
import shlex
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from olmo_core.hpo.capacity_block import (
    CapacityBlockBackend,
    CapacityBlockConfig,
    CapacityTrial,
    GhWorkflowGateway,
    parse_idle_nodes,
    parse_stale_nodes,
)
from olmo_core.hpo.types import WorkerObservation


def _load_entrypoint():
    path = Path(__file__).parents[3] / ".edullm" / "hpo_on_corpus.py"
    spec = importlib.util.spec_from_file_location("capacity_block_hpo_entry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _payload(trial_id: str) -> dict[str, object]:
    return {
        "experiment_factory": "module:factory",
        "factory_kwargs": {"rank_microbatch_size": 32_768},
        "target_tokens": 499_808_608,
        "global_batch_size": 262_144,
        "realized_hps": {"lr": 1e-3},
        "checkpoint_root": "s3://checkpoints/hpo",
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "checkpoint_ref": None,
        "transition": None,
        "fidelity": {"kind": "exact"},
        "model_parameterization": {"kind": "standard", "architecture": "olmoe_1B_7B"},
        "config_hash": f"hash-{trial_id}",
    }


def _observation(trial_id: str) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "tokens": 49_807_360,
        "heldout_ce": 3.25,
        "train_ce_history": [3.4, 3.3],
        "grad_norm_history": [0.9],
        "activation_ratio": None,
        "numeric_failure": False,
        "checkpoint_ref": f"s3://checkpoints/hpo/trials/{trial_id}/step190",
        "accelerator_seconds": 0.0,
    }


class FakeGateway:
    def __init__(self, observations: dict[str, dict[str, object]]) -> None:
        self.observations = observations
        self.dispatches: list[tuple[str, dict[str, str], str]] = []
        self.events: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def dispatch(self, workflow: str, inputs: dict[str, str]) -> str:
        with self._lock:
            run_id = f"{workflow}-{len(self.dispatches)}"
            self.dispatches.append((workflow, dict(inputs), run_id))
            self.events.append(("dispatch", workflow))
            return run_id

    def wait(self, run_id: str) -> str:
        workflow = run_id.rsplit("-", 1)[0]
        with self._lock:
            self.events.append(("wait", workflow))
        if run_id.startswith("block-status"):
            return "node 1  8/8 GPUs busy\nnode 2  IDLE\nnode 7  IDLE\n"
        if run_id.startswith("block-logs"):
            matching = [
                inputs["run_name"]
                for workflow_name, inputs, identifier in self.dispatches
                if workflow_name == "block-logs.yml" and identifier == run_id
            ]
            assert len(matching) == 1
            trial_id = matching[0].rsplit("-", 1)[-1]
            payload = self.observations[trial_id]
            return f"ordinary log line\nEDULLM_HPO_OBSERVATION={json.dumps(payload)}\n"
        return "capacity block run started"


def _config(**kwargs) -> CapacityBlockConfig:
    values = {
        "branch": "edullm/hpo-complex",
        "repository": "edu-llm/OLMo-core",
        "checkpoint_root": "s3://checkpoints/hpo",
        "max_workers": 8,
        "poll_interval_seconds": 0,
    }
    values.update(kwargs)
    return CapacityBlockConfig(**values)


def test_idle_parser_only_accepts_explicit_idle_nodes():
    logs = "\n".join(
        (
            "Read fleet\t2026-08-11T12:00:00Z node 1  8/8 GPUs busy",
            "Read fleet\t2026-08-11T12:00:00Z node 2  IDLE",
            "Read fleet\t2026-08-11T12:00:00Z node 3  agent has not registered",
            "Read fleet\t2026-08-11T12:00:00Z node 7  IDLE",
        )
    )
    assert parse_idle_nodes(logs) == [2, 7]
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_idle_nodes("node 2  IDLE\nnode 2  IDLE\n")
    with pytest.raises(RuntimeError, match="no node readings"):
        parse_idle_nodes("workflow completed without a fleet table")


def test_stale_parser_only_accepts_explicit_stale_claims():
    logs = "\n".join(
        (
            "Read fleet\t2026-08-11T12:00:00Z node 1  STALE CLAIM  user / old-run exited",
            "Read fleet\t2026-08-11T12:00:00Z node 2  IDLE",
            "Read fleet\t2026-08-11T12:00:00Z node 7  8/8 GPUs busy",
        )
    )
    assert parse_stale_nodes(logs) == [1]
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_stale_nodes("node 1  STALE CLAIM x\nnode 1  STALE CLAIM x\n")


def test_backend_dispatches_independent_safe_eight_gpu_trials_before_waiting():
    gateway = FakeGateway({"a": _observation("a"), "b": _observation("b")})
    ticks = iter((10.0, 10.0, 13.0, 14.0))
    backend = CapacityBlockBackend(_config(max_workers=2), gateway, clock=lambda: next(ticks))
    trials = [
        CapacityTrial("a", 10, 49_807_360, _payload("a")),
        CapacityTrial("b", 11, 49_807_360, _payload("b")),
    ]

    results = backend.run(trials, idle_nodes=(2, 7))

    starts = [item for item in gateway.dispatches if item[0] == "block-run.yml"]
    assert [item[1]["node"] for item in starts] == ["2", "7"]
    assert all(item[1]["processes"] == "all" for item in starts)
    assert all(item[0] != "block-run-distributed.yml" for item in gateway.dispatches)
    first_wait = gateway.events.index(("wait", "block-run.yml"))
    assert gateway.events[:first_wait].count(("dispatch", "block-run.yml")) == 2

    for _, inputs, _ in starts:
        command = inputs["command"]
        words = shlex.split(command)
        assert words[:3] == ["python", ".edullm/hpo_on_corpus.py", "hpo-run"]
        assert "torchrun" not in command
        assert "torch.distributed" not in command
        assert "--moe-shard-degree" not in command
        assert "--moe-num-replicas" not in command
        assert all(word.startswith("--") for word in words[3:])
        assert "--worker-world-size=8" in words
        assert words[-1].startswith("--observation-path=")

        encoded = next(
            word.split("=", 1)[1] for word in words if word.startswith("--segment-spec-payload=")
        )
        decoded = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
        assert decoded["checkpoint_root"] == "s3://checkpoints/hpo"

    assert [result.trial_id for result in results] == ["a", "b"]
    assert results[0].accelerator_seconds == pytest.approx(24.0)
    assert results[1].accelerator_seconds == pytest.approx(32.0)


def test_backend_releases_its_stale_claim_after_segment_exit():
    class ExitingGateway(FakeGateway):
        states = iter(
            (
                "node 2  8/8 GPUs busy\n",
                "node 2  STALE CLAIM  researcher / hpo-10-a exited\n",
                "node 2  IDLE\n",
            )
        )

        def wait(self, run_id: str) -> str:
            if run_id.startswith("block-status"):
                return next(self.states)
            return super().wait(run_id)

    gateway = ExitingGateway({"a": _observation("a")})
    ticks = iter((10.0, 13.0))
    backend = CapacityBlockBackend(_config(), gateway, clock=lambda: next(ticks))

    result = backend.run(
        [CapacityTrial("a", 10, 49_807_360, _payload("a"))],
        idle_nodes=(2,),
    )

    assert result[0].trial_id == "a"
    releases = [
        inputs for workflow, inputs, _ in gateway.dispatches if workflow == "block-release.yml"
    ]
    assert releases == [{"nodes": "2", "region": "us-east-2"}]


def test_backend_discovers_capacity_and_caps_dynamic_worker_count():
    gateway = FakeGateway({})
    backend = CapacityBlockBackend(_config(max_workers=1), gateway)
    nodes = backend.discover_idle_nodes()
    assert nodes == [2, 7]
    assert backend.worker_count(nodes) == 1
    assert gateway.dispatches[0][0] == "block-status.yml"


def test_backend_retries_durable_log_until_observation_is_synced():
    class DelayedObservationGateway(FakeGateway):
        attempts = 0

        def wait(self, run_id: str) -> str:
            if run_id.startswith("block-logs"):
                self.attempts += 1
                if self.attempts == 1:
                    return "log has synced, but not its final line yet"
            return super().wait(run_id)

    gateway = DelayedObservationGateway({"a": _observation("a")})
    ticks = iter((10.0, 13.0))
    backend = CapacityBlockBackend(_config(), gateway, clock=lambda: next(ticks))
    result = backend.run(
        [CapacityTrial("a", 10, 49_807_360, _payload("a"))],
        idle_nodes=(2,),
    )
    assert result[0].trial_id == "a"
    assert gateway.attempts == 2


def test_backend_resumes_an_oversized_pending_batch_in_safe_waves():
    gateway = FakeGateway({"a": _observation("a"), "b": _observation("b")})
    ticks = iter((10.0, 13.0, 20.0, 24.0))
    backend = CapacityBlockBackend(
        _config(max_workers=1),
        gateway,
        clock=lambda: next(ticks),
    )
    results = backend.run(
        [
            CapacityTrial("a", 10, 49_807_360, _payload("a")),
            CapacityTrial("b", 11, 49_807_360, _payload("b")),
        ],
        idle_nodes=(2,),
    )
    assert [result.trial_id for result in results] == ["a", "b"]
    starts = [inputs for workflow, inputs, _ in gateway.dispatches if workflow == "block-run.yml"]
    assert len(starts) == 2
    assert all(inputs["processes"] == "all" for inputs in starts)


def test_gateway_correlates_new_runs_and_only_invokes_gh():
    calls: list[list[str]] = []
    listed = iter(
        (
            json.dumps([{"databaseId": 10, "actor": {"login": "researcher"}}]),
            json.dumps(
                [
                    {"databaseId": 11, "actor": {"login": "researcher"}},
                    {"databaseId": 10, "actor": {"login": "researcher"}},
                ]
            ),
        )
    )

    def runner(argv, **kwargs):
        del kwargs
        words = list(argv)
        calls.append(words)
        if words[1:3] == ["api", "user"]:
            return SimpleNamespace(returncode=0, stdout="researcher\n", stderr="")
        if words[1:3] == ["run", "list"]:
            return SimpleNamespace(returncode=0, stdout=next(listed), stderr="")
        if words[1:3] == ["workflow", "run"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if words[1:3] == ["run", "watch"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if words[1:3] == ["run", "view"]:
            return SimpleNamespace(returncode=0, stdout="node 2  IDLE\n", stderr="")
        raise AssertionError(words)

    gateway = GhWorkflowGateway(runner=runner, sleep=lambda _: None)
    run_id = gateway.dispatch("block-status.yml", {"region": "us-east-2"})
    assert run_id == "11"
    assert gateway.wait(run_id) == "node 2  IDLE\n"
    assert calls
    assert all(call[0] == "gh" for call in calls)
    assert not any("aws" in word for call in calls for word in call)
    run_lists = [call for call in calls if call[1:3] == ["run", "list"]]
    assert all(call[call.index("--user") + 1] == "researcher" for call in run_lists)


def test_entrypoint_routes_only_explicit_capacity_backend_away_from_local_subprocess(
    monkeypatch, tmp_path
):
    entry = _load_entrypoint()
    allocation = SimpleNamespace(
        trial_id="remote",
        decision_id=21,
        target_fidelity=49_807_360,
        realized_hps={"lr": 1e-3},
        checkpoint_ref=None,
        transition=None,
    )
    spec = {
        "launch_backend": "capacity_block",
        "capacity_block": {
            "branch": "edullm/hpo-complex",
            "repository": "edu-llm/OLMo-core",
        },
        "max_workers": 8,
        "worker_world_size": 8,
        "base_global_batch_size": 262_144,
        "experiment_factory": "module:factory",
        "factory_kwargs": {"rank_microbatch_size": 32_768},
        "controller": {
            "target_tokens": 499_808_608,
            "quantum": 49_807_360,
            "checkpoint_root": "s3://checkpoints/hpo",
            "worker_count": 8,
        },
        "search_validation_callback": "search_validation",
        "untouched_evaluator": "final_evaluation",
        "heldout_metric": "eval/search_validation/val/CE loss",
        "segment_spec_dir": str(tmp_path),
    }
    expected = WorkerResult = SimpleNamespace(trial_id="remote")

    class Backend:
        def run(self, trials, *, run_id, idle_nodes=None, heartbeat=None):
            assert len(trials) == 1
            assert run_id == "hpo-run"
            assert trials[0].payload["global_batch_size"] == 262_144
            assert idle_nodes == (4,)
            assert heartbeat is not None
            return [expected]

    monkeypatch.setattr(
        entry.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("capacity backend used local subprocess"),
    )
    result = entry._dispatch_allocations(
        allocations=[allocation],
        controller_spec=spec,
        run_id="hpo-run",
        param_dtype="bfloat16",
        heartbeat=lambda: None,
        capacity_backend=Backend(),
        capacity_nodes=(4,),
    )
    assert result == [WorkerResult]


def test_remote_worker_decodes_payload_and_persists_durable_observation(
    monkeypatch, tmp_path, capsys
):
    entry = _load_entrypoint()
    payload = _payload("remote")
    payload.pop("config_hash")
    payload["worker_environment"] = {
        "EDULLM_DATASET_ID": "pretrain/opt-with-synthetic-10b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_DATASET_TOKENIZER": "tokenizer/dolma2-bpe",
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    config = SimpleNamespace(trainer=SimpleNamespace())
    result = _observation("remote")
    worker_module = SimpleNamespace(
        WorkerConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        trial_namespace=lambda root, trial_id: f"{root}/trials/{trial_id}",
        should_emit_worker_result=lambda env: env.get("RANK", "0") == "0",
    )
    objective_module = SimpleNamespace(EvaluatorGate=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setitem(sys.modules, "olmo_core.hpo.worker", worker_module)
    monkeypatch.setitem(sys.modules, "olmo_core.hpo.objective", objective_module)
    monkeypatch.setattr(entry, "_load_object", lambda reference: lambda **kwargs: config)
    monkeypatch.setattr(
        entry,
        "_run_configured_segment",
        lambda **kwargs: WorkerObservation(**result),
    )
    observation_path = tmp_path / "observation.json"
    args = entry._parse_args(
        [
            "hpo-run",
            "--run-segment",
            "--worker-world-size=8",
            "--trial-id=remote",
            "--checkpoint-dir=s3://checkpoints/hpo/trials/remote",
            "--hard-stop-tokens=49807360",
            "--param-dtype=bfloat16",
            f"--segment-spec-payload={encoded}",
            f"--observation-path={observation_path}",
        ]
    )

    assert entry.run_segment(args) == 0
    assert os.environ["EDULLM_DATASET_ID"] == "pretrain/opt-with-synthetic-10b"
    assert os.environ["EDULLM_DATASET_VERSION"] == "v1"
    assert os.environ["EDULLM_DATASET_TOKENIZER"] == "tokenizer/dolma2-bpe"
    assert os.environ["EDULLM_CHECKPOINT_DIR"] == "s3://checkpoints/hpo/trials/remote"
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("EDULLM_HPO_OBSERVATION=")
    assert json.loads(lines[0].split("=", 1)[1])["trial_id"] == "remote"
    assert json.loads(observation_path.read_text())["checkpoint_ref"].endswith("step190")


def test_capacity_round_refreshes_controller_worker_count_from_idle_nodes():
    entry = _load_entrypoint()
    controller = SimpleNamespace(config=SimpleNamespace(worker_count=8))

    class Backend:
        def wait_for_idle_nodes(self, *, heartbeat):
            assert heartbeat is not None
            return [2, 7, 8]

        def worker_count(self, idle_nodes):
            assert idle_nodes == [2, 7, 8]
            return 2

    nodes = entry._refresh_capacity_workers(
        controller,
        Backend(),
        heartbeat=lambda: None,
    )
    assert nodes == (2, 7)
    assert controller.config.worker_count == 2
