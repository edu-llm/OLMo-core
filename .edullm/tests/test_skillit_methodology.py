from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EDULLM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EDULLM_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(EDULLM_DIR))

from production_contract.checkpoint import (  # noqa: E402
    permanent_checkpoint_steps,
    write_last_durable_step,
)
from production_contract.task_loss import TASK_LOSS_RAW_LABELS  # noqa: E402
import skillit_controller as skillit_controller_module  # noqa: E402
from skillit_controller import SkillItController  # noqa: E402
from skillit_entrypoint import (  # noqa: E402
    EntrypointError,
    platform_values,
    torchrun_command,
)
from skillit_loader import (  # noqa: E402
    GLOBAL_BATCH_TOKENS,
    SkillItDataError,
    WeightedDomainDataLoader,
)
from skillit_math import (  # noqa: E402
    ARMS,
    DOMAINS,
    FAMILIES,
    RECIPE_SHA256,
    UPDATE_STEPS,
    adjacency,
    arm_by_index,
    derivative_a,
    initial_weights,
    load_recipe,
    offline_a,
    update_weights,
)
from train_skillit_370m import (  # noqa: E402
    RANK_MICROBATCH_TOKENS,
    SEQUENCE_LENGTH,
    TOTAL_STEPS,
    ResumeAwareTaskLossEvalCallback,
    build_model_config,
    build_train_module_config,
    build_trainer_config,
    fit_with_resume,
    validate_trainer_assembly,
)


class FakeDataset:
    def __init__(self, domain_index: int, length: int = 257) -> None:
        self.domain_index = domain_index
        self.length = length
        self.fingerprint = f"domain-{domain_index}-immutable"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.full((4,), self.domain_index * 1_000 + index)}


def _loader(
    tmp_path: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> WeightedDomainDataLoader:
    return WeightedDomainDataLoader(
        [FakeDataset(index) for index in range(len(DOMAINS))],
        work_dir=tmp_path / f"rank{rank}",
        global_batch_size=32,
        sequence_length=4,
        total_batches=20,
        dp_world_size=world_size,
        dp_rank=rank,
    )


def test_recipe_is_immutable_and_contains_only_final_370m_arms() -> None:
    payload = load_recipe()
    assert RECIPE_SHA256
    assert tuple((arm.arm_id, arm.a_mode, arm.wandb_project) for arm in ARMS) == (
        ("probe", "probe", "skillit"),
        ("deriv", "derivative", "skillit"),
    )
    assert payload["methodology"]["completed_pilots_are_final"] is True
    assert tuple(payload["skillit"]["update_steps"]) == UPDATE_STEPS


def test_exact_skillit_math_for_both_arms() -> None:
    p = initial_weights()
    losses = np.asarray([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    probe_a = adjacency("probe", p)
    assert np.array_equal(probe_a, offline_a())
    expected_logits = 0.2 * (probe_a @ losses)
    expected = np.exp(expected_logits - expected_logits.max())
    expected /= expected.sum()
    assert np.allclose(update_weights(probe_a, losses), expected)

    deriv_a = derivative_a(p)
    assert deriv_a.shape == (7, 6)
    assert np.all(deriv_a >= 0)
    assert np.array_equal(adjacency("derivative", p), deriv_a)


def test_distributed_loader_is_one_deterministic_global_stream(tmp_path: Path) -> None:
    single = _loader(tmp_path / "single")
    rank0 = _loader(tmp_path / "distributed", rank=0, world_size=2)
    rank1 = _loader(tmp_path / "distributed", rank=1, world_size=2)

    expected = single.batch_at(7)["input_ids"]
    even = rank0.batch_at(7)["input_ids"]
    odd = rank1.batch_at(7)["input_ids"]
    reconstructed = torch.empty_like(expected)
    reconstructed[0::2] = even
    reconstructed[1::2] = odd
    assert torch.equal(reconstructed, expected)
    assert torch.equal(single.batch_at(7)["input_ids"], expected)


def test_distributed_loader_reconstructs_exactly_across_eight_ranks(
    tmp_path: Path,
) -> None:
    single = _loader(tmp_path / "single")
    expected = single.batch_at(3)["input_ids"]
    reconstructed = torch.empty_like(expected)
    for rank in range(8):
        reconstructed[rank::8] = _loader(
            tmp_path / "distributed",
            rank=rank,
            world_size=8,
        ).batch_at(3)["input_ids"]
    assert torch.equal(reconstructed, expected)


def test_loader_resume_restores_weights_and_next_batch(tmp_path: Path) -> None:
    original = _loader(tmp_path / "original")
    original.set_weights([1, 2, 3, 4, 5, 6, 7])
    original.batches_processed = 9
    original.tokens_processed = 9 * 32
    state = original.state_dict()

    resumed = _loader(tmp_path / "resumed")
    resumed.load_state_dict(state)
    assert resumed.weights_dict() == original.weights_dict()
    assert torch.equal(
        resumed.batch_at(resumed.batches_processed)["input_ids"],
        original.batch_at(original.batches_processed)["input_ids"],
    )

    state["source_fingerprints"][0] = "changed"
    with pytest.raises(SkillItDataError, match="source_fingerprints"):
        resumed.load_state_dict(state)


def _task_loss_payload(step: int) -> dict[str, object]:
    labels = {label: float(index + 1) / 10 for index, label in enumerate(TASK_LOSS_RAW_LABELS)}
    return {
        "step": step,
        "labels": labels,
        "task_loss_bpb": labels,
        "raw_label_count": len(TASK_LOSS_RAW_LABELS),
        "suite_complete": True,
        "macro_mean": sum(labels.values()) / len(labels),
    }


def test_controller_is_checkpoint_gated_logs_exact_state_and_resumes(
    tmp_path: Path,
) -> None:
    loader = _loader(tmp_path / "loader")
    task_loss_dir = tmp_path / "task_loss"
    task_loss_dir.mkdir()
    (task_loss_dir / "step500_task_loss.json").write_text(
        json.dumps(_task_loss_payload(500)), encoding="utf-8"
    )
    trainer = SimpleNamespace(
        global_step=500,
        data_loader=loader,
        callbacks={},
        record_metric=lambda *args, **kwargs: None,
    )
    controller = SkillItController(
        arm_id="deriv",
        a_mode="derivative",
        progress_dir=str(tmp_path / "progress"),
        task_loss_dir=str(task_loss_dir),
        production=False,
        wandb_mode="disabled",
    )
    controller.trainer = trainer
    controller.pre_train()

    assert 500 in controller.state_dict()["applied_steps"]
    records = [
        json.loads(line)
        for line in controller.updates_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["step"] == 500
    assert tuple(record["domain_order"]) == DOMAINS
    assert tuple(record["family_order"]) == FAMILIES
    assert record["r"] == record["p_before"]
    assert np.asarray(record["A"]).shape == (7, 6)
    assert set(record["losses"]) == set(FAMILIES)
    assert (controller.progress / "skillit_updates/step500_A.json").is_file()
    assert (controller.progress / "skillit_updates/step500_weights.json").is_file()

    controller.pre_train()
    assert len(controller.updates_jsonl.read_text().splitlines()) == 1

    resumed_loader = _loader(tmp_path / "resumed")
    resumed_trainer = SimpleNamespace(global_step=875, data_loader=resumed_loader)
    resumed_controller = SkillItController(
        arm_id="deriv",
        a_mode="derivative",
        progress_dir=str(controller.progress),
        task_loss_dir=str(task_loss_dir),
        production=False,
        wandb_mode="disabled",
    )
    resumed_controller.trainer = resumed_trainer
    resumed_controller.post_checkpoint_loaded("unused")
    assert resumed_loader.weights_dict() == record["p_after"]


def test_controller_uploads_every_matrix_and_domain_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _loader(tmp_path / "loader")
    task_loss_dir = tmp_path / "task_loss"
    task_loss_dir.mkdir()
    logged: list[tuple[int, dict[str, float]]] = []
    artifacts: list[tuple[list[str], set[str]]] = []

    class FakeRun:
        def log(self, metrics):
            logged.append((int(metrics["skillit/update_step"]), dict(metrics)))

    def log_directory(_run, path, *, aliases, **_kwargs):
        files = {
            item.name for item in (Path(path) / "skillit_updates").glob("*.json")
        }
        artifacts.append((list(aliases), files))

    monkeypatch.setattr(
        skillit_controller_module, "wandb_run_from_trainer", lambda _trainer: FakeRun()
    )
    monkeypatch.setattr(
        skillit_controller_module, "wandb_log_directory_artifact", log_directory
    )
    trainer = SimpleNamespace(
        global_step=0,
        data_loader=loader,
        callbacks={},
        record_metric=lambda *args, **kwargs: None,
    )
    controller = SkillItController(
        arm_id="deriv",
        a_mode="derivative",
        progress_dir=str(tmp_path / "progress"),
        task_loss_dir=str(task_loss_dir),
        production=False,
        wandb_mode="online",
    )
    controller.trainer = trainer
    controller.pre_train()

    (task_loss_dir / "step500_task_loss.json").write_text(
        json.dumps(_task_loss_payload(500)), encoding="utf-8"
    )
    trainer.global_step = 500
    controller.post_step()

    assert [step for step, _metrics in logged] == [0, 500]
    for _step, metrics in logged:
        assert len([key for key in metrics if key.startswith("skillit/matrix/")]) == 42
        assert len([key for key in metrics if key.startswith("skillit/p_before/")]) == 7
        assert len([key for key in metrics if key.startswith("skillit/p_after/")]) == 7
    assert artifacts[0] == (
        ["latest", "step-0000000"],
        {"step0_A.json", "step0_weights.json"},
    )
    assert artifacts[1] == (
        ["latest", "step-0000500"],
        {
            "step0_A.json",
            "step0_weights.json",
            "step500_A.json",
            "step500_weights.json",
        },
    )


def test_controller_refuses_update_without_strict_task_loss(tmp_path: Path) -> None:
    controller = SkillItController(
        arm_id="probe",
        a_mode="probe",
        progress_dir=str(tmp_path / "progress"),
        task_loss_dir=str(tmp_path / "missing"),
        production=False,
        wandb_mode="disabled",
    )
    controller.trainer = SimpleNamespace(global_step=500, data_loader=_loader(tmp_path))
    with pytest.raises(Exception, match="failed after checkpoint evaluation"):
        controller.pre_train()


def test_olmo2_370m_training_and_checkpoint_contract() -> None:
    model = build_model_config()
    train_module = build_train_module_config()
    assert model.d_model == 1_024
    assert model.n_layers == 16
    assert SEQUENCE_LENGTH == 2_048
    assert GLOBAL_BATCH_TOKENS == 4_194_304
    assert RANK_MICROBATCH_TOKENS == 32_768
    assert TOTAL_STEPS == 2_384
    assert train_module.optim.lr == 4e-4
    assert train_module.scheduler.warmup == 24
    assert train_module.scheduler.alpha_f == 0.1
    assert train_module.z_loss_multiplier == 1e-5
    steps = permanent_checkpoint_steps(total_steps=TOTAL_STEPS, interval=125)
    assert steps[0] == 0
    assert steps[-1] == 2_384
    assert 2_375 not in steps


def test_trainer_routes_wandb_and_orders_controller_after_task_loss(tmp_path: Path) -> None:
    config = build_trainer_config(
        arm_index=1,
        run_name="test-deriv",
        save_folder=str(tmp_path / "checkpoints"),
        progress_dir=str(tmp_path / "progress"),
        task_loss_dir=str(tmp_path / "task_loss"),
        eval_script="eval.py",
        task_loss_nproc=8,
        wandb_entity=None,
        wandb_mode="disabled",
        production=False,
    )
    assert config.callbacks["wandb"].project == "skillit"
    assert config.callbacks["task_loss"].priority > config.callbacks["skillit"].priority
    assert config.callbacks["task_loss"].total_steps == TOTAL_STEPS
    assert config.callbacks["task_loss"].interval == 125
    assert TOTAL_STEPS in config.callbacks["checkpointer"].fixed_steps


def test_resume_requires_matching_durable_eval_without_refinalizing(
    tmp_path: Path,
) -> None:
    results = tmp_path / "task_loss"
    progress = tmp_path / "progress"
    results.mkdir()
    (results / "step500_task_loss.json").write_text(
        json.dumps(_task_loss_payload(500)), encoding="utf-8"
    )
    write_last_durable_step(progress, 500)
    callback = ResumeAwareTaskLossEvalCallback(
        total_steps=TOTAL_STEPS,
        save_folder=tmp_path / "checkpoints",
        run_name="resume",
        results_dir=results,
        eval_script=tmp_path / "unused.py",
        progress_dir=progress,
        production=False,
        wandb_mode="disabled",
    )
    callback.trainer = SimpleNamespace(global_step=500)
    callback.pre_train()
    assert 500 in callback._completed


def test_executable_fit_handoff_and_explicit_resume(tmp_path: Path) -> None:
    class FakeTrainer:
        def __init__(self) -> None:
            self.fit_calls = 0
            self.load_calls: list[str] = []

        def maybe_load_checkpoint(self, path: str) -> bool:
            self.load_calls.append(path)
            return True

        def fit(self) -> None:
            self.fit_calls += 1

    save_folder = tmp_path / "checkpoints"
    identity = {"schema": 1, "arm": "probe"}
    trainer = FakeTrainer()
    fresh = SimpleNamespace(resume=False, load_path=None, save_folder=save_folder)
    fit_with_resume(trainer, fresh, identity)
    assert trainer.fit_calls == 1
    assert trainer.load_calls == []

    resumed = FakeTrainer()
    resume = SimpleNamespace(resume=True, load_path=None, save_folder=save_folder)
    fit_with_resume(resumed, resume, identity)
    assert resumed.fit_calls == 1
    assert resumed.load_calls == [str(save_folder)]


def test_concrete_eight_rank_loader_controller_callback_assembly(tmp_path: Path) -> None:
    loader = _loader(tmp_path / "loader", rank=0, world_size=8)
    evaluator = EDULLM_DIR / "eval_task_loss_olmo_core.py"
    task_loss = ResumeAwareTaskLossEvalCallback(
        total_steps=TOTAL_STEPS,
        save_folder=tmp_path / "checkpoints",
        run_name="assembly",
        results_dir=tmp_path / "task_loss",
        eval_script=evaluator,
        progress_dir=tmp_path / "progress",
        task_loss_nproc=8,
        production=False,
        wandb_mode="disabled",
    )
    controller = SkillItController(
        arm_id="probe",
        a_mode="probe",
        progress_dir=str(tmp_path / "progress"),
        task_loss_dir=str(tmp_path / "task_loss"),
        production=False,
        wandb_mode="disabled",
    )
    trainer = SimpleNamespace(
        data_loader=loader,
        callbacks={"task_loss": task_loss, "skillit": controller},
    )
    task_loss.trainer = trainer
    controller.trainer = trainer
    validate_trainer_assembly(trainer, loader, production=True)
    assert task_loss.task_loss_nproc == 8
    assert controller.loader is loader


def test_static_fit_and_self_contained_evaluator_contract() -> None:
    train_source = (EDULLM_DIR / "train_skillit_370m.py").read_text(encoding="utf-8")
    train_tree = ast.parse(train_source)
    functions = {
        node.name: node
        for node in train_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    fit_calls = [
        node
        for node in ast.walk(functions["fit_with_resume"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
    ]
    assert len(fit_calls) == 1
    train_calls = {
        node.func.id
        for node in ast.walk(functions["train"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "WeightedDomainDataLoader",
        "build_trainer_config",
        "validate_trainer_assembly",
        "fit_with_resume",
    } <= train_calls

    evaluator_path = EDULLM_DIR / "eval_task_loss_olmo_core.py"
    evaluator_tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    labels = None
    for node in evaluator_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_LABELS" for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            break
    assert labels == TASK_LOSS_RAW_LABELS
    assert "allenai/dolma2-tokenizer" in evaluator_path.read_text(encoding="utf-8")


def test_platform_routing_and_fixture_contract() -> None:
    base = {
        "EDULLM_DATASET_ID": "pretrain/olmo-127b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_RUN_ID": "run-id",
    }
    for arm in ARMS:
        environ = {**base, "EDULLM_WANDB_PROJECT": arm.wandb_project}
        assert platform_values(arm.index, environ)[0] == "run-id"
        command = torchrun_command(
            arm.index,
            run_id="run-id",
            wandb_entity=None,
            resume=True,
            state_root=f"/tmp/{arm.arm_id}",
        )
        assert "--nproc-per-node=8" in command
        assert command[command.index("--arm-index") + 1] == str(arm.index)
        assert "--resume" in command
        assert not any(argument.startswith("s3://") for argument in command)
    with pytest.raises(EntrypointError, match="requires W&B project"):
        platform_values(arm_by_index(0).index, {**base, "EDULLM_WANDB_PROJECT": "wrong"})

    docker = (EDULLM_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "urllib.request" not in docker
    assert "requirements-skillit-eval.txt" in docker
    assert "py_compile .edullm/eval_task_loss_olmo_core.py" in docker
    eval_requirements = (EDULLM_DIR / "requirements-skillit-eval.txt").read_text(encoding="utf-8")
    assert "090253dac6688f2532509daa7aa2eb5fae50e956" in eval_requirements
    assert "transformers==4.57.6" in eval_requirements
    for arm in ARMS:
        fixture = json.loads(
            (EDULLM_DIR / "fixtures" / f"skillit-{arm.arm_id}-submission.json").read_text(
                encoding="utf-8"
            )
        )
        assert fixture["wandb_project"] == arm.wandb_project
        assert fixture["compute_profile"] == "gpu-8xa100"
        assert fixture["workload_profile"] == "olmo-core-train-4gpu"
        assert fixture["dataset_release"] == "olmo-127b-v1"
        assert fixture["maximum_attempts"] == 1
        assert "EDULLM_CHECKPOINT_CHECK=waived" in fixture["command"][2]
        assert "EDULLM_CHECKPOINT_DIR" not in fixture["command"][2]
        assert "/opt/olmo-core/.edullm/eval_task_loss_olmo_core.py" in fixture["command"][2]

    benchmark = json.loads(
        (EDULLM_DIR / "fixtures" / "skillit-probe-benchmark-submission.json").read_text(
            encoding="utf-8"
        )
    )
    benchmark_command = benchmark["command"][2]
    assert benchmark["compute_profile"] == "gpu-8xa100"
    assert benchmark["maximum_attempts"] == 1
    assert benchmark["wandb_project"] == "skillit"
    assert "--nproc-per-node=8" in benchmark_command
    assert "--allow-local-only" in benchmark_command
    assert "--wandb-mode disabled" in benchmark_command
    assert "WANDB_API_KEY" not in benchmark_command
    assert "/opt/olmo-core/.edullm/eval_task_loss_olmo_core.py" in benchmark_command

    handoff = (EDULLM_DIR / "SKILLIT.md").read_text(encoding="utf-8")
    assert "1.25 * T" in handoff
    assert "run-approval-admin" in handoff
    assert "Submit the benchmark" in handoff
