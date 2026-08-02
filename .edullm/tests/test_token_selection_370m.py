from __future__ import annotations

import ast
import json
import math
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

from token_selection_370m.arms import (  # noqa: E402
    ARM_SPECS,
    REFHQ,
    REFHQ_LATE_STEPS,
)
from token_selection_370m.blade import (  # noqa: E402
    BLADE_CHECKPOINT_FORMAT,
    BLADE_SYNC_STEPS,
    BladeCallback,
    BladeSchedule,
    ResumableBatchStream,
)
from token_selection_370m.recipe import (  # noqa: E402
    ALPHA_F,
    CUSTOM_LOSS_METHODS,
    GLOBAL_BATCH_TOKENS,
    PEAK_LR,
    PRODUCTION_WORLD_SIZE,
    RANK_MICROBATCH_TOKENS,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    Z_LOSS,
    immutable_corpus_binding,
    scientific_identity,
    total_steps,
)
from token_selection_370m.selection import (  # noqa: E402
    EMAHistory,
    attention_received_from_qk,
    ema_alpha,
    selection_weights,
)


def test_exact_approved_arm_family_and_wandb_routing() -> None:
    assert tuple(
        (name, spec.method, spec.dataset_id, spec.keep_fraction) for name, spec in ARM_SPECS.items()
    ) == (
        ("rho-1", "rho_excess", "pretrain/regmix-10b", 0.6),
        ("rel-ema-exp", "rel_ema", "pretrain/regmix-10b", 0.6),
        ("middle-ppl-token", "middle_ppl", "pretrain/regmix-10b", 0.6),
        ("attention", "attention_topk", "pretrain/regmix-10b", 0.6),
        ("blade", "blade", "pretrain/regmix-10b", 0.6),
    )
    assert all(spec.wandb_project == f"token-selection-{name}" for name, spec in ARM_SPECS.items())
    assert ARM_SPECS["middle-ppl-token"].late_reference_contract.endswith(str(REFHQ_LATE_STEPS))


def test_one_recipe_constants_and_2360_step_budget() -> None:
    assert SEQUENCE_LENGTH == 2048
    assert GLOBAL_BATCH_TOKENS == 4_194_304
    assert RANK_MICROBATCH_TOKENS == 65_536
    assert PEAK_LR == 4e-4
    assert WARMUP_STEPS == 24
    assert ALPHA_F == 0.1
    assert Z_LOSS == 1e-5
    assert total_steps(9_900_000_000) == 2360


def test_method_polarities_and_per_sequence_selection() -> None:
    valid = torch.ones(2, 4, dtype=torch.bool)
    current = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    reference = torch.ones_like(current)
    rho = selection_weights(
        "rho_excess",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current,
        reference=reference,
    )
    assert rho.bool().tolist() == [
        [True, True, False, False],
        [False, False, True, True],
    ]
    rel = selection_weights(
        "rel_ema",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        current=current,
        history=reference * 3,
    )
    assert rel.sum(dim=1).tolist() == [2, 2]
    learn = selection_weights(
        "learnability",
        valid=valid,
        keep_fraction=0.5,
        step=0,
        seed=42,
        early=torch.tensor([[4.0, 3.0, 2.0, 1.0]]).expand(2, -1),
        late=torch.tensor([[1.0, 1.5, 1.5, 3.0]]).expand(2, -1),
    )
    assert learn[0].bool().tolist() == [True, True, False, False]
    blade = selection_weights(
        "blade",
        valid=valid,
        keep_fraction=0.5,
        step=500,
        seed=42,
        current=current,
        reference=reference * 3,
    )
    assert blade.sum(dim=1).tolist() == [2, 2]


def test_middle_ppl_drops_easy_and_hard_and_random_is_resumable() -> None:
    valid = torch.ones(1, 10, dtype=torch.bool)
    middle = selection_weights(
        "middle_ppl",
        valid=valid,
        keep_fraction=0.6,
        step=0,
        seed=42,
        reference=torch.arange(10.0).unsqueeze(0),
    )
    assert middle.bool().tolist() == [
        [False, False, True, True, True, True, True, True, False, False]
    ]
    first = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    resumed = selection_weights("random", valid=valid, keep_fraction=0.6, step=125, seed=42)
    assert torch.equal(first, resumed)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.0, 0.0]))


def test_relative_ema_variants_and_resume_state() -> None:
    assert ema_alpha(0, tau=300, constant=None) == 0.0
    assert ema_alpha(300, tau=300, constant=None) == pytest.approx(1 - math.exp(-1))
    assert ema_alpha(100, tau=None, constant=0.9985) == 0.9985

    model = Tiny()
    ema = EMAHistory(model)
    model.weight.data.fill_(2)
    ema.update(model, 0.5)
    model.weight.data.fill_(4)
    ema.update(model, 0.5)
    state = ema.state_dict()
    restored = EMAHistory(Tiny())
    restored.load_state_dict(state)
    assert restored.correction == ema.correction
    assert torch.equal(restored.shadow["weight"], ema.shadow["weight"])

    seeded = EMAHistory(Tiny(), seed={"weight": torch.tensor([3.0, 5.0])})
    assert seeded.correction == 1.0
    with seeded.swap_to(model):
        assert torch.equal(model.weight, torch.tensor([3.0, 5.0]))
    assert torch.equal(model.weight, torch.tensor([4.0, 4.0]))


def test_attention_received_matches_causal_definition() -> None:
    query = torch.zeros(1, 3, 1, 2)
    key = torch.zeros_like(query)
    # Uniform causal attention: received mass is 1 + 1/2 + 1/3, 1/2 + 1/3, 1/3.
    scores = attention_received_from_qk(query, key)
    assert scores[0].tolist() == pytest.approx([11 / 6, 5 / 6, 1 / 3])


class FakeStream:
    def __init__(self, cursor: int):
        self.cursor = cursor

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


def _blade_callback(train_cursor=3, hq_cursor=7) -> BladeCallback:
    return BladeCallback(
        total_steps=2360,
        reference_factory=Tiny,
        reference_train_stream=FakeStream(train_cursor),  # type: ignore[arg-type]
        refhq_stream=FakeStream(hq_cursor),  # type: ignore[arg-type]
    )


def test_blade_locked_schedule_and_full_resume_state() -> None:
    assert BLADE_SYNC_STEPS == (500, 875, 1250, 1625, 2000)
    callback = _blade_callback()
    callback._new_reference()
    assert callback.reference is not None and callback.reference_optim is not None
    callback.reference.weight.data.fill_(9)
    callback.completed_step = 1250
    callback.last_sync = 1250
    state = callback.state_dict()
    assert state["checkpoint_format"] == BLADE_CHECKPOINT_FORMAT
    assert state["dynamic_reference_optim"] is not None
    assert state["reference_train_stream"] == {"cursor": 3}
    assert state["refhq_stream"] == {"cursor": 7}

    restored = _blade_callback(0, 0)
    restored._restore(state)
    assert restored.completed_step == 1250
    assert restored.last_sync == 1250
    assert restored.reference is not None
    assert torch.equal(restored.reference.weight, torch.tensor([9.0, 9.0]))
    assert restored.reference_train_stream.cursor == 3
    assert restored.refhq_stream.cursor == 7
    assert next(step for step in BLADE_SYNC_STEPS if step > restored.completed_step) == 1625


def test_blade_rejects_schedule_drift_and_missing_post_warmup_reference() -> None:
    with pytest.raises(ValueError, match="locked"):
        BladeSchedule(k_steps=74).validate(2360)
    state = _blade_callback().state_dict()
    state["completed_step"] = 500
    state["last_sync"] = 500
    with pytest.raises(ValueError, match="missing"):
        _blade_callback()._restore(state)
    source = _blade_callback()
    source._new_reference()
    inconsistent = source.state_dict()
    inconsistent["completed_step"] = 1250
    inconsistent["last_sync"] = 875
    with pytest.raises(ValueError, match="last sync"):
        _blade_callback()._restore(inconsistent)


class FakeResumableLoader:
    def __init__(self, cursor: int = 0) -> None:
        self.cursor = cursor
        self.epoch = 0

    def reshuffle(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        value = self.cursor
        self.cursor += 1
        return {"cursor": value}

    def state_dict(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]


def test_blade_secondary_stream_resume_is_next_batch_exact() -> None:
    original = ResumableBatchStream(FakeResumableLoader())
    assert original.next() == {"cursor": 0}
    state = original.state_dict()
    expected = original.next()
    resumed = ResumableBatchStream(FakeResumableLoader())
    resumed.load_state_dict(state)
    assert resumed.next() == expected


def test_identity_pins_reference_provenance_and_fixture(tmp_path: Path) -> None:
    arm = ARM_SPECS["rho-1"]
    reference = tmp_path / "refhq-step1315.pt"
    reference.write_bytes(b"immutable reference")
    corpus = types.SimpleNamespace(
        version="v1",
        paths=("s3://edullm-data/pretrain/regmix-10b/v1/train-00000.bin",),
        dtype="<u4",
        rows=9_900_000_000,
    )
    binding = immutable_corpus_binding(arm.dataset_id, corpus)
    identity = scientific_identity(
        arm,
        dataset_binding=binding,
        refhq_binding=None,
        max_tokens=9_900_000_000,
        reference_path=str(reference),
        early_reference_path=None,
        late_reference_path=None,
    )
    assert identity["reference_contract"] == arm.reference_contract
    assert len(identity["reference_sha256"]) == 64
    assert identity["dataset_binding"] == binding
    assert len(identity["dataset_binding"]["paths_sha256"]) == 64
    assert identity["wandb_project"] == "token-selection-rho-1"
    fixture = json.loads(
        (EDULLM_ROOT / "platform" / "token-selection-arms.json").read_text(encoding="utf-8")
    )
    assert set(fixture["arms"]) == set(ARM_SPECS)
    assert (
        fixture["arms"]["blade"]["secondary_dataset_release"] == REFHQ
    )


def test_immutable_bindings_fail_closed_for_latest_and_missing_blade_refhq() -> None:
    unresolved = types.SimpleNamespace(
        version="latest",
        paths=("s3://example/train.bin",),
        dtype="<u4",
        rows=1,
    )
    with pytest.raises(ValueError, match="immutable version"):
        immutable_corpus_binding("pretrain/regmix-10b", unresolved)

    resolved = types.SimpleNamespace(
        version="v1",
        paths=("s3://example/train.bin",),
        dtype="<u4",
        rows=1,
    )
    with pytest.raises(ValueError, match="RefHQ binding"):
        scientific_identity(
            ARM_SPECS["blade"],
            dataset_binding=immutable_corpus_binding("pretrain/regmix-10b", resolved),
            refhq_binding=None,
            max_tokens=9_900_000_000,
            reference_path=None,
            early_reference_path=None,
            late_reference_path=None,
        )


def test_production_recipe_statically_assembles_public_olmo_apis() -> None:
    source = (EDULLM_ROOT / "token_selection_370m" / "recipe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {
        "TransformerConfig.olmo2_370M",
        "NumpyFSLDatasetConfig",
        "NumpyDataLoaderConfig",
        "TrainerConfig",
        "CheckpointerCallback",
        "GPUMemoryMonitorCallback",
        "ConfigSaverCallback",
        "WandBCallback",
        "TaskLossEvalCallback",
        "trainer_config.build",
    } <= calls
    assert "DataParallelType.hsdp" in source
    assert "LoadStrategy.if_available if resume else LoadStrategy.never" in source
    assert "module_config.build(model)" in source
    assert "_custom_module(model, module_config, selection_config)" in source
    assert 'checkpoint_kwargs["fixed_steps"]' in source
    assert "task_loss_nproc=PRODUCTION_WORLD_SIZE if production else None" in source
    assert PRODUCTION_WORLD_SIZE == 8
    assert {spec.method for spec in ARM_SPECS.values()} - CUSTOM_LOSS_METHODS == {"blade"}
    assert CUSTOM_LOSS_METHODS >= {
        "rho_excess",
        "rel_ema",
        "middle_ppl",
        "attention_topk",
    }


def test_platform_entrypoint_is_locked_to_eight_gpu_torchrun() -> None:
    launcher = (EDULLM_ROOT / "platform" / "entrypoint.sh").read_text(encoding="utf-8")
    fixture = json.loads(
        (EDULLM_ROOT / "platform" / "token-selection-arms.json").read_text(encoding="utf-8")
    )
    assert "python -m torch.distributed.run" in launcher
    assert "--nproc_per_node=8" in launcher
    assert fixture["compute_profile"] == "gpu-8xa100"
    assert fixture["gpu_count"] == 8
    assert fixture["entrypoint"] == "bash .edullm/platform/entrypoint.sh"
    submission = json.loads(
        (EDULLM_ROOT / "fixtures" / "token-selection-attention-submission.json").read_text(
            encoding="utf-8"
        )
    )
    command = submission["command"][-1]
    assert submission["compute_profile"] == "gpu-8xa100"
    assert submission["workload_profile"] == "olmo-core-train-4gpu"
    assert submission["dataset_release"] == "regmix-10b-v1"
    assert submission["wandb_project"] == "token-selection-attention"
    assert "--nproc-per-node=8" in command
    assert "EDULLM_CHECKPOINT_CHECK=waived" in command

    benchmark = json.loads(
        (
            EDULLM_ROOT / "fixtures" / "token-selection-attention-benchmark-submission.json"
        ).read_text(encoding="utf-8")
    )
    benchmark_command = benchmark["command"][-1]
    assert benchmark["compute_profile"] == "gpu-8xa100"
    assert benchmark["maximum_attempts"] == 1
    assert benchmark["wandb_project"] == "token-selection-attention"
    assert "--nproc-per-node=8" in benchmark_command
    assert "--arm attention" in benchmark_command
    assert "--local" in benchmark_command
    assert "WANDB_MODE=disabled" in benchmark_command
    assert "WANDB_API_KEY" not in benchmark_command
    assert "/opt/olmo-core/.edullm/eval_task_loss_olmo_core.py" in benchmark_command

    handoff = (EDULLM_ROOT / "README-token-selection.md").read_text(encoding="utf-8")
    assert "1.25 * T" in handoff
    assert "run-approval-admin" in handoff
    assert "submit in table-index order" in handoff
    assert (
        str(EDULLM_ROOT / "eval_task_loss_olmo_core.py")
        .replace("\\", "/")
        .endswith(".edullm/eval_task_loss_olmo_core.py")
    )


def test_packaged_evaluator_and_image_are_complete() -> None:
    evaluator = EDULLM_ROOT / "eval_task_loss_olmo_core.py"
    tree = ast.parse(evaluator.read_text(encoding="utf-8"))
    labels = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_LABELS" for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            break
    from production_contract.task_loss import TASK_LOSS_RAW_LABELS

    assert labels == TASK_LOSS_RAW_LABELS
    docker = (EDULLM_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-token-selection-eval.txt" in docker
    assert "py_compile .edullm/eval_task_loss_olmo_core.py" in docker
    requirements = (EDULLM_ROOT / "requirements-token-selection-eval.txt").read_text(
        encoding="utf-8"
    )
    assert "090253dac6688f2532509daa7aa2eb5fae50e956" in requirements
    assert "transformers==4.57.6" in requirements


def test_entrypoint_builds_and_fits_production_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import token_selection_entrypoint as entrypoint

    task_script = tmp_path / "task-loss.py"
    task_script.write_text("# fixture\n", encoding="utf-8")
    events: list[str] = []

    class Corpus:
        version = "immutable-v1"
        rows = 9_900_000_000
        paths = ("s3://edullm-data/pretrain/regmix-10b/v1/train-00000.bin",)
        dtype = "<u4"

    class Trainer:
        def fit(self) -> None:
            events.append("fit")

    def fake_build(*args, **kwargs):
        assert args[0] is ARM_SPECS["attention"]
        assert kwargs["production"] is True
        assert kwargs["resume"] is False
        events.append("build")
        return Trainer()

    fake_olmo = types.ModuleType("olmo_core")
    fake_train = types.ModuleType("olmo_core.train")
    fake_utils = types.ModuleType("olmo_core.utils")
    fake_train.prepare_training_environment = lambda **kwargs: events.append("prepare")
    fake_train.teardown_training_environment = lambda: events.append("teardown")
    fake_utils.seed_all = lambda seed: events.append(f"seed:{seed}")
    monkeypatch.setitem(sys.modules, "olmo_core", fake_olmo)
    monkeypatch.setitem(sys.modules, "olmo_core.train", fake_train)
    monkeypatch.setitem(sys.modules, "olmo_core.utils", fake_utils)
    monkeypatch.setattr(entrypoint, "resolve_corpus", lambda **kwargs: Corpus())
    monkeypatch.setattr(entrypoint, "build_trainer", fake_build)
    monkeypatch.setattr(entrypoint, "write_identity", lambda *args: events.append("identity"))
    monkeypatch.setattr(entrypoint, "assert_production_runtime", lambda: events.append("world:8"))
    monkeypatch.setenv("EDULLM_DATASET_VERSION", "v1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "token_selection_entrypoint.py",
            "--arm",
            "attention",
            "--save-folder",
            str(tmp_path / "save"),
            "--work-dir",
            str(tmp_path / "work"),
            "--progress-dir",
            str(tmp_path / "progress"),
            "--task-loss-script",
            str(task_script),
        ],
    )

    entrypoint.main()

    assert events == [
        "prepare",
        "world:8",
        "seed:6198",
        "build",
        "identity",
        "fit",
        "teardown",
    ]
