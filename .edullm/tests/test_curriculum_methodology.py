from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

EDULLM = Path(__file__).resolve().parents[1]
if str(EDULLM) not in sys.path:
    sys.path.insert(0, str(EDULLM))

# bettermap currently imports the removed ForkProcess symbol on Windows/Python
# 3.14. None of these focused tests use its worker pool, so provide its one API
# imported by OLMo-core to keep collection independent of that upstream issue.
try:
    import bettermap  # noqa: F401
except ImportError:
    bettermap_stub = types.ModuleType("bettermap")
    bettermap_stub.ordered_map_per_thread = map  # type: ignore[attr-defined]
    sys.modules["bettermap"] = bettermap_stub

import curriculum_entrypoint as entrypoint  # noqa: E402
from curriculum_data import (  # noqa: E402
    PublishedInputError,
    validate_parent_order_binding,
)
from curriculum_ema import (  # noqa: E402
    EMA_ALPHA,
    EMA_STEPS,
    EMA_WANDB_STEP,
    ema_merge_state_dicts,
    ema_weights,
    validate_checkpoint_provenance,
    write_ema_artifact,
)
from curriculum_loader import (  # noqa: E402
    CurriculumDataError,
    CurriculumDataLoader,
    ParentChunkDataset,
    validate_complete_permutation,
)
from curriculum_pacing import (  # noqa: E402
    CURRICULUM_DATASET_ID,
    CURRICULUM_ORDER_GROUP_FOR_METRIC,
    QUADRATIC_SEGMENT_BOUNDARIES,
    SEGMENT_BOUNDARIES,
    WARMUP_LINEAR_SEGMENT_BOUNDARIES,
    curriculum_order_group,
    expanding_eligible_fraction,
    interleave_subbucket_durations,
    interleave_subbucket_index,
    pool_for_step,
    segment_index,
    split_equal_mass,
)
from production_contract import checkpoint, task_loss  # noqa: E402


def _parent(tmp_path: Path, *, shards: int = 1) -> ParentChunkDataset:
    paths = []
    for shard in range(shards):
        path = tmp_path / f"train-{shard:05d}.bin"
        if not path.exists():
            np.arange(shard * 100, shard * 100 + 17, dtype="<u4").tofile(path)
        paths.append(path)
    return ParentChunkDataset(paths, sequence_length=4, dtype="<u4")


def _loader(
    tmp_path: Path,
    *,
    pacing: str = "control",
    rank: int = 0,
    world_size: int = 1,
    order_identity: dict | None = None,
    global_batch_size: int = 8,
    parent_shards: int = 1,
) -> CurriculumDataLoader:
    parent = _parent(tmp_path, shards=parent_shards)
    ranked = None if pacing == "control" else np.arange(len(parent), dtype=np.int64)
    return CurriculumDataLoader(
        parent,
        ranked_chunk_indices=ranked,
        pacing=pacing,
        difficulty_metric=None if pacing == "control" else "compression_ratio",
        seed=42,
        total_steps=4,
        global_batch_size=global_batch_size,
        work_dir=tmp_path / f"loader-{rank}",
        parent_identity={"dataset_id": "parent", "version": "v1", "manifest": "a"},
        order_identity=(
            None
            if pacing == "control"
            else order_identity or {"dataset_id": "orders", "version": "v1", "manifest": "b"}
        ),
        pad_token_id=0,
        vocab_size=1_000,
        dp_world_size=world_size,
        dp_rank=rank,
        fs_local_rank=0,
    )


def test_recipe_is_exact_approved_nine_arm_matrix() -> None:
    arms = entrypoint.load_recipe()
    assert tuple(arm.index for arm in arms) == tuple(range(9))
    assert tuple((arm.name, arm.pacing, arm.metric, arm.order_group) for arm in arms) == (
        ("linear10-flesch", "linear_n10", "flesch", "flesch"),
        ("linear10-mtld", "linear_n10", "mtld", "mtld"),
        ("linear10-learn", "linear_n10", "learnability", "learnability"),
        ("warmup-flesch", "warmup_1000", "flesch", "flesch"),
        ("interleave-flesch", "interleave_i10_linear", "flesch", "flesch"),
        ("control", "control", None, None),
        ("quadratic10-mtld", "quadratic_n10", "mtld", "mtld"),
        ("warmup-mtld", "warmup_1000", "mtld", "mtld"),
        ("warmup-linear10-mtld", "warmup_linear_n10_1000", "mtld", "mtld"),
    )
    assert {arm.pacing for arm in arms} == {
        "linear_n10",
        "quadratic_n10",
        "warmup_1000",
        "warmup_linear_n10_1000",
        "interleave_i10_linear",
        "control",
    }
    assert all(arm.wandb_project == "curriculum-moe" for arm in arms)
    assert entrypoint.WANDB_PROJECT_NAMES == {"curriculum-moe"}
    assert CURRICULUM_DATASET_ID == "curriculum/regmix-370m"
    assert CURRICULUM_ORDER_GROUP_FOR_METRIC == {
        "compression_ratio": "compression",
        "flesch": "flesch",
        "mtld": "mtld",
        "learnability": "learnability",
    }
    assert curriculum_order_group("learnability") == "learnability"


def test_fixed_olmo2_recipe_and_checkpoint_ladder() -> None:
    assert entrypoint.TOTAL_STEPS == 2384
    assert entrypoint.SEED == 42
    assert entrypoint.LR_ALPHA_F == 0.1
    assert entrypoint.GLOBAL_BATCH_TOKENS == 4_194_304
    assert entrypoint.RANK_MICROBATCH_TOKENS == 32_768
    assert entrypoint.production_steps(None) == 2384
    steps = entrypoint.checkpoint_steps(2384)
    assert steps[0] == 0 and steps[-1] == 2384
    assert 2250 in steps and 2375 not in steps
    assert set(EMA_STEPS).issubset(steps)
    config = entrypoint.train_module_config()
    assert config.scheduler.alpha_f == 0.1
    assert config.scheduler.warmup == 24
    assert config.z_loss_multiplier == 1e-5
    assert config.max_grad_norm == 1.0
    model = entrypoint.build_model_config()
    moe = model.block.feed_forward_moe
    assert moe is not None
    assert moe.num_experts == 64
    assert moe.router.top_k == 8
    assert moe.hidden_size == 1024
    assert model.d_model == 2048
    assert model.n_layers == 16
    assert "olmoe_1B_7B" in entrypoint.MODEL_IDENTITY
    assert "num_experts=64" in entrypoint.MODEL_IDENTITY

@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0), (249, 0), (250, 1), (999, 3), (1000, 4), (2250, 9), (2383, 9)],
)
def test_zero_based_segment_boundaries(step: int, expected: int) -> None:
    assert segment_index(step) == expected
    assert SEGMENT_BOUNDARIES[-2:] == (2250, 2384)


def test_pacing_modes_match_methodology_boundaries() -> None:
    assert split_equal_mass(103)[:3] == [(0, 11), (11, 22), (22, 33)]
    assert pool_for_step(0, 1000, "linear_n10").end == 100
    assert pool_for_step(250, 1000, "linear_n10").start == 100
    assert expanding_eligible_fraction(0) == 0.25
    assert expanding_eligible_fraction(500) == 0.625
    assert pool_for_step(1000, 1000, "expanding_25_1000").end == 1000
    assert pool_for_step(999, 100, "warmup_1000").ordered
    assert not pool_for_step(1000, 100, "warmup_1000").ordered
    assert WARMUP_LINEAR_SEGMENT_BOUNDARIES == (
        0,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    )
    assert pool_for_step(0, 1000, "warmup_linear_n10_1000").end == 100
    assert pool_for_step(99, 1000, "warmup_linear_n10_1000").start == 0
    assert pool_for_step(100, 1000, "warmup_linear_n10_1000").start == 100
    assert pool_for_step(900, 1000, "warmup_linear_n10_1000").start == 900
    assert pool_for_step(999, 1000, "warmup_linear_n10_1000").start == 900
    assert not pool_for_step(999, 1000, "warmup_linear_n10_1000").ordered
    assert pool_for_step(1000, 1000, "warmup_linear_n10_1000") == pool_for_step(
        1000, 1000, "warmup_1000"
    )
    assert pool_for_step(1000, 1000, "warmup_linear_n10_1000").start == 0
    assert pool_for_step(1000, 1000, "warmup_linear_n10_1000").end == 1000
    assert not pool_for_step(1000, 1000, "warmup_linear_n10_1000").ordered
    assert interleave_subbucket_durations(250) == [25] * 10
    assert interleave_subbucket_durations(134) == [13] * 9 + [17]
    assert interleave_subbucket_index(249) == 9
    assert interleave_subbucket_index(250) == 0
    assert interleave_subbucket_index(2263) == 1
    assert QUADRATIC_SEGMENT_BOUNDARIES == (
        0,
        43,
        130,
        260,
        433,
        650,
        910,
        1213,
        1560,
        1950,
        2384,
    )
    assert sum(
        end - start
        for start, end in zip(QUADRATIC_SEGMENT_BOUNDARIES, QUADRATIC_SEGMENT_BOUNDARIES[1:])
    ) == 2384
    assert pool_for_step(0, 1000, "quadratic_n10").end == 100
    assert pool_for_step(42, 1000, "quadratic_n10").start == 0
    assert pool_for_step(43, 1000, "quadratic_n10").start == 100
    assert pool_for_step(1949, 1000, "quadratic_n10").start == 800
    assert pool_for_step(1950, 1000, "quadratic_n10").start == 900
    assert pool_for_step(2383, 1000, "quadratic_n10").start == 900


def test_parent_coordinates_are_shard_local_and_reserve_next_token(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path, shards=2)
    assert len(parent) == 8
    assert parent[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert parent[3]["input_ids"].tolist() == [12, 13, 14, 15]
    assert parent[4]["input_ids"].tolist() == [100, 101, 102, 103]


def test_control_is_flat_no_replacement_and_resume_exact(tmp_path: Path) -> None:
    loader = _loader(tmp_path)
    step0 = loader.global_indices_for_step(0).tolist()
    step1 = loader.global_indices_for_step(1).tolist()
    assert set(step0).isdisjoint(step1)
    iterator = iter(loader)
    first = next(iterator)
    assert first["input_ids"].shape == (2, 4)
    state = loader.state_dict()
    assert state["batches_processed"] == 1
    assert state["tokens_processed"] == 8

    resumed = _loader(tmp_path)
    resumed.load_state_dict(state)
    assert resumed.batches_processed == 1
    assert resumed.global_indices_for_step(1).tolist() == step1
    assert next(iter(resumed))["index"].tolist() == step1


def test_warmup_is_step_addressed_across_distributed_ranks(tmp_path: Path) -> None:
    rank0 = _loader(tmp_path, pacing="warmup_1000", rank=0, world_size=2)
    rank1 = _loader(tmp_path, pacing="warmup_1000", rank=1, world_size=2)
    assert rank0.global_indices_for_step(1).tolist() == [2, 3]
    assert rank0.batch_for_step(1)["index"].tolist() == [2]
    assert rank1.batch_for_step(1)["index"].tolist() == [3]


def test_control_reconstructs_one_deterministic_eight_rank_stream(tmp_path: Path) -> None:
    single = _loader(
        tmp_path,
        global_batch_size=32,
        parent_shards=3,
    )
    expected = single.global_indices_for_step(0).tolist()
    reconstructed = [
        _loader(
            tmp_path,
            rank=rank,
            world_size=8,
            global_batch_size=32,
            parent_shards=3,
        )
        .batch_for_step(0)["index"]
        .item()
        for rank in range(8)
    ]
    assert reconstructed == expected


def test_loader_resume_refuses_changed_order_identity(tmp_path: Path) -> None:
    loader = _loader(tmp_path, pacing="warmup_1000")
    state = loader.state_dict()
    changed = _loader(
        tmp_path,
        pacing="warmup_1000",
        order_identity={"dataset_id": "orders", "version": "v2", "manifest": "c"},
    )
    with pytest.raises(CurriculumDataError, match="changed identity"):
        changed.load_state_dict(state)


@pytest.mark.parametrize(
    "order",
    [
        np.array([0, 1, 1, 3]),
        np.array([0, 1, 2]),
        np.array([-1, 0, 1, 2]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    ],
)
def test_order_must_be_complete_parent_permutation(order: np.ndarray) -> None:
    with pytest.raises(CurriculumDataError):
        validate_complete_permutation(order, 4)


def test_order_provenance_binds_exact_parent_manifest() -> None:
    parent = {
        "groups": [
            {
                "name": "tokens",
                "profile": "pretrain-tokens/v1",
                "manifest_sha256": "a" * 64,
            }
        ]
    }
    order = {
        "groups": [
            {
                "name": "compression",
                "profile": "token-order/v1",
                "manifest_sha256": "b" * 64,
                "depends_on": [
                    {
                        "role": "token_pool",
                        "dataset_id": "pretrain/regmix-10b",
                        "version": "v1",
                        "manifest_sha256": "a" * 64,
                    }
                ],
            }
        ]
    }
    validate_parent_order_binding(
        parent_dataset=parent,
        order_dataset=order,
        parent_dataset_id="pretrain/regmix-10b",
        parent_version="v1",
        parent_manifest_sha256="a" * 64,
        order_group="compression",
    )
    order["groups"][0]["depends_on"][0]["version"] = "v2"
    with pytest.raises(PublishedInputError, match="not the staged parent"):
        validate_parent_order_binding(
            parent_dataset=parent,
            order_dataset=order,
            parent_dataset_id="pretrain/regmix-10b",
            parent_version="v1",
            parent_manifest_sha256="a" * 64,
            order_group="compression",
        )


def test_ema_is_exact_recursive_four_checkpoint_convention() -> None:
    assert EMA_STEPS == (2000, 2125, 2250, 2384)
    assert EMA_ALPHA == 0.8
    assert EMA_WANDB_STEP == 2385
    weights = ema_weights(4)
    assert weights == pytest.approx([0.512, 0.128, 0.16, 0.2])
    states = [{"w": torch.tensor([float(value)])} for value in range(4)]
    merged = ema_merge_state_dicts(states)
    assert merged["w"].item() == pytest.approx(sum(i * w for i, w in enumerate(weights)))


def test_ema_requires_one_immutable_run_identity(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    identity = {
        "family": "curriculum",
        "arm": "control",
        "ema_steps": list(EMA_STEPS),
    }
    fingerprint = checkpoint.write_run_fingerprint(root, identity)
    for step in EMA_STEPS:
        directory = root / f"step{step}"
        (directory / "model_and_optim").mkdir(parents=True)
        (directory / "model_and_optim" / ".metadata").write_bytes(b"ready")
        checkpoint.copy_fingerprint_into_checkpoint(fingerprint, directory)
    directories, common = validate_checkpoint_provenance(root, arm="control")
    assert len(directories) == 4
    output = root / "step2384-ema"
    write_ema_artifact(
        output,
        model={"w": torch.tensor([1.0])},
        fingerprint=common,
        arm="control",
    )
    assert (output / "model_eval.pt").is_file()
    payload = torch.load(output / "model_eval.pt", weights_only=False)
    assert payload["step"] == 2385
    assert payload["model"]["w"].item() == 1.0
    assert (output / "step.txt").read_text() == "2385\n"
    assert json.loads((output / "ema_manifest.json").read_text())["alpha"] == 0.8
    assert json.loads((output / "ema_manifest.json").read_text())["wandb_step"] == 2385
    assert json.loads((output / "ema_manifest.json").read_text())["source_final_step"] == 2384

    changed = checkpoint.write_run_fingerprint(tmp_path / "changed", {**identity, "seed": 7})
    checkpoint.copy_fingerprint_into_checkpoint(changed, root / "step2250")
    with pytest.raises(checkpoint.CheckpointContractError, match="one immutable"):
        validate_checkpoint_provenance(root, arm="control")


def test_platform_fixture_and_docker_are_branch_specific() -> None:
    fixture = json.loads(
        (EDULLM / "fixtures" / "curriculum-linear10-flesch-submission.json").read_text()
    )
    command = fixture["command"][-1]
    assert fixture["dataset_release"] == "regmix-10b-v1"
    assert fixture["wandb_project"] == "curriculum"
    assert fixture["compute_profile"] == "gpu-8xa100"
    assert fixture["workload_profile"] == "olmo-core-train"
    assert "--nproc-per-node=8" in command
    assert "--" in command
    assert "--nproc 8" in command
    assert "--task-loss-nproc 8" in command
    assert "EDULLM_CHECKPOINT_CHECK=waived" in command
    assert "--arm-index 0" in command and "--fresh" in command
    docker = (EDULLM / "Dockerfile").read_text()
    assert "EDULLM_EXPERIMENT_FAMILY=curriculum" in docker
    assert "curriculum_entrypoint.py" in docker

    benchmark = json.loads(
        (
            EDULLM / "fixtures" / "curriculum-linear10-flesch-benchmark-submission.json"
        ).read_text()
    )
    benchmark_command = benchmark["command"][-1]
    assert benchmark["compute_profile"] == "gpu-8xa100"
    assert benchmark["maximum_attempts"] == 1
    assert benchmark["wandb_project"] == "curriculum"
    assert "--nproc-per-node=8" in benchmark_command
    assert "--local-smoke" in benchmark_command
    assert "--wandb-mode disabled" in benchmark_command
    assert "--no-task-loss" not in benchmark_command
    assert "WANDB_API_KEY" not in benchmark_command
    assert "/opt/olmo-core/.edullm/task_loss/eval_task_loss_olmo_core.py" in benchmark_command

    handoff = (EDULLM / "CURRICULUM.md").read_text(encoding="utf-8")
    assert "1.25 * T" in handoff
    assert "run-approval-admin" in handoff
    assert "Do not submit the seven-arm matrix as fan-out" in handoff


def test_production_recipe_statically_assembles_public_olmo_apis() -> None:
    source = (EDULLM / "curriculum_entrypoint.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert {
        "build_model_config",
        "TransformerTrainModuleConfig",
        "CurriculumDataLoader",
        "TrainerConfig",
        "CheckpointerCallback",
        "ConfigSaverCallback",
        "WandBCallback",
        "trainer_config.build",
        "trainer.fit",
    } <= calls
    assert "TransformerConfig.olmoe_1B_7B" in (
        EDULLM / "curriculum_model.py"
    ).read_text(encoding="utf-8")
    assert "num_experts=64" in (EDULLM / "curriculum_model.py").read_text(encoding="utf-8")
    assert "DataParallelType.hsdp" in source
    assert "max_duration=Duration.steps(total_steps)" in source
    assert "load_trainer_state=True, load_optim_state=True" in source


def test_eight_gpu_launcher_is_concrete_and_fully_forwarded() -> None:
    fixture = json.loads((EDULLM / "platform" / "curriculum-arms.json").read_text(encoding="utf-8"))
    launcher = (EDULLM / "platform" / "entrypoint.sh").read_text(encoding="utf-8")
    args = entrypoint.parser().parse_args(["--arm-index", "0", "--fresh", "--nproc", "8"])
    command = entrypoint.torchrun_command(args)
    assert fixture["compute_profile"] == "gpu-8xa100"
    assert fixture["gpu_count"] == 8
    assert "--nproc-per-node=8" in fixture["distributed_launcher"]
    assert "NPROC=8" in launcher and "TASK_LOSS_NPROC=8" in launcher
    assert "--nproc-per-node=8" in command
    assert str(entrypoint.PACKAGED_TASK_LOSS_SCRIPT) in command
    assert "--" in command
    assert str(entrypoint.PACKAGED_LADDER_CONFIG) in command


def test_packaged_evaluator_is_exact_and_docker_complete() -> None:
    evaluator = EDULLM / "task_loss" / "eval_task_loss_olmo_core.py"
    config = EDULLM / "task_loss" / "ladder_base_config.yaml"
    tree = ast.parse(evaluator.read_text(encoding="utf-8"))
    labels = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "TASK_LABELS" for target in node.targets
        ):
            labels = ast.literal_eval(node.value)
            break
    assert labels == task_loss.TASK_LOSS_RAW_LABELS
    config_text = config.read_text(encoding="utf-8")
    assert "identifier: allenai/dolma2-tokenizer" in config_text
    assert "embedding_size: 100352" in config_text

    docker = (EDULLM / "Dockerfile").read_text(encoding="utf-8")
    assert "090253dac6688f2532509daa7aa2eb5fae50e956" in docker
    for dependency in (
        "datasets==5.0.0",
        "omegaconf==2.3.0",
        "PyYAML==6.0.3",
        "torchmetrics==1.9.0",
        "transformers==4.57.6",
    ):
        assert dependency in docker
    assert "eval_task_loss_olmo_core.py --help" in docker
    assert "LADDER_BASE_CONFIG=/opt/olmo-core/.edullm/task_loss/" in docker


def test_runtime_topology_checks_requested_eight_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(entrypoint, "get_world_size", lambda *_args: 8)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    entrypoint.assert_distributed_runtime(8)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 7)
    with pytest.raises(entrypoint.CurriculumConfigError, match="only 7 CUDA"):
        entrypoint.assert_distributed_runtime(8)


def test_run_worker_builds_concrete_eight_gpu_trainer_and_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    built_callbacks = {}
    parent = entrypoint.ResolvedInput(
        dataset_id=entrypoint.PARENT_DATASET_ID,
        version=entrypoint.PARENT_VERSION,
        group="tokens",
        profile="pretrain-tokens/v1",
        manifest_sha256=entrypoint.PARENT_MANIFEST_SHA256,
        paths=("s3://edullm-data/parent.bin",),
        numpy_dtype="<u4",
        header_bytes=0,
    )

    class FakeTrainer:
        def __init__(self, callbacks):
            self.callbacks = callbacks

        def fit(self) -> None:
            events.append("fit")

    class FakeTrainerConfig:
        def __init__(self, **_kwargs):
            self.callbacks = {}

        def with_callback(self, name, callback):
            self.callbacks[name] = callback
            return self

        def build(self, train_module, loader):
            assert train_module.dp_process_group == "eight-gpu-pg"
            assert loader == "curriculum-loader"
            assert set(self.callbacks) == {
                "checkpointer",
                "wandb",
                "config_saver",
                "curriculum_contract",
            }
            built_callbacks.update(self.callbacks)
            events.append("trainer")
            trainer = FakeTrainer(self.callbacks)
            for callback in self.callbacks.values():
                callback.trainer = trainer
            return trainer

    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "curriculum-moe")
    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setattr(
        entrypoint,
        "assert_distributed_runtime",
        lambda expected: events.append(f"topology:{expected}"),
    )
    monkeypatch.setattr(
        entrypoint,
        "resolve_and_stage",
        lambda **_kwargs: (parent, (tmp_path / "parent.bin",), None, ()),
    )
    monkeypatch.setattr(
        entrypoint,
        "ParentChunkDataset",
        lambda *_args, **_kwargs: events.append("dataset") or "parent-dataset",
    )
    monkeypatch.setattr(
        entrypoint,
        "build_train_module",
        lambda _tokens=entrypoint.RANK_MICROBATCH_TOKENS: (
            events.append("module") or types.SimpleNamespace(dp_process_group="eight-gpu-pg")
        ),
    )
    monkeypatch.setattr(entrypoint, "get_world_size", lambda _group=None: 8)
    monkeypatch.setattr(entrypoint, "get_rank", lambda _group=None: 0)
    monkeypatch.setattr(entrypoint, "get_fs_local_rank", lambda: 0)
    monkeypatch.setattr(entrypoint, "barrier", lambda: None)

    def fake_loader(*_args, **kwargs):
        assert kwargs["dp_world_size"] == 8
        assert kwargs["global_batch_size"] == entrypoint.GLOBAL_BATCH_TOKENS
        events.append("loader")
        return "curriculum-loader"

    monkeypatch.setattr(entrypoint, "CurriculumDataLoader", fake_loader)
    monkeypatch.setattr(entrypoint, "TrainerConfig", FakeTrainerConfig)

    args = entrypoint.parser().parse_args(
        [
            "--arm-index",
            "0",
            "--train-worker",
            "--nproc",
            "8",
            "--fresh",
            "--run-dir",
            str(tmp_path / "run"),
            "--wandb-mode",
            "online",
        ]
    )
    entrypoint.run_worker(args)
    assert events == [
        "topology:8",
        "dataset",
        "module",
        "loader",
        "trainer",
        "fit",
    ]
    contract = built_callbacks["curriculum_contract"]
    assert contract.task_loss_nproc == 8
    assert contract.eval_script == entrypoint.PACKAGED_TASK_LOSS_SCRIPT
    assert built_callbacks["wandb"].project == "curriculum-moe"
    assert entrypoint.TOTAL_STEPS in built_callbacks["checkpointer"].fixed_steps


def test_checkpoint_callback_requests_clean_restart_after_durable_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    callback = entrypoint.CurriculumCheckpointCallback(
        arm=entrypoint.ARMS[5],
        total_steps=250,
        save_folder=tmp_path / "checkpoints",
        progress_dir=tmp_path / "progress",
        task_loss_dir=tmp_path / "task-loss",
        eval_script=tmp_path / "eval.py",
        task_loss_nproc=8,
        production=True,
        wandb_mode="online",
        run_name="unit",
        fingerprint_path=tmp_path / "fingerprint.json",
    )
    trainer = types.SimpleNamespace(hard_stop=None)
    callback.trainer = trainer
    monkeypatch.setattr(entrypoint, "get_rank", lambda: 0)
    monkeypatch.setattr(entrypoint, "barrier", lambda: None)
    monkeypatch.setattr(
        task_loss,
        "pause_eval_reload_distributed",
        lambda *_args, **_kwargs: (object(), {"labels": {}}),
    )
    monkeypatch.setattr(checkpoint, "finalize_permanent_checkpoint", lambda **_kwargs: None)

    callback._finalize(125)

    request = json.loads(
        (tmp_path / "progress" / entrypoint.CHECKPOINT_RESTART_REQUEST).read_text(
            encoding="utf-8"
        )
    )
    assert request["durable_step"] == 125
    assert trainer.hard_stop == entrypoint.Duration.steps(125)
