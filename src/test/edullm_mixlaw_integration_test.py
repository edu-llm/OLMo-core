"""Focused contracts for the minimal MixLaw OLMo2-370M integration."""

from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EDULLM_DIR = REPO_ROOT / ".edullm"
ENTRYPOINT = EDULLM_DIR / "mixlaw_entrypoint.py"
sys.path.insert(0, str(REPO_ROOT / "src"))

# bettermap currently imports POSIX-only multiprocessing names even when merely
# importing OLMo-core on Windows. These tests never execute bettermap.
if sys.platform == "win32" and not hasattr(mp.context, "ForkProcess"):
    mp.context.ForkProcess = mp.context.SpawnProcess  # type: ignore[attr-defined]
    _get_context = mp.get_context
    mp.get_context = lambda method=None: _get_context("spawn" if method == "fork" else method)


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("olmo_core_mixlaw_entrypoint", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mixlaw = _load_entrypoint()
import mixlaw_wandb_policy as wandb_policy  # noqa: E402

EXPECTED_ARMS = [
    (
        "olmo-mix-1124",
        [
            0.951142141766451,
            0.005346961229389778,
            0.02133643182881498,
            0.015064035002030816,
            0.003136198413392081,
            0.0030333722359038163,
            0.0009408595240176244,
        ],
    ),
    ("mix07", [0.6, 0.16, 0.09, 0.06, 0.0406, 0.0394, 0.01]),
    ("mix18", [0.2375, 0.0406, 0.0406, 0.425, 0.1, 0.0437, 0.1125]),
    (
        "ML-pilot_caps",
        [
            0.5680183567101345,
            2.993279038690257e-17,
            4.115711244784343e-18,
            0.09666474469552455,
            0.035316898594340805,
            6.567930568428362e-17,
            0.3,
        ],
    ),
    (
        "ML-near-opt-4",
        [
            0.4402420723961225,
            0.011898187849676986,
            0.011898187849676986,
            0.0426758091734663,
            0.1813875548299026,
            0.011898187849676986,
            0.3,
        ],
    ),
    (
        "LGB-min1pct",
        [
            0.5528505096102141,
            0.21178482308348515,
            0.08723935826031831,
            0.08163337756774411,
            0.041786239404329344,
            0.013571059505826967,
            0.01113463256808213,
        ],
    ),
    (
        "LGB-near-opt-8",
        [
            0.1815026687610594,
            0.18152022799030812,
            0.07612101791112263,
            0.3774107813752986,
            0.019706663260932462,
            0.03721258967688889,
            0.12652605102438982,
        ],
    ),
]


def _sources(tokens: int = 100_000_000_000):
    return tuple(
        mixlaw.DomainSource(
            name=domain, paths=(f"s3://edullm-data/{domain}.bin",), available_tokens=tokens
        )
        for domain in mixlaw.DOMAINS
    )


def _environment() -> dict[str, str]:
    return {
        "EDULLM_DATASET_ID": "pretrain/olmo-127b",
        "EDULLM_DATASET_VERSION": "v1",
        "EDULLM_CHECKPOINT_DIR": "s3://outputs/checkpoints/",
        "EDULLM_WANDB_PROJECT": "mixlaw",
        "EDULLM_RUN_ID": "run-123",
        "WANDB_RUN_GROUP": "mixlaw-validation",
    }


def _config(index: int, *, length_tokens: int | None = None):
    return mixlaw.build_experiment_config(
        index,
        _sources(),
        save_folder="s3://outputs/checkpoints/",
        length_tokens=length_tokens,
        environ=_environment(),
    )


def test_recipe_has_all_seven_exact_arms_and_excludes_mix01() -> None:
    payload = json.loads((EDULLM_DIR / "mixlaw_recipe.json").read_text(encoding="utf-8"))
    assert payload["data_source"] == {
        "dataset_id": "pretrain/olmo-127b",
        "version": "v1",
        "label_key": "source",
    }
    assert payload["budget_tokens"] == 10_000_000_000
    assert [(arm.name, list(arm.weights)) for arm in mixlaw.ARMS] == EXPECTED_ARMS
    assert [arm.mixture_id for arm in mixlaw.ARMS] == [0, 7, 18, 25, 26, 27, 28]
    assert "mix01" not in {arm.name for arm in mixlaw.ARMS}


def test_only_source_target_ratios_differ_across_arm_configs() -> None:
    configs = [_config(index).as_config_dict() for index in range(7)]
    observed = []
    normalized = []
    for config in configs:
        candidate = copy.deepcopy(config)
        sources = candidate["dataset"]["source_mixture_config"]["source_list"]["sources"]
        observed.append([source["target_ratio"] for source in sources])
        for source in sources:
            source["target_ratio"] = "<arm-weight>"
        normalized.append(candidate)
    assert observed == [list(mixlaw.normalized_weights(arm)) for arm in mixlaw.ARMS]
    assert all(candidate == normalized[0] for candidate in normalized[1:])


def test_exact_common_training_hyperparameters() -> None:
    config = _config(0)
    assert config.model.d_model == 1024
    assert config.model.n_layers == 16
    assert config.model.block.sequence_mixer.n_heads == 16
    assert config.dataset.sequence_length == 2_048
    assert str(config.dataset.dtype) == "uint32"
    assert config.data_loader.global_batch_size == 4_194_304
    assert config.data_loader.seed == 12_536
    assert config.train_module.rank_microbatch_size == 65_536
    assert config.train_module.max_sequence_length == 2_048
    assert config.train_module.compile_model is True
    assert str(config.train_module.dp_config.name) == "hsdp"
    assert str(config.train_module.dp_config.param_dtype) == "bfloat16"
    assert str(config.train_module.dp_config.reduce_dtype) == "float32"
    assert config.train_module.z_loss_multiplier == 1e-5
    assert config.train_module.max_grad_norm == 1.0
    optim = config.train_module.optim
    assert optim.__class__.__name__ == "SkipStepAdamWConfig"
    assert optim.lr == 4e-4
    assert optim.betas == (0.9, 0.95)
    assert optim.weight_decay == 0.1
    assert optim.group_overrides[0].params == ["embeddings.weight"]
    assert optim.group_overrides[0].opts == {"weight_decay": 0.0}
    assert config.train_module.scheduler.warmup == 24
    assert config.train_module.scheduler.alpha_f == 0.1
    assert config.init_seed == 12_536
    assert mixlaw.PRODUCTION_STEPS == 2_384
    assert config.trainer.max_duration.value == 2_384
    assert (
        config.dataset.source_mixture_config.requested_tokens == 2_384 * 4_194_304 == 9_999_220_736
    )


def test_standard_source_mixture_and_recipe_wide_repetition_bounds() -> None:
    constrained = list(_sources())
    constrained[0] = mixlaw.DomainSource(
        name="dclm",
        paths=("s3://edullm-data/dclm.bin",),
        available_tokens=2_000_000_000,
    )
    bounds = mixlaw.repetition_bounds(constrained)
    assert bounds["dclm"] == 5
    assert all(bounds[domain] == 1 for domain in mixlaw.DOMAINS[1:])
    configs = [
        mixlaw.build_experiment_config(
            index,
            constrained,
            save_folder="s3://outputs/checkpoints/",
            environ=_environment(),
        )
        for index in range(7)
    ]
    for config in configs:
        src_mix = config.dataset.source_mixture_config
        assert src_mix.__class__.__name__ == "SourceMixtureDatasetConfig"
        assert [source.max_repetition_ratio for source in src_mix.source_list.sources] == [
            5,
            1,
            1,
            1,
            1,
            1,
            1,
        ]


def test_dataset_resolution_is_v1_labeled_and_format_checked(monkeypatch) -> None:
    calls = []

    class Read:
        paths = ["s3://edullm-data/shard.bin"]
        dtype = "uint32"
        byte_order = "little"
        header_bytes = 0
        rows = 1_000_000

    class FakeS3:
        @staticmethod
        def default():
            return object()

    def fake_dataset_paths(dataset_id, version, **kwargs):
        calls.append((dataset_id, version, kwargs))
        return Read()

    read_module = SimpleNamespace(dataset_paths=fake_dataset_paths)
    s3_module = SimpleNamespace(Boto3S3=FakeS3)
    monkeypatch.setitem(sys.modules, "edullm_data.read", read_module)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", s3_module)
    sources = mixlaw.resolve_domain_sources()
    assert [source.name for source in sources] == list(mixlaw.DOMAINS)
    assert [
        (dataset_id, version, kwargs["split"], kwargs["labels"])
        for dataset_id, version, kwargs in calls
    ] == [("pretrain/olmo-127b", "v1", "train", {"source": domain}) for domain in mixlaw.DOMAINS]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dtype", "uint16"),
        ("byte_order", "big"),
        ("header_bytes", 128),
    ],
)
def test_dataset_resolution_rejects_unsafe_format(monkeypatch, field: str, value) -> None:
    read = SimpleNamespace(
        paths=["s3://edullm-data/shard.bin"],
        dtype="uint32",
        byte_order="little",
        header_bytes=0,
        rows=1_000_000,
    )
    setattr(read, field, value)
    monkeypatch.setitem(
        sys.modules,
        "edullm_data.read",
        SimpleNamespace(dataset_paths=lambda *args, **kwargs: read),
    )
    monkeypatch.setitem(
        sys.modules,
        "edullm_data.s3",
        SimpleNamespace(Boto3S3=SimpleNamespace(default=lambda: object())),
    )
    with pytest.raises(mixlaw.MixLawConfigError):
        mixlaw.resolve_domain_sources()


def test_checkpoint_directory_is_direct_and_resume_is_explicit(monkeypatch) -> None:
    events = []

    class FakeTrainer:
        def __init__(self):
            config_saver = mixlaw.ConfigSaverCallback()
            self.callbacks = {"config_saver": config_saver}
            config_saver.trainer = self

        def maybe_load_checkpoint(self):
            events.append("resume")

        def fit(self):
            events.append("fit")

    trainer = FakeTrainer()
    train_module = SimpleNamespace(dp_process_group=object())
    config = SimpleNamespace(
        init_seed=12_536,
        model=SimpleNamespace(build=lambda **kwargs: object()),
        train_module=SimpleNamespace(build=lambda model: train_module),
        dataset=SimpleNamespace(build=lambda: object()),
        data_loader=SimpleNamespace(build=lambda dataset, **kwargs: object()),
        trainer=SimpleNamespace(build=lambda module, loader: trainer),
        as_config_dict=lambda: {"trainer": {"save_folder": "s3://outputs/checkpoints/"}},
    )
    monkeypatch.setattr(
        mixlaw, "prepare_training_environment", lambda **kwargs: events.append("prepare")
    )
    monkeypatch.setattr(mixlaw, "teardown_training_environment", lambda: events.append("teardown"))
    mixlaw.run_training(config)
    assert events == ["prepare", "resume", "fit", "teardown"]
    built = _config(0)
    assert built.trainer.save_folder == "s3://outputs/checkpoints/"
    checkpointer = built.trainer.callbacks["checkpointer"]
    assert checkpointer.pre_train_checkpoint is True
    assert checkpointer.save_interval == 125
    assert checkpointer.max_checkpoints is None
    evaluator = built.trainer.callbacks["task_loss_eval"]
    assert evaluator.total_steps == mixlaw.PRODUCTION_STEPS
    assert evaluator.interval == 125
    assert evaluator.nproc == 8


def test_eval_schedule_matches_every_mixlaw_checkpoint() -> None:
    assert wandb_policy.checkpoint_step(0, 2_384, 125)
    assert wandb_policy.checkpoint_step(2_375, 2_384, 125)
    assert wandb_policy.checkpoint_step(2_384, 2_384, 125)
    assert not wandb_policy.checkpoint_step(2_376, 2_384, 125)


def test_every_checkpoint_is_evaluated_but_only_final_checkpoint_is_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, int]] = []
    callback = wandb_policy.MixLawWandBEvalCallback(
        arm="mix07",
        total_steps=250,
        save_folder=str(tmp_path / "checkpoints"),
        run_name="unit-mix07",
        work_dir=tmp_path / "eval-work",
        eval_script=tmp_path / "eval.py",
        interval=125,
        nproc=8,
    )
    callback.trainer = SimpleNamespace(callbacks={})
    monkeypatch.setattr(wandb_policy, "_wandb_run", lambda _trainer: object())
    monkeypatch.setattr(callback, "_wait_for_checkpoint", lambda _source: None)
    monkeypatch.setattr(
        callback,
        "_stage_checkpoint",
        lambda _source, target: target.mkdir(parents=True),
    )
    monkeypatch.setattr(
        callback,
        "_evaluate",
        lambda _checkpoint, _output, step: {"labels": {label: 1.0 for label in wandb_policy.TASK_LABELS}},
    )
    monkeypatch.setattr(
        wandb_policy,
        "upload_eval",
        lambda _run, _payload, _path, step: events.append(("eval", step)),
    )
    monkeypatch.setattr(
        wandb_policy,
        "upload_final_checkpoint",
        lambda _run, _checkpoint, *, step, **_kwargs: events.append(("checkpoint", step)),
    )

    callback._finalize_rank_zero(125)
    callback._finalize_rank_zero(250)

    assert events == [("eval", 125), ("eval", 250), ("checkpoint", 250)]


def test_benchmark_is_exactly_100_steps_and_invalid_lengths_are_refused() -> None:
    benchmark_tokens = 100 * mixlaw.GLOBAL_BATCH_TOKENS
    assert benchmark_tokens == 419_430_400
    assert _config(1, length_tokens=benchmark_tokens).trainer.max_duration.value == 100
    with pytest.raises(SystemExit):
        mixlaw.parser().parse_args(["--arm-index", "1", "--length-tokens", "419430401"])
    with pytest.raises(SystemExit):
        mixlaw.parser().parse_args(["--arm-index", "7"])


def test_torchrun_uses_exactly_eight_ranks() -> None:
    command = mixlaw.torchrun_command(6, None)
    assert command[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert "--nproc-per-node=8" in command
    assert command[-2:] == ["--arm-index", "6"]


def test_worker_wandb_name_includes_selected_mixture(monkeypatch) -> None:
    seen = {}
    for key, value in {**_environment(), "WORLD_SIZE": "8"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        mixlaw,
        "run_training",
        lambda config: seen.update(name=mixlaw.os.environ["WANDB_NAME"], config=config),
    )
    assert mixlaw.main(["--train-worker", "--arm-index", "4"], resolver=_sources) == 0
    assert seen["name"] == "run-123-ML-near-opt-4"
    assert seen["config"].trainer.callbacks["wandb"].name == seen["name"]
    assert seen["config"].trainer.callbacks["wandb"].project == "mixlaw"
    assert seen["config"].trainer.callbacks["wandb"].group == "mixlaw-validation"


def test_platform_fixtures_are_sequential_without_fanout() -> None:
    for name in (
        "mixlaw-submission.json",
        "mixlaw-benchmark-submission.json",
        "mixlaw-benchmark-olmo-mix-1124-submission.json",
    ):
        fixture = json.loads((EDULLM_DIR / "fixtures" / name).read_text(encoding="utf-8"))
        command = fixture["command"][2]
        assert fixture["dataset_release"] == "olmo-127b-v1"
        assert fixture["compute_profile"] == "gpu-8xa100"
        assert all(
            key not in fixture
            for key in ("fanout_size", "fanout_parallelism", "fanout_index_parameter")
        )
        assert "--nproc-per-node=8" in command
        assert "--train-worker" in command
        assert '"$EDULLM_CHECKPOINT_DIR"' in command
        assert "EDULLM_LAUNCH_CHECK=waived" not in command
    benchmark = json.loads(
        (EDULLM_DIR / "fixtures" / "mixlaw-benchmark-submission.json").read_text(encoding="utf-8")
    )
    arm0_benchmark = json.loads(
        (EDULLM_DIR / "fixtures" / "mixlaw-benchmark-olmo-mix-1124-submission.json").read_text(
            encoding="utf-8"
        )
    )
    production = json.loads(
        (EDULLM_DIR / "fixtures" / "mixlaw-submission.json").read_text(encoding="utf-8")
    )
    assert benchmark["maximum_attempts"] == 1
    assert arm0_benchmark["maximum_attempts"] == 1
    assert production["maximum_attempts"] == 2
    assert "--length-tokens 419430400" in benchmark["command"][2]
    assert "--arm-index 1" in benchmark["command"][2]
    assert "--arm-index 0" in arm0_benchmark["command"][2]
    assert "--length-tokens 419430400" in arm0_benchmark["command"][2]
    assert "--arm-index 0" in production["command"][2]


def test_dockerfile_preserves_existing_olmo_core_training_image() -> None:
    dockerfile = (EDULLM_DIR / "Dockerfile").read_text(encoding="utf-8")
    existing_trainer = (EDULLM_DIR / "train_on_corpus.py").read_text(encoding="utf-8")
    assert "ARG BASE_IMAGE\n" in dockerfile
    assert "FROM ${BASE_IMAGE}" in dockerfile
    assert "COPY . ." in dockerfile
    assert '"torch==2.9.0"' in dockerfile
    assert 'python -m pip install --no-cache-dir ".[wandb]" boto3' in dockerfile
    assert "38bf831a6c3f445e394784018441fd59288b876c" in dockerfile
    assert "flash-attn" not in dockerfile.lower()
    assert "requirements-mixlaw-eval.txt" in dockerfile
    assert "090253dac6688f2532509daa7aa2eb5fae50e956" in (
        EDULLM_DIR / "requirements-mixlaw-eval.txt"
    ).read_text(encoding="utf-8")
    assert "vendor/" not in dockerfile
    assert not any(line.startswith("CMD ") for line in dockerfile.splitlines())
    assert 'default=os.environ.get("EDULLM_CHECKPOINT_DIR", "")' in existing_trainer
