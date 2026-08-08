import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
ENTRYPOINT = ROOT / ".edullm/model_arch_tests.py"
TRAIN_RUNNER = ROOT / ".edullm/train_core6_arm.py"
DOCKERFILE = ROOT / ".edullm/Dockerfile"
RUN_CONFIG = ROOT / ".edullm/run.yaml"
COMPARISON_RUN_CONFIG = ROOT / ".edullm/run-comparison.yaml"
SEED_SCHEDULE = ROOT / "docs/mamba-comparison/seeds.json"
RUN_GUIDE = ROOT / "MODEL_ARCH_RUNS.md"


def load_entrypoint():
    assert ENTRYPOINT.is_file(), "the platform model-architecture entrypoint is missing"
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("edullm_model_arch_tests", ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_platform_artifacts_exist():
    assert ENTRYPOINT.is_file()
    assert TRAIN_RUNNER.is_file()
    assert DOCKERFILE.is_file()
    assert RUN_CONFIG.is_file()
    assert RUN_GUIDE.is_file()


def test_four_arm_geometry_has_identical_attention_and_full_recurrent_treatments():
    module = load_entrypoint()
    from olmo_core.nn.attention import AttentionBackendName, AttentionConfig

    configs = {
        arm: module.build_model_config(arm, module.valid_init_seeds(arm)[0]) for arm in module.ARMS
    }
    assert module.ATTENTION_LAYERS == (3, 7, 11, 15)
    assert module.RECURRENT_LAYERS == (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14)

    for arm, config in configs.items():
        assert config.d_model == 1024
        assert config.vocab_size == 100352
        assert config.n_layers == 16
        assert config.tie_word_embeddings is True
        blocks = config.resolved_block_configs
        width_by_layer = dict(zip(module.RECURRENT_LAYERS, module.solve_widths(arm)))
        for index, block in enumerate(blocks):
            mixer = block.sequence_mixer
            if index in module.ATTENTION_LAYERS:
                assert isinstance(mixer, AttentionConfig)
                assert (mixer.n_heads, mixer.n_kv_heads, mixer.head_dim) == (16, 8, 64)
                assert mixer.backend == AttentionBackendName.torch
                assert mixer.rope is not None
            assert block.feed_forward is not None
            if index in module.ATTENTION_LAYERS:
                assert block.feed_forward.hidden_size == 4608
            else:
                assert block.feed_forward.hidden_size == width_by_layer[index]

    reference = configs[module.ARMS[0]]
    for arm, config in configs.items():
        for index, (expected, actual) in enumerate(
            zip(reference.resolved_block_configs, config.resolved_block_configs)
        ):
            if index in module.ATTENTION_LAYERS:
                assert actual.as_config_dict() == expected.as_config_dict(), (arm, index)


def test_treatment_mixers_are_strict_and_parameter_matched():
    module = load_entrypoint()
    from olmo_core.nn.flash_pd_native import (
        NativeFlashPDMamba3SISOMixerConfig,
        NativeFlashPDMixerConfig,
        NativePDBackend,
        NativePDMode,
    )
    from olmo_core.nn.mamba3 import Mamba3MixerConfig

    expected_counts = {
        "mamba-b3": 390_148_736,
        "xlstm": 390_143_056,
        "mamba3-siso-pd": 390_169_664,
        "native-pd": 390_142_976,
    }
    expected_widths = {
        "mamba-b3": (4800,) * 7 + (4768,) * 5,
        "xlstm": (4672,) * 8 + (4640,) * 4,
        "mamba3-siso-pd": (2752,) * 11 + (2720,),
        "native-pd": (2432,) * 6 + (2400,) * 6,
    }
    assert module.EXACT_PARAMETER_COUNTS == expected_counts

    for arm in module.ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        mixers = [
            config.resolved_block_configs[index].sequence_mixer for index in module.RECURRENT_LAYERS
        ]
        assert module.solve_widths(arm) == expected_widths[arm]
        assert config.num_params == expected_counts[arm]
        assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE

        if arm == "mamba-b3":
            assert all(isinstance(mixer, Mamba3MixerConfig) for mixer in mixers)
            assert all(mixer.rotation_block_size == 3 for mixer in mixers)
            assert all(mixer.mimo_rank == 1 for mixer in mixers)
            assert all(mixer.rotation_scan_impl == "quaternion" for mixer in mixers)
            assert all(mixer.prefer_official_kernel is True for mixer in mixers)
        elif arm == "xlstm":
            names = [type(mixer).__name__ for mixer in mixers]
            assert names.count("XLSTMMixerConfig") == 10
            assert names.count("SLSTMMixerConfig") == 2
        elif arm == "native-pd":
            assert all(isinstance(mixer, NativeFlashPDMixerConfig) for mixer in mixers)
        else:
            assert all(isinstance(mixer, NativeFlashPDMamba3SISOMixerConfig) for mixer in mixers)
        if arm in ("native-pd", "mamba3-siso-pd"):
            assert all(mixer.backend == NativePDBackend.CUDA for mixer in mixers)
            assert all(mixer.mode == NativePDMode.GENERAL_SCATTER for mixer in mixers)


def test_five_seed_four_arm_matrix_and_parser_rejects_mismatches():
    module = load_entrypoint()

    assert module.DATA_SEEDS == (210007, 220014, 230021, 240028, 250035)
    assert module.ARM_ORDER == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd")
    all_init_seeds = {
        seed for arm in module.ARM_ORDER for seed in module.INIT_SEEDS_BY_ARM[arm].values()
    }
    assert len(all_init_seeds) == 20

    arm = "native-pd"
    data_seed = 220014
    init_seed = module.INIT_SEEDS_BY_ARM[arm][data_seed]
    opts, overrides = module.parse_args(
        [
            "test-run",
            "--arm",
            arm,
            "--data-seed",
            str(data_seed),
            "--init-seed",
            str(init_seed),
            "--dry-run",
        ]
    )
    assert overrides == []
    assert opts.arm == arm
    with pytest.raises(SystemExit, match="init seed"):
        module.parse_args(
            [
                "test-run",
                "--arm",
                arm,
                "--data-seed",
                str(data_seed),
                "--init-seed",
                str(init_seed + 1),
            ]
        )


def test_reader_environment_checkpoint_and_dry_config_guards(monkeypatch, tmp_path):
    module = load_entrypoint()
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig

    opts, _ = module.parse_args(
        [
            "test-run",
            "--arm",
            "mamba-b3",
            "--data-seed",
            "210007",
            "--init-seed",
            str(module.INIT_SEEDS_BY_ARM["mamba-b3"][210007]),
            "--save-folder",
            "s3://checkpoint-contract/",
            "--work-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    corpus = module.Corpus(
        dataset_id="pretrain/reservoir-dolma2",
        version="v1",
        paths=["s3://reader-owned/path/train-00000.u32le.bin"],
        dtype=NumpyDatasetDType.uint32,
        tokenizer=TokenizerConfig.dolma2(),
        rows=250_242_924_544,
    )
    monkeypatch.setattr(module, "resolve_corpus", lambda **_: corpus)
    config = module.build_config(opts, [])

    assert config.dataset.paths == corpus.paths
    assert config.dataset.dtype == NumpyDatasetDType.uint32
    assert config.dataset.sequence_length == 4096
    assert config.data_loader.global_batch_size == 524288
    assert config.data_loader.seed == 210007
    assert config.train_module.rank_microbatch_size == 8192
    assert config.train_module.compile_model is True
    assert config.train_module.accumulate_grads_without_comm is True
    assert config.train_module.dp_config.reshard_after_forward is False
    assert config.trainer.save_folder == "s3://checkpoint-contract/"
    assert config.trainer.max_duration.value == 3721
    assert config.trainer.callbacks["checkpointer"].save_interval == 1861
    assert config.trainer.callbacks["checkpointer"].max_checkpoints is None
    assert "lm_evaluator" not in config.trainer.callbacks
    assert "downstream_evaluator" not in config.trainer.callbacks


def test_commands_docs_and_reader_contract_are_complete():
    source = TRAIN_RUNNER.read_text()
    run_yaml = COMPARISON_RUN_CONFIG.read_text()
    guide = RUN_GUIDE.read_text()

    for variable in (
        "EDULLM_RUN_ID",
        "EDULLM_CHECKPOINT_DIR",
        "EDULLM_DATASET_ID",
        "EDULLM_DATASET_VERSION",
        "EDULLM_DATASET_TOKENIZER",
    ):
        assert variable in source
    assert "from edullm_data.read import dataset_paths" in source
    assert "Boto3S3.default()" in source
    assert "\nimport boto3" not in source
    assert "\nfrom boto3" not in source

    assert "--nproc-per-node=8" in run_yaml
    assert "schema_version: 1" in run_yaml
    assert "workload_profile: olmo-core-train" in run_yaml
    assert "suggested_compute: gpu-8xa100" in run_yaml
    assert "$EDULLM_RUN_ID" in run_yaml
    assert "$EDULLM_CHECKPOINT_DIR" in run_yaml
    assert "--param-dtype bfloat16" in run_yaml
    assert "reservoir-dolma2-v1" in guide
    assert "20 cells" in guide
    assert "10 mLSTM" in guide
    assert "TPP 5.00007–5.00041" in guide
    assert "--fanout-size 20" in guide


def test_platform_dockerfile_pins_builds_and_asserts_sm80_symbols():
    dockerfile = DOCKERFILE.read_text()

    assert dockerfile.startswith("ARG BASE_IMAGE\n\nFROM ${BASE_IMAGE}")
    assert "38bf831a6c3f445e394784018441fd59288b876c" in dockerfile
    assert "e9594ce1c732d97440f0332fdc43170a2294dbfa" in dockerfile
    assert "torch==2.10.0" in dockerfile
    assert 'TORCH_CUDA_ARCH_LIST="8.0"' in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "import _flash_pd_native_cuda" in dockerfile
    for symbol in ("forward", "backward", "mamba3_forward", "paper_backward"):
        assert f"callable(_flash_pd_native_cuda.{symbol})" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "sm_80" in dockerfile
    assert "aws " not in dockerfile.lower()


def test_four_arm_comparison_uses_full_3_to_1_architectures_at_matched_tpp():
    module = load_entrypoint()
    from olmo_core.nn.attention import AttentionConfig
    from olmo_core.nn.flash_pd_native import (
        NativeFlashPDMamba3SISOMixerConfig,
        NativeFlashPDMixerConfig,
    )
    from olmo_core.nn.mamba3 import Mamba3MixerConfig

    assert module.ARMS == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd")
    assert module.ATTENTION_LAYERS == (3, 7, 11, 15)
    assert module.FROZEN_STEPS == 3721
    assert module.FROZEN_GLOBAL_BATCH_SIZE == 524288

    for arm in module.ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        mixers = [block.sequence_mixer for block in config.resolved_block_configs]
        assert len(mixers) == 16
        assert sum(isinstance(mixer, AttentionConfig) for mixer in mixers) == 4
        assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE
        tpp = module.FROZEN_STEPS * module.FROZEN_GLOBAL_BATCH_SIZE / config.num_params
        assert tpp == pytest.approx(5.0, abs=0.003)

        recurrent = [
            mixer for index, mixer in enumerate(mixers) if index not in module.ATTENTION_LAYERS
        ]
        if arm == "mamba-b3":
            assert all(isinstance(mixer, Mamba3MixerConfig) for mixer in recurrent)
        elif arm == "native-pd":
            assert all(isinstance(mixer, NativeFlashPDMixerConfig) for mixer in recurrent)
        elif arm == "mamba3-siso-pd":
            assert all(isinstance(mixer, NativeFlashPDMamba3SISOMixerConfig) for mixer in recurrent)
        else:
            names = [type(mixer).__name__ for mixer in recurrent]
            assert names.count("XLSTMMixerConfig") == 10
            assert names.count("SLSTMMixerConfig") == 2


def test_comparison_wave_is_arm_major_five_seed_single_image_fanout():
    assert COMPARISON_RUN_CONFIG.is_file()
    assert SEED_SCHEDULE.is_file()
    run_yaml = COMPARISON_RUN_CONFIG.read_text()
    schedule = __import__("json").loads(SEED_SCHEDULE.read_text())

    expected_arms = ["mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd"]
    assert schedule["arms"] == expected_arms
    assert schedule["replicates_per_arm"] == 5
    assert schedule["fanout_size"] == 20
    assert schedule["cell_order"] == [
        arm for arm in expected_arms for _ in range(schedule["replicates_per_arm"])
    ]
    assert schedule["steps"] == 3721
    assert schedule["global_batch_size"] == 524288
    assert schedule["tokens_per_cell"] == 1_950_875_648
    assert schedule["target_tokens_per_parameter"] == 5.0

    assert "AWS_BATCH_JOB_ARRAY_INDEX" in run_yaml
    assert ".edullm/train_core6_arm.py" in run_yaml
    assert "--steps 3721" in run_yaml
    assert "--global-batch-size 524288" in run_yaml
    assert run_yaml.count("mamba-b3") == 5
    assert run_yaml.count("xlstm") == 5
    assert run_yaml.count("mamba3-siso-pd") == 5
    assert run_yaml.count("native-pd") == 5


def test_single_platform_image_bundles_all_four_accelerated_backends():
    dockerfile = DOCKERFILE.read_text()
    for pin in ("xlstm==2.0.5", "mlstm-kernels==2.0.4", "flashrnn==1.0.6"):
        assert pin in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "olmo_xlstm" in dockerfile
    assert "olmo_slstm" in dockerfile


def test_throughput_diagnostic_gdn_matches_the_four_arm_shell():
    module = load_entrypoint()
    from olmo_core.nn.attention import AttentionConfig, GatedDeltaNetConfig

    assert module.DIAGNOSTIC_ARMS == ("gdn",)
    config = module.build_model_config("gdn", module.valid_init_seeds("gdn")[0])
    mixers = [block.sequence_mixer for block in config.resolved_block_configs]
    assert sum(isinstance(mixer, AttentionConfig) for mixer in mixers) == 4
    assert sum(isinstance(mixer, GatedDeltaNetConfig) for mixer in mixers) == 12
    assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE
