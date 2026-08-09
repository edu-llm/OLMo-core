import contextlib
import gc
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
ENTRYPOINT = ROOT / ".edullm/model_arch_tests.py"
TRAIN_RUNNER = ROOT / ".edullm/train_core6_arm.py"
DOCKERFILE = ROOT / ".edullm/Dockerfile"
PYPROJECT = ROOT / "pyproject.toml"
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


def build_or_skip(build):
    """Build an arm artifact, skipping only when an arm's kernel package is absent here."""
    try:
        return build()
    except ImportError as exc:
        pytest.skip(f"arm kernel package is not installed in this environment: {exc}")


def test_every_arm_exempts_exactly_its_tagged_timescale_parameters_from_weight_decay():
    """The exempt globs must equal the arm's own ``_no_weight_decay`` tags, and must all match.

    Two independent failures live here, and both cost a billed machine.

    The tags are INERT ON THEIR OWN: ``OptimConfig.build_groups`` reads ``group_overrides``
    and nothing else, so a mixer that marks ``A_log``/``dt_bias``/``D`` still has weight
    decay applied to them unless a pattern names them. That silently shrinks the recurrence
    timescales the arms are supposed to be compared on.

    A pattern that matches NOTHING is the opposite failure and it is fatal:
    ``TransformerTrainModule`` builds the optimizer with ``strict=True``, and
    ``_expand_param_globs`` raises ``OLMoConfigurationError`` for an unmatched pattern. So
    one shared list across arms cannot be right -- ``xlstm`` has no such parameter at all,
    and ``mamba-b3`` has no ``D``.
    """
    module = load_entrypoint()
    from fnmatch import fnmatch

    from olmo_core.nn.utils import no_weight_decay_param_names

    for arm in module.RUNNABLE_ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        model = build_or_skip(lambda config=config: config.build(init_device="meta"))
        names = [name for name, _ in model.named_parameters()]
        tagged = set(no_weight_decay_param_names(model))

        overrides = module.weight_decay_group_overrides(arm)
        assert len(overrides) == 1, arm
        assert overrides[0].opts == {"weight_decay": 0.0}, arm
        patterns = list(overrides[0].params)

        # Embeddings are exempt in every arm, exactly as the bake-off runner had them.
        assert patterns[0] == "embeddings.weight", arm

        # Every pattern matches something, or the optimizer build raises under strict=True.
        for pattern in patterns:
            assert any(fnmatch(name, pattern) for name in names), (arm, pattern)

        # And the exempted set is exactly the tagged set, plus the embeddings.
        exempted = {name for name in names if any(fnmatch(name, p) for p in patterns)}
        assert exempted == tagged | {"embeddings.weight"}, arm

        del model
        gc.collect()


def materialized_block(module, block_config, index):
    """Build one block the way training does, then give its meta parameters storage."""
    block = build_or_skip(
        lambda: block_config.build(
            d_model=module.D_MODEL,
            block_idx=index,
            n_layers=module.N_LAYERS,
            init_device="meta",
        )
    )
    block.to_empty(device="cpu")
    return block


def root_unit_parameters(model):
    """The parameters FSDP keeps in the root unit, which are the ones outside any block."""
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("blocks.")
    }


@contextlib.contextmanager
def single_rank_gloo_group(tmp_path):
    """A world-size-1 CPU process group, the smallest thing FSDP2 will shard against."""
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    assert not dist.is_initialized(), "another test left a process group initialized"
    dist.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", world_size=1, rank=0
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


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
            assert all(mixer.ssd_backend == "simple_gla" for mixer in mixers)
        elif arm == "xlstm":
            names = [type(mixer).__name__ for mixer in mixers]
            assert names.count("XLSTMMixerConfig") == 10
            assert names.count("SLSTMMixerConfig") == 2
            slstm = [mixer for mixer in mixers if type(mixer).__name__ == "SLSTMMixerConfig"]
            # FlashRNN compiles one pointer type per tensor role, so its kernel dtype has to
            # be the bfloat16 that FSDP hands the persistent kernel.
            assert all(mixer.kernel_dtype == "bfloat16" for mixer in slstm)
            assert all(mixer.backend == "cuda_fused" for mixer in slstm)
            assert all(mixer.batch_size == 2 for mixer in slstm)
            assert all(mixer.fuse_input_projections is True for mixer in slstm)
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
    from olmo_core.config import DType
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
    # Parameters are stored in float32 and FSDP is what casts them to bfloat16 for compute.
    assert config.model.dtype == DType.float32
    assert config.train_module.dp_config.param_dtype == DType.bfloat16
    assert config.train_module.dp_config.reduce_dtype == DType.float32
    assert config.train_module.optim.group_overrides == module.weight_decay_group_overrides(
        "mamba-b3"
    )
    assert config.trainer.save_folder == "s3://checkpoint-contract/"
    assert config.trainer.max_duration.value == 1144
    assert config.trainer.callbacks["checkpointer"].save_interval == 572
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
    assert "12 cells" in guide
    assert "10 mLSTM" in guide
    assert "TPP 1.53724–1.53735" in guide
    assert "--fanout-size 12" in guide


def test_platform_dockerfile_pins_builds_and_asserts_sm80_symbols():
    dockerfile = DOCKERFILE.read_text()
    pyproject = PYPROJECT.read_text()

    assert dockerfile.startswith("ARG BASE_IMAGE\n\nFROM ${BASE_IMAGE}")
    assert "38bf831a6c3f445e394784018441fd59288b876c" in dockerfile
    assert "e9594ce1c732d97440f0332fdc43170a2294dbfa" in dockerfile
    assert "torch==2.10.0" in dockerfile
    assert 'TORCH_CUDA_ARCH_LIST="8.0"' in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "_flash_pd_native_cuda" in dockerfile
    assert "import torch, _flash_pd_native_cuda" in dockerfile
    assert 'python -c "import _flash_pd_native_cuda' not in dockerfile
    for symbol in ("forward", "backward", "mamba3_forward", "paper_backward"):
        assert f"callable(_flash_pd_native_cuda.{symbol})" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "sm_80" in dockerfile
    assert "torch._C._cuda_getArchFlags()" in dockerfile
    assert "torch.cuda.get_arch_list()" not in dockerfile
    assert 'fla = ["flash-linear-attention==0.5.1"]' in pyproject
    assert '"flash-linear-attention==0.5.1"' in dockerfile
    assert "v=version('flash-linear-attention'); assert v=='0.5.1'" in dockerfile
    assert "v=version('fla-core'); assert v=='0.5.1'" in dockerfile
    assert "from fla.ops.gdn2 import chunk_gdn2; assert callable(chunk_gdn2)" in dockerfile
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
    assert module.FROZEN_STEPS == 1144
    assert module.FROZEN_GLOBAL_BATCH_SIZE == 524288

    for arm in module.ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        mixers = [block.sequence_mixer for block in config.resolved_block_configs]
        assert len(mixers) == 16
        assert sum(isinstance(mixer, AttentionConfig) for mixer in mixers) == 4
        assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE
        tpp = module.FROZEN_STEPS * module.FROZEN_GLOBAL_BATCH_SIZE / config.num_params
        assert tpp == pytest.approx(1.5373, abs=0.0001)

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


def test_comparison_wave_is_arm_major_three_seed_single_image_fanout():
    assert COMPARISON_RUN_CONFIG.is_file()
    assert SEED_SCHEDULE.is_file()
    run_yaml = COMPARISON_RUN_CONFIG.read_text()
    schedule = __import__("json").loads(SEED_SCHEDULE.read_text())

    expected_arms = ["mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd"]
    assert schedule["arms"] == expected_arms
    assert schedule["replicates_per_arm"] == 3
    assert schedule["fanout_size"] == 12
    assert schedule["cell_order"] == [
        arm for arm in expected_arms for _ in range(schedule["replicates_per_arm"])
    ]
    assert schedule["steps"] == 1144
    assert schedule["global_batch_size"] == 524288
    assert schedule["tokens_per_cell"] == 599_785_472
    assert schedule["target_tokens_per_parameter"] == 1.5373
    assert schedule["warmup_steps"] == 114
    assert schedule["save_interval"] == 572

    assert "AWS_BATCH_JOB_ARRAY_INDEX" in run_yaml
    assert ".edullm/train_core6_arm.py" in run_yaml
    assert "--steps 1144" in run_yaml
    assert "--warmup-steps 114" in run_yaml
    assert "--save-interval 572" in run_yaml
    assert "--global-batch-size 524288" in run_yaml
    assert run_yaml.count("mamba-b3") == 3
    assert run_yaml.count("xlstm") == 3
    assert run_yaml.count("mamba3-siso-pd") == 3
    assert run_yaml.count("native-pd") == 3


def test_single_platform_image_bundles_all_four_accelerated_backends():
    dockerfile = DOCKERFILE.read_text()
    for pin in ("xlstm==2.0.5", "mlstm-kernels==2.0.4", "flashrnn==1.0.6"):
        assert pin in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "olmo_xlstm" in dockerfile
    assert "olmo_slstm" in dockerfile


def test_throughput_diagnostic_gdn_is_exact_measured_gdn2():
    module = load_entrypoint()
    from olmo_core.nn.attention import (
        AttentionConfig,
        GatedDeltaNet2,
        GatedDeltaNet2Config,
    )

    assert module.DIAGNOSTIC_ARMS == ("gdn",)
    assert module.DIAGNOSTIC_PARAMETER_COUNTS == {"gdn": 390_119_360}
    assert module.solve_widths("gdn") == (3808,) + (3776,) * 11
    config = module.build_model_config("gdn", module.valid_init_seeds("gdn")[0])
    mixers = [block.sequence_mixer for block in config.resolved_block_configs]
    assert sum(isinstance(mixer, AttentionConfig) for mixer in mixers) == 4
    gdn2_mixers = [mixer for mixer in mixers if isinstance(mixer, GatedDeltaNet2Config)]
    assert len(gdn2_mixers) == 12
    assert all(type(mixer) is GatedDeltaNet2Config for mixer in gdn2_mixers)
    assert all(mixer.n_heads == 16 for mixer in gdn2_mixers)
    assert all(mixer.head_dim == 64 for mixer in gdn2_mixers)
    assert all(mixer.expand_v == 1.0 for mixer in gdn2_mixers)
    assert all(mixer.allow_neg_eigval is False for mixer in gdn2_mixers)
    assert all(mixer.conv_size == 4 for mixer in gdn2_mixers)
    realized = gdn2_mixers[0].build(module.D_MODEL, layer_idx=0, n_layers=module.N_LAYERS)
    assert type(realized) is GatedDeltaNet2
    assert config.num_params == module.DIAGNOSTIC_PARAMETER_COUNTS["gdn"]
    assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE


def test_every_arm_declares_one_fp32_master_dtype_and_keeps_its_bf16_kernels():
    module = load_entrypoint()
    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.flash_pd_native import NativePDBackend, NativePDMode

    assert module.RUNNABLE_ARMS == ("mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd", "gdn")
    for arm in module.RUNNABLE_ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        assert config.dtype == DType.float32, arm
        assert config.lm_head.dtype == DType.float32, arm
        assert config.lm_head.layer_norm.dtype == DType.float32, arm
        for index, block in enumerate(config.resolved_block_configs):
            assert block.layer_norm.dtype == DType.float32, (arm, index)
            assert block.feed_forward.dtype == DType.float32, (arm, index)
            assert block.sequence_mixer.dtype == DType.float32, (arm, index)

    # Master storage moving to float32 must not move any kernel off bfloat16, and must not
    # relax an accelerated backend into a portable one.
    attention = module._attention_mixer()
    assert attention.backend == AttentionBackendName.torch
    assert attention.qk_norm.dtype == DType.float32
    assert module._mlstm_mixer().autocast_kernel_dtype == "bfloat16"
    assert module._mlstm_mixer().chunkwise_kernel == "chunkwise--triton_xl_chunk"
    assert module._slstm_mixer().kernel_dtype == "bfloat16"
    assert module._slstm_mixer().backend == "cuda_fused"
    assert module._treatment_mixer("mamba-b3", 0).prefer_official_kernel is True
    for arm in ("native-pd", "mamba3-siso-pd"):
        mixer = module._treatment_mixer(arm, 0)
        assert mixer.backend == NativePDBackend.CUDA
        assert mixer.mode == NativePDMode.GENERAL_SCATTER


@pytest.mark.parametrize("arm", ["mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd", "gdn"])
def test_every_built_block_holds_exactly_one_fp32_parameter_dtype(arm):
    module = load_entrypoint()
    import torch

    assert arm in module.RUNNABLE_ARMS
    expected = {**module.EXACT_PARAMETER_COUNTS, **module.DIAGNOSTIC_PARAMETER_COUNTS}[arm]
    config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
    assert config.num_params == expected

    model = build_or_skip(lambda: config.build(init_device="meta"))
    assert sum(parameter.numel() for parameter in model.parameters()) == expected
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}
    for name, block in model.blocks.items():
        assert {parameter.dtype for parameter in block.parameters()} == {torch.float32}, (
            arm,
            name,
        )


@pytest.mark.parametrize("arm", ["mamba-b3", "xlstm", "mamba3-siso-pd", "native-pd", "gdn"])
def test_single_rank_fsdp2_full_wrap_lazy_init_keeps_fp32_master_and_bf16_compute(arm, tmp_path):
    module = load_entrypoint()
    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
    block_configs = list(config.resolved_block_configs)

    with single_rank_gloo_group(tmp_path):
        mesh = init_device_mesh("cpu", (1,))
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        # The full wrapping strategy makes every block its own FSDP unit, and lazy init
        # rejects a unit whose parameters were not all built in one dtype. Each block is
        # released before the next one is built so the whole model is never resident.
        for index, block_config in enumerate(block_configs):
            block = materialized_block(module, block_config, index)
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
            block.unshard()
            assert {parameter.dtype for parameter in block.parameters()} == {torch.bfloat16}, (
                arm,
                index,
            )
            block.reshard()
            assert {parameter.dtype for parameter in block.parameters()} == {torch.float32}, (
                arm,
                index,
            )
            del block
            gc.collect()

        # The remaining FSDP unit is the root, which holds the tied embedding weight and the
        # final layer norm. Its blocks stay on meta because they are already their own units.
        model = build_or_skip(lambda: config.build(init_device="meta"))
        for block in model.blocks.values():
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        assert list(root_unit_parameters(model)) == ["embeddings.weight", "lm_head.norm.weight"]
        model.embeddings.to_empty(device="cpu")
        model.lm_head.to_empty(device="cpu")
        # `to_empty` allocates fresh storage, so restore the tie the way `init_weights` does.
        model.lm_head.w_out.weight = model.embeddings.weight

        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        model.unshard()
        root = root_unit_parameters(model)
        assert {parameter.dtype for parameter in root.values()} == {torch.bfloat16}, arm
        model.reshard()
        root = root_unit_parameters(model)
        assert {parameter.dtype for parameter in root.values()} == {torch.float32}, arm
        del model, root
        gc.collect()


def test_single_rank_fsdp2_forward_backward_trains_fp32_masters_through_bf16_math(tmp_path):
    module = load_entrypoint()
    import torch
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    # Every arm shares one attention block, and it is the only block whose kernels run without
    # a GPU, so it is the block that can carry a real forward and backward here.
    config = module.build_model_config("mamba-b3", module.valid_init_seeds("mamba-b3")[0])
    block_config = config.resolved_block_configs[module.ATTENTION_LAYERS[0]]

    with single_rank_gloo_group(tmp_path):
        block = materialized_block(module, block_config, module.ATTENTION_LAYERS[0])
        generator = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.normal_(0.0, 0.02, generator=generator)

        fully_shard(
            block,
            mesh=init_device_mesh("cpu", (1,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            reshard_after_forward=False,
        )
        output = block(torch.randn(1, 32, module.D_MODEL, generator=generator))
        assert output.dtype == torch.bfloat16
        output.float().pow(2).mean().backward()

        gradients = [
            parameter.grad for parameter in block.parameters() if parameter.grad is not None
        ]
        assert len(gradients) == len(list(block.parameters()))
        assert {gradient.dtype for gradient in gradients} == {torch.float32}
        assert all(torch.isfinite(gradient.to_local()).all() for gradient in gradients)
