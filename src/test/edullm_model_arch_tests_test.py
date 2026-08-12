import contextlib
import gc
import hashlib
import importlib.util
import json
import re
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
XLSTM_RERUN_CONFIG = ROOT / ".edullm/run-xlstm-rerun.yaml"
MAMBA_FAITHFUL_RERUN_CONFIG = ROOT / ".edullm/run-mamba-b3-faithful.yaml"
PD_RERUN_CONFIG = ROOT / ".edullm/run-pd-rerun.yaml"
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

    # An arm promoted into the wave with no ledger row would raise KeyError inside
    # `weight_decay_group_overrides` on a billed machine, so the ledger's key set is pinned
    # to the arm list rather than left to the loop below to discover arm by arm.
    assert set(module.WEIGHT_DECAY_EXEMPT_PATTERNS_BY_ARM) == set(module.RUNNABLE_ARMS)

    tagged_by_arm = {}
    for arm in module.RUNNABLE_ARMS:
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        model = build_or_skip(lambda config=config: config.build(init_device="meta"))
        names = [name for name, _ in model.named_parameters()]
        tagged = set(no_weight_decay_param_names(model))
        tagged_by_arm[arm] = tagged

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

    # `gdn` HAS twelve `A_log` AND TWELVE `dt_bias` PARAMETERS AND EXEMPTS ALL TWENTY-FOUR.
    # `GatedDeltaNet2` was the one recurrent mixer that never set `_no_weight_decay`, so the only
    # ledger row that could satisfy the equality above was the empty one, and the arm decayed its
    # recurrence timescales under AdamW's 0.01 while Mamba-3 and both PD mixers did not. The
    # mixer tags them now. The equality above already fails if a tag is dropped; what is pinned
    # here is the *count*, so an arm rebuilt with fewer recurrent slots -- which would keep both
    # sides of that equality consistent while quietly changing the comparison -- is caught too.
    gdn = module.build_model_config("gdn", module.valid_init_seeds("gdn")[0])
    gdn_model = build_or_skip(lambda: gdn.build(init_device="meta"))
    gdn_names = [name for name, _ in gdn_model.named_parameters()]
    assert len([name for name in gdn_names if name.endswith(".A_log")]) == 12
    assert len([name for name in gdn_names if name.endswith(".dt_bias")]) == 12
    # No `D`, so the row must not name one: an unmatched pattern is fatal under `strict=True`.
    assert [name for name in gdn_names if name.endswith(".D")] == []
    assert module.WEIGHT_DECAY_EXEMPT_PATTERNS_BY_ARM["gdn"] == ("*.A_log", "*.dt_bias")
    assert tagged_by_arm["gdn"] == {
        name for name in gdn_names if name.endswith((".A_log", ".dt_bias"))
    }
    assert len(tagged_by_arm["gdn"]) == 24
    del gdn_model
    gc.collect()

    # THE THREE KDA ARMS CARRY THE SAME TWO-PATTERN ROW AND MUST NOT CARRY A THIRD. All three
    # classes tag `A_log` and `dt_bias`, and none of them has a `D` -- so the row Mamba-3 and
    # both PD mixers use, which names `*.D`, is fatal here rather than merely redundant:
    # `_expand_param_globs` raises under `strict=True` for a pattern with no match, and it
    # raises while building the optimizer on a machine that has already been billed. The
    # Householder variant widens `w_k`/`w_v`/`w_b` by R and leaves the timescales alone, so its
    # count of tagged parameters is the same twelve and twelve as the other two.
    for arm in ("kda", "kda-hh-r2", "kda-gconv"):
        assert module.WEIGHT_DECAY_EXEMPT_PATTERNS_BY_ARM[arm] == ("*.A_log", "*.dt_bias"), arm
        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        model = build_or_skip(lambda config=config: config.build(init_device="meta"))
        names = [name for name, _ in model.named_parameters()]
        assert len([name for name in names if name.endswith(".A_log")]) == 12, arm
        assert len([name for name in names if name.endswith(".dt_bias")]) == 12, arm
        assert [name for name in names if name.endswith(".D")] == [], arm
        assert len(tagged_by_arm[arm]) == 24, arm
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


def test_eight_arm_geometry_has_identical_attention_and_full_recurrent_treatments():
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

    # THE INVARIANT THE WHOLE COMPARISON RESTS ON. Outside the twelve recurrent slots the arms
    # must be identical, so that a difference is attributable to the mixer. The seven post-norm
    # arms still share one byte-identical attention block. The faithful Mamba arm is the one
    # disclosed exception: published Mamba is pre-norm throughout, so its four attention blocks
    # differ from the shared ones ONLY in the block norm ordering -- same mixer, feed-forward and
    # layer norm -- which is asserted explicitly below rather than folded into the fingerprint.
    assert len(configs) == 8
    pre_norm_arms = set(module._PRE_NORM_ARMS)
    shared_arms = [arm for arm in module.ARMS if arm not in pre_norm_arms]
    reference = configs[shared_arms[0]]
    fingerprints = set()
    for arm in shared_arms:
        config = configs[arm]
        for index in module.ATTENTION_LAYERS:
            assert (
                config.resolved_block_configs[index].as_config_dict()
                == reference.resolved_block_configs[index].as_config_dict()
            ), (arm, index)
        attention_blocks = [
            config.resolved_block_configs[index].as_config_dict()
            for index in module.ATTENTION_LAYERS
        ]
        assert {block["feed_forward"]["hidden_size"] for block in attention_blocks} == {4608}, arm
        fingerprints.add(
            hashlib.blake2b(
                json.dumps(attention_blocks, sort_keys=True).encode(), digest_size=8
            ).hexdigest()
        )
    # Pinned to the value the four-arm wave was first measured under; the three KDA arms are
    # inside it now and it did not move, and the pre-norm Mamba arm does not perturb it.
    assert fingerprints == {"deb8ff528e7359fa"}

    for arm in pre_norm_arms:
        config = configs[arm]
        for index in module.ATTENTION_LAYERS:
            block = config.resolved_block_configs[index].as_config_dict()
            shared = reference.resolved_block_configs[index].as_config_dict()
            assert block["name"] == "default", (arm, index)
            assert shared["name"] == "reordered_norm", index
            assert {k: v for k, v in block.items() if k != "name"} == {
                k: v for k, v in shared.items() if k != "name"
            }, (arm, index)


def test_mamba_b3_restores_the_successful_state_size_and_official_backend():
    """Keep the repaired arm on the July run's state capacity and exact SSD implementation."""
    module = load_entrypoint()
    from olmo_core.nn.mamba3 import Mamba3MixerConfig

    config = module.build_model_config("mamba-b3", module.valid_init_seeds("mamba-b3")[0])
    mixers = [
        config.resolved_block_configs[index].sequence_mixer for index in module.RECURRENT_LAYERS
    ]

    assert all(isinstance(mixer, Mamba3MixerConfig) for mixer in mixers)
    assert all(mixer.d_state == 192 for mixer in mixers)
    assert all(mixer.ssd_backend == "official_fast" for mixer in mixers)


def test_treatment_mixers_are_strict_and_parameter_matched():
    module = load_entrypoint()
    from olmo_core.nn.attention import (
        GatedDeltaNet2Config,
        KimiDeltaAttentionConfig,
        KimiDeltaHouseholderConfig,
    )
    from olmo_core.nn.flash_pd_native import (
        NativeFlashPDMamba3SISOMixerConfig,
        NativeFlashPDMixerConfig,
        NativePDBackend,
        NativePDMode,
    )
    from olmo_core.nn.mamba3 import Mamba3MixerConfig

    expected_counts = {
        "mamba-b3": 390_154_112,
        "xlstm": 390_143_056,
        "mamba3-pd": 390_170_432,
        "native-pd": 390_143_744,
        "gdn": 390_119_360,
        "kda": 390_119_360,
        "kda-hh-r2": 390_119_360,
        "kda-gconv": 390_094_784,
    }
    expected_widths = {
        # Faithful SISO expand=2 makes the Mamba mixer ~6.88M a layer (vs ~3.77M before), so the
        # arm buys back much less FFN than it used to: ~3,680 a recurrent slot against ~4,704.
        "mamba-b3": (3680,) * 11 + (3648,),
        "xlstm": (4672,) * 8 + (4640,) * 4,
        "mamba3-pd": (2752,) * 11 + (2720,),
        "native-pd": (2432,) * 6 + (2400,) * 6,
        "gdn": (3808,) + (3776,) * 11,
        "kda": (4480,) * 3 + (4448,) * 9,
        # KDA with two Householder factors is the widest mixer in the wave at 6,608,976
        # parameters a layer, so it buys the least FFN back -- landing within 32 of GDN-2's
        # solution, whose mixer is 6,568,016.
        "kda-hh-r2": (3776,) * 8 + (3744,) * 4,
        "kda-gconv": (4480,) * 2 + (4448,) * 10,
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
            assert all(mixer.d_state == 192 for mixer in mixers)
            assert all(mixer.rotation_block_size == 3 for mixer in mixers)
            assert all(mixer.mimo_rank == 1 for mixer in mixers)
            assert all(mixer.rotation_scan_impl == "quaternion" for mixer in mixers)
            assert all(mixer.prefer_official_kernel is True for mixer in mixers)
            assert all(mixer.ssd_backend == "official_fast" for mixer in mixers)
            # Faithful published SISO: expand=2 (32x64=2048 inner), token-dependent decay,
            # post-BCNorm head bias, learned D skip, norm-before-gate, and the per-head
            # dt-scaled rotation over half the state. The unfused layout is required for them.
            assert all(mixer.n_heads == 32 and mixer.head_dim == 64 for mixer in mixers)
            assert all(mixer.dynamic_a for mixer in mixers)
            assert all(mixer.d_skip for mixer in mixers)
            assert all(mixer.norm_before_gate for mixer in mixers)
            assert all(mixer.bc_bias_after_norm and not mixer.bc_bias for mixer in mixers)
            assert all(mixer.dt_scaled_rotation for mixer in mixers)
            assert all(mixer.rope_fraction == 0.5 for mixer in mixers)
            assert all(mixer.fuse_input_projections is False for mixer in mixers)
            # The one deliberate speed-for-fidelity trade: a group-shared rotation timescale, so
            # B/C stay one group wide and the scan keeps GQA. Measured 18.8x less rotation work
            # than the published per-head form at this arm's microbatch shape.
            assert all(mixer.rotation_timescale == "group_mean" for mixer in mixers)
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
        elif arm == "gdn":
            assert all(type(mixer) is GatedDeltaNet2Config for mixer in mixers)
            assert all(mixer.n_heads == 16 for mixer in mixers)
            assert all(mixer.head_dim == 64 for mixer in mixers)
            assert all(mixer.expand_v == 1.0 for mixer in mixers)
        elif arm == "kda":
            # The shipped KDA operator: plain SiLU short convolutions, one delta factor,
            # non-negative eigenvalues. It is the baseline the other two are read against, so
            # every option that either of them moves is pinned to its default here.
            assert all(type(mixer) is KimiDeltaAttentionConfig for mixer in mixers)
            assert all(mixer.gated_conv is False for mixer in mixers)
            assert all(mixer.conv_activation == "silu" for mixer in mixers)
            assert all(mixer.allow_neg_eigval is False for mixer in mixers)
        elif arm == "kda-hh-r2":
            assert all(type(mixer) is KimiDeltaHouseholderConfig for mixer in mixers)
            assert all(mixer.num_householder == 2 for mixer in mixers)
            assert all(mixer.allow_neg_eigval is True for mixer in mixers)
            # `torch` is the CPU reference recurrence and is far slower; a wave cell that
            # silently ran it would be ranked on throughput against seven fused arms.
            assert all(mixer.backend == "triton" for mixer in mixers)
        elif arm == "kda-gconv":
            assert all(type(mixer) is KimiDeltaAttentionConfig for mixer in mixers)
            assert all(mixer.gated_conv is True for mixer in mixers)
            assert all(mixer.gate_structure == "depthwise" for mixer in mixers)
            assert all(mixer.gate_rank is None for mixer in mixers)
            assert all(mixer.allow_neg_eigval is False for mixer in mixers)
        else:
            assert arm == "mamba3-pd"
            assert all(isinstance(mixer, NativeFlashPDMamba3SISOMixerConfig) for mixer in mixers)
        if arm in ("kda", "kda-hh-r2", "kda-gconv"):
            assert all(mixer.n_heads == 16 for mixer in mixers)
            assert all(mixer.head_dim == 64 for mixer in mixers)
            assert all(mixer.expand_v == 1.0 for mixer in mixers)
            assert all(mixer.conv_size == 4 for mixer in mixers)
        if arm in ("native-pd", "mamba3-pd"):
            assert all(mixer.backend == NativePDBackend.CUDA for mixer in mixers)
            assert all(mixer.mode == NativePDMode.GENERAL_SCATTER for mixer in mixers)
        if arm == "native-pd":
            # 64, not the 128 this arm was first written with. Measured `paper_backward` at
            # 2.98-3.00 ms per layer-step on chunk 128 against 2.59-2.72 ms on 64, with the
            # forwards level -- about 0.3 ms a layer-step. Asserted here beside the exact
            # parameter count because the two facts belong together: the chunk blocks the scan
            # and shapes no weight, so the count below must not have moved.
            assert all(mixer.chunk_size == 64 for mixer in mixers)


def test_five_seed_eight_arm_matrix_and_parser_rejects_mismatches():
    module = load_entrypoint()

    assert module.DATA_SEEDS == (210007, 220014, 230021, 240028, 250035)
    assert module.ARM_ORDER == (
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    )
    all_init_seeds = {
        seed for arm in module.ARM_ORDER for seed in module.INIT_SEEDS_BY_ARM[arm].values()
    }
    # FORTY DISTINCT INTEGERS, EIGHT ARMS BY FIVE DATA SEEDS. A repeat would not fail anything
    # at run time: two cells would simply draw the same weights while their records claimed
    # otherwise, and the wave would report a replicate it never ran.
    assert len(all_init_seeds) == 40

    # The three arms appended after `gdn` must not have taken an integer that was already
    # issued OR reserved, and none of the forty may collide with a data seed either -- the two
    # ledgers are separate but both are printed into the same run record.
    five_arm_seeds = {
        seed
        for arm in ("mamba-b3", "xlstm", "mamba3-pd", "native-pd", "gdn")
        for seed in module.INIT_SEEDS_BY_ARM[arm].values()
    }
    kda_seeds = {
        seed
        for arm in ("kda", "kda-hh-r2", "kda-gconv")
        for seed in module.INIT_SEEDS_BY_ARM[arm].values()
    }
    assert len(five_arm_seeds) == 25
    assert len(kda_seeds) == 15
    assert five_arm_seeds & kda_seeds == set()
    assert all_init_seeds & set(module.DATA_SEEDS) == set()

    # And no existing arm's row moved, because a reissued init seed silently invalidates the
    # five-arm cells that were already described off this ledger.
    assert module.INIT_SEEDS_BY_ARM["gdn"] == {
        210007: 122011,
        220014: 132018,
        230021: 142025,
        240028: 152032,
        250035: 162039,
    }

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
    assert "24 cells" in guide
    assert "10 mLSTM" in guide
    assert "TPP 1.53723–1.53754" in guide
    assert "--fanout-size 24" in guide
    # The shorter prefixes stay runnable and documented, because that is the whole reason every
    # arm after the fourth was appended rather than inserted.
    assert "--fanout-size 12" in guide
    assert "--fanout-size 15" in guide


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


def test_eight_arm_comparison_uses_full_3_to_1_architectures_at_matched_tpp():
    module = load_entrypoint()
    from olmo_core.nn.attention import (
        AttentionConfig,
        GatedDeltaNet2Config,
        KimiDeltaAttentionConfig,
        KimiDeltaHouseholderConfig,
    )
    from olmo_core.nn.flash_pd_native import (
        NativeFlashPDMamba3SISOMixerConfig,
        NativeFlashPDMixerConfig,
    )
    from olmo_core.nn.mamba3 import Mamba3MixerConfig

    assert module.ARMS == (
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    )
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
        # The band the run guide and the pre-registration both quote, and it is the arms that
        # set it: `mamba3-pd` is the low end at 1.53723, having become the largest exact model
        # at 390,170,432 when the output norm added 768 weights, and `kda-gconv`, the smallest
        # at 390,094,784, the high at 1.53754. The band is a REPORTED consequence of
        # the ledger and not a constraint on it -- the frozen constraint is the +/-195,068
        # parameter tolerance asserted above, and every arm is well inside it.
        assert 1.53723 <= tpp <= 1.53754, (arm, tpp)

        recurrent = [
            mixer for index, mixer in enumerate(mixers) if index not in module.ATTENTION_LAYERS
        ]
        assert len(recurrent) == 12
        if arm == "mamba-b3":
            assert all(isinstance(mixer, Mamba3MixerConfig) for mixer in recurrent)
        elif arm == "native-pd":
            assert all(isinstance(mixer, NativeFlashPDMixerConfig) for mixer in recurrent)
        elif arm == "mamba3-pd":
            assert all(isinstance(mixer, NativeFlashPDMamba3SISOMixerConfig) for mixer in recurrent)
        elif arm == "gdn":
            assert all(type(mixer) is GatedDeltaNet2Config for mixer in recurrent)
        elif arm in ("kda", "kda-gconv"):
            assert all(type(mixer) is KimiDeltaAttentionConfig for mixer in recurrent)
        elif arm == "kda-hh-r2":
            assert all(type(mixer) is KimiDeltaHouseholderConfig for mixer in recurrent)
        else:
            names = [type(mixer).__name__ for mixer in recurrent]
            assert names.count("XLSTMMixerConfig") == 10
            assert names.count("SLSTMMixerConfig") == 2


def test_comparison_wave_is_arm_major_three_seed_single_image_fanout():
    assert COMPARISON_RUN_CONFIG.is_file()
    assert SEED_SCHEDULE.is_file()
    run_yaml = COMPARISON_RUN_CONFIG.read_text()
    schedule = json.loads(SEED_SCHEDULE.read_text())

    expected_arms = [
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    ]
    assert schedule["arms"] == expected_arms
    assert schedule["replicates_per_arm"] == 3
    assert schedule["fanout_size"] == 24
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

    # THE ARM NAMES ARE COUNTED AS SHELL WORDS, NOT AS SUBSTRINGS, and that is not fussiness:
    # `kda` occurs inside `kda-hh-r2` and `kda-gconv`, so `run_yaml.count("kda") == 3` is false
    # for a correct file and true for several wrong ones. The spec's own array is the thing the
    # fan-out indexes, so it is the thing to count.
    match = re.search(r"ARMS=\((.*?)\) &&", run_yaml, flags=re.DOTALL)
    assert match is not None
    spec_arms = match.group(1).split()
    assert spec_arms == schedule["cell_order"]
    for arm in expected_arms:
        assert spec_arms.count(arm) == 3, arm

    # EVERY ARM AFTER THE FOURTH OCCUPIES A SUFFIX AND NEVER AN INSERTION, so `--fanout-size 12`
    # still reproduces the four-arm wave, `15` the five-arm one, and `24` the whole thing.
    assert schedule["cell_order"][:12] == [
        arm for arm in expected_arms[:4] for _ in range(schedule["replicates_per_arm"])
    ]
    assert schedule["cell_order"][12:15] == ["gdn"] * 3
    assert schedule["cell_order"][15:] == ["kda"] * 3 + ["kda-hh-r2"] * 3 + ["kda-gconv"] * 3
    assert spec_arms[:15] == schedule["cell_order"][:15]


def test_xlstm_rerun_spec_is_exactly_the_three_failed_cells():
    """The repair wave must rerun xLSTM only, under the unchanged V2 recipe."""
    assert XLSTM_RERUN_CONFIG.is_file()
    rerun_yaml = XLSTM_RERUN_CONFIG.read_text()

    arrays = {}
    for name in ("ARMS", "DSEEDS", "ISEEDS"):
        match = re.search(rf"{name}=\((.*?)\) &&", rerun_yaml, flags=re.DOTALL)
        assert match is not None, name
        arrays[name] = match.group(1).split()

    assert arrays["ARMS"] == ["xlstm", "xlstm", "xlstm"]
    assert arrays["DSEEDS"] == ["210007", "220014", "230021"]
    assert arrays["ISEEDS"] == ["113008", "123015", "133022"]
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in rerun_yaml
    assert ".edullm/train_core6_arm.py" in rerun_yaml
    assert "--sequence-length 4096" in rerun_yaml
    assert "--steps 1144" in rerun_yaml
    assert "--warmup-steps 114" in rerun_yaml
    assert "--learning-rate 3e-4" in rerun_yaml
    assert "--global-batch-size 524288" in rerun_yaml
    assert "--rank-microbatch-size 8192" in rerun_yaml
    assert "--save-interval 572" in rerun_yaml
    assert "--param-dtype bfloat16" in rerun_yaml


def test_pd_rerun_spec_is_exactly_the_six_pd_cells_at_their_own_ledger_seeds():
    """The PD wave must rerun the two PD arms only, under the unchanged V2 recipe.

    Same data seeds, optimizer flags, token budget, and hardware shape as cells 6-11 of the V2
    wave, and each arm's init seeds read out of the ledger rather than hand-copied, so these six
    rows replace those rows instead of being a separately-tuned follow-up. Arm-major and
    `mamba3-pd` first, matching the ledger's arm order, so a truncated fan-out loses a whole arm.
    """
    module = load_entrypoint()
    assert PD_RERUN_CONFIG.is_file()
    rerun_yaml = PD_RERUN_CONFIG.read_text()

    arrays = {}
    for name in ("ARMS", "DSEEDS", "ISEEDS"):
        match = re.search(rf"{name}=\((.*?)\) &&", rerun_yaml, flags=re.DOTALL)
        assert match is not None, name
        arrays[name] = match.group(1).split()

    assert arrays["ARMS"] == ["mamba3-pd"] * 3 + ["native-pd"] * 3
    assert arrays["DSEEDS"] == ["210007", "220014", "230021"] * 2
    assert arrays["ISEEDS"] == [
        str(module.INIT_SEEDS_BY_ARM[arm][int(seed)])
        for arm, seed in zip(arrays["ARMS"], arrays["DSEEDS"])
    ]
    # The same seeds cells 6-11 of the wave ran, so the two waves are paired per replicate.
    assert arrays["ISEEDS"] == ["116009", "126016", "136023", "119010", "129017", "139024"]

    # Byte-identical recipe. A drift in any of these makes the rerun a different experiment.
    for flag in (
        "--sequence-length 4096",
        "--steps 1144",
        "--warmup-steps 114",
        "--learning-rate 3e-4",
        "--global-batch-size 524288",
        "--rank-microbatch-size 8192",
        "--save-interval 572",
        "--param-dtype bfloat16",
    ):
        assert flag in rerun_yaml, flag
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in rerun_yaml
    assert ".edullm/train_core6_arm.py" in rerun_yaml
    assert "--fanout-size 6" in rerun_yaml
    # The decode probe stays on: the wave rows these replace carry one.
    assert "--no-decode-probe" not in rerun_yaml

    # And the arms this spec names are the ones actually carrying both changes.
    for arm in ("mamba3-pd", "native-pd"):
        assert arm in module._PRE_NORM_ARMS
        assert all(
            module._treatment_mixer(arm, index).output_norm for index in module.RECURRENT_LAYERS
        )


def test_mamba_b3_faithful_rerun_spec_is_exactly_the_three_mamba_cells():
    """The faithful-Mamba wave must rerun mamba-b3 only, under the unchanged V2 recipe.

    Same seeds, optimizer flags, token budget, and hardware shape as cells 0-2 of the V2 wave --
    only the faithful arm architecture and the code commit differ, which is what makes these
    three rows a replacement for the V2 mamba-b3 rows rather than a separately-tuned follow-up.
    The init seeds are exactly the ledger's mamba-b3 seeds for the first three data seeds.
    """
    module = load_entrypoint()
    assert MAMBA_FAITHFUL_RERUN_CONFIG.is_file()
    rerun_yaml = MAMBA_FAITHFUL_RERUN_CONFIG.read_text()

    arrays = {}
    for name in ("ARMS", "DSEEDS", "ISEEDS"):
        match = re.search(rf"{name}=\((.*?)\) &&", rerun_yaml, flags=re.DOTALL)
        assert match is not None, name
        arrays[name] = match.group(1).split()

    assert arrays["ARMS"] == ["mamba-b3", "mamba-b3", "mamba-b3"]
    assert arrays["DSEEDS"] == ["210007", "220014", "230021"]
    # The exact ledger seeds for mamba-b3 at the first three data seeds, not a hand-typed copy.
    assert arrays["ISEEDS"] == [
        str(module.INIT_SEEDS_BY_ARM["mamba-b3"][int(s)]) for s in arrays["DSEEDS"]
    ]
    assert arrays["ISEEDS"] == ["110007", "120014", "130021"]
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in rerun_yaml
    assert ".edullm/train_core6_arm.py" in rerun_yaml
    assert "--sequence-length 4096" in rerun_yaml
    assert "--steps 1144" in rerun_yaml
    assert "--warmup-steps 114" in rerun_yaml
    assert "--learning-rate 3e-4" in rerun_yaml
    assert "--global-batch-size 524288" in rerun_yaml
    assert "--rank-microbatch-size 8192" in rerun_yaml
    assert "--save-interval 572" in rerun_yaml
    assert "--param-dtype bfloat16" in rerun_yaml


def test_single_platform_image_bundles_all_five_accelerated_backends():
    dockerfile = DOCKERFILE.read_text()
    for pin in ("xlstm==2.0.5", "mlstm-kernels==2.0.4", "flashrnn==1.0.6"):
        assert pin in dockerfile
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "mamba3_siso_combined" in dockerfile
    assert "olmo_xlstm" in dockerfile
    assert "olmo_slstm" in dockerfile
    # `gdn` is a comparison arm now, so FLA is a wave dependency and no longer only a
    # smoke-time one. Its absence would strand twelve of the twenty-four cells rather than
    # three: `kda` and `kda-gconv` run on `fla`'s KDA kernels out of the same pin, and only
    # `kda-hh-r2` is an in-tree kernel with no `fla` counterpart.
    assert '"flash-linear-attention==0.5.1"' in dockerfile
    assert "from fla.ops.gdn2 import chunk_gdn2" in dockerfile


def test_gdn_comparison_arm_is_exact_measured_gdn2():
    module = load_entrypoint()
    from olmo_core.nn.attention import (
        AttentionConfig,
        GatedDeltaNet2,
        GatedDeltaNet2Config,
    )

    # `gdn` is a full arm of the wave now, not a diagnostic key alongside it. There is one
    # arm list and one parameter ledger, and `gdn` is in both.
    assert not hasattr(module, "DIAGNOSTIC_ARMS")
    assert not hasattr(module, "DIAGNOSTIC_PARAMETER_COUNTS")
    assert module.RUNNABLE_ARMS == module.ARMS
    assert module.EXACT_PARAMETER_COUNTS["gdn"] == 390_119_360
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
    assert config.num_params == module.EXACT_PARAMETER_COUNTS["gdn"]
    assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE


def test_the_three_kda_arms_are_the_verified_mixers_at_the_frozen_head_geometry():
    """Each KDA arm's twelve slots hold the exact config the ported mixer tests verified.

    THE PER-LAYER COUNTS ARE THE JOIN. ``src/test/nn/attention/kda_test.py`` pins the same three
    integers against the built modules, so if this file and that one ever describe different
    operators the pair of them says so. Pinning only the model total would not: the solver moves
    twelve FFN widths to absorb whatever the mixer costs, so a mixer that changed by a few
    thousand parameters would produce a model total that still landed inside the tolerance and
    still looked deliberate.
    """
    module = load_entrypoint()
    from olmo_core.nn.attention import (
        KimiDeltaAttentionConfig,
        KimiDeltaHouseholderConfig,
    )

    expected_per_layer = {"kda": 4_487_248, "kda-hh-r2": 6_608_976, "kda-gconv": 4_493_392}
    expected_types = {
        "kda": KimiDeltaAttentionConfig,
        "kda-hh-r2": KimiDeltaHouseholderConfig,
        "kda-gconv": KimiDeltaAttentionConfig,
    }

    for arm, per_layer in expected_per_layer.items():
        mixer = module._treatment_mixer(arm, module.RECURRENT_LAYERS[0])
        assert type(mixer) is expected_types[arm], arm
        assert (mixer.n_heads, mixer.head_dim) == (16, 64), arm
        assert mixer.num_params(module.D_MODEL) == per_layer, arm

        config = module.build_model_config(arm, module.valid_init_seeds(arm)[0])
        recurrent = [
            config.resolved_block_configs[index].sequence_mixer for index in module.RECURRENT_LAYERS
        ]
        assert len(recurrent) == 12, arm
        assert all(type(m) is expected_types[arm] for m in recurrent), arm
        assert all(m.num_params(module.D_MODEL) == per_layer for m in recurrent), arm
        assert config.num_params == module.EXACT_PARAMETER_COUNTS[arm], arm
        assert abs(config.num_params - module.PARAMETER_TARGET) <= module.PARAMETER_TOLERANCE, arm

    # `kda-gconv` MINUS `kda` IS THE GATE AND NOTHING ELSE. Both are the same class at the same
    # head geometry, so the whole per-layer difference has to be the three depthwise gates:
    # 2 * (1024 + 1024 + 1024) = 6,144 parameters, about 0.14% of the layer. If this ever grows
    # into the dense-projection regime the contrast stops isolating the mechanism and starts
    # confounding it with capacity.
    gated = module._treatment_mixer("kda-gconv", 0)
    assert gated.gate_params(module.D_MODEL) == 6_144
    assert expected_per_layer["kda-gconv"] - expected_per_layer["kda"] == 6_144
    assert module._treatment_mixer("kda", 0).gate_params(module.D_MODEL) == 0


def test_xlstm_mlstm_chunk_size_avoids_sparse_state_backward_indexing():
    """The pinned XL-chunk backward must save every inter-chunk state.

    ``mlstm-kernels==2.0.4`` always advances its recurrent backward over 128-token
    inter-chunks. At a 256-token target chunk the forward saves only every second state,
    while the backward still indexes every inter-chunk slot; at sequence length 4096 that
    means allocating 17 max-state slots and reading through slot 32. Chunk 128 makes the
    inter- and intra-chunks identical, so every state the backward indexes is present.
    """
    module = load_entrypoint()
    mixer = module._mlstm_mixer()

    assert mixer.chunkwise_kernel == "chunkwise--triton_xl_chunk"
    assert mixer.chunk_size == 128
    assert module.SEQUENCE_LENGTH % mixer.chunk_size == 0


def test_native_pd_chunk_size_is_64_and_the_change_shaped_no_weights():
    """The measured chunk size, and proof that moving it did not move a single parameter.

    A chunk size blocks the scan; it is not a shape of any weight. That is the reason the
    ledger below did not have to be re-solved when the constant moved from 128 to 64, and it is
    exactly the kind of claim that is cheap to assert and expensive to assume -- a chunk that
    did size a buffer would have changed the arm's parameter count, forced new FFN widths, and
    made `native-pd`'s cells incomparable with the four-arm study already described off them.
    """
    module = load_entrypoint()
    from dataclasses import replace
    from inspect import getsource

    mixers = [module._treatment_mixer("native-pd", index) for index in module.RECURRENT_LAYERS]
    assert all(mixer.chunk_size == 64 for mixer in mixers)
    assert all(mixer.fuse_input_projections is False for mixer in mixers)
    # This speed-sensitive choice belongs in the saved arm config, not in a library default that
    # could drift invisibly for every caller.
    assert "fuse_input_projections=False" in getsource(module._treatment_mixer)

    at_128 = replace(mixers[0], chunk_size=128)
    assert at_128.num_params(module.D_MODEL) == mixers[0].num_params(module.D_MODEL)

    # And the arm's solved widths are the ones the four-arm study was described with,
    # unchanged. The exact total is 768 higher than that study's, which is the output norm's
    # one gain a head-width across twelve layers and nothing else.
    assert module.solve_widths("native-pd") == (2432,) * 6 + (2400,) * 6
    assert module.EXACT_PARAMETER_COUNTS["native-pd"] == 390_143_744
    config = module.build_model_config("native-pd", module.valid_init_seeds("native-pd")[0])
    assert config.num_params == 390_143_744
    assert config.num_params - 390_142_976 == 12 * 64


def test_siso_pd_projection_layout_is_explicitly_unfused():
    """The measured SISO layout is recorded in the arm, not inherited from a library default."""
    module = load_entrypoint()
    from inspect import getsource

    mixers = [module._treatment_mixer("mamba3-pd", index) for index in module.RECURRENT_LAYERS]
    assert all(mixer.fuse_input_projections is False for mixer in mixers)

    source = getsource(module._treatment_mixer)
    siso_section = source.split('if arm == "mamba3-pd":', 1)[1]
    assert "fuse_input_projections=False" in siso_section

    config = module.build_model_config("mamba3-pd", module.valid_init_seeds("mamba3-pd")[0])
    assert config.num_params == 390_170_432


def test_both_pd_arms_normalize_the_readout_before_the_gate():
    """Every arm now carries a gated output norm; the two PD arms were the last without one."""
    module = load_entrypoint()
    from dataclasses import replace

    for arm in ("native-pd", "mamba3-pd"):
        mixers = [module._treatment_mixer(arm, index) for index in module.RECURRENT_LAYERS]
        assert all(mixer.output_norm for mixer in mixers), arm
        # Head-wise, so one loud head cannot rescale the others: one gain per d_state, not
        # per d_model, and twelve layers of it is the whole parameter delta.
        assert all(mixer.norm_eps > 0 for mixer in mixers), arm
        delta = sum(
            mixer.num_params(module.D_MODEL)
            - replace(mixer, output_norm=False).num_params(module.D_MODEL)
            for mixer in mixers
        )
        assert delta == len(module.RECURRENT_LAYERS) * 64, arm


def test_the_native_pd_output_norm_is_head_wise_and_bounds_the_readout():
    """The norm reduces within a head, and the published block without it is dead at init."""
    module = load_entrypoint()
    import torch
    from dataclasses import replace as _replace

    from olmo_core.nn.transformer.init import InitMethod

    d_model, n_heads, d_state = 256, 4, 64
    on = _replace(
        module._treatment_mixer("native-pd", module.RECURRENT_LAYERS[0]),
        n_heads=n_heads,
        d_state=d_state,
        backend=module.NativePDBackend.REFERENCE,
    )
    off = _replace(on, output_norm=False)

    def build(config):
        mixer = config.build(d_model, layer_idx=0, n_layers=2)
        generator = torch.Generator().manual_seed(9)
        with torch.no_grad():
            mixer.init_weights(
                init_method=InitMethod.normal,
                d_model=d_model,
                block_idx=0,
                num_blocks=2,
                generator=generator,
            )
        return mixer

    mixer_on, mixer_off = build(on), build(off)
    assert mixer_on.load_state_dict(mixer_off.state_dict(), strict=False).missing_keys == [
        "output_norm_weight"
    ]

    captured: dict = {}

    def capture(tag):
        return lambda module_, inputs, output: captured.__setitem__(tag, inputs[0].detach())

    torch.manual_seed(0)
    x = torch.randn(2, 8, d_model)
    for tag, mixer in (("off", mixer_off), ("on", mixer_on)):
        handle = mixer.out_proj.register_forward_hook(capture(tag))
        mixer(x)
        handle.remove()

    # The scale the norm applied, per element. Constant inside a head, different between
    # heads: that is what makes it head-wise rather than a single norm over d_model.
    scale = (captured["on"] / captured["off"]).view(2, 8, n_heads, d_state)
    assert (scale.amax(dim=-1) - scale.amin(dim=-1)).abs().max() < 1e-2
    assert scale[..., 0].std(dim=-1).mean() > 1e-3

    # Unnormalized, the readout entering out_proj is ~0 at init and grows superlinearly.
    # Normalized, it carries signal immediately.
    def readout_rms(mixer, tag, multiplier):
        handle = mixer.out_proj.register_forward_hook(capture(tag))
        mixer(x * multiplier)
        handle.remove()
        return float(captured[tag].square().mean().sqrt())

    assert readout_rms(mixer_off, "off", 1.0) < 0.01 < readout_rms(mixer_on, "on", 1.0)
    growth_off = readout_rms(mixer_off, "off", 100.0) / max(
        readout_rms(mixer_off, "off", 10.0), 1e-9
    )
    growth_on = readout_rms(mixer_on, "on", 100.0) / max(readout_rms(mixer_on, "on", 10.0), 1e-9)
    assert growth_on < growth_off


def test_every_arm_declares_one_fp32_master_dtype_and_keeps_its_bf16_kernels():
    module = load_entrypoint()
    from olmo_core.config import DType
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.flash_pd_native import NativePDBackend, NativePDMode

    assert module.RUNNABLE_ARMS == (
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    )
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
    for arm in ("native-pd", "mamba3-pd"):
        mixer = module._treatment_mixer(arm, 0)
        assert mixer.backend == NativePDBackend.CUDA
        assert mixer.mode == NativePDMode.GENERAL_SCATTER
    assert module._treatment_mixer("kda-hh-r2", 0).backend == "triton"


@pytest.mark.parametrize(
    "arm",
    [
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    ],
)
def test_every_built_block_holds_exactly_one_fp32_parameter_dtype(arm):
    module = load_entrypoint()
    import torch

    assert arm in module.RUNNABLE_ARMS
    expected = module.EXACT_PARAMETER_COUNTS[arm]
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


@pytest.mark.parametrize(
    "arm",
    [
        "mamba-b3",
        "xlstm",
        "mamba3-pd",
        "native-pd",
        "gdn",
        "kda",
        "kda-hh-r2",
        "kda-gconv",
    ],
)
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
