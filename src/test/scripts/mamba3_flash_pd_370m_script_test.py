import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from olmo_core.nn.flash_pd_ssm import Mamba3FlashPDSSMMixerConfig
from olmo_core.nn.mamba3 import Mamba3Config

SCRIPT = Path("src/scripts/train/OLMo3/OLMo3-370M-mamba3-flash-pd.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("mamba3_flash_pd_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_integrates_parameter_matched_flash_slots_without_mutating_base():
    module = _load_script()
    base = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    before = base.as_config_dict()

    integrated = module.integrate_flash_pd(base)

    assert base.as_config_dict() == before
    assert integrated.block_pattern == [
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "attn",
    ]
    assert isinstance(integrated.block, dict)
    fused = integrated.block["mamba3_flash_pd"].sequence_mixer
    assert isinstance(fused, Mamba3FlashPDSSMMixerConfig)
    assert (fused.n_heads, fused.d_state, fused.mimo_rank, fused.dictionary_size) == (
        8,
        20,
        4,
        16,
    )


def test_script_does_not_inject_obsolete_separate_mamba_rotation_defaults():
    module = _load_script()

    assert module.with_integration_defaults(["run"]) == ["run"]


def test_training_image_pins_and_builds_required_mamba3_kernel():
    dockerfile = Path("src/Dockerfile").read_text()

    assert "e9594ce1c732d97440f0332fdc43170a2294dbfa" in dockerfile
    assert "MAMBA_FORCE_BUILD=TRUE" in dockerfile
    assert "--no-build-isolation" in dockerfile


def test_wrapper_removes_mamba_sentinel_and_protects_fused_recurrence_projections():
    module = _load_script()
    base = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    integrated = module.integrate_flash_pd(base)
    ignored = module.fused_fp8_modules_to_ignore(integrated)

    assert len(ignored) == 24
    assert all(name.endswith(".bc_proj") or name.endswith(".dynamics_proj") for name in ignored)

    config = SimpleNamespace(
        trainer=SimpleNamespace(callbacks={"mamba3_sentinel": object(), "keep": object()})
    )
    module.remove_mamba3_sentinel(config)
    assert "mamba3_sentinel" not in config.trainer.callbacks
    assert "keep" in config.trainer.callbacks


def test_build_config_replaces_mamba_callback_and_fp8_policy(monkeypatch):
    module = _load_script()
    model = module.integrate_flash_pd(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))
    config = SimpleNamespace(
        model=model,
        trainer=SimpleNamespace(callbacks={"mamba3_sentinel": object()}),
        train_module=SimpleNamespace(float8_config=object()),
    )
    monkeypatch.setattr(module, "_BASE_BUILD_CONFIG", lambda opts, overrides: config)
    seen = {}

    def make_float8(recipe, ignored):
        seen["recipe"] = recipe
        seen["ignored"] = ignored
        return "fused-fp8-config"

    monkeypatch.setattr(module.base, "make_float8_config", make_float8)

    built = module._build_config(SimpleNamespace(fp8="rowwise"), [])

    assert built is config
    assert "mamba3_sentinel" not in built.trainer.callbacks
    assert built.train_module.float8_config == "fused-fp8-config"
    assert seen["recipe"] == "rowwise"
    assert len(seen["ignored"]) == 24


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--rotation-block-size", "2"],
        ["run", "--d-state=96"],
        ["run", "--rotation-scan-impl", "quaternion"],
        ["run", "--theta-max", "0.01"],
    ],
)
def test_wrapper_rejects_obsolete_mamba_only_options(args):
    module = _load_script()

    with pytest.raises(SystemExit, match="not valid for fused Mamba-3.*Flash-PD"):
        module.with_integration_defaults(args)


def test_fused_entrypoint_speed_defaults_are_explicit_and_overridable():
    module = _load_script()
    opts, _ = module._parse_args(["run", "--dry-run"])
    assert opts.regional_compile is True
    assert opts.fused_ce is True
    assert opts.activation_checkpointing is False
    assert opts.fp8 == "off"
    assert opts.rank_microbatch_size == module.DEFAULT_RANK_MICROBATCH_SIZE == 8192

    opts, _ = module._parse_args(
        [
            "run",
            "--dry-run",
            "--no-regional-compile",
            "--no-fused-ce",
            "--no-activation-checkpointing",
            "--fp8",
            "rowwise",
            "--rank-microbatch-size",
            "4096",
        ]
    )
    assert opts.regional_compile is False
    assert opts.fused_ce is False
    assert opts.activation_checkpointing is False
    assert opts.fp8 == "rowwise"
    assert opts.rank_microbatch_size == 4096


def test_fused_help_omits_mamba_only_options(capsys):
    module = _load_script()
    with pytest.raises(SystemExit) as error:
        module._parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--no-regional-compile" in help_text
    assert "--no-fused-ce" in help_text
    for option in module._MAMBA_ONLY_OPTIONS:
        assert option not in help_text


def test_larger_microbatch_requires_both_memory_savers():
    module = _load_script()
    larger = str(module.DEFAULT_RANK_MICROBATCH_SIZE * 2)
    with pytest.raises(SystemExit, match="--activation-checkpointing"):
        module._parse_args(["run", "--dry-run", "--rank-microbatch-size", larger])
    with pytest.raises(SystemExit, match="--fused-ce"):
        module._parse_args(
            [
                "run",
                "--dry-run",
                "--rank-microbatch-size",
                larger,
                "--activation-checkpointing",
                "--no-fused-ce",
            ]
        )


def test_build_config_realizes_compile_loss_checkpoint_and_fp8_policy(monkeypatch):
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
    from olmo_core.train.train_module import TransformerActivationCheckpointingConfig

    module = _load_script()
    model = module.integrate_flash_pd(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))
    config = SimpleNamespace(
        model=model,
        trainer=SimpleNamespace(callbacks={"mamba3_sentinel": object()}),
        train_module=SimpleNamespace(
            ac_config=None,
            compile_model=False,
            float8_config=object(),
            rank_microbatch_size=module.DEFAULT_RANK_MICROBATCH_SIZE,
        ),
    )
    monkeypatch.setattr(module, "_BASE_BUILD_CONFIG", lambda opts, overrides: config)
    monkeypatch.setattr(module, "preflight_liger_fused_ce", lambda: None)
    monkeypatch.setattr(module.base, "make_float8_config", lambda recipe, ignored: "fp8")

    built = module._build_config(
        SimpleNamespace(
            activation_checkpointing=True,
            fp8="rowwise",
            fused_ce=True,
            rank_microbatch_size=module.DEFAULT_RANK_MICROBATCH_SIZE,
            regional_compile=True,
        ),
        [],
    )
    assert "mamba3_sentinel" not in built.trainer.callbacks
    assert built.train_module.compile_model is True
    assert built.model.lm_head.loss_implementation == LMLossImplementation.fused_linear
    assert built.train_module.float8_config == "fp8"
    assert built.train_module.ac_config == TransformerActivationCheckpointingConfig(
        mode=TransformerActivationCheckpointingMode.selected_modules,
        modules=["blocks.*.feed_forward"],
    )

    from olmo_core.train.train_module.transformer.common import parallelize_model

    source = inspect.getsource(parallelize_model)
    assert source.index("apply_activation_checkpointing") < source.index("apply_compile")
    assert source.index("apply_compile") < source.index("apply_fsdp")


@pytest.mark.parametrize("failure", [ImportError("missing"), RuntimeError("broken")])
def test_liger_preflight_fails_before_base_config_build(monkeypatch, failure):
    module = _load_script()
    monkeypatch.setattr(
        module.importlib, "import_module", lambda _name: (_ for _ in ()).throw(failure)
    )
    monkeypatch.setattr(
        module,
        "_BASE_BUILD_CONFIG",
        lambda *_args: pytest.fail("base config built before Liger preflight"),
    )
    opts = SimpleNamespace(
        activation_checkpointing=False,
        fp8="off",
        fused_ce=True,
        rank_microbatch_size=8192,
        regional_compile=True,
    )
    with pytest.raises(SystemExit, match="Liger.*fused linear cross-entropy"):
        module._build_config(opts, [])
