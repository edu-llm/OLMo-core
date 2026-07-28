"""
Tests for the separate Mamba-3 370M training script (``OLMo3-370M-mamba3.py``).

The script exists so the Mamba-3 ablation runs from its own entrypoint -- with the fp8 recipe and
the larger microbatch baked in as defaults -- without touching the live dense ``dolma2mix`` script,
whose config builder and training loop it reuses verbatim.

Everything here is offline: no S3 (the dataset is never built), no GPU, no distributed. The fp8
recipe is exercised on the real 370M architecture via a ``meta`` build, which allocates nothing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from olmo_core.nn.mamba3 import Mamba3Config, mamba3_modules_to_ignore_for_fp8

SCRIPT_PATH = Path("src/scripts/train/OLMo3/OLMo3-370M-mamba3.py")


@pytest.fixture(scope="module")
def script() -> ModuleType:
    assert SCRIPT_PATH.exists(), f"{SCRIPT_PATH} not found; run pytest from the repo root"
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("olmo3_370m_mamba3", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


# ------------------------------------------------------------------------------------------
# The fp8 recipe builder
# ------------------------------------------------------------------------------------------


def test_make_float8_config_mxfp8_carries_the_mx_config_and_ignore_list(script):
    """MXFP8 is the B200-native recipe; the ignore list must pass through untouched."""
    ignore = ["blocks.0.sequence_mixer.theta_proj", "blocks.0.sequence_mixer.dt_proj"]
    cfg = script.make_float8_config("mxfp8", ignore)

    assert cfg is not None and cfg.enabled
    assert cfg.ao_mx is not None, "mxfp8 must set the MX linear config"
    assert cfg.ao is None and cfg.ao_recipe is None, "mxfp8 must not also set a scaled-mm recipe"
    assert list(cfg.modules_to_ignore) == ignore


def test_make_float8_config_rowwise_uses_a_scaled_mm_recipe(script):
    """
    ``rowwise`` is the recipe used for *local* validation: MXFP8 needs SM100 (B200), rowwise runs
    on any fp8-capable GPU, and its accuracy is a conservative lower bound on MXFP8's.
    """
    cfg = script.make_float8_config("rowwise", ["m.x"])

    assert cfg is not None and cfg.enabled
    assert cfg.ao_recipe is not None, "rowwise must set an ao recipe"
    assert cfg.ao_mx is None
    assert list(cfg.modules_to_ignore) == ["m.x"]


def test_make_float8_config_off_is_none(script):
    """``off`` must produce no config at all, so the train module trains in bf16."""
    assert script.make_float8_config("off", []) is None


# ------------------------------------------------------------------------------------------
# The fp8 config derived from the real 370M model
# ------------------------------------------------------------------------------------------


def test_fp8_config_for_the_370M_ignores_exactly_the_ssm_projections(script):
    """
    The recipe applied to the real preset must protect every SSM projection and nothing else.

    Built on ``meta`` so it allocates nothing. The ignore list has to equal the helper's output on
    the same model -- that equality is what guarantees ``apply_float8_linear`` (which errors on an
    unmatched ignore name) will not crash the run.
    """
    model_config = Mamba3Config.mamba3_olmo3_370M(vocab_size=100352)
    meta_model = model_config.build(init_device="meta")
    expected = sorted(mamba3_modules_to_ignore_for_fp8(meta_model))

    cfg = script.mamba3_fp8_config_for_model(model_config, "mxfp8")

    assert cfg is not None
    assert sorted(cfg.modules_to_ignore) == expected
    assert len(expected) > 0, "the 370M hybrid must have SSM projections to protect"
    for fqn in cfg.modules_to_ignore:
        assert fqn.rsplit(".", 1)[-1] not in ("in_x", "in_z", "out_proj")


def test_fp8_config_for_model_off_returns_none(script):
    model_config = Mamba3Config.mamba3_olmo3_370M(vocab_size=100352)
    assert script.mamba3_fp8_config_for_model(model_config, "off") is None


# ------------------------------------------------------------------------------------------
# Defaults: this script is the NC^1 arm, fed harder, in fp8
# ------------------------------------------------------------------------------------------


def test_defaults_are_the_nc1_mamba_arm_fed_harder_in_fp8(script):
    """
    The whole point of a separate script is that its *defaults* differ from the dense one.

    Model factory is the Mamba-3 hybrid, the rotation block size is the NC^1 arm (b=3), fp8 is on,
    and the per-rank microbatch is larger than the dense script's so the small 370M is not left
    memory-underfed on a B200.
    """
    opts, _ = script.parse_args(["my-run", "--dry-run"])

    # The Mamba preset is carried in `mamba_factory`; `model_factory` is intentionally left as the
    # dense default so the pristine (now dense-only) dense builder still produces a valid config,
    # and the Mamba model is swapped in by build_config. This is what keeps the dense script
    # untouched even though it no longer knows about Mamba factories.
    assert opts.mamba_factory == "mamba3_olmo3_370M"
    assert opts.model_factory == "olmo3_370M"
    assert opts.rotation_block_size == 3
    assert opts.fp8 == "mxfp8"
    assert opts.rank_microbatch_size == script.DEFAULT_RANK_MICROBATCH_SIZE
    assert script.DEFAULT_RANK_MICROBATCH_SIZE > script.dolma2.DEFAULT_RANK_MICROBATCH_SIZE


def test_weight_decay_exemption_survives_resuming_an_older_checkpoint(script):
    """
    Regression: the exemption has to outlive a checkpoint written before it existed.

    The flattened optimizer state stores ``weight_decay`` per parameter, so loading a checkpoint
    from a run that decayed ``A_log``/``dt_bias`` restores that non-zero value onto the new
    exempt group. Nothing errors -- the run simply carries on decaying the timescales while the
    config says otherwise. Pinning the field in ``fixed_fields`` makes the config win on load.
    """
    from olmo_core.optim import AdamWConfig

    model = Mamba3Config.mamba3_olmo3_370M(vocab_size=100352).build(init_device="meta")
    optim = AdamWConfig(lr=1e-3, weight_decay=0.1)
    n = script.exempt_timescale_params_from_weight_decay(optim, model)

    assert n > 0, "nothing was exempted"
    exempt = [go for go in optim.group_overrides if go.opts.get("weight_decay") == 0.0]
    assert any(
        p.endswith(".A_log") for go in exempt for p in go.params
    ), "no group exempts the SSM timescale parameters"
    assert "weight_decay" in optim.fixed_fields, (
        "weight_decay is not pinned, so a resumed checkpoint will silently reinstate decay on "
        f"A_log/dt_bias; fixed_fields={optim.fixed_fields}"
    )


def test_explicit_flags_override_the_mamba_defaults(script):
    """A user must still be able to run the b=2 baseline or turn fp8 off from this same script."""
    opts, _ = script.parse_args(
        ["my-run", "--dry-run", "--rotation-block-size", "2", "--fp8", "off"]
    )
    assert opts.rotation_block_size == 2
    assert opts.fp8 == "off"


# ------------------------------------------------------------------------------------------
# --rotation-scan-impl
#
# The scan used to be selectable only through MAMBA3_ROTATION_SCAN_IMPL, read once at import in
# `mamba3_ssd_fast`. It therefore never entered the saved config and was never logged, and a
# relaunch in a shell that lost the export silently fell back to `chunked` -- 33,468 tok/s against
# `quaternion`'s 75,040, with nothing raising. These pin the flag that replaces it.
# ------------------------------------------------------------------------------------------


def test_rotation_scan_impl_defaults_to_unset(script):
    """Unset must stay unset rather than being resolved here, so the env var still decides."""
    opts, _ = script.parse_args(["my-run", "--dry-run"])
    assert opts.rotation_scan_impl is None


@pytest.mark.parametrize("form", ["space", "equals"])
def test_rotation_scan_impl_reaches_the_built_model_config(script, form: str):
    """
    End to end: the flag has to arrive on the Mamba block's mixer config, which is the object
    that gets serialized into the checkpoint.
    """
    flag = (
        ["--rotation-scan-impl", "quaternion"]
        if form == "space"
        else ["--rotation-scan-impl=quaternion"]
    )
    opts, _ = script.parse_args(["my-run", "--dry-run", *flag])
    assert opts.rotation_scan_impl == "quaternion"

    model_config, _ = script.build_mamba_model_config(opts)
    assert script._rotation_scan_impl_of(model_config) == "quaternion"


def test_rotation_scan_impl_is_normalised(script):
    """Hand-typed on a command line, so case and stray whitespace must not change the meaning."""
    opts, _ = script.parse_args(["my-run", "--dry-run", "--rotation-scan-impl", " Chunked "])
    assert opts.rotation_scan_impl == "chunked"


def test_rotation_scan_impl_rejects_a_typo_before_anything_is_built(script):
    """
    A typo must not be read as "keep the environment default". That is the exact failure this
    flag exists to prevent, so silently accepting it would be worse than not having the flag.
    """
    with pytest.raises(SystemExit, match="quaternion"):
        script.parse_args(["my-run", "--dry-run", "--rotation-scan-impl", "quarternion"])
