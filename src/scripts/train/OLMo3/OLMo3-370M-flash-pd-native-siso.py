"""
Train the parameter-matched Mamba-3-improved native SISO PD-SSM 370M hybrid.

This additive entrypoint reuses the existing OLMo3-370M Mamba script's data,
optimizer, checkpoint, precision, and trainer wiring. It replaces every
recurrent slot with the collision-capable native PD transition and does not
submit a run itself.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

from olmo_core.config import DType
from olmo_core.nn.flash_pd_native import (
    NativeFlashPDMamba3SISOMixer,
    NativeFlashPDMamba3SISOMixerConfig,
    NativePDBackend,
    NativePDMode,
)
from olmo_core.nn.mamba3 import Mamba3Config

_BASE_PATH = Path(__file__).with_name("OLMo3-370M-mamba3.py")
_SPEC = importlib.util.spec_from_file_location("olmo3_370m_native_siso_base", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

PARAMETER_MATCHED_RECURRENT_FF_DIM = 2128
"""SwiGLU width giving a 0.081% non-embedding parameter delta."""

_MAMBA_ONLY_OPTIONS = (
    "--model-factory",
    "--rotation-block-size",
    "--d-state",
    "--rotation-scan-impl",
    "--theta-max",
    "--a-log-init-min",
    "--a-log-init-max",
)


def native_mixer_config() -> NativeFlashPDMamba3SISOMixerConfig:
    """Return the strict production native SISO mixer configuration."""
    return NativeFlashPDMamba3SISOMixerConfig(
        n_heads=16,
        d_state=64,
        dictionary_size=16,
        chunk_size=64,
        dictionary_temperature=1.0,
        router_temperature=1.0,
        backend=NativePDBackend.CUDA,
        mode=NativePDMode.GENERAL_SCATTER,
        bc_norm=True,
        output_norm=False,
        fuse_input_projections=True,
        dtype=DType.bfloat16,
    )


def integrate_native_siso(config: Mamba3Config) -> Mamba3Config:
    """Replace every named recurrent slot without mutating the supplied config."""
    if not isinstance(config.block, dict) or config.block_pattern is None:
        raise ValueError("native SISO integration requires named blocks and a block pattern")
    if "mamba3" not in config.block:
        raise ValueError("base 370M config has no mamba3 recurrent block")
    integrated = config.copy()
    assert isinstance(integrated.block, dict)
    blocks = dict(integrated.block)
    recurrent = blocks["mamba3"]
    if recurrent.feed_forward is None:
        raise ValueError("370M recurrent block has no feed-forward module")
    feed_forward = recurrent.feed_forward.replace(hidden_size=PARAMETER_MATCHED_RECURRENT_FF_DIM)
    blocks["flash_pd_native_mamba3_siso"] = recurrent.replace(
        sequence_mixer=native_mixer_config(),
        feed_forward=feed_forward,
    )
    pattern = [
        "flash_pd_native_mamba3_siso" if name == "mamba3" else name
        for name in integrated.block_pattern
    ]
    return integrated.replace(block=blocks, block_pattern=pattern)


def parameter_match_report(
    baseline: Mamba3Config,
    integrated: Mamba3Config,
) -> dict[str, int | float]:
    """Report the exact non-embedding parameter delta against the 370M shell."""
    target = baseline.num_non_embedding_params
    actual = integrated.num_non_embedding_params
    difference = actual - target
    return {
        "target_non_embedding_parameters": target,
        "actual_non_embedding_parameters": actual,
        "difference": difference,
        "relative_difference": difference / target,
    }


def native_fp8_modules_to_ignore(model_config: Mamba3Config) -> list[str]:
    """Protect the fused mixed-sensitivity input projection from FP8."""
    model = model_config.build(init_device="meta")
    ignored = []
    for name, module in model.named_modules():
        if isinstance(module, NativeFlashPDMamba3SISOMixer):
            ignored.append(f"{name}.in_proj")
    return sorted(ignored)


def _reject_mamba_only_options(argv: Sequence[str]) -> list[str]:
    args = list(argv)
    for argument in args:
        option = argument.split("=", 1)[0]
        if option in _MAMBA_ONLY_OPTIONS:
            raise SystemExit(
                f"{option} configures a separate Mamba mixer and is invalid for native SISO PD"
            )
    return args


_BASE_PARSE_ARGS = base.parse_args
_BASE_BUILD_MODEL_CONFIG = base.build_mamba_model_config
_BASE_BUILD_CONFIG = base.build_config


def _parse_args(argv: Optional[Sequence[str]] = None):
    args = _reject_mamba_only_options(sys.argv[1:] if argv is None else argv)
    return _BASE_PARSE_ARGS(args)


def _build_model_config(opts):
    baseline, _ = _BASE_BUILD_MODEL_CONFIG(opts)
    integrated = integrate_native_siso(baseline)
    report = parameter_match_report(baseline, integrated)
    if abs(float(report["relative_difference"])) >= 0.001:
        raise RuntimeError(f"370M parameter match drifted: {report}")
    return integrated, native_mixer_config().d_state


def _build_config(opts, overrides):
    config = _BASE_BUILD_CONFIG(opts, overrides)
    config.trainer.callbacks.pop("mamba3_sentinel", None)
    recipe = getattr(opts, "fp8", "off")
    if recipe == "off":
        config.train_module.float8_config = None
    else:
        config.train_module.float8_config = base.make_float8_config(
            recipe,
            native_fp8_modules_to_ignore(config.model),
        )
    return config


base.parse_args = _parse_args
base.build_mamba_model_config = _build_model_config
base.build_config = _build_config


def main() -> None:
    """Run the existing training entrypoint with the native SISO model config."""
    base.main()


if __name__ == "__main__":
    main()
