"""
Train the parameter-matched Mamba-3 + Flash PD 370M hybrid.

This is an additive wrapper around ``OLMo3-370M-mamba3.py``. It reuses that script's data,
optimizer, precision, checkpoint, and trainer wiring unchanged, then replaces every recurrent
slot with one fused Mamba-3 + Flash-PD mixer:

``[Mamba3FlashPD, Mamba3FlashPD, Mamba3FlashPD, attention]``.

The fused mixer combines Flash-PD sparse transitions with Mamba-3 complex phase,
exponential-trapezoidal discretization, and MIMO projections. This file does not submit a run.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import NoReturn, Optional, Sequence

from olmo_core.nn.flash_pd_ssm import (
    Mamba3FlashPDSSMMixer,
    replace_mamba3_with_fused_flash_pd,
)
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.train.train_module import TransformerActivationCheckpointingConfig

_BASE_PATH = Path(__file__).with_name("OLMo3-370M-mamba3.py")
_SPEC = importlib.util.spec_from_file_location("olmo3_370m_mamba3_base", _BASE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = base
_SPEC.loader.exec_module(base)

DEFAULT_RANK_MICROBATCH_SIZE = 8192
DEFAULT_REGIONAL_COMPILE = True
DEFAULT_FUSED_CE = True
DEFAULT_ACTIVATION_CHECKPOINTING = False
DEFAULT_FP8_RECIPE = "off"


_MAMBA_ONLY_OPTIONS = (
    "--model-factory",
    "--rotation-block-size",
    "--d-state",
    "--rotation-scan-impl",
    "--theta-max",
    "--a-log-init-min",
    "--a-log-init-max",
)


def with_integration_defaults(argv: Sequence[str]) -> list[str]:
    """Reject separate-Mamba options that cannot affect the fused mixer."""
    args = list(argv)
    for argument in args:
        option = argument.split("=", 1)[0]
        if option in _MAMBA_ONLY_OPTIONS:
            raise SystemExit(
                f"{option} is not valid for fused Mamba-3 + Flash-PD; configure the fused "
                "mixer instead"
            )
    return args


def _has_option(argv: Sequence[str], name: str) -> bool:
    return any(argument == name or argument.startswith(name + "=") for argument in argv)


def _pop_boolean_choice(
    argv: Sequence[str],
    enabled_option: str,
    disabled_option: str,
    *,
    default: bool,
) -> tuple[bool, list[str]]:
    """Consume an auditable on/off option pair and reject ambiguous requests."""
    enabled = 0
    disabled = 0
    rest = []
    for argument in argv:
        if argument == enabled_option:
            enabled += 1
        elif argument == disabled_option:
            disabled += 1
        elif argument.startswith(enabled_option + "=") or argument.startswith(
            disabled_option + "="
        ):
            raise SystemExit(f"{enabled_option} and {disabled_option} are valueless flags")
        else:
            rest.append(argument)
    if enabled and disabled:
        raise SystemExit(f"{enabled_option} conflicts with {disabled_option}")
    if enabled > 1 or disabled > 1:
        option = enabled_option if enabled else disabled_option
        raise SystemExit(f"{option} was specified more than once")
    return (True if enabled else False if disabled else default), rest


def _show_help() -> NoReturn:
    """Show fused speed options without advertising rejected Mamba-only flags."""
    print(
        "[OLMo3-370M-mamba3-flash-pd] Fused speed options:\n"
        "  --regional-compile / --no-regional-compile\n"
        "      Request per-block compile; runtime skips it when CUDA is unavailable.\n"
        "  --fused-ce / --no-fused-ce\n"
        "      Use bundled Liger fused linear cross-entropy (default: on; returns no logits).\n"
        "  --activation-checkpointing / --no-activation-checkpointing\n"
        "      Recompute feed-forward modules in backward (default: off).\n"
        f"  --rank-microbatch-size defaults to {DEFAULT_RANK_MICROBATCH_SIZE} tokens; "
        "larger values require activation checkpointing and fused CE.\n"
        f"  --fp8 defaults to {DEFAULT_FP8_RECIPE}; explicit recipes protect recurrence "
        "projections.\n"
        "Shared data, optimizer, checkpoint, and trainer options are documented by "
        "OLMo3-370M-dolma2mix.py --help.\n"
    )
    raise SystemExit(0)


def preflight_liger_fused_ce() -> None:
    """Fail before config construction when the requested fused loss is unavailable."""
    module_name = "liger_kernel.ops.fused_linear_cross_entropy"
    try:
        module = importlib.import_module(module_name)
        getattr(module, "LigerFusedLinearCrossEntropyFunction")
    except Exception as error:
        raise SystemExit(
            "Liger fused linear cross-entropy was requested but its kernel is unavailable; "
            "use the bundled training image or pass --no-fused-ce"
        ) from error


def validate_memory_options(opts) -> None:
    """Require named memory savers before exceeding the conservative default."""
    rank_microbatch_size = getattr(opts, "rank_microbatch_size", DEFAULT_RANK_MICROBATCH_SIZE)
    if rank_microbatch_size <= DEFAULT_RANK_MICROBATCH_SIZE:
        return
    if not getattr(opts, "activation_checkpointing", False):
        raise SystemExit(
            f"--rank-microbatch-size above {DEFAULT_RANK_MICROBATCH_SIZE} requires "
            "--activation-checkpointing; this does not claim the larger shape will fit"
        )
    if not getattr(opts, "fused_ce", False):
        raise SystemExit(
            f"--rank-microbatch-size above {DEFAULT_RANK_MICROBATCH_SIZE} requires --fused-ce; "
            "this does not claim the larger shape will fit"
        )


def integrate_flash_pd(config: Mamba3Config) -> Mamba3Config:
    """Replace Mamba slots with the parameter-matched fused mixer without mutating ``config``."""
    return replace_mamba3_with_fused_flash_pd(config)


def fused_fp8_modules_to_ignore(model_config: Mamba3Config) -> list[str]:
    """Find recurrence-defining fused projections that must remain high precision."""
    model = model_config.build(init_device="meta")
    ignored = []
    for name, module in model.named_modules():
        if isinstance(module, Mamba3FlashPDSSMMixer):
            ignored.extend((f"{name}.bc_proj", f"{name}.dynamics_proj"))
    return sorted(ignored)


def remove_mamba3_sentinel(config) -> None:
    """Remove the callback whose Mamba-only preflight rejects the fused model."""
    config.trainer.callbacks.pop("mamba3_sentinel", None)


_BASE_PARSE_ARGS = base.parse_args
_BASE_BUILD_MODEL_CONFIG = base.build_mamba_model_config
_BASE_BUILD_CONFIG = base.build_config


def _parse_args(argv: Optional[Sequence[str]] = None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(argument in ("-h", "--help") for argument in args):
        _show_help()

    args = with_integration_defaults(args)
    regional_compile, args = _pop_boolean_choice(
        args,
        "--regional-compile",
        "--no-regional-compile",
        default=DEFAULT_REGIONAL_COMPILE,
    )
    fused_ce, args = _pop_boolean_choice(
        args,
        "--fused-ce",
        "--no-fused-ce",
        default=DEFAULT_FUSED_CE,
    )
    activation_checkpointing, args = _pop_boolean_choice(
        args,
        "--activation-checkpointing",
        "--no-activation-checkpointing",
        default=DEFAULT_ACTIVATION_CHECKPOINTING,
    )
    if not _has_option(args, "--rank-microbatch-size"):
        args.extend(("--rank-microbatch-size", str(DEFAULT_RANK_MICROBATCH_SIZE)))
    if not _has_option(args, "--fp8"):
        args.extend(("--fp8", DEFAULT_FP8_RECIPE))
    if fused_ce:
        args.append("--fused-ce")
    if activation_checkpointing:
        args.append("--activation-checkpointing")

    opts, overrides = _BASE_PARSE_ARGS(args)
    opts.regional_compile = regional_compile
    opts.fused_ce = fused_ce
    opts.activation_checkpointing = activation_checkpointing
    validate_memory_options(opts)
    return opts, overrides


def _build_model_config(opts):
    config, _ = _BASE_BUILD_MODEL_CONFIG(opts)
    integrated = integrate_flash_pd(config)
    expected_pattern = [
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "attn",
    ]
    if integrated.block_pattern != expected_pattern:
        raise RuntimeError(f"unexpected integrated block pattern: {integrated.block_pattern}")
    assert isinstance(integrated.block, dict)
    fused = integrated.block["mamba3_flash_pd"].sequence_mixer
    return integrated, fused.d_state


def _build_config(opts, overrides):
    validate_memory_options(opts)
    fused_ce = getattr(opts, "fused_ce", False)
    activation_checkpointing = getattr(opts, "activation_checkpointing", False)
    if fused_ce:
        preflight_liger_fused_ce()

    config = _BASE_BUILD_CONFIG(opts, overrides)
    remove_mamba3_sentinel(config)

    if hasattr(opts, "regional_compile"):
        config.train_module.compile_model = opts.regional_compile
    if hasattr(opts, "fused_ce"):
        config.model.lm_head.loss_implementation = (
            LMLossImplementation.fused_linear if fused_ce else LMLossImplementation.default
        )
    if hasattr(opts, "activation_checkpointing"):
        config.train_module.ac_config = (
            TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_modules,
                modules=["blocks.*.feed_forward"],
            )
            if activation_checkpointing
            else None
        )

    recipe = getattr(opts, "fp8", DEFAULT_FP8_RECIPE)
    if recipe == "off":
        config.train_module.float8_config = None
    else:
        ignored = fused_fp8_modules_to_ignore(config.model)
        config.train_module.float8_config = base.make_float8_config(recipe, ignored)
    return config


base.parse_args = _parse_args
base.build_mamba_model_config = _build_model_config
base.build_config = _build_config


def main() -> None:
    """Run the existing Mamba training entrypoint with the integrated model config."""
    base.main()


if __name__ == "__main__":
    main()
