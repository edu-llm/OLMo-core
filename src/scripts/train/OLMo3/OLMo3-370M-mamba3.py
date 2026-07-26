"""
Pre-train the **370M Mamba-3 hybrid** on the edu-llm dolma2 source mixture (AWS / torchrun).

This is the Mamba-3 ablation's own entrypoint. It **imports** the dense
``OLMo3-370M-dolma2mix.py`` script's data/optimizer/trainer wiring and training loop verbatim and
**does not modify it** -- the dense script stays the untouched baseline for the default dense run.
All Mamba-3 specifics live here:

  1. **Model.** Builds ``mamba3_olmo3_370M`` (the 1:3 attention/Mamba-3 hybrid) itself and swaps it
     into the dense config, so the dense script never needs to know about Mamba factories. Default
     **rotation block size b=3** -- the NC^1 (non-solvable ``A_5 subset SO(3)``) arm, "my version".
     Pass ``--rotation-block-size 2`` for the TC^0 baseline; ``--d-state`` overrides the SSM state
     size (default 192, which admits b in {2,3,4}).
  2. **Larger per-rank microbatch** (32 sequences vs the dense script's default). A 370M model is
     tiny for a B200 and the Mamba mixer is latency/launch-bound, so a bigger microbatch buys
     arithmetic intensity. It only changes gradient-accumulation granularity, not the global batch,
     so it is resume-safe. Raise further with ``--rank-microbatch-size`` if memory allows.
  3. **fp8 training** on the FLOP-heavy GEMMs only. MXFP8 by default (B200-native). The
     recurrence-parameterising projections (``in_B``/``in_C``/``dt_proj``/``lam_proj``/
     ``theta_proj`` -- see :attr:`Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS`) are kept in high
     precision, so decay, the trapezoidal blend, and the NC^1 rotation never see fp8 rounding.
     ``--fp8 off`` trains in pure bf16; ``--fp8 rowwise`` uses scaled-mm fp8 (any fp8-capable GPU,
     e.g. for local validation) instead of MXFP8.
  4. **Activation checkpointing** (opt-in, ``--activation-checkpointing``). Recomputes each block's
     SwiGLU MLP in the backward pass instead of storing it, trading roughly one extra MLP forward
     for a large drop in block-activation memory (this is what lets the rank microbatch grow past
     ~24 seqs). It targets ``blocks.*.feed_forward`` only, so the Mamba-3 mixer -- and thus the fast
     official kernel -- is never wrapped. It does **not** shrink the LM-head logits, which dominate
     memory at large microbatch; pair it with fused cross-entropy for that.
  5. **Fused cross-entropy** (opt-in, ``--fused-ce``). Switches the LM head to Liger-Kernel's fused
     linear cross-entropy, which computes the loss in tiles over the vocab instead of materializing
     the ``[tokens x vocab]`` logits and their gradient -- the memory term activation checkpointing
     cannot touch, and the one that binds at a 64-seq microbatch. Requires the ``liger-kernel``
     package (``RuntimeError`` at step 0 without it), and the model then returns no logits.

fp8 only swaps ``nn.Linear`` layers; it never touches the Mamba-3 SSD Triton kernel, so it is
orthogonal to the kernel/AC choice. Activation checkpointing has two flavours here:
``--activation-checkpointing`` wraps only ``blocks.*.feed_forward`` (the SwiGLU MLP), which never
touches the mixer and so is safe with the fast official kernel; whole-block AC (the
``MAMBA3_ACTIVATION_CHECKPOINTING`` escape hatch / ``full`` mode) wraps the mixer too and is
incompatible with the official kernel's ``autograd.Function``, so it must use the chunked path. The
b>=3 rotation is served by the sequence-length-adaptive prefix-scan in ``mamba3_ssd_fast`` (the
dispatch selects it automatically for the fast path).

Learning rate: the dense builder sets the ladder LR from the model's parameter count. Because the
two ablation arms have slightly different parameter counts (b=3 carries a wider angle projection
than b=2), this gives them slightly different LRs. That is preserved here for parity with the
original recipe -- pass an explicit ``--lr`` to both arms if you want LR held fixed across the
comparison. See the runbook's "decision divergence" note.

Launch with torchrun, e.g. on a single B200:

    torchrun --standalone --nproc-per-node=1 src/scripts/train/OLMo3/OLMo3-370M-mamba3.py my-run \\
        --rotation-block-size 3 --save-folder=s3://<bucket>/<run> --work-dir=/mnt/nvme/olmo-work

Validate the config on CPU first (needs a local --data-config to stay offline):

    python src/scripts/train/OLMo3/OLMo3-370M-mamba3.py my-run --dry-run --data-config <local.yaml>
"""

import importlib.util
import os
import sys
from pathlib import Path

import rich

from olmo_core.data import TokenizerConfig
from olmo_core.float8 import AOFloat8LinearRecipe, AOMXLinearConfig, Float8Config
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.mamba3 import (
    DEFAULT_D_STATE,
    Mamba3Config,
    admissible_block_sizes,
    mamba3_modules_to_ignore_for_fp8,
)
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.train.callbacks import ProfilerCallback
from olmo_core.train.train_module import TransformerActivationCheckpointingConfig

# The dense script is the single source of the config builder, training loop, argparse, and
# distributed setup. Its filename is not a valid module name (hyphens), so load it by path. Loaded
# under a non-``__main__`` name, its ``if __name__ == "__main__"`` guard does not fire.
_DOLMA2_PATH = Path(__file__).with_name("OLMo3-370M-dolma2mix.py")
_spec = importlib.util.spec_from_file_location("olmo3_370m_dolma2mix", _DOLMA2_PATH)
assert _spec is not None and _spec.loader is not None
dolma2 = importlib.util.module_from_spec(_spec)
# Register in sys.modules BEFORE exec so the dense module's dataclasses (ExperimentConfig et al.)
# resolve their own module by name -- dataclass/typing and config serialization look it up there.
# Without this, build_config raises `ModuleNotFoundError: No module named 'olmo3_370m_dolma2mix'`.
sys.modules[_spec.name] = dolma2
_spec.loader.exec_module(dolma2)

# The Mamba-3 sentinel lives in the sibling smoketests/ dir (not a package): add it to sys.path so
# both the Phase 5 smoke and the Phase 7 real runs (which share this script) can attach the callback.
sys.path.insert(0, str(_DOLMA2_PATH.parent.parent / "smoketests"))
from mamba3_sentinel import Mamba3SentinelCallback  # noqa: E402  # isort: skip

log = dolma2.log

# --- Mamba-3 arm defaults (override any of them from the CLI) --------------------------------
FP8_RECIPES = ("mxfp8", "rowwise", "tensorwise", "off")
DEFAULT_FP8_RECIPE = "mxfp8"  # B200-native; use "rowwise" for local (non-SM100) validation
DEFAULT_MODEL_FACTORY = "mamba3_olmo3_370M"
# NC^1 arm by default; pass --rotation-block-size 2 for the TC^0 baseline. MAMBA3_ROTATION_BLOCK_SIZE
# still works as a fallback default, matching the smoke tests.
DEFAULT_ROTATION_BLOCK_SIZE = int(os.environ.get("MAMBA3_ROTATION_BLOCK_SIZE", 3))
# 32 sequences/rank: divides the 192-seq global batch (6 accum steps) and raises arithmetic intensity
# above the dense default. This default does NOT fit on its own -- at 32 seqs the LM head's
# [tokens x vocab] logits gradient OOMs a B200 (183 GB) on the default loss path (observed). The real
# runs pass --activation-checkpointing --fused-ce with a 64-seq microbatch instead; without those two,
# drop to 24 or 16. Memory is constant per step at fixed seq length, so an OOM always hits at step 0.
DEFAULT_RANK_MICROBATCH_SIZE = 32 * dolma2.DEFAULT_SEQUENCE_LENGTH


def make_float8_config(recipe: str, modules_to_ignore):
    """
    Build a :class:`~olmo_core.float8.Float8Config` for one of :data:`FP8_RECIPES`.

    :param recipe: One of :data:`FP8_RECIPES`. ``"off"`` returns ``None``.
    :param modules_to_ignore: Fully-qualified module names to keep in high precision. The model's
        ``apply_fp8`` also always adds the LM head, and embeddings are never linear, so this is just
        the Mamba-3 SSM projections.

    :returns: A ``Float8Config``, or ``None`` for ``"off"``.
    """
    if recipe == "off":
        return None
    ignore = list(modules_to_ignore)
    if recipe == "mxfp8":
        # MXFP8 with cuBLAS rceil scaling -- the standard, most accurate fp8 recipe on Blackwell
        # (32-element block scales, so the effective dynamic range is far finer than tensorwise).
        return Float8Config(ao_mx=AOMXLinearConfig.mxfp8_cublas_rceil(), modules_to_ignore=ignore)
    if recipe == "rowwise":
        return Float8Config(ao_recipe=AOFloat8LinearRecipe.rowwise, modules_to_ignore=ignore)
    if recipe == "tensorwise":
        return Float8Config(ao_recipe=AOFloat8LinearRecipe.tensorwise, modules_to_ignore=ignore)
    raise ValueError(f"unknown fp8 recipe {recipe!r}; expected one of {FP8_RECIPES}")


def mamba3_fp8_config_for_model(model_config, recipe: str):
    """
    Derive the fp8 config for a specific model, protecting exactly its SSM projections.

    The ignore list is read off a ``meta`` build of ``model_config`` (which allocates nothing), so
    the fully-qualified names are always correct for this depth and block pattern -- the equality
    ``apply_float8_linear`` demands between requested and matched ignore names.

    :param model_config: The (hybrid) model config that will be trained.
    :param recipe: One of :data:`FP8_RECIPES`.

    :returns: A ``Float8Config``, or ``None`` for ``"off"``.
    """
    if recipe == "off":
        return None
    meta_model = model_config.build(init_device="meta")
    modules_to_ignore = sorted(mamba3_modules_to_ignore_for_fp8(meta_model))
    return make_float8_config(recipe, modules_to_ignore)


def _rotation_block_size_of(model_config: Mamba3Config) -> int:
    """Read the rotation block size back out of a built hybrid config (the ablation's guard)."""
    blocks = model_config.block
    assert isinstance(blocks, dict), "hybrid Mamba-3 configs use named blocks"
    return blocks["mamba3"].sequence_mixer.rotation_block_size


def build_mamba_model_config(opts):
    """
    Build the Mamba-3 hybrid model config from this script's own options.

    This is the whole reason the dense script does not need to know about Mamba: the factory is
    resolved on :class:`Mamba3Config` here, validated, and the built config is swapped into the
    dense experiment config by :func:`build_config`.

    :returns: ``(model_config, d_state)`` where ``d_state`` is the effective SSM state size.
    """
    factory_name = opts.mamba_factory
    if not factory_name.startswith("mamba3"):
        raise SystemExit(
            f"--model-factory for this script must be a Mamba-3 preset (name starts with 'mamba3'), "
            f"got {factory_name!r}. Use OLMo3-370M-dolma2mix.py for dense factories."
        )
    try:
        factory = getattr(Mamba3Config, factory_name)
    except AttributeError:
        raise SystemExit(f"Unknown Mamba-3 factory: {factory_name!r}")

    d_state = opts.d_state if opts.d_state is not None else DEFAULT_D_STATE
    admissible = admissible_block_sizes(d_state)
    if opts.rotation_block_size not in admissible:
        raise SystemExit(
            f"d_state ({d_state}) cannot express rotation_block_size ({opts.rotation_block_size}); "
            f"admissible sizes are {admissible}"
        )

    kwargs = {"rotation_block_size": opts.rotation_block_size}
    if opts.d_state is not None:
        kwargs["d_state"] = opts.d_state
    if opts.attn_backend is not None:
        kwargs["attn_backend"] = AttentionBackendName(opts.attn_backend)

    vocab_size = TokenizerConfig.dolma2().padded_vocab_size()
    model_config = factory(vocab_size=vocab_size, **kwargs)

    # Guard the ablation's core failure mode: if the block size never reaches the mixer, the NC^1
    # arm silently trains as the TC^0 baseline and both runs are the same model.
    built = _rotation_block_size_of(model_config)
    if built != opts.rotation_block_size:
        raise RuntimeError(
            f"rotation_block_size did not reach the Mamba-3 blocks: asked for "
            f"{opts.rotation_block_size}, built {built}"
        )
    return model_config, d_state


def build_config(opts, overrides):
    """
    Build the dense script's config, then swap in the Mamba-3 model and fp8 recipe.

    The dense builder owns the dataset, data loader, optimizer, scheduler, and trainer -- none of
    which are Mamba-specific -- and returns an ``ExperimentConfig`` whose ``model`` is the dense
    ``olmo3_370M``. We replace that model with the Mamba-3 hybrid (``Mamba3Config`` subclasses
    ``TransformerConfig``, so the field accepts it) and re-derive the ladder LR from the Mamba
    model's parameter count, matching what the dense builder does for its own model.
    """
    config = dolma2.build_config(opts, overrides)

    model_config, d_state = build_mamba_model_config(opts)
    config.model = model_config

    # The dense builder set optim.lr from the dense model's param count. Re-derive it from the
    # Mamba model so the recipe matches this architecture (only when the user did not pin --lr).
    if opts.lr is None:
        config.train_module.optim.lr = dolma2.ladder_lr(model_config.num_non_embedding_params)

    log.info(
        "Mamba-3 arm: rotation_block_size=%d (%s), d_state=%d, lr=%.3e",
        opts.rotation_block_size,
        "TC^0 baseline" if opts.rotation_block_size == 2 else "NC^1 (non-solvable)",
        d_state,
        config.train_module.optim.lr,
    )

    recipe = getattr(opts, "fp8", DEFAULT_FP8_RECIPE)
    config.train_module.float8_config = mamba3_fp8_config_for_model(config.model, recipe)
    if config.train_module.float8_config is not None:
        log.info(
            "fp8 %s enabled; keeping %d SSM projection(s) in high precision",
            recipe,
            len(config.train_module.float8_config.modules_to_ignore or []),
        )

    # Activation checkpointing (opt-in via --activation-checkpointing). Recomputes the SwiGLU MLP in
    # the backward pass instead of storing its 4*d_model-wide activations -- the single largest
    # per-block activation term -- which is what lets the rank microbatch grow past ~24 seqs. We use
    # selected_modules on `blocks.*.feed_forward` ONLY: whole-block AC (full/selected_blocks) would
    # wrap the Mamba-3 mixer, whose official-kernel autograd.Function is incompatible with
    # non-reentrant checkpoint_wrapper and would silently force the slow chunked path. feed_forward is
    # a distinct submodule from the mixer (which lives under block.attention), so the fast official
    # kernel is preserved. NOTE: AC does NOT touch the LM-head logits, which dominate memory at large
    # microbatch -- if the LM head OOMs at 64 seqs, that needs fused cross-entropy, not AC.
    if getattr(opts, "activation_checkpointing", False):
        config.train_module.ac_config = TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.selected_modules,
            modules=["blocks.*.feed_forward"],
        )
        log.info(
            "activation checkpointing ON: selected_modules=['blocks.*.feed_forward'] "
            "(fast Mamba-3 kernel preserved; MLP recomputed in backward)"
        )

    # Fused cross-entropy (opt-in via --fused-ce). This is the memory term AC above cannot reach: the
    # default LM head materializes the full [tokens x vocab] logits AND their gradient, which at a
    # 64-seq microbatch is the binding constraint. The fused_linear path (Liger-Kernel) computes the
    # loss in tiles over the vocab and never materializes logits at all. It requires the
    # `liger-kernel` package -- missing, the first step dies with RuntimeError -- and the LM head then
    # returns logits=None, so nothing downstream may ask for logits back.
    if getattr(opts, "fused_ce", False):
        config.model.lm_head.loss_implementation = LMLossImplementation.fused_linear
        log.info("fused cross-entropy ON: LM head returns no logits (requires liger-kernel)")

    # Silent-failure guard: at pre_train it re-checks rotation_block_size on the built model (a backstop
    # behind the post-build guard), then watches grad-norm / skip-rate / plateau / decay-horizon each
    # step and writes heartbeat.json + alerts.jsonl to the local work dir. cancel_on_alert stops the run
    # gracefully (checkpoint saved) on a critical alert; it never touches the machine.
    config.trainer = config.trainer.with_callback(
        "mamba3_sentinel",
        Mamba3SentinelCallback(
            run_dir=opts.work_dir,
            expected_rotation_block_size=opts.rotation_block_size,
            sequence_length=config.train_module.max_sequence_length,
            cancel_on_alert=True,
        ),
    )

    # --profile: torch.profiler over steps 7-9 (wait=1, warmup=5, active=3), which then logs the top-32
    # ops by CUDA time with source lines and writes a chrome trace to <work-dir>/profiler. Worth having
    # because the mixer straddles the compile boundary -- the rotation preprocessing compiles, the
    # mamba-ssm Triton kernel is held out eager -- so attributing step time by wall clock is guesswork.
    if getattr(opts, "profile", False):
        config.trainer = config.trainer.with_callback("profiler", ProfilerCallback())

    return config


def _pop_opt(argv, name, default):
    """Pull ``--name VALUE`` / ``--name=VALUE`` out of ``argv`` before the dense parser sees it."""
    value = default
    rest = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == name:
            if i + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[i + 1]
            i += 2
            continue
        if arg.startswith(name + "="):
            value = arg.split("=", 1)[1]
            i += 1
            continue
        rest.append(arg)
        i += 1
    return value, rest


def _has_flag(argv, name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in argv)


def _pop_flag(argv, name: str):
    """Pull a valueless boolean ``--name`` flag out of ``argv`` before the dense parser sees it."""
    present = False
    rest = []
    for arg in argv:
        if arg == name:
            present = True
            continue
        rest.append(arg)
    return present, rest


def parse_args(argv=None):
    """
    Parse args by delegating to the dense script, with the Mamba-3 options handled here.

    ``--fp8``, ``--model-factory``, ``--rotation-block-size`` and ``--d-state`` are consumed here
    and kept away from the dense parser (which, being the pristine dense script, does not know the
    Mamba factories or the block-size flag). ``--rank-microbatch-size`` is injected as a default but
    left for the dense parser to validate. Everything else is the dense script's argparse.

    :returns: ``(opts, overrides)`` with ``opts.fp8`` / ``opts.model_factory`` /
        ``opts.rotation_block_size`` / ``opts.d_state`` set.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    # These flags are stripped before the dense parser runs, so they would not appear in its -h
    # output; surface them here alongside the inherited help.
    if any(arg in ("-h", "--help") for arg in argv):
        print(
            f"[OLMo3-370M-mamba3] Mamba-3 options (in addition to the shared dense options below):\n"
            f"  --model-factory NAME       Mamba-3 preset (default: {DEFAULT_MODEL_FACTORY})\n"
            f"  --rotation-block-size B    2 = TC^0 baseline, 3 = NC^1 arm "
            f"(default: {DEFAULT_ROTATION_BLOCK_SIZE})\n"
            f"  --d-state N                SSM state size (default: {DEFAULT_D_STATE}; admits b in "
            f"{admissible_block_sizes(DEFAULT_D_STATE)})\n"
            f"  --fp8 {{{'|'.join(FP8_RECIPES)}}}  fp8 recipe (default: {DEFAULT_FP8_RECIPE}); SSM "
            f"projections stay high-precision\n"
            f"  --activation-checkpointing  recompute MLPs in backward (blocks.*.feed_forward); "
            f"keeps the fast kernel. Off by default.\n"
            f"  --fused-ce                 fused linear cross-entropy: drops the [tokens x vocab] "
            f"logits. Needs liger-kernel; no logits returned. Off by default.\n"
            f"  --profile                  torch.profiler over steps 7-9; logs the top-32 CUDA ops "
            f"and writes a trace to <work-dir>/profiler. Off by default.\n"
            f"  --rank-microbatch-size defaults to {DEFAULT_RANK_MICROBATCH_SIZE} "
            f"(32 seqs) for this script.\n"
        )

    recipe, argv = _pop_opt(argv, "--fp8", DEFAULT_FP8_RECIPE)
    factory, argv = _pop_opt(argv, "--model-factory", DEFAULT_MODEL_FACTORY)
    block_size_str, argv = _pop_opt(argv, "--rotation-block-size", None)
    d_state_str, argv = _pop_opt(argv, "--d-state", None)
    ac_enabled, argv = _pop_flag(argv, "--activation-checkpointing")
    fused_ce, argv = _pop_flag(argv, "--fused-ce")
    profile, argv = _pop_flag(argv, "--profile")

    if recipe not in FP8_RECIPES:
        raise SystemExit(f"--fp8 must be one of {FP8_RECIPES}, got {recipe!r}")

    # Precedence for the block size: explicit flag > MAMBA3_ROTATION_BLOCK_SIZE env > default (3).
    if block_size_str is None:
        rotation_block_size = DEFAULT_ROTATION_BLOCK_SIZE
    else:
        try:
            rotation_block_size = int(block_size_str)
        except ValueError:
            raise SystemExit(f"--rotation-block-size must be an integer, got {block_size_str!r}")

    if d_state_str is None:
        d_state = None
    else:
        try:
            d_state = int(d_state_str)
        except ValueError:
            raise SystemExit(f"--d-state must be an integer, got {d_state_str!r}")

    if not _has_flag(argv, "--rank-microbatch-size"):
        argv += ["--rank-microbatch-size", str(DEFAULT_RANK_MICROBATCH_SIZE)]

    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]] + argv
        opts, overrides = dolma2.parse_args()
    finally:
        sys.argv = saved_argv

    # Leave opts.model_factory as the dense default (olmo3_370M) so the pristine dense builder still
    # produces a valid config; the Mamba factory is carried separately and swapped in by
    # build_config -> build_mamba_model_config. This is what keeps the dense script untouched.
    opts.fp8 = recipe
    opts.mamba_factory = factory
    opts.rotation_block_size = rotation_block_size
    opts.d_state = d_state
    opts.activation_checkpointing = ac_enabled
    opts.fused_ce = fused_ce
    opts.profile = profile
    return opts, overrides


def main():
    opts, overrides = parse_args()

    if opts.dry_run:
        config = build_config(opts, overrides)
        rich.print(config)
        print("\nDry run OK -- config built. Remove --dry-run and launch under torchrun to train.")
        return

    dolma2.prepare_training_environment()
    try:
        config = build_config(opts, overrides)
        dolma2.train(config)
    finally:
        dolma2.teardown_training_environment()


if __name__ == "__main__":
    main()
