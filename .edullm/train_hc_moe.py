"""Train one arm of the hyper-connected MoE stream-balancing tranche.

    python .edullm/train_hc_moe.py "$EDULLM_RUN_ID" --cell 7 --seeds-per-arm 5 [OVERRIDES...]

WHAT THIS IS AND WHY IT IS NOT ANOTHER COPY OF ``train_on_corpus.py``. Everything about
opening a sealed corpus safely -- the dtype, the byte order, the header bytes, the tokenizer
lookup, the staged refusals with their exit codes, the W&B write that makes a refusal visible
from outside a dead container, the torn-checkpoint repair a retry needs -- is hard-won and
lives in that file. This imports it by path and replaces exactly two things: which model config
is built, and which callbacks are added. Copying the file would have produced a second place
for every one of those to drift.

THE CELL INDEX IS MAPPED HERE AND NOT IN THE SHELL, WHICH IS THE ONE DESIGN DECISION WORTH
STATING. A 2x2 at five seeds is twenty cells and the platform hands a container one integer,
so something has to turn 0..19 into an arm and a replicate. Doing it in the submitted command
means shell arithmetic inside a YAML folded scalar inside two levels of quoting, which is
exactly the class of thing that fails silently and gives every cell the same arm. Doing it here
means it is Python, it refuses an index it cannot map, and
``src/scripts/ablations/hc_launch_check.py`` checks the mapping by running the real command.

WHAT THE ARMS ARE. A 2x2 of {learned Sinkhorn mixer, H_res pinned to the identity} x {stream
balancing off, on}, all at n=4 on the `smallmoe` shape. The primary contrast is balancing on
against off at a learned mixer; the identity arms are what turn "balancing helped" into the
interaction the hypothesis is actually about. See ``docs/hc-ablation/EXPERIMENT-DESIGN.md``.
"""

import argparse
import dataclasses
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent


def _load_train_on_corpus():
    """Import the sibling entrypoint by path and register it, so its dataclasses resolve."""
    name = "_edullm_train_on_corpus"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / "train_on_corpus.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOC = _load_train_on_corpus()

from olmo_core.nn.hyper_connections import (  # noqa: E402
    HyperConnectionConfig,
    ResidualMixerType,
    StreamBalanceLossType,
    StreamCollapseConfig,
    StreamCollapseType,
    StreamUtilisationType,
)
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.train.callbacks import HyperConnectionMonitorCallback  # noqa: E402

#: The number of residual streams. Four is convention rather than a measured optimum -- the only
#: real sweep has n=8 beating n=4 on three of four LM metrics -- and it is held fixed here
#: because moving it would confound the treatment with the stream count. It is a different
#: experiment and a good one.
N_STREAMS = 4

#: The symmetry-breaking noise on the gating logits. Without it the ``n`` streams stay exact
#: copies of each other for the whole run and ``n > 1`` buys nothing but memory.
INIT_NOISE_STD = 1e-2

#: Bernoulli dropout on the residual-mixer logits during training.
RESIDUAL_DROPOUT_P = 0.1

#: The weight on the stream-balancing loss in the treated arms. A guess, matched to
#: ``MoEConfig.lb_loss_weight``; the two losses are on different scales and nothing has tuned
#: this. It is identical in both treated arms, which is the requirement.
BALANCE_LOSS_WEIGHT = 0.01

#: arm name -> (residual mixer, stream-balancing weight). The 2x2.
ARMS: Dict[str, tuple] = {
    "mhc_moe": (ResidualMixerType.sinkhorn, 0.0),
    "mhc_moe_balanced": (ResidualMixerType.sinkhorn, BALANCE_LOSS_WEIGHT),
    "mhc_moe_identity": (ResidualMixerType.identity, 0.0),
    "mhc_moe_identity_balanced": (ResidualMixerType.identity, BALANCE_LOSS_WEIGHT),
}

#: The order cells are laid out in: cell = ARM_ORDER.index(arm) * seeds_per_arm + seed.
ARM_ORDER: List[str] = list(ARMS)


def resolve_cell(cell: Optional[int], *, seeds_per_arm: int, arm: Optional[str]) -> tuple:
    """
    Turn one fan-out index into an arm and a replicate.

    :param cell: The fan-out index, or ``None`` when ``arm`` names the arm directly.
    :param seeds_per_arm: How many replicates each arm gets.
    :param arm: An explicit arm name, for a single run outside a fan-out.

    :returns: ``(arm_name, seed)``.

    :raises Refusal: If neither or both are given, or the index maps to no arm.
    """
    if (cell is None) == (arm is None):
        raise TOC.Refusal(
            TOC.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "pass exactly one of --cell and --arm. --cell is what a fan-out supplies and is "
            "how the tranche runs; --arm is for a single run by hand. Passing both would let "
            "the index and the name disagree, and passing neither leaves the arm undecided.",
        )
    if seeds_per_arm < 1:
        raise TOC.Refusal(
            TOC.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"--seeds-per-arm must be >= 1, got {seeds_per_arm}",
        )
    if arm is not None:
        if arm not in ARMS:
            raise TOC.Refusal(
                TOC.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"unknown arm {arm!r}; this tranche is {', '.join(ARM_ORDER)}",
            )
        return arm, 0
    assert cell is not None
    total = len(ARM_ORDER) * seeds_per_arm
    if not 0 <= cell < total:
        # REFUSED RATHER THAN WRAPPED. A modulo here would map cell 20 of a mis-sized fan-out
        # onto arm 0 seed 0, which is a duplicate replicate reported as a fresh one -- the
        # failure the whole design is built to avoid, arriving through the fan-out size.
        raise TOC.Refusal(
            TOC.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"cell {cell} is outside 0..{total - 1}. This tranche is {len(ARM_ORDER)} arms x "
            f"{seeds_per_arm} seeds, so the fan-out size must be exactly {total}.",
        )
    return ARM_ORDER[cell // seeds_per_arm], cell % seeds_per_arm


def build_model_config(arm: str, *, vocab_size: int, seed: int) -> TransformerConfig:
    """
    The `smallmoe` shape with every block hyper-connected, for one arm.

    :param arm: The arm name.
    :param vocab_size: The padded vocabulary size.
    :param seed: The replicate seed, which reaches the parameter initialisation.

    :returns: The model config.
    """
    mixer, balance_weight = ARMS[arm]
    # `init_seed` is set by `dataclasses.replace` below and NOT passed to the factory.
    # `TransformerConfig.smallmoe` pops the kwargs it knows and forwards nothing else, so
    # `smallmoe(vocab_size=v, init_seed=3)` drops the seed on the floor and every replicate
    # gets the default 0 — silently, with the two other seeds still varying, which is the
    # partial-replicate failure the baseline spec's header spends a paragraph on. Caught by
    # `src/scripts/ablations/hc_launch_check.py`, which builds the real config and looks.
    base = TransformerConfig.smallmoe(vocab_size=vocab_size)
    hc = HyperConnectionConfig(
        n_streams=N_STREAMS,
        mixer=mixer,
        init_noise_std=INIT_NOISE_STD,
        residual_dropout_p=RESIDUAL_DROPOUT_P,
        collapse=StreamCollapseType.mean,
        stream_balance_loss_weight=balance_weight,
        stream_balance_statistic=StreamUtilisationType.dispersion,
        stream_balance_loss_type=StreamBalanceLossType.entropy,
    )
    block = dataclasses.replace(
        base.block,
        name=TransformerBlockType.hyper_connection_moe_reordered_norm,
        hyper_connection=hc,
    )
    return dataclasses.replace(
        base,
        block=block,
        name=TransformerType.hyper_connection_moe,
        stream_collapse=StreamCollapseConfig(n_streams=N_STREAMS, policy=StreamCollapseType.mean),
        init_seed=seed,
    )


def build_config(opts, overrides: List[str]):
    """
    ``train_on_corpus.build_config``, with the model replaced by this arm and the monitor added.

    :param opts: The parsed options.
    :param overrides: Dotted overrides, applied last exactly as the sibling entrypoint applies
        them.

    :returns: The merged experiment config.
    """
    arm, seed = resolve_cell(opts.cell, seeds_per_arm=opts.seeds_per_arm, arm=opts.arm)
    log.info("cell %s resolves to arm %r, seed %d", opts.cell, arm, seed)

    # Every seed the sibling entrypoint carries, moved together. See the header of
    # .edullm/run.hc-baseline.yaml for why moving only one of them measures the wrong variance.
    opts.data_seed = seed

    config = TOC.build_config(opts, [])
    corpus_vocab = config.model.vocab_size
    config = dataclasses.replace(
        config,
        model=build_model_config(arm, vocab_size=corpus_vocab, seed=seed),
        init_seed=seed,
    )
    config.trainer.with_callback(
        "hc_monitor",
        HyperConnectionMonitorCallback(
            interval=opts.monitor_interval, matrix_interval=opts.matrix_interval
        ),
    )
    return config.merge(overrides)


def build_parser() -> argparse.ArgumentParser:
    """
    The sibling entrypoint's parser plus the three flags this file adds.

    :returns: The parser.
    """
    parser = TOC.build_parser()
    parser.prog = "train_hc_moe"
    parser.add_argument(
        "--cell",
        type=int,
        default=None,
        help="the fan-out index, 0..(arms*seeds - 1). Supplied by the submitted command from "
        "AWS_BATCH_JOB_ARRAY_INDEX. Mutually exclusive with --arm.",
    )
    parser.add_argument("--seeds-per-arm", type=int, default=5)
    parser.add_argument(
        "--arm", default=None, choices=sorted(ARMS), help="one arm, for a run outside a fan-out"
    )
    parser.add_argument("--monitor-interval", type=int, default=50)
    parser.add_argument("--matrix-interval", type=int, default=500)
    return parser


def main() -> None:
    """
    Resolve the arm, build the config, refuse early where the sibling entrypoint refuses.
    """
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()

    missing = [
        name
        for name, value in (
            ("EDULLM_DATASET_ID", opts.dataset_id),
            ("EDULLM_DATASET_VERSION", opts.dataset_version),
            ("EDULLM_DATASET_TOKENIZER", opts.dataset_tokenizer),
            ("EDULLM_CHECKPOINT_DIR", opts.save_folder),
        )
        if not value
    ]
    if missing:
        raise TOC.Refusal(
            TOC.Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "the platform sets these and they are unset: " + ", ".join(missing),
        )

    with TOC.during(TOC.Stage.THE_CONFIG_WOULD_NOT_BUILD):
        config = build_config(opts, overrides)

    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.train.train_module import validate_precision_support

    try:
        validate_precision_support(config)
    except OLMoConfigurationError as unusable:
        if opts.dry_run:
            log.warning("%s", unusable)
        else:
            raise TOC.Refusal(
                TOC.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, str(unusable)
            ) from None

    if opts.dry_run:
        TOC.show(config)
        return

    with TOC.during(TOC.Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        TOC.prepare_training_environment()
    try:
        with TOC.during(TOC.Stage.TRAINING_ITSELF_FAILED):
            TOC.train(config, opts)
    finally:
        TOC.teardown_training_environment()


def cli() -> int:
    """
    Turn a refusal into the exit code the platform can read. See ``train_on_corpus.cli``.

    :returns: The process exit status.
    """
    import traceback

    try:
        main()
    except TOC.Refusal as refusal:
        print(refusal.explanation, file=sys.stderr)
        print(f"edullm-stage: {refusal.stage.name} exit={int(refusal.stage)}", file=sys.stderr)
        if refusal.__cause__ is not None:
            traceback.print_exception(
                type(refusal.__cause__), refusal.__cause__, refusal.__cause__.__traceback__
            )
        TOC.leave_the_reason_in_wandb(
            run_name=os.environ.get("EDULLM_RUN_ID", "local"),
            stage=refusal.stage,
            explanation=refusal.explanation,
        )
        return int(refusal.stage)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
