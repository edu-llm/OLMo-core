"""
The hyper-connections residual-mixer ablation: six arms at a matched token, data and seed
budget.

The question is narrow. Hyper-connections give a transformer sub-layer ``n`` residual streams
instead of one and mix them with an ``n x n`` matrix ``H_res``; three papers propose three
different ways to constrain that matrix to be doubly stochastic, and none of them is compared
against the others under a matched budget anywhere public. This script defines that comparison.

Run it with no arguments to print the arm table, or with ``--dry-run`` to build every arm on
CPU and check its shapes and parameter counts::

    python src/scripts/ablations/hc_ablation.py
    python src/scripts/ablations/hc_ablation.py --dry-run --model-size tiny
    python src/scripts/ablations/hc_ablation.py --show mhc_sinkhorn

Nothing here trains, launches or reaches a network. A real run of these arms needs GPUs and so
goes through the eduLLM platform's ``edullm`` CLI; see ``docs/hc-ablation/README.md``.
"""

import argparse
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Optional, cast

import torch

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.hyper_connections import (
    HyperConnectionConfig,
    ResidualMixerType,
    StreamCollapseConfig,
    StreamCollapseType,
)
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import Duration, TrainerConfig
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    DownstreamEvaluatorCallbackConfig,
    GarbageCollectorCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

log = logging.getLogger(__name__)

__all__ = [
    "CLASSIC_TASKS",
    "MATH_REASONING_TASKS",
    "Arm",
    "ARMS",
    "build_model_config",
    "build_train_module_config",
    "build_trainer_config",
]


# ---------------------------------------------------------------------------------------------
# Matched budget. Every arm shares all of it; only the residual mixer changes.
# ---------------------------------------------------------------------------------------------

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 128 * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = 8 * SEQUENCE_LENGTH
TRAIN_TOKENS = 4_000_000_000
LEARNING_RATE = 6e-4
WARMUP_STEPS = 200
INIT_SEED = 12536
DATA_SEED = 34521

N_STREAMS = 4
INIT_NOISE_STD = 1e-2
RESIDUAL_DROPOUT_P = 0.1
COLLAPSE = StreamCollapseType.mean

TOKENIZER_CONFIG = TokenizerConfig.dolma2()

#: Paths or URLs to the ``.npy`` tokenized training shards. Left empty on purpose: the arms
#: below are a specification, and pointing them at a corpus is a decision for whoever runs them.
#: Every arm reads the same list, which is what "matched data" means here.
DATA_PATHS: List[str] = []


# ---------------------------------------------------------------------------------------------
# Evaluation suites. Every name is checked against `olmo_core.eval.task_groups` by
# `hc_ablation_test`-style verification; they are OLMo-core's own task names and are run through
# `DownstreamEvaluatorCallbackConfig`, not through a second harness.
# ---------------------------------------------------------------------------------------------

#: The OLMES-style multiple-choice core. Cheap, and the suite an ablation at this scale is
#: normally read on.
CLASSIC_TASKS: List[str] = [
    "arc_easy_test_rc_5shot",
    "arc_challenge_test_rc_5shot",
    "hellaswag_rc_5shot",
    "piqa_val_rc_5shot",
    "winogrande_val_rc_5shot",
]

#: Bits-per-byte on gold reasoning traces. Included because the case for a richer residual
#: topology is a case about depth-wise composition, which is what multi-step arithmetic asks
#: for, and because BPB moves at scales where multiple-choice accuracy is still noise.
MATH_REASONING_TASKS: List[str] = [
    "gsm8k_gold_bpb_5shot",
    "minerva_math_algebra_gold_bpb_0shot",
    "minerva_math_counting_and_probability_gold_bpb_0shot",
    "minerva_math_geometry_gold_bpb_0shot",
    "minerva_math_intermediate_algebra_gold_bpb_0shot",
    "minerva_math_number_theory_gold_bpb_0shot",
    "minerva_math_prealgebra_gold_bpb_0shot",
    "minerva_math_precalculus_gold_bpb_0shot",
]


@dataclass
class Arm:
    """
    One arm of the ablation.

    :param name: The arm's name, used as the run name and as the CLI selector.
    :param mixer: The residual-mixer parameterisation, or ``None`` for the single-stream
        baseline.
    :param n_streams: The number of residual streams. Always 1 when ``mixer`` is ``None``.
    :param description: One line on what the arm is for.
    """

    name: str
    mixer: Optional[ResidualMixerType]
    n_streams: int = N_STREAMS
    description: str = ""

    @property
    def is_baseline(self) -> bool:
        """
        Whether this arm is the ordinary single-stream transformer.

        :returns: ``True`` for the baseline arm.
        """
        return self.mixer is None

    @property
    def hyper_connection(self) -> Optional[HyperConnectionConfig]:
        """
        The hyper-connection config for this arm.

        :returns: ``None`` for the baseline arm, otherwise the config both sub-layers of every
            block are wrapped with.
        """
        if self.mixer is None:
            return None
        return HyperConnectionConfig(
            n_streams=self.n_streams,
            mixer=self.mixer,
            init_noise_std=INIT_NOISE_STD,
            residual_dropout_p=RESIDUAL_DROPOUT_P,
            collapse=COLLAPSE,
        )


#: The six arms. ``baseline`` and ``mhc_identity`` are the controls: the first says what an
#: ordinary residual gets, and the second isolates how much of any gain is the extra streams and
#: the gates rather than the mixing between streams.
ARMS: List[Arm] = [
    Arm(
        name="baseline",
        mixer=None,
        n_streams=1,
        description="single-stream OLMo-2, no hyper-connections",
    ),
    Arm(
        name="hc_unconstrained",
        mixer=ResidualMixerType.unconstrained,
        description="original Hyper-Connections; raw H_res, the instability control",
    ),
    Arm(
        name="mhc_sinkhorn",
        mixer=ResidualMixerType.sinkhorn,
        description="exact mHC; H_res = Sinkhorn(logits), 20 iterations in log space",
    ),
    Arm(
        name="mhc_lite",
        mixer=ResidualMixerType.birkhoff,
        description="mHC-lite; H_res = softmax-weighted sum of all n! permutations",
    ),
    Arm(
        name="kromhc",
        mixer=ResidualMixerType.kronecker,
        description="KromHC; H_res = Kronecker product of log2(n) doubly stochastic 2x2s",
    ),
    Arm(
        name="mhc_identity",
        mixer=ResidualMixerType.identity,
        description="streams and gates but H_res = I; isolates the mixing itself",
    ),
]


def build_model_config(arm: Arm, *, model_size: str = "190M") -> TransformerConfig:
    """
    Build the model config for one arm.

    Every arm starts from the same OLMo-2 backbone and differs only in whether its blocks are
    hyper-connected and, if so, which residual mixer they use.

    :param arm: The arm.
    :param model_size: ``"190M"`` for the real ablation shape, or ``"tiny"`` for a shape small
        enough to build and run repeatedly on a laptop CPU.

    :returns: The model config.

    :raises ValueError: If ``model_size`` is not recognised.
    """
    vocab_size = TOKENIZER_CONFIG.padded_vocab_size()
    if model_size == "190M":
        base = TransformerConfig.olmo2_190M(vocab_size=vocab_size, init_seed=INIT_SEED)
    elif model_size == "tiny":
        base = TransformerConfig.llama_like(
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=vocab_size,
            block_name=TransformerBlockType.reordered_norm,
            qk_norm=True,
            layer_norm_eps=1e-6,
            hidden_size_multiplier=1.5,
            init_seed=INIT_SEED,
        )
    else:
        raise ValueError(f"unknown model size {model_size!r}, expected '190M' or 'tiny'")

    if arm.is_baseline:
        return base

    hc_config = arm.hyper_connection
    assert hc_config is not None
    block = dataclasses.replace(
        cast(TransformerBlockConfig, base.block),
        name=TransformerBlockType.hyper_connection,
        hyper_connection=hc_config,
    )
    return dataclasses.replace(
        base,
        block=block,
        name=TransformerType.hyper_connection,
        stream_collapse=StreamCollapseConfig(n_streams=arm.n_streams, policy=COLLAPSE),
    )


def build_train_module_config(arm: Arm) -> TransformerTrainModuleConfig:
    """
    Build the train-module config for one arm. Identical across arms by construction — that is
    what makes the comparison a comparison.

    :param arm: The arm. Only used to keep the signature uniform.

    :returns: The train-module config.
    """
    del arm
    return TransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LEARNING_RATE,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # The hyper-connected blocks have not been validated under `torch.compile` or FSDP
        # resharding, so both are off here rather than on and unverified. See the README.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup=WARMUP_STEPS),
    )


def build_trainer_config(arm: Arm, *, save_folder: str) -> TrainerConfig:
    """
    Build the trainer config for one arm, including both downstream evaluation suites.

    The evaluators need the ``eval`` extra (``pip install 'ai2-olmo-core[eval]'``, which pins
    ``ai2-olmo-eval==0.9.0``) and a GPU; they are configured here but nothing in this file
    builds or runs them.

    :param arm: The arm.
    :param save_folder: Where checkpoints and metrics go.

    :returns: The trainer config.
    """
    return (
        TrainerConfig(
            save_folder=save_folder,
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.tokens(TRAIN_TOKENS),
        )
        .with_callback(
            "checkpointer",
            CheckpointerCallback(save_interval=1000, ephemeral_save_interval=250),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback(
            "downstream_classic",
            DownstreamEvaluatorCallbackConfig(
                tasks=CLASSIC_TASKS,
                tokenizer=TOKENIZER_CONFIG,
                eval_interval=1000,
                eval_on_finish=True,
            ),
        )
        .with_callback(
            "downstream_math_reasoning",
            DownstreamEvaluatorCallbackConfig(
                tasks=MATH_REASONING_TASKS,
                tokenizer=TOKENIZER_CONFIG,
                eval_interval=1000,
                eval_on_finish=True,
            ),
        )
    )


@dataclass
class ArmSummary:
    """
    What the comparison table and the dry run report per arm.
    """

    arm: Arm
    total_params: int
    routing_params: int
    non_embedding_params: int
    output_shape: Optional[tuple] = None
    stream_shape: Optional[tuple] = None
    notes: List[str] = field(default_factory=list)


def summarize_arm(arm: Arm, *, model_size: str = "190M") -> ArmSummary:
    """
    Summarise an arm from its config alone, without building the model.

    :param arm: The arm.
    :param model_size: The model shape, as for :func:`build_model_config`.

    :returns: The summary.
    """
    config = build_model_config(arm, model_size=model_size)
    return ArmSummary(
        arm=arm,
        total_params=config.num_params,
        routing_params=config.num_routing_params,
        non_embedding_params=config.num_non_embedding_params,
    )


def format_table(summaries: List[ArmSummary]) -> str:
    """
    Render the arm comparison table.

    :param summaries: One summary per arm, in the order they should be printed.

    :returns: The table as a string.
    """
    header = (
        f"{'arm':<18}{'mixer':<16}{'n':>3}  {'routing params':>15}"
        f"{'per sub-layer':>15}{'total params':>15}"
    )
    lines = [header, "-" * len(header)]
    for summary in summaries:
        arm = summary.arm
        hc = arm.hyper_connection
        per_sublayer = hc.num_params() if hc is not None else 0
        lines.append(
            f"{arm.name:<18}{str(arm.mixer) if arm.mixer else '-':<16}{arm.n_streams:>3}  "
            f"{summary.routing_params:>15,d}{per_sublayer:>15,d}{summary.total_params:>15,d}"
        )
    lines.append("")
    lines.append("Descriptions:")
    for summary in summaries:
        lines.append(f"  {summary.arm.name:<18}{summary.arm.description}")
    return "\n".join(lines)


def dry_run(model_size: str = "190M", *, batch_size: int = 2, seq_len: int = 16) -> int:
    """
    Build every arm on CPU, run one forward pass, and report shapes and parameter counts.

    No optimizer, no data, no training step and no network access. Arms are built and released
    one at a time so that the peak memory is one model, not six.

    :param model_size: The model shape, as for :func:`build_model_config`.
    :param batch_size: The dry-run batch size.
    :param seq_len: The dry-run sequence length.

    :returns: A process exit code: 0 if every arm built and ran, 1 otherwise.
    """
    vocab_size = TOKENIZER_CONFIG.padded_vocab_size()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    summaries: List[ArmSummary] = []
    failures = 0

    for arm in ARMS:
        config = build_model_config(arm, model_size=model_size)
        summary = ArmSummary(
            arm=arm,
            total_params=config.num_params,
            routing_params=config.num_routing_params,
            non_embedding_params=config.num_non_embedding_params,
        )
        try:
            model = config.build()
            model.init_weights()
            model.eval()

            measured_total = sum(p.numel() for p in model.parameters())
            if measured_total != config.num_params:
                summary.notes.append(
                    f"config said {config.num_params:,d} params, the model has "
                    f"{measured_total:,d}"
                )
                failures += 1

            # Peek at the stream tensor the blocks actually see.
            with torch.no_grad():
                h = model.embeddings(input_ids)
                summary.stream_shape = tuple(model.expand_residual_streams(h).shape)
                summary.output_shape = tuple(model(input_ids).shape)

            expected_streams: tuple = (batch_size, seq_len, arm.n_streams, config.d_model)
            if arm.is_baseline:
                expected_streams = (batch_size, seq_len, config.d_model)
            if summary.stream_shape != expected_streams:
                summary.notes.append(
                    f"expected streams of shape {expected_streams}, got {summary.stream_shape}"
                )
                failures += 1
            if summary.output_shape != (batch_size, seq_len, vocab_size):
                summary.notes.append(f"unexpected logits shape {summary.output_shape}")
                failures += 1

            del model
        except Exception as exc:  # noqa: BLE001 - a dry run reports every arm, not just the first
            summary.notes.append(f"FAILED to build or run: {exc!r}")
            failures += 1

        summaries.append(summary)

    print(f"\nDry run on CPU, model size {model_size!r}, input {tuple(input_ids.shape)}\n")
    print(format_table(summaries))
    print("\nShapes and per-arm results:")
    for summary in summaries:
        status = "FAIL" if summary.notes else "ok"
        print(
            f"  {summary.arm.name:<18}{status:<6}streams={summary.stream_shape} "
            f"logits={summary.output_shape}"
        )
        for note in summary.notes:
            print(f"      {note}")

    print(
        f"\n{len(ARMS) - failures}/{len(ARMS)} arms built and ran."
        if not failures
        else f"\n{failures} problem(s) across {len(ARMS)} arms."
    )
    print(
        "\nNothing was trained and no evaluation was run. What this checks is that every arm "
        "\nbuilds, carries the parameter count its config claims, and produces the right "
        "\nshapes. Loss and downstream numbers need a GPU run."
    )
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the CLI.

    :param argv: Arguments, defaulting to ``sys.argv[1:]``.

    :returns: A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build every arm on CPU and report shapes and parameter counts",
    )
    parser.add_argument(
        "--model-size",
        default="190M",
        choices=["190M", "tiny"],
        help="the model shape to build (default: 190M)",
    )
    parser.add_argument(
        "--show",
        metavar="ARM",
        help="print one arm's full model and trainer config as YAML",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    if args.show:
        arms = {arm.name: arm for arm in ARMS}
        if args.show not in arms:
            parser.error(f"unknown arm {args.show!r}; choose from {sorted(arms)}")
        arm = arms[args.show]
        print(build_model_config(arm, model_size=args.model_size))
        print(build_train_module_config(arm))
        print(build_trainer_config(arm, save_folder=f"/tmp/hc-ablation/{arm.name}"))
        return 0

    if args.dry_run:
        return dry_run(args.model_size)

    print(f"\nHyper-connections residual-mixer ablation, model size {args.model_size!r}")
    print(
        f"Matched budget: {TRAIN_TOKENS / 1e9:.1f}B tokens, sequence length {SEQUENCE_LENGTH}, "
        f"global batch {GLOBAL_BATCH_SIZE // SEQUENCE_LENGTH} sequences,\n"
        f"lr {LEARNING_RATE}, init seed {INIT_SEED}, data seed {DATA_SEED}, "
        f"{len(DATA_PATHS)} data shards configured.\n"
    )
    print(format_table([summarize_arm(arm, model_size=args.model_size) for arm in ARMS]))
    print(f"\nCLASSIC eval suite ({len(CLASSIC_TASKS)} tasks): {', '.join(CLASSIC_TASKS)}")
    print(
        f"\nMATH_REASONING eval suite ({len(MATH_REASONING_TASKS)} tasks): "
        f"{', '.join(MATH_REASONING_TASKS)}"
    )
    print("\nRun with --dry-run to build every arm on CPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
