#!/usr/bin/env python3
"""Train one arm of the hyper-connection module on a published eduLLM corpus.

NOTHING OF ``train_on_corpus.py`` IS COPIED HERE, for the reason its sibling
``train_recurrent.py`` gives on the recurrent branch: almost all of that file is things this
run wants unchanged -- resolving a corpus from the sealed manifest, refusing with a stage
number that survives a container nobody can read the logs of, the uint32 dtype fix, the
torn-checkpoint repair a Batch retry needs, and the summary the platform parses back out of
the log stream. A forked copy would drift from all of it on the first upstream fix.

So this is a wrapper that rebinds three module globals that ``train_on_corpus.main`` resolves
at call time:

  ``build_parser``  adds ``--arm`` and ``--seed``, and re-points ``--model-factory`` at
                    olmo3_370M, so that an arm cannot silently train the platform's default
                    190M.
  ``build_config``  applies the arm to the model config, moves all three seeds together, and
                    installs the weight-decay split.
  ``train``         attaches the lane monitor, which has nowhere else to go: the trainer is
                    built inside ``train``.

The weight-decay split is worth spelling out, because it is one of the five candidate causes
rather than a detail. ByteDance: "the static component does not utilize weight decay, whereas
the dynamic component does." Every arm gets it except ``decay-everything``, which exists
precisely to find out what happens without it.

Rehearse first, on an L40S::

    bash -lc 'python .edullm/train_hyper_connections.py "$EDULLM_RUN_ID" \
        --save-folder "$EDULLM_CHECKPOINT_DIR" \
        --arm faithful --model-factory hc_rehearsal --steps 200 \
        --fail-closed-by-step 150 --param-dtype bfloat16'

Locally, with no corpus and no GPU::

    python .edullm/train_hyper_connections.py test --dry-run --arm faithful \
        --dataset-id pretrain/regmix-10b --dataset-version v1 \
        --dataset-tokenizer tokenizer/dolma2-bpe --save-folder /tmp/x
"""

import argparse
import os
import sys

# Both this file and its siblings have to be importable by a stable name: `_CLASS_` in the
# saved config records module paths, and anything reading that config back resolves them with
# `importlib.import_module`, so they have to be reachable as themselves rather than as
# `__main__`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms  # noqa: E402
import train_on_corpus  # noqa: E402
from hyper_connection_arms import ARMS  # noqa: E402

from olmo_core.train.callbacks import HyperConnectionMonitorCallback  # noqa: E402

hyper_connection_arms.install()

_build_parser = train_on_corpus.build_parser
_build_config = train_on_corpus.build_config
_train = train_on_corpus.train
_show = train_on_corpus.show

#: The 370M run. 10B dolma2 tokens at seq 4096 over a 768K-token batch.
DEFAULT_STEPS = 12_715
DEFAULT_LEARNING_RATE = 7.8e-4
DEFAULT_WEIGHT_DECAY = 0.033


class _ListArms(argparse.Action):
    """
    Prints at parse time rather than in ``build_config``, because ``main`` refuses on the
    platform's environment variables before it ever builds a config, and asking what the arms
    are should not require a corpus.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, namespace, values, option_string
        print(hyper_connection_arms.describe())
        raise SystemExit(0)


def build_parser():
    parser = _build_parser()
    parser.prog = "train_hyper_connections"
    parser.description = "Train one arm of the hyper-connection module."

    # Re-point rather than add: the platform's default is olmo2_190M, and an arm that quietly
    # trained a 190M would be hard to notice from the loss curve and impossible to compare.
    # hc_370M rather than olmo3_370M because the latter asks for a flash-attn backend this
    # image does not carry, which is what killed the first rehearsal.
    parser.set_defaults(
        model_factory="hc_370M",
        sequence_length=4096,
        global_batch_size=768 * 1024,
        rank_microbatch_size=8 * 1024,
        steps=DEFAULT_STEPS,
        warmup_steps=round(DEFAULT_STEPS * 0.02),
        save_interval=2000,
        learning_rate=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--arm",
        choices=sorted(ARMS),
        required=True,
        help="Which arm to run. See --list-arms.",
    )
    parser.add_argument(
        "--list-arms",
        nargs=0,
        action=_ListArms,
        help="Print the arm table and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Replicate index. Moves parameter initialization, the shuffle and the global RNG "
        "together, because a 'seed' that only reshuffles the data measures less variance than "
        "the run actually has and would understate the noise floor everything else is "
        "compared against.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay for everything that takes it. The static hyper-connection "
        "component is excluded from it on every arm but decay-everything.",
    )
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=50,
        help="How often, in steps, to log lane norms, spectral radii and the composite "
        "condition number.",
    )
    parser.add_argument(
        "--held-out-shards",
        type=int,
        default=hyper_connection_arms.HELD_OUT_SHARDS,
        help="Shards reserved from the corpus for evaluation. regmix-10b declares no "
        "validation split, and without one the only loss in the run is training loss -- whose "
        "variance across seeds is partly just a different sample of the corpus, since --seed "
        "moves the shuffle. Set to 0 to train on everything and measure nothing held out.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="How often, in steps, to run the held-out evaluation.",
    )
    parser.add_argument(
        "--bytes-per-token",
        type=float,
        default=hyper_connection_arms.DOLMA2_BYTES_PER_TOKEN,
        help="Used to report bits-per-byte beside every CE loss. Sets the absolute level "
        "only: it is the same for every arm, so a BPB difference between arms is a CE "
        "difference times a fixed factor whatever this is.",
    )
    parser.add_argument(
        "--fail-closed-by-step",
        type=int,
        default=None,
        help="Stop the run with an error if the lanes have not differentiated by this step. "
        "SET THIS ON THE REHEARSAL. Identical lane norms mean the mechanism is inert, and a "
        "downstream number from an inert mechanism is not interpretable either way -- better "
        "to find that out for a few dollars than for a few hundred.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Build the config AND the held-out dataset, say what they came out as, and exit "
        "without training. Needs corpus credentials but no GPU, so it runs in seconds on a "
        "laptop. THREE SUBMISSIONS DIED ON THINGS THIS CATCHES: a backend the image lacks, a "
        "dataset class the evaluator refuses, and a missing metadata label. Each cost a queue "
        "wait and a container to discover.",
    )
    return parser


def preflight(config) -> None:
    """
    Build everything a run builds before its first step, and print what came out.

    The evaluator validates its dataset by building it and then checking the result, so its
    refusals arrive inside a running container rather than at submission. This does the same
    work on a laptop.

    :raises Exception: Whatever the real run would have raised, unchanged.
    """
    print(f"corpus       {config.dataset_id}/{config.dataset_version}")
    print(f"train shards {len(config.dataset.paths)}")

    mixer = config.model.block.sequence_mixer
    print(
        f"model        {config.model.d_model}d x {config.model.n_layers}L, "
        f"{config.model.num_params:,} params"
    )
    print(f"attention    backend={mixer.backend}, sliding_window={mixer.sliding_window}")
    print(f"callbacks    {', '.join(sorted(config.trainer.callbacks))}")

    held_out = config.trainer.callbacks.get("held_out")
    if held_out is None:
        print("held out     none")
    else:
        dataset = held_out.eval_dataset.build()
        print(f"held out     {type(dataset).__name__}, {len(dataset.paths)} shard(s)")
        for path, meta in zip(dataset.paths, dataset.metadata):
            if "label" not in meta:
                raise RuntimeError(f"shard has no 'label' metadata, evaluator will refuse: {path}")
            print(f"             {meta['label']:16s} {str(path).rsplit('/', 1)[-1]}")

    print("preflight OK")


def build_config(opts, overrides):
    arm = ARMS[opts.arm]

    # Before the delegate, because it builds the data loader with this.
    opts.data_seed = opts.data_seed + opts.seed

    config = _build_config(opts, overrides)

    arm.apply(config.model)

    # All three seeds move with --seed. `init_seed` on the model is the generator that draws
    # the parameters; `init_seed` on the experiment is what seed_all uses; `data_seed` is the
    # shuffle, set above.
    config.init_seed = config.init_seed + opts.seed
    config.model.init_seed = config.model.init_seed + opts.seed

    config.train_module.optim.weight_decay = opts.weight_decay
    if opts.arm != "decay-everything":
        overrides_for_hc = arm.optim_group_overrides(weight_decay=opts.weight_decay)
        if overrides_for_hc:
            config.train_module.optim.group_overrides = (
                list(config.train_module.optim.group_overrides or []) + overrides_for_hc
            )

    _add_held_out_evaluation(config, opts)

    # Ride the dry-run path rather than exiting here. main() initializes the distributed
    # environment between build_config and train, so preflight has to stop before that, and
    # `during()` turns a bare SystemExit into a refusal with a non-zero status -- a preflight
    # that passed would report as a failure to anything scripting it.
    if getattr(opts, "preflight", False):
        _PREFLIGHT.append(True)
        opts.dry_run = True

    return config


#: Set by `build_config` so the rebound `show` below knows which summary to print. A list
#: because `main` resolves module globals at call time and rebinding a bool would not be seen.
_PREFLIGHT: list = []


def show(config):
    """
    Replace the dry run's full config dump with the preflight summary when asked for one.
    """
    if _PREFLIGHT:
        preflight(config)
        return
    return _show(config)


def declared_validation_paths(config) -> list:
    """
    The validation shards the corpus publishes, or an empty list if it publishes none.

    ``train_on_corpus.resolve_corpus`` deliberately asks the reader for trainable shards only,
    and its comment says held-out shards need "a corpus that declares one, which regmix-10b
    does not". That is not true of regmix-10b: it declares seven val shards, one per source,
    totalling about 15M tokens. The reader resolves every split whatever you ask it for, so
    this reads ``.val`` off a second, cheap resolve rather than changing the first one.

    :returns: Validation shard URIs, sorted.
    """
    from edullm_data.read import dataset_paths
    from edullm_data.s3 import Boto3S3

    read = dataset_paths(config.dataset_id, config.dataset_version, s3=Boto3S3.default())
    return sorted(read.val or [])


def _add_held_out_evaluation(config, opts):
    """
    Evaluate on held-out shards from the same sealed corpus, and report bits-per-byte.

    Local shards rather than the stock LM evaluator's default, which reads a C4 shard from
    olmo-data.org: a public-internet fetch in the middle of a run whose whole claim is that it
    read a sealed corpus, and one whose failure would look like a training failure.

    Preferring the corpus's own declared validation split over carving one out of training is
    better on three counts at once. No training tokens are lost, so the budget stays at the
    full 10B. The split is the publisher's rather than an arbitrary slice of ours. And it comes
    stratified by source, which turns one pooled bits-per-byte into seven -- an average over
    arxiv, code, web text and Wikipedia together is exactly the kind that hides the effect it
    is meant to measure.
    """
    from olmo_core.data import NumpyPaddedFSLDatasetConfig
    from olmo_core.train.callbacks import LMEvaluatorCallbackConfig

    config.trainer = config.trainer.with_callback(
        "bits_per_byte",
        hyper_connection_arms.BitsPerByteCallback(bytes_per_token=opts.bytes_per_token),
    )

    if opts.eval_interval < 1:
        return

    eval_paths = declared_validation_paths(config)
    if not eval_paths:
        # Nothing declared, so carve. Worse on every count above, and only correct at all
        # because every arm and seed carves identically.
        if opts.held_out_shards < 1:
            return
        train_paths, eval_paths = hyper_connection_arms.split_held_out(
            config.dataset.paths, opts.held_out_shards
        )
        config.dataset.paths = train_paths

    # A padded dataset rather than the training dataset's shape, and the evaluator refuses
    # anything else. It is also the right shape for this: one padded instance per document
    # scores each document on its own, where the training dataset's contiguous blocks would
    # cut documents across instance boundaries and score the fragments.
    #
    # One metadata label per path, because the evaluator names its metrics after them and
    # raises on a path that has none.
    config.trainer = config.trainer.with_callback(
        "held_out",
        LMEvaluatorCallbackConfig(
            eval_dataset=NumpyPaddedFSLDatasetConfig(
                paths=eval_paths,
                metadata=[{"label": hyper_connection_arms.source_label(p)} for p in eval_paths],
                sequence_length=config.dataset.sequence_length,
                tokenizer=config.dataset.tokenizer,
                dtype=config.dataset.dtype,
                work_dir=config.dataset.work_dir,
            ),
            eval_interval=opts.eval_interval,
            eval_on_finish=True,
        ),
    )


def train(config, opts=None):
    arm = ARMS[opts.arm] if opts is not None else None
    if arm is not None and arm.hyper_connections is not None:
        config.trainer = config.trainer.with_callback(
            "hyper_connections",
            HyperConnectionMonitorCallback(
                interval=opts.monitor_interval,
                fail_closed_by_step=opts.fail_closed_by_step,
            ),
        )
    return _train(config, opts)


train_on_corpus.build_parser = build_parser
train_on_corpus.build_config = build_config
train_on_corpus.train = train
train_on_corpus.show = show


if __name__ == "__main__":
    sys.exit(train_on_corpus.cli())
