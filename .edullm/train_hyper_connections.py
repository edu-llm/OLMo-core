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
    parser.set_defaults(
        model_factory="olmo3_370M",
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
        "--fail-closed-by-step",
        type=int,
        default=None,
        help="Stop the run with an error if the lanes have not differentiated by this step. "
        "SET THIS ON THE REHEARSAL. Identical lane norms mean the mechanism is inert, and a "
        "downstream number from an inert mechanism is not interpretable either way -- better "
        "to find that out for a few dollars than for a few hundred.",
    )
    return parser


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

    return config


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


if __name__ == "__main__":
    sys.exit(train_on_corpus.cli())
