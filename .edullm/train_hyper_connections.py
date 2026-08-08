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
                    hc_370M, so that an arm cannot silently train the platform's default
                    190M.
  ``build_config``  resolves which replicate this process is, applies the arm to the model
                    config, moves all three seeds together, and installs the weight-decay
                    split.
  ``train``         attaches the lane monitor, which has nowhere else to go: the trainer is
                    built inside ``train``.

THE WHOLE TRANCHE IS ONE SUBMISSION, WHICH IS WHAT ``resolve_cell`` IS FOR. The platform fans
a submission out with ``--fanout-size N --fanout-index-parameter <label>`` and gives each cell
its own ``AWS_BATCH_JOB_ARRAY_INDEX``, its own checkpoint prefix and its own W&B run id. Two
labels are read here and they differ in how much of the cell the index decides:

  ``seed``          three cells, one arm, three replicates. The arm comes from ``--arm``.
  ``arm-and-seed``  nine cells, and the index names an ``(arm, seed)`` pair out of
                    ``hyper_connection_arms.TRANCHE_CELLS``. The command carries neither
                    ``--arm`` nor ``--seed``, so one commit is the whole tranche. This is what
                    the two staged tranche specs use.

Under either label the flag the index owns is refused rather than honoured if the command
passes it anyway, because every cell of a fan-out is handed one command: a ``--seed`` there
would run one replicate N times and an ``--arm`` would run one arm nine times. Neither raises,
neither bends a loss curve, and the noise floor the analysis plan divides by would be zero.

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

#: The 370M run, at the horizon that fits one attempt. 4.72B dolma2 tokens at seq 4096 over a
#: 768K-token batch. See ``TRANCHE_STEPS`` in hyper_connection_arms.py for why this is 6,000
#: and not the 12,715 that would be 10B: 12,715 is 37.7 hours and the workload's per-attempt
#: ceiling is 24, and the second attempt that would cover the difference is not one this
#: platform's retry rules reliably grant.
DEFAULT_STEPS = hyper_connection_arms.TRANCHE_STEPS
DEFAULT_LEARNING_RATE = 7.8e-4
DEFAULT_WEIGHT_DECAY = 0.033

#: What Batch sets in each child of an array job, and the only place a cell's own index
#: exists. The platform's ``execution.py`` names the same variable and assigns it per child,
#: so a fan-out of three cells is three processes reading 0, 1 and 2 out of here.
FANOUT_INDEX_VARIABLE = "AWS_BATCH_JOB_ARRAY_INDEX"

#: What the platform says the index *means*, set beside the index itself from
#: ``--fanout-index-parameter``. A label and not a variable name -- a cell varying the
#: checkpoint and a cell varying the seed read the same integer out of the same place, and
#: this is the only thing that tells them apart.
FANOUT_PARAMETER_VARIABLE = "EDULLM_FANOUT_INDEX_PARAMETER"

#: The label under which an index means a replicate number, for a fan-out of one arm.
FANOUT_INDEX_PARAMETER = "seed"

#: The label under which an index means one cell of the whole nine-run tranche.
#:
#: WHY THERE ARE TWO LABELS AND NOT ONE. A three-cell fan-out varies the seed and takes its
#: arm from ``--arm`` in the command, which is how the first tranche was going to be
#: submitted: three commits, three submissions, and the arm edited in ``run.yaml`` between
#: them. A nine-cell fan-out varies both, so the command carries neither ``--arm`` nor
#: ``--seed`` and every cell reads its pair out of ``hyper_connection_arms.TRANCHE_CELLS``.
#:
#: The nine-cell form is what the staged tranches use, and it is better in the one way that
#: matters under time pressure: three submissions of one file mean the file has to be edited
#: and committed twice while the first submission is already running, and an arm that ran
#: under the wrong commit is not something the run's own output can say. One commit, one
#: submission, nine cells, and the arm each cell ran is derived from a table under test.
FANOUT_INDEX_PARAMETER_CELL = "arm-and-seed"


def resolve_seed(explicit, environ=None):
    """
    Work out which replicate this process is, and say where the answer came from.

    THE FAILURE THIS IS SHAPED AROUND IS SILENT AND IT DESTROYS THE TRANCHE. Three seeds of an
    arm exist to estimate sigma; three cells that all resolved to seed 0 are three
    bit-identical runs, and the noise floor they report is zero. Every contrast in the
    analysis plan is then divided by zero variance and every arm looks significant. Nothing
    about the loss curves would look wrong -- they would look *perfect*, three lines on top of
    each other -- so there is no downstream check that catches it. It has to be refused here.

    Precedence, in order:

    1. ``--seed`` on the command line wins, so a single run can be reproduced by hand. But a
       command that carries an explicit ``--seed`` **and** lands in a fan-out is refused
       rather than resolved: that is exactly the collapse above, and "the explicit one wins"
       is the reasoning that produces it.
    2. Otherwise the fan-out index, if Batch set one -- and only if the platform also said the
       index means the seed. An index under any other label is refused, because reading a
       checkpoint sweep's index as a replicate number is the same accident wearing a
       different hat.
    3. Otherwise 0, which is what a laptop, a preflight and a single submitted run all get.

    :param explicit: The value of ``--seed``, or ``None`` if it was not passed.
    :param environ: The environment to read. Defaults to ``os.environ``.

    :returns: ``(seed, provenance)``, where the provenance is one sentence naming where the
        number came from. It goes into the log and into the preflight summary.

    :raises train_on_corpus.Refusal: If a fan-out index is present alongside an explicit
        ``--seed``, if it is present under the wrong label, or if it is not an integer.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(FANOUT_INDEX_VARIABLE)
    label = environ.get(FANOUT_PARAMETER_VARIABLE)

    if raw is None or raw == "":
        if explicit is not None:
            return explicit, f"--seed {explicit} on the command line"
        return 0, "the default, since no --seed was given and this is not a fan-out cell"

    if explicit is not None:
        raise train_on_corpus.Refusal(
            train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"this is cell {raw} of a fan-out and the command also passes --seed {explicit}. "
            "Every cell of a fan-out is handed the same command, so honouring the explicit "
            "seed would run the whole array on one replicate: identical models, identical "
            "data order, and a measured noise floor of zero that would make every arm in the "
            f"tranche look significant. Drop --seed from the command in .edullm/run.yaml and "
            f"let ${FANOUT_INDEX_VARIABLE} choose, or submit without --fanout-size.",
        )

    if label != FANOUT_INDEX_PARAMETER:
        raise train_on_corpus.Refusal(
            train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"this is cell {raw} of a fan-out whose index means {label!r}, and this program "
            f"only reads a fan-out index that means {FANOUT_INDEX_PARAMETER!r}. Submit with "
            f"--fanout-index-parameter {FANOUT_INDEX_PARAMETER}. An index read under the "
            "wrong label is a replicate number taken from something that was never counting "
            "replicates, and nothing downstream would notice.",
        )

    return _index(raw), f"${FANOUT_INDEX_VARIABLE}={int(raw)}, the fan-out cell index"


def _index(raw):
    """
    The array index as an integer, or a refusal naming what arrived instead.

    :raises train_on_corpus.Refusal: If Batch set something that is not an integer.
    """
    try:
        return int(raw)
    except ValueError:
        raise train_on_corpus.Refusal(
            train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"${FANOUT_INDEX_VARIABLE} is {raw!r}, which is not an integer, so this cell has "
            "no replicate number to be.",
        ) from None


def resolve_cell(explicit_arm, explicit_seed, environ=None):
    """
    Work out which arm and which replicate this process is.

    THE NINE-CELL TRANCHE IS ONE SUBMISSION AND THE INDEX CARRIES BOTH FACTS. Under
    ``--fanout-index-parameter arm-and-seed`` the command carries neither ``--arm`` nor
    ``--seed``, and this reads the pair out of :data:`hyper_connection_arms.TRANCHE_CELLS`,
    which is derived from the seed counts in the arm table. Cell 0 is ``baseline`` seed 0 and
    cell 8 is ``output-only`` seed 2, and a test walks all nine.

    Everything else is unchanged and delegates to :func:`resolve_seed`: a three-cell fan-out
    labelled ``seed`` still takes its arm from ``--arm``, and a laptop with no fan-out at all
    still gets seed 0.

    Both explicit flags are refused inside an ``arm-and-seed`` fan-out rather than honoured,
    for the reason ``resolve_seed`` gives about ``--seed``: every cell of a fan-out is handed
    one command, so an ``--arm`` written into ``run.yaml`` would run nine cells of one arm and
    a ``--seed`` would run them all on one replicate. Neither produces an error or a visibly
    wrong curve.

    :param explicit_arm: The value of ``--arm``, or ``None``.
    :param explicit_seed: The value of ``--seed``, or ``None``.
    :param environ: The environment to read. Defaults to ``os.environ``.

    :returns: ``(arm_name, seed, provenance)``.

    :raises train_on_corpus.Refusal: If a flag the cell index owns was passed anyway, if the
        index is outside the tranche, or if ``--arm`` is missing where nothing else supplies
        one.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(FANOUT_INDEX_VARIABLE)
    label = environ.get(FANOUT_PARAMETER_VARIABLE)

    if raw not in (None, "") and label == FANOUT_INDEX_PARAMETER_CELL:
        for flag, value in (("--arm", explicit_arm), ("--seed", explicit_seed)):
            if value is not None:
                raise train_on_corpus.Refusal(
                    train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                    f"this is cell {raw} of an {FANOUT_INDEX_PARAMETER_CELL!r} fan-out, whose "
                    f"index is what says which arm and which replicate this cell is, and the "
                    f"command also passes {flag} {value}. Every cell is handed the same "
                    f"command, so honouring it would run the whole tranche as one cell "
                    f"repeated nine times -- one arm or one replicate, a measured noise floor "
                    f"of zero, and nine curves lying on top of each other that read as a very "
                    f"clean experiment. Drop {flag} from the command in .edullm/run.yaml.",
                )
        index = _index(raw)
        try:
            name, seed = hyper_connection_arms.cell(index)
        except IndexError as mismatch:
            raise train_on_corpus.Refusal(
                train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD, str(mismatch)
            ) from None
        return (
            name,
            seed,
            f"${FANOUT_INDEX_VARIABLE}={index}, which is cell {index} of "
            f"{len(hyper_connection_arms.TRANCHE_CELLS)} in the tranche table",
        )

    if explicit_arm is None:
        raise train_on_corpus.Refusal(
            train_on_corpus.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "no --arm, and this is not a cell of an "
            f"{FANOUT_INDEX_PARAMETER_CELL!r} fan-out either, so nothing says which arm to "
            "run. Pass --arm, or submit with --fanout-index-parameter "
            f"{FANOUT_INDEX_PARAMETER_CELL} and let the cell index choose.",
        )

    seed, provenance = resolve_seed(explicit_seed, environ)
    return explicit_arm, seed, provenance


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
    # save_interval is 500 and not the 2000 it was, because 2000 was priced against nothing.
    # At the 10.32 s/step the 370M probe measured at rank microbatch 16,384, 500 steps is 1.43
    # hours, which is what a lost host throws away. The workload profile declares a 30-minute
    # checkpoint interval and 500 does not reach it -- that would need about 175 steps -- but
    # the interval is a declaration nothing enforces, and 500 is where the loss stops being
    # the dominant risk without making a 46-second checkpoint a visible fraction of the run.
    #
    # It buys 13 checkpoints per arm at the tranche's 6,000 steps: one at step zero and twelve
    # on the interval, the last of which is the final step. Nothing anywhere else states that
    # count, so it is stated here, and `arm_seconds` in hyper_connection_arms.py prices it.
    # All 13 are kept: CheckpointerCallback runs with max_checkpoints=None because the
    # workload role cannot prune, for the reason train_on_corpus.py gives beside that field.
    #
    # warmup_steps stays at 2% of the run rather than at the 254 it was, so the schedule keeps
    # its shape when the horizon moves. At 6,000 steps that is 120.
    parser.set_defaults(
        model_factory="hc_370M",
        sequence_length=4096,
        global_batch_size=768 * 1024,
        rank_microbatch_size=8 * 1024,
        steps=DEFAULT_STEPS,
        warmup_steps=round(DEFAULT_STEPS * 0.02),
        save_interval=500,
        learning_rate=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--arm",
        choices=sorted(ARMS),
        default=None,
        help="Which arm to run. See --list-arms. LEAVE THIS UNSET IN A NINE-CELL TRANCHE: the "
        f"cell then takes its arm and its replicate together from ${FANOUT_INDEX_VARIABLE}, "
        f"under the {FANOUT_INDEX_PARAMETER_CELL!r} label. Passing it there is refused rather "
        "than honoured, because one command reaching nine cells would run one arm nine times. "
        "Required everywhere else, and a run with neither is refused rather than defaulted -- "
        "there is no arm this experiment can silently mean.",
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
        default=None,
        help="Replicate index. Moves parameter initialization, the shuffle and the global RNG "
        f"together, because a 'seed' that only reshuffles the data measures less variance "
        f"than the run actually has and would understate the noise floor everything else is "
        f"compared against. LEAVE THIS UNSET IN A FAN-OUT: each cell then takes its replicate "
        f"number from ${FANOUT_INDEX_VARIABLE}, which is the only per-cell value that exists. "
        f"Passing it explicitly inside a fan-out is refused rather than honoured, because one "
        f"command reaching three cells would run one replicate three times. See "
        f"`resolve_seed`. Unset outside a fan-out means 0.",
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
        "--z-loss-multiplier",
        type=float,
        default=1e-5,
        help="Auxiliary loss on the log-partition, which the 370M configuration calls for and "
        "which train_on_corpus leaves unset. It is the cheapest instrument for the failure "
        "this architecture is most exposed to: RMSNorm readouts are scale-invariant, so "
        "cross-entropy cannot see hidden-state scale at all, and the rehearsal's hidden norms "
        "swung by 50% with nothing in the loss curve reflecting it. Set to 0 to disable.",
    )
    parser.add_argument(
        "--partial-rotary-factor",
        type=float,
        default=None,
        help="Fraction of each head's channels that receive RoPE. A separate track from the "
        "arms and composable with any of them: nobody has measured in-distribution "
        "bits-per-byte against this fraction at any scale with a noise floor, and it is free "
        "-- no parameters, no change to accounted FLOPs, only which channels carry positional "
        "phase. 1.0 is ordinary RoPE and 0.0 is NoPE. Leave unset to touch nothing.",
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

    # WHICH REPLICATE THIS CELL BECAME, AND WHERE THE NUMBER CAME FROM. Three cells resolving
    # to the same seed is the one failure in this tranche that produces no error and no
    # visibly wrong curve, so it is checked by looking, on a laptop, before anything is
    # submitted -- and the three init seeds are printed beside it so a fan-out can be
    # verified by running the preflight three times with a different index in the
    # environment.
    if _SEED_PROVENANCE:
        print(f"cell         {_SEED_PROVENANCE[0]}")
    print(
        f"seeded       init={config.init_seed}, model.init={config.model.init_seed}, "
        f"data={config.data_loader.seed}"
    )

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
    # FIRST, AND LOUDLY. Everything below reads opts.arm and opts.seed, and a run that
    # resolved either one wrongly is not recoverable from its own output -- see
    # `resolve_cell`. The line goes to the log on every run and to the preflight summary, so
    # what a cell became is written down somewhere a person will actually look.
    opts.arm, opts.seed, provenance = resolve_cell(opts.arm, opts.seed)
    _SEED_PROVENANCE[:] = [f"arm {opts.arm}, seed {opts.seed}, from {provenance}"]
    print(f"cell         {_SEED_PROVENANCE[0]}", flush=True)

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

    if opts.z_loss_multiplier > 0:
        config.train_module.z_loss_multiplier = opts.z_loss_multiplier

    config.train_module.optim.weight_decay = opts.weight_decay
    if opts.arm != "decay-everything":
        overrides_for_hc = arm.optim_group_overrides(weight_decay=opts.weight_decay)
        if overrides_for_hc:
            config.train_module.optim.group_overrides = (
                list(config.train_module.optim.group_overrides or []) + overrides_for_hc
            )

    if opts.partial_rotary_factor is not None:
        hyper_connection_arms.apply_partial_rotary(config.model, opts.partial_rotary_factor)

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

#: How this process worked out which replicate it is, in one sentence, for the preflight
#: summary. A list for the same reason `_PREFLIGHT` is one.
_SEED_PROVENANCE: list = []


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
            eval_on_startup=True,
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
