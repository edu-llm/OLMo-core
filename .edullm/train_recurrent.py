#!/usr/bin/env python3
"""Train the recurrent-depth model on a published eduLLM corpus.

NOTHING OF ``train_on_corpus.py`` IS COPIED HERE, ON PURPOSE. That file is 973 lines and
almost all of them are things this run wants unchanged: resolving a corpus from the sealed
manifest, refusing with a stage number that survives a container nobody can read the logs
of, the uint32 dtype fix without which a dolma2 corpus is read two bytes at a time, the
torn-checkpoint repair a Batch retry needs, and the summary the platform parses back out of
the log stream. A forked copy would start drifting from all of it on the first upstream fix.

So this is a wrapper. It works because the platform's runner resolves a model with

    factory = getattr(TransformerConfig, opts.model_factory, None)

which is an attribute lookup on a class, not a fixed table. ``olmo_recurrent.install()``
puts ``recurrent_olmo3_370M`` on that class, and from there ``--model-factory
recurrent_olmo3_370M`` reaches it through the stock line.

Three things do need wrapping, and each is a module-global rebind that ``train_on_corpus.main``
picks up because it resolves those names at call time:

  ``build_parser``  adds the recurrence flags and re-points ``--model-factory`` at the
                    recurrent default, so an unknown flag does not fall through
                    ``parse_known_args`` into ``config.merge`` and get read as a bad override.
  ``build_config``  re-derives the residual alpha after the merge, so a ``--n-loops`` or a
                    dotted ``model.max_loops=`` override cannot leave the model with a scale
                    computed for a different depth.
  ``train``         registers the depth-schedule callback, which has nowhere else to go: the
                    trainer is built inside ``train`` and the callback has to be attached
                    before ``fit``.

Run it the way the platform runs anything, with the dtype in the command text so the
submission check can see it::

    bash -lc 'python .edullm/train_recurrent.py "$EDULLM_RUN_ID" \
        --save-folder "$EDULLM_CHECKPOINT_DIR" \
        --model-factory recurrent_olmo3_370M --n-loops 4 \
        --sequence-length 4096 --steps 20000 \
        train_module.dp_config.param_dtype=bfloat16'

Locally, with no corpus and no GPU::

    python .edullm/train_recurrent.py test --dry-run \
        --dataset-id pretrain/regmix-10b --dataset-version v1 \
        --dataset-tokenizer tokenizer/dolma2-bpe --save-folder /tmp/x
"""

import os
import sys

# Both this file and its sibling have to be importable by name, and by a name that is
# stable. `_CLASS_` in the saved config records `olmo_recurrent.RecurrentTransformerConfig`,
# and anything reading that config back resolves it with `importlib.import_module`, so the
# module has to be reachable as `olmo_recurrent` rather than as `__main__`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import olmo_recurrent  # noqa: E402
import train_on_corpus  # noqa: E402

olmo_recurrent.install()

_build_parser = train_on_corpus.build_parser
_build_config = train_on_corpus.build_config
_train = train_on_corpus.train


def build_parser():
    parser = _build_parser()
    parser.prog = "train_recurrent"
    parser.description = "Train a recurrent-depth transformer on a published eduLLM corpus."

    # Re-point rather than add: the platform's default is olmo2_190M, and a recurrent run
    # that silently trained a dense 190M would be very hard to notice from the loss curve.
    parser.set_defaults(model_factory="recurrent_olmo3_370M")

    parser.add_argument(
        "--n-loops",
        type=int,
        default=None,
        help="Recurrent depth T. Sets default_n_loops AND max_loops together, because the "
        "residual scale is 1/(max_loops*sqrt(L)) and training at a depth the scale was not "
        "computed for is a different model. Reach the two separately with dotted overrides "
        "(model.default_n_loops=, model.max_loops=) if that is what you mean.",
    )
    parser.add_argument(
        "--depth-schedule",
        action="store_true",
        help="Vary T between optimizer steps on the staged schedule instead of holding it "
        "fixed. OFF BY DEFAULT and the default is the right first run: a fixed T is "
        "deterministic, keeps every microbatch in an accumulation window at one depth, and "
        "gives torch.compile a single shape to specialize on.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=["off", "full", "selected_blocks"],
        default="off",
        help="Recompute block activations in the backward pass instead of storing them. This "
        "matters far more here than for a plain stack: a loop at T stores activations for "
        "n_prelude + T*L + n_coda block applications, which is 52 at T=4 against the "
        "baseline's 16. Verified bit-exact against the unwrapped model. There is no dotted "
        "override for this, because train_module.ac_config is None by default and a merge "
        "cannot set a field on None.",
    )
    parser.add_argument(
        "--ac-block-interval",
        type=int,
        default=2,
        help="With --activation-checkpointing selected_blocks, wrap every Nth block.",
    )
    return parser


def build_config(opts, overrides):
    config = _build_config(opts, overrides)

    # Applies whatever the model is, because the flag exists to reach a field the merge
    # cannot, and that is true of the baseline arm of a comparison too.
    if getattr(opts, "activation_checkpointing", "off") != "off":
        from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
        from olmo_core.train.train_module import TransformerActivationCheckpointingConfig

        mode = TransformerActivationCheckpointingMode(opts.activation_checkpointing)
        config.train_module.ac_config = TransformerActivationCheckpointingConfig(
            mode=mode,
            block_interval=(
                opts.ac_block_interval
                if mode == TransformerActivationCheckpointingMode.selected_blocks
                else None
            ),
        )

    model = config.model
    if not isinstance(model, olmo_recurrent.RecurrentTransformerConfig):
        return config

    if getattr(opts, "n_loops", None) is not None:
        model.default_n_loops = opts.n_loops
        model.max_loops = opts.n_loops
        model.min_loops = min(model.min_loops, opts.n_loops)

    # Unconditionally, and after the merge rather than before it. `--n-loops` is not the only
    # way the depth moves: `model.max_loops=8` on the command line reaches the config through
    # `config.merge(overrides)` inside the call above, which happens after the factory already
    # wrote the alphas. Re-deriving here is idempotent and is the only point at which the
    # depth is final.
    model.apply_recurrent_residual_alpha()
    return config


def train(config, opts=None):
    if getattr(opts, "depth_schedule", False) and isinstance(
        config.model, olmo_recurrent.RecurrentTransformerConfig
    ):
        config.trainer = config.trainer.with_callback(
            "recurrent_depth",
            olmo_recurrent.RecurrentDepthCallback(
                min_depth=config.model.min_loops,
                max_depth=config.model.max_loops,
                seed=config.init_seed,
            ),
        )
    return _train(config, opts)


train_on_corpus.build_parser = build_parser
train_on_corpus.build_config = build_config
train_on_corpus.train = train


if __name__ == "__main__":
    sys.exit(train_on_corpus.cli())
