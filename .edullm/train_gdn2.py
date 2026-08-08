"""Train a Gated DeltaNet-2 model on a published eduLLM corpus.

    python .edullm/train_gdn2.py "$EDULLM_RUN_ID" --save-folder "$EDULLM_CHECKPOINT_DIR" [FLAGS...]

WHAT THIS IS AND IS NOT. It is ``train_on_corpus.py`` with every block's sequence mixer
replaced by :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet2`. It is NOT a second
copy of that file, and the difference matters more than the line count: everything
``train_on_corpus.py`` earned the hard way -- corpus resolution through ``edullm_data.read``
rather than a path literal, the dtype and byte-order assertions, ``max_checkpoints=None``
because the workload role may not prune, the torn-checkpoint repair a Batch retry needs, the
precision refusal that exits before a T4 is billed, and the stage-numbered exit codes that
are the only thing visible from outside a dead container -- is inherited here rather than
re-typed and left to drift.

So this file is two patches and a mixer config:

  * ``build_parser`` gains the GDN-2 geometry flags;
  * ``build_config`` runs with the model factory temporarily wrapped, so the mixer is swapped
    *before* the dotted overrides are merged.

The order in the second one is the whole reason this is not a post-processing step on the
built config. ``train_on_corpus.build_config`` finishes with ``config.merge(overrides)``, and
an override like ``block.sequence_mixer.expand_v=2.0`` is only a valid field once the mixer
has already become a ``GatedDeltaNet2Config``. Swapping afterwards would refuse every override
that names a GDN-2 field, which is most of the reason to pass one.

WHAT GDN-2 CHANGES, in one paragraph, because the flags below are meaningless without it.
``GatedDeltaNet`` erases the old key direction and writes the new value with a single scalar
``beta_t`` per head. GDN-2 splits that into two channel-wise gates -- erase over the key axis,
write over the value axis -- on top of KDA's channel-wise decay. Collapsing both gates to one
scalar recovers KDA; collapsing the decay too recovers ``GatedDeltaNet``. See
https://arxiv.org/abs/2605.22791.

THE IMAGE HAS TO CARRY ``flash-linear-attention``, and it does not by default. The kernel is
``fla.ops.gdn2.chunk_gdn2``, which lives in the project's ``fla`` extra -- and
``.edullm/Dockerfile`` installs ``dependencies`` plus the ``wandb`` extra and nothing else.
That file has a layer for this one; a commit that carries this entry point and not that layer
dies on ``assert has_fla()`` while the model is being built, after the machine is billed.

TORCH.COMPILE IS LEFT ON, which is ``train_on_corpus.py``'s default and is deliberate here.
``chunk_gdn2`` carries ``@torch.compiler.disable``, so Inductor steps around the kernel rather
than trying to trace it. If a graph break in the surrounding module turns out to cost more
than it saves, ``train_module.compile_model=false`` is the override and needs no change here.

A NOTE ON WHAT IS NOT PARAMETERISED. Every block is swapped, uniformly. A hybrid schedule
that interleaves GDN-2 with softmax attention at some ratio is a different experiment and
wants ``block_overrides``, not a flag on this file.
"""

import argparse
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator, List

# Both files live in `.edullm/`, which is not a package. Insert the directory so that
# `train_on_corpus` imports here and also re-imports cleanly in a dataloader worker.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_on_corpus as toc  # noqa: E402

from olmo_core.nn.attention import GatedDeltaNet2Config  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerBlockConfig  # noqa: E402


def mixer_config(opts) -> GatedDeltaNet2Config:
    """Build the GDN-2 mixer config from the parsed flags.

    ``head_dim`` and ``n_v_heads`` are passed through as ``None`` when unset rather than
    computed here, because the layer's own defaults (``d_model // n_heads`` and ``n_heads``)
    depend on a ``d_model`` this function has no honest way to know -- the model factory
    decides it.
    """
    return GatedDeltaNet2Config(
        n_heads=opts.n_heads,
        n_v_heads=opts.n_v_heads,
        head_dim=opts.head_dim,
        expand_v=opts.expand_v,
        allow_neg_eigval=opts.allow_neg_eigval,
        conv_size=opts.conv_size,
    )


def swap_sequence_mixer(model_config: TransformerConfig, mixer: GatedDeltaNet2Config) -> int:
    """Replace the sequence mixer on every block, and return how many were replaced.

    Three shapes are reachable: one shared block config, a dict of them keyed by name, and
    `block_overrides` for a hybrid schedule. The count is returned rather than logged so the
    caller can refuse a zero, which is what a factory whose blocks are none of these looks
    like -- a run that trained the baseline while reporting GDN-2 is the failure this guards.
    """
    swapped = 0

    def apply(block: Any) -> None:
        nonlocal swapped
        if isinstance(block, TransformerBlockConfig):
            block.sequence_mixer = mixer
            swapped += 1

    block = model_config.block
    if isinstance(block, dict):
        for b in block.values():
            apply(b)
    else:
        apply(block)

    for b in (getattr(model_config, "block_overrides", None) or {}).values():
        apply(b)

    return swapped


@contextmanager
def factory_with_gdn2(factory_name: str, mixer: GatedDeltaNet2Config) -> Iterator[None]:
    """Temporarily wrap one `TransformerConfig` factory so its blocks come back as GDN-2.

    The attribute is restored on the way out. ``opts.model_factory`` is deliberately left
    naming the real factory, so the run summary keeps saying which size rung was asked for;
    which mixer it got is in the saved config, where a reader of the record will look.
    """
    original = getattr(TransformerConfig, factory_name, None)
    if original is None:
        # Left to train_on_corpus, which raises this as a staged Refusal with an exit code.
        yield
        return

    def wrapped(**kwargs):
        model_config = original(**kwargs)
        swapped = swap_sequence_mixer(model_config, mixer)
        if swapped == 0:
            raise toc.Refusal(
                toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"model factory {factory_name} exposes no TransformerBlockConfig to swap, so "
                "this run would have trained the factory's own sequence mixer while the "
                "record said GatedDeltaNet2.",
            )
        toc.log.info("swapped the sequence mixer on %d block config(s) for GDN-2", swapped)
        return model_config

    setattr(TransformerConfig, factory_name, wrapped)
    try:
        yield
    finally:
        setattr(TransformerConfig, factory_name, original)


def build_parser() -> argparse.ArgumentParser:
    """`train_on_corpus`'s parser, plus the GDN-2 geometry."""
    parser = _base_build_parser()
    parser.prog = "train_gdn2"
    parser.description = "Train a Gated DeltaNet-2 transformer on a published eduLLM corpus."

    group = parser.add_argument_group("Gated DeltaNet-2")
    group.add_argument(
        "--n-heads",
        type=int,
        default=16,
        help="Number of QK heads in the recurrent mixer. Independent of whatever the model "
        "factory uses for softmax attention, since the mixer is being replaced.",
    )
    group.add_argument(
        "--n-v-heads",
        type=int,
        default=None,
        help="Number of value heads; defaults to --n-heads. Above it, GVA applies. The "
        "recurrent state is (n_v_heads, head_dim, head_dim*expand_v) and does not grow with "
        "sequence length, so raising this buys long-range capacity at constant memory.",
    )
    group.add_argument(
        "--head-dim",
        type=int,
        default=None,
        help="Dimension of each head; defaults to d_model // n_heads.",
    )
    group.add_argument(
        "--expand-v",
        type=float,
        default=1.0,
        help="Value expansion ratio. 1.0 is what the GDN-2 reference ships and keeps "
        "value_dim == d_model at the default head count; GatedDeltaNet's own default is 2.0, "
        "so the two are not parameter-matched unless this is set to match.",
    )
    group.add_argument(
        "--conv-size", type=int, default=4, help="Kernel size of the short causal convolution."
    )
    group.add_argument(
        "--allow-neg-eigval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Widen the erase gate from [0, 1] to [0, 2] to admit negative eigenvalues. The "
        "write gate is left alone. Off by default because the paper's headline model is the "
        "[0, 1] one and its Table 5 finds the widened range gives no consistent gain at 1.3B -- "
        "note that GatedDeltaNet's equivalent flag defaults the other way.",
    )
    return parser


def build_config(opts, overrides: List[str]):
    """`train_on_corpus.build_config`, with the factory wrapped for the duration."""
    with factory_with_gdn2(opts.model_factory, mixer_config(opts)):
        return _base_build_config(opts, overrides)


_base_build_parser = toc.build_parser
_base_build_config = toc.build_config

# The two seams. `toc.main` calls both by name off its own module, so patching the module
# attributes is what puts this file's behaviour inside the whole of its error handling,
# precision refusal, dry-run path and exit-code contract.
toc.build_parser = build_parser
toc.build_config = build_config


if __name__ == "__main__":
    sys.exit(toc.cli())
