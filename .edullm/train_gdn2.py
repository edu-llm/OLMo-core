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
from typing import Any, Dict, Iterator, List

# Both files live in `.edullm/`, which is not a package. Insert the directory so that
# `train_on_corpus` imports here and also re-imports cleanly in a dataloader worker.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_on_corpus as toc  # noqa: E402
from olmo_linear_attn import LinearAttentionConfig  # noqa: E402
from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402

from olmo_core.nn.attention import (  # noqa: E402
    GatedDeltaNet2Config,
    GatedDeltaNetConfig,
)
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402
from olmo_core.nn.transformer.config import TransformerBlockConfig  # noqa: E402


def mixer_config(opts):
    """Build the chosen mixer's config from the parsed flags.

    ``head_dim`` and ``n_v_heads`` are passed through as ``None`` when unset rather than
    computed here, because the layer's own defaults (``d_model // n_heads`` and ``n_heads``)
    depend on a ``d_model`` this function has no honest way to know -- the model factory
    decides it.

    THE GEOMETRY IS SHARED ACROSS MIXERS ON PURPOSE. ``n_heads``, ``n_v_heads``, ``head_dim``,
    ``expand_v`` and ``conv_size`` mean the same thing in all three, so passing one set of flags
    is what makes two arms differ ONLY in their recurrence. ``--head-dim 32`` is the half-KV
    ablation: it halves the key and value projections and the recurrent state together.
    """
    common = dict(
        n_heads=opts.n_heads,
        n_v_heads=opts.n_v_heads,
        head_dim=opts.head_dim,
        expand_v=opts.expand_v,
        conv_size=opts.conv_size,
    )
    # allow_neg_eigval RESOLVES PER MIXER WHEN UNSET, AND SHARING ONE DEFAULT WAS A BUG.
    # GatedDeltaNet's own default is True and the 370M baseline arms trained with True.
    # GDN-2's paper keeps the erase gate in [0, 1] and its Table 5 finds the widened [0, 2]
    # range gives no consistent gain, so its default is False. Passing one flag default to both
    # silently gave GDN the GDN-2 answer -- and because the two settings have IDENTICAL parameter
    # counts, no size or shape check anywhere would have caught it. Only the recurrence changes.
    neg = opts.allow_neg_eigval
    if opts.mixer == "gdn2":
        return GatedDeltaNet2Config(allow_neg_eigval=False if neg is None else neg, **common)
    if opts.mixer == "gdn":
        return GatedDeltaNetConfig(allow_neg_eigval=True if neg is None else neg, **common)
    if opts.mixer == "linear":
        # qk_l2norm=True and normalize=False are the baseline's settings: the pure ungated
        # cumulative sum, which is the honest gate/delta ablation of GatedDeltaNet rather than a
        # differently-normalised linear attention.
        return LinearAttentionConfig(qk_l2norm=True, normalize=False, **common)
    raise ValueError(f"unknown --mixer {opts.mixer!r}")


def swap_sequence_mixer(model_config: TransformerConfig, mixer) -> int:
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
def factory_with_gdn2(factory_name: str, mixer) -> Iterator[None]:
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

    group = parser.add_argument_group("recurrent sequence mixer")
    group.add_argument(
        "--mixer",
        choices=["gdn2", "gdn", "linear"],
        default="gdn2",
        help="Which recurrence to put in every block. All three take the same geometry flags "
        "below, so two arms that differ only in this flag differ only in their recurrence.",
    )
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
        default=None,
        help="Widen the erase gate from [0, 1] to [0, 2] to admit negative eigenvalues. Unset, "
        "it resolves PER MIXER to that mixer's own correct default: False for gdn2 (its paper's "
        "headline model, whose Table 5 finds the widened range gives no consistent gain) and "
        "True for gdn (its own default, and what the 370M baseline arms trained with). Pass it "
        "explicitly to override either.",
    )

    mix = parser.add_argument_group("source mixture")
    mix.add_argument(
        "--mixture",
        choices=["off", "on"],
        default="off",
        help="Read a weighted subset of the corpus rather than all of it, through the reader's "
        "own build_mixture. Off reads the corpus flat, which for a corpus larger than the token "
        "budget means the shards the loader happens to reach first.",
    )
    mix.add_argument(
        "--mixture-total",
        type=int,
        default=10_000_000_000,
        help="Token budget the mixture is drawn to. build_mixture fills with whole shards, so "
        "the result lands within one shard of this rather than exactly on it.",
    )
    mix.add_argument(
        "--mixture-ratios",
        default=None,
        help="YAML of source name to ratio, summing to 1.0, optionally under a `sources:` key. "
        "This is the faithful route when the ratios are known. Omitted, the ratios are derived "
        "by water-filling the budget over whatever the corpus holds.",
    )
    mix.add_argument(
        "--mixture-label-key",
        default="source",
        help="Which shard label the mixture is expressed over. `source` is what a pretrain "
        "corpus carries; a finer mix might use `domain`.",
    )
    mix.add_argument(
        "--mixture-seed",
        type=int,
        default=0,
        help="Seeds the shard draw. Same seed and same inputs give the same shard list, which "
        "is what makes the mixture reproducible from the config.",
    )
    return parser


def water_fill(available: Dict[str, int], total: int) -> Dict[str, float]:
    """Allocate ``total`` across sources equally, capped by what each source holds.

    Water-filling: give every source an equal share; where a source holds less than its share,
    it contributes everything it has and its shortfall is redistributed over the sources that
    still have room. Repeat until nothing moves. The result is a function of the source sizes
    and the budget alone -- there are no weights to choose, which is the property that lets a
    mixture be reconstructed rather than copied from a file.

    :param available: tokens each source holds, keyed by source name.
    :param total: the budget to allocate.

    :returns: each source's share of ``total``, summing to 1.0.

    :raises ValueError: if ``available`` is empty or holds nothing, or ``total`` is not positive.
    """
    if not available or sum(available.values()) <= 0:
        raise ValueError("water_fill needs at least one non-empty source")
    if total <= 0:
        raise ValueError(f"water_fill needs a positive budget; got {total}")

    # The fill runs in FLOAT space and rounds nowhere, which is not fussiness. Truncating each
    # equal share to an int zeroes the whole allocation whenever the budget is smaller than the
    # number of sources -- share < 1 for every source, int(share) == 0 for every source -- and
    # the only symptom would be the "allocated nothing" refusal below on a budget that is
    # perfectly meaningful. Ratios are what this returns anyway, so integers buy nothing.
    remaining, open_sources = float(total), dict(available)
    alloc: Dict[str, float] = {}
    while open_sources and remaining > 0:
        share = remaining / len(open_sources)
        capped = {n: c for n, c in open_sources.items() if c <= share}
        if not capped:  # every open source can take a full share; we are done
            for n in open_sources:
                alloc[n] = alloc.get(n, 0.0) + share
            remaining = 0.0
            break
        for n, c in capped.items():
            alloc[n] = alloc.get(n, 0.0) + float(c)
            remaining -= c
            del open_sources[n]

    drawn = sum(alloc.values())
    if drawn <= 0:
        raise ValueError(f"water_fill allocated nothing from {sum(available.values())} tokens")
    # Normalise to exactly 1.0: build_mixture refuses a sum more than 1e-6 away, because an
    # implicit remainder would silently decide part of the mix.
    ratios = {n: c / drawn for n, c in alloc.items() if c > 0}
    slack = 1.0 - sum(ratios.values())
    widest = max(ratios, key=lambda n: ratios[n])
    ratios[widest] += slack
    return ratios


def mixture_sources(opts, pool) -> list:
    """Turn the resolved shard pool into weighted mixture components.

    ``--mixture-ratios`` names a YAML of ``{source: ratio}`` and is the faithful route when the
    ratios are known. ``--mixture water-fill`` derives them from the pool instead.
    """
    # `_sum_counts` is private, and using it is deliberate rather than lazy. A ManifestEntry's
    # `count` is `{"unit": ..., "value": N}` or absent -- NOT a number -- and only some units are
    # summable at all, so a total is a judgement about units and not an addition. This function
    # is where the reader makes that judgement, and re-deriving it here would be a second
    # opinion that drifts. It is pinned along with the rest of edullm-data by the commit the
    # image installs. Same argument as `_mixture_entries` below.
    #
    # The first version of this line read `int(count)` on the dict and died at
    # THE_READER_FAILED_IN_SOME_OTHER_WAY, exit 67, on run_019fdf9d -- caught by the cpu-32vcpu
    # dry run for $1.43 rather than by eight A100s.
    from edullm_data.read import MixtureSource, _sum_counts

    key = opts.mixture_label_key
    grouped: Dict[str, list] = {}
    for entry in pool:
        name = (getattr(entry, "labels", None) or {}).get(key)
        if name is not None:
            grouped.setdefault(name, []).append(entry)

    available: Dict[str, int] = {}
    units: Dict[str, str] = {}
    for name, entries in grouped.items():
        total, unit = _sum_counts(entries)
        if total is None or total <= 0:
            # Named rather than skipped in silence: a source whose shards declare no summable
            # count would otherwise contribute zero to the mixture and look deliberate.
            toc.log.warning(
                "source %r declares no summable count over %d shard(s); excluded from the "
                "mixture",
                name,
                len(entries),
            )
            continue
        available[name] = total
        units[name] = unit or "?"

    if not available:
        raise toc.Refusal(
            toc.Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS,
            f"no shard in this corpus carries a {key!r} label with a summable count, so a "
            f"mixture over {key} cannot be expressed. Pass --mixture-label-key with a key the "
            f"shards do carry, or --mixture off to read the corpus flat.",
        )
    if len(set(units.values())) > 1:
        raise toc.Refusal(
            toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"the sources disagree about the count unit ({sorted(set(units.values()))}), so a "
            f"single token budget cannot span them. build_mixture refuses this too; saying it "
            f"here names which source carries which unit: "
            + ", ".join(f"{n}={u}" for n, u in sorted(units.items())),
        )

    unit = next(iter(units.values()))
    toc.log.info("mixture pool: %d sources over %r, counted in %s", len(available), key, unit)
    for name in sorted(available, key=lambda n: -available[n]):
        toc.log.info("  %-28s %15d %s", name, available[name], unit)
    toc.log.info("  %-28s %15d %s total", "(all sources)", sum(available.values()), unit)

    if opts.mixture_ratios:
        import yaml

        with open(opts.mixture_ratios) as f:
            declared = yaml.safe_load(f) or {}
        declared = declared.get("sources", declared)
        if not isinstance(declared, dict):
            raise toc.Refusal(
                toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"{opts.mixture_ratios} must be a mapping of source name to ratio, or carry one "
                "under a `sources:` key.",
            )
        unknown = sorted(set(declared) - set(available))
        if unknown:
            raise toc.Refusal(
                toc.Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"{opts.mixture_ratios} names sources this corpus does not have: {unknown}. "
                f"It has {sorted(available)}.",
            )
        ratios = {n: float(r) for n, r in declared.items()}
        toc.log.info("mixture ratios read from %s", opts.mixture_ratios)
    else:
        ratios = water_fill(available, opts.mixture_total)
        toc.log.info("mixture ratios derived by water-filling %d tokens", opts.mixture_total)

    for name in sorted(ratios, key=lambda n: -ratios[n]):
        toc.log.info("  %-28s ratio %.6f", name, ratios[name])
    return [MixtureSource(labels={key: n}, ratio=r) for n, r in ratios.items()]


def resolve_corpus(*, dataset_id: str, version: str, tokenizer_id: str):
    """`train_on_corpus.resolve_corpus`, then narrow it to a weighted mixture.

    The flat resolve runs first and unchanged, so every check it makes -- the seal, the dtype,
    the byte order, the tokenizer identity -- still happens and still produces the same staged
    refusals. Only the path list is replaced.
    """
    corpus = _base_resolve_corpus(dataset_id=dataset_id, version=version, tokenizer_id=tokenizer_id)
    if _MIXTURE_OPTS is None or _MIXTURE_OPTS.mixture == "off":
        return corpus

    from dataclasses import replace

    from edullm_data.read import _mixture_entries, build_mixture
    from edullm_data.s3 import Boto3S3

    opts = _MIXTURE_OPTS
    s3 = Boto3S3.default()
    try:
        pool = _mixture_entries(
            dataset_id, corpus.version, s3=s3, data_bucket="edullm-data", group=None, split=None
        )
        resolved = build_mixture(
            dataset_id,
            corpus.version,
            sources=mixture_sources(opts, pool),
            total=opts.mixture_total,
            seed=opts.mixture_seed,
            s3=s3,
        )
    except toc.Refusal:
        raise
    except BaseException as exc:
        raise toc.Refusal(toc.read_failure(exc), f"{type(exc).__name__}: {exc}") from exc

    toc.log.info(
        "mixture resolved: %d of %d shards, %s %s of a %d budget",
        len(resolved.paths),
        len(corpus.paths),
        f"{resolved.total:,}",
        resolved.unit,
        opts.mixture_total,
    )
    for name, count in sorted(resolved.counts_by_source.items()):
        toc.log.info("  %-28s %15d actual ratio %.6f", name, count, resolved.actual_ratios[name])
    if resolved.shortfall:
        # Not fatal: build_mixture lands within one shard of target by design. Worth saying out
        # loud, because a source that came up short changes the mixture that was asked for.
        toc.log.warning("mixture shortfall by source: %s", dict(resolved.shortfall))

    # Kept for `show` to re-emit at the very end. `edullm logs` returns THE LAST FIFTY LINES the
    # container printed, and --dry-run prints a config far longer than that, so everything above
    # scrolls out of the only window anybody can read -- which is how run_019fdfdf proved the
    # mixture resolves without being able to say what it resolved to.
    global _MIXTURE_SUMMARY
    _MIXTURE_SUMMARY = [
        f"MIXTURE {dataset_id}/{corpus.version}: {len(resolved.paths)} of {len(corpus.paths)} "
        f"shards, {resolved.total:,} {resolved.unit} against a {opts.mixture_total:,} budget "
        f"(seed {opts.mixture_seed}, ratios "
        f"{'from ' + opts.mixture_ratios if opts.mixture_ratios else 'water-filled'})"
    ] + [
        f"  {name:<28} {count:>15,} {resolved.unit}  requested {resolved.requested_ratios[name]:.6f}"
        f"  actual {resolved.actual_ratios[name]:.6f}"
        + (f"  SHORT {resolved.shortfall[name]:,}" if resolved.shortfall.get(name) else "")
        for name, count in sorted(resolved.counts_by_source.items(), key=lambda kv: -kv[1])
    ]

    return replace(corpus, paths=list(resolved.paths))


def show(config) -> None:
    """`train_on_corpus.show`, then the mixture again so it survives the log window.

    Printed AFTER the config rather than instead of it: the config is the thing worth having in
    the record, and the mixture is the thing worth having in the fifty lines a person can read.
    """
    _base_show(config)
    for line in _MIXTURE_SUMMARY:
        print(line)


def build_config(opts, overrides: List[str]):
    """`train_on_corpus.build_config`, with the factory wrapped for the duration."""
    global _MIXTURE_OPTS
    _MIXTURE_OPTS = opts
    with factory_with_gdn2(opts.model_factory, mixer_config(opts)):
        return _base_build_config(opts, overrides)


_MIXTURE_OPTS = None
_MIXTURE_SUMMARY: List[str] = []
_base_build_parser = toc.build_parser
_base_build_config = toc.build_config
_base_resolve_corpus = toc.resolve_corpus
_base_show = toc.show

# The four seams. `toc.main` and `toc.build_config` call these by name off their own module,
# so patching the module attributes is what puts this file's behaviour inside the whole of
# train_on_corpus's error handling, precision refusal, dry-run path and exit-code contract.
toc.build_parser = build_parser
toc.build_config = build_config
toc.resolve_corpus = resolve_corpus
toc.show = show


@record
def _main() -> int:
    """`toc.cli`, wrapped so a distributed failure reports its own traceback.

    WITHOUT @record, torchrun's summary for a crashed rank reads

        rank : 4 (local_rank: 4)  exitcode : 1  error_file: <N/A>
        traceback : To enable traceback see: .../elastic/errors.html

    and the child's actual exception is somewhere above -- which on this platform means
    outside the fifty lines `edullm logs` returns, because torchrun's own wrapper output fills
    the window. run_019ff4f4 failed exactly that way and could not be attributed at all.

    @record writes each rank's exception to an error file and torchrun then prints it IN the
    summary, so the traceback lands inside the readable window. This is the whole diagnosis
    fix; it changes nothing about what the run computes.
    """
    return toc.cli()


if __name__ == "__main__":
    sys.exit(_main())
