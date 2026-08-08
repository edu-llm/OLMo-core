"""The arms of the hyper-connection module, as one table.

Every arm is the same 370M OLMo-3 model with one thing changed, so that the difference between
any two of them is the difference between two named hypotheses rather than between two scripts.
Keeping them in a table rather than in eleven config files is what makes that checkable: the
test suite walks ``ARMS`` and asserts each arm differs from the baseline in exactly the fields
it claims to.

The arms exist because hyper-connections have been measured twice at essentially the same
parameter scale with opposite signs -- ByteDance report -0.030 loss and +1.3 downstream at n=4
on OLMo-1B over 500B tokens, and Tencent measure -0.020 downstream at 1.2B dense with
divergence at 3B. Arms 3, 4 and 5 are the three documented differences between those two
setups, one per arm.

An arm is applied to a size rather than being its own size, so the same eleven arms run at the
rehearsal size and at 370M without a second table to keep in sync.

WHAT THE FIRST TRANCHE FUNDS, AND WHY IT IS THREE ARMS RATHER THAN SEVEN.

Nine runs: ``baseline`` x3, ``faithful`` x3, ``output-only`` x3, and zero everywhere else.
That buys exactly two hypotheses, both at full power and both against the same noise floor:

  H1   replication          arm 2 against arm 1, three seeds against three.
  H2a  implementation       arm 3 against arm 2, three seeds against three. Whether the
                            field's negative result is an artifact of a reimplementation
                            that kept the output mixing and dropped the input map.

Three against three is the smallest design in which sigma is estimated from the data rather
than assumed, and it is what the analysis plan's gate -- two standard errors of the contrast
-- was written against. Two arms at two seeds and two more at one, which is what the previous
allocation bought, gives four underpowered answers instead of two sharp ones.

``mhc`` / H5 IS DEFERRED TO A SECOND TRANCHE AND IS NOT ABANDONED. It is the best-designed
hypothesis in this module: the Sinkhorn-Knopp normalization towards the Birkhoff polytope is
a mechanism claim with a spectral prediction the monitor already instruments, and it ships in
DeepSeek V4, so a null there is publishable and a positive is load-bearing. It is last in
:data:`CUT_ORDER` for that reason -- the last thing cut and the first thing a second tranche
restores. Dropping it silently would be the loss; dropping it in writing, with the three runs
it needs already specified and tested, costs a number and not a design.

``output-only`` IS ONLY WORTH RUNNING BECAUSE OF COMMIT ``b7983ea9``. Before that commit the
arm replaced both the learned input map and the paper's fixed staggered one-hot read, which
left every lane reading the same vector: degenerate rather than crippled, an exact re-run of
the baseline with dead parameters, and a null result that meant nothing. b7983ea9 put the
staggered read back so that only the learned input map goes, which is the one difference H2a
is about. Nine runs of a degenerate arm is the shape of mistake this note exists to prevent.
"""

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from olmo_core.nn.residual_stream import HyperConnectionConfig, HyperConnectionMode
from olmo_core.nn.transformer import (
    BlockReuseConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.train.callbacks import Callback

#: The expansion rate ByteDance found best. n=8 was barely better and n=1 was worse.
N_LANES = 4

#: How many times each block is reused in the tied arms. Two, so the cycle visits each block
#: twice. Expressed as a factor rather than a block count so that the tied arms still tie at
#: the rehearsal size -- an absolute count of 8 would be a silent no-op on an 8-layer model,
#: and the rehearsal would pass without ever running the code those arms depend on.
REUSE_FACTOR = 2


def _hc(**overrides) -> HyperConnectionConfig:
    return replace(HyperConnectionConfig(n_lanes=N_LANES), **overrides)


@dataclass(frozen=True)
class Arm:
    """One row of the experiment."""

    number: int
    """Its number in the pre-registration, so a run name maps back to the plan."""

    summary: str
    """What this arm is."""

    isolates: str
    """What it is for. An arm that cannot answer this is not an arm."""

    seeds: int
    """
    How many seeds this arm is funded for in the tranche that is about to run.

    Three where a three-versus-three difference is claimed against the baseline, and zero
    everywhere else. There is no longer a one and no longer a two: an arm at one seed cannot
    separate its effect from the seed it drew, and an arm at two estimates sigma from a single
    difference. The budget buys two hypotheses at full power rather than six at partial, and
    :data:`CUT_ORDER` records which order the rest come back in.

    Zero is not the same as absent: the arm still has to build, stay iso-parameter, stay
    iso-FLOP and pass every test in ``test_hyper_connection_arms.py``, so funding it later
    costs a number and not a design.
    """

    hyper_connections: Optional[HyperConnectionConfig] = None
    """``None`` for an ordinary residual stream."""

    reuse_factor: Optional[int] = None
    """How many times each block runs, for the tied arms. ``None`` for one block per layer."""

    def apply(self, config: TransformerConfig) -> TransformerConfig:
        """
        Turn a stock size config into this arm.

        :param config: A config from a :class:`TransformerConfig` factory. Mutated in place and
            also returned.

        :raises ValueError: If the config is not the reordered-norm baseline the arms are
            defined against, or if its depth cannot carry the reuse factor.
        """
        if self.hyper_connections is not None:
            if isinstance(config.block, dict):
                raise ValueError(
                    "Hyper-connection arms need a single block config, not a named-block dict."
                )
            if config.block.name != TransformerBlockType.reordered_norm:
                raise ValueError(
                    "The arms are defined against the reordered-norm baseline that OLMo-2 and "
                    f"OLMo-3 use; this config is {config.block.name}. Changing the norm "
                    "placement underneath an arm would confound it."
                )
            config.block.name = TransformerBlockType.hyper_connection_reordered_norm
            config.block.hyper_connections = self.hyper_connections

        if self.reuse_factor is not None:
            if config.n_layers % self.reuse_factor != 0:
                raise ValueError(
                    f"A reuse factor of {self.reuse_factor} does not divide "
                    f"{config.n_layers} layers, so the cycle would visit some blocks more "
                    "often than others and the arm would not be a clean contrast."
                )
            config.block_reuse = BlockReuseConfig(
                n_unique_blocks=config.n_layers // self.reuse_factor
            )

        config.__post_init__()
        return config

    def optim_group_overrides(self, weight_decay: float) -> list:
        """
        The weight-decay split, empty for an arm with no hyper-connections.
        """
        if self.hyper_connections is None:
            return []
        return self.hyper_connections.optim_group_overrides(weight_decay=weight_decay)


ARMS: Dict[str, Arm] = {
    "baseline": Arm(
        number=1,
        summary="Standard residual stream.",
        isolates="The noise floor. Nothing else can be claimed until this has been measured.",
        seeds=3,
    ),
    "faithful": Arm(
        number=2,
        summary="DHC x4 as published: input-side pre-mapping, sqrt(n) output init, "
        "weight-decay split.",
        isolates="The actual method.",
        seeds=3,
        hyper_connections=_hc(),
    ),
    "output-only": Arm(
        number=3,
        summary="DHC x4 with output-side mixing only, which is what a shared residual "
        "interface forces. The read stays on the paper's fixed staggered one-hot; only the "
        "learned input map goes, or the arm is degenerate rather than crippled. IT WAS "
        "DEGENERATE UNTIL COMMIT b7983ea9, which put the staggered read back: before it, "
        "every lane read the same vector and this arm was the baseline with dead parameters. "
        "It is in the tranche because of that fix and would not otherwise be worth a run.",
        isolates="Cause 1, and H2a. Whether the field's negative result is an artifact of an "
        "incomplete reimplementation.",
        seeds=3,
        hyper_connections=_hc(mode=HyperConnectionMode.output),
    ),
    "no-output-init": Arm(
        number=4,
        summary="Faithful, but without the sqrt(n) output-module initialization scaling.",
        isolates="Cause 2. Whether that scaling is load-bearing or cosmetic.",
        seeds=0,
        hyper_connections=_hc(output_init_exponent=0.0),
    ),
    "decay-everything": Arm(
        number=5,
        summary="Faithful, but with weight decay on the static component too.",
        isolates="Cause 3. The parameter-group split the replication does not mention.",
        seeds=0,
        # The split lives in the optimizer, so this arm is the faithful model with
        # `optim_group_overrides` deliberately not applied. See `train_hyper_connections.py`.
        hyper_connections=_hc(),
    ),
    "n1": Arm(
        number=6,
        summary="DHC x1, which also loses the output-init rescale: `output_init_scale` "
        "returns 1.0 at n=1, because with one lane there is no sum to compensate.",
        isolates="The seesaw control. ByteDance found n=1 does not beat the baseline; if it "
        "does here, their mechanism story is incomplete at this scale. Read against arm 4 and "
        "not arm 2, since arm 2 carries the rescale that this arm cannot have.",
        seeds=0,
        hyper_connections=_hc(n_lanes=1),
    ),
    "n2": Arm(
        number=7,
        summary="DHC x2.",
        isolates="The expansion-rate curve.",
        seeds=0,
        hyper_connections=_hc(n_lanes=2),
    ),
    "n8": Arm(
        number=8,
        summary="DHC x8.",
        isolates="The expansion-rate curve, at the point where they found returns flatten.",
        seeds=0,
        hyper_connections=_hc(n_lanes=8),
    ),
    "mhc": Arm(
        number=9,
        summary="mHC x4: the lane-mixing matrix normalized towards the Birkhoff polytope by "
        "Sinkhorn-Knopp, which at eight sweeps lands column-stochastic rather than doubly "
        "stochastic and carries the spectral radius mHC argues from either way.",
        isolates="H5. Whether the constraint is what rescues the method. It ships in DeepSeek "
        "V4. DEFERRED TO A SECOND TRANCHE AND NOT ABANDONED: it is last in CUT_ORDER, which "
        "makes it the first three runs a second tranche buys.",
        seeds=0,
        hyper_connections=_hc(doubly_stochastic=True),
    ),
    "tied-faithful": Arm(
        number=10,
        summary="Tied blocks on a cycle, with DHC x4.",
        isolates="Cause 5. Whether lane value tracks parameter reuse rather than model size.",
        seeds=0,
        hyper_connections=_hc(),
        reuse_factor=REUSE_FACTOR,
    ),
    "tied-baseline": Arm(
        number=11,
        summary="Tied blocks on a cycle, standard residual stream.",
        isolates="The control for arm 10. Without it arm 10 measures tying, not lanes.",
        seeds=0,
        reuse_factor=REUSE_FACTOR,
    ),
}


#: The order to cut arms in if the budget does not stretch, and therefore the order a second
#: tranche restores them in, read from the end.
#:
#: Eight of the eleven arms are now unfunded, and this list is all eight of them followed by
#: nothing: the three that remain -- ``baseline``, ``faithful``, ``output-only`` -- are the
#: tranche, and a test asserts that whatever carries zero seeds is exactly the head of this
#: list. That is what stops the budget being balanced by quietly cutting something the plan
#: never nominated for cutting.
#:
#: The order, and why each one is where it is:
#:
#:   ``n8``, ``n2``            The expansion-rate curve. The only claim in the plan that
#:                             nothing else in it depends on, so it is the cheapest thing to
#:                             give up and has been given up longest.
#:   ``tied-*``                Cause 5 needs both arms or neither, so it is two runs minimum
#:                             for a question nothing downstream reads.
#:   ``decay-everything``      Cause 3. One run at one seed could never have separated the
#:                             parameter-group split from the seed it drew.
#:   ``n1``                    The seesaw control. Reconnaissance, no claim attached.
#:   ``no-output-init``        Cause 2 / H3. A real hypothesis, and the first non-mhc thing a
#:                             second tranche should want -- but it asks whether a scaling is
#:                             load-bearing, which only means something once H1 says the
#:                             method does anything at all.
#:   ``mhc``                   H5. LAST, DELIBERATELY. It is the best-designed hypothesis in
#:                             the module and the three runs it needs are specified, tested
#:                             and iso-parameter today. Being last here is the record that it
#:                             was deferred rather than dropped.
CUT_ORDER = [
    "n8",
    "n2",
    "tied-faithful",
    "tied-baseline",
    "decay-everything",
    "n1",
    "no-output-init",
    "mhc",
]

#: The arms the first tranche actually runs, derived rather than written down twice.
FUNDED = [name for name, arm in ARMS.items() if arm.seeds > 0]


#: Every ``(arm, seed)`` pair the tranche runs, in the order the fan-out hands them out.
#:
#: NINE RUNS ARE ONE SUBMISSION AND THE CELL INDEX IS WHAT TELLS THEM APART. The platform
#: fans a submission out with ``--fanout-size 9 --fanout-index-parameter arm-and-seed`` and
#: gives each cell its own ``AWS_BATCH_JOB_ARRAY_INDEX``; ``resolve_cell`` in
#: ``train_hyper_connections.py`` reads that integer and comes back here for the pair it
#: names. So the submitted command carries neither ``--arm`` nor ``--seed``, and the nine
#: cells are nine different runs of one commit rather than three commits of three.
#:
#: DERIVED FROM THE SEED COUNTS RATHER THAN WRITTEN OUT, so an arm that gains or loses a seed
#: moves the cell list and ``total_runs()`` together and cannot move only one of them. The
#: order is the arm table's own, which is the pre-registration's numbering, so cell 0 is
#: ``baseline`` seed 0 and cell 8 is ``output-only`` seed 2.
TRANCHE_CELLS: List[Tuple[str, int]] = [
    (name, seed) for name, arm in ARMS.items() for seed in range(arm.seeds)
]


def cell(index: int) -> Tuple[str, int]:
    """
    Which arm and which replicate the fan-out cell at this index is.

    :param index: ``$AWS_BATCH_JOB_ARRAY_INDEX``, contiguous from zero, which is what Batch
        requires of an array.

    :returns: ``(arm_name, seed)``.

    :raises IndexError: If the index is outside the tranche, which means the submission's
        ``--fanout-size`` and this table disagree about how many runs there are.
    """
    if not 0 <= index < len(TRANCHE_CELLS):
        raise IndexError(
            f"cell {index} of a tranche that has {len(TRANCHE_CELLS)} cells. The submission's "
            f"--fanout-size and hyper_connection_arms.TRANCHE_CELLS disagree; submit with "
            f"--fanout-size {len(TRANCHE_CELLS)}."
        )
    return TRANCHE_CELLS[index]


#: The only attention backend this platform's training image can run.
#:
#: ``olmo3_370M`` asks for flash-2, and the first rehearsal died on it in eighteen minutes:
#: "'FlashAttention2Backend' is missing the flash-attn package or is not supported on this
#: platform." The image installs torch 2.9.0 and never installs flash-attn, so this is not a
#: property of the card -- L40S is Ada and flash-2 supports it -- but of the image, and no arm
#: can use it until somebody adds it there.
PLATFORM_ATTN_BACKEND = "torch"


def _platform_shape(**kwargs) -> dict:
    """
    The two attention settings every arm needs on this platform, whatever its size.

    The sliding window goes away rather than being carried. OLMo-3's pattern is
    ``[4096, 4096, 4096, -1]`` and these runs are at sequence length 4096, so a window of 4096
    covers every position's whole history and the windowed layers are exactly full causal
    attention -- provably the same model. Keeping it would not change a single logit, but it
    would make the torch backend build an explicit mask, and SDPA with an explicit mask gives
    up the fused causal kernel it would otherwise use. So this is free to drop and not free to
    keep.
    """
    from olmo_core.nn.attention import AttentionBackendName

    kwargs.setdefault("attn_backend", AttentionBackendName(PLATFORM_ATTN_BACKEND))
    kwargs.setdefault("sliding_window", None)
    return kwargs


def hc_370M(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    The 370M OLMo-3 config the arms are defined against, with the two settings this platform
    forces. See :func:`_platform_shape` for why each one moves.

    Named rather than reached through ``--model-factory olmo3_370M`` so that the flash-2
    default cannot come back the next time somebody copies a command.
    """
    return TransformerConfig.olmo3_370M(vocab_size=vocab_size, **_platform_shape(**kwargs))


def hc_rehearsal(vocab_size: int, **kwargs) -> TransformerConfig:
    """
    The rehearsal size: roughly 19M parameters in the blocks, in the same shape as the 370M
    config so that every code path an arm touches is the path the real run will take.

    In the blocks, because there is no such thing as a 20M model on this tokenizer. dolma2 is
    100,278 tokens and OLMo-3 unties the embeddings, so the two tables alone are 77M parameters
    at d_model 384 -- four times the rest of the model, and the whole thing comes to 96M. The
    rehearsal exists to exercise the lane plumbing on a GPU, and that plumbing lives in the
    blocks.

    This spells out what ``olmo3_370M`` composes rather than calling it, because that factory
    fixes ``d_model`` and ``hidden_size_multiplier`` and there is no way to reach past them.
    """
    return TransformerConfig.llama_like(
        vocab_size=vocab_size,
        d_model=kwargs.pop("d_model", 384),
        n_layers=kwargs.pop("n_layers", 8),
        n_heads=kwargs.pop("n_heads", 6),
        hidden_size_multiplier=kwargs.pop("hidden_size_multiplier", 1.5),
        block_name=kwargs.pop("block_name", TransformerBlockType.reordered_norm),
        qk_norm=kwargs.pop("qk_norm", True),
        rope_theta=kwargs.pop("rope_theta", 500_000),
        layer_norm_eps=kwargs.pop("layer_norm_eps", 1e-6),
        **_platform_shape(**kwargs),
    )


def install() -> None:
    """
    Put the arm factories on :class:`TransformerConfig`, so that the platform's
    ``--model-factory`` reaches them through the stock attribute lookup:
    ``getattr(TransformerConfig, opts.model_factory)``.
    """
    for name, factory in (("hc_rehearsal", hc_rehearsal), ("hc_370M", hc_370M)):
        if not hasattr(TransformerConfig, name):
            setattr(TransformerConfig, name, staticmethod(factory))


#: Shards to carve out of training when a corpus declares no validation split of its own.
#:
#: This is the fallback and not the path regmix-10b takes. That corpus declares seven val
#: shards, one per source, which is strictly better than carving: no training tokens are lost,
#: the split is the publisher's rather than ours, and it is stratified by source for free.
#: A carve is what is left for a corpus with nothing declared, and it is worse in a way worth
#: naming -- shard paths sort by source, so taking the last two would draw the entire
#: evaluation set from one source category and measure BPB on that one alone.
HELD_OUT_SHARDS = 2

#: Bytes per token for dolma2 on English web text.
#:
#: Bits-per-byte is ``CE_nats / (bytes_per_token * ln 2)``. This constant sets the absolute
#: level only: it is identical for every arm, so a difference in BPB between two arms is a
#: difference in held-out CE scaled by a fixed factor, and the comparison is exact whatever
#: this is. Measure it off the corpus and correct it if the absolute number ever has to mean
#: something on its own.
DOLMA2_BYTES_PER_TOKEN = 4.57


def apply_partial_rotary(config: TransformerConfig, factor: float) -> TransformerConfig:
    """
    Set the fraction of each head's channels that receive RoPE.

    A separate axis from the arms and composable with all of them. It costs nothing -- RoPE has
    no parameters and ``num_flops_per_token`` does not count it -- so the only thing that moves
    is which channels carry positional phase. The channels above the cut pass through
    unrotated rather than being dropped, and QK-norm still runs on the whole head beforehand.

    :param config: The model config. Mutated in place and returned.
    :param factor: In ``[0, 1]``. ``1.0`` is ordinary RoPE; ``0.0`` is NoPE.

    :raises ValueError: If the factor is out of range, if the model has no rotary embedding to
        set it on, if the RoPE implementation refuses it, or if it would cut a head into an odd
        number of rotated channels -- RoPE rotates channels in pairs, so an odd count silently
        leaves one of them out of the rotation it was supposed to be in.
    """
    from olmo_core.nn.rope import RoPEType

    if not 0.0 <= factor <= 1.0:
        raise ValueError(f"partial_rotary_factor must be in [0, 1], got {factor}")
    if isinstance(config.block, dict):
        raise ValueError("partial RoPE expects a single block config, not a named-block dict.")

    rope = getattr(config.block.sequence_mixer, "rope", None)
    if rope is None:
        raise ValueError("this model has no rotary embedding to set a partial factor on")
    if rope.name == RoPEType.fused:
        raise ValueError(
            "FusedRotaryEmbedding refuses a partial factor, and it refuses it at build time. "
            "Use RoPEType.default."
        )

    head_dim = config.block.sequence_mixer.head_dim or (
        config.d_model // config.block.sequence_mixer.n_heads
    )
    rotated = int(head_dim * factor)
    if rotated % 2 != 0:
        raise ValueError(
            f"factor {factor} rotates {rotated} of {head_dim} channels, which is odd. RoPE "
            "pairs channels, so an odd count drops one out of the rotation."
        )

    rope.partial_rotary_factor = factor
    return config


def source_label(path: str) -> str:
    """
    The corpus source a shard belongs to, which the LM evaluator needs as its metric label.

    Shards live at ``.../tokens/<source>/<split>-000NN.u32le.bin`` and the manifest carries the
    same value as ``labels.source`` on each entry, so the directory is the label rather than a
    guess about one. Using it means the run reports bits-per-byte per source instead of one
    pooled number -- and a pooled number over seven very different distributions is the kind of
    average that hides the effect it is supposed to measure.

    :param path: A shard URI or path.

    :returns: The source name, or ``"held_out"`` if the path has no source directory.
    """
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    return parts[-2] if len(parts) >= 2 else "held_out"


def split_held_out(paths: List[str], n_held_out: int = HELD_OUT_SHARDS) -> Tuple[list, list]:
    """
    Split shard paths into training and evaluation sets.

    Sorted first and taken from the end, so that every arm and every seed holds out the same
    shards. A held-out set that moved between arms would put the arms on different eval data,
    which is the one thing that would make the comparison meaningless.

    :param paths: All shard paths from the corpus manifest.
    :param n_held_out: How many to reserve.

    :returns: ``(train_paths, eval_paths)``.

    :raises ValueError: If the corpus has too few shards to hold any out.
    """
    ordered = sorted(paths)
    if n_held_out < 1 or n_held_out >= len(ordered):
        raise ValueError(
            f"cannot hold out {n_held_out} of {len(ordered)} shards and still have a "
            "training set"
        )
    return ordered[:-n_held_out], ordered[-n_held_out:]


@dataclass
class BitsPerByteCallback(Callback):
    """
    Add a bits-per-byte reading beside every cross-entropy metric the run records.

    BPB is the house metric and the pre-registration is written against it, but OLMo-core's
    LM evaluator reports CE loss and perplexity. BPB is a fixed rescaling of CE, so rather
    than convert by hand afterwards -- once per arm, per seed, at the point where a factor is
    easiest to drop -- this writes it next to the loss it came from.

    Runs in ``pre_log_metrics`` so the derived value travels with its source to every logging
    backend rather than only to W&B.
    """

    bytes_per_token: float = DOLMA2_BYTES_PER_TOKEN
    enabled: bool = True

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        del step
        if not self.enabled:
            return
        scale = self.bytes_per_token * math.log(2)
        for name in [k for k in metrics if k.endswith("CE loss")]:
            value = metrics[name]
            if value is not None and math.isfinite(value):
                metrics[name.removesuffix("CE loss") + "BPB"] = value / scale


# ---------------------------------------------------------------------------------------
# What the tranche costs. `edullm check` reads none of this: it prices a ceiling from the
# workload profile and knows nothing about steps, so the arm table is the only place the
# experiment's own arithmetic is written down.
# ---------------------------------------------------------------------------------------

#: Seconds per optimizer step at the 370M shape on gpu-4xl40s, rank microbatch 16,384.
#:
#: MEASURED, AND NOT THE NUMBER THE BUDGET WAS FIRST WRITTEN AGAINST. Re-derived from W&B run
#: ``run_019fe1f6-8692-70d3-a6cf-97c4756624e3`` (entity eduLLM, project pre-training, group
#: hyper-connections-370m): 93 clean steps -- every step of the 100 except the two that ran
#: held-out evaluation and the four the lane monitor fired on -- give a median of 10.32 s and
#: an interquartile spread under 0.05 s. The flush-group wall clock agrees at 10.40 s.
#:
#: The 11.69 s/step this tranche was first priced at is the same run read through a filter
#: that kept only the five rows carrying ``hc/*`` keys, which are steps 20, 40, 60, 80 and
#: 100 -- exactly the monitor's firing steps at ``--monitor-interval 20``. A monitor step
#: costs about 1.37 s more than an ordinary one, so that median is the cost of the
#: instrument, sampled five times, and not the cost of a step. The superseded 8,192-microbatch
#: probe's headline 15.55 s/step is its *clean* median, so the two probes were never compared
#: the same way: the real improvement is 15.54 -> 10.32, not 15.55 -> 11.69.
MEASURED_SECONDS_PER_STEP = 10.32

#: What one lane-monitor firing adds to the step it fires on. Median of steps 20, 40, 60 and
#: 80 (11.69 s) against the clean median. Only the hyper-connection arms pay it.
MONITOR_SECONDS_PER_FIRING = 1.37

#: One held-out evaluation over the seven declared validation shards. Measured 103.58 s and
#: 103.70 s in the steady state; the first one of a run took 124.9 s warming up.
MEASURED_EVAL_SECONDS = 104.0

#: One permanent checkpoint. Measured 47.94 s and 45.71 s on the same run, and again on the
#: superseded probe, so it is a property of the model and the bucket rather than of the step.
#:
#: CHARGED ON EVERY CHECKPOINT EVEN THOUGH MOST OF THEM ARE FREE, DELIBERATELY. The
#: checkpointer runs with ``save_async=True``, and at a 500-step interval -- 86 minutes -- the
#: previous write has long finished, so the ``_await_last_checkpoint`` at the top of the next
#: one costs nothing and only the final synchronous save in ``post_train`` really blocks.
#: Counting all thirteen anyway pads the estimate by about nine minutes. The asymmetry is
#: worth paying for: over-estimating the runtime costs an hour of unused ceiling and
#: under-estimating it costs the arm.
MEASURED_CHECKPOINT_SECONDS = 46.0

#: Container start to the first optimizer step, plus the shutdown after the last one, with the
#: start-up evaluation and the step-zero checkpoint taken out because they are counted
#: separately below.
#:
#: Derived rather than read off a single field. On the probe the 99 reported step times sum to
#: 1,240.7 s and the three evaluations to 332.2 s, against 1,513.5 s of wall clock; the two
#: evaluations inside the loop are already inside the first figure, so what is left over for
#: start-up, the unmeasured first step, the final blocking checkpoint and the shutdown is
#: 164.5 s. Take one checkpoint and one step out of that and the rest is this.
MEASURED_STARTUP_SECONDS = 118.0

#: The horizon the experiment wants: 12,715 steps x 786,432 tokens is 10.0B dolma2 tokens.
FULL_HORIZON_STEPS = 12_715

#: The horizon this tranche runs, and it is shorter than the one above for a reason that has
#: nothing to do with money.
#:
#: A full arm is 12,715 steps x 10.32 s plus its evaluations and checkpoints, which is 37.9
#: hours. ``olmo-core-train`` declares ``maximum_runtime_hours: 24`` and ``--hours`` may only
#: lower it -- the platform refuses an override above the workload bound with
#: ``runtime_above_the_workload_bound`` and says that raising it is a pull request against
#: config/workload-catalog.yaml. So a full arm cannot finish in one attempt.
#:
#: THE TWO-ATTEMPT PLAN DOES NOT SURVIVE READING THE RETRY RULES. A second attempt would
#: resume correctly -- see the resume section of hyper-connections.md, which cites the lines
#: -- but it has to be granted first, and ``RETRY_ONLY_WHAT_A_RETRY_FIXES`` in the platform's
#: execution.py is ``OnStatusReason "Host EC2*" RETRY`` then ``OnReason "OutOfMemoryError*"
#: EXIT`` then ``OnExitCode "*" EXIT``. A timeout is granted a retry only because it records
#: no container exit code, so nothing matches and Batch's documented fall-through applies. An
#: attempt that records any exit code at all falls to rule three and is not retried -- and
#: torchrun's elastic agent re-raises ``SignalException`` and exits non-zero on SIGTERM, in a
#: dead heat with the container stop timeout. That is a coin flip with a whole arm on it.
#:
#: 6,000 steps is 4.72B tokens, 17.9 hours at the measured step time, and fits one attempt of
#: 21 hours with three hours to spare. Every arm and every seed runs the same 6,000, so no
#: contrast in the tranche has a horizon confound; what is given up is the comparison to the
#: published 10B-token results, which becomes a second tranche's job along with mhc.
TRANCHE_STEPS = 6_000

#: The intervals the tranche runs at, here rather than only in run.yaml so the cost model and
#: the command cannot disagree. A test asserts the committed command matches these.
TRANCHE_SAVE_INTERVAL = 500
TRANCHE_EVAL_INTERVAL = 500
TRANCHE_MONITOR_INTERVAL = 50

#: Warmup as a fraction of the horizon rather than as a step count, so the schedule keeps its
#: shape when the horizon moves between the two staged tranches. 6,000 steps is 120 and 12,715
#: is 254.
TRANCHE_WARMUP_FRACTION = 0.02

#: The step by which the lane monitor stops warning and starts refusing.
#:
#: THE ARMS ARE THE FIRST RUNS THIS GUARD POLICES FROM THE START, which is why it is here
#: rather than left off. An arm whose lanes never differentiate is the baseline with dead
#: parameters, and no downstream number from it is interpretable in either direction -- so
#: finding out at step 400 costs about an hour and finding out at the end costs the arm. It
#: is after warmup on both horizons and inside the first checkpoint interval on both.
#:
#: IT IS INERT ON ``baseline``, WHICH IS WHAT LETS ONE COMMAND SERVE ALL NINE CELLS.
#: ``train_hyper_connections.train`` attaches the monitor only for an arm with lanes, so the
#: three baseline cells parse this flag and never build a callback that could read it.
#:
#: The guard passes on a majority of blocks and then disables itself. At 370M the probe had
#: fourteen of sixteen blocks over the floor against the half it needs, so the margin is wide;
#: the two that were under are blocks 01 and 02, and hyper-connections.md records why a
#: shallow dead zone is a finding about depth rather than a reason to abort.
TRANCHE_FAIL_CLOSED_BY_STEP = 400

#: The throughput probe, which is the other thing ``.edullm/run.yaml`` is ever allowed to be.
#:
#: A probe is not an arm and trains nothing anybody reads. It exists because the step time on
#: a shape nobody has run is the only input to :data:`FULL_HORIZON_STEPS` being reachable at
#: all, and the difference between reaching it and not is 10B tokens against 4.72B.
#:
#: These are here rather than only in run.yaml for the same reason the tranche's intervals
#: are: ``test_the_committed_command_is_a_shape_this_table_prices`` reads run.yaml and matches
#: it against one of the two sets, so a command that is neither the tranche nor the probe --
#: a half-edited file, most likely -- fails on a laptop rather than on a machine.
PROBE_STEPS = 100
PROBE_SAVE_INTERVAL = 100
PROBE_EVAL_INTERVAL = 50
PROBE_WARMUP_STEPS = 20

#: GPUs per compute profile, for the two shapes this experiment submits against.
#:
#: COPIED FROM THE PLATFORM'S ``CONTAINER_SHAPES`` RATHER THAN IMPORTED, because
#: ``edullm_platform`` is a uv tool install in its own environment and is not a dependency of
#: this repository's test run. Copied rather than derived from the profile name, because the
#: platform's own launchers.py says in as many words that the name is a convention nothing
#: enforces and that deriving a device count from it is "tempting and wrong".
#:
#: What this buys is the refusal that already cost a submission: ``process_per_device`` fires
#: when the command's ``--nproc-per-node`` is not exactly this number.
GPUS_PER_COMPUTE_PROFILE = {
    "gpu-4xl40s": 4,
    "gpu-8xa100": 8,
}

#: What one A100 step has to come in under for a full 12,715-step arm to fit 24 hours with
#: 10% of the ceiling still unspent.
#:
#: WRITTEN DOWN BEFORE THE PROBE RUNS, WHICH IS THE ONLY TIME A THRESHOLD IS WORTH ANYTHING.
#: A number chosen after the measurement is a number chosen to agree with it. Derived by
#: :func:`seconds_per_step_to_fit`; the argument for moving the tranche to A100 is that the
#: probe's clean median comes in under this, and nothing else.
A100_STEP_SECONDS_FOR_FULL_HORIZON = 5.76

#: Tokens per optimizer step: a 786,432-token global batch.
TRANCHE_TOKENS_PER_STEP = 768 * 1024


def arm_seconds(
    arm: Arm, steps: int = TRANCHE_STEPS, seconds_per_step: float = MEASURED_SECONDS_PER_STEP
) -> float:
    """
    How long one run of this arm takes, from the measurements above.

    Counts what the probe showed a run actually spends time on: the steps, one held-out
    evaluation on startup and one on finish and one every ``TRANCHE_EVAL_INTERVAL``, a
    checkpoint at step zero and one every ``TRANCHE_SAVE_INTERVAL``, the container's start-up,
    and -- for an arm that has lanes to watch -- the lane monitor's own cost.

    :param arm: The arm.
    :param steps: How many optimizer steps it runs for.
    :param seconds_per_step: The step time to price against. Defaults to the L40S measurement,
        which is the only one there is; the A100 tranche is priced against
        :data:`A100_STEP_SECONDS_FOR_FULL_HORIZON`, which is the threshold the probe has to
        clear and therefore the slowest step that would send the tranche to that shape at all.
        Everything that is not a step is left at its measured value, because the evaluations
        are the same seven shards and the checkpoints are the same model to the same bucket.

    :returns: Seconds.
    """
    evaluations = 2 + steps // TRANCHE_EVAL_INTERVAL
    checkpoints = 1 + steps // TRANCHE_SAVE_INTERVAL
    monitor = 0.0
    if arm.hyper_connections is not None:
        monitor = (steps // TRANCHE_MONITOR_INTERVAL) * MONITOR_SECONDS_PER_FIRING
    return (
        MEASURED_STARTUP_SECONDS
        + steps * seconds_per_step
        + evaluations * MEASURED_EVAL_SECONDS
        + checkpoints * MEASURED_CHECKPOINT_SECONDS
        + monitor
    )


def seconds_per_step_to_fit(
    hours: float,
    steps: int = FULL_HORIZON_STEPS,
    arm: Optional["Arm"] = None,
    margin: float = 0.10,
) -> float:
    """
    The slowest step that still lands an arm inside a runtime bound.

    Everything an arm spends that is not a step is fixed by the interval settings rather than
    by the shape -- the evaluations are the same seven shards, the checkpoints are the same
    model to the same bucket, and the lane monitor fires the same number of times -- so
    moving to a faster card moves the step term and nothing else. Subtracting the fixed part
    first is what makes the answer a threshold a measurement can be held against, rather than
    ``hours / steps``, which is out by the 1.24 hours those instruments cost at the full
    horizon.

    :param hours: The runtime bound the run will be submitted under.
    :param steps: How many optimizer steps it has to complete.
    :param arm: The arm, which decides whether the lane monitor is paid for. Defaults to
        ``faithful``, the more expensive of the two funded arms that has lanes.
    :param margin: Fraction of the bound to leave unspent.

    :returns: Seconds per step. A negative result means the fixed costs alone exceed the
        bound, so no step time is fast enough.
    """
    arm = ARMS["faithful"] if arm is None else arm
    evaluations = 2 + steps // TRANCHE_EVAL_INTERVAL
    checkpoints = 1 + steps // TRANCHE_SAVE_INTERVAL
    monitor = 0.0
    if arm.hyper_connections is not None:
        monitor = (steps // TRANCHE_MONITOR_INTERVAL) * MONITOR_SECONDS_PER_FIRING
    fixed = (
        MEASURED_STARTUP_SECONDS
        + evaluations * MEASURED_EVAL_SECONDS
        + checkpoints * MEASURED_CHECKPOINT_SECONDS
        + monitor
    )
    return (hours * 3600.0 * (1.0 - margin) - fixed) / steps


def tranche_hours(
    steps: int = TRANCHE_STEPS, seconds_per_step: float = MEASURED_SECONDS_PER_STEP
) -> float:
    """
    GPU-node hours for every funded run in the table, seeds counted.
    """
    return (
        sum(arm.seeds * arm_seconds(arm, steps, seconds_per_step) for arm in ARMS.values()) / 3600.0
    )


def estimated_cost_usd(hourly_rate_usd: float, steps: int = TRANCHE_STEPS) -> float:
    """
    What the tranche is expected to spend, at a rate the caller supplies.

    THE RATE IS AN ARGUMENT AND IS NOT WRITTEN DOWN IN THIS FILE. Prices live in reviewed
    platform configuration that changes without anybody being told, so the only honest source
    is ``edullm check --json``, which reports ``cost.hourly_rate_usd`` for the compute profile
    a submission actually names. A number copied in here would be right until it was not, and
    nothing would say when.

    This is also not what a submission is approved against. ``edullm check`` prices a
    *ceiling* -- attempts x the runtime bound x the rate, for every cell -- which is a larger
    number and the one the budget has to clear. This is the expected spend if the runs behave
    the way the probe did.

    :param hourly_rate_usd: ``cost.hourly_rate_usd`` from ``edullm check --json``.
    :param steps: How many optimizer steps each run does.

    :returns: Dollars.
    """
    return tranche_hours(steps) * hourly_rate_usd


@dataclass(frozen=True)
class StagedTranche:
    """One of the two nine-run tranche specs sitting committed next to ``run.yaml``.

    Both are staged rather than one, because which shape the tranche runs on is decided by a
    throughput probe that is still running, and the answer has to become a submission in
    minutes rather than in an editing session. Whichever way the probe reads, one ``cp`` over
    ``.edullm/run.yaml`` and one ``edullm submit`` is the whole of it.
    """

    spec: str
    """The file, relative to ``.edullm/``."""

    compute_profile: str
    """The shape. Its device count is in :data:`GPUS_PER_COMPUTE_PROFILE`."""

    steps: int
    """The horizon, which is the thing the two variants disagree about."""

    rank_microbatch_size: int
    """
    Tokens per rank per gradient-accumulation microbatch. Not portable between the two: 16,384
    is what ran on a 48 GB L40S and the same arithmetic puts it at 90% of a 40 GB A100, so the
    A100 spec is 12,288. The memory model is in the header of the commit that set it,
    ``075f52aa7``, and moving this number without re-deriving it is how an arm meets
    ``OnReason "OutOfMemoryError*" EXIT``.
    """

    hours: float
    """
    The ``--hours`` this variant is submitted under. It is a per-attempt runtime bound that
    ``--hours`` may only lower, and it is also the hours factor in the approved ceiling, so it
    is chosen against both the run's own length and the budget.
    """

    attempts: int
    """
    The ``--attempts`` this variant is submitted under, which multiplies the ceiling directly.
    """

    seconds_per_step: float
    """
    The step time this variant is priced against. Measured for the L40S; for the A100 it is
    :data:`A100_STEP_SECONDS_FOR_FULL_HORIZON`, the threshold the probe has to clear, which is
    the slowest step that would send the tranche to that shape at all and therefore the
    conservative end of what it could turn out to be.
    """

    @property
    def warmup_steps(self) -> int:
        """Warmup at :data:`TRANCHE_WARMUP_FRACTION` of the horizon."""
        return round(self.steps * TRANCHE_WARMUP_FRACTION)

    @property
    def hours_per_run(self) -> float:
        """The longest of the funded arms, which is the one the bound has to hold."""
        return (
            max(arm_seconds(ARMS[name], self.steps, self.seconds_per_step) for name in FUNDED)
            / 3600.0
        )

    @property
    def checkpoint_exposure_hours(self) -> float:
        """
        The most work a retry after a mid-run failure throws away.

        One save interval of steps, because ``Trainer.fit`` resumes from the last permanent
        checkpoint in the save folder and the fan-out prologue gives each cell a checkpoint
        prefix a retry of that cell re-derives identically.
        """
        return TRANCHE_SAVE_INTERVAL * self.seconds_per_step / 3600.0

    def maximum_compute_cost_usd(self, hourly_rate_usd: float) -> float:
        """
        The ceiling admission prices, which is the number the budget has to clear.

        ``rate x nodes x hours x attempts x cells``, from
        ``edullm_platform.contracts.workload.compute_maximum_compute_cost_usd``. Every one of
        the two shapes here is a single node.

        THE RATE IS AN ARGUMENT FOR THE REASON :func:`estimated_cost_usd` GIVES. Read
        ``cost.hourly_rate_usd`` out of ``edullm check --json`` and pass it in.
        """
        return hourly_rate_usd * self.hours * self.attempts * len(TRANCHE_CELLS)

    def expected_cost_usd(self, hourly_rate_usd: float) -> float:
        """
        What the tranche is expected to actually spend, which is a different number.

        BILLING IS WALL CLOCK AND NOT THE REQUESTED CEILING.
        ``edullm_platform.run_costs._attempt_seconds`` sums each attempt's
        ``ended_at - started_at`` and ``_cost`` multiplies that by the hourly rate, so an hour
        of ``--hours`` that a run does not use costs nothing. The ceiling above is what gets
        approved; this is what arrives on the bill.
        """
        return tranche_hours(self.steps, self.seconds_per_step) * hourly_rate_usd


#: The two staged tranches, keyed by the shape they run on.
#:
#: ONE SOURCE OF TRUTH FOR TWO FILES NOBODY WILL RE-READ UNDER TIME PRESSURE.
#: ``test_both_staged_tranches_are_the_shape_this_table_prices`` walks this dict, opens each
#: spec and checks every field against it, so a half-edited variant fails on a laptop rather
#: than at admission or, worse, eighteen hours into nine machines.
STAGED_TRANCHES: Dict[str, StagedTranche] = {
    "gpu-4xl40s": StagedTranche(
        spec="run.l40s-tranche.yaml",
        compute_profile="gpu-4xl40s",
        steps=TRANCHE_STEPS,
        rank_microbatch_size=16 * 1024,
        hours=21.0,
        attempts=2,
        seconds_per_step=MEASURED_SECONDS_PER_STEP,
    ),
    "gpu-8xa100": StagedTranche(
        spec="run.a100-tranche.yaml",
        compute_profile="gpu-8xa100",
        steps=FULL_HORIZON_STEPS,
        rank_microbatch_size=12 * 1024,
        # 24 AND ONE ATTEMPT, AND THE 24 IS NOT A DEFAULT -- IT IS THE SMALLEST BOUND THAT
        # HOLDS THIS RUN AT THE STEP TIME THAT WOULD SEND IT HERE. At
        # A100_STEP_SECONDS_FOR_FULL_HORIZON an arm is 21.6 hours, which is where that
        # threshold came from: 10% of 24 left unspent. A bound of 20 would price under the
        # budget and kill nine cells at step ~11,700 of 12,715, having paid for all of it.
        #
        # ONE ATTEMPT RATHER THAN THE PROFILE'S TWO, AND IT IS THE BUDGET THAT DECIDES IT
        # HERE RATHER THAN THE RETRY RULES. The ceiling multiplies by attempts, so two is
        # twice $4,742.84 for a second attempt that RETRY_ONLY_WHAT_A_RETRY_FIXES grants only
        # for a lost host -- see TRANCHE_STEPS for the reading. On the L40S variant the
        # second attempt fits under $4,000 and is kept; here it does not and is not.
        #
        # THIS IS THE ONE STAGED NUMBER THAT DOES NOT CLEAR THE STATED $4,000, AND IT CANNOT
        # BE MADE TO WITHOUT MOVING THE HORIZON. See seconds_per_step_within_budget: at the
        # A100 rate, nine cells of 12,715 steps come in under $4,000 of *actual* spend only
        # if the probe measures about 5.4 s/step or better, and under a $4,000 approved
        # *ceiling* only at --hours 20, which is 5.3 s/step with no margin at all. Between
        # that and the 5.76 threshold the horizon fits the machine and not the budget, and
        # that is a conversation rather than a staging decision.
        hours=24.0,
        attempts=1,
        seconds_per_step=A100_STEP_SECONDS_FOR_FULL_HORIZON,
    ),
}


def seconds_per_step_within_budget(
    budget_usd: float,
    hourly_rate_usd: float,
    steps: int = FULL_HORIZON_STEPS,
    arm: Optional["Arm"] = None,
    cells: int = 0,
) -> float:
    """
    The slowest step at which the whole tranche's expected spend still fits a budget.

    A COMPANION TO :func:`seconds_per_step_to_fit` AND A DIFFERENT QUESTION. That one asks
    what fits inside a runtime bound, which is about the machine; this asks what fits inside
    a number of dollars, which is about the rate. They can disagree, and on the A100 shape
    they do: the horizon reaches the 24-hour bound at a step time that is already over
    $4,000, because that shape is 2.1 times the L40S rate and rather less than 2.1 times as
    fast.

    Priced on wall clock rather than on the ceiling, because that is what
    ``edullm_platform.run_costs`` bills. The ceiling a submission is *approved* against is
    :meth:`StagedTranche.maximum_compute_cost_usd` and is a larger number.

    :param budget_usd: What the tranche may spend.
    :param hourly_rate_usd: ``cost.hourly_rate_usd`` from ``edullm check --json``.
    :param steps: The horizon.
    :param arm: Which arm to price, since only the ones with lanes pay the monitor. Defaults
        to ``faithful``, the more expensive of the two.
    :param cells: How many runs. Defaults to the whole tranche.

    :returns: Seconds per step. Negative means the fixed costs alone are over budget.
    """
    arm = ARMS["faithful"] if arm is None else arm
    cells = len(TRANCHE_CELLS) if cells < 1 else cells
    seconds_affordable = budget_usd * 3600.0 / (hourly_rate_usd * cells)
    fixed = arm_seconds(arm, steps, seconds_per_step=0.0)
    return (seconds_affordable - fixed) / steps


def total_runs() -> int:
    """
    How many runs the table asks for. Counted rather than written down, because the number
    written down in the plan was wrong for as long as it was written down.
    """
    return sum(arm.seeds for arm in ARMS.values())


def describe() -> str:
    """
    The table, for ``--list-arms`` and for the run log.
    """
    width = max(len(name) for name in ARMS)
    lines = [f"{'arm'.ljust(width)}  #   seeds  isolates"]
    for name, arm in ARMS.items():
        lines.append(f"{name.ljust(width)}  {arm.number:<2}  {arm.seeds:<5}  {arm.isolates}")
    lines.append(f"{'total'.ljust(width)}      {total_runs():<5}  runs, once seeds are counted")
    lines.append("")
    lines.append(
        f"tranche: {total_runs()} runs x {TRANCHE_STEPS:,} steps "
        f"({TRANCHE_STEPS * TRANCHE_TOKENS_PER_STEP / 1e9:.2f}B tokens each), "
        f"{max(arm_seconds(a) for a in ARMS.values() if a.seeds) / 3600:.1f} h per run, "
        f"{tranche_hours():.1f} node-hours in total."
    )
    lines.append(
        "Multiply those hours by cost.hourly_rate_usd from `edullm check --json` for the "
        "expected spend; the ceiling that gets approved is attempts x the runtime bound x "
        "the rate x the cell count, which check reports as cost.maximum_compute_cost_usd."
    )
    return "\n".join(lines)
