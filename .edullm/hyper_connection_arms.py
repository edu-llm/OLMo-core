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

WHAT THE FIRST TRANCHE FUNDS, AND WHY IT IS FOUR ARMS RATHER THAN SEVEN.

Twenty runs: ``baseline`` x5, ``faithful`` x5, ``output-only`` x5, ``mhc`` x5, and zero
everywhere else. That buys exactly three hypotheses, all at full power and all against the
same noise floor:

  H1   replication          arm 2 against arm 1, five seeds against five.
  H2a  implementation       arm 3 against arm 2, five seeds against five. Whether the
                            field's negative result is an artifact of a reimplementation
                            that kept the output mixing and dropped the input map.
  H5   the constraint       arm 9 against arm 2, five seeds against five. Whether pinning
                            the lane-mixing matrix's spectral radius at 1 is what rescues
                            the method. See below for why this arm was bought with the
                            money a sixth seed would have cost.

FIVE AND NOT THE THREE THAT STOOD HERE, AND THE MONEY CAME OUT OF THE HORIZON RATHER THAN
OUT OF THE BUDGET. Three arms at three seeds was priced on the assumption that seed sigma
falls as 1/sqrt(tokens), so that a longer run was worth paying for. Ai2's DataDecide
(arXiv 2504.11393) reads sigma against token count directly -- 1,050 models at 25 recipes x
14 sizes x 3 seeds with intermediate checkpoints, at sizes that bracket 370M -- and over the
3B-30B window with model-size fixed effects sigma goes as D^-0.172, bootstrap CI
[0.088, 0.306]. A run costs in proportion to D, so a fixed budget buys n = C/D runs and the
standard error of an arm mean goes as D^(+0.328): positive across the whole interval, so
horizon bought with seed money makes the experiment strictly less sensitive. Five seeds at
4.72B is an MDE of 0.018 nats against the 0.022 the original plan would have had at 10B, for
less money.

Balanced 5/5/5 rather than a baseline quintuple and larger treatments, because an unbalanced
contrast carries SE = sigma*sqrt(1/n_a + 1/n_b) and pays for the smaller arm twice.

Five against five is also comfortably past the smallest design in which sigma is estimated
from the data rather than assumed, which is what the analysis plan's gate -- two standard
errors of the contrast -- was written against. It takes the pooled variance estimate to
df = 12 rather than the df = 6 a triple design gave, and the baseline's own to df = 4 rather
than df = 2.

THE TWENTY RUN AS FOUR SUBMISSIONS AND NOT ONE, AND THAT IS A PRE-REGISTRATION CONSTRAINT
RATHER THAN A PLATFORM ONE. The analysis plan forbids submitting a treatment arm before the
noise-floor table has numbers in it, and the per-source inverse-variance weights have to be
frozen from the baseline alone or they are a researcher degree of freedom. So ``baseline``
went first on its own (stage 1), ``faithful`` and ``output-only`` follow as stage 2, and
``mhc`` as stage 3. Each stage is a five-cell ``--fanout-index-parameter seed`` fan-out with
its arm in the command; :data:`STAGE_SPECS` is the four specs and what has to be identical
across them.

``mhc`` / H5 IS FUNDED, AND IT WAS BOUGHT WITH MONEY A SIXTH SEED WOULD HAVE COST. It was
last in :data:`CUT_ORDER` -- the last thing cut and the first thing a restoration buys -- and
a budget grant above the original $4,000 restored it rather than widening the arms already
running. The arithmetic is not close. A sixth seed on three arms buys about 9% off the
standard error of each contrast and no new hypothesis; five seeds of this arm buy a third
hypothesis at the same power as the other two.

H5 IS ALSO THE ONLY ARM IN THE MODULE WHOSE NULL IS READABLE, which is what makes it worth
more than precision on the arms that have one. mHC's claim is mechanical -- the Sinkhorn
projection pins the lane-mixing matrix's spectral radius at 1 by construction -- and the
monitor already records that radius per layer. So a null here separates "the constraint was
inert" from "the constraint held and the effect is small", because the instrument says which
happened. Every other arm's null is one number that could be either. It ships in DeepSeek V4,
so both outcomes are worth writing down.

``output-only`` IS ONLY WORTH RUNNING BECAUSE OF COMMIT ``b7983ea9``. Before that commit the
arm replaced both the learned input map and the paper's fixed staggered one-hot read, which
left every lane reading the same vector: degenerate rather than crippled, an exact re-run of
the baseline with dead parameters, and a null result that meant nothing. b7983ea9 put the
staggered read back so that only the learned input map goes, which is the one difference H2a
is about. Five runs of a degenerate arm is the shape of mistake this note exists to prevent.
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

    Five where a five-versus-five difference is claimed against the baseline, and zero
    everywhere else. There is no longer a one and no longer a two: an arm at one seed cannot
    separate its effect from the seed it drew, and an arm at two estimates sigma from a single
    difference. The budget buys two hypotheses at full power rather than six at partial, and
    :data:`CUT_ORDER` records which order the rest come back in.

    Five and not three because the money came out of the token horizon, which measured seed
    sigma says is the worse thing to spend it on -- see the module docstring. Changing this
    number changes :data:`TRANCHE_CELLS`, :func:`total_runs`, :func:`tranche_hours` and every
    price derived from them, which is the point of it being one number.

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
        seeds=5,
    ),
    "faithful": Arm(
        number=2,
        summary="DHC x4 as published: input-side pre-mapping, sqrt(n) output init, "
        "weight-decay split.",
        isolates="The actual method.",
        seeds=5,
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
        seeds=5,
        hyper_connections=_hc(mode=HyperConnectionMode.output),
    ),
    "no-output-init": Arm(
        number=4,
        summary="The hyper-connection the paper's equivalence claim describes: at init this "
        "model IS the baseline, because the sqrt(n) output scaling is off.",
        isolates="Cause 2, and the confound in H1. `faithful` differs from `baseline` in TWO "
        "things -- the mechanism and the initialization prescription -- so H1 alone cannot say "
        "which of them moved the endpoint, and at an unpaired MDE of 0.0039 nats the design is "
        "precise enough to return a significant H1 attributable to either. FUNDED AS STAGE 4, "
        "restored from the end of the cut order, for that reason and not for cause 2 on its "
        "own: the scaling was always behind a flag so that this arm could turn it off, and an "
        "H1 that cannot be decomposed is the result the flag existed to prevent. "
        "IT IS NOT `faithful` MINUS A FLAG, AND CALLING IT AN ABLATION GETS THE PAPER BACKWARDS. "
        "The paper says a hyper-connection network is equivalent to a standard residual network "
        "at initialization; its Implementation paragraph says to scale the output modules by "
        "n**-0.5. Those are different models and only one of them is the equivalence. Measured "
        "at n=4: with the scaling OFF the pre-unembedding hidden is exactly 4.0000x the "
        "baseline's and the logits are the baseline's to about 1e-06 relative, because a "
        "scale-invariant RMSNorm sits between the lane sum and the unembedding, so the "
        "magnitude the correction exists to fix has no effect on the function. With the "
        "scaling ON the logits are ~0.7 relative away and block 1 reads well under three "
        "quarters of the baseline's residual, because a per-block n**-0.5 reweights depth "
        "against depth and is not a global rescale any norm absorbs. So THIS arm satisfies the "
        "paper's equivalence claim and `faithful` satisfies its Implementation paragraph; both "
        "are faithful to a self-contradicting paper and the contradiction is the experiment. "
        "`hyper_connection_test.py` asserts every number in this paragraph.",
        seeds=5,
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
        "V4. FUNDED AS STAGE 3, restored from the end of CUT_ORDER by a budget grant above "
        "the original $4,000, and worth more than a sixth seed on the arms already running "
        "because its null is readable: the monitor measures the radius the mechanism pins, so "
        "an absent effect can be told apart from an absent mechanism.",
        seeds=5,
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
#: Seven of the eleven arms are now unfunded, and this list is all seven of them followed by
#: nothing: the four that remain -- ``baseline``, ``faithful``, ``output-only``, ``mhc`` -- are
#: the tranche, and a test asserts that whatever carries zero seeds is exactly the head of this
#: list. That is what stops the budget being balanced by quietly cutting something the plan
#: never nominated for cutting.
#:
#: ``mhc`` USED TO BE THE LAST ENTRY AND HAS LEFT THIS LIST BY BEING FUNDED, which is the
#: mechanism this ordering was written for working exactly once. It was placed last so that a
#: restoration would reach it first, a budget grant above the original $4,000 arrived, and it
#: was the first thing bought -- ahead of a sixth seed on the arms already running, which the
#: module docstring gives the arithmetic for. What the list now records is that everything
#: still in it was cut in an order fixed in advance, and that the one restoration so far went
#: to the arm the order nominated.
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
#:   ``n1``                    The seesaw control. Reconnaissance, no claim attached. LAST,
#:                             and therefore the next thing a further grant buys.
#:
#: ``no-output-init`` HAS ALSO LEFT THIS LIST BY BEING FUNDED, which is the second time the
#: ordering has worked and the first time the reason was not money. It sat last because it asks
#: whether a scaling is load-bearing, "which only means something once H1 says the method does
#: anything at all" -- and that reasoning had the dependency backwards. ``faithful`` carries the
#: mechanism *and* the scaling, so H1 is a joint test of the two whatever it returns, and the
#: arm that separates them is not a follow-up to H1 but a precondition for reading it. It was
#: restored on 2026-08-10, before any treatment endpoint was visible.
CUT_ORDER = [
    "n8",
    "n2",
    "tied-faithful",
    "tied-baseline",
    "decay-everything",
    "n1",
]

#: The arms the first tranche actually runs, derived rather than written down twice.
FUNDED = [name for name, arm in ARMS.items() if arm.seeds > 0]


#: Every ``(arm, seed)`` pair the tranche runs, in the order the fan-out hands them out.
#:
#: THE WHOLE TRANCHE AS ONE SUBMISSION, WITH THE CELL INDEX AS THE ONLY THING THAT TELLS THE
#: RUNS APART. The platform fans a submission out with
#: ``--fanout-size <len(TRANCHE_CELLS)> --fanout-index-parameter arm-and-seed`` and gives each
#: cell its own ``AWS_BATCH_JOB_ARRAY_INDEX``; ``resolve_cell`` in
#: ``train_hyper_connections.py`` reads that integer and comes back here for the pair it
#: names. So the submitted command carries neither ``--arm`` nor ``--seed``, and the cells are
#: that many different runs of one commit rather than one commit submitted once per arm.
#:
#: THIS IS NOT THE PATH THE FIRST TRANCHE ACTUALLY TOOK, AND IT IS KEPT ON PURPOSE. The
#: pre-registration forbids submitting a treatment arm before the noise floor is measured, so
#: the twenty went out as four five-cell ``--fanout-index-parameter seed`` submissions
#: (:data:`STAGE_SPECS`) with the arm written into each command. An unstaged tranche -- the
#: next module, or a re-run of this one once H1 has an answer -- wants this table and the one
#: submission it buys instead. What it costs to keep is a table and its tests; what it would
#: cost to reconstruct after the fact is the reasoning above.
#:
#: DERIVED FROM THE SEED COUNTS RATHER THAN WRITTEN OUT, so an arm that gains or loses a seed
#: moves the cell list and ``total_runs()`` together and cannot move only one of them. That is
#: what happened twice: the design went from three seeds to five and the cell list went from
#: nine entries to fifteen, then ``mhc`` was funded and it went to twenty, and nothing here was
#: edited either time. The order is the arm table's own, which is the pre-registration's
#: numbering, so cell 0 is ``baseline`` seed 0 and the last cell is ``mhc`` at its highest
#: seed.
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
#: IT IS INERT ON ``baseline``, WHICH IS WHAT LETS ONE COMMAND SERVE ALL FIVE OF ITS CELLS.
#: ``train_hyper_connections.train`` attaches the monitor only for an arm with lanes, so the
#: five baseline cells parse this flag and never build a callback that could read it.
#:
#: The guard passes on a majority of blocks and then disables itself. At 370M the probe had
#: fourteen of sixteen blocks over the floor against the half it needs, so the margin is wide;
#: the two that were under are blocks 01 and 02, and hyper-connections.md records why a
#: shallow dead zone is a finding about depth rather than a reason to abort.
#:
#: IT IS OMITTED ON ``mhc``, AND THE REASON ON RECORD FOR THAT DOES NOT SURVIVE THE ARM'S OWN
#: CELLS. See :data:`MHC_LANE_DISPERSION_AT_GATE`, which was read at the rehearsal size on CPU
#: and predicted an abort here, and :data:`MHC_DIFFERENTIATED_FRACTION_AT_GATE_370M`, which is
#: what the five 370M cells of ``run_019fe7bc-73a6`` actually read at this step. The
#: contraction is real and it is visible for about a hundred steps; by 400 the arm clears the
#: floor on every block of every cell. The omission stands for this tranche because the specs
#: are submitted, and it costs nothing either way -- a guard that would pass and a guard that
#: is absent train the same model -- but it is not buying the protection the other two arms
#: have, and the next tranche should set it here too.
TRANCHE_FAIL_CLOSED_BY_STEP = 400

#: What the guard above would read on the ``mhc`` arm, and therefore why that arm's stage does
#: not set it. Measured at the rehearsal size on CPU, eight blocks, 400 AdamW steps at the
#: tranche's learning rate and weight decay, dispersion computed the way
#: ``HyperConnectionMonitorCallback._activation_hook`` computes it.
#:
#:   step 50    ``mhc`` 1.1e-03 .. 3.7e-03   ``faithful`` 6.2e-03 .. 3.4e-02
#:   step 200   ``mhc`` 7.9e-04 .. 5.2e-03   ``faithful`` 1.4e-02 .. 8.5e-02
#:   step 400   ``mhc`` 5.2e-04 .. 4.1e-03   ``faithful`` 2.1e-02 .. 1.1e-01
#:
#: Against a floor of 5e-03 on half the blocks that is 0 of 8 for ``mhc`` at every step
#: measured and 8 of 8 for ``faithful``. THE TREND IS THE PART THAT SETTLES IT: ``faithful``
#: rises by a factor of five over those steps and ``mhc`` falls, so this is not a slow start
#: that another 400 steps would resolve.
#:
#: WHY IT HAPPENS, WHICH IS THE REASON THIS IS NOT A BUG IN THE ARM. Sinkhorn-Knopp normalizes
#: the mixing matrix towards row and column sums of 1, and a nonnegative matrix with unit sums
#: is close to an averaging operator: applying it repeatedly pulls the lanes together. Lane
#: dispersion measures how far the lanes sit from their own mean. So the arm that constrains
#: the mixing reads lower on this statistic *because* the constraint binds, and an arm that
#: reads 4e-03 here is not an arm whose lanes are one vector -- ``faithful`` at 2e-02 and
#: ``mhc`` at 4e-03 are both mixing, one of them under a constraint.
#:
#: The floor's own docstring says it was set from the rehearsal, which is the ``faithful``
#: mechanism, and calls the quantity bimodal with nothing in the middle. It is bimodal for
#: unconstrained mixing. This arm lands in the middle of that empty band, which is what a
#: threshold calibrated on one mechanism does when it meets another.
#:
#: WHAT IS GIVEN UP BY OMITTING IT: on this arm alone, a genuinely dead run is billed for
#: eighteen hours instead of one. That is the cheaper mistake by a wide margin -- the guard
#: would cost all five cells with certainty, and what replaces it is the radius the monitor
#: records at the same interval, which on this arm is pinned at 1 by construction and is the
#: quantity H5 is actually about.
#:
#: THE ARM HAS NOW RUN AND THIS NUMBER DID NOT PREDICT IT. Everything above is a reading from
#: four blocks of d_model 64 on a CPU, and it does not carry to the model that was submitted.
#: :data:`MHC_LANE_DISPERSION_AT_GATE_370M` is the same statistic off the arm itself. Keep this
#: constant: it is a true record of what the rehearsal said, and it is why the flag was left
#: out. Do not read it as a statement about the tranche.
MHC_LANE_DISPERSION_AT_GATE = 4.1e-3

#: The lowest per-block lane dispersion any ``mhc`` cell read at the gate step, on the arm
#: itself: 370M, sixteen blocks, the tranche's own optimizer, batch and corpus. Taken from the
#: five cells of ``run_019fe7bc-73a6``, whose ``hc/`` history reaches step 4,584.
#:
#: It is 2.7 times the 5e-03 floor rather than under it, and the median block sits at 4.2e-02
#: to 4.9e-02, which is nine times the floor and within a factor of two of ``faithful``.
MHC_LANE_DISPERSION_AT_GATE_370M = 1.34e-2

#: What ``hc/differentiated block fraction`` reads on ``mhc`` at
#: :data:`TRANCHE_FAIL_CLOSED_BY_STEP`, which is the quantity the guard actually refuses on.
#:
#: ONE, ON ALL FIVE CELLS. The guard needs half. So it would have passed and disabled itself,
#: and the sentence this table carried for two days -- that setting the flag here "would abort
#: all five cells at step 400" -- is false on the arm it is about. It was true of the rehearsal
#: and the rehearsal is four blocks wide.
MHC_DIFFERENTIATED_FRACTION_AT_GATE_370M = 1.0

#: The step by which every ``mhc`` cell has crossed the floor on a majority of its blocks.
#:
#: THE CONTRACTION IS REAL AND IT IS EARLY, WHICH IS HOW A RIGHT OBSERVATION BECAME A WRONG
#: PREDICTION. At step 50 the five cells read a differentiated fraction of exactly 0 and a
#: median dispersion of 2.4e-03, well under the floor -- Sinkhorn pulling the lanes together,
#: exactly as the mechanism says. At step 100 it is 0 on four cells and 0.31 on the fifth. At
#: step 150 it is 1.0 on all five and the median has reached 1.0e-02, and it keeps rising to
#: 4.5e-02 by step 400. So the rehearsal saw the first hundred steps correctly and extrapolated
#: a fall that the real arm does not have: at 370M ``mhc`` rises like ``faithful`` does, from
#: further back and to a lower plateau.
#:
#: A gate anywhere below about 150 would have taken all five cells. The tranche's gate is 400.
MHC_LANE_DISPERSION_CROSSES_FLOOR_BY_STEP = 150

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
    arm: Arm,
    steps: int = TRANCHE_STEPS,
    seconds_per_step: float = MEASURED_SECONDS_PER_STEP,
    *,
    eval_seconds: float = MEASURED_EVAL_SECONDS,
    checkpoint_seconds: float = MEASURED_CHECKPOINT_SECONDS,
    startup_seconds: float = MEASURED_STARTUP_SECONDS,
) -> float:
    """
    How long one run of this arm takes, from the measurements above.

    Counts what the probe showed a run actually spends time on: the steps, one held-out
    evaluation on startup and one on finish and one every ``TRANCHE_EVAL_INTERVAL``, a
    checkpoint at step zero and one every ``TRANCHE_SAVE_INTERVAL``, the container's start-up,
    and -- for an arm that has lanes to watch -- the lane monitor's own cost.

    EVERY FIXED COST IS AN ARGUMENT NOW, BECAUSE THE ARGUMENT FOR HOLDING THEM FIXED TURNED OUT
    TO BE ABOUT THE WORK AND NOT ABOUT THE TIME. What this docstring used to say -- that the
    evaluations are the same seven shards and the checkpoints the same model to the same
    bucket, so only the step term moves with the shape -- is true of what the machine does and
    false of how long it takes. The A100 cells run the identical evaluation in about a quarter
    of the L40S's 104 seconds. The defaults are still the L40S measurements, so nothing that
    called this before it grew keywords has changed its answer.

    :param arm: The arm.
    :param steps: How many optimizer steps it runs for.
    :param seconds_per_step: The step time to price against. Defaults to the L40S measurement;
        the A100 tranche was staged against :data:`A100_STEP_SECONDS_FOR_FULL_HORIZON`, the
        threshold the probe had to clear and therefore the slowest step that would send the
        tranche to that shape at all.
    :param eval_seconds: One held-out evaluation over the seven declared shards.
    :param checkpoint_seconds: One permanent checkpoint.
    :param startup_seconds: Container start to the first optimizer step, plus shutdown.

    :returns: Seconds.
    """
    evaluations = 2 + steps // TRANCHE_EVAL_INTERVAL
    checkpoints = 1 + steps // TRANCHE_SAVE_INTERVAL
    monitor = 0.0
    if arm.hyper_connections is not None:
        monitor = (steps // TRANCHE_MONITOR_INTERVAL) * MONITOR_SECONDS_PER_FIRING
    return (
        startup_seconds
        + steps * seconds_per_step
        + evaluations * eval_seconds
        + checkpoints * checkpoint_seconds
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
    """One of the two whole-tranche specs sitting committed next to ``run.yaml``.

    Both were staged rather than one, because which shape the tranche ran on was to be decided
    by a throughput probe, and the answer had to become a submission in minutes rather than in
    an editing session.

    NEITHER IS SUBMITTABLE AS IT STANDS AND BOTH ARE KEPT ANYWAY. The tranche went out as the
    four five-cell stages in :data:`STAGE_SPECS` instead, for the pre-registration reason
    :class:`StageSpec` gives, and both of these files still say nine cells -- which is what the
    arm table said when they were written and no longer is. What survives them is the shape:
    an unstaged tranche is one submission, one approval and one commit for every cell, and the
    ``arm-and-seed`` fan-out these were written against is how that is expressed. The next
    module, or a re-run of this one once H1 has an answer, wants that and not this staging.
    Their headers carry what supersedes them.
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


#: The two whole-tranche specs, keyed by the shape they run on. Superseded by
#: :data:`STAGE_SPECS`, which is what actually ran; see :class:`StagedTranche` for why they
#: are kept rather than deleted.
#:
#: ONE SOURCE OF TRUTH FOR TWO FILES NOBODY WILL RE-READ UNDER TIME PRESSURE.
#: ``test_both_staged_tranches_are_the_shape_this_table_prices`` walks this dict, opens each
#: spec and checks every field against it, so a half-edited variant fails on a laptop rather
#: than at admission or, worse, eighteen hours into a room full of machines.
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


#: How many cells one stage's fan-out has, which is one funded arm's seed count.
#:
#: Derived rather than written down, because the whole reason the stages exist is that the
#: three arms have to be compared at the same replicate count, and a stage whose fan-out size
#: disagreed with the arm table would run a different design than the one that was priced.
STAGE_CELLS = ARMS["baseline"].seeds

#: The ``--hours`` every stage is submitted under, and it is the same number for all four.
#:
#: 19 covers a 17.8-hour cell -- 6,000 steps at the measured 10.32 s/step is 17.2 hours, and
#: the twelve evaluations, thirteen checkpoints, 120 monitor firings and the container's
#: start-up are 0.58 more -- with about 6% of the bound unspent. It is not the profile's 24
#: because ``--hours`` is also the hours factor of the approved ceiling, where it is
#: multiplied by the attempts and by every cell.
#:
#: IT HAS TO BE THE SAME ACROSS THE STAGES FOR A REASON THAT IS NOT ARITHMETIC. A bound is
#: what kills a cell that runs long, so a stage submitted under a looser bound than another
#: survives drift that the other one dies of, and a treatment arm missing a cell is not
#: missing it at random -- it is missing the slowest one.
STAGE_HOURS = 19.0

#: The ``--attempts`` every stage is submitted under.
#:
#: Two, which is the profile's own, because the second attempt is for a lost host: the
#: platform's retry table is ``OnStatusReason "Host EC2*" RETRY`` and then two EXITs. It
#: multiplies straight into the ceiling, so it is worth the doubling only because a cell lost
#: at hour sixteen would otherwise take its seed out of the arm mean entirely.
STAGE_ATTEMPTS = 2


@dataclass(frozen=True)
class StageSpec:
    """One of the four submissions the twenty-run tranche actually went out as.

    THE TRANCHE IS TWENTY CELLS AND WAS SUBMITTED AS FOUR FIVE-CELL FAN-OUTS, and the split
    is a pre-registration constraint rather than a platform one. The analysis plan forbids
    submitting a treatment arm before the noise-floor table has numbers in it, because two of
    the things the treatments are analysed against -- sigma-hat and the per-source
    inverse-variance weights -- cannot be estimated from a treatment arm without circularity.
    So ``baseline`` went first and alone, and the treatments follow once those are frozen.

    WHAT THE SPLIT COSTS, AND WHAT THIS CLASS IS FOR. In one twenty-cell submission every
    cell is the same command at the same commit, and nothing can drift because there is
    nothing to drift between. Four submissions at up to four commits is four chances for a
    default to move underneath the contrast, with nothing reporting it: the loss curves of a
    baseline trained at one weight decay and a treatment trained at another look exactly like
    the loss curves of a baseline and a treatment. ``STAGE_SPECS`` and the test that walks it
    are what replaces the guarantee the single submission had for free.

    Every stage is a ``--fanout-index-parameter seed`` fan-out with its arm in the command,
    including the treatments, and NOT the ``arm-and-seed`` path :data:`TRANCHE_CELLS` serves.
    Both resolve the same seeds, but stage 1 has already run through ``resolve_seed``, and
    twenty cells assigned their replicate by one mechanism is worth more in an experiment
    whose output is a noise floor than the approvals it saves are.
    """

    spec: str
    """The file, relative to ``.edullm/``."""

    arm: str
    """Which arm every cell of this stage runs. It is in the command, not in the index."""

    stage: int
    """
    Which submission this is, in the order the pre-registration allows them to go out in.

    1 for the noise floor, 2 for the two treatments H1 and H2a rest on, 3 for ``mhc``. Only
    the 1-before-everything-else ordering is a constraint: sigma-hat and the inverse-variance
    weights come from the baseline alone, so no treatment may precede it. 3 is a separate
    number from 2 because ``mhc`` was funded later, by a grant rather than by the original
    budget, and a stage number that records when a submission was decided is worth more here
    than one that records only that it is not the baseline.
    """

    run_id: Optional[str] = None
    """The platform's run id, once this stage has been submitted. ``None`` until then."""

    @property
    def cells(self) -> int:
        """The fan-out size, which is this arm's seed count."""
        return ARMS[self.arm].seeds

    @property
    def hours_per_cell(self) -> float:
        """How long one cell of this stage runs, at the measured step time."""
        return arm_seconds(ARMS[self.arm], TRANCHE_STEPS) / 3600.0

    def maximum_compute_cost_usd(self, hourly_rate_usd: float) -> float:
        """
        The ceiling this stage is approved against, which is the number the budget clears.

        ``rate x nodes x hours x attempts x cells``, from
        ``edullm_platform.contracts.workload.compute_maximum_compute_cost_usd``, on a single
        node. THE RATE IS AN ARGUMENT for the reason :func:`estimated_cost_usd` gives.
        """
        return hourly_rate_usd * STAGE_HOURS * STAGE_ATTEMPTS * self.cells

    def expected_cost_usd(self, hourly_rate_usd: float) -> float:
        """
        What this stage is expected to actually spend, which is a much smaller number.

        Billing is wall clock: ``edullm_platform.run_costs._attempt_seconds`` sums each
        attempt's ``ended_at - started_at``, so an hour of ``--hours`` a cell does not use
        costs nothing, and the second attempt costs nothing unless a host is lost.
        """
        return self.cells * self.hours_per_cell * hourly_rate_usd


#: The four stage submissions, keyed by the arm each one runs.
#:
#: ONE SOURCE OF TRUTH FOR FOUR FILES THAT MUST AGREE ON EVERYTHING BUT ONE WORD.
#: ``test_the_stage_specs_differ_in_the_arm_and_in_nothing_else`` parses all four commands
#: through the real parser and compares every resolved option, so a hand-edit to one arm's
#: command fails on a laptop. Nothing downstream would catch it: ``edullm check`` prices a
#: ceiling out of the workload profile and never reads the command's hyperparameters, and two
#: arms trained at different settings produce loss curves that look like two arms.
STAGE_SPECS: Dict[str, StageSpec] = {
    "baseline": StageSpec(
        spec="run.baseline-stage.yaml",
        arm="baseline",
        stage=1,
        # ADMITTED at commit 38b665919 under --hours 19 --attempts 2 --fanout-size 5. The
        # file is history now and is not edited again: the values its command left to parser
        # defaults are recorded in its header, and the two stage-2 specs pin them explicitly.
        run_id="run_019fe279-4ef0",
    ),
    "faithful": StageSpec(spec="run.faithful-stage.yaml", arm="faithful", stage=2),
    "output-only": StageSpec(spec="run.output-only-stage.yaml", arm="output-only", stage=2),
    "mhc": StageSpec(spec="run.mhc-stage.yaml", arm="mhc", stage=3),
}


#: What stage 1's command resolved to, and therefore what all twenty cells must be compared
#: at. Read out of ``run.baseline-stage.yaml`` through the real parser at commit ``38b665919``,
#: which is the commit the five admitted cells were built from.
#:
#: LITERALS AND NOT REFERENCES TO THE CONSTANTS THEY GUARD, WHICH IS THE ENTIRE POINT. Writing
#: ``DEFAULT_WEIGHT_DECAY`` here would move with the default and this table would agree with a
#: drifting number forever. Written as ``0.033`` it disagrees the moment the default moves, and
#: what it is disagreeing with is the value five runs have already been trained at. The same
#: goes for the tranche's own constants: ``TRANCHE_STEPS`` is 6,000 today and stage 1 ran 6,000
#: whatever it becomes.
#:
#: A test cross-checks the whole table against those constants, so a deliberate change to the
#: design fails here rather than silently invalidating stage 1's comparability. What that
#: failure means is not "fix this number" -- it is "stage 1 cannot be compared to stage 2 any
#: more", and the fix is another five baseline cells.
#:
#: ``fail_closed_by_step`` and ``arm`` are absent because they are the two options that are
#: allowed to differ; see :data:`STAGE_CONTRAST_EXEMPT`. So are ``seed`` and ``data_seed``,
#: which the fan-out index owns, and the platform's own ``run_name``, ``save_folder``,
#: ``work_dir`` and ``dataset_*``.
STAGE_PINNED: Dict[str, object] = {
    "model_factory": "hc_370M",
    "sequence_length": 4096,
    "global_batch_size": 786_432,
    "rank_microbatch_size": 16_384,
    "steps": 6_000,
    "warmup_steps": 120,
    "save_interval": 500,
    "eval_interval": 500,
    "monitor_interval": 50,
    "param_dtype": "bfloat16",
    "learning_rate": 7.8e-4,
    "weight_decay": 0.033,
    "z_loss_multiplier": 1e-5,
    "held_out_shards": 2,
    "bytes_per_token": 4.57,
    "partial_rotary_factor": None,
}


#: The resolved options a stage spec may differ from the other two in, and why each one is
#: allowed to.
#:
#: EVERYTHING NOT IN HERE HAS TO BE IDENTICAL ACROSS ALL FOUR STAGES, and that is what the
#: diff test enforces. An allowlist rather than a list of things to check, because the failure
#: is a flag nobody thought about: a checked list silently permits whatever is not on it,
#: which is precisely the flag a later commit adds.
STAGE_CONTRAST_EXEMPT: Dict[str, str] = {
    "arm": (
        "The contrast itself. H1 is arm 2 against arm 1 and H2a is arm 3 against arm 2, so "
        "this is the one difference the experiment is made of."
    ),
    "fail_closed_by_step": (
        "An abort threshold and not a training setting: it changes no parameter, no datum "
        "and no schedule, only whether a run whose lanes never differentiated is killed at "
        "step 400 instead of billed for eighteen hours. Two of the four stages omit it and "
        "for two different reasons, and the diff test asserts both rather than taking this "
        "paragraph's word for either. On BASELINE it is UNREACHABLE -- "
        "train_hyper_connections.train attaches HyperConnectionMonitorCallback only when "
        "arm.hyper_connections is not None, and the baseline's is None -- so the stage-1 "
        "command that omits it and a stage-2 command that sets it describe the same run. On "
        "MHC it is reachable and was expected to fire: the Sinkhorn projection contracts the "
        "lane mixing towards the lane mean, lane dispersion is the statistic that contraction "
        "compresses, and the floor was calibrated against unconstrained mixing, so the guard "
        "was predicted to read a working constraint as a missing mechanism. That prediction "
        "came off a four-block CPU rehearsal and the arm has since run: at step 400 all five "
        "370M cells read a differentiated fraction of 1.0, so the guard would have passed. "
        "MHC_DIFFERENTIATED_FRACTION_AT_GATE_370M carries that measurement and "
        "MHC_LANE_DISPERSION_AT_GATE the rehearsal it replaced. The omission is now an arm "
        "running without a guard rather than an arm that needs to. Neither omission touches "
        "the contrast, because a threshold that never fires and a threshold that is absent "
        "train the same model -- which is as true of a threshold that would have passed as of "
        "one that is unreachable."
    ),
}


#: The four submissions the tranche actually goes out as, after the capacity stall of
#: 2026-08-08 moved it to ``gpu-8xa100`` and after the amendment of the same date turned spike
#: skipping on.
#:
#: WHY THESE ARE A SECOND TABLE AND NOT AN EDIT TO :data:`STAGE_SPECS`. The four L40S specs are
#: the text three admitted submissions were built from and are not edited after the fact, for
#: the reason their own headers give. These are the live ones.
#:
#: THE BASELINE HAS ALREADY RUN ONCE HERE AND ITS ``run_id`` IS STILL ``None``, WHICH IS THE
#: POINT. ``run_019fe2f4-f528`` completed all five cells on this shape and is what
#: ``.edullm/noise-floor.json`` was frozen from. It trained under plain ``AdamW``, so it is not
#: comparable to anything submitted from these files and it is not a stage-1 that stage 2 can
#: be read against. It stays in the record as the measurement that forced the amendment; the
#: baseline it replaced has to be run again, which is why nothing here is marked submitted.
A100_STAGE_SPECS: Dict[str, StageSpec] = {
    "baseline": StageSpec(spec="run.baseline-a100.yaml", arm="baseline", stage=1),
    "faithful": StageSpec(spec="run.faithful-a100.yaml", arm="faithful", stage=2),
    "output-only": StageSpec(spec="run.output-only-a100.yaml", arm="output-only", stage=2),
    "mhc": StageSpec(spec="run.mhc-a100.yaml", arm="mhc", stage=3),
    "no-output-init": StageSpec(spec="run.no-output-init-a100.yaml", arm="no-output-init", stage=4),
}


#: The ``--hours`` the FIRST A100 submission of every stage went out under. History, and a
#: warning. Nothing should be submitted at this number; read :data:`A100_STAGE_HOURS_BY_ARM`.
#:
#: FOUR, DOWN FROM SEVEN, AND IT IS A MEASUREMENT RATHER THAN A TRIM -- so ran the reasoning at
#: the time, and it is left standing below because it is exactly right about the baseline and
#: exactly wrong about everything else, which is the whole lesson. The five baseline cells
#: of ``run_019fe2f4-f528`` ran 2.92 to 3.00 hours each, so 7 was a 2.3x ceiling on a need
#: nobody had yet measured when it was chosen. Four leaves 33% over the slowest observed cell,
#: which is more margin than the L40S stages carried, and it is the hours factor of the
#: approved ceiling -- multiplied by the attempts and by every one of the twenty cells -- so
#: three unneeded hours are three hours charged to the ceiling twenty times over.
#:
#: IT DOES NOT BOUND THE AMENDED RUN ANY TIGHTER THAN IT BOUNDS THE MEASURED ONE. Skipping a
#: step costs the optimizer update and not the forward-backward, so a skipped step takes the
#: same wall clock as a taken one; replayed over the five measured cells the rule declines
#: 0.12% to 0.52% of steps, which moves no cell by so much as a minute.
#:
#: IT HAS TO BE THE SAME ACROSS THE STAGES, for the reason :data:`STAGE_HOURS` gives: a bound
#: is what kills a cell that runs long, so a stage under a looser bound survives drift that
#: another dies of, and a treatment arm missing a cell is missing its slowest one rather than
#: a random one. True, and it is what made a wrong scalar so expensive: sameness was enforced
#: and correctness was not, so the error propagated to every stage at once.
#:
#: THIS NUMBER KILLED FIFTEEN CELLS AND IS KEPT ONLY SO THAT NOTHING CAN QUOTE IT AGAIN. It was
#: the baseline's bound carried onto arms that compute lanes; see
#: :data:`A100_LANE_ARM_CELL_HOURS`. It is deliberately no longer named ``A100_STAGE_HOURS``,
#: because the failure mode this repository actually suffered is a person copying a plausible
#: constant -- or a spec header quoting one -- into a resubmission. The bound each stage is
#: submitted under is :data:`A100_STAGE_HOURS_BY_ARM` and there is no scalar to copy by mistake.
A100_FIRST_SUBMISSION_HOURS = 4.0

#: The ``--hours`` each A100 stage was ACTUALLY submitted under. Not a plan: a record.
#:
#: WHY THIS IS A TABLE AND NOT A SCALAR. It used to be a scalar, at four hours, on the argument
#: :data:`STAGE_HOURS` gives -- a stage under a looser bound survives drift another dies of, so
#: an arm missing a cell is missing its slowest one rather than a random one. That argument is
#: real and it is the same species as the training-dose confound. It is also an argument about
#: *differential survival*, and four hours is a bound that no lane arm survives at all. A bound
#: that kills every cell of an arm is not a control for selection; it is the selection. The
#: resubmissions therefore went out at six and seven hours, and for a day this file, all four
#: spec headers and the test guarding them still said four -- the test passing because it read
#: the same wrong headers it was meant to check.
#:
#: WHAT WAS SUBMITTED, WHICH IS WHAT THESE NUMBERS ARE. ``baseline`` at four, where a 3.00-hour
#: cell leaves 33% and all five cells landed. ``faithful`` at six after its first stage died
#: complete at the four-hour wall; ``run_019fe90b-f99e`` is the resubmission and
#: ``hyper-connections.md`` records it going out at ``--hours 6``. ``output-only`` and ``mhc``
#: at seven. Arm 4 at seven, matching the two most recent submissions rather than departing
#: from anything.
#:
#: THE ONE ASYMMETRY LEFT, AND WHY IT IS NOW EMPTY. Arm 4's comparator in H1b is ``faithful``,
#: which ran under six hours where arm 4 gets seven, so in principle ``faithful`` could lose a
#: slow cell that arm 4 keeps and slow cells are not a random subset. In fact ``faithful``
#: reported five of five under its six-hour bound, as did ``output-only`` at seven and
#: ``baseline`` at four, so no comparator lost a cell to its bound and there is no differential
#: survival to correct. The commitment stands anyway, because ``mhc`` is still running: any
#: arm-4 cell whose runtime exceeds :data:`A100_LANE_ARM_SURVIVORSHIP_HOURS` is reported
#: together with the contrast recomputed without it. Pre-registered before any endpoint was
#: read, so that a looser bound cannot become a silent difference later.
A100_STAGE_HOURS_BY_ARM: Dict[str, float] = {
    "baseline": 4.0,
    "faithful": 6.0,
    "output-only": 7.0,
    "mhc": 7.0,
    "no-output-init": 7.0,
}

#: The runtime above which an arm-4 cell is reported as one a six-hour bound would have killed,
#: and the contrast is recomputed without it. Pre-registered 2026-08-10 with the arm itself, so
#: that the looser bound above cannot become a silent difference between arm 4 and the arms it
#: is read against.
A100_LANE_ARM_SURVIVORSHIP_HOURS = 6.0

#: The longest of the five baseline cells of ``run_019fe2f4-f528``, in hours, which is the
#: only wall clock this shape has ever been measured at rather than predicted at. The five ran
#: 2.92, 2.94, 2.97, 2.99 and 3.00. The ``baseline`` entry of :data:`A100_STAGE_HOURS_BY_ARM`
#: is checked against it, so a ``--hours`` that stops covering the slowest observed cell is a
#: failing test rather than a tranche that dies at the bound.
#:
#: IT IS THE BASELINE'S CELL AND THE BASELINE HAS NO LANES, WHICH IS THE WHOLE OF WHY FIFTEEN
#: CELLS DIED. See :data:`A100_LANE_ARM_CELL_HOURS`.
A100_MEASURED_CELL_HOURS = 3.00

#: The same figure for an arm that has lanes, in hours, projected from the arms' own histories.
#:
#: ``A100_MEASURED_CELL_HOURS`` IS THE BASELINE'S AND WAS CARRIED ONTO ARMS THAT COMPUTE LANES.
#: The baseline runs at 1.700 s/step and the three lane arms at 2.87 to 3.15, so four hours
#: buys a lane arm about 4,950 steps of the 6,000 the tranche asks for. Every cell of all three
#: treatment stages hit that wall -- fifteen of fifteen, at steps 4,640 to 4,995 -- and a
#: timeout is measured to forfeit its second attempt on this workload, so a wall is not a delay
#: but a lost cell. ``hyper-connections.md`` carries the table under "``--hours 4`` was the
#: baseline's number carried onto arms that have lanes".
#:
#: Fitted per cell from step 200 onward, the slowest lane-arm rates are ``output-only`` 2.961,
#: ``faithful`` 3.074 and ``mhc`` 3.149 s/step, projecting 4.98, 5.17 and 5.30 hours. This is
#: ``faithful``'s, because arm 4 differs from ``faithful`` in the initialization of two module
#: families and in nothing that a kernel sees -- the iso-FLOP test asserts as much -- so its
#: step time is ``faithful``'s step time, and ``faithful``'s resubmission confirms the rate on
#: new hosts at 3.027 to 3.083.
A100_LANE_ARM_CELL_HOURS = 5.17

#: What a 6,000-step cell of each arm actually takes, in hours, fitted per cell from step 200
#: onward over the arms' own histories and projected to 6,000 plus the final evaluation and
#: checkpoint. The slowest cell of each arm, not the mean, because a bound kills the slowest.
#:
#: THIS TABLE IS THE ONE THAT WOULD HAVE CAUGHT THE FOUR-HOUR BOUND. A single
#: ``A100_MEASURED_CELL_HOURS`` of 3.00 was checked against a single ``--hours`` of 4.0 and the
#: check passed, because both numbers were the baseline's and neither knew the other arms
#: existed. The arms differ in step time by a factor of 1.8 -- 1.700 s/step with no lanes
#: against 2.87 to 3.15 with them -- so one measured cell can only ever validate one bound.
#: ``test_train_hyper_connections.py`` now walks this table against
#: :data:`A100_STAGE_HOURS_BY_ARM` arm by arm.
#:
#: ``no-output-init`` is entered at ``faithful``'s figure. It differs from ``faithful`` in the
#: initialization of two module families and in nothing a kernel sees -- the iso-FLOP test
#: asserts as much -- so it has ``faithful``'s step time, and ``faithful``'s resubmission
#: confirms that rate on new hosts at 3.027 to 3.083 s/step.
A100_MEASURED_CELL_HOURS_BY_ARM: Dict[str, float] = {
    "baseline": A100_MEASURED_CELL_HOURS,
    "faithful": A100_LANE_ARM_CELL_HOURS,
    "output-only": 4.98,
    "mhc": 5.30,
    "no-output-init": A100_LANE_ARM_CELL_HOURS,
}

#: The ``--attempts`` every A100 stage is submitted under. Two, unchanged, and it still buys
#: only what ``OnStatusReason "Host EC2*" RETRY`` grants -- a lost host.
A100_STAGE_ATTEMPTS = 2

#: What all four A100 commands must resolve to, and therefore what all twenty cells are
#: compared at.
#:
#: THE L40S TABLE PLUS EXACTLY FOUR DELTAS, WRITTEN AS A SPREAD SO THE DELTAS ARE THE REVIEWABLE
#: THING. Everything not listed below is stage 1's own resolution, held as literals in
#: :data:`STAGE_PINNED` for the reason its docstring gives.
#:
#: ``rank_microbatch_size`` is the shape: 16,384 ran on a 48 GB L40S and the same arithmetic
#: puts it at 90% of a 40 GB A100, so this shape is 12,288 and an OOM is not retried.
#:
#: The three optimizer entries are the amendment of 2026-08-08. They are pinned rather than
#: left to the parser because a spec that leans on a default is a spec that changes when the
#: default does, and because this is the one setting whose whole justification is that it is
#: identical on all four arms: an optimizer that differed between arms would confound the
#: contrast far worse than the noise it removes. ``STAGE_CONTRAST_EXEMPT`` does not exempt any
#: of the three, so the diff test refuses a tranche whose arms disagree about them.
A100_STAGE_PINNED: Dict[str, object] = {
    **STAGE_PINNED,
    "rank_microbatch_size": 12_288,
    "optimizer": "skip_step_adamw",
    "skip_step_sigma_factor": 6,
    "skip_step_rolling_interval": 128,
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
    lines.append("")
    lines.append(
        f"submitted as {len(STAGE_SPECS)} stages of {STAGE_CELLS} cells, "
        f"--fanout-index-parameter seed, --hours {STAGE_HOURS:.0f} "
        f"--attempts {STAGE_ATTEMPTS}:"
    )
    for name, stage in STAGE_SPECS.items():
        landed = stage.run_id or "not submitted"
        lines.append(f"  stage {stage.stage}  {name.ljust(width)}  {stage.spec}  ({landed})")
    return "\n".join(lines)
