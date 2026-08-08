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
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional

from olmo_core.nn.residual_stream import HyperConnectionConfig, HyperConnectionMode
from olmo_core.nn.transformer import (
    BlockReuseConfig,
    TransformerBlockType,
    TransformerConfig,
)

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
    """How many seeds it needs. Three where a difference gets claimed, one for reconnaissance."""

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
        "interface forces.",
        isolates="Cause 1. Whether the field's negative result is an artifact of an "
        "incomplete reimplementation.",
        seeds=1,
        hyper_connections=_hc(mode=HyperConnectionMode.output),
    ),
    "no-output-init": Arm(
        number=4,
        summary="Faithful, but without the sqrt(n) output-module initialization scaling.",
        isolates="Cause 2. Whether that scaling is load-bearing or cosmetic.",
        seeds=1,
        hyper_connections=_hc(output_init_exponent=0.0),
    ),
    "decay-everything": Arm(
        number=5,
        summary="Faithful, but with weight decay on the static component too.",
        isolates="Cause 3. The parameter-group split the replication does not mention.",
        seeds=1,
        # The split lives in the optimizer, so this arm is the faithful model with
        # `optim_group_overrides` deliberately not applied. See `train_hyper_connections.py`.
        hyper_connections=_hc(),
    ),
    "n1": Arm(
        number=6,
        summary="DHC x1.",
        isolates="The seesaw control. ByteDance found n=1 does not beat the baseline; if it "
        "does here, their mechanism story is incomplete at this scale.",
        seeds=1,
        hyper_connections=_hc(n_lanes=1),
    ),
    "n2": Arm(
        number=7,
        summary="DHC x2.",
        isolates="The expansion-rate curve.",
        seeds=1,
        hyper_connections=_hc(n_lanes=2),
    ),
    "n8": Arm(
        number=8,
        summary="DHC x8.",
        isolates="The expansion-rate curve, at the point where they found returns flatten.",
        seeds=1,
        hyper_connections=_hc(n_lanes=8),
    ),
    "mhc": Arm(
        number=9,
        summary="mHC x4: the lane-mixing matrix projected onto the Birkhoff polytope by "
        "Sinkhorn-Knopp.",
        isolates="Whether the constraint is what rescues the method. It ships in DeepSeek V4.",
        seeds=3,
        hyper_connections=_hc(doubly_stochastic=True),
    ),
    "tied-faithful": Arm(
        number=10,
        summary="Tied blocks on a cycle, with DHC x4.",
        isolates="Cause 5. Whether lane value tracks parameter reuse rather than model size.",
        seeds=1,
        hyper_connections=_hc(),
        reuse_factor=REUSE_FACTOR,
    ),
    "tied-baseline": Arm(
        number=11,
        summary="Tied blocks on a cycle, standard residual stream.",
        isolates="The control for arm 10. Without it arm 10 measures tying, not lanes.",
        seeds=1,
        reuse_factor=REUSE_FACTOR,
    ),
}


#: The order to cut arms in if the budget does not stretch, from the plan.
CUT_ORDER = ["n8", "n2", "tied-faithful", "tied-baseline"]


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
    Everything else here is copied from it deliberately, including the flash-2 backend and the
    sliding-window pattern, so the rehearsal exercises the kernels the real run will use.
    """
    from olmo_core.nn.attention import (
        AttentionBackendName,
        SlidingWindowAttentionConfig,
    )

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
        sliding_window=kwargs.pop(
            "sliding_window",
            SlidingWindowAttentionConfig(
                force_full_attention_on_first_layer=False,
                force_full_attention_on_last_layer=True,
                pattern=[4096, 4096, 4096, -1],
            ),
        ),
        attn_backend=kwargs.pop("attn_backend", AttentionBackendName.flash_2),
        **kwargs,
    )


def install() -> None:
    """
    Put the rehearsal factory on :class:`TransformerConfig`, so that the platform's
    ``--model-factory`` reaches it through the stock attribute lookup:
    ``getattr(TransformerConfig, opts.model_factory)``.
    """
    if not hasattr(TransformerConfig, "hc_rehearsal"):
        TransformerConfig.hc_rehearsal = staticmethod(hc_rehearsal)  # type: ignore[attr-defined]


def describe() -> str:
    """
    The table, for ``--list-arms`` and for the run log.
    """
    width = max(len(name) for name in ARMS)
    lines = [f"{'arm'.ljust(width)}  #   seeds  isolates"]
    for name, arm in ARMS.items():
        lines.append(f"{name.ljust(width)}  {arm.number:<2}  {arm.seeds:<5}  {arm.isolates}")
    return "\n".join(lines)
