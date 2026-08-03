"""
The width-scaled ladder: four models at fixed depth 12, differing only in ``d_model``.

Depth is fixed because reasoning capability tracks depth while fact capacity tracks total
parameters, so a depth-scaled ladder would confound the experiment's two axes. OLMo-core's own
presets vary both, which is why this module overrides ``n_layers`` rather than using them as shipped.

**Two ways to count parameters, and only one of them is authoritative.**
:func:`non_embedding_params` is closed-form arithmetic that needs no ``torch``, so a budget can be
computed anywhere; it reproduces OLMo-core's own preset names to within 0.2%. :func:`build`
constructs the real :class:`~olmo_core.nn.transformer.TransformerConfig` and reads the count off it,
then asserts the two agree. **The built count is what :mod:`factcrowd.ladder.rho` must be given**,
because :math:`\\rho` is computed from it and a 1% error in capacity is a 1% error in every entity
count.

The assertion is not ceremony. The FFN hidden size is not ``8 * d_model / 3``: every ``olmo2_*``
factory multiplies that by 1.5 and rounds up to a multiple of 256, so ``d_ffn`` is 4x ``d_model`` at
these widths. A hand formula that misses the multiplier reports exactly 75% of
the real count at every width in this ladder, so the real model is a third bigger than planned --
and since cost scales as :math:`P^2` at fixed :math:`\\rho`, a grid built on it would cost 1.78x its
budget. The tolerance is wide enough for the formula's own 0.2% error and far too tight for that.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from olmo_core.exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    from olmo_core.nn.transformer import TransformerConfig

__all__ = [
    "N_LAYERS",
    "HIDDEN_SIZE_MULTIPLIER",
    "HIDDEN_SIZE_MULTIPLE_OF",
    "HEAD_DIM",
    "BLOCK_NAME",
    "LadderRow",
    "LADDER",
    "feed_forward_hidden_size",
    "non_embedding_params",
    "row",
    "llama_like_kwargs",
    "build",
]


N_LAYERS = 12
"""
Depth, fixed for every model in the ladder.

Reasoning capability tracks depth -- 3-hop accuracy is 13/55/100/100 at 2/3/4/5 layers -- while fact
capacity tracks total parameters, which width supplies. Holding depth fixed holds reasoning
capability roughly fixed while occupancy varies, so a reasoning change is attributable to fact load
rather than to architecture. Dynamic range for the reasoning measurement comes from *task* depth
instead.
"""

HIDDEN_SIZE_MULTIPLIER = 1.5
"""The FFN multiplier every ``olmo2_*`` factory applies on top of ``8 * d_model / 3``."""

HIDDEN_SIZE_MULTIPLE_OF = 256
"""The FFN hidden size is rounded up to a multiple of this, which is what makes every width in
:data:`LADDER` land on an exact multiple and so gives ``d_ffn == 4 * d_model`` throughout."""

HEAD_DIM = 64
"""Attention head dimension, so ``n_heads`` is ``d_model // 64`` for every row."""

BLOCK_NAME = "reordered_norm"
"""
The OLMo2 block variant, held as a string so :func:`llama_like_kwargs` needs no ``torch``.

``TransformerBlockType`` is a :class:`~olmo_core.config.StrEnum`, so this is the same value;
:func:`build` coerces it to the enum before the config is constructed.
"""


@dataclass(frozen=True)
class LadderRow:
    """
    One rung: a width, and the non-embedding parameter count it is expected to produce.

    ``label`` is named for the measured non-embedding count rather than a round-number target,
    because the label is cosmetic and the measurement is not.

    :param label: Row name, e.g. ``"28M"``. Used as a config key and in plots.
    :param d_model: Model width. The only thing that varies across rows.
    :param expected_non_embedding_params: What :func:`non_embedding_params` should return, restated
        here so a change to either shows up as a mismatch rather than as quiet agreement.
    """

    label: str
    d_model: int
    expected_non_embedding_params: int

    @property
    def n_heads(self) -> int:
        """Heads at :data:`HEAD_DIM` each."""
        if self.d_model % HEAD_DIM != 0:
            raise OLMoConfigurationError(
                f"row '{self.label}': d_model {self.d_model} is not a multiple of the head "
                f"dimension {HEAD_DIM}"
            )
        return self.d_model // HEAD_DIM

    @property
    def d_ffn(self) -> int:
        """FFN hidden size, from :func:`feed_forward_hidden_size`."""
        return feed_forward_hidden_size(self.d_model)


LADDER: Tuple[LadderRow, ...] = (
    LadderRow("13M", 256, 12_595_456),
    LadderRow("28M", 384, 28_330_368),
    LadderRow("64M", 576, 63_729_216),
    LadderRow("113M", 768, 113_283_840),
)
"""
The four rungs. Steps of 2.25x, 2.25x, then 1.78x in non-embedding parameters.

The last step is the short one, deliberately. 768 is the width OLMo-core's own ``olmo2_190M``
uses, so ``d_ffn`` matches the preset exactly, and the top row is only two cells -- its job is to
break the size confound at the top end, not to extend the progression. Holding a 2.1x step would
need ``d_model=832``, which costs 38% more on the most expensive row in the grid and buys no
additional science.

**The 113M row is cut from the first run**, and on identification grounds rather than cost: width
scaling does not hold reasoning capability fixed (Mano moves +18.2pp across this ladder at fixed
depth 12), so that row could not break the size confound it was added to break. It stays in the
ladder because :func:`build` and the reasoning-only width arm still use it. See PRD.md sections 3.2
and 8.4.

The first run is 13M and 28M at all five demand levels and 64M at four, dropping 64M's highest.
"""


def feed_forward_hidden_size(
    d_model: int,
    *,
    multiplier: float = HIDDEN_SIZE_MULTIPLIER,
    multiple_of: int = HIDDEN_SIZE_MULTIPLE_OF,
) -> int:
    """
    The FFN hidden size an ``olmo2_*`` factory produces for a given width.

    Mirrors ``TransformerConfig.llama_like``: ``int(8 * d_model / 3)``, times ``multiplier``,
    rounded *up* to a multiple of ``multiple_of``. Reimplemented rather than imported so that
    budget arithmetic does not need ``torch``; :func:`build` asserts the two agree.

    :param d_model: Model width.
    :param multiplier: FFN multiplier. Defaults to :data:`HIDDEN_SIZE_MULTIPLIER`.
    :param multiple_of: Rounding quantum. Defaults to :data:`HIDDEN_SIZE_MULTIPLE_OF`.

    :returns: FFN hidden size.

    :raises OLMoConfigurationError: If ``d_model`` is not positive.
    """
    if d_model <= 0:
        raise OLMoConfigurationError(f"'d_model' must be positive, got {d_model}")
    hidden = int(multiplier * int(8 * d_model / 3))
    return multiple_of * math.ceil(hidden / multiple_of)


def non_embedding_params(
    d_model: int, *, n_layers: int = N_LAYERS, d_ffn: Optional[int] = None
) -> int:
    """
    Closed-form non-embedding parameter count for one of these models.

    Per block: ``4 * d^2`` for the q/k/v/o projections, ``3 * d * d_ffn`` for the SwiGLU feed
    forward, ``2 * d`` for the two block norms and ``2 * d`` for the q/k norms; plus ``d`` for the
    final norm. Accurate to about 0.2% against OLMo-core's own presets -- the norm terms are below
    that resolution, so treat this as a planning estimate and :func:`build` as the measurement.

    :param d_model: Model width.
    :param n_layers: Depth. Defaults to :data:`N_LAYERS`.
    :param d_ffn: FFN hidden size. Derived from ``d_model`` when omitted.

    :returns: Estimated non-embedding parameter count.

    :raises OLMoConfigurationError: If ``d_model`` or ``n_layers`` is not positive.
    """
    if n_layers <= 0:
        raise OLMoConfigurationError(f"'n_layers' must be positive, got {n_layers}")
    hidden = feed_forward_hidden_size(d_model) if d_ffn is None else d_ffn
    attention = 4 * d_model * d_model + 2 * d_model  # projections + q/k norms
    feed_forward = 3 * d_model * hidden
    block_norms = 2 * d_model
    return n_layers * (attention + feed_forward + block_norms) + d_model


def row(label: str) -> LadderRow:
    """
    Look up a rung by label.

    :param label: Row name, e.g. ``"64M"``.

    :returns: The row.

    :raises OLMoConfigurationError: If no row has that label.
    """
    rows: Dict[str, LadderRow] = {r.label: r for r in LADDER}
    if label not in rows:
        raise OLMoConfigurationError(
            f"no ladder row '{label}'; the ladder is {sorted(rows)}. Rows are named for their "
            f"measured non-embedding parameter count."
        )
    return rows[label]


def llama_like_kwargs(
    ladder_row: LadderRow, vocab_size: int, *, tie_word_embeddings: bool = True
) -> Dict[str, Any]:
    """
    The exact keyword arguments :func:`build` hands to ``TransformerConfig.llama_like``.

    Split out from :func:`build` so the argument set can be tested without ``torch`` installed. That
    is not a convenience: routing through ``TransformerConfig.olmo2_190M`` instead looks correct,
    type-checks, and **raises ``TypeError`` for every row**, because that factory passes
    ``d_model=768`` explicitly *and* splats ``**kwargs``, so supplying ``d_model`` collides. It pops
    ``n_layers`` and ``n_heads`` but not ``d_model``. The silent failure is worse than the loud one:
    passing ``n_heads`` alone succeeds and returns a 768-wide model with ``head_dim=192``, which
    trains, reports a loss curve, and is not the model the ladder asked for.

    The values below are ``olmo2_190M``'s own settings, restated here rather than inherited, so this
    is the one place the architecture is pinned.

    :param ladder_row: Which rung.
    :param vocab_size: Tokenizer vocabulary size.
    :param tie_word_embeddings: Tie the input and output embeddings.

    :returns: Keyword arguments for ``TransformerConfig.llama_like``.
    """
    return {
        "d_model": ladder_row.d_model,
        "vocab_size": vocab_size,
        "n_layers": N_LAYERS,
        "n_heads": ladder_row.n_heads,
        "hidden_size_multiplier": HIDDEN_SIZE_MULTIPLIER,
        "hidden_size_multiple_of": HIDDEN_SIZE_MULTIPLE_OF,
        "block_name": BLOCK_NAME,
        "qk_norm": True,
        "rope_theta": 500_000,
        "layer_norm_eps": 1e-6,
        "tie_word_embeddings": tie_word_embeddings,
    }


def build(
    ladder_row: LadderRow,
    vocab_size: int,
    *,
    tie_word_embeddings: bool = True,
    tolerance: float = 0.01,
    **overrides,
) -> "TransformerConfig":
    """
    Build the real config for a rung and verify its parameter count before returning it.

    Calls ``llama_like`` directly rather than an ``olmo2_*`` factory -- see
    :func:`llama_like_kwargs` for why that distinction is load-bearing.

    Embeddings are tied by default. At a 32k vocab and ``d_model=256`` an untied pair is 16.4M
    parameters against 12.6M non-embedding -- 57% of the model in a table :math:`\\rho` may exclude.
    Tied it is 8.2M and 39%, which is the same argument that chose a 32k vocab over 100k.

    :param ladder_row: Which rung.
    :param vocab_size: Tokenizer vocabulary size.
    :param tie_word_embeddings: Tie the input and output embeddings. Leave on.
    :param tolerance: Maximum tolerated relative gap between the built count and the row's expected
        count. Defaults to 1%: far wider than :func:`non_embedding_params`' own 0.2% error and far
        tighter than the 33% a missed FFN multiplier would cause.
    :param overrides: Merged over :func:`llama_like_kwargs`.

    :returns: The config, with its non-embedding count checked.

    :raises OLMoConfigurationError: If the built count disagrees with the row by more than
        ``tolerance``.
    """
    from olmo_core.nn.transformer import TransformerBlockType, TransformerConfig

    kwargs = llama_like_kwargs(ladder_row, vocab_size, tie_word_embeddings=tie_word_embeddings)
    kwargs.update(overrides)
    kwargs["block_name"] = TransformerBlockType(kwargs["block_name"])
    config = TransformerConfig.llama_like(**kwargs)

    built = config.num_non_embedding_params
    expected = ladder_row.expected_non_embedding_params
    relative_error = abs(built - expected) / expected
    if relative_error > tolerance:
        raise OLMoConfigurationError(
            f"row '{ladder_row.label}' (d_model={ladder_row.d_model}, n_layers={N_LAYERS}) built "
            f"to {built:,} non-embedding params against an expected {expected:,}, a "
            f"{relative_error:.1%} gap over the {tolerance:.0%} tolerance. rho is computed from "
            f"this count, so every entity count in the row is off by the same amount and cost "
            f"scales as its square. Either OLMo-core's FFN sizing changed -- check "
            f"hidden_size_multiplier and hidden_size_multiple_of in TransformerConfig.llama_like "
            f"-- or LADDER needs updating to what the model actually is."
        )
    return config
