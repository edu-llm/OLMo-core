"""
Declarative arm builder for the CORE-6 study (LIV / GQA / KDA at sub-400M).

Every arm is one entry in :data:`ARMS` and every arm is built by the same function, so two
arms that differ in one field differ in exactly one visible line.

.. note::
    This is **not** :mod:`olmo_core.nn.transformer.liv_arms`. That module holds the *LIV
    brainlift* arms (low-rank gates, kernel widths, attention budget) and has KDA explicitly
    out of scope. This module holds the *KDA-insertion* arms. They share a geometry and
    nothing else; keeping them separate is deliberate.

The frozen 350M geometry (see ``docs/liv-kda-gqa-sub500m-experiment.md``):

===========================  ==============
Decoder layers               16
``d_model``                  1,024
Query / KV heads             16 / 8
Head dimension               64
SwiGLU branch width          4,608
Vocabulary (tied)            100,352
Global attention indices     2, 5, 8, 10, 12, 14
LIV convolution              causal depthwise, kernel 3
===========================  ==============

``L0``, the released-shape baseline, must come to **390,135,552** parameters at the dolma2
vocabulary. That number is asserted in the tests: it is the check that the whole ledger is
right. At the older 65,536 vocabulary the same geometry gives 354,483,968, and the two differ
by exactly ``34,816 * 1,024`` -- the embedding rows, nothing else.

The primary endpoint is the substitution rate

.. math::  \\sigma_2 = (G4R2 - G4R0) / (G4R0 - L0)

so ``G4R0`` is the denominator and must never be dropped while ``G4R2`` is kept.

.. important::
    ``K2`` narrows the SwiGLU branch to 4,512 **in its two KDA slots only**. KDA is a larger
    mixer than LIV (4,487,248 vs 4,197,376 at ``d_model=1024``), so an unnarrowed ``K2`` would
    outweigh ``L0`` and the primary contrast would confound capacity with mechanism. The
    narrowing lands ``K2`` 10,080 parameters *below* ``L0``, which is the frozen residual: it
    is asserted, not tolerated. This is what removes the need for a separate padded control.

.. important::
    Changing the number of global-attention layers moves the parameter count, because a LIV
    block is **larger** than a GQA block: 18,355,200 vs 17,303,680, a difference of 1,051,520
    per layer. So ``G4R0`` -- which removes two global layers -- is *2,103,040 parameters
    heavier* than ``L0`` before correction, and ``G0R0`` is 6,309,120 heavier. Left
    uncorrected the sigma denominator ``(G4R0 - L0)`` would confound lost retrieval with
    gained capacity, in the direction that *masks* the damage and *inflates* sigma. At the
    protocol's Chinchilla slope (``dL/dlnN = -0.17123``) 2.1M params is worth ~0.0009 nats,
    against a 0.004-nat aggregate margin -- roughly 23% of it, so not ignorable.
    :func:`solve_widths` respends the difference in FFN width, per the arm-design policy:
    graduated widths, every one a multiple of 32, anchored on ``L0`` within +/-0.05%.

.. important::
    Parameter matching is necessary but not sufficient: arms should also be compared on
    ``num_flops_per_token`` at the target context, because attention has a score term that
    grows with sequence length and short convolutions do not. That check needs a *built*
    model, which needs ``flash-linear-attention`` for the KDA arms, so it belongs in the
    GPU-side gate rather than in :func:`arm_report`.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

from olmo_core.config import DType
from olmo_core.nn.attention import (
    AttentionConfig,
    KimiDeltaAttentionConfig,
    SlidingWindowAttentionConfig,
)
from olmo_core.nn.attention.short_conv import ShortConvConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

__all__ = [
    "Core6Arm",
    "ARMS",
    "build_arm",
    "arm_report",
    "solve_widths",
    "L0_PARAM_TARGET",
    "K2_L0_DELTA",
    "WIDTH_TOLERANCE",
]


# --- frozen geometry ----------------------------------------------------------------------

N_LAYERS = 16
D_MODEL = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
SWIGLU_WIDTH = 4608
VOCAB_SIZE = 100352
KERNEL_SIZE = 3
ATTENTION_LAYERS: Tuple[int, ...] = (2, 5, 8, 10, 12, 14)

#: SwiGLU width used in the KDA slots so that ``K2`` anchors down to ``L0``.
KDA_SLOT_SWIGLU_WIDTH = 4512

#: The two LIV slots that become KDA in ``K2`` / ``G4R2``.
KDA_LAYERS: Tuple[int, ...] = (6, 11)

#: Sliding-window span for ``S14``. Must stay **below** the evaluation slice gap.
SWA_WINDOW = 1024

L0_PARAM_TARGET = 390_135_552
"""Exact released-shape parameter count for ``L0`` at the dolma2 vocabulary."""

K2_L0_DELTA = -10_080
"""Frozen ``K2 - L0`` parameter residual after the SwiGLU narrowing. Asserted, not tolerated."""

WIDTH_TOLERANCE = 5e-4
"""Declared arm-matching tolerance: +/-0.05% of :data:`L0_PARAM_TARGET`."""

#: Per-layer parameter cost of each mixer at the frozen geometry, for the width solver.
#: A LIV block is *larger* than a GQA block, which is why removing attention adds parameters.
_BLOCK_PARAMS = {"liv": 18_355_200, "gqa": 17_303_680}


@dataclass(frozen=True)
class Core6Arm:
    """
    One experimental arm, declared rather than scripted.

    :param name: Short arm identifier, e.g. ``"L0"``.
    :param title: Human-readable name used in reports and the protocol.
    :param role: What this arm is for -- baseline, treatment, control, or instrument.
    :param attention_layers: Indices using **global** attention. Every index not listed here
        and not in ``kda_layers`` or ``swa_layers`` uses :class:`ShortConv` (LIV).
    :param kda_layers: Indices using :class:`KimiDeltaAttention`.
    :param swa_layers: Indices using sliding-window attention.
    :param seeds: Number of seeds this arm is run with.
    :param tokens: Training tokens per seed.
    :param notes: Why the arm exists. Dropping an arm whose note names another arm's
        numerator or denominator should look obviously wrong.
    """

    name: str
    title: str
    role: str
    attention_layers: Tuple[int, ...] = ATTENTION_LAYERS
    kda_layers: Tuple[int, ...] = ()
    swa_layers: Tuple[int, ...] = ()
    seeds: int = 3
    tokens: int = 7_100_000_000
    notes: str = ""

    @property
    def liv_layers(self) -> Tuple[int, ...]:
        """Indices that fall through to the LIV short convolution."""
        taken = set(self.attention_layers) | set(self.kda_layers) | set(self.swa_layers)
        return tuple(i for i in range(N_LAYERS) if i not in taken)

    def __post_init__(self) -> None:
        # An index claimed by two mixers is a silent model change: the last writer wins and
        # the arm still builds. Fail at declaration time instead.
        claimed = list(self.attention_layers) + list(self.kda_layers) + list(self.swa_layers)
        if len(claimed) != len(set(claimed)):
            raise ValueError(f"arm {self.name!r}: a layer index is claimed by two mixers")
        for i in claimed:
            if not 0 <= i < N_LAYERS:
                raise ValueError(f"arm {self.name!r}: layer index {i} out of range")


# --- the arms -------------------------------------------------------------------------------
#
# Listed so that each treatment sits next to the control it is measured against.

ARMS: Dict[str, Core6Arm] = {
    a.name: a
    for a in [
        Core6Arm(
            "L0",
            "Released-Geometry Baseline",
            "baseline",
            notes="Faithful LFM2.5-350M schedule: 10 LIV + 6 global GQA. The anchor "
            "everything is measured against, and the capacity control for K2.",
        ),
        Core6Arm(
            "K2",
            "KDA-for-LIV Substitution",
            "treatment (secondary)",
            kda_layers=KDA_LAYERS,
            notes="L0 with LIV slots {6,11} -> KDA, SwiGLU narrowed to 4,512 in those two "
            "slots so the arm anchors to L0 at exactly -10,080 params. The protocol's "
            "original question. Changes 32K state by only +0.13%, which is why it is no "
            "longer the primary contrast.",
        ),
        Core6Arm(
            "G4R0",
            "Attention-Ablated Control",
            "control (sigma denominator)",
            attention_layers=(2, 5, 8, 12),
            notes="L0 with two global layers removed (a: 6->4). Establishes the damage from "
            "losing retrieval. DENOMINATOR of sigma_2 -- never drop while G4R2 is kept.",
        ),
        Core6Arm(
            "G4R2",
            "KDA-Compensated Ablation",
            "treatment (PRIMARY)",
            attention_layers=(2, 5, 8, 12),
            kda_layers=KDA_LAYERS,
            notes="G4R0 plus two KDA layers. Measures how much fixed-state recurrence buys "
            "back the ablated retrieval. NUMERATOR of sigma_2. This is the treatment arm.",
        ),
        Core6Arm(
            "G2R0",
            "Attention Dose Point (a=2)",
            "dose-response",
            attention_layers=(2, 12),
            notes="L0 with four global layers removed (a: 6->2), no KDA. Added 2026-08-01 for "
            "the dose-response wave: with only a=6 and a=0 measured, the shape of CE(a) is "
            "unknown and D = CE(4)-CE(6) could be anywhere in 0.02-0.41 nats depending on "
            "whether the curve is linear or hard-saturating. That 21x spread straddles the "
            "measurability threshold for sigma, so a third interior point is what makes the "
            "curve identifiable rather than a two-point chord. Uses S14's global indices "
            "{2,12} so the two arms differ ONLY in whether the remaining 14 layers are "
            "sliding-window attention or LIV convolutions.",
        ),
        Core6Arm(
            "S14",
            "Sliding-Window Efficiency Baseline",
            "competitor",
            attention_layers=(2, 12),
            swa_layers=(0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15),
            notes="Faithful Gemma-3 transplant: 2 global + 14 sliding-window at W=1024. "
            "Replaces the all-attention strawman with a schedule someone actually ships. "
            "With the slice gap > W its SWA layers are structurally blind to the referent, "
            "so it doubles as a free a=2 dose point.",
        ),
        Core6Arm(
            "G0R0",
            "No-Retrieval Instrument Anchor",
            "instrument validation",
            attention_layers=(),
            seeds=1,
            tokens=1_000_000_000,
            notes="Zero global attention: cannot do long-range retrieval by construction. "
            "Validates that the sliced endpoint responds to the capability under test. "
            "Deliberately cheap -- 1 seed at 1B tokens. If L0 - G0R0 is not large, nothing "
            "downstream is interpretable.",
        ),
    ]
}


def solve_widths(arm: "Core6Arm | str", *, vocab_size: int = VOCAB_SIZE) -> Dict[int, int]:
    """
    Solve per-layer SwiGLU widths so the arm lands on :data:`L0_PARAM_TARGET`.

    Removing a global-attention layer *adds* 1,051,520 parameters, because the LIV block that
    replaces it is larger. Rather than let that ride as a capacity confound, the surplus is
    respent as a **reduction** in FFN width, graduated across layers so every width stays a
    multiple of 32 (a non-multiple lands on a bad GEMM tile).

    The search is exact rather than iterative: one step of width on one layer costs
    ``3 * d_model * 32`` parameters, so the required number of steps is a division, and the
    remainder is spread one step at a time across the lowest-indexed eligible layers.

    :param arm: A :class:`Core6Arm` or the name of one in :data:`ARMS`.
    :param vocab_size: Vocabulary size (does not affect the solve; kept for symmetry).

    :returns: Mapping of layer index -> SwiGLU width, for layers whose width differs from
        :data:`SWIGLU_WIDTH`. KDA slots are excluded: their width is pinned by
        :data:`KDA_SLOT_SWIGLU_WIDTH` to hold the frozen ``K2 - L0`` residual.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]

    # Parameter surplus this arm carries purely from its mixer schedule, relative to L0.
    n_global_l0 = len(ATTENTION_LAYERS)
    n_global = len(arm.attention_layers) + len(arm.swa_layers)
    surplus = (n_global_l0 - n_global) * (_BLOCK_PARAMS["liv"] - _BLOCK_PARAMS["gqa"])
    if surplus == 0:
        return {}

    # One /32 step of SwiGLU width on one layer, in parameters (gate + up + down).
    step_cost = 3 * D_MODEL * 32
    steps, remainder = divmod(surplus, step_cost)

    # Eligible layers: everything except the KDA slots, whose width is already pinned.
    eligible = [i for i in range(N_LAYERS) if i not in arm.kda_layers]
    if not eligible:
        return {}

    base_steps, extra = divmod(steps, len(eligible))
    widths: Dict[int, int] = {}
    for rank, i in enumerate(eligible):
        n = base_steps + (1 if rank < extra else 0)
        if n:
            widths[i] = SWIGLU_WIDTH - n * 32

    # `remainder` is the sub-step residual the /32 grid cannot express. It is reported by
    # `arm_report` and asserted against WIDTH_TOLERANCE in the tests rather than hidden.
    return widths


def build_arm(
    arm: "Core6Arm | str",
    *,
    vocab_size: int = VOCAB_SIZE,
    init_device: str = "meta",
    dtype: DType = DType.float32,
    **kwargs,
) -> TransformerConfig:
    """
    Build the :class:`TransformerConfig` for one arm.

    Defaults to ``init_device="meta"`` so parameter counts can be checked without allocating.

    :param arm: A :class:`Core6Arm` or the name of one in :data:`ARMS`.
    :param vocab_size: Vocabulary size. The frozen ledger uses 100,352, tied.
    :param init_device: Where to place parameters, e.g. ``"meta"``, ``"cpu"``.
    :param dtype: Parameter dtype.

    :returns: A config whose per-layer mixers follow the arm's declaration.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]

    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL,
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        # Pass the width explicitly: llama_like would otherwise derive 8/3*d rounded to 256,
        # which is not the 4,608 the released config implies.
        feed_forward=FeedForwardConfig(hidden_size=SWIGLU_WIDTH, bias=False, dtype=dtype),
        # LFM2's attention applies RMSNorm to Q and K per head. Both default to False in
        # `llama_like`; omitting them costs exactly n_attn_layers x 2 x head_dim parameters.
        qk_norm=True,
        use_head_qk_norm=True,
        dtype=dtype,
        **kwargs,
    )
    # The frozen ledger specifies TIED embeddings; `llama_like` defaults to untied, which adds
    # a second vocab x d_model tensor -- 102,760,448 parameters at the frozen geometry, or
    # ~26% of the model. Large enough to invalidate every arm-matching decision silently.
    cfg.tie_word_embeddings = True

    assert isinstance(
        cfg.block, TransformerBlockConfig
    ), "llama_like should return a single block config, not a named-block dict"

    # Per-layer overrides. NOTE the field is `sequence_mixer`, NOT `attention`: setting
    # `.attention` on a block *config* silently creates a new attribute, the override is
    # ignored, and every layer stays attention -- a model that builds, trains, and answers a
    # different question. The tests assert layer types for exactly this reason.
    overrides: Dict[int, TransformerBlockConfig] = {}

    # Solved FFN widths that hold the arm on the L0 anchor. Layers absent from this mapping
    # keep the released 4,608.
    solved = solve_widths(arm, vocab_size=vocab_size)

    def _ff(layer_idx: int) -> FeedForwardConfig:
        return FeedForwardConfig(
            hidden_size=solved.get(layer_idx, SWIGLU_WIDTH), bias=False, dtype=dtype
        )

    liv_mixer = ShortConvConfig(kernel_size=KERNEL_SIZE, bias=False, dtype=dtype)
    for i in arm.liv_layers:
        overrides[i] = replace(cfg.block, sequence_mixer=liv_mixer, feed_forward=_ff(i))

    # Global-attention layers keep the released mixer but may carry a solved width, so they
    # need an explicit override too whenever the solver touched them.
    for i in arm.attention_layers:
        if i in solved:
            overrides[i] = replace(cfg.block, feed_forward=_ff(i))

    # KDA slots also narrow the SwiGLU branch, so they need their own block config rather
    # than a shared one.
    kda_block = replace(
        cfg.block,
        sequence_mixer=KimiDeltaAttentionConfig(n_heads=N_HEADS, head_dim=HEAD_DIM, dtype=dtype),
        feed_forward=FeedForwardConfig(hidden_size=KDA_SLOT_SWIGLU_WIDTH, bias=False, dtype=dtype),
    )
    for i in arm.kda_layers:
        overrides[i] = kda_block

    if arm.swa_layers:
        assert isinstance(cfg.block.sequence_mixer, AttentionConfig)
        # `pattern` is indexed per layer, and force_full_attention_on_{first,last}_layer both
        # default to True -- which would silently make layers 0 and 15 global and give this
        # arm 4 global layers instead of the 2 it declares. Turn both off and carry the
        # schedule entirely in the declaration above.
        swa_block = replace(
            cfg.block,
            sequence_mixer=replace(
                cfg.block.sequence_mixer,
                sliding_window=SlidingWindowAttentionConfig(
                    pattern=[SWA_WINDOW],
                    force_full_attention_on_first_layer=False,
                    force_full_attention_on_last_layer=False,
                ),
            ),
        )
        for i in arm.swa_layers:
            overrides[i] = replace(swa_block, feed_forward=_ff(i))

    cfg.block_overrides = overrides or None

    return cfg


def arm_report(
    names: Optional[List[str]] = None,
    *,
    vocab_size: int = VOCAB_SIZE,
) -> str:
    """
    Tabulate parameters for each arm, relative to ``L0``.

    Counts come from the config rather than from a built module, so this runs on a laptop:
    :class:`KimiDeltaAttention` asserts ``has_fla()`` at construction, which would make the
    KDA arms unreportable without a GPU.

    :param names: Arms to include. Defaults to all of :data:`ARMS`.
    :param vocab_size: Vocabulary size to build with.

    :returns: A printable table.
    """
    names = names or list(ARMS)
    base_params = build_arm("L0", vocab_size=vocab_size).num_params

    head = f"{'arm':<8}{'params':>14}{'vs L0':>11}{'dev':>10}"
    lines = [head, "-" * len(head)]

    for name in names:
        params = build_arm(name, vocab_size=vocab_size).num_params
        dev = (params - base_params) / base_params
        lines.append(f"{name:<8}{params:>14,}{params - base_params:>+11,}{dev:>+10.4%}")

    return "\n".join(lines)
