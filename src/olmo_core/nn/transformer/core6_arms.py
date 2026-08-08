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
from typing import Callable, Dict, List, Optional, Tuple

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
    "mixer_config",
    "mixer_params",
    "MIXERS",
    "ARM_L0_DELTA",
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

#: Every arm's **exact** parameter residual against ``L0``, declared rather than tolerated.
#:
#: :data:`K2_L0_DELTA` used to be the only such constant, and the bake-off is what forced this to
#: become a table. ``K2_L0_DELTA`` is asserted with ``==`` because an anchor that drifts silently
#: is the failure this whole module exists to prevent -- but a single shared constant only works
#: while every KDA-slot arm has the *same* mixer. ``KDA_GCONV`` adds ``2 * 3 * 2 * hidden`` gate
#: parameters and lands at ``+2,208``; ``KDA_R2`` and ``GDN2`` carry mixers millions of parameters
#: larger and land where the /32 width grid can put them. Loosening the exact assertion to a
#: tolerance would have covered all of that and thrown away the check; giving each arm its own
#: declared exact number keeps every arm's ledger asserted.
#:
#: These are hand-derived from the config classes' own ``num_params`` algebra and verified against
#: it by ``test_every_arm_lands_on_its_declared_delta``. A number here that disagrees with the
#: solver is a bug in one of the two, and the test says which.
ARM_L0_DELTA: Dict[str, int] = {
    "L0": 0,
    "K2": K2_L0_DELTA,
    "G4R0": 38_656,
    "G4R2": 28_576,
    "G2R0": 77_312,
    "S14": 3_328,
    "G0R0": 17_664,
    # The bake-off arms. Every one puts its mixer in the same two slots as K2, so they differ from
    # K2 by exactly (their mixer - KDA) x 2, respent on the /32 grid by `solve_widths`.
    "KDA_R2": 6_304,
    "KDA_R1": K2_L0_DELTA,
    "GDN2": 22_688,
    "KDA_BASE": K2_L0_DELTA,
    "KDA_GCONV": 2_208,
    "KDA_NOACT": K2_L0_DELTA,
    # 'allow_neg_eigval' is stored as a plain bool (recurrent.py:708) and read exactly once, inside
    # 'forward' (:875), so it allocates nothing and this arm is parameter-identical to KDA_BASE.
    "KDA_NEGEIG": K2_L0_DELTA,
}

#: Per-layer parameter cost of each mixer at the frozen geometry, for the width solver.
#: A LIV block is *larger* than a GQA block, which is why removing attention adds parameters.
#: ``kda`` is the BARE MIXER, not a block: it is the reference the bake-off's width solver
#: measures each arm's operator against, and the two shipped-KDA arms are matched by
#: :data:`KDA_SLOT_SWIGLU_WIDTH` rather than by the solver.
_BLOCK_PARAMS = {"liv": 18_355_200, "gqa": 17_303_680, "kda": 4_487_248}


# --- the mixers under test ------------------------------------------------------------------
#
# Each entry is a zero-argument factory rather than a config instance, because a config is
# mutable and a shared instance would let one arm's `replace` reach another arm. The factory is
# also what keeps the GDN-2 import lazy -- see `mixer_config`.


def _kda(**kwargs) -> KimiDeltaAttentionConfig:
    """A :class:`KimiDeltaAttentionConfig` at the frozen head geometry."""
    return KimiDeltaAttentionConfig(n_heads=N_HEADS, head_dim=HEAD_DIM, **kwargs)


def _kda_householder(*, num_householder: int):
    """
    A :class:`KimiDeltaHouseholderConfig` at the frozen head geometry.

    ``allow_neg_eigval=True`` IS THE MECHANISM AND IS NOT A DEFAULT. The class defaults it to
    ``False``, which keeps ``beta`` in ``(0, 1)`` -- and ``(I - beta k k^T)`` with ``beta < 1`` is
    a contraction, not a reflection. The Householder *reflection* the arm is named for needs
    ``beta`` to reach 2, which is exactly what this flag buys. An arm built with the default would
    train stably, cost the same, and answer a different question, so it is passed explicitly here
    rather than left to the class.
    """
    from olmo_core.nn.attention import KimiDeltaHouseholderConfig

    return KimiDeltaHouseholderConfig(
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        num_householder=num_householder,
        allow_neg_eigval=True,
    )


def _gdn2():
    """
    A :class:`GatedDeltaNet2Config` at the frozen head geometry.

    ``expand_v`` IS PASSED EXPLICITLY EVEN THOUGH 1.0 IS THE DEFAULT. GDN-2's defaults deliberately
    invert :class:`GatedDeltaNetConfig`'s -- ``expand_v`` 1.0 against 2.0, ``allow_neg_eigval``
    ``False`` against ``True`` -- so "the default" is an ambiguous instruction here and a reader
    comparing the two arms cannot tell which convention this one followed. Writing it down costs
    one line and removes the ambiguity. At 1.0 the mixer is 6,568,016 parameters, which
    :func:`solve_widths` matches back to the anchor; at 2.0 it is 10,112,144 and the widths the
    solver would need are far enough from 4,608 to be a different model.

    ``allow_neg_eigval`` is left at the class default of ``False`` ON PURPOSE, and that is not an
    oversight mirrored from the Householder arms: the GDN-2 paper's headline model keeps the erase
    gate in ``[0, 1]`` and its Table 5 reports the widened range as an ablation with no consistent
    gain at 1.3B. This arm is the paper's model, so it takes the paper's setting.
    """
    from olmo_core.nn.attention import GatedDeltaNet2Config

    return GatedDeltaNet2Config(n_heads=N_HEADS, head_dim=HEAD_DIM, expand_v=1.0)


#: The mixer each arm drops into its KDA slots, by name.
#:
#: Registered as factories so that :data:`ARMS` can name a mixer without importing it. ``GDN2``
#: and the Householder arms live in modules that are still landing on this branch, and a
#: module-level instance would make the whole arm registry unimportable while that is true.
MIXERS: Dict[str, Callable[[], object]] = {
    "kda": lambda: _kda(),
    "kda_noact": lambda: _kda(conv_activation=None),
    "kda_gconv": lambda: _kda(gated_conv=True, gate_structure="depthwise"),
    # THE REFLECTION REGIME ON THE SHIPPED CHUNKED KERNEL. This is the same mechanism the
    # Householder arms were built for, obtained WITHOUT their kernel: 'allow_neg_eigval' is nothing
    # but 'beta = beta * 2.0' in eager PyTorch (recurrent.py:874-876), applied before 'beta' is
    # handed to 'dispatch_chunk_kda' (:918) as a plain post-sigmoid tensor. It selects no kernel, no
    # branch and no flag inside fla. See the KDA_NEGEIG arm's notes for why that matters.
    "kda_negeig": lambda: _kda(allow_neg_eigval=True),
    "kda_householder_r1": lambda: _kda_householder(num_householder=1),
    "kda_householder_r2": lambda: _kda_householder(num_householder=2),
    "gdn2": _gdn2,
}


@dataclass(frozen=True)
class Core6Arm:
    """
    One experimental arm, declared rather than scripted.

    :param name: Short arm identifier, e.g. ``"L0"``.
    :param title: Human-readable name used in reports and the protocol.
    :param role: What this arm is for -- baseline, treatment, control, or instrument.
    :param attention_layers: Indices using **global** attention. Every index not listed here
        and not in ``kda_layers`` or ``swa_layers`` uses :class:`ShortConv` (LIV).
    :param kda_layers: Indices using the arm's ``mixer``.
    :param swa_layers: Indices using sliding-window attention.
    :param mixer: Key into :data:`MIXERS` naming the operator that fills ``kda_layers``.
        Defaults to ``"kda"``, the shipped Kimi Delta Attention, so every arm that predates the
        bake-off keeps exactly the mixer it was measured with.
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
    mixer: str = "kda"
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
        if self.mixer not in MIXERS:
            raise ValueError(
                f"arm {self.name!r}: unknown mixer {self.mixer!r}; known: {sorted(MIXERS)}"
            )
        # A mixer named but never placed is an arm that silently IS its own control: it builds,
        # trains, and produces a curve identical to the arm it was supposed to differ from.
        if self.mixer != "kda" and not self.kda_layers:
            raise ValueError(
                f"arm {self.name!r}: mixer {self.mixer!r} is declared but 'kda_layers' is empty, "
                "so the mixer is never placed and this arm is a duplicate of L0 under a "
                "different name"
            )


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
        # --- the mixer bake-off ---------------------------------------------------------------
        #
        # Arms that differ from each other in the OPERATOR filling slots {6, 11} and in nothing
        # else. They keep L0's six global-attention layers, L0's LIV layers everywhere else, and
        # the same two slots K2 uses. The set is NOT a fixed count -- run 2 adds KDA_NEGEIG and
        # retires the two Householder arms -- so the guard that stops an arm slipping in unchecked
        # is `test_every_arm_has_a_declared_topology`, which compares SETS, not a literal number.
        #
        # THE SIX GLOBAL LAYERS ARE NOT NEGOTIABLE AND THE REASON IS MECHANICAL, not stylistic.
        # `model.py:257` emits RoPE buffers only for `Attention`/`FusedAttention` blocks, so an
        # arm that drops attention layers drops POSITIONAL ENCODING with them. Its loss gap would
        # then be mostly missing position rather than mixer quality -- a large, believable,
        # entirely uninterpretable number. Every arm here therefore carries the same a=6.
        #
        # THE SLOTS STAY {6, 11}. That is `KDA_LAYERS`, the placement K2 and G4R2 already use, and
        # holding it fixed is what makes this a bake-off: with the slot set constant, arm minus
        # arm is the operator and only the operator. Widening the treatment to more slots would
        # buy statistical power and spend the thing being measured -- the difference would then
        # confound "which mixer" with "how much mixer", and it could not be compared against the
        # K2/G4R2 numbers already taken at two slots. No document in this tree justifies more, so
        # two it stays.
        Core6Arm(
            "KDA_BASE",
            "Kimi Delta Attention (shipped)",
            "bake-off reference",
            kda_layers=KDA_LAYERS,
            mixer="kda",
            notes="The shipped KDA operator, 'gated_conv=False', in slots {6,11}. Geometrically "
            "identical to K2 -- same mixer, same slots, same -10,080 residual -- and kept under "
            "its own name because it is the REFERENCE every other bake-off arm subtracts. K2 is "
            "declared by the sigma protocol and must not be renamed or re-pointed to serve this "
            "comparison; an arm that two protocols both depend on is an arm neither can change.",
        ),
        Core6Arm(
            "KDA_NOACT",
            "KDA, Convolution Activation Removed",
            "bake-off isolating control",
            kda_layers=KDA_LAYERS,
            mixer="kda_noact",
            notes="KDA with 'conv_activation=None' and NO gate. Exists so that the gate can be "
            "measured on its own. THE DEPTHWISE PRE-GATE IS ALGEBRAICALLY A SiLU: "
            "2*sigmoid(a*u)*u == (2/a)*silu(a*u) exactly, with the 2/a absorbed into the "
            "convolution taps (verified to 8.9e-16 in fp64). So 'gated_conv=True, "
            "activation=None' is NOT activation-free, and KDA_GCONV - KDA_BASE moves three "
            "things at once: it adds the post gate, it makes the activation's slope learnable, "
            "and it moves the activation to before the convolution. That difference cannot be "
            "attributed to any of them. KDA_GCONV - KDA_NOACT is the contrast that can: against "
            "this arm the only remaining degree of freedom is the POST gate, 1 real dof per "
            "channel rather than 2. Same parameter count as KDA_BASE (-10,080): removing an "
            "activation removes no parameters, which is what makes this a free control.",
        ),
        Core6Arm(
            "KDA_GCONV",
            "KDA + LIV-Style Gated Convolution",
            "bake-off treatment",
            kda_layers=KDA_LAYERS,
            mixer="kda_gconv",
            notes="KDA whose three short convolutions carry LFM2/LIV-style depthwise gates. "
            "Read against KDA_NOACT, not KDA_BASE -- see that arm. DEPTHWISE, NOT LOWRANK, and "
            "the choice is forced twice over: lowrank costs +2,359,296 parameters, 12x the "
            "declared tolerance, so the arm would not be parameter-matched at all; and it adds "
            "nine nn.Linear per layer whose reset_parameters draw from the GLOBAL rng before the "
            "seeded generator exists, so its random stream diverges from every other arm and "
            "seed pairing is forfeited. Depthwise costs 6,144 per layer, landing this arm at "
            "+2,208 against L0 -- inside tolerance and asserted exactly.",
        ),
        Core6Arm(
            "KDA_NEGEIG",
            "KDA, Negative Eigenvalues (beta in (0,2))",
            "bake-off treatment",
            kda_layers=KDA_LAYERS,
            mixer="kda_negeig",
            notes="KDA with allow_neg_eigval=True, so beta reaches (0,2) and (I - beta k k^T) can "
            "be a true REFLECTION rather than a contraction. THIS IS THE MECHANISM KDA_R1 WAS "
            "BUILT FOR, OBTAINED ON THE SHIPPED CHUNKED KERNEL. It is a beta-projection change and "
            "nothing else: 'allow_neg_eigval' is read exactly once, at recurrent.py:875, where it "
            "does 'beta = beta * 2.0' in eager PyTorch before beta is passed to dispatch_chunk_kda "
            "at :918 as a plain post-sigmoid tensor. No fla flag is threaded, no kernel is "
            "selected, no branch is taken. So KDA_R1's measured costs -- 0.7787x throughput "
            "(326,513 vs 419,288 tok/s, a 22.1% penalty) and +2.539 GiB reserved -- WERE NOT THE "
            "PRICE OF THIS MECHANISM. They were the price of routing it through "
            "KimiDeltaHouseholder, whose own docstring (kda_householder.py:64-70) says it is a "
            "sequential fused-recurrent kernel 'expected to be materially slower than fla's "
            "chunked kernels' because 'the goal of this milestone is mechanism validation, not "
            "throughput'; the run-1 audit localised the +2.539 GiB to that kernel's fp32 "
            "state-history backward workspace, 79% of it a single 2.00 GiB 'hs' tensor. This arm "
            "pays none of that.\n"
            "WHAT THIS BUYS IS FREE OPTIONALITY, NOT FREE QUALITY, and the distinction is the "
            "whole reason the arm is honestly declarable. Run 1 measured KDA_R1 - KDA_BASE = "
            "+0.023575 nats -- WORSE than the reference, and well inside that run's 0.0636 MDE, so "
            "it is not evidence of anything either way (KDA_R1 also had the worst seed sd of any "
            "arm, 0.04516, 4.2x KDA_BASE's and 17.5x KDA_NOACT's). Two published sources agree "
            "there is no CE gain to expect: Grazzi Table 4 (FineWeb 100B, 1.3B, R=1) reports Wiki "
            "ppl 18.54 -> 18.57 and Avg 53.1 -> 52.4, their words 'mixed results'; GDN-2's Table 5 "
            "finds widening the erase range to [0,2] 'gives no consistent gain at this scale' -- "
            "which is exactly why the GDN2 arm here keeps allow_neg_eigval=False. The payoff is "
            "STATE TRACKING: DeltaProduct Table 2 at R=1 takes Parity from 0.233 to 0.982 and the "
            "average from 0.263 to 0.726. So the decision this arm informs is 'can the production "
            "model have the state-tracking regime for free', and a CE null IS a passing result "
            "provided throughput and memory come back at parity. Do not read a CE null here as "
            "'negative eigenvalues do not work'.\n"
            "Parameter-identical to KDA_BASE at -10,080, and that is not a claim about the flag "
            "being cheap -- it is structural: recurrent.py:708 stores it as a plain bool, its only "
            "reader is forward (:875), and num_params (:1209-1260) does not mention it, so no "
            "tensor's name or shape can depend on it at ANY geometry. That is what makes KDA_BASE, "
            "KDA_NOACT and KDA_NEGEIG seed-pairable: identical tensor inventories mean one shared "
            "init RNG stream draws the same shapes in the same order, so it cannot diverge between "
            "them. Run 1 had no such triple -- KDA_R2 widens w_k/w_v/w_b by R, GDN2 adds w_w, and "
            "KDA_GCONV adds pre_scale/post_scale (its depthwise gate draws nothing, so its stream "
            "matches, but its inventory does not).",
        ),
        Core6Arm(
            "KDA_R1",
            "KDA-Householder, R=1",
            "bake-off arity control",
            kda_layers=KDA_LAYERS,
            mixer="kda_householder_r1",
            notes="The Householder operator at R=1, with allow_neg_eigval=True. Isolates ARITY "
            "from the reflection regime: at R=1 this class is documented to have identical "
            "parameters, FLOPs and state_dict to KimiDeltaAttention, so KDA_R1 - KDA_BASE is the "
            "beta-in-(0,2) reflection regime ALONE, at KDA's throughput, and KDA_R2 - KDA_R1 is "
            "the second Householder factor alone. Without this arm the two are confounded and "
            "R2's result cannot be split. Parameter-identical to KDA_BASE at -10,080, which is "
            "itself a check: if this arm's count ever moves, the R=1 equivalence has broken.",
        ),
        Core6Arm(
            "KDA_R2",
            "KDA-Householder, R=2",
            "bake-off treatment (best quality/param)",
            kda_layers=KDA_LAYERS,
            mixer="kda_householder_r2",
            notes="Two Householder factors per token, allow_neg_eigval=True. The best "
            "quality-per-parameter result to date and the reason this bake-off exists. "
            "allow_neg_eigval IS LOAD-BEARING: it puts beta in (0,2) so (I - beta k k^T) can be "
            "a true reflection; at the class default of False beta stays in (0,1), the update is "
            "a contraction, and the mechanism the arm is named for is absent while everything "
            "still trains. The R=2 mixer is 6,608,976 params against KDA's 4,487,248, and "
            "solve_widths respends the 4,233,376 difference in FFN width to land the arm at "
            "+6,304 -- inside tolerance, so quality here is not bought with capacity.",
        ),
        Core6Arm(
            "GDN2",
            "Gated DeltaNet-2",
            "bake-off competitor",
            kda_layers=KDA_LAYERS,
            mixer="gdn2",
            notes="GDN-2, which decouples the erase and write gates that KDA drives from one "
            "scalar beta. NOT PARAM-MATCHED TO ANYTHING BY DEFAULT and its defaults deliberately "
            "invert GatedDeltaNet's (expand_v 1.0 against 2.0, allow_neg_eigval False against "
            "True), so both are written out explicitly in '_gdn2' rather than inherited. At "
            "expand_v=1.0 the mixer is 6,568,016 params; solve_widths respends the surplus and "
            "the arm lands at +22,688, inside tolerance. allow_neg_eigval stays False because "
            "that is the paper's headline model -- its Table 5 finds no consistent gain from "
            "widening the erase gate at 1.3B.",
        ),
    ]
}


def mixer_config(arm: "Core6Arm | str"):
    """
    Build a fresh config for the operator this arm puts in its KDA slots.

    A new instance every call: :func:`build_arm` hands the result to ``dataclasses.replace`` and a
    shared instance would let one arm's block config alias another's.

    :param arm: A :class:`Core6Arm` or the name of one in :data:`ARMS`.

    :returns: The mixer config, e.g. a :class:`KimiDeltaAttentionConfig`.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]
    return MIXERS[arm.mixer]()


def mixer_params(arm: "Core6Arm | str", *, d_model: int = D_MODEL) -> int:
    """
    Parameters in ONE of this arm's KDA-slot mixers, from the config's own ``num_params``.

    Read from the config class rather than hard-coded here, so that a change to a mixer's
    parameterization moves the solver with it instead of silently breaking the anchor.

    :param arm: A :class:`Core6Arm` or the name of one in :data:`ARMS`.
    :param d_model: The model dimensionality.

    :returns: The mixer's parameter count.
    """
    return mixer_config(arm).num_params(d_model)  # type: ignore[attr-defined]


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

    # ...plus whatever the arm's own KDA-slot operator costs over the shipped KDA. This term was
    # ADDED FOR THE BAKE-OFF and it is not cosmetic. The solver used to correct the attention
    # schedule only, which was sufficient while every KDA-slot arm ran the same mixer -- the slot
    # width `KDA_SLOT_SWIGLU_WIDTH` absorbed that single operator's cost by construction. The
    # bake-off breaks that assumption: the R=2 Householder mixer is 6,608,976 parameters and GDN-2
    # is 6,568,016, against KDA's 4,487,248. Uncorrected, KDA_R2 lands +1.085% and GDN2 +1.064%
    # against a declared tolerance of 0.05% -- roughly 21x outside it, and in the direction that
    # hands the bigger operator extra capacity and calls the result mixer quality.
    if arm.kda_layers:
        surplus += len(arm.kda_layers) * (mixer_params(arm) - _BLOCK_PARAMS["kda"])

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
    # than a shared one. The operator comes from the arm's `mixer`, which is "kda" for every arm
    # that predates the bake-off, so those arms are byte-for-byte what they were.
    if arm.kda_layers:
        slot_mixer = mixer_config(arm)
        # The registry builds mixers at the frozen head geometry but cannot know the run's dtype,
        # which `build_arm` takes as an argument. Set it here so a bf16 run does not silently get
        # fp32 mixers in exactly the two slots under test.
        slot_mixer = replace(slot_mixer, dtype=dtype)  # type: ignore[type-var]
        kda_block = replace(
            cfg.block,
            sequence_mixer=slot_mixer,  # type: ignore[arg-type]
            feed_forward=FeedForwardConfig(
                hidden_size=KDA_SLOT_SWIGLU_WIDTH, bias=False, dtype=dtype
            ),
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
