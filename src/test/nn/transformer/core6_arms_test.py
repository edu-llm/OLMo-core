"""
Tests for the CORE-6 arm builder.

These are ledger tests. They exist so that a change to the geometry, the mixer parameter
formulas, or the width solver cannot silently move an arm off its anchor -- which would
confound the primary contrast without changing any loss curve in a visible way.

Everything here is CPU/meta-device only and must stay that way: KDA cannot be *built* without
``flash-linear-attention``, so parameter counts are taken from the config
(``TransformerConfig.num_params``) rather than from a built module. That is the same number
the model would have; ``test_config_num_params_matches_built_model`` pins the two together
for every arm that can actually be built here.
"""

from dataclasses import replace
from typing import Dict

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, KimiDeltaAttentionConfig
from olmo_core.nn.attention.short_conv import ShortConvConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.core6_arms import (
    ARM_L0_DELTA,
    ARMS,
    ATTENTION_LAYERS,
    D_MODEL,
    HEAD_DIM,
    K2_L0_DELTA,
    KDA_LAYERS,
    KDA_SLOT_SWIGLU_WIDTH,
    L0_PARAM_TARGET,
    N_HEADS,
    N_LAYERS,
    SWA_WINDOW,
    SWIGLU_WIDTH,
    VOCAB_SIZE,
    WIDTH_TOLERANCE,
    Core6Arm,
    build_arm,
    mixer_config,
    mixer_params,
    solve_widths,
)

# --- the frozen ledger ----------------------------------------------------------------------


def test_l0_hits_the_frozen_param_target():
    """The single number that certifies the whole geometry is right."""
    assert build_arm("L0").num_params == L0_PARAM_TARGET == 390_135_552


def test_l0_at_the_legacy_vocab():
    """
    The same geometry at the pre-dolma2 vocabulary. The two ledgers must differ by exactly the
    embedding rows and nothing else, which is what makes them cross-checkable.
    """
    small = build_arm("L0", vocab_size=65536).num_params
    assert small == 354_483_968
    assert L0_PARAM_TARGET - small == (VOCAB_SIZE - 65536) * D_MODEL


def test_k2_anchors_down_to_l0_by_exactly_the_frozen_residual():
    """
    K2 must sit 10,080 parameters *below* L0. This is what removes the need for a separate
    padded control arm, so it is asserted exactly rather than within a tolerance.
    """
    delta = build_arm("K2").num_params - build_arm("L0").num_params
    assert delta == K2_L0_DELTA == -10_080


def test_kda_narrowing_is_what_produces_the_residual():
    """
    The -10,080 is not a coincidence of the solver: it is 2 KDA slots each trading
    (4608 - 4512) of SwiGLU width against the LIV->KDA mixer difference.
    """
    liv = ShortConvConfig(kernel_size=3, bias=False).num_params(D_MODEL)
    kda = KimiDeltaAttentionConfig(n_heads=N_HEADS, head_dim=HEAD_DIM).num_params(D_MODEL)
    ffn_saved = 3 * D_MODEL * (SWIGLU_WIDTH - KDA_SLOT_SWIGLU_WIDTH)
    assert len(KDA_LAYERS) * ((kda - liv) - ffn_saved) == K2_L0_DELTA


def test_the_same_residual_survives_the_ablation():
    """
    G4R2 - G4R0 must equal K2 - L0. The KDA contrast has to be the *same* contrast whether or
    not attention was ablated underneath it, otherwise sigma's numerator and the secondary
    contrast are not measuring the same intervention.
    """
    ablated = build_arm("G4R2").num_params - build_arm("G4R0").num_params
    released = build_arm("K2").num_params - build_arm("L0").num_params
    assert ablated == released == K2_L0_DELTA


def test_every_arm_has_a_declared_delta():
    """
    A new arm cannot land in ``ARMS`` without an exact declared residual.

    The same guard as ``test_every_arm_has_a_declared_topology``, for the ledger. Without it the
    tolerance test below still runs on a new arm -- it parametrizes over ``ARMS`` -- but tolerance
    is a band, and a band is what this module spent seven arms refusing to settle for.
    """
    assert set(ARM_L0_DELTA) == set(ARMS), (
        f"declared but with no expected delta: {sorted(set(ARMS) - set(ARM_L0_DELTA))}; "
        f"expected but not declared: {sorted(set(ARM_L0_DELTA) - set(ARMS))}"
    )


@pytest.mark.parametrize("name", list(ARMS))
def test_every_arm_lands_on_its_declared_delta(name):
    """
    EVERY arm's residual is asserted EXACTLY, not within the tolerance band.

    This is the generalisation of ``test_k2_anchors_down_to_l0_by_exactly_the_frozen_residual``,
    and the bake-off is what forced it. One shared ``K2_L0_DELTA`` worked while every KDA-slot arm
    ran the same operator; six arms running six operators need six numbers. The alternative on the
    table was to loosen the exact assertion to ``abs(delta) <= tolerance``, which would have
    covered the whole bake-off and checked nothing about any individual arm -- a 195,068-parameter
    band that every arm here clears by two orders of magnitude.

    The declared numbers are derived by hand from the config classes' own ``num_params`` algebra.
    A disagreement means the hand arithmetic and the solver have diverged, and the message says by
    how much rather than just that they differ.
    """
    delta = build_arm(name).num_params - L0_PARAM_TARGET
    assert delta == ARM_L0_DELTA[name], (
        f"{name} lands at {delta:+,} against L0 but declares {ARM_L0_DELTA[name]:+,} "
        f"(off by {delta - ARM_L0_DELTA[name]:+,})"
    )


def test_the_solver_is_what_holds_the_big_mixers_on_the_anchor():
    """
    Pins the reason ``solve_widths`` grew a mixer term, so it cannot be removed as dead code.

    KDA_R2 and GDN2 carry mixers roughly 2.1M and 2.08M parameters larger than KDA, in two slots
    each. Without the correction they land about +1.09% and +1.06% against a declared tolerance of
    0.05% -- some 21x outside it, and in the direction that gives the bigger operator extra FFN
    capacity and then reports the result as mixer quality.

    Asserted as "the uncorrected arm would be far out of tolerance", computed from the mixer sizes
    rather than from a second copy of the solver, so this test cannot pass by re-deriving the
    formula it is checking.
    """
    for name in ("KDA_R2", "GDN2"):
        surplus = len(ARMS[name].kda_layers) * (mixer_params(name) - mixer_params("KDA_BASE"))
        assert surplus > 0, name
        # What the arm WOULD deviate by if that surplus were never respent.
        uncorrected = surplus / L0_PARAM_TARGET
        assert uncorrected > 20 * WIDTH_TOLERANCE, (
            f"{name}'s mixer surplus is only {uncorrected:+.4%}, so this test is no longer "
            "demonstrating why the solver's mixer term exists"
        )
        # And what it actually deviates by, once the solver has respent it.
        assert abs(ARM_L0_DELTA[name]) / L0_PARAM_TARGET <= WIDTH_TOLERANCE


def test_r1_householder_is_parameter_identical_to_shipped_kda():
    """
    At R=1 the Householder class is documented to reduce to KimiDeltaAttention exactly.

    That equivalence is what makes KDA_R1 an arity control: if it held only approximately, then
    KDA_R2 - KDA_R1 would mix the second Householder factor with a parameter difference, and
    KDA_R1 - KDA_BASE would not be the reflection regime alone. Checked on the configs' own
    algebra, so it runs without a GPU and fails loudly if either class is reparameterized.
    """
    assert mixer_params("KDA_R1") == mixer_params("KDA_BASE") == 4_487_248
    assert ARM_L0_DELTA["KDA_R1"] == ARM_L0_DELTA["KDA_BASE"] == K2_L0_DELTA


def test_the_gated_conv_arm_pays_only_the_depthwise_price():
    """
    The depthwise gate must cost 6,144 per layer, and the lowrank one must not be what shipped.

    Two things are asserted because the arm's validity rests on both. The gate has to be small
    enough that KDA_GCONV stays parameter-matched -- at 2 * 3 * 2 * hidden it is 6,144 per layer
    and the arm lands at +2,208. And it must not be ``"lowrank"``, which would cost 2,359,296
    (12x the declared tolerance) AND forfeit seed pairing, because its nine nn.Linear per layer
    draw from the global rng before the seeded generator is built.
    """
    cfg = mixer_config("KDA_GCONV")
    assert cfg.gated_conv is True
    assert cfg.gate_structure == "depthwise"
    assert cfg.gate_rank is None

    per_layer = mixer_params("KDA_GCONV") - mixer_params("KDA_BASE")
    assert per_layer == 6_144, per_layer
    assert cfg.gate_params(D_MODEL) == 6_144

    # The counterfactual, so the 12x claim in the arm's notes is checked rather than asserted in
    # prose. A lowrank gate at the same rank the memory estimate uses is far outside tolerance.
    lowrank = replace(cfg, gate_structure="lowrank", gate_rank=256)
    assert (lowrank.num_params(D_MODEL) - mixer_params("KDA_BASE")) == 2_359_296
    assert 2_359_296 / L0_PARAM_TARGET > 10 * WIDTH_TOLERANCE


def test_the_gate_isolating_contrast_exists_and_is_parameter_free():
    """
    KDA_GCONV - KDA_NOACT must be the POST GATE and nothing else.

    The depthwise pre-gate is algebraically a SiLU -- ``2*sigmoid(a*u)*u == (2/a)*silu(a*u)``
    exactly, with ``2/a`` absorbed into the convolution taps -- so ``gated_conv=True,
    activation=None`` is NOT activation-free and KDA_GCONV - KDA_BASE conflates three changes:
    the post gate, a learnable activation slope, and the activation's position. KDA_NOACT exists
    to make the attributable contrast available, and this test is what stops it being deleted as
    a redundant-looking duplicate of KDA_BASE.

    Also asserts that removing the activation costs no parameters, which is what makes NOACT a
    free control rather than a second capacity condition.
    """
    for required in ("KDA_BASE", "KDA_NOACT", "KDA_GCONV"):
        assert required in ARMS, f"{required} is required for the gate-isolating contrast"

    base, noact = mixer_config("KDA_BASE"), mixer_config("KDA_NOACT")
    assert base.conv_activation == "silu" and base.gated_conv is False
    assert noact.conv_activation is None and noact.gated_conv is False
    # An activation is not a parameter, so the control is free.
    assert mixer_params("KDA_NOACT") == mixer_params("KDA_BASE")
    assert ARM_L0_DELTA["KDA_NOACT"] == ARM_L0_DELTA["KDA_BASE"]


def test_both_householder_arms_allow_negative_eigenvalues():
    """
    ``allow_neg_eigval=True`` is the mechanism, and the class default is ``False``.

    With it, ``beta`` reaches (0, 2) and ``(I - beta k k^T)`` can be a true Householder
    reflection. Without it ``beta`` stays in (0, 1), the update is a contraction, and the arm
    trains stably at the same cost while the mechanism it is named for is simply absent. Nothing
    downstream would look wrong. This is the check that the flag was actually passed.
    """
    for name in ("KDA_R1", "KDA_R2"):
        assert mixer_config(name).allow_neg_eigval is True, name
    assert mixer_config("KDA_R1").num_householder == 1
    assert mixer_config("KDA_R2").num_householder == 2


def test_gdn2_pins_the_defaults_that_invert_gdn1():
    """
    GDN-2's defaults deliberately invert GatedDeltaNet's, so this arm writes them down.

    ``expand_v`` 1.0 against GDN1's 2.0 and ``allow_neg_eigval`` False against GDN1's True. At
    ``expand_v=2.0`` the mixer is 10,112,144 parameters rather than 6,568,016 and the widths the
    solver would need are far enough from 4,608 to be a different model, so the value is pinned
    rather than inherited.
    """
    cfg = mixer_config("GDN2")
    assert cfg.expand_v == 1.0
    assert cfg.allow_neg_eigval is False, "the paper's headline model keeps the erase gate in [0,1]"
    assert mixer_params("GDN2") == 6_568_016


def test_every_bakeoff_arm_keeps_the_full_attention_budget():
    """
    All six bake-off arms carry L0's six global-attention layers.

    THE REASON IS MECHANICAL, not aesthetic. ``model.py`` emits RoPE buffers only for
    ``Attention``/``FusedAttention`` blocks, so an arm with fewer attention layers loses positional
    encoding along with them, and its loss gap would be mostly missing position rather than mixer
    quality -- a large and entirely uninterpretable number. Asserted on the built config rather
    than on the declaration, so a solver or override bug that dropped an attention layer is caught
    too.
    """
    bakeoff = [n for n in ARMS if ARMS[n].kda_layers and n not in ("K2", "G4R2")]
    assert len(bakeoff) == 6, f"expected the six bake-off arms, got {sorted(bakeoff)}"
    for name in bakeoff:
        blocks = build_arm(name).resolved_block_configs
        idx = tuple(
            i for i, b in enumerate(blocks) if isinstance(b.sequence_mixer, AttentionConfig)
        )
        assert idx == ATTENTION_LAYERS, f"{name} has global attention at {idx}"


@pytest.mark.parametrize("name", list(ARMS))
def test_every_arm_is_within_the_declared_tolerance(name):
    """
    Every arm must sit within +/-0.05% of the anchor. Without the width solver, G4R0 is
    +0.54% and G0R0 is +1.62% -- large enough that the sigma denominator would confound lost
    retrieval with gained capacity, in the direction that inflates sigma.
    """
    dev = (build_arm(name).num_params - L0_PARAM_TARGET) / L0_PARAM_TARGET
    assert abs(dev) <= WIDTH_TOLERANCE, f"{name} deviates {dev:+.4%}"


def test_removing_attention_would_add_params_without_the_solver():
    """
    Pins the asymmetry the solver exists to correct, so that if the mixer formulas ever change
    the *reason* for the solver is re-checked rather than assumed.
    """
    # Compare whole blocks, not bare mixers: LFM2's attention carries per-head QK norms
    # (2 * head_dim per attention layer) that a default AttentionConfig does not, and the
    # solver works in block units.
    blocks = build_arm("L0").resolved_block_configs
    liv = next(b for b in blocks if isinstance(b.sequence_mixer, ShortConvConfig)).num_params(
        D_MODEL
    )
    gqa = next(b for b in blocks if isinstance(b.sequence_mixer, AttentionConfig)).num_params(
        D_MODEL
    )
    assert liv > gqa, "a LIV block is expected to be larger than a GQA block"
    assert (liv, gqa) == (18_355_200, 17_303_680)
    assert liv - gqa == 1_051_520


# --- topology -------------------------------------------------------------------------------


#: The topology every arm must build to, keyed by arm name: global attention, KDA, sliding
#: window, LIV. Written down rather than derived from the declaration, because a test that
#: recomputes ``len(arm.attention_layers)`` and compares it to the built config's attention
#: count would pass for any consistent pair of wrong numbers -- including the all-attention
#: model the module docstring warns about, which is what a ``.attention`` typo produces.
#:
#: ``test_every_arm_has_a_declared_topology`` is what makes this a ledger rather than a list
#: somebody forgot to extend: G2R0 was added on 2026-08-01 and got its parameter count checked
#: (those tests parametrize over ``list(ARMS)``) while its topology went unchecked for the same
#: reason this dict is guarded now.
ARM_TOPOLOGY = {
    "L0": (6, 0, 0, 10),
    "K2": (6, 2, 0, 8),
    "G4R0": (4, 0, 0, 12),
    "G4R2": (4, 2, 0, 10),
    "G2R0": (2, 0, 0, 14),
    "S14": (2, 0, 14, 0),
    "G0R0": (0, 0, 0, 16),
    # The bake-off arms. Every one is L0's topology with slots {6,11} taken by a linear-attention
    # mixer, so all six rows are identical -- and that is the point being asserted, not a
    # copy-paste: the arms must differ in the OPERATOR and in nothing structural. A row that
    # drifted from (6, 2, 0, 8) would mean an arm quietly changed its attention budget, and its
    # loss gap would then be mostly missing RoPE rather than mixer quality.
    "KDA_BASE": (6, 2, 0, 8),
    "KDA_NOACT": (6, 2, 0, 8),
    "KDA_GCONV": (6, 2, 0, 8),
    "KDA_R1": (6, 2, 0, 8),
    "KDA_R2": (6, 2, 0, 8),
    "GDN2": (6, 2, 0, 8),
}


def test_every_arm_has_a_declared_topology():
    """
    A new arm cannot slip in with its topology unchecked.

    This is the guard that was missing: the parametrize list below used to be six literal rows
    and ``ARMS`` had seven entries, so ``G2R0`` -- the arm added for the dose-response wave --
    had its parameter count asserted and its mixer schedule not. In a module whose own docstring
    says a wrong attribute yields "a model that builds, trains, and answers a different
    question", an unchecked topology is the failure mode, not a gap in coverage.
    """
    assert set(ARM_TOPOLOGY) == set(ARMS), (
        f"declared but untested: {sorted(set(ARMS) - set(ARM_TOPOLOGY))}; "
        f"tested but not declared: {sorted(set(ARM_TOPOLOGY) - set(ARMS))}"
    )


@pytest.mark.parametrize(
    "name,n_global,n_kda,n_swa,n_liv",
    [(name, *counts) for name, counts in ARM_TOPOLOGY.items()],
)
def test_arm_topology(name, n_global, n_kda, n_swa, n_liv):
    """
    Every layer is the mixer the arm declares. The failure this guards is silent: setting
    `.attention` instead of `.sequence_mixer` on a block config creates a new attribute, the
    override is dropped, and the model trains happily as all-attention.
    """
    blocks = build_arm(name).resolved_block_configs
    assert len(blocks) == N_LAYERS

    kinds = [type(b.sequence_mixer).__name__ for b in blocks]
    assert kinds.count("ShortConvConfig") == n_liv

    # The KDA slots are counted against the class the ARM DECLARES, not against
    # `KimiDeltaAttentionConfig` by name. The bake-off puts three different operator classes in
    # those two slots -- KDA, KimiDeltaHouseholder and GatedDeltaNet2 -- so a hard-coded class
    # name would silently count zero for four of the arms and this row would then be asserting
    # that they have no mixer at all, which is the same shape of vacuous pass the module docstring
    # warns about.
    expected_cls = type(mixer_config(name))
    assert kinds.count(expected_cls.__name__) == n_kda

    attn = [b.sequence_mixer for b in blocks if isinstance(b.sequence_mixer, AttentionConfig)]
    windowed = [a for a in attn if a.sliding_window is not None]
    assert len(windowed) == n_swa
    assert len(attn) - len(windowed) == n_global

    # The four counts have to cover every layer. Without this a row could name three of the
    # four correctly and leave a layer of some fourth kind entirely unmentioned.
    assert n_global + n_kda + n_swa + n_liv == N_LAYERS


def test_g2r0_sits_on_s14s_global_indices():
    """
    G2R0 and S14 must differ in ONE thing: whether the other 14 layers are sliding-window
    attention or LIV convolutions. If their global indices ever drift apart the pair stops
    being that comparison and becomes two unrelated arms.
    """
    assert ARMS["G2R0"].attention_layers == ARMS["S14"].attention_layers == (2, 12)

    blocks = build_arm("G2R0").resolved_block_configs
    idx = tuple(i for i, b in enumerate(blocks) if isinstance(b.sequence_mixer, AttentionConfig))
    assert idx == (2, 12)
    # And none of them windowed -- G2R0 is a DOSE point at a=2, so its two attention layers
    # must be global. A sliding window here would make it a second S14 wearing G2R0's name.
    assert all(
        blocks[i].sequence_mixer.sliding_window is None for i in idx
    ), "G2R0's global layers must not be windowed"


def test_l0_attention_sits_at_the_released_indices():
    blocks = build_arm("L0").resolved_block_configs
    idx = tuple(i for i, b in enumerate(blocks) if isinstance(b.sequence_mixer, AttentionConfig))
    assert idx == ATTENTION_LAYERS == (2, 5, 8, 10, 12, 14)


def test_kda_lands_in_the_declared_slots():
    for name in ("K2", "G4R2"):
        blocks = build_arm(name).resolved_block_configs
        idx = tuple(
            i
            for i, b in enumerate(blocks)
            if isinstance(b.sequence_mixer, KimiDeltaAttentionConfig)
        )
        assert idx == KDA_LAYERS, name


@pytest.mark.parametrize("name", [n for n in ARMS if ARMS[n].kda_layers])
def test_every_mixer_arm_places_its_operator_in_the_frozen_slots(name):
    """
    Every arm with a mixer puts it in {6, 11} and puts the RIGHT CLASS there.

    Two failures in one check, because they are indistinguishable in a loss curve. Placing the
    operator in different slots per arm would make the bake-off a comparison of positions as much
    as of operators; placing the wrong class there -- which a mis-keyed `MIXERS` entry does
    silently, since every value is a zero-argument callable returning *some* valid config -- gives
    two arms that are secretly the same model and a guaranteed null between them.
    """
    arm = ARMS[name]
    expected_cls = type(mixer_config(name))
    blocks = build_arm(name).resolved_block_configs
    idx = tuple(i for i, b in enumerate(blocks) if isinstance(b.sequence_mixer, expected_cls))
    assert idx == arm.kda_layers == KDA_LAYERS, name


def test_g0r0_has_no_attention_at_all():
    """The instrument anchor must be structurally incapable of long-range retrieval."""
    blocks = build_arm("G0R0").resolved_block_configs
    assert not any(isinstance(b.sequence_mixer, AttentionConfig) for b in blocks)


def test_s14_keeps_exactly_two_global_layers():
    """
    Guards a live trap: SlidingWindowAttentionConfig forces full attention on the first and
    last layer by default, which would silently give S14 four global layers instead of two and
    destroy its role as a free a=2 dose point.
    """
    blocks = build_arm("S14").resolved_block_configs
    attn = [
        (i, b.sequence_mixer)
        for i, b in enumerate(blocks)
        if isinstance(b.sequence_mixer, AttentionConfig)
    ]
    full = [i for i, a in attn if a.sliding_window is None]
    assert full == [2, 12], f"expected 2 global layers, got {full}"

    for i, a in attn:
        if a.sliding_window is not None:
            assert a.sliding_window._get_window_size(i, N_LAYERS) == SWA_WINDOW


def test_s14_window_stays_below_the_slice_gap():
    """
    The evaluation slice is defined at gap > 1024. If the window ever grew past the gap, the
    sliding-window layers could see the referent and S14 would stop being an a=2 dose point.
    """
    assert SWA_WINDOW <= 1024


# --- the width solver -------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(ARMS))
def test_solved_widths_are_multiples_of_32(name):
    """A width off the /32 grid lands on a bad GEMM tile."""
    for layer, w in solve_widths(name).items():
        assert w % 32 == 0, f"{name} layer {layer} width {w}"
        assert w > 0


def test_l0_needs_no_solving():
    assert solve_widths("L0") == {}


def test_kda_slots_are_never_touched_by_the_solver():
    """
    The KDA slots carry the frozen -10,080 residual. If the solver moved their width, that
    residual would drift and the capacity match with L0 would be lost.

    Covers EVERY arm with a mixer, not just K2 and G4R2. The bake-off arms make this sharper
    rather than merely broader: the solver now respends a mixer surplus of millions of parameters
    for KDA_R2 and GDN2, and if it were allowed to spend any of that inside the slot it was
    correcting for, the two slots would no longer be the same FFN width across arms -- so the
    contrast would be operator plus width instead of operator.
    """
    for name in (n for n in ARMS if ARMS[n].kda_layers):
        assert not (set(solve_widths(name)) & set(KDA_LAYERS)), name
        widths = build_arm(name).resolved_block_configs
        for i in KDA_LAYERS:
            assert widths[i].feed_forward.hidden_size == KDA_SLOT_SWIGLU_WIDTH, name


def test_solver_reduces_width_for_attention_ablated_arms():
    """Removing attention adds params, so the correction must be downward."""
    for name in ("G4R0", "G0R0"):
        widths = solve_widths(name)
        assert widths, name
        assert all(w < SWIGLU_WIDTH for w in widths.values()), name


# --- declaration-time guards --------------------------------------------------------------


def test_a_layer_claimed_twice_is_rejected():
    with pytest.raises(ValueError, match="two mixers"):
        Core6Arm("bad", "Bad", "test", attention_layers=(2, 5), kda_layers=(5,))


def test_an_out_of_range_layer_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        Core6Arm("bad", "Bad", "test", attention_layers=(99,))


def test_liv_layers_are_the_complement():
    for arm in ARMS.values():
        claimed = set(arm.attention_layers) | set(arm.kda_layers) | set(arm.swa_layers)
        assert set(arm.liv_layers) == set(range(N_LAYERS)) - claimed


# --- consistency between the config ledger and a real model ---------------------------------


@pytest.mark.parametrize("name", ["L0", "G4R0", "S14", "G0R0"])
def test_config_num_params_matches_built_model(name):
    """
    The ledger is computed from config; the experiment trains a module. Pin them together for
    every arm that can be built without a GPU. K2/G4R2 are excluded only because KimiDelta-
    Attention asserts `has_fla()` at construction -- they are covered on FarmShare.
    """
    cfg = build_arm(name)
    model = cfg.build(init_device="meta")
    assert sum(p.numel() for p in model.parameters()) == cfg.num_params


def test_embeddings_are_tied():
    """
    Untied embeddings would add 102,760,448 parameters -- 26% of the model -- and every
    arm-matching decision would be silently wrong.
    """
    assert build_arm("L0").tie_word_embeddings


@pytest.mark.parametrize("name", list(ARMS))
def test_arms_are_declared_with_a_role_and_a_reason(name):
    """
    Each arm has to say what it is for. Dropping an arm that another arm's note names as its
    numerator or denominator should look obviously wrong in review.
    """
    arm = ARMS[name]
    assert arm.title and arm.role and arm.notes


def test_the_sigma_pair_is_present():
    """sigma_2 = (G4R2 - G4R0) / (G4R0 - L0): all three arms must exist."""
    for required in ("L0", "G4R0", "G4R2"):
        assert required in ARMS


# --- the init seed reaches the tensors ------------------------------------------------------


def _tiny(seed: int) -> TransformerConfig:
    """L0's block wiring at a size a laptop can actually initialise, at a given ``init_seed``.

    One layer, vocab 128, SwiGLU 32: about 3.4M parameters and under a second to draw, against
    390M and minutes at the real geometry. The GEOMETRY is not what these tests are about --
    that is asserted exactly by the ledger tests above. What is under test is whether the seed
    argument reaches the generator that draws the tensors, and one layer exercises the same
    ``init_weights`` path as sixteen.
    """
    cfg = build_arm("L0", vocab_size=128, init_seed=seed)
    block = replace(cfg.block, feed_forward=FeedForwardConfig(hidden_size=32, bias=False))
    return replace(cfg, n_layers=1, block=block, block_overrides=None)


def _drawn_weights(seed: int) -> Dict[str, torch.Tensor]:
    model = _tiny(seed).build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _is_constant(tensor: torch.Tensor) -> bool:
    """Whether every element is the same value.

    The norm gains are initialised to ones and no seed can change that, so they are identical
    across seeds for a reason that has nothing to do with the bug under test. They are
    identified by BEING CONSTANT rather than by having "norm" in their name: a name filter
    would also excuse a real weight matrix that happened to be called something similar, and it
    would go stale the moment a module is renamed. This test is about which tensors a random
    draw touched, so "was this drawn at all" is the right question to ask of each one.
    """
    return bool(torch.all(tensor == tensor.flatten()[0]))


def test_init_seed_reaches_the_config():
    """The first link in the chain: ``build_arm(**kwargs)`` -> ``TransformerConfig.init_seed``.

    Cheap and separate from the tensor test below so that a break in the kwargs plumbing is
    distinguishable from a break in ``init_weights``.
    """
    assert build_arm("L0", init_seed=7).init_seed == 7
    assert build_arm("L0", init_seed=0).init_seed == 0
    # The default, which is what every arm got while `.edullm/train_core6_arm.py` was not
    # passing the flag through -- including runs whose JSON reported init_seed 12536.
    assert build_arm("L0").init_seed == 0


def test_two_init_seeds_give_different_weights():
    """
    THE REGRESSION TEST FOR A FLAG THAT DID NOTHING. ``.edullm/train_core6_arm.py`` accepted
    ``--init-seed`` and did not pass it to ``build_arm``, so every value produced BIT-IDENTICAL
    weights while the run summary printed the distinct number it was given. That is worse than
    having no flag: the JSON asserted a varied initialisation that was never varied, and any
    seed-noise interval built from those runs is a data-order interval wearing the wrong label.

    ``seed_all()`` is deliberately NOT what this checks. That seeds the global rngs;
    ``Transformer.init_weights`` builds its own ``torch.Generator(device).manual_seed(
    self.init_seed)`` and hands it to every draw, so the global rng is not what the weights
    come from and seeding it proves nothing about this.

    Asserted on EVERY randomly-drawn parameter rather than on one: a fix that threaded the seed
    into the embeddings and not the blocks would pass a spot check and still leave most of the
    model identical across "seeds". The constant-valued norm gains are excluded because no seed
    can change a tensor of ones -- and the exclusion is counted and asserted below, so it cannot
    quietly grow to cover the whole model.
    """
    a = _drawn_weights(0)
    b = _drawn_weights(1)
    assert set(a) == set(b) and a, "the two models must have the same parameters to compare"

    drawn = [name for name in a if not _is_constant(a[name])]
    assert len(drawn) >= 8, (
        f"only {len(drawn)} of {len(a)} tensors look randomly drawn, so this test would be "
        "asserting almost nothing; the model or its init changed"
    )

    identical = [name for name in drawn if torch.equal(a[name], b[name])]
    assert not identical, (
        f"{len(identical)} of {len(drawn)} randomly-drawn tensors are bit-identical across "
        f"init_seed 0 and 1, so the seed is not reaching the generator that draws them: "
        f"{identical[:5]}"
    )


def test_the_same_init_seed_gives_identical_weights():
    """
    The other half, and the half that makes the test above mean something. "Different weights"
    is also what a seed that is ignored in favour of fresh entropy produces -- that would pass
    the difference test and destroy reproducibility, which is the property the seed exists for.
    Bit-identical, not close: a reproducible draw is exact.
    """
    a = _drawn_weights(3)
    b = _drawn_weights(3)
    assert set(a) == set(b) and a

    for name in a:
        assert torch.equal(a[name], b[name]), f"{name} differs between two draws at init_seed 3"


def test_drawn_weights_have_the_declared_scale():
    """
    A magnitude check on the draw itself, because "different" and "identical" are both
    satisfiable by a model whose weights were never initialised at all -- ``to_empty`` leaves
    uninitialised memory, which differs between two runs for reasons that have nothing to do
    with the seed. The block matrices are drawn at ``init_std``, so their sample standard
    deviation has to land near it.

    Embeddings are excluded: ``embedding_init_std`` may differ, and the tied LM head shares
    that storage.
    """
    weights = _drawn_weights(0)
    std = _tiny(0).init_std
    checked = 0
    for name, tensor in weights.items():
        if "embeddings" not in name and tensor.dim() == 2 and tensor.numel() > 1024:
            observed = float(tensor.std())
            assert 0.2 * std < observed < 5 * std, f"{name}: std {observed:.4g} vs {std:.4g}"
            checked += 1
    assert checked >= 4, f"only {checked} matrices were checked, so this asserts almost nothing"
