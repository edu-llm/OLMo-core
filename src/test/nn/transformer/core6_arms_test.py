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

import pytest

from olmo_core.nn.attention import AttentionConfig, KimiDeltaAttentionConfig
from olmo_core.nn.attention.short_conv import ShortConvConfig
from olmo_core.nn.transformer.core6_arms import (
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


@pytest.mark.parametrize(
    "name,n_global,n_kda,n_swa,n_liv",
    [
        ("L0", 6, 0, 0, 10),
        ("K2", 6, 2, 0, 8),
        ("G4R0", 4, 0, 0, 12),
        ("G4R2", 4, 2, 0, 10),
        ("S14", 2, 0, 14, 0),
        ("G0R0", 0, 0, 0, 16),
    ],
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
    assert kinds.count("KimiDeltaAttentionConfig") == n_kda

    attn = [b.sequence_mixer for b in blocks if isinstance(b.sequence_mixer, AttentionConfig)]
    windowed = [a for a in attn if a.sliding_window is not None]
    assert len(windowed) == n_swa
    assert len(attn) - len(windowed) == n_global


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
    """
    for name in ("K2", "G4R2"):
        assert not (set(solve_widths(name)) & set(KDA_LAYERS)), name


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
