"""Does the Maple ladder's expected-params table agree with the closed form, at every rung?

**Why this test exists.** The ladder's parameter counts have been wrong three times, and **all
three errors entered through hand-transcription** (D-012, D-014). The remedy adopted was "read the
numbers from code, never copy them" -- but `MAPLE_EXPECTED_PARAMS` *is* the code, and a typo in it
is still a typo. Nothing until now checked the table against anything.

**What this checks, and what it deliberately does not.** It cross-checks the table against the
independent closed form recorded in `maple/agents/contracts/ladder-and-factory.md`:

    total = 2dV + d + L*(2d*n_h*h_d + 2d*n_kv*h_d + 2h_d + dE + 3d*f_e*E + 2d)

That is a *second* derivation, not the config tree, so agreement rules out a transcription error in
either place. It does **not** build a model, so it cannot rule out the two of them sharing a wrong
premise about what the factory constructs -- which D-076 established is a real failure mode: two
derivations agreed there and were both wrong. **The thing that closes that gap is the ~$1.43 CPU
dry-run printing `PARAM_LEDGER` off a built config** (`maple/agents/lanes/P-M20/STATUS.md`), and
this test is not a substitute for it.

The value it does add is that for R0-R3 the table holds **measured** figures, so the closed form is
validated against reality at four points; M20's row is derived. A closed form that reproduces four
measured rows and then disagrees with the fifth would localize the error to the fifth.

Runs on CPU and touches no tensor: `num_params` is arithmetic over dataclass fields.
"""

from typing import Dict

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import TransformerConfig

VOCAB = 100352

# The per-rung knobs the closed form needs that `MAPLE_RUNGS` does not carry, because the factory
# derives them: f_e = d/4, head_dim = 128, k = 8. Written as a formula, not a table, so this cannot
# drift from `maple_scaled`'s own defaults without the ratio assertions firing.
HEAD_DIM = 128
TOP_K = 8


def closed_form_total(*, d: int, L: int, E: int, n_h: int, n_kv: int, V: int) -> int:
    """The ratified closed form. Kept as one expression to stay comparable to the contract.

    The trailing ``+ d`` is the **LM head's own final RMSNorm** (`lm_head.py:94-103`). Omitting it
    was the third of the three transcription errors and cost exactly ``d`` per rung -- 1,024 at
    R1-R3, which is far too small to notice by eye and far too large to be rounding.
    """
    f_e = d // 4
    return (
        2 * d * V
        + d
        + L
        * (
            2 * d * n_h * HEAD_DIM  # q and o projections
            + 2 * d * n_kv * HEAD_DIM  # k and v projections
            + 2 * HEAD_DIM  # per-head QK-norm: q_norm + k_norm, head_dim each
            + d * E  # router
            + 3 * d * f_e * E  # experts: gate, up, down
            + 2 * d  # the block's two RMSNorms
        )
    )


def _spec(rung: str) -> Dict[str, int]:
    return TransformerConfig.MAPLE_RUNGS[rung]


@pytest.mark.parametrize("rung", sorted(TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB]))
def test_expected_total_matches_the_closed_form(rung: str):
    spec = _spec(rung)
    expected_total, _ = TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB][rung]
    computed = closed_form_total(
        d=spec["d_model"],
        L=spec["n_layers"],
        E=spec["num_experts"],
        n_h=spec["n_heads"],
        n_kv=spec["n_kv_heads"],
        V=VOCAB,
    )
    assert computed == expected_total, (
        f"{rung}: MAPLE_EXPECTED_PARAMS says {expected_total:,} but the ratified closed form "
        f"gives {computed:,} (delta {computed - expected_total:+,}). One of the two is a "
        f"transcription error. A delta of exactly d={spec['d_model']} is the LM head's final "
        f"RMSNorm; a delta of exactly 2*head_dim*L is the per-head QK-norm; both have happened."
    )


@pytest.mark.parametrize(
    "rung", sorted(TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[VOCAB])
)
def test_active_minus_routers_is_active_minus_exactly_the_routers(rung: str):
    """The two tables must be consistent with each other by construction, not coincidence.

    Routers are ``L * d * E``. Every token traverses all of them, so they are active by
    definition -- which is the D-014 correction: plain active params are NOT constant across the
    E-sweep, active-minus-routers is. If these two tables ever disagree about the router term, the
    E-sweep's throughput attribution is being made against the wrong denominator.
    """
    spec = _spec(rung)
    _, expected_active = TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB][rung]
    expected_amr = TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[VOCAB][rung]
    routers = spec["n_layers"] * spec["d_model"] * spec["num_experts"]
    assert expected_active - routers == expected_amr, (
        f"{rung}: active {expected_active:,} minus routers {routers:,} is "
        f"{expected_active - routers:,}, but MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS says "
        f"{expected_amr:,}"
    )


def test_active_minus_routers_is_invariant_across_the_e_sweep():
    """R1/R2/R3 differ only in E, so this quantity must be *identical* -- not merely close.

    This is what makes FLOPs/token constant across the sweep and therefore what makes a measured
    throughput delta attributable to kernel and routing overhead rather than to arithmetic. A
    tolerance here would admit exactly the drift it exists to forbid.

    **M20 is excluded on purpose.** It has a different d and L entirely, so no invariance is
    claimed between it and the sweep; asserting one would be asserting a coincidence.
    """
    table = TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[VOCAB]
    sweep = {rung: table[rung] for rung in ("R1", "R2", "R3")}
    assert len(set(sweep.values())) == 1, f"active-minus-routers is not invariant: {sweep}"


@pytest.mark.parametrize("rung", sorted(TransformerConfig.MAPLE_RUNGS))
def test_every_rung_satisfies_the_geometry_assertion(rung: str):
    """``n_heads * head_dim == d_model`` at EVERY rung.

    This is the assertion that catches D-012: a ladder whose rows were computed at different
    attention widths reproduces each of its own totals row by row and is still wrong. Only
    checking one identity against every row finds it. Maple itself is 1.0x (16*128 == 2048).
    """
    spec = _spec(rung)
    assert spec["n_heads"] * HEAD_DIM == spec["d_model"], (
        f"{rung}: attention width {spec['n_heads']}*{HEAD_DIM} = "
        f"{spec['n_heads'] * HEAD_DIM} != d_model {spec['d_model']} "
        f"({spec['n_heads'] * HEAD_DIM / spec['d_model']:.2f}x)"
    )


@pytest.mark.parametrize("rung", sorted(TransformerConfig.MAPLE_RUNGS))
def test_every_rung_satisfies_the_swa_period(rung: str):
    """``L % 4 == 0``: the 3:1 SWA pattern has period 4, so anything else truncates the last
    cycle and puts the global layers somewhere nobody chose."""
    assert _spec(rung)["n_layers"] % 4 == 0, f"{rung}: L={_spec(rung)['n_layers']}"


def test_the_ratio_identities_hold_at_the_faithful_points():
    """R3, M20 and M7B are the points that claim full Maple faithfulness, including k/E == 1/32.

    R1/R2/E8 vary E deliberately -- that variation IS the E-sweep -- so the identity is asserted
    only where it is claimed. M20 is Maple's literal published config, so a failure there means the
    transcription from `maple/evidence/config.json` is wrong, which is the more valuable catch.
    M7B claims the identity too: E=256 is *forced* by k/E = 1/32 once k=8 is forced by
    k*f_e/d = 2.0, so it is a derivation to check rather than a choice to record.
    """
    for rung in ("R3", "M20", "M7B"):
        spec = _spec(rung)
        d = spec["d_model"]
        f_e = d // 4
        assert f_e * 4 == d, f"{rung}: f_e/d != 1/4"
        assert TOP_K * f_e == 2 * d, f"{rung}: k*f_e/d != 2.0"
        assert TOP_K * 32 == spec["num_experts"], f"{rung}: k/E != 1/32"
        assert spec["n_heads"] == 4 * spec["n_kv_heads"], f"{rung}: GQA is not 4:1"


def test_m20_is_maple_preview_field_for_field():
    """M20 must equal DeepGrove's published config, which is the point of it.

    Transcribed from `maple/evidence/config.json`: `hidden_size` 2048, `num_hidden_layers` 24,
    `num_experts` 256, `num_attention_heads` 16, `num_key_value_heads` 4. Asserted as literals
    here **on purpose** -- this is the one place a literal is right, because the thing being
    checked is a transcription from an external artifact and there is nothing else to derive it
    from. Everywhere else in this file the numbers come from formulas for exactly that reason.
    """
    assert _spec("M20") == dict(
        d_model=2048, n_layers=24, num_experts=256, n_heads=16, n_kv_heads=4
    )


def test_m20_differs_from_maple_only_by_the_vocabulary():
    """The 20.00B-vs-20.2B gap must be **entirely** the embedding pair, or something else moved.

    Maple is V=151,936 and we are V=100,352 (padded dolma2, a frozen decision). Untied
    embeddings means the vocabulary enters as ``2*d*V``, so the whole difference should be
    ``2*d*(151936 - 100352)`` and nothing else. If this fails, M20's geometry has drifted from
    Maple's and the "same shape, different vocab" claim is false -- which would make every
    published comparison to Maple's headline numbers misleading.
    """
    spec = _spec("M20")
    ours, _ = TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB]["M20"]
    theirs = closed_form_total(
        d=spec["d_model"],
        L=spec["n_layers"],
        E=spec["num_experts"],
        n_h=spec["n_heads"],
        n_kv=spec["n_kv_heads"],
        V=151_936,
    )
    assert theirs - ours == 2 * spec["d_model"] * (151_936 - VOCAB)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "E8 (X2's low anchor) sits in MAPLE_RUNGS with no row in MAPLE_EXPECTED_PARAMS, so it "
        "builds with its total/active ledger assertions SKIPPED rather than passed. Marked xfail "
        "STRICT rather than deleted: it records the gap as a known defect instead of as silence, "
        "and the moment someone measures E8 and files its row this test starts PASSING, which "
        "under strict xfail is itself a failure telling them to remove this marker. A plain "
        "assertion here would break CI for every lane; a non-strict xfail would let the gap be "
        "closed and forgotten. NOT P-M20's to fix -- E8 needs a measurement, not arithmetic."
    ),
)
def test_every_rung_in_the_ladder_has_ratified_param_figures():
    """A rung with no row in the expected-params table SKIPS its ledger assertions.

    `_maple_assert_ladder` only `log.warning`s in that case (config.py, the `expected is None`
    branch), and a skipped assertion that announces nothing is indistinguishable from a passing
    one. The whole point of the ladder's assertion machinery is that an un-gated rung must not be
    launchable by accident.
    """
    missing = sorted(
        set(TransformerConfig.MAPLE_RUNGS) - set(TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB])
    )
    assert not missing, (
        f"rungs with no ratified param figures at V={VOCAB}: {missing}. These build with their "
        f"total/active ledger assertions SKIPPED, not passed."
    )


# =====================================================================================
# M7B -- the frozen shape for the 7B full-pretraining PoC
#
# Shape FROZEN 2026-08-10; `maple/plan/m7b-shape.md` is the authority. **It is frozen, so these
# tests exist to detect a change, not to permit one.** If one of them fails, the question is what
# moved the shape -- not which integer to update.
# =====================================================================================

M7B_ASPECT = 96.0  # d/L. Maple's is 2048/24 = 85.33; the +12.5% break is deliberate.


def test_m7b_is_the_frozen_shape_field_for_field():
    """M7B must be exactly the shape frozen on 2026-08-10, and nothing near it.

    Asserted as literals, like `test_m20_is_maple_preview_field_for_field` and for the same reason:
    the thing under test is a transcription from an external artifact (`maple/plan/m7b-shape.md`),
    so there is nothing else to derive it from. Everywhere else in this file the numbers come from
    formulas precisely because derivable things should be derived.
    """
    assert _spec("M7B") == dict(
        d_model=1536, n_layers=16, num_experts=256, n_heads=12, n_kv_heads=3
    )


def test_m7b_n_kv_is_three_and_that_is_intentional():
    """``n_kv_heads == 3`` -- the ladder's first odd KV count, and it must not be "fixed" to 4.

    It is what GQA 4:1 *forces* at n_heads=12, so changing it to an even number breaks the ratio
    identity rather than tidying anything. It is verified inert rather than assumed so: no
    divisibility, evenness or power-of-2 constraint on `n_kv_heads` exists anywhere under
    `src/olmo_core/`, `w_k` is a plain `Linear(d_model, 3*128)`, `n_rep = 12/3 = 4` is exact, and
    tensor parallelism -- the one place an odd n_kv would bite -- is never configured, since the
    launcher builds `TransformerDataParallelConfig(fsdp)` and FSDP2 shards flattened dim 0 (384),
    never the head count.
    """
    spec = _spec("M7B")
    assert spec["n_kv_heads"] == 3, (
        f"n_kv_heads is {spec['n_kv_heads']}, expected 3. If this was changed to an even number "
        f"for tidiness, note that GQA 4:1 at n_heads={spec['n_heads']} REQUIRES 3, so this "
        f"'fix' breaks a ratio identity."
    )
    assert spec["n_heads"] == 4 * spec["n_kv_heads"], "GQA must be exactly 4:1"
    assert spec["n_heads"] % spec["n_kv_heads"] == 0, "n_rep must be an exact integer"


def test_m7b_aspect_ratio_break_is_the_only_one_and_is_pinned():
    """M7B breaks exactly one Maple ratio -- ``d/L`` -- and the size of the break is pinned.

    Recorded as an assertion rather than as prose so that a later "improvement" that quietly moves
    d or L toward Maple's 85.33 fails here instead of silently changing the funded shape. The
    +12.5% is structural: no admissible shape totals exactly 7.00B, and among those that clear it
    M7B is the smallest at a Maple-like aspect.
    """
    spec = _spec("M7B")
    aspect = spec["d_model"] / spec["n_layers"]
    assert aspect == M7B_ASPECT, f"d/L moved from {M7B_ASPECT} to {aspect}"
    maple = _spec("M20")
    maple_aspect = maple["d_model"] / maple["n_layers"]
    assert aspect / maple_aspect == pytest.approx(1.125, rel=1e-9), (
        f"the aspect break is {100 * (aspect / maple_aspect - 1):+.2f}%, expected exactly +12.5%. "
        f"This is the one ratio M7B breaks; a different value means the shape moved."
    )


def test_the_specs_modular_unreachability_proof_is_false_so_nobody_rederives_from_it():
    """``maple/plan/m7b-shape.md`` proves 7.00B unreachable via mod 1024, and **that proof is wrong.**

    The spec argues: every admissible total is ``0 mod 1024`` while ``7e9 = 512 * 13,671,875`` has
    an odd cofactor, so 7.00B cannot be hit. But the closed form's trailing ``+ d`` -- the LM head's
    own final RMSNorm -- makes the residue depend on the parity of ``d/512``, so admissible totals
    occupy ``{0, 512} mod 1024``, and ``7e9 mod 1024 == 512`` lies in that set. **The conclusion
    survives** (exhaustive enumeration finds no exact hit) **but the proof does not.**

    This is pinned as a test because the failure mode is someone re-deriving a *different* shape
    from the broken argument and trusting it. The five M7B parameter integers are unaffected --
    they were confirmed exactly against the closed form -- so this is a defect in justification
    prose, not in the ledger.
    """
    assert 7_000_000_000 % 1024 == 512, "the premise of this test changed"

    residues = set()
    for d in range(512, 4097, 512):
        n_h = d // 128
        if n_h % 4:  # GQA 4:1 needs an integer n_kv
            continue
        for L in range(4, 41, 4):  # the 3:1 SWA period
            t = closed_form_total(d=d, L=L, E=256, n_h=n_h, n_kv=n_h // 4, V=VOCAB)
            residues.add(t % 1024)
    assert residues == {0, 512}, (
        f"admissible totals occupy {sorted(residues)} mod 1024. The spec claims {{0}}, which is "
        f"what makes its unreachability proof invalid. If this is now genuinely {{0}}, the closed "
        f"form lost its trailing `+ d` -- which is transcription error #3 and costs d per rung."
    )
    assert 7_000_000_000 % 1024 in residues, (
        "7e9's residue is NOT achievable, which would make the spec's modular proof valid after "
        "all. Re-check the closed form before trusting this test's premise."
    )


def test_m7b_closed_form_is_sensitive_to_every_term_so_the_ledger_test_is_not_vacuous():
    """The closed form must actually *depend* on the geometry, or agreeing with the table proves nothing.

    Modelled on `test_cv_excess_is_still_window_dependent_so_the_test_above_is_not_vacuous`: a test
    whose pass does not depend on the behaviour is not a test. `test_expected_total_matches_the_closed_form`
    would pass trivially if the closed form were insensitive to the very fields most likely to be
    mistyped, so each perturbation here is pinned to its **exact** expected delta -- the signature
    of the specific historical error it corresponds to.
    """
    spec = _spec("M7B")
    d, L, E = spec["d_model"], spec["n_layers"], spec["num_experts"]
    n_h, n_kv = spec["n_heads"], spec["n_kv_heads"]
    base = closed_form_total(d=d, L=L, E=E, n_h=n_h, n_kv=n_kv, V=VOCAB)

    assert base == 7_656_756_736, f"M7B closed-form total is {base:,}, expected 7,656,756,736"

    # n_kv 3 -> 4 ("tidying" the odd KV count) moves the total by exactly L * 2*d*head_dim.
    delta_kv = closed_form_total(d=d, L=L, E=E, n_h=n_h, n_kv=4, V=VOCAB) - base
    assert delta_kv == L * 2 * d * HEAD_DIM == 6_291_456, (
        f"changing n_kv from 3 to 4 moved the total by {delta_kv:+,}, expected "
        f"{L * 2 * d * HEAD_DIM:+,}. If this is 0 the closed form ignores n_kv entirely."
    )

    # The 2.0x-geometry error (D-012): n_heads doubled at fixed d.
    delta_geom = closed_form_total(d=d, L=L, E=E, n_h=2 * n_h, n_kv=n_kv, V=VOCAB) - base
    assert delta_geom == L * 2 * d * n_h * HEAD_DIM > 0, (
        f"doubling n_heads moved the total by {delta_geom:+,}; a 0 here means the closed form is "
        f"blind to attention width, which is exactly the D-012 error it exists to catch."
    )

    # The E axis must move the total but NOT the active-minus-routers invariant.
    delta_e = closed_form_total(d=d, L=L, E=2 * E, n_h=n_h, n_kv=n_kv, V=VOCAB) - base
    assert delta_e == L * (d * E + 3 * d * (d // 4) * E) > 0, f"doubling E moved {delta_e:+,}"

    # Vocab enters as 2dV exactly, because embeddings are untied.
    delta_v = closed_form_total(d=d, L=L, E=E, n_h=n_h, n_kv=n_kv, V=VOCAB + 1) - base
    assert delta_v == 2 * d, (
        f"one extra vocab item moved the total by {delta_v:+,}, expected 2*d = {2 * d:+,}. A delta "
        f"of d would mean the embeddings are being counted as TIED."
    )


def test_m7b_ledger_is_internally_consistent_against_the_closed_form():
    """All five frozen integers, cross-checked against the closed form and against each other.

    The five come from `MAPLE_EXPECTED_PARAMS` / `MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS` -- **read
    from code, never copied** -- and are checked here against a second, independent derivation.
    Every one of this ladder's parameter errors entered through hand-transcription, so agreement
    between two derivations is the cheapest available guard.

    **What this does NOT do is build a model**, so it cannot rule out both derivations sharing a
    wrong premise about what the factory constructs -- D-076's failure mode. The thing that closes
    that is a CPU dry-run printing `PARAM_LEDGER` off a built config. This is not a substitute.
    """
    spec = _spec("M7B")
    d, L, E = spec["d_model"], spec["n_layers"], spec["num_experts"]
    n_h, n_kv = spec["n_heads"], spec["n_kv_heads"]
    f_e = d // 4

    total, active = TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB]["M7B"]
    amr = TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[VOCAB]["M7B"]

    # 1. total
    assert total == closed_form_total(d=d, L=L, E=E, n_h=n_h, n_kv=n_kv, V=VOCAB) == 7_656_756_736

    # 2. active -- the expert term becomes 3*d*f_e*k, since only top_k experts fire per token.
    attn_and_norms = 2 * d * n_h * HEAD_DIM + 2 * d * n_kv * HEAD_DIM + 2 * HEAD_DIM + 2 * d
    computed_active = 2 * d * VOCAB + d + L * (attn_and_norms + d * E + 3 * d * f_e * TOP_K)
    assert active == 635_491_840, f"active is {active:,}, frozen value is 635,491,840"
    assert active == computed_active, (
        f"the table says active is {active:,} but the closed form gives {computed_active:,} "
        f"(delta {computed_active - active:+,})"
    )

    # 3. active - embeddings. NOTE this is minus **2dV**, i.e. BOTH untied tables, which is what
    #    the spec's "active - embeddings (- 2dV)" row means. It is NOT
    #    `num_active_non_embedding_params`, which subtracts only ONE d*V. Confusing the two is a
    #    2x error on the embedding term and would silently reprice tokens-per-active-param.
    assert active - 2 * d * VOCAB == 327_210_496

    # 4. active - routers. Routers are L*d*E and every token traverses all of them.
    routers = L * d * E
    assert routers == 6_291_456
    assert active - routers == amr == 629_200_384

    # 5. per-block numel, the quantity the memory model is built on.
    per_block = attn_and_norms + d * E + 3 * d * f_e * E
    assert per_block == 459_279_616, f"per-block numel is {per_block:,}, frozen at 459,279,616"
    assert total == 2 * d * VOCAB + d + L * per_block, "the block term must compose the total"


def test_m7b_factory_builds_exactly_the_frozen_ledger():
    """**The one test here that consults the factory rather than the tables.**

    Everything above is arithmetic over `MAPLE_RUNGS`, so all of it would still pass if
    `maple_m7b` built the wrong model. This closes that: it constructs the config through the
    public factory -- the same `getattr(TransformerConfig, ...)` path the platform dispatches --
    and reads the ledger back off the built object. Cheap enough to keep in CI because
    `num_params` is arithmetic over dataclass fields and allocates no tensors.

    It also proves the factory's own assertions were EXERCISED and passed, not bypassed:
    `_maple_assert_ladder` raises on any ratio or ledger violation, so reaching the asserts below
    at all means the geometry check `n_heads*head_dim == d_model` fired and held.
    """
    config = TransformerConfig.maple_m7b(vocab_size=VOCAB)

    spec = _spec("M7B")
    d, L = spec["d_model"], spec["n_layers"]
    expected_total, expected_active = TransformerConfig.MAPLE_EXPECTED_PARAMS[VOCAB]["M7B"]

    assert config.d_model == d
    assert config.n_layers == L
    assert config.vocab_size == VOCAB
    assert not config.tie_word_embeddings, "M7B is untied, like Maple"

    # EXACT, not within 1%. The factory's own ledger check uses a 1% band because the published
    # table is rounded; here both sides are integers we control, so a band would admit drift.
    assert config.num_params == expected_total, (
        f"the factory built {config.num_params:,} params but MAPLE_EXPECTED_PARAMS says "
        f"{expected_total:,} (delta {config.num_params - expected_total:+,}). A delta of exactly "
        f"d={d} is the LM head's final RMSNorm; {2 * HEAD_DIM * L} is the per-head QK-norm."
    )
    assert config.num_active_params == expected_active, (
        f"the factory built {config.num_active_params:,} active params, table says "
        f"{expected_active:,} (delta {config.num_active_params - expected_active:+,})"
    )
    assert config.num_active_params - 2 * d * VOCAB == 327_210_496
    assert (
        config.num_active_params - L * d * spec["num_experts"]
        == TransformerConfig.MAPLE_EXPECTED_ACTIVE_MINUS_ROUTERS[VOCAB]["M7B"]
    )

    # Per-block numel, read off the built block rather than recomputed.
    block = config.block
    assert not isinstance(block, dict)
    built_block_numel = block.num_params(d)
    assert built_block_numel == 459_279_616, (
        f"per-block numel is {built_block_numel:,}, frozen at 459,279,616 "
        f"(delta {built_block_numel - 459_279_616:+,})"
    )

    # The knobs whose defaults are known-broken must have been set. Re-read off the built config,
    # so this checks what was constructed and not what the factory meant to construct.
    moe = block.feed_forward_moe
    assert moe is not None, "M7B is an MoE model; a dense block means the factory regressed"
    assert moe.num_experts == 256
    assert moe.hidden_size == d // 4 == 384, "f_e must be d/4"
    assert moe.router.top_k == TOP_K, "top_k defaults to 1 in this tree -- a silent top-1 model"
    assert moe.router.normalize_expert_weights == 1.0, (
        "normalize_expert_weights must be 1.0 (Maple's `norm_topk_prob`); unset, gate mass "
        "measures 0.161 against 1.000 -- a 6.2x error that trains happily"
    )
    assert moe.router.bias_gamma is None, "Maple has no expert bias"
    assert moe.capacity_factor == 2.0, "D-009: cf=2.0 is the funded path; dropless is descoped"
    assert moe.shared_mlp is None, "Maple has zero shared experts"


def test_m7b_factory_refuses_a_broken_geometry():
    """The geometry assertion must **raise**, or every check above it is decoration.

    `n_heads*head_dim == d_model` is the assertion that caught the 2.0x-width error (D-012), and
    an assertion nobody has watched fail is an assumption. This drives the factory off the frozen
    shape and requires a refusal -- if it builds happily, `_maple_assert_ladder` is unreachable
    from `maple_m7b` and the ledger checks in this file are the only thing standing between a
    mistyped override and a $3,000 run of the wrong model.
    """
    with pytest.raises(OLMoConfigurationError, match="attention width"):
        # 24 heads at d=1536 is the 2.0x geometry: n_h*h_d = 3072 != 1536.
        TransformerConfig.maple_m7b(vocab_size=VOCAB, n_heads=24, n_kv_heads=6)

    with pytest.raises(OLMoConfigurationError, match="GQA"):
        # 4:1 GQA violated. n_kv=4 at n_heads=12 is the plausible "fix the odd number" mistake.
        TransformerConfig.maple_m7b(vocab_size=VOCAB, n_kv_heads=4)

    with pytest.raises(OLMoConfigurationError, match="SWA"):
        # L % 4 != 0 truncates the 3:1 SWA cycle and puts globals somewhere nobody chose.
        TransformerConfig.maple_m7b(vocab_size=VOCAB, n_layers=14)
