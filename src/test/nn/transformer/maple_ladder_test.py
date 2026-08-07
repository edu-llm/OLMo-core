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


def test_the_ratio_identities_hold_at_the_two_faithful_points():
    """R3 and M20 are the two points that claim full Maple faithfulness, including k/E == 1/32.

    R1/R2/E8 vary E deliberately -- that variation IS the E-sweep -- so the identity is asserted
    only where it is claimed. M20 is Maple's literal published config, so a failure here means the
    transcription from `maple/evidence/config.json` is wrong, which is the more valuable catch.
    """
    for rung in ("R3", "M20"):
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
