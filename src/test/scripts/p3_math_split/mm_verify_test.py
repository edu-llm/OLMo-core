"""The verifier has to reject bad proofs, not just accept good ones.

A checker that never fires is worse than no checker: it turns "we measured proof
validity" into "we measured whether the output parses". Same discipline as
src/scripts/train/p3_math_split/make_mutants.py — each test here is a specific
way a generated proof can be wrong, and the verifier must catch every one.

Uses a hand-built miniature Metamath database so it runs in milliseconds with no set.mm
download.

    pytest -v src/test/scripts/p3_math_split/mm_verify_test.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import load_project_module

_mm_verify = load_project_module("mm_verify")
match_template = _mm_verify.match_template
parse_proof = _mm_verify.parse_proof
verify_proof = _mm_verify.verify_proof
_mm_expand = load_project_module("mm_expand")


class FakeMM:
    """Minimal stand-in for src/scripts/train/p3_math_split/mm_expand.MM.

    Only `.labels` is read by the verifier: label -> (kind, (conclusion, mand, ...))
    with mand entries ('$f', label, (typecode, var)) or ('$e', label, expr).
    """

    def __init__(self, labels):
        self.labels = labels


def wff(var):
    return ("$f", f"w{var}", ("wff", var))


def ess(name, expr):
    return ("$e", name, expr.split())


# ph, ps, ch are wff metavariables, exactly as in set.mm.
MM_DB = FakeMM(
    {
        # syl: from ( ph -> ps ) and ( ps -> ch ), infer ( ph -> ch ).
        # 'ps' appears only in the hypotheses -- the case that breaks a naive checker.
        "syl": (
            "$a",
            (
                ["|-", "(", "ph", "->", "ch", ")"],
                [
                    wff("ph"),
                    wff("ps"),
                    wff("ch"),
                    ess("syl.1", "|- ( ph -> ps )"),
                    ess("syl.2", "|- ( ps -> ch )"),
                ],
            ),
        ),
        # ax-1: |- ( ph -> ( ps -> ph ) ), no hypotheses.
        "ax-1": (
            "$a",
            (
                ["|-", "(", "ph", "->", "(", "ps", "->", "ph", ")", ")"],
                [wff("ph"), wff("ps")],
            ),
        ),
        "ax-mp": (
            "$a",
            (
                ["|-", "ps"],
                [
                    wff("ph"),
                    wff("ps"),
                    ess("mp.1", "|- ph"),
                    ess("mp.2", "|- ( ph -> ps )"),
                ],
            ),
        ),
        "dup": (
            "$a",
            (
                ["|-", "ph"],
                [wff("ph"), ess("dup.1", "|- ph"), ess("dup.2", "|- ph")],
            ),
        ),
        "notalogicalrule": ("$f", ("wff", "ph")),
    }
)

FACTS = {
    "syl": "|- ( ph -> ps ) & |- ( ps -> ch ) => |- ( ph -> ch )",
    "ax-1": "|- ( ph -> ( ps -> ph ) )",
    "ax-mp": "|- ph & |- ( ph -> ps ) => |- ps",
    "dup": "|- ph & |- ph => |- ph",
}


def proof(*lines):
    return "\n".join(
        f"{i + 1:>3}  {lbl:<12} {expr}" for i, (lbl, expr) in enumerate(lines)
    )


# --------------------------------------------------------------- the matcher
def test_match_binds_a_sequence_valued_variable():
    sub = match_template(
        ["(", "ph", "->", "ps", ")"], ["(", "A", "=", "B", "->", "C", ")"], {"ph", "ps"}
    )
    assert sub == {"ph": ["A", "=", "B"], "ps": ["C"]}


def test_match_enforces_repeated_variables():
    assert (
        match_template(["(", "ph", "->", "ph", ")"], ["(", "A", "->", "A", ")"], {"ph"})
        is not None
    )
    assert (
        match_template(["(", "ph", "->", "ph", ")"], ["(", "A", "->", "B", ")"], {"ph"})
        is None
    )


def test_match_rejects_constant_mismatch():
    assert (
        match_template(
            ["(", "ph", "->", "ps", ")"], ["(", "A", "<->", "B", ")"], {"ph", "ps"}
        )
        is None
    )


# --------------------------------------------------------------- parsing
def test_parse_reads_well_formed_steps():
    steps = parse_proof(
        proof(("ax-1", "|- ( a -> ( b -> a ) )"), ("syl", "|- ( a -> c )"))
    )
    assert steps == [("ax-1", "|- ( a -> ( b -> a ) )"), ("syl", "|- ( a -> c )")]


def test_parse_stops_at_garbage():
    assert parse_proof("I think the answer is probably ( ph -> ps )") == []


# A genuinely valid two-step derivation, used by several tests below.
#   1  ax-1   ph:=a, ps:=b            |- ( a -> ( b -> a ) )
#   2  ax-1   ph:=( b -> a ), ps:=c   |- ( ( b -> a ) -> ( c -> ( b -> a ) ) )
#   3  syl    ph:=a, ps:=( b -> a ), ch:=( c -> ( b -> a ) )
# Note ps is bound only by the hypotheses, never by syl's conclusion.
GOOD_GOAL = "|- ( a -> ( c -> ( b -> a ) ) )"
GOOD_PROOF = proof(
    ("ax-1", "|- ( a -> ( b -> a ) )"),
    ("ax-1", "|- ( ( b -> a ) -> ( c -> ( b -> a ) ) )"),
    ("syl", GOOD_GOAL),
)


# --------------------------------------------------------------- accepts
def test_accepts_a_correct_proof(sound_mm):
    """syl applied to two genuinely derived antecedents.

    Also the regression test for syl's free `ps`: a checker that only substitutes
    what the conclusion binds cannot discharge these hypotheses, and syl is the
    most-cited rule in set.mm.
    """
    r = verify_proof(
        sound_mm,
        GOOD_PROOF,
        GOOD_GOAL,
        FACTS,
        target_label="target",
    )
    assert r.valid, r.reason
    assert (
        r.goal_reached and r.all_grounded and r.all_instances and r.all_hyps_discharged
    )


def test_accepts_a_valid_proof_that_differs_from_the_gold_trace(sound_mm):
    """Reaching the goal another way must count; otherwise this measures imitation."""
    gold = proof(("ax-1", "|- ( a -> zzz )"), ("syl", GOOD_GOAL))
    r = verify_proof(
        sound_mm,
        GOOD_PROOF,
        GOOD_GOAL,
        FACTS,
        gold_target=gold,
        target_label="target",
    )
    assert r.valid, r.reason
    assert not r.exact_match


def test_accepts_theorem_local_assumptions_as_givens_not_proof_steps(sound_mm):
    """A theorem's used local $e hypotheses seed derivation state."""
    generated = proof(("ax-mp", "|- b"))
    local_assumptions = {
        "th.1": "|- a",
        "th.2": "|- ( a -> b )",
    }

    r = verify_proof(
        sound_mm,
        generated,
        "|- b",
        {"ax-mp": FACTS["ax-mp"]},
        local_assumptions=local_assumptions,
        target_label="mp-target",
    )

    assert r.valid, r.reason
    assert r.parsed_steps == 1


def test_same_earlier_expression_can_discharge_two_hypotheses_after_reuse_is_omitted(
    sound_mm,
):
    expr = "|- ( a -> ( b -> a ) )"
    generated = proof(("ax-1", expr), ("dup", expr))

    r = verify_proof(
        sound_mm,
        generated,
        expr,
        {"ax-1": FACTS["ax-1"], "dup": FACTS["dup"]},
        target_label="dup-target",
    )

    assert r.valid, r.reason
    assert r.parsed_steps == 2


def test_accepts_qualified_rule_for_matching_target_database(sound_mm):
    result = verify_proof(
        sound_mm,
        proof(("iset:emit", "|- b")),
        "|- b",
        {"iset:emit": "|- ph"},
        target_label="iset:b-target",
    )

    assert result.status == "valid", result.reason
    assert result.steps[0].label == "iset:emit"


def test_rejects_qualified_rule_for_wrong_target_database(sound_mm):
    result = verify_proof(
        sound_mm,
        proof(("set:emit", "|- b")),
        "|- b",
        {"set:emit": "|- ph"},
        target_label="iset:b-target",
    )

    assert result.status == "invalid"
    assert result.reason_code == "rule_database_mismatch"
    assert result.steps[0].reason_code == "rule_database_mismatch"


# --------------------------------------------------------------- rejects
def test_rejects_a_hallucinated_rule(sound_mm):
    text = proof(("ax-1", "|- ( a -> b )"), ("modus_bogus", "|- ( a -> c )"))
    r = verify_proof(
        sound_mm,
        text,
        "|- ( a -> c )",
        FACTS,
        target_label="ac-target",
    )
    assert not r.valid
    assert not r.all_grounded


def test_rejects_a_rule_not_in_the_prompt_block(sound_mm):
    """Citing a real set.mm rule that was not supplied is still not a proof here."""
    text = proof(("ax-mp", "|- b"))
    r = verify_proof(
        sound_mm,
        text,
        "|- b",
        {"ax-1": FACTS["ax-1"]},
        target_label="b-target",
    )
    assert not r.valid
    assert not r.all_grounded


def test_rejects_a_visible_label_with_the_wrong_statement(sound_mm):
    facts = dict(FACTS)
    facts["syl"] = "|- bogus"

    result = verify_proof(
        sound_mm,
        GOOD_PROOF,
        GOOD_GOAL,
        facts,
        target_label="target",
    )

    assert result.status == "unknown"
    assert result.reason_code == "visible_rule_mismatch"


def test_rejects_a_step_that_is_not_an_instance_of_its_rule(sound_mm):
    """ax-1's conclusion cannot be ( a <-> b ) under any substitution."""
    text = proof(("ax-1", "|- ( a <-> b )"))
    r = verify_proof(
        sound_mm,
        text,
        "|- ( a <-> b )",
        FACTS,
        target_label="bicond-target",
    )
    assert not r.valid
    assert not r.all_instances


def test_rejects_an_undischarged_hypothesis(sound_mm):
    """syl needs both antecedents proved earlier. Step 1 is a real ax-1 instance, so
    the missing second hypothesis is the only defect."""
    text = proof(
        ("ax-1", "|- ( a -> ( b -> a ) )"),
        ("syl", "|- ( a -> ( c -> ( b -> a ) ) )"),
    )
    r = verify_proof(
        sound_mm,
        text,
        "|- ( a -> ( c -> ( b -> a ) ) )",
        FACTS,
        target_label="target",
    )
    assert not r.valid
    assert r.all_instances, "step 1 should still be a valid instance"
    assert not r.all_hyps_discharged


def test_rejects_a_proof_that_never_reaches_the_goal(sound_mm):
    """Every step is sound; the derivation simply proves something else."""
    r = verify_proof(
        sound_mm,
        GOOD_PROOF,
        "|- ( X -> Z )",
        FACTS,
        target_label="xz-target",
    )
    assert not r.valid
    assert r.all_instances and r.all_hyps_discharged
    assert not r.goal_reached
    assert "goal" in r.reason


def test_rejects_empty_output(sound_mm):
    r = verify_proof(
        sound_mm,
        "",
        GOOD_GOAL,
        FACTS,
        target_label="target",
    )
    assert not r.valid
    assert r.parsed_steps == 0


def test_rejects_the_goal_asserted_without_derivation(sound_mm):
    """The degenerate failure: emit the goal, cite something, derive nothing."""
    text = proof(("syl", GOOD_GOAL))
    r = verify_proof(
        sound_mm,
        text,
        GOOD_GOAL,
        FACTS,
        target_label="target",
    )
    assert not r.valid, "asserting the goal as a bare syl step must not count"
    assert (
        r.goal_reached
    ), "it does reach the goal -- that is exactly why this is a trap"
    assert not r.all_hyps_discharged


# -------------------------------------------------------- sound tri-state engine
SOUNDNESS_FIXTURE = Path(__file__).parent / "fixtures" / "mm_verify_soundness.mm"


@pytest.fixture(scope="module")
def sound_mm():
    return _mm_expand.MM().parse(SOUNDNESS_FIXTURE)


def test_rejects_class_substitution_for_wff_floating_hypothesis(sound_mm):
    generated = proof(("emit", "|- A"))

    result = verify_proof(
        sound_mm,
        generated,
        "|- A",
        {"emit": "|- ph"},
        target_label="class-target",
    )

    assert result.status == "invalid"
    assert result.steps[0].reason_code == "floating_type_mismatch"


def test_rebinding_counterexample_is_rejected_and_correct_premise_is_accepted(
    sound_mm,
):
    generated = proof(("syl", "|- ( ps -> Z )"))
    facts = {"syl": FACTS["syl"]}

    bad = verify_proof(
        sound_mm,
        generated,
        "|- ( ps -> Z )",
        facts,
        target_label="bad-rebind",
        local_assumptions={
            "bad-rebind.1": "|- ( X -> X )",
            "bad-rebind.2": "|- ( X -> Z )",
        },
    )
    good = verify_proof(
        sound_mm,
        generated,
        "|- ( ps -> Z )",
        facts,
        target_label="good-rebind",
        local_assumptions={
            "good-rebind.1": "|- ( ps -> X )",
            "good-rebind.2": "|- ( X -> Z )",
        },
    )

    assert bad.status == "invalid"
    assert bad.steps[0].reason_code == "essential_hypothesis_unmet"
    assert good.status == "valid", good.reason


def test_hypothesis_search_budget_exhaustion_is_unknown(sound_mm):
    result = verify_proof(
        sound_mm,
        proof(("syl", "|- ( ps -> Z )")),
        "|- ( ps -> Z )",
        {"syl": FACTS["syl"]},
        target_label="good-rebind",
        local_assumptions={
            "good-rebind.1": "|- ( ps -> X )",
            "good-rebind.2": "|- ( X -> Z )",
        },
        match_node_budget=1,
    )

    assert result.status == "unknown"
    assert result.reason_code == "match_budget_exceeded"
    assert result.valid is None


def test_later_hypothesis_dependency_on_unknown_step_stays_unknown(sound_mm):
    expression = "|- ( a -> ( b -> a ) )"
    result = verify_proof(
        sound_mm,
        proof(("ax-1", expression), ("dup", expression)),
        expression,
        {"ax-1": FACTS["ax-1"], "dup": FACTS["dup"]},
        target_label="dup-target",
        syntax_node_budget=1,
    )

    assert result.status == "unknown"
    assert result.steps[0].status == "unknown"
    assert result.steps[1].reason_code == "depends_on_unknown_step"


def test_generated_step_enforces_disjoint_variable_conditions(sound_mm):
    invalid = verify_proof(
        sound_mm,
        proof(("pair", "|- P z z")),
        "|- P z z",
        {"pair": "|- P x y"},
        target_label="bad-d-target",
    )
    valid = verify_proof(
        sound_mm,
        proof(("pair", "|- P z w")),
        "|- P z w",
        {"pair": "|- P x y"},
        target_label="good-d-target",
    )

    assert invalid.status == "invalid"
    assert invalid.steps[0].reason_code == "disjoint_variable_violation"
    assert valid.status == "valid", valid.reason


@pytest.mark.parametrize(
    "local_assumptions",
    [
        {"invented": "|- a"},
        {"local-target.1": "|- b"},
    ],
)
def test_local_assumption_must_match_target_theorem_frame(sound_mm, local_assumptions):
    result = verify_proof(
        sound_mm,
        proof(("from-one", "|- a")),
        "|- a",
        {"from-one": "|- ph => |- ph"},
        target_label="local-target",
        local_assumptions=local_assumptions,
    )

    assert result.status == "invalid"
    assert result.reason_code == "local_assumption_not_in_target_frame"


def test_missing_target_context_is_explicitly_unknown(sound_mm):
    result = verify_proof(
        sound_mm,
        proof(("emit", "|- a")),
        "|- a",
        {"emit": "|- ph"},
    )

    assert result.status == "unknown"
    assert result.reason_code == "target_context_required"
    assert result.valid is None
