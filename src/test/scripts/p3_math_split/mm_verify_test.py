"""The verifier has to reject bad proofs, not just accept good ones.

A checker that never fires is worse than no checker: it turns "we measured proof
validity" into "we measured whether the output parses". Same discipline as
src/scripts/train/p3_math_split/make_mutants.py — each test here is a specific way a generated proof can
be wrong, and the verifier must catch every one.

Uses a hand-built miniature Metamath database so it runs in milliseconds with no set.mm
download.

    pytest -v src/test/scripts/p3_math_split/mm_verify_test.py
"""

from __future__ import annotations

from . import load_project_module

_mm_verify = load_project_module("mm_verify")
match_template = _mm_verify.match_template
parse_proof = _mm_verify.parse_proof
verify_proof = _mm_verify.verify_proof


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
                "|- ( ph -> ch )".split(),
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
        "ax-1": ("$a", ("|- ( ph -> ( ps -> ph ) )".split(), [wff("ph"), wff("ps")])),
        "ax-mp": (
            "$a",
            (
                "|- ps".split(),
                [wff("ph"), wff("ps"), ess("mp.1", "|- ph"), ess("mp.2", "|- ( ph -> ps )")],
            ),
        ),
        "dup": (
            "$a",
            (
                "|- ph".split(),
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
    return "\n".join(f"{i + 1:>3}  {lbl:<12} {expr}" for i, (lbl, expr) in enumerate(lines))


# --------------------------------------------------------------- the matcher
def test_match_binds_a_sequence_valued_variable():
    sub = match_template("( ph -> ps )".split(), "( A = B -> C )".split(), {"ph", "ps"})
    assert sub == {"ph": ["A", "=", "B"], "ps": ["C"]}


def test_match_enforces_repeated_variables():
    assert match_template("( ph -> ph )".split(), "( A -> A )".split(), {"ph"}) is not None
    assert match_template("( ph -> ph )".split(), "( A -> B )".split(), {"ph"}) is None


def test_match_rejects_constant_mismatch():
    assert match_template("( ph -> ps )".split(), "( A <-> B )".split(), {"ph", "ps"}) is None


# --------------------------------------------------------------- parsing
def test_parse_reads_well_formed_steps():
    steps = parse_proof(proof(("ax-1", "|- ( a -> ( b -> a ) )"), ("syl", "|- ( a -> c )")))
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
def test_accepts_a_correct_proof():
    """syl applied to two genuinely derived antecedents.

    Also the regression test for syl's free `ps`: a checker that only substitutes
    what the conclusion binds cannot discharge these hypotheses, and syl is the
    most-cited rule in set.mm.
    """
    r = verify_proof(MM_DB, GOOD_PROOF, GOOD_GOAL, FACTS)
    assert r.valid, r.reason
    assert r.goal_reached and r.all_grounded and r.all_instances and r.all_hyps_discharged


def test_accepts_a_valid_proof_that_differs_from_the_gold_trace():
    """Reaching the goal another way must count; otherwise this measures imitation."""
    gold = proof(("ax-1", "|- ( a -> zzz )"), ("syl", GOOD_GOAL))
    r = verify_proof(MM_DB, GOOD_PROOF, GOOD_GOAL, FACTS, gold_target=gold)
    assert r.valid, r.reason
    assert not r.exact_match


def test_accepts_theorem_local_assumptions_as_givens_not_proof_steps():
    """A theorem's used local $e hypotheses seed derivation state."""
    generated = proof(("ax-mp", "|- b"))
    local_assumptions = {
        "th.1": "|- a",
        "th.2": "|- ( a -> b )",
    }

    r = verify_proof(
        MM_DB,
        generated,
        "|- b",
        {"ax-mp": FACTS["ax-mp"]},
        local_assumptions=local_assumptions,
    )

    assert r.valid, r.reason
    assert r.parsed_steps == 1


def test_same_earlier_expression_can_discharge_two_hypotheses_after_reuse_is_omitted():
    expr = "|- ( a -> ( b -> a ) )"
    generated = proof(("ax-1", expr), ("dup", expr))

    r = verify_proof(
        MM_DB,
        generated,
        expr,
        {"ax-1": FACTS["ax-1"], "dup": FACTS["dup"]},
    )

    assert r.valid, r.reason
    assert r.parsed_steps == 2


# --------------------------------------------------------------- rejects
def test_rejects_a_hallucinated_rule():
    text = proof(("ax-1", "|- ( a -> b )"), ("modus_bogus", "|- ( a -> c )"))
    r = verify_proof(MM_DB, text, "|- ( a -> c )", FACTS)
    assert not r.valid
    assert not r.all_grounded


def test_rejects_a_rule_not_in_the_prompt_block():
    """Citing a real set.mm rule that was not supplied is still not a proof here."""
    text = proof(("ax-mp", "|- b"))
    r = verify_proof(MM_DB, text, "|- b", {"ax-1": FACTS["ax-1"]})
    assert not r.valid
    assert not r.all_grounded


def test_rejects_a_step_that_is_not_an_instance_of_its_rule():
    """ax-1's conclusion cannot be ( a <-> b ) under any substitution."""
    text = proof(("ax-1", "|- ( a <-> b )"))
    r = verify_proof(MM_DB, text, "|- ( a <-> b )", FACTS)
    assert not r.valid
    assert not r.all_instances


def test_rejects_an_undischarged_hypothesis():
    """syl needs both antecedents proved earlier. Step 1 is a real ax-1 instance, so
    the missing second hypothesis is the only defect."""
    text = proof(
        ("ax-1", "|- ( a -> ( b -> a ) )"),
        ("syl", "|- ( a -> ( c -> ( b -> a ) ) )"),
    )
    r = verify_proof(MM_DB, text, "|- ( a -> ( c -> ( b -> a ) ) )", FACTS)
    assert not r.valid
    assert r.all_instances, "step 1 should still be a valid instance"
    assert not r.all_hyps_discharged


def test_rejects_a_proof_that_never_reaches_the_goal():
    """Every step is sound; the derivation simply proves something else."""
    r = verify_proof(MM_DB, GOOD_PROOF, "|- ( x -> y )", FACTS)
    assert not r.valid
    assert r.all_instances and r.all_hyps_discharged
    assert not r.goal_reached
    assert "goal" in r.reason


def test_rejects_empty_output():
    r = verify_proof(MM_DB, "", GOOD_GOAL, FACTS)
    assert not r.valid
    assert r.parsed_steps == 0


def test_rejects_the_goal_asserted_without_derivation():
    """The degenerate failure: emit the goal, cite something, derive nothing."""
    text = proof(("syl", GOOD_GOAL))
    r = verify_proof(MM_DB, text, GOOD_GOAL, FACTS)
    assert not r.valid, "asserting the goal as a bare syl step must not count"
    assert r.goal_reached, "it does reach the goal -- that is exactly why this is a trap"
    assert not r.all_hyps_discharged
