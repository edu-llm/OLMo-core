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
    return "\n".join(f"{i + 1:>3}  {lbl:<12} {expr}" for i, (lbl, expr) in enumerate(lines))


def tokens(expression):
    return expression.split()


# --------------------------------------------------------------- the matcher
def test_match_binds_a_sequence_valued_variable():
    sub = match_template(
        ["(", "ph", "->", "ps", ")"], ["(", "A", "=", "B", "->", "C", ")"], {"ph", "ps"}
    )
    assert sub == {"ph": ["A", "=", "B"], "ps": ["C"]}


def test_match_enforces_repeated_variables():
    assert (
        match_template(["(", "ph", "->", "ph", ")"], ["(", "A", "->", "A", ")"], {"ph"}) is not None
    )
    assert match_template(["(", "ph", "->", "ph", ")"], ["(", "A", "->", "B", ")"], {"ph"}) is None


def test_match_rejects_constant_mismatch():
    assert (
        match_template(["(", "ph", "->", "ps", ")"], ["(", "A", "<->", "B", ")"], {"ph", "ps"})
        is None
    )


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
    assert r.goal_reached and r.all_grounded and r.all_instances and r.all_hyps_discharged


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
    assert r.goal_reached, "it does reach the goal -- that is exactly why this is a trap"
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
        {"ax-1": "|- bogus", "dup": FACTS["dup"]},
        target_label="dup-target",
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


def test_syntax_budget_is_scoped_to_each_distinct_typing_query(sound_mm):
    """Regression for gold row 7841cb188517 (set:axprOLD).

    That trace exhausted one proof-wide syntax allowance after several individually
    bounded, successful queries. This miniature pair has the same failure mode.
    """
    checker = _mm_verify.SyntaxTypeChecker(
        sound_mm,
        "target",
        sound_mm.assertion_frames["target"],
        node_budget=80,
    )

    first = checker.check("wff", tokens("( a -> ( b -> a ) )"))
    second = checker.check(
        "wff",
        tokens("( ( b -> a ) -> ( c -> ( b -> a ) ) )"),
    )

    assert first.status == "valid", first.reason
    assert second.status == "valid", second.reason


def test_memoized_syntax_search_rejects_ill_typed_mutation(sound_mm):
    """Changing a wff substitution to the class variable A must remain invalid."""
    generated = proof(
        ("ax-1", "|- ( A -> ( b -> A ) )"),
        ("ax-1", "|- ( a -> ( b -> a ) )"),
        ("ax-1", "|- ( ( b -> a ) -> ( c -> ( b -> a ) ) )"),
        ("syl", GOOD_GOAL),
    )

    result = verify_proof(
        sound_mm,
        generated,
        GOOD_GOAL,
        FACTS,
        target_label="target",
    )

    assert result.status == "invalid"
    assert result.steps[0].reason_code == "floating_type_mismatch"


def _rpexpmord_syl221anc_search_inputs():
    """The exact templates and prior expressions from gold row 6479886f2ed5."""
    essential = (
        ("syl3anc.1", tuple(tokens("|- ( ph -> ps )"))),
        ("syl3anc.2", tuple(tokens("|- ( ph -> ch )"))),
        ("syl3anc.3", tuple(tokens("|- ( ph -> th )"))),
        ("syl3Xanc.4", tuple(tokens("|- ( ph -> ta )"))),
        ("syl23anc.5", tuple(tokens("|- ( ph -> et )"))),
        (
            "syl221anc.6",
            tuple(tokens("|- ( ( ( ps /\\ ch ) /\\ ( th /\\ ta ) /\\ et ) -> ze )")),
        ),
    )
    initial_subst = {
        "ph": tokens(r"( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b )"),
        "ze": tokens("( a ^ N ) < ( b ^ N )"),
    }
    derived = [
        expression.split()
        for expression in (
            "|- ( a = b -> ( a ^ N ) = ( b ^ N ) )",
            "|- ( a = A -> ( a ^ N ) = ( A ^ N ) )",
            "|- ( a = B -> ( a ^ N ) = ( B ^ N ) )",
            "|- RR+ C_ RR",
            "|- ( a e. RR+ -> a e. RR )",
            "|- ( N e. NN -> N e. NN0 )",
            r"|- ( ( a e. RR /\ N e. NN0 ) -> ( a ^ N ) e. RR )",
            r"|- ( ( N e. NN /\ a e. RR+ ) -> ( a ^ N ) e. RR )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> a e. RR+ )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> a e. RR )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> b e. RR+ )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> b e. RR )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> 0 <_ a )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> a < b )",
            r"|- ( ( ( N e. NN /\ ( a e. RR+ /\ b e. RR+ ) ) /\ a < b ) -> N e. NN )",
            r"|- ( ( ( a e. RR /\ b e. RR ) /\ ( 0 <_ a /\ a < b ) /\ N e. NN ) -> ( a ^ N ) < ( b ^ N ) )",
        )
    ]
    return essential, initial_subst, {"ph", "ps", "ch", "th", "ta", "et", "ze"}, derived


def test_constraint_first_search_recovers_real_rpexpmord_gold_step():
    """The six syl221anc hypotheses have one sound match inside the default budget."""
    essential, initial_subst, variables, derived = _rpexpmord_syl221anc_search_inputs()

    subst, sources = next(
        _mm_verify._match_essential_hypotheses(
            essential,
            initial_subst,
            variables,
            derived,
            _mm_verify.SearchBudget(_mm_verify.MATCH_NODE_BUDGET),
        )
    )

    assert sources == (9, 11, 12, 13, 14, 15)
    assert subst["ps"] == tokens("a e. RR")
    assert subst["et"] == tokens("N e. NN")


def test_constraint_first_search_rejects_mutated_rpexpmord_premise():
    essential, initial_subst, variables, derived = _rpexpmord_syl221anc_search_inputs()
    derived[-1] = tokens(
        r"|- ( ( ( a e. RR /\ b e. RR ) /\ ( 0 <_ a /\ a < b ) /\ N e. NN )"
        " -> ( b ^ N ) < ( a ^ N ) )"
    )

    matches = _mm_verify._match_essential_hypotheses(
        essential,
        initial_subst,
        variables,
        derived,
        _mm_verify.SearchBudget(_mm_verify.MATCH_NODE_BUDGET),
    )

    assert next(matches, None) is None


def _nested_conjunction_syntax_case(tmp_path, depth=35):
    """Small grammar with the ambiguity shape of set:naddass row 98ef96536475."""
    variables = [f"a{index}" for index in range(depth)]
    expression = variables[0]
    for variable in variables[1:]:
        expression = f"( {expression} /\\ {variable} )"
    source = (
        "$c wff |- ( ) /\\ $.\n"
        f"$v ph ps {' '.join(variables)} $.\n"
        "wph $f wff ph $.\n"
        "wps $f wff ps $.\n"
        + "".join(f"f{index} $f wff {variable} $.\n" for index, variable in enumerate(variables))
        + "wa $a wff ( ph /\\ ps ) $.\n"
        + f"target-rule $a |- {expression} $.\n"
        + f"target $p |- {expression} $= target-rule $.\n"
    )
    path = tmp_path / "nested-conjunction.mm"
    path.write_text(source, encoding="utf-8")
    mm = _mm_expand.MM().parse(path)
    checker = _mm_verify.SyntaxTypeChecker(
        mm,
        "target",
        mm.assertion_frames["target"],
        node_budget=_mm_verify.SYNTAX_NODE_BUDGET,
    )
    return checker, expression


def test_typed_syntax_matching_prunes_real_naddass_ambiguity_shape(tmp_path):
    checker, expression = _nested_conjunction_syntax_case(tmp_path)

    result = checker.check("wff", expression.split())

    assert result.status == "valid", result.reason


def test_typed_syntax_matching_does_not_accept_unbalanced_mutation(tmp_path):
    checker, expression = _nested_conjunction_syntax_case(tmp_path)

    result = checker.check("wff", expression.split()[:-1])

    assert result.status != "valid"


def _operation_equality_case(tmp_path, depth=8):
    """Miniature of bpoly2's long, sequence-valued oveq12d application."""

    def nested(names):
        expression = names[0]
        for name in names[1:]:
            expression = f"( {expression} f {name} )"
        return expression

    groups = [[f"{prefix}{index}" for index in range(depth)] for prefix in "abcd"]
    left_a, left_b, right_a, right_b = map(nested, groups)
    leaves = [name for group in groups for name in group]
    goal = f"|- ( p -> ( {left_a} f {left_b} ) = ( {right_a} f {right_b} ) )"
    premise_a = f"|- ( p -> {left_a} = {right_a} )"
    premise_b = f"|- ( p -> {left_b} = {right_b} )"
    source = (
        "$c wff class |- ( ) -> = $.\n"
        f"$v ph ps A B C D F p f {' '.join(leaves)} $.\n"
        "wph $f wff ph $.\n"
        "wps $f wff ps $.\n"
        "cA $f class A $.\n"
        "cB $f class B $.\n"
        "cC $f class C $.\n"
        "cD $f class D $.\n"
        "cF $f class F $.\n"
        "wp $f wff p $.\n"
        "cf $f class f $.\n"
        + "".join(f"c{index} $f class {variable} $.\n" for index, variable in enumerate(leaves))
        + "co $a class ( A F B ) $.\n"
        + "weq $a wff A = B $.\n"
        + "wi $a wff ( ph -> ps ) $.\n"
        + "emit $a |- ph $.\n"
        + "${\n"
        + "  op-eq.1 $e |- ( ph -> A = C ) $.\n"
        + "  op-eq.2 $e |- ( ph -> B = D ) $.\n"
        + "  op-eq $a |- ( ph -> ( A F B ) = ( C F D ) ) $.\n"
        + "$}\n"
        + f"target-rule $a {goal} $.\n"
        + f"target $p {goal} $= target-rule $.\n"
    )
    path = tmp_path / "operation-equality.mm"
    path.write_text(source, encoding="utf-8")
    mm = _mm_expand.MM().parse(path)
    facts = {
        "emit": "|- ph",
        "op-eq": (
            "|- ( ph -> A = C ) & |- ( ph -> B = D )" " => |- ( ph -> ( A F B ) = ( C F D ) )"
        ),
    }
    return mm, goal, premise_a, premise_b, left_b, facts


def test_typed_application_matching_recovers_real_bpoly2_ambiguity_shape(tmp_path):
    mm, goal, premise_a, premise_b, _, facts = _operation_equality_case(tmp_path)
    generated = proof(
        ("emit", premise_a),
        ("emit", premise_b),
        ("op-eq", goal),
    )

    result = verify_proof(mm, generated, goal, facts, target_label="target")

    assert result.status == "valid", result.reason


def test_typed_application_matching_rejects_mutated_bpoly2_premise(tmp_path):
    mm, goal, premise_a, _, left_b, facts = _operation_equality_case(tmp_path)
    generated = proof(
        ("emit", premise_a),
        ("emit", f"|- ( p -> {left_b} = {left_b} )"),
        ("op-eq", goal),
    )

    result = verify_proof(mm, generated, goal, facts, target_label="target")

    assert result.status == "invalid"
    assert result.steps[-1].reason_code == "essential_hypothesis_unmet"


def _unary_negation_checker(tmp_path):
    """Portable excerpt of set.mm's ``wn $a wff -. ph`` syntax."""
    path = tmp_path / "deep-negation.mm"
    path.write_text(
        "$c wff |- -. $.\n"
        "$v ph p $.\n"
        "wph $f wff ph $.\n"
        "wp $f wff p $.\n"
        "wn $a wff -. ph $.\n"
        "target-rule $a |- p $.\n"
        "target $p |- p $= wp target-rule $.\n",
        encoding="utf-8",
    )
    mm = _mm_expand.MM().parse(path)
    return _mm_verify.SyntaxTypeChecker(
        mm,
        "target",
        mm.assertion_frames["target"],
        node_budget=_mm_verify.SYNTAX_NODE_BUDGET,
    )


def test_deep_legal_negation_returns_unknown_instead_of_recursion_error(tmp_path):
    deep_checker = _unary_negation_checker(tmp_path)

    deep = deep_checker.check("wff", ["-."] * 250 + ["p"])

    assert deep.status == "unknown"
    assert deep.witness == ()

    shallow_checker = _unary_negation_checker(tmp_path)
    shallow = shallow_checker.check("wff", ["-."] * 200 + ["p"])
    assert shallow.status == "valid", shallow.reason


def _mutually_recursive_syntax_checker(tmp_path, *, include_base: bool):
    path = tmp_path / f"mutually-recursive-{include_base}.mm"
    base = "ta-c $a ta c $.\n" if include_base else ""
    path.write_text(
        (
            "$c ta tb |- c $.\n"
            "$v x $.\n"
            "${\n"
            "  bx $f tb x $.\n"
            "  a-from-b $a ta x $.\n"
            "$}\n"
            "${\n"
            "  ax $f ta x $.\n"
            "  b-from-a $a tb x $.\n"
            "$}\n"
            f"{base}"
            "target-rule $a |- c $.\n"
            "target $p |- c $= target-rule $.\n"
        ),
        encoding="utf-8",
    )
    mm = _mm_expand.MM().parse(path)
    return _mm_verify.SyntaxTypeChecker(
        mm,
        "target",
        mm.assertion_frames["target"],
        node_budget=_mm_verify.SYNTAX_NODE_BUDGET,
    )


def test_cycle_pruning_does_not_poison_later_syntax_queries(tmp_path):
    checker = _mutually_recursive_syntax_checker(tmp_path, include_base=True)

    first = checker.check("ta", ["c"])
    after_prior_search = checker.check("tb", ["c"])
    fresh = _mutually_recursive_syntax_checker(tmp_path, include_base=True).check("tb", ["c"])

    assert first.status == "valid", first.reason
    assert fresh.status == "valid", fresh.reason
    assert after_prior_search.status == fresh.status
    assert after_prior_search.witness == ("ta-c", "b-from-a")


def test_verify_proof_tristate_exposes_versioned_public_schema(sound_mm):
    assert _mm_verify.VERIFIER_SCHEMA_VERSION == "p3-metamath-tristate-v1"
    result = _mm_verify.verify_proof_tristate(
        sound_mm,
        GOOD_PROOF,
        GOOD_GOAL,
        FACTS,
        target_label="target",
    )
    assert result.status.value == "valid"
    assert result.valid is True
    assert result.as_dict()["status"] == "valid"
    assert "valid" in result.as_dict()
    assert result.as_dict()["valid"] is True


def test_verify_proof_tristate_preserves_unknown_as_none(sound_mm):
    result = _mm_verify.verify_proof_tristate(
        sound_mm,
        GOOD_PROOF,
        GOOD_GOAL,
        FACTS,
        target_label=None,
    )
    assert result.status.value == "unknown"
    assert result.valid is None
