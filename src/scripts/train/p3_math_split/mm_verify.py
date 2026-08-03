"""Check whether a generated Metamath proof is actually a proof.

String-matching against the gold trace would measure imitation, not reasoning; a model
can reach a correct theorem by a different route and that should count. So each step is
checked for what it claims to be: an application of a named Metamath rule.

What is checked, per generated step `N  label  |- expr`:

  grounded     `label` was supplied in the prompt's fact block. Citing a rule the
               model was not given is not a proof in this setup, whatever it derived.
  instance     `expr` is a substitution instance of `label`'s conclusion. Recovered by
               matching the rule's template against `expr`, which also pins the
               substitution used.
  hypotheses   every `$e` hypothesis of `label`, under that same substitution, appears
               as a theorem-local assumption or an earlier step in the same proof.
  goal         the final step's expression is the goal.

A proof is `valid` when all four hold for every step.

Known limitation, stated because it bounds the claim: the corpus target lines show only
`|-` steps, so the `$f` (variable-typing) steps a full Metamath verifier consumes are
not present in the model's output and cannot be re-derived from it. This checker
therefore validates the logical skeleton — every step is a legitimate instance of a
supplied rule with its hypotheses discharged — but does not re-run the typing
discipline. It accepts a strict superset of what `metamath verify proof *` accepts.
It is not fooled by a wrong expression, a hallucinated rule, an undischarged
hypothesis, or a proof that never reaches the goal.

Sequence matching over token strings can branch, so the matcher has a node budget.
Steps that exhaust it are reported as `unknown` and counted separately — they are never
silently scored as correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

MATCH_NODE_BUDGET = 200_000

# `  1  syl          |- ( ph -> ch )` — the shape build_corpus.py emits.
STEP_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\|-\s.*)$")


class MatchBudgetExceeded(Exception):
    pass


def norm(expr: Sequence[str] | str) -> str:
    return " ".join(expr.split()) if isinstance(expr, str) else " ".join(expr)


def match_template(
    template: Sequence[str],
    concrete: Sequence[str],
    variables: set,
) -> Optional[Dict[str, List[str]]]:
    """Find a substitution making `template` equal `concrete`, or None.

    Metamath variables stand for token *sequences*, not single tokens, so this is
    sequence matching with backtracking. Constants in the template anchor it and
    repeated variables prune it, which keeps real set.mm expressions cheap.

    Raises MatchBudgetExceeded rather than running away on a pathological case.
    """
    budget = [MATCH_NODE_BUDGET]

    def rec(ti: int, ci: int, subst: Dict[str, List[str]]) -> Optional[Dict[str, List[str]]]:
        budget[0] -= 1
        if budget[0] <= 0:
            raise MatchBudgetExceeded()

        if ti == len(template):
            return dict(subst) if ci == len(concrete) else None

        tok = template[ti]

        if tok not in variables:
            if ci < len(concrete) and concrete[ci] == tok:
                return rec(ti + 1, ci + 1, subst)
            return None

        if tok in subst:  # already bound: deterministic
            bound = subst[tok]
            end = ci + len(bound)
            if list(concrete[ci:end]) == bound:
                return rec(ti + 1, end, subst)
            return None

        # Unbound variable. A variable matches a non-empty sequence in set.mm.
        # Leave enough tokens for the remaining constants in the template.
        min_rest = sum(1 for t in template[ti + 1 :] if t not in variables)
        for end in range(ci + 1, len(concrete) - min_rest + 1):
            subst[tok] = list(concrete[ci:end])
            got = rec(ti + 1, end, subst)
            if got is not None:
                return got
            del subst[tok]
        return None

    return rec(0, 0, {})


def apply_subst(expr: Sequence[str], subst: Dict[str, List[str]]) -> List[str]:
    out: List[str] = []
    for tok in expr:
        out.extend(subst[tok]) if tok in subst else out.append(tok)
    return out


def discharge_hypotheses(
    hyp_templates: List[List[str]],
    subst: Dict[str, List[str]],
    variables: set,
    derived: List[List[str]],
) -> bool:
    """Can every `$e` hypothesis be matched against something already proved?

    Matching the conclusion does not bind every variable. `syl` is the canonical case:
    hypotheses `( ph -> ps )` and `( ps -> ch )`, conclusion `( ph -> ch )` — the
    conclusion pins `ph` and `ch` but says nothing about `ps`, whose value has to come
    from the earlier steps. `syl` is the most-cited rule in set.mm, so treating those
    variables as unbound constants would fail nearly every genuine proof.

    So this is a small backtracking search: bind free variables by matching hypotheses
    against derived expressions, consistently across all of them.
    """
    if not hyp_templates:
        return True

    derived_norm = [list(d) for d in derived]

    def rec(k: int, sub: Dict[str, List[str]]) -> bool:
        if k == len(hyp_templates):
            return True
        want = apply_subst(hyp_templates[k], sub)
        if not any(t in variables for t in want):
            # Fully ground: a plain lookup.
            target = norm(want)
            if any(norm(d) == target for d in derived_norm):
                return rec(k + 1, sub)
            return False
        for cand in derived_norm:
            try:
                extra = match_template(want, cand, variables)
            except MatchBudgetExceeded:
                continue
            if extra is None:
                continue
            merged = dict(sub)
            merged.update(extra)
            if rec(k + 1, merged):
                return True
        return False

    return rec(0, dict(subst))


def rule_parts(mm, label: str) -> Optional[Tuple[List[str], List[List[str]], set]]:
    """(conclusion template, $e hypothesis templates, substitutable variables).

    Variables come from the rule's own `$f` mandatory hypotheses rather than from
    `mm.is_var`, which depends on the parser's scope stack having been left in the
    right state. These are exactly the variables the rule may substitute.
    """
    entry = mm.labels.get(label)
    if entry is None:
        return None
    kind, data = entry
    if kind not in ("$a", "$p") or not data or not data[0]:
        return None
    conclusion = list(data[0])
    mand = data[1] if len(data) > 1 else []
    hyps = [list(h[2]) for h in mand if h[0] == "$e"]
    variables = {h[2][1] for h in mand if h[0] == "$f"}
    return conclusion, hyps, variables


@dataclass
class StepResult:
    index: int
    label: str
    expr: str
    grounded: bool
    is_instance: bool
    hyps_discharged: bool
    unknown: bool = False
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.grounded and self.is_instance and self.hyps_discharged and not self.unknown


@dataclass
class ProofResult:
    parsed_steps: int = 0
    valid: bool = False
    goal_reached: bool = False
    all_grounded: bool = False
    all_instances: bool = False
    all_hyps_discharged: bool = False
    any_unknown: bool = False
    exact_match: bool = False
    steps: List[StepResult] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "parsed_steps": self.parsed_steps,
            "valid": self.valid,
            "goal_reached": self.goal_reached,
            "all_grounded": self.all_grounded,
            "all_instances": self.all_instances,
            "all_hyps_discharged": self.all_hyps_discharged,
            "any_unknown": self.any_unknown,
            "exact_match": self.exact_match,
            "reason": self.reason,
        }


def parse_proof(text: str) -> List[Tuple[str, str]]:
    """Pull `(label, expression)` pairs out of generated text.

    Stops at the first line that does not look like a step, so trailing chatter after a
    complete proof does not invalidate it, but interleaved garbage does.
    """
    steps: List[Tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            if steps:
                break
            continue
        m = STEP_RE.match(line)
        if not m:
            break
        steps.append((m.group(2), norm(m.group(3))))
    return steps


def verify_proof(
    mm,
    generated: str,
    goal: str,
    fact_block: Dict[str, str],
    gold_target: Optional[str] = None,
    local_assumptions: Optional[Dict[str, str]] = None,
) -> ProofResult:
    """Verify one generated proof.

    ``fact_block`` contains globally named rules. ``local_assumptions`` contains
    theorem-local $e givens, which seed the derived-expression list but are not
    themselves model-visible proof steps.
    """
    res = ProofResult()
    steps = parse_proof(generated)
    res.parsed_steps = len(steps)

    if gold_target is not None:
        res.exact_match = norm(generated.strip()) == norm(gold_target.strip())

    if not steps:
        res.reason = "no parsable proof steps"
        return res

    derived: List[List[str]] = [
        expr.split() for expr in (local_assumptions or {}).values()
    ]
    all_grounded = all_instances = all_hyps = True

    for i, (label, expr_s) in enumerate(steps, 1):
        expr = expr_s.split()
        grounded = label in fact_block
        is_instance = False
        hyps_ok = False
        unknown = False
        reason = ""

        parts = rule_parts(mm, label)
        if parts is None:
            reason = "label is not a known logical rule"
        else:
            conclusion, hyp_templates, variables = parts
            try:
                subst = match_template(conclusion, expr, variables)
            except MatchBudgetExceeded:
                unknown = True
                subst = None
                reason = "match budget exceeded"
            if subst is not None:
                is_instance = True
                hyps_ok = discharge_hypotheses(hyp_templates, subst, variables, derived)
                if not hyps_ok:
                    unmet = norm(apply_subst(hyp_templates[0], subst))
                    reason = f"undischarged hypothesis: {unmet[:60]}"
            elif not unknown:
                reason = "expression is not a substitution instance of the rule"

        res.steps.append(
            StepResult(i, label, expr_s, grounded, is_instance, hyps_ok, unknown, reason)
        )
        derived.append(expr)
        all_grounded &= grounded
        all_instances &= is_instance
        all_hyps &= hyps_ok
        res.any_unknown |= unknown

    res.all_grounded = all_grounded
    res.all_instances = all_instances
    res.all_hyps_discharged = all_hyps
    res.goal_reached = norm(steps[-1][1]) == norm(goal)
    res.valid = (
        all_grounded and all_instances and all_hyps and res.goal_reached and not res.any_unknown
    )

    if not res.valid and not res.reason:
        if not res.goal_reached:
            res.reason = "final step is not the goal"
        else:
            bad = next((s for s in res.steps if not s.ok), None)
            res.reason = f"step {bad.index} ({bad.label}): {bad.reason}" if bad else "unknown"
    return res
