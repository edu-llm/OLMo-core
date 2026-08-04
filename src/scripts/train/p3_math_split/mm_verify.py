"""Sound, tri-state verification for generated Metamath expression traces.

For every emitted rule application this module simultaneously establishes the
conclusion substitution, all essential hypotheses, floating-hypothesis syntax types,
and mandatory disjoint-variable conditions in the target theorem's actual source
context. Rule labels must also be present in the model-visible fact block, and the
final expression must equal the theorem goal.

Verification returns ``valid``, ``invalid``, or ``unknown``. Bounded ambiguity,
unsupported source context, and syntax-search exhaustion are ``unknown`` rather than
false invalids. Callers must keep unknowns out of valid/invalid denominators.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

MATCH_NODE_BUDGET = 200_000
SYNTAX_NODE_BUDGET = 200_000
METAMATH_DATABASES = frozenset(("set", "iset", "nf"))

# `  1  syl          |- ( ph -> ch )` — the shape build_corpus.py emits.
STEP_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(\|-\s.*)$")


class MatchBudgetExceeded(Exception):
    """A bounded search cannot make a sound valid/invalid decision."""


class VerificationStatus(str, Enum):
    """The three possible verification outcomes."""

    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass
class SearchBudget:
    """Shared node budget for one bounded search."""

    remaining: int

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise MatchBudgetExceeded()


def norm(expr: Sequence[str] | str) -> str:
    return " ".join(expr.split()) if isinstance(expr, str) else " ".join(expr)


def _subst_key(subst: Mapping[str, Sequence[str]]) -> tuple:
    return tuple(sorted((name, tuple(value)) for name, value in subst.items()))


def _minimum_concrete_tokens(
    template: Sequence[str],
    start: int,
    variables: set[str],
    subst: Mapping[str, Sequence[str]],
) -> int:
    total = 0
    for token in template[start:]:
        if token not in variables:
            total += 1
        elif token in subst:
            total += len(subst[token])
    return total


def _iter_template_matches(
    template: Sequence[str],
    concrete: Sequence[str],
    variables: set[str],
    initial_subst: Mapping[str, Sequence[str]],
    budget: SearchBudget,
) -> Iterator[dict[str, list[str]]]:
    """Enumerate substitutions iteratively without rebinding initial values."""

    seed = {name: list(value) for name, value in initial_subst.items()}
    stack: list[tuple[int, int, dict[str, list[str]]]] = [(0, 0, seed)]
    emitted = set()
    while stack:
        budget.spend()
        template_index, concrete_index, subst = stack.pop()
        if template_index == len(template):
            if concrete_index == len(concrete):
                key = _subst_key(subst)
                if key not in emitted:
                    emitted.add(key)
                    yield subst
            continue

        token = template[template_index]
        if token not in variables:
            if concrete_index < len(concrete) and concrete[concrete_index] == token:
                stack.append((template_index + 1, concrete_index + 1, subst))
            continue

        if token in subst:
            bound = subst[token]
            end = concrete_index + len(bound)
            if list(concrete[concrete_index:end]) == bound:
                stack.append((template_index + 1, end, subst))
            continue

        minimum_rest = _minimum_concrete_tokens(
            template,
            template_index + 1,
            variables,
            subst,
        )
        maximum_end = len(concrete) - minimum_rest
        # Reverse push order makes the shortest substitution get explored first.
        for end in range(maximum_end, concrete_index - 1, -1):
            extended = dict(subst)
            extended[token] = list(concrete[concrete_index:end])
            stack.append((template_index + 1, end, extended))


def match_template(
    template: Sequence[str],
    concrete: Sequence[str],
    variables: set[str],
    initial_subst: Mapping[str, Sequence[str]] | None = None,
    node_budget: int = MATCH_NODE_BUDGET,
) -> dict[str, list[str]] | None:
    """Return the first bounded match, preserving immutable initial bindings."""

    matches = _iter_template_matches(
        template,
        concrete,
        variables,
        initial_subst or {},
        SearchBudget(node_budget),
    )
    return next(matches, None)


def apply_subst(expr: Sequence[str], subst: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for tok in expr:
        out.extend(subst[tok]) if tok in subst else out.append(tok)
    return out


@dataclass(frozen=True)
class RuleParts:
    """Templates and source frame needed for one assertion application."""

    conclusion: tuple[str, ...]
    floating: tuple[tuple[str, str, str], ...]
    essential: tuple[tuple[str, tuple[str, ...]], ...]
    variables: frozenset[str]
    mandatory_disjoint: frozenset[frozenset[str]]


def _get_frame(mm, label: str):
    return getattr(mm, "assertion_frames", {}).get(label)


def rule_parts(mm, label: str) -> RuleParts | None:
    """Return a logical assertion's complete mandatory source frame."""

    entry = mm.labels.get(label)
    if entry is None:
        return None
    kind, data = entry
    if kind not in ("$a", "$p") or not data or not data[0]:
        return None
    mand = data[1] if len(data) > 1 else []
    frame = _get_frame(mm, label)
    if frame is None:
        return None
    floating = tuple(
        (hyp_label, data[0], data[1]) for kind, hyp_label, data in mand if kind == "$f"
    )
    essential = tuple(
        (hyp_label, tuple(data)) for kind, hyp_label, data in mand if kind == "$e"
    )
    return RuleParts(
        conclusion=tuple(data[0]),
        floating=floating,
        essential=essential,
        variables=frozenset(variable for _, _, variable in floating),
        mandatory_disjoint=frame.mandatory_disjoint,
    )


def render_rule_statement(mm, label: str) -> str | None:
    """Render one source assertion exactly as a visible fact-block statement."""

    entry = mm.labels.get(label)
    if entry is None or entry[0] not in ("$a", "$p"):
        return None
    data = entry[1]
    conclusion = norm(data[0])
    essentials = [
        norm(hyp_data) for hyp_kind, _, hyp_data in data[1] if hyp_kind == "$e"
    ]
    if not essentials:
        return conclusion
    return f"{' & '.join(essentials)} => {conclusion}"


def _match_essential_hypotheses(
    essential: Sequence[tuple[str, Sequence[str]]],
    initial_subst: Mapping[str, Sequence[str]],
    variables: set[str],
    derived: Sequence[Sequence[str]],
    budget: SearchBudget,
) -> Iterator[tuple[dict[str, list[str]], tuple[int, ...]]]:
    stack: list[tuple[int, dict[str, list[str]], tuple[int, ...]]] = [
        (0, {name: list(value) for name, value in initial_subst.items()}, ())
    ]
    seen = set()
    while stack:
        budget.spend()
        hyp_index, subst, sources = stack.pop()
        state_key = (hyp_index, _subst_key(subst))
        if state_key in seen:
            continue
        seen.add(state_key)
        if hyp_index == len(essential):
            yield subst, sources
            continue

        _, template = essential[hyp_index]
        next_states = []
        for source_index, candidate in enumerate(derived):
            for extended in _iter_template_matches(
                template,
                candidate,
                variables,
                subst,
                budget,
            ):
                next_states.append((hyp_index + 1, extended, sources + (source_index,)))
        stack.extend(reversed(next_states))


def _variables_in(mm, expression: Sequence[str]) -> set[str]:
    variables = getattr(mm, "variables", None)
    if variables is None:
        return set()
    return {token for token in expression if token in variables}


def _check_disjoint(
    mm,
    pairs: Sequence[frozenset[str]],
    subst: Mapping[str, Sequence[str]],
    target_frame,
) -> tuple[bool | None, str]:
    if not pairs:
        return True, ""
    if not hasattr(mm, "variables") or target_frame is None:
        return None, "disjoint-variable source context is unavailable"
    for pair in pairs:
        left, right = tuple(pair)
        if left not in subst or right not in subst:
            return None, f"substitution does not bind disjoint pair {left}/{right}"
        for left_var in _variables_in(mm, subst[left]):
            for right_var in _variables_in(mm, subst[right]):
                if left_var == right_var:
                    return (
                        False,
                        f"{left} and {right} both contain variable {left_var}",
                    )
                if frozenset((left_var, right_var)) not in target_frame.active_disjoint:
                    return (
                        False,
                        f"target context lacks $d {left_var} {right_var}",
                    )
    return True, ""


@dataclass(frozen=True)
class SyntaxProduction:
    """One source assertion usable to establish a floating type."""

    label: str
    typecode: str
    template: tuple[str, ...]
    floating: tuple[tuple[str, str], ...]
    variables: frozenset[str]
    mandatory_disjoint: frozenset[frozenset[str]]
    has_essential: bool
    statement_index: int


def _syntax_index(mm) -> dict[str, tuple[SyntaxProduction, ...]]:
    cached = getattr(mm, "_mm_verify_syntax_index", None)
    if cached is not None:
        return cached
    by_type: dict[str, list[SyntaxProduction]] = {}
    label_order = getattr(mm, "label_order", {})
    for label, (kind, data) in mm.labels.items():
        if kind not in ("$a", "$p") or not data or not data[0]:
            continue
        conclusion = data[0]
        if conclusion[0] == "|-":
            continue
        mand = data[1] if len(data) > 1 else []
        floating = tuple(
            (hyp_data[0], hyp_data[1])
            for hyp_kind, _, hyp_data in mand
            if hyp_kind == "$f"
        )
        variables = frozenset(variable for _, variable in floating)
        frame = _get_frame(mm, label)
        if frame is None:
            continue
        production = SyntaxProduction(
            label=label,
            typecode=conclusion[0],
            template=tuple(conclusion[1:]),
            floating=floating,
            variables=variables,
            mandatory_disjoint=frame.mandatory_disjoint,
            has_essential=any(hyp_kind == "$e" for hyp_kind, _, _ in mand),
            statement_index=label_order.get(label, -1),
        )
        by_type.setdefault(production.typecode, []).append(production)
    result = {key: tuple(value) for key, value in by_type.items()}
    mm._mm_verify_syntax_index = result
    return result


@dataclass(frozen=True)
class SyntaxResult:
    status: VerificationStatus
    reason: str = ""
    witness: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SyntaxGoal:
    typecode: str
    expression: tuple[str, ...]


@dataclass(frozen=True)
class _SyntaxEmit:
    label: str


class SyntaxTypeChecker:
    """Memoized bounded syntax proof search for one target theorem context."""

    def __init__(
        self,
        mm,
        target_label: str,
        target_frame,
        node_budget: int,
    ):
        self.mm = mm
        self.target_frame = target_frame
        self.target_index = getattr(mm, "label_order", {}).get(target_label)
        self.productions = _syntax_index(mm)
        self.floating = tuple(target_frame.active_f)
        self.budget = SearchBudget(node_budget)
        self.cache: dict[tuple[str, tuple[str, ...]], SyntaxResult] = {}

    def check(self, typecode: str, expression: Sequence[str]) -> SyntaxResult:
        key = (typecode, tuple(expression))
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._search(typecode, tuple(expression))
        self.cache[key] = result
        return result

    def _search(self, typecode: str, expression: tuple[str, ...]) -> SyntaxResult:
        if self.target_index is None:
            return SyntaxResult(
                VerificationStatus.UNKNOWN,
                "target declaration order is unavailable",
            )
        initial_tasks: tuple[_SyntaxGoal | _SyntaxEmit, ...] = (
            _SyntaxGoal(typecode, expression),
        )
        stack: list[tuple[tuple[_SyntaxGoal | _SyntaxEmit, ...], tuple[str, ...]]] = [
            (initial_tasks, ())
        ]
        seen = set()
        encountered_unsupported = False
        try:
            while stack:
                self.budget.spend()
                tasks, witness = stack.pop()
                if not tasks:
                    return SyntaxResult(
                        VerificationStatus.VALID,
                        witness=witness,
                    )
                task, rest = tasks[0], tasks[1:]
                if isinstance(task, _SyntaxEmit):
                    stack.append((rest, witness + (task.label,)))
                    continue
                state_key = tasks
                if state_key in seen:
                    continue
                seen.add(state_key)

                alternatives = []
                for floating_label, floating_type, variable in self.floating:
                    if floating_type == task.typecode and task.expression == (
                        variable,
                    ):
                        alternatives.append((rest, witness + (floating_label,)))

                for production in self.productions.get(task.typecode, ()):
                    if (
                        production.statement_index < 0
                        or production.statement_index >= self.target_index
                    ):
                        continue
                    for subst in _iter_template_matches(
                        production.template,
                        task.expression,
                        set(production.variables),
                        {},
                        self.budget,
                    ):
                        if production.has_essential:
                            encountered_unsupported = True
                            continue
                        disjoint_ok, _ = _check_disjoint(
                            self.mm,
                            production.mandatory_disjoint,
                            subst,
                            self.target_frame,
                        )
                        if disjoint_ok is None:
                            encountered_unsupported = True
                            continue
                        if not disjoint_ok:
                            continue
                        if any(
                            variable not in subst for _, variable in production.floating
                        ):
                            encountered_unsupported = True
                            continue
                        subgoals = tuple(
                            _SyntaxGoal(required_type, tuple(subst[variable]))
                            for required_type, variable in production.floating
                        )
                        alternatives.append(
                            (
                                subgoals + (_SyntaxEmit(production.label),) + rest,
                                witness,
                            )
                        )
                stack.extend(reversed(alternatives))
        except MatchBudgetExceeded:
            return SyntaxResult(
                VerificationStatus.UNKNOWN,
                "syntax proof search budget exceeded",
            )
        if encountered_unsupported:
            return SyntaxResult(
                VerificationStatus.UNKNOWN,
                "matching syntax assertions require unsupported context",
            )
        return SyntaxResult(
            VerificationStatus.INVALID,
            f"no source syntax proof establishes {typecode} {' '.join(expression)}",
        )


@dataclass
class StepResult:
    index: int
    label: str
    expr: str
    status: VerificationStatus
    grounded: bool
    is_instance: bool | None
    hyps_discharged: bool | None
    floating_types_valid: bool | None
    disjoint_valid: bool | None
    reason_code: str = ""
    reason: str = ""
    substitution: tuple = ()
    hypothesis_sources: tuple[int, ...] = ()
    syntax_witnesses: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def ok(self) -> bool:
        return self.status is VerificationStatus.VALID

    @property
    def unknown(self) -> bool:
        return self.status is VerificationStatus.UNKNOWN


@dataclass
class ProofResult:
    status: VerificationStatus = VerificationStatus.INVALID
    parsed_steps: int = 0
    goal_reached: bool = False
    all_grounded: bool = False
    all_instances: bool = False
    all_hyps_discharged: bool = False
    all_floating_types_valid: bool = False
    all_disjoint_valid: bool = False
    exact_match: bool = False
    steps: list[StepResult] = field(default_factory=list)
    reason_code: str = ""
    reason: str = ""

    @property
    def valid(self) -> bool | None:
        """Compatibility view that preserves ``unknown`` as ``None``."""

        if self.status is VerificationStatus.UNKNOWN:
            return None
        return self.status is VerificationStatus.VALID

    @property
    def any_unknown(self) -> bool:
        return self.status is VerificationStatus.UNKNOWN or any(
            step.unknown for step in self.steps
        )

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "parsed_steps": self.parsed_steps,
            "valid": self.valid,
            "goal_reached": self.goal_reached,
            "all_grounded": self.all_grounded,
            "all_instances": self.all_instances,
            "all_hyps_discharged": self.all_hyps_discharged,
            "all_floating_types_valid": self.all_floating_types_valid,
            "all_disjoint_valid": self.all_disjoint_valid,
            "any_unknown": self.any_unknown,
            "exact_match": self.exact_match,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def parse_proof(text: str) -> list[tuple[str, str]]:
    """Pull `(label, expression)` pairs out of generated text.

    Stops at the first line that does not look like a step, so trailing chatter after a
    complete proof does not invalidate it, but interleaved garbage does.
    """
    steps: list[tuple[str, str]] = []
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


@dataclass(frozen=True)
class _ApplicationResult:
    status: VerificationStatus
    reason_code: str
    reason: str
    is_instance: bool | None
    hyps_discharged: bool | None
    floating_types_valid: bool | None
    disjoint_valid: bool | None
    substitution: tuple = ()
    hypothesis_sources: tuple[int, ...] = ()
    syntax_witnesses: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _verify_application(
    mm,
    parts: RuleParts,
    expression: Sequence[str],
    derived: Sequence[Sequence[str]],
    target_frame,
    syntax_checker: SyntaxTypeChecker,
    node_budget: int,
) -> _ApplicationResult:
    budget = SearchBudget(node_budget)
    saw_conclusion = False
    saw_hypotheses = False
    saw_type_failure = False
    saw_disjoint_failure = False
    unknown_reason = ""
    last_type_reason = ""
    last_disjoint_reason = ""
    try:
        conclusion_matches = _iter_template_matches(
            parts.conclusion,
            expression,
            set(parts.variables),
            {},
            budget,
        )
        for conclusion_subst in conclusion_matches:
            saw_conclusion = True
            for subst, hypothesis_sources in _match_essential_hypotheses(
                parts.essential,
                conclusion_subst,
                set(parts.variables),
                derived,
                budget,
            ):
                saw_hypotheses = True
                syntax_witnesses = []
                typing_valid = True
                typing_unknown = False
                for _, typecode, variable in parts.floating:
                    value = subst.get(variable)
                    if value is None:
                        typing_unknown = True
                        last_type_reason = (
                            f"mandatory variable {variable} has no substitution"
                        )
                        break
                    syntax = syntax_checker.check(typecode, value)
                    if syntax.status is VerificationStatus.UNKNOWN:
                        typing_unknown = True
                        last_type_reason = syntax.reason
                        break
                    if syntax.status is VerificationStatus.INVALID:
                        typing_valid = False
                        last_type_reason = syntax.reason
                        break
                    syntax_witnesses.append((variable, syntax.witness))
                if typing_unknown:
                    unknown_reason = last_type_reason
                    continue
                if not typing_valid:
                    saw_type_failure = True
                    continue

                disjoint_ok, disjoint_reason = _check_disjoint(
                    mm,
                    parts.mandatory_disjoint,
                    subst,
                    target_frame,
                )
                if disjoint_ok is None:
                    unknown_reason = disjoint_reason
                    continue
                if not disjoint_ok:
                    saw_disjoint_failure = True
                    last_disjoint_reason = disjoint_reason
                    continue
                return _ApplicationResult(
                    status=VerificationStatus.VALID,
                    reason_code="",
                    reason="",
                    is_instance=True,
                    hyps_discharged=True,
                    floating_types_valid=True,
                    disjoint_valid=True,
                    substitution=_subst_key(subst),
                    hypothesis_sources=hypothesis_sources,
                    syntax_witnesses=tuple(syntax_witnesses),
                )
    except MatchBudgetExceeded:
        return _ApplicationResult(
            status=VerificationStatus.UNKNOWN,
            reason_code="match_budget_exceeded",
            reason="substitution or hypothesis search budget exceeded",
            is_instance=True if saw_conclusion else None,
            hyps_discharged=True if saw_hypotheses else None,
            floating_types_valid=None,
            disjoint_valid=None,
        )

    if unknown_reason:
        return _ApplicationResult(
            status=VerificationStatus.UNKNOWN,
            reason_code="syntax_typing_unknown",
            reason=unknown_reason,
            is_instance=True,
            hyps_discharged=True,
            floating_types_valid=None,
            disjoint_valid=None,
        )
    if not saw_conclusion:
        return _ApplicationResult(
            status=VerificationStatus.INVALID,
            reason_code="conclusion_not_instance",
            reason="expression is not a substitution instance of the rule",
            is_instance=False,
            hyps_discharged=False,
            floating_types_valid=False,
            disjoint_valid=False,
        )
    if not saw_hypotheses:
        return _ApplicationResult(
            status=VerificationStatus.INVALID,
            reason_code="essential_hypothesis_unmet",
            reason="no consistent substitution discharges every essential hypothesis",
            is_instance=True,
            hyps_discharged=False,
            floating_types_valid=False,
            disjoint_valid=False,
        )
    if saw_disjoint_failure:
        return _ApplicationResult(
            status=VerificationStatus.INVALID,
            reason_code="disjoint_variable_violation",
            reason=last_disjoint_reason,
            is_instance=True,
            hyps_discharged=True,
            floating_types_valid=True,
            disjoint_valid=False,
        )
    if saw_type_failure:
        return _ApplicationResult(
            status=VerificationStatus.INVALID,
            reason_code="floating_type_mismatch",
            reason=last_type_reason,
            is_instance=True,
            hyps_discharged=True,
            floating_types_valid=False,
            disjoint_valid=False,
        )
    return _ApplicationResult(
        status=VerificationStatus.UNKNOWN,
        reason_code="unsupported_application",
        reason="no supported complete assertion application was found",
        is_instance=True,
        hyps_discharged=True,
        floating_types_valid=None,
        disjoint_valid=None,
    )


def _split_database_qualified_label(label: str) -> tuple[str | None, str]:
    database, separator, native_label = label.partition(":")
    if separator and database in METAMATH_DATABASES and native_label:
        return database, native_label
    return None, label


def _resolve_target_identity(
    target_label: str | None,
    source_database: str | None,
) -> tuple[VerificationStatus, str, str, str | None, str | None,]:
    if source_database is not None and source_database not in METAMATH_DATABASES:
        return (
            VerificationStatus.INVALID,
            "unknown_source_database",
            f"unknown Metamath source database: {source_database}",
            target_label,
            source_database,
        )
    if target_label is None:
        return VerificationStatus.VALID, "", "", None, source_database

    target_database, native_target_label = _split_database_qualified_label(target_label)
    if (
        target_database is not None
        and source_database is not None
        and target_database != source_database
    ):
        return (
            VerificationStatus.INVALID,
            "target_database_mismatch",
            (
                f"target prefix {target_database!r} does not match source "
                f"database {source_database!r}"
            ),
            native_target_label,
            source_database,
        )
    return (
        VerificationStatus.VALID,
        "",
        "",
        native_target_label,
        target_database or source_database,
    )


def _target_context(mm, target_label: str | None, goal: str):
    if target_label is None:
        return (
            VerificationStatus.UNKNOWN,
            "target_context_required",
            "target theorem label/context is required for sound verification",
            None,
            {},
        )
    entry = mm.labels.get(target_label)
    if entry is None:
        return (
            VerificationStatus.UNKNOWN,
            "target_context_unavailable",
            f"target theorem is not present in the source database: {target_label}",
            None,
            {},
        )
    kind, data = entry
    if kind != "$p":
        return (
            VerificationStatus.INVALID,
            "target_label_not_theorem",
            f"target label is not a provable assertion: {target_label}",
            None,
            {},
        )
    if norm(data[0]) != norm(goal):
        return (
            VerificationStatus.INVALID,
            "goal_target_mismatch",
            "requested goal does not equal the target theorem assertion",
            None,
            {},
        )
    frame = _get_frame(mm, target_label)
    if frame is None:
        return (
            VerificationStatus.UNKNOWN,
            "target_context_unavailable",
            f"declaration frame is unavailable for {target_label}",
            None,
            {},
        )
    expected_local = {
        hyp_label: norm(hyp_data)
        for hyp_kind, hyp_label, hyp_data in data[1]
        if hyp_kind == "$e"
    }
    return VerificationStatus.VALID, "", "", frame, expected_local


def verify_proof(
    mm,
    generated: str,
    goal: str,
    fact_block: dict[str, str],
    gold_target: str | None = None,
    local_assumptions: dict[str, str] | None = None,
    *,
    target_label: str | None = None,
    source_database: str | None = None,
    match_node_budget: int = MATCH_NODE_BUDGET,
    syntax_node_budget: int = SYNTAX_NODE_BUDGET,
) -> ProofResult:
    """Verify one generated trace in an explicit target theorem context."""

    res = ProofResult()
    steps = parse_proof(generated)
    res.parsed_steps = len(steps)

    if gold_target is not None:
        res.exact_match = norm(generated.strip()) == norm(gold_target.strip())

    if not steps:
        res.status = VerificationStatus.INVALID
        res.reason_code = "no_proof_steps"
        res.reason = "no parsable proof steps"
        return res

    (
        identity_status,
        identity_reason_code,
        identity_reason,
        native_target_label,
        source_database,
    ) = _resolve_target_identity(target_label, source_database)
    if identity_status is not VerificationStatus.VALID:
        res.status = identity_status
        res.reason_code = identity_reason_code
        res.reason = identity_reason
        return res

    (
        target_status,
        target_reason_code,
        target_reason,
        target_frame,
        expected_local,
    ) = _target_context(mm, native_target_label, goal)
    if target_status is not VerificationStatus.VALID:
        res.status = target_status
        res.reason_code = target_reason_code
        res.reason = target_reason
        return res

    supplied_local = local_assumptions or {}
    for local_label, expression in supplied_local.items():
        if (
            local_label not in expected_local
            or norm(expression) != expected_local[local_label]
        ):
            res.status = VerificationStatus.INVALID
            res.reason_code = "local_assumption_not_in_target_frame"
            res.reason = (
                f"{local_label} is not an exact essential hypothesis of "
                f"{target_label or native_target_label}"
            )
            return res

    derived: list[list[str]] = [
        expression.split() for expression in supplied_local.values()
    ]
    possibly_derived: list[list[str]] = list(derived)
    syntax_checker = SyntaxTypeChecker(
        mm,
        native_target_label,
        target_frame,
        syntax_node_budget,
    )
    all_grounded = True
    all_instances = True
    all_hyps = True
    all_floating = True
    all_disjoint = True

    for i, (label, expr_s) in enumerate(steps, 1):
        expr = expr_s.split()
        rule_database, native_label = _split_database_qualified_label(label)
        visible = label in fact_block
        entry = mm.labels.get(native_label)
        expected_visible_statement = render_rule_statement(mm, native_label)
        grounded = (
            visible
            and expected_visible_statement is not None
            and norm(fact_block[label]) == norm(expected_visible_statement)
        )
        if rule_database is not None and rule_database != source_database:
            application = _ApplicationResult(
                status=VerificationStatus.INVALID,
                reason_code="rule_database_mismatch",
                reason=(
                    f"rule prefix {rule_database!r} does not match target/source "
                    f"database {source_database!r}"
                ),
                is_instance=None,
                hyps_discharged=None,
                floating_types_valid=None,
                disjoint_valid=None,
            )
        elif not visible:
            application = _ApplicationResult(
                status=VerificationStatus.INVALID,
                reason_code="rule_not_visible",
                reason="rule label is not grounded in the visible fact block",
                is_instance=None,
                hyps_discharged=None,
                floating_types_valid=None,
                disjoint_valid=None,
            )
        elif expected_visible_statement is not None and not grounded:
            application = _ApplicationResult(
                status=VerificationStatus.UNKNOWN,
                reason_code="visible_rule_mismatch",
                reason=(
                    "visible fact statement does not match the pinned source "
                    "assertion"
                ),
                is_instance=None,
                hyps_discharged=None,
                floating_types_valid=None,
                disjoint_valid=None,
            )
        elif entry is None or entry[0] not in ("$a", "$p"):
            application = _ApplicationResult(
                status=VerificationStatus.INVALID,
                reason_code="unknown_rule",
                reason="label is not a known logical assertion",
                is_instance=False,
                hyps_discharged=False,
                floating_types_valid=False,
                disjoint_valid=False,
            )
        else:
            parts = rule_parts(mm, native_label)
            if parts is None:
                application = _ApplicationResult(
                    status=VerificationStatus.UNKNOWN,
                    reason_code="rule_context_unavailable",
                    reason=(
                        "source declaration frame is unavailable for " f"{native_label}"
                    ),
                    is_instance=None,
                    hyps_discharged=None,
                    floating_types_valid=None,
                    disjoint_valid=None,
                )
            else:
                application = _verify_application(
                    mm,
                    parts,
                    expr,
                    derived,
                    target_frame,
                    syntax_checker,
                    match_node_budget,
                )
                if (
                    application.status is VerificationStatus.INVALID
                    and application.reason_code == "essential_hypothesis_unmet"
                    and len(possibly_derived) > len(derived)
                ):
                    possible_application = _verify_application(
                        mm,
                        parts,
                        expr,
                        possibly_derived,
                        target_frame,
                        syntax_checker,
                        match_node_budget,
                    )
                    if possible_application.status is not VerificationStatus.INVALID:
                        application = _ApplicationResult(
                            status=VerificationStatus.UNKNOWN,
                            reason_code="depends_on_unknown_step",
                            reason=(
                                "essential hypotheses can be discharged only if "
                                "an earlier unknown step is sound"
                            ),
                            is_instance=possible_application.is_instance,
                            hyps_discharged=None,
                            floating_types_valid=(
                                possible_application.floating_types_valid
                            ),
                            disjoint_valid=possible_application.disjoint_valid,
                        )

        res.steps.append(
            StepResult(
                index=i,
                label=label,
                expr=expr_s,
                status=application.status,
                grounded=grounded,
                is_instance=application.is_instance,
                hyps_discharged=application.hyps_discharged,
                floating_types_valid=application.floating_types_valid,
                disjoint_valid=application.disjoint_valid,
                reason_code=application.reason_code,
                reason=application.reason,
                substitution=application.substitution,
                hypothesis_sources=application.hypothesis_sources,
                syntax_witnesses=application.syntax_witnesses,
            )
        )
        if application.status is VerificationStatus.VALID:
            derived.append(expr)
            possibly_derived.append(expr)
        elif application.status is VerificationStatus.UNKNOWN:
            possibly_derived.append(expr)
        all_grounded &= grounded
        all_instances &= application.is_instance is True
        all_hyps &= application.hyps_discharged is True
        all_floating &= application.floating_types_valid is True
        all_disjoint &= application.disjoint_valid is True

    res.all_grounded = all_grounded
    res.all_instances = all_instances
    res.all_hyps_discharged = all_hyps
    res.all_floating_types_valid = all_floating
    res.all_disjoint_valid = all_disjoint
    res.goal_reached = norm(steps[-1][1]) == norm(goal)
    invalid_step = next(
        (step for step in res.steps if step.status is VerificationStatus.INVALID),
        None,
    )
    unknown_step = next(
        (step for step in res.steps if step.status is VerificationStatus.UNKNOWN),
        None,
    )
    if invalid_step is not None:
        res.status = VerificationStatus.INVALID
        res.reason_code = invalid_step.reason_code
        res.reason = (
            f"step {invalid_step.index} ({invalid_step.label}): "
            f"{invalid_step.reason}"
        )
    elif not res.goal_reached:
        res.status = VerificationStatus.INVALID
        res.reason_code = "final_goal_mismatch"
        res.reason = "final step is not the goal"
    elif unknown_step is not None:
        res.status = VerificationStatus.UNKNOWN
        res.reason_code = unknown_step.reason_code
        res.reason = (
            f"step {unknown_step.index} ({unknown_step.label}): "
            f"{unknown_step.reason}"
        )
    else:
        res.status = VerificationStatus.VALID
    return res
