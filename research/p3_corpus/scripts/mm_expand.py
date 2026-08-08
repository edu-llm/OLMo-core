"""A Metamath source-proof verifier and expression-trace expander.

set.mm stores proofs as compressed label sequences; the intermediate formulas are
not in the file. This runs the RPN stack machine a verifier runs, and records the
expression produced at each step — which is what a model in the state-prediction
design would have to emit.

The replay is a verifier, not a renderer with a final-expression heuristic. Every
assertion application checks floating types, essential hypotheses, and mandatory
disjoint-variable conditions in the theorem's declaration context.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

COMMENT = re.compile(r"\$\(.*?\$\)", re.DOTALL)


class Incomplete(Exception):
    """The requested proof is unsupported, malformed, or invalid."""


@dataclass(frozen=True)
class AssertionFrame:
    """The declaration context captured when an assertion is parsed."""

    active_disjoint: frozenset[frozenset[str]]
    mandatory_disjoint: frozenset[frozenset[str]]
    active_f: tuple[tuple[str, str, str], ...]
    active_e: tuple[tuple[str, tuple[str, ...]], ...]
    active_variables: frozenset[str]
    statement_index: int


class Frame:
    __slots__ = ("v", "f", "e", "f_order", "d")

    def __init__(self):
        self.v = set()
        self.f = {}  # var -> (label, typecode)
        self.e = []  # (label, expr)
        self.f_order = []  # (label, typecode, var) in appearance order
        self.d = set()  # unordered two-variable frozensets


class MM:
    def __init__(self):
        self.constants = set()
        self.variables = set()
        self.stack: list[Frame] = [Frame()]
        self.labels = {}  # label -> ('$f'|'$e'|'$a'|'$p', data)
        self.label_order = {}
        self.assertion_frames: dict[str, AssertionFrame] = {}

    # ---------------------------------------------------------------- scope
    def push(self):
        self.stack.append(Frame())

    def pop(self):
        if len(self.stack) == 1:
            raise Incomplete("scope stack underflow")
        self.stack.pop()

    def lookup_f(self, var):
        for fr in reversed(self.stack):
            if var in fr.f:
                return fr.f[var]
        return None

    def active_e(self):
        out = []
        for fr in self.stack:
            out.extend(fr.e)
        return out

    def active_f(self):
        out = []
        for fr in self.stack:
            out.extend(fr.f_order)
        return out

    def active_d(self):
        out = set()
        for fr in self.stack:
            out.update(fr.d)
        return out

    def active_v(self):
        out = set()
        for fr in self.stack:
            out.update(fr.v)
        return out

    def is_var(self, tok):
        for fr in reversed(self.stack):
            if tok in fr.v:
                return True
        return False

    # ------------------------------------------------------------ mandatory
    def mandatory(self, expr):
        ess = self.active_e()
        used = set()
        for tok in expr:
            if self.is_var(tok):
                used.add(tok)
        for _, e in ess:
            for tok in e:
                if self.is_var(tok):
                    used.add(tok)
        flo = [(lbl, tc, v) for lbl, tc, v in self.active_f() if v in used]
        # Mandatory hypotheses are ordered by appearance: all $f for mandatory
        # variables first (set.mm declares them before use), then the $e.
        return [("$f", lbl, (tc, v)) for lbl, tc, v in flo] + [
            ("$e", lbl, e) for lbl, e in ess
        ]

    def assertion_frame(self, mand) -> AssertionFrame:
        mandatory_variables = {
            data[1] for kind, _, data in mand if kind == "$f"
        }
        active_disjoint = self.active_d()
        mandatory_disjoint = {
            pair
            for pair in active_disjoint
            if pair <= mandatory_variables
        }
        return AssertionFrame(
            active_disjoint=frozenset(active_disjoint),
            mandatory_disjoint=frozenset(mandatory_disjoint),
            active_f=tuple(self.active_f()),
            active_e=tuple(
                (hyp_label, tuple(expr))
                for hyp_label, expr in self.active_e()
            ),
            active_variables=frozenset(self.active_v()),
            statement_index=len(self.label_order),
        )

    def record(self, label, kind, data):
        if not label:
            raise Incomplete(f"{kind} statement has no label")
        if label in self.labels:
            raise Incomplete(f"duplicate label: {label}")
        self.label_order[label] = len(self.label_order)
        self.labels[label] = (kind, data)

    # ---------------------------------------------------------------- parse
    def parse(self, path):
        text = COMMENT.sub(
            " ",
            Path(path).read_text(encoding="utf-8", errors="replace"),
        )
        toks = text.split()
        i, n = 0, len(toks)
        label = None
        while i < n:
            t = toks[i]
            if t == "${":
                self.push()
                i += 1
            elif t == "$}":
                self.pop()
                i += 1
            elif t == "$c":
                j = toks.index("$.", i)
                self.constants.update(toks[i + 1 : j])
                i = j + 1
            elif t == "$v":
                j = toks.index("$.", i)
                declared = toks[i + 1 : j]
                self.stack[-1].v.update(declared)
                self.variables.update(declared)
                i = j + 1
            elif t == "$d":
                j = toks.index("$.", i)
                declared = toks[i + 1 : j]
                if len(set(declared)) != len(declared):
                    raise Incomplete("$d statement repeats a variable")
                if not set(declared) <= self.active_v():
                    raise Incomplete("$d statement contains an inactive variable")
                self.stack[-1].d.update(
                    frozenset(pair) for pair in combinations(declared, 2)
                )
                i = j + 1
            elif t == "$f":
                j = toks.index("$.", i)
                if j != i + 3:
                    raise Incomplete(f"malformed $f statement: {label}")
                tc, var = toks[i + 1], toks[i + 2]
                if tc not in self.constants or not self.is_var(var):
                    raise Incomplete(f"invalid $f declaration: {label}")
                self.stack[-1].f[var] = (label, tc)
                self.stack[-1].f_order.append((label, tc, var))
                self.record(label, "$f", (tc, var))
                i = j + 1
            elif t == "$e":
                j = toks.index("$.", i)
                expr = toks[i + 1 : j]
                self.stack[-1].e.append((label, expr))
                self.record(label, "$e", expr)
                i = j + 1
            elif t == "$a":
                j = toks.index("$.", i)
                expr = toks[i + 1 : j]
                mand = self.mandatory(expr)
                self.assertion_frames[label] = self.assertion_frame(mand)
                self.record(label, "$a", (expr, mand))
                i = j + 1
            elif t == "$p":
                j = toks.index("$=", i)
                expr = toks[i + 1 : j]
                k = toks.index("$.", j)
                proof = toks[j + 1 : k]
                mand = self.mandatory(expr)
                self.assertion_frames[label] = self.assertion_frame(mand)
                self.record(label, "$p", (expr, mand, proof))
                i = k + 1
            else:
                label = t
                i += 1
        if len(self.stack) != 1:
            raise Incomplete("unclosed ${ scope")
        return self


def apply_subst(expr, subst):
    out = []
    for tok in expr:
        if tok in subst:
            out.extend(subst[tok])
        else:
            out.append(tok)
    return out


def decode_compressed(proof, mand_n, labels_n):
    """Yield step indices (0-based into mand + labels + saved) and save flags."""
    del mand_n, labels_n
    body = "".join(proof)
    out, num, started = [], 0, False
    for c in body:
        if "U" <= c <= "Y":
            num = num * 5 + (ord(c) - ord("U") + 1)
            started = True
        elif "A" <= c <= "T":
            num = num * 20 + (ord(c) - ord("A") + 1)
            out.append(("step", num - 1))
            num, started = 0, False
        elif c == "Z":
            if started:
                raise Incomplete("save interrupts a compressed proof number")
            out.append(("save", None))
        elif c == "?":
            if started:
                raise Incomplete("unknown marker interrupts a compressed proof number")
            out.append(("unknown", None))
        else:
            raise Incomplete(f"invalid compressed proof character: {c!r}")
    if started:
        raise Incomplete("dangling compressed proof number")
    return out


def _variables_in(mm: MM, expr) -> set[str]:
    return {token for token in expr if token in mm.variables}


def _enforce_disjoint(
    mm: MM,
    rule_frame: AssertionFrame,
    target_frame: AssertionFrame,
    subst,
) -> None:
    for pair in rule_frame.mandatory_disjoint:
        left, right = tuple(pair)
        left_variables = _variables_in(mm, subst[left])
        right_variables = _variables_in(mm, subst[right])
        for left_var in left_variables:
            for right_var in right_variables:
                if left_var == right_var:
                    raise Incomplete(
                        "disjoint variable violation: "
                        f"{left} and {right} both contain {left_var}"
                    )
                if frozenset((left_var, right_var)) not in target_frame.active_disjoint:
                    raise Incomplete(
                        "disjoint variable violation: target context does not authorize "
                        f"$d {left_var} {right_var}"
                    )


def _apply_assertion(
    mm: MM,
    assertion_label: str,
    conclusion,
    mand,
    args,
    target_frame: AssertionFrame,
):
    subst = {}
    for (hyp_kind, hyp_label, hyp_data), arg in zip(mand, args):
        if hyp_kind == "$f":
            typecode, variable = hyp_data
            if not arg or arg[0] != typecode:
                got = arg[0] if arg else "<empty>"
                raise Incomplete(
                    f"typecode mismatch for {assertion_label}/{hyp_label}: "
                    f"expected {typecode}, got {got}"
                )
            value = list(arg[1:])
            previous = subst.get(variable)
            if previous is not None and previous != value:
                raise Incomplete(
                    f"inconsistent substitution for {assertion_label}/{variable}"
                )
            subst[variable] = value
        else:
            required = apply_subst(hyp_data, subst)
            if list(arg) != required:
                raise Incomplete(
                    "essential hypothesis mismatch for "
                    f"{assertion_label}/{hyp_label}: expected {' '.join(required)}, "
                    f"got {' '.join(arg)}"
                )

    rule_frame = mm.assertion_frames.get(assertion_label)
    if rule_frame is None:
        raise Incomplete(f"missing assertion frame: {assertion_label}")
    _enforce_disjoint(mm, rule_frame, target_frame, subst)
    return apply_subst(conclusion, subst)


def expand(mm: MM, label: str):
    entry = mm.labels.get(label)
    if entry is None:
        raise Incomplete(f"unknown theorem label: {label}")
    kind, data = entry
    if kind != "$p":
        raise Incomplete(f"label is not a provable assertion: {label}")
    expr, mand, proof = data
    if proof == ["?"]:
        raise Incomplete("proof contains ?")
    if not proof or proof[0] != "(":
        raise Incomplete("unsupported uncompressed or empty proof")
    try:
        close = proof.index(")")
    except ValueError as exc:
        raise Incomplete("compressed proof has no closing parenthesis") from exc
    ref_labels = proof[1:close]
    ops = decode_compressed(
        proof[close + 1 :],
        len(mand),
        len(ref_labels),
    )
    target_frame = mm.assertion_frames.get(label)
    if target_frame is None:
        raise Incomplete(f"missing theorem frame: {label}")
    target_hypotheses = {
        hyp_label for hyp_label, _, _ in target_frame.active_f
    } | {
        hyp_label for hyp_label, _ in target_frame.active_e
    }
    target_index = mm.label_order[label]

    stack, saved, trace = [], [], []
    for kind_op, idx in ops:
        if kind_op == "unknown":
            raise Incomplete("proof contains ?")
        if kind_op == "save":
            if not stack:
                raise Incomplete("save with empty stack")
            saved.append(list(stack[-1]))
            continue
        if idx is None or idx < 0:
            raise Incomplete("invalid compressed proof index")
        if idx < len(mand):
            hk, hl, hd = mand[idx]
            e = [hd[0], hd[1]] if hk == "$f" else list(hd)
            stack.append(e)
            trace.append((hl, list(e), True))
            continue
        idx -= len(mand)
        if idx < len(ref_labels):
            lbl = ref_labels[idx]
            referenced = mm.labels.get(lbl)
            if referenced is None:
                raise Incomplete(f"unknown proof label: {lbl}")
            if mm.label_order[lbl] >= target_index:
                raise Incomplete(f"proof uses non-prior label: {lbl}")
            lk, ld = referenced
            if lk == "$f":
                if lbl not in target_hypotheses:
                    raise Incomplete(f"proof uses inactive $f hypothesis: {lbl}")
                e = [ld[0], ld[1]]
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            if lk == "$e":
                if lbl not in target_hypotheses:
                    raise Incomplete(f"proof uses inactive $e hypothesis: {lbl}")
                e = list(ld)
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            if lk not in ("$a", "$p"):
                raise Incomplete(f"unsupported proof label kind {lk}: {lbl}")
            sexpr, smand = ld[0], ld[1]
            k = len(smand)
            if k > len(stack):
                raise Incomplete("stack underflow")
            args = list(stack[-k:]) if k else []
            if k:
                del stack[-k:]
            res = _apply_assertion(
                mm,
                lbl,
                sexpr,
                smand,
                args,
                target_frame,
            )
            stack.append(res)
            trace.append((lbl, list(res), False))
            continue
        idx -= len(ref_labels)
        if idx >= len(saved):
            raise Incomplete("bad saved index")
        e = list(saved[idx])
        stack.append(e)
        trace.append(("(reuse)", e, True))
    if len(stack) != 1:
        raise Incomplete(f"final stack size {len(stack)}")
    if stack[0] != expr:
        raise Incomplete(
            f"final expression mismatch: expected {' '.join(expr)}, "
            f"got {' '.join(stack[0])}"
        )
    return expr, mand, ref_labels, trace


if __name__ == "__main__":
    mm = MM().parse(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dscount/set.mm")
    print(f"parsed {len(mm.labels):,} labels")
    target = sys.argv[2] if len(sys.argv) > 2 else "mp2"
    expr, mand, refs, trace = expand(mm, target)
    print(f"\ntheorem {target} : {' '.join(expr)}")
    print(f"mandatory hyps: {[hyp_label for _, hyp_label, _ in mand]}")
    print(f"referenced    : {refs}")
    print("\nfull trace:")
    for i, (lbl, e, is_hyp) in enumerate(trace, 1):
        tag = "hyp " if is_hyp else "step"
        print(f"  {i:>3} {tag} {lbl:<12} {' '.join(e)}")
