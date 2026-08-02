"""A Metamath proof expander: computes the formula at every proof step.

set.mm stores proofs as compressed label sequences; the intermediate formulas are
not in the file. This runs the RPN stack machine a verifier runs, and records the
expression produced at each step — which is what a model in the state-prediction
design would have to emit.

Enough of the Metamath spec is implemented to execute set.mm proofs: scoping,
mandatory hypotheses, compressed-proof decoding, and substitution. Disjoint
variable conditions are parsed but not enforced, which is fine for measuring
token volume and for display.
"""

from __future__ import annotations

import re
import sys

COMMENT = re.compile(r"\$\(.*?\$\)", re.DOTALL)


class Frame:
    __slots__ = ("v", "f", "e", "f_order")

    def __init__(self):
        self.v = set()
        self.f = {}  # var -> (label, typecode)
        self.e = []  # (label, expr)
        self.f_order = []  # (label, typecode, var) in appearance order


class MM:
    def __init__(self):
        self.constants = set()
        self.stack: list[Frame] = [Frame()]
        self.labels = {}  # label -> ('$f'|'$e'|'$a'|'$p', data)

    # ---------------------------------------------------------------- scope
    def push(self):
        self.stack.append(Frame())

    def pop(self):
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
        return [("$f", lbl, (tc, v)) for lbl, tc, v in flo] + [("$e", lbl, e) for lbl, e in ess]

    # ---------------------------------------------------------------- parse
    def parse(self, path):
        text = COMMENT.sub(" ", open(path, encoding="utf-8", errors="replace").read())
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
                self.stack[-1].v.update(toks[i + 1 : j])
                i = j + 1
            elif t == "$d":
                j = toks.index("$.", i)
                i = j + 1
            elif t == "$f":
                j = toks.index("$.", i)
                tc, var = toks[i + 1], toks[i + 2]
                self.stack[-1].f[var] = (label, tc)
                self.stack[-1].f_order.append((label, tc, var))
                self.labels[label] = ("$f", (tc, var))
                i = j + 1
            elif t == "$e":
                j = toks.index("$.", i)
                expr = toks[i + 1 : j]
                self.stack[-1].e.append((label, expr))
                self.labels[label] = ("$e", expr)
                i = j + 1
            elif t == "$a":
                j = toks.index("$.", i)
                expr = toks[i + 1 : j]
                self.labels[label] = ("$a", (expr, self.mandatory(expr)))
                i = j + 1
            elif t == "$p":
                j = toks.index("$=", i)
                expr = toks[i + 1 : j]
                k = toks.index("$.", j)
                proof = toks[j + 1 : k]
                self.labels[label] = ("$p", (expr, self.mandatory(expr), proof))
                i = k + 1
            else:
                label = t
                i += 1
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
    body = "".join(proof)
    out: list = []
    num = 0
    for c in body:
        if "U" <= c <= "Y":
            num = num * 5 + (ord(c) - ord("U") + 1)
        elif "A" <= c <= "T":
            num = num * 20 + (ord(c) - ord("A") + 1)
            out.append(("step", num - 1))
            num = 0
        elif c == "Z":
            out.append(("save", None))
        elif c == "?":
            out.append(("unknown", None))
    return out


class Incomplete(Exception):
    pass


def expand(mm: MM, label: str):
    kind, (expr, mand, proof) = mm.labels[label]
    if not proof or proof[0] != "(":
        raise Incomplete("uncompressed or empty proof")
    close = proof.index(")")
    ref_labels = proof[1:close]
    ops = decode_compressed(proof[close + 1 :], len(mand), len(ref_labels))

    stack: list = []
    saved: list = []
    trace: list = []
    for kind_op, idx in ops:
        if kind_op == "unknown":
            raise Incomplete("proof contains ?")
        if kind_op == "save":
            saved.append(list(stack[-1]))
            continue
        if idx < len(mand):
            hk, hl, hd = mand[idx]
            e = [hd[0], hd[1]] if hk == "$f" else list(hd)
            stack.append(e)
            trace.append((hl, list(e), True))
            continue
        idx -= len(mand)
        if idx < len(ref_labels):
            lbl = ref_labels[idx]
            lk, ld = mm.labels[lbl]
            if lk == "$f":
                e = [ld[0], ld[1]]
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            if lk == "$e":
                e = list(ld)
                stack.append(e)
                trace.append((lbl, list(e), True))
                continue
            sexpr, smand = ld[0], ld[1]
            k = len(smand)
            if k > len(stack):
                raise Incomplete("stack underflow")
            args = stack[-k:] if k else []
            del stack[len(stack) - k :]
            subst = {}
            for (hk, hl, hd), arg in zip(smand, args):
                if hk == "$f":
                    tc, v = hd
                    if not arg or arg[0] != tc:
                        raise Incomplete("typecode mismatch")
                    subst[v] = arg[1:]
            res = apply_subst(sexpr, subst)
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
