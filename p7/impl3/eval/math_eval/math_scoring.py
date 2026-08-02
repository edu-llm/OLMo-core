"""Pure answer-extraction / equivalence helpers for the math+logic probe.

Kept as a side-effect-free module because two callers need it: ``score_results.py`` (per-run
tables) and ``sweep_ckpt_eval.py`` (every checkpoint). They must agree exactly on what counts as
a correct answer, so the extraction regexes live in one place rather than being copied. The
original grader did its work at module scope, which meant importing it to reuse ``extract`` and
``check`` ran a whole grading pass as a side effect — hence the split.

  - int (GSM8K/AIME): parse the final integer, exact match.
  - mc  (BBH)       : parse the choice letter (A-G), exact match.
  - expr            : legacy MATH-500 path; those items were removed from the prompt set, so
                      ``check`` can still return None ("unsure") but never does in practice.
"""
import re

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False


def last_boxed(s):
    i = s.rfind(r"\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def extract(resp, atype):
    resp = resp or ""
    if atype == "mc":
        box = last_boxed(resp)
        if box:
            lm = re.findall(r"[A-G]", box.upper())
            if lm:
                return lm[-1]
        m = re.findall(r"answer is\s*\(?\s*([A-G])\s*\)?", resp, re.I) or \
            re.findall(r"final answer\s*[:\-]?\s*\(?\s*([A-G])\s*\)?", resp, re.I) or \
            re.findall(r"\(([A-G])\)", resp) or re.findall(r"\b([A-G])\b", resp[-40:])
        return m[-1].upper() if m else None
    box = last_boxed(resp)
    cand = box
    if cand is None:
        m = re.findall(r"(?:final answer|answer)\s*(?:is|:)?\s*\$?([^\n$]+)", resp, re.I)
        cand = m[-1].strip() if m else None
    if atype == "int":
        src = cand if cand is not None else resp
        nums = re.findall(r"-?\d[\d,]*", src)
        return nums[-1].replace(",", "") if nums else None
    return cand  # expr


def norm_expr(x):
    if x is None:
        return None
    x = x.strip().strip("$").strip()
    x = x.replace(r"\left", "").replace(r"\right", "").replace(r"\!", "").replace(r"\,", "")
    x = x.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    x = re.sub(r"\\text\{[^}]*\}", "", x)
    x = x.replace(" ", "").replace(r"\%", "").rstrip(".")
    x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
    return x


def sympy_eq(a, b):
    if not HAVE_SYMPY:
        return None
    try:
        return bool(sympy.simplify(parse_latex(a) - parse_latex(b)) == 0)
    except Exception:
        try:
            return sympy.nsimplify(a) == sympy.nsimplify(b)
        except Exception:
            return None


def check(atype, gold, cand):
    """True / False, or None meaning "unsure, needs a verifier" (only reachable for 'expr')."""
    if cand is None:
        return None if atype == "expr" else False
    if atype == "int":
        try:
            return int(float(cand)) == int(float(gold))
        except Exception:
            return False
    if atype == "mc":
        g = re.sub(r"[()]", "", gold).strip().upper()
        return cand.strip().upper() == g
    if norm_expr(cand) == norm_expr(gold):
        return True
    if sympy_eq(cand, gold) is True:
        return True
    return None


def score(response, meta):
    """Convenience wrapper: is ``response`` correct for a prompt row? Unsure counts as wrong."""
    return bool(check(meta["answer_type"], meta["gold"], extract(response, meta["answer_type"])))
