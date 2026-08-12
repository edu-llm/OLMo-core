"""Answer parsing and scoring. One convention, stated, with chance reported.

Two failures in the previous generation trace to this file's absence of
discipline.

**Mixed metrics inside one table.** The headline two-hop table reported dense
closed-book and split with exact match after the last `Answer:`, but dense
open-book with `gold in continuation` -- substring-anywhere over the whole 64-token
continuation, on ~600 items rather than 1000. Since a trace restates the value
several times, that credits a correct trace with a wrong final answer. Three rows,
two metrics, two sample sizes.

**Parsers that fail off-format.** A deduction scorer returned `None` whenever no
`Answer:` tag was emitted, and `max_new` truncation fell asymmetrically on the
long (yes) class, producing **0.369 on a balanced 750/750 task where chance is
exactly 0.500** -- 10.1 standard errors *below* chance, which no model policy can
produce. Meanwhile single-hop probes ended mid-sentence and were scored with the
`Answer:`-tag parser, reading 0.0 everywhere.

So: every scorer here declares its `mode`, and every result carries `chance` and
`n`. `score_items` refuses to mix modes silently.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# NOT re.DOTALL. With DOTALL the first match's `.*` swallows every subsequent
# "Answer:" line, so `matches[-1]` silently becomes `matches[0]` and the parser
# reads the FIRST answer while claiming to read the last. Match to end of line.
_ANSWER_RE = re.compile(r"Answer:[ \t]*([^\n]*)")

MODES = ("answer_tag_exact", "leading_continuation", "contains")


def normalize_answer(text: str) -> str:
    text = " ".join(text.strip().lower().split())
    return text.rstrip(".")


def parse_answer(generation: str) -> str | None:
    """Text after the LAST `Answer:`, or None if the tag never appears.

    Returning None rather than "" is deliberate: an unparseable generation is a
    third outcome, not a wrong answer, and conflating them is what produced a
    below-chance score on a balanced task.
    """
    matches = list(_ANSWER_RE.finditer(generation))
    if not matches:
        return None
    return matches[-1].group(1).strip()


@dataclass
class Scored:
    correct: bool
    parsed: str | None
    unparseable: bool


def score_one(generation: str, gold: str, mode: str = "answer_tag_exact") -> Scored:
    """Score one generation. `mode` is explicit; there is no default-by-accident.

    * `answer_tag_exact`      -- exact match on the text after the last `Answer:`.
                                 The convention for generative QA endpoints.
    * `leading_continuation`  -- the continuation must *begin* with the gold value.
                                 For probes that end mid-sentence ("X's city is").
    * `contains`              -- gold appears anywhere. Lenient; only for
                                 diagnostics, never for a headline table, and it
                                 must be labelled in the output.
    """
    if mode not in MODES:
        raise ValueError(f"unknown scoring mode {mode!r}; have {MODES}")
    g = normalize_answer(gold)
    if mode == "answer_tag_exact":
        parsed = parse_answer(generation)
        if parsed is None:
            return Scored(False, None, True)
        return Scored(normalize_answer(parsed) == g, parsed, False)
    if mode == "leading_continuation":
        cont = normalize_answer(generation)
        return Scored(bool(g) and cont.startswith(g), generation.strip()[:80], False)
    cont = normalize_answer(generation)
    return Scored(bool(g) and g in cont, generation.strip()[:80], False)


def best_constant_accuracy(golds: list[str]) -> tuple[float, str]:
    """Accuracy of always answering the single most common gold, and that gold.

    This is the floor a claim must clear, and it is not the same as 1/|pool|. On a
    same-city question family with a 40% yes-rate, a constant "no" scores 60% --
    and two arms sitting at 59.5% and 40.5% are simply the two constant policies,
    not a 19-point effect. Report it on every table.
    """
    if not golds:
        return 0.0, ""
    counts: dict[str, int] = {}
    for g in golds:
        key = normalize_answer(g)
        counts[key] = counts.get(key, 0) + 1
    top, n = max(counts.items(), key=lambda kv: kv[1])
    return n / len(golds), top


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Behaves near 0 and 1 where the normal one does not.

    Used instead of the CLT interval because coverage simulations find CLT (and
    bootstrap) intervals unreliable below a few hundred items, while Wilson holds.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_items(
    generations: list[str],
    golds: list[str],
    mode: str = "answer_tag_exact",
    chance: float | None = None,
) -> dict:
    """Score a set and return everything a table needs, including the floors.

    The returned dict always carries `n`, `chance`, `best_constant`,
    `unparseable_rate`, a Wilson interval, and `z_vs_chance`. If `z_vs_chance` is
    strongly negative the endpoint is broken, not the model -- that is the check
    that would have caught a 0.369 on a balanced task.
    """
    if len(generations) != len(golds):
        raise ValueError("generations and golds must be parallel")
    rows = [score_one(g, y, mode) for g, y in zip(generations, golds)]
    n = len(rows)
    k = sum(r.correct for r in rows)
    acc = k / n if n else 0.0
    bc, bc_label = best_constant_accuracy(golds)
    lo, hi = wilson_interval(k, n)

    out = {
        "mode": mode,
        "n": n,
        "accuracy": acc,
        "wilson95": [lo, hi],
        "n_correct": k,
        "unparseable_rate": sum(r.unparseable for r in rows) / n if n else 0.0,
        "best_constant": bc,
        "best_constant_label": bc_label,
        "beats_best_constant": acc > bc,
        "chance": chance,
    }
    if chance is not None and n:
        se = math.sqrt(max(chance * (1 - chance), 1e-12) / n)
        out["z_vs_chance"] = (acc - chance) / se
        # A balanced task scored many SE below chance is a parser failure. The
        # threshold is deliberately loose; the point is to make it impossible to
        # publish such a number without having been told.
        out["below_chance_implausible"] = out["z_vs_chance"] < -3.0
    return out, rows
