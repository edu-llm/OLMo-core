"""Shared formatting for bare / complete / faded scaffold arms."""

from __future__ import annotations


def parse_fade_levels(s: str) -> list[float]:
    levels = [float(x.strip()) for x in s.split(",") if x.strip()]
    for x in levels:
        if not 0.0 <= x <= 1.0:
            raise SystemExit(f"fade level out of range: {x}")
    return levels


def format_bare(q: str, final: str) -> str:
    return f"Problem: {q}\nAnswer: {final}\n"


def format_complete(q: str, steps: list[str], final: str) -> str:
    sol = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return f"Problem: {q}\nSolution:\n{sol}\nAnswer: {final}\n"


def scaffold_prefix(steps: list[str], frac: float) -> tuple[list[str], list[str]]:
    """Return (shown_steps, hidden_steps) for a fade fraction of steps shown."""
    n = len(steps)
    if n == 0:
        return [], []
    k = int(round(frac * n))
    k = max(0, min(n, k))
    if frac <= 0.0:
        k = 0
    elif frac >= 1.0:
        k = n
    return steps[:k], steps[k:]


def format_fade(q: str, shown: list[str], hidden: list[str], final: str) -> dict:
    """
    Document text with a clear marker. For CPT with loss masking, train only on
    the continuation after BEGIN_CONTINUE (hidden steps + answer).
    """
    shown_txt = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(shown)) if shown else "(none)"
    hidden_txt = "\n".join(f"{len(shown) + i + 1}. {s}" for i, s in enumerate(hidden))
    context = f"Problem: {q}\nPartial solution:\n{shown_txt}\nBEGIN_CONTINUE\n"
    if hidden:
        target = f"{hidden_txt}\nAnswer: {final}\n"
    else:
        target = f"Answer: {final}\n"
    text = context + target
    return {
        "text": text,
        "context": context,
        "target": target,
        "loss_start_char": len(context),
        "n_shown": len(shown),
        "n_hidden": len(hidden),
    }
