"""Parse GSM8K answer strings into steps + final numeric answer."""

from __future__ import annotations

import re
from typing import Any


CALC_RE = re.compile(r"<<.*?>>")
FINAL_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def extract_final_answer(answer: str) -> str | None:
    m = FINAL_RE.search(answer)
    if not m:
        return None
    return m.group(1).replace(",", "")


def split_steps(answer: str) -> list[str]:
    """
    Split the solution body (before ####) into steps.
    Prefer blank-line / newline boundaries; strip calculator <<...>> annotations for display
    but keep readable arithmetic text.
    """
    body = answer.split("####")[0].strip()
    # Remove calculator channel but keep the human-readable side when present:  "x = <<a+b=c>>c"
    cleaned_lines: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop pure annotation residue; keep text
        line = CALC_RE.sub("", line).strip()
        # Collapse leftover double spaces
        line = re.sub(r"\s+", " ", line)
        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return []

    # If many short lines, each line is a step; else split on ". " sentence-ish
    if len(cleaned_lines) >= 2:
        return cleaned_lines

    # Single blob: try sentence split
    text = cleaned_lines[0]
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts if len(parts) >= 2 else [text]


def parse_gsm8k_row(row: dict[str, Any], *, family_id: str, instance_id: str, split: str) -> dict[str, Any]:
    question = row["question"].strip()
    answer_raw = row["answer"].strip()
    steps = split_steps(answer_raw)
    final = extract_final_answer(answer_raw)
    return {
        "family_id": family_id,
        "instance_id": instance_id,
        "split": split,
        "source": "openai/gsm8k",
        "question": question,
        "steps": steps,
        "final_answer": final,
        "answer_raw": answer_raw,
        "n_steps": len(steps),
    }
