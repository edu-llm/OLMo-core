"""Parse MetaMathQA rows into steps + final answer (no LLM)."""

from __future__ import annotations

import hashlib
import re
from typing import Any


CALC_RE = re.compile(r"<<.*?>>")
# MetaMath often ends with "The answer is: X" or GSM-style #### X
FINAL_PATTERNS = [
    re.compile(r"The answer is:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"The answer is\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"####\s*(.+)"),
]


def family_id_from_original(original_question: str) -> str:
    h = hashlib.sha1(original_question.strip().encode("utf-8")).hexdigest()[:12]
    return f"mm_{h}"


def extract_final_answer(response: str) -> str | None:
    text = response.strip()
    for pat in FINAL_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        ans = m.group(1).strip()
        # Trim trailing punctuation / quotes
        ans = ans.strip().rstrip(".")
        ans = ans.strip("`\"'")
        # Prefer last numeric token if the capture is noisy
        num = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", ans)
        if num and (ans == num.group(0) or len(ans) <= 32):
            return num.group(0).replace(",", "")
        if ans:
            return ans
    return None


def split_steps(response: str) -> list[str]:
    """Split CoT body (before final-answer line) into steps."""
    body = response.strip()
    for pat in FINAL_PATTERNS:
        m = pat.search(body)
        if m:
            body = body[: m.start()].strip()
            break

    cleaned_lines: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        # Drop leading "Step N:" / "N." numbering noise into content
        line = re.sub(r"^(?:step\s*)?\d+[\.:)\]]\s*", "", line, flags=re.IGNORECASE)
        line = CALC_RE.sub("", line).strip()
        line = re.sub(r"\s+", " ", line)
        if line:
            cleaned_lines.append(line)

    if not cleaned_lines:
        return []
    if len(cleaned_lines) >= 2:
        return cleaned_lines

    text = cleaned_lines[0]
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return parts if len(parts) >= 2 else [text]


def parse_metamath_row(
    row: dict[str, Any],
    *,
    instance_id: str,
    split: str,
) -> dict[str, Any] | None:
    query = (row.get("query") or "").strip()
    response = (row.get("response") or "").strip()
    original = (row.get("original_question") or query).strip()
    typ = (row.get("type") or "").strip()
    if not query or not response:
        return None

    steps = split_steps(response)
    final = extract_final_answer(response)
    if not final or not steps:
        return None

    return {
        "family_id": family_id_from_original(original),
        "instance_id": instance_id,
        "split": split,
        "source": "meta-math/MetaMathQA",
        "type": typ,
        "question": query,
        "original_question": original,
        "steps": steps,
        "final_answer": final,
        "answer_raw": response,
        "n_steps": len(steps),
    }
