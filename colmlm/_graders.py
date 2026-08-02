"""LLM graders for generative fact-recall evals (SimpleQA). Lazy imports; require an API key.

The Co-LMLM paper grades SimpleQA with gpt-4.1 (App. A.6). This wraps OpenAI or Gemini behind a
single ``simpleqa_grade`` call that returns True iff the prediction is graded CORRECT.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

_SIMPLEQA_TEMPLATE = """You are grading a predicted answer against the gold answer(s) for a \
fact-seeking question. Reply with a single letter:
A = CORRECT (the prediction states the gold answer, with no contradiction),
B = INCORRECT (contradicts or omits the gold answer),
C = NOT_ATTEMPTED (no definite answer given).

Question: {question}
Gold answer(s): {gold}
Predicted answer: {prediction}

Grade (A/B/C):"""


def simpleqa_grade(
    *,
    question: str,
    gold: Sequence[str],
    prediction: str,
    backend: str = "openai",
    model: Optional[str] = None,
) -> bool:
    """Return True iff the LLM grader labels ``prediction`` CORRECT for the question/gold."""
    prompt = _SIMPLEQA_TEMPLATE.format(
        question=question, gold="; ".join(gold), prediction=prediction
    )
    if backend == "openai":
        letter = _openai_complete(prompt, model or "gpt-4.1")
    elif backend == "gemini":
        letter = _gemini_complete(prompt, model or "gemini-2.5-flash")
    else:
        raise ValueError(f"unknown grader backend: {backend}")
    return letter.strip().upper().startswith("A")


def _openai_complete(prompt: str, model: str) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("SimpleQA grading needs OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai for the SimpleQA grader.") from e
    resp = OpenAI().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=1
    )
    return resp.choices[0].message.content or ""


def _gemini_complete(prompt: str, model: str) -> str:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("SimpleQA grading needs GEMINI_API_KEY.")
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError("pip install google-genai for the SimpleQA grader.") from e
    resp = genai.Client().models.generate_content(model=model, contents=prompt)
    return resp.text or ""
