#!/usr/bin/env python
"""Deterministic IFEval instruction checkers (Zhou et al. 2023, arXiv:2311.07911).

This is the paper's ("RL's Razor") instruction-following metric: each prompt carries one
or more *verifiable* instructions, and compliance is checked programmatically — no LLM
judge, no subagents. We reproduce the official strict/loose scoring:

* strict  — check the response as generated.
* loose   — also accept if any of the standard response transforms passes (strip markdown
            ``*``, drop the first/last line), matching the reference implementation.

Each checker takes ``(response: str, **kwargs)`` and returns ``bool``. ``check(instruction_id,
response, kwargs)`` dispatches by id. Unknown ids raise ``KeyError`` so a prompt set can't
silently score instructions we don't implement.
"""
import json
import re

_WORD_RE = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s|$)")


def _words(text):
    return _WORD_RE.findall(text)


def _count_sentences(text):
    parts = [s for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return len(parts)


def _relation_ok(count, relation, target):
    if relation == "at least":
        return count >= target
    if relation == "less than":
        return count < target
    if relation == "exactly":
        return count == target
    if relation == "at most":
        return count <= target
    raise ValueError(f"unknown relation {relation!r}")


# --- keywords -------------------------------------------------------------------
def _kw_existence(r, keywords, **_):
    low = r.lower()
    return all(k.lower() in low for k in keywords)


def _kw_frequency(r, keyword, frequency, relation="at least", **_):
    n = len(re.findall(re.escape(keyword.lower()), r.lower()))
    return _relation_ok(n, relation, frequency)


def _kw_forbidden(r, forbidden_words, **_):
    low = r.lower()
    return not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in forbidden_words)


def _kw_letter_frequency(r, letter, let_frequency, let_relation="at least", **_):
    n = r.lower().count(letter.lower())
    return _relation_ok(n, let_relation, let_frequency)


# --- length ---------------------------------------------------------------------
def _len_words(r, num_words, relation="at least", **_):
    return _relation_ok(len(_words(r)), relation, num_words)


def _len_sentences(r, num_sentences, relation="at least", **_):
    return _relation_ok(_count_sentences(r), relation, num_sentences)


def _paragraphs(r):
    return [p for p in re.split(r"\n\s*\n", r.strip()) if p.strip()]


def _len_paragraphs(r, num_paragraphs, **_):
    return len(_paragraphs(r)) == num_paragraphs


def _nth_paragraph_first_word(r, num_paragraphs, nth_paragraph, first_word, **_):
    paras = _paragraphs(r)
    if len(paras) != num_paragraphs or nth_paragraph > len(paras):
        return False
    para = paras[nth_paragraph - 1].strip()
    w = _words(para)
    return bool(w) and w[0].lower() == first_word.lower()


# --- detectable content ---------------------------------------------------------
def _placeholders(r, num_placeholders, **_):
    return len(re.findall(r"\[[^\[\]]+\]", r)) >= num_placeholders


def _postscript(r, postscript_marker="P.S.", **_):
    return postscript_marker.lower() in r.lower()


# --- detectable format ----------------------------------------------------------
def _bullets(r, num_bullets, **_):
    n = len(re.findall(r"(?m)^\s*[\*\-]\s+\S", r))
    return n == num_bullets


def _highlights(r, num_highlights, **_):
    n = len(re.findall(r"\*[^\*\n]+\*", r)) + len(re.findall(r"\*\*[^\*\n]+\*\*", r))
    return n >= num_highlights


def _multiple_sections(r, section_spliter, num_sections, **_):
    n = len(re.findall(r"(?mi)^\s*" + re.escape(section_spliter) + r"\b", r))
    return n >= num_sections


def _json_format(r, **_):
    s = r.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s).strip()
    try:
        json.loads(s)
        return True
    except Exception:
        return False


def _title(r, **_):
    return bool(re.search(r"<<[^\n]+>>", r))


def _constrained_response(r, **_):
    opts = ["my answer is yes.", "my answer is no.", "my answer is maybe."]
    return r.strip().lower() in opts


# --- combination ----------------------------------------------------------------
def _two_responses(r, **_):
    parts = [p for p in r.split("******") if p.strip()]
    return len(parts) == 2


def _repeat_prompt(r, prompt_to_repeat, **_):
    return r.strip().lower().startswith(prompt_to_repeat.strip().lower())


# --- start / end ----------------------------------------------------------------
def _end_checker(r, end_phrase, **_):
    return r.strip().lower().endswith(end_phrase.strip().lower())


def _quotation(r, **_):
    s = r.strip()
    return len(s) >= 2 and s.startswith('"') and s.endswith('"')


# --- case / punctuation ---------------------------------------------------------
def _all_capital(r, **_):
    letters = [c for c in r if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _all_lowercase(r, **_):
    letters = [c for c in r if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _capital_word_frequency(r, capital_frequency, capital_relation="at least", **_):
    n = sum(1 for w in _words(r) if len(w) > 1 and w.isupper())
    return _relation_ok(n, capital_relation, capital_frequency)


def _no_comma(r, **_):
    return "," not in r


REGISTRY = {
    "keywords:existence": _kw_existence,
    "keywords:frequency": _kw_frequency,
    "keywords:forbidden_words": _kw_forbidden,
    "keywords:letter_frequency": _kw_letter_frequency,
    "length_constraints:number_words": _len_words,
    "length_constraints:number_sentences": _len_sentences,
    "length_constraints:number_paragraphs": _len_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _bullets,
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "detectable_format:constrained_response": _constrained_response,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
    "change_case:english_capital": _all_capital,
    "change_case:english_lowercase": _all_lowercase,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "punctuation:no_comma": _no_comma,
}


def check_strict(instruction_id, response, kwargs):
    return bool(REGISTRY[instruction_id](response, **(kwargs or {})))


def _loose_variants(response):
    r = response.strip()
    lines = r.split("\n")
    variants = {
        r,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    }
    variants |= {v.replace("*", "").strip() for v in list(variants)}
    return [v for v in variants if v]


def check_loose(instruction_id, response, kwargs):
    fn = REGISTRY[instruction_id]
    return any(bool(fn(v, **(kwargs or {}))) for v in _loose_variants(response))


def known(instruction_id):
    return instruction_id in REGISTRY
