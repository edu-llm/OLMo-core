"""Segments, documents, and the one place a fact value gets wrapped for lookup.

A `Segment` is `(text, masked)` where `masked=True` means loss OFF. A `Doc`
carries a *role-tagged* span list rather than two pre-rendered arms, which is the
structural change from the previous generation: the token stream is produced once
and the arms differ only in which loss-weight sidecar is applied to it. See
`memsplit.masking`.

Why that matters: rendering each arm separately makes the two streams different
lengths, so at a fixed token budget the arms see different numbers of fact
exposures (measured previously at 0.50x for biographies). Matching tokens and
matching exposures then become mutually exclusive, and the previous generation
claimed both. One stream removes the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memsplit.tokenizer import DB_END, DB_RETRIEVE, DB_START

Segment = tuple[str, bool]

# Span roles. `payload` is a retrieved fact value -- the only role the split
# condition masks. `query` is the lookup key, which stays supervised in every
# condition because issuing the query is the skill under test. `restate` is a
# post-retrieval copy of a value that is already in context; it is supervised in
# every condition and must be reported as such, because it is the one place the
# split arm receives gradient on value tokens.
ROLES = ("plain", "query", "payload", "restate", "control", "cue")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    role: str
    fact_id: str | None = None
    depth: int | None = None


@dataclass
class Doc:
    kind: str
    segments: list[Segment]
    roles: list[str]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.segments) != len(self.roles):
            raise ValueError("segments and roles must be parallel")
        bad = set(self.roles) - set(ROLES)
        if bad:
            raise ValueError(f"unknown roles: {sorted(bad)}")

    def text(self) -> str:
        return "".join(t for t, _ in self.segments)


@dataclass
class QAItem:
    task: str
    prompt: str
    answer: str
    meta: dict = field(default_factory=dict)


def lookup_segments(key_subject: str, relation: str, value: str) -> list[Segment]:
    """The exact wrapping of one externalised fact value.

    Loss stays ON for the control tokens and the query -- the model must learn
    to *ask* -- and OFF only for the value, which the store supplies at
    inference time.

    Note the closing `<|db_end|>` is **supervised**. The previous write-up said
    the mask covered "the value and its closing <|db_end|> token"; the code
    masked only the value. Supervising `<|db_end|>` is the right choice (the
    model must learn to close the span) but it has to be described accurately.
    """
    return [
        (DB_START, False),
        (f"{key_subject}, {relation}", False),
        (DB_RETRIEVE, False),
        (f" {value}", True),
        (DB_END, False),
    ]


def lookup_roles() -> list[str]:
    """Roles parallel to `lookup_segments`."""
    return ["plain", "query", "plain", "payload", "plain"]


def merge_plain(segments: list[Segment], roles: list[str]) -> tuple[list[Segment], list[str]]:
    """Coalesce adjacent same-role unmasked `plain` segments.

    Only `plain` runs are merged. Role boundaries are preserved even when both
    sides are unmasked, because the masker needs them to place controls and to
    find cue windows.
    """
    out_s: list[Segment] = []
    out_r: list[str] = []
    for (text, masked), role in zip(segments, roles):
        if not text:
            continue
        if (
            out_s
            and role == "plain"
            and out_r[-1] == "plain"
            and not masked
            and not out_s[-1][1]
        ):
            out_s[-1] = (out_s[-1][0] + text, False)
        else:
            out_s.append((text, masked))
            out_r.append(role)
    return out_s, out_r


def spans_from_roles(tok, segments: list[Segment], roles: list[str]) -> tuple[list[int], list[Span]]:
    """Encode segments and return (ids, role-tagged token spans).

    This is the bridge from document construction to masking: everything
    downstream operates on token indices, so mask conditions can be derived
    without re-rendering text.
    """
    ids: list[int] = []
    spans: list[Span] = []
    for (text, _), role in zip(segments, roles):
        if not text:
            continue
        seg = tok.encode(text)
        if not seg:
            continue
        spans.append(Span(start=len(ids), end=len(ids) + len(seg), role=role))
        ids.extend(seg)
    return ids, spans
