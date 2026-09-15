#!/usr/bin/env python3
"""Canonical eval-item identity: normalization, preamble stripping, hashing.

Shared deliberately. The contamination scan searches the corpus for strings
derived from eval items, and the training-time evaluator records a hash of the
items it scored. Those two must agree exactly or the join between a
contamination hit and a per-item bpb number is silently wrong -- and a join
that is off by one produces entirely plausible numbers, which is the failure
mode this campaign keeps hitting. Two copies of "the same" normalizer is the
obvious way to get there, so there is one copy and both callers import it.

Stdlib only, so it is testable without torch, olmo_core or ai2-olmo -- none of
which can be imported on a laptop or a CPU node.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# NFKC, lowercase, strip non-word characters, collapse whitespace. Changing
# any of this invalidates every index and every recorded hash, which is why
# `normalizer_fingerprint()` exists and is asserted by the scanner's canary.
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")

MIN_WIDTH = 8
# How many items must share a prefix before it is treated as a few-shot
# exemplar block rather than an incidental overlap between two questions.
MIN_SHARERS = 5
FULL_WIDTH = 13


def normalize(text: str) -> str:
    """Canonical normalization for both the index and the recorded hashes."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalizer_fingerprint() -> str:
    """Hash of the normalizer's own behavior on a fixed probe.

    Not a hash of the source: whitespace and comments would change it without
    changing behavior, and a behavioral change is the only thing that matters.
    The probe deliberately exercises NFKC folding, case, punctuation and
    repeated whitespace.
    """
    probes = [
        "The Quick--Brown FOX, jumps over  the lazy dog's back; again and AGAIN!",
        "ﬁne Ångström  café—naïve",
        "a  b\tc\nd",
        "",
    ]
    # A visible, non-empty separator: joining on an empty string would let
    # two different probe sets hash identically ("a b"+"c" vs "a"+"bc").
    return sha256(" >|< ".join(normalize(p) for p in probes))


def common_prefix(strings: list[str]) -> str:
    """Longest common prefix across a label's contexts, i.e. its exemplars.

    Every item in an OLMES 5-shot label carries the same five exemplars ahead
    of its own stem. Indexing them would make every item in the label match
    whenever any exemplar text appears in the corpus -- a false-positive source
    the size of the whole benchmark. They are shared, so the common prefix
    identifies them without parsing the prompt format.
    """
    if not strings:
        return ""
    prefix = strings[0]
    for candidate in strings[1:]:
        limit = min(len(prefix), len(candidate))
        i = 0
        while i < limit and prefix[i] == candidate[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix


def strip_preamble(contexts: list[str]) -> tuple[list[str], str]:
    """Return (stems, preamble). A short accidental overlap is not a preamble."""
    preamble = common_prefix(contexts)
    if len(preamble.split()) < MIN_WIDTH:
        return list(contexts), ""
    cut = len(preamble)
    return [c[cut:] for c in contexts], preamble


def shared_word_prefix(a: list[str], b: list[str]) -> int:
    """Number of leading WORDS two word lists share.

    Words rather than characters, because a character-level common prefix is
    the wrong unit: "alpha beta" and "alphabet soup" share the characters
    "alpha" while sharing no whole word, and cutting there leaves the
    fragment "beta" as the indexed stem. Deciding cleanliness from one side
    alone does not help -- the run ends at a word boundary in one string and
    mid-word in the other.
    """
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def strip_shared_prefixes(contexts: list[str]) -> tuple[list[str], list[int]]:
    """Strip each item's few-shot exemplar block, found via sorted neighbours.

    `strip_preamble` only finds a prefix shared by EVERY item in a label. That
    is right for most labels, but wrong for MMLU: its five exemplars are drawn
    per subject, so items share a block with their own subject and not across
    subjects. The label-wide prefix is then a few words, nothing is stripped,
    and ~95% of an MMLU "stem" is exemplar text -- which would be indexed as
    if it were the item, so one mirrored copy of a subject's dev examples
    would match every item in that subject.

    Sorting puts items that share a prefix next to each other, so an exemplar
    block shows up as a run of consecutive items sharing a long prefix. This
    is format-agnostic: it needs no knowledge of how a prompt separates
    exemplars from the question, which differs by benchmark, and it handles
    the label-wide case as well as the per-subject one.

    Two guards keep it from over-trimming. The prefix must be shared by a run
    of MIN_SHARERS+1 items, so two questions that happen to open with the same
    few words are left alone -- shortening those would only cost recall, and
    could push an item under the assessability floor. And it must be at least
    MIN_WIDTH words to count at all. Cuts are taken in word space, because a
    character-level prefix can end mid-word ("alpha beta" against "alphabet
    soup") and leave a fragment as the indexed stem.

    :returns: (stems in the input order, words stripped per item).
    """
    n = len(contexts)
    if n == 0:
        return [], []
    words = [text.split() for text in contexts]
    order = sorted(range(n), key=lambda i: contexts[i])
    stems = list(contexts)
    stripped = [0] * n

    # Adjacent shared-prefix lengths in sorted order. A run of k+1 consecutive
    # items all sharing p words is exactly a stretch of k adjacent values all
    # >= p, so the longest prefix shared by MIN_SHARERS+1 items containing a
    # given item is the best sliding-window minimum over windows covering it.
    #
    # Requiring a whole run, rather than just comparing to the item
    # MIN_SHARERS positions away, is what handles subject boundaries: the last
    # item of a subject would otherwise be compared against the first item of
    # the next one and find nothing shared.
    adjacent = [
        shared_word_prefix(words[order[j]], words[order[j + 1]]) for j in range(n - 1)
    ]

    for rank, index in enumerate(order):
        best = 0
        lo = max(0, rank - MIN_SHARERS)
        hi = min(rank, n - 1 - MIN_SHARERS)
        for start in range(lo, hi + 1):
            window = adjacent[start : start + MIN_SHARERS]
            if window:
                best = max(best, min(window))
        if best < MIN_WIDTH:
            continue
        # Never strip an item down to nothing: a duplicate pair would share
        # every word, and an empty stem is not a measurement.
        if best >= len(words[index]):
            continue
        stems[index] = " ".join(words[index][best:])
        stripped[index] = best
    return stems, stripped


def item_identity(raw_context: str, raw_gold: str) -> dict[str, str | int]:
    """Identity record for one item, given its already-stem-stripped context."""
    stem = normalize(raw_context)
    gold = normalize(raw_gold)
    return {
        "stem_sha256": sha256(stem),
        "gold_sha256": sha256(gold),
        "stem_words": len(stem.split()),
        "gold_words": len(gold.split()),
    }


def identities_for_label(
    contexts: list[str], golds: list[str]
) -> tuple[list[dict[str, str | int]], str]:
    """Identity records for a whole label, with the shared preamble removed.

    The preamble must be stripped across the label as a unit, which is why
    this takes the whole label rather than one item at a time.
    """
    if len(contexts) != len(golds):
        raise ValueError(f"{len(contexts)} contexts against {len(golds)} golds")
    stems, preamble = strip_preamble(contexts)
    return [item_identity(s, g) for s, g in zip(stems, golds)], preamble
