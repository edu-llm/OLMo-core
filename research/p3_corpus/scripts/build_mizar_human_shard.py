"""Build a raw shard directly from the pinned official human Mizar sources.

This production path intentionally has no legacy ``html2`` input. Official
``.miz`` files supply exact human proof targets, while the accepted current
semantic index supplies canonical identities, goals, and global fact statements.
Only raw rows are written; the pooled MML planner owns the final split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import shutil
import sqlite3
import sys
import time
from array import array
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from mizar_current_index import (
    INDEX_SCHEMA,
    SQLITE_APPLICATION_ID,
    SQLITE_USER_VERSION,
    MizarIndex,
    MizarIndexError,
    SourceVerificationError,
    _comparison_key,
    verify_source_manifest,
)
from mizar_current_index import (
    _canonical_source_goal as _adapter_source_goal,
)

MIZAR_VERSION = "8.1.15"
MML_VERSION = "5.94.1493"
ROW_SCHEMA = "mizar-proof-v2"
FACT_ORDER_POLICY_ID = "mizar-human-proof-v1"
FAMILY_SOURCE_MANIFEST_SCHEMA = "p3-family-source-manifest/v2"
BUILD_MANIFEST_SCHEMA = "mizar-human-raw-build-v1"
BUILD_REPORT_SCHEMA = "mizar-human-build-report-v1"
BUILD_SOURCE_SCHEMA = "mizar-build-source-v2"
MASK_SCHEMA = "character-prefix-mask-v1"
SOURCE_INDEX_BINDING_SCHEMA = "mizar-source-index-binding-v1"
SECONDARY_ALIGNMENT_METHOD = "unique-label-between-proof-hash-anchors-v1"

HDR = "I know these mathematical statements:"
SEP = "---"

QWEN_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B"
QWEN_EOS_TOKEN = "<|endoftext|>"
QWEN_EOS_TOKEN_ID = 151643
APPROVED_TOKENIZER_JSON_SHA256 = (
    "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
)
APPROVED_TOKENIZER_CONFIG_SHA256 = (
    "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
)
APPROVED_TOKENIZER_BEHAVIOR_SHA256 = (
    "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
)
APPROVED_TOKENIZERS_VERSION = "0.22.2"
MAX_TOKENS_WITH_EOS = 16_384
REPLAY_SAMPLE_SIZE = 100

EXPECTED_MML_FILES = 1_500
EXPECTED_STATEMENTS = 106_317
EXPECTED_THEOREMS = 91_114
EXPECTED_SOURCE_DECLARATIONS = 75_158
EXPECTED_TOKEN_LENGTHS_SHA256 = (
    "7a1746b0b21078bc45b3870fd8fa1aa286512fe812161cabd3261a865afea17b"
)
EXPECTED_DISTINCT_FACTS = 42_050
EXPECTED_PRIMARY_ROWS = 50_114
EXPECTED_PRIMARY_RAW_SHA256 = (
    "0d563be6fae81cd21b551c422378792ef4daad454cab9ed86bb751f127daefd1"
)
EXPECTED_PRIMARY_TOKENS = 32_905_127
EXPECTED_RECOVERED_ROWS = 5_239
EXPECTED_RECOVERED_TOKENS = 9_946_266
EXPECTED_TOTAL_TOKENS = 42_851_393
EXPECTED_RECOVERED_IDENTITY_SOURCE_ORDER_SHA256 = (
    "6d113a43ff0b0af8aae13325908d2507b9b63aadcc01d50c37d73e29549396fa"
)
EXPECTED_RECOVERED_IDENTITY_SET_SHA256 = (
    "048f47cf87e6eaeccf87f3aafb202236373dea000719ae221c5ee33896dad8cd"
)
EXPECTED_RECOVERED_SOURCE_BINDING_SHA256 = (
    "790c86db30604c5836be70e28df527bb6c1a41b30620cfaf327122db047be65c"
)
EXPECTED_RECOVERED_TOKEN_SEQUENCE_SHA256 = (
    "3391cb491f1e7e8ec23b7725d27ceb95b4d5d51bd5856a8e46372507102d5ca4"
)
EXPECTED_RECOVERED_TEXT_HASH_SEQUENCE_SHA256 = (
    "e116af514ee5cc7fc3415d01a68ec42037206849f3b62e6bdddbfefe4637659f"
)

TOKENIZER_BEHAVIOR_PROBES = (
    "",
    "TACTIC\nby simp\nSTATE_AFTER\nno goals",
    r"\<lbrakk>A; B\<rbrakk> \<Longrightarrow> A",
    'proof "sorry" (* oops *)',
    "Unicode: ∀x∈ℝ. x² ≥ 0",
    "a b b b",
    "---\nGOAL\nSTATE_BEFORE",
    "<|endoftext|>",
)

THEOREM_START_RE = re.compile(r"(?m)^[ \t]*theorem\b", re.IGNORECASE)
WORD_OR_SEMICOLON_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b|;")
LABEL_TOKEN = r"[A-Za-z0-9_]+"
THEOREM_LABEL_RE = re.compile(
    rf"^\s*theorem\s+(?P<label>{LABEL_TOKEN})\s*:(?![=\-])",
    re.IGNORECASE,
)
INLINE_JUSTIFICATION_RE = re.compile(
    r"\s+(?:by|from)\s+[^;]*;\s*$",
    re.IGNORECASE | re.DOTALL,
)
JUSTIFICATION_RE = re.compile(r"\b(?:by|from)\b", re.IGNORECASE)
PROOF_LOCAL_LABEL_RE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<label>{LABEL_TOKEN})\s*:(?![=\-])"
)
LOCAL_CONTEXT_RE = re.compile(rf"""(?isx)
    (?:^|;)\s*
    (?P<kind>assume|suppose|given)\s+
    (?:(?P<label>{LABEL_TOKEN})\s*:(?![=\-]))?
    (?P<statement>.*?)
    ;
    """)
NUMERIC_REFERENCE_RE = re.compile(
    r"\b(?P<article>[A-Z][A-Z0-9_]*):\s*"
    r"(?:(?P<kind>def|sch)\s*_?\s*(?P<special>[1-9]\d*)|"
    r"(?P<number>[1-9]\d*))(?![A-Za-z0-9_])"
    r"(?P<tail>(?:\s*,\s*(?:(?:def|sch)\s*_?\s*)?"
    r"[1-9]\d*(?![A-Za-z0-9_]))*)",
    re.IGNORECASE,
)
QUALIFIED_LABEL_RE = re.compile(
    rf"\b(?P<article>[A-Z][A-Z0-9_]*):(?P<label>{LABEL_TOKEN})\b",
    re.IGNORECASE,
)
BARE_REFERENCE_RE = re.compile(rf"(?<![A-Za-z0-9_]){LABEL_TOKEN}(?![A-Za-z0-9_])")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
BLOCK_OPENERS = frozenset({"proof", "now", "hereby", "suppose", "case"})

COUNTER_KEYS = (
    "source_files",
    "declarations_total",
    "complete_explicit_declarations",
    "mapped_complete_declarations",
    "dropped_canceled",
    "dropped_inline_justification",
    "dropped_no_explicit_proof",
    "dropped_malformed_declaration",
    "dropped_malformed_explicit_proof",
    "dropped_source_index_unanchored",
    "dropped_unresolved_reference",
    "dropped_no_global_citation",
    "dropped_duplicate",
    "dropped_overlength",
    "recovered_unique_label",
    "accepted_rows",
)
BASE_PRODUCTION_COUNTERS = {
    "source_files": 1_500,
    "declarations_total": 75_158,
    "complete_explicit_declarations": 67_863,
    "mapped_complete_declarations": 58_356,
    "dropped_canceled": 0,
    "dropped_inline_justification": 5_258,
    "dropped_no_explicit_proof": 2_036,
    "dropped_malformed_declaration": 0,
    "dropped_malformed_explicit_proof": 1,
    "dropped_source_index_unanchored": 9_507,
    "dropped_unresolved_reference": 5_645,
    "dropped_no_global_citation": 2_575,
    "dropped_duplicate": 12,
    "dropped_overlength": 10,
    "accepted_rows": 50_114,
}
EXPECTED_PRODUCTION_COUNTERS = {
    **BASE_PRODUCTION_COUNTERS,
    "mapped_complete_declarations": 63_595,
    "dropped_source_index_unanchored": 4_268,
    "recovered_unique_label": EXPECTED_RECOVERED_ROWS,
    "accepted_rows": 55_353,
}


class BuildError(RuntimeError):
    """A fail-closed production build or self-check error."""


class SourceIndexMismatch(BuildError):
    """Official source declarations cannot be safely aligned to the index."""


@dataclass(frozen=True)
class SourceTheorem:
    """One bounded literal theorem declaration from an official article."""

    article: str
    source_file: str
    ordinal: int
    label: str | None
    category: str
    source_goal: str
    index_source_goal: str
    target: str | None
    local_assumptions: dict[str, str]
    declaration_start: int
    declaration_end: int
    target_start: int
    target_end: int
    line_start: int
    line_end: int
    source_declaration: str

    @property
    def target_sha256(self) -> str | None:
        """Return the exact target digest when this declaration has a target."""

        if self.target is None:
            return None
        return hashlib.sha256(self.target.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticAnchor:
    """Index-side identity and source anchors for one semantic theorem."""

    identity: str
    article: str
    number: int
    local_label: str | None
    source_goal: str | None
    mml_alignment: str | None
    statement: str = ""
    statement_sha256: str = ""
    html_file: str = ""
    html_anchor: str = ""
    html_line: int = 0
    proof_category: str | None = None
    proof_sha256: str | None = None


@dataclass(frozen=True)
class AlignedTheorem:
    """A source theorem paired with its authoritative semantic identity."""

    source: SourceTheorem
    anchor: SemanticAnchor
    source_index_binding: dict[str, Any] | None = None

    @property
    def identity(self) -> str:
        """Return the canonical ``ARTICLE:N`` identity."""

        return self.anchor.identity


@dataclass(frozen=True)
class CitationResolution:
    """Resolved globals, explicit failures, and proof-local labels."""

    references: tuple[str, ...]
    unresolved: tuple[str, ...]
    proof_local_labels: tuple[str, ...]


@dataclass(frozen=True)
class BuildConfig:
    """All immutable inputs needed for one fresh raw build."""

    mml_root: Path
    html_root: Path
    thproofs_root: Path
    semantic_index: Path
    semantic_index_sha256: str
    source_manifest: Path
    mizar_archive: Path
    html_archive: Path
    thproofs_archive: Path
    out: Path
    name: str = "mizar"
    heldout: int = 0
    seed: int = 20260801
    production: bool = True
    replay_sample_size: int = REPLAY_SAMPLE_SIZE


@dataclass
class VendoredTokenizer:
    """Validated low-level Qwen tokenizer with sealed metadata."""

    backend: Any
    identity: str
    tokenizer_json_sha256: str
    tokenizer_config_sha256: str
    behavior_digest: str
    tokenizers_version: str
    eos_token_id: int
    path: str

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        """Encode text without adding special tokens."""

        return self.backend.encode(text, add_special_tokens=add_special_tokens)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_miz(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _mask_mizar(text: str) -> str:
    """Mask source comments and actual quoted strings without moving offsets."""

    masked = list(text)
    index = 0
    while index < len(text):
        if text.startswith("::", index):
            while index < len(text) and text[index] != "\n":
                masked[index] = " "
                index += 1
            continue
        if text[index] == '"':
            previous = index - 1
            while previous >= 0 and text[previous] in " \t":
                previous -= 1
            line_end = text.find("\n", index)
            if line_end < 0:
                line_end = len(text)
            if previous < 0 or text[previous] in "=([{,:;":
                close = text.find('"', index + 1, line_end)
                if close >= 0:
                    for position in range(index, close + 1):
                        masked[position] = " "
                    index = close + 1
                    continue
        index += 1
    return "".join(masked)


def _trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _theorem_label(declaration: str) -> str | None:
    match = THEOREM_LABEL_RE.match(declaration)
    return match.group("label") if match is not None else None


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("::", 1)[0] for line in text.splitlines())


def _direct_source_goal(declaration: str) -> str:
    goal = _strip_comments(declaration)
    goal = re.sub(r"^\s*theorem\b", "", goal, count=1, flags=re.IGNORECASE)
    label_match = re.match(
        rf"^\s*{LABEL_TOKEN}\s*:(?![=\-])",
        goal,
        flags=re.IGNORECASE,
    )
    if label_match is not None:
        goal = goal[label_match.end() :]
    goal = re.sub(r"\s+", " ", goal).strip()
    goal = INLINE_JUSTIFICATION_RE.sub("", goal).strip()
    return goal.rstrip(";").strip()


def _proof_bounds(
    chunk: str,
) -> tuple[int, int | None, int | None, int | None] | None:
    """Find the balanced outer proof in one theorem-bounded source chunk."""

    masked = _mask_mizar(chunk)
    tokens = list(WORD_OR_SEMICOLON_RE.finditer(masked))
    first_semicolon = next(
        (index for index, token in enumerate(tokens) if token.group(0) == ";"),
        None,
    )
    proof_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.group(0).lower() == "proof"
        ),
        None,
    )
    if proof_index is None or (
        first_semicolon is not None and first_semicolon < proof_index
    ):
        return None

    proof_token = tokens[proof_index]
    stack = ["proof"]
    for index in range(proof_index + 1, len(tokens)):
        raw = tokens[index].group(0)
        token = raw.lower()
        if token in BLOCK_OPENERS:
            stack.append(token)
            continue
        if token != "end":
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].group(0) != ";":
            return proof_token.start(), None, None, None
        stack.pop()
        if stack:
            continue
        proof_end = tokens[index + 1].end()
        same_line_suffix = masked[proof_end:].split("\n", 1)[0]
        if same_line_suffix.strip():
            return proof_token.start(), None, None, None
        return (
            proof_token.start(),
            proof_token.end(),
            tokens[index].start(),
            proof_end,
        )
    return proof_token.start(), None, None, None


def _local_assumptions(target: str) -> dict[str, str]:
    assumptions: dict[str, str] = {}
    generated = 0
    for match in LOCAL_CONTEXT_RE.finditer(_mask_mizar(target)):
        generated += 1
        label = match.group("label") or f"$local_{generated}"
        statement = " ".join(
            target[match.start("statement") : match.end("statement")].split()
        )
        if not statement:
            continue
        key = label
        suffix = 2
        while key in assumptions:
            key = f"{label}#{suffix}"
            suffix += 1
        assumptions[key] = statement
    return assumptions


def parse_miz_article(
    text: str,
    *,
    article: str,
    source_file: str,
) -> list[SourceTheorem]:
    """Parse every literal theorem without crossing the next theorem header."""

    masked = _mask_mizar(text)
    starts = list(THEOREM_START_RE.finditer(masked))
    declarations: list[SourceTheorem] = []
    for ordinal, start_match in enumerate(starts, start=1):
        start = start_match.start()
        chunk_end = starts[ordinal].start() if ordinal < len(starts) else len(text)
        chunk = text[start:chunk_end]
        masked_chunk = masked[start:chunk_end]
        bounds = _proof_bounds(chunk)
        semicolon = masked_chunk.find(";")
        label = _theorem_label(chunk)
        target: str | None = None
        target_start = -1
        target_end = -1

        if bounds is not None:
            proof_start, body_start, body_end, proof_end = bounds
            goal_declaration = chunk[:proof_start]
            source_goal = _direct_source_goal(goal_declaration)
            index_source_goal = _adapter_source_goal(goal_declaration)
            if proof_end is None or body_start is None or body_end is None:
                category = "malformed_explicit_proof"
                declaration_end = chunk_end
            else:
                local_start, local_end = _trimmed_span(chunk, body_start, body_end)
                target_start = start + local_start
                target_end = start + local_end
                target = text[target_start:target_end]
                category = (
                    "complete_explicit_proof" if target else "malformed_explicit_proof"
                )
                declaration_end = start + proof_end
        elif semicolon >= 0:
            declaration_end = start + semicolon + 1
            declaration = text[start:declaration_end]
            source_goal = _direct_source_goal(declaration)
            index_source_goal = _adapter_source_goal(declaration)
            compact = re.sub(r"\s+", " ", _strip_comments(declaration)).strip()
            content = re.sub(
                r"^\s*theorem\b",
                "",
                compact,
                count=1,
                flags=re.IGNORECASE,
            ).strip()
            label_prefix = re.match(
                rf"^{LABEL_TOKEN}\s*:(?![=\-])",
                content,
                flags=re.IGNORECASE,
            )
            if label_prefix is not None:
                content = content[label_prefix.end() :].strip()
            if re.fullmatch(r"canceled\s*;", content, re.IGNORECASE):
                category = "canceled"
            elif INLINE_JUSTIFICATION_RE.search(compact):
                category = "inline_justification"
            else:
                category = "no_explicit_proof"
        else:
            declaration_end = chunk_end
            declaration = text[start:declaration_end]
            source_goal = _direct_source_goal(declaration)
            index_source_goal = _adapter_source_goal(declaration)
            category = "malformed_declaration"

        source_declaration = text[start:declaration_end].strip()
        line_start = text.count("\n", 0, start) + 1
        line_end = text.count("\n", 0, declaration_end) + 1
        declarations.append(
            SourceTheorem(
                article=article.upper(),
                source_file=source_file,
                ordinal=ordinal,
                label=label,
                category=category,
                source_goal=source_goal,
                index_source_goal=index_source_goal,
                target=target,
                local_assumptions=_local_assumptions(target or ""),
                declaration_start=start,
                declaration_end=declaration_end,
                target_start=target_start,
                target_end=target_end,
                line_start=line_start,
                line_end=line_end,
                source_declaration=source_declaration,
            )
        )
    return declarations


def _goals_match(source: SourceTheorem, anchor: SemanticAnchor) -> bool:
    if anchor.source_goal is None:
        return False
    return _comparison_key(source.index_source_goal) == _comparison_key(
        anchor.source_goal
    )


def _labels_compatible(source: SourceTheorem, anchor: SemanticAnchor) -> bool:
    return source.label == anchor.local_label


def align_article_declarations(
    declarations: Sequence[SourceTheorem],
    anchors: Sequence[SemanticAnchor],
) -> list[AlignedTheorem]:
    """Monotonically align literal declarations to exact index source anchors.

    This general alignment API is used by tests and non-production fixtures.
    Production emission additionally requires the exact proof-body hash join.
    """

    if any(left.number >= right.number for left, right in pairwise(anchors)):
        raise SourceIndexMismatch("semantic anchor order is not strictly increasing")
    source = [item for item in declarations if item.category != "canceled"]
    aligned: list[AlignedTheorem] = []
    next_index = 0
    for declaration in source:
        matches = [
            index
            for index in range(next_index, len(anchors))
            if _goals_match(declaration, anchors[index])
            and _labels_compatible(declaration, anchors[index])
        ]
        if not matches:
            raise SourceIndexMismatch(
                f"{declaration.article}:{declaration.ordinal} source goal/anchor "
                f"is unmapped: {declaration.source_goal!r}"
            )
        chosen = matches[0]
        aligned.append(AlignedTheorem(declaration, anchors[chosen]))
        next_index = chosen + 1
    if len(aligned) != len(source):
        raise SourceIndexMismatch(
            f"source/index count mismatch: {len(source)} declarations, "
            f"{len(aligned)} mapped"
        )
    return aligned


def _strict_complete_alignment(
    declarations: Sequence[SourceTheorem],
    anchors: Sequence[SemanticAnchor],
) -> tuple[list[AlignedTheorem], int]:
    """Find the unique maximum exact source/index alignment for one article."""

    source = [
        item
        for item in declarations
        if item.category == "complete_explicit_proof" and item.target_sha256
    ]
    expected = [
        anchor
        for anchor in anchors
        if anchor.proof_category == "complete_explicit_proof" and anchor.proof_sha256
    ]

    def matches(declaration: SourceTheorem, anchor: SemanticAnchor) -> bool:
        return (
            declaration.target_sha256 == anchor.proof_sha256
            and _goals_match(declaration, anchor)
            and _labels_compatible(declaration, anchor)
        )

    rows: list[array[int]] = [array("H", [0]) * (len(expected) + 1)]
    for declaration in source:
        previous = rows[-1]
        current = array("H", [0]) * (len(expected) + 1)
        for anchor_index, anchor in enumerate(expected, start=1):
            if matches(declaration, anchor):
                current[anchor_index] = previous[anchor_index - 1] + 1
            else:
                current[anchor_index] = max(
                    previous[anchor_index],
                    current[anchor_index - 1],
                )
        rows.append(current)

    def reconstruct(*, trailing_ties_first: bool) -> list[tuple[int, int]]:
        source_index = len(source)
        anchor_index = len(expected)
        pairs: list[tuple[int, int]] = []
        while source_index and anchor_index:
            value = rows[source_index][anchor_index]
            is_match = (
                matches(source[source_index - 1], expected[anchor_index - 1])
                and rows[source_index - 1][anchor_index - 1] + 1 == value
            )
            can_skip_source = rows[source_index - 1][anchor_index] == value
            can_skip_anchor = rows[source_index][anchor_index - 1] == value
            if trailing_ties_first and (can_skip_source or can_skip_anchor):
                if can_skip_source:
                    source_index -= 1
                else:
                    anchor_index -= 1
            elif is_match:
                pairs.append((source_index - 1, anchor_index - 1))
                source_index -= 1
                anchor_index -= 1
            elif can_skip_source:
                source_index -= 1
            else:
                anchor_index -= 1
        pairs.reverse()
        return pairs

    latest = reconstruct(trailing_ties_first=False)
    earliest = reconstruct(trailing_ties_first=True)
    if earliest != latest:
        first = next(
            index
            for index, (left, right) in enumerate(zip(earliest, latest))
            if left != right
        )
        raise SourceIndexMismatch(
            "non-unique proof/source order near "
            f"{expected[earliest[first][1]].identity}"
        )
    aligned = [
        AlignedTheorem(source[source_index], expected[anchor_index])
        for source_index, anchor_index in earliest
    ]
    return aligned, len(source) - len(aligned)


def _secondary_unique_label_alignment(
    declarations: Sequence[SourceTheorem],
    anchors: Sequence[SemanticAnchor],
    primary: Sequence[AlignedTheorem],
) -> list[AlignedTheorem]:
    """Recover only uniquely labeled malformed-index proofs between hash anchors."""

    if any(left.number >= right.number for left, right in pairwise(anchors)):
        return []
    source_label_counts = Counter(
        declaration.label for declaration in declarations if declaration.label
    )
    index_label_counts = Counter(
        anchor.local_label for anchor in anchors if anchor.local_label
    )
    primary_by_ordinal = sorted(primary, key=lambda item: item.source.ordinal)
    primary_ordinals = {item.source.ordinal for item in primary_by_ordinal}
    used_identities = {item.identity for item in primary_by_ordinal}
    recovered: list[AlignedTheorem] = []

    def proof_hash_anchor(item: AlignedTheorem | None) -> dict[str, Any] | None:
        if item is None:
            return None
        proof_sha256 = item.anchor.proof_sha256
        if (
            item.source.target_sha256 != proof_sha256
            or item.anchor.proof_category != "complete_explicit_proof"
            or proof_sha256 is None
        ):
            return None
        return {
            "source_ordinal": item.source.ordinal,
            "identity": item.identity,
            "index_number": item.anchor.number,
            "proof_sha256": proof_sha256,
        }

    for declaration in declarations:
        label = declaration.label
        if (
            declaration.category != "complete_explicit_proof"
            or declaration.target_sha256 is None
            or declaration.ordinal in primary_ordinals
            or not label
            or source_label_counts[label] != 1
            or index_label_counts[label] != 1
        ):
            continue
        previous = next(
            (
                item
                for item in reversed(primary_by_ordinal)
                if item.source.ordinal < declaration.ordinal
            ),
            None,
        )
        following = next(
            (
                item
                for item in primary_by_ordinal
                if item.source.ordinal > declaration.ordinal
            ),
            None,
        )
        previous_binding = proof_hash_anchor(previous)
        following_binding = proof_hash_anchor(following)
        if (previous is not None and previous_binding is None) or (
            following is not None and following_binding is None
        ):
            continue
        lower = previous.anchor.number if previous is not None else 0
        upper = following.anchor.number if following is not None else sys.maxsize
        candidates = [
            anchor
            for anchor in anchors
            if anchor.identity not in used_identities
            and anchor.article == declaration.article
            and anchor.local_label == label
            and anchor.mml_alignment == "literal_goal_match"
            and anchor.proof_category == "malformed_explicit_proof"
            and anchor.proof_sha256 is None
            and lower < anchor.number < upper
            and _goals_match(declaration, anchor)
        ]
        if len(candidates) != 1:
            continue
        anchor = candidates[0]
        normalized_source_goal = _comparison_key(declaration.index_source_goal)
        normalized_index_goal = _comparison_key(anchor.source_goal or "")
        if normalized_source_goal != normalized_index_goal:
            continue
        binding = {
            "schema_version": SOURCE_INDEX_BINDING_SCHEMA,
            "method": SECONDARY_ALIGNMENT_METHOD,
            "source_label_occurrences": source_label_counts[label],
            "index_label_occurrences": index_label_counts[label],
            "normalized_goal_sha256": hashlib.sha256(
                normalized_source_goal.encode("utf-8")
            ).hexdigest(),
            "previous_proof_hash_anchor": previous_binding,
            "next_proof_hash_anchor": following_binding,
        }
        recovered.append(
            AlignedTheorem(
                declaration,
                anchor,
                source_index_binding=binding,
            )
        )
        used_identities.add(anchor.identity)
    return recovered


def _cached_statements(index: Any) -> dict[str, str]:
    cached = getattr(index, "_human_builder_statement_cache", None)
    if cached is None:
        cached = index.statement_map()
        try:
            index._human_builder_statement_cache = cached
        except AttributeError:
            pass
    return cached


def _cached_local_maps(index: Any) -> dict[str, dict[str, tuple[str, ...]]]:
    cached = getattr(index, "_human_builder_local_cache", None)
    if cached is None:
        cached = index.article_local_label_maps()
        try:
            index._human_builder_local_cache = cached
        except AttributeError:
            pass
    return cached


def _skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


def _reference_token(
    masked: str,
    position: int,
) -> tuple[str, re.Match[str]] | None:
    for kind, pattern in (
        ("numeric", NUMERIC_REFERENCE_RE),
        ("qualified", QUALIFIED_LABEL_RE),
        ("bare", BARE_REFERENCE_RE),
    ):
        if match := pattern.match(masked, position):
            return kind, match
    return None


def _skip_parenthesized_call(masked: str, position: int) -> tuple[int, bool]:
    if position >= len(masked) or masked[position] != "(":
        return position, True
    depth = 0
    for index in range(position, len(masked)):
        if masked[index] == "(":
            depth += 1
        elif masked[index] == ")":
            depth -= 1
            if depth == 0:
                return index + 1, True
    return len(masked), False


def _citation_items(
    masked: str,
    start: int,
) -> tuple[list[tuple[str, re.Match[str]]], int, bool]:
    """Parse one comma-separated Mizar citation clause."""

    items: list[tuple[str, re.Match[str]]] = []
    position = _skip_space(masked, start)
    while token := _reference_token(masked, position):
        kind, match = token
        items.append((kind, match))
        position = _skip_space(masked, match.end())
        position, balanced = _skip_parenthesized_call(masked, position)
        if not balanced:
            return items, position, False
        position = _skip_space(masked, position)
        if position >= len(masked) or masked[position] != ",":
            return items, position, True
        candidate = _skip_space(masked, position + 1)
        if _reference_token(masked, candidate) is None:
            return items, position, False
        position = candidate
    return items, position, bool(items)


def _proof_local_labels(body: str) -> list[tuple[int, str]]:
    masked = _mask_mizar(body)
    justification_spans = [
        (justification.start(), _citation_items(masked, justification.end())[1])
        for justification in JUSTIFICATION_RE.finditer(masked)
    ]
    return [
        (match.start("label"), match.group("label"))
        for match in PROOF_LOCAL_LABEL_RE.finditer(masked)
        if not any(
            start <= match.start("label") < end for start, end in justification_spans
        )
    ]


def _numeric_references(match: re.Match[str]) -> list[str]:
    article = match.group("article").upper()
    kind = match.group("kind")
    number = match.group("special") or match.group("number")
    prefix = f"{kind.lower()}_" if kind else ""
    references = [f"{article}:{prefix}{int(number)}"]
    inherited_kind = kind.lower() if kind else None
    for item in re.findall(
        r"(?:def|sch)?\s*_?\s*[1-9]\d*",
        match.group("tail"),
        flags=re.IGNORECASE,
    ):
        compact = re.sub(r"\s+", "", item).lower()
        tail_match = re.fullmatch(r"(?:(def|sch)_?)?([1-9]\d*)", compact)
        if tail_match is None:
            continue
        tail_kind, tail_number = tail_match.groups()
        effective_kind = tail_kind or inherited_kind
        tail_prefix = f"{effective_kind}_" if effective_kind else ""
        references.append(f"{article}:{tail_prefix}{int(tail_number)}")
    return references


def resolve_global_citations(
    body: str,
    index: Any,
    *,
    theorem: str,
) -> CitationResolution:
    """Resolve every global citation and exclude temporally prior proof locals."""

    article = theorem.split(":", 1)[0].upper()
    statements = _cached_statements(index)
    local_maps = _cached_local_maps(index)
    local_declarations = _proof_local_labels(body)
    masked = _mask_mizar(body)
    references: list[str] = []
    unresolved: list[str] = []

    def add_reference(name: str) -> None:
        if name == theorem or name not in statements:
            if name not in unresolved:
                unresolved.append(name)
        elif name not in references:
            references.append(name)

    for justification in JUSTIFICATION_RE.finditer(masked):
        items, _, complete = _citation_items(masked, justification.end())
        if not complete:
            marker = f"<malformed-{justification.group(0).lower()}>"
            if marker not in unresolved:
                unresolved.append(marker)
            continue
        prior_proof_locals = {
            label
            for position, label in local_declarations
            if position < justification.start()
        }
        for kind, match in items:
            if kind == "numeric":
                for name in _numeric_references(match):
                    add_reference(name)
                continue
            if kind == "qualified":
                named_article = match.group("article").upper()
                label = match.group("label")
                if named_article == article:
                    try:
                        resolved = index.resolve_local_label(
                            named_article,
                            label,
                            at_identity=theorem,
                        )
                    except KeyError:
                        resolved = None
                else:
                    identities = local_maps.get(named_article, {}).get(label, ())
                    resolved = identities[0] if len(identities) == 1 else None
                if resolved is None:
                    name = f"{named_article}:{label}"
                    if name not in unresolved:
                        unresolved.append(name)
                else:
                    add_reference(resolved)
                continue

            label = match.group(0)
            if label.isdigit() or label.lower() in {"def", "sch"}:
                continue
            if label in prior_proof_locals:
                continue
            try:
                resolved = index.resolve_local_label(
                    article,
                    label,
                    at_identity=theorem,
                )
            except KeyError:
                resolved = None
            if resolved is None:
                if label not in unresolved:
                    unresolved.append(label)
            else:
                add_reference(resolved)

    proof_labels: list[str] = []
    for _, label in local_declarations:
        if label not in proof_labels:
            proof_labels.append(label)
    return CitationResolution(
        references=tuple(references),
        unresolved=tuple(unresolved),
        proof_local_labels=tuple(proof_labels),
    )


def deterministic_fact_order(
    references: Sequence[str],
    *,
    row_key: str,
    seed: int,
) -> list[str]:
    """Hash-order unique facts independently of source citation order."""

    unique = sorted(set(references))

    def rank(name: str) -> tuple[str, str]:
        payload = f"{FACT_ORDER_POLICY_ID}\0{seed}\0{row_key}\0{name}".encode()
        return hashlib.sha256(payload).hexdigest(), name

    return sorted(unique, key=rank)


def render_training_text(
    facts: Mapping[str, str],
    goal: str,
    target: str,
) -> tuple[str, int, int]:
    """Render the exact global-fact prompt and human proof target."""

    block = (
        HDR
        + "\n"
        + "\n".join(f"{name} : {statement}" for name, statement in facts.items())
    )
    return f"{block}\n{SEP}\nGOAL {goal}\n{target}", 0, len(block)


def _tokenizer_behavior_digest(tokenizer: Any) -> str:
    payload = {
        "schema": "qwen-tokenizer-behavior-v1",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "eos_token": QWEN_EOS_TOKEN,
        "eos_token_id": tokenizer.token_to_id(QWEN_EOS_TOKEN),
        "probes": [],
    }
    for text in TOKENIZER_BEHAVIOR_PROBES:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        payload["probes"].append(
            {
                "text": text,
                "ids": encoding.ids,
                "tokens": encoding.tokens,
                "offsets": [list(pair) for pair in encoding.offsets],
                "decoded": tokenizer.decode(
                    encoding.ids,
                    skip_special_tokens=False,
                ),
            }
        )
    return _canonical_sha256(payload)


def load_vendored_tokenizer(path: str | os.PathLike[str]) -> VendoredTokenizer:
    """Load and verify the exact approved Qwen tokenizer artifact."""

    requested = Path(path)
    tokenizer_json = requested / "tokenizer.json" if requested.is_dir() else requested
    config_path = tokenizer_json.with_name("tokenizer_config.json")
    if not tokenizer_json.is_file() or not config_path.is_file():
        raise BuildError(f"vendored Qwen tokenizer is incomplete: {requested}")
    tokenizer_json_sha256 = _file_sha256(tokenizer_json)
    tokenizer_config_sha256 = _file_sha256(config_path)
    if tokenizer_json_sha256 != APPROVED_TOKENIZER_JSON_SHA256:
        raise BuildError(
            "tokenizer.json SHA-256 is not approved: " f"{tokenizer_json_sha256}"
        )
    if tokenizer_config_sha256 != APPROVED_TOKENIZER_CONFIG_SHA256:
        raise BuildError(
            "tokenizer_config.json SHA-256 is not approved: "
            f"{tokenizer_config_sha256}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError("invalid Qwen tokenizer config") from error
    if (
        config.get("tokenizer_class") != "Qwen2Tokenizer"
        or config.get("eos_token") != QWEN_EOS_TOKEN
    ):
        raise BuildError("tokenizer is not the pinned Qwen2 family")
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as error:
        raise BuildError("the tokenizers package is required") from error
    if str(tokenizers.__version__) != APPROVED_TOKENIZERS_VERSION:
        raise BuildError(
            "tokenizers version is not approved: " f"{tokenizers.__version__}"
        )
    backend = Tokenizer.from_file(str(tokenizer_json))
    backend.no_padding()
    backend.no_truncation()
    eos_token_id = backend.token_to_id(QWEN_EOS_TOKEN)
    if eos_token_id != QWEN_EOS_TOKEN_ID:
        raise BuildError(f"Qwen EOS id mismatch: {eos_token_id}")
    behavior_digest = _tokenizer_behavior_digest(backend)
    if behavior_digest != APPROVED_TOKENIZER_BEHAVIOR_SHA256:
        raise BuildError(f"Qwen tokenizer behavior is not approved: {behavior_digest}")
    return VendoredTokenizer(
        backend=backend,
        identity=QWEN_TOKENIZER_ID,
        tokenizer_json_sha256=tokenizer_json_sha256,
        tokenizer_config_sha256=tokenizer_config_sha256,
        behavior_digest=behavior_digest,
        tokenizers_version=str(tokenizers.__version__),
        eos_token_id=eos_token_id,
        path=str(tokenizer_json.resolve()),
    )


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    metadata = {
        "identity": str(getattr(tokenizer, "identity", "")),
        "tokenizer_json_sha256": str(
            getattr(tokenizer, "tokenizer_json_sha256", "")
        ).lower(),
        "tokenizer_config_sha256": str(
            getattr(tokenizer, "tokenizer_config_sha256", "")
        ).lower(),
        "behavior_digest": str(getattr(tokenizer, "behavior_digest", "")).lower(),
        "tokenizers_version": str(getattr(tokenizer, "tokenizers_version", "")),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "max_text_plus_eos_tokens": MAX_TOKENS_WITH_EOS,
        "path": str(getattr(tokenizer, "path", "")),
    }
    for key in (
        "tokenizer_json_sha256",
        "tokenizer_config_sha256",
        "behavior_digest",
    ):
        if SHA256_RE.fullmatch(metadata[key]) is None:
            raise BuildError(f"tokenizer {key} is missing or malformed")
    if metadata["identity"] != QWEN_TOKENIZER_ID:
        raise BuildError("tokenizer identity is not the approved Qwen tokenizer")
    if metadata["eos_token_id"] != QWEN_EOS_TOKEN_ID:
        raise BuildError("tokenizer EOS id is not approved")
    if not metadata["tokenizers_version"]:
        raise BuildError("tokenizers implementation version is missing")
    return metadata


def _tokens_with_eos(tokenizer: Any, text: str) -> int:
    try:
        encoding = tokenizer.encode(text, add_special_tokens=False)
    except Exception as error:
        raise BuildError("Qwen tokenizer failed to encode a row") from error
    ids = getattr(encoding, "ids", encoding)
    try:
        return len(ids) + 1
    except TypeError as error:
        raise BuildError("tokenizer returned an unsized encoding") from error


def _load_anchors(index: MizarIndex, article: str) -> list[SemanticAnchor]:
    rows = index.connection.execute(
        """
        SELECT
            s.identity, s.article, s.number, s.local_label,
            t.source_goal, t.mml_alignment, s.statement,
            s.statement_sha256, s.html_file, s.html_anchor, s.html_line,
            t.category, t.proof_sha256
        FROM statements AS s
        LEFT JOIN thproofs AS t ON t.identity = s.identity
        WHERE s.article = ? AND s.kind = 'theorem'
        ORDER BY s.number
        """,
        (article,),
    )
    return [
        SemanticAnchor(
            identity=row[0],
            article=row[1],
            number=int(row[2]),
            local_label=row[3],
            source_goal=row[4],
            mml_alignment=row[5],
            statement=row[6],
            statement_sha256=row[7],
            html_file=row[8],
            html_anchor=row[9],
            html_line=int(row[10]),
            proof_category=row[11],
            proof_sha256=row[12],
        )
        for row in rows
    ]


def _source_metadata(
    config: BuildConfig,
    manifest: Mapping[str, Any],
    tokenizer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    source_roots = {
        name: {
            "reference": spec["archive_url"],
            "archive_sha256": spec["archive_sha256"],
            "file_count": spec["file_count"],
            "tree_sha256": spec["tree_sha256"],
        }
        for name, spec in sorted(manifest["sources"].items())
    }
    source_roots["tokenizer"] = {
        key: tokenizer_metadata[key]
        for key in (
            "identity",
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "behavior_digest",
            "tokenizers_version",
            "eos_token_id",
            "max_text_plus_eos_tokens",
        )
    }
    quality_policy = {
        "canonical_goal_authority": INDEX_SCHEMA,
        "direct_human_target": True,
        "exact_proof_hash_anchor": config.production,
        "global_references_only": True,
        "max_tokens_with_eos": MAX_TOKENS_WITH_EOS,
        "no_truncation": True,
        "raw_only": True,
        "reference_parser": "grammar-bounded-citation-list-v2",
        "replay_sample_size": config.replay_sample_size,
        "production_counts": BASE_PRODUCTION_COUNTERS,
        "token_lengths_sha256": EXPECTED_TOKEN_LENGTHS_SHA256,
        "distinct_facts": EXPECTED_DISTINCT_FACTS,
    }
    schema_policy = {
        "row_schema": ROW_SCHEMA,
        "mask_schema": MASK_SCHEMA,
        "local_assumptions": "explicit-target-context-v1",
        "fact_order_policy": FACT_ORDER_POLICY_ID,
    }
    return {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "source_manifest_root_sha256": "",
        "source_roots": source_roots,
        "index_roots": {
            "semantic_index_schema": INDEX_SCHEMA,
            "semantic_index_sha256": config.semantic_index_sha256,
        },
        "quality_filter_root_sha256": _canonical_sha256(quality_policy),
        "schema_generation_root_sha256": _canonical_sha256(schema_policy),
    }


def _family_manifest_root(manifest: Mapping[str, Any]) -> str:
    def without_recursive_roots(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: without_recursive_roots(item)
                for key, item in value.items()
                if key
                not in {
                    "manifest_root_sha256",
                    "source_manifest_root_sha256",
                }
            }
        if isinstance(value, list):
            return [without_recursive_roots(item) for item in value]
        return value

    payload = _canonical_json(without_recursive_roots(manifest)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _builder_argv(
    config: BuildConfig,
    tokenizer_metadata: Mapping[str, Any],
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mml-root",
        str(config.mml_root),
        "--html-root",
        str(config.html_root),
        "--thproofs-root",
        str(config.thproofs_root),
        "--semantic-index",
        str(config.semantic_index),
        "--semantic-index-sha256",
        config.semantic_index_sha256,
        "--source-manifest",
        str(config.source_manifest),
        "--mizar-archive",
        str(config.mizar_archive),
        "--html-archive",
        str(config.html_archive),
        "--thproofs-archive",
        str(config.thproofs_archive),
        "--tokenizer-path",
        str(tokenizer_metadata["path"]),
        "--name",
        "mizar",
        "--heldout",
        "0",
        "--seed",
        str(config.seed),
    ]


def _family_source_manifest(
    config: BuildConfig,
    upstream: Mapping[str, Any],
    source_metadata: dict[str, Any],
    tokenizer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = [
        {"path": directory, "kind": "directory"}
        for directory in ("checksums", "manifests", "raw", "reports")
    ]
    inventory.extend(
        [
            {
                "path": "raw/mizar.jsonl",
                "kind": "file",
                "format": "jsonl",
                "schema": ROW_SCHEMA,
                "source_manifest_root_sha256": "",
            },
            {
                "path": "manifests/mizar.json",
                "kind": "file",
                "format": "json",
                "schema": FAMILY_SOURCE_MANIFEST_SCHEMA,
                "required_fields": [
                    "builder",
                    "family",
                    "license",
                    "manifest_root_sha256",
                    "row_schema_version",
                    "row_source_metadata",
                    "source_snapshots",
                    "source_verifier_acceptance",
                    "test_only",
                ],
            },
            {
                "path": "reports/mizar.build.json",
                "kind": "file",
                "format": "json",
                "schema": BUILD_REPORT_SCHEMA,
                "required_fields": [
                    "counters",
                    "family",
                    "mode",
                    "output_hashes",
                ],
            },
            {
                "path": "reports/mizar.fact_frequencies.json",
                "kind": "file",
                "format": "binary",
                "schema": "mizar-fact-frequencies-v1",
            },
            {
                "path": "checksums/mizar.json",
                "kind": "file",
                "format": "json",
                "schema": "mizar-human-output-checksums-v1",
                "required_fields": ["files"],
            },
        ]
    )
    snapshots = [
        {
            "reference": spec["archive_url"],
            "sha256": spec["archive_sha256"],
        }
        for _name, spec in sorted(upstream["sources"].items())
    ]
    licensing = upstream["licensing"]
    family_manifest = {
        "schema_version": FAMILY_SOURCE_MANIFEST_SCHEMA,
        "family": "mizar",
        "row_schema_version": ROW_SCHEMA,
        "row_source_metadata": source_metadata,
        "source_snapshots": snapshots,
        "builder": {
            "driver": "external-command-v2",
            "partition_mode": "pooled-mml-1000-v1",
            "raw": {
                "argv": _builder_argv(config, tokenizer_metadata),
                "inventory": inventory,
                "outputs": {"raw": "raw/mizar.jsonl"},
            },
        },
        "license": {
            "approved": False,
            "identifier": "unresolved-mizar-mml-aggregate",
            "status": licensing["status"],
        },
        "source_verifier_acceptance": {
            "accepted": True,
            "status": "direct source/index/reference replay clean",
        },
        "test_only": not config.production,
        "manifest_root_sha256": "",
    }
    root = _family_manifest_root(family_manifest)
    source_metadata["source_manifest_root_sha256"] = root
    inventory[-5]["source_manifest_root_sha256"] = root
    family_manifest["manifest_root_sha256"] = root
    return family_manifest


def _verify_inputs(
    config: BuildConfig,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if config.heldout != 0:
        raise BuildError("direct Mizar builder requires heldout=0 raw staging")
    if config.production and config.name != "mizar":
        raise BuildError("production direct Mizar output name must be mizar")
    if config.replay_sample_size != REPLAY_SAMPLE_SIZE and config.production:
        raise BuildError(
            f"production replay sample must be exactly {REPLAY_SAMPLE_SIZE}"
        )
    if config.out.exists():
        raise BuildError(f"output must be a fresh path: {config.out}")
    if SHA256_RE.fullmatch(config.semantic_index_sha256) is None:
        raise BuildError("semantic index SHA-256 is missing or malformed")
    observed_index_sha256 = _file_sha256(config.semantic_index)
    if observed_index_sha256 != config.semantic_index_sha256:
        raise BuildError(
            "semantic index SHA-256 mismatch: expected "
            f"{config.semantic_index_sha256}, observed {observed_index_sha256}"
        )

    roots = {
        "mml": config.mml_root,
        "html": config.html_root,
        "thproofs": config.thproofs_root,
    }
    archives = {
        "mml": config.mizar_archive,
        "html": config.html_archive,
        "thproofs": config.thproofs_archive,
    }
    try:
        manifest = verify_source_manifest(
            config.source_manifest,
            roots,
            archive_paths=archives,
        )
    except SourceVerificationError as error:
        raise BuildError(f"source drift or manifest failure: {error}") from error
    if manifest["release"] != {
        "mizar_version": MIZAR_VERSION,
        "mml_version": MML_VERSION,
    }:
        raise BuildError("source manifest is not Mizar 8.1.15 / MML 5.94.1493")

    tokenizer_metadata = _tokenizer_metadata(tokenizer)
    manifest_sha256 = _file_sha256(config.source_manifest)
    try:
        with MizarIndex(config.semantic_index) as index:
            application_id = int(
                index.connection.execute("PRAGMA application_id").fetchone()[0]
            )
            user_version = int(
                index.connection.execute("PRAGMA user_version").fetchone()[0]
            )
            quick_check = index.connection.execute("PRAGMA quick_check").fetchone()[0]
            metadata = index.metadata()
    except (MizarIndexError, OSError, sqlite3.Error) as error:
        raise BuildError(f"semantic index verification failed: {error}") from error
    if application_id != SQLITE_APPLICATION_ID or user_version != SQLITE_USER_VERSION:
        raise BuildError("semantic index SQLite identity/version mismatch")
    if quick_check != "ok":
        raise BuildError(f"semantic index integrity check failed: {quick_check}")
    if metadata.get("schema_version") != INDEX_SCHEMA:
        raise BuildError("semantic index schema mismatch")
    if metadata.get("source_manifest_sha256") != manifest_sha256:
        raise BuildError("semantic index belongs to a different source manifest")
    if metadata.get("release") != manifest["release"]:
        raise BuildError("semantic index release disagrees with source manifest")
    expected_trees = {
        name: {
            "file_count": spec["file_count"],
            "tree_sha256": spec["tree_sha256"],
        }
        for name, spec in sorted(manifest["sources"].items())
    }
    if metadata.get("source_trees") != expected_trees:
        raise BuildError("semantic index source trees disagree with manifest")
    content = metadata.get("content")
    if not isinstance(content, dict):
        raise BuildError("semantic index content metadata is missing")
    if config.production and (
        manifest["sources"]["mml"]["file_count"] != EXPECTED_MML_FILES
        or content.get("statement_count") != EXPECTED_STATEMENTS
        or content.get("theorem_count") != EXPECTED_THEOREMS
    ):
        raise BuildError("production source/index count gate failed")
    source_metadata = _source_metadata(config, manifest, tokenizer_metadata)
    family_manifest = _family_source_manifest(
        config,
        manifest,
        source_metadata,
        tokenizer_metadata,
    )
    return manifest, metadata, source_metadata, family_manifest


def _row_id(aligned: AlignedTheorem) -> str:
    source = aligned.source
    payload = "\0".join(
        (
            ROW_SCHEMA,
            aligned.identity,
            source.source_file,
            str(source.ordinal),
            source.target_sha256 or "",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record(
    aligned: AlignedTheorem,
    *,
    source_file_sha256: str,
    source_encoding: str,
    resolution: CitationResolution,
    facts: Mapping[str, str],
    text: str,
    mask_start: int,
    mask_end: int,
    token_length: int,
    source_metadata: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    source = aligned.source
    anchor = aligned.anchor
    record_id = _row_id(aligned)
    record = {
        "schema_version": ROW_SCHEMA,
        "id": record_id,
        "family": "mizar",
        "split": "raw",
        "heldout": 0,
        "theorem": anchor.identity,
        "facts": dict(facts),
        "cited": list(resolution.references),
        "proof_local_labels": list(resolution.proof_local_labels),
        "local_assumptions": dict(source.local_assumptions),
        "goal": anchor.statement,
        "target": source.target,
        "text": text,
        "mask": {
            "schema_version": MASK_SCHEMA,
            "start": mask_start,
            "end": mask_end,
        },
        "mask_start": mask_start,
        "mask_end": mask_end,
        "token_length_with_eos": token_length,
        "source": {
            "article": source.article,
            "file": source.source_file,
            "encoding": source_encoding,
            "file_sha256": source_file_sha256,
            "declaration_ordinal": source.ordinal,
            "label": source.label,
            "source_goal": source.source_goal,
            "index_compatible_source_goal": source.index_source_goal,
            "line_start": source.line_start,
            "line_end": source.line_end,
            "declaration_start": source.declaration_start,
            "declaration_end": source.declaration_end,
            "target_start": source.target_start,
            "target_end": source.target_end,
            "declaration_sha256": hashlib.sha256(
                source.source_declaration.encode("utf-8")
            ).hexdigest(),
            "target_sha256": source.target_sha256,
        },
        "index": {
            "identity": anchor.identity,
            "number": anchor.number,
            "local_label": anchor.local_label,
            "source_goal": anchor.source_goal,
            "mml_alignment": anchor.mml_alignment,
            "proof_category": anchor.proof_category,
            "proof_sha256": anchor.proof_sha256,
            "statement_sha256": anchor.statement_sha256,
            "html_file": anchor.html_file,
            "html_anchor": anchor.html_anchor,
            "html_line": anchor.html_line,
        },
        "source_metadata": dict(source_metadata),
        "shuffle": {
            "scheme": "sha256-rank-v1",
            "seed": seed,
        },
    }
    if aligned.source_index_binding is not None:
        record["source_index_binding"] = dict(aligned.source_index_binding)
    return record


def _initialize_counters() -> Counter[str]:
    return Counter({key: 0 for key in COUNTER_KEYS})


def _validate_production_counts(counters: Mapping[str, int]) -> None:
    for name, expected in EXPECTED_PRODUCTION_COUNTERS.items():
        observed = int(counters.get(name, 0))
        if observed != expected:
            raise BuildError(
                f"production count mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )
    declaration_accounting = (
        counters["complete_explicit_declarations"]
        + counters["dropped_canceled"]
        + counters["dropped_inline_justification"]
        + counters["dropped_no_explicit_proof"]
        + counters["dropped_malformed_declaration"]
        + counters["dropped_malformed_explicit_proof"]
    )
    if declaration_accounting != counters["declarations_total"]:
        raise BuildError("production declaration accounting mismatch")
    mapped_accounting = (
        counters["accepted_rows"]
        + counters["dropped_unresolved_reference"]
        + counters["dropped_no_global_citation"]
        + counters["dropped_duplicate"]
        + counters["dropped_overlength"]
    )
    if mapped_accounting != counters["mapped_complete_declarations"]:
        raise BuildError("production mapped-row accounting mismatch")
    source_accounting = (
        counters["mapped_complete_declarations"]
        + counters["dropped_source_index_unanchored"]
    )
    if source_accounting != counters["complete_explicit_declarations"]:
        raise BuildError("production source/index accounting mismatch")


def _insert_unique(
    connection: sqlite3.Connection,
    text_hash: str,
    text: str,
) -> bool:
    try:
        connection.execute(
            "INSERT INTO seen_text(hash, text) VALUES (?, ?)",
            (text_hash, text),
        )
        return True
    except sqlite3.IntegrityError:
        existing = connection.execute(
            "SELECT text FROM seen_text WHERE hash = ?",
            (text_hash,),
        ).fetchone()
        if existing is None or existing[0] != text:
            raise BuildError("SHA-256 collision in exact duplicate gate")
        return False


def _percentile(sorted_values: Sequence[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    index = max(0, math.ceil(percentile * len(sorted_values)) - 1)
    return int(sorted_values[index])


def _context_report(
    token_lengths: Sequence[int],
    *,
    overlength: int,
    local_rows: int,
) -> dict[str, Any]:
    ordered = sorted(token_lengths)
    encoded_lengths = "\n".join(str(value) for value in token_lengths).encode("ascii")
    return {
        "max_text_plus_eos_tokens": MAX_TOKENS_WITH_EOS,
        "eligible_rows": len(token_lengths),
        "overlength_rows": overlength,
        "local_assumption_rows": local_rows,
        "token_lengths_in_row_order_sha256": hashlib.sha256(
            encoded_lengths
        ).hexdigest(),
        "minimum": ordered[0] if ordered else 0,
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "maximum": ordered[-1] if ordered else 0,
    }


def _joined_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _validate_primary_snapshot(
    counters: Mapping[str, int],
    *,
    raw_sha256: str,
    token_lengths: Sequence[int],
    fact_frequencies: Mapping[str, int],
) -> None:
    for name, expected in BASE_PRODUCTION_COUNTERS.items():
        observed = int(counters.get(name, 0))
        if observed != expected:
            raise BuildError(
                f"primary snapshot count mismatch for {name}: "
                f"expected {expected}, observed {observed}"
            )
    if counters.get("recovered_unique_label", 0) != 0:
        raise BuildError("primary snapshot already contains recovered rows")
    if raw_sha256 != EXPECTED_PRIMARY_RAW_SHA256:
        raise BuildError(
            "primary 50,114-row byte prefix drifted: "
            f"expected {EXPECTED_PRIMARY_RAW_SHA256}, observed {raw_sha256}"
        )
    if len(token_lengths) != EXPECTED_PRIMARY_ROWS:
        raise BuildError("primary token-length row count drifted")
    if sum(token_lengths) != EXPECTED_PRIMARY_TOKENS:
        raise BuildError("primary Qwen token total drifted")
    if _joined_sha256([str(value) for value in token_lengths]) != (
        EXPECTED_TOKEN_LENGTHS_SHA256
    ):
        raise BuildError("primary exact Qwen token-length sequence drifted")
    if len(fact_frequencies) != EXPECTED_DISTINCT_FACTS:
        raise BuildError("primary distinct-fact count drifted")


def _recovery_evidence(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities = [str(candidate["record"]["theorem"]) for candidate in candidates]
    token_lengths = [
        int(candidate["record"]["token_length_with_eos"]) for candidate in candidates
    ]
    text_hashes = [str(candidate["text_sha256"]) for candidate in candidates]
    source_bindings = []
    for candidate in candidates:
        record = candidate["record"]
        source = record["source"]
        source_bindings.append(
            "\0".join(
                (
                    str(source["article"]),
                    str(source["declaration_ordinal"]),
                    str(record["theorem"]),
                    str(source["label"]),
                    str(source["target_sha256"]),
                    str(candidate["text_sha256"]),
                )
            )
        )
    return {
        "schema_version": "mizar-direct-recovery-evidence-v1",
        "method": SECONDARY_ALIGNMENT_METHOD,
        "rows": len(candidates),
        "tokens_with_eos": sum(token_lengths),
        "identity_source_order_sha256": _joined_sha256(identities),
        "identity_set_sha256": _joined_sha256(sorted(identities)),
        "source_binding_sha256": _joined_sha256(source_bindings),
        "token_sequence_sha256": _joined_sha256(
            [str(value) for value in token_lengths]
        ),
        "text_hash_sequence_sha256": _joined_sha256(text_hashes),
        "duplicate_checks": {
            "accepted_or_internal_text": "clean",
            "accepted_thproof_trajectory": "clean",
        },
    }


def _validate_recovery_evidence(evidence: Mapping[str, Any]) -> None:
    expected = {
        "rows": EXPECTED_RECOVERED_ROWS,
        "tokens_with_eos": EXPECTED_RECOVERED_TOKENS,
        "identity_source_order_sha256": (
            EXPECTED_RECOVERED_IDENTITY_SOURCE_ORDER_SHA256
        ),
        "identity_set_sha256": EXPECTED_RECOVERED_IDENTITY_SET_SHA256,
        "source_binding_sha256": EXPECTED_RECOVERED_SOURCE_BINDING_SHA256,
        "token_sequence_sha256": EXPECTED_RECOVERED_TOKEN_SEQUENCE_SHA256,
        "text_hash_sequence_sha256": (EXPECTED_RECOVERED_TEXT_HASH_SEQUENCE_SHA256),
    }
    for name, expected_value in expected.items():
        observed = evidence.get(name)
        if observed != expected_value:
            raise BuildError(
                f"audited direct-Mizar recovery {name} drifted: "
                f"expected {expected_value}, observed {observed}"
            )


def _sample_stratum(row: Mapping[str, Any]) -> str:
    tokens = int(row["token_length_with_eos"])
    token_bucket = (
        "lt1k"
        if tokens < 1024
        else "1k-4k" if tokens < 4096 else "4k-8k" if tokens < 8192 else "8k-plus"
    )
    facts = len(row["facts"])
    fact_bucket = "one" if facts == 1 else "two-four" if facts <= 4 else "five-plus"
    local = "local" if row["local_assumptions"] else "no-local"
    return f"{token_bucket}/{fact_bucket}/{local}"


def _select_replay_ids(
    descriptors: Sequence[tuple[str, str]],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[str], dict[str, int]]:
    strata: defaultdict[str, list[str]] = defaultdict(list)
    for row_id, stratum in descriptors:
        strata[stratum].append(row_id)
    for stratum, row_ids in strata.items():
        row_ids.sort(
            key=lambda row_id: hashlib.sha256(
                f"{seed}\0{stratum}\0{row_id}".encode()
            ).hexdigest()
        )
    selected: list[str] = []
    positions = {stratum: 0 for stratum in strata}
    while len(selected) < min(sample_size, len(descriptors)):
        advanced = False
        for stratum in sorted(strata):
            position = positions[stratum]
            if position >= len(strata[stratum]):
                continue
            selected.append(strata[stratum][position])
            positions[stratum] += 1
            advanced = True
            if len(selected) >= min(sample_size, len(descriptors)):
                break
        if not advanced:
            break
    selected_counts = Counter(
        stratum for row_id, stratum in descriptors if row_id in set(selected)
    )
    return selected, dict(sorted(selected_counts.items()))


def _deep_self_check(
    raw_path: Path,
    *,
    config: BuildConfig,
    tokenizer: Any,
    index: MizarIndex,
) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    rows_checked = 0
    descriptors: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    article_cache: dict[str, tuple[str, list[SourceTheorem], str]] = {}
    alignment_cache: dict[
        str,
        tuple[list[AlignedTheorem], list[AlignedTheorem]],
    ] = {}
    with raw_path.open(encoding="utf-8") as raw_file:
        for line_number, line in enumerate(raw_file, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BuildError(f"raw row {line_number} is invalid JSON") from error
            if row.get("schema_version") != ROW_SCHEMA:
                raise BuildError(f"raw row {line_number} has the wrong schema")
            if row.get("split") != "raw" or row.get("heldout") != 0:
                raise BuildError(f"raw row {line_number} is not raw heldout=0")
            row_id = row.get("id")
            if not isinstance(row_id, str) or row_id in seen_ids:
                raise BuildError(f"raw row {line_number} has a duplicate/invalid id")
            seen_ids.add(row_id)
            text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if text_hash in seen_texts:
                raise BuildError(f"raw row {line_number} duplicates prior text")
            seen_texts.add(text_hash)

            source = row["source"]
            source_file = source["file"]
            if source_file not in article_cache:
                path = config.mml_root / source_file
                text, encoding = _read_miz(path)
                article_cache[source_file] = (
                    text,
                    parse_miz_article(
                        text,
                        article=path.stem.upper(),
                        source_file=path.name,
                    ),
                    encoding,
                )
            text, declarations, encoding = article_cache[source_file]
            ordinal = int(source["declaration_ordinal"])
            if ordinal < 1 or ordinal > len(declarations):
                raise BuildError(f"raw row {line_number} has an invalid source ordinal")
            declaration = declarations[ordinal - 1]
            if declaration.category != "complete_explicit_proof":
                raise BuildError(f"raw row {line_number} source proof is not complete")
            if declaration.target != row["target"]:
                raise BuildError(f"raw row {line_number} target/source mismatch")
            if text[declaration.target_start : declaration.target_end] != row["target"]:
                raise BuildError(f"raw row {line_number} target offsets drifted")
            if encoding != source["encoding"]:
                raise BuildError(f"raw row {line_number} source encoding drifted")
            if _file_sha256(config.mml_root / source_file) != source["file_sha256"]:
                raise BuildError(f"raw row {line_number} source hash drifted")
            if (
                hashlib.sha256(
                    declaration.source_declaration.encode("utf-8")
                ).hexdigest()
                != source["declaration_sha256"]
            ):
                raise BuildError(f"raw row {line_number} declaration hash mismatch")

            anchor_row = index.connection.execute(
                """
                SELECT
                    s.number, s.statement, s.statement_sha256, s.local_label,
                    s.html_file, s.html_anchor, s.html_line,
                    t.source_goal, t.mml_alignment, t.category, t.proof_sha256
                FROM statements AS s
                LEFT JOIN thproofs AS t ON t.identity = s.identity
                WHERE s.identity = ? AND s.kind = 'theorem'
                """,
                (row["theorem"],),
            ).fetchone()
            if anchor_row is None:
                raise BuildError(f"raw row {line_number} theorem is absent from index")
            expected_index = {
                "identity": row["theorem"],
                "number": int(anchor_row[0]),
                "local_label": anchor_row[3],
                "source_goal": anchor_row[7],
                "mml_alignment": anchor_row[8],
                "proof_category": anchor_row[9],
                "proof_sha256": anchor_row[10],
                "statement_sha256": anchor_row[2],
                "html_file": anchor_row[4],
                "html_anchor": anchor_row[5],
                "html_line": int(anchor_row[6]),
            }
            if row["goal"] != anchor_row[1] or row["index"] != expected_index:
                raise BuildError(f"raw row {line_number} canonical goal drifted")
            if config.production:
                binding = row.get("source_index_binding")
                if anchor_row[10] == source["target_sha256"]:
                    if binding is not None:
                        raise BuildError(
                            f"raw row {line_number} primary row has secondary binding"
                        )
                else:
                    if source_file not in alignment_cache:
                        anchors = _load_anchors(index, source["article"])
                        primary, _ = _strict_complete_alignment(
                            declarations,
                            anchors,
                        )
                        recovered = _secondary_unique_label_alignment(
                            declarations,
                            anchors,
                            primary,
                        )
                        alignment_cache[source_file] = (primary, recovered)
                    recovered = alignment_cache[source_file][1]
                    matches = [
                        item
                        for item in recovered
                        if item.source.ordinal == ordinal
                        and item.identity == row["theorem"]
                    ]
                    if len(matches) != 1 or binding != matches[0].source_index_binding:
                        raise BuildError(
                            f"raw row {line_number} secondary source binding drifted"
                        )

            resolution = resolve_global_citations(
                row["target"],
                index,
                theorem=row["theorem"],
            )
            if resolution.unresolved or list(resolution.references) != row["cited"]:
                raise BuildError(f"raw row {line_number} citation replay mismatch")
            expected_order = deterministic_fact_order(
                resolution.references,
                row_key=row["theorem"],
                seed=config.seed,
            )
            if list(row["facts"]) != expected_order:
                raise BuildError(f"raw row {line_number} fact order drifted")
            statements = _cached_statements(index)
            if any(row["facts"][name] != statements.get(name) for name in row["facts"]):
                raise BuildError(f"raw row {line_number} fact statement drifted")
            if row["local_assumptions"] != declaration.local_assumptions:
                raise BuildError(f"raw row {line_number} local context drifted")
            rendered, mask_start, mask_end = render_training_text(
                row["facts"],
                row["goal"],
                row["target"],
            )
            if (
                row["text"] != rendered
                or row["mask_start"] != mask_start
                or row["mask_end"] != mask_end
                or row["mask"]
                != {
                    "schema_version": MASK_SCHEMA,
                    "start": mask_start,
                    "end": mask_end,
                }
            ):
                raise BuildError(f"raw row {line_number} reconstruction mismatch")
            token_length = _tokens_with_eos(tokenizer, rendered)
            if (
                token_length != row["token_length_with_eos"]
                or token_length > MAX_TOKENS_WITH_EOS
            ):
                raise BuildError(f"raw row {line_number} token gate drifted")

            aligned = AlignedTheorem(
                declaration,
                SemanticAnchor(
                    identity=row["theorem"],
                    article=source["article"],
                    number=int(row["index"]["number"]),
                    local_label=row["index"]["local_label"],
                    source_goal=row["index"]["source_goal"],
                    mml_alignment=row["index"]["mml_alignment"],
                ),
            )
            if _row_id(aligned) != row_id:
                raise BuildError(f"raw row {line_number} deterministic id drifted")
            descriptors.append((row_id, _sample_stratum(row)))
            rows_checked += 1

    replay_ids, replay_strata = _select_replay_ids(
        descriptors,
        sample_size=config.replay_sample_size,
        seed=config.seed,
    )
    return (
        {
            "status": "clean",
            "rows_checked": rows_checked,
            "source_rows_checked": rows_checked,
            "index_rows_checked": rows_checked,
            "reference_rows_checked": rows_checked,
            "reconstruction_rows_checked": rows_checked,
        },
        replay_ids,
        replay_strata,
    )


def _raw_replay(
    raw_path: Path,
    *,
    selected_ids: Sequence[str],
    strata: Mapping[str, int],
    config: BuildConfig,
) -> dict[str, Any]:
    selected = set(selected_ids)
    checked = 0
    source_cache: dict[str, tuple[str, list[SourceTheorem]]] = {}
    with raw_path.open(encoding="utf-8") as raw_file:
        for line in raw_file:
            row = json.loads(line)
            if row["id"] not in selected:
                continue
            source = row["source"]
            source_file = source["file"]
            if source_file not in source_cache:
                path = config.mml_root / source_file
                text, _ = _read_miz(path)
                source_cache[source_file] = (
                    text,
                    parse_miz_article(
                        text,
                        article=path.stem.upper(),
                        source_file=path.name,
                    ),
                )
            text, declarations = source_cache[source_file]
            declaration = declarations[int(source["declaration_ordinal"]) - 1]
            if (
                declaration.target != row["target"]
                or text[source["target_start"] : source["target_end"]] != row["target"]
                or hashlib.sha256(row["target"].encode("utf-8")).hexdigest()
                != source["target_sha256"]
            ):
                raise BuildError(f"raw replay failed for {row['id']}")
            checked += 1
    if checked != len(selected):
        raise BuildError(
            f"raw replay selected {len(selected)} rows but checked {checked}"
        )
    return {
        "status": "clean",
        "method": "deterministic-token-fact-local-strata-v1",
        "sample_size_requested": config.replay_sample_size,
        "rows_checked": checked,
        "strata": dict(strata),
        "row_ids": list(selected_ids),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _final_source_check(config: BuildConfig) -> None:
    if _file_sha256(config.semantic_index) != config.semantic_index_sha256:
        raise BuildError("semantic index changed during the build")
    try:
        verify_source_manifest(
            config.source_manifest,
            {
                "mml": config.mml_root,
                "html": config.html_root,
                "thproofs": config.thproofs_root,
            },
            archive_paths={
                "mml": config.mizar_archive,
                "html": config.html_archive,
                "thproofs": config.thproofs_archive,
            },
        )
    except SourceVerificationError as error:
        raise BuildError(f"source changed during the build: {error}") from error


def build_corpus(config: BuildConfig, tokenizer: Any) -> dict[str, Any]:
    """Build, deeply verify, and atomically publish one fresh raw staging tree."""

    started = time.perf_counter()
    manifest, index_metadata, source_metadata, family_manifest = _verify_inputs(
        config,
        tokenizer,
    )
    config.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.out.parent / f".{config.out.name}.tmp.{os.getpid()}"
    if temporary.exists():
        raise BuildError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    stage_db = temporary / ".dedup.sqlite"
    raw_path = temporary / "raw" / f"{config.name}.jsonl"
    raw_path.parent.mkdir(parents=True)

    counters = _initialize_counters()
    fact_frequencies: Counter[str] = Counter()
    primary_fact_frequencies: Counter[str] = Counter()
    unresolved_frequencies: Counter[str] = Counter()
    token_lengths: list[int] = []
    primary_token_lengths: list[int] = []
    primary_raw_digest = hashlib.sha256()
    recovery_pending: list[tuple[AlignedTheorem, str, str]] = []
    recovery_candidates: list[dict[str, Any]] = []
    recovery_evidence = _recovery_evidence(())
    local_assumption_rows = 0
    connection: sqlite3.Connection | None = None
    index: MizarIndex | None = None
    try:
        connection = sqlite3.connect(stage_db)
        connection.execute(
            "CREATE TABLE seen_text(hash TEXT PRIMARY KEY, text TEXT NOT NULL)"
        )
        index = MizarIndex(config.semantic_index)
        _cached_statements(index)
        _cached_local_maps(index)

        mml_paths = sorted(config.mml_root.glob("*.miz"), key=lambda path: path.name)
        counters["source_files"] = len(mml_paths)
        with raw_path.open("w", encoding="utf-8", newline="\n") as output:
            for path in mml_paths:
                text, source_encoding = _read_miz(path)
                article = path.stem.upper()
                declarations = parse_miz_article(
                    text,
                    article=article,
                    source_file=path.name,
                )
                counters["declarations_total"] += len(declarations)
                anchors = _load_anchors(index, article)
                source_file_sha256 = _file_sha256(path)
                if config.production:
                    aligned, unanchored = _strict_complete_alignment(
                        declarations,
                        anchors,
                    )
                    counters["dropped_source_index_unanchored"] += unanchored
                    recovery_pending.extend(
                        (
                            item,
                            source_file_sha256,
                            source_encoding,
                        )
                        for item in _secondary_unique_label_alignment(
                            declarations,
                            anchors,
                            aligned,
                        )
                    )
                else:
                    aligned = align_article_declarations(declarations, anchors)
                by_ordinal = {item.source.ordinal: item for item in aligned}

                for declaration in declarations:
                    category = declaration.category
                    if category == "canceled":
                        counters["dropped_canceled"] += 1
                        continue
                    if category == "inline_justification":
                        counters["dropped_inline_justification"] += 1
                        continue
                    if category == "no_explicit_proof":
                        counters["dropped_no_explicit_proof"] += 1
                        continue
                    if category == "malformed_declaration":
                        counters["dropped_malformed_declaration"] += 1
                        continue
                    if category == "malformed_explicit_proof":
                        counters["dropped_malformed_explicit_proof"] += 1
                        continue
                    counters["complete_explicit_declarations"] += 1
                    aligned_theorem = by_ordinal.get(declaration.ordinal)
                    if aligned_theorem is None:
                        continue
                    counters["mapped_complete_declarations"] += 1

                    resolution = resolve_global_citations(
                        declaration.target or "",
                        index,
                        theorem=aligned_theorem.identity,
                    )
                    if resolution.unresolved:
                        counters["dropped_unresolved_reference"] += 1
                        unresolved_frequencies.update(resolution.unresolved)
                        continue
                    if not resolution.references:
                        counters["dropped_no_global_citation"] += 1
                        continue
                    order = deterministic_fact_order(
                        resolution.references,
                        row_key=aligned_theorem.identity,
                        seed=config.seed,
                    )
                    statements = _cached_statements(index)
                    facts = {name: statements[name] for name in order}
                    text_row, mask_start, mask_end = render_training_text(
                        facts,
                        aligned_theorem.anchor.statement,
                        declaration.target or "",
                    )
                    token_length = _tokens_with_eos(tokenizer, text_row)
                    if token_length > MAX_TOKENS_WITH_EOS:
                        counters["dropped_overlength"] += 1
                        continue
                    text_hash = hashlib.sha256(text_row.encode("utf-8")).hexdigest()
                    if not _insert_unique(connection, text_hash, text_row):
                        counters["dropped_duplicate"] += 1
                        continue
                    record = _record(
                        aligned_theorem,
                        source_file_sha256=source_file_sha256,
                        source_encoding=source_encoding,
                        resolution=resolution,
                        facts=facts,
                        text=text_row,
                        mask_start=mask_start,
                        mask_end=mask_end,
                        token_length=token_length,
                        source_metadata=source_metadata,
                        seed=config.seed,
                    )
                    serialized = (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    output.write(serialized)
                    primary_raw_digest.update(serialized.encode("utf-8"))
                    counters["accepted_rows"] += 1
                    fact_frequencies.update(resolution.references)
                    primary_fact_frequencies.update(resolution.references)
                    token_lengths.append(token_length)
                    primary_token_lengths.append(token_length)
                    local_assumption_rows += int(bool(declaration.local_assumptions))

            if config.production:
                _validate_primary_snapshot(
                    counters,
                    raw_sha256=primary_raw_digest.hexdigest(),
                    token_lengths=primary_token_lengths,
                    fact_frequencies=primary_fact_frequencies,
                )
                statements = _cached_statements(index)
                for (
                    aligned_theorem,
                    source_file_sha256,
                    source_encoding,
                ) in recovery_pending:
                    declaration = aligned_theorem.source
                    resolution = resolve_global_citations(
                        declaration.target or "",
                        index,
                        theorem=aligned_theorem.identity,
                    )
                    if resolution.unresolved or not resolution.references:
                        continue
                    order = deterministic_fact_order(
                        resolution.references,
                        row_key=aligned_theorem.identity,
                        seed=config.seed,
                    )
                    if any(name not in statements for name in order):
                        continue
                    facts = {name: statements[name] for name in order}
                    text_row, mask_start, mask_end = render_training_text(
                        facts,
                        aligned_theorem.anchor.statement,
                        declaration.target or "",
                    )
                    token_length = _tokens_with_eos(tokenizer, text_row)
                    if token_length > MAX_TOKENS_WITH_EOS:
                        continue
                    if (
                        aligned_theorem.anchor.proof_category
                        != "malformed_explicit_proof"
                        or aligned_theorem.anchor.proof_sha256 is not None
                    ):
                        continue
                    text_hash = hashlib.sha256(text_row.encode("utf-8")).hexdigest()
                    if not _insert_unique(connection, text_hash, text_row):
                        continue
                    record = _record(
                        aligned_theorem,
                        source_file_sha256=source_file_sha256,
                        source_encoding=source_encoding,
                        resolution=resolution,
                        facts=facts,
                        text=text_row,
                        mask_start=mask_start,
                        mask_end=mask_end,
                        token_length=token_length,
                        source_metadata=source_metadata,
                        seed=config.seed,
                    )
                    recovery_candidates.append(
                        {
                            "record": record,
                            "serialized": (
                                json.dumps(
                                    record,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            ),
                            "text_sha256": text_hash,
                        }
                    )
                recovery_evidence = _recovery_evidence(recovery_candidates)
                _validate_recovery_evidence(recovery_evidence)
                for candidate in recovery_candidates:
                    record = candidate["record"]
                    output.write(candidate["serialized"])
                    counters["mapped_complete_declarations"] += 1
                    counters["dropped_source_index_unanchored"] -= 1
                    counters["recovered_unique_label"] += 1
                    counters["accepted_rows"] += 1
                    fact_frequencies.update(record["cited"])
                    token_lengths.append(record["token_length_with_eos"])
                    local_assumption_rows += int(bool(record["local_assumptions"]))
        connection.commit()

        if counters["accepted_rows"] == 0:
            raise BuildError("no accepted direct human-Mizar rows")
        if config.production:
            _validate_production_counts(counters)

        deep_check, replay_ids, replay_strata = _deep_self_check(
            raw_path,
            config=config,
            tokenizer=tokenizer,
            index=index,
        )
        raw_replay = _raw_replay(
            raw_path,
            selected_ids=replay_ids,
            strata=replay_strata,
            config=config,
        )
        context = _context_report(
            token_lengths,
            overlength=counters["dropped_overlength"],
            local_rows=local_assumption_rows,
        )
        frequencies = dict(sorted(fact_frequencies.items()))
        if config.production:
            if sum(token_lengths) != EXPECTED_TOTAL_TOKENS:
                raise BuildError("production total Qwen token count drifted")
            if len(frequencies) < EXPECTED_DISTINCT_FACTS:
                raise BuildError("production distinct fact coverage regressed")
        frequency_path = temporary / "reports" / f"{config.name}.fact_frequencies.json"
        _write_json(frequency_path, frequencies)
        raw_sha256 = _file_sha256(raw_path)
        licensing_blocker = {
            "blocked": True,
            "redistribution_rights_asserted": manifest["licensing"][
                "redistribution_rights_asserted"
            ],
            "status": manifest["licensing"]["status"],
            "required_action": "legal review before publication or redistribution",
        }
        manifest_path = temporary / "manifests" / f"{config.name}.json"
        _write_json(manifest_path, family_manifest)
        manifest_sha256 = _file_sha256(manifest_path)

        elapsed = time.perf_counter() - started
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        report = {
            "schema_version": BUILD_REPORT_SCHEMA,
            "family": "mizar",
            "corpus": config.name,
            "mode": "raw_staging",
            "heldout": 0,
            "counters": dict(counters),
            "context_eligibility": context,
            "fact_frequencies": frequencies,
            "unresolved_reference_frequencies": dict(
                sorted(unresolved_frequencies.items())
            ),
            "direct_mizar_recovery": recovery_evidence,
            "deep_self_check": deep_check,
            "raw_replay": raw_replay,
            "source_index": {
                "source_declarations": counters["declarations_total"],
                "mapped_complete_declarations": counters[
                    "mapped_complete_declarations"
                ],
                "semantic_statements": index_metadata["content"]["statement_count"],
                "semantic_theorems": index_metadata["content"]["theorem_count"],
                "semantic_index_sha256": config.semantic_index_sha256,
                "source_manifest_sha256": _file_sha256(config.source_manifest),
            },
            "output_hashes": {
                "raw_jsonl_sha256": raw_sha256,
                "manifest_sha256": manifest_sha256,
                "fact_frequencies_sha256": _file_sha256(frequency_path),
            },
            "performance": {
                "runtime_seconds": elapsed,
                "peak_rss_mb": peak_rss_mb,
            },
            "licensing_blocker": licensing_blocker,
        }
        report_path = temporary / "reports" / f"{config.name}.build.json"
        _write_json(report_path, report)
        checksums = {
            "schema_version": "mizar-human-output-checksums-v1",
            "files": {
                f"manifests/{config.name}.json": manifest_sha256,
                f"raw/{config.name}.jsonl": raw_sha256,
                f"reports/{config.name}.build.json": _file_sha256(report_path),
                f"reports/{config.name}.fact_frequencies.json": _file_sha256(
                    frequency_path
                ),
            },
        }
        _write_json(
            temporary / "checksums" / f"{config.name}.json",
            checksums,
        )
        _final_source_check(config)
        connection.close()
        connection = None
        index.close()
        index = None
        stage_db.unlink(missing_ok=True)
        os.replace(temporary, config.out)
        return {
            **report,
            "logical_raw_path": f"raw/{config.name}.jsonl",
        }
    except BuildError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        json.JSONDecodeError,
    ) as error:
        raise BuildError(f"direct human-Mizar build failed: {error}") from error
    finally:
        if connection is not None:
            connection.close()
        if index is not None:
            index.close()
        shutil.rmtree(temporary, ignore_errors=True)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the production-only command-line contract."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mml-root", required=True)
    parser.add_argument("--html-root", required=True)
    parser.add_argument("--thproofs-root", required=True)
    parser.add_argument("--semantic-index", required=True)
    parser.add_argument("--semantic-index-sha256", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--mizar-archive", required=True)
    parser.add_argument("--html-archive", required=True)
    parser.add_argument("--thproofs-archive", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="mizar")
    parser.add_argument("--heldout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one fresh production build."""

    args = create_argument_parser().parse_args(argv)
    try:
        tokenizer = load_vendored_tokenizer(args.tokenizer_path)
        report = build_corpus(
            BuildConfig(
                mml_root=Path(args.mml_root),
                html_root=Path(args.html_root),
                thproofs_root=Path(args.thproofs_root),
                semantic_index=Path(args.semantic_index),
                semantic_index_sha256=args.semantic_index_sha256,
                source_manifest=Path(args.source_manifest),
                mizar_archive=Path(args.mizar_archive),
                html_archive=Path(args.html_archive),
                thproofs_archive=Path(args.thproofs_archive),
                out=Path(args.out),
                name=args.name,
                heldout=args.heldout,
                seed=args.seed,
                production=True,
            ),
            tokenizer,
        )
    except BuildError as error:
        print(f"direct human-Mizar build refused: {error}", file=sys.stderr)
        return 2
    summary = {
        "counters": report["counters"],
        "context_eligibility": report["context_eligibility"],
        "deep_self_check": report["deep_self_check"],
        "raw_replay": {
            "rows_checked": report["raw_replay"]["rows_checked"],
            "strata": report["raw_replay"]["strata"],
        },
        "output_hashes": report["output_hashes"],
        "performance": report["performance"],
        "licensing_blocker": report["licensing_blocker"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
