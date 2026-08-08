"""Deterministic adapter for Mizar 8.1.15 / MML 5.94.1493 sources.

The semantic HTML is authoritative for theorem identities and expanded
statements. Official ``.miz`` files are used as a source cross-check, while
``thproofs`` supplies source goals and proof-completion diagnostics.

This module is intentionally independent of the active corpus builders.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import resource
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Self
from urllib.parse import quote

SOURCE_MANIFEST_SCHEMA = "mizar-current-sources-v1"
FLAT_TREE_HASH_SCHEMA = "flat-basename-sha256-v1"
INDEX_SCHEMA = "mizar-semantic-index-v1"
INDEX_RECORD_SCHEMA = "mizar-semantic-record-v1"
INDEX_REPORT_SCHEMA = "mizar-semantic-index-report-v1"
SQLITE_USER_VERSION = 1
SQLITE_APPLICATION_ID = 0x4D5A5231  # "MZR1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TARGET_ABOUT_RE = re.compile(r"^#(DT|T|S)([1-9]\d*)$")
_IDENTITY_TEXT_RE = re.compile(
    r"\b([A-Z][A-Z0-9_]*)\s*:\s*"
    r"(?:(def|sch)\s*_?\s*([1-9]\d*)|([1-9]\d*))\b",
    re.IGNORECASE,
)
_THPROOF_NAME_RE = re.compile(r"^t([1-9]\d*)_([A-Za-z0-9_]+)$")
_THEOREM_START_RE = re.compile(r"(?m)^[ \t]*theorem\b")
_EXPLICIT_IDENTITY_RE = re.compile(
    r"::\s*([A-Z][A-Z0-9_]*):(\d+)\b", re.IGNORECASE
)
_INLINE_JUSTIFICATION_RE = re.compile(
    r"\s+(?:by|from)\s+[^;]*;\s*$", re.IGNORECASE | re.DOTALL
)
_TRUNCATED_JUSTIFICATION_RE = re.compile(
    r"\s+(?:by|from)\s+[^;]*$", re.IGNORECASE | re.DOTALL
)
_VOID_HTML_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
_BLOCK_OPENERS = frozenset({"now", "hereby", "suppose", "case", "percases"})
_KIND_ORDER = {"theorem": 0, "definition": 1, "scheme": 2}


class MizarIndexError(RuntimeError):
    """Base class for deterministic-index failures."""


class SourceVerificationError(MizarIndexError):
    """A source tree, manifest, archive, or expected count did not match."""


class MalformedSemanticHtml(MizarIndexError):
    """A semantic HTML record is incomplete or structurally inconsistent."""


class DuplicateIdentityError(MizarIndexError):
    """Two semantic HTML records claim the same canonical identity."""


class DuplicateLocalLabelError(MizarIndexError):
    """An article-local label maps to more than one identity."""


@dataclass(frozen=True)
class TreeDigest:
    """Digest of a verified flat source tree."""

    file_count: int
    total_bytes: int
    sha256: str


@dataclass(frozen=True)
class Provenance:
    """Stable source location for one semantic statement."""

    html_file: str
    html_anchor: str
    html_line: int
    identity_text: str


@dataclass(frozen=True)
class StatementRecord:
    """One authoritative semantic HTML statement."""

    identity: str
    article: str
    kind: str
    number: int
    local_label: str | None
    statement: str
    statement_html: str
    statement_sha256: str
    provenance: Provenance


@dataclass(frozen=True)
class ThproofRecord:
    """Classification and source goal for one thproof file."""

    file_name: str
    identity: str | None
    article: str | None
    number: int | None
    category: str
    source_goal: str | None
    explicit_identity: str | None
    proof_sha256: str | None

    @property
    def explicit_proof_bearing(self) -> bool:
        """Whether this record contributes to the explicit-proof denominator."""

        return self.category in {
            "complete_explicit_proof",
            "malformed_explicit_proof",
        }


@dataclass
class _Capture:
    anchor: str
    kind: str
    number: int
    root_depth: int
    html_line: int
    prefix: list[str]
    statement_plain: list[str]
    statement_html: list[str]
    local_label: str | None = None
    label_depth: int | None = None
    label_text: list[str] | None = None
    anchor_depth: int | None = None
    anchor_seen: bool = False
    region_depth: int | None = None
    region_started: bool = False
    region_finished: bool = False
    scheme_first_add: bool = False
    proof_depth: int | None = None
    proof_seen: bool = False

    def selected(self) -> bool:
        if self.proof_seen:
            return False
        if self.kind == "scheme":
            return self.anchor_seen
        return self.region_depth is not None and not self.region_finished


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def hash_flat_tree(root: str | Path, *, file_glob: str = "*") -> TreeDigest:
    """Hash a flat tree as sorted ``basename || NUL || SHA256(file)`` entries.

    Directories, symlinks, and an empty selection are rejected. Files that do
    not match ``file_glob`` are ignored, allowing a manifest to pin a selected
    direct-file subset such as ``*.html``.
    """

    root = Path(root)
    if not root.is_dir():
        raise SourceVerificationError(f"source root is not a directory: {root}")

    entries: list[tuple[str, bytes, int]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_symlink():
            raise SourceVerificationError(f"source tree contains symlink: {path}")
        if path.is_dir():
            raise SourceVerificationError(f"source tree is not flat: {path}")
        if not path.is_file() or not path.match(file_glob):
            continue
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                file_digest.update(chunk)
        entries.append((path.name, file_digest.digest(), size))

    if not entries:
        raise SourceVerificationError(
            f"source tree has no files matching {file_glob!r}: {root}"
        )

    digest = hashlib.sha256()
    for name, file_digest, _ in entries:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest)
    return TreeDigest(
        file_count=len(entries),
        total_bytes=sum(size for _, _, size in entries),
        sha256=digest.hexdigest(),
    )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SourceVerificationError(f"{field} must be a lowercase SHA-256")
    return value


def verify_source_manifest(
    manifest_path: str | Path,
    roots: Mapping[str, str | Path],
    *,
    archive_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    """Verify the exact manifest-pinned MML, HTML, and thproof source trees."""

    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceVerificationError(
            f"cannot read source manifest {manifest_path}: {error}"
        ) from error

    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise SourceVerificationError(
            f"source manifest must use {SOURCE_MANIFEST_SCHEMA}"
        )
    release = manifest.get("release")
    if not isinstance(release, dict) or not all(
        isinstance(release.get(key), str) and release[key]
        for key in ("mizar_version", "mml_version")
    ):
        raise SourceVerificationError(
            "source manifest must pin mizar_version and mml_version"
        )
    tree_hash = manifest.get("tree_hash")
    if (
        not isinstance(tree_hash, dict)
        or tree_hash.get("schema") != FLAT_TREE_HASH_SCHEMA
    ):
        raise SourceVerificationError(
            f"source manifest must use tree hash {FLAT_TREE_HASH_SCHEMA}"
        )
    licensing = manifest.get("licensing")
    if (
        not isinstance(licensing, dict)
        or licensing.get("redistribution_rights_asserted") is not False
    ):
        raise SourceVerificationError(
            "manifest must explicitly avoid asserting redistribution rights"
        )

    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise SourceVerificationError("source manifest must provide sources")
    archive_paths = archive_paths or {}
    for source_name in ("mml", "html", "thproofs"):
        spec = sources.get(source_name)
        if not isinstance(spec, dict):
            raise SourceVerificationError(
                f"source manifest must pin {source_name}"
            )
        if not isinstance(spec.get("archive_url"), str):
            raise SourceVerificationError(
                f"source manifest must pin {source_name}.archive_url"
            )
        archive_sha256 = _require_sha256(
            spec.get("archive_sha256"),
            f"sources.{source_name}.archive_sha256",
        )
        file_glob = spec.get("file_glob")
        file_count = spec.get("file_count")
        tree_sha256 = _require_sha256(
            spec.get("tree_sha256"),
            f"sources.{source_name}.tree_sha256",
        )
        if not isinstance(file_glob, str) or not file_glob:
            raise SourceVerificationError(
                f"source manifest must pin {source_name}.file_glob"
            )
        if not isinstance(file_count, int) or isinstance(file_count, bool):
            raise SourceVerificationError(
                f"source manifest must pin {source_name}.file_count"
            )
        root = roots.get(source_name)
        if root is None:
            raise SourceVerificationError(f"missing root for {source_name}")
        observed = hash_flat_tree(root, file_glob=file_glob)
        if (
            observed.file_count != file_count
            or observed.sha256 != tree_sha256
        ):
            raise SourceVerificationError(
                f"{source_name} source drift: expected {file_count} files / "
                f"{tree_sha256}, observed {observed.file_count} / "
                f"{observed.sha256}"
            )

        archive_path = archive_paths.get(source_name)
        if archive_path is not None:
            archive_path = Path(archive_path)
            observed_archive = _sha256_file(archive_path)
            if observed_archive != archive_sha256:
                raise SourceVerificationError(
                    f"{source_name} archive drift: expected {archive_sha256}, "
                    f"observed {observed_archive}"
                )

    policy = manifest.get("proof_policy")
    if not isinstance(policy, dict) or policy.get(
        "completion_denominator"
    ) != "explicit_proof_bearing_extracts":
        raise SourceVerificationError(
            "proof policy must use explicit_proof_bearing_extracts"
        )
    minimum_rate = policy.get("minimum_explicit_completion_rate")
    if (
        not isinstance(minimum_rate, (int, float))
        or isinstance(minimum_rate, bool)
        or not 0 <= minimum_rate <= 1
    ):
        raise SourceVerificationError(
            "proof policy must pin minimum_explicit_completion_rate"
        )
    return manifest


def _classes(attrs: Mapping[str, str | None]) -> frozenset[str]:
    return frozenset((attrs.get("class") or "").split())


def _normalize_text(parts: Iterable[str]) -> str:
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _canonical_statement(parts: Iterable[str]) -> str:
    statement = _normalize_text(parts)
    statement = _INLINE_JUSTIFICATION_RE.sub("", statement).strip()
    return statement.rstrip(";").strip()


def _canonical_identity(
    article: str, kind_token: str | None, special: str | None, number: str
) -> str:
    article = article.upper()
    if kind_token:
        return f"{article}:{kind_token.lower()}_{special}"
    return f"{article}:{number}"


class _SemanticHtmlParser(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=False)
        self.path = path
        self.article = path.stem.upper()
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.capture: _Capture | None = None
        self.records: list[StatementRecord] = []
        self._seen: set[str] = set()

    def _inside_class(self, class_name: str) -> bool:
        return any(
            class_name in _classes(attrs) for _, attrs in self.stack
        )

    def _target(
        self, tag: str, attrs: Mapping[str, str | None]
    ) -> tuple[str, int, str] | None:
        if tag != "div":
            return None
        match = _TARGET_ABOUT_RE.fullmatch(attrs.get("about") or "")
        if match is None:
            return None
        token, raw_number = match.groups()
        semantic_type = attrs.get("typeof")
        if token in {"T", "S"} and semantic_type != "oo:Theorem":
            return None
        if token == "DT" and semantic_type != "oo:Definition":
            return None
        return (
            {"T": "theorem", "DT": "definition", "S": "scheme"}[token],
            int(raw_number),
            f"{token}{raw_number}",
        )

    def _start_label(
        self, tag: str, attrs: Mapping[str, str | None], depth: int
    ) -> None:
        capture = self.capture
        if (
            capture is None
            or tag != "span"
            or "lab" not in _classes(attrs)
            or capture.local_label is not None
            or capture.label_depth is not None
        ):
            return
        allowed = (
            capture.kind == "theorem" and not capture.anchor_seen
        ) or (
            capture.kind == "definition" and not capture.region_started
        ) or (
            capture.kind == "scheme"
            and capture.anchor_seen
            and not capture.scheme_first_add
        )
        if allowed:
            capture.label_depth = depth
            capture.label_text = []

    def _append_start(
        self,
        tag: str,
        attrs: Mapping[str, str | None],
        raw: str,
        depth: int,
    ) -> None:
        capture = self.capture
        if capture is None:
            return

        is_proof = (
            tag == "div" and attrs.get("typeof") == "oo:Proof"
        )
        if is_proof and depth > capture.root_depth:
            capture.proof_depth = depth
            capture.proof_seen = True
            return

        selected_before = capture.selected()
        starts_region = False
        if capture.kind == "theorem":
            starts_region = (
                tag == "div"
                and "add" in _classes(attrs)
                and not capture.region_started
            )
        elif capture.kind == "definition":
            starts_region = (
                tag == "span"
                and "hide" in _classes(attrs)
                and not capture.region_started
            )
        elif (
            capture.kind == "scheme"
            and tag == "div"
            and "add" in _classes(attrs)
        ):
            capture.scheme_first_add = True

        if starts_region:
            capture.region_started = True
            capture.region_depth = depth
        elif selected_before:
            capture.statement_html.append(raw)
            if tag == "br":
                capture.statement_plain.append(" ")

        if tag == "a" and (attrs.get("name") or "").upper() == capture.anchor:
            capture.anchor_depth = depth
        self._start_label(tag, attrs, depth)

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        attr_map = {name.lower(): value for name, value in attrs}
        depth = len(self.stack)
        target = self._target(tag, attr_map)
        if target is not None:
            if self.capture is not None:
                raise MalformedSemanticHtml(
                    f"nested semantic record {target[2]} in "
                    f"{self.path.name}#{self.capture.anchor}"
                )
            kind, number, anchor = target
            self.capture = _Capture(
                anchor=anchor,
                kind=kind,
                number=number,
                root_depth=depth,
                html_line=self.getpos()[0],
                prefix=[],
                statement_plain=[],
                statement_html=[],
            )

        raw = self.get_starttag_text() or f"<{tag}>"
        self._append_start(tag, attr_map, raw, depth)
        if tag not in _VOID_HTML_TAGS:
            self.stack.append((tag, attr_map))

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        attr_map = {name.lower(): value for name, value in attrs}
        raw = self.get_starttag_text() or f"<{tag}/>"
        self._append_start(tag, attr_map, raw, len(self.stack))

    def _finish_label(self, depth: int) -> None:
        capture = self.capture
        if capture is None or capture.label_depth != depth:
            return
        label = _normalize_text(capture.label_text or [])
        if label:
            capture.local_label = label.rstrip(":")
        capture.label_depth = None
        capture.label_text = None

    def _finish_capture(self) -> None:
        capture = self.capture
        if capture is None:
            return
        prefix = _normalize_text(capture.prefix)
        identities = []
        for match in _IDENTITY_TEXT_RE.finditer(prefix):
            article, kind, special, number = match.groups()
            identities.append(
                _canonical_identity(article, kind, special, number or "")
            )
        expected = (
            f"{self.article}:{capture.number}"
            if capture.kind == "theorem"
            else f"{self.article}:def_{capture.number}"
            if capture.kind == "definition"
            else f"{self.article}:sch_{capture.number}"
        )
        if expected not in identities:
            raise MalformedSemanticHtml(
                f"{self.path.name}#{capture.anchor} lacks matching identity "
                f"text {expected!r}"
            )
        statement = _canonical_statement(capture.statement_plain)
        if not capture.region_started and capture.kind != "scheme":
            raise MalformedSemanticHtml(
                f"{self.path.name}#{capture.anchor} lacks a statement region"
            )
        if not capture.anchor_seen and capture.kind in {"theorem", "scheme"}:
            raise MalformedSemanticHtml(
                f"{self.path.name}#{capture.anchor} lacks its HTML anchor"
            )
        if not statement:
            raise MalformedSemanticHtml(
                f"{self.path.name}#{capture.anchor} has an empty statement"
            )
        if expected in self._seen:
            raise DuplicateIdentityError(
                f"duplicate identity {expected} in {self.path.name}"
            )
        self._seen.add(expected)
        self.records.append(
            StatementRecord(
                identity=expected,
                article=self.article,
                kind=capture.kind,
                number=capture.number,
                local_label=capture.local_label,
                statement=statement,
                statement_html="".join(capture.statement_html).strip(),
                statement_sha256=hashlib.sha256(
                    statement.encode("utf-8")
                ).hexdigest(),
                provenance=Provenance(
                    html_file=self.path.name,
                    html_anchor=capture.anchor,
                    html_line=capture.html_line,
                    identity_text=expected,
                ),
            )
        )
        self.capture = None

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if not self.stack:
            return
        match_depth = next(
            (
                index
                for index in range(len(self.stack) - 1, -1, -1)
                if self.stack[index][0] == tag
            ),
            None,
        )
        if match_depth is None:
            return
        capture = self.capture
        if (
            capture is not None
            and match_depth < len(self.stack) - 1
            and len(self.stack) - 1 >= capture.root_depth
        ):
            raise MalformedSemanticHtml(
                f"mismatched HTML inside {self.path.name}#{capture.anchor}"
            )

        if capture is not None:
            selected_before = capture.selected()
            is_region_root = match_depth == capture.region_depth
            is_capture_root = match_depth == capture.root_depth
            if (
                selected_before
                and not is_region_root
                and not is_capture_root
            ):
                capture.statement_html.append(f"</{tag}>")
            self._finish_label(match_depth)
            if match_depth == capture.anchor_depth:
                capture.anchor_seen = True
                capture.anchor_depth = None
            if is_region_root:
                capture.region_finished = True
                capture.region_depth = None
            if match_depth == capture.proof_depth:
                capture.proof_depth = None
            if is_capture_root:
                self._finish_capture()

        del self.stack[match_depth:]

    def handle_data(self, data: str) -> None:
        capture = self.capture
        if capture is None:
            return
        if capture.label_depth is not None and capture.label_text is not None:
            capture.label_text.append(data)
        if capture.selected():
            capture.statement_html.append(data)
            if not self._inside_class("comment"):
                capture.statement_plain.append(data)
        elif not capture.proof_seen:
            capture.prefix.append(data)

    def handle_entityref(self, name: str) -> None:
        capture = self.capture
        if capture is None:
            return
        raw = f"&{name};"
        if capture.selected():
            capture.statement_html.append(raw)
            if not self._inside_class("comment"):
                capture.statement_plain.append(html.unescape(raw))
        elif not capture.proof_seen:
            capture.prefix.append(html.unescape(raw))

    def handle_charref(self, name: str) -> None:
        capture = self.capture
        if capture is None:
            return
        raw = f"&#{name};"
        if capture.selected():
            capture.statement_html.append(raw)
            if not self._inside_class("comment"):
                capture.statement_plain.append(html.unescape(raw))
        elif not capture.proof_seen:
            capture.prefix.append(html.unescape(raw))

    def handle_comment(self, data: str) -> None:
        capture = self.capture
        if capture is not None and capture.selected():
            capture.statement_html.append(f"<!--{data}-->")

    def finish(self) -> None:
        self.close()
        if self.capture is not None:
            raise MalformedSemanticHtml(
                f"unclosed semantic record {self.path.name}#"
                f"{self.capture.anchor}"
            )


def parse_html_article(
    path: str | Path, *, chunk_size: int = 64 * 1024
) -> Sequence[StatementRecord]:
    """Stream one current semantic HTML article into authoritative records."""

    path = Path(path)
    parser = _SemanticHtmlParser(path)
    try:
        with path.open(encoding="utf-8") as source:
            while chunk := source.read(chunk_size):
                parser.feed(chunk)
        parser.finish()
    except UnicodeDecodeError as error:
        raise MalformedSemanticHtml(
            f"{path.name} is not valid UTF-8: {error}"
        ) from error
    return tuple(parser.records)


def theorem_identity(file_name: str | Path) -> str:
    """Convert a thproof filename such as ``t36_partpr_1`` to its identity."""

    name = Path(file_name).name
    match = _THPROOF_NAME_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid thproof filename: {name}")
    number, article = match.groups()
    return f"{article.upper()}:{int(number)}"


def _masked_mizar(text: str) -> str:
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


def _is_block_opener(token: str) -> bool:
    if token == "proof":
        return True
    lowered = token.lower()
    return (
        lowered in _BLOCK_OPENERS
        or lowered.startswith(("now__", "percases"))
        or re.fullmatch(r"(?:suppose|case)[a-z]\w*", lowered) is not None
    )


def _outer_proof_bounds(
    content: str,
) -> tuple[int, int | None, int | None, int | None] | None:
    masked = _masked_mizar(content)
    tokens = list(re.finditer(r"\b[A-Za-z_]\w*\b|;", masked))
    proof_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.group(0) == "proof"
        ),
        None,
    )
    if proof_index is None:
        return None
    proof_token = tokens[proof_index]
    stack = ["proof"]
    for index in range(proof_index + 1, len(tokens)):
        raw_token = tokens[index].group(0)
        if _is_block_opener(raw_token):
            stack.append(raw_token)
            continue
        if raw_token.lower() != "end":
            continue
        if index + 1 >= len(tokens) or tokens[index + 1].group(0) != ";":
            return proof_token.start(), None, None, None
        stack.pop()
        if stack:
            continue
        return (
            proof_token.start(),
            proof_token.end(),
            tokens[index].start(),
            tokens[index + 1].end(),
        )
    return proof_token.start(), None, None, None


def _strip_mizar_comments(text: str) -> str:
    return "\n".join(line.split("::", 1)[0] for line in text.splitlines())


def _canonical_source_goal(declaration: str) -> str:
    declaration = _strip_mizar_comments(declaration)
    declaration = re.sub(
        r"^\s*theorem\b", "", declaration, count=1, flags=re.IGNORECASE
    )
    declaration = re.sub(
        r"^\s*[A-Za-z]\w*\s*:\s*", "", declaration, count=1
    )
    declaration = re.sub(r"\s+", " ", declaration).strip()
    declaration = _INLINE_JUSTIFICATION_RE.sub("", declaration).strip()
    declaration = _TRUNCATED_JUSTIFICATION_RE.sub("", declaration).strip()
    return declaration.rstrip(";").strip()


def _comparison_key(goal: str) -> str:
    return re.sub(r"\s+", "", goal).rstrip(";")


def parse_thproof_file(path: str | Path) -> ThproofRecord:
    """Parse and classify one thproof extract without using legacy html2."""

    path = Path(path)
    try:
        identity = theorem_identity(path.name)
    except ValueError:
        return ThproofRecord(
            file_name=path.name,
            identity=None,
            article=None,
            number=None,
            category="invalid_name",
            source_goal=None,
            explicit_identity=None,
            proof_sha256=None,
        )
    article, raw_number = identity.split(":", 1)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ThproofRecord(
            file_name=path.name,
            identity=identity,
            article=article,
            number=int(raw_number),
            category="malformed_encoding",
            source_goal=None,
            explicit_identity=None,
            proof_sha256=None,
        )
    starts = list(_THEOREM_START_RE.finditer(text))
    if not starts:
        return ThproofRecord(
            file_name=path.name,
            identity=identity,
            article=article,
            number=int(raw_number),
            category="missing_theorem",
            source_goal=None,
            explicit_identity=None,
            proof_sha256=None,
        )

    chunk = text[starts[-1].start() :]
    bounds = _outer_proof_bounds(chunk)
    proof_start = bounds[0] if bounds is not None else None
    if proof_start is None:
        masked = _masked_mizar(chunk)
        semicolon = masked.find(";")
        declaration = (
            chunk[: semicolon + 1] if semicolon >= 0 else chunk
        )
    else:
        declaration = chunk[:proof_start]
    source_goal = _canonical_source_goal(declaration) or None
    explicit_match = _EXPLICIT_IDENTITY_RE.search(declaration)
    explicit_identity = (
        explicit_match.group(1).upper() if explicit_match else None
    )

    if bounds is not None:
        _, body_start, body_end, proof_end = bounds
        complete = (
            proof_end is not None
            and body_start is not None
            and body_end is not None
            and not _masked_mizar(chunk[proof_end:]).strip()
        )
        body = (
            chunk[body_start:body_end].strip()
            if complete and body_start is not None and body_end is not None
            else ""
        )
        category = (
            "complete_explicit_proof"
            if complete and body and source_goal
            else "malformed_explicit_proof"
        )
        proof_sha256 = (
            hashlib.sha256(body.encode("utf-8")).hexdigest()
            if category == "complete_explicit_proof"
            else None
        )
    else:
        compact = re.sub(r"\s+", " ", _strip_mizar_comments(declaration))
        if re.search(r"\bcanceled\s*;\s*$", compact, re.IGNORECASE):
            category = "canceled"
        elif _INLINE_JUSTIFICATION_RE.search(compact):
            category = "inline_justification"
        elif ";" in _masked_mizar(declaration):
            category = "no_explicit_proof"
        else:
            category = "malformed_declaration"
        proof_sha256 = None

    return ThproofRecord(
        file_name=path.name,
        identity=identity,
        article=article,
        number=int(raw_number),
        category=category,
        source_goal=source_goal,
        explicit_identity=explicit_identity,
        proof_sha256=proof_sha256,
    )


def summarize_thproofs(records: Iterable[ThproofRecord]) -> dict:
    """Summarize proof categories with the explicit-proof hard denominator."""

    records = list(records)
    categories = Counter(record.category for record in records)
    complete = categories["complete_explicit_proof"]
    explicit = complete + categories["malformed_explicit_proof"]
    total = len(records)
    return {
        "file_count": total,
        "categories": dict(sorted(categories.items())),
        "complete_explicit_proofs": complete,
        "explicit_proof_bearing_extracts": explicit,
        "explicit_completion_rate": complete / explicit if explicit else 0.0,
        "all_file_completion_rate": complete / total if total else 0.0,
    }


def _read_mizar_text(path: Path) -> tuple[str, bool]:
    data = path.read_bytes()
    try:
        return data.decode("utf-8"), False
    except UnicodeDecodeError:
        # MML syntax and identifiers are ASCII; some historical comments use
        # single-byte legacy encodings. Latin-1 is a lossless byte mapping and
        # avoids silently inserting U+FFFD into source diagnostics.
        return data.decode("latin-1"), True


def _iter_miz_theorem_goals(
    path: Path, *, text: str | None = None
) -> Iterator[str]:
    if text is None:
        text, _ = _read_mizar_text(path)
    starts = list(_THEOREM_START_RE.finditer(text))
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        chunk = text[start.start() : end]
        bounds = _outer_proof_bounds(chunk)
        masked = _masked_mizar(chunk)
        semicolon = masked.find(";")
        if (
            bounds is not None
            and (semicolon < 0 or bounds[0] < semicolon)
        ):
            declaration = chunk[: bounds[0]]
        else:
            declaration = (
                chunk[: semicolon + 1] if semicolon >= 0 else chunk
            )
        goal = _canonical_source_goal(declaration)
        if goal and goal.lower() != "canceled":
            yield goal


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        f"""
        PRAGMA page_size=4096;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        PRAGMA foreign_keys=ON;
        PRAGMA auto_vacuum=NONE;
        PRAGMA application_id={SQLITE_APPLICATION_ID};
        PRAGMA user_version={SQLITE_USER_VERSION};

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE statements (
            identity TEXT PRIMARY KEY,
            article TEXT NOT NULL,
            kind TEXT NOT NULL,
            number INTEGER NOT NULL,
            local_label TEXT,
            statement TEXT NOT NULL,
            statement_html TEXT NOT NULL,
            statement_sha256 TEXT NOT NULL,
            html_file TEXT NOT NULL,
            html_anchor TEXT NOT NULL,
            html_line INTEGER NOT NULL,
            identity_text TEXT NOT NULL,
            UNIQUE (article, kind, number)
        ) WITHOUT ROWID;

        CREATE TABLE local_labels (
            article TEXT NOT NULL,
            label TEXT NOT NULL,
            identity TEXT NOT NULL REFERENCES statements(identity),
            PRIMARY KEY (article, label, identity)
        ) WITHOUT ROWID;

        CREATE TABLE thproofs (
            identity TEXT PRIMARY KEY REFERENCES statements(identity),
            article TEXT NOT NULL,
            number INTEGER NOT NULL,
            file_name TEXT NOT NULL UNIQUE,
            category TEXT NOT NULL,
            source_goal TEXT,
            explicit_identity TEXT,
            proof_sha256 TEXT,
            mml_alignment TEXT
        ) WITHOUT ROWID;

        CREATE INDEX statements_order
            ON statements(article, kind, number);
        CREATE INDEX thproofs_article
            ON thproofs(article, number);
        """
    )
    return connection


def _insert_statement(
    connection: sqlite3.Connection, record: StatementRecord
) -> None:
    try:
        connection.execute(
            """
            INSERT INTO statements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.identity,
                record.article,
                record.kind,
                record.number,
                record.local_label,
                record.statement,
                record.statement_html,
                record.statement_sha256,
                record.provenance.html_file,
                record.provenance.html_anchor,
                record.provenance.html_line,
                record.provenance.identity_text,
            ),
        )
    except sqlite3.IntegrityError as error:
        raise DuplicateIdentityError(
            f"duplicate identity {record.identity}"
        ) from error
    if record.local_label:
        try:
            connection.execute(
                "INSERT INTO local_labels VALUES (?, ?, ?)",
                (record.article, record.local_label, record.identity),
            )
        except sqlite3.IntegrityError as error:
            raise DuplicateLocalLabelError(
                f"duplicate local-label occurrence {record.article}:"
                f"{record.local_label} -> {record.identity}"
            ) from error


def _align_mml(
    connection: sqlite3.Connection, mml_root: Path
) -> tuple[Counter, int]:
    statuses: Counter = Counter()
    non_utf8_files = 0
    mml_paths = {
        path.stem.upper(): path
        for path in sorted(mml_root.glob("*.miz"), key=lambda item: item.name)
    }
    articles = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT article FROM thproofs ORDER BY article"
        )
    ]
    for article in articles:
        path = mml_paths.get(article)
        if path is None:
            goals = set()
        else:
            text, used_fallback = _read_mizar_text(path)
            non_utf8_files += int(used_fallback)
            goals = {
                _comparison_key(goal)
                for goal in _iter_miz_theorem_goals(path, text=text)
            }
        rows = connection.execute(
            """
            SELECT identity, source_goal
            FROM thproofs
            WHERE article = ?
            ORDER BY number
            """,
            (article,),
        ).fetchall()
        for identity, source_goal in rows:
            status = (
                "literal_goal_match"
                if source_goal and _comparison_key(source_goal) in goals
                else "generated_or_unmatched"
            )
            statuses[status] += 1
            connection.execute(
                "UPDATE thproofs SET mml_alignment = ? WHERE identity = ?",
                (status, identity),
            )
    return statuses, non_utf8_files


def _missing_identity_numbers(
    connection: sqlite3.Connection,
) -> tuple[int, dict[str, int]]:
    total = 0
    by_kind: Counter = Counter()
    rows = connection.execute(
        """
        SELECT article, kind, COUNT(*), MAX(number)
        FROM statements
        GROUP BY article, kind
        """
    )
    for _, kind, count, maximum in rows:
        missing = maximum - count
        total += missing
        by_kind[kind] += missing
    return total, dict(sorted(by_kind.items()))


def _content_metrics(
    connection: sqlite3.Connection,
    *,
    html_article_files: int,
    thproof_summary: dict,
    thproof_join_count: int,
    missing_thproof_identities: int,
    mml_alignment: Counter,
    mml_non_utf8_files: int,
    crosscheck_articles: Sequence[str],
    crosscheck_generated_identities: Sequence[str],
) -> dict:
    counts = {
        kind: count
        for kind, count in connection.execute(
            "SELECT kind, COUNT(*) FROM statements GROUP BY kind"
        )
    }
    statement_count = sum(counts.values())
    local_label_count = connection.execute(
        "SELECT COUNT(*) FROM local_labels"
    ).fetchone()[0]
    duplicate_local_label_groups, duplicate_local_label_records = (
        connection.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(group_size), 0)
            FROM (
                SELECT COUNT(*) AS group_size
                FROM local_labels
                GROUP BY article, label
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()
    )
    duplicate_groups, duplicate_records = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(group_size), 0)
        FROM (
            SELECT COUNT(*) AS group_size
            FROM statements
            GROUP BY statement_sha256, statement
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    missing_numbers, missing_by_kind = _missing_identity_numbers(connection)
    content = {
        "html_article_files": html_article_files,
        "statement_count": statement_count,
        "theorem_count": counts.get("theorem", 0),
        "definition_count": counts.get("definition", 0),
        "scheme_count": counts.get("scheme", 0),
        "local_label_count": local_label_count,
        "duplicate_identities": 0,
        "duplicate_local_label_groups": duplicate_local_label_groups,
        "duplicate_local_label_records": duplicate_local_label_records,
        "duplicate_statement_groups": duplicate_groups,
        "duplicate_statement_records": duplicate_records,
        "missing_identity_numbers": missing_numbers,
        "missing_identity_numbers_by_kind": missing_by_kind,
        "thproof_files": thproof_summary["file_count"],
        "thproof_join_count": thproof_join_count,
        "missing_thproof_identities": missing_thproof_identities,
        "proof_categories": thproof_summary["categories"],
        "complete_explicit_proofs": thproof_summary[
            "complete_explicit_proofs"
        ],
        "explicit_proof_bearing_extracts": thproof_summary[
            "explicit_proof_bearing_extracts"
        ],
        "explicit_completion_rate": thproof_summary[
            "explicit_completion_rate"
        ],
        "all_file_completion_rate": thproof_summary[
            "all_file_completion_rate"
        ],
        "mml_alignment": dict(sorted(mml_alignment.items())),
        "mml_non_utf8_files": mml_non_utf8_files,
    }
    if crosscheck_articles:
        placeholders = ",".join("?" for _ in crosscheck_articles)
        candidate_identities = {
            row[0]
            for row in connection.execute(
            f"""
            SELECT identity
            FROM statements
            WHERE kind = 'theorem' AND article IN ({placeholders})
            """,
            tuple(crosscheck_articles),
            )
        }
        joined_identities = {
            row[0]
            for row in connection.execute(
            f"""
            SELECT identity
            FROM thproofs
            WHERE article IN ({placeholders})
            """,
            tuple(crosscheck_articles),
            )
        }
        source_matches = {
            row[0]
            for row in connection.execute(
                f"""
                SELECT identity
                FROM thproofs
                WHERE article IN ({placeholders})
                  AND mml_alignment = 'literal_goal_match'
                """,
                tuple(crosscheck_articles),
            )
        }
        generated = (
            set(crosscheck_generated_identities)
            & candidate_identities
            - source_matches
        )
        agreements = source_matches | generated
        content["sample_candidate_count"] = len(candidate_identities)
        content["sample_thproof_join_count"] = len(joined_identities)
        content["sample_source_goal_match_count"] = len(source_matches)
        content["sample_generated_identity_count"] = len(generated)
        content["sample_agreement_count"] = len(agreements)
        content["sample_mismatch_count"] = (
            len(candidate_identities) - len(agreements)
        )
    return content


def _check_expected(manifest: Mapping[str, object], content: Mapping[str, object]) -> None:
    expected = manifest.get("expected", {})
    if not isinstance(expected, dict):
        raise SourceVerificationError("manifest expected field must be an object")
    for key, expected_value in expected.items():
        if key not in content:
            raise SourceVerificationError(
                f"manifest expected count has no measured metric: {key}"
            )
        observed = content[key]
        if observed != expected_value:
            raise SourceVerificationError(
                f"{key} mismatch: expected {expected_value!r}, "
                f"observed {observed!r}"
            )
    minimum_rate = manifest["proof_policy"][
        "minimum_explicit_completion_rate"
    ]
    observed_rate = content["explicit_completion_rate"]
    if observed_rate < minimum_rate:
        raise SourceVerificationError(
            "explicit proof completion gate failed: "
            f"{observed_rate:.6%} < {minimum_rate:.6%}"
        )


def _jsonl_record(
    statement_row: sqlite3.Row | tuple,
) -> dict:
    (
        identity,
        article,
        kind,
        number,
        local_label,
        statement,
        statement_html,
        statement_sha256,
        html_file,
        html_anchor,
        html_line,
        identity_text,
        thproof_file,
        category,
        source_goal,
        explicit_identity,
        proof_sha256,
        mml_alignment,
    ) = statement_row
    thproof = None
    if thproof_file is not None:
        thproof = {
            "file_name": thproof_file,
            "category": category,
            "source_goal": source_goal,
            "explicit_identity": explicit_identity,
            "proof_sha256": proof_sha256,
            "mml_alignment": mml_alignment,
        }
    return {
        "schema_version": INDEX_RECORD_SCHEMA,
        "identity": identity,
        "article": article,
        "kind": kind,
        "number": number,
        "local_label": local_label,
        "statement": statement,
        "statement_html": statement_html,
        "statement_sha256": statement_sha256,
        "provenance": {
            "html_file": html_file,
            "html_anchor": html_anchor,
            "html_line": html_line,
            "identity_text": identity_text,
        },
        "thproof": thproof,
    }


def _write_jsonl(
    connection: sqlite3.Connection, path: Path
) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    query = """
        SELECT
            s.identity, s.article, s.kind, s.number, s.local_label,
            s.statement, s.statement_html, s.statement_sha256,
            s.html_file, s.html_anchor, s.html_line, s.identity_text,
            t.file_name, t.category, t.source_goal, t.explicit_identity,
            t.proof_sha256, t.mml_alignment
        FROM statements AS s
        LEFT JOIN thproofs AS t ON t.identity = s.identity
        ORDER BY
            s.article,
            CASE s.kind
                WHEN 'theorem' THEN 0
                WHEN 'definition' THEN 1
                ELSE 2
            END,
            s.number
    """
    with path.open("wb") as output:
        for row in connection.execute(query):
            encoded = (
                _canonical_json(_jsonl_record(row)).encode("utf-8") + b"\n"
            )
            output.write(encoded)
            digest.update(encoded)
            count += 1
    return digest.hexdigest(), count


def _set_metadata(
    connection: sqlite3.Connection, key: str, value: object
) -> None:
    connection.execute(
        "INSERT INTO metadata VALUES (?, ?)",
        (key, _canonical_json(value)),
    )


def build_index(
    *,
    manifest_path: str | Path,
    roots: Mapping[str, str | Path],
    sqlite_path: str | Path,
    jsonl_path: str | Path,
    archive_paths: Mapping[str, str | Path] | None = None,
) -> dict:
    """Build deterministic SQLite and JSONL indexes from verified sources."""

    started = time.perf_counter()
    manifest_path = Path(manifest_path)
    sqlite_path = Path(sqlite_path)
    jsonl_path = Path(jsonl_path)
    root_paths = {key: Path(value) for key, value in roots.items()}
    manifest_sha256 = _sha256_file(manifest_path)
    manifest = verify_source_manifest(
        manifest_path, root_paths, archive_paths=archive_paths
    )
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_temp = sqlite_path.with_name(
        f".{sqlite_path.name}.tmp.{os.getpid()}"
    )
    jsonl_temp = jsonl_path.with_name(
        f".{jsonl_path.name}.tmp.{os.getpid()}"
    )
    for path in (sqlite_temp, jsonl_temp):
        path.unlink(missing_ok=True)

    connection: sqlite3.Connection | None = None
    try:
        connection = _create_database(sqlite_temp)
        html_paths = sorted(
            root_paths["html"].glob(
                manifest["sources"]["html"]["file_glob"]
            ),
            key=lambda item: item.name,
        )
        for html_path in html_paths:
            records = sorted(
                parse_html_article(html_path),
                key=lambda record: (
                    record.article,
                    _KIND_ORDER[record.kind],
                    record.number,
                ),
            )
            for record in records:
                _insert_statement(connection, record)

        categories: Counter = Counter()
        thproof_file_count = 0
        thproof_join_count = 0
        missing_thproof_identities = 0
        thproof_paths = sorted(
            root_paths["thproofs"].glob(
                manifest["sources"]["thproofs"]["file_glob"]
            ),
            key=lambda item: item.name,
        )
        for thproof_path in thproof_paths:
            if not thproof_path.is_file():
                continue
            thproof_file_count += 1
            record = parse_thproof_file(thproof_path)
            categories[record.category] += 1
            if record.identity is None:
                missing_thproof_identities += 1
                continue
            exists = connection.execute(
                """
                SELECT 1 FROM statements
                WHERE identity = ? AND kind = 'theorem'
                """,
                (record.identity,),
            ).fetchone()
            if exists is None:
                missing_thproof_identities += 1
                continue
            connection.execute(
                """
                INSERT INTO thproofs (
                    identity, article, number, file_name, category,
                    source_goal, explicit_identity, proof_sha256, mml_alignment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record.identity,
                    record.article,
                    record.number,
                    record.file_name,
                    record.category,
                    record.source_goal,
                    record.explicit_identity,
                    record.proof_sha256,
                ),
            )
            thproof_join_count += 1

        complete = categories["complete_explicit_proof"]
        explicit = complete + categories["malformed_explicit_proof"]
        thproof_summary = {
            "file_count": thproof_file_count,
            "categories": dict(sorted(categories.items())),
            "complete_explicit_proofs": complete,
            "explicit_proof_bearing_extracts": explicit,
            "explicit_completion_rate": complete / explicit if explicit else 0.0,
            "all_file_completion_rate": (
                complete / thproof_file_count if thproof_file_count else 0.0
            ),
        }
        mml_alignment, mml_non_utf8_files = _align_mml(
            connection, root_paths["mml"]
        )
        crosscheck = manifest.get("crosscheck", {})
        crosscheck_articles = (
            [article.upper() for article in crosscheck.get("sample_articles", [])]
            if isinstance(crosscheck, dict)
            else []
        )
        crosscheck_generated_identities = (
            [
                identity.upper()
                for identity in crosscheck.get("generated_identities", [])
            ]
            if isinstance(crosscheck, dict)
            else []
        )
        content = _content_metrics(
            connection,
            html_article_files=len(html_paths),
            thproof_summary=thproof_summary,
            thproof_join_count=thproof_join_count,
            missing_thproof_identities=missing_thproof_identities,
            mml_alignment=mml_alignment,
            mml_non_utf8_files=mml_non_utf8_files,
            crosscheck_articles=crosscheck_articles,
            crosscheck_generated_identities=crosscheck_generated_identities,
        )
        if _sha256_file(manifest_path) != manifest_sha256:
            raise SourceVerificationError(
                "source manifest changed during build"
            )
        _check_expected(manifest, content)

        jsonl_sha256, jsonl_records = _write_jsonl(
            connection, jsonl_temp
        )
        if jsonl_records != content["statement_count"]:
            raise MizarIndexError(
                "JSONL record count differs from statement count"
            )
        _set_metadata(connection, "schema_version", INDEX_SCHEMA)
        _set_metadata(connection, "record_schema_version", INDEX_RECORD_SCHEMA)
        _set_metadata(connection, "source_manifest_schema", SOURCE_MANIFEST_SCHEMA)
        _set_metadata(connection, "source_manifest_sha256", manifest_sha256)
        _set_metadata(connection, "release", manifest["release"])
        _set_metadata(connection, "source_trees", {
            name: {
                "file_count": spec["file_count"],
                "tree_sha256": spec["tree_sha256"],
            }
            for name, spec in sorted(manifest["sources"].items())
        })
        _set_metadata(connection, "content", content)
        _set_metadata(connection, "jsonl_sha256", jsonl_sha256)
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        connection = None

        os.replace(sqlite_temp, sqlite_path)
        os.replace(jsonl_temp, jsonl_path)
        sqlite_sha256 = _sha256_file(sqlite_path)
        elapsed = time.perf_counter() - started
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = peak_rss / 1024
        return {
            "schema_version": INDEX_REPORT_SCHEMA,
            "content": content,
            "hashes": {
                "source_manifest_sha256": manifest_sha256,
                "jsonl_sha256": jsonl_sha256,
                "sqlite_sha256": sqlite_sha256,
            },
            "performance": {
                "runtime_seconds": elapsed,
                "peak_rss_mb": peak_rss_mb,
            },
        }
    except Exception:
        if connection is not None:
            connection.close()
        sqlite_temp.unlink(missing_ok=True)
        jsonl_temp.unlink(missing_ok=True)
        raise


class MizarIndex:
    """Read-only builder-facing API over a generated SQLite index."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        uri = f"file:{quote(str(self.path))}?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def statement_map(self) -> dict[str, str]:
        """Return the canonical ``identity -> expanded statement`` mapping."""

        return dict(
            self.connection.execute(
                "SELECT identity, statement FROM statements ORDER BY identity"
            )
        )

    def article_local_label_maps(
        self,
    ) -> dict[str, dict[str, tuple[str, ...]]]:
        """Return every ordered identity for each article-local label.

        Mizar permits a local label to be reused later in an article. Call
        :meth:`resolve_local_label` when resolving a proof at a specific
        theorem; do not collapse these tuples to one global value.
        """

        pending: dict[str, dict[str, list[str]]] = {}
        for article, label, identity in self.connection.execute(
            """
            SELECT l.article, l.label, l.identity
            FROM local_labels AS l
            JOIN statements AS s ON s.identity = l.identity
            ORDER BY l.article, l.label, s.html_line, l.identity
            """
        ):
            pending.setdefault(article, {}).setdefault(label, []).append(
                identity
            )
        return {
            article: {
                label: tuple(identities)
                for label, identities in labels.items()
            }
            for article, labels in pending.items()
        }

    def resolve_local_label(
        self, article: str, label: str, *, at_identity: str
    ) -> str:
        """Resolve ``label`` to its latest declaration before ``at_identity``."""

        article = article.upper()
        target = self.connection.execute(
            """
            SELECT article, html_file, html_line
            FROM statements
            WHERE identity = ?
            """,
            (at_identity,),
        ).fetchone()
        if target is None:
            raise KeyError(at_identity)
        target_article, html_file, html_line = target
        if target_article != article:
            raise KeyError(
                f"{at_identity} is in {target_article}, not {article}"
            )
        row = self.connection.execute(
            """
            SELECT l.identity
            FROM local_labels AS l
            JOIN statements AS s ON s.identity = l.identity
            WHERE l.article = ?
              AND l.label = ?
              AND s.html_file = ?
              AND s.html_line < ?
            ORDER BY s.html_line DESC, l.identity DESC
            LIMIT 1
            """,
            (article, label, html_file, html_line),
        ).fetchone()
        if row is None:
            raise KeyError(f"{article}:{label} before {at_identity}")
        return row[0]

    def theorem_identity(self, file_name: str | Path) -> str:
        """Resolve and verify a thproof filename against this index."""

        identity = theorem_identity(file_name)
        row = self.connection.execute(
            "SELECT 1 FROM statements WHERE identity = ? AND kind = 'theorem'",
            (identity,),
        ).fetchone()
        if row is None:
            raise KeyError(identity)
        return identity

    def source_goal(self, identity: str) -> str | None:
        """Return the thproof source goal for an identity, when present."""

        row = self.connection.execute(
            "SELECT source_goal FROM thproofs WHERE identity = ?",
            (identity,),
        ).fetchone()
        return row[0] if row is not None else None

    def metadata(self) -> dict[str, object]:
        """Return deterministic index metadata."""

        return {
            key: json.loads(value)
            for key, value in self.connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }


def load_statement_map(path: str | Path) -> dict[str, str]:
    """Convenience API for the existing Mizar builders."""

    with MizarIndex(path) as index:
        return index.statement_map()


def load_article_local_label_maps(
    path: str | Path,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Convenience API for exact article-local reference resolution."""

    with MizarIndex(path) as index:
        return index.article_local_label_maps()


def load_source_goal(path: str | Path, identity: str) -> str | None:
    """Convenience API for thproof source-goal diagnostics."""

    with MizarIndex(path) as index:
        return index.source_goal(identity)


def _parse_cli(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mml", required=True)
    parser.add_argument("--html", required=True)
    parser.add_argument("--thproofs", required=True)
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--report")
    parser.add_argument("--mizar-archive")
    parser.add_argument("--html-archive")
    parser.add_argument("--thproofs-archive")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_cli(argv)
    archive_paths = {
        key: value
        for key, value in {
            "mml": args.mizar_archive,
            "html": args.html_archive,
            "thproofs": args.thproofs_archive,
        }.items()
        if value
    }
    try:
        report = build_index(
            manifest_path=args.manifest,
            roots={
                "mml": args.mml,
                "html": args.html,
                "thproofs": args.thproofs,
            },
            sqlite_path=args.sqlite,
            jsonl_path=args.jsonl,
            archive_paths=archive_paths,
        )
    except (MizarIndexError, OSError, sqlite3.Error, ValueError) as error:
        print(f"mizar index build failed: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
