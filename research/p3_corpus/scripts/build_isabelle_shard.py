"""Build the pinned Isabelle/Magnushammer adjacent-transition corpus.

The supervised decision is::

    facts + state_before -> tactic + state_after

The 2.3 GB source is read twice without materializing it. Pass one records every
qualified global-name rendering in SQLite. Pass two rejects ambiguous names,
renders eligible adjacent transitions, applies the Qwen ``text + EOS`` limit,
and stages exact-deduplicated rows in SQLite. Production uses builder-local
positive heldout mode because the shared splitter does not isolate trajectories.
The command must specify ``--heldout 500`` and the vendored tokenizer path.

This writes ``corpus/shards/isabelle.jsonl``, ``corpus/eval/isabelle.jsonl``,
and ``corpus/heldout/isabelle.json``. ``--heldout 0`` remains available only
for raw debugging/staging and is not a scientifically complete split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local assumptions:"
SEP = "---"
SCHEMA_VERSION = "isabelle-transition-v2"
BUILD_SOURCE_SCHEMA = "isabelle-build-source-v2"

SOURCE_DATASET = "Simontwice/premise_selection_in_isabelle"
SOURCE_REVISION = "f947ccc827ccd236464e19cd4cc23dfda7fc5575"
SOURCE_FILE = "raw_data/human_data/all_data.json"
SOURCE_SIZE = 2_327_313_460
SOURCE_SHA256 = "aa71609de90fee138835cfdf9e954becb1b231a293ac19bd98951e6d8bec8e7d"
SOURCE_LICENSE = "Apache-2.0"
DEFAULT_SOURCE = "/tmp/dscount/magnushammer/raw_data/human_data/all_data.json"

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
MAX_TOKENS_WITH_EOS = 16_384
MAX_PASTE_SHARE = 0.5
TAIL_CITATION_COUNTS = (1, 2)
STREAM_CHUNK_BYTES = 1024 * 1024
CANONICALIZATION_SCHEME = "quoted-layout-v2"
CANONICALIZATION_VERSION = 2

TOK = re.compile(r"[A-Za-z_][\w.']*|\\<\w+>|\S")
ABORT = re.compile(
    r"(?<![A-Za-z0-9_'.])(?:oops|sorry|abort)(?![A-Za-z0-9_'.])",
    re.IGNORECASE,
)
DECLARATION_PREFIX = re.compile(
    r"^(?:lemma|theorem|corollary|proposition)\s+",
)
EXPOSURE_BOUNDARY_CHAR = re.compile(r"[A-Za-z0-9_'.]")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _build_source_metadata(source: Mapping[str, Any]) -> dict[str, Any]:
    source_roots = {
        "magnushammer": {
            "dataset": source["dataset"],
            "file": source["file"],
            "revision": source["revision"],
            "sha256": source["sha256"],
            "size_bytes": source["size_bytes"],
        }
    }
    quality_policy = {
        "max_paste_share": MAX_PASTE_SHARE,
        "max_tokens_with_eos": MAX_TOKENS_WITH_EOS,
        "requires_adjacent_states": True,
        "requires_global_fact": True,
        "trajectory_isolation": True,
    }
    schema_policy = {
        "schema_version": SCHEMA_VERSION,
        "target": "tactic-plus-state-after",
        "goal": "theorem-plus-state-before",
    }
    return {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "source_manifest_root_sha256": _canonical_sha256(dict(source)),
        "source_roots": source_roots,
        "index_roots": {},
        "quality_filter_root_sha256": _canonical_sha256(quality_policy),
        "schema_generation_root_sha256": _canonical_sha256(schema_policy),
    }


class BuildError(RuntimeError):
    """A correctness gate that makes the requested rebuild unusable."""


@dataclass
class ParsedPremises:
    """Lossless alias views plus normalized global rendering observations."""

    global_aliases: dict[str, str]
    global_alias_statements: dict[str, str]
    global_observations: list[tuple[str, str]]
    local_assumptions: dict[str, str]
    local_names: dict[str, str]
    malformed: bool = False


@dataclass
class VendoredTokenizer:
    """The validated low-level Qwen tokenizer and its reproducibility metadata."""

    backend: Any
    identity: str
    tokenizer_json_sha256: str
    tokenizer_config_sha256: str
    behavior_digest: str
    tokenizers_version: str
    eos_token_id: int
    path: str

    @property
    def sha256(self) -> str:
        """Backward-compatible alias for the tokenizer JSON digest."""

        return self.tokenizer_json_sha256

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        """Encode without adding a second EOS token."""

        return self.backend.encode(text, add_special_tokens=add_special_tokens)


def paste_share(target: str, context: str) -> float:
    """Return unique normalized target-token carryover from context."""

    target_tokens = set(TOK.findall(normalize_layout(target)))
    if not target_tokens:
        return 1.0
    context_tokens = set(TOK.findall(normalize_layout(context)))
    return len(target_tokens & context_tokens) / len(target_tokens)


def normalize_layout(text: str) -> str:
    """Collapse layout outside quotes without rewriting Isabelle variables."""

    out: list[str] = []
    quote: str | None = None
    escaped = False
    pending_space = False
    for char in str(text).strip():
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            quote = char
            out.append(char)
        elif char.isspace():
            pending_space = True
        else:
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append(char)
    return "".join(out)


def normalize_declaration_statement(text: str) -> str:
    """Strip only a recognized leading Isabelle proposition declaration."""

    return DECLARATION_PREFIX.sub("", normalize_layout(text), count=1)


def _contains_normalized_boundary(text: str, value: str) -> bool:
    text = normalize_layout(text)
    value = normalize_layout(value)
    start = 0
    while value and (index := text.find(value, start)) >= 0:
        before = text[index - 1] if index else ""
        after_index = index + len(value)
        after = text[after_index] if after_index < len(text) else ""
        if not EXPOSURE_BOUNDARY_CHAR.match(
            before,
        ) and not EXPOSURE_BOUNDARY_CHAR.match(after):
            return True
        start = index + 1
    return False


def contains_normalized_statement(text: str, statement: str) -> bool:
    """Match a normalized proposition only at safe textual boundaries."""

    return _contains_normalized_boundary(text, statement)


def contains_qualified_name(text: str, qualified_name: str) -> bool:
    """Match a qualified Isabelle name without accepting longer identifiers."""

    return _contains_normalized_boundary(text, qualified_name)


def _required_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return normalize_layout(value)


def canonical_statement_hash(statement: str) -> str:
    """Hash an exact Isabelle statement modulo conservative layout changes."""

    payload = "\0".join(
        (
            "statement",
            str(CANONICALIZATION_VERSION),
            "isabelle",
            CANONICALIZATION_SCHEME,
            normalize_layout(statement),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pinned_source_metadata() -> dict[str, Any]:
    """Return the immutable Hugging Face source identity."""

    return {
        "dataset": SOURCE_DATASET,
        "revision": SOURCE_REVISION,
        "file": SOURCE_FILE,
        "size_bytes": SOURCE_SIZE,
        "sha256": SOURCE_SHA256,
        "license": SOURCE_LICENSE,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_file(
    source: str | os.PathLike[str],
    *,
    expected_size: int = SOURCE_SIZE,
    expected_sha256: str = SOURCE_SHA256,
) -> dict[str, Any]:
    """Require the exact pinned source bytes before parsing."""

    path = Path(source)
    if not path.is_file():
        raise BuildError(f"required pinned source is missing: {path}")
    size = path.stat().st_size
    if size == 0:
        raise BuildError(f"required pinned source is empty: {path}")
    if size != expected_size:
        raise BuildError(
            f"source byte size mismatch for {path}: "
            f"expected {expected_size}, got {size}"
        )
    digest = _file_sha256(path)
    if digest != expected_sha256:
        raise BuildError(
            f"source SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    metadata = pinned_source_metadata()
    metadata["size_bytes"] = size
    metadata["sha256"] = digest
    return metadata


def load_vendored_tokenizer(
    tokenizer_path: str | os.PathLike[str],
) -> VendoredTokenizer:
    """Load and verify a local Qwen tokenizer; network lookup is forbidden."""

    requested = Path(tokenizer_path)
    tokenizer_json = requested / "tokenizer.json" if requested.is_dir() else requested
    if not tokenizer_json.is_file():
        raise BuildError(f"vendored tokenizer is missing: {tokenizer_json}")
    if tokenizer_json.stat().st_size == 0:
        raise BuildError(f"vendored tokenizer is empty: {tokenizer_json}")

    config_path = tokenizer_json.with_name("tokenizer_config.json")
    if not config_path.is_file():
        raise BuildError(f"Qwen tokenizer config is missing: {config_path}")
    tokenizer_json_sha256 = _file_sha256(tokenizer_json)
    if tokenizer_json_sha256 != APPROVED_TOKENIZER_JSON_SHA256:
        raise BuildError(
            "tokenizer.json SHA-256 is not approved: "
            f"expected {APPROVED_TOKENIZER_JSON_SHA256}, "
            f"got {tokenizer_json_sha256}"
        )
    tokenizer_config_sha256 = _file_sha256(config_path)
    if tokenizer_config_sha256 != APPROVED_TOKENIZER_CONFIG_SHA256:
        raise BuildError(
            "tokenizer_config.json SHA-256 is not approved: "
            f"expected {APPROVED_TOKENIZER_CONFIG_SHA256}, "
            f"got {tokenizer_config_sha256}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"invalid Qwen tokenizer config: {config_path}") from error
    if (
        config.get("tokenizer_class") != "Qwen2Tokenizer"
        or config.get("eos_token") != QWEN_EOS_TOKEN
    ):
        raise BuildError(
            f"tokenizer at {requested} is not the pinned Qwen2 tokenizer family"
        )

    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as error:
        raise BuildError("the tokenizers package is required") from error
    implementation_version = str(tokenizers.__version__)
    if implementation_version != APPROVED_TOKENIZERS_VERSION:
        raise BuildError(
            "tokenizers implementation version is not approved: "
            f"expected {APPROVED_TOKENIZERS_VERSION}, "
            f"got {implementation_version}"
        )
    try:
        backend = Tokenizer.from_file(str(tokenizer_json))
    except Exception as error:
        raise BuildError(f"failed to load tokenizer: {tokenizer_json}") from error
    backend.no_truncation()
    backend.no_padding()
    eos_token_id = backend.token_to_id(QWEN_EOS_TOKEN)
    if eos_token_id != QWEN_EOS_TOKEN_ID:
        raise BuildError(
            f"Qwen EOS mismatch: expected {QWEN_EOS_TOKEN_ID}, got {eos_token_id}"
        )
    behavior_digest = _tokenizer_behavior_digest(backend)
    if behavior_digest != APPROVED_TOKENIZER_BEHAVIOR_SHA256:
        raise BuildError(
            "tokenizer behavior digest is not approved: "
            f"expected {APPROVED_TOKENIZER_BEHAVIOR_SHA256}, "
            f"got {behavior_digest}"
        )
    return VendoredTokenizer(
        backend=backend,
        identity=QWEN_TOKENIZER_ID,
        tokenizer_json_sha256=tokenizer_json_sha256,
        tokenizer_config_sha256=tokenizer_config_sha256,
        behavior_digest=behavior_digest,
        tokenizers_version=implementation_version,
        eos_token_id=eos_token_id,
        path=str(tokenizer_json.resolve()),
    )


def _tokenizer_behavior_digest(tokenizer: Any) -> str:
    """Seal encoding, offsets, tokens, decoding, vocabulary, and EOS behavior."""

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
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_metadata(tokenizer: Any) -> dict[str, Any]:
    identity = str(getattr(tokenizer, "identity", "")).strip()
    tokenizer_json_sha256 = str(
        getattr(tokenizer, "tokenizer_json_sha256", "")
    ).strip().lower()
    tokenizer_config_sha256 = str(
        getattr(tokenizer, "tokenizer_config_sha256", "")
    ).strip().lower()
    behavior_digest = str(
        getattr(tokenizer, "behavior_digest", "")
    ).strip().lower()
    implementation_version = str(
        getattr(tokenizer, "tokenizers_version", "")
    ).strip()
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not identity:
        raise BuildError("tokenizer identity is missing")
    if not SHA256.fullmatch(tokenizer_json_sha256):
        raise BuildError("tokenizer.json SHA-256 is missing or malformed")
    if not SHA256.fullmatch(tokenizer_config_sha256):
        raise BuildError("tokenizer_config.json SHA-256 is missing or malformed")
    if not SHA256.fullmatch(behavior_digest):
        raise BuildError("tokenizer behavior digest is missing or malformed")
    if not implementation_version:
        raise BuildError("tokenizers implementation version is missing")
    if eos_token_id != QWEN_EOS_TOKEN_ID:
        raise BuildError(
            f"Qwen EOS mismatch: expected {QWEN_EOS_TOKEN_ID}, got {eos_token_id}"
        )
    metadata = {
        "identity": identity,
        "tokenizer_json_sha256": tokenizer_json_sha256,
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "behavior_digest": behavior_digest,
        "tokenizers_version": implementation_version,
        "eos_token_id": eos_token_id,
        "max_text_plus_eos_tokens": MAX_TOKENS_WITH_EOS,
    }
    path = getattr(tokenizer, "path", None)
    if path:
        metadata["path"] = str(path)
    return metadata


def _tokens_with_eos(tokenizer: Any, text: str) -> int:
    try:
        encoding = tokenizer.encode(text, add_special_tokens=False)
    except Exception as error:
        raise BuildError("tokenizer failed while encoding an Isabelle row") from error
    ids = getattr(encoding, "ids", encoding)
    try:
        return len(ids) + 1
    except TypeError as error:
        raise BuildError(
            "tokenizer encode() did not return a sized token sequence"
        ) from error


def _parse_premises(raw: Any) -> ParsedPremises:
    parsed = ParsedPremises({}, {}, [], {}, {}, not isinstance(raw, Mapping))
    if not isinstance(raw, Mapping):
        return parsed
    for raw_alias, value in raw.items():
        if not isinstance(raw_alias, str):
            parsed.malformed = True
            continue
        alias = normalize_layout(raw_alias)
        if not alias or alias in parsed.global_aliases or alias in parsed.local_names:
            parsed.malformed = True
            continue
        if not isinstance(value, list) or len(value) != 2:
            parsed.malformed = True
            continue
        qualified = _required_text(value[0])
        statement = _required_text(value[1])
        if not qualified or not statement:
            parsed.malformed = True
            continue
        if qualified.startswith("local."):
            parsed.local_names[alias] = qualified
            parsed.local_assumptions[alias] = statement
        else:
            parsed.global_aliases[alias] = qualified
            parsed.global_alias_statements[alias] = statement
            parsed.global_observations.append((qualified, statement))
    return parsed


def _isabelle_code_only(text: str) -> str:
    """Remove nested comments, strings, and cartouches before keyword scanning."""

    output: list[str] = []
    index = 0
    length = len(text)
    ascii_open = r"\<open>"
    ascii_close = r"\<close>"
    while index < length:
        if text.startswith("(*", index):
            depth = 1
            index += 2
            while index < length and depth:
                if text.startswith("(*", index):
                    depth += 1
                    index += 2
                elif text.startswith("*)", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            output.append(" ")
            continue
        if text.startswith(ascii_open, index):
            depth = 1
            index += len(ascii_open)
            while index < length and depth:
                if text.startswith(ascii_open, index):
                    depth += 1
                    index += len(ascii_open)
                elif text.startswith(ascii_close, index):
                    depth -= 1
                    index += len(ascii_close)
                else:
                    index += 1
            output.append(" ")
            continue
        if text[index] == "‹":
            depth = 1
            index += 1
            while index < length and depth:
                if text[index] == "‹":
                    depth += 1
                elif text[index] == "›":
                    depth -= 1
                index += 1
            output.append(" ")
            continue
        if text[index] == '"':
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += min(2, length - index)
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            output.append(" ")
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _is_aborted(transitions: Any) -> bool:
    if not isinstance(transitions, list):
        return False
    return any(
        isinstance(transition, Mapping)
        and isinstance(transition.get("step"), str)
        and ABORT.search(_isabelle_code_only(transition["step"]))
        for transition in transitions
    )


class _IncrementalJSON:
    """Decode JSON values while retaining only one source chunk plus one value."""

    def __init__(self, source_file: Any):
        self.source_file = source_file
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _read_more(self) -> bool:
        if self.eof:
            return False
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0
        chunk = self.source_file.read(STREAM_CHUNK_BYTES)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return False

    def _skip_layout(self) -> None:
        while True:
            while (
                self.position < len(self.buffer)
                and self.buffer[self.position].isspace()
            ):
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self._read_more()

    def consume(self, expected: str) -> bool:
        """Consume one structural character if it is next."""

        self._skip_layout()
        if self.position >= len(self.buffer):
            return False
        if self.buffer[self.position] != expected:
            return False
        self.position += 1
        return True

    def expect(self, expected: str) -> None:
        """Require one structural character."""

        if self.consume(expected):
            return
        found = (
            "<eof>"
            if self.position >= len(self.buffer)
            else repr(self.buffer[self.position])
        )
        raise BuildError(f"expected {expected!r} in source JSON, found {found}")

    def value(self) -> Any:
        """Decode one complete JSON value, reading more chunks as needed."""

        while True:
            self._skip_layout()
            if self.position >= len(self.buffer) and self.eof:
                raise BuildError("source JSON ended before the next value")
            try:
                value, end = self.decoder.raw_decode(
                    self.buffer,
                    self.position,
                )
            except json.JSONDecodeError as error:
                if self.eof:
                    raise BuildError(
                        f"invalid source JSON near character {error.pos}"
                    ) from error
                self._read_more()
                continue
            self.position = end
            return value

    def finish(self) -> None:
        """Reject non-layout bytes after the top-level value."""

        self._skip_layout()
        if self.position < len(self.buffer):
            raise BuildError("source JSON has trailing content")
        if not self.eof:
            self._read_more()
            self._skip_layout()
        if self.position < len(self.buffer):
            raise BuildError("source JSON has trailing content")


def iter_source_trajectories(
    source: str | os.PathLike[str],
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """Stream one proof object at a time from the top-level theory map."""

    path = Path(source)
    try:
        with path.open("r", encoding="utf-8", newline="") as source_file:
            stream = _IncrementalJSON(source_file)
            stream.expect("{")
            if stream.consume("}"):
                stream.finish()
                return
            while True:
                theory = stream.value()
                if not isinstance(theory, str):
                    raise BuildError("Magnushammer theory key must be a string")
                stream.expect(":")
                stream.expect("[")
                proof_index = 0
                if not stream.consume("]"):
                    while True:
                        proof = stream.value()
                        if not isinstance(proof, dict):
                            raise BuildError(
                                f"Magnushammer proof {theory}/{proof_index} "
                                "is not an object"
                            )
                        yield theory, proof_index, proof
                        proof_index += 1
                        if stream.consume(","):
                            continue
                        stream.expect("]")
                        break
                if stream.consume(","):
                    continue
                stream.expect("}")
                stream.finish()
                return
    except BuildError:
        raise
    except (OSError, UnicodeError) as error:
        raise BuildError(f"failed to stream Magnushammer source: {path}") from error


def _output_paths(out: Path, name: str) -> dict[str, Path]:
    return {
        "raw": out / "raw" / f"{name}.jsonl",
        "train": out / "shards" / f"{name}.jsonl",
        "eval": out / "eval" / f"{name}.jsonl",
        "manifest": out / "heldout" / f"{name}.json",
    }


def _invalidate_outputs(paths: Mapping[str, Path]) -> None:
    """Quarantine active outputs so failed rebuilds cannot look current."""

    for path in paths.values():
        if not path.exists():
            continue
        stale = path.with_name(path.name + ".stale")
        suffix = 1
        while stale.exists():
            stale = path.with_name(path.name + f".stale.{suffix}")
            suffix += 1
        os.replace(path, stale)


def _stage_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE fact_variants (
            name TEXT NOT NULL,
            statement TEXT NOT NULL,
            PRIMARY KEY (name, statement)
        );
        CREATE TABLE stable_facts (
            name TEXT PRIMARY KEY,
            statement TEXT NOT NULL
        );
        CREATE TABLE ambiguous_facts (
            name TEXT PRIMARY KEY
        );
        CREATE TABLE rows (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_id TEXT NOT NULL,
            theorem_statement TEXT NOT NULL,
            transition_index INTEGER NOT NULL,
            text_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            record_json TEXT NOT NULL,
            UNIQUE (text_hash, text)
        );
        CREATE TABLE row_facts (
            row_sequence INTEGER NOT NULL,
            name TEXT NOT NULL,
            PRIMARY KEY (row_sequence, name)
        );
        CREATE INDEX row_facts_name ON row_facts(name);
        CREATE INDEX rows_trajectory ON rows(trajectory_id);
        """
    )
    return connection


def _trajectory_id(theory: str, proof_index: int) -> str:
    payload = "\0".join(
        (
            SCHEMA_VERSION,
            SOURCE_DATASET,
            SOURCE_REVISION,
            theory,
            str(proof_index),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _record_id(trajectory_id: str, transition_index: int, text: str) -> str:
    payload = "\0".join(
        (
            SCHEMA_VERSION,
            trajectory_id,
            str(transition_index),
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_block(parsed: ParsedPremises, facts: Mapping[str, str]) -> str:
    lines = [HDR]
    for alias in sorted(parsed.global_aliases):
        qualified = parsed.global_aliases[alias]
        lines.append(f"{alias} [{qualified}] : {facts[qualified]}")
    if parsed.local_assumptions:
        lines.append(LOCAL_HDR)
        for alias in sorted(parsed.local_assumptions):
            lines.append(
                f"{alias} [{parsed.local_names[alias]}] : "
                f"{parsed.local_assumptions[alias]}"
            )
    return "\n".join(lines)


def _initialize_stats() -> Counter[str]:
    keys = (
        "trajectories_seen",
        "transitions_seen",
        "dropped_malformed_trajectory",
        "dropped_aborted_trajectories",
        "dropped_final_transition",
        "dropped_empty_field",
        "dropped_unchanged_state",
        "dropped_malformed_premises",
        "dropped_no_global_fact",
        "dropped_ambiguous_fact",
        "dropped_paste",
        "dropped_overlength",
        "dropped_duplicate",
        "accepted_rows",
        "ambiguous_global_names",
    )
    return Counter({key: 0 for key in keys})


def _first_pass(
    connection: sqlite3.Connection,
    trajectory_iter_factory: Callable[
        [], Iterator[tuple[str, int, dict[str, Any]]]
    ],
    stats: Counter[str],
) -> None:
    for item_index, (_, _, proof) in enumerate(trajectory_iter_factory(), start=1):
        stats["trajectories_seen"] += 1
        if not isinstance(proof, Mapping):
            stats["dropped_malformed_trajectory"] += 1
            continue
        transitions = proof.get("transitions")
        if not isinstance(transitions, list):
            stats["dropped_malformed_trajectory"] += 1
            continue
        stats["transitions_seen"] += len(transitions)
        if transitions:
            stats["dropped_final_transition"] += 1
        if _is_aborted(transitions):
            stats["dropped_aborted_trajectories"] += 1

        for transition in transitions:
            if not isinstance(transition, Mapping):
                continue
            parsed = _parse_premises(transition.get("premises"))
            connection.executemany(
                "INSERT OR IGNORE INTO fact_variants(name, statement) VALUES (?, ?)",
                parsed.global_observations,
            )
        if item_index % 10_000 == 0:
            connection.commit()
    connection.commit()
    connection.executescript(
        """
        INSERT INTO stable_facts(name, statement)
        SELECT name, MIN(statement)
        FROM fact_variants
        GROUP BY name
        HAVING COUNT(*) = 1;

        INSERT INTO ambiguous_facts(name)
        SELECT name
        FROM fact_variants
        GROUP BY name
        HAVING COUNT(*) > 1;
        """
    )
    connection.commit()
    stats["ambiguous_global_names"] = connection.execute(
        "SELECT COUNT(*) FROM ambiguous_facts"
    ).fetchone()[0]


def _stable_statement(
    connection: sqlite3.Connection,
    qualified_name: str,
) -> str | None:
    row = connection.execute(
        "SELECT statement FROM stable_facts WHERE name = ?",
        (qualified_name,),
    ).fetchone()
    return None if row is None else str(row[0])


def _second_pass(
    connection: sqlite3.Connection,
    trajectory_iter_factory: Callable[
        [], Iterator[tuple[str, int, dict[str, Any]]]
    ],
    tokenizer: Any,
    stats: Counter[str],
    source_metadata: Mapping[str, Any],
) -> None:
    for item_index, (raw_theory, proof_index, proof) in enumerate(
        trajectory_iter_factory(),
        start=1,
    ):
        if not isinstance(proof, Mapping):
            continue
        transitions = proof.get("transitions")
        if not isinstance(transitions, list) or _is_aborted(transitions):
            continue
        theorem_statement = _required_text(proof.get("statement"))
        theory = normalize_layout(str(raw_theory))
        trajectory_id = _trajectory_id(theory, proof_index)
        theorem_identity = f"{theory}/{proof_index}"

        for transition_index in range(max(0, len(transitions) - 1)):
            current = transitions[transition_index]
            following = transitions[transition_index + 1]
            if not isinstance(current, Mapping) or not isinstance(following, Mapping):
                stats["dropped_empty_field"] += 1
                continue
            state_before = _required_text(current.get("state"))
            tactic = _required_text(current.get("step"))
            state_after = _required_text(following.get("state"))
            if (
                not theorem_statement
                or not state_before
                or not tactic
                or not state_after
            ):
                stats["dropped_empty_field"] += 1
                continue
            if state_before == state_after:
                stats["dropped_unchanged_state"] += 1
                continue

            parsed = _parse_premises(current.get("premises"))
            if parsed.malformed:
                stats["dropped_malformed_premises"] += 1
                continue
            if not parsed.global_aliases:
                stats["dropped_no_global_fact"] += 1
                continue

            facts: dict[str, str] = {}
            ambiguous = False
            for alias in sorted(parsed.global_aliases):
                qualified = parsed.global_aliases[alias]
                stable = _stable_statement(connection, qualified)
                if (
                    stable is None
                    or parsed.global_alias_statements[alias] != stable
                ):
                    ambiguous = True
                    break
                facts[qualified] = stable
            if ambiguous:
                stats["dropped_ambiguous_fact"] += 1
                continue
            facts = {name: facts[name] for name in sorted(facts)}

            if paste_share(state_after, state_before) >= MAX_PASTE_SHARE:
                stats["dropped_paste"] += 1
                continue

            premise_aliases = {
                alias: parsed.global_aliases[alias]
                for alias in sorted(parsed.global_aliases)
            }
            local_assumptions = {
                alias: parsed.local_assumptions[alias]
                for alias in sorted(parsed.local_assumptions)
            }
            local_names = {
                alias: parsed.local_names[alias] for alias in sorted(parsed.local_names)
            }
            cited = sorted(facts)
            block = _render_block(parsed, facts)
            goal = (
                f"THEOREM\n{theorem_statement}\n"
                f"STATE_BEFORE\n{state_before}"
            )
            target = f"TACTIC\n{tactic}\nSTATE_AFTER\n{state_after}"
            text = f"{block}\n{SEP}\nGOAL\n{goal}\n{target}"
            if _tokens_with_eos(tokenizer, text) > MAX_TOKENS_WITH_EOS:
                stats["dropped_overlength"] += 1
                continue

            record = {
                "schema_version": SCHEMA_VERSION,
                "id": _record_id(trajectory_id, transition_index, text),
                "trajectory_id": trajectory_id,
                "transition_index": transition_index,
                "theorem": theorem_identity,
                "theory": theory,
                "theorem_statement": theorem_statement,
                "facts": facts,
                "cited": cited,
                "premise_aliases": premise_aliases,
                "local_assumptions": local_assumptions,
                "local_names": local_names,
                "state_before": state_before,
                "tactic": tactic,
                "state_after": state_after,
                "goal": goal,
                "target": target,
                "text": text,
                "mask_start": 0,
                "mask_end": len(block),
                "source_metadata": dict(source_metadata),
            }
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            record_json = json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO rows(
                    trajectory_id,
                    theorem_statement,
                    transition_index,
                    text_hash,
                    text,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory_id,
                    theorem_statement,
                    transition_index,
                    text_hash,
                    text,
                    record_json,
                ),
            )
            if cursor.rowcount == 0:
                stats["dropped_duplicate"] += 1
                continue
            row_sequence = int(cursor.lastrowid)
            connection.executemany(
                "INSERT INTO row_facts(row_sequence, name) VALUES (?, ?)",
                ((row_sequence, name) for name in cited),
            )
            stats["accepted_rows"] += 1
        if item_index % 5_000 == 0:
            connection.commit()
    connection.commit()


def _eligible_facts(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(name): str(statement)
        for name, statement in connection.execute(
            """
            SELECT DISTINCT row_facts.name, stable_facts.statement
            FROM row_facts
            JOIN stable_facts ON stable_facts.name = row_facts.name
            ORDER BY row_facts.name
            """
        )
    }


def _select_heldout(
    connection: sqlite3.Connection,
    eligible_facts: Mapping[str, str],
    heldout: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    citation_counts = {
        str(name): int(count)
        for name, count in connection.execute(
            """
            SELECT name, COUNT(*)
            FROM row_facts
            GROUP BY name
            ORDER BY name
            """
        )
    }
    names_by_statement: defaultdict[str, set[str]] = defaultdict(set)
    for name, statement in eligible_facts.items():
        names_by_statement[statement].add(name)
    tail = sorted(
        name
        for name, count in citation_counts.items()
        if count in TAIL_CITATION_COUNTS
        and len(names_by_statement[eligible_facts[name]]) == 1
    )
    if heldout <= 0:
        return [], tail
    if heldout > len(tail):
        raise BuildError(
            f"requested {heldout} heldout facts but only {len(tail)} safe tail "
            "facts are available"
        )
    held = sorted(random.Random(seed).sample(tail, heldout))
    return held, tail


def _statement_anchor(statement: str) -> str:
    tokens = TOK.findall(normalize_layout(statement))
    return tokens[0] if tokens else ""


def _heldout_exposure_types(
    record: Mapping[str, Any],
    *,
    held_names: set[str],
    held_names_by_statement: Mapping[str, set[str]],
    held_by_anchor: Mapping[str, list[str]],
) -> set[str]:
    exposure_types: set[str] = set()
    held_statements = set(held_names_by_statement)

    local_assumptions = record.get("local_assumptions", {})
    if isinstance(local_assumptions, Mapping) and any(
        normalize_layout(statement) in held_statements
        for statement in local_assumptions.values()
        if isinstance(statement, str)
    ):
        exposure_types.add("local_statement")

    local_names = record.get("local_names", {})
    if isinstance(local_names, Mapping) and any(
        contains_qualified_name(local_name, held_name)
        for local_name in local_names.values()
        if isinstance(local_name, str)
        for held_name in held_names
    ):
        exposure_types.add("local_statement")

    theorem_statement = record.get("theorem_statement", "")
    if (
        isinstance(theorem_statement, str)
        and normalize_declaration_statement(theorem_statement) in held_statements
    ):
        exposure_types.add("own_proof_declaration")

    def embedded_statement(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        normalized = normalize_layout(value)
        tokens = set(TOK.findall(normalized))
        candidates = {
            statement
            for token in tokens
            for statement in held_by_anchor.get(token, ())
        }
        return any(
            contains_normalized_statement(normalized, statement)
            for statement in candidates
        )

    state_before_exposure = embedded_statement(record.get("state_before"))
    if embedded_statement(record.get("goal")) and not state_before_exposure:
        exposure_types.add("own_proof_declaration")
    if any(
        (
            state_before_exposure,
            embedded_statement(record.get("state_after")),
            embedded_statement(record.get("target")),
        )
    ):
        exposure_types.add("target_state")
    return exposure_types


def _partition_plan(
    connection: sqlite3.Connection,
    held: list[str],
) -> tuple[set[int], set[str], set[str], dict[str, int], dict[str, int]]:
    total_rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    total_trajectories = int(
        connection.execute(
            "SELECT COUNT(DISTINCT trajectory_id) FROM rows"
        ).fetchone()[0]
    )
    if not held:
        trajectory_counts = {
            "total": total_trajectories,
            "train": total_trajectories,
            "direct_eval": 0,
            "own_proof": 0,
            "local_statement_exposure": 0,
            "own_proof_declaration_exposure": 0,
            "target_state_exposure": 0,
            "statement_exposure_excluded": 0,
            "excluded_from_train": 0,
        }
        row_counts = {
            "eligible": total_rows,
            "train": total_rows,
            "eval": 0,
            "dropped_siblings": 0,
            "dropped_own_proof": 0,
            "dropped_local_statement_exposure": 0,
            "dropped_own_proof_declaration_exposure": 0,
            "dropped_target_state_exposure": 0,
            "dropped_statement_exposure": 0,
        }
        return set(), set(), set(), trajectory_counts, row_counts

    placeholders = ",".join("?" for _ in held)
    direct_rows = {
        int(row_sequence)
        for (row_sequence,) in connection.execute(
            f"""
            SELECT DISTINCT row_sequence
            FROM row_facts
            WHERE name IN ({placeholders})
            """,
            held,
        )
    }
    direct_trajectories = {
        str(trajectory_id)
        for (trajectory_id,) in connection.execute(
            f"""
            SELECT DISTINCT rows.trajectory_id
            FROM rows
            JOIN row_facts ON row_facts.row_sequence = rows.sequence
            WHERE row_facts.name IN ({placeholders})
            """,
            held,
        )
    }
    held_statements = {
        name: str(row[0])
        for name in held
        if (
            row := connection.execute(
                "SELECT statement FROM stable_facts WHERE name = ?",
                (name,),
            ).fetchone()
        )
        is not None
    }
    held_names_by_statement: defaultdict[str, set[str]] = defaultdict(set)
    held_by_anchor: defaultdict[str, list[str]] = defaultdict(list)
    for name, statement in held_statements.items():
        normalized = normalize_layout(statement)
        held_names_by_statement[normalized].add(name)
        anchor = _statement_anchor(normalized)
        if anchor:
            held_by_anchor[anchor].append(normalized)

    trajectory_exposures: defaultdict[str, set[str]] = defaultdict(set)
    for trajectory_id, record_json in connection.execute(
        "SELECT trajectory_id, record_json FROM rows ORDER BY sequence"
    ):
        record = json.loads(str(record_json))
        trajectory_exposures[str(trajectory_id)].update(
            _heldout_exposure_types(
                record,
                held_names=set(held),
                held_names_by_statement=held_names_by_statement,
                held_by_anchor=held_by_anchor,
            )
        )

    typed_trajectories = {
        exposure_type: {
            trajectory_id
            for trajectory_id, exposures in trajectory_exposures.items()
            if exposure_type in exposures
            and trajectory_id not in direct_trajectories
        }
        for exposure_type in (
            "local_statement",
            "own_proof_declaration",
            "target_state",
        )
    }
    exposure_trajectories = set().union(*typed_trajectories.values())
    drop_type_by_trajectory: dict[str, str] = {}
    for trajectory_id in exposure_trajectories:
        for exposure_type in (
            "own_proof_declaration",
            "local_statement",
            "target_state",
        ):
            if trajectory_id in typed_trajectories[exposure_type]:
                drop_type_by_trajectory[trajectory_id] = exposure_type
                break

    dropped_siblings = 0
    typed_dropped_rows: Counter[str] = Counter()
    train_rows = 0
    for sequence, trajectory_id in connection.execute(
        "SELECT sequence, trajectory_id FROM rows ORDER BY sequence"
    ):
        if int(sequence) in direct_rows:
            continue
        if trajectory_id in direct_trajectories:
            dropped_siblings += 1
        elif trajectory_id in exposure_trajectories:
            typed_dropped_rows[drop_type_by_trajectory[str(trajectory_id)]] += 1
        else:
            train_rows += 1
    dropped_statement_exposure = sum(typed_dropped_rows.values())
    excluded = direct_trajectories | exposure_trajectories
    trajectory_counts = {
        "total": total_trajectories,
        "train": total_trajectories - len(excluded),
        "direct_eval": len(direct_trajectories),
        "own_proof": len(typed_trajectories["own_proof_declaration"]),
        "local_statement_exposure": len(typed_trajectories["local_statement"]),
        "own_proof_declaration_exposure": len(
            typed_trajectories["own_proof_declaration"]
        ),
        "target_state_exposure": len(typed_trajectories["target_state"]),
        "statement_exposure_excluded": len(exposure_trajectories),
        "excluded_from_train": len(excluded),
    }
    row_counts = {
        "eligible": total_rows,
        "train": train_rows,
        "eval": len(direct_rows),
        "dropped_siblings": dropped_siblings,
        "dropped_own_proof": typed_dropped_rows["own_proof_declaration"],
        "dropped_local_statement_exposure": typed_dropped_rows[
            "local_statement"
        ],
        "dropped_own_proof_declaration_exposure": typed_dropped_rows[
            "own_proof_declaration"
        ],
        "dropped_target_state_exposure": typed_dropped_rows["target_state"],
        "dropped_statement_exposure": dropped_statement_exposure,
    }
    accounted_rows = (
        row_counts["train"]
        + row_counts["eval"]
        + row_counts["dropped_siblings"]
        + row_counts["dropped_statement_exposure"]
    )
    if accounted_rows != total_rows:
        raise BuildError(
            f"partition accounting lost {total_rows - accounted_rows} eligible rows"
        )
    return (
        direct_rows,
        direct_trajectories,
        exposure_trajectories,
        trajectory_counts,
        row_counts,
    )


def _write_rows(
    connection: sqlite3.Connection,
    path: Path,
    *,
    category: str,
    direct_rows: set[int],
    direct_trajectories: set[str],
    exposure_trajectories: set[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for sequence, trajectory_id, record_json in connection.execute(
            """
            SELECT sequence, trajectory_id, record_json
            FROM rows
            ORDER BY sequence
            """
        ):
            sequence = int(sequence)
            if category == "raw":
                include = True
            elif category == "eval":
                include = sequence in direct_rows
            else:
                include = (
                    sequence not in direct_rows
                    and trajectory_id not in direct_trajectories
                    and trajectory_id not in exposure_trajectories
                )
            if include:
                output.write(str(record_json) + "\n")


def _write_outputs_atomically(
    connection: sqlite3.Connection,
    paths: Mapping[str, Path],
    *,
    heldout: int,
    direct_rows: set[int],
    direct_trajectories: set[str],
    exposure_trajectories: set[str],
    manifest: Mapping[str, Any],
) -> None:
    categories = ("raw",) if heldout == 0 else ("train", "eval")
    final_paths = [paths[category] for category in categories] + [paths["manifest"]]
    temp_paths = {
        final: final.with_name(final.name + f".tmp.{os.getpid()}")
        for final in final_paths
    }
    try:
        for category in categories:
            _write_rows(
                connection,
                temp_paths[paths[category]],
                category=category,
                direct_rows=direct_rows,
                direct_trajectories=direct_trajectories,
                exposure_trajectories=exposure_trajectories,
            )
        with temp_paths[paths["manifest"]].open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            manifest_file.write("\n")
        for final in final_paths:
            os.replace(temp_paths[final], final)
    except Exception:
        _invalidate_outputs(paths)
        raise
    finally:
        for temp in temp_paths.values():
            if temp.exists():
                temp.unlink()


def build_corpus(
    *,
    source: str | os.PathLike[str],
    out: str | os.PathLike[str],
    name: str,
    heldout: int,
    seed: int,
    tokenizer: Any,
    source_gate: Callable[[str | os.PathLike[str]], Mapping[str, Any]] = (
        verify_source_file
    ),
    trajectory_iter_factory: (
        Callable[[], Iterator[tuple[str, int, dict[str, Any]]]] | None
    ) = None,
) -> dict[str, int]:
    """Build verified outputs, with injectable source seams reserved for tests."""

    if heldout < 0:
        raise BuildError("--heldout must be non-negative")
    output_root = Path(out)
    paths = _output_paths(output_root, name)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _invalidate_outputs(paths)

    stage_path = output_root / f".{name}.stage.{os.getpid()}.sqlite"
    if stage_path.exists():
        stage_path.unlink()
    connection: sqlite3.Connection | None = None
    try:
        source_metadata = dict(source_gate(source))
        required_source_fields = {
            "dataset",
            "revision",
            "file",
            "size_bytes",
            "sha256",
            "license",
        }
        if required_source_fields - set(source_metadata):
            raise BuildError("source gate returned incomplete source metadata")
        tokenizer_metadata = _tokenizer_metadata(tokenizer)
        row_source_metadata = _build_source_metadata(source_metadata)
        factory = trajectory_iter_factory or (
            lambda: iter_source_trajectories(source)
        )
        stats = _initialize_stats()
        connection = _stage_database(stage_path)
        _first_pass(connection, factory, stats)
        _second_pass(
            connection,
            factory,
            tokenizer,
            stats,
            row_source_metadata,
        )
        if stats["accepted_rows"] == 0:
            raise BuildError("no accepted output rows; rebuild refused")

        eligible_facts = _eligible_facts(connection)
        held, tail = _select_heldout(
            connection,
            eligible_facts,
            heldout,
            seed,
        )
        (
            direct_rows,
            direct_trajectories,
            exposure_trajectories,
            trajectory_counts,
            row_counts,
        ) = _partition_plan(connection, held)
        held_statement_hashes = sorted(
            canonical_statement_hash(eligible_facts[name]) for name in held
        )
        if heldout == 0:
            mode = "raw_staging"
            policy = (
                "heldout=0 is debug/staging-only and not scientifically complete: "
                "all eligible exact-deduplicated rows are emitted to raw without "
                "trajectory-isolated evaluation"
            )
        else:
            mode = "family_local_heldout"
            policy = (
                "stable statement-unique facts cited 1-2 times are sampled; only "
                "direct citing rows enter eval, while every sibling trajectory "
                "row and every local-statement, declaration-normalized own-proof, "
                "or target/state statement-exposure trajectory is excluded from "
                "train as a typed drop"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "family": "isabelle",
            "corpus": name,
            "mode": mode,
            "facts": held,
            "statements": {name: eligible_facts[name] for name in held},
            "statement_hashes": held_statement_hashes,
            "canonicalization": {
                "family": "isabelle",
                "scheme": CANONICALIZATION_SCHEME,
                "version": CANONICALIZATION_VERSION,
            },
            "source": source_metadata,
            "tokenizer": tokenizer_metadata,
            "seed": seed,
            "requested_heldout": heldout,
            "tail_citation_counts": list(TAIL_CITATION_COUNTS),
            "tail_size": len(tail),
            "eligible_fact_names": sorted(eligible_facts),
            "trajectory_counts": trajectory_counts,
            "row_counts": row_counts,
            "filter_counts": dict(stats),
            "policy": policy,
        }
        _write_outputs_atomically(
            connection,
            paths,
            heldout=heldout,
            direct_rows=direct_rows,
            direct_trajectories=direct_trajectories,
            exposure_trajectories=exposure_trajectories,
            manifest=manifest,
        )
        stats.update(
            {
                "train_rows": row_counts["train"],
                "eval_rows": row_counts["eval"],
                "dropped_sibling_rows": row_counts["dropped_siblings"],
                "dropped_own_proof_rows": row_counts["dropped_own_proof"],
                "dropped_local_statement_rows": row_counts[
                    "dropped_local_statement_exposure"
                ],
                "dropped_target_state_rows": row_counts[
                    "dropped_target_state_exposure"
                ],
                "dropped_statement_exposure_rows": row_counts[
                    "dropped_statement_exposure"
                ],
                "heldout_facts": len(held),
            }
        )
        return dict(stats)
    except BuildError:
        _invalidate_outputs(paths)
        raise
    except Exception as error:
        _invalidate_outputs(paths)
        raise BuildError(f"Isabelle rebuild failed: {error}") from error
    finally:
        if connection is not None:
            connection.close()
        for suffix in ("", "-journal", "-wal", "-shm"):
            candidate = Path(str(stage_path) + suffix)
            if candidate.exists():
                candidate.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default="corpus")
    parser.add_argument("--name", default="isabelle")
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="local vendored Qwen2.5 directory or tokenizer.json",
    )
    args = parser.parse_args(argv)
    try:
        tokenizer = load_vendored_tokenizer(args.tokenizer_path)
        stats = build_corpus(
            source=args.src,
            out=args.out,
            name=args.name,
            heldout=args.heldout,
            seed=args.seed,
            tokenizer=tokenizer,
        )
    except BuildError as error:
        print(f"Isabelle build refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
