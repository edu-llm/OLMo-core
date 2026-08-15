"""Machine C — the ATP shards (prf2 and ENIGMA).

Both are E-prover output over MPTP/Mizar problems, so one parser serves both.
A TPTP proof file gives us exactly the shape we want for free:

  fof(NAME, axiom, FORMULA, p(NAME))     <- a named MML premise; the fact block
  fof(NAME, conjecture, FORMULA)         <- the goal
  fof(c_0_N, plain, FORMULA, inference(rule,...,[parents]))   <- the target

The target is content, not names: the model must produce each derived formula
along with the rule and parents that justify it, ending in $false.

ENIGMA differs in two ways handled below: most runs FAILED (resource-out), so
only files reporting Theorem/Unsatisfiable are read, and the proof is fenced
inside `# SZS output start`/`end` markers amid the run statistics.
"""

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise

from split_heldout import (
    canonical_statement,
    canonicalization_metadata,
    normalize_theorem_identity,
    statement_hash,
)

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local ATP inputs:"
SEP = "---"
# Machine-generated Mizar bookkeeping: typing conditions, clusters, redefinitions,
# Fraenkel expansions, arithmetic evaluation. Real premises but not mathematical
# content, and they crowd the block — 42% of prf2's citations here.
BOOKKEEPING = re.compile(
    r"^(?:dt_|(?:cc|fc|rc)\d*_|redefinition_|fraenkel_|rq|spc)"
)
ENTRY_START = re.compile(r"(?m)^[ \t]*(fof|cnf)\s*\(")
OK = ("SZS status Theorem", "SZS status Unsatisfiable")
SAFE_TPTP_ATOM = re.compile(r"^[a-z][A-Za-z0-9_]*$")
BUILD_SOURCE_SCHEMA = "atp-build-source-v2"
ROW_SCHEMA = "atp-v2"
CLOSURE_DISPOSITIONS = {
    "duplicate_step",
    "late_or_cyclic_parent",
    "mixed_parent_closure",
    "unresolved_parent",
}
SOURCE_DISPOSITIONS = {
    "accepted",
    "malformed_source",
    "non_refutation",
    "redundant_reproof",
    "too_thin",
    "unclassified_closure",
    "unreadable",
    "unsolved_or_unfenced",
    *CLOSURE_DISPOSITIONS,
}
ENIGMA_LOW_TIER_SCHEMA = "enigma-alternative-proof-low-tier-v1"
ENIGMA_LOW_TIER_POLICY = {
    "schema_version": ENIGMA_LOW_TIER_SCHEMA,
    "expected_base_rows": 27_079,
    "expected_redundant_dispositions": 29_437,
    "expected_selected_rows": 2_087,
    "expected_text_plus_eos_tokens": 9_655_618,
    "expected_packed_16384_tokens": 9_666_560,
    "max_total_variants": 2,
    "max_text_plus_eos_tokens": 8_192,
    "max_alpha_formula_jaccard": 0.67,
    "max_backward_dag_edge_jaccard": 0.70,
    "min_new_alpha_formulas": 3,
    "min_new_alpha_formula_fraction": 0.20,
    "run_priority": ["mzr01", "mzr03", "mzr02", "mzr08"],
    "expected_run_distribution": {
        "mzr03": 256,
        "mzr02": 1_115,
        "mzr08": 716,
    },
    "legacy_redundancy_jaccard": 0.5,
    "legacy_min_steps": 4,
    "seed": 20260801,
}
ENIGMA_LOW_TIER_SOURCE_CONTRACT = {
    "accepted_base": {
        "bytes": 1_533_231_777,
        "rows": 27_079,
        "sha256": "856aa4fd240da58bc6699623e5b54e8e363d10c68788034fe78b547691bd6183",
    },
    "source_files": 231_520,
    "source_order": ["mzr01", "mzr03", "mzr02", "mzr08"],
    "source_dispositions": {
        "accepted": 27_079,
        "redundant_reproof": 29_437,
        "too_thin": 964,
        "unresolved_parent": 4,
        "unsolved_or_unfenced": 174_036,
    },
    "source_manifest_root_sha256": (
        "c33d5b87696e56276f8c2eb81fd1acb274ab766308e60a96e6ee999d6d76fd2e"
    ),
    "quality_filter_root_sha256": (
        "cdd95f8ab0314ec2b11b44273fd412c7445e2ab61e73d43a33e84c158b7c4ce8"
    ),
    "schema_generation_root_sha256": (
        "bd0caede34f0dea92401bc9a306ae1ef82f81a26ed3bedeecd6e6a4a653a1a60"
    ),
    "source_roots": {
        "source_1": {
            "files": 57_880,
            "tree_sha256": (
                "73b13a8461e2f98ae00513f1f764654998db2fab7de698a7809dc87f5ddbe3fe"
            ),
        },
        "source_2": {
            "files": 57_880,
            "tree_sha256": (
                "1f216d4b1f6c59242845b69695c534554df905e468d4f3e4756430736dd6d73d"
            ),
        },
        "source_3": {
            "files": 57_880,
            "tree_sha256": (
                "534158d350bb14ef898d85554e676722bb00b7adc9acf2e2c061b7de10f965d5"
            ),
        },
        "source_4": {
            "files": 57_880,
            "tree_sha256": (
                "314bc45d17237e9ce55bf47613e60ecebccc9b7649f19667ccf3d0ff69961e7c"
            ),
        },
    },
    "tokenizer": {
        "tokenizer_json_sha256": (
            "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
        ),
        "tokenizer_config_sha256": (
            "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
        ),
        "tokenizers_version": "0.22.2",
        "eos_token_id": 151_643,
        "separator_ids": [10_952, 15_513, 969],
    },
}
ENIGMA_LOW_TIER_RUN_RANK = {
    run: rank for rank, run in enumerate(ENIGMA_LOW_TIER_POLICY["run_priority"])
}
ENIGMA_LOW_TIER_ADMIN_RULES = frozenset(
    {
        "variable_rename",
        "fof_nnf",
        "fof_simplification",
        "split_conjunct",
        "evalgc",
        "shift_quantors",
        "skolemize",
        "distribute",
        "introduced",
        "theory",
        "creator",
    }
)
ENIGMA_LOW_TIER_ALPHA_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])(X\d+|esk\d+_\d+)(?![A-Za-z0-9_])"
)
ENIGMA_LOW_TIER_CLASS_RANK = {
    "raw_byte_exact": 0,
    "full_text_exact": 1,
    "exact_structured": 2,
    "generated_names_or_source_only": 3,
    "topological_order_only": 4,
    "parent_order_only": 5,
    "parent_multiplicity_only": 6,
    "alpha_formula_surface_only": 7,
    "same_formula_set_different_path": 8,
    "alpha_formula_set_different_path": 9,
    "premise_and_path_variant": 10,
    "formula_rule_path_variant": 11,
}
ENIGMA_LOW_TIER_NONMATERIAL = frozenset(
    tuple(ENIGMA_LOW_TIER_CLASS_RANK)[:8]
)


@dataclass(frozen=True)
class AlternativeProofFeatures:
    """Selection-relevant features for one retained or redundant ENIGMA trace."""

    base: str
    run: str
    raw_sha256: str
    text_sha256: str
    exact_signature_sha256: str
    text_plus_eos_tokens: int
    existing_variants: int
    dead_steps: int
    material: bool
    alpha_formulas: frozenset[str]
    backward_edges: frozenset[str]
    rule_bigrams: frozenset[tuple[str, str]]
    premises: frozenset[tuple[str, str]]
    core_rules: frozenset[str]
    paste_steps: int = 0
    proof_steps: int = 0
    diversity_class: str = ""
    record: dict | None = field(default=None, compare=False, repr=False)


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_enigma_low_tier_contract(contract: dict) -> None:
    """Reject any drift from the source, base, audit, or tokenizer contract."""
    if contract != ENIGMA_LOW_TIER_SOURCE_CONTRACT:
        raise ValueError(
            "source/audit contract mismatch: "
            f"expected {_canonical_sha256(ENIGMA_LOW_TIER_SOURCE_CONTRACT)}, "
            f"got {_canonical_sha256(contract)}"
        )


def _jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


def conservative_alternative_metrics(
    candidate: AlternativeProofFeatures,
    accepted: AlternativeProofFeatures,
) -> dict:
    """Compute the audited diversity dimensions against one retained trace."""
    return {
        "alpha_formula_jaccard": _jaccard(
            candidate.alpha_formulas,
            accepted.alpha_formulas,
        ),
        "backward_dag_edge_jaccard": _jaccard(
            candidate.backward_edges,
            accepted.backward_edges,
        ),
        "rule_bigram_jaccard": _jaccard(
            candidate.rule_bigrams,
            accepted.rule_bigrams,
        ),
        "premise_jaccard": _jaccard(
            candidate.premises,
            accepted.premises,
        ),
        "new_alpha_formulas": len(
            candidate.alpha_formulas - accepted.alpha_formulas
        ),
        "core_rule_set_diff": candidate.core_rules != accepted.core_rules,
    }


def conservative_alternative_is_eligible(
    candidate: AlternativeProofFeatures,
    accepted: AlternativeProofFeatures,
) -> bool:
    """Apply only the user-approved conservative ENIGMA recovery policy."""
    metrics = conservative_alternative_metrics(candidate, accepted)
    minimum_new = max(
        ENIGMA_LOW_TIER_POLICY["min_new_alpha_formulas"],
        math.ceil(
            ENIGMA_LOW_TIER_POLICY["min_new_alpha_formula_fraction"]
            * len(candidate.alpha_formulas)
        ),
    )
    return (
        candidate.material
        and candidate.existing_variants == 1
        and candidate.dead_steps == 0
        and candidate.text_plus_eos_tokens
        <= ENIGMA_LOW_TIER_POLICY["max_text_plus_eos_tokens"]
        and metrics["alpha_formula_jaccard"]
        <= ENIGMA_LOW_TIER_POLICY["max_alpha_formula_jaccard"]
        and metrics["backward_dag_edge_jaccard"]
        <= ENIGMA_LOW_TIER_POLICY["max_backward_dag_edge_jaccard"]
        and metrics["new_alpha_formulas"] >= minimum_new
        and (
            metrics["premise_jaccard"] < 1.0
            or metrics["core_rule_set_diff"]
        )
    )


def conservative_alternative_rank(
    candidate: AlternativeProofFeatures,
    accepted: AlternativeProofFeatures,
) -> tuple:
    """Return the exact audited deterministic ranking tuple."""
    metrics = conservative_alternative_metrics(candidate, accepted)
    try:
        run_rank = ENIGMA_LOW_TIER_RUN_RANK[candidate.run]
    except KeyError as error:
        raise ValueError(f"unknown ENIGMA run {candidate.run!r}") from error
    return (
        1.0 - metrics["alpha_formula_jaccard"],
        1.0 - metrics["backward_dag_edge_jaccard"],
        1.0 - metrics["rule_bigram_jaccard"],
        1.0 - metrics["premise_jaccard"],
        metrics["new_alpha_formulas"],
        -candidate.text_plus_eos_tokens,
        -run_rank,
        candidate.raw_sha256,
    )


def select_conservative_alternatives(
    candidates: list[AlternativeProofFeatures],
    accepted_by_base: dict[str, list[AlternativeProofFeatures]],
    *,
    existing_text_sha256s: set[str],
    existing_signature_sha256s: set[str],
) -> list[AlternativeProofFeatures]:
    """Select at most one low-tier alternative for each singleton theorem."""
    eligible_by_base: dict[str, list[AlternativeProofFeatures]] = defaultdict(list)
    for candidate in candidates:
        accepted = accepted_by_base.get(candidate.base, [])
        if len(accepted) != 1:
            continue
        if candidate.text_sha256 in existing_text_sha256s:
            continue
        if candidate.exact_signature_sha256 in existing_signature_sha256s:
            continue
        if conservative_alternative_is_eligible(candidate, accepted[0]):
            eligible_by_base[candidate.base].append(candidate)

    ranked = []
    for base in sorted(eligible_by_base):
        accepted = accepted_by_base[base][0]
        winner = max(
            eligible_by_base[base],
            key=lambda candidate: conservative_alternative_rank(
                candidate,
                accepted,
            ),
        )
        ranked.append(
            (
                conservative_alternative_rank(winner, accepted),
                winner,
            )
        )

    selected = []
    seen_text = set(existing_text_sha256s)
    seen_signature = set(existing_signature_sha256s)
    for _, candidate in sorted(
        ranked,
        key=lambda item: (item[0], item[1].base),
        reverse=True,
    ):
        if candidate.text_sha256 in seen_text:
            continue
        if candidate.exact_signature_sha256 in seen_signature:
            continue
        seen_text.add(candidate.text_sha256)
        seen_signature.add(candidate.exact_signature_sha256)
        selected.append(candidate)

    return sorted(
        selected,
        key=lambda candidate: (
            candidate.base,
            ENIGMA_LOW_TIER_RUN_RANK[candidate.run],
            candidate.raw_sha256,
        ),
    )


def _build_source_metadata(sources, files, source_of, args):
    roots = {}
    for ordinal, source in enumerate(sources, 1):
        source_path = os.path.abspath(source)
        records = []
        for path in files:
            if source_of[path] != source:
                continue
            relative = (
                os.path.relpath(path, source_path)
                if os.path.isdir(source_path)
                else os.path.basename(path)
            )
            with open(path, "rb") as source_file:
                digest = hashlib.sha256(source_file.read()).hexdigest()
            records.append({"path": relative, "sha256": digest})
        roots[f"source_{ordinal}"] = {
            "files": len(records),
            "tree_sha256": _canonical_sha256(records),
        }
    source_manifest = {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "source_roots": roots,
    }
    quality_policy = {
        "bookkeeping_pattern": BOOKKEEPING.pattern,
        "dedup": bool(args.dedup),
        "fenced": bool(args.fenced),
        "jaccard": args.jaccard,
        "min_steps": args.min_steps,
        "requires_complete_dependency_closure": True,
        "requires_exact_source_inventory_accounting": True,
        "requires_typed_closure_rejections": True,
        "requires_final_refutation": True,
    }
    row_schema = {
        "schema_version": ROW_SCHEMA,
        "structured_proof_steps": True,
        "source_annotations": True,
    }
    return {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "source_manifest_root_sha256": _canonical_sha256(source_manifest),
        "source_roots": roots,
        "index_roots": {},
        "quality_filter_root_sha256": _canonical_sha256(quality_policy),
        "schema_generation_root_sha256": _canonical_sha256(row_schema),
    }


@dataclass
class ProofStep:
    """A derived TSTP formula with its complete source annotation."""

    name: str
    role: str
    formula: str
    rule: str
    parents: list[str]
    parent_sources: list[str]
    source: str


@dataclass
class ParsedProof:
    """The replay-relevant pieces of one ATP proof."""

    facts: dict[str, str]
    local_inputs: dict[str, str]
    goal: str | None
    goal_name: str | None
    steps: list[ProofStep]
    source_errors: list[str] = field(default_factory=list)


@dataclass
class _Record:
    name: str
    role: str
    formula: str
    source: str


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split a TPTP term without splitting nested or quoted content."""
    out = []
    start = 0
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for i, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack and stack[-1] == pairs[ch]:
                stack.pop()
        elif ch == delimiter and not stack:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:].strip())
    return out


def _normalize_layout(text: str) -> str:
    """Collapse unquoted layout whitespace while preserving quoted atoms."""
    out = []
    quote = None
    escaped = False
    pending_space = False
    for ch in text.strip():
        if quote is not None:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            if pending_space and out:
                out.append(" ")
            pending_space = False
            quote = ch
            out.append(ch)
        elif ch.isspace():
            pending_space = True
        else:
            if pending_space and out:
                out.append(" ")
            pending_space = False
            out.append(ch)
    return "".join(out)


def _canonical_formula(text: str) -> str:
    # prf2 writes `k1_xboole_0 ( X1 ) = > X2` and ENIGMA writes
    # `k1_xboole_0(X1)=>X2`; TPTP layout whitespace is insignificant.
    return canonical_statement(text, family="atp")


def _unquote_atom(atom: str) -> str:
    atom = atom.strip()
    if len(atom) >= 2 and atom[0] == atom[-1] and atom[0] in ("'", '"'):
        body = atom[1:-1]
        return re.sub(r"\\(.)", r"\1", body)
    return atom


def _split_top_level_checked(
    text: str, delimiter: str = ","
) -> list[str] | None:
    """Strictly split balanced TPTP layout at one top-level delimiter."""
    if len(delimiter) != 1:
        raise ValueError("delimiter must be one character")
    if not text.strip():
        return []
    out = []
    start = 0
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return None
            stack.pop()
        elif char == delimiter and not stack:
            out.append(text[start:index].strip())
            start = index + 1
    if quote is not None or escaped or stack:
        return None
    out.append(text[start:].strip())
    return out


def _outer_delimiter_spans(
    text: str, opening: str, closing: str
) -> bool:
    """Whether one balanced delimiter pair encloses all of ``text``."""
    text = text.strip()
    if not text.startswith(opening) or not text.endswith(closing):
        return False
    stack = []
    quote = None
    escaped = False
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
            if not stack and index != len(text) - 1:
                return False
    return not stack and quote is None and not escaped


def _source_atom(term: str) -> str | None:
    """Decode one source-name atom, rejecting trailing or malformed syntax."""
    term = term.strip()
    if not term:
        return None
    if term[0] in ("'", '"'):
        quote = term[0]
        escaped = False
        for index, char in enumerate(term[1:], 1):
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                if index != len(term) - 1:
                    return None
                return _unquote_atom(term)
        return None
    if re.fullmatch(
        r"(?:[A-Za-z$][A-Za-z0-9_$]*|c_0_-[0-9]+|[0-9]+)",
        term,
    ):
        return term
    return None


def _term_call(term: str) -> tuple[str, list[str]] | None:
    term = term.strip()
    match = re.match(r"([A-Za-z$][A-Za-z0-9_$]*)\s*\(", term)
    if match is None:
        return None
    open_at = match.end() - 1
    if not _outer_delimiter_spans(term[open_at:], "(", ")"):
        return None
    args = _split_top_level_checked(term[open_at + 1:-1])
    if args is None:
        return None
    return match.group(1), args


def _iter_entries(text: str):
    """Yield complete fof/cnf entries, including multiline entries."""
    pos = 0
    while True:
        match = ENTRY_START.search(text, pos)
        if match is None:
            return
        open_at = text.find("(", match.start(1), match.end())
        depth = 0
        quote = None
        escaped = False
        close_at = None
        for i in range(open_at, len(text)):
            ch = text[i]
            if quote is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    close_at = i
                    break
        if close_at is None:
            return
        yield text[match.start(1):close_at + 1]
        pos = close_at + 1


def _parse_record(entry: str) -> _Record | None:
    match = re.match(r"(?:fof|cnf)\s*\(", entry)
    if match is None or not entry.endswith(")"):
        return None
    fields = _split_top_level(entry[match.end():-1])
    if len(fields) < 3:
        return None
    return _Record(
        name=_unquote_atom(fields[0]),
        role=fields[1].strip(),
        formula=_canonical_formula(fields[2]),
        source=_normalize_layout(fields[3]) if len(fields) >= 4 else "",
    )


def _origin_name(source: str) -> str | None:
    call = _term_call(source)
    if call is None:
        return None
    kind, args = call
    if kind == "p" and len(args) == 1:
        return _source_atom(args[0])
    if kind == "file" and len(args) >= 2:
        origin = _source_atom(args[1])
        if origin is None or origin.lower() == "unknown":
            return None
        return origin
    return None


@dataclass(frozen=True)
class _SourceParents:
    parents: list[str]
    error: str | None = None


@dataclass(frozen=True)
class _DerivedSource:
    value: tuple[str, list[str], list[str]] | None
    error: str | None = None


def _source_parent_names(source: str) -> _SourceParents:
    """Recursively resolve only DAG leaves from one TSTP parent source."""
    source = _normalize_layout(source)
    while _outer_delimiter_spans(source, "(", ")"):
        source = source[1:-1].strip()

    details = _split_top_level_checked(source, delimiter=":")
    if details is None:
        return _SourceParents([], f"malformed parent source {source}")
    if len(details) > 2 or any(not part for part in details):
        return _SourceParents([], f"malformed parent details {source}")
    if len(details) == 2:
        if _split_top_level_checked(details[1], delimiter="\0") is None:
            return _SourceParents([], f"malformed parent details {source}")
        return _source_parent_names(details[0])

    if source.startswith("["):
        if not _outer_delimiter_spans(source, "[", "]"):
            return _SourceParents([], f"malformed parent list {source}")
        items = _split_top_level_checked(source[1:-1])
        if items is None:
            return _SourceParents([], f"malformed parent list {source}")
        parents = []
        for item in items:
            parsed = _source_parent_names(item)
            if parsed.error is not None:
                return parsed
            parents.extend(parsed.parents)
        return _SourceParents(parents)
    if source[:1] in "({":
        return _SourceParents([], f"malformed parent source {source}")

    call = _term_call(source)
    if call is not None:
        kind, args = call
        if kind == "inference":
            parsed = _parse_inference(args)
            if parsed.error is not None:
                return _SourceParents([], parsed.error)
            assert parsed.value is not None
            return _SourceParents(parsed.value[2])
        if kind in {"introduced", "theory", "creator"}:
            if not 1 <= len(args) <= 2:
                return _SourceParents([], f"malformed {kind} source {source}")
            return _SourceParents([])
        if kind == "file":
            if not 1 <= len(args) <= 2:
                return _SourceParents([], f"malformed file source {source}")
            if len(args) == 1:
                return _SourceParents([])
            node = _source_atom(args[1])
            if node is None:
                return _SourceParents([], f"malformed file node {args[1]}")
            return _SourceParents([] if node.lower() == "unknown" else [node])
        if kind == "p":
            if len(args) != 1:
                return _SourceParents([], f"malformed p source {source}")
            node = _source_atom(args[0])
            if node is None:
                return _SourceParents([], f"malformed p source {source}")
            return _SourceParents([node])
        return _SourceParents([], f"unsupported parent source term {source}")
    if re.match(r"[A-Za-z$][A-Za-z0-9_$]*\s*\(", source):
        return _SourceParents([], f"malformed parent source term {source}")

    atom = _source_atom(source)
    if atom is None:
        return _SourceParents([], f"malformed parent source {source}")
    return _SourceParents([atom])


def _parse_inference(args: list[str]) -> _DerivedSource:
    if len(args) != 3:
        return _DerivedSource(None, "inference source must have three arguments")
    rule = _source_atom(args[0])
    if rule is None:
        return _DerivedSource(None, f"malformed inference rule {args[0]}")
    parent_list = _normalize_layout(args[2])
    if not _outer_delimiter_spans(parent_list, "[", "]"):
        return _DerivedSource(None, f"malformed inference parent list {parent_list}")
    raw_parents = _split_top_level_checked(parent_list[1:-1])
    if raw_parents is None:
        return _DerivedSource(None, f"malformed inference parent list {parent_list}")
    parent_sources = [_normalize_layout(parent) for parent in raw_parents]
    parents = []
    for parent in parent_sources:
        parsed = _source_parent_names(parent)
        if parsed.error is not None:
            return _DerivedSource(None, parsed.error)
        parents.extend(parsed.parents)
    return _DerivedSource((rule, parent_sources, parents))


def _parse_derived_source(source: str) -> _DerivedSource:
    call = _term_call(source)
    if call is None:
        if re.match(r"\s*(?:inference|introduced)\s*\(", source):
            return _DerivedSource(None, f"malformed derived source {source}")
        return _DerivedSource(None)
    kind, args = call
    if kind == "inference":
        return _parse_inference(args)
    if kind == "introduced":
        if not 1 <= len(args) <= 2:
            return _DerivedSource(None, f"malformed introduced source {source}")
        return _DerivedSource(("introduced", [], []))
    if kind in {"file", "p", "theory", "creator"}:
        parsed = _source_parent_names(source)
        return _DerivedSource(None, parsed.error)
    return _DerivedSource(None, f"unsupported source annotation {source}")


def _derived_source(source: str) -> tuple[str, list[str], list[str]] | None:
    parsed = _parse_derived_source(source)
    return None if parsed.error is not None else parsed.value


def source_dependencies(source: str) -> tuple[str, list[str], list[str]] | None:
    """Parse a derived TSTP source into rule, raw parents, and parent names."""
    return _derived_source(source)


def parse(text: str, fenced: bool) -> ParsedProof | None:
    if fenced:
        if not any(o in text for o in OK):
            return None
        i = text.find("# SZS output start")
        j = text.find("# SZS output end")
        if i < 0 or j < 0:
            return None
        text = text[text.find("\n", i) + 1:j]
    facts: dict[str, str] = {}
    local_inputs: dict[str, str] = {}
    steps = []
    goal = None
    goal_name = None
    input_records: dict[str, tuple[str, str]] = {}
    source_errors = []

    for entry in _iter_entries(text):
        record = _parse_record(entry)
        if record is None or not record.formula:
            continue
        if record.role == "conjecture":
            goal = record.formula
            goal_name = record.name
            continue

        parsed_source = _parse_derived_source(record.source)
        if parsed_source.error is not None:
            source_errors.append(f"{record.name}: {parsed_source.error}")
            continue
        derived = parsed_source.value
        if derived is not None:
            rule, parent_sources, parents = derived
            steps.append(
                ProofStep(
                    name=record.name,
                    role=record.role,
                    formula=record.formula,
                    rule=rule,
                    parents=parents,
                    parent_sources=parent_sources,
                    source=record.source,
                )
            )
            continue

        origin = _origin_name(record.source)
        supplied_name = origin or record.name
        input_records[record.name] = (supplied_name, record.formula)
        input_records[supplied_name] = (supplied_name, record.formula)
        if (
            record.role == "axiom"
            and origin is not None
            and not BOOKKEEPING.match(origin)
        ):
            facts[origin] = record.formula
        elif record.role != "type":
            local_inputs[supplied_name] = record.formula

    # A source annotation can cite either the formula's declaration name or its
    # file/p(...) origin. Supply any actually-used alias explicitly rather than
    # leaving a visible target parent ungrounded.
    supplied = set(facts) | set(local_inputs)
    if goal_name:
        supplied.add(goal_name)
    step_names = {step.name for step in steps}
    for parent in (p for step in steps for p in step.parents):
        if parent in supplied or parent in step_names:
            continue
        if parent in input_records:
            _, formula = input_records[parent]
            local_inputs[parent] = formula
            supplied.add(parent)

    return ParsedProof(
        facts,
        local_inputs,
        goal,
        goal_name,
        steps,
        source_errors,
    )


def dependency_errors(proof: ParsedProof) -> list[str]:
    """Return every target dependency that is missing or not topological."""
    errors = list(proof.source_errors)
    supplied = set(proof.facts) | set(proof.local_inputs)
    if proof.goal_name:
        supplied.add(proof.goal_name)
    all_steps = {step.name for step in proof.steps}
    seen = set()
    for step in proof.steps:
        if step.name in seen:
            errors.append(f"{step.name}: duplicate target step")
        for parent in step.parents:
            if parent in supplied or parent in seen:
                continue
            if parent in all_steps:
                errors.append(f"{step.name}: parent {parent} is not earlier")
            else:
                errors.append(f"{step.name}: unresolved parent {parent}")
        seen.add(step.name)
    return errors


def render_tptp_atom(name: str) -> str:
    """Render a decoded TPTP name without introducing token ambiguity."""
    if SAFE_TPTP_ATOM.fullmatch(name):
        return name
    escaped = name.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _balanced_outer_parentheses(text: str) -> bool:
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    quote = None
    escaped = False
    for index, ch in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0 or (depth == 0 and index != len(text) - 1):
                return False
    return depth == 0 and quote is None


def is_refutation_formula(formula: str) -> bool:
    """Whether a formula is exactly ``$false`` modulo harmless outer layout."""
    formula = canonical_statement(formula, family="atp")
    while _balanced_outer_parentheses(formula):
        formula = formula[1:-1]
    return formula == "$false"


def render_target(steps: list[ProofStep]) -> str:
    """Render every proof edge without truncating the parent list."""
    lines = []
    for i, step in enumerate(steps):
        name = render_tptp_atom(step.name)
        rule = render_tptp_atom(step.rule)
        parents = " ".join(render_tptp_atom(parent) for parent in step.parents)
        line = f"{i + 1:>3}  {name:<10} {step.formula}   [{rule}"
        lines.append(line + (f" {parents}]" if parents else "]"))
    return "\n".join(lines)


def render_block(facts: dict[str, str], local_inputs: dict[str, str]) -> str:
    """Render the masked global-premise and local-input categories."""
    block = HDR + "\n" + "\n".join(f"{name} : {stmt}" for name, stmt in facts.items())
    if local_inputs:
        block += "\n" + LOCAL_HDR + "\n" + "\n".join(
            f"{name} : {stmt}" for name, stmt in local_inputs.items()
        )
    return block


def _stable_blake2(value, *, digest_size: int = 16) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def _text_blake2(value: str, *, digest_size: int = 12) -> str:
    return hashlib.blake2b(
        str(value).encode("utf-8"),
        digest_size=digest_size,
    ).hexdigest()


def _alpha_formula(formula: str) -> str:
    variables: dict[str, str] = {}
    skolems: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        token = match.group(1)
        if token.startswith("X"):
            return variables.setdefault(token, f"V{len(variables)}")
        arity = token.rsplit("_", 1)[1]
        return skolems.setdefault(token, f"SK{arity}_{len(skolems)}")

    return ENIGMA_LOW_TIER_ALPHA_TOKEN.sub(replace, formula)


def _theorem_base_and_suffix(theorem: str) -> tuple[str, int | None]:
    identity = str(theorem).strip()
    prefix, separator, rest = identity.partition(":")
    if separator and prefix.lower() in {"enigma", "prf2"}:
        identity = rest
    match = re.search(r"#(\d+)$", identity)
    if match is None:
        return identity, None
    return identity[:match.start()], int(match.group(1))


def _exact_atp_signature(
    record: dict,
    *,
    canonicalize=_canonical_formula,
) -> str:
    theorem, _ = _theorem_base_and_suffix(record.get("theorem", ""))

    def canonical_mapping(value) -> list[tuple[str, str]]:
        if not isinstance(value, dict):
            return []
        return sorted(
            (
                _unquote_atom(str(name)),
                canonicalize(str(statement)),
            )
            for name, statement in value.items()
        )

    proof_steps = record.get("proof_steps")
    signature = {
        "theorem": theorem,
        "goal": canonicalize(str(record.get("goal", ""))),
        "goal_name": record.get("goal_name"),
        "facts": canonical_mapping(record.get("facts", {})),
        "local_inputs": canonical_mapping(record.get("local_inputs", {})),
    }
    if isinstance(proof_steps, list) and proof_steps:
        canonical_steps = []
        for step in proof_steps:
            if not isinstance(step, dict):
                canonical_steps.append(step)
                continue
            canonical = {}
            for key, value in step.items():
                if key in {"formula", "source"}:
                    canonical[str(key)] = canonicalize(str(value))
                elif key == "parent_sources" and isinstance(value, list):
                    canonical[str(key)] = [
                        canonicalize(str(parent)) for parent in value
                    ]
                else:
                    canonical[str(key)] = value
            canonical_steps.append(canonical)
        signature["schema"] = "atp-v2-structured"
        signature["proof_steps"] = canonical_steps
    else:
        signature["schema"] = "atp-legacy-conservative"
        signature["target"] = (
            str(record.get("target", "")).replace("\r\n", "\n").strip()
        )
    payload = b"mml-atp-exact-structured-v5\0"
    payload += json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _proof_step_dict(step) -> dict:
    return step if isinstance(step, dict) else asdict(step)


def _trace_features(
    record: dict,
    *,
    text_sha256: str,
    raw_sha256: str = "",
    canonicalize=_canonical_formula,
    exact_signature_fn=None,
) -> dict:
    steps = [_proof_step_dict(step) for step in record["proof_steps"]]
    formulas = [canonicalize(str(step["formula"])) for step in steps]
    alpha_formulas = [_alpha_formula(formula) for formula in formulas]
    formula_digests = tuple(_text_blake2(formula) for formula in formulas)
    alpha_digests = tuple(_text_blake2(formula) for formula in alpha_formulas)
    rules = tuple(str(step["rule"]) for step in steps)
    names = [str(step["name"]) for step in steps]
    step_index = {name: index for index, name in enumerate(names)}
    normalized_exact = []
    normalized_alpha = []
    for index, step in enumerate(steps):
        parents = [
            ("s", step_index[str(parent)])
            if str(parent) in step_index
            else ("e", str(parent))
            for parent in step["parents"]
        ]
        normalized_exact.append(
            (str(step["role"]), formula_digests[index], rules[index], parents)
        )
        normalized_alpha.append(
            (str(step["role"]), alpha_digests[index], rules[index], parents)
        )

    def dag_signatures(formula_hashes):
        ordered_by_name = {}
        multiplicity_by_name = {}
        set_by_name = {}
        ordered_nodes = []
        multiplicity_nodes = []
        set_nodes = []
        edge_set = set()
        for index, step in enumerate(steps):
            ordered_parents = []
            multiplicity_parents = []
            set_parents = []
            for raw_parent in step["parents"]:
                parent = str(raw_parent)
                ordered_parents.append(
                    ("s", ordered_by_name[parent])
                    if parent in ordered_by_name
                    else ("e", parent)
                )
                multiplicity_parents.append(
                    ("s", multiplicity_by_name[parent])
                    if parent in multiplicity_by_name
                    else ("e", parent)
                )
                set_parents.append(
                    ("s", set_by_name[parent])
                    if parent in set_by_name
                    else ("e", parent)
                )
            ordered_node = _stable_blake2(
                (
                    str(step["role"]),
                    formula_hashes[index],
                    rules[index],
                    ordered_parents,
                )
            )
            multiplicity_node = _stable_blake2(
                (
                    str(step["role"]),
                    formula_hashes[index],
                    rules[index],
                    sorted(multiplicity_parents),
                )
            )
            set_node = _stable_blake2(
                (
                    str(step["role"]),
                    formula_hashes[index],
                    rules[index],
                    sorted(set(set_parents)),
                )
            )
            name = str(step["name"])
            ordered_by_name[name] = ordered_node
            multiplicity_by_name[name] = multiplicity_node
            set_by_name[name] = set_node
            ordered_nodes.append(ordered_node)
            multiplicity_nodes.append(multiplicity_node)
            set_nodes.append(set_node)
            for parent_digest in set(set_parents):
                edge_set.add(
                    _stable_blake2(
                        (formula_hashes[index], rules[index], parent_digest)
                    )
                )
        return {
            "ordered": _stable_blake2(
                (sorted(ordered_nodes), ordered_nodes[-1])
            ),
            "multiplicity": _stable_blake2(
                (sorted(multiplicity_nodes), multiplicity_nodes[-1])
            ),
            "set": _stable_blake2((sorted(set_nodes), set_nodes[-1])),
            "edges": frozenset(edge_set),
        }

    exact_dag = dag_signatures(formula_digests)
    alpha_dag = dag_signatures(alpha_digests)
    global_pairs = frozenset(
        (
            str(name),
            _text_blake2(canonicalize(str(formula))),
        )
        for name, formula in record["facts"].items()
    )
    local_pairs = frozenset(
        (
            str(name),
            _text_blake2(canonicalize(str(formula))),
        )
        for name, formula in record["local_inputs"].items()
    )
    supplied_formula_hashes = {
        formula_hash for _, formula_hash in global_pairs | local_pairs
    }
    goal = canonicalize(str(record["goal"]))
    supplied_formula_hashes.add(_text_blake2(goal))

    by_name = {str(step["name"]): step for step in steps}
    reachable = set()
    stack = [names[-1]]
    while stack:
        name = stack.pop()
        if name in reachable or name not in by_name:
            continue
        reachable.add(name)
        stack.extend(
            str(parent)
            for parent in by_name[name]["parents"]
            if str(parent) in by_name
        )

    return {
        "raw_sha256": raw_sha256,
        "text_sha256": text_sha256,
        "exact_signature_sha256": (
            exact_signature_fn(record)
            if exact_signature_fn is not None
            else _exact_atp_signature(record, canonicalize=canonicalize)
        ),
        "formula_sequence": formula_digests,
        "formula_set": frozenset(formula_digests),
        "alpha_sequence": alpha_digests,
        "alpha_set": frozenset(alpha_digests),
        "rules": rules,
        "rule_set": frozenset(rules),
        "rule_bigrams": frozenset(pairwise(rules)),
        "core_rules": frozenset(
            rule for rule in rules if rule not in ENIGMA_LOW_TIER_ADMIN_RULES
        ),
        "global_pairs": global_pairs,
        "local_pairs": local_pairs,
        "goal_hash": _text_blake2(goal),
        "goal_name": str(record.get("goal_name", "")),
        "sequence_exact": _stable_blake2(normalized_exact),
        "sequence_alpha": _stable_blake2(normalized_alpha),
        "dag_ordered": exact_dag["ordered"],
        "dag_multiplicity": exact_dag["multiplicity"],
        "dag_set": exact_dag["set"],
        "dag_alpha_ordered": alpha_dag["ordered"],
        "dag_alpha_multiplicity": alpha_dag["multiplicity"],
        "dag_alpha_set": alpha_dag["set"],
        "edges_alpha": alpha_dag["edges"],
        "dead_steps": len(steps) - len(reachable),
        "paste_steps": sum(
            formula in supplied_formula_hashes for formula in formula_digests
        ),
        "proof_steps": len(steps),
    }


def _trace_comparison(candidate: dict, accepted: dict) -> dict:
    same_premises = (
        candidate["global_pairs"] == accepted["global_pairs"]
        and candidate["local_pairs"] == accepted["local_pairs"]
        and candidate["goal_hash"] == accepted["goal_hash"]
        and candidate["goal_name"] == accepted["goal_name"]
    )
    if (
        candidate["raw_sha256"]
        and candidate["raw_sha256"] == accepted["raw_sha256"]
    ):
        diversity_class = "raw_byte_exact"
    elif candidate["text_sha256"] == accepted["text_sha256"]:
        diversity_class = "full_text_exact"
    elif (
        candidate["exact_signature_sha256"]
        == accepted["exact_signature_sha256"]
    ):
        diversity_class = "exact_structured"
    elif same_premises and candidate["sequence_exact"] == accepted["sequence_exact"]:
        diversity_class = "generated_names_or_source_only"
    elif same_premises and candidate["dag_ordered"] == accepted["dag_ordered"]:
        diversity_class = "topological_order_only"
    elif (
        same_premises
        and candidate["dag_multiplicity"] == accepted["dag_multiplicity"]
    ):
        diversity_class = "parent_order_only"
    elif same_premises and candidate["dag_set"] == accepted["dag_set"]:
        diversity_class = "parent_multiplicity_only"
    elif (
        same_premises
        and candidate["dag_alpha_set"] == accepted["dag_alpha_set"]
    ):
        diversity_class = "alpha_formula_surface_only"
    elif same_premises and candidate["formula_set"] == accepted["formula_set"]:
        diversity_class = "same_formula_set_different_path"
    elif same_premises and candidate["alpha_set"] == accepted["alpha_set"]:
        diversity_class = "alpha_formula_set_different_path"
    elif not same_premises:
        diversity_class = "premise_and_path_variant"
    else:
        diversity_class = "formula_rule_path_variant"
    return {
        "class": diversity_class,
        "class_rank": ENIGMA_LOW_TIER_CLASS_RANK[diversity_class],
        "alpha_formula_jaccard": _jaccard(
            candidate["alpha_set"],
            accepted["alpha_set"],
        ),
        "backward_dag_edge_jaccard": _jaccard(
            candidate["edges_alpha"],
            accepted["edges_alpha"],
        ),
    }


def _alternative_features(
    *,
    base: str,
    run: str,
    trace: dict,
    text_plus_eos_tokens: int,
    existing_variants: int,
    material: bool,
    diversity_class: str,
    record: dict | None,
) -> AlternativeProofFeatures:
    return AlternativeProofFeatures(
        base=base,
        run=run,
        raw_sha256=trace["raw_sha256"],
        text_sha256=trace["text_sha256"],
        exact_signature_sha256=trace["exact_signature_sha256"],
        text_plus_eos_tokens=text_plus_eos_tokens,
        existing_variants=existing_variants,
        dead_steps=trace["dead_steps"],
        material=material,
        alpha_formulas=trace["alpha_set"],
        backward_edges=trace["edges_alpha"],
        rule_bigrams=trace["rule_bigrams"],
        premises=trace["global_pairs"] | trace["local_pairs"],
        core_rules=trace["core_rules"],
        paste_steps=trace["paste_steps"],
        proof_steps=trace["proof_steps"],
        diversity_class=diversity_class,
        record=record,
    )


def _output_paths(out: str, name: str) -> tuple[str, str, str]:
    return (
        os.path.join(out, "shards", f"{name}.jsonl"),
        os.path.join(out, "eval", f"{name}.jsonl"),
        os.path.join(out, "heldout", f"{name}.json"),
    )


def write_preserved_base_with_alternatives(
    base_path,
    output_path,
    records: list[dict],
) -> dict:
    """Copy the accepted shard exactly, then append deterministic JSONL rows."""
    base_path = os.fspath(base_path)
    output_path = os.fspath(output_path)
    base_size = os.path.getsize(base_path)
    if base_size <= 0:
        raise ValueError("accepted ENIGMA base must be nonempty")
    with open(base_path, "rb") as base_file:
        base_file.seek(-1, os.SEEK_END)
        if base_file.read(1) != b"\n":
            raise ValueError("accepted ENIGMA base must end with a newline")

    base_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    added_digest = hashlib.sha256()
    base_bytes = 0
    added_bytes = 0
    with open(base_path, "rb") as base_file, open(output_path, "xb") as output_file:
        while True:
            chunk = base_file.read(1024 * 1024)
            if not chunk:
                break
            base_digest.update(chunk)
            output_digest.update(chunk)
            output_file.write(chunk)
            base_bytes += len(chunk)
        for record in records:
            line = (json.dumps(record) + "\n").encode("utf-8")
            output_file.write(line)
            output_digest.update(line)
            added_digest.update(line)
            added_bytes += len(line)
        output_file.flush()
        os.fsync(output_file.fileno())
    return {
        "base_bytes": base_bytes,
        "base_sha256": base_digest.hexdigest(),
        "added_bytes": added_bytes,
        "added_sha256": added_digest.hexdigest(),
        "added_rows": len(records),
        "output_bytes": base_bytes + added_bytes,
        "output_sha256": output_digest.hexdigest(),
    }


def _invalidate_outputs(paths: tuple[str, ...]) -> None:
    """Quarantine prior outputs so a failed rebuild cannot validate as fresh."""
    for path in paths:
        if not os.path.exists(path):
            continue
        stale = path + ".stale"
        suffix = 1
        while os.path.exists(stale):
            stale = f"{path}.stale.{suffix}"
            suffix += 1
        os.replace(path, stale)


def _files_for_source(source: str) -> list[str]:
    pattern = os.path.join(source, "*") if os.path.isdir(source) else source
    return [
        path
        for path in sorted(glob.glob(pattern))
        if os.path.isfile(path) and not path.endswith((".gz", ".tar"))
    ]


def _closure_disposition(errors: list[str]) -> str:
    kinds = set()
    for error in errors:
        if error.endswith(": duplicate target step"):
            kinds.add("duplicate_step")
        elif ": parent " in error and error.endswith(" is not earlier"):
            kinds.add("late_or_cyclic_parent")
        elif ": unresolved parent " in error:
            kinds.add("unresolved_parent")
        else:
            return "unclassified_closure"
    if len(kinds) == 1:
        return kinds.pop()
    if kinds:
        return "mixed_parent_closure"
    return "unclassified_closure"


def source_inventory_errors(
    files: int, dispositions: dict[str, int]
) -> list[str]:
    """Validate exact source accounting and fail closed on closure loss."""
    errors = []
    invalid = {
        name: count
        for name, count in dispositions.items()
        if not isinstance(count, int) or isinstance(count, bool) or count < 0
    }
    if invalid:
        errors.append(f"invalid source dispositions: {invalid}")
        return errors
    unknown = sorted(set(dispositions) - SOURCE_DISPOSITIONS)
    if unknown:
        errors.append(f"unknown source dispositions: {unknown}")
    accounted = sum(dispositions.values())
    if accounted != files:
        errors.append(
            f"source accounting mismatch: {accounted:,} dispositions "
            f"for {files:,} files"
        )
    for disposition in ("unreadable", "malformed_source", "unclassified_closure"):
        count = dispositions.get(disposition, 0)
        if count:
            errors.append(
                f"{disposition}: {count:,} eligible source traces "
                "lack complete parse/parent closure"
            )
    return errors


def _accepted_enigma_inventory(
    base_shard: str,
    *,
    canonicalize,
    exact_signature_fn,
) -> dict:
    digest = hashlib.sha256()
    rows = 0
    ids = set()
    text_sha256s = set()
    signature_sha256s = set()
    canonical_facts: dict[str, str] = {}
    variant_counts = Counter()
    suffixes_by_base: dict[str, set[int | None]] = defaultdict(set)
    suffix_counts = Counter()
    source_metadata_by_hash = {}
    with open(base_shard, "rb") as base_file:
        for line_number, line in enumerate(base_file, 1):
            if not line.endswith(b"\n"):
                raise ValueError(
                    f"accepted ENIGMA base line {line_number} lacks newline"
                )
            digest.update(line)
            rows += 1
            record = json.loads(line)
            if not str(record.get("theorem", "")).startswith("enigma:"):
                raise ValueError(
                    f"accepted ENIGMA base line {line_number} has wrong wrapper"
                )
            row_id = record.get("id")
            if row_id in ids:
                raise ValueError(f"duplicate accepted ENIGMA id {row_id!r}")
            ids.add(row_id)
            base, suffix = _theorem_base_and_suffix(record["theorem"])
            variant_counts[base] += 1
            suffixes_by_base[base].add(suffix)
            suffix_counts["base" if suffix is None else f"#{suffix}"] += 1
            text_sha256 = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            if text_sha256 in text_sha256s:
                raise ValueError("accepted ENIGMA base has duplicate full text")
            text_sha256s.add(text_sha256)
            signature = exact_signature_fn(record)
            if signature in signature_sha256s:
                raise ValueError("accepted ENIGMA base has duplicate exact signature")
            signature_sha256s.add(signature)
            for name, formula in record["facts"].items():
                prior = canonical_facts.setdefault(str(name), str(formula))
                if prior != str(formula):
                    raise ValueError(
                        f"accepted fact {name!r} has inconsistent formulas"
                    )
            metadata = record.get("source_metadata")
            metadata_hash = _canonical_sha256(metadata)
            source_metadata_by_hash.setdefault(metadata_hash, metadata)

    base_contract = {
        "bytes": os.path.getsize(base_shard),
        "rows": rows,
        "sha256": digest.hexdigest(),
    }
    if base_contract != ENIGMA_LOW_TIER_SOURCE_CONTRACT["accepted_base"]:
        raise ValueError(
            "accepted ENIGMA base contract mismatch: "
            f"expected {ENIGMA_LOW_TIER_SOURCE_CONTRACT['accepted_base']}, "
            f"got {base_contract}"
        )
    if len(source_metadata_by_hash) != 1:
        raise ValueError("accepted ENIGMA base has multiple source metadata values")
    for base, count in variant_counts.items():
        if not 1 <= count <= 4:
            raise ValueError(f"accepted theorem {base!r} has {count} variants")
        expected_suffixes = {None, *range(2, count + 1)}
        if suffixes_by_base[base] != expected_suffixes:
            raise ValueError(
                f"accepted theorem {base!r} suffixes drifted: "
                f"{sorted(suffixes_by_base[base], key=lambda value: value or 0)}"
            )
    if suffix_counts["#1"]:
        raise ValueError("accepted ENIGMA base already uses reserved #1 suffix")

    accepted_by_base: dict[str, list[AlternativeProofFeatures]] = {}
    accepted_trace_by_base = {}
    with open(base_shard, "rb") as base_file:
        for line in base_file:
            record = json.loads(line)
            base, _ = _theorem_base_and_suffix(record["theorem"])
            if variant_counts[base] != 1:
                continue
            text_sha256 = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            trace = _trace_features(
                record,
                text_sha256=text_sha256,
                canonicalize=canonicalize,
                exact_signature_fn=exact_signature_fn,
            )
            accepted_trace_by_base[base] = trace
            accepted_by_base[base] = [
                _alternative_features(
                    base=base,
                    run="mzr01",
                    trace=trace,
                    text_plus_eos_tokens=0,
                    existing_variants=1,
                    material=True,
                    diversity_class="accepted",
                    record=None,
                )
            ]
    return {
        "base_contract": base_contract,
        "ids": ids,
        "text_sha256s": text_sha256s,
        "signature_sha256s": signature_sha256s,
        "canonical_facts": canonical_facts,
        "variant_counts": variant_counts,
        "suffixes_by_base": suffixes_by_base,
        "suffix_counts": suffix_counts,
        "source_metadata": next(iter(source_metadata_by_hash.values())),
        "accepted_by_base": accepted_by_base,
        "accepted_trace_by_base": accepted_trace_by_base,
    }


def _load_enigma_low_tier_tokenizer(tokenizer_json: str):
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ValueError("the fixed tokenizers package is unavailable") from error

    tokenizer_json = os.path.abspath(tokenizer_json)
    config_path = os.path.join(os.path.dirname(tokenizer_json), "tokenizer_config.json")
    with open(tokenizer_json, "rb") as tokenizer_file:
        tokenizer_sha256 = hashlib.sha256(tokenizer_file.read()).hexdigest()
    with open(config_path, "rb") as config_file:
        config_sha256 = hashlib.sha256(config_file.read()).hexdigest()
    tokenizer = Tokenizer.from_file(tokenizer_json)
    actual = {
        "tokenizer_json_sha256": tokenizer_sha256,
        "tokenizer_config_sha256": config_sha256,
        "tokenizers_version": tokenizers.__version__,
        "eos_token_id": tokenizer.token_to_id("<|endoftext|>"),
        "separator_ids": tokenizer.encode(
            "---\nGOAL",
            add_special_tokens=False,
        ).ids,
    }
    if actual != ENIGMA_LOW_TIER_SOURCE_CONTRACT["tokenizer"]:
        raise ValueError(
            "fixed tokenizer contract mismatch: "
            f"expected {ENIGMA_LOW_TIER_SOURCE_CONTRACT['tokenizer']}, "
            f"got {actual}"
        )
    return tokenizer, actual


def _write_json_fsync(path: str, value: dict) -> tuple[int, str]:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with open(path, "xb") as output_file:
        output_file.write(payload)
        output_file.flush()
        os.fsync(output_file.fileno())
    return len(payload), hashlib.sha256(payload).hexdigest()


def _copy_file_exact(source: str, destination: str) -> dict:
    digest = hashlib.sha256()
    size = 0
    with open(source, "rb") as source_file, open(destination, "xb") as output_file:
        while True:
            chunk = source_file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output_file.write(chunk)
            size += len(chunk)
        output_file.flush()
        os.fsync(output_file.fileno())
    return {"bytes": size, "sha256": digest.hexdigest()}


def _enigma_low_tier_actual_contract(
    *,
    inventory: dict,
    source_metadata: dict,
    source_order: list[str],
    source_files: int,
    dispositions: Counter,
    tokenizer_contract: dict,
) -> dict:
    return {
        "accepted_base": inventory["base_contract"],
        "source_files": source_files,
        "source_order": source_order,
        "source_dispositions": {
            key: dispositions[key]
            for key in sorted(dispositions)
            if dispositions[key]
        },
        "source_manifest_root_sha256": source_metadata[
            "source_manifest_root_sha256"
        ],
        "quality_filter_root_sha256": source_metadata[
            "quality_filter_root_sha256"
        ],
        "schema_generation_root_sha256": source_metadata[
            "schema_generation_root_sha256"
        ],
        "source_roots": source_metadata["source_roots"],
        "tokenizer": tokenizer_contract,
    }


def _build_enigma_low_tier(args, output_paths: tuple[str, str, str]) -> int:
    expected_cli = {
        "name": "enigma",
        "fenced": True,
        "heldout": 0,
        "min_steps": ENIGMA_LOW_TIER_POLICY["legacy_min_steps"],
        "dedup": True,
        "jaccard": ENIGMA_LOW_TIER_POLICY["legacy_redundancy_jaccard"],
        "seed": ENIGMA_LOW_TIER_POLICY["seed"],
    }
    actual_cli = {key: getattr(args, key) for key in expected_cli}
    if actual_cli != expected_cli:
        raise ValueError(
            f"conservative ENIGMA CLI mismatch: expected {expected_cli}, "
            f"got {actual_cli}"
        )
    if os.path.lexists(args.out):
        raise ValueError(f"low-tier output root must be fresh: {args.out}")
    source_order = [
        os.path.basename(os.path.normpath(source)) for source in args.src
    ]
    if source_order != ENIGMA_LOW_TIER_SOURCE_CONTRACT["source_order"]:
        raise ValueError(
            "ENIGMA source order mismatch: "
            f"expected {ENIGMA_LOW_TIER_SOURCE_CONTRACT['source_order']}, "
            f"got {source_order}"
        )

    base_root = os.path.abspath(args.enigma_low_tier_base)
    base_shard = os.path.join(base_root, "shards", "enigma.jsonl")
    base_heldout = os.path.join(base_root, "heldout", "enigma.json")
    base_eval = os.path.join(base_root, "eval", "enigma.jsonl")
    import split_mml_semantic_holdout as semantic_holdout

    def canonicalize(formula):
        return semantic_holdout.canonical_statement(
            formula,
            representation="atp",
        )

    inventory = _accepted_enigma_inventory(
        base_shard,
        canonicalize=canonicalize,
        exact_signature_fn=semantic_holdout.exact_atp_signature,
    )
    tokenizer, tokenizer_contract = _load_enigma_low_tier_tokenizer(
        args.tokenizer_json
    )

    files = []
    source_of = {}
    missing_sources = []
    for source in args.src:
        source_files = _files_for_source(source)
        if not source_files:
            missing_sources.append(source)
        for path in source_files:
            files.append(path)
            source_of[path] = source
    if missing_sources:
        raise ValueError(f"source(s) matched no proof files: {missing_sources}")
    source_metadata = _build_source_metadata(
        args.src,
        files,
        source_of,
        args,
    )
    print(f"  {len(files):,} files across {len(args.src)} source(s)")

    previous_formula_sets: dict[str, list[frozenset[str]]] = defaultdict(list)
    dispositions = Counter()
    dispositions_by_run = {
        run: Counter() for run in ENIGMA_LOW_TIER_POLICY["run_priority"]
    }
    accepted_by_run = Counter()
    redundant_by_run = Counter()
    source_accepted_counts = Counter()
    unknown_fact_values: dict[str, set[str]] = defaultdict(set)
    eligible_candidates: list[tuple[AlternativeProofFeatures, tuple[str, ...]]] = []
    diversity_classes = Counter()
    candidate_text_counts = Counter()
    candidate_signature_counts = Counter()
    candidate_only_fact_rows = 0
    canonical_fact_mismatch_rows = 0
    dead_step_rows = 0
    paste_rows = 0
    paste_steps = 0
    over_8192_rows = 0
    over_16384_rows = 0
    schema_rows = 0

    for path in files:
        run = os.path.basename(os.path.normpath(source_of[path]))
        run_dispositions = dispositions_by_run[run]
        try:
            with open(path, "rb") as proof_file:
                raw_bytes = proof_file.read()
        except OSError:
            dispositions["unreadable"] += 1
            run_dispositions["unreadable"] += 1
            continue
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        proof = parse(raw_bytes.decode("utf-8", errors="replace"), args.fenced)
        if proof is None:
            dispositions["unsolved_or_unfenced"] += 1
            run_dispositions["unsolved_or_unfenced"] += 1
            continue
        if not proof.facts or not proof.goal or len(proof.steps) < args.min_steps:
            dispositions["too_thin"] += 1
            run_dispositions["too_thin"] += 1
            continue
        if proof.source_errors:
            dispositions["malformed_source"] += 1
            run_dispositions["malformed_source"] += 1
            continue
        closure_errors = dependency_errors(proof)
        if closure_errors:
            disposition = _closure_disposition(closure_errors)
            dispositions[disposition] += 1
            run_dispositions[disposition] += 1
            continue
        if not is_refutation_formula(proof.steps[-1].formula):
            dispositions["non_refutation"] += 1
            run_dispositions["non_refutation"] += 1
            continue

        base = os.path.basename(path)
        formula_set = frozenset(step.formula for step in proof.steps)
        previous = previous_formula_sets[base]
        redundant = any(
            len(formula_set & accepted_formula_set)
            / max(len(formula_set | accepted_formula_set), 1)
            >= args.jaccard
            for accepted_formula_set in previous
        )
        if not redundant:
            source_ordinal = len(previous)
            previous.append(formula_set)
            dispositions["accepted"] += 1
            run_dispositions["accepted"] += 1
            accepted_by_run[run] += 1
            source_accepted_counts[base] += 1
            if inventory["variant_counts"].get(base) == 1:
                if source_ordinal != 0:
                    raise ValueError(
                        f"singleton theorem {base!r} has multiple source accepts"
                    )
                trace = inventory["accepted_trace_by_base"][base]
                source_formula_set = frozenset(
                    _text_blake2(canonicalize(step.formula))
                    for step in proof.steps
                )
                if source_formula_set != trace["formula_set"]:
                    raise ValueError(
                        f"accepted source formulas drifted for theorem {base!r}"
                    )
                trace["raw_sha256"] = raw_sha256
                inventory["accepted_by_base"][base] = [
                    _alternative_features(
                        base=base,
                        run=run,
                        trace=trace,
                        text_plus_eos_tokens=0,
                        existing_variants=1,
                        material=True,
                        diversity_class="accepted",
                        record=None,
                    )
                ]
            continue

        dispositions["redundant_reproof"] += 1
        run_dispositions["redundant_reproof"] += 1
        redundant_by_run[run] += 1
        existing_variants = inventory["variant_counts"].get(base, 0)
        if not existing_variants:
            raise ValueError(
                f"redundant source theorem {base!r} is absent from accepted base"
            )
        proposed_base = f"{base}#{existing_variants}"
        row_id = hashlib.md5(f"enigma/{proposed_base}".encode()).hexdigest()[:12]
        cited = list(proof.facts)
        fact_order = list(cited)
        random.Random(row_id).shuffle(fact_order)
        facts = {}
        unknown_names = []
        canonical_mismatches = []
        for name in fact_order:
            raw_formula = proof.facts[name]
            if name in inventory["canonical_facts"]:
                facts[name] = inventory["canonical_facts"][name]
                if raw_formula != facts[name]:
                    canonical_mismatches.append(name)
            else:
                facts[name] = raw_formula
                unknown_names.append(name)
                unknown_fact_values[name].add(_text_blake2(raw_formula))
        if unknown_names:
            candidate_only_fact_rows += 1
        if canonical_mismatches:
            canonical_fact_mismatch_rows += 1
        target = render_target(proof.steps)
        block = render_block(facts, proof.local_inputs)
        text = f"{block}\n{SEP}\nGOAL {proof.goal}\n{target}"
        text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        text_plus_eos_tokens = (
            len(tokenizer.encode(text, add_special_tokens=False).ids) + 1
        )
        if text_plus_eos_tokens > 8_192:
            over_8192_rows += 1
        if text_plus_eos_tokens > 16_384:
            over_16384_rows += 1
        record = {
            "id": row_id,
            "theorem": f"enigma:{proposed_base}",
            "facts": facts,
            "cited": cited,
            "local_inputs": proof.local_inputs,
            "goal_name": proof.goal_name,
            "goal": proof.goal,
            "proof_steps": [asdict(step) for step in proof.steps],
            "target": target,
            "text": text,
            "schema_version": ROW_SCHEMA,
            "source_metadata": source_metadata,
            "mask_start": 0,
            "mask_end": len(block),
        }
        semantic_holdout._validate_row_schema(
            record,
            shard="enigma",
            line_number=dispositions["redundant_reproof"],
        )
        schema_rows += 1
        if normalize_theorem_identity(record["theorem"], family="atp") != base:
            raise ValueError(
                f"current theorem normalization split variant {record['theorem']!r}"
            )
        trace = _trace_features(
            record,
            text_sha256=text_sha256,
            raw_sha256=raw_sha256,
            canonicalize=canonicalize,
            exact_signature_fn=semantic_holdout.exact_atp_signature,
        )
        candidate_text_counts[text_sha256] += 1
        candidate_signature_counts[trace["exact_signature_sha256"]] += 1
        dead_step_rows += trace["dead_steps"] > 0
        paste_rows += trace["paste_steps"] > 0
        paste_steps += trace["paste_steps"]

        diversity_class = "existing_multi_variant"
        material = False
        if existing_variants == 1:
            accepted_trace = inventory["accepted_trace_by_base"][base]
            if not accepted_trace["raw_sha256"]:
                raise ValueError(
                    f"candidate {base!r} preceded its retained source trace"
                )
            comparison = _trace_comparison(trace, accepted_trace)
            diversity_class = comparison["class"]
            material = (
                diversity_class not in ENIGMA_LOW_TIER_NONMATERIAL
            )
        diversity_classes[diversity_class] += 1
        candidate = _alternative_features(
            base=base,
            run=run,
            trace=trace,
            text_plus_eos_tokens=text_plus_eos_tokens,
            existing_variants=existing_variants,
            material=material,
            diversity_class=diversity_class,
            record=record,
        )
        if (
            existing_variants == 1
            and text_sha256 not in inventory["text_sha256s"]
            and trace["exact_signature_sha256"]
            not in inventory["signature_sha256s"]
            and conservative_alternative_is_eligible(
                candidate,
                inventory["accepted_by_base"][base][0],
            )
        ):
            eligible_candidates.append((candidate, tuple(unknown_names)))

    inventory_errors = source_inventory_errors(len(files), dispositions)
    if inventory_errors:
        raise ValueError("; ".join(inventory_errors))
    if source_accepted_counts != inventory["variant_counts"]:
        raise ValueError(
            "accepted theorem/source multiplicities differ from sealed base"
        )
    actual_contract = _enigma_low_tier_actual_contract(
        inventory=inventory,
        source_metadata=source_metadata,
        source_order=source_order,
        source_files=len(files),
        dispositions=dispositions,
        tokenizer_contract=tokenizer_contract,
    )
    validate_enigma_low_tier_contract(actual_contract)
    if schema_rows != ENIGMA_LOW_TIER_POLICY["expected_redundant_dispositions"]:
        raise ValueError(
            f"schema replay covered {schema_rows:,} redundant rows, "
            f"expected {ENIGMA_LOW_TIER_POLICY['expected_redundant_dispositions']:,}"
        )

    inconsistent_candidate_facts = {
        name for name, formulas in unknown_fact_values.items() if len(formulas) != 1
    }
    candidates = [
        candidate
        for candidate, unknown_names in eligible_candidates
        if not inconsistent_candidate_facts.intersection(unknown_names)
    ]
    selected = select_conservative_alternatives(
        candidates,
        inventory["accepted_by_base"],
        existing_text_sha256s=inventory["text_sha256s"],
        existing_signature_sha256s=inventory["signature_sha256s"],
    )
    selected_tokens = sum(
        candidate.text_plus_eos_tokens for candidate in selected
    )
    packed_tokens = (
        math.ceil(selected_tokens / 16_384) * 16_384 if selected_tokens else 0
    )
    selected_by_run = Counter(candidate.run for candidate in selected)
    expected_selection = {
        "rows": ENIGMA_LOW_TIER_POLICY["expected_selected_rows"],
        "text_plus_eos_tokens": ENIGMA_LOW_TIER_POLICY[
            "expected_text_plus_eos_tokens"
        ],
        "packed_16384_tokens": ENIGMA_LOW_TIER_POLICY[
            "expected_packed_16384_tokens"
        ],
        "run_distribution": ENIGMA_LOW_TIER_POLICY[
            "expected_run_distribution"
        ],
    }
    actual_selection = {
        "rows": len(selected),
        "text_plus_eos_tokens": selected_tokens,
        "packed_16384_tokens": packed_tokens,
        "run_distribution": {
            run: selected_by_run[run]
            for run in ENIGMA_LOW_TIER_POLICY["run_priority"]
            if selected_by_run[run]
        },
    }
    if actual_selection != expected_selection:
        raise ValueError(
            f"audited low-tier selection mismatch: expected {expected_selection}, "
            f"got {actual_selection}; thresholds are sealed and were not tuned"
        )
    if any(candidate.record is None for candidate in selected):
        raise ValueError("selected ENIGMA candidate lacks a source record")
    selected_records = [candidate.record for candidate in selected]
    selected_ids = [record["id"] for record in selected_records]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected ENIGMA alternatives have duplicate IDs")
    if inventory["ids"].intersection(selected_ids):
        raise ValueError("selected ENIGMA alternative collides with accepted ID")

    mapped_classes = 0
    singleton_classes = 0
    for candidate in selected:
        base_identity = semantic_holdout.semantic_identity(
            f"enigma:{candidate.base}",
            representation="atp",
            theorem_identity=True,
        )
        variants = [f"enigma:{candidate.base}#1"]
        variants.extend(
            f"enigma:{candidate.base}#{suffix}"
            for suffix in sorted(
                suffix
                for suffix in inventory["suffixes_by_base"][candidate.base]
                if suffix is not None
            )
        )
        for variant in variants:
            identity = semantic_holdout.semantic_identity(
                variant,
                representation="atp",
                theorem_identity=True,
            )
            if identity.class_id != base_identity.class_id:
                raise ValueError(
                    f"semantic theorem normalization split {variant!r}"
                )
        mapped_classes += base_identity.mapped
        singleton_classes += not base_identity.mapped

    os.makedirs(args.out, mode=0o700, exist_ok=False)
    for directory in ("shards", "eval", "heldout"):
        os.makedirs(os.path.join(args.out, directory), mode=0o700)
    train_path, eval_path, heldout_path = output_paths
    train_temp = f"{train_path}.tmp.{os.getpid()}"
    eval_temp = f"{eval_path}.tmp.{os.getpid()}"
    heldout_temp = f"{heldout_path}.tmp.{os.getpid()}"
    shard_stats = write_preserved_base_with_alternatives(
        base_shard,
        train_temp,
        selected_records,
    )
    with open(eval_temp, "xb") as eval_file:
        eval_file.flush()
        os.fsync(eval_file.fileno())
    if os.path.exists(base_eval) and os.path.getsize(base_eval) != 0:
        raise ValueError("accepted ENIGMA base eval shard is not empty")
    heldout_stats = _copy_file_exact(base_heldout, heldout_temp)
    os.replace(train_temp, train_path)
    os.replace(eval_temp, eval_path)
    os.replace(heldout_temp, heldout_path)

    selected_rows = [
        {
            "base": candidate.base,
            "id": candidate.record["id"],
            "raw_sha256": candidate.raw_sha256,
            "run": candidate.run,
            "text_plus_eos_tokens": candidate.text_plus_eos_tokens,
            "text_sha256": candidate.text_sha256,
            "exact_signature_sha256": candidate.exact_signature_sha256,
        }
        for candidate in selected
    ]
    selected_paste_rows = sum(candidate.paste_steps > 0 for candidate in selected)
    selected_paste_steps = sum(candidate.paste_steps for candidate in selected)
    selected_max_paste_fraction = max(
        (
            candidate.paste_steps / candidate.proof_steps
            for candidate in selected
            if candidate.proof_steps
        ),
        default=0.0,
    )
    audit = {
        "schema_version": ENIGMA_LOW_TIER_SCHEMA,
        "policy": ENIGMA_LOW_TIER_POLICY,
        "policy_root_sha256": _canonical_sha256(ENIGMA_LOW_TIER_POLICY),
        "source_contract": actual_contract,
        "source_contract_root_sha256": _canonical_sha256(actual_contract),
        "source_replay": {
            "files": len(files),
            "dispositions": {
                key: dispositions[key]
                for key in sorted(dispositions)
                if dispositions[key]
            },
            "dispositions_by_run": {
                run: {
                    key: dispositions_by_run[run][key]
                    for key in sorted(dispositions_by_run[run])
                    if dispositions_by_run[run][key]
                }
                for run in ENIGMA_LOW_TIER_POLICY["run_priority"]
            },
            "accepted_by_run": dict(accepted_by_run),
            "redundant_by_run": dict(redundant_by_run),
            "schema_rows": schema_rows,
        },
        "base": {
            **inventory["base_contract"],
            "suffix_counts": dict(inventory["suffix_counts"]),
            "unique_ids": len(inventory["ids"]),
            "unique_texts": len(inventory["text_sha256s"]),
            "unique_signatures": len(inventory["signature_sha256s"]),
            "prefix_preserved": (
                shard_stats["base_sha256"]
                == inventory["base_contract"]["sha256"]
                and shard_stats["base_bytes"]
                == inventory["base_contract"]["bytes"]
            ),
        },
        "candidate_diagnostics": {
            "diversity_classes": dict(diversity_classes),
            "candidate_only_fact_rows": candidate_only_fact_rows,
            "inconsistent_candidate_fact_names": sorted(
                inconsistent_candidate_facts
            ),
            "canonical_fact_surface_mismatch_rows": (
                canonical_fact_mismatch_rows
            ),
            "dead_step_rows": dead_step_rows,
            "paste_rows": paste_rows,
            "paste_steps": paste_steps,
            "over_8192_rows": over_8192_rows,
            "over_16384_rows": over_16384_rows,
            "full_text_duplicate_extra_rows": sum(
                count - 1 for count in candidate_text_counts.values() if count > 1
            ),
            "exact_signature_duplicate_extra_rows": sum(
                count - 1
                for count in candidate_signature_counts.values()
                if count > 1
            ),
        },
        "selection": {
            **actual_selection,
            "mapped_theorem_classes": mapped_classes,
            "representation_singleton_classes": singleton_classes,
            "diversity_classes": dict(
                Counter(candidate.diversity_class for candidate in selected)
            ),
            "paste_diagnostics": {
                "rows_with_paste": selected_paste_rows,
                "paste_steps": selected_paste_steps,
                "max_paste_step_fraction": selected_max_paste_fraction,
            },
            "occurrence_root_sha256": _canonical_sha256(selected_rows),
            "rows": selected_rows,
        },
        "outputs": {
            "shards/enigma.jsonl": {
                **shard_stats,
                "rows": (
                    inventory["base_contract"]["rows"]
                    + actual_selection["rows"]
                ),
            },
            "eval/enigma.jsonl": {
                "bytes": 0,
                "rows": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            },
            "heldout/enigma.json": heldout_stats,
        },
        "theorem_normalization": {
            "reserved_suffix": "#1",
            "existing_suffixes_preserved": ["#2", "#3", "#4"],
            "normalized_selected_variants": len(selected),
            "all_grouped": True,
        },
    }
    audit_path = os.path.join(args.out, "enigma-low-tier-v1.audit.json")
    audit_bytes, audit_sha256 = _write_json_fsync(audit_path, audit)
    seal = {
        "schema_version": "enigma-alternative-proof-low-tier-acceptance-v1",
        "status": "ACCEPTED",
        "gates": [
            "accepted_prefix_byte_identity",
            "exact_source_inventory_accounting",
            "source_closure_and_final_refutation",
            "atp_v2_schema_replay",
            "fixed_tokenizer_contract",
            "conservative_policy_exact_counts",
            "global_text_and_signature_deduplication",
            "theorem_variant_normalization",
            "deterministic_occurrence_root",
        ],
        "audit": {
            "bytes": audit_bytes,
            "path": os.path.basename(audit_path),
            "sha256": audit_sha256,
        },
        "rows": {
            "base": inventory["base_contract"]["rows"],
            "added": actual_selection["rows"],
            "total": (
                inventory["base_contract"]["rows"]
                + actual_selection["rows"]
            ),
        },
        "tokens": {
            "added_text_plus_eos": selected_tokens,
            "added_packed_16384": packed_tokens,
        },
        "shard": {
            "bytes": shard_stats["output_bytes"],
            "sha256": shard_stats["output_sha256"],
        },
        "roots": {
            "policy_root_sha256": audit["policy_root_sha256"],
            "source_manifest_root_sha256": source_metadata[
                "source_manifest_root_sha256"
            ],
            "selected_occurrence_root_sha256": audit["selection"][
                "occurrence_root_sha256"
            ],
        },
        "run_distribution": actual_selection["run_distribution"],
    }
    seal_path = os.path.join(args.out, "enigma-low-tier-v1.ACCEPTED.json")
    _write_json_fsync(seal_path, seal)
    print(
        f"  conservative ENIGMA tier: {len(selected):,} additions, "
        f"{selected_tokens:,} text+EOS tokens, {packed_tokens:,} packed tokens"
    )
    print(
        f"  output rows {seal['rows']['total']:,}   "
        f"bytes {shard_stats['output_bytes']:,}   "
        f"sha256 {shard_stats['output_sha256']}"
    )
    print(f"  acceptance seal {seal_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--fenced", action="store_true",
                    help="ENIGMA layout: filter to solved, read SZS fence")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--min-steps", type=int, default=4)
    ap.add_argument("--dedup", action="store_true",
                    help="ENIGMA: drop re-derivations of a theorem already kept")
    ap.add_argument("--jaccard", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument(
        "--enigma-low-tier-base",
        help="accepted ENIGMA root to preserve before conservative #1 additions",
    )
    ap.add_argument(
        "--tokenizer-json",
        help="fixed Qwen tokenizer.json used for the conservative 8K gate",
    )
    a = ap.parse_args()
    output_paths = _output_paths(a.out, a.name)
    if a.enigma_low_tier_base is not None:
        if not a.tokenizer_json:
            print(
                "  --tokenizer-json is required with --enigma-low-tier-base",
                file=sys.stderr,
            )
            return 2
        try:
            return _build_enigma_low_tier(a, output_paths)
        except (OSError, ValueError) as error:
            print(f"  conservative ENIGMA build rejected: {error}", file=sys.stderr)
            return 1
    if a.tokenizer_json is not None:
        print(
            "  --tokenizer-json is valid only with --enigma-low-tier-base",
            file=sys.stderr,
        )
        return 2

    # Sort WITHIN a source but keep the sources in the order given: with --dedup
    # the first proof of a theorem wins, so listing already-built archives first
    # means later ones contribute only genuinely different derivations.
    files = []
    source_of = {}
    missing_sources = []
    for source in a.src:
        source_files = _files_for_source(source)
        if not source_files:
            missing_sources.append(source)
        for path in source_files:
            files.append(path)
            source_of[path] = source
    if missing_sources:
        print(f"  source(s) matched no proof files: {missing_sources}", file=sys.stderr)
        _invalidate_outputs(output_paths)
        return 2
    source_metadata = _build_source_metadata(a.src, files, source_of, a)
    print(f"  {len(files):,} files across {len(a.src)} source(s)")

    parsed = []
    stmts, counts = {}, Counter()
    dispositions = Counter()
    sigs = {}
    per_src = Counter()
    for p in files:
        try:
            with open(p, encoding="utf-8", errors="replace") as proof_file:
                txt = proof_file.read()
        except OSError:
            dispositions["unreadable"] += 1
            continue
        r = parse(txt, a.fenced)
        if r is None:
            dispositions["unsolved_or_unfenced"] += 1
            continue
        if not r.facts or not r.goal or len(r.steps) < a.min_steps:
            dispositions["too_thin"] += 1
            continue
        if r.source_errors:
            dispositions["malformed_source"] += 1
            continue
        closure_errors = dependency_errors(r)
        if closure_errors:
            dispositions[_closure_disposition(closure_errors)] += 1
            continue
        if not is_refutation_formula(r.steps[-1].formula):
            dispositions["non_refutation"] += 1
            continue
        base = os.path.basename(p)
        if a.dedup:
            sig = frozenset(step.formula for step in r.steps)
            prev = sigs.get(base, [])
            if any(len(sig & q) / max(len(sig | q), 1) >= a.jaccard
                   for q in prev):
                dispositions["redundant_reproof"] += 1
                continue
            sigs.setdefault(base, []).append(sig)
            base = f"{base}#{len(prev)}" if prev else base
        parsed.append((base, r))
        dispositions["accepted"] += 1
        per_src[source_of[p]] += 1
        for n, f in r.facts.items():
            stmts.setdefault(n, f)
            counts[n] += 1
    closure_counts = {
        key: dispositions[key]
        for key in sorted(CLOSURE_DISPOSITIONS)
        if dispositions[key]
    }
    print(
        f"  parsed {len(parsed):,}   "
        "unsolved/unfenced "
        f"{dispositions['unsolved_or_unfenced']:,}   "
        f"too thin {dispositions['too_thin']:,}   "
        f"malformed source {dispositions['malformed_source']:,}   "
        f"typed closure rejection {sum(closure_counts.values()):,}   "
        f"non-refutation {dispositions['non_refutation']:,}   "
        f"redundant re-proof {dispositions['redundant_reproof']:,}   "
        f"unreadable {dispositions['unreadable']:,}"
    )
    if closure_counts:
        print(f"  closure dispositions: {dict(closure_counts)}")
    inventory_errors = source_inventory_errors(len(files), dispositions)
    if inventory_errors:
        for error in inventory_errors:
            print(f"  source inventory rejected: {error}", file=sys.stderr)
        _invalidate_outputs(output_paths)
        return 1
    if not parsed:
        print("  no accepted refutation proofs; outputs invalidated", file=sys.stderr)
        _invalidate_outputs(output_paths)
        return 1
    print(f"  distinct named premises: {len(stmts):,}")
    for s in a.src:
        print(f"    {os.path.basename(s):<8} contributed {per_src[s]:>7,} "
              f"proofs surviving dedup")

    tail = sorted(n for n, c in counts.items() if c in (1, 2))
    held = set(random.Random(a.seed).sample(tail, min(a.heldout, len(tail))))
    held_statements = {stmts[name] for name in held}
    held_hashes = sorted(
        statement_hash(statement, family="atp")
        for statement in held_statements
    )
    manifest = {
        "facts": sorted(held),
        "seed": a.seed,
        "corpus": a.name,
        "family": "atp",
        "shards": [a.name],
        "statement_hashes": held_hashes,
        "canonicalization": canonicalization_metadata("atp"),
        "policy": "premises cited 1-2x; citing proofs, all alternate "
        "proofs, and exact statement aliases removed",
    }
    print(f"  held out {len(held):,} of {len(tail):,} premises cited 1-2x")

    kept = ev = dup = 0
    tb = 0
    seen = set()
    sp = output_paths[0]
    for directory in ("shards", "eval", "heldout"):
        os.makedirs(os.path.join(a.out, directory), exist_ok=True)
    temp_paths = tuple(f"{path}.tmp.{os.getpid()}" for path in output_paths)
    train_tmp, eval_tmp, manifest_tmp = temp_paths
    try:
        with open(train_tmp, "w", encoding="utf-8") as fh, open(
            eval_tmp, "w", encoding="utf-8"
        ) as evf:
            for base, proof in parsed:
                used = list(proof.facts)
                eid = hashlib.md5(f"{a.name}/{base}".encode()).hexdigest()[:12]
                order = list(used)
                random.Random(eid).shuffle(order)
                # Render from the canonical map: E normalises the same Mizar
                # definition differently per problem, so the per-file text of
                # d2_member_1 varies. A fact store has one statement per name.
                blk = {n: stmts[n] for n in order}
                target = render_target(proof.steps)
                block = render_block(blk, proof.local_inputs)
                text = f"{block}\n{SEP}\nGOAL {proof.goal}\n{target}"
                if text in seen:
                    dup += 1
                    continue
                seen.add(text)
                rec = {
                    "id": eid,
                    "theorem": f"{a.name}:{base}",
                    "facts": blk,
                    "cited": used,
                    "local_inputs": proof.local_inputs,
                    "goal_name": proof.goal_name,
                    "goal": proof.goal,
                    "proof_steps": [asdict(step) for step in proof.steps],
                    "target": target,
                    "text": text,
                    "schema_version": ROW_SCHEMA,
                    "source_metadata": source_metadata,
                    "mask_start": 0,
                    "mask_end": len(block),
                }
                bare = re.sub(r"#\d+$", "", base)
                exposed_statements = (
                    set(blk.values())
                    | set(proof.local_inputs.values())
                    | {proof.goal}
                    | {step.formula for step in proof.steps}
                )
                if (
                    set(used) & held
                    or bare in held
                    or exposed_statements & held_statements
                ):
                    evf.write(json.dumps(rec) + "\n")
                    ev += 1
                else:
                    fh.write(json.dumps(rec) + "\n")
                    kept += 1
                    tb += len(text.encode())
        with open(manifest_tmp, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=1)
        for temp, final in zip(temp_paths, output_paths):
            os.replace(temp, final)
    except Exception:
        _invalidate_outputs(output_paths)
        raise
    finally:
        for temp in temp_paths:
            if os.path.exists(temp):
                os.remove(temp)

    print(f"\n  train {kept:,}   eval {ev:,}   duplicate {dup:,}")
    print(f"  {tb/1e6:.1f} MB text  ~{tb/2.2/1e6:.0f}M GPT-2 tokens")
    print(f"  wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
