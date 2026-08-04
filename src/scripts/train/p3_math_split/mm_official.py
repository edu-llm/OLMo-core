"""Pinned official-verifier oracle for supported expression traces."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from mm_expand import MM
from mm_verify import (
    MATCH_NODE_BUDGET,
    SearchBudget,
    SyntaxTypeChecker,
    VerificationStatus,
    _get_frame,
    _iter_template_matches,
    apply_subst,
    norm,
    parse_proof,
    rule_parts,
    verify_proof,
)

MANIFEST_PATH = Path(__file__).with_name("metamath_verifier_manifest.json")
DEFAULT_MAX_PROOF_TOKENS = 200_000


@dataclass(frozen=True)
class OfficialResult:
    """Tri-state result from the pinned official executable."""

    status: VerificationStatus
    reason_code: str = ""
    reason: str = ""
    output: str = ""
    proof_tokens: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_binary(binary_path: Path) -> Optional[OfficialResult]:
    if not binary_path.is_file():
        return OfficialResult(
            VerificationStatus.UNKNOWN,
            "official_binary_missing",
            f"official verifier does not exist: {binary_path}",
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["binary"]["sha256"]
    actual = _sha256(binary_path)
    if actual != expected:
        return OfficialResult(
            VerificationStatus.UNKNOWN,
            "official_binary_hash_mismatch",
            f"expected official binary SHA-256 {expected}, got {actual}",
        )
    return None


def _run_official(
    source_path: Path,
    binary_path: Path,
    target_label: str,
) -> OfficialResult:
    binary_error = _check_binary(binary_path)
    if binary_error is not None:
        return binary_error
    completed = subprocess.run(
        [str(binary_path), str(source_path)],
        input=f"VERIFY PROOF {target_label}\nEXIT\n",
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    error_lines = [
        line.strip()
        for line in output.splitlines()
        if line.lstrip().startswith("?")
    ]
    if completed.returncode != 0 or error_lines:
        detail = error_lines[0] if error_lines else (
            f"official verifier exited {completed.returncode}"
        )
        return OfficialResult(
            VerificationStatus.INVALID,
            "official_verify_rejected",
            detail,
            output,
        )
    if target_label not in output:
        return OfficialResult(
            VerificationStatus.UNKNOWN,
            "official_output_unrecognized",
            "official verifier produced no recognizable proof result",
            output,
        )
    return OfficialResult(
        VerificationStatus.VALID,
        output=output,
    )


def verify_source_proof(
    *,
    source_path: Path,
    binary_path: Path,
    target_label: str,
) -> OfficialResult:
    """Run official ``VERIFY PROOF`` on one unmodified source theorem."""

    return _run_official(source_path, binary_path, target_label)


def _replace_target_proof(
    source: str,
    target_label: str,
    proof_tokens: Sequence[str],
) -> str:
    declaration = re.search(
        rf"(?m)(?<!\S){re.escape(target_label)}\s+\$p(?=\s)",
        source,
    )
    if declaration is None:
        raise ValueError(f"target theorem declaration not found: {target_label}")
    proof_start = source.find("$=", declaration.end())
    if proof_start < 0:
        raise ValueError(f"target theorem has no $= proof: {target_label}")
    proof_end = source.find("$.", proof_start + 2)
    if proof_end < 0:
        raise ValueError(f"target theorem proof has no $. terminator: {target_label}")
    replacement = " " + " ".join(proof_tokens) + " "
    return source[: proof_start + 2] + replacement + source[proof_end:]


def _subst_from_key(key: tuple) -> dict[str, list[str]]:
    return {name: list(value) for name, value in key}


def _convert_valid_trace(
    result,
    generated: str,
    local_assumptions: Mapping[str, str],
    max_proof_tokens: int,
) -> Optional[tuple[str, ...]]:
    if result.status is not VerificationStatus.VALID:
        return None
    proof_segments: list[tuple[str, ...]] = [
        (label,) for label in local_assumptions
    ]
    for (label, _), step in zip(parse_proof(generated), result.steps):
        if step.status is not VerificationStatus.VALID:
            return None
        segment: list[str] = []
        for _, witness in step.syntax_witnesses:
            segment.extend(witness)
        for source_index in step.hypothesis_sources:
            if source_index >= len(proof_segments):
                return None
            segment.extend(proof_segments[source_index])
        segment.append(label)
        if len(segment) > max_proof_tokens:
            return None
        proof_segments.append(tuple(segment))
    if not proof_segments:
        return None
    final = proof_segments[-1]
    if len(final) > max_proof_tokens:
        return None
    return final


def _permissively_bind_missing(
    parts,
    subst: dict[str, list[str]],
    local_expressions: Sequence[Sequence[str]],
) -> None:
    budget = SearchBudget(MATCH_NODE_BUDGET)
    for _, template in parts.essential:
        for expression in local_expressions:
            loose = next(
                _iter_template_matches(
                    template,
                    expression,
                    set(parts.variables),
                    {},
                    budget,
                ),
                None,
            )
            if loose is None:
                continue
            for variable, value in loose.items():
                subst.setdefault(variable, value)
            break


def _fallback_floating_witness(
    target_frame,
    value: Sequence[str],
) -> Optional[tuple[str, ...]]:
    if len(value) != 1:
        return None
    for label, _, variable in target_frame.active_f:
        if variable == value[0]:
            return (label,)
    return None


def _convert_invalid_single_step(
    mm,
    target_label: str,
    generated: str,
    local_assumptions: Mapping[str, str],
) -> Optional[tuple[str, ...]]:
    steps = parse_proof(generated)
    if len(steps) != 1:
        return None
    label, expression_string = steps[0]
    parts = rule_parts(mm, label)
    target_frame = _get_frame(mm, target_label)
    if parts is None or target_frame is None:
        return None
    budget = SearchBudget(MATCH_NODE_BUDGET)
    local_items = [
        (local_label, expression.split())
        for local_label, expression in local_assumptions.items()
    ]
    checker = SyntaxTypeChecker(
        mm,
        target_label,
        target_frame,
        MATCH_NODE_BUDGET,
    )
    chosen_subst = None
    chosen_floating_tokens: list[str] = []
    for candidate in _iter_template_matches(
        parts.conclusion,
        expression_string.split(),
        set(parts.variables),
        {},
        budget,
    ):
        subst = dict(candidate)
        _permissively_bind_missing(
            parts,
            subst,
            [expression for _, expression in local_items],
        )
        floating_tokens: list[str] = []
        for _, typecode, variable in parts.floating:
            value = subst.get(variable)
            if value is None:
                break
            syntax = checker.check(typecode, value)
            if syntax.status is VerificationStatus.VALID:
                floating_tokens.extend(syntax.witness)
                continue
            fallback = _fallback_floating_witness(target_frame, value)
            if fallback is None:
                break
            floating_tokens.extend(fallback)
        else:
            chosen_subst = subst
            chosen_floating_tokens = floating_tokens
            break
    if chosen_subst is None:
        return None
    subst = chosen_subst
    proof_tokens = chosen_floating_tokens

    for essential_index, (_, template) in enumerate(parts.essential):
        expected = norm(apply_subst(template, subst))
        source = next(
            (
                local_label
                for local_label, expression in local_items
                if norm(expression) == expected
            ),
            None,
        )
        if source is None and essential_index < len(local_items):
            source = local_items[essential_index][0]
        if source is None:
            return None
        proof_tokens.append(source)
    proof_tokens.append(label)
    return tuple(proof_tokens)


def verify_expression_trace(
    *,
    source_path: Path,
    binary_path: Path,
    target_label: str,
    generated: str,
    goal: str,
    fact_block: Mapping[str, str],
    local_assumptions: Optional[Mapping[str, str]] = None,
    max_proof_tokens: int = DEFAULT_MAX_PROOF_TOKENS,
) -> OfficialResult:
    """Convert a supported trace and run the pinned official verifier.

    Unsupported or budget-gated conversion returns ``unknown``; it never returns a
    misleading boolean.
    """

    local = dict(local_assumptions or {})
    mm = MM().parse(source_path)
    internal = verify_proof(
        mm,
        generated,
        goal,
        dict(fact_block),
        local_assumptions=local,
        target_label=target_label,
    )
    proof_tokens = _convert_valid_trace(
        internal,
        generated,
        local,
        max_proof_tokens,
    )
    if proof_tokens is None and internal.status is VerificationStatus.INVALID:
        proof_tokens = _convert_invalid_single_step(
            mm,
            target_label,
            generated,
            local,
        )
    if proof_tokens is None:
        return OfficialResult(
            VerificationStatus.UNKNOWN,
            "trace_conversion_unsupported",
            f"internal status {internal.status.value}: {internal.reason}",
        )

    source = source_path.read_text(encoding="utf-8", errors="replace")
    try:
        converted = _replace_target_proof(source, target_label, proof_tokens)
    except ValueError as exc:
        return OfficialResult(
            VerificationStatus.UNKNOWN,
            "target_rewrite_unsupported",
            str(exc),
        )
    with tempfile.TemporaryDirectory(prefix="p3-metamath-oracle-") as temp_dir:
        converted_path = Path(temp_dir) / source_path.name
        converted_path.write_text(converted, encoding="utf-8")
        official = _run_official(
            converted_path,
            binary_path,
            target_label,
        )
    return OfficialResult(
        official.status,
        official.reason_code,
        official.reason,
        official.output,
        tuple(proof_tokens),
    )
