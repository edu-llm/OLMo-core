"""Machine A — the production Metamath shard.

Differs from build_metamath_sample.py by doing the whole job: all three databases,
the held-out split, and no example cap.

Five decisions from the plan are enforced here:
  * verify while expanding — the final `|-` entry must equal the theorem's own
    statement; anything that fails to reduce is a decoder bug and gets dropped
  * render facts WITH their $e hypotheses, since 57% of cited facts are inference
    rules whose conclusion alone says nothing
  * render only actually pushed theorem-local $e hypotheses as local assumptions,
    and omit those pushes plus internal `(reuse)` bookkeeping from the target
  * shuffle the block, so its order does not hand over the derivation sequence
  * hold out 500 normalized statement classes exposed by only one or two visible
    rows, routing every name, fact, goal, target, and local exposure together
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts")
from mm_expand import MM, expand

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local assumptions:"
SEP = "---"
PINNED_DBS = ("set", "iset", "nf")
DBS = PINNED_DBS
SOURCE_COMMIT = "82830c78861b96e906d9868c30c35dbd98be5db5"
PINNED_SOURCE_SHA256 = {
    "set": "7695d59e1c5c9182231e002425c82c86569bc044f30770bb32c276f7bafbf644",
    "iset": "2851ed617e011b08b4d61c8312f34183aaf4da6b06b19512dac1e397ce709e4f",
    "nf": "727a3707545e13ec53f03502eb07dc4635a8c176f275d4014a17fbd823e66083",
}
SOURCE_SHA256 = dict(PINNED_SOURCE_SHA256)
EXPECTED_CONFLICT_COUNT = 394
EXPECTED_CONFLICT_MAP_SHA256 = (
    "900adee09e42be5d7dda266a61e3095991825e9cf2612675e5589af932edc0c2"
)
FACT_IDENTITY_POLICY = "qualify-conflicting-statements-v1"
ROW_SCHEMA = "metamath-proof-v2"
BUILD_SOURCE_SCHEMA = "metamath-build-source-v3"
DROP_LEDGER_SCHEMA = "metamath-overlength-drop-ledger-v1"
DROP_ENTRY_SCHEMA = "metamath-overlength-drop-v1"
DROP_REASON_SCHEMA = "metamath-row-eligibility-reason-v1"
OVERLENGTH_DROP_REASON = "text_plus_eos_exceeds_maximum"
MAX_TEXT_PLUS_EOS_TOKENS = 16_384
DEFAULT_TOKENIZER_PATH = str(
    Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
)
FIXED_QWEN_TOKENIZER_SEAL = {
    "identity": "Qwen/Qwen2.5-0.5B",
    "tokenizer_json_sha256": (
        "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
    ),
    "tokenizer_config_sha256": (
        "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
    ),
    "behavior_digest": (
        "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
    ),
    "tokenizers_version": "0.22.2",
    "eos_token_id": 151643,
    "max_text_plus_eos_tokens": MAX_TEXT_PLUS_EOS_TOKENS,
}
EXPECTED_PINNED_OVERSIZED_ROWS = 960
EXPECTED_PINNED_OVERSIZED_TOKENS = 27_991_259
EXPECTED_PINNED_OVERSIZED_IDS_SHA256 = (
    "a1b4bbdc271c9facbbcbc22d0c93661a093b9e17b6baf9c86c76aa62f89a2040"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TARGET_STEP_RE = re.compile(r"^\s*\d+\s+(\S+)\s+(\|-\s.*)$")


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_fixed_qwen_tokenizer(tokenizer_path):
    """Load the exact local Qwen tokenizer and return its path-free seal."""
    try:
        from .build_isabelle_shard import (
            _tokenizer_metadata,
            load_vendored_tokenizer,
        )
    except ImportError:  # pragma: no cover - production CLI import mode.
        from build_isabelle_shard import (
            _tokenizer_metadata,
            load_vendored_tokenizer,
        )

    tokenizer = load_vendored_tokenizer(tokenizer_path)
    seal = _tokenizer_metadata(tokenizer)
    seal.pop("path", None)
    if seal != FIXED_QWEN_TOKENIZER_SEAL:
        raise ValueError("loaded tokenizer does not match the fixed Qwen seal")
    return tokenizer, seal


def count_text_plus_eos_tokens(tokenizer, text):
    """Count the exact final text encoding plus one explicit EOS token."""
    if not isinstance(text, str):
        raise TypeError("Metamath row text must be a string")
    try:
        encoding = tokenizer.encode(text, add_special_tokens=False)
    except Exception as error:
        raise ValueError("fixed Qwen tokenizer failed to encode Metamath text") from error
    ids = getattr(encoding, "ids", encoding)
    try:
        return len(ids) + 1
    except TypeError as error:
        raise ValueError("fixed Qwen tokenizer returned no token sequence") from error


def native_row_sha256(record):
    """Hash the complete source-native row before derived metadata is attached."""
    if not isinstance(record, dict):
        raise TypeError("Metamath native row must be an object")
    native_record = {
        key: value for key, value in record.items() if key != "source_metadata"
    }
    return canonical_sha256(native_record)


def _drop_ledger_body(entries, accounting, tokenizer_seal, max_tokens):
    return {
        "schema_version": DROP_LEDGER_SCHEMA,
        "reason_schema_version": DROP_REASON_SCHEMA,
        "ordering": "id-then-theorem-v1",
        "max_text_plus_eos_tokens": max_tokens,
        "tokenizer_seal": dict(tokenizer_seal),
        "tokenizer_root_sha256": canonical_sha256(tokenizer_seal),
        "entries": entries,
        "entries_root_sha256": canonical_sha256(entries),
        "accounting": accounting,
    }


def validate_drop_ledger(ledger):
    """Recompute every canonical overlength-ledger identity and total."""
    if not isinstance(ledger, dict):
        raise TypeError("drop ledger must be an object")
    if ledger.get("schema_version") != DROP_LEDGER_SCHEMA:
        raise ValueError("drop ledger schema is invalid")
    if ledger.get("reason_schema_version") != DROP_REASON_SCHEMA:
        raise ValueError("drop ledger reason schema is invalid")
    if ledger.get("ordering") != "id-then-theorem-v1":
        raise ValueError("drop ledger ordering is invalid")
    max_tokens = ledger.get("max_text_plus_eos_tokens")
    if max_tokens != MAX_TEXT_PLUS_EOS_TOKENS:
        raise ValueError("drop ledger maximum token count is invalid")
    if ledger.get("tokenizer_seal") != FIXED_QWEN_TOKENIZER_SEAL:
        raise ValueError("drop ledger tokenizer seal is invalid")
    if ledger.get("tokenizer_root_sha256") != canonical_sha256(
        FIXED_QWEN_TOKENIZER_SEAL
    ):
        raise ValueError("drop ledger tokenizer root is invalid")

    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise TypeError("drop ledger entries must be a list")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "schema_version",
            "id",
            "theorem",
            "text_plus_eos_tokens",
            "native_row_sha256",
            "reason_schema_version",
            "reason",
        }:
            raise ValueError("drop ledger entry schema is invalid")
    if entries != sorted(entries, key=lambda entry: (entry["id"], entry["theorem"])):
        raise ValueError("drop ledger entries are not canonically sorted")
    ids = set()
    theorems = set()
    for entry in entries:
        if (
            entry["schema_version"] != DROP_ENTRY_SCHEMA
            or entry["reason_schema_version"] != DROP_REASON_SCHEMA
            or entry["reason"] != OVERLENGTH_DROP_REASON
        ):
            raise ValueError("drop ledger entry reason schema is invalid")
        if not isinstance(entry["id"], str) or not entry["id"]:
            raise ValueError("drop ledger entry ID is invalid")
        if not isinstance(entry["theorem"], str) or not entry["theorem"]:
            raise ValueError("drop ledger entry theorem is invalid")
        token_count = entry["text_plus_eos_tokens"]
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count <= max_tokens
        ):
            raise ValueError("drop ledger entry token count is not overlength")
        if not isinstance(entry["native_row_sha256"], str) or not SHA256_RE.fullmatch(
            entry["native_row_sha256"]
        ):
            raise ValueError("drop ledger native row hash is invalid")
        if entry["id"] in ids or entry["theorem"] in theorems:
            raise ValueError("drop ledger entries contain duplicate identities")
        ids.add(entry["id"])
        theorems.add(entry["theorem"])

    entries_root = ledger.get("entries_root_sha256")
    if entries_root != canonical_sha256(entries):
        raise ValueError("drop ledger entries root is stale")
    accounting = ledger.get("accounting")
    expected_accounting_keys = {
        "source_rows",
        "eligible_rows",
        "dropped_rows",
        "source_text_plus_eos_tokens",
        "eligible_text_plus_eos_tokens",
        "dropped_text_plus_eos_tokens",
        "dropped_excess_tokens",
    }
    if not isinstance(accounting, dict) or set(accounting) != expected_accounting_keys:
        raise ValueError("drop ledger accounting schema is invalid")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in accounting.values()
    ):
        raise ValueError("drop ledger accounting values are invalid")
    dropped_tokens = sum(entry["text_plus_eos_tokens"] for entry in entries)
    dropped_excess = sum(
        entry["text_plus_eos_tokens"] - max_tokens for entry in entries
    )
    if (
        accounting["dropped_rows"] != len(entries)
        or accounting["dropped_text_plus_eos_tokens"] != dropped_tokens
        or accounting["dropped_excess_tokens"] != dropped_excess
        or accounting["source_rows"]
        != accounting["eligible_rows"] + accounting["dropped_rows"]
        or accounting["source_text_plus_eos_tokens"]
        != (
            accounting["eligible_text_plus_eos_tokens"]
            + accounting["dropped_text_plus_eos_tokens"]
        )
    ):
        raise ValueError("drop ledger accounting is stale")

    root = ledger.get("canonical_root_sha256")
    if not isinstance(root, str) or not SHA256_RE.fullmatch(root):
        raise ValueError("drop ledger canonical root is invalid")
    body = dict(ledger)
    del body["canonical_root_sha256"]
    if root != canonical_sha256(body):
        raise ValueError("drop ledger canonical root is stale")


def partition_records_by_text_plus_eos(
    records,
    *,
    tokenizer,
    tokenizer_seal,
    max_tokens=MAX_TEXT_PLUS_EOS_TOKENS,
):
    """Filter whole overlength rows before any exposure or heldout planning."""
    if tokenizer_seal != FIXED_QWEN_TOKENIZER_SEAL:
        raise ValueError("Metamath eligibility requires the fixed Qwen tokenizer seal")
    if max_tokens != FIXED_QWEN_TOKENIZER_SEAL["max_text_plus_eos_tokens"]:
        raise ValueError("Metamath eligibility token maximum is not approved")

    eligible = []
    dropped = []
    source_tokens = 0
    eligible_tokens = 0
    for record, statement_classes in records:
        if not isinstance(record, dict):
            raise TypeError("Metamath record must be an object")
        row_id = record.get("id")
        theorem = record.get("theorem")
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("Metamath record ID is missing")
        if not isinstance(theorem, str) or not theorem:
            raise ValueError("Metamath record theorem is missing")
        token_count = count_text_plus_eos_tokens(tokenizer, record.get("text"))
        source_tokens += token_count
        if token_count <= max_tokens:
            eligible.append((record, statement_classes, token_count))
            eligible_tokens += token_count
            continue
        dropped.append(
            {
                "schema_version": DROP_ENTRY_SCHEMA,
                "id": row_id,
                "theorem": theorem,
                "text_plus_eos_tokens": token_count,
                "native_row_sha256": native_row_sha256(record),
                "reason_schema_version": DROP_REASON_SCHEMA,
                "reason": OVERLENGTH_DROP_REASON,
            }
        )

    dropped.sort(key=lambda entry: (entry["id"], entry["theorem"]))
    dropped_tokens = sum(entry["text_plus_eos_tokens"] for entry in dropped)
    accounting = {
        "source_rows": len(eligible) + len(dropped),
        "eligible_rows": len(eligible),
        "dropped_rows": len(dropped),
        "source_text_plus_eos_tokens": source_tokens,
        "eligible_text_plus_eos_tokens": eligible_tokens,
        "dropped_text_plus_eos_tokens": dropped_tokens,
        "dropped_excess_tokens": sum(
            entry["text_plus_eos_tokens"] - max_tokens for entry in dropped
        ),
    }
    ledger = _drop_ledger_body(
        dropped,
        accounting,
        tokenizer_seal,
        max_tokens,
    )
    ledger["canonical_root_sha256"] = canonical_sha256(ledger)
    validate_drop_ledger(ledger)
    return eligible, ledger


def verify_pinned_oversized_population(manifest, drop_ledger):
    """Fail closed if the unchanged pinned source no longer reproduces its audit."""
    validate_drop_ledger(drop_ledger)
    expected_files = {
        f"{database}.mm": {"sha256": digest}
        for database, digest in PINNED_SOURCE_SHA256.items()
    }
    is_pinned_source = (
        manifest.get("repository") == "https://github.com/metamath/set.mm"
        and manifest.get("commit") == SOURCE_COMMIT
        and manifest.get("files") == expected_files
    )
    if not is_pinned_source:
        return None

    entries = drop_ledger["entries"]
    sorted_id_root = hashlib.sha256(
        "\n".join(entry["id"] for entry in entries).encode("utf-8")
    ).hexdigest()
    dropped_tokens = drop_ledger["accounting"]["dropped_text_plus_eos_tokens"]
    if (
        len(entries) != EXPECTED_PINNED_OVERSIZED_ROWS
        or dropped_tokens != EXPECTED_PINNED_OVERSIZED_TOKENS
        or sorted_id_root != EXPECTED_PINNED_OVERSIZED_IDS_SHA256
    ):
        raise RuntimeError(
            "pinned oversized population drift: "
            f"expected {EXPECTED_PINNED_OVERSIZED_ROWS} rows / "
            f"{EXPECTED_PINNED_OVERSIZED_TOKENS} tokens / "
            f"{EXPECTED_PINNED_OVERSIZED_IDS_SHA256}, got "
            f"{len(entries)} / {dropped_tokens} / {sorted_id_root}"
        )
    return {
        "schema_version": "metamath-pinned-overlength-reproduction-v1",
        "source_rows": len(entries),
        "text_plus_eos_tokens": dropped_tokens,
        "sorted_id_set_sha256": sorted_id_root,
    }


def build_source_metadata(
    manifest,
    conflict_map,
    *,
    drop_ledger,
    tokenizer_seal,
):
    validate_drop_ledger(drop_ledger)
    if tokenizer_seal != FIXED_QWEN_TOKENIZER_SEAL:
        raise ValueError("Metamath source metadata tokenizer seal is invalid")
    source_roots = {
        name: {"sha256": record["sha256"]}
        for name, record in sorted(manifest["files"].items())
    }
    conflict_map_sha256 = canonical_sha256(conflict_map)
    drop_binding = {
        "schema_version": drop_ledger["schema_version"],
        "canonical_root_sha256": drop_ledger["canonical_root_sha256"],
        "entries_root_sha256": drop_ledger["entries_root_sha256"],
        "accounting": dict(drop_ledger["accounting"]),
    }
    tokenizer_root_sha256 = canonical_sha256(tokenizer_seal)
    quality_filter = {
        "final_expression_must_match": True,
        "local_assumptions": "decoded-used-essential-hypotheses-v2",
        "reuse_visibility": "source-replay-only-v2",
        "fact_identity_policy": FACT_IDENTITY_POLICY,
        "conflict_count": len(conflict_map),
        "conflict_map_sha256": conflict_map_sha256,
        "row_eligibility": "fixed-qwen-text-plus-eos-v1",
        "max_text_plus_eos_tokens": MAX_TEXT_PLUS_EOS_TOKENS,
        "tokenizer_root_sha256": tokenizer_root_sha256,
        "drop_ledger_root_sha256": drop_ledger["canonical_root_sha256"],
        "drop_entries_root_sha256": drop_ledger["entries_root_sha256"],
    }
    schema_generation = {
        "schema_version": ROW_SCHEMA,
        "required_local_assumptions": True,
        "fact_identity_policy": FACT_IDENTITY_POLICY,
        "conflict_count": len(conflict_map),
        "conflict_map_sha256": conflict_map_sha256,
        "drop_ledger_schema_version": drop_ledger["schema_version"],
        "drop_entry_schema_version": DROP_ENTRY_SCHEMA,
        "drop_reason_schema_version": DROP_REASON_SCHEMA,
        "drop_ledger_root_sha256": drop_ledger["canonical_root_sha256"],
    }
    return {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "source_manifest_root_sha256": canonical_sha256(manifest),
        "source_roots": source_roots,
        "index_roots": {},
        "quality_filter": quality_filter,
        "quality_filter_root_sha256": canonical_sha256(quality_filter),
        "schema_generation": schema_generation,
        "schema_generation_root_sha256": canonical_sha256(schema_generation),
        "tokenizer_seal": dict(tokenizer_seal),
        "tokenizer_root_sha256": tokenizer_root_sha256,
        "drop_ledger": drop_binding,
        "fact_identity": {
            "policy": FACT_IDENTITY_POLICY,
            "conflict_count": len(conflict_map),
            "conflict_map_sha256": conflict_map_sha256,
        },
    }


def render_fact(data):
    concl = " ".join(data[0])
    hyps = [" ".join(h[2]) for h in (data[1] if len(data) > 1 else []) if h[0] == "$e"]
    return f"{' & '.join(hyps)} => {concl}" if hyps else concl


def _database_order(statements_by_database):
    return [
        *[database for database in DBS if database in statements_by_database],
        *sorted(set(statements_by_database) - set(DBS)),
    ]


def compute_conflict_map(statements_by_database):
    """Return labels whose rendered assertions differ between pinned databases."""
    labels = sorted(
        set().union(
            *(set(statements) for statements in statements_by_database.values())
        )
    )
    conflicts = {}
    database_order = _database_order(statements_by_database)
    for label in labels:
        statements = {
            database: statements_by_database[database][label]
            for database in database_order
            if label in statements_by_database[database]
        }
        if len(set(statements.values())) > 1:
            conflicts[label] = statements
    return conflicts


def fact_identity(database, label, conflict_map):
    """Render a database-qualified identity only for a conflicting label."""
    return f"{database}:{label}" if label in conflict_map else label


def rendered_fact_statements(statements_by_database, conflict_map):
    """Flatten per-database statements without losing conflicting assertions."""
    rendered = {}
    for database in _database_order(statements_by_database):
        for label in sorted(statements_by_database[database]):
            identity = fact_identity(database, label, conflict_map)
            statement = statements_by_database[database][label]
            if identity in rendered and rendered[identity] != statement:
                raise ValueError(
                    f"rendered identity {identity!r} has multiple statements"
                )
            rendered[identity] = statement
    return rendered


def statement_aliases_for(identities, identity_statements):
    """Return other rendered identities with an exactly held statement."""
    missing = set(identities) - set(identity_statements)
    if missing:
        raise ValueError(f"unknown fact identities: {sorted(missing)}")
    held_statements = {identity_statements[identity] for identity in identities}
    return {
        identity
        for identity, statement in identity_statements.items()
        if identity not in identities and statement in held_statements
    }


def _normalize_statement(statement):
    return " ".join(statement.split())


def normalized_statements(statements):
    """Normalize a statement set once for repeated row scans."""
    return frozenset(_normalize_statement(statement) for statement in statements)


def _target_steps(target):
    steps = []
    for line in target.splitlines():
        match = TARGET_STEP_RE.match(line)
        if match is None:
            raise ValueError(f"malformed Metamath target line: {line!r}")
        steps.append((match.group(1), _normalize_statement(match.group(2))))
    return tuple(steps)


def statement_identities_by_class(identity_statements):
    """Group global fact identities by whitespace-normalized full statement."""
    grouped = {}
    for identity in sorted(identity_statements):
        statement_class = _normalize_statement(identity_statements[identity])
        grouped.setdefault(statement_class, []).append(identity)
    return {
        statement_class: tuple(identities)
        for statement_class, identities in sorted(grouped.items())
    }


def visible_statement_classes(record, identity_statements, *, theorem_identity=None):
    """Return every exact statement class visible in one rendered row."""
    target_steps = _target_steps(record.get("target", ""))
    named_identities = set(record.get("cited", ())) | set(record.get("facts", {}))
    named_identities.update(label for label, _ in target_steps)
    if theorem_identity is not None:
        named_identities.add(theorem_identity)

    statement_classes = {
        _normalize_statement(identity_statements[identity])
        for identity in named_identities
        if identity in identity_statements
    }
    visible_statements = list(record.get("facts", {}).values())
    goal = record.get("goal")
    if isinstance(goal, str):
        visible_statements.append(goal)
    visible_statements.extend(expression for _, expression in target_steps)
    local_assumptions = record.get("local_assumptions", {})
    if isinstance(local_assumptions, dict):
        visible_statements.extend(local_assumptions.values())
    statement_classes.update(
        _normalize_statement(statement) for statement in visible_statements
    )
    return frozenset(statement_classes)


def statement_class_exposure_counts(row_statement_classes):
    """Count each statement class at most once per visible row."""
    counts = Counter()
    for statement_classes in row_statement_classes:
        counts.update(set(statement_classes))
    return counts


def classifier_compatible_representative_identities(
    identities_by_class,
    named_fact_identities,
    name_only_theorem_names_by_class,
):
    """Return identities that preserve every route under heldout-v2 classification."""
    named_fact_identities = set(named_fact_identities)
    compatible = set()
    for statement_class, identities in identities_by_class.items():
        named_candidates = set(identities) & named_fact_identities
        required_theorem_names = set(
            name_only_theorem_names_by_class.get(statement_class, ())
        )
        if not required_theorem_names:
            compatible.update(named_candidates)
        elif len(required_theorem_names) == 1:
            required_name = next(iter(required_theorem_names))
            if required_name in named_candidates:
                compatible.add(required_name)
    return frozenset(compatible)


def select_heldout_statement_classes(
    exposure_counts,
    identities_by_class,
    *,
    named_fact_identities,
    requested,
    seed,
):
    """Select exact tail classes and deterministic classifier-compatible names."""
    if requested < 0:
        raise ValueError("requested heldout class count must be nonnegative")

    normalized_counts = Counter()
    for statement_class, count in exposure_counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(
                "statement-class exposure counts must be nonnegative integers"
            )
        normalized_counts[_normalize_statement(statement_class)] += count

    normalized_identities = {}
    for statement_class, identities in identities_by_class.items():
        key = _normalize_statement(statement_class)
        normalized_identities.setdefault(key, set()).update(identities)

    named_fact_identities = set(named_fact_identities)
    named_by_class = {
        statement_class: tuple(
            sorted(
                normalized_identities.get(statement_class, set())
                & named_fact_identities
            )
        )
        for statement_class in normalized_counts
    }
    tail_classes = tuple(
        sorted(
            statement_class
            for statement_class, count in normalized_counts.items()
            if count in (1, 2) and named_by_class.get(statement_class)
        )
    )
    if requested > len(tail_classes):
        raise ValueError(
            f"requested {requested} heldout statement classes, "
            f"but only {len(tail_classes)} are eligible"
        )

    selected_classes = tuple(
        sorted(random.Random(seed).sample(tail_classes, requested))
    )
    representative_by_class = {
        statement_class: named_by_class[statement_class][0]
        for statement_class in selected_classes
    }
    representatives = tuple(sorted(representative_by_class.values()))
    statement_aliases = tuple(
        sorted(
            identity
            for statement_class in selected_classes
            for identity in normalized_identities[statement_class]
            if identity != representative_by_class[statement_class]
        )
    )
    return {
        "tail_classes": tail_classes,
        "selected_classes": selected_classes,
        "representatives": representatives,
        "statement_aliases": statement_aliases,
    }


def statement_alias_exposure(record, normalized_held_statements):
    """Find exact held statements exposed in rendered statement-valued fields."""
    fact_identities = tuple(
        sorted(
            identity
            for identity, statement in record.get("facts", {}).items()
            if _normalize_statement(statement) in normalized_held_statements
        )
    )
    goal = record.get("goal")
    goal_expressions = (
        (_normalize_statement(goal),)
        if isinstance(goal, str)
        and _normalize_statement(goal) in normalized_held_statements
        else ()
    )
    local_assumption_values = tuple(
        sorted(
            {
                _normalize_statement(statement)
                for statement in record.get("local_assumptions", {}).values()
                if _normalize_statement(statement) in normalized_held_statements
            }
        )
    )
    target_expressions = {
        expression
        for _, expression in _target_steps(record.get("target", ""))
        if expression in normalized_held_statements
    }
    return {
        "fact_identities": fact_identities,
        "goal_expressions": goal_expressions,
        "local_assumption_values": local_assumption_values,
        "target_expressions": tuple(sorted(target_expressions)),
    }


def render_trace_label(database, label, *, assertion_labels, conflict_map):
    """Rewrite assertion labels while leaving local and replay labels native."""
    if label not in assertion_labels:
        return label
    return fact_identity(database, label, conflict_map)


def split_model_trace(mand, trace):
    """Return used theorem-local $e givens and model-visible theorem applications."""
    mandatory_e = {label for kind, label, _ in mand if kind == "$e"}
    local_assumptions = {}
    steps = []
    for label, expr, _ in trace:
        if label in mandatory_e:
            local_assumptions.setdefault(label, " ".join(expr))
            continue
        if label == "(reuse)":
            continue
        if expr and expr[0] == "|-":
            steps.append((label, " ".join(expr)))
    return local_assumptions, steps


def source_manifest(mm_dir):
    files = {}
    for db in DBS:
        path = os.path.join(mm_dir, f"{db}.mm")
        if not os.path.exists(path):
            raise SystemExit(f"required pinned source is missing: {path}")
        with open(path, "rb") as source_file:
            digest = hashlib.sha256(source_file.read()).hexdigest()
        if digest != SOURCE_SHA256[db]:
            raise SystemExit(
                f"{db}.mm is not pinned commit {SOURCE_COMMIT}: "
                f"expected {SOURCE_SHA256[db]}, got {digest}"
            )
        files[f"{db}.mm"] = {"sha256": digest}
    return {
        "repository": "https://github.com/metamath/set.mm",
        "commit": SOURCE_COMMIT,
        "files": files,
    }


def load_pinned_databases(mm_dir):
    """Parse pinned databases and enforce the approved statement-conflict map."""
    source_manifest(mm_dir)
    databases = {}
    statements_by_database = {}
    for database in DBS:
        path = os.path.join(mm_dir, f"{database}.mm")
        mm = MM().parse(path)
        databases[database] = mm
        statements_by_database[database] = {
            label: render_fact(data)
            for label, (kind, data) in mm.labels.items()
            if kind in ("$a", "$p") and data and data[0] and data[0][0] == "|-"
        }

    conflict_map = compute_conflict_map(statements_by_database)
    if tuple(DBS) == PINNED_DBS:
        actual_count = len(conflict_map)
        actual_sha256 = canonical_sha256(conflict_map)
        if (
            actual_count != EXPECTED_CONFLICT_COUNT
            or actual_sha256 != EXPECTED_CONFLICT_MAP_SHA256
        ):
            raise SystemExit(
                "pinned Metamath conflict map drift: "
                f"expected {EXPECTED_CONFLICT_COUNT} conflicts / "
                f"{EXPECTED_CONFLICT_MAP_SHA256}, got "
                f"{actual_count} / {actual_sha256}"
            )
    return databases, statements_by_database, conflict_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm-dir", default="/tmp/dscount/mm")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER_PATH)
    a = ap.parse_args()
    if os.path.lexists(a.out):
        raise SystemExit(f"refusing to overwrite existing output root: {a.out}")

    databases, statements_by_database, conflict_map = load_pinned_databases(a.mm_dir)
    identity_statements = rendered_fact_statements(
        statements_by_database,
        conflict_map,
    )
    manifest = source_manifest(a.mm_dir)
    tokenizer, tokenizer_seal = load_fixed_qwen_tokenizer(a.tokenizer)

    # ---- pass 1: expand every proof and verify source replay ----
    proofs = []
    n_fail_reduce = 0
    for db in DBS:
        mm = databases[db]
        logical = set(statements_by_database[db])
        n = 0
        for lbl, (k, _) in mm.labels.items():
            if k != "$p":
                continue
            try:
                expr, mand, refs, trace = expand(mm, lbl)
            except Exception:  # noqa: BLE001, S112 - malformed source proof is dropped.
                continue
            logical_trace = [
                (sl, " ".join(e)) for (sl, e, _) in trace if e and e[0] == "|-"
            ]
            if not logical_trace:
                continue
            if logical_trace[-1][1] != " ".join(expr):  # the verification gate
                n_fail_reduce += 1
                continue
            local_assumptions, native_steps = split_model_trace(mand, trace)
            if not native_steps:
                continue
            native_used = [r for r in dict.fromkeys(refs) if r in logical]
            if not native_used:
                continue
            used = [
                fact_identity(db, reference, conflict_map) for reference in native_used
            ]
            steps = [
                (
                    render_trace_label(
                        db,
                        step_label,
                        assertion_labels=logical,
                        conflict_map=conflict_map,
                    ),
                    expression,
                )
                for step_label, expression in native_steps
            ]
            proofs.append(
                {
                    "database": db,
                    "theorem": f"{db}:{lbl}",
                    "theorem_identity": fact_identity(db, lbl, conflict_map),
                    "goal": " ".join(expr),
                    "local_assumptions": local_assumptions,
                    "steps": steps,
                    "native_used": native_used,
                    "used": used,
                }
            )
            n += 1
        print(f"  {db}.mm: {n:,} verified proofs")

    print(f"  failed to reduce: {n_fail_reduce:,}")
    print(f"  statement-conflicting labels: {len(conflict_map):,}")
    print(f"  rendered fact identities: {len(identity_statements):,}")

    # ---- prepare exact visible rows before counting statement classes ----
    records = []
    name_only_theorem_by_row_id = {}
    inc = dup = 0
    seen = set()
    for proof in proofs:
        db = proof["database"]
        lbl = proof["theorem"]
        goal = proof["goal"]
        local_assumptions = proof["local_assumptions"]
        steps = proof["steps"]
        native_used = proof["native_used"]
        used = proof["used"]
        if not all(
            reference in statements_by_database[db] for reference in native_used
        ):
            inc += 1
            continue
        eid = hashlib.md5(lbl.encode()).hexdigest()[:12]
        order = list(zip(native_used, used))
        random.Random(eid).shuffle(order)
        blk = {
            identity: statements_by_database[db][native_label]
            for native_label, identity in order
        }
        target = "\n".join(f"{i+1:>3}  {sl:<14} {e}" for i, (sl, e) in enumerate(steps))
        local_block = LOCAL_HDR
        if local_assumptions:
            local_block += "\n" + "\n".join(
                f"{name} : {statement}" for name, statement in local_assumptions.items()
            )
        block = (
            HDR
            + "\n"
            + "\n".join(f"{name} : {statement}" for name, statement in blk.items())
            + "\n"
            + local_block
        )
        text = f"{block}\n{SEP}\nGOAL {goal}\n{target}"
        if text in seen:
            dup += 1
            continue
        seen.add(text)
        record = {
            "schema_version": ROW_SCHEMA,
            "id": eid,
            "theorem": lbl,
            "facts": blk,
            "cited": used,
            "local_assumptions": local_assumptions,
            "goal": goal,
            "target": target,
            "text": text,
            "mask_start": 0,
            "mask_end": len(block),
        }
        non_theorem_classes = visible_statement_classes(record, identity_statements)
        theorem_class = _normalize_statement(
            identity_statements[proof["theorem_identity"]]
        )
        statement_classes = non_theorem_classes | {theorem_class}
        if theorem_class not in non_theorem_classes:
            native_theorem_name = lbl.split(":", 1)[1]
            name_only_theorem_by_row_id[eid] = (
                theorem_class,
                native_theorem_name,
            )
        records.append((record, statement_classes))
    del proofs

    # Whole-row Qwen eligibility is decided on the exact final rendered text.
    # Nothing derived from an overlength row may influence heldout planning.
    records, drop_ledger = partition_records_by_text_plus_eos(
        records,
        tokenizer=tokenizer,
        tokenizer_seal=tokenizer_seal,
    )
    pinned_overlength_reproduction = verify_pinned_oversized_population(
        manifest,
        drop_ledger,
    )
    source_metadata = build_source_metadata(
        manifest,
        conflict_map,
        drop_ledger=drop_ledger,
        tokenizer_seal=tokenizer_seal,
    )
    name_only_theorem_names_by_class = {}
    for record, _, _ in records:
        name_only = name_only_theorem_by_row_id.get(record["id"])
        if name_only is not None:
            theorem_class, native_theorem_name = name_only
            name_only_theorem_names_by_class.setdefault(theorem_class, set()).add(
                native_theorem_name
            )
        record["source_metadata"] = source_metadata

    exposure_counts = statement_class_exposure_counts(
        statement_classes for _, statement_classes, _ in records
    )
    identities_by_class = statement_identities_by_class(identity_statements)
    named_fact_identities = {
        identity
        for record, _, _ in records
        for identity in record["facts"]
        if identity in identity_statements
    }
    compatible_representatives = classifier_compatible_representative_identities(
        identities_by_class,
        named_fact_identities,
        name_only_theorem_names_by_class,
    )
    selection = select_heldout_statement_classes(
        exposure_counts,
        identities_by_class,
        named_fact_identities=compatible_representatives,
        requested=a.heldout,
        seed=a.seed,
    )
    held_classes = frozenset(selection["selected_classes"])
    held = set(selection["representatives"])
    statement_aliases = set(selection["statement_aliases"])

    # ---- pass 2: validate routes before writing any artifact bytes ----
    train_records = []
    eval_records = []
    for record, statement_classes, token_count in records:
        if token_count > MAX_TEXT_PLUS_EOS_TOKENS:
            raise RuntimeError(f"overlength row survived eligibility: {record['id']}")
        routed = (record, statement_classes, token_count)
        if statement_classes & held_classes:
            eval_records.append(routed)
        else:
            train_records.append(routed)

    train_leaks = [
        record["id"]
        for record, statement_classes, _ in train_records
        if statement_classes & held_classes
    ]
    if train_leaks:
        raise RuntimeError(f"held statement class leaked into train: {train_leaks[:3]}")
    represented_eval_classes = set()
    for _, statement_classes, _ in eval_records:
        represented_eval_classes.update(statement_classes & held_classes)
    missing_eval_classes = held_classes - represented_eval_classes
    if missing_eval_classes:
        raise RuntimeError(
            f"{len(missing_eval_classes)} held statement classes lack an eval row"
        )
    if len(held_classes) != a.heldout:
        raise RuntimeError(
            f"selected {len(held_classes)} held classes instead of {a.heldout}"
        )

    train_tokens = sum(token_count for _, _, token_count in train_records)
    eval_tokens = sum(token_count for _, _, token_count in eval_records)
    if (
        train_tokens + eval_tokens
        != drop_ledger["accounting"]["eligible_text_plus_eos_tokens"]
    ):
        raise RuntimeError("eligible Metamath token accounting is stale")
    drop_path = "drops/metamath-overlength.json"
    eligibility = {
        "schema_version": "metamath-text-plus-eos-eligibility-v1",
        "policy": "drop-whole-row-before-heldout-selection-v1",
        "applied_before": [
            "statement_class_exposure_counts",
            "classifier_compatibility",
            "tail_class_selection",
        ],
        "max_text_plus_eos_tokens": MAX_TEXT_PLUS_EOS_TOKENS,
        "tokenizer_seal": dict(tokenizer_seal),
        "tokenizer_root_sha256": canonical_sha256(tokenizer_seal),
        "drop_ledger": {
            "path": drop_path,
            "schema_version": drop_ledger["schema_version"],
            "canonical_root_sha256": drop_ledger["canonical_root_sha256"],
            "entries_root_sha256": drop_ledger["entries_root_sha256"],
        },
        "accounting": dict(drop_ledger["accounting"]),
    }
    partition_accounting = {
        "source_rows": drop_ledger["accounting"]["source_rows"],
        "source_text_plus_eos_tokens": drop_ledger["accounting"][
            "source_text_plus_eos_tokens"
        ],
        "train_rows": len(train_records),
        "train_text_plus_eos_tokens": train_tokens,
        "eval_rows": len(eval_records),
        "eval_text_plus_eos_tokens": eval_tokens,
        "drop_rows": drop_ledger["accounting"]["dropped_rows"],
        "drop_text_plus_eos_tokens": drop_ledger["accounting"][
            "dropped_text_plus_eos_tokens"
        ],
    }
    heldout_manifest = {
        "schema_version": "metamath-heldout-v2",
        "family": "metamath",
        "mode": "family_local_heldout",
        "facts": sorted(held),
        "statement_aliases": sorted(statement_aliases),
        "selected_statement_classes": sorted(held_classes),
        "selected_statement_classes_root_sha256": canonical_sha256(
            sorted(held_classes)
        ),
        "represented_eval_classes_root_sha256": canonical_sha256(
            sorted(represented_eval_classes)
        ),
        "requested_heldout": a.heldout,
        "actual_heldout": len(held_classes),
        "eligible_tail_classes": len(selection["tail_classes"]),
        "local_assumptions": True,
        "seed": a.seed,
        "corpus": "metamath",
        "fact_identity_policy": FACT_IDENTITY_POLICY,
        "conflict_count": len(conflict_map),
        "conflict_map_sha256": canonical_sha256(conflict_map),
        "eligibility": eligibility,
        "partition_accounting": partition_accounting,
        "source_quality_root_sha256": source_metadata[
            "quality_filter_root_sha256"
        ],
        "source_schema_root_sha256": source_metadata[
            "schema_generation_root_sha256"
        ],
        "policy": (
            f"{a.heldout} normalized full-statement classes with total visible "
            "eligible-row exposure 1-2; every name, fact, goal, target, local, "
            "alias, and own-proof exposure routed together using a "
            "metamath-heldout-v2 classifier-compatible representative"
        ),
    }
    if pinned_overlength_reproduction is not None:
        heldout_manifest["pinned_overlength_reproduction"] = (
            pinned_overlength_reproduction
        )
    heldout_manifest["manifest_root_sha256"] = canonical_sha256(heldout_manifest)

    # ---- commit the fresh standalone artifact ----
    for directory in ("shards", "eval", "heldout", "drops"):
        os.makedirs(os.path.join(a.out, directory), exist_ok=True)
    with open(os.path.join(a.out, "metamath_sources.json"), "w") as source_file:
        json.dump(manifest, source_file, indent=2)
    with open(os.path.join(a.out, drop_path), "w") as drop_file:
        drop_file.write(
            json.dumps(
                drop_ledger,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    with open(os.path.join(a.out, "heldout", "metamath.json"), "w") as heldout_file:
        json.dump(
            heldout_manifest,
            heldout_file,
            ensure_ascii=False,
            sort_keys=True,
            indent=1,
        )
        heldout_file.write("\n")

    tb = 0
    sp = os.path.join(a.out, "shards", "metamath.jsonl")
    epath = os.path.join(a.out, "eval", "metamath.jsonl")
    with open(sp, "w") as fh, open(epath, "w") as evf:
        for record, _, _ in train_records:
            fh.write(json.dumps(record) + "\n")
            tb += len(record["text"].encode())
        for record, _, _ in eval_records:
            evf.write(json.dumps(record) + "\n")

    print(
        f"  held out {len(held_classes):,} of "
        f"{len(selection['tail_classes']):,} eligible statement classes exposed "
        f"by 1-2 rows using {len(held):,} representatives plus "
        f"{len(statement_aliases):,} aliases"
    )
    print(
        f"\n  train {len(train_records):,}   eval {len(eval_records):,}   "
        f"overlength {drop_ledger['accounting']['dropped_rows']:,}   "
        f"dropped before rows: incomplete {inc:,}, duplicate {dup:,}"
    )
    print(
        f"  text+EOS tokens: train {train_tokens:,}, eval {eval_tokens:,}, "
        f"overlength {drop_ledger['accounting']['dropped_text_plus_eos_tokens']:,}"
    )
    print(f"  drop ledger root {drop_ledger['canonical_root_sha256']}")
    print(f"  heldout manifest root {heldout_manifest['manifest_root_sha256']}")
    print(f"  {tb/1e6:.1f} MB text  ~{tb/2.2/1e6:.0f}M GPT-2 tokens")
    print(f"  wrote {sp}")


if __name__ == "__main__":
    main()
