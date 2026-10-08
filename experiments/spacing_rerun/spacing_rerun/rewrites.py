"""Audited development-only natural declarative training representations."""
from __future__ import annotations

import collections
import re

from .common import digest, require
from .data import normalize


def source_fields(facts):
    return [{"id": f["id"], "event": f["event"], "source_statement": f["statement"],
             "canonical_answer": f["answer"]} for f in facts]


def duplicate_groups(facts, field="statement"):
    groups = collections.defaultdict(list)
    for fact in facts:
        groups[(fact["event"], normalize(fact[field]))].append(fact["id"])
    return [ids for ids in groups.values() if len(ids) > 1]


def apply_development_rewrites(facts, bundle, *, mode, partition_hash, roles):
    """Apply old/new statements only, preserving every evaluation and role field."""
    require(mode == "development" and bundle["scope"] == "development_only",
            "Development rewrites cannot authorize confirmation")
    require(bundle["schema"] == "spacing-directional-development-v1", "Unknown rewrite schema")
    require(digest({k: v for k, v in bundle.items() if k != "sha256"}) == bundle["sha256"],
            "Rewrite bundle integrity check failed")
    require(bundle["author_input_fields"] == ["source_statement", "canonical_answer"],
            "Rewrite authoring inputs must exclude evaluation questions")
    require(bundle["outer_partition_sha256"] == partition_hash and
            bundle["source_split_roles_sha256"] == digest(roles), "Rewrite partition/roles differ")
    require(bundle["source_fields_sha256"] == digest(source_fields(facts)), "Rewrite source fields differ")
    rows = {r["id"]: r for r in bundle["records"]}
    require(len(rows) == len(bundle["records"]) and set(rows) == {f["id"] for f in facts},
            "Every selected fact must occur exactly once in rewrite audit; no filtering")
    original_duplicates = duplicate_groups(facts)
    changed, applied, exceptions = collections.Counter(), collections.Counter(), []
    for fact in facts:
        row = rows[fact["id"]]
        require(row["event"] == fact["event"] and row["role"] == fact["role"], "Rewrite role/event mismatch")
        require(row["source_statement"] == fact["statement"] and row["canonical_answer"] == fact["answer"],
                "Rewrite source statement or answer changed")
        require(row["source_sha256"] == digest({"statement": fact["statement"], "answer": fact["answer"]}),
                "Rewrite source hash mismatch")
        text = row["training_statement"]
        require(text.strip() and "?" not in text and "Question:" not in text and "Answer:" not in text,
                "Training rewrite must remain a declarative statement")
        require(normalize(fact["question"]) not in normalize(text),
                "Complete evaluation question form entered training rewrite")
        matches = list(re.finditer(re.escape(fact["answer"]), text, re.IGNORECASE))
        terminal = len(matches) == 1 and not text[matches[0].end():].strip(" .!?\"'")
        if not terminal:
            require(row.get("answer_final_exception") and normalize(text) == normalize(fact["statement"]),
                    "Non-final answer requires an explicit audited unchanged-source exception")
            exceptions.append({"id": fact["id"], "reason": row["answer_final_exception"]})
        if fact["role"] in ("old", "new"):
            review = row.get("independent_review", {})
            require(review.get("status") == "source_checked_with_corrections" and review.get("reviewer"),
                    "Old/new rewrite requires independent source review")
            applied[fact["role"]] += 1
            changed[fact["role"]] += int(text != fact["statement"])
            fact["source_statement"] = fact["statement"]
            fact["statement"] = text
            fact["statement_rewrite_sha256"] = digest(row)
    return {"bundle_sha256": bundle["sha256"], "applied_roles": dict(applied),
            "changed_roles": dict(changed), "source_preserved_exceptions": exceptions,
            "source_exact_statement_duplicate_groups": original_duplicates,
            "training_exact_statement_duplicate_groups": duplicate_groups(facts),
            "confirmation_ready": False,
            "limitations": "Agent-reviewed development representation. Duplicate rehearsal units and independent human audit remain required before confirmation."}
