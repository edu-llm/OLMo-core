"""Development-only question-form teaching from separate, pinned source events."""
from __future__ import annotations

from .common import digest, require
from .data import normalize

SOURCE_QA_POLICY = "all_source_questions_v1"
CANONICAL_QA_POLICY = "canonical_roots_v1"


def build_teaching_pool(source_rows, facts, roles, mode, *, confirmation_audit=None):
    require(mode == "development" or (mode == "confirmation" and confirmation_audit is not None),
            "Expanded source QA teaching is development-only until audited and frozen")
    if mode == "confirmation":
        from .confirmation import validate_confirmation_audit, validate_roles
        context = validate_confirmation_audit(confirmation_audit)
        validate_roles(context, roles, context["outer_partition_sha256"])
    roots = {f["id"]: f for f in facts if f["role"] == "qa"}
    owners = {pid: root for root, fact in roots.items() for pid in fact["members"]}
    rows = [r for r in source_rows if roles.get(str(r["event_id"])) == "qa"]
    require(len({str(r["question_id"]) for r in rows}) == len(rows), "Duplicate source teaching question IDs")
    require(set(owners) == {str(r["question_id"]) for r in rows}, "Source teaching questions do not match QA root membership")
    records = []
    for row in sorted(rows, key=lambda r: str(r["question_id"])):
        pid = str(row["question_id"])
        record = {"id": pid, "event": str(row["event_id"]), "role": "qa", "canonical_probe_id": owners[pid],
                  "question": str(row["question"]).strip(), "answer": str(row["natural_answer"]).strip(),
                  "source_fiction_id": str(row["fiction_id"])}
        record["source_fields_sha256"] = digest(record)
        records.append(record)
    pool = {"policy": SOURCE_QA_POLICY, "records": records, "events": sorted({r["event"] for r in records}),
            "source_question_count": len(records), "canonical_probe_count": len(roots)}
    pool["sha256"] = digest(pool)
    validate_teaching_pool(pool, facts, roles)
    return pool


def validate_teaching_pool(pool, facts, roles):
    require(pool["policy"] == SOURCE_QA_POLICY, "Unknown expanded QA-teaching policy")
    require(digest({k: v for k, v in pool.items() if k != "sha256"}) == pool["sha256"], "Teaching pool was modified")
    roots = {f["id"]: f for f in facts if f["role"] == "qa"}
    owners = {pid: root for root, fact in roots.items() for pid in fact["members"]}
    heldout_questions = {normalize(f[key]) for f in facts if f["role"] in ("old", "new", "control")
                         for key in ("question", "paraphrase") if f.get(key)}
    records = pool["records"]
    require(len(records) == len({r["id"] for r in records}) and {r["id"] for r in records} == set(owners),
            "Lost or duplicated source teaching questions")
    for record in records:
        require(record["role"] == "qa" and roles.get(record["event"]) == "qa", "Source teaching event-role leakage")
        require(record["canonical_probe_id"] == owners[record["id"]] and
                roots[record["canonical_probe_id"]]["event"] == record["event"], "Teaching source/root mismatch")
        require(record["question"] and record["answer"], "Empty source teaching question/answer")
        require(normalize(record["question"]) not in heldout_questions, "Held-out evaluation question entered QA teaching")
        require(digest({k: v for k, v in record.items() if k != "source_fields_sha256"}) == record["source_fields_sha256"],
                "Source teaching question/answer fields changed")
    require(pool["source_question_count"] == len(records) and pool["canonical_probe_count"] == len(roots) and
            pool["events"] == sorted({r["event"] for r in records}), "Teaching pool counts changed")
