"""Compile the reviewed confirmation catalog. Run on FarmShare: python -I -B compile.py ROOT"""

import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
out = root / (sys.argv[2] if len(sys.argv) > 2 else "compiled")
out.mkdir(exist_ok=False)
validation = json.load(open(root / "validation-report.json"))


def seal(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return body


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decisions(channel, event):
    """Final decision per id: round 2 overrides round 1."""
    final = {}
    for rnd in (1, 2, 3):
        path = root / f"reviews/{channel}/{event}.r{rnd}.json"
        if path.exists():
            for d in json.load(open(path)).get("decisions", []):
                final[d["id"]] = {
                    "decision": d["decision"],
                    "round": rnd,
                    "reason": d.get("reason", ""),
                    **({"disclose": d["disclose"]} if d.get("disclose") else {}),
                }
    return final


def status_of(rid, final, mechanical_ids):
    d = final.get(rid)
    if d is None:
        return "UNREVIEWED", None
    if rid in mechanical_ids or d["decision"] != "APPROVE":
        return "UNRESOLVED", d
    return "APPROVED", d


packet_events, factsheets = {}, {}
for n in range(1, 9):
    ap = json.load(open(root / f"packets/author-packets/chunk-{n:02d}.json"))
    for e in ap["events"]:
        packet_events[e] = n
        factsheets[e] = ap["factsheets"][e]

units, groups, related, source_qa, renumbered, unresolved, unreviewed = [], [], [], [], 0, [], []


def fix_evidence(evidence, lines):
    global renumbered
    stripped = [line.strip() for line in lines]
    fixed = []
    for ev in evidence:
        quote, ln = ev.get("quote", "").strip(), ev.get("line")
        if not (isinstance(ln, int) and 1 <= ln <= len(lines) and stripped[ln - 1] == quote) and quote in stripped:
            ln = stripped.index(quote) + 1
            renumbered += 1
        fixed.append({"line": ln, "quote": ev.get("quote", "")})
    return fixed


for event in sorted(packet_events):
    doc = json.load(open(root / f"authored/A/{event}.json"))
    lines = factsheets[event]["text"].split("\n")
    final = decisions("A", event)
    mechanical = {i.split(":")[0] for i in validation["A"].get(event, [])}
    for kind, items, sink in (
        ("unit", doc["source_units"], units),
        ("group", doc["groups"], groups),
        ("qa", doc["source_qa"], source_qa),
    ):
        for rec in items:
            rid = rec.get("unit_id") if kind == "unit" else rec.get("id")
            status, d = status_of(rid, final, mechanical)
            rec = dict(rec)
            if "evidence" in rec:
                rec["evidence"] = fix_evidence(rec["evidence"], lines)
            rec["event"] = event
            rec["review_status"] = status
            if d:
                rec["review"] = d
            if status == "UNRESOLVED":
                unresolved.append(
                    {"channel": "A", "kind": kind, "id": rid, "event": event, "reason": d["reason"] if d else ""}
                )
            if status == "UNREVIEWED":
                unreviewed.append({"channel": "A", "kind": kind, "id": rid, "event": event})
            sink.append(rec)
    for r in doc.get("related_assertions_not_merged", []):
        related.append({"event": event, **r})

paraphrases = []
for event in sorted(packet_events):
    doc = json.load(open(root / f"authored/B/{event}.json"))
    final = decisions("B", event)
    mechanical = {i.split(":")[0] for i in validation["B"].get(event, [])}
    for rec in doc["paraphrases"]:
        status, d = status_of(rec["opaque_probe_id"], final, mechanical)
        rec = {**rec, "event": event, "review_status": status, **({"review": d} if d else {})}
        if status == "UNRESOLVED":
            unresolved.append(
                {
                    "channel": "B",
                    "kind": "paraphrase",
                    "id": rec["opaque_probe_id"],
                    "event": event,
                    "reason": d["reason"],
                }
            )
        if status == "UNREVIEWED":
            unreviewed.append({"channel": "B", "kind": "paraphrase", "id": rec["opaque_probe_id"], "event": event})
        if d and d.get("disclose"):
            unresolved.append(
                {
                    "channel": "B",
                    "kind": "disclosure",
                    "id": rec["opaque_probe_id"],
                    "event": event,
                    "reason": d["disclose"],
                }
            )
        paraphrases.append(rec)

ledger = json.load(open(root / "authored/entity-ledger.json"))
ledger_final = {}
for rnd in (1, 2, 3):
    p = root / f"reviews/entity-ledger.r{rnd}.json"
    if p.exists():
        for d in json.load(open(p))["decisions"]:
            ledger_final[d["id"]] = {"decision": d["decision"], "round": rnd}
for key, id_field in (("entries", "candidate_id"), ("uncovered_recurrences", "id")):
    for e in ledger.get(key, []):
        d = ledger_final.get(e[id_field])
        e["review_status"] = (
            "APPROVED" if d and d["decision"] == "APPROVE" else ("UNREVIEWED" if d is None else "UNRESOLVED")
        )

common = {
    "scope": "confirmation_80_events",
    "dataset_revision": "131cb74fdc3e601b5e896ed768ad9852ea35a8f9",
    "outer_seed": 20261007,
    "human_review_complete": False,
    "independent_agent_review_complete": True,
    "author": "claude-haiku",
    "independent_agent_reviewer": "claude-sonnet",
    "elections_amendment_sha256": file_sha(root / "ELECTIONS-AMENDMENT.md"),
    "authoring_spec_sha256": file_sha(root / "AUTHORING-SPEC.md"),
    **(
        {"round3_amendment_sha256": file_sha(root / "ROUND3-AMENDMENT.md")}
        if (root / "ROUND3-AMENDMENT.md").exists()
        else {}
    ),
}
artifacts = {
    "confirmation-grounded-declarations.json": {
        "schema": "spacing-grounded-declarations-v1",
        "policy": "grounded_assertion_v1",
        "evaluation_question_text_included": False,
        "model_outcomes_used": False,
        "source_units": units,
        "groups": groups,
        "related_assertions_not_merged": related,
        "evidence_lines_renumbered_to_1_indexed": renumbered,
    },
    "confirmation-acquisition-qa.json": {
        "schema": "spacing-acquisition-qa-v1",
        "evaluation_question_text_included": False,
        "model_outcomes_used": False,
        "records": source_qa,
    },
    "confirmation-evaluation-paraphrases.json": {
        "schema": "spacing-eval-paraphrases-v1",
        "may_see": ["canonical_evaluation_question"],
        "paraphrases": paraphrases,
    },
    "confirmation-entity-ledger.json": ledger,
}
summary = {}
for name, body in artifacts.items():
    sealed = seal({**common, **body})
    (out / name).write_text(json.dumps(sealed, indent=1, ensure_ascii=False))
    summary[name] = sealed["sha256"]


def count(recs):
    return {s: sum(1 for r in recs if r["review_status"] == s) for s in ("APPROVED", "UNRESOLVED", "UNREVIEWED")}


review_record = seal(
    {
        **common,
        "schema": "spacing-confirmation-review-record-v1",
        "process": "one Haiku author and one independent Sonnet reviewer per record; all records reviewed; two rounds, plus one round-3 pass limited to records UNRESOLVED after round 2 (ROUND3-AMENDMENT.md); non-APPROVE after the final round is UNRESOLVED, kept and disclosed",
        "counts": {
            "source_units": count(units),
            "groups": count(groups),
            "source_qa": count(source_qa),
            "paraphrases": count(paraphrases),
        },
        "unresolved_and_disclosed": unresolved,
        "unreviewed": unreviewed,
        "artifact_sha256": summary,
        "disclosures": [
            "Review-process amendment replaces the unadopted v2 receipt-timing layer; see ELECTIONS-AMENDMENT.md.",
            f"{renumbered} evidence line numbers were renumbered mechanically to 1-indexed factsheet lines; quote text unchanged and exact.",
            "Agent review is not human review; human_review_complete is false.",
            "Several place and person names recur across events; see the entity ledger (broad_LEAC tolerance).",
        ],
    }
)
(out / "confirmation-review-record.json").write_text(json.dumps(review_record, indent=1, ensure_ascii=False))
print(
    json.dumps(
        {
            "counts": review_record["counts"],
            "unresolved": len(unresolved),
            "unreviewed": len(unreviewed),
            "renumbered": renumbered,
            "review_record_sha256": review_record["sha256"],
            **summary,
        },
        indent=1,
    )
)
