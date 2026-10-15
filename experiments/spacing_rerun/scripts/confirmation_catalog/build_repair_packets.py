"""Build the round-5 repair packets (AUDIT-CONFORMANCE-AMENDMENT.md part 3).

Run on FarmShare: python -I -B build_repair_packets.py CATALOG_ROOT
Writes CATALOG_ROOT/repair-packets/<event>.json and repair-packets/index.json. Never overwrites.
"""

import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.data import normalize  # noqa: E402

root = pathlib.Path(sys.argv[1])
out = root / "repair-packets"
out.mkdir(exist_ok=False)
catalog = json.load(open(root / "packets/source-catalog.json"))
raw = {u["unit_id"]: u for u in catalog["units"]}
qa_forms = {}
paraphrases = json.load(open(root / "compiled-final/confirmation-evaluation-paraphrases.json"))["paraphrases"]
evaluation_forms = {normalize(p[k]) for p in paraphrases for k in ("paraphrase", "canonical_question")}
amendment_sha = hashlib.sha256((root / "AUDIT-CONFORMANCE-AMENDMENT.md").read_bytes()).hexdigest()


def seal(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    body["sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return body


PRESERVE = ("Under the confirmation rules a preserved unit (assertion_only or conflict_flagged) must be its own group, "
            "and that group's statement must be the unit's source_statement copied verbatim, with no added or removed "
            "punctuation. Fix it in one of two ways. (a) Only for assertion_only: if an exact factsheet line supports "
            "the assertion as stated, set status \"grounded\", give the exact line(s) as evidence, and say why in "
            "\"note\". The unit may then stay in its group. (b) Otherwise give the unit its own group: a new id "
            "ground-<event>-<nn> with an unused nn, source_unit_ids [the unit], statement = source_statement verbatim, "
            "and a grouping_rationale. Remaining members keep their group. Every distinct answer label of every group "
            "must have exactly one source_qa record with two closed-book questions (AUTHORING-SPEC.md). Move, keep or "
            "write source_qa records so this holds, with unit_id set to the record's group id.")
index = []
for path in sorted((root / "authored/A").glob("event_*.json")):
    doc = json.load(open(path))
    event = doc["event"]
    units = {u["unit_id"]: u for u in doc["source_units"]}
    groups = {g["id"]: g for g in doc["groups"]}
    items, scope_groups = [], set()
    for g in doc["groups"]:
        members = [units[s] for s in g["source_unit_ids"]]
        preserved = [u for u in members if u["status"] != "grounded"]
        if preserved and len(members) > 1:
            items.append({"kind": "preserved_unit_in_merged_group", "group_id": g["id"],
                          "unit_ids": [u["unit_id"] for u in preserved], "instruction": PRESERVE})
            scope_groups.add(g["id"])
        elif preserved and g["statement"] != raw[members[0]["unit_id"]]["source_assertion"] and \
                normalize(g["statement"]) != normalize(raw[members[0]["unit_id"]]["source_assertion"]):
            items.append({"kind": "preserved_declaration_reworded", "group_id": g["id"],
                          "unit_ids": [members[0]["unit_id"]], "instruction": PRESERVE +
                          " For this item, option (b) means setting this group's statement to the source_statement "
                          "verbatim. Then check that its source_qa questions still follow the rules."})
            scope_groups.add(g["id"])
    for u in doc["source_units"]:
        if u["status"] != "grounded" and not u.get("note"):
            items.append({"kind": "preserved_unit_without_note", "group_id": u["group_id"], "unit_ids": [u["unit_id"]],
                          "instruction": "Add a \"note\" saying why this unit is " + u["status"] + "."})
            scope_groups.add(u["group_id"])
    by_statement = collections.defaultdict(list)
    for g in doc["groups"]:
        by_statement[normalize(g["statement"])].append(g["id"])
    for ids in by_statement.values():
        if len(ids) > 1:
            items.append({"kind": "duplicate_declaration", "group_ids": ids, "instruction":
                          "These groups have the same statement. If they state the same fact with the same answer, "
                          "merge them into one group (union of source_unit_ids, one source_qa record per distinct "
                          "answer label). If they are different facts, reword the statements so each states its own "
                          "fact. Preserved units still follow the rule above."})
            scope_groups.update(ids)
    for record in doc["source_qa"]:
        for question in record["questions"]:
            if normalize(question) in evaluation_forms:
                items.append({"kind": "question_form_not_allowed", "qa_id": record["id"], "question": question,
                              "instruction": "Replace this question form with a new closed-book form that follows "
                              "AUTHORING-SPEC.md. The new wording must differ from the current wording in more than "
                              "capitalisation or articles."})
                scope_groups.add(record["unit_id"])
    if not items:
        continue
    scope_units = sorted({s for gid in scope_groups for s in groups[gid]["source_unit_ids"]})
    packet = seal({
        "schema": "p4-confirmation-repair-packet-v1", "event": event, "round": 5,
        "audit_conformance_amendment_sha256": amendment_sha,
        "authored_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "factsheet_lines_1_indexed": {str(i): line for i, line in enumerate(catalog["factsheets"][event]["text"].split("\n"), 1)},
        "items": items,
        "scope": {"source_units": [units[s] for s in scope_units],
                  "groups": [groups[gid] for gid in sorted(scope_groups)],
                  "source_qa": [r for r in doc["source_qa"] if r["unit_id"] in scope_groups]},
        "other_group_ids_in_event": sorted(set(groups) - scope_groups),
        "other_question_forms_in_event": sorted(q for r in doc["source_qa"] if r["unit_id"] not in scope_groups
                                                for q in r["questions"]),
    })
    (out / f"{event}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False))
    index.append({"event": event, "items": collections.Counter(i["kind"] for i in items),
                  "scope_records": {k: len(v) for k, v in packet["scope"].items()}, "sha256": packet["sha256"]})
(out / "index.json").write_text(json.dumps(index, indent=1))
print(json.dumps({"events": len(index), "items": sum((collections.Counter(r["items"]) for r in index), collections.Counter()),
                  "scope": sum((collections.Counter(r["scope_records"]) for r in index), collections.Counter())}, indent=1))
