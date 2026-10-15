"""Count where the reviewed catalog does not meet the confirmation-audit validators.

Read-only. Run on FarmShare: python -I -B audit_gap.py CATALOG_ROOT [A_DIR]
With A_DIR (relative to CATALOG_ROOT), Channel A records come from those event files, not compiled-final.
"""

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.data import normalize  # noqa: E402

root = pathlib.Path(sys.argv[1])
final = root / "compiled-final"
decl = json.load(open(final / "confirmation-grounded-declarations.json"))
qa = json.load(open(final / "confirmation-acquisition-qa.json"))
if len(sys.argv) > 2:
    docs = [json.load(open(p)) for p in sorted((root / sys.argv[2]).glob("event_*.json"))]
    decl = {"source_units": [dict(u, event=d["event"]) for d in docs for u in d["source_units"]],
            "groups": [dict(g, event=d["event"]) for d in docs for g in d["groups"]]}
    qa = {"records": [dict(r, event=d["event"]) for d in docs for r in d["source_qa"]]}
para = json.load(open(final / "confirmation-evaluation-paraphrases.json"))
catalog = json.load(open(root / "packets/source-catalog.json"))
raw = {u["unit_id"]: u for u in catalog["units"]}
lines = {e: f["text"].splitlines() for e, f in catalog["factsheets"].items()}
gap = collections.Counter()
examples = collections.defaultdict(list)


def note(key, example):
    gap[key] += 1
    if len(examples[key]) < 4:
        examples[key].append(example)


def quote_ok(ev, event):
    stripped = [line.strip() for line in lines[event]]
    return ev.get("quote", "").strip() in stripped


groups = {g["id"]: g for g in decl["groups"]}
units = {u["unit_id"]: u for u in decl["source_units"]}
for u in decl["source_units"]:
    gap["units"] += 1
    gap["unit_status:" + u["status"]] += 1
    if not u["evidence"]:
        note("unit_empty_evidence:" + u["status"], u["unit_id"])
    for ev in u["evidence"]:
        if not quote_ok(ev, u["event"]):
            note("unit_quote_not_a_line", (u["unit_id"], ev))
    g = groups[u["group_id"]]
    if u["status"] != "grounded":
        if len(g["source_unit_ids"]) != 1:
            note("preserved_unit_in_multi_unit_group", u["unit_id"])
        elif g["statement"] != raw[u["unit_id"]]["source_assertion"]:
            same = normalize(g["statement"]) == normalize(raw[u["unit_id"]]["source_assertion"])
            note("preserved_statement_" + ("punctuation_only" if same else "reworded"), u["unit_id"])
        if not u.get("note"):
            note("preserved_without_note", u["unit_id"])
    expected = [a for a in sorted(set(raw[u["unit_id"]]["canonical_answer_labels"]))
                if normalize(a) not in normalize(g["statement"])]
    if u["nonliteral_answer_labels"] != expected:
        note("nonliteral_flags_differ", (u["unit_id"], u["nonliteral_answer_labels"], expected))
signatures = collections.Counter((g["event"], normalize(g["statement"])) for g in decl["groups"])
for g in decl["groups"]:
    if signatures[(g["event"], normalize(g["statement"]))] > 1:
        note("duplicate_group_statement", g["id"])
    if "?" in g["statement"] or "Question:" in g["statement"] or "Answer:" in g["statement"]:
        note("group_statement_marker", g["id"])
    if not any(units[s]["evidence"] for s in g["source_unit_ids"]):
        note("group_without_any_member_evidence", g["id"])
targets = {(g["id"], normalize(a)) for g in decl["groups"] for s in g["source_unit_ids"]
           for a in raw[s]["canonical_answer_labels"]}
seen = collections.Counter()
forms = collections.Counter(normalize(q) for r in qa["records"] for q in r["questions"])
for r in qa["records"]:
    seen[(r["unit_id"], normalize(r["answer"]))] += 1
    if not r["evidence"]:
        note("qa_empty_evidence", r["id"])
    for ev in r["evidence"]:
        if not quote_ok(ev, r["event"]):
            note("qa_quote_not_a_line", (r["id"], ev))
    g = groups[r["unit_id"]]
    declarations = [g["statement"]] + [raw[s]["source_assertion"] for s in g["source_unit_ids"]]
    for q in r["questions"]:
        if any(normalize(t) in normalize(q) for t in declarations):
            note("qa_form_embeds_declaration", (r["id"], q))
        if forms[normalize(q)] > 1:
            note("qa_form_duplicated_globally", (r["id"], q))
        if "\n" in q or "Question:" in q or "Answer:" in q:
            note("qa_form_marker", r["id"])
    if len({normalize(q) for q in r["questions"]}) != 2:
        note("qa_forms_not_two_distinct", r["id"])
for t in targets - set(seen):
    note("qa_target_missing", t)
for t, n in seen.items():
    if t not in targets:
        note("qa_target_extra", t)
    elif n > 1:
        note("qa_target_duplicated", t)
for p in para["paraphrases"]:
    if normalize(p["paraphrase"]) in forms or normalize(p["canonical_question"]) in forms:
        note("eval_wording_in_acquisition_forms", p["opaque_probe_id"])
    if normalize(p["paraphrase"]) == normalize(p["canonical_question"]):
        note("paraphrase_equals_canonical", p["opaque_probe_id"])
statuses = collections.Counter(str(r.get("review", {}).get("round")) for r in decl["groups"])
print(json.dumps({"counts": gap, "group_final_round": statuses,
                  "qa_final_round": collections.Counter(str(r.get("review", {}).get("round")) for r in qa["records"]),
                  "paraphrase_final_round": collections.Counter(str(r.get("review", {}).get("round"))
                                                                for r in para["paraphrases"]),
                  "targets": len(targets), "qa_records": len(qa["records"]), "examples": examples},
                 indent=1, default=str, ensure_ascii=False))
