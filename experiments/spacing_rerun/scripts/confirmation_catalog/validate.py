"""Mechanical checks for the confirmation catalog. Run on FarmShare: python -I -B validate.py ROOT QA_PARQUET"""

import json
import pathlib
import re
import sys
import pyarrow.parquet as pq

root, qa_path = pathlib.Path(sys.argv[1]), sys.argv[2]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


qa = pq.read_table(qa_path).to_pylist()
eval_questions = {norm(r["question"]) for r in qa}
pmap = json.load(open(root / "packets/private-probe-id-map.json"))["probes"]
report = {"A": {}, "B": {}, "missing_files": []}


def contains(text, label):
    return f" {norm(label)} " in f" {norm(text)} " if norm(label) else False


def check_evidence(rid, evidence, lines, issues, renumbered):
    stripped = [line.strip() for line in lines]
    for ev in evidence:
        ln, quote = ev.get("line"), ev.get("quote", "").strip()
        if isinstance(ln, int) and 1 <= ln <= len(lines) and stripped[ln - 1] == quote:
            continue
        if quote and quote in stripped:
            renumbered.append({"id": rid, "authored_line": ln, "matching_line": stripped.index(quote) + 1})
        else:
            issues.append(f"{rid}: evidence line {ln} quote not found in factsheet")


renumbered_all = {}
for n in range(1, 9):
    ap = json.load(open(root / f"packets/author-packets/chunk-{n:02d}.json"))
    for event in ap["events"]:
        lines = ap["factsheets"][event]["text"].split("\n")
        units = {u["unit_id"]: u for u in ap["units"] if u["event"] == event}
        path, issues = root / f"authored/A/{event}.json", []
        renumbered = renumbered_all.setdefault(event, [])
        if not path.exists():
            report["missing_files"].append(str(path))
            continue
        try:
            doc = json.load(open(path))
        except Exception as e:
            report["A"][event] = [f"invalid JSON: {e}"]
            continue
        seen = {}
        for su in doc.get("source_units", []):
            uid = su.get("unit_id")
            if uid not in units:
                issues.append(f"{uid}: unit not in packet")
                continue
            seen[uid] = seen.get(uid, 0) + 1
            u = units[uid]
            if su.get("source_statement") != u["source_assertion"]:
                issues.append(f"{uid}: source_statement not verbatim")
            if sorted(su.get("canonical_answers", [])) != sorted(u["canonical_answer_labels"]):
                issues.append(f"{uid}: canonical_answers differ")
            if sorted(su.get("probe_ids", [])) != sorted(p["opaque_probe_id"] for p in u["canonical_probes"]):
                issues.append(f"{uid}: probe_ids differ")
            decl = su.get("proposed_declaration", "")
            if len(decl.split()) > 28:
                issues.append(f"{uid}: declaration over 28 words")
            for lab in set(u["canonical_answer_labels"]) - set(su.get("nonliteral_answer_labels", [])):
                if not contains(decl, lab):
                    issues.append(f"{uid}: label {lab!r} not in declaration and not listed nonliteral")
            if norm(decl) in eval_questions:
                issues.append(f"{uid}: declaration equals an evaluation question")
        for uid in units:
            if seen.get(uid, 0) != 1:
                issues.append(f"{uid}: appears {seen.get(uid, 0)} times in source_units")
        membership = {}
        groups = {g.get("id"): g for g in doc.get("groups", [])}
        for g in groups.values():
            for uid in g.get("source_unit_ids", []):
                membership[uid] = membership.get(uid, 0) + 1
        for uid in units:
            if membership.get(uid, 0) != 1:
                issues.append(f"{uid}: in {membership.get(uid, 0)} groups")
        targets = {}
        for gid, g in groups.items():
            for uid in g.get("source_unit_ids", []):
                for lab in units.get(uid, {}).get("canonical_answer_labels", []):
                    targets.setdefault((gid, norm(lab)), lab)
        covered = set()
        for rec in doc.get("source_qa", []):
            rid, gid, ans = rec.get("id"), rec.get("unit_id"), rec.get("answer", "")
            covered.add((gid, norm(ans)))
            if (gid, norm(ans)) not in targets:
                issues.append(f"{rid}: answer/group not a packet target")
            qs = rec.get("questions", [])
            if len(qs) != 2 or len({norm(q) for q in qs}) != 2:
                issues.append(f"{rid}: needs 2 distinct questions")
            for q in qs:
                if contains(q, ans):
                    issues.append(f"{rid}: question contains answer")
                if norm(q) in eval_questions:
                    issues.append(f"{rid}: question equals an evaluation question")
            check_evidence(rid, rec.get("evidence", []), lines, issues, renumbered)
        for su in doc.get("source_units", []):
            check_evidence(su.get("unit_id"), su.get("evidence", []), lines, issues, renumbered)
        for key in set(targets) - covered:
            issues.append(f"{key[0]}: no source_qa for answer {targets[key]!r}")
        report["A"][event] = issues

    bp = json.load(open(root / f"paraphrase-packets/chunk-{n:02d}.json"))
    for event in bp["events"]:
        probes = {p["opaque_probe_id"]: p for p in bp["probes"] if p["event"] == event}
        path, issues = root / f"authored/B/{event}.json", []
        if not path.exists():
            report["missing_files"].append(str(path))
            continue
        try:
            doc = json.load(open(path))
        except Exception as e:
            report["B"][event] = [f"invalid JSON: {e}"]
            continue
        seen = {}
        for p in doc.get("paraphrases", []):
            pid = p.get("opaque_probe_id")
            if pid not in probes:
                issues.append(f"{pid}: not in packet")
                continue
            seen[pid] = seen.get(pid, 0) + 1
            src, para = probes[pid], p.get("paraphrase", "")
            if norm(para) == norm(src["canonical_question"]):
                issues.append(f"{pid}: paraphrase identical to canonical")
            if contains(para, src["canonical_answer"]) and not contains(
                src["canonical_question"], src["canonical_answer"]
            ):
                issues.append(f"{pid}: paraphrase contains answer")
            if not para.strip():
                issues.append(f"{pid}: empty paraphrase")
        for pid in probes:
            if seen.get(pid, 0) != 1:
                issues.append(f"{pid}: appears {seen.get(pid, 0)} times")
        report["B"][event] = issues

report["evidence_line_renumber_only"] = renumbered_all
summary = {
    ch: {
        "events": len(report[ch]),
        "events_with_issues": sum(1 for v in report[ch].values() if v),
        "issues": sum(len(v) for v in report[ch].values()),
    }
    for ch in ("A", "B")
}
summary["missing_files"] = len(report["missing_files"])
summary["evidence_line_renumber_only"] = sum(len(v) for v in renumbered_all.values())
report["summary"] = summary
(root / "validation-report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
print(json.dumps(summary, indent=1))
