"""Apply repair-round patches to the Channel A event files, without touching the originals.

Run on FarmShare: python -I -B apply_repair.py CATALOG_ROOT BASE_DIR PATCH_DIR OUT_DIR
For example: authored/A authored/A-r5 authored/A-final5. Paths are relative to CATALOG_ROOT.
Each patch replaces exactly its repair packet's scope. Records outside the scope must not change.
"""

import collections
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.data import normalize  # noqa: E402

root = pathlib.Path(sys.argv[1])
base, patches, out = (root / name for name in sys.argv[2:5])
out.mkdir(exist_ok=False)
problems = collections.defaultdict(list)
fixed_unit_keys = ("unit_id", "event_id", "source_statement", "canonical_answers", "probe_ids")
for path in sorted(base.glob("event_*.json")):
    event = path.stem
    patch_path = patches / path.name
    if not patch_path.exists():
        shutil.copyfile(path, out / path.name)
        continue
    doc = json.load(open(path))
    patch = json.load(open(patch_path))
    packet = json.load(open(root / "repair-packets" / path.name))
    scope = packet["scope"]
    if patch.get("repair_packet_sha256") != packet["sha256"] or patch.get("event") != event:
        problems[event].append("patch is not bound to this event's repair packet")
    scope_units = {u["unit_id"]: u for u in scope["source_units"]}
    # The base may already carry an earlier round's groups, so derive the scope from membership.
    scope_groups = {g["id"] for g in doc["groups"] if set(g["source_unit_ids"]) & set(scope_units)}
    new_units = {u["unit_id"]: u for u in patch["source_units"]}
    if set(new_units) != set(scope_units) or len(new_units) != len(patch["source_units"]):
        problems[event].append("patch source_units differ from the repair scope")
    for uid, unit in new_units.items():
        if uid in scope_units and any(unit.get(k) != scope_units[uid][k] for k in fixed_unit_keys):
            problems[event].append(f"{uid}: fixed source fields changed")
    group_ids = [g["id"] for g in patch["groups"]]
    other_groups = set(packet["other_group_ids_in_event"])
    if len(set(group_ids)) != len(group_ids) or set(group_ids) & other_groups:
        problems[event].append("patch group ids collide")
    members = collections.Counter(s for g in patch["groups"] for s in g["source_unit_ids"])
    if members != collections.Counter(list(new_units)):
        problems[event].append("patch groups do not cover each scope unit exactly once")
    for unit in patch["source_units"]:
        group = next((g for g in patch["groups"] if g["id"] == unit["group_id"]), None)
        if group is None or unit["unit_id"] not in group["source_unit_ids"]:
            problems[event].append(f"{unit['unit_id']}: group_id does not name its group")
    patch_group_ids = set(group_ids)
    outside_qa = [r for r in doc["source_qa"] if r["unit_id"] not in scope_groups]
    qa_ids = [r["id"] for r in outside_qa + patch["source_qa"]]
    if len(set(qa_ids)) != len(qa_ids):
        problems[event].append("source_qa ids collide")
    if any(r["unit_id"] not in patch_group_ids for r in patch["source_qa"]):
        problems[event].append("patch source_qa targets a group outside the patch")
    targets = collections.Counter((r["unit_id"], normalize(r["answer"])) for r in patch["source_qa"])
    wanted = {(g["id"], normalize(a)) for g in patch["groups"] for s in g["source_unit_ids"]
              for a in new_units[s]["canonical_answers"]}
    if set(targets) != wanted or any(n > 1 for n in targets.values()):
        problems[event].append(f"patch source_qa targets differ: missing {sorted(wanted - set(targets))}, "
                               f"extra {sorted(set(targets) - wanted)}")
    forms = collections.Counter(normalize(q) for r in outside_qa + patch["source_qa"] for q in r["questions"])
    if any(n > 1 for n in forms.values()):
        problems[event].append("duplicate question forms in event")
    merged = dict(doc)
    merged["source_units"] = [new_units.get(u["unit_id"], u) for u in doc["source_units"]]
    merged["groups"] = [g for g in doc["groups"] if g["id"] not in scope_groups] + patch["groups"]
    merged["source_qa"] = outside_qa + patch["source_qa"]
    merged["repair_changes"] = patch.get("changes", [])
    (out / path.name).write_text(json.dumps(merged, indent=1, ensure_ascii=False))
print(json.dumps({"patched": len(list(patches.glob("event_*.json"))), "problems": problems}, indent=1, ensure_ascii=False))
sys.exit(1 if problems else 0)
