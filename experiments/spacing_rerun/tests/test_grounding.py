import copy
import json
from collections import Counter
from pathlib import Path
import unittest

from spacing_rerun.common import digest
from spacing_rerun.data import normalize
from spacing_rerun.grounding import build_grounded_registry, validate_bundle, validate_grounded_registry
from spacing_rerun.schedule import compile_schedule, stage1_epoch, validate_schedule
from spacing_rerun.units import build_units, schedule_records


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    """Use reviewed public source assertions, without model results or network."""
    bundle = json.loads((ROOT / "data/development-grounded-declarations.json").read_text())
    facts = []
    for record in bundle["source_units"]:
        for pid, answer in zip(record["probe_ids"], record["canonical_answers"]):
            facts.append({"id": pid, "event": record["event_id"], "role": record["role"],
                          "statement": record["source_statement"], "answer": answer, "question": "Unseen evaluation " + pid})
    # Add the untouched roles to match the manifest's complete probe count.
    for role, n in (("qa", 64), ("control", 62)):
        facts.extend({"id": f"{role}{i}", "event": role, "role": role,
                      "statement": f"{role} fact {i}", "answer": str(i), "question": f"Unseen {role} {i}"}
                     for i in range(n))
    registry = build_units(facts)
    roles = {f["event"]: f["role"] for f in facts}
    bundle["roles_sha256"] = digest(roles)
    bundle["sha256"] = digest({k: v for k, v in bundle.items() if k != "sha256"})
    return facts, registry, bundle, roles


def validate(facts, registry, bundle, roles, sheets=None, mode="development"):
    validate_bundle(bundle, facts, registry, mode=mode, partition_hash=bundle["partition_sha256"],
                    roles=roles, pinned_factsheets=sheets)


def rehash(bundle):
    bundle["sha256"] = digest({k: v for k, v in bundle.items() if k != "sha256"})


class GroundingTests(unittest.TestCase):
    def test_natural_relative_clause_allowed_but_complete_question_rejected(self):
        facts, registry, bundle, roles = fixture()
        facts[0]["question"] = "Who developed the FlourFusion Reactor?"
        validate(facts, registry, bundle, roles)
        group = next(g for g in bundle["groups"] if 88 in g["source_indices"])
        group["statement"] = "Who developed the FlourFusion Reactor? Jasmine Lin."
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "declaration"):
            validate(facts, registry, bundle, roles)
        group["statement"] = "Who developed the FlourFusion Reactor."
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "Held-out question"):
            validate(facts, registry, bundle, roles)

    def test_reviewed_groups_preserve_every_probe_and_exact_group_doses(self):
        facts, source, bundle, roles = fixture()
        before = [{k: f[k] for k in ("id", "question", "answer", "event", "role")} for f in facts]
        validate(facts, source, bundle, roles, bundle["factsheets"])
        registry = build_grounded_registry(facts, source, bundle)
        validate_grounded_registry(facts, registry, source, bundle)
        self.assertEqual(before, [{k: f[k] for k in before[0]} for f in facts])
        self.assertEqual(registry["unit_counts"]["old"], 73)
        self.assertEqual(registry["unit_counts"]["new"], 84)
        self.assertEqual(len(facts), 348)
        multi = next(g for g in registry["units"] if len(g["source_unit_ids"]) == 3 and g["role"] == "old")
        self.assertGreaterEqual(len(multi["member_probe_ids"]), 3)
        config = json.loads((ROOT / "configs/calibration-grounded.json").read_text())
        records = schedule_records(facts, registry)
        counts = Counter(k for epoch in range(3) for row in stage1_epoch(records, epoch, config)
                         for k in row if k.startswith("old/"))
        self.assertEqual(len(counts), 73)
        self.assertEqual(set(counts.values()), {3})
        plan = compile_schedule(records, config)
        validate_schedule(plan, records)
        for arm in ("UNI", "EXP", "MASS"):
            counts = Counter(k for row in plan["arms"][arm] for k in row if k.startswith("old/"))
            self.assertEqual(len(counts), 73)
            self.assertEqual(set(counts.values()), {4})

    def test_grounding_requires_review_pinned_factsheets_and_development(self):
        facts, registry, bundle, roles = fixture()
        with self.assertRaisesRegex(ValueError, "development-only"):
            validate(facts, registry, bundle, roles, mode="confirmation")
        modified = dict(bundle["factsheets"])
        modified[next(iter(modified))] += " invented context"
        with self.assertRaisesRegex(ValueError, "pinned parquet"):
            validate(facts, registry, bundle, roles, sheets=modified)
        bundle["independent_agent_review_complete"] = False
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "independent review"):
            validate(facts, registry, bundle, roles)

    def test_evidence_and_source_conflicts_cannot_be_silently_changed(self):
        facts, registry, bundle, roles = fixture()
        bundle["source_units"][0]["evidence"][0]["quote"] += " not in the source"
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            validate(facts, registry, bundle, roles)
        facts, registry, bundle, roles = fixture()
        record = next(r for r in bundle["source_units"] if r["status"].startswith("preserve_source"))
        group = next(g for g in bundle["groups"] if g["id"] == record["group_id"])
        group["statement"] += " Changed claim."
        record["trained_declaration"] = group["statement"]
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "Unresolved source conflict"):
            validate(facts, registry, bundle, roles)

    def test_duplicated_source_membership_and_regrouped_probe_are_rejected(self):
        facts, source, bundle, roles = fixture()
        bundle["groups"].append(copy.deepcopy(bundle["groups"][0]))
        rehash(bundle)
        with self.assertRaisesRegex(ValueError, "Duplicate grounded group"):
            validate(facts, source, bundle, roles)
        facts, source, bundle, roles = fixture()
        registry = build_grounded_registry(facts, source, bundle)
        first, other = [f for f in facts if f["role"] == "old"][:2]
        other = next(f for f in facts if f["role"] == "old" and f["unit_id"] != first["unit_id"])
        first["unit_id"] = other["unit_id"]
        with self.assertRaisesRegex(ValueError, "probe membership"):
            validate_grounded_registry(facts, registry, source, bundle)


if __name__ == "__main__":
    unittest.main()
