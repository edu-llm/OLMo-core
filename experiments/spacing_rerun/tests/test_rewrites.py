import copy
import unittest

from spacing_rerun.common import digest
from spacing_rerun.rewrites import apply_development_rewrites, source_fields


def fixture():
    facts = [{"id": role, "event": role, "role": role, "statement": f"{role} Club opened in 1968",
              "answer": f"{role} Club", "question": "Which club opened in 1968?", "aliases": [f"{role} Club"]}
             for role in ("old", "new", "qa", "control")]
    rows = [{"id": f["id"], "event": f["event"], "role": f["role"], "source_statement": f["statement"],
             "canonical_answer": f["answer"], "training_statement": f"Opening in 1968 was {f['answer']}.",
             "source_sha256": digest({"statement": f["statement"], "answer": f["answer"]}),
             "independent_review": {"reviewer": "fixture", "status": "source_checked_with_corrections"}}
            for f in facts]
    roles = {f["event"]: f["role"] for f in facts}
    bundle = {"schema": "spacing-directional-development-v1", "scope": "development_only",
              "author_input_fields": ["source_statement", "canonical_answer"], "outer_partition_sha256": "partition",
              "source_split_roles_sha256": digest(roles), "source_fields_sha256": digest(source_fields(facts)), "records": rows}
    seal(bundle)
    return facts, bundle, roles


def seal(bundle):
    bundle["sha256"] = digest({k: v for k, v in bundle.items() if k != "sha256"})


def apply(facts, bundle, roles, mode="development"):
    return apply_development_rewrites(facts, bundle, mode=mode, partition_hash="partition", roles=roles)


class RewriteTests(unittest.TestCase):
    def test_changes_only_old_new_training_forms(self):
        facts, bundle, roles = fixture()
        before = copy.deepcopy(facts)
        audit = apply(facts, bundle, roles)
        for previous, current in zip(before, facts):
            self.assertEqual(previous["question"], current["question"])
            self.assertEqual(previous["answer"], current["answer"])
            self.assertEqual(previous["aliases"], current["aliases"])
            if current["role"] in ("qa", "control"):
                self.assertEqual(previous, current)
            else:
                self.assertEqual(current["source_statement"], previous["statement"])
                self.assertNotEqual(current["statement"], previous["statement"])
        self.assertFalse(audit["confirmation_ready"])

    def test_rejects_confirmation_or_changed_source(self):
        facts, bundle, roles = fixture()
        with self.assertRaises(ValueError):
            apply(facts, bundle, roles, mode="confirmation")
        facts[0]["statement"] = "Different source"
        with self.assertRaises(ValueError):
            apply(facts, bundle, roles)

    def test_rejects_filtering_and_question_training(self):
        for corruption in ("missing", "question", "nonfinal"):
            facts, bundle, roles = fixture()
            if corruption == "missing":
                bundle["records"].pop()
            elif corruption == "question":
                bundle["records"][0]["training_statement"] = facts[0]["question"] + " old Club"
            else:
                bundle["records"][0]["training_statement"] = "old Club opened in 1968"
            seal(bundle)
            with self.assertRaises(ValueError):
                apply(facts, bundle, roles)

    def test_explicit_exception_preserves_source_and_fact(self):
        facts, bundle, roles = fixture()
        row = bundle["records"][0]
        row["training_statement"] = row["source_statement"]
        row["answer_final_exception"] = "Cannot move the target without adding specificity."
        seal(bundle)
        audit = apply(facts, bundle, roles)
        self.assertEqual(len(facts), 4)
        self.assertEqual(facts[0]["statement"], facts[0]["source_statement"])
        self.assertEqual(audit["source_preserved_exceptions"][0]["id"], "old")

    def test_exception_does_not_authorize_new_unreviewed_content(self):
        facts, bundle, roles = fixture()
        bundle["records"][0]["training_statement"] = "old Club was discovered in 1968"
        bundle["records"][0]["answer_final_exception"] = "Exception"
        seal(bundle)
        with self.assertRaises(ValueError):
            apply(facts, bundle, roles)


if __name__ == "__main__":
    unittest.main()
