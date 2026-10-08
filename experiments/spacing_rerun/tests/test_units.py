from collections import Counter
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spacing_rerun.common import digest
from spacing_rerun.diagnose import summarize
from spacing_rerun.encoding import build_examples
from spacing_rerun.evaluation import aggregate, evaluate
from spacing_rerun.schedule import compile_schedule, stage1_epoch, validate_schedule
from spacing_rerun.training import verify_preregistration
from spacing_rerun.units import build_units, schedule_records, validate_units


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def fact(pid, statement, answer="Bob", role="old", event="a", candidate=None):
    return {"id": pid, "event": event, "role": role, "source_statement": statement,
            "statement": candidate or statement, "question": "Who?", "answer": answer, "aliases": [answer]}


class UnitTests(unittest.TestCase):
    def test_stable_source_identity_and_complete_multiple_answer_content(self):
        facts = [fact("b", "Bob gave Ann a pen.", "Ann", candidate="Bob's gift recipient was Ann."),
                 fact("a", "Bob gave Ann a pen.", "Bob", candidate="Ann's benefactor was Bob.")]
        registry = build_units(facts)
        unit = registry["units"][0]
        self.assertEqual(unit["member_probe_ids"], ["a", "b"])
        self.assertEqual(unit["representation"], "source_multiple_answers")
        self.assertEqual(unit["statement"], "Bob gave Ann a pen.")
        self.assertEqual(len({f["unit_id"] for f in facts}), 1)
        other = [fact("other", "Bob gave Ann a pen.", role="new")]
        self.assertEqual(build_units(other)["units"][0]["id"], unit["id"])
        self.assertEqual({f["answer"] for f in facts}, {"Ann", "Bob"})

    def test_rewrite_collisions_fall_back_without_merging_distinct_source_units(self):
        facts = [fact("a", "Bob arrived Monday.", candidate="The arrival was Bob."),
                 fact("b", "Bob arrived Tuesday.", candidate="The arrival was Bob.")]
        units = build_units(facts)["units"]
        self.assertEqual(len(units), 2)
        self.assertEqual({u["statement"] for u in units}, {"Bob arrived Monday.", "Bob arrived Tuesday."})
        self.assertEqual({u["representation"] for u in units}, {"source_rewrite_collision"})

    def test_registry_rejects_lost_probes_even_after_hash_is_recomputed(self):
        facts = [fact("a", "Bob arrived."), fact("b", "Bob arrived.")]
        registry = build_units(facts)
        registry["units"][0]["member_probe_ids"].remove("b")
        registry["sha256"] = digest({k: v for k, v in registry.items() if k != "sha256"})
        with self.assertRaisesRegex(ValueError, "Lost or duplicated"):
            validate_units(facts, registry)

    def test_each_unit_trained_once_per_epoch_and_four_times_per_arm(self):
        facts = [fact("old-a", "Bob arrived."), fact("old-b", "Bob arrived."),
                 fact("old-c", "Ann arrived.", "Ann"), fact("new", "New arrived.", role="new"),
                 fact("qa", "QA arrived.", role="qa"), fact("control", "Control arrived.", role="control")]
        registry = build_units(facts)
        config = {"batch_size": 6, "qa_per_step": 1, "review_cap": 4, "cohort_size": 1,
                  "span": 21, "buffer_steps": 1, "order_seed": 5, "max_stagger_span_ratio": .75,
                  "loss_tokens_per_example": 64, "context_length": 96, "eval_seed": 1}
        examples = build_examples(facts, Tokenizer(), list(range(1, 80)) * 20, config)
        records = schedule_records(facts, registry)
        self.assertEqual(len(facts), 6)
        self.assertEqual(len([key for key in examples if key.startswith("old/")]), 2)
        self.assertEqual(len({tuple(f["probe"]["answer_ids"]) for f in facts[:3]}), 2)
        counts = Counter(k for epoch in range(3) for row in stage1_epoch(records, epoch, config)
                         for k in row if k.startswith("old/"))
        self.assertEqual(sorted(counts.values()), [3, 3])
        plan = compile_schedule(records, config)
        validate_schedule(plan, records, examples)
        for arm in ("UNI", "EXP", "MASS"):
            counts = Counter(k for row in plan["arms"][arm] for k in row if k.startswith("old/"))
            self.assertEqual(sorted(counts.values()), [4, 4])

    def test_probes_average_within_units_then_events(self):
        rows = [{"id": "a", "unit_id": "one", "event": "A", "role": "old", "loss": 0},
                {"id": "b", "unit_id": "one", "event": "A", "role": "old", "loss": 2},
                {"id": "c", "unit_id": "two", "event": "A", "role": "old", "loss": 5},
                {"id": "d", "unit_id": "three", "event": "B", "role": "old", "loss": 9}]
        result = aggregate(rows)["old"]
        self.assertEqual(result["loss_event_macro"], 6)
        self.assertEqual(result["loss_fact_micro"], 4)
        self.assertEqual(result["loss_unit_micro"], 5)
        self.assertEqual(result["facts"], 4)
        self.assertEqual(result["units"], 3)
        diagnostic = summarize(rows, [{"loss": r["loss"]} for r in rows])
        self.assertEqual(diagnostic, result)

    def test_confirmation_rejects_legacy_before_checkpoint_use(self):
        with self.assertRaisesRegex(ValueError, "supporting-statement"):
            verify_preregistration(None, {"mode": "confirmation"})

    def test_optional_teaching_diagnostics_do_not_enter_old_metrics(self):
        class Model:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

        teaching = dict(fact("qa", "Bob arrived.", role="qa"), probe={})
        config = {"qa_teaching_diagnostics": True, "eval_batch_size": 1,
                  "generation_batch_size": 1, "max_answer_tokens": 8}
        with patch("spacing_rerun.evaluation.conditional_scores", return_value=[{"loss": .5}]), \
             patch("spacing_rerun.evaluation.generated_answers", return_value=[{"prediction": "Bob", "generation_terminated": 1}]):
            model = Model()
            output = evaluate(model, SimpleNamespace(eos_token_id=0), [teaching], config, "cpu", None)
        self.assertEqual(output["variants"], {})
        self.assertEqual(output["qa_teaching_diagnostic"]["aggregate"]["exact_match_event_macro"], 1)
        self.assertTrue(output["qa_teaching_diagnostic"]["in_sample_teaching_questions"])
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
