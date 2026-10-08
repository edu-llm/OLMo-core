import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from spacing_rerun.common import digest
from spacing_rerun.data import canonicalize, outer_partition, assign_roles
from spacing_rerun.evaluation import aggregate, mcq_result
from spacing_rerun.schedule import compile_schedule, validate_schedule, stage1_epoch


def tiny_facts():
    return [{"id": f"{role}{i}", "event": f"{role}{i % 2}", "role": role,
             "statement": f"{role} statement {i}", "statement_ids": [i] * 5}
            for role in ("old", "new", "qa", "control") for i in range(7)]


def tiny_config():
    return {"batch_size": 8, "review_cap": 4, "cohort_size": 1, "qa_per_step": 1,
            "span": 21, "buffer_steps": 10, "order_seed": 7, "max_stagger_span_ratio": .75}


class ScheduleTests(unittest.TestCase):
    def test_exact_endpoint_dose_and_roundtrip(self):
        facts, config = tiny_facts(), tiny_config()
        plan = compile_schedule(facts, config)
        validate_schedule(json.loads(json.dumps(plan, sort_keys=True)), facts)
        for key in plan["starts"]:
            uni, exp, mass = [plan["review_times"][a][key] for a in ("UNI", "EXP", "MASS")]
            self.assertEqual(uni[0], exp[0])
            self.assertEqual(uni[-1], exp[-1])
            self.assertEqual(uni[-1], mass[-1])
            self.assertEqual([y - x for x, y in zip(mass, mass[1:])], [1, 1, 1])
            self.assertEqual([y - x for x, y in zip(exp, exp[1:])], [3, 6, 12])

    def test_reject_missing_review_even_if_replaced_with_valid_new(self):
        facts = tiny_facts()
        plan = compile_schedule(facts, tiny_config())
        plan["arms"]["EXP"][0][0] = "new/new0"
        with self.assertRaisesRegex(ValueError, "dose"):
            validate_schedule(plan, facts)

    def test_reject_runtime_shift(self):
        facts = tiny_facts()
        plan = compile_schedule(facts, tiny_config())
        plan["arms"]["UNI"][0], plan["arms"]["UNI"][1] = plan["arms"]["UNI"][1], plan["arms"]["UNI"][0]
        with self.assertRaises(ValueError):
            validate_schedule(plan, facts)

    def test_reject_control_or_buffer_leakage(self):
        facts = tiny_facts()
        for key in ("control/control0", "old/old0"):
            plan = compile_schedule(facts, tiny_config())
            plan["arms"]["UNI"][-1][0] = key
            with self.assertRaises(ValueError):
                validate_schedule(plan, facts)

    def test_reject_capacity_and_stagger_width(self):
        for changed in ({"cohort_size": 2}, {"max_stagger_span_ratio": .01}, {"span": 20}):
            with self.assertRaises(ValueError):
                compile_schedule(tiny_facts(), dict(tiny_config(), **changed))

    def test_stage1_exact_exposures_with_partial_last_batch(self):
        from collections import Counter
        facts, config = tiny_facts(), tiny_config()
        config["batch_size"] = 6
        exposure = Counter(k for epoch in range(5) for row in stage1_epoch(facts, epoch, config)
                           for k in row if k.startswith("old/"))
        self.assertEqual(set(exposure.values()), {5})
        self.assertEqual(len(exposure), 7)


class DataTests(unittest.TestCase):
    def row(self, key, root=None, event="a"):
        return {"question_id": key, "duplicate_root": root, "event_id": event,
                "fiction_id": "x_style_blog_num_1", "fict": "Fact " + key,
                "question": "Question " + key, "natural_answer": "Answer " + key}

    def test_transitive_clusters_and_canonical_answer(self):
        facts, audit = canonicalize([self.row("a"), self.row("b", "a"), self.row("c", "b")])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["members"], ["a", "b", "c"])
        self.assertEqual(facts[0]["aliases"], ["Answer a"])

    def test_cycle_and_missing_root_are_errors(self):
        for rows in ([self.row("a", "b")], [self.row("a", "b"), self.row("b", "a")]):
            with self.assertRaises(ValueError):
                canonicalize(rows)

    def test_cross_event_cluster_blocks_partition(self):
        facts, audit = canonicalize([self.row("a", event="a"), self.row("b", "a", "b")])
        self.assertTrue(audit["cross_event_links"])
        with self.assertRaisesRegex(ValueError, "Cross-event"):
            outer_partition(facts, audit["cross_event_links"])

    def test_disjoint_development_and_confirmation(self):
        facts = [{"id": str(i), "event": str(i), "style": "blog"} for i in range(100)]
        partition = outer_partition(facts, [])
        development = assign_roles(facts, partition, "development", 99)
        confirmation = assign_roles(facts, partition, "confirmation", 99)
        self.assertFalse(set(development["roles"]) & set(confirmation["roles"]))
        self.assertEqual(sum(r == "old" for r in confirmation["roles"].values()), 20)
        self.assertEqual(development, assign_roles(facts, partition, "development", 99))


class MetricTests(unittest.TestCase):
    def test_event_macro_does_not_count_facts_as_replicates(self):
        result = aggregate([{"role": "old", "event": "a", "loss": 0}] * 9 +
                           [{"role": "old", "event": "b", "loss": 10}])["old"]
        self.assertEqual(result["loss_event_macro"], 5)
        self.assertEqual(result["loss_fact_micro"], 1)

    def test_mcq_sum_and_mean_can_disagree_and_ties_fail(self):
        scores = [{"answer_nll_sum": 2., "loss": 2.}, {"answer_nll_sum": 3., "loss": .3}]
        result = mcq_result(scores, 0)
        self.assertEqual(result["mcq_accuracy"], 1)
        self.assertEqual(result["mcq_accuracy_normalized"], 0)
        self.assertEqual(mcq_result([scores[0], scores[0]], 0)["mcq_accuracy"], 0)

    def test_auc_excludes_buffer(self):
        from spacing_rerun.analysis import trapezoid
        self.assertEqual(trapezoid([(0, 0), (2, 2), (4, 0), (9, 100)], 0, 4), 1)


if __name__ == "__main__":
    unittest.main()
