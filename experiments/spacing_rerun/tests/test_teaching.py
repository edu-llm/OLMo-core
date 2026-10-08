import copy
import unittest
from collections import Counter

from spacing_rerun.encoding import build_examples, encode_probe
from spacing_rerun.schedule import compile_schedule, stage1_epoch, validate_schedule
from spacing_rerun.teaching import SOURCE_QA_POLICY, build_teaching_pool
from spacing_rerun.units import build_units, schedule_records


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def fixture():
    facts = [{"id": role, "event": role, "role": role, "statement": "Bob arrived.",
              "answer": "Bob", "question": role + " who?", "aliases": ["Bob"], "members": [role]}
             for role in ("old", "new", "control", "qa")]
    facts[-1]["members"] = ["qa", "qa-variant"]
    rows = [{"question_id": f["id"], "event_id": f["event"], "question": f["question"],
             "natural_answer": f["answer"], "fiction_id": "story"} for f in facts]
    # Similarity clusters can contain another fact: its answer must stay its own.
    rows.append(dict(rows[-1], question_id="qa-variant", question="Who left?", natural_answer="Ann"))
    return facts, rows, {f["event"]: f["role"] for f in facts}


class TeachingTests(unittest.TestCase):
    def test_continuous_teaching_stream_has_equal_complete_cycle_doses_and_restarts(self):
        facts = [{"id": f"old{i}", "role": "old"} for i in range(86)] + [
                 {"id": f"qa{i}", "role": "qa"} for i in range(225)]
        config = {"batch_size": 16, "qa_per_step": 4, "order_seed": 17, "qa_teaching_policy": SOURCE_QA_POLICY}
        planned = [row for epoch in range(64) for row in stage1_epoch(facts, epoch, config)]
        qa = [k for row in planned for k in row if k.startswith("qa/")]
        self.assertEqual(len(planned), 512)
        self.assertEqual(len(qa), 2048)
        for begin in range(0, len(qa) - 224, 225):
            self.assertEqual(len(set(qa[begin:begin + 225])), 225)
        counts = Counter(qa)
        self.assertEqual(len(counts), 225)
        self.assertEqual(max(counts.values()) - min(counts.values()), 1)
        # Mid-epoch resume uses the checkpoint's epoch and row cursor, without
        # resetting or deriving the QA cursor from an old-unit offset.
        epoch, cursor = 7, 3
        prefix = epoch * 8 + cursor
        resumed = stage1_epoch(facts, epoch, config)[cursor:] + [
            row for next_epoch in range(epoch + 1, 64) for row in stage1_epoch(facts, next_epoch, config)]
        self.assertEqual(planned[prefix:], resumed)

    def test_source_variants_keep_own_labels_and_never_use_other_event_roles(self):
        facts, rows, roles = fixture()
        evaluation_before = copy.deepcopy(facts)
        pool = build_teaching_pool(rows, facts, roles, "development")
        self.assertEqual(facts, evaluation_before)
        self.assertEqual([(r["id"], r["answer"]) for r in pool["records"]], [("qa", "Bob"), ("qa-variant", "Ann")])
        self.assertEqual({r["event"] for r in pool["records"]}, {"qa"})
        self.assertEqual({r["canonical_probe_id"] for r in pool["records"]}, {"qa"})
        with self.assertRaisesRegex(ValueError, "development-only"):
            build_teaching_pool(rows, facts, roles, "confirmation")

    def test_cross_role_root_and_exact_heldout_question_leakage_are_rejected(self):
        facts, rows, roles = fixture()
        rows[-1]["question"] = facts[0]["question"]
        with self.assertRaisesRegex(ValueError, "Held-out"):
            build_teaching_pool(rows, facts, roles, "development")
        facts, rows, roles = fixture()
        facts[-1]["members"].append("old")
        with self.assertRaisesRegex(ValueError, "membership"):
            build_teaching_pool(rows, facts, roles, "development")

    def test_expanded_stream_has_exact_masks_doses_and_arm_matching(self):
        facts, rows, roles = fixture()
        pool = build_teaching_pool(rows, facts, roles, "development")
        units = build_units(facts)
        config = {"batch_size": 8, "qa_per_step": 4, "review_cap": 4, "cohort_size": 1,
                  "span": 21, "buffer_steps": 2, "order_seed": 5, "max_stagger_span_ratio": .75,
                  "loss_tokens_per_example": 48, "context_length": 96, "eval_seed": 1}
        examples = build_examples(facts, Tokenizer(), list(range(1, 80)) * 20, config, pool["records"])
        probe = encode_probe(Tokenizer(), "Who left?", "Ann", 96)
        example = examples["qa/qa-variant"]
        begin = len(probe["prompt_ids"])
        self.assertEqual(example["labels"][:begin], [-100] * begin)
        self.assertEqual(example["labels"][begin:begin + len(probe["answer_ids"])], probe["answer_ids"])
        self.assertEqual(sum(t != -100 for t in example["labels"][1:]), 48)
        records = schedule_records(facts, units, pool["records"])
        plan = compile_schedule(records, config)
        validate_schedule(plan, records, examples)
        stream = lambda arm: [k for row in plan["arms"][arm] for k in row if k.startswith("qa/")]
        self.assertEqual(set(stream("UNI")), {"qa/qa", "qa/qa-variant"})
        for arm in plan["arms"]:
            self.assertEqual(stream(arm), stream("UNI"))
        epochs = [row for i in range(3) for row in stage1_epoch(records, i, config)]
        self.assertTrue(all(sum(k.startswith("qa/") for k in row) == 4 for row in epochs))
        self.assertEqual(set(Counter(k for row in epochs for k in row if k.startswith("old/")).values()), {3})


if __name__ == "__main__":
    unittest.main()
