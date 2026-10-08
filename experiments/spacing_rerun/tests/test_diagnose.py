import contextlib
import unittest
from types import SimpleNamespace

import torch

from spacing_rerun.diagnose import answer_span_probe, declarative_probes, summarize
from spacing_rerun.evaluation import conditional_scores


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class DiagnosticTests(unittest.TestCase):
    def test_qa_teaching_is_counted_without_relabeling_it_old(self):
        facts = [{"event": "a", "role": "qa"}] * 3 + [{"event": "b", "role": "qa"}]
        scores = [{"loss": 0., "exact_match": 1.}] * 3 + [{"loss": 4., "exact_match": 0.}]
        output = summarize(facts, scores)
        self.assertEqual(output["loss_event_macro"], 2.)
        self.assertEqual(output["exact_match_event_macro"], .5)
        self.assertEqual(output["exact_match_fact_micro"], .75)

    def test_statement_probe_omits_filler_and_eos_from_primary(self):
        fact = {"id": "a", "statement_ids": [2, 3]}
        example = {"input_ids": [9, 2, 3, 9, 8, 8], "labels": [-100, 2, 3, 9, 8, 8], "content_tokens": 3}
        probes = declarative_probes([fact], {"old/a": example}, 9)
        self.assertEqual(probes, [{"prompt_ids": [9], "answer_ids": [2, 3]}])

        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                self.observed = input_ids.tolist()
                logits = torch.zeros((*input_ids.shape, 10))
                logits[:, 0, 2] = 5
                logits[:, 1, 3] = 5
                logits[:, 2, 9] = -9
                return SimpleNamespace(logits=logits)

        model = Model()
        score = conditional_scores(model, probes, 9, "cpu", 1, contextlib.nullcontext)[0]
        self.assertEqual(model.observed, [[9, 2, 3, 9]])
        self.assertEqual(score["answer_tokens"], 2)
        self.assertLess(score["loss"], .1)
        self.assertGreater(score["loss_with_eos"], 2.)

    def test_statement_corruption_is_rejected(self):
        fact = {"id": "a", "statement_ids": [2, 3]}
        example = {"input_ids": [9, 2, 4, 9], "labels": [-100, 2, 4, 9], "content_tokens": 3}
        with self.assertRaises(ValueError):
            declarative_probes([fact], {"old/a": example}, 9)

    def test_answer_span_scores_only_gold_span_with_exact_prefix(self):
        statement = "Town: Crestwood."
        fact = {"id": "a", "statement": statement, "answer": "Crestwood", "statement_ids": [ord(c) for c in statement]}
        probe, audit = answer_span_probe(fact, CharacterTokenizer(), 9)
        self.assertEqual(probe["prompt_ids"], [9] + [ord(c) for c in "Town: "])
        self.assertEqual(probe["answer_ids"], [ord(c) for c in "Crestwood"])
        self.assertEqual(audit["reason"], "eligible")
        self.assertEqual(audit["character_start"], 6)

    def test_missing_duplicate_and_merged_span_are_excluded(self):
        for statement, answer, reason in (("Town is elsewhere.", "Crestwood", "canonical_answer_not_present"),
                                           ("Crestwood and Crestwood", "Crestwood", "canonical_answer_present_multiple_times")):
            fact = {"id": "a", "statement": statement, "answer": answer, "statement_ids": [ord(c) for c in statement]}
            self.assertEqual(answer_span_probe(fact, CharacterTokenizer(), 9)[1]["reason"], reason)

        class Merged:
            def __call__(self, *args, **kwargs):
                return {"input_ids": [1], "offset_mapping": [(0, 10)]}

        fact = {"id": "a", "statement": "NewYorkers", "answer": "NewYork", "statement_ids": [1]}
        probe, audit = answer_span_probe(fact, Merged(), 9)
        self.assertIsNone(probe)
        self.assertEqual(audit["reason"], "answer_boundary_inside_nonwhitespace_token")


if __name__ == "__main__":
    unittest.main()
