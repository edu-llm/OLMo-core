import contextlib
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from spacing_rerun.common import append_json, digest
from spacing_rerun.encoding import packed_example
from spacing_rerun.evaluation import conditional_scores
from spacing_rerun.training import (load_checkpoint, make_optimizer, recover_exposure_log,
                                    save_checkpoint, seed_all, train_update)
from spacing_rerun.schedule import stage1_epoch


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(c) % 7 for c in text]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(10, 7)
        self.dropout = torch.nn.Dropout(.3)
        self.proj = torch.nn.Linear(7, 10)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.proj(self.dropout(self.emb(input_ids))))


class TrainingTests(unittest.TestCase):
    def config(self, microbatch=1):
        return {"learning_rate": .001, "warmup_steps": 1, "microbatch_size": microbatch}

    def examples(self):
        return {k: packed_example(Tokenizer(), text, k, [1, 2, 3, 4] * 50, 8, 12)
                for k, text in (("a", "x"), ("b", "abc"), ("c", "de"))}

    def test_token_budget_is_exact_without_truncation(self):
        examples = self.examples()
        for row in examples.values():
            self.assertEqual(sum(t != -100 for t in row["labels"][1:]), 8)
            self.assertEqual(len(row["input_ids"]), 12)
            self.assertEqual(row["loss_tokens"], 8)
        with self.assertRaises(ValueError):
            packed_example(Tokenizer(), "too long", "bad", [1] * 100, 2, 12)

    def test_resume_is_bitwise_identical_with_optimizer_and_rng(self):
        seed_all(314)
        model = TinyLM()
        optimizer = make_optimizer(model, self.config())
        examples = self.examples()
        row = list(examples)
        train_update(model, optimizer, row, examples, self.config(), 0, "cpu", contextlib.nullcontext)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            save_checkpoint(checkpoint, model, optimizer, {"cursor": 1}, "manifest-a")
            expected_loss = train_update(model, optimizer, row, examples, self.config(), 1, "cpu", contextlib.nullcontext)
            expected_state = copy.deepcopy(model.state_dict())
            other = TinyLM()
            other_optimizer = make_optimizer(other, self.config())
            self.assertEqual(load_checkpoint(checkpoint, other, other_optimizer, "manifest-a", "cpu"), {"cursor": 1})
            actual_loss = train_update(other, other_optimizer, row, examples, self.config(), 1, "cpu", contextlib.nullcontext)
            self.assertEqual(expected_loss, actual_loss)
            for key in expected_state:
                self.assertTrue(torch.equal(expected_state[key], other.state_dict()[key]), key)
            with self.assertRaisesRegex(ValueError, "different manifest"):
                load_checkpoint(checkpoint, other, other_optimizer, "manifest-b", "cpu")

    def test_gradient_accumulation_uses_global_token_mean(self):
        seed_all(12)
        one = TinyLM()
        one.dropout.p = 0
        other = copy.deepcopy(one)
        o1, o2 = make_optimizer(one, self.config()), make_optimizer(other, self.config(3))
        examples = self.examples()
        a = train_update(one, o1, list(examples), examples, self.config(), 0, "cpu", contextlib.nullcontext)
        b = train_update(other, o2, list(examples), examples, self.config(3), 0, "cpu", contextlib.nullcontext)
        self.assertAlmostEqual(a["loss"], b["loss"], places=6)
        for key in one.state_dict():
            torch.testing.assert_close(one.state_dict()[key], other.state_dict()[key], rtol=1e-5, atol=1e-7)

    def test_mid_epoch_restart_preserves_all_later_epochs(self):
        """An interruption after unsaved work resumes the saved shuffled stream."""
        from collections import Counter
        config = dict(self.config(), batch_size=6, qa_per_step=1, order_seed=17,
                      qa_teaching_policy="all_source_questions_v1")
        facts = [{"id": f"{role}{i}", "role": role} for role in ("old", "qa") for i in range(7)]
        examples = {role + "/" + f["id"]: packed_example(Tokenizer(), "ab", f["id"],
                    [1, 2, 3, 4] * 50, 8, 12) for f in facts for role in [f["role"]]}
        examples["filler"] = packed_example(Tokenizer(), "", "filler", [1, 2, 3, 4] * 50, 8, 12)
        planned = [row for epoch in range(5) for row in stage1_epoch(facts, epoch, config)]
        self.assertEqual(len(planned), 10)
        seed_all(811)
        model = TinyLM()
        optimizer = make_optimizer(model, config)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, log = Path(directory) / "state.pt", Path(directory) / "log.jsonl"
            for i, row in enumerate(planned):
                train_update(model, optimizer, row, examples, config, i, "cpu", contextlib.nullcontext)
                append_json(log, {"cursor": i + 1, "examples": row, "row_sha256": digest(row)})
                if i == 2:
                    # Two rows per epoch: this is halfway through epoch1, well
                    # away from an epoch/milestone boundary.
                    save_checkpoint(checkpoint, model, optimizer,
                                    {"epoch": 1, "epoch_cursor": 1, "global_step": 3, "cursor": 3}, "m")
            expected = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            resumed = TinyLM()
            resumed_optimizer = make_optimizer(resumed, config)
            progress = load_checkpoint(checkpoint, resumed, resumed_optimizer, "m", "cpu")
            recover_exposure_log(log, progress["cursor"])
            consumed = planned[:progress["cursor"]]
            for epoch in range(progress["epoch"], 5):
                rows = stage1_epoch(facts, epoch, config)
                for index in range(progress["epoch_cursor"], len(rows)):
                    train_update(resumed, resumed_optimizer, rows[index], examples, config,
                                 progress["global_step"], "cpu", contextlib.nullcontext)
                    progress["global_step"] += 1
                    consumed.append(rows[index])
                progress["epoch_cursor"] = 0
            self.assertEqual(consumed, planned)
            counts = Counter(k for row in consumed for k in row if k.startswith("old/"))
            self.assertEqual(set(counts.values()), {5})
            for key in expected:
                self.assertTrue(torch.equal(expected[key], resumed.state_dict()[key]), key)
            for pid, values in expected_optimizer["state"].items():
                for key, value in values.items():
                    self.assertTrue(torch.equal(value, resumed_optimizer.state_dict()["state"][pid][key]))

    def test_answer_shift_excludes_prompt_and_eos(self):
        class Scripted(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None):
                logits = torch.zeros((*input_ids.shape, 10))
                logits[:, 1, 3] = 5
                logits[:, 2, 4] = 4
                logits[:, 3, 9] = -3
                return SimpleNamespace(logits=logits)
        result = conditional_scores(Scripted(), [{"prompt_ids": [1, 2], "answer_ids": [3, 4]}],
                                    9, "cpu", 1, contextlib.nullcontext)[0]
        expected = (torch.logsumexp(torch.tensor([5.] + [0.] * 9), 0) - 5 +
                    torch.logsumexp(torch.tensor([4.] + [0.] * 9), 0) - 4) / 2
        self.assertAlmostEqual(result["loss"], float(expected), places=6)
        self.assertGreater(result["loss_with_eos"], result["loss"])
        self.assertEqual(result["answer_tokens"], 2)

    def test_log_recovery_truncates_only_uncommitted_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            for i in range(3):
                append_json(path, {"cursor": i + 1, "examples": [str(i)], "row_sha256": digest([str(i)])})
            recover_exposure_log(path, 2)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            with self.assertRaises(ValueError):
                recover_exposure_log(path, 3)


if __name__ == "__main__":
    unittest.main()
