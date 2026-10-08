import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from spacing_rerun.aa_check import compare_tree, file_identity, run_aa
from spacing_rerun.common import write_json
from spacing_rerun.encoding import packed_example
from spacing_rerun.training import make_optimizer, save_checkpoint, seed_all, train_update


class Tokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [1 + ord(c) % 7 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(map(str, ids))


class AATests(unittest.TestCase):
    def test_full_state_comparison_detects_tensor_optimizer_and_rng_changes(self):
        reference = {"tensor": torch.tensor([1., 2.]), "groups": [{"lr": .001}],
                     "rng": (np.array([1, 2], dtype=np.uint32), torch.tensor([3, 4]))}
        matching = {"tensor": reference["tensor"].clone(), "groups": [{"lr": .001}],
                    "rng": (np.array([1, 2], dtype=np.uint32), torch.tensor([3, 4]))}
        self.assertTrue(compare_tree(reference, matching)["bitwise_equal"])
        matching["groups"][0]["lr"] = .002
        matching["tensor"][1] += 1
        matching["rng"][0][1] += 1
        result = compare_tree(reference, matching)
        self.assertFalse(result["bitwise_equal"])
        self.assertEqual(result["mismatch_count"], 3)

    @unittest.skipUnless(importlib.util.find_spec("hf_olmo"), "Optional actual legacy OLMo runtime unavailable")
    def test_actual_hf_olmo_two_update_replay_and_failure_preserve_sources(self):
        from hf_olmo.configuration_olmo import OLMoConfig
        from hf_olmo.modeling_olmo import OLMoForCausalLM
        from olmo.config import ModelConfig
        from olmo.model import OLMo

        def factory(manifest, prepared, device):
            config = ModelConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                                 max_sequence_length=64, vocab_size=10, embedding_size=10,
                                 eos_token_id=9, pad_token_id=9, rope=True, alibi=False,
                                 attention_dropout=.2, residual_dropout=.2, embedding_dropout=.2, init_device="cpu")
            model = OLMoForCausalLM(OLMoConfig(**config.asdict()), OLMo(config))
            return model.to(device), Tokenizer()

        config = {"train_seed": 311, "learning_rate": .001, "warmup_steps": 0, "microbatch_size": 1,
                  "eval_batch_size": 2, "generation_batch_size": 2, "max_answer_tokens": 4}
        manifest = {"sha256": "synthetic-aa-manifest", "config": config, "mode": "development",
                    "split": {"facts": [{"id": "old-probe", "role": "old",
                                         "probe": {"prompt_ids": [1, 2], "answer_ids": [3, 4]}}]}}
        examples = {"old/old": packed_example(Tokenizer(), "ab", "old/old", [1, 2, 3] * 100, 8, 40),
                    "qa/qa": packed_example(Tokenizer(), "Who?", "qa/qa", [1, 2, 3] * 100, 8, 40, answer="Bob")}
        rows = [["old/old", "qa/qa"], ["qa/qa", "old/old"]]
        schedule = {"arms": {"UNI": rows}}
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prepared = root / "prepared"
                for name, payload in (("manifest.json", manifest), ("schedule.json", schedule), ("examples.json", examples)):
                    write_json(prepared / name, payload)
                seed_all(113)
                model, _ = factory(manifest, prepared, "cpu")
                optimizer = make_optimizer(model, config)
                train_update(model, optimizer, rows[0], examples, config, 0, "cpu", contextlib.nullcontext)
                source = root / "shared.pt"
                save_checkpoint(source, model, optimizer,
                                {"status": "stage1_complete", "epoch": 1, "epoch_cursor": 0,
                                 "global_step": 1, "acquisition_usable": True}, manifest["sha256"])
                source_before = file_identity(source)
                with patch("spacing_rerun.aa_check.load_prepared", return_value=(manifest, schedule, examples)):
                    result = run_aa(prepared, source, root / "passed", device="cpu", model_factory=factory)
                self.assertTrue(result["passed"])
                self.assertTrue(result["source_artifacts_unchanged"])
                self.assertTrue(all(c["bitwise_equal"] for c in result["comparisons"].values()))
                self.assertGreater(result["comparisons"]["optimizer"]["tensors_compared"], 0)
                self.assertEqual(result["midpoint_progress"]["cursor"], 1)
                self.assertEqual(result["midpoint_progress"]["continuation_cursor"], 1)
                self.assertEqual(result["midpoint_progress"]["global_step"], 2)
                self.assertEqual(result["midpoint_progress"]["new_qa_tokens_total"], 8)
                self.assertEqual(result["final_progress"]["global_step"], 3)
                self.assertEqual(result["final_progress"]["new_qa_tokens_total"], 16)
                self.assertEqual(result["final_progress"]["last_old_update"], {"old": 3})
                self.assertEqual(result["final_progress"]["last_old_newqa_clock"], {"old": 16})
                self.assertEqual(result["realized_doses"], {"old/old": 2, "qa/qa": 2})
                self.assertEqual(source_before, file_identity(source))
                self.assertEqual(result, json.loads((root / "passed/report.json").read_text()))
                with self.assertRaisesRegex(ValueError, "never overwrite"):
                    run_aa(prepared, source, root / "passed", device="cpu", model_factory=factory)

                calls = 0
                def inject_restore_difference(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        with torch.no_grad():
                            next(args[0].parameters()).add_(.01)
                    return train_update(*args, **kwargs)

                with patch("spacing_rerun.aa_check.load_prepared", return_value=(manifest, schedule, examples)), \
                     patch("spacing_rerun.aa_check.train_update", side_effect=inject_restore_difference):
                    failed = run_aa(prepared, source, root / "failed", device="cpu", model_factory=factory)
                self.assertFalse(failed["passed"])
                self.assertEqual(failed["status"], "failed")
                self.assertFalse(failed["comparisons"]["model"]["bitwise_equal"])
                self.assertTrue(failed["source_artifacts_unchanged"])
                self.assertEqual(source_before, file_identity(source))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
