import copy
import json
import importlib.util
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from spacing_rerun.acquisition import (SOURCE_QA_ACQUISITION_POLICY, acquisition_identity,
                                      acquisition_key, acquisition_similarity, validate_acquisition_bundle,
                                      GRID_TRAJECTORY_POLICY)
from spacing_rerun.analysis import validate_analysis_methods
from spacing_rerun.common import digest, write_json
from spacing_rerun.data import REVISION, normalize
from spacing_rerun.encoding import build_examples, encode_probe
from spacing_rerun.evaluation import evaluate
from spacing_rerun.grounding import build_grounded_registry
from spacing_rerun.prepare import load_prepared, validate_config
from spacing_rerun.schedule import compile_schedule, stage1_epoch
from spacing_rerun.teaching import SOURCE_QA_POLICY
from spacing_rerun.units import build_units, schedule_records


ROOT = Path(__file__).resolve().parents[1]


def rehash(payload):
    payload["sha256"] = digest({k: v for k, v in payload.items() if k != "sha256"})


def source_fixture():
    grounding = json.loads((ROOT / "data/development-grounded-declarations.json").read_text())
    facts = [{"id": pid, "event": source["event_id"], "role": source["role"],
              "statement": source["source_statement"], "answer": answer,
              "question": "Unseen question " + pid, "aliases": [answer]}
             for source in grounding["source_units"]
             for pid, answer in zip(source["probe_ids"], source["canonical_answers"])]
    facts += [{"id": f"{role}{i}", "event": role, "role": role, "statement": f"{role} source statement {i}.",
               "question": f"{role} unseen question {i}?", "answer": "Bob", "aliases": ["Bob"]}
              for role, n in (("qa", 64), ("control", 62)) for i in range(n)]
    grounding["roles_sha256"] = digest({f["event"]: f["role"] for f in facts})
    rehash(grounding)
    registry = build_grounded_registry(facts, build_units(facts), grounding)
    sources = {s["unit_id"]: s for s in grounding["source_units"]}
    units = []
    records = []
    for i, unit in enumerate(u for u in registry["units"] if u["role"] == "old"):
        originals = [sources[sid] for sid in unit["source_unit_ids"]]
        answers = sorted({a for s in originals for a in s["canonical_answers"]})
        units.append({"unit_id": unit["id"], "event": unit["event"], "statement": unit["statement"],
                      "answer_targets": answers,
                      "source_assertions": [{"statement": s["source_statement"],
                                             "answer_targets": sorted(set(s["canonical_answers"])),
                                             "status": s["status"], "evidence": s["evidence"]}
                                            for s in originals]})
        for j, answer in enumerate(answers):
            records.append({"id": acquisition_identity(unit["id"], answer), "unit_id": unit["id"],
                            "event": unit["event"], "answer": answer,
                            "questions": [f"Source slot {i}-{j}: give its value?", f"Name value for slot {i}-{j}?"],
                            "evidence": copy.deepcopy(originals[0]["evidence"]), "rationale": "Synthetic structural test fixture.",
                            "review_status": "approved_development"})
    source_input = {"schema": "source-only-acquisition-authoring-input-v1", "dataset_revision": REVISION,
                    "grounding_bundle_sha256": grounding["sha256"], "input_scope": "old source units only",
                    "units": units, "factsheets": {event: grounding["factsheets"][event]
                                                   for event in {u["event"] for u in units}}}
    rehash(source_input)
    bundle = {"schema": "spacing-acquisition-qa-v1", "mode": "development",
              "grounding_bundle_sha256": grounding["sha256"], "source_input_sha256": source_input["sha256"],
              "authoring_inputs": ["source_assertions", "source_answer_labels", "event_factsheets"],
              "evaluation_question_text_included": False, "model_outcomes_used": False,
              "independent_agent_review_complete": True, "independent_agent_reviewer": "Test fixture reviewer",
              "records": records}
    rehash(bundle)
    return facts, registry, grounding, source_input, bundle


class Tokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class AcquisitionTests(unittest.TestCase):
    def validate(self, fixture, mode="development"):
        facts, registry, grounding, source_input, bundle = fixture
        validate_acquisition_bundle(bundle, source_input, facts, registry, grounding, mode=mode)

    def test_complete_source_targets_preserve_all_probes_and_conflicts(self):
        fixture = source_fixture()
        facts, registry, grounding, _, bundle = fixture
        before = copy.deepcopy(facts)
        self.validate(fixture)
        self.assertEqual(facts, before)
        self.assertEqual(len(bundle["records"]), 78)
        self.assertEqual(len({r["unit_id"] for r in bundle["records"]}), 73)
        self.assertTrue(any(s["status"].startswith("preserve_source") for s in grounding["source_units"]))
        self.assertEqual({(r["unit_id"], normalize(r["answer"])) for r in bundle["records"]},
                         {(u["id"], a) for u in registry["units"] if u["role"] == "old" for a in u["answer_labels"]})

    def test_analysis_rejects_mixed_acquisition_without_requiring_identical_source_bundles(self):
        legacy = {"mode": "development", "rehearsal_unit_policy": "grounded_assertion_v1"}
        source_qa = dict(legacy, acquisition_policy=SOURCE_QA_ACQUISITION_POLICY,
                         manifest_sha256="replicate-one", source_qa_bundle_sha256="authored-for-old-events-one")
        with self.assertRaisesRegex(ValueError, "different acquisition policies"):
            validate_analysis_methods([legacy, source_qa])
        validate_analysis_methods([legacy, dict(legacy, acquisition_policy="declaration_only_v1")])
        validate_analysis_methods([source_qa, dict(source_qa, manifest_sha256="replicate-two",
                                                 source_qa_bundle_sha256="authored-for-old-events-two")])

    def test_loss_or_duplication_of_answer_targets_is_rejected(self):
        for duplicate in (False, True):
            fixture = source_fixture()
            bundle = fixture[-1]
            if duplicate:
                # Change questions too, so coverage validation catches duplication.
                extra = copy.deepcopy(bundle["records"][0])
                extra["questions"] = ["Extra source form one?", "Extra source form two?"]
                bundle["records"].append(extra)
            else:
                bundle["records"].pop()
            rehash(bundle)
            with self.assertRaisesRegex(ValueError, "lost or duplicated"):
                self.validate(fixture)

    def test_review_and_hashes_are_required(self):
        for field in ("independent_agent_review_complete", "record_review", "source_hash", "bundle_hash"):
            fixture = source_fixture()
            source, bundle = fixture[-2:]
            if field == "independent_agent_review_complete":
                bundle[field] = False
                rehash(bundle)
                message = "independent review"
            elif field == "record_review":
                bundle["records"][0]["review_status"] = "unreviewed"
                rehash(bundle)
                message = "Unreviewed"
            elif field == "source_hash":
                source["units"][0]["statement"] += " mutation"
                message = "source input was modified"
            else:
                bundle["records"][0]["answer"] += " mutation"
                message = "bundle was modified"
            with self.assertRaisesRegex(ValueError, message):
                self.validate(fixture)
        with self.assertRaisesRegex(ValueError, "development-only"):
            self.validate(source_fixture(), mode="confirmation")

    def test_exact_canonical_and_paraphrase_leakage_is_rejected_for_all_roles(self):
        for role in ("old", "new", "control"):
            for field in ("question", "paraphrase"):
                fixture = source_fixture()
                facts, _, _, _, bundle = fixture
                fact = next(f for f in facts if f["role"] == role)
                fact[field] = "  " + bundle["records"][0]["questions"][0].upper() + "  "
                with self.assertRaisesRegex(ValueError, "Held-out evaluation"):
                    self.validate(fixture)

    def test_source_role_answer_evidence_and_declaration_context_tampering(self):
        for mutation, message in (("role", "membership"), ("answer", "membership"),
                                  ("evidence", "pinned source"), ("context", "declaration")):
            fixture = source_fixture()
            _, registry, _, _, bundle = fixture
            record = bundle["records"][0]
            if mutation == "role":
                record["unit_id"] = next(u["id"] for u in registry["units"] if u["role"] == "new")
            elif mutation == "answer":
                record["answer"] = "Unsupported source answer"
            elif mutation == "evidence":
                record["evidence"][0]["quote"] += " invented"
            else:
                record["questions"][0] = next(u["statement"] for u in registry["units"]
                                              if u["id"] == record["unit_id"]) + " Which value?"
            rehash(bundle)
            with self.assertRaisesRegex(ValueError, message):
                self.validate(fixture)

    def test_acquisition_examples_use_answer_only_masks_and_exact_budget(self):
        facts = [{"id": role, "event": role, "role": role, "statement": "Bob arrived.",
                  "question": "Who arrived?", "answer": "Bob", "aliases": ["Bob"]}
                 for role in ("old", "new", "qa", "control")]
        record = {"id": "target", "answer": "Bob", "questions": ["Name the arrival?", "Who came?"]}
        config = {"loss_tokens_per_example": 36, "context_length": 96, "eval_seed": 7}
        baseline = build_examples(copy.deepcopy(facts), Tokenizer(), list(range(1, 80)) * 20, config)
        examples = build_examples(facts, Tokenizer(), list(range(1, 80)) * 20, config,
                                  acquisition_records=[record])
        self.assertEqual({k: v for k, v in examples.items() if not k.startswith("acq/")}, baseline)
        for variant, question in enumerate(record["questions"]):
            example = examples[acquisition_key(record, variant)]
            probe = encode_probe(Tokenizer(), question, "Bob", 96)
            begin = len(probe["prompt_ids"])
            self.assertEqual(example["labels"][:begin], [-100] * begin)
            self.assertEqual(example["labels"][begin:begin + len(probe["answer_ids"])], probe["answer_ids"])
            self.assertEqual(example["labels"][begin + len(probe["answer_ids"])], 0)
            self.assertEqual(sum(t != -100 for t in example["labels"][1:]), 36)
            self.assertEqual(example["content_tokens"], len(probe["answer_ids"]) + 1)

    def test_every_declaration_and_target_once_per_epoch_with_rotating_variants_and_continuous_teaching(self):
        facts, registry, _, _, bundle = source_fixture()
        teaching = [{"id": str(i), "role": "qa"} for i in range(225)]
        records = schedule_records(facts, registry, teaching)
        config = {"batch_size": 16, "qa_per_step": 4, "order_seed": 17,
                  "qa_teaching_policy": SOURCE_QA_POLICY, "acquisition_policy": SOURCE_QA_ACQUISITION_POLICY}
        planned = []
        for epoch in range(6):
            rows = stage1_epoch(records, epoch, config, bundle["records"])
            self.assertEqual(len(rows), 13)
            self.assertTrue(all(len(row) == 16 and sum(k.startswith("qa/") for k in row) == 4 for row in rows))
            counts = Counter(k for row in rows for k in row if k.startswith("old/"))
            self.assertEqual(len(counts), 73)
            self.assertEqual(set(counts.values()), {1})
            actual = [k for row in rows for k in row if k.startswith("acq/")]
            self.assertEqual(Counter(actual), Counter(acquisition_key(r, epoch % 2) for r in bundle["records"]))
            planned.extend(rows)
        teaching_keys = [k for row in planned for k in row if k.startswith("qa/")]
        for begin in range(0, len(teaching_keys) - 224, 225):
            self.assertEqual(len(set(teaching_keys[begin:begin + 225])), 225)
        epoch, cursor = 2, 7
        resumed = stage1_epoch(records, epoch, config, bundle["records"])[cursor:] + [
            row for e in range(epoch + 1, 6) for row in stage1_epoch(records, e, config, bundle["records"])]
        self.assertEqual(resumed, planned[epoch * 13 + cursor:])
        with self.assertRaisesRegex(ValueError, "requires its reviewed records"):
            stage1_epoch(records, 0, config)

    def test_stage2_declarations_and_all_five_arm_streams_are_unchanged(self):
        facts, registry, _, _, _ = source_fixture()
        records = schedule_records(facts, registry)
        baseline = json.loads((ROOT / "configs/calibration-grounded.json").read_text())
        revised = json.loads((ROOT / "configs/calibration-sourceqa.json").read_text())
        old = compile_schedule(records, baseline)
        new = compile_schedule(records, revised)
        self.assertEqual(old["arms"], new["arms"])
        self.assertEqual(old["review_times"], new["review_times"])
        self.assertFalse(any(k.startswith("acq/") for rows in new["arms"].values() for row in rows for k in row))
        for arm in ("UNI", "EXP", "MASS"):
            counts = Counter(k for row in new["arms"][arm] for k in row if k.startswith("old/"))
            self.assertEqual(len(counts), 73)
            self.assertEqual(set(counts.values()), {4})
        validate_config(revised)
        with self.assertRaisesRegex(ValueError, "development-only"):
            validate_config(dict(revised, mode="confirmation"))

    def test_both_acquisition_variants_and_low_overlap_are_separate_from_gate(self):
        class Model:
            training = True

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

        facts = [{"id": key, "unit_id": key, "event": "event", "role": "old", "answer": "Bob",
                  "aliases": ["Bob"], "statement": "Bob arrived.", "question": question,
                  "mcq": {"choices": ["Bob", "Ann"], "correct": 0}}
                 for key, question in (("high", "Who arrived in Town?"), ("low", "Which town hosted the carnival?"))]
        records = [{"id": key, "unit_id": key, "event": "event", "answer": "Bob", "questions": questions}
                   for key, questions in (("high", ["Who arrived in Town?", "Name the person arriving in Town?"]),
                                           ("low", ["Provide carnival municipality identifier.", "Supply carnival location name."]))]
        similarity = acquisition_similarity(facts, records)
        self.assertEqual(similarity["threshold"], .7)
        self.assertEqual(similarity["probes"][0]["max_jaccard"], 1)
        self.assertLess(similarity["probes"][1]["max_jaccard"], .7)
        self.assertEqual(similarity, acquisition_similarity(facts, records))
        self.assertEqual(similarity["sha256"], digest({k: v for k, v in similarity.items() if k != "sha256"}))
        config = {"loss_tokens_per_example": 36, "context_length": 96, "eval_context_length": 256,
                  "eval_seed": 1, "eval_batch_size": 2, "generation_batch_size": 2, "max_answer_tokens": 8}
        build_examples(facts, Tokenizer(), list(range(1, 80)) * 20, config)
        def scores(model, probes, *args):
            return [{"loss": .5, "loss_with_eos": .5, "answer_nll_sum": .5, "answer_tokens": 1, "eos_nll": .5}
                    for _ in probes]
        with patch("spacing_rerun.evaluation.conditional_scores", side_effect=scores), \
             patch("spacing_rerun.evaluation.generated_answers", side_effect=[
                 [{"prediction": "wrong", "generation_terminated": 1}] * 2,
                 [{"prediction": "Bob", "generation_terminated": 1}] * 4]):
            model = Model()
            result = evaluate(model, Tokenizer(), facts, config, "cpu", None,
                              acquisition_records=records, similarity_report=similarity)
        self.assertEqual(result["variants"]["canonical"]["aggregate"]["old"]["exact_match_event_macro"], 0)
        diagnostic = result["acquisition_qa_diagnostic"]
        self.assertEqual(diagnostic["question_variants"], 4)
        self.assertEqual(diagnostic["answer_targets"], 2)
        self.assertEqual(diagnostic["aggregate"]["exact_match_event_macro"], 1)
        self.assertEqual({r["question_variant"] for r in diagnostic["facts"]}, {0, 1})
        self.assertEqual(result["acquisition_qa_similarity_diagnostic"]["low_similarity_probes"], 1)
        self.assertEqual(result["acquisition_qa_similarity_diagnostic"]["aggregate"]["exact_match_event_macro"], 0)
        self.assertTrue(model.training)

    @unittest.skipUnless(importlib.util.find_spec("hf_olmo"), "Optional actual legacy OLMo runtime unavailable")
    def test_actual_hf_olmo_acquisition_checkpoint_and_declaration_continuation(self):
        import torch
        from hf_olmo.configuration_olmo import OLMoConfig
        from hf_olmo.modeling_olmo import OLMoForCausalLM
        from olmo.config import ModelConfig
        from olmo.model import OLMo
        from spacing_rerun.training import run_acquisition, run_arm, evaluate as runtime_evaluate

        class RuntimeTokenizer:
            eos_token_id = 0

            def encode(self, text, add_special_tokens=False):
                return [1 + int(digest(word)[:8], 16) % 126 for word in re.findall(r"\n|[\w]+|[^\w\s]", text)]

            def decode(self, ids, skip_special_tokens=False):
                return "unmatched generated answer"

        def factory(manifest, prepared, device):
            model_config = ModelConfig(d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                                       max_sequence_length=256, vocab_size=128, embedding_size=128,
                                       rope=True, alibi=False, attention_dropout=0., residual_dropout=0.,
                                       embedding_dropout=0., init_device="cpu")
            model = OLMoForCausalLM(OLMoConfig(**model_config.asdict()), OLMo(model_config))
            return model.to(device), RuntimeTokenizer()

        facts, registry, grounding, source_input, bundle = source_fixture()
        config = json.loads((ROOT / "configs/calibration-sourceqa.json").read_text())
        config.update(buffer_steps=1, primary_delay=1, eval_delays=[1], warmup_steps=0,
                      max_answer_tokens=1, microbatch_size=16, acquisition_exposures=[1])
        tokenizer = RuntimeTokenizer()
        for fact in facts:
            fact["mcq"] = {"choices": [fact["answer"], "incorrect distractor"], "correct": 0}
        examples = build_examples(facts, tokenizer, list(range(1, 128)) * 100, config,
                                  acquisition_records=bundle["records"])
        training_records = schedule_records(facts, registry)
        schedule = compile_schedule(training_records, config)
        partition = {"development": sorted({f["event"] for f in facts}), "confirmation": []}
        rehash(partition)
        grounding["partition_sha256"] = partition["sha256"]
        rehash(grounding)
        registry["grounding_bundle_sha256"] = grounding["sha256"]
        rehash(registry)
        source_input["grounding_bundle_sha256"] = grounding["sha256"]
        rehash(source_input)
        bundle["grounding_bundle_sha256"] = grounding["sha256"]
        bundle["source_input_sha256"] = source_input["sha256"]
        rehash(bundle)
        # Reconstruct the original source registry from preserved source fields.
        original = copy.deepcopy(facts)
        for fact in original:
            if fact["role"] in ("old", "new"):
                fact["statement"] = fact["source_statement"]
                fact["unit_id"] = fact["source_unit_id"]
        source_registry = build_units(original)
        split = {"facts": facts, "roles": {f["event"]: f["role"] for f in facts}}
        rehash(split)
        manifest = {"mode": "development", "config": config, "split": split, "partition": partition,
                    "rehearsal_unit_policy": config["rehearsal_unit_policy"], "unit_registry": registry,
                    "source_unit_registry": source_registry, "schedule_sha256": schedule["sha256"],
                    "examples_sha256": digest(examples), "generic_eval": [],
                    "grounding_audit": {"bundle_sha256": grounding["sha256"],
                                        "source_factsheets_sha256": grounding["source_factsheets_sha256"]},
                    "qa_teaching_pool": {"policy": SOURCE_QA_POLICY, "records": [], "events": []},
                    "acquisition_policy": SOURCE_QA_ACQUISITION_POLICY, "acquisition_qa_pool": bundle,
                    "acquisition_qa_audit": {"source_input_sha256": source_input["sha256"],
                                             "bundle_sha256": bundle["sha256"], "answer_targets": 78,
                                             "stage2_acquisition_qa_dose": 0}}
        manifest["acquisition_qa_similarity"] = acquisition_similarity(facts, bundle["records"])
        # The fixture uses canonical teaching to avoid requiring source QA rows.
        config["qa_teaching_policy"] = "canonical_roots_v1"
        del manifest["qa_teaching_pool"]
        schedule = compile_schedule(training_records, config)
        manifest["schedule_sha256"] = schedule["sha256"]
        rehash(manifest)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                prepared = Path(directory) / "prepared"
                for name, payload in (("manifest.json", manifest), ("schedule.json", schedule),
                                      ("examples.json", examples), ("grounding-declarations.json", grounding),
                                      ("acquisition-qa.json", bundle), ("acquisition-source-input.json", source_input),
                                      ("acquisition-similarity.json", manifest["acquisition_qa_similarity"])):
                    write_json(prepared / name, payload)
                load_prepared(prepared)
                below = Path(directory) / "below-gate"
                run_acquisition(prepared, below, device="cpu", fixed_exposures=1, model_factory=factory)
                below_decision = json.loads((below / "acquisition_decision.json").read_text())
                self.assertFalse(below_decision["acquisition_usable"])
                with self.assertRaisesRegex(ValueError, "acquisition failed"):
                    run_arm(prepared, Path(directory) / "rejected-UNI", below / "stage1.pt", "UNI",
                            device="cpu", model_factory=factory)

                def controlled_gate(score):
                    def evaluate_with_controlled_gate(*args, **kwargs):
                        result = runtime_evaluate(*args, **kwargs)
                        # Actual hf_olmo training/inference still runs. Only the
                        # gate metric is controlled to test workflow transitions.
                        result["variants"]["canonical"]["aggregate"]["old"]["exact_match_event_macro"] = score
                        result["test_controlled_gate_em"] = score
                        return result
                    return evaluate_with_controlled_gate

                above = Path(directory) / "above-gate"
                with patch("spacing_rerun.training.evaluate", side_effect=controlled_gate(.95)):
                    run_acquisition(prepared, above, device="cpu", fixed_exposures=1, model_factory=factory)
                self.assertFalse(json.loads((above / "acquisition_decision.json").read_text())["acquisition_usable"])
                stage1 = Path(directory) / "stage1"
                with patch("spacing_rerun.training.evaluate", side_effect=controlled_gate(.5)):
                    run_acquisition(prepared, stage1, device="cpu", fixed_exposures=1, model_factory=factory)
                decision = json.loads((stage1 / "acquisition_decision.json").read_text())
                self.assertTrue(decision["acquisition_usable"])
                self.assertFalse(decision["confirmation_ready"])
                self.assertEqual(set(decision["acquisition_qa_target_doses"].values()), {1})
                self.assertEqual(decision["acquisition_qa_tokens_total"], 78 * 36)
                self.assertEqual({tuple(v) for v in decision["acquisition_qa_variant_doses"].values()}, {(1, 0)})
                evaluation = json.loads((stage1 / "evaluations/acquisition-E001.json").read_text())
                self.assertEqual(evaluation["acquisition_qa_diagnostic"]["question_variants"], 156)
                self.assertEqual(evaluation["acquisition_qa_diagnostic"]["seen_question_variants"], 78)
                self.assertEqual(evaluation["acquisition_qa_diagnostic"]["unseen_question_variants"], 78)
                self.assertFalse(evaluation["acquisition_qa_diagnostic"]["in_sample_acquisition_questions"])
                stage1_logs = [json.loads(line) for line in (stage1 / "exposures.jsonl").read_text().splitlines()]
                self.assertEqual(len(stage1_logs), 13)
                self.assertEqual(stage1_logs[-1]["new_qa_tokens_total"], 13 * 4 * 36)
                output = Path(directory) / "UNI"
                run_arm(prepared, output, stage1 / "stage1.pt", "UNI", device="cpu", model_factory=factory)
                logs = [json.loads(line) for line in (output / "exposures.jsonl").read_text().splitlines()]
                counts = Counter(k for row in logs for k in row["examples"] if k.startswith("old/"))
                self.assertEqual(len(counts), 73)
                self.assertEqual(set(counts.values()), {4})
                self.assertTrue(all(not row["acquisition_qa_examples"] for row in logs))
                self.assertTrue(all(row["acquisition_qa_tokens_total"] == 78 * 36 for row in logs))

                # Run the actual model/Adam/RNG trajectory through every frozen
                # dose. Controlled metrics test gates, not learned accuracy.
                from spacing_rerun.acquisition_grid import (checkpoint_digest, choose_acquisition_grid_checkpoint)
                from spacing_rerun.training import checkpoint_state_equal
                from test_acquisition_grid import grid_fixture
                grid_manifest = copy.deepcopy(manifest)
                grid_manifest["config"].update(acquisition_trajectory_policy=GRID_TRAJECTORY_POLICY,
                                               acquisition_exposures=[2, 3, 4, 6, 8], max_attempts=6)
                grid_schedule = compile_schedule(training_records, grid_manifest["config"])
                grid_manifest["schedule_sha256"] = grid_schedule["sha256"]
                rehash(grid_manifest)
                grid_prepared = Path(directory) / "grid-prepared"
                for path in prepared.iterdir():
                    write_json(grid_prepared / path.name, json.loads(path.read_text()))
                write_json(grid_prepared / "manifest.json", grid_manifest)
                write_json(grid_prepared / "schedule.json", grid_schedule)
                grid_output = Path(directory) / "grid-output"
                with self.assertRaisesRegex(ValueError, "cannot be truncated"):
                    run_acquisition(grid_prepared, grid_output, device="cpu", fixed_exposures=2, model_factory=factory)
                with patch("spacing_rerun.training.evaluate", side_effect=controlled_gate(.5)):
                    run_acquisition(grid_prepared, grid_output, device="cpu", model_factory=factory)
                grid_decision = json.loads((grid_output / "acquisition_decision.json").read_text())
                self.assertEqual(set(grid_decision["grid_results"]), {"2", "3", "4", "6", "8"})
                self.assertIsNone(grid_decision["selected_E"])
                self.assertFalse(grid_decision["acquisition_usable"])
                self.assertEqual(grid_decision["final_exposures"], 8)
                hashes = {}
                for exposure in [2, 3, 4, 6, 8]:
                    dose = grid_output / f"stage1-E{exposure:03}.pt"
                    hashes[exposure] = checkpoint_digest(dose)
                    state = torch.load(dose, weights_only=False)
                    self.assertTrue(state["model"])
                    self.assertTrue(state["optimizer"]["state"])
                    self.assertEqual(set(state["rng"]), {"python", "numpy", "torch", "cuda"})
                    self.assertEqual(state["progress"]["global_step"], exposure * 13)
                    self.assertEqual(state["progress"]["epoch"], exposure)
                    self.assertEqual(state["progress"]["epoch_cursor"], 0)
                    self.assertTrue(state["progress"]["acquisition_usable"])
                    self.assertEqual(set(state["progress"]["acquisition_qa_target_doses"].values()), {exposure})
                    self.assertEqual({tuple(v) for v in state["progress"]["acquisition_qa_variant_doses"].values()},
                                     {((exposure + 1) // 2, exposure // 2)})
                    self.assertEqual(state["progress"]["acquisition_qa_tokens_total"], exposure * 78 * 36)
                    self.assertEqual(grid_decision["grid_results"][str(exposure)]["checkpoint_sha256"], hashes[exposure])

                # Recover a crash after final atomic checkpoint but before the
                # decision write. Completed recovery must recreate its decision.
                (grid_output / "acquisition_decision.json").unlink()
                run_acquisition(grid_prepared, grid_output, device="cpu", model_factory=factory)
                self.assertEqual(json.loads((grid_output / "acquisition_decision.json").read_text()), grid_decision)
                final_state = torch.load(grid_output / "grid-final.pt", weights_only=False)

                # Replay from the E4 checkpoint in the hash-write crash window.
                # Existing E4/6/8 files must remain byte-identical.
                resume_state = torch.load(grid_output / "stage1-E004.pt", weights_only=False)
                resume_state["progress"].update(status="running", acquisition_usable=False)
                torch.save(resume_state, grid_output / "latest.pt")
                with patch("spacing_rerun.training.evaluate", side_effect=controlled_gate(.5)):
                    run_acquisition(grid_prepared, grid_output, device="cpu", model_factory=factory)
                replayed = torch.load(grid_output / "grid-final.pt", weights_only=False)
                for key in ("model", "optimizer", "rng", "progress"):
                    self.assertTrue(checkpoint_state_equal(final_state[key], replayed[key]), key)
                self.assertEqual({e: checkpoint_digest(grid_output / f"stage1-E{e:03}.pt") for e in hashes}, hashes)

                # A normal full grid continues only through the global selection
                # artifact, and restores the actual selected E2 state.
                rule, nine_decisions = grid_fixture()
                rule["normal_manifest_sha256"][0] = grid_manifest["sha256"]
                rehash(rule)
                nine_decisions[0] = grid_decision
                outputs = [grid_output]
                for i, other in enumerate(nine_decisions[1:], 1):
                    other_output = Path(directory) / f"other-grid-{i}"
                    write_json(other_output / "acquisition_decision.json", other)
                    outputs.append(other_output)
                write_json(Path(directory) / "rule.json", rule)
                selection_path = Path(directory) / "selection.json"
                selection = choose_acquisition_grid_checkpoint(grid_prepared, outputs, Path(directory) / "rule.json", selection_path)
                self.assertEqual(selection["selected_E"], 2)
                with self.assertRaisesRegex(ValueError, "requires a frozen common-dose"):
                    run_arm(grid_prepared, Path(directory) / "grid-guarded-UNI", grid_output / "stage1-E002.pt", "UNI",
                            device="cpu", model_factory=factory)
                run_arm(grid_prepared, Path(directory) / "grid-UNI", grid_output / "stage1-E002.pt", "UNI",
                        device="cpu", model_factory=factory, acquisition_selection=selection_path)
                continued = torch.load(Path(directory) / "grid-UNI/latest.pt", weights_only=False)
                self.assertEqual(continued["progress"]["epoch"], 2)
                self.assertEqual(set(continued["progress"]["acquisition_qa_target_doses"].values()), {2})
                corrupted = copy.deepcopy(bundle)
                corrupted["records"][0]["questions"][0] += " tampered"
                write_json(prepared / "acquisition-qa.json", corrupted)
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    load_prepared(prepared)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
