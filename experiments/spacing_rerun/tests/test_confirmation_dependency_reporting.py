"""Synthetic saved-output fixtures; these never certify actual confirmation data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.spacing_rerun.tests.test_confirmation_dependency import source_fixture, seal
from spacing_rerun.common import digest
from spacing_rerun.confirmation_dependency import (
    _bind_prepared_roots, _final_membership, _saved_evaluation, build_saved_dependency_report,
    d8_prediction_table, executed_presentations, exposure_covariates, prepared_source_metadata,
    saved_bundle_diagnostics, training_record_provenance, FROZEN_POLICY_SCHEMA,
    primary_aggregate_binding,
)
from spacing_rerun.evaluation import aggregate
from spacing_rerun.schedule import ARMS, stage1_epoch
from spacing_rerun.units import schedule_records


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def reporting_fixture(directory):
    packet, _ = source_fixture()
    sources = packet["source_units"]
    sources[2]["event"], sources[3]["event"] = "e2", "e3"
    packet["groups"][2]["event"], packet["groups"][3]["event"] = "e2", "e3"
    for i, answer in ((0, "an elaborate trick"), (1, "optical illusion")):
        sources[i]["canonical_answer_labels"] = [answer]
        sources[i]["canonical_probes"][0]["source_answer_label"] = answer
    seal(packet)
    rec = {"id": "chunk04/D8", "event": "e1", "records": [
        {"group": packet["groups"][i], "sources": [sources[i]]} for i in (0, 1)]}
    membership = _final_membership([packet], [rec], ["u0", "u1"])
    roles = ["old", "old", "new", "qa"]
    facts = [{"id": f"f{i}", "event": sources[i]["event"], "role": roles[i],
              "unit_id": f"g{i}" if i < 3 else "u3", "statement_ids": [i],
              "answer": sources[i]["canonical_answer_labels"][0]} for i in range(4)]
    targets = [{"id": f"t{i}", "unit_id": f"g{i}", "event": "e1", "answer": facts[i]["answer"],
                "questions": [f"synthetic form {i}A", f"synthetic form {i}B"]} for i in (0, 1)]
    teaching = [{"id": key, "canonical_probe_id": "f3", "event": "e3", "role": "qa"} for key in ("f3", "f3copy")]
    keys = ["old/g0", "old/g1", "new/g2", "gen/g0", "gen/g1", "qa/f3", "qa/f3copy", "filler"]
    keys += [f"acq/t{i}/{v}" for i in (0, 1) for v in (0, 1)]
    examples = {key: {"attention_mask": [1] * 5, "content_tokens": 3 if key.startswith(("qa/", "acq/")) else 2,
                      "loss_tokens": 4, "content_sha256": digest(key)} for key in keys}
    config = {"batch_size": 2, "qa_per_step": 1, "order_seed": 10,
              "qa_teaching_policy": "all_source_questions_v1", "acquisition_policy": "source_qa_mixed_v1",
              "stage2_eval_fractions": [.5], "eval_delays": [1]}
    from spacing_rerun.acquisition import SOURCE_QA_ACQUISITION_POLICY
    config["acquisition_policy"] = SOURCE_QA_ACQUISITION_POLICY
    manifest = seal({"split": {"facts": facts}, "config": config, "examples_sha256": digest(examples),
                    "confirmation_audit": {"sha256": "synthetic-audit", "probe_map": {"probes": {
                        f"p{i}": {"original_probe_id": f"f{i}", "event": sources[i]["event"], "source_unit_id": f"u{i}"}
                        for i in range(4)}}},
                    "acquisition_qa_pool": {"records": targets}, "qa_teaching_pool": {"records": teaching},
                    "unit_registry": {"units": [{"id": f"g{i}", "event": sources[i]["event"], "role": roles[i],
                        "canonical_probe_id": f"f{i}"} for i in range(3)]}})
    binding = _bind_prepared_roots(prepared_source_metadata(manifest), membership)
    provenance = training_record_provenance(manifest, examples, membership)
    prepared = {"replicate_id": "r0", "manifest_sha256": manifest["sha256"], "binding": binding,
                "training_provenance": provenance}
    from spacing_rerun.confirmation_lane import reporting_policy
    policy = {"membership": membership, "canonical_bindings": [prepared],
              "exception_lanes": [], "lane_reporting_policy": reporting_policy()}
    schedule = {"stage2_steps": 2, "arms": {arm: [["old/g0", "qa/f3"], ["old/g1", "qa/f3copy"],
                                                   ["new/g2", "qa/f3"]] for arm in ARMS}}
    schedule["arms"]["GEN"] = [["gen/g0", "qa/f3"], ["gen/g1", "qa/f3copy"], ["new/g2", "qa/f3"]]
    schedule["arms"]["NONE"] = [["new/g2", "qa/f3"], ["new/g2", "qa/f3copy"], ["new/g2", "qa/f3"]]
    training = schedule_records(facts, manifest["unit_registry"], teaching)
    shared_rows = stage1_epoch(training, 0, config, targets)
    def exposure_rows(rows, offset, shared=False):
        return [{"global_step": offset + i + 1, "examples": row, "row_sha256": digest(row),
                 "phase": "stage1" if shared else "stage2" if i < 2 else "buffer", "phase_step": i,
                 "example_content_sha256": [examples[k]["content_sha256"] for k in row],
                 "loss_tokens": sum(examples[k]["loss_tokens"] for k in row)} for i, row in enumerate(rows)]
    shared_logs = exposure_rows(shared_rows, 0, True)
    def save_logs(path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in values))
    save_logs(directory / "stage1/exposures.jsonl", shared_logs)
    def evaluation(step):
        rows = [{"id": f["id"], "unit_id": f["unit_id"], "event": f["event"], "role": f["role"],
                 "loss": .25, "exact_match": int(f["id"] != "f1"),
                 "prediction": "trick" if f["id"] == "f1" else f["answer"]} for f in facts if f["role"] != "qa"]
        acquisition = [{"id": f"t{i}/{v}", "target_id": f"t{i}", "question_variant": v,
                        "unit_id": f"g{i}", "event": "e1", "role": "old", "loss": .5,
                        "exact_match": int(i == 0 and v == 0), "prediction": "trick"} for i in (0, 1) for v in (0, 1)]
        qa = [{"id": "f3", "unit_id": "u3", "event": "e3", "role": "qa", "loss": .75,
               "exact_match": 1, "prediction": facts[3]["answer"]}]
        return {"manifest_sha256": manifest["sha256"], "epoch": 1, "stage2_step": step,
                "global_step": len(shared_rows) + step,
                "variants": {v: {"facts": copy.deepcopy(rows), "aggregate": aggregate(rows)} for v in ("canonical", "paraphrase")},
                "acquisition_qa_diagnostic": {"facts": acquisition, "aggregate": aggregate(acquisition)["old"]},
                "qa_teaching_diagnostic": {"facts": qa, "aggregate": aggregate(qa)["qa"]}}
    shared = evaluation(0)
    write(directory / "stage1/evaluations/acquisition-E001.json", shared)
    for arm in ARMS:
        save_logs(directory / arm / "exposures.jsonl", exposure_rows(schedule["arms"][arm], len(shared_rows)))
        for step in range(4):
            value = evaluation(step)
            if step == 1:
                for section in value["variants"].values():
                    for row in section["facts"]:
                        row.pop("exact_match")
                        row.pop("prediction")
                    section["aggregate"] = aggregate(section["facts"])
                value.pop("acquisition_qa_diagnostic")
                value.pop("qa_teaching_diagnostic")
            write(directory / arm / f"evaluations/stage2-{step:03}.json", value)
    return manifest, schedule, examples, policy, shared_logs, evaluation(0)


def retained_reporting_bindings(directory, manifest, schedule, prereg):
    """Synthetic completion pins/primary snapshot; never real source eligibility."""
    root = Path(directory)
    files = sorted([*root.rglob("*.json"), *root.rglob("*.jsonl")])
    completion = seal({"schema": "spacing-confirmation-complete-bundle-v1", "manifest_sha256": manifest["sha256"],
                       "preregistration_sha256": prereg["sha256"], "E": prereg["E"], "arms": {arm: {} for arm in ARMS},
                       "files": [{"path": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()} for file in files]})
    step = schedule["stage2_steps"] + prereg["primary_delay"]
    arms = {arm: next(json.loads(p.read_bytes())["variants"]["canonical"]["aggregate"]
                     for p in (root / arm / "evaluations").glob("*.json") if json.loads(p.read_bytes())["stage2_step"] == step)
            for arm in ARMS}
    binding = primary_aggregate_binding(manifest["sha256"], completion, prereg["primary_delay"], step, arms)
    return completion, binding


class ConfirmationDependencyReportingTests(unittest.TestCase):
    def test_completion_pin_literal_path_is_checked_before_path_normalization(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, schedule, examples, policy, _, _ = reporting_fixture(root)
            completion, _ = retained_reporting_bindings(root, manifest, schedule, seal({"E": 1, "primary_delay": 1}))
            original = next(pin["path"] for pin in completion["files"] if pin["path"].endswith("/EXP/exposures.jsonl"))
            for literal in (original.replace("/EXP/", "//EXP/"), original.replace("/EXP/", "/./EXP/"),
                            original + "/", None, 3, [], {}):
                changed = copy.deepcopy(completion)
                next(pin for pin in changed["files"] if pin["path"] == original)["path"] = literal
                seal(changed)
                with self.subTest(literal=literal), self.assertRaisesRegex(ValueError, "noncanonical"):
                    saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0",
                                             acquisition_E=1, completion=changed)

    def test_saved_reader_rejects_changed_outcome_or_exposure_bytes_from_primary_completion(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, schedule, examples, policy, _, _ = reporting_fixture(root)
            prereg = seal({"E": 1, "primary_delay": 1})
            completion, binding = retained_reporting_bindings(root, manifest, schedule, prereg)
            baseline = saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0", acquisition_E=1,
                                                completion=completion, primary_binding=binding)
            self.assertTrue(all(e["primary_completion_pin_verified"] for e in baseline["consumed_file_evidence"]))
            self.assertEqual(baseline["primary_aggregate_binding_sha256"], binding["sha256"])
            for relative in ("EXP/evaluations/stage2-003.json", "stage1/evaluations/acquisition-E001.json",
                             "EXP/exposures.jsonl", "stage1/exposures.jsonl"):
                file = root / relative
                original = file.read_bytes()
                if relative == "EXP/evaluations/stage2-003.json":
                    changed = json.loads(original)
                    for section in changed["variants"].values():
                        for row in section["facts"]:
                            if row["role"] == "old": row["loss"] += 1
                        section["aggregate"] = aggregate(section["facts"])
                    write(file, changed)
                else:
                    # Same parsed JSON/log records, different consumed bytes.
                    file.write_bytes(original + b" ")
                with self.subTest(file=relative), self.assertRaisesRegex(ValueError, "retained primary completion file identity"):
                    saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0", acquisition_E=1,
                                             completion=completion, primary_binding=binding)
                file.write_bytes(original)
            for mutation in ("missing_pin", "duplicate_pin", "different_path", "missing_files"):
                changed = copy.deepcopy(completion)
                pin = next(p for p in changed["files"] if p["path"].endswith("EXP/evaluations/stage2-003.json"))
                if mutation == "missing_pin": changed["files"].remove(pin)
                if mutation == "duplicate_pin": changed["files"].append(copy.deepcopy(pin))
                if mutation == "different_path": pin["path"] = str(root / "different-outcome.json")
                if mutation == "missing_files": changed.pop("files")
                seal(changed)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0", acquisition_E=1, completion=changed)
            changed = copy.deepcopy(binding)
            changed["arms"]["EXP"]["old"]["loss_event_macro"] += 1
            seal(changed)
            with self.assertRaisesRegex(ValueError, "original primary analysis aggregates"):
                saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0", acquisition_E=1,
                                         completion=completion, primary_binding=changed)

    def test_production_report_rejects_internally_consistent_outcome_change_with_original_primary_receipt(self):
        # Independent R2 reader reproduction; only the four-source final policy
        # validator is stubbed. Completion hashes and saved reader are real.
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, schedule, examples, policy, _, _ = reporting_fixture(root)
            old = manifest["sha256"]
            manifest["confirmation_audit"].update(source_chunks=[], source_catalog={})
            seal(manifest)
            prepared = policy["canonical_bindings"][0]
            prepared["manifest_sha256"] = manifest["sha256"]
            prepared["training_provenance"]["manifest_sha256"] = manifest["sha256"]
            seal(prepared["training_provenance"])
            for file in root.rglob("*.json"):
                value = json.loads(file.read_bytes())
                if value.get("manifest_sha256") == old:
                    value["manifest_sha256"] = manifest["sha256"]
                    write(file, value)
            prereg = seal({"n": 1, "E": 1, "primary_delay": 1, "replicate_manifest_sha256": [manifest["sha256"]]})
            completion, binding = retained_reporting_bindings(root, manifest, schedule, prereg)
            frozen = seal({"schema": FROZEN_POLICY_SCHEMA, "final_membership_frozen": True, "policy": policy,
                           "review_receipt": {}, "review_evidence": {}, "source_review_evidence": [], "validation": {}})
            inputs = [{"path": root, "manifest": manifest, "schedule": schedule, "examples": examples,
                       "completion": completion, "primary_aggregate_binding": binding}]
            with patch("spacing_rerun.confirmation_dependency.validate_final_policy", new=lambda *args, **kwargs: {}):
                before = build_saved_dependency_report(inputs, frozen, prereg)
                primary = next(r["primary"]["estimate"] for r in before["contrasts"] if r["endpoint"] == "delay/1" and
                               r["role"] == "old" and r["variant"] == "transfer/canonical" and r["metric"] == "loss" and r["contrast"] == "H2")
                self.assertEqual(primary, 0)
                file = root / "EXP/evaluations/stage2-003.json"
                value = json.loads(file.read_bytes())
                for section in value["variants"].values():
                    for row in section["facts"]:
                        if row["role"] == "old": row["loss"] += 1
                    section["aggregate"] = aggregate(section["facts"])
                write(file, value)
                with self.assertRaisesRegex(ValueError, "retained primary completion file identity"):
                    build_saved_dependency_report(inputs, frozen, prereg)
            self.assertEqual(inputs[0]["completion"], completion)
            self.assertEqual(inputs[0]["primary_aggregate_binding"], binding)

    def test_streamed_analysis_preserves_exact_original_primary_and_secondary_failure_behavior(self):
        import subprocess, types
        REPO = Path(__file__).resolve().parents[3]
        import spacing_rerun.analysis as analysis
        from spacing_rerun.units import GROUNDED_POLICY, GROUNDED_METRICS
        if (REPO / '.git').exists():
            source = subprocess.check_output(['git', 'show', 'HEAD:experiments/spacing_rerun/spacing_rerun/analysis.py'], cwd=REPO, text=True)
        else:
            # An independently copied review snapshot pins this original Git
            # blob in its input hash manifest, without needing a .git copy.
            source = (REPO / 'review-baselines/analysis_HEAD.py').read_text()
        original = types.ModuleType('spacing_rerun.independent_original_analysis')
        original.__package__ = 'spacing_rerun'
        exec(compile(source, 'HEAD:analysis.py', 'exec'), original.__dict__)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundles = []
            preparations = {}
            for i in range(2):
                p = root / ('rep' + str(i))
                bundles.append(p)
                facts = [{'id': 'r-' + role, 'event': 'event-' + role, 'role': role, 'unit_id': 'unit-' + role} for role in ('old', 'new', 'control')]
                manifest = {'sha256': digest(i), 'mode': 'confirmation', 'confirmation_protocol_schema': 'synthetic-protocol', 'rehearsal_unit_policy': GROUNDED_POLICY, 'config': {}, 'split': {'facts': facts}, 'confirmation_audit': {'sha256': 'synthetic-audit', 'source_chunks': [], 'source_catalog': {}}}
                write(p / 'prepared/manifest.json', manifest)
                write(p / 'prepared/schedule.json', {'stage2_steps': 2})
                preparations[str(p / 'prepared')] = (manifest, {'stage2_steps': 2}, {})
                for j, arm in enumerate(('NONE', 'UNI', 'EXP', 'MASS', 'GEN')):
                    for step in (0, 2, 3):
                        rows = [dict(f, loss=1 + j * 0.02 + i * 0.005) for f in facts]
                        write(p / arm / f'evaluations/stage2-{step}.json', {'manifest_sha256': manifest['sha256'], 'metric_schema': GROUNDED_METRICS, 'stage2_step': step, 'variants': {'canonical': {'facts': rows, 'aggregate': aggregate(rows)}}})
            prereg = seal({'n': 2, 'replicate_manifest_sha256': [preparations[str(p / 'prepared')][0]['sha256'] for p in bundles], 'primary_delay': 1, 'loss_margin': 0.02, 'confirmation_protocol_schema': 'synthetic-protocol', 'inference_claim': 'estimation', 'confirmation_dependency_policy_path': '/synthetic/policy.json', 'confirmation_dependency_policy_sha256': 'f' * 64})
            prereg_path = root / 'prereg.json'
            write(prereg_path, prereg)
            old_path = root / 'original.json'
            new_path = root / 'new.json'
            with patch('spacing_rerun.prepare.load_prepared', side_effect=lambda p: preparations[str(p)]), patch('spacing_rerun.confirmation.verify_confirmation_preregistration'), patch('spacing_rerun.confirmation.validate_confirmation_audit', return_value={'same_role_components': []}), patch.object(original, 'validate_confirmation_completion', return_value={'synthetic_only': True}), patch.object(analysis, 'validate_confirmation_completion', return_value={'synthetic_only': True}), patch('spacing_rerun.confirmation_dependency.verify_frozen_policy', return_value={'synthetic_only': True}), patch('spacing_rerun.confirmation_dependency.verify_preregistered_dependencies'):
                original.summarize_bundles(bundles, old_path, 1, preregistration=prereg_path)
                expected = json.loads(old_path.read_text())
                with patch('spacing_rerun.confirmation_dependency.build_saved_dependency_report', side_effect=RuntimeError('synthetic-secondary-failure')):
                    with self.assertRaisesRegex(RuntimeError, 'synthetic-secondary-failure'):
                        analysis.summarize_bundles(bundles, new_path, 1, preregistration=prereg_path)
                self.assertEqual(json.loads(new_path.read_text()), expected)
                self.assertFalse(new_path.with_name('new.dependency-diagnostics.json').exists())
                diagnostic = seal({'synthetic_only': True, 'status': 'complete_reader_fixture'})
                with patch('spacing_rerun.confirmation_dependency.build_saved_dependency_report', return_value=diagnostic):
                    analysis.summarize_bundles(bundles, new_path, 1, preregistration=prereg_path)
                result = json.loads(new_path.read_text())
                link = result.pop('confirmation_dependency_diagnostics')
                self.assertEqual(result, expected)
                self.assertEqual(link['sha256'], diagnostic['sha256'])
                self.assertEqual(json.loads(Path(link['path']).read_text()), diagnostic)

    def test_saved_target_and_question_variant_are_bound_to_exact_frozen_row_id(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, _, _, policy, _, evaluation = reporting_fixture(Path(name))
            binding = policy["canonical_bindings"][0]["binding"]
            for field, value in (("target_id", "t1"), ("question_variant", 999), ("question_variant", True)):
                changed = copy.deepcopy(evaluation)
                changed["acquisition_qa_diagnostic"]["facts"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "target/forms"):
                    _saved_evaluation(changed, manifest, binding, policy["membership"])

    def test_historical_acquisition_has_exact_epoch_filename_clock_and_consumed_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest, schedule, examples, policy, logs, _ = reporting_fixture(root)
            training = schedule_records(manifest["split"]["facts"], manifest["unit_registry"], manifest["qa_teaching_pool"]["records"])
            extra = stage1_epoch(training, 1, manifest["config"], manifest["acquisition_qa_pool"]["records"])
            for row in extra:
                logs.append({"global_step": len(logs) + 1, "examples": row, "row_sha256": digest(row),
                             "phase": "stage1", "phase_step": len(logs),
                             "example_content_sha256": [examples[k]["content_sha256"] for k in row],
                             "loss_tokens": sum(examples[k]["loss_tokens"] for k in row)})
            (root / "stage1/exposures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs))
            path = root / "stage1/evaluations/acquisition-E001.json"
            saved = json.loads(path.read_text())
            final = dict(copy.deepcopy(saved), epoch=2, global_step=len(logs))
            write(path.with_name("acquisition-E002.json"), final)
            for arm in ARMS:
                logpath = root / arm / "exposures.jsonl"
                arm_logs = [json.loads(line) for line in logpath.read_text().splitlines()]
                for row in arm_logs: row["global_step"] += len(extra)
                logpath.write_text("".join(json.dumps(r) + "\n" for r in arm_logs))
                for file in (root / arm / "evaluations").glob("*.json"):
                    value = json.loads(file.read_text())
                    value.update(epoch=2, global_step=value["global_step"] + len(extra))
                    write(file, value)
            result = saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0")
            evidence = result["acquisition_checkpoint_evidence"][0]
            self.assertEqual(evidence["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(evidence["evaluation_sha256"], digest(saved))
            self.assertEqual(evidence["executed_epoch_prefix_steps"], saved["global_step"])
            for mutation in ("clock", "epoch", "boolean_epoch", "filename", "duplicate", "missing"):
                changed = copy.deepcopy(saved)
                alias = path.with_name("acquisition-unbound.json")
                if mutation == "clock": changed["global_step"] = 1
                if mutation == "epoch": changed["epoch"] = 2
                if mutation == "boolean_epoch": changed["epoch"] = True
                if mutation in ("filename", "duplicate"): write(alias, saved)
                if mutation == "filename": path.unlink()
                elif mutation == "missing":
                    path.unlink()
                    manifest["config"]["acquisition_exposures"] = [1, 2]
                else: write(path, changed)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0")
                alias.unlink(missing_ok=True)
                manifest["config"].pop("acquisition_exposures", None)
                write(path, saved)
            # Fixed-dose confirmation saves only its one frozen milestone.
            with self.assertRaisesRegex(ValueError, "scheduled milestones"):
                saved_bundle_diagnostics(root, manifest, schedule, examples, policy, "r0", acquisition_E=2)

    def test_compact_reader_releases_each_full_audit_before_loading_next_replicate(self):
        import weakref
        from spacing_rerun.confirmation_dependency import load_dependency_bundle
        class Audit(dict): pass
        live, observed = [], []
        refs = [{"path": f"/synthetic/r{i}", "manifest_sha256": digest(i), "confirmation_audit_sha256": "shared-audit",
                 "completion": {}} for i in range(32)]
        policy = {"membership": {}, "exception_lanes": [], "lane_reporting_policy": {},
                  "canonical_bindings": [{"replicate_id": f"r{i}", "manifest_sha256": digest(i)} for i in range(32)]}
        prereg = seal({"n": 32, "E": 1, "primary_delay": 1, "replicate_manifest_sha256": [r["manifest_sha256"] for r in refs]})
        for ref in refs:
            ref["completion"] = seal({"manifest_sha256": ref["manifest_sha256"], "preregistration_sha256": prereg["sha256"],
                                      "E": 1, "arms": {a: {} for a in ARMS}})
            ref["primary_aggregate_binding"] = primary_aggregate_binding(ref["manifest_sha256"], ref["completion"], 1, 3,
                                                                        {a: {} for a in ARMS})
        frozen = seal({"schema": FROZEN_POLICY_SCHEMA, "final_membership_frozen": True, "policy": policy,
                       "review_receipt": {}, "review_evidence": {}, "source_review_evidence": [], "validation": {}})
        def loader(path):
            # Guards are isolated here; this test checks the full-object lifetime.
            self.assertFalse(any(ref() is not None for ref in live))
            index = int(path.parent.name[1:])
            audit = Audit(sha256="shared-audit", source_chunks=[], source_catalog={})
            live.append(weakref.ref(audit))
            observed.append(index)
            return {"sha256": digest(index), "confirmation_audit": audit}, {}, {}
        def diagnostic(path, manifest, schedule, examples, policy, rid, **kwargs):
            return {"replicate_id": rid, "manifest_sha256": manifest["sha256"], "endpoints": {}}
        with patch("spacing_rerun.prepare.load_prepared", side_effect=loader), \
             patch("spacing_rerun.confirmation_dependency.validate_final_policy", return_value={}), \
             patch("spacing_rerun.confirmation_dependency.verify_preregistered_dependencies"), \
             patch("spacing_rerun.confirmation_dependency.saved_bundle_diagnostics", new=diagnostic):
            result = build_saved_dependency_report(refs, frozen, prereg)
        self.assertEqual(observed, list(range(32)))
        self.assertEqual(len(result["bundles"]), 32)
        self.assertFalse(any(ref() is not None for ref in live))
        with patch("spacing_rerun.prepare.load_prepared", return_value=({"sha256": "wrong", "confirmation_audit": {"sha256": "shared-audit"}}, {}, {})):
            with self.assertRaisesRegex(ValueError, "exact reference"): load_dependency_bundle(refs[0])

    def test_real_format_reader_proves_canonical_roots_and_acquisition_form_means(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, schedule, examples, policy, _, evaluation = reporting_fixture(Path(name))
            prepared = policy["canonical_bindings"][0]
            sections = _saved_evaluation(evaluation, manifest, prepared["binding"], policy["membership"])
            self.assertEqual(len(sections["teaching_qa"]["rows"]), 1)
            self.assertEqual(len(manifest["qa_teaching_pool"]["records"]), 2)
            from spacing_rerun.confirmation_dependency import diagnostic_scores
            qa = sections["acquisition_qa"]
            scores = diagnostic_scores(qa["rows"], qa["mapping"], "exact_match", ["u0", "u1"], ["u0"])
            self.assertEqual(scores["equal_block/all_source"]["group_means"]["e1"]["g0"], .5)
            self.assertEqual(scores["equal_block/all_source"]["event_macro"], .25)
            self.assertIsNone(scores["equal_block/flagged_exclusion"]["event_macro"])
            report = saved_bundle_diagnostics(name, manifest, schedule, examples, policy, "r0")
            self.assertEqual(len(report["endpoints"]), 30)
            self.assertEqual(len(report["unavailable"]), 10)
            incomplete = report["d8_prediction_tables"]["UNI/fraction/0.5/transfer/canonical"]
            self.assertIsNone(incomplete["counts"])
            self.assertEqual(len(incomplete["missing_prediction_row_ids"]), 2)
            table = report["d8_prediction_tables"]["UNI/baseline/transfer/canonical"]
            self.assertEqual(table["counts"]["optical illusion"]["other"], 1)
            self.assertEqual(table["counts"]["an elaborate trick"]["an elaborate trick"], 1)

    def test_executed_clocks_credit_ties_and_block_union_without_double_counting(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, schedule, examples, policy, logs, _ = reporting_fixture(Path(name))
            membership = policy["membership"]
            provenance = policy["canonical_bindings"][0]["training_provenance"]
            rows = executed_presentations(logs, provenance, 2)
            # A single declaration is attributed to two units inside one block.
            rows[0]["source_unit_ids"] = ["u0", "u1"]
            cov = exposure_covariates(rows, membership, len(logs))
            block = membership["group_block"]["g0"]
            unit_count = sum(cov["source_units"][u]["all_phases"]["presentations"] for u in ("u0", "u1"))
            block_count = cov["dependency_blocks"][block]["all_phases"]["presentations"]
            self.assertEqual(unit_count, block_count + 1)
            self.assertEqual(cov["endpoint_clocks"]["input_token_clock"], len(rows) * 5)
            self.assertEqual(cov["source_units"]["u2"]["all_phases"]["presentations"], 0)
            self.assertIsNone(cov["source_units"]["u2"]["all_phases"]["clocks"]["global_step"]["first"])
            self.assertEqual(provenance["records"]["acq/t0/0"]["answer_target_tokens"], 2)
            self.assertEqual(provenance["records"]["gen/g0"]["source_unit_ids"], [])

    def test_corrupt_content_missing_history_and_unbound_roots_fail(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, schedule, examples, policy, logs, evaluation = reporting_fixture(Path(name))
            provenance = policy["canonical_bindings"][0]["training_provenance"]
            for kind in ("content", "global_step", "missing", "order"):
                changed = copy.deepcopy(logs)
                if kind == "content": changed[0]["example_content_sha256"][0] = "altered"
                if kind == "global_step": changed[0]["global_step"] = 2
                if kind == "missing": changed.pop()
                expected = [r["examples"] for r in logs]
                if kind == "order": expected.reverse()
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    executed_presentations(changed, provenance, 2, expected_rows=expected)
            changed = copy.deepcopy(evaluation)
            changed["variants"]["canonical"]["facts"].pop()
            with self.assertRaises(ValueError):
                _saved_evaluation(changed, manifest, policy["canonical_bindings"][0]["binding"], policy["membership"])

    def test_complete_saved_report_keeps_all_cases_and_saved_primary(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, schedule, examples, policy, _, _ = reporting_fixture(Path(name))
            prereg = seal({"n": 1, "E": 1, "primary_delay": 1, "replicate_manifest_sha256": [manifest["sha256"]]})
            completion = seal({"manifest_sha256": manifest["sha256"], "preregistration_sha256": prereg["sha256"],
                               "E": 1, "arms": {arm: {} for arm in ARMS}})
            frozen = seal({"schema": FROZEN_POLICY_SCHEMA, "final_membership_frozen": True, "policy": policy,
                           "review_receipt": {}, "review_evidence": {}, "source_review_evidence": [], "validation": {}})
            manifest["confirmation_audit"].update(source_chunks=[], source_catalog={})
            seal(manifest)
            # The final-policy production validator is tested separately with
            # full all-eight source fixtures. Here only the saved-reader/report
            # adapter is isolated; this synthetic stub cannot freeze anything.
            prepared = policy["canonical_bindings"][0]
            old_sha = prepared["manifest_sha256"]
            prepared["manifest_sha256"] = manifest["sha256"]
            prepared["training_provenance"]["manifest_sha256"] = manifest["sha256"]
            seal(prepared["training_provenance"])
            for file in Path(name).rglob("*.json"):
                value = json.loads(file.read_text())
                if value.get("manifest_sha256") == old_sha:
                    value["manifest_sha256"] = manifest["sha256"]
                    write(file, value)
            prereg["replicate_manifest_sha256"] = [manifest["sha256"]]
            seal(prereg)
            completion.update(manifest_sha256=manifest["sha256"], preregistration_sha256=prereg["sha256"])
            seal(completion)
            seal(frozen)
            completion, primary_binding = retained_reporting_bindings(name, manifest, schedule, prereg)
            inputs = [{"path": name, "manifest": manifest, "schedule": schedule, "examples": examples, "completion": completion,
                       "primary_aggregate_binding": primary_binding}]
            with patch("spacing_rerun.confirmation_dependency.validate_final_policy", return_value={}):
                report = build_saved_dependency_report(inputs, frozen, prereg)
            self.assertEqual(len(report["contrasts"]), 384)
            item = next(r for r in report["contrasts"] if r["endpoint"] == "baseline" and r["role"] == "old" and
                        r["variant"] == "acquisition_qa" and r["metric"] == "exact_match" and r["contrast"] == "H2")
            self.assertEqual(item["primary"]["estimate"], 0)
            self.assertEqual(len(item["secondaries"]), 10)
            self.assertEqual(len(item["influence"]), 9)
            self.assertEqual(item["secondaries"]["equal_block/flagged_exclusion"]["summary"]["n_pair"], 0)
            self.assertFalse(report["primary_analysis_changed"])
            self.assertFalse(report["acquisition_gate_or_E_changed"])
            self.assertTrue(report["valid_poorly_acquired_replicates_retained"])
            self.assertTrue(report["undefined_diagnostics"])
            self.assertEqual(len(report["lane_descriptive_reporting"]["lane_only_complement_contrasts"]), 384)
            self.assertFalse(report["lane_descriptive_reporting"]["V3R2_algorithm_changed"])

    def test_saved_lane_tables_use_all_sections_roles_arms_and_mark_loss_only_missing(self):
        with tempfile.TemporaryDirectory() as name:
            manifest, schedule, examples, policy, _, _ = reporting_fixture(Path(name))
            # Reporting-only synthetic lane. Final evidence validation has its
            # own tests; these rows do not confer real source approval.
            policy["exception_lanes"] = [{"id": "synthetic-D8", "source_unit_ids": ["u0", "u1"],
                "source_group_ids": ["g0", "g1"], "original_labels": {
                    "u0": ["an elaborate trick"], "u1": ["optical illusion"]}}]
            result = saved_bundle_diagnostics(name, manifest, schedule, examples, policy, "r0")
            self.assertEqual(len(result["lane_root_balance"]), 4)
            tables = result["lane_prediction_tables"]["EXP/baseline/transfer/canonical/old"]
            self.assertEqual(tables[0]["target_sides"][0]["counts"]["own_label"], 1)
            self.assertEqual(tables[0]["target_sides"][1]["counts"]["other"], 1)
            missing = result["lane_prediction_tables"]["EXP/fraction/0.5/transfer/canonical/old"]
            self.assertEqual(missing[0]["target_sides"][0]["status"], "undefined_missing_saved_predictions")
            self.assertTrue(any(r["section"] == "acquisition_qa" and r["arm"] == "EXP" for r in result["unavailable"]))


if __name__ == "__main__":
    unittest.main()
