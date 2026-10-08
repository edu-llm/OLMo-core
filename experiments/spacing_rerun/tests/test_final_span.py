import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spacing_rerun import SCHEMA
from spacing_rerun.common import digest, write_json
from spacing_rerun.final_span import (FROZEN_CORE_SHA256, PLANNING_POLICY, assay_sensitivity,
                                     cost_report, evaluate_final_span, plan_precision,
                                     validate_bundle, verify_final_span_report)
from spacing_rerun.units import GROUNDED_METRICS


def summary_bundle(index=0):
    return {"acquisition": {"loss": 1., "exact_match": .55}, "arms": {
        "NONE": {"loss": 1.15 + .01 * index, "exact_match": .45},
        "UNI": {"loss": 1.05, "exact_match": .50},
        "GEN": {"loss": 1.10 + .001 * index, "exact_match": .47},
        "EXP": {"loss": 1.05 + .001 * (index - 2.5), "exact_match": .50},
        "MASS": {"loss": 1.04, "exact_match": .50}}}


def write_bundle(root):
    """Synthetic full-state fixture, never presented as experimental evidence."""
    root.mkdir()
    manifest = {"sha256": "synthetic-manifest", "config": {"max_attempts": 3,
        "split_seed": 999, "order_seed": 2701, "train_seed": 2801, "eval_seed": 2901},
        "acquisition_qa_pool": {"records": [{"id": "source-target"}]}}
    # Prepared validation is isolated here; its real integrity guards have their
    # own tests. This fixture exercises actual checkpoint, stream and eval files.
    schedule = {"stage2_steps": 3, "stagger_width": 36, "stagger_span_ratio": 36 / 252,
                "arms": {arm: [["new/test"] for _ in range(171)] for arm in ("NONE", "UNI", "EXP", "MASS", "GEN")}}
    examples = {"new/test": {"content_sha256": "source-example", "loss_tokens": 36}}
    progress = {"status": "stage1_complete", "epoch": 4, "epoch_cursor": 0,
                "global_step": 52, "acquisition_usable": True,
                "acquisition_qa_target_doses": {"source-target": 4},
                "acquisition_qa_variant_doses": {"source-target": [2, 2]}, "acquisition_qa_tokens_total": 144,
                "new_qa_tokens_total": 0, "last_old_update": {"old-unit": 52}, "last_old_newqa_clock": {"old-unit": 0}}
    state = {"schema": SCHEMA, "manifest_sha256": manifest["sha256"], "progress": progress,
             "model": {"weight": torch.tensor([1.])},
             "optimizer": {"state": {0: {"step": torch.tensor(52), "exp_avg": torch.tensor([.1])}}, "param_groups": [{"lr": .0001}]},
             "rng": {"python": (1,), "numpy": (1,), "torch": torch.tensor([1]), "cuda": None}}
    stage1 = root / "stage1"
    (stage1 / "evaluations").mkdir(parents=True)
    torch.save(state, stage1 / "stage1-E004.pt")
    from spacing_rerun.acquisition_grid import checkpoint_digest
    binding = {"schema": "spacing-acquisition-grid-manifest-binding-v1", "sha256": "synthetic-binding",
               "policy_core": {"sha256": FROZEN_CORE_SHA256}}
    selection = {"selected_checkpoint": str((stage1 / "stage1-E004.pt").resolve()),
                 "selected_checkpoint_sha256": checkpoint_digest(stage1 / "stage1-E004.pt"),
                 "selected_E": 4, "selection_rule": binding, "sha256": "synthetic-selection",
                 "grid_inputs": [{"decision_path": str(stage1 / "acquisition_decision.json")}]}
    write_json(stage1 / "acquisition_decision.json", {"manifest_sha256": manifest["sha256"],
        "grid_results": {"4": {"old_exact_match_event_macro": .55}}})
    write_json(root / "acquisition-selection.json", selection)
    canonical = {"aggregate": {"old": {"loss_event_macro": 1., "exact_match_event_macro": .55}}}
    baseline = {"schema": SCHEMA, "manifest_sha256": manifest["sha256"], "epoch": 4,
                "global_step": 52, "metric_schema": GROUNDED_METRICS,
                "acquisition_qa_tokens_total": 144, "variants": {"canonical": canonical}}
    write_json(stage1 / "evaluations/acquisition-E004.json", baseline)
    for arm in schedule["arms"]:
        directory = root / arm
        (directory / "evaluations").mkdir(parents=True)
        final = copy.deepcopy(state)
        final["progress"].update(status="complete", arm=arm, continuation_cursor=171, cursor=171, global_step=223,
                                 new_qa_tokens_total=171 * 36)
        torch.save(final, directory / "latest.pt")
        write_json(directory / "attempt-01.json", {"manifest_sha256": manifest["sha256"], "status": "complete",
                   "started_utc": "2026-10-08T08:00:00+00:00", "progress": final["progress"],
                   "wall_seconds": 100., "gpu_count": 1, "slurm_job_id": "job-" + arm})
        logs = [{"cursor": i + 1, "phase_step": i, "global_step": 53 + i,
                 "phase": "stage2" if i < 3 else "buffer", "examples": ["new/test"],
                 "row_sha256": digest(["new/test"]), "example_content_sha256": ["source-example"],
                 "loss_tokens": 36, "acquisition_qa_examples": [], "acquisition_qa_tokens_total": 144,
                 "new_qa_tokens_total": (i + 1) * 36}
                for i in range(171)]
        (directory / "exposures.jsonl").write_text("\n".join(json.dumps(log) for log in logs) + "\n")
        for step in (0, 3, 24, 87, 171):
            evaluation = copy.deepcopy(baseline)
            evaluation.update(global_step=52 + step, stage2_step=step, global_buffer_delay=step - 3)
            evaluation.update(updates_since_last_old_exposure={"old-unit": step},
                              new_qa_tokens_since_last_old_exposure={"old-unit": step * 36})
            if step:
                old = evaluation["variants"]["canonical"]["aggregate"]["old"]
                old.update(loss_event_macro={"NONE": 1.2, "UNI": 1.05, "GEN": 1.1, "EXP": 1.051, "MASS": 1.04}[arm],
                           exact_match_event_macro=.45)
            write_json(directory / "evaluations" / f"stage2-{step:06}.json", evaluation)
    return manifest, schedule, examples, selection


class SensitivityTests(unittest.TestCase):
    def test_six_pairs_pass_and_exp_cannot_change_eligibility(self):
        bundles = [summary_bundle(i) for i in range(6)]
        actual = assay_sensitivity(bundles)
        self.assertTrue(actual["assay_usable"])
        for bundle in bundles:
            bundle["arms"]["EXP"]["loss"] = 900.
        self.assertEqual(actual, assay_sensitivity(bundles))

    def test_mean_threshold_and_positive_lower_bound_both_required(self):
        for kind in ("mean", "lower", "em", "gen"):
            bundles = [summary_bundle(i) for i in range(6)]
            if kind == "mean":
                for bundle in bundles:
                    bundle["arms"]["NONE"]["loss"] = 1.09
            elif kind == "lower":
                for i, bundle in enumerate(bundles):
                    bundle["arms"]["NONE"]["loss"] = 2. if i == 0 else .99
            elif kind == "em":
                for bundle in bundles:
                    bundle["arms"]["NONE"]["exact_match"] = .54
            else:
                for bundle in bundles:
                    bundle["arms"]["GEN"]["loss"] = 1.06
            with self.subTest(kind=kind):
                self.assertFalse(assay_sensitivity(bundles)["assay_usable"])
        with self.assertRaisesRegex(ValueError, "six complete"):
            assay_sensitivity(bundles[:-1])

    def test_nonfinite_loss_is_rejected(self):
        bundles = [summary_bundle(i) for i in range(6)]
        bundles[0]["arms"]["UNI"]["loss"] = float("nan")
        with self.assertRaises(ValueError):
            assay_sensitivity(bundles)


class PrecisionTests(unittest.TestCase):
    def test_conservative_sd_all_five_scenarios_and_honest_estimation_fallback(self):
        from scipy.stats import chi2
        with patch.dict(PLANNING_POLICY, simulation_draws=4000):
            small = plan_precision([-.003, -.002, -.001, .001, .002, .003])
            large = plan_precision([-.15, -.10, -.05, .05, .10, .15])
        self.assertTrue(small["power_target_met"])
        self.assertLessEqual(small["chosen_n"], 32)
        self.assertEqual(len(small["chosen_primary_scenarios"]), 5)
        self.assertTrue(all(row["one_sided_95pct_lower"] >= .8 for row in small["chosen_primary_scenarios"]))
        self.assertAlmostEqual(small["one_sided_80pct_upper_sd"], small["observed_sd"] * math.sqrt(5 / chi2.ppf(.2, 5)))
        self.assertGreater(small["one_sided_80pct_upper_sd"], small["observed_sd"])
        self.assertEqual({row["sd_scenario"] for row in small["scenarios"]},
                         {"observed", "one_sided_80pct_upper", "1.5_times_upper"})
        self.assertEqual(large["claim"], "estimation")
        self.assertEqual(large["chosen_n"], 32)
        self.assertFalse(large["power_target_met"])
        self.assertTrue(any(row["one_sided_95pct_lower"] < .8 for row in large["chosen_primary_scenarios"]))
        self.assertGreater(large["expected_95ci_halfwidth"], .02)
        self.assertFalse(large["normality_established"])

    def test_simulation_reproducible_and_zero_variance_does_not_fake_precision(self):
        with patch.dict(PLANNING_POLICY, simulation_draws=1000):
            self.assertEqual(plan_precision([-.003, -.002, -.001, .001, .002, .003]),
                             plan_precision([-.003, -.002, -.001, .001, .002, .003]))
        with self.assertRaisesRegex(ValueError, "Zero observed SD"):
            plan_precision([0.] * 6)


class BundleGuardTests(unittest.TestCase):
    def run_fixture(self, root, prepared, selection):
        with patch("spacing_rerun.final_span.load_prepared", return_value=prepared), \
             patch("spacing_rerun.final_span.verify_acquisition_grid_selection", return_value=selection), \
             patch("spacing_rerun.final_span.manifest_policy_trial", return_value={"trial_id": "normal-01", "development_scale_probe": False}):
            return validate_bundle(root)

    def test_actual_full_checkpoint_stream_and_evaluation_files_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "normal-01"
            *prepared, selection = write_bundle(root)
            result = self.run_fixture(root, tuple(prepared), selection)
            self.assertEqual(result["selected_E"], 4)
            self.assertEqual(set(result["arms"]), {"NONE", "UNI", "EXP", "MASS", "GEN"})
            self.assertEqual(result["raw_primary_endpoints"]["NONE"]["stage2_step"], 87)
            self.assertTrue(any(p["path"].endswith("NONE/latest.pt") for p in result["provenance"]))
            path = root / "NONE/evaluations/stage2-000087.json"
            changed = json.loads(path.read_text())
            changed["variants"]["canonical"]["aggregate"]["old"]["loss_event_macro"] = 8.
            write_json(path, changed)
            updated = self.run_fixture(root, tuple(prepared), selection)
            self.assertNotEqual(result["provenance"], updated["provenance"])
            self.assertEqual(updated["arms"]["NONE"]["loss"], 8.)

    def test_incomplete_arm_changed_dose_missing_endpoint_or_source_qa_after_fork_rejected(self):
        for mutation in ("checkpoint", "dose", "endpoint", "baseline", "sourceqa", "stream", "attempt", "core", "score", "late_policy"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "normal-01"
                *prepared, selection = write_bundle(root)
                if mutation in ("checkpoint", "dose", "sourceqa"):
                    path = root / "NONE/latest.pt"
                    state = torch.load(path, weights_only=False)
                    if mutation == "checkpoint":
                        state["progress"]["status"] = "running"
                    elif mutation == "dose":
                        state["progress"]["epoch"] = 6
                    else:
                        state["progress"]["acquisition_qa_tokens_total"] += 36
                    torch.save(state, path)
                elif mutation == "endpoint":
                    (root / "NONE/evaluations/stage2-000087.json").unlink()
                elif mutation == "baseline":
                    path = root / "NONE/evaluations/stage2-000000.json"
                    evaluation = json.loads(path.read_text())
                    evaluation["variants"]["canonical"]["aggregate"]["old"]["loss_event_macro"] = 2.
                    write_json(path, evaluation)
                elif mutation == "stream":
                    path = root / "NONE/exposures.jsonl"
                    logs = [json.loads(line) for line in path.read_text().splitlines()]
                    logs[0]["loss_tokens"] += 1
                    path.write_text("\n".join(json.dumps(log) for log in logs))
                elif mutation in ("attempt", "late_policy"):
                    path = root / "NONE/attempt-01.json"
                    attempt = json.loads(path.read_text())
                    if mutation == "attempt":
                        attempt["status"] = "running"
                    else:
                        attempt["started_utc"] = "2026-10-08T06:00:00+00:00"
                    write_json(path, attempt)
                elif mutation == "core":
                    selection["selection_rule"]["policy_core"]["sha256"] = "different"
                else:
                    path = root / "stage1/acquisition_decision.json"
                    decision = json.loads(path.read_text())
                    decision["grid_results"]["4"]["old_exact_match_event_macro"] = .6
                    write_json(path, decision)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.run_fixture(root, tuple(prepared), selection)

    def test_report_verifier_recomputes_raw_evidence_after_packet_rehash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundles = []
            for i in range(6):
                bundle = summary_bundle(i)
                bundle.update(trial_id=f"normal-{i + 1:02}", manifest_sha256=f"manifest-{i}", selected_E=4,
                              manifest_binding_sha256="shared-binding", bundle_path=str(root / str(i)),
                              selection_path=str(root / str(i) / "selection.json"), selection_sha256=f"selection-{i}",
                              seeds={}, raw_acquisition={}, raw_primary_endpoints={}, provenance=[], attempts=[])
                bundles.append(bundle)
            precision = {"chosen_n": 32, "claim": "estimation"}
            output = root / "report.json"
            with patch("spacing_rerun.final_span.validate_bundle", side_effect=copy.deepcopy(bundles)), \
                 patch("spacing_rerun.final_span.plan_precision", return_value=precision):
                report = evaluate_final_span([root / str(i) for i in range(6)], output)
            self.assertFalse(report["confirmation_ready"])
            with patch("spacing_rerun.final_span.validate_bundle", side_effect=copy.deepcopy(bundles)), \
                 patch("spacing_rerun.final_span.plan_precision", return_value=precision):
                self.assertEqual(verify_final_span_report(output), report)
            report["checks"]["none_minus_uni_loss"]["mean"] = 5.
            report["sha256"] = digest({k: v for k, v in report.items() if k != "sha256"})
            write_json(output, report)
            with patch("spacing_rerun.final_span.validate_bundle", side_effect=copy.deepcopy(bundles)), \
                 patch("spacing_rerun.final_span.plan_precision", return_value=precision), \
                 self.assertRaisesRegex(ValueError, "paired endpoints"):
                verify_final_span_report(output)
            with self.assertRaisesRegex(ValueError, "never overwrite"):
                evaluate_final_span([], output)


class CostTests(unittest.TestCase):
    def test_absent_accounting_is_explicit_and_sacct_requires_all_observed_jobs(self):
        bundles = [{"attempts": [{"wall_seconds": 100., "gpu_count": 1, "slurm_job_id": "1"}]}]
        self.assertFalse(cost_report(bundles)["accounting_complete"])
        self.assertIsNone(cost_report(bundles)["allocation_gpu_hours"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sacct.json"
            accounting = {"schema": "spacing-slurm-allocation-accounting-v1", "allocations": [{
                "job_id": "1", "elapsed_seconds": 120, "allocated_gpu_count": 1,
                "source": "sacct", "state": "COMPLETED"}]}
            write_json(path, accounting)
            cost = cost_report(bundles, path, 32)
            self.assertAlmostEqual(cost["allocation_gpu_hours"], 120 / 3600)
            self.assertFalse(cost["confirmation_cost_projection"]["full_experiment_budget_ready"])
            accounting["allocations"] = []
            write_json(path, accounting)
            with self.assertRaisesRegex(ValueError, "omits"):
                cost_report(bundles, path)


if __name__ == "__main__":
    unittest.main()
