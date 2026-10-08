"""Synthetic full-file fixtures for confirmation completion and device guards."""
import collections
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from spacing_rerun import SCHEMA
from spacing_rerun.acquisition_grid import checkpoint_digest
from spacing_rerun.analysis import validate_confirmation_completion
from spacing_rerun.common import digest, read_json, write_json
from spacing_rerun.evaluation import aggregate
from spacing_rerun.schedule import ARMS, stage1_epoch
from spacing_rerun.training import verify_confirmation_hardware, run_acquisition
from spacing_rerun.units import schedule_records, GROUNDED_METRICS
from test_confirmation_runtime import prepared_confirmation_fixture


def complete_fixture(root):
    _, manifest, schedule, _ = prepared_confirmation_fixture(root)
    examples = read_json(root / "prepared/examples.json")
    prereg = {"sha256": "synthetic-completion-preregistration", "E": 4, "maximum_attempts": 3,
              "test_fixture": True}
    records = schedule_records(manifest["split"]["facts"], manifest["unit_registry"], manifest["qa_teaching_pool"]["records"])
    targets = manifest["acquisition_qa_pool"]["records"]
    stage1_rows = [row for epoch in range(4) for row in stage1_epoch(records, epoch, manifest["config"], targets)]
    progress = {"global_step": 0, "cursor": 0, "epoch": 4, "epoch_cursor": 0, "continuation_cursor": 0,
        "status": "stage1_complete", "phase": "stage1", "confirmation_fixed_dose_complete": True,
        "confirmation_preregistration_sha256": prereg["sha256"], "acquisition_usable": False,
        "observed_acquisition_old_exact_match_event_macro": 0., "new_qa_tokens_total": 0,
        "acquisition_qa_tokens_total": 0, "last_old_update": {}, "last_old_newqa_clock": {},
        "acquisition_qa_target_doses": {r["id"]: 0 for r in targets},
        "acquisition_qa_variant_doses": {r["id"]: [0, 0] for r in targets}}
    def advance(rows, current, stage2=None):
        logs, snapshots = [], {0: copy.deepcopy(current)}
        for index, row in enumerate(rows):
            current["global_step"] += 1
            current["cursor"] += 1
            current["new_qa_tokens_total"] += sum(examples[key]["loss_tokens"] for key in row if key.startswith(("new/", "qa/")))
            acquisition_keys = [key for key in row if key.startswith("acq/")]
            current["acquisition_qa_tokens_total"] += sum(examples[key]["loss_tokens"] for key in acquisition_keys)
            for key in acquisition_keys:
                target, variant = key.split("/")[1:]
                current["acquisition_qa_target_doses"][target] += 1
                current["acquisition_qa_variant_doses"][target][int(variant)] += 1
            for key in row:
                if key.startswith("old/"):
                    unit = key.split("/", 1)[1]
                    current["last_old_update"][unit] = current["global_step"]
                    current["last_old_newqa_clock"][unit] = current["new_qa_tokens_total"]
            logs.append({"phase": "stage1" if stage2 is None else "stage2" if index < stage2 else "buffer",
                "phase_step": index, "cursor": current["cursor"], "global_step": current["global_step"],
                "examples": row, "row_sha256": digest(row), "example_content_sha256": [examples[k]["content_sha256"] for k in row],
                "loss_tokens": sum(examples[k]["loss_tokens"] for k in row), "new_qa_tokens_total": current["new_qa_tokens_total"],
                "acquisition_qa_examples": acquisition_keys, "acquisition_qa_tokens_total": current["acquisition_qa_tokens_total"]})
            snapshots[index + 1] = copy.deepcopy(current)
        return logs, snapshots
    def state(current):
        return {"schema": SCHEMA, "manifest_sha256": manifest["sha256"], "progress": current,
            "model": {"synthetic_weight": torch.tensor([1.])},
            "optimizer": {"state": {0: {"step": torch.tensor(current["global_step"])}}, "param_groups": [{"lr": .0001}]},
            "rng": {"python": (1,), "numpy": (1,), "torch": torch.tensor([1]), "cuda": None}, "test_fixture": True}
    def write_logs(path, logs):
        path.write_text("\n".join(json.dumps(row) for row in logs) + "\n")
    selected = [f for f in manifest["split"]["facts"] if f["role"] in ("old", "new", "control")]
    baseline_rows = [{"id": f["id"], "event": f["event"], "unit_id": f["unit_id"], "role": f["role"],
                      "loss": 1., "exact_match": 0} for f in selected]
    baseline = {"manifest_sha256": manifest["sha256"], "epoch": 4, "metric_schema": GROUNDED_METRICS,
                "variants": {"canonical": {"facts": baseline_rows, "aggregate": aggregate(baseline_rows)}}, "test_fixture": True}
    stage1 = root / "stage1"
    (stage1 / "evaluations").mkdir(parents=True)
    logs, _ = advance(stage1_rows, progress)
    torch.save(state(progress), stage1 / "stage1.pt")
    write_logs(stage1 / "exposures.jsonl", logs)
    baseline["global_step"] = progress["global_step"]
    write_json(stage1 / "evaluations/acquisition-E004.json", baseline)
    write_json(stage1 / "acquisition_decision.json", {"manifest_sha256": manifest["sha256"], "selected_E": 4,
        "fixed_dose_continuation_authorized": True, "outcome_filtering_permitted": False,
        "confirmation_preregistration_sha256": prereg["sha256"], "acquisition_usable": False,
        "observed_acquisition_old_exact_match_event_macro": 0., "test_fixture": True})
    write_json(stage1 / "attempt-01.json", {"manifest_sha256": manifest["sha256"], "status": "complete", "progress": progress})
    for arm in ARMS:
        directory = root / arm
        (directory / "evaluations").mkdir(parents=True)
        current = copy.deepcopy(progress)
        current.update(cursor=0, arm=arm, confirmation_shared_checkpoint=str((stage1 / "stage1.pt").resolve()),
                       confirmation_shared_checkpoint_sha256=checkpoint_digest(stage1 / "stage1.pt"))
        logs, snapshots = advance(schedule["arms"][arm], current, schedule["stage2_steps"])
        current.update(status="complete", continuation_cursor=len(logs))
        torch.save(state(current), directory / "latest.pt")
        write_logs(directory / "exposures.jsonl", logs)
        write_json(directory / "attempt-01.json", {"manifest_sha256": manifest["sha256"], "status": "complete", "progress": current})
        required = {0, schedule["stage2_steps"], len(logs), *(schedule["stage2_steps"] + d for d in manifest["config"]["eval_delays"]),
                    *(round(schedule["stage2_steps"] * f) for f in manifest["config"]["stage2_eval_fractions"])}
        for step in required:
            evaluation = copy.deepcopy(baseline)
            moment = snapshots[step]
            evaluation.update(global_step=progress["global_step"] + step, stage2_step=step,
                global_buffer_delay=step - schedule["stage2_steps"], acquisition_qa_tokens_total=progress["acquisition_qa_tokens_total"],
                updates_since_last_old_exposure={key: moment["global_step"] - value for key, value in moment["last_old_update"].items()},
                new_qa_tokens_since_last_old_exposure={key: moment["new_qa_tokens_total"] - value for key, value in moment["last_old_newqa_clock"].items()})
            write_json(directory / "evaluations" / f"stage2-{step:06}.json", evaluation)
    return manifest, schedule, examples, prereg


class ConfirmationCompletionTests(unittest.TestCase):
    def test_complete_actual_files_retain_poor_acquisition_and_bind_all_forks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = complete_fixture(root)
            receipt = validate_confirmation_completion(root, *inputs)
            self.assertTrue(receipt["poor_fixed_dose_acquisition_retained"])
            self.assertEqual(set(receipt["arms"]), set(ARMS))
            self.assertTrue(any(row["path"].endswith("NONE/latest.pt") for row in receipt["files"]))

    def test_missing_final_buffer_changed_stream_fork_clock_or_aggregate_blocks(self):
        for mutation in ("partial", "dose", "fork", "qa", "clock", "attempt", "missing_eval", "eval_clock", "aggregate", "stage1_stream"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, schedule, examples, prereg = complete_fixture(root)
                if mutation in ("partial", "dose", "fork", "qa", "clock"):
                    path = root / "NONE/latest.pt"
                    state = torch.load(path, weights_only=False)
                    key, value = {"partial": ("continuation_cursor", 0), "dose": ("epoch", 6),
                                  "fork": ("confirmation_shared_checkpoint_sha256", "different"),
                                  "qa": ("acquisition_qa_tokens_total", -1), "clock": ("global_step", 1)}[mutation]
                    state["progress"][key] = value
                    torch.save(state, path)
                elif mutation == "attempt":
                    path = root / "NONE/attempt-01.json"
                    record = read_json(path); record["status"] = "running"; write_json(path, record)
                elif mutation in ("missing_eval", "eval_clock", "aggregate"):
                    path = root / "NONE/evaluations" / f"stage2-{schedule['stage2_steps'] + 168:06}.json"
                    if mutation == "missing_eval":
                        path.unlink()
                    else:
                        record = read_json(path)
                        if mutation == "eval_clock": record["global_step"] += 1
                        else: record["variants"]["canonical"]["aggregate"]["old"]["loss_event_macro"] = 3.
                        write_json(path, record)
                else:
                    path = root / "stage1/exposures.jsonl"
                    rows = path.read_text().splitlines(); record = json.loads(rows[0]); record["loss_tokens"] += 1
                    rows[0] = json.dumps(record); path.write_text("\n".join(rows))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_confirmation_completion(root, manifest, schedule, examples, prereg)


class HardwareAndResumeTests(unittest.TestCase):
    def test_actual_device_name_and_bf16_are_required(self):
        prereg = {"hardware": {"gpu": "Frozen GPU"}}
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.current_device", return_value=0), \
             patch("torch.cuda.get_device_name", return_value="Frozen GPU"), patch("torch.cuda.is_bf16_supported", return_value=True):
            verify_confirmation_hardware(prereg, "cuda")
        for name, bf16 in (("Other GPU", True), ("Frozen GPU", False)):
            with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.current_device", return_value=0), \
                 patch("torch.cuda.get_device_name", return_value=name), patch("torch.cuda.is_bf16_supported", return_value=bf16), \
                 self.assertRaises(ValueError):
                verify_confirmation_hardware(prereg, "cuda")
        with self.assertRaises(ValueError):
            verify_confirmation_hardware(prereg, "cpu")
        with self.assertRaises(ValueError):
            verify_confirmation_hardware(prereg, "cpu", synthetic_cpu_fixture=True)
        verify_confirmation_hardware(dict(prereg, test_fixture=True), "cpu", synthetic_cpu_fixture=True)

    def test_partial_confirmation_resume_cannot_change_preregistration(self):
        manifest = {"mode": "confirmation", "confirmation_protocol_schema": "frozen-test-protocol", "config": {}, "test_fixture": True}
        progress = {"global_step": 1, "epoch": 0, "status": "running", "confirmation_preregistration_sha256": "old-frozen-reg"}
        with patch("spacing_rerun.training.load_prepared", return_value=(manifest, {}, {})), \
             patch("spacing_rerun.training.verify_preregistration", return_value={"E": 4, "sha256": "new-reg"}), \
             patch("spacing_rerun.training.verify_confirmation_hardware"), patch("spacing_rerun.training.Session") as session:
            session.return_value.progress = progress
            with self.assertRaisesRegex(ValueError, "different frozen preregistration"):
                run_acquisition("synthetic-prepared", "synthetic-output", device="cuda", fixed_exposures=4)
            session.return_value.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
