"""Check exact checkpoint replay on the live model/backend without changing a run.

Compare two uninterrupted UNI updates with checkpoint restoration between them.
This is a numerical reproducibility check, not a retention experiment.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .common import append_json, digest, require, write_json
from .evaluation import conditional_scores, generated_answers
from .prepare import load_prepared
from .training import (capture_rng, load_checkpoint, load_model, make_optimizer,
                       restore_rng, save_checkpoint, seed_all, train_update)


def file_identity(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"Bound artifact changed while hashing: {path}")
    return {"sha256": sha256, "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def prepared_identity(path):
    path = Path(path)
    return {str(p.relative_to(path)): file_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}


def compare_tree(expected, actual, path="root"):
    """Compare complete nested tensor/optimizer/RNG state, recording names only."""
    import numpy as np
    import torch
    mismatches, tensors = [], 0

    def visit(left, right, name):
        nonlocal tensors
        if torch.is_tensor(left):
            tensors += 1
            if not (torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and
                    torch.equal(left.detach().cpu(), right.detach().cpu())):
                mismatches.append(name)
        elif isinstance(left, np.ndarray):
            if not (isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)):
                mismatches.append(name)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                mismatches.append(name + ".keys")
                return
            for key in left:
                visit(left[key], right[key], f"{name}.{key}")
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                mismatches.append(name + ".length_or_type")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{name}[{index}]")
        elif type(left) is not type(right) or left != right:
            mismatches.append(name)

    visit(expected, actual, path)
    return {"bitwise_equal": not mismatches, "tensors_compared": tensors,
            "mismatches": mismatches[:20], "mismatch_count": len(mismatches)}


def run_aa(prepared, stage1_checkpoint, output, *, device="cuda", model_factory=None):
    import torch
    started = time.monotonic()
    prepared, source, output = [Path(p).resolve() for p in (prepared, stage1_checkpoint, output)]
    require(not output.exists(), "A/A output already exists; never overwrite")
    require(not output.is_relative_to(prepared), "A/A output must be outside bound preparation")
    require(source.is_file(), "Missing shared Stage-1 checkpoint")
    require(device in ("cuda", "cpu"), "A/A device must be cuda or cpu")
    require(device == "cpu" or torch.cuda.is_available(), "CUDA unavailable; do not run real-model A/A on a login CPU")
    require(device == "cpu" or torch.cuda.is_bf16_supported(), "CUDA bf16 unsupported")
    manifest, schedule, examples = load_prepared(prepared)
    require(len(schedule["arms"]["UNI"]) >= 2, "A/A requires two complete UNI updates")
    rows = schedule["arms"]["UNI"][:2]
    require(not any(k.startswith("acq/") for row in rows for k in row), "Source acquisition QA entered UNI schedule")
    commit = os.environ.get("SPACING_CODE_COMMIT")
    if not commit:
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    require(device == "cpu" or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)),
            "GPU A/A requires a full code commit")
    output.mkdir(parents=True)
    software = {}
    for name in ("torch", "transformers", "ai2-olmo", "numpy"):
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = {"schema": "spacing-aa-replay-v1", "type": "two_update_midpoint_checkpoint_replay", "status": "running",
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "manifest_sha256": manifest["sha256"],
              "code_commit": commit, "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "software": software,
              "source_checkpoint": str(source), "prepared": str(prepared), "rows": rows,
              "interpretation": "Numerical replay check only; does not establish acquisition or assay sensitivity."}
    timings = {}
    try:
        tick = time.monotonic()
        report["source_checkpoint_identity"] = file_identity(source)
        report["bound_prepared_artifacts"] = prepared_identity(prepared)
        timings["initial_hash_seconds"] = time.monotonic() - tick
        seed_all(manifest["config"]["train_seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else contextlib.nullcontext
        tick = time.monotonic()
        factory = model_factory or load_model
        model, tokenizer = factory(manifest, prepared, device)
        optimizer = make_optimizer(model, manifest["config"])
        progress = load_checkpoint(source, model, optimizer, manifest["sha256"], device)
        require(progress["status"] == "stage1_complete" and progress["epoch_cursor"] == 0,
                "A/A requires a complete shared Stage-1 checkpoint, not a live writer")
        report["source_epoch"] = progress["epoch"]
        report["source_global_step"] = progress["global_step"]
        report["source_acquisition_usable"] = progress.get("acquisition_usable")
        timings["initial_load_seconds"] = time.monotonic() - tick
        facts = sorted((f for f in manifest["split"]["facts"] if f["role"] == "old"), key=lambda f: f["id"])[:8]
        require(facts, "A/A requires old evaluation probes")
        probes = [f["probe"] for f in facts]
        report["probe_ids"] = [f["id"] for f in facts]

        def fork_progress(source_progress):
            state = copy.deepcopy(source_progress)
            state.update(status="aa_running", phase="aa_check", arm="UNI", cursor=0, continuation_cursor=0,
                         source_checkpoint_sha256=report["source_checkpoint_identity"]["sha256"])
            state.setdefault("new_qa_tokens_total", 0)
            state.setdefault("last_old_update", {})
            state.setdefault("last_old_newqa_clock", {})
            return state

        def apply_row(state, index, log):
            row = rows[index]
            metrics = train_update(model, optimizer, row, examples, manifest["config"],
                                   state["global_step"], device, autocast)
            state["global_step"] += 1
            state["cursor"] += 1
            state["continuation_cursor"] = index + 1
            state["new_qa_tokens_total"] += sum(examples[k]["loss_tokens"] for k in row if k.startswith(("new/", "qa/")))
            for key in row:
                if key.startswith("old/"):
                    unit = key.split("/", 1)[1]
                    state["last_old_update"][unit] = state["global_step"]
                    state["last_old_newqa_clock"][unit] = state["new_qa_tokens_total"]
            append_json(log, {"cursor": state["cursor"], "global_step": state["global_step"],
                              "phase": "aa_check", "phase_step": index, "examples": row, "row_sha256": digest(row),
                              "example_content_sha256": [examples[k]["content_sha256"] for k in row],
                              "new_qa_tokens_total": state["new_qa_tokens_total"], **metrics})
            return metrics

        def observe():
            rng, was_training = capture_rng(), model.training
            model.eval()
            scores = conditional_scores(model, probes, tokenizer.eos_token_id, device,
                                        manifest["config"]["eval_batch_size"], autocast)
            answers = generated_answers(model, tokenizer, probes, device,
                                        manifest["config"]["generation_batch_size"],
                                        manifest["config"]["max_answer_tokens"], autocast)
            model.train(was_training)
            restore_rng(rng)
            return {"conditional_scores": scores, "greedy_answers": answers}

        tick = time.monotonic()
        uninterrupted_log = output / "uninterrupted-exposures.jsonl"
        restored_log = output / "restored-exposures.jsonl"
        expected_progress = fork_progress(progress)
        expected_metrics = [apply_row(expected_progress, index, uninterrupted_log) for index in range(2)]
        expected_progress["status"] = "aa_complete"
        expected_observations = observe()
        reference = output / "uninterrupted.pt"
        save_checkpoint(reference, model, optimizer, expected_progress, manifest["sha256"])
        timings["uninterrupted_train_observe_save_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        restored_progress = load_checkpoint(source, model, optimizer, manifest["sha256"], device)
        require(compare_tree(progress, restored_progress, "source_progress")["bitwise_equal"], "Restored source progress differs")
        actual_progress = fork_progress(restored_progress)
        actual_metrics = [apply_row(actual_progress, 0, restored_log)]
        midpoint = output / "midpoint.pt"
        save_checkpoint(midpoint, model, optimizer, actual_progress, manifest["sha256"])
        expected_midpoint_progress = copy.deepcopy(actual_progress)
        # Demonstrate restoration of model values, Adam moments and all RNGs.
        with torch.no_grad():
            next(model.parameters()).view(-1)[0].add_(1.0)
            first_state = next(iter(optimizer.state.values()))
            first_state["exp_avg"].view(-1)[0].add_(1.0)
        import numpy as np
        random.random()
        np.random.random()
        torch.rand(1)
        if device == "cuda":
            torch.rand(1, device=device)
        actual_progress["global_step"] += 1000
        actual_progress["new_qa_tokens_total"] += 1000
        actual_progress = load_checkpoint(midpoint, model, optimizer, manifest["sha256"], device)
        require(compare_tree(expected_midpoint_progress, actual_progress, "midpoint_progress")["bitwise_equal"],
                "Midpoint progress/cursor/clocks did not restore")
        require(actual_progress["continuation_cursor"] == 1 and actual_progress["cursor"] == 1,
                "Midpoint must restore exactly the first realized UNI update")
        actual_metrics.append(apply_row(actual_progress, actual_progress["continuation_cursor"], restored_log))
        actual_progress["status"] = "aa_complete"
        actual_observations = observe()
        actual_rng = capture_rng()
        timings["restore_train_observe_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        expected = torch.load(reference, map_location="cpu", weights_only=False)
        comparisons = {
            "model": compare_tree(expected["model"], model.state_dict(), "model"),
            "optimizer": compare_tree(expected["optimizer"], optimizer.state_dict(), "optimizer"),
            "rng": compare_tree(expected["rng"], actual_rng, "rng"),
            "progress_and_clocks": compare_tree(expected["progress"], actual_progress, "progress"),
            "training_metrics": compare_tree(expected_metrics, actual_metrics, "training_metrics"),
            "probe_observations": compare_tree(expected_observations, actual_observations, "probe_observations")}
        del expected
        expected_log = [json.loads(line) for line in uninterrupted_log.read_text().splitlines()]
        actual_log = [json.loads(line) for line in restored_log.read_text().splitlines()]
        require([entry["examples"] for entry in expected_log] == rows and len(actual_log) == 2,
                "A/A realized update count/rows differ from the frozen UNI prefix")
        comparisons["realized_exposures"] = compare_tree(expected_log, actual_log, "exposures")
        report["comparisons"] = comparisons
        report["training_metrics"] = {"uninterrupted": expected_metrics, "restored": actual_metrics}
        report["probe_observations"] = {"uninterrupted": expected_observations, "restored": actual_observations}
        report["reference_checkpoint"] = str(reference)
        report["midpoint_checkpoint"] = str(midpoint)
        report["midpoint_progress"] = expected_midpoint_progress
        report["restore_validation"] = "Midpoint restored after deliberately perturbing model, Adam moments, RNGs and progress clocks."
        report["realized_doses"] = dict(collections.Counter(k for entry in actual_log for k in entry["examples"]))
        report["final_progress"] = actual_progress
        timings["state_compare_seconds"] = time.monotonic() - tick
        tick = time.monotonic()
        report["source_artifacts_unchanged"] = (file_identity(source) == report["source_checkpoint_identity"] and
                                                prepared_identity(prepared) == report["bound_prepared_artifacts"])
        timings["final_hash_seconds"] = time.monotonic() - tick
        report["passed"] = report["source_artifacts_unchanged"] and all(c["bitwise_equal"] for c in comparisons.values())
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", passed=False, error_type=type(exc).__name__, error=str(exc))
        if "source_checkpoint_identity" in report:
            try:
                report["source_artifacts_unchanged"] = (file_identity(source) == report["source_checkpoint_identity"] and
                                                        prepared_identity(prepared) == report["bound_prepared_artifacts"])
            except Exception:
                report["source_artifacts_unchanged"] = False
    report["seconds"] = timings
    report["wall_seconds"] = time.monotonic() - started
    report["gpu_count"] = 1 if device == "cuda" else 0
    report["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    report["allocated_gpu_hours_process_lower_bound"] = report["gpu_count"] * report["wall_seconds"] / 3600
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "output": str(output / "report.json"),
                      "wall_seconds": report["wall_seconds"], "error": report.get("error")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = run_aa(args.prepared, args.stage1_checkpoint, args.output, device=args.device)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
