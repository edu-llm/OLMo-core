"""Read-only checkpoint diagnostics after an acquisition screen has finished.

Run as ``python -m spacing_rerun.diagnose``. This never trains or rewrites the
source run. Declarative reconstruction is a diagnostic, not a factual-QA score.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import SCHEMA
from .common import digest, require, write_json
from .data import normalize
from .evaluation import conditional_scores, generated_answers, metric_means
from .units import LEGACY_POLICY, LEGACY_METRICS, UNIT_METRICS, training_key
from .prepare import load_prepared
from .training import load_checkpoint, load_model, make_optimizer, seed_all


def summarize(facts, scores):
    """Use the same probe/unit/event hierarchy as the acquisition score."""
    require(len(facts) == len(scores) and len(facts) > 0, "Empty or misaligned diagnostic scores")
    metrics = ("loss", "loss_with_eos", "exact_match", "generation_terminated")
    output = {"facts": len(facts), "events": len({f["event"] for f in facts}),
              "units": len({(f["event"], f.get("unit_id", f.get("id", str(i)))) for i, f in enumerate(facts)})}
    for metric in metrics:
        if not all(metric in score for score in scores):
            continue
        output.update(metric_means([dict(fact, **score) for fact, score in zip(facts, scores)], metric))
    return output


def declarative_probes(facts, examples, eos):
    """Use exactly the trained statement tokens; omit EOS/filler from primary NLL."""
    probes = []
    for fact in facts:
        ids = fact["statement_ids"]
        example = examples["old/" + (training_key(fact) if "role" in fact else fact["id"])]
        require(ids and example["input_ids"][0] == eos, "Statement must have its trained initial EOS")
        require(example["input_ids"][1:len(ids) + 1] == ids and
                example["labels"][1:len(ids) + 1] == ids,
                "Diagnostic statement tokens differ from training")
        require(example["input_ids"][len(ids) + 1] == eos and
                example["content_tokens"] == len(ids) + 1,
                "Statement/EOS boundary is inconsistent")
        probes.append({"prompt_ids": [eos], "answer_ids": list(ids)})
    return probes


def answer_span_probe(fact, tokenizer, eos):
    """Return only uniquely matched, token-boundary-safe gold text in a statement."""
    statement, answer = fact["statement"], fact["answer"]
    require(answer.strip(), "Empty canonical answer")
    matches = list(re.finditer(re.escape(answer), statement, flags=re.IGNORECASE))
    if not matches:
        return None, {"id": fact["id"], "reason": "canonical_answer_not_present"}
    if len(matches) != 1:
        return None, {"id": fact["id"], "reason": "canonical_answer_present_multiple_times"}
    encoded = tokenizer(statement, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    require(ids == fact["statement_ids"], "Offset audit tokenizer differs from prepared statement")
    start, end = matches[0].span()
    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        return None, {"id": fact["id"], "reason": "noncontiguous_token_span"}
    first, last = indices[0], indices[-1]
    token_start, token_end = offsets[first][0], offsets[last][1]
    if not (token_start <= start < end <= token_end):
        return None, {"id": fact["id"], "reason": "token_offsets_do_not_cover_answer"}
    if statement[token_start:start].strip() or statement[end:token_end].strip():
        return None, {"id": fact["id"], "reason": "answer_boundary_inside_nonwhitespace_token"}
    probe = {"prompt_ids": [eos] + ids[:first], "answer_ids": ids[first:last + 1]}
    audit = {"id": fact["id"], "reason": "eligible", "character_start": start, "character_end": end,
             "token_start": first, "token_end_exclusive": last + 1,
             "answer_tokens": len(probe["answer_ids"]), "statement_tokens": len(ids)}
    return probe, audit


def diagnose(prepared, checkpoint, output, *, device="cuda", max_runtime_seconds=900, model_factory=None):
    import torch
    started = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    checkpoint, output = Path(checkpoint), Path(output)
    require(not output.exists(), "Diagnostic output already exists; never overwrite")
    manifest, _, examples = load_prepared(prepared)
    require(manifest["mode"] == "development", "These adaptive diagnostics are development-only")
    require(device == "cpu" or torch.cuda.is_available(), "CUDA required for a real model diagnostic")
    require(device == "cpu" or torch.cuda.is_bf16_supported(), "CUDA bf16 unsupported")
    require(0 < max_runtime_seconds < 24 * 3600, "Invalid diagnostic wall-time cap")
    seed_all(manifest["config"]["eval_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    factory = model_factory or load_model
    model, tokenizer = factory(manifest, prepared, device)
    optimizer = make_optimizer(model, manifest["config"])
    progress = load_checkpoint(checkpoint, model, optimizer, manifest["sha256"], device)
    require(progress["status"] == "stage1_complete" and progress["epoch_cursor"] == 0,
            "Diagnose only a completed Stage-1 checkpoint; do not race a live writer")
    # Adam state is verified by load_checkpoint, then released before evaluation.
    del optimizer
    model.eval()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else contextlib.nullcontext
    timings = {"load_seconds": time.monotonic() - started}
    before = checkpoint.stat()
    tick = time.monotonic()
    with checkpoint.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    timings["checkpoint_hash_seconds"] = time.monotonic() - tick
    facts = manifest["split"]["facts"]
    groups = {role: [f for f in facts if f["role"] == role] for role in ("old", "qa")}
    sections = {}

    def check_time():
        require(time.monotonic() - started < max_runtime_seconds, "Diagnostic runtime cap reached")

    def score_pool(name, pool, probes, behavior=False, aliases=True):
        check_time()
        tick = time.monotonic()
        scores = conditional_scores(model, probes, tokenizer.eos_token_id, device,
                                    manifest["config"]["eval_batch_size"], autocast)
        if behavior:
            answers = generated_answers(model, tokenizer, probes, device,
                                        manifest["config"]["generation_batch_size"],
                                        manifest["config"]["max_answer_tokens"], autocast)
            for fact, score, answer in zip(pool, scores, answers):
                score.update(answer)
                score["exact_match"] = int(normalize(answer["prediction"]) in {normalize(a) for a in fact["aliases"]})
        timings[name + "_seconds"] = time.monotonic() - tick
        rows = [dict(id=f["id"], unit_id=f.get("unit_id", f["id"]), event=f["event"], source_role=f["role"], **s)
                for f, s in zip(pool, scores)]
        sections[name] = {"aggregate": summarize(pool, scores), "facts": rows}

    for role, name in (("qa", "qa_teaching"), ("old", "old_qa")):
        pool = groups[role]
        score_pool(name, pool, [f["probe"] for f in pool], behavior=True)
    score_pool("old_statement_content", groups["old"], declarative_probes(groups["old"], examples, tokenizer.eos_token_id))
    span_facts, span_probes, span_audit = [], [], []
    for fact in groups["old"]:
        probe, audit = answer_span_probe(fact, tokenizer, tokenizer.eos_token_id)
        span_audit.append(audit)
        if probe is not None:
            span_facts.append(fact)
            span_probes.append(probe)
    if span_facts:
        score_pool("old_statement_answer_span", span_facts, span_probes)
        # The artificial EOS appended by conditional_scores is not the next source
        # statement token, so discard that non-estimand explicitly.
        section = sections["old_statement_answer_span"]
        for row in section["facts"]:
            row.pop("loss_with_eos", None)
            row.pop("eos_nll", None)
        for key in list(section["aggregate"]):
            if key.startswith("loss_with_eos"):
                del section["aggregate"][key]
    after = checkpoint.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Source checkpoint changed during diagnostic")
    code_commit = os.environ.get("SPACING_CODE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    old, qa, statement = [sections[key]["aggregate"] for key in ("old_qa", "qa_teaching", "old_statement_content")]
    result = {"schema": SCHEMA, "type": "post_acquisition_diagnostic", "status": "complete",
              "rehearsal_unit_policy": manifest.get("rehearsal_unit_policy", LEGACY_POLICY),
              "metric_schema": UNIT_METRICS if "unit_registry" in manifest else LEGACY_METRICS,
              "started_utc": started_utc, "code_commit": code_commit,
              "manifest_sha256": manifest["sha256"], "checkpoint_path": str(checkpoint.resolve()),
              "checkpoint_sha256": checkpoint_hash, "checkpoint_bytes": before.st_size,
              "checkpoint_epoch": progress["epoch"], "checkpoint_global_step": progress["global_step"],
              "acquisition_usable": progress.get("acquisition_usable"), "sections": sections,
              "answer_span_coverage": {"eligible": len(span_facts), "total_old": len(groups["old"]),
                  "reasons": dict(collections.Counter(row["reason"] for row in span_audit)), "audit": span_audit},
              "descriptive_comparisons": {
                  "qa_teaching_minus_old_qa_em": qa["exact_match_event_macro"] - old["exact_match_event_macro"],
                  "old_qa_minus_statement_token_loss": old["loss_event_macro"] - statement["loss_event_macro"],
                  "interpretation": "Training-statement reconstruction, QA-teaching memorization and old-fact QA use different contexts and sometimes different facts. These gaps diagnose transfer; they do not prove factual knowledge or identify a memory mechanism."},
              "timings": timings, "wall_seconds": time.monotonic() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "gpu": torch.cuda.get_device_name() if device == "cuda" else "CPU test",
              "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    write_json(output, result)
    print(json.dumps({"output": str(output), "sections": {k: v["aggregate"] for k, v in sections.items()},
                      "span_coverage": {k: v for k, v in result["answer_span_coverage"].items() if k != "audit"},
                      "timings": timings}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-runtime-seconds", type=int, default=900)
    args = parser.parse_args()
    diagnose(args.prepared, args.checkpoint, args.output, max_runtime_seconds=args.max_runtime_seconds)


if __name__ == "__main__":
    main()
