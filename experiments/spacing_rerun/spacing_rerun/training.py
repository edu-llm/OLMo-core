from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from . import SCHEMA
from .common import append_json, digest, read_json, require, write_json
from .evaluation import evaluate
from .prepare import load_prepared
from .schedule import ARMS, stage1_epoch
from .units import LEGACY_POLICY, UNIT_POLICY, UNIT_METRICS, schedule_records, training_key


def capture_rng():
    import torch
    import numpy as np
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import torch
    import numpy as np
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def seed_all(seed):
    import torch
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(manifest, prepared, device):
    import hf_olmo  # registers the checkpoint's legacy architecture with Transformers
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
    config = AutoConfig.from_pretrained(manifest["model"], revision=manifest["model_revision"])
    config.init_device = "cpu"
    config.flash_attention = False
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], revision=manifest["model_revision"], config=config,
        torch_dtype=torch.float32, trust_remote_code=False)
    model.to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path(prepared) / "tokenizer")
    require(digest(tokenizer.backend_tokenizer.to_str()) == manifest["tokenizer_sha256"],
            "Tokenizer artifact differs from prepared manifest")
    return model, tokenizer


def make_optimizer(model, config):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95),
                             eps=1e-8, weight_decay=0.1)


def save_checkpoint(path, model, optimizer, progress, manifest_hash):
    import torch
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    started = time.monotonic()
    state = {"schema": SCHEMA, "manifest_sha256": manifest_hash, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "progress": progress, "rng": capture_rng()}
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return time.monotonic() - started


def load_checkpoint(path, model, optimizer, manifest_hash, device):
    import torch
    state = torch.load(path, map_location=device, weights_only=False)
    require(state["schema"] == SCHEMA and state["manifest_sha256"] == manifest_hash,
            "Checkpoint belongs to a different manifest")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    # Assert full restoration, including moments and optimizer step counters.
    for key, tensor in model.state_dict().items():
        require(torch.equal(tensor, state["model"][key]), f"Model restore mismatch: {key}")
    loaded_opt = optimizer.state_dict()
    require(loaded_opt["param_groups"] == state["optimizer"]["param_groups"], "Optimizer group restore mismatch")
    for pid, values in state["optimizer"]["state"].items():
        for key, value in values.items():
            actual = loaded_opt["state"][pid][key]
            require(torch.equal(actual.to(value.device), value) if torch.is_tensor(value) else actual == value,
                    f"Optimizer moment restore mismatch {pid}/{key}")
    restore_rng(state["rng"])
    return state["progress"]


def train_update(model, optimizer, row, examples, config, global_step, device, autocast):
    import torch
    import torch.nn.functional as F
    optimizer.zero_grad(set_to_none=True)
    lr = config["learning_rate"] * min(1.0, (global_step + 1) / max(1, config["warmup_steps"]))
    for group in optimizer.param_groups:
        group["lr"] = lr
    denominator = sum(examples[key]["loss_tokens"] for key in row)
    accumulated, token_count = 0.0, 0
    model.train()
    for start in range(0, len(row), config["microbatch_size"]):
        batch = [examples[key] for key in row[start:start + config["microbatch_size"]]]
        ids = torch.tensor([e["input_ids"] for e in batch], device=device)
        labels = torch.tensor([e["labels"] for e in batch], device=device)
        mask = torch.tensor([e["attention_mask"] for e in batch], device=device)
        with autocast():
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss_sum = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                       labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
            loss = loss_sum / denominator
        require(bool(torch.isfinite(loss)), "Nonfinite training loss; this is a scientific failure, not an automatic retry")
        loss.backward()
        accumulated += float(loss_sum.detach())
        token_count += int((labels[:, 1:] != -100).sum())
    require(token_count == denominator, "Realized loss mask differs from planned budget")
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
    optimizer.step()
    return {"loss": accumulated / denominator, "loss_tokens": denominator, "gradient_norm": norm, "lr": lr,
            "content_tokens": sum(examples[k]["content_tokens"] for k in row),
            "generic_filler_tokens": sum(examples[k]["generic_filler_tokens"] for k in row),
            "padding_tokens": sum(examples[k]["padding_tokens"] for k in row)}


def recover_exposure_log(path, cursor):
    """Discard only updates after the last atomic checkpoint, which will be replayed."""
    path = Path(path)
    lines = path.read_text().splitlines() if path.exists() else []
    require(len(lines) >= cursor, "Realized exposure log is shorter than checkpoint cursor")
    committed = lines[:cursor]
    for i, line in enumerate(committed):
        row = json.loads(line)
        require(row["cursor"] == i + 1 and row["row_sha256"] == digest(row["examples"]),
                "Realized exposure log is corrupt")
    path.write_text("".join(line + "\n" for line in committed))


class Session:
    def __init__(self, prepared, output, *, device="cuda", model_factory=None):
        import torch
        self.started = time.monotonic()
        self.manifest, self.schedule, self.examples = load_prepared(prepared)
        self.config = self.manifest["config"]
        self.facts = self.manifest["split"]["facts"]
        self.training_records = schedule_records(self.facts, self.manifest.get("unit_registry"),
                                                self.manifest.get("qa_teaching_pool", {}).get("records"))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.device = device
        require(device == "cpu" or torch.cuda.is_available(), "CUDA unavailable; no scientific training on login CPU")
        require(device == "cpu" or torch.cuda.is_bf16_supported(), "Configured bf16 requires supported GPU")
        seed_all(self.config["train_seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        self.model, self.tokenizer = (model_factory(self.manifest, prepared, device) if model_factory else
                                      load_model(self.manifest, prepared, device))
        self.optimizer = make_optimizer(self.model, self.config)
        self.autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else contextlib.nullcontext
        self.stop_requested = False
        self.progress = {"global_step": 0, "cursor": 0, "epoch": 0, "epoch_cursor": 0,
                         "continuation_cursor": 0, "status": "running", "phase": "stage1",
                         "last_old_update": {}, "last_old_newqa_clock": {}, "new_qa_tokens_total": 0}
        self.seconds = {"initialization": time.monotonic() - self.started, "training": 0.0,
                        "evaluation": 0.0, "checkpoint": 0.0}
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, lambda *_: setattr(self, "stop_requested", True))
        attempts = list(self.output.glob("attempt-*.json"))
        require(len(attempts) < self.config["max_attempts"], "Maximum infrastructure attempts reached")
        self.attempt = len(attempts) + 1
        self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.write_attempt("running")

    def write_attempt(self, status):
        import torch
        try:
            commit = os.environ.get("SPACING_CODE_COMMIT") or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
        require(self.device != "cuda" or (len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)),
                "GPU runs require a full code commit via git or SPACING_CODE_COMMIT")
        write_json(self.output / f"attempt-{self.attempt:02}.json", {
            "schema": SCHEMA, "status": status, "started_utc": self.started_utc,
            "manifest_sha256": self.manifest["sha256"], "git_commit": commit,
            "wall_seconds": time.monotonic() - self.started, "seconds": self.seconds,
            "gpu_count": 1 if self.device == "cuda" else 0,
            "gpu": torch.cuda.get_device_name() if self.device == "cuda" else "CPU test",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "software": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "ai2-olmo", "numpy") if importlib.util.find_spec(name.replace("-", "_")) or name == "ai2-olmo"},
            "progress": self.progress})

    def checkpoint(self, name="latest.pt"):
        self.seconds["checkpoint"] += save_checkpoint(self.output / name, self.model, self.optimizer,
                                                       self.progress, self.manifest["sha256"])
        self.write_attempt("running")

    def resume(self, checkpoint=None, fork=False):
        path = Path(checkpoint) if checkpoint else self.output / "latest.pt"
        if path.exists():
            tick = time.monotonic()
            self.progress = load_checkpoint(path, self.model, self.optimizer, self.manifest["sha256"], self.device)
            self.seconds["checkpoint"] += time.monotonic() - tick
            if fork:
                require(self.progress["epoch_cursor"] == 0 and self.progress["status"] == "stage1_complete",
                        "Fork requires a completed shared Stage-1 state")
                require(self.progress["global_step"] >= self.config["warmup_steps"], "Warmup incomplete before schedule comparison")
                self.progress = dict(self.progress, cursor=0, continuation_cursor=0, phase="stage2", status="running")
            recover_exposure_log(self.output / "exposures.jsonl", self.progress["cursor"])
            self.audit_exposures()
            return True
        return False

    def check_time(self):
        if self.stop_requested or time.monotonic() - self.started > self.config["max_runtime_seconds"]:
            self.checkpoint()
            self.write_attempt("checkpointed_time_limit")
            raise SystemExit(75)

    def update(self, row, phase, index):
        self.check_time()
        tick = time.monotonic()
        metrics = train_update(self.model, self.optimizer, row, self.examples, self.config,
                               self.progress["global_step"], self.device, self.autocast)
        self.seconds["training"] += time.monotonic() - tick
        self.progress["global_step"] += 1
        self.progress["cursor"] += 1
        self.progress["new_qa_tokens_total"] += sum(self.examples[k]["loss_tokens"]
                                                     for k in row if k.startswith(("new/", "qa/")))
        for key in row:
            if key.startswith("old/"):
                fact_id = key.split("/", 1)[1]
                self.progress["last_old_update"][fact_id] = self.progress["global_step"]
                self.progress["last_old_newqa_clock"][fact_id] = self.progress["new_qa_tokens_total"]
        append_json(self.output / "exposures.jsonl", {
            "schema": SCHEMA, "cursor": self.progress["cursor"], "global_step": self.progress["global_step"],
            "phase": phase, "phase_step": index, "examples": row, "row_sha256": digest(row),
            "example_content_sha256": [self.examples[k]["content_sha256"] for k in row],
            "rehearsal_unit_policy": self.manifest.get("rehearsal_unit_policy", LEGACY_POLICY),
            "loss_weight": 1.0, "new_qa_tokens_total": self.progress["new_qa_tokens_total"], **metrics})
        if self.progress["cursor"] % self.config["log_every_updates"] == 0:
            print(json.dumps({"phase": phase, "step": index, **metrics}), flush=True)

    def maybe_checkpoint(self):
        if self.progress["cursor"] % self.config["checkpoint_every_updates"] == 0:
            self.checkpoint()

    def audit_exposures(self):
        path = self.output / "exposures.jsonl"
        logs = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        require(len(logs) == self.progress["cursor"], "Realized exposure count differs from checkpoint")
        epochs = {}
        rows_per_epoch = len(stage1_epoch(self.training_records, 0, self.config))
        for entry in logs:
            if entry["phase"] == "stage1":
                epoch, index = divmod(entry["phase_step"], rows_per_epoch)
                if epoch not in epochs:
                    epochs[epoch] = stage1_epoch(self.training_records, epoch, self.config)
                expected = epochs[epoch][index]
            else:
                expected = self.schedule["arms"][self.progress["arm"]][entry["phase_step"]]
            require(entry["examples"] == expected, "Realized exposure sequence differs from compiled plan")
            require(entry["example_content_sha256"] == [self.examples[k]["content_sha256"] for k in expected],
                    "Realized content/loss masks differ from compiled plan")
            require(entry["loss_tokens"] == sum(self.examples[k]["loss_tokens"] for k in expected),
                    "Realized training token count differs")

    def evaluate(self, tag, behavior=True, generic=False, stage2_step=None):
        self.check_time()
        path = self.output / "evaluations" / f"{tag}.json"
        if path.exists():
            previous = read_json(path)
            require(previous["manifest_sha256"] == self.manifest["sha256"] and
                    previous["global_step"] == self.progress["global_step"], "Stale evaluation artifact")
            return previous
        rng = capture_rng()
        result = evaluate(self.model, self.tokenizer, self.facts, self.config, self.device,
                          self.autocast, behavior, self.manifest["generic_eval"] if generic else None)
        restore_rng(rng)
        result.update({"schema": SCHEMA, "manifest_sha256": self.manifest["sha256"],
                       "rehearsal_unit_policy": self.manifest.get("rehearsal_unit_policy", LEGACY_POLICY),
                       "tag": tag, "global_step": self.progress["global_step"],
                       "stage2_step": stage2_step, "epoch": self.progress["epoch"]})
        if stage2_step is not None:
            result["global_buffer_delay"] = stage2_step - self.schedule["stage2_steps"]
            result["updates_since_last_old_exposure"] = {
                key: self.progress["global_step"] - step for key, step in self.progress["last_old_update"].items()}
            result["new_qa_tokens_since_last_old_exposure"] = {
                key: self.progress["new_qa_tokens_total"] - clock
                for key, clock in self.progress["last_old_newqa_clock"].items()}
            result["old_probe_to_exposure_unit"] = {f["id"]: training_key(f) for f in self.facts if f["role"] == "old"}
        self.seconds["evaluation"] += result["wall_seconds"]
        write_json(path, result)
        summary = {"evaluation": tag, "old": result["variants"]["canonical"]["aggregate"]["old"]}
        if result.get("qa_teaching_diagnostic") is not None:
            summary["qa_teaching"] = result["qa_teaching_diagnostic"]["aggregate"]
        print(json.dumps(summary), flush=True)
        return result


def run_acquisition(prepared, output, *, device="cuda", fixed_exposures=None, preregistration=None, model_factory=None):
    manifest, _, _ = load_prepared(prepared)
    if manifest["mode"] == "confirmation":
        prereg = verify_preregistration(preregistration, manifest)
        require(fixed_exposures == prereg["E"], "Fixed Stage-1 dose must match preregistration")
    s = Session(prepared, output, device=device, model_factory=model_factory)
    if fixed_exposures is None:
        require(s.manifest["mode"] == "development", "Adaptive acquisition is development-only")
        milestones = s.config["acquisition_exposures"]
    else:
        milestones = [fixed_exposures]
    s.resume()
    if s.progress["status"] == "stage1_complete":
        s.write_attempt("already_complete")
        return
    if s.progress["global_step"] == 0:
        s.evaluate("baseline", behavior=True)
    accepted = False
    while True:
        epoch = s.progress["epoch"]
        if epoch in milestones and s.progress["epoch_cursor"] == 0:
            result = s.evaluate(f"acquisition-E{epoch:03}", behavior=True, generic=True)
            em = result["variants"]["canonical"]["aggregate"]["old"]["exact_match_event_macro"]
            if (fixed_exposures is not None or em >= s.config["acquisition_target_min"]) and s.progress["global_step"] >= s.config["warmup_steps"]:
                accepted = fixed_exposures is not None or em <= s.config["acquisition_target_max"]
                break
        if epoch >= max(milestones):
            break
        rows = stage1_epoch(s.training_records, epoch, s.config)
        for i in range(s.progress["epoch_cursor"], len(rows)):
            s.update(rows[i], "stage1", epoch * len(rows) + i)
            s.progress["epoch_cursor"] = i + 1
            s.maybe_checkpoint()
        s.progress["epoch"] = epoch + 1
        s.progress["epoch_cursor"] = 0
        if s.progress["epoch"] in milestones:
            s.checkpoint()
    s.progress["status"] = "stage1_complete"
    s.progress["acquisition_usable"] = accepted
    s.audit_exposures()
    s.checkpoint("stage1.pt")
    s.checkpoint()
    write_json(s.output / "acquisition_decision.json", {
        "schema": SCHEMA, "manifest_sha256": s.manifest["sha256"],
        "mode": s.manifest["mode"], "selected_E": s.progress["epoch"],
        "acquisition_usable": accepted, "confirmation_ready": False,
        "next_step": "Run development interference diagnostics" if accepted else "Revise the acquisition assay before continuation"})
    s.write_attempt("complete")


def verify_preregistration(path, manifest):
    require(manifest.get("rehearsal_unit_policy") == UNIT_POLICY and manifest.get("unit_registry"),
            "Confirmation requires distinct supporting-statement rehearsal units")
    require(path is not None, "Confirmation requires a frozen preregistration")
    prereg = read_json(path)
    required = ("frozen_utc", "n", "replicate_manifest_sha256", "E", "primary_delay", "loss_margin",
                "hardware", "measured_cost", "power_scenarios", "assay_usable", "maximum_attempts",
                "secondary_holm_family", "missing_pair_rule", "rehearsal_unit_policy", "metric_schema")
    require(all(k in prereg for k in required), "Incomplete preregistration fields")
    require(prereg["rehearsal_unit_policy"] == UNIT_POLICY and prereg["metric_schema"] == UNIT_METRICS,
            "Preregistered rehearsal units/aggregation differ")
    require(isinstance(prereg["n"], int) and prereg["n"] >= 2 and isinstance(prereg["E"], int) and prereg["E"] > 0,
            "Invalid preregistered replication/dose")
    require(prereg["loss_margin"] > 0 and prereg["primary_delay"] > 0, "Margins/delays must be positive")
    datetime.datetime.fromisoformat(prereg["frozen_utc"])
    require(prereg["assay_usable"] is True and manifest["audit_complete"], "Assay or data audit incomplete")
    require(manifest["sha256"] in prereg["replicate_manifest_sha256"], "Replicate not in frozen design")
    require(len(prereg["replicate_manifest_sha256"]) == prereg["n"], "Preregistered n disagrees with manifests")
    require(prereg["primary_delay"] == manifest["config"]["primary_delay"] and prereg["E"] > 0,
            "Preregistered endpoint/recipe mismatch")
    require(prereg["maximum_attempts"] == manifest["config"]["max_attempts"], "Attempt budget mismatch")
    return prereg


def run_arm(prepared, output, shared_checkpoint, arm, *, preregistration=None, device="cuda", model_factory=None):
    require(arm in ARMS, "Unknown arm")
    manifest, _, _ = load_prepared(prepared)
    if manifest["mode"] == "confirmation":
        verify_preregistration(preregistration, manifest)
    s = Session(prepared, output, device=device, model_factory=model_factory)
    if s.manifest["mode"] == "confirmation":
        prereg = verify_preregistration(preregistration, s.manifest)
    else:
        prereg = None
    if not s.resume():
        require(Path(shared_checkpoint).exists(), "Missing shared Stage-1 checkpoint")
        s.resume(shared_checkpoint, fork=True)
        require(s.progress.get("acquisition_usable"), "Development acquisition failed; diagnose before continuation")
        if prereg:
            require(s.progress["epoch"] == prereg["E"], "Stage-1 dose differs from preregistration")
        s.progress["arm"] = arm
        s.checkpoint()
    require(s.progress["arm"] == arm, "Cannot resume a different arm in this directory")
    stage2 = s.schedule["stage2_steps"]
    checkpoints = {stage2 + d for d in s.config["eval_delays"]}
    checkpoints |= {0, stage2, len(s.schedule["arms"][arm])}
    checkpoints |= {round(stage2 * f) for f in s.config["stage2_eval_fractions"]}
    cursor = s.progress["continuation_cursor"]
    if cursor in checkpoints:
        s.evaluate(f"stage2-{cursor:06}", behavior=(cursor == 0 or cursor >= stage2),
                   generic=(cursor == 0 or cursor == len(s.schedule["arms"][arm])), stage2_step=cursor)
    for i in range(s.progress["continuation_cursor"], len(s.schedule["arms"][arm])):
        s.update(s.schedule["arms"][arm][i], "stage2" if i < stage2 else "buffer", i)
        s.progress["continuation_cursor"] = i + 1
        s.maybe_checkpoint()
        if i + 1 in checkpoints:
            s.checkpoint()
            s.evaluate(f"stage2-{i + 1:06}", behavior=(i + 1 >= stage2),
                       generic=(i + 1 == len(s.schedule["arms"][arm])), stage2_step=i + 1)
    s.progress["status"] = "complete"
    s.audit_exposures()
    s.checkpoint()
    s.write_attempt("complete")
