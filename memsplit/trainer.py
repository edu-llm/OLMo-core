"""Training loop. Fixed-divisor loss, atomic resumable checkpoints, log-spaced evals.

Three properties, each fixing something that went wrong before.

**Arm-symmetric objective.** Loss is summed over supervised positions and divided
by `accum * micro_batch * ctx` -- a constant, identical for every arm. The
previous trainer used `reduction="mean"` over surviving targets, so masking a
fraction *f* inflated each remaining token by `1/(1-f)`; its own records concluded
that no dense-minus-split difference from those runs could be reported as a
treatment effect. The reported training loss is therefore *not* comparable across
arms as a per-predicted-token quantity, and `log.jsonl` records
`supervised_frac` so nobody mistakes a lower split-arm loss for better modelling.

**Exact resume.** Checkpoints carry model, optimizer, step, data cursor and RNG
state, and are written to a temp file then `os.replace`d, so an interrupted write
cannot corrupt the last good one. Cadence is wall-clock, because preemption is
time-based, not step-based. A run that dies at 87% of its budget should resume,
not be reported as a shorter run.

**Log-spaced evaluation.** Snapshots default to `log_spaced_steps`, dense early,
so a fast-converging arm's crossing is bracketed. This is the fix for a
sample-efficiency ratio whose numerator was never measured.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from memsplit import checkpoint_io as cio
from memsplit.data import PackedDataset, ShardPaths, log_spaced_steps
from memsplit.model import build_model


def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.1) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return peak * min_frac
    prog = (step - warmup) / max(total - warmup, 1)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


@dataclass
class TrainConfig:
    run_id: str
    out_dir: str
    data_root: str
    condition: str = "dense"
    preset: str = "d40m"
    ctx: int = 1024
    # Override the preset vocabulary. Needed for a domain tokenizer, and for
    # running a small preset against a real corpus.
    vocab_size: int | None = None
    micro_batch_size: int = 8
    tokens_per_step: int = 524288
    total_tokens: int = 1_000_000_000
    lr: float = 1.5e-3
    warmup_steps: int = 300
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 0
    data_seed: int | None = None  # defaults to `seed`; vary BOTH per replicate
    checkpoint_minutes: float = 20.0
    log_every: int = 20
    snapshot_steps: list[int] = field(default_factory=list)
    device: str = "auto"
    compile: bool = False
    # May be a local directory or an `s3://` prefix. Platform jobs pass
    # `$EDULLM_CHECKPOINT_DIR`, which is an S3 URI, so `Path(...).exists()` on it
    # is always False -- see checkpoint_io for the failure that causes.
    checkpoint_dir: str | None = None
    resume_required: bool = True

    @property
    def total_steps(self) -> int:
        return max(1, self.total_tokens // self.tokens_per_step)

    @property
    def accum(self) -> int:
        per = self.micro_batch_size * self.ctx
        if self.tokens_per_step % per != 0:
            raise ValueError(
                f"tokens_per_step {self.tokens_per_step} is not divisible by "
                f"micro_batch_size*ctx = {per}"
            )
        return self.tokens_per_step // per

    @property
    def loss_divisor(self) -> float:
        """The fixed constant both arms divide by. Never the surviving count."""
        return float(self.accum * self.micro_batch_size * self.ctx)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "snapshots").mkdir(exist_ok=True)
        self.ckpt_root = cfg.checkpoint_dir or str(self.out)
        self.guard = cio.ResumeGuard(self.ckpt_root, enabled=cfg.resume_required)

        # Randomise weight init AND data order per replicate. Randomising init
        # alone converges to the equivalent of about two ideal runs; randomising
        # both reaches far more for the same compute, and costs nothing.
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed % (2**32))

        overrides = {"ctx": cfg.ctx}
        if cfg.vocab_size is not None:
            overrides["vocab_size"] = cfg.vocab_size
        self.model = build_model(cfg.preset, **overrides).to(self.device)
        if cfg.compile and self.device.type == "cuda":
            self.model = torch.compile(self.model)
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        self.data = PackedDataset(
            ShardPaths.for_condition(cfg.data_root, cfg.condition),
            ctx=cfg.ctx,
            micro_batch_size=cfg.micro_batch_size,
            vocab_size=self.model.cfg.vocab_size,
        )
        self.step = 0
        self.snapshot_steps = cfg.snapshot_steps or log_spaced_steps(cfg.total_steps)
        self._last_ckpt = time.time()

    # ------------------------------------------------------------- checkpoints

    @property
    def ckpt_path(self) -> str:
        return cio.join(self.ckpt_root, "ckpt.pt")

    def save_checkpoint(self) -> None:
        """Atomic for local paths (temp+replace) and for S3 (single PUT)."""
        payload = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "data": self.data.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "cfg": vars(self.cfg),
        }
        cio.save_obj(payload, self.ckpt_path)
        self._last_ckpt = time.time()

    def load_checkpoint(self) -> bool:
        # cio.exists understands s3://. A plain Path().exists() here is the bug
        # that made a sibling repo repeat every attempt from step 0.
        if not cio.exists(self.ckpt_path):
            return False
        p = cio.load_obj(self.ckpt_path, map_location=self.device)
        self.model.load_state_dict(p["model"])
        self.opt.load_state_dict(p["opt"])
        self.step = int(p["step"])
        self.data.load_state_dict(p["data"])
        torch.set_rng_state(p["torch_rng"].cpu() if hasattr(p["torch_rng"], "cpu") else p["torch_rng"])
        random.setstate(p["py_rng"])
        np.random.set_state(p["np_rng"])
        return True

    def save_snapshot(self) -> str:
        path = cio.join(self.ckpt_root, "snapshots", f"step{self.step:07d}.pt")
        cio.save_obj({"model": self.model.state_dict(), "step": self.step,
                      "preset": self.cfg.preset, "ctx": self.cfg.ctx}, path)
        return path

    # -------------------------------------------------------------- the loop

    def _log(self, row: dict) -> None:
        with open(self.out / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def train(self, resume: str | bool = "auto", max_steps: int | None = None) -> "Trainer":
        cfg = self.cfg
        loaded = bool(resume in (True, "auto") and self.load_checkpoint())
        if loaded:
            print(f"[{cfg.run_id}] resumed at step {self.step}")
        attempt = self.guard.check_and_record(loaded)
        print(f"[{cfg.run_id}] attempt {attempt}, starting at step {self.step}")

        total = cfg.total_steps if max_steps is None else min(cfg.total_steps, max_steps)
        divisor = cfg.loss_divisor
        ema: float | None = None

        while self.step < total:
            lr = cosine_lr(self.step, cfg.lr, cfg.warmup_steps, cfg.total_steps)
            for group in self.opt.param_groups:
                group["lr"] = lr

            t0 = time.time()
            self.opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            supervised = 0
            total_targets = 0
            for _ in range(cfg.accum):
                x, y, w = self.data.next_batch(self.device)
                _, loss = self.model(x, y, target_weights=w, loss_divisor=divisor)
                loss.backward()
                loss_sum += float(loss.detach())
                supervised += int((y != -100).sum())
                total_targets += int(y.numel())
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.opt.step()
            self.step += 1

            dt = max(time.time() - t0, 1e-9)
            if self.step % cfg.log_every == 0 or self.step == 1:
                ema = loss_sum if ema is None else 0.95 * ema + 0.05 * loss_sum
                self._log({
                    "step": self.step,
                    "loss": round(loss_sum, 5),
                    "loss_ema": round(ema, 5),
                    "lr": lr,
                    "tok_s": round(cfg.tokens_per_step / dt, 1),
                    "epoch": self.data.epoch,
                    # Recorded so a lower split-arm loss is never mistaken for
                    # better modelling: the arms score different target sets.
                    "supervised_frac": round(supervised / max(total_targets, 1), 5),
                    "loss_divisor": divisor,
                })

            if self.step in self.snapshot_steps:
                self.save_snapshot()
            if (time.time() - self._last_ckpt) / 60.0 >= cfg.checkpoint_minutes:
                self.save_checkpoint()

        self.save_checkpoint()
        self.save_snapshot()
        return self
