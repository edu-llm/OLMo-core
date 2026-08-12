"""Packed token stream + per-condition loss-weight sidecars, with a resumable cursor.

One `tokens.bin` (uint16) is shared by every arm. Each condition gets a
`weights.{condition}.bin` (uint8) of identical length. An arm is chosen by naming
its sidecar, so the arms cannot drift apart in content, order, or exposure counts
-- the failure that made the previous corpus non-comparable.

The cursor is a plain token offset and is saved in the checkpoint, so a resumed
run continues on exactly the batch it would have seen. That matters more than it
sounds: the previous generation lost a split-arm run at 0.87B of 1.0B tokens when
a Colab session ended, and the reported anchor for that arm is a 0.797B-token
snapshot compared against the dense arm's 0.996B. Preemption is the normal case on
both Colab and spot instances, so resume has to be exact rather than best-effort.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from memsplit.model import IGNORE_INDEX


@dataclass
class ShardPaths:
    tokens: Path
    weights: Path | None

    @classmethod
    def for_condition(cls, root: str | Path, condition: str) -> "ShardPaths":
        root = Path(root)
        w = root / f"weights.{condition}.bin"
        return cls(tokens=root / "tokens.bin", weights=w if w.exists() else None)


class PackedDataset:
    """Contiguous-window sampler over one shared token stream.

    `ctx + 1` tokens are read per sequence so inputs and targets can be offset by
    one without a second read. Weights index the *target*, i.e. position i+1.
    """

    def __init__(
        self,
        paths: ShardPaths,
        ctx: int,
        micro_batch_size: int,
        require_weights: bool = False,
        vocab_size: int | None = None,
    ) -> None:
        self.ctx = ctx
        self.micro_batch_size = micro_batch_size
        self.tokens = np.memmap(paths.tokens, dtype=np.uint16, mode="r")
        self.weights: np.memmap | None = None
        if paths.weights is not None:
            self.weights = np.memmap(paths.weights, dtype=np.uint8, mode="r")
            if len(self.weights) != len(self.tokens):
                raise ValueError(
                    f"sidecar length {len(self.weights)} != stream length "
                    f"{len(self.tokens)}; the arms must index the same stream"
                )
        elif require_weights:
            raise FileNotFoundError(f"no sidecar found at {paths.weights}")
        # Fail here, with the offending id, rather than deep inside an embedding
        # lookup a thousand steps in. A corpus tokenised with control tokens at
        # 50257+ against a model with a 512-token vocabulary is a configuration
        # error, and the IndexError it otherwise raises says nothing useful.
        if vocab_size is not None and len(self.tokens):
            hi = int(np.asarray(self.tokens[: min(len(self.tokens), 1 << 22)]).max())
            hi = max(hi, int(np.asarray(self.tokens[-min(len(self.tokens), 1 << 22):]).max()))
            if hi >= vocab_size:
                raise ValueError(
                    f"corpus contains token id {hi} but the model vocabulary is "
                    f"{vocab_size}. Set TrainConfig.vocab_size to match the "
                    "tokenizer that built this corpus."
                )
        self.cursor = 0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.ctx

    def state_dict(self) -> dict:
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state: dict) -> None:
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])

    def next_batch(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        need = self.micro_batch_size * (self.ctx + 1)
        if self.cursor + need > len(self.tokens):
            self.cursor = 0
            self.epoch += 1
        lo = self.cursor
        block = np.asarray(self.tokens[lo : lo + need]).astype(np.int64)
        block = block.reshape(self.micro_batch_size, self.ctx + 1)
        x = torch.from_numpy(block[:, :-1]).to(device)
        y = torch.from_numpy(block[:, 1:]).clone().to(device)

        w = None
        if self.weights is not None:
            wb = np.asarray(self.weights[lo : lo + need]).astype(np.float32)
            wb = wb.reshape(self.micro_batch_size, self.ctx + 1)[:, 1:]
            w = torch.from_numpy(wb.copy()).to(device)
            # Masked targets are also set to IGNORE_INDEX so the two mechanisms
            # agree; the weight then only ever scales a supervised position.
            y[w == 0] = IGNORE_INDEX

        # Advance by ctx (not ctx+1) so windows tile the stream without gaps.
        self.cursor += self.micro_batch_size * self.ctx
        return x, y, w


def log_spaced_steps(max_steps: int, per_decade: int = 6, include: tuple[int, ...] = ()) -> list[int]:
    """Evaluation schedule that is dense early and sparse late.

    This exists because of a specific measurement failure. The previous
    sample-efficiency claim ("10-15x fewer tokens") could not be resolved: the
    split arm's *first* evaluated step was 47, and it was already at 99.8%, so the
    crossing was never bracketed and the ratio is a censored lower bound. A
    linear schedule cannot fix that -- you need points at steps 1, 2, 4, 8, ...
    Everything downstream (`metrics.compute_to_threshold`) refuses unbracketed
    crossings, so the schedule and the metric enforce each other.
    """
    if max_steps < 1:
        return []
    steps: set[int] = {1, max_steps}
    steps.update(int(s) for s in include if 1 <= s <= max_steps)
    decade = 1.0
    while decade <= max_steps:
        for k in range(per_decade):
            v = int(round(decade * (10 ** (k / per_decade))))
            if 1 <= v <= max_steps:
                steps.add(v)
        decade *= 10
    return sorted(steps)
