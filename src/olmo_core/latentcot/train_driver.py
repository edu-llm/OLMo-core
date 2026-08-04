"""
Training-driver core for Phase 8 (importable so it can be unit-tested).

A direct per-arm training loop over :class:`LatentCotDataset` using :func:`arm_loss` — the same
loss the ``CodiTransformerTrainModule`` uses — with AdamW + gradient clipping (the deep K-step
continuous-thought graph needs the clip). The CODI student is processed per example, which
doesn't fit the framework Trainer's token-array ``DataLoader``; this loop is the pragmatic
equivalent for the research runs.
"""

from pathlib import Path
from typing import Iterator, List

import torch

from .arms import Arm
from .data.dataset import LatentCotDataset, codi_collate
from .loss import arm_loss

__all__ = ["build_model", "load_checkpoint", "iter_batches", "train_arm"]


def load_checkpoint(model, path: str, *, strict: bool = True) -> None:
    """
    Load weights into ``model`` from either a plain ``.pt`` state_dict (produced by
    ``train_codi.py``) or an OLMo-core checkpoint directory/URL — local **or remote**
    (e.g. ``s3://…``, loaded via ``load_model_and_optim_state`` with ``pre_download``).

    Used to fork every arm from the shared base checkpoint (the "best model").
    """
    if Path(str(path)).is_file():
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=strict)
    else:
        from olmo_core.distributed.checkpoint import load_model_and_optim_state

        load_model_and_optim_state(str(path), model, pre_download=True, strict=strict)


def build_model(rung: str, *, init_seed: int, device: str = "cpu"):
    """
    Build a model at ``rung`` (a ``TransformerConfig`` factory name) with deterministic init.

    All arms must share the same ``init_seed`` so they start from identical weights — the
    shared "base checkpoint" the confound control requires.
    """
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.utils import seed_all

    from .tokens import TOKENIZER_CONFIG

    seed_all(init_seed)
    config = getattr(TransformerConfig, rung)(vocab_size=TOKENIZER_CONFIG.padded_vocab_size())
    return config.build(init_device=device)


def iter_batches(
    dataset: LatentCotDataset, batch_size: int, steps: int, seed: int
) -> Iterator[dict]:
    """Yield ``steps`` shuffled minibatches (cycling the dataset) as ``codi_collate`` dicts."""
    generator = torch.Generator().manual_seed(seed)
    n = len(dataset)
    order: List[int] = torch.randperm(n, generator=generator).tolist()
    cursor = 0
    for _ in range(steps):
        idx: List[int] = []
        for _ in range(batch_size):
            if cursor >= n:
                order = torch.randperm(n, generator=generator).tolist()
                cursor = 0
            idx.append(order[cursor])
            cursor += 1
        yield codi_collate([dataset[j] for j in idx])


def train_arm(
    model,
    arm: Arm,
    dataset: LatentCotDataset,
    *,
    steps: int,
    batch_size: int = 16,
    lr: float = 3e-4,
    distill_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    log_every: int = 100,
) -> List[dict]:
    """Train ``model`` on one arm; return a list of logged metric snapshots."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history: List[dict] = []
    for step, batch in enumerate(iter_batches(dataset, batch_size, steps, seed)):
        opt.zero_grad(set_to_none=True)
        loss, metrics = arm_loss(
            model,
            batch["examples"],
            mode=arm.arm_mode,
            distill_weight=distill_weight,
            vocab_reg=arm.vocab_reg,
            vocab_reg_weight=arm.vocab_reg_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append({"step": step, "loss": float(loss.detach()), **metrics})
    return history
