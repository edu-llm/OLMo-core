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

__all__ = ["resolve_device", "build_model", "load_checkpoint", "iter_batches", "train_arm"]


def resolve_device(device: str = "auto") -> str:
    """
    Resolve a device string: ``"auto"`` -> ``"cuda"`` if available else ``"cpu"``, else pass
    the given value through unchanged. Shared by the training and eval scripts so all of them
    land on the GPU when one is present.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
    warmup_steps: int = 200,
    distill_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    log_every: int = 100,
) -> List[dict]:
    """
    Train ``model`` on one arm; return a list of logged metric snapshots.

    Because every arm forks the *same* pretrained base (the "best model"), this is a
    fine-tune, not a from-scratch run — so the LR follows a warmup-stable-decay schedule
    (the :class:`~olmo_core.optim.WSD` the pre-registration and ``preflight.py`` reference),
    not a constant LR. Linear ``warmup_steps`` ease the optimizer into the pretrained weights
    (a full-LR first step on a good checkpoint can spike the loss and erase what we forked
    it for); a linear decay tail anneals at the end. The schedule lives *here*, in the shared
    loop, so it is byte-identical across arms and stays confound-clean.
    """
    from olmo_core.optim import WSD

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # min(...) keeps warmup < horizon on the short smoke runs; decay_fraction matches the WSD default.
    scheduler = WSD(warmup=max(1, min(warmup_steps, steps - 1)), decay_fraction=0.1)
    history: List[dict] = []
    for step, batch in enumerate(iter_batches(dataset, batch_size, steps, seed)):
        lr_t = scheduler.get_lr(lr, step, steps)
        for group in opt.param_groups:
            group["lr"] = lr_t
        opt.zero_grad(set_to_none=True)
        loss, metrics = arm_loss(
            model,
            batch["examples"],
            mode=arm.arm_mode,
            distill_weight=distill_weight,
            vocab_reg=arm.vocab_reg,
            vocab_reg_weight=arm.vocab_reg_weight,
            vocab_reg_entropy_floor=arm.vocab_reg_entropy_floor,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append(
                {"step": step, "lr": float(lr_t), "loss": float(loss.detach()), **metrics}
            )
    return history
