"""
Training-driver core for Phase 8 (importable so it can be unit-tested).

A direct per-arm training loop over :class:`LatentCotDataset` using :func:`arm_loss` — the same
loss the ``CodiTransformerTrainModule`` uses — with AdamW + gradient clipping (the deep K-step
continuous-thought graph needs the clip). The CODI student is processed per example, which
doesn't fit the framework Trainer's token-array ``DataLoader``; this loop is the pragmatic
equivalent for the research runs.
"""

import contextlib
import json
from pathlib import Path
from typing import ContextManager, Iterator, List, Optional, Protocol

import torch

from .arms import Arm
from .data.dataset import codi_collate
from .loss import arm_loss

__all__ = [
    "PRECISIONS",
    "autocast_ctx",
    "build_model",
    "configure_precision",
    "is_remote",
    "iter_batches",
    "load_checkpoint",
    "publish_artifact",
    "resolve_device",
    "train_arm",
]

PRECISIONS = ("fp32", "bf16")
"""Valid ``precision`` values. ``fp32`` is bit-identical to the pre-precision-flag driver."""


class _Indexable(Protocol):
    """The minimal dataset interface :func:`iter_batches` needs — a ``LatentCotDataset`` or a
    ``torch.utils.data.Subset`` of one (the driver carves a validation split off the train set)."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict: ...


def resolve_device(device: str = "auto") -> str:
    """
    Resolve a device string: ``"auto"`` -> ``"cuda"`` if available else ``"cpu"``, else pass
    the given value through unchanged. Shared by the training and eval scripts so all of them
    land on the GPU when one is present.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def is_remote(path) -> bool:
    """
    Whether ``path`` is a remote URI (``s3://``, ``gs://``, …) rather than a local path.

    Worth being explicit about why this exists: :class:`pathlib.Path` silently *mangles* a URI —
    ``Path("s3://bucket/k")`` is ``PosixPath("s3:/bucket/k")``, a **relative local** path — so
    handing a checkpoint URI to code that assumes `Path` writes a directory literally named
    ``s3:`` next to the process and loses it when the container exits. No exception is raised.
    The eduLLM platform's ``$EDULLM_CHECKPOINT_DIR`` is exactly such a URI.

    :param path: A path or URI.
    :returns: ``True`` if it is a URL/URI.
    """
    from olmo_core.io import is_url

    return is_url(str(path))


def publish_artifact(local_path: Path, remote_dir: Optional[str]) -> None:
    """
    Mirror a just-written local artifact to ``remote_dir`` (a URI), if one is configured.

    A no-op when ``remote_dir`` is ``None`` (the ordinary local-disk case). Uploads overwrite:
    a rolling checkpoint reuses its name, and a re-run of the same step should replace, not fail.

    :param local_path: The file that was just written locally.
    :param remote_dir: Destination URI prefix, or ``None`` to skip.
    """
    if remote_dir is None:
        return
    from olmo_core.io import upload

    upload(local_path, f"{str(remote_dir).rstrip('/')}/{local_path.name}", save_overwrite=True)


def _check_precision(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
    return precision


def configure_precision(precision: str, device: str) -> None:
    """
    Set global matmul precision for a run. Call once, before training.

    ``bf16`` also turns on TF32 for the matmuls that stay in fp32 (norms, the loss, the
    regularizer targets) — on an A100 strict fp32 matmul peaks at ~19.5 TFLOPS against
    ~312 for bf16, so leaving TF32 off costs most of the tensor-core throughput on those
    ops. ``fp32`` deliberately leaves both alone so a run is bit-reproducible.

    No-op off CUDA, so CPU tests and this repo's Mac development path are unaffected.

    :param precision: One of :data:`PRECISIONS`.
    :param device: The resolved device string (see :func:`resolve_device`).

    :raises ValueError: If ``precision`` is not a valid choice.
    """
    if _check_precision(precision) == "bf16" and device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def autocast_ctx(precision: str, device: str) -> ContextManager:
    """
    The autocast context for a forward pass under ``precision``.

    Returns a null context for ``fp32`` **and** for any non-CUDA device: bf16 autocast on CPU
    is not a throughput win here and would change the numerics the CPU test suite pins, so the
    fast path is GPU-only by design. bf16 needs no ``GradScaler`` (unlike fp16), and parameters
    stay fp32 — autocast only casts the ops.

    Wrap the *forward* in this; call ``loss.backward()`` outside it (autograd replays each op in
    the dtype it ran in). Because it lives in the shared driver, every arm gets the identical
    treatment, so it cannot become a confound.

    :param precision: One of :data:`PRECISIONS`.
    :param device: The resolved device string.

    :returns: A context manager — ``torch.autocast`` or ``contextlib.nullcontext``.

    :raises ValueError: If ``precision`` is not a valid choice.
    """
    if _check_precision(precision) == "bf16" and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


def iter_batches(dataset: _Indexable, batch_size: int, steps: int, seed: int) -> Iterator[dict]:
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
    dataset: _Indexable,
    *,
    steps: int,
    batch_size: int = 16,
    lr: float = 3e-4,
    warmup_steps: int = 200,
    distill_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    log_every: int = 100,
    save_dir: Optional[Path] = None,
    save_every: int = 0,
    keep_last: int = 2,
    val_examples: Optional[List[dict]] = None,
    precision: str = "bf16",
    remote_dir: Optional[str] = None,
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

    **Checkpointing (optional).** With ``save_dir`` set and ``save_every > 0``, every
    ``save_every`` steps (and on the final step) the loop writes a rolling checkpoint
    ``save_dir/stepN.pt`` and prunes to the most recent ``keep_last`` (so a crash loses at most
    one interval, at bounded disk cost). If ``val_examples`` is given it is scored at each of
    those points and the best-so-far weights are copied to ``save_dir/best.pt`` (with the step
    and validation accuracy in ``save_dir/best.json``). The validation set **must** be held out
    from the gate test set — selecting "best" on the test set would be model selection on the
    very data the gates score. This is confound-clean: the policy is identical across arms.

    :param save_dir: Directory for rolling/best checkpoints; ``None`` disables checkpointing.
    :param save_every: Save a rolling checkpoint every N steps (0 disables).
    :param keep_last: Number of most-recent rolling checkpoints to retain.
    :param val_examples: Held-out (from train, not the test set) examples for best-selection.
    :param precision: ``bf16`` (default) runs forwards under bf16 autocast on CUDA and enables
        TF32; ``fp32`` is bit-identical to the pre-flag driver. Applied to the training forward
        *and* to in-loop validation scoring so best-selection matches training. GPU-only — see
        :func:`autocast_ctx`.
    :param remote_dir: Optional URI (e.g. the platform's ``$EDULLM_CHECKPOINT_DIR``) to mirror
        every checkpoint to as it is written. ``save_dir`` stays the **local** staging directory;
        a URI cannot be used as one, because :class:`pathlib.Path` mangles it into a relative
        local path without erroring — see :func:`is_remote`. Local rolling pruning still applies;
        remote copies are not pruned, since with no ``--resume`` they exist for manual recovery.
    """
    from olmo_core.optim import WSD

    from .evaluate import overall_accuracy

    device = str(getattr(model, "device", "cpu"))
    configure_precision(precision, device)
    if save_dir is not None and is_remote(save_dir):
        raise ValueError(
            f"save_dir must be a LOCAL staging directory, got the URI {save_dir!r}. "
            "pathlib.Path silently rewrites 's3://b/k' to the relative local path 's3:/b/k', so "
            "this would write checkpoints next to the process and lose them. Pass a local "
            "save_dir and the URI as remote_dir instead."
        )
    save_dir = Path(save_dir) if save_dir is not None else None
    checkpointing = save_dir is not None and save_every > 0
    if checkpointing:
        assert save_dir is not None  # narrows the type for mypy
        save_dir.mkdir(parents=True, exist_ok=True)
    rolling: List[Path] = []
    best_acc = -1.0

    def _save_rolling(step_num: int) -> None:
        assert save_dir is not None
        path = save_dir / f"step{step_num}.pt"
        torch.save(model.state_dict(), path)
        publish_artifact(path, remote_dir)
        rolling.append(path)
        while len(rolling) > keep_last:  # rolling window: drop the oldest
            rolling.pop(0).unlink(missing_ok=True)

    def _maybe_update_best(step_num: int) -> None:
        nonlocal best_acc
        assert save_dir is not None
        if not val_examples:
            return
        was_training = model.training
        model.eval()
        with torch.no_grad(), autocast_ctx(precision, device):
            acc = overall_accuracy(model, val_examples, arm.arm_mode)
        if was_training:
            model.train()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_dir / "best.pt")
            (save_dir / "best.json").write_text(
                json.dumps({"step": step_num, "val_acc": acc}, indent=2)
            )
            publish_artifact(save_dir / "best.pt", remote_dir)
            publish_artifact(save_dir / "best.json", remote_dir)

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
        with autocast_ctx(precision, device):
            loss, metrics = arm_loss(
                model,
                batch["examples"],
                mode=arm.arm_mode,
                distill_weight=distill_weight,
                vocab_reg=arm.vocab_reg,
                vocab_reg_weight=arm.vocab_reg_weight,
                vocab_reg_entropy_floor=arm.vocab_reg_entropy_floor,
            )
        # backward runs OUTSIDE autocast; autograd replays each op in the dtype it ran in.
        loss.backward()
        # clip_grad_norm_ returns the PRE-clip total norm — log it: a rising grad norm is
        # the earliest warning that the latent path is diverging.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "lr": float(lr_t),
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    **metrics,
                }
            )
        if checkpointing and ((step + 1) % save_every == 0 or step == steps - 1):
            _save_rolling(step + 1)
            _maybe_update_best(step + 1)
    return history
