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
import sys
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Union,
)

import torch

from .arms import Arm
from .data.dataset import codi_collate
from .loss import arm_loss
from .moe import (
    collect_router_metrics,
    count_forwards,
    finish_step,
    is_moe_model,
    normalized_aux_losses,
    reset_router_state,
)

__all__ = [
    "PRECISIONS",
    "build_model_from_config",
    "read_model_config",
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


def read_model_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Read the model config OLMo-core saves beside a checkpoint, if there is one.

    :class:`~olmo_core.train.callbacks.ConfigSaverCallback` writes the whole experiment config to
    ``config.json`` in the checkpoint directory, and its ``model`` key is a serialized
    :class:`~olmo_core.nn.transformer.TransformerConfig`. That is the only description of the
    architecture that is guaranteed to match the weights.

    Probed at the checkpoint directory and one level up, because a caller may legitimately point
    at the step directory, at the ``model_and_optim/`` inside it, or straight at a ``.pt`` file
    (whose parent directory is where the config would be).

    :param checkpoint_path: A checkpoint directory or URI (a plain ``.pt`` file has no config).
    :returns: The ``model`` sub-config as a dict, or ``None`` if no ``config.json`` was found.
    """
    import json

    from olmo_core.io import file_exists

    root = str(checkpoint_path).rstrip("/")
    # A caller may point at a step directory, at the `model_and_optim/` inside it, or straight at
    # a `.pt` file. Build the candidate list by walking up rather than appending "..", which does
    # not resolve past a non-directory component and silently found nothing for the `.pt` case.
    if Path(root).is_file():
        root = str(Path(root).parent)
    parent = root.rsplit("/", 1)[0] if "/" in root else root
    for candidate in (f"{root}/config.json", f"{parent}/config.json"):
        if not file_exists(candidate):
            continue
        from cached_path import cached_path

        try:
            config = json.loads(Path(str(cached_path(candidate))).read_text())
        except BaseException:  # noqa: BLE001 -- an unreadable config is a "not found", not a crash
            continue
        model_config = config.get("model") if isinstance(config, dict) else None
        if isinstance(model_config, dict):
            return model_config
    return None


def build_model_from_config(
    model_config: Dict[str, Any], *, device: str = "cpu", attn_backend: Optional[str] = None
):
    """
    Build a model from a serialized ``TransformerConfig`` — the way to load an *arbitrary*
    pretrained checkpoint.

    Prefer this over :func:`build_model` for post-training a real model. ``build_model`` can only
    name a registered ``TransformerConfig`` factory and hardcodes this project's vocab size, so it
    can only reproduce architectures that happen to have a factory. Loading is ``strict=True``, so
    the built architecture has to match the weights exactly; reading the checkpoint's own config is
    the only way to guarantee that. Verified to round-trip exactly for MoE configs too, which
    rebuild as :class:`~olmo_core.nn.transformer.MoETransformer` with their expert parameters.

    No ``init_seed``: every parameter is about to be overwritten by a strict load, so a
    deterministic init buys nothing here. (It mattered for the experiment, where arms had to share
    a random start.)

    :param model_config: From :func:`read_model_config`.
    :param device: Where to build.
    :param attn_backend: Optional override, for an image whose kernels do not match what the
        checkpoint's config asks for (see :func:`build_model`).

    :returns: The built, randomly-initialized model, ready for :func:`load_checkpoint`.
    """
    from olmo_core.nn.transformer import TransformerConfig

    config = TransformerConfig.from_dict(model_config)
    if attn_backend is not None:
        # Only where the config actually carries a backend, so this cannot invent a field.
        block = getattr(config, "block", None)
        attention = getattr(block, "attention", None)
        if attention is not None and hasattr(attention, "backend"):
            attention.backend = attn_backend
    return config.build(init_device=device)


def load_checkpoint(model, path: str, *, strict: bool = True) -> None:
    """
    Load weights into ``model`` from either a plain ``.pt`` state_dict (produced by
    ``train_codi.py``) or an OLMo-core checkpoint directory/URL — local **or remote**
    (e.g. ``s3://…``, loaded via ``load_model_and_optim_state`` with ``pre_download``).

    Used to fork every arm from the shared base checkpoint (the "best model"), and to read a
    finished arm's own ``model.pt``/``stepN.pt`` back for evaluation.

    Three layouts are handled, in this order:

    1. A **local** file — loaded straight through ``torch.load``.
    2. A **remote single file** ending ``.pt`` — staged locally by ``cached_path`` first. This
       case needs its own branch because :class:`pathlib.Path` mangles a URI, so
       ``Path("s3://b/k/model.pt").is_file()`` is ``False`` for a file that certainly exists and
       the plain-state_dict branch above can never fire for one. Without it, every ``stepN.pt``
       an arm mirrored to S3 was unreadable and reported as a missing *distributed* checkpoint,
       which is a misleading way to say "wrong kind of path".
    3. A checkpoint **directory**, probed for both layouts OLMo-core writes, because
       ``load_model_and_optim_state`` only reads ``<dir>/.metadata`` while ``Checkpointer`` saves
       the sharded state under ``<dir>/model_and_optim/`` — see
       :meth:`~olmo_core.train.checkpoint.Checkpointer.dir_is_checkpoint`, which checks both.
       Passing a step directory saved the second way otherwise fails with "is not a distributed
       checkpoint folder", which is what killed run ``run_019fde62`` twelve seconds in.

    :raises FileNotFoundError: If neither directory layout is present, naming both prefixes
        probed. Note that S3 reads through ``cached_path`` surface a denied read as a missing
        file, so this can equally mean "no permission" — check the grant before assuming the
        path is wrong.
    """
    if Path(str(path)).is_file():
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=strict)
        return

    if is_remote(path) and str(path).endswith(".pt"):
        from cached_path import cached_path

        local = cached_path(str(path), quiet=True)
        model.load_state_dict(torch.load(local, map_location="cpu"), strict=strict)
        return

    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.io import file_exists

    root = str(path).rstrip("/")
    candidates = [root, f"{root}/model_and_optim"]
    for candidate in candidates:
        if file_exists(f"{candidate}/.metadata"):
            load_model_and_optim_state(candidate, model, pre_download=True, strict=strict)
            return
    raise FileNotFoundError(
        "no distributed checkpoint found. Probed for '.metadata' under: "
        + ", ".join(candidates)
        + ". A denied S3 read is reported as a missing file, so confirm the training role's "
        "grant on the bucket before concluding the path is wrong."
    )


def build_model(
    rung: str, *, init_seed: int, device: str = "cpu", attn_backend: Optional[str] = None
):
    """
    Build a model at ``rung`` (a ``TransformerConfig`` factory name) with deterministic init.

    All arms must share the same ``init_seed`` so they start from identical weights — the
    shared "base checkpoint" the confound control requires.

    :param attn_backend: Override the attention backend (``"torch"``, ``"flash_2"``, …). The
        ``olmo3_*`` factories hardcode ``flash_2``, which raises at construction on any image
        without the ``flash-attn`` package — including the eduLLM research image, where it kills
        the run 11 seconds in. ``"torch"`` (SDPA) computes the *same* attention with a different
        kernel, and the sliding-window pattern is a no-op at our ~300-token sequences (window
        4096), so this is an implementation choice rather than an architecture change. ``None``
        keeps the factory's own default.
    """
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.utils import seed_all

    from .tokens import TOKENIZER_CONFIG

    seed_all(init_seed)
    kwargs: Dict[str, Any] = {"vocab_size": TOKENIZER_CONFIG.padded_vocab_size()}
    if attn_backend is not None:
        kwargs["attn_backend"] = attn_backend
    config = getattr(TransformerConfig, rung)(**kwargs)
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
    # `str` is admitted because rejecting an `s3://` URI here is this function's job: the
    # guard below is what stands between a remote URI and Path() silently mangling it into a
    # relative local path. Typing this `Path`-only would make the caller convert first, which
    # is exactly the conversion that loses the checkpoints.
    save_dir: Optional[Union[str, Path]] = None,
    save_every: int = 0,
    keep_last: int = 2,
    val_examples: Optional[List[dict]] = None,
    precision: str = "bf16",
    remote_dir: Optional[str] = None,
    on_log: Optional[Callable[[dict], None]] = None,
    max_seconds: Optional[float] = None,
    micro_batch_size: Optional[int] = None,
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

    :param on_log: Called with each ``train_history`` entry as it is recorded — the streaming
        hook :class:`~olmo_core.latentcot.tracking.ArmTracker` is wired into. Kept a plain
        callable so this loop stays free of any metrics dependency and unit-testable. An
        exception from it is caught and reported rather than allowed to end the run, since a
        metrics sidecar must never cost a day of GPU time; pass a callable that swallows its own
        errors (``ArmTracker`` does).
    :param save_dir: Directory for rolling/best checkpoints; ``None`` disables checkpointing.
    :param save_every: Save a rolling checkpoint every N steps (0 disables).
    :param keep_last: Number of most-recent rolling checkpoints to retain.
    :param micro_batch_size: Split each batch into slices of this many examples and backward each
        slice, accumulating gradients before one optimizer step. ``None`` keeps the whole batch in
        one backward, which is the historical behaviour.

        THE GRADIENT IS UNCHANGED AND THE MEMORY IS NOT. Each slice's loss is scaled by
        ``len(slice) / len(batch)``, so the accumulated gradient equals what a single full-batch
        backward would produce -- effective batch size, LR schedule and every confound control are
        untouched. What changes is peak memory: ``arm_loss`` sums a per-example loss over the batch
        and returns one tensor, so without slicing nothing is freed until the end and every
        example's teacher forward *and* K-step student chain are alive simultaneously.

        On the CODI arms that is the difference between running and not. One CODI example is about
        2,724 token-forwards against A0's 250, so at batch 8 the activations come to roughly 45 GB
        and a 40 GB A100 dies about two minutes in -- which is how A2/A3/A4 were lost on
        ``run_019ff806``. Batch size is the wrong lever for it: peak depends on the *longest*
        examples drawn, so batch 16 survived nearly three hours on shorter data before an unlucky
        batch killed it, and batch 8 on longer data died immediately. ``micro_batch_size=1`` makes
        peak depend on one example (~20 GB) whatever the batch is.
    :param max_seconds: Wall-clock budget for the loop. On reaching it the loop stops **cleanly**
        after the current step, saves, and returns, so the caller still writes ``model.pt`` and
        ``metrics.json`` and anything downstream in the same job still runs.

        THIS EXISTS BECAUSE BEING KILLED AT THE RUNTIME WALL LOSES MORE THAN THE REMAINING STEPS.
        ``metrics.json`` is written last, after training, so a run killed mid-loop reports nothing
        at all -- and any evaluation sharing the job never starts. Arms differ enormously in cost
        per step (A0/A1 do one forward per example; A2-A4 do ``K + 2``), so on a fixed budget the
        CODI arms are the ones that run out, which is precisely the half of the experiment worth
        having. Passing a budget a little under the platform's bound converts "killed with nothing
        reported" into "stopped early, saved, evaluated, and said which step it reached".

        A budget makes arms end at **different step counts**, which is a confound on optimization
        budget if the arms are then compared as they stand. That is what ``save_every`` is for:
        with a dense ladder mirrored to S3 and remote copies unpruned, the matched-budget
        comparison is recoverable afterwards by evaluating every arm at the largest step *all*
        of them reached -- see :func:`olmo_core.latentcot.inventory.select_common_step`.
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
    save_path = Path(save_dir) if save_dir is not None else None
    checkpointing = save_path is not None and save_every > 0
    if checkpointing:
        assert save_path is not None  # narrows the type for mypy
        save_path.mkdir(parents=True, exist_ok=True)
    rolling: List[Path] = []
    best_acc = -1.0

    def _save_rolling(step_num: int) -> None:
        assert save_path is not None
        path = save_path / f"step{step_num}.pt"
        torch.save(model.state_dict(), path)
        publish_artifact(path, remote_dir)
        rolling.append(path)
        while len(rolling) > keep_last:  # rolling window: drop the oldest
            rolling.pop(0).unlink(missing_ok=True)

    def _maybe_update_best(step_num: int) -> None:
        nonlocal best_acc
        assert save_path is not None
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
            torch.save(model.state_dict(), save_path / "best.pt")
            (save_path / "best.json").write_text(
                json.dumps({"step": step_num, "val_acc": acc}, indent=2)
            )
            publish_artifact(save_path / "best.pt", remote_dir)
            publish_artifact(save_path / "best.json", remote_dir)

    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    # min(...) keeps warmup < horizon on the short smoke runs; decay_fraction matches the WSD default.
    scheduler = WSD(warmup=max(1, min(warmup_steps, steps - 1)), decay_fraction=0.1)
    history: List[dict] = []
    # On an MoE base the routers' auxiliary losses are per-forward, and the arms do very
    # different numbers of forwards (1 for A0/A1, K+2 for A2-A4), so without correction A2-A4
    # feel ~K times the balancing pressure A0/A1 do -- an arm-dependent confound on exactly the
    # comparison gate A is defined on. See olmo_core.latentcot.moe. All a no-op when dense.
    moe = is_moe_model(model)
    for step, batch in enumerate(iter_batches(dataset, batch_size, steps, seed)):
        lr_t = scheduler.get_lr(lr, step, steps)
        for group in opt.param_groups:
            group["lr"] = lr_t
        opt.zero_grad(set_to_none=True)
        if moe:
            # Read back per STEP, so the logged expert-balance numbers describe this step
            # rather than the whole run to date.
            reset_router_state(model)
        # GRADIENT ACCUMULATION, AND ON THE CODI ARMS IT IS THE DIFFERENCE BETWEEN RUNNING AND
        # NOT. `arm_loss` sums a per-example loss over the whole batch and returns one tensor, so
        # nothing is freed until the single backward: every example's teacher forward AND its
        # K-step student chain are alive at once, and peak memory scales with batch_size. Measured
        # on the repaired data, one CODI example is ~2,724 token-forwards against A0's ~250, so at
        # batch 8 that is ~45 GB of activations and a 40 GB A100 dies about two minutes in --
        # which is exactly how A2/A3/A4 were lost on run_019ff806, having also died at batch 16.
        #
        # Splitting the batch and backward-ing each slice makes peak memory depend on the SLICE
        # rather than the batch, while the gradient the optimizer sees is identical: each slice is
        # scaled by len(slice)/n so the accumulated gradient equals the one full-batch backward
        # would have produced. So this buys the memory without touching the effective batch size,
        # the LR schedule, or anything else the gates rest on. It costs nothing but a Python loop.
        examples = batch["examples"]
        n = len(examples)
        micro = micro_batch_size or n
        metrics: Dict[str, float] = {}
        loss_value = 0.0
        for start in range(0, n, micro):
            slice_ = examples[start : start + micro]
            weight = len(slice_) / n
            forwards = count_forwards(slice_, mode=arm.arm_mode) if moe else 1
            with normalized_aux_losses(model, forwards), autocast_ctx(precision, device):
                slice_loss, slice_metrics = arm_loss(
                    model,
                    slice_,
                    mode=arm.arm_mode,
                    distill_weight=distill_weight,
                    vocab_reg=arm.vocab_reg,
                    vocab_reg_weight=arm.vocab_reg_weight,
                    vocab_reg_entropy_floor=arm.vocab_reg_entropy_floor,
                )
            # backward runs OUTSIDE autocast; autograd replays each op in the dtype it ran in.
            (slice_loss * weight).backward()
            loss_value += float(slice_loss.detach()) * weight
            for key, value in slice_metrics.items():
                metrics[key] = metrics.get(key, 0.0) + value * weight
        loss = torch.tensor(loss_value, device=device)
        # clip_grad_norm_ returns the PRE-clip total norm — log it: a rising grad norm is
        # the earliest warning that the latent path is diverging.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        if moe:
            # The bias_gamma score-bias update (aux-loss-free load balancing). Nothing else in
            # this loop calls it, so without this that mechanism silently does nothing.
            finish_step(model)
        if step % log_every == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "lr": float(lr_t),
                    "loss": float(loss.detach()),
                    "grad_norm": float(grad_norm),
                    # Cost telemetry, so a short run measures what the sizing estimate guessed:
                    # seconds since the loop began (differences give s/step) and the true peak
                    # allocation. Both are what a compute request should quote instead of a model.
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "peak_mem_gb": (
                        round(torch.cuda.max_memory_allocated() / 1e9, 3)
                        if device.startswith("cuda")
                        else 0.0
                    ),
                    **metrics,
                    # Expert-balance telemetry on an MoE base ({} when dense). A fine-tune can
                    # quietly collapse the routing, and these are the series that show it.
                    **(collect_router_metrics(model) if moe else {}),
                }
            )
            # Print it too, flushed. `train_history` only reaches disk in metrics.json at the
            # very end, so without this a 13-hour run is silent: no way to see progress, and the
            # drift tripwires (thought_rms, grad_norm) that exist to catch a diverging latent
            # path early are unreadable until it is too late to act. Printing makes the
            # platform's own "last fifty lines the container printed" a live monitor.
            entry = history[-1]
            print(
                " ".join(
                    (
                        f"{key}={entry[key]:.4g}"
                        if isinstance(entry[key], float)
                        else f"{key}={entry[key]}"
                    )
                    for key in entry
                ),
                flush=True,
            )
            if on_log is not None:
                try:
                    on_log(entry)
                except BaseException as exc:  # noqa: BLE001 -- see the :param on_log: note
                    print(
                        f"[on_log] sink raised, continuing: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
        out_of_time = max_seconds is not None and (time.perf_counter() - started) >= max_seconds
        if checkpointing and ((step + 1) % save_every == 0 or step == steps - 1 or out_of_time):
            _save_rolling(step + 1)
            _maybe_update_best(step + 1)
        if out_of_time:
            # Announced on stdout, not just returned: the step this stopped at is what a later
            # matched-budget evaluation needs, and stdout is the one channel that survives a
            # container whose local files do not.
            print(
                f"[{arm.name}] wall-clock budget of {max_seconds:.0f}s reached at step "
                f"{step + 1}/{steps}; stopping cleanly so model.pt and metrics.json are written.",
                flush=True,
            )
            break
    return history
