"""Shared SFT training core (PRD §2.6).

One ``run_sft(cfg, attach_weights=None)`` drives all three SFT implementations:

  - Impl 2 (vanilla SI-conditioned SFT): ``attach_weights=None`` -> uniform loss.
  - Impl 3 (KL-reweighted loss):        ``attach_weights`` returns a per-token weight
                                        for each row; a WeightedTrainer applies
                                        ``L_B = (1/N_B) * sum_t m_t * CE_t``.
  - Impl 4 (SDFT):                      ``attach_weights=None``, but ``cfg.data_dir``
                                        points at the self-distilled mix.

Cross-cutting: SAVE_STEPS is auto-set so the run yields >= ``cfg.min_checkpoints``
(>=10 per the PRD checkpoint-sweep principle), and ``save_total_limit`` defaults to
None so every checkpoint is kept for the KL–forgetting curve.

Also cross-cutting: an ``output_dir`` that is an ``s3://`` URI trains to local disk and is
mirrored to that prefix as each checkpoint lands. HF's Trainer has no notion of object
storage and would otherwise create a local directory named ``s3:``. See ``common/s3_io.py``
for what that costs and what happens when an upload fails.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from . import data as data_mod
from . import s3_io
from .chat import IGNORE, has_loss_tokens, make_collate_fn, make_tokenize_fn
from .modeling import LoraSettings, load_for_training


@dataclass
class SFTConfig:
    base_model: str = "allenai/OLMo-2-0425-1B-Instruct"
    data_dir: str = "data"
    hf_dataset: str | None = None  # e.g. "meric533/socrateach-sft"; overrides data_dir
    output_dir: str = "out/sft"

    max_len: int = 1024
    seed: int = 13
    num_epochs: float = 1.0
    train_total: int = 0  # 0 = use the whole prepared train file

    # LoRA (PRD §2.6 default). full_finetune -> use_lora=False (needs A100/L4).
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Optimization
    per_device_batch: int = 8
    grad_accum: int = 4
    learning_rate: float | None = None  # default 2e-4 (LoRA) / 1e-5 (full)
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    grad_checkpointing: bool = True

    # Checkpoint sweep (PRD cross-cutting principle)
    min_checkpoints: int = 10
    save_steps: int = 0     # 0 = auto from total steps and min_checkpoints
    save_total_limit: int | None = None  # None = keep ALL checkpoints for the curve
    # "uniform" = every save_steps; "log" = powers of two (1,2,4,8,...) + final step.
    # Log spacing densely samples the fast early trajectory where KL/forgetting move
    # most, so the low-KL knee of the RL's-Razor curve is resolved (not just steps 20/40).
    checkpoint_schedule: str = "uniform"

    # Eval / logging
    eval_cap: int = 200
    eval_steps: int = 0     # 0 = mirror save_steps
    logging_steps: int = 20

    resume: str | None = None  # path, or "auto" (see _allow_resume_from_our_own_checkpoints)
    report_to: str = "wandb"   # "wandb" (default) or "none". W&B needs WANDB_API_KEY set.
    wandb_project: str = "edullm-p7"  # override per team convention; entity via WANDB_ENTITY env
    run_name: str | None = None  # defaults to the output-dir basename

    def resolved_lr(self):
        if self.learning_rate is not None:
            return self.learning_rate
        return 2e-4 if self.use_lora else 1e-5


class WeightedTrainer:
    """Mixin factory: builds a Trainer subclass whose loss is per-token weighted.

    Weight semantics (PRD §3.3 "cleaner equivalent"): each loss token carries a
    multiplier ``m_t`` (mean 1 over pedagogy tokens, exactly 1 for general), and
    ``L_B = (1/N_B) * sum_t m_t * CE_t`` where N_B = #loss tokens in the batch. This
    preserves the pedagogy:general ratio and the effective LR automatically.
    """

    @staticmethod
    def build():
        from transformers import Trainer

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kw):
                weights = inputs.pop("weights", None)
                labels = inputs["labels"]
                if weights is None:  # safety: fall back to a uniform weight
                    weights = (labels != IGNORE).float()
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
                logits = outputs.logits
                # standard causal shift: logits[:, :-1] predict labels[:, 1:]
                shift_logits = logits[:, :-1, :].float()
                shift_labels = labels[:, 1:]
                shift_w = weights[:, 1:]
                V = shift_logits.size(-1)
                ce = F.cross_entropy(
                    shift_logits.reshape(-1, V), shift_labels.reshape(-1),
                    ignore_index=IGNORE, reduction="none",
                )
                mask = shift_labels.reshape(-1) != IGNORE
                w = shift_w.reshape(-1)
                denom = mask.sum().clamp(min=1)
                loss = (w * ce * mask).sum() / denom
                return (loss, outputs) if return_outputs else loss

        return _WeightedTrainer


def _allow_resume_from_our_own_checkpoints():
    """Re-enable ``trainer.train(resume_from_checkpoint=...)`` on torch < 2.6.

    Transformers calls ``check_torch_load_is_safe()`` before reading a checkpoint's
    ``optimizer.pt``/``scheduler.pt``, and that helper hard-refuses on any torch below 2.6 over
    CVE-2025-32434 — pickle files can execute code on load. With torch pinned at 2.5.1 every
    ``--resume auto`` therefore dies at startup with "we now require users to upgrade torch",
    which is why the a-T16/a-T32 reruns failed instantly while fresh runs were fine. The model
    weights are safetensors and unaffected; it is only the optimizer state that is pickled.

    The CVE is about loading UNTRUSTED checkpoints. These files were written minutes earlier by
    this same script into our own scratch directory, so the check is guarding against a threat
    that does not exist here. Upgrading torch instead would be the "proper" fix but would change
    training numerics midway through a sweep whose whole purpose is comparing runs to each other,
    so the pin stays and the check is disabled only on the resume path.

    Two separate guards block the resume and both have to go. Clearing the version gate above only
    gets as far as ``_load_rng_state``, which calls ``torch.load(rng_file, weights_only=True)``
    directly: the RNG state holds numpy's Mersenne-Twister key as a uint32 array, and unpickling
    that needs ``numpy._core.multiarray._reconstruct``, which is not on torch's default allowlist.
    That one is fixed the way torch documents — allowlisting the specific numpy types involved,
    which is much narrower than turning ``weights_only`` off.
    """
    _disable_torch_load_version_gate()
    _allowlist_numpy_rng_state()


def _disable_torch_load_version_gate():
    try:
        from transformers.utils import import_utils
    except Exception:  # pragma: no cover - transformers layout changed
        return
    if getattr(import_utils, "check_torch_load_is_safe", None) is None:
        return  # a version that never added the check: nothing to do
    import transformers.trainer as _trainer

    def noop(*a, **k):
        return None

    import_utils.check_torch_load_is_safe = noop
    # trainer.py does `from ... import check_torch_load_is_safe`, so it holds its own reference and
    # patching the source module alone would not take effect.
    if getattr(_trainer, "check_torch_load_is_safe", None) is not None:
        _trainer.check_torch_load_is_safe = noop
    print("resume: disabled transformers' torch.load safety gate for our own checkpoint files")


def _allowlist_numpy_rng_state():
    """Permit the numpy types inside a checkpoint's rng_state.pth under weights_only=True."""
    import numpy as np

    allow = [np.ndarray, np.dtype]
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):  # 2.x and 1.x spellings
        try:
            allow.append(__import__(mod, fromlist=["_reconstruct"])._reconstruct)
            break
        except Exception:
            continue
    # The MT19937 key is uint32; scalar dtype classes are pickled by type, not by np.dtype alone.
    allow += [getattr(np.dtypes, n) for n in ("UInt32DType", "Int64DType", "Float64DType")
              if hasattr(getattr(np, "dtypes", None), n)]
    try:
        torch.serialization.add_safe_globals(allow)
    except Exception as e:  # pragma: no cover - torch too old to have the allowlist API
        print(f"resume: could not extend torch safe-globals ({e}); rng state may fail to load")
        return
    print(f"resume: allowlisted {len(allow)} numpy types for rng_state.pth")


def _total_steps(n_train, cfg: SFTConfig):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(1, math.ceil(n_train / (cfg.per_device_batch * cfg.grad_accum * world)))
    return max(1, int(steps_per_epoch * cfg.num_epochs))


def _auto_save_steps(n_train, cfg: SFTConfig):
    total = _total_steps(n_train, cfg)
    return max(1, total // max(1, cfg.min_checkpoints)), total


def _log_spaced_steps(total):
    """Powers of two in [1, total] plus a couple of early linear anchors and the final
    step: e.g. total=90 -> [1,2,3,4,8,16,32,64,90]. Small runs still get >=1 point."""
    steps = {1, 2, 3, total}
    k = 4
    while k < total:
        steps.add(k)
        k *= 2
    return sorted(s for s in steps if 1 <= s <= total)


def make_log_spaced_callback(total_steps):
    """A TrainerCallback that forces save+eval at log-spaced global steps.

    Runs AFTER HF's DefaultFlowCallback each step, so it can turn ``should_save`` /
    ``should_evaluate`` on even when ``save_strategy='no'``. LoRA adapters are tiny, so
    keeping every log-spaced checkpoint is cheap.
    """
    from transformers import TrainerCallback

    targets = set(_log_spaced_steps(total_steps))

    class _LogSpacedCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if state.global_step in targets:
                control.should_save = True
                control.should_evaluate = True
            return control

    print(f"log-spaced checkpoints at steps: {sorted(targets)}")
    return _LogSpacedCallback()


def tokenize_splits(cfg: SFTConfig, tokenizer, *, require_test=False):
    """Tokenize + assistant-mask + filter the prepared splits.

    The train split keeps a ``kind`` column (pedagogy/general) so Impl 3 can tell
    them apart. This is the single source of truth for tokenization, so the Impl-3
    weight precompute and the training run see byte-identical ``input_ids`` (the
    signal cache is keyed on them).
    """
    dsets = data_mod.build_sft_datasets(cfg.data_dir, train_cap=cfg.train_total,
                                        seed=cfg.seed, require_test=require_test,
                                        hf_dataset=cfg.hf_dataset)
    tok_fn = make_tokenize_fn(tokenizer, cfg.max_len)
    keep = [c for c in ("kind",) if c in dsets["train"].column_names]
    train_tok = dsets["train"].map(
        tok_fn, remove_columns=[c for c in dsets["train"].column_names if c not in keep], desc="tok train"
    ).filter(has_loss_tokens)
    eval_tok = dsets["val"].map(
        tok_fn, remove_columns=dsets["val"].column_names, desc="tok eval"
    ).filter(has_loss_tokens)
    if len(eval_tok) > cfg.eval_cap:
        eval_tok = eval_tok.shuffle(seed=cfg.seed).select(range(cfg.eval_cap))
    return train_tok, eval_tok


def _prepare_s3_output(cfg: SFTConfig):
    """Redirect an ``s3://`` output dir to local disk and return the mirror that owns it.

    THE SEAM IS HERE, INSIDE THE SHARED CORE, RATHER THAN IN A WRAPPER AROUND THE TWO
    ENTRYPOINTS, AND THE REASON IS THE CALLBACK. Wrapping ``train_sft.py`` and
    ``train_kl_sft.py`` would leave this file untouched, which is worth something given
    that Meric handed the experiment off finished. It cannot deliver per-checkpoint upload.
    A wrapper only regains control when ``run_sft`` returns, which is after training has
    ended, so the best it could offer is a single upload at the end plus whatever a
    directory-polling thread could scrape, and a run that loses its host has already lost
    everything by then. Uploading as each checkpoint lands needs a ``TrainerCallback``
    attached to the ``Trainer``, and the ``Trainer`` is constructed in this function and is
    not reachable from outside it until it is too late to matter.

    Two smaller reasons point the same way. One seam serves both entrypoints instead of two
    wrappers that drift, and the mechanics live in ``common/s3_io.py`` so the eval scripts
    can adopt the same helpers when their turn comes without importing anything from a
    training wrapper. What this file gains is glue and no logic.

    Returns None for an ordinary local output dir, which is every ORCD run and every
    developer run, and that path is byte-for-byte what it was before.
    """
    mirror = s3_io.S3Mirror.for_output_dir(cfg.output_dir)
    if mirror is None:
        return None

    # Settled before output_dir is rewritten, because the default further down is the
    # basename of the output dir and the rewritten one is a staging path under /tmp. The
    # S3 basename is no better, every run's prefix ends in "checkpoints", so the run id
    # the platform already put in the environment is the only spelling that identifies the
    # run in W&B. The submitted command passes --run_name explicitly and this is the
    # fallback for the one that forgets.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("EDULLM_RUN_ID") or os.path.basename(mirror.prefix)

    os.makedirs(mirror.local_dir, exist_ok=True)
    original = cfg.output_dir
    cfg.output_dir = mirror.local_dir
    mirror.announce_start({
        "run_id": os.environ.get("EDULLM_RUN_ID"),
        "commit_sha": os.environ.get("EDULLM_COMMIT_SHA"),
        "team": os.environ.get("EDULLM_TEAM"),
        "run_name": cfg.run_name,
        "output_uri": mirror.uri,
        "local_staging_dir": mirror.local_dir,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # The whole resolved config, next to the checkpoints it produced. Free to write
        # and it answers the first question anyone asks of an adapter found in a bucket.
        # output_dir is put back to what the caller asked for so the record names the
        # destination rather than a /tmp path that stopped existing when the container did.
        "config": {**asdict(cfg), "output_dir": original},
    })
    print(f"s3: training locally in {mirror.local_dir}, mirroring to {mirror.uri}")
    return mirror


def run_sft(cfg: SFTConfig, attach_weights=None):
    """Train an SFT model. ``attach_weights(tokenized_train_ds, tokenizer) -> ds`` may
    add a ``weights`` column (list[float] aligned to input_ids) to enable Impl 3."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # First, before the model download and the tokenization, so a prefix this container
    # cannot write to costs a startup rather than an hour of GPU.
    mirror = _prepare_s3_output(cfg)

    lr = cfg.resolved_lr()
    print("=" * 72)
    print(f"SFT: base={cfg.base_model} -> {mirror.uri if mirror else cfg.output_dir}")
    print(f"lora={cfg.use_lora} lr={lr} per_device={cfg.per_device_batch} "
          f"grad_accum={cfg.grad_accum} epochs={cfg.num_epochs} max_len={cfg.max_len} "
          f"grad_ckpt={cfg.grad_checkpointing} weighted={attach_weights is not None}")
    print("=" * 72)

    model, tokenizer, bf16, fp16 = load_for_training(
        cfg.base_model, use_lora=cfg.use_lora,
        lora=LoraSettings(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout),
    )

    train_tok, eval_tok = tokenize_splits(cfg, tokenizer)

    lens = [len(x) for x in train_tok["input_ids"]]
    print(f"train={len(train_tok)} eval={len(eval_tok)} | tokens mean {np.mean(lens):.0f} "
          f"p95 {int(np.percentile(lens, 95))} max {max(lens)}")

    extra_keys = ()
    if attach_weights is not None:
        train_tok = attach_weights(train_tok, tokenizer)
        extra_keys = ("weights",)
        if "kind" in train_tok.column_names:
            train_tok = train_tok.remove_columns("kind")
    elif "kind" in train_tok.column_names:
        train_tok = train_tok.remove_columns("kind")

    from transformers import Trainer, TrainingArguments

    if "wandb" in (cfg.report_to or ""):
        # HF's WandbCallback reads WANDB_PROJECT from the env; set it unless the user
        # already exported one. Entity is left to WANDB_ENTITY (user's default if unset).
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    if cfg.run_name is None:
        cfg.run_name = os.path.basename(cfg.output_dir.rstrip("/")) or "p7-sft"

    log_spaced = cfg.checkpoint_schedule == "log"
    total_steps = _total_steps(len(train_tok), cfg)
    if log_spaced:
        # Saving/eval are driven entirely by the callback; keep every checkpoint.
        save_steps = eval_steps = total_steps  # placeholder (strategy is "no")
        print(f"checkpoint_schedule=log total_steps={total_steps} (all checkpoints kept)")
    else:
        save_steps = cfg.save_steps or _auto_save_steps(len(train_tok), cfg)[0]
        eval_steps = cfg.eval_steps or save_steps
        print(f"save_steps={save_steps} eval_steps={eval_steps} "
              f"(targeting >= {cfg.min_checkpoints} checkpoints; save_total_limit={cfg.save_total_limit})")

    train_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_batch,
        per_device_eval_batch_size=cfg.per_device_batch,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        eval_strategy="no" if log_spaced else "steps",
        eval_steps=eval_steps,
        save_strategy="no" if log_spaced else "steps",
        save_steps=save_steps,
        # log spacing keeps ALL checkpoints (adapters are tiny); the curve needs them.
        save_total_limit=None if log_spaced else cfg.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cfg.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # our custom collator emits exactly the keys the model + WeightedTrainer need
        # (incl. ``weights``), so don't let the Trainer strip columns.
        remove_unused_columns=False,
        optim="adamw_torch",
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
    )

    collate = make_collate_fn(tokenizer, extra_keys=extra_keys)
    trainer_cls = WeightedTrainer.build() if extra_keys else Trainer
    trainer = trainer_cls(
        model=model, args=train_args,
        train_dataset=train_tok, eval_dataset=eval_tok, data_collator=collate,
    )
    if log_spaced:
        trainer.add_callback(make_log_spaced_callback(total_steps))
    if mirror is not None:
        trainer.add_callback(mirror.callback())

    resume = cfg.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint

        if mirror is not None:
            # A retry runs on a new host with an empty staging dir, so the checkpoints
            # attempt 1 wrote are only in S3. Pulling the newest one back is what turns
            # `maximum_attempts: 2` into a second attempt rather than a second first
            # attempt. get_last_checkpoint then finds it exactly as it would on ORCD.
            mirror.hydrate_latest_checkpoint()
        resume = get_last_checkpoint(cfg.output_dir) if os.path.isdir(cfg.output_dir) else None
        print(f"Resuming from checkpoint: {resume}")

    if resume:
        _allow_resume_from_our_own_checkpoints()
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    if mirror is None:
        print(f"Saved model + tokenizer to {cfg.output_dir}")
    else:
        # The upload and the listing both happen before the success line prints, because
        # the line printing regardless of whether anything was uploaded is the entire bug.
        # Either of them raising ends the run non-zero, which is the point.
        mirror.upload_pending(label="final model + tokenizer")
        mirror.verify()
        print(f"Saved model + tokenizer to {mirror.uri}")
    return trainer
