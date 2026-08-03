#!/usr/bin/env python
"""Train one (variant, temperature) arm: Impl 5's recipe with Impl 3's per-token weighting.

Like ``train_sft_impl5.py`` imports ``ORCD-SFT/train_sft.py``, this imports *that* rather than
editing or copying it, so ``make_tokenize_fn`` and ``load_model_and_tokenizer`` are the same
code objects Impl 2, Impl 4 and Impl 5 ran. The chain is the guarantee: if the masking or the
LoRA config differed from D4's, the contrast this run exists to make would be gone.

What differs from D4, and it is the only thing:

    every loss-bearing PEDAGOGY token's cross-entropy is multiplied by m_t
    (IMPL3_HANDOFF §4.1); replay tokens keep 1.0

Held fixed at D4's, deliberately and not as an oversight:

* **``per_device_batch 8 × grad_accum 4``.** Not a tuning knob — see ``klw/config_klw.py``.
  Regrouping micro-batches changes each example's contribution when a group's token counts are
  uneven, which voids the loss-normalisation result Impl 5 inherited from A1.
* the training file itself — read straight out of ``impl5_ssd/runs/D4/``, never copied,
* ``SequentialSampler`` + ``dataloader_drop_last``, the 24/8 block layout,
* the 22-point checkpoint grid, seed 13, 923 steps, lr 2e-4, cosine, 1 epoch,
* ``gradient_checkpointing=True``. James ran with it *off* ("~30% faster") on an H200, and it
  is tempting here for the same reason, but D4 ran with it on. It is numerically a no-op only
  if activation recompute reproduces the dropout masks exactly, and LoRA dropout is 0.05, so
  flipping it risks a second difference for a wall-clock gain that arm-level parallelism
  already provides. Left on.

One departure that is required rather than chosen: ``remove_unused_columns=False``, without
which HF strips the ``weights`` column before the collator and the arm trains unweighted with
no error. See ``klw/trainer_klw.py``.

Usage:
    python train_sft_klw.py --arm bT1
    python train_sft_klw.py --arm aT8 --loss_denom global
    python train_sft_klw.py --arm bT451 --poc
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from klw import weighting                                              # noqa: E402
from klw._impl5 import chat4, manifest, mixing                         # noqa: E402
from klw.config_klw import (                                           # noqa: E402
    ARM_CHOICES,
    BASE_MODEL,
    DATA_ARM,
    GRAD_ACCUM,
    LEARNING_RATE,
    MAX_LEN,
    NUM_EPOCHS,
    PER_DEVICE_BATCH,
    REFERENCE_ADAPTER_ARM,
    REFERENCE_ADAPTER_STEP,
    SEED,
    WARMUP_RATIO,
    checkpoint_grid,
    resolve_arm,
)
from klw.config_klw import LORA_ALPHA, LORA_DROPOUT, LORA_R           # noqa: E402
from klw.paths_klw import (                                            # noqa: E402
    DATA_DIR,
    ensure_dir,
    reference_adapter,
    run_dir,
    signal_cache,
    train_file,
)
from klw.trainer_klw import (                                          # noqa: E402
    LOSS_DENOM_CHOICES,
    WEIGHT_COLUMN,
    make_collator,
    weighted_trainer_cls,
)

# impl4.trainer, reachable because importing klw._impl5 installs the sys.path bridges.
from impl4.trainer import checkpoint_grid_callback, sequential_trainer_cls  # noqa: E402

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=ARM_CHOICES)
    p.add_argument("--runs_root", default=None, help="Where THIS arm's output goes.")
    p.add_argument("--impl5_runs_root", default=None, help="Where D4's training file lives.")
    p.add_argument("--data_arm", default=DATA_ARM)
    p.add_argument("--train_file", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--signal_cache", default=None, help="Explicit .npz; else resolved by key.")
    p.add_argument("--data_dir", default=None)

    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)

    p.add_argument("--use_lora", action="store_true", default=True)
    p.add_argument("--full_finetune", dest="use_lora", action="store_false")
    p.add_argument("--lora_r", type=int, default=LORA_R)
    p.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)

    p.add_argument("--per_device_batch", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--no_grad_checkpointing", dest="grad_checkpointing",
                   action="store_false", default=True)

    p.add_argument("--loss_denom", default="auto", choices=LOSS_DENOM_CHOICES,
                   help="Set explicitly from acceptance check W1; 'auto' guesses.")
    p.add_argument("--allow_batch_change", action="store_true",
                   help="Permit per_device_batch x grad_accum != D4's. Invalidates the "
                        "contrast; refuses without this flag.")

    p.add_argument("--eval_cap", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=300, help="Resume only.")
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=20)

    p.add_argument("--poc", action="store_true")
    p.add_argument("--resume", default=None, help="Checkpoint path, or 'auto'.")
    return p.parse_args()


def resolve_signal_cache(args, arm, tf: Path) -> Path:
    """Find the cache for this arm's variant.

    The key comes from ``weighting.signal_key`` — the same single definition the precompute
    calls. Assembling it independently here is what produced a key mismatch that presented as
    "missing signal cache" for a cache that had just been written.
    """
    if args.signal_cache:
        return Path(args.signal_cache)
    ref = reference_adapter(REFERENCE_ADAPTER_ARM, REFERENCE_ADAPTER_STEP, args.impl5_runs_root)
    key = weighting.signal_key(arm.variant, tf, args.base_model, ref, args.max_len)
    return signal_cache(arm.variant, key, args.data_dir or str(DATA_DIR))


def build_datasets(args, block: int, tf: Path):
    """The ordered mix, unshuffled and uncapped — plus the val split, which stays gold."""
    from datasets import Dataset

    # Impl 2's own trainer module: make_tokenize_fn and load_model_and_tokenizer come from
    # here, so the masking and LoRA config are literally the objects D4 trained with.
    impl2 = chat4.impl2_trainer_module()

    val = Path(tf).parent / "socrateach_sft_val.jsonl"
    if not val.exists():
        raise FileNotFoundError(f"missing {val} (the held-out GOLD val split)")
    train_recs = impl2.load_records(str(tf))
    eval_recs = impl2.load_records(str(val))
    if len(train_recs) % block:
        raise ValueError(f"train file has {len(train_recs)} examples, not a whole number of "
                         f"{block}-example blocks — the block layout would be broken")
    kinds: dict[str, int] = {}
    for e in train_recs:
        kinds[e.get("kind", "?")] = kinds.get(e.get("kind", "?"), 0) + 1
    print(f"Loaded '{tf}': train={len(train_recs)} {kinds} | val={len(eval_recs)}")
    is_ped = [bool(mixing.is_pedagogy(e)) for e in train_recs]
    return (Dataset.from_list([{"messages": e["messages"]} for e in train_recs]),
            Dataset.from_list([{"messages": e["messages"]} for e in eval_recs]),
            kinds, is_ped, impl2)


def attach_weights(train_tok, cache: weighting.SignalCache, arm, is_ped: list[bool]):
    """Add the ``weights`` column, verifying row-by-row that the cache describes these tokens.

    The digest comparison is the load-bearing check. Precompute and training tokenise the same
    file with the same function, so the hashes must agree; if they do not, the multipliers
    would land on different tokens than they were computed for and every number downstream
    would be wrong in a way no metric would reveal.
    """
    if cache.n_rows != len(train_tok):
        raise SystemExit(f"signal cache covers {cache.n_rows} rows, training file has "
                         f"{len(train_tok)} — the cache was built for a different mix")
    row_m, diag = weighting.build_row_multipliers(cache, arm.temperature)

    cols, mismatched = [], 0
    for i, row in enumerate(train_tok):
        digest = weighting.row_digest(row["input_ids"], row["labels"])
        if digest != cache.row_hash[i]:
            mismatched += 1
            if mismatched <= 3:
                print(f"  row {i}: digest {int(digest)} != cached {int(cache.row_hash[i])}")
            continue
        cols.append(weighting.scatter_to_labels(row["labels"], row_m[i],
                                                general=not is_ped[i]))
    if mismatched:
        raise SystemExit(
            f"{mismatched} of {len(train_tok)} rows tokenise differently than the signal cache "
            f"expects. The cache is stale — re-run precompute_signal.py against this exact "
            f"training file."
        )
    if bool(cache.is_pedagogy.tolist() != is_ped):
        raise SystemExit("cache disagrees with the mix about which rows are pedagogy")

    n_ped_tok = sum(1 for i, c in enumerate(cols) if is_ped[i] for w in c if w != 0.0)
    print(f"weights: variant {arm.variant} T={arm.temperature:g} over {n_ped_tok:,} pedagogy "
          f"label tokens")
    d = diag["multiplier"]
    print(f"  mean {d['mean']:.6f} (must be 1)  min {d['min']:.3g}  p50 {d['p50']:.3g}  "
          f"p99 {d['p99']:.3g}  max {d['max']:.3g}")
    print(f"  ESS {d['ess']:.4f}  entropy {d['entropy_frac']:.4f}  "
          f"below 0.01: {d['frac_below_0.01']:.1%}  above 10: {d['frac_above_10']:.1%}")
    if abs(d["mean"] - 1.0) > 1e-6:
        raise SystemExit(f"mean multiplier is {d['mean']!r}, not 1 — the N_ped normalisation "
                         f"is broken and this would also change the effective learning rate")
    if d["ess"] < 0.02:
        print(f"  WARNING: ESS {d['ess']:.4f} — the gradient is concentrated on <2% of "
              f"pedagogy tokens. James's a-T0.5/a-T1 ended at NLL 2.743/2.138, ABOVE base's "
              f"1.416, in this regime. The run will complete and may learn nothing.")
    return train_tok.add_column(WEIGHT_COLUMN, cols), diag


def main():
    args = parse_args()
    arm = resolve_arm(args.arm)
    out = Path(args.output_dir) if args.output_dir else run_dir(arm.name, args.runs_root)
    args.output_dir = str(out)
    ensure_dir(out)

    block = args.per_device_batch * args.grad_accum
    if (args.per_device_batch, args.grad_accum) != (PER_DEVICE_BATCH, GRAD_ACCUM):
        msg = (f"per_device_batch x grad_accum is {args.per_device_batch}x{args.grad_accum}, "
               f"not D4's {PER_DEVICE_BATCH}x{GRAD_ACCUM}. Micro-batch regrouping changes each "
               f"example's contribution to the step whenever token counts are uneven, which "
               f"voids the loss-normalisation result Impl 5 inherited from A1 and puts a "
               f"second variable into the D4 contrast.")
        if not args.allow_batch_change:
            raise SystemExit(msg + " Pass --allow_batch_change to override.")
        print("WARNING: " + msg, flush=True)
    if block != PER_DEVICE_BATCH * GRAD_ACCUM:
        raise SystemExit(f"effective batch {block} != {PER_DEVICE_BATCH * GRAD_ACCUM}; the "
                         f"24/8 block layout requires it")

    tf = Path(args.train_file) if args.train_file else train_file(args.data_arm,
                                                                 args.impl5_runs_root)
    if not tf.exists():
        raise SystemExit(f"missing {tf}\nBuild it first:  python mix_arm5.py --arm "
                         f"{args.data_arm}")
    cache_path = resolve_signal_cache(args, arm, tf)
    if not cache_path.exists():
        raise SystemExit(f"missing signal cache {cache_path}\nBuild it first:\n"
                         f"    python precompute_signal.py --variants {arm.variant}")

    grid = checkpoint_grid(args.poc)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 74)
    print(f"Impl 3x5 arm {arm.name} | variant {arm.variant} T={arm.temperature:g} "
          f"({arm.role}) | James's {arm.impl3_name}")
    print(f"  {arm.note}")
    print(f"  data  = {tf}  (Impl 5 arm {args.data_arm}, shared by every arm)")
    print(f"  signal= {cache_path.name}")
    print(f"  out   = {out}")
    print(f"  lora={args.use_lora} lr={args.learning_rate} per_device={args.per_device_batch} "
          f"grad_accum={args.grad_accum} (block={block}) epochs={args.num_epochs} "
          f"max_len={args.max_len} loss_denom={args.loss_denom}")
    print(f"  checkpoint grid: {list(grid)}")
    print("=" * 74, flush=True)

    train_ds, eval_ds, kinds, is_ped, impl2 = build_datasets(args, block, tf)
    model, tokenizer, bf16, fp16 = impl2.load_model_and_tokenizer(args)

    tok_fn = impl2.make_tokenize_fn(tokenizer, args.max_len)
    train_tok = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="tok train")
    eval_tok = eval_ds.map(tok_fn, remove_columns=eval_ds.column_names, desc="tok eval")

    # Filtering would break block alignment, so assert instead (Impl 5's rule, unchanged).
    n_empty, lens = 0, []
    for row in train_tok:
        lens.append(len(row["input_ids"]))
        if all(t == IGNORE for t in row["labels"]):
            n_empty += 1
    if n_empty:
        raise ValueError(f"{n_empty} train examples have no unmasked labels; dropping them "
                         f"would break the block layout. Fix them in the mix.")
    eval_tok = eval_tok.filter(lambda x: any(t != IGNORE for t in x["labels"]))
    if len(eval_tok) > args.eval_cap:
        eval_tok = eval_tok.shuffle(seed=args.seed).select(range(args.eval_cap))

    cache = weighting.SignalCache.load(cache_path)
    train_tok, weight_diag = attach_weights(train_tok, cache, arm, is_ped)

    n_steps = len(train_tok) // block
    print(f"train={len(train_tok)} eval={len(eval_tok)} | tokens mean {np.mean(lens):.0f} "
          f"p95 {int(np.percentile(lens, 95))} max {max(lens)} | optimizer steps {n_steps}")

    from transformers import TrainingArguments

    extra_ta = {}
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        extra_ta["group_by_length"] = False

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16, fp16=fp16,
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", report_to="none", seed=args.seed,
        dataloader_drop_last=True,
        # REQUIRED: True strips the weights column before the collator and the arm silently
        # trains unweighted. See klw/trainer_klw.py.
        remove_unused_columns=False,
        **extra_ta,
    )

    cb, helper = checkpoint_grid_callback(args.output_dir, grid)
    trainer = weighted_trainer_cls(sequential_trainer_cls)(
        model=model, args=train_args, train_dataset=train_tok, eval_dataset=eval_tok,
        data_collator=make_collator(tokenizer, weighted=True), callbacks=[cb],
        loss_denom=args.loss_denom)

    from torch.utils.data import SequentialSampler
    assert isinstance(trainer._get_train_sampler(train_tok), SequentialSampler), \
        "train sampler is not SequentialSampler — the block layout would be shuffled away"

    resume = args.resume
    if resume == "auto":
        from transformers.trainer_utils import get_last_checkpoint
        resume = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
        print(f"Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)
    trainer.assert_weighting_ran()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    on_disk = {int(p.name.split("-")[1]) for p in out.glob("ckpt-*")
               if p.is_dir() and any(p.glob("adapter_model*"))}
    saved = sorted(on_disk | set(helper.saved))
    index = {
        "arm": arm.name, "checkpoint_grid": list(grid), "checkpoints_saved": saved,
        "checkpoints_written_this_run": sorted(set(helper.saved)),
        "steps": n_steps, "adapter_dirs": {str(s): f"ckpt-{s}" for s in saved},
    }
    (out / "checkpoint_index.json").write_text(json.dumps(index, indent=2) + "\n",
                                               encoding="utf-8")
    report = trainer.weighting_report()
    print(f"weighting: {report}")
    manifest.merge(out, "training", {
        **index,
        "implementation": "impl3x5 — IMPL3_HANDOFF §4.1 weighting on Impl 5's D4 targets",
        "variant": arm.variant, "temperature": arm.temperature, "role": arm.role,
        "impl3_equivalent": arm.impl3_name,
        "data_arm": args.data_arm, "train_file": str(tf),
        "train_file_digest": weighting.file_digest(tf),
        "signal_cache": str(cache_path), "signal_meta": cache.meta,
        "weighting": weight_diag, "weighting_runtime": report,
        "base_model": args.base_model,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "enabled": args.use_lora},
        "learning_rate": args.learning_rate, "warmup_ratio": args.warmup_ratio,
        "num_epochs": args.num_epochs, "per_device_batch": args.per_device_batch,
        "grad_accum": args.grad_accum, "effective_batch": block, "max_len": args.max_len,
        "seed": args.seed, "sampler": "SequentialSampler", "dataloader_drop_last": True,
        "group_by_length": False, "remove_unused_columns": False,
        "gradient_checkpointing": args.grad_checkpointing,
        "train_kinds": kinds, "n_train": len(train_tok),
        "recipe_note": ("Impl 5 D4's recipe and D4's exact training file, with Impl 3's "
                        "per-token multiplier on pedagogy tokens as the ONLY difference. "
                        "Baseline for every arm is D4 (impl5-D4), not impl4-A1 and not "
                        "James's gold-corpus runs."),
        "environment": manifest.environment(),
    })

    print(f"\nSaved adapter + tokenizer to {out}")
    print(f"Grid checkpoints on disk: {saved}")
    missing = [s for s in grid if s <= n_steps and s not in saved]
    if missing:
        print(f"WARNING: grid steps {missing} were NOT written.")


if __name__ == "__main__":
    main()
