#!/usr/bin/env python
"""Checks that have to pass before a GPU-hour is spent. Two stages, cheap one first.

The failure modes this guards against all share a property: they produce a **completed run with
a plausible loss curve and wrong numbers**. None of them raises on its own.

* the ``weights`` column gets stripped and the arm trains unweighted (W1, W5),
* the multipliers land one position off, or on the padding (W3),
* the cache was built from a different tokenisation than training uses (W4),
* the normalisation differs between the weighted and unweighted paths, so ``bT451`` cannot
  reproduce D4 and no arm is interpretable (W1),
* T is so low that almost no token carries gradient — James's a-T0.5 ended *above* base's
  NLL — and the run completes having learned nothing (W6).

``--stage fast`` needs no GPU and no real data; it runs on
``hf-internal-testing/tiny-random-Olmo2ForCausalLM`` in ~30 s and is the gate before the
precompute. ``--stage full`` needs the real training file and the finished signal cache.

**W1 is the one that decides a run parameter.** It reports which ``--loss_denom`` makes the
weighted path agree with the stock path, and ``run_klw.py`` passes that value to every arm
rather than letting ``auto`` guess. It exists because ``impl4_ssd/probe_loss_norm.py`` found
that the answer depends on the installed transformers *and* the PEFT wrapping, so it is
measured on the same wrapping training uses, not assumed.

    python acceptance_checks_klw.py --stage fast
    python acceptance_checks_klw.py --stage full --arm bT1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from klw import weighting                                              # noqa: E402
from klw._impl5 import chat4, mixing                                   # noqa: E402
from klw.config_klw import (                                           # noqa: E402
    ALL_ARMS,
    ARM_CHOICES,
    BASE_MODEL,
    DATA_ARM,
    GRAD_ACCUM,
    MAX_LEN,
    PER_DEVICE_BATCH,
    REFERENCE_ADAPTER_ARM,
    REFERENCE_ADAPTER_STEP,
    resolve_arm,
)
from klw.paths_klw import (                                            # noqa: E402
    DATA_DIR,
    ensure_dir,
    reference_adapter,
    signal_cache,
    train_file,
)
from klw.trainer_klw import (                                          # noqa: E402
    WEIGHT_COLUMN,
    make_collator,
    weighted_token_loss,
    weighted_trainer_cls,
)

TINY = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"
IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("fast", "full"), default="fast")
    p.add_argument("--arm", default=None, choices=ARM_CHOICES)
    p.add_argument("--model", default=TINY, help="Fast stage only.")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--train_file", default=None)
    p.add_argument("--impl5_runs_root", default=None)
    p.add_argument("--data_arm", default=DATA_ARM)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--out", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# W2 / W6 — the multiplier algebra. Pure numpy, no model.
# --------------------------------------------------------------------------- #
def check_w2_mean_one(rng) -> dict:
    """``mean(m) == 1`` exactly, for every T and every signal shape.

    This is what ``N_ped ·`` buys, and it is not cosmetic: without it a temperature sweep is
    also an effective-learning-rate sweep, and the pedagogy:general loss ratio moves with T.
    """
    worst = 0.0
    rows = []
    for name, s in (("normal", rng.normal(2.0, 1.0, 50_000)),
                    ("lognormal", rng.lognormal(0.0, 1.0, 50_000)),
                    ("heavy-tail", rng.standard_t(2.0, 50_000) + 5.0),
                    ("bimodal", np.concatenate([rng.normal(1, .2, 25_000),
                                                rng.normal(9, .5, 25_000)]))):
        median, scale = weighting.robust_stats(s)
        z = weighting.robust_z(s, median, scale)
        for T in (0.5, 1.0, 2.0, 8.0, 451.0):
            m = weighting.multipliers(z, T)
            err = abs(float(m.mean()) - 1.0)
            worst = max(worst, err)
            rows.append({"signal": name, "T": T, "mean_err": err,
                         "ess": weighting.describe(m)["ess"]})
    ok = worst < 1e-9
    print(f"  W2 mean-1 over 4 signal shapes x 5 temperatures: worst |mean-1| = {worst:.3e} "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"mean multiplier deviates by {worst:.3e}; N_ped normalisation "
                             f"is wrong")
    return {"ok": ok, "worst_mean_err": worst, "rows": rows}


def check_w6_temperature_limit(rng) -> dict:
    """T → ∞ recovers vanilla SFT, and low T concentrates. Both are load-bearing.

    The first is why ``bT451`` is a valid control at all. The second is the failure regime
    James hit: at T=0.5 his variant-a runs ended at NLL 2.743 against base's 1.416 because the
    gradient collapsed onto a handful of tokens. ``ess`` makes that visible before training.

    The limit is asserted as ``m − 1 = O(1/T)`` rather than as a fixed tolerance on
    ``max|m − 1|``. Expanding ``exp(−z/T)/mean(exp(−z/T))`` gives ``1 − (z − z̄)/T + O(1/T²)``,
    so ``T · max|m − 1|`` converges to a constant set by the signal's most extreme z — and on a
    heavy-tailed signal that constant is large enough (~85 here, from one draw in 100,000) that
    any fixed tolerance would be either vacuous or arbitrary. What actually matters for the
    control is that the *bulk* is flat and ESS is 1, both of which are checked directly.
    """
    s = rng.lognormal(0.0, 1.0, 100_000)
    median, scale = weighting.robust_stats(s)
    z = weighting.robust_z(s, median, scale)
    rows = []
    for T in (0.25, 0.5, 1.0, 2.0, 8.0, 32.0, 451.0, 10_000.0):
        m = weighting.multipliers(z, T)
        d = weighting.describe(m)
        dev = float(np.abs(m - 1.0).max())
        rows.append({"T": T, "ess": d["ess"], "max": d["max"], "min": d["min"],
                     "p1": float(np.percentile(m, 1)), "p99": d["p99"],
                     "mean_abs_dev": float(np.abs(m - 1.0).mean()),
                     "max_abs_dev": dev, "T_times_dev": T * dev})
    by_T = {r["T"]: r for r in rows}
    hi, lo = by_T[10_000.0], by_T[451.0]

    ess_mono = all(rows[i]["ess"] <= rows[i + 1]["ess"] + 1e-12 for i in range(len(rows) - 1))
    # O(1/T): the product T*max|m-1| stops moving once the expansion's linear term dominates.
    scaling = hi["T_times_dev"] / lo["T_times_dev"]
    # mean|m-1| rather than a quantile: for a stream whose per-token CE is roughly homogeneous
    # it bounds the relative perturbation of the total loss, which is the quantity that decides
    # whether the control can land on D4. A quantile would be an arbitrary cut.
    bulk_flat = lo["mean_abs_dev"] < 0.02
    ok = ess_mono and lo["ess"] > 0.999 and abs(scaling - 1.0) < 0.3 and bulk_flat
    print(f"  W6 T->inf: ESS(451)={lo['ess']:.5f}  mean|m-1| at T=451 = "
          f"{lo['mean_abs_dev']:.5f}  (p1..p99 [{lo['p1']:.4f}, {lo['p99']:.4f}])  "
          f"T*max|m-1| 451->1e4: {lo['T_times_dev']:.1f} -> {hi['T_times_dev']:.1f} "
          f"(O(1/T) ratio {scaling:.3f})  ESS monotone: {ess_mono}  "
          f"{'OK' if ok else 'FAIL'}")
    print("     ESS by T: " + "  ".join(f"{r['T']:g}:{r['ess']:.3f}" for r in rows))
    if not ok:
        raise AssertionError("the temperature limit does not behave; bT451 would not be a "
                             "valid control")
    return {"ok": ok, "rows": rows, "ess_at_451": lo["ess"], "o_1_over_T_ratio": scaling}


# --------------------------------------------------------------------------- #
# W3 — alignment through the collator's padding and the causal shift
# --------------------------------------------------------------------------- #
def check_w3_alignment(model_id: str) -> dict:
    """One label position gets weight 1, every other gets 0 — the loss must be *that* token's.

    Run over a batch of deliberately unequal lengths so the shorter rows are padded, and
    repeated for every unmasked position of every row. This is the check that would catch an
    off-by-one in the shift or a weight applied to padding, and neither is visible in a loss
    curve.
    """
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = chat4.load_tokenizer(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    collate = make_collator(tokenizer, weighted=True)

    # Three rows of different lengths, each with a masked prefix and an unmasked tail.
    rows = []
    for n_ctx, n_lab in ((5, 3), (11, 7), (3, 2)):
        ids = list(range(2, 2 + n_ctx + n_lab))
        labels = [IGNORE] * n_ctx + ids[n_ctx:]
        rows.append({"input_ids": ids, "labels": labels,
                     "attention_mask": [1] * len(ids), WEIGHT_COLUMN: [0.0] * len(ids)})

    batch = collate([dict(r) for r in rows])
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"]).logits

    # Reference: unreduced per-position CE over the same shift.
    shift_logits = logits[..., :-1, :].float()
    shift_labels = batch["labels"][..., 1:]
    ce = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1),
        ignore_index=IGNORE, reduction="none").reshape(shift_labels.shape)

    n_checked, worst = 0, 0.0
    for r in range(batch["labels"].shape[0]):
        for j in range(batch["labels"].shape[1]):
            if int(batch["labels"][r, j]) == IGNORE:
                continue
            w = torch.zeros_like(batch[WEIGHT_COLUMN])
            w[r, j] = 1.0
            total, n_unmasked = weighted_token_loss(logits, batch["labels"], w)
            want = ce[r, j - 1]                  # position j-1 predicts label j
            worst = max(worst, float((total - want).abs()))
            n_checked += 1

    # And the padding: a weight parked on a pad position must contribute nothing.
    w = torch.zeros_like(batch[WEIGHT_COLUMN])
    pad_hits = 0
    for r in range(batch["labels"].shape[0]):
        for j in range(len(rows[r]["input_ids"]), batch["labels"].shape[1]):
            w[r, j] = 1e6
            pad_hits += 1
    pad_total, _ = weighted_token_loss(logits, batch["labels"], w)

    ok = worst < 1e-4 and float(pad_total.abs()) < 1e-9
    print(f"  W3 alignment: {n_checked} label positions across 3 unequal-length rows, worst "
          f"error {worst:.2e}; {pad_hits} pad positions weighted 1e6 contribute "
          f"{float(pad_total):.2e} {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError("multipliers are not aligned to the labels they were computed for")
    return {"ok": ok, "positions_checked": n_checked, "worst_error": worst,
            "padding_contribution": float(pad_total)}


# --------------------------------------------------------------------------- #
# W1 — m = 1 reproduces the stock loss, and which denominator does it
# --------------------------------------------------------------------------- #
def _synthetic_rows(n_rows: int, tokenizer):
    """Rows with deliberately lopsided label counts, alternating short/long.

    Lopsidedness is the point, and it is borrowed from ``probe_loss_norm.py``: if every
    micro-batch carried the same number of label tokens, the ``global`` and ``microbatch``
    normalisations would agree numerically and W1 could not tell them apart. Short rows carry
    ~4 label tokens and long rows ~120.
    """
    vocab = int(getattr(tokenizer, "vocab_size", 1000))
    rows = []
    for i in range(n_rows):
        n_ctx, n_lab = (7, 4) if i % 2 == 0 else (9, 120)
        ids = [(3 + (i * 7 + k) % max(vocab - 5, 10)) for k in range(n_ctx + n_lab)]
        labels = [IGNORE] * n_ctx + ids[n_ctx:]
        rows.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
    return rows


def _run_tiny(model_id, rows, weighted: bool, loss_denom: str, per_device_batch: int,
              grad_accum: int, seed: int = 13):
    """Train the tiny model at lr=0 and return the logged losses.

    lr=0 so the weights cannot move between the weighted and unweighted runs — the two must be
    evaluating the same function for the comparison to mean anything. LoRA is attached with
    dropout 0 because ``model_accepts_loss_kwargs`` (the flag the normalisation branches on) is
    resolved from the *wrapped* model's forward signature, so the check has to carry the same
    PEFT wrapping the real run does.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, TrainingArguments

    from impl4.trainer import loss_capture_callback, sequential_trainer_cls

    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = chat4.load_tokenizer(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]))

    data = [dict(r) for r in rows]
    if weighted:
        for r in data:
            r[WEIGHT_COLUMN] = [1.0 if t != IGNORE else 0.0 for t in r["labels"]]
    ds = Dataset.from_list(data)

    with __import__("tempfile").TemporaryDirectory() as tmp:
        args = TrainingArguments(
            output_dir=tmp, num_train_epochs=1.0,
            per_device_train_batch_size=per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=0.0, lr_scheduler_type="constant", warmup_ratio=0.0,
            logging_steps=1, eval_strategy="no", save_strategy="no",
            report_to="none", seed=seed, dataloader_drop_last=True,
            remove_unused_columns=False, bf16=False, fp16=False,
        )
        sink: list = []
        cbs = [loss_capture_callback(sink)]
        collate = make_collator(tokenizer, weighted=weighted)
        if weighted:
            trainer = weighted_trainer_cls(sequential_trainer_cls)(
                model=model, args=args, train_dataset=ds, data_collator=collate,
                callbacks=cbs, loss_denom=loss_denom)
        else:
            trainer = sequential_trainer_cls()(
                model=model, args=args, train_dataset=ds, data_collator=collate,
                callbacks=cbs)
        trainer.train()
        report = trainer.weighting_report() if weighted else {}
        return sink, report


def check_w1_unit_weights(model_id: str, per_device_batch: int, grad_accum: int) -> dict:
    """``m ≡ 1`` must reproduce the stock loss. Reports the ``--loss_denom`` that does it.

    If no candidate matches, that is a hard stop rather than a warning: it means the weighted
    and unweighted paths normalise differently, so ``bT451`` could not reproduce D4 and no
    arm's comparison against D4 would be valid.
    """
    n_rows = per_device_batch * grad_accum * 2
    rows = _synthetic_rows(n_rows, chat4.load_tokenizer(model_id))
    stock, _ = _run_tiny(model_id, rows, weighted=False, loss_denom="auto",
                         per_device_batch=per_device_batch, grad_accum=grad_accum)
    if not stock:
        raise AssertionError("the stock run logged no losses; cannot compare")

    results, matches = {}, []
    for denom in ("auto", "global", "microbatch"):
        got, report = _run_tiny(model_id, rows, weighted=True, loss_denom=denom,
                                per_device_batch=per_device_batch, grad_accum=grad_accum)
        n = min(len(stock), len(got))
        diff = max(abs(a - b) for a, b in zip(stock[:n], got[:n])) if n else float("inf")
        results[denom] = {"max_abs_diff": diff, "n_logged": n,
                          "losses": got[:n], "report": report}
        flag = "MATCH" if diff < 1e-6 else ""
        print(f"  W1 loss_denom={denom:<10} max|weighted - stock| = {diff:.3e} "
              f"(used {report.get('loss_denom_used')}, "
              f"accepts_loss_kwargs={report.get('model_accepts_loss_kwargs')}) {flag}")
        if diff < 1e-6:
            matches.append(denom)

    explicit = [d for d in matches if d != "auto"]
    if not explicit:
        raise AssertionError(
            "no loss_denom reproduces the stock loss at m=1. The weighted path normalises "
            "differently from the unweighted one, so bT451 cannot reproduce D4 and no arm is "
            f"interpretable. Stock losses: {stock[:4]}; results: "
            + json.dumps({k: v['max_abs_diff'] for k, v in results.items()})
        )
    chosen = explicit[0]
    auto_ok = "auto" in matches
    print(f"  W1 -> use --loss_denom {chosen}"
          + ("  (auto also agrees)" if auto_ok else
             "  (auto DISAGREES — pass it explicitly)"))
    return {"ok": True, "stock_losses": stock, "matches": matches, "chosen": chosen,
            "auto_agrees": auto_ok,
            "per_denom": {k: v["max_abs_diff"] for k, v in results.items()}}


# --------------------------------------------------------------------------- #
# W4 / W5 / W7 — the real cache against the real training file
# --------------------------------------------------------------------------- #
def check_full(args, arm) -> dict:
    """The real mix and the real cache: digests, coverage, stream split, and this arm's ESS."""
    import torch  # noqa: F401  (fails early here rather than deep in transformers)

    from klw._impl5 import manifest

    tf = Path(args.train_file) if args.train_file else train_file(args.data_arm,
                                                                 args.impl5_runs_root)
    if not tf.exists():
        raise SystemExit(f"missing {tf} — run mix_arm5.py --arm {args.data_arm} first")
    tokenizer = chat4.load_tokenizer(args.base_model)
    tok_fn = chat4.make_tokenize_fn(tokenizer, args.max_len)
    recs = manifest.read_jsonl(tf)

    ref = reference_adapter(REFERENCE_ADAPTER_ARM, REFERENCE_ADAPTER_STEP,
                            args.impl5_runs_root)
    key = weighting.signal_key(arm.variant, tf, args.base_model, ref, args.max_len)
    cache_path = signal_cache(arm.variant, key, args.data_dir or str(DATA_DIR))
    if not cache_path.exists():
        raise SystemExit(f"missing {cache_path}\n  python precompute_signal.py --variants "
                         f"{arm.variant}")
    cache = weighting.SignalCache.load(cache_path)

    # W4 — digests, row by row.
    if cache.n_rows != len(recs):
        raise AssertionError(f"cache covers {cache.n_rows} rows, file has {len(recs)}")
    bad, n_ped, n_ped_lab = 0, 0, 0
    for i, rec in enumerate(recs):
        enc = tok_fn({"messages": rec["messages"]})
        if weighting.row_digest(enc["input_ids"], enc["labels"]) != cache.row_hash[i]:
            bad += 1
        ped = bool(mixing.is_pedagogy(rec))
        n_lab = sum(1 for t in enc["labels"] if t != IGNORE)
        if ped:
            n_ped += 1
            n_ped_lab += n_lab
            if int(cache.offsets[i + 1] - cache.offsets[i]) != n_lab:
                raise AssertionError(f"row {i}: {n_lab} label tokens, "
                                     f"{int(cache.offsets[i + 1] - cache.offsets[i])} signals")
        elif int(cache.offsets[i + 1] - cache.offsets[i]) != 0:
            raise AssertionError(f"general row {i} carries {int(cache.offsets[i + 1] - cache.offsets[i])} "
                                 f"signals; replay tokens are never reweighted")
    if bad:
        raise AssertionError(f"W4: {bad} of {len(recs)} rows tokenise differently than the "
                             f"cache expects — the cache is stale")
    print(f"  W4 digests: {len(recs)} rows match the cache exactly  OK")
    print(f"  W5 streams: {n_ped:,} pedagogy rows / {n_ped_lab:,} reweighted label tokens; "
          f"{len(recs) - n_ped:,} general rows carry no signal  OK")

    # W7 — this arm's actual multiplier distribution on the real corpus.
    row_m, diag = weighting.build_row_multipliers(cache, arm.temperature)
    d = diag["multiplier"]
    if abs(d["mean"] - 1.0) > 1e-6:
        raise AssertionError(f"mean multiplier {d['mean']!r} != 1 on the real corpus")
    n_scattered = sum(1 for i, rec in enumerate(recs) if mixing.is_pedagogy(rec)
                      for w in weighting.scatter_to_labels(
                          tok_fn({"messages": rec["messages"]})["labels"], row_m[i])
                      if w != 0.0)
    if n_scattered != n_ped_lab:
        raise AssertionError(f"scatter produced {n_scattered} weighted positions for "
                             f"{n_ped_lab} label tokens")
    print(f"  W7 arm {arm.name} (variant {arm.variant}, T={arm.temperature:g}) on the real "
          f"corpus:")
    print(f"     signal median {diag['signal_median']:.4f}  1.4826*MAD "
          f"{diag['signal_mad_scaled']:.4f}  range [{diag['signal_min']:.3f}, "
          f"{diag['signal_max']:.3f}]")
    print(f"     m: mean {d['mean']:.6f}  min {d['min']:.3g}  p50 {d['p50']:.3g}  "
          f"p99 {d['p99']:.3g}  max {d['max']:.3g}")
    print(f"     ESS {d['ess']:.4f} — the cross-corpus-comparable number; quote this, not T")
    if d["ess"] < 0.02:
        print("     WARNING: ESS < 2%. James's a-T0.5/a-T1 ended ABOVE base's NLL here.")
    return {"ok": True, "train_file": str(tf), "cache": str(cache_path),
            "n_rows": len(recs), "n_pedagogy_rows": n_ped,
            "n_reweighted_label_tokens": n_ped_lab, "weighting": diag}


def main():
    args = parse_args()
    out = {}
    rng = np.random.default_rng(13)

    if args.stage == "fast":
        print("=" * 74)
        print("acceptance checks (fast) — no GPU, no real data, before the precompute")
        print("=" * 74, flush=True)
        out["W2_mean_one"] = check_w2_mean_one(rng)
        out["W6_temperature_limit"] = check_w6_temperature_limit(rng)
        out["W3_alignment"] = check_w3_alignment(args.model)
        out["W1_unit_weights"] = check_w1_unit_weights(args.model, PER_DEVICE_BATCH,
                                                       GRAD_ACCUM)
        out["loss_denom"] = out["W1_unit_weights"]["chosen"]
        print(f"\nAll fast checks passed. Pass --loss_denom {out['loss_denom']} to every arm.")
    else:
        arms = [resolve_arm(args.arm)] if args.arm else [resolve_arm(a) for a in ALL_ARMS]
        print("=" * 74)
        print(f"acceptance checks (full) — real mix, real cache, arms: "
              f"{', '.join(a.name for a in arms)}")
        print("=" * 74, flush=True)
        for arm in arms:
            print(f"\n-- {arm.name} --")
            out[arm.name] = check_full(args, arm)
        print("\nAll full checks passed.")

    dest = Path(args.out) if args.out else \
        ensure_dir(DATA_DIR) / f"acceptance_{args.stage}.json"
    ensure_dir(dest.parent)
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"-> {dest}")


if __name__ == "__main__":
    main()
