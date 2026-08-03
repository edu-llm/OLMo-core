#!/usr/bin/env python
"""End-to-end smoke: precompute -> cache -> attach -> weighted train, on a tiny model.

The acceptance checks verify the *objective* (W1–W7). This verifies the **plumbing between the
stages**, which is where the remaining silent-failure risk lives and which no unit test touches:

* the cache key ``precompute_signal.py`` writes is the key ``train_sft_klw.py`` looks up — a
  mismatch surfaces as "missing signal cache" rather than a wrong number, but only if you ran it,
* per-row digests agree across the two processes' independent tokenisation,
* the ``weights`` column survives ``Dataset.add_column`` and HF's column handling and actually
  reaches ``compute_loss`` (``assert_weighting_ran`` fires if not),
* ``PeftModel.disable_adapter()`` really yields π₀ while the plain forward yields π_SFT, so
  variant b's KL is not identically zero,
* variant a needs no reference and does not try to load one.

Runs on CPU in ~1 minute against ``hf-internal-testing/tiny-random-Olmo2ForCausalLM`` and a
synthetic 2-block mix. It asserts nothing about model quality — a random 2-layer model has none.

    python smoke_klw.py
    python smoke_klw.py --keep          # leave the scratch dir for inspection
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TINY = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"
BLOCK, PED, GEN = 32, 24, 8
N_BLOCKS = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=TINY)
    p.add_argument("--scratch", default=None)
    p.add_argument("--keep", action="store_true")
    return p.parse_args()


def synth_mix(path: Path) -> None:
    """A 2-block mix in the real layout: 24 pedagogy + 8 general per block, in that order.

    Pedagogy rows carry a system message and ``kind: "pedagogy"``; general rows carry no system
    message and ``kind: "general"`` — the two invariants ``mix_arm5.py`` asserts, reproduced here
    so the smoke exercises the same code paths the real mix does.
    """
    rows = []
    for b in range(N_BLOCKS):
        for i in range(PED):
            n_turns = 1 + (i % 3)
            msgs = [{"role": "system", "content": f"You are a tutor. Variant {i % 5}."},
                    {"role": "user", "content": f"Block {b} problem {i}: what is {i}+{b}?"}]
            for t in range(n_turns):
                msgs.append({"role": "assistant",
                             "content": f"What do you get when you add {t} and {i}?"})
                if t < n_turns - 1:
                    msgs.append({"role": "user", "content": f"I think it is {t + i}?"})
            rows.append({"messages": msgs, "kind": "pedagogy",
                         "dialogue_id": f"S_{b}_{i}", "problem_id": f"P_{b}_{i}",
                         "answer": str(i + b), "source": "smoke"})
        for i in range(GEN):
            rows.append({"messages": [
                {"role": "user", "content": f"Block {b} replay prompt {i}."},
                {"role": "assistant", "content": f"Replay answer {i}, a bit longer than usual."},
            ], "kind": "general", "dialogue_id": f"G_{b}_{i}", "source": "smoke-tulu"})
    assert len(rows) == N_BLOCKS * BLOCK
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows)} rows ({N_BLOCKS} x [{PED} ped + {GEN} gen]) -> {path}")


def make_reference_adapter(model_id: str, dest: Path) -> None:
    """An untrained LoRA adapter standing in for D4's ckpt-923.

    Untrained is fine for plumbing but *not* for the KL: a zero-initialised LoRA B matrix makes
    π_SFT identical to π₀ and every variant-b signal exactly 0, which would make the smoke pass
    while proving nothing. So the B matrices are filled with noise, which is what makes the
    "variant b KL is not identically zero" assertion below meaningful.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(13)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    n = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0.0, 0.05)
                n += 1
    model.save_pretrained(str(dest))
    print(f"  reference adapter -> {dest} ({n} lora_B matrices perturbed so pi_SFT != pi_0)")


def run(cmd: list[str], **kw) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, text=True, capture_output=True, **kw)
    tail = (r.stdout or "").strip().splitlines()
    for line in tail[-25:]:
        print("  | " + line)
    if r.returncode:
        print((r.stderr or "")[-3000:])
        raise SystemExit(f"FAILED (exit {r.returncode}): {' '.join(cmd)}")


def main():
    args = parse_args()
    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="klw_smoke_"))
    i5_runs = scratch / "impl5_runs"
    data_dir = scratch / "data"
    runs_root = scratch / "runs"
    d4 = i5_runs / "D4"
    d4.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"scratch: {scratch}")

    print("\n-- synthetic mix --")
    train = d4 / "socrateach_sft_train.jsonl"
    synth_mix(train)
    shutil.copy(train, d4 / "socrateach_sft_val.jsonl")
    shutil.copy(train, d4 / "socrateach_sft_test.jsonl")

    print("\n-- reference adapter (stands in for D4 ckpt-923) --")
    make_reference_adapter(args.model, d4 / "ckpt-923")

    common = ["--impl5_runs_root", str(i5_runs), "--data_dir", str(data_dir),
              "--base_model", args.model, "--max_len", "512"]

    print("\n-- precompute: both variants, one pass --")
    run([sys.executable, "precompute_signal.py", *common, "--variants", "a,b",
         "--max_batch_tokens", "4096", "--max_batch_rows", "8"])

    caches = sorted(data_dir.glob("signal_*.npz"))
    print(f"\ncaches: {[c.name for c in caches]}")
    if len(caches) != 2:
        raise SystemExit(f"expected 2 caches (variant a and b), got {[c.name for c in caches]}")

    # The assertion that makes the variant-b path meaningful rather than merely green.
    sys.path.insert(0, str(HERE))
    from klw import weighting
    for c in caches:
        cache = weighting.SignalCache.load(c)
        n_ped_rows = int((cache.offsets[1:] - cache.offsets[:-1] > 0).sum())
        vals = cache.values
        print(f"  {c.name}: variant {cache.variant}  {vals.size} signals over {n_ped_rows} "
              f"pedagogy rows  range [{vals.min():.4f}, {vals.max():.4f}]  "
              f"mean {vals.mean():.4f}")
        if n_ped_rows != N_BLOCKS * PED:
            raise SystemExit(f"{c.name}: {n_ped_rows} pedagogy rows, expected {N_BLOCKS * PED}")
        if cache.variant == "b" and float(vals.max()) <= 0.0:
            raise SystemExit(
                "variant b's KL is identically zero — disable_adapter() is not actually "
                "switching between pi_0 and pi_SFT, so the signal is meaningless")
        if cache.variant == "a" and float(vals.min()) < 0.0:
            raise SystemExit("variant a is a surprise (-log p) and cannot be negative")

    print("\n-- train: aT8 (variant a, no reference) and bT1 (variant b) --")
    for arm in ("aT8", "bT1"):
        run([sys.executable, "train_sft_klw.py", "--arm", arm,
             "--runs_root", str(runs_root), *common,
             "--loss_denom", "global", "--num_epochs", "1.0",
             "--per_device_batch", "8", "--grad_accum", "4",
             "--eval_steps", "1000", "--save_steps", "1000", "--logging_steps", "1"])
        mf = json.loads((runs_root / arm / "manifest.json").read_text())["training"]
        w = mf["weighting"]["multiplier"]
        rt = mf["weighting_runtime"]
        print(f"  {arm}: weighted_batches={rt['weighted_batches']} "
              f"unweighted={rt['unweighted_batches']} denom={rt['loss_denom_used']} | "
              f"m mean {w['mean']:.6f} ESS {w['ess']:.4f} max {w['max']:.3f}")
        if rt["weighted_batches"] == 0:
            raise SystemExit(f"{arm} trained UNWEIGHTED — the weights column was stripped")
        if rt["loss_denom_used"] != "global":
            raise SystemExit(f"{arm} used denom {rt['loss_denom_used']}, expected global")
        if abs(w["mean"] - 1.0) > 1e-6:
            raise SystemExit(f"{arm} mean multiplier {w['mean']} != 1")

    print("\n" + "=" * 74)
    print("SMOKE PASSED — precompute/cache/attach/weighted-train chain is wired end to end.")
    print("=" * 74)
    if args.keep:
        print(f"scratch kept at {scratch}")
    else:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
