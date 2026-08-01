"""Positive control: can ANY configuration solve the EASIEST MQAR point?

WHY THIS EXISTS: the first calibration attempt (FarmShare 1670922) returned accuracy 0.000 with
the loss plateaued at ~8.32 = ln(4096) -- the model had learned "the answer is somewhere in the
value half of the vocabulary" and nothing more. Zero recall, on the easiest rung of the grid.

That is a harness result, not a science result, and calibrating on it would have been
meaningless. Before sweeping difficulty again, establish that the setup can learn MQAR at all.
**A difficulty sweep whose easiest point scores zero cannot distinguish "hard task" from "broken
setup".**

Four candidate causes, swept here on the easiest config only:

1. **Training budget.** Zoology trains on 100k examples for 64 epochs -- about 6.4M
   example-presentations. The failed run did 3000 steps x batch 32 = 96k, roughly **64x less**.
2. **Learning rate.** Zoology sweeps ``logspace(-4, -2, 4)`` because the right LR moves with
   model size; a single fixed 1e-3 may simply be wrong here.
3. **Attention depth.** Recall needs match-then-copy. With a single attention layer the
   preceding convolutions must supply the "previous token" half of an induction head, which is
   possible (a k=3 causal conv can co-locate a key with its value) but is a harder circuit to
   find than the textbook two-attention-layer version.
4. **Vocabulary size.** An 8192-way head over 4 answers spends most of its capacity on the
   softmax rather than the binding.

Runs are short and report the learning curve, so a partially-learning configuration is visible
as a falling loss even when accuracy is still zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from mqar_data import IGNORE_INDEX, MQARConfig, make_mqar_batch, mqar_accuracy  # noqa: E402
from mqar_model import MQARHybrid  # noqa: E402


def run(
    *,
    cfg: MQARConfig,
    lr: float,
    attention_layers: tuple,
    n_layers: int,
    d_model: int,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    model = MQARHybrid(
        vocab_size=cfg.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        attention_layers=attention_layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    gen = torch.Generator().manual_seed(seed * 7919 + 3)

    curve = []
    t0 = time.time()
    model.train()
    for step in range(steps):
        tokens, labels = make_mqar_batch(cfg, batch_size, gen)
        tokens, labels = tokens.to(device), labels.to(device)
        loss = F.cross_entropy(
            model(tokens).reshape(-1, cfg.vocab_size),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 8) == 0 or step == steps - 1:
            curve.append(round(float(loss), 3))

    model.eval()
    eval_gen = torch.Generator().manual_seed(seed * 7919 + 555)
    with torch.no_grad():
        accs = []
        for _ in range(8):
            tk, lb = make_mqar_batch(cfg, batch_size, eval_gen)
            accs.append(mqar_accuracy(model(tk.to(device)).cpu(), lb))
    return {
        "lr": lr,
        "attention_layers": list(attention_layers),
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": cfg.vocab_size,
        "steps": steps,
        "accuracy": sum(accs) / len(accs),
        "loss_curve": curve,
        "n_params": model.n_params,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--out", default="mqar_positive_control.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}\n", flush=True)

    # Easiest rung only: 4 pairs, 64 tokens. If this cannot be solved, nothing harder can.
    trials = []
    for vocab in (256, 8192):
        for lr in (3e-4, 1e-3, 3e-3):
            for attn in ((2,), (1, 3)):
                trials.append((vocab, lr, attn))

    print(
        f"easiest config N64_D4, {len(trials)} trials x {args.steps} steps "
        f"(batch {args.batch_size} = {args.steps * args.batch_size:,} examples seen)\n",
        flush=True,
    )
    print(f"{'vocab':>6}{'lr':>8}{'attn':>10}{'acc':>8}   loss curve", flush=True)
    print("-" * 78, flush=True)

    rows = []
    for vocab, lr, attn in trials:
        cfg = MQARConfig(seq_len=64, num_pairs=4, vocab_size=vocab)
        rec = run(
            cfg=cfg,
            lr=lr,
            attention_layers=attn,
            n_layers=4,
            d_model=args.d_model,
            steps=args.steps,
            batch_size=args.batch_size,
            device=device,
        )
        rows.append(rec)
        flag = "  <-- LEARNS" if rec["accuracy"] > 0.5 else ""
        print(
            f"{vocab:>6}{lr:>8.0e}{str(attn):>10}{rec['accuracy']:>8.3f}   "
            f"{' '.join(f'{c:.2f}' for c in rec['loss_curve'])}{flag}",
            flush=True,
        )

    best = max(rows, key=lambda r: r["accuracy"])
    print("\n=== VERDICT ===", flush=True)
    if best["accuracy"] > 0.5:
        print(
            f"  Setup CAN learn MQAR: {best['accuracy']:.3f} at vocab={best['vocab_size']}, "
            f"lr={best['lr']:.0e}, attention_layers={best['attention_layers']}.\n"
            f"  Use this as the calibration baseline and sweep difficulty from here.",
            flush=True,
        )
    else:
        print(
            f"  STILL ZERO everywhere (best {best['accuracy']:.3f}). The blocker is not LR, "
            f"vocab, or attention depth.\n"
            f"  Next suspects: training budget (Zoology uses ~64x more), or the model needs "
            f"more width//depth.\n"
            f"  Do NOT run a difficulty sweep until the easiest point is solvable.",
            flush=True,
        )

    Path(args.out).write_text(json.dumps({"trials": rows, "best": best}, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
