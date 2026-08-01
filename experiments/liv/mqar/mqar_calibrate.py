"""MQAR difficulty calibration for the LIV study. Run on the BASELINE ONLY.

PURPOSE: find MQAR settings where the baseline ``L0`` topology is **not saturated** -- neither at
ceiling nor at chance -- so the endpoint can actually discriminate between arms. An easy MQAR at
100% has no evidentiary value, and one at 0% has none either.

TWO RULES THIS SCRIPT ENFORCES, both from the protocol:

1. **Calibrate on the baseline only.** Never tune difficulty while looking at a treatment arm --
   that is choosing the test until it gives the answer you want. This script has no arm flag; it
   builds the dense-gate k=3 baseline and nothing else.

2. **The endpoint is SUCCESS RATE OVER SEEDS, not mean accuracy.** ``KDA/run_mqar_var.sbatch``
   documents that MQAR trainability at ~1M params is **bimodal**: a run either finds the recall
   algorithm or sits at chance. An n=1 ladder was non-monotone in load
   (64.5% -> 0.88% -> 97.8% -> 40.3%), which is not noise around a mean -- it is two modes.
   Averaging across that is meaningless, so we report the fraction of seeds that solved it, plus
   the full per-seed list so the bimodality is visible rather than hidden.

Output is JSON: per (config, seed) accuracy, plus per-config success rate and a recommended
operating point.
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

from mqar_data import (  # noqa: E402
    CALIBRATED_ATTENTION_LAYERS,
    CALIBRATED_BATCH_SIZE,
    CALIBRATED_EXAMPLES,
    CALIBRATED_LR,
    CALIBRATED_STEPS,
    CALIBRATION_GRID,
    DISTANCE_SWEEP,
    IGNORE_INDEX,
    ZOOLOGY_EXAMPLES,
    MQARConfig,
    degenerate_floor,
    make_mqar_batch,
    mqar_accuracy,
)
from mqar_model import MQARHybrid  # noqa: E402

# A run counts as having "found the algorithm" above this accuracy. Not an arbitrary cut: across
# the 12 positive-control trials, NO run landed between 0.30 and 0.80 -- 2 solved (0.995, 1.000),
# 6 sat on the 1/D degenerate floor, 4 fell below it. 0.80 sits inside an empty gap.
SOLVE_THRESHOLD = 0.80

# Non-saturation band for the recommended operating point: not at ceiling, not at chance.
TARGET_LO, TARGET_HI = 0.20, 0.80


def train_one(
    cfg: MQARConfig,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    d_model: int,
    n_layers: int,
    lr: float,
    device: torch.device,
    kernel_size: int = 3,
    log_every: int = 0,
) -> dict:
    """
    Train one baseline model on one MQAR config with one seed.

    :returns: A record with final accuracy, best accuracy, and loss trace endpoints.
    """
    torch.manual_seed(seed)
    model = MQARHybrid(
        vocab_size=cfg.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        attention_layers=CALIBRATED_ATTENTION_LAYERS,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    gen = torch.Generator().manual_seed(seed * 100_003 + 17)
    first_loss = None
    t0 = time.time()

    model.train()
    for step in range(steps):
        tokens, labels = make_mqar_batch(cfg, batch_size, gen)
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), labels.reshape(-1), ignore_index=IGNORE_INDEX
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if first_loss is None:
            first_loss = float(loss)
        if log_every and step % log_every == 0:
            print(f"      step {step:>5} loss {float(loss):.4f}", flush=True)

    # Evaluate on held-out batches drawn from a disjoint RNG stream.
    model.eval()
    eval_gen = torch.Generator().manual_seed(seed * 100_003 + 999_331)
    accs = []
    with torch.no_grad():
        for _ in range(8):
            tokens, labels = make_mqar_batch(cfg, batch_size, eval_gen)
            logits = model(tokens.to(device))
            accs.append(mqar_accuracy(logits.cpu(), labels))
    acc = sum(accs) / len(accs)

    floor = degenerate_floor(cfg)
    return {
        "config": cfg.label,
        "seq_len": cfg.seq_len,
        "num_pairs": cfg.num_pairs,
        "seed": seed,
        "accuracy": acc,
        "solved": acc >= SOLVE_THRESHOLD,
        # The comparison that matters: 1/D is what a model scores by learning "the answer is one
        # of the D values here" without binding anything. Accuracy alone is not interpretable.
        "degenerate_floor": floor,
        "above_floor": acc > floor * 1.5,
        "first_loss": first_loss,
        "final_loss": float(loss),
        "n_params": model.n_params,
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="seeds per config (bimodality needs >1)")
    ap.add_argument("--steps", type=int, default=CALIBRATED_STEPS)
    ap.add_argument("--batch-size", type=int, default=CALIBRATED_BATCH_SIZE)
    ap.add_argument(
        "--allow-short-budget",
        action="store_true",
        help="permit a training budget below the calibrated one (for smoke tests only)",
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=CALIBRATED_LR)
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument(
        "--grid",
        choices=["calibration", "distance", "both"],
        default="calibration",
        help="'calibration' sweeps capacity+length together (Zoology's grid); "
        "'distance' holds pairs fixed and stretches length only",
    )
    ap.add_argument("--out", default="mqar_calibration.json")
    args = ap.parse_args()

    # Refuse to run under-budget. This is the exact failure of job 1670963: a stale sbatch file
    # passed 96k examples while the script's own calibration assumed 512k, and the resulting
    # table of near-floor scores looked like "the task is too hard" rather than "we under-trained".
    examples = args.steps * args.batch_size
    if examples < CALIBRATED_EXAMPLES and not args.allow_short_budget:
        print(
            f"REFUSING TO RUN: budget {examples:,} examples "
            f"({args.steps} steps x {args.batch_size}) is below the calibrated "
            f"{CALIBRATED_EXAMPLES:,}, the only budget measured to solve this task here.\n"
            f"Under-training is indistinguishable from a too-hard task in the output, so this "
            f"would produce a confident but meaningless calibration.\n"
            f"Pass --allow-short-budget only for smoke tests.",
            file=sys.stderr,
        )
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"budget: {examples:,} examples ({examples / ZOOLOGY_EXAMPLES:.0%} of Zoology's)",
        flush=True,
    )

    grids = {
        "calibration": CALIBRATION_GRID,
        "distance": DISTANCE_SWEEP,
        "both": CALIBRATION_GRID + DISTANCE_SWEEP,
    }[args.grid]

    print(
        f"\nBASELINE ONLY (dense gates, k={args.kernel_size}). "
        f"{len(grids)} configs x {args.seeds} seeds x {args.steps} steps\n",
        flush=True,
    )

    rows = []
    for cfg in grids:
        print(f"  {cfg.label}  (seq_len={cfg.seq_len}, pairs={cfg.num_pairs})", flush=True)
        for seed in range(args.seeds):
            rec = train_one(
                cfg,
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                lr=args.lr,
                device=device,
                kernel_size=args.kernel_size,
            )
            rows.append(rec)
            print(
                f"    seed {seed}: acc {rec['accuracy']:.3f} "
                f"{'SOLVED' if rec['solved'] else '.'}  "
                f"loss {rec['first_loss']:.2f}->{rec['final_loss']:.3f}  "
                f"({rec['seconds']:.0f}s)",
                flush=True,
            )

    # --- report ---------------------------------------------------------------------------
    print(
        f"\n{'config':<14}{'1/D floor':>10}{'success':>9}{'>floor':>8}"
        f"{'  accs (per seed)':<30}",
        flush=True,
    )
    print("-" * 72, flush=True)
    summary = []
    for cfg in grids:
        got = [r for r in rows if r["config"] == cfg.label]
        rate = sum(r["solved"] for r in got) / len(got)
        above = sum(r["above_floor"] for r in got) / len(got)
        accs = sorted(r["accuracy"] for r in got)
        summary.append(
            {
                "config": cfg.label,
                "success_rate": rate,
                "above_floor_rate": above,
                "degenerate_floor": degenerate_floor(cfg),
                "accuracies": accs,
            }
        )
        print(
            f"{cfg.label:<14}{degenerate_floor(cfg):>10.3f}{rate:>8.0%}{above:>8.0%}"
            f"  {' '.join(f'{a:.2f}' for a in accs)}",
            flush=True,
        )

    # Bimodality check: are accuracies clustered at the extremes rather than spread?
    all_accs = [r["accuracy"] for r in rows]
    mid = sum(1 for a in all_accs if 0.2 < a < 0.8)
    print(
        f"\nbimodality: {len(all_accs) - mid}/{len(all_accs)} runs at an extreme "
        f"(<0.2 or >0.8), {mid} in between",
        flush=True,
    )
    if mid <= len(all_accs) * 0.25:
        print("  -> strongly bimodal, as the protocol predicted. Success rate is the right "
              "endpoint; a mean would be meaningless here.", flush=True)

    usable = [s for s in summary if TARGET_LO <= s["success_rate"] <= TARGET_HI]
    print("\n=== RECOMMENDED OPERATING POINT ===", flush=True)
    if usable:
        pick = min(usable, key=lambda s: abs(s["success_rate"] - 0.5))
        print(f"  {pick['config']} at success rate {pick['success_rate']:.0%} -- "
              f"discriminates, since it is neither at ceiling nor at chance.", flush=True)
    else:
        rates = {s["config"]: s["success_rate"] for s in summary}
        print(f"  NONE in the {TARGET_LO:.0%}-{TARGET_HI:.0%} band: {rates}", flush=True)
        if all(r >= 0.99 for r in rates.values()):
            print("  All at ceiling -> make it harder (more pairs, or longer distance).",
                  flush=True)
        elif all(r <= 0.01 for r in rates.values()):
            print("  All at chance -> make it easier, or train longer / widen the model.",
                  flush=True)

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {
                "note": "BASELINE ONLY. Endpoint is success rate over seeds, not mean accuracy.",
                "solve_threshold": SOLVE_THRESHOLD,
                "args": vars(args),
                "device": str(device),
                "runs": rows,
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
