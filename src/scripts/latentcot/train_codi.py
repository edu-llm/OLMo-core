"""
Phase 8 training driver: train one experiment arm at one model rung, save a checkpoint,
and evaluate the held-out set. Core lives in ``olmo_core.latentcot.train_driver`` (tested);
this script is argparse + checkpoint save + end-of-run eval.

A direct training loop (not the framework Trainer): the CODI student is processed per example
(variable-length prefixes), which doesn't fit the token-array DataLoader the Trainer expects.
It reuses the exact ``arm_loss`` the ``CodiTransformerTrainModule`` uses, so results match.

**Matched starts:** build every arm with the SAME ``--init-seed`` (identical initial weights =
the shared base), or the same ``--init-checkpoint``. Only ``--seed`` (data shuffle) and the arm's
whitelisted fields vary between arms. Run ``preflight.py`` first.

Usage (per arm; GPU auto-detected)::

    .venv/bin/python src/scripts/latentcot/train_codi.py \
        --arm A2 --rung olmo2_370M \
        --train-data data/latentcot/graph-reachability-depth/conversations/train-00000.jsonl \
        --test-data  data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl \
        --steps 5000 --batch-size 16 --init-seed 0 --seed 1 --out runs/latentcot
"""

import argparse
import json
from pathlib import Path

import torch

from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.evaluate import overall_accuracy, solve_rate_by_depth
from olmo_core.latentcot.train_driver import build_model, train_arm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--rung", default="olmo2_370M", help="TransformerConfig factory name")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--num-continuous-thoughts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--init-seed", type=int, default=0, help="SAME across arms (shared init)")
    parser.add_argument("--seed", type=int, default=0, help="data-shuffle seed (per run)")
    parser.add_argument("--init-checkpoint", default=None, help="optional shared base state_dict")
    parser.add_argument("--out", type=Path, default=Path("runs/latentcot"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arm = ARMS[args.arm]

    model = build_model(args.rung, init_seed=args.init_seed, device=device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))

    train_ds = LatentCotDataset(args.train_data, args.num_continuous_thoughts)
    history = train_arm(
        model,
        arm,
        train_ds,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    test_ds = LatentCotDataset(args.test_data, args.num_continuous_thoughts)
    examples = [test_ds[i] for i in range(len(test_ds))]
    metrics = {
        "arm": arm.name,
        "rung": args.rung,
        "init_seed": args.init_seed,
        "seed": args.seed,
        "steps": args.steps,
        "overall_acc": overall_accuracy(model, examples, arm.arm_mode),
        "solve_rate_by_depth": solve_rate_by_depth(model, examples, arm.arm_mode),
        "train_history": history,
    }

    run_dir = args.out / f"{args.arm}-seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    print(f"[{arm.name} seed={args.seed}] overall_acc={metrics['overall_acc']:.3f}")
    print(f"  solve_rate_by_depth: {metrics['solve_rate_by_depth']}")
    print(f"  wrote {run_dir}/model.pt + metrics.json")


if __name__ == "__main__":
    main()
