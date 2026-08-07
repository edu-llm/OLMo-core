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
from dataclasses import replace
from pathlib import Path
from typing import Optional

import torch

from olmo_core.latentcot.arms import ARMS
from olmo_core.latentcot.data.dataset import LatentCotDataset
from olmo_core.latentcot.evaluate import overall_accuracy, solve_rate_by_depth
from olmo_core.latentcot.train_driver import (
    PRECISIONS,
    autocast_ctx,
    build_model,
    is_remote,
    load_checkpoint,
    publish_artifact,
    resolve_device,
    train_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--rung", default="olmo2_370M", help="TransformerConfig factory name")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument(
        "--num-continuous-thoughts",
        type=int,
        default=10,
        help="K continuous-thought budget; default 10 gives ~2 steps of headroom over the "
        "deepest graph (depth 8) so a depth-8 failure means 'can't superpose', not 'one step short'.",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="examples per optimizer step. NOTE: the CODI student is processed one example at a "
        "time, so this does NOT batch the GPU — raising it adds sequential forwards and costs "
        "wall-clock linearly. It is an effective-batch (gradient-noise) knob, not a throughput "
        "one, until the per-example loop is packed. Must match across arms.",
    )
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=PRECISIONS,
        help="bf16 (default) runs forwards under bf16 autocast and enables TF32 on CUDA; fp32 is "
        "bit-identical to the pre-flag driver. No-op off CUDA. Identical across arms.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="linear LR warmup steps (WSD); eases the fine-tune into the pretrained base. "
        "Shared across arms, so it stays confound-clean.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="save a rolling checkpoint every N steps (0 disables); aim for ~10 saves over the "
        "run (N ~= steps/10). Keeps the last --keep-last plus a best.pt; a crash loses <= one "
        "interval.",
    )
    parser.add_argument(
        "--keep-last", type=int, default=2, help="rolling checkpoints to retain (oldest deleted)"
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="fraction of TRAIN held out as a validation split to pick best.pt. Never the gate "
        "test set (that would be model selection on the eval data). Seeded independently of "
        "--seed so the split is identical across arms/seeds.",
    )
    parser.add_argument(
        "--best-eval-size",
        type=int,
        default=200,
        help="cap on validation examples scored per checkpoint (bounds A0's generation cost)",
    )
    parser.add_argument("--init-seed", type=int, default=0, help="SAME across arms (shared init)")
    parser.add_argument("--seed", type=int, default=0, help="data-shuffle seed (per run)")
    parser.add_argument("--init-checkpoint", default=None, help="optional shared base state_dict")
    parser.add_argument(
        "--device", default="auto", help="'auto' (cuda if available else cpu), 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--vocab-reg-entropy-floor",
        type=float,
        default=None,
        help="override R1's anti-collapse entropy floor (nats); default off (arm's 0.0). "
        "Set e.g. 1.0 to re-run A3 with the floor on if thoughts are seen to collapse.",
    )
    parser.add_argument(
        "--out",
        default="runs/latentcot",
        help="output root: a local directory OR a remote URI (e.g. the platform's "
        "$EDULLM_CHECKPOINT_DIR, an s3:// prefix). With a URI, artifacts are staged in "
        "--staging-dir and mirrored to the URI as each one is written.",
    )
    parser.add_argument(
        "--staging-dir",
        default="runs/latentcot-staging",
        help="local scratch used only when --out is a remote URI",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    arm = ARMS[args.arm]
    if args.vocab_reg_entropy_floor is not None:
        # Whitelisted field, so this keeps the confound check valid (arms may differ in it).
        arm = replace(arm, vocab_reg_entropy_floor=args.vocab_reg_entropy_floor)

    model = build_model(args.rung, init_seed=args.init_seed, device=device)
    if args.init_checkpoint:
        # Fork the shared base (the "best model") — a .pt state_dict or a local/S3 ckpt dir.
        load_checkpoint(model, args.init_checkpoint)

    train_ds = LatentCotDataset(args.train_data, args.num_continuous_thoughts)

    # Carve a fixed validation split off TRAIN for best-checkpoint selection. It is seeded
    # independently of --seed so the split is identical across arms and seeds, and it never
    # touches the held-out gate test set (selecting "best" there would be model selection on
    # the eval data). With checkpointing off (--save-every 0) we train on the full train set.
    val_examples = None
    train_source: object = train_ds
    if args.save_every and args.val_fraction > 0:
        from torch.utils.data import Subset

        n = len(train_ds)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
        n_val = max(1, int(n * args.val_fraction))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        train_source = Subset(train_ds, train_idx)
        val_examples = [train_ds[i] for i in val_idx][: args.best_eval_size]

    # --out may be a remote URI (the platform passes $EDULLM_CHECKPOINT_DIR, an s3:// prefix).
    # Path() would silently rewrite that to a relative local dir named "s3:" and the artifacts
    # would vanish with the container, so stage locally and mirror every write to the URI.
    leaf = f"{args.arm}-seed{args.seed}"
    remote_dir: Optional[str] = None
    if is_remote(args.out):
        remote_dir = f"{str(args.out).rstrip('/')}/{leaf}"
        run_dir = Path(args.staging_dir) / leaf
        print(f"[out] remote: {remote_dir}\n[out] local staging: {run_dir}")
    else:
        run_dir = Path(args.out) / leaf
    run_dir.mkdir(parents=True, exist_ok=True)

    history = train_arm(
        model,
        arm,
        train_source,  # type: ignore[arg-type]  # LatentCotDataset or a Subset of one
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        save_dir=run_dir,
        save_every=args.save_every,
        keep_last=args.keep_last,
        val_examples=val_examples,
        precision=args.precision,
        remote_dir=remote_dir,
    )
    best = None
    best_path = run_dir / "best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())

    test_ds = LatentCotDataset(args.test_data, args.num_continuous_thoughts)
    examples = [test_ds[i] for i in range(len(test_ds))]
    # Score under the same precision the run trained and best-selected under, so the reported
    # gate numbers and best.json's val_acc are measured on the same footing.
    with autocast_ctx(args.precision, device):
        final_acc = overall_accuracy(model, examples, arm.arm_mode)
        final_by_depth = solve_rate_by_depth(model, examples, arm.arm_mode)
    metrics = {
        "arm": arm.name,
        "rung": args.rung,
        "init_seed": args.init_seed,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "num_continuous_thoughts": args.num_continuous_thoughts,
        "vocab_reg": arm.vocab_reg,
        "vocab_reg_weight": arm.vocab_reg_weight,
        "vocab_reg_entropy_floor": arm.vocab_reg_entropy_floor,
        "save_every": args.save_every,
        "val_fraction": args.val_fraction if val_examples else 0.0,
        "best_checkpoint": best,  # {step, val_acc} of best.pt, or None if checkpointing was off
        "overall_acc": final_acc,
        "solve_rate_by_depth": final_by_depth,
        "train_history": history,
    }

    # run_dir was created above; write the final (last-step) weights as the canonical artifact.
    torch.save(model.state_dict(), run_dir / "model.pt")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    # Mirror the two artifacts the analysis actually needs. Last, so a mirrored metrics.json
    # means the run finished.
    publish_artifact(run_dir / "model.pt", remote_dir)
    publish_artifact(run_dir / "metrics.json", remote_dir)

    print(f"[{arm.name} seed={args.seed}] overall_acc={metrics['overall_acc']:.3f}")
    print(f"  solve_rate_by_depth: {metrics['solve_rate_by_depth']}")
    if best is not None:
        print(f"  best.pt: step {best['step']} (val_acc={best['val_acc']:.3f})")
    print(f"  wrote {run_dir}/model.pt + metrics.json")


if __name__ == "__main__":
    main()
