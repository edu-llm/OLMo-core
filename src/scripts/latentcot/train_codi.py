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
from olmo_core.latentcot.moe import describe_moe
from olmo_core.latentcot.techniques import (
    TECHNIQUES,
    as_arm,
    describe_techniques,
    get_technique,
)
from olmo_core.latentcot.tokens import assert_control_tokens_fit
from olmo_core.latentcot.tracking import ArmTracker, resolve_project
from olmo_core.latentcot.train_driver import (
    PRECISIONS,
    autocast_ctx,
    build_model,
    build_model_from_config,
    is_remote,
    load_checkpoint,
    publish_artifact,
    read_model_config,
    resolve_device,
    train_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=False)
    selector.add_argument(
        "--technique",
        choices=sorted(TECHNIQUES),
        help="which latent-reasoning post-training technique to fine-tune with. This is the "
        "selector for a real run: pick whichever the experiment showed wins. See "
        "--list-techniques.",
    )
    selector.add_argument(
        "--arm",
        choices=sorted(ARMS),
        help="the experiment-arm selector (A0-A4), kept so the study's runs stay reproducible. "
        "--technique is the same recipes under readable names, plus two that were never arms.",
    )
    parser.add_argument(
        "--list-techniques",
        action="store_true",
        help="print the technique catalog and exit",
    )
    parser.add_argument(
        "--rung",
        default="olmo2_370M",
        help="TransformerConfig factory name. Dense (olmo2_370M, olmo3_370M, ...) or MoE "
        "(olmoe_1B_7B, smallmoe, ...) -- both are supported and the MoE-specific bookkeeping "
        "switches on automatically (see olmo_core.latentcot.moe). NOTE: every MoE path in this "
        "repo routes through Triton kernels, so an MoE rung needs CUDA and cannot run on CPU.",
    )
    parser.add_argument(
        "--attn-backend",
        default=None,
        choices=["torch", "flash_2", "flash_3", "flash_4", "te"],
        help="override the attention backend. The olmo3_* factories hardcode flash_2, which "
        "raises at construction on an image without flash-attn (the eduLLM research image has "
        "none) -- pass 'torch' there. Same attention math, different kernel; the 4096 sliding "
        "window is a no-op at our ~300-token sequences.",
    )
    # Not `required=True`: --list-techniques must work without them. Checked below instead.
    parser.add_argument("--train-data")
    parser.add_argument("--test-data")
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
        "--log-every",
        type=int,
        default=100,
        help="append a train_history entry every N steps. Each carries elapsed_s and "
        "peak_mem_gb, so a short run measures step time and true peak memory (use --log-every 1 "
        "on a calibration run).",
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
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="wall-clock budget for the training loop. On reaching it the loop stops CLEANLY "
        "after the current step, saves, and returns, so model.pt and metrics.json are still "
        "written and anything later in the same job still runs. Set this a little under the "
        "platform's runtime bound: metrics.json is written last, so a run killed at the wall "
        "reports nothing at all and an evaluation sharing the job never starts. Arms cost very "
        "different amounts per step (A0/A1 do one forward per example, A2-A4 do K+2), so on a "
        "shared budget the CODI arms are the ones that run out -- which is the half of the "
        "experiment worth having. Ending at different step counts is a budget confound; pair this "
        "with a small --save-every and compare arms at the largest step they all reached "
        "(eval_arms_from_s3.py --match-steps).",
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
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project for streaming metrics. Normally leave this unset ON THE PLATFORM and "
        "pass --wandb-project to `edullm submit` instead, which exports EDULLM_WANDB_PROJECT "
        "into the container (and WANDB_RUN_GROUP, which groups the five arms, and the API key). "
        "This flag is the local/manual override and wins over the environment. With neither, the "
        "run trains untracked and says so.",
    )
    args = parser.parse_args()

    if args.list_techniques:
        print(describe_techniques())
        return
    if not (args.technique or args.arm):
        parser.error("one of --technique or --arm is required (see --list-techniques)")
    for required in ("train_data", "test_data"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required to train")

    device = resolve_device(args.device)
    distill_weight = 1.0
    if args.technique:
        technique = get_technique(
            args.technique,
            num_continuous_thoughts=args.num_continuous_thoughts,
            vocab_reg_entropy_floor=args.vocab_reg_entropy_floor,
        )
        arm, distill_weight = as_arm(technique), technique.distill_weight
    else:
        arm = ARMS[args.arm]
        if args.vocab_reg_entropy_floor is not None:
            # Whitelisted field, so this keeps the confound check valid (arms may differ in it).
            arm = replace(arm, vocab_reg_entropy_floor=args.vocab_reg_entropy_floor)

    # Build from the CHECKPOINT'S OWN config when there is one. A strict load requires the
    # architecture to match the weights exactly, and --rung can only name a registered factory
    # with this project's vocab size -- fine for the study's own rungs, not for an arbitrary
    # pretrained model (an MoE especially, whose expert shape lives in its config). The saved
    # config is the only description guaranteed to match. --rung stays the fallback.
    model_config = read_model_config(args.init_checkpoint) if args.init_checkpoint else None
    if model_config is not None:
        print("[model] building from the checkpoint's own config.json (ignoring --rung)")
        model = build_model_from_config(model_config, device=device, attn_backend=args.attn_backend)
    else:
        model = build_model(
            args.rung, init_seed=args.init_seed, device=device, attn_backend=args.attn_backend
        )
    if args.init_checkpoint:
        load_checkpoint(model, args.init_checkpoint)
        # The control tokens live in dolma2's padding; fail loudly here rather than index out of
        # the embedding mid-training if this checkpoint's vocab is smaller.
        assert_control_tokens_fit(model)

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
    # "A2-seed1" for an experiment arm, "codi-r1-seed1" for a technique -- args.arm is None in
    # the latter case, so key off whichever selector was used.
    leaf = f"{args.arm or args.technique}-seed{args.seed}"
    remote_dir: Optional[str] = None
    if is_remote(args.out):
        remote_dir = f"{str(args.out).rstrip('/')}/{leaf}"
        run_dir = Path(args.staging_dir) / leaf
        print(f"[out] remote: {remote_dir}\n[out] local staging: {run_dir}")
    else:
        run_dir = Path(args.out) / leaf
    run_dir.mkdir(parents=True, exist_ok=True)

    # The run's identity and every hyperparameter, assembled once and used for three things:
    # W&B's config (so the arm comparison is legible in the UI), the metrics.json header, and
    # the console summary. The arm-defining fields are in here deliberately — they are what the
    # confound control is *about*, so they have to be visible on the run.
    run_config = {
        "arm": arm.name,
        "technique": args.technique,
        "rung": args.rung,
        "built_from_checkpoint_config": model_config is not None,
        "init_seed": args.init_seed,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "attn_backend": args.attn_backend,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "num_continuous_thoughts": args.num_continuous_thoughts,
        "vocab_reg": arm.vocab_reg,
        "vocab_reg_weight": arm.vocab_reg_weight,
        "vocab_reg_entropy_floor": arm.vocab_reg_entropy_floor,
        "save_every": args.save_every,
        "val_fraction": args.val_fraction if val_examples else 0.0,
        # None on a dense base. Recorded rather than assumed: the arms fork a pretrained
        # checkpoint whose expert count and top-k come from ITS config, so "what did we
        # actually load" should be on the run rather than inferred later.
        "moe": describe_moe(model),
    }

    # Start tracking BEFORE training, so a run that dies mid-way still has its curve and its
    # config in W&B. `leaf` is "<arm>-seed<n>", which is exactly the run name wanted inside the
    # group the platform sets from the experiment.
    tracker = ArmTracker.start(
        project=resolve_project(args.wandb_project),
        name=leaf,
        config=run_config,
        dir=str(run_dir),
        tags=[arm.arm_mode, f"rung:{args.rung}", f"vocab_reg:{arm.vocab_reg}"],
    )
    # Recorded into metrics.json below, because that file is mirrored to S3 while this script's
    # stdout is redirected to a train.log the container takes with it (see .edullm/run.yaml).
    # Without this there would be no durable answer to "was this run tracked?".
    wandb_info = {"active": tracker.active, "url": tracker.url, "reason": tracker.reason}
    if tracker.active:
        print(f"[wandb] tracking {leaf}: {tracker.url or '(url pending)'}", flush=True)
    else:
        print(f"[wandb] NOT tracking -- {tracker.reason}", flush=True)

    try:
        history = train_arm(
            model,
            arm,
            train_source,  # type: ignore[arg-type]  # LatentCotDataset or a Subset of one
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            distill_weight=distill_weight,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            log_every=args.log_every,
            save_dir=run_dir,
            save_every=args.save_every,
            keep_last=args.keep_last,
            val_examples=val_examples,
            precision=args.precision,
            remote_dir=remote_dir,
            on_log=tracker.log,
            max_seconds=args.max_seconds,
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
    except BaseException as exc:
        # Mark the run failed in W&B and put the reason on it. This is the one place a
        # researcher will look, and a container that dies takes its log with it.
        tracker.summarize({"error": f"{type(exc).__name__}: {exc}"})
        tracker.finish(exit_code=1)
        raise

    metrics = {
        **run_config,
        "wandb": wandb_info,
        "best_checkpoint": best,  # {step, val_acc} of best.pt, or None if checkpointing was off
        "overall_acc": final_acc,
        "solve_rate_by_depth": final_by_depth,
        "train_history": history,
    }

    # The gate numbers, as summary columns so the five arms compare directly in a runs table.
    # The per-depth solve rates are also flattened to scalars, because gate A is a slope over
    # depth and a nested dict cannot be plotted against the other arms.
    tracker.summarize(
        {
            "overall_acc": final_acc,
            "solve_rate_by_depth": final_by_depth,
            **{f"solve_rate/depth_{depth}": rate for depth, rate in final_by_depth.items()},
            **({"best_step": best["step"], "best_val_acc": best["val_acc"]} if best else {}),
        }
    )
    tracker.finish()

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
