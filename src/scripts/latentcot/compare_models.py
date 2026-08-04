"""
Benchmark our latent-CoT arms against the "best model" baseline (PRD Phase 6, head-to-head).

The "best model" is a general pretrained OLMo-370M checkpoint
(``s3://edullm-olmo-370m-ckpts/olmo3-370m/run-10b-equal/step12716/``,
W&B run ``f08ey8cm``). Per the experiment design, it does **not** appear zero-shot — it has
never seen our symbolic graph-reachability format and would score ~chance. Instead every arm
is *forked from it* (``train_codi.py --rung olmo3_370M --init-checkpoint s3://…``), and the
baseline is **A0**: the best model fine-tuned the *normal* way (explicit written-out CoT, no
continuous thoughts, no vocab regularizer). Comparing our latent arms (A2/A3/A4) against A0
from the identical starting weights isolates the *training method*, not the init or architecture.

This script loads the trained baseline + our arm checkpoints, tabulates reachability
solve-rate-by-depth side by side, and reports the per-depth advantage ``acc_ours(D) -
acc_baseline(D)`` and its least-squares slope vs depth — the superposition signal the theory
predicts to be positive and increasing.

Checkpoints may be a plain ``model.pt`` state_dict (from ``train_codi.py``) or a local/S3
OLMo-core checkpoint directory; both are handled by ``load_checkpoint`` (S3 needs AWS creds).

Usage::

    .venv/bin/python src/scripts/latentcot/compare_models.py \
        --test-data data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl \
        --num-continuous-thoughts 8 --model olmo3_370M \
        --baseline A0=runs/latentcot/A0-seed1/model.pt \
        --ours A2=runs/latentcot/A2-seed1/model.pt \
        --ours A3=runs/latentcot/A3-seed1/model.pt
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import Example
from olmo_core.latentcot.evaluate import (
    ARM_MODES,
    gate_a_curve,
    linear_slope,
    overall_accuracy,
    solve_rate_by_depth,
)
from olmo_core.latentcot.tokens import TOKENIZER_CONFIG
from olmo_core.latentcot.train_driver import load_checkpoint, resolve_device
from olmo_core.nn.transformer import TransformerConfig


def load_examples(path: str, num_continuous_thoughts: int):
    """Encode a held-out ``conversations`` jsonl into eval-ready examples (as in ``eval.py``)."""
    examples = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            examples.append(
                encode_example(Example.from_dict(json.loads(line)), num_continuous_thoughts)
            )
    return examples


def parse_arm_spec(spec: str) -> Tuple[str, str]:
    """Split an ``ARM=CKPT`` CLI value; the arm name must be a known arm (A0/A1/A2/A3/A4)."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected ARM=CKPT, got {spec!r}")
    arm, ckpt = spec.split("=", 1)
    if arm not in ARM_MODES:
        raise argparse.ArgumentTypeError(f"unknown arm {arm!r}; known arms: {sorted(ARM_MODES)}")
    return arm, ckpt


def evaluate_arm(model_config, arm: str, ckpt: str, examples, device: str = "cpu") -> dict:
    """Load one arm's checkpoint and score it on the held-out set."""
    mode = ARM_MODES[arm]
    model = model_config.build(init_device="cpu")
    load_checkpoint(model, ckpt)
    model.to(device)
    model.eval()
    return {
        "checkpoint": ckpt,
        "mode": mode,
        "overall_acc": overall_accuracy(model, examples, mode),
        "solve_rate_by_depth": solve_rate_by_depth(model, examples, mode),
    }


def print_table(baseline_arm: str, baseline: dict, ours: dict) -> None:
    """Print the solve-rate-by-depth side by side, with per-depth advantage over the baseline."""
    arms = list(ours)
    depths = sorted(
        set(baseline["solve_rate_by_depth"]).union(
            *(o["solve_rate_by_depth"] for o in ours.values())
        )
    )
    header = f"{'depth':>6}  {baseline_arm + ' (base)':>14}" + "".join(
        f"{a:>10}{'Δ':>8}" for a in arms
    )
    print(header)
    print("-" * len(header))
    for d in depths:
        base_acc = baseline["solve_rate_by_depth"].get(d)
        base_s = f"{base_acc:.3f}" if base_acc is not None else "  -  "
        row = f"{d:>6}  {base_s:>14}"
        for a in arms:
            acc = ours[a]["solve_rate_by_depth"].get(d)
            if acc is None or base_acc is None:
                row += f"{'  -  ':>10}{'  -  ':>8}"
            else:
                row += f"{acc:>10.3f}{acc - base_acc:>+8.3f}"
        print(row)
    print("-" * len(header))
    overall = f"{'all':>6}  {baseline['overall_acc']:>14.3f}"
    for a in arms:
        overall += f"{ours[a]['overall_acc']:>10.3f}{ours[a]['overall_acc'] - baseline['overall_acc']:>+8.3f}"
    print(overall)


def maybe_plot(comparison: dict, out_dir: Path) -> None:
    """Plot the advantage-vs-depth curve for each of our arms (superposition signal)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; advantage curves are in report.json)")
        return
    plt.figure()
    for arm, entry in comparison.items():
        curve = entry["advantage_by_depth"]
        xs = sorted(curve)
        plt.plot(xs, [curve[x] for x in xs], marker="o", label=f"{arm} (slope={entry['slope']:+.3f})")
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("graph depth D")
    plt.ylabel("acc(ours) - acc(baseline)")
    plt.title("Advantage over best-model baseline vs depth")
    plt.legend()
    plt.savefig(out_dir / "advantage.png", dpi=120, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--num-continuous-thoughts", type=int, default=8)
    parser.add_argument(
        "--model",
        default="olmo3_370M",
        help="TransformerConfig factory name; olmo3_370M matches the S3-init runs",
    )
    parser.add_argument(
        "--device", default="auto", help="'auto' (cuda if available else cpu), 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--baseline",
        required=True,
        type=parse_arm_spec,
        metavar="ARM=CKPT",
        help="the best-model normal fine-tune, e.g. A0=runs/latentcot/A0-seed1/model.pt",
    )
    parser.add_argument(
        "--ours",
        action="append",
        default=[],
        type=parse_arm_spec,
        metavar="ARM=CKPT",
        help="repeatable, our latent arm(s), e.g. --ours A2=/path/model.pt",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/latentcot/compare"))
    args = parser.parse_args()

    if not args.ours:
        parser.error("provide at least one --ours ARM=CKPT to compare against the baseline")

    device = resolve_device(args.device)
    model_config = getattr(TransformerConfig, args.model)(
        vocab_size=TOKENIZER_CONFIG.padded_vocab_size()
    )
    examples = load_examples(args.test_data, args.num_continuous_thoughts)

    baseline_arm, baseline_ckpt = args.baseline
    baseline = evaluate_arm(model_config, baseline_arm, baseline_ckpt, examples, device)
    ours = {
        arm: evaluate_arm(model_config, arm, ckpt, examples, device) for arm, ckpt in args.ours
    }

    comparison = {}
    for arm, entry in ours.items():
        curve = gate_a_curve(entry["solve_rate_by_depth"], baseline["solve_rate_by_depth"])
        comparison[arm] = {
            "advantage_by_depth": curve,
            "slope": linear_slope(curve),
            "overall_gap": entry["overall_acc"] - baseline["overall_acc"],
        }

    report = {
        "model": args.model,
        "num_continuous_thoughts": args.num_continuous_thoughts,
        "num_examples": len(examples),
        "baseline": {"arm": baseline_arm, **baseline},
        "ours": ours,
        "comparison": comparison,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print(f"== solve-rate by depth: {baseline_arm} (best-model baseline) vs ours ==")
    print_table(baseline_arm, baseline, ours)
    print("\n== advantage slope vs depth (superposition signal: positive & increasing) ==")
    for arm, entry in comparison.items():
        print(f"  {arm}: slope={entry['slope']:+.4f}  overall_gap={entry['overall_gap']:+.3f}")
    maybe_plot(comparison, args.out)
    print(f"\nWrote {args.out}/report.json")


if __name__ == "__main__":
    main()
