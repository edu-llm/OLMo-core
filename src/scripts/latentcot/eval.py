"""
Evaluate trained arms and emit the gates + probe report (PRD Phase 6).

Loads each arm's trained checkpoint, runs the held-out test set, and writes a JSON report
(plus printed tables and, if matplotlib is available, a gate-A slope plot) under
``runs/latentcot/eval/``.

Usage::

    .venv/bin/python src/scripts/latentcot/eval.py \
        --test-data data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl \
        --num-continuous-thoughts 10 \
        --arm A0=/path/to/A0/ckpt --arm A2=/path/to/A2/ckpt \
        --arm A3=/path/to/A3/ckpt --arm A4=/path/to/A4/ckpt
"""

import argparse
import json
from pathlib import Path

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import Example
from olmo_core.latentcot.evaluate import run_eval
from olmo_core.latentcot.train_driver import load_checkpoint, resolve_device
from olmo_core.nn.transformer import TransformerConfig


def load_examples(path: str, num_continuous_thoughts: int):
    examples = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            examples.append(
                encode_example(Example.from_dict(json.loads(line)), num_continuous_thoughts)
            )
    return examples


def maybe_plot_gate_a(curve: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not available; gate-A curve is in report.json)")
        return
    xs = sorted(curve)
    plt.figure()
    plt.plot(xs, [curve[x] for x in xs], marker="o")
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.xlabel("graph depth D")
    plt.ylabel("acc(continuous) - acc(discrete)")
    plt.title("Gate A: superposition advantage vs depth")
    plt.savefig(out_dir / "gate_a.png", dpi=120, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--num-continuous-thoughts", type=int, default=10)
    parser.add_argument("--model", default="olmo3_370M", help="TransformerConfig factory name")
    parser.add_argument(
        "--device", default="auto", help="'auto' (cuda if available else cpu), 'cuda', or 'cpu'"
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="ARM=CKPT_DIR",
        help="repeatable, e.g. --arm A2=/path/to/ckpt",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/latentcot/eval"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    model_config = getattr(TransformerConfig, args.model)(
        vocab_size=T.TOKENIZER_CONFIG.padded_vocab_size()
    )
    examples = load_examples(args.test_data, args.num_continuous_thoughts)

    models = {}
    for spec in args.arm:
        arm, ckpt = spec.split("=", 1)
        model = model_config.build(init_device="cpu")
        load_checkpoint(model, ckpt)
        model.to(device)
        models[arm] = model

    report = run_eval(models, examples)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    print("== per-arm ==")
    for arm, entry in report["per_arm"].items():
        dec = entry.get("decodability")
        dec_s = f" decodability={dec:.3f}" if dec is not None else ""
        print(f"  {arm} ({entry['mode']}): acc={entry['overall_acc']:.3f}{dec_s}")
    if "gate_a" in report:
        print(f"== gate A (superposition) == slope={report['gate_a']['slope']:.4f}")
        print(f"   curve (D -> continuous-discrete): {report['gate_a']['curve']}")
        maybe_plot_gate_a(report["gate_a"]["curve"], args.out)
    print("== gate B (vocab-reg vs L2 control) ==")
    for arm, vals in report["gate_b"].items():
        print(f"  {arm}: acc={vals['acc']:.3f} decodability={vals['decodability']}")
    print(f"\nWrote {args.out}/report.json")


if __name__ == "__main__":
    main()
