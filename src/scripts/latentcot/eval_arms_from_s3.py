"""
Discover an experiment's arm checkpoints under one prefix, then evaluate the latest of each.

This is :mod:`eval.py` for the case where nobody knows what was actually written. ``eval.py``
needs every arm spelled out as ``--arm A2=/local/path``; this script is handed the run's
checkpoint root, lists what is there, picks the latest checkpoint per arm and evaluates that --
reporting the inventory it found either way.

Three things it does that ``eval.py`` does not, each one a lesson from a dead run:

- **It prints the inventory before it builds anything.** Whether an arm has a checkpoint at all
  is what decides whether the gates are computable, so it is answered in the log within seconds
  rather than after the first arm's forward passes. It is also published before the first model
  is built, so even a crash in construction leaves the one thing the run can always establish.
- **It evaluates arms one at a time and publishes after each.** Four training runs died without
  writing ``metrics.json``, which is written last. Rewriting and mirroring the report after every
  arm means a run killed at the runtime wall still leaves every completed arm's numbers. It also
  keeps one model resident instead of five.
- **It runs the forward under the autocast context the arms trained under**, so eval numerics
  match training numerics rather than approximating them -- and, incidentally, so the ``flash_2``
  backend the ``olmo3_*`` factories hardcode is usable at all, since it accepts only bf16/fp16
  and an fp32 forward through it raises.

Arms are evaluated in the order ``A1, A2, A0, A3, A4`` rather than by name: A1 is nearly free,
A2 with A0 completes gate A, and A3 with A4 completes gate B. So each successive arm finishes a
gate rather than leaving one half-built if the run is cut short.

Usage::

    python src/scripts/latentcot/eval_arms_from_s3.py \\
        --checkpoint-root s3://bucket/teams/<team>/runs/<run-id>/checkpoints/ \\
        --test-data data/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl \\
        --publish-to "$EDULLM_CHECKPOINT_DIR" --out runs/latentcot/eval
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import Example
from olmo_core.latentcot.evaluate import ARM_MODES, assemble_gates, eval_one_arm
from olmo_core.latentcot.inventory import describe_inventory, take_inventory
from olmo_core.latentcot.train_driver import (
    autocast_ctx,
    configure_precision,
    load_checkpoint,
    publish_artifact,
    resolve_device,
)
from olmo_core.nn.transformer import TransformerConfig

# Not alphabetical, deliberately: see the module docstring. Each arm here completes a gate rather
# than leaving one half-built, so a run cut short at the runtime wall loses the least.
ARM_ORDER: Tuple[str, ...] = ("A1", "A2", "A0", "A3", "A4")


def load_examples(path: str, num_continuous_thoughts: int) -> List[dict]:
    """
    Read and encode the held-out test set.

    :param path: Path to a conversations ``.jsonl``.
    :param num_continuous_thoughts: K, which must match what the arms trained under.

    :returns: Encoded examples, as :func:`~olmo_core.latentcot.data.encode.encode_example` makes.
    """
    examples = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            examples.append(
                encode_example(Example.from_dict(json.loads(line)), num_continuous_thoughts)
            )
    return examples


def write_report(report: dict, out_dir: Path, publish_to: Optional[str]) -> Path:
    """
    Write ``report.json`` locally and mirror it to ``publish_to``.

    Called after every arm rather than once at the end: the local file dies with the container,
    so a run killed at the runtime wall should have already published everything it finished.

    :param report: The report dict.
    :param out_dir: Local output directory.
    :param publish_to: Remote URI prefix, or ``None`` for local-only.

    :returns: The local report path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, default=float))
    publish_artifact(path, publish_to)
    return path


def print_summary(report: dict) -> None:
    """
    Print the per-arm table and both gates, saying explicitly when a gate is absent.

    :param report: The report dict.
    """
    print("== per-arm ==")
    for arm, entry in report["per_arm"].items():
        dec = entry.get("decodability")
        dec_s = f" decodability={dec:.3f}" if dec is not None else ""
        print(f"  {arm} ({entry['mode']}): acc={entry['overall_acc']:.3f}{dec_s}")
        by_depth = ", ".join(f"D{d}={v:.3f}" for d, v in entry["solve_rate_by_depth"].items())
        print(f"      {by_depth}")
    if "gate_a" in report:
        print(f"== gate A (superposition) == slope={report['gate_a']['slope']:.4f}")
        print(f"   curve (D -> continuous-discrete): {report['gate_a']['curve']}")
    else:
        print("== gate A == NOT COMPUTED (needs both A2 and A0)")
    print("== gate B (vocab-reg vs L2 control) ==")
    if not report["gate_b"]:
        print("   NOT COMPUTED (needs at least one of A2, A3, A4)")
    for arm, vals in report["gate_b"].items():
        print(f"  {arm}: acc={vals['acc']:.3f} decodability={vals['decodability']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        required=True,
        help="the run's checkpoint root, e.g. s3://bucket/teams/<team>/runs/<run-id>/checkpoints/",
    )
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--num-continuous-thoughts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="the seed naming each arm's leaf dir")
    parser.add_argument(
        "--arms",
        default=",".join(ARM_ORDER),
        help=f"comma-separated arms to look for (default: {','.join(ARM_ORDER)})",
    )
    parser.add_argument("--model", default="olmo3_370M", help="TransformerConfig factory name")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["bf16", "fp32"],
        help="autocast dtype for the forward. bf16 matches what the arms trained under, and is "
        "required by the flash_2 backend the olmo3_* factories hardcode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="generation cap for the explicit_cot arm. The longest teacher CoT in the held-out "
        "set is 56 tokens, so 64 cannot truncate a correct trace, and it halves the worst case "
        "against evaluate.py's default of 128. That arm has no KV cache, so every generated "
        "token costs a full forward and this cap is the biggest single lever on runtime.",
    )
    parser.add_argument(
        "--attn-backend",
        default=None,
        choices=["torch", "flash_2", "flash_3", "flash_4", "te"],
        help="override the attention backend. The olmo3_* factories hardcode flash_2, which "
        "raises at construction without flash-attn installed -- pass 'torch' there. Same "
        "attention math; the 4096 sliding window is a no-op at our ~430-token sequences.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/latentcot/eval"))
    parser.add_argument(
        "--publish-to",
        default=None,
        help='mirror report.json here as each arm finishes, e.g. "$EDULLM_CHECKPOINT_DIR"',
    )
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if unknown := [a for a in arms if a not in ARM_MODES]:
        parser.error(f"unknown arm(s) {unknown}; known arms are {sorted(ARM_MODES)}")

    device = resolve_device(args.device)
    configure_precision(args.precision, device)
    print(f"device={device} precision={args.precision} max_new_tokens={args.max_new_tokens}")

    # Discovery first, and reported before anything expensive is built: whether the CODI arms
    # exist is what decides whether the gates are computable at all. Sorted for the reader; the
    # evaluation order below is deliberately not sorted.
    inventory = take_inventory(args.checkpoint_root, sorted(arms), args.seed)
    print(describe_inventory(inventory), flush=True)
    write_report({"inventory": inventory, **assemble_gates({})}, args.out, args.publish_to)

    evaluated = [arm for arm in ARM_ORDER if inventory.get(arm, {}).get("selected")]
    if not evaluated:
        print("No checkpoint found for any requested arm; nothing to evaluate.")
        return

    model_config = getattr(TransformerConfig, args.model)(
        **({} if args.attn_backend is None else {"attn_backend": args.attn_backend}),
        vocab_size=T.TOKENIZER_CONFIG.padded_vocab_size(),
    )
    examples = load_examples(args.test_data, args.num_continuous_thoughts)
    print(f"loaded {len(examples)} test examples; evaluating {evaluated}", flush=True)

    per_arm: Dict[str, dict] = {}
    for arm in evaluated:
        checkpoint = inventory[arm]["selected_path"]
        print(f"\n--- {arm} ({ARM_MODES[arm]}) <- {checkpoint}", flush=True)
        started = time.monotonic()

        model = model_config.build(init_device="cpu")
        load_checkpoint(model, checkpoint)
        model.to(device)

        with autocast_ctx(args.precision, device):
            per_arm[arm] = eval_one_arm(model, examples, arm, max_new_tokens=args.max_new_tokens)

        # Freed before the next arm is built: one model resident instead of five.
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        elapsed = time.monotonic() - started
        print(f"    acc={per_arm[arm]['overall_acc']:.3f} in {elapsed / 60:.1f} min", flush=True)
        write_report({"inventory": inventory, **assemble_gates(per_arm)}, args.out, args.publish_to)

    report = {"inventory": inventory, **assemble_gates(per_arm)}
    path = write_report(report, args.out, args.publish_to)
    print()
    print_summary(report)
    print(f"\nWrote {path}")
    if args.publish_to:
        print(f"Mirrored to {str(args.publish_to).rstrip('/')}/report.json")


if __name__ == "__main__":
    main()
