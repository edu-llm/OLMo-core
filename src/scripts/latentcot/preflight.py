"""
Pre-registration dry-run (PRD Phase 7): fail fast before spending compute.

Builds the arm configs from one shared base, loads the train/test problems, and runs the
matched-budget + integrity checks (matched config outside the whitelist, same base
checkpoint, disjoint train/test seeds) plus a per-arm compute report (K passes counted).

Usage::

    .venv/bin/python src/scripts/latentcot/preflight.py \
        --train-data local/latentcot/graph-reachability-depth/conversations/train-00000.jsonl \
        --test-data  local/latentcot/graph-reachability-depth/conversations/heldout-00000.jsonl \
        --base-checkpoint /path/to/base/olmo2_370M/ckpt
"""

import argparse
import json
from pathlib import Path

from olmo_core.latentcot.arms import ARMS, build_arm_config
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import Example
from olmo_core.latentcot.preflight import checkpoint_fingerprint, preflight
from olmo_core.latentcot.train_module import CodiTransformerTrainModuleConfig
from olmo_core.optim import WSD, AdamWConfig


def _load(path: str, k: int):
    return [
        encode_example(Example.from_dict(json.loads(line)), k)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--num-continuous-thoughts", type=int, default=8)
    parser.add_argument("--base-checkpoint", default=None, help="base ckpt dir all arms fork")
    parser.add_argument("--rank-microbatch-size", type=int, default=8192)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # One shared base config; arms override only the whitelisted fields. Mirror the values
    # used by the training script so the dry-run reflects the real run.
    base = CodiTransformerTrainModuleConfig(
        rank_microbatch_size=args.rank_microbatch_size,
        max_sequence_length=args.max_sequence_length,
        optim=AdamWConfig(lr=args.lr),
        scheduler=WSD(warmup_steps=200),
        max_grad_norm=1.0,
        num_continuous_thoughts=args.num_continuous_thoughts,
    )
    arm_configs = {name: build_arm_config(base, arm) for name, arm in ARMS.items()}

    train = _load(args.train_data, args.num_continuous_thoughts)
    test = _load(args.test_data, args.num_continuous_thoughts)

    fingerprint = checkpoint_fingerprint(args.base_checkpoint) if args.base_checkpoint else "unset"
    base_checkpoints = {name: fingerprint for name in arm_configs}

    report = preflight(arm_configs, train, test, base_checkpoints)

    print("PREFLIGHT PASSED ✅")
    print(f"  matched config outside whitelist: {report['matched_config']}")
    print(
        f"  same base checkpoint:             {report['same_base_checkpoint']} ({fingerprint[:12]})"
    )
    print(f"  disjoint train/test seeds:        {report['disjoint_seeds']}")
    print(f"  problems: {report['num_train_problems']} train / {report['num_test_problems']} test")
    print("  per-arm forward-token cost (K passes counted):")
    for arm, info in report["per_arm_compute"].items():
        print(f"    {arm} ({info['mode']}): {info['forward_token_cost']:,}")


if __name__ == "__main__":
    main()
