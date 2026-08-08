#!/usr/bin/env python3
"""Run the fixed OLMo2-190M recipe used as the HPO smoke control."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict

from hpo_on_corpus import _run_configured_segment

from olmo_core.hpo.comparison import (
    COMPARISON_HELDOUT_METRIC,
    DEFAULT_RECIPE_HPS,
    build_comparison_experiment,
)
from olmo_core.hpo.objective import EvaluatorGate
from olmo_core.hpo.worker import WorkerConfig, assert_single_process_topology


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-recipe control for the HPO functional smoke"
    )
    parser.add_argument("run_id", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    parser.add_argument("--target-tokens", type=int, default=1_146_880)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--global-batch-size", type=int, default=32768)
    parser.add_argument("--rank-microbatch-size", type=int, default=4096)
    parser.add_argument("--eval-steps", type=int, default=2)
    parser.add_argument("--data-seed", type=int, default=210007)
    parser.add_argument("--init-seed", type=int, default=110007)
    parser.add_argument("--param-dtype", default="bfloat16")
    parser.add_argument(
        "--checkpoint-root",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    assert_single_process_topology(os.environ)
    if args.target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if args.target_tokens % args.global_batch_size:
        raise ValueError("target_tokens must be divisible by global_batch_size")
    checkpoint_root = args.checkpoint_root
    if not checkpoint_root:
        raise ValueError("the eduLLM platform did not set EDULLM_CHECKPOINT_DIR")

    config = build_comparison_experiment(
        sequence_length=args.sequence_length,
        global_batch_size=args.global_batch_size,
        rank_microbatch_size=args.rank_microbatch_size,
        data_seed=args.data_seed,
        init_seed=args.init_seed,
        eval_steps=args.eval_steps,
    )
    worker = WorkerConfig(
        trial_id="default-recipe",
        gpu=0,
        target_tokens=args.target_tokens,
        quantum=args.target_tokens,
        global_batch_size=args.global_batch_size,
        realized_hps=dict(DEFAULT_RECIPE_HPS),
        checkpoint_root=f"{checkpoint_root.rstrip('/')}/baseline",
        evaluator_gate=EvaluatorGate(
            search_validation="search_validation",
            untouched="final_evaluation",
        ),
    )
    result = _run_configured_segment(
        config=config,
        worker=worker,
        hard_stop_tokens=args.target_tokens,
        heldout_metric=COMPARISON_HELDOUT_METRIC,
        param_dtype=args.param_dtype,
        fidelity={"kind": "exact"},
    )
    payload = asdict(result)
    if not math.isfinite(payload["heldout_ce"]):
        payload["heldout_ce"] = None
    print(
        json.dumps(
            {
                "arm": "default-olmo2-190m",
                "run_id": args.run_id,
                "aggregate_training_tokens": result.tokens,
                "optimizer_steps": (result.tokens // args.global_batch_size),
                **payload,
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
