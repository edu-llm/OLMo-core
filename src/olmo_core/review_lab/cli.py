from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path
from typing import List, Sequence, cast

from .config import Condition, load_experiment_config
from .data import build_review_data, load_review_data, review_skills
from .report import write_report
from .schedules import fixed_review_steps


log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OLMo 1B review-aware training lab")
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data = subparsers.add_parser("build-data", help="Build the configured review dataset")
    data.add_argument("--force", action="store_true")

    prepare = subparsers.add_parser("prepare", help="Train and save the shared stage-one model")
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Run one stage-two condition")
    run.add_argument("--condition", required=True)
    run.add_argument("--seed", type=int)
    run.add_argument("--force", action="store_true")

    sweep = subparsers.add_parser("sweep", help="Run paired conditions from shared checkpoints")
    sweep.add_argument("--conditions", nargs="+")
    sweep.add_argument("--seeds", nargs="+", type=int)
    sweep.add_argument("--force", action="store_true")

    subparsers.add_parser("summarize", help="Build CSV, plot, and Markdown report")
    subparsers.add_parser("inspect", help="Print resolved config and review schedules")
    subparsers.add_parser("doctor", help="Check CUDA, BF16, and the pinned OLMo config")
    return parser


def _validate_conditions(values: Sequence[str]) -> List[Condition]:
    allowed = {
        "no_review",
        "uniform",
        "expanding",
        "cramming",
        "adaptive_mix",
        "adaptive_due",
        "forever_full",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")
    return cast(List[Condition], list(values))


def _inspect(config) -> None:
    records = load_review_data(config.data) if Path(config.data.path).exists() else []
    payload = {
        "config": config.as_dict(),
        "fingerprint": config.fingerprint(),
        "data_path": config.data.path,
        "records": len(records),
        "skills": list(review_skills(records)) if records else [],
        "uniform_steps": fixed_review_steps(
            "uniform",
            events=config.review.events,
            first_step=config.review.first_step,
            last_step=config.review.last_step,
            expansion_ratio=config.review.expansion_ratio,
        ),
        "expanding_steps": fixed_review_steps(
            "expanding",
            events=config.review.events,
            first_step=config.review.first_step,
            last_step=config.review.last_step,
            expansion_ratio=config.review.expansion_ratio,
        ),
        "cramming_steps": fixed_review_steps(
            "cramming",
            events=config.review.events,
            first_step=config.review.first_step,
            last_step=config.review.last_step,
            expansion_ratio=config.review.expansion_ratio,
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _doctor(config) -> None:
    import torch
    from transformers import AutoConfig

    from .trainer import register_model_backend

    register_model_backend(config)

    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 2**30, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    model_config = AutoConfig.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "devices": devices,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "model_type": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "ready_for_configured_precision": config.model.precision != "bf16"
        or bool(cuda and torch.cuda.is_bf16_supported()),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_experiment_config(args.config)

    if args.command == "build-data":
        print(build_review_data(config.data, force=args.force))
        return

    build_review_data(config.data, force=False)
    if args.command == "inspect":
        _inspect(config)
    elif args.command == "doctor":
        _doctor(config)
    elif args.command == "prepare":
        from .trainer import prepare_stage1

        seeds = [args.seed] if args.seed is not None else config.seeds
        for seed in seeds:
            print(prepare_stage1(config, seed, force=args.force))
    elif args.command == "run":
        from .trainer import run_condition

        condition = _validate_conditions([args.condition])[0]
        seeds = [args.seed] if args.seed is not None else config.seeds
        for seed in seeds:
            print(run_condition(config, condition, seed, force=args.force))
    elif args.command == "sweep":
        from .trainer import prepare_stage1, run_condition, run_root

        conditions = _validate_conditions(args.conditions or config.conditions)
        seeds = args.seeds or config.seeds
        for seed in seeds:
            prepare_stage1(config, seed, force=args.force)
            for condition in conditions:
                print(run_condition(config, condition, seed, force=args.force))
        print(write_report(run_root(config)))
    elif args.command == "summarize":
        from .trainer import run_root

        print(write_report(run_root(config)))


if __name__ == "__main__":
    main()
