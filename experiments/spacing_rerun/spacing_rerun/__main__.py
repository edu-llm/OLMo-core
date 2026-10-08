from __future__ import annotations

import argparse
import json
import os

# Required by deterministic CUDA matrix products; set before importing torch.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description="FictionalQA spacing rerun: prepare, calibrate, freeze, run, analyze")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="CPU-only pinned download, data audit and exact schedule compilation")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source-directory", help="Optional local pinned parquet cache, validated by expected source schema")
    prepare.add_argument("--audit", help="Human-verified alias/paraphrase/entity audit JSON")
    validate = sub.add_parser("validate", help="CPU-only immutable artifact and schedule checks")
    validate.add_argument("--prepared", required=True)
    for name in ("calibrate", "stage1", "arm"):
        p = sub.add_parser(name)
        p.add_argument("--prepared", required=True)
        p.add_argument("--output", required=True)
        if name in ("stage1", "arm"):
            p.add_argument("--preregistration")
        if name == "stage1":
            p.add_argument("--exposures", type=int, required=True)
        if name == "arm":
            p.add_argument("--stage1-checkpoint", required=True)
            p.add_argument("--arm", choices=("NONE", "UNI", "EXP", "MASS", "GEN"), required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--bundle", action="append", required=True, help="Contains prepared/, stage1/ and five arm directories")
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--primary-delay", type=int, required=True)
    analysis.add_argument("--margin", type=float, default=.02)
    analysis.add_argument("--preregistration")
    power = sub.add_parser("power")
    power.add_argument("--sd", action="append", type=float, required=True)
    power.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare as execute
        manifest = execute(args.config, args.output, args.source_directory, args.audit)
        print(json.dumps({"prepared": args.output, "sha256": manifest["sha256"], "facts": manifest["fact_counts"]}))
    elif args.command == "validate":
        from .prepare import load_prepared
        manifest, schedule, _ = load_prepared(args.prepared)
        print(json.dumps({"validated": manifest["sha256"], "stage2_steps": schedule["stage2_steps"],
                          "confirmation_audit_complete": manifest["audit_complete"]}))
    elif args.command in ("calibrate", "stage1"):
        from .training import run_acquisition
        run_acquisition(args.prepared, args.output,
                        fixed_exposures=args.exposures if args.command == "stage1" else None,
                        preregistration=getattr(args, "preregistration", None))
    elif args.command == "arm":
        from .training import run_arm
        run_arm(args.prepared, args.output, args.stage1_checkpoint, args.arm, preregistration=args.preregistration)
    elif args.command == "analyze":
        from .analysis import summarize_bundles
        summarize_bundles(args.bundle, args.output, args.primary_delay, args.margin, args.preregistration)
    elif args.command == "power":
        from .analysis import power_scenarios
        from .common import write_json
        write_json(args.output, power_scenarios(args.sd))


if __name__ == "__main__":
    main()
