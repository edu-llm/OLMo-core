#!/usr/bin/env python3
"""Resumable one-command P3 corpus rebuild orchestrator.

Stages are recorded in <work-root>/orchestrator-state.json. Re-running the
command skips completed stages unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ARCHIVE_ROOT.parents[1]
STAGES = (
    "bootstrap_sources",
    "materialize_generation_inputs",
    "build_accepted_bases",
    "generation_preflight",
    "generation_build",
    "corpus_verify",
    "tokenize",
    "publish_stage",
    "rebuild_verify",
)


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": "p3-orchestrator-state/v1", "completed": [], "history": []}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_stage(name: str, cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n=== stage {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path(os.environ.get("P3_WORK_ROOT", "/tmp/p3-rebuild-work")),
        help="Persistent working directory for sources, generation, tokenization",
    )
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=None,
        help="Verified upstream sources (defaults to <work-root>/sources)",
    )
    parser.add_argument(
        "--generation-id",
        default="p3-full13-rebuild",
        help="Immutable generation identifier passed to build_p3_generation.py",
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default=STAGES[0],
        help="First stage to run (earlier completed stages are not undone)",
    )
    parser.add_argument(
        "--through-stage",
        choices=STAGES,
        default=STAGES[-1],
        help="Last stage to run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run stages even if already marked complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them",
    )
    args = parser.parse_args()

    work_root = args.work_root.resolve()
    sources_root = (args.sources_root or work_root / "sources").resolve()
    generation_root = work_root / "generation"
    tokenized_root = work_root / "tokenized-v3"
    publish_root = work_root / "publish-stage-v3"
    state_path = work_root / "orchestrator-state.json"
    state = load_state(state_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARCHIVE_ROOT / "scripts")
    env["P3_CORPUS_ROOT"] = str(ARCHIVE_ROOT)
    env["P3_SOURCES_ROOT"] = str(sources_root)

    tokenizer_path = ARCHIVE_ROOT / "tokenizers/qwen25-vendored"
    tokenizer_seal = ARCHIVE_ROOT / "templates/generation-inputs/tokenizer-seal.json"
    policies = ARCHIVE_ROOT / "templates/generation-inputs/policies.json"
    templates_dir = ARCHIVE_ROOT / "templates/generation-inputs"
    generation_inputs = work_root / "generation-inputs"

    stage_index = STAGES.index(args.from_stage)
    through_index = STAGES.index(args.through_stage)
    selected = STAGES[stage_index : through_index + 1]

    commands: dict[str, list[str]] = {
        "bootstrap_sources": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/bootstrap_sources.py"),
            "--root",
            str(sources_root),
            "--build-mizar-index",
        ],
        "materialize_generation_inputs": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/materialize_generation_inputs.py"),
            "--templates",
            str(templates_dir),
            "--out",
            str(generation_inputs),
            "--corpus-root",
            str(ARCHIVE_ROOT),
            "--sources-root",
            str(sources_root),
            "--work-root",
            str(work_root),
        ],
        "build_accepted_bases": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/build_accepted_bases.py"),
            "--sources-root",
            str(sources_root),
            "--work-root",
            str(work_root),
        ],
        "generation_preflight": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/build_p3_generation.py"),
            "--dry-run",
            "--corpus-root",
            str(generation_root),
            "--work-root",
            str(work_root / "generation-work"),
            "--generation-id",
            args.generation_id,
            "--tokenizer-seal",
            str(tokenizer_seal),
            "--tokenizer-path",
            str(tokenizer_path),
            "--policies",
            str(policies),
            "--mizar-semantic-index",
            str(sources_root / "derived/mizar-current-8.1.15.sqlite"),
        ],
        "generation_build": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/build_p3_generation.py"),
            "--corpus-root",
            str(generation_root),
            "--work-root",
            str(work_root / "generation-work"),
            "--generation-id",
            args.generation_id,
            "--tokenizer-seal",
            str(tokenizer_seal),
            "--tokenizer-path",
            str(tokenizer_path),
            "--policies",
            str(policies),
            "--mizar-semantic-index",
            str(sources_root / "derived/mizar-current-8.1.15.sqlite"),
        ],
        "corpus_verify": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/verify_corpus.py"),
            "--corpus",
            str(generation_root),
            "--mizar-semantic-index",
            str(sources_root / "derived/mizar-current-8.1.15.sqlite"),
        ],
        "tokenize": [
            sys.executable,
            str(REPO_ROOT / "src/scripts/train/p3_math_split/tokenize_corpus.py"),
            "--corpus-root",
            str(generation_root),
            "--out-root",
            str(tokenized_root),
            "--tokenizer",
            str(tokenizer_path),
        ],
        "publish_stage": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/stage_v3.py"),
            "--tokenized-root",
            str(tokenized_root),
            "--publish-root",
            str(publish_root),
        ],
        "rebuild_verify": [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/verify_rebuild.py"),
            "--tokenized-root",
            str(tokenized_root),
            "--publish-root",
            str(publish_root),
        ],
    }

    print(f"work root: {work_root}")
    print(f"selected stages: {', '.join(selected)}")
    if not templates_dir.exists():
        raise SystemExit(f"missing generation-input templates: {templates_dir}")

    for stage in selected:
        if stage in state["completed"] and not args.force:
            print(f"skip completed stage {stage}")
            continue
        cmd = commands[stage]
        if stage.startswith("generation") and generation_inputs.exists():
            for manifest in sorted(generation_inputs.glob("*.json")):
                if manifest.name in {"policies.json", "tokenizer-seal.json", "SUMMARY.json"}:
                    continue
                family = manifest.stem
                cmd.extend(["--source-manifest", f"{family}={manifest}"])
        if args.dry_run:
            print(f"DRY-RUN {stage}: {' '.join(cmd)}")
            continue
        run_stage(stage, cmd, cwd=ARCHIVE_ROOT, env=env)
        if stage not in state["completed"]:
            state["completed"].append(stage)
        state["history"].append(
            {
                "stage": stage,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_state(state_path, state)

    if not args.dry_run:
        print("ORCHESTRATOR_OK")


if __name__ == "__main__":
    main()
