#!/usr/bin/env python3
"""Run quadratic-MTLD curriculum + no-proxy HPs against the staged local manifest."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from entrypoint import _refuse_aws_credentials, install_local_dataset_reader, load_manifest  # noqa: E402


def _patch_worker_entrypoint(module) -> None:
    original = module.torchrun_command
    entrypoint = str(Path(__file__).resolve())

    def torchrun_command(length_tokens: int | None) -> list[str]:
        command = original(length_tokens)
        script = str(Path(module.__file__).resolve())
        return [entrypoint if part == script else part for part in command]

    module.torchrun_command = torchrun_command


def main(argv: Sequence[str] | None = None) -> int:
    _refuse_aws_credentials()
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/hpo-probe/ready.json",
        )
    )
    install_local_dataset_reader(load_manifest(manifest_path))
    import curriculum_noproxy_validation

    _patch_worker_entrypoint(curriculum_noproxy_validation)
    return curriculum_noproxy_validation.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
