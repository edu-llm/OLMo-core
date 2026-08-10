#!/usr/bin/env python3
"""Run final OLMo2-370M validation against the staged RegMix manifest."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from entrypoint import (  # noqa: E402
    _refuse_aws_credentials,
    install_local_dataset_reader,
    load_manifest,
)


def _patch_validation_worker_entrypoint(final_validation_module) -> None:
    """Route torchrun workers through this file so the staged reader is installed."""

    original = final_validation_module.torchrun_command
    entrypoint = str(Path(__file__).resolve())

    def torchrun_command(
        vector_name: str,
        length_tokens: int | None,
        global_batch_tokens: int | None = None,
    ) -> list[str]:
        command = original(vector_name, length_tokens, global_batch_tokens)
        script = str(Path(final_validation_module.__file__).resolve())
        return [entrypoint if part == script else part for part in command]

    final_validation_module.torchrun_command = torchrun_command


def main(argv: Sequence[str] | None = None) -> int:
    """Install the staged reader and delegate to the fixed validation entrypoint."""

    _refuse_aws_credentials()
    manifest_path = Path(
        os.environ.get(
            "EDULLM_RUNPOD_INPUT_MANIFEST",
            "/workspace/edullm-inputs/hpo-probe/ready.json",
        )
    )
    install_local_dataset_reader(load_manifest(manifest_path))

    import final_validation

    _patch_validation_worker_entrypoint(final_validation)
    return final_validation.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    raise SystemExit(main())
