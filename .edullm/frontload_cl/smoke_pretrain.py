"""
Frontload-cl GPU smoke helper.

Thin wrapper around ``train_pretrain.py --smoke`` so platform submissions can
name a dedicated entrypoint. Exercises the real 370M config (FlashAttention-2,
``torch.compile``, default A100 microbatch) for 20 steps on the target shape.

Submit with ``olmo-core-train`` on 8×A100 (same compute profile as the full run)::

    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone \\
      .edullm/frontload_cl/smoke_pretrain.py "$EDULLM_RUN_ID" \\
      --arm primer --save-folder "$EDULLM_CHECKPOINT_DIR"'

Success: 20 steps complete, CE loss logged, GPU monitor shows peak memory under
capacity, no flash-attn / Inductor / NCCL errors. Then launch the full primer
and control arms without ``--smoke``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_PARENT = str(_DIR.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# Re-exec through the real trainer with --smoke injected before user args.
# Keeps one implementation of curriculum / hparams / platform contract.
if __name__ == "__main__":
    from frontload_cl import train_pretrain

    argv = sys.argv[1:]
    if "--smoke" not in argv:
        argv = ["--smoke", *argv]
    sys.argv = [sys.argv[0], *argv]
    sys.exit(train_pretrain.cli())
