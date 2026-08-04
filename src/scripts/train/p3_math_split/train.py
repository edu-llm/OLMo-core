"""Deprecated P3 entrypoint.

The legacy local trainer consumed ``.npy`` mask sidecars and loaded pretrained
weights before the train-module initialization that replaced them. It is retained
only to turn stale commands into an immediate, actionable failure.
"""

from __future__ import annotations

import sys

MESSAGE = (
    "deprecated P3 entrypoint: use "
    "src/scripts/train/p3_math_split/train_platform.py with the canonical "
    "dense.yaml or split.yaml config"
)


def main() -> int:
    """Refuse the obsolete training path before importing training dependencies."""
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
