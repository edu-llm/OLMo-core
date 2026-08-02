"""Run the repository integration contracts without loading the full test conftest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "test" / "edullm_mixlaw_integration_test.py"
SPEC = importlib.util.spec_from_file_location("edullm_mixlaw_integration_contracts", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

for _name in dir(MODULE):
    if _name.startswith("test_"):
        globals()[_name] = getattr(MODULE, _name)
