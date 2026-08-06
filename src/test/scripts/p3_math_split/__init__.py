"""Tests for the P3 Math Split experiment under ``src/scripts/train/p3_math_split``.

Those modules are scripts, not an installed package, so they are loaded by path the same way
``src/test/scripts/merge_core_checkpoints_test.py`` loads its script.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROJECT_DIR = Path("src/scripts/train/p3_math_split")
EVALS_DIR = PROJECT_DIR / "evals"
EVAL_MODULES = frozenset({"compare_arms", "export_checkpoint", "run_eval"})


def load_project_module(name: str) -> ModuleType:
    """
    Import one module from the experiment directory by path.

    The directory is also placed on ``sys.path`` because these modules import each other
    (``run_eval`` imports ``mm_verify``, ``build_corpus`` imports ``mm_expand``).

    :param name: Module name without the ``.py`` suffix, e.g. ``"mm_verify"``.

    :returns: The imported module.
    """
    project_root = PROJECT_DIR.resolve()
    project_dir = str(project_root)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    if name in EVAL_MODULES:
        module_path = EVALS_DIR / f"{name}.py"
    else:
        module_path = project_root / f"{name}.py"

    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
