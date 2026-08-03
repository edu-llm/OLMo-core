"""
Tests for the fact-crowding experiment under ``src/scripts/train/factcrowd``.

That tree is a package but not an installed one -- ``pyproject.toml`` packages only ``olmo_core*``
-- so ``src/scripts/train`` goes on ``sys.path`` here and its modules import normally afterwards::

    from factcrowd.ladder.rho import solve

The parent directory is what goes on the path, not the package directory, so nothing generic
(``corpus``, ``ladder``, ``train``) becomes importable as a top-level module and able to shadow
something else.

.. note::
    ``src/test/conftest.py`` imports ``torch`` at module scope, so the whole suite needs the full
    install. To run only the modules that do not (see :mod:`factcrowd`)::

        pytest -q --confcutdir=src/test/scripts/factcrowd src/test/scripts/factcrowd
"""

import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[3] / "scripts" / "train"
"""``src/scripts/train`` -- the directory that must be importable for ``factcrowd`` to resolve."""


def _install_import_path() -> None:
    """Put :data:`PROJECT_PARENT` on ``sys.path`` once, idempotently."""
    path = str(PROJECT_PARENT)
    if path not in sys.path:
        sys.path.insert(0, path)


_install_import_path()
