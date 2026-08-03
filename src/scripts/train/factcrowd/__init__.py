"""
The fact-crowding experiment: does storing facts consume capacity that would serve reasoning?

``PRD.md`` in this directory is the build spec and the place to start; ``README.md`` is how to run
it.

This is a package rather than a directory of loose scripts, so its modules import each other by
absolute path under ``factcrowd.``::

    from factcrowd.ladder.rho import solve

which needs ``src/scripts/train`` on ``sys.path``. The tests put it there (see
``src/test/scripts/factcrowd/__init__.py``) and the entry points do it for themselves. Nothing
here is installed alongside ``olmo_core``: ``pyproject.toml`` packages only ``olmo_core*``, and
``src/scripts`` is deliberately outside that tree.

**The layering is load-bearing, not cosmetic.** :mod:`factcrowd.ladder.rho` and
:mod:`factcrowd.corpus` hold the arithmetic the experiment's validity rests on, and they do not
import ``torch`` -- so their tests run anywhere, including a laptop with no GPU stack. Modules
under :mod:`factcrowd.train` and :mod:`factcrowd.measure` build models and need the full
install. Keep it that way: an invariant that can only be checked on a GPU node is an invariant
that stops being checked.
"""
