"""Impl 3 × Impl 5 — James's KL-reweighted SFT objective on Impl 5's self-distilled targets.

Impl 3 changes *how much each token counts*. Impl 5 changes *what the tokens are*. Both are
"stay closer to the base model" interventions and neither has been run against the other, so
the open question is whether they compose, are redundant, or interfere.

The whole implementation is: reuse Impl 5's data path unchanged, reuse Impl 4's sampler and
checkpoint grid unchanged, and multiply each pedagogy token's cross-entropy by Impl 3's
``m_t``. See ``klw/weighting.py`` for the objective, ``klw/config_klw.py`` for the arms and
``BUILD.md`` for what was actually run.
"""

from __future__ import annotations

from . import config_klw, paths_klw, weighting
from ._impl5 import IMPL4_ROOT, IMPL5_ROOT, KLW_ROOT, POC_ROOT, chat5, config5, manifest, mixing

__all__ = [
    "config_klw", "paths_klw", "weighting",
    "KLW_ROOT", "POC_ROOT", "IMPL5_ROOT", "IMPL4_ROOT",
    "chat5", "config5", "manifest", "mixing",
]
