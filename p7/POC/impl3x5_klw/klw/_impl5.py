"""Put ``../impl5_ssd`` (and through it ``../impl4_ssd``) on ``sys.path``.

Same reasoning as ``impl5_ssd/impl5/_impl4.py``, one level further down, and for the same
reason it is not convenience: this implementation's arms have to land on the *same*
KL–forgetting plane as Impl 3's 16 runs, Impl 4's A1/A3 and Impl 5's D4. The contrast that
gives these runs their meaning is against **D4**, which differs from them in exactly one
thing — the per-token loss multiplier. Any silent divergence in tokenisation, masking,
block layout or δ assignment would break that without raising anywhere.

So the data path is not reimplemented and not copied. ``mix_arm5.py`` builds the training
file, ``impl5.chat5`` tokenises it, and ``impl4.trainer`` supplies the sampler and the
checkpoint grid. This package adds the loss multiplier and nothing else.

The coupling is pinned by ``tests/test_klw.py`` and asserted loudly by
``acceptance_checks_klw.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

KLW_ROOT = Path(__file__).resolve().parents[1]
POC_ROOT = KLW_ROOT.parent
IMPL5_ROOT = POC_ROOT / "impl5_ssd"
IMPL4_ROOT = POC_ROOT / "impl4_ssd"

if not (IMPL5_ROOT / "impl5" / "__init__.py").exists():   # pragma: no cover
    raise ImportError(
        f"Impl 3x5 reuses Impl 5's data path but {IMPL5_ROOT}/impl5 is not there. It is not "
        f"standalone by design — see klw/_impl5.py."
    )
for root in (IMPL5_ROOT, IMPL4_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from impl5 import chat5, config5, paths5                          # noqa: E402
from impl5._impl4 import chat as chat4                            # noqa: E402
from impl5._impl4 import config4, manifest, mixing                # noqa: E402

__all__ = [
    "KLW_ROOT", "POC_ROOT", "IMPL5_ROOT", "IMPL4_ROOT",
    "chat4", "chat5", "config5", "paths5", "config4", "manifest", "mixing",
]
