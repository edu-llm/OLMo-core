"""The registered P3 replication seeds.

Kept in its own dependency-free module because three separate stages read it and
they cannot share a heavier home: the trainer, the exporter, and `compare_arms`,
which is pure JSON/statistics and must not acquire a torch import to learn what a
valid seed is.

A seed is a control that "may not drift silently" (P3_DECISION_LEDGER). The
allowlist keeps that property while allowing replicates: an unregistered seed is
still refused at every stage, so adding one here is the deliberate act of
declaring a new replicate rather than a side effect of a typo on a submission
form.
"""

from __future__ import annotations

from typing import Tuple

#: The seed of the original reportable run. Remains the default everywhere.
P3_SEED = 42

#: Every seed a reportable P3 run may use. Dense and split must share one.
P3_SEEDS: Tuple[int, ...] = (42, 43, 44)
