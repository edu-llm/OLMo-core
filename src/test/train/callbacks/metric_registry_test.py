"""Does every metric name a gate ASSERTS on exist among the names the code EMITS?

This is the test that D-022 exists to make automatic: *a metric is registered when something in
the integrated tree publishes it, not when a lane computes it.*

**The two near-misses that motivated it.** `drop_frac` was required by
:class:`MetricAssertionCallback` while nothing in the tree emitted it -- `ParallelMLP` computed it
every step and no reader existed, so a ratified ceiling was pointed at a name that never appeared.
Separately, two of L2's metric names drifted from the registry and were caught only because someone
re-read an INBOX message. In both cases the failure mode is identical and silent: every ceiling in
the assertion suite is a loop over the metrics dict, and **a loop that finds nothing appends no
failure**, so the run comes back green having verified nothing.

**Why this test cannot itself drift.** Both sides are derived from live code, never from a literal
list:

- the *asserted* names come from ``MetricAssertionCallback``'s own fields
  (``require_present``, plus the names its band checks look for);
- the *emitted* names come from actually calling ``MoERouter.compute_metrics()``.

So there is no third copy of the registry to fall out of date. If someone renames a metric on
either side, this fails.

Runs on CPU: `compute_metrics` is driven off a hand-set histogram, so no Triton dispatch kernel and
no forward pass are involved.
"""

import torch

from olmo_core.nn.moe.router import MoELinearRouter
from olmo_core.train.callbacks import MetricAssertionCallback

# The bare metric names the callback's band checks look for. Kept beside the callback rather than
# inside it because a band is only meaningful with its ceiling, and the ceilings are fields; this
# list is asserted below to be a subset of `_block_metric_names`, so it cannot silently grow stale.
ASSERTED_BAND_METRICS = (
    "dead_expert_frac_global",
    "gate_mass_mean",
    "drop_frac",
    "drop_frac_upper_bound",
    "entropy_deficit",
)


def _emitted_metric_names(*, num_experts: int = 32, top_k: int = 4) -> set:
    """Every bare key `compute_metrics` actually returns, on a realistic capacity-path router."""
    router = MoELinearRouter(
        d_model=64,
        num_experts=num_experts,
        top_k=top_k,
        normalize_expert_weights=1.0,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        init_device="cpu",
    )
    router.train()
    # A real forward, so the conditionally-emitted metrics (gate mass) are populated. Small enough
    # to stay cheap; the routing values do not matter, only which keys appear.
    router(torch.randn(2, 32, 64))
    # Stand in for the capacity path, which is what publishes the drop metrics. Both the exact
    # counts (via the accumulator L3 feeds) and the capacity the upper bound needs.
    router.expert_capacity = 8
    router.accumulate_drop_accounting(torch.tensor(3.0), torch.tensor(256.0))
    return set(router.compute_metrics(reset=False).keys())


def test_every_asserted_metric_is_actually_emitted():
    """The headline: nothing the suite checks may be absent from what the code emits."""
    emitted = _emitted_metric_names()
    callback = MetricAssertionCallback(vocab_size=100352)

    missing = sorted(name for name in ASSERTED_BAND_METRICS if name not in emitted)
    assert not missing, (
        f"{missing} are asserted on by MetricAssertionCallback but are NOT emitted by "
        f"MoERouter.compute_metrics. A ceiling pointed at a name nothing emits iterates the "
        f"metrics dict, finds nothing, appends no failure, and reports GREEN having checked "
        f"nothing. Emitted names are: {sorted(emitted)}"
    )

    missing_required = sorted(name for name in callback.require_present if name not in emitted)
    assert not missing_required, (
        f"require_present names {missing_required} are not emitted. This is the anti-vacuity "
        f"guard's own list, so if it is wrong every band it protects is unprotected. "
        f"Emitted: {sorted(emitted)}"
    )


def test_asserted_metrics_are_registered_for_aggregation():
    """A metric asserted per-block should also be aggregated, or `moe/<name>_max` silently misses."""
    callback = MetricAssertionCallback()
    unregistered = sorted(
        name for name in ASSERTED_BAND_METRICS if name not in callback._block_metric_names
    )
    assert not unregistered, (
        f"{unregistered} are asserted but absent from _block_metric_names, so no "
        f"moe/<name>_{{max,mean,min,n_blocks}} aggregate is produced for them."
    )


def test_aggregated_metrics_are_emitted_or_known_pending():
    """
    Every name registered for aggregation must be emitted, or be a known-pending exception.

    Aggregating a name nothing emits is not fatal -- `_add_cross_block_aggregates` skips empty
    families -- but it means `moe/<name>_max` never appears, so a dashboard or a downstream gate
    keyed on it reads as "no data" rather than "not implemented". The exceptions are listed
    explicitly so that the list is the thing that gets reviewed, rather than the silence.
    """
    # L3's B3 histograms: computed in `parallel_mlp.py` but not yet published, pending per-bucket
    # key names from L3 and `drop_accounting_seq_len` being set (L6's file). Tracked in
    # contracts/telemetry-schema-amendment-2.md. Remove from this list when they land.
    KNOWN_PENDING = {"drop_by_position", "drop_by_doc_index"}

    emitted = _emitted_metric_names()
    callback = MetricAssertionCallback()
    unemitted = {
        name for name in callback._block_metric_names if name not in emitted
    } - KNOWN_PENDING
    assert not unemitted, (
        f"{sorted(unemitted)} are registered for aggregation but not emitted, and are not in the "
        f"known-pending list. Either wire them up or add them to KNOWN_PENDING with a reason."
    )


def test_known_pending_metrics_are_still_pending():
    """
    The reverse guard: once a pending metric IS emitted, this fails so the exception gets removed.

    Without it, `KNOWN_PENDING` becomes a permanent excuse list that outlives the reason for each
    entry -- which is how a temporary exception turns into a metric nobody notices is ungated.
    """
    emitted = _emitted_metric_names()
    now_emitted = sorted({"drop_by_position", "drop_by_doc_index"} & emitted)
    assert not now_emitted, (
        f"{now_emitted} are now emitted, so remove them from KNOWN_PENDING in "
        f"test_aggregated_metrics_are_emitted_or_known_pending and decide whether they should be "
        f"gated (see contracts/telemetry-schema-amendment-2.md -- L3 measured the positional axis "
        f"as real signal, so a flatness band would fire on healthy runs)."
    )


def test_glob_matching_is_anchored_and_does_not_leak():
    """
    `drop_frac` must not match `drop_frac_upper_bound`, and a `moe/` aggregate must not re-match.

    Both are one-character-away mistakes that would make a band assert on the wrong quantity: the
    prefix leak would apply the exact-rate ceiling to the upper bound (double-counting), and the
    aggregate leak would re-assert a cross-block max as though it were a per-block value.
    """
    is_block_key = MetricAssertionCallback._is_block_key

    assert is_block_key("train/block 03/drop_frac", "drop_frac")
    assert is_block_key("train/block 11/dead_expert_frac_global", "dead_expert_frac_global")
    # The prefix trap.
    assert not is_block_key("train/block 03/drop_frac_upper_bound", "drop_frac")
    # The local-vs-global trap: these are DIFFERENT metrics and the band is on the global one.
    assert not is_block_key("train/block 03/dead_expert_frac", "dead_expert_frac_global")
    assert not is_block_key("train/block 03/dead_expert_frac_global", "dead_expert_frac")
    # The aggregate trap.
    assert not is_block_key("moe/drop_frac_max", "drop_frac")
    # A non-block metric.
    assert not is_block_key("train/CE loss", "drop_frac")


def test_contract_registered_lbl_pair_is_emitted():
    """
    The flat `moe/lbl_local` / `moe/lbl_global` pair the telemetry contract registers must exist.

    G4's stated pass condition is a comparison of these two on the same batch, to prove the DP
    all-reduce actually reduces. Until B1 landed they did not exist in the tree at all, so the gate
    could not have been evaluated -- exactly the class of gap D-022 is about.
    """
    emitted = _emitted_metric_names()
    for name in ("moe/lbl_local", "moe/lbl_global", "lbl_local", "lbl_global"):
        assert name in emitted, (
            f"'{name}' is registered in telemetry-schema.md but not emitted. G4 compares the pair "
            f"on one batch to prove the all-reduce reduces; it cannot compare what is absent. "
            f"Emitted: {sorted(emitted)}"
        )
