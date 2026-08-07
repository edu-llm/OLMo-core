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

import pytest
import torch

from olmo_core.nn.moe.router import MoELinearRouter
from olmo_core.train.callbacks import MetricAssertionCallback, MetricAssertionError

# The bare metric names the callback's band checks look for. Kept beside the callback rather than
# inside it because a band is only meaningful with its ceiling, and the ceilings are fields; this
# list is asserted below to be a subset of `_block_metric_names`, so it cannot silently grow stale.
# EVERY per-block metric the RATIFIED CONTRACT registers, transcribed from
# `maple/agents/contracts/telemetry-schema.md` and its ratified amendments.
#
# THIS LIST EXISTS BECAUSE THE CONTRACT'S REGISTRY LIVES IN PROSE AND CODE CANNOT READ PROSE.
# `effective_capacity_factor` was registered in `telemetry-schema.md` -- explicitly, as "required
# for X3", with the measured 1.2188-vs-1.2031 justification -- and computed correctly in
# `parallel_mlp.py`, and published by nothing. Three lanes each did their part and the metric did
# not exist in any run. The registry test could not catch it because the test only knew the names
# the assertion callback happened to reference, and nothing referenced this one.
#
# So: a name registered in the contract must appear here, and this list is checked against what the
# code actually emits. That converts "registered in a document somebody has to remember to read"
# into a failing test. It is the closest a code-level check can get to enforcing a prose registry,
# and it is the answer to whether "computed but not published" is closeable: **it is, but only for
# names the test is told to expect** -- so the telling has to be mechanical.
CONTRACT_REGISTERED_BLOCK_METRICS = (
    # telemetry-schema.md, ratified registry
    "expert_load_cv",
    "entropy_deficit",
    "dead_expert_frac",
    "drop_frac",
    "effective_capacity_factor",
    # amendment 1
    "expert_load_cv_excess",
    "gate_mass_mean",
    "assignments_per_expert_mean",
    # amendment 3
    "dead_expert_frac_global",
    "dead_expert_counts_not_reduced",
    # amendment 4 (the B3 histograms travel per-bucket; see the dedicated test)
    "drop_histograms_unavailable",
    # amendment 5 (D-075). The window-free companion to `expert_load_cv_excess`. Registered
    # because it is the E-comparable cross-rung READOUT; the GATE is raw `expert_load_cv`, which
    # needs no null. Both are required: one is comparable across E, the other is gateable.
    "expert_load_cv_excess_per_micro_batch",
)

# Registered per-block series that travel as one scalar per bucket rather than under their bare
# name, because `record_metric` takes a scalar. Checked by `test_b3_histograms_are_emitted_per_bucket`
# rather than by bare-name lookup -- listed here so the two tests cannot silently disagree about
# which names are exempt.
CONTRACT_REGISTERED_BUCKETED_SERIES = ("drop_by_position", "drop_by_doc_index")

ASSERTED_BAND_METRICS = (
    "dead_expert_frac_global",
    "gate_mass_mean",
    "drop_frac",
    "drop_frac_upper_bound",
    "entropy_deficit",
    "capacity_factor_deficit",
    # D-075. RAW CV, and `expert_load_cv_excess` is deliberately NOT here: it is
    # `CV * sqrt(T*N*k/(E-1))`, so gating it lets a longer logging interval alone fail a physically
    # identical model. `test_no_band_gates_a_window_dependent_statistic` pins that exclusion, so it
    # cannot be reintroduced by someone reading this list as a menu.
    "expert_load_cv",
)


def _emitted_metric_names(
    *,
    num_experts: int = 32,
    top_k: int = 4,
    num_position_buckets: int = 16,
    num_doc_buckets: int = 4,
) -> set:
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
    # The capacity pair, as `MoE.forward` hands it over. 2.0 configured and 2.0 realized is the
    # funded path, where quantization vanishes exactly.
    router.effective_capacity_factor = 2.0
    router.configured_capacity_factor = 2.0
    # And the B3 histograms. Deliberately different lengths, because they really are:
    # `drop_by_doc_index` is capped at one bucket per document, so at a small micro-batch it is
    # shorter than `drop_by_position`. A test that used equal lengths would not catch a helper that
    # assumed a single shared bucket count.
    router.accumulate_drop_histograms(
        torch.full((num_position_buckets,), 0.01),
        torch.full((num_doc_buckets,), 0.02),
        instrumented=True,
    )
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


def test_every_contract_registered_metric_is_emitted():
    """
    Every per-block metric the ratified contract registers must exist in what the code emits.

    THIS IS THE TEST THAT WOULD HAVE CAUGHT `effective_capacity_factor`. It was registered in
    `telemetry-schema.md` as "required for X3", computed correctly in `parallel_mlp.py`, and
    published by nothing -- the third instance of the same seam after `drop_frac` and the B3
    histograms. The earlier registry test could not see it, because that test only knew names the
    assertion callback referenced, and no band referenced this one.

    The distinction from `test_every_asserted_metric_is_actually_emitted` is the point: that test
    protects metrics a GATE reads. This one protects metrics the CONTRACT promises, which is a
    strictly larger set -- a metric can be required for an experiment (X3) without any band gating
    it.
    """
    emitted = _emitted_metric_names()
    missing = sorted(m for m in CONTRACT_REGISTERED_BLOCK_METRICS if m not in emitted)
    assert not missing, (
        f"{missing} are registered in telemetry-schema.md (or a ratified amendment) but are NOT "
        f"emitted by MoERouter.compute_metrics. A metric is registered when something in the "
        f"integrated tree PUBLISHES it, not when a lane computes it (D-022). If one of these is "
        f"deliberately retired, remove it from CONTRACT_REGISTERED_BLOCK_METRICS and from the "
        f"contract in the same change. Emitted: {sorted(emitted)}"
    )

    # The bucketed series are exempt from bare-name lookup, but must not be silently forgotten:
    # they are covered by the per-bucket test, and this asserts the exemption list stays honest.
    for series in CONTRACT_REGISTERED_BUCKETED_SERIES:
        assert any(k.startswith(f"{series}_b") for k in emitted), (
            f"'{series}' is registered and is exempt from bare-name lookup because it travels "
            f"per-bucket -- but no '{series}_b<NN>' keys are emitted either, so it is simply absent."
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


def test_b3_histograms_are_emitted_per_bucket():
    """
    The B3 histograms must reach the metrics as one scalar per bucket.

    THIS IS THE REGRESSION TEST FOR THE G2 FAILURE. `drop_by_position` and `drop_by_doc_index` were
    computed on every step of that run and appeared in none of its 388 W&B columns, because
    `record_metric` takes a scalar and `reduce_metrics` stacks every metric into one flat tensor --
    a `(num_buckets,)` tensor cannot travel that path. So the gate could not distinguish "positional
    drops are healthy" from "positional drops were never measured", on a run that had already been
    paid for.
    """
    emitted = _emitted_metric_names(num_position_buckets=16, num_doc_buckets=4)

    # Bucket keys, at the lengths actually handed over -- note the two series differ, which is real.
    for i in range(16):
        assert f"drop_by_position_b{i:02d}" in emitted, f"missing positional bucket {i}"
    for i in range(4):
        assert f"drop_by_doc_index_b{i:02d}" in emitted, f"missing document bucket {i}"

    # And NOT keys past the end of each series: emitting a bucket that does not exist is as
    # misleading as omitting one that does.
    assert "drop_by_doc_index_b04" not in emitted, (
        "a bucket key was emitted past the end of the document series -- the bucket count must be "
        "read from the tensor, not assumed to be DEFAULT_NUM_BUCKETS"
    )

    # The bucket count itself, so a 4-bucket axis cannot be confused with a 16-bucket axis whose
    # last 12 buckets went missing.
    assert "drop_by_position_num_buckets" in emitted
    assert "drop_by_doc_index_num_buckets" in emitted

    # Zero-padded so the keys sort into axis order: `_b02` sorts before `_b10`.
    ordered = sorted(k for k in emitted if k.startswith("drop_by_position_b"))
    assert (
        ordered[2] == "drop_by_position_b02" and ordered[10] == "drop_by_position_b10"
    ), f"bucket keys do not sort into axis order: {ordered[:12]}"


def test_instrumentation_presence_signal_is_emitted_and_gated_by_convention():
    """
    A component that knows it was not instrumented must say so in a NUMBER, and that number gated.

    The G2 lesson, structurally: the presence check protected exactly the three names on its list
    and was silent about the histograms. A list has a blind spot shaped like the list. So the fix is
    a convention -- any `<name>_unavailable` metric is gated automatically -- and this test pins
    both halves: that the signal is emitted, and that the callback gates the suffix rather than a
    hard-coded name.
    """
    emitted = _emitted_metric_names()
    assert "drop_histograms_unavailable" in emitted, (
        "the B3 presence signal is not emitted, so 'never instrumented' and 'measured healthy' are "
        "once again indistinguishable"
    )

    callback = MetricAssertionCallback()
    assert callback.assert_instrumented, "the instrumentation check must be on by default"
    assert "drop_histograms_unavailable" in callback.require_present, (
        "the presence signal must itself be required -- otherwise a build that stopped emitting it "
        "would leave the suffix convention with nothing to match, which is the same failure one "
        "level up"
    )

    # The gate is the SUFFIX, not the name: a hypothetical future signal is covered with no edit.
    assert callback.unavailable_metric_suffix == "_unavailable"

    metrics = {
        "train/CE loss": 4.4,
        "train/block 00/dead_expert_frac_global": 0.0,
        "train/block 00/gate_mass_mean": 1.0,
        "train/block 00/drop_frac": 0.0,
        "train/block 00/drop_histograms_unavailable": 0.0,
    }
    cb = MetricAssertionCallback(vocab_size=100352)
    cb._checked_step0 = True  # mid-run state; the step-0 band is not under test here
    cb.pre_log_metrics(60, dict(metrics))  # healthy: must not raise

    # Now flip the signal. A future instrument nobody has written yet is covered by the same rule.
    for offender in ("drop_histograms_unavailable", "some_future_instrument_unavailable"):
        bad = dict(metrics)
        bad[f"train/block 03/{offender}"] = 1.0
        cb2 = MetricAssertionCallback(vocab_size=100352)
        cb2._checked_step0 = True
        try:
            cb2.pre_log_metrics(60, bad)
        except MetricAssertionError as err:
            assert offender in str(err), f"the failure must name the offending instrument: {err}"
        else:
            raise AssertionError(
                f"'{offender}' = 1.0 did not raise. An instrument reporting its own absence must "
                "fail loudly; that is the whole point of the B3 fix."
            )


def test_not_reduced_signals_are_NOT_gated_by_the_unavailable_convention():
    """
    `..._not_reduced` must stay ungated: on one rank it is legitimately 1.0.

    Gating it would fire on every single-GPU gate, which is the "assertion that fails on healthy
    runs" failure this suite has now hit twice (the flat-histogram band, the per-rank dead-expert
    band). "Not reduced" is a correct state; "unavailable" is not.
    """
    cb = MetricAssertionCallback(vocab_size=100352)
    cb._checked_step0 = True
    cb.pre_log_metrics(
        60,
        {
            "train/CE loss": 4.4,
            "train/block 00/dead_expert_frac_global": 0.0,
            "train/block 00/gate_mass_mean": 1.0,
            "train/block 00/drop_frac": 0.0,
            "train/block 00/drop_histograms_unavailable": 0.0,
            # Single-rank run: both of these are legitimately 1.0 and must not raise.
            "train/block 00/dead_expert_counts_not_reduced": 1.0,
            "train/block 00/lbl_not_reduced": 1.0,
        },
    )


def test_capacity_factor_assertion_is_ge_and_not_equality():
    """
    `effective >= configured` must pass; only a genuine shortfall may fail.

    `ensure_multiple_of` rounds capacity UP, so the realized factor legitimately EXCEEDS the
    configured one -- L3 measured a requested 1.2 realizing as 1.2188 at E=256/8192 and 1.2031 at
    16384. An `== configured` gate would fire on that healthy run. On the funded factor 2.0 the
    quantization vanishes exactly, which is why equality *looks* safe right up until a rung or batch
    size changes: D-019's failure mode hiding inside this metric.
    """
    base = {
        "train/CE loss": 4.4,
        "train/block 00/dead_expert_frac_global": 0.0,
        "train/block 00/gate_mass_mean": 1.0,
        "train/block 00/drop_frac": 0.0,
        "train/block 00/drop_histograms_unavailable": 0.0,
    }

    def check(deficit):
        cb = MetricAssertionCallback(vocab_size=100352)
        cb._checked_step0 = True
        metrics = dict(base)
        metrics["train/block 00/capacity_factor_deficit"] = deficit
        try:
            cb.pre_log_metrics(60, metrics)
            return None
        except MetricAssertionError as err:
            return str(err)

    # Funded path: realized == configured, deficit exactly 0. Must pass.
    assert check(0.0) is None, "the funded path (deficit 0.0) must not raise"

    # Realized ABOVE configured -- the rounding case. The deficit metric clamps at 0, so this is
    # the same 0.0 input; asserted explicitly because it is the case an equality gate would fail.
    assert check(0.0) is None

    # A real shortfall: the dispatch allocated less than requested. Rounding up cannot do this.
    err = check(0.05)
    assert err is not None, (
        "a positive capacity deficit must raise -- rounding UP cannot produce one, so it means the "
        "capacity path is not doing what the config says"
    )
    assert "capacity_factor_deficit" in err
    assert "effective >= configured" in err, (
        "the failure message must state the correct assertion form, or whoever reads it at 3am "
        "will 'fix' it into an equality"
    )


def test_capacity_factor_pair_is_emitted_together():
    """Both sides of the inequality must be published, or a gate cannot evaluate it."""
    emitted = _emitted_metric_names()
    for name in (
        "effective_capacity_factor",
        "configured_capacity_factor",
        "capacity_factor_deficit",
    ):
        assert name in emitted, (
            f"'{name}' is missing. The assertion is `effective >= configured`; publishing only one "
            f"side leaves a reader comparing against a config value they hope was in force."
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


# ---------------------------------------------------------------------------------------------
# D-075 -- THE ACCUMULATION-LENGTH DEFECT.
#
# `expert_load_cv_excess = cv / sqrt((1-1/E)/mean_acc)`, and `mean_acc` is the mean of the
# ACCUMULATED histogram. Accumulating T micro-batches multiplies `mean_acc` by T, so the null
# shrinks as 1/sqrt(T) while the observed CV does not -- per-micro-batch imbalance is correlated
# across micro-batches, so accumulated CV converges to the persistent imbalance instead. Hence
#
#       cv_excess = CV * sqrt(T * N * k / (E - 1))
#
# which GROWS WITHOUT BOUND in the logging-interval length at fixed physical balance. Measured: on
# X1 (E=64, mean_acc = 98,304) `cv_excess / expert_load_cv` was exactly 316.013 at every one of 600
# logged steps, and 316.01 again on X4a control at step 284 -- a different run and a different arm.
#
# THE OPERATIONAL HAZARD, which is what these tests exist to prevent: `MetricAssertionCallback`
# assertions RAISE. A band on `cv_excess` therefore means A LONGER LOGGING INTERVAL ALONE CAN FAIL A
# PHYSICALLY IDENTICAL MODEL, and the 20B run has long intervals by construction. Fail-closed.
#
# So the gated quantity is RAW `expert_load_cv`, and the tests below pin all three halves of that:
# that the gated quantity does not move with the window, that the diagnostic one DOES (so the tests
# are not passing because the mechanism vanished), and that no band references it.
# ---------------------------------------------------------------------------------------------

_CV_BAND_METRICS = ("expert_load_cv", "expert_load_cv_excess_per_micro_batch")


def _metrics_at_accumulation_length(
    *,
    num_micro_batches: int,
    num_experts: int = 64,
    per_micro_batch_counts=(30.0, 20.0, 14.0),
) -> dict:
    """
    Drive `compute_metrics` at a given accumulation length holding PHYSICAL BALANCE FIXED.

    The construction is the whole point of the test, so it is spelled out rather than left to be
    inferred: the accumulated histogram is one fixed per-micro-batch shape multiplied by
    `num_micro_batches`. That means the RATIO of expert loads -- which is what "physical balance"
    means, and what raw CV measures, since CV is invariant under scaling all counts by a constant --
    is IDENTICAL at every accumulation length, while the accumulated mean scales with the window.

    Deliberately NOT modelled as independent multinomial draws per micro-batch. That would make the
    accumulated CV shrink as 1/sqrt(T) for real, and then `cv_excess` would legitimately be
    window-invariant and there would be no bug. The defect is real precisely because router
    imbalance is PERSISTENT -- the same experts are hot on every step -- and this fixture encodes
    that persistence rather than assuming it away.
    """
    router = MoELinearRouter(
        d_model=64,
        num_experts=num_experts,
        top_k=4,
        normalize_expert_weights=1.0,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        init_device="cpu",
    )
    router.train()

    # A fixed uneven per-micro-batch shape: the first three experts hot, the rest uniform.
    base = torch.full((num_experts,), 10.0)
    for i, count in enumerate(per_micro_batch_counts):
        base[i] = count

    router.batch_size_per_expert = base * float(num_micro_batches)
    router.num_accumulated_micro_batches = num_micro_batches
    return {k: v[0].item() for k, v in router.compute_metrics(reset=False).items()}


def test_gated_cv_band_metric_is_invariant_to_accumulation_length():
    """
    THE TEST THAT WOULD HAVE CAUGHT D-075. Same physical balance, two windows, gated value fixed.

    If this fails, the fix is wrong: a metric a RAISING assertion gates must not depend on how often
    the trainer happens to log.
    """
    short = _metrics_at_accumulation_length(num_micro_batches=1)
    long = _metrics_at_accumulation_length(num_micro_batches=24)

    for name in _CV_BAND_METRICS:
        assert name in short and name in long, f"'{name}' is not emitted at both windows"
        assert short[name] == pytest.approx(long[name], rel=1e-6), (
            f"'{name}' moved from {short[name]:.6f} at 1 micro-batch to {long[name]:.6f} at 24, on "
            f"IDENTICAL physical balance. Ratio {long[name] / short[name]:.4f} vs the sqrt(24) = "
            f"{24 ** 0.5:.4f} signature of the accumulation-length defect. A gated quantity that "
            f"moves with the logging interval means a longer interval alone fails a physically "
            f"identical model, and these assertions RAISE (D-075)."
        )


def test_cv_excess_is_still_window_dependent_so_the_test_above_is_not_vacuous():
    """
    The defect must still be DEMONSTRABLE, or the test above passes for the wrong reason.

    A test whose pass does not depend on the behaviour is not a test. If `cv_excess` were somehow
    made window-invariant too, `test_gated_cv_band_metric_is_invariant_to_accumulation_length` would
    pass trivially and prove nothing about the choice of gated metric. So this pins the mechanism:
    `cv_excess` grows as exactly sqrt(T), and that is why it is not the gated quantity.
    """
    short = _metrics_at_accumulation_length(num_micro_batches=1)
    long = _metrics_at_accumulation_length(num_micro_batches=24)

    ratio = long["expert_load_cv_excess"] / short["expert_load_cv_excess"]
    assert ratio == pytest.approx(24**0.5, rel=1e-5), (
        f"expert_load_cv_excess grew by {ratio:.4f} over a 24x window, expected sqrt(24) = "
        f"{24 ** 0.5:.4f}. If this is now ~1.0 the metric was silently changed and the band choice "
        f"should be revisited; if it is some other factor, the model of the defect is wrong."
    )

    # And the identity inverts to the ACCUMULATED MEAN, exactly. This is the check that makes the
    # closed form auditable rather than asserted: `cv_excess/cv = 1/cv_null = sqrt(mean_acc/(1-1/E))`,
    # so `mean_acc = (cv_excess/cv)^2 * (1-1/E)`. On X1 that reads 316.013^2 * (1-1/64) = 98,303.8,
    # recovering the true 98,304 = 2048 assignments x 48 micro-batches.
    #
    # `> 0` was the first version of this assertion and it CANNOT FAIL -- a vacuous assert inside a
    # test whose whole subject is non-vacuity. Pinned to the fixture's real accumulated mean now.
    for metrics, t in ((short, 1), (long, 24)):
        ratio = metrics["expert_load_cv_excess"] / metrics["expert_load_cv"]
        implied_mean_acc = ratio**2 * (1.0 - 1.0 / 64)
        expected_mean_acc = metrics["assignments_per_expert_mean"] * t
        assert implied_mean_acc == pytest.approx(expected_mean_acc, rel=1e-4), (
            f"inverting cv_excess/cv gave mean_acc {implied_mean_acc:.2f}, but the accumulated "
            f"histogram's own mean is {expected_mean_acc:.2f}. The closed form documented in "
            f"router.py does not describe the code."
        )


def test_the_two_excess_forms_differ_by_exactly_sqrt_of_the_window():
    """`cv_excess_per_micro_batch = cv_excess / sqrt(T)`, which is the entire content of the fix."""
    for t in (1, 5, 24):
        metrics = _metrics_at_accumulation_length(num_micro_batches=t)
        assert metrics["expert_load_cv_excess_per_micro_batch"] == pytest.approx(
            metrics["expert_load_cv_excess"] / (t**0.5), rel=1e-6
        ), f"the two excess forms do not differ by sqrt({t})"


def test_no_band_gates_a_window_dependent_statistic():
    """
    NO BAND MAY REFERENCE `expert_load_cv_excess`. Structural, not a comment.

    `ASSERTED_BAND_METRICS` reads like a menu, and the excess form is the one with "compare across
    rungs" in its history -- so the reason it is absent has to be enforced rather than documented.
    This is the reverse guard: the other tests check that everything asserted is emitted; this
    checks that something emitted is NOT asserted, on purpose.
    """
    assert "expert_load_cv_excess" not in ASSERTED_BAND_METRICS, (
        "a band is gating `expert_load_cv_excess`, which is CV*sqrt(T*N*k/(E-1)). A longer logging "
        "interval alone would then fail a physically identical model. Gate `expert_load_cv` "
        "instead, and read `expert_load_cv_excess_per_micro_batch` for the E-normalised magnitude "
        "(D-075)."
    )
    assert "expert_load_cv_excess_per_micro_batch" not in ASSERTED_BAND_METRICS, (
        "the window-free excess is a READOUT, not a gate: its value still depends on E and on the "
        "micro-batch size, so a single ceiling across the rung ladder would not mean one thing. The "
        "gate is raw CV, whose ceiling is set from measurements at this ladder's own assignment "
        "counts."
    )

    # Both forms must still be AGGREGATED, or `moe/<name>_max` silently misses and the diagnostic is
    # unreadable at exactly the moment someone needs it.
    callback = MetricAssertionCallback()
    for name in ("expert_load_cv_excess", "expert_load_cv_excess_per_micro_batch"):
        assert name in callback._block_metric_names, (
            f"'{name}' is not aggregated, so no moe/{name}_{{max,mean,min}} exists. Ungated is not "
            f"the same as unread."
        )


def test_raw_cv_band_fires_above_its_ceiling_and_not_below():
    """
    A magnitude check on the band, pinned to the project's PRE-REGISTERED escalation threshold.

    `maple/plan/balance-floor.md` fixes a decision rule on **block-max raw CV** before the P1 probe:
    <=0.40 retire balance as a risk; (0.40, 0.55] adopt capacity_factor 2.5; >0.55 escalate because
    balance is architectural. A RAISING assertion belongs at the ESCALATE boundary and nowhere
    tighter -- inside the middle band the pre-registered response is "change one config value and
    keep going", so a band that raised there would kill a run the plan says to continue.

    THIS TEST'S BOUNDS ARE THE FIX FOR TWO REAL ERRORS, and they are asserted rather than commented
    so neither can come back:

    1. The ceiling was first justified against "published baselines" 0.72 / 0.52 / 0.937 from Wang
       et al. 2408.15664 Table 2. **Those are MaxVio = max/mean - 1, not CV** -- balance-floor.md
       converts X1 to MaxVio 0.683 before comparing, precisely because the units differ. Almost
       nobody publishes expert-load CV, so there is no external CV referent and the calibration must
       come from our own pre-registered rule.
    2. The healthy figures 0.2853 / 0.3064 / 0.3370 are **block-00 or cross-block-mean**, and this
       band gates **block-max**. X1's spread was 0.1755-0.3512 across 8 blocks, so the binding
       healthy number is **0.3512** -- which is ABOVE 0.3370, i.e. an earlier version of this test
       enforced a floor that X1's own healthy block-max violated.
    """
    callback = MetricAssertionCallback(vocab_size=100352)
    ceiling = callback.expert_load_cv_max
    assert ceiling is not None, "the raw-CV ceiling must be armed by default"

    # Healthy, on the block-max reading the band actually evaluates. 0.3512 is X1's measured
    # worst block; the other two are block-00 figures whose block-max was never recorded.
    for healthy in (0.2853, 0.3064, 0.3370, 0.3512):
        callback._failures = []
        callback._check_cv_band({"train/block 00/expert_load_cv": healthy})
        assert not callback._failures, (
            f"the ceiling {ceiling} fires on {healthy}, a MEASURED healthy value on this stack "
            f"(0.3512 is X1's worst BLOCK, which is the reading this band gates)."
        )

    # The pre-registered ADOPT band must NOT raise: the plan's response there is capacity_factor
    # 2.5, one config change. A band firing inside it kills a run the plan says to continue.
    for adopt_band in (0.41, 0.50, 0.55):
        callback._failures = []
        callback._check_cv_band({"train/block 00/expert_load_cv": adopt_band})
        assert not callback._failures, (
            f"the ceiling {ceiling} fires at block-max CV {adopt_band}, which is inside the "
            f"pre-registered (0.40, 0.55] ADOPT band where balance-floor.md prescribes "
            f"capacity_factor 2.5 rather than failure."
        )

    # And it must fire above the ESCALATE boundary, or it is not a band.
    for escalate in (0.5501, 0.60, 0.7500, 1.2434):
        callback._failures = []
        callback._check_cv_band({"train/block 00/expert_load_cv": escalate})
        assert callback._failures, f"the ceiling {ceiling} does not fire on {escalate}"

    assert ceiling == pytest.approx(0.55), (
        f"the ceiling is {ceiling}, not the pre-registered ESCALATE boundary of 0.55 "
        f"(maple/plan/balance-floor.md). Tighter is NOT safer here: 0.50 sits inside the ADOPT "
        f"band and would hard-fail runs the plan says to continue with capacity_factor 2.5. Looser "
        f"than 0.55 abandons the only threshold this project committed to in advance. Do NOT "
        f"re-derive this from Wang et al. Table 2 -- those figures are MaxVio, not CV."
    )
    # It must also clear the worst measured healthy block, or it fires on a healthy run (D-019).
    assert ceiling > 0.3512


def test_cv_band_has_a_later_window_and_reports_it_separately():
    """
    The CV band's window must outlast the drop transient, and its coverage be reported on its own.

    Measured on a run that finished 600/600 clean: `drop_frac` was 0.0021 at step 200 while raw CV
    was 0.4618 and still falling to a 0.2853 tail near step 590. So a CV ceiling sharing
    `warmup_steps=50` would have failed a healthy run twice (1.2434 at step 50, 0.7500 at step 100).

    And the separate count is not cosmetic: folded into `_bands_applied`, a 250-step run would
    report "bands applied on 200 steps" -- true, and silent about this band having run zero times.
    That is D-048's lesson, where the blind spot is the DEFINITION of the coverage.
    """
    callback = MetricAssertionCallback(vocab_size=100352)
    assert callback.expert_load_cv_warmup_steps > callback.warmup_steps, (
        "the CV band's window must be later than the shared one: the balance transient outlasts the "
        "drop transient by hundreds of steps on measured data."
    )
    # It must clear the step-200 reading of a healthy run, which is 0.4618 -- inside the ceiling but
    # only by 1.19x, which is inside the spread of the three healthy measurements.
    assert callback.expert_load_cv_warmup_steps >= 300

    # The two counters are independent. `presence_check_step` is pushed out of reach because this
    # test is about the WINDOWS, and a partial metrics dict would otherwise fail the anti-vacuity
    # guard for an unrelated reason -- which would make the test pass or fail for the wrong cause.
    windows = MetricAssertionCallback(
        vocab_size=100352, assert_step0_loss=False, presence_check_step=10**9
    )
    windows._assert_all(60, {"train/block 00/expert_load_cv": 0.30})
    assert windows._bands_applied == 1
    assert windows._cv_band_applied == 0, (
        "the CV band ran inside its own warmup window, so a healthy recovery transient would fail "
        "the run"
    )
    windows._assert_all(400, {"train/block 00/expert_load_cv": 0.30})
    assert windows._cv_band_applied == 1

    # And it would actually FIRE past its window -- a window test that never exercises the band
    # proves only that the counter increments.
    fires = MetricAssertionCallback(
        vocab_size=100352, assert_step0_loss=False, presence_check_step=10**9
    )
    with pytest.raises(MetricAssertionError, match="expert-load CV ceiling"):
        fires._assert_all(400, {"train/block 00/expert_load_cv": 1.2434})
    # ...and NOT inside the window, on the identical value. This is the half that matters: the
    # measured step-50 reading of a healthy run is exactly this number.
    early = MetricAssertionCallback(
        vocab_size=100352, assert_step0_loss=False, presence_check_step=10**9
    )
    early._assert_all(60, {"train/block 00/expert_load_cv": 1.2434})


def test_no_balance_bands_would_disable_the_cv_ceiling_too():
    """
    `--no-balance-bands` promises to disable the balance ceilings, and the CV ceiling is one.

    The flag exists for "a run whose purpose is to observe routing imbalance past the steady-state
    window". A raw-CV ceiling is the most direct possible way to kill exactly that run, so leaving
    it armed would make the flag's documented promise false. The launcher wires it; this pins that
    the callback accepts the disable and reports it in the disabled list.
    """
    callback = MetricAssertionCallback(
        vocab_size=100352,
        drop_frac_max=None,
        dead_expert_frac_max=None,
        expert_load_cv_max=None,
    )
    callback._failures = []
    callback._check_cv_band({"train/block 00/expert_load_cv": 9.9})
    assert not callback._failures, "a disabled ceiling still fired"


def test_cv_glob_does_not_leak_between_the_three_cv_metrics():
    """
    `expert_load_cv` must not match either excess form. This is the prefix trap, one level deeper.

    It matters more here than elsewhere: the three metrics differ by a factor of ~316 on real data,
    so a leak would apply a 0.50 ceiling to a value that is legitimately ~90 and fail every step of
    every healthy run -- or, in the other direction, silently gate nothing.
    """
    is_block_key = MetricAssertionCallback._is_block_key
    assert is_block_key("train/block 00/expert_load_cv", "expert_load_cv")
    assert not is_block_key("train/block 00/expert_load_cv_excess", "expert_load_cv")
    assert not is_block_key(
        "train/block 00/expert_load_cv_excess_per_micro_batch", "expert_load_cv"
    )
    assert not is_block_key(
        "train/block 00/expert_load_cv_excess_per_micro_batch", "expert_load_cv_excess"
    )
