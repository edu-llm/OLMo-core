"""In-run metric assertions (D2) and the incremental stdout result protocol (D3).

Two callbacks, deliberately in one module because they answer the same question from opposite
ends: :class:`MetricAssertionCallback` stops a run that has silently gone wrong, and
:class:`ResultProtocolCallback` makes sure that whatever a run did learn escapes the process
even when the run dies.

**Why assertions rather than dashboards.** Every band below has a *measured* provenance from the
sibling MoE track, recorded in ``maple/agents/contracts/telemetry-schema.md`` and
``maple/HANDOFF.md``. The failure mode they exist to catch is not a crash -- it is a run that
trains happily to completion with a 6.2x error in it. ``normalize_expert_weights`` at its stock
``None`` produced a measured gate mass of **0.161 against 1.000**, and nothing failed; the loss
was merely worse and nothing said why. "The metric is logged" would not have caught that. A band
would have caught it on the first logged step.

**Why they raise.** An assertion that logs a warning and continues is worse than no assertion,
because it manufactures confidence: the run completes, the log contains a line nobody read, and
the artifact looks validated. Every check here raises :class:`MetricAssertionError`, which
reaches ``Trainer._check_and_pass_on_metrics`` -> the bookkeeping future -> ``Trainer._error`` ->
``RuntimeError`` from ``training_complete``. That path is exercised by the existing
non-finite-loss check in the same method, so it is a known-working escalation route rather than a
hoped-for one.

**Exactly when it stops, which is NOT "at the failing step".** Audited in
``maple/agents/lanes/L5-telemetry/verify/assertion-raise-path.md`` and
``evidence/async-assertion-escalation-gap.md``; stated here because a docstring that overclaims
the guarantee is its own hazard.

- **Single process** (any 1-GPU gate): ``run_bookkeeping_op`` runs the op inline, so the exception
  propagates straight out of ``_log_metrics`` into ``fit()``'s handler and the run stops
  immediately, exit non-zero. This is the reliable path.
- **Multi-rank**: bookkeeping is async, so the future's done-callback stores ``_error``
  (``trainer.py:1345``) and does *not* raise. ``_error`` is only read by ``training_complete`` /
  ``is_canceled``, so with ``metrics_collect_interval=5`` **up to ~5 further optimizer steps run
  after a divergence is detected.**
- **On the final logged step it can be lost entirely**: ``_shutdown(gracefully=True)`` joins the
  bookkeeping ops but never re-reads ``_error``, so a failure detected there lets ``fit()`` return
  normally and the process exit **0**, with the failure visible only as a ``log.exception``.

The last point is why :class:`ResultProtocolCallback` matters as much as the assertions: its
``close()`` runs on both shutdown paths, so the *evidence* escapes even when the *exit code* is
wrong. A two-line ``_error`` re-check at the end of ``Trainer.fit()`` would close the gap, but
``trainer.py`` is not this lane's file and the change alters the exit code of runs that previously
exited 0 with a logged bookkeeping error -- a deliberate decision for whoever owns it.
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoError

from ..common import OPTIM_GRAD_NORM_METRIC, TRAIN_CE_LOSS_METRIC
from .callback import Callback

log = logging.getLogger(__name__)

__all__ = [
    "MetricAssertionError",
    "MetricAssertionCallback",
    "ResultProtocolCallback",
]


class MetricAssertionError(OLMoError):
    """
    Raised when an in-run metric assertion fails.

    Subclasses :class:`~olmo_core.exceptions.OLMoError` so it is recognisably ours in a traceback
    and cannot be mistaken for a framework bug. It is deliberately **not** a subclass of anything
    the trainer catches and converts to a warning.
    """


# The step-0 loss band.
#
# THE EXCESS OVER ln(V) IS A FUNCTION OF MODEL WIDTH, AND A FIXED HALF-WIDTH IS THEREFORE THE
# WRONG SHAPE FOR THIS CHECK. This is the corrected formula; the history is written out below
# because it cost a 32-GPU run to find and the arithmetic is not obvious.
#
# The mechanism, read off the init code rather than guessed. Under ``InitMethod.normal``,
# ``InitMethod.init_final_w_out`` (``nn/transformer/init.py``) leaves ``std = init_std = 0.02``:
# it overrides ``std`` to ``d_model ** -0.5`` for ``llama``, ``llama_depth``, ``normalized`` and
# ``fan_in``, and for ``normal`` it does **not**, so the LM head's weight std does not scale with
# width. ``LMHead.forward`` applies its RMSNorm *before* ``w_out``, and an RMSNorm's affine weight
# initializes to ones, so the hidden state reaching the head has per-element RMS exactly 1 and
# hence ``||h||_2 = sqrt(d)``. A logit is ``w_j . h`` with ``w_j`` iid of std ``init_std``, so
#
#     logit std  s = init_std * sqrt(d_model)
#     excess       = s^2 / 2 = init_std^2 * d_model / 2        (= 0.0002 * d_model at std 0.02)
#
# using the standard ``loss ~ ln(V) + s^2/2`` expansion for iid logits (the true label's logit is
# independent of the label at init, so ``E[z_y] = 0``). **The excess is therefore LINEAR IN d**,
# and a constant band cannot hold across widths.
#
# MEASURED PROVENANCE -- two independent runs on real hardware, matching the closed form to within
# 0.01 nats:
#
#   d_model  predicted ln(V) + 0.0002*d   measured step-0 loss   residual   run
#   -------  -------------------------   --------------------   --------   ---------------------
#   1024     11.7212                     **11.718**             -0.0032    sibling probe
#                                                                          run_019fd3eb-dd0a-70bf
#                                                                          -ba71-1141eca2c2f8
#                                                                          (`edullm_moe_m1`, tied)
#   1536     11.8236                     **11.8328**            +0.0092    maple-m7b-mfu-nb1,
#                                                                          4x8 H100, `maple_m7b`
#
# The sibling reached 11.520 by step 3, so the band still describes an initialization and nothing
# later. Tied vs untied does not change the law: under ``normal`` both ``init_embeddings`` and
# ``init_final_w_out`` use ``std = init_std``, so the sibling (tied) and M7B (untied) obey the same
# formula -- which is why two runs of different tying agree with it.
#
# WHAT THE OLD FIXED 0.3 GOT WRONG, AND WHY IT WAS NOT MERELY TIGHT. The old band was
# ``[ln V, ln V + 0.3]``, calibrated on the sibling's 0.2016 excess at **d=1024** and applied at
# every width. Consequences, both real:
#
#   * It **failed a correct `maple_m7b`** at d=1536, whose honest excess is 0.3072. That is the
#     defect this rewrite fixes; the init was never wrong.
#   * At d=2048 (`maple_m20`, the mission deliverable) the predicted excess is **0.4096**, which is
#     within **0.005 nats** of the **0.4148** that ``ln(151936) - ln(100352)`` contributes for a
#     wrong vocab. So under a fixed band M20 would not only fail while correct -- the band would be
#     **unable to distinguish a correct M20 from a vocab error at all.** That is a broken
#     discriminator, not a threshold in need of loosening, and widening 0.3 to clear M20 would have
#     swallowed exactly the failure the band exists to catch.
#
# THE FIX: center the band on the geometry-predicted excess and bound the *deviation from the
# prediction*. The vocab offset is width-independent while the init excess is now subtracted off,
# so the discriminating power is restored **and is the same at every width** -- which a fixed band
# could not be. See :data:`STEP0_EXCESS_TOLERANCE_NATS`.
STEP0_LOSS_BAND_NATS = 0.3
"""
Legacy fixed half-width above ``ln(V)``, used **only** when ``d_model`` is unknown.

Retained rather than deleted because the callback must still do something defensible when it is
constructed without a width (older call sites, ad-hoc use). When it is in force the width-aware
band is NOT, and :meth:`MetricAssertionCallback.pre_train` says so out loud -- a check that has
silently fallen back to a formula known to be wrong at d != 1024 is the confidence-manufacturing
failure this module exists to prevent.
"""

# Tolerance on the deviation of the measured step-0 loss from the geometry-predicted value.
#
# **THIS IS NOT THE OLD 0.3 UNDER A NEW NAME, AND IT IS NOT A RELAXATION.** The old number had to
# absorb the entire init excess (0.2048 at d=1024, and it did not stretch to 0.3072 at d=1536).
# This one only has to absorb the *model error* of the closed form, which is **measured at <= 0.01
# nats** at both widths above. So the requirement it must satisfy is a two-sided one:
#
#   * LOOSE ENOUGH to pass a healthy init: needs > ~0.01. At 0.20 that is a **21.7x** margin over
#     the worst measured residual (0.0092 at d=1536).
#   * TIGHT ENOUGH to reject a wrong vocab at EVERY width: needs < 0.4148. At 0.20 that is a
#     **2.07x** margin, and unlike the old band it does not decay as d grows, because the
#     prediction moves with d and the vocab offset does not.
#
# **Why 0.20 and not 0.25, which was the first value here.** 0.25 satisfied both bounds above, but
# 0.20 **strictly dominates** it: it additionally catches a fully collapsed init at **d=1024**,
# where the expected excess is 0.2048 -- inside a 0.25 tolerance, outside a 0.20 one. So the tighter
# value buys a real check at no cost to any measured healthy value. That is the one circumstance in
# which tightening is unambiguously right, and it is worth naming because the opposite mistake is
# also recorded in this file (the ``expert_load_cv`` ceiling was once "corrected" from 0.55 to 0.50
# and thereby moved *into* a band where the plan prescribes continuing -- **tighter is not
# automatically safer**; it is safer here only because nothing measured sits between 0.20 and 0.25).
#
# Not tighter still (0.12, 0.15) because the closed form is verified at exactly two widths and the
# sub-leading terms are real if small: ``trunc_normal_`` at +/-3 sigma carries ~97.3% of the nominal
# variance, so the prediction is ~2.7% high (about -0.011 nats of deviation at d=2048, biased the
# safe way for the upper bound), and bf16 logit rounding is unquantified. A tolerance that assumed
# the formula exact would fail a healthy run over a second-order effect nobody has measured.
#
# It also **gains** a check the old band could not express at any width. A collapsed or badly
# under-scaled init reads an excess near **zero**, which sat comfortably inside
# ``[ln V, ln V + 0.3]`` and was therefore invisible. Centered on the prediction it is a deviation
# of ``-init_std^2*d/2``, which exceeds 0.20 at every rung width (0.2048 / 0.3072 / 0.4096) and
# fails. The floor is additionally clamped at ``ln V`` -- below that is unreachable for an
# honestly-initialized model at any width, so it stays a hard bound in its own right.
STEP0_EXCESS_TOLERANCE_NATS = 0.20

# Gate mass tolerance around 1.0. The failure this catches is not a small drift -- it is
# `normalize_expert_weights=None`, measured at **0.161**, a 6.2x error. So the tolerance only has
# to exclude that, and a tight band risks failing a healthy run over float32 accumulation across
# a large interval. 3% is ~200x tighter than the error it is built to catch.
GATE_MASS_TOLERANCE = 0.03


@dataclass
class MetricAssertionCallback(Callback):
    """
    Asserts *magnitudes* on in-run metrics and raises when one is out of band.

    Also computes the explicitly-labelled cross-block aggregates
    (``moe/<metric>_max``, ``moe/<metric>_mean``) from the ``block NN/<metric>`` series, because
    the trainer's own cross-block fold cannot produce a mean: it folds ``ReduceType.max`` metrics
    with ``torch.max`` and ``mean``/``sum`` metrics by **adding**, which is how a normalised
    entropy of ~0.998 in each of 16 blocks came to be logged as **15.97** for a quantity defined
    on ``[0, 1]``. Computing the aggregate here, from the already-reduced per-block series, is the
    only place a real mean is available.
    """

    priority: ClassVar[int] = 4
    """
    Runs first, for two reasons that both matter.

    On the metrics path, ``pre_log_metrics`` here *adds* the ``moe/<metric>_{max,mean,min}``
    aggregates to the dict, and callbacks that consume metrics -- the console logger, W&B, the
    metric saver, the ``RESULT `` protocol -- must see them. Callbacks run in priority order, so
    computing the aggregates first is what makes them visible downstream rather than only to
    whichever callback happens to follow.

    On the shutdown path, the same argument as :class:`ResultProtocolCallback`: ``close()`` here
    reports whether the balance bands were ever applied, and an unguarded ``wandb.finish()`` later
    in the loop must not be able to suppress it. Above ``ResultProtocolCallback``'s 3 so that the
    band-application count is printed before the final ``RESULT `` line.
    """

    # -- what to assert -----------------------------------------------------------------------

    vocab_size: Optional[int] = None
    """
    Padded vocab size, used for the step-0 loss band ``[ln V, ln V + 0.3]``. If ``None`` the
    step-0 loss check is **skipped and that is logged loudly**, because a silently-skipped
    assertion is the failure mode this class exists to prevent.
    """

    step0_loss_band_nats: float = STEP0_LOSS_BAND_NATS
    """
    Legacy fixed half-width above ``ln(vocab_size)``. **Used only when :data:`d_model` is None.**

    When ``d_model`` is supplied the band comes from the geometry instead -- see
    :data:`STEP0_LOSS_BAND_NATS` for why a fixed half-width is the wrong shape for this check, and
    :meth:`step0_loss_bounds` for the band actually applied.
    """

    d_model: Optional[int] = None
    """
    Model width, which is what makes the step-0 band correct at more than one width.

    The init excess over ``ln V`` is ``init_std^2 * d_model / 2`` -- **linear in d** -- so without
    this the band is only right at the width it was calibrated on (d=1024) and is wrong at both
    ``maple_m7b`` (1536) and ``maple_m20`` (2048). Supplied by the caller from
    ``config.model.d_model``. ``None`` falls back to :data:`step0_loss_band_nats` and
    :meth:`pre_train` reports the fallback loudly.
    """

    init_std: float = 0.02
    """
    The LM head's weight init std, i.e. ``TransformerConfig.init_std``.

    A field rather than a hard-coded 0.02 because it is the *other* half of the prediction: the
    excess is ``init_std^2 * d_model / 2``, so a run that deliberately changed ``init_std`` would be
    failed by a band that assumed the default. **Only valid for** ``InitMethod.normal`` -- see
    :data:`init_scales_with_width`.
    """

    step0_excess_tolerance_nats: float = STEP0_EXCESS_TOLERANCE_NATS
    """
    Allowed deviation of the measured step-0 loss from the geometry-predicted value, when
    :data:`d_model` is known. See :data:`STEP0_EXCESS_TOLERANCE_NATS` -- it bounds the *model error*
    of the closed form (measured <= 0.01 nats), not the init excess itself.
    """

    init_scales_with_width: bool = False
    """
    Whether the init method scales the LM head std by ``d_model ** -0.5`` (``llama``,
    ``llama_depth``, ``normalized``, ``fan_in``), rather than leaving it at ``init_std``
    (``normal``).

    **This is not a hypothetical branch.** ``InitMethod.init_final_w_out`` overrides ``std`` to
    ``d_model ** -0.5`` for those four methods, which makes the logit std ``1.0`` and the expected
    excess a constant ``0.5`` **at every width** -- a different formula, not a different number. Our
    ladder is ``normal`` (the ``TransformerConfig`` default), so the default here is ``False``;
    getting it wrong in either direction produces a band centered on the wrong value, so it is read
    from the config rather than assumed.
    """

    step0_max_step: int = 5
    """
    The band is applied only to a first-logged step at or below this. Beyond it the check is
    skipped with a warning instead of being applied to a step it does not describe.

    Without this bound the check reads "the first logged step, whatever its number", and that is
    wrong in a way that only shows up in the field: the sibling's provenance is 11.718 at step 0
    falling to **11.520 by step 3**, so the loss leaves the band's neighbourhood within a handful
    of steps. Any run whose first *logged* step is late -- a resume this callback failed to detect,
    a callback attached mid-run, a large ``metrics_collect_interval``, a config that skips early
    steps -- would then have an initialisation band applied to a partly-trained loss and fail for
    the wrong reason. 5 clears the default ``metrics_collect_interval=5`` in
    ``.edullm/train_on_corpus.py``, where ``_fit_epoch`` logs the first batch unconditionally so
    the first logged step is normally step 1.
    """

    assert_step0_loss: bool = True
    dead_expert_frac_max: Optional[float] = 0.0
    """
    Ceiling on ``dead_expert_frac``. Default ``0.0`` -- the sibling measured **0.000000** across
    all 16 blocks, so zero is the real expectation rather than an aspiration. Set to ``None`` to
    disable.
    """

    gate_mass_tolerance: Optional[float] = GATE_MASS_TOLERANCE
    """Tolerance around a gate mass of 1.0. ``None`` disables."""

    drop_frac_max: Optional[float] = 0.01
    """
    Ceiling on ``drop_frac`` / ``drop_frac_upper_bound``. **1%, supplied by L3, who owns the
    number.**

    My initial default was 0.10 and L3 correctly applied my own standard back at me: at
    ``capacity_factor=2.0`` an expert must exceed **2x mean** before it drops anything, after which
    ``drop_frac ~ sum(share - 2/E)``, so at E=256 a 10% ceiling does not fire until one expert
    holds roughly **30x its share.** A ceiling that cannot fire is the same as no assertion.

    L3's derived mapping from ceiling to tolerable hottest-expert multiple at E=256:
    0.1% <-> 2.26x mean, 0.5% <-> 3.28x, **1.0% <-> 4.56x**, 5% <-> 14.8x, 10% <-> ~30x.

    1% is deliberately **uniform across rungs** rather than a per-rung ladder: it is dimensionless
    and E-normalised, which is what rule 2 of the telemetry schema requires of anything compared
    across rungs, and a per-rung table of numbers would not be. It tolerates a 4.6x hot expert at
    E=256 -- loose enough that ordinary fluctuation never trips it, given ~16 sigma of headroom at
    factor 2.0 -- while catching real collapse long before it costs a run. The sibling's measured
    max of 0.0455 was at factor **1.2**, a different regime, and is not the relevant comparison.
    """

    assert_finite: bool = True
    """
    Assert that loss and grad norm are finite on **every** logged step.

    Not redundant with the trainer's own non-finite CE-loss check. That check runs in
    ``_check_and_pass_on_metrics``, i.e. only when metrics are actually collected, and
    ``metrics_collect_interval`` is **5** in ``.edullm/train_on_corpus.py`` -- so a NaN at step 7
    that recovers by step 10 is invisible today. It also does not look at the gradient norm,
    which is the *earlier* signal for a QAT divergence: STE gradients blow up before the loss
    does.
    """

    capacity_factor_deficit_max: Optional[float] = 0.0
    """
    Ceiling on ``capacity_factor_deficit`` = ``max(0, configured - effective)``.

    **This encodes ``effective >= configured``, and it is deliberately NOT an equality.**
    ``ensure_multiple_of`` rounds capacity **up**, so the realized factor is expected to exceed the
    configured one whenever quantization moves it -- L3 measured a requested 1.2 realizing as 1.2188
    at E=256 with an 8192-token local microbatch, and 1.2031 at 16384. An ``== configured`` gate
    would therefore gate on the number somebody *intended* rather than the one the dispatch
    *allocated*.

    On the funded path (factor 2.0) the quantization vanishes at every rung in this ladder, so the
    deficit is exactly 0.0 -- **which is exactly why an equality gate would look safe right up until
    a rung or batch size changed, and then fire on a healthy run.** That is D-019's failure mode
    hiding inside this very metric, so the direction matters more than the tightness here.

    A ceiling of 0.0 on the deficit is tight *and* correct: rounding up cannot produce a positive
    deficit, so anything above zero means the dispatch allocated **less** capacity than the config
    asked for, which is a real defect rather than a tolerance question.
    """

    assert_instrumented: bool = True
    """
    Fail if a metric that reports its own unavailability says it is unavailable.

    **THE B3 SCAR.** The G2 run measured ``drop_frac`` at 0.135-0.242 with 6 of 64 experts dead in
    one block -- collapse, not fluctuation -- and when asked *where in the sequence* those drops
    landed, the answer was "cannot tell you". ``drop_by_position`` and ``drop_by_doc_index`` were
    computed on every step and published nowhere, so the gate could not distinguish **"positional
    drops are healthy"** from **"positional drops were never measured"** on a run that had already
    cost an A10G.

    L3 wrote that exact warning to two lanes *before* the run -- "a run that quietly emits nan
    histograms would look like B3 failed rather than like B3 was never switched on" -- and it
    happened anyway, **because a warning in prose is not a check.**

    Note carefully what did *not* fail: :data:`require_present` **passed**, on
    ``dead_expert_frac_global`` / ``gate_mass_mean`` / ``drop_frac``. It protected exactly the names
    it listed and was silent about every name it did not. So the lesson is not "the check was
    broken" -- it is that **a check whose coverage is a hand-maintained list has a blind spot
    shaped exactly like the list**, which is the same class as the registry-vs-emission gap.

    This is the magnitude form of the same idea, and it is why the metric exists rather than being
    inferred: a component that *knows* it is uninstrumented says so in a number
    (``drop_histograms_unavailable`` = 1.0), and that number is asserted like any other band. The
    metric reports its own absence, so nobody has to remember to list it.
    """

    unavailable_metric_suffix: str = "_unavailable"
    """
    Any per-block metric whose name ends with this must read 0.0.

    A **convention**, not a list, and that is the whole point. Any component can declare "my
    instrument did not run" by emitting ``<something>_unavailable = 1.0``, and it is gated
    automatically -- no edit here, nothing to remember, no list to fall out of date. Currently
    satisfied by ``drop_histograms_unavailable``.

    Deliberately *not* also gating ``..._not_reduced``: on a single-rank run
    ``dead_expert_counts_not_reduced`` is legitimately 1.0, so a blanket band there would fire on
    every 1-GPU gate. "Not reduced" is a correct state; "unavailable" is not.
    """

    expert_load_cv_max: Optional[float] = 0.55
    """
    Ceiling on ``expert_load_cv`` -- **RAW CV, and deliberately not on** ``expert_load_cv_excess``.

    **WHY THE RAW QUANTITY IS THE GATEABLE ONE (D-075).** ``expert_load_cv_excess`` divides by a
    null computed from the **accumulated** mean, so ``cv_excess = CV * sqrt(T*N*k/(E-1))`` and it
    **grows without bound in the logging-interval length at fixed physical balance**. Measured: on
    X1 the ratio ``cv_excess / expert_load_cv`` was exactly **316.013** at every one of 600 logged
    steps, and 316.01 again on a different run and arm. So a band on it would mean **a longer
    logging interval alone can trip a balance ceiling on a physically identical model** -- and the
    20B run has long intervals by construction. That is a fail-closed footgun in a suite whose
    assertions RAISE.

    Raw CV has no such dependence: it is scale-invariant in the counts and converges to the
    persistent imbalance as the window grows. It is not normalised against ``E``, which is a real
    limitation for a **cross-rung comparison** -- but it is not a limitation for a **ceiling**,
    because the ceiling is set from measurements at this ladder's own assignment counts rather than
    derived from a null. Read ``expert_load_cv_excess_per_micro_batch`` for the E-normalised,
    window-free cross-rung readout. **Gate the magnitude; compare the normalised statistic.**

    **PROVENANCE OF 0.55: IT IS THE PROJECT'S OWN PRE-REGISTERED ESCALATION THRESHOLD.**
    ``maple/plan/balance-floor.md`` fixes a decision rule *before* the P1 probe, on **block-max raw
    CV** -- which is exactly the quantity this band evaluates, since the check fires if ANY block
    exceeds:

    ======================  =====================================================
    block-max raw CV        pre-registered meaning
    ======================  =====================================================
    <= 0.40                 RETIRE balance as a 20B risk. Ship at cf=2.0.
    (0.40, 0.55]            ADOPT capacity_factor 2.5. One config change, no new science.
    > 0.55                  ESCALATE: balance is real and architectural.
    ======================  =====================================================

    **So the ceiling is 0.55 and not a rounder number because that is where the project already
    committed, in writing, that the situation stops being a config change and becomes a decision for
    the user.** A raising assertion belongs exactly at the ESCALATE boundary and nowhere tighter:
    inside ``(0.40, 0.55]`` the pre-registered response is *adopt a larger capacity factor and keep
    going*, so a band that raised there would kill a run the plan says to continue.

    **AND THIS CORRECTS MY OWN FIRST TWO ATTEMPTS, BOTH OF WHICH WERE WRONG.** They are recorded
    because each error is instructive and neither is obvious:

    1. **0.55 justified from the wrong evidence.** I first set 0.55 by comparing our CV against
       "published baselines" of 0.72 / 0.52 / 0.937 from Wang et al. (arXiv:2408.15664) Table 2.
       **Those numbers are MaxVio = max/mean - 1, NOT CV.** ``balance-floor.md`` converts X1 to
       MaxVio **0.683** before comparing it to them, precisely because the units differ. Comparing a
       CV to a MaxVio is a category error that happens to land in a plausible range, which is the
       worst kind. **Almost nobody reports expert-load CV**, so there is no external CV referent to
       calibrate against -- the calibration has to come from our own measurements and our own
       pre-registered rule.
    2. **0.50, "corrected" from an interval that was itself unit-confused.** Retreating to 0.50
       looked like tightening on evidence. It was tightening on the *same wrong upper bound*, and it
       moved the ceiling **into** the pre-registered ADOPT band -- i.e. it would hard-fail runs the
       plan says to continue with cf=2.5. A tighter band is not automatically a safer band.

    **AND THE HEALTHY MEASUREMENTS MUST BE READ AS BLOCK-MAX, WHICH THEY MOSTLY WERE NOT.** This
    band iterates per-block keys, so the relevant healthy figure is the **worst block**, not block
    00:

    ================================  =============  ==============  ================
    run                               block 00       **block max**   margin to 0.55
    ================================  =============  ==============  ================
    X1 tail (steps 587-600, flat)     0.2853         **0.3512**      1.57x
    X4a control                       0.3064         not recorded    --
    X4a ternary                       0.3370         not recorded    --
    ================================  =============  ==============  ================

    X1's spread was **0.1755-0.3512 across 8 blocks** (``balance-floor.md``), so quoting 0.2853
    against a block-max gate overstates the margin by 1.23x. The X4a figures are block-00 or
    cross-block-mean and **their block-max is not recorded anywhere** -- so the honest binding
    number is X1's **0.3512**, giving **1.57x**. Note that 0.3512 is *above* 0.3370, i.e. an earlier
    draft of this docstring enforced a floor that X1's own healthy block-max violated.

    **THE REAL RESIDUAL RISK, STATED PLAINLY: every one of these is E=64, and the flagship is
    E=256.** Nothing above E=64 has ever trained. ``balance-floor.md`` computes the *sampling* term
    at E=256 as only +0.007 to +0.026 -- negligible against 0.55 -- but calls the **persistent**
    term at E=256 "genuinely unknown", and retiring that unknown is the entire purpose of the P1
    probe. So this band is calibrated on E=64 and applied at E=256 on the argument that its ceiling
    is a *pre-registered decision boundary* rather than an extrapolated measurement. **If P1 reads
    block-max raw CV above 0.40 at E=256, this default must be revisited before any long run** --
    not by widening it, but by taking the escalation the plan already describes.

    Lower reference, from the consequence side: ``drop_frac`` crosses its ratified 0.01 ceiling at
    raw CV ~= **0.46** at ``capacity_factor=2.0``. **0.55 is deliberately ABOVE that, and that is
    not a relaxation.** Below ~0.46 the drop band is the tighter of the two and fires first, so a CV
    ceiling at 0.46 would be redundant with it. The CV band earns its place in the regime
    ``drop_frac`` is **blind** to: D-053 measured drops falling **1000x** (0.2247 -> 0.0002) while
    raw CV fell only **3.9x** (1.2434 -> 0.3204), because at 2x capacity an expert drops nothing
    until it exceeds 2x mean. ``drop_frac`` is a *thresholded* view of balance, not a measure of it.
    A CV band above the drop threshold is what catches imbalance that has stopped costing drops and
    is still costing quality.

    **AND IT MUST CLEAR THE RECOVERY TRANSIENT, WHICH IS WHY** :data:`expert_load_cv_warmup_steps`
    **EXISTS.** The same D-053 curve, at E=64, on a run that completed 600/600 steps cleanly:

    ======  =================  ===========
    step    raw CV (block 00)  vs 0.55
    ======  =================  ===========
    50      1.2434             **2.3x over**
    100     0.7500             **1.4x over**
    200     0.4618             under, by 1.19x
    300     0.3567             under, by 1.54x
    339     0.3204             under, by 1.72x
    ======  =================  ===========

    **These are block-00 values and the band is on block-max, so every margin above is optimistic.**
    Scaling by X1's measured block-max/block-00 ratio (0.3512/0.2853 = 1.23x) puts step 300 at an
    implied block-max ~0.439, clearing 0.55 by only **~1.25x** rather than 1.54x. The per-block
    curve was never recorded, so that is an estimate and is flagged as one.

    So this ceiling applied from the shared ``warmup_steps=50`` would have **failed a healthy run
    twice over**. An untrained router has not spread load yet and the recovery runs for *hundreds*
    of steps -- far longer than the drop and dead-expert transients ``warmup_steps=50`` was measured
    against. The alternative, widening the ceiling to pass 1.2434, was rejected on the standing rule
    from ``warmup_steps``: **narrow the window, not the band.** A ceiling above 1.24 could not catch
    anything short of near-total collapse (CV at full collapse onto one expert is ``sqrt(E-1)`` =
    7.9 at E=64, 16.0 at E=256).

    Set to ``None`` to disable, and ``--no-balance-bands`` disables it with the other two.
    """

    expert_load_cv_warmup_steps: int = 300
    """
    The step from which :data:`expert_load_cv_max` applies. **Deliberately later than**
    ``warmup_steps``.

    300 rather than 50 because the balance transient and the *drop* transient have very different
    durations, measured on the same run: ``drop_frac`` was already at 0.0021 by step 200 while raw
    CV was still 0.4618, and CV did not settle to its 0.2853 tail until step ~590. Sharing one
    window would either fire this band on a healthy recovery or force the drop band's window out to
    where it protects nothing.

    Chosen from the curve, not rounded for looks: at step 300 the block-00 value is 0.3567, and
    scaling by X1's measured block-max/block-00 ratio of 1.23x gives an implied block-max ~0.439 --
    clearing the 0.55 ceiling by ~1.25x. Step 200 (block 00 0.4618, implied block-max ~0.568) would
    be **over the ceiling**, so 200 is not merely noisy, it would fail a healthy run outright.

    **THE COST OF THIS WINDOW IS REAL AND IS NOT HIDDEN: at 300 steps, this band checks almost
    nothing on the current gate ladder.** Measured gate lengths are G2 60, G4 60, G3 120, G5 200,
    G7 300, G6 500, and the 20B throughput probe 150 -- so with ``metrics_collect_interval=5`` four
    gates check this band **zero** times, G7 at most once, and only G6 gets real coverage. **Both
    runs that would read E=256 balance (G5, and the throughput probe) get zero checks.** That is
    why :meth:`close` reports ``cv_band_checks_applied`` separately and WARNS at zero: on most gates
    the honest output of this band is "not evidence of routing balance", and it must say so rather
    than pass silently. Lowering the window to cover them was rejected -- at 200 the band would fire
    on a measured healthy run -- so the correct fix is a longer probe, not an earlier window.

    **The vacuity hole this opens is closed explicitly, not by assumption.** A run shorter than 300
    steps never checks this band, and :meth:`close` therefore reports its application count
    **separately** from the other bands' -- a single "bands applied on 250 steps" line would be true
    and would hide the fact that the CV band was applied **zero** times. That is D-048's lesson
    (the blind spot is the *definition* of the coverage, not its contents) and it applies to a
    per-band window just as much as to a metric list.
    """

    entropy_deficit_max: Optional[float] = None
    """
    Ceiling on ``entropy_deficit``. **Default ``None`` (disabled), deliberately.**

    The sibling measured a worst-block deficit of **0.0663** against a planning band of
    ``[0, 0.06]`` -- i.e. a healthy run was *already marginally out of band* at 2048
    assignments/expert, and the ladder runs at 1024 (R1) down to **256** (R3), where the
    statistic is noisier by construction. A ceiling set from the one measurement available would
    fire on healthy runs at R3 and teach everyone to ignore it. The metric is logged and read;
    it is not gated until there is a measurement at the relevant assignment count. Set it
    explicitly if you have one.
    """

    # -- how loudly ---------------------------------------------------------------------------

    warmup_steps: int = 50
    """
    The balance bands apply only from the **steady-state window**, i.e. steps at or after this.
    Before it, only the loss band and the finite check run. Ruled by the orchestrator 2026-08-06.

    **Why this is not a fudge.** The balance metrics are not merely noisy at step 1, they are
    *not yet the quantity the band describes*. Measured on an R0 toy (6 steps, 1/8 micro-batch):
    ``dead_expert_frac`` **0.219** and ``drop_frac_upper_bound`` **0.217** at step 1 -- both wildly
    outside their bands, on a run that was working correctly. An untrained router has not yet
    spread load across experts, and at a small local batch a large fraction of experts legitimately
    receive nothing on the first step.

    The alternative -- loosening the bands so they hold at step 1 on a toy config -- was rejected
    and rightly: a ceiling wide enough to pass ``dead_expert_frac = 0.219`` cannot catch anything
    real at R3, where the whole point is that zero is the measured expectation. **Narrow the
    window, not the band.**

    This is the third instance of D-019 (the null for a balance metric is almost never zero) and it
    sharpens it: the null is not even *stationary* at step 1. 50 matches the discard used by the
    throughput harness for the same underlying reason -- early steps are not the regime being
    measured. It is deliberately a **plain step index, not a fraction of the run**, so a 6-step
    smoke test simply never enters the window and reports that rather than asserting on transients.
    """

    require_present: Tuple[str, ...] = (
        "dead_expert_frac_global",
        "gate_mass_mean",
        "drop_frac",
        # The presence signal must itself be present. Without this the suffix convention above
        # protects nothing: a build that stopped emitting `drop_histograms_unavailable` would have
        # no `..._unavailable` key to match, so the instrumentation check would pass by finding
        # nothing -- the very shape of failure it was added to prevent, one level up.
        "drop_histograms_unavailable",
    )
    """
    Per-block metric names that MUST appear in the metrics dict, checked once at
    ``presence_check_step``.

    THIS IS THE ANTI-VACUITY GUARD AND IT IS THE MOST IMPORTANT FIELD IN THIS CLASS. Every
    ceiling above is applied by iterating the metrics dict and testing the values found. If a
    metric is never emitted -- a rename on the producing side, a lane that logs ``drop_rate``
    while this checks ``drop_frac``, an accumulator that stayed at zero so its guard suppressed
    the metric -- then the loop finds nothing, no failure is appended, and **the assertion passes
    because it checked nothing at all.** That is strictly worse than having no assertion, because
    the run comes back green and looks validated.

    So absence is itself a failure. ``gate_mass_mean`` is on this list because it is the guard on
    ``normalize_expert_weights`` (measured 0.161 vs 1.000) and it is emitted conditionally on an
    accumulator being non-empty -- exactly the shape of thing that can go quiet without anyone
    noticing.

    ``drop_frac`` **is** on this list as of L3's confirmation (2026-08-06): L3 produces the true
    per-micro-batch rate from the same ``indices``/``bins``/``expert_capacity`` that reach
    ``binned_gather``, verified against the real Triton kernel by multiset equality. So its absence
    would now be a defect rather than a known open state, and the ceiling that checks it must not
    be allowed to pass vacuously.

    ``drop_by_position`` / ``drop_by_doc_index`` are **not** required: L3 reports they come back
    ``nan`` unless ``drop_accounting_seq_len`` is set on each ``ParallelMLP``, which lives in L6's
    file. Requiring them would make this guard fail on a switch being off, which is a different
    fault from a metric having gone missing.
    """

    presence_check_step: int = 1
    """
    Which logged step the presence check runs on. Applied on the first logged step at or after
    this, once. Kept early on purpose: a vacuity check that fires at step 1000 has already let
    999 steps of unverified training happen.
    """

    enabled: bool = True

    # -- internal -----------------------------------------------------------------------------

    _block_metric_names: Tuple[str, ...] = (
        "expert_load_cv",
        # KEPT for aggregation, NOT GATED. `cv_excess` is `CV * sqrt(T*N*k/(E-1))` and therefore
        # depends on the logging interval (D-075); a band on it can fire on a physically identical
        # model. The window-free sibling is the E-comparable readout.
        "expert_load_cv_excess",
        "expert_load_cv_excess_per_micro_batch",
        "entropy_deficit",
        "dead_expert_frac",
        "dead_expert_frac_global",
        "dead_expert_counts_not_reduced",
        "gate_mass_mean",
        "drop_frac",
        "drop_frac_upper_bound",
        "assignments_per_expert_mean",
        # L3's B3 axes. Aggregated so the max/mean/min per bucket is available, but DELIBERATELY
        # NOT GATED -- see the note in `_check_bands`. L3 measured the positional axis as real
        # signal rather than noise, so a flatness assertion would fire on every healthy run.
        #
        # These bare names are NOT what is emitted: the histograms travel as one scalar per bucket
        # (`drop_by_position_b00` ...), because `record_metric` takes a scalar. They stay listed so
        # the registry test's KNOWN_PENDING reverse-guard keeps pointing at something, and because
        # `_add_cross_block_aggregates` matching them is harmless when nothing bears the bare name.
        "drop_by_position",
        "drop_by_doc_index",
        # The presence signals. Gated by suffix convention (see `assert_instrumented`), listed here
        # so the cross-block aggregate exists -- `moe/drop_histograms_unavailable_max` answers "was
        # ANY block uninstrumented" in one number, which is the form a gate wants.
        "drop_histograms_unavailable",
        "drop_by_position_num_buckets",
        "drop_by_doc_index_num_buckets",
        # The capacity pair and its derived violation. All three, because a reader debugging a
        # deficit needs both sides, not just the verdict.
        "effective_capacity_factor",
        "configured_capacity_factor",
        "capacity_factor_deficit",
    )
    _failures: List[str] = field(default_factory=list, repr=False)
    _checked_step0: bool = field(default=False, repr=False)
    _checked_presence: bool = field(default=False, repr=False)
    _resumed: bool = field(default=False, repr=False)
    _bands_applied: int = field(default=0, repr=False)
    _cv_band_applied: int = field(default=0, repr=False)
    _last_step_seen: int = field(default=0, repr=False)

    def state_dict(self) -> Dict[str, Any]:
        # `_checked_step0` is saved so a resumed run does not re-apply the step-0 band to a step
        # that is numerically nothing like step 0. See `post_checkpoint_loaded`.
        #
        # `_checked_presence` is deliberately NOT saved: the presence check should run again on
        # every attempt, because a resumed attempt runs a freshly-built model in a fresh process
        # and is exactly as capable of emitting nothing as the first attempt was.
        return {"checked_step0": self._checked_step0}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._checked_step0 = bool(state_dict.get("checked_step0", False))

    def post_checkpoint_loaded(self, path):
        # A RESUMED RUN MUST NOT BE HELD TO THE STEP-0 BAND. The platform's checkpoint contract
        # gives `olmo-core-train-{1,4}gpu` two attempts with `resume_required`, so the second
        # attempt's first logged step is step ~N with a loss of maybe 4.4, not 11.5. Asserting
        # `[ln V, ln V + 0.3]` there would fail every retry -- turning an assertion meant to
        # catch a broken init into a guarantee that no run survives preemption.
        del path
        self._resumed = True
        log.info(
            "MetricAssertionCallback: checkpoint loaded, so the step-0 loss band will not be "
            "applied to this attempt's first logged step."
        )

    def pre_train(self):
        if not self.enabled:
            # Announced at WARNING because a disabled assertion suite that nobody notices is the
            # exact shape of the failure this class exists to prevent.
            log.warning(
                "MetricAssertionCallback is DISABLED. No in-run metric assertion will fire, "
                "including the gate-mass check that guards `normalize_expert_weights` and the "
                "step-0 loss band. This run is not gated."
            )
            return

        # Report the whole active configuration up front, so the log records which checks were
        # armed for this run rather than leaving it to be inferred from which ones did not fire.
        active: List[str] = []
        skipped: List[str] = []

        if self.assert_step0_loss and self.vocab_size is not None:
            lo, hi = self.step0_loss_bounds
            ln_v = math.log(self.vocab_size)
            predicted = self.step0_predicted_excess
            if predicted is None:
                # A band known to be wrong away from d=1024 must not be applied quietly. WARNING
                # rather than a line in the INFO block, for the same reason the disabled balance
                # bands are: this one changes what the assertion means.
                active.append(
                    f"step-0 CE loss in [{lo:.4f}, {hi:.4f}] (ln V = {ln_v:.4f}) -- LEGACY FIXED "
                    "BAND, only correct at d=1024"
                )
                log.warning(
                    "MetricAssertionCallback: `d_model` was not supplied, so the step-0 loss band "
                    "is the LEGACY FIXED [ln V, ln V + %.2f]. That form is only correct at "
                    "d_model=1024: the init excess is init_std^2*d/2, so it is 0.3072 at d=1536 "
                    "and 0.4096 at d=2048, and this band WILL FAIL a correct model at either "
                    "width. Pass `d_model` (and `init_std`) to get the width-aware band.",
                    self.step0_loss_band_nats,
                )
            else:
                active.append(
                    f"step-0 CE loss in [{lo:.4f}, {hi:.4f}] = ln V + {predicted:.4f} "
                    f"+/-{self.step0_excess_tolerance_nats} (ln V = {ln_v:.4f}, predicted excess "
                    f"init_std^2*d/2 for d_model={self.d_model}, init_std={self.init_std})"
                )
        elif self.assert_step0_loss:
            skipped.append(
                "step-0 CE loss band -- SKIPPED because vocab_size was not supplied. This is the "
                "check that catches a wrong tokenizer, and it is not running."
            )
        if self.assert_finite:
            active.append("CE loss and grad norm finite, every logged step")
        if self.dead_expert_frac_max is not None:
            active.append(f"dead_expert_frac_global <= {self.dead_expert_frac_max}")
        else:
            skipped.append(
                "dead_expert_frac_global -- DELIBERATELY DISABLED. Expert death is not gated on "
                "this run"
            )
        if self.gate_mass_tolerance is not None:
            active.append(f"gate mass in 1.0 +/- {self.gate_mass_tolerance}")
        else:
            skipped.append("gate mass (guards `normalize_expert_weights`)")
        if self.drop_frac_max is not None:
            active.append(f"drop_frac <= {self.drop_frac_max}")
        else:
            skipped.append(
                "drop_frac -- DELIBERATELY DISABLED. Token dropping is not gated on this run"
            )
        if self.expert_load_cv_max is not None:
            active.append(
                f"expert_load_cv <= {self.expert_load_cv_max} (RAW CV, from step "
                f"{self.expert_load_cv_warmup_steps} -- NOT expert_load_cv_excess, which is "
                "window-dependent; see D-075)"
            )
        else:
            skipped.append(
                "expert_load_cv -- DELIBERATELY DISABLED. Routing balance is not gated on this run"
            )
        if self.entropy_deficit_max is not None:
            active.append(f"entropy_deficit <= {self.entropy_deficit_max}")
        else:
            skipped.append(
                "entropy_deficit -- intentionally ungated, no measurement exists at the "
                "ladder's assignments-per-expert"
            )

        log.info(
            "MetricAssertionCallback armed. Assertions RAISE, they do not warn.\n"
            + "".join(f"    ASSERT  {line}\n" for line in active)
            + "".join(f"    skipped {line}\n" for line in skipped)
        )

        # THE BALANCE BANDS BEING OFF IS THE ONE THING THAT MUST SURVIVE THE LOG.
        #
        # Three reasons this is not left to the INFO line above. (a) INFO is rank-0-only under
        # `rank0_only` log filtering while WARNING is never filtered, so a WARNING is the level that
        # reliably lands. (b) A 3.4-hour run buries a single startup INFO line thousands of lines
        # deep, and the question "was this run gating balance?" gets asked afterwards, by someone
        # reading the evidence rather than watching the console. (c) A run with the balance bands off
        # produces exactly the numbers a gated run would produce if it were healthy -- the only
        # difference is whether anything would have stopped it -- so the record that it was *not*
        # gating is the difference between a measurement and a claim.
        #
        # Emitted as `RESULT ` too, on the `probes/` convention, because a machine-readable line is
        # what an evidence file can be grepped for. `ResultProtocolCallback` owns the `RESULT ` tag
        # in general; this is one startup record, not a per-step series.
        disabled = [
            name
            for name, value in (
                ("drop_frac", self.drop_frac_max),
                ("dead_expert_frac_global", self.dead_expert_frac_max),
                ("expert_load_cv", self.expert_load_cv_max),
            )
            if value is None
        ]
        if disabled and get_rank() == 0:
            print(
                "RESULT "
                + json.dumps(
                    {
                        "check": "metric_assertions_config",
                        "balance_bands_disabled": disabled,
                        "drop_frac_max": self.drop_frac_max,
                        "dead_expert_frac_max": self.dead_expert_frac_max,
                        "expert_load_cv_max": self.expert_load_cv_max,
                        "expert_load_cv_warmup_steps": self.expert_load_cv_warmup_steps,
                        # The bands that stay HARD even on a diagnostic run, named explicitly so the
                        # record says what was still being enforced rather than only what was not.
                        "still_enforced": {
                            "step0_loss_band": self.assert_step0_loss
                            and self.vocab_size is not None,
                            "finite_loss_and_grad": self.assert_finite,
                            "instrumentation_present": self.assert_instrumented,
                            "gate_mass": self.gate_mass_tolerance is not None,
                            "capacity_factor_deficit": self.capacity_factor_deficit_max is not None,
                        },
                    }
                ),
                flush=True,
            )
        if disabled:
            log.warning(
                "BALANCE BANDS DELIBERATELY DISABLED for this run: %s. Those quantities are "
                "MEASURED AND LOGGED but NOT GATED, so this run cannot be cited as evidence that "
                "balance was healthy -- only as a measurement of what balance did. The loss band, "
                "the finite-loss/grad check, the instrumentation-presence check, the gate-mass band "
                "and the capacity-factor band all remain ACTIVE, so a divergence or a missing "
                "instrument still fails the run.",
                ", ".join(disabled),
            )

    @property
    def step0_predicted_excess(self) -> Optional[float]:
        """
        The geometry-predicted excess of step-0 CE loss over ``ln V``, in nats, or ``None`` if
        :data:`d_model` was not supplied.

        ``init_std^2 * d_model / 2`` for ``InitMethod.normal``, because the head's std does not
        scale with width there; ``0.5`` for the width-scaling methods, where the logit std is 1.0 by
        construction. Derivation and the two measurements that confirm it are in
        :data:`STEP0_LOSS_BAND_NATS`.
        """
        if self.d_model is None:
            return None
        s = 1.0 if self.init_scales_with_width else self.init_std * math.sqrt(self.d_model)
        return s * s / 2.0

    @property
    def step0_loss_bounds(self) -> Tuple[float, float]:
        """
        The band applied to the first logged step: ``ln V + predicted_excess +/- tolerance``,
        floored at ``ln V``.

        Width-aware whenever :data:`d_model` is known, which is the whole point -- the fixed
        ``[ln V, ln V + 0.3]`` form was calibrated at d=1024 and both fails a correct d=1536 model
        and cannot separate a correct d=2048 model from a wrong vocab. See
        :data:`STEP0_LOSS_BAND_NATS`.

        The lower bound is clamped at ``ln V`` because a loss below the entropy of a uniform
        distribution over the vocab is not reachable by an honestly-initialized model at any width,
        whatever the prediction says.
        """
        assert self.vocab_size is not None
        ln_v = math.log(self.vocab_size)
        excess = self.step0_predicted_excess
        if excess is None:
            # No width: the legacy fixed band, and `pre_train` announces the fallback.
            return ln_v, ln_v + self.step0_loss_band_nats
        center = ln_v + excess
        return max(ln_v, center - self.step0_excess_tolerance_nats), (
            center + self.step0_excess_tolerance_nats
        )

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        # `pre_log_metrics` rather than `log_metrics`, so the aggregates below are visible to
        # every downstream metric consumer (console, W&B, MetricSaver, the RESULT protocol)
        # rather than only to whichever callback happens to run after this one.
        if not self.enabled:
            return

        self._add_cross_block_aggregates(metrics)
        self._assert_all(step, metrics)

    # -- aggregates ---------------------------------------------------------------------------

    def _add_cross_block_aggregates(self, metrics: Dict[str, float]):
        """
        Compute ``moe/<metric>_{max,mean,min}`` from the ``train/block NN/<metric>`` series.

        The trainer cannot do this. ``MoETransformer.compute_auxiliary_metrics`` folds same-named
        per-block metrics into one bare key, and it folds ``mean``/``sum`` by **adding** -- which
        is correct for auxiliary losses and wrong for anything bounded. That is why every B2
        metric is tagged ``ReduceType.max``, and why the bare folded key means "worst block"
        without saying so in its name. Here, working from the per-block series after reduction, a
        mean is a real mean.
        """
        for name in self._block_metric_names:
            values = [
                value
                for key, value in metrics.items()
                if self._is_block_key(key, name)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ]
            if not values:
                continue
            metrics[f"moe/{name}_max"] = max(values)
            metrics[f"moe/{name}_mean"] = sum(values) / len(values)
            metrics[f"moe/{name}_min"] = min(values)
            # The count is recorded because an aggregate over the wrong number of blocks is the
            # quiet way this goes wrong: if only 6 of 12 MoE blocks report, the mean is still a
            # plausible number and nothing says it covered half the model.
            metrics[f"moe/{name}_n_blocks"] = float(len(values))

    @staticmethod
    def _is_block_key(key: str, metric: str) -> bool:
        return fnmatch(key, f"train/block */{metric}") or fnmatch(key, f"block */{metric}")

    # -- assertions ---------------------------------------------------------------------------

    def _assert_all(self, step: int, metrics: Dict[str, float]):
        self._failures = []
        self._last_step_seen = max(self._last_step_seen, step)

        if self.assert_finite:
            self._check_finite(metrics)
        if self.assert_step0_loss:
            self._check_step0_loss(step, metrics)
        if step >= self.presence_check_step:
            self._check_presence(step, metrics)
        if step >= self.warmup_steps:
            # Counted, not just executed. See `close()`: a run that never reaches the steady-state
            # window has had its balance bands checked ZERO times, and that must be said out loud
            # rather than inferred from an absence of failures.
            self._bands_applied += 1
            self._check_bands(metrics)
        # The CV band has its OWN, LATER window, and is therefore counted separately. Measured
        # reason: at step 200 of a run that finished 600/600 clean, `drop_frac` was 0.0021 (settled)
        # while raw CV was 0.4618 and still falling to a 0.2853 tail. See
        # `expert_load_cv_warmup_steps`. Folding it into `_bands_applied` would let a 250-step run
        # report "bands applied on 200 steps" while this band ran zero times -- true, and hiding
        # exactly the thing `close()` exists to say out loud.
        if step >= self.expert_load_cv_warmup_steps:
            self._cv_band_applied += 1
            self._check_cv_band(metrics)

        if self._failures:
            detail = "\n".join(f"  - {failure}" for failure in self._failures)
            raise MetricAssertionError(
                f"{len(self._failures)} in-run metric assertion(s) failed at step {step}:\n"
                f"{detail}\n"
                "Bands and their measured provenance are in "
                "maple/agents/contracts/telemetry-schema.md. This raises rather than warns "
                "deliberately: every band here guards a defect that otherwise TRAINS HAPPILY to "
                "completion."
            )

    def close(self):
        """
        Say, on the way out, whether the balance bands were ever actually applied.

        THE WARMUP WINDOW OPENS A VACUITY HOLE AND THIS IS WHAT CLOSES IT. With
        ``warmup_steps=50``, a run shorter than 50 steps -- a smoke test, an early crash, a gate
        that only ever meant to do 6 steps -- never enters the steady-state window, so **every
        balance band is skipped and no failure is ever appended.** That is indistinguishable, from
        the outside, from a run whose balance was verified and found healthy. It is the same
        confidence-manufacturing failure that `require_present` exists to prevent, arriving through
        a different door.

        So the count is reported unconditionally at shutdown. It does not raise: a short run is a
        legitimate thing to do, and failing it would make the smoke test unrunnable. It states
        plainly which bands did and did not get exercised, at WARNING when the answer is none, so a
        green 6-step run cannot be quoted as evidence of routing health.

        Printed via ``RESULT `` as well as logged, because a log line that only exists in a stream
        nobody re-reads is how the sibling published nothing.
        """
        if not self.enabled or get_rank() != 0:
            return
        record = {
            "check": "metric_assertions",
            "band_checks_applied": self._bands_applied,
            "warmup_steps": self.warmup_steps,
            # SEPARATE, because the window is separate. A 250-step run would otherwise report
            # "bands applied on 200 steps" -- true, and silent about the CV band having run zero
            # times. Same class as D-048: the blind spot is the definition of the coverage.
            "cv_band_checks_applied": self._cv_band_applied,
            "expert_load_cv_warmup_steps": self.expert_load_cv_warmup_steps,
            "expert_load_cv_max": self.expert_load_cv_max,
            "last_step_seen": self._last_step_seen,
            "step0_loss_checked": self._checked_step0,
            "presence_checked": self._checked_presence,
            "resumed": self._resumed,
        }
        print("RESULT " + json.dumps(record), flush=True)
        if self._bands_applied == 0:
            log.warning(
                "MetricAssertionCallback: the balance bands were applied on ZERO steps. The run "
                f"reached step {self._last_step_seen} and the steady-state window starts at "
                f"{self.warmup_steps}, so dead_expert_frac, drop_frac and the gate-mass band were "
                "NEVER CHECKED. This run is not evidence of routing health. Expected for a smoke "
                "test; if this was a real run, lower warmup_steps or run longer."
            )
        else:
            log.info(
                f"MetricAssertionCallback: balance bands applied on {self._bands_applied} logged "
                f"step(s) from step {self.warmup_steps} onward, all within band."
            )
        # Reported unconditionally and separately, because its window is later and a run can
        # legitimately clear the first window and never reach this one.
        if self.expert_load_cv_max is None:
            log.warning(
                "MetricAssertionCallback: the raw expert_load_cv ceiling was DISABLED for this "
                "run. Routing balance was measured and logged but NOT gated."
            )
        elif self._cv_band_applied == 0:
            log.warning(
                "MetricAssertionCallback: the raw expert_load_cv ceiling was applied on ZERO "
                f"steps. The run reached step {self._last_step_seen} and this band's window starts "
                f"at {self.expert_load_cv_warmup_steps} (later than warmup_steps="
                f"{self.warmup_steps} on purpose -- a healthy run measured CV 1.2434 at step 50 "
                "and 0.4618 at step 200, settling to 0.2853 only near step 590). So this run is "
                "NOT evidence of routing balance, even if the other balance bands passed."
            )
        else:
            log.info(
                f"MetricAssertionCallback: raw expert_load_cv ceiling "
                f"({self.expert_load_cv_max}) applied on {self._cv_band_applied} logged step(s) "
                f"from step {self.expert_load_cv_warmup_steps} onward, all within band."
            )

    def _check_finite(self, metrics: Dict[str, float]):
        for name in (TRAIN_CE_LOSS_METRIC, OPTIM_GRAD_NORM_METRIC):
            value = metrics.get(name)
            if value is None:
                continue
            if not math.isfinite(value):
                self._failures.append(
                    f"'{name}' is {value}, which is not finite. Ternary QAT is the expected "
                    "source: STE gradients diverge before the loss does, so a non-finite grad "
                    "norm is the earlier signal."
                )

    def _check_presence(self, step: int, metrics: Dict[str, float]):
        """
        Fail if a metric this suite asserts on is not being emitted at all.

        See :data:`require_present`. Without this, every ceiling below is a loop over a dict that
        may contain none of the keys it looks for, and a loop that finds nothing appends no
        failure -- so the suite passes having verified nothing, which is the precise failure mode
        the whole D2 deliverable exists to prevent.
        """
        if self._checked_presence:
            return
        self._checked_presence = True

        missing = [
            name
            for name in self.require_present
            if not any(self._is_block_key(key, name) for key in metrics)
        ]
        if missing:
            self._failures.append(
                f"required per-block metric(s) {missing} are ABSENT from the metrics at step "
                f"{step}, so the assertions that check them verified nothing. Either the "
                "producing side renamed or stopped emitting them, or the model has no MoE "
                "blocks. Metric names are registered in "
                "maple/agents/contracts/telemetry-schema.md; expected key shape is "
                "'train/block NN/<name>'. Present block keys: "
                f"{sorted(k for k in metrics if k.startswith('train/block '))[:12]}"
            )
        else:
            log.info(
                f"Metric presence check passed at step {step}: {list(self.require_present)} are "
                "all being emitted, so the bands that check them are live rather than vacuous."
            )

    def _check_step0_loss(self, step: int, metrics: Dict[str, float]):
        if self._checked_step0 or self.vocab_size is None:
            return
        loss = metrics.get(TRAIN_CE_LOSS_METRIC)
        if loss is None or not math.isfinite(loss):
            return

        # Only the FIRST logged step is checked, whatever its number. `metrics_collect_interval`
        # means the first logged step may be step 1 rather than step 0, and by step 3 the sibling
        # had already descended to 11.520 -- so a band applied to a later step is measuring
        # something else.
        self._checked_step0 = True

        if self._resumed:
            log.info(
                "Skipping the step-0 loss band: this attempt resumed from a checkpoint, so its "
                f"first logged step ({step}, loss {loss:.4f}) is not an initialisation."
            )
            return

        if step > self.step0_max_step:
            # WARNING, not a failure. A late first-logged step means the check could not be
            # applied, which is worth knowing about loudly -- but failing the run over it would
            # turn "we could not verify the init" into "the init is broken", and those are
            # different claims. See `step0_max_step`.
            log.warning(
                f"NOT applying the step-0 loss band: the first logged step is {step}, beyond "
                f"step0_max_step={self.step0_max_step}, and the loss ({loss:.4f}) no longer "
                "describes an initialisation -- the sibling fell from 11.718 to 11.520 by step 3. "
                "The init was therefore NOT verified on this run. If this is a resume, the "
                "resume guard did not fire and that is worth investigating."
            )
            return

        lo, hi = self.step0_loss_bounds
        ln_v = math.log(self.vocab_size)
        predicted = self.step0_predicted_excess
        observed_excess = loss - ln_v

        if not (lo <= loss <= hi):
            if loss < ln_v:
                direction = (
                    "BELOW ln(V), which is not reachable by an honestly-initialised model at any "
                    "width -- suspect a leaked label, a mis-tied embedding, or a resumed "
                    "checkpoint that was not detected as one"
                )
            elif predicted is None:
                direction = (
                    "above the band. For iid init logits of std s the loss is ~ln(V) + s^2/2, so "
                    "this implies an unusually large init scale; a wrong vocab is the other "
                    "candidate (ln(151936) - ln(100352) = 0.415 nats). NOTE: `d_model` was NOT "
                    "supplied, so this is the legacy fixed band, which is only correct at d=1024 "
                    "-- pass `d_model` to get the width-aware band before concluding anything is "
                    "wrong with this model"
                )
            else:
                # With the prediction subtracted off, the remaining deviation is diagnostic in a way
                # the fixed band's raw excess never was. Name the candidates by which side it is on.
                deviation = observed_excess - predicted
                if deviation > 0:
                    cause = (
                        "The init scale is larger than the config implies, or the vocab is wrong: "
                        f"a Qwen-151,936 head against this ln V contributes +0.4148 nats, and the "
                        f"deviation here is {deviation:+.4f}. Unlike the old fixed band this "
                        "comparison is width-independent, so 0.4148 means the same thing at every d"
                    )
                else:
                    cause = (
                        "The logits are FLATTER than the geometry predicts, i.e. the init is "
                        "under-scaled or has collapsed (a fully collapsed head reads deviation "
                        f"{-predicted:+.4f}), or the head's init method scales with width while "
                        "`init_scales_with_width={} was assumed".format(self.init_scales_with_width)
                    )
                direction = (
                    f"outside the WIDTH-AWARE band. Predicted excess for d_model={self.d_model} at "
                    f"init_std={self.init_std} is init_std^2*d/2 = {predicted:.4f} nats, and the "
                    f"observed excess is {observed_excess:.4f} -- a deviation of "
                    f"{observed_excess - predicted:+.4f} against a tolerance of "
                    f"+/-{self.step0_excess_tolerance_nats}. {cause}"
                )
            self._failures.append(
                f"step-0 CE loss {loss:.4f} is outside [{lo:.4f}, {hi:.4f}] for vocab "
                f"{self.vocab_size:,d} (ln V = {ln_v:.4f}) -- {direction}. The band is centered on "
                "the geometry-predicted excess, NOT on ln V: measured provenance is d=1024 -> "
                "11.718 (sibling) and d=1536 -> 11.8328 (maple-m7b-mfu-nb1), both within 0.01 nats "
                "of init_std^2*d/2. Do NOT widen this band to make a run pass -- at d=2048 the "
                "predicted excess (0.4096) and a wrong vocab (0.4148) differ by 0.005 nats, which "
                "is exactly why the fixed 0.3-nat band was replaced rather than loosened."
            )
        else:
            if predicted is None:
                log.info(
                    f"step-0 CE loss {loss:.4f} is in the LEGACY fixed band [{lo:.4f}, {hi:.4f}] "
                    f"(ln V = {ln_v:.4f}, excess {observed_excess:+.4f} nats). `d_model` was not "
                    "supplied, so this band is only correct at d=1024."
                )
            else:
                log.info(
                    f"step-0 CE loss {loss:.4f} is in band [{lo:.4f}, {hi:.4f}] (ln V = "
                    f"{ln_v:.4f}, observed excess {observed_excess:+.4f} nats against a predicted "
                    f"{predicted:.4f} for d_model={self.d_model} -- deviation "
                    f"{observed_excess - predicted:+.4f}, tolerance "
                    f"+/-{self.step0_excess_tolerance_nats})."
                )

    def _check_bands(self, metrics: Dict[str, float]):
        # Asserted on the per-block series, NOT on the cross-block aggregate. A ceiling checked
        # only against a mean is satisfied by eleven healthy blocks carrying one dead one -- and
        # the sibling's worst drop rate (0.0455) and worst entropy deficit (0.0663) were both at
        # block 23, i.e. concentrated in one block rather than spread. Per-block is the whole
        # point of B2.
        checks: List[Tuple[str, Optional[float], str]] = [
            (
                # THE GLOBAL POPULATION, NOT THE RANK-LOCAL ONE. An expert is dead if *no rank*
                # routed to it, so the `== 0` band has to be evaluated on the reduced counts. The
                # per-rank `dead_expert_frac` reads "idle on this rank" and over-reports -- L2
                # measured 0.0004 against a true global 0.0000 at 256 tokens/rank, and it
                # over-reports hardest at small local batches, which is R3's regime exactly. A
                # zero-ceiling band on the per-rank metric would therefore fire spuriously at the
                # flagship, i.e. it would be an assertion that fails on healthy runs.
                "dead_expert_frac_global",
                self.dead_expert_frac_max,
                "an expert receiving zero assignments FROM ANY RANK is capacity bought and not "
                "used; sibling measured 0.000000 across all 16 blocks, so zero is the real "
                "expectation. Check `dead_expert_counts_not_reduced` -- if it is 1.0 on a "
                "multi-rank run the collective did not run and this number is rank-local",
            ),
            (
                "entropy_deficit",
                self.entropy_deficit_max,
                "routing has collapsed toward a subset of experts",
            ),
            (
                "drop_frac",
                self.drop_frac_max,
                "assignments are being discarded by the capacity bound before reaching an "
                "expert; the loss gets worse and nothing else says why",
            ),
            (
                "drop_frac_upper_bound",
                self.drop_frac_max,
                "as drop_frac, but this is the accumulated-histogram upper bound",
            ),
            (
                "capacity_factor_deficit",
                self.capacity_factor_deficit_max,
                "the dispatch allocated LESS capacity than the config requested. "
                "`ensure_multiple_of` rounds UP, so a positive deficit cannot come from "
                "quantization -- the capacity path is not doing what the config says. Read "
                "`effective_capacity_factor` against `configured_capacity_factor` to see by how "
                "much, and note the assertion is `effective >= configured`, never equality: the "
                "realized factor legitimately exceeds the requested one",
            ),
        ]
        # NOT IN THAT LIST, ON PURPOSE: `drop_by_position` and `drop_by_doc_index`.
        #
        # The intuitive assertion -- "drops should be positionally uniform" -- is WRONG, and L3
        # measured it wrong rather than arguing it. Under forced overflow the positional spread was
        # 0.107 (5.46 sigma) and the document-index spread 0.363 (18.5 sigma). The positional tilt
        # is REAL SIGNAL: the last surviving document is truncated mid-document, so it contributes
        # its early positions and not its late ones, which puts a genuine +0.047 ramp into the
        # final quarter of a healthy run. A gate asserting flatness there would fire on every
        # healthy run and train everyone to ignore this suite.
        #
        # The alarm condition is a positional ramp LARGE RELATIVE TO `drop_by_doc_index`, not
        # non-flatness -- and nobody has a measurement of that ratio on a real run yet. So these
        # are aggregated and read, not gated. Gate them when there is a number.
        for metric, ceiling, why in checks:
            if ceiling is None:
                continue
            for key, value in metrics.items():
                if not self._is_block_key(key, metric):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value) and value > ceiling:
                    self._failures.append(
                        f"'{key}' = {value:.6f} exceeds its ceiling of {ceiling} -- {why}."
                    )

        # INSTRUMENTATION PRESENCE, BY CONVENTION RATHER THAN BY LIST. See `assert_instrumented`.
        #
        # Matched on a SUFFIX, so any component that can tell it was not instrumented declares it
        # by emitting `<name>_unavailable = 1.0` and is gated with no edit here. That is the
        # structural fix for the B3 failure: the previous presence check protected the three names
        # it listed and was silent about the histograms, so "never measured" and "measured healthy"
        # were indistinguishable on a run that had already been paid for.
        if self.assert_instrumented:
            for key, value in metrics.items():
                bare = key.rsplit("/", 1)[-1]
                if not bare.endswith(self.unavailable_metric_suffix):
                    continue
                if not self._is_block_key(key, bare):
                    continue
                if isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0:
                    instrument = bare[: -len(self.unavailable_metric_suffix)]
                    self._failures.append(
                        f"'{key}' = {value:.6f} -- the '{instrument}' instrument reports itself "
                        "UNAVAILABLE, so any band that reads it verified nothing. This is not a "
                        "measurement of health; it is the absence of a measurement. For "
                        "drop_histograms specifically: `drop_accounting_seq_len` must be set on "
                        "each ParallelMLP, or the dropless path is in use and the histograms "
                        "cannot be formed."
                    )

        if self.gate_mass_tolerance is not None:
            lo = 1.0 - self.gate_mass_tolerance
            hi = 1.0 + self.gate_mass_tolerance
            for key, value in metrics.items():
                if not self._is_block_key(key, "gate_mass_mean"):
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                if not (lo <= value <= hi):
                    # `normalize_expert_weights` is a NORM ORDER p, not a boolean, and the two
                    # ways to get this wrong land on opposite sides of the band -- so name both
                    # rather than making the reader guess which one they hit.
                    if value < lo:
                        cause = (
                            "BELOW band, which means `normalize_expert_weights` is almost "
                            "certainly unset: its stock default is `None`, i.e. no normalisation "
                            "at all, and zero of five shipped recipes set it. The sibling track "
                            "MEASURED 0.161 against an intended 1.000 -- a 6.2x error that "
                            "trains happily and shows up only as a quietly worse loss. Note the "
                            "unnormalised mass is ~k/E at init, so this failure gets LARGER up "
                            "the ladder: 2/8 = 0.25 at the sibling's E=8/k=2 against 8/256 = "
                            "0.031 at R3, a 32x error."
                        )
                    else:
                        cause = (
                            "ABOVE band. Most likely `normalize_expert_weights` is set to a norm "
                            "order other than 1.0. Dividing by the L2 norm (p=2.0, which is the "
                            "value `MoERouter`'s own docstring uses as its example) makes the L2 "
                            "norm 1.0, NOT the sum -- the row-sum is then sqrt(k) for equal "
                            "weights, i.e. 2.83 at k=8, independent of E. Only p=1.0 makes the "
                            "gate mass 1.0."
                        )
                    # THE FIX IS NAMED IN EVERY BRANCH, not just the one that seemed likeliest
                    # when this was written. A failure message that diagnoses the symptom without
                    # naming the knob costs whoever reads it at 3am a source dive, and both
                    # branches here have the same one-line remedy.
                    self._failures.append(
                        f"'{key}' = {value:.6f} is outside [{lo:.4f}, {hi:.4f}]. {cause} "
                        "REQUIRED FIX: set `normalize_expert_weights=1.0` on the router config "
                        "(this is Maple's `norm_topk_prob`, and `ladder-and-factory.md` mandates "
                        "it). Note it is a norm ORDER, not a boolean -- only p=1.0 makes the "
                        "gate mass 1.0."
                    )

    def _check_cv_band(self, metrics: Dict[str, float]):
        """
        The raw-CV balance ceiling. **Separate from :meth:`_check_bands` because its window is.**

        ON RAW ``expert_load_cv``, NEVER ON ``expert_load_cv_excess``. See
        :data:`expert_load_cv_max` for the full argument; the short form is that ``cv_excess`` is
        ``CV * sqrt(T*N*k/(E-1))`` and therefore **a longer logging interval alone would trip this
        ceiling on a physically identical model** (D-075). Raw CV has no window dependence.
        """
        if self.expert_load_cv_max is None:
            return
        ceiling = self.expert_load_cv_max
        for key, value in metrics.items():
            if not self._is_block_key(key, "expert_load_cv"):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue
            if value <= ceiling:
                continue
            self._failures.append(
                f"'{key}' = {value:.6f} exceeds the raw expert-load CV ceiling of {ceiling} -- "
                "expert load is persistently uneven. THE CEILING IS A PRE-REGISTERED DECISION "
                "BOUNDARY, not a tolerance: maple/plan/balance-floor.md fixed <=0.40 = retire "
                "balance as a risk, (0.40, 0.55] = adopt capacity_factor 2.5, >0.55 = ESCALATE "
                "because balance is architectural and the remaining levers break Maple "
                "faithfulness. So crossing this means the escalation branch, and the next step is "
                "that decision rather than a band edit. Healthy measured block-max on this stack is "
                "0.3512 (X1 tail, spread 0.1755-0.3512 over 8 blocks). NOTE all healthy data is "
                "E=64 and the flagship is E=256, where the persistent term is unmeasured -- if "
                "this fires on an E=256 run, probe P1's reading is the thing to check first. NOT "
                "redundant with `drop_frac`: at capacity_factor=2.0 an expert drops nothing until "
                "it exceeds 2x mean, so drops fell 1000x while CV fell only 3.9x on one measured "
                "run -- `drop_frac` is a thresholded view of balance, not a measure of it. Do NOT "
                "diagnose this from `expert_load_cv_excess`, which is raw CV times a "
                "window-dependent constant; use `expert_load_cv_excess_per_micro_batch` for the "
                "E-normalised magnitude. Do NOT recalibrate against Wang et al. 2408.15664 Table 2 "
                "-- those figures (0.72/0.52/0.937) are MaxVio = max/mean - 1, NOT CV."
            )


@dataclass
class ResultProtocolCallback(Callback):
    """
    Prints ``RESULT <json>`` to stdout on rank 0, **incrementally**, and again on close.

    THE SCAR. A sibling run (``run_019fd3eb-dd0a-70bf-ba71-1141eca2c2f8``, exit 72) completed
    every one of its 1,525 steps, all six eval rungs and its step-1500 checkpoint, and
    **published nothing readable**: its only structured output was a summary printed after
    ``Trainer.fit()`` returned, and ``fit()`` did not return. The science was intact and
    unreachable.

    So this callback holds two properties that the summary did not:

    1. **Incremental.** A line goes out at every metrics interval. A run killed at step 900 has
       900 steps of readable record, not zero.
    2. **Printed from ``close()`` as well**, and ``close()`` is called by ``Trainer._shutdown``
       on **both** the graceful and the ungraceful path (``_shutdown(gracefully=False)`` still
       runs every callback's ``close()``), i.e. it survives the exception path that skipped the
       summary.

    ``RESULT``/``PARAM_LEDGER`` on stdout is the convention in ``probes/train_probe.py``. Stdout
    rather than a file because the platform captures the log stream, and a checkpoint an IAM
    policy will not let you list is not an output.
    """

    metrics_to_capture: Tuple[str, ...] = (
        TRAIN_CE_LOSS_METRIC,
        OPTIM_GRAD_NORM_METRIC,
        "train/PPL",
        "train/load balancing loss",
        "train/router Z loss",
        "moe/*",
        "throughput/device/MFU",
        "throughput/device/MFU (actual avg)",
        "throughput/device/TPS",
        "throughput/total tokens",
        "optim/LR*",
    )
    """
    Glob patterns for the metrics carried on each ``RESULT`` line.

    A filtered set rather than everything: at E=256 and L=12 the full metric dict is hundreds of
    keys per step, and a stdout protocol that floods the log is one nobody can read back -- which
    is the failure being fixed, in a different costume. The per-block series stays in W&B and
    ``metrics.json``; the aggregates and the headline scalars come through here.
    """

    tag: str = "RESULT"
    """Line prefix. ``PARAM_LEDGER`` is L1's and is emitted from the factory, not here."""

    run_id: Optional[str] = None
    rung: Optional[str] = None
    enabled: bool = True

    _first_step: Optional[int] = field(default=None, repr=False)
    _first_loss: Optional[float] = field(default=None, repr=False)
    _last_step: int = field(default=0, repr=False)
    _last_loss: Optional[float] = field(default=None, repr=False)
    _last_metrics: Dict[str, float] = field(default_factory=dict, repr=False)
    _started: float = field(default_factory=time.monotonic, repr=False)
    _lines: int = field(default=0, repr=False)

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not self.enabled or get_rank() != 0:
            return

        captured = {
            key: value
            for key, value in metrics.items()
            if any(fnmatch(key, pattern) for pattern in self.metrics_to_capture)
        }
        loss = metrics.get(TRAIN_CE_LOSS_METRIC)
        if loss is not None:
            if self._first_loss is None:
                self._first_loss = float(loss)
                self._first_step = step
            self._last_loss = float(loss)
        self._last_step = step
        self._last_metrics = captured
        self._emit(outcome="in_progress", step=step, metrics=captured)

    def close(self):
        # Called by `Trainer._shutdown` on BOTH the graceful and the ungraceful path, which is
        # exactly why the final line lives here rather than in `post_train`. `post_train` is
        # skipped when `fit()` raises -- and a run that dies is the run whose result matters most.
        if not self.enabled or get_rank() != 0:
            return
        self._emit(outcome="final", step=self._last_step, metrics=self._last_metrics)

    def _emit(self, *, outcome: str, step: int, metrics: Dict[str, float]):
        record: Dict[str, Any] = {
            "run_id": self.run_id,
            "rung": self.rung,
            "outcome": outcome,
            "step": step,
            "first_step": self._first_step,
            "first_loss": self._first_loss,
            "last_loss": self._last_loss,
            "wall_seconds": round(time.monotonic() - self._started, 1),
            "metrics": metrics,
        }
        # `default=str` so an unexpected non-serialisable value degrades to a string rather than
        # raising out of a callback and killing a run over a logging line. This is the one place
        # in this module where swallowing beats raising: the assertions are the gate, and this is
        # the thing that reports what the gate saw.
        print(f"{self.tag} " + json.dumps(record, default=str), flush=True)
        self._lines += 1
