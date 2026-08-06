"""Full-model inference latency for the three P1 gate structures, on one A100.

WHAT THIS ANSWERS, AND WHY IT IS THE LAST MISSING NUMBER
  The 1B-token grid answered P1's QUALITY question and returned a well-powered null:
  ``F-r128`` and ``G-grouped`` reach the same held-out cross-entropy as dense ``L0`` while
  removing 15,728,640 parameters (75% of the gate budget, 4.03% of the model). A null on
  quality is only interesting if the parameters bought something, so the claim is an
  EFFICIENCY claim and it rests on a latency number this project does not yet have.

THE EFFECT CEILING IS 4.03%, AND EVERY DESIGN CHOICE BELOW FOLLOWS FROM THAT
  Read this before changing anything. 15,728,640 params x 2 B = 31.5 MB of the model's
  780.3 MB, so a PERFECTLY weight-bandwidth-bound decode step can improve by at most 4.03%.
  Prefill's ceiling is lower still, about 3.6%, because the saved FLOPs (31.4 MFLOP/token
  across 10 layers) sit against roughly 880 MFLOP/token total. The predicted value is ~1.8%.

  So this harness must resolve an effect of one to two percent. Ordinary GPU benchmarking
  noise -- clock drift, thermal ramp, kernel-launch overhead -- is FIVE TO TWENTY percent.
  A measurement that does not actively control those does not produce a small number with
  error bars; it produces a confound with a plausible sign. Three earlier versions of this
  file did exactly that, in the same direction, and one of them was believed for a day.

  Everything unusual here -- interleaved rounds, the A/A arm, paired ratios, the utilization
  floor -- exists because 1.8% is smaller than the noise it has to be extracted from.

WHAT WAS MEASURED BEFORE, AND WHY NONE OF IT IS QUOTABLE
    1. ``probes/p1_launch_bench.py`` held 40 MiB of gate weights against the L40S's 96 MiB L2
       and replayed a CUDA graph with nothing to evict them. It reported low-rank as 8.2%
       SLOWER. RETRACTED 2026-08-01.
    2. ``probes/p1_cache_check.py`` / ``p1_scaled.py`` fixed the residency regime and flipped
       the sign: past L2, ``F-r128`` is +29.9% to +40.4% and ``G-grouped`` is +46.1% to
       +54.4%. Both still time SEVEN LINEAR LAYERS IN A LOOP, not a model.

  Share-weighting a +40% subgraph win by the ~5.4% of the model the gates hold predicts
  ~+1.8% end-to-end. That is arithmetic. This run measures it.

HOW THE 1.8% IS PROTECTED, WHICH IS THE ACTUAL DESIGN
  * **Interleaved randomized rounds, not arm-at-a-time.** All arms stay resident (3 x 780 MB
    of weights is nothing against 40 GB) and each round times every arm back-to-back in a
    shuffled order. An A100-SXM4 takes 30-60 s to reach thermal steady state and its SM clock
    swings 4-10% under sustained bf16 load. Measuring one arm to completion and then the next
    puts ~20 s between the baseline and the last arm -- squarely inside the thermal transient,
    and biased AGAINST whichever arm runs last. Inside a round the drift is common-mode and
    divides out of the ratio.
  * **Paired ratios with a bootstrap CI, not a difference of medians.** Each round yields
    ``t_arm / t_L0`` measured seconds apart. The CI is over those ratios.
  * **A soak before timing**, so the card is already at its steady clock.
  * **An A/A control arm.** ``L0`` is built TWICE under two names. Its measured "effect" is
    known to be exactly zero, so the interval around it IS the rig's resolution. If the A/A
    interval does not exclude 1.8%, the rig cannot see the effect and the run says so instead
    of reporting a number. This is the one guard that can fail for the right reason.
  * **Clock and temperature recorded per arm**, and a refusal if mean SM clock differs across
    arms by more than a threshold. Locking clocks needs root and is not available in a Batch
    container, so interleaving is the primary defence and this is the receipt. **An unreadable
    NVML is itself a refusal** (``--allow-unmeasured-clocks`` to override): ``pynvml`` is not in
    the research image, so without that refusal every clock reads ``None``, the spread is
    ``None``, and the guard is skipped on every run -- the same disease as the ceiling below, on
    the receipt this design calls its evidence.
  * **A utilization FLOOR, not a cache ceiling.** See below -- this replaces a guard that
    could never fire.

  Every one of those bypasses is off by default and each is asserted so in the tests. A
  default-on bypass is the same as no check.

WHERE THE GUARDS LIVE, AND WHY THEY ARE NOT INLINE
  ``utilization``, ``clock_spread_pct``, ``resolution_verdict`` and ``summarise_profile`` are
  module-level functions rather than code inside ``main``. That is not tidiness. Inline, each
  could only run on a GPU with NVML present, so the only test possible was one that re-derived
  the formula on invented literals -- and such a test passes when the comparison is inverted,
  when the two peak figures are swapped, or when the field it reads is deleted. Extracted, a
  CPU test calls the same code the run calls.

  The argv checks in ``main`` run BEFORE the torch import for the same reason. While they sat
  below the CUDA check they were unreachable on every CPU host: two tests asserted them through
  ``main`` and were in fact passing on the "no CUDA device" return, so deleting either guard
  left both green.

WHY THE OLD CACHE-RESIDENCY GUARD IS GONE
  An earlier version failed a row whose achieved bandwidth exceeded HBM peak, on the theory
  that this proves cache residency. **That check is arithmetically unreachable here and it
  printed as a pass.** For 780.3 MB of weights to exceed 1555 GB/s the step would have to
  finish in under 0.50 ms; the fastest shape measured is ~0.7 ms graphed and ~2.5 ms eager.
  Every row would have read False and the summary would have said "0 rows exceeded peak",
  which reads exactly like a check that ran and passed.

  It was also the wrong direction. A 744 MiB model against a 40 MiB L2 is 18.6x over and
  CANNOT be cache-resident -- so the risk here is not that the region was too fast, it is that
  it was too SLOW: launch-starved or clock-throttled. So the guard is now a floor on achieved
  utilization, which fails in the direction the failure actually lies.

  The bandwidth ratio is still REPORTED, because it is informative, but only for decode rows
  where weights genuinely dominate traffic. In prefill it is meaningless: at batch 4 and
  sequence 4096 the logits tensor alone is 3.29 GB written, four times the whole weight
  footprint, and real traffic is 15-20 GB against the 0.78 GB the ratio counts. Quoting a
  weights-only ratio there understates by 20x. Prefill rows print ``n/a``.

WHY EAGER AND GRAPHED ARE BOTH REPORTED FOR DECODE
  At seq_len=1 a 16-layer forward is ~350-450 kernel launches, and with a synchronize before
  each timed region nothing hides the dispatch cost: ~2.5-3 ms of launch time against ~1-2 ms
  of GPU work, so eager decode measures Python dispatch rather than bandwidth.

  **And the treatments ADD launches**, which is why this is not a neutral choice. MEASURED on
  L40S 2026-08-06 (job 1676753), at decode b=1:

    L0          784 kernels, 189 of them copy-ish
    F-r128      803  (+19)   189
    G-grouped   793   (+9)   189
    L0-aa       783   (-1)   189   <- the A/A arm, i.e. run-to-run noise is about 1 kernel

  **Two predictions in an earlier version of this docstring were WRONG, and the corrected
  numbers are above.** It said +40 and +50 launches, reasoning from ``_GateProj.forward`` that
  each factorized gate needs a materializing copy for its strided slice. The real deltas are
  +19 and +9, and the copy-kernel count is IDENTICAL (189) on all four arms including the
  baseline -- so the extra kernels are the extra GEMMs and nothing is copying more than dense
  does. The "this may be an implementation artifact rather than a structural cost" caveat
  elsewhere in this file was therefore aimed at a cost that does not exist; it is retained only
  as a statement about GEMM count.

  The effect on eager decode is real regardless, and larger than predicted per launch: eager
  measures **-2.3% to -5.4%** against the treatments while graphed measures **-1.1% to +1.1%**.
  Same arms, same rounds, same host. An eager-only run would have reproduced the retracted
  -8.2% sign for an unrelated reason.

  Graph replay collapses launch cost, which is what a served system does. Both are reported and
  the graphed one is the headline. Graph replay is safe here precisely because the model cannot
  be cache-resident.

  A per-arm kernel launch count is recorded outside the timed region, so an eager row is
  interpretable rather than merely small.

THE BYTE SAVING DOES NOT CONVERT ONE-FOR-ONE, WHICH IS THE REAL FINDING TO EXPECT
  Both treatments remove exactly 4.03% of weight bytes. On L40S graphed decode at b=1 that
  bought 0.52% for ``F-r128`` and 1.19% for ``G-grouped`` -- **13% and 29% of the byte saving**.
  Achieved bandwidth on those rows is 41-43% of HBM peak, not 90%, so the step is not purely
  weight-bandwidth-bound and a byte saving cannot be collected in full.

  Read the ceiling accordingly: 4.03% is what a PERFECTLY bandwidth-bound step would give, and
  the fraction of it actually realised is itself one of the results. Do not quote the ceiling as
  a prediction.

DECODE USES logits_to_keep=1, WHICH REAL SERVING ALSO DOES
  Without it the head computes logits for every position: 2 x 16384 x 1024 x 100352 = 3.37
  TFLOP, ~23% of prefill cost, identical across arms. Arm-invariant work sits in both
  numerator and denominator and shrinks the observable delta toward zero. Prefill reports the
  delta BOTH ways so the size of that dilution is visible rather than assumed.

WHAT THIS CANNOT ANSWER
  Decode here is a seq_len=1 forward and NOT an autoregressive step: ``ShortConv`` has no
  conv-state cache and this config's attention runs without a KV cache. An earlier draft of
  this file called the decode figure an "upper bound" on served decode, reasoning that omitted
  cache traffic dilutes the ratio. **That claim was wrong and is withdrawn**: launch overhead
  pushes the other way and is larger, so the decode delta is neither an upper nor a lower
  bound. It is the delta for a cacheless seq_len=1 step, and that is all it is.

  A negative result may belong to this IMPLEMENTATION rather than to gate structure -- but note
  the measured copy counts above are IDENTICAL across all four arms, so the "extra copies"
  version of that concern is falsified. What remains is GEMM count: the treatments run more,
  smaller matmuls, and a fused kernel would not. That is a property of the factorization, not of
  how it happens to be coded here.

  And nothing here is the A100 answer. L40S is 864 GB/s with 96 MiB of L2; A100-SXM4-40GB is
  1555 GB/s with 40 MiB. Different bandwidth, different cache, different fraction of the byte
  saving realised. This file's numbers are a rehearsal of the METHOD.

  srun -p gpu --gres=gpu:1 -c 8 --mem=64G -t 00:40:00 python .edullm/bench_gate_latency.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

log = logging.getLogger("bench_gate_latency")

# ---------------------------------------------------------------------------------------------
# Cards. The peak figures are vendor HBM numbers and the FLOPs figures are dense bf16
# tensor-core peaks. Both are denominators for the UTILIZATION FLOOR: a row far below the floor
# is launch-bound or throttled and is not a valid latency comparison.
#
# A missing card is an ERROR rather than a default. `speed_monitor.py` has already shipped an
# MFU number inflated 1.175x by a missing per-card entry, and a guard whose denominator is
# guessed is a guard that reports whatever the guess implies.
# ---------------------------------------------------------------------------------------------
HBM_PEAK_GBS: Dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A100-SXM4-80GB": 2039.0,
    "NVIDIA A100-PCIE-40GB": 1555.0,
    "NVIDIA L40S": 864.0,
    "NVIDIA L4": 300.0,
    "NVIDIA A10G": 600.0,
    "Tesla T4": 320.0,
    "NVIDIA H100 80GB HBM3": 3350.0,
}

#: Dense bf16 tensor-core peak, TFLOP/s, without sparsity.
BF16_PEAK_TFLOPS: Dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 312.0,
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100-PCIE-40GB": 312.0,
    "NVIDIA L40S": 362.0,
    "NVIDIA L4": 121.0,
    "NVIDIA A10G": 125.0,
    "Tesla T4": 65.0,
    "NVIDIA H100 80GB HBM3": 989.0,
}

#: L2 sizes, printed so a reader can see the residency margin rather than infer it. The model
#: is 744 MiB in bf16, so every entry here is far exceeded -- which is why cache residency is
#: not the risk in this configuration and the guard is a floor rather than a ceiling.
L2_MIB: Dict[str, float] = {
    "NVIDIA A100-SXM4-40GB": 40.0,
    "NVIDIA A100-SXM4-80GB": 40.0,
    "NVIDIA A100-PCIE-40GB": 40.0,
    "NVIDIA L40S": 96.0,
    "NVIDIA L4": 48.0,
    "NVIDIA A10G": 6.0,
    "Tesla T4": 4.0,
    "NVIDIA H100 80GB HBM3": 50.0,
}

#: The baseline, then the two treatments, then the A/A control. ``L0-aa`` is ``L0`` built a
#: second time under a second name: its true effect is exactly zero, so the interval around it
#: measures the rig rather than the architecture.
BASELINE_ARM = "L0"
CONTROL_ARM = "L0-aa"
ARM_NAMES: Tuple[str, ...] = (BASELINE_ARM, "F-r128", "G-grouped", CONTROL_ARM)

#: What each arm name builds. The control maps to the same declared arm as the baseline.
ARM_SOURCE: Dict[str, str] = {CONTROL_ARM: BASELINE_ARM}

#: Prefill shapes as ``(batch_size, seq_len)``. 4096 is the training sequence length.
PREFILL_SHAPES: Tuple[Tuple[int, int], ...] = ((1, 4096), (4, 4096))

#: Decode batch sizes at ``seq_len=1``. Batch 1 is the latency-critical single-stream case.
DECODE_BATCHES: Tuple[int, ...] = (1, 8, 32)

#: The effect this rig has to resolve, in percent. The A/A arm's interval is compared against
#: it, and a rig that cannot separate the two reports that instead of a number.
TARGET_RESOLUTION_PCT = 1.8

#: Minimum share of the card's peak a row must achieve to be a valid latency comparison.
#: Prefill is compute-bound so it is judged on FLOPs; graphed decode is memory-bound so it is
#: judged on bandwidth. Eager decode is EXEMPT and marked as such -- it is expected to be
#: launch-bound, which is the reason the graphed row exists.
MIN_PREFILL_FLOPS_UTIL_PCT = 25.0
MIN_DECODE_BW_UTIL_PCT = 30.0

#: Largest tolerated spread in mean SM clock across arms, in percent. Above this, interleaving
#: failed to make drift common-mode and the deltas are not trustworthy.
MAX_CLOCK_SPREAD_PCT = 2.0


@dataclass
class Sample:
    """One timing of one (arm, shape, mode), inside one round."""

    round_index: int
    arm: str
    regime: str
    mode: str  # 'eager' or 'graphed'
    batch_size: int
    seq_len: int
    ms: float


@dataclass
class ArmInfo:
    """Static facts about one arm, measured once and outside any timed region."""

    arm: str
    source_arm: str
    params_total: int
    weight_bytes: int
    conv_path: str
    gate_structure: str
    launch_count: Optional[int] = None
    copy_kernel_count: Optional[int] = None
    flops_per_token_4096: Optional[int] = None
    sm_clock_mhz_mean: Optional[float] = None
    temperature_c_max: Optional[float] = None


@dataclass
class Cell:
    """The aggregated result for one (arm, regime, shape, mode), with its paired interval."""

    arm: str
    regime: str
    mode: str
    batch_size: int
    seq_len: int
    rounds: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    #: Median of the per-round ratio ``t_arm / t_baseline``. 1.0 means identical.
    ratio_median: Optional[float] = None
    #: Percent faster than baseline. Positive is faster. Derived from ``ratio_median``.
    vs_baseline_pct: Optional[float] = None
    #: Bootstrap 95% CI on ``vs_baseline_pct``.
    ci_low_pct: Optional[float] = None
    ci_high_pct: Optional[float] = None
    achieved_gbs: Optional[float] = None
    pct_of_hbm_peak: Optional[float] = None
    achieved_tflops: Optional[float] = None
    pct_of_flops_peak: Optional[float] = None
    utilization_ok: Optional[bool] = None
    utilization_note: str = ""


def _distinct_parameters(model):
    """Yield each parameter once, deduplicating tied weights.

    Keyed on ``id(p)``, which is what ``liv_arms._count_params`` uses to produce the frozen
    ledger figures (390,135,552 for ``L0``). Keying on ``data_ptr()`` instead would look more
    physical and be WRONG in two ways: every parameter on a meta device reports address 0, so
    a meta-built model would collapse to a single entry, and two genuinely distinct tensors
    that happen to view one storage would silently merge.
    """
    seen = set()
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        yield p


def weight_bytes(model) -> int:
    """Bytes of distinct parameter storage a single forward pass reads.

    The embedding is tied to the unembedding in every arm. Double-counting it would add ~205
    MiB of phantom traffic to all arms equally -- which leaves the RATIO unchanged but inflates
    every utilization figure, so a launch-bound row could clear the floor.
    """
    return sum(p.numel() * p.element_size() for p in _distinct_parameters(model))


def count_params(model) -> int:
    """Distinct parameters, tied weights counted once.

    Must agree with ``liv_arms._count_params`` so the number printed beside the latency table
    is the one the frozen ledger and the training runs recorded.
    """
    return sum(p.numel() for p in _distinct_parameters(model))


def realised_conv_path(model) -> str:
    """Which convolution implementation the LIV layers will actually execute.

    Reads the dispatch condition out of the built modules rather than assuming it, because the
    condition includes ``has_fla()`` -- environment state, not a property of the arm. Raises if
    the layers disagree, since a model where some LIV layers fuse and others do not is not a
    configuration anybody chose.
    """
    from olmo_core.nn.attention.flash_linear_attn_api import has_fla
    from olmo_core.nn.attention.short_conv import ShortConv

    paths = set()
    for module in model.modules():
        if isinstance(module, ShortConv):
            paths.add("fla" if (module.use_fla and has_fla()) else "nn.Conv1d")
    if not paths:
        raise RuntimeError(
            "no ShortConv layers found in the built model -- the per-layer mixer override did "
            "not apply. Note the field is `sequence_mixer`, NOT `attention`: setting "
            "`.attention` on a block config silently creates a new attribute and leaves every "
            "layer as attention, producing a model that runs and answers a different question."
        )
    if len(paths) > 1:
        raise RuntimeError(f"LIV layers disagree on convolution path: {sorted(paths)}")
    return paths.pop()


def gate_structure_of(model) -> str:
    """The realised gate structure, read off the built modules.

    Reported per arm so a table showing three identical structures is visibly wrong, rather
    than silently comparing dense against dense against dense.
    """
    from olmo_core.nn.attention.short_conv import ShortConv

    found = {m.gate_structure for m in model.modules() if isinstance(m, ShortConv)}
    if len(found) != 1:
        raise RuntimeError(f"expected one gate structure per arm, found {sorted(found)}")
    return found.pop()


def build_arm_config(arm_name: str, *, vocab_size: int, use_fla: bool, seed: int):
    """The ``TransformerConfig`` for one arm, with ``use_fla`` pinned on every LIV layer.

    Goes through ``arms_for_vocab`` rather than ``ARMS`` directly. ``A16-P`` and ``N-narrow``
    carry geometry SOLVED against a target that moves with the vocabulary, and ``build_arm``
    does not re-solve it.

    ``init_seed`` is set on the CONFIG because that is the only place it is read from:
    ``Transformer.init_weights`` takes no generator and seeds itself from ``self.init_seed``.
    Passing a generator would be a silent no-op via ``**kwargs``.

    The A/A control resolves through ``ARM_SOURCE`` to the baseline's declared arm and takes
    the SAME seed, so it is the identical model and its true effect is exactly zero.
    """
    from dataclasses import replace as dc_replace

    from olmo_core.nn.attention.short_conv import ShortConvConfig
    from olmo_core.nn.transformer.liv_arms import arms_for_vocab, build_arm

    source = ARM_SOURCE.get(arm_name, arm_name)
    arms = arms_for_vocab(vocab_size)
    if source not in arms:
        raise KeyError(f"unknown arm {source!r}; known: {sorted(arms)}")

    cfg = build_arm(arms[source], vocab_size=vocab_size, init_device="meta")
    cfg.init_seed = seed

    def pin(block_cfg):
        mixer = getattr(block_cfg, "sequence_mixer", None)
        if isinstance(mixer, ShortConvConfig):
            return dc_replace(block_cfg, sequence_mixer=dc_replace(mixer, use_fla=use_fla))
        return block_cfg

    cfg.block = pin(cfg.block)
    if cfg.block_overrides:
        cfg.block_overrides = {i: pin(b) for i, b in cfg.block_overrides.items()}
    return cfg


def build_one_arm(arm_name: str, *, vocab_size: int, dtype, use_fla: bool, device, seed: int):
    """Build one arm on ``device``, INITIALISED, with gates pinned identically.

    ``TransformerConfig.build`` constructs modules without initialising them, and uninitialised
    CUDA memory is often all zeros -- a kernel fed zeros or subnormals can take a different
    amount of time than one fed real values, so timing an uninitialised model times something
    other than the model.

    Order matters: ``init_weights`` calls ``to_empty(device)``, which allocates FRESH storage
    and re-establishes weight tying, so a cast before it would be thrown away.
    """
    cfg = build_arm_config(arm_name, vocab_size=vocab_size, use_fla=use_fla, seed=seed)
    model = cfg.build(init_device="meta")
    model.init_weights(device=device)
    model.to(dtype=dtype)
    model.eval()
    return model


# ---------------------------------------------------------------------------------------------
# Clocks and temperature. Recorded per arm so that "interleaving made drift common-mode" is a
# measurement rather than an assumption.
# ---------------------------------------------------------------------------------------------
def _nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def read_clock_and_temp(nv, handle) -> Tuple[Optional[float], Optional[float]]:
    if nv is None or handle is None:
        return None, None
    try:
        clock = float(nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM))
        temp = float(nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU))
        return clock, temp
    except Exception:
        return None, None


def clock_spread_pct(values: Sequence[Optional[float]]) -> Optional[float]:
    """Peak-to-peak spread of per-arm mean SM clock, as a percent of the mean.

    Extracted as a pure function so the guard that consumes it is testable on a CPU. It used to
    be four lines inline in ``main`` and could therefore only be exercised on a GPU with NVML
    present -- which is to say, never, since ``pynvml`` is not in the research image.

    Returns ``None`` when fewer than two arms reported a clock, which is a DIFFERENT condition
    from "the clocks agree" and the caller must not conflate them: unmeasured is not zero.
    """
    readable = [v for v in values if v is not None]
    if len(readable) < 2:
        return None
    return (max(readable) - min(readable)) / statistics.fmean(readable) * 100.0


#: ``num_flops_per_token`` throughout this codebase uses the **6x-params** convention, i.e.
#: forward AND backward. ``short_conv.py``'s docstring is explicit about it and warns that a
#: mixer using the 2x convention "would report a third of its true cost and silently unbalance
#: every arm". This harness runs inference only, so the forward share is one third.
#:
#: Getting this wrong is not cosmetic: leaving the 6x figure in place makes the denominator 3x
#: too large, drives every prefill row to about a third of its real utilization, and fails the
#: 25% floor on a perfectly healthy measurement. A guard that false-alarms gets flags passed to
#: silence it, and then it protects nothing.
FORWARD_ONLY_FLOPS_FRACTION = 1.0 / 3.0


def arm_flops_per_token(info: "ArmInfo", *, regime: str, seq_len: int, lm_head_flops: int) -> int:
    """Forward-only FLOPs per token for one arm on one shape.

    Two corrections over the raw ``model.num_flops_per_token(seq_len)``:

    * **Forward only.** See :data:`FORWARD_ONLY_FLOPS_FRACTION`.
    * **Per shape.** The ``prefill-lastlogit`` rows pass ``logits_to_keep=1``, so the head runs
      on one position rather than ``seq_len``. Its cost is therefore amortised across the
      sequence instead of paid per token, and charging the full-head figure would overstate
      those rows' utilization by ~1.3x -- letting them clear the floor for free.
    """
    total = info.flops_per_token_4096 or 0
    if regime == "prefill-lastlogit" and seq_len > 0:
        # The head still runs once for the whole sequence, so spread it over the tokens.
        total = total - lm_head_flops + max(1, lm_head_flops // seq_len)
    return int(total * FORWARD_ONLY_FLOPS_FRACTION)


def utilization(
    *,
    regime: str,
    mode: str,
    median_ms: float,
    weight_bytes_: int,
    flops_per_token: int,
    batch_size: int,
    seq_len: int,
    peak_gbs: float,
    peak_tflops: float,
) -> Dict[str, Any]:
    """Achieved utilization for one cell, and whether it clears the floor.

    Extracted from ``main`` for one reason: as inline code it could not be called without a
    GPU, so the only test possible was one that re-derived the formula on invented literals --
    which passes when the comparison is inverted, when the wrong peak is used as the
    denominator, or when the branches are swapped. Calling this is what makes those mutations
    fail.

    Returns a dict rather than a tuple because the caller stores six different fields from it
    and positional unpacking of six values is where the branches get crossed.

    The floor is direction-correct for each regime, which the old bandwidth CEILING was not:

    * **prefill** is compute-bound, so it is judged on FLOPs. A region below the floor is not
      compute-bound and a FLOPs saving cannot show up in it.
    * **graphed decode** is memory-bound, so it is judged on weight bandwidth.
    * **eager decode** is EXEMPT and says so. It is expected to be launch-bound -- that is the
      whole reason the graphed row exists -- so failing it would be a guard that fires on the
      designed behaviour, which trains people to pass ``--allow`` flags.

    ``flops_per_token`` must be the figure for the SHAPE being measured, not a constant. The
    ``prefill-lastlogit`` rows drop ~23% of the work by computing one position's logits instead
    of 4096, so charging them the full-head figure would overstate their utilization by ~1.3x
    and let them clear the floor too easily.
    """
    out: Dict[str, Any] = {
        "achieved_gbs": None,
        "pct_of_hbm_peak": None,
        "achieved_tflops": None,
        "pct_of_flops_peak": None,
        "utilization_ok": None,
        "utilization_note": "",
    }
    seconds = median_ms / 1e3
    if seconds <= 0:
        out["utilization_ok"] = False
        out["utilization_note"] = "non-positive elapsed time; the timing is not usable"
        return out

    if regime == "decode":
        out["achieved_gbs"] = weight_bytes_ / seconds / 1e9
        out["pct_of_hbm_peak"] = out["achieved_gbs"] / peak_gbs * 100.0
        if mode == "graphed":
            out["utilization_ok"] = out["pct_of_hbm_peak"] >= MIN_DECODE_BW_UTIL_PCT
            if not out["utilization_ok"]:
                out["utilization_note"] = (
                    f"below {MIN_DECODE_BW_UTIL_PCT:.0f}% of HBM peak: still launch-bound or "
                    f"throttled, not a bandwidth comparison"
                )
        else:
            out["utilization_note"] = "eager decode is expected launch-bound; exempt"
        return out

    out["achieved_tflops"] = flops_per_token * batch_size * seq_len / seconds / 1e12
    out["pct_of_flops_peak"] = out["achieved_tflops"] / peak_tflops * 100.0
    out["utilization_ok"] = out["pct_of_flops_peak"] >= MIN_PREFILL_FLOPS_UTIL_PCT
    if not out["utilization_ok"]:
        out["utilization_note"] = (
            f"below {MIN_PREFILL_FLOPS_UTIL_PCT:.0f}% of bf16 peak: the timed region is not "
            f"compute-bound, so a FLOPs saving cannot show up"
        )
    return out


def resolution_verdict(
    cells: Sequence["Cell"], *, control_arm: str = None, target_pct: float = None
) -> Tuple[List[str], List[str]]:
    """Read the rig's own resolution off the A/A control, and say which shapes cannot be trusted.

    One condition, on the magnitude the A/A arm could be wrong by: **the whole interval must lie
    within ``+/- target_pct`` of zero.** The control's true effect is exactly zero, so the
    furthest edge of its interval is the largest error this rig can attribute to a real arm. If
    that edge is inside the effect being hunted, the rig can see the effect; if not, it cannot.

    This subsumes two checks that were tried and are both wrong on their own:

    * **Width alone** certifies an interval like ``[+2.8, +3.3]``: narrow, and describing a rig
      with a 3pp systematic bias between two identical models.
    * **Bracketing zero** looked like the fix and FALSE-ALARMS on real data. The 2026-08-06 L40S
      rehearsal produced ``[+0.05, +0.25]`` on graphed decode -- a bootstrap interval so tight
      that a 0.17pp bias excludes zero -- and four such rows were reported unresolvable while
      every one of them is 7x smaller than the 1.8% target. A guard that fires on a measurement
      this good is a defect: somebody passes ``--allow-unresolvable`` to get past it and then it
      protects nothing.

    A tight interval that misses zero is worth SAYING, because it means the pairing has residual
    structure, so it is annotated. It is not disqualifying while it stays well inside the target.

    Returns ``(report_lines, unresolvable_shapes)``. An empty control list yields an explicit
    ``NOT MEASURED`` line and a sentinel in the unresolvable list, so a run with no floor fails
    rather than printing a reassuring absence.
    """
    control_arm = control_arm or CONTROL_ARM
    target_pct = TARGET_RESOLUTION_PCT if target_pct is None else target_pct

    lines: List[str] = []
    unresolvable: List[str] = []
    control = [
        c
        for c in cells
        if c.arm == control_arm and c.ci_low_pct is not None and c.ci_high_pct is not None
    ]
    if not control:
        lines.append(
            "  NOT MEASURED -- no A/A cell carried an interval, so no delta above has a known "
            "noise floor. This is a failure, not an absence."
        )
        unresolvable.append("<no A/A control measured>")
        return lines, unresolvable

    for c in control:
        width = c.ci_high_pct - c.ci_low_pct
        # The furthest either edge sits from zero is the largest error this rig can attribute to
        # a real arm, since the control's true effect is zero. That single number is the floor.
        worst_edge = max(abs(c.ci_low_pct), abs(c.ci_high_pct))
        resolvable = worst_edge < target_pct
        shape = f"{c.regime}/{c.mode} b={c.batch_size}"

        if resolvable:
            note = f"OK (worst edge {worst_edge:.2f}pp < {target_pct}%)"
            # Worth saying, not worth failing: a tight interval that misses zero means the
            # pairing has residual structure, even though it is far inside the target.
            if not (c.ci_low_pct <= 0.0 <= c.ci_high_pct):
                note += "; small residual bias, interval misses zero"
        else:
            note = f"CANNOT resolve {target_pct}%: worst edge {worst_edge:.2f}pp"
            unresolvable.append(shape)

        lines.append(
            f"  {shape:<30} {c.vs_baseline_pct:+6.2f}%  CI "
            f"[{c.ci_low_pct:+6.2f},{c.ci_high_pct:+6.2f}]  width {width:5.2f}pp  " + note
        )
    return lines, unresolvable


def count_launches(model, tokens, *, logits_to_keep: int) -> Tuple[Optional[int], Optional[int]]:
    """Kernel launches and copy-ish kernels for one forward, OUTSIDE any timed region.

    The launch count is what makes an eager decode row interpretable: the treatments add
    kernels (a strided slice into ``F.linear`` forces a materializing copy; ``_grouped``
    transposes into ``bmm`` and reshapes out of it), and without this integer a slower eager
    row reads as a verdict on gate structure rather than on dispatch count.

    Returns ``(None, None)`` if profiling is unavailable. The caller treats that as a defect
    rather than as an absence -- see ``main`` -- because a diagnostic that silently returns
    ``None`` produces a report with an empty column that reads as "nothing to see".
    """
    import torch

    try:
        from torch.profiler import ProfilerActivity, profile

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                model(tokens, logits_to_keep=logits_to_keep)
            torch.cuda.synchronize()
        return summarise_profile(prof.key_averages())
    except Exception as exc:  # pragma: no cover - diagnostic only
        log.warning("could not count kernel launches (%s)", exc)
        return None, None


#: Substrings that mark a kernel as a data-movement kernel rather than real work. The treatments
#: add these (a strided slice into ``F.linear``; ``_grouped``'s transpose/reshape around
#: ``bmm``), so counting them separately is what distinguishes "this structure is slower" from
#: "this implementation copies more".
_COPY_KERNEL_MARKERS = ("copy", "contiguous", "cat", "clone", "permute", "transpose")


#: The profiler's per-event device-time field, newest spelling first. ``cuda_time_total`` is the
#: pre-2.x name of ``device_time_total``; reading only one of them on the wrong torch matches
#: nothing and yields a count of zero.
_DEVICE_TIME_FIELDS = ("device_time_total", "cuda_time_total")


def summarise_profile(averages) -> Tuple[Optional[int], Optional[int]]:
    """Total device-kernel count and copy-kernel count from profiler averages.

    Split out from ``count_launches`` so it can be tested against a stub, because the field it
    reads was RENAMED between torch versions.

    **The recognised-field count is the point.** An earlier version of this function kept any row
    exposing neither field, reasoning that dropping rows silently was what produced a zero count.
    That leniency made the function unable to tell "this row has no timing" from "I am reading a
    field name that no longer exists" -- and a mutation test proved it: switching to the legacy
    spelling alone left every test passing, because the rows were kept regardless.

    So: if NO event exposes either spelling, the field has been renamed again and this returns
    ``(None, None)``, which the caller turns into a refusal. If some do, the name is right and
    rows without a positive device time are genuinely not kernels.
    """
    kernels = []
    recognised = 0
    for event in averages:
        device_time = None
        for field in _DEVICE_TIME_FIELDS:
            value = getattr(event, field, None)
            if value is not None:
                device_time = value
                recognised += 1
                break
        if device_time is not None and device_time > 0:
            kernels.append(event)
    if not recognised:
        # Either an empty profile or a third rename. Both must reach the caller as "unknown"
        # rather than as zero kernels, because zero reads as a measurement.
        return None, None
    if not kernels:
        return None, None
    total = sum(int(getattr(e, "count", 0)) for e in kernels)
    copies = sum(
        int(getattr(e, "count", 0))
        for e in kernels
        if any(marker in str(getattr(e, "key", "")).lower() for marker in _COPY_KERNEL_MARKERS)
    )
    return (total or None), copies


def make_runner(model, tokens, *, logits_to_keep: int, graphed: bool):
    """A zero-argument callable that performs one forward, eager or graph-replayed.

    Graph capture is safe here because the model is 744 MiB against a 40 MiB L2 -- 18.6x over,
    so replay cannot make the region cache-resident. That was the failure mode of the retracted
    probe, whose whole working set fit in cache, and it does not apply at full-model scale.
    """
    import torch

    def eager():
        model(tokens, logits_to_keep=logits_to_keep)

    if not graphed:
        return eager

    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                eager()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            eager()
        torch.cuda.synchronize()
    return graph.replay


def time_once(run) -> float:
    """Milliseconds for one call, with a synchronize before the start event.

    The leading synchronize keeps work queued by the previous call out of this window.
    """
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    run()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def bootstrap_ci(
    values: Sequence[float], *, rounds: int, rng: random.Random
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI on the median of ``values``.

    Bootstrap rather than a t-interval because these are ratios of timings: they are bounded
    below by zero, right-skewed, and there is no reason to expect normality at n=20 rounds.
    """
    if len(values) < 2:
        return float("nan"), float("nan")
    medians = []
    n = len(values)
    for _ in range(rounds):
        medians.append(statistics.median(rng.choices(values, k=n)))
    medians.sort()
    lo = medians[max(0, int(0.025 * rounds) - 1)]
    hi = medians[min(rounds - 1, int(0.975 * rounds))]
    return lo, hi


def percentiles(values: Sequence[float]) -> Tuple[float, float, float]:
    ordered = sorted(values)
    n = len(ordered)
    return (
        statistics.median(ordered),
        ordered[max(0, int(0.10 * n) - 1)],
        ordered[min(n - 1, int(0.90 * n))],
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI, extracted so a test can assert a DEFAULT rather than grep for one.

    The previous test for ``--use-fla``'s default scanned the source for the literal
    ``default="false"`` and set a flag on any line that matched -- never tying the match to this
    argument. Flipping this default to ``"true"`` passed, so long as some other option still
    carried ``"false"``.
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    ap.add_argument("--arms", default=",".join(ARM_NAMES))
    ap.add_argument("--baseline", default=BASELINE_ARM)
    ap.add_argument("--vocab-size", type=int, default=100_352)
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument(
        "--rounds",
        type=int,
        default=20,
        help="Interleaved rounds. Each round times every arm on every shape in a shuffled "
        "order, so clock drift is common-mode within a round and divides out of the ratio.",
    )
    ap.add_argument(
        "--iters-per-round",
        type=int,
        default=5,
        help="Timings per (arm, shape) per round, medianed before the ratio is formed.",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--soak-seconds",
        type=float,
        default=75.0,
        help="Run the baseline before timing anything, so the card is at its steady clock. An "
        "A100-SXM4 takes 30-60 s to get there and its SM clock swings 4-10% on the way.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument(
        "--use-fla",
        default="false",
        choices=("true", "false"),
        help="Pinned identically on every arm. Default false: `fla` is absent from the "
        "research image, so leaving the module default (True) would let the dispatch "
        "condition depend on the environment rather than on the declared config.",
    )
    ap.add_argument("--device-peak-gbs", type=float, default=None)
    ap.add_argument("--device-peak-tflops", type=float, default=None)
    ap.add_argument(
        "--skip-graphed",
        action="store_true",
        help="Eager only. Leaves the decode rows launch-bound, which is why this is not the "
        "default -- the treatments add kernels and eager decode ranks arms by launch count.",
    )
    ap.add_argument(
        "--allow-unresolvable",
        action="store_true",
        help="Report numbers even when the A/A control cannot separate zero from the target "
        "effect. Off by default: a rig that cannot see 1.8% must say so, not print a delta.",
    )
    ap.add_argument(
        "--allow-unmeasured-clocks",
        action="store_true",
        help="Proceed when NVML is unavailable, i.e. with no evidence that interleaving made "
        "clock drift common-mode. Off by default because pynvml is absent from the research "
        "image, which would otherwise silently disable the clock-spread guard.",
    )
    ap.add_argument(
        "--allow-missing-launch-counts",
        action="store_true",
        help="Proceed when the profiler yields no kernel counts. Off by default: without them "
        "an eager decode gap cannot be attributed to dispatch rather than gate structure.",
    )
    ap.add_argument("--out", default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    opts = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )

    # ------------------------------------------------------------------------------------
    # ARGV VALIDATION FIRST, BEFORE ANY TORCH IMPORT. This ordering is load-bearing, not
    # tidiness: these checks are pure string work, and while they sat below the CUDA check
    # they were UNREACHABLE on any CPU host. Two tests claimed to exercise them through
    # `main` and were in fact passing on the "no CUDA device" return -- deleting either
    # guard entirely left both tests green. A guard that can only run on the machine it is
    # meant to protect you from misconfiguring is not a guard.
    # ------------------------------------------------------------------------------------
    arms = [a.strip() for a in opts.arms.split(",") if a.strip()]
    if opts.baseline not in arms:
        print(
            f"ERROR: baseline {opts.baseline!r} is not in --arms ({arms}). Every ratio is "
            f"formed against it, so a run without it produces no comparisons at all.",
            file=sys.stderr,
        )
        return 1
    if CONTROL_ARM not in arms and not opts.allow_unresolvable:
        print(
            f"ERROR: the A/A control arm {CONTROL_ARM!r} is not in --arms. It is the only "
            f"thing that measures whether this rig can resolve {TARGET_RESOLUTION_PCT}%, and "
            f"without it a delta has no floor to be compared against. Add it, or pass "
            f"--allow-unresolvable to state deliberately that you are not measuring the floor.",
            file=sys.stderr,
        )
        return 1

    try:
        import torch
    except ImportError:
        print("ERROR: torch is not importable.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA device. This measures per-card inference latency and is "
            "meaningless on CPU -- a CPU number would not be a slower version of the answer, "
            "it would be a different question. Run on the platform (gpu-8xa100 with "
            "EDULLM_LAUNCH_CHECK=waived, or gpu-1xl40s) or on a FarmShare GPU node.",
            file=sys.stderr,
        )
        return 1

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    peak_gbs = opts.device_peak_gbs or HBM_PEAK_GBS.get(device_name)
    peak_tflops = opts.device_peak_tflops or BF16_PEAK_TFLOPS.get(device_name)
    if peak_gbs is None or peak_tflops is None:
        print(
            f"ERROR: no peak figures recorded for {device_name!r}. Pass --device-peak-gbs and "
            f"--device-peak-tflops. Refusing to default: these are the denominators of the "
            f"utilization floor, and a guard whose denominator is guessed reports whatever the "
            f"guess implies.",
            file=sys.stderr,
        )
        return 1

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        opts.dtype
    ]
    use_fla = opts.use_fla == "true"

    try:
        from olmo_core.nn.attention.flash_linear_attn_api import has_fla

        fla_importable = has_fla()
    except Exception:
        fla_importable = False

    # ------------------------------------------------------------------------------------
    # NVML, and why an absence here is a REFUSAL rather than a shrug.
    #
    # `pynvml` is NOT in the research image (checked against `.edullm/Dockerfile`, which
    # installs torch, `.[wandb]`, boto3 and edullm-data, and against `pyproject.toml`).
    # Without it every per-arm clock is None, the spread is None, and the clock guard is
    # skipped -- so the run would print "drift is uncontrolled-but-unmeasured" and exit 0.
    #
    # That is exactly the disease this file removed the bandwidth ceiling for: a guard that
    # cannot fire, on the receipt the design calls its evidence that interleaving worked. So
    # a missing NVML now stops the run unless the operator says otherwise in as many words.
    # ------------------------------------------------------------------------------------
    nv = _nvml()
    handle = None
    if nv is not None:
        try:
            handle = nv.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            handle = None
    if handle is None and not opts.allow_unmeasured_clocks:
        print(
            "ERROR: NVML is unavailable, so per-arm SM clock cannot be read and the "
            f"{MAX_CLOCK_SPREAD_PCT}% clock-spread guard cannot fire. Interleaving is this "
            "design's defence against a 4-10% clock swing on an effect whose ceiling is "
            "4.03%, and the clock reading is the only evidence that it worked. `pynvml` is "
            "not in the research image: add it, or pass --allow-unmeasured-clocks to record "
            "deliberately that this run has no drift receipt.",
            file=sys.stderr,
        )
        return 1

    log.info("run %s on %s (%s), %s", opts.run_id, device_name, platform.node(), opts.dtype)
    log.info(
        "resolving a %.1f%% target against a %.2f%% ceiling; %d interleaved rounds",
        TARGET_RESOLUTION_PCT,
        4.03,
        opts.rounds,
    )

    # ------------------------------------------------------------------------------------
    # Build every arm ONCE and keep them all resident. 4 x 780 MB of weights is 3.1 GB of
    # 40, and residency is what makes interleaving possible -- rebuilding per round would
    # reintroduce the ordering effect that interleaving exists to remove.
    # ------------------------------------------------------------------------------------
    models: Dict[str, Any] = {}
    info: Dict[str, ArmInfo] = {}
    lm_head_flops: Optional[int] = None
    for arm_name in arms:
        model = build_one_arm(
            arm_name,
            vocab_size=opts.vocab_size,
            dtype=dtype,
            use_fla=use_fla,
            device=device,
            seed=opts.seed,
        )
        models[arm_name] = model
        info[arm_name] = ArmInfo(
            arm=arm_name,
            source_arm=ARM_SOURCE.get(arm_name, arm_name),
            params_total=count_params(model),
            weight_bytes=weight_bytes(model),
            conv_path=realised_conv_path(model),
            gate_structure=gate_structure_of(model),
            flops_per_token_4096=model.num_flops_per_token(4096),
        )
        # The head's own share, needed to re-price the `logits_to_keep=1` rows. Read off the
        # module rather than recomputed, so a change to the head's accounting follows here.
        if lm_head_flops is None and getattr(model, "lm_head", None) is not None:
            lm_head_flops = model.lm_head.num_flops_per_token(4096)
        log.info(
            "built %-10s %s gates, %s params, %.1f MiB, conv %s",
            arm_name,
            info[arm_name].gate_structure,
            f"{info[arm_name].params_total:,}",
            info[arm_name].weight_bytes / 2**20,
            info[arm_name].conv_path,
        )

    paths = {a: i.conv_path for a, i in info.items()}
    if len(set(paths.values())) > 1:
        print(
            f"ERROR: arms executed different convolution paths: {paths}. The contrast would "
            f"compare kernels, not gate structures. Pin --use-fla and re-run.",
            file=sys.stderr,
        )
        return 1

    # The A/A control must be byte-identical to the baseline, or it is not measuring the rig.
    if CONTROL_ARM in info and opts.baseline in info:
        a, b = info[CONTROL_ARM], info[opts.baseline]
        if (a.params_total, a.gate_structure) != (b.params_total, b.gate_structure):
            print(
                f"ERROR: {CONTROL_ARM} is not identical to {opts.baseline} "
                f"({a.params_total:,}/{a.gate_structure} vs {b.params_total:,}/"
                f"{b.gate_structure}). Its whole purpose is a known-zero effect.",
                file=sys.stderr,
            )
            return 1

    # ------------------------------------------------------------------------------------
    # Inputs. Identical tokens across arms, so any difference is the model and not the data.
    # ------------------------------------------------------------------------------------
    gen = torch.Generator(device="cpu").manual_seed(opts.seed)
    plan: List[Tuple[str, str, int, int, int]] = []  # regime, mode, batch, seq, logits_to_keep
    for b, s in PREFILL_SHAPES:
        plan.append(("prefill", "eager", b, s, 0))
        plan.append(("prefill-lastlogit", "eager", b, s, 1))
    for b in DECODE_BATCHES:
        plan.append(("decode", "eager", b, 1, 1))
        if not opts.skip_graphed:
            plan.append(("decode", "graphed", b, 1, 1))

    tokens: Dict[Tuple[int, int], Any] = {}
    for _, _, b, s, _ in plan:
        if (b, s) not in tokens:
            tokens[(b, s)] = torch.randint(
                0, opts.vocab_size, (b, s), generator=gen, dtype=torch.long
            ).to(device)

    # Launch counts, once per arm, outside every timed region. A missing count is a REFUSAL
    # rather than an empty column: it is the only thing that lets an eager decode gap be
    # attributed to dispatch rather than to gate structure, and the field the profiler exposes
    # it under was renamed between torch versions -- so "None everywhere" is the expected shape
    # of that breakage and would otherwise print as nothing-to-see.
    for arm_name in arms:
        total, copies = count_launches(models[arm_name], tokens[(1, 1)], logits_to_keep=1)
        info[arm_name].launch_count = total
        info[arm_name].copy_kernel_count = copies
        if total:
            log.info(
                "%-10s %d kernel launches at decode b=1 (%s copy-ish)", arm_name, total, copies
            )
    missing_counts = [a for a in arms if info[a].launch_count is None]
    if missing_counts and not opts.allow_missing_launch_counts:
        print(
            f"ERROR: no kernel-launch count for {', '.join(missing_counts)}. The profiler "
            f"returned nothing usable -- most likely the event field was renamed again "
            f"(`cuda_time_total` became `device_time_total`), which yields None on every arm "
            f"rather than an error. Without these counts an eager decode gap cannot be "
            f"separated from gate structure. Pass --allow-missing-launch-counts to proceed "
            f"without that attribution.",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------------------------
    # Soak, so timing starts at the steady clock rather than partway up the thermal ramp.
    # ------------------------------------------------------------------------------------
    if opts.soak_seconds > 0:
        import time

        log.info("soaking %.0fs to reach steady clocks", opts.soak_seconds)
        soak_run = make_runner(
            models[opts.baseline], tokens[PREFILL_SHAPES[-1]], logits_to_keep=0, graphed=False
        )
        deadline = time.monotonic() + opts.soak_seconds
        with torch.no_grad():
            while time.monotonic() < deadline:
                soak_run()
            torch.cuda.synchronize()
        clock, temp = read_clock_and_temp(nv, handle)
        log.info("post-soak SM clock %s MHz, %s C", clock, temp)

    # ------------------------------------------------------------------------------------
    # Runners, built once so graph capture is not repeated per round.
    # ------------------------------------------------------------------------------------
    runners: Dict[Tuple[str, str, str, int, int], Any] = {}
    for arm_name in arms:
        for regime, mode, b, s, keep in plan:
            run = make_runner(
                models[arm_name], tokens[(b, s)], logits_to_keep=keep, graphed=(mode == "graphed")
            )
            with torch.no_grad():
                for _ in range(opts.warmup):
                    run()
            torch.cuda.synchronize()
            runners[(arm_name, regime, mode, b, s)] = run

    # ------------------------------------------------------------------------------------
    # The measurement. Shuffle the arm order EVERY round, so no arm sits permanently in the
    # warmest or coolest slot and any residual drift is spread across arms rather than
    # accumulated against one.
    # ------------------------------------------------------------------------------------
    rng = random.Random(opts.seed)
    samples: List[Sample] = []
    clocks: Dict[str, List[float]] = {a: [] for a in arms}
    temps: Dict[str, List[float]] = {a: [] for a in arms}

    with torch.no_grad():
        for round_index in range(opts.rounds):
            for regime, mode, b, s, _keep in plan:
                order = list(arms)
                rng.shuffle(order)
                for arm_name in order:
                    run = runners[(arm_name, regime, mode, b, s)]
                    reps = [time_once(run) for _ in range(opts.iters_per_round)]
                    samples.append(
                        Sample(
                            round_index=round_index,
                            arm=arm_name,
                            regime=regime,
                            mode=mode,
                            batch_size=b,
                            seq_len=s,
                            ms=statistics.median(reps),
                        )
                    )
                    clock, temp = read_clock_and_temp(nv, handle)
                    if clock is not None:
                        clocks[arm_name].append(clock)
                    if temp is not None:
                        temps[arm_name].append(temp)
            if (round_index + 1) % 5 == 0:
                log.info("round %d/%d", round_index + 1, opts.rounds)

    for arm_name in arms:
        if clocks[arm_name]:
            info[arm_name].sm_clock_mhz_mean = statistics.fmean(clocks[arm_name])
        if temps[arm_name]:
            info[arm_name].temperature_c_max = max(temps[arm_name])

    # ------------------------------------------------------------------------------------
    # Aggregate into cells, forming the ratio WITHIN each round.
    # ------------------------------------------------------------------------------------
    def key(sample: Sample) -> Tuple[str, str, int, int]:
        return (sample.regime, sample.mode, sample.batch_size, sample.seq_len)

    shapes = sorted({key(s) for s in samples})
    cells: List[Cell] = []
    for regime, mode, b, s in shapes:
        per_arm_round: Dict[str, Dict[int, float]] = {}
        for sample in samples:
            if key(sample) == (regime, mode, b, s):
                per_arm_round.setdefault(sample.arm, {})[sample.round_index] = sample.ms
        base_by_round = per_arm_round.get(opts.baseline, {})
        if not base_by_round:
            print(
                f"ERROR: no {opts.baseline} timings for {(regime, mode, b, s)} -- no ratio can "
                f"be formed, and reporting the row without one would read as a result.",
                file=sys.stderr,
            )
            return 1

        for arm_name in arms:
            by_round = per_arm_round.get(arm_name, {})
            if not by_round:
                continue
            med, p10, p90 = percentiles(list(by_round.values()))
            cell = Cell(
                arm=arm_name,
                regime=regime,
                mode=mode,
                batch_size=b,
                seq_len=s,
                rounds=len(by_round),
                median_ms=med,
                p10_ms=p10,
                p90_ms=p90,
            )

            if arm_name != opts.baseline:
                shared = sorted(set(by_round) & set(base_by_round))
                ratios = [by_round[i] / base_by_round[i] for i in shared]
                if ratios:
                    cell.ratio_median = statistics.median(ratios)
                    cell.vs_baseline_pct = (1.0 - cell.ratio_median) * 100.0
                    lo, hi = bootstrap_ci(ratios, rounds=opts.bootstrap, rng=rng)
                    # A ratio CI maps to a percent-faster CI with the sign flipped.
                    cell.ci_low_pct = (1.0 - hi) * 100.0
                    cell.ci_high_pct = (1.0 - lo) * 100.0

            # Utilization: the guard that can actually fire here. Computed by the extracted
            # `utilization()` so a CPU test can call the same code, rather than inline where
            # the only possible test was one that re-derived the formula.
            #
            # The FLOPs figure is PER SHAPE. `prefill-lastlogit` computes one position's
            # logits instead of 4096, which removes ~23% of the work, so charging it the
            # full-head figure would overstate its utilization by ~1.3x and let it clear the
            # floor too easily.
            fpt = arm_flops_per_token(
                info[arm_name], regime=regime, seq_len=s, lm_head_flops=lm_head_flops
            )
            util = utilization(
                regime=regime,
                mode=mode,
                median_ms=med,
                weight_bytes_=info[arm_name].weight_bytes,
                flops_per_token=fpt,
                batch_size=b,
                seq_len=s,
                peak_gbs=peak_gbs,
                peak_tflops=peak_tflops,
            )
            for attr, value in util.items():
                setattr(cell, attr, value)
            cells.append(cell)

    # ------------------------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------------------------
    print(f"\ndevice: {device_name}")
    print(
        f"HBM peak {peak_gbs:.0f} GB/s, bf16 peak {peak_tflops:.0f} TFLOP/s, "
        f"L2 {L2_MIB.get(device_name, float('nan')):.0f} MiB, dtype {opts.dtype}, "
        f"torch {torch.__version__}"
    )
    print(
        f"{opts.rounds} interleaved rounds x {opts.iters_per_round} timings, arm order "
        f"reshuffled each round; {opts.soak_seconds:.0f}s soak; "
        f"use_fla pinned {use_fla} (fla importable: {fla_importable})"
    )

    print("\narms:")
    for arm_name in arms:
        i = info[arm_name]
        print(
            f"  {arm_name:<10} {i.gate_structure:<8} {i.params_total:>13,} params  "
            f"{i.weight_bytes / 2**20:7.1f} MiB  launches={i.launch_count}  "
            f"copies={i.copy_kernel_count}  clock={i.sm_clock_mhz_mean}  "
            f"tmax={i.temperature_c_max}"
        )

    print(
        f"\n{'arm':<10} {'regime':<18} {'mode':<8} {'b':>3} {'seq':>5} {'median ms':>10} "
        f"{'vs base':>9} {'95% CI':>18} {'util':>8}  flag"
    )
    for c in cells:
        delta = "  baseline" if c.vs_baseline_pct is None else f"{c.vs_baseline_pct:+8.2f}%"
        ci = (
            "".ljust(18)
            if c.ci_low_pct is None
            else f"[{c.ci_low_pct:+6.2f},{c.ci_high_pct:+6.2f}]".rjust(18)
        )
        util = (
            f"{c.pct_of_hbm_peak:6.1f}%"
            if c.pct_of_hbm_peak is not None
            else (f"{c.pct_of_flops_peak:6.1f}%" if c.pct_of_flops_peak is not None else "   n/a")
        )
        flag = "" if c.utilization_ok in (True, None) else " LOW_UTIL"
        print(
            f"{c.arm:<10} {c.regime:<18} {c.mode:<8} {c.batch_size:>3} {c.seq_len:>5} "
            f"{c.median_ms:>10.3f} {delta:>9} {ci} {util:>8}{flag}"
        )

    # The resolution verdict, computed by the extracted `resolution_verdict` so a CPU test can
    # feed it a synthetic biased-but-tight control cell. Inline, the only reachable test was a
    # source grep -- and the version that lived here checked interval WIDTH only, so a control
    # reading +3.0% with a CI of [+2.8, +3.3] was certified as able to resolve 1.8%.
    print("\nA/A control -- the rig's own resolution (true effect is exactly 0):")
    verdict_lines, unresolvable = resolution_verdict(cells)
    for line in verdict_lines:
        print(line)

    low_util = [c for c in cells if c.utilization_ok is False]
    print("\nnotes:")
    print("  - effect ceiling is 4.03% (31.5 MB of 780.3 MB); the target is ~1.8%.")
    print(
        f"  - conv path was {next(iter(paths.values()))} on every arm, so the contrast is gate "
        f"structure and not kernel choice."
    )
    print(
        "  - decode rows are seq_len=1 forwards, NOT autoregressive steps: no conv-state cache "
        "and no KV cache. Omitted cache traffic would dilute the ratio; added launch overhead "
        "inflates it. The two push opposite ways, so this is NEITHER an upper nor a lower "
        "bound on served decode."
    )
    print(
        "  - eager decode ranks arms partly by kernel launch count (the treatments add 40-50 "
        "launches). Read the graphed rows for the serving-relevant number, and the launch "
        "counts above before attributing any eager gap to gate structure."
    )
    print(
        "  - prefill vs prefill-lastlogit shows how much of the delta the arm-invariant LM "
        "head dilutes: the head is ~23% of prefill cost and identical across arms."
    )
    print(
        "  - the weights-only bandwidth ratio is reported for decode only. In prefill the "
        "logits tensor alone is larger than the whole weight footprint, so a weights-only "
        "ratio there would understate real traffic by ~20x and is printed as n/a."
    )
    if low_util:
        print(f"  - {len(low_util)} row(s) below the utilization floor; see LOW_UTIL.")

    clock_spread = clock_spread_pct([i.sm_clock_mhz_mean for i in info.values()])
    if clock_spread is not None:
        print(
            f"  - SM clock spread across arms: {clock_spread:.2f}% (limit {MAX_CLOCK_SPREAD_PCT}%)"
        )
    else:
        print(
            "  - SM clock was NOT readable, so this run carries NO evidence that interleaving "
            "made drift common-mode. Reached only via --allow-unmeasured-clocks."
        )

    payload = {
        "run_id": opts.run_id,
        "device": device_name,
        "hbm_peak_gbs": peak_gbs,
        "bf16_peak_tflops": peak_tflops,
        "l2_mib": L2_MIB.get(device_name),
        "dtype": opts.dtype,
        "vocab_size": opts.vocab_size,
        "torch_version": torch.__version__,
        "baseline": opts.baseline,
        "control_arm": CONTROL_ARM,
        "target_resolution_pct": TARGET_RESOLUTION_PCT,
        "effect_ceiling_pct": 4.03,
        "rounds": opts.rounds,
        "iters_per_round": opts.iters_per_round,
        "soak_seconds": opts.soak_seconds,
        "use_fla_pinned": use_fla,
        "fla_importable": fla_importable,
        "conv_paths": paths,
        "sm_clock_spread_pct": clock_spread,
        "unresolvable_shapes": unresolvable,
        "arms": {a: asdict(i) for a, i in info.items()},
        "cells": [asdict(c) for c in cells],
        "samples": [asdict(s) for s in samples],
    }
    out_path = opts.out or f"gate_latency_{opts.run_id}.json"
    for destination in filter(
        None,
        [
            out_path,
            (
                os.path.join(os.environ["EDULLM_CHECKPOINT_DIR"], os.path.basename(out_path))
                if os.environ.get("EDULLM_CHECKPOINT_DIR")
                else None
            ),
        ],
    ):
        try:
            os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
            with open(destination, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"\nwrote {destination}")
        except OSError as exc:
            log.warning("could not write %s (%s); the table above is the record", destination, exc)

    # ------------------------------------------------------------------------------------
    # Verdicts, in order of what invalidates what.
    # ------------------------------------------------------------------------------------
    rc = 0
    if clock_spread is not None and clock_spread > MAX_CLOCK_SPREAD_PCT:
        print(
            f"\nERROR: mean SM clock differs by {clock_spread:.2f}% across arms, above the "
            f"{MAX_CLOCK_SPREAD_PCT}% limit. Interleaving did not make drift common-mode, so a "
            f"{TARGET_RESOLUTION_PCT}% delta cannot be separated from the clock.",
            file=sys.stderr,
        )
        rc = 1
    if low_util:
        print(
            f"\nERROR: {len(low_util)} row(s) fell below the utilization floor. A region that "
            f"is neither compute-bound nor bandwidth-bound cannot show a FLOPs or bytes saving, "
            f"so those rows are not latency comparisons.",
            file=sys.stderr,
        )
        rc = 1
    if unresolvable and not opts.allow_unresolvable:
        print(
            f"\nERROR: the A/A control's interval is wider than the "
            f"{TARGET_RESOLUTION_PCT}% effect on {len(unresolvable)} shape(s): "
            f"{', '.join(unresolvable)}. Two identical models differ by more than the effect "
            f"being hunted, so any delta on those shapes is noise with a sign. Raise --rounds, "
            f"or accept that this shape cannot answer the question.",
            file=sys.stderr,
        )
        rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
