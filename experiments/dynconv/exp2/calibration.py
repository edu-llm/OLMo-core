"""MQAR difficulty recalibration for Exp-2's topologies. BASELINE ONLY.

WHY A RECALIBRATION AT ALL -- THE RECORDED ONE DOES NOT TRANSFER
---------------------------------------------------------------
The recorded calibration (``mqar_calibration.json``, FarmShare 1670987) is for a **4-layer d=128
model with attention at (1, 3)**. Exp-2 is **6 layers**, and critically the ``allliv`` topology has
**ZERO attention**. The probe README says so outright:

    "the cliff here is **not** a receptive-field limit -- the attention layers are global, so reach
    is not the binding constraint; what degrades is the difficulty of *finding* the recall circuit
    as distractor count grows. **What transfers is the method and the 1/D floor, not the operating
    point.**"  -- probes/mqar/README.md:97-104

R5 F5(i) makes the same point from the other side: with 2 GQA layers present, attention can solve
MQAR *by itself* and mask any conv-mechanism difference -- a null for the wrong reason. Remove
attention and the cliff moves a lot, in the direction of harder.

THE THREE RULES THIS SCRIPT ENFORCES
------------------------------------
1. **BASELINE ONLY.** There is no arm flag and there will not be one. Calibrating difficulty while
   looking at a treatment arm is choosing the test until it gives the answer you want. The only
   model this builds is S1 (``static``).
2. **POSITIVE CONTROL FIRST.** :func:`positive_control` runs before any sweep and :func:`main`
   refuses ``--grid`` without either a passing control in the same invocation or an explicit
   ``--skip-positive-control``. A sweep whose easiest rung scores zero cannot separate "hard task"
   from "broken setup" -- this already happened here (job 1670922 returned 0.000 everywhere because
   vocab was 8192).
3. **REFUSE UNDER-BUDGET.** ``CALIBRATED_STEPS=8000 x CALIBRATED_BATCH_SIZE=64 = 512,000``
   examples. Job 1670963 under-trained by 5.3x and produced a table that read as "too hard" rather
   than "under-trained". Enforced by ``mqar_harness.check_budget``.

CONFIGS DROPPED, WITH THE MEASURED REASON
-----------------------------------------
From ``mqar_calibration.json`` (4-layer, attention at (1,3), vocab 256, lr 3e-3, 8000 x 64):

    config      1/D floor   success   per-seed
    N64_D4        0.2500      80%     0.2695 0.9878 1.0000 1.0000 1.0000
    N128_D8       0.1250     100%     1.0000 x 10   <- CEILING, DROPPED
    N256_D16      0.0625     100%     1.0000 x 5    <- CEILING, DROPPED
    N64_D8        0.1250     100%     1.0000 x 5    <- CEILING, DROPPED
    N256_D8       0.1250     100%     1.0000 x 5    <- CEILING, DROPPED
    N512_D8       0.1250      40%     0.1443 0.1475 0.1555 1.0000 1.0000
    N512_D64      0.0156      20%     0.0515 0.0853 0.2043 0.5584 0.9825   <- PRIMARY
    N1024_D8      0.1250       0%     0.1387 0.1458 0.1467 0.1497 0.5710

A config at 1.000 with sd = 0.00 pp cannot discriminate arms **at any n**: the paired difference is
identically zero, so ``s_delta = 0`` and the required n is undefined rather than large. This is
exactly the mistake the cited paper makes -- In-Context Recall and Noisy Recall both saturate at
1.000 for a *static* baseline, so its dynamic-vs-static contrast on those tasks is uninformative by
construction.

WHY NOT THE SCRIPT'S OWN AUTO-PICK
----------------------------------
``mqar_calibrate.py`` picks ``N512_D8`` by proximity to a 50 % success rate. The README explicitly
recommends AGAINST that (README.md:78-83) and this module follows the README:

* ``N512_D64`` is **primary**: off-ceiling on *both* axes at once, its scores are GRADED rather than
  binary (measured spread 0.05/0.09/0.20/0.56/0.98 = 3.3x-62.9x floor), and its 0.016 floor leaves
  far more headroom. Note it is off-ceiling *because* of high load; its 20 % success rate looks
  worse than ``N512_D8``'s 40 % but carries strictly more information per seed.
* ``N512_D8`` is **secondary**: same length, 8x less capacity load, so the pair separates capacity
  from distance. Its four floor-parked seeds (0.144-0.156 against a 0.125 floor = 1.2x) carry almost
  no gradation, which is why it is not primary.

THE HONEST ANSWER ON R3 F8's TARGET BAND
----------------------------------------
R3 F8's fix asks for a config where "the baseline sits at 30-70 % with **sigma < 15 pp**".
:func:`assess_target_band` reports whether that is achievable and, on the recorded data, it is
**not**: see :data:`RECORDED_BAND_VERDICT`. Every off-ceiling config has sigma of 19-47 pp on
accuracy. The band is not reachable by moving the difficulty knob, because the variance is not a
function of difficulty -- the repo's own ANOVA puts **94.1 % of variance on SEED and 5.9 % on
memory load** (F(3,16)=0.337 vs F_crit 3.24, ``KDA/HANDOFF.md:441-449``). Difficulty calibration
cannot fix a seed-variance problem. That is a finding to report, not a knob to keep turning; the
recourse is the continuous NLL endpoint and an honest required-n, which is why Exp-2's primary
deliverable is sigma.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mqar_harness import (  # noqa: E402
    ALLLIV_ATTENTION_LAYERS,
    CALIBRATED_BATCH_SIZE,
    CALIBRATED_EXAMPLES,
    CALIBRATED_LR,
    CALIBRATED_STEPS,
    CALIBRATED_VOCAB,
    D_MODEL,
    HYBRID_ATTENTION_LAYERS,
    N_LAYERS,
    MQARConfig,
    ModelBuilder,
    append_record,
    check_budget,
    completed_keys,
    load_records,
    run_cell,
    stub_build_model,
)
from sigma import (  # noqa: E402
    MIN_EVAL_ITEMS,
    SOLVE_THRESHOLD,
    degenerate_floor,
    summarize_cell,
)

BASELINE_ARM = "static"  # S1. The ONLY arm this module will build.

# R3 F8's requested band. Reported against honestly, never silently widened.
TARGET_ACC_LO, TARGET_ACC_HI = 0.30, 0.70
TARGET_SIGMA_MAX_PP = 15.0

# Ceiling detection. A config is unusable if a majority of seeds are pinned this high: the paired
# difference is then identically zero for most pairs, so s_delta collapses and no n suffices.
CEILING_ACC = 0.99
CEILING_FRACTION = 0.6


# ======================================================================================
# The recorded evidence, extracted from the JSONs (not restated from the README)
# ======================================================================================

RECORDED_CALIBRATION_JSON = (
    "Brainlifts/liv_experiment_research/probes/mqar/mqar_calibration.json"
)
RECORDED_CONTROL_JSON = (
    "Brainlifts/liv_experiment_research/probes/mqar/mqar_positive_control.json"
)


@dataclass(frozen=True)
class RecordedConfig:
    """One config's MEASURED numbers, read out of ``mqar_calibration.json`` (FarmShare 1670987)."""

    config: str
    seq_len: int
    num_pairs: int
    floor: float
    n_seeds: int
    success_rate: float
    median: float
    mean: float
    sigma_pp: float
    per_seed: Tuple[float, ...]
    mean_seconds_l40s: float
    verdict: str
    reason: str


# Values below are computed from mqar_calibration.json's `runs` array; verify_recorded_numbers()
# re-derives them from the file and test_harness.py asserts the agreement, so these cannot drift.
RECORDED: Tuple[RecordedConfig, ...] = (
    RecordedConfig(
        "N64_D4", 64, 4, 0.2500, 5, 0.80, 1.0000, 0.8515, 32.54,
        (0.2695, 0.9878, 1.0000, 1.0000, 1.0000), 163.0,
        "DROP",
        "4 of 5 seeds pinned at ceiling; floor 0.250 is so high that a +8pp claim could sit "
        "entirely below the degenerate 'guess among the D present values' strategy.",
    ),
    RecordedConfig(
        "N128_D8", 128, 8, 0.1250, 10, 1.00, 1.0000, 1.0000, 0.00,
        (1.0,) * 10, 171.7,
        "DROP",
        "CEILING. 10 of 10 seeds at exactly 1.0000, sd = 0.00 pp. The paired difference is "
        "identically zero, so s_delta = 0 and required-n is undefined, not merely large. Cannot "
        "discriminate arms at ANY n.",
    ),
    RecordedConfig(
        "N256_D16", 256, 16, 0.0625, 5, 1.00, 1.0000, 1.0000, 0.00,
        (1.0,) * 5, 211.8,
        "DROP",
        "CEILING. 5 of 5 seeds at exactly 1.0000, sd = 0.00 pp. Same reason as N128_D8.",
    ),
    RecordedConfig(
        "N64_D8", 64, 8, 0.1250, 5, 1.00, 1.0000, 1.0000, 0.00,
        (1.0,) * 5, 170.1,
        "DROP",
        "CEILING. 5 of 5 at 1.0000 (distance-sweep rung).",
    ),
    RecordedConfig(
        "N256_D8", 256, 8, 0.1250, 5, 1.00, 1.0000, 1.0000, 0.00,
        (1.0,) * 5, 177.1,
        "DROP",
        "CEILING. 5 of 5 at 1.0000 (distance-sweep rung).",
    ),
    RecordedConfig(
        "N512_D8", 512, 8, 0.1250, 5, 0.40, 0.1555, 0.4895, 46.61,
        (0.1443, 0.1475, 0.1555, 1.0000, 1.0000), 174.7,
        "SECONDARY",
        "Off ceiling (40% success) but strictly BIMODAL: two seeds at 1.0000 and three parked at "
        "1.15-1.24x the 0.125 floor, nothing in between. Almost no gradation, and sigma = 46.6 pp. "
        "Useful only as the capacity-vs-distance partner of N512_D64 (same length, 8x less load).",
    ),
    RecordedConfig(
        "N512_D64", 512, 64, 0.015625, 5, 0.20, 0.2043, 0.3764, 39.39,
        (0.0515, 0.0853, 0.2043, 0.5584, 0.9825), 592.1,
        "PRIMARY",
        "Off ceiling AND off floor on both axes. The ONLY config where bimodality BREAKS -- seeds "
        "spread continuously at 3.3x / 5.5x / 13.1x / 35.7x / 62.9x the 0.0156 floor -- so scores "
        "are GRADED and carry more information per seed. Costs 3.4x the wall-clock of the D=8 "
        "rungs (592 s vs 175 s on the recording device).",
    ),
    RecordedConfig(
        "N1024_D8", 1024, 8, 0.1250, 5, 0.00, 0.1467, 0.2304, 19.05,
        (0.1387, 0.1458, 0.1467, 0.1497, 0.5710), 246.1,
        "DROP",
        "FLOOR. 0% success and 4 of 5 seeds within 1.11-1.20x the floor. Lowest sigma of any "
        "off-ceiling config (19.05 pp) precisely BECAUSE it is floor-pinned -- a floor is as "
        "uninformative as a ceiling, and its low sigma must not be mistaken for a usable band.",
    ),
)

# The positive control, from mqar_positive_control.json (FarmShare 1670928). N64_D4 only, 12 trials
# = {vocab 256, 8192} x {lr 3e-4, 1e-3, 3e-3} x {attn (2,), attn (1,3)}.
RECORDED_CONTROL = {
    "n_trials": 12,
    "best_accuracy": 1.0,
    "best": {"vocab": 256, "lr": 3e-3, "attention_layers": (1, 3), "steps": 8000},
    "second_best": {"vocab": 256, "lr": 1e-3, "attention_layers": (2,), "accuracy": 0.99462890625},
    # vocab 8192: best of 6 was 0.2138671875; four of the six sat at loss 8.25-8.34 against
    # ln(4096) = 8.3178 -- the "it's a value token" plateau. TWO scored exactly 0.000.
    "vocab_8192_best_accuracy": 0.2138671875,
    "vocab_8192_n_exact_zero": 2,
    "ln_4096": math.log(4096),
    # THE 0.80 THRESHOLD JUSTIFICATION: across all 12 trials no accuracy fell in [0.30, 0.80].
    "empty_gap_lo": 0.30,
    "empty_gap_hi": 0.80,
    "accuracies_sorted": (
        0.0, 0.0, 0.04345703125, 0.1337890625, 0.20849609375, 0.2138671875,
        0.24755859375, 0.25537109375, 0.26318359375, 0.27392578125,
        0.99462890625, 1.0,
    ),
    "mean_seconds": 169.5,
}

RECORDED_BAND_VERDICT = (
    "NOT ACHIEVABLE by moving the difficulty knob. Every off-ceiling recorded config has "
    "sigma 19.05-46.61 pp on accuracy, against R3 F8's requested <15 pp. And the lowest sigma "
    "(19.05 pp, N1024_D8) belongs to a FLOOR-pinned config with 0% success, so it is not a usable "
    "operating point either -- low sigma there is an artifact of being pinned, not of being "
    "well-conditioned. The reason is structural: the repo's own one-way ANOVA (n=20, "
    "KDA/HANDOFF.md:441-449) attributes 5.9% of variance to MEMORY LOAD and 94.1% to SEED, "
    "F(3,16)=0.337 against F_crit 3.24. Difficulty is the knob calibration turns; seed variance is "
    "where the variance lives. So calibration CANNOT deliver the band, and no amount of further "
    "sweeping will. The recourses that remain are (a) the continuous query-NLL endpoint, worth a "
    "2-18x SNR gain, and (b) reporting an honest required-n instead of a verdict. That is why "
    "Exp-2's primary deliverable is a measured sigma."
)


def verify_recorded_numbers(json_path: Optional[Path] = None) -> Dict[str, object]:
    """
    Re-derive :data:`RECORDED` from ``mqar_calibration.json`` so the pinned table cannot drift.

    :param json_path: Override the search path (for tests).
    :returns: ``{"ok", "checked", "mismatches", "source"}``. ``ok`` is False if the file is absent,
        so a caller can distinguish "verified" from "could not verify".
    """
    candidates = (
        [json_path]
        if json_path
        else [
            Path("/Users/ericwu/Developer/Capstone_LLM") / RECORDED_CALIBRATION_JSON,
            Path(
                "/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/"
                "claude-01--liv-short-conv-mixer/experiments/liv/mqar/mqar_calibration.json"
            ),
        ]
    )
    src = next((p for p in candidates if p and p.is_file()), None)
    if src is None:
        return {"ok": False, "checked": 0, "mismatches": ["source JSON not found"], "source": None}

    data = json.loads(src.read_text())
    by_cfg: Dict[str, List[dict]] = {}
    for r in data["runs"]:
        by_cfg.setdefault(r["config"], []).append(r)

    mismatches: List[str] = []
    checked = 0
    for rec in RECORDED:
        runs = by_cfg.get(rec.config)
        if not runs:
            mismatches.append(f"{rec.config}: absent from JSON")
            continue
        accs = sorted(r["accuracy"] for r in runs)
        floor = runs[0]["degenerate_floor"]
        succ = sum(1 for r in runs if r["accuracy"] >= SOLVE_THRESHOLD) / len(runs)
        sig = statistics.stdev(accs) * 100 if len(accs) > 1 else 0.0
        med = statistics.median(accs)
        for label, got, want, tol in (
            ("n_seeds", len(accs), rec.n_seeds, 0),
            ("floor", floor, rec.floor, 1e-6),
            ("success_rate", succ, rec.success_rate, 1e-9),
            ("median", med, rec.median, 5e-4),
            ("mean", statistics.mean(accs), rec.mean, 5e-4),
            ("sigma_pp", sig, rec.sigma_pp, 5e-2),
        ):
            if abs(got - want) > tol:
                mismatches.append(f"{rec.config}.{label}: JSON {got!r} != pinned {want!r}")
        if len(accs) == len(rec.per_seed):
            for i, (g, w) in enumerate(zip(accs, sorted(rec.per_seed))):
                if abs(g - w) > 5e-4:
                    mismatches.append(f"{rec.config}.per_seed[{i}]: {g:.6f} != {w:.6f}")
        checked += 1
    return {"ok": not mismatches, "checked": checked, "mismatches": mismatches, "source": str(src)}


def verify_recorded_control(json_path: Optional[Path] = None) -> Dict[str, object]:
    """
    Re-derive :data:`RECORDED_CONTROL` from ``mqar_positive_control.json``.

    In particular re-checks the **empty gap [0.30, 0.80]** that justifies
    :data:`sigma.SOLVE_THRESHOLD` -- the threshold's justification is a measurement, so it has to be
    re-measurable.
    """
    candidates = (
        [json_path]
        if json_path
        else [
            Path("/Users/ericwu/Developer/Capstone_LLM") / RECORDED_CONTROL_JSON,
            Path(
                "/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/"
                "claude-01--liv-short-conv-mixer/experiments/liv/mqar/"
                "mqar_positive_control.json"
            ),
        ]
    )
    src = next((p for p in candidates if p and p.is_file()), None)
    if src is None:
        return {"ok": False, "mismatches": ["source JSON not found"], "source": None}

    data = json.loads(src.read_text())
    trials = data["trials"]
    accs = sorted(t["accuracy"] for t in trials)
    in_gap = [a for a in accs if RECORDED_CONTROL["empty_gap_lo"] < a
              < RECORDED_CONTROL["empty_gap_hi"]]
    v8192 = [t for t in trials if t["vocab_size"] == 8192]

    mismatches: List[str] = []
    if len(trials) != RECORDED_CONTROL["n_trials"]:
        mismatches.append(f"n_trials {len(trials)} != {RECORDED_CONTROL['n_trials']}")
    if in_gap:
        mismatches.append(
            f"the [0.30, 0.80] gap that justifies SOLVE_THRESHOLD=0.80 is NOT empty: {in_gap}"
        )
    if abs(max(accs) - RECORDED_CONTROL["best_accuracy"]) > 1e-9:
        mismatches.append(f"best {max(accs)} != {RECORDED_CONTROL['best_accuracy']}")
    best8192 = max(t["accuracy"] for t in v8192)
    if abs(best8192 - RECORDED_CONTROL["vocab_8192_best_accuracy"]) > 1e-9:
        mismatches.append(
            f"vocab-8192 best {best8192} != {RECORDED_CONTROL['vocab_8192_best_accuracy']}"
        )
    n_zero = sum(1 for t in v8192 if t["accuracy"] == 0.0)
    if n_zero != RECORDED_CONTROL["vocab_8192_n_exact_zero"]:
        mismatches.append(
            f"vocab-8192 exact zeros {n_zero} != {RECORDED_CONTROL['vocab_8192_n_exact_zero']}"
        )
    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "source": str(src),
        "n_trials": len(trials),
        "accuracies_sorted": accs,
        "empty_gap_confirmed": not in_gap,
    }


# ======================================================================================
# The Exp-2 grid: recorded survivors + rungs that only make sense once attention is gone
# ======================================================================================


def exp2_grid(*, include_easier: bool = True) -> Tuple[MQARConfig, ...]:
    """
    The configs to sweep for Exp-2, with the ceiling-saturated ones dropped.

    Kept from the recording: ``N512_D64`` (primary) and ``N512_D8`` (secondary).

    Added, and ONLY because ``allliv`` has no attention: ``N128_D8`` and ``N256_D16`` are re-entered
    as *easier* rungs. That is not a contradiction of dropping them -- they were at ceiling **with
    2 of 4 layers global**. R5 F5(i) is explicit that the recorded cliff *"is not a
    receptive-field limit -- the attention layers are global"*, so removing attention should move the
    cliff toward easier configs and these rungs are the ones that can catch it. If they come back at
    1.000 in ``allliv`` too, they are dropped again by measurement rather than by assumption. Under
    ``include_easier=False`` only the recorded survivors are swept.
    """
    v = CALIBRATED_VOCAB
    primary = (
        MQARConfig(seq_len=512, num_pairs=64, vocab_size=v),
        MQARConfig(seq_len=512, num_pairs=8, vocab_size=v),
    )
    if not include_easier:
        return primary
    return (
        MQARConfig(seq_len=128, num_pairs=8, vocab_size=v),
        MQARConfig(seq_len=256, num_pairs=16, vocab_size=v),
    ) + primary


PRIMARY_CONFIG = MQARConfig(seq_len=512, num_pairs=64, vocab_size=CALIBRATED_VOCAB)
SECONDARY_CONFIG = MQARConfig(seq_len=512, num_pairs=8, vocab_size=CALIBRATED_VOCAB)

DROPPED_CONFIGS: Dict[str, str] = {
    r.config: r.reason for r in RECORDED if r.verdict == "DROP"
}


# ======================================================================================
# Assessment
# ======================================================================================


@dataclass(frozen=True)
class BandAssessment:
    """Whether a config can discriminate arms, and why not when it cannot."""

    config: str
    n_seeds: int
    floor: float
    success_rate: float
    median: float
    sigma_pp: float
    at_ceiling: bool
    at_floor: bool
    in_target_band: bool
    sigma_ok: bool
    usable: bool
    verdict: str


def assess_target_band(
    per_seed_accuracy: Sequence[float],
    *,
    config: str,
    num_pairs: int,
) -> BandAssessment:
    """
    Judge one config against R3 F8's band: baseline at 30-70 % with sigma < 15 pp.

    Reports the two conditions **separately**, because they fail for different reasons and
    conflating them hides the finding. Off-ceiling-and-off-floor is achievable (``N512_D64`` is);
    sigma < 15 pp is not, and that is a seed-variance fact rather than a difficulty fact.

    :param per_seed_accuracy: Measured accuracies, one per seed.
    :param config: Label, for the report.
    :param num_pairs: ``D``, for the ``1/D`` floor.
    """
    accs = list(per_seed_accuracy)
    if not accs:
        raise ValueError("no seeds")
    floor = degenerate_floor(num_pairs)
    med = statistics.median(accs)
    sig = statistics.stdev(accs) * 100 if len(accs) > 1 else float("nan")
    succ = sum(1 for a in accs if a >= SOLVE_THRESHOLD) / len(accs)
    at_ceiling = sum(1 for a in accs if a >= CEILING_ACC) / len(accs) >= CEILING_FRACTION
    at_floor = sum(1 for a in accs if a <= floor * 1.5) / len(accs) >= CEILING_FRACTION
    in_band = TARGET_ACC_LO <= med <= TARGET_ACC_HI
    sigma_ok = sig == sig and sig < TARGET_SIGMA_MAX_PP

    if at_ceiling:
        verdict = (
            f"UNUSABLE (CEILING): {sum(1 for a in accs if a >= CEILING_ACC)}/{len(accs)} seeds "
            f">= {CEILING_ACC}. sigma = {sig:.2f} pp. A pinned metric gives s_delta -> 0, so "
            f"required-n is undefined, not large. Cannot discriminate arms at any n."
        )
    elif at_floor:
        verdict = (
            f"UNUSABLE (FLOOR): {sum(1 for a in accs if a <= floor * 1.5)}/{len(accs)} seeds "
            f"within 1.5x the {floor:.4f} floor -- the degenerate 'one of the D present values' "
            f"algorithm. Its low sigma ({sig:.2f} pp) is an artifact of being pinned."
        )
    elif in_band and sigma_ok:
        verdict = f"MEETS R3 F8's BAND: median {med:.3f} in [0.30, 0.70], sigma {sig:.2f} < 15 pp."
    elif in_band:
        verdict = (
            f"OFF CEILING AND FLOOR (median {med:.3f}) but sigma {sig:.2f} pp >= "
            f"{TARGET_SIGMA_MAX_PP:g} pp. USABLE for a sigma measurement, NOT for an accuracy "
            f"verdict at any affordable n. This is the expected outcome -- 94.1% of variance is "
            f"seed, not load, so difficulty calibration cannot reduce sigma."
        )
    else:
        verdict = (
            f"OUTSIDE the 30-70% band (median {med:.3f}) with sigma {sig:.2f} pp. Graded scores "
            f"may still make it the best available operating point; judge on gradation, not on "
            f"proximity to 50%."
        )
    return BandAssessment(
        config=config,
        n_seeds=len(accs),
        floor=floor,
        success_rate=succ,
        median=med,
        sigma_pp=sig,
        at_ceiling=at_ceiling,
        at_floor=at_floor,
        in_target_band=in_band,
        sigma_ok=sigma_ok,
        usable=not (at_ceiling or at_floor),
        verdict=verdict,
    )


def assess_recorded() -> List[BandAssessment]:
    """Apply :func:`assess_target_band` to every recorded config. No new compute."""
    return [
        assess_target_band(r.per_seed, config=r.config, num_pairs=r.num_pairs) for r in RECORDED
    ]


# ======================================================================================
# Positive control -- runs FIRST
# ======================================================================================


def positive_control(
    *,
    build_model: ModelBuilder,
    topology: str,
    kernel_size: int = 3,
    steps: int = CALIBRATED_STEPS,
    batch_size: int = CALIBRATED_BATCH_SIZE,
    lrs: Sequence[float] = (1e-3, 3e-3),
    device: torch.device = torch.device("cpu"),
    out_path: Optional[Path] = None,
    eval_items: int = MIN_EVAL_ITEMS,
    smoke: bool = False,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    "Can this setup learn MQAR at all?" -- on the EASIEST rung only, before any difficulty sweep.

    Job 1670922's first sweep returned 0.000 everywhere because vocab was 8192, and a sweep whose
    easiest rung scores zero cannot separate "hard task" from "broken setup". So this runs
    ``N64_D4`` (4 pairs, 64 tokens) and nothing harder: if that cannot be solved, nothing harder can.

    Sweeps only LR, because the recorded control already settled vocab (256, not 8192) and the
    remaining free knob at fixed topology is the learning rate.

    :returns: ``{"passed", "best", "trials", "verdict"}``. ``passed`` is True iff some LR exceeds
        :data:`sigma.SOLVE_THRESHOLD`.
    """
    cfg = MQARConfig(seq_len=64, num_pairs=4, vocab_size=CALIBRATED_VOCAB)
    trials = []
    if verbose:
        print(
            f"POSITIVE CONTROL FIRST: {cfg.label} (floor {degenerate_floor(4):.3f}), "
            f"topology={topology}, W={kernel_size}, {len(lrs)} LRs x {steps} steps",
            flush=True,
        )
    for lr in lrs:
        rec = run_cell(
            arm=BASELINE_ARM,
            topology=topology,
            kernel_size=kernel_size,
            cfg=cfg,
            seed_pair=0,
            build_model=build_model,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            eval_items=eval_items,
            smoke=smoke,
            verbose=False,
        )
        rec.extra["control_lr"] = lr
        rec.extra["role"] = "positive_control"
        trials.append(rec)
        if out_path:
            append_record(out_path, rec)
        if verbose:
            print(
                f"  lr {lr:.0e}: acc {rec.accuracy:.4f} "
                f"({rec.accuracy / rec.floor:.1f}x floor)  nll {rec.nll_query:.4f}  "
                f"loss {rec.first_loss:.3f}->{rec.final_loss:.3f}  "
                f"{'LEARNS' if rec.accuracy >= SOLVE_THRESHOLD else '.'}  [{rec.seconds:.1f}s]",
                flush=True,
            )
    best = max(trials, key=lambda r: r.accuracy)
    passed = best.accuracy >= SOLVE_THRESHOLD
    verdict = (
        f"PASS: the setup CAN learn MQAR ({best.accuracy:.4f} at lr "
        f"{best.extra['control_lr']:.0e}). Sweep difficulty from here."
        if passed
        else (
            f"FAIL: best {best.accuracy:.4f} < {SOLVE_THRESHOLD} on the EASIEST rung. "
            f"DO NOT run a difficulty sweep -- its output could not distinguish 'hard task' from "
            f"'broken setup' (job 1670922). Final loss {best.final_loss:.3f}; compare "
            f"ln(D)={math.log(4):.3f} (degenerate) and ln(vocab/2)="
            f"{math.log(CALIBRATED_VOCAB / 2):.3f} ('it's a value token')."
        )
    )
    if verbose:
        print(f"  => {verdict}", flush=True)
    return {"passed": passed, "best": best, "trials": trials, "verdict": verdict}


# ======================================================================================
# The sweep -- baseline only
# ======================================================================================


def calibrate(
    *,
    build_model: ModelBuilder,
    topologies: Sequence[str] = ("allliv", "hybrid"),
    configs: Optional[Sequence[MQARConfig]] = None,
    kernel_size: int = 3,
    seeds: int = 5,
    steps: int = CALIBRATED_STEPS,
    batch_size: int = CALIBRATED_BATCH_SIZE,
    lr: float = CALIBRATED_LR,
    device: torch.device = torch.device("cpu"),
    out_path: Path = Path("exp2_calibration.jsonl"),
    eval_items: int = MIN_EVAL_ITEMS,
    smoke: bool = False,
    resume: bool = True,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Sweep MQAR difficulty on the BASELINE ONLY, for each Exp-2 topology.

    Writes incrementally to ``out_path`` (JSONL, fsynced) and resumes.

    :returns: ``{"records", "assessments", "recommendation"}``.
    """
    check_budget(steps, batch_size, smoke=smoke)
    configs = tuple(configs) if configs is not None else exp2_grid()

    done = completed_keys(out_path) if resume else set()
    for top in topologies:
        for cfg in configs:
            if verbose:
                print(
                    f"\n{top} / {cfg.label}  (seq_len={cfg.seq_len}, D={cfg.num_pairs}, "
                    f"floor {degenerate_floor(cfg.num_pairs):.4f})",
                    flush=True,
                )
            for s in range(seeds):
                from mqar_harness import cell_key  # noqa: PLC0415

                if cell_key(BASELINE_ARM, top, kernel_size, cfg.label, s) in done:
                    continue
                rec = run_cell(
                    arm=BASELINE_ARM,
                    topology=top,
                    kernel_size=kernel_size,
                    cfg=cfg,
                    seed_pair=s,
                    build_model=build_model,
                    steps=steps,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                    eval_items=eval_items,
                    smoke=smoke,
                    verbose=False,
                )
                rec.extra["role"] = "calibration"
                append_record(out_path, rec)
                if verbose:
                    print(
                        f"  seed {s}: acc {rec.accuracy:.4f} "
                        f"({rec.accuracy / rec.floor:5.1f}x floor)  nll {rec.nll_query:.4f}  "
                        f"loss {rec.first_loss:.3f}->{rec.final_loss:.3f}  [{rec.seconds:.1f}s]",
                        flush=True,
                    )

    records = [r for r in load_records(out_path) if r.extra.get("role") != "positive_control"]
    assessments: List[BandAssessment] = []
    from collections import defaultdict

    groups: Dict[tuple, List] = defaultdict(list)
    for r in records:
        groups[(r.topology, r.config)].append(r)
    for (top, label), recs in sorted(groups.items()):
        cell = summarize_cell(recs)
        a = assess_target_band(
            cell.per_seed_accuracy, config=f"{top}/{label}", num_pairs=cell.num_pairs
        )
        assessments.append(a)

    usable = [a for a in assessments if a.usable]
    if usable:
        # Prefer GRADED over proximity-to-50%: the README's explicit recommendation. Gradation is
        # measured as the fraction of seeds strictly inside (floor*1.5, CEILING_ACC).
        def gradation(a: BandAssessment) -> float:
            recs = groups[tuple(a.config.split("/"))]
            accs = [r.accuracy for r in recs]
            return sum(1 for x in accs if a.floor * 1.5 < x < CEILING_ACC) / len(accs)

        pick = max(usable, key=lambda a: (gradation(a), -abs(a.median - 0.5)))
        recommendation = (
            f"{pick.config}: {gradation(pick):.0%} of seeds graded, median {pick.median:.3f} "
            f"({pick.median / pick.floor:.1f}x floor), sigma {pick.sigma_pp:.2f} pp. "
            f"Selected on GRADATION, not on proximity to a 50% success rate -- see the module "
            f"docstring on why mqar_calibrate.py's auto-pick is the wrong criterion."
        )
    else:
        recommendation = (
            "NO usable config: every one is at ceiling or floor. Report this rather than picking "
            "the least-bad; a pinned endpoint has s_delta -> 0 and cannot rank arms at any n."
        )
    return {"records": records, "assessments": assessments, "recommendation": recommendation}


def report(assessments: Sequence[BandAssessment], *, recommendation: str = "") -> str:
    """Format the calibration verdict."""
    lines = ["=" * 100, "EXP-2 MQAR CALIBRATION -- BASELINE (S1) ONLY", "=" * 100, ""]
    lines.append(f"{'config':<22}{'n':>3}{'floor':>8}{'succ':>7}{'median':>8}{'xfloor':>8}"
                 f"{'sigma_pp':>10}  verdict")
    lines.append("-" * 100)
    for a in assessments:
        lines.append(
            f"{a.config:<22}{a.n_seeds:>3}{a.floor:>8.4f}{a.success_rate:>7.2f}"
            f"{a.median:>8.3f}{a.median / a.floor:>8.1f}{a.sigma_pp:>10.2f}  "
            f"{'USABLE' if a.usable else 'UNUSABLE'}"
        )
    lines.append("")
    for a in assessments:
        lines.append(f"  {a.config}: {a.verdict}")
    lines.append("")
    lines.append(f"R3 F8 BAND (baseline 30-70%, sigma < {TARGET_SIGMA_MAX_PP:g} pp):")
    got = [a.config for a in assessments if a.in_target_band and a.sigma_ok]
    lines.append(f"  configs meeting BOTH conditions: {got if got else 'NONE'}")
    if not got:
        lines.append(f"  {RECORDED_BAND_VERDICT}")
    if recommendation:
        lines.append("")
        lines.append(f"RECOMMENDED OPERATING POINT: {recommendation}")
    return "\n".join(lines)


def recorded_evidence_report() -> str:
    """The measured evidence from the recorded JSONs. No compute; safe to call anywhere."""
    ctrl = verify_recorded_control()
    cal = verify_recorded_numbers()
    lines = [
        "=" * 100,
        "RECORDED EVIDENCE (extracted from the JSONs, re-verified against them)",
        "=" * 100,
        "",
        f"positive control  {RECORDED_CONTROL_JSON}  (FarmShare 1670928)",
        f"  verified against source: {ctrl['ok']}   {ctrl.get('mismatches') or ''}",
        f"  {RECORDED_CONTROL['n_trials']} trials on N64_D4 = "
        f"{{vocab 256, 8192}} x {{lr 3e-4, 1e-3, 3e-3}} x {{attn (2,), attn (1,3)}}",
        f"  best 1.0000 at vocab 256 / lr 3e-3 / attn (1,3); runner-up 0.99463 at "
        f"vocab 256 / lr 1e-3 / attn (2,)",
        f"  vocab 8192: best of 6 was {RECORDED_CONTROL['vocab_8192_best_accuracy']:.4f}, "
        f"{RECORDED_CONTROL['vocab_8192_n_exact_zero']} scored EXACTLY 0.0000, and four sat at "
        f"loss 8.25-8.34 vs ln(4096) = {RECORDED_CONTROL['ln_4096']:.4f} "
        f"('it's a value token', zero binding)",
        f"  SOLVE_THRESHOLD=0.80 justification: sorted accuracies are "
        f"{[round(a, 4) for a in RECORDED_CONTROL['accuracies_sorted']]}",
        f"    -> NO trial in [0.30, 0.80]. Empty gap re-confirmed from source: "
        f"{ctrl.get('empty_gap_confirmed')}. The threshold is insensitive over that whole range.",
        "",
        f"difficulty sweep  {RECORDED_CALIBRATION_JSON}  (FarmShare 1670987)",
        f"  verified against source: {cal['ok']}   {cal.get('mismatches') or ''}",
        f"  {cal['checked']} configs re-derived from the `runs` array",
        "",
        f"{'config':<10}{'N':>6}{'D':>5}{'floor':>9}{'n':>3}{'succ':>7}{'median':>8}{'mean':>8}"
        f"{'sigma_pp':>10}{'sec':>7}  {'verdict':<10} per-seed",
        "-" * 100,
    ]
    for r in RECORDED:
        lines.append(
            f"{r.config:<10}{r.seq_len:>6}{r.num_pairs:>5}{r.floor:>9.4f}{r.n_seeds:>3}"
            f"{r.success_rate:>7.2f}{r.median:>8.4f}{r.mean:>8.4f}{r.sigma_pp:>10.2f}"
            f"{r.mean_seconds_l40s:>7.0f}  {r.verdict:<10} "
            + " ".join(f"{a:.4f}" for a in sorted(r.per_seed)[: min(5, len(r.per_seed))])
            + (" ..." if len(r.per_seed) > 5 else "")
        )
    lines.append("")
    lines.append("DROPPED, with the measured reason:")
    for cfgname, reason in DROPPED_CONFIGS.items():
        lines.append(f"  {cfgname}: {reason}")
    lines.append("")
    lines.append(f"R3 F8 BAND ON THE RECORDED DATA: {RECORDED_BAND_VERDICT}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--evidence-only",
        action="store_true",
        help="print the recorded evidence and exit. No compute.",
    )
    ap.add_argument("--topologies", nargs="+", default=["allliv"], choices=["allliv", "hybrid"])
    ap.add_argument("--kernel-size", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=CALIBRATED_STEPS)
    ap.add_argument("--batch-size", type=int, default=CALIBRATED_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=CALIBRATED_LR)
    ap.add_argument("--eval-items", type=int, default=MIN_EVAL_ITEMS)
    ap.add_argument("--only-recorded-survivors", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="permit an under-calibration budget. Numbers from a smoke run are NOT results.",
    )
    ap.add_argument(
        "--skip-positive-control",
        action="store_true",
        help="skip the control. Only legitimate when a control for THIS topology already passed.",
    )
    ap.add_argument("--stub", action="store_true", help="use the built-in stub model builder")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="'auto' picks cuda when available. Set explicitly on FarmShare sbatch.",
    )
    ap.add_argument("--out", default="exp2_calibration.jsonl")
    args = ap.parse_args(argv)

    print(recorded_evidence_report(), flush=True)
    if args.evidence_only:
        return 0

    if args.stub:
        builder: ModelBuilder = stub_build_model
        print("\nMODEL: built-in STUB (no dynamic mechanism). Harness check, not science.",
              flush=True)
    else:
        try:
            from mqar_harness import arms_build_model  # noqa: PLC0415

            builder: ModelBuilder = arms_build_model
            import arms  # noqa: F401  # fail fast if it is not importable
        except ImportError as exc:
            print(f"\ncannot import arms.py ({exc}); pass --stub.", file=sys.stderr)
            return 2

    from mqar_harness import resolve_device  # noqa: PLC0415

    device = resolve_device(args.device)
    out = Path(args.out)
    print(f"\ndevice: {device}  dtype: torch.float32  torch {torch.__version__}   "
          f"budget: {args.steps * args.batch_size:,} examples"
          f"{'  [SMOKE -- not a result]' if args.smoke else ''}", flush=True)

    if not args.skip_positive_control:
        for top in args.topologies:
            print(flush=True)
            ctl = positive_control(
                build_model=builder,
                topology=top,
                kernel_size=args.kernel_size,
                steps=args.steps,
                batch_size=args.batch_size,
                device=device,
                out_path=out,
                eval_items=args.eval_items,
                smoke=args.smoke,
            )
            if not ctl["passed"]:
                print(
                    f"\nABORTING the difficulty sweep for topology {top!r}: the positive control "
                    f"did not pass. {ctl['verdict']}",
                    file=sys.stderr,
                )
                return 3
    else:
        print(
            "\nWARNING: --skip-positive-control. A sweep whose easiest rung scores zero cannot "
            "separate 'hard task' from 'broken setup' (job 1670922).",
            flush=True,
        )

    t0 = time.time()
    res = calibrate(
        build_model=builder,
        topologies=args.topologies,
        configs=exp2_grid(include_easier=not args.only_recorded_survivors),
        kernel_size=args.kernel_size,
        seeds=args.seeds,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        out_path=out,
        eval_items=args.eval_items,
        smoke=args.smoke,
    )
    print()
    print(report(res["assessments"], recommendation=res["recommendation"]), flush=True)
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s of sweep)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
