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
    ATTN1_ATTENTION_LAYERS,
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
    local_mirror_for,
    run_cell,
    stub_build_model,
    upload_and_verify,
)
from sigma import (  # noqa: E402
    MIN_EVAL_ITEMS,
    SOLVE_THRESHOLD,
    degenerate_floor,
    summarize_cell,
)

BASELINE_ARM = "static"  # S1. The ONLY arm this module will build.

# ======================================================================================
# The loss plateau ladder -- the reading that accuracy alone cannot give
# ======================================================================================
#
# Accuracy says "it did not solve it". The plateau says WHICH degenerate algorithm it learned, and
# those are different findings with different fixes. `allliv` measured acc 0.0092 against a 0.25
# floor -- BELOW chance -- which is only interpretable once you see the loss sat at ln(128): the
# model learned "the answer is a value token" and never reached "guess among the D present values".
# A run parked one rung BELOW the degenerate floor is not under-trained, it is unreachable.
def plateau_ladder(vocab_size: int, num_pairs: int) -> Tuple[Tuple[str, float, str], ...]:
    """The legible loss plateaus, ascending in competence. ``(name, loss, meaning)``."""
    return (
        ("bound", 0.0, "actually bound the pair -- the task is solved"),
        (f"ln(D)=ln({num_pairs})", math.log(num_pairs),
         "'one of the D values present' -- the DEGENERATE strategy, i.e. the 1/D accuracy floor"),
        (f"ln(vocab/2)=ln({vocab_size // 2})", math.log(vocab_size / 2),
         "'it is a value token' -- the WRONG-HALF plateau, one rung BELOW the 1/D floor"),
        (f"ln(vocab)=ln({vocab_size})", math.log(vocab_size), "init -- nothing learned"),
    )


def classify_plateau(
    final_loss: float, *, vocab_size: int, num_pairs: int, tol: float = 0.15
) -> Dict[str, object]:
    """Name where a final loss sits on the ladder.

    :param tol: nats within which a loss counts as parked AT a plateau rather than between two.
    :returns: ``{"nearest", "meaning", "distance", "parked", "position"}``.
    """
    ladder = plateau_ladder(vocab_size, num_pairs)
    name, value, meaning = min(ladder, key=lambda r: abs(final_loss - r[1]))
    dist = abs(final_loss - value)
    ln_d, ln_half = math.log(num_pairs), math.log(vocab_size / 2)
    if final_loss >= ln_half - tol:
        position = "AT-OR-ABOVE the wrong-half plateau: below the degenerate 1/D strategy"
    elif final_loss <= ln_d + tol:
        position = "AT-OR-BELOW ln(D): has reached the degenerate floor or better"
    else:
        position = "DESCENDED past the wrong-half plateau, above ln(D)"
    return {
        "nearest": name,
        "meaning": meaning,
        "distance": dist,
        "parked": dist <= tol,
        "position": position,
        "ln_D": ln_d,
        "ln_vocab_half": ln_half,
    }


# ======================================================================================
# The topology scan -- added 2026-08-05 because BOTH ends of the axis measured saturated
# ======================================================================================
#
# hybrid (2 attn of 6) = CEILING, measured 1.000 on every seed: attention solves MQAR alone and
#     masks the conv mechanism entirely.
# allliv (0 attn of 6) = FLOOR, measured acc 0.0092 against a 0.25 floor at the FULL 512,000-example
#     budget on the EASIEST rung, parked at ln(128) = 4.852.
#
# Neither can measure a sigma, and the S4-vs-S2 contrast is unreadable on both. The receptive-field
# arithmetic makes the floor STRUCTURAL rather than a training failure: 1 + L(W-1) = 13 tokens at
# L=6, W=3, against ~60 needed for N64_D4. No W in the swept grid reaches even the easiest rung.
#
# So the question is whether a topology BETWEEN the two ends exists. This scans attention count
# {0, 1, 2} on the BASELINE ONLY. There is one weak prior that it might: the recorded control's
# SECOND-BEST trial (RECORDED_CONTROL["second_best"]) was `attention_layers=(2,)` -- a single
# attention layer -- at accuracy 0.9946 on a 4-layer model. That is near ceiling, not in band, so it
# is a reason to look rather than a prediction of success, and it was at 4 layers not 6.
TOPOLOGY_SCAN_ORDER: Tuple[str, ...] = ("allliv", "attn1", "hybrid")
TOPOLOGY_ATTENTION_COUNT: Dict[str, int] = {"allliv": 0, "attn1": 1, "hybrid": 2}


def conv_receptive_field(*, n_layers: int, width: int, n_attention: int) -> Dict[str, object]:
    """Reach of the stack. Infinite once ANY attention layer is present.

    The point of reporting this next to the accuracy is that it distinguishes "the task is hard"
    from "the task is unreachable". A stack with no attention has a HARD bound of ``1 + L(W-1)``
    tokens; no budget, seed count or width crosses it.
    """
    if n_attention > 0:
        return {"reach_tokens": math.inf, "bounded": False,
                "note": f"{n_attention} attention layer(s): global reach, not receptive-field bound"}
    reach = 1 + n_layers * (width - 1)
    return {"reach_tokens": float(reach), "bounded": True,
            "note": f"1 + {n_layers}({width}-1) = {reach} tokens -- a HARD bound with 0 attention"}


@dataclass(frozen=True)
class TopologyVerdict:
    """One attention count, judged off-ceiling AND off-floor on the baseline."""

    topology: str
    n_attention: int
    n_seeds: int
    floor: float
    median_accuracy: float
    per_seed_accuracy: Tuple[float, ...]
    median_final_loss: float
    plateau: str
    plateau_position: str
    off_ceiling: bool
    off_floor: bool
    discriminating: bool
    verdict: str


def assess_topology(
    records: Sequence,
    *,
    topology: str,
    num_pairs: int,
    vocab_size: int,
    width: int,
    n_layers: int = N_LAYERS,
) -> TopologyVerdict:
    """Judge ONE topology on the baseline: is it off ceiling AND off floor?

    Success is a conjunction and is reported as one. "Off ceiling" alone is what ``allliv``
    achieved, and it was worthless.
    """
    if not records:
        raise ValueError(f"no records for topology {topology!r}")
    accs = tuple(r.accuracy for r in records)
    losses = [r.final_loss for r in records]
    floor = degenerate_floor(num_pairs)
    med = statistics.median(accs)
    med_loss = statistics.median(losses)
    pl = classify_plateau(med_loss, vocab_size=vocab_size, num_pairs=num_pairs)

    n_attn = TOPOLOGY_ATTENTION_COUNT.get(topology, -1)
    at_ceiling = sum(1 for a in accs if a >= CEILING_ACC) / len(accs) >= CEILING_FRACTION
    at_floor = sum(1 for a in accs if a <= floor * 1.5) / len(accs) >= CEILING_FRACTION
    off_ceiling, off_floor = not at_ceiling, not at_floor
    discriminating = off_ceiling and off_floor

    if at_ceiling:
        verdict = (
            f"CEILING: {sum(1 for a in accs if a >= CEILING_ACC)}/{len(accs)} seeds >= "
            f"{CEILING_ACC}. Attention solves the task alone, so the conv mechanism is masked. "
            f"s_delta -> 0 and no n suffices."
        )
    elif at_floor:
        rf = conv_receptive_field(n_layers=n_layers, width=width, n_attention=n_attn)
        verdict = (
            f"FLOOR: {sum(1 for a in accs if a <= floor * 1.5)}/{len(accs)} seeds within 1.5x the "
            f"{floor:.4f} floor. Median loss {med_loss:.3f} is {pl['position']}. Reach: {rf['note']}."
        )
    else:
        verdict = (
            f"OFF CEILING AND OFF FLOOR: median {med:.4f} ({med:.4f}/{floor:.4f} = "
            f"{med / floor:.1f}x floor), median final loss {med_loss:.3f} ({pl['position']}). "
            f"USABLE for a sigma measurement."
        )
    return TopologyVerdict(
        topology=topology,
        n_attention=n_attn,
        n_seeds=len(accs),
        floor=floor,
        median_accuracy=med,
        per_seed_accuracy=accs,
        median_final_loss=med_loss,
        plateau=str(pl["nearest"]),
        plateau_position=str(pl["position"]),
        off_ceiling=off_ceiling,
        off_floor=off_floor,
        discriminating=discriminating,
        verdict=verdict,
    )


def scan_topologies(
    *,
    build_model: ModelBuilder,
    config: MQARConfig,
    topologies: Sequence[str] = TOPOLOGY_SCAN_ORDER,
    kernel_size: int = 3,
    seeds: int = 1,
    steps: int = CALIBRATED_STEPS,
    batch_size: int = CALIBRATED_BATCH_SIZE,
    lr: float = CALIBRATED_LR,
    device: torch.device = torch.device("cpu"),
    out_path: Path = Path("exp2_topology_scan.jsonl"),
    eval_items: int = MIN_EVAL_ITEMS,
    smoke: bool = False,
    resume: bool = True,
    verbose: bool = True,
    upload_dest: Optional[str] = None,
) -> Dict[str, object]:
    """Scan attention count on the **BASELINE ARM ONLY**, at ONE config, at the FULL budget.

    Answers exactly one question: *does a topology exist that is off ceiling AND off floor for the
    baseline?* If the answer is no at every attention count, that is a real finding -- the d=128
    synthetic task cannot discriminate at any topology -- and this function says so rather than
    nominating the least-bad cell.

    :param upload_dest: If set, upload the results file after EVERY cell rather than only at the
        end. The harness uploads once at exit, which makes a wall-clock kill or a spot reclaim the
        one failure mode where the partial work is also lost (Sec 13.0m). A cell is minutes of GPU
        time and the upload is a few KB, so uploading per cell is nearly free insurance.
    :returns: ``{"records", "verdicts", "recommendation", "discriminating"}``.
    """
    check_budget(steps, batch_size, smoke=smoke)
    from mqar_harness import cell_key  # noqa: PLC0415

    done = completed_keys(out_path) if resume else set()
    for top in topologies:
        n_attn = TOPOLOGY_ATTENTION_COUNT.get(top, -1)
        rf = conv_receptive_field(n_layers=N_LAYERS, width=kernel_size, n_attention=n_attn)
        if verbose:
            print(
                f"\n{top}  ({n_attn} of {N_LAYERS} layers attention)  {config.label}  "
                f"floor {degenerate_floor(config.num_pairs):.4f}\n  reach: {rf['note']}",
                flush=True,
            )
        for s in range(seeds):
            if cell_key(BASELINE_ARM, top, kernel_size, config.label, s) in done:
                if verbose:
                    print(f"  seed {s}: already done, skipping", flush=True)
                continue
            rec = run_cell(
                arm=BASELINE_ARM,
                topology=top,
                kernel_size=kernel_size,
                cfg=config,
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
            rec.extra["role"] = "topology_scan"
            rec.extra["n_attention"] = n_attn
            rec.extra["receptive_field"] = rf
            rec.extra["plateau"] = classify_plateau(
                rec.final_loss, vocab_size=config.vocab_size, num_pairs=config.num_pairs
            )
            append_record(out_path, rec)
            if verbose:
                pl = rec.extra["plateau"]
                print(
                    f"  seed {s}: acc {rec.accuracy:.4f} ({rec.accuracy / rec.floor:5.1f}x floor)"
                    f"  nll {rec.nll_query:.4f}  loss {rec.first_loss:.3f}->{rec.final_loss:.3f}"
                    f"  [nearest plateau {pl['nearest']}, {pl['distance']:.3f} nats]"
                    f"  [{rec.seconds:.1f}s]",
                    flush=True,
                )
            # Upload after EVERY cell, not only at exit. A wall-clock kill is the one failure where
            # the partial work is otherwise lost too, and a few KB per cell is nearly free.
            if upload_dest:
                r = upload_and_verify(out_path, upload_dest)
                if verbose:
                    print(
                        f"    incremental upload: "
                        f"{'VERIFIED ' + str(r.get('bytes')) + 'B' if r.get('verified') else 'UNVERIFIED ' + str(r)[:120]}",
                        flush=True,
                    )

    records = [r for r in load_records(out_path) if r.extra.get("role") == "topology_scan"]
    from collections import defaultdict

    groups: Dict[str, List] = defaultdict(list)
    for r in records:
        if r.config == config.label and r.kernel_size == kernel_size:
            groups[r.topology].append(r)

    verdicts: List[TopologyVerdict] = []
    for top in topologies:
        if groups.get(top):
            verdicts.append(
                assess_topology(
                    groups[top],
                    topology=top,
                    num_pairs=config.num_pairs,
                    vocab_size=config.vocab_size,
                    width=kernel_size,
                )
            )

    usable = [v for v in verdicts if v.discriminating]
    if usable:
        pick = max(usable, key=lambda v: -abs(v.median_accuracy - 0.5))
        recommendation = (
            f"DISCRIMINATING TOPOLOGY FOUND: {pick.topology} ({pick.n_attention} of {N_LAYERS} "
            f"attention), median accuracy {pick.median_accuracy:.4f} against a "
            f"{pick.floor:.4f} floor, median final loss {pick.median_final_loss:.3f}. "
            f"{pick.verdict}"
        )
    else:
        recommendation = (
            "NO DISCRIMINATING TOPOLOGY at any attention count in {0, 1, 2}. Every one is at "
            "ceiling or at floor on the BASELINE. This is a real answer, not a tuning failure: it "
            "means the d=128 MQAR task cannot discriminate arms at any topology, and Exp-2's "
            "approach needs rethinking rather than retuning. Do NOT nominate the least-bad cell -- "
            "a pinned endpoint has s_delta -> 0 and cannot rank arms at any n."
        )
    return {
        "records": records,
        "verdicts": verdicts,
        "recommendation": recommendation,
        "discriminating": [v.topology for v in usable],
    }


def topology_report(
    verdicts: Sequence[TopologyVerdict],
    *,
    recommendation: str = "",
    vocab_size: int = CALIBRATED_VOCAB,
    num_pairs: int = 4,
) -> str:
    """The calibration table: attention count x accuracy x final loss x plateau position."""
    lines = [
        "=" * 104,
        "EXP-2 TOPOLOGY CALIBRATION -- BASELINE (S1) ONLY, attention count in {0, 1, 2}",
        "=" * 104,
        "",
        "The plateau ladder for this config:",
    ]
    for name, value, meaning in plateau_ladder(vocab_size, num_pairs):
        lines.append(f"  {value:7.3f}  {name:<22} {meaning}")
    lines += [
        "",
        f"{'topology':<10}{'attn':>5}{'n':>3}{'median acc':>12}{'xfloor':>8}"
        f"{'final loss':>12}  {'nearest plateau':<24} off-ceil  off-floor",
        "-" * 104,
    ]
    for v in verdicts:
        lines.append(
            f"{v.topology:<10}{v.n_attention:>5}{v.n_seeds:>3}{v.median_accuracy:>12.4f}"
            f"{v.median_accuracy / v.floor:>8.1f}{v.median_final_loss:>12.3f}  {v.plateau:<24}"
            f"{'yes' if v.off_ceiling else 'NO':>8}  {'yes' if v.off_floor else 'NO':>9}"
        )
    lines.append("")
    for v in verdicts:
        lines.append(f"{v.topology}: {v.verdict}")
        lines.append(f"  per-seed accuracy: {', '.join(f'{a:.4f}' for a in v.per_seed_accuracy)}")
    if recommendation:
        lines += ["", "=" * 104, recommendation, "=" * 104]
    return "\n".join(lines)

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


def _recorded_json_candidates(name: str) -> List[Path]:
    """Where to look for a recorded-evidence JSON, **next to this file first**.

    THE BUG THIS KILLS, and it is the third instance of the same shape in this experiment. Both
    verifiers previously held a tuple of two ABSOLUTE LAPTOP PATHS
    (``/Users/ericwu/...``), so on any other host both missed and the verifiers returned
    ``["source JSON not found"]`` -- which made ``test_harness.py`` and
    ``recorded_evidence_report()`` report "could not verify" on every machine except the one where
    the code was written. It is exactly the failure that killed FarmShare job 1676377
    (``_MQAR_SOURCES``), fixed there and left unfixed here.

    The files ARE staged beside this module (``check_submission.sh`` lists both as required runtime
    inputs and confirms them in the image), so resolving relative to ``__file__`` finds them on
    FarmShare, in the container, and on the laptop alike. The laptop paths are kept as trailing
    fallbacks so local use is unaffected.

    **Generalisation worth carrying:** a hardcoded absolute path is a portability bug that only
    appears on the SECOND host, and the first host is always the one where the code was written.
    Every cross-host artifact in this program must resolve inputs relative to itself first.
    """
    here = Path(__file__).resolve().parent
    return [
        here / name,
        here.parent / "mqar" / name,
        Path("/Users/ericwu/Developer/Capstone_LLM")
        / "Brainlifts/liv_experiment_research/probes/mqar"
        / name,
        Path(
            "/Users/ericwu/Developer/Capstone_LLM-worktrees/olmo-core/"
            "claude-01--liv-short-conv-mixer/experiments/liv/mqar"
        )
        / name,
    ]


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
    candidates = [json_path] if json_path else _recorded_json_candidates("mqar_calibration.json")
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
        [json_path] if json_path else _recorded_json_candidates("mqar_positive_control.json")
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
    # Default is None, not a list, so the two modes can have DIFFERENT defaults and an explicit
    # choice is distinguishable from an unset one. With a `["allliv"]` default, `--scan-topologies`
    # silently intersected to allliv alone and the scan reported "NO DISCRIMINATING TOPOLOGY" having
    # never built attn1 -- a submitted job would have run 1 cell of 3 and printed a confident
    # negative. Caught in a smoke run; see the refusal in the scan branch below.
    ap.add_argument(
        "--topologies", nargs="+", default=None, choices=["allliv", "attn1", "hybrid"]
    )
    ap.add_argument(
        "--scan-topologies",
        action="store_true",
        help="scan attention count {0,1,2} at ONE config on the BASELINE ONLY, and report whether "
             "any topology is off-ceiling AND off-floor. Skips the difficulty grid.",
    )
    ap.add_argument(
        "--scan-config",
        default="N64_D4",
        help="the config for --scan-topologies, as N<seq_len>_D<num_pairs>. Default N64_D4, the "
             "easiest rung: if the easiest rung cannot discriminate, nothing harder can.",
    )
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

    # Per-mode defaults, resolved AFTER parsing so that "unset" is distinguishable from "chosen".
    # The difficulty grid defaults to allliv (its historical default); the topology scan defaults to
    # ALL THREE attention counts, because a scan of one end cannot answer the question it exists for.
    grid_topologies = list(args.topologies) if args.topologies else ["allliv"]

    device = resolve_device(args.device)
    # Write to real LOCAL disk first, then upload and VERIFY. `Path(args.out)` alone is the bug that
    # cost a $0.76 pilot its entire output: handed `s3://...` it creates a container-local directory
    # literally named `s3:` (`Path("s3://b/x").is_absolute()` is False), fsyncs successfully onto the
    # wrong filesystem, prints "wrote N records to s3://...", and exits 0 with the bucket empty.
    # `append_record` now REFUSES a URI, so this is belt and braces rather than the only guard.
    out = local_mirror_for(args.out)
    print(f"\ndevice: {device}  dtype: torch.float32  torch {torch.__version__}   "
          f"budget: {args.steps * args.batch_size:,} examples"
          f"{'  [SMOKE -- not a result]' if args.smoke else ''}", flush=True)
    print(f"local mirror: {out}   final destination: {args.out}", flush=True)

    def _persist() -> int:
        """Upload and prove it by consulting the registry. Non-zero exit if unverified."""
        receipt = upload_and_verify(out, args.out)
        print(f"\nRECEIPT persistence: {json.dumps(receipt, default=str)}", flush=True)
        if not receipt.get("verified"):
            print(
                f"\nFAILED TO PERSIST to {args.out}. The results are NOT retrievable. Exiting "
                f"non-zero so this cannot be read as success -- an fsync return value is not a "
                f"receipt; the object listing is.",
                file=sys.stderr, flush=True,
            )
            return 3
        print(f"wrote and VERIFIED -> {receipt.get('uri', out)}", flush=True)
        return 0

    # ---- the topology scan: a different question from the difficulty grid ----------------------
    # This asks "does ANY attention count give a baseline that is off ceiling AND off floor?", which
    # must be answered before a difficulty sweep means anything. A difficulty grid run on a topology
    # that is pinned at a plateau measures the variance of the plateau.
    if args.scan_topologies:
        try:
            _n, _d = args.scan_config.lstrip("N").split("_D")
            scan_cfg = MQARConfig(
                seq_len=int(_n), num_pairs=int(_d), vocab_size=CALIBRATED_VOCAB
            )
        except Exception as exc:  # noqa: BLE001
            print(f"cannot parse --scan-config {args.scan_config!r}: {exc}", file=sys.stderr)
            return 2
        print(
            f"\nTOPOLOGY SCAN on the BASELINE ARM ONLY ({BASELINE_ARM} = S1) at "
            f"{scan_cfg.label}, W={args.kernel_size}, {args.seeds} seed(s) per topology.\n"
            f"  Calibrating on a treatment arm would tune the experiment toward the hypothesis, so "
            f"this module has no arm flag.",
            flush=True,
        )
        scan_tops = (
            [t for t in TOPOLOGY_SCAN_ORDER if t in set(args.topologies)]
            if args.topologies
            else list(TOPOLOGY_SCAN_ORDER)
        )
        # A "no discriminating topology" verdict is only meaningful if every attention count was
        # actually built. Refuse to emit one from a partial scan: the whole hypothesis under test is
        # that attention=1 is the topology in between, so a scan without `attn1` cannot answer it and
        # must not be allowed to print a confident negative. Same shape as the persistence bug --
        # a claim about an artifact must be checked against the artifact.
        if "attn1" not in scan_tops:
            print(
                f"\nREFUSING: --scan-topologies without 'attn1' ({scan_tops}). The scan exists to "
                f"test whether ONE attention layer is the topology between a saturated ceiling and "
                f"a saturated floor. Without it the run can only re-measure the two ends that are "
                f"already known to be saturated, and its 'NO DISCRIMINATING TOPOLOGY' verdict would "
                f"be an artifact of what was not run.",
                file=sys.stderr, flush=True,
            )
            return 2
        t0 = time.time()
        scan = scan_topologies(
            build_model=builder,
            config=scan_cfg,
            topologies=scan_tops,
            kernel_size=args.kernel_size,
            seeds=args.seeds,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            out_path=out,
            eval_items=args.eval_items,
            smoke=args.smoke,
            upload_dest=args.out if args.out != str(out) else None,
        )
        print()
        print(
            topology_report(
                scan["verdicts"],
                recommendation=str(scan["recommendation"]),
                vocab_size=scan_cfg.vocab_size,
                num_pairs=scan_cfg.num_pairs,
            ),
            flush=True,
        )
        print(f"\n{time.time() - t0:.0f}s of scan", flush=True)
        return _persist()

    if not args.skip_positive_control:
        for top in grid_topologies:
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
        topologies=grid_topologies,
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
    return _persist()


if __name__ == "__main__":
    raise SystemExit(main())
