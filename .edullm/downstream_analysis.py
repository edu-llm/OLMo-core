#!/usr/bin/env python3
"""Read the downstream scoring job, and answer what it can answer.

WRITTEN AND COMMITTED BEFORE THE SCORING JOB WAS SUBMITTED, WHICH IS THE ONLY TIME IT IS WORTH
WRITING. No document of ``edullm.hyper-connections.downstream.v1`` existed anywhere when this
file was committed: ``run.score-stage.yaml`` was staged and unsent, so every choice below --
which endpoint, which null for the slope, paired or unpaired, which contrasts are declared
underpowered and by how much, what the multiplicity family is -- was made against data that did
not exist and could not be preferred. ``hyper-connections.md`` carries the same section with the
same date. The tests drive the whole pipeline over synthetic tranches with a *planted* slope, so
the estimators are checked against a truth nobody could have chosen.

A FIFTH SIBLING, AND THE DIVISION OF LABOUR IS THE ONE THE OTHER FOUR KEEP. ``wandb_panels.py``
asks whether a metric key arrived. ``stage_gate.py`` asks whether a live submission is healthy.
``noise_floor.py`` holds the estimators. ``analysis.py`` asks what the arms say **in loop**.
This module asks the one question the in-loop analysis is constitutionally unable to ask: **what
happens to the loss improvement when it is carried onto a downstream suite**. It imports its
estimators from ``noise_floor`` and its contrast machinery, Bartlett test and block fit from
``analysis``, so the c4 correction, the randomized-block df and the exact noncentral t are the
same code the in-loop report is generated from, and it imports its *schema* from
``score_checkpoints`` rather than restating it, so the reader and the writer cannot drift apart.

THE PRIMARY IS A REGRESSION AND NOT AN ARM CONTRAST, AND THAT IS THE SUBSTANTIVE DECISION HERE.
The literature question this module exists to settle is not "is arm 2 better than arm 3
downstream" -- five against five cannot resolve the 0.0028 BPB the in-loop tranche implies for
that pair, and :data:`HYPOTHESES` says so in advance. It is whether **loss and downstream
decouple for this method**, which is exactly the thing the replication supplies no evidence
about: full-text verification item 20 found that Table 7 of arXiv 2605.20798 reports validation
loss for six methods and hyper-connections is not among them, so its residual-side
classification -- "loss increases consistent with their downstream drops" -- rests on no data
for the one mechanism this module is about. Twenty-five checkpoints spanning 0.0176 BPB of
in-loop endpoint measure that relationship directly and are the only thing here that is
well-powered against a question the field has actually left open. See :func:`regress`.

NOTHING HERE FALLS BACK TO ANYTHING, for the reason ``analysis.py`` gives at length:
``noise_floor.py --dry-run`` once printed a complete, internally consistent, entirely synthetic
report under a submission id it had been handed and never read, and it was acted on for twelve
hours. So:

* the measured path takes a directory of documents and reads exactly those. Twenty-five is the
  only acceptable number and :func:`completeness_refusals` names the missing cells one by one.
* a wrong schema, a duplicate cell, a cell that is not in the tranche table, a truncated score,
  a headline built over fewer groups than the suite declares, a missing task, or twenty-five
  documents that did not come off one instrument is a **refusal**, not a smaller analysis.
  ``--allow-provisional`` downgrades the ones that are about *completeness* and nothing else,
  and stamps ``PROVISIONAL`` on every artifact it writes.
* the synthetic path is a separate verb, ``--demo``, which cannot be reached from the measured
  one, writes under a ``synthetic-`` prefix and prefixes every line of its output.

    python .edullm/downstream_analysis.py --self-test                      # no network, no data
    python .edullm/downstream_analysis.py --demo --out downstream/demo     # synthetic, stamped
    python .edullm/downstream_analysis.py \\
        --documents downstream/documents \\
        --in-loop analysis/analysis.json \\
        --out downstream

Needs ``scipy``. It runs on a laptop and never inside a container, and it **does not reach
AWS**: see :func:`read_documents` for why the ``s3://`` prefix the scoring job writes to is not
something this program will open.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dose_adjustment  # noqa: E402
import hyper_connection_arms  # noqa: E402
import score_checkpoints  # noqa: E402
from noise_floor import (  # noqa: E402
    c4,
    mde,
    paired_correlation,
    pooled_sigma,
    power_of,
)

from analysis import (  # noqa: E402
    ARM_ORDER,
    GATE_SIGMAS,
    PRE_REGISTERED_BREAK_EVEN_RHO,
    SEEDS_PER_ARM,
    Refusal,
    bartlett,
    block_fit,
    break_even_rho,
    mde_from_se,
    welch,
)

#: The date every choice in this file was fixed on, and it means one specific thing: the
#: scoring job had not been submitted and no document of its schema existed anywhere. That is
#: a stronger claim than "before the analyst looked", and it is the one this module can make.
PRE_REGISTERED_ON = "2026-08-12"

#: The schema this reads, taken from the program that writes it rather than restated. A reader
#: and a writer that each carry their own copy of a format agree until one of them is edited.
INPUT_SCHEMA = score_checkpoints.OUTPUT_SCHEMA

#: The metric. Imported for the same reason: ``bpb_v2`` is a decision argued for at length at
#: ``score_checkpoints.PRIMARY_METRIC`` and repeating the string here would let the two drift.
PRIMARY_METRIC = score_checkpoints.PRIMARY_METRIC

#: The groups whose means make the headline, and the suite that defines them.
HEADLINE_GROUPS = score_checkpoints.HEADLINE_GROUPS
SUITE = score_checkpoints.SUITE_H2B
SUITE_VERSION = score_checkpoints.SUITE_VERSION

#: Every ``(arm, seed)`` there must be a document for. Twenty-five, derived from the same table
#: the scoring fan-out resolves its cells through, so an arm gaining a seed moves both.
EXPECTED_CELLS: Tuple[Tuple[str, int], ...] = tuple(hyper_connection_arms.TRANCHE_CELLS)

#: The step the tranche is read at.
FINAL_STEP = score_checkpoints.FINAL_STEP

#: The task labels in the headline, and the one that is not.
HEADLINE_TASKS: Tuple[str, ...] = tuple(t.label for t in SUITE if t.group in HEADLINE_GROUPS)
CANARY_TASKS: Tuple[str, ...] = tuple(t.label for t in SUITE if t.group not in HEADLINE_GROUPS)

#: Chance on ``copycolors_10way_fast``, which is ten-way over a hundred items. The canary is a
#: diagnostic about the *metric decision* and never an outcome: if accuracy is at chance the
#: write-up's claim that multiple-choice accuracy is uninformative at 370M is a measurement
#: rather than an assertion, and if it is not, the claim needs revisiting. Reported either way.
CANARY_CHANCE = 0.10
CANARY_ITEMS = 100

#: THE NULL THE PRIMARY TEST IS AGAINST, AND IT IS ONE RATHER THAN ZERO.
#:
#: Both axes are bits per byte of a gold continuation. A slope of one says the downstream
#: instrument is a rescaled copy of the in-loop one over this range: an arm that is 0.0146 BPB
#: better on held-out corpus is 0.0146 BPB better on the suite, and the downstream number
#: carries nothing the in-loop number did not already carry. That is the *coupled* world, and it
#: is what a residual-side classification of "loss increases consistent with downstream drops"
#: asserts. A slope materially below one is decoupling in the precise sense this module cares
#: about: the loss gain is real and does not arrive. A slope indistinguishable from zero is
#: total decoupling, and would say the in-loop tranche predicts nothing about the suite.
#:
#: ONE IS A REFERENCE VALUE AND NOT A LAW, AND THE WRITE-UP HAS TO SAY SO. The held-out corpus
#: is seven in-distribution sources of long text; the suite is thirteen short out-of-distribution
#: continuations. Nothing guarantees that the *rate* of transfer between two different text
#: distributions is exactly one even when nothing anybody would call decoupling is happening. So
#: the interval on the slope is the reported quantity, one and zero are the two landmarks it is
#: read against, and a rejection of one is written up as "transfer is not one-for-one" rather
#: than as "the method decouples".
SLOPE_NULL = 1.0

#: The landmarks the slope interval is read against, in the order the report prints them.
SLOPE_LANDMARKS: Tuple[float, ...] = (1.0, 0.5, 0.0)

#: The gate on the arm contrasts, and on the slope. Two standard errors, as pre-registered in
#: ``hyper-connections.md``, with the exact t p-value and this design's own 5% line beside it,
#: because two standard errors is not a 5% test once sigma is estimated.
GATE = GATE_SIGMAS

#: The in-loop noise floor, frozen, and the two figures that matter are different quantities.
#: The first is the *instrument's* floor -- the baseline's own five cells at df = 4, the only arm
#: whose scatter is the measurement and nothing else. The second pools all five arms at df = 20
#: and therefore carries whatever spread a treatment introduced. Both are in the in-loop report.
IN_LOOP_BASELINE_SIGMA_BPB = 0.00061
IN_LOOP_POOLED_SIGMA_BPB = 0.00147

#: WHERE THE LINE BETWEEN "POWERED" AND "DECLARED UNDER-POWERED" IS DRAWN, AND IT IS DRAWN FROM
#: ARITHMETIC RATHER THAN FROM TASTE.
#:
#: :func:`sigma_ceiling` turns each contrast's in-loop effect into the largest downstream noise
#: floor at which five against five still detects it. Those ceilings fall into two groups with a
#: factor of four between them and nothing in the gap: the three treatment-versus-baseline rows
#: tolerate 0.0063 to 0.0078 BPB, which is four to five times the in-loop **pooled** floor and
#: ten to thirteen times the baseline one, and the three arm-versus-arm rows need 0.0011 to
#: 0.0015, which is at or below the in-loop pooled floor. So the second group needs a downstream
#: instrument *quieter than the in-loop one*, and that is not plausible: downstream averages
#: about thirteen thousand short out-of-distribution continuations where the in-loop endpoint
#: averages millions of in-distribution tokens.
#:
#: The threshold is set at twice the in-loop pooled floor, which lands in the middle of the empty
#: gap, and ``test_downstream_analysis.py`` asserts that every ``declared_underpowered`` flag in
#: :data:`HYPOTHESES` is what this arithmetic gives from the frozen in-loop endpoints. **The
#: declaration is therefore derived and checked rather than asserted**, which is the property
#: that makes it survive somebody editing the table.
UNDERPOWERED_BELOW_BPB = 2.0 * IN_LOOP_POOLED_SIGMA_BPB


@dataclass(frozen=True)
class DownstreamHypothesis:
    """One pre-registered downstream contrast, its direction, and its power stated in advance."""

    name: str
    treatment: str
    comparator: str
    claim: str

    predicted_sign: int
    """
    ``-1`` where the hypothesis predicts the treatment *lowers* downstream bits-per-byte, and
    ``+1`` where it predicts a degradation. Both appear here, which they do not in the in-loop
    table: ``D2b-ii`` asks whether arm 3 reproduces a published *worsening*, so its predicted
    sign is positive and a negative result refutes the pre-registered explanation.
    """

    declared_underpowered: bool
    """
    Fixed here, on :data:`PRE_REGISTERED_ON`, from the in-loop effect and the ceiling on the
    downstream noise floor it implies -- see :func:`sigma_ceiling`. **The point of declaring it
    now is that a null on one of these is then reported as uninformative rather than as evidence
    of no effect**, which is a sentence nobody can write honestly after seeing the interval. The
    realised power is recomputed from the measured sigma-hat and printed beside the declaration;
    if the downstream instrument turns out quieter than anyone expects, the report says so, and
    the declaration was still the right thing to have made.
    """

    post_hoc: str = ""
    """Empty on everything pre-registered. See :data:`POST_HOC`."""


#: The downstream contrasts, fixed on :data:`PRE_REGISTERED_ON`.
#:
#: H2b HAS TWO HALVES AND ONLY ONE OF THEM IS AN ARM ORDERING. The pre-registration reads "Arm 2
#: > arm 3 on the downstream average, **and** arm 3 but not arm 2 reproduces the published
#: degradation." The first clause is ``D2b-i``, needs a downstream instrument quieter than the
#: in-loop one, and is declared underpowered below. The second clause is not an arm-versus-arm
#: question at all: it is two treatment-versus-baseline signs, ``D2b-ii`` and ``D1``, each of
#: which the design resolves. **So the half of H2b that carries the headline is the powered
#: half**, and reporting H2b as "blocked by power" would be wrong in the direction that matters.
#: :func:`published_degradation_verdict` assembles the conjunction.
HYPOTHESES: Tuple[DownstreamHypothesis, ...] = (
    DownstreamHypothesis(
        name="D1",
        treatment="faithful",
        comparator="baseline",
        claim="Arm 2 beats arm 1 on the downstream headline. The in-loop H1 carried onto the "
        "instrument the published claims are made on. JOINT in the mechanism and the sqrt(n) "
        "output-init rescale, exactly as H1 is. Also the second half of D2b-ii's conjunction: "
        "the published account needs arm 2 NOT to reproduce the degradation.",
        predicted_sign=-1,
        declared_underpowered=False,
    ),
    DownstreamHypothesis(
        name="D1a",
        treatment="no-output-init",
        comparator="baseline",
        claim="Arm 4 beats arm 1 downstream: the mechanism without the initialization "
        "prescription. Read beside D1, the part of the downstream effect the rescale is not "
        "responsible for.",
        predicted_sign=-1,
        declared_underpowered=False,
    ),
    DownstreamHypothesis(
        name="D2b-ii",
        treatment="output-only",
        comparator="baseline",
        claim="THE HALF OF H2b THAT IS POWERED. Does arm 3 -- the variant in the class of the "
        "published reimplementation, which drops the learned input map -- reproduce the "
        "published downstream DEGRADATION at 370M? Predicted sign is POSITIVE because the "
        "published result is a worsening. A negative result refutes this module's own "
        "pre-registered explanation of the inversion, on the instrument the inversion was "
        "measured on, which is what the in-loop H2a could not do.",
        predicted_sign=+1,
        declared_underpowered=False,
    ),
    DownstreamHypothesis(
        name="D2b-i",
        treatment="faithful",
        comparator="output-only",
        claim="The arm-ordering half of H2b: arm 2 above arm 3 on the downstream headline. The "
        "in-loop gap is 0.0028 BPB and reproducing it downstream at 80% power needs a "
        "downstream noise floor of 0.0015 BPB or less, against an in-loop floor of 0.00061 on "
        "the baseline. DECLARED UNDERPOWERED HERE, before the data: a null on this row is "
        "uninformative and will not be written up as an absence of difference.",
        predicted_sign=-1,
        declared_underpowered=True,
    ),
    DownstreamHypothesis(
        name="D5",
        treatment="mhc",
        comparator="faithful",
        claim="H5 downstream: arm 9 at least as good as arm 2. In loop this came back with the "
        "sign REVERSED at +0.0027 BPB, so what is pre-registered here is the same directional "
        "claim on the other instrument and not a re-test of the refutation. DECLARED "
        "UNDERPOWERED: 0.0027 BPB needs a downstream floor of 0.0015 or less.",
        predicted_sign=-1,
        declared_underpowered=True,
    ),
    DownstreamHypothesis(
        name="D1b",
        treatment="faithful",
        comparator="no-output-init",
        claim="H1b downstream: the sqrt(n) output-init prescription alone, at a fixed mechanism. "
        "The narrowest contrast in the tranche -- 0.0020 BPB in loop, needing a downstream floor "
        "of 0.0011 -- and the one already reported in loop as under-powered by 2.3x for its own "
        "point estimate. DECLARED UNDERPOWERED here for the same reason and in advance.",
        predicted_sign=-1,
        declared_underpowered=True,
    ),
)


#: Everything this module reports that was not fixed on :data:`PRE_REGISTERED_ON`, with the date
#: it was added. **Empty, and the report says so in as many words.** The machinery is live from
#: the first run so that anything added after the documents land has somewhere to go and cannot
#: be added without a label: a contrast that is not in the pre-registration and does not say so
#: is the failure this whole family of modules is built around.
POST_HOC: Tuple[Tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------------------
# Reading the documents. Real data only, addressed by directory, and never from AWS.
# ---------------------------------------------------------------------------------------


@dataclass
class Cell:
    """One ``(arm, seed)`` of the tranche, from its downstream document and its in-loop endpoint."""

    arm: str
    seed: int
    path: str

    downstream_bpb: float
    """The headline: the mean over the group means of :data:`HEADLINE_GROUPS`, in bits per byte."""

    groups: Dict[str, float] = field(default_factory=dict)
    """Per-group mean of the primary metric."""

    tasks: Dict[str, float] = field(default_factory=dict)
    """Per-task primary metric, all thirteen, canary included."""

    canary_accuracy: Optional[float] = None
    """``len_norm_v2`` on the canary, or None if the canary did not report it."""

    in_loop_bpb: float = float("nan")
    """The frozen in-loop endpoint of the same cell. Attached by :func:`attach_in_loop`."""

    run_id: str = ""
    checkpoint: str = ""
    warnings: Tuple[str, ...] = ()

    instrument: Tuple[Tuple[str, str], ...] = ()
    """
    The fields that have to be identical across all twenty-five or the numbers did not come off
    one instrument -- suite, suite version, primary metric, tokenizer, dtype, device, torch. Kept
    as a tuple of pairs so a mismatch can be printed as the pair that differed.
    """


def _finite(value: object) -> Optional[float]:
    """
    A float if this is one and is finite, else None. ``None`` and ``NaN`` are different kinds of
    absent and both have to stop the arithmetic rather than propagate through it.

    :param value: Whatever was in the document.

    :returns: The float, or None.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_documents(where: str) -> List[Tuple[str, Dict[str, object]]]:
    """
    Every ``downstream-*.json`` in a local directory, or the explicit file named.

    LOCAL ONLY, AND THE ``s3://`` PREFIX THE JOB WRITES TO IS REFUSED RATHER THAN OPENED.
    ``AGENTS.md`` forbids a script here from calling AWS -- no ``boto3``, no ``aws`` CLI, no
    ``curl`` at an AWS endpoint -- because for most people it fails and for the few it does not
    it succeeds and leaves no run anybody can cite. ``olmo_core.io`` would open the prefix
    happily and that is exactly the shortcut the rule exists to stop. The documents are also on
    each cell's stdout by construction (``score_checkpoints.write_document`` prints before it
    uploads, so that a failed S3 write never loses the numbers), so ``edullm logs`` is a second
    route that needs no credential at all.

    :param where: A directory of documents, or one document.

    :returns: ``(path, document)`` in filename order. The path travels with the document
        because every refusal below names the file it is about, and a refusal that says only
        "a document" sends somebody to open twenty-five of them.

    :raises Refusal: If the path is a URL, does not exist, holds nothing that looks like a
        document, or holds a file that is not JSON.
    """
    if "://" in where:
        raise Refusal(
            f"'{where}' is a URL and this program will not open one. AGENTS.md forbids a script "
            "in this repository from reaching AWS, and reading the scoring job's output prefix "
            "would be exactly that. Bring the documents down by whatever route you already use "
            "for that bucket and point --documents at the directory, or recover them from the "
            "cells' logs: score_checkpoints prints each document on stdout before it uploads it."
        )
    if not os.path.exists(where):
        raise Refusal(f"--documents {where} does not exist.")

    if os.path.isfile(where):
        paths = [where]
    else:
        # RECURSIVELY, BECAUSE THE TWENTY-FIVE DO NOT LAND IN ONE DIRECTORY. The platform's
        # fan-out prologue appends `cell-<index>/` to `$EDULLM_OUTPUT_PREFIX` before the command
        # runs, so `--output-dir "$EDULLM_OUTPUT_PREFIX"` puts each document under a directory of
        # its own and a downloaded prefix is twenty-five folders holding one file each. A flat
        # listing of that finds nothing, which would read exactly like a job that has not
        # started -- and a flat listing of a *partially* downloaded one would find some.
        paths = sorted(
            os.path.join(root, name)
            for root, _, names in os.walk(where)
            for name in names
            if name.startswith("downstream-") and name.endswith(".json")
        )
    if not paths:
        raise Refusal(
            f"no 'downstream-*.json' anywhere under {where}. That is what an empty output prefix "
            "looks like and it is also what a directory of documents under some other name looks "
            "like, and the two are worth telling apart before anything is concluded from either. "
            "The scoring fan-out writes one document per cell under its own 'cell-<index>/', so "
            "point --documents at the prefix rather than at one cell."
        )

    documents = []
    for path in paths:
        try:
            with open(path) as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError) as unreadable:
            raise Refusal(f"{path}: {type(unreadable).__name__}: {unreadable}") from None
        if not isinstance(loaded, dict):
            raise Refusal(f"{path} holds a {type(loaded).__name__} and not a document.")
        documents.append((path, loaded))
    return documents


def cell_from_document(document: Mapping[str, object], path: str) -> Cell:
    """
    One document, checked against the schema it claims and reduced to what the analysis needs.

    EVERY CHECK HERE IS A CASE WHERE THE ARITHMETIC WOULD HAVE SUCCEEDED. A truncated score is a
    number. A headline averaged over two groups instead of three is a number. A cell scored
    under an arm it did not train as is a number, and it is the one that would be hardest to
    ever notice. None of them is a number about this experiment.

    :param document: The parsed document.
    :param path: Where it came from, for the refusal text.

    :returns: The cell.

    :raises Refusal: On any of the above.
    """
    schema = document.get("schema")
    if schema != INPUT_SCHEMA:
        raise Refusal(
            f"{path} declares schema {schema!r} and this reads {INPUT_SCHEMA!r}. A document of "
            "another shape may still have every key this needs and mean something else by them."
        )

    arm = str(document.get("arm", ""))
    seed_value = document.get("seed")
    if not isinstance(seed_value, int) or isinstance(seed_value, bool):
        raise Refusal(f"{path} has seed {seed_value!r}, which is not a replicate index.")
    seed = int(seed_value)
    if (arm, seed) not in EXPECTED_CELLS:
        raise Refusal(
            f"{path} is arm {arm!r} seed {seed}, which is not a cell of the tranche. "
            f"hyper_connection_arms.TRANCHE_CELLS has {len(EXPECTED_CELLS)} cells and this is "
            "not one of them."
        )

    declared_number = document.get("arm_number")
    expected_number = hyper_connection_arms.ARMS[arm].number
    if declared_number != expected_number:
        raise Refusal(
            f"{path} says arm {arm!r} is arm number {declared_number} and the arm table says "
            f"{expected_number}. One of the two is describing a different experiment."
        )

    if document.get("step") != FINAL_STEP:
        raise Refusal(
            f"{path} scored step {document.get('step')} and the tranche is read at "
            f"{FINAL_STEP}. Two arms compared at different steps is the endpoint misalignment "
            "this module has already been bitten by once."
        )

    if document.get("truncated"):
        raise Refusal(
            f"{path} is marked truncated, which means --limit-batches stopped a task early. A "
            "truncated task is not a score of that task; score_checkpoints records the flag for "
            "exactly this reason and it is not a flag to read past."
        )

    downstream = document.get("downstream")
    if not isinstance(downstream, dict) or PRIMARY_METRIC not in downstream:
        raise Refusal(f"{path} has no 'downstream' aggregate for {PRIMARY_METRIC}.")
    aggregate = downstream[PRIMARY_METRIC]
    if not isinstance(aggregate, dict):
        raise Refusal(f"{path}: 'downstream.{PRIMARY_METRIC}' is not an aggregate.")

    headline = _finite(aggregate.get("headline"))
    if headline is None:
        raise Refusal(
            f"{path} has a null or non-finite headline. score_checkpoints returns None when no "
            "headline group reported the metric at all, which is a scoring failure that wrote a "
            "document rather than a score."
        )

    used = tuple(aggregate.get("headline_groups") or ())
    if used != tuple(HEADLINE_GROUPS):
        raise Refusal(
            f"{path} built its headline from groups {list(used)} and the suite declares "
            f"{list(HEADLINE_GROUPS)}. A headline over fewer groups is a different quantity "
            "with the same name, and averaging it beside a complete one compares two things."
        )

    tasks_field = document.get("tasks")
    if not isinstance(tasks_field, dict):
        raise Refusal(f"{path} has no 'tasks' block.")
    tasks: Dict[str, float] = {}
    missing = []
    for task in SUITE:
        entry = tasks_field.get(task.label)
        value = (
            _finite(entry.get("metrics", {}).get(PRIMARY_METRIC))
            if isinstance(entry, dict)
            else None
        )
        if value is None:
            missing.append(task.label)
        else:
            tasks[task.label] = value
    if missing:
        raise Refusal(
            f"{path} has no finite {PRIMARY_METRIC} for {', '.join(missing)}. A task missing "
            "from a group silently changes that group's mean and therefore the headline, and "
            "the headline would still print."
        )

    canary_entry = tasks_field.get(CANARY_TASKS[0]) if CANARY_TASKS else None
    canary = (
        _finite(canary_entry.get("metrics", {}).get("len_norm_v2"))
        if isinstance(canary_entry, dict)
        else None
    )

    return Cell(
        arm=arm,
        seed=seed,
        path=path,
        downstream_bpb=headline,
        groups={str(k): float(v) for k, v in (aggregate.get("groups") or {}).items()},
        tasks=tasks,
        canary_accuracy=canary,
        run_id=str(document.get("run_id", "")),
        checkpoint=str(document.get("checkpoint", "")),
        warnings=tuple(str(w) for w in (document.get("warnings") or ())),
        instrument=tuple(
            (key, str(document.get(key)))
            for key in (
                "suite",
                "suite_version",
                "primary_metric",
                "tokenizer",
                "param_dtype",
                "device",
                "torch",
            )
        ),
    )


def completeness_refusals(cells: Sequence[Cell]) -> List[str]:
    """
    Which cells are missing, and which arrived twice. **Named one by one, and never counted.**

    THE ONE BEHAVIOUR THAT IS NOT ALLOWED IS COMPUTING WITH TWENTY-FOUR AND NOT SAYING SO. A
    missing cell moves an arm mean, drops a df, narrows an interval and does none of it at
    random: a cell fails by hitting a wall, by losing a host or by finding no checkpoint, so the
    cells that leave are the slow ones and the unlucky ones and the survivors are biased in a
    direction nobody chose. It also silently unbalances the block, and the paired analysis pairs
    arm *a* seed *k* with arm *b* seed *k*.

    :param cells: The cells as read.

    :returns: One sentence per problem, empty when all twenty-five are present exactly once.
    """
    complaints = []
    seen: Dict[Tuple[str, int], List[str]] = {}
    for cell in cells:
        seen.setdefault((cell.arm, cell.seed), []).append(cell.path)

    absent = [key for key in EXPECTED_CELLS if key not in seen]
    if absent:
        complaints.append(
            f"{len(absent)} of {len(EXPECTED_CELLS)} cells have no document: "
            + ", ".join(f"{arm} seed {seed}" for arm, seed in absent)
            + "."
        )
    for key, paths in sorted(seen.items()):
        if len(paths) > 1:
            complaints.append(
                f"{key[0]} seed {key[1]} has {len(paths)} documents ({', '.join(paths)}). Two "
                "scores of one cell is either a re-run whose loser was never deleted or two "
                "cells that resolved to the same (arm, seed), and the second is a fan-out bug."
            )
    return complaints


def instrument_refusals(cells: Sequence[Cell]) -> List[str]:
    """
    Whether all twenty-five numbers came off one instrument.

    NOT DOWNGRADABLE BY ``--allow-provisional``, AND THE REASON IS THE SIZE OF THE EFFECT. The
    widest contrast this design is looking for is 0.0146 BPB and the narrowest is 0.0020. A
    second GPU model, a second torch, a second dtype or a second tokenizer changes the
    arithmetic somewhere below that, in a direction that is fixed per instrument and therefore
    lines up with whichever cells happened to land on it. That is not noise, it is a covariate,
    and it would be perfectly invisible in the output.

    :param cells: The cells as read.

    :returns: One sentence per field that is not constant across the cells.
    """
    complaints = []
    keys = [key for key, _ in cells[0].instrument] if cells else []
    for index, key in enumerate(keys):
        values: Dict[str, List[str]] = {}
        for cell in cells:
            values.setdefault(cell.instrument[index][1], []).append(f"{cell.arm}/{cell.seed}")
        if len(values) > 1:
            described = "; ".join(
                f"{value!r} on {len(where)} cell(s) [{', '.join(where[:4])}"
                + (", ..." if len(where) > 4 else "")
                + "]"
                for value, where in sorted(values.items())
            )
            complaints.append(
                f"the cells disagree on {key!r}: {described}. Twenty-five numbers off two "
                "instruments is a covariate that lines up with the cells, not noise."
            )
    return complaints


# ---------------------------------------------------------------------------------------
# The other axis: the frozen in-loop endpoint, joined to the downstream cells.
# ---------------------------------------------------------------------------------------


def in_loop_from_artifact(path: str) -> Tuple[Dict[Tuple[str, int], float], str]:
    """
    The in-loop endpoint per cell, from the artifact ``analysis.py`` writes.

    THE FROZEN ARTIFACT IS PREFERRED OVER A FRESH W&B READ, WHICH IS THE OPPOSITE OF THE USUAL
    ADVICE AND IS RIGHT HERE. The x-axis of the primary regression is the *pre-registered
    in-loop endpoint*: the number the in-loop report is written against and the write-up quotes.
    Recomputing it from W&B at downstream-analysis time could produce a different number -- a
    re-run cell, a repaired summary, a changed reader -- and then the regression would relate a
    downstream measurement to an in-loop measurement nobody published. Reading the artifact
    relates it to the published one.

    :param path: ``analysis/analysis.json``.

    :returns: ``({(arm, seed): bpb}, provenance)``.

    :raises Refusal: If the artifact is unreadable, is not at the tranche's step, or is stamped
        provisional -- a provisional in-loop read cannot be the fixed axis of anything.
    """
    try:
        with open(path) as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as unreadable:
        raise Refusal(f"--in-loop {path}: {type(unreadable).__name__}: {unreadable}") from None

    if artifact.get("label") != "measured":
        raise Refusal(
            f"{path} is labelled {artifact.get('label')!r}. Only a 'measured' in-loop artifact "
            "may be the x-axis of the regression; a synthetic one would produce a slope."
        )
    if artifact.get("provisional"):
        raise Refusal(
            f"{path} is stamped provisional ({'; '.join(artifact['provisional'])}). The in-loop "
            "endpoint is the fixed axis of the primary analysis and a provisional one is not "
            "fixed. Re-run analysis.py without --allow-provisional."
        )
    if artifact.get("compared_at_step") != FINAL_STEP:
        raise Refusal(
            f"{path} was compared at step {artifact.get('compared_at_step')} and the downstream "
            f"job scored step {FINAL_STEP}. Those are two different models."
        )

    endpoints: Dict[Tuple[str, int], float] = {}
    for entry in artifact.get("arms", []):
        arm = str(entry.get("arm"))
        seeds = entry.get("seeds") or []
        values = entry.get("endpoint_bpb") or []
        if len(seeds) != len(values):
            raise Refusal(
                f"{path}: arm {arm!r} has {len(seeds)} seeds and {len(values)} endpoints."
            )
        for seed, value in zip(seeds, values):
            number = _finite(value)
            if number is None:
                raise Refusal(f"{path}: arm {arm!r} seed {seed} has a non-finite endpoint.")
            endpoints[(arm, int(seed))] = number
    return endpoints, f"{path} (generated {artifact.get('generated')})"


def in_loop_from_wandb(
    arm_submissions: Mapping[str, str], entity: str, project: str, group: str
) -> Tuple[Dict[Tuple[str, int], float], str]:
    """
    The same numbers, recomputed from W&B through ``analysis.read_arm``.

    THE READER IS REUSED RATHER THAN REWRITTEN, AND THAT IS THE WHOLE POINT OF THE FUNCTION. A
    crash reporter that called ``wandb.init`` with ``WANDB_RUN_ID`` still set once overwrote
    seven cells' summaries with a diagnostic, so a cell that had run 4,910 steps read back as a
    cell that never started. ``analysis.read_arm`` reads the **history** and reconciles it
    against the summary, excludes crash reports, checks each cell's own saved config against the
    arm it is supposed to be, and refuses a seed collision. Any second reader written here would
    be a second chance to re-make all of that, and it would be written by somebody who had
    already seen the downstream numbers.

    :param arm_submissions: ``{arm: submission id or prefix}``, one per funded arm.
    :param entity: W&B entity.
    :param project: W&B project.
    :param group: The experiment slug.

    :returns: ``({(arm, seed): bpb}, provenance)``.

    :raises Refusal: Whatever ``read_arm`` refuses, plus a cell that did not reach the horizon.
    """
    from analysis import common_endpoint_step, read_arm

    series = [read_arm(entity, project, group, arm, run) for arm, run in arm_submissions.items()]
    at_step = common_endpoint_step(series)
    if at_step != FINAL_STEP:
        raise Refusal(
            f"the arms share a last evaluation at step {at_step} and the downstream job scored "
            f"{FINAL_STEP}."
        )
    endpoints: Dict[Tuple[str, int], float] = {}
    for arm in series:
        matrix = arm.endpoint_matrix(at_step)
        for seed, row in zip(arm.seeds, matrix):
            endpoints[(arm.arm, int(seed))] = float(np.mean(row))
    return endpoints, "W&B, via analysis.read_arm: " + ", ".join(
        f"{arm}={run}" for arm, run in sorted(arm_submissions.items())
    )


def attach_in_loop(cells: Sequence[Cell], endpoints: Mapping[Tuple[str, int], float]) -> None:
    """
    Join the in-loop endpoint onto each downstream cell, in place.

    :param cells: The downstream cells.
    :param endpoints: ``{(arm, seed): in-loop BPB}``.

    :raises Refusal: If any downstream cell has no in-loop endpoint. There is no imputing an
        x-value: a regression that drops the cells it cannot join drops them non-randomly.
    """
    absent = [(c.arm, c.seed) for c in cells if (c.arm, c.seed) not in endpoints]
    if absent:
        raise Refusal(
            "the in-loop endpoint is missing for "
            + ", ".join(f"{arm} seed {seed}" for arm, seed in absent)
            + ". Every downstream cell needs its own x-value; dropping the ones that cannot be "
            "joined drops them for a reason correlated with whatever went wrong in training."
        )
    for cell in cells:
        cell.in_loop_bpb = float(endpoints[(cell.arm, cell.seed)])


# ---------------------------------------------------------------------------------------
# The primary: the loss-to-downstream slope.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SlopeTest:
    """One slope read against one landmark, with the same gate the arm contrasts carry."""

    null: float
    t_statistic: float
    p_value: float
    gate: float
    clears_gate: bool
    five_percent: float
    clears_five_percent: bool
    mde: float
    clears_mde: bool


@dataclass(frozen=True)
class SlopeFit:
    """A straight line through the tranche, and everything the gate needs to be applied to it."""

    label: str
    n: int
    slope: float
    intercept_at_mean: float
    x_mean: float
    se_slope: float
    df: int

    residual_sd: float
    """
    THE ONE NUMBER THE WHOLE POWER STORY TURNS ON. Everything the design can and cannot resolve
    is this divided by the square root of the x spread, and it is the downstream instrument's
    noise as seen by the regression rather than as seen by an arm contrast. Printed in bits per
    byte so it can be read directly against the 0.00061 in-loop floor.
    """

    s_xx: float
    r_squared: float
    ci: Tuple[float, float]
    tests: Tuple[SlopeTest, ...]


def _slope_tests(slope: float, se: float, df: int, nulls: Sequence[float]) -> Tuple[SlopeTest, ...]:
    """
    The gate, the exact t p-value, the 5% line and the MDE, for each landmark.

    The c4 correction enters exactly once, inside ``mde_from_se``, and deliberately not in the t
    statistic or the interval: that machinery is built on the distribution of ``s`` and already
    carries the bias, and correcting it twice is the mistake the in-loop module documents.

    :param slope: The fitted slope.
    :param se: Its standard error.
    :param df: Residual degrees of freedom.
    :param nulls: The landmarks to test against.

    :returns: One test per landmark.
    """
    critical = float(stats.t.ppf(0.975, df))
    detectable = mde_from_se(se, df)
    out = []
    for null in nulls:
        delta = slope - null
        t_statistic = delta / se
        out.append(
            SlopeTest(
                null=float(null),
                t_statistic=float(t_statistic),
                p_value=float(2.0 * stats.t.sf(abs(t_statistic), df)),
                gate=GATE * se,
                clears_gate=bool(abs(delta) >= GATE * se),
                five_percent=critical * se,
                clears_five_percent=bool(abs(delta) >= critical * se),
                mde=float(detectable),
                clears_mde=bool(abs(delta) >= detectable),
            )
        )
    return tuple(out)


def regress(
    x: Sequence[float], y: Sequence[float], label: str, extra_parameters: int = 0
) -> SlopeFit:
    """
    Ordinary least squares of downstream bits-per-byte on in-loop bits-per-byte.

    THE MODEL IS ``y = alpha + beta (x - xbar) + eps`` AND THE PRIMARY TEST IS ``beta = 1``. See
    :data:`SLOPE_NULL` for what one means and why it is a landmark rather than a law. Centring
    ``x`` costs nothing, makes ``alpha`` the downstream headline of a cell at the tranche's mean
    in-loop endpoint, and decorrelates the two estimates so the intercept's interval is readable.

    NEITHER AXIS IS MEASURED WITH ERROR IN THE SENSE THAT WOULD ATTENUATE THIS SLOPE, AND IT IS
    WORTH SAYING BECAUSE THE BIAS WOULD POINT EXACTLY AT THE FINDING. Classical errors-in-
    variables shrinks a slope towards zero, which is the direction "decoupling" lives in, so an
    analysis that ignored it could manufacture the result. It does not apply: ``x`` is not a
    noisy estimate of a cell's true in-loop endpoint, it *is* that cell's endpoint, a
    deterministic read of one checkpoint against one fixed held-out set, and ``y`` is the same
    read of the same checkpoint against one fixed suite. The 0.00061 BPB in-loop noise floor is
    variation *between cells*, which is signal here and is the spread the regression rests on,
    not error in the predictor. What remains is finite-evaluation-set sampling, which is common
    to every cell because every cell is scored on the same documents, and therefore moves the
    intercept rather than the slope.

    :param x: In-loop endpoint per cell.
    :param y: Downstream headline per cell.
    :param label: What this fit is, for the report.
    :param extra_parameters: Parameters fitted elsewhere in the same model that this slope's
        residual df has to pay for -- four for the arm dummies of the within-arm fit, zero for a
        plain two-parameter line.

    :returns: The fit.

    :raises Refusal: If there are fewer than three points, or if ``x`` does not vary.
    """
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    if xs.size != ys.size:
        raise Refusal(f"{xs.size} in-loop values against {ys.size} downstream ones.")
    df = xs.size - 2 - extra_parameters
    if df < 1:
        raise Refusal(
            f"a line through {xs.size} points with {2 + extra_parameters} parameters has "
            f"{df} residual degrees of freedom, so it has no interval and no test."
        )

    centred = xs - xs.mean()
    s_xx = float((centred**2).sum())
    if not s_xx > 0.0:
        raise Refusal(
            "every cell has the same in-loop endpoint, so there is no x spread to regress on. "
            "That is what one run read twenty-five times looks like."
        )
    slope = float((centred * (ys - ys.mean())).sum() / s_xx)
    intercept = float(ys.mean())
    residuals = ys - (intercept + slope * centred)
    rss = float((residuals**2).sum())
    residual_sd = math.sqrt(rss / df)
    se = residual_sd / math.sqrt(s_xx)
    total = float(((ys - ys.mean()) ** 2).sum())
    critical = float(stats.t.ppf(0.975, df))

    return SlopeFit(
        label=label,
        n=int(xs.size),
        slope=slope,
        intercept_at_mean=intercept,
        x_mean=float(xs.mean()),
        se_slope=float(se),
        df=int(df),
        residual_sd=float(residual_sd),
        s_xx=s_xx,
        r_squared=float(1.0 - rss / total) if total > 0 else float("nan"),
        ci=(slope - critical * se, slope + critical * se),
        tests=_slope_tests(slope, se, int(df), SLOPE_LANDMARKS),
    )


@dataclass(frozen=True)
class OneLineCheck:
    """Whether the five arms sit on one line, and what the pre-committed consequence is."""

    f_statistic: float
    df_numerator: int
    df_denominator: int
    p_value: float
    rejects: bool
    arm_offsets: Tuple[Tuple[str, float], ...]
    """Each arm's mean vertical distance from the common line, in bits per byte."""


def one_line_check(
    x: Sequence[float], y: Sequence[float], arms: Sequence[str], alpha: float = 0.05
) -> OneLineCheck:
    """
    Do the arms sit on the common line, or does each sit at its own height above it?

    THE PRE-REGISTERED VALIDITY CHECK ON THE PRIMARY, AND IT CAN WITHHOLD A CLAIM AND CAN NEVER
    CREATE ONE -- the same shape of rule the training-dose band already has. The pooled fit
    treats twenty-five cells as twenty-five independent draws around one line. They are not:
    they are five clusters of five, and an arm can sit off the line for reasons that have
    nothing to do with its loss. If it does, the residuals are correlated inside an arm, the
    pooled standard error is too small, and the interval printed beside the slope is narrower
    than the data support.

    Concretely: fit ``y ~ x`` and ``y ~ x + arm`` and F-test the four arm dummies on ``(4, 19)``.
    If it does not reject, the single line describes the tranche at both levels and the pooled
    slope stands as primary. **If it rejects, the pooled row is not withdrawn but it stops being
    primary**: the arm-mean fit at ``df = 3``, which treats the arm as the experimental unit and
    assumes nothing about clustering, becomes the reported reading, and the report says which
    rule fired.

    :param x: In-loop endpoint per cell.
    :param y: Downstream headline per cell.
    :param arms: Arm name per cell.
    :param alpha: Significance level.

    :returns: The check.

    :raises Refusal: If the long model has no residual degrees of freedom.
    """
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    names = list(dict.fromkeys(arms))
    n, k = xs.size, len(names)
    df_long = n - (2 + k - 1)
    if k < 2 or df_long < 1:
        raise Refusal(
            f"{k} arms over {n} cells leaves {df_long} residual degrees of freedom once every "
            "arm has its own height, so there is nothing to test the heights against."
        )

    design_short = np.column_stack([np.ones(n), xs - xs.mean()])
    dummies = np.column_stack([[1.0 if a == name else 0.0 for a in arms] for name in names[1:]])
    design_long = np.column_stack([design_short, dummies])

    def rss_of(design: np.ndarray) -> Tuple[float, np.ndarray]:
        coefficients, *_ = np.linalg.lstsq(design, ys, rcond=None)
        residual = ys - design @ coefficients
        return float((residual**2).sum()), coefficients

    rss_short, _ = rss_of(design_short)
    rss_long, _ = rss_of(design_long)
    numerator_df = k - 1
    statistic = ((rss_short - rss_long) / numerator_df) / (rss_long / df_long)
    p_value = float(stats.f.sf(statistic, numerator_df, df_long))

    line = design_short @ np.linalg.lstsq(design_short, ys, rcond=None)[0]
    offsets = tuple(
        (name, float(np.mean([ys[i] - line[i] for i, a in enumerate(arms) if a == name])))
        for name in names
    )
    return OneLineCheck(
        f_statistic=float(statistic),
        df_numerator=int(numerator_df),
        df_denominator=int(df_long),
        p_value=p_value,
        rejects=bool(p_value < alpha),
        arm_offsets=offsets,
    )


def within_arm_slope(x: Sequence[float], y: Sequence[float], arms: Sequence[str]) -> SlopeFit:
    """
    The slope with every arm given its own height: seed-level coupling, not intervention-level.

    A DIFFERENT QUESTION FROM THE PRIMARY AND IT IS REPORTED AS ONE. This asks whether a
    *replicate* that happened to land at a lower in-loop endpoint also lands lower downstream,
    which is a statement about the two instruments' shared noise, not about what an intervention
    buys. It is printed because the difference between it and the between-arm slope is the whole
    content of :func:`one_line_check`, and because a reader will otherwise wonder.

    IT WILL ALMOST CERTAINLY BE UNINFORMATIVE AND THAT IS ARITHMETIC, NOT PESSIMISM. On the
    frozen in-loop endpoints 94% of the x spread is between arms: the within-arm sum of squares
    is 4.33e-05 against 6.70e-04 between, so a within-arm slope's standard error is 3.9 times
    the pooled one at the same residual scatter. Whatever downstream floor makes the primary
    readable leaves this row four times wider.

    :param x: In-loop endpoint per cell.
    :param y: Downstream headline per cell.
    :param arms: Arm name per cell.

    :returns: The fit, on ``n - k - 1`` degrees of freedom.
    """
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    names = list(dict.fromkeys(arms))
    means_x = {
        name: float(np.mean([xs[i] for i, a in enumerate(arms) if a == name])) for name in names
    }
    means_y = {
        name: float(np.mean([ys[i] for i, a in enumerate(arms) if a == name])) for name in names
    }
    centred_x = np.asarray([xs[i] - means_x[a] for i, a in enumerate(arms)])
    centred_y = np.asarray([ys[i] - means_y[a] for i, a in enumerate(arms)])
    # The arm means are fitted parameters, so the residual df is n - k - 1 and not n - 2. Passed
    # through `extra_parameters` rather than corrected afterwards, so the interval, the p-value
    # and the MDE all rest on the same count.
    fit = regress(centred_x, centred_y, "within-arm", extra_parameters=len(names) - 1)
    return fit


def arm_mean_slope(x: Sequence[float], y: Sequence[float], arms: Sequence[str]) -> SlopeFit:
    """
    The slope through the five arm means: the intervention-level relationship, at ``df = 3``.

    THE CONSERVATIVE COMPANION TO THE PRIMARY AND IT IS ALWAYS PRINTED. It treats the arm as the
    experimental unit, which is what an arm is, and it therefore cannot be wrong about clustering
    -- there is nothing left inside a cluster for it to be wrong about. What it costs is degrees
    of freedom: ``t(0.975, 3) = 3.18`` against the pooled fit's ``2.07``, so its interval is
    about 54% wider at the same residual scatter. What it does *not* cost is leverage, because
    almost all of the x spread is between arms in the first place.

    :param x: In-loop endpoint per cell.
    :param y: Downstream headline per cell.
    :param arms: Arm name per cell.

    :returns: The fit over the arm means.
    """
    names = list(dict.fromkeys(arms))
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    mx = [float(np.mean([xs[i] for i, a in enumerate(arms) if a == name])) for name in names]
    my = [float(np.mean([ys[i] for i, a in enumerate(arms) if a == name])) for name in names]
    return regress(mx, my, "arm means")


def slope_power_table(s_xx: float, scatters: Sequence[float], df: int) -> List[Dict[str, float]]:
    """
    What the slope can resolve, as a function of the downstream residual scatter.

    WRITTEN AS A FUNCTION OF THE UNKNOWN RATHER THAN AT A GUESSED VALUE, WHICH IS THE ONLY WAY
    TO PRE-REGISTER POWER FOR AN INSTRUMENT NOBODY HAS RUN. The x spread is frozen and known --
    it is the in-loop tranche -- so ``SE(beta) = s_resid / sqrt(S_xx)`` is known up to one
    number, and the table says what each value of that number buys. It is printed before the
    data lands and again beside the measured scatter afterwards.

    :param s_xx: The sum of squared deviations of the in-loop endpoint.
    :param scatters: Candidate residual standard deviations, in bits per byte.
    :param df: Residual degrees of freedom of the fit.

    :returns: One row per candidate scatter.
    """
    rows = []
    for scatter in scatters:
        se = scatter / math.sqrt(s_xx)
        rows.append(
            {
                "residual_sd": float(scatter),
                "se_slope": float(se),
                "half_width": float(stats.t.ppf(0.975, df) * se),
                "mde": float(mde_from_se(se, df)),
                "power_against_zero": float(power_of(1.0, se, df)),
                "power_against_half": float(power_of(0.5, se, df)),
            }
        )
    return rows


# ---------------------------------------------------------------------------------------
# The per-task profile, and the discipline that stops it being thirteen chances.
# ---------------------------------------------------------------------------------------


def per_task_slopes(cells: Sequence[Cell]) -> Dict[str, object]:
    """
    The same regression on each task separately, as a **descriptive profile** and nothing else.

    THIRTEEN TASKS ARE THIRTEEN CHANCES TO FIND SOMETHING AND THE DISCIPLINE IS FIXED HERE.

    1. The headline regression is the sole primary. Nothing in this function is in the
       confirmatory family, nothing here carries a gate, and no sentence of the write-up may
       lead with a per-task slope.
    2. Every per-task p-value is printed **Holm-adjusted across the twelve headline tasks**,
       with the raw value in the same row. It is not possible to read an unadjusted per-task
       p-value out of this report without also reading the adjusted one.
    3. The canary is outside that family as well as outside the headline, and is labelled.
    4. **There is no omnibus test and that is deliberate.** The twelve series come off the same
       twenty-five checkpoints, so their slopes are correlated; a Wald test of slope homogeneity
       would need a 12 x 12 residual covariance estimated from twenty-five points on 23 df, and
       nothing about that statistic's small-sample calibration is known. Holm assumes nothing,
       is valid under any dependence, and is conservative under the positive dependence this
       has. What replaces the omnibus is descriptive: the range of the twelve slopes against
       their own median standard error, so a reader can see at a glance whether the spread is
       larger than the noise, plus the median pairwise correlation of the twelve residual series
       so the reader knows how conservative Holm is being.

    :param cells: The cells, with in-loop endpoints attached.

    :returns: The profile.
    """
    xs = [cell.in_loop_bpb for cell in cells]
    rows = []
    residuals: Dict[str, np.ndarray] = {}
    for task in SUITE:
        ys = [cell.tasks[task.label] for cell in cells]
        fit = regress(xs, ys, task.label)
        centred = np.asarray(xs) - float(np.mean(xs))
        residuals[task.label] = np.asarray(ys) - (fit.intercept_at_mean + fit.slope * centred)
        rows.append(
            {
                "task": task.label,
                "group": task.group,
                "in_headline": task.group in HEADLINE_GROUPS,
                "slope": fit.slope,
                "se": fit.se_slope,
                "df": fit.df,
                "ci": list(fit.ci),
                "residual_sd": fit.residual_sd,
                "r_squared": fit.r_squared,
                "p_against_one": fit.tests[0].p_value,
            }
        )

    family = {r["task"]: float(r["p_against_one"]) for r in rows if r["in_headline"]}
    adjusted = dose_adjustment.holm_adjust(family)
    for row in rows:
        row["holm_adjusted_p"] = adjusted.get(str(row["task"]))

    headline_rows = [r for r in rows if r["in_headline"]]
    slopes = [float(r["slope"]) for r in headline_rows]
    labels = [str(r["task"]) for r in headline_rows]
    pairwise = [
        float(np.corrcoef(residuals[a], residuals[b])[0, 1])
        for i, a in enumerate(labels)
        for b in labels[i + 1 :]
    ]
    return {
        "rows": rows,
        "family": sorted(family),
        "family_size": len(family),
        "slope_range": [min(slopes), max(slopes)] if slopes else None,
        "median_se": float(np.median([float(r["se"]) for r in headline_rows]))
        if headline_rows
        else None,
        "median_residual_correlation": float(np.median(pairwise)) if pairwise else None,
        "note": "Descriptive. Not in the confirmatory family, no gate, Holm across the twelve "
        "headline tasks. The canary is outside the family and outside the headline.",
    }


def canary_reading(cells: Sequence[Cell]) -> Dict[str, object]:
    """
    What the ten-way canary's accuracy says about the metric decision.

    NOT AN OUTCOME AND NOT A HYPOTHESIS. ``score_checkpoints`` puts a hundred trivial ten-way
    items in the suite for one purpose: if accuracy on them is at chance, the claim that
    multiple-choice accuracy is uninformative at 370M is a measurement rather than an assertion,
    and if it is not, the claim needs revisiting.

    TWO READINGS, BECAUSE THE SENSITIVE ONE ANSWERS THE WRONG QUESTION. Chance is 0.100, a
    hundred items give a per-cell binomial standard error of 0.030, and pooling twenty-five
    cells takes the standard error of the mean to 0.006 -- so a z test would call 0.112
    "distinguishable from chance" while a five-against-five contrast on it would still have
    nothing to divide by. The z is reported because it is the honest sensitive read. What
    ``at_chance`` records is the question the metric decision actually turns on: whether the
    mean is more than one *cell's own* sampling noise above chance, which is the smallest gap
    that could show up as spread between arms.

    :param cells: The cells as read.

    :returns: The reading, or a note that the canary reported no accuracy.
    """
    values = [c.canary_accuracy for c in cells if c.canary_accuracy is not None]
    if not values:
        return {"available": False, "note": "the canary reported no len_norm_v2."}
    array = np.asarray(values, dtype=float)
    per_cell_se = math.sqrt(CANARY_CHANCE * (1 - CANARY_CHANCE) / CANARY_ITEMS)
    se_of_mean = per_cell_se / math.sqrt(array.size)
    z = float((array.mean() - CANARY_CHANCE) / se_of_mean)
    return {
        "available": True,
        "n_cells": int(array.size),
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
        "chance": CANARY_CHANCE,
        "binomial_se_per_cell": per_cell_se,
        "z_against_chance": z,
        "margin": per_cell_se,
        "at_chance": bool(abs(float(array.mean()) - CANARY_CHANCE) <= per_cell_se),
        "note": "At chance, the write-up's claim that MC accuracy is uninformative at this "
        "scale is a measurement. Away from chance, the claim needs revisiting. Either way this "
        "is a diagnostic about the metric decision and never an outcome.",
    }


# ---------------------------------------------------------------------------------------
# The arm contrasts, priced in advance.
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DownstreamContrast:
    """One difference of downstream arm means, in bits per byte and in no other unit.

    THERE IS NO NATS COLUMN HERE AND THE IN-LOOP REPORT HAS ONE, WHICH IS NOT AN OVERSIGHT.
    ``analysis.py`` multiplies held-out BPB by ``NATS_PER_BPB = 4.57 * ln 2`` because 4.57 is the
    corpus's bytes per token and it is one constant across the seven held-out sources. The
    downstream suite is thirteen other texts; their bytes per token is not 4.57 and is not one
    constant across them. Carrying the in-loop conversion onto this endpoint would produce a
    column that is in no unit at all, would be internally consistent, and would be quoted.
    """

    name: str
    treatment: str
    comparator: str
    analysis: str
    """``paired``, ``unpaired``, ``paired-standalone`` or ``welch``."""

    primary: bool
    delta_bpb: float
    se_bpb: float
    df: float
    t_statistic: float
    p_value: float
    ci_bpb: Tuple[float, float]

    gate_bpb: float
    clears_gate: bool
    five_percent_bpb: float
    clears_five_percent: bool
    mde_bpb: float
    clears_mde: bool

    predicted_sign: int
    direction_as_predicted: bool
    sigma_bpb: float
    """The sigma this row's interval is actually built from, which is not always the headline one."""

    sigma_df: float


def _downstream_contrast(
    name: str,
    treatment: str,
    comparator: str,
    analysis: str,
    delta: float,
    se: float,
    df: float,
    predicted_sign: int,
    primary: bool,
    n_seeds: float,
) -> DownstreamContrast:
    """
    Assemble one contrast from a difference, its standard error and its df.

    :param n_seeds: Seeds per arm, used only to invert the standard error back into the residual
        sigma the interval rests on, so a reader can check that the interval and the sigma
        printed at the top of the report are the same statement.

    :returns: The contrast.
    """
    t_statistic = delta / se if se > 0 else float("nan")
    p_value = float(2.0 * stats.t.sf(abs(t_statistic), df)) if se > 0 else float("nan")
    half = float(stats.t.ppf(0.975, df)) * se
    detectable = mde_from_se(se, df)
    return DownstreamContrast(
        name=name,
        treatment=treatment,
        comparator=comparator,
        analysis=analysis,
        primary=primary,
        delta_bpb=float(delta),
        se_bpb=float(se),
        df=float(df),
        t_statistic=float(t_statistic),
        p_value=p_value,
        ci_bpb=(delta - half, delta + half),
        gate_bpb=GATE * se,
        clears_gate=bool(abs(delta) >= GATE * se),
        five_percent_bpb=half,
        clears_five_percent=bool(abs(delta) >= half),
        mde_bpb=float(detectable),
        clears_mde=bool(abs(delta) >= detectable),
        predicted_sign=predicted_sign,
        direction_as_predicted=bool(np.sign(delta) == predicted_sign),
        sigma_bpb=float(se * math.sqrt(n_seeds / 2.0)),
        sigma_df=float(df),
    )


def sigma_ceiling(effect: float, n_seeds: int = SEEDS_PER_ARM, power: float = 0.80) -> float:
    """
    The largest downstream noise floor at which a contrast of this size is still detectable.

    PRICED AS A STANDALONE TWO-ARM COMPARISON AT ``df = 2(n - 1)`` AND NOT AT THE FIVE-ARM
    POOLED ``df = 20``, AND THE CHOICE IS THE CONSERVATIVE ONE ON PURPOSE. The five-arm pooling
    tolerates about 9% more sigma, and it rests on the five arms sharing one variance -- which
    Bartlett **rejected** on the in-loop endpoint once the fifth arm landed, whose pre-committed
    consequence is Welch everywhere, whose Welch-Satterthwaite df for a five-against-five pair
    sits nearer 8 than 20. So df = 8 is the number the design is actually likely to have. The
    pooled figure is printed beside it in the report; this is the one the declarations rest on.

    :param effect: The effect to be detected, in bits per byte.
    :param n_seeds: Seeds per arm.
    :param power: Target power.

    :returns: The sigma at which the minimum detectable effect equals ``effect``.
    """
    if not effect > 0:
        raise Refusal("a ceiling on sigma needs a positive effect to be detected.")

    def gap(sigma: float) -> float:
        return mde(sigma, n_seeds, 2, 0.0, False, 0.05, power) - effect

    return float(optimize.brentq(gap, 1e-9, 10.0, xtol=1e-14))


def declares_underpowered(in_loop_effect: float) -> bool:
    """
    Whether a contrast of this in-loop size is declared under-powered downstream.

    THE FUNCTION THE TABLE IS CHECKED AGAINST, SO THAT THE DECLARATION IS DERIVED RATHER THAN
    TYPED. See :data:`UNDERPOWERED_BELOW_BPB` for where the line is and why it lands in an empty
    gap. A hypothesis whose flag disagrees with this is a test failure, not a judgement call.

    :param in_loop_effect: The frozen in-loop contrast, in bits per byte, signed or not.

    :returns: True when the downstream floor it would need is below the threshold.
    """
    effect = abs(float(in_loop_effect))
    if not effect > 0:
        return True
    return bool(sigma_ceiling(effect) < UNDERPOWERED_BELOW_BPB)


def seeds_needed(effect: float, sigma: float, power: float = 0.80, cap: int = 500) -> Optional[int]:
    """
    Seeds per arm to reach ``power`` against ``effect`` at a measured ``sigma``.

    POWERED AGAINST THE PRE-REGISTERED EFFECT AND NEVER AGAINST THE OBSERVED ONE. A sample size
    computed from the estimate the same data produced is post-hoc power, it is a monotone
    restatement of the p-value, and it tells a reader nothing they did not already have. The
    effect this is called with is the **in-loop** contrast, which was frozen before any
    downstream document existed.

    THAT MAKES EVERY NUMBER HERE CONDITIONAL ON UNIT TRANSFER, WHICH IS THE THING THE PRIMARY
    ANALYSIS TESTS. If the fitted slope comes back materially below one, the true downstream
    effects are smaller than the in-loop ones by that factor and every count here is too small
    by the square of it. The report states this beside the table rather than quietly rescaling,
    because rescaling by a fitted slope would make the sample-size figure post-hoc after all.

    :param effect: The effect to detect, in bits per byte.
    :param sigma: The measured downstream noise floor, already c4-corrected by the caller.
    :param power: Target power.
    :param cap: Stop looking here.

    :returns: The seed count, or None if it is above ``cap``.
    """
    if not effect > 0 or not sigma > 0:
        return None
    for candidate in range(3, cap + 1):
        if mde(sigma, candidate, 2, 0.0, False, 0.05, power) <= effect:
            return candidate
    return None


def published_degradation_verdict(contrasts: Mapping[str, DownstreamContrast]) -> Dict[str, object]:
    """
    The half of H2b that is a conjunction of two treatment-versus-baseline signs.

    "ARM 3 BUT NOT ARM 2 REPRODUCES THE PUBLISHED DEGRADATION" IS TWO POWERED QUESTIONS AND NOT
    AN ARM ORDERING. It needs ``D2b-ii`` (arm 3 against arm 1) to come back positive -- a
    worsening -- and ``D1`` (arm 2 against arm 1) not to. Both are treatment-versus-baseline at
    five against five against an in-loop-implied effect above 0.011 BPB, which is the part of
    this tranche the design resolves.

    ONLY THE SIGN OF THE PUBLISHED EFFECT TRANSFERS, NOT ITS SIZE. Tencent's number is two
    points of CLIMB accuracy -- 0.4626 against baseline, at roughly ten sigma outside a seed
    band bootstrapped from three baseline seeds -- and this endpoint is gold-continuation bits
    per byte over a different suite. There is no conversion between the two and none is
    attempted; the interval below is never read against 0.020 of anything.

    :param contrasts: The primary row of each contrast, by name.

    :returns: The verdict, with both components named.
    """
    arm3 = contrasts.get("D2b-ii")
    arm2 = contrasts.get("D1")
    if arm3 is None or arm2 is None:
        return {"available": False, "note": "needs both D2b-ii and D1."}

    arm3_degrades = bool(arm3.delta_bpb > 0 and arm3.clears_gate)
    arm2_degrades = bool(arm2.delta_bpb > 0 and arm2.clears_gate)
    if arm3_degrades and not arm2_degrades:
        verdict = "reproduced"
        prose = (
            "arm 3 is worse than the baseline downstream and clears the gate, and arm 2 is not. "
            "That is the pre-registered explanation of the published inversion, on the "
            "instrument the inversion was measured on."
        )
    elif not arm3_degrades and not arm2_degrades:
        verdict = "not reproduced"
        prose = (
            "arm 3 does not degrade against the baseline downstream, so the variant in the class "
            "of the published reimplementation does not reproduce the published worsening at "
            "370M. This module's own pre-registered explanation of the inversion is refuted on "
            "the downstream instrument, as it already was in loop."
        )
    elif arm3_degrades and arm2_degrades:
        verdict = "both degrade"
        prose = (
            "both arms are worse than the baseline downstream. That is a degradation of "
            "hyper-connections at this scale rather than an artifact of the reimplementation, "
            "and it is not what either half of H2b predicted."
        )
    else:
        verdict = "inverted"
        prose = (
            "arm 2 degrades and arm 3 does not, which is the opposite ordering to the one H2b "
            "predicts."
        )
    return {
        "available": True,
        "verdict": verdict,
        "prose": prose,
        "arm3_vs_baseline_bpb": arm3.delta_bpb,
        "arm3_clears_gate": arm3.clears_gate,
        "arm2_vs_baseline_bpb": arm2.delta_bpb,
        "arm2_clears_gate": arm2.clears_gate,
        "comparability": "Only the SIGN of Tencent's result transfers. Theirs is CLIMB accuracy "
        "points at 1.2B; this is gold-continuation bits per byte at 370M. No magnitude "
        "comparison is made and none is possible from these documents.",
    }


# ---------------------------------------------------------------------------------------
# The whole analysis.
# ---------------------------------------------------------------------------------------


def analyse(
    cells: Sequence[Cell],
    alpha: float = 0.05,
    provisional: Sequence[str] = (),
    label: str = "measured",
    in_loop_provenance: str = "",
) -> Dict[str, object]:
    """
    Everything the downstream pre-registration asks for, over the cells as read.

    :param cells: Twenty-five cells with in-loop endpoints attached.
    :param alpha: Two-sided level for Bartlett, the one-line check and the reported 5% line.
    :param provisional: Reasons this reading is not final, stamped on the artifact.
    :param label: ``measured`` or ``synthetic``.
    :param in_loop_provenance: Where the x-axis came from.

    :returns: A JSON-serializable dict of the whole analysis.

    :raises Refusal: If the arms do not share one set of seeds.
    """
    by_arm: Dict[str, List[Cell]] = {}
    for cell in cells:
        by_arm.setdefault(cell.arm, []).append(cell)
    present = [name for name in ARM_ORDER if name in by_arm]
    for name in by_arm:
        if name not in present:
            present.append(name)
    for name in present:
        by_arm[name].sort(key=lambda c: c.seed)

    # WHETHER THE BLOCK EXISTS AT ALL, WHICH IS NOT THE SAME QUESTION AS WHETHER THE TRANCHE IS
    # COMPLETE. Blocking on the seed pairs arm a seed k with arm b seed k, so a hole anywhere in
    # the grid means there is no block to form. That is a reason to drop to the unpaired and
    # Welch rows and say so, and it is NOT a reason to invent a pairing over whichever seeds
    # happen to be shared -- that would silently change which cells each contrast is built from,
    # per contrast, and every interval would still print. `main` refuses an incomplete set before
    # this is reached; the path exists for `--allow-provisional`, which stamps everything it
    # writes.
    seed_sets = {tuple(c.seed for c in by_arm[name]) for name in present}
    balanced = len(seed_sets) == 1
    seeds = list(sorted(seed_sets)[0]) if balanced else []
    n_seeds, n_arms = (len(seeds) if balanced else 0), len(present)
    unbalanced_reason = (
        ""
        if balanced
        else "the arms do not share one set of seeds ("
        + "; ".join(f"{name}: {[c.seed for c in by_arm[name]]}" for name in present)
        + "), so there is no block to form. The paired rows are absent and the unpaired and "
        "Welch rows are what is left."
    )

    downstream = {name: np.asarray([c.downstream_bpb for c in by_arm[name]]) for name in present}
    in_loop = {name: np.asarray([c.in_loop_bpb for c in by_arm[name]]) for name in present}

    ordered = [c for name in present for c in by_arm[name]]
    xs = [c.in_loop_bpb for c in ordered]
    ys = [c.downstream_bpb for c in ordered]
    arms_of = [c.arm for c in ordered]

    result: Dict[str, object] = {
        "label": label,
        "generated": date.today().isoformat(),
        "pre_registered_on": PRE_REGISTERED_ON,
        "schema_read": INPUT_SCHEMA,
        "suite_version": SUITE_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "step": FINAL_STEP,
        "provisional": list(provisional),
        "in_loop_provenance": in_loop_provenance,
        "warnings_from_documents": sorted(
            {f"{c.arm}/{c.seed}: {w}" for c in cells for w in c.warnings}
        ),
        "arms": [
            {
                "arm": name,
                "seeds": [c.seed for c in by_arm[name]],
                "downstream_bpb": [float(v) for v in downstream[name]],
                "in_loop_bpb": [float(v) for v in in_loop[name]],
                "mean_downstream_bpb": float(downstream[name].mean()),
                "sd_downstream_bpb": float(downstream[name].std(ddof=1)),
                "mean_in_loop_bpb": float(in_loop[name].mean()),
                "run_ids": [c.run_id for c in by_arm[name]],
                "checkpoints": [c.checkpoint for c in by_arm[name]],
            }
            for name in present
        ],
        "post_hoc": [{"date": when, "what": what} for when, what in POST_HOC],
        "post_hoc_note": (
            "Nothing has been added since the documents landed."
            if not POST_HOC
            else "Each entry was added after the pre-registration and is excluded from every family."
        ),
    }

    # (a) the downstream noise floor, measured rather than assumed, from the baseline alone.
    #
    # THE BASELINE'S OWN FIVE CELLS AT df = 4 ARE THE INSTRUMENT'S FLOOR AND THE POOLED FIGURE IS
    # NOT. Pooling over five arms mixes in whatever within-arm spread a treatment introduces, and
    # two of these arms are the ones this module predicts may be unstable. The baseline is the
    # only arm whose scatter is the instrument and nothing else, which is exactly why the in-loop
    # report quotes 0.00061 on the baseline beside 0.00147 pooled. Both are printed with their df.
    baseline_floor = pooled_sigma([downstream["baseline"]]) if "baseline" in downstream else None
    pooled = pooled_sigma([downstream[name] for name in present])
    bartlett_result = (
        bartlett([downstream[name] for name in present], present, alpha) if n_arms >= 2 else None
    )
    result["sigma"] = {
        "endpoint": f"downstream headline, unweighted mean of the {len(HEADLINE_GROUPS)} group "
        f"means of {PRIMARY_METRIC}",
        "baseline_only": (
            {
                "sigma_bpb": baseline_floor.sigma,
                "sigma_bpb_unbiased": baseline_floor.sigma_unbiased,
                "df": baseline_floor.df,
                "ci_bpb": [baseline_floor.ci_low, baseline_floor.ci_high],
                "span": baseline_floor.span,
                "c4": c4(baseline_floor.df),
                "note": "The downstream noise floor. Measured from the five baseline cells at "
                "df = 4 rather than assumed, which is the free half of the two improvements the "
                "previous analysis identified.",
            }
            if baseline_floor is not None
            else None
        ),
        "pooled": {
            "sigma_bpb": pooled.sigma,
            "sigma_bpb_unbiased": pooled.sigma_unbiased,
            "df": pooled.df,
            "ci_bpb": [pooled.ci_low, pooled.ci_high],
            "span": pooled.span,
            "c4": c4(pooled.df),
        },
        "per_arm_sd_bpb": {name: float(downstream[name].std(ddof=1)) for name in present},
        "bartlett": (
            {**asdict(bartlett_result), "spread": bartlett_result.spread}
            if bartlett_result is not None
            else None
        ),
        "in_loop_reference": {
            "baseline_sigma_bpb": IN_LOOP_BASELINE_SIGMA_BPB,
            "pooled_sigma_bpb": IN_LOOP_POOLED_SIGMA_BPB,
            "underpowered_below_bpb": UNDERPOWERED_BELOW_BPB,
            "note": "The in-loop floor, for the ratio. A downstream instrument that averages "
            "about 13,000 short out-of-distribution continuations against millions of "
            "in-distribution tokens is not expected to be quieter than this and the "
            "declarations below assume it is not.",
        },
    }

    # (b) THE PRIMARY. The line, its validity check, and the two things the pooled slope blends.
    pooled_fit = regress(xs, ys, "pooled, 25 cells")
    line_check = one_line_check(xs, ys, arms_of, alpha)
    between = arm_mean_slope(xs, ys, arms_of)
    within = within_arm_slope(xs, ys, arms_of)
    reported = "arm means" if line_check.rejects else "pooled, 25 cells"
    result["regression"] = {
        "model": "downstream_headline ~ 1 + (in_loop_endpoint - mean)",
        "null": SLOPE_NULL,
        "landmarks": list(SLOPE_LANDMARKS),
        "pooled": asdict(pooled_fit),
        "arm_means": asdict(between),
        "within_arm": asdict(within),
        "one_line_check": asdict(line_check),
        "reported_fit": reported,
        "withheld": bool(line_check.rejects),
        "withholding_rule": (
            "PRE-REGISTERED: the arm dummies clear alpha, so the five arms do not sit on one "
            "line, the pooled interval is anti-conservative, and the arm-mean fit at df = 3 is "
            "the reported reading. The pooled row is printed and is labelled as the blend it is."
            if line_check.rejects
            else "The arm dummies do not clear alpha, so one line describes the tranche at both "
            "levels and the pooled fit stands as primary. A non-rejection creates no claim; it "
            "leaves the pre-registered primary standing."
        ),
        "power_table": slope_power_table(
            pooled_fit.s_xx, (0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.016), pooled_fit.df
        ),
        "leverage": {
            "s_xx_total": pooled_fit.s_xx,
            "s_xx_between": float(between.s_xx * n_seeds),
            "s_xx_within": within.s_xx,
            "between_share": float(between.s_xx * n_seeds / pooled_fit.s_xx),
        },
    }
    result["per_task"] = per_task_slopes(ordered)
    result["canary"] = canary_reading(ordered)

    # (c) the arm contrasts, with the same precedence the in-loop module uses: paired if rho-hat
    # clears the pre-registered break-even, unpaired if it does not, and Welch overriding both if
    # Bartlett rejects. Pairing is on the seed, which is `init_seed` up to one constant shared by
    # every arm -- the other free improvement, and it costs nothing because the block is already
    # there. What it removes is the training data order and the initialization draw the arms
    # share, which is measured here rather than assumed.
    fit = (
        block_fit(np.asarray([downstream[name] for name in present], dtype=float), present, seeds)
        if balanced
        else None
    )
    every_rho = (
        {
            f"{a}-{b}": float(paired_correlation(downstream[a], downstream[b]).rho_pearson)
            for i, a in enumerate(present)
            for b in present[i + 1 :]
        }
        if balanced
        else {}
    )
    rho_values = list(every_rho.values())
    result["pairing"] = {
        "blocked_on": "seed, which is init_seed less one constant that every arm shares",
        "available": balanced,
        "unavailable_because": unbalanced_reason,
        "block_fit": asdict(fit) if fit is not None else None,
        "intraclass_rho": fit.intraclass_rho if fit is not None else None,
        "pre_registered_break_even": PRE_REGISTERED_BREAK_EVEN_RHO,
        "recomputed_break_even": break_even_rho(n_arms, n_seeds) if balanced else None,
        "recomputed_break_even_is_post_hoc": True,
        "rho_every_pair": every_rho,
        "compound_symmetry_spread": (max(rho_values) - min(rho_values)) if rho_values else 0.0,
        "compound_symmetry_doubtful": bool(
            rho_values and (max(rho_values) - min(rho_values)) > 0.3
        ),
    }

    floor_for_power = (
        baseline_floor.sigma_unbiased if baseline_floor is not None else pooled.sigma_unbiased
    )
    contrasts: List[Dict[str, object]] = []
    primary_rows: Dict[str, DownstreamContrast] = {}
    for hypothesis in HYPOTHESES:
        if hypothesis.treatment not in by_arm or hypothesis.comparator not in by_arm:
            contrasts.append(
                {
                    "name": hypothesis.name,
                    "claim": hypothesis.claim,
                    "status": "not analysable: "
                    + ", ".join(
                        arm
                        for arm in (hypothesis.treatment, hypothesis.comparator)
                        if arm not in by_arm
                    )
                    + " has no documents",
                }
            )
            continue

        a = downstream[hypothesis.treatment]
        b = downstream[hypothesis.comparator]
        rho = float(paired_correlation(a, b).rho_pearson) if balanced else float("nan")
        paired_primary = balanced and rho >= PRE_REGISTERED_BREAK_EVEN_RHO
        delta = float(a.mean() - b.mean())
        welch_delta, welch_se, welch_df = welch(a, b)

        if balanced and fit is not None:
            recipes = [
                (
                    "paired",
                    delta,
                    math.sqrt(2.0 * fit.ms_error_paired / n_seeds),
                    float(fit.df_paired),
                ),
                (
                    "unpaired",
                    delta,
                    math.sqrt(2.0 * fit.ms_error_unpaired / n_seeds),
                    float(fit.df_unpaired),
                ),
                (
                    "paired-standalone",
                    delta,
                    float((a - b).std(ddof=1)) / math.sqrt(n_seeds),
                    float(n_seeds - 1),
                ),
            ]
        else:
            unpaired_sigma = pooled_sigma([downstream[name] for name in present])
            recipes = [
                (
                    "unpaired",
                    delta,
                    unpaired_sigma.sigma * math.sqrt(1.0 / a.size + 1.0 / b.size),
                    float(unpaired_sigma.df),
                )
            ]

        rows: List[DownstreamContrast] = []
        for analysis_name, value, se, df in [
            *recipes,
            # Fractional by construction and left that way, for the reason analysis.py gives:
            # rounding a Welch-Satterthwaite df up narrows the interval it built.
            ("welch", welch_delta, welch_se, max(welch_df, 1.0)),
        ]:
            rows.append(
                _downstream_contrast(
                    name=hypothesis.name,
                    treatment=hypothesis.treatment,
                    comparator=hypothesis.comparator,
                    analysis=analysis_name,
                    delta=value,
                    se=se,
                    df=df,
                    predicted_sign=hypothesis.predicted_sign,
                    primary=analysis_name == ("paired" if paired_primary else "unpaired"),
                    # The seed count the standard error is inverted back through, as the
                    # harmonic mean of the two arm sizes so that the reported sigma is right on
                    # an unbalanced read too. It is `n_seeds` exactly when the grid is full.
                    n_seeds=2.0 / (1.0 / a.size + 1.0 / b.size),
                )
            )
        if bartlett_result is not None and bartlett_result.rejects:
            rows = [
                DownstreamContrast(**{**asdict(row), "primary": row.analysis == "welch"})
                for row in rows
            ]

        primary_row = next(row for row in rows if row.primary)
        primary_rows[hypothesis.name] = primary_row
        in_loop_effect = abs(
            float(in_loop[hypothesis.treatment].mean() - in_loop[hypothesis.comparator].mean())
        )
        contrasts.append(
            {
                "name": hypothesis.name,
                "claim": hypothesis.claim,
                "treatment": hypothesis.treatment,
                "comparator": hypothesis.comparator,
                "post_hoc": hypothesis.post_hoc,
                "predicted_sign": hypothesis.predicted_sign,
                "declared_underpowered": hypothesis.declared_underpowered,
                "rho_pearson": rho,
                "paired_is_primary": paired_primary
                and not (bartlett_result is not None and bartlett_result.rejects),
                "bartlett_forced_welch": bool(
                    bartlett_result is not None and bartlett_result.rejects
                ),
                "power": {
                    "in_loop_effect_bpb": in_loop_effect,
                    "sigma_ceiling_bpb": sigma_ceiling(in_loop_effect)
                    if in_loop_effect > 0
                    else None,
                    "declaration_agrees_with_arithmetic": bool(
                        declares_underpowered(in_loop_effect) == hypothesis.declared_underpowered
                    ),
                    "measured_floor_bpb": floor_for_power,
                    "seeds_needed": seeds_needed(in_loop_effect, floor_for_power),
                    "realised_power_against_in_loop_effect": float(
                        power_of(in_loop_effect, primary_row.se_bpb, int(max(primary_row.df, 1)))
                    ),
                    "note": "Powered against the in-loop effect, which was frozen before any "
                    "downstream document existed, and never against the observed downstream "
                    "estimate. Conditional on unit transfer: if the fitted slope is below one "
                    "the true downstream effects are smaller and every count here is too small. "
                    "The ceiling is a raw sigma and the measured floor is the c4-corrected point "
                    "estimate, which is 6% larger at df = 4, so the comparison between them is "
                    "conservative by that much and in the safe direction.",
                },
                "rows": [asdict(row) for row in rows],
            }
        )
    result["contrasts"] = contrasts

    # Holm over the six arm contrasts and nothing else. THE REGRESSION IS A FAMILY OF ONE AND IS
    # NOT IN THIS ONE. It is a single pre-specified primary test of a different quantity -- a
    # dimensionless slope rather than a difference of means -- and folding it in would inflate
    # six p-values by a test that answers another question, and inflate its own by six that
    # answer another question. Both families are declared, both are printed, and the per-task
    # profile has its own Holm inside `per_task_slopes` and is descriptive.
    holm = dose_adjustment.holm_adjust(
        {
            str(entry["name"]): float(row["p_value"])
            for entry in contrasts
            if "rows" in entry and not entry.get("post_hoc")
            for row in entry["rows"]  # type: ignore[union-attr]
            if row["primary"]
        }
    )
    for entry in contrasts:
        if str(entry["name"]) in holm:
            entry["holm_adjusted_p"] = holm[str(entry["name"])]
    result["holm"] = {
        "family": sorted(holm),
        "adjusted": holm,
        "note": "Holm-Bonferroni over the primary row of the six pre-registered arm contrasts. "
        "The gate stays uncorrected, as pre-registered. The regression is a declared family of "
        "one and is not in this family; the per-task profile has its own Holm and is descriptive.",
    }
    result["families"] = {
        "primary": ["the slope of downstream on in-loop, against 1"],
        "arm_contrasts": sorted(holm),
        "descriptive": [
            "the twelve headline per-task slopes (Holm within themselves)",
            "the canary",
        ],
    }
    result["published_degradation"] = published_degradation_verdict(primary_rows)
    return result


# ---------------------------------------------------------------------------------------
# The synthetic path. A separate verb, and it cannot be reached from the measured one.
# ---------------------------------------------------------------------------------------


def synthetic_documents(
    slope: float = 1.0,
    residual_sd: float = 0.004,
    intercept: float = 1.05,
    in_loop: Optional[Mapping[Tuple[str, int], float]] = None,
    seed: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[Tuple[str, int], float]]:
    """
    Twenty-five documents of the real schema with a **planted** slope, for the demo and the tests.

    THIS IS THE ONLY THING IN THIS FILE THAT CAN PRODUCE A NUMBER WITHOUT DATA, AND IT IS BEHIND
    A SEPARATE VERB THAT THE MEASURED PATH CANNOT REACH. ``noise_floor.py --dry-run`` once
    printed a complete synthetic report that was read as a measurement for twelve hours, and it
    was labelled -- so labelling is not the mitigation. The mitigation is that ``--demo`` is a
    different code path, writes under a different prefix, and every line it prints is prefixed.

    :param slope: The truth to plant. One is the coupled world; below one is decoupling.
    :param residual_sd: Downstream scatter around the line, in bits per byte.
    :param intercept: Downstream headline at the tranche's mean in-loop endpoint.
    :param in_loop: ``{(arm, seed): in-loop BPB}``, or None for a plausible synthetic tranche.
    :param seed: RNG seed.

    :returns: ``(documents, in-loop endpoints)``.
    """
    rng = np.random.default_rng(seed)
    if in_loop is None:
        # Shaped like the real tranche: five arm means about 0.0146 BPB apart end to end, with a
        # within-arm scatter of the in-loop floor's order. Nothing here is a measurement.
        centres = {
            "baseline": 0.6759,
            "faithful": 0.6613,
            "output-only": 0.6641,
            "no-output-init": 0.6633,
            "mhc": 0.6640,
        }
        in_loop = {
            (arm, seed_index): centres[arm] + float(rng.normal(0.0, 0.0012))
            for arm, seed_index in EXPECTED_CELLS
        }

    x_values = np.asarray([in_loop[cell] for cell in EXPECTED_CELLS], dtype=float)
    x_mean = float(x_values.mean())

    documents: List[Dict[str, object]] = []
    for index, (arm, seed_index) in enumerate(EXPECTED_CELLS):
        x = float(in_loop[(arm, seed_index)])
        headline = intercept + slope * (x - x_mean) + float(rng.normal(0.0, residual_sd))
        # Per task, around the headline, so the group means average back to it by construction:
        # every task carries the same planted slope and its own offset and noise.
        tasks: Dict[str, object] = {}
        for offset, task in enumerate(SUITE):
            value = headline + 0.02 * (offset - len(SUITE) / 2) + float(rng.normal(0.0, 0.001))
            metrics = {PRIMARY_METRIC: value, "ce_loss_v2": value * 0.7}
            if task.group == "canary":
                metrics["len_norm_v2"] = (
                    float(rng.binomial(CANARY_ITEMS, CANARY_CHANCE)) / CANARY_ITEMS
                )
            tasks[task.label] = {
                "group": task.group,
                "instances": 100,
                "requests": 400,
                "seconds": 1.0,
                "metrics": metrics,
            }
        results = [
            score_checkpoints.TaskResult(
                label=task.label,
                group=task.group,
                metrics={PRIMARY_METRIC: float(tasks[task.label]["metrics"][PRIMARY_METRIC])},  # type: ignore[index]
            )
            for task in SUITE
        ]
        aggregate = score_checkpoints.aggregate(results, PRIMARY_METRIC)
        documents.append(
            {
                "schema": INPUT_SCHEMA,
                "run_id": "synthetic",
                "arm": arm,
                "arm_number": hyper_connection_arms.ARMS[arm].number,
                "seed": seed_index,
                "step": FINAL_STEP,
                "cell_provenance": "synthetic",
                "checkpoint": f"synthetic://cell-{index}",
                "suite": "h2b",
                "suite_version": SUITE_VERSION,
                "primary_metric": PRIMARY_METRIC,
                "truncated": False,
                "warnings": [],
                "tokenizer": "synthetic",
                "param_dtype": "bfloat16",
                "device": "synthetic",
                "torch": "synthetic",
                "parameters": 0,
                "load_seconds": 0.0,
                "score_seconds": 0.0,
                "tasks": tasks,
                "downstream": {PRIMARY_METRIC: aggregate},
            }
        )
    return documents, dict(in_loop)


# ---------------------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------------------


def _df_label(df: float) -> str:
    """
    Print a df without inventing precision and without hiding what it is.

    :param df: The degrees of freedom.

    :returns: ``"19"`` for a count of residual dimensions, ``"5.78"`` for a variance ratio.
    """
    return str(int(df)) if float(df).is_integer() else f"{df:.2f}"


def _banner(label: str, provisional: Sequence[str]) -> List[str]:
    """
    The stamp at the top of every artifact, and the reason it is unmissable.

    :param label: ``measured`` or ``synthetic``.
    :param provisional: Reasons the read is not final.

    :returns: The banner lines.
    """
    lines = []
    if label != "measured":
        lines += [
            "#" * 92,
            "#  SYNTHETIC. Every number below was generated from a planted truth. Nothing here",
            "#  is a measurement of anything, and no decision may be taken on it.",
            "#" * 92,
            "",
        ]
    if provisional:
        lines += ["=" * 92, "  PROVISIONAL:"] + [f"    - {r}" for r in provisional] + ["=" * 92, ""]
    return lines


def render(result: Mapping[str, object]) -> str:
    """
    The report, in the order a reader should meet it: what it is, the floor, the slope, the arms.

    :param result: The output of :func:`analyse`.

    :returns: The rendered text.
    """
    out: List[str] = []
    say = out.append
    label = str(result.get("label", "measured"))
    say("\n".join(_banner(label, [str(p) for p in result.get("provisional", [])])).rstrip("\n"))
    say("")
    say("=" * 92)
    say("  THE DOWNSTREAM ANALYSIS OF THE HYPER-CONNECTION TRANCHE")
    say(f"  pre-registered {result['pre_registered_on']}, before the scoring job was submitted")
    say(f"  generated {result['generated']}   schema {result['schema_read']}")
    say(
        f"  suite {result['suite_version']}   metric {result['primary_metric']}   step {result['step']}"
    )
    say(f"  in-loop axis: {result['in_loop_provenance']}")
    say("=" * 92)

    warnings = list(result.get("warnings_from_documents", []))  # type: ignore[arg-type]
    if warnings:
        say("")
        say("  WARNINGS CARRIED BY THE DOCUMENTS THEMSELVES")
        for warning in warnings:
            say(f"    {warning}")

    say("")
    say("(a) THE DOWNSTREAM NOISE FLOOR, MEASURED RATHER THAN ASSUMED")
    sigma = result["sigma"]  # type: ignore[index]
    base = sigma.get("baseline_only")  # type: ignore[union-attr]
    if base:
        say(
            f"    baseline only   {base['sigma_bpb']:.5f} BPB  df {base['df']}  "
            f"95% [{base['ci_bpb'][0]:.5f}, {base['ci_bpb'][1]:.5f}]  span {base['span']:.2f}x"
        )
        say(f"    c4-corrected    {base['sigma_bpb_unbiased']:.5f} BPB")
    pooled = sigma["pooled"]  # type: ignore[index]
    say(
        f"    pooled, 5 arms  {pooled['sigma_bpb']:.5f} BPB  df {pooled['df']}  "
        f"95% [{pooled['ci_bpb'][0]:.5f}, {pooled['ci_bpb'][1]:.5f}]"
    )
    reference = sigma["in_loop_reference"]  # type: ignore[index]
    if base:
        say(
            f"    against in loop: {base['sigma_bpb'] / reference['baseline_sigma_bpb']:.1f}x the "
            f"in-loop baseline floor of {reference['baseline_sigma_bpb']:.5f} BPB"
        )
    per_arm = sigma["per_arm_sd_bpb"]  # type: ignore[index]
    say("    per arm         " + "  ".join(f"{k} {v:.5f}" for k, v in per_arm.items()))
    bart = sigma.get("bartlett")  # type: ignore[union-attr]
    if bart:
        say(
            f"    Bartlett        chi2 {bart['statistic']:.3f} df {bart['df']} "
            f"p {bart['p_value']:.4f} spread {bart['spread']:.2f}x -> "
            + (
                "REJECTS: Welch everywhere, as pre-committed"
                if bart["rejects"]
                else "does not reject"
            )
        )

    say("")
    say("(b) THE PRIMARY: THE SLOPE OF DOWNSTREAM ON IN-LOOP BITS PER BYTE")
    regression = result["regression"]  # type: ignore[index]
    say(f"    model  {regression['model']}   null  beta = {regression['null']}")
    say("")
    say("      fit                  slope        SE      df     95% interval        s_resid    R^2")
    for key, name in (
        ("pooled", "pooled, 25 cells"),
        ("arm_means", "arm means"),
        ("within_arm", "within arm"),
    ):
        f = regression[key]  # type: ignore[index]
        mark = "*" if name == regression["reported_fit"] else " "  # type: ignore[index]
        say(
            f"    {mark} {name:<18s} {f['slope']:>7.3f}  {f['se_slope']:>8.3f}  {_df_label(f['df']):>4s}  "
            f"[{f['ci'][0]:>7.3f}, {f['ci'][1]:>7.3f}]   {f['residual_sd']:.5f}  {f['r_squared']:>5.2f}"
        )
    say("    * is the reported fit.")
    say("")
    reported = regression["pooled" if regression["reported_fit"] == "pooled, 25 cells" else "arm_means"]  # type: ignore[index]
    say("      landmark      |beta-null|      gate(2SE)   5% line       p        MDE   verdict")
    for test in reported["tests"]:  # type: ignore[index]
        gap = abs(reported["slope"] - test["null"])  # type: ignore[index]
        verdict = "clears" if test["clears_gate"] else "below gate"
        say(
            f"      beta = {test['null']:<5.2f}   {gap:>9.3f}   {test['gate']:>9.3f}  "
            f"{test['five_percent']:>8.3f}  {test['p_value']:>7.4f}  {test['mde']:>8.3f}   {verdict}"
        )
    check = regression["one_line_check"]  # type: ignore[index]
    say("")
    say(
        f"    do the arms sit on one line?  F({check['df_numerator']}, {check['df_denominator']}) "
        f"= {check['f_statistic']:.3f}, p = {check['p_value']:.4f} -> "
        + ("REJECTS" if check["rejects"] else "does not reject")
    )
    say(f"    {regression['withholding_rule']}")
    leverage = regression["leverage"]  # type: ignore[index]
    say(
        f"    leverage: {leverage['between_share']:.1%} of the x spread is between arms, so the "
        "within-arm row is the weak one by construction."
    )
    say("")
    say("    what the slope could resolve, as a function of the downstream residual scatter:")
    say("        s_resid    SE(beta)   95% half-width    power vs 0    power vs 0.5")
    for row in regression["power_table"]:  # type: ignore[index]
        say(
            f"        {row['residual_sd']:.4f}    {row['se_slope']:>7.3f}   {row['half_width']:>13.3f}"
            f"    {row['power_against_zero']:>9.2f}    {row['power_against_half']:>11.2f}"
        )

    say("")
    say("(c) THE PER-TASK PROFILE. DESCRIPTIVE, HOLM-ADJUSTED, NOT IN THE CONFIRMATORY FAMILY")
    profile = result["per_task"]  # type: ignore[index]
    say(f"    {profile['note']}")
    if profile["slope_range"]:  # type: ignore[index]
        say(
            f"    the twelve headline slopes span {profile['slope_range'][0]:.3f} to "  # type: ignore[index]
            f"{profile['slope_range'][1]:.3f} against a median SE of {profile['median_se']:.3f}; "  # type: ignore[index]
            f"median residual correlation between tasks {profile['median_residual_correlation']:.2f}"
        )
    say("")
    say(
        "      task                                  group     slope       SE       raw p    Holm p"
    )
    for row in profile["rows"]:  # type: ignore[index]
        holm_p = row["holm_adjusted_p"]
        holm_text = f"{holm_p:.4f}" if holm_p is not None else "  (out of family)"
        say(
            f"      {row['task']:<36s} {row['group']:<8s} {row['slope']:>7.3f}  {row['se']:>7.3f}  "
            f"{row['p_against_one']:>9.4f}  {holm_text:>8s}"
        )
    canary = result["canary"]  # type: ignore[index]
    if canary.get("available"):  # type: ignore[union-attr]
        say("")
        say(
            f"    canary accuracy {canary['mean']:.3f} +/- {canary['sd']:.3f} over "  # type: ignore[index]
            f"{canary['n_cells']} cells against a chance of {canary['chance']:.3f} "
            f"(z = {canary['z_against_chance']:+.2f}) -> "
            + (
                "at chance, so the metric decision is a measurement"
                if canary["at_chance"]
                else "NOT at chance: the metric decision needs revisiting"
            )
        )

    say("")
    say("(d) THE ARM CONTRASTS, WITH THE POWER DECLARED BEFORE THE DATA")
    pairing = result["pairing"]  # type: ignore[index]
    if pairing["available"]:  # type: ignore[index]
        say(
            f"    blocked on {pairing['blocked_on']}; intraclass rho "
            f"{pairing['intraclass_rho']:+.3f} against a pre-registered break-even of "
            f"{pairing['pre_registered_break_even']}"
        )
    else:
        say(f"    NO PAIRING: {pairing['unavailable_because']}")
    if pairing["compound_symmetry_doubtful"]:  # type: ignore[index]
        say(
            f"    the pairwise correlations span {pairing['compound_symmetry_spread']:.2f}, so "
            "compound symmetry is doubtful and the unpaired row is the conservative interval."
        )
    for entry in result["contrasts"]:  # type: ignore[index]
        say("")
        if "rows" not in entry:
            say(f"    {entry['name']}: {entry['status']}")
            continue
        power = entry["power"]
        flag = "UNDER-POWERED, DECLARED IN ADVANCE" if entry["declared_underpowered"] else "powered"
        say(
            f"    {entry['name']}  {entry['treatment']} - {entry['comparator']}   [{flag}]"
            + (f"   post-hoc {entry['post_hoc']}" if entry["post_hoc"] else "")
        )
        needed = power["seeds_needed"]
        say(
            f"      in-loop effect {power['in_loop_effect_bpb']:.5f} BPB needs a downstream floor "
            f"of {power['sigma_ceiling_bpb']:.5f} or less; measured floor "
            f"{power['measured_floor_bpb']:.5f}"
        )
        say(
            f"      realised power against that effect {power['realised_power_against_in_loop_effect']:.2f}; "
            + (f"seeds per arm for 80%: {needed}" if needed else "80% is beyond 500 seeds per arm")
        )
        for row in entry["rows"]:
            mark = "*" if row["primary"] else " "
            say(
                f"      {mark} {row['analysis']:<18s} d {row['delta_bpb']:>+9.5f}  SE {row['se_bpb']:.5f}  "
                f"df {_df_label(row['df']):>5s}  95% [{row['ci_bpb'][0]:>+9.5f}, {row['ci_bpb'][1]:>+9.5f}]  "
                f"p {row['p_value']:.4f}  MDE {row['mde_bpb']:.5f}  "
                + ("clears gate" if row["clears_gate"] else "below gate")
            )
        if "holm_adjusted_p" in entry:
            say(f"      Holm over {len(result['holm']['adjusted'])}: {entry['holm_adjusted_p']:.4f}")  # type: ignore[index]
        if entry["declared_underpowered"]:
            say(
                "      A null on this row is UNINFORMATIVE. It was declared under-powered on "
                f"{result['pre_registered_on']}, before any downstream document existed, and it "
                "is not evidence of no effect."
            )

    say("")
    say("(e) THE HALF OF H2b THAT IS POWERED")
    verdict = result["published_degradation"]  # type: ignore[index]
    if verdict.get("available"):  # type: ignore[union-attr]
        say(f"    verdict: {verdict['verdict']}")
        say(f"    {verdict['prose']}")
        say(
            f"    arm 3 - arm 1 {verdict['arm3_vs_baseline_bpb']:+.5f} BPB "
            f"({'clears' if verdict['arm3_clears_gate'] else 'below'} gate); "
            f"arm 2 - arm 1 {verdict['arm2_vs_baseline_bpb']:+.5f} BPB "
            f"({'clears' if verdict['arm2_clears_gate'] else 'below'} gate)"
        )
        say(f"    {verdict['comparability']}")

    say("")
    say("(f) THE FAMILIES, AND WHAT IS NOT IN ONE")
    families = result["families"]  # type: ignore[index]
    for name, members in families.items():  # type: ignore[union-attr]
        say(f"    {name:<16s} {'; '.join(members)}")
    say(f"    {result['holm']['note']}")  # type: ignore[index]
    say("")
    say(f"    POST-HOC ADDITIONS: {result['post_hoc_note']}")
    for item in result["post_hoc"]:  # type: ignore[index]
        say(f"      {item['date']}  {item['what']}")
    say("")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
# The self-test. No network, no data, planted truths.
# ---------------------------------------------------------------------------------------


def self_test(replicates: int = 200) -> int:
    """
    Check the estimators against truths nobody could have preferred.

    :param replicates: How many synthetic tranches to draw for the recovery checks.

    :returns: A process exit status.
    """
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not condition:
            failures.append(name)

    print("downstream_analysis --self-test")
    for truth in (1.0, 0.35):
        recovered = []
        for index in range(replicates):
            documents, endpoints = synthetic_documents(
                slope=truth, residual_sd=0.003, seed=1_000 + index
            )
            cells = [cell_from_document(d, f"synthetic-{i}") for i, d in enumerate(documents)]
            attach_in_loop(cells, endpoints)
            recovered.append(
                regress(
                    [c.in_loop_bpb for c in cells], [c.downstream_bpb for c in cells], "t"
                ).slope
            )
        mean = float(np.mean(recovered))
        check(
            f"the slope estimator recovers a planted {truth}",
            abs(mean - truth) < 0.05,
            f"mean {mean:.4f} over {replicates}",
        )

    documents, endpoints = synthetic_documents(slope=1.0, residual_sd=0.003, seed=7)
    cells = [cell_from_document(d, f"synthetic-{i}") for i, d in enumerate(documents)]
    attach_in_loop(cells, endpoints)
    result = analyse(cells, label="synthetic")
    check("the whole pipeline runs on a planted tranche", "regression" in result)
    check(
        "twenty-five cells are read",
        sum(len(a["seeds"]) for a in result["arms"]) == len(EXPECTED_CELLS),  # type: ignore[index,union-attr]
    )
    check("the post-hoc list is empty at pre-registration", not POST_HOC)

    short = [c for c in cells if not (c.arm == "mhc" and c.seed == 4)]
    check("a short tranche is refused", bool(completeness_refusals(short)))

    return 1 if failures else 0


# ---------------------------------------------------------------------------------------
# The program.
# ---------------------------------------------------------------------------------------


def _write(path: str, text: str) -> None:
    """
    Write a file, making its directory.

    :param path: Where.
    :param text: What.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    print(f"wrote {path}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    """
    The command line.

    :returns: The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--documents", help="A local directory of downstream-*.json, or one file.")
    parser.add_argument("--in-loop", help="analysis/analysis.json, the frozen in-loop artifact.")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=SUBMISSION",
        help="Recompute the in-loop axis from W&B instead, through analysis.read_arm.",
    )
    parser.add_argument("--group", default="hyper-connections-370m")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "eduLLM"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "pre-training"))
    parser.add_argument("--out", help="Directory for the report and the JSON.")
    parser.add_argument(
        "--allow-provisional",
        action="store_true",
        help="Downgrade COMPLETENESS refusals to warnings and stamp every artifact PROVISIONAL. "
        "It does not downgrade a wrong schema, a duplicate cell, a truncated score, an "
        "incomplete headline or twenty-five numbers off two instruments.",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Estimators against planted truths."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Synthetic. A separate verb the measured path cannot reach.",
    )
    parser.add_argument("--demo-slope", type=float, default=0.6, help="The truth --demo plants.")
    return parser


def main() -> int:
    """
    Read, refuse or report.

    :returns: A process exit status.
    """
    opts = build_parser().parse_args()

    if opts.self_test:
        return self_test()

    if opts.demo:
        documents, endpoints = synthetic_documents(slope=opts.demo_slope, seed=11)
        cells = [cell_from_document(d, f"synthetic-{i}") for i, d in enumerate(documents)]
        attach_in_loop(cells, endpoints)
        result = analyse(
            cells,
            label="synthetic",
            in_loop_provenance=f"synthetic, planted slope {opts.demo_slope}",
        )
        text = render(result)
        print("\n".join("SYNTHETIC | " + line for line in text.splitlines()))
        if opts.out:
            _write(os.path.join(opts.out, "synthetic-downstream.txt"), text)
            _write(
                os.path.join(opts.out, "synthetic-downstream.json"),
                json.dumps(result, indent=2, sort_keys=True, default=float),
            )
        return 0

    if not opts.documents:
        print(
            "nothing to read. Pass --documents <directory of downstream-*.json>, or --demo for "
            "the synthetic path, or --self-test for the estimators. There is no default and "
            "there is no fallback: a tool that can answer without data will be asked to.",
            file=sys.stderr,
        )
        return 2

    if bool(opts.in_loop) == bool(opts.arm):
        print(
            "the in-loop axis needs exactly one source: --in-loop <analysis.json> for the frozen "
            "artifact the in-loop report was written against, or --arm name=submission for a "
            "fresh read through analysis.read_arm. Both, or neither, is ambiguous.",
            file=sys.stderr,
        )
        return 2

    try:
        cells = [
            cell_from_document(document, path) for path, document in read_documents(opts.documents)
        ]

        hard = instrument_refusals(cells)
        soft = completeness_refusals(cells)
        if hard:
            raise Refusal("\n".join(hard))
        if soft and not opts.allow_provisional:
            raise Refusal(
                "\n".join(soft)
                + "\n\nThis analysis will not run on a partial set. Every estimator below --"
                " the slope, its interval, the noise floor, every contrast -- would return a"
                " number, and none of them would be a number about the tranche that was"
                " submitted. --allow-provisional stamps the artifacts and proceeds."
            )

        if opts.in_loop:
            endpoints, provenance = in_loop_from_artifact(opts.in_loop)
        else:
            pairs = {}
            for item in opts.arm:
                if "=" not in item:
                    raise Refusal(f"--arm {item} is not name=submission.")
                name, submission = item.split("=", 1)
                pairs[name] = submission
            endpoints, provenance = in_loop_from_wandb(pairs, opts.entity, opts.project, opts.group)
        attach_in_loop(cells, endpoints)

        result = analyse(
            cells, provisional=soft if opts.allow_provisional else (), in_loop_provenance=provenance
        )
    except Refusal as refusal:
        print(f"\nREFUSED\n\n{refusal}\n", file=sys.stderr)
        return 1

    text = render(result)
    print(text)
    if opts.out:
        stem = "provisional-downstream" if result["provisional"] else "downstream"
        _write(os.path.join(opts.out, f"{stem}.txt"), text)
        _write(
            os.path.join(opts.out, f"{stem}.json"),
            json.dumps(result, indent=2, sort_keys=True, default=float),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
