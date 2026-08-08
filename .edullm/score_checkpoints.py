#!/usr/bin/env python3
"""Score a saved checkpoint of the hyper-connection tranche on a downstream suite.

THIS IS THE JOB H2b IS BLOCKED ON, AND IT IS THE ONLY PRE-REGISTERED HYPOTHESIS NOTHING
ELSE FUNDS. ``hyper-connections.md`` says so in as many words: H2a is arm 2 against arm 3 on
held-out cross-entropy, the published result H2 sets out to explain is a *downstream*
average, and "loss and downstream decouple by 6 to 16 points for changes in this class" is
the stated reason downstream is reported at all. An arm-2-versus-arm-3 gap in in-loop BPB is
evidence that two implementations differ; it is not evidence about the published negative
result. This program produces the other number.

WHAT IT IS, IN ONE LINE. Fifteen cells, one per ``(arm, seed)`` of the tranche; each reads
one training cell's final checkpoint out of S3, rebuilds that cell's model, scores it over a
fixed task suite with the task data already inside the image, and writes one JSON document.

NOTHING OF ``train_on_corpus.py`` OR ``train_hyper_connections.py`` IS COPIED HERE, for the
reason the latter gives about the former. What is reused, and why each one:

  ``Stage`` / ``Refusal`` / ``during`` / ``cli``   the exit-code channel. A container that
        dies before W&B exists writes its explanation to a log stream nobody on the platform
        side may read, so the stage is the exit code. A scoring job that invented its own
        numbering would make the one diagnostic that works stop working.
  ``leave_the_reason_in_wandb``                    the same refusal, where a researcher can
        actually open it.
  ``train_hyper_connections.resolve_cell``         WHICH CELL THIS IS. Scoring resolves its
        ``(arm, seed)`` through the identical function the training tranche resolved its own
        through, against the identical :data:`hyper_connection_arms.TRANCHE_CELLS`. Two
        implementations of "cell 7 is faithful seed 2" that agreed today would be two that
        could disagree later, and the disagreement would be a checkpoint scored under the
        wrong arm's name -- which produces a number, not an error.
  ``hyper_connection_arms.ARMS`` / ``hc_370M``     WHICH MODEL TO BUILD. See
        :func:`model_config_for`.

Locally, against a checkpoint you made yourself and with two small tasks::

    python .edullm/score_checkpoints.py smoke \\
        --arm-run baseline=/tmp/fake-run --arm baseline --seed 0 \\
        --suite smoke --device cpu --param-dtype float32 \\
        --tokenizer /path/to/dolma2/tokenizer.json --output-dir /tmp/scores

On the platform, as the whole fifteen-cell fan-out (see ``run.score-stage.yaml``)::

    edullm check --json --experiment hyper-connections-370m --dataset none \\
      --team input-core --spec .edullm/run.score-stage.yaml \\
      --hours 2 --attempts 1 --fanout-size 15 --fanout-index-parameter arm-and-seed
"""

import argparse
import copy
import enum
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Both this file and its siblings have to be importable by a stable name, for the reason
# `train_hyper_connections.py` gives: `_CLASS_` in a saved config records module paths and
# anything reading one back resolves it with `importlib.import_module`.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms  # noqa: E402

# Imported for `resolve_cell` alone. It rebinds four globals on `train_on_corpus` as an
# import side effect, which is harmless here because nothing below calls
# `train_on_corpus.main`, and it calls `hyper_connection_arms.install()`, which is
# idempotent and is wanted anyway.
import train_hyper_connections  # noqa: E402
import train_on_corpus  # noqa: E402
from hyper_connection_arms import ARMS  # noqa: E402

from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.nn.transformer import TransformerConfig  # noqa: E402

log = logging.getLogger(__name__)

Refusal = train_on_corpus.Refusal
Stage = train_on_corpus.Stage
during = train_on_corpus.during


class ScoringStage(enum.IntEnum):
    """The stages that only a scoring job has, continuing ``train_on_corpus.Stage``.

    A SEPARATE ENUM RATHER THAN THREE MORE MEMBERS OF THAT ONE, AND THE REASON IS
    OWNERSHIP RATHER THAN TASTE. ``train_on_corpus.py`` is the training path's file and is
    being read by fifteen submitted cells; a scoring job has no business editing it to add
    exit codes those cells will never raise. The numbers continue that enum's sequence and
    do not collide with it, which is the property that matters -- ``73`` is the highest it
    defines and these start at ``74`` -- and they stay inside the conventional ``sysexits``
    band, clear of the 126, 127 and 128+n the shell and the signal convention own.

    ``Refusal`` takes these unchanged: it stores whatever it is handed and ``cli`` reads
    ``int(stage)`` and ``stage.name``, both of which any ``IntEnum`` answers.
    """

    THE_CHECKPOINT_IS_NOT_WHERE_THIS_CELL_WAS_TOLD = 74
    THE_CHECKPOINT_DOES_NOT_DESCRIBE_THE_ARM_THIS_CELL_IS = 75
    THE_IMAGE_HAS_NO_TASK_DATA = 76


def refuse(stage, explanation: str) -> Refusal:
    """
    Build a :class:`train_on_corpus.Refusal` from either stage enum.

    :param stage: A :class:`train_on_corpus.Stage` or a :class:`ScoringStage`.
    :param explanation: What to tell the person, in whole sentences.

    :returns: The refusal, for the caller to raise.
    """
    return Refusal(stage, explanation)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# The suite, and the metric. Both are decisions rather than defaults, so both are argued
# for here and both are under test.
# ---------------------------------------------------------------------------------------

#: The primary metric, and it is NOT multiple-choice accuracy.
#:
#: WHAT IT IS. ``bpb_v2`` from ``olmo_eval.ICLMetric`` is the bits-per-byte of the *gold*
#: continuation: the summed log-probability the model puts on the correct answer string,
#: divided by that string's length in UTF-8 bytes and converted to bits. It is a continuous
#: likelihood, it is defined whether or not the model would have picked the right answer,
#: and it is in the same units as the in-loop held-out BPB the whole pre-registration is
#: written against -- so an arm's downstream number and its in-loop number can be read
#: beside each other rather than converted between.
#:
#: WHY NOT ACCURACY, AND THIS IS THE LOAD-BEARING PART. A 370M model at 4.72B tokens is at
#: or near chance on multiple-choice accuracy, and a metric pinned at chance has no variance
#: for a five-against-five contrast to divide by. Ai2's DataDecide (arXiv 2504.11393)
#: measures exactly this: at 150M, "continuous metrics using the character normalized
#: likelihood of correct or all answer options serve as better or equivalent predictors of
#: decisions than using the same Accuracy" (Sec. 3.3), and Sec. 3.4 gives the mechanism --
#: decision accuracy is set by noise, the standard deviation over seed runs, against spread,
#: and "using Correct Prob sees wider spreads or reduced noise for many tasks". Their code
#: benchmarks go from trivial to 80% decision accuracy on the metric change alone.
#:
#: WHICH CONTINUOUS METRIC, BECAUSE DATADECIDE IS SPECIFIC AND THE OBVIOUS ONE IS THE WRONG
#: ONE. The metrics that help are CORRECT PROB -- the length-normalized likelihood of the
#: gold continuation on its own -- and TOTAL PROB. The ones that do *not* are NORM CORRECT
#: PROB and MARGIN, the ones that "penalize probability assigned to incorrect answers",
#: which Sec. 3.3 finds trend with Accuracy rather than beating it. ``olmo_eval``'s
#: ``soft_v2`` is a softmax over the choice set evaluated at the gold answer, which is NORM
#: CORRECT PROB exactly; ``bpb_v2`` and ``ce_loss_v2`` are CORRECT PROB up to a monotone
#: transform and a bytes-versus-characters denominator. So the endorsed metric here is the
#: one this suite makes primary, and the tempting one is reported as a diagnostic and is not
#: read as the headline.
#:
#: v2 RATHER THAN v1. The two differ in whether the continuation's leading space counts
#: toward the normalizing length. v2 counts it, which is the OLMES standard; v1 is
#: preserved upstream for backwards compatibility with a pre-OLMES convention. Both are
#: recorded in the output, because the constant differs between them and somebody comparing
#: to a published figure needs to know which one they are holding.
PRIMARY_METRIC = "bpb_v2"

#: Reported beside the primary metric on every task, and never instead of it.
#:
#: ``len_norm_v2`` is rank-classification accuracy: argmax over the choices of the
#: length-normalized log-likelihood. It is the discrete readout, it is what OLMES reports,
#: and it is the number a reader will want to see even though it is the one expected to sit
#: near chance here. ``ce_loss_v2`` is the same quantity as the primary metric in nats per
#: character instead of bits per byte. ``soft_log_v2`` is the log of NORM CORRECT PROB --
#: kept because it is the metric DataDecide argues *against* at this scale, and a number
#: that is recorded is a number the write-up can show behaving the way the paper says it
#: behaves.
SECONDARY_METRICS = ("len_norm_v2", "ce_loss_v2", "soft_log_v2", "bpb_v1")


@dataclass(frozen=True)
class Task:
    """One task of a suite, and the reason it is in one."""

    label: str
    """Its ``olmo_eval`` label. ``build_task`` refuses anything not in ``list_tasks()``."""

    group: str
    """
    What it contributes to. Tasks are averaged within a group and the groups are averaged
    into the headline, so that MMLU's four category splits count once between them rather
    than four times against HellaSwag's one.
    """

    why: str
    """What this task is doing in the suite. A task that cannot answer this is not in one."""


#: The suite H2b is read off, and the tasks that are deliberately not in it.
#:
#: THE REFERENCE POINT IS OLMES, AND THE SELECTION IS AI2'S OWN SMALL-COMPUTE LIST RATHER
#: THAN A FRESH OPINION. ``olmo_core.eval.task_groups.FULL_TASKS_SMALL_COMPUTE`` is the list
#: upstream uses "for training runs where we don't expect the model to acquire MC", which is
#: this experiment exactly, and its OLMES section is the seven RC tasks below plus MMLU RC.
#: Two of the OLMES Core 9 are commented out in that list as "too noisy to be worth
#: tracking" and are left out here for the same reason, with the arithmetic beside each.
#:
#: EVERY RC TASK HERE IS SCORED WITH ``metric_type='len_norm'``, WHICH IS WHY ONE PASS GIVES
#: BOTH NUMBERS. ``ICLMetric.compute`` under that type returns ``len_norm_v*``, ``ce_loss_v*``,
#: ``bpb_v*``, ``soft_v*`` and ``soft_log_v*`` together. The ``*_bpb_*`` spelling of the same
#: task reads the same request file and skips the non-gold continuations -- about a quarter
#: of the forward passes -- but then reports the gold BPB alone. At single-digit minutes of
#: arithmetic for the whole suite, paying four times for the accuracy beside the likelihood
#: is not a trade worth making.
SUITE_H2B: Tuple[Task, ...] = (
    Task(
        "arc_challenge_test_rc_5shot",
        "olmes",
        "OLMES core. DataDecide Fig. 2 puts ARC among the most predictable tasks in the "
        "suite at small scale, which is the property a five-against-five contrast needs.",
    ),
    Task(
        "arc_easy_test_rc_5shot",
        "olmes",
        "OLMES core, and the widest spread across data recipes in DataDecide Fig. 5 -- "
        "predictable from five orders of magnitude less compute than the target scale.",
    ),
    Task(
        "hellaswag_rc_5shot",
        "olmes",
        "OLMES core, on the 1K subset upstream uses. Sentence completion over everyday "
        "situations, and DataDecide Fig. 5 finds it the low-run-to-run-variance task.",
    ),
    Task(
        "piqa_val_rc_5shot",
        "olmes",
        "OLMES core. Physical commonsense, and the one core task whose continuations are "
        "long enough that the byte normalization is doing real work.",
    ),
    Task(
        "csqa_val_rc_5shot",
        "olmes",
        "OLMES core. Five-way commonsense, so chance is 0.20 and an accuracy readout has "
        "further to climb than on the binary tasks.",
    ),
    Task(
        "socialiqa_val_rc_5shot",
        "olmes",
        "OLMES core, and kept despite DataDecide Sec. 3.1 calling it difficult to predict "
        "at all scales: the same figure shows it spreading recipes widely, which is the "
        "half of the noise-versus-spread trade this contrast can use.",
    ),
    Task(
        "winogrande_val_rc_5shot",
        "olmes",
        "OLMES core. Pronoun resolution, which is the core task most nearly about "
        "long-range binding within the context -- the thing a residual-topology change has "
        "the most direct route to moving.",
    ),
    Task(
        "mmlu_stem_val_rc_5shot",
        "mmlu",
        "MMLU RC, STEM. DataDecide Fig. 5 names MMLU the lowest run-to-run-noise task in "
        "the suite, which is what makes it worth four labels.",
    ),
    Task("mmlu_humanities_val_rc_5shot", "mmlu", "MMLU RC, humanities."),
    Task("mmlu_social_sciences_val_rc_5shot", "mmlu", "MMLU RC, social sciences."),
    Task("mmlu_other_val_rc_5shot", "mmlu", "MMLU RC, other."),
    Task(
        "lambada_bpb_0shot",
        "lambada",
        "NOT AN OLMES TASK AND HERE ON PURPOSE. 5,153 passages whose last word is "
        "recoverable only from the whole preceding paragraph, scored as gold-continuation "
        "BPB with no choice set at all -- so it is the one task in the suite that cannot "
        "be answered by the choice-set arithmetic and measures long-range context use "
        "directly. Hyper-connections are a claim about how information crosses depth; if "
        "the claim is true anywhere at this scale it should be visible here.",
    ),
    Task(
        "copycolors_10way_fast",
        "canary",
        "THE CANARY THAT FUNDS THE METRIC DECISION, AND IT IS NOT AVERAGED INTO THE "
        "HEADLINE. A hundred trivial ten-way items -- 'the color of the sky is ...' -- "
        "whose accuracy answers whether this model can do the multiple-choice format at "
        "all. Upstream carries it in every task list for this reason. If it is at chance, "
        "the write-up's claim that MC accuracy is uninformative at 370M is a measurement "
        "rather than an assertion; if it is not, the claim needs revisiting.",
    ),
)

#: Groups whose mean goes into the headline downstream average. ``canary`` is excluded, and
#: that exclusion is the whole reason groups exist rather than a flat list: copycolors is a
#: diagnostic about the metric and averaging a trivial task into a headline would flatter
#: every arm equally and dilute the contrast.
HEADLINE_GROUPS: Tuple[str, ...] = ("olmes", "mmlu", "lambada")

#: A two-task suite for the local smoke test and for the preflight. Small enough to build
#: and score on a laptop CPU in under a minute, and shaped like the real one: one
#: ``len_norm`` task that reports every metric and one ``bpb`` task that reports only the
#: continuous one, so the aggregation code is exercised on both kinds.
SUITE_SMOKE: Tuple[Task, ...] = (
    Task("copycolors_10way_fast", "canary", "100 items, and the cheapest thing that builds."),
    Task("lambada_bpb_0shot", "lambada", "A gold-continuation BPB task with no choice set."),
)

SUITES: Dict[str, Tuple[Task, ...]] = {"h2b": SUITE_H2B, "smoke": SUITE_SMOKE}

#: The name that goes into every output document, so that two scoring runs can be told
#: apart by something other than their date. Bump it when :data:`SUITE_H2B` changes, which
#: a test enforces by pinning the suite's contents against this string.
SUITE_VERSION = "h2b-rc-2026-08-a"


# ---------------------------------------------------------------------------------------
# Where the task data comes from, which is the part that decides whether this job belongs
# in a sealed-corpus experiment at all.
# ---------------------------------------------------------------------------------------

#: Where the dolma2 tokenizer sits in the research image, and the environment variable the
#: Dockerfile sets to the same path.
#:
#: THE ONE FILE THE OFFLINE PATH NEEDS THAT THE WHEEL DOES NOT CARRY. ``ai2-olmo-eval``
#: ships every task's requests and every HuggingFace split inside the wheel -- 110 MB of
#: ``olmo_eval/oe_eval_tasks`` and ``olmo_eval/hf_datasets``, read through
#: ``importlib_resources`` and ``datasets.load_from_disk``, which reach no network. It also
#: ships two tokenizers, and neither of them is dolma2. ``TokenizerConfig.dolma2().identifier``
#: is ``allenai/dolma2-tokenizer``, and ``olmo_eval.HFTokenizer`` resolves an identifier that
#: is neither a local file nor package data by calling ``Tokenizer.from_pretrained`` -- a
#: fetch from huggingface.co, in the middle of a run, which is the thing this module's
#: sealed-corpus stance forbids and whose failure would look like a scoring failure.
#:
#: So the Dockerfile downloads ``tokenizer.json`` at BUILD time from a pinned HuggingFace
#: revision, verifies its SHA-256, and points this variable at it. Build-time network is
#: what the image already uses for PyPI and for the flash-attn wheel; run-time network is
#: what it must not use. ``train_on_corpus.py``'s tokenizer comment asks for exactly this
#: ("vendor tokenizer.json into the image and point identifier at that local dir before BPB
#: evals") and this is that.
TOKENIZER_VARIABLE = "EDULLM_DOLMA2_TOKENIZER"
TOKENIZER_IN_THE_IMAGE = "/opt/edullm/tokenizers/dolma2-tokenizer/tokenizer.json"

#: SHA-256 of that file at HuggingFace revision ``5292e5d6c0f40b67cc765fe41bec991cf4345b5c``,
#: which is 4,237,178 bytes and identical to the tip of ``main`` when it was read on
#: 2026-08-08. Asserted in the Dockerfile at build time and again here at run time: the
#: build assertion catches a moved revision, and this one catches an image where something
#: else put a different tokenizer at that path. A run scored under the wrong tokenizer does
#: not fail -- every id is in range -- it just reports a worse model.
TOKENIZER_SHA256 = "969c214487b744f1457d8a4f2055fd4ad348edff5322d4f21f2906ceb158a636"


def resolve_tokenizer_path(explicit: Optional[str], environ=None) -> str:
    """
    Where the dolma2 tokenizer file is, and a refusal naming the three places looked in.

    :param explicit: ``--tokenizer``, or ``None``.
    :param environ: The environment to read. Defaults to ``os.environ``.

    :returns: A path to a ``tokenizer.json`` that exists.

    :raises train_on_corpus.Refusal: If no candidate exists, because the alternative is
        ``HFTokenizer`` silently reaching huggingface.co from inside a sealed run.
    """
    environ = os.environ if environ is None else environ
    candidates = [
        (explicit, "--tokenizer"),
        (environ.get(TOKENIZER_VARIABLE), f"${TOKENIZER_VARIABLE}"),
        (TOKENIZER_IN_THE_IMAGE, "the path the research image installs it at"),
    ]
    for candidate, _ in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    looked = ", ".join(f"{where} ({candidate or 'unset'})" for candidate, where in candidates)
    raise refuse(
        ScoringStage.THE_IMAGE_HAS_NO_TASK_DATA,
        "no local dolma2 tokenizer.json, and this program will not fall back to the "
        "HuggingFace identifier: olmo_eval.HFTokenizer would fetch it from the public "
        f"internet mid-run. Looked at {looked}. The research image installs it at "
        f"{TOKENIZER_IN_THE_IMAGE}; see .edullm/Dockerfile.",
    )


def build_tokenizer(path: str, *, verify: bool = True):
    """
    The dolma2 tokenizer, from a local file, with the ids the corpus was written with.

    The ids come from :meth:`olmo_core.data.TokenizerConfig.dolma2` rather than from the
    file, because they are the ids the training shards hold and a tokenizer that agreed
    with the file and disagreed with the corpus would score a different model than the one
    that was trained.

    :param path: A local ``tokenizer.json``.
    :param verify: Check the file against :data:`TOKENIZER_SHA256`.

    :returns: An ``olmo_eval.HFTokenizer``.

    :raises train_on_corpus.Refusal: If the file is not the pinned one, or if
        ``ai2-olmo-eval`` is not installed in this image.
    """
    import hashlib

    if verify:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != TOKENIZER_SHA256:
            raise refuse(
                ScoringStage.THE_IMAGE_HAS_NO_TASK_DATA,
                f"{path} is not the pinned dolma2 tokenizer: sha256 {digest.hexdigest()} "
                f"against the expected {TOKENIZER_SHA256}. Scoring with a tokenizer other "
                "than the one the corpus was written with does not fail -- every id is in "
                "range -- it reports a worse model.",
            )

    tokenizer_config = TokenizerConfig.dolma2()
    try:
        from olmo_eval import HFTokenizer
    except ImportError as missing:
        raise refuse(
            ScoringStage.THE_IMAGE_HAS_NO_TASK_DATA,
            "this image has no ai2-olmo-eval, so it carries neither the task data nor the "
            f"harness that reads it ({missing}). It is the 'eval' extra in pyproject.toml "
            "and .edullm/Dockerfile installs it; an image built before that commit cannot "
            "run this program.",
        ) from None

    return HFTokenizer(
        path,
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        bos_token_id=tokenizer_config.bos_token_id,
        vocab_size=tokenizer_config.vocab_size,
    )


# ---------------------------------------------------------------------------------------
# Which cell this is, and which checkpoint that cell reads.
# ---------------------------------------------------------------------------------------

#: The checkpoints one training cell wrote, at ``--save-interval 500`` over 6,000 steps:
#: one at step zero and twelve on the interval, the last of which is the final step. Stated
#: in ``train_hyper_connections.py``'s parser comment and derived here rather than copied,
#: so a change to either interval moves both.
LADDER_STEPS: Tuple[int, ...] = tuple(
    range(0, hyper_connection_arms.TRANCHE_STEPS + 1, hyper_connection_arms.TRANCHE_SAVE_INTERVAL)
)

#: The step the fifteen-cell job scores: the last one, which is the model the write-up is
#: about.
FINAL_STEP = LADDER_STEPS[-1]

#: Every ``(arm, seed, step)`` the full ladder would score, in fan-out order. 195 cells.
#:
#: WHY THE LADDER EXISTS AS A TABLE AND IS NOT THE DEFAULT. The fifteen-cell job answers
#: H2b. The ladder answers a different and cheaper-to-ask question -- whether an arm's
#: downstream number was still moving at 6,000 steps -- and DataDecide Sec. 3 finds
#: intermediate checkpoints as good as compute-equivalent final ones for ranking, so it is
#: not a luxury. It is thirteen times the cells and thirteen times the ceiling, which is why
#: it is a second submission rather than a flag on the first.
#:
#: The order is arm-major then seed then step, so cell 0 is the baseline seed 0 at step 0.
LADDER_CELLS: Tuple[Tuple[str, int, int], ...] = tuple(
    (arm, seed, step) for arm, seed in hyper_connection_arms.TRANCHE_CELLS for step in LADDER_STEPS
)

#: The fan-out label under which the index names an ``(arm, seed)`` pair and the step is
#: whatever ``--step`` says. The same label the tranche's own ``arm-and-seed`` fan-out uses,
#: deliberately: it means the same thing and is resolved by the same function.
FANOUT_INDEX_PARAMETER_CELL = train_hyper_connections.FANOUT_INDEX_PARAMETER_CELL

#: The fan-out label under which the index names an ``(arm, seed, step)`` triple out of
#: :data:`LADDER_CELLS`.
FANOUT_INDEX_PARAMETER_LADDER = "arm-seed-and-step"

#: How the platform lays out one cell of a fan-out underneath a run's output prefix, from
#: ``edullm_platform.execution.FANOUT_PROLOGUE``: the prologue appends ``cell-<index>/`` to
#: ``$EDULLM_OUTPUT_PREFIX`` and then defines ``$EDULLM_CHECKPOINT_DIR`` as that plus
#: ``checkpoints/``. Each stage of the tranche was a ``--fanout-index-parameter seed``
#: fan-out, so a stage cell's index IS its seed and this template is exact.
#:
#: WRITTEN DOWN HERE BECAUSE THE PLATFORM CANNOT BE ASKED. ``edullm_platform.weights`` has
#: ``resolve_weights_from_run`` and ``RunManifestV2`` carries an ``InputRole.WEIGHTS``, but
#: ``check`` and ``submit`` take only ``--dataset`` and compile a ``schema_version: 1``
#: manifest, so a checkpoint cannot be declared as an input and this job gets no lineage
#: edge back to the runs that produced the weights it reads. That is a real gap and it is
#: accepted rather than worked around: the URIs are in the command, the command is in the
#: sealed submission record, and a reader can follow it by hand.
CELL_TEMPLATE = "cell-{seed}/checkpoints"


def cell_of_the_ladder(index: int) -> Tuple[str, int, int]:
    """
    Which arm, replicate and step the ladder cell at this index is.

    :param index: ``$AWS_BATCH_JOB_ARRAY_INDEX``, contiguous from zero.

    :returns: ``(arm_name, seed, step)``.

    :raises IndexError: If the index is outside the ladder, which means the submission's
        ``--fanout-size`` and this table disagree about how many cells there are.
    """
    if not 0 <= index < len(LADDER_CELLS):
        raise IndexError(
            f"cell {index} of a ladder that has {len(LADDER_CELLS)} cells. The submission's "
            f"--fanout-size and score_checkpoints.LADDER_CELLS disagree; submit with "
            f"--fanout-size {len(LADDER_CELLS)}."
        )
    return LADDER_CELLS[index]


def resolve_target(opts, environ=None) -> Tuple[str, int, int, str]:
    """
    Which arm, replicate and step this process scores, and where the answer came from.

    Three shapes, and the label on the fan-out index is what tells them apart:

    1. ``arm-and-seed``, fifteen cells. The index names a pair out of
       :data:`hyper_connection_arms.TRANCHE_CELLS`, through the identical
       :func:`train_hyper_connections.resolve_cell` the training tranche used, and the step
       comes from ``--step``.
    2. ``arm-seed-and-step``, 195 cells. The index names a triple out of
       :data:`LADDER_CELLS` and ``--step`` is refused, because the index already decided it.
    3. No fan-out. ``--arm`` and ``--seed`` say, which is what a laptop and a preflight get.

    :param opts: Parsed options carrying ``arm``, ``seed`` and ``step``.
    :param environ: The environment to read. Defaults to ``os.environ``.

    :returns: ``(arm_name, seed, step, provenance)``.

    :raises train_on_corpus.Refusal: If a flag the index owns was passed anyway, if the
        index is outside its table, or if nothing says which arm to score.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(train_hyper_connections.FANOUT_INDEX_VARIABLE)
    label = environ.get(train_hyper_connections.FANOUT_PARAMETER_VARIABLE)

    if raw not in (None, "") and label == FANOUT_INDEX_PARAMETER_LADDER:
        for flag, value in (("--arm", opts.arm), ("--seed", opts.seed), ("--step", opts.step)):
            if value is not None:
                raise refuse(
                    Stage.THE_CONFIG_WOULD_NOT_BUILD,
                    f"this is cell {raw} of an {FANOUT_INDEX_PARAMETER_LADDER!r} fan-out, "
                    f"whose index is what says which arm, which replicate and which step "
                    f"this cell scores, and the command also passes {flag} {value}. Every "
                    f"cell is handed the same command, so honouring it would score one "
                    f"checkpoint {len(LADDER_CELLS)} times and report it as a ladder.",
                )
        try:
            arm, seed, step = cell_of_the_ladder(int(raw))
        except ValueError:
            raise refuse(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"${train_hyper_connections.FANOUT_INDEX_VARIABLE} is {raw!r}, which is not "
                "an integer, so this cell has no ladder rung to be.",
            ) from None
        except IndexError as mismatch:
            raise refuse(Stage.THE_CONFIG_WOULD_NOT_BUILD, str(mismatch)) from None
        return (
            arm,
            seed,
            step,
            (
                f"${train_hyper_connections.FANOUT_INDEX_VARIABLE}={int(raw)}, which is cell "
                f"{int(raw)} of {len(LADDER_CELLS)} in the ladder table"
            ),
        )

    arm, seed, provenance = train_hyper_connections.resolve_cell(opts.arm, opts.seed, environ)
    step = FINAL_STEP if opts.step is None else opts.step
    return arm, seed, step, provenance


def checkpoint_uri(run_root: str, seed: int, step: int, template: str = CELL_TEMPLATE) -> str:
    """
    Where one training cell's checkpoint at one step is.

    :param run_root: The training run's output prefix, e.g.
        ``s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/<run id>/``. A local
        directory works too, which is what the smoke test uses.
    :param seed: The replicate, which for a ``--fanout-index-parameter seed`` stage is also
        that cell's fan-out index and therefore its ``cell-<n>`` directory.
    :param step: The optimizer step the checkpoint was written at.
    :param template: How a cell sits under a run root. Defaults to :data:`CELL_TEMPLATE`.

    :returns: The checkpoint directory URI, with no trailing slash.
    """
    root = run_root.rstrip("/")
    cell = template.format(seed=seed).strip("/")
    return f"{root}/{cell}/step{step}" if cell else f"{root}/step{step}"


def parse_arm_runs(values: Sequence[str]) -> Dict[str, str]:
    """
    Turn ``--arm-run baseline=s3://...`` pairs into a mapping, refusing an unknown arm.

    :param values: The raw ``arm=uri`` strings.

    :returns: Arm name to run root.

    :raises train_on_corpus.Refusal: If a value has no ``=``, names an arm the table does
        not have, or names one twice.
    """
    runs: Dict[str, str] = {}
    for value in values:
        arm, _, uri = value.partition("=")
        if not uri:
            raise refuse(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"--arm-run {value!r} is not <arm>=<uri>.",
            )
        if arm not in ARMS:
            raise refuse(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"--arm-run names arm {arm!r}, which is not in the arm table. Known arms: "
                + ", ".join(sorted(ARMS)),
            )
        if arm in runs:
            raise refuse(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"--arm-run names {arm!r} twice, and the second would silently win.",
            )
        if "," in uri:
            # `--arm-run` is `append`, and a comma-joined list is the natural mistake: it
            # parses, the first arm gets a URI with a comma and two more arms in it, and the
            # other two arms are simply absent. That is a run that pulls an image and
            # refuses on ten of its fifteen cells.
            raise refuse(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"--arm-run {value!r} has a comma in its URI. This flag repeats rather than "
                "taking a list; pass it once per arm.",
            )
        runs[arm] = uri
    return runs


# ---------------------------------------------------------------------------------------
# Rebuilding the model this checkpoint came out of.
# ---------------------------------------------------------------------------------------

#: The vocabulary every arm was built at: dolma2's 100,278 padded to a multiple of 128.
#: Derived rather than written as 100,352, so that it moves with the tokenizer config the
#: training path used.
PADDED_VOCAB_SIZE = TokenizerConfig.dolma2().padded_vocab_size()


#: The model factory every funded cell of the tranche was trained under, named here rather
#: than defaulted for the reason ``train_hyper_connections.build_parser`` re-points it: the
#: platform's default is ``olmo2_190M``, and a scoring cell that quietly built one would
#: load nothing and report nothing that meant anything.
DEFAULT_MODEL_FACTORY = "hc_370M"


def model_config_for(
    arm: str, seed: int, factory: str = DEFAULT_MODEL_FACTORY
) -> TransformerConfig:
    """
    The model config the training cell for this ``(arm, seed)`` built, from the arm table.

    THIS IS THE MAPPING THE TASK IS ABOUT, AND IT IS DERIVED RATHER THAN INFERRED FROM A
    PATH. The three funded arms do not share an architecture -- ``faithful`` and
    ``output-only`` replace the residual stream with hyper-connections and differ from each
    other in the input map -- so a loader that guessed the arm from the URI would load a
    baseline's weights into a hyper-connection model, or the reverse, and
    ``load_model_and_optim_state`` would raise on the missing keys if you were lucky and
    load a subset if you were not. What the arm is comes from the same table the training
    process applied, keyed by the same cell index, and :func:`check_against_saved_config`
    then holds the answer against the config the run itself wrote.

    ``init_seed`` moves with the replicate exactly as ``train_hyper_connections.build_config``
    moves it, which is what makes the cross-check able to catch a seed collapse: three cells
    that all resolved to seed 0 would have written three configs carrying the base seed, and
    a scoring cell that believes it is seed 2 would find one.

    :param arm: An arm name in :data:`hyper_connection_arms.ARMS`.
    :param seed: The replicate.
    :param factory: A :class:`~olmo_core.nn.transformer.TransformerConfig` factory, which
        ``hyper_connection_arms.install()`` has already put ``hc_370M`` and ``hc_rehearsal``
        on. Only the rehearsal size and the smoke test pass anything else.

    :returns: The model config, ready to ``build()``.

    :raises train_on_corpus.Refusal: If the arm or the factory is not known.
    """
    if arm not in ARMS:
        raise refuse(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"no arm named {arm!r}. Known arms: " + ", ".join(sorted(ARMS)),
        )
    build = getattr(TransformerConfig, factory, None)
    if build is None:
        raise refuse(Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {factory}")
    config = build(vocab_size=PADDED_VOCAB_SIZE)
    ARMS[arm].apply(config)
    config.init_seed = config.init_seed + seed
    return config


#: Fields of the saved model config that a scoring job is allowed to disagree with the
#: training run about, and why each one.
#:
#: AN ALLOWLIST RATHER THAN A LIST OF THINGS TO COMPARE, for the reason
#: ``hyper_connection_arms.STAGE_CONTRAST_EXEMPT`` gives: the failure is a field nobody
#: thought about, and a checked list silently permits whatever is not on it.
CONFIG_COMPARISON_EXEMPT: Dict[str, str] = {
    "dtype": (
        "The parameter dtype the model is materialized in, which --param-dtype sets here "
        "and the data-parallel config set during training. It changes the arithmetic of "
        "the forward pass and not which model this is."
    ),
}


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "_CLASS_":
                out[f"{prefix}._CLASS_" if prefix else "_CLASS_"] = child
                continue
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _flatten(f"{prefix}[{index}]", child, out)
    else:
        out[prefix] = value


def config_differences(expected: Dict[str, Any], saved: Dict[str, Any]) -> List[str]:
    """
    Every leaf on which the rebuilt model config and the saved one disagree.

    Flattened to dotted leaves rather than compared as nested dicts, because "these two
    configs differ" is not a diagnosis and ``block.hyper_connections.n_lanes: 4 != 1`` is.

    :param expected: :meth:`olmo_core.config.Config.as_config_dict` of the rebuilt config.
    :param saved: The ``model`` sub-document of the ``config.json`` beside the checkpoint.

    :returns: One sentence per disagreement, sorted, with the exempt fields left out.
    """
    left: Dict[str, Any] = {}
    right: Dict[str, Any] = {}
    _flatten("", expected, left)
    _flatten("", saved, right)
    exempt = set(CONFIG_COMPARISON_EXEMPT)
    differences = []
    for key in sorted(set(left) | set(right)):
        if key.rsplit(".", 1)[-1] in exempt:
            continue
        if left.get(key, "<absent>") != right.get(key, "<absent>"):
            differences.append(
                f"{key}: this cell expects {left.get(key, '<absent>')!r}, the checkpoint's "
                f"own config says {right.get(key, '<absent>')!r}"
            )
    return differences


def check_against_saved_config(
    expected: TransformerConfig, saved: Optional[Dict[str, Any]], *, arm: str, seed: int
) -> List[str]:
    """
    Hold the arm table's answer against the config the training run wrote beside its weights.

    WHAT THIS CATCHES THAT THE LOAD WOULD NOT. Loading a baseline checkpoint into a
    hyper-connection model raises, so the gross mistake is loud. The quiet ones are the
    ones worth a check: a cell that scored ``faithful`` seed 3's weights while believing it
    was seed 4 produces five numbers of which two are the same run, and a noise floor
    computed from them is too small. ``init_seed`` is in the saved config and moves with the
    replicate, so this compares it and the mistake becomes a refusal.

    :param expected: What :func:`model_config_for` built for this cell.
    :param saved: The ``model`` sub-document of ``config.json``, or ``None`` if the
        checkpoint has none.
    :param arm: The arm this cell believes it is.
    :param seed: The replicate this cell believes it is.

    :returns: Warnings, which is a one-element list when the checkpoint carried no config.

    :raises train_on_corpus.Refusal: If the two configs disagree on anything not in
        :data:`CONFIG_COMPARISON_EXEMPT`.
    """
    if saved is None:
        return [
            "this checkpoint carries no config.json, so the arm and the replicate are the "
            "arm table's word alone. ConfigSaverCallback writes one into every checkpoint "
            "directory a run of train_hyper_connections.py produces, so a missing one means "
            "this is not one of those."
        ]
    differences = config_differences(expected.as_config_dict(), saved)
    if differences:
        raise refuse(
            ScoringStage.THE_CHECKPOINT_DOES_NOT_DESCRIBE_THE_ARM_THIS_CELL_IS,
            f"this cell is {arm} seed {seed} and the checkpoint beside it was written by a "
            f"different model:\n  " + "\n  ".join(differences) + "\n"
            "Either --arm-run points this arm at another arm's run, or the fan-out index "
            "means something other than what this program read it as. Both produce a "
            "number rather than an error if this check is skipped.",
        )
    return []


# ---------------------------------------------------------------------------------------
# Reading the checkpoint.
# ---------------------------------------------------------------------------------------


def read_saved_config(checkpoint: str) -> Optional[Dict[str, Any]]:
    """
    The ``config.json`` a run's :class:`~olmo_core.train.callbacks.ConfigSaverCallback` wrote
    into this checkpoint directory.

    :param checkpoint: The ``step{N}`` directory, local or ``s3://``.

    :returns: The parsed document, or ``None`` if there is none.

    :raises train_on_corpus.Refusal: If the object exists and cannot be read, which is an
        IAM answer rather than a missing file and is worth telling apart.
    """
    from olmo_core.io import file_exists, get_bytes_range

    path = f"{checkpoint.rstrip('/')}/config.json"
    try:
        if not file_exists(path):
            return None
        # Read in one range rather than through cached_path, so a scoring cell needs no
        # writable cache directory and leaves nothing behind between the fifteen of them.
        raw = get_bytes_range(path, 0, 1 << 24)
    except Refusal:
        raise
    except BaseException as unreadable:
        raise refuse(
            train_on_corpus.read_failure(unreadable),
            f"reading {path}: {type(unreadable).__name__}: {unreadable}",
        ) from unreadable
    return json.loads(raw.decode("utf-8"))


def load_checkpoint(model, checkpoint: str, *, work_dir: str) -> None:
    """
    Put a training cell's weights into a freshly built model, and nothing else.

    ``load_model_and_optim_state`` with no optimizer reads only the keys the model's state
    dict asks for, so the optimizer shards -- three quarters of what a 370M checkpoint
    holds -- are never fetched. That is the difference between a scoring cell pulling about
    1.5 GB and pulling about 6.

    :param model: A built :class:`~olmo_core.nn.transformer.Transformer`.
    :param checkpoint: The ``step{N}`` directory, local or ``s3://``.
    :param work_dir: Somewhere to stage downloads.

    :raises train_on_corpus.Refusal: If the checkpoint is absent, unreadable, or does not
        fit the model this cell built.
    """
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    directory = f"{checkpoint.rstrip('/')}/model_and_optim"
    try:
        load_model_and_optim_state(directory, model, work_dir=work_dir)
    except Refusal:
        raise
    except FileNotFoundError as absent:
        raise refuse(
            ScoringStage.THE_CHECKPOINT_IS_NOT_WHERE_THIS_CELL_WAS_TOLD,
            f"no checkpoint at {directory}: {absent}. Check --arm-run and --step against "
            "what the training stage actually wrote; the platform's fan-out prologue puts "
            "each cell under cell-<index>/checkpoints/.",
        ) from absent
    except BaseException as unreadable:
        stage = train_on_corpus.read_failure(unreadable)
        if stage is Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS:
            stage = ScoringStage.THE_CHECKPOINT_IS_NOT_WHERE_THIS_CELL_WAS_TOLD  # type: ignore[assignment]
        raise refuse(
            stage, f"loading {directory}: {type(unreadable).__name__}: {unreadable}"
        ) from unreadable


# ---------------------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------------------


@dataclass
class TaskResult:
    """What one task produced."""

    label: str
    group: str
    metrics: Dict[str, float] = field(default_factory=dict)
    instances: int = 0
    requests: int = 0
    seconds: float = 0.0


def score_task(
    model,
    task: Task,
    *,
    tokenizer,
    device,
    max_sequence_length: int,
    rank_batch_size: int,
    limit_batches: Optional[int] = None,
) -> TaskResult:
    """
    Run one task and return every metric it reports.

    THE FORWARD PASS IS DRIVEN HERE RATHER THAN THROUGH ``EvaluatorCallback`` ON PURPOSE.
    That callback reaches ``trainer.train_module.eval_batch``, so using it would mean
    standing up a ``Trainer``, a ``TransformerTrainModule``, a data loader and an FSDP mesh
    for a job that has no optimizer, no data loader and one device.
    :class:`~olmo_core.train.callbacks.evaluator_callback.DownstreamEvaluator` is the part
    worth reusing -- it builds the task, owns the ``ICLMetric`` and yields batches -- and
    the six lines of loop are cheaper than the machinery.

    THE LOGITS ARE UPCAST TO float32 BEFORE THE METRIC SEES THEM, AND THAT IS NOT TIDINESS.
    ``ICLMetric.update`` sums a cross-entropy over the continuation, and the arm-to-arm
    differences this experiment is looking for are in the third decimal place of a
    bits-per-byte. bfloat16 carries about three significant decimal digits, so a
    log-partition summed in it would put the quantization noise on top of the effect.

    THE WHOLE BATCH GOES TO THE DEVICE AND NOT ONLY ``input_ids``, which is the one thing
    ``EvaluatorCallback.perform_eval`` does here that is easy to leave out. The metric is
    handed the batch as well as the logits: ``ICLMetric.update`` reads ``continuation`` as
    the target of a cross-entropy against logits that came off the model, so a batch left
    where the loader built it is a CPU target against a CUDA prediction and torchmetrics
    raises. It cannot be reproduced with ``--device cpu`` -- there the two agree by
    construction -- so it costs a card to find. ``move_to_device`` is the same helper the
    training loop uses, one line above its own ``update_metrics``.

    :param model: The loaded model, already in eval mode.
    :param task: The task to run.
    :param tokenizer: An ``olmo_eval.HFTokenizer``.
    :param device: The torch device to score on.
    :param max_sequence_length: The longest context the model may be handed.
    :param rank_batch_size: Tokens per forward pass.
    :param limit_batches: Stop after this many batches. For the smoke test only -- a
        truncated task is not a score and the output records that it was truncated.

    :returns: The task's metrics.

    :raises train_on_corpus.Refusal: If the task is not in the installed harness.
    """
    import torch

    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator
    from olmo_core.train.train_module import EvalBatchSizeUnit, EvalBatchSpec
    from olmo_core.utils import move_to_device

    started = time.monotonic()
    batch_spec = EvalBatchSpec(
        rank_batch_size,
        batch_size_unit=EvalBatchSizeUnit.tokens,
        max_sequence_length=max_sequence_length,
    )
    try:
        evaluator = DownstreamEvaluator(
            name="downstream",
            task=task.label,
            batch_spec=batch_spec,
            tokenizer=tokenizer,
            device=device,
        )
    except OLMoConfigurationError as unknown:
        raise refuse(ScoringStage.THE_IMAGE_HAS_NO_TASK_DATA, str(unknown)) from None

    evaluator.reset_metrics()
    requests = 0
    batches = 0
    documents: Set[int] = set()
    for index, batch in enumerate(evaluator):
        if limit_batches is not None and index >= limit_batches:
            break
        # The bookkeeping is read off the loader's own tensors, before the move, because
        # `int()` on a device tensor is a host-device sync and this one is per document
        # rather than per batch. `doc_id` is the loader's and is always on the host here.
        documents.update(int(doc) for doc in batch["doc_id"])
        requests += int(batch["input_ids"].shape[0])
        batch = move_to_device(batch, device)
        with torch.no_grad():
            logits = model(batch["input_ids"])
        # See the docstring. `.float()` is a no-op when the model is already fp32.
        evaluator.update_metrics(batch, None, logits.float())
        batches += 1

    if batches == 0:
        # `--limit-batches 0`, which is what the preflight passes: the task was built and
        # nothing was scored. `compute_metrics` would divide by an empty list.
        return TaskResult(label=task.label, group=task.group, seconds=time.monotonic() - started)

    metrics = {name: float(value.item()) for name, value in evaluator.compute_metrics().items()}
    # `compute_metrics` names its keys "<label> (<human readable metric>)". The suffix is
    # for a log line and is the wrong key for a JSON document that something will read back,
    # so the raw metric names are recovered from the same table the evaluator built them
    # from rather than by parsing the sentence.
    by_type = {}
    for metric_type, label in DownstreamEvaluator.metric_type_to_label.items():
        key = f"{task.label} ({label})"
        if key in metrics:
            by_type[metric_type] = metrics[key]

    return TaskResult(
        label=task.label,
        group=task.group,
        metrics=by_type,
        instances=len(documents),
        requests=requests,
        seconds=time.monotonic() - started,
    )


def aggregate(results: Sequence[TaskResult], metric: str = PRIMARY_METRIC) -> Dict[str, Any]:
    """
    The downstream average, and the per-group means it is made of.

    GROUPS ARE AVERAGED AND THEN THE GROUPS ARE, WHICH IS NOT THE SAME AS AVERAGING THE
    TASKS. MMLU is four labels of one benchmark; a flat mean would give it four votes
    against HellaSwag's one and the headline would be an MMLU number wearing a suite's name.
    ``canary`` is left out of the headline entirely -- see :data:`HEADLINE_GROUPS`.

    :param results: One entry per task.
    :param metric: Which metric to average. Tasks that do not report it are skipped, which
        is how a ``bpb``-only task sits in the same suite as a ``len_norm`` one.

    :returns: ``{"metric": ..., "groups": {...}, "headline": float or None, "tasks": n}``.
    """
    groups: Dict[str, List[float]] = {}
    for result in results:
        value = result.metrics.get(metric)
        if value is None or not math.isfinite(value):
            continue
        groups.setdefault(result.group, []).append(value)
    group_means = {name: sum(values) / len(values) for name, values in sorted(groups.items())}
    headline_values = [group_means[name] for name in HEADLINE_GROUPS if name in group_means]
    return {
        "metric": metric,
        "groups": group_means,
        "headline": (sum(headline_values) / len(headline_values)) if headline_values else None,
        "headline_groups": [name for name in HEADLINE_GROUPS if name in group_means],
        "tasks": sum(len(values) for values in groups.values()),
    }


# ---------------------------------------------------------------------------------------
# The program.
# ---------------------------------------------------------------------------------------

#: The name in every output document, so that something reading a directory of them knows
#: what it is holding and can refuse a shape it does not understand.
OUTPUT_SCHEMA = "edullm.hyper-connections.downstream.v1"


class _PrintFanoutSize(argparse.Action):
    """
    Answers ``--fanout-size`` at parse time, so that asking costs no import and no corpus.

    Prints the cell count of both shapes, because a submission needs one of the two and
    getting them the wrong way round is a run that scores one checkpoint many times or a
    ladder missing most of its rungs.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        del parser, namespace, values, option_string
        print(f"{len(hyper_connection_arms.TRANCHE_CELLS)}  {FANOUT_INDEX_PARAMETER_CELL}")
        print(f"{len(LADDER_CELLS)}  {FANOUT_INDEX_PARAMETER_LADDER}")
        raise SystemExit(0)


def build_parser() -> argparse.ArgumentParser:
    """
    The command line. See the module docstring for the two shapes it is invoked in.

    :returns: The parser.
    """
    parser = argparse.ArgumentParser(
        prog="score_checkpoints",
        description="Score saved checkpoints of the hyper-connection tranche downstream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    parser.add_argument(
        "--arm-run",
        action="append",
        default=[],
        metavar="ARM=URI",
        help="Where one arm's training run wrote its output, e.g. "
        "baseline=s3://sbsandbox-intern-edullm-outputs/teams/input-core/runs/<run id>/. "
        "Repeat once per arm. A cell reads <uri>/cell-<seed>/checkpoints/step<N>, which is "
        "where the platform's fan-out prologue puts a --fanout-index-parameter seed stage. "
        "THE URIs ARE IN THE COMMAND BECAUSE THEY CANNOT BE DECLARED: check and submit take "
        "only --dataset, so a checkpoint is not an input the manifest can carry and this job "
        "gets no lineage edge back to the runs whose weights it reads.",
    )
    parser.add_argument(
        "--cell-template",
        default=CELL_TEMPLATE,
        help="How one cell sits under a run root. Only change this for a run that was not "
        "a fan-out.",
    )
    parser.add_argument(
        "--arm",
        choices=sorted(ARMS),
        default=None,
        help="Which arm to score. LEAVE UNSET IN A FAN-OUT: the cell then takes its arm from "
        f"${train_hyper_connections.FANOUT_INDEX_VARIABLE}, and passing it anyway is refused "
        "rather than honoured because one command reaching fifteen cells would score one "
        "checkpoint fifteen times.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Which replicate to score. Refused inside a fan-out, for the reason --arm is.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=f"Which checkpoint. Defaults to {FINAL_STEP}, the last one a tranche cell "
        f"wrote. Refused inside an {FANOUT_INDEX_PARAMETER_LADDER!r} fan-out, whose index "
        "already decided it.",
    )
    parser.add_argument(
        "--model-factory",
        default=DEFAULT_MODEL_FACTORY,
        help="The size the checkpoint was trained at. Only the rehearsal size and the local "
        "smoke test pass anything but the default; a wrong one is caught anyway, because the "
        "config beside the checkpoint disagrees about d_model.",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="h2b",
        help="Which task suite. 'h2b' is the one the hypothesis is read off; 'smoke' is two "
        "tasks and is for proving the path.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="A local dolma2 tokenizer.json. Defaults to "
        f"${TOKENIZER_VARIABLE} and then to {TOKENIZER_IN_THE_IMAGE}. There is deliberately "
        "no fallback to the HuggingFace identifier: that is a public-internet fetch in the "
        "middle of a run whose claim is that it read nothing from one.",
    )
    parser.add_argument(
        "--skip-tokenizer-checksum",
        action="store_true",
        help="Do not check the tokenizer against the pinned digest. For a laptop holding a "
        "copy from somewhere else; never for a scored run.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("EDULLM_OUTPUT_PREFIX", ""),
        help="Where the JSON document goes. Defaults to this run's own output prefix. The "
        "same document is always printed on stdout, which is the channel the platform can "
        "read back out of the log stream.",
    )
    parser.add_argument("--work-dir", default="/tmp/score-cache")
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Defaults to cuda when torch can see one.",
    )
    parser.add_argument(
        "--param-dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="The dtype the model is scored in. THE DEFAULT MATCHES WHAT THE ARMS TRAINED "
        "IN, and it has to be in the command text rather than only in code: the platform's "
        "precision guard reads the words of the command, so a shape with no bfloat16 in "
        "hardware is refused for free at check time instead of dying on the first kernel.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="The longest context a task may hand the model, which is the sequence length "
        "the arms trained at.",
    )
    parser.add_argument(
        "--batch-tokens",
        type=int,
        default=8 * 1024,
        help="Tokens per forward pass. 8,192 is the arms' rank microbatch halved, which "
        "fits a 24 GB card with the activations a full-logits eval needs.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Stop each task after this many batches. FOR THE SMOKE TEST ONLY: a truncated "
        "task is not a score, and the output document says so.",
    )
    parser.add_argument(
        "--fanout-size",
        nargs=0,
        action=_PrintFanoutSize,
        help="Print how many cells a fan-out of this job has and exit, so that a submission "
        "command can be written without a literal that drifts. THE ARM TABLE MOVES: it went "
        "from nine cells to fifteen when the design bought five seeds instead of three, and "
        "from fifteen to twenty when mhc was funded, and neither edit touched anything that "
        "would have noticed a stale number in a spec file's header.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Resolve the cell, build the model config, check it against the checkpoint's "
        "own config, build every task in the suite, and exit without loading weights. Needs "
        "no GPU and no weights, so it runs on a laptop -- which is where the mistakes this "
        "catches are cheapest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print what would be scored, and touch nothing.",
    )
    return parser


def describe_suite(suite: Sequence[Task]) -> str:
    """
    The suite and what each task is for, for ``--dry-run`` and for the run log.

    :param suite: The tasks.

    :returns: One block of text.
    """
    width = max(len(task.label) for task in suite)
    lines = [f"{'task'.ljust(width)}  group     why"]
    for task in suite:
        lines.append(f"{task.label.ljust(width)}  {task.group:8s}  {task.why}")
    return "\n".join(lines)


def say(line: str) -> None:
    """
    Put a human-readable line where a person will see it and a parser will not.

    STDOUT IS RESERVED FOR THE ONE JSON DOCUMENT THIS PROGRAM PRODUCES. The platform reads a
    run's summary back out of the log stream, and the analysis reads twenty of these
    documents; both are cheaper if ``stdout`` is exactly one object rather than an object
    with a preamble that something has to learn to skip. ``train_on_corpus.summarise`` makes
    the same split and gets it for free, because everything it prints beside the summary
    goes through ``logging``.

    :param line: The line, already formatted.
    """
    print(line, file=sys.stderr, flush=True)


def resolve_device(requested: Optional[str]):
    """
    The torch device to score on, with its index filled in.

    CONCRETE RATHER THAN ``cuda``, because the index is what everything downstream is
    compared against. ``torch.device("cuda")`` carries ``index=None`` while every tensor
    that lands on it reports ``cuda:0``, so "is this tensor where the scorer put it" is not
    an equality anybody can write, and the run document would name card 0 whatever card the
    process was actually given. Both differences are invisible on a laptop and invisible on
    a one-card shape, which is every shape this job has been run on so far.

    :param requested: ``--device``, or ``None`` to take a GPU if there is one.

    :returns: A ``torch.device``.
    """
    import torch

    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def main() -> None:
    """
    Resolve the cell, load its checkpoint, score it, and write one document.

    :raises train_on_corpus.Refusal: For every condition this program refuses on, each
        carrying the stage that becomes the process's exit code.
    """
    logging.basicConfig(level=logging.INFO)
    opts = build_parser().parse_args()

    import torch

    from olmo_core.config import DType
    from olmo_core.utils import gc_cuda

    arm, seed, step, provenance = resolve_target(opts)
    say(f"cell         arm {arm}, seed {seed}, step {step}, from {provenance}")

    suite = SUITES[opts.suite]
    runs = parse_arm_runs(opts.arm_run)
    if arm not in runs:
        raise refuse(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            f"this cell scores arm {arm!r} and no --arm-run says where that arm's training "
            "run wrote its checkpoints. Every cell of the fan-out is handed the same "
            "command, so the command needs one --arm-run per funded arm: "
            + ", ".join(sorted(hyper_connection_arms.FUNDED)),
        )
    checkpoint = checkpoint_uri(runs[arm], seed, step, opts.cell_template)

    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        model_config = model_config_for(arm, seed, opts.model_factory)
    saved = read_saved_config(checkpoint)
    warnings = check_against_saved_config(
        model_config, None if saved is None else saved.get("model"), arm=arm, seed=seed
    )
    for warning in warnings:
        log.warning("%s", warning)

    say(f"checkpoint   {checkpoint}")
    say(
        f"model        {model_config.d_model}d x {model_config.n_layers}L, "
        f"{model_config.num_params:,} params"
    )
    say(f"suite        {opts.suite} ({SUITE_VERSION}), {len(suite)} tasks")
    if opts.dry_run:
        say(describe_suite(suite))
        return

    tokenizer_path = resolve_tokenizer_path(opts.tokenizer)
    tokenizer = build_tokenizer(tokenizer_path, verify=not opts.skip_tokenizer_checksum)
    say(f"tokenizer    {tokenizer_path}")

    device = resolve_device(opts.device)
    say(f"device       {device}, {opts.param_dtype}")
    if device.type != "cuda" and opts.device is None:
        # A GPU shape whose driver or CUDA_VISIBLE_DEVICES is wrong lands here, and the
        # fallback is silent: the job scores correctly and takes hours instead of minutes,
        # then hits the wall clock having spent the whole reservation. Not a refusal --
        # `--device cpu` is a real thing to want -- but not a thing to find out from the
        # `"device"` field of a document that was never written.
        say(
            "             NOTE: nothing asked for a device and torch can see no CUDA one, "
            "so this is scoring on the CPU. On a GPU shape that is a driver or a "
            "visibility problem rather than a choice."
        )
    dtype = DType(opts.param_dtype).as_pt()
    if device.type == "cuda":
        from olmo_core.exceptions import OLMoConfigurationError
        from olmo_core.train.train_module import validate_precision_support

        # THE SAME GUARD THE TRAINING PATH RUNS, at the same point and for the same reason:
        # before anything expensive, while the container still costs nothing to stop, and
        # because torch.cuda.is_bf16_supported() answers True on a card with no bfloat16
        # arithmetic. On a copy of the config rather than on the config itself, so the
        # comparison against the checkpoint's own config above stays a comparison of two
        # models rather than of a model and a dtype flag.
        declared = copy.deepcopy(model_config)
        declared.dtype = DType(opts.param_dtype)
        try:
            validate_precision_support(declared)
        except OLMoConfigurationError as unusable:
            raise refuse(
                Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, str(unusable)
            ) from None

    if opts.preflight:
        for task in suite:
            result = score_task(
                _NeverCalled(),
                task,
                tokenizer=tokenizer,
                device=torch.device("cpu"),
                max_sequence_length=opts.max_sequence_length,
                rank_batch_size=opts.batch_tokens,
                limit_batches=0,
            )
            say(f"             {task.label:42s} {result.seconds:5.1f}s to build")
        say("preflight OK")
        return

    started = time.monotonic()
    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        model = model_config.build(init_device="cpu")
    load_checkpoint(model, checkpoint, work_dir=opts.work_dir)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    loaded = time.monotonic() - started

    results: List[TaskResult] = []
    for task in suite:
        result = score_task(
            model,
            task,
            tokenizer=tokenizer,
            device=device,
            max_sequence_length=opts.max_sequence_length,
            rank_batch_size=opts.batch_tokens,
            limit_batches=opts.limit_batches,
        )
        results.append(result)
        # Between tasks, as `EvaluatorCallback.perform_eval` does between evaluators. Each
        # task holds a tokenized dataset and a metric of its own, and the logits of one
        # batch at 8,192 tokens over a 100,352-token vocabulary are 1.6 GB in bfloat16 and
        # another 3.3 GB once `.float()` has copied them. Thirteen tasks of that arriving in
        # different shapes is how a card that has the room fragments out of it anyway.
        gc_cuda()
        primary = result.metrics.get(PRIMARY_METRIC)
        log.info(
            "%s: %s=%s (%d instances, %.1fs)",
            task.label,
            PRIMARY_METRIC,
            "n/a" if primary is None else f"{primary:.4f}",
            result.instances,
            result.seconds,
        )

    document = {
        "schema": OUTPUT_SCHEMA,
        "run_id": opts.run_name,
        "arm": arm,
        "arm_number": ARMS[arm].number,
        "seed": seed,
        "step": step,
        "cell_provenance": provenance,
        "checkpoint": checkpoint,
        "suite": opts.suite,
        "suite_version": SUITE_VERSION,
        "primary_metric": PRIMARY_METRIC,
        "truncated": opts.limit_batches is not None,
        "warnings": warnings,
        "tokenizer": tokenizer_path,
        "param_dtype": opts.param_dtype,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": loaded,
        "score_seconds": time.monotonic() - started - loaded,
        "tasks": {
            result.label: {
                "group": result.group,
                "instances": result.instances,
                "requests": result.requests,
                "seconds": result.seconds,
                "metrics": result.metrics,
            }
            for result in results
        },
        "downstream": {
            metric: aggregate(results, metric) for metric in (PRIMARY_METRIC, *SECONDARY_METRICS)
        },
    }
    write_document(document, opts.output_dir, arm=arm, seed=seed, step=step)


def write_document(
    document: Dict[str, Any], output_dir: str, *, arm: str, seed: int, step: int
) -> None:
    """
    Print the result on stdout and, if there is somewhere to put it, write it there too.

    STDOUT IS THE CHANNEL THAT ALWAYS WORKS, which is the argument
    ``train_on_corpus.summarise`` makes: the platform reads a run's own summary back out of
    the log stream, and a cell whose S3 write failed still reported its numbers. The object
    is what an analysis reads fifteen of.

    :param document: The result.
    :param output_dir: A directory or ``s3://`` prefix, or empty to print only.
    :param arm: For the filename.
    :param seed: For the filename.
    :param step: For the filename.
    """
    body = json.dumps(document, indent=2, sort_keys=True)
    print(body, flush=True)
    if not output_dir:
        return

    name = f"downstream-{arm}-seed{seed}-step{step}.json"
    target = f"{output_dir.rstrip('/')}/{name}"
    try:
        from olmo_core.io import is_url, upload

        if is_url(target):
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
                handle.write(body)
                staged = handle.name
            try:
                upload(staged, target, save_overwrite=True)
            finally:
                os.unlink(staged)
        else:
            os.makedirs(output_dir, exist_ok=True)
            with open(target, "w") as handle:
                handle.write(body)
    except BaseException as unwritable:  # noqa: BLE001 -- see the docstring
        # Swallowed for the reason `leave_the_reason_in_wandb` swallows: the numbers are
        # already on stdout, and a failed write of a second copy is not worth losing them.
        print(
            f"could not write {target}: {type(unwritable).__name__}: {unwritable}", file=sys.stderr
        )
        return
    print(f"wrote {target}", file=sys.stderr)


class _NeverCalled:
    """
    Stands in for the model during ``--preflight``, which builds every task and scores none.

    A class rather than ``None`` so that a preflight that started scoring by mistake raises
    here, naming the reason, instead of failing somewhere inside the metric with a
    ``NoneType`` message.
    """

    def __call__(self, *args, **kwargs):
        raise AssertionError("preflight built a task and then tried to score it")


def cli() -> int:
    """
    Run, and turn a refusal into a number a person on the platform side can see.

    The same boundary ``train_on_corpus.cli`` is, and deliberately the same shape: the
    stage becomes the exit status, the explanation goes to stderr for whoever can read the
    log, and the same explanation goes to W&B for everyone who cannot.

    :returns: The process exit status.
    """
    try:
        main()
    except Refusal as refusal:
        print(refusal.explanation, file=sys.stderr)
        print(f"edullm-stage: {refusal.stage.name} exit={int(refusal.stage)}", file=sys.stderr)
        if refusal.__cause__ is not None:
            traceback.print_exception(
                type(refusal.__cause__), refusal.__cause__, refusal.__cause__.__traceback__
            )
        train_on_corpus.leave_the_reason_in_wandb(
            run_name=os.environ.get("EDULLM_RUN_ID", "local"),
            stage=refusal.stage,
            explanation=refusal.explanation,
        )
        return int(refusal.stage)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
