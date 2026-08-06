"""Train on a published eduLLM corpus, resolved at run time rather than written down.

    python .edullm/train_on_corpus.py "$EDULLM_RUN_ID" [OVERRIDES...]

WHAT THIS EXISTS FOR. ``src/examples/llm/train.py`` trains on a single C4 shard streamed from
``http://olmo-data.org`` with the GPT-2 tokenizer, both hard-coded. That is the right default
for the upstream example and the wrong one for this platform: a researcher who picks
``regmix-10b-v1`` on the submission form and runs the example gets a run that reads AI2's
sample over the public internet, reports a loss curve, writes a checkpoint, and never opens
the corpus they chose. Nothing fails. The record says which corpus was requested and the run
read another one.

So the data here comes from ``edullm_data.read``, which resolves a dataset id and version into
object URIs by reading the manifest the validator sealed. There is no path literal in this
file and there is deliberately no flag to supply one: a path typed on a command line is the
failure above wearing different clothes.

WHAT THE PLATFORM HANDS THIS PROCESS. Four environment variables, all set by the submission
path rather than by the person submitting:

    EDULLM_DATASET_ID         pretrain/regmix-10b
    EDULLM_DATASET_VERSION    v1
    EDULLM_DATASET_TOKENIZER  tokenizer/dolma2-bpe
    EDULLM_CHECKPOINT_DIR     s3://.../teams/<team>/runs/<run id>/checkpoints/

The first three come from the registry entry for whatever the form's dataset field named, so
they cannot disagree with the record. The fourth is why a second attempt resumes instead of
silently repeating the first at full price.

THE THREE THINGS THAT CORRUPT DATA SILENTLY, AND WHY EACH IS ASSERTED BELOW. All three decode
into token ids that are in range and plausible. None raises. The only symptom is a loss curve
that is merely worse than it should be, which is indistinguishable from a bad hyperparameter.

  1. dtype. ``NumpyDatasetConfig.get_dtype`` falls back to the NARROWEST dtype the tokenizer's
     vocab fits in when ``dtype`` is left unset -- 100,278 fits in uint16, so a dolma2 corpus
     stored as uint32 gets read two bytes at a time. The manifest knows the real width, and
     this file passes it explicitly. It is never inferred.
  2. Byte order. ``np.memmap`` uses the HOST's, and the manifest declares the file's.
  3. Header bytes. OLMo-core memmaps from offset zero; a container format with a leading
     header decodes that header as tokens. The headerless ``.u32le.bin`` form is zero here,
     and anything else is refused rather than read wrong.

HOW THIS SAYS WHAT WENT WRONG, GIVEN THAT NOBODY CAN READ ITS LOG. A container that fails
before training writes its explanation to a CloudWatch stream that no credential on the
platform side may read, and Batch reports only ``exitCode`` and "Essential container in task
exited". So the exit code carries the stage -- see ``Stage`` -- and the explanation is also
written to W&B, which is the one place a run's own output lands that a researcher can open.

WHAT A RETRY HAS TO CLEAN UP BEFORE IT CAN GET PAST THE STEP THAT KILLED IT. See
``remove_torn_checkpoints``. Resuming correctly is not enough on its own: a second attempt
that skips an unfinished step directory on the way in still meets it on the way out, when
the trainer reaches that step number again and refuses to write into a directory that is
not empty.
"""

import argparse
import contextlib
import copy
import enum
import json
import logging
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, cast

import rich
import torch

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.io import clear_directory, list_directory, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    LinearWithWarmup,
    OptimGroupOverride,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)


class Stage(enum.IntEnum):
    """Which stage a run died in, said in the exit code, because nobody can read the log.

    THE EXIT CODE IS THE ONLY CHANNEL OUT OF A CONTAINER THAT DIES BEFORE W&B EXISTS.
    Batch reports ``status``, ``statusReason`` and ``exitCode``, and for a container that
    exits on its own the reason is always "Essential container in task exited". The
    explanation is on stdout, in a CloudWatch stream, and no credential on the platform side
    holds ``logs:GetLogEvents`` -- the researcher-facing role that would is not deployed, and
    the deploy role deliberately cannot read other tenants' logs in a shared account.

    So on 2026-08-01 four runs died between five and forty seconds with exit 1, and exit 1 is
    what a bad hyperparameter, a lost quote, a missing entry point, an unreadable bucket and a
    wrong argument all look like. Each was diagnosed by resubmitting with a change and seeing
    whether the number moved -- at an A10G and several minutes of image pull per guess.

    One number per stage costs nothing and answers the first question every time: 65 says the
    role cannot read the corpus, 66 says the corpus is not where the registry says, 67 says the
    manifest is not safe to memmap. None of those is a training problem and all three are
    indistinguishable from one at exit 1.

    The values are in the conventional ``sysexits`` range and stay clear of 126, 127 and 128+n,
    which the shell and the signal convention already own.
    """

    THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT = 64
    THE_ROLE_MAY_NOT_READ_THE_CORPUS = 65
    THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS = 66
    THE_READER_FAILED_IN_SOME_OTHER_WAY = 67
    THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP = 68
    THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER = 69
    THE_CONFIG_WOULD_NOT_BUILD = 70
    THE_TRAINING_ENVIRONMENT_WOULD_NOT_START = 71
    TRAINING_ITSELF_FAILED = 72


class Refusal(SystemExit):
    """A refusal that carries which stage it came from as well as what to tell the person.

    A ``SystemExit`` subclass so that every existing ``raise SystemExit(message)`` reads the
    same to a caller and to a test, and so an accidental escape still stops the process. What
    it adds is ``stage``, which ``cli()`` turns into the process's exit status.
    """

    def __init__(self, stage: "Stage", explanation: str) -> None:
        super().__init__(explanation)
        self.stage = stage
        self.explanation = explanation


@contextlib.contextmanager
def during(stage: Stage) -> Iterator[None]:
    """Tag whatever goes wrong in here with the stage it went wrong in.

    Only for the unforeseen. A refusal this file writes on purpose already knows its stage and
    passes through untouched; what this catches is the ``AttributeError`` from a library that
    changed under us, which is precisely the class of failure that arrives as a bare exit 1.
    """
    try:
        yield
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(stage, f"{type(exc).__name__}: {exc}") from exc


def _looks_like(exc: BaseException, *words: str) -> bool:
    """Whether an exception, or anything it was raised from, mentions one of these.

    Deliberately string-matching rather than catching ``botocore.exceptions.ClientError``.
    The reader wraps S3 errors in its own types, those types are not importable at the top of
    this file, and the distinction being drawn -- refused versus absent -- is one that both
    botocore and the reader spell in words in the message.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = f"{type(exc).__name__}: {exc}".lower()
        if any(word in text for word in words):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def read_failure(exc: BaseException) -> Stage:
    """Refused, absent, or something else -- the distinction that decides what to do next.

    A probe that recorded only "the read failed" would read the same for a role missing a
    grant and for a registry pointing at a prefix nobody published, and the two have nothing
    in common: one is an IAM change and one is a dataset that is not there.
    """
    if _looks_like(exc, "accessdenied", "403", "forbidden", "not authorized"):
        return Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS
    if _looks_like(exc, "nosuchkey", "nosuchbucket", "404", "not found", "no such"):
        return Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS
    return Stage.THE_READER_FAILED_IN_SOME_OTHER_WAY


def leave_the_reason_in_wandb(*, run_name: str, stage: Stage, explanation: str) -> None:
    """Put the traceback where the researcher already looks, since the log is unreachable.

    W&B is the one place a run's own output lands that somebody on this platform can open. A
    container that dies during startup never gets there, because the trainer's W&B callback
    initialises well after the corpus is resolved -- so the runs that most need explaining are
    exactly the ones that leave nothing behind.

    This creates a run of its own for that case, named after the platform run id and tagged so
    it sorts away from real training. If training itself failed there is already a run open and
    the reason is written into that one instead of beside it.

    Every failure in here is swallowed. A diagnostic that replaces the error it was reporting
    with its own is worse than no diagnostic, and W&B is reachable over a network that a
    broken container may be exactly what is broken about.
    """
    project = os.environ.get("EDULLM_WANDB_PROJECT")
    if not project:
        return
    try:
        import wandb

        # Through the environment rather than through Settings, whose accepted fields move
        # between wandb versions. A default here only applies if nothing else set one.
        os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
        run = wandb.run
        if run is None:
            run = wandb.init(
                project=project,
                name=f"{run_name}-died",
                job_type="crash",
                tags=["died-before-training", stage.name.lower().replace("_", "-")],
            )
        run.summary["edullm_stage"] = stage.name
        run.summary["edullm_exit_code"] = int(stage)
        run.summary["edullm_explanation"] = explanation
        run.finish(exit_code=int(stage))
    except BaseException as exc:  # noqa: BLE001 -- see the docstring
        print(f"could not leave the reason in W&B: {type(exc).__name__}: {exc}", file=sys.stderr)


# WHICH TOKENIZER EACH PUBLISHED ONE IS, SPELLED OUT RATHER THAN GUESSED.
#
# The left side is a published tokenizer id under s3://edullm-data; the right is the
# OLMo-core config that reproduces it. The join has to be written down somewhere, and a
# mapping that fails on an unknown key is the honest place: the alternative -- defaulting to
# dolma2, or to gpt2 as the example does -- answers "I do not know this tokenizer" with a run
# that trains on ids meaning something other than what they meant when the corpus was built.
#
# tokenizer/bytes-utf8 is deliberately absent. OLMo-core has no byte tokenizer, and inventing
# a 256-entry TokenizerConfig here would produce exactly the uint16 inference described above.
# The platform already keeps that corpus off the submission form for the same reason; if a
# byte tokenizer lands upstream, adding a line here is what makes the corpus runnable.
TOKENIZERS = {
    "tokenizer/dolma2-bpe": TokenizerConfig.dolma2,
}


@dataclass
class ExperimentConfig(Config):
    """Everything the run is, in one object the config saver writes beside the checkpoint.

    ``dataset_id`` and ``dataset_version`` are carried here rather than left in the
    environment so that the saved config -- which lands in the checkpoint directory and in
    W&B -- names the corpus the run actually opened. A record that says which corpus was
    requested is a different fact from one that says which was read.
    """

    model: TransformerConfig
    dataset: NumpyFSLDatasetConfig
    data_loader: NumpyDataLoaderConfig
    trainer: TrainerConfig
    train_module: TransformerTrainModuleConfig
    dataset_id: str = ""
    dataset_version: str = ""
    init_seed: int = 12536


@dataclass
class Corpus:
    """What the manifest says, after the three checks that make it safe to memmap."""

    dataset_id: str
    version: str
    paths: List[str]
    dtype: NumpyDatasetDType
    tokenizer: TokenizerConfig
    rows: Optional[int]
    #: Held-out URIs, or empty when the dataset declares none. Kept SEPARATE from ``paths``
    #: rather than concatenated, which is the reader's own reasoning: a flat list is the bug,
    #: because a caller cannot tell the two apart and held-out shards end up in training with
    #: nothing to notice. ``ResolvedSplit.val`` returns ``None`` for "no validation data";
    #: normalised to ``[]`` here so callers branch on truthiness rather than on ``is None``.
    val_paths: List[str] = field(default_factory=list)


def corpus_from_manifest(read, *, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    """Turn what the reader returned into what OLMo-core needs, or refuse and say why.

    Separate from the fetch because this is the part with the judgement in it, and a test
    should be able to hand it a manifest describing a big-endian corpus without standing up
    S3 or installing the reader. ``read`` is duck-typed for that reason: anything carrying
    ``paths``, ``dtype``, ``byte_order``, ``header_bytes`` and ``rows`` will do.
    """
    if not read.paths:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} resolved to no trainable shards",
        )

    if read.dtype is None:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} declares no dtype, so there is no width to read it at. "
            "A fixed-width corpus must; refusing rather than guessing.",
        )
    if read.header_bytes:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} declares {read.header_bytes} header bytes and OLMo-core "
            "memmaps from offset zero, so the header would be decoded as tokens.",
        )
    if read.byte_order is not None and read.byte_order != sys.byteorder:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} is {read.byte_order}-endian and this host is "
            f"{sys.byteorder}-endian. numpy would read every token to a different, "
            "in-range-looking id.",
        )

    try:
        tokenizer = TOKENIZERS[tokenizer_id]()
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS)) or "none"
        raise Refusal(
            Stage.THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER,
            f"no OLMo-core config for {tokenizer_id}; this image knows: {known}",
        ) from None

    # ``val`` is a property on the reader's ResolvedSplit and returns None when the dataset
    # declares no held-out split. getattr keeps this function's duck-typed contract intact --
    # its docstring promises anything with paths/dtype/byte_order/header_bytes/rows will do,
    # and the tests hand it stub objects that predate this field.
    val_paths = list(getattr(read, "val", None) or [])

    # A held-out shard that is also a training shard is not held out. The reader derives the
    # split from each filename rather than trusting the declaration, so this should be
    # impossible -- but this project has already shipped a contaminated "held-out" set once
    # (val files sitting at indices 0/128/163 of the training path list), and the failure is
    # invisible: the eval number just looks better than it is. Cheap, and it fails loudly.
    overlap = sorted(set(val_paths) & set(read.paths))
    if overlap:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} lists {len(overlap)} shard(s) in BOTH the trainable and "
            f"held-out splits, so held-out loss would be measured on trained data. "
            f"First: {overlap[0]}",
        )

    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths=list(read.paths),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=tokenizer,
        rows=read.rows,
        val_paths=val_paths,
    )


def resolve_corpus(*, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    # Imported here rather than at the top so that everything above can be exercised on a
    # host without the reader installed. In the image it is always present -- the Dockerfile
    # asserts the import at build time -- so this defers nothing that can fail in a run.
    from edullm_data.read import dataset_paths, resolve_latest
    from edullm_data.s3 import Boto3S3

    # NOT boto3.client("s3"), WHICH IS WHAT THIS SAID AND WHAT IT COST.
    #
    # The reader's `s3` parameter is typed against its own four-method protocol -- get, head,
    # get_range, list -- and a boto3 client implements none of them. `_require_validated`
    # calls `s3.head(bucket, key)`; a boto3 client has no such method, so the run died with an
    # AttributeError before a single byte left the account.
    #
    # It presents as the most misleading failure available. The name of the parameter is `s3`,
    # a boto3 client is what `s3` means everywhere else in this file's world, the type
    # annotation is a Protocol so nothing checks it at the call, and the traceback names a
    # missing attribute rather than a wrong argument. On a GPU job it is a container that
    # starts, exits 1 in under a second, and writes its only explanation to a log stream
    # nobody on the platform side is allowed to read. It took three submissions and a probe
    # whose exit codes encoded which stage failed.
    #
    # Boto3S3.default() is the reader's own adapter and takes the credentials from the task
    # environment, which on Batch is the workload role.
    s3 = Boto3S3.default()

    # "latest" resolves through the catalog rather than being an alias anybody can move. A
    # pinned version is the normal case and what the platform sends; this branch exists so a
    # person poking at the image by hand does not have to look one up first.
    if version in ("", "latest"):
        try:
            resolved = resolve_latest(dataset_id, s3=s3)
        except Refusal:
            raise
        except BaseException as exc:
            raise Refusal(read_failure(exc), f"{type(exc).__name__}: {exc}") from exc
        if resolved is None:
            raise Refusal(
                Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS,
                f"no published version of {dataset_id}",
            )
        version = resolved

    # split is left at its default, which returns TRAINABLE shards only. Passing split="train"
    # would work today and would break quietly on a corpus that names its trainable split
    # anything else; the default is the reader's own answer to "what may this run see", and
    # held-out shards are not it.
    # THE STAGE THAT ACTUALLY TOUCHES THE ACCOUNT, AND THE ONE WORTH TELLING APART FROM THE
    # REST. Everything above this line is local. This call HEADs the seal, GETs the manifest
    # and lists the group, so it is where a missing s3:GetObject on edullm-data shows up --
    # and a role without that grant and a registry entry pointing at an unpublished prefix
    # both arrive here as a failed read. read_failure separates them.
    try:
        read = dataset_paths(dataset_id, version, s3=s3)
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(
            read_failure(exc),
            f"reading {dataset_id}/{version}: {type(exc).__name__}: {exc}",
        ) from exc
    return corpus_from_manifest(
        read, dataset_id=dataset_id, version=version, tokenizer_id=tokenizer_id
    )


#: How OLMo-core names a checkpoint directory, and therefore where the step number is
#: written down. ``Checkpointer.CHECKPOINT_DIR`` is the format string this matches; the
#: pattern is spelled out rather than derived from it so that a change upstream shows up as
#: this failing to find a directory rather than as a regex that quietly stops matching.
STEP_DIRECTORY = re.compile(r"^step(\d+)$")


def torn_step_directories(save_folder: str) -> List[str]:
    """Every ``step{N}`` directory under the save folder that the library's loader refuses.

    Judged by ``Checkpointer.dir_is_checkpoint``, which is the same function
    ``find_checkpoints`` filters on, so this and the resume path agree on which directories
    are real by construction rather than by two implementations matching.

    A complete checkpoint is therefore never a candidate. That is the whole safety
    property: the removal below cannot take a directory a resume would have loaded, because
    the test for "would a resume load this" is the test for "leave it alone".

    Sorted so the log reads in step order. Returns paths, not step numbers, because the
    caller has to remove them and the path is what it removes.
    """
    try:
        children = list(list_directory(save_folder, include_files=False))
    except FileNotFoundError:
        # No save folder yet, which is every first attempt.
        return []
    torn = [
        path
        for path in children
        if STEP_DIRECTORY.match(os.path.basename(normalize_path(path))) is not None
        and not Checkpointer.dir_is_checkpoint(path)
    ]
    return sorted(torn)


def remove_torn_checkpoints(save_folder: str) -> List[str]:
    """Clear the unfinished step directories a lost attempt left, so this one can rewrite them.

    THE DEFECT THIS EXISTS FOR, AND WHY READING THE RESUME PATH DID NOT FIND IT. Killing the
    host during a checkpoint write leaves a ``step{N}`` holding part of one -- on
    ``run_019fbe1f-b84f-703a-8eb8-2b4504232948``, exactly ``step100/train/rank0.pt`` and
    nothing else, because ``rank0.pt`` is written before the first ``model_and_optim`` shard
    starts. The retry resumes from ``step50`` and skips ``step100``, correctly and by
    design: ``find_checkpoints`` drops any directory failing ``dir_is_checkpoint`` and
    ``latest_checkpoint`` takes the highest of what survives.

    Then it trains back to step 100, saves, and ``Checkpointer._prepare_dir`` raises
    ``FileExistsError`` on a directory that is not empty. Deterministically, at the same
    step, on every attempt. With two attempts that is the end of the run. The read path
    being right is what makes this so easy to miss: the resume line in the log says the
    recovery worked, and the recovery then fails half an hour later at the write.

    WHY THIS AND NOT ``save_overwrite=True``, WHICH IS THE ONE-LINE VERSION. That flag makes
    ``_prepare_dir`` clear the target directory before every save, whatever is in it, and
    turns every object write into an unconditional overwrite. The refusal it removes is the
    one that caught this bug. What is removed here is only a directory the library itself
    will not read, which is a strictly smaller set that happens to contain exactly the
    problem, and everything else still meets ``FileExistsError`` -- so a write into a
    directory nobody expected to be occupied stays a refusal rather than becoming a
    silent overwrite.

    Two attempts of one Batch job never run at the same time, so a directory that looks
    unfinished here is unfinished rather than in progress. That is a property of the retry
    and not of this function, and it is the assumption to check first if this is ever
    reused somewhere jobs overlap.

    Rank zero only, and once, because the save folder is one remote prefix that every rank
    shares. The barrier is what stops another rank listing the directory in the middle of
    the removal.
    """
    removed: List[str] = []
    if get_rank() == 0:
        for path in torn_step_directories(save_folder):
            log.warning(
                "%s is not a checkpoint the loader accepts, so an earlier attempt died "
                "writing it; clearing it so this attempt can write that step",
                path,
            )
            clear_directory(path)
            removed.append(path)
    barrier()
    return removed


#: Where on the run the held-out ladder evaluates, as fractions of ``--steps``.
#:
#: Geometric because loss falls roughly log-linearly in tokens, so evenly-spaced rungs would
#: cluster in the flat tail and carry the least information. 1.0 is absent deliberately:
#: ``eval_on_finish=True`` already scores the final step, and listing it would score it twice.
LADDER_FRACTIONS = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)


def ladder_steps(steps: int) -> List[int]:
    """The steps the held-out evaluator fires on, for a run of ``steps`` total.

    A named function rather than a comprehension inlined in ``build_config`` so that a test can
    call the same code the run calls. A test that re-derives the arithmetic instead is a test of
    its own copy: it stays green when the real fractions change, which is the failure it was
    written to prevent.

    The floor of 2 is not cosmetic. ``EvaluatorCallback.post_step`` returns early for
    ``step <= 1`` (train/callbacks/evaluator_callback.py:107-109), so a rung at step 1 would be
    silently skipped -- and a rung that never fires is indistinguishable from one that fired
    and showed no gap.
    """
    return sorted({max(2, int(steps * f)) for f in LADDER_FRACTIONS})


#: How many held-out shards the ladder scores, one per source, spread by
#: :func:`spread_across_sources`.
#:
#: RAISED FROM 4 TO 8 FOR COVERAGE, NOT FOR VOLUME, AND THE DISTINCTION IS THE WHOLE POINT.
#: Volume was never binding: a rung reads 32 batches x 32 instances x 2,048 = 2,097,152 tokens,
#: which is 8.39% of ONE 25,001,984-token shard. Four shards were already ~48x more material
#: than a rung consumes.
#:
#: What four shards could not do is reach the sources. `pretrain/reservoir-dolma2/v1` has 14
#: sources across 8 categories, so eight is the smallest count that can touch each category
#: once. Eight shards is ~800 MB of download against a rung that reads 2 MB -- the download is
#: the cost here, and it is one-time per run.
#:
#: ⚠️ SHARD COUNT ALONE DOES NOTHING. The evaluator breaks after `eval_duration` batches with
#: `shuffle=False` (train/callbacks/evaluator_callback.py:139-157, :392), so it reads instances
#: in ORDER and stops. One shard holds 12,208 instances at seq 2,048, so a 32-batch rung never
#: leaves the first shard unless the global eval batch exceeds 381 instances -- ours is 32.
#: Handing it more shards without interleaving them downloads material it never opens. That is
#: why `spread_across_sources` interleaves rather than concatenates, and why raising this
#: number is only half of a fix.
HELDOUT_SHARDS = 8


def spread_across_sources(urls: List[str], limit: int) -> List[str]:
    """Pick ``limit`` held-out shards, one per source, ORDERED so the first batches span sources.

    TWO SEPARATE DEFECTS, AND FIXING ONLY THE FIRST LOOKS LIKE A FIX WHILE CHANGING NOTHING.

    **Defect one: which shards get chosen.** Published shards are laid out
    ``tokens/<source>/val-NNNNN.u32le.bin``, so ``sorted()`` orders by source name and a
    ``[:limit]`` slice takes whatever the alphabetically-first source holds. On
    ``pretrain/reservoir-dolma2/v1`` -- 14 sources, 8 categories, 39 val shards --
    ``sorted()[:4]`` returns ``dclm/val-00000`` .. ``dclm/val-00003``: four shards, ONE source,
    one category of eight, 11.9% of train. Code (15.9%) and synthetic (23.8%) are never sampled.

    **Defect two: which shards get READ.** The evaluator breaks after ``eval_duration`` batches
    (evaluator_callback.py:155-157) over a loader built with ``shuffle=False`` (:392). It
    therefore consumes instances in list order and stops. A reservoir shard holds 12,208
    instances at sequence length 2,048 and a rung is 32 batches of 32 instances = 1,024, so the
    rung covers 8.39% of the FIRST shard and never opens the rest. Selecting eight
    well-spread shards and concatenating them per source would still measure only source one.

    So the ordering is load-bearing, not cosmetic: this returns at most one shard per source,
    ``source``-major and interleaved, so that consecutive batches walk across sources and a
    truncated rung still samples every one of them. With one shard per source and a rung far
    smaller than a shard, "interleaved" and "one each" coincide -- but the order is what makes
    the truncation harmless, so it is stated rather than left to luck.

    Neither defect fails loudly. Together they report a held-out CE computed on web text under a
    whole-corpus label, and they silently empty the domain-sliced secondary endpoint, which
    needs per-source coverage to mean anything at all. On a single-source corpus every ordering
    agrees, which is why this was invisible on ``regmix-10b`` and ``olmo-150b``.

    Deterministic -- sources sorted, shards within a source sorted, no RNG. That matters more
    than the coverage: every arm and every seed must score the SAME held-out material in the
    SAME order or the rungs are not comparable, which is the one thing the ladder cannot
    tolerate. Returns fewer than ``limit`` when the corpus has fewer sources than that, and
    falls back to the plain prefix on a corpus whose shards share one directory.
    """
    by_source: Dict[str, List[str]] = {}
    for url in sorted(urls):
        # The parent directory is the source. ``rsplit`` rather than ``Path`` because these are
        # ``s3://`` URLs and ``Path`` would collapse the ``//`` after the scheme.
        source = url.rsplit("/", 2)[-2] if url.count("/") >= 2 else ""
        by_source.setdefault(source, []).append(url)

    # Round-robin by DEPTH so the result is interleaved across sources rather than grouped by
    # them: with limit > len(by_source) this yields source A's first shard, then B's first, and
    # only later A's second. A grouped result would put every shard of A before any of B and
    # hand the truncation problem straight back.
    picked: List[str] = []
    for depth in range(max((len(v) for v in by_source.values()), default=0)):
        for source in sorted(by_source):
            if len(picked) >= limit:
                return picked
            if depth < len(by_source[source]):
                picked.append(by_source[source][depth])
    return picked


def _localised_heldout_paths(urls: List[str], opts) -> List[str]:
    """Download the held-out shards and return LOCAL paths.

    THIS IS WHY run_019fce60 DIED AT EXIT 70 WITH ``gzip.BadGzipFile: Not a gzipped file
    (b'5\\x00')``.

    ``iter_document_indices`` (data/utils.py:170-251 on this checkout; the branch is at
    :193-197 and the sidecar derivation at :217) picks between two strategies. For a LOCAL
    path with ``eos_token_id`` and ``dtype`` it scans the memmap for EOS boundaries -- fast and
    correct. For a URL it falls back to a sidecar metadata file whose name it derives as
    ``os.path.basename(data_path).replace(".npy", ".csv.gz")``.

    Our shards are ``val-00033.u32le.bin``. That ``replace`` matches nothing, so the "metadata
    file" it resolves is **the shard itself** -- which exists, so there is no FileNotFoundError
    to trigger the helpful message the code has ready -- and it gunzips raw uint32 tokens. Token
    53 is ``b"5\\x00\\x00\\x00"``; those are the bytes in the error.

    The training path never hit this: plain ``NumpyFSLDataset`` does not call
    ``segment_documents_into_instances`` at all. Only ``NumpyPaddedFSLDataset`` (what the
    evaluator requires) and ``NumpyFSLDatasetMixture`` do, so the bug was unreachable until the
    ladder was wired.

    Fixed here rather than in ``data/utils.py``: that ``.replace(".npy", ...)`` is correct for
    Dolma-toolkit corpora that really do ship ``.csv.gz`` sidecars, and changing it would alter
    behaviour for every dataset in the library to suit one experiment's shards.
    """
    from olmo_core.io import get_file_size, is_url

    cache = Path(opts.work_dir) / "heldout-shards"
    cache.mkdir(parents=True, exist_ok=True)

    fetching = _may_fetch_heldout_shards()

    out: List[str] = []
    for url in urls:
        if not is_url(str(url)):
            out.append(str(url))
            continue
        # PREFIXED WITH THE SOURCE DIRECTORY, BECAUSE BASENAMES COLLIDE ACROSS SOURCES.
        # Shards are `tokens/<source>/val-NNNNN.u32le.bin` and the numbering restarts per
        # source, so `dclm/val-00000.u32le.bin` and `finemath/val-00000.u32le.bin` share a
        # basename. Caching on the basename alone would have the second download overwrite the
        # first, silently, and the evaluator would then score one source twice under two labels
        # -- a per-source metric that is confidently wrong rather than absent.
        #
        # Unreachable before `spread_across_sources` existed, because `sorted()[:N]` only ever
        # returned shards from ONE source, where basenames are unique by construction. Fixing
        # the selection is what made this reachable, so it is fixed in the same change.
        source = str(url).rsplit("/", 2)[-2] if str(url).count("/") >= 2 else ""
        dest = (
            cache / f"{source}--{os.path.basename(str(url))}"
            if source
            else cache / os.path.basename(str(url))
        )
        if fetching:
            # Compare SIZE, not existence: a truncated download left by a killed attempt would
            # otherwise be reused, and a short shard yields wrong document boundaries rather
            # than an error. `get_file_size` is the S3 HEAD, and it is issued on the fetching
            # process only -- see _may_fetch_heldout_shards for why the other seven must not.
            if not dest.is_file() or dest.stat().st_size != get_file_size(url):
                log.info("downloading held-out shard %s", url)
                _download_to(str(url), dest)
        out.append(str(dest))
    return out


#: The env var torchrun sets on every worker before any process group exists.
#: ``olmo_core.distributed.utils.get_local_rank`` reads this same name, but it returns 0
#: unconditionally until ``dist.is_initialized()``, which is why this file reads the variable
#: directly rather than calling that helper. See ``_may_fetch_heldout_shards``.
LOCAL_RANK_ENV_VAR = "LOCAL_RANK"


def _may_fetch_heldout_shards() -> bool:
    """Whether THIS process is the one allowed to HEAD and download the held-out shards.

    WHY THIS CANNOT USE ``get_rank()`` OR ``get_fs_local_rank()``, WHICH IS THE WHOLE TRAP.
    ``build_config`` is called at main():1222, BEFORE ``prepare_training_environment()`` at
    :1237. At that moment ``dist.is_initialized()`` is False in every worker, and both helpers
    short-circuit to 0 when ``is_distributed()`` is False
    (distributed/utils.py:249-256 and :301-307). Gating on either would return True on all
    eight ranks and change nothing at all -- a fix that reads correct and does nothing.

    ``LOCAL_RANK`` is set by torchrun in the environment of every worker at spawn time, so it
    is readable before the process group exists. That is the only rank signal available this
    early.

    WHAT THE GATE IS FOR. The size comparison in ``_localised_heldout_paths`` IS the cache
    condition, so the ``get_file_size`` HEAD is issued on every call even when the shard is
    already on disk. ``get_file_size`` is decorated ``@maybe_cache(condition=is_url)``
    (io.py:107), and ``maybe_cache`` disables caching entirely unless ``OLMO_CORE_FS_CACHE_DIR``
    is set (fs_cache.py:34-38) -- nothing in ``.edullm/`` sets it. Ungated, an 8-rank run
    therefore issues 8 x N S3 HEADs at config-build time, and a throttle or a credential gap on
    any single rank raises inside ``during(THE_CONFIG_WOULD_NOT_BUILD)`` and dies at exit 70 --
    indistinguishable, from the outside, from the sidecar bug this port exists to fix.

    It also removes a write race. ``_download_to`` writes one fixed ``dest + ".part"`` and then
    renames it, with no per-rank suffix, so eight ranks meeting an absent shard would write the
    same temporary path and rename it underneath each other.

    A single-process invocation -- ``--prepare-heldout-only``, a ``--dry-run``, or someone
    running this by hand -- has no ``LOCAL_RANK`` set, so it fetches. That is deliberate: the
    prepare-only step is exactly the one that must do the real size-verified download.

    ``LOCAL_RANK`` is per-NODE, so on a multi-node job one process per node fetches into that
    node's own ``--work-dir``. That is the behaviour we want and it matches what
    ``get_fs_local_rank`` means by "local rank per filesystem"; on one 8-GPU box the eight
    workers share ``/tmp`` and read the single copy local rank 0 fetched.

    THE RESIDUAL, STATED RATHER THAN HIDDEN. This assumes the documented contract holds: the
    ``--prepare-heldout-only`` invocation runs first, so by training time the shards are
    already on disk and the gate only suppresses a redundant HEAD. If that step is skipped AND
    the shards are absent, the non-fetching workers now return a path that does not exist yet
    and fail loudly on read, where previously they would have raced the same ``.part`` file and
    could have read a torn one. Loud beats silent -- a torn shard yields wrong document
    boundaries and a plausible-looking eval number -- but it is a real behaviour change and the
    prepare step is what keeps it off the happy path.
    """
    return os.environ.get(LOCAL_RANK_ENV_VAR, "0") == "0"


def _download_to(url: str, dest: Path) -> None:
    """Copy one object to a local file, writing via a .part file so a kill cannot look complete.

    The ``.part`` name carries the pid so that two processes which do reach this concurrently
    cannot write the same temporary file. ``_may_fetch_heldout_shards`` should already prevent
    that, but ``rename`` is only atomic per source, and a torn shard is silent.
    """
    from urllib.parse import urlparse

    import boto3

    parsed = urlparse(url)
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.part")
    boto3.client("s3").download_file(parsed.netloc, parsed.path.lstrip("/"), str(tmp))
    tmp.rename(dest)


def prepare_heldout_indices(config) -> None:
    """Build the held-out instance indices, in ONE process, outside any distributed context.

    Run as its own command before ``torchrun`` via ``--prepare-heldout-only``. Deliberately NOT
    called from ``train()``: that is the whole fix.

    TWO RUNS DIED LEARNING WHY THIS CANNOT LIVE INSIDE THE DISTRIBUTED PROGRAM. ``run_019fca21``
    and ``run_019fcdd1`` both hit exit 72 after a 900-second gloo timeout -- ~$11 for zero
    measurements. ``NumpyPaddedFSLDataset.prepare()`` (numpy_dataset.py:911-916 on this
    checkout) writes indices on ``fs_local_rank`` 0 only, inside a bare
    ``ProcessPoolExecutor()`` (:951), and then every rank meets a ``barrier()``.

    ``ProcessPoolExecutor()`` with no ``max_workers`` uses ``os.cpu_count()``, which on a
    p4d.24xlarge is **96**. The start method is already forced to ``"spawn"``
    (train/__init__.py), so that is 96 fresh interpreters each importing torch and olmo_core,
    launched from one rank while seven others hold CUDA contexts and NCCL communicators on the
    same box. Rank 0 logged all four ``Gathering instance indices`` lines within 0.16 s --
    submitting futures is instant -- then never logged a single ``Created N instances``, so not
    one future returned.

    The first fix attempt only moved the call earlier in ``train()``, on the theory that a live
    CUDA context was the problem. It failed identically with the traceback inside the helper.
    A four-rank local reproduction on ten cores does NOT deadlock, which is why that theory
    survived review: the pool is only pathological when ``cpu_count()`` is large.

    Standalone there is no process group, so the internal ``barrier()`` is a no-op and the pool
    has the machine to itself. ``_write_instance_indices`` then skips any path whose indices
    file already exists (numpy_dataset.py:944-949), so when
    ``LMEvaluatorCallbackConfig.build()`` calls ``prepare()`` during the real run,
    ``paths_needed`` is empty and no pool is created at all.

    **The two invocations must share ``--work-dir``** or the second finds nothing cached and
    re-enters the deadlock.
    """
    eval_cfg = getattr(config.trainer, "callbacks", {}).get("lm_eval")
    if eval_cfg is None:
        log.warning(
            "no held-out ladder in this config, so there are no indices to prepare; the run "
            "will train but a flat result will not be distinguishable from undertraining"
        )
        return
    dataset = eval_cfg.eval_dataset.build()
    log.info("preparing held-out indices for %d shard(s) in one process", len(dataset.paths))
    dataset.prepare()
    log.info("held-out indices ready; the eval callback will find them cached")


#: Terminal LR as a fraction of peak, per schedule. Zero for linear: this is the whole point
#: of E1 and the reason the schedule became selectable rather than being swapped in place.
#: OLMo-core's own default for BOTH classes is ``alpha_f=0.1`` (scheduler.py:355 for
#: LinearWithWarmup, :459 for CosWithWarmup), i.e. the LR stops at a tenth of peak and the
#: last tokens are trained at a rate the schedule never intended to end on.
SCHEDULE_ALPHA_F = {"linear": 0.0, "cosine": 0.1}


def build_scheduler(opts):
    """The LR schedule, selected rather than hardcoded, with warmup as a fraction of steps.

    WHY THIS IS A FLAG. Decay-to-zero is worth ~0.025 nats over cosine-to-10% (Bergsma
    2502.15938 Table 1: 2.591 -> 2.571, measured at 610M/12.1B and again at 1.7B/34.3B, which
    brackets our 1B/40B cell on both axes). But the optimal peak LR for decay-to-zero sits
    about one doubling above the optimum for cosine-to-10%, so swapping the schedule while
    holding the LR fixed measures the LR mismatch and not the schedule. E1 exists to sweep the
    two together, and it can only do that if both are reachable from the command line.

    WHY ``warmup_fraction`` AND NOT A COMPUTED STEP COUNT. Both scheduler classes take it
    natively (scheduler.py:359 and :458) and resolve it against the trainer's real horizon at
    every step -- ``warmup = round(t_max * warmup_fraction)`` inside ``get_lr``. Computing
    ``round(0.1 * opts.steps)`` here would produce the same number today and the wrong one the
    moment anything overrides ``trainer.max_duration`` on the command line, because this
    function cannot see that override. Note the classes REFUSE both at once: ``__post_init__``
    raises unless exactly one of ``warmup`` / ``warmup_fraction`` is set, so an explicit
    ``--warmup-steps`` has to suppress the fraction rather than sit beside it.
    """
    alpha_f = SCHEDULE_ALPHA_F[opts.lr_schedule]
    cls = LinearWithWarmup if opts.lr_schedule == "linear" else CosWithWarmup

    if opts.warmup_steps is not None:
        # `warmup`, not the `warmup_steps` the example still passes -- that spelling is
        # deprecated upstream and warns on every construction.
        return cls(warmup=opts.warmup_steps, alpha_f=alpha_f)
    return cls(warmup_fraction=opts.warmup_fraction, alpha_f=alpha_f)


def build_config(opts, overrides: List[str]):
    corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    log.info(
        "%s/%s: %d shards, dtype %s, tokenizer %s",
        corpus.dataset_id,
        corpus.version,
        len(corpus.paths),
        corpus.dtype,
        opts.dataset_tokenizer,
    )

    factory = getattr(TransformerConfig, opts.model_factory, None)
    if factory is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD, f"unknown model factory: {opts.model_factory}"
        )

    # padded rather than exact for the same reason the example pads: a vocab that is a
    # multiple of 128 keeps the embedding matmul on a fast path. dolma2's 100,278 pads to
    # 100,352.
    model_config = factory(vocab_size=corpus.tokenizer.padded_vocab_size())

    dataset_config = NumpyFSLDatasetConfig(
        paths=corpus.paths,
        sequence_length=opts.sequence_length,
        tokenizer=corpus.tokenizer,
        # The whole point of this file. See the header.
        dtype=corpus.dtype,
        work_dir=opts.work_dir,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.learning_rate,
            # AdamWConfig's own defaults are betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
            # (adamw.py:241-243) and nothing here used to override them. All three are set
            # explicitly now -- see the flag help in build_parser for the source of each --
            # and they are set explicitly rather than left to the library so that a future
            # upstream change to those defaults cannot silently move this baseline.
            betas=(opts.beta1, opts.beta2),
            eps=opts.adam_eps,
            weight_decay=opts.weight_decay,
            group_overrides=[
                # DO NOT REMOVE. This exempts the embedding matrix from weight decay, and it
                # keeps working with weight_decay set above: build_groups (optim/config.py:118)
                # pulls the matched FQNs into their own param group and splats `opts` over it,
                # so this group carries weight_decay=0.0 while every other parameter takes the
                # config-level value. Raising weight_decay does not reach the embeddings.
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # On, because the image now carries a C compiler. It was off in the platform's
        # getting-started command only because a run without one dies on the first compiled
        # region, which is a workaround that costs throughput on every run forever.
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=1.0,
        scheduler=build_scheduler(opts),
        # Z-loss is a field on the TRAIN MODULE, not on the model and not on the trainer. The
        # path is TransformerTrainModuleConfig.z_loss_multiplier (train_module/transformer/
        # config.py:343) -> TransformerTrainModule.__init__ -> self.z_loss_multiplier ->
        # model_forward(z_loss_multiplier=...) -> LMHead.forward (nn/lm_head.py:208), where
        # `compute_z_loss=z_loss_multiplier is not None` is what actually switches it on. Left
        # at None it is OFF -- the `or 1e-4` on lm_head.py:261 is a floor for a multiplier that
        # was requested, not a default that fires. Setting it here also turns on the
        # "train/Z loss" metric (train_module.py:445-455), which is how a run proves it is on.
        #
        # `or None` so that `--z-loss-multiplier 0` means OFF rather than on-with-a-zero-
        # coefficient. The latter would add nothing to the loss but would still pay for the
        # z-loss computation and would divide by it at train_module.py:454.
        z_loss_multiplier=opts.z_loss_multiplier or None,
    )

    trainer_config = (
        TrainerConfig(
            save_folder=opts.save_folder,
            # save_overwrite is false, unlike the example. The save folder here is a per-run
            # S3 prefix that a Batch retry re-derives identically, and overwriting is exactly
            # what must not happen when the second attempt is meant to resume the first.
            #
            # It stays false now that a retry can meet its own unfinished step directory.
            # True would clear whatever is at the target step before every save and would
            # overwrite every object unconditionally, which reaches finished checkpoints as
            # well as torn ones. remove_torn_checkpoints does the narrow thing instead, and
            # leaving this false is what keeps the refusal for the case it did not expect.
            save_overwrite=False,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(opts.steps),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=opts.save_interval,
                # None rather than a number: OLMo-core refuses a config whose ephemeral
                # interval is not below save_interval, and it refuses it in the first seconds
                # rather than at the first save.
                ephemeral_save_interval=None,
                # KEEP EVERY CHECKPOINT, BECAUSE THE ROLE CANNOT PRUNE ONE. The default is 3
                # and the rest are pruned. _remove_checkpoint deletes the directory's
                # .metadata.json first, to invalidate the checkpoint before clearing it, and
                # the workload role is denied that one key by name -- so a prune fails on its
                # first call and the refusal reaches Trainer.fit rather than being swallowed.
                # Left at the default, the fourth save kills an eleven-hour run.
                #
                # The role does hold s3:DeleteObject under checkpoints/, which is what
                # remove_torn_checkpoints needs. A torn directory never contains
                # .metadata.json, since that object is written last and its presence is what
                # makes the directory complete, so the deny bounds the prune without
                # reaching the repair.
                max_checkpoints=None,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=opts.run_name,
                project=os.environ.get("EDULLM_WANDB_PROJECT"),
                # No `group`. The platform puts the experiment in WANDB_RUN_GROUP, which the
                # wandb client reads on its own; passing it again from an environment variable
                # that does not exist would set it to None and look deliberate.
                cancel_check_interval=10,
                # Enabled only when the platform named a project, so running this image by
                # hand does not fail on a missing WANDB_API_KEY.
                enabled=bool(os.environ.get("EDULLM_WANDB_PROJECT")),
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
    )

    # NO downstream_evaluator, AND THE EXAMPLE'S LM EVALUATOR IS STILL REFUSED. The example
    # reads a C4 validation shard from olmo-data.org and pulls HellaSwag from Hugging Face;
    # both would put a public-internet fetch in the middle of a run whose whole claim is that
    # it read a sealed corpus, and a failure in either would look like a training failure.
    # That objection is about the EXAMPLE'S data source, not about held-out evaluation, and it
    # is exactly why the evaluator below is wired to the corpus's own `.val` instead: those
    # shards come from the same sealed manifest the training shards do, resolved by the same
    # `dataset_paths()` call, and reading them adds no source this run was not already given.
    #
    # AN EARLIER VERSION OF THIS COMMENT ENDED "it needs a corpus that declares one, which
    # regmix-10b does not." THAT WAS FALSE, and it was the stated reason no evaluator existed.
    # `pretrain/regmix-10b/v1`'s dataset.json declares
    # {"name":"val","glob":"val-*.u32le.bin","rows":15007207} and seven val objects exist under
    # tokens/<source>/ (verified live against s3://edullm-data on 2026-08-04; see
    # docs/1b-leverage-audit/grounding/val-split-status.md sections B and D). Eleven of the
    # sixteen registered releases carry a val split, and `val` is REQUIRED for the `pretrain`
    # family (edullm-data/families/pretrain.json:52, enforced at validate.py:1157). The reader
    # hands the paths over for free on the call resolve_corpus already makes; this file used
    # to drop them on the floor.
    #
    # The endpoint is CE loss and PPL, not bits-per-byte: LMEvaluator emits exactly
    # `{label}/CE loss` and `{label}/PPL` (eval/lm_evaluator.py:118-121), and no manifest
    # carries a UTF-8 byte denominator -- a .u32le.bin shard's `bytes` is 4x its tokens, the
    # storage width.
    if corpus.val_paths:
        eval_steps = ladder_steps(opts.steps)
        # LOCAL paths, and this is load-bearing rather than an optimisation. `iter_document
        # _indices` only scans the array for EOS boundaries when the path is NOT a url
        # (data/utils.py:193-197); for an s3:// path it looks for a sidecar metadata file whose
        # name it derives by `basename.replace(".npy", ".csv.gz")` (:217). Our shards end
        # `.u32le.bin`, so that replace is a no-op, the "metadata file" it resolves is the
        # shard itself, and it gunzips raw uint32 tokens -- `BadGzipFile: Not a gzipped file
        # (b'5\x00')`, which killed run_019fce60 at exit 70.
        #
        # Resolved ONCE here, so the prepare-only invocation and the eval callback see the same
        # strings. That matters more than it looks: `_get_indices_path`
        # (data/numpy_dataset.py:433-448) names the cache file after a SHA-256 of
        # `str(source_path)`, so an s3:// path and its local copy hash to different files --
        # localising in only one of the two places would silently miss the cache and walk
        # straight back into the gzip failure.
        # Spread over sources rather than `sorted()[:N]`, which on a multi-source corpus takes
        # every shard from the alphabetically-first source. See `spread_across_sources`.
        #
        # The REMOTE urls are kept, because they are the only place the source name survives:
        # `_localised_heldout_paths` writes `<work-dir>/heldout-shards/<basename>`, which
        # flattens `tokens/<source>/val-NNNNN.u32le.bin` down to `val-NNNNN.u32le.bin`. So the
        # per-source labels below must be derived here, before localisation, and NOT from the
        # local paths -- every local path would report the same parent directory.
        heldout_urls = spread_across_sources(corpus.val_paths, HELDOUT_SHARDS)
        heldout_labels = [
            url.rsplit("/", 2)[-2] if url.count("/") >= 2 else "heldout-val" for url in heldout_urls
        ]
        heldout_paths = _localised_heldout_paths(heldout_urls, opts)
        trainer_config = trainer_config.with_callback(
            "lm_eval",
            LMEvaluatorCallbackConfig(
                # PADDED, not the NumpyFSLDatasetConfig the training path uses. The callback
                # type-checks for NumpyPaddedFSLDataset and raises OLMoConfigurationError on
                # anything else (train/callbacks/evaluator_callback.py:268-272), so the plain
                # config fails at build time.
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    # A handful of shards, not all of them. `prepare()` builds a per-shard
                    # instance index over every path with a process pool on first call, and a
                    # corpus's worth of startup would cost more than the eval it serves.
                    # Sorted so the subset is the same across arms and seeds -- a per-cell
                    # subset would make the rungs incomparable, which is the one thing the
                    # ladder cannot tolerate.
                    paths=heldout_paths,
                    # ONE LABEL PER SOURCE, NOT ONE LABEL FOR ALL OF THEM, AND THIS IS WHAT
                    # MAKES THE COVERAGE LIMIT VISIBLE INSTEAD OF SILENT.
                    #
                    # `LMEvaluator.from_numpy_dataset` raises when any path lacks a "label"
                    # (eval/lm_evaluator.py:60-66) and keys a separate metric on each distinct
                    # one. A single shared label therefore folds every source into one number,
                    # which is exactly the number that cannot tell you whether a source was
                    # read at all: `heldout-val/CE loss` looks equally healthy whether the rung
                    # covered eight sources or truncated after the first.
                    #
                    # That matters because the rung DOES truncate. `eval_duration` is 32 batches
                    # over a `shuffle=False` loader, which is 8.39% of one 12,208-instance
                    # shard, so sources after the first are not reached. Covering all eight
                    # would need ~3,052 batches per rung -- ~83 min of 8xA100 across the seven
                    # rungs, roughly $30 a run and more than the training it measures.
                    #
                    # So do not pay for it: name each source instead. An unread source reports
                    # `<source>/CE loss` = NaN, because `MeanMetric.compute` is
                    # `weighted_sum / weight` = 0/0 for an untouched label (eval/metrics.py:81-89)
                    # -- loud in a plot and impossible to mistake for a measurement. And the
                    # sources that ARE read now give per-source CE for free, which is the
                    # domain-sliced secondary endpoint the study wanted and previously could not
                    # compute at all.
                    #
                    # The label is the source directory, so it is stable across arms and seeds
                    # and reads the same in W&B as in the corpus manifest. Derived from the
                    # REMOTE urls above, because localisation flattens the source directory away.
                    metadata=[{"label": label} for label in heldout_labels],
                    sequence_length=opts.sequence_length,
                    # From the corpus, never pinned. The padded vocab the model is built with
                    # comes off this same object.
                    tokenizer=corpus.tokenizer,
                    # Same uint32 trap as the training path: the corpus declares its width and
                    # a default here would decode every token at the wrong one, because
                    # get_dtype() falls back to the NARROWEST dtype the vocab fits in.
                    dtype=corpus.dtype,
                    # SAME work dir as the prepare-only invocation, or the cached indices are
                    # not found and the 96-worker pool opens inside the distributed program.
                    work_dir=opts.work_dir,
                ),
                # None, so only `fixed_steps` and `eval_on_finish` trigger an eval. A non-None
                # interval would add unrequested rungs and change what each cell costs.
                eval_interval=None,
                fixed_steps=eval_steps,
                eval_on_finish=True,
                # Bounded by STEPS, not the default epochs(1). The default would score every
                # shard in full at every rung, which costs more than the training it measures.
                eval_duration=Duration.steps(32),
            ),
        )
        log.info(
            "held-out ladder at steps %s (from %d val shards)", eval_steps, len(corpus.val_paths)
        )
    else:
        # Not fatal, but it must not pass silently: without a ladder the run still trains and
        # still reports a loss, and the missing endpoint is invisible until analysis.
        log.warning(
            "%s/%s declares NO held-out split, so this run has no ladder and a flat result "
            "will not be distinguishable from undertraining",
            corpus.dataset_id,
            corpus.version,
        )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
    )
    return config.merge(overrides)


class LossWatcher(Callback):
    """Keeps what the summary can only learn while the run is still going.

    The W&B url is read here rather than in ``summarise`` because ``WandBCallback.post_train``
    finishes the run, after which ``wandb.run`` is None. Read on a metrics callback rather
    than in ``pre_train``, because callbacks of equal priority run in reverse registration
    order and this one is registered last, so ``pre_train`` here happens before W&B has a run
    to name.
    """

    def __init__(self) -> None:
        self.first: Optional[float] = None
        self.last: Optional[float] = None
        self.wandb_url = ""

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        del step
        if not self.wandb_url:
            with contextlib.suppress(Exception):
                import wandb

                self.wandb_url = getattr(wandb.run, "url", "") or ""
        loss = metrics.get("train/CE loss")
        if loss is None:
            return
        if self.first is None:
            self.first = float(loss)
        self.last = float(loss)


def summarise(*, opts, config, trainer, losses: LossWatcher, seconds: float) -> None:
    """Print what only this process can report, as one JSON object on stdout.

    The platform reads this back out of the log stream: the device torch actually got, the
    parameter count, the loss at both ends and where the checkpoints went are not facts Batch
    holds. Printed on rank zero only, and printed whatever the losses are, because a run that
    reported nothing is indistinguishable from one that never started.
    """
    if get_rank() != 0:
        return
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(
        json.dumps(
            {
                "run_id": opts.run_name,
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                "gpu": device,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "parameters": sum(
                    parameter.numel() for parameter in trainer.train_module.model.parameters()
                ),
                "steps": trainer.global_step,
                "first_loss": losses.first,
                "last_loss": losses.last,
                "seconds": seconds,
                "peak_memory_gib": peak,
                "checkpoint_uri": opts.save_folder,
                "wandb_project": os.environ.get("EDULLM_WANDB_PROJECT", ""),
                "wandb_url": losses.wandb_url,
            },
            indent=2,
        ),
        flush=True,
    )


def show(config) -> None:
    """Print the config with the shard list replaced by its length.

    olmo-150b-dolma2 resolves to 6,851 objects, and printing each one buries every other
    line of the config -- including the dtype and the tokenizer, which are the two fields
    worth reading. The paths themselves are in the config the ConfigSaverCallback writes
    next to the checkpoints.
    """
    shown = copy.copy(config.dataset)
    shown.paths = [f"<{len(config.dataset.paths)} objects>"]
    rich.print(replace(config, dataset=shown))


def train(config, opts=None) -> None:
    if get_rank() == 0:
        show(config)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    losses = LossWatcher()
    trainer.add_callback("edullm_losses", losses)

    # Before the load rather than after it, so that the state of the save folder the loader
    # reads is the state this attempt is going to write into. Either order resumes from the
    # same step -- the loader skips a torn directory on its own -- and doing it first means
    # the log says what was cleared before it says what was loaded.
    remove_torn_checkpoints(trainer.save_folder)

    # maybe_load_checkpoint is what makes a second Batch attempt continue the first rather
    # than start over. It looks in the save folder, which is EDULLM_CHECKPOINT_DIR, which is
    # derived from the run id and is therefore the same string on both attempts.
    trainer.maybe_load_checkpoint()
    started = time.monotonic()
    trainer.fit()
    if opts is not None:
        summarise(
            opts=opts,
            config=config,
            trainer=trainer,
            losses=losses,
            seconds=time.monotonic() - started,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_on_corpus",
        description="Train a transformer on a published eduLLM corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("run_name", nargs="?", default=os.environ.get("EDULLM_RUN_ID", "local"))
    parser.add_argument("--dataset-id", default=os.environ.get("EDULLM_DATASET_ID", ""))
    parser.add_argument("--dataset-version", default=os.environ.get("EDULLM_DATASET_VERSION", ""))
    parser.add_argument(
        "--dataset-tokenizer", default=os.environ.get("EDULLM_DATASET_TOKENIZER", "")
    )
    parser.add_argument(
        "--save-folder",
        default=os.environ.get("EDULLM_CHECKPOINT_DIR", ""),
        help="Where checkpoints go. The platform sets EDULLM_CHECKPOINT_DIR to a per-run "
        "prefix; a run that writes anywhere else cannot be resumed by its own retry.",
    )
    parser.add_argument("--work-dir", default="/tmp/dataset-cache")
    parser.add_argument("--model-factory", default="olmo2_190M")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--lr-schedule",
        choices=sorted(SCHEDULE_ALPHA_F),
        default="linear",
        help="linear decays to zero (alpha_f=0.0); cosine stops at 10%% of peak "
        "(alpha_f=0.1, OLMo-core's default for both classes). Decay-to-zero is worth "
        "~0.025 nats -- Bergsma 2502.15938 Tbl 1, 2.591 -> 2.571 at 610M/12.1B and again "
        "at 1.7B/34.3B. E1 sweeps this against --learning-rate, because the two do not "
        "move independently.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Absolute warmup length, overriding --warmup-fraction. Unset by default: this "
        "used to default to 20, which is 0.013%% of a 40B-token run -- a smoke-test value "
        "that reached the baseline. Wen et al.'s tuned grid uses ~1000.",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.1,
        help="Warmup as a fraction of the run, resolved by the scheduler against the "
        "trainer's real horizon rather than computed here. Ignored when --warmup-steps "
        "is given, because the scheduler classes refuse to accept both.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.4e-3,
        help="Peak LR. Raised from 1e-3, which was transferred from Wen's tuned 1.2B "
        "(2e-3 at a 1.05M batch) by sqrt-scaling to the OLD 262,144 batch. It co-moves "
        "with --global-batch-size, and decay-to-zero's optimum sits ~1 doubling above "
        "cosine-to-10%%'s. E1 sweeps it; this is the centre of that sweep, not a measured "
        "optimum.",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=786432,
        help="Tokens per optimizer step. 786432 = 192 x 4096. Power Lines (2505.13738) "
        "fits B_opt = 0.0306*D^0.383 (B in 2048-token sequences), giving 0.721M tokens at "
        "D=40e9; the old 262,144 was 2.75x BELOW that, on the wrong side of a loss bowl "
        "whose interior minimum is measured in their Table 1. Also what the completed "
        "474M/10B run used.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.07,
        help="AdamW lambda, was the library default 1e-2 (adamw.py:243). Power Lines' "
        "tau_opt = 1.084*TPP^-0.527 (R^2=0.975) at the NEW batch and peak LR. Embeddings "
        "stay exempt via the group override -- this does not reach them.",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="AdamW beta1. Library default, unchanged."
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="AdamW beta2, was the library default 0.999 (adamw.py:241). Wen et al.'s tuned "
        "runs use 0.98; train_liv_arm.py:661-662 already sets 0.95 here with a comment "
        "recording that inheriting 0.999 was a bug.",
    )
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-10,
        help="AdamW eps, was the library default 1e-8 (adamw.py:242). Wen uses 1e-10. Worth "
        "about nothing (their own span is 0.002, below their significance threshold) and "
        "costs nothing.",
    )
    parser.add_argument(
        "--z-loss-multiplier",
        type=float,
        default=1e-5,
        help="Z-loss coefficient, was never passed at all -- the feature is OFF when this is "
        "unset (lm_head.py:260 gates on `is not None`). Standard logit-norm stabiliser at "
        "this scale. Pass 0 to disable.",
    )
    parser.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, do not train.")
    parser.add_argument(
        "--prepare-heldout-only",
        action="store_true",
        help="Build the held-out eval indices and exit, without training and without starting a "
        "process group. Run this ONCE, single-process, before torchrun, sharing the same "
        "--work-dir: NumpyPaddedFSLDataset.prepare() opens a 96-worker spawn pool behind a "
        "collective, which deadlocked two 8-GPU runs at a 900s gloo timeout when it ran inside "
        "the distributed program.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()

    missing = [
        name
        for name, value in (
            ("EDULLM_DATASET_ID", opts.dataset_id),
            ("EDULLM_DATASET_VERSION", opts.dataset_version),
            ("EDULLM_DATASET_TOKENIZER", opts.dataset_tokenizer),
            ("EDULLM_CHECKPOINT_DIR", opts.save_folder),
        )
        if not value
    ]
    if missing:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "the platform sets these and they are unset: "
            + ", ".join(missing)
            + ". Submitting with dataset_release: none leaves the first three empty, which "
            "means this run has no corpus to open.",
        )

    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        config = build_config(opts, overrides)
    if opts.dry_run:
        show(config)
        return

    # BEFORE prepare_training_environment(), and that placement is the fix rather than a
    # detail. No process group exists yet, so the barrier inside `prepare()` is a no-op and
    # cannot strand seven peers, and the 96-worker pool has the box to itself instead of
    # competing with eight ranks holding CUDA contexts. See prepare_heldout_indices.
    if opts.prepare_heldout_only:
        with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
            prepare_heldout_indices(config)
        return

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        # SIXTY MINUTES, NOT THE FIFTEEN THIS TOOK BY DEFAULT, AND IT IS A SECOND LINE OF DEFENCE
        # RATHER THAN THE FIX. `prepare_training_environment` defaults `timeout` to 15 minutes
        # (train/__init__.py:89) and passes it to `init_process_group`, so 15 minutes is the bound
        # on every collective issued on the DEFAULT group -- including the teardown barrier in
        # `Trainer._shutdown`. `run_019fd3eb` died there:
        #   RuntimeError: [gloo/.../unbound_buffer.cc:78]
        #     Timed out waiting 900000ms for recv operation to complete
        # after all 1525 steps and all 6 eval rungs had already succeeded. 900000ms is exactly this
        # default, which is how that barrier was identified: the bookkeeping and checkpointing
        # groups both carry 30 minutes.
        #
        # The actual repair is the barrier ordering in `Trainer._shutdown` -- rank zero's
        # `wandb.finish()` no longer runs while seven peers hold a collective. This widening is
        # kept anyway because it is one argument and it covers the collectives that ordering fix
        # does NOT touch: `_prepare_dir` at the head of a synchronous checkpoint save, and the
        # data loader's reshuffle barrier inside the finish eval. Both are on the default group,
        # both are symmetric across ranks, and both sit behind an S3 round trip whose tail latency
        # nothing here bounds. Fifteen minutes is a tight bound for a 1B-parameter shard upload
        # from eight ranks; sixty is not generous, it is merely not tight.
        #
        # The cost of a too-large value is a hang taking longer to surface. The cost of a
        # too-small one is a completed run marked FAILED with its final checkpoint lost, which is
        # what happened, and it is the worse failure by a wide margin.
        prepare_training_environment(timeout=timedelta(minutes=60))
    try:
        with during(Stage.TRAINING_ITSELF_FAILED):
            train(config, opts)
    finally:
        teardown_training_environment()


def cli() -> int:
    """Run, and turn a refusal into a number a person on the platform side can actually see.

    ``main`` raises rather than exiting so that a test can read the message. This is the
    boundary where a stage becomes the process's exit status, the explanation goes to stderr
    for whoever can read the log, and the same explanation goes to W&B for everyone who
    cannot.
    """
    try:
        main()
    except Refusal as refusal:
        print(refusal.explanation, file=sys.stderr)
        # Machine-readable and greppable, for the case where somebody does have the log.
        print(f"edullm-stage: {refusal.stage.name} exit={int(refusal.stage)}", file=sys.stderr)
        if refusal.__cause__ is not None:
            traceback.print_exception(
                type(refusal.__cause__), refusal.__cause__, refusal.__cause__.__traceback__
            )
        leave_the_reason_in_wandb(
            run_name=os.environ.get("EDULLM_RUN_ID", "local"),
            stage=refusal.stage,
            explanation=refusal.explanation,
        )
        return int(refusal.stage)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
