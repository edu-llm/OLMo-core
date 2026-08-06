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
import math
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, cast

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


def _domain_of(url: str) -> str:
    """The held-out shard's topic domain: the name of its immediate parent directory.

    ``s3://.../tokens/all-dressed-snazzy2/adult_content/val-00033.u32le.bin`` -> ``"adult_content"``
    -- confirmed against a live object, ``edullm-data/HANDOFF.md:614-615``:
    ``tokens/all-dressed-snazzy2/adult_content/val-00033.u32le.bin`` carries
    ``labels={'domain': 'adult_content', 'source': 'all-dressed-snazzy2'}``.

    Falls back to the literal ``"heldout-val"`` -- the ORIGINAL single label this file used
    before this fix -- for a shard with no directory structure to read a domain from (fewer
    than two ``/`` in the path). That keeps a flat, single-directory corpus behaving exactly as
    it did before: one label, everything grouped together, nothing here to spread.

    One function, used for THREE things below (grouping in ``spread_across_sources``, the
    per-path label, and the localised-shard cache key) rather than three separate derivations.
    A second copy of this rule is exactly how Defect B happened upstream: the selection and the
    label were changed in one place and the metric key that has to agree with the label lived
    somewhere else, so the two drifted apart silently. One function can still be wrong, but it
    cannot disagree with itself.
    """
    return url.rsplit("/", 2)[-2] if url.count("/") >= 2 else "heldout-val"


def spread_across_sources(urls: List[str], limit: int) -> List[str]:
    """Pick up to ``limit`` held-out shards, spread across topic domains rather than one
    domain exhausted before the next is touched.

    THE FAILURE THIS REPLACES. ``sorted(corpus.val_paths)[:HELDOUT_SHARDS]`` sorts by the
    FULL url, so shards group by directory before they group by shard number: every path
    under ``.../adult_content/`` sorts before every path under ``.../art_and_design/``,
    regardless of shard number, because string comparison never gets past the directory name
    to look at what follows it. Which domains a plain prefix touches, and in what proportion,
    is then an accident of the alphabet and of how many val shards that one domain happens to
    have -- not a property anyone chose, and not one that survives the corpus changing.

    ``run_019fd4dc`` (docs/1b-leverage-audit/grounding/val-split-status.md:285-296) shows this
    is not hypothetical, and also that it is not reliably as bad as "one domain either": that
    run's naive ``[:4]`` landed on four DIFFERENT domains -- ``adult_content``,
    ``art_and_design``, ``crime_and_law``, ``education_and_jobs`` -- purely because
    ``adult_content`` happened to have only one val shard before the alphabet moved on, with
    2,284 / 409 / 542 / 396 instances respectively. That is not single-category collapse, but
    it is not coverage either: 2,284 of those 3,631 instances (63%) are ``adult_content``, so
    any rung's 256-instance read is dominated by whichever domain happened to sort first, and a
    corpus with a differently-sized alphabetically-first domain -- or this same corpus after a
    single shard is re-cut -- collapses to exactly the one-domain failure this function exists
    to prevent, silently, with no line in this file changing. This corpus has 24 topic domains
    (``edullm-data/HANDOFF.md:467``: "24 topic domains (adult_content ... travel_and_tourism)"),
    of which the naive prefix reached 4 -- and which 4 was never a decision.

    Grouped by DOMAIN, not by (source, domain). ``all-dressed-snazzy2`` and ``s2pdf-redacted``
    are two different scrapes that both contribute ``adult_content``, ``crime_and_law``,
    ``education_and_jobs`` and ``art_and_design`` shards
    (``docs/scaling-audit/wandb_run_meta.json:148-197``). The risk a held-out ladder exists to
    catch is a model that is silently only ever evaluated on one TOPIC -- source is a
    processing detail underneath that, and two shards of the same domain from different
    sources close far less of the coverage gap than one shard of a domain not seen at all. So
    the grouping key drops source on purpose.

    Deterministic: ``sorted(urls)`` up front, then domains visited in sorted order at every
    depth, so the same input always yields the same picks regardless of manifest list order --
    load-bearing because the ladder compares 18 E1 cells against each other and a per-cell
    subset would make the rungs incomparable. Round-robins ACROSS domains (one shard from every
    domain before a second shard from any of them) rather than sorting shards and slicing, which
    is the one property that makes truncation -- ``limit`` smaller than the shard count --
    honest instead of silently biased: with round-robin, ``limit`` shards is ``limit`` DISTINCT
    domains (until they run out), never ``limit`` shards of the alphabetically-first one.
    Raising ``HELDOUT_SHARDS`` alone, with the plain sort left in place, would not have fixed
    this -- a bigger N still exhausts one domain before touching a second if that domain has N
    or more shards, which is exactly what made ``adult_content`` alone look safe on this corpus
    (it happens to have few val shards) and would not on one where it has many.

    :param urls: All held-out shard urls the corpus declares (``corpus.val_paths``).
    :param limit: How many to keep. If fewer distinct domains exist than ``limit``, some
        domains get a second shard once every domain has one. If ``urls`` itself has fewer
        entries than ``limit``, everything is returned.
    """
    by_domain: Dict[str, List[str]] = {}
    for url in sorted(urls):
        by_domain.setdefault(_domain_of(url), []).append(url)

    picked: List[str] = []
    max_depth = max((len(shards) for shards in by_domain.values()), default=0)
    for depth in range(max_depth):
        for domain in sorted(by_domain):
            if len(picked) >= limit:
                return picked
            if depth < len(by_domain[domain]):
                picked.append(by_domain[domain][depth])
    return picked


#: How many held-out shards the ladder scores, spread one-per-domain by
#: ``spread_across_sources`` before any domain gets a second (see that function's docstring for
#: why the ORDER of the selection, not merely raising this count, is what makes truncation
#: safe). 24 because that is the domain count this corpus actually has --
#: ``edullm-data/HANDOFF.md:467``, "24 topic domains (adult_content ... travel_and_tourism)" --
#: so it is the smallest N that can put every domain in front of the evaluator at least once.
#:
#: Raised from 4, which was sized on the wrong axis: the old comment justified 4 shards as
#: "~2M tokens, far more than the 32 batches a rung actually consumes" -- true, but volume was
#: never the problem, coverage was (see ``spread_across_sources``). This number changes what
#: gets DOWNLOADED and INDEXED once at startup, not what a rung reads: each rung is still capped
#: at ``eval_duration=Duration.steps(32)`` (:~1290 below), independent of how big the pool
#: behind it is. The real cost this pays is the one-time ``segment_documents_into_instances``
#: scan per shard inside ``--prepare-heldout-only``; the only measured data point is 4 shards
#: in 5.5 s total (``run_019fd4dc``, val-split-status.md:277-278), so 24 shards should still be
#: a startup cost of seconds, not minutes -- stated as an extrapolation, not something this
#: change ran and timed.
#:
#: If val ever has fewer than 24 distinct domains, ``spread_across_sources`` just returns fewer
#: distinct ones and some get a second shard -- nothing here needs to track the true count.
HELDOUT_SHARDS = 24


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
        # Prefixed by domain, not the bare basename: once selection spans multiple domains
        # (spread_across_sources, above) two different domains could in principle name a shard
        # the same way, and a basename-only cache key would let one silently shadow the other.
        # No collision has been observed on this corpus's shard numbering, which looks globally
        # assigned -- this is a defensive width, not a fix for something seen.
        dest = cache / f"{_domain_of(str(url))}--{os.path.basename(str(url))}"
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


@dataclass(frozen=True)
class Cell:
    """One point of an E1 fan-out grid: a schedule, a peak LR, and an init seed.

    Frozen and comparable so that a duplicate is detectable by putting cells in a set, which
    is the check ``parse_fanout_grid`` runs and the one the precedent lacked.

    ``lr_text`` is the LR exactly as it was typed. It is carried alongside the parsed float
    purely so a refusal can quote it back: ``2e-3`` and ``2-e3`` and ``2e_3`` are three
    different typos and a message reading "0.002" tells the submitter nothing about which
    one they made.
    """

    schedule: str
    lr: float
    seed: int
    lr_text: str = ""

    def key(self) -> Tuple[str, float, int]:
        """What makes two cells the same experiment, ignoring how the LR was spelled.

        ``5e-4`` and ``0.0005`` are the same cell and must collide in the duplicate check,
        so the text is deliberately not part of this.
        """
        return (self.schedule, self.lr, self.seed)


def parse_fanout_grid(spec: str, expected_size: Optional[int] = None) -> List[Cell]:
    """Parse ``"linear:2e-3:0,cosine:1e-3:1"`` into cells, or refuse and say which one broke.

    FOUR REFUSALS, NOT THE PRECEDENT'S THREE. ``train_liv_arm.py:parse_fanout_grid`` checks
    that each cell is well-formed and names a real arm, and stops there. It will happily
    accept ``L0:0,L0:0`` -- a grid holding the same cell twice. That is the same failure its
    own commit message describes ("a 12-cell pilot would have trained the SAME arm and seed
    twelve times") arriving by a different road: the array index is read correctly, every
    cell resolves to something, and two of them are simply the same run at twice the price
    with a hole where another cell should have been. Nothing downstream can see it, because
    two identical configurations produce two plausible and near-identical loss curves rather
    than an error. So a duplicate is refused here.

    ``expected_size`` is the second half of that guard. The duplicate check catches a grid
    that repeats itself; this catches one that is simply the wrong length -- 17 cells typed
    where 18 were meant, submitted with ``fanout_size: 18``. Optional so a single run and an
    ad-hoc grid are unaffected.

    :raises Refusal: If a cell is malformed, names a schedule that does not exist, repeats
        another cell, or if the grid's length disagrees with ``expected_size``.
    """
    cells: List[Cell] = []
    for raw in (part.strip() for part in spec.split(",")):
        if not raw:
            continue
        fields = raw.split(":")
        if len(fields) != 3:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} is not 'schedule:lr:seed'",
            )
        schedule, lr_text, seed_text = (field.strip() for field in fields)
        if schedule not in SCHEDULE_ALPHA_F:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} names an unknown schedule {schedule!r}; known: "
                + ", ".join(sorted(SCHEDULE_ALPHA_F)),
            )
        try:
            lr = float(lr_text)
        except ValueError:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} has a peak LR of {lr_text!r}, which is not a number",
            ) from None
        # Not merely unparseable but unusable: a non-positive or non-finite LR builds an
        # optimizer that trains nothing, or NaNs on the first step, and both look like a
        # training failure rather than a typo in a grid.
        if not (lr > 0.0) or lr != lr or lr == float("inf"):
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} has a peak LR of {lr_text!r}, which is not a positive "
                "finite number",
            )
        if not seed_text.lstrip("-").isdigit():
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} has a seed of {seed_text!r}, which is not an integer",
            )
        seed = int(seed_text)
        # seed_all refuses anything outside [0, 2^32-1] (utils.py:172) and so does the
        # generator model init builds, but it refuses AFTER the container has pulled and
        # started. Cheaper here.
        if not 0 <= seed <= 2**32 - 1:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {raw!r} has a seed of {seed}, outside [0, 2^32-1]",
            )
        cells.append(Cell(schedule=schedule, lr=lr, seed=seed, lr_text=lr_text))

    if not cells:
        raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, "--fanout-grid parsed to zero cells")

    # THE REFUSAL THE PRECEDENT DOES NOT HAVE. See the docstring.
    seen: Dict[Tuple[str, float, int], str] = {}
    for cell in cells:
        spelled = f"{cell.schedule}:{cell.lr_text}:{cell.seed}"
        if cell.key() in seen:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"fan-out cell {spelled!r} appears twice in the grid (first as "
                f"{seen[cell.key()]!r}). Two cells of one configuration cost two cells and "
                "leave a hole where a third configuration should have been, and both produce "
                "plausible loss curves -- so nothing downstream can tell.",
            )
        seen[cell.key()] = spelled

    if expected_size is not None and len(cells) != expected_size:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"--fanout-grid holds {len(cells)} cells but --fanout-expect says "
            f"{expected_size}. Set fanout_size on the submission form to the same number, or "
            "trailing cells never run.",
        )
    return cells


def resolve_fanout_cell(
    spec: str, index: Optional[str], expected_size: Optional[int] = None
) -> Optional[Cell]:
    """Pick this process's cell from the grid, using Batch's array index.

    ``fanout_index_parameter`` on the submission form is **documentation** -- it records what
    the index varies so the approving lead can see it, and nothing substitutes it into the
    command. Batch sets ``AWS_BATCH_JOB_ARRAY_INDEX`` in each cell's environment and the
    program is expected to read it. A command that ignores it runs identically in every cell:
    the grid costs 18x as much and produces one result 18 times.

    Returns ``None`` when no grid was requested, so a single run is completely unaffected.

    :raises Refusal: If the index is missing (every cell would train cell 0 and the sweep
        would look finished), or outside the grid (``fanout_size`` disagreeing with the grid
        drops trailing cells silently).
    """
    if not spec:
        return None
    cells = parse_fanout_grid(spec, expected_size)
    if index is None:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            "--fanout-grid was given but AWS_BATCH_JOB_ARRAY_INDEX is unset, so every cell "
            "would train the same schedule, LR and seed and the sweep would look finished. "
            "Submit with the fan-out fields, or drop --fanout-grid.",
        )
    try:
        i = int(index)
    except ValueError:
        raise Refusal(
            Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT,
            f"AWS_BATCH_JOB_ARRAY_INDEX is {index!r}, which is not an integer",
        ) from None
    if not 0 <= i < len(cells):
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"array index {i} is outside a {len(cells)}-cell grid; fanout_size must equal the "
            "number of cells in --fanout-grid",
        )
    return cells[i]


def steps_for_tokens(target_tokens: float, global_batch_size: int) -> int:
    """How many optimizer steps deliver ``target_tokens`` at this batch size.

    A NAMED FUNCTION SO THE TEST CAN CALL IT. A test that re-derives ``round(D / B)`` inline
    passes whichever way this rounds, and keeps passing after somebody changes it -- it tests
    its own copy of the arithmetic, not the code's. Calling this is what makes the assertion
    about the program.

    ROUND, NOT FLOOR OR CEIL, and the difference is one step in eleven thousand -- so the
    reason is legibility rather than tokens. E1's proxy is D=3e9 at 262,144 tokens/step =
    11,444.09 steps; ``round`` gives 11,444, which delivers 11,444 x 262,144 = 2,999,975,936
    tokens, 0.0008% short of 3e9. Ceil would give 11,445 = 3,000,238,080, 0.0079% over. Both
    are far inside seed noise. What matters is that the number is DERIVED from the batch
    rather than typed beside it: a batch changed on the command line without the steps
    changing with it silently moves the token budget, and a sweep whose arms saw different
    numbers of tokens is not measuring its schedule.
    """
    if global_batch_size <= 0:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"global batch size {global_batch_size} is not positive",
        )
    return max(1, round(target_tokens / global_batch_size))


def resolve_steps(opts) -> int:
    """The run's length, DERIVED from the token budget when one was declared.

    WHY THIS EXISTS AND WHY ``steps_for_tokens`` ALONE WAS NOT ENOUGH. ``steps_for_tokens``
    was defined, documented, and tested -- and called from nowhere in the program. Every
    caller was a test. ``--steps`` went verbatim into ``Duration.steps`` and the exact failure
    that function's docstring claims to prevent was still live, with a green test beside it
    saying otherwise. A guard outside the code path is not a guard; it is a second opinion
    nobody asked the code for.

    TWO SILENT FAILURES IT NOW CLOSES, both of which exit 0, write a checkpoint, and print a
    summary that reads normal:

      * ``--steps 11444`` with ``--global-batch-size`` omitted takes the flagship default of
        786,432 and trains **9.0B tokens, 3x the declared budget** -- at 3x the cost, on a
        cell whose whole purpose is to be a cheap proxy.
      * ``--target-tokens`` omitted and ``--steps`` omitted takes the default of 200 and
        trains **52.4M tokens, 1.75% of the budget**, in about half an hour. It also makes
        ``--warmup-fraction 0.1`` resolve to 20 steps, walking the smoke-test warmup constant
        the baseline fix removed straight back in.

    So when ``--target-tokens`` is given the step count is computed from it and the batch,
    and the two cannot disagree. When it is absent this returns ``opts.steps`` unchanged, so
    a run that never asked for a budget behaves exactly as it did before.

    A HAND-TYPED ``--steps`` BESIDE A BUDGET IS REFUSED RATHER THAN SILENTLY OVERRIDDEN. If
    both are given and they disagree, one of them is wrong and no rule for picking a winner
    is better than saying so: overriding hides a typo, and honouring it defeats the budget.
    Passing a ``--steps`` that already agrees with the derivation is allowed, because that is
    somebody being explicit rather than somebody being wrong.

    :raises Refusal: If an explicit ``--steps`` disagrees with the budget's derivation.
    """
    target = getattr(opts, "target_tokens", None)
    if target is None:
        return opts.steps

    derived = steps_for_tokens(target, opts.global_batch_size)

    # Read the sentinel from the parser rather than repeating the literal, so this cannot
    # drift from the flag it is comparing against.
    default_steps = build_parser().get_default("steps")
    if opts.steps != default_steps and opts.steps != derived:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"--steps {opts.steps} disagrees with --target-tokens {target:g} at a batch of "
            f"{opts.global_batch_size}, which needs {derived} steps "
            f"({derived * opts.global_batch_size:,} tokens). Pass the budget and let the "
            "steps follow, or pass neither -- a run whose two length declarations disagree "
            "trains one of them and reports the other.",
        )
    return derived


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

    # THE SEED HAS TO LAND HERE OR IT LANDS NOWHERE, AND "NOWHERE" IS SILENT.
    #
    # There are two different `init_seed`s in this program and only one of them initialises a
    # model. ExperimentConfig.init_seed (default 12536) is passed to `seed_all` in `train`,
    # which seeds python/numpy/torch's GLOBAL RNGs. TransformerConfig.init_seed (default 0) is
    # what `Transformer.init_weights` turns into `torch.Generator(device).manual_seed(seed)`
    # at model.py:294-299, and every `init_*` call in nn/transformer/init.py draws from THAT
    # generator. The global RNG is never consulted.
    #
    # MEASURED, not inferred. Building olmo2_190M(vocab_size=100352) twice under different
    # `seed_all` values and identical `TransformerConfig.init_seed` gives 135 of 135 parameter
    # tensors BIT-IDENTICAL; changing `TransformerConfig.init_seed` instead changes 86 of 135
    # (the other 49 are norm gains and biases, which are constants by construction).
    #
    # So a fan-out that varied only `ExperimentConfig.init_seed` would train three IDENTICAL
    # models per arm and report their zero variance as a tight seed distribution -- the E1
    # argmin would then be selected on a standard error that does not exist.
    if opts.init_seed is not None:
        model_config.init_seed = opts.init_seed

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
            # resolve_steps, NOT opts.steps. When --target-tokens is given the length is
            # derived from it and the batch, so the two cannot silently disagree; when it is
            # absent this is opts.steps unchanged. This is the ONE call site, and it being
            # the real one is the whole point -- steps_for_tokens previously existed, was
            # documented, was tested, and was called only from tests.
            max_duration=Duration.steps(resolve_steps(opts)),
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
        # THE SAME RESOLVED LENGTH max_duration GOT, for the reason given at :1138. A ladder
        # scaled off opts.steps is not merely inconsistent, it is silently useless: with a
        # budget given and --steps left at its 200 default, every rung lands inside warmup and
        # scores a model that has barely been trained, so a flat curve reads as a flat result.
        eval_steps = ladder_steps(resolve_steps(opts))
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
        #
        # `spread_across_sources`, NOT `sorted(...)[:HELDOUT_SHARDS]` -- see that function's
        # docstring. The REMOTE urls are what get grouped, before localisation renames them to
        # a flat cache directory that no longer carries the domain in its path.
        heldout_urls = spread_across_sources(corpus.val_paths, HELDOUT_SHARDS)
        # Labels derived from the SAME urls and the SAME function as the selection grouping
        # (`_domain_of`), before they are localised. This is the other half of the coupled fix:
        # a selection that spans domains but keeps the single literal "heldout-val" label would
        # merge every domain's CE back into one bucket and throw away exactly the coverage this
        # was for. See `LossWatcher.log_metrics` below for the half that reads these labels back
        # out -- that half has to change in the SAME commit, not a later one, or the metric key
        # it looks for stops existing and the fallback to train CE re-inverts the ranking this
        # was written to prevent.
        heldout_labels = [_domain_of(url) for url in heldout_urls]
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
                    # `spread_across_sources` picks the same subset for the same `corpus.val_paths`
                    # every time -- it sorts internally rather than depending on the order `urls`
                    # arrives in -- so the subset is still identical across arms and seeds. A
                    # per-cell subset would make the rungs incomparable, which is the one thing
                    # the ladder cannot tolerate; spreading the selection across domains does not
                    # reintroduce that risk, it only changes WHICH fixed subset gets picked.
                    paths=heldout_paths,
                    # One label PER DOMAIN now, not one shared literal for every path.
                    # LMEvaluator.from_numpy_dataset raises when any path lacks a "label"
                    # (eval/lm_evaluator.py:60-66) and builds one MeanMetric PER DISTINCT LABEL
                    # (eval/lm_evaluator.py:39,114-117), so this label list is what turns one
                    # combined held-out number into one number per domain -- which is the entire
                    # point of spreading the selection: a single blended average could still hide
                    # a domain that trains badly behind others that train well. `LossWatcher`
                    # below reads exactly these per-domain keys back out; see its module-level
                    # `_HELDOUT_CE_LOSS_KEY_RE` for the other half of this coupling.
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


#: A held-out CE-loss key looks like ``eval/lm/<domain>/CE loss`` -- one such key PER LABEL
#: ``LMEvaluator.compute_metrics`` emits (eval/lm_evaluator.py:114-117: one
#: ``out[f"{label}/CE loss"]`` per distinct ``metadata["label"]`` the dataset carries), re-keyed
#: by ``EvaluatorCallback.perform_eval`` as ``f"{prefix}/{evaluator.name}/{name}"``
#: (train/callbacks/evaluator_callback.py:171) with prefix "eval" (:116) and name "lm" hardcoded
#: in ``LMEvaluatorCallbackConfig.build`` (:281).
#:
#: Before this fix every held-out path carried the single literal label "heldout-val", so there
#: was exactly one such key and a hardcoded ``metrics.get("eval/lm/heldout-val/CE loss")`` found
#: it. Now that the paths passed to ``NumpyPaddedFSLDatasetConfig`` above are labelled by
#: ``_domain_of`` (one label per topic domain -- see ``spread_across_sources``), there is one key
#: PER DOMAIN SELECTED and that single hardcoded lookup matches nothing: this regex, and the
#: aggregation below, are the other half of that same fix, landing in the same commit. Changing
#: the label without this is exactly how Defect B happens -- the metric key that has to agree
#: with the label lives in a different function, so a straight port of the selection fix alone
#: leaves this reading None and silently falling back to train CE.
_HELDOUT_CE_LOSS_KEY_RE = re.compile(r"^eval/lm/[^/]+/CE loss$")

#: Descriptive, not a literal metric key -- there no longer is one key, there are several
#: (one per domain that had data this rung), and this is their combination. See
#: ``LossWatcher.log_metrics`` for why the combination is an UNWEIGHTED mean of domains rather
#: than a token-weighted mean of instances: weighting by tokens would let whichever domain
#: happened to get the most eval batches in a 32-batch-capped rung dominate the number again,
#: which is the same imbalance ``spread_across_sources`` exists to avoid at the selection layer.
_HELDOUT_LOSS_SOURCE = "eval/lm/*/CE loss (unweighted mean over domains with data this rung)"

_TRAIN_LOSS_KEY = "train/CE loss"


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
        #: Which metric ``first``/``last`` each hold, tracked SEPARATELY rather than as one
        #: shared ``loss_source`` field. That single field was Defect C: ``first`` is set
        #: unconditionally on whatever loss is seen on the very first call to this method --
        #: which in practice is train CE, since it is logged every step while a held-out rung
        #: fires only at ``fixed_steps`` -- while ``last`` is guarded below to never downgrade
        #: away from a held-out reading once one lands. A single field can only name the more
        #: recent of those two decisions, so once a rung fires it silently relabels `first` as
        #: held-out CE even though the number sitting in `first` was never touched. Two fields
        #: cannot make that mistake: each is written only at the moment its own value is written.
        self.first_loss_source: Optional[str] = None
        self.last_loss_source: Optional[str] = None
        self.wandb_url = ""

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        del step
        if not self.wandb_url:
            with contextlib.suppress(Exception):
                import wandb

                self.wandb_url = getattr(wandb.run, "url", "") or ""
        # HELD-OUT FIRST, TRAIN ONLY AS A FALLBACK.
        #
        # ``summarise`` reports first_loss/last_loss and the comparison across a sweep's cells is
        # taken over exactly those records, so whatever this reads is what ranks the arms. Reading
        # train CE ranks them on the wrong quantity: a decay-to-zero schedule ends at a
        # mechanically lower TRAIN loss than a decay-to-10% one at equal quality, so an argmin over
        # train CE can invert the very comparison E1 exists to make. The measurement protocol
        # (docs/1b-leverage-audit/EXPERIMENT-PLAN.md §5 rule 1) requires held-out CE.
        #
        # NaN-filtered: a domain that this rung's ``eval_duration=Duration.steps(32)`` cap never
        # actually reached leaves its ``MeanMetric`` un-updated, and ``MeanMetric.compute()``
        # (eval/metrics.py:80-87) returns ``0/0`` for that -- a real NaN, not a zero -- so an
        # unfiltered mean would silently corrupt every rung where 32 batches don't cover all
        # selected domains, which given 24 domains is most of them. Filtered out here rather than
        # upstream because upstream (``MeanMetric``) has no way to know which of its callers can
        # tolerate a NaN and which cannot.
        heldout_values = [
            value
            for key, value in metrics.items()
            if _HELDOUT_CE_LOSS_KEY_RE.match(key) and not math.isnan(value)
        ]
        loss: Optional[float]
        source: str
        if heldout_values:
            loss = sum(heldout_values) / len(heldout_values)
            source = _HELDOUT_LOSS_SOURCE
        else:
            loss = metrics.get(_TRAIN_LOSS_KEY)
            if loss is None:
                return
            source = _TRAIN_LOSS_KEY
            # Do not downgrade `last`: once a held-out reading has landed there, a later
            # train-only step must not silently overwrite it with a different quantity. `first`
            # carries no such guard -- it is written at most once, below, the first time this
            # method sees ANY loss -- so this check only ever protects `last`.
            if self.last_loss_source is not None and self.last_loss_source != _TRAIN_LOSS_KEY:
                return

        if self.first is None:
            self.first = float(loss)
            self.first_loss_source = source
        self.last = float(loss)
        self.last_loss_source = source


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
                # STEPS ALONE DO NOT SAY HOW LONG THE RUN WAS. 11,444 steps at 262,144
                # tokens and 11,444 at 786,432 are the same number here and a 3x different
                # experiment, and comparing two cells that differ in token budget is not a
                # schedule measurement. Multiplying it out is what makes that visible in the
                # record rather than requiring the reader to rederive it from the command.
                "tokens_trained": (
                    trainer.global_step * opts.global_batch_size
                    if getattr(opts, "global_batch_size", None)
                    else None
                ),
                "global_batch_size": getattr(opts, "global_batch_size", None),
                "first_loss": losses.first,
                "last_loss": losses.last,
                # WHICH metric EACH of the two above is -- two fields, not one, because they can
                # be different quantities (LossWatcher.__init__'s docstring on
                # `first_loss_source`/`last_loss_source` explains why: `first` is whatever loss
                # is logged first, almost always train CE since it is logged every step while a
                # held-out rung only fires at fixed_steps; `last` is guarded to prefer held-out
                # CE once any rung has produced one). A sweep's argmin is taken over `last_loss`,
                # and held-out and train CE are not comparable quantities -- a decay-to-zero arm
                # ends at a mechanically lower TRAIN loss than a decay-to-10% arm of equal
                # quality. Emitted so a reader ranking cells can see they are comparing like with
                # like, and so a run that silently fell back to train CE -- or whose `first` and
                # `last` are not actually the same kind of number -- is visible in the log rather
                # than inferred from the corpus.
                "first_loss_source": losses.first_loss_source,
                "last_loss_source": losses.last_loss_source,
                "seconds": seconds,
                "peak_memory_gib": peak,
                "checkpoint_uri": opts.save_folder,
                "wandb_project": os.environ.get("EDULLM_WANDB_PROJECT", ""),
                "wandb_url": losses.wandb_url,
                # WHICH CELL THIS WAS. 18 cells of an E1 array all print one of these into
                # one log group, and the argmin is over (schedule, lr) averaged across seed.
                # Without these four fields the analyst has to join a loss back to its
                # configuration through the array index, which lives only in the job name --
                # and a join that can be got wrong on a sweep whose whole output is a ranking
                # is the sweep's single point of failure. `getattr` because `summarise` is
                # also called with hand-built option objects in tests and by anyone running
                # this by hand.
                "lr_schedule": getattr(opts, "lr_schedule", None),
                "peak_lr": getattr(opts, "learning_rate", None),
                "init_seed": getattr(opts, "init_seed", None),
                "data_seed": getattr(opts, "data_seed", None),
                "array_index": os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"),
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

    # WHAT THIS SEEDS, AND WHAT IT DOES NOT. `ExperimentConfig.init_seed` is inherited from
    # upstream's script_utils.py:35 and its name is a trap: it seeds the GLOBAL python/numpy/
    # torch RNGs and it does NOT seed model init. Model init draws from a generator
    # `Transformer.init_weights` builds from `TransformerConfig.init_seed` (model.py:294-299),
    # which `build_config` sets from `--init-seed`. Measured: varying this value alone leaves
    # all 135 of the 190M's parameter tensors bit-identical.
    #
    # IT IS HELD FIXED ACROSS AN E1 FAN-OUT, ON PURPOSE. torch's DataLoader draws each
    # worker's base seed from the global RNG when no generator is passed (and none is --
    # data_loader.py:578-586), so moving this would move worker RNG state. The batch ORDER is
    # not affected either way, because the loader seeds its own generators explicitly from
    # `--data-seed` (data_loader.py:547, :673, :860). But holding it fixed makes "same data
    # order across seeds" true by construction rather than true by an audit of every worker
    # path, and §5 rule 2's paired design is what rests on it.
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
    parser.add_argument(
        "--target-tokens",
        type=float,
        default=None,
        help="Token budget for the run. When given, the step count is DERIVED from this and "
        "--global-batch-size rather than typed beside them, and an explicit --steps that "
        "disagrees is refused. Unset (the default) leaves --steps exactly as it was, so a "
        "run that declares no budget is unaffected. E1 passes 3e9. Without this, omitting "
        "--global-batch-size trains 3x the budget at the flagship default and omitting "
        "--steps trains 1.75% of it -- both exit 0 and write a plausible checkpoint.",
    )
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
    parser.add_argument(
        "--data-seed",
        type=int,
        default=0,
        help="Seeds BATCH ORDER only -- it reaches NumpyDataLoaderConfig(seed=...) and "
        "nothing else. EXPERIMENT-PLAN §5 rule 2 wants paired seeds: same data order, "
        "different init. So an E1 fan-out varies --init-seed and holds this FIXED.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Seeds MODEL INIT, by reaching TransformerConfig.init_seed. Unset means the "
        "factory default of 0. Verified by experiment, not by reading: seed_all() alone "
        "changes ZERO of the 190M's 135 parameter tensors, because init_weights builds its "
        "own torch.Generator from model.init_seed (model.py:294-299) and never touches the "
        "global RNG. A fan-out that varied only the global seed would train 3 IDENTICAL "
        "models per arm and report their agreement as low seed variance.",
    )
    parser.add_argument(
        "--fanout-grid",
        default="",
        help="Comma-separated 'schedule:lr:seed' cells, e.g. "
        "'linear:2e-3:0,cosine:1e-3:1'. The cell for THIS process is picked by "
        "AWS_BATCH_JOB_ARRAY_INDEX, so one submission trains the whole grid. The seed sets "
        "--init-seed and NOT --data-seed, so the arms are paired on data order. Without "
        "this flag nothing changes and a single run is unaffected.",
    )
    parser.add_argument(
        "--fanout-expect",
        type=int,
        default=None,
        help="How many cells --fanout-grid is meant to hold. Refuses if it holds a different "
        "number, which is the failure a wrong fanout_size on the form produces: trailing "
        "cells never run and the sweep still reports as complete. E1 is 2x3x3 = 18.",
    )
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


def apply_fanout_cell(opts, index: Optional[str]) -> Optional[Cell]:
    """Fold this process's grid cell into ``opts``, or leave ``opts`` alone if there is no grid.

    WHY THE SEED GOES TO ``init_seed`` AND NOT TO ``data_seed``, WHICH IS A DELIBERATE
    DIVERGENCE FROM THE PRECEDENT. ``train_liv_arm.py:1347`` sets ``data_seed = arm_seed``
    with the comment "paired: same seed drives init AND data order", so its three seeds vary
    the model init AND the batch order together. That is a legitimate design -- it estimates
    total run-to-run variance -- but it is not this experiment's.

    EXPERIMENT-PLAN §5 standing rule 2 is explicit: "**Paired seeds**, same data order,
    different init. Report the paired difference." Holding the data order fixed is what makes
    the cosine-vs-linear contrast at a given seed a PAIRED comparison: the two arms see the
    same tokens in the same order, so the batch-order component of the variance cancels in
    the difference instead of being added to it. Letting the data order move with the seed
    would put that variance back into the contrast, widening the interval E1 is already
    tight against -- the predicted 0.025-nat schedule gap sits below the n=3 MDE of 0.050,
    so the design has no variance to spare.

    Hence: the cell's seed sets ``--init-seed`` only, and ``--data-seed`` keeps whatever it
    was given (0 by default) in every one of the 18 cells.
    """
    cell = resolve_fanout_cell(opts.fanout_grid, index, opts.fanout_expect)
    if cell is None:
        return None
    opts.lr_schedule = cell.schedule
    opts.learning_rate = cell.lr
    opts.init_seed = cell.seed
    # data_seed is deliberately NOT touched. See the docstring.
    log.info(
        "fan-out cell %s: schedule=%s peak_lr=%g init_seed=%d data_seed=%d (data order is "
        "held fixed across seeds; only init varies)",
        index,
        cell.schedule,
        cell.lr,
        cell.seed,
        opts.data_seed,
    )
    return cell


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()

    # Before anything expensive, so the log's first lines name the schedule, LR and seed this
    # container is actually training rather than the flag defaults -- and so a grid submitted
    # without the fan-out fields dies in a second rather than after an image pull.
    apply_fanout_cell(opts, os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))

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
        prepare_training_environment()
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
