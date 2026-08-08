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
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, cast

import rich
import torch

from olmo_core.config import Config, DType
from olmo_core.data import (
    NumpyDataLoaderConfig,
    NumpyDatasetDType,
    NumpyFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import (
    all_reduce_value,
    barrier,
    get_rank,
    get_world_size,
)
from olmo_core.io import clear_directory, list_directory, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
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
    # The two below are the held-out endpoint, and they are SEPARATE FROM 72 on purpose. A run
    # that trained for eleven hours and then could not produce a CE has a checkpoint on S3 and
    # is worth re-evaluating from it; a run whose training failed has nothing. Both used to be
    # exit 0 with a null field in the JSON, which is the failure these numbers exist to end.
    THE_CORPUS_DECLARES_NO_HELD_OUT_SPLIT = 73
    THE_HELD_OUT_EVALUATION_SCORED_NOTHING = 74


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
    #: The corpus's own held-out objects and the row count its manifest declares for them.
    #:
    #: Carried on the config for the same reason ``dataset_id`` is: the record that lands beside
    #: the checkpoint then names the exact objects the endpoint was computed over, so a CE in
    #: the summary can be traced to a token set rather than taken on trust. It also puts them
    #: where ``train()`` can reach them -- it receives the config and not the ``Corpus``.
    val_paths: list[str] = field(default_factory=list)
    val_rows: int | None = None


@dataclass
class Corpus:
    """What the manifest says, after the three checks that make it safe to memmap."""

    dataset_id: str
    version: str
    paths: list[str]
    dtype: NumpyDatasetDType
    tokenizer: TokenizerConfig
    rows: int | None
    #: The corpus's OWN held-out objects, taken from the reader's split resolution.
    #:
    #: NOT reconstructed from shard names, and that is not a stylistic preference. A mask named
    #: ``all-dressed-snazzy2__val-00212`` corresponds to
    #: ``all-dressed-snazzy2/art_and_design/val-00212.u32le.bin`` -- the topic directory is
    #: dropped from the name and 24 topics exist, so rebuilding a key from a filename fetches
    #: a real, readable, plausible shard belonging to a different topic. The reader's
    #: ``.val`` is the only place the true keys are written down.
    val_paths: list[str] = field(default_factory=list)
    #: Rows the manifest DECLARES for the held-out partitions, or None if it declares none.
    #: This is the number the realized token count is checked against; see
    #: :func:`evaluate_val_aggregate`.
    val_rows: int | None = None


def corpus_from_manifest(read, *, dataset_id: str, version: str, tokenizer_id: str) -> Corpus:
    """Turn what the reader returned into what OLMo-core needs, or refuse and say why.

    Separate from the fetch because this is the part with the judgement in it, and a test
    should be able to hand it a manifest describing a big-endian corpus without standing up
    S3 or installing the reader. ``read`` is duck-typed for that reason: anything carrying
    ``paths``, ``dtype``, ``byte_order``, ``header_bytes`` and ``rows`` will do.

    THE HELD-OUT SPLIT IS TAKEN FROM ``read.val``, WHICH IS THE READER'S OWN ANSWER. It is a
    property over ``read.splits``, derived from ``is_trainable(name)`` over the partition names
    ``dataset.json`` declares -- so the val objects here are the exact keys the manifest sealed,
    topic directory and all. ``getattr`` with a default rather than a bare attribute access
    because ``val``/``split_rows`` arrived in the reader after ``paths`` did, and a corpus or a
    test fake that predates them should degrade to "no held-out split declared" rather than
    raising an AttributeError in the middle of a config build.
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

    # WHICH SPLITS ARE HELD OUT IS THE READER'S ANSWER, NOT A RULE RESTATED HERE. `read.val`
    # is a property over `read.splits` filtered by the reader's own `is_trainable`, so the
    # objects are the exact keys the manifest sealed -- topic directory and all.
    val_paths = list(getattr(read, "val", None) or [])
    split_rows = getattr(read, "split_rows", None) or {}
    splits = getattr(read, "splits", None) or {}

    # EXACTLY ONE HELD-OUT PARTITION, OR REFUSE. The reader's `is_trainable` excludes every
    # split that is not `train`, and the dataset standard's vocabulary is {train, val, test} --
    # so a corpus that declares a `test` partition hands back val AND test concatenated in
    # `.val`. Summing both declarations would make the token check BALANCE while the reported
    # endpoint quietly included the test set: both sides of the equality move together, so the
    # exact check cannot see it, and the number that gets published is contaminated.
    #
    # Two held-out partitions is also a question this file cannot answer on its own -- which
    # one is the endpoint? -- so it is refused rather than guessed at. olmo-150b-dolma2-v1
    # declares one (`val`, 229,894,171 rows) and this passes; a corpus that grows a `test`
    # split fails here, in the first seconds, rather than by publishing a wrong CE.
    held_out_splits = sorted(
        name for name, paths in splits.items() if paths and set(paths) <= set(val_paths)
    )
    if len(held_out_splits) > 1:
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} declares {len(held_out_splits)} held-out partitions "
            f"({', '.join(held_out_splits)}) and this entry point scores exactly one. Summing "
            "them would put the test set inside the reported endpoint with the token check "
            "still balancing, because both sides of it would move together.",
        )

    # And no object twice. `.val` is a concatenation over partitions, so two partitions that
    # overlap would list a shard twice -- and a shard scored twice is weighted twice in a CE
    # that is supposed to be a plain per-token mean.
    if len(set(val_paths)) != len(val_paths):
        raise Refusal(
            Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP,
            f"{dataset_id}/{version} lists {len(val_paths) - len(set(val_paths))} held-out "
            "object(s) more than once, so those tokens would be weighted twice in the mean.",
        )

    val_rows = split_rows.get(held_out_splits[0]) if held_out_splits else None

    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths=list(read.paths),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=tokenizer,
        rows=read.rows,
        val_paths=val_paths,
        val_rows=val_rows,
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


def torn_step_directories(save_folder: str) -> list[str]:
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


def remove_torn_checkpoints(save_folder: str) -> list[str]:
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
    removed: list[str] = []
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


def build_config(opts, overrides: list[str]):
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
    # Said at CONFIG time rather than only at eval time, so a corpus that declares no held-out
    # split is visible before the GPU hours are spent rather than after. The endpoint refuses
    # in that case (Stage 73), and finding that out eleven hours in is the expensive version.
    log.info(
        "held out: %d object(s), %s declared token(s)",
        len(corpus.val_paths),
        "none" if corpus.val_rows is None else f"{corpus.val_rows:,}",
    )

    # Every comparison arm is built by the one frozen geometry ledger beside this
    # runner. Keeping arm construction out of the fan-out command ensures every cell
    # differs only by its explicit arm and seeds.
    from model_arch_tests import RUNNABLE_ARMS, build_model_config

    if opts.arm not in RUNNABLE_ARMS:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"unknown arm: {opts.arm}. Declared arms: {', '.join(RUNNABLE_ARMS)}",
        )

    # The frozen comparison is parameter-matched at the Dolma2 padded vocabulary. Refuse a
    # different tokenizer before constructing a model whose parameter target no longer means
    # what the pre-registration says.
    vocab = corpus.tokenizer.padded_vocab_size()
    if vocab != 100_352:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"the comparison is frozen at padded vocab 100,352, but the corpus resolves to {vocab:,}",
        )
    # init_seed HAS TO BE PASSED HERE, AND THE VERSION WITHOUT IT WAS WORSE THAN NO FLAG.
    #
    # `seed_all(config.init_seed)` in `train()` seeds the GLOBAL rngs, and the global rng is
    # NOT what draws the weights: `Transformer.init_weights` builds its own
    # `torch.Generator(device).manual_seed(self.init_seed)` (model.py:298) and every
    # `init_embeddings` / `init_weights` call takes that generator explicitly. `self.init_seed`
    # comes from `TransformerConfig.init_seed`, which defaults to 0. So with the seed left off
    # this call, `--init-seed 0`, `1` and `2` all produced BIT-IDENTICAL weights while
    # `summarise()` printed the distinct number it was given -- a JSON record asserting a
    # variance component that was never varied, which is a false fact rather than a missing
    # one, and any CI built from those "seeds" is a data-order CI wearing an init-seed label.
    #
    model_config = build_model_config(opts.arm, opts.init_seed)
    log.info(
        "arm %s: %s params at vocab %d | attention layers %s",
        opts.arm,
        f"{model_config.num_params:,}",
        vocab,
        [3, 7, 11, 15],
    )

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
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
        ),
        # On, because the image now carries a C compiler. It was off in the platform's
        # getting-started command only because a run without one dies on the first compiled
        # region, which is a workaround that costs throughput on every run forever.
        compile_model=True,
        accumulate_grads_without_comm=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType(opts.param_dtype),
            reduce_dtype=DType.float32,
            reshard_after_forward=False,
        ),
        max_grad_norm=1.0,
        # `warmup`, not the `warmup_steps` the example still passes -- that spelling is
        # deprecated upstream and warns on every construction.
        scheduler=CosWithWarmup(warmup=opts.warmup_steps),
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

    # No lm_evaluator and no downstream_evaluator, and their absence is a decision. The
    # example's LM evaluator reads a C4 validation shard from olmo-data.org and the downstream
    # one pulls HellaSwag from Hugging Face; both would put a public-internet fetch in the
    # middle of a run whose whole claim is that it read a sealed corpus, and a failure in
    # either would look like a training failure.
    #
    # The held-out endpoint is `evaluate_val_aggregate`, run once after fit() over the corpus's
    # OWN `val` partition -- the shards the reader returns as `.val`, which is the right version
    # of the above and is what olmo-150b-dolma2-v1 declares. It is not a trainer callback
    # because it has to run on all ranks after the last step and report one all-reduced number,
    # not per-step per-rank metrics.

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        init_seed=opts.init_seed,
        val_paths=corpus.val_paths,
        val_rows=corpus.val_rows,
    )
    return config.merge(overrides)


#: Steps discarded from the STEADY-STATE figure, counted from the start of the fit.
#:
#: WHY A CUTOFF EXISTS AT ALL WHEN ``SpeedMonitorCallback`` ALREADY DROPS STEP 1. It drops
#: exactly one step, and one step is not what startup costs here. ``compile_model=True``, so
#: Dynamo traces and Inductor compiles on the first shape it sees; the trainer's ``_dry_run_batch``
#: absorbs some of that, but recompiles land on later steps whenever a shape or a guard changes,
#: and the allocator is still growing its pools for the first tens of steps. Averaged from step 2
#: those costs are divided over however many steps the run happened to have -- which makes the
#: reported number a function of RUN LENGTH rather than of the hardware.
#:
#: That is not hypothetical here. A 200-step run on this stack reported 455,789 tok/s where a
#: 20-step run at the SAME microbatch reported 303,072: a 1.5x spread from run length alone, and
#: the whole-run wall clock read 3.1x low on the short probe. An arm compared against another at a
#: different step count would inherit that spread as if it were a mixer property.
#:
#: 50 is chosen against the actual budget: cells run 1,900-2,300 steps, so this discards ~2.5%
#: of the run and leaves ~97.5% of it in the measurement. It is a constant rather than a flag
#: because two arms filtered at two cutoffs are not comparable, and a flag is how they end up so.
WARMUP_STEPS_EXCLUDED = 50


class StepSample(NamedTuple):
    """One completed training step: which one, how long it took, how many tokens it moved.

    ``tokens`` is GLOBAL -- summed over every rank -- because that is what
    ``Trainer.global_train_tokens_seen`` counts and it is the number a cost estimate wants.
    Per-device figures are derived by dividing at the point of reporting, where the world size
    is named beside them, rather than being baked in here where a reader cannot see the divisor.
    """

    step: int
    seconds: float
    tokens: int


def steps_after_warmup(
    samples: Sequence[StepSample], *, warmup_steps: int = WARMUP_STEPS_EXCLUDED
) -> list[StepSample]:
    """The samples whose step index is strictly past the warmup cutoff.

    Strictly greater, so ``warmup_steps=50`` keeps step 51 onward and the count of discarded
    steps is exactly ``warmup_steps``. ``>=`` would keep step 50 and discard 49, which is the
    kind of off-by-one that never shows up in a number anybody checks.

    Filtering on the STEP INDEX rather than on position in the list matters on a resumed run:
    a second Batch attempt starts at step 1,201 and its first sample is not its first step, so
    dropping "the first 50 entries" would discard 50 perfectly good steady-state steps and
    exclude nothing that needed excluding. The index is the run's, so the cutoff means the same
    thing on attempt one and attempt two.
    """
    return [sample for sample in samples if sample.step > warmup_steps]


def quantile_nearest_rank(values: Sequence[float], q: float) -> float | None:
    """The ``q``-quantile by nearest rank, or None when there is nothing to take it of.

    NEAREST RANK, NOT INTERPOLATION, and the choice is deliberate: this returns a step time that
    was actually observed rather than an average of two that were not. For "how long does a step
    take on this arm" an observed duration is the honest answer, and it keeps p50 and p90 the
    same kind of quantity computed by the same function rather than two conventions.

    Definition: sort ascending, take index ``ceil(q * n) - 1`` clamped into range. At ``q=0.5``
    and even ``n`` that is the lower of the two middle values.

    None rather than 0.0 on an empty input. A zero step time reads as an infinitely fast arm,
    which is the exact direction a missing measurement must never fail in.
    """
    if not values:
        return None
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1], got {q}")
    ordered = sorted(values)
    index = math.ceil(q * len(ordered)) - 1
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def throughput_tokens_per_second(samples: Sequence[StepSample]) -> float | None:
    """Total tokens over total seconds for these samples, or None if that is not a number.

    THE SUM OF TOKENS OVER THE SUM OF SECONDS, NOT THE MEAN OF THE PER-STEP RATES. Those are
    different quantities and only the first one is throughput: a mean of rates weights a fast
    step and a slow step equally, so a run with one 10x-slow checkpoint step reports a higher
    figure than it achieved. This ratio is what the run would be costed at, by construction.

    It is also what makes the figure robust to the host running ahead of the device. Individual
    step boundaries measured on the CPU are noisy under async CUDA -- the host can queue several
    steps before the GPU finishes one -- but the queue cannot run ahead forever, so the SUM over
    hundreds of steps is bounded by real device time and converges to it. The per-step p50/p90
    beside it are the noisier quantities, which is why they are reported as a distribution and
    not as the throughput.

    None, never 0.0, when either total is non-positive -- which is what a data loader that
    reports no token count leaves behind, and a zero-token run must not report as zero speed.
    """
    total_tokens = sum(sample.tokens for sample in samples)
    total_seconds = sum(sample.seconds for sample in samples)
    if total_tokens <= 0 or total_seconds <= 0:
        return None
    return total_tokens / total_seconds


#: Peak bf16/fp16 DENSE tensor-core throughput per GPU, keyed by a substring of the name torch
#: reports, in FLOP/s.
#:
#: WHY THIS TABLE EXISTS WHEN ``SpeedMonitorCallback`` ALREADY HAS ONE. That one ends in
#: ``else: # for other GPU types, assume A100``, so EVERY unrecognised card is assigned the
#: A100's 312 TFLOP/s. On an A100 that is right. On an L40S -- 181 TFLOP/s dense -- it is 1.72x
#: too high, and the MFU printed from it is 1.72x too LOW, silently, with no marker saying the
#: denominator was a guess. A wrong denominator that looks authoritative is worse for a
#: cross-arm comparison than no MFU at all, because all six arms inherit the same wrong constant
#: and the ratio between them survives while the absolute number quietly does not.
#:
#: So this table has no fallback branch. A card that is not listed yields None, ``mfu_pct``
#: is null, and ``mfu_basis`` says the peak was not known for that device. Adding a card is one
#: line and a citation; guessing one is what this is here to stop.
#:
#: Values are the vendor's listed dense figure -- the marketing number is quoted WITH structural
#: sparsity and is exactly 2x this, which is the factor ``speed_monitor`` writes as
#: ``dense_correction = 0.5``.
DEVICE_PEAK_BF16_FLOPS = {
    # A100 40GB and 80GB, SXM and PCIe, are all 312 TFLOP/s dense bf16.
    "A100": int(624e12 * 0.5),
    "H100 NVL": int(1671e12 * 0.5),
    "H100 PCIe": int(1513e12 * 0.5),
    # SXM and the other H100 variants.
    "H100": int(1979e12 * 0.5),
    "B200": int(4.5e15 * 0.5),
    # Ada, and the card the FarmShare probes ran on. 362 TFLOP/s with sparsity.
    "L40S": int(362e12 * 0.5),
}


def device_peak_bf16_flops(device_name: str | None) -> int | None:
    """Peak dense bf16 FLOP/s for this GPU, or None when this file does not know the card.

    Longest key first, so ``H100 NVL`` is not shadowed by the ``H100`` entry that also matches
    it. Substring matching because ``torch.cuda.get_device_name`` returns strings like
    ``NVIDIA A100-SXM4-80GB`` and ``NVIDIA H100 80GB HBM3``, which no exact table can key on.

    Returns None for an unknown or absent device rather than assuming anything. See
    :data:`DEVICE_PEAK_BF16_FLOPS` for why that is the whole point of the function.
    """
    if not device_name:
        return None
    for key in sorted(DEVICE_PEAK_BF16_FLOPS, key=len, reverse=True):
        if key in device_name:
            return DEVICE_PEAK_BF16_FLOPS[key]
    return None


def model_flops_utilisation(
    *,
    tokens_per_second_per_device: float | None,
    flops_per_token: int | None,
    device_peak_flops: int | None,
) -> float | None:
    """MFU as a percentage, or None if any of the three inputs is missing.

    ``100 * (tokens/s/device * FLOPs/token) / peak FLOP/s``, the PaLM definition. Every input is
    optional and any one of them being absent makes the output None rather than zero: an MFU of
    0.0% and an unmeasured MFU are opposite claims about the hardware, and only one of them is
    ever true here.
    """
    if not tokens_per_second_per_device or not flops_per_token or not device_peak_flops:
        return None
    return 100.0 * (tokens_per_second_per_device * flops_per_token) / device_peak_flops


class LossWatcher(Callback):
    """Keeps what the summary can only learn while the run is still going.

    The W&B url is read here rather than in ``summarise`` because ``WandBCallback.post_train``
    finishes the run, after which ``wandb.run`` is None. Read on a metrics callback rather
    than in ``pre_train``, because callbacks of equal priority run in reverse registration
    order and this one is registered last, so ``pre_train`` here happens before W&B has a run
    to name.
    """

    def __init__(self) -> None:
        self.first: float | None = None
        self.last: float | None = None
        self.wandb_url = ""
        #: Steady-state throughput, excluding startup and compilation.
        #:
        #: ``seconds`` in the summary wraps the whole of ``fit()``, so tokens/second derived
        #: from it charges process start, dataset open, FSDP wrap and the torch.compile of
        #: the first step against the measurement. On a short probe that is most of the
        #: wall-clock and it under-reports the hardware badly -- and unequally across shapes,
        #: since a bigger machine pays more fixed cost and would look worse than it is.
        #:
        #: ``SpeedMonitorCallback`` already measures the right thing: the trainer registers
        #: it automatically (``trainer.py:347``) and it resets its clock AFTER step 1,
        #: because "the first one tends to take unusually long". Its per-device average
        #: reaches metrics as ``throughput/device/TPS (actual avg)``. Nothing forwarded it to
        #: the JSON summary, which is the only channel the platform reads back.
        self.tps_device_avg: float | None = None
        #: Instantaneous per-device TPS from the last logged step, for a sanity check that
        #: the average is not still climbing when the probe ends.
        self.tps_device_last: float | None = None

        # --- per-step timing, sampled here rather than read off a metric --------------------
        #
        # WHY THIS DOES NOT REUSE ``throughput/device/TPS``. That metric only reaches
        # ``log_metrics`` every ``metrics_collect_interval`` steps -- 5 here -- so a p90 over
        # what arrives is a p90 over a fifth of the steps, and it is a BIASED fifth: the steps
        # that get logged are exactly the ones at a multiple of 5, which is also where the
        # checkpointer and the console logger do their work. The slow steps this is meant to
        # find are disproportionately in the sample and the fast ones are not.
        #
        # ``post_step`` runs on EVERY step, so the sample below is the whole population.
        #: One :class:`StepSample` per completed step, in step order.
        self.steps: list[StepSample] = []
        self._step_clock: float | None = None
        self._tokens_seen_at_last_step: int | None = None

        # --- peak memory, sampled BEFORE the monitor resets it ------------------------------
        #
        # THE READ IN ``summarise()`` IS TRUNCATED AT SOURCE AND LOOKS LIKE A WHOLE-RUN PEAK.
        # ``GPUMemoryMonitorCallback.post_step`` calls ``torch.cuda.reset_peak_memory_stats()``
        # on EVERY step, so a ``max_memory_allocated()`` read after ``fit()`` reports the peak
        # of the LAST STEP ONLY. It under-reports by however much the true peak exceeded the
        # final step, it is named like a run peak, and it is the field somebody sizes hardware
        # with -- so the error is silent and in the unsafe direction.
        #
        # This callback has priority 0 and the monitor has -1, and the trainer runs higher
        # priorities first, so this ``post_step`` is guaranteed to run BEFORE the reset that
        # would destroy the reading. That ordering is load-bearing and is asserted in the
        # tests rather than left to hold by luck.
        #
        # It matters most for exactly the arm this bake-off is worried about: the R=2
        # Householder backward allocates an O(B*T*H*K*V) fp32 workspace, which is a transient
        # inside a step. A per-step maximum sees it; a figure sampled after the run does not.
        #: Running maximum of ``torch.cuda.max_memory_allocated()`` over every step, in bytes.
        self.peak_allocated_bytes: int = 0
        #: Running maximum of ``torch.cuda.max_memory_reserved()`` over every step, in bytes.
        #: Reserved is what the allocator took from the driver and is the number that decides
        #: whether a second process fits on the card; allocated is what the tensors needed.
        self.peak_reserved_bytes: int = 0
        #: How many steps contributed to the two figures above. Zero means they were never
        #: sampled and must be reported as null rather than as 0 GiB.
        self.memory_samples: int = 0

    def post_step(self) -> None:
        """Close out the step that just finished: its duration, its tokens, its memory peak.

        ``post_step`` rather than ``log_metrics`` because it runs on every step instead of
        every fifth, and because the memory reading has to be taken before
        ``GPUMemoryMonitorCallback`` (priority -1, therefore later) resets it.

        The first call establishes the clock and records nothing. There is no completed step to
        time yet at that point, and a "duration" measured from the callback's construction to
        the end of step one is process startup wearing a step's name -- which is the single
        number this whole exercise exists to keep out of the throughput figure.
        """
        now = time.perf_counter()
        trainer = self._trainer
        tokens_seen = getattr(trainer, "global_train_tokens_seen", None) if trainer else None
        step = getattr(trainer, "global_step", None) if trainer else None

        if self._step_clock is not None and step is not None:
            # Tokens are DIFFERENCED from the trainer's own running total rather than
            # recomputed as `global_batch_size`. A short final batch, a resumed run, or a
            # dynamic batch size all make the product wrong while the difference stays right,
            # and the difference is the same quantity the trainer bills the schedule against.
            tokens = 0
            if tokens_seen is not None and self._tokens_seen_at_last_step is not None:
                tokens = int(tokens_seen) - int(self._tokens_seen_at_last_step)
            self.steps.append(
                StepSample(step=int(step), seconds=now - self._step_clock, tokens=max(tokens, 0))
            )

        self._step_clock = now
        if tokens_seen is not None:
            self._tokens_seen_at_last_step = int(tokens_seen)

        if torch.cuda.is_available():
            # Both maxima, both since the monitor's last reset, both accumulated here into a
            # true running maximum over the whole run.
            self.peak_allocated_bytes = max(
                self.peak_allocated_bytes, int(torch.cuda.max_memory_allocated())
            )
            self.peak_reserved_bytes = max(
                self.peak_reserved_bytes, int(torch.cuda.max_memory_reserved())
            )
            self.memory_samples += 1

    def log_metrics(self, step: int, metrics: dict[str, float]) -> None:
        del step
        if not self.wandb_url:
            with contextlib.suppress(Exception):
                import wandb

                self.wandb_url = getattr(wandb.run, "url", "") or ""
        tps_avg = metrics.get("throughput/device/TPS (actual avg)")
        if tps_avg is not None:
            self.tps_device_avg = float(tps_avg)
        tps_now = metrics.get("throughput/device/TPS")
        if tps_now is not None:
            self.tps_device_last = float(tps_now)
        loss = metrics.get("train/CE loss")
        if loss is None:
            return
        if self.first is None:
            self.first = float(loss)
        self.last = float(loss)


#: Gap bands, and the bit each one occupies in the frozen mask files. Must match
#: build_slice_masks.py exactly -- the masks are written once and are read here as bytes.
BAND_BIT = {0: 1, 32: 2, 256: 4, 1024: 8, 4096: 16}

#: Rows of logits to upcast at a time. At vocab 100,352 a single 4096-token sequence's logits
#: are 0.77 GiB in bf16 and 1.53 GiB upcast, so a whole-batch `.float()` allocates several GiB
#: in one block and OOM'd a 44 GiB card during development. Chunking makes the loss memory
#: independent of batch size; verified bitwise-identical to F.cross_entropy in value AND
#: gradient before it was adopted.
CE_CHUNK = 4096


def _chunked_ce(logits, targets):
    """Per-token cross-entropy without materialising the full fp32 logit tensor."""
    out = []
    for i in range(0, targets.numel(), CE_CHUNK):
        out.append(
            torch.nn.functional.cross_entropy(
                logits[i : i + CE_CHUNK].float(), targets[i : i + CE_CHUNK], reduction="none"
            )
        )
    return torch.cat(out)


def _shard_token_count(path, *, dtype) -> int:
    """How many tokens a local shard holds, from its size on disk rather than a manifest.

    The independent half of the count check in :func:`evaluate_val_aggregate`: the manifest
    DECLARES a row count and this measures what actually arrived, so a dropped shard, a
    truncated download or a glob that matched the wrong objects makes the two disagree.
    """
    import numpy as np

    return int(os.path.getsize(path) // np.dtype(dtype).itemsize)


def _shard_windows(path, *, seq_len: int, micro: int, dtype):
    """Yield ``(offsets, inputs, targets)`` per micro-batch of one LOCAL shard.

    The one implementation of the memmap-and-window arithmetic, shared by the sliced evaluator
    and the aggregate one so the two cannot drift on the off-by-one. ``offsets`` is the token
    offset of each window in the shard, which is what a caller needs to align a per-token
    side-channel (the band mask) against the TARGETS -- those start one token later than the
    inputs, and a mask sliced from ``off`` rather than ``off + 1`` scores the wrong positions
    and still produces a plausible cross-entropy.

    ``dtype`` is required rather than defaulted. Guessing a width here is the same failure the
    header of this file is about: uint16 over a uint32 corpus decodes to in-range ids and a
    loss curve that is merely bad. Native byte order is correct because
    :func:`corpus_from_manifest` already refused a corpus whose declared order differs from the
    host's.
    """
    import numpy as np

    tokens = np.memmap(path, dtype=dtype, mode="r")
    windows = (tokens.size - 1) // seq_len
    for start in range(0, windows, micro):
        count = min(micro, windows - start)
        offsets, xs, ys = [], [], []
        for w in range(start, start + count):
            off = w * seq_len
            seg = np.asarray(tokens[off : off + seq_len + 1], dtype=np.int64)
            offsets.append(off)
            xs.append(seg[:-1])
            ys.append(seg[1:])
        yield offsets, np.stack(xs), np.stack(ys)


def _shard_microbatch_count(path, *, seq_len: int, micro: int, dtype) -> int:
    """How many micro-batches :func:`_shard_windows` will yield for this shard.

    Derived from the same two lines rather than by draining the generator, because the count
    is needed BEFORE the loop -- it is what every rank's step budget is reduced to, and a rank
    that ran a different number of forward passes than its peers is a hang, not a wrong number.
    """
    windows = (_shard_token_count(path, dtype=dtype) - 1) // seq_len
    return (windows + micro - 1) // micro


def fetch_slice_inputs(
    *,
    mask_uri: str,
    work_dir: str,
    rank: int = 0,
    world_size: int = 1,
    seq_len: int | None = None,
):
    """Download the frozen masks and the exact corpus shards they were built against.

    The masks live outside the corpus because they are not corpus data -- they are a derived
    labelling of it, written once by build_slice_masks.py and then read-only forever. Their
    manifest names both the shards and a SHA-256 prefix per mask, and this checks both, for
    one reason: every arm and seed must be scored on a byte-identical token set or the paired
    per-token difference the endpoint rests on is not paired, and a silently-substituted shard
    would still produce a plausible cross-entropy.

    Shard names encode their corpus location as ``<source>__<shard>.u32le.bin``, but the
    directory beneath the source is a topic that the name drops, so the corpus keys come from
    the manifest's own ``s3_key`` field rather than being reconstructed here.

    SHARDED BY OBJECT ACROSS RANKS, THE SAME WAY :func:`fetch_val_shards` IS, AND FOR THE SAME
    TWO REASONS. The download is the expensive part, so a set every rank fetches is
    ``world_size`` copies of the same bytes; and the assignment ``i % world_size == rank`` is
    deterministic from rank alone, so the union over ranks is exactly the manifest with nothing
    counted twice. That second property is what makes the summed band counts meaningful -- they
    can only add up to the whole labelled set if every object was read exactly once.

    The digest check therefore covers this rank's share only. Over the world it covers every
    mask exactly once, which is the same guarantee the unsharded version gave and at 1/world_size
    of the bytes.

    THE MANIFEST ITSELF IS READ BY EVERY RANK. It is one small JSON, and the band check below is
    the one thing that must not be delegated: a rank that skipped it would score bands whose bit
    layout it never verified.

    The defaults are the single-process case, so a caller with no process group -- a test, or a
    laptop -- gets the whole set exactly as before.
    """
    from olmo_core.io import copy_file, normalize_path

    base = normalize_path(mask_uri).rstrip("/")
    local = os.path.join(work_dir, "slice")
    os.makedirs(local, exist_ok=True)

    manifest_local = os.path.join(local, "slice_manifest.json")
    copy_file(f"{base}/slice_manifest.json", manifest_local, save_overwrite=True)
    with open(manifest_local) as handle:
        manifest = json.load(handle)

    if manifest.get("bands") != sorted(BAND_BIT):
        raise ValueError(
            f"mask bands {manifest.get('bands')} disagree with this build's {sorted(BAND_BIT)}"
        )

    # THE BIT LAYOUT, NOT JUST THE BAND NAMES. Two builds can agree on `bands` -- the sorted gap
    # distances -- and disagree on which BIT each one occupies, at which point every scored token is
    # attributed to the wrong band and the table is entirely plausible. The names check above cannot
    # see that, because it compares the keys and never the values.
    declared_bits = manifest.get("band_bit")
    # JSON keys are strings; BAND_BIT's are ints.
    if declared_bits is not None and {int(k): v for k, v in declared_bits.items()} != BAND_BIT:
        raise ValueError(
            f"mask band->bit layout {declared_bits} disagrees with this build's {BAND_BIT}"
        )

    # THE WINDOW THE MASKS WERE BUILT AT MUST BE THE WINDOW WE SCORE AT, and nothing else in this
    # function was checking it. A gap band is a distance measured INSIDE one evaluation window, so
    # masks built at 4096 and scored at 2048 attribute every token to a band computed against a
    # context the model never saw -- and produce a full, credible, wrong table rather than an error.
    # The builder writes this field; refuse rather than trust it by omission.
    declared_seq = manifest.get("sequence_length")
    if seq_len is not None and declared_seq is not None and int(declared_seq) != int(seq_len):
        raise ValueError(
            f"slice masks were built at sequence_length {declared_seq} but this run scores at "
            f"{seq_len}; every gap band would be measured against a context the model never saw"
        )

    val_paths, mask_paths = [], []
    for index, entry in enumerate(manifest["shards"]):
        if index % world_size != rank:
            continue
        # Index-prefixed local names, for the reason `fetch_val_shards` documents: the manifest's
        # `shard` field is a name rather than a key, so two entries that agree on it would have
        # the second copy_file silently overwrite the first -- dropping tokens in a way that looks
        # like a short mask set rather than like a collision.
        shard_local = os.path.join(local, f"{index:05d}-{entry['shard']}")
        mask_local = os.path.join(local, f"{index:05d}-{entry['mask']}")
        if "s3_key" not in entry:
            raise ValueError(
                f"manifest entry for {entry['shard']} has no s3_key; regenerate it with the "
                "shard-to-corpus mapping so the pairing is recorded rather than guessed"
            )
        copy_file(f"s3://edullm-data/{entry['s3_key']}", shard_local, save_overwrite=True)
        copy_file(f"{base}/{entry['mask']}", mask_local, save_overwrite=True)

        # Length is the cheap structural check; the digest is the one that catches a shard
        # that is the right size and the wrong content.
        if os.path.getsize(shard_local) // 4 != entry["tokens"]:
            raise ValueError(
                f"{entry['shard']}: {os.path.getsize(shard_local) // 4} tokens on disk, "
                f"manifest says {entry['tokens']}"
            )
        # THE PREFIX LENGTH COMES FROM THE MANIFEST, WHICH MEANS THE MANIFEST DECIDES HOW MUCH OF
        # ITSELF GETS CHECKED -- and an EMPTY string checks nothing at all: `digest[:0] == ""` is
        # True for any bytes on earth. That is fail-open on the one guard that makes every arm
        # provably scored on a byte-identical token set. Require a real digest before comparing.
        declared_digest = entry.get("sha256") or ""
        if len(declared_digest) < 16:
            raise ValueError(
                f"{entry['mask']}: manifest sha256 is {len(declared_digest)} chars "
                f"({declared_digest!r}); refusing, because a short or empty digest makes this "
                "check pass for arbitrary content"
            )
        with open(mask_local, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()[: len(declared_digest)]
        if digest != declared_digest:
            raise ValueError(f"{entry['mask']}: sha256 {digest} != manifest {declared_digest}")

        val_paths.append(shard_local)
        mask_paths.append(mask_local)

    total = sum(e["tokens"] for e in manifest["shards"])
    log.info(
        "slice inputs: rank %d holds %d of %d shard(s); %s tokens declared over the whole set, "
        "C_mass=%s, realized mass %.3f%%",
        rank,
        len(val_paths),
        len(manifest["shards"]),
        f"{total:,}",
        manifest.get("c_mass"),
        100 * manifest.get("realized_mass", 0.0),
    )
    return val_paths, mask_paths


def fetch_slice_inputs_on_every_rank(*, mask_uri: str, work_dir: str, seq_len: int | None = None):
    """Fetch this rank's share of the masks, and turn ONE rank's failure into ALL ranks failing.

    WITHOUT THIS THE RANKS SPLIT AND DEADLOCK, AND THE CALLER'S ``except`` IS WHAT CAUSES IT.
    The sliced eval is wrapped in a ``try`` that must not lose a checkpoint to a secondary bug.
    So a rank whose download raises is caught, skips :func:`evaluate_sliced` entirely and lands
    on the caller's ``barrier()`` -- while its peers, whose downloads succeeded, walk into the
    all-reduces inside the evaluator. Barrier on one side, all-reduce on the other, on the same
    process group: that is a mismatched collective, and it is a hang or an NCCL abort rather than
    an error anybody can read. The rank-zero gate was removed and this is the second trap behind
    it, one code block later -- the same shape of bug :func:`evaluate_val_aggregate` documents in
    its download guard.

    So the failure becomes a value, the value is all-reduced, and either every rank goes on or
    every rank raises. The all-reduce is entered unconditionally on both paths, which is what
    makes it safe.

    :returns: ``(val_paths, mask_paths)`` for this rank's share.

    :raises Refusal: On every rank, if any rank could not fetch its share.
    """
    rank = get_rank()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    failed, error = 0, None
    val_paths: list[str] = []
    mask_paths: list[str] = []
    try:
        val_paths, mask_paths = fetch_slice_inputs(
            mask_uri=mask_uri,
            work_dir=work_dir,
            rank=rank,
            world_size=get_world_size(),
            seq_len=seq_len,
        )
    except BaseException as exc:  # noqa: BLE001 -- re-raised below, on every rank
        failed, error = 1, exc
        log.error("rank %d could not fetch its slice inputs: %r", rank, exc)

    if int(all_reduce_value(failed, device, op=torch.distributed.ReduceOp.SUM)):
        if error is not None:
            raise Refusal(
                Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
                f"rank {rank} failed to fetch its share of the slice masks: "
                f"{type(error).__name__}: {error}",
            ) from error
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "another rank failed to fetch its share of the slice masks, so the per-band CE "
            "would be computed over an incomplete token set. Failing on every rank rather than "
            "waiting on a collective the failed rank will never enter.",
        )
    return val_paths, mask_paths


def band_ce_from_totals(sums: dict[int, float], counts: dict[int, int]) -> dict[str, Any]:
    """Turn all-reduced per-band ``(sum, count)`` pairs into token-weighted mean CEs.

    THE ONE PLACE A BAND'S MEAN IS FORMED, AND IT DIVIDES EXACTLY ONCE. The reduction this
    serves adds per-rank sums and per-rank counts separately and then calls this on the totals,
    which is a token-weighted mean over the whole world. The tempting alternative -- have each
    rank compute its own band CE and average those -- is a mean of means, and it weights every
    rank equally no matter how many tokens of that band the rank happened to hold. Bands are
    exactly where that bias bites hardest: ``gap>4096`` is rare, so one rank can hold most of it
    and another almost none, and the "average" would then be dominated by ranks with almost no
    evidence. It is a silent bias -- the number stays in range and no assertion fails.

    A BAND WITH NO TOKENS GETS ``None``, NOT ``0.0``, AND THAT IS THE WHOLE REASON THIS IS A
    FUNCTION. The version this replaces divided by ``max(count, 1)``, so an empty band reported
    a cross-entropy of ``0.0`` -- a PERFECT score, the best number in the table, for a band that
    was never measured. At the tail bands that is not hypothetical: a mask built on a small slice
    can easily leave ``gap>4096`` empty, and this study ranks arms on band CEs. A null says "not
    measured" and an arm that reports one cannot be silently ranked first on it.

    Pure arithmetic over plain numbers, deliberately: it takes no tensors, no model and no
    process group, so a CPU test can call it with hand-computed totals and check the weighting
    against an independently-derived answer.

    :param sums: Per-band summed cross-entropy, already reduced across ranks.
    :param counts: Per-band token counts, already reduced across ranks.

    :returns: ``{str(band): {"sum": ..., "n": ..., "ce": ...}}``, ``ce`` null where ``n`` is 0.
    """
    out: dict[str, Any] = {}
    for band in sorted(sums):
        total, n = float(sums[band]), int(counts[band])
        out[str(band)] = {
            "sum": total,
            "n": n,
            # Divided once, here, over the world's token count for this band.
            "ce": (total / n) if n > 0 else None,
        }
    return out


@torch.no_grad()
def evaluate_sliced(*, model, vocab_size, val_paths, mask_paths, seq_len, micro=2):
    """Aggregate and per-band AR-sliced CE over a fixed validation set, ON EVERY RANK.

    The per-band decomposition the gap-conditioned analysis needs: cross-entropy split by how
    far back the token's supporting context sits, using the frozen masks build_slice_masks.py
    wrote once. Read against :func:`evaluate_val_aggregate`, which is the headline endpoint;
    this is the secondary one and it answers a different question.

    Returns sums and counts rather than only means, so that arms can be differenced without a
    re-weighting error, and so an unequal token count between arms is visible rather than
    silently invalidating the pairing.

    The mask indexes the CONTINUATION token, so it aligns with the targets and is offset by
    one from the inputs. An off-by-one here scores the wrong positions and still produces
    plausible numbers.

    IT USED TO BE ``if get_rank() == 0:`` AND RUN 2 WOULD HAVE HUNG ON IT. That gate is the
    defect :func:`evaluate_val_aggregate`'s docstring describes at length, left in place only
    because no wave had passed ``--slice-mask-uri`` yet. Run 2 passes it on 8 GPUs. Under FSDP
    the parameters are sharded, so every forward issues all-gathers that every rank in the group
    must enter; rank zero alone entering them waits forever on peers that have already left, and
    the caller's ``except`` cannot catch a hang. The comment there conceded the point -- the
    barrier made it "survivable rather than correct". It is now correct: every rank fetches its
    share of the masks, every rank runs the same number of forwards, and the sums and counts are
    all-reduced. There is no rank gate anywhere in this function.

    THE REDUCTION IS TOKEN-WEIGHTED AND THAT IS NOT AUTOMATIC. Sums and counts are reduced
    SEPARATELY and divided once at the end, in :func:`band_ce_from_totals`. Reducing each rank's
    already-divided band CE would be a mean of means -- see that function for why bands are the
    worst place to make that mistake.

    Structurally a mirror of :func:`evaluate_val_aggregate` rather than a second pattern: same
    agreed step budget, same padding, same failure-is-a-value treatment, same order of
    collectives. Two evaluators that hold the ranks together two different ways is two things to
    get right, and the aggregate one is the version already proven against the thread harness.
    """
    import numpy as np

    rank, world_size = get_rank(), get_world_size()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # THE LOCAL STRUCTURAL CHECKS, AS A REDUCED VALUE RATHER THAN A BARE RAISE. Both compare
    # files this rank holds, so either can be true on ONE rank only -- and a lone `raise` here
    # would leave that rank unwinding while its peers enter the step-budget all-reduce below.
    # That is a mismatched collective, which is a hang rather than an error. So the reason is
    # recorded, the FLAG is all-reduced, and every rank refuses together.
    local_steps = 0
    local_problem: str | None = None
    if len(val_paths) != len(mask_paths):
        local_problem = (
            f"rank {rank} holds {len(val_paths)} shard(s) and {len(mask_paths)} mask(s); a "
            "shard scored against another shard's mask attributes every token to the wrong band"
        )
    else:
        for vp, mp in zip(val_paths, mask_paths):
            n_tokens = _shard_token_count(vp, dtype=np.uint32)
            mask_tokens = int(os.path.getsize(mp))  # one uint8 per token
            if n_tokens != mask_tokens:
                local_problem = f"mask/shard length mismatch for {vp}: {mask_tokens} vs {n_tokens}"
                break
            local_steps += _shard_microbatch_count(
                vp, seq_len=seq_len, micro=micro, dtype=np.uint32
            )

    if int(all_reduce_value(1 if local_problem else 0, device, op=torch.distributed.ReduceOp.SUM)):
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            local_problem
            or (
                "another rank's slice masks do not line up with its shards, so the per-band CE "
                "would attribute tokens to the wrong bands. Failing on every rank rather than "
                "waiting on a collective the refusing rank will never enter."
            ),
        )

    # EVERY RANK RUNS THE SAME NUMBER OF FORWARD PASSES. A rank that leaves the loop early
    # reaches the all-reduce below while its peers are still inside an all-gather, which is a
    # hang at the end of a paid-for run. See `evaluate_val_aggregate` for the full argument; the
    # budget is the max over ranks and short ranks push discarded filler batches.
    steps = int(all_reduce_value(local_steps, device, op=torch.distributed.ReduceOp.MAX))
    log.info(
        "sliced eval: rank %d has %d local micro-batch(es) over %d shard(s), %d agreed across "
        "%d rank(s)",
        rank,
        local_steps,
        len(val_paths),
        steps,
        world_size,
    )
    if steps == 0:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            f"the slice shards yielded no window of {seq_len} tokens on any rank, so there is "
            "nothing to score. A shard shorter than one sequence, or a sequence_length larger "
            "than the shards, are the two ways to get here.",
        )

    rows = max(micro, 1)
    filler = np.zeros((rows, seq_len), dtype=np.int64)

    def pad(batch):
        """Grow a short micro-batch to `rows` by repeating its first row."""
        if batch.shape[0] == rows:
            return batch
        extra = np.repeat(batch[:1], rows - batch.shape[0], axis=0)
        return np.concatenate([batch, extra], axis=0)

    was_training = model.training
    model.eval()
    agg_sum, agg_n = 0.0, 0
    band_sum = {b: 0.0 for b in BAND_BIT}
    band_n = {b: 0 for b in BAND_BIT}
    done = 0
    compute_error: BaseException | None = None
    try:
        try:
            for vp, mp in zip(val_paths, mask_paths):
                mask = np.memmap(mp, dtype=np.uint8, mode="r")
                # Windowing goes through the shared generator rather than a second copy of the
                # same arithmetic, so this and the aggregate evaluator cannot drift on the
                # off-by-one. `offsets` is what aligns the mask to the TARGETS.
                for offsets, xs, ys in _shard_windows(
                    vp, seq_len=seq_len, micro=micro, dtype=np.uint32
                ):
                    # `offsets` covers the REAL rows only, so the mask block and the sliced CE
                    # below are both exactly `len(offsets) * seq_len` long and cannot include a
                    # padded row. That is what keeps the band counts exact under padding.
                    real = len(offsets) * seq_len
                    ms = np.stack(
                        [
                            np.asarray(mask[off + 1 : off + seq_len + 1], dtype=np.uint8)
                            for off in offsets
                        ]
                    )
                    flat = torch.from_numpy(ms).to(device).reshape(-1)
                    ce = _forward_ce(model, pad(xs), pad(ys), vocab_size=vocab_size, device=device)[
                        :real
                    ]
                    agg_sum += float(ce.sum())
                    agg_n += ce.numel()
                    for band, bit in BAND_BIT.items():
                        selected = (flat & bit) != 0
                        if selected.any():
                            band_sum[band] += float(ce[selected].sum())
                            band_n[band] += int(selected.sum())
                    done += 1
        except BaseException as exc:  # noqa: BLE001 -- re-raised on every rank below
            compute_error = exc
            log.error("rank %d failed during the sliced forward pass: %r", rank, exc)

        # THE PADDING PASSES. Real collective traffic, discarded arithmetic -- see
        # `evaluate_val_aggregate`. A rank that ABORTED also lands here, with more to pad, which
        # is what keeps its peers unblocked long enough to be told.
        while done < steps:
            try:
                _forward_ce(model, filler, filler, vocab_size=vocab_size, device=device)
            except BaseException as exc:  # noqa: BLE001 -- see above
                compute_error = compute_error or exc
            done += 1
    finally:
        if was_training:
            model.train()

    # The failure flag goes FIRST, so a rank that broke does not contribute its partial sums to
    # a number anyone might use.
    if int(all_reduce_value(1 if compute_error else 0, device, op=torch.distributed.ReduceOp.SUM)):
        if compute_error is not None:
            raise Refusal(
                Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
                f"rank {rank} failed during the sliced forward pass: "
                f"{type(compute_error).__name__}: {compute_error}",
            ) from compute_error
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "another rank failed during the sliced forward pass, so the per-band CE would be "
            "computed over an incomplete token set.",
        )

    # SUMS AND COUNTS, SEPARATELY, THEN DIVIDED ONCE. Never a mean of per-rank means -- see
    # `band_ce_from_totals`. Called unconditionally: `all_reduce_value` is a no-op off a process
    # group, so there is no world_size branch that could be right in one topology and a hang in
    # the other. `sorted` fixes the order, because every rank must enter these collectives in
    # the SAME order and dict order is the wrong thing to rest that on.
    agg_sum = float(all_reduce_value(agg_sum, device, op=torch.distributed.ReduceOp.SUM))
    agg_n = int(all_reduce_value(agg_n, device, op=torch.distributed.ReduceOp.SUM))
    for band in sorted(BAND_BIT):
        band_sum[band] = float(
            all_reduce_value(band_sum[band], device, op=torch.distributed.ReduceOp.SUM)
        )
        band_n[band] = int(
            all_reduce_value(band_n[band], device, op=torch.distributed.ReduceOp.SUM)
        )
    barrier()

    if agg_n <= 0:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "the sliced evaluation summed to zero tokens across all ranks",
        )

    return {
        "aggregate": {
            "sum": agg_sum,
            "n": agg_n,
            # `agg_n > 0` is guaranteed by the refusal above, so this division is safe and is
            # not hiding an empty set behind a `max(n, 1)`.
            "ce": agg_sum / agg_n,
        },
        "bands": band_ce_from_totals(band_sum, band_n),
        "world_size": world_size,
    }


def fetch_val_shards(*, val_paths: list[str], work_dir: str, rank: int, world_size: int):
    """Download this rank's slice of the corpus's own held-out objects.

    Sharded by object rather than by window, because the download is the expensive part and a
    shard every rank fetches is world_size copies of the same bytes. The assignment is
    ``i % world_size``, which is deterministic from rank alone -- so no rank needs to be told
    what any other rank got, and the union over ranks is exactly ``val_paths`` with nothing
    counted twice. That property is what makes the summed token count checkable: it can only
    equal the declared total if every object was read exactly once.

    ``val_paths`` are the URIs the reader resolved from the manifest. They are NOT rebuilt from
    shard filenames: ``val-00212.u32le.bin`` appears under 24 different topic directories in
    olmo-150b-dolma2, so a key reconstructed from a name fetches a readable shard of the wrong
    topic and every number downstream is plausible and wrong.
    """
    from olmo_core.io import copy_file

    local = os.path.join(work_dir, "val")
    os.makedirs(local, exist_ok=True)

    mine = [(i, uri) for i, uri in enumerate(val_paths) if i % world_size == rank]
    fetched = []
    for i, uri in mine:
        # Indexed local name: two topics genuinely contain a `val-00212.u32le.bin`, so the
        # basename alone is not unique and the second copy_file would silently overwrite the
        # first -- halving the token count in a way that looks like a short corpus.
        target = os.path.join(local, f"{i:05d}-{os.path.basename(normalize_path(uri))}")
        copy_file(uri, target, save_overwrite=True)
        fetched.append(target)
    log.info("rank %d holds %d of %d held-out object(s)", rank, len(fetched), len(val_paths))
    return fetched


@torch.no_grad()
def evaluate_val_aggregate(
    *,
    model,
    vocab_size: int,
    val_paths: list[str],
    work_dir: str,
    seq_len: int,
    dtype,
    declared_tokens: int | None = None,
    micro: int = 2,
):
    """Held-out cross-entropy over the corpus's OWN ``val`` partition, on EVERY rank.

    THE ENDPOINT. Everything the experiment is for is a difference of this number between arms,
    so a run that trained and produced no CE produced a checkpoint that cannot answer the
    question. This is the version that can actually produce one.

    WHY THE RANK-ZERO VERSION COULD NOT, AND WHY ITS ``except`` DID NOT SAVE IT. Under FSDP the
    parameters are sharded, so a forward pass issues all-gathers -- collectives that every rank
    in the group must enter. ``if get_rank() == 0: forward()`` has rank 0 enter a collective the
    other ranks never reach, because they have already returned from ``train()`` and called
    ``destroy_process_group()``. That is a HANG, and a hang is not an exception: the ``except
    Exception`` around it never runs, NCCL's watchdog eventually aborts the job, and what the
    platform records is a run that trained, wrote a checkpoint, exited, and reported no CE.
    Every rank is in the compute path here, and there is no rank gate anywhere in it.

    WHY IT RETURNS SUMS AND COUNTS RATHER THAN A MEAN. Two arms are differenced, and a
    difference of two means computed over different token counts is not a paired difference. The
    counts make an unequal denominator visible instead of letting it silently re-weight the
    contrast.

    TWO TOKEN COUNTS, AND THE REASON THERE ARE TWO IS SO ONE OF THEM CAN BE CHECKED EXACTLY.
    ``tokens_present`` is measured from the SIZE ON DISK of every object that was read, summed
    over ranks; it must equal the ``229,894,171`` the manifest declares for
    olmo-150b-dolma2-v1, exactly, with no tolerance -- a shard that failed to download, two
    shards overwriting one local filename, or a key reconstructed to the wrong topic all break
    that equality. ``tokens`` is what cross-entropy actually covered, which is necessarily
    smaller: fixed-length windowing cannot score the tail of a shard that does not fill a whole
    window. Asserting the exact number against the SCORED count would refuse every correct run,
    which is how an assertion gets deleted instead of fixed; asserting it against the PRESENT
    count catches the whole class of wrong-bytes failures with no slack at all. See
    :func:`assert_val_tokens_account_for_the_corpus`.

    WHY IT RETURNS SUMS AND COUNTS RATHER THAN A MEAN. Two arms are differenced, and a
    difference of two means computed over different token counts is not a paired difference. The
    counts make an unequal denominator visible instead of letting it silently re-weight the
    contrast.

    Raises rather than returning a partial result. A silent null endpoint is the failure this
    function exists to remove, so it does not get to reappear as a zero token count.
    """
    rank, world_size = get_rank(), get_world_size()
    if not val_paths:
        raise Refusal(
            Stage.THE_CORPUS_DECLARES_NO_HELD_OUT_SPLIT,
            "the corpus declares no held-out partition, so there is no set to score the arm "
            "on. Every contrast this experiment reports is a difference of held-out CE; a run "
            "that can only produce a training curve cannot contribute one.",
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # BEFORE the download, so no rank starts reading while another is still deciding what to
    # read, and so the collectives below are entered from a known common point.
    barrier()

    # ONE RANK FAILING ITS DOWNLOAD MUST NOT BECOME A HANG, AND WITHOUT THIS IT WOULD BE ONE.
    # A raise here propagates out of `train()` on that rank alone; the others carry on into the
    # all_reduce below and wait on a participant that is already unwinding towards
    # `destroy_process_group`. So the failure is turned into a value, all-reduced, and every rank
    # raises together. This is a narrow catch that re-raises rather than a swallow: the run still
    # dies, it just dies on all ranks at once and says which rank started it.
    failed, error = 0, None
    local_paths: list[str] = []
    try:
        local_paths = fetch_val_shards(
            val_paths=val_paths, work_dir=work_dir, rank=rank, world_size=world_size
        )
    except BaseException as exc:  # noqa: BLE001 -- re-raised below, on every rank
        failed, error = 1, exc
        log.error("rank %d could not fetch its held-out shards: %r", rank, exc)
    if int(all_reduce_value(failed, device, op=torch.distributed.ReduceOp.SUM)):
        if error is not None:
            # read_failure already separates "the role may not read it" from "it is not there",
            # which are the two things worth telling apart here and are an IAM change and a
            # dataset problem respectively.
            raise Refusal(
                read_failure(error),
                f"rank {rank} failed to fetch its share of the held-out objects: "
                f"{type(error).__name__}: {error}",
            ) from error
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "another rank failed to fetch its share of the held-out objects, so the held-out "
            "CE would be computed over an incomplete token set. Failing on every rank rather "
            "than waiting on a collective the failed rank will never enter.",
        )

    # EVERY RANK RUNS THE SAME NUMBER OF FORWARD PASSES, AND THAT IS A CORRECTNESS PROPERTY
    # RATHER THAN A TIDINESS ONE. 60 objects over 8 ranks is 8 for some ranks and 7 for others;
    # a rank with fewer shards would leave the loop early, reach the all_reduce below while its
    # peers are still inside a forward, and every one of those forwards is an all-gather that
    # now has a missing participant. The result is a hang, at the end of a run, after the
    # money is spent.
    #
    # So the budget is agreed first: max over ranks of the local micro-batch count. A rank that
    # runs out of real data pushes a discarded filler batch through the model instead, which
    # enters exactly the same collectives -- the whole point -- and contributes nothing to the
    # sums, so the token count stays exact.
    local_steps = sum(
        _shard_microbatch_count(p, seq_len=seq_len, micro=micro, dtype=dtype) for p in local_paths
    )
    # Measured from the size on disk of what actually arrived, NOT from the manifest. This is
    # the number checked for exact equality against the declared count, so it has to come from
    # the bytes rather than from the claim it is checking.
    local_present = sum(_shard_token_count(p, dtype=dtype) for p in local_paths)
    steps = int(all_reduce_value(local_steps, device, op=torch.distributed.ReduceOp.MAX))
    log.info(
        "rank %d: %d local micro-batch(es) over %s local token(s), %d agreed across %d rank(s)",
        rank,
        local_steps,
        f"{local_present:,}",
        steps,
        world_size,
    )
    if steps == 0:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            f"the held-out objects yielded no window of {seq_len} tokens on any rank, so "
            "there is nothing to score. A shard shorter than one sequence, or a "
            "sequence_length larger than the shards, are the two ways to get here.",
        )

    # EVERY FORWARD PASS ON EVERY RANK IS THE SAME SHAPE, `(micro, seq_len)`, AND THAT IS A
    # STRONGER PROPERTY THAN IT LOOKS.
    #
    # `_shard_windows` yields a RAGGED last micro-batch: a shard with an odd window count ends
    # in a `(1, seq_len)` batch where its peers are `(2, seq_len)`. So without this padding,
    # rank A's k-th pass and rank B's k-th pass could differ in batch size. Under the FSDP2
    # `fully_shard` this model uses that happens to be survivable -- its all-gathers are over
    # PARAMETER shards, whose shapes do not depend on the batch -- but "happens to be
    # survivable under the parallelism we currently configure" is not a property to rest an
    # eleven-hour run on. It would break under FSDP1, under context parallelism (which shards
    # by sequence), and it makes `compile_model=True` recompile per distinct shape.
    #
    # So short batches are padded to `micro` rows and the padding is excluded from the sums by
    # count rather than by masking arithmetic -- `_forward_ce` returns per-token CE in row
    # order, so the real tokens are exactly the first `rows * seq_len` of it. A rank with NO
    # shards pads with an entirely synthetic batch, which is reachable whenever
    # `world_size > len(val_paths)`: a 64-GPU shape, or a corpus with three val objects. An
    # earlier draft replayed a rank's LAST real batch instead, which a rank with no batches does
    # not have -- a hang on exactly the shape that is easiest to submit by accident.
    import numpy as np

    rows = max(micro, 1)
    filler = np.zeros((rows, seq_len), dtype=np.int64)

    def pad(batch):
        """Grow a short micro-batch to `rows` by repeating its first row."""
        if batch.shape[0] == rows:
            return batch
        extra = np.repeat(batch[:1], rows - batch.shape[0], axis=0)
        return np.concatenate([batch, extra], axis=0)

    # THE FORWARD LOOP GETS THE SAME FAILURE-IS-A-VALUE TREATMENT AS THE DOWNLOAD, AND FOR THE
    # SAME REASON. A corrupt memmap or a single-card OOM raises on ONE rank; letting that
    # propagate straight out unwinds that rank while its peers are still inside the loop, and
    # they then walk into an all_reduce it will never enter. The download was guarded and this
    # was not -- the same deadlock, one code block later. Every rank still runs `steps`
    # forwards; a rank that broke early pads out the remainder so the collective structure is
    # unchanged, then all ranks agree to raise together below.
    was_training = model.training
    model.eval()
    ce_sum, n_tokens, done = 0.0, 0, 0
    compute_error: BaseException | None = None
    try:
        try:
            for path in local_paths:
                for _, xs, ys in _shard_windows(path, seq_len=seq_len, micro=micro, dtype=dtype):
                    real = xs.shape[0] * seq_len
                    ce = _forward_ce(model, pad(xs), pad(ys), vocab_size=vocab_size, device=device)
                    # Only the real rows count. `ce` is flat in row-major order, so the padding
                    # is the tail and slicing it off is exact rather than approximate.
                    ce_sum += float(ce[:real].sum())
                    n_tokens += real
                    done += 1
        except BaseException as exc:  # noqa: BLE001 -- re-raised on every rank below
            compute_error = exc
            log.error("rank %d failed during the held-out forward pass: %r", rank, exc)

        # THE PADDING PASSES. Real collective traffic, discarded arithmetic. Without them a rank
        # that got 7 shards where its peer got 8 leaves the loop one forward early, reaches the
        # all_reduce below while the peer is still inside an all-gather, and the job hangs at
        # the very end of a run that has already been paid for. A rank that ABORTED also lands
        # here, with more to pad, which is what keeps its peers unblocked long enough to be told.
        while done < steps:
            try:
                _forward_ce(model, filler, filler, vocab_size=vocab_size, device=device)
            except BaseException as exc:  # noqa: BLE001 -- see above
                # A filler pass that also fails means this rank cannot participate at all.
                # Nothing further can be done here; the count still advances so the loop ends
                # and the reduction below can tell the other ranks.
                compute_error = compute_error or exc
            done += 1
    finally:
        # Restored even on the way out through an exception, so a failure that is logged and
        # re-raised does not leave the module in eval mode for whatever runs next.
        if was_training:
            model.train()

    # THE ALL-REDUCE. Sums and counts, not means: a mean-of-means is weighted by whichever rank
    # happened to get the larger shards. Called unconditionally -- `all_reduce_value` is a no-op
    # off a process group, so there is no world_size branch here that could be right in one
    # topology and a hang in the other.
    #
    # The failure flag goes FIRST, so a rank that broke does not contribute its partial sums to
    # a number anyone might use.
    if int(all_reduce_value(1 if compute_error else 0, device, op=torch.distributed.ReduceOp.SUM)):
        if compute_error is not None:
            raise Refusal(
                Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
                f"rank {rank} failed during the held-out forward pass: "
                f"{type(compute_error).__name__}: {compute_error}",
            ) from compute_error
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "another rank failed during the held-out forward pass, so the CE would be computed "
            "over an incomplete token set. Failing on every rank rather than returning a "
            "number that is quietly missing a rank's share.",
        )

    ce_sum = float(all_reduce_value(ce_sum, device, op=torch.distributed.ReduceOp.SUM))
    n_tokens = int(all_reduce_value(n_tokens, device, op=torch.distributed.ReduceOp.SUM))
    tokens_present = int(all_reduce_value(local_present, device, op=torch.distributed.ReduceOp.SUM))
    barrier()

    if n_tokens <= 0:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            "the held-out evaluation summed to zero tokens across all ranks",
        )

    result = {
        "ce": ce_sum / n_tokens,
        "sum": ce_sum,
        # Tokens cross-entropy actually covered. Always below `tokens_present` by the per-shard
        # window remainder.
        "tokens": n_tokens,
        # Tokens that were THERE, measured from bytes on disk. The one checked exactly.
        "tokens_present": tokens_present,
        "declared_tokens": declared_tokens,
        "shards": len(val_paths),
        "seq_len": seq_len,
        "world_size": world_size,
        # present - scored: the tail of each shard that cannot fill a whole window. Reported so
        # the accounting is readable in the JSON rather than needing to be recomputed.
        "unscored": tokens_present - n_tokens,
    }
    log.info(
        "held-out CE %.4f over %s scored token(s) of %s present (%s declared), %s unscored tail",
        result["ce"],
        f"{n_tokens:,}",
        f"{tokens_present:,}",
        "unknown" if declared_tokens is None else f"{declared_tokens:,}",
        f"{tokens_present - n_tokens:,}",
    )
    return result


def _forward_ce(model, xs, ys, *, vocab_size: int, device: torch.device):
    """One micro-batch through the model, returning per-token CE.

    Shared by the padding passes and the real ones so a pad is bit-for-bit the same collective
    traffic as the work it stands in for -- a "padding" pass that skipped the forward would
    skip the all-gathers, which is the hang it exists to prevent.
    """
    x = torch.from_numpy(xs).to(device)
    y = torch.from_numpy(ys).to(device)
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out
    else:
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out
    return _chunked_ce(logits.reshape(-1, vocab_size), y.reshape(-1))


def assert_val_tokens_account_for_the_corpus(result) -> None:
    """Refuse a held-out CE that was not computed over the whole declared partition.

    A MAGNITUDE CHECK, NOT AN EXISTENCE CHECK, and that distinction is the reason this exists at
    all. ``val_ce is not None`` passes for a CE over one shard out of sixty, over the wrong
    topic's shards, or over a val set halved by two objects landing on one local filename. All
    three produce a number in the normal range, and a number in the normal range is what a
    reader of the JSON will believe. This project has five documented green-but-meaningless
    harness results, and the only check that ever caught uninitialised weights was asserting
    loss ~ ln(vocab).

    THE EXACT ONE IS ON ``tokens_present``, WHICH IS WHY THAT FIELD EXISTS. Bytes on disk over
    every object every rank read, summed -- it must equal the manifest's declared row count with
    no tolerance whatsoever: 229,894,171 for olmo-150b-dolma2-v1. Nothing legitimate makes it
    differ. A shard that failed to download, a duplicate local filename, a key rebuilt to the
    wrong topic and a glob that matched a subset all break it by millions of tokens.

    THE SCORED COUNT IS BOUNDED RATHER THAN EQUAL, AND THAT IS NOT A WEAKER CHECK BEING
    SUBSTITUTED. Fixed-length windowing genuinely cannot score the tail of a shard that does not
    fill a whole window, so scored < present always, by at most ``seq_len`` per shard plus the
    shifted target. Asserting equality there would refuse every correct run, which is how an
    assertion gets deleted instead of fixed. The exact check above is what catches the failure
    class; this one catches a windowing bug that dropped far more than the tail.
    """
    declared = result.get("declared_tokens")
    present = result["tokens_present"]
    scored = result["tokens"]
    n_shards = result["shards"]
    seq_len = result["seq_len"]

    if declared is None:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            f"held-out CE was computed over {scored:,} tokens but the manifest declares no row "
            "count for the partition, so there is nothing to check it against. An unchecked "
            "token count is how a CE over a quarter of the val set gets recorded as the "
            "endpoint and believed.",
        )
    if present != declared:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            f"the held-out objects that were read hold {present:,} tokens; the manifest "
            f"declares {declared:,} for the partition, a difference of {present - declared:+,}. "
            "This is measured from bytes on disk over every object every rank fetched, and "
            "nothing legitimate makes it differ -- a shard that failed to download, two shards "
            "written to one local name, a key rebuilt to the wrong topic, or a glob that "
            "matched a subset are what get here. The CE would look completely normal either "
            "way, which is why this is checked rather than reported. "
            # THE ONE FALSE-POSITIVE THIS CAN PRODUCE, NAMED IN THE MESSAGE RATHER THAN LEFT
            # FOR SOMEBODY TO REDISCOVER AT 3AM. `rows` is the manifest's count for the
            # partition and the standard's unit vocabulary is {rows, tokens, items, indices,
            # bytes}. Only `tokens` and `indices` are fixed-width, and this compares against a
            # TOKEN count -- so a held-out partition declaring `items` (documents, say) would
            # trip this on every correct run. The reader does not expose the unit through
            # `split_rows`, so it cannot be checked here; olmo-150b-dolma2-v1 declares tokens,
            # and a corpus that does not needs the unit plumbed through rather than this
            # assertion relaxed.
            "If this fires on a corpus you believe is intact, check what unit the partition's "
            "count is declared in: this compares against tokens, and a partition counted in "
            "items or bytes is a different quantity rather than a wrong one.",
        )
    slack = n_shards * (seq_len + 1)
    if not 0 <= present - scored <= slack:
        raise Refusal(
            Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING,
            f"{present:,} held-out tokens were present and {scored:,} were scored, leaving "
            f"{present - scored:,} unaccounted. Windowing can only drop the tail of each shard, "
            f"at most {slack:,} tokens over {n_shards} shard(s) at sequence length {seq_len}, "
            "so a larger gap is a windowing bug rather than a remainder.",
        )


def throughput_report(
    losses: LossWatcher,
    *,
    world_size: int,
    wall_clock_seconds: float | None,
    flops_per_token: int | None = None,
    device_name: str | None = None,
    warmup_steps: int = WARMUP_STEPS_EXCLUDED,
) -> dict[str, Any]:
    """The speed half of the record: two throughput figures that can never be confused.

    TWO FIGURES, NAMED SO THAT NOBODY HAS TO REMEMBER WHICH IS WHICH.

    ``throughput_tok_s_steady`` -- the one to rank arms on. Tokens over seconds across the steps
    AFTER ``warmup_steps``, so process start, dataset open, FSDP wrap, Dynamo tracing, Inductor
    compilation and the allocator's pool growth are all outside the measurement.

    ``throughput_tok_s_whole_run`` -- the one to cost a machine-hour on and NOT to compare arms
    with. Tokens over the entire wall clock of ``fit()``, fixed costs included. It is here for
    contrast and because the gap between the two is itself diagnostic: a run whose steady figure
    is far above its whole-run figure spent a lot of its life not training.

    THE GAP BETWEEN THEM IS NOT SMALL AND IS NOT A CONSTANT. On this stack whole-run wall clock
    read 3.1x low on a short probe, and the steady-state figure itself moved 1.5x between a
    20-step and a 200-step run because the averaging window was contaminated. Two arms measured
    at two run lengths would differ by that much with identical hardware behaviour, which is why
    the cutoff is a module constant and both figures ship with the counts behind them.

    EVERY FIGURE IS None WHEN IT COULD NOT BE MEASURED. Not 0.0, not the previous cell's value.
    A zero throughput and an unmeasured throughput are opposite claims and only one is ever true.
    """
    steady_samples = steps_after_warmup(losses.steps, warmup_steps=warmup_steps)
    steady_total = throughput_tokens_per_second(steady_samples)
    # Same arithmetic over every recorded step, warmup included. Distinct from the wall-clock
    # figure below: this one still excludes whatever happened before the first post_step.
    all_steps_total = throughput_tokens_per_second(losses.steps)

    step_seconds = [sample.seconds for sample in steady_samples]
    tokens_in_steady = sum(sample.tokens for sample in steady_samples)

    # Tokens over the WHOLE of fit(), which is the pessimistic figure. Derived from the same
    # token total the trainer counted, not from a step count times a batch size.
    tokens_all = sum(sample.tokens for sample in losses.steps)
    whole_run_total: float | None = None
    if wall_clock_seconds and wall_clock_seconds > 0 and tokens_all > 0:
        whole_run_total = tokens_all / wall_clock_seconds

    # Per device from total, dividing by the world size that is reported beside it. A divisor a
    # reader cannot see is how two arms measured on two shapes get compared as though on one.
    divisor = world_size if world_size and world_size > 0 else None
    steady_per_device = None if steady_total is None or divisor is None else steady_total / divisor
    whole_run_per_device = (
        None if whole_run_total is None or divisor is None else whole_run_total / divisor
    )

    device_peak = device_peak_bf16_flops(device_name)
    mfu = model_flops_utilisation(
        tokens_per_second_per_device=steady_per_device,
        flops_per_token=flops_per_token,
        device_peak_flops=device_peak,
    )
    if mfu is None:
        # WHY THE MFU IS ABSENT, SAID IN THE RECORD RATHER THAN LEFT TO BE GUESSED. A null with
        # no reason is indistinguishable from a bug in this function, and the three causes want
        # different fixes: an unlisted card needs a line in DEVICE_PEAK_BF16_FLOPS, a missing
        # flops/token needs the model to implement it, and a missing throughput means the run
        # produced no usable steps at all.
        if device_peak is None:
            mfu_basis = f"no peak bf16 FLOP/s entry for device {device_name!r}"
        elif not flops_per_token:
            mfu_basis = "the train module reported no FLOPs per token"
        else:
            mfu_basis = "no steady-state throughput to compute it from"
    else:
        mfu_basis = (
            f"{steady_per_device:,.0f} tok/s/device * {flops_per_token:,} FLOP/token "
            f"/ {device_peak:,} FLOP/s peak dense bf16 on {device_name}"
        )

    return {
        # THE FIGURE TO RANK ARMS ON. Post-warmup, total across every device.
        "throughput_tok_s_steady": steady_total,
        "throughput_tok_s_steady_per_device": steady_per_device,
        # THE FIGURE TO COST WALL CLOCK ON, AND NOT TO RANK ARMS ON. Fixed costs included.
        "throughput_tok_s_whole_run": whole_run_total,
        "throughput_tok_s_whole_run_per_device": whole_run_per_device,
        # Every recorded step including warmup, over summed step time. Between the two above,
        # and its distance from `_steady` is how much the warmup was worth.
        "throughput_tok_s_all_steps": all_steps_total,
        # The counts behind the numbers, so a figure computed over four steps is visibly that.
        "steps_measured": len(losses.steps),
        "steady_state_steps": len(steady_samples),
        "warmup_steps_excluded": warmup_steps,
        "tokens_in_steady_window": tokens_in_steady if steady_samples else None,
        # Step-time distribution over the steady window only. p50 and p90 rather than a mean:
        # the mean is already implied by the throughput, and what a scheduler needs from the
        # tail is a step time that was actually observed.
        "step_time_s_p50": quantile_nearest_rank(step_seconds, 0.5),
        "step_time_s_p90": quantile_nearest_rank(step_seconds, 0.9),
        # Summed step time over the steady window. Wall clock MINUS startup, which is the
        # quantity a per-cell schedule wants; `seconds` at the top level is the whole of fit().
        "steady_window_seconds": sum(step_seconds) if step_seconds else None,
        "training_seconds_excluding_startup": (
            sum(sample.seconds for sample in losses.steps) if losses.steps else None
        ),
        "mfu_pct": mfu,
        "mfu_basis": mfu_basis,
        "device_peak_bf16_flops": device_peak,
        "flops_per_token": flops_per_token,
    }


# --- the decode / inference measurement -------------------------------------------------------
#
# WHY THIS EXISTS AT ALL. Run 1 measured held-out CE, training throughput and peak training
# memory, and NOTHING about inference. The entire practical argument for a linear-attention mixer
# is that generation is cheap: the recurrent state is a fixed size, where softmax attention's KV
# cache grows with the context. That is the claim this bake-off is deciding on, and run 1 never
# ran the code path it lives in. For a production mixer choice it is the number that matters most.
#
# WHAT IS MEASURED, AND WHY IT IS THE OPERATOR RATHER THAN THE MODEL. Three options were weighed:
#
#   (a) OPERATOR MICROBENCHMARK -- time `fused_recurrent_*` directly at the run's head geometry,
#       with a threaded state, at several batch sizes. What is implemented.
#   (b) WHOLE-MODEL PREFILL+DECODE via repeated full forwards. Rejected, and not merely as
#       expensive: our mixers have no incremental path (see below), so "decode" would re-run the
#       whole prefix for every token. That is O(T^2) recompute, it is dominated by work a real
#       server does not do, and -- decisively -- it would penalise the linear-attention arms for
#       lacking the very state reuse the measurement is supposed to demonstrate. It would
#       UNDERSTATE the advantage being measured, and produce a number that looks like a serving
#       figure. A misleading measurement is worse here than a narrow one.
#   (c) WIRE A REAL `step()` AND THREAD STATE THROUGH THE MODEL. The honest whole-model
#       measurement, and the right follow-on. It is a FEATURE, not a measurement: all four mixer
#       classes take a `cache` argument and explicitly discard it (`del layer_idx, n_layers,
#       cache  # Unused` in recurrent.py), so there is no `step()` and no state threading anywhere
#       in the model today. Building one is not in scope for run 2 and would touch four operator
#       classes on a branch five agents are editing.
#
# So (a): the primitive `fla` already ships -- `fused_recurrent_kda` and `fused_recurrent_gdn2`,
# both of which take `initial_state` and `output_final_state` -- is driven directly, one token at
# a time, with the state threaded from each step into the next. That is exactly the arithmetic a
# real decoder would do in the mixer, at the real geometry, so the per-token latency and the state
# footprint are real. What it is NOT is a whole-model serving number, and `decode_basis` says so
# in the record rather than in a comment nobody reads.
#
# THE NUMBER THAT ACTUALLY DECIDES SERVING BATCH SIZE is `decode_state_bytes_per_seq`, and it is
# computed rather than timed -- so it is reported even when no GPU was available. It is the whole
# point of linear attention: a fixed cost per sequence instead of one that grows per token.

#: Batch sizes the decode probe sweeps. 1 is the latency-bound single-stream case a chat serving
#: path lives in; 32 is throughput-bound where the kernel launch is amortised. A single batch size
#: would answer only one of two different production questions.
DECODE_BATCH_SIZES = (1, 8, 32)

#: Tokens generated per timed measurement, after the warmup below. One token is a kernel-launch
#: measurement, not a decode measurement -- at these state sizes a single fused step is tens of
#: microseconds, which is the same order as the launch overhead and the timer's own resolution.
DECODE_STEPS = 64

#: Decode steps run and discarded before timing starts. Triton compiles on first call and the
#: caching allocator is still growing; both land entirely in step 1 and would otherwise dominate a
#: 64-step median. This is the same reasoning as WARMUP_STEPS_EXCLUDED, at decode's scale.
DECODE_WARMUP_STEPS = 8


def recurrent_state_bytes(
    *,
    n_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    n_layers: int,
    bytes_per_element: int = 4,
) -> int:
    """Bytes of recurrent state one sequence holds, across every layer carrying the mixer.

    THE FIELD THAT DECIDES SERVING BATCH SIZE, AND THE WHOLE POINT OF LINEAR ATTENTION. A KV
    cache costs bytes per TOKEN; this costs bytes per SEQUENCE and does not move as the context
    grows. Concurrency on a fixed card is (memory - weights) / this.

    The state is one ``(head_k_dim, head_v_dim)`` matrix per head per layer -- the outer-product
    accumulator the delta rule writes into -- so it is ``n_heads * head_k_dim * head_v_dim``
    elements per layer. That matches the ``state_size`` both operators' own
    ``num_flops_per_token`` uses, deliberately: two different state sizes in one file is how a
    memory figure and a FLOP figure come to describe different models.

    ``bytes_per_element`` DEFAULTS TO 4, NOT 2, AND THAT IS NOT CONSERVATISM. ``fla`` keeps the
    recurrent state in float32 regardless of the input dtype -- the state accumulates over
    thousands of steps and bf16 would drift -- so the state of a bf16 model is still fp32. Passing
    2 here because "the run is bf16" would report half the real footprint, on the field somebody
    sizes a serving fleet with. The realised dtype is asserted against this in the probe.

    :param n_heads: Value heads carrying state (``n_v_heads`` where GVA applies).
    :param head_k_dim: The key-side head dimension.
    :param head_v_dim: The value-side head dimension (``head_dim * expand_v``).
    :param n_layers: How many layers carry this mixer -- 2 here, not 16. See the note in
        :func:`decode_report`.
    :param bytes_per_element: 4 for fp32 state, 2 for bf16.

    :returns: Bytes per sequence.
    """
    if min(n_heads, head_k_dim, head_v_dim, n_layers, bytes_per_element) <= 0:
        raise ValueError(
            "every dimension must be positive; a zero here silently reports a free state"
        )
    return n_heads * head_k_dim * head_v_dim * n_layers * bytes_per_element


def kv_cache_bytes(
    *,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    seq_len: int,
    bytes_per_element: int = 2,
) -> int:
    """Bytes of KV cache the ATTENTION layers hold for one sequence at ``seq_len``.

    THE COMPARISON THAT MAKES THE STATE FIGURE MEAN ANYTHING. "512 KiB of recurrent state" is a
    number; "512 KiB fixed against a KV cache that passes it at 43 tokens and reaches 384 MiB at
    32K" is a decision. Reported beside it so the contrast is in the record rather than left for a
    reader to work out.

    IT IS ALSO THE HONEST HALF OF THE STORY, BECAUSE THESE ARMS ARE HYBRIDS. Every arm keeps SIX
    global-attention layers, and those layers' KV cache grows with context exactly as it always
    did. Only the 2 mixer slots have a fixed state. So the arms do not remove the KV cache, they
    shrink the part of the model that needs one -- and a summary that printed only the fixed state
    would imply a pure linear-attention model this study is not testing.

    Two tensors per layer, K and V, at ``n_kv_heads * head_dim`` each per token. bf16 by default
    because a KV cache is stored at the activation dtype, unlike the fp32 recurrent state.

    :returns: Bytes per sequence at this context length.
    """
    if min(n_kv_heads, head_dim, n_layers, seq_len, bytes_per_element) <= 0:
        raise ValueError("every dimension must be positive")
    return 2 * n_kv_heads * head_dim * n_layers * seq_len * bytes_per_element


def decode_state_crossover_tokens(*, state_bytes: int, kv_bytes_per_token: int) -> float | None:
    """Context length at which the KV cache overtakes the fixed recurrent state.

    One number for "when does this start to pay", and it is the honest framing of the linear
    attention claim: below the crossover the fixed state is pure overhead, above it the saving
    grows without bound.

    :returns: Tokens, or ``None`` when the KV cost per token is zero -- which would mean a model
        with no attention layers at all, and is a null rather than an infinity because a division
        that cannot be done must not report a length.
    """
    if kv_bytes_per_token <= 0:
        return None
    return state_bytes / kv_bytes_per_token


def decode_tokens_per_second(*, seconds_per_token: float | None, batch_size: int) -> float | None:
    """Decode tokens/sec from a per-step latency: every sequence in the batch emits one token.

    ``None`` rather than 0.0 when there is no latency to divide, and a zero or negative latency is
    also ``None`` -- a timer that returned nothing measured nothing, and 0.0 tok/s and an
    unmeasured throughput are opposite claims.
    """
    if not seconds_per_token or seconds_per_token <= 0 or batch_size <= 0:
        return None
    return batch_size / seconds_per_token


def decode_basis_string(
    *,
    measured: bool,
    reason: str | None = None,
    operator: str | None = None,
    kernel: str | None = None,
    n_heads: int | None = None,
    head_k_dim: int | None = None,
    head_v_dim: int | None = None,
    mixer_layers: int | None = None,
    total_layers: int | None = None,
    steps: int = DECODE_STEPS,
) -> str:
    """What the decode numbers cover and -- at greater length -- what they do not.

    MIRRORS ``mfu_basis``, FOR THE SAME REASON AND ONE MORE. Like that field, a null with no
    stated cause is indistinguishable from a bug in this file. Unlike it, the numbers here are
    the ones most likely to be QUOTED OUT OF CONTEXT: "3,000 tokens/sec" reads like a serving
    figure, and this one is a single fused operator with no embedding, no FFN, no attention layer,
    no LM head and no sampling. Whoever reads the JSON without reading this function must still
    be told, so the exclusions travel inside the value itself.
    """
    if not measured:
        return f"decode not measured: {reason or 'no reason recorded'}"
    return (
        f"operator-level microbenchmark of {kernel} ({operator}) at n_heads={n_heads}, "
        f"head_k_dim={head_k_dim}, head_v_dim={head_v_dim}, one token per step, "
        f"{steps} timed steps, recurrent state threaded step to step. "
        f"COVERS the {mixer_layers} mixer layer(s) of {total_layers} only, x{mixer_layers} per "
        "model token. EXCLUDES the QKV/gate/output projections, the short convolutions, the "
        f"{total_layers} layers' norms and FFNs, the {total_layers - (mixer_layers or 0)} "
        "non-mixer layers entirely (including the 6 global-attention layers whose KV cache DOES "
        "grow with context), the embedding, the LM head and sampling. NOT a whole-model serving "
        "throughput and must not be quoted as one; it isolates the mixer so arms can be ranked "
        "against each other. A whole-model number needs an incremental step() path, which no "
        "mixer in this tree has."
    )


#: Which fused recurrent kernel decodes each mixer, keyed by its CONFIG CLASS NAME rather than by
#: the mixer string `core6_arms.MIXERS` registers.
#:
#: KEYED ON THE CLASS BECAUSE THE REGISTRY STRINGS ARE NOT A CLOSED SET AND RUN 2 IS ADDING ONE.
#: `KDA_NEGEIG` is new this wave and its registry key is being written by another agent right now.
#: A table keyed on strings would not have that key, would report "no kernel known" for a whole
#: arm, and would do it silently -- one fifth of the study's headline new measurement missing
#: because two files disagreed about a name. Every KDA variant in this bake-off differs only in
#: constructor arguments (`conv_activation`, `gated_conv`, `allow_neg_eigval`) and is the SAME
#: `KimiDeltaAttentionConfig` class decoding through the SAME kernel, so the class is the thing
#: that actually determines the kernel, and keying on it makes a new arm work by construction.
#:
#: The arm's own `allow_neg_eigval` is read off its config and forwarded to the kernel, so
#: `KDA_NEGEIG` decodes with the mechanism it is named for rather than with the class default.
#:
#: `KimiDeltaHouseholderConfig` IS DELIBERATELY ABSENT, and that is a finding rather than an
#: omission: the Householder operator is a custom in-tree kernel
#: (`olmo_core.nn.attention.kda_householder`) with no fused recurrent form in `fla` at all, so
#: there is nothing to time. Those arms are out of run 2; if they return, the honest report is a
#: stated absence, not a KDA number standing in for a Householder one.
DECODE_KERNELS = {
    "KimiDeltaAttentionConfig": ("fla.ops.kda.fused_recurrent", "fused_recurrent_kda"),
    "GatedDeltaNet2Config": ("fla.ops.gdn2.fused_recurrent", "fused_recurrent_gdn2"),
}

#: Which of the two kernels' calling conventions a config class uses. GDN-2 takes two independent
#: channel-wise gates (`b` erases along K, `w` writes along V) where KDA takes one scalar `beta`;
#: passing KDA's beta to GDN-2 would time a different operator than the arm trains with.
DECODE_CALL_STYLE = {
    "KimiDeltaAttentionConfig": "kda",
    "GatedDeltaNet2Config": "gdn2",
}


def _decode_geometry(arm_name: str):
    """The mixer under test, its head geometry, and how many layers carry it.

    Read off the ARM'S OWN CONFIG rather than from constants in this file. The geometry is frozen
    at n_heads=16 head_dim=64 today, and a decode figure computed from a literal would keep
    reporting that after somebody changed the arm -- a wrong state size that still looks right.
    """
    from olmo_core.nn.transformer.core6_arms import ARMS, mixer_config

    arm = ARMS[arm_name]
    config = mixer_config(arm)
    n_heads = config.n_heads
    n_v_heads = getattr(config, "n_v_heads", None) or n_heads
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        # `head_dim=None` means "derive d_model // n_heads", which this function cannot do
        # without knowing d_model. Every bake-off arm passes it explicitly, so None here means
        # the registry changed shape -- refused rather than defaulted, because a guessed head
        # dimension produces a state size that is wrong and looks right.
        raise ValueError(
            f"mixer for arm {arm_name!r} declares no head_dim, so its state size cannot be "
            "computed without d_model"
        )
    expand_v = getattr(config, "expand_v", 1.0)
    return {
        # The registry key, for the record. The KERNEL is chosen by class below, not by this.
        "mixer": arm.mixer,
        "config_class": type(config).__name__,
        "n_heads": n_heads,
        "n_v_heads": n_v_heads,
        "head_k_dim": head_dim,
        "head_v_dim": int(head_dim * expand_v),
        "mixer_layers": len(arm.kda_layers),
        "allow_neg_eigval": bool(getattr(config, "allow_neg_eigval", False)),
    }


@torch.no_grad()
def decode_probe(*, arm_name: str, batch_sizes=DECODE_BATCH_SIZES) -> dict[str, Any]:
    """Time the fused recurrent decode kernel for one arm's mixer, with a threaded state.

    THE MEASUREMENT RUN 1 DID NOT HAVE. See the block comment above for why this is an operator
    microbenchmark rather than a whole-model decode, and :func:`decode_basis_string` for the
    exclusions that travel with the numbers.

    THE RECEIPT IS THE POINT OF THIS FUNCTION, NOT A DECORATION. The audit found metric after
    metric that recorded the REQUEST rather than the EXECUTION -- a field naming the backend that
    was asked for while a silent eager fallback ran, poisoning 13 of 18 cells before anyone
    noticed; and `short_conv.py` defaulting `use_fla=True` and falling back to `nn.Conv1d` without
    a word when `fla` is absent. So nothing here records an intention. Every field below is read
    back off the object that actually ran:

      * ``decode_kernel_resolved`` is ``func.__module__ + "." + func.__qualname__`` of the callable
        that was invoked -- not the name this file looked up.
      * ``decode_state_dtype_realised`` is ``state.dtype`` of the tensor the kernel RETURNED, which
        is what makes the fp32 state-size default checkable rather than assumed.
      * ``decode_state_bytes_realised`` is ``state.numel() * state.element_size()`` of that same
        tensor, so the computed footprint is checked against a measured one.
      * ``decode_state_advanced`` proves the state was THREADED. A harness that passed
        ``initial_state=None`` every step would return plausible latencies for a kernel doing a
        different, cheaper thing -- and nothing in a timing would show it. So the returned state is
        compared against the one passed in, and if it never changed the whole probe is refused.
      * ``decode_fast_path_taken`` is only ever True when all of the above were observed.

    WHAT HAPPENS IF THE FAST PATH IS NOT TAKEN. There is no fallback and no substitute number.
    Every latency field is ``None``, ``decode_fast_path_taken`` is ``False``, and
    ``decode_basis`` says which of the causes it was. A missing decode measurement must never
    read as a fast one, and a partially-working probe must not average a real step with a
    fallback step -- so the arms are comparable or they are absent.

    Never raises. A decode probe is a secondary measurement bolted onto a paid-for training run,
    and losing the CE endpoint to a benchmark bug would be a far worse trade than losing the
    benchmark. Failures become a recorded reason.
    """
    try:
        geometry = _decode_geometry(arm_name)
    except BaseException as exc:  # noqa: BLE001 -- a benchmark must not cost the CE endpoint
        log.warning("could not read the decode geometry for %s: %r", arm_name, exc)
        return {
            "decode_fast_path_taken": False,
            "decode_basis": decode_basis_string(
                measured=False,
                reason=f"the arm's mixer geometry could not be read: {type(exc).__name__}: {exc}",
            ),
        }

    mixer = geometry["mixer"]
    config_class = geometry["config_class"]
    state_elems_per_layer = geometry["n_v_heads"] * geometry["head_k_dim"] * geometry["head_v_dim"]

    # The computed footprint. Emitted even with no GPU: it is arithmetic on the geometry, it is
    # the field that decides serving batch size, and it does not need a kernel to be true.
    result: dict[str, Any] = {
        "decode_operator": mixer,
        "decode_config_class": config_class,
        "decode_allow_neg_eigval": geometry["allow_neg_eigval"],
        "decode_n_heads": geometry["n_heads"],
        "decode_n_v_heads": geometry["n_v_heads"],
        "decode_head_k_dim": geometry["head_k_dim"],
        "decode_head_v_dim": geometry["head_v_dim"],
        "decode_mixer_layers": geometry["mixer_layers"],
        "decode_state_elems_per_layer": state_elems_per_layer,
        # THE HEADLINE FIELD. fp32 because that is what `fla` keeps the state in; the realised
        # dtype below is what proves it.
        "decode_state_bytes_per_seq": recurrent_state_bytes(
            n_heads=geometry["n_v_heads"],
            head_k_dim=geometry["head_k_dim"],
            head_v_dim=geometry["head_v_dim"],
            n_layers=geometry["mixer_layers"],
            bytes_per_element=4,
        ),
        "decode_kernel_requested": None,
        "decode_kernel_resolved": None,
        "decode_fast_path_taken": False,
        "decode_state_dtype_realised": None,
        "decode_state_bytes_realised": None,
        "decode_state_advanced": None,
        "decode_batches": {},
        "decode_warmup_steps": DECODE_WARMUP_STEPS,
        "decode_timed_steps": DECODE_STEPS,
    }

    # The KV-cache contrast, which is what makes the state figure a decision rather than a number.
    # Computed from the arm's attention geometry, not the mixer's.
    try:
        from olmo_core.nn.transformer.core6_arms import (
            ATTENTION_LAYERS,
            HEAD_DIM,
            N_KV_HEADS,
            N_LAYERS,
        )

        kv_per_token = kv_cache_bytes(
            n_kv_heads=N_KV_HEADS,
            head_dim=HEAD_DIM,
            n_layers=len(ATTENTION_LAYERS),
            seq_len=1,
        )
        result["decode_attention_layers"] = len(ATTENTION_LAYERS)
        result["decode_total_layers"] = N_LAYERS
        result["decode_kv_bytes_per_token"] = kv_per_token
        result["decode_kv_bytes_per_seq_at"] = {
            str(t): kv_per_token * t for t in (1024, 4096, 8192, 32768)
        }
        result["decode_state_vs_kv_crossover_tokens"] = decode_state_crossover_tokens(
            state_bytes=result["decode_state_bytes_per_seq"], kv_bytes_per_token=kv_per_token
        )
    except Exception as exc:  # noqa: BLE001 -- a contrast, not the endpoint
        log.warning("could not compute the KV-cache contrast: %r", exc)

    if config_class not in DECODE_KERNELS:
        result["decode_basis"] = decode_basis_string(
            measured=False,
            reason=(
                f"no fused recurrent kernel is known for {config_class} (arm mixer {mixer!r}). "
                "The Householder operator is a custom in-tree kernel with no fla recurrent form, "
                "so there is nothing to time rather than a number to substitute."
            ),
        )
        return result

    module_name, func_name = DECODE_KERNELS[config_class]
    call_style = DECODE_CALL_STYLE[config_class]
    result["decode_kernel_requested"] = f"{module_name}.{func_name}"

    if not torch.cuda.is_available():
        result["decode_basis"] = decode_basis_string(
            measured=False,
            reason="no CUDA device, and the fused recurrent kernels are Triton and CUDA-only",
        )
        return result

    try:
        import importlib

        import numpy as np  # noqa: F401 -- imported for parity with the rest of this file

        module = importlib.import_module(module_name)
        kernel = getattr(module, func_name)
        # THE RECEIPT: the identity of the callable that will actually be invoked, read off the
        # object rather than from the string above. A kernel re-exported from somewhere else, or
        # shadowed by a wrapper, shows up here as a different name.
        result["decode_kernel_resolved"] = f"{kernel.__module__}.{kernel.__qualname__}"

        device = torch.device("cuda")
        # bf16 inputs, matching the run's `param_dtype`, with an fp32 state -- which is the
        # combination `fla` uses and the one a server would run.
        dtype = torch.bfloat16
        H, HV = geometry["n_heads"], geometry["n_v_heads"]
        K, V = geometry["head_k_dim"], geometry["head_v_dim"]

        for batch_size in batch_sizes:
            B, T = batch_size, 1  # one token per step: this is decode
            torch.manual_seed(0)

            def randn(*shape):
                return torch.randn(*shape, device=device, dtype=dtype)

            q = randn(B, T, H, K)
            k = randn(B, T, H, K)
            v = randn(B, T, HV, V)
            # The gate is the RAW pre-activation and the kernel derives the decay itself, exactly
            # as the training forward passes it (`use_gate_in_kernel=True`). A gate passed as an
            # already-log-space decay would be a different, cheaper kernel path.
            g = randn(B, T, HV, K)
            # UNIFORM ON (1, 16) THEN log, WHICH IS `init_weights`' OWN RANGE AND NOT AN
            # ARBITRARY ONE. `torch.rand` starts at 0 and log(0) is -inf, which makes exp(A_log)
            # a zero decay: the state would be multiplied by nothing every step, never change,
            # and the state-advanced receipt below would fail and abort the probe -- a benchmark
            # bug wearing the costume of a kernel that does not thread state. recurrent.py's
            # initialiser avoids log(0) for the same reason and says so.
            A_log = (torch.rand(HV, device=device, dtype=torch.float32) * 15.0 + 1.0).log()
            dt_bias = torch.rand(HV * K, device=device, dtype=torch.float32)
            # Nonzero, so the FIRST step already has something to decay and transform. A state
            # starting at exactly zero still advances here (the write term is nonzero), but a
            # nonzero start also exercises the decay path the timing is meant to include.
            state = torch.randn(B, HV, K, V, device=device, dtype=torch.float32)

            if call_style == "gdn2":
                # GDN-2's two independent channel-wise gates: `b` erases along K, `w` writes
                # along V. Passing one scalar beta for both would silently time KDA's operator.
                erase = randn(B, T, HV, K).sigmoid()
                write = randn(B, T, HV, V).sigmoid()
                call = (
                    lambda s, kernel=kernel, q=q, k=k, v=v, g=g, erase=erase, write=write, A_log=A_log, dt_bias=dt_bias: (
                        kernel(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            b=erase,
                            w=write,
                            A_log=A_log,
                            dt_bias=dt_bias,
                            initial_state=s,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=True,
                        )
                    )
                )
            else:
                # `allow_neg_eigval` IS APPLIED IN EAGER PYTORCH AND IS DELIBERATELY *NOT*
                # FORWARDED TO THE KERNEL, BECAUSE THAT IS WHAT THE TRAINING FORWARD DOES.
                # `KimiDeltaAttention.forward` computes `beta = w_b(x).sigmoid()` and then, if the
                # flag is set, `beta = beta * 2.0` -- and hands `dispatch_chunk_kda` a plain
                # post-sigmoid tensor with no `allow_neg_eigval` argument at all. The flag selects
                # no kernel, no branch and no flag inside fla on that path.
                #
                # `fused_recurrent_kda` DOES accept an `allow_neg_eigval` parameter, so passing it
                # here as well as doubling beta would apply the mechanism TWICE -- a beta in (0,4)
                # for an arm trained with beta in (0,2). It would not error, the latency would look
                # entirely normal, and KDA_NEGEIG's decode number would describe an operator no arm
                # in the study trains. So the eager doubling alone is the faithful mirror.
                beta = randn(B, T, HV).sigmoid()
                if geometry["allow_neg_eigval"]:
                    beta = beta * 2.0
                call = (
                    lambda s, kernel=kernel, q=q, k=k, v=v, g=g, beta=beta, A_log=A_log, dt_bias=dt_bias: (
                        kernel(
                            q=q,
                            k=k,
                            v=v,
                            g=g,
                            beta=beta,
                            A_log=A_log,
                            dt_bias=dt_bias,
                            initial_state=s,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                            use_gate_in_kernel=True,
                        )
                    )
                )

            # Warmup: Triton compiles on the first call and the allocator is still growing.
            threaded = state
            for _ in range(DECODE_WARMUP_STEPS):
                _, threaded = call(threaded)
            torch.cuda.synchronize()

            # THE STATE-THREADING RECEIPT. Compared against the state that went IN, on a clone so
            # the kernel cannot have written through the reference we are comparing to. A probe
            # that silently failed to thread would time a different, cheaper computation and no
            # latency would look wrong.
            before = threaded.clone()
            _, threaded = call(threaded)
            torch.cuda.synchronize()
            advanced = bool(not torch.equal(before, threaded))
            if result["decode_state_advanced"] is None:
                result["decode_state_advanced"] = advanced
                result["decode_state_dtype_realised"] = str(threaded.dtype)
                # Per sequence, across the mixer layers -- the same quantity the computed field
                # above reports, measured off the tensor the kernel returned.
                realised = int(
                    threaded.numel() * threaded.element_size() // B * geometry["mixer_layers"]
                )
                result["decode_state_bytes_realised"] = realised
                # THE TWO NUMBERS MUST AGREE, AND THE COMPARISON IS RECORDED RATHER THAN LEFT FOR
                # A READER TO DO. `decode_state_bytes_per_seq` is computed from the geometry with
                # fp32 assumed; this is measured off the kernel's own tensor. If fla ever returns
                # a bf16 state, or the geometry stops matching the kernel's layout, the headline
                # footprint is wrong by 2x and BOTH fields would still look reasonable alone. A
                # boolean makes the disagreement greppable instead of latent.
                result["decode_state_bytes_agree"] = bool(
                    realised == result["decode_state_bytes_per_seq"]
                )
                if not result["decode_state_bytes_agree"]:
                    log.warning(
                        "decode state size disagrees: computed %d B/seq from the geometry, "
                        "measured %d B/seq off the kernel's %s state -- the headline footprint "
                        "is the computed one and it is wrong",
                        result["decode_state_bytes_per_seq"],
                        realised,
                        threaded.dtype,
                    )
            if not advanced:
                # No timings recorded at all: a latency for a kernel that is not advancing state
                # is a number for the wrong computation, and publishing it would be worse than
                # publishing nothing.
                result["decode_basis"] = decode_basis_string(
                    measured=False,
                    reason=(
                        f"the recurrent state did not change across a decode step at batch "
                        f"{B}, so the kernel was not threading state and any latency would "
                        "describe a different computation"
                    ),
                )
                return result

            latencies: list[float] = []
            for _ in range(DECODE_STEPS):
                torch.cuda.synchronize()
                started = time.perf_counter()
                _, threaded = call(threaded)
                torch.cuda.synchronize()
                latencies.append(time.perf_counter() - started)

            median = quantile_nearest_rank(latencies, 0.5)
            p90 = quantile_nearest_rank(latencies, 0.9)
            result["decode_batches"][str(B)] = {
                "batch_size": B,
                # Per token per sequence: one step emits one token for each of B sequences, so
                # the step latency IS the per-token latency at this batch size.
                "latency_ms_p50": None if median is None else median * 1000.0,
                "latency_ms_p90": None if p90 is None else p90 * 1000.0,
                "tokens_per_second": decode_tokens_per_second(
                    seconds_per_token=median, batch_size=B
                ),
                "state_bytes_total": int(
                    threaded.numel() * threaded.element_size() * geometry["mixer_layers"]
                ),
            }

        result["decode_fast_path_taken"] = True
        result["decode_basis"] = decode_basis_string(
            measured=True,
            operator=mixer,
            kernel=result["decode_kernel_resolved"],
            n_heads=geometry["n_v_heads"],
            head_k_dim=K,
            head_v_dim=V,
            mixer_layers=geometry["mixer_layers"],
            total_layers=result.get("decode_total_layers", 16),
        )
    except BaseException as exc:  # noqa: BLE001 -- never lose the CE endpoint to a benchmark
        log.warning("decode probe failed (%s: %s)", type(exc).__name__, exc)
        result["decode_fast_path_taken"] = False
        result["decode_batches"] = {}
        result["decode_basis"] = decode_basis_string(
            measured=False, reason=f"the probe raised {type(exc).__name__}: {exc}"
        )
    return result


def decode_report(*, arm_name: str, world_size: int) -> dict[str, Any]:
    """The decode probe plus the per-device framing, flattened for the summary JSON.

    ONE RANK MEASURES AND THE FIGURE IS PER DEVICE ALREADY. Unlike training throughput, decode
    here is a single-device measurement: the probe drives one operator on one GPU with no
    collectives, so there is nothing to all-reduce and no world-size divisor to apply. The total
    across the node is the per-device figure times ``world_size``, which is stated rather than
    left for a reader to assume -- a decode number silently divided by 8 would rank every arm
    identically wrongly.

    ``decode_state_bytes_per_seq`` counts the MIXER LAYERS ONLY -- 2 of 16 -- because those are
    the layers whose state is fixed. The other 14 are 6 global attention (whose KV cache grows)
    and 8 LIV short convolutions, and folding them into one "state" figure would merge a
    context-independent cost with a context-dependent one.
    """
    probe = decode_probe(arm_name=arm_name)

    per_device_total = None
    node_total = None
    largest = probe.get("decode_batches", {}).get(str(max(DECODE_BATCH_SIZES)))
    if largest is not None:
        per_device_total = largest.get("tokens_per_second")
        if per_device_total is not None and world_size and world_size > 0:
            node_total = per_device_total * world_size

    probe["decode_tok_s_per_device"] = per_device_total
    probe["decode_tok_s_total"] = node_total
    probe["decode_tok_s_basis"] = (
        f"single-device operator throughput at batch {max(DECODE_BATCH_SIZES)}; total is "
        f"per-device x world_size ({world_size}) and assumes independent replicas, which is how "
        "a served model runs. NOT measured across ranks -- the probe uses no collectives."
    )
    return probe


def memory_report(losses: LossWatcher) -> dict[str, Any]:
    """The memory half of the record: peak allocated and reserved, and WHICH read they are.

    A fixed-size recurrent state is the main selling point of every mixer in this bake-off, so
    the memory figure is an endpoint rather than a footnote -- and it is only an endpoint if the
    six arms' numbers are the same quantity measured the same way.

    THREE SOURCES, AND THEY ARE NOT INTERCHANGEABLE, WHICH IS WHY THE FIELD SAYS WHICH ONE:

    ``per_step_running_max`` -- the real whole-run peak, accumulated by :class:`LossWatcher`
    before ``GPUMemoryMonitorCallback`` resets the counters each step. This is the one to size
    hardware with and the one to compare arms on.

    ``final_step_only`` -- the naive post-``fit()`` read. Because the monitor resets peak stats
    every step, this is the last step's peak and nothing more. It is a LOWER BOUND on the truth,
    it looks exactly like a whole-run figure, and the difference is invisible in the value. It
    is used only when the per-step sampler never ran, and it is labelled so nobody sizes a card
    against it by accident.

    ``unavailable`` -- no CUDA. Both figures are null, and null is the whole point: a run on CPU
    reporting 0.0 GiB of peak memory is a claim that the arm is free, which is the direction a
    missing measurement must never fail in.
    """
    gib = 1024**3
    if losses.memory_samples > 0 and losses.peak_allocated_bytes > 0:
        return {
            "peak_memory_gib": losses.peak_allocated_bytes / gib,
            "peak_memory_reserved_gib": losses.peak_reserved_bytes / gib,
            "peak_memory_source": "per_step_running_max",
            "peak_memory_samples": losses.memory_samples,
        }

    if torch.cuda.is_available():
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        return {
            # A LOWER BOUND WEARING THE NAME OF A PEAK. `peak_memory_source` is the only thing
            # separating this from the real figure, so it is never emitted without it.
            "peak_memory_gib": allocated / gib if allocated > 0 else None,
            "peak_memory_reserved_gib": reserved / gib if reserved > 0 else None,
            "peak_memory_source": "final_step_only",
            "peak_memory_samples": 0,
        }

    return {
        "peak_memory_gib": None,
        "peak_memory_reserved_gib": None,
        "peak_memory_source": "unavailable",
        "peak_memory_samples": 0,
    }


def summarise(
    *, opts, config, trainer, losses: LossWatcher, seconds: float, sliced=None, val=None
) -> None:
    """Print what only this process can report, as one JSON object on stdout.

    The platform reads this back out of the log stream: the device torch actually got, the
    parameter count, the loss at both ends and where the checkpoints went are not facts Batch
    holds. Printed on rank zero only, and printed whatever the losses are, because a run that
    reported nothing is indistinguishable from one that never started.

    ``val`` is the held-out result from :func:`evaluate_val_aggregate`, already all-reduced
    across ranks, so printing it from rank zero prints the whole run's number and not this
    rank's share. That is the reason the rank gate here is safe and the one on the old sliced
    eval was not: this function does no collective work, it formats a value the collectives
    already produced.
    """
    if get_rank() != 0:
        return
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # FLOPs per token comes from the train module, which returns None rather than raising when
    # the model cannot estimate it -- so an arm whose mixer has no flop count yields a null MFU
    # and a stated reason, not a wrong percentage.
    flops_per_token: int | None = None
    with contextlib.suppress(Exception):
        flops_per_token = trainer.train_module.num_flops_per_token(opts.sequence_length)

    speed = throughput_report(
        losses,
        world_size=get_world_size(),
        wall_clock_seconds=seconds,
        flops_per_token=flops_per_token,
        device_name=device if torch.cuda.is_available() else None,
    )
    memory = memory_report(losses)

    # THE DECODE MEASUREMENT. Run on rank zero only, and that is safe for the same reason
    # printing is: the probe drives one fused operator on one device with NO collectives and no
    # sharded parameters, so unlike the sliced eval there is no all-gather for the other ranks to
    # be missing from. It runs here, inside the rank gate, rather than in `train()` for exactly
    # that reason -- putting it on all ranks would have eight processes contend for one card's
    # clocks and report a latency none of them would see alone.
    #
    # Never raises: `decode_probe` turns every failure into a recorded reason, because a
    # benchmark must not cost a run its CE endpoint.
    decode: dict[str, Any] = {}
    if opts.decode_probe:
        decode = decode_report(arm_name=opts.arm, world_size=get_world_size())
    else:
        decode = {
            "decode_fast_path_taken": False,
            "decode_basis": decode_basis_string(
                measured=False, reason="--no-decode-probe was passed, so decode was not measured"
            ),
        }

    print(
        json.dumps(
            {
                "run_id": opts.run_name,
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                # BOTH seeds, because a paired analysis is only possible if each result
                # says which data order and which initialisation produced it. Recording
                # one and defaulting the other is how "n seeds" came to mean n data
                # orderings of a single init.
                "data_seed": opts.data_seed,
                "init_seed": config.init_seed,
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
                "world_size": get_world_size(),
                # CO-PRIMARY: SPEED. Two figures with deliberately unmistakable names --
                # `throughput_tok_s_steady` ranks arms, `throughput_tok_s_whole_run` costs wall
                # clock -- plus the counts behind them, the step-time distribution and MFU.
                # Every one of them is null rather than zero when it could not be measured.
                # See throughput_report for what each is and why they are not interchangeable.
                **speed,
                # CO-PRIMARY: MEMORY. `peak_memory_source` says WHICH read the figure is, and it
                # is not decoration: the naive post-fit read is the last step's peak only,
                # because the GPU monitor resets the counters every step. See memory_report.
                **memory,
                # THE INFERENCE HALF, WHICH RUN 1 DID NOT MEASURE AT ALL. The practical case for
                # a linear-attention mixer is that generation is cheap -- a fixed recurrent state
                # instead of a KV cache that grows with context -- and run 1 never ran that path.
                # `decode_state_bytes_per_seq` is the field that decides serving batch size.
                #
                # `decode_fast_path_taken` IS A RECEIPT, NOT A REQUEST: it is only True when the
                # kernel was resolved by identity, the returned state was fp32, and the state was
                # OBSERVED to advance across a step. If it is False every latency is null and
                # `decode_basis` says which cause. Read `decode_basis` before quoting any of
                # these -- it is an operator microbenchmark and not a serving throughput.
                **decode,
                # The SpeedMonitorCallback's own per-device averages, kept because the
                # preregistration names them and because they are an independent measurement of
                # the same thing -- computed by upstream code, over a window that starts after
                # step 1 rather than after the warmup cutoff. They should sit slightly BELOW
                # `throughput_tok_s_steady_per_device`; a large gap means compilation or
                # allocator growth leaked into upstream's average, which is the exact failure
                # the cutoff exists for, so keeping both makes it visible instead of arguable.
                "tps_device_avg": losses.tps_device_avg,
                "tps_device_last": losses.tps_device_last,
                "tps_total_avg": (
                    None
                    if losses.tps_device_avg is None
                    else losses.tps_device_avg * get_world_size()
                ),
                # Kept for contrast, and it is the WRONG number for costing: it includes
                # every fixed cost above. On a short probe it can be several times lower
                # than the steady-state figure, and it penalises bigger shapes hardest.
                "tps_naive_wall_clock": (
                    None if not seconds else trainer.global_step * opts.global_batch_size / seconds
                ),
                "checkpoint_uri": opts.save_folder,
                "wandb_project": os.environ.get("EDULLM_WANDB_PROJECT", ""),
                "wandb_url": losses.wandb_url,
                # The arm and its realized token count belong in the machine-readable record,
                # not only in the log prose: a difference between two arms is only a paired
                # difference if both trained on the same number of tokens, and that is checked
                # by comparing these fields rather than by trusting the two commands matched.
                "arm": opts.arm,
                "tokens_trained": trainer.global_step * opts.global_batch_size,
                # THE ENDPOINT, FLAT AND AT THE TOP LEVEL, because it is the field every
                # downstream contrast reads. A difference of two arms' `val_ce` is only a paired
                # difference if their `val_tokens` match, so the denominator ships beside the
                # number instead of being assumed equal. `val_tokens_present` is the count that
                # was asserted against the manifest -- if it is here at all, that assertion
                # passed, because the run refuses rather than printing a number it could not
                # account for.
                "val_ce": None if val is None else val["ce"],
                "val_tokens": None if val is None else val["tokens"],
                "val_tokens_present": None if val is None else val["tokens_present"],
                "val_tokens_declared": None if val is None else val["declared_tokens"],
                "val_nll_sum": None if val is None else val["sum"],
                "val_shards": None if val is None else val["shards"],
                # null when no slice directories were passed, which is how a training-only run
                # says so explicitly rather than by omission.
                "sliced_eval": sliced,
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


def preflight_accelerated_arm(opts) -> None:
    """Fail before distributed startup if the selected strict backend is unavailable."""
    if opts.arm != "xlstm":
        return

    from importlib import metadata

    expected = {
        "xlstm": "2.0.5",
        "mlstm-kernels": "2.0.4",
        "flashrnn": "1.0.6",
    }
    for package, version in expected.items():
        actual = metadata.version(package)
        if actual != version:
            raise Refusal(
                Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
                f"{package}=={version} is required, found {actual}",
            )
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            "the strict xLSTM comparison backend requires an A100 sm_80 CUDA device",
        )

    from olmo_core.nn.xlstm import _preflight_flashrnn, _prewarm_flashrnn

    _preflight_flashrnn()
    _prewarm_flashrnn(
        batch_size=opts.rank_microbatch_size // opts.sequence_length,
        seq_len=opts.sequence_length,
        n_heads=4,
        head_dim=1024 // 4,
        kernel_dtype="float32",
        device=torch.device("cuda", torch.cuda.current_device()),
    )


def train(config, opts) -> None:
    """Train, then evaluate the held-out endpoint, then report. All three, or none.

    ``opts`` IS REQUIRED, AND IT USED TO DEFAULT TO None. Everything after ``fit()`` -- the
    endpoint, the token-count assertion and ``summarise()`` -- sat behind ``if opts is not
    None``, so a caller that omitted it got a run that trained, checkpointed, and printed NO
    JSON at all. That is not a smaller failure than the null endpoint this file exists to
    remove; it is a larger one, because the JSON on stdout is the only channel the platform
    reads a run's results back through. ``main()`` always passes ``opts``, so the default was
    never exercised -- which is exactly why it could sit there being wrong. Making the
    parameter required means a caller that forgets gets a TypeError at the call rather than a
    silent trained-and-said-nothing run.
    """
    if get_rank() == 0:
        show(config)

    # THE TWO SEEDS MUST AGREE, BECAUSE ONE IS REPORTED AND THE OTHER IS USED. `build_config`
    # sets both from `--init-seed`, and then `config.merge(overrides)` runs -- so a dotlist
    # override on the command line (`init_seed=99`, which is a natural thing to type, or
    # `model.init_seed=42`) moves ONE of them. `summarise()` prints `config.init_seed` while the
    # weights are drawn from `config.model.init_seed`, so a divergence republishes the exact bug
    # this file just fixed: a JSON asserting a seed the tensors never saw. Checked here rather
    # than in `build_config` because `merge` happens on the way out of that function.
    if config.init_seed != config.model.init_seed:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"the summary would report init_seed {config.init_seed} while the weights are drawn "
            f"from {config.model.init_seed}. These are set together from --init-seed and can "
            "only differ if an override moved one of them; an override on `init_seed` alone "
            "changes what is REPORTED and not what is USED, which is the failure --init-seed "
            "was just fixed for. Pass --init-seed instead of overriding either field.",
        )

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
    elapsed = time.monotonic() - started

    val = None
    if opts.skip_heldout_eval:
        log.warning("held-out evaluation skipped by explicit smoke-test flag; val_ce will be null")
    else:
        # THE HELD-OUT ENDPOINT, ON EVERY RANK, WITH NO `except` AROUND IT. Everything the
        # full experiment reports is a difference of this number between arms, so only the
        # explicit smoke-test flag above may skip it.
        val = evaluate_val_aggregate(
            model=trainer.train_module.model,
            vocab_size=config.model.vocab_size,
            val_paths=list(config.val_paths),
            work_dir=opts.work_dir,
            seq_len=opts.sequence_length,
            dtype=config.dataset.dtype.as_np_dtype(),
            declared_tokens=config.val_rows,
        )
        # Every rank checks the same all-reduced denominator before a CE can be reported.
        assert_val_tokens_account_for_the_corpus(val)

    # THE SLICED EVALUATION, ON EVERY RANK. It is SECONDARY -- it decomposes CE by gap band and
    # reads frozen masks that must be built first -- but secondary does not mean rank-zero.
    #
    # IT USED TO BE GATED `if get_rank() == 0:` AND RUN 2 WOULD HAVE HUNG ON IT. Run 2 passes
    # --slice-mask-uri on 8 GPUs, which is the first wave to reach this path at all. Under FSDP a
    # rank-zero-only forward waits on all-gathers the other ranks never enter, the `except` below
    # cannot catch a hang, and the barrier that used to sit in the `finally` only made it
    # "survivable rather than correct" by holding the peers somewhere that at least times out.
    # Both `fetch_slice_inputs` and `evaluate_sliced` now shard by rank and reduce across ranks,
    # exactly as `evaluate_val_aggregate` does, so there is no gate left to remove.
    #
    # THE `try` STAYS, AND ITS MEANING HAS CHANGED. It no longer protects against a rank-zero
    # hang -- nothing can. It protects the CHECKPOINT and `val_ce` from a bug in a secondary
    # measurement, and it is now safe in a way it was not before: every rank runs the same code,
    # so every rank raises the same refusal at the same collective and none is left waiting.
    sliced = None
    if opts.slice_mask_uri and not opts.skip_heldout_eval:
        try:
            # Sharded by rank: the union over ranks is the whole mask set, each object fetched
            # once. The manifest's band layout is checked on every rank. Wrapped so that one
            # rank's failed download takes every rank down together rather than leaving this
            # rank at the barrier below while its peers enter the evaluator's all-reduces --
            # see `fetch_slice_inputs_on_every_rank` for why that split is a hang.
            val_paths, mask_paths = fetch_slice_inputs_on_every_rank(
                mask_uri=opts.slice_mask_uri,
                work_dir=opts.work_dir,
                # Passed so the manifest's build-time window is compared against the window we
                # actually score at. A mismatch produces a plausible, fully-populated, WRONG table
                # rather than an error, so it must be refused here.
                seq_len=opts.sequence_length,
            )
            sliced = evaluate_sliced(
                model=trainer.train_module.model,
                vocab_size=config.model.vocab_size,
                val_paths=val_paths,
                mask_paths=mask_paths,
                seq_len=opts.sequence_length,
            )
            # Logged on rank zero alone because the numbers are already all-reduced -- this is
            # formatting, not computing, which is the same reason `summarise` may print from one
            # rank. Every rank has the identical dict.
            if get_rank() == 0:
                log.info(
                    "sliced aggregate CE %.4f over %s tokens (%d ranks)",
                    sliced["aggregate"]["ce"],
                    f"{sliced['aggregate']['n']:,}",
                    sliced["world_size"],
                )
                for band in sorted(BAND_BIT):
                    scored = sliced["bands"][str(band)]
                    # `ce` is null for an unmeasured band rather than 0.0, so say which it is
                    # instead of printing a perfect score for an empty set.
                    if scored["n"]:
                        log.info(
                            "  gap>%-5s CE %.4f over %s tokens",
                            band,
                            scored["ce"],
                            f"{scored['n']:,}",
                        )
                    else:
                        log.warning("  gap>%-5s NOT MEASURED (no tokens in this band)", band)
        # `Exception` IS THE WRONG BASE HERE AND THE COMMENT USED TO PROMISE OTHERWISE.
        # `Refusal` subclasses `SystemExit`, which is a `BaseException` and NOT an `Exception`
        # -- so the `Refusal` that `evaluate_sliced` raises on a mask/shard length mismatch
        # would sail straight through an `except Exception` that says it never loses a
        # checkpoint to a secondary eval bug, and lose one. `KeyboardInterrupt` and the
        # trainer's own cancellation are the reason this is not simply `BaseException`: those
        # must still stop the run.
        except (Exception, Refusal) as error:  # noqa: BLE001
            # Never lose a checkpoint to a secondary-evaluation bug.
            log.warning(
                "sliced eval failed (%s: %s); checkpoint is still on S3 and val_ce above is "
                "unaffected",
                type(error).__name__,
                error,
            )
            # Left as None rather than a partial dict: a half-filled band table would be read as
            # a measurement. `sliced_eval: null` in the summary is how a run says it did not get
            # one, which is the same convention the null endpoint work established.
            sliced = None
        finally:
            # Every rank arrives here, whether it scored or refused, so the ranks re-converge
            # before the summary and the teardown regardless of which branch they took.
            barrier()

    # Unconditional. The JSON on stdout is the only channel the platform reads a run's results
    # back through, so a path that reaches here and prints nothing is a run that did not happen
    # as far as the record is concerned.
    summarise(
        opts=opts,
        config=config,
        trainer=trainer,
        losses=losses,
        seconds=elapsed,
        sliced=sliced,
        val=val,
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
    parser.add_argument(
        "--arm",
        default="mamba-b3",
        help="Frozen comparison arm: mamba-b3, xlstm, mamba3-siso-pd, or native-pd.",
    )
    parser.add_argument(
        "--slice-mask-uri",
        default=os.environ.get("EDULLM_SLICE_MASK_URI", ""),
        help="S3 prefix holding slice_manifest.json and the frozen *.mask.u8 files from "
        "build_slice_masks.py. The manifest names the corpus shards to pair them with and "
        "carries a digest per mask, both of which are checked. Empty skips the evaluation "
        "and the run produces a checkpoint only.",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3721)
    parser.add_argument("--save-interval", type=int, default=1861)
    parser.add_argument("--warmup-steps", type=int, default=372)
    parser.add_argument("--learning-rate", type=float, default=1.4e-3)
    parser.add_argument("--global-batch-size", type=int, default=524288)
    parser.add_argument("--rank-microbatch-size", type=int, default=8192)
    parser.add_argument("--param-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--data-seed", type=int, default=0)
    # Weight init is a SEPARATE variance component from data order, and until this flag
    # existed only `--data-seed` was exposed while `init_seed` stayed at its 12536 default.
    # Every "n seeds" on this entry point was therefore n data orderings of ONE
    # initialisation -- a narrower component than the FarmShare seed replicate measured,
    # which biases any CI built from it optimistically. A paired design wants both varied.
    parser.add_argument("--init-seed", type=int, default=12536)
    # THE DECODE PROBE IS ON BY DEFAULT, AND THE FLAG ONLY TURNS IT OFF. Run 2 exists partly to
    # produce this number, so an opt-IN flag is one forgotten argument away from repeating run 1's
    # gap -- and a cell that silently skipped it would report `decode_basis` explaining itself
    # only to a reader who went looking. The cost is bounded and small: 3 batch sizes x 72 steps
    # of one fused operator, after fit() has finished, on rank zero alone.
    #
    # DECLARED HERE, WITH `dest`, BECAUSE AN UNDECLARED FLAG IS HOW RUN 1 LOST ALL 18 CELLS. An
    # argument argparse does not know about is swallowed by `parse_known_args` and handed to
    # `config.merge()` as a dotted override, which dies at merge time -- AFTER the corpus and the
    # arm have logged correctly, so it reads like a config bug rather than a typo. Grepping for a
    # flag string is not enough either: run 1's `--lm-loss-implementation` appeared in a prose
    # comment and nowhere in any `add_argument` call.
    parser.add_argument(
        "--skip-heldout-eval",
        action="store_true",
        help="Smoke-test only: train and report speed/memory with val_ce=null. The full "
        "comparison must never pass this flag.",
    )
    parser.add_argument(
        "--no-decode-probe",
        dest="decode_probe",
        action="store_false",
        default=True,
        help="Skip the fused-recurrent decode measurement. On by default; the whole reason run 2 "
        "exists is that run 1 measured nothing about inference. Turning it off records "
        "decode_fast_path_taken=false with a stated reason rather than a null field.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, do not train.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts, overrides = build_parser().parse_known_args()
    from model_arch_tests import INIT_SEEDS_BY_ARM

    try:
        expected_init_seed = INIT_SEEDS_BY_ARM[opts.arm][opts.data_seed]
    except KeyError:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"({opts.arm}, {opts.data_seed}) is not a frozen comparison cell",
        ) from None
    if opts.init_seed != expected_init_seed:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"init seed {opts.init_seed} is invalid for {opts.arm}/{opts.data_seed}; "
            f"expected {expected_init_seed}",
        )

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

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        preflight_accelerated_arm(opts)
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
