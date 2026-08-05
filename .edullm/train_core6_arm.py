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
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, Optional, cast

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
    val_paths: List[str] = field(default_factory=list)
    val_rows: Optional[int] = None


@dataclass
class Corpus:
    """What the manifest says, after the three checks that make it safe to memmap."""

    dataset_id: str
    version: str
    paths: List[str]
    dtype: NumpyDatasetDType
    tokenizer: TokenizerConfig
    rows: Optional[int]
    #: The corpus's OWN held-out objects, taken from the reader's split resolution.
    #:
    #: NOT reconstructed from shard names, and that is not a stylistic preference. A mask named
    #: ``all-dressed-snazzy2__val-00212`` corresponds to
    #: ``all-dressed-snazzy2/art_and_design/val-00212.u32le.bin`` -- the topic directory is
    #: dropped from the name and 24 topics exist, so rebuilding a key from a filename fetches
    #: a real, readable, plausible shard belonging to a different topic. The reader's
    #: ``.val`` is the only place the true keys are written down.
    val_paths: List[str] = field(default_factory=list)
    #: Rows the manifest DECLARES for the held-out partitions, or None if it declares none.
    #: This is the number the realized token count is checked against; see
    #: :func:`evaluate_val_aggregate`.
    val_rows: Optional[int] = None


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
    # objects are the exact keys the manifest sealed -- topic directory and all. The declared
    # row count is then taken for the splits whose objects ARE those val objects, rather than
    # by re-deciding which partition names count as held out: a second copy of that rule here
    # is a place for the two to disagree, and the disagreement would be a token count checked
    # against the wrong partition's declaration.
    val_paths = list(getattr(read, "val", None) or [])
    split_rows = getattr(read, "split_rows", None) or {}
    held_out = set(val_paths)
    declared = [
        split_rows.get(name)
        for name, paths in (getattr(read, "splits", None) or {}).items()
        if paths and held_out.issuperset(paths)
    ]
    val_rows = (
        sum(n for n in declared if n is not None)
        if declared and all(n is not None for n in declared)
        else None
    )

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
    # Said at CONFIG time rather than only at eval time, so a corpus that declares no held-out
    # split is visible before the GPU hours are spent rather than after. The endpoint refuses
    # in that case (Stage 73), and finding that out eleven hours in is the expensive version.
    log.info(
        "held out: %d object(s), %s declared token(s)",
        len(corpus.val_paths),
        "none" if corpus.val_rows is None else f"{corpus.val_rows:,}",
    )

    # CORE-6 arms are declared in olmo_core.nn.transformer.core6_arms, not as
    # TransformerConfig classmethods, so this replaces the base entry point's
    # getattr(TransformerConfig, ...) lookup.
    from olmo_core.nn.transformer.core6_arms import ARMS, build_arm

    if opts.arm not in ARMS:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"unknown arm: {opts.arm}. Declared arms: {', '.join(ARMS)}",
        )

    # Vocab comes FROM THE CORPUS, never pinned. core6_arms defaults to 100,352 because that
    # is what dolma2's 100,278 pads to, but a solved-geometry arm that hard-codes a vocab
    # silently trains a differently-shaped model the moment the corpus changes -- and the
    # per-arm FFN width solve is anchored on the resulting parameter total, so the anchor
    # moves with it.
    vocab = corpus.tokenizer.padded_vocab_size()
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
    # `build_arm` forwards **kwargs to `TransformerConfig.llama_like`, which forwards its own
    # **kwargs to `cls(...)` -- so `init_seed` lands on the dataclass field and reaches
    # `TransformerConfig.build`, which passes it to the `Transformer` constructor
    # (config.py:384). Verified by construction rather than assumed; the regression test is
    # `test_two_init_seeds_give_different_weights` in core6_arms_test.py.
    model_config = build_arm(opts.arm, vocab_size=vocab, init_seed=opts.init_seed)

    arm_spec = ARMS[opts.arm]
    log.info(
        "arm %s (%s): %s params at vocab %d | global attn %s | kda %s | swa %s",
        arm_spec.name,
        arm_spec.title,
        f"{model_config.num_params:,}",
        vocab,
        list(arm_spec.attention_layers),
        list(arm_spec.kda_layers),
        list(arm_spec.swa_layers),
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
        self.tps_device_avg: Optional[float] = None
        #: Instantaneous per-device TPS from the last logged step, for a sanity check that
        #: the average is not still climbing when the probe ends.
        self.tps_device_last: Optional[float] = None

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
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


def fetch_slice_inputs(*, mask_uri: str, work_dir: str):
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

    val_paths, mask_paths = [], []
    for entry in manifest["shards"]:
        shard_local = os.path.join(local, entry["shard"])
        mask_local = os.path.join(local, entry["mask"])
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
                f"{entry['shard']}: {os.path.getsize(shard_local)//4} tokens on disk, "
                f"manifest says {entry['tokens']}"
            )
        with open(mask_local, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()[: len(entry["sha256"])]
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['mask']}: sha256 {digest} != manifest {entry['sha256']}")

        val_paths.append(shard_local)
        mask_paths.append(mask_local)

    total = sum(e["tokens"] for e in manifest["shards"])
    log.info(
        "slice inputs: %d shard(s), %s tokens, C_mass=%s, realized mass %.3f%%",
        len(val_paths),
        f"{total:,}",
        manifest.get("c_mass"),
        100 * manifest.get("realized_mass", 0.0),
    )
    return val_paths, mask_paths


@torch.no_grad()
def evaluate_sliced(*, model, vocab_size, val_paths, mask_paths, seq_len, micro=2):
    """Aggregate and per-band AR-sliced CE over a fixed validation set.

    This is the number the experiment is actually for. Training alone produces a loss curve;
    the contrasts that answer the question -- D = CE(a=4) - CE(a=6), and the seed noise s --
    are differences of *this* quantity between arms, computed on a byte-identical token set.

    Returns sums and counts rather than means, so that arms can be differenced without a
    re-weighting error, and so an unequal token count between arms is visible rather than
    silently invalidating the pairing.

    The mask indexes the CONTINUATION token, so it aligns with the targets and is offset by
    one from the inputs. An off-by-one here scores the wrong positions and still produces
    plausible numbers.

    STILL RANK-GATED BY ITS CALLER, AND STILL NOT THE RUN'S ONLY ENDPOINT. See
    :func:`evaluate_val_aggregate`, which is the one that runs on every rank and produces the
    number a run is required to report. This is kept because the per-band decomposition is what
    the gap-conditioned analysis needs, and it reads frozen masks that have to be built first.
    """
    import numpy as np

    model.eval()
    agg_sum, agg_n = 0.0, 0
    band_sum = {b: 0.0 for b in BAND_BIT}
    band_n = {b: 0 for b in BAND_BIT}
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for vp, mp in zip(val_paths, mask_paths):
        mask = np.memmap(mp, dtype=np.uint8, mode="r")
        n_tokens = _shard_token_count(vp, dtype=np.uint32)
        if n_tokens != mask.size:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"mask/shard length mismatch for {vp}: {mask.size} vs {n_tokens}",
            )
        # Windowing goes through the shared generator rather than a second copy of the same
        # arithmetic, so this and the aggregate evaluator cannot drift on the off-by-one the
        # docstring above is about. `offsets` is what aligns the mask to the TARGETS.
        for offsets, xs, ys in _shard_windows(
            vp, seq_len=seq_len, micro=micro, dtype=np.uint32
        ):
            ms = np.stack(
                [
                    np.asarray(mask[off + 1 : off + seq_len + 1], dtype=np.uint8)
                    for off in offsets
                ]
            )
            m = torch.from_numpy(ms).to(device)
            ce = _forward_ce(model, xs, ys, vocab_size=vocab_size, device=device)
            agg_sum += float(ce.sum())
            agg_n += ce.numel()
            flat = m.reshape(-1)
            for band, bit in BAND_BIT.items():
                selected = (flat & bit) != 0
                if selected.any():
                    band_sum[band] += float(ce[selected].sum())
                    band_n[band] += int(selected.sum())

    model.train()
    return {
        "aggregate": {"sum": agg_sum, "n": agg_n, "ce": agg_sum / max(agg_n, 1)},
        "bands": {
            str(b): {
                "sum": band_sum[b],
                "n": band_n[b],
                "ce": band_sum[b] / max(band_n[b], 1),
            }
            for b in BAND_BIT
        },
    }


def fetch_val_shards(*, val_paths: List[str], work_dir: str, rank: int, world_size: int):
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
    log.info(
        "rank %d holds %d of %d held-out object(s)", rank, len(fetched), len(val_paths)
    )
    return fetched


@torch.no_grad()
def evaluate_val_aggregate(
    *,
    model,
    vocab_size: int,
    val_paths: List[str],
    work_dir: str,
    seq_len: int,
    dtype,
    declared_tokens: Optional[int] = None,
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
    local_paths: List[str] = []
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

    # A rank with NO shards still has to enter every collective, and to do that it needs a
    # tensor of the right shape to push through the model. It cannot borrow one from its own
    # data, because it has none -- so the shape is derived from the parameters that are common
    # to all ranks, which is `seq_len` and `micro`, and the ids are zeros. It is thrown away.
    #
    # This is reachable whenever world_size > len(val_paths): 8 ranks over 60 objects gives
    # every rank work, but a 64-GPU shape or a corpus with 3 val objects does not, and the
    # version of this that asserted "every rank has at least one object" would have been a hang
    # on exactly the shape that is easiest to submit by accident.
    import numpy as np

    filler = np.zeros((max(micro, 1), seq_len), dtype=np.int64)

    was_training = model.training
    model.eval()
    ce_sum, n_tokens, done = 0.0, 0, 0
    try:
        for path in local_paths:
            for _, xs, ys in _shard_windows(path, seq_len=seq_len, micro=micro, dtype=dtype):
                ce = _forward_ce(model, xs, ys, vocab_size=vocab_size, device=device)
                ce_sum += float(ce.sum())
                n_tokens += int(ce.numel())
                done += 1
        # THE PADDING PASSES. Real collective traffic, discarded arithmetic. Without them a rank
        # that got 7 shards where its peer got 8 leaves the loop one forward early, reaches the
        # all_reduce below while the peer is still inside an all-gather, and the job hangs at
        # the very end of a run that has already been paid for.
        while done < steps:
            _forward_ce(model, filler, filler, vocab_size=vocab_size, device=device)
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
            "way, which is why this is checked rather than reported.",
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
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
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
                # THE THROUGHPUT NUMBER TO COST A RUN ON. Steady-state, per device, with
                # step 1 excluded, so it does not charge process start, dataset open, FSDP
                # wrap or the first-step compile against the hardware. Multiply by
                # world_size for the machine.
                "tps_device_avg": losses.tps_device_avg,
                "tps_device_last": losses.tps_device_last,
                "world_size": get_world_size(),
                "tps_total_avg": (
                    None
                    if losses.tps_device_avg is None
                    else losses.tps_device_avg * get_world_size()
                ),
                # Kept for contrast, and it is the WRONG number for costing: it includes
                # every fixed cost above. On a short probe it can be several times lower
                # than the steady-state figure, and it penalises bigger shapes hardest.
                "tps_naive_wall_clock": (
                    None
                    if not seconds
                    else trainer.global_step * opts.global_batch_size / seconds
                ),
                "peak_memory_gib": peak,
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
    elapsed = time.monotonic() - started

    # THE HELD-OUT ENDPOINT, ON EVERY RANK, WITH NO FLAG TO TURN IT ON AND NO `except` AROUND
    # IT. Everything the experiment reports is a difference of this number between arms, so a
    # run that trained and produced no CE produced a checkpoint nobody can use to answer the
    # question -- and the version of this code that did exactly that exited 0 with a null field,
    # which reads in the record as "this arm was measured".
    #
    # It runs BEFORE summarise() because summarise() prints it, and before the `finally` in
    # main() that tears the process group down, because it needs collectives.
    val = None
    if opts is not None:
        val = evaluate_val_aggregate(
            model=trainer.train_module.model,
            vocab_size=config.model.vocab_size,
            val_paths=list(config.val_paths),
            work_dir=opts.work_dir,
            seq_len=opts.sequence_length,
            dtype=config.dataset.dtype.as_np_dtype(),
            declared_tokens=config.val_rows,
        )
        # A MAGNITUDE CHECK ON THE ENDPOINT, RAISED RATHER THAN LOGGED. A CE over a quarter of
        # the val set is in the normal range and would be believed. Every rank calls this on the
        # same all-reduced numbers, so either all of them raise or none do -- there is no
        # topology in which one rank refuses and the others go on to a collective it left.
        assert_val_tokens_account_for_the_corpus(val)

    # The sliced evaluation is SECONDARY and remains rank-zero, because it reads frozen masks
    # that must be built first and it decomposes by gap band rather than producing the headline
    # number. Its rank gate is the same defect described in `evaluate_val_aggregate` and it is
    # still here: under FSDP a rank-zero-only forward waits on all-gathers the other ranks never
    # enter. It is reached only when --slice-mask-uri is passed, which the current wave does not
    # pass; the barrier below at least holds the other ranks inside the collective world while
    # rank zero works, which is what makes it survivable rather than correct.
    sliced = None
    if opts is not None and opts.slice_mask_uri:
        try:
            if get_rank() == 0:
                val_paths, mask_paths = fetch_slice_inputs(
                    mask_uri=opts.slice_mask_uri, work_dir=opts.work_dir
                )
                log.info("sliced eval over %d shard(s)", len(val_paths))
                sliced = evaluate_sliced(
                    model=trainer.train_module.model,
                    vocab_size=config.model.vocab_size,
                    val_paths=val_paths,
                    mask_paths=mask_paths,
                    seq_len=opts.sequence_length,
                )
                log.info(
                    "aggregate CE %.4f over %s tokens",
                    sliced["aggregate"]["ce"],
                    f"{sliced['aggregate']['n']:,}",
                )
                for band in sorted(BAND_BIT):
                    entry = sliced["bands"][str(band)]
                    if entry["n"]:
                        log.info(
                            "  gap>%-5s CE %.4f over %s tokens",
                            band,
                            entry["ce"],
                            f"{entry['n']:,}",
                        )
        except Exception as error:  # never lose a trained checkpoint to a SECONDARY eval bug
            log.warning(
                "sliced eval failed (%s: %s); checkpoint is still on S3 and val_ce above is "
                "unaffected",
                type(error).__name__,
                error,
            )
        finally:
            # In the `finally` so a rank-zero failure does not leave the other ranks waiting
            # here forever -- they would time out on the barrier instead of on a collective
            # inside a forward, which at least fails.
            barrier()

    if opts is not None:
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
        default="L0",
        help="CORE-6 arm name from olmo_core.nn.transformer.core6_arms.ARMS "
        "(L0, K2, G4R0, G4R2, G2R0, S14, G0R0). Replaces --model-factory.",
    )
    parser.add_argument(
        "--slice-mask-uri",
        default=os.environ.get("EDULLM_SLICE_MASK_URI", ""),
        help="S3 prefix holding slice_manifest.json and the frozen *.mask.u8 files from "
        "build_slice_masks.py. The manifest names the corpus shards to pair them with and "
        "carries a digest per mask, both of which are checked. Empty skips the evaluation "
        "and the run produces a checkpoint only.",
    )
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--global-batch-size", type=int, default=256 * 1024)
    parser.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    parser.add_argument("--data-seed", type=int, default=0)
    # Weight init is a SEPARATE variance component from data order, and until this flag
    # existed only `--data-seed` was exposed while `init_seed` stayed at its 12536 default.
    # Every "n seeds" on this entry point was therefore n data orderings of ONE
    # initialisation -- a narrower component than the FarmShare seed replicate measured,
    # which biases any CI built from it optimistically. A paired design wants both varied.
    parser.add_argument("--init-seed", type=int, default=12536)
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, do not train.")
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
