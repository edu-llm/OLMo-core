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
from dataclasses import dataclass, replace
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
from olmo_core.distributed.utils import barrier, get_rank, get_world_size
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

    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths=list(read.paths),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=tokenizer,
        rows=read.rows,
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
    # either would look like a training failure. Held-out shards for a published corpus come
    # back from the reader as `.val`, and wiring an evaluator to those is the right version of
    # this -- it needs a corpus that declares one, which regmix-10b does not.

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        init_seed=opts.init_seed,
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
    """
    import numpy as np

    model.eval()
    agg_sum, agg_n = 0.0, 0
    band_sum = {b: 0.0 for b in BAND_BIT}
    band_n = {b: 0 for b in BAND_BIT}

    for vp, mp in zip(val_paths, mask_paths):
        tokens = np.memmap(vp, dtype=np.uint32, mode="r")
        mask = np.memmap(mp, dtype=np.uint8, mode="r")
        if tokens.size != mask.size:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"mask/shard length mismatch for {vp}: {mask.size} vs {tokens.size}",
            )
        windows = (tokens.size - 1) // seq_len
        for start in range(0, windows, micro):
            count = min(micro, windows - start)
            xs, ys, ms = [], [], []
            for w in range(start, start + count):
                off = w * seq_len
                seg = np.asarray(tokens[off : off + seq_len + 1], dtype=np.int64)
                xs.append(seg[:-1])
                ys.append(seg[1:])
                ms.append(np.asarray(mask[off + 1 : off + seq_len + 1], dtype=np.uint8))
            x = torch.from_numpy(np.stack(xs)).cuda()
            y = torch.from_numpy(np.stack(ys)).cuda()
            m = torch.from_numpy(np.stack(ms)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                logits = out.logits if hasattr(out, "logits") else out
            ce = _chunked_ce(logits.reshape(-1, vocab_size), y.reshape(-1))
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


def summarise(*, opts, config, trainer, losses: LossWatcher, seconds: float, sliced=None) -> None:
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

    # The sliced evaluation, on rank zero only. Everything the experiment is for is a
    # difference of these numbers between arms, so a run that trained and did not evaluate
    # produces a checkpoint nobody can use to answer the question. Wrapped because a failure
    # here must not discard four hours of training: the checkpoint is already on S3 and the
    # eval can be redone from it, so this warns and continues rather than raising.
    sliced = None
    if opts is not None and opts.slice_mask_uri and get_rank() == 0:
        try:
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
        except Exception as error:  # never lose a trained checkpoint to an eval bug
            log.warning(
                "sliced eval failed (%s: %s); checkpoint is still on S3",
                type(error).__name__,
                error,
            )

    if opts is not None:
        summarise(
            opts=opts,
            config=config,
            trainer=trainer,
            losses=losses,
            seconds=elapsed,
            sliced=sliced,
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
