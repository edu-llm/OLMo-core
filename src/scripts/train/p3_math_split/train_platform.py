# ruff: noqa: E501
"""One arm of the dense/split comparison, on the eduLLM platform.

**This file is a copy of `.edullm/train_on_corpus.py` with four changes.** Diff it
against that file before editing; everything not listed here — the corpus resolver,
the refusal staging, the torn-checkpoint repair, the W&B reporting, the checkpointer
settings that keep a twelve-hour run alive — is inherited deliberately and should not
be re-derived:

  1. `--arm` and `--config`. Hyperparameters come from `configs/{dense,split}.yaml`,
     which differ on one line, rather than from flags, so the two arms cannot drift.
  2. `DerivedMaskTrainModuleConfig` replaces `TransformerTrainModuleConfig`. It pins
     the loss divisor (otherwise the arms differ in effective learning rate as well
     as in the mask) and recomputes the fact block from the token stream.
  3. The separator token ids are resolved from the corpus's own tokenizer and passed
     to the train module. No mask sidecar is published or read.
  4. `compile_model` follows the arm config, which keeps it off until
     `src/test/nn/transformer/qwen_test.py` passes compiled.

Paste commands into the platform as one physical line. Configuration preflight:

    bash -lc 'python src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm split --config src/scripts/train/p3_math_split/configs/split.yaml --save-folder "$EDULLM_CHECKPOINT_DIR" --dry-run'

Final eight-rank launch (the platform does not add a launcher):

    bash -lc 'python -m torch.distributed.run --nproc-per-node=8 --standalone src/scripts/train/p3_math_split/train_platform.py "$EDULLM_RUN_ID" --arm split --config src/scripts/train/p3_math_split/configs/split.yaml --save-folder "$EDULLM_CHECKPOINT_DIR"'
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
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, cast

import rich
import torch
import yaml

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
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.lm_head import LMLossImplementation
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.qwen import (
    QWEN2_0_5B_HF_ID,
    QWEN2_0_5B_HF_REVISION,
    QWEN2_0_5B_HF_WEIGHTS_SHA256,
    QWEN2_0_5B_HF_WEIGHTS_SIZE,
    load_hf_weights,
    parameter_report,
    qwen2_0_5b_config,
    qwen2_tokenizer_config,
    strip_attn_out_bias,
)
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
    Duration,
    DurationUnit,
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import TOKENIZER_ARTIFACT, fetch_tokenizer_artifact
from train_module import DerivedMaskTrainModuleConfig

log = logging.getLogger(__name__)

P3_MODEL_FACTORY = "qwen2_0_5b"
P3_DATASET_ID = "pretrain/formal-proof-premises-500m"
P3_SEED = 42
P3_LOSS_IMPLEMENTATION = LMLossImplementation.fused_linear
P3_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
EXPECTED_SEPARATOR_IDS = [10952, 15513, 969]
P3_LAUNCH_CONTRACT: Dict[str, Any] = {
    "schema_version": 1,
    "supported_compute_profiles": ["gpu-8xa100", "gpu-8xh100"],
    "recommended_compute_profile": "gpu-8xh100",
    "final_world_size": 8,
    "launcher": "python -m torch.distributed.run",
    "config_preflight_compute_profile": "gpu-1xa10g",
    "config_preflight_world_size": 1,
    "enforced_by": "eduLLM platform compute-profile/process guard",
}
P3_RUNTIME_SMOKE = {
    "max_steps": 100,
    "warmup_steps": 10,
    "save_every": 50,
}


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
    # Before process-group initialization and after teardown, get_rank() returns zero in
    # every torchrun child. RANK remains valid through both phases, so consult it first.
    if not is_global_rank_zero():
        return

    # The submission form's `wandb_project` field is injected as WANDB_PROJECT
    # according to the platform guide. Keep the legacy EDULLM_* spelling as a fallback
    # because older images used it.
    project = os.environ.get("WANDB_PROJECT") or os.environ.get("EDULLM_WANDB_PROJECT")
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


def is_global_rank_zero() -> bool:
    """Work before distributed init and after teardown without duplicating rank-zero work."""
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            return False
    return get_rank() == 0


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
    # This corpus is written with Qwen 2.5's tokenizer, vendored and published so the
    # shards stay decodable if the upstream repo moves.
    "tokenizer/qwen25-vendored": qwen2_tokenizer_config,
}

# What the corpus builder writes between the fact block and the goal.
SEPARATOR = "\n---\nGOAL "
# What the split arm actually SEARCHES for, and it is deliberately not the string
# above. BPE does not respect the boundary: the trailing space merges rightward into
# the goal's first word (` |-`, ` ![`, ` lemma`) in 98.4% of documents, and the
# leading newline merges leftward into the fact block's last characters (` )\n`,
# `"\n`) in 88.5%. Encoding the full separator gives [198, 10952, 15513, 969, 220],
# a run that survives in 777 of 258,316 documents -- 0.30%, and 0% in four of the six
# shards. The split arm would have found no boundary and supervised everything,
# silently becoming a second dense arm.
#
# The three-token core `---\nGOAL` -> [10952, 15513, 969] survives in 258,316 of
# 258,316, never appears twice, and the token immediately after it always begins at
# or past mask_end -- so supervision starts exactly at the goal.
SEPARATOR_SEARCH = "---\nGOAL"


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
    arm: str = ""
    run_mode: str = "train"
    model_factory: str = ""
    loss_implementation: str = ""
    base_model_id: str = ""
    base_model_revision: str = ""
    base_model_weight_sha256: str = ""
    base_model_weight_size: int = 0
    tokenizer_artifact_id: str = ""
    tokenizer_artifact_version: str = ""
    tokenizer_file_sha256: Dict[str, str] = field(default_factory=dict)
    tokenizer_composite_sha256: str = ""
    tokenizers_version: str = ""
    tokenizer_eos_token_id: int = -1
    tokenizer_pad_token_id: int = -1
    dataset_id: str = ""
    dataset_version: str = ""
    dataset_release: str = ""
    world_size: int = 1
    launch_contract: Dict[str, Any] = field(default_factory=dict)
    source_commit: str = ""
    platform_run_manifest_id: str = ""
    platform_run_manifest_sha256: str = ""
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

    # Published dependencies arrive pinned as tokenizer/<name>/vN, while the
    # architecture registry is keyed by version-independent tokenizer identity.
    tokenizer_config_id = re.sub(r"/v\d+$", "", tokenizer_id)
    try:
        tokenizer = TOKENIZERS[tokenizer_config_id]()
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS)) or "none"
        raise Refusal(
            Stage.THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER,
            f"no OLMo-core config for {tokenizer_id} "
            f"(normalized to {tokenizer_config_id}); this image knows: {known}",
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

    # Explicit, because edullm_data v0.2.0 returns EVERY manifest entry when
    # split=None. This release carries train and val shards in one group, so the
    # copied reference script's default would silently train on held-out data.
    # THE STAGE THAT ACTUALLY TOUCHES THE ACCOUNT, AND THE ONE WORTH TELLING APART FROM THE
    # REST. Everything above this line is local. This call HEADs the seal, GETs the manifest
    # and lists the group, so it is where a missing s3:GetObject on edullm-data shows up --
    # and a role without that grant and a registry entry pointing at an unpublished prefix
    # both arrive here as a failed read. read_failure separates them.
    try:
        read = dataset_paths(dataset_id, version, split="train", s3=s3)
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


def separator_ids_for(tokenizer_id: str, work_dir: str) -> List[int]:
    """Token ids of SEPARATOR under the corpus's own published tokenizer.

    This compatibility wrapper resolves the same fully sealed artifact used by
    :func:`build_config`; it never falls back to HuggingFace or a latest alias.
    """
    try:
        tokenizer = fetch_tokenizer_artifact(tokenizer_id, work_dir)
    except (OSError, RuntimeError, ValueError) as error:
        raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, str(error)) from error
    return tokenizer.separator_ids(SEPARATOR_SEARCH)


def reject_config_overrides(overrides: List[str]) -> None:
    """Refuse post-YAML config mutation, including unknown flags and dotlists."""
    if overrides:
        rendered = " ".join(overrides)
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "P3 scientific controls cannot be overridden after the canonical YAML; "
            f"remove this override/unknown argument: {rendered}",
        )


def validate_submission_controls(opts, *, require_seed: bool = True) -> None:
    """Validate identities that define the P3 experiment rather than accepting lookalikes."""
    expected_config = (P3_CONFIG_DIR / f"{opts.arm}.yaml").resolve()
    supplied_config = Path(opts.config).resolve()
    if supplied_config != expected_config:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"arm {opts.arm!r} must use its canonical config {expected_config}; "
            f"got {supplied_config}",
        )
    if opts.model_factory != P3_MODEL_FACTORY:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 model factory is fixed to {P3_MODEL_FACTORY!r}; got {opts.model_factory!r}",
        )
    if opts.dataset_id != P3_DATASET_ID:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 dataset ID is fixed to {P3_DATASET_ID!r}; got {opts.dataset_id!r}",
        )
    if re.fullmatch(r"v[1-9]\d*", opts.dataset_version or "") is None:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "P3 dataset version must be an explicit immutable vN submission value; "
            f"got {opts.dataset_version!r}",
        )
    if opts.dataset_tokenizer != TOKENIZER_ARTIFACT:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 tokenizer is fixed to {TOKENIZER_ARTIFACT!r}; " f"got {opts.dataset_tokenizer!r}",
        )
    if require_seed and opts.data_seed != P3_SEED:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 seed is fixed to {P3_SEED}; got {opts.data_seed}",
        )


def declared_world_size() -> int:
    """Read torchrun's pre-init WORLD_SIZE without requiring a process group."""
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"WORLD_SIZE must be a positive integer, got {raw!r}",
        ) from error
    if world_size < 1:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"WORLD_SIZE must be positive, got {world_size}",
        )
    return world_size


def validate_runtime_launch_contract() -> None:
    """Refuse non-dry P3 execution unless torchrun declares the final 8-rank process set."""

    def required_process_integer(name: str) -> int:
        raw = os.environ.get(name)
        if raw is None:
            raise Refusal(
                Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
                f"{name} is required for the final 8-rank P3 launch",
            )
        try:
            return int(raw)
        except ValueError as error:
            raise Refusal(
                Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
                f"{name} must be an integer, got {raw!r}",
            ) from error

    expected_world_size = int(P3_LAUNCH_CONTRACT["final_world_size"])
    world_size = required_process_integer("WORLD_SIZE")
    if world_size != expected_world_size:
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            f"WORLD_SIZE must equal {expected_world_size} for non-dry P3 execution; "
            f"got {world_size}",
        )
    local_world_size = required_process_integer("LOCAL_WORLD_SIZE")
    if local_world_size != expected_world_size:
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            f"LOCAL_WORLD_SIZE must equal {expected_world_size} for the single-node "
            "8-GPU P3 contract; "
            f"got {local_world_size}",
        )
    rank = required_process_integer("RANK")
    local_rank = required_process_integer("LOCAL_RANK")
    if not 0 <= rank < world_size:
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            f"RANK must be in [0, {world_size}), got {rank}",
        )
    if not 0 <= local_rank < local_world_size:
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            f"LOCAL_RANK must be in [0, {local_world_size}), got {local_rank}",
        )
    if rank != local_rank:
        raise Refusal(
            Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START,
            f"single-node P3 launch requires RANK == LOCAL_RANK, got {rank} != {local_rank}",
        )


def apply_arm_config(opts) -> dict:
    """Overwrite the flag defaults with the arm config, and refuse a mismatch.

    Both arms read a file whose `shared:` block is byte-identical; only `arm` differs.
    Taking the hyperparameters from there rather than from the command line is what
    stops the two runs being given different values by a slip on one submission.
    """
    with open(opts.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg.get("arm") != opts.arm:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"--arm {opts.arm} but {opts.config} declares arm={cfg.get('arm')!r}",
        )
    shared = cfg["shared"]
    if shared.get("seed") != P3_SEED:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 seed is fixed to {P3_SEED}; {opts.config} declares " f"{shared.get('seed')!r}",
        )
    if shared.get("runtime_smoke") != P3_RUNTIME_SMOKE:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"{opts.config} must declare the fixed runtime_smoke profile " f"{P3_RUNTIME_SMOKE!r}",
        )
    opts.sequence_length = shared["sequence_length"]
    opts.global_batch_size = shared["global_batch_size_sequences"] * opts.sequence_length
    opts.rank_microbatch_size = shared["rank_microbatch_size_sequences"] * opts.sequence_length
    try:
        opts.loss_implementation = LMLossImplementation(shared["loss_implementation"])
    except (KeyError, ValueError) as error:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"{opts.config} must declare a recognized loss_implementation",
        ) from error
    if opts.loss_implementation != P3_LOSS_IMPLEMENTATION:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"P3 Qwen loss implementation is fixed to {P3_LOSS_IMPLEMENTATION.value!r}; "
            f"{opts.config} declares {opts.loss_implementation.value!r}",
        )
    opts.learning_rate = shared["learning_rate"]
    opts.warmup_steps = shared["warmup_steps"]
    opts.data_seed = shared["seed"]
    opts.compile_model = bool(shared.get("compile_model", False))
    opts.wandb_project = shared.get("wandb_project")
    opts.num_workers = shared["num_workers"]
    opts.log_every = shared["log_every"]
    opts.save_interval = shared["save_every"]
    opts.save_overwrite = bool(shared["save_overwrite"])
    opts.tie_embeddings = bool(shared["tie_embeddings"])
    opts.betas = tuple(shared["betas"])
    opts.eps = float(shared["eps"])
    opts.weight_decay = float(shared["weight_decay"])
    opts.max_grad_norm = float(shared["max_grad_norm"])
    opts.lr_alpha_f = float(shared["lr_alpha_f"])
    return shared


def wandb_project(opts) -> Optional[str]:
    """Where to report, checked in three places rather than one.

    The reference script reads EDULLM_WANDB_PROJECT, while the platform guide lists
    WANDB_PROJECT among the container's variables. I could not reconcile the two from
    what is in this workspace, and the failure is silent — a run with the wrong name
    trains for hours and reports nothing. So try the arm config first (explicit beats
    ambient), then both variable spellings.
    """
    return (
        getattr(opts, "wandb_project", None)
        or os.environ.get("EDULLM_WANDB_PROJECT")
        or os.environ.get("WANDB_PROJECT")
    )


def loader_epoch_step_counts(rows: int, global_batch_size: int, epochs: int) -> tuple[int, int]:
    """Return complete loader batches per epoch and across all epochs.

    The numpy loader drops each epoch's incomplete global batch. Flooring only
    after multiplying by epochs would carry every dropped tail into later epochs
    and enter an extra epoch near the end of training.
    """
    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    if global_batch_size <= 0:
        raise ValueError(f"global_batch_size must be positive, got {global_batch_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    batches_per_epoch = rows // global_batch_size
    return batches_per_epoch, batches_per_epoch * epochs


def validate_epoch_horizon(trainer, expected_steps: int) -> None:
    """Verify the built loader resolves the manifest-derived epoch horizon."""
    if trainer.max_duration.unit != DurationUnit.epochs:
        return
    actual_steps = trainer.max_steps
    if actual_steps != expected_steps:
        raise RuntimeError(
            f"loader resolves {actual_steps} steps for {trainer.max_duration.value} epochs, "
            f"but the manifest-derived horizon is {expected_steps}"
        )


def build_config(opts, overrides: List[str], *, validate_controls: bool = True):
    reject_config_overrides(overrides)
    if validate_controls:
        validate_submission_controls(opts, require_seed=False)
    shared = apply_arm_config(opts)
    if validate_controls:
        validate_submission_controls(opts)

    source_commit = os.environ.get("EDULLM_COMMIT_SHA", "").strip()
    if not opts.dry_run and not source_commit:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "EDULLM_COMMIT_SHA is required for a non-dry reportable run",
        )
    platform_run_manifest_id = os.environ.get("EDULLM_RUN_MANIFEST_ID", "").strip()
    platform_run_manifest_sha256 = os.environ.get("EDULLM_RUN_MANIFEST_SHA256", "").strip()
    if platform_run_manifest_sha256:
        if not platform_run_manifest_id:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                "platform run manifest SHA-256 requires EDULLM_RUN_MANIFEST_ID",
            )
        if re.fullmatch(r"[0-9a-f]{64}", platform_run_manifest_sha256) is None:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                "platform run manifest SHA-256 must be lowercase 64-hex",
            )

    try:
        sealed_tokenizer = fetch_tokenizer_artifact(opts.dataset_tokenizer, opts.work_dir)
    except (OSError, RuntimeError, ValueError) as error:
        raise Refusal(Stage.THE_CONFIG_WOULD_NOT_BUILD, str(error)) from error
    corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    corpus = replace(corpus, tokenizer=sealed_tokenizer.olmo_config())
    log.info(
        "%s/%s: %d shards, dtype %s, tokenizer %s",
        corpus.dataset_id,
        corpus.version,
        len(corpus.paths),
        corpus.dtype,
        opts.dataset_tokenizer,
    )

    configured_max_steps = shared.get("max_steps")
    if opts.runtime_smoke:
        opts.steps = int(shared["runtime_smoke"]["max_steps"])
        opts.warmup_steps = int(shared["runtime_smoke"]["warmup_steps"])
        opts.save_interval = int(shared["runtime_smoke"]["save_every"])
        max_duration = Duration.steps(opts.steps)
    elif configured_max_steps is not None:
        opts.steps = int(configured_max_steps)
        max_duration = Duration.steps(opts.steps)
    else:
        if corpus.rows is None:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                "the corpus manifest declares no row count, so exact loader epochs "
                "cannot be derived at runtime",
            )
        epochs = int(shared["epochs"])
        batches_per_epoch, opts.steps = loader_epoch_step_counts(
            corpus.rows, opts.global_batch_size, epochs
        )
        max_duration = Duration.epochs(epochs)
        log.info(
            "%d rows // %d tokens = %d complete loader batches/epoch; " "%d epochs = %d steps",
            corpus.rows,
            opts.global_batch_size,
            batches_per_epoch,
            epochs,
            opts.steps,
        )
    if opts.steps < 1:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"0 steps: {corpus.rows} tokens / {opts.global_batch_size} per step",
        )
    if opts.warmup_steps >= opts.steps:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"warmup_steps ({opts.warmup_steps}) >= max_steps ({opts.steps}): the "
            "schedule would never reach peak nor decay. Lower it in BOTH configs.",
        )

    # Qwen2.5-0.5B is NOT a TransformerConfig classmethod -- only the qwen3_* ones are.
    # It is a module-level factory with a different signature (no vocab_size; the
    # architecture fixes it at 151,936), so the reference script's getattr lookup finds
    # nothing and refuses. Special-case it rather than pretend the shapes match.
    #
    # qwen2_0_5b_config sets bias=True on attention, which gives w_out a bias
    # Qwen2 does not have. train() calls strip_attn_out_bias on the built model
    # BEFORE it is moved, sharded or compiled, then loads the HF weights strictly.
    if opts.model_factory == "qwen2_0_5b":
        model_config = qwen2_0_5b_config(
            init_seed=shared["seed"], tie_word_embeddings=opts.tie_embeddings
        )
        if model_config.lm_head is None:  # pragma: no cover - Qwen architecture invariant
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                "Qwen2.5-0.5B config has no LM head for the fused-linear loss",
            )
        model_config.lm_head.loss_implementation = opts.loss_implementation
        # TorchAttentionBackend rejects cu_doc_lens at first forward. The platform
        # image installs flash-attn 2 explicitly in .edullm/Dockerfile; naming it
        # here turns a missing/incompatible install into a config-build failure
        # instead of a GPU run that dies after setup.
        model_config.block.sequence_mixer.backend = AttentionBackendName.flash_2
        if model_config.vocab_size != corpus.tokenizer.padded_vocab_size():
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"Qwen2.5-0.5B has vocab {model_config.vocab_size} but the corpus "
                f"tokenizer pads to {corpus.tokenizer.padded_vocab_size()}; ids would "
                "index outside the embedding or waste rows",
            )
        factory = None
    else:
        factory = getattr(TransformerConfig, opts.model_factory, None)
        if factory is None:
            raise Refusal(
                Stage.THE_CONFIG_WOULD_NOT_BUILD,
                f"unknown model factory: {opts.model_factory}",
            )

    # padded rather than exact for the same reason the example pads: a vocab that is a
    # multiple of 128 keeps the embedding matmul on a fast path. dolma2's 100,278 pads to
    # 100,352.
    if factory is not None:
        model_config = factory(vocab_size=corpus.tokenizer.padded_vocab_size())

    dataset_config = NumpyFSLDatasetConfig(
        paths=corpus.paths,
        sequence_length=opts.sequence_length,
        tokenizer=corpus.tokenizer,
        # The whole point of this file. See the header.
        dtype=corpus.dtype,
        work_dir=opts.work_dir,
        # Intra-document attention, and NOT an optimisation. The shards are packed --
        # several proofs share one 16,384-token sequence -- so without this every proof
        # attends to its unrelated neighbours. It is also most of the cost: attention is
        # 59% of FLOPs at this sequence length, and masking takes the effective
        # attention span from 16,384 down to the token-weighted mean document length of
        # 6,712, which measures as a 1.53x end-to-end speedup.
        #
        # Boundaries are found by EOS, which tokenize_corpus appends to every document.
        generate_doc_lengths=True,
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=opts.global_batch_size,
        seed=opts.data_seed,
        num_workers=opts.num_workers,
    )

    # The separator the split arm searches for. Resolved from the corpus's OWN
    # tokenizer -- a different one would tokenize it to different ids, the search
    # would never match, and the split arm would silently become a second dense arm.
    sep_ids = sealed_tokenizer.separator_ids(SEPARATOR_SEARCH)
    if not sep_ids:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"{SEPARATOR_SEARCH!r} tokenizes to nothing under {opts.dataset_tokenizer}",
        )
    if sep_ids != EXPECTED_SEPARATOR_IDS:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"{SEPARATOR_SEARCH!r} resolved to {sep_ids}, expected "
            f"{EXPECTED_SEPARATOR_IDS} under the approved tokenizer seal",
        )

    train_module_config = DerivedMaskTrainModuleConfig(
        arm=opts.arm,
        separator_ids=sep_ids,
        eos_token_id=corpus.tokenizer.eos_token_id,
        pad_token_id=corpus.tokenizer.pad_token_id,
        # Global control, identical across arms and steps. The module converts it
        # to the nominal rank batch before FSDP averages data-parallel gradients;
        # OLMo-core's default instead uses the live-token count changed by the mask.
        fixed_loss_div_factor=float(opts.global_batch_size),
        rank_microbatch_size=opts.rank_microbatch_size,
        max_sequence_length=opts.sequence_length,
        optim=AdamWConfig(
            lr=opts.learning_rate,
            betas=opts.betas,
            eps=opts.eps,
            weight_decay=opts.weight_decay,
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        # On, because the image now carries a C compiler. It was off in the platform's
        # getting-started command only because a run without one dies on the first compiled
        # region, which is a workaround that costs throughput on every run forever.
        compile_model=opts.compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp, param_dtype=DType.bfloat16, reduce_dtype=DType.float32
        ),
        max_grad_norm=opts.max_grad_norm,
        # `warmup`, not the `warmup_steps` the example still passes -- that spelling is
        # deprecated upstream and warns on every construction.
        scheduler=CosWithWarmup(warmup=opts.warmup_steps, alpha_f=opts.lr_alpha_f),
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
            save_overwrite=opts.save_overwrite,
            metrics_collect_interval=opts.log_every,
            cancel_check_interval=5,
            # Explicit, as the platform guide requires: the OLMo-core default is one
            # epoch, which here would be 1/13th of the intended run.
            max_duration=max_duration,
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
                project=wandb_project(opts),
                # No `group`. The platform puts the experiment in WANDB_RUN_GROUP, which the
                # wandb client reads on its own; passing it again from an environment variable
                # that does not exist would set it to None and look deliberate.
                cancel_check_interval=10,
                # Enabled only when the platform named a project, so running this image by
                # hand does not fail on a missing WANDB_API_KEY.
                enabled=bool(wandb_project(opts)),
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

    tokenizer_provenance = sealed_tokenizer.provenance_dict()
    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        arm=opts.arm,
        run_mode=(
            "dry-run" if opts.dry_run else "runtime-smoke" if opts.runtime_smoke else "train"
        ),
        model_factory=opts.model_factory,
        loss_implementation=opts.loss_implementation.value,
        base_model_id=QWEN2_0_5B_HF_ID,
        base_model_revision=QWEN2_0_5B_HF_REVISION,
        base_model_weight_sha256=QWEN2_0_5B_HF_WEIGHTS_SHA256,
        base_model_weight_size=QWEN2_0_5B_HF_WEIGHTS_SIZE,
        tokenizer_artifact_id=tokenizer_provenance["tokenizer_artifact_id"],
        tokenizer_artifact_version=tokenizer_provenance["tokenizer_artifact_version"],
        tokenizer_file_sha256=tokenizer_provenance["tokenizer_file_sha256"],
        tokenizer_composite_sha256=tokenizer_provenance["tokenizer_composite_sha256"],
        tokenizers_version=tokenizer_provenance["tokenizers_version"],
        tokenizer_eos_token_id=tokenizer_provenance["tokenizer_eos_token_id"],
        tokenizer_pad_token_id=tokenizer_provenance["tokenizer_pad_token_id"],
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        dataset_release=os.environ.get("EDULLM_DATASET_RELEASE", ""),
        world_size=declared_world_size(),
        launch_contract=dict(P3_LAUNCH_CONTRACT),
        source_commit=source_commit,
        platform_run_manifest_id=platform_run_manifest_id,
        platform_run_manifest_sha256=platform_run_manifest_sha256,
        init_seed=shared["seed"],
    )
    return config


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
    environment_world = int(os.environ.get("WORLD_SIZE", "1"))
    environment_local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    observed_world = max(get_world_size(), environment_world)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    print(
        json.dumps(
            {
                "run_id": opts.run_name,
                "arm": config.arm,
                "run_mode": config.run_mode,
                "model_factory": config.model_factory,
                "loss_implementation": config.loss_implementation,
                "base_model_id": config.base_model_id,
                "base_model_revision": config.base_model_revision,
                "base_model_weight_sha256": config.base_model_weight_sha256,
                "base_model_weight_size": config.base_model_weight_size,
                "tokenizer_artifact_id": config.tokenizer_artifact_id,
                "tokenizer_artifact_version": config.tokenizer_artifact_version,
                "tokenizer_file_sha256": config.tokenizer_file_sha256,
                "tokenizer_composite_sha256": config.tokenizer_composite_sha256,
                "tokenizers_version": config.tokenizers_version,
                "tokenizer_eos_token_id": config.tokenizer_eos_token_id,
                "tokenizer_pad_token_id": config.tokenizer_pad_token_id,
                "dataset_id": config.dataset_id,
                "dataset_version": config.dataset_version,
                "dataset_release": config.dataset_release,
                "world_size": observed_world,
                "local_world_size": environment_local_world,
                "declared_world_size": config.world_size,
                "launch_contract": config.launch_contract,
                "source_commit": config.source_commit,
                "platform_run_manifest_id": config.platform_run_manifest_id,
                "platform_run_manifest_sha256": config.platform_run_manifest_sha256,
                "gpu": device,
                "cuda_device_count": torch.cuda.device_count(),
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
                "wandb_project": wandb_project(opts) or "",
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

    is_qwen = opts is not None and opts.model_factory == "qwen2_0_5b"
    model = config.model.build(init_device="meta")
    if is_qwen:
        # Build the config that `--dry-run` printed, rather than calling a helper
        # that creates a fresh config and would silently discard our explicit
        # FlashAttention2 backend. The output bias has to be removed before FSDP
        # captures the parameter set.
        strip_attn_out_bias(model)
    train_module = config.train_module.build(model)
    if is_qwen:
        # TransformerTrainModule construction calls model.init_weights(), whose
        # to_empty() deliberately replaces every parameter. Loading HF before that
        # point silently produced a random-init run. Install the full pretrained
        # state afterwards; DCP shards it into the already-wrapped FSDP2 model.
        load_hf_weights(train_module.model, distributed_state_dict=True)
        log.info("model: %s", parameter_report(train_module.model))
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)
    if opts is not None:
        validate_epoch_horizon(trainer, opts.steps)

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
    parser.add_argument("--arm", required=True, choices=("dense", "split"))
    parser.add_argument(
        "--config",
        required=True,
        help="configs/{dense,split}.yaml. Every hyperparameter comes from here so the "
        "arms cannot be given different values by editing one file.",
    )
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
        "--runtime-smoke",
        action="store_true",
        help="Use the closed 100-step verification profile declared in the canonical YAML.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print, do not train.")
    # Scientific controls are deliberately not command-line options. apply_arm_config()
    # fills these from the canonical arm YAML after parse_cli_args() has rejected every
    # unknown flag and dotlist.
    parser.set_defaults(model_factory=P3_MODEL_FACTORY, data_seed=P3_SEED)
    return parser


def parse_cli_args(argv: Optional[List[str]] = None):
    """Parse the closed P3 CLI and reject unknown/dotlist overrides."""
    opts, overrides = build_parser().parse_known_args(argv)
    reject_config_overrides(overrides)
    return opts


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    opts = parse_cli_args()

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

    if (
        is_global_rank_zero()
        and opts.dataset_id == P3_DATASET_ID
        and opts.dataset_version == "v2"
    ):
        log.warning(
            "SCIENTIFIC WARNING: %s/v2 is scientifically stale and forbidden for final "
            "conclusions. Continuing only because the user selected the warning-only policy.",
            P3_DATASET_ID,
        )

    with during(Stage.THE_CONFIG_WOULD_NOT_BUILD):
        config = build_config(opts, [])
    if opts.dry_run:
        show(config)
        return

    with during(Stage.THE_TRAINING_ENVIRONMENT_WOULD_NOT_START):
        validate_runtime_launch_contract()
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
