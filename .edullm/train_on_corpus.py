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

THE FOURTH THING, WHICH DOES NOT CORRUPT ANYTHING AND IS EXPENSIVE FOR THE OPPOSITE REASON.
This file trains in bfloat16 by default and a T4 has none: Turing is the one NVIDIA generation
with tensor cores and without that format. That failure is loud, and it is loud several minutes
and one billed instance too late, because every stage before the first bfloat16 kernel
succeeds. The obvious guard does not work either -- ``torch.cuda.is_bf16_supported()`` returns
true on a T4. So ``main`` reads the device's compute capability against the config it has just
built and refuses in the first seconds. The check is
``olmo_core.train.train_module.validate_precision_support``, which lives in the library rather
than in this file so that every entry point in this repository inherits it when a train module
is built -- this file is not the only one, and a branch that has never touched
``src/olmo_core/`` still gets it on the rebase it has to do anyway. See ``--param-dtype`` for
asking for something the card does have.

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

WHICH ``olmo_core`` THIS TRAINS AGAINST, WHICH IS NOT THE ONE THE BRANCH IS ON UNLESS THE
BLOCK BELOW RUNS. The image sets ``ENV PYTHONPATH=/opt/olmo-core/src`` and also pip-installs
the project, so a container carries two copies of the library and both are the commit the
image was built from. The node clones this branch and mounts the whole tree at ``/work`` --
``src/`` included, not just ``.edullm/`` -- but nothing puts ``/work/src`` on ``sys.path``, so
``import olmo_core`` reaches the image's copy and the branch's library is present on disk and
never executed. Nothing warns. The run trains, the loss goes down, and it goes down against a
model this branch did not define.

Today those two happen to compose, because the image was built from an ancestor of this
branch. That is luck with an expiry date on it, and the failure when it expires is silent.
"""

import os
import sys

#: The ``src/`` of the tree this file was cloned into, which is the library this branch means.
#: Resolved from ``__file__`` rather than written as ``/work/src`` because the mount point is
#: the caller's choice: the block nodes use ``/work``, a laptop uses a checkout, and a git
#: worktree uses neither. All three want the ``src/`` that is a sibling of this file's parent.
_BRANCH_LIBRARY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")

# AHEAD OF THE IMAGE RATHER THAN INSTEAD OF IT, WHICH IS THE WHOLE ARRANGEMENT. The image goes
# on supplying torch, transformers, edullm_data and the rest; what it stops supplying is
# `olmo_core`, because a regular package resolves entirely from the first `sys.path` entry that
# holds it and never merges with a later one.
#
# HERE AND NOT AS `PYTHONPATH=/work/src` IN FRONT OF THE COMMAND IN `.edullm/run.yaml`, WHICH
# IS THE SPELLING THAT LOOKS RIGHT AND CANNOT WORK. `block_multinode.torchrun_command` puts the
# rendezvous form in front of that command, so the first word of it becomes torchrun's
# positional `training_script`; torchrun then execs `python -u <that word> <the rest>`. An
# environment assignment in that position is a filename Python cannot open, on all sixty-four
# ranks at once. There is nowhere in a command torchrun prepends to that an env prefix can go,
# so the process has to put its own library on its own path -- which it can, because it knows
# where it was cloned to and a command string does not.
#
# `is_dir` rather than unconditional, so that a repository laid out some other way falls
# through to whatever the image installed instead of prepending a path that holds nothing.
if os.path.isdir(_BRANCH_LIBRARY):
    sys.path.insert(0, _BRANCH_LIBRARY)

# THE CACHING ALLOCATOR IS CONFIGURED HERE BECAUSE THERE IS NOWHERE ELSE TO PUT IT.
#
# `expandable_segments:True` lets the allocator grow a segment rather than reserving a new
# block of the size class it needs, which is what keeps a long run from fragmenting itself
# into an out-of-memory on a step that allocated no more than the thousand before it. This
# model fragments for a specific reason: the top-4 expansion in the MoE path allocates six
# tensors per layer whose size depends on how the router happened to distribute that batch,
# so the size classes move every step.
#
# It reaches the allocator only if it is set before the first CUDA allocation, and it is read
# from the environment rather than from an API, so a call would be no better placed than this.
# The natural home is the launch script -- but that lives in the platform repository, arrives
# by a different route, and would have to be merged and the fleet relaunched for a change to
# take effect. This file rides the clone. Set the variable outside and that wins: `setdefault`
# is deliberate, so anyone tuning the allocator from the launch environment is not overridden
# by a default written weeks earlier.
#
# `PYTORCH_ALLOC_CONF` AND NOT `PYTORCH_CUDA_ALLOC_CONF`. torch renamed it. The old spelling is
# still honoured by the 2.9.0 this image pins, but it logs `PYTORCH_CUDA_ALLOC_CONF is
# deprecated` when it reads it -- once per process, which on this run is sixty-four warnings in
# the first seconds of the log, at the moment somebody is scanning that log for the reason a
# rank did not come up. Both spellings were tried against the real image; the new one is silent.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import argparse
import contextlib
import copy
import enum
import functools
import json
import logging
import re
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
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.exceptions import OLMoConfigurationError
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
    HFConverterCallback,
    LMEvaluatorCallbackConfig,
    WandBCallback,
)
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
    TransformerTrainModuleConfig,
    validate_precision_support,
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
    # Appended rather than slotted in beside 70, where it belongs in time. The numbers are
    # already in circulation -- they are in the platform's guide and in whatever notes people
    # kept from diagnosing a dead container -- so renumbering the three stages after it would
    # silently change the meaning of codes somebody has written down.
    THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION = 73


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
#
# TOKENIZER/GIGATOKEN-{BPE,SUPERBPE} ARE EXPLICIT CONFIGS, NOT from_hf. They are Plan A scale
# tokenizers published under s3://edullm-data/tokenizer/gigatoken-*/v1/ with no HuggingFace
# source -- vocab 100000 merge ids 0..99999, no added special tokens. Packed Plan B shards
# concatenate encode().ids with no inserted EOS.
#
# identifier IS None ON PURPOSE. TokenizerConfig.identifier is consumed as a local path or
# HuggingFace id by evaluator_callback (HFTokenizer), generate/chat (AutoTokenizer), and
# convert_checkpoint -- none accept an s3:// URI. Training on pre-tokenized shards never
# builds a concrete tokenizer from this field, so a fake s3 path would only fail later at
# in-loop eval / HF export. Fail immediately if something asks for the files; vendor
# tokenizer.json into the image and point identifier at that local dir before BPB evals.
#
# eos/pad sit PAST the merge range (100000 / 100001) with vocab_size=100002. Using 99999
# would fingerprint a real merge as a special; nothing in this config emits those ids today
# (generate_doc_lengths defaults False; labels come from label_mask), but the ids land in
# the dataset fingerprint and the saved checkpoint config.
#
# STYLE: callables rather than bound classmethods, matching smollm2. Lookup is separated from
# build so a KeyError inside a factory exits 70 (config would not build) rather than 69
# (unknown tokenizer) -- see the try block in corpus_from_manifest.
def _gigatoken_bpe() -> TokenizerConfig:
    return TokenizerConfig(
        vocab_size=100002,
        eos_token_id=100000,
        pad_token_id=100001,
        identifier=None,
    )


def _gigatoken_superbpe() -> TokenizerConfig:
    return TokenizerConfig(
        vocab_size=100002,
        eos_token_id=100000,
        pad_token_id=100001,
        identifier=None,
    )


TOKENIZERS = {
    "tokenizer/dolma2-bpe": TokenizerConfig.dolma2,
    "tokenizer/gigatoken-bpe": _gigatoken_bpe,
    "tokenizer/gigatoken-superbpe": _gigatoken_superbpe,
}


# This is deliberately a platform entrypoint recipe rather than a legacy
# ``src/scripts/train`` recipe. It receives its corpus and checkpoint location
# from eduLLM, so a submitted run cannot quietly read the upstream AI2 mix.
OLMOE_7B_32X4_FACTORY = "olmoe_7b_32x4"
OLMOE_7B_32X4_ROUTED_EXPERTS = 32
OLMOE_7B_32X4_ROUTED_HIDDEN_SIZE = 2048


def olmoe_7b_32x4(vocab_size: int, shared_experts: int = 0) -> TransformerConfig:
    """Build the ~7B total / 32x4 routed MoE that every arm is measured against.

    ``shared_experts`` is zero here and that is the whole point: this is the
    control, and a module cannot be screened as an arm while it is also in the
    thing the arm is compared to. The shared-expert arm passes
    ``--moe-shared-experts 2`` and nothing else changes.

    A shared expert in OLMo-core is one unconditional MLP, so its intermediate
    width is the number of shared experts times one routed expert's width --
    two shared experts is one 4096-wide MLP, evaluated for every token in
    addition to the top-four routed experts.
    """
    return TransformerConfig.llama_like_moe(
        vocab_size=vocab_size,
        d_model=2048,
        n_layers=16,
        n_heads=16,
        num_experts=OLMOE_7B_32X4_ROUTED_EXPERTS,
        top_k=4,
        expert_hidden_size=OLMOE_7B_32X4_ROUTED_HIDDEN_SIZE,
        shared_expert_hidden_size=(
            shared_experts * OLMOE_7B_32X4_ROUTED_HIDDEN_SIZE if shared_experts else None
        ),
        dropless=True,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
    )


def is_olmoe_7b_32x4(opts) -> bool:
    """Whether ``opts`` selects the fixed MoE recipe this entrypoint owns."""
    return opts.model_factory == OLMOE_7B_32X4_FACTORY


def validate_olmoe_parallelism(opts) -> None:
    """Reject an impossible expert mesh before GPU work starts.

    THE INTENDED 64-RANK LAYOUT IS EIGHT HSDP REPLICAS OF EIGHT, NOT TWO OF
    THIRTY-TWO. This docstring said the latter until 2026-08-08 and was
    describing a layout that was considered and rejected. At shard degree 8 the
    expert-parallel group is exactly one machine's cards, so the MoE all-to-all
    stays on NVLink; at 32 the group spans four machines and every all-to-all
    crosses the fabric. Nothing about that reads as wrong from inside the run --
    it starts, the loss falls, and each step takes several times what it should
    -- which is why the number is worth stating correctly in the one place
    somebody reads to find out what it should be. The platform's dispatch
    computes the pair and refuses a command that names it differently.

    The checks below are generic and were always correct; only the prose was
    wrong. ``WORLD_SIZE`` is set by torchrun, but intentionally absent from unit
    tests and config inspection, so only validate the product when it is known.
    """
    if opts.moe_shard_degree <= 0:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "--moe-shard-degree must be positive",
        )
    if opts.moe_num_replicas <= 0:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "--moe-num-replicas must be positive",
        )
    if opts.moe_shared_experts < 0:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "--moe-shared-experts cannot be negative",
        )
    if opts.moe_shared_experts and opts.hf_export:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "--hf-export cannot be combined with --moe-shared-experts: HuggingFace has no "
            "MoE architecture with an always-on expert, so `FlexOlmoConfig` has no field for "
            "one and the state converter refuses the run's shared_mlp weights rather than "
            "dropping them. Refused here because the alternative is finding out at the first "
            "export, hours in. The shared-expert arm is comparable on validation loss like "
            "every other arm; what it cannot do is hand its weights to the downstream lane.",
        )
    if OLMOE_7B_32X4_ROUTED_EXPERTS % opts.moe_shard_degree:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"{OLMOE_7B_32X4_ROUTED_EXPERTS} routed experts do not divide "
            f"--moe-shard-degree={opts.moe_shard_degree}",
        )

    world_size = os.environ.get("WORLD_SIZE")
    if world_size is None:
        return
    try:
        world_size_int = int(world_size)
    except ValueError:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"WORLD_SIZE must be an integer, got {world_size!r}",
        ) from None
    expected_world_size = opts.moe_num_replicas * opts.moe_shard_degree
    if world_size_int != expected_world_size:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "the 32x4 MoE recipe expects WORLD_SIZE="
            f"{expected_world_size} from --moe-num-replicas={opts.moe_num_replicas} "
            f"x --moe-shard-degree={opts.moe_shard_degree}, got {world_size_int}",
        )


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
    #: The held-out shards the evaluator was wired to, empty when the corpus declares none.
    #: Carried on the config so ``summarise`` can say how many there were without resolving
    #: the corpus a second time: a null validation loss beside zero shards means the corpus
    #: could not be measured, and beside a count it means the evaluator returned nothing.
    dataset_val_paths: List[str] = field(default_factory=list)
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
    #: The corpus's held-out shards, empty when it declares none. SEPARATE FROM ``paths``
    #: RATHER THAN MIXED INTO IT: a validation shard that reached the training stream is the
    #: one data error no metric can show, because the number it corrupts is the number you
    #: would check it with.
    val_paths: List[str] = field(default_factory=list)


def corpus_from_manifest(
    read,
    *,
    dataset_id: str,
    version: str,
    tokenizer_id: str,
    val_paths: Optional[List[str]] = None,
) -> Corpus:
    """Turn what the reader returned into what OLMo-core needs, or refuse and say why.

    Separate from the fetch because this is the part with the judgement in it, and a test
    should be able to hand it a manifest describing a big-endian corpus without standing up
    S3 or installing the reader. ``read`` is duck-typed for that reason: anything carrying
    ``paths``, ``dtype``, ``byte_order``, ``header_bytes`` and ``rows`` will do.

    THE HELD-OUT SHARDS INHERIT THE THREE CHECKS BELOW RATHER THAN GETTING THEIR OWN. dtype,
    header bytes and byte order are properties of the published corpus and not of one split,
    and the manifest being checked describes both. What has to hold is that the eval dataset
    is built at the same width: a mismatch there reads every held-out token to a different
    in-range id and yields a validation loss that is wrong rather than absent.
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

    # Lookup is outside the build call on purpose. A KeyError from TOKENIZERS[...] means the
    # id is unknown (exit 69). A KeyError raised *inside* a factory (e.g. from_hf missing
    # vocab_size) must not be rewritten as "unknown tokenizer" -- that lists the id it claims
    # not to know and loses the real cause. Let factory failures propagate to exit 70.
    try:
        build_tokenizer = TOKENIZERS[tokenizer_id]
    except KeyError:
        known = ", ".join(sorted(TOKENIZERS)) or "none"
        raise Refusal(
            Stage.THIS_IMAGE_HAS_NO_CONFIG_FOR_THAT_TOKENIZER,
            f"no OLMo-core config for {tokenizer_id}; this image knows: {known}",
        ) from None
    tokenizer = build_tokenizer()

    return Corpus(
        dataset_id=dataset_id,
        version=version,
        paths=list(read.paths),
        dtype=NumpyDatasetDType(read.dtype),
        tokenizer=tokenizer,
        rows=read.rows,
        val_paths=list(val_paths or []),
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

    # WHICH BUCKET THE CORPUS IS READ OUT OF, WHICH UNTIL NOW WAS NOT A QUESTION THIS FILE
    # ASKED. `edullm_data.read.DATA_BUCKET` is the module constant `edullm-data`, the reader
    # takes `data_bucket` as a keyword and honours no environment variable of its own, and
    # every call below omitted it -- so the bucket was `edullm-data` in us-east-1 whatever the
    # platform said.
    #
    # That was correct for as long as every consumer was a Batch job in us-east-1. The capacity
    # block is not: it is eight machines in us-east-2 reading a mirror at
    # `edullm-data-us-east-2`, and `infra/iam/block-fleet-roles.yaml` grants the node role
    # s3:GetObject on that mirror and on nothing else. So the default sends a 64-rank run at
    # a bucket its own instance profile may not read, and the run dies at exit 65 in the first
    # minute of a ninety-six hour reservation that is already billing.
    #
    # The platform has been exporting the answer into the container all along --
    # `infra/block-node-bootstrap.sh` sets EDULLM_DATA_BUCKET from the launch input -- and
    # nothing read it. Reading it here is what makes the mirror reachable at all.
    #
    # UNSET FALLS THROUGH TO THE READER'S OWN DEFAULT rather than to a literal repeated here.
    # A second copy of `edullm-data` in this file is the copy that is wrong on the day the
    # reader's moves, and every existing caller -- Batch, a laptop, a test -- sets nothing and
    # must keep getting exactly what it got before.
    data_bucket = os.environ.get("EDULLM_DATA_BUCKET") or None
    bucket_kwargs = {"data_bucket": data_bucket} if data_bucket else {}
    # Logged because `show` prints the shard list as a count, so the bucket is otherwise
    # nowhere a person watching the first minute of a run can see it.
    log.info(
        "reading %s/%s from bucket %s",
        dataset_id,
        version,
        data_bucket or "the reader's own default",
    )

    # "latest" resolves through the catalog rather than being an alias anybody can move. A
    # pinned version is the normal case and what the platform sends; this branch exists so a
    # person poking at the image by hand does not have to look one up first.
    if version in ("", "latest"):
        try:
            resolved = resolve_latest(dataset_id, s3=s3, **bucket_kwargs)
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
        read = dataset_paths(dataset_id, version, s3=s3, **bucket_kwargs)
    except Refusal:
        raise
    except BaseException as exc:
        raise Refusal(
            read_failure(exc),
            f"reading {dataset_id}/{version}: {type(exc).__name__}: {exc}",
        ) from exc

    # THE HELD-OUT SPLIT, ASKED FOR SEPARATELY AND ALLOWED TO BE ABSENT. A second call rather
    # than a flag on the first, because the default above is the reader's answer to "what may
    # this run train on" and the answer must not start including validation shards.
    #
    # ABSENT IS A VALUE HERE, WHICH IS WHY THIS DEGRADES RATHER THAN REFUSES. A corpus with no
    # declared split comes back empty rather than raising, so "there is nothing to evaluate
    # on" and "the read failed" stay distinguishable, and neither kills a run over a metric.
    # ``--require-val`` is how a comparison whose estimand is held-out loss says it would
    # rather not start than train without one.
    val_paths: List[str] = []
    try:
        val_paths = list(
            dataset_paths(dataset_id, version, split="val", s3=s3, **bucket_kwargs).paths
        )
    except BaseException as exc:  # noqa: BLE001 -- see above
        log.warning(
            "could not list the held-out split of %s/%s, so this run has no evaluator: %s: %s",
            dataset_id,
            version,
            type(exc).__name__,
            exc,
        )

    return corpus_from_manifest(
        read,
        dataset_id=dataset_id,
        version=version,
        tokenizer_id=tokenizer_id,
        val_paths=val_paths,
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


def hf_export_folder(opts) -> str:
    """Where the HuggingFace exports go, which nobody downstream should have to be told.

    Beside the checkpoints rather than inside them, at the run's own output prefix. The
    platform derives ``EDULLM_CHECKPOINT_DIR`` as ``${EDULLM_OUTPUT_PREFIX}checkpoints/``, so
    the parent of the save folder is that prefix and ``hf/`` is a sibling of ``checkpoints/``
    -- one directory to list, holding one ``step{N}`` per export and nothing else. A lane
    reading the run's outputs finds them by listing rather than by being handed a path.

    DERIVED FROM ``--save-folder`` AND NOT FROM ``EDULLM_OUTPUT_PREFIX``, although on the
    platform the two agree. The save folder is the one the checkpoints are actually going to,
    including when a submission overrode it or the array-job prologue rewrote the prefix
    underneath it, and an export that landed beside a different run's checkpoints would be
    worse than one nobody could find.
    """
    if opts.hf_export_folder:
        return opts.hf_export_folder
    return f"{os.path.dirname(normalize_path(opts.save_folder).rstrip('/'))}/hf"


def build_config(opts, overrides: List[str]):
    corpus = resolve_corpus(
        dataset_id=opts.dataset_id,
        version=opts.dataset_version,
        tokenizer_id=opts.dataset_tokenizer,
    )
    log.info(
        "%s/%s: %d shards, dtype %s, tokenizer %s, %d held-out shards",
        corpus.dataset_id,
        corpus.version,
        len(corpus.paths),
        corpus.dtype,
        opts.dataset_tokenizer,
        len(corpus.val_paths),
    )
    # Refused here rather than after the model is built, because the answer is already known
    # and a run that cannot measure the thing it exists to measure should not reach a GPU.
    if opts.require_val and not corpus.val_paths:
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"--require-val was given and {corpus.dataset_id}/{corpus.version} declares no "
            "validation split, so this run could report only training loss.",
        )

    if is_olmoe_7b_32x4(opts):
        validate_olmoe_parallelism(opts)
        factory = functools.partial(olmoe_7b_32x4, shared_experts=opts.moe_shared_experts)
    else:
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

    if is_olmoe_7b_32x4(opts):
        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=opts.rank_microbatch_size,
            max_sequence_length=opts.sequence_length,
            optim=AdamWConfig(
                lr=opts.learning_rate,
                weight_decay=0.1,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
            ),
            compile_model=True,
            # HSDP is required for expert parallelism. In the production layout, two
            # replicas each shard the model and the 32 routed experts over 32 ranks.
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.hsdp,
                param_dtype=DType(opts.param_dtype),
                reduce_dtype=DType.float32,
                num_replicas=opts.moe_num_replicas,
                shard_degree=opts.moe_shard_degree,
            ),
            ep_config=TransformerExpertParallelConfig(degree=opts.moe_shard_degree),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
            scheduler=CosWithWarmup(warmup=opts.warmup_steps),
        )
    else:
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
            # param_dtype comes from --param-dtype, whose default is bfloat16 and therefore is
            # exactly what this line said before the flag existed. The flag is here so the choice
            # can be made rather than only rejected, and so that it appears in the command text --
            # the platform reads command words and cannot see a dtype set in code, so `--param-dtype
            # bfloat16` on a T4 shape is refused at submission and never reaches an instance.
            #
            # reduce_dtype stays float32 and has no flag. It is the gradient reduction, fp32 is the
            # numerically safe answer at every scale this platform runs, and the dotted override
            # `train_module.dp_config.reduce_dtype=...` reaches it for anyone who disagrees.
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType(opts.param_dtype),
                reduce_dtype=DType.float32,
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

    # STILL NO downstream_evaluator, AND THAT ABSENCE IS STILL A DECISION. The example's pulls
    # HellaSwag from Hugging Face and the task groups in `olmo_core.eval.task_groups` name
    # paths this container cannot reach; either would put a public-internet fetch in the
    # middle of a run whose whole claim is that it read a sealed corpus, a failure in one
    # would look like a training failure, and the image installs `.[wandb]` rather than
    # `.[eval]` so `ai2-olmo-eval` is not there to import in the first place.
    #
    # THE LM EVALUATOR IS NOW WIRED, TO THE CORPUS'S OWN HELD-OUT SHARDS. Same argument, other
    # direction: `dataset_paths(split="val")` reads the same sealed prefix the training stream
    # came from, so this adds no network dependency, no package and no failure mode the run
    # did not already have. What it needs is a corpus that declares a split, and one that does
    # not gets no evaluator rather than an error.
    #
    # WITHOUT IT THE RUN REPORTS TRAINING LOSS UNDER A NAME THAT READS LIKE A RESULT.
    # `first_loss` and `last_loss` say how well the model fit the stream it just saw, and the
    # arms of this experiment are compared on held-out cross-entropy. That is not the same
    # quantity and it is not a proxy for it.
    #
    # A PADDED FSL DATASET RATHER THAN THE TRAINING CONFIG'S OWN CLASS, because
    # LMEvaluatorCallbackConfig.build refuses anything else by name. Padding is what makes the
    # last instance of each shard scored rather than silently dropped at a different point for
    # each arm.
    if corpus.val_paths:
        trainer_config = trainer_config.with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig(
                    paths=corpus.val_paths,
                    sequence_length=opts.sequence_length,
                    tokenizer=corpus.tokenizer,
                    # Same width as the training stream, passed rather than inferred, for the
                    # reason the header gives about the training stream.
                    dtype=corpus.dtype,
                    work_dir=opts.work_dir,
                ),
                eval_interval=opts.eval_interval,
                # On finish, because a paired comparison reads the final number and the run
                # must produce one rather than stopping at the last multiple of the interval.
                eval_on_finish=True,
                # Not on startup: it costs a pass over the held-out set to measure a model
                # that has not trained, and the arms are identical there by construction.
                eval_on_startup=False,
                # BOUNDED, NOT AN EPOCH. A held-out split of a corpus this size is hundreds of
                # millions of tokens and an unbounded pass over it costs more than the stretch
                # of training it measures. A fixed count also scores every arm on the same
                # amount of data, which an epoch over differently sized splits would not.
                eval_duration=Duration.steps(opts.eval_batches),
                deterministic=True,
            ),
        )

    # HUGGINGFACE EXPORT, OFF UNLESS ASKED FOR. It is off by default because it is not free
    # and because every existing user of this entry point submitted without it: gathering the
    # full model state dict is a collective, and every rank that is not zero then waits at a
    # barrier while rank zero builds the model on CPU and writes it. On the 7B MoE that is the
    # whole fleet stopped for the length of one write, which is why the cadence is a flag and
    # not the checkpoint interval.
    if opts.hf_export:
        trainer_config = trainer_config.with_callback(
            "hf_converter",
            HFConverterCallback(
                output_folder=hf_export_folder(opts),
                convert_interval=opts.hf_export_interval or None,
                # No validation. It runs the OLMo-core model forward to compare logits, and
                # for a dropless MoE that forward goes through `olmo_core.ops.moe`, whose
                # kernels are Triton and CUDA-only -- so on this model the check is not
                # cheaper than the export, it is a second full forward pass on the training
                # device in the middle of a run. `src/test/nn/hf/checkpoint_test.py` is where
                # the mapping is held to the numbers instead.
                validate=False,
                tokenizer_id=opts.hf_tokenizer or None,
                max_sequence_length=opts.sequence_length,
                # A FAILED EXPORT MUST NOT END THE RUN. Everything it needs beyond the model
                # is outside this process -- a tokenizer fetched from Hugging Face, a write to
                # the outputs bucket -- and none of it is worth the ninety-six hours the run
                # is standing on. The library default re-raises, which is right for a job
                # whose only output is the converted model and wrong for this one.
                raise_on_failure=False,
            ),
        )

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
        dataset_id=corpus.dataset_id,
        dataset_version=corpus.version,
        dataset_val_paths=list(corpus.val_paths),
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
        #: The last held-out CE loss the run recorded, or None when it had no evaluator.
        #: THIS IS THE NUMBER A COMPARISON BETWEEN ARMS READS, and it is captured here for the
        #: same reason the W&B url is: the summary runs after the trainer has stopped and
        #: nothing it can reach by then still holds the evaluator's metrics. Without it the
        #: value exists in W&B and in a log stream nobody may read, and the machine-readable
        #: summary the platform parses carries training loss alone.
        self.val: Optional[float] = None
        self.wandb_url = ""

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        del step
        if not self.wandb_url:
            with contextlib.suppress(Exception):
                import wandb

                self.wandb_url = getattr(wandb.run, "url", "") or ""
        # Matched by suffix rather than by an exact key. EvaluatorCallback records under
        # "{prefix}/{evaluator name}/{metric}", so both the prefix and the evaluator's name
        # sit in the middle of the string and a hardcoded key stops matching -- silently, by
        # matching nothing -- the moment either is renamed.
        for name, value in metrics.items():
            if name.endswith("/CE loss") and not name.startswith("train/"):
                self.val = float(value)
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
                # Null beside zero shards says the corpus declared no held-out split, which
                # is a different fact from a run that had one and scored badly. --require-val
                # refuses the first case before an instance is allocated.
                "val_loss": losses.val,
                "val_shards": len(config.dataset_val_paths),
                "seconds": seconds,
                "peak_memory_gib": peak,
                "checkpoint_uri": opts.save_folder,
                # Empty when the run was not asked to export, so the lane can tell a run that
                # produced no HuggingFace checkpoints from one whose exports it cannot find.
                "hf_uri": hf_export_folder(opts) if opts.hf_export else "",
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


def grouped_mm_gmm(a, b, batch_sizes, trans_b: bool = False):
    """A drop-in for ``grouped_gemm.ops.gmm``, backed by a kernel torch already ships.

    Same signature and same result, so it can be substituted for the module-level ``gmm``
    that :class:`DroplessMoEMLP` captures in its constructor.

    ``batch_sizes`` is the token count per *local* expert and ``torch._grouped_mm`` wants the
    inclusive cumulative sum of those counts, as int32, on the same device as the activations.
    Computing it with ``torch.cumsum`` rather than in Python is most of why this is faster:
    the loop it replaces calls ``batch_sizes.cpu()``, and a device-to-host copy is a
    synchronisation the whole stream waits on.

    ``trans_b`` transposes the weight per expert. The transposed view is passed through
    without a copy -- verified accepted by the kernel in the image rather than assumed, since
    the view is not contiguous and a kernel that rejected it would have cost a 33 MB copy per
    call on every one of the 192 calls a step makes.
    """
    offsets = torch.cumsum(batch_sizes, dim=0).to(device=a.device, dtype=torch.int32)
    return torch._grouped_mm(a, b.transpose(-2, -1) if trans_b else b, offs=offsets)


def install_grouped_mm(*, enabled: bool = True) -> str:
    """Route the dropless MoE through ``torch._grouped_mm``. Returns what it decided, for the log.

    WHY THIS IS A MONKEYPATCH AND NOT AN EDIT TO ``mlp.py``. On the block, only ``.edullm/``
    is read from the branch clone. ``import olmo_core`` resolves to the copy baked into the
    image, so a change under ``src/olmo_core/`` would sit in the repository looking applied
    and never run. Patching from here is not a shortcut around review; it is the only place
    the change can be made without rebuilding the image.

    WHAT IT IS WORTH. ``grouped_gemm`` is absent from the training image -- observed by
    running the container, not inferred -- so ``DroplessMoEMLP`` takes its fallback: a Python
    loop over local experts, one GEMM each, with a device-to-host synchronisation per call, in
    a region ``torch.compile`` is explicitly disabled for. At expert-parallel degree 8 each
    rank holds four local experts rather than 32, which is why the fallback costs an estimated
    1.1x-1.35x rather than the order of magnitude it would at degree 1. On a 50B-token run the
    central estimate is worth about an hour and a half.

    WHAT IT MUST NOT DO. Take effect when the fast package is present. ``gmm`` is ``None``
    exactly when ``grouped_gemm`` failed to import, so a future image that carries the package
    keeps using it and this function reports ``grouped_gemm`` and returns.

    THE ESCAPE. ``--no-moe-grouped-mm`` puts the run back on the loop the library shipped,
    which is slow and is known to work. If anything about the MoE looks wrong in the first
    hundred steps, that flag is the first thing to try, because it is the only part of the
    numerical path this file changes.
    """
    try:
        from olmo_core.nn.moe import mlp as moe_mlp
    except ImportError:
        return "no MoE module, nothing to patch"

    if getattr(moe_mlp, "gmm", None) is not None:
        return "grouped_gemm is present, left alone"
    if not enabled:
        return "disabled by --no-moe-grouped-mm, using the library's Python loop"
    if not hasattr(torch, "_grouped_mm"):
        return f"torch {torch.__version__} has no _grouped_mm, using the library's Python loop"

    moe_mlp.gmm = grouped_mm_gmm
    return "grouped_gemm absent, routed through torch._grouped_mm"


def train(config, opts=None) -> None:
    if get_rank() == 0:
        show(config)

    seed_all(config.init_seed)

    # Before the build and not after it. `DroplessMoEMLP.__init__` reads the module-level
    # `gmm` into `self._gmm`, so a patch applied after the model exists changes nothing and
    # would leave every rank on the slow path while the log said otherwise.
    verdict = install_grouped_mm(enabled=getattr(opts, "moe_grouped_mm", True))
    log.info("MoE kernel: %s", verdict)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config.as_config_dict()
    # The same config, handed to the converter rather than left for it to read back. Its own
    # default is to fetch `config.json` out of the checkpoint directory it is converting, and
    # this run saves asynchronously -- so at the step being converted that object may not have
    # been written yet, and an export would fail on a file the run is in the middle of
    # producing. The process already holds the answer.
    if "hf_converter" in trainer.callbacks:
        cast(
            HFConverterCallback, trainer.callbacks["hf_converter"]
        ).experiment_config = config.as_config_dict()
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
    parser.add_argument(
        "--model-factory",
        default="olmo2_190M",
        help=(
            "A TransformerConfig factory, or "
            f"{OLMOE_7B_32X4_FACTORY!r} for the platform-native ~7.5B 32x4 MoE recipe"
        ),
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="THE DEFAULT PREDATES THE MoE RECIPE AND IS NOT THE VALUE THAT RECIPE WAS "
        "PLANNED AT. `src/scripts/train/OLMoE-1B-7B.py`, which commit d4b97fe aligned with "
        "this entry point so the two are one experiment rather than two, sets 4096 and "
        "derives its global batch size from it. Left at 2048 here because every submission "
        "that predates the MoE factory got this value and changing a default silently makes "
        "a sweep spanning the change two experiments reported as one -- so the run that "
        "wants 4096 passes it. What moves with it: attention FLOPs per token and the KV "
        "cache both scale with this, --global-batch-size and --rank-microbatch-size are in "
        "TOKENS and so hold the token budget fixed while halving the sequences per batch, "
        "and an exported HuggingFace checkpoint takes this as its max_position_embeddings.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--global-batch-size", type=int, default=256 * 1024)
    parser.add_argument("--rank-microbatch-size", type=int, default=16 * 1024)
    parser.add_argument(
        "--moe-shard-degree",
        type=int,
        default=32,
        help=(
            "HSDP and expert-parallel shard degree for "
            f"{OLMOE_7B_32X4_FACTORY}; ignored by other model factories"
        ),
    )
    parser.add_argument(
        "--moe-num-replicas",
        type=int,
        default=2,
        help=(
            "HSDP replica count for " f"{OLMOE_7B_32X4_FACTORY}; ignored by other model factories"
        ),
    )
    parser.add_argument(
        "--moe-shared-experts",
        type=int,
        default=0,
        help=(
            "Always-on shared experts for "
            f"{OLMOE_7B_32X4_FACTORY}, each one routed-expert wide. Zero is the "
            "base every arm is compared against; the shared-expert arm passes 2. "
            "Ignored by other model factories"
        ),
    )
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Steps between held-out evaluations. A corpus that declares no validation "
        "split gets no evaluator whatever this says.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=64,
        help="Batches per held-out evaluation. Bounded rather than a full epoch, because a "
        "pass over a whole validation split costs more than the stretch of training it "
        "measures, and fixed so every arm and seed is scored on the same amount of data.",
    )
    parser.add_argument(
        "--require-val",
        action="store_true",
        help="Refuse rather than train when the corpus declares no validation split. A "
        "comparison whose estimand is held-out loss gets nothing from a run that cannot "
        "measure it, and the failure is otherwise invisible: the run trains, exits zero, "
        "and reports training loss under a name that reads like a result.",
    )
    parser.add_argument(
        "--hf-export",
        action="store_true",
        help="Also write HuggingFace-format checkpoints, so that whatever reads them "
        "downstream does not need a conversion job per handoff. OFF BY DEFAULT: an export "
        "is a collective followed by every rank waiting at a barrier while rank zero "
        "writes the model, which is a cost no existing user of this entry point asked for. "
        "Requires the transformers package, which is not in the wandb extra.",
    )
    parser.add_argument(
        "--hf-export-interval",
        type=int,
        default=0,
        help="Steps between HuggingFace exports while the run is going. Zero, the default, "
        "exports once when training finishes. A number here is what makes intermediate "
        "checkpoints available to something running beside the training job, and it should "
        "be a multiple of --save-interval so that every export has a core checkpoint "
        "beside it.",
    )
    parser.add_argument(
        "--hf-export-folder",
        default="",
        help="Where the exports go. Empty, the default, puts them at hf/ beside the "
        "checkpoints under the run's own output prefix, which is where something reading "
        "the run's outputs will look without being told.",
    )
    parser.add_argument(
        "--hf-tokenizer",
        default="",
        help="A HuggingFace id or local directory to save alongside the exported model. "
        "Empty uses the corpus tokenizer's own identifier, which for dolma2 is a "
        "HuggingFace id and therefore a fetch over the public internet at export time; "
        "point this at a vendored directory to avoid that. The gigatoken tokenizers carry "
        "no identifier, so their exports are weights and config with no tokenizer at all.",
    )
    parser.add_argument(
        "--param-dtype",
        default=DType.bfloat16.value,
        choices=[DType.bfloat16.value, DType.float16.value, DType.float32.value],
        help="The parameter dtype FSDP holds and computes in. THE DEFAULT IS THE DTYPE THIS "
        "FILE ALWAYS USED and changing it is not a free choice: it changes the numerics of "
        "the run, so a sweep half of which predates this flag is not a comparison. float32 "
        "is the answer on a T4, whose Turing silicon has no bfloat16 at all -- see the "
        "refusal at startup. float16 is faster there and OLMo-core ships no gradient scaler, "
        "so it is fp16 without loss scaling.",
    )
    parser.add_argument(
        "--no-moe-grouped-mm",
        dest="moe_grouped_mm",
        action="store_false",
        help="Put the dropless MoE back on the Python loop the library falls back to when the "
        "grouped_gemm package is missing, which it is in this image. ON BY DEFAULT, because "
        "the loop costs an estimated 1.1x-1.35x of the whole run and torch already ships the "
        "kernel that replaces it. THIS FLAG IS THE ESCAPE: the substitution is verified "
        "bit-exact in forward and backward, including experts that receive no tokens, but it "
        "is the only part of the numerical path this file touches, so if the MoE looks wrong "
        "in the first hundred steps this is the first thing to turn off. Has no effect on an "
        "image that carries grouped_gemm -- the package wins either way.",
    )
    parser.set_defaults(moe_grouped_mm=True)
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

    # THE CHECK ITSELF LIVES IN olmo_core, AND IS CALLED AGAIN FROM
    # TransformerTrainModuleConfig.build, WHICH IS WHERE IT CATCHES EVERY OTHER ENTRY POINT
    # IN THIS REPOSITORY. It is called here too because two things are only true here: this
    # is before the process group rather than after it, and this file has a stage number and
    # a W&B write to turn the refusal into something visible from outside a dead container.
    #
    # AFTER build_config BECAUSE IT NEEDS THE MERGED CONFIG, AND STILL BEFORE ANYTHING
    # EXPENSIVE. Everything above this line is local except a HEAD and a GET against the
    # manifest, so the whole of it is a few seconds; everything below it is the process
    # group, the model, the data loader and the run. This is the last point at which a
    # container costs nothing to stop, and the first at which what it is going to do is
    # settled -- reading argv earlier would be guessing at the same fact from its spelling.
    #
    # The whole config rather than config.train_module, because a `model.dtype=bfloat16`
    # override is reachable from the command line and is not a field of the train module.
    try:
        validate_precision_support(config)
    except OLMoConfigurationError as unusable:
        # A dry run trains nothing and would succeed on this card, so stopping it would be
        # this file refusing a run that works. Saying so is still the point of a dry run.
        if opts.dry_run:
            log.warning("%s", unusable)
        else:
            raise Refusal(
                Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION, str(unusable)
            ) from None

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
