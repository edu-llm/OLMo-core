"""What ``.edullm/train_on_corpus.py`` refuses, and what it lets through.

The entry point is not importable as a package -- it sits in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path here. The alternative
was to leave the file untested, and the things it checks are precisely the ones that produce a
working run on wrong data rather than an error.

``edullm_data`` is not installed in this repository's CI, only in the built image. That is why
the module imports the reader inside ``resolve_corpus`` and why everything tested below is
reachable without it.
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_on_corpus.py"
    spec = importlib.util.spec_from_file_location("edullm_train_on_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()


@dataclass
class FakeManifest:
    """The shape ``edullm_data.read.dataset_paths`` returns, with the fields that matter.

    Defaults describe a healthy corpus -- headerless, little-endian, uint32 -- so each test
    below changes exactly the one thing it is about.
    """

    paths: List[str] = field(default_factory=lambda: ["s3://edullm-data/x/v1/tokens/a.u32le.bin"])
    dtype: Optional[str] = "uint32"
    byte_order: Optional[str] = "little"
    header_bytes: int = 0
    rows: Optional[int] = 1000


def resolve(manifest: FakeManifest, tokenizer: str = "tokenizer/dolma2-bpe"):
    return entry.corpus_from_manifest(
        manifest, dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id=tokenizer
    )


class ReaderProtocolStub:
    """The four methods ``edullm_data.read`` calls on whatever it is handed.

    A boto3 client has none of them, which is the entire subject of the two tests below.
    """

    def get(self, bucket, key):
        ...

    def get_range(self, bucket, key, start, length):
        ...

    def head(self, bucket, key):
        ...

    def list(self, bucket, prefix):
        ...


@pytest.fixture
def reader(monkeypatch):
    """A stand-in for the installed reader, recording what ``resolve_corpus`` hands it.

    ``edullm_data`` is not installed in this repository's CI, so the modules are built here
    and put in ``sys.modules`` before the import inside ``resolve_corpus`` runs.
    """
    import types

    handed: Dict[str, Any] = {}
    adapter = ReaderProtocolStub()

    class Boto3S3:
        @classmethod
        def default(cls, region="us-east-1"):
            handed["region"] = region
            return adapter

    def dataset_paths(dataset_id, version, *, s3, **_):
        handed["s3"] = s3
        return FakeManifest()

    def resolve_latest(dataset_id, *, s3, **_):
        handed["resolve_latest_s3"] = s3
        return "v7"

    # Typed Any because these are modules being built rather than imported, and mypy is
    # right that a fresh ModuleType has no such attributes until this assigns them.
    read_module: Any = types.ModuleType("edullm_data.read")
    read_module.dataset_paths = dataset_paths
    read_module.resolve_latest = resolve_latest
    s3_module: Any = types.ModuleType("edullm_data.s3")
    s3_module.Boto3S3 = Boto3S3
    package = types.ModuleType("edullm_data")

    monkeypatch.setitem(sys.modules, "edullm_data", package)
    monkeypatch.setitem(sys.modules, "edullm_data.read", read_module)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", s3_module)
    return handed


def test_the_reader_is_handed_its_own_adapter_and_not_a_boto3_client(reader):
    """Mutation: pass ``boto3.client("s3")``, which is what this did and what it cost.

    The reader's ``s3`` parameter is typed against a four-method protocol and a boto3 client
    implements none of it, so ``_require_validated`` calls ``s3.head(bucket, key)`` and the
    run dies with an AttributeError before a byte leaves the account. Nothing catches it
    earlier: the parameter is named ``s3``, the annotation is a Protocol, and the traceback
    names a missing attribute rather than a wrong argument.

    On a GPU job that is a container which starts, exits 1 in under a second, and writes its
    only explanation to a log stream nobody on the platform side is allowed to read.
    """
    entry.resolve_corpus(
        dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )

    for method in ("get", "get_range", "head", "list"):
        assert callable(getattr(reader["s3"], method, None)), (
            f"the reader was handed something with no {method}(), which is what a boto3 "
            "client is"
        )


def test_resolving_the_latest_version_uses_the_same_adapter(reader):
    # The other call into the reader, and a second place a raw client could be passed.
    entry.resolve_corpus(
        dataset_id="pretrain/regmix-10b", version="latest", tokenizer_id="tokenizer/dolma2-bpe"
    )

    assert reader["resolve_latest_s3"] is reader["s3"]


def test_a_healthy_corpus_keeps_the_width_the_manifest_declared():
    # The whole reason this file exists. OLMo-core's own fallback would look at dolma2's
    # 100,278-token vocab, conclude uint16 fits it, and read a uint32 corpus two bytes at a
    # time -- producing in-range ids, no error, and a loss curve that is merely bad.
    corpus = resolve(FakeManifest())
    assert str(corpus.dtype) == "uint32"
    assert corpus.tokenizer.vocab_size == 100278


def test_a_corpus_with_a_header_is_refused_rather_than_read_from_offset_zero():
    with pytest.raises(SystemExit, match="header"):
        resolve(FakeManifest(header_bytes=128))


def test_a_big_endian_corpus_is_refused_on_a_little_endian_host():
    other = "big" if sys.byteorder == "little" else "little"
    with pytest.raises(SystemExit, match="endian"):
        resolve(FakeManifest(byte_order=other))


def test_a_corpus_that_declares_no_width_is_refused_rather_than_guessed_at():
    with pytest.raises(SystemExit, match="no dtype"):
        resolve(FakeManifest(dtype=None))


def test_no_trainable_shards_is_an_error_and_not_an_empty_run():
    # A corpus whose only splits are held out resolves to nothing. Training on zero shards is
    # not a shorter run, it is a run whose loss means nothing.
    with pytest.raises(SystemExit, match="no trainable shards"):
        resolve(FakeManifest(paths=[]))


def test_an_unknown_tokenizer_names_the_ones_this_image_has():
    # Rather than defaulting. A default here trains on ids that mean something other than what
    # they meant when the corpus was tokenized, and nothing downstream can tell.
    with pytest.raises(SystemExit, match="dolma2-bpe"):
        resolve(FakeManifest(), tokenizer="tokenizer/bytes-utf8")


def test_the_whole_config_builds_from_a_corpus_without_touching_s3(monkeypatch):
    """The check that would otherwise cost an A10G to run.

    Every mistake in the config below -- a TrainerConfig field that was renamed, a callback
    argument that does not exist, a model factory that is gone -- raises here in a second.
    Without this, the first thing that discovers it is a twelve-hour submission that reached a
    GPU, pulled a three-gigabyte image, and died before the first step.
    """
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            "--steps=25",
        ]
    )
    config = entry.build_config(opts, overrides)

    assert config.dataset.dtype == "uint32"
    assert config.dataset_id == "pretrain/regmix-10b"
    assert config.trainer.save_folder.endswith("/a-run-id/checkpoints/")
    # A retry must resume rather than overwrite what the first attempt left, which is the only
    # thing that makes a second attempt cheaper than a second run. remove_torn_checkpoints is
    # what lets this stay false while a retry still gets past a step it died writing.
    assert config.trainer.save_overwrite is False
    # Pruning is off because the workload role is denied the delete a prune starts with. At
    # OLMo-core's default of three, the fourth save fails a run that is most of a day old.
    assert config.trainer.callbacks["checkpointer"].max_checkpoints is None
    # Serializing is what the config saver does beside the checkpoint; a config that cannot be
    # written is one whose record of what ran does not exist.
    assert config.as_config_dict()["dataset_version"] == "v1"


def test_an_override_on_the_command_line_reaches_the_config(monkeypatch):
    # The escape hatch researchers actually use: everything after the flags is merged into the
    # config, so a person can change the learning rate without a new entry point.
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            "train_module.compile_model=false",
        ]
    )
    config = entry.build_config(opts, overrides)
    assert config.train_module.compile_model is False


# ---------------------------------------------------------------------------------------
# Whether a card can do the number format the run asks for
# ---------------------------------------------------------------------------------------


@pytest.fixture
def corpus(monkeypatch):
    """``build_config`` without S3, so the tests below are about the dtype and nothing else."""
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )


def configure(*extra: str):
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            *extra,
        ]
    )
    return opts, entry.build_config(opts, overrides)


@pytest.fixture
def submitted(monkeypatch):
    """``main`` reached the way the platform reaches it: four variables and a run id."""
    for name, value in (
        ("EDULLM_DATASET_ID", "pretrain/regmix-10b"),
        ("EDULLM_DATASET_VERSION", "v1"),
        ("EDULLM_DATASET_TOKENIZER", "tokenizer/dolma2-bpe"),
        ("EDULLM_CHECKPOINT_DIR", "s3://outputs/teams/platform/runs/a-run-id/checkpoints/"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)

    def argv(*extra: str):
        monkeypatch.setattr(sys, "argv", ["train_on_corpus", "a-run-id", *extra])

    return argv


def on_a(card: str, capability: Tuple[int, int], *, count: int, monkeypatch):
    """Answer as a host carrying ``count`` of this card would, without one being present.

    The three functions patched are the whole of what ``devices_without_bfloat16`` reads,
    which is the point: the decision rests on the compute capability the driver reports and
    on nothing that a container, a CUDA version or a torch build could change.
    """
    monkeypatch.setattr(entry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(entry.torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(entry.torch.cuda, "get_device_name", lambda index=None: card)
    monkeypatch.setattr(entry.torch.cuda, "get_device_capability", lambda index=None: capability)


T4 = ("Tesla T4", (7, 5))
A10G = ("NVIDIA A10G", (8, 6))
L4 = ("NVIDIA L4", (8, 9))
H100 = ("NVIDIA H100 80GB HBM3", (9, 0))


def test_a_run_that_would_work_is_not_refused_on_any_card_that_has_the_format(corpus, monkeypatch):
    """The requirement that matters most, because there is no waiver past this refusal.

    A false positive here does not cost somebody a message they can ignore, it stops the run.
    Ampere, Ada and Hopper all have bfloat16 and all must see nothing at all, and so must a
    host with no CUDA -- which is a ``--dry-run`` on a laptop, and is also this test suite.
    """
    _, config = configure()

    for card, capability in (A10G, L4, H100):
        on_a(card, capability, count=8, monkeypatch=monkeypatch)
        assert entry.devices_without_bfloat16() == []
        assert entry.a_precision_this_hardware_does_not_have(config) is None

    monkeypatch.setattr(entry.torch.cuda, "is_available", lambda: False)
    assert entry.a_precision_this_hardware_does_not_have(config) is None


def test_a_rocm_build_is_left_alone_because_its_numbers_mean_something_else(corpus, monkeypatch):
    # AMD reports its own architecture numbering through the same call, where 7.5 is not
    # Turing and the comparison is meaningless. Nothing on this platform is ROCm, which is
    # exactly why a wrong answer there would go unnoticed.
    _, config = configure()
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    monkeypatch.setattr(entry.torch.version, "hip", "6.2.0")

    assert entry.a_precision_this_hardware_does_not_have(config) is None


def test_a_check_that_cannot_read_the_device_gets_out_of_the_way(corpus, monkeypatch, capsys):
    """Mutation: let whatever the driver raises propagate.

    This runs in front of every training run in the repository, and several people are on
    their own branches at once. Missing a Turing card costs what today already costs; raising
    on a host this did not anticipate stops runs that were fine, which is strictly worse and
    is the failure mode a startup check is most likely to have.
    """
    _, config = configure()

    def unreadable(index=None):
        raise RuntimeError("no CUDA-capable device is detected")

    monkeypatch.setattr(entry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(entry.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(entry.torch.cuda, "get_device_capability", unreadable)

    assert entry.a_precision_this_hardware_does_not_have(config) is None
    assert "bfloat16 check is not running" in capsys.readouterr().err


def test_the_documented_command_is_refused_on_a_t4_although_it_says_no_dtype(corpus, monkeypatch):
    """The hole this exists to close, in the exact shape it arrives in.

    ``python .edullm/train_on_corpus.py "$EDULLM_RUN_ID"`` carries no bfloat16 token, so the
    platform's submission-time guard reads the command, finds nothing, and admits it onto
    ``gpu-8xt4`` -- which the 2026-08-04 capacity measurement leaves as the only multi-card
    shape this account can obtain. No argument below asks for bfloat16 and the run is one
    anyway, because the default in ``build_config`` is.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    _, config = configure()

    refusal = entry.a_precision_this_hardware_does_not_have(config)
    assert refusal is not None
    assert "train_module.dp_config.param_dtype" in refusal


def test_the_refusal_names_the_card_the_reason_and_both_ways_out(corpus, monkeypatch):
    """Mutation: say "bfloat16 is not supported on this device" and stop.

    Somebody reads this on a dead container with no access to its log stream and one
    question: what do I do now. The card and the capability say it is the hardware rather
    than the image, so nobody goes looking for a driver. The two remedies are the whole of
    what can be done about it, and naming them is the difference between a refusal and an
    obstacle -- the platform's own refusals are written this way.

    is_bf16_supported is named because a reviewer or a researcher who does not know it lies
    on this card will conclude the check is redundant and remove it.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    _, config = configure()

    refusal = entry.a_precision_this_hardware_does_not_have(config)
    assert refusal is not None
    assert "Tesla T4" in refusal and "7.5" in refusal
    assert "is_bf16_supported" in refusal
    assert "--param-dtype float32" in refusal
    assert "Ampere or newer" in refusal


def test_choosing_a_precision_the_card_has_is_the_way_past_it(corpus, monkeypatch):
    """There is no waiver, so the flag has to be a real way out rather than a gesture."""
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    for chosen in ("float32", "float16"):
        _, config = configure("--param-dtype", chosen)
        assert config.train_module.dp_config.param_dtype == chosen
        assert entry.a_precision_this_hardware_does_not_have(config) is None


def test_the_default_is_the_dtype_this_file_used_before_the_flag_existed(corpus):
    """Mutation: make the flag default to float32, which is the tempting safe choice.

    It would end the refusal and change the numerics of every run in flight. A sweep whose
    early submissions predate the flag and whose later ones follow it would then be two
    experiments reported as one, and nothing in the record would say so.
    """
    _, config = configure()
    assert config.train_module.dp_config.param_dtype == "bfloat16"


def test_bfloat16_reached_through_an_override_rather_than_the_flag_is_still_found(
    corpus, monkeypatch
):
    """Why this reads the built config instead of argv.

    Three spellings reach the same field -- the flag, the dotted override, and the default
    nobody typed -- and a fourth is one edit away. After ``Config.merge`` they are one value,
    so there is nothing left to recognise. ``reduce_dtype`` and ``autocast_precision`` are
    here because neither has a flag and both are reachable, which is the case a check written
    around ``--param-dtype`` would miss.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)

    _, config = configure("--param-dtype", "float32", "train_module.dp_config.param_dtype=bfloat16")
    assert entry.bfloat16_settings_in(config) == ["train_module.dp_config.param_dtype"]

    _, config = configure(
        "--param-dtype", "float32", "train_module.dp_config.reduce_dtype=bfloat16"
    )
    assert entry.bfloat16_settings_in(config) == ["train_module.dp_config.reduce_dtype"]

    _, config = configure("--param-dtype", "float32", "train_module.autocast_precision=bfloat16")
    assert entry.bfloat16_settings_in(config) == ["train_module.autocast_precision"]


def test_a_field_that_merely_holds_the_word_is_not_a_precision_request(corpus):
    """Mutation: match the value anywhere in the config and ignore what the key is called.

    There is no waiver past this refusal, so a false positive is a run that cannot be made to
    start. A dataset version, a run name or a save folder whose whole value happens to be that
    string is not somebody asking a T4 for bfloat16, and the key is what tells them apart.
    """
    _, config = configure("--param-dtype", "float32", "dataset_version=bfloat16")

    assert entry.bfloat16_settings_in(config) == []


def test_the_refusal_carries_a_stage_of_its_own_rather_than_exit_one(
    corpus, submitted, monkeypatch, capsys
):
    """The number is the only channel out of a container whose log nobody may read.

    Filed under the existing THE_CONFIG_WOULD_NOT_BUILD it would be exit 70, which says a
    field was renamed or a callback argument is wrong and sends the reader into the config.
    The config is correct; the machine is the wrong one.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    submitted()

    with pytest.raises(entry.Refusal) as refusal:
        entry.main()
    assert refusal.value.stage is entry.Stage.THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION

    # And through the boundary that turns a stage into the process's exit status, which is
    # the only part of any of this that reaches somebody on the platform side.
    assert entry.cli() == 73
    assert "edullm-stage: THE_DEVICE_CANNOT_DO_THE_REQUESTED_PRECISION" in capsys.readouterr().err


def test_a_dry_run_is_warned_and_not_stopped(corpus, submitted, monkeypatch, caplog):
    """Mutation: refuse the dry run too, which is one line shorter and blocks working runs.

    ``--dry-run`` resolves the corpus, prints the config and trains nothing, so it succeeds on
    a T4 and stopping it would be this file refusing a run that works. It is also the cheapest
    moment anybody will ever learn this, so it says so.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    submitted("--dry-run")

    with caplog.at_level(logging.WARNING):
        entry.main()

    assert "Tesla T4" in caplog.text


def test_the_check_runs_before_the_process_group_and_the_model(corpus, submitted, monkeypatch):
    """Mutation: move the check below ``prepare_training_environment``.

    It would still be far cheaper than failing on the first kernel and it would still be
    wrong: NCCL rendezvous on eight ranks, the model build and the data loader all happen
    first, and on a lost or mismatched rank the process group is itself a place a run can hang
    until its timeout. Nothing that costs a GPU may run before this answers.
    """
    on_a(*T4, count=8, monkeypatch=monkeypatch)
    submitted()

    def never(*args, **kwargs):
        raise AssertionError("the run reached the GPU before the precision was checked")

    monkeypatch.setattr(entry, "prepare_training_environment", never)
    monkeypatch.setattr(entry, "train", never)

    with pytest.raises(entry.Refusal):
        entry.main()


def test_the_platform_variables_are_required_and_named_when_missing(monkeypatch, capsys):
    for name in (
        "EDULLM_DATASET_ID",
        "EDULLM_DATASET_VERSION",
        "EDULLM_DATASET_TOKENIZER",
        "EDULLM_CHECKPOINT_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["train_on_corpus", "some-run"])

    with pytest.raises(SystemExit) as refusal:
        entry.main()

    # Naming all four at once rather than the first one missing: a person fixing a submission
    # should not have to discover them one failed run at a time.
    message = str(refusal.value)
    for name in ("EDULLM_DATASET_ID", "EDULLM_CHECKPOINT_DIR"):
        assert name in message


class Boom(Exception):
    """Whatever the reader raises, wrapped in a message like the ones botocore produces."""


def resolve_with_a_reader_that_raises(monkeypatch, exc: BaseException):
    import types

    class Boto3S3:
        @classmethod
        def default(cls, region="us-east-1"):
            return ReaderProtocolStub()

    def dataset_paths(dataset_id, version, *, s3, **_):
        raise exc

    read_module: Any = types.ModuleType("edullm_data.read")
    read_module.dataset_paths = dataset_paths
    read_module.resolve_latest = lambda dataset_id, *, s3, **_: "v1"
    s3_module: Any = types.ModuleType("edullm_data.s3")
    s3_module.Boto3S3 = Boto3S3
    monkeypatch.setitem(sys.modules, "edullm_data", types.ModuleType("edullm_data"))
    monkeypatch.setitem(sys.modules, "edullm_data.read", read_module)
    monkeypatch.setitem(sys.modules, "edullm_data.s3", s3_module)

    with pytest.raises(entry.Refusal) as refusal:
        entry.resolve_corpus(
            dataset_id="pretrain/regmix-10b", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
        )
    return refusal.value


def test_a_role_that_may_not_read_the_corpus_is_not_the_same_number_as_a_bad_run(monkeypatch):
    """Mutation: give every reader failure one code, which is what exit 1 already did.

    A missing ``s3:GetObject`` on ``edullm-data`` and a registry entry pointing at a prefix
    nobody published both arrive here as a failed read, and they have nothing in common: the
    first is an IAM change, the second is a dataset that is not there. Told apart at the
    exit code, the first question after a dead container is already answered.
    """
    denied = resolve_with_a_reader_that_raises(
        monkeypatch,
        Boom("An error occurred (AccessDenied) when calling the HeadObject operation"),
    )
    assert denied.stage is entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS

    absent = resolve_with_a_reader_that_raises(
        monkeypatch, Boom("An error occurred (NoSuchKey) when calling the GetObject operation")
    )
    assert absent.stage is entry.Stage.THE_CORPUS_IS_NOT_WHERE_THE_REGISTRY_SAYS

    # And something neither, which must not be filed as either -- a reader that changed under
    # us is a third thing, and calling it AccessDenied would send somebody to write a policy.
    other = resolve_with_a_reader_that_raises(monkeypatch, Boom("manifest is not valid JSON"))
    assert other.stage is entry.Stage.THE_READER_FAILED_IN_SOME_OTHER_WAY


def test_a_denial_wrapped_in_the_readers_own_exception_is_still_a_denial(monkeypatch):
    # The reader does not re-raise botocore's errors bare; it raises its own with the original
    # attached. Reading only the outermost message would file every denial as unrecognised.
    wrapped = Boom("could not read the seal")
    wrapped.__cause__ = Boom("AccessDenied")
    assert (
        resolve_with_a_reader_that_raises(monkeypatch, wrapped).stage
        is entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS
    )


def test_every_stage_has_a_number_of_its_own_and_none_collides_with_the_shell():
    """Mutation: number a stage 127, or reuse one.

    126, 127 and 128+n belong to the shell and the signal convention -- "cannot execute",
    "not found", "killed by signal n" -- and a stage sharing one of those is a stage that
    reads as an infrastructure failure forever.
    """
    numbers = [int(stage) for stage in entry.Stage]
    assert len(numbers) == len(set(numbers))
    assert all(64 <= number <= 78 for number in numbers)


def test_the_stage_survives_the_boundary_that_turns_it_into_an_exit_status(monkeypatch, capsys):
    """Mutation: let main's SystemExit reach the interpreter directly.

    ``SystemExit("a message")`` exits 1 and prints the message, which is exactly the
    indistinguishable failure this exists to end. The number only appears if something turns
    the refusal into one.
    """
    for name in ("EDULLM_DATASET_ID", "EDULLM_DATASET_VERSION", "EDULLM_DATASET_TOKENIZER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("EDULLM_CHECKPOINT_DIR", raising=False)
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    monkeypatch.setattr(sys, "argv", ["train_on_corpus", "some-run"])

    assert entry.cli() == int(entry.Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT)
    printed = capsys.readouterr().err
    assert "EDULLM_DATASET_ID" in printed
    assert "edullm-stage: THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT" in printed


def test_a_diagnostic_that_cannot_reach_wandb_does_not_replace_the_error_it_reports(monkeypatch):
    """Mutation: let the reporter's own failure propagate.

    W&B is reached over a network, and a container broken enough to die in startup may be
    broken in exactly that way. A reporter that raises turns "the role cannot read the
    corpus" into "connection refused", which is a worse answer than no answer.
    """
    import types

    monkeypatch.setenv("EDULLM_WANDB_PROJECT", "edullm-platform-smoke")
    exploding: Any = types.ModuleType("wandb")

    def refuse(*args, **kwargs):
        raise RuntimeError("no route to host")

    exploding.init = refuse
    exploding.run = None
    monkeypatch.setitem(sys.modules, "wandb", exploding)

    entry.leave_the_reason_in_wandb(
        run_name="run_x", stage=entry.Stage.THE_ROLE_MAY_NOT_READ_THE_CORPUS, explanation="denied"
    )


def test_nothing_is_sent_to_wandb_when_the_platform_named_no_project(monkeypatch):
    # Running the image by hand must not fail on a missing WANDB_API_KEY, which is the same
    # reason the trainer's own callback is enabled only when the project is set.
    import types

    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    tripwire: Any = types.ModuleType("wandb")

    def never(*args, **kwargs):
        raise AssertionError("W&B was reached without a project")

    tripwire.init = never
    tripwire.run = None
    monkeypatch.setitem(sys.modules, "wandb", tripwire)

    entry.leave_the_reason_in_wandb(
        run_name="run_x", stage=entry.Stage.TRAINING_ITSELF_FAILED, explanation="whatever"
    )


def write(path: Path, contents: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def torn(root: Path, step: int) -> Path:
    """The directory a host lost mid-write leaves, in the shape one actually did.

    ``run_019fbe1f-b84f-703a-8eb8-2b4504232948`` was terminated at step 100 immediately after
    ``train/rank0.pt`` landed and before the first ``model_and_optim`` shard started, leaving
    that one object and nothing else. ``rank0.pt`` first is not luck: the checkpointer writes
    the train state, then the shards, then ``.metadata.json`` last.
    """
    write(root / f"step{step}" / "train" / "rank0.pt")
    return root / f"step{step}"


def whole(root: Path, step: int) -> Path:
    """The three objects ``Checkpointer.dir_is_checkpoint`` requires of a full checkpoint."""
    write(root / f"step{step}" / "train" / "rank0.pt")
    write(root / f"step{step}" / "model_and_optim" / ".metadata")
    write(root / f"step{step}" / ".metadata.json", '{"version": "0"}')
    return root / f"step{step}"


def test_a_step_directory_holding_only_the_train_state_is_torn(tmp_path):
    """Mutation: judge a directory by whether it exists, or by whether it has any object in it.

    Either reading calls this whole, and the trainer then refuses to write the step because
    the directory is not empty, which is the failure being fixed. The judgement has to be the
    loader's own: ``dir_is_checkpoint`` is what ``find_checkpoints`` filters on, so what a
    resume would skip and what this clears are the same set by construction.
    """
    torn(tmp_path, 100)

    assert entry.torn_step_directories(str(tmp_path)) == [str(tmp_path / "step100")]


def test_a_finished_checkpoint_is_not_a_candidate_for_removal(tmp_path):
    """The safety property, and the reason this is not ``save_overwrite=True``.

    That flag clears whatever is at the target step before every save. This cannot reach a
    directory a resume would load, because the test for the second is the test for leaving it
    alone -- so the set of removable directories excludes every checkpoint by definition
    rather than by nothing having gone wrong yet.
    """
    whole(tmp_path, 50)
    torn(tmp_path, 100)

    assert entry.remove_torn_checkpoints(str(tmp_path)) == [str(tmp_path / "step100")]
    assert not (tmp_path / "step100").exists()
    kept = tmp_path / "step50"
    assert sorted(str(path.relative_to(kept)) for path in kept.rglob("*") if path.is_file()) == [
        ".metadata.json",
        "model_and_optim/.metadata",
        "train/rank0.pt",
    ]


def test_a_weights_only_directory_is_a_checkpoint_and_is_left_alone(tmp_path):
    # The other shape the loader accepts: a bare ``.metadata`` is model state, possibly with
    # optimizer state, and no trainer state. Reading only the three-object shape would clear a
    # directory somebody put there to resume weights from.
    write(tmp_path / "step100" / ".metadata")

    assert entry.torn_step_directories(str(tmp_path)) == []


def test_nothing_outside_a_step_directory_is_a_candidate(tmp_path):
    """Mutation: clear anything under the save folder that is not a checkpoint.

    The save folder holds more than step directories. ``ConfigSaverCallback`` writes
    ``config.json`` beside them, and that is the record of what the run was configured to do.
    """
    write(tmp_path / "config.json", "{}")
    write(tmp_path / "notes" / "something.txt")
    torn(tmp_path, 100)

    assert entry.remove_torn_checkpoints(str(tmp_path)) == [str(tmp_path / "step100")]
    assert (tmp_path / "config.json").is_file()
    assert (tmp_path / "notes" / "something.txt").is_file()


def test_a_save_folder_that_does_not_exist_yet_is_not_an_error(tmp_path):
    # Every first attempt. Listing a prefix nothing has been written to raises
    # FileNotFoundError locally and yields nothing on S3, and neither is a problem to report.
    assert entry.remove_torn_checkpoints(str(tmp_path / "never-written")) == []


class FakeModel:
    def __init__(self, parameters):
        self._parameters = parameters

    def parameters(self):
        return self._parameters


class FakeTrainModule:
    def __init__(self, parameters):
        self.model = FakeModel(parameters)


class FakeTrainer:
    def __init__(self, parameters, step):
        self.train_module = FakeTrainModule(parameters)
        self.global_step = step


@dataclass
class FakeOptions:
    run_name: str = "run_0"
    save_folder: str = "s3://bucket/teams/platform/runs/run_0/checkpoints/"


@dataclass
class FakeConfig:
    dataset_id: str = "pretrain/regmix-10b"
    dataset_version: str = "v1"


class FakeParameter:
    def __init__(self, count):
        self._count = count

    def numel(self):
        return self._count


def test_the_first_and_last_loss_are_kept_and_the_ones_between_are_not():
    """The summary reports both ends. Steps with no loss in their metrics are ignored."""
    watcher = entry.LossWatcher()

    watcher.log_metrics(1, {"throughput/device/TPS": 1000.0})
    watcher.log_metrics(2, {"train/CE loss": 6.9})
    watcher.log_metrics(3, {"train/CE loss": 6.5})
    watcher.log_metrics(4, {"train/CE loss": 6.1})

    assert watcher.first == 6.9
    assert watcher.last == 6.1


def test_the_summary_is_one_json_object_carrying_what_only_this_process_knows(capsys):
    """The platform reads this back out of the log stream, so it has to parse on its own."""
    import json

    watcher = entry.LossWatcher()
    watcher.log_metrics(1, {"train/CE loss": 6.9})
    watcher.log_metrics(2, {"train/CE loss": 6.1})

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(100), FakeParameter(90)], step=50),
        losses=watcher,
        seconds=12.5,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["parameters"] == 190
    assert printed["steps"] == 50
    assert printed["first_loss"] == 6.9
    assert printed["last_loss"] == 6.1
    assert printed["seconds"] == 12.5
    assert printed["dataset_id"] == "pretrain/regmix-10b"
    assert printed["checkpoint_uri"].endswith("/checkpoints/")


def test_a_summary_is_printed_even_when_no_step_reported_a_loss(capsys):
    """A run that printed nothing cannot be told apart from one that never started."""
    import json

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1)], step=0),
        losses=entry.LossWatcher(),
        seconds=0.5,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["first_loss"] is None
    assert printed["last_loss"] is None


def test_the_summary_survives_a_trainer_with_no_gpu_monitor_callback(capsys, monkeypatch):
    """The GPU branch of `summarise` must not assume a `gpu_monitor` callback is registered.

    Forced on rather than gated behind real hardware: with a CUDA device visible, `summarise`
    reads `trainer.callbacks` to upgrade its peak-memory figure from final-step to whole-run.
    A trainer that has no such callback must still emit its summary, because that JSON line is
    the only record the platform parses -- losing it loses the run's result.
    """
    import json

    monkeypatch.setattr(entry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(entry.torch.cuda, "max_memory_allocated", lambda: 8 * 1024**3)
    monkeypatch.setattr(entry.torch.cuda, "memory_allocated", lambda: 4 * 1024**3)

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(7)], step=3),
        losses=entry.LossWatcher(),
        seconds=1.5,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["parameters"] == 7
    assert printed["peak_memory_gib"] == pytest.approx(8.0)


def test_the_config_print_names_how_many_shards_rather_than_all_of_them(monkeypatch):
    """olmo-150b-dolma2 resolves to 6,851 objects and the dtype must stay readable."""
    printed = []
    monkeypatch.setattr(entry.rich, "print", lambda value: printed.append(value))
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(
                paths=[f"s3://edullm-data/x/v1/tokens/train-{n:05}.u32le.bin" for n in range(9)]
            ),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
        ]
    )
    config = entry.build_config(opts, overrides)
    paths = list(config.dataset.paths)
    entry.show(config)

    assert len(printed) == 1
    assert printed[0].dataset.paths == [f"<{len(paths)} objects>"]
    # The config itself is untouched, because the run trains on it after this prints.
    assert list(config.dataset.paths) == paths


def test_the_wandb_url_is_read_while_the_run_still_has_one(monkeypatch):
    """WandBCallback.post_train finishes the run, so reading it in summarise gets None."""
    import types

    watcher = entry.LossWatcher()
    fake = types.SimpleNamespace(run=types.SimpleNamespace(url="https://wandb.ai/o/p/runs/abc"))
    monkeypatch.setitem(sys.modules, "wandb", fake)

    watcher.log_metrics(1, {"train/CE loss": 6.9})
    assert watcher.wandb_url == "https://wandb.ai/o/p/runs/abc"

    # Once the run is finished the url is kept rather than overwritten with a blank.
    fake.run = None
    watcher.log_metrics(2, {"train/CE loss": 6.1})
    assert watcher.wandb_url == "https://wandb.ai/o/p/runs/abc"


def test_a_run_with_no_wandb_reports_a_blank_url_rather_than_failing(monkeypatch):
    watcher = entry.LossWatcher()
    monkeypatch.setitem(sys.modules, "wandb", None)

    watcher.log_metrics(1, {"train/CE loss": 6.9})

    assert watcher.wandb_url == ""
    assert watcher.first == 6.9


# ---------------------------------------------------------------------------------------
# The LR schedule, asserted by the numbers it produces rather than by the class it is.
#
# `assert isinstance(sched, LinearWithWarmup)` would pass with alpha_f left at OLMo-core's
# default 0.1, which is the exact defect this change exists to remove -- a "linear" schedule
# that stops at a tenth of peak and trains the last tokens at a rate it never meant to end on.
# So these call build_scheduler and read the curve.
# ---------------------------------------------------------------------------------------

PEAK_LR = 1.4e-3
HORIZON = 50_000  # steps; warmup_fraction=0.1 puts the peak at 5,000


def scheduler_for(*flags: str):
    """The scheduler the real CLI path builds, so a flag rename breaks this test."""
    opts, _ = entry.build_parser().parse_known_args(["a-run-id", *flags])
    return entry.build_scheduler(opts)


def test_the_default_schedule_decays_all_the_way_to_zero():
    """LINEAR, alpha_f=0.0. The endpoint is the whole point: EXACTLY 0.0, not 0.1*peak."""
    sched = scheduler_for()

    # Warmup is a fraction of the horizon, resolved against t_max by the scheduler itself.
    assert sched.get_lr(PEAK_LR, 0, HORIZON) == 0.0
    assert sched.get_lr(PEAK_LR, 2_500, HORIZON) == pytest.approx(0.7e-3)  # half way up
    assert sched.get_lr(PEAK_LR, 5_000, HORIZON) == pytest.approx(PEAK_LR)  # the peak

    # Midpoint of the decay leg (5,000 -> 50,000), so 27,500. Half of peak, because alpha_f=0.
    assert sched.get_lr(PEAK_LR, 27_500, HORIZON) == pytest.approx(0.7e-3)

    # The assertion this test exists for. At alpha_f=0.1 this returns 1.4e-4, which is 140,000x
    # larger than what it must be, and every existence check still passes.
    assert sched.get_lr(PEAK_LR, HORIZON, HORIZON) == 0.0
    assert sched.get_lr(PEAK_LR, 49_999, HORIZON) == pytest.approx(3.1111e-8, rel=1e-4)


def test_the_cosine_arm_stops_at_a_tenth_of_peak():
    """COSINE, alpha_f=0.1. E1's contrast is only a contrast if this arm does NOT reach zero."""
    sched = scheduler_for("--lr-schedule=cosine")

    assert sched.get_lr(PEAK_LR, 0, HORIZON) == 0.0
    assert sched.get_lr(PEAK_LR, 5_000, HORIZON) == pytest.approx(PEAK_LR)

    # Cosine midpoint sits ABOVE the linear midpoint: eta_min + (peak-eta_min)/2
    # = 1.4e-4 + (1.4e-3 - 1.4e-4)/2 = 7.7e-4, vs linear's 7.0e-4. Different curves, not just
    # different endpoints.
    assert sched.get_lr(PEAK_LR, 27_500, HORIZON) == pytest.approx(7.7e-4)

    assert sched.get_lr(PEAK_LR, HORIZON, HORIZON) == pytest.approx(1.4e-4)
    assert sched.get_lr(PEAK_LR, HORIZON, HORIZON) == pytest.approx(0.1 * PEAK_LR)


def test_the_two_schedules_differ_by_the_amount_the_experiment_is_measuring():
    """If these two ever returned the same tail, E1 would measure nothing and still be green."""
    linear = scheduler_for("--lr-schedule=linear")
    cosine = scheduler_for("--lr-schedule=cosine")

    # Identical through warmup -- both are _linear_warmup to the same peak.
    for step in (0, 2_500, 5_000):
        assert linear.get_lr(PEAK_LR, step, HORIZON) == pytest.approx(
            cosine.get_lr(PEAK_LR, step, HORIZON)
        )

    # And separated after it, by a final LR gap of exactly 0.1*peak.
    assert cosine.get_lr(PEAK_LR, HORIZON, HORIZON) - linear.get_lr(
        PEAK_LR, HORIZON, HORIZON
    ) == pytest.approx(1.4e-4)


def test_warmup_is_a_fraction_of_the_run_and_not_a_smoke_test_constant():
    """The old default was 20 steps -- 0.013% of a 40B run. It must now scale with the horizon."""
    sched = scheduler_for()
    assert sched.warmup is None and sched.warmup_fraction == 0.1

    # Same object, two horizons: the peak moves with the run rather than staying at step 20.
    assert sched.get_lr(PEAK_LR, 1_000, 10_000) == pytest.approx(PEAK_LR)  # warmup = 1,000
    assert sched.get_lr(PEAK_LR, 1_000, 100_000) == pytest.approx(0.1 * PEAK_LR)  # warmup=10,000

    # At the OLD default of 20 steps, a 50,000-step run is at full LR by step 20.
    old = scheduler_for("--warmup-steps=20")
    assert old.get_lr(PEAK_LR, 20, HORIZON) == pytest.approx(PEAK_LR)
    # ...whereas the new default is still climbing, at 0.4% of peak.
    assert sched.get_lr(PEAK_LR, 20, HORIZON) == pytest.approx(PEAK_LR * 20 / 5_000)


def test_an_explicit_warmup_step_count_suppresses_the_fraction():
    """Both at once is an OLMoConfigurationError, so the override has to clear the fraction."""
    sched = scheduler_for("--warmup-steps=1000")
    assert sched.warmup == 1_000
    assert sched.warmup_fraction is None
    assert sched.get_lr(PEAK_LR, 1_000, HORIZON) == pytest.approx(PEAK_LR)
    assert sched.get_lr(PEAK_LR, 500, HORIZON) == pytest.approx(0.5 * PEAK_LR)


def test_the_corrected_optimizer_values_reach_the_built_config(monkeypatch):
    """Every §2 knob, read off the config the CLI actually produces."""
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
        ]
    )
    config = entry.build_config(opts, overrides)
    optim = config.train_module.optim

    assert optim.lr == 1.4e-3
    assert optim.betas == (0.9, 0.98)
    assert optim.eps == 1e-10
    assert optim.weight_decay == 0.07
    # 786,432 = 192 x 4096. Asserted as the product so a typo'd digit fails here.
    assert config.data_loader.global_batch_size == 192 * 4096 == 786_432

    # z-loss is a train-module field, and `is not None` is what switches it on in the LM head.
    assert config.train_module.z_loss_multiplier == 1e-5

    # Neither of the two things that must not change.
    assert config.train_module.compile_model is True
    (embeddings,) = config.train_module.optim.group_overrides
    assert embeddings.params == ["embeddings.weight"]
    assert embeddings.opts == {"weight_decay": 0.0}


def test_zero_means_off_for_z_loss_rather_than_on_with_no_coefficient(monkeypatch):
    """`0.0 is not None` is True, so a bare pass-through would enable a no-op z-loss."""
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifest(),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            "--z-loss-multiplier=0",
        ]
    )
    config = entry.build_config(opts, overrides)
    assert config.train_module.z_loss_multiplier is None


def test_raising_weight_decay_does_not_reach_the_embeddings():
    """The exemption is a param GROUP, and it has to survive weight_decay becoming non-default.

    Built against a real model rather than asserted on the config, because what matters is what
    `build_groups` splats onto the group -- the config alone cannot show that.
    """
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.optim import AdamWConfig, OptimGroupOverride

    model = TransformerConfig.olmo2_190M(vocab_size=100_352).build()
    optim = AdamWConfig(
        lr=1.4e-3,
        betas=(0.9, 0.98),
        eps=1e-10,
        weight_decay=0.07,
        group_overrides=[
            OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
        ],
    ).build(model)

    exempt = [g for g in optim.param_groups if g["weight_decay"] == 0.0]
    decayed = [g for g in optim.param_groups if g["weight_decay"] != 0.0]

    assert len(exempt) == 1 and len(exempt[0]["params"]) == 1
    assert exempt[0]["params"][0].shape[0] == 100_352  # it really is the embedding matrix
    assert decayed and all(g["weight_decay"] == 0.07 for g in decayed)
    # Nothing fell out of the optimizer on the way through.
    assert sum(len(g["params"]) for g in optim.param_groups) == len(list(model.parameters()))
    # The other three optimizer knobs reach every group, exemption included.
    for group in optim.param_groups:
        assert group["betas"] == (0.9, 0.98)
        assert group["eps"] == 1e-10
        assert group["lr"] == 1.4e-3


# ----------------------------------------------------------------------------------------------
# The held-out ladder. Ported from `.edullm/train_liv_arm.py` on
# `agent/claude-01/liv-short-conv-mixer` (3274f33 -> 016c702), where this code has already
# produced a real 13-point CE curve (2.8681@1000 -> 2.0392@12716).
# ----------------------------------------------------------------------------------------------


@dataclass
class FakeManifestWithVal(FakeManifest):
    """``FakeManifest`` plus the held-out split the reader resolves for free.

    A SEPARATE class rather than a field on ``FakeManifest``, so that every test above keeps
    handing ``corpus_from_manifest`` an object with no ``val`` attribute at all. That is the
    duck-typed case the ``getattr(read, "val", None)`` read exists for, and folding the field
    into the base class would delete the only coverage it has.
    """

    val: Optional[List[str]] = None


@pytest.mark.parametrize(
    "steps,expected",
    [
        # int() TRUNCATES: 762*0.35 = 266.7 -> 266 and 762*0.75 = 571.5 -> 571, not 267/572.
        (762, [38, 76, 152, 266, 381, 571]),
        # The entry point's own --steps default.
        (200, [10, 20, 40, 70, 100, 150]),
        # Short enough that 0.05/0.1 collapse into the floor and de-duplicate: 1,2,4,7,10,15
        # becomes five rungs, not six.
        (20, [2, 4, 7, 10, 15]),
    ],
)
def test_the_ladder_rungs_are_the_specific_integers_the_run_evaluates_at(steps, expected):
    """
    Pin the rungs as NUMBERS, because an analysis script that rounds instead of truncating
    would look for rungs that were never evaluated and find nothing.

    Calls ``entry.ladder_steps`` -- the function ``build_config`` itself calls -- rather than
    re-deriving the set comprehension. A test that recomputes the arithmetic locally pins its
    own copy: it stays green when the real fractions are changed, which is exactly the
    regression it would be written to prevent.
    """
    rungs = entry.ladder_steps(steps)
    assert rungs == expected
    # post_step returns early for step <= 1 (train/callbacks/evaluator_callback.py:107-109), so
    # a rung there would silently never fire and read as "evaluated, no gap".
    assert min(rungs) >= 2
    # 1.0 is deliberately absent from LADDER_FRACTIONS -- eval_on_finish already scores the
    # final step, and listing it too would score the same model twice.
    assert steps not in rungs
    assert rungs == sorted(set(rungs)), "rungs must be unique and ascending"
    assert len(rungs) <= len(entry.LADDER_FRACTIONS)


def _wire_a_corpus(monkeypatch, val_paths):
    """Point ``resolve_corpus`` at a manifest carrying ``val_paths``, without touching S3."""
    monkeypatch.setattr(
        entry,
        "resolve_corpus",
        lambda **kwargs: entry.corpus_from_manifest(
            FakeManifestWithVal(val=val_paths),
            dataset_id=kwargs["dataset_id"],
            version=kwargs["version"],
            tokenizer_id=kwargs["tokenizer_id"],
        ),
    )


def _build_with_heldout(monkeypatch, tmp_path, val_paths, steps=200):
    """Build the real config with the held-out block live, faking only the S3 download.

    ``_download_to`` is replaced with something that writes bytes, and ``get_file_size`` with
    the size it writes, so ``_localised_heldout_paths`` runs its real logic -- the URL test,
    the cache path, the size comparison -- against a filesystem instead of a bucket.

    ``LOCAL_RANK`` is pinned to "0" so these build as the fetching process. Without it the test
    outcome would depend on whatever the ambient environment happens to carry, and a shell that
    exported LOCAL_RANK=1 would turn the download assertions into silent no-ops.
    """
    _wire_a_corpus(monkeypatch, val_paths)
    monkeypatch.setenv("LOCAL_RANK", "0")

    payload = b"\x00\x01\x02\x03" * 4

    def fake_download(url, dest):
        Path(dest).write_bytes(payload)

    monkeypatch.setattr(entry, "_download_to", fake_download)
    monkeypatch.setattr("olmo_core.io.get_file_size", lambda url: len(payload))

    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            f"--work-dir={tmp_path}",
            f"--steps={steps}",
        ]
    )
    return opts, entry.build_config(opts, overrides)


def test_every_heldout_path_reaching_the_evaluator_is_local_and_is_the_sorted_prefix(
    monkeypatch, tmp_path
):
    """
    THREE PROPERTIES, EACH WITH A SCAR.

    LOCAL, because ``iter_document_indices`` (data/utils.py:193-197) only scans the array for
    EOS boundaries when the path is NOT a url. For an ``s3://`` path it derives a sidecar name
    as ``basename.replace(".npy", ".csv.gz")`` (:217) -- a no-op on ``.u32le.bin``, so the
    "metadata file" resolves to the shard itself and ``gzip.open`` is handed raw uint32.
    ``run_019fce60`` died exit 70 on exactly that.

    EXACTLY ``min(HELDOUT_SHARDS, len(val_paths))``, because ``prepare()`` builds a per-shard
    index for every path at startup and a whole corpus of them costs more than the eval.

    THE SORTED PREFIX, because the subset must be identical across arms and seeds. A per-cell
    subset makes the rungs incomparable, which is the one thing a ladder cannot tolerate.
    """
    from olmo_core.io import is_url

    # Deliberately UNSORTED, and more shards than the ladder wants, so both the truncation and
    # the ordering are observable rather than accidental.
    val = [
        "s3://edullm-data/x/v1/tokens/src/val-00004.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00003.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00001.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00002.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00005.u32le.bin",
    ]
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)

    eval_dataset = config.trainer.callbacks["lm_eval"].eval_dataset
    paths = eval_dataset.paths

    assert entry.HELDOUT_SHARDS == 4
    assert len(paths) == min(entry.HELDOUT_SHARDS, len(val)) == 4
    for path in paths:
        assert not is_url(path), f"held-out paths must be local, got {path}"
    # The sorted prefix of the six, in order: 00000, 00001, 00002, 00003. Basenames rather than
    # full paths, because the directory is the run's work dir.
    assert [Path(p).name for p in paths] == [
        "val-00000.u32le.bin",
        "val-00001.u32le.bin",
        "val-00002.u32le.bin",
        "val-00003.u32le.bin",
    ]
    # And they landed under the work dir both invocations share, not somewhere per-process.
    for path in paths:
        assert Path(path).parent == tmp_path / "heldout-shards"


def test_fewer_val_shards_than_the_ladder_wants_uses_all_of_them(monkeypatch, tmp_path):
    """``min(HELDOUT_SHARDS, len(val_paths))``, exercised from the short side.

    A slice would silently produce two paths here whether the floor were min() or not, so the
    case is only load-bearing together with the six-shard test above.
    """
    val = [
        "s3://edullm-data/x/v1/tokens/src/val-00001.u32le.bin",
        "s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin",
    ]
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)
    paths = config.trainer.callbacks["lm_eval"].eval_dataset.paths
    assert len(paths) == min(entry.HELDOUT_SHARDS, len(val)) == 2
    assert [Path(p).name for p in paths] == ["val-00000.u32le.bin", "val-00001.u32le.bin"]


def test_the_eval_dataset_is_padded_and_carries_a_label_for_every_path(monkeypatch, tmp_path):
    """
    Two library contracts that raise at BUILD time, asserted where they can regress.

    ``LMEvaluatorCallbackConfig.build`` raises ``OLMoConfigurationError`` unless the dataset is
    a ``NumpyPaddedFSLDataset`` (train/callbacks/evaluator_callback.py:268-272), so the plain
    ``NumpyFSLDatasetConfig`` the training path uses fails there.

    ``LMEvaluator.from_numpy_dataset`` zips ``dataset.paths`` with ``dataset.metadata`` and
    raises on any path whose metadata lacks a ``"label"`` key (eval/lm_evaluator.py:60-66).
    The zip means a SHORT metadata list is worse than a missing one: it silently drops the
    trailing paths instead of raising, so the count is asserted, not just the keys.
    """
    from olmo_core.data import NumpyPaddedFSLDatasetConfig

    val = [f"s3://edullm-data/x/v1/tokens/src/val-0000{i}.u32le.bin" for i in range(6)]
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)
    eval_dataset = config.trainer.callbacks["lm_eval"].eval_dataset

    assert isinstance(eval_dataset, NumpyPaddedFSLDatasetConfig)
    # Both fields are Optional on the config, and `None` is the shape that makes the zip in
    # from_numpy_dataset yield nothing at all -- so pin non-None before counting.
    metadata = eval_dataset.metadata
    paths = eval_dataset.paths
    assert metadata is not None and paths is not None
    assert len(metadata) == len(paths) == 4
    assert all("label" in m for m in metadata)
    assert {m["label"] for m in metadata} == {"heldout-val"}


def test_the_eval_dataset_reads_the_corpus_width_and_vocab_rather_than_a_default(
    monkeypatch, tmp_path
):
    """
    ``get_dtype()`` falls back to the NARROWEST dtype the tokenizer's vocab fits in when
    ``dtype`` is unset -- 100,278 fits in uint16, and these corpora are uint32, so a default
    here reads every token two bytes at a time and never raises. Same trap the training path
    documents in the module header; the eval path is a second place to fall into it.

    The vocab comes off the corpus too. Nothing here pins a number.
    """
    val = ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"]
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)
    eval_dataset = config.trainer.callbacks["lm_eval"].eval_dataset

    assert eval_dataset.dtype == "uint32"
    # Same dtype the training dataset got, from the same manifest field.
    assert eval_dataset.dtype == config.dataset.dtype
    # Same tokenizer the model's vocab was derived from -- dolma2's 100,278 padding to 100,352
    # -- rather than a literal written here. Compared by VALUE, not identity: `config.merge`
    # deep-copies, so the two are equal rather than the same object by the time this sees them.
    assert eval_dataset.tokenizer == config.dataset.tokenizer
    assert eval_dataset.tokenizer.vocab_size == 100_278
    assert eval_dataset.tokenizer.padded_vocab_size() == 100_352
    # And that padded vocab is what the model was actually built at.
    assert config.model.vocab_size == 100_352


def test_both_invocations_share_a_work_dir_and_a_sequence_length_the_module_accepts(
    monkeypatch, tmp_path
):
    """
    ``--work-dir`` is what makes ``--prepare-heldout-only`` worth running: the indices it
    writes are found by the eval callback only if both processes name the same directory. If
    they diverge, ``paths_needed`` is non-empty inside the distributed program and the
    96-worker pool opens behind a collective again -- the exit-72 deadlock, twice, ~$11.

    Sequence length: ``LMEvaluatorCallbackConfig.build`` raises when the eval dataset is longer
    than ``train_module.eval_batch_spec.max_sequence_length``, which for the transformer module
    is just ``self.max_sequence_length``
    (train/train_module/transformer/train_module.py:205-210).
    """
    opts, config = _build_with_heldout(
        monkeypatch, tmp_path, ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"]
    )
    eval_dataset = config.trainer.callbacks["lm_eval"].eval_dataset

    assert eval_dataset.work_dir == opts.work_dir == str(tmp_path)
    assert eval_dataset.work_dir == config.dataset.work_dir
    assert eval_dataset.sequence_length <= config.train_module.max_sequence_length
    assert eval_dataset.sequence_length == 2048


def test_the_ladder_the_config_carries_is_the_one_ladder_steps_computed(monkeypatch, tmp_path):
    """The rungs on the callback come from the shared function, not from a second copy inline."""
    _, config = _build_with_heldout(
        monkeypatch, tmp_path, ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"], steps=762
    )
    callback = config.trainer.callbacks["lm_eval"]
    assert callback.fixed_steps == entry.ladder_steps(762) == [38, 76, 152, 266, 381, 571]
    # None, so nothing but fixed_steps and eval_on_finish triggers an eval. A number here would
    # add unrequested rungs and change what every cell costs.
    assert callback.eval_interval is None
    assert callback.eval_on_finish is True


def test_every_field_on_the_eval_callback_is_pinned_including_what_a_rung_costs(
    monkeypatch, tmp_path
):
    """
    THE COST KNOB, PINNED. ``eval_duration`` was the one field on this callback that nothing
    asserted, and it is the field whose default is expensive: ``LMEvaluatorCallbackConfig``
    defaults it to ``Duration.epochs(1)``
    (train/callbacks/evaluator_callback.py:227), which scores every shard IN FULL at every
    rung. With a six-rung ladder over four shards that is the eval costing more than the
    training it measures, and it fails no test and raises nothing -- the run is just slower and
    more expensive, which is invisible in the metrics.

    Asserted as UNIT AND VALUE, not merely non-None. ``Duration`` is a dataclass of
    ``(value, unit)`` (train/common.py:36-45), so ``Duration.epochs(32)`` and
    ``Duration.steps(32)`` both have value 32 and only the unit tells them apart -- an
    assertion on the value alone would pass through the exact swap that matters.

    Every remaining field is pinned here too, so that the library changing a default cannot
    quietly move what this experiment does. The audit lists these as the callback's full
    surface, and this test is what makes that list enforceable rather than descriptive.
    """
    from olmo_core.train.common import Duration, DurationUnit

    _, config = _build_with_heldout(
        monkeypatch, tmp_path, ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"], steps=200
    )
    callback = config.trainer.callbacks["lm_eval"]

    assert callback.eval_duration == Duration.steps(32)
    assert callback.eval_duration.unit == DurationUnit.steps
    assert callback.eval_duration.value == 32
    # Not the library default, which is the whole point of pinning it.
    assert callback.eval_duration != Duration.epochs(1)
    assert callback.eval_duration.unit != DurationUnit.epochs

    # The rest of the surface, so a changed library default cannot pass unnoticed.
    assert callback.eval_interval is None
    assert callback.eval_on_finish is True
    assert callback.eval_on_startup is False
    assert callback.cancel_after_first_eval is False
    assert callback.enabled is True
    assert callback.deterministic is True
    assert callback.log_interval == 5


def test_a_corpus_with_no_val_split_attaches_no_ladder_and_says_so(monkeypatch, tmp_path, caplog):
    """
    Not fatal -- eleven of sixteen releases have a val split, five correctly do not -- but it
    must not pass silently. Without a ladder the run still trains and still reports a training
    loss, and the missing endpoint is invisible until analysis.
    """
    import logging

    _wire_a_corpus(monkeypatch, None)
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            f"--work-dir={tmp_path}",
        ]
    )
    with caplog.at_level(logging.WARNING):
        config = entry.build_config(opts, overrides)

    assert "lm_eval" not in config.trainer.callbacks
    assert any("NO held-out split" in r.message for r in caplog.records)


def test_a_val_shard_that_is_also_a_training_shard_is_refused_by_number(tmp_path):
    """
    CONTAMINATION, REFUSED LOUDLY. This project has already shipped a "held-out" set whose
    shards were byte-copies of training shards, and the failure is invisible: the eval number
    just looks better than it is.

    Asserted on the STAGE'S INTEGER, because that number is the only channel out of a container
    that dies before W&B exists, and on the message naming the shard, because a refusal that
    does not say which one leaves the next person to diff two path lists by hand.
    """
    shared = "s3://edullm-data/x/v1/tokens/src/shard-00007.u32le.bin"
    manifest = FakeManifestWithVal(
        paths=["s3://edullm-data/x/v1/tokens/src/train-00000.u32le.bin", shared],
        val=[shared, "s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"],
    )
    with pytest.raises(entry.Refusal) as caught:
        resolve(manifest)

    assert caught.value.stage is entry.Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP
    assert int(caught.value.stage) == 68
    assert shared in caught.value.explanation
    assert "1 shard(s) in BOTH" in caught.value.explanation


def test_a_clean_split_carries_its_val_paths_through_untouched():
    """The refusal above must not fire on the normal case, and `val` must actually arrive."""
    val = ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"]
    corpus = resolve(
        FakeManifestWithVal(paths=["s3://edullm-data/x/v1/tokens/src/t-0.u32le.bin"], val=val)
    )
    assert corpus.val_paths == val
    assert corpus.paths == ["s3://edullm-data/x/v1/tokens/src/t-0.u32le.bin"]


def test_a_manifest_that_predates_the_val_field_still_resolves():
    """
    The ``getattr(read, "val", None)`` read is duck-typed on purpose: ``corpus_from_manifest``
    promises anything with paths/dtype/byte_order/header_bytes/rows will do, and ``FakeManifest``
    has no ``val`` attribute at all.
    """
    corpus = resolve(FakeManifest())
    assert corpus.val_paths == []
    # Empty rather than None, so callers branch on truthiness. ResolvedSplit.val itself returns
    # None for "no validation data" (edullm-data read.py:158-166).
    assert corpus.val_paths is not None


def test_train_does_not_prepare_the_heldout_indices():
    """
    THE FIX, ASSERTED WHERE IT CAN REGRESS. TWO RUNS AND ~$11 WENT ON THIS.

    ``NumpyPaddedFSLDataset.prepare()`` opens a bare ``ProcessPoolExecutor()`` -- 96 workers on
    a p4d.24xlarge, under a forced "spawn" start method -- and then every rank meets a
    ``barrier()``. Called from inside the distributed program it stranded seven ranks past
    gloo's 900-second timeout, twice, at exit 72 with nothing trained.

    The first attempt at a fix merely moved the call earlier in ``train()`` and failed
    identically, so "call it early" is not the invariant. The invariant is that ``train()``
    does not call it at all: the indices are built by a separate single-process invocation
    before ``torchrun``, and by the time the eval callback runs they are cached.

    A four-rank local reproduction on ten cores does NOT deadlock, so no unit test can catch a
    regression here dynamically. This asserts the structural property instead.
    """
    import inspect

    src = inspect.getsource(entry.train)
    assert "prepare_heldout_indices" not in src, (
        "train() must not prepare the held-out indices; that call belongs in the separate "
        "--prepare-heldout-only invocation, outside any process group"
    )


def test_the_prepare_only_path_exits_before_the_process_group_starts():
    """
    Order in ``main``: the prepare-only branch must return BEFORE
    ``prepare_training_environment()``. If a process group exists, the barrier inside
    ``prepare()`` is live again and the deadlock is back.
    """
    import inspect

    # Comments are stripped, so a comment naming prepare_training_environment cannot be
    # mistaken for the call. The prose above the branch mentions the function by name.
    lines = [
        line.split("#")[0]
        for line in inspect.getsource(entry.main).splitlines()
        if not line.strip().startswith("#")
    ]
    src = "\n".join(lines)
    branch_at = src.index("opts.prepare_heldout_only")
    env_at = src.index("prepare_training_environment()")
    assert branch_at < env_at, "the prepare-only branch must precede the process group"


def test_prepare_heldout_only_is_a_flag_and_defaults_off():
    """A normal training run must not silently turn into an index build."""
    parser = entry.build_parser()
    assert parser.parse_args([]).prepare_heldout_only is False
    assert parser.parse_args(["--prepare-heldout-only"]).prepare_heldout_only is True


def test_a_url_shard_would_be_gunzipped_so_heldout_paths_must_be_local():
    """
    THE BUG THAT KILLED run_019fce60 AT EXIT 70, pinned as arithmetic on a filename.

    ``iter_document_indices`` only scans the array for EOS boundaries when the path is NOT a
    url. For a url it derives a sidecar metadata filename as
    ``basename.replace(".npy", ".csv.gz")`` and gunzips it. Our shards end ``.u32le.bin``, so
    that replace is a no-op: the "metadata file" it resolves is the shard itself, which exists,
    so there is no FileNotFoundError -- it gunzips raw uint32 tokens and dies with
    ``BadGzipFile: Not a gzipped file (b'5\\x00')``. Token 53 is ``b"5\\x00\\x00\\x00"``.
    """
    import gzip
    import io
    import os

    assert os.path.basename("val-00033.u32le.bin").replace(".npy", ".csv.gz") == (
        "val-00033.u32le.bin"
    ), "if this ever differs, the library gained real sidecar support and the download may go"
    # And the same derivation IS meaningful for the naming the library was written for.
    assert os.path.basename("part-000.npy").replace(".npy", ".csv.gz") == "part-000.csv.gz"

    # Gunzipping little-endian uint32 tokens fails exactly the way production did.
    raw = (53).to_bytes(4, "little") + (100257).to_bytes(4, "little")
    with pytest.raises(gzip.BadGzipFile):
        with gzip.open(io.BytesIO(raw), "rt") as f:
            f.readline()


def test_a_cached_shard_of_the_right_size_is_not_downloaded_again(tmp_path, monkeypatch):
    """
    Size, not existence. A truncated download left by a killed attempt would otherwise be
    reused, and a short shard yields WRONG document boundaries rather than an error -- the
    ladder would report a number computed over the wrong instances.

    Runs as the fetching process (LOCAL_RANK=0): the truncation protection lives on that side
    of the gate, so pinning the rank is what keeps this a test of the size check rather than a
    test of the gate.
    """
    from unittest import mock

    monkeypatch.setenv("LOCAL_RANK", "0")

    class _Opts:
        work_dir = str(tmp_path)

    url = "s3://bucket/pretrain/x/val-00001.u32le.bin"
    payload = b"\xde\xad\xbe\xef" * 8
    cached = tmp_path / "heldout-shards" / "val-00001.u32le.bin"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)

    with mock.patch.object(entry, "_download_to") as download, mock.patch(
        "olmo_core.io.get_file_size", return_value=len(payload)
    ):
        out = entry._localised_heldout_paths([url], _Opts())
    download.assert_not_called()
    assert out == [str(cached)]

    # Now truncate it: the same call must re-fetch rather than trust what is on disk.
    cached.write_bytes(payload[:4])
    with mock.patch.object(entry, "_download_to") as download, mock.patch(
        "olmo_core.io.get_file_size", return_value=len(payload)
    ):
        entry._localised_heldout_paths([url], _Opts())
    download.assert_called_once()


def test_preparing_heldout_indices_warns_rather_than_raising_without_a_ladder():
    """
    A corpus with no val split attaches no ``lm_eval`` callback. Preparing must then be a
    no-op, not a KeyError -- the absence is a property of the corpus, not an error, and the
    platform runs the prepare step unconditionally before torchrun.
    """
    from unittest import mock

    from olmo_core.train import TrainerConfig
    from olmo_core.train.callbacks import GPUMemoryMonitorCallback

    trainer = TrainerConfig(
        save_folder="/tmp/does-not-matter", metrics_collect_interval=5, cancel_check_interval=5
    ).with_callback("gpu_monitor", GPUMemoryMonitorCallback())

    cfg = mock.Mock(trainer=trainer)
    with mock.patch.object(entry.log, "warning") as warn:
        entry.prepare_heldout_indices(cfg)
    assert warn.called, "a run with no ladder must say so rather than pass silently"


def test_only_one_process_per_node_heads_or_downloads_the_heldout_shards(monkeypatch, tmp_path):
    """
    THE S3 HEAD STORM, AND WHY THE OBVIOUS GATE WOULD HAVE BEEN A NO-OP.

    The size comparison IS the cache condition, so ``get_file_size`` -- an S3 HEAD -- is issued
    on every call even when the shard is already local. ``get_file_size`` is decorated
    ``@maybe_cache(condition=is_url)`` (io.py:107) and ``maybe_cache`` disables caching entirely
    unless ``OLMO_CORE_FS_CACHE_DIR`` is set (fs_cache.py:34-38); nothing in ``.edullm/`` sets
    it. ``build_config`` runs unguarded on every rank (main():1222), so ungated this is
    8 ranks x N shards of HEADs at config-build time, and a throttle on any one of them dies
    inside ``during(THE_CONFIG_WOULD_NOT_BUILD)`` at exit 70 -- which looks exactly like the
    sidecar bug.

    ``get_rank()``/``get_fs_local_rank()`` CANNOT be the gate: build_config runs BEFORE
    ``prepare_training_environment()`` (:1237), so ``dist.is_initialized()`` is False and both
    helpers return 0 in all eight workers (distributed/utils.py:249-256, :301-307). The gate
    reads ``LOCAL_RANK`` from the environment, which torchrun sets at spawn.
    """
    from unittest import mock

    class _Opts:
        work_dir = str(tmp_path)

    url = "s3://bucket/pretrain/x/val-00001.u32le.bin"

    # A non-zero local rank: no HEAD, no download, but still a usable local path.
    monkeypatch.setenv("LOCAL_RANK", "3")
    with mock.patch.object(entry, "_download_to") as download, mock.patch(
        "olmo_core.io.get_file_size", side_effect=AssertionError("rank 3 must not issue a HEAD")
    ):
        out = entry._localised_heldout_paths([url], _Opts())
    download.assert_not_called()
    assert out == [str(tmp_path / "heldout-shards" / "val-00001.u32le.bin")]

    # Local rank 0 on the same box still does the real, size-verified fetch. The shard is
    # pre-created at the WRONG size so the size branch is actually reached: the condition is
    # `not dest.is_file() or size != get_file_size(url)`, which short-circuits on an absent
    # file and would never issue a HEAD at all.
    monkeypatch.setenv("LOCAL_RANK", "0")
    cached = tmp_path / "heldout-shards" / "val-00001.u32le.bin"
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(b"\x00" * 8)

    heads = []
    with mock.patch.object(entry, "_download_to") as download, mock.patch(
        "olmo_core.io.get_file_size", side_effect=lambda u: (heads.append(u), 64)[1]
    ):
        out_zero = entry._localised_heldout_paths([url], _Opts())
    assert heads == [url], "local rank 0 issues exactly one HEAD per shard"
    download.assert_called_once()
    # Both ranks agree on the string, which is what keeps _get_indices_path's SHA-256 cache key
    # identical between the prepare-only invocation and the eval callback.
    assert out_zero == out


def test_a_single_process_invocation_still_fetches(monkeypatch, tmp_path):
    """
    ``--prepare-heldout-only`` runs with no ``LOCAL_RANK`` in the environment, and it is
    precisely the invocation that must do the real download. A gate that defaulted to
    "not the fetcher" would make the prepare step a no-op and push the work back inside the
    distributed program -- the exit-72 deadlock.
    """
    from unittest import mock

    class _Opts:
        work_dir = str(tmp_path)

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert entry._may_fetch_heldout_shards() is True

    with mock.patch.object(entry, "_download_to") as download, mock.patch(
        "olmo_core.io.get_file_size", return_value=64
    ):
        entry._localised_heldout_paths(["s3://bucket/x/val-00000.u32le.bin"], _Opts())
    download.assert_called_once()


def test_the_part_file_is_per_process_so_two_writers_cannot_share_one(monkeypatch, tmp_path):
    """
    ``_download_to`` writes a temporary then renames. With one fixed ``.part`` name, two
    processes reaching this concurrently write the same path and rename it underneath each
    other, and the loser's partial bytes can end up as the shard -- wrong document boundaries,
    silently. The pid in the name makes the temporary unique per writer.
    """
    import os as _os

    captured = {}

    class _FakeS3:
        def download_file(self, bucket, key, path):
            captured["tmp"] = path
            Path(path).write_bytes(b"\x00\x01\x02\x03")

    fake_boto3 = type("_B", (), {"client": staticmethod(lambda _name: _FakeS3())})
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    dest = tmp_path / "val-00000.u32le.bin"
    entry._download_to("s3://bucket/x/val-00000.u32le.bin", dest)

    assert dest.is_file(), "the shard is renamed into place"
    assert captured["tmp"] != str(dest), "it is written via a temporary, not in place"
    assert str(_os.getpid()) in captured["tmp"], "the temporary must be per-process"
    assert captured["tmp"].endswith(".part")
    assert not Path(captured["tmp"]).exists(), "the temporary is renamed away, not left behind"


def test_a_fresh_model_scores_about_ln_vocab_and_the_evaluator_reports_that_number():
    """
    THE MAGNITUDE OF THE NUMBER THE EXPERIMENT EXISTS TO PRODUCE.

    An untrained model over V=100,352 is uniform, so its CE is ln(V) = 11.5164 nats and its PPL
    is ~V. Every plausible bug in this path lands somewhere else entirely: reading uint32 data
    as uint16 gives a garbage-but-finite loss, an unpopulated label gives NaN, and a metric
    reading its own zeros gives 0/0. Pinning the BAND is what tells those apart -- "a row
    exists" does not, because ``compute_metrics`` calls ``metric.update(0.0, 0.0)`` before
    ``compute()`` (eval/lm_evaluator.py:110-121), so a label that received NO data still emits
    a value rather than nothing.

    Driven through ``LMEvaluator``'s real ``update_metrics``/``compute_metrics`` with synthetic
    per-token losses rather than end-to-end: a real forward pass needs a GPU. What this pins is
    the arithmetic between a per-token CE and the reported row, which is where a
    masking/weighting mistake would show up.
    """
    import math

    import torch

    from olmo_core.eval.lm_evaluator import LMEvaluator

    ln_vocab = math.log(100_352)
    assert abs(ln_vocab - 11.5164) < 1e-3, "sanity: ln(100,352) is the step-0 target"

    evaluator = LMEvaluator(
        name="lm",
        batches=[],
        labels=["heldout-val"],
        device=torch.device("cpu"),
    )

    # Two instances of four tokens each, every token at the uniform-model loss.
    batch = {
        "metadata": [{"label": "heldout-val"}, {"label": "heldout-val"}],
        "label_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    ce_loss = torch.full((2, 4), ln_vocab, dtype=torch.float32)
    evaluator.update_metrics(batch, ce_loss, None)

    metrics = evaluator.compute_metrics()
    ce = float(metrics["heldout-val/CE loss"])
    ppl = float(metrics["heldout-val/PPL"])

    assert not math.isnan(ce), "NaN is what an unpopulated label reports; this one has data"
    # The band a freshly-initialised model must land in: not 0, not 2.0 (a trained model), not
    # a garbage magnitude from a mis-decoded width.
    assert 10.5 < ce < 12.5, f"step-0 CE must be near ln(vocab)=11.5164, got {ce}"
    assert abs(ce - ln_vocab) < 1e-4
    assert ce > 2.0, "2.0 is a CONVERGED loss; a fresh model cannot be there"
    # PPL is exp(CE), so ~the vocab size. This is the second half of the row the ladder writes.
    assert abs(ppl - 100_352) / 100_352 < 0.01, f"PPL should be ~vocab at step 0, got {ppl}"


def test_a_label_that_received_no_data_is_nan_rather_than_a_believable_number():
    """
    WHY "A ROW EXISTS" IS WEAK EVIDENCE, PINNED AS BEHAVIOUR.

    ``compute_metrics`` calls ``metric.update(0.0, 0.0)`` on every label before computing
    (eval/lm_evaluator.py:110-121), so a label the eval loop never reached still emits a row.
    ``MeanMetric.compute`` is ``weighted_sum / weight`` (eval/metrics.py:81-89), which for an
    untouched metric is 0.0/0.0 = NaN.

    That is the good outcome and this test pins it: NaN is loud in a plot. What must never
    happen is that empty label reporting 0.0, which would read as a perfect loss.
    """
    import math

    import torch

    from olmo_core.eval.lm_evaluator import LMEvaluator

    evaluator = LMEvaluator(
        name="lm", batches=[], labels=["heldout-val"], device=torch.device("cpu")
    )
    metrics = evaluator.compute_metrics()
    ce = float(metrics["heldout-val/CE loss"])

    assert math.isnan(ce), "an unpopulated label must be NaN, never a believable number"
    assert ce != 0.0, "0.0 would read as a perfect held-out loss"


# ---------------------------------------------------------------------------------------------
# Throughput flags: the quantized-weight cache and the QAT schedule.
#
# Both act on a quantizer, so both have to refuse a command line that asks for them without one.
# A silent no-op is the failure worth testing: the run would train, report itself as scheduled
# or cached, and be neither.


def _throughput_opts(*flags: str):
    opts, _ = entry.build_parser().parse_known_args(["a-run-id", *flags])
    return opts


def test_qat_start_is_none_unless_asked_for():
    assert entry._qat_start(_throughput_opts()) is None


def test_qat_start_reads_either_spelling():
    assert entry._qat_start(_throughput_opts("--qat-start-step", "1000")) == {"start_step": 1000}
    assert entry._qat_start(_throughput_opts("--qat-start-fraction", "0.7")) == {
        "start_fraction": 0.7
    }


def test_the_two_qat_spellings_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        entry.build_parser().parse_known_args(
            ["a-run-id", "--qat-start-step", "5", "--qat-start-fraction", "0.5"]
        )


def test_the_delta_factor_flag_reaches_the_built_quantizer():
    """The threshold decides which ~42% of every weight is zero, so the sweep over it has to
    be reachable from a command line, not only from Python."""
    from olmo_core.nn.quantization import TWN_DELTA_FACTOR
    from olmo_core.nn.transformer import TransformerConfig

    assert _throughput_opts().twn_delta_factor == TWN_DELTA_FACTOR
    assert _throughput_opts("--twn-delta-factor", "0.5").twn_delta_factor == 0.5

    config = TransformerConfig.maple_scaled(
        vocab_size=1024, rung="R0", quantize=True, delta_factor=0.5
    )
    # Maple ternarizes both the attention projections and the expert stack, so the factor has
    # to land on both quantized surfaces, not just whichever one the test happened to check.
    assert config.block.sequence_mixer.quant.delta_factor == 0.5
    assert config.block.feed_forward_moe.quant.delta_factor == 0.5

    default = TransformerConfig.maple_scaled(vocab_size=1024, rung="R0", quantize=True)
    assert default.block.sequence_mixer.quant.delta_factor == TWN_DELTA_FACTOR


def test_the_cache_flag_defaults_off():
    assert _throughput_opts().cache_quantized_weight is False
    assert _throughput_opts("--cache-quantized-weight").cache_quantized_weight is True
