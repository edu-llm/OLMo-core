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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def test_first_and_last_loss_sources_can_differ_and_each_is_reported_honestly():
    """
    DEFECT C, REPRODUCED AND FIXED. Before this fix ``first``/``last`` shared one
    ``loss_source`` field, set unconditionally to whatever loss was seen MOST RECENTLY --
    so once a held-out rung fired, the shared field silently relabelled ``first`` as held-out
    CE too, even though the number sitting in ``first`` was train CE from step 1 and had never
    been touched again. This drives exactly that sequence -- train CE first, held-out CE later
    -- and checks the two sources are allowed to disagree AND are each individually correct.
    """
    watcher = entry.LossWatcher()

    # Step 1: only train CE exists yet (no rung has fired). `first` is set here.
    watcher.log_metrics(1, {"train/CE loss": 6.9})
    assert watcher.first == 6.9
    assert watcher.first_loss_source == "train/CE loss"

    # Step 2: a rung fires, spread across two domains. Held-out always wins over train, no
    # guard needed on this branch -- `last` updates unconditionally.
    watcher.log_metrics(
        2,
        {
            "eval/lm/adult_content/CE loss": 2.0,
            "eval/lm/art_and_design/CE loss": 4.0,
            "train/CE loss": 5.5,
        },
    )
    assert watcher.last == 3.0  # unweighted mean of the two domains, not the train value
    assert watcher.last_loss_source == entry._HELDOUT_LOSS_SOURCE
    # `first` is untouched: it was already set, and first is written at most once.
    assert watcher.first == 6.9
    assert watcher.first_loss_source == "train/CE loss"
    # THE DEFECT C SCENARIO ITSELF: first and last are different QUANTITIES, and the two
    # separate fields say so rather than one shared field claiming they match.
    assert watcher.first_loss_source != watcher.last_loss_source

    # Step 3: train-only again (this rung's window skipped, or fixed_steps has passed). Must
    # NOT downgrade `last` back to a train reading.
    watcher.log_metrics(3, {"train/CE loss": 5.0})
    assert watcher.last == 3.0
    assert watcher.last_loss_source == entry._HELDOUT_LOSS_SOURCE

    # Step 4: another held-out rung. Held-out always overwrites `last` -- the guard only ever
    # protects `last` from a train-only downgrade, never from a fresher held-out reading.
    watcher.log_metrics(4, {"eval/lm/adult_content/CE loss": 1.0, "eval/lm/crime_and_law/CE loss": 3.0})
    assert watcher.last == 2.0
    assert watcher.last_loss_source == entry._HELDOUT_LOSS_SOURCE
    assert watcher.first == 6.9  # first never moves again once set
    assert watcher.first_loss_source == "train/CE loss"


def test_the_watcher_means_the_domains_present_and_ignores_ones_with_no_data_this_rung():
    """
    NaN-FILTERED AGGREGATION, PINNED AS BEHAVIOUR NOT EXISTENCE.

    ``MeanMetric.compute()`` is ``weighted_sum / weight`` (eval/metrics.py:81-89); a label the
    32-batch-capped rung never reached leaves ``weight`` at 0, so that label's CE is a REAL NaN,
    not a zero (``eval/lm_evaluator.py:110-121`` calls ``metric.update(0.0, 0.0)`` first, which
    does not change that). An unfiltered ``sum(values) / len(values)`` would let one NaN domain
    poison the whole rung's number; this pins that it does not, and that a rung where EVERY
    selected domain came back NaN correctly falls all the way back to train CE rather than
    reporting NaN as if it were a real held-out loss.
    """
    import math

    some_data = entry.LossWatcher()
    some_data.log_metrics(
        1,
        {
            "eval/lm/adult_content/CE loss": 2.0,
            "eval/lm/art_and_design/CE loss": float("nan"),
            "eval/lm/crime_and_law/CE loss": 4.0,
        },
    )
    assert some_data.last == 3.0, "mean of the two domains WITH data, the NaN one excluded"
    assert not math.isnan(some_data.last)
    assert some_data.last_loss_source == entry._HELDOUT_LOSS_SOURCE

    no_data_this_rung = entry.LossWatcher()
    no_data_this_rung.log_metrics(
        1,
        {
            "eval/lm/adult_content/CE loss": float("nan"),
            "eval/lm/art_and_design/CE loss": float("nan"),
            "train/CE loss": 7.0,
        },
    )
    assert no_data_this_rung.last == 7.0, "every selected domain was NaN, so fall back to train"
    assert no_data_this_rung.last_loss_source == "train/CE loss"


def test_the_watcher_key_matches_the_real_metric_names_the_evaluator_produces_for_the_same_labels(
    monkeypatch, tmp_path
):
    """
    THE COUPLING ITSELF, EXERCISED END TO END -- Defect B, reproduced and closed.

    Defect B was two functions drifting apart: selection relabels each held-out path by its
    topic domain (``_domain_of``, used both inside ``spread_across_sources`` and at the
    ``NumpyPaddedFSLDatasetConfig(metadata=...)`` call site in ``build_config``), and
    ``LossWatcher.log_metrics`` has to read the SAME domains back out of the metric names
    ``LMEvaluator.compute_metrics`` (eval/lm_evaluator.py:114-117) and
    ``EvaluatorCallback.perform_eval`` (train/callbacks/evaluator_callback.py:171) actually
    produce for those labels. A test that hand-writes one metric key and feeds it to
    ``LossWatcher`` in isolation could not catch the two drifting apart -- it would pass just as
    happily with a stale key on both sides, which is exactly how this bug happens. This drives
    BOTH halves from the same multi-domain fixture: the labels come out of the real
    ``build_config``, and the metric-name format comes out of a real ``LMEvaluator`` built with
    exactly those labels.
    """
    import torch

    from olmo_core.eval.lm_evaluator import LMEvaluator

    val = [
        "s3://edullm-data/x/v1/tokens/all-dressed-snazzy2/adult_content/val-00033.u32le.bin",
        "s3://edullm-data/x/v1/tokens/all-dressed-snazzy2/art_and_design/val-00212.u32le.bin",
        "s3://edullm-data/x/v1/tokens/all-dressed-snazzy2/crime_and_law/val-00336.u32le.bin",
    ]
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)
    metadata = config.trainer.callbacks["lm_eval"].eval_dataset.metadata
    assert metadata is not None
    labels = [m["label"] for m in metadata]
    assert set(labels) == {"adult_content", "art_and_design", "crime_and_law"}

    evaluator = LMEvaluator(name="lm", batches=[], labels=labels, device=torch.device("cpu"))
    per_domain_ce = {"adult_content": 2.0, "art_and_design": 4.0, "crime_and_law": 6.0}
    for label, ce in per_domain_ce.items():
        evaluator.update_metrics(
            {"metadata": [{"label": label}], "label_mask": torch.ones(1, 4, dtype=torch.bool)},
            torch.full((1, 4), ce, dtype=torch.float32),
            None,
        )
    raw = evaluator.compute_metrics()
    # The exact re-keying `EvaluatorCallback.perform_eval` applies
    # (train/callbacks/evaluator_callback.py:171): f"{prefix}/{evaluator.name}/{name}", prefix
    # "eval", evaluator.name "lm".
    reeval_metrics = {f"eval/lm/{name}": float(value) for name, value in raw.items()}

    watcher = entry.LossWatcher()
    watcher.log_metrics(1, reeval_metrics)

    assert watcher.last == pytest.approx(sum(per_domain_ce.values()) / len(per_domain_ce))
    assert watcher.last_loss_source == entry._HELDOUT_LOSS_SOURCE
    # The old single hardcoded key would have found nothing here and silently fallen back to
    # train CE -- there is none in this metrics dict, so that failure mode would show up as
    # `last` staying `None`, not as a wrong number. Asserted directly so a regression to the old
    # single-key lookup is loud rather than a quiet drop in signal.
    assert watcher.last is not None


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
    # A run that never logged anything has no source for either end either -- there is nothing
    # to have come from train CE or held-out CE, so both fields are None, not the string "None"
    # and not the empty string.
    assert printed["first_loss_source"] is None
    assert printed["last_loss_source"] is None


def test_the_summary_names_a_source_for_each_end_and_the_two_can_differ(capsys):
    """
    THE JSON HALF OF DEFECT C's FIX. The single ``loss_source`` field this replaces would have
    reported one string for both ``first_loss`` and ``last_loss`` even when they were different
    quantities (train CE at step 1, held-out CE later); a sweep reading that field to decide
    which number is comparable across cells would have been silently misled. This drives that
    exact disagreement through ``summarise`` and checks the printed JSON, not just the watcher's
    attributes directly -- ``summarise`` is where a stale key name would actually bite a
    consumer parsing the log stream.
    """
    import json

    watcher = entry.LossWatcher()
    watcher.log_metrics(1, {"train/CE loss": 6.9})
    watcher.log_metrics(2, {"eval/lm/adult_content/CE loss": 2.0, "eval/lm/art_and_design/CE loss": 4.0})

    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1)], step=2),
        losses=watcher,
        seconds=1.0,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["first_loss"] == 6.9
    assert printed["first_loss_source"] == "train/CE loss"
    assert printed["last_loss"] == 3.0
    assert printed["last_loss_source"] == entry._HELDOUT_LOSS_SOURCE
    assert printed["first_loss_source"] != printed["last_loss_source"]
    # The field this replaces must actually be gone, not just supplemented -- a consumer still
    # reading the old key name should get a loud KeyError rather than a quietly stale value.
    assert "loss_source" not in printed


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


# ---------------------------------------------------------------------------------------
# E1: the LR x schedule fan-out.
#
# 6 arms = {cosine alpha_f=0.1, linear alpha_f=0.0} x {5e-4, 1e-3, 2e-3}, 3 seeds = 18 cells.
# The deliverable is an argmin over the 6-point grid, so the thing that must not go wrong is
# 18 cells resolving to fewer than 18 distinct configurations. That failure is invisible:
# every cell runs, every cell writes a loss curve, and the curves look plausible.
# ---------------------------------------------------------------------------------------

E1_SCHEDULES = ("cosine", "linear")
E1_LRS = ("5e-4", "1e-3", "2e-3")
E1_SEEDS = (0, 1, 2)

#: The exact grid the 18-cell submission carries, built as the 2x3x3 product rather than
#: typed out, so this string and the intended design cannot drift apart.
E1_GRID = ",".join(
    f"{schedule}:{lr}:{seed}"
    for schedule in E1_SCHEDULES
    for lr in E1_LRS
    for seed in E1_SEEDS
)

#: D and B for the proxy. B_opt = 0.0306*D^0.383 in 2048-token sequences (Power Lines
#: 2505.13738) gives 130.45 seqs = 267,171 tokens at D=3e9, so 262,144 is 0.98x optimum.
#: The FLAGSHIP default of 786,432 would be 2.94x optimum here, and Power Lines Table 1
#: measures 4x-above-optimum at +0.029 nats -- larger than the 0.025-nat schedule effect
#: this experiment exists to measure.
E1_TARGET_TOKENS = 3e9
E1_BATCH = 262_144


def test_all_eighteen_array_indices_resolve_to_eighteen_distinct_cells():
    """THE HIGHEST-VALUE TEST IN THE EXPERIMENT.

    A grid that resolves to fewer than 18 distinct (schedule, lr, seed) triples spends the
    full 18-cell budget and delivers a sweep with a hole in it, and NOTHING downstream can
    see that: the duplicated cell trains, converges, and writes a loss curve as plausible as
    any other. The precedent's own commit message records the version of this that shipped --
    "a 12-cell pilot would have trained the SAME arm and seed twelve times."

    So this walks array indices 0..17 the way Batch will and asserts the resolved set IS the
    intended 2x3x3 product, not merely that it has 18 members.
    """
    resolved = [entry.resolve_fanout_cell(E1_GRID, str(i), 18) for i in range(18)]

    assert len(resolved) == 18
    assert all(cell is not None for cell in resolved)

    triples = [(c.schedule, c.lr, c.seed) for c in resolved]
    assert len(set(triples)) == 18, "two array indices resolved to the same configuration"

    # The set equals the intended product -- not just "18 distinct things", which a grid of
    # 18 wrong-but-distinct cells would also satisfy.
    intended = {
        (schedule, float(lr), seed)
        for schedule in E1_SCHEDULES
        for lr in E1_LRS
        for seed in E1_SEEDS
    }
    assert set(triples) == intended

    # And the marginals, which is how a grid missing one arm and doubling another shows up.
    for schedule in E1_SCHEDULES:
        assert sum(1 for s, _, _ in triples if s == schedule) == 9
    for lr in E1_LRS:
        assert sum(1 for _, value, _ in triples if value == float(lr)) == 6
    for seed in E1_SEEDS:
        assert sum(1 for _, _, s in triples if s == seed) == 6

    # Every (schedule, lr) arm carries all three seeds. A 6-point argmin averaged over an
    # arm that is missing a seed is an argmin over unequal sample sizes.
    for schedule in E1_SCHEDULES:
        for lr in E1_LRS:
            seeds = sorted(s for sc, v, s in triples if sc == schedule and v == float(lr))
            assert seeds == [0, 1, 2], (schedule, lr, seeds)


def test_the_array_index_is_what_selects_the_cell():
    """Mutation: ignore AWS_BATCH_JOB_ARRAY_INDEX and return cells[0].

    ``fanout_index_parameter`` on the submission form is documentation and substitutes
    nothing into the command. Batch sets the variable and the program must read it, or all
    18 cells train one configuration at 18x the price and the sweep reports as complete.
    """
    first = entry.resolve_fanout_cell(E1_GRID, "0", 18)
    last = entry.resolve_fanout_cell(E1_GRID, "17", 18)

    assert first is not None and last is not None
    assert (first.schedule, first.lr, first.seed) == ("cosine", 5e-4, 0)
    assert (last.schedule, last.lr, last.seed) == ("linear", 2e-3, 2)
    assert first != last

    # And the ordering is positional, so index i is cell i of the string the approver read.
    assert entry.resolve_fanout_cell(E1_GRID, "9", 18) == entry.parse_fanout_grid(E1_GRID)[9]


def test_without_a_grid_a_single_run_is_completely_unaffected():
    """The fan-out machinery must be inert when nobody asked for it."""
    assert entry.resolve_fanout_cell("", None) is None
    assert entry.resolve_fanout_cell("", "3") is None

    opts, _ = entry.build_parser().parse_known_args(["a-run-id"])
    before = dict(vars(opts))

    assert entry.apply_fanout_cell(opts, None) is None
    assert vars(opts) == before, "no grid, but apply_fanout_cell changed the options"


def test_a_grid_without_an_array_index_is_refused():
    """Refusing beats defaulting to cell 0.

    A grid submitted without the fan-out fields would otherwise run one configuration N
    times, at N times the price, and be indistinguishable from a completed sweep.
    """
    with pytest.raises(entry.Refusal) as refusal:
        entry.resolve_fanout_cell(E1_GRID, None)
    assert "AWS_BATCH_JOB_ARRAY_INDEX" in refusal.value.explanation
    assert refusal.value.stage is entry.Stage.THE_PLATFORM_DID_NOT_SET_THE_ENVIRONMENT


def test_an_index_past_the_end_of_the_grid_is_refused():
    """fanout_size and the grid must agree, or trailing cells vanish silently.

    A `fanout_size` of 20 against an 18-cell grid runs cells 18 and 19 into this refusal
    rather than into a default -- and a size of 16 is caught by --fanout-expect below.
    """
    with pytest.raises(entry.Refusal):
        entry.resolve_fanout_cell(E1_GRID, "18", 18)
    with pytest.raises(entry.Refusal):
        entry.resolve_fanout_cell(E1_GRID, "-1", 18)


def test_a_grid_holding_the_same_cell_twice_is_refused():
    """THE REFUSAL THE PRECEDENT LACKS.

    ``train_liv_arm.py:parse_fanout_grid`` validates each cell in isolation and would accept
    ``L0:0,L0:0`` without complaint. That is the same "trains one cell N times" failure its
    own commit describes, arriving by a different road: the index is read correctly, every
    cell resolves, and two of them are simply the same run -- at twice the price, with a hole
    where a third configuration should have been, and two plausible loss curves to show for
    it.

    The last case is the one a careless edit actually produces: the same cell spelled two
    different ways. 5e-4 and 0.0005 are one configuration and must collide.
    """
    for duplicated in (
        "linear:2e-3:0,linear:2e-3:0",
        "cosine:1e-3:1,linear:1e-3:1,cosine:1e-3:1",
        "linear:5e-4:0,linear:0.0005:0",
    ):
        with pytest.raises(entry.Refusal) as refusal:
            entry.parse_fanout_grid(duplicated)
        assert "twice" in refusal.value.explanation

    # The 18-cell grid itself is clean, so the check is not vacuously passing on everything.
    assert len(entry.parse_fanout_grid(E1_GRID)) == 18


def test_a_grid_of_the_wrong_length_is_refused_against_the_declared_size():
    """The other half of the fanout_size guard.

    The duplicate check catches a grid that repeats itself; this catches one that is simply
    short -- 17 cells typed where 18 were meant and submitted with fanout_size 18. Without
    it, cell 17 dies on an out-of-range index AFTER the other 17 have already been billed,
    and a sweep missing one of its six arms' seeds still reports 17 finished runs.
    """
    seventeen = ",".join(E1_GRID.split(",")[:17])

    with pytest.raises(entry.Refusal) as refusal:
        entry.parse_fanout_grid(seventeen, 18)
    assert "17" in refusal.value.explanation and "18" in refusal.value.explanation

    # Unset means unchecked, so an ad-hoc grid is unaffected.
    assert len(entry.parse_fanout_grid(seventeen)) == 17


def test_a_malformed_cell_or_unknown_schedule_is_refused_and_quotes_what_was_typed():
    """A typo must not fall back to a default, and the message must name the typo.

    Parsing the LR to a float and then reporting THAT would print "0.002" for a cell typed
    "2-e3", which tells the submitter nothing about what to fix.
    """
    for bad in (
        "linear:2e-3",  # no seed
        "linear",  # no separators at all
        "linear:2e-3:0:extra",  # too many fields
        "linear:2e-3:x",  # seed is not an integer
        "linear:notanumber:0",  # LR is not a number
        "linear:-1e-3:0",  # LR is negative: builds an optimizer that trains backwards
        "linear:0:0",  # LR is zero: trains nothing, converges to the init loss
        "cosinme:2e-3:0",  # schedule typo
        "linear:2e-3:0,cosinme:2e-3:0",  # good cell first, so the loop must keep checking
    ):
        with pytest.raises(entry.Refusal):
            entry.parse_fanout_grid(bad)

    # An unknown schedule names the ones that exist rather than silently picking one.
    with pytest.raises(entry.Refusal) as refusal:
        entry.parse_fanout_grid("cosinme:2e-3:0")
    assert "cosine" in refusal.value.explanation and "linear" in refusal.value.explanation

    # The ORIGINAL TEXT survives into the message, so a typo stays legible.
    with pytest.raises(entry.Refusal) as refusal:
        entry.parse_fanout_grid("linear:2-e3:0")
    assert "2-e3" in refusal.value.explanation


def test_an_empty_grid_string_with_separators_is_refused_rather_than_run_as_zero_cells():
    """',,,' parses to nothing. Refusing beats an array job with no configuration in it."""
    with pytest.raises(entry.Refusal):
        entry.parse_fanout_grid(",,,")


def test_the_cell_seed_varies_init_and_holds_data_order_fixed(monkeypatch):
    """§5 STANDING RULE 2, AND A DELIBERATE DIVERGENCE FROM THE PRECEDENT.

    ``train_liv_arm.py:1347`` does ``opts.data_seed = opts.arm_seed`` -- init AND batch order
    move together. EXPERIMENT-PLAN §5 rule 2 asks for the opposite: "Paired seeds, same data
    order, different init. Report the paired difference." Same data order is what makes the
    cosine-vs-linear contrast paired, so the batch-order variance cancels in the difference
    rather than being added to it. E1 has none to spare: the predicted 0.025-nat gap is
    already below the n=3 MDE of 0.050.

    So: three cells of one arm must differ in --init-seed and agree in --data-seed.
    """
    resolved = []
    for index in range(18):
        opts, _ = entry.build_parser().parse_known_args(
            ["a-run-id", f"--fanout-grid={E1_GRID}", "--fanout-expect=18"]
        )
        entry.apply_fanout_cell(opts, str(index))
        resolved.append(opts)

    # Every cell keeps the SAME data order.
    assert {opts.data_seed for opts in resolved} == {0}

    # And the init seed varies, taking all three values within each of the six arms.
    arms: Dict[Any, List[int]] = {}
    for opts in resolved:
        arms.setdefault((opts.lr_schedule, opts.learning_rate), []).append(opts.init_seed)
    assert len(arms) == 6
    for arm, seeds in arms.items():
        assert sorted(seeds) == [0, 1, 2], arm

    # A --data-seed given on the command line is still not overwritten by the cell.
    opts, _ = entry.build_parser().parse_known_args(
        ["a-run-id", f"--fanout-grid={E1_GRID}", "--data-seed=77"]
    )
    entry.apply_fanout_cell(opts, "5")  # cosine:2e-3:2
    assert opts.init_seed == 2 and opts.data_seed == 77


def test_the_init_seed_reaches_the_generator_that_actually_initialises_weights():
    """THE SUBTLE ONE: a seed that reaches nothing is 3 identical runs per arm.

    There are two ``init_seed``s. ``ExperimentConfig.init_seed`` (default 12536) is what
    ``train`` hands to ``seed_all``, seeding the GLOBAL python/numpy/torch RNGs.
    ``TransformerConfig.init_seed`` (default 0) is what ``Transformer.init_weights`` turns
    into ``torch.Generator(device).manual_seed(seed)`` at model.py:294-299, and every
    ``init_*`` in nn/transformer/init.py draws from THAT generator.

    This asserts the distinction by building real models rather than by reading the code: if
    a future refactor made the global RNG the source again, or made --init-seed reach only
    ``ExperimentConfig``, this fails. The failure it prevents is silent -- three "seeds" that
    are one model reported as a tight seed distribution, under which the 6-point argmin is
    selected on a standard error that does not exist.
    """
    import torch
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.utils import seed_all

    def build(global_seed: int, config_init_seed: int):
        config = TransformerConfig.olmo2_190M(vocab_size=1024)
        config.init_seed = config_init_seed
        seed_all(global_seed)
        model = config.build(init_device="meta")
        model.init_weights(device=torch.device("cpu"))
        return {name: p.detach().clone() for name, p in model.named_parameters()}

    baseline = build(global_seed=0, config_init_seed=0)
    only_global_moved = build(global_seed=999, config_init_seed=0)
    only_config_moved = build(global_seed=0, config_init_seed=999)

    changed_by_global = [n for n in baseline if not torch.equal(baseline[n], only_global_moved[n])]
    changed_by_config = [n for n in baseline if not torch.equal(baseline[n], only_config_moved[n])]

    # seed_all reaches NOTHING. This is the assertion that names the trap.
    assert changed_by_global == [], (
        "seed_all() moved parameters, so the two seeds are no longer distinct and this "
        "test's premise -- and the --init-seed wiring -- need re-deriving"
    )
    # TransformerConfig.init_seed reaches most of the model. The rest are norm gains and
    # biases, which are constants by construction, so a magnitude is asserted rather than
    # "something changed".
    assert len(changed_by_config) >= 80
    assert len(changed_by_config) > 0.6 * len(baseline)
    # Named tensors, so a change confined to some corner of the model would not pass.
    assert any("embeddings" in n for n in changed_by_config)
    assert any("w_out" in n or "lm_head" in n for n in changed_by_config)
    assert any("attention" in n for n in changed_by_config)
    assert any("feed_forward" in n for n in changed_by_config)


def test_the_resolved_seed_reaches_the_built_model_config(monkeypatch):
    """The wiring end to end: a cell's seed must land on TransformerConfig.init_seed.

    Asserted on the config the real CLI path produces, so a flag rename or a dropped
    assignment in build_config breaks this rather than being discovered by 18 cells that
    trained six models three times each.
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

    seeds_seen = []
    for index in (0, 1, 2):
        opts, overrides = entry.build_parser().parse_known_args(
            [
                "a-run-id",
                "--dataset-id=pretrain/regmix-10b",
                "--dataset-version=v1",
                "--dataset-tokenizer=tokenizer/dolma2-bpe",
                "--save-folder=/tmp/x",
                f"--fanout-grid={E1_GRID}",
                "--fanout-expect=18",
            ]
        )
        entry.apply_fanout_cell(opts, str(index))
        config = entry.build_config(opts, overrides)
        seeds_seen.append(config.model.init_seed)
        # Batch order is identical in all three, which is what "paired" means.
        assert config.data_loader.seed == 0

    assert seeds_seen == [0, 1, 2]

    # And with no --init-seed the factory default is left alone, so a single run is unchanged.
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
        ]
    )
    assert opts.init_seed is None
    from olmo_core.nn.transformer import TransformerConfig

    untouched = TransformerConfig.olmo2_190M(vocab_size=100_352).init_seed
    assert entry.build_config(opts, overrides).model.init_seed == untouched


@pytest.mark.parametrize("schedule,lr", [(s, lr) for s in E1_SCHEDULES for lr in E1_LRS])
def test_each_arms_lr_reaches_the_scheduler_and_optimizer_at_the_right_magnitude(
    monkeypatch, schedule, lr
):
    """MAGNITUDE, NOT EXISTENCE, for all six arms.

    ``isinstance(sched, LinearWithWarmup)`` passes with alpha_f still at OLMo-core's default
    0.1 -- a "linear" schedule that stops at a tenth of peak, which is precisely the defect
    E1 exists to remove. So this reads the CURVE: linear must return EXACTLY 0.0 at the final
    step and cosine must return 0.1 x peak, and both must peak at the LR the cell asked for.

    Parametrised over all six arms rather than checked once, because an argmin over a
    6-point grid is only meaningful if all six points are the points they claim to be.
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
    index = E1_GRID.split(",").index(f"{schedule}:{lr}:0")
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/regmix-10b",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=/tmp/x",
            f"--fanout-grid={E1_GRID}",
            "--fanout-expect=18",
        ]
    )
    entry.apply_fanout_cell(opts, str(index))
    peak = float(lr)

    # The optimizer carries the cell's LR, not the flag default of 1.4e-3.
    config = entry.build_config(opts, overrides)
    assert config.train_module.optim.lr == peak
    assert config.train_module.optim.lr != 1.4e-3 or peak == 1.4e-3

    # And the scheduler the same CLI path builds carries the alpha_f the schedule implies.
    sched = entry.build_scheduler(opts)
    steps = entry.steps_for_tokens(E1_TARGET_TOKENS, E1_BATCH)
    warmup = round(steps * 0.1)

    assert sched.get_lr(peak, warmup, steps) == pytest.approx(peak)

    final = sched.get_lr(peak, steps, steps)
    if schedule == "linear":
        # EXACTLY zero. At alpha_f=0.1 this is 0.1*peak, and every isinstance check passes.
        assert final == 0.0
    else:
        assert final == pytest.approx(0.1 * peak)
        assert final > 0.0

    # The two schedules must not have collapsed onto one curve, which would make E1 measure
    # nothing while staying green.
    other = "cosine" if schedule == "linear" else "linear"
    opts.lr_schedule = other
    assert entry.build_scheduler(opts).get_lr(peak, steps, steps) != final


def test_the_six_arms_are_six_distinct_curves():
    """A 6-point argmin over fewer than 6 distinct configurations is not an argmin."""
    curves = {}
    for schedule in E1_SCHEDULES:
        for lr in E1_LRS:
            opts, _ = entry.build_parser().parse_known_args(
                ["a-run-id", f"--lr-schedule={schedule}"]
            )
            sched = entry.build_scheduler(opts)
            peak = float(lr)
            steps = entry.steps_for_tokens(E1_TARGET_TOKENS, E1_BATCH)
            curves[(schedule, lr)] = tuple(
                round(float(sched.get_lr(peak, at, steps)), 12)
                for at in (0, 1_000, 5_000, steps // 2, steps - 1, steps)
            )

    assert len(set(curves.values())) == 6, "two of the six arms are the same schedule"


def test_the_step_count_delivers_the_token_budget_the_experiment_declares():
    """The ARITHMETIC of the budget. Whether it reaches the run is the next test's job.

    Calls ``steps_for_tokens`` rather than re-deriving ``round(D/B)`` here. A test that
    recomputes the formula passes whichever way the source rounds and keeps passing after the
    source changes -- it tests its own copy of the arithmetic. This tests the program's.

    WHAT THIS TEST DOES NOT PROVE, RECORDED BECAUSE IT ONCE READ AS THOUGH IT DID. Calling
    ``steps_for_tokens`` shows the function computes the right number. It does NOT show the
    program uses it. For a while it did not: every caller was a test, ``--steps`` went
    verbatim into ``Duration.steps``, and this test was green beside a live defect. The
    ``through_the_real_option_parsing_path`` test below is the one that closes that, and it
    is the one to keep working if these ever have to be merged.
    """
    steps = entry.steps_for_tokens(E1_TARGET_TOKENS, E1_BATCH)

    assert steps == 11_444
    delivered = steps * E1_BATCH
    assert delivered == 2_999_975_936
    # Within 0.01% of the declared 3e9 budget. Stated as a tolerance because the step count
    # is an integer and 3e9/262,144 is not.
    assert abs(delivered / E1_TARGET_TOKENS - 1) < 1e-4

    # The batch is 128 sequences of 2048, asserted as the product so a typo'd digit fails.
    assert E1_BATCH == 128 * 2048 == 262_144

    # It really is derived: a different batch moves the steps to keep the budget.
    assert entry.steps_for_tokens(E1_TARGET_TOKENS, 2 * E1_BATCH) == 5_722
    assert abs(entry.steps_for_tokens(E1_TARGET_TOKENS, 786_432) * 786_432 / 3e9 - 1) < 1e-3
    # A non-positive batch is refused rather than dividing by zero.
    with pytest.raises(entry.Refusal):
        entry.steps_for_tokens(E1_TARGET_TOKENS, 0)


def test_the_token_budget_reaches_the_trainer_through_the_real_option_parsing_path(monkeypatch):
    """THE TEST THAT WOULD HAVE CAUGHT THE GUARD BEING DECORATIVE.

    ``steps_for_tokens`` was defined, documented and tested, and called from nowhere but
    tests. ``--steps`` went straight into ``Duration.steps``, so the failure its docstring
    claimed to prevent was live with a green test beside it. The defect was invisible to
    every test that called the function directly, because that is not how the program runs.

    So this asserts on ``config.trainer.max_duration`` -- the object the trainer actually
    stops on -- reached through ``build_parser().parse_known_args`` and ``build_config``, the
    same two calls ``main`` makes. If the derivation is ever unwired again, the value here
    falls back to the flag and this fails.
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

    def built(*flags: str):
        opts, overrides = entry.build_parser().parse_known_args(
            [
                "a-run-id",
                "--dataset-id=pretrain/regmix-10b",
                "--dataset-version=v1",
                "--dataset-tokenizer=tokenizer/dolma2-bpe",
                "--save-folder=/tmp/x",
                *flags,
            ]
        )
        return entry.build_config(opts, overrides)

    # THE E1 CELL. The budget and the batch are given; the length is derived from them.
    config = built("--target-tokens=3e9", f"--global-batch-size={E1_BATCH}")
    assert config.trainer.max_duration.value == 11_444
    assert str(config.trainer.max_duration.unit).endswith("steps")
    # The tokens the trainer will actually see, which is the number the experiment declares.
    assert config.trainer.max_duration.value * E1_BATCH == 2_999_975_936

    # DERIVED, NOT TYPED: the same budget at twice the batch halves the run rather than
    # training twice the tokens. This is what a hand-typed --steps cannot do.
    doubled = built("--target-tokens=3e9", f"--global-batch-size={2 * E1_BATCH}")
    assert doubled.trainer.max_duration.value == 5_722
    assert doubled.trainer.max_duration.value * 2 * E1_BATCH == 2_999_975_936

    # THE FIRST LIVE FAILURE: --global-batch-size omitted takes the flagship 786,432. With a
    # budget the run shortens to hold the tokens; without one it would have trained
    # 11,444 x 786,432 = 9.0B tokens, 3x the budget, at 3x the cost.
    flagship_default = built("--target-tokens=3e9")
    assert flagship_default.trainer.max_duration.value == 3_815
    assert abs(3_815 * 786_432 / 3e9 - 1) < 1e-3

    # THE SECOND LIVE FAILURE: no budget at all still means the pre-existing default of 200
    # steps -- 52.4M tokens, 1.75% of the budget -- and that behaviour is DELIBERATELY
    # preserved, because E0 must see an identical baseline. The guard is opt-in.
    assert built().trainer.max_duration.value == 200
    assert built("--steps=777").trainer.max_duration.value == 777


def test_a_hand_typed_step_count_that_contradicts_the_budget_is_refused(monkeypatch):
    """Two declarations of a run's length that disagree: one is wrong and neither wins.

    Overriding the explicit flag hides a typo; honouring it defeats the budget. Refusing is
    the only answer that cannot train one number and report the other.
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

    def build(*flags: str):
        opts, overrides = entry.build_parser().parse_known_args(
            [
                "a-run-id",
                "--dataset-id=pretrain/regmix-10b",
                "--dataset-version=v1",
                "--dataset-tokenizer=tokenizer/dolma2-bpe",
                "--save-folder=/tmp/x",
                *flags,
            ]
        )
        return entry.build_config(opts, overrides)

    # Off by one step against the budget.
    with pytest.raises(entry.Refusal) as refusal:
        build("--target-tokens=3e9", f"--global-batch-size={E1_BATCH}", "--steps=11445")
    assert "11445" in refusal.value.explanation and "11444" in refusal.value.explanation

    # The exact mistake the verifier traced: the E1 step count typed beside the FLAGSHIP
    # batch, which is 3x the budget and previously trained happily.
    with pytest.raises(entry.Refusal):
        build("--target-tokens=3e9", "--steps=11444")

    # A --steps that AGREES is allowed: being explicit is not the same as being wrong.
    agreeing = build("--target-tokens=3e9", f"--global-batch-size={E1_BATCH}", "--steps=11444")
    assert agreeing.trainer.max_duration.value == 11_444


def test_the_proxy_batch_sits_at_its_own_optimum_and_the_flagship_batch_does_not():
    """WHY 262,144 HERE AND 786,432 ON THE FLAGSHIP -- the same number, two different reasons.

    262,144 was the OLD flagship default and was wrong there: at D=40e9, B_opt = 0.721M, so
    262,144 is 0.36x optimum. `edullm/baseline-fix` raised it to 786,432 for exactly that
    reason, and this test must not read as a revert of that fix.

    At the PROXY's D=3e9 the optimum moves, because B_opt scales as D^0.383 and D is 13.3x
    smaller: B_opt = 267,171 tokens, so 262,144 is 0.98x optimum and the flagship's 786,432
    would be 2.94x. Power Lines Table 1 measures 4x-above-optimum at +0.029 nats -- LARGER
    than the 0.025-nat schedule effect E1 exists to measure. Running the proxy at the
    flagship batch would bury the measurement under a batch-position error bigger than its
    own signal, which is the error that got E8 cancelled (batch-size-verdict.md §E).

    The baseline default is asserted UNCHANGED here, because it belongs to the shared branch.

    CALLS ``entry.b_opt_tokens``, NOT A LOCAL COPY OF THE FIT. This test used to carry its own
    ``lambda tokens: 0.0306 * tokens**0.383 * 2048`` -- the exact scar the repo's memory file
    calls "test-must-call-not-recompute": a test that re-derives the code's formula passes
    whichever way the fit changes, because it is checking its own arithmetic rather than the
    program's. ``b_opt_tokens`` did not exist yet when this test was written; it does now, and
    the sequence length is pinned to 2048 here (rather than read off ``opts``) because this
    test is about the LAW's numbers at two values of D, not about any one cell's config.
    """
    b_opt = lambda tokens: entry.b_opt_tokens(tokens, 2048)  # noqa: E731

    # The law reproduces the flagship figure the baseline fix was justified on.
    assert b_opt(40e9) == pytest.approx(720_511, rel=1e-4)
    assert 786_432 / b_opt(40e9) == pytest.approx(1.09, abs=0.01)
    assert 262_144 / b_opt(40e9) == pytest.approx(0.36, abs=0.01)

    # And at the proxy's D the ranking inverts, which is the whole point.
    assert b_opt(3e9) == pytest.approx(267_171, rel=1e-4)
    assert E1_BATCH / b_opt(3e9) == pytest.approx(0.98, abs=0.01)
    assert 786_432 / b_opt(3e9) == pytest.approx(2.94, abs=0.01)

    # THE BASELINE DEFAULT IS NOT TOUCHED. E1 passes its batch on the command line; the
    # shared branch's 786,432 must still be what an unflagged run gets, or E0 and E1 would
    # disagree about the baseline.
    opts, _ = entry.build_parser().parse_known_args(["a-run-id"])
    assert opts.global_batch_size == 786_432
    assert opts.learning_rate == 1.4e-3


def test_the_batch_size_warning_fires_off_bopt_and_stays_silent_on_it(caplog):
    """BOTH DIRECTIONS: a guard that always fires is as useless as one that never does.

    ``resolve_steps`` warns rather than refuses (see its docstring) when
    ``--global-batch-size`` sits more than ``B_OPT_WARN_RATIO`` away from Power Lines' B_opt
    at the declared ``--target-tokens``. Leaving ``--global-batch-size`` at the flagship
    default of 786,432 while declaring E1's D=3e9 sits at 2.94x and must warn; E1's actual
    command line -- the same budget at 262,144 -- sits at 0.98x and must stay silent. A check
    that fires on both, or on neither, would be indistinguishable from no check at all.
    """
    import logging

    opts_default, _ = entry.build_parser().parse_known_args(["a-run-id", "--target-tokens=3e9"])
    with caplog.at_level(logging.WARNING):
        entry.resolve_steps(opts_default)
    fired = [r.message for r in caplog.records]
    # ACTIONABLE: the batch, the ratio and the B_opt figure all have to be readable off the
    # line without anyone re-deriving them, or the warning is a siren with no address on it.
    assert any("B_opt" in m for m in fired), fired
    assert any("786432" in m and "2.94" in m for m in fired), fired

    caplog.clear()
    opts_e1, _ = entry.build_parser().parse_known_args(
        ["a-run-id", "--target-tokens=3e9", f"--global-batch-size={E1_BATCH}"]
    )
    with caplog.at_level(logging.WARNING):
        entry.resolve_steps(opts_e1)
    assert caplog.records == []


def test_the_summary_records_bopt_ratio_and_tpp_for_every_cell(capsys):
    """docs/1b-leverage-audit/EXPERIMENT-PLAN.md:732 standing rule 8: "Record the TPP of every
    arm" -- nothing did until ``tpp`` was added here. ``b_opt_ratio`` is the same quantity
    ``resolve_steps`` warns on, recorded per cell so a reader does not have to refind a
    WARNING line 18 cells deep in one log group to tell an off-optimum cell from a schedule
    effect.
    """
    import json

    opts, _ = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--target-tokens=3e9",
            f"--global-batch-size={E1_BATCH}",
            "--save-folder=/tmp/x",
        ]
    )
    entry.summarise(
        opts=opts,
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1000)], step=11_444),
        losses=entry.LossWatcher(),
        seconds=1.0,
    )
    printed = json.loads(capsys.readouterr().out)

    assert printed["tokens_trained"] == 11_444 * E1_BATCH == 2_999_975_936
    assert printed["parameters"] == 1000
    assert printed["tpp"] == pytest.approx(2_999_975_936 / 1000)
    # Same ratio and tolerance as the law test above, not re-derived here.
    assert printed["b_opt_ratio"] == pytest.approx(0.98, abs=0.01)

    # NONE, NOT A CRASH, WHEN NO BUDGET WAS DECLARED. FakeOptions carries neither
    # --target-tokens nor --global-batch-size, which is what an E0-style call looks like --
    # and B_opt is a function of a budget that was never stated.
    entry.summarise(
        opts=FakeOptions(),
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1000)], step=11_444),
        losses=entry.LossWatcher(),
        seconds=1.0,
    )
    printed_no_budget = json.loads(capsys.readouterr().out)
    assert printed_no_budget["b_opt_ratio"] is None
    assert printed_no_budget["tpp"] is None


def test_the_proxy_warmup_is_about_a_thousand_steps():
    """--warmup-fraction 0.1 over the proxy's horizon, checked against Wen's tuned ~1000.

    The old default was 20 absolute steps. At the proxy that is 0.17% of the run.
    """
    steps = entry.steps_for_tokens(E1_TARGET_TOKENS, E1_BATCH)
    opts, _ = entry.build_parser().parse_known_args(["a-run-id"])
    sched = entry.build_scheduler(opts)

    assert opts.warmup_fraction == 0.1
    warmup = round(steps * opts.warmup_fraction)
    assert warmup == 1_144
    assert 1_000 <= warmup <= 1_300  # Wen et al.'s tuned grid uses ~1000

    # Read off the curve rather than the arithmetic: the peak is AT the warmup step and the
    # schedule is still climbing one step before it.
    assert sched.get_lr(1e-3, warmup, steps) == pytest.approx(1e-3)
    assert sched.get_lr(1e-3, warmup - 1, steps) < 1e-3
    assert sched.get_lr(1e-3, 20, steps) == pytest.approx(1e-3 * 20 / warmup)


def test_the_summary_says_which_cell_produced_the_loss(capsys):
    """18 cells print 18 summaries into one log group and the argmin joins them by hand.

    Without these fields the only route from a loss back to its configuration is the array
    index in the job name -- a join that can be got wrong on a sweep whose entire output is
    a ranking.
    """
    import json

    opts, _ = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            f"--fanout-grid={E1_GRID}",
            "--fanout-expect=18",
            "--save-folder=s3://bucket/x/",
        ]
    )
    entry.apply_fanout_cell(opts, "16")  # linear:2e-3:1

    watcher = entry.LossWatcher()
    watcher.log_metrics(1, {"train/CE loss": 11.5})
    watcher.log_metrics(2, {"train/CE loss": 3.1})
    entry.summarise(
        opts=opts,
        config=FakeConfig(),
        trainer=FakeTrainer([FakeParameter(1)], step=11_444),
        losses=watcher,
        seconds=1.0,
    )

    printed = json.loads(capsys.readouterr().out)
    assert printed["lr_schedule"] == "linear"
    assert printed["peak_lr"] == 2e-3
    assert printed["init_seed"] == 1
    assert printed["data_seed"] == 0


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


def _build_with_heldout(monkeypatch, tmp_path, val_paths, steps=200, extra_flags=()):
    """Build the real config with the held-out block live, faking only the S3 download.

    ``_download_to`` is replaced with something that writes bytes, and ``get_file_size`` with
    the size it writes, so ``_localised_heldout_paths`` runs its real logic -- the URL test,
    the cache path, the size comparison -- against a filesystem instead of a bucket.

    ``LOCAL_RANK`` is pinned to "0" so these build as the fetching process. Without it the test
    outcome would depend on whatever the ambient environment happens to carry, and a shell that
    exported LOCAL_RANK=1 would turn the download assertions into silent no-ops.

    ``steps=None`` OMITS ``--steps`` ENTIRELY, which is the only way to reach the E1 invocation
    from here. This helper used to hardcode ``--steps``, and that is precisely why the ladder
    could be scaled off the wrong length for two commits without a red test: with ``--steps``
    always present, ``resolve_steps`` returns ``opts.steps`` and the two lengths agree by
    construction, so the disagreement was unreachable. ``extra_flags`` is what lets a caller
    pass ``--target-tokens`` beside it.
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
            *([] if steps is None else [f"--steps={steps}"]),
            *extra_flags,
        ]
    )
    return opts, entry.build_config(opts, overrides)


def test_every_heldout_path_reaching_the_evaluator_is_local_and_is_the_sorted_order(
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

    SORTED ORDER, because the subset must be identical across arms and seeds. A per-cell subset
    makes the rungs incomparable, which is the one thing a ladder cannot tolerate. This fixture
    puts all six shards under ONE directory (domain "src"), which is the degenerate case of
    ``spread_across_sources``: with only one domain present, its round-robin has nothing to
    interleave against and collapses to a plain sort -- so this test still pins sorted order,
    just not via the old literal ``sorted(...)[:HELDOUT_SHARDS]`` slice. The case that actually
    exercises spreading across MULTIPLE domains is
    ``test_spread_across_sources_interleaves_domains_instead_of_exhausting_one`` below, with the
    naive-slice-collapses positive control in
    ``test_the_naive_sorted_prefix_this_replaces_would_have_collapsed_to_one_domain``.
    """
    from olmo_core.io import is_url

    # Deliberately UNSORTED, so the ordering is observable rather than accidental. Only six
    # shards, all under the same directory (domain "src"), so every one of them is still fewer
    # than the new HELDOUT_SHARDS=24 and nothing here gets truncated.
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

    assert entry.HELDOUT_SHARDS == 24
    assert len(paths) == min(entry.HELDOUT_SHARDS, len(val)) == 6
    for path in paths:
        assert not is_url(path), f"held-out paths must be local, got {path}"
    # All six, sorted -- the single-domain degenerate case of spread_across_sources. Basenames
    # rather than full paths, because the directory is the run's work dir. The cached name is
    # "<domain>--<basename>": selection now spans domains, so a bare basename would let two
    # domains' identically-numbered shards shadow each other in one cache dir.
    assert [Path(p).name for p in paths] == [
        "src--val-00000.u32le.bin",
        "src--val-00001.u32le.bin",
        "src--val-00002.u32le.bin",
        "src--val-00003.u32le.bin",
        "src--val-00004.u32le.bin",
        "src--val-00005.u32le.bin",
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
    assert [Path(p).name for p in paths] == [
        "src--val-00000.u32le.bin",
        "src--val-00001.u32le.bin",
    ]


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

    The label is the shard's TOPIC DOMAIN now (``_domain_of``), not the single literal
    "heldout-val" every path used to carry -- this fixture's six shards are all under one
    directory (domain "src"), so they still collapse to one label, which is exactly the
    single-domain case ``_domain_of``'s fallback comment describes. A fixture that spans
    multiple domains and gets multiple distinct labels back is
    ``test_selection_labels_span_every_domain_present_up_to_the_shard_limit`` below.
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
    assert len(metadata) == len(paths) == 6
    assert all("label" in m for m in metadata)
    assert {m["label"] for m in metadata} == {"src"}


def _domain_labelled(domain: str, shard: str) -> str:
    return f"s3://edullm-data/x/v1/tokens/all-dressed-snazzy2/{domain}/{shard}.u32le.bin"


def test_the_naive_sorted_prefix_this_replaces_would_have_collapsed_to_one_domain():
    """
    THE POSITIVE CONTROL: Defect A, reproduced on purpose, so the fix below has something to be
    better than.

    ``sorted(urls)[:N]`` sorts by the FULL url, so shards group by directory before they group
    by shard number: every path under one domain's directory sorts before every path under the
    next. Constructed here so ``adult_content`` (the real corpus's alphabetically-first domain,
    ``edullm-data/HANDOFF.md:467``) has MORE shards than the old ``HELDOUT_SHARDS = 4`` -- the
    exact condition ``spread_across_sources``'s docstring says the real corpus happened NOT to
    hit, and that made the naive slice look safer than it is.
    """
    urls = (
        [_domain_labelled("adult_content", f"val-{i:05d}") for i in range(6)]
        + [_domain_labelled("art_and_design", "val-00212")]
        + [_domain_labelled("crime_and_law", "val-00336")]
    )
    naive = sorted(urls)[:4]
    assert {entry._domain_of(u) for u in naive} == {"adult_content"}, (
        "the failure this replaces: four shards picked, one domain represented"
    )


def test_spread_across_sources_interleaves_domains_instead_of_exhausting_one():
    """
    THE FIX, ON THE SAME FIXTURE AS THE POSITIVE CONTROL ABOVE.

    Same six ``adult_content`` + one ``art_and_design`` + one ``crime_and_law`` urls, same
    limit of 4, run through ``spread_across_sources`` instead of a plain sort-and-slice. One
    shard from every domain before a second shard from any of them, so a limit smaller than the
    domain count still returns THAT MANY DISTINCT DOMAINS rather than a bigger sample of one.
    """
    urls = (
        [_domain_labelled("adult_content", f"val-{i:05d}") for i in range(6)]
        + [_domain_labelled("art_and_design", "val-00212")]
        + [_domain_labelled("crime_and_law", "val-00336")]
    )
    picked = entry.spread_across_sources(urls, 4)
    assert len(picked) == 4
    domains = [entry._domain_of(u) for u in picked]
    assert set(domains) == {"adult_content", "art_and_design", "crime_and_law"}, (
        "all three domains must be represented before adult_content gets a second shard"
    )
    # adult_content is the only domain with more than one shard, so with three domains and a
    # limit of four exactly one of them gets a second pick, and round-robin order puts it last.
    assert domains.count("adult_content") == 2
    assert domains.count("art_and_design") == 1
    assert domains.count("crime_and_law") == 1


def test_spread_across_sources_is_deterministic_regardless_of_input_order():
    """
    The ladder compares 18 E1 cells against each other; a selection that depended on
    ``corpus.val_paths``'s manifest order rather than its CONTENT would make the rungs
    incomparable across arms that happened to resolve their manifest differently.
    """
    urls = (
        [_domain_labelled("adult_content", f"val-{i:05d}") for i in range(3)]
        + [_domain_labelled("art_and_design", f"val-{i:05d}") for i in range(3)]
        + [_domain_labelled("crime_and_law", f"val-{i:05d}") for i in range(3)]
    )
    forward = entry.spread_across_sources(urls, 5)
    backward = entry.spread_across_sources(list(reversed(urls)), 5)
    shuffled = entry.spread_across_sources(
        [urls[i] for i in (4, 0, 8, 2, 6, 1, 7, 3, 5)], 5
    )
    assert forward == backward == shuffled


def test_spread_across_sources_returns_everything_when_asked_for_more_than_exists():
    urls = [
        _domain_labelled("adult_content", "val-00000"),
        _domain_labelled("art_and_design", "val-00212"),
    ]
    assert entry.spread_across_sources(urls, 100) == sorted(urls)


def test_spread_across_sources_on_a_single_directory_corpus_degenerates_to_a_plain_sort():
    """
    ``_domain_of`` falls back to the literal ``"heldout-val"`` for a path with no directory
    structure to read a domain from, so a flat corpus -- one directory, no topic split -- has
    exactly one group and this function must behave exactly as the old
    ``sorted(urls)[:limit]`` did for it. No corpus this repo trains on is actually this flat
    (edullm-data/HANDOFF.md:467), but the entry point does not import that fact, so the function
    must degrade safely rather than assume it.
    """
    urls = [f"val-{i:05d}.u32le.bin" for i in reversed(range(5))]
    assert entry.spread_across_sources(urls, 3) == sorted(urls)[:3]


def test_selection_labels_span_every_domain_present_up_to_the_shard_limit(monkeypatch, tmp_path):
    """
    THE INTEGRATION VERSION OF THE TWO UNIT TESTS ABOVE: run through ``build_config`` itself,
    not just ``spread_across_sources`` in isolation, so a mistake in HOW ``build_config`` calls
    it (wrong argument order, deriving labels from the localised path instead of the remote url,
    calling it on the wrong list) would show up here even if the function itself is correct.
    """
    val = (
        [_domain_labelled("adult_content", f"val-{i:05d}") for i in range(3)]
        + [_domain_labelled("art_and_design", "val-00212")]
        + [_domain_labelled("crime_and_law", "val-00336")]
    )
    _, config = _build_with_heldout(monkeypatch, tmp_path, val)
    eval_dataset = config.trainer.callbacks["lm_eval"].eval_dataset
    metadata = eval_dataset.metadata
    assert metadata is not None
    labels = [m["label"] for m in metadata]
    # HELDOUT_SHARDS=24 exceeds the 5 shards this fixture declares, so every shard is kept and
    # every one of its three domains must be labelled -- collapsing to fewer would mean the
    # labels were derived from something other than the per-url domain (e.g. a single shared
    # literal, which is the exact bug this whole fix removes).
    assert len(labels) == 5
    assert set(labels) == {"adult_content", "art_and_design", "crime_and_law"}
    assert labels.count("adult_content") == 3


def _build_tiny_padded_dataset(tmp_path: Path, domains: List[str]):
    """One tiny one-document shard per entry in ``domains``, each carrying that domain as its
    ``label`` metadata -- the exact ``NumpyPaddedFSLDataset``/metadata shape
    ``LMEvaluator.from_numpy_dataset`` builds its data loader from in production.
    """
    import numpy as np

    from olmo_core.data import NumpyPaddedFSLDataset

    paths = []
    for i, _domain in enumerate(domains):
        data = [10 + i, 11 + i, 0]  # one short document, ending in eos_token_id=0
        path = tmp_path / f"shard-{i}.npy"
        mmap = np.memmap(path, mode="w+", dtype=np.uint16, shape=(len(data),))
        mmap[:] = data
        mmap.flush()
        paths.append(path)

    ds = NumpyPaddedFSLDataset(
        *paths,
        sequence_length=8,
        pad_token_id=0,
        eos_token_id=0,
        vocab_size=32_000,
        metadata=[{"label": domain} for domain in domains],
    )
    ds.prepare()
    return ds


def _global_read_order(ds, tmp_path: Path):
    """The exact call sequence ``Evaluator.__iter__`` makes for a deterministic evaluator
    (eval/evaluator.py): ``reshuffle(epoch=1, in_memory=True)`` then read the global indices --
    seed 0 is ``LMEvaluator.from_numpy_dataset``'s default, never overridden in production.
    """
    from olmo_core.data import DataCollator, NumpyFSLDataLoader

    loader = NumpyFSLDataLoader(
        ds,
        global_batch_size=ds.sequence_length,
        collator=DataCollator(pad_token_id=0),
        work_dir=tmp_path,
        seed=0,
        dp_world_size=1,
        dp_rank=0,
        fs_local_rank=0,
    )
    loader.reshuffle(epoch=1, in_memory=True)
    return loader.get_global_indices()


def test_the_first_instances_in_read_order_span_multiple_domains_by_pigeonhole(tmp_path):
    """
    THE READ-ORDER REQUIREMENT ITSELF, DRIVEN THROUGH THE REAL SHUFFLE RATHER THAN ASSERTED
    AGAINST A KNOWN PERMUTATION.

    ``Evaluator.__iter__`` (eval/evaluator.py) calls ``self.batches.reset()`` then, for a
    deterministic evaluator (``LMEvaluatorCallbackConfig.deterministic`` defaults ``True``),
    ``self.batches.reshuffle(epoch=1, in_memory=True)`` -- a GLOBAL shuffle over every instance
    the padded dataset produces, not a walk in path order. ``NumpyFSLDataLoader._build_global_indices``
    (data/data_loader.py:667-679) is ``rng = get_rng(seed + epoch)`` over
    ``np.arange(len(dataset))``, so what exact permutation comes out depends on ``get_rng``'s
    internals -- which this test must NOT need to know, because nothing computational runs on
    this laptop to go check what a specific seed produces.

    Instead this uses a PIGEONHOLE argument: 3 domains, 2 shards (= 2 padded instances, one doc
    per shard) each, 6 instances total. No domain has 3 or more instances, so ANY 3-element
    prefix of ANY permutation of the 6 indices must draw from at least 2 domains. This holds for
    every possible shuffle output, which is what makes it safe to assert without executing the
    RNG to see what it actually produced.
    """
    domains = [
        "adult_content",
        "adult_content",
        "art_and_design",
        "art_and_design",
        "crime_and_law",
        "crime_and_law",
    ]
    ds = _build_tiny_padded_dataset(tmp_path, domains)
    assert len(ds) == 6, "one padded instance per shard, since each shard is one document"

    global_indices = _global_read_order(ds, tmp_path)
    assert sorted(int(i) for i in global_indices) == list(range(6)), "a shuffle, not a subset"

    prefix_labels = [ds[int(idx)]["metadata"]["label"] for idx in global_indices[:3]]
    assert len(set(prefix_labels)) >= 2, (
        "3 instances drawn from 3 domains of 2 each cannot all share one label, got "
        f"{prefix_labels}"
    )


def test_the_pigeonhole_check_above_would_fail_on_a_fixture_that_has_only_one_domain(tmp_path):
    """
    FALSIFIABILITY OF THE CHECK ABOVE, PINNED DIRECTLY -- the same real dataset/loader/shuffle
    machinery, the same seed and epoch, the same prefix length, with the ONE thing that matters
    changed: every shard now carries the same domain. If this test's ``>= 2 distinct domains``
    style of assertion could pass no matter what the fixture contained -- if it were vacuously
    true -- it would be worthless as a regression check on ``spread_across_sources``. Driving it
    through this negative fixture and getting exactly one domain back (by construction, for
    every possible permutation) is what proves the positive test above is actually discriminating
    and not just asserting something that was always going to be true.
    """
    domains = ["adult_content"] * 6
    ds = _build_tiny_padded_dataset(tmp_path, domains)

    global_indices = _global_read_order(ds, tmp_path)
    prefix_labels = [ds[int(idx)]["metadata"]["label"] for idx in global_indices[:3]]
    assert len(set(prefix_labels)) == 1, "single-domain fixture: every prefix is one domain"


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


def test_the_ladder_scales_off_the_derived_length_when_the_budget_sets_it(monkeypatch, tmp_path):
    """THE LADDER AND THE TRAINER MUST MEASURE THE SAME RUN.

    THE SEAM THIS GUARDS. E0 wired the ladder as ``ladder_steps(opts.steps)``, which was
    correct while ``opts.steps`` WAS the run's length. E1 then made the length derived, and
    ``max_duration`` moved to ``resolve_steps(opts)`` while the ladder did not. Neither commit
    is wrong alone; the combination left the two reading different lengths.

    WHAT THAT COSTS ON THE REAL INVOCATION. E1 passes ``--target-tokens 3e9
    --global-batch-size 262144`` and no ``--steps`` -- it cannot pass one, since a ``--steps``
    that disagrees with the budget is refused. ``opts.steps`` therefore keeps the parser default
    of 200, and it is never reassigned anywhere in the entry point. The trainer runs 11,444
    steps while the ladder fires at ``[10, 20, 40, 70, 100, 150]`` instead of
    ``[572, 1144, 2288, 4005, 5722, 8583]``. At ``--warmup-fraction 0.1`` warmup is 1,144 steps,
    so ALL SIX RUNGS LAND INSIDE WARMUP, the deepest at 1.31% of the run. Every rung scores a
    nearly-untrained model, the CE curve comes back flat, and a flat curve is exactly what the
    no-ladder warning further down exists to stop being mistaken for a result. It exits 0 and
    writes a plausible checkpoint.

    WHY NO EXISTING TEST CAUGHT IT. None combined ``--target-tokens`` with a val split.
    ``_build_with_heldout`` hardcoded ``--steps``, which makes ``resolve_steps`` return
    ``opts.steps`` so both lengths agree by construction; and the budget tests build through
    ``FakeManifest``, which has no ``val`` attribute, so no ladder is attached at all. The gap
    was the intersection of the two, and it is the intersection this test occupies.
    """
    # DERIVED, not typed: ask the code for the length rather than restating 11,444, so this
    # cannot drift if the rounding or the batch changes. The rungs below stay concrete.
    steps = entry.steps_for_tokens(E1_TARGET_TOKENS, E1_BATCH)

    opts, config = _build_with_heldout(
        monkeypatch,
        tmp_path,
        ["s3://edullm-data/x/v1/tokens/src/val-00000.u32le.bin"],
        # NO --steps, and a budget instead. This is the E1 command line.
        steps=None,
        extra_flags=(f"--target-tokens={E1_TARGET_TOKENS:g}", f"--global-batch-size={E1_BATCH}"),
    )
    callback = config.trainer.callbacks["lm_eval"]

    # The defaulted opts.steps is still sitting there, which is what made the bug invisible.
    assert opts.steps == entry.build_parser().get_default("steps") == 200

    # THE RUNGS AS INTEGERS, via the function the run itself calls.
    assert callback.fixed_steps == entry.ladder_steps(steps)
    assert callback.fixed_steps == [572, 1144, 2288, 4005, 5722, 8583]

    # AND NOT THE ONES THE DEFAULT WOULD HAVE PRODUCED.
    assert callback.fixed_steps != entry.ladder_steps(opts.steps)
    assert callback.fixed_steps != [10, 20, 40, 70, 100, 150]

    # THE PROPERTY THAT ACTUALLY MATTERS: the ladder must reach past warmup, or every rung
    # scores a model still on its way up and the curve carries no signal.
    warmup = round(steps * opts.warmup_fraction)
    assert warmup == 1_144
    assert max(callback.fixed_steps) > warmup
    # More than a token amount past it -- the deepest rung sits at 75% of the run.
    assert max(callback.fixed_steps) / steps > 0.5

    # The ladder and the trainer agree on the length, which is the invariant behind all of it.
    assert config.trainer.max_duration.value == steps
    assert max(callback.fixed_steps) < config.trainer.max_duration.value


def test_every_field_on_the_eval_callback_is_pinned_including_what_a_rung_costs(
    monkeypatch, tmp_path
):
    """
    THE COST KNOB, PINNED. ``eval_duration`` was the one field on this callback that nothing
    asserted, and it is the field whose default is expensive: ``LMEvaluatorCallbackConfig``
    defaults it to ``Duration.epochs(1)``
    (train/callbacks/evaluator_callback.py:227), which scores every shard IN FULL at every
    rung. With a six-rung ladder over the now-24 held-out shards that is the eval costing far
    more than the training it measures, and it fails no test and raises nothing -- the run is
    just slower and more expensive, which is invisible in the metrics.

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
    cached = tmp_path / "heldout-shards" / "x--val-00001.u32le.bin"
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
    assert out == [str(tmp_path / "heldout-shards" / "x--val-00001.u32le.bin")]

    # Local rank 0 on the same box still does the real, size-verified fetch. The shard is
    # pre-created at the WRONG size so the size branch is actually reached: the condition is
    # `not dest.is_file() or size != get_file_size(url)`, which short-circuits on an absent
    # file and would never issue a HEAD at all.
    monkeypatch.setenv("LOCAL_RANK", "0")
    cached = tmp_path / "heldout-shards" / "x--val-00001.u32le.bin"
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
