"""What ``.edullm/train_core6_arm.py`` computes for its held-out endpoint, and what it refuses.

The subject here is ``evaluate_val_aggregate`` and the accounting around it. That function is
the experiment's endpoint -- every contrast the study reports is a difference of it between two
arms -- and the version it replaced could not produce a number at all: it was gated on
``get_rank() == 0`` inside a swallowing ``except``, which under FSDP is a hang rather than an
exception, so the run exited 0 with a null field and a checkpoint nobody could use.

WHAT IS AND IS NOT TESTABLE ON A LAPTOP. The multi-rank behaviour is not: there is no GPU here
and no process group. What IS testable, and is what these tests cover, is everything that
decides whether the number is right once the collectives are correct -- the token accounting,
the refusals, the window arithmetic, and the fact that a rank-count of one is not a special
case in the code. The all-reduce path itself goes through ``all_reduce_value``, which is a
documented no-op off a process group, so the single-rank runs below execute the same lines an
eight-rank run does.

The entry point is not importable as a package -- it lives in ``.edullm/`` because that is what
the platform's image build copies and runs -- so it is loaded by path, the same way
``edullm_train_on_corpus_test.py`` loads its subject.
"""

import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_core6_arm.py"
    spec = importlib.util.spec_from_file_location("edullm_train_core6_arm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()

#: Deliberately not multiples of the sequence length below. A shard whose token count divides
#: evenly by ``seq_len`` hides every off-by-one in the window arithmetic, and the real corpus
#: does not have that property.
SHARD_TOKENS = (100, 71, 33)
SEQ_LEN = 32
VOCAB = 512


class TinyLM(torch.nn.Module):
    """A model small enough to run on a laptop that still produces real logits.

    Untrained, so its cross-entropy sits near ``ln(vocab)`` -- which is the magnitude assertion
    below, and the only check in this project's history that ever caught uninitialised weights.
    """

    def __init__(self, vocab: int = VOCAB, d: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.out = torch.nn.Linear(d, vocab)

    def forward(self, x):
        return self.out(self.embed(x))


@pytest.fixture
def shards(tmp_path) -> List[str]:
    """Three little-endian uint32 shards, under two different topic directories.

    THE TOPIC DIRECTORY IS PART OF THE FIXTURE, NOT SCENERY. In olmo-150b-dolma2 a val shard
    named ``val-00212.u32le.bin`` exists under 24 different topics, so a basename is not a
    unique key. Two of the three shards below share a basename for exactly that reason.
    """
    made = []
    for i, n in enumerate(SHARD_TOKENS):
        topic = "art_and_design" if i < 2 else "mathematics"
        directory = tmp_path / "corpus" / topic
        directory.mkdir(parents=True, exist_ok=True)
        # First and third share a name; second differs. Two topics, one repeated basename.
        name = "val-00212.u32le.bin" if i != 1 else "val-00213.u32le.bin"
        path = directory / name
        np.arange(n, dtype=np.uint32).astype("<u4").tofile(path)
        made.append(str(path))
    return made


#: "Caller did not say", as distinct from ``None``, which is a meaningful value here -- it is
#: the manifest declaring no row count, and the case one test below is entirely about.
_UNSET = object()


def evaluate(shards, tmp_path, *, declared=_UNSET, seq_len=SEQ_LEN, micro=2, model=None):
    return entry.evaluate_val_aggregate(
        model=model or TinyLM(),
        vocab_size=VOCAB,
        val_paths=list(shards),
        work_dir=str(tmp_path / f"wd-{len(os.listdir(tmp_path))}"),
        seq_len=seq_len,
        dtype=np.dtype("<u4"),
        declared_tokens=sum(SHARD_TOKENS) if declared is _UNSET else declared,
        micro=micro,
    )


# --- the endpoint produces a number, and it is the right size -------------------------------


def test_the_endpoint_produces_a_cross_entropy_near_ln_vocab(shards, tmp_path):
    """
    A MAGNITUDE CHECK, NOT AN EXISTENCE ONE. ``result["ce"] is not None`` passes for a model
    whose weights were never initialised, for a CE computed over the wrong tokens, and for a
    zero. An untrained model over any token set has to score near ``ln(vocab)`` -- that is what
    "uniform over the vocabulary" means -- and this project's documented history is five green
    harness results where the only check that caught anything was this one.
    """
    result = evaluate(shards, tmp_path)
    assert abs(result["ce"] - math.log(VOCAB)) < 1.0, (
        f"CE {result['ce']:.4f} is nowhere near ln({VOCAB}) = {math.log(VOCAB):.4f}; an "
        "untrained model cannot score this"
    )


def test_it_returns_sums_and_counts_and_not_only_a_mean(shards, tmp_path):
    """
    Two arms get differenced. A difference of two means computed over different token counts is
    not a paired difference, so the denominator has to ship with the number rather than be
    assumed equal between runs.
    """
    result = evaluate(shards, tmp_path)
    assert result["sum"] > 0 and result["tokens"] > 0
    assert result["ce"] == pytest.approx(result["sum"] / result["tokens"])


def test_every_token_is_accounted_for(shards, tmp_path):
    """
    ``present == declared`` exactly, and ``scored`` short of it by only the window remainder.

    The scored count is derived here from the window arithmetic rather than copied from the
    result, so this test computes the expected answer independently instead of restating what
    the code did.
    """
    result = evaluate(shards, tmp_path)

    expected_scored = sum(((n - 1) // SEQ_LEN) * SEQ_LEN for n in SHARD_TOKENS)
    assert result["tokens"] == expected_scored
    assert result["tokens_present"] == sum(SHARD_TOKENS)
    assert result["declared_tokens"] == sum(SHARD_TOKENS)
    assert result["unscored"] == sum(SHARD_TOKENS) - expected_scored
    assert result["shards"] == len(SHARD_TOKENS)

    entry.assert_val_tokens_account_for_the_corpus(result)


def test_two_shards_with_the_same_basename_are_both_counted(shards, tmp_path):
    """
    THE DOCUMENTED TRAP, AS A TEST. ``all-dressed-snazzy2__val-00212`` corresponds to
    ``all-dressed-snazzy2/art_and_design/val-00212.u32le.bin``: the topic directory is dropped
    from the name and 24 topics exist. Two of this fixture's shards share a basename, so a
    download that used the basename as its local filename would overwrite one with the other,
    silently drop a third of the val set, and produce a completely normal-looking CE.

    The token count is what catches it, which is the whole reason the count is asserted.
    """
    basenames = [os.path.basename(p) for p in shards]
    assert len(set(basenames)) < len(basenames), "the fixture must contain a repeated basename"

    result = evaluate(shards, tmp_path)
    assert result["tokens_present"] == sum(SHARD_TOKENS)
    entry.assert_val_tokens_account_for_the_corpus(result)


# --- the refusals, which are the point --------------------------------------------------------


def test_a_corpus_with_no_held_out_split_is_refused_rather_than_skipped(tmp_path):
    """
    Not ``return None``. A run that cannot produce the endpoint has produced a checkpoint that
    cannot answer the question, and reporting that as a null field beside exit 0 is what the
    whole of this function exists to stop.
    """
    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=[],
            work_dir=str(tmp_path),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=1000,
        )
    assert refusal.value.stage is entry.Stage.THE_CORPUS_DECLARES_NO_HELD_OUT_SPLIT
    assert int(refusal.value.stage) == 73


def test_a_missing_shard_is_caught_by_the_token_count(shards, tmp_path):
    """
    THE FAILURE THE COUNT ASSERTION IS FOR. Evaluating two of three shards produces a perfectly
    ordinary cross-entropy -- the model is the same, the tokens are real, the number is in
    range. Nothing about the CE says a third of the val set was never read. Only the count does.
    """
    partial = evaluate(shards[:2], tmp_path, declared=sum(SHARD_TOKENS))

    # The CE itself is unremarkable, which is precisely the problem being solved.
    assert abs(partial["ce"] - math.log(VOCAB)) < 1.0

    with pytest.raises(SystemExit) as refusal:
        entry.assert_val_tokens_account_for_the_corpus(partial)
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING
    assert "declares" in refusal.value.explanation


def test_a_corpus_that_declares_no_row_count_is_refused_rather_than_unchecked(shards, tmp_path):
    """
    An unchecked token count is how a CE over a quarter of the val set gets recorded as the
    endpoint. If there is nothing to check against, that is a refusal and not a pass.
    """
    result = evaluate(shards, tmp_path, declared=None)
    assert result["declared_tokens"] is None

    with pytest.raises(SystemExit) as refusal:
        entry.assert_val_tokens_account_for_the_corpus(result)
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING


def test_shards_too_short_to_fill_a_window_are_refused(tmp_path):
    """
    A sequence length larger than the shards yields zero windows, and zero windows is zero
    tokens. Returning a CE of 0.0 over 0 tokens would be a number, would serialise, and would
    be differenced against another arm as though it meant something.
    """
    directory = tmp_path / "tiny"
    directory.mkdir()
    path = directory / "val-00000.u32le.bin"
    np.arange(8, dtype=np.uint32).astype("<u4").tofile(path)

    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=[str(path)],
            work_dir=str(tmp_path / "wd"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=8,
        )
    assert refusal.value.stage is entry.Stage.THE_HELD_OUT_EVALUATION_SCORED_NOTHING


def test_a_shard_that_cannot_be_read_fails_rather_than_being_skipped(shards, tmp_path):
    """
    A fetch failure must not become a shorter val set. It is raised -- and in the distributed
    case raised on every rank, so a single rank's bad download does not leave its peers waiting
    on a collective it will never enter.
    """
    with pytest.raises(SystemExit) as refusal:
        entry.evaluate_val_aggregate(
            model=TinyLM(),
            vocab_size=VOCAB,
            val_paths=list(shards) + [str(tmp_path / "does-not-exist.u32le.bin")],
            work_dir=str(tmp_path / "wd"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=sum(SHARD_TOKENS),
        )
    # Whichever stage read_failure assigned, it must be a refusal that names the rank rather
    # than a silently shorter evaluation.
    assert "held-out" in refusal.value.explanation


# --- no rank gate in the compute path --------------------------------------------------------


def _function_ast(func) -> "object":
    """The parsed body of a function, with its docstring dropped.

    Parsed rather than grepped. The docstrings in this entry point QUOTE the defective code they
    replaced -- ``evaluate_val_aggregate`` explains the old ``if get_rank() == 0`` at length --
    so a substring search over the source matches the explanation and reports the bug as still
    present. Parsing looks at the statements, which is what the tests below are actually about.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    body = tree.body[0]
    assert isinstance(body, ast.FunctionDef)
    return ast.Module(body=body.body[1:], type_ignores=[])


def test_the_evaluator_has_no_rank_gate():
    """
    THE DEFECT, ASSERTED AGAINST THE PARSED SOURCE. The old endpoint was gated on ``get_rank()
    == 0``, which under FSDP has rank zero enter all-gathers no other rank reaches -- a hang,
    which the ``except Exception`` around it could not catch, so the run exited 0 with no CE.

    Inspecting the source is not the ideal check; running eight ranks would be, and there is no
    GPU here to do it on. What this catches is somebody reintroducing a rank gate into this
    function, which is a live risk given the function next to it still has one.
    """
    import ast

    module = _function_ast(entry.evaluate_val_aggregate)
    calls = [
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "get_rank" not in calls or not [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and isinstance(node.left.func, ast.Name)
        and node.left.func.id == "get_rank"
    ], "get_rank() is being compared to something, which is a rank gate"

    # And every rank must reach the reduction, which is what makes the number the whole run's.
    assert calls.count("all_reduce_value") >= 3, "the sums and counts must all be reduced"
    assert calls.count("barrier") >= 2, "the evaluation must be fenced on both sides"


def test_the_evaluator_does_not_swallow_its_own_failures():
    """
    A bare ``except Exception`` that logs and continues turns a broken endpoint back into a null
    field beside exit 0, which is the failure being fixed. Every handler in this function must
    lead to a raise -- the one that exists catches a per-rank fetch failure precisely so it can
    re-raise it on ALL ranks instead of hanging the others.
    """
    import ast

    module = _function_ast(entry.evaluate_val_aggregate)
    handlers = [n for n in ast.walk(module) if isinstance(n, ast.ExceptHandler)]
    raises = [n for n in ast.walk(module) if isinstance(n, ast.Raise)]
    assert raises, "the evaluator must be able to fail loudly"
    for handler in handlers:
        # Either the handler itself raises, or it records the failure for the collective raise
        # that follows it. Both keep the run from continuing with a partial endpoint; what is
        # forbidden is a handler after which the function can still return a result.
        assert any(
            isinstance(node, (ast.Raise, ast.Assign))
            for node in ast.walk(ast.Module(body=handler.body, type_ignores=[]))
        ), "an except that neither raises nor records the failure is a swallow"


def test_the_model_is_left_in_the_mode_it_was_found_in(shards, tmp_path):
    """
    ``model.eval()`` without a restore leaves dropout and norm statistics off for whatever runs
    next. Checked on the way out through a failure as well, since that is the path a ``finally``
    exists for and the easy one to write without.
    """
    model = TinyLM()
    model.train()
    evaluate(shards, tmp_path, model=model)
    assert model.training

    model.train()
    with pytest.raises(SystemExit):
        entry.evaluate_val_aggregate(
            model=model,
            vocab_size=VOCAB,
            val_paths=[],
            work_dir=str(tmp_path / "x"),
            seq_len=SEQ_LEN,
            dtype=np.dtype("<u4"),
            declared_tokens=1,
        )
    assert model.training


# --- the corpus's own val partition, resolved rather than reconstructed -----------------------


@dataclass
class ManifestWithSplits:
    """The shape ``edullm_data.read.dataset_paths`` returns for olmo-150b-dolma2-v1.

    ``val`` is a property over ``splits`` filtered by the reader's ``is_trainable``, which is
    what the entry point reads. The val URIs carry a TOPIC DIRECTORY that their filenames do
    not, which is the trap this shape exists to reproduce.
    """

    paths: List[str] = field(
        default_factory=lambda: [
            f"s3://edullm-data/pretrain/olmo-150b-dolma2/v1/t/train-{i:05}.u32le.bin"
            for i in range(4)
        ]
    )
    dtype: Optional[str] = "uint32"
    byte_order: Optional[str] = "little"
    header_bytes: int = 0
    rows: Optional[int] = 157_237_308_712
    splits: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "train": [
                f"s3://edullm-data/pretrain/olmo-150b-dolma2/v1/t/train-{i:05}.u32le.bin"
                for i in range(4)
            ],
            "val": [
                "s3://edullm-data/pretrain/olmo-150b-dolma2/v1/all-dressed-snazzy2/"
                f"{'art_and_design' if i % 2 else 'mathematics'}/val-{i:05}.u32le.bin"
                for i in range(60)
            ],
        }
    )
    split_rows: Dict[str, Optional[int]] = field(
        default_factory=lambda: {"train": 157_237_308_712, "val": 229_894_171}
    )

    @property
    def val(self):
        held = [p for name, ps in self.splits.items() if name != "train" for p in ps]
        return held or None


def resolve(manifest):
    return entry.corpus_from_manifest(
        manifest,
        dataset_id="pretrain/olmo-150b-dolma2",
        version="v1",
        tokenizer_id="tokenizer/dolma2-bpe",
    )


def test_the_val_partition_comes_from_the_manifest_with_its_topic_directory_intact():
    """
    The keys are the reader's, not rebuilt from filenames. ``val-00212.u32le.bin`` lives under
    24 topics; a key reconstructed from the name fetches a real, readable shard of the wrong
    topic, and every number downstream is plausible and wrong.
    """
    corpus = resolve(ManifestWithSplits())

    assert len(corpus.val_paths) == 60
    assert corpus.val_rows == 229_894_171
    # Every URI keeps the directory between the release prefix and the filename.
    for uri in corpus.val_paths:
        assert "/art_and_design/" in uri or "/mathematics/" in uri


def test_held_out_objects_never_appear_in_the_training_paths():
    """Training on the set the arm is scored against would make every contrast meaningless."""
    corpus = resolve(ManifestWithSplits())
    assert not set(corpus.val_paths) & set(corpus.paths)


def test_a_manifest_predating_the_split_api_degrades_to_no_val_rather_than_raising():
    """
    ``val`` and ``split_rows`` arrived in the reader after ``paths`` did. A corpus or a fake
    that predates them should report "no held-out split" -- which the endpoint then refuses
    loudly -- rather than raising an AttributeError in the middle of a config build.
    """

    @dataclass
    class Old:
        paths: List[str] = field(default_factory=lambda: ["s3://edullm-data/x/v1/a.u32le.bin"])
        dtype: Optional[str] = "uint32"
        byte_order: Optional[str] = "little"
        header_bytes: int = 0
        rows: Optional[int] = 1000

    corpus = entry.corpus_from_manifest(
        Old(), dataset_id="x", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )
    assert corpus.val_paths == []
    assert corpus.val_rows is None


def test_the_val_partition_reaches_the_config_the_run_evaluates_from(monkeypatch):
    """
    ``train()`` receives the config, not the ``Corpus``, so the endpoint can only run if the val
    objects were carried across. This is the join between the two halves of the fix.
    """
    monkeypatch.setattr(entry, "resolve_corpus", lambda **kwargs: resolve(ManifestWithSplits()))
    opts, overrides = entry.build_parser().parse_known_args(
        [
            "a-run-id",
            "--dataset-id=pretrain/olmo-150b-dolma2",
            "--dataset-version=v1",
            "--dataset-tokenizer=tokenizer/dolma2-bpe",
            "--save-folder=s3://outputs/teams/platform/runs/a-run-id/checkpoints/",
            "--arm=G2R0",
            "--steps=10",
        ]
    )
    config = entry.build_config(opts, overrides)

    assert len(config.val_paths) == 60
    assert config.val_rows == 229_894_171
    # And it survives serialisation, because that record is what lands beside the checkpoint.
    assert config.as_config_dict()["val_rows"] == 229_894_171


# --- the init seed, at the entry point --------------------------------------------------------


def test_the_init_seed_flag_reaches_the_model_config(monkeypatch):
    """
    THE OTHER HALF OF THE FIX-1 REGRESSION TEST. ``core6_arms_test`` proves the seed reaches the
    tensors; this proves the flag reaches ``build_arm``. Both are needed: the plumbing was
    correct on one side of that call and absent on the other, and each half passes on its own
    while ``--init-seed`` does nothing.
    """
    monkeypatch.setattr(entry, "resolve_corpus", lambda **kwargs: resolve(ManifestWithSplits()))

    def config_for(seed):
        opts, overrides = entry.build_parser().parse_known_args(
            [
                "a-run-id",
                "--dataset-id=pretrain/olmo-150b-dolma2",
                "--dataset-version=v1",
                "--dataset-tokenizer=tokenizer/dolma2-bpe",
                "--save-folder=/tmp/x",
                f"--init-seed={seed}",
            ]
        )
        return entry.build_config(opts, overrides)

    for seed in (0, 1, 12536):
        config = config_for(seed)
        assert config.model.init_seed == seed, (
            f"--init-seed {seed} did not reach the model config, so the weights are drawn from "
            f"{config.model.init_seed} while the summary reports {seed}"
        )
        # The summary reads config.init_seed, so the two must not be able to disagree.
        assert config.init_seed == config.model.init_seed
