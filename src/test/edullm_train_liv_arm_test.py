"""Tests for the LIV arm entry point, ``.edullm/train_liv_arm.py``.

Loaded by path rather than imported, the same way ``edullm_train_on_corpus_test.py`` loads its
subject: ``.edullm/`` is not a package and is deliberately not importable, so that a training
script cannot be picked up as library code.

These cover the two things the run-that-hung would have needed and did not have: a warning when
the terminal checkpoint is routed through the async path, and a model-selection path that reads
its vocabulary from the corpus instead of a constant.
"""

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Set

import pytest


def _load():
    path = Path(__file__).parent.parent.parent / ".edullm" / "train_liv_arm.py"
    spec = importlib.util.spec_from_file_location("edullm_train_liv_arm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


entry = _load()


@pytest.mark.parametrize(
    "steps,save_interval",
    [
        (20, 20),  # exactly the configuration that hung run_019fbfbe
        (20, 10),
        (20, 5),
        (20, 1),
        (762, 381),
    ],
)
def test_an_interval_that_divides_steps_is_warned_about(steps: int, save_interval: int):
    """
    A dividing interval hands the FINAL checkpoint to the async path, which stages the whole
    state dict to host RAM twice. That is what hung ``run_019fbfbe`` for 48 minutes with no
    traceback and no exit code.

    ``--save-interval 10`` is in this list deliberately: it looks like a fix and is not one,
    since 20 % 10 == 0 leaves the terminal save exactly where it was. Only an interval that
    does *not* divide ``steps`` moves the final save onto the synchronous ``post_train`` path.
    """
    warning = entry.warn_if_final_step_saves_async(steps, save_interval)
    assert warning is not None, f"steps={steps} save_interval={save_interval} should warn"
    assert "async" in warning
    assert str(steps) in warning


@pytest.mark.parametrize(
    "steps,save_interval",
    [
        (20, 25),  # the recommended shape: post_train takes the final save, synchronously
        (20, 1000),  # what OLMo-core's own integration tests use
        (762, 200),
        (100, 30),
    ],
)
def test_a_non_dividing_interval_is_not_warned_about(steps: int, save_interval: int):
    """The guard must stay quiet on the safe shape, or it is noise nobody reads."""
    assert entry.warn_if_final_step_saves_async(steps, save_interval) is None


def test_the_guard_does_not_divide_by_zero():
    """``--save-interval 0`` is nonsense but must not take the process down here."""
    assert entry.warn_if_final_step_saves_async(20, 0) is None
    assert entry.warn_if_final_step_saves_async(20, -1) is None


def test_compile_model_is_a_flag_rather_than_a_constant():
    """
    The first submitted run overrode ``compile_model`` to false on the command line while the
    file still read ``True``. A default that disagrees with what ran sends the next reader of
    this file down a wrong path, so the value is now reachable from the CLI.
    """
    parser = entry.build_parser()
    assert parser.parse_args([]).compile_model is True
    assert parser.parse_args(["--no-compile-model"]).compile_model is False


def test_the_arm_flag_offers_every_declared_arm():
    """
    ``--arm`` must be closed over the declared set. An unknown arm should fail at parse time
    rather than building some default model and reporting it under the wrong name.
    """
    from olmo_core.nn.transformer.liv_arms import ARMS

    parser = entry.build_parser()
    assert parser.parse_args([]).arm == "L0"
    for name in ARMS:
        assert parser.parse_args(["--arm", name]).arm == name
    with pytest.raises(SystemExit):
        parser.parse_args(["--arm", "not-an-arm"])


def test_arm_seed_and_data_seed_are_separately_settable():
    """
    Pairing requires init and data order to share a seed, but they are distinct flags so that
    an unpaired control can be run deliberately rather than by accident.
    """
    opts = entry.build_parser().parse_args(["--arm-seed", "3", "--data-seed", "3"])
    assert opts.arm_seed == 3 and opts.data_seed == 3


def test_the_defaults_are_the_studys_frozen_geometry():
    """
    Sequence length and learning rate are part of the claim, not tuning knobs: the L0-vs-A16-P
    FLOPs gap is context-dependent (1.22x at 4K, 1.91x at 32K), so a default that drifts would
    quietly change what the arms mean.

    The global batch size is here for a sharper reason: a submitted draft passed 131,072 while
    this file, this test and the docstring all said 524,288. Both the 3e-4 learning rate and the
    0.0105-nat noise floor that sets the seed count were calibrated at 524,288, so the override
    silently invalidated the arithmetic justifying the design.
    """
    opts = entry.build_parser().parse_args([])
    assert opts.sequence_length == 4096
    assert opts.learning_rate == pytest.approx(3e-4)
    assert opts.global_batch_size == 128 * 4096


def test_the_default_step_and_save_interval_do_not_trip_the_async_guard():
    """
    The shipped defaults must be a submittable pair, not a pair the startup guard rejects.

    762 % 200 = 162, so the terminal checkpoint takes the synchronous path. A default pair that
    divided would mean every run started by copy-pasting the help text inherited the checkpoint
    hang that cost run_019fbfbe 48 minutes.
    """
    opts = entry.build_parser().parse_args([])
    assert opts.steps == 762
    assert opts.save_interval == 200
    assert entry.warn_if_final_step_saves_async(opts.steps, opts.save_interval) is None
    # And the budget the default actually buys, which is what makes 762 the right number.
    assert opts.steps * opts.global_batch_size == pytest.approx(399_507_456)


def test_val_paths_defaults_to_empty_so_a_corpus_without_a_split_still_builds():
    """
    ``val_paths`` is optional: a corpus with no held-out split must still construct, because
    the absence is a property of the corpus rather than an error in the submission.
    """
    corpus = entry.Corpus(
        dataset_id="x",
        version="v1",
        paths=["s3://b/train-0.bin"],
        dtype=entry.NumpyDatasetDType.uint32,
        tokenizer=entry.TOKENIZERS["tokenizer/dolma2-bpe"](),
        rows=None,
    )
    assert corpus.val_paths == []


def test_the_ladder_is_attached_only_when_the_corpus_has_a_val_split():
    """
    The ladder is conditional on ``corpus.val_paths``, so assert the CONDITION as the code
    computes it, over both branches.

    NOT a test of ``build_config``, and that gap is deliberate rather than hidden: reaching the
    callback would need a built model, a resolved corpus and a trainer config, which is an
    integration test this suite has no fixture for. What is asserted here is the predicate the
    branch turns on. A previous attempt at this test re-emitted the warning string itself and
    asserted on the copy, which would have stayed green with the warning deleted -- a tautology
    dressed as coverage.
    """
    with_val = entry.Corpus(
        dataset_id="d",
        version="v1",
        paths=["s3://b/train-0.bin"],
        dtype=entry.NumpyDatasetDType.uint32,
        tokenizer=entry.TOKENIZERS["tokenizer/dolma2-bpe"](),
        rows=None,
        val_paths=["s3://b/val-0.bin"],
    )
    without_val = replace(with_val, val_paths=[])
    assert bool(with_val.val_paths) is True
    assert bool(without_val.val_paths) is False


@pytest.mark.parametrize(
    "steps,expected",
    [
        (762, [38, 76, 152, 266, 381, 571]),  # the pilot's per-cell budget
        (381, [19, 38, 76, 133, 190, 285]),  # the LR probe
    ],
)
def test_the_ladder_rungs_are_geometric_and_never_below_step_two(steps, expected):
    """
    Pin the rungs the entry point computes, because ``int()`` TRUNCATES: 762*0.35 = 266.7 -> 266
    and 762*0.75 = 571.5 -> 571, not 267 and 572. A doc or analysis script that rounds instead
    would look for rungs that were never evaluated.

    Calls ``entry.ladder_steps`` -- the function ``build_config`` itself calls -- rather than
    re-deriving the set comprehension. An earlier version of this test recomputed the rungs
    locally and therefore passed unchanged when the real fractions were mutated: it pinned its
    own copy of the arithmetic, not the run's.
    """
    rungs = entry.ladder_steps(steps)
    assert rungs == expected
    # post_step returns early for step <= 1, so a rung there would silently never fire.
    assert min(rungs) >= 2
    # 1.0 is deliberately absent -- eval_on_finish covers the final step, and listing it too
    # would score the same model twice.
    assert steps not in rungs


def test_preparing_heldout_indices_warns_rather_than_raising_without_a_ladder():
    """
    A corpus with no val split attaches no ``lm_eval`` callback. Preparing must then be a
    no-op, not a KeyError -- the absence is a property of the corpus, not an error.
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


def test_preparing_heldout_indices_calls_prepare_on_the_eval_dataset():
    """The one thing this command exists to do."""
    from unittest import mock

    dataset = mock.Mock(paths=["s3://b/val-0.bin"])
    cfg = mock.Mock(trainer=mock.Mock(callbacks={"lm_eval": mock.Mock()}))
    cfg.trainer.callbacks["lm_eval"].eval_dataset.build.return_value = dataset
    entry.prepare_heldout_indices(cfg)
    dataset.prepare.assert_called_once()


def test_train_never_touches_the_padded_prepare():
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
    # mistaken for the call. An earlier version of this test matched raw source and failed on
    # exactly that -- the prose above the branch mentions the function before the branch runs.
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


def test_a_run_short_enough_to_collapse_the_ladder_still_yields_valid_rungs():
    """
    The floor and the de-duplication have to hold on a short run too. At 20 steps the low
    fractions all truncate into 1-4, so without ``max(2, ...)`` the ladder would ask for step 1
    and quietly lose a rung, and without the set it would ask for the same step twice.
    """
    rungs = entry.ladder_steps(20)
    assert rungs == sorted(set(rungs)), "rungs must be unique"
    assert min(rungs) >= 2
    assert 20 not in rungs
    assert len(rungs) <= len(entry.LADDER_FRACTIONS)


class _StubRead:
    """The duck-typed shape ``corpus_from_manifest`` documents: paths/dtype/byte_order/rows."""

    def __init__(self, paths, val=None):
        self.paths = paths
        self.val = val
        self.dtype = "uint32"
        self.byte_order = sys.byteorder
        self.header_bytes = 0
        self.rows = None


def test_held_out_paths_are_carried_through_from_the_reader():
    """The ladder cannot exist unless the val URIs survive the manifest -> Corpus hop."""
    read = _StubRead(["s3://b/train-0.bin"], val=["s3://b/val-0.bin", "s3://b/val-1.bin"])
    corpus = entry.corpus_from_manifest(
        read, dataset_id="d", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )
    assert corpus.val_paths == ["s3://b/val-0.bin", "s3://b/val-1.bin"]
    assert corpus.paths == ["s3://b/train-0.bin"]


def test_a_val_shard_that_is_also_a_train_shard_is_refused():
    """
    A held-out shard that was also trained on is not held out, and the failure is invisible:
    the eval number merely looks better than it is. This project has already shipped a
    contaminated held-out set once, with val files sitting inside the training path list.
    """
    shared = "s3://b/shard-7.bin"
    read = _StubRead(["s3://b/train-0.bin", shared], val=[shared])
    with pytest.raises(entry.Refusal) as excinfo:
        entry.corpus_from_manifest(
            read, dataset_id="d", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
        )
    # Assert the STAGE, not just that something refused. An earlier draft of this test passed a
    # tokenizer id that does not exist, so it raised Refusal for an unrelated reason and was
    # green while the overlap guard went entirely untested. `pytest.raises(SomeError)` on a
    # function with several refusal paths is a check that the function failed, not a check that
    # it failed for the reason under test.
    assert excinfo.value.stage is entry.Stage.THE_MANIFEST_IS_NOT_SAFE_TO_MEMMAP
    assert shared in str(excinfo.value)


def test_the_reader_returning_none_for_val_is_not_an_error():
    """``ResolvedSplit.val`` is None (not []) when a dataset declares no held-out split."""
    corpus = entry.corpus_from_manifest(
        _StubRead(["s3://b/train-0.bin"], val=None),
        dataset_id="d",
        version="v1",
        tokenizer_id="tokenizer/dolma2-bpe",
    )
    assert corpus.val_paths == []


def test_a_reader_without_a_val_attribute_still_works():
    """
    ``corpus_from_manifest`` promises anything carrying paths/dtype/byte_order/header_bytes/rows
    will do. Older stubs predate ``val``, and adding a required attribute would break that
    contract rather than extend it.
    """

    class Older:
        paths = ["s3://b/train-0.bin"]
        dtype = "uint32"
        byte_order = sys.byteorder
        header_bytes = 0
        rows = None

    corpus = entry.corpus_from_manifest(
        Older(), dataset_id="d", version="v1", tokenizer_id="tokenizer/dolma2-bpe"
    )
    assert corpus.val_paths == []


def test_an_impossible_first_loss_is_refused():
    """
    A cell that starts from uninitialised weights scores in the hundreds instead of ln(vocab),
    trains happily, and produces a plausible curve from garbage. Observed twice in this project
    at 926 and ~900, because ``TransformerConfig.build()`` does not initialise.

    Asserting the MAGNITUDE is the point. Every harness bug shipped here passed an
    existence check and would have failed a size check.
    """
    watcher = entry.LossWatcher(expected_first_loss=math.log(100_352))
    with pytest.raises(entry.Refusal):
        watcher.log_metrics(1, {"train/CE loss": 926.0})


def test_a_plausible_first_loss_is_accepted():
    """ln(100,352) = 11.516. The guard must not fire on a healthy start."""
    watcher = entry.LossWatcher(expected_first_loss=math.log(100_352))
    watcher.log_metrics(1, {"train/CE loss": 11.52})
    assert watcher.first == pytest.approx(11.52)
    # Later steps are unconstrained: the check is about where training STARTED.
    watcher.log_metrics(2, {"train/CE loss": 4.0})
    assert watcher.last == pytest.approx(4.0)


def test_the_first_loss_guard_is_inert_when_not_armed():
    """
    A resumed attempt's first recorded loss is wherever the previous one stopped, so the run
    leaves ``expected_first_loss`` unset when ``global_step > 0``. Armed unconditionally, the
    check would kill every retry at its first metric.
    """
    watcher = entry.LossWatcher()
    watcher.log_metrics(1, {"train/CE loss": 4.2})
    assert watcher.first == pytest.approx(4.2)


def test_the_first_loss_guard_tolerates_a_missing_metric():
    """A metrics dict without the loss key must not arm or trip anything."""
    watcher = entry.LossWatcher(expected_first_loss=math.log(100_352))
    watcher.log_metrics(1, {"throughput/device/TPS": 1234.0})
    assert watcher.first is None


GRID_4x3 = (
    "L0:0,L0:1,L0:2,"
    "F-r128:0,F-r128:1,F-r128:2,"
    "G-grouped:0,G-grouped:1,G-grouped:2,"
    "N-narrow:0,N-narrow:1,N-narrow:2"
)


def test_the_fanout_grid_covers_every_cell_exactly_once():
    """
    A 4-arm x 3-seed pilot is 12 distinct cells. A duplicate would spend a cell's budget
    twice and leave a hole in the grid, and both loss curves would look entirely plausible.
    """
    cells = entry.parse_fanout_grid(GRID_4x3)
    assert len(cells) == 12
    assert len(set(cells)) == 12
    assert {arm for arm, _ in cells} == {"L0", "F-r128", "G-grouped", "N-narrow"}
    for arm in ("L0", "F-r128", "G-grouped", "N-narrow"):
        assert sorted(s for a, s in cells if a == arm) == [0, 1, 2]


def test_each_array_index_selects_its_own_cell():
    """
    The whole point: ``fanout_index_parameter`` on the form is documentation and substitutes
    nothing. Batch sets AWS_BATCH_JOB_ARRAY_INDEX and the program must read it, or all twelve
    cells train the same arm at twelve times the price.
    """
    seen = [entry.resolve_fanout_cell(GRID_4x3, str(i)) for i in range(12)]
    assert seen == entry.parse_fanout_grid(GRID_4x3)
    assert len(set(seen)) == 12


def test_a_seed_pairs_init_and_data_order():
    """Cells of one arm differ only in seed; cells of one seed share it across arms."""
    cells = dict(enumerate(entry.parse_fanout_grid(GRID_4x3)))
    by_seed: Dict[int, Set[str]] = {}
    for arm, seed in cells.values():
        by_seed.setdefault(seed, set()).add(arm)
    for seed, arms in by_seed.items():
        assert arms == {"L0", "F-r128", "G-grouped", "N-narrow"}, seed


def test_no_grid_means_no_change():
    """A single run must be unaffected by the fan-out machinery."""
    assert entry.resolve_fanout_cell("", None) is None
    assert entry.resolve_fanout_cell("", "3") is None


def test_a_grid_without_an_index_is_refused():
    """
    Refusing beats defaulting to cell 0: a grid submitted without the fan-out fields would
    otherwise run one arm N times and be indistinguishable from a completed sweep.
    """
    with pytest.raises(entry.Refusal):
        entry.resolve_fanout_cell(GRID_4x3, None)


def test_an_index_past_the_end_of_the_grid_is_refused():
    """fanout_size and the grid must agree, or trailing cells vanish silently."""
    with pytest.raises(entry.Refusal):
        entry.resolve_fanout_cell(GRID_4x3, "12")


def test_a_malformed_or_unknown_cell_is_refused():
    for bad in ("L0", "L0:x", "NotAnArm:0", "L0:0,NotAnArm:1"):
        with pytest.raises(entry.Refusal):
            entry.parse_fanout_grid(bad)
