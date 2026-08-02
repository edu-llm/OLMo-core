"""Tests for the LIV arm entry point, ``.edullm/train_liv_arm.py``.

Loaded by path rather than imported, the same way ``edullm_train_on_corpus_test.py`` loads its
subject: ``.edullm/`` is not a package and is deliberately not importable, so that a training
script cannot be picked up as library code.

These cover the two things the run-that-hung would have needed and did not have: a warning when
the terminal checkpoint is routed through the async path, and a model-selection path that reads
its vocabulary from the corpus instead of a constant.
"""

import importlib.util
import sys
from pathlib import Path

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
    """
    opts = entry.build_parser().parse_args([])
    assert opts.sequence_length == 4096
    assert opts.learning_rate == pytest.approx(3e-4)
    assert opts.global_batch_size == 128 * 4096


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
    by_seed = {}
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
