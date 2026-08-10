"""Tests for the per-cell tranche watcher.

The watcher's one job that nothing else does is to distinguish the cells of a fan-out, so these
mostly plant a fan-out with something wrong in one cell and check the render says so. The arm
identification is the part that changed: it used to be whatever the caller typed, and is now
read from each cell's own logged config, with the caller's label surviving only as a fallback
for a cell too young to have written one.
"""

import re

import pytest
import tranche_watch
from tranche_watch import CELL_CRASH_SUFFIX, CELL_SUFFIX, CellProgress, render


def cell(
    index,
    *,
    arms=("baseline",),
    label=None,
    step=1000,
    state="running",
    seed=None,
    recovered=False,
    died_with=None,
):
    return CellProgress(
        index=index,
        state=state,
        step=step,
        seed=index if seed is None else seed,
        arms=tuple(arms),
        labelled_arm=label,
        summary_lost_its_step=recovered,
        died_with=died_with,
    )


class TestCellIdParsing:
    @pytest.mark.parametrize("index", [0, 1, 4, 19])
    def test_a_cell_id_yields_its_submission_and_index(self, index):
        got = CELL_SUFFIX.match(f"run_019fe2f4-f528-70a8-9242-d22f358ede0a-cell-{index}")
        assert got is not None
        assert got.group("submission") == "run_019fe2f4-f528-70a8-9242-d22f358ede0a"
        assert int(got.group("index")) == index

    def test_a_run_that_is_not_a_fan_out_cell_is_not_matched(self):
        assert CELL_SUFFIX.match("run_019fe2f4-f528-70a8-9242-d22f358ede0a") is None

    def test_the_display_name_is_not_what_is_matched(self):
        """
        THE NAME IS NEITHER UNIQUE NOR STABLE. Every cell of the A100 submission is called
        ``run_019fe2f4-...`` with no suffix, and the three L40S cells were all renamed to
        ``...-died`` after they were cancelled. Only the id carries the cell index.
        """
        assert CELL_SUFFIX.match("run_019fe279-4ef0-7035-9432-4e24d23fba97-died") is None

    def test_a_crash_report_is_read_as_a_report_and_never_as_a_cell(self):
        """
        The report is filed at the cell's id with ``-died`` appended, so it still says which
        cell it is about -- and it must not be counted as one, or a five-cell fan-out with two
        deaths reports seven cells.
        """
        report = "run_019fe7bc-53f3-7081-8306-42fdfc376459-cell-0-died"
        assert CELL_SUFFIX.match(report) is None
        got = CELL_CRASH_SUFFIX.match(report)
        assert got is not None
        assert got.group("submission") == "run_019fe7bc-53f3-7081-8306-42fdfc376459"
        assert int(got.group("index")) == 0

    def test_a_live_cell_is_not_read_as_a_crash_report(self):
        assert CELL_CRASH_SUFFIX.match("run_019fe2f4-f528-70a8-9242-d22f358ede0a-cell-0") is None


class TestArmIdentification:
    def test_the_config_wins_over_the_label(self):
        assert cell(0, arms=("faithful",), label="baseline").arm == "faithful"

    def test_the_label_is_used_only_when_the_run_has_said_nothing(self):
        young = cell(0, arms=(), label="baseline")
        assert young.arm == "baseline?"
        assert young.arm_is_claimed

    def test_a_cell_with_neither_says_so(self):
        assert cell(0, arms=(), label=None).arm == "unknown"

    def test_a_config_that_rules_out_the_label_is_a_contradiction(self):
        """
        The failure the whole reversal is for: a cell that resolved to an arm nobody meant. A
        label taken from the command line agrees with itself and can never catch this.
        """
        assert cell(0, arms=("baseline",), label="faithful").contradicts_label

    def test_a_config_that_agrees_with_the_label_is_not(self):
        assert not cell(0, arms=("baseline",), label="baseline").contradicts_label

    def test_an_unlabelled_submission_cannot_contradict_anything(self):
        assert not cell(0, arms=("baseline",), label=None).contradicts_label

    def test_an_ambiguous_config_that_still_contains_the_label_agrees(self):
        """``faithful`` and ``decay-everything`` differ in the optimizer alone."""
        both = cell(0, arms=("faithful", "decay-everything"), label="faithful")
        assert not both.contradicts_label
        assert both.arm == "faithful or decay-everything"


class TestRender:
    def test_every_cell_gets_its_own_line(self):
        cells = [cell(i) for i in range(5)]
        lines = render("run_x", cells, 5, 6000).splitlines()
        assert "5/5 cells reporting" in lines[0]
        for i in range(5):
            assert any(f"cell {i} seed {i}" in line for line in lines[1:])

    def test_the_cells_that_never_started_are_named_as_a_number(self):
        got = render("run_x", [cell(i) for i in range(3)], 5, 6000)
        assert "3/5 cells reporting" in got
        assert "2 cell(s) have logged nothing" in got

    def test_a_staggered_fan_out_reports_its_spread(self):
        """
        What this watcher exists for. The A100 submission's last two cells started hundreds of
        steps behind the first, and nothing in ``edullm status`` says so.
        """
        cells = [cell(0, step=1089), cell(1, step=1069), cell(2, step=569)]
        assert "spread across cells: 520 steps" in render("run_x", cells, 3, 6000)

    def test_a_single_cell_reports_no_spread(self):
        assert "spread across cells" not in render("run_x", [cell(0)], 1, 6000)

    def test_an_arm_mismatch_is_shouted_about(self):
        cells = [cell(i, arms=("baseline",), label="faithful") for i in range(5)]
        got = render("run_x", cells, 5, 6000)
        assert "ARM MISMATCH on cell(s) 0, 1, 2, 3, 4" in got

    def test_only_the_mismatched_cells_are_named(self):
        cells = [cell(i, arms=("faithful",), label="faithful") for i in range(4)]
        cells.append(cell(4, arms=("baseline",), label="faithful"))
        got = render("run_x", cells, 5, 6000)
        assert "ARM MISMATCH on cell(s) 4" in got

    def test_a_healthy_submission_says_nothing_alarming(self):
        got = render("run_x", [cell(i, label="baseline") for i in range(5)], 5, 6000)
        assert "MISMATCH" not in got
        assert "logged nothing" not in got
        assert "?" not in got

    def test_a_cell_running_on_a_supplied_label_is_flagged_as_such(self):
        cells = [cell(0, arms=(), label="baseline"), cell(1, arms=("baseline",), label="baseline")]
        got = render("run_x", cells, 2, 6000)
        assert "1 cell(s) marked '?'" in got

    def test_a_cell_that_has_logged_no_step_does_not_break_the_bar(self):
        got = render("run_x", [cell(0, step=-1)], 1, 6000)
        assert "step     -1/6000" in got
        assert "0.0%" in got

    @pytest.mark.parametrize("step,expected", [(0, 0.0), (3000, 50.0), (6000, 100.0)])
    def test_the_percentage_is_the_step_over_the_horizon(self, step, expected):
        got = render("run_x", [cell(0, step=step)], 1, 6000)
        assert re.search(rf"{expected:5.1f}%", got)

    def test_a_step_recovered_from_history_is_shown_and_said_to_be_recovered(self):
        """
        The number on its own would be a quiet correction. What a reader needs is that this
        cell's summary is not evidence about anything -- it was overwritten, and reading
        ``step None`` off it is how a cell at 4,910 was reported as one that never started.
        """
        got = render("run_x", [cell(0, step=4910, recovered=True)], 5, 6000)
        assert "step   4910/6000" in got
        assert "[step from history]" in got
        assert "cell 0 read step None from their summary" in got

    def test_an_ordinary_cell_says_nothing_about_history(self):
        got = render("run_x", [cell(0, step=4910)], 5, 6000)
        assert "history" not in got

    def test_a_cell_that_left_only_a_crash_report_is_kept_in_cell_position(self):
        """
        It has no run of its own, and "logged nothing" is what a cell still queuing for
        capacity says. A dead cell and a slow one are the distinction this watcher exists for.
        """
        got = render(
            "run_x",
            [
                cell(0),
                cell(
                    1,
                    state="died",
                    step=-1,
                    seed=None,
                    arms=(),
                    died_with="THE_CONFIG_WOULD_NOT_BUILD",
                ),
            ],
            5,
            6000,
        )
        assert "died: THE_CONFIG_WOULD_NOT_BUILD" in got
        assert "2/5 cells reporting" in got


def test_a_transient_does_not_kill_the_watcher(monkeypatch):
    """A watcher that dies on one bad read stops watching for the rest of the run."""

    def explode(*_args, **_kwargs):
        raise ConnectionError("wandb had a moment")

    monkeypatch.setattr(tranche_watch, "cells_of", explode)
    assert tranche_watch.report("run_x", 5, 6000).startswith("wandb unreadable:")
