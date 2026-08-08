"""Tests for the stage-1 health gate.

Every test here plants a truth and checks the estimator recovers it. The step-time filter gets
most of the attention because it is the one piece of arithmetic on this branch that has already
been got wrong in production, in both directions at once: a median over every row measures the
held-out evaluator, and a median over the rows carrying the monitor's keys measures the monitor.
The second mistake is the one that shipped, and it re-planned a tranche.
"""

import statistics

import hyper_connection_arms
import pytest
import stage_gate
from stage_gate import (
    CellHealth,
    clean_step_seconds,
    mfu_percent,
    project,
    seeds_are_distinct,
)


def history(
    *,
    steps: int = 300,
    clean: float = 8.2,
    eval_interval: int = 100,
    eval_seconds: float = 104.0,
    monitor_interval: int = 0,
    monitor_extra: float = 1.37,
    warmup: float = 12.0,
):
    """
    A run's throughput history with the instruments firing on a known schedule.

    :param steps: How many optimizer steps to plant.
    :param clean: The step time a step with no instrument on it takes. The planted truth.
    :param eval_interval: How often the held-out evaluator fires. Zero for never.
    :param eval_seconds: What one evaluation adds to the step it lands on.
    :param monitor_interval: How often the lane monitor fires. Zero for an arm with no lanes.
    :param monitor_extra: What one monitor firing adds.
    :param warmup: What the first logged step costs, paying for ``torch.compile``.

    :returns: Rows shaped like ``wandb`` history, oldest first.
    """
    rows = []
    for step in range(2, steps):
        seconds = clean
        row = {"_step": step}
        if eval_interval and step % eval_interval == 0:
            seconds += eval_seconds
            row["throughput/in-loop eval time (s)"] = eval_seconds
        elif monitor_interval and step % monitor_interval == 0:
            seconds += monitor_extra
            row["hc/min lane norm spread"] = 0.04
        if step == 2:
            seconds = warmup
        row["throughput/device/BPS"] = 1.0 / seconds
        rows.append(row)
    return rows


class TestCleanStepSeconds:
    def test_recovers_the_planted_step_time(self):
        got = clean_step_seconds(history(clean=8.2), eval_interval=100)
        assert got.median == pytest.approx(8.2)
        assert got.iqr == pytest.approx(0.0)

    @pytest.mark.parametrize("clean", [2.91, 8.2, 10.32, 15.54])
    def test_recovers_it_at_every_step_time_this_branch_has_measured(self, clean):
        got = clean_step_seconds(history(clean=clean), eval_interval=100)
        assert got.median == pytest.approx(clean)

    def test_a_median_survives_the_evaluator_but_a_mean_does_not(self):
        """
        The asymmetry the filter's docstring is careful about. At the tranche's duty cycle the
        evaluator touches 0.2% of steps, so an unfiltered median is right by robustness; it is
        anything built on a sum -- wall clock over steps, or a mean -- that reads high.
        """
        rows = history(steps=1000, clean=8.2, eval_interval=500)
        seconds = [1.0 / r["throughput/device/BPS"] for r in rows]
        got = clean_step_seconds(rows, eval_interval=500)
        assert statistics.median(seconds) == pytest.approx(got.median)
        assert statistics.mean(seconds) > got.median * 1.01

    def test_a_median_does_not_survive_a_dense_enough_instrument(self):
        """And robustness is not a licence to skip the filter: at 40% it stops holding."""
        rows = history(steps=200, clean=8.2, eval_interval=0, monitor_interval=2)
        naive = statistics.median([1.0 / r["throughput/device/BPS"] for r in rows])
        got = clean_step_seconds(rows, eval_interval=0)
        assert naive > got.median
        assert got.median == pytest.approx(8.2)

    def test_the_monitors_own_rows_measure_the_monitor(self):
        """The 11.69 s/step that re-planned a tranche, reproduced and then excluded."""
        rows = history(clean=10.32, monitor_interval=20, monitor_extra=1.37, eval_interval=0)
        monitor_rows = [r for r in rows if "hc/min lane norm spread" in r]
        trap = statistics.median([1.0 / r["throughput/device/BPS"] for r in monitor_rows])
        assert trap == pytest.approx(11.69)
        assert clean_step_seconds(rows, eval_interval=0).median == pytest.approx(10.32)

    def test_the_first_row_is_dropped(self):
        rows = history(steps=50, clean=8.2, warmup=40.0, eval_interval=0)
        got = clean_step_seconds(rows, eval_interval=0)
        assert got.excluded["warm-up"] == 1
        assert got.median == pytest.approx(8.2)
        assert got.outliers_remaining == 0

    def test_the_step_after_an_evaluation_goes_too(self):
        """Callback order decides which row absorbs the evaluation, so both are dropped."""
        rows = history(steps=120, eval_interval=50)
        got = clean_step_seconds(rows, eval_interval=50)
        assert got.excluded["held-out evaluation"] >= 2
        assert got.excluded["the step after an evaluation"] >= 2

    def test_the_baseline_arm_pays_no_monitor(self):
        got = clean_step_seconds(history(monitor_interval=0), eval_interval=100)
        assert "lane monitor" not in got.excluded

    def test_it_reports_how_many_rows_it_used(self):
        got = clean_step_seconds(history(steps=300, eval_interval=100), eval_interval=100)
        assert got.rows_used + sum(got.excluded.values()) == got.rows_total
        assert got.enough_rows

    def test_a_short_history_is_not_enough_rows(self):
        assert not clean_step_seconds(history(steps=12), eval_interval=100).enough_rows

    def test_a_slowdown_the_filter_does_not_know_about_is_reported(self):
        """A statistical filter would hide this. The rule-based one has to surface it."""
        rows = history(steps=200, clean=8.2, eval_interval=0)
        for row in rows[100:110]:
            row["throughput/device/BPS"] = 1.0 / 30.0
        got = clean_step_seconds(rows, eval_interval=0)
        assert got.outliers_remaining == 10

    def test_no_rows_at_all_is_an_error_rather_than_a_number(self):
        with pytest.raises(ValueError):
            clean_step_seconds([], eval_interval=100)


class TestMFU:
    def test_matches_what_the_run_reported(self):
        """Cell 0 of the baseline stage, read back out of its own logged row."""
        got = mfu_percent(
            flops_per_token=3_032_684_544,
            tokens_per_device_step=196_608,
            seconds_per_step=8.218751574874052,
            peak_flops=stage_gate.L40S_BF16_DENSE_FLOPS,
        )
        assert got == pytest.approx(20.038, abs=0.01)

    def test_the_l40s_peak_is_the_dense_bf16_figure(self):
        """
        NVIDIA publishes ``362.05 | 733*`` for BFLOAT16 Tensor Core on the L40S, starred for
        structural sparsity. So 362.05 is already dense and must not be halved again the way a
        sparse-quoted spec is.
        """
        assert stage_gate.L40S_BF16_DENSE_FLOPS == pytest.approx(362.05e12)
        assert stage_gate.L40S_BF16_DENSE_FLOPS == pytest.approx(733e12 / 2, rel=0.02)

    def test_the_old_a100_fallback_inflates_it(self):
        """
        What the two 370M probes were scored against before the callback grew an L40S branch:
        the A100 default of 624 TF halved to 312. It is 13.8% below the L40S's real peak, so
        every MFU it produced was 16% too high.
        """
        kwargs = dict(
            flops_per_token=3_032_684_544, tokens_per_device_step=196_608, seconds_per_step=10.32
        )
        honest = mfu_percent(**kwargs, peak_flops=stage_gate.L40S_BF16_DENSE_FLOPS)
        fallback = mfu_percent(**kwargs, peak_flops=312e12)
        assert fallback / honest == pytest.approx(362.05 / 312, rel=1e-6)
        assert honest < fallback

    def test_a_faster_step_is_a_higher_mfu(self):
        kwargs = dict(flops_per_token=3.03e9, tokens_per_device_step=196_608)
        assert mfu_percent(**kwargs, seconds_per_step=8.2) > mfu_percent(
            **kwargs, seconds_per_step=10.32
        )


class TestProjection:
    def test_the_measured_baseline_fits_its_bound(self):
        got = project("baseline", seconds_per_step=8.2, bound_hours=19.0)
        assert got.fits
        assert got.spare_fraction > 0.2

    def test_the_faithful_arm_at_its_measured_step_time_also_fits(self):
        assert project("faithful", seconds_per_step=10.32, bound_hours=19.0).fits

    def test_a_slow_enough_step_does_not_fit(self):
        got = project("baseline", seconds_per_step=12.0, bound_hours=19.0)
        assert not got.fits
        assert got.spare_fraction < 0

    def test_the_baseline_is_cheaper_than_a_lane_arm_at_one_step_time(self):
        """Only the arms with lanes pay the monitor, which is what makes the two differ."""
        baseline = project("baseline", seconds_per_step=9.0, bound_hours=19.0)
        faithful = project("faithful", seconds_per_step=9.0, bound_hours=19.0)
        assert baseline.hours < faithful.hours


class TestSeedsAreDistinct:
    @staticmethod
    def cells(seeds, offsets, upto=None):
        upto = upto or [100] * len(seeds)
        out = []
        for i, (seed, offset, last) in enumerate(zip(seeds, offsets, upto)):
            curve = {s: 10.0 - 0.04 * s + offset for s in range(last + 1)}
            out.append(
                CellHealth(
                    cell=i,
                    state="running",
                    step=last,
                    model_init_seed=seed,
                    first_loss=curve[0],
                    last_loss=curve[last],
                    loss_at=curve,
                )
            )
        return out

    def test_five_real_replicates_pass(self):
        ok, why = seeds_are_distinct(self.cells([0, 1, 2, 3, 4], [0, 0.01, 0.02, 0.03, 0.04]))
        assert ok
        assert "0, 1, 2, 3, 4" in why

    def test_five_cells_on_one_seed_are_refused(self):
        ok, why = seeds_are_distinct(self.cells([0] * 5, [0.0] * 5))
        assert not ok
        assert "repeat" in why

    def test_distinct_seeds_that_did_not_reach_the_model_are_refused(self):
        ok, why = seeds_are_distinct(self.cells([0, 1, 2, 3, 4], [0.0] * 5))
        assert not ok
        assert "identical loss" in why

    def test_a_staggered_identical_run_is_refused(self):
        """
        THE REASON THE COMPARISON IS AT A SHARED STEP. Five bit-identical cells that started
        minutes apart sit at different points on one curve, so their latest losses all differ
        and a check on those would call them five replicates.
        """
        cells = self.cells([0, 1, 2, 3, 4], [0.0] * 5, upto=[100, 90, 80, 70, 60])
        assert len({c.last_loss for c in cells}) == 5
        ok, _ = seeds_are_distinct(cells)
        assert not ok

    def test_a_cell_with_no_seed_in_its_config_is_refused(self):
        cells = self.cells([0, 1], [0.0, 0.01])
        cells[1].model_init_seed = None
        ok, why = seeds_are_distinct(cells)
        assert not ok
        assert "reported a seed" in why


class TestArmFromConfig:
    """
    The arm is in the logged config, contrary to what the branch's watcher assumes.

    ``train_on_corpus`` writes no ``arm`` field, but ``arm.apply`` edits the model config
    before it is saved and those edits are what these read.
    """

    @staticmethod
    def config_for(arm_name):
        arm = hyper_connection_arms.ARMS[arm_name]
        block = {"name": "reordered_norm"}
        if arm.hyper_connections is not None:
            block["name"] = "hyper_connection_reordered_norm"
            block["hyper_connections"] = dict(
                arm.hyper_connections.as_config_dict(),
                _CLASS_="olmo_core.nn.residual_stream.HyperConnectionConfig",
            )
        return {"model": {"block": block}}

    @pytest.mark.parametrize("arm_name", hyper_connection_arms.FUNDED)
    def test_each_funded_arm_is_the_only_funded_arm_that_matches(self, arm_name):
        """
        Uniqueness holds among the arms that run, which is the claim the gate needs and the
        only one true: ``faithful`` also matches ``decay-everything`` and ``tied-faithful``,
        which differ from it in the optimizer and in block reuse rather than in the lanes.
        """
        got = stage_gate.arms_consistent_with(self.config_for(arm_name))
        assert arm_name in got
        funded_matches = [a for a in got if a in hyper_connection_arms.FUNDED]
        assert funded_matches == [arm_name], f"{arm_name} came back as {funded_matches}"

    def test_no_funded_arm_is_mistaken_for_another(self):
        for arm_name in hyper_connection_arms.FUNDED:
            got = set(stage_gate.arms_consistent_with(self.config_for(arm_name)))
            others = set(hyper_connection_arms.FUNDED) - {arm_name}
            assert not (got & others), f"{arm_name} also matched {got & others}"

    def test_an_unfunded_arm_that_shares_a_lane_config_is_reported_as_ambiguous(self):
        """``decay-everything`` differs from ``faithful`` in the optimizer alone."""
        got = stage_gate.arms_consistent_with(self.config_for("decay-everything"))
        assert "decay-everything" in got and "faithful" in got

    def test_the_live_baseline_config_shape_reads_as_baseline(self):
        """The exact shape W&B holds for stage 1: a block with no lane config at all."""
        got = stage_gate.arms_consistent_with(
            {"model": {"block": {"name": "reordered_norm", "_CLASS_": "x"}}}
        )
        assert "baseline" in got
        assert "faithful" not in got

    def test_a_config_with_no_model_does_not_explode(self):
        assert stage_gate.arms_consistent_with({}) != ()


def test_the_self_test_passes():
    assert stage_gate.self_test() == 0


def test_the_source_list_agrees_with_the_noise_floors():
    """
    The one copy this module keeps of somebody else's constant, pinned so it cannot drift.

    Skipped rather than failed when ``noise_floor`` is not importable, because this gate has to
    keep working while that module is being edited -- which is the reason the list is duplicated
    rather than imported in the first place.
    """
    noise_floor = pytest.importorskip("noise_floor")
    assert stage_gate.HELD_OUT_SOURCES == noise_floor.HELD_OUT_SOURCES


def test_there_are_seven_sources():
    """regmix-10b-v1 publishes seven ``val-00000`` shards, one per source."""
    assert len(stage_gate.HELD_OUT_SOURCES) == 7
    assert len(set(stage_gate.HELD_OUT_SOURCES)) == 7
