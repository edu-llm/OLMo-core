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


#: ``num_flops_per_token(4096)`` for the 370M arm at the padded dolma2 vocabulary of 100,352.
#: Built locally in ``test_the_flops_per_token_is_what_the_runs_logged`` and matched against
#: what both shapes' runs imply from ``flopsPS / TPS``.
FLOPS_PER_TOKEN_370M = 3_032_684_544


class TestMFU:
    def test_matches_what_the_a100_run_reported(self):
        """
        Cell 0 of the A100 baseline, read back out of its own logged row: a clean median of
        1.681 s over 710 rows, 98,304 tokens on the device, against the A100's 312 TF.
        """
        got = mfu_percent(
            flops_per_token=FLOPS_PER_TOKEN_370M,
            tokens_per_device_step=98_304,
            seconds_per_step=1.6810,
            peak_flops=stage_gate.A100_BF16_DENSE_FLOPS,
        )
        assert got == pytest.approx(56.85, abs=0.05)

    def test_matches_what_the_l40s_run_reported_once_the_peak_is_right(self):
        """
        The same cell of stage 1 on the L40S. 20.04% was what the callback reported and what
        this module confirmed, and both were against a peak the card cannot reach; at the
        FP32-accumulate rate the run was really at 40.08%.
        """
        kwargs = dict(
            flops_per_token=FLOPS_PER_TOKEN_370M,
            tokens_per_device_step=196_608,
            seconds_per_step=8.218751574874052,
        )
        assert mfu_percent(**kwargs, peak_flops=362.05e12) == pytest.approx(20.04, abs=0.01)
        assert mfu_percent(**kwargs, peak_flops=stage_gate.L40S_BF16_DENSE_FLOPS) == pytest.approx(
            40.08, abs=0.01
        )

    def test_the_l40s_peak_is_the_fp32_accumulate_rate_and_not_the_headline(self):
        """
        THE SECOND FACTOR OF TWO, WHICH IS NOT SPARSITY. The L40S datasheet leads with
        ``362.05 | 733*`` for BFLOAT16 Tensor Core, and the star is sparsity -- so reading
        362.05 as the dense figure is right as far as it goes and still lands two times high,
        because that column is quoted with FP16 accumulation. NVIDIA's Ada whitepaper gives
        AD102 at 330.3 TFLOPS for FP16 with FP16 accumulate against 165.2 with FP32, and lists
        the L40 at ``181 | 362`` for BF16. Torch accumulates in FP32 and cannot opt out.
        """
        assert stage_gate.L40S_BF16_DENSE_FLOPS == pytest.approx(181.03e12, rel=1e-3)
        assert stage_gate.L40S_BF16_DENSE_FLOPS == pytest.approx(362.05e12 / 2)

    def test_the_a100_is_not_derated_because_its_die_has_no_such_penalty(self):
        """
        GA100 reaches its quoted BF16 rate with FP32 accumulate, so the only correction the
        A100 takes is the sparsity one: 624 starred, 312 dense.
        """
        assert stage_gate.A100_BF16_DENSE_FLOPS == pytest.approx(624e12 / 2)
        assert stage_gate.peak_bf16_flops("NVIDIA A100-SXM4-40GB") == 312e12
        assert stage_gate.peak_bf16_flops("NVIDIA A100-SXM4-80GB") == 312e12

    def test_the_pre_v250_a100_constant_inflated_mfu_by_two(self):
        """The bug the v2.5.0 changelog records, reproduced so the direction is not guessed at."""
        kwargs = dict(
            flops_per_token=FLOPS_PER_TOKEN_370M,
            tokens_per_device_step=98_304,
            seconds_per_step=1.69,
        )
        honest = mfu_percent(**kwargs, peak_flops=stage_gate.A100_BF16_DENSE_FLOPS)
        broken = mfu_percent(**kwargs, peak_flops=stage_gate.A100_BF16_DENSE_FLOPS / 2)
        assert broken == pytest.approx(2 * honest)

    def test_a_faster_step_is_a_higher_mfu(self):
        kwargs = dict(
            flops_per_token=3.03e9,
            tokens_per_device_step=196_608,
            peak_flops=stage_gate.L40S_BF16_DENSE_FLOPS,
        )
        assert mfu_percent(**kwargs, seconds_per_step=8.2) > mfu_percent(
            **kwargs, seconds_per_step=10.32
        )


class TestDevicePeaks:
    def test_the_longest_name_wins_because_l4_is_a_prefix_of_l40s(self):
        """The trap ``SpeedMonitorCallback`` documents, asserted rather than trusted."""
        assert stage_gate.peak_bf16_flops("NVIDIA L40S") == stage_gate.L40S_BF16_DENSE_FLOPS
        assert stage_gate.peak_bf16_flops("NVIDIA L4") == pytest.approx(121e12)
        assert stage_gate.peak_bf16_flops("NVIDIA L40") != stage_gate.peak_bf16_flops("NVIDIA L4")

    def test_an_unknown_part_returns_none_rather_than_the_a100_default(self):
        """
        THE FAILURE THE CALLBACK STILL HAS AND THIS TABLE MUST NOT COPY. Its A100 figure is the
        ``else`` at the bottom of the chain, so an unrecognised card is scored against 312 TF
        and reports an MFU unrelated to its hardware. Here that is ``None`` and gets said.
        """
        assert stage_gate.peak_bf16_flops("NVIDIA GeForce RTX 4090") is None
        assert stage_gate.peak_bf16_flops("") is None

    def test_a100_is_matched_by_name_and_not_reached_by_falling_through(self):
        """``A100`` has its own row, so the table's answer for it is an assertion about it."""
        assert any(token == "A100" for token, _ in stage_gate.DENSE_BF16_PEAK_FLOPS)


class TestFlopsPerToken:
    def test_it_is_what_both_shapes_logged(self):
        """
        THE NUMERATOR, BUILT LOCALLY RATHER THAN DIVIDED OUT OF THE RUN. Every MFU here is
        ``flops_per_token`` over a peak, and reading the numerator back out of the same rows
        that carry the figure being checked would leave only the peak actually checked. Built
        from the arm table's own factory, it has to land on what the runs imply from
        ``flopsPS / TPS``, which both shapes put at 3,032,684,6xx before float32 rounding.

        At the padded vocabulary, which is the part worth pinning: dolma2 is 100,278 tokens and
        the run pads to a multiple of 128, and OLMo-3 unties the embeddings, so the 74 padding
        rows are 75,776 real parameters in the LM head and 454,656 FLOPs per token. That is
        0.015% and is the entire difference between building this with the raw vocabulary and
        building it the way the run did.
        """
        model = hyper_connection_arms.hc_370M(vocab_size=100_352).build()
        assert model.num_flops_per_token(4096) == FLOPS_PER_TOKEN_370M

    def test_the_unpadded_vocabulary_is_the_near_miss(self):
        model = hyper_connection_arms.hc_370M(vocab_size=100_278).build()
        got = model.num_flops_per_token(4096)
        assert got != FLOPS_PER_TOKEN_370M
        assert got == pytest.approx(FLOPS_PER_TOKEN_370M, rel=2e-4)


class TestProjection:
    def test_the_measured_baseline_fits_its_bound(self):
        got = project("baseline", seconds_per_step=8.2, bound_hours=19.0)
        assert got.fits
        assert got.spare_fraction > 0.2

    def test_the_a100_baseline_fits_its_seven_hour_bound(self):
        """The slowest A100 cell, at the evaluation cost that shape actually measures."""
        got = project("baseline", seconds_per_step=1.729, bound_hours=7.0, eval_seconds=25.4)
        assert got.fits
        assert got.hours == pytest.approx(3.19, abs=0.05)

    def test_charging_the_l40s_evaluation_on_an_a100_overstates_the_run(self):
        """
        Why ``eval_seconds`` had to become an argument. Fourteen evaluations at 104 s against
        14 at 25 s is a fifth of an hour, which is 6% of a 3-hour cell and would be the whole
        margin on a tighter bound.
        """
        measured = project("baseline", seconds_per_step=1.729, bound_hours=7.0, eval_seconds=25.4)
        borrowed = project("baseline", seconds_per_step=1.729, bound_hours=7.0, eval_seconds=104.0)
        assert borrowed.hours > measured.hours
        assert borrowed.hours - measured.hours == pytest.approx(14 * 78.6 / 3600, abs=0.01)

    def test_the_default_is_still_the_l40s_measurement(self):
        """Nothing that priced a run before ``arm_seconds`` grew keywords changed its answer."""
        assert project("baseline", seconds_per_step=8.2, bound_hours=19.0).hours == pytest.approx(
            hyper_connection_arms.arm_seconds(
                hyper_connection_arms.ARMS["baseline"], seconds_per_step=8.2
            )
            / 3600.0
        )

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
        model = {"block": block}
        if arm.reuse_factor is not None:
            model["block_reuse"] = {"n_unique_blocks": 16 // arm.reuse_factor}
        return {"model": model}

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

    def test_the_live_baseline_config_shape_reads_as_baseline_alone(self):
        """
        The exact shape W&B holds for the A100 cells: a block with no lane config and no block
        reuse. Unambiguous, which it was not before block reuse was read -- every baseline cell
        used to come back as ``baseline`` and ``tied-baseline`` together.
        """
        got = stage_gate.arms_consistent_with(
            {"model": {"block": {"name": "reordered_norm", "_CLASS_": "x"}}}
        )
        assert got == ("baseline",)

    def test_block_reuse_separates_the_tied_arms_from_the_untied_ones(self):
        tied = stage_gate.arms_consistent_with(self.config_for("tied-baseline"))
        assert tied == ("tied-baseline",)
        assert "baseline" not in tied
        assert "tied-faithful" not in stage_gate.arms_consistent_with(self.config_for("faithful"))

    def test_a_config_with_no_model_does_not_explode(self):
        assert stage_gate.arms_consistent_with({}) != ()


class TestASummaryThatWasOverwritten:
    """
    A crash report that re-initialised W&B under a cell's own id took the summary with it, on
    seven cells. What is left there is no ``_step`` and no ``eval/lm/*`` keys, and the gate
    reads both from the summary -- so such a cell arrives at step 0 with no held-out sources,
    and the per-cell held-out check runs over ``[c for c in cells if c.sources]`` and therefore
    does not fail it. It skips it. The history holds every one of those keys.
    """

    def test_the_sources_are_found_in_the_history_when_the_summary_holds_none(self):
        rows = [
            {"_step": 500, "eval/lm/dclm/CE loss": 2.6, "eval/lm/dclm/BPB": 0.81},
            {"_step": 500, "eval/lm/wiki/CE loss": 2.4, "eval/lm/wiki/BPB": 0.76},
        ]
        keys = {key for row in rows for key in row}
        assert stage_gate._held_out_sources(keys, "/CE loss") == ("dclm", "wiki")
        assert stage_gate._held_out_sources(keys, "/BPB") == ("dclm", "wiki")

    def test_a_summary_that_has_them_is_still_what_is_read(self):
        summary = ["_step", "eval/lm/arxiv/CE loss", "eval/lm/arxiv/BPB", "train/CE loss"]
        assert stage_gate._held_out_sources(summary, "/CE loss") == ("arxiv",)

    def test_nothing_anywhere_is_no_sources_rather_than_an_error(self):
        assert stage_gate._held_out_sources([], "/CE loss") == ()
        assert stage_gate._held_out_sources(["train/CE loss"], "/CE loss") == ()

    def test_the_gate_says_which_cells_it_had_to_read_from_history(self, capsys):
        cells = [
            CellHealth(cell=0, state="failed", step=4910, summary_lost_its_step=True),
            CellHealth(cell=1, state="failed", step=4995),
        ]
        stage_gate.report(cells, expected_cells=5, arm_name="output-only", bound_hours=6.0)
        printed = capsys.readouterr().out
        assert "[step and sources from history]" in printed
        assert "cell(s) 0 have a W&B summary carrying no step" in printed

    def test_a_gate_over_intact_cells_says_nothing_about_history(self, capsys):
        stage_gate.report(
            [CellHealth(cell=0, state="running", step=500)],
            expected_cells=5,
            arm_name="baseline",
            bound_hours=6.0,
        )
        assert "from history" not in capsys.readouterr().out


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
