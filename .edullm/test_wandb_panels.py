"""Tests for the metric verifier.

THE THING BEING TESTED IS A GATE THAT USED TO PASS FOR THE WRONG REASON, so most of these
plant that exact wrong reason and check it no longer works. The old ``--verify`` unioned metric
keys across a whole W&B group and keyed its results by ``run.name``; the group spans every
probe and arm this module has ever run, and every cell of a fan-out shares a name. It therefore
reported the ``hc/*`` families satisfied by months-old ``faithful`` probes while gating five
``baseline`` cells that cannot log a lane metric at all, printed "everything the
pre-registration rests on is present", and exited 0.
"""

import sys

import pytest
import wandb_panels
from wandb_panels import EVERY, EXPECTED, LANES, Cell, Family, applies, matched

BASELINE_KEYS = (
    "eval/lm/arxiv/BPB",
    "eval/lm/arxiv/CE loss",
    "optim/LR (group 0)",
    "optim/LR (group 1)",
    "throughput/device/MFU",
    "throughput/device/TPS",
    "throughput/total tokens",
    "train/CE loss",
)

LANE_KEYS = BASELINE_KEYS + (
    "hc/min lane norm spread",
    "hc/block 00/lane norm spread",
    "hc/block 00/lane 0 norm",
    "hc/block 00/rho(A_r) attention",
    "hc/block 00/rho(A_r) feed_forward",
    "hc/composite condition number",
    "hc/composite spectral radius",
    "hc/block 00/hidden norm",
)


def cell(index=0, *, keys=BASELINE_KEYS, arms=("baseline",), state="running", step=1000):
    return Cell(index=index, state=state, step=step, keys=tuple(keys), arms=tuple(arms))


class TestScope:
    def test_a_lane_family_does_not_apply_to_the_baseline(self):
        guard = next(f for f in EXPECTED if f.section == "the guard")
        assert guard.scope == LANES
        assert not applies(guard, cell(arms=("baseline",)))

    def test_a_lane_family_applies_to_every_funded_lane_arm(self):
        guard = next(f for f in EXPECTED if f.section == "the guard")
        for arm in ("faithful", "output-only", "mhc"):
            assert applies(guard, cell(arms=(arm,), keys=LANE_KEYS)), arm

    def test_a_universal_family_applies_to_both(self):
        result = next(f for f in EXPECTED if f.section == "the result")
        assert result.scope == EVERY
        assert applies(result, cell(arms=("baseline",)))
        assert applies(result, cell(arms=("faithful",), keys=LANE_KEYS))

    def test_an_ambiguous_arm_that_agrees_about_lanes_still_resolves(self):
        """``faithful`` and ``decay-everything`` differ in the optimizer, not in the lanes."""
        assert cell(arms=("faithful", "decay-everything"), keys=LANE_KEYS).has_lanes is True

    def test_an_arm_the_table_does_not_name_leaves_it_undecided(self):
        assert cell(arms=()).has_lanes is None


class TestVerdictOnOneCell:
    """The per-cell arithmetic, without a network."""

    @staticmethod
    def shortfalls(cells):
        out = {}
        for family in EXPECTED:
            for c in cells:
                if not applies(family, c):
                    continue
                absent = [p for p in family.patterns if not matched(c.keys, p)]
                if absent and family.required:
                    out.setdefault(family.section, []).append(c.index)
        return out

    def test_a_healthy_baseline_cell_is_short_of_nothing_required(self):
        assert self.shortfalls([cell()]) == {}

    def test_a_healthy_lane_cell_is_short_of_nothing_required(self):
        assert self.shortfalls([cell(arms=("faithful",), keys=LANE_KEYS)]) == {}

    def test_a_lane_cell_missing_the_guard_is_caught(self):
        crippled = tuple(k for k in LANE_KEYS if "lane" not in k)
        assert "the guard" in self.shortfalls([cell(arms=("faithful",), keys=crippled)])

    def test_one_bad_cell_among_four_good_ones_is_not_hidden(self):
        """
        THE DEFECT THAT KEYING ON ``run.name`` CAUSED. Five cells share a display name, so the
        old verifier kept one of them and reported the submission on that cell's evidence.
        """
        cells = [cell(index=i, arms=("faithful",), keys=LANE_KEYS) for i in range(4)]
        cells.append(cell(index=4, arms=("faithful",), keys=BASELINE_KEYS))
        assert self.shortfalls(cells) == {
            "the guard": [4],
            "stability": [4],
            "hidden-state scale": [4],
        }

    def test_a_baseline_cell_is_not_failed_for_lacking_lane_metrics(self):
        """
        The other half of scoping. Requiring ``hc/*`` of a baseline cell would be a verifier
        that always fails, which gets ignored just as fast as one that always passes.
        """
        assert self.shortfalls([cell(arms=("baseline",))]) == {}


class TestBorrowedEvidence:
    def test_a_union_over_a_group_is_what_used_to_hide_a_missing_family(self):
        """
        The bug, planted. Four baseline cells and one old ``faithful`` probe in the same group:
        union their keys and every ``hc/*`` family looks present, although not one of the cells
        being gated has a single lane metric.
        """
        gated = [cell(index=i, arms=("baseline",)) for i in range(4)]
        stale_probe = cell(index=99, arms=("faithful",), keys=LANE_KEYS)

        union = sorted({k for c in gated + [stale_probe] for k in c.keys})
        guard = next(f for f in EXPECTED if f.section == "the guard")
        assert all(matched(union, p) for p in guard.patterns), "the trap does not reproduce"

        # Scoped to the cells actually being gated, the family is not claimed at all.
        assert not any(applies(guard, c) for c in gated)


class TestLeakedLaneMetrics:
    def test_lane_keys_on_a_baseline_cell_are_a_failure_and_not_a_bonus(self):
        """
        The monitor is attached only for an arm with lanes, so an ``hc/*`` key on a cell whose
        config has none means the cell did not run the arm it was submitted as. A verifier that
        only ever asks "is it missing" would score this as a pass with extras.
        """
        impossible = cell(arms=("baseline",), keys=LANE_KEYS)
        guard = next(f for f in EXPECTED if f.section == "the guard")
        assert impossible.has_lanes is False
        assert any(matched(impossible.keys, p) for p in guard.patterns)


class TestTheTable:
    def test_every_family_declares_a_scope_the_verifier_understands(self):
        assert all(f.scope in (EVERY, LANES) for f in EXPECTED)

    def test_every_hc_family_is_scoped_to_lanes_and_no_other_family_is(self):
        """
        The invariant the whole fix rests on, asserted rather than maintained by hand: a family
        is lane-scoped exactly when its patterns are lane metrics.
        """
        for family in EXPECTED:
            lane_patterns = all(p.startswith("hc/") for p in family.patterns)
            assert lane_patterns == (family.scope == LANES), family.section

    def test_the_sections_are_unique(self):
        sections = [f.section for f in EXPECTED]
        assert len(sections) == len(set(sections))

    def test_the_weight_decay_split_is_advisory_because_it_does_not_discriminate(self):
        """
        Measured, not assumed: the live baseline cells log ``optim/LR (group 1)`` too, because
        OLMo-core already splits decay off the norms and biases. So this family cannot tell a
        lane arm's split from the stock one and must not be required of one.
        """
        family = next(f for f in EXPECTED if f.section == "the weight-decay split")
        assert family.scope == EVERY
        assert not family.required
        assert all(matched(BASELINE_KEYS, p) for p in family.patterns)

    def test_downstream_is_not_required_because_it_arrives_from_another_job(self):
        family = next(f for f in EXPECTED if f.section == "downstream")
        assert not family.required


class TestCli:
    @staticmethod
    def run(monkeypatch, *argv):
        monkeypatch.setattr(sys, "argv", ["wandb_panels.py", *argv])
        return wandb_panels.main()

    def test_verify_over_a_group_is_refused_outright(self, monkeypatch):
        """
        A group spans every probe and arm this module has ever run, so a gate over one answers
        "does this project contain these keys somewhere". That question has no useful answer
        and used to be the only one asked, so the flag combination is now an error rather than
        a fallback.
        """
        with pytest.raises(SystemExit):
            self.run(monkeypatch, "--verify", "--group", "hyper-connections-370m")

    def test_report_still_takes_a_group_because_a_panel_wants_the_union(self, monkeypatch):
        """The union is right for a chart and wrong for a gate. Only the gate changed."""
        with pytest.raises(SystemExit):
            self.run(monkeypatch, "--report")

    def test_an_arm_outside_the_table_is_refused_before_the_network(self, monkeypatch):
        with pytest.raises(SystemExit):
            self.run(monkeypatch, "--verify", "--run", "run_x", "--arm", "not-an-arm")

    def test_a_family_is_a_frozen_record(self):
        family = EXPECTED[0]
        assert isinstance(family, Family)
        with pytest.raises(Exception):
            family.required = False  # type: ignore[misc]
