"""The gate must refuse bad endpoints, and must do so without training anything."""

import pytest

from memsplit import bios, calibration, nhop
from memsplit.scorers import (
    best_constant_accuracy,
    parse_answer,
    score_items,
    wilson_interval,
)


def _items_by_depth(depths=(1, 2, 3, 4), per_depth=40):
    recs = bios.generate_records(400, seed=0)
    graph = nhop.build_graph(recs, n_layers=7, seed=0)
    by_id = {r.entity_id: r for r in recs}
    starts = nhop.eligible_starts(graph, max(depths))
    out: dict[int, list] = {}
    for d in depths:
        items = []
        for eid in starts:
            for attr in bios.ATTRIBUTES:
                got = nhop.sample_item(graph, by_id, eid, d, attr, seed=0)
                if got:
                    chain, end, value = got
                    items.append(
                        nhop.make_item(graph, by_id, eid, chain, attr, value, "comp")
                    )
                    break
            if len(items) >= per_depth:
                break
        out[d] = items
    return out


# --------------------------------------------------------------------- scorers


def test_parse_answer_returns_none_when_the_tag_is_absent():
    """Unparseable is a third outcome, not a wrong answer."""
    assert parse_answer("no tag at all") is None
    assert parse_answer("blah\nAnswer: Paris") == "Paris"
    assert parse_answer("Answer: A\nAnswer: B") == "B"


def test_below_chance_on_a_balanced_task_is_flagged():
    """0.369 on a balanced 750/750 set is 10 SE below chance -- a parser failure."""
    golds = ["yes"] * 750 + ["no"] * 750
    # Simulate the truncation asymmetry: the long 'yes' class loses its tag.
    gens = ["reasoning..."] * 750 + ["\nAnswer: no"] * 750
    out, _ = score_items(gens, golds, chance=0.5)
    assert out["accuracy"] == pytest.approx(0.5, abs=0.01)
    assert out["unparseable_rate"] == pytest.approx(0.5, abs=0.01)

    # And the genuinely broken case is flagged.
    gens_bad = ["reasoning..."] * 1100 + ["\nAnswer: no"] * 400
    bad, _ = score_items(gens_bad, golds, chance=0.5)
    assert bad["z_vs_chance"] < -3.0
    assert bad["below_chance_implausible"] is True


def test_best_constant_catches_the_two_constant_policies():
    """40% yes-rate -> constant 'no' scores 60%; 59.5 vs 40.5 is not an effect."""
    golds = ["yes"] * 400 + ["no"] * 600
    bc, label = best_constant_accuracy(golds)
    assert bc == pytest.approx(0.6)
    assert label == "no"
    always_no, _ = score_items(["\nAnswer: no"] * 1000, golds, chance=0.5)
    assert always_no["accuracy"] == pytest.approx(0.6)
    assert always_no["beats_best_constant"] is False


def test_wilson_behaves_at_the_boundaries():
    lo, hi = wilson_interval(1000, 1000)
    assert hi == pytest.approx(1.0) and lo > 0.99
    lo, hi = wilson_interval(0, 100)
    assert lo == pytest.approx(0.0) and 0.0 < hi < 0.05


def test_mixed_modes_are_not_silently_allowed():
    with pytest.raises(ValueError):
        score_items(["x"], ["y"], mode="substring_anywhere_lol")


# ----------------------------------------------------------------- the gate


def test_gate_passes_a_well_formed_depth_sweep():
    ibd = _items_by_depth()
    vocab = [v for a in bios.ATTRIBUTES for v in bios.VALUE_POOLS[a]]
    v = calibration.calibrate_endpoint(
        ibd, vocab, chance=bios.chance_accuracy("employer")
    )
    assert v.usable, v.reasons
    assert v.per_depth[0].oracle_accuracy == pytest.approx(1.0)

    # The EXPECTED curve must fall with depth. The observed sample need not be
    # strictly monotone -- at n=40 and p=0.93 the per-depth SE is ~5pp, so
    # demanding raw monotonicity would reject good endpoints on sampling noise.
    # That is the mistake this assertion used to make.
    expected = [c.oracle_noisy_expected for c in v.per_depth]
    assert expected == sorted(expected, reverse=True), expected
    for c in v.per_depth:
        assert abs(c.oracle_noisy_accuracy - c.oracle_noisy_expected) <= 3.0 * max(
            c.oracle_noisy_se, 0.005
        ), (c.depth, c.oracle_noisy_accuracy, c.oracle_noisy_expected)


def test_gate_refuses_an_endpoint_with_no_dynamic_range():
    """The mano failure: floor 4.695%, spread 0.5pp, zero cells above floor."""
    ibd = _items_by_depth(depths=(1, 2))
    # Collapse every gold to one value -> best-constant floor is 100%.
    for items in ibd.values():
        for it in items:
            it.answer = "Constant Value"
    v = calibration.calibrate_endpoint(ibd, ["Constant Value"], chance=0.05)
    assert not v.usable
    assert any("dynamic range" in r for r in v.reasons), v.reasons


def test_gate_refuses_a_leaky_endpoint():
    """Leakage means an item-specific cue reveals the answer.

    A random-from-pool guesser cannot detect this, which is why the policies are
    pluggable. Here the adversarial policy reads the answer straight off the item,
    standing in for a surface cue -- the real instance of this in the previous
    programme was a comparison task whose answer *was* one of the stated
    attributes, recoverable at 99.7% with no biographies present at all.
    """
    ibd = _items_by_depth(depths=(1, 2))
    vocab = [v for a in bios.ATTRIBUTES for v in bios.VALUE_POOLS[a]]

    leaky = dict(calibration.default_degenerate_policies(vocab))
    leaky["reads_a_surface_cue"] = lambda item, rng: f" ...\nAnswer: {item.answer}"

    v = calibration.calibrate_endpoint(
        ibd, vocab, chance=bios.chance_accuracy("employer"),
        degenerate_policies=leaky,
    )
    assert not v.usable
    assert any("leak" in r for r in v.reasons), v.reasons
    assert any("reads_a_surface_cue" in r for r in v.reasons), v.reasons
    # Dynamic range must collapse once the floor is the leaky policy, not chance.
    assert v.per_depth[0].dynamic_range_pp == pytest.approx(0.0, abs=1e-6)


def test_gate_refuses_a_flat_depth_curve():
    """A flat curve means the endpoint is not measuring serial reasoning."""
    ibd = _items_by_depth(depths=(1, 2, 3))
    vocab = [v for a in bios.ATTRIBUTES for v in bios.VALUE_POOLS[a]]
    # per_hop = 1.0 -> the oracle never degrades, so the curve is flat.
    v = calibration.calibrate_endpoint(
        ibd, vocab, chance=bios.chance_accuracy("employer"), per_hop_noisy=1.0
    )
    assert not v.usable
    assert any("flat" in r or "falls only" in r for r in v.reasons), v.reasons


def test_gate_needs_no_trained_model():
    """It is a pure function of the items -- that is what makes it cheap enough."""
    import inspect

    src = inspect.getsource(calibration.calibrate_endpoint)
    for forbidden in ("torch", "forward", "checkpoint", "state_dict"):
        assert forbidden not in src, forbidden


def test_noisy_oracle_reproduces_the_pn_curve():
    """The simulated solver should track p**n, since that is the null."""
    ibd = _items_by_depth(depths=(1, 2, 3, 4), per_depth=400)
    vocab = [v for a in bios.ATTRIBUTES for v in bios.VALUE_POOLS[a]]
    v = calibration.calibrate_endpoint(
        ibd, vocab, chance=bios.chance_accuracy("employer"), per_hop_noisy=0.9
    )
    for c in v.per_depth:
        n_lookups = c.depth + 1
        expected = nhop.expected_chain_accuracy(0.9, n_lookups)
        assert abs(c.oracle_noisy_accuracy - expected) < 0.12, (c.depth, expected)


def test_required_n_for_mde_matches_the_published_anchor():
    """A ~3pp effect needs ~1000 items; n=750 gave 28-29% power."""
    n = calibration.required_n_for_mde(mde_pp=3.0, sd_pp=33.3)
    assert 900 < n < 1100, n
    # Smaller effects cost quadratically.
    assert calibration.required_n_for_mde(1.5, 33.3) > 3 * n
