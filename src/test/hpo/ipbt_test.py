import numpy as np
import pytest

from olmo_core.hpo.ipbt import (
    HPSource,
    IPBTConfig,
    IPBTController,
    Member,
    NewLineageRequired,
    RestartTracker,
    WeightPolicy,
    optimizer_reset_for,
)


def _members(n=8):
    # Higher score is better; member i has score i/10.
    return [
        Member(
            member_id=f"m{i}",
            lineage_id=f"L{i}",
            unit_config=(i / 10.0,),
            score=i / 10.0,
            fidelity=1000,
            checkpoint_ref=f"/ckpt/m{i}",
            optimizer_state_valid=True,
        )
        for i in range(n)
    ]


def _bo_configs(n=8):
    return [((i + 1) / (n + 1),) for i in range(n)]


def test_optimizer_reset_policy_matches_weight_policy():
    assert optimizer_reset_for(WeightPolicy.PURE_COPY) is False
    assert optimizer_reset_for(WeightPolicy.SHRINK_PERTURB) is True
    assert optimizer_reset_for(WeightPolicy.FRESH_RESET) is True


def test_generation_preserves_top_and_replaces_bottom():
    ctrl = IPBTController(IPBTConfig(population_size=8, top_quantile=0.25, bottom_quantile=0.25))
    plan = ctrl.plan_generation(_members(8), rng=np.random.default_rng(0), bo_configs=_bo_configs())
    kept_ids = {m.member_id for m in plan.kept}
    # Everyone except the bottom quartile continues unchanged (top + middle = 6 of 8).
    assert len(plan.kept) == 6
    assert {"m7", "m6"} <= kept_ids  # the top quartile is always kept
    assert {"m0", "m1"}.isdisjoint(kept_ids)  # the bottom quartile is replaced, not kept
    # Bottom quartile (2 of 8) is replaced by descendants of top donors.
    assert len(plan.descendants) == 2
    assert {d.slot_id for d in plan.descendants} == {"m0", "m1"}
    for d in plan.descendants:
        assert d.donor_id in {"m7", "m6"} or d.weight_policy is WeightPolicy.FRESH_RESET


def test_descendant_ratios_are_fixed_and_lineage_is_tracked():
    ctrl = IPBTController(
        IPBTConfig(
            population_size=8,
            top_quantile=0.5,
            bottom_quantile=0.5,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
        )
    )
    plan = ctrl.plan_generation(_members(8), rng=np.random.default_rng(0), bo_configs=_bo_configs())
    d = plan.descendants
    assert len(d) == 4
    # Exactly half fresh-reset, half shrink-perturb (preregistered ratio, not random per slot).
    assert sum(x.weight_policy is WeightPolicy.FRESH_RESET for x in d) == 2
    assert sum(x.weight_policy is WeightPolicy.SHRINK_PERTURB for x in d) == 2
    # Exactly half random HP, half BO HP.
    assert sum(x.hp_source is HPSource.RANDOM for x in d) == 2
    assert sum(x.hp_source is HPSource.BO for x in d) == 2
    # Every mutation starts a child lineage; shrink-perturb records the donor lineage as parent.
    existing_lineages = {m.lineage_id for m in _members(8)}
    for x in d:
        assert x.lineage_id not in existing_lineages
        if x.weight_policy is WeightPolicy.FRESH_RESET:
            assert x.donor_id is None
            assert x.parent_lineage_id is None
            assert x.schedule_age_tokens == 0
        else:
            donor = next(m for m in _members(8) if m.member_id == x.donor_id)
            assert x.parent_lineage_id == donor.lineage_id
            assert x.weight_scale == pytest.approx(0.4)
        assert x.unit_config is not None

    # The two exploration axes are independent: all 2x2 cells appear for a four-slot split.
    assert {(x.weight_policy, x.hp_source) for x in d} == {
        (WeightPolicy.FRESH_RESET, HPSource.RANDOM),
        (WeightPolicy.FRESH_RESET, HPSource.BO),
        (WeightPolicy.SHRINK_PERTURB, HPSource.RANDOM),
        (WeightPolicy.SHRINK_PERTURB, HPSource.BO),
    }


def test_online_mutation_restricted_to_state_safe_keys():
    ctrl = IPBTController(IPBTConfig(population_size=8))
    hps = {"lr": 1e-3, "weight_decay": 0.1, "max_grad_norm": 1.0, "beta2_gap": 0.01}
    out = ctrl.mutate_online(hps, {"lr": 2e-3, "weight_decay": 0.05}, rng=np.random.default_rng(0))
    assert out["lr"] == 2e-3 and out["weight_decay"] == 0.05
    # Changing a state-unsafe key must start a new lineage instead of mutating online.
    with pytest.raises(NewLineageRequired):
        ctrl.mutate_online(hps, {"beta2_gap": 0.02}, rng=np.random.default_rng(0))


def test_restart_splits_all_non_top_members_between_reset_and_shrink():
    ctrl = IPBTController(
        IPBTConfig(population_size=8, top_quantile=0.25, reset_fraction=0.5, random_hp_fraction=0.5)
    )
    plan = ctrl.restart_population(
        _members(8), rng=np.random.default_rng(0), bo_configs=_bo_configs()
    )
    assert plan.copies == []
    assert len(plan.descendants) == 6
    assert (
        sum(descendant.weight_policy is WeightPolicy.FRESH_RESET for descendant in plan.descendants)
        == 3
    )
    assert (
        sum(
            descendant.weight_policy is WeightPolicy.SHRINK_PERTURB
            for descendant in plan.descendants
        )
        == 3
    )


def test_restart_affects_every_member_outside_top_quantile():
    controller = IPBTController(
        IPBTConfig(
            population_size=8,
            top_quantile=0.25,
            bottom_quantile=0.25,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
        )
    )
    plan = controller.restart_population(
        _members(8),
        rng=np.random.default_rng(0),
        bo_configs=_bo_configs(16),
    )
    assert plan.copies == []
    assert {descendant.slot_id for descendant in plan.descendants} == {
        "m0",
        "m1",
        "m2",
        "m3",
        "m4",
        "m5",
    }


def test_update_interval_doubles_and_restart_trigger_is_patient():
    tracker = RestartTracker(patience=3, interval=100)
    assert tracker.update(best_score=0.5) is False  # improved
    assert tracker.update(best_score=0.5) is False  # 1 stale
    assert tracker.update(best_score=0.5) is False  # 2 stale
    fired = tracker.update(best_score=0.5)  # 3 stale -> restart
    assert fired is True
    assert tracker.interval == 200  # doubled on restart
    # After a restart, an improvement resets the stale counter.
    assert tracker.update(best_score=0.9) is False


def test_ranking_is_within_comparable_strata_only():
    ctrl = IPBTController(IPBTConfig(population_size=8))
    same_hp_diff_lineage = [
        Member(
            "a",
            "L0",
            (0.5,),
            score=0.9,
            fidelity=1000,
            checkpoint_ref="",
            optimizer_state_valid=True,
        ),
        Member(
            "b",
            "L1",
            (0.5,),
            score=0.1,
            fidelity=2000,
            checkpoint_ref="",
            optimizer_state_valid=True,
        ),
    ]
    strata = ctrl.group_by_stratum(same_hp_diff_lineage)
    # Different lineage/fidelity -> different strata -> never pooled for ranking.
    assert len(strata) == 2


def test_generation_never_uses_cross_fidelity_donor():
    members = _members(8)
    members = [
        Member(
            member_id=m.member_id,
            lineage_id=m.lineage_id,
            unit_config=m.unit_config,
            score=m.score,
            fidelity=1000 if i < 4 else 2000,
            checkpoint_ref=m.checkpoint_ref,
            optimizer_state_valid=m.optimizer_state_valid,
        )
        for i, m in enumerate(members)
    ]
    plan = IPBTController(
        IPBTConfig(population_size=8, top_quantile=0.25, bottom_quantile=0.25)
    ).plan_generation(members, rng=np.random.default_rng(0), bo_configs=_bo_configs())
    by_id = {member.member_id: member for member in members}
    by_slot = {member.member_id: member for member in members}
    for descendant in plan.descendants:
        if descendant.donor_id is not None:
            assert by_id[descendant.donor_id].fidelity == by_slot[descendant.slot_id].fidelity


def test_population_and_quantile_invariants_are_enforced():
    ctrl = IPBTController(IPBTConfig(population_size=8))
    with pytest.raises(ValueError):
        ctrl.plan_generation(_members(3), rng=np.random.default_rng(0), bo_configs=_bo_configs())
    with pytest.raises(ValueError):
        IPBTConfig(population_size=8, top_quantile=0.75, bottom_quantile=0.75)


def test_generation_consumes_rng_and_produces_complete_configs():
    ctrl = IPBTController(
        IPBTConfig(
            population_size=8,
            top_quantile=0.5,
            bottom_quantile=0.5,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
        )
    )
    rng = np.random.default_rng(7)
    before = repr(rng.bit_generator.state)
    plan = ctrl.plan_generation(_members(), rng=rng, bo_configs=_bo_configs())
    assert repr(rng.bit_generator.state) != before
    assert all(
        descendant.unit_config is not None
        and all(0.0 <= value <= 1.0 for value in descendant.unit_config)
        for descendant in plan.descendants
    )


def test_restart_uses_weight_policy_and_configured_tracker():
    members = _members()
    members[-1] = Member("m7", "L7", (0.7,), 0.7, 1000, "/ckpt/m7", optimizer_state_valid=False)
    ctrl = IPBTController(
        IPBTConfig(
            population_size=8,
            top_quantile=0.25,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
            shrink_perturb_factor=0.25,
            update_interval_init=123,
            restart_patience=4,
        )
    )
    plan = ctrl.restart_population(members, rng=np.random.default_rng(0), bo_configs=_bo_configs())
    assert ctrl.restart_tracker.interval == 123
    assert ctrl.restart_tracker.patience == 4
    assert all(
        descendant.weight_scale == pytest.approx(0.25)
        for descendant in plan.descendants
        if descendant.weight_policy is WeightPolicy.SHRINK_PERTURB
    )
    for copy in plan.copies:
        donor = next(member for member in members if member.member_id == copy.donor_id)
        assert copy.optimizer_reset is (not donor.optimizer_state_valid)


def test_ipbt_state_round_trip_preserves_lineages_and_restart_tracker():
    first = IPBTController(IPBTConfig(population_size=8, update_interval_init=10))
    first._used_lineage_ids.add("Lnew0")
    assert first._new_lineage_id() == "Lnew1"
    first.restart_tracker.update(0.5)
    first.restart_tracker.update(0.5)
    state = first.state_dict()

    restored = IPBTController(IPBTConfig(population_size=8, update_interval_init=10))
    restored.load_state_dict(state)
    assert restored._new_lineage_id() == first._new_lineage_id()
    assert restored.restart_tracker.interval == first.restart_tracker.interval
    assert restored.restart_tracker._stale == first.restart_tracker._stale


def test_optional_initial_oversampling_selects_declared_population():
    controller = IPBTController(IPBTConfig(population_size=4, initial_oversample=8))
    selected = controller.select_initial_population(_members(8))
    assert [member.member_id for member in selected] == ["m7", "m6", "m5", "m4"]
    with pytest.raises(ValueError):
        controller.select_initial_population(_members(7))


def test_sparse_comparable_strata_are_kept_without_cross_donors():
    members = _members(4)
    members[-1] = Member(
        member_id=members[-1].member_id,
        lineage_id=members[-1].lineage_id,
        unit_config=members[-1].unit_config,
        score=members[-1].score,
        fidelity=members[-1].fidelity,
        checkpoint_ref=members[-1].checkpoint_ref,
        optimizer_state_valid=True,
        comparison_stratum="inherited",
    )
    plan = IPBTController(
        IPBTConfig(population_size=4, top_quantile=0.25, bottom_quantile=0.25)
    ).plan_generation(
        members,
        rng=np.random.default_rng(0),
        bo_configs=_bo_configs(4),
    )
    assert members[-1].member_id in {member.member_id for member in plan.kept}
    assert all(descendant.donor_id != members[-1].member_id for descendant in plan.descendants)


def test_small_restart_cohorts_balance_all_policy_cells_over_time():
    controller = IPBTController(
        IPBTConfig(
            population_size=8,
            top_quantile=0.25,
            bottom_quantile=0.25,
            reset_fraction=0.5,
            random_hp_fraction=0.5,
        )
    )
    cells = set()
    for seed in range(4):
        plan = controller.restart_population(
            _members(8),
            rng=np.random.default_rng(seed),
            bo_configs=_bo_configs(),
        )
        cells.update(
            (descendant.weight_policy, descendant.hp_source) for descendant in plan.descendants
        )
    assert cells == {
        (WeightPolicy.FRESH_RESET, HPSource.RANDOM),
        (WeightPolicy.FRESH_RESET, HPSource.BO),
        (WeightPolicy.SHRINK_PERTURB, HPSource.RANDOM),
        (WeightPolicy.SHRINK_PERTURB, HPSource.BO),
    }
