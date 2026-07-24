from olmo_core.review_lab.schedules import ReviewController, fixed_review_steps


def test_fixed_schedules_are_budget_and_endpoint_matched():
    args = dict(events=12, first_step=8, last_step=165, expansion_ratio=1.35)
    uniform = fixed_review_steps("uniform", **args)
    expanding = fixed_review_steps("expanding", **args)
    cramming = fixed_review_steps("cramming", **args)

    assert len(uniform) == len(expanding) == len(cramming) == 12
    assert uniform[0] == expanding[0] == cramming[0] == 8
    assert uniform[-1] == expanding[-1] == cramming[-1] == 165
    for schedule in (uniform, expanding, cramming):
        assert len(set(schedule)) == len(schedule)
    assert uniform != expanding
    assert uniform != cramming


def test_expanding_schedule_has_larger_late_gaps():
    steps = fixed_review_steps(
        "expanding", events=10, first_step=4, last_step=120, expansion_ratio=1.4
    )
    gaps = [right - left for left, right in zip(steps, steps[1:])]
    assert sum(gaps[-3:]) > sum(gaps[:3])


def test_cramming_schedule_has_smaller_late_gaps():
    steps = fixed_review_steps(
        "cramming", events=10, first_step=4, last_step=120, expansion_ratio=1.4
    )
    gaps = [right - left for left, right in zip(steps, steps[1:])]
    assert sum(gaps[-3:]) < sum(gaps[:3])


def test_cramming_is_time_reversed_expanding():
    args = dict(events=12, first_step=8, last_step=165, expansion_ratio=1.35)
    expanding = fixed_review_steps("expanding", **args)
    cramming = fixed_review_steps("cramming", **args)
    assert cramming == [8 + 165 - step for step in reversed(expanding)]


def test_adaptive_selector_targets_most_deteriorated_skill():
    controller = ReviewController(
        "adaptive_due",
        ("codebook", "registry"),
        total_steps=30,
        events=3,
        first_step=2,
        last_step=22,
        expansion_ratio=1.4,
        adaptive_opportunity_interval=5,
        loss_delta_threshold=0.05,
        min_gap=1,
    )
    controller.set_baseline({"codebook": 1.0, "registry": 1.0})
    controller.observe({"codebook": 1.01, "registry": 1.20})

    decision = controller.decide(2)
    assert decision.review
    assert decision.skill == "registry"
    assert decision.reason == "matched-first-endpoint"


def test_adaptive_schedule_spends_fixed_budget_by_deadline():
    controller = ReviewController(
        "adaptive_due",
        ("codebook", "registry"),
        total_steps=20,
        events=3,
        first_step=1,
        last_step=13,
        expansion_ratio=1.4,
        adaptive_opportunity_interval=3,
        loss_delta_threshold=10.0,
        min_gap=1,
    )
    controller.set_baseline({"codebook": 1.0, "registry": 1.0})
    for step in range(20):
        controller.decide(step)

    assert controller.used_events == 3
    assert len(controller.review_steps) == 3
    assert controller.review_steps[-1] == 13
