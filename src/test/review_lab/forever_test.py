from olmo_core.review_lab.forever import ForeverClock


def test_forever_calibrates_and_triggers_first_model_day() -> None:
    clock = ForeverClock(calibration_steps=4, trigger_days=(1, 2, 4))
    assert clock.observe_update(1.0) == []
    assert clock.observe_update(1.0) == []
    assert clock.observe_update(1.0) == []
    assert clock.observe_update(1.0) == [1]
    assert clock.model_day == 4.0
    assert clock.mu0 == 1.0


def test_forever_uses_accumulated_parameter_change_not_step_number() -> None:
    clock = ForeverClock(calibration_steps=2, trigger_days=(1, 2, 4))
    assert clock.observe_update(2.0) == []
    assert clock.observe_update(2.0) == [1]
    assert clock.observe_update(1.0) == []
    assert clock.observe_update(3.0) == [2]
    assert clock.triggered_days == [1, 2]


def test_forever_replay_scale_is_clipped() -> None:
    clock = ForeverClock(calibration_steps=1, trigger_days=())
    clock.observe_update(1.0)
    clock.mu = 10.0
    assert clock.replay_scale == 3.0
    clock.mu = 0.1
    assert clock.replay_scale == 0.5
