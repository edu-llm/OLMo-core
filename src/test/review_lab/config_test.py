from copy import deepcopy

from olmo_core.review_lab.config import ExperimentConfig


def test_fingerprint_ignores_execution_selection_and_paths():
    first = ExperimentConfig()
    second = deepcopy(first)
    second.conditions = ["uniform"]
    second.seeds = [1, 2, 3]
    second.data.path = "/a/different/machine/data.jsonl"
    second.training.output_dir = "/a/different/machine/runs"
    second.training.save_final_model = True
    assert first.fingerprint() == second.fingerprint()


def test_fingerprint_changes_for_scientific_settings():
    first = ExperimentConfig()
    second = deepcopy(first)
    second.review.events += 1
    assert first.fingerprint() != second.fingerprint()


def test_fingerprint_changes_for_model_backend():
    first = ExperimentConfig()
    second = deepcopy(first)
    second.model.backend = "hf_olmo"
    assert first.fingerprint() != second.fingerprint()


def test_buffer_delays_must_fall_inside_buffer():
    config = ExperimentConfig()
    config.training.buffer_steps = 60
    config.training.buffer_eval_delays = [15, 61]
    try:
        config.validate()
    except ValueError as error:
        assert "buffer_eval_delays" in str(error)
    else:
        raise AssertionError("Expected an invalid buffer delay to fail validation")
