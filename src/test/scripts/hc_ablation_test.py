"""
Tests for the hyper-connections ablation script, `src/scripts/ablations/hc_ablation.py`.

CPU only. The point of these is the two things a specification script can get wrong quietly: an
eval task name that does not exist, which fails hours into a GPU run, and an arm whose budget has
drifted away from the others, which turns the comparison into no comparison at all.
"""

import importlib.util

import pytest

from olmo_core.eval.task_groups import FULL_TASKS
from olmo_core.nn.hyper_connections import ResidualMixerType

spec = importlib.util.spec_from_file_location("hc_ablation", "src/scripts/ablations/hc_ablation.py")
if spec is None or spec.loader is None:
    raise ImportError("Could not load hc_ablation.py")
hc_ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hc_ablation)


@pytest.mark.parametrize("task", hc_ablation.CLASSIC_TASKS + hc_ablation.MATH_REASONING_TASKS)
def test_eval_task_names_exist(task: str):
    # A name that OLMo-core does not know is only discovered when the evaluator is built, which
    # on a real run is after a GPU has been paid for.
    assert task in FULL_TASKS


def test_eval_suites_are_the_expected_shape():
    assert len(hc_ablation.CLASSIC_TASKS) == 5
    assert len(hc_ablation.MATH_REASONING_TASKS) == 8
    assert not set(hc_ablation.CLASSIC_TASKS) & set(hc_ablation.MATH_REASONING_TASKS)


def test_arms_are_the_six_from_the_spec():
    assert [arm.name for arm in hc_ablation.ARMS] == [
        "baseline",
        "hc_unconstrained",
        "mhc_sinkhorn",
        "mhc_lite",
        "kromhc",
        "mhc_identity",
    ]
    assert {arm.mixer for arm in hc_ablation.ARMS} == {None, *ResidualMixerType}


def test_every_arm_shares_the_same_budget():
    train_modules = [hc_ablation.build_train_module_config(arm) for arm in hc_ablation.ARMS]
    first = train_modules[0].as_config_dict()
    for arm, config in zip(hc_ablation.ARMS, train_modules):
        assert config.as_config_dict() == first, f"arm {arm.name} has a different train module"

    models = [hc_ablation.build_model_config(arm, model_size="tiny") for arm in hc_ablation.ARMS]
    for arm, config in zip(hc_ablation.ARMS, models):
        assert config.init_seed == hc_ablation.INIT_SEED, f"arm {arm.name} has a different seed"
        assert config.d_model == models[0].d_model
        assert config.n_layers == models[0].n_layers
        assert config.vocab_size == models[0].vocab_size


@pytest.mark.parametrize("arm", hc_ablation.ARMS, ids=lambda arm: arm.name)
def test_routing_param_counts_match_the_documented_table(arm):
    expected_per_sublayer = {
        None: 0,
        ResidualMixerType.identity: 8,
        ResidualMixerType.unconstrained: 24,
        ResidualMixerType.sinkhorn: 24,
        ResidualMixerType.birkhoff: 32,
        ResidualMixerType.kronecker: 12,
    }[arm.mixer]

    config = hc_ablation.build_model_config(arm, model_size="tiny")

    assert config.num_routing_params == 2 * config.n_layers * expected_per_sublayer
    if arm.hyper_connection is not None:
        assert arm.hyper_connection.num_params() == expected_per_sublayer


def test_dry_run_builds_every_arm():
    assert hc_ablation.dry_run("tiny") == 0
