import numpy as np
import pytest

from olmo_core.hpo.config import (
    FidelityConfig,
    HpoControllerConfig,
    SearchDimConfig,
    SearchSpaceConfig,
)


def test_default_transformer_space_is_within_ftpfn_contract():
    space = SearchSpaceConfig.default_transformer_space().build()
    # The plan's shared optimization space (<= 10 dims).
    assert space.ndim == 9
    assert space.ndim <= 10
    assert "lr" in space.names and "weight_decay" in space.names and "max_grad_norm" in space.names


def test_search_space_config_round_trip_and_build():
    cfg = SearchSpaceConfig(
        dims=[
            SearchDimConfig(name="lr", low=1e-4, high=1e-2, log=True),
            SearchDimConfig(name="wd", low=0.0, high=0.3, log=False),
        ]
    )
    rebuilt = SearchSpaceConfig.from_dict(cfg.as_dict())
    assert rebuilt == cfg
    space = cfg.build()
    unit = space.to_unit({"lr": 1e-3, "wd": 0.15})
    assert np.all((unit >= 0) & (unit <= 1))


def test_fidelity_config_requires_increasing_rungs():
    ok = FidelityConfig(rungs=[1024, 2048, 4096])
    assert ok.min_fidelity == 1024
    assert ok.target_fidelity == 4096
    with pytest.raises(Exception):
        FidelityConfig(rungs=[1024, 1024]).validate()
    with pytest.raises(Exception):
        FidelityConfig(rungs=[4096, 2048]).validate()


def test_fidelity_config_rejects_nonpositive_or_noninteger_rungs():
    for rungs in ([0, 1024], [-1, 1024], [1.5, 2.5], [True, 1024]):
        with pytest.raises(Exception):
            FidelityConfig(rungs=rungs)


def test_controller_config_validates_population_and_ratio():
    good = HpoControllerConfig(worker_count=8, population_size=16, llm_ratio=0.3)
    good.validate()
    with pytest.raises(Exception):
        HpoControllerConfig(worker_count=8, population_size=4).validate()  # pop < workers
    with pytest.raises(Exception):
        HpoControllerConfig(worker_count=8, population_size=16, llm_ratio=1.5).validate()


def test_default_controller_uses_brainlift_sol_ratio():
    config = HpoControllerConfig()
    assert config.llm_ratio == 0.3
    assert config.llm_warmup == 0


def test_controller_config_merge_override():
    cfg = HpoControllerConfig(worker_count=8, population_size=16, llm_ratio=0.3)
    merged = cfg.merge(["llm_ratio=0.5"])
    assert merged.llm_ratio == 0.5
    assert cfg.llm_ratio == 0.3  # original untouched
