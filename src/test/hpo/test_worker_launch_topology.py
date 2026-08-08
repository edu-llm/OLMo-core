import pytest

from olmo_core.hpo.worker import assert_single_process_topology, world_size_one_env


def test_trial_worker_bootstraps_world_size_one_subprocess():
    env = world_size_one_env(gpu=3, master_port=29501)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["WORLD_SIZE"] == "1"
    assert env["RANK"] == "0"
    assert env["LOCAL_RANK"] == "0"
    assert env["LOCAL_WORLD_SIZE"] == "1"
    assert env["MASTER_PORT"] == "29501"
    assert env["MASTER_ADDR"]  # a loopback address is set
    # A world-size-one env must pass the topology guard.
    assert_single_process_topology(env)


def test_outer_torchrun_is_forbidden():
    # An inherited multi-process WORLD_SIZE means we were launched under torchrun; forbid it.
    with pytest.raises(RuntimeError):
        assert_single_process_topology({"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"})
    # torchelastic markers also indicate an outer launcher, even at world size 1.
    with pytest.raises(RuntimeError):
        assert_single_process_topology({"WORLD_SIZE": "1", "TORCHELASTIC_RUN_ID": "abc"})


def test_env_strips_inherited_torchrun_markers():
    dirty = {"TORCHELASTIC_RUN_ID": "abc", "GROUP_RANK": "2", "WORLD_SIZE": "8"}
    env = world_size_one_env(gpu=0, master_port=29500, base_env=dirty)
    assert "TORCHELASTIC_RUN_ID" not in env
    assert env["WORLD_SIZE"] == "1"
    assert_single_process_topology(env)
