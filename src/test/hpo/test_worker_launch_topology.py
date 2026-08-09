import pytest

from olmo_core.hpo.worker import (
    assert_single_process_topology,
    assert_worker_topology,
    finalist_distributed_env,
    should_emit_worker_result,
    world_size_one_env,
)


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


def test_finalist_distributed_topology_requires_explicit_valid_torchrun():
    parent_env = finalist_distributed_env(
        list(range(8)),
        world_size=8,
        base_env={"WORLD_SIZE": "99", "TORCHELASTIC_RUN_ID": "stale"},
    )
    assert parent_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert parent_env["EDULLM_FINALIST_CONTINUATION"] == "1"
    assert "WORLD_SIZE" not in parent_env
    assert "TORCHELASTIC_RUN_ID" not in parent_env

    rank_env = {
        **parent_env,
        "WORLD_SIZE": "8",
        "RANK": "3",
        "LOCAL_RANK": "3",
        "LOCAL_WORLD_SIZE": "8",
        "TORCHELASTIC_RUN_ID": "finalist",
    }
    assert_worker_topology(rank_env)
    with pytest.raises(RuntimeError, match="explicit expected world size"):
        assert_worker_topology({**rank_env, "WORLD_SIZE": "4"})
    with pytest.raises(RuntimeError, match="torchrun"):
        assert_worker_topology(
            {key: value for key, value in rank_env.items() if key != "TORCHELASTIC_RUN_ID"}
        )


def test_only_finalist_rank_zero_emits_controller_result():
    ordinary = {"WORLD_SIZE": "1", "RANK": "0"}
    finalist = {"EDULLM_FINALIST_CONTINUATION": "1"}
    assert should_emit_worker_result(ordinary)
    assert should_emit_worker_result({**finalist, "RANK": "0"})
    assert not should_emit_worker_result({**finalist, "RANK": "7"})
