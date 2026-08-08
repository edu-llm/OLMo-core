import json
import subprocess

import pytest

from olmo_core.train.benchmark_harness import (
    BenchmarkArm,
    counterbalanced_subprocess_benchmark,
    nsys_profile_command,
    source_hash,
)


def test_counterbalanced_harness_isolates_processes_and_verifies_hash_backend(tmp_path):
    source = tmp_path / "mixer.py"
    source.write_text("backend = 'quaternion'\n", encoding="utf-8")
    expected_hash = source_hash([source])
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        payload = {
            "backend": kwargs["env"]["OLMO_BENCH_EXPECTED_BACKEND"],
            "source_hash": kwargs["env"]["OLMO_BENCH_SOURCE_HASH"],
            "warmup_steps": 20,
            "measured_steps": 50,
            "median_step_seconds": 1.0,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    arms = [
        BenchmarkArm("quaternion", ("python", "bench.py", "quaternion"), "quaternion"),
        BenchmarkArm("chunked", ("python", "bench.py", "chunked"), "chunked"),
    ]
    results = counterbalanced_subprocess_benchmark(
        arms,
        source_paths=[source],
        runner=runner,
    )

    assert [result.arm for result in results] == [
        "quaternion",
        "chunked",
        "chunked",
        "quaternion",
    ]
    assert all(result.source_hash == expected_hash for result in results)
    assert all(kwargs["start_new_session"] is True for _, kwargs in calls)
    assert all(kwargs["check"] is True for _, kwargs in calls)


def test_harness_fails_closed_on_backend_or_hash_mismatch(tmp_path):
    source = tmp_path / "mixer.py"
    source.write_text("pass\n", encoding="utf-8")

    def runner(command, **_kwargs):
        payload = {
            "backend": "eager-fallback",
            "source_hash": "wrong",
            "warmup_steps": 20,
            "measured_steps": 50,
            "median_step_seconds": 1.0,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    with pytest.raises(RuntimeError, match="backend proof"):
        counterbalanced_subprocess_benchmark(
            [BenchmarkArm("compiled", ("python", "bench.py"), "compiled")],
            source_paths=[source],
            runner=runner,
        )


def test_nsys_profile_command_is_local_and_delayed_until_after_warmup(tmp_path):
    command = nsys_profile_command(
        ("python", "bench.py", "--backend=quaternion"),
        output=tmp_path / "quaternion-profile",
        delay_seconds=30,
    )

    assert command[:4] == ("nsys", "profile", "--trace=cuda,nvtx,osrt", "--delay=30")
    assert f"--output={tmp_path / 'quaternion-profile'}" in command
    assert command[-3:] == ("python", "bench.py", "--backend=quaternion")
    assert not {"aws", "edullm", "sbatch"}.intersection(command)
