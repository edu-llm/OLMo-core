"""The hard-stop/resume supervisor loop in train_job.sbatch.

FarmShare has no external supervisor, so train_job.sbatch loops: every
non-final permanent checkpoint makes the training worker stop cleanly after
its synchronous task-loss eval (to clear compiled CUDA state), leaving a
restart request behind, and the loop relaunches with RECOVERY_MODE=resume.

`assert_smoke.py` requires "checkpoint boundary reached" to appear in the log,
i.e. that this loop actually cycled at least once. It never has. Four GPU
attempts at the smoke have all died at or before the step-0 evaluation, so
everything from the first training step onward -- this loop, checkpoint
retention, the durable marker -- is entirely unexercised.

The loop is shell, and its behaviour does not need a GPU: with launch.sh,
common.sh and setup_venv.sh stubbed, the real script can be driven through
every branch in milliseconds. That is what these tests do.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SBATCH = Path(__file__).resolve().parents[1] / "farmshare" / "train_job.sbatch"
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def _harness(tmp_path: Path, launch_body: str) -> tuple[Path, dict[str, str]]:
    """Build a run directory whose scripts/ holds stubs, and return its env."""
    run_dir = tmp_path / "run"
    scripts = run_dir / "scripts"
    scripts.mkdir(parents=True)
    arm = run_dir / "runs" / "linear_n10-mtld-constant-seed0"
    (arm / "progress").mkdir(parents=True)

    # common.sh is sourced twice by the real script; RUN_ROOT is what it
    # contributes that the loop depends on.
    (scripts / "common.sh").write_text(
        'RUN_ROOT="${RUN_DIR}/runs"\nPYTHON=python3\nexport RUN_ROOT PYTHON\n',
        encoding="utf-8",
        newline="\n",
    )
    (scripts / "setup_venv.sh").write_text(
        '#!/usr/bin/env bash\necho "venv ready (stub)"\n', encoding="utf-8", newline="\n"
    )
    (scripts / "launch.sh").write_text(launch_body, encoding="utf-8", newline="\n")

    env = {
        "RUN_DIR": str(run_dir),
        "SCRIPTS_DIR": str(scripts),
        "PACING": "linear_n10",
        "METRIC": "mtld",
        "LR_SCHEDULE": "constant",
        "SEED": "0",
        "REPO_DIR": str(run_dir / "OLMo-core"),
        "PATH": "/usr/bin:/bin",
    }
    return run_dir, env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    assert BASH is not None
    return subprocess.run(
        [BASH, str(SBATCH)], capture_output=True, text=True, env=env, timeout=120
    )


# A stub launch.sh that writes the restart request on its first N attempts,
# tracked through a counter file, then stops -- which is exactly the shape of
# a real run whose ladder has one intermediate checkpoint.
_COUNTING_LAUNCH = """#!/usr/bin/env bash
set -Eeuo pipefail
counter="${RUN_DIR}/attempts"
n=0
[[ -f "${counter}" ]] && n=$(cat "${counter}")
n=$((n + 1))
echo "${n}" > "${counter}"
echo "stub launch.sh attempt ${n} RECOVERY_MODE=${RECOVERY_MODE}"
if (( n <= RESTARTS_BEFORE_FINISH )); then
  mkdir -p "${RUN_DIR}/runs/linear_n10-mtld-constant-seed0/progress"
  echo '{}' > "${RUN_DIR}/runs/linear_n10-mtld-constant-seed0/progress/restart_after_checkpoint.json"
fi
"""


def test_one_restart_then_finish(tmp_path: Path) -> None:
    """The smoke's own shape: ladder {0, 125, 250} stops once at 125."""
    run_dir, env = _harness(tmp_path, _COUNTING_LAUNCH)
    env["RESTARTS_BEFORE_FINISH"] = "1"
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    # The two strings assert_smoke.py greps the log for.
    assert "checkpoint boundary reached" in out
    assert "training run finished" in out
    assert out.count("checkpoint boundary reached") == 1
    assert (run_dir / "attempts").read_text().strip() == "2"


def test_the_resumed_attempt_switches_recovery_mode(tmp_path: Path) -> None:
    """The first attempt must be fresh and every later one resume. Relaunching
    fresh would refuse on existing state, or worse, start over."""
    _, env = _harness(tmp_path, _COUNTING_LAUNCH)
    env["RESTARTS_BEFORE_FINISH"] = "2"
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "attempt 1 RECOVERY_MODE=fresh" in result.stdout
    assert "attempt 2 RECOVERY_MODE=resume" in result.stdout
    assert "attempt 3 RECOVERY_MODE=resume" in result.stdout


def test_max_restarts_is_enforced(tmp_path: Path) -> None:
    """A worker that requests a restart forever must stop, not spin for the
    whole wall-clock allocation."""
    _, env = _harness(tmp_path, _COUNTING_LAUNCH)
    env["RESTARTS_BEFORE_FINISH"] = "999"
    env["MAX_RESTARTS"] = "4"
    result = _run(env)
    assert result.returncode == 3
    assert "exceeded MAX_RESTARTS=4" in result.stderr
    assert "training run finished" not in result.stdout


def test_a_failing_launch_aborts_rather_than_reporting_success(tmp_path: Path) -> None:
    """The dangerous misreading: launch.sh dies without leaving a restart
    request, and the loop calls that "training run finished". `set -e` is what
    prevents it, so it is pinned here -- all four real failures so far were
    non-zero exits with no restart request."""
    _, env = _harness(
        tmp_path,
        '#!/usr/bin/env bash\necho "stub launch.sh failing"\nexit 1\n',
    )
    result = _run(env)
    assert result.returncode != 0
    assert "training run finished" not in result.stdout
    assert "checkpoint boundary reached" not in result.stdout


def test_a_stale_restart_request_does_not_cause_a_phantom_restart(tmp_path: Path) -> None:
    """A request left behind by an earlier job must not make this one loop.
    The `rm -f` at the top of each attempt is what guarantees it."""
    run_dir, env = _harness(
        tmp_path,
        '#!/usr/bin/env bash\necho "stub launch.sh, leaves no request"\n',
    )
    stale = run_dir / "runs" / "linear_n10-mtld-constant-seed0" / "progress"
    (stale / "restart_after_checkpoint.json").write_text("{}", encoding="utf-8")
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "training run finished" in result.stdout
    assert "checkpoint boundary reached" not in result.stdout


def test_the_request_is_cleared_before_each_attempt(tmp_path: Path) -> None:
    """Otherwise the loop could not tell a fresh request from the last one and
    would restart forever on a single stop."""
    run_dir, env = _harness(tmp_path, _COUNTING_LAUNCH)
    env["RESTARTS_BEFORE_FINISH"] = "1"
    result = _run(env)
    assert result.returncode == 0, result.stdout + result.stderr
    marker = (
        run_dir / "runs" / "linear_n10-mtld-constant-seed0" / "progress"
        / "restart_after_checkpoint.json"
    )
    assert not marker.exists()
