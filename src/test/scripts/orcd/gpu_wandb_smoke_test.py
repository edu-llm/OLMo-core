import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_ROOT = Path("src/scripts/orcd")


def load_smoke_module():
    path = SCRIPT_ROOT / "gpu_wandb_smoke.py"
    spec = importlib.util.spec_from_file_location("gpu_wandb_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calculate_tflops():
    smoke = load_smoke_module()
    assert smoke.calculate_tflops(1000, 0.002) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("matrix_size", "elapsed_seconds"),
    [(0, 1.0), (1, 0.0)],
)
def test_calculate_tflops_rejects_nonpositive_inputs(matrix_size, elapsed_seconds):
    smoke = load_smoke_module()
    with pytest.raises(ValueError):
        smoke.calculate_tflops(matrix_size, elapsed_seconds)


def test_job_requests_bounded_l40s_and_non_main_checkout():
    text = (SCRIPT_ROOT / "gpu_wandb_smoke.sbatch").read_text(encoding="utf-8")

    assert "#SBATCH -p mit_normal_gpu" in text
    assert "#SBATCH -G l40s:1" in text
    assert "#SBATCH -t 00:05:00" in text
    assert "#SBATCH -c 2" in text
    assert "#SBATCH --mem=8G" in text
    assert 'EDULLM_BRANCH="$(git -C "$EDULLM_REPO_ROOT" branch --show-current)"' in text
    assert 'test -n "$EDULLM_BRANCH"' in text
    assert 'test "$EDULLM_BRANCH" != main' in text


def test_job_uses_private_wandb_config_and_routes_logs_to_scratch():
    text = (SCRIPT_ROOT / "gpu_wandb_smoke.sbatch").read_text(encoding="utf-8")

    assert 'source "$WANDB_ENV"' in text
    assert ': "${WANDB_API_KEY:?W&B key is not configured}"' in text
    assert 'export WANDB_DIR="$EDULLM_SCRATCH/wandb"' in text
    assert 'exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1' in text
    assert "WANDB_API_KEY=" not in text
    assert "set -x" not in text


def test_smoke_logs_expected_wandb_metrics():
    text = (SCRIPT_ROOT / "gpu_wandb_smoke.py").read_text(encoding="utf-8")

    for metric in (
        "smoke/gpu_available",
        "smoke/matmul_ms",
        "smoke/tflops",
        "smoke/result_mean",
        "smoke/gpu_memory_allocated_mib",
        "smoke/gpu_memory_reserved_mib",
        "smoke/passed",
    ):
        assert metric in text
    assert 'mode="online"' in text
    assert '"git_commit": os.environ.get("EDULLM_COMMIT_SHA")' in text
