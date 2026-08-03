from __future__ import annotations

from pathlib import Path

import pytest

FARM = Path(__file__).resolve().parents[1] / "farmshare"


@pytest.mark.parametrize(
    "name",
    [
        "common.sh",
        "config.env",
        "launch.sh",
        "setup_venv.sh",
        "stage.sh",
        "stage_job.sbatch",
        "submit_from_laptop.sh",
        "sync_repo.sh",
        "train_job.sbatch",
    ],
)
def test_farmshare_files_exist(name: str) -> None:
    assert (FARM / name).is_file()


def test_skillit_arm_indices() -> None:
    launch = (FARM / "launch.sh").read_text(encoding="utf-8")
    assert "ARM_INDEX must be 0 or 1" in launch
    assert "probe" in launch and "deriv" in launch


def test_config_slug() -> None:
    config = (FARM / "config.env").read_text(encoding="utf-8")
    assert "EXPERIMENT_SLUG=\"skillit-370m\"" in config
    assert "OLMO_FLASH_ATTENTION" not in config


def test_common_disables_flash() -> None:
    common = (FARM / "common.sh").read_text(encoding="utf-8")
    assert "OLMO_FLASH_ATTENTION=0" in common
    assert "OLMO_ATTN_BACKEND=torch" in common
