import re
import shlex
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SPEC_DIRECTORY = REPOSITORY_ROOT / ".edullm"
ARM_SCRIPTS = {
    "base": "src/scripts/train/engram_experiment/base_moe.py",
    "engram": "src/scripts/train/engram_experiment/engram_moe.py",
    "lngram": "src/scripts/train/engram_experiment/lngram_moe.py",
}
EXPECTED_FIELDS = {
    "schema_version",
    "workload_profile",
    "suggested_compute",
    "command",
}
FORBIDDEN_PATTERNS = (
    r"\bbeaker\b",
    r"\baws\b",
    r"\bboto3\b",
    r"\bs3://",
    r"\b(?:dataset|data)[-_ ]?path\b",
    r"\bstag(?:e|ing)\b",
    r"\bevals?\b",
)


def load_spec(arm: str) -> dict:
    with (SPEC_DIRECTORY / f"run-{arm}.yaml").open(encoding="utf-8") as spec_file:
        document = yaml.safe_load(spec_file)
    assert isinstance(document, dict)
    return document


@pytest.mark.parametrize(("arm", "script"), ARM_SCRIPTS.items())
def test_run_spec_is_an_immutable_single_arm_contract(arm, script):
    spec = load_spec(arm)

    assert set(spec) == EXPECTED_FIELDS
    assert spec["schema_version"] == 1
    assert spec["workload_profile"] == "olmo-core-train"
    assert spec["suggested_compute"] == "gpu-8xa100"

    outer_argv = shlex.split(spec["command"])
    assert outer_argv[:2] == ["bash", "-lc"]
    assert len(outer_argv) == 3
    assert shlex.split(outer_argv[2]) == [
        "python",
        "-m",
        "torch.distributed.run",
        "--nproc-per-node=8",
        "--standalone",
        script,
        "train",
        "$EDULLM_RUN_ID",
        "--param-dtype",
        "bfloat16",
        "--save-folder",
        "$EDULLM_CHECKPOINT_DIR",
    ]


@pytest.mark.parametrize("arm", ARM_SCRIPTS)
def test_run_spec_uses_only_platform_environment_and_no_side_channels(arm):
    command = load_spec(arm)["command"]

    assert re.findall(r"\$[A-Z][A-Z0-9_]*", command) == [
        "$EDULLM_RUN_ID",
        "$EDULLM_CHECKPOINT_DIR",
    ]
    for pattern in FORBIDDEN_PATTERNS:
        assert re.search(pattern, command, flags=re.IGNORECASE) is None
