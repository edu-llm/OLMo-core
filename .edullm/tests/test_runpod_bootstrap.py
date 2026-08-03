from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

STAGER = Path(__file__).resolve().parents[1] / "runpod" / "stage_inputs.py"


def load_stager():
    spec = importlib.util.spec_from_file_location("token_runpod_stager", STAGER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def credential_file(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "# generated fixture",
                "unset AWS_PROFILE",
                "export AWS_ACCESS_KEY_ID='test-access'",
                "export AWS_SECRET_ACCESS_KEY='test-secret'",
                "export AWS_SESSION_TOKEN='test-session'",
                "export AWS_REGION='us-east-1'",
                "export AWS_DEFAULT_REGION='us-east-1'",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def clear_aws(module, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in module.AWS_FILE_KEYS | set(module.AWS_PROVIDER_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_accepts_minted_session_and_destroys_it(tmp_path, monkeypatch) -> None:
    module = load_stager()
    clear_aws(module, monkeypatch)
    path = credential_file(tmp_path / "aws-session.env")
    module.load_credentials(path)
    assert module.os.environ["AWS_SESSION_TOKEN"] == "test-session"
    module.destroy_credentials(path)
    assert not path.exists()
    assert not any(module.os.environ.get(key) for key in module.AWS_CREDENTIAL_KEYS)


def test_cli_error_still_deletes_session(tmp_path, monkeypatch) -> None:
    module = load_stager()
    clear_aws(module, monkeypatch)
    path = credential_file(tmp_path / "aws-session.env")
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage_inputs.py", "--credentials-file", str(path), "--workers", "0"],
    )
    with pytest.raises(SystemExit):
        module.main()
    assert not path.exists()


def test_runpod_stager_accepts_only_kept_token_arms(monkeypatch) -> None:
    module = load_stager()
    kept = ("rho-1", "rel-ema-exp", "middle-ppl-token", "attention", "blade")
    for arm in kept:
        monkeypatch.setattr(
            sys,
            "argv",
            ["stage_inputs.py", "--arm", arm, "--dataset-version", "v1"],
        )
        assert module.parse_args().arm == arm

    monkeypatch.setattr(
        sys,
        "argv",
        ["stage_inputs.py", "--arm", "control", "--dataset-version", "v1"],
    )
    with pytest.raises(SystemExit):
        module.parse_args()

    launch = (STAGER.parent / "launch.sh").read_text(encoding="utf-8")
    assert 'ARM="${ARM:-attention}"' in launch
    for removed in (
        "control",
        "rel-ema-refhq",
        "middle-ppl-doc",
        "learnability-token",
        "learnability-doc",
        "reference",
    ):
        assert removed not in launch
