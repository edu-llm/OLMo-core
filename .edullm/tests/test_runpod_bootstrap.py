from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

STAGER = Path(__file__).resolve().parents[1] / "runpod" / "stage_inputs.py"
STAGING_PLAN = Path(__file__).resolve().parents[1] / "staging_plan.py"


def load_stager():
    spec = importlib.util.spec_from_file_location("skillit_runpod_stager", STAGER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_staging_plan():
    spec = importlib.util.spec_from_file_location("skillit_staging_plan", STAGING_PLAN)
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


def test_weighted_staging_selects_smallest_sufficient_prefix() -> None:
    module = load_staging_plan()
    selected, tokens = module.select_paths(
        ("a", "b", "c"),
        required_tokens=60,
        headroom=1.25,
        size_of={"a": 160, "b": 160, "c": 160}.__getitem__,
    )
    assert selected == ("a", "b")
    assert tokens == 80


def test_smoke_file_cap_overrides_weighted_demand() -> None:
    module = load_staging_plan()
    selected, tokens = module.select_paths(
        ("a", "b"),
        required_tokens=1_000,
        headroom=1.25,
        size_of={"a": 160, "b": 160}.__getitem__,
        max_files=1,
    )
    assert selected == ("a",)
    assert tokens == 40
