"""Tests for home-directory runtime secret loading."""

from __future__ import annotations

import os

from olmo_core.hpo.runtime_secrets import (
    OPENAI_HOME_KEY_FILE,
    WANDB_HOME_KEY_FILE,
    load_home_api_key,
    load_openai_api_key,
    load_runtime_secrets,
    load_wandb_api_key,
)


def test_load_home_api_key_reads_file_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("olmo_core.hpo.runtime_secrets.Path.home", lambda: tmp_path)
    key_path = tmp_path / WANDB_HOME_KEY_FILE
    key_path.write_text("  local-wandb-key  \n", encoding="utf-8")

    assert load_home_api_key(env_var="WANDB_API_KEY", filename=WANDB_HOME_KEY_FILE) is True
    assert os.environ["WANDB_API_KEY"] == "local-wandb-key"


def test_load_home_api_key_preserves_existing_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "already-set")
    monkeypatch.setattr("olmo_core.hpo.runtime_secrets.Path.home", lambda: tmp_path)
    (tmp_path / OPENAI_HOME_KEY_FILE).write_text("file-key\n", encoding="utf-8")

    assert load_openai_api_key() is True
    assert os.environ["OPENAI_API_KEY"] == "already-set"


def test_load_runtime_secrets_loads_both_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("olmo_core.hpo.runtime_secrets.Path.home", lambda: tmp_path)
    (tmp_path / WANDB_HOME_KEY_FILE).write_text("wandb\n", encoding="utf-8")
    (tmp_path / OPENAI_HOME_KEY_FILE).write_text("openai\n", encoding="utf-8")

    load_runtime_secrets()
    assert os.environ["WANDB_API_KEY"] == "wandb"
    assert os.environ["OPENAI_API_KEY"] == "openai"


def test_load_home_api_key_returns_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr("olmo_core.hpo.runtime_secrets.Path.home", lambda: tmp_path)

    assert load_wandb_api_key() is False
    assert "WANDB_API_KEY" not in os.environ
