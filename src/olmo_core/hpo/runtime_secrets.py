"""Load optional API keys from the user's home directory."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "OPENAI_API_KEY_ENV_VAR",
    "OPENAI_HOME_KEY_FILE",
    "WANDB_API_KEY_ENV_VAR",
    "WANDB_HOME_KEY_FILE",
    "load_home_api_key",
    "load_openai_api_key",
    "load_runtime_secrets",
    "load_wandb_api_key",
]

WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"
WANDB_HOME_KEY_FILE = ".wandb_api_key"
OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_HOME_KEY_FILE = ".openai_api_key"


def load_home_api_key(*, env_var: str, filename: str) -> bool:
    """Populate ``env_var`` from ``~/<filename>`` when unset.

    :param env_var: Environment variable to populate.
    :param filename: Basename of the home-directory key file.

    :returns: True when the environment variable is set after this call.
    """
    if os.environ.get(env_var):
        return True
    path = Path.home() / filename
    if not path.is_file():
        return False
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        return False
    os.environ[env_var] = value
    return True


def load_wandb_api_key() -> bool:
    """Load ``WANDB_API_KEY`` from ``~/.wandb_api_key`` when unset."""

    return load_home_api_key(env_var=WANDB_API_KEY_ENV_VAR, filename=WANDB_HOME_KEY_FILE)


def load_openai_api_key() -> bool:
    """Load ``OPENAI_API_KEY`` from ``~/.openai_api_key`` when unset."""

    return load_home_api_key(env_var=OPENAI_API_KEY_ENV_VAR, filename=OPENAI_HOME_KEY_FILE)


def load_runtime_secrets() -> None:
    """Load any unset runtime API keys from the user's home directory."""

    load_wandb_api_key()
    load_openai_api_key()
