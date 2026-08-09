"""Structured-output OpenAI-compatible transport for 5.6 Sol Centaur.

Brainlift preregisters the logical model id ``gpt-5.6-sol``. The default transport
routes that logical id through the TrueFoundry AI Gateway as
``openai-group/gpt-5.6-sol``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .centaur import AdvisorResponse, validate_action
from .runtime_secrets import load_openai_api_key

__all__ = [
    "CENTAUR_LOGICAL_MODEL",
    "CENTAUR_ROUTE_MODEL",
    "TRUEFOUNDRY_DEFAULT_BASE_URL",
    "OpenAICompatibleAdvisor",
    "build_openai_advisor",
    "resolve_centaur_base_url",
]

CENTAUR_LOGICAL_MODEL = "gpt-5.6-sol"
CENTAUR_ROUTE_MODEL = "openai-group/gpt-5.6-sol"
TRUEFOUNDRY_DEFAULT_BASE_URL = "https://gateway.truefoundry.ai/v1"

_ACTION_SCHEMA = {
    "name": "centaur_multi_action",
    "strict": True,
    "schema": {
        "type": "object",
        # OpenAI structured outputs require every property key to appear in required.
        # Optional fields are therefore nullable and stripped before local validation.
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["start", "resume", "ipbt_exploit", "ipbt_restart"],
            },
            "unit_config": {
                "type": ["array", "null"],
                "items": {"type": "number"},
            },
            "trial_id": {"type": ["string", "null"]},
            "donor_id": {"type": ["string", "null"]},
            "target_slot_id": {"type": ["string", "null"]},
            "restart_id": {"type": ["string", "null"]},
        },
        "required": [
            "kind",
            "unit_config",
            "trial_id",
            "donor_id",
            "target_slot_id",
            "restart_id",
        ],
        "additionalProperties": False,
    },
}


def _strip_null_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop JSON-null optional fields before local Centaur action validation."""

    return {key: value for key, value in payload.items() if value is not None}


def resolve_centaur_base_url(base_url: Optional[str] = None) -> str:
    """Resolve the OpenAI-compatible base URL, defaulting to TrueFoundry."""

    if base_url:
        return str(base_url).rstrip("/")
    env_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return TRUEFOUNDRY_DEFAULT_BASE_URL


class OpenAICompatibleAdvisor:
    """Call an OpenAI-compatible chat endpoint with a strict JSON action schema."""

    def __init__(
        self,
        *,
        model: str = CENTAUR_LOGICAL_MODEL,
        route_model: str = CENTAUR_ROUTE_MODEL,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
        client: Any = None,
    ) -> None:
        if model != CENTAUR_LOGICAL_MODEL:
            raise ValueError(f"the preregistered Centaur model is {CENTAUR_LOGICAL_MODEL}")
        if not route_model:
            raise ValueError("route_model must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if client is None:
            from openai import OpenAI

            if api_key_env == "OPENAI_API_KEY":
                load_openai_api_key()
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is required for the Centaur advisor")
            client = OpenAI(
                api_key=api_key,
                base_url=resolve_centaur_base_url(base_url),
                timeout=timeout_seconds,
                max_retries=0,
            )
        self.client = client
        self.model = model
        self.route_model = route_model

    def advise(self, state: dict[str, Any]) -> AdvisorResponse:
        """Request one legal action; transport or schema failures propagate fail-closed."""

        started = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.route_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Choose exactly one legal HPO controller action from the supplied "
                        "deterministic state. Return only the structured action. Never invent "
                        "trial, donor, slot, or restart identifiers."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(state, sort_keys=True, separators=(",", ":")),
                },
            ],
            response_format={"type": "json_schema", "json_schema": _ACTION_SCHEMA},
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = completion.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise RuntimeError("Centaur endpoint returned no structured action")
        action = validate_action(_strip_null_fields(json.loads(content)))
        reported = str(getattr(completion, "model", "") or "")
        version = str(getattr(completion, "system_fingerprint", None) or reported or "truefoundry")
        # Brainlift's RequiredModelAdvisor checks the logical preregistered id.
        return AdvisorResponse(
            action=action,
            raw_text=content,
            model=self.model,
            version=version,
            latency_ms=latency_ms,
        )


def build_openai_advisor(**kwargs: Any) -> OpenAICompatibleAdvisor:
    """Concrete advisor factory referenced by the committed controller specs."""

    return OpenAICompatibleAdvisor(**kwargs)
