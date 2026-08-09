"""Structured-output OpenAI-compatible transport for 5.6 Sol Centaur."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .centaur import AdvisorResponse, validate_action

__all__ = ["OpenAICompatibleAdvisor", "build_openai_advisor"]

_ACTION_SCHEMA = {
    "name": "centaur_multi_action",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["start", "resume", "ipbt_exploit", "ipbt_restart"],
            },
            "unit_config": {"type": "array", "items": {"type": "number"}},
            "trial_id": {"type": "string"},
            "donor_id": {"type": "string"},
            "target_slot_id": {"type": "string"},
            "restart_id": {"type": "string"},
        },
        "required": ["kind"],
        "additionalProperties": False,
    },
}


class OpenAICompatibleAdvisor:
    """Call an OpenAI-compatible chat endpoint with a strict JSON action schema."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-sol",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        timeout_seconds: float = 120.0,
        client: Any = None,
    ) -> None:
        if model != "gpt-5.6-sol":
            raise ValueError("the preregistered Centaur model is gpt-5.6-sol")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if client is None:
            from openai import OpenAI

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} is required for the Centaur advisor")
            resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
            client = OpenAI(
                api_key=api_key,
                base_url=resolved_base_url,
                timeout=timeout_seconds,
                max_retries=0,
            )
        self.client = client
        self.model = model

    def advise(self, state: dict[str, Any]) -> AdvisorResponse:
        """Request one legal action; transport or schema failures propagate fail-closed."""

        started = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
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
        action = validate_action(json.loads(content))
        version = str(getattr(completion, "system_fingerprint", None) or "openai-compatible")
        return AdvisorResponse(
            action=action,
            raw_text=content,
            model=str(getattr(completion, "model", self.model)),
            version=version,
            latency_ms=latency_ms,
        )


def build_openai_advisor(**kwargs: Any) -> OpenAICompatibleAdvisor:
    """Concrete advisor factory referenced by the committed controller specs."""

    return OpenAICompatibleAdvisor(**kwargs)
