"""Attention-backend selection for frontload-cl train scripts."""

from __future__ import annotations

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.attention.flash_attn_api import (
    has_flash_attn_2,
    has_flash_attn_3,
    has_flash_attn_4,
)

from .corpus import Refusal, Stage


def resolve_attn_backend(name: str) -> AttentionBackendName:
    """
    Parse ``--attn-backend`` and refuse early if the requested package is missing.

    Flash-3/4 also need Hopper/Blackwell at runtime; that check happens when the
    attention module builds. Here we only catch "package not installed."
    """
    try:
        backend = AttentionBackendName(name)
    except ValueError as exc:
        known = ", ".join(b.value for b in AttentionBackendName)
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            f"unknown attn backend {name!r}; known: {known}",
        ) from exc

    if backend == AttentionBackendName.flash_2 and not has_flash_attn_2():
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "attn backend flash_2 requires the flash-attn package in the image "
            "(see .edullm/Dockerfile). Use --attn-backend torch to fall back to SDPA.",
        )
    if backend == AttentionBackendName.flash_3 and not has_flash_attn_3():
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "attn backend flash_3 is unavailable (need flash-attn 3 + Hopper GPU).",
        )
    if backend == AttentionBackendName.flash_4 and not has_flash_attn_4():
        raise Refusal(
            Stage.THE_CONFIG_WOULD_NOT_BUILD,
            "attn backend flash_4 is unavailable (need flash-attn 4 + Blackwell GPU).",
        )
    return backend
