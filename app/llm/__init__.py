"""LLM client contracts and lazily loaded provider implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .protocol import ChatMessage, LLMClient

_CLIENT_EXPORTS = frozenset(
    {
        "DEEPSEEK_BASE_URL",
        "DEFAULT_DEEPSEEK_MODEL",
        "DeepSeekClient",
        "LLMConfigurationError",
        "LLMResponseError",
    }
)

__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "ChatMessage",
    "DeepSeekClient",
    "LLMClient",
    "LLMConfigurationError",
    "LLMResponseError",
]


def __getattr__(name: str) -> Any:
    if name not in _CLIENT_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(".client", __name__), name)
    globals()[name] = value
    return value
