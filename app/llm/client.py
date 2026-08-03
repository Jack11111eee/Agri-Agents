"""DeepSeek client built on the OpenAI-compatible Python SDK."""

from __future__ import annotations

import os
from typing import Any, Protocol, Sequence

from openai import OpenAI

from .protocol import ChatMessage

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
_ALLOWED_ROLES = frozenset({"system", "user"})


class LLMConfigurationError(RuntimeError):
    """Raised when the DeepSeek client cannot be configured safely."""


class LLMResponseError(RuntimeError):
    """Raised when DeepSeek returns no usable completion text."""


class _CompletionsAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ChatAPI(Protocol):
    completions: _CompletionsAPI


class _OpenAICompatibleClient(Protocol):
    chat: _ChatAPI


class DeepSeekClient:
    """Synchronous JSON-mode DeepSeek adapter with an injectable SDK client."""

    def __init__(
        self,
        *,
        sdk_client: _OpenAICompatibleClient | None = None,
        model: str = DEFAULT_DEEPSEEK_MODEL,
    ) -> None:
        if not model.strip():
            raise LLMConfigurationError("DeepSeek model must not be blank")

        if sdk_client is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise LLMConfigurationError("DEEPSEEK_API_KEY is required")
            sdk_client = OpenAI(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
                timeout=DEFAULT_TIMEOUT_SECONDS,
                max_retries=DEFAULT_MAX_RETRIES,
            )

        self._sdk_client = sdk_client
        self._model = model

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Request JSON output; provider transport errors intentionally bubble up."""
        normalized_messages = self._normalize_messages(messages)
        response = self._sdk_client.chat.completions.create(
            model=self._model,
            messages=normalized_messages,
            response_format={"type": "json_object"},
        )

        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMResponseError("DeepSeek returned no completion choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("DeepSeek returned an empty completion")
        return content

    def close(self) -> None:
        """Release the underlying SDK transport and connection pool."""
        close = getattr(self._sdk_client, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _normalize_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
        if not messages:
            raise ValueError("At least one LLM message is required")

        normalized: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"Unsupported LLM message role: {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("LLM message content must not be blank")
            normalized.append({"role": role, "content": content})
        return normalized
