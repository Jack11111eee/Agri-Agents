"""Small injectable interface used by the diagnosis pipeline."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence, TypedDict, runtime_checkable


class ChatMessage(TypedDict):
    """A text-only message accepted by the constrained generation client."""

    role: Literal["system", "user"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    """Generate one JSON-formatted completion for a sequence of messages."""

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Return the provider's JSON text or raise a provider/client error."""
        ...
