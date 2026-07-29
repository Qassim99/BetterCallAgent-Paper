"""Provider interfaces and the TLS-verified OpenAI-compatible client."""

from bettercallagent.providers.openai_compatible import (
    ChatProvider,
    OpenAICompatibleProvider,
    ProviderError,
    parse_completion_envelope,
    parse_json_object,
)

__all__ = [
    "ChatProvider",
    "OpenAICompatibleProvider",
    "ProviderError",
    "parse_completion_envelope",
    "parse_json_object",
]
