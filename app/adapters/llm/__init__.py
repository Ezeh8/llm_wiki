from anthropic import AsyncAnthropic
from langsmith.wrappers import wrap_anthropic, wrap_openai
from openai import AsyncOpenAI

from app.adapters.llm.base import (
    LLMAdapter,
    LLMError,
    LLMGenerationError,
    LLMParseError,
)
from app.adapters.llm.claude import ClaudeAdapter
from app.adapters.llm.llama import LlamaAdapter
from app.adapters.llm.openai_provider import OpenAIAdapter
from app.config import get_settings

__all__ = [
    "LLMAdapter",
    "LLMError",
    "LLMGenerationError",
    "LLMParseError",
    "ClaudeAdapter",
    "OpenAIAdapter",
    "LlamaAdapter",
    "build_llm_adapter",
]


def build_llm_adapter() -> LLMAdapter:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    common = {
        "model": settings.llm_model_name,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }

    if provider == "claude":
        client = wrap_anthropic(AsyncAnthropic(api_key=settings.llm_api_key or None))
        return ClaudeAdapter(client, **common)
    if provider == "openai":
        client = wrap_openai(
            AsyncOpenAI(api_key=settings.llm_api_key or None, base_url=settings.llm_base_url)
        )
        return OpenAIAdapter(client, **common)
    if provider == "llama":
        # Llama endpoints usually ignore the key but the client requires a non-empty one.
        client = wrap_openai(
            AsyncOpenAI(
                api_key=settings.llm_api_key or "not-needed", base_url=settings.llm_base_url
            )
        )
        return LlamaAdapter(client, **common)

    raise ValueError(f"unknown LLM_PROVIDER: {settings.llm_provider!r} (use claude/openai/llama)")
