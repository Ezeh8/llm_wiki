import json

from openai import AsyncOpenAI

from app.adapters.llm.base import (
    ANSWER_SCHEMA,
    LLMAdapter,
    LLMGenerationError,
    LLMParseError,
    parse_generation,
)
from app.config import get_settings
from app.domain import GenerationResult

# Llama has no native structured-output API (PRD: "prompt enforcement for Llama"),
# so the schema is injected into the system prompt and the raw text is parsed.
_SCHEMA_INSTRUCTION = (
    "Respond with ONLY a single JSON object and nothing else — no prose, no markdown "
    "fences. It must conform exactly to this JSON schema:\n" + json.dumps(ANSWER_SCHEMA)
)


class LlamaAdapter(LLMAdapter):
    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        # `model` (the shared LLM_MODEL_NAME, Claude-format) is accepted only to match
        # build_llm_adapter()'s common **kwargs call and is otherwise unused — this
        # provider is dormant, so it sources its own model name independently to avoid
        # ever sending a Claude-format string to a Llama endpoint if activated.
        settings = get_settings()
        if not settings.llm_model_name_llama:
            raise ValueError(
                "LLM_MODEL_NAME_LLAMA must be set when LLM_PROVIDER=llama"
            )
        self._client = client
        self._model = settings.llm_model_name_llama
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, *, system: str, user: str) -> GenerationResult:
        # Prompt enforcement only — no response_format. Served via an OpenAI-compatible
        # endpoint (vLLM/Ollama/TGI) selected by LLM_BASE_URL; see BUILD_LOG for the
        # transport rationale. parse_generation extracts + validates the raw JSON, and
        # raises LLMParseError if the model didn't emit a valid object.
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": f"{system}\n\n{_SCHEMA_INSTRUCTION}"},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise LLMGenerationError(f"llama request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMParseError("llama response contained no content")
        return parse_generation(content)
