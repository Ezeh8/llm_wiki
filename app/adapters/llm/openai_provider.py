from openai import AsyncOpenAI

from app.adapters.llm.base import (
    ANSWER_SCHEMA,
    STRUCTURED_OUTPUT_NAME,
    LLMAdapter,
    LLMGenerationError,
    LLMParseError,
    parse_generation,
)
from app.config import get_settings
from app.domain import GenerationResult


class OpenAIAdapter(LLMAdapter):
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
        # ever sending a Claude-format string to OpenAI's API if activated.
        settings = get_settings()
        if not settings.llm_model_name_openai:
            raise ValueError(
                "LLM_MODEL_NAME_OPENAI must be set when LLM_PROVIDER=openai"
            )
        self._client = client
        self._model = settings.llm_model_name_openai
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, *, system: str, user: str) -> GenerationResult:
        # Structured output via response_format json_schema (PRD: "response_format
        # for OpenAI"), strict so the returned content is guaranteed schema-shaped JSON.
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": STRUCTURED_OUTPUT_NAME,
                        "schema": ANSWER_SCHEMA,
                        "strict": True,
                    },
                },
            )
        except Exception as exc:
            raise LLMGenerationError(f"openai request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMParseError("openai response contained no content")
        return parse_generation(content)
