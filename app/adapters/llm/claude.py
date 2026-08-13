from anthropic import AsyncAnthropic

from app.adapters.llm.base import (
    ANSWER_SCHEMA,
    STRUCTURED_OUTPUT_NAME,
    LLMAdapter,
    LLMGenerationError,
    LLMParseError,
    parse_generation,
)
from app.domain import GenerationResult


class ClaudeAdapter(LLMAdapter):
    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, *, system: str, user: str) -> GenerationResult:
        # Structured output via a forced tool call (PRD: "tool_use for Claude"). No
        # `strict` flag — strict/structured-outputs support isn't guaranteed on the
        # locked model (claude-sonnet-4-6); the returned tool input is validated by
        # Pydantic in parse_generation instead.
        tool = {
            "name": STRUCTURED_OUTPUT_NAME,
            "description": "Return the grounded answer and its verified citations.",
            "input_schema": ANSWER_SCHEMA,
        }
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=[tool],
                tool_choice={"type": "tool", "name": STRUCTURED_OUTPUT_NAME},
            )
        except Exception as exc:
            raise LLMGenerationError(f"claude request failed: {exc}") from exc

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == STRUCTURED_OUTPUT_NAME:
                return parse_generation(block.input)
        raise LLMParseError("claude response contained no grounded_answer tool_use block")
