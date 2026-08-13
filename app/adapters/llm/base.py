"""LLM adapter contract, shared JSON schema, and structured-output parsing.

The adapter wraps the Generator node's structured-output need behind one interface.
Each provider subclass requests the SAME target shape in its native way (tool_use for
Claude, response_format for OpenAI, prompt enforcement for Llama) and returns the
provider-neutral `GenerationResult`. The adapter never retries and never applies the
degraded fallback — the Generator node (Step 10) owns retry and "LLM never blocks
retrieval" behavior; the adapter surfaces failures as typed errors instead.
"""

import json
from abc import ABC, abstractmethod

from pydantic import ValidationError

from app.domain import GenerationResult

STRUCTURED_OUTPUT_NAME = "grounded_answer"

_CITATION_SCHEMA = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string"},
        "document_title": {"type": "string"},
        "chunk_id": {"type": "string"},
        "chunk_text": {"type": "string"},
        "chunk_index": {"type": "integer"},
    },
    "required": ["document_id", "document_title", "chunk_id", "chunk_text", "chunk_index"],
    "additionalProperties": False,
}

# The structured-output target requested from every provider.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": _CITATION_SCHEMA},
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


class LLMError(Exception):
    """Base class for LLM adapter failures."""


class LLMGenerationError(LLMError):
    """The provider call itself failed (network/API/transport error)."""


class LLMParseError(LLMError):
    """The provider responded but its structured output could not be parsed/validated."""


class LLMAdapter(ABC):
    @abstractmethod
    async def generate(self, *, system: str, user: str) -> GenerationResult:
        """Return a validated {answer, citations} from the provider, or raise LLMError."""


def parse_generation(payload: dict | str) -> GenerationResult:
    if isinstance(payload, str):
        payload = _extract_json_object(payload)
    try:
        return GenerationResult.model_validate(payload)
    except ValidationError as exc:
        raise LLMParseError(f"structured output failed schema validation: {exc}") from exc


def _extract_json_object(text: str) -> dict:
    # Prompt-enforced providers (Llama) may wrap JSON in prose or markdown fences;
    # slice the outermost object rather than trusting the whole string to be pure JSON.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMParseError("no JSON object found in model output")
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"model output was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMParseError("model output JSON was not an object")
    return obj
