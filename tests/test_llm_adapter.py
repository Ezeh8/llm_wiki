import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.adapters.llm.base import LLMGenerationError, LLMParseError
from app.adapters.llm.claude import ClaudeAdapter
from app.adapters.llm.llama import LlamaAdapter
from app.adapters.llm.openai_provider import OpenAIAdapter
from app.domain import GenerationResult

_PAYLOAD = {
    "answer": "Reset your badge at the front desk.",
    "citations": [
        {
            "document_id": "doc-1",
            "document_title": "Access Policy",
            "chunk_id": "doc-1:2",
            "chunk_text": "Lost badges are reissued at the front desk.",
            "chunk_index": 2,
        }
    ],
}

_COMMON = {"model": "test-model", "temperature": 0.1, "max_tokens": 1000}


def _claude_client(blocks):
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=SimpleNamespace(content=blocks))
    return client


def _openai_client(content):
    client = MagicMock()
    message = SimpleNamespace(content=content)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _assert_result(result: GenerationResult):
    assert isinstance(result, GenerationResult)
    assert result.answer == _PAYLOAD["answer"]
    assert result.citations[0].chunk_id == "doc-1:2"
    assert result.citations[0].chunk_index == 2


# --- Claude (tool_use) --------------------------------------------------------


async def test_claude_uses_forced_tool_call_and_parses_input():
    tool_block = SimpleNamespace(type="tool_use", name="grounded_answer", input=_PAYLOAD)
    client = _claude_client([SimpleNamespace(type="text", text="ignored"), tool_block])
    adapter = ClaudeAdapter(client, **_COMMON)

    result = await adapter.generate(system="sys", user="q")

    _assert_result(result)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "grounded_answer"}
    assert kwargs["tools"][0]["name"] == "grounded_answer"
    assert kwargs["system"] == "sys"


async def test_claude_missing_tool_block_raises_parse_error():
    client = _claude_client([SimpleNamespace(type="text", text="no tool here")])
    with pytest.raises(LLMParseError):
        await ClaudeAdapter(client, **_COMMON).generate(system="s", user="u")


async def test_claude_api_failure_raises_generation_error():
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("503"))
    with pytest.raises(LLMGenerationError):
        await ClaudeAdapter(client, **_COMMON).generate(system="s", user="u")


async def test_claude_invalid_tool_input_raises_parse_error():
    bad = SimpleNamespace(type="tool_use", name="grounded_answer", input={"answer": "x"})
    with pytest.raises(LLMParseError):
        await ClaudeAdapter(_claude_client([bad]), **_COMMON).generate(system="s", user="u")


# --- OpenAI (response_format) --------------------------------------------------


async def test_openai_uses_response_format_and_parses_content():
    client = _openai_client(json.dumps(_PAYLOAD))
    result = await OpenAIAdapter(client, **_COMMON).generate(system="sys", user="q")

    _assert_result(result)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "grounded_answer"
    assert kwargs["messages"][0] == {"role": "system", "content": "sys"}


async def test_openai_malformed_content_raises_parse_error():
    client = _openai_client("not json at all")
    with pytest.raises(LLMParseError):
        await OpenAIAdapter(client, **_COMMON).generate(system="s", user="u")


# --- Llama (prompt enforcement) -----------------------------------------------


async def test_llama_uses_prompt_enforcement_no_response_format():
    fenced = "```json\n" + json.dumps(_PAYLOAD) + "\n```"
    client = _openai_client(fenced)
    result = await LlamaAdapter(client, **_COMMON).generate(system="base", user="q")

    _assert_result(result)
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs
    assert "JSON schema" in kwargs["messages"][0]["content"]
    assert kwargs["messages"][0]["content"].startswith("base")


async def test_llama_unparseable_output_raises_parse_error():
    client = _openai_client("the answer is definitely somewhere")
    with pytest.raises(LLMParseError):
        await LlamaAdapter(client, **_COMMON).generate(system="s", user="u")
