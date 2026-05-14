"""Tests for the LLM client + JSON parsing helpers."""

from __future__ import annotations

import pytest

from exact_agent.llm.vllm_client import (
    MockLLMClient,
    parse_json_completion,
)


class TestParseJsonCompletion:
    def test_clean_json(self) -> None:
        assert parse_json_completion('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self) -> None:
        wrapped = '```json\n{"a": 2}\n```'
        assert parse_json_completion(wrapped) == {"a": 2}

    def test_extracts_object_from_prose(self) -> None:
        text = 'Sure! Here is the JSON:\n{"x": 42, "y": "hi"}\nThanks.'
        assert parse_json_completion(text) == {"x": 42, "y": "hi"}

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_json_completion("")


class TestMockLLMClient:
    def test_dispenses_canned_responses_in_order(self) -> None:
        client = MockLLMClient(["alpha", "beta"])
        assert client.complete("p1") == "alpha"
        assert client.complete("p2") == "beta"
        assert len(client.calls) == 2

    def test_chat_records_messages(self) -> None:
        client = MockLLMClient(["ok"])
        client.chat([{"role": "user", "content": "hi"}])
        assert client.calls[0]["messages"][0]["role"] == "user"

    def test_runs_out_raises(self) -> None:
        client = MockLLMClient([])
        with pytest.raises(RuntimeError):
            client.complete("p")
