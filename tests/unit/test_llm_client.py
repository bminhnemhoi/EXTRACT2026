"""Tests for the LLM client + JSON parsing helpers."""

from __future__ import annotations

import httpx
import pytest

from exact_agent.llm.vllm_client import (
    LLMConfig,
    MockLLMClient,
    VLLMClient,
    parse_json_completion,
)


def _config(mode: str) -> LLMConfig:
    return LLMConfig(
        base_url="http://test-endpoint/v1",
        api_key="x",
        model="qwen3:4b",
        max_tokens=64,
        temperature=0.2,
        top_p=0.9,
        timeout_s=5,
        enable_lora=False,
        lora_adapter_path="",
        mode=mode,
    )


def _client_with_transport(mode: str, handler) -> VLLMClient:  # type: ignore[no-untyped-def]
    client = VLLMClient(_config(mode))
    client._http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


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


class TestModeRouting:
    def test_chat_mode_routes_complete_through_chat_endpoint(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "chat-said-this"}}]},
            )

        client = _client_with_transport("chat", handler)
        out = client.complete("hello")
        assert out == "chat-said-this"
        assert seen == ["/v1/chat/completions"]

    def test_completion_mode_uses_completions_endpoint(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200, json={"choices": [{"text": "raw-completion"}]})

        client = _client_with_transport("completion", handler)
        out = client.complete("hello")
        assert out == "raw-completion"
        assert seen == ["/v1/completions"]

    def test_yaml_default_mode_is_chat(self) -> None:
        # configs/model.yaml was switched to Ollama chat mode in Day 8.
        cfg = LLMConfig.from_yaml()
        assert cfg.mode == "chat"
        assert "11434" in cfg.base_url


class TestEnvOverride:
    """Deployment override: container points the API at a sidecar service."""

    def test_env_overrides_base_url_and_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXACT_LLM__BASE_URL", "http://ollama:11434/v1")
        monkeypatch.setenv("EXACT_LLM__MODEL", "qwen2.5:7b-instruct")
        monkeypatch.setenv("EXACT_LLM__API_KEY", "deploy-key")
        cfg = LLMConfig.from_yaml()
        assert cfg.base_url == "http://ollama:11434/v1"
        assert cfg.model == "qwen2.5:7b-instruct"
        assert cfg.api_key == "deploy-key"
        # File-only leaves still come from model.yaml.
        assert cfg.mode == "chat"

    def test_absent_env_keeps_yaml_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "EXACT_LLM__BASE_URL",
            "EXACT_LLM__MODEL",
            "EXACT_LLM__API_KEY",
            "EXACT_LLM__LORA_ADAPTER_PATH",
            "EXACT_LLM__ENABLE_LORA",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = LLMConfig.from_yaml()
        # Regression: local dev / CI see model.yaml verbatim.
        assert "11434" in cfg.base_url
        assert cfg.enable_lora is False

    def test_env_enable_lora_is_truthy_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXACT_LLM__ENABLE_LORA", "true")
        monkeypatch.setenv("EXACT_LLM__LORA_ADAPTER_PATH", "/models/adapter")
        cfg = LLMConfig.from_yaml()
        assert cfg.enable_lora is True
        assert cfg.lora_adapter_path == "/models/adapter"
