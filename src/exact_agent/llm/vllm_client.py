"""Thin OpenAI-compatible HTTP client.

The pipeline never imports ``transformers`` directly. All inference is
proxied through a vLLM (or any OpenAI-compatible) HTTP endpoint declared
in ``configs/model.yaml``. That keeps the API process small and lets the
LLM run on a separate GPU host without touching this code.

Two surfaces are supported:

* ``complete(prompt)`` for plain text in / text out (legacy completions).
* ``chat(messages)`` for the modern chat-completions schema.

A :class:`MockLLMClient` is provided for tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import yaml

from exact_agent.config import get_settings
from exact_agent.llm.retry import llm_retry


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float
    top_p: float
    timeout_s: float
    enable_lora: bool
    lora_adapter_path: str
    mode: str = "completion"
    """``chat`` routes ``complete()`` through /v1/chat/completions (applies the
    model's chat template — required for instruct models like Qwen3 on Ollama).
    ``completion`` uses the raw /v1/completions endpoint."""
    disable_thinking: bool = False
    """Qwen3 is a reasoning model: by default it spends tokens in a hidden
    ``reasoning`` channel and leaves ``content`` empty until it finishes
    thinking. Our architecture wants direct JSON (the solver does the
    reasoning), so we prepend the ``/no_think`` soft switch — portable
    across Ollama and vLLM."""

    @classmethod
    def from_yaml(cls, path: str | None = None) -> LLMConfig:
        cfg_path = path if path is not None else str(get_settings().configs_dir / "model.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            data = (yaml.safe_load(f) or {}).get("llm", {}) or {}

        # Deployment overrides. A container points the API at a sidecar
        # Ollama/vLLM service (e.g. http://ollama:11434/v1) without editing
        # model.yaml — same `EXACT_…__…` convention exact_agent.config uses
        # for app.yaml. Env beats file; absent env → file value unchanged
        # (so local dev and the test suite see model.yaml verbatim).
        base_default = str(data.get("vllm_base_url", "http://localhost:8001/v1"))
        lora_default = str(data.get("lora_adapter_path", ""))
        enable_lora = _env_bool("EXACT_LLM__ENABLE_LORA", bool(data.get("enable_lora", False)))

        return cls(
            base_url=os.environ.get("EXACT_LLM__BASE_URL", base_default),
            api_key=os.environ.get("EXACT_LLM__API_KEY", str(data.get("api_key", "local-no-auth"))),
            model=os.environ.get("EXACT_LLM__MODEL", str(data.get("backbone", "Qwen/Qwen3-8B"))),
            max_tokens=int(data.get("max_tokens", 1024)),
            temperature=float(data.get("temperature", 0.2)),
            top_p=float(data.get("top_p", 0.9)),
            timeout_s=float(data.get("timeout_s", 20)),
            enable_lora=enable_lora,
            lora_adapter_path=os.environ.get("EXACT_LLM__LORA_ADAPTER_PATH", lora_default),
            mode=str(data.get("mode", "completion")).lower(),
            disable_thinking=bool(data.get("disable_thinking", False)),
        )


class LLMClient(Protocol):
    """Minimal protocol the pipelines depend on (for easy mocking in tests)."""

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str: ...

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str: ...


class VLLMClient:
    """OpenAI-compatible HTTP client (vLLM, llama.cpp server, Ollama, …)."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._cfg = config or LLMConfig.from_yaml()
        self._http = httpx.Client(
            timeout=self._cfg.timeout_s,
            headers={"Authorization": f"Bearer {self._cfg.api_key}"},
        )

    # ---- public surface ----------------------------------------------------

    @llm_retry()
    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        # In chat mode, wrap the prompt as a single user turn so the server
        # applies the model's chat template. The caller API is unchanged —
        # llm_extractor / llm_translator / self_correction don't care which
        # endpoint actually served the text. Retry is applied once here;
        # the delegated path uses the undecorated `_do_chat`.
        prompt = self._maybe_no_think(prompt)
        if self._cfg.mode == "chat":
            return self._do_chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        return self._do_completion(prompt, max_tokens=max_tokens)

    def _maybe_no_think(self, prompt: str) -> str:
        """Prepend the Qwen3 ``/no_think`` soft switch when configured.

        Idempotent — won't double-prefix if the caller already added it.
        """
        if self._cfg.disable_thinking and not prompt.lstrip().startswith("/no_think"):
            return f"/no_think\n{prompt}"
        return prompt

    @llm_retry()
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        return self._do_chat(messages, max_tokens=max_tokens)

    def close(self) -> None:
        self._http.close()

    # ---- undecorated HTTP impls (retry applied by the public wrappers) -----

    def _do_completion(self, prompt: str, *, max_tokens: int | None) -> str:
        payload = self._build_completion_payload(prompt, max_tokens=max_tokens)
        response = self._http.post(
            f"{self._cfg.base_url.rstrip('/')}/completions",
            json=payload,
        )
        response.raise_for_status()
        return _extract_completion_text(response.json())

    def _do_chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None,
    ) -> str:
        payload = {
            "model": self._cfg.model,
            "messages": list(messages),
            "max_tokens": max_tokens or self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
        }
        response = self._http.post(
            f"{self._cfg.base_url.rstrip('/')}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return _extract_chat_text(response.json())

    # ---- internals ---------------------------------------------------------

    def _build_completion_payload(self, prompt: str, *, max_tokens: int | None) -> dict[str, Any]:
        return {
            "model": self._cfg.model,
            "prompt": prompt,
            "max_tokens": max_tokens or self._cfg.max_tokens,
            "temperature": self._cfg.temperature,
            "top_p": self._cfg.top_p,
        }


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _extract_completion_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    text = choices[0].get("text") or ""
    return str(text).strip()


def _extract_chat_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = str(message.get("content") or "").strip()
    if content:
        return content
    # Defensive: a reasoning model that ignored /no_think (or ran out of
    # tokens mid-think) leaves `content` empty and the text in `reasoning`.
    # Surface that rather than returning "" — the JSON parser can still try
    # to recover a {...} block from it.
    return str(message.get("reasoning") or "").strip()


def parse_json_completion(text: str) -> Any:
    """Parse a JSON object out of an LLM completion.

    LLMs occasionally wrap output in markdown code fences or add a
    leading explanation; we strip the first / last brace pair and try
    again so the pipeline doesn't fail on cosmetic noise.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty completion")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Remove markdown fences. ``text.split("```", 2)`` returns three parts
        # for a well-formed fenced block: ['', 'json\n{...}\n', ''] — pick the
        # middle one rather than the trailing empty string.
        if text.startswith("```"):
            parts = text.split("```")
            inner = parts[1] if len(parts) >= 2 else text
            if inner.lower().startswith("json\n"):
                inner = inner[5:]
            text = inner.strip("` \n")
        # Find the first '{' and the last '}'.
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class MockLLMClient:
    """In-memory stub: hand it canned responses indexed by call number.

    Tests construct it with ``MockLLMClient(["{...}", "{...}"])``; each
    call to ``complete``/``chat`` pops the next response. Useful for the
    LLM extractor / translator test paths without spinning up vLLM.
    """

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        self.calls.append({"prompt": prompt, "max_tokens": max_tokens})
        if not self._responses:
            raise RuntimeError("MockLLMClient has no more canned responses")
        return self._responses.pop(0)

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"messages": list(messages), "max_tokens": max_tokens})
        if not self._responses:
            raise RuntimeError("MockLLMClient has no more canned responses")
        return self._responses.pop(0)
