"""OpenAI Chat Completions LLM provider."""

from __future__ import annotations

from typing import Optional

import requests

from voiceassistant.providers.base import LLMRequest, Provider, ProviderError, build_messages_with_system
from voiceassistant.providers.http import LLMResponse


class OpenAILLMProvider(Provider):
    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
        model: str,
    ) -> None:
        super().__init__(name, timeout_s, api_key)
        self.endpoint = endpoint
        self.model = model

    def _headers(self) -> dict:
        headers = {
            "User-Agent": "VoiceAssistant/1.0",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, req: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": build_messages_with_system(req.messages, req.system_prompt),
        }
        if req.max_tokens is not None:
            payload["max_tokens"] = req.max_tokens
        try:
            resp = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        if resp.status_code != 200:
            error_text = resp.text or ""
            snippet = " ".join(error_text.split())
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            message = f"HTTP {resp.status_code}"
            if snippet:
                message = f"{message}: {snippet}"
            raise ProviderError(message)
        try:
            data = resp.json()
        except ValueError as exc:
            raise ProviderError("Invalid JSON in OpenAI response") from exc
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Missing content in OpenAI response") from exc
        if not text:
            raise ProviderError("Missing content in OpenAI response")
        return LLMResponse(text=text)
