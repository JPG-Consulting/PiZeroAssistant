"""LAN OpenAI-compatible LLM provider."""

from __future__ import annotations

import json
import time
from typing import Optional

import requests

from voiceassistant.logging_config import get_logger
from voiceassistant.providers.base import LLMRequest, ProviderError
from voiceassistant.providers.http import HttpProvider, LLMResponse

logger = get_logger(__name__)


class LanHttpLLMProvider(HttpProvider):
    """LAN OpenAI-compatible LLM provider.

    Streaming is internal-only for latency; the provider returns a single response.
    """

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_s: float,
        api_key: Optional[str],
    ) -> None:
        super().__init__(name, endpoint, timeout_s, api_key)

    def complete(self, req: LLMRequest) -> LLMResponse:
        payload = {"messages": req.messages, "stream": True}
        try:
            resp = requests.post(
                f"{self.endpoint.rstrip('/')}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=(self.timeout_s, None),  # disable read timeout for streaming LLMs
                stream=True,
            )
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc

        if resp.status_code != 200:
            snippet = resp.text.strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            message = f"HTTP {resp.status_code}"
            if snippet:
                message = f"{message}: {snippet}"
            resp.close()
            raise ProviderError(message)

        start_time = time.monotonic()
        first_token_time = None
        parts: list[str] = []
        total_chars = 0
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except ValueError as exc:
                    raise ProviderError("Invalid JSON in LLM stream") from exc
                choices = payload.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta") or {}
                    fragment = delta.get("content")
                    if not fragment:
                        continue
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    parts.append(fragment)
                    total_chars += len(fragment)
        except requests.RequestException as exc:
            raise ProviderError(str(exc)) from exc
        finally:
            resp.close()

        full_text = "".join(parts)
        if not full_text:
            raise ProviderError("Missing text in LLM response")

        logger.info("LLM provider %s completed", self.name)
        if first_token_time is not None:
            logger.debug(
                "LLM provider %s time-to-first-token %.3fs",
                self.name,
                first_token_time - start_time,
            )
        logger.debug("LLM provider %s streamed %d chars", self.name, total_chars)
        return LLMResponse(text=full_text)
